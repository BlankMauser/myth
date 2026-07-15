//! Texture and Image operations
//!
//! Core concepts:
//! - `GpuImage`: Physical texture resource, containing `wgpu::Texture` and default view
//! - `GpuSampler`: Sampler state, globally cached for reuse
//! - `TextureBinding`: Maps `TextureHandle` to (`ImageId`, `ViewId`, `SamplerId`)
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::gpu::{PhysicalImageGenerationKey, Tracked, generate_gpu_resource_id};
use myth_assets::{AssetServer, ImageHandle, TextureHandle};
use myth_resources::image::Image;
use myth_resources::texture::{Texture, TextureSampler, TextureSource};

use super::environment::snapshot_ready_asset_texture_source;
use super::{GpuResourceStore, MipmapRequest, ResourceManager};

/// Texture resource mapping
///
/// Maps `TextureHandle` to the corresponding `GpuImage` ID, View ID, and `GpuSampler` ID
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureBinding {
    /// GPU-side image view ID
    pub view_id: u64,
    /// CPU-side image ID
    pub image_handle: ImageHandle,
    pub sampler_id: usize,
    /// CPU-side Texture asset version. This covers sampler, colour-space,
    /// view-dimension, image-reference, and mip-policy changes.
    pub texture_version: u32,
    /// Exact loaded Texture asset identity; unlike `texture_version`, this
    /// cannot ABA when a reserved slot is populated again at version 1.
    pub texture_identity: u64,
    /// CPU-side Image content/descriptor version.
    pub image_version: u32,
    /// Exact loaded Image asset identity.
    pub image_identity: u64,
}

impl TextureBinding {
    #[inline]
    #[must_use]
    pub(crate) fn matches_residency(self, lease: TextureImageLease, image: &GpuImage) -> bool {
        self.view_id == image.id
            && self.image_handle == lease.image_handle
            && lease.image_handle == image.image_handle
            && self.texture_version == lease.texture_version
            && self.texture_identity == lease.texture_identity
            && self.image_version == lease.image_version
            && self.image_identity == lease.image_identity
            && lease.image_version == image.image_version
            && lease.image_identity == image.image_identity
    }
}

/// CPU-only, exact-version description of an asset texture.
///
/// This value is safe to capture while building the graph. It carries no WGPU
/// owner and can therefore be stored directly in an arena PassNode. Live
/// preparation revalidates both asset versions before materializing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AssetTextureSnapshot {
    pub(crate) texture_handle: TextureHandle,
    pub(crate) image_handle: ImageHandle,
    pub(crate) texture_version: u32,
    pub(crate) texture_identity: u64,
    pub(crate) image_version: u32,
    pub(crate) image_identity: u64,
    pub(crate) logical_version: u64,
    pub(crate) extent: wgpu::Extent3d,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) dimension: wgpu::TextureDimension,
    pub(crate) view_dimension: wgpu::TextureViewDimension,
    pub(crate) mip_count: u32,
    pub(crate) usage: wgpu::TextureUsages,
    pub(crate) generate_mipmaps: bool,
}

/// Exact CPU-side state of one Asset Texture at graph construction time.
///
/// `ready == None` is still a meaningful snapshot. The surrounding stamps
/// distinguish a genuinely unchanged pending/missing dependency from a
/// same-handle replacement that also happens to be unavailable, closing the
/// pending-state ABA hole without creating a GPU resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AssetTextureStateSnapshot {
    pub(crate) texture_handle: TextureHandle,
    pub(crate) texture_version: Option<u32>,
    pub(crate) texture_identity: Option<u64>,
    pub(crate) image_handle: Option<ImageHandle>,
    pub(crate) image_version: Option<u32>,
    pub(crate) image_identity: Option<u64>,
    pub(crate) ready: Option<AssetTextureSnapshot>,
}

impl AssetTextureSnapshot {
    pub(crate) fn from_loaded(
        texture_handle: TextureHandle,
        texture: &Texture,
        texture_version: u32,
        texture_identity: u64,
        image: &Image,
        image_version: u32,
        image_identity: u64,
    ) -> Option<Self> {
        let source = TextureSource::Asset(texture_handle);
        let logical = snapshot_ready_asset_texture_source(
            source,
            texture,
            texture_version,
            texture_identity,
            image,
            image_version,
            image_identity,
        );
        if !logical.ready {
            return None;
        }

        let generate_mipmaps = texture.generate_mipmaps;
        let mip_count = requested_mip_count(image.width, image.height, generate_mipmaps);
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        if mip_count > 1 {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }

        Some(Self {
            texture_handle,
            image_handle: texture.image,
            texture_version,
            texture_identity,
            image_version,
            image_identity,
            logical_version: logical.version,
            extent: wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: image.depth,
            },
            format: texture.resolve_wgpu_format(image.format),
            dimension: image.dimension.to_wgpu(),
            view_dimension: texture.view_dimension,
            mip_count,
            usage,
            generate_mipmaps,
        })
    }

    #[inline]
    pub(crate) fn physical_spec(self) -> GpuImageSpec {
        GpuImageSpec {
            image_handle: self.image_handle,
            extent: self.extent,
            format: self.format,
            dimension: self.dimension,
            view_dimension: self.view_dimension,
            mip_count: self.mip_count,
            sample_count: 1,
            usage: self.usage,
        }
    }
}

/// Capture a ready asset texture without creating or updating GPU resources.
#[must_use]
pub(crate) fn snapshot_asset_texture(
    assets: &AssetServer,
    handle: TextureHandle,
) -> Option<AssetTextureSnapshot> {
    let (texture, texture_version, texture_identity) =
        assets.textures.get_entry_with_identity(handle)?;
    let (image, image_version, image_identity) =
        assets.images.get_entry_with_identity(texture.image)?;
    AssetTextureSnapshot::from_loaded(
        handle,
        &texture,
        texture_version,
        texture_identity,
        &image,
        image_version,
        image_identity,
    )
}

/// Capture both ready and unavailable Asset Texture states without touching
/// GPU resources. Material plans use the complete value for exact post-DPE
/// revalidation; only `ready` snapshots become logical RDG imports.
#[must_use]
#[cfg(test)]
pub(crate) fn snapshot_asset_texture_state(
    assets: &AssetServer,
    handle: TextureHandle,
) -> AssetTextureStateSnapshot {
    let Some((texture, texture_version, texture_identity)) =
        assets.textures.get_entry_with_identity(handle)
    else {
        return AssetTextureStateSnapshot {
            texture_handle: handle,
            texture_version: None,
            texture_identity: None,
            image_handle: None,
            image_version: None,
            image_identity: None,
            ready: None,
        };
    };

    let image_handle = texture.image;
    let Some((image, image_version, image_identity)) =
        assets.images.get_entry_with_identity(image_handle)
    else {
        return AssetTextureStateSnapshot {
            texture_handle: handle,
            texture_version: Some(texture_version),
            texture_identity: Some(texture_identity),
            image_handle: Some(image_handle),
            image_version: None,
            image_identity: None,
            ready: None,
        };
    };
    let ready = AssetTextureSnapshot::from_loaded(
        handle,
        &texture,
        texture_version,
        texture_identity,
        &image,
        image_version,
        image_identity,
    );

    AssetTextureStateSnapshot {
        texture_handle: handle,
        texture_version: Some(texture_version),
        texture_identity: Some(texture_identity),
        image_handle: Some(image_handle),
        image_version: Some(image_version),
        image_identity: Some(image_identity),
        ready,
    }
}

#[inline]
fn requested_mip_count(width: u32, height: u32, generate_mipmaps: bool) -> u32 {
    if generate_mipmaps {
        width.max(height).max(1).ilog2() + 1
    } else {
        1
    }
}

slotmap::new_key_type! {
    /// Generational identity of one physical image allocation in the GPU
    /// resource store.
    pub(crate) struct PhysicalImageId;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GpuImageSpec {
    pub(crate) image_handle: ImageHandle,
    pub(crate) extent: wgpu::Extent3d,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) dimension: wgpu::TextureDimension,
    pub(crate) view_dimension: wgpu::TextureViewDimension,
    pub(crate) mip_count: u32,
    pub(crate) sample_count: u32,
    pub(crate) usage: wgpu::TextureUsages,
}

/// Per-TextureHandle logical residency metadata.
///
/// Destructor-bearing WGPU state lives in the physical image arena. Texture
/// and Image stamps remain lease-local so the exact-spec pool can attach
/// several TextureHandles to one allocation without overwriting each other's
/// logical identity or TTL.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TextureImageLease {
    pub(crate) physical_id: PhysicalImageId,
    pub(crate) image_handle: ImageHandle,
    pub(crate) texture_version: u32,
    pub(crate) texture_identity: u64,
    pub(crate) image_version: u32,
    pub(crate) image_identity: u64,
    pub(crate) last_used_frame: u64,
}

/// Physical result produced by live asset materialization.
pub(crate) struct MaterializedAssetTexture {
    pub(crate) view: Tracked<wgpu::TextureView>,
    pub(crate) generation_key: PhysicalImageGenerationKey,
    pub(crate) sampler_id: usize,
}

/// Copy physical identity exported by the canonical Asset Texture producer to
/// later material consumers in the same graph attempt.
///
/// The generation key is ABA-safe for the physical allocation/content, while
/// `sampler_id` identifies the exact sampler slot installed by the producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AssetTexturePhysicalBinding {
    pub(crate) generation_key: PhysicalImageGenerationKey,
    pub(crate) sampler_id: usize,
}

impl MaterializedAssetTexture {
    #[inline]
    #[must_use]
    pub(crate) const fn physical_binding(&self) -> AssetTexturePhysicalBinding {
        AssetTexturePhysicalBinding {
            generation_key: self.generation_key,
            sampler_id: self.sampler_id,
        }
    }
}

/// A drift-safe live materialization attempt.
///
/// `texture == None` is a normal outcome when the CPU asset changed after its
/// Composer snapshot. The optional mip request remains graph-attempt-owned and
/// is published only if its logical producer encodes and the frame submits.
pub(crate) struct AssetTextureMaterialization {
    pub(crate) texture: Option<MaterializedAssetTexture>,
    pub(crate) mipmap_request: Option<MipmapRequest>,
}

impl AssetTextureMaterialization {
    #[inline]
    #[must_use]
    pub(crate) const fn unavailable() -> Self {
        Self {
            texture: None,
            mipmap_request: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ResourceState {
    /// Resource is fully loaded and GPU-ready
    Ready,
    /// Underlying data is still loading
    Pending,
    /// Resource is missing or failed to load (e.g. image decoding failed)
    Unknown,
}

/// GPU-side image resource
///
/// Contains the physical texture and default view, excluding sampler
pub struct GpuImage {
    pub id: u64,
    pub texture: wgpu::Texture,
    pub default_view: wgpu::TextureView,
    pub default_view_dimension: wgpu::TextureViewDimension,
    pub dimension: wgpu::TextureDimension,
    pub size: wgpu::Extent3d,
    pub format: wgpu::TextureFormat,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub usage: wgpu::TextureUsages,
    pub image_handle: ImageHandle,
    pub image_version: u32,
    pub image_identity: u64,
    pub generation_id: u64,
    submitted_mipmap_generation: Arc<AtomicU64>,
    pub last_used_frame: u64,
}

impl GpuImage {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &Image,
        spec: GpuImageSpec,
        image_version: u32,
        image_identity: u64,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: spec.extent,
            mip_level_count: spec.mip_count,
            sample_count: spec.sample_count,
            dimension: spec.dimension,
            format: spec.format,
            usage: spec.usage,
            view_formats: &[],
        });

        Self::upload_data(
            queue,
            &texture,
            image,
            image.width,
            image.height,
            image.depth,
            spec.format,
        );

        let default_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(spec.format),
            dimension: Some(spec.view_dimension),
            ..Default::default()
        });

        let generation_id = generate_gpu_resource_id();
        let submitted_mipmap_generation = Arc::new(AtomicU64::new(if spec.mip_count <= 1 {
            generation_id
        } else {
            0
        }));
        Self {
            id: generate_gpu_resource_id(),
            texture,
            default_view,
            default_view_dimension: spec.view_dimension,
            dimension: spec.dimension,
            size: spec.extent,
            format: spec.format,
            mip_level_count: spec.mip_count,
            sample_count: spec.sample_count,
            usage: spec.usage,
            image_handle: spec.image_handle,
            image_version,
            image_identity,
            generation_id,
            submitted_mipmap_generation,
            last_used_frame: 0,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn mipmaps_submitted(&self) -> bool {
        self.mip_level_count <= 1
            || self.submitted_mipmap_generation.load(Ordering::Acquire) == self.generation_id
    }

    #[inline]
    fn advance_content_generation(&mut self) {
        self.generation_id = generate_gpu_resource_id();
        self.submitted_mipmap_generation.store(
            if self.mip_level_count <= 1 {
                self.generation_id
            } else {
                0
            },
            Ordering::Release,
        );
    }

    #[must_use]
    pub(crate) fn mipmap_request(&self, physical_id: PhysicalImageId) -> Option<MipmapRequest> {
        if self.mipmaps_submitted() {
            return None;
        }

        Some(MipmapRequest::new(
            physical_id,
            self.texture.clone(),
            self.id,
            self.generation_id,
            self.format,
            self.submitted_mipmap_generation.clone(),
        ))
    }

    #[inline]
    #[must_use]
    pub(crate) const fn generation_key(
        &self,
        physical_id: PhysicalImageId,
    ) -> PhysicalImageGenerationKey {
        PhysicalImageGenerationKey {
            physical: physical_id,
            allocation: self.id,
            generation: self.generation_id,
        }
    }

    /// Upload a new base level into an allocation whose immutable physical
    /// specification has already been validated by the exact-pool entry.
    pub(crate) fn update_content(
        &mut self,
        queue: &wgpu::Queue,
        image: &Image,
        image_version: u32,
        image_identity: u64,
    ) {
        if self.image_version != image_version || self.image_identity != image_identity {
            Self::upload_data(
                queue,
                &self.texture,
                image,
                self.size.width,
                self.size.height,
                self.size.depth_or_array_layers,
                self.format,
            );
            self.image_version = image_version;
            self.image_identity = image_identity;
            self.advance_content_generation();
        }
    }

    fn upload_data(
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        image: &Image,
        src_width: u32,
        src_height: u32,
        src_depth: u32,
        src_format: wgpu::TextureFormat,
    ) {
        image.with_data(|data| {
            let block_size = src_format.block_copy_size(None).unwrap_or(4);
            let bytes_per_row = src_width * block_size;

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(src_height),
                },
                wgpu::Extent3d {
                    width: src_width,
                    height: src_height,
                    depth_or_array_layers: src_depth,
                },
            );
        });
    }
}

/// Materialize a revalidated snapshot through its per-TextureHandle lease.
///
/// The returned WGPU owners are cheap handle clones. Callers decide whether
/// the mip request belongs to the legacy pending queue or to frame-owned RDG
/// storage.
pub(crate) fn materialize_asset_texture_snapshot(
    resources: &mut GpuResourceStore,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frame_index: u64,
    snapshot: AssetTextureSnapshot,
    image: &Image,
    sampler_id: usize,
) -> AssetTextureMaterialization {
    let spec = snapshot.physical_spec();
    resources.materialize_texture_image(
        device,
        queue,
        frame_index,
        snapshot.texture_handle,
        image,
        spec,
        snapshot.texture_version,
        snapshot.texture_identity,
        snapshot.image_version,
        snapshot.image_identity,
    );

    let gpu_image = resources
        .texture_image(snapshot.texture_handle)
        .expect("asset texture missing after materialization");
    let texture = MaterializedAssetTexture {
        view: Tracked::with_id(gpu_image.default_view.clone(), gpu_image.id),
        generation_key: resources
            .texture_generation_key(snapshot.texture_handle)
            .expect("asset texture missing a physical generation"),
        sampler_id,
    };
    let mipmap_request = snapshot
        .generate_mipmaps
        .then(|| resources.texture_mipmap_request(snapshot.texture_handle))
        .flatten();

    // Maintain the ImageHandle-only compatibility index even when an existing
    // physical allocation was reused by this deferred consumer.
    resources.touch_texture_image_alias(snapshot.image_handle, snapshot.texture_handle);

    AssetTextureMaterialization {
        texture: Some(texture),
        mipmap_request,
    }
}

impl ResourceManager {
    /// Observe an exact resident Asset Texture before the graph coordinator's
    /// selected T -> I -> T postflight.
    #[must_use]
    pub(crate) fn exact_asset_texture_residency(
        &self,
        snapshot: AssetTextureSnapshot,
    ) -> Option<super::ExactAssetTextureResidency> {
        self.resources.exact_asset_texture_residency(snapshot)
    }

    /// Complete the cache-side CAS for an exact resident Asset Texture after
    /// AssetStorage postflight. The warm path only touches residency and
    /// clones graph-attempt owners.
    pub(crate) fn reuse_exact_asset_texture_for_graph(
        &mut self,
        snapshot: AssetTextureSnapshot,
        expected: super::ExactAssetTextureResidency,
    ) -> Option<AssetTextureMaterialization> {
        self.resources
            .reuse_exact_asset_texture(snapshot, expected, self.frame_index)
    }

    /// Materialize one DPE-live logical Asset Texture and publish the exact
    /// sampler/view binding consumed by graph-planned materials.
    ///
    /// Unlike [`Self::prepare_texture`], this path never enqueues a legacy mip
    /// request. The request remains inside the returned graph-attempt owner and
    /// is published only after the containing command buffer is submitted.
    pub(crate) fn materialize_asset_texture_for_graph(
        &mut self,
        assets: &AssetServer,
        snapshot: AssetTextureSnapshot,
    ) -> AssetTextureMaterialization {
        let Some((texture, texture_version, texture_identity)) = assets
            .textures
            .get_entry_with_identity(snapshot.texture_handle)
        else {
            return AssetTextureMaterialization::unavailable();
        };
        if texture_version != snapshot.texture_version
            || texture_identity != snapshot.texture_identity
            || texture.image != snapshot.image_handle
        {
            return AssetTextureMaterialization::unavailable();
        }
        let Some((image, image_version, image_identity)) =
            assets.images.get_entry_with_identity(snapshot.image_handle)
        else {
            return AssetTextureMaterialization::unavailable();
        };
        if image_version != snapshot.image_version
            || image_identity != snapshot.image_identity
            || snapshot_asset_texture(assets, snapshot.texture_handle) != Some(snapshot)
        {
            return AssetTextureMaterialization::unavailable();
        }

        let sampler_id = self.get_or_create_sampler(texture.sampler);
        let materialized = materialize_asset_texture_snapshot(
            &mut self.resources,
            &self.device,
            &self.queue,
            self.frame_index,
            snapshot,
            &image,
            sampler_id,
        );

        // Dynamic images and AssetStorage entries may change while bytes are
        // uploaded. Never publish a logical binding under a stale snapshot.
        if snapshot_asset_texture(assets, snapshot.texture_handle) != Some(snapshot) {
            return AssetTextureMaterialization::unavailable();
        }
        let Some(materialized_texture) = materialized.texture.as_ref() else {
            return AssetTextureMaterialization::unavailable();
        };
        self.resources.insert_texture_binding(
            snapshot.texture_handle,
            TextureBinding {
                view_id: materialized_texture.generation_key.allocation,
                image_handle: snapshot.image_handle,
                sampler_id,
                texture_version: snapshot.texture_version,
                texture_identity: snapshot.texture_identity,
                image_version: snapshot.image_version,
                image_identity: snapshot.image_identity,
            },
        );

        materialized
    }

    /// Whether the persistent binding currently installed for an Asset
    /// Texture is the exact logical and physical state captured by the graph.
    #[must_use]
    pub(crate) fn asset_texture_binding_matches_snapshot(
        &self,
        snapshot: AssetTextureSnapshot,
    ) -> bool {
        let Some(binding) = self.resources.texture_binding(snapshot.texture_handle) else {
            return false;
        };
        let Some(image) = self.resources.texture_image(snapshot.texture_handle) else {
            return false;
        };
        binding.image_handle == snapshot.image_handle
            && binding.texture_version == snapshot.texture_version
            && binding.texture_identity == snapshot.texture_identity
            && binding.image_version == snapshot.image_version
            && binding.image_identity == snapshot.image_identity
            && binding.view_id == image.id
            && image.image_handle == snapshot.image_handle
            && image.image_version == snapshot.image_version
            && image.image_identity == snapshot.image_identity
            && image.size == snapshot.extent
            && image.format == snapshot.format
            && image.dimension == snapshot.dimension
            && image.default_view_dimension == snapshot.view_dimension
            && image.mip_level_count == snapshot.mip_count
            && image.sample_count == 1
            && image.usage == snapshot.usage
    }

    /// Whether the exact physical content selected by this snapshot already
    /// has submitted mip levels. This is generation/content based rather than
    /// TextureHandle based, so a new logical lease can reuse a shared complete
    /// chain without adding an empty RDG mip node.
    #[must_use]
    pub(crate) fn asset_snapshot_mipmaps_submitted(&self, snapshot: AssetTextureSnapshot) -> bool {
        self.resources.physical_mipmaps_submitted_for_spec(
            snapshot.physical_spec(),
            snapshot.image_version,
            snapshot.image_identity,
        )
    }

    /// Prepare GPU resources for a `Texture` asset.
    ///
    /// Performs a **two-level query**: first retrieves the `Texture` config
    /// (always immediately available after [`AssetServer::load_texture`]),
    /// then checks whether the underlying `Image` has finished decoding.
    /// If the image is not yet ready, no binding is created and the
    /// material system falls back to a placeholder texture.
    ///
    /// Version tracking ensures GPU resources are only rebuilt when the
    /// underlying data actually changes.
    pub fn prepare_texture(
        &mut self,
        assets: &AssetServer,
        handle: TextureHandle,
    ) -> ResourceState {
        if handle == TextureHandle::dummy_env_map() {
            return ResourceState::Ready;
        }

        let Some((texture_asset, texture_version, texture_identity)) =
            assets.textures.get_entry_with_identity(handle)
        else {
            return ResourceState::Unknown;
        };

        let image_handle: ImageHandle = texture_asset.image;

        if assets.images.is_loading(image_handle) {
            return ResourceState::Pending;
        }

        let Some((observed_image_version, observed_image_identity)) =
            assets.images.get_version_and_identity(image_handle)
        else {
            // Image still decoding in the background — no binding is created,
            // the material system will use a fallback placeholder texture.
            return ResourceState::Unknown;
        };

        // ── Fast path: exact logical identity + physical policy match ──
        if let Some(binding) = self.resources.texture_binding(handle).copied() {
            let sampler_match = self
                .sampler_registry
                .lookup_index(&texture_asset.sampler)
                .is_some_and(|idx| idx == binding.sampler_id);
            let exact = if let Some(gpu_image) = self.resources.texture_image(handle) {
                let expected_mips = requested_mip_count(
                    gpu_image.size.width,
                    gpu_image.size.height,
                    texture_asset.generate_mipmaps,
                );
                let mut expected_usage =
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
                if expected_mips > 1 {
                    expected_usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
                }
                let matches = binding.texture_version == texture_version
                    && binding.texture_identity == texture_identity
                    && binding.image_version == observed_image_version
                    && binding.image_identity == observed_image_identity
                    && binding.image_handle == image_handle
                    && binding.view_id == gpu_image.id
                    && gpu_image.image_version == observed_image_version
                    && gpu_image.image_identity == observed_image_identity
                    && gpu_image.default_view_dimension == texture_asset.view_dimension
                    && gpu_image.mip_level_count == expected_mips
                    && gpu_image.sample_count == 1
                    && gpu_image.usage == expected_usage;
                matches && sampler_match
            } else {
                false
            };
            if exact {
                let texture_unchanged = assets.textures.get_version_and_identity(handle)
                    == Some((texture_version, texture_identity));
                let image_unchanged = assets.images.get_version_and_identity(image_handle)
                    == Some((observed_image_version, observed_image_identity));
                if !texture_unchanged || !image_unchanged {
                    return ResourceState::Pending;
                }
                let request = texture_asset
                    .generate_mipmaps
                    .then(|| self.resources.texture_mipmap_request(handle))
                    .flatten();
                assert!(
                    self.resources.touch_texture_image(handle, self.frame_index),
                    "validated texture binding lost its physical residency"
                );
                self.resources
                    .touch_texture_image_alias(image_handle, handle);
                if let Some(request) = request {
                    self.resources.queue_mipmap_request(request);
                }
                return ResourceState::Ready;
            }
        }

        // ── Slow path: something changed, do the full update ──
        let Some((image_arc, image_version, image_identity)) =
            assets.images.get_entry_with_identity(image_handle)
        else {
            return ResourceState::Unknown;
        };
        if assets.textures.get_version_and_identity(handle)
            != Some((texture_version, texture_identity))
        {
            return ResourceState::Pending;
        }
        let Some(snapshot) = AssetTextureSnapshot::from_loaded(
            handle,
            &texture_asset,
            texture_version,
            texture_identity,
            &image_arc,
            image_version,
            image_identity,
        ) else {
            return ResourceState::Unknown;
        };
        let sampler_id = self.get_or_create_sampler(texture_asset.sampler);
        let materialized = materialize_asset_texture_snapshot(
            &mut self.resources,
            &self.device,
            &self.queue,
            self.frame_index,
            snapshot,
            &image_arc,
            sampler_id,
        );

        // Do not publish a binding under a snapshot that drifted while its
        // dynamic image bytes were being uploaded.
        let texture_unchanged = assets.textures.get_version_and_identity(handle)
            == Some((texture_version, texture_identity));
        let image_unchanged = assets.images.get_version_and_identity(image_handle)
            == Some((image_version, image_identity));
        if !texture_unchanged || !image_unchanged {
            return ResourceState::Pending;
        }

        let Some(materialized_texture) = materialized.texture else {
            return ResourceState::Unknown;
        };
        if let Some(request) = materialized.mipmap_request {
            self.resources.queue_mipmap_request(request);
        }

        let binding = TextureBinding {
            view_id: materialized_texture.generation_key.allocation,
            image_handle,
            sampler_id,
            texture_version,
            texture_identity,
            image_version,
            image_identity,
        };
        self.resources.insert_texture_binding(handle, binding);

        ResourceState::Ready
    }

    pub(crate) fn get_or_create_sampler(&mut self, descriptor: TextureSampler) -> usize {
        self.sampler_registry
            .get_custom(&self.device, &descriptor)
            .0
    }

    #[inline]
    pub fn get_texture_binding(&self, handle: TextureHandle) -> Option<&TextureBinding> {
        self.resources.texture_binding(handle)
    }

    /// Resolve the exact physical allocation owned by one Texture asset.
    #[inline]
    pub fn get_texture_image(&self, texture_handle: TextureHandle) -> Option<&GpuImage> {
        self.resources.texture_image(texture_handle)
    }

    /// Compatibility lookup for callers that still only carry an ImageHandle.
    /// New render paths must prefer [`get_texture_image`](Self::get_texture_image)
    /// because multiple Texture assets may interpret the same Image differently.
    #[inline]
    pub fn get_image(&self, image_handle: ImageHandle) -> Option<&GpuImage> {
        self.resources.latest_image(image_handle)
    }

    #[inline]
    pub fn get_sampler_by_index(&self, index: usize) -> Option<&wgpu::Sampler> {
        self.sampler_registry.get_sampler_by_index(index)
    }
}

#[cfg(test)]
mod tests;
