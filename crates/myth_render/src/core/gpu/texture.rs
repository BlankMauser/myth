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
mod tests {
    use super::*;
    use crate::settings::{RendererInitConfig, RendererSettings};
    use crate::test_support::SharedTestContext as WgpuContext;
    use myth_resources::image::{ColorSpace, ImageDimension, PixelFormat};

    fn image_8x8(fill: u8) -> Image {
        Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![fill; 8 * 8 * 4]),
        )
    }

    fn resource_manager() -> ResourceManager {
        let context = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        ResourceManager::new(context.device.clone(), context.queue.clone(), 1)
    }

    fn prepare_ready(
        resources: &mut ResourceManager,
        assets: &AssetServer,
        texture: TextureHandle,
    ) {
        assert!(matches!(
            resources.prepare_texture(assets, texture),
            ResourceState::Ready
        ));
    }

    #[test]
    fn snapshots_canonicalize_specs_and_separate_content_versions() {
        let assets = AssetServer::new();
        let image = assets.images.add(image_8x8(255));
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("snapshot"), image));

        let first = snapshot_asset_texture(&assets, texture).expect("ready snapshot missing");
        assert_eq!(
            (first.texture_version, first.image_version, first.mip_count),
            (1, 1, 1)
        );
        assert_eq!(first.format, wgpu::TextureFormat::Rgba8UnormSrgb);

        let mut changed = Texture::new_2d(Some("linear mipped"), image);
        changed.color_space = ColorSpace::Linear;
        changed.generate_mipmaps = true;
        assert_eq!(assets.textures.update(texture, changed), Some(2));

        let second = snapshot_asset_texture(&assets, texture).expect("updated snapshot missing");
        assert_eq!(
            (
                second.texture_version,
                second.image_version,
                second.mip_count
            ),
            (2, 1, 4),
            "texture policy drift must not masquerade as image-content drift"
        );
        assert_eq!(second.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_ne!(first.logical_version, second.logical_version);
        assert_eq!(second.physical_spec().image_handle, image);
    }

    #[test]
    fn exact_residency_cas_fails_closed_across_logical_and_content_drift() {
        let mut resources = resource_manager();
        let assets = AssetServer::new();
        let image = assets.images.add(image_8x8(255));
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("resident"), image));

        let initial = snapshot_asset_texture(&assets, texture).expect("initial snapshot missing");
        assert!(
            resources
                .materialize_asset_texture_for_graph(&assets, initial)
                .texture
                .is_some()
        );
        let initial_token = resources
            .exact_asset_texture_residency(initial)
            .expect("initial residency token missing");
        let initial_generation = resources
            .resources
            .texture_generation_key(texture)
            .expect("initial generation missing");
        assert_eq!(
            resources
                .reuse_exact_asset_texture_for_graph(initial, initial_token)
                .and_then(|materialized| materialized.texture)
                .expect("unchanged residency must be reusable")
                .generation_key,
            initial_generation
        );

        let mut sampler_changed = Texture::new_2d(Some("nearest"), image);
        sampler_changed.sampler.mag_filter = wgpu::FilterMode::Nearest;
        assert_eq!(assets.textures.update(texture, sampler_changed), Some(2));
        let sampler_snapshot =
            snapshot_asset_texture(&assets, texture).expect("sampler snapshot missing");
        assert!(
            resources
                .materialize_asset_texture_for_graph(&assets, sampler_snapshot)
                .texture
                .is_some()
        );
        let sampler_token = resources
            .exact_asset_texture_residency(sampler_snapshot)
            .expect("sampler residency token missing");
        let sampler_generation = resources
            .resources
            .texture_generation_key(texture)
            .expect("sampler generation missing");
        assert_eq!(sampler_generation, initial_generation);
        assert_ne!(sampler_token, initial_token);
        let sampler_binding = *resources
            .get_texture_binding(texture)
            .expect("sampler binding missing");
        assert!(
            resources
                .reuse_exact_asset_texture_for_graph(initial, initial_token)
                .is_none()
        );
        assert_eq!(
            *resources
                .get_texture_binding(texture)
                .expect("failed CAS removed the current binding"),
            sampler_binding
        );

        assert_eq!(assets.images.update(image, image_8x8(37)), Some(2));
        let image_snapshot =
            snapshot_asset_texture(&assets, texture).expect("image snapshot missing");
        assert!(
            resources
                .materialize_asset_texture_for_graph(&assets, image_snapshot)
                .texture
                .is_some()
        );
        let image_generation = resources
            .resources
            .texture_generation_key(texture)
            .expect("refreshed generation missing");
        assert_eq!(image_generation.allocation, sampler_generation.allocation);
        assert_ne!(image_generation, sampler_generation);
        let image_binding = *resources
            .get_texture_binding(texture)
            .expect("refreshed binding missing");
        assert!(
            resources
                .reuse_exact_asset_texture_for_graph(sampler_snapshot, sampler_token)
                .is_none()
        );
        assert_eq!(
            *resources
                .get_texture_binding(texture)
                .expect("generation CAS failure removed the winner"),
            image_binding
        );
        assert_eq!(
            resources.resources.texture_generation_key(texture),
            Some(image_generation)
        );
    }

    #[test]
    fn canonical_physical_pool_shares_exact_specs_and_isolates_lease_ownership() {
        let mut resources = resource_manager();
        let assets = AssetServer::new();
        let image = assets.images.add(image_8x8(255));

        let mut first_asset = Texture::new_2d(Some("first"), image);
        first_asset.generate_mipmaps = true;
        let first = assets.textures.add(first_asset);
        let mut second_asset = Texture::new_2d(Some("second nearest"), image);
        second_asset.generate_mipmaps = true;
        second_asset.sampler.mag_filter = wgpu::FilterMode::Nearest;
        let second = assets.textures.add(second_asset);
        let mut linear_asset = Texture::new_2d(Some("linear"), image);
        linear_asset.color_space = ColorSpace::Linear;
        linear_asset.generate_mipmaps = true;
        let linear = assets.textures.add(linear_asset);

        for handle in [first, second, linear] {
            prepare_ready(&mut resources, &assets, handle);
        }
        resources.resources.assert_texture_pool_consistent();

        let first_lease = resources
            .resources
            .texture_image_lease(first)
            .expect("first lease missing");
        let second_lease = resources
            .resources
            .texture_image_lease(second)
            .expect("second lease missing");
        let linear_lease = resources
            .resources
            .texture_image_lease(linear)
            .expect("linear lease missing");
        assert_eq!(first_lease.physical_id, second_lease.physical_id);
        assert_ne!(first_lease.physical_id, linear_lease.physical_id);
        assert_eq!(
            resources
                .resources
                .physical_image_member_count(first_lease.physical_id),
            Some(2)
        );
        assert_eq!(
            resources
                .get_texture_image(first)
                .expect("shared image missing")
                .id,
            resources
                .get_texture_image(second)
                .expect("second shared image missing")
                .id
        );
        assert_ne!(
            resources
                .get_texture_binding(first)
                .expect("first binding missing")
                .sampler_id,
            resources
                .get_texture_binding(second)
                .expect("second binding missing")
                .sampler_id,
            "samplers remain lease-local even when the image allocation is shared"
        );
        assert_eq!(
            resources
                .get_texture_image(linear)
                .expect("linear image missing")
                .format,
            wgpu::TextureFormat::Rgba8Unorm
        );
        let requests = resources.take_mipmap_requests();
        assert_eq!(requests.len(), 2, "one request per physical generation");

        assert_eq!(
            assets
                .textures
                .update(first, Texture::new_2d(Some("first without mips"), image),),
            Some(2)
        );
        prepare_ready(&mut resources, &assets, first);
        let migrated = resources
            .resources
            .texture_image_lease(first)
            .expect("migrated lease missing")
            .physical_id;
        assert_ne!(migrated, second_lease.physical_id);
        assert_eq!(
            resources
                .resources
                .physical_image_member_count(second_lease.physical_id),
            Some(1)
        );

        resources.resources.remove_texture_image(first);
        assert!(!resources.resources.contains_physical_image(migrated));
        assert!(resources.get_texture_image(second).is_some());
        resources.resources.remove_texture_image(second);
        assert!(
            !resources
                .resources
                .contains_physical_image(second_lease.physical_id)
        );
        assert!(resources.get_texture_image(linear).is_some());
        resources.resources.assert_texture_pool_consistent();
    }

    #[test]
    fn same_handle_same_version_replacements_refresh_exact_gpu_identity() {
        let mut resources = resource_manager();
        let assets = AssetServer::new();
        let image = assets.images.add(image_8x8(255));
        let mut initial_asset = Texture::new_2d(Some("ABA sRGB"), image);
        initial_asset.generate_mipmaps = true;
        let texture = assets.textures.add(initial_asset);

        let initial_snapshot =
            snapshot_asset_texture(&assets, texture).expect("initial snapshot missing");
        prepare_ready(&mut resources, &assets, texture);
        let initial_binding = *resources
            .get_texture_binding(texture)
            .expect("initial binding missing");
        let initial_allocation = resources
            .get_texture_image(texture)
            .expect("initial image missing")
            .id;
        let initial_physical = resources
            .resources
            .texture_image_lease(texture)
            .expect("initial lease missing")
            .physical_id;
        let _ = resources.take_mipmap_requests();

        let texture_uuid = assets
            .textures
            .get(texture)
            .expect("initial texture missing")
            .uuid;
        let mut replacement = Texture::new_2d(Some("ABA linear"), image);
        replacement.uuid = texture_uuid;
        replacement.color_space = ColorSpace::Linear;
        replacement.generate_mipmaps = true;
        assets.textures.insert_ready(texture, replacement);

        let replacement_snapshot =
            snapshot_asset_texture(&assets, texture).expect("replacement snapshot missing");
        assert_eq!(
            replacement_snapshot.texture_version,
            initial_snapshot.texture_version
        );
        assert_ne!(
            replacement_snapshot.texture_identity,
            initial_snapshot.texture_identity
        );
        prepare_ready(&mut resources, &assets, texture);
        let replacement_binding = *resources
            .get_texture_binding(texture)
            .expect("replacement binding missing");
        let replacement_image = resources
            .get_texture_image(texture)
            .expect("replacement image missing");
        let replacement_allocation = replacement_image.id;
        let replacement_generation = replacement_image.generation_id;
        let replacement_image_identity = replacement_image.image_identity;
        let replacement_physical = resources
            .resources
            .texture_image_lease(texture)
            .expect("replacement lease missing")
            .physical_id;
        assert_ne!(replacement_allocation, initial_allocation);
        assert_ne!(replacement_physical, initial_physical);
        assert_ne!(
            replacement_binding.texture_identity,
            initial_binding.texture_identity
        );
        let _ = resources.take_mipmap_requests();

        let mut sampler_replacement = Texture::new_2d(Some("ABA nearest"), image);
        sampler_replacement.uuid = texture_uuid;
        sampler_replacement.color_space = ColorSpace::Linear;
        sampler_replacement.generate_mipmaps = true;
        sampler_replacement.sampler.mag_filter = wgpu::FilterMode::Nearest;
        assets.textures.insert_ready(texture, sampler_replacement);
        let sampler_snapshot =
            snapshot_asset_texture(&assets, texture).expect("sampler snapshot missing");
        let deferred = resources.materialize_asset_texture_for_graph(&assets, sampler_snapshot);
        assert!(deferred.texture.is_some() && deferred.mipmap_request.is_some());
        let sampler_binding = *resources
            .get_texture_binding(texture)
            .expect("sampler binding missing");
        assert_eq!(
            resources
                .resources
                .texture_image_lease(texture)
                .expect("sampler lease missing")
                .physical_id,
            replacement_physical
        );
        assert_ne!(sampler_binding.sampler_id, replacement_binding.sampler_id);
        assert_eq!(
            sampler_binding.texture_identity,
            sampler_snapshot.texture_identity
        );
        assert!(resources.take_mipmap_requests().is_empty());
        drop(deferred);

        let image_uuid = assets
            .images
            .get(image)
            .expect("initial image missing")
            .uuid;
        let mut image_replacement = image_8x8(31);
        image_replacement.uuid = image_uuid;
        assets.images.insert_ready(image, image_replacement);
        prepare_ready(&mut resources, &assets, texture);
        let updated = resources
            .get_texture_image(texture)
            .expect("updated image missing");
        assert_eq!(updated.id, replacement_allocation);
        assert_ne!(updated.image_identity, replacement_image_identity);
        assert_ne!(updated.generation_id, replacement_generation);
        assert_eq!(
            resources
                .resources
                .texture_image_lease(texture)
                .expect("updated lease missing")
                .physical_id,
            replacement_physical
        );
        assert_eq!(
            resources
                .get_texture_binding(texture)
                .expect("updated binding missing")
                .image_identity,
            updated.image_identity
        );
        assert_eq!(resources.take_mipmap_requests().len(), 1);
    }

    #[test]
    fn shared_refresh_revalidates_leases_and_rejects_stale_mip_and_alias_state() {
        let mut resources = resource_manager();
        let assets = AssetServer::new();
        let image = assets.images.add(image_8x8(255));
        let mut first_asset = Texture::new_2d(Some("first"), image);
        first_asset.generate_mipmaps = true;
        let first = assets.textures.add(first_asset);
        let mut second_asset = Texture::new_2d(Some("second"), image);
        second_asset.generate_mipmaps = true;
        let second = assets.textures.add(second_asset);

        prepare_ready(&mut resources, &assets, first);
        let stale = resources
            .take_mipmap_requests()
            .pop()
            .expect("initial mip request missing");
        prepare_ready(&mut resources, &assets, second);
        let _ = resources.take_mipmap_requests();
        let stale_after_drain = resources
            .resources
            .texture_mipmap_request(first)
            .expect("stale generation owner missing");
        let shared_physical = resources
            .resources
            .texture_image_lease(first)
            .expect("shared lease missing")
            .physical_id;
        let shared_allocation = resources
            .get_texture_image(first)
            .expect("shared image missing")
            .id;

        assert_eq!(assets.images.update(image, image_8x8(17)), Some(2));
        prepare_ready(&mut resources, &assets, first);
        let current_generation = resources
            .resources
            .texture_generation_key(first)
            .expect("current generation missing");
        assert_eq!(current_generation.physical, shared_physical);
        assert_eq!(current_generation.allocation, shared_allocation);
        assert_ne!(current_generation.generation, stale.generation_id);
        assert!(
            resources.get_texture_binding(second).is_none(),
            "other logical leases must fail closed until their new content stamp is revalidated"
        );

        stale.mark_submitted();
        assert!(
            !resources
                .get_texture_image(first)
                .expect("current image missing")
                .mipmaps_submitted()
        );
        resources.resources.queue_mipmap_request(stale);
        let current = resources
            .take_mipmap_requests()
            .pop()
            .expect("current generation retry missing");
        assert_eq!(current.key(), current_generation);
        current.mark_submitted();
        assert!(
            resources
                .get_texture_image(first)
                .expect("submitted image missing")
                .mipmaps_submitted()
        );
        stale_after_drain.mark_submitted();
        resources.resources.queue_mipmap_request(stale_after_drain);
        assert!(resources.take_mipmap_requests().is_empty());
        assert!(
            resources
                .get_texture_image(first)
                .expect("late stale publication removed the current image")
                .mipmaps_submitted()
        );

        prepare_ready(&mut resources, &assets, second);
        assert_eq!(
            resources
                .resources
                .texture_image_lease(second)
                .expect("revalidated lease missing")
                .physical_id,
            shared_physical
        );
        assert_eq!(
            resources
                .resources
                .physical_image_member_count(shared_physical),
            Some(2)
        );

        let other_image = assets.images.add(image_8x8(91));
        assert_eq!(
            assets
                .textures
                .update(first, Texture::new_2d(Some("redirected"), other_image),),
            Some(2)
        );
        prepare_ready(&mut resources, &assets, first);
        assert_eq!(
            resources.resources.latest_texture_alias(image),
            Some(second),
            "redirecting a lease must repair the old ImageHandle compatibility alias"
        );
        assert_eq!(
            resources.resources.latest_texture_alias(other_image),
            Some(first)
        );
        resources.resources.remove_texture_image(second);
        assert!(!resources.resources.contains_physical_image(shared_physical));
        assert!(resources.get_texture_image(first).is_some());
        resources.resources.assert_texture_pool_consistent();
    }
}
