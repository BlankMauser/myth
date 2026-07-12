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
#[derive(Debug, Clone, Copy)]
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
    fn from_loaded(
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
    fn physical_spec(self) -> GpuImageSpec {
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
    fn invalidate_mipmaps(&mut self) {
        if self.mip_level_count <= 1 {
            return;
        }
        self.generation_id = generate_gpu_resource_id();
        self.submitted_mipmap_generation.store(0, Ordering::Release);
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
            if self.mip_level_count > 1 {
                self.invalidate_mipmaps();
            }
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
    use crate::core::WgpuContext;
    use crate::settings::{RendererInitConfig, RendererSettings};
    use myth_resources::image::{ColorSpace, ImageDimension, PixelFormat};

    fn image_8x8() -> Image {
        Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![255; 8 * 8 * 4]),
        )
    }

    #[test]
    fn snapshots_keep_texture_and_image_versions_separate() {
        let assets = AssetServer::new();
        let image = assets.images.add(image_8x8());
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("Snapshot"), image));

        let first = snapshot_asset_texture(&assets, texture).expect("texture must be ready");
        assert_eq!(first.texture_version, 1);
        assert_eq!(first.image_version, 1);
        assert_eq!(first.mip_count, 1);
        assert_eq!(first.format, wgpu::TextureFormat::Rgba8UnormSrgb);

        let mut changed = Texture::new_2d(Some("Snapshot Linear Mipped"), image);
        changed.color_space = ColorSpace::Linear;
        changed.generate_mipmaps = true;
        assert_eq!(assets.textures.update(texture, changed), Some(2));

        let second = snapshot_asset_texture(&assets, texture).expect("texture must stay ready");
        assert_eq!(second.texture_version, 2);
        assert_eq!(second.image_version, 1);
        assert_eq!(second.mip_count, 4);
        assert_eq!(second.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_ne!(first.logical_version, second.logical_version);
    }

    #[test]
    fn unavailable_texture_states_reject_same_handle_pending_aba() {
        let assets = AssetServer::new();
        let first_missing = ImageHandle::from(slotmap::KeyData::from_ffi(0x101));
        let second_missing = ImageHandle::from(slotmap::KeyData::from_ffi(0x202));
        let texture_handle = assets
            .textures
            .add(Texture::new_2d(Some("pending A"), first_missing));

        let first = snapshot_asset_texture_state(&assets, texture_handle);
        assert!(first.ready.is_none());
        assert_eq!(first.image_handle, Some(first_missing));

        assets.textures.insert_ready(
            texture_handle,
            Texture::new_2d(Some("pending B"), second_missing),
        );
        let second = snapshot_asset_texture_state(&assets, texture_handle);
        assert!(second.ready.is_none());
        assert_eq!(second.texture_version, first.texture_version);
        assert_ne!(second.texture_identity, first.texture_identity);
        assert_eq!(second.image_handle, Some(second_missing));
        assert_ne!(first, second);
    }

    #[test]
    fn incompatible_texture_specs_own_independent_physical_images_and_mip_requests() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image = assets.images.add(image_8x8());

        let srgb = assets
            .textures
            .add(Texture::new_2d(Some("Shared Image sRGB"), image));
        let mut linear_asset = Texture::new_2d(Some("Shared Image Linear"), image);
        linear_asset.color_space = ColorSpace::Linear;
        linear_asset.generate_mipmaps = true;
        let linear = assets.textures.add(linear_asset);

        assert!(matches!(
            resources.prepare_texture(&assets, srgb),
            ResourceState::Ready
        ));
        assert!(matches!(
            resources.prepare_texture(&assets, linear),
            ResourceState::Ready
        ));

        let srgb_image = resources
            .get_texture_image(srgb)
            .expect("sRGB image missing");
        let linear_image = resources
            .get_texture_image(linear)
            .expect("linear image missing");
        let srgb_physical = resources
            .resources
            .texture_image_lease(srgb)
            .expect("sRGB lease missing")
            .physical_id;
        let linear_physical = resources
            .resources
            .texture_image_lease(linear)
            .expect("linear lease missing")
            .physical_id;
        assert_ne!(srgb_physical, linear_physical);
        assert_ne!(srgb_image.id, linear_image.id);
        assert_eq!(srgb_image.format, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert_eq!(srgb_image.mip_level_count, 1);
        assert_eq!(linear_image.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(linear_image.mip_level_count, 4);

        let mut srgb_changed = Texture::new_2d(Some("sRGB now mipped"), image);
        srgb_changed.generate_mipmaps = true;
        assert_eq!(assets.textures.update(srgb, srgb_changed), Some(2));
        let old_srgb_id = srgb_image.id;
        assert!(matches!(
            resources.prepare_texture(&assets, srgb),
            ResourceState::Ready
        ));
        let rebuilt_srgb = resources
            .get_texture_image(srgb)
            .expect("rebuilt sRGB image missing");
        let mipped_srgb_physical = resources
            .resources
            .texture_image_lease(srgb)
            .expect("rebuilt sRGB lease missing")
            .physical_id;
        assert_ne!(mipped_srgb_physical, srgb_physical);
        assert_ne!(rebuilt_srgb.id, old_srgb_id);
        assert_eq!(rebuilt_srgb.mip_level_count, 4);
        let mipped_srgb_id = rebuilt_srgb.id;
        let binding = resources
            .get_texture_binding(srgb)
            .expect("sRGB binding missing");
        assert_eq!(binding.texture_version, 2);
        assert_eq!(binding.image_version, 1);

        let srgb_generation = resources
            .resources
            .texture_generation_key(srgb)
            .expect("sRGB generation missing");
        let linear_generation = resources
            .resources
            .texture_generation_key(linear)
            .expect("linear generation missing");

        let requests = resources.take_mipmap_requests();
        assert_eq!(requests.len(), 2);
        assert!(
            requests
                .iter()
                .any(|request| request.key() == srgb_generation)
        );
        assert!(
            requests
                .iter()
                .any(|request| request.key() == linear_generation)
        );
        assert_ne!(requests[0].key(), requests[1].key());

        // Leave a fresh request pending so the true -> false transition also
        // proves that recreating the physical allocation releases the
        // obsolete generation owner.
        assert!(matches!(
            resources.prepare_texture(&assets, srgb),
            ResourceState::Ready
        ));

        let srgb_without_mips = Texture::new_2d(Some("sRGB mips disabled"), image);
        assert_eq!(assets.textures.update(srgb, srgb_without_mips), Some(3));
        assert!(matches!(
            resources.prepare_texture(&assets, srgb),
            ResourceState::Ready
        ));
        let rebuilt_without_mips = resources
            .get_texture_image(srgb)
            .expect("sRGB no-mip image missing");
        let no_mip_physical = resources
            .resources
            .texture_image_lease(srgb)
            .expect("sRGB no-mip lease missing")
            .physical_id;
        assert_ne!(no_mip_physical, mipped_srgb_physical);
        assert_ne!(rebuilt_without_mips.id, mipped_srgb_id);
        assert_eq!(rebuilt_without_mips.mip_level_count, 1);
        assert!(resources.take_mipmap_requests().is_empty());
    }

    #[test]
    fn exact_spec_handles_share_physical_image_and_release_per_lease() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image_handle = assets.images.add(image_8x8());

        let mut first_texture = Texture::new_2d(Some("shared first"), image_handle);
        first_texture.generate_mipmaps = true;
        let first = assets.textures.add(first_texture);
        let mut second_texture = Texture::new_2d(Some("shared nearest"), image_handle);
        second_texture.generate_mipmaps = true;
        second_texture.sampler.mag_filter = wgpu::FilterMode::Nearest;
        let second = assets.textures.add(second_texture);

        assert!(matches!(
            resources.prepare_texture(&assets, first),
            ResourceState::Ready
        ));
        assert!(matches!(
            resources.prepare_texture(&assets, second),
            ResourceState::Ready
        ));
        resources.resources.assert_texture_pool_consistent();

        let first_lease = resources
            .resources
            .texture_image_lease(first)
            .expect("first lease missing");
        let second_lease = resources
            .resources
            .texture_image_lease(second)
            .expect("second lease missing");
        assert_eq!(first_lease.physical_id, second_lease.physical_id);
        assert_eq!(
            resources
                .resources
                .physical_image_member_count(first_lease.physical_id),
            Some(2)
        );
        assert_eq!(
            resources
                .get_texture_image(first)
                .expect("first physical image missing")
                .id,
            resources
                .get_texture_image(second)
                .expect("second physical image missing")
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
            "sampler state must remain TextureHandle-local"
        );

        let generation = resources
            .resources
            .texture_generation_key(first)
            .expect("shared generation missing");
        let mut initial_requests = resources.take_mipmap_requests();
        assert_eq!(initial_requests.len(), 1);
        let old_graph_owner = initial_requests
            .pop()
            .expect("shared generation owner missing");
        assert_eq!(old_graph_owner.key(), generation);
        assert!(matches!(
            resources.prepare_texture(&assets, first),
            ResourceState::Ready
        ));

        assert_eq!(
            assets.textures.update(
                first,
                Texture::new_2d(Some("first migrated without mips"), image_handle),
            ),
            Some(2)
        );
        assert!(matches!(
            resources.prepare_texture(&assets, first),
            ResourceState::Ready
        ));
        let migrated_physical = resources
            .resources
            .texture_image_lease(first)
            .expect("migrated lease missing")
            .physical_id;
        assert_ne!(migrated_physical, second_lease.physical_id);
        assert_eq!(
            resources
                .resources
                .physical_image_member_count(migrated_physical),
            Some(1)
        );
        assert_eq!(
            resources
                .resources
                .physical_image_member_count(second_lease.physical_id),
            Some(1)
        );

        resources.resources.remove_texture_image(first);
        resources.resources.assert_texture_pool_consistent();
        assert!(resources.get_texture_image(first).is_none());
        assert!(resources.get_texture_binding(first).is_none());
        assert!(resources.get_texture_image(second).is_some());
        assert_eq!(
            resources
                .resources
                .physical_image_member_count(second_lease.physical_id),
            Some(1)
        );
        let pending = resources.take_mipmap_requests();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].key(), generation);

        assert!(matches!(
            resources.prepare_texture(&assets, second),
            ResourceState::Ready
        ));
        resources.resources.remove_texture_image(second);
        resources.resources.assert_texture_pool_consistent();
        assert!(
            !resources
                .resources
                .contains_physical_image(second_lease.physical_id)
        );
        assert!(resources.take_mipmap_requests().is_empty());

        let mut replacement_texture = Texture::new_2d(Some("replacement owner"), image_handle);
        replacement_texture.generate_mipmaps = true;
        let replacement = assets.textures.add(replacement_texture);
        assert!(matches!(
            resources.prepare_texture(&assets, replacement),
            ResourceState::Ready
        ));
        let replacement_physical = resources
            .resources
            .texture_image_lease(replacement)
            .expect("replacement lease missing")
            .physical_id;
        assert_ne!(replacement_physical, second_lease.physical_id);

        old_graph_owner.mark_submitted();
        assert!(
            !resources
                .get_texture_image(replacement)
                .expect("replacement physical image missing")
                .mipmaps_submitted(),
            "an old graph owner must not publish a reused physical slot"
        );
        resources.resources.queue_mipmap_request(old_graph_owner);
        let replacement_requests = resources.take_mipmap_requests();
        assert_eq!(replacement_requests.len(), 1);
        assert_eq!(replacement_requests[0].physical_id, replacement_physical);
        resources.resources.assert_texture_pool_consistent();
    }

    #[test]
    fn shared_content_refresh_invalidates_other_lease_until_revalidated() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image_handle = assets.images.add(image_8x8());
        let first = assets
            .textures
            .add(Texture::new_2d(Some("content first"), image_handle));
        let second = assets
            .textures
            .add(Texture::new_2d(Some("content second"), image_handle));

        assert!(matches!(
            resources.prepare_texture(&assets, first),
            ResourceState::Ready
        ));
        assert!(matches!(
            resources.prepare_texture(&assets, second),
            ResourceState::Ready
        ));
        resources.resources.assert_texture_pool_consistent();
        let physical_id = resources
            .resources
            .texture_image_lease(first)
            .expect("shared lease missing")
            .physical_id;
        let allocation = resources
            .get_texture_image(first)
            .expect("shared image missing")
            .id;

        let replacement = Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![23; 8 * 8 * 4]),
        );
        assert_eq!(assets.images.update(image_handle, replacement), Some(2));
        assert!(matches!(
            resources.prepare_texture(&assets, first),
            ResourceState::Ready
        ));
        let refreshed_generation = resources
            .get_texture_image(first)
            .expect("refreshed image missing")
            .generation_id;
        assert!(
            resources.get_texture_binding(second).is_none(),
            "an unvalidated logical lease must not publish refreshed physical content"
        );

        assert!(matches!(
            resources.prepare_texture(&assets, second),
            ResourceState::Ready
        ));
        let second_image = resources
            .get_texture_image(second)
            .expect("revalidated image missing");
        assert_eq!(second_image.id, allocation);
        assert_eq!(second_image.generation_id, refreshed_generation);
        assert_eq!(
            resources
                .resources
                .texture_image_lease(second)
                .expect("second lease missing")
                .physical_id,
            physical_id
        );
        assert_eq!(
            resources.resources.physical_image_member_count(physical_id),
            Some(2)
        );
    }

    #[test]
    fn physical_pending_queue_rejects_a_late_stale_generation() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image_handle = assets.images.add(image_8x8());
        let mut texture = Texture::new_2d(Some("Physical pending"), image_handle);
        texture.generate_mipmaps = true;
        let texture_handle = assets.textures.add(texture);

        assert!(matches!(
            resources.prepare_texture(&assets, texture_handle),
            ResourceState::Ready
        ));
        let mut first = resources.take_mipmap_requests();
        assert_eq!(first.len(), 1);
        let stale = first.pop().expect("initial mip request missing");
        let physical_id = stale.physical_id;
        let stale_generation = stale.generation_id;
        let stale_after_drain = resources
            .resources
            .texture_mipmap_request(texture_handle)
            .expect("second stale request missing");

        let replacement = Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![17; 8 * 8 * 4]),
        );
        assert_eq!(assets.images.update(image_handle, replacement), Some(2));
        assert!(matches!(
            resources.prepare_texture(&assets, texture_handle),
            ResourceState::Ready
        ));
        let current_generation = resources
            .get_texture_image(texture_handle)
            .expect("updated physical image missing")
            .generation_id;
        assert_ne!(current_generation, stale_generation);
        assert_eq!(
            resources
                .resources
                .texture_image_lease(texture_handle)
                .expect("updated lease missing")
                .physical_id,
            physical_id,
            "content updates must preserve the physical allocation"
        );

        stale.mark_submitted();
        assert!(
            !resources
                .get_texture_image(texture_handle)
                .expect("current physical image missing")
                .mipmaps_submitted(),
            "publishing an old generation must not make the current one ready"
        );
        resources.resources.queue_mipmap_request(stale);
        let pending = resources.take_mipmap_requests();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].physical_id, physical_id);
        assert_eq!(pending[0].generation_id, current_generation);
        pending[0].mark_submitted();
        assert!(
            resources
                .get_texture_image(texture_handle)
                .expect("submitted physical image missing")
                .mipmaps_submitted()
        );

        stale_after_drain.mark_submitted();
        assert!(
            resources
                .get_texture_image(texture_handle)
                .expect("current physical image missing after stale publication")
                .mipmaps_submitted(),
            "late publication of an old generation must not regress readiness"
        );
        resources.resources.queue_mipmap_request(stale_after_drain);
        assert!(
            resources.take_mipmap_requests().is_empty(),
            "a stale generation must remain rejected after the current request was drained"
        );
    }

    #[test]
    fn same_handle_same_version_asset_replacements_refresh_gpu_identity() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image_handle = assets.images.add(image_8x8());
        let mut texture = Texture::new_2d(Some("ABA sRGB"), image_handle);
        texture.generate_mipmaps = true;
        let texture_handle = assets.textures.add(texture);

        let first_snapshot =
            snapshot_asset_texture(&assets, texture_handle).expect("initial snapshot missing");
        assert!(matches!(
            resources.prepare_texture(&assets, texture_handle),
            ResourceState::Ready
        ));
        let first_binding = *resources
            .get_texture_binding(texture_handle)
            .expect("initial binding missing");
        let first_physical_id = resources
            .get_texture_image(texture_handle)
            .expect("initial GPU image missing")
            .id;
        let first_pool_id = resources
            .resources
            .texture_image_lease(texture_handle)
            .expect("initial texture lease missing")
            .physical_id;

        let original_texture_uuid = assets
            .textures
            .get(texture_handle)
            .expect("initial texture missing")
            .uuid;
        let mut replacement = Texture::new_2d(Some("ABA linear"), image_handle);
        replacement.uuid = original_texture_uuid;
        replacement.color_space = ColorSpace::Linear;
        replacement.generate_mipmaps = true;
        assets.textures.insert_ready(texture_handle, replacement);

        let replacement_snapshot =
            snapshot_asset_texture(&assets, texture_handle).expect("replacement snapshot missing");
        assert_eq!(replacement_snapshot.texture_version, 1);
        assert_eq!(replacement_snapshot.image_version, 1);
        assert_ne!(
            replacement_snapshot.texture_identity,
            first_snapshot.texture_identity
        );
        assert_ne!(
            replacement_snapshot.logical_version,
            first_snapshot.logical_version
        );

        assert!(matches!(
            resources.prepare_texture(&assets, texture_handle),
            ResourceState::Ready
        ));
        let replacement_binding = *resources
            .get_texture_binding(texture_handle)
            .expect("replacement binding missing");
        let replacement_physical_id = {
            let image = resources
                .get_texture_image(texture_handle)
                .expect("replacement GPU image missing");
            assert_eq!(image.format, wgpu::TextureFormat::Rgba8Unorm);
            image.id
        };
        let replacement_pool_id = resources
            .resources
            .texture_image_lease(texture_handle)
            .expect("replacement texture lease missing")
            .physical_id;
        assert_ne!(replacement_physical_id, first_physical_id);
        assert_ne!(replacement_pool_id, first_pool_id);
        assert_ne!(
            replacement_binding.texture_identity,
            first_binding.texture_identity
        );
        assert_eq!(replacement_binding.sampler_id, first_binding.sampler_id);
        let requests = resources.take_mipmap_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].key().allocation, replacement_physical_id);

        // A sampler-only same-version replacement reuses the physical view.
        // Graph-owned materialization must still publish the replacement's
        // exact sampler binding without leaking its mip request into the
        // legacy queue.
        let mut sampler_replacement = Texture::new_2d(Some("ABA nearest"), image_handle);
        sampler_replacement.uuid = original_texture_uuid;
        sampler_replacement.color_space = ColorSpace::Linear;
        sampler_replacement.generate_mipmaps = true;
        sampler_replacement.sampler.mag_filter = wgpu::FilterMode::Nearest;
        assets
            .textures
            .insert_ready(texture_handle, sampler_replacement);
        let sampler_snapshot =
            snapshot_asset_texture(&assets, texture_handle).expect("sampler snapshot missing");
        let deferred = resources.materialize_asset_texture_for_graph(&assets, sampler_snapshot);
        assert!(deferred.texture.is_some());
        assert!(deferred.mipmap_request.is_some());
        assert_eq!(
            resources
                .get_texture_image(texture_handle)
                .expect("deferred image missing")
                .id,
            replacement_physical_id
        );
        assert_eq!(
            resources
                .resources
                .texture_image_lease(texture_handle)
                .expect("deferred sampler lease missing")
                .physical_id,
            replacement_pool_id
        );
        let graph_binding = *resources
            .get_texture_binding(texture_handle)
            .expect("graph materialization did not publish its exact binding");
        assert_ne!(graph_binding.sampler_id, replacement_binding.sampler_id);
        assert_eq!(
            graph_binding.texture_identity,
            sampler_snapshot.texture_identity
        );
        assert!(
            resources.take_mipmap_requests().is_empty(),
            "graph materialization must not enqueue a legacy mip request"
        );
        drop(deferred);
        // The graph attempt above deliberately did not submit its mip request;
        // normal compatibility prepare therefore observes the same incomplete
        // generation and may queue an independently owned retry.
        assert!(matches!(
            resources.prepare_texture(&assets, texture_handle),
            ResourceState::Ready
        ));
        let sampler_binding = *resources
            .get_texture_binding(texture_handle)
            .expect("sampler replacement binding missing");
        assert_ne!(sampler_binding.sampler_id, replacement_binding.sampler_id);
        assert_eq!(
            sampler_binding.texture_identity,
            sampler_snapshot.texture_identity
        );
        let (replacement_generation, replacement_image_identity) = {
            let image = resources
                .get_texture_image(texture_handle)
                .expect("sampler replacement GPU image missing");
            assert_eq!(image.id, replacement_physical_id);
            (image.generation_id, image.image_identity)
        };
        assert_eq!(resources.take_mipmap_requests().len(), 1);

        let original_image_uuid = assets
            .images
            .get(image_handle)
            .expect("initial image missing")
            .uuid;
        let mut replacement_image = Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![31; 8 * 8 * 4]),
        );
        replacement_image.uuid = original_image_uuid;
        assets.images.insert_ready(image_handle, replacement_image);

        assert!(matches!(
            resources.prepare_texture(&assets, texture_handle),
            ResourceState::Ready
        ));
        let updated_binding = *resources
            .get_texture_binding(texture_handle)
            .expect("updated image binding missing");
        let updated_image = resources
            .get_texture_image(texture_handle)
            .expect("updated GPU image missing");
        assert_eq!(updated_image.id, replacement_physical_id);
        assert_eq!(
            resources
                .resources
                .texture_image_lease(texture_handle)
                .expect("updated image lease missing")
                .physical_id,
            replacement_pool_id
        );
        assert_ne!(updated_image.image_identity, replacement_image_identity);
        assert_ne!(updated_image.generation_id, replacement_generation);
        assert_eq!(updated_binding.image_version, 1);
        assert_eq!(updated_binding.image_identity, updated_image.image_identity);
        let updated_generation_key = resources
            .resources
            .texture_generation_key(texture_handle)
            .expect("updated texture generation missing");
        let requests = resources.take_mipmap_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].key(), updated_generation_key);
    }

    #[test]
    fn ttl_prune_releases_mipmap_requests_and_repairs_image_aliases() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image_handle = assets.images.add(image_8x8());

        let mut stale_texture = Texture::new_2d(Some("stale mipped"), image_handle);
        stale_texture.generate_mipmaps = true;
        let stale = assets.textures.add(stale_texture);
        let mut survivor_texture = Texture::new_2d(Some("surviving alias"), image_handle);
        survivor_texture.generate_mipmaps = true;
        let survivor = assets.textures.add(survivor_texture);

        assert!(matches!(
            resources.prepare_texture(&assets, stale),
            ResourceState::Ready
        ));
        let stale_physical = resources
            .resources
            .texture_image_lease(stale)
            .expect("stale lease missing")
            .physical_id;
        resources.next_frame();
        assert!(matches!(
            resources.prepare_texture(&assets, survivor),
            ResourceState::Ready
        ));
        let survivor_id = resources
            .get_texture_image(survivor)
            .expect("survivor missing")
            .id;
        let survivor_physical = resources
            .resources
            .texture_image_lease(survivor)
            .expect("survivor lease missing")
            .physical_id;
        assert_eq!(
            stale_physical, survivor_physical,
            "identical specifications must share one physical image"
        );

        // Simulate the compatibility alias having last pointed at an older
        // interpretation of the same Image. Prune must repair it to the live
        // survivor rather than leaving the lookup on its O(n) fallback.
        resources
            .resources
            .touch_texture_image_alias(image_handle, stale);
        assert_eq!(
            resources.resources.latest_texture_alias(image_handle),
            Some(stale)
        );

        resources.next_frame();
        resources.prune(1);
        resources.resources.assert_texture_pool_consistent();

        assert!(resources.get_texture_image(stale).is_none());
        assert!(
            resources
                .resources
                .contains_physical_image(survivor_physical)
        );
        assert_eq!(
            resources
                .resources
                .physical_image_member_count(survivor_physical),
            Some(1)
        );
        assert_eq!(
            resources
                .get_texture_image(survivor)
                .expect("live alias was pruned")
                .id,
            survivor_id
        );
        assert!(resources.get_texture_binding(stale).is_none());
        assert!(matches!(
            resources.prepare_texture(&assets, survivor),
            ResourceState::Ready
        ));
        let pending = resources.take_mipmap_requests();
        assert_eq!(
            pending.len(),
            1,
            "pruning one lease must retain the shared physical generation"
        );
        assert_eq!(
            pending[0].key(),
            resources
                .resources
                .texture_generation_key(survivor)
                .expect("survivor generation missing")
        );
        assert_eq!(
            resources.resources.latest_texture_alias(image_handle),
            Some(survivor)
        );
        assert_eq!(
            resources.get_image(image_handle).expect("alias missing").id,
            survivor_id
        );

        let moved = assets
            .textures
            .add(Texture::new_2d(Some("moved alias"), image_handle));
        assert!(matches!(
            resources.prepare_texture(&assets, moved),
            ResourceState::Ready
        ));
        assert_eq!(
            resources.resources.latest_texture_alias(image_handle),
            Some(moved)
        );
        let other_image = assets.images.add(image_8x8());
        assert_eq!(
            assets.textures.update(
                moved,
                Texture::new_2d(Some("moved to another image"), other_image),
            ),
            Some(2)
        );
        assert!(matches!(
            resources.prepare_texture(&assets, moved),
            ResourceState::Ready
        ));
        assert_eq!(
            resources.resources.latest_texture_alias(image_handle),
            Some(survivor),
            "changing a TextureHandle's Image must repair the old compatibility alias immediately"
        );
        assert_eq!(
            resources.resources.latest_texture_alias(other_image),
            Some(moved)
        );
    }
}
