//! GPU buffer storage and lifecycle management.
//!
//! # Architecture
//!
//! GPU buffers are stored in a [`SlotMap`] keyed by [`GpuBufferHandle`], providing
//! O(1) insertion, removal, and lookup by handle. A secondary index
//! (`buffer_index: FxHashMap<u64, GpuBufferHandle>`) maps CPU-side buffer IDs to
//! their corresponding slot, bridging the gap for code paths that only have a
//! `BufferRef` or raw `u64` id.
//!
//! [`CpuBuffer`] caches its assigned [`GpuBufferHandle`] in an `AtomicU64` field,
//! enabling a lock-free O(1) fast path in [`ResourceManager::ensure_buffer`].
//! Every arena entry also records its CPU logical ID, so a handle is only a
//! hint until both its SlotMap generation and exact owner have been validated.
//!
//! # Resize Strategy
//!
//! When incoming data exceeds the current `wgpu::Buffer` capacity, the old buffer
//! is destroyed and a new, larger one is created **in the same SlotMap slot**. The
//! handle remains stable, but `GpuBuffer::id` is regenerated to signal downstream
//! consumers (e.g. `ResourceIdSet` fingerprints) that the physical resource has
//! changed and dependent `BindGroup`s must be rebuilt.

use myth_assets::AssetServer;
use myth_resources::buffer::{BufferRef, CpuBufferStamp};
use myth_resources::texture::TextureSource;

use super::environment::{snapshot_attachment_texture_source, snapshot_ready_asset_texture_source};
use super::{
    EnsureResult, GpuImageSpec, InternalTextureRegistry, ResourceManager, Tracked,
    generate_gpu_resource_id,
};

/// Narrow post-compile capability for persistent CPU→GPU uploads and texture
/// source resolution.
///
/// This context borrows the GPU resource store plus an immutable internal
/// texture registry. It can update persistent buffers, materialize a ready
/// asset texture, or resolve an attachment without exposing the monolithic
/// [`ResourceManager`] to live pass preparation.
pub struct BufferUploadContext<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    frame_index: u64,
    resources: &'a mut super::GpuResourceStore,
    internal_textures: &'a InternalTextureRegistry,
}

/// Physical texture binding resolved during live pass preparation.
///
/// `view_id` fingerprints the concrete WGPU view for bind-group caching, while
/// `view_dimension` lets callers validate the pipeline/layout variant selected
/// from their earlier CPU-only source snapshot.
pub(crate) struct PreparedTextureSource {
    pub(crate) view: wgpu::TextureView,
    pub(crate) view_id: u64,
    pub(crate) view_dimension: wgpu::TextureViewDimension,
    pub(crate) logical_version: u64,
}

/// Exact, allocation-free identity of one post-DPE materialized CPU buffer.
///
/// The SlotMap handle is generation checked, while `logical_id` on the arena
/// entry and `physical_id` below close the remaining wrong-owner and in-slot
/// resize/ABA cases. Keeping this token Copy lets graph-attempt state retain
/// the exact physical result without cloning a `wgpu::Buffer` on the warm path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MaterializedBufferToken {
    pub(crate) stamp: CpuBufferStamp,
    pub(crate) handle: GpuBufferHandle,
    pub(crate) physical_id: u64,
}

impl MaterializedBufferToken {
    #[inline]
    #[must_use]
    pub(crate) const fn new(
        stamp: CpuBufferStamp,
        handle: GpuBufferHandle,
        physical_id: u64,
    ) -> Self {
        Self {
            stamp,
            handle,
            physical_id,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn buffer_ref(self) -> BufferRef {
        BufferRef::with_fixed_id(
            self.stamp.id,
            self.stamp.size,
            self.stamp.usage,
            self.stamp.version,
            None,
        )
    }
}

type BufferSnapshotInput<'a> = Option<(&'a BufferRef, &'a [u8])>;

#[derive(Clone, Copy)]
pub(crate) struct HintedBufferSnapshotInput<'a> {
    pub(crate) stamp: CpuBufferStamp,
    pub(crate) bytes: &'a [u8],
    pub(crate) cached_handle: Option<GpuBufferHandle>,
}

#[inline]
fn buffer_snapshot_ids_are_unique<const N: usize>(
    snapshots: &[BufferSnapshotInput<'_>; N],
) -> bool {
    snapshots.iter().enumerate().all(|(index, current)| {
        let Some((current, _)) = current else {
            return true;
        };
        snapshots[..index]
            .iter()
            .flatten()
            .all(|(previous, _)| previous.id() != current.id())
    })
}

#[inline]
fn hinted_buffer_snapshot_ids_are_unique<const N: usize>(
    snapshots: &[HintedBufferSnapshotInput<'_>; N],
) -> bool {
    snapshots.iter().enumerate().all(|(index, current)| {
        snapshots[..index]
            .iter()
            .all(|previous| previous.stamp.id != current.stamp.id)
    })
}

#[cfg(test)]
mod snapshot_batch_tests {
    use super::*;

    #[test]
    fn duplicate_logical_buffer_ids_fail_before_materialization() {
        let first = BufferRef::with_fixed_id(
            7,
            4,
            wgpu::BufferUsages::COPY_DST,
            1,
            Some("first duplicate snapshot"),
        );
        let second = BufferRef::with_fixed_id(
            7,
            8,
            wgpu::BufferUsages::COPY_DST,
            2,
            Some("second duplicate snapshot"),
        );
        let first_bytes = [1_u8; 4];
        let second_bytes = [2_u8; 8];

        assert!(!buffer_snapshot_ids_are_unique(&[
            Some((&first, &first_bytes)),
            Some((&second, &second_bytes)),
        ]));
        assert!(buffer_snapshot_ids_are_unique(&[
            Some((&first, &first_bytes)),
            None,
        ]));
    }
}

impl<'a> BufferUploadContext<'a> {
    pub(crate) fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        frame_index: u64,
        resources: &'a mut super::GpuResourceStore,
        internal_textures: &'a InternalTextureRegistry,
    ) -> Self {
        Self {
            device,
            queue,
            frame_index,
            resources,
            internal_textures,
        }
    }

    /// Ensure a persistent buffer exists and contains the latest CPU version.
    pub fn ensure_buffer<T: super::GpuData>(
        &mut self,
        cpu_buffer: &super::CpuBuffer<T>,
    ) -> (GpuBufferHandle, EnsureResult) {
        if let Some(handle) = cpu_buffer.gpu_handle() {
            if let Some(gpu_buf) = self.resources.buffer_mut(handle)
                && gpu_buf.logical_id == cpu_buffer.id()
            {
                let buffer_ref = cpu_buffer.handle();
                let mut was_recreated = false;

                if buffer_ref.version > gpu_buf.last_uploaded_version {
                    let guard = cpu_buffer.read();
                    let data: &[u8] = bytemuck::cast_slice(guard.as_bytes());
                    was_recreated = gpu_buf.write_to_gpu(self.device, self.queue, data);
                    gpu_buf.last_uploaded_version = buffer_ref.version;
                }
                gpu_buf.last_used_frame = self.frame_index;
                return (handle, EnsureResult::new(gpu_buf.id, was_recreated));
            }
            cpu_buffer.clear_gpu_handle();
        }

        let buffer_ref = cpu_buffer.handle();
        let guard = cpu_buffer.read();
        let data: &[u8] = bytemuck::cast_slice(guard.as_bytes());
        let (handle, result) = self.resources.write_buffer(
            self.device,
            self.queue,
            self.frame_index,
            &buffer_ref,
            data,
        );
        cpu_buffer.set_gpu_handle(handle);
        (handle, result)
    }

    /// Materialize one fixed semantic batch after validating every member
    /// before the first mutation.
    ///
    /// Distinct logical buffer IDs cannot invalidate one another while this
    /// context holds the resource store exclusively, so each member needs only
    /// the batch preflight plus the canonical `write_buffer` lookup. Duplicate
    /// IDs fail closed before mutation: two versions of one physical buffer
    /// cannot both remain exact after the later upload overwrites or resizes it.
    pub(crate) fn materialize_buffer_snapshot_batch<const N: usize>(
        &mut self,
        snapshots: [BufferSnapshotInput<'_>; N],
    ) -> Option<[Option<Tracked<wgpu::Buffer>>; N]> {
        if !buffer_snapshot_ids_are_unique(&snapshots)
            || snapshots
                .iter()
                .flatten()
                .any(|&(buffer_ref, bytes)| !self.preflight_buffer_snapshot(buffer_ref, bytes))
        {
            return None;
        }

        let mut failed = false;
        let materialized = std::array::from_fn(|index| {
            if failed {
                return None;
            }
            let (buffer_ref, bytes) = snapshots[index]?;
            let buffer = self.materialize_buffer_snapshot_after_preflight(buffer_ref, bytes);
            if buffer.is_none() {
                failed = true;
            }
            buffer
        });
        (!failed).then_some(materialized)
    }

    /// Materialize an exact fixed-size batch using each `CpuBuffer`'s cached
    /// arena handle as a generation-checked fast-path hint.
    ///
    /// Every member is preflighted before the first mutation. A current hint
    /// resolves directly through SlotMap and verifies the arena entry's CPU
    /// logical ID, avoiding the reverse-index hash lookup. Missing or stale
    /// hints conservatively fall back to the canonical CPU-ID path.
    pub(crate) fn materialize_hinted_buffer_snapshot_batch<const N: usize>(
        &mut self,
        snapshots: [HintedBufferSnapshotInput<'_>; N],
    ) -> Option<[MaterializedBufferToken; N]> {
        if !hinted_buffer_snapshot_ids_are_unique(&snapshots)
            || snapshots.iter().any(|snapshot| {
                !self.preflight_hinted_buffer_snapshot(
                    snapshot.stamp,
                    snapshot.bytes,
                    snapshot.cached_handle,
                )
            })
        {
            return None;
        }

        let mut materialized = [None; N];
        for (index, snapshot) in snapshots.into_iter().enumerate() {
            let (handle, result) = self.resources.write_buffer_snapshot_with_hint(
                self.device,
                self.queue,
                self.frame_index,
                snapshot.stamp,
                snapshot.bytes,
                snapshot.cached_handle,
            );
            materialized[index] = Some(MaterializedBufferToken::new(
                snapshot.stamp,
                handle,
                result.resource_id,
            ));
        }

        Some(materialized.map(|token| token.expect("materialized batch member missing")))
    }

    fn materialize_buffer_snapshot_after_preflight(
        &mut self,
        buffer_ref: &BufferRef,
        bytes: &[u8],
    ) -> Option<Tracked<wgpu::Buffer>> {
        let handle = self
            .resources
            .write_buffer(self.device, self.queue, self.frame_index, buffer_ref, bytes)
            .0;
        let gpu_buffer = self
            .resources
            .buffer(handle)
            .expect("freshly ensured GPU buffer handle is missing");
        (gpu_buffer.last_uploaded_version == buffer_ref.version
            && gpu_buffer.usage == buffer_ref.usage()
            && gpu_buffer.size >= bytes.len() as u64)
            .then(|| Tracked::with_id(gpu_buffer.buffer.clone(), gpu_buffer.id))
    }

    /// Validate an exact snapshot without mutating ResourceManager state.
    ///
    /// A fixed-size batch validates every live input before the first GPU
    /// upload. An equal-version allocation must already cover the snapshot's
    /// logical byte range because `write_buffer` correctly skips an upload for
    /// an identical version and therefore cannot resize it afterward.
    #[inline]
    #[must_use]
    pub(crate) fn preflight_buffer_snapshot(&self, buffer_ref: &BufferRef, bytes: &[u8]) -> bool {
        if bytes.len() != buffer_ref.size() {
            return false;
        }
        let Some(buffer) = self.resources.buffer_by_cpu_id(buffer_ref.id()) else {
            return true;
        };
        buffer.usage == buffer_ref.usage()
            && buffer.last_uploaded_version <= buffer_ref.version
            && (buffer.last_uploaded_version != buffer_ref.version
                || buffer.size >= bytes.len() as u64)
    }

    #[inline]
    #[must_use]
    fn preflight_hinted_buffer_snapshot(
        &self,
        stamp: CpuBufferStamp,
        bytes: &[u8],
        cached_handle: Option<GpuBufferHandle>,
    ) -> bool {
        if bytes.len() != stamp.size {
            return false;
        }
        let Some(buffer) = self
            .resources
            .buffer_by_cpu_id_with_hint(stamp.id, cached_handle)
        else {
            return true;
        };
        buffer.usage == stamp.usage
            && buffer.last_uploaded_version <= stamp.version
            && (buffer.last_uploaded_version != stamp.version || buffer.size >= bytes.len() as u64)
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn get_gpu_buffer(&self, handle: GpuBufferHandle) -> Option<&GpuBuffer> {
        self.resources.buffer(handle)
    }

    /// Materialize and resolve one texture source during live-node prepare.
    ///
    /// Asset decoding readiness is decided during the CPU-only snapshot stage.
    /// This method verifies that exact logical version before performing the
    /// deferred GPU upload, after DPE has kept the source-conversion node alive.
    /// The returned view is cloned deliberately so callers can release the
    /// mutable upload borrow before assembling a bind group from the other
    /// `PrepareContext` partitions.
    pub(crate) fn resolve_texture_source(
        &mut self,
        assets: &AssetServer,
        source: TextureSource,
        expected_logical_version: u64,
    ) -> Option<PreparedTextureSource> {
        match source {
            TextureSource::Asset(handle) => {
                let (texture_asset, texture_version, texture_identity) =
                    assets.textures.get_entry_with_identity(handle)?;
                let image_handle = texture_asset.image;
                let (image, image_version, image_identity) =
                    assets.images.get_entry_with_identity(image_handle)?;
                let logical = snapshot_ready_asset_texture_source(
                    source,
                    &texture_asset,
                    texture_version,
                    texture_identity,
                    &image,
                    image_version,
                    image_identity,
                );
                if !logical.ready || logical.version != expected_logical_version {
                    return None;
                }
                let resolved_format = texture_asset.resolve_wgpu_format(image.format);
                let required_usage =
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;

                // Source-conversion shaders sample level zero explicitly. Keep
                // an existing full mip allocation when compatible, but create
                // only a base-level allocation for a source used solely here.
                let base_spec = GpuImageSpec {
                    image_handle,
                    extent: wgpu::Extent3d {
                        width: image.width,
                        height: image.height,
                        depth_or_array_layers: image.depth,
                    },
                    format: resolved_format,
                    dimension: image.dimension.to_wgpu(),
                    view_dimension: texture_asset.view_dimension,
                    mip_count: 1,
                    sample_count: 1,
                    usage: required_usage,
                };

                let materialization_spec = self
                    .resources
                    .texture_image(handle)
                    .filter(|gpu_image| {
                        gpu_image.image_handle == base_spec.image_handle
                            && gpu_image.size == base_spec.extent
                            && gpu_image.default_view_dimension == base_spec.view_dimension
                            && gpu_image.dimension == base_spec.dimension
                            && gpu_image.format == base_spec.format
                            && gpu_image.sample_count == base_spec.sample_count
                            && gpu_image.usage.contains(required_usage)
                    })
                    .map_or(base_spec, |gpu_image| GpuImageSpec {
                        image_handle: gpu_image.image_handle,
                        extent: gpu_image.size,
                        format: gpu_image.format,
                        dimension: gpu_image.dimension,
                        view_dimension: gpu_image.default_view_dimension,
                        mip_count: gpu_image.mip_level_count,
                        sample_count: gpu_image.sample_count,
                        usage: gpu_image.usage,
                    });

                self.resources.materialize_texture_image(
                    self.device,
                    self.queue,
                    self.frame_index,
                    handle,
                    &image,
                    materialization_spec,
                    texture_version,
                    texture_identity,
                    image_version,
                    image_identity,
                );

                // `Image` data is mutable behind its Arc. Re-read both storage
                // versions after `GpuImage::new/update` has consumed the bytes,
                // so an in-place edit cannot be published under the older
                // Composer snapshot key.
                let texture_unchanged = assets
                    .textures
                    .get_entry_with_identity(handle)
                    .is_some_and(|(current, current_version, current_identity)| {
                        current_version == texture_version
                            && current_identity == texture_identity
                            && current.image == image_handle
                    });
                let image_unchanged = assets
                    .images
                    .get_entry_with_identity(image_handle)
                    .is_some_and(|(_, current_version, current_identity)| {
                        current_version == image_version && current_identity == image_identity
                    });
                if !texture_unchanged || !image_unchanged {
                    return None;
                }

                self.resources
                    .texture_image(handle)
                    .map(|gpu_image| PreparedTextureSource {
                        view: gpu_image.default_view.clone(),
                        view_id: gpu_image.id,
                        view_dimension: gpu_image.default_view_dimension,
                        logical_version: logical.version,
                    })
            }
            TextureSource::Attachment(id, view_dimension) => {
                let logical =
                    snapshot_attachment_texture_source(source, self.internal_textures.revision(id));
                if !logical.ready || logical.version != expected_logical_version {
                    return None;
                }
                let view = self.internal_textures.get(id)?.clone();
                let view_id = self.internal_textures.view_id(id)?;
                Some(PreparedTextureSource {
                    view,
                    view_id,
                    view_dimension,
                    logical_version: logical.version,
                })
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Handle — imported from `myth_resources` to ensure CpuBuffer←→SlotMap
// identity agreement.
// ────────────────────────────────────────────────────────────────────────────

pub use myth_resources::GpuBufferHandle;

// ────────────────────────────────────────────────────────────────────────────
// GpuBuffer
// ────────────────────────────────────────────────────────────────────────────

/// A GPU-resident buffer managed by [`ResourceManager`].
///
/// The `id` field is a *physical* resource identity: it changes whenever the
/// underlying `wgpu::Buffer` is recreated (e.g. on resize), allowing
/// `ResourceIdSet` fingerprints to detect the change and trigger `BindGroup`
/// rebuilds.
pub struct GpuBuffer {
    pub id: u64,
    /// CPU-side logical owner for exact cached-handle validation.
    pub(crate) logical_id: u64,
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pub label: String,
    pub last_used_frame: u64,
    pub version: u64,
    pub last_uploaded_version: u64,
}

impl GpuBuffer {
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        logical_id: u64,
        data: &[u8],
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: data,
            usage,
        });

        Self {
            id: generate_gpu_resource_id(),
            logical_id,
            buffer,
            size: data.len() as u64,
            usage,
            label: label.unwrap_or("Buffer").to_string(),
            last_used_frame: 0,
            version: 0,
            last_uploaded_version: 0,
        }
    }

    pub fn with_capacity(
        device: &wgpu::Device,
        logical_id: u64,
        capacity_bytes: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Self {
        let min_size = if usage.contains(wgpu::BufferUsages::UNIFORM) {
            256
        } else {
            16
        };
        let size = capacity_bytes.max(min_size);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: false,
        });

        Self {
            id: generate_gpu_resource_id(),
            logical_id,
            buffer,
            size,
            usage,
            label: label.unwrap_or("Buffer").to_string(),
            last_used_frame: 0,
            version: 0,
            last_uploaded_version: 0,
        }
    }

    /// Write `data` to the GPU, resizing the buffer in-place if necessary.
    ///
    /// Returns `true` when the physical `wgpu::Buffer` was recreated (callers
    /// must rebuild any `BindGroup`s that reference this buffer).
    pub(crate) fn write_to_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> bool {
        let new_size = data.len() as u64;
        if new_size > self.size {
            self.resize(device, new_size);
            queue.write_buffer(&self.buffer, 0, data);
            return true;
        }
        queue.write_buffer(&self.buffer, 0, data);
        false
    }

    /// Destroy the current `wgpu::Buffer` and allocate a larger one.
    ///
    /// The handle in the SlotMap is unchanged; only `id` is regenerated to
    /// propagate the physical-resource change through the fingerprint system.
    pub(crate) fn resize(&mut self, device: &wgpu::Device, new_size: u64) {
        self.buffer.destroy();
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: new_size,
            usage: self.usage,
            mapped_at_creation: false,
        });
        self.size = new_size;
        self.id = generate_gpu_resource_id();
    }
}

impl ResourceManager {
    // ────────────────────────────────────────────────────────────────────────
    // Public ensure_buffer family
    // ────────────────────────────────────────────────────────────────────────

    /// Ensure the GPU buffer for a [`CpuBuffer`] exists and contains the
    /// latest data.
    ///
    /// Uses the `CpuBuffer`'s internal atomic handle cache for an O(1) fast
    /// path on subsequent calls (no hash lookup required).
    pub fn ensure_buffer<T: super::GpuData>(
        &mut self,
        cpu_buffer: &super::CpuBuffer<T>,
    ) -> (GpuBufferHandle, EnsureResult) {
        // ── Fast path: CpuBuffer already knows its slot ────────────
        if let Some(handle) = cpu_buffer.gpu_handle() {
            if let Some(gpu_buf) = self.resources.buffer_mut(handle)
                && gpu_buf.logical_id == cpu_buffer.id()
            {
                let buffer_ref = cpu_buffer.handle();
                let mut was_recreated = false;

                if buffer_ref.version > gpu_buf.last_uploaded_version {
                    let guard = cpu_buffer.read();
                    let data: &[u8] = bytemuck::cast_slice(guard.as_bytes());
                    was_recreated = gpu_buf.write_to_gpu(&self.device, &self.queue, data);
                    gpu_buf.last_uploaded_version = buffer_ref.version;
                }
                gpu_buf.last_used_frame = self.frame_index;
                return (handle, EnsureResult::new(gpu_buf.id, was_recreated));
            }
            // Handle went stale (shouldn't happen under normal operation).
            // Clear the cache and fall through to the slow path.
            cpu_buffer.clear_gpu_handle();
        }

        // ── Slow path: first call or stale handle ──────────────────
        let buffer_ref = cpu_buffer.handle();
        let guard = cpu_buffer.read();
        let data: &[u8] = bytemuck::cast_slice(guard.as_bytes());

        let (handle, result) = self.resources.write_buffer(
            &self.device,
            &self.queue,
            self.frame_index,
            &buffer_ref,
            data,
        );
        cpu_buffer.set_gpu_handle(handle);
        (handle, result)
    }

    /// Ensure a GPU buffer from a [`BufferRef`] and raw byte data.
    ///
    /// Used by generic interfaces (e.g. `MaterialTrait`) that don't hold a
    /// `CpuBuffer`.
    pub fn ensure_buffer_ref(
        &mut self,
        buffer_ref: &BufferRef,
        data: &[u8],
    ) -> (GpuBufferHandle, EnsureResult) {
        self.resources.write_buffer(
            &self.device,
            &self.queue,
            self.frame_index,
            buffer_ref,
            data,
        )
    }

    /// Convenience wrapper returning only the physical resource ID.
    #[inline]
    pub fn ensure_buffer_id<T: super::GpuData>(&mut self, cpu_buffer: &super::CpuBuffer<T>) -> u64 {
        self.ensure_buffer(cpu_buffer).1.resource_id
    }

    // ────────────────────────────────────────────────────────────────────────
    // Attribute / slot-based helpers
    // ────────────────────────────────────────────────────────────────────────

    /// Ensure the GPU buffer for a geometry attribute is created and current.
    pub fn prepare_attribute(
        &mut self,
        attr: &myth_resources::geometry::Attribute,
    ) -> EnsureResult {
        match &attr.data {
            Some(d) => self.ensure_buffer_ref(&attr.buffer, d).1,
            None => self.ensure_buffer_ref(&attr.buffer, &[0u8]).1,
        }
    }

    pub fn prepare_index(
        &mut self,
        indices: &myth_resources::geometry::IndexAttribute,
    ) -> EnsureResult {
        match &indices.data {
            Some(d) => self.ensure_buffer_ref(&indices.buffer, d).1,
            None => self.ensure_buffer_ref(&indices.buffer, &[0u8]).1,
        }
    }

    /// Ensure a uniform slot buffer exists, creating it on first access and
    /// uploading new data on subsequent calls when content differs.
    pub fn prepare_uniform_slot_data(
        &mut self,
        slot_id: u64,
        data: &[u8],
        label: &str,
    ) -> EnsureResult {
        self.resources.write_uniform_slot(
            &self.device,
            &self.queue,
            self.frame_index,
            slot_id,
            data,
            label,
        )
    }

    // ────────────────────────────────────────────────────────────────────────
    // Lookup helpers
    // ────────────────────────────────────────────────────────────────────────

    /// Look up a [`GpuBuffer`] by the CPU-side buffer ID.
    ///
    /// This goes through the `buffer_index` reverse map and is slightly
    /// slower than a direct `gpu_buffers.get(handle)`.
    #[inline]
    pub fn get_gpu_buffer_by_cpu_id(&self, cpu_id: u64) -> Option<&GpuBuffer> {
        self.resources.buffer_by_cpu_id(cpu_id)
    }
}
