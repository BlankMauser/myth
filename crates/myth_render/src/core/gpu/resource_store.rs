//! GPU-side asset arenas and reverse indices.

use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::SecondaryMap;

use myth_assets::{GeometryHandle, ImageHandle, MaterialHandle, TextureHandle};
use myth_resources::buffer::BufferRef;
use myth_resources::image::Image;

use super::{
    EnsureResult, GpuBuffer, GpuBufferHandle, GpuGeometry, GpuImage, GpuImageSpec, GpuMaterial,
    MipmapRequest, PhysicalImageGenerationKey, PhysicalImageId, TextureBinding, TextureImageLease,
};

/// GPU-side asset arenas and reverse indices.
///
/// This is the part of `ResourceManager` that mirrors CPU asset handles to
/// persistent GPU allocations. Keeping it separate lets callers borrow asset
/// storage independently from bind-group and environment state.
pub(crate) struct GpuResourceStore {
    gpu_geometries: SecondaryMap<GeometryHandle, GpuGeometry>,
    gpu_materials: SecondaryMap<MaterialHandle, GpuMaterial>,
    /// Destructor-bearing physical image arena. The first migration stage
    /// deliberately keeps one lease per entry; exact-spec sharing is enabled
    /// only after mipmap ownership has been fully detached from TextureHandle.
    physical_images: slotmap::SlotMap<PhysicalImageId, GpuImage>,
    /// Logical TextureHandle residency pointing into the physical arena.
    texture_image_leases: SecondaryMap<TextureHandle, TextureImageLease>,

    /// Compatibility lookup for callers that still only carry an ImageHandle.
    /// Exact render paths must use `texture_image(TextureHandle)` instead.
    latest_texture_for_image: SecondaryMap<ImageHandle, TextureHandle>,

    /// Mapping from `TextureHandle` to (`ImageId`, `SamplerId`)
    texture_bindings: SecondaryMap<TextureHandle, TextureBinding>,

    /// All GPU buffers stored in a contiguous arena for O(1) handle-based access.
    gpu_buffers: slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
    /// Reverse index: CPU-side buffer ID -> SlotMap handle.
    buffer_index: FxHashMap<u64, GpuBufferHandle>,

    pending_mipmap_requests: Vec<MipmapRequest>,
    /// At most the newest content generation for one physical allocation is
    /// retained before ownership moves into GraphStorage.
    pending_mipmap_indices: FxHashMap<PhysicalImageId, usize>,
}

impl GpuResourceStore {
    #[must_use]
    pub(super) fn new(
        gpu_buffers: slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
        buffer_index: FxHashMap<u64, GpuBufferHandle>,
    ) -> Self {
        Self {
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SecondaryMap::new(),
            physical_images: slotmap::SlotMap::with_key(),
            texture_image_leases: SecondaryMap::new(),
            latest_texture_for_image: SecondaryMap::new(),
            texture_bindings: SecondaryMap::new(),
            gpu_buffers,
            buffer_index,
            pending_mipmap_requests: Vec::new(),
            pending_mipmap_indices: FxHashMap::default(),
        }
    }

    pub(super) fn queue_mipmap_request(&mut self, request: MipmapRequest) {
        let is_current_generation = self
            .physical_images
            .get(request.physical_id)
            .is_some_and(|image| image.generation_key(request.physical_id) == request.key());
        if !is_current_generation {
            // Reject requests detached by content replacement, TTL/removal,
            // or a SlotMap generation reuse even when the pending table has
            // already been drained into GraphStorage.
            return;
        }

        // Only the newest content generation of one physical allocation can
        // matter. TextureHandle remains diagnostic/legacy claim metadata, not
        // the pending work's correctness identity.
        // This also prevents repeated begin_frame calls without render from
        // retaining an unbounded chain of obsolete WGPU texture owners.
        if let Some(&index) = self.pending_mipmap_indices.get(&request.physical_id) {
            let queued_generation = self.pending_mipmap_requests[index].generation_id;
            if queued_generation > request.generation_id {
                // A stale request may arrive after a newer coordinator result;
                // it must never replace the newest physical generation.
                return;
            }
            self.pending_mipmap_requests[index] = request;
            return;
        }

        let index = self.pending_mipmap_requests.len();
        self.pending_mipmap_indices
            .insert(request.physical_id, index);
        self.pending_mipmap_requests.push(request);
    }

    fn remove_pending_mipmap_at(&mut self, index: usize) -> MipmapRequest {
        let stale = self.pending_mipmap_requests.swap_remove(index);
        self.pending_mipmap_indices.remove(&stale.physical_id);
        if let Some(moved) = self.pending_mipmap_requests.get(index) {
            self.pending_mipmap_indices.insert(moved.physical_id, index);
        }
        stale
    }

    pub(super) fn discard_mipmap_requests(&mut self, texture_handle: TextureHandle) {
        let Some(physical_id) = self
            .texture_image_leases
            .get(texture_handle)
            .map(|lease| lease.physical_id)
        else {
            return;
        };
        self.discard_mipmap_request_for_physical(physical_id);
    }

    fn discard_mipmap_request_for_physical(&mut self, physical_id: PhysicalImageId) {
        if let Some(index) = self.pending_mipmap_indices.get(&physical_id).copied() {
            self.remove_pending_mipmap_at(index);
        }
    }

    #[must_use]
    pub(super) fn take_mipmap_requests(&mut self) -> Vec<MipmapRequest> {
        self.pending_mipmap_indices.clear();
        std::mem::take(&mut self.pending_mipmap_requests)
    }

    pub(super) fn buffer_storage_mut(
        &mut self,
    ) -> (
        &mut slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
        &mut FxHashMap<u64, GpuBufferHandle>,
    ) {
        (&mut self.gpu_buffers, &mut self.buffer_index)
    }

    #[inline]
    #[must_use]
    pub(super) fn geometry(&self, handle: GeometryHandle) -> Option<&GpuGeometry> {
        self.gpu_geometries.get(handle)
    }

    #[inline]
    pub(super) fn geometry_mut(&mut self, handle: GeometryHandle) -> Option<&mut GpuGeometry> {
        self.gpu_geometries.get_mut(handle)
    }

    #[inline]
    pub(super) fn insert_geometry(&mut self, handle: GeometryHandle, geometry: GpuGeometry) {
        self.gpu_geometries.insert(handle, geometry);
    }

    #[inline]
    #[must_use]
    pub(super) fn material(&self, handle: MaterialHandle) -> Option<&GpuMaterial> {
        self.gpu_materials.get(handle)
    }

    #[inline]
    pub(super) fn material_mut(&mut self, handle: MaterialHandle) -> Option<&mut GpuMaterial> {
        self.gpu_materials.get_mut(handle)
    }

    #[inline]
    #[must_use]
    pub(super) fn contains_material(&self, handle: MaterialHandle) -> bool {
        self.gpu_materials.contains_key(handle)
    }

    #[inline]
    pub(super) fn insert_material(&mut self, handle: MaterialHandle, material: GpuMaterial) {
        self.gpu_materials.insert(handle, material);
    }

    #[inline]
    #[must_use]
    pub(super) fn texture_image(&self, handle: TextureHandle) -> Option<&GpuImage> {
        let lease = self.texture_image_leases.get(handle)?;
        self.physical_images.get(lease.physical_id)
    }

    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(super) fn texture_image_lease(&self, handle: TextureHandle) -> Option<TextureImageLease> {
        self.texture_image_leases.get(handle).copied()
    }

    #[cfg(test)]
    #[must_use]
    pub(super) fn contains_physical_image(&self, id: PhysicalImageId) -> bool {
        self.physical_images.contains_key(id)
    }

    /// Create or update one TextureHandle's physical residency through the
    /// arena/lease split. This first stage intentionally does not share entries
    /// across handles; the API is the single acquisition point that the exact
    /// spec index will extend in the next migration batch.
    pub(super) fn materialize_texture_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_index: u64,
        handle: TextureHandle,
        image: &Image,
        spec: GpuImageSpec,
        texture_version: u32,
        texture_identity: u64,
        image_version: u32,
        image_identity: u64,
    ) -> PhysicalImageId {
        let previous_image_handle = self.texture_image(handle).map(|image| image.image_handle);
        let reusable = self
            .texture_image(handle)
            .is_some_and(|gpu_image| gpu_image.matches_spec(spec));

        let physical_id = if reusable {
            self.texture_image_leases
                .get(handle)
                .expect("texture image exists without a lease")
                .physical_id
        } else {
            self.remove_texture_image(handle);
            let mut gpu_image =
                GpuImage::new(device, queue, image, spec, image_version, image_identity);
            gpu_image.last_used_frame = frame_index;
            let physical_id = self.physical_images.insert(gpu_image);
            self.texture_image_leases.insert(
                handle,
                TextureImageLease {
                    physical_id,
                    image_handle: spec.image_handle,
                    texture_version,
                    texture_identity,
                    image_version,
                    image_identity,
                    last_used_frame: frame_index,
                },
            );
            physical_id
        };

        if reusable {
            let generation_before = self
                .physical_images
                .get(physical_id)
                .expect("texture lease points to a missing physical image")
                .generation_id;
            let gpu_image = self
                .physical_images
                .get_mut(physical_id)
                .expect("texture lease points to a missing physical image");
            gpu_image.update_content(queue, image, image_version, image_identity);
            gpu_image.last_used_frame = frame_index;
            if gpu_image.generation_id != generation_before {
                self.discard_mipmap_request_for_physical(physical_id);
            }
            let lease = self
                .texture_image_leases
                .get_mut(handle)
                .expect("texture image exists without a lease");
            lease.texture_version = texture_version;
            lease.texture_identity = texture_identity;
            lease.image_version = image_version;
            lease.image_identity = image_identity;
            lease.last_used_frame = frame_index;
        }

        self.latest_texture_for_image
            .insert(spec.image_handle, handle);
        if let Some(previous_image_handle) = previous_image_handle
            && previous_image_handle != spec.image_handle
        {
            self.repair_texture_image_alias(previous_image_handle);
        }
        physical_id
    }

    pub(super) fn touch_texture_image(&mut self, handle: TextureHandle, frame_index: u64) -> bool {
        let Some(lease) = self.texture_image_leases.get_mut(handle) else {
            return false;
        };
        lease.last_used_frame = frame_index;
        if let Some(image) = self.physical_images.get_mut(lease.physical_id) {
            image.last_used_frame = frame_index;
            true
        } else {
            false
        }
    }

    #[must_use]
    pub(super) fn texture_mipmap_request(&self, handle: TextureHandle) -> Option<MipmapRequest> {
        let lease = self.texture_image_leases.get(handle)?;
        self.physical_images
            .get(lease.physical_id)?
            .mipmap_request(lease.physical_id, handle)
    }

    #[must_use]
    pub(super) fn texture_generation_key(
        &self,
        handle: TextureHandle,
    ) -> Option<PhysicalImageGenerationKey> {
        let lease = self.texture_image_leases.get(handle)?;
        Some(
            self.physical_images
                .get(lease.physical_id)?
                .generation_key(lease.physical_id),
        )
    }

    #[inline]
    pub(super) fn remove_texture_image(&mut self, handle: TextureHandle) {
        self.discard_mipmap_requests(handle);
        self.texture_bindings.remove(handle);
        if let Some(lease) = self.texture_image_leases.remove(handle) {
            if let Some(image) = self.physical_images.remove(lease.physical_id) {
                debug_assert_eq!(image.image_handle, lease.image_handle);
            }
            self.repair_texture_image_alias(lease.image_handle);
        }
    }

    fn repair_texture_image_alias(&mut self, image_handle: ImageHandle) {
        if self
            .latest_texture_for_image
            .get(image_handle)
            .and_then(|texture| self.texture_image(*texture))
            .is_some_and(|image| image.image_handle == image_handle)
        {
            return;
        }

        self.latest_texture_for_image.remove(image_handle);
        if let Some((texture_handle, _)) = self
            .texture_image_leases
            .iter()
            .filter(|(texture, _)| {
                self.texture_image(*texture)
                    .is_some_and(|image| image.image_handle == image_handle)
            })
            .max_by_key(|(_, lease)| lease.last_used_frame)
        {
            self.latest_texture_for_image
                .insert(image_handle, texture_handle);
        }
    }

    /// Compatibility lookup for legacy call sites that only know the CPU image.
    #[inline]
    #[must_use]
    pub(super) fn latest_image(&self, handle: ImageHandle) -> Option<&GpuImage> {
        self.latest_texture_for_image
            .get(handle)
            .and_then(|texture| self.texture_image(*texture))
            .filter(|image| image.image_handle == handle)
            .or_else(|| {
                self.texture_image_leases.iter().find_map(|(texture, _)| {
                    self.texture_image(texture)
                        .filter(|image| image.image_handle == handle)
                })
            })
    }

    #[inline]
    #[must_use]
    pub(super) fn texture_binding(&self, handle: TextureHandle) -> Option<&TextureBinding> {
        let binding = self.texture_bindings.get(handle)?;
        let lease = self.texture_image_leases.get(handle).copied()?;
        self.physical_images
            .get(lease.physical_id)
            .filter(|image| binding.matches_residency(lease, image))
            .map(|_| binding)
    }

    #[inline]
    pub(super) fn insert_texture_binding(
        &mut self,
        handle: TextureHandle,
        binding: TextureBinding,
    ) {
        self.latest_texture_for_image
            .insert(binding.image_handle, handle);
        self.texture_bindings.insert(handle, binding);
    }

    #[inline]
    pub(super) fn touch_texture_image_alias(
        &mut self,
        image_handle: ImageHandle,
        texture_handle: TextureHandle,
    ) {
        self.latest_texture_for_image
            .insert(image_handle, texture_handle);
    }

    #[cfg(test)]
    #[must_use]
    pub(super) fn latest_texture_alias(&self, image_handle: ImageHandle) -> Option<TextureHandle> {
        self.latest_texture_for_image.get(image_handle).copied()
    }

    pub(super) fn prune(&mut self, cutoff: u64) {
        self.gpu_geometries
            .retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_materials
            .retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_buffers.retain(|_, v| v.last_used_frame >= cutoff);
        self.buffer_index
            .retain(|_, h| self.gpu_buffers.contains_key(*h));
        let stale_textures: Vec<_> = self
            .texture_image_leases
            .iter()
            .filter_map(|(handle, lease)| (lease.last_used_frame < cutoff).then_some(handle))
            .collect();
        for handle in stale_textures {
            self.remove_texture_image(handle);
        }
        let texture_image_leases = &self.texture_image_leases;
        let physical_images = &self.physical_images;
        self.texture_bindings.retain(|handle, binding| {
            texture_image_leases
                .get(handle)
                .and_then(|lease| {
                    physical_images
                        .get(lease.physical_id)
                        .map(|image| (*lease, image))
                })
                .is_some_and(|(lease, image)| binding.matches_residency(lease, image))
        });
        self.pending_mipmap_requests.retain(|request| {
            physical_images
                .get(request.physical_id)
                .is_some_and(|image| image.generation_key(request.physical_id) == request.key())
        });
        self.pending_mipmap_indices.clear();
        self.pending_mipmap_indices.extend(
            self.pending_mipmap_requests
                .iter()
                .enumerate()
                .map(|(index, request)| (request.physical_id, index)),
        );
        self.latest_texture_for_image.retain(|image, texture| {
            texture_image_leases
                .get(*texture)
                .and_then(|lease| physical_images.get(lease.physical_id))
                .is_some_and(|gpu_image| gpu_image.image_handle == image)
        });

        // Preserve the compatibility lookup when its most-recent alias was
        // evicted but another TextureHandle for the same Image remains live.
        let live_images: FxHashSet<_> = self
            .texture_image_leases
            .keys()
            .filter_map(|texture| self.texture_image(texture).map(|image| image.image_handle))
            .collect();
        for image_handle in live_images {
            if self.latest_texture_for_image.contains_key(image_handle) {
                continue;
            }
            if let Some((texture_handle, _)) = self
                .texture_image_leases
                .iter()
                .filter(|(texture, _)| {
                    self.texture_image(*texture)
                        .is_some_and(|image| image.image_handle == image_handle)
                })
                .max_by_key(|(_, lease)| lease.last_used_frame)
            {
                self.latest_texture_for_image
                    .insert(image_handle, texture_handle);
            }
        }
    }

    /// Upload `data` for the buffer identified by `buffer_ref`, creating or
    /// resizing the GPU-side buffer as needed.
    pub(super) fn write_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_index: u64,
        buffer_ref: &BufferRef,
        data: &[u8],
    ) -> (GpuBufferHandle, EnsureResult) {
        let cpu_id = buffer_ref.id();

        if let Some(&handle) = self.buffer_index.get(&cpu_id) {
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
                let mut was_recreated = false;

                if buffer_ref.version > gpu_buf.last_uploaded_version {
                    let old_id = gpu_buf.id;
                    was_recreated = gpu_buf.write_to_gpu(device, queue, data);
                    if !was_recreated && gpu_buf.id != old_id {
                        was_recreated = true;
                    }
                    gpu_buf.last_uploaded_version = buffer_ref.version;
                }
                gpu_buf.last_used_frame = frame_index;
                return (handle, EnsureResult::new(gpu_buf.id, was_recreated));
            }
            self.buffer_index.remove(&cpu_id);
        }

        let mut buf = GpuBuffer::new(device, data, buffer_ref.usage, buffer_ref.label());
        buf.last_uploaded_version = buffer_ref.version;
        buf.last_used_frame = frame_index;
        let phys_id = buf.id;
        let handle = self.gpu_buffers.insert(buf);
        self.buffer_index.insert(cpu_id, handle);
        (handle, EnsureResult::created(phys_id))
    }

    pub(super) fn write_uniform_slot(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_index: u64,
        slot_id: u64,
        data: &[u8],
        label: &str,
    ) -> EnsureResult {
        if let Some(&handle) = self.buffer_index.get(&slot_id) {
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
                let was_recreated = gpu_buf.write_to_gpu(device, queue, data);
                gpu_buf.last_used_frame = frame_index;
                return EnsureResult::new(gpu_buf.id, was_recreated);
            }
            self.buffer_index.remove(&slot_id);
        }

        let mut buf = GpuBuffer::new(
            device,
            data,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some(label),
        );
        buf.last_used_frame = frame_index;
        let phys_id = buf.id;
        let handle = self.gpu_buffers.insert(buf);
        self.buffer_index.insert(slot_id, handle);
        EnsureResult::created(phys_id)
    }

    pub(super) fn touch_buffer_by_cpu_id(&mut self, cpu_id: u64, frame_index: u64) -> bool {
        let Some(&handle) = self.buffer_index.get(&cpu_id) else {
            return false;
        };

        if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
            gpu_buf.last_used_frame = frame_index;
            return true;
        }

        self.buffer_index.remove(&cpu_id);
        false
    }

    #[inline]
    pub(super) fn buffer_mut(&mut self, handle: GpuBufferHandle) -> Option<&mut GpuBuffer> {
        self.gpu_buffers.get_mut(handle)
    }

    #[inline]
    pub(super) fn buffer(&self, handle: GpuBufferHandle) -> Option<&GpuBuffer> {
        self.gpu_buffers.get(handle)
    }

    #[inline]
    pub(super) fn buffer_by_cpu_id(&self, cpu_id: u64) -> Option<&GpuBuffer> {
        self.buffer_index
            .get(&cpu_id)
            .and_then(|&handle| self.gpu_buffers.get(handle))
    }
}
