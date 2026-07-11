//! GPU-side asset arenas and reverse indices.

use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::SecondaryMap;

use myth_assets::{GeometryHandle, ImageHandle, MaterialHandle, TextureHandle};
use myth_resources::buffer::BufferRef;

use super::{
    AssetTextureGenerationKey, EnsureResult, GpuBuffer, GpuBufferHandle, GpuGeometry, GpuImage,
    GpuMaterial, MipmapRequest, TextureBinding,
};

/// GPU-side asset arenas and reverse indices.
///
/// This is the part of `ResourceManager` that mirrors CPU asset handles to
/// persistent GPU allocations. Keeping it separate lets callers borrow asset
/// storage independently from bind-group and environment state.
pub(crate) struct GpuResourceStore {
    gpu_geometries: SecondaryMap<GeometryHandle, GpuGeometry>,
    gpu_materials: SecondaryMap<MaterialHandle, GpuMaterial>,
    /// One physical image per logical `TextureHandle`.
    ///
    /// Two texture assets may intentionally interpret the same CPU `Image`
    /// with different colour spaces, view dimensions, or mip policies. Keying
    /// this table by `ImageHandle` made those incompatible specifications
    /// overwrite each other every frame.
    gpu_images: SecondaryMap<TextureHandle, GpuImage>,

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
    pending_mipmap_keys: FxHashSet<AssetTextureGenerationKey>,
    pending_mipmap_indices: FxHashMap<TextureHandle, usize>,
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
            gpu_images: SecondaryMap::new(),
            latest_texture_for_image: SecondaryMap::new(),
            texture_bindings: SecondaryMap::new(),
            gpu_buffers,
            buffer_index,
            pending_mipmap_requests: Vec::new(),
            pending_mipmap_keys: FxHashSet::default(),
            pending_mipmap_indices: FxHashMap::default(),
        }
    }

    pub(super) fn queue_mipmap_request(&mut self, request: MipmapRequest) {
        // Only the newest physical generation of one texture asset can matter.
        // This also prevents repeated begin_frame calls without render from
        // retaining an unbounded chain of obsolete WGPU texture owners.
        if let Some(&index) = self.pending_mipmap_indices.get(&request.texture_handle) {
            let stale_key = self.pending_mipmap_requests[index].key();
            if stale_key == request.key() {
                self.pending_mipmap_requests[index] = request;
                return;
            }
            if self.pending_mipmap_keys.contains(&request.key()) {
                // A future physical-sharing layer may produce one generation
                // for multiple TextureHandles. The other queued owner is
                // sufficient; retire this handle's now-obsolete request.
                self.remove_pending_mipmap_at(index);
                return;
            }
            self.pending_mipmap_requests[index] = request;
            self.pending_mipmap_keys.remove(&stale_key);
            self.pending_mipmap_keys
                .insert(self.pending_mipmap_requests[index].key());
            return;
        }

        if !self.pending_mipmap_keys.insert(request.key()) {
            return;
        }

        let index = self.pending_mipmap_requests.len();
        self.pending_mipmap_indices
            .insert(request.texture_handle, index);
        self.pending_mipmap_requests.push(request);
    }

    fn remove_pending_mipmap_at(&mut self, index: usize) -> MipmapRequest {
        let stale = self.pending_mipmap_requests.swap_remove(index);
        self.pending_mipmap_keys.remove(&stale.key());
        self.pending_mipmap_indices.remove(&stale.texture_handle);
        if let Some(moved) = self.pending_mipmap_requests.get(index) {
            self.pending_mipmap_indices
                .insert(moved.texture_handle, index);
        }
        stale
    }

    pub(super) fn discard_mipmap_requests(&mut self, texture_handle: TextureHandle) {
        let Some(index) = self.pending_mipmap_indices.remove(&texture_handle) else {
            return;
        };
        self.remove_pending_mipmap_at(index);
    }

    #[must_use]
    pub(super) fn take_mipmap_requests(&mut self) -> Vec<MipmapRequest> {
        self.pending_mipmap_keys.clear();
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
        self.gpu_images.get(handle)
    }

    #[inline]
    pub(super) fn texture_image_mut(&mut self, handle: TextureHandle) -> Option<&mut GpuImage> {
        self.gpu_images.get_mut(handle)
    }

    #[inline]
    pub(super) fn insert_texture_image(&mut self, handle: TextureHandle, image: GpuImage) {
        let previous_image_handle = self
            .gpu_images
            .get(handle)
            .map(|previous| previous.image_handle);
        let image_handle = image.image_handle;
        self.gpu_images.insert(handle, image);
        self.latest_texture_for_image.insert(image_handle, handle);
        if let Some(previous_image_handle) = previous_image_handle
            && previous_image_handle != image_handle
        {
            self.repair_texture_image_alias(previous_image_handle);
        }
    }

    #[inline]
    pub(super) fn remove_texture_image(&mut self, handle: TextureHandle) {
        self.discard_mipmap_requests(handle);
        self.texture_bindings.remove(handle);
        if let Some(image) = self.gpu_images.remove(handle) {
            self.repair_texture_image_alias(image.image_handle);
        }
    }

    fn repair_texture_image_alias(&mut self, image_handle: ImageHandle) {
        if self
            .latest_texture_for_image
            .get(image_handle)
            .and_then(|texture| self.gpu_images.get(*texture))
            .is_some_and(|image| image.image_handle == image_handle)
        {
            return;
        }

        self.latest_texture_for_image.remove(image_handle);
        if let Some((texture_handle, _)) = self
            .gpu_images
            .iter()
            .filter(|(_, image)| image.image_handle == image_handle)
            .max_by_key(|(_, image)| image.last_used_frame)
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
            .and_then(|texture| self.gpu_images.get(*texture))
            .filter(|image| image.image_handle == handle)
            .or_else(|| {
                self.gpu_images
                    .values()
                    .find(|image| image.image_handle == handle)
            })
    }

    #[inline]
    #[must_use]
    pub(super) fn texture_binding(&self, handle: TextureHandle) -> Option<&TextureBinding> {
        let binding = self.texture_bindings.get(handle)?;
        self.gpu_images
            .get(handle)
            .filter(|image| binding.matches_gpu_image(image))
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
        self.gpu_images.retain(|_, v| v.last_used_frame >= cutoff);
        let gpu_images = &self.gpu_images;
        self.texture_bindings.retain(|handle, binding| {
            gpu_images
                .get(handle)
                .is_some_and(|image| binding.matches_gpu_image(image))
        });
        self.pending_mipmap_requests.retain(|request| {
            gpu_images
                .get(request.texture_handle)
                .is_some_and(|image| image.generation_key() == request.key())
        });
        self.pending_mipmap_keys.clear();
        self.pending_mipmap_keys
            .extend(self.pending_mipmap_requests.iter().map(MipmapRequest::key));
        self.pending_mipmap_indices.clear();
        self.pending_mipmap_indices.extend(
            self.pending_mipmap_requests
                .iter()
                .enumerate()
                .map(|(index, request)| (request.texture_handle, index)),
        );
        self.latest_texture_for_image.retain(|image, texture| {
            self.gpu_images
                .get(*texture)
                .is_some_and(|gpu_image| gpu_image.image_handle == image)
        });

        // Preserve the compatibility lookup when its most-recent alias was
        // evicted but another TextureHandle for the same Image remains live.
        let live_images: FxHashSet<_> = self
            .gpu_images
            .values()
            .map(|image| image.image_handle)
            .collect();
        for image_handle in live_images {
            if self.latest_texture_for_image.contains_key(image_handle) {
                continue;
            }
            if let Some((texture_handle, _)) = self
                .gpu_images
                .iter()
                .filter(|(_, image)| image.image_handle == image_handle)
                .max_by_key(|(_, image)| image.last_used_frame)
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
