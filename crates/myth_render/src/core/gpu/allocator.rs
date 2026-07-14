//! Model Buffer Allocator
//!
//! Manages per-frame dynamic model uniform allocation. Maintains a CPU-side
//! staging vector and a [`GpuBufferHandle`] pointing directly into the
//! `ResourceManager`'s SlotMap arena, bypassing the `CpuBuffer` + `RwLock`
//! intermediary for zero-overhead GPU uploads.

use std::num::NonZero;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::gpu::GpuBuffer;
use myth_resources::buffer::BufferRef;
use myth_resources::uniforms::DynamicModelUniforms;

use super::buffer::GpuBufferHandle;

const DEFAULT_INITIAL_CAPACITY: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelBufferPhase {
    Collecting,
    Frozen,
}

/// Model Buffer allocator.
///
/// Each frame the sequence is:
/// 1. [`Self::reset`] — clear the staging vector
/// 2. [`Self::allocate`] — push uniforms and return byte offsets
/// 3. Caller freezes and uploads through `ResourceManager::flush_model_buffers`
pub struct ModelBufferAllocator {
    /// CPU-side staging data for the current frame.
    host_data: Vec<DynamicModelUniforms>,
    /// Exact payload last submitted to the current physical allocation.
    ///
    /// The two vectors rotate at frame boundaries, avoiding a copy solely for
    /// change detection. Byte equality is required: a hash-only shortcut could
    /// silently retain stale matrices after a collision.
    submitted_data: Vec<DynamicModelUniforms>,
    capacity: usize,
    logical_ref: BufferRef,
    cached_gpu_handle: AtomicU64,
    phase: ModelBufferPhase,
    #[cfg(test)]
    upload_count: usize,
}

impl ModelBufferAllocator {
    #[must_use]
    pub fn new() -> Self {
        Self::with_initial_capacity(DEFAULT_INITIAL_CAPACITY)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn new_for_test(initial_capacity: usize) -> Self {
        Self::with_initial_capacity(initial_capacity)
    }

    fn with_initial_capacity(initial_capacity: usize) -> Self {
        assert!(
            initial_capacity > 0,
            "model buffer capacity must be non-zero"
        );
        let logical_ref = BufferRef::empty(
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("GlobalModelBuffer"),
        );

        Self {
            host_data: Vec::with_capacity(initial_capacity),
            submitted_data: Vec::with_capacity(initial_capacity),
            capacity: initial_capacity,
            logical_ref,
            cached_gpu_handle: AtomicU64::new(0),
            phase: ModelBufferPhase::Collecting,
            #[cfg(test)]
            upload_count: 0,
        }
    }

    /// Reset at the beginning of each frame.
    pub fn reset(&mut self) {
        if self.phase == ModelBufferPhase::Frozen {
            // `host_data` is now the exact payload most recently offered to
            // the persistent allocation. Rotate owners so the next frame can
            // compare without copying the previous byte stream.
            std::mem::swap(&mut self.host_data, &mut self.submitted_data);
        }
        self.host_data.clear();
        self.phase = ModelBufferPhase::Collecting;
    }

    pub fn buffer_handle(&self) -> &BufferRef {
        &self.logical_ref
    }

    pub fn uniform_stride() -> NonZero<u64> {
        std::mem::size_of::<DynamicModelUniforms>()
            .try_into()
            .ok()
            .and_then(NonZero::new)
            .expect("DynamicModelUniforms size should be non-zero")
    }

    /// Allocate a model uniform slot, returning the byte offset.
    pub fn allocate(&mut self, data: DynamicModelUniforms) -> u32 {
        assert_eq!(
            self.phase,
            ModelBufferPhase::Collecting,
            "model uniforms cannot be allocated after the frame buffer is frozen"
        );
        let index = self.host_data.len();

        if index >= self.capacity {
            self.expand_capacity();
        }

        self.host_data.push(data);
        (index * std::mem::size_of::<DynamicModelUniforms>()) as u32
    }

    /// Shrink the capacity to fit the current number of uniforms.
    /// Only intended to be called during scene transitions when we expect a significant drop in the number of active models, to free up GPU memory.
    /// Not called every frame to avoid fragmentation and performance overhead.
    pub fn shrink_to_fit(&mut self) {
        assert_eq!(
            self.phase,
            ModelBufferPhase::Collecting,
            "model buffer capacity cannot change after the frame buffer is frozen"
        );
        self.capacity = self.host_data.len().max(128);
        // Scene-transition trimming deliberately forgets the previous payload;
        // the next live flush uploads conservatively and re-establishes it.
        self.submitted_data.clear();
        self.host_data.shrink_to(self.capacity);
        self.submitted_data.shrink_to(self.capacity);
    }

    pub fn flush_to_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_buffers: &mut slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
        buffer_index: &mut rustc_hash::FxHashMap<u64, GpuBufferHandle>,
        frame_index: u64,
    ) -> bool {
        assert_eq!(
            self.phase,
            ModelBufferPhase::Collecting,
            "model buffer cannot be flushed twice without a frame reset"
        );
        self.phase = ModelBufferPhase::Frozen;
        let mut was_resized = false;
        #[cfg(test)]
        let mut uploaded = false;

        let target_size = (self.capacity * std::mem::size_of::<DynamicModelUniforms>()) as u64;
        let active_data = bytemuck::cast_slice(&self.host_data);
        let payload_unchanged = self.payload_matches_last_submission();

        let handle_bits = self.cached_gpu_handle.load(Ordering::Acquire);
        let cached_handle = GpuBufferHandle::from_bits(handle_bits);
        let live_handle = cached_handle.filter(|handle| gpu_buffers.contains_key(*handle));

        let (handle, was_created) = if let Some(handle) = live_handle {
            (handle, false)
        } else {
            was_resized = true;
            // First allocation, or recovery after the general GPU store
            // evicted this long-idle system buffer.
            let mut gpu_buf = GpuBuffer::with_capacity(
                device,
                self.logical_ref.id(),
                target_size,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("GlobalModelBuffer"),
            );

            // If there is data in the first frame, upload it immediately
            if !active_data.is_empty() {
                queue.write_buffer(&gpu_buf.buffer, 0, active_data);
                #[cfg(test)]
                {
                    uploaded = true;
                }
            }

            gpu_buf.last_used_frame = frame_index;
            let h = gpu_buffers.insert(gpu_buf);
            buffer_index.insert(self.logical_ref.id(), h);
            self.cached_gpu_handle.store(h.to_bits(), Ordering::Release);
            (h, true)
        };

        // Check if the existing buffer needs to be resized (capacity expansion)
        if !was_created {
            let gpu_buf = gpu_buffers
                .get_mut(handle)
                .expect("validated model buffer handle disappeared during flush");
            if gpu_buf.size < target_size {
                gpu_buf.resize(device, target_size);
                was_resized = true;
            }

            if !active_data.is_empty() && (was_resized || !payload_unchanged) {
                queue.write_buffer(&gpu_buf.buffer, 0, active_data);
                #[cfg(test)]
                {
                    uploaded = true;
                }
            }
            gpu_buf.last_used_frame = frame_index;
        }
        #[cfg(test)]
        if uploaded {
            self.upload_count += 1;
        }
        was_resized
    }

    pub(crate) fn assert_frozen(&self) {
        assert_eq!(
            self.phase,
            ModelBufferPhase::Frozen,
            "object BindGroups require the final frozen model buffer"
        );
    }

    pub fn host_data(&self) -> &[DynamicModelUniforms] {
        &self.host_data
    }

    #[inline]
    fn payload_matches_last_submission(&self) -> bool {
        bytemuck::cast_slice::<_, u8>(&self.host_data)
            == bytemuck::cast_slice::<_, u8>(&self.submitted_data)
    }

    #[cfg(test)]
    pub(crate) const fn upload_count(&self) -> usize {
        self.upload_count
    }

    fn expand_capacity(&mut self) {
        let new_cap = (self.capacity * 2).max(128);
        log::info!(
            "Model Buffer expanding capacity: {} -> {}",
            self.capacity,
            new_cap
        );
        self.capacity = new_cap;
    }

    pub fn len(&self) -> usize {
        self.host_data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.host_data.is_empty()
    }
}

impl Default for ModelBufferAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelBufferAllocator, ModelBufferPhase};
    use myth_resources::uniforms::DynamicModelUniforms;

    #[test]
    #[should_panic(expected = "cannot be allocated after the frame buffer is frozen")]
    fn allocation_after_freeze_is_rejected() {
        let mut allocator = ModelBufferAllocator::new_for_test(1);
        allocator.phase = ModelBufferPhase::Frozen;
        allocator.allocate(DynamicModelUniforms::default());
    }

    #[test]
    fn frame_reset_reopens_collection() {
        let mut allocator = ModelBufferAllocator::new_for_test(1);
        allocator.phase = ModelBufferPhase::Frozen;
        allocator.reset();

        assert_eq!(allocator.allocate(DynamicModelUniforms::default()), 0);
    }

    #[test]
    fn frozen_payload_rotates_without_copy_and_matches_exact_bytes() {
        let mut allocator = ModelBufferAllocator::new_for_test(1);
        let baseline = DynamicModelUniforms::default();
        allocator.allocate(baseline);
        let first_allocation = allocator.host_data.as_ptr();
        allocator.phase = ModelBufferPhase::Frozen;

        allocator.reset();
        assert_eq!(allocator.submitted_data.as_ptr(), first_allocation);
        allocator.allocate(baseline);
        assert!(allocator.payload_matches_last_submission());

        allocator.host_data[0].instance_tint.x = -0.0;
        allocator.submitted_data[0].instance_tint.x = 0.0;
        assert!(
            !allocator.payload_matches_last_submission(),
            "bit-distinct float payloads must never alias through numeric equality"
        );
    }

    #[test]
    fn unflushed_collection_does_not_replace_the_last_submitted_payload() {
        let mut allocator = ModelBufferAllocator::new_for_test(1);
        allocator
            .submitted_data
            .push(DynamicModelUniforms::default());
        allocator.host_data.push(DynamicModelUniforms::default());
        allocator.host_data[0].instance_tint.x = 0.5;

        allocator.reset();

        assert_eq!(allocator.submitted_data.len(), 1);
        assert_eq!(allocator.submitted_data[0].instance_tint.x, 0.0);
        assert!(allocator.host_data.is_empty());
    }
}
