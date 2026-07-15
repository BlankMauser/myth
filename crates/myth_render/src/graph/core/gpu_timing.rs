//! Asynchronous Render Graph GPU timestamp sampling.
//!
//! This module exists only behind `rdg_bench_support`. Shipping builds neither
//! request timestamp features nor allocate query/readback resources. Samples
//! rotate through a small owned ring and are dropped when the GPU/readback path
//! falls behind; rendering is never allowed to wait for profiler data.

use std::mem::size_of;

const QUERY_CAPACITY: u32 = 4096;
const SLOT_COUNT: usize = 4;
const DEFAULT_SAMPLE_INTERVAL: u64 = 0;
const TIMESTAMP_SIZE: u64 = size_of::<u64>() as u64;
const QUERY_BUFFER_SIZE: u64 = QUERY_CAPACITY as u64 * TIMESTAMP_SIZE;

const REQUIRED_FEATURES: wgpu::Features =
    wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

/// GPU duration attributed to one live RDG pass.
#[derive(Debug, Clone, PartialEq)]
pub struct RdgGpuPassTiming {
    name: &'static str,
    duration_ns: f64,
}

impl RdgGpuPassTiming {
    #[inline]
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    #[inline]
    #[must_use]
    pub const fn duration_ns(&self) -> f64 {
        self.duration_ns
    }

    #[inline]
    #[must_use]
    pub const fn duration_ms(&self) -> f64 {
        self.duration_ns / 1_000_000.0
    }
}

/// Latest asynchronously completed GPU timing sample.
#[derive(Debug, Clone, PartialEq)]
pub struct RdgGpuFrameTimings {
    renderer_frame: u64,
    sample_lag: u64,
    gpu_total_ns: f64,
    passes: Vec<RdgGpuPassTiming>,
    dropped_samples: u64,
    map_errors: u64,
}

impl RdgGpuFrameTimings {
    #[inline]
    #[must_use]
    pub const fn renderer_frame(&self) -> u64 {
        self.renderer_frame
    }

    #[inline]
    #[must_use]
    pub const fn sample_lag(&self) -> u64 {
        self.sample_lag
    }

    #[inline]
    #[must_use]
    pub const fn gpu_total_ns(&self) -> f64 {
        self.gpu_total_ns
    }

    #[inline]
    #[must_use]
    pub const fn gpu_total_ms(&self) -> f64 {
        self.gpu_total_ns / 1_000_000.0
    }

    #[inline]
    #[must_use]
    pub fn passes(&self) -> &[RdgGpuPassTiming] {
        &self.passes
    }

    #[inline]
    #[must_use]
    pub const fn dropped_samples(&self) -> u64 {
        self.dropped_samples
    }

    #[inline]
    #[must_use]
    pub const fn map_errors(&self) -> u64 {
        self.map_errors
    }
}

struct GpuTimingSlot {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    pass_names: Vec<&'static str>,
    renderer_frame: u64,
    query_count: u32,
}

enum GpuTimingCompletion {
    Recycle(GpuTimingSlot),
    Mapped {
        slot: GpuTimingSlot,
        result: Result<(), wgpu::BufferAsyncError>,
    },
}

/// Feature-gated renderer owner of the timestamp/readback ring.
pub(crate) struct RdgGpuTimingRecorder {
    available: Vec<GpuTimingSlot>,
    sender: flume::Sender<GpuTimingCompletion>,
    receiver: flume::Receiver<GpuTimingCompletion>,
    timestamp_period_ns: f64,
    sample_interval: u64,
    latest: Option<RdgGpuFrameTimings>,
    dropped_samples: u64,
    map_errors: u64,
    supported: bool,
}

impl RdgGpuTimingRecorder {
    #[must_use]
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // At most one completion exists for each owned slot. A fixed-capacity
        // channel therefore has room for every legitimate completion and
        // avoids the per-message allocation of an unbounded queue. Producers
        // still use `try_send`: profiler bookkeeping must never block render
        // or a wgpu callback even if an internal invariant is violated.
        let (sender, receiver) = flume::bounded(SLOT_COUNT);
        let supported = device.features().contains(REQUIRED_FEATURES);
        let mut available = Vec::new();
        if supported {
            available.reserve_exact(SLOT_COUNT);
            for _ in 0..SLOT_COUNT {
                let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("RDG GPU Timestamp Query Set"),
                    ty: wgpu::QueryType::Timestamp,
                    count: QUERY_CAPACITY,
                });
                let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("RDG GPU Timestamp Resolve Buffer"),
                    size: QUERY_BUFFER_SIZE,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("RDG GPU Timestamp Readback Buffer"),
                    size: QUERY_BUFFER_SIZE,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                available.push(GpuTimingSlot {
                    query_set,
                    resolve_buffer,
                    readback_buffer,
                    pass_names: Vec::with_capacity(64),
                    renderer_frame: 0,
                    query_count: 0,
                });
            }
        }
        Self {
            available,
            sender,
            receiver,
            timestamp_period_ns: f64::from(queue.get_timestamp_period()),
            sample_interval: DEFAULT_SAMPLE_INTERVAL,
            latest: None,
            dropped_samples: 0,
            map_errors: 0,
            supported,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn supported(&self) -> bool {
        self.supported
    }

    pub(crate) fn set_sample_interval(&mut self, interval: u64) {
        self.sample_interval = interval;
    }

    #[must_use]
    pub(crate) const fn latest(&self) -> Option<&RdgGpuFrameTimings> {
        self.latest.as_ref()
    }

    /// Drive callbacks and consume every completed slot without waiting.
    pub(crate) fn poll(&mut self, device: &wgpu::Device, current_frame: u64) {
        if !self.supported || self.available.len() == SLOT_COUNT {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        while let Ok(completion) = self.receiver.try_recv() {
            match completion {
                GpuTimingCompletion::Recycle(mut slot) => {
                    slot.pass_names.clear();
                    slot.query_count = 0;
                    self.available.push(slot);
                }
                GpuTimingCompletion::Mapped { mut slot, result } => {
                    if result.is_err() || !self.publish_mapped(&slot, current_frame) {
                        self.map_errors = self.map_errors.saturating_add(1);
                    }
                    slot.readback_buffer.unmap();
                    slot.pass_names.clear();
                    slot.query_count = 0;
                    self.available.push(slot);
                }
            }
        }
    }

    fn publish_mapped(&mut self, slot: &GpuTimingSlot, current_frame: u64) -> bool {
        let byte_size = u64::from(slot.query_count) * TIMESTAMP_SIZE;
        let mapped = slot.readback_buffer.slice(0..byte_size).get_mapped_range();
        let timestamps = bytemuck::cast_slice::<u8, u64>(&mapped);
        debug_assert_eq!(timestamps.len(), slot.pass_names.len() + 1);
        if self
            .latest
            .as_ref()
            .is_some_and(|latest| latest.renderer_frame >= slot.renderer_frame)
        {
            // Queue completion is ordered, but callback delivery is not part
            // of the public contract. Never let a delayed callback regress the
            // observable frame serial.
            drop(mapped);
            return true;
        }
        let mut passes = self.latest.take().map_or_else(Vec::new, |mut latest| {
            latest.passes.clear();
            latest.passes
        });
        passes.reserve(slot.pass_names.len());
        for (index, &name) in slot.pass_names.iter().enumerate() {
            let ticks = timestamps[index + 1].wrapping_sub(timestamps[index]);
            passes.push(RdgGpuPassTiming {
                name,
                duration_ns: ticks as f64 * self.timestamp_period_ns,
            });
        }
        let total_ticks = timestamps
            .last()
            .copied()
            .unwrap_or(0)
            .wrapping_sub(timestamps.first().copied().unwrap_or(0));
        drop(mapped);
        self.latest = Some(RdgGpuFrameTimings {
            renderer_frame: slot.renderer_frame,
            sample_lag: current_frame.saturating_sub(slot.renderer_frame),
            gpu_total_ns: total_ticks as f64 * self.timestamp_period_ns,
            passes,
            dropped_samples: self.dropped_samples,
            map_errors: self.map_errors,
        });
        true
    }

    pub(crate) fn begin_sample(
        &mut self,
        renderer_frame: u64,
        pass_count: usize,
    ) -> Option<RdgGpuTimingSample> {
        if !self.supported
            || self.sample_interval == 0
            || !renderer_frame.is_multiple_of(self.sample_interval)
        {
            return None;
        }
        if pass_count == 0 || pass_count >= QUERY_CAPACITY as usize {
            self.dropped_samples = self.dropped_samples.saturating_add(1);
            return None;
        }
        let Some(mut slot) = self.available.pop() else {
            self.dropped_samples = self.dropped_samples.saturating_add(1);
            return None;
        };
        slot.pass_names.clear();
        slot.pass_names.reserve(pass_count);
        slot.renderer_frame = renderer_frame;
        slot.query_count =
            u32::try_from(pass_count + 1).expect("RDG timestamp query count was prevalidated");
        Some(RdgGpuTimingSample {
            slot: Some(slot),
            sender: self.sender.clone(),
            next_query: 0,
        })
    }
}

/// One RAII timestamp reservation. Dropping before successful Queue submission
/// returns its owned ring slot through the same non-blocking completion queue.
pub(crate) struct RdgGpuTimingSample {
    slot: Option<GpuTimingSlot>,
    sender: flume::Sender<GpuTimingCompletion>,
    next_query: u32,
}

impl RdgGpuTimingSample {
    pub(crate) fn write_start(&mut self, encoder: &mut wgpu::CommandEncoder) {
        debug_assert_eq!(self.next_query, 0);
        let slot = self.slot.as_ref().expect("GPU timing sample lost its slot");
        encoder.write_timestamp(&slot.query_set, 0);
        self.next_query = 1;
    }

    pub(crate) fn write_pass_end(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        pass_name: &'static str,
    ) {
        let slot = self.slot.as_mut().expect("GPU timing sample lost its slot");
        slot.pass_names.push(pass_name);
        encoder.write_timestamp(&slot.query_set, self.next_query);
        self.next_query += 1;
    }

    pub(crate) fn encode_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let slot = self.slot.as_ref().expect("GPU timing sample lost its slot");
        debug_assert_eq!(self.next_query, slot.query_count);
        let byte_size = u64::from(slot.query_count) * TIMESTAMP_SIZE;
        encoder.resolve_query_set(
            &slot.query_set,
            0..slot.query_count,
            &slot.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(&slot.resolve_buffer, 0, &slot.readback_buffer, 0, byte_size);
    }

    /// Start asynchronous readback only after the command buffer was accepted
    /// by Queue::submit. The callback owns the slot until it is recycled.
    pub(crate) fn map_after_submit(mut self) {
        let slot = self.slot.take().expect("GPU timing sample lost its slot");
        let byte_size = u64::from(slot.query_count) * TIMESTAMP_SIZE;
        let readback = slot.readback_buffer.clone();
        let sender = self.sender.clone();
        readback
            .slice(0..byte_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.try_send(GpuTimingCompletion::Mapped { slot, result });
            });
    }
}

impl Drop for RdgGpuTimingSample {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            let _ = self.sender.try_send(GpuTimingCompletion::Recycle(slot));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WgpuContext;
    use crate::settings::{RendererInitConfig, RendererSettings};

    #[test]
    fn timestamp_layout_accepts_the_max_live_pass_count() {
        assert_eq!(QUERY_CAPACITY - 1, 4095);
        assert_eq!(QUERY_BUFFER_SIZE, 32 * 1024);
    }

    #[test]
    fn timestamp_delta_handles_counter_wrap() {
        let start = u64::MAX - 4;
        let end = 3u64;
        assert_eq!(end.wrapping_sub(start), 8);
    }

    #[test]
    fn timestamp_round_trip_or_cleanly_reports_unavailable() {
        let Ok(context) = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            4,
            4,
            None,
        )) else {
            return;
        };
        let mut recorder = RdgGpuTimingRecorder::new(&context.device, &context.queue);
        assert!(
            recorder.begin_sample(8, 1).is_none(),
            "GPU timestamps must remain opt-in even in benchmark-support builds"
        );
        if !recorder.supported() {
            assert!(recorder.latest().is_none());
            return;
        }
        recorder.set_sample_interval(1);
        let mut sample = recorder
            .begin_sample(7, 1)
            .expect("supported timestamp recorder refused an idle slot");
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RDG GPU Timestamp Test Encoder"),
            });
        sample.write_start(&mut encoder);
        sample.write_pass_end(&mut encoder, "Empty_Test_Pass");
        sample.encode_resolve(&mut encoder);
        context.queue.submit([encoder.finish()]);
        sample.map_after_submit();
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("timestamp test device poll failed");
        recorder.poll(&context.device, 8);

        let latest = recorder
            .latest()
            .expect("submitted timestamp sample did not publish");
        assert_eq!(latest.renderer_frame(), 7);
        assert_eq!(latest.sample_lag(), 1);
        assert_eq!(latest.passes().len(), 1);
        assert_eq!(latest.passes()[0].name(), "Empty_Test_Pass");
    }

    #[test]
    fn timestamp_ring_drops_without_blocking_and_recycles_raii_slots() {
        let Ok(context) = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            4,
            4,
            None,
        )) else {
            return;
        };
        let mut recorder = RdgGpuTimingRecorder::new(&context.device, &context.queue);
        if !recorder.supported() {
            return;
        }
        recorder.set_sample_interval(1);
        let reservations: Vec<_> = (0..SLOT_COUNT)
            .map(|frame| {
                recorder
                    .begin_sample(frame as u64, 1)
                    .expect("idle timestamp slot missing")
            })
            .collect();
        assert!(recorder.begin_sample(SLOT_COUNT as u64, 1).is_none());
        assert_eq!(recorder.dropped_samples, 1);

        drop(reservations);
        recorder.poll(&context.device, SLOT_COUNT as u64);
        assert_eq!(recorder.available.len(), SLOT_COUNT);
    }
}
