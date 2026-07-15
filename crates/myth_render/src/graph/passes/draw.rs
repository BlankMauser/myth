//! Unified Draw Command Submission
//!
//! Provides [`submit_scene_submission`], the single function responsible for
//! translating a backend-neutral scene submission into GPU render-pass calls. All
//! scene-drawing passes invoke this function, ensuring consistent state
//! management and eliminating duplicated draw-loop boilerplate.
//!
//! [`DrawCommand`] stores compact slots into a graph-attempt-owned resource
//! table. Stable physical IDs carried by that table provide identity without
//! depending on WGPU wrapper addresses.

use crate::core::binding::PreparedBindGroupId;
use crate::graph::core::scene_submission::{
    GpuIndexedIndirectBatch, GpuIndexedIndirectStream, ResolvedSceneSubmission,
    SceneSubmissionPlanId, SceneSubmissionResolver,
};
use crate::graph::frame::{DrawBindGroupSlot, DrawBufferSlot, DrawCommand};
use crate::pipeline::{PipelineCache, RenderPipelineId};

/// Submit a pre-baked draw plan through a raw render pass.
///
/// The pipeline cache is append-only for the lifetime of a graph attempt.
/// Its deprecated `clear()` API must not run between stream finalization and
/// execute; the composer's immutable cache borrow enforces that contract on
/// the current serial path. Pipelines are resolved only when their stable ID
/// changes.
///
/// # Bind Group Convention
///
/// * **Group 0** (frame / camera / lights) — set by the caller.
/// * **Group 1** (material) — selected by the draw command.
/// * **Group 2** (object / transform) — CPU-direct commands select a dynamic
///   offset; GPU-indirect batches select one prepared, immutable scene binding.
/// * **Group 3** (screen / transient + clustered lighting) — set by the
///   caller at pass level.
pub(crate) fn submit_scene_submission<'pass>(
    pass: &mut wgpu::RenderPass<'pass>,
    submissions: SceneSubmissionResolver<'pass>,
    plan: SceneSubmissionPlanId,
    pipeline_cache: &'pass PipelineCache,
) {
    match submissions.resolve(plan) {
        ResolvedSceneSubmission::CpuDirect(commands) => {
            submit_cpu_direct(pass, submissions, commands, pipeline_cache);
        }
        ResolvedSceneSubmission::GpuIndexedIndirect(stream) => {
            submit_gpu_indexed_indirect(pass, submissions, stream, pipeline_cache);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bit directly gates one independent WGPU state setter in the blind execute loop"
)]
struct GpuIndirectStateDelta {
    pipeline: bool,
    material_bind_group: bool,
    object_bind_group: bool,
    vertex_buffers: bool,
    index_buffer: bool,
}

#[derive(Default)]
struct GpuIndirectSubmissionState<'a> {
    pipeline: Option<RenderPipelineId>,
    material_bind_group: Option<DrawBindGroupSlot>,
    object_bind_group: Option<PreparedBindGroupId>,
    vertex_buffers: Option<&'a [DrawBufferSlot]>,
    index_buffer: Option<(DrawBufferSlot, wgpu::IndexFormat)>,
}

impl<'a> GpuIndirectSubmissionState<'a> {
    fn transition(&mut self, batch: GpuIndexedIndirectBatch<'a>) -> GpuIndirectStateDelta {
        let next_index = (batch.index_buffer, batch.index_format);
        let delta = GpuIndirectStateDelta {
            pipeline: self.pipeline != Some(batch.pipeline),
            material_bind_group: self.material_bind_group != Some(batch.material_bind_group),
            object_bind_group: self.object_bind_group != Some(batch.object_bind_group),
            vertex_buffers: self.vertex_buffers != Some(batch.vertex_buffers),
            index_buffer: self.index_buffer != Some(next_index),
        };
        self.pipeline = Some(batch.pipeline);
        self.material_bind_group = Some(batch.material_bind_group);
        self.object_bind_group = Some(batch.object_bind_group);
        self.vertex_buffers = Some(batch.vertex_buffers);
        self.index_buffer = Some(next_index);
        delta
    }
}

/// Submit one compact GPU-indirect stream while retaining the currently bound
/// state across adjacent batches. Every indirect draw remains unconditional;
/// only WGPU setters whose exact physical IDs/slices are unchanged are elided.
fn submit_gpu_indexed_indirect<'pass>(
    pass: &mut wgpu::RenderPass<'pass>,
    submissions: SceneSubmissionResolver<'pass>,
    stream: GpuIndexedIndirectStream<'pass>,
    pipeline_cache: &'pass PipelineCache,
) {
    debug_assert!(!stream.is_empty());
    let mut state = GpuIndirectSubmissionState::default();

    for batch in stream.batches() {
        debug_assert_ne!(
            batch.instance_table_physical.physical_id, 0,
            "GPU Scene batch lost its exact persistent table identity"
        );
        let delta = state.transition(batch);

        if delta.pipeline {
            pass.set_pipeline(pipeline_cache.get_render_pipeline(batch.pipeline));
        }
        if delta.material_bind_group {
            let material_bind_group = submissions.bind_group(batch.material_bind_group);
            pass.set_bind_group(1, &**material_bind_group, &[]);
        }
        if delta.object_bind_group {
            pass.set_bind_group(
                2,
                submissions.prepared_bind_group(batch.object_bind_group),
                &[],
            );
        }
        if delta.vertex_buffers {
            for (binding, &slot) in batch.vertex_buffers.iter().enumerate() {
                let buffer = submissions.buffer(slot);
                pass.set_vertex_buffer(binding as u32, buffer.slice(..));
            }
        }
        if delta.index_buffer {
            let index_buffer = submissions.buffer(batch.index_buffer);
            pass.set_index_buffer(index_buffer.slice(..), batch.index_format);
        }

        let indirect_arguments = submissions.buffer(batch.indirect_arguments);
        pass.draw_indexed_indirect(indirect_arguments, batch.indirect_offset);
    }
}

/// Preserve the CPU-direct backend's original zero-copy command slice and
/// state-change elision. Backend selection happens once before entering this
/// loop; no per-command tag dispatch is introduced.
fn submit_cpu_direct<'pass>(
    pass: &mut wgpu::RenderPass<'pass>,
    submissions: SceneSubmissionResolver<'pass>,
    commands: &'pass [DrawCommand],
    pipeline_cache: &'pass PipelineCache,
) {
    if commands.is_empty() {
        return;
    }

    let mut cur_pipeline: Option<RenderPipelineId> = None;
    let mut cur_bg1: Option<DrawBindGroupSlot> = None;
    let mut cur_vertex_b: [Option<DrawBufferSlot>; 8] = [None; 8];
    let mut cur_index_b: Option<(DrawBufferSlot, wgpu::IndexFormat)> = None;
    let mut cur_stencil: Option<u32> = None;

    for cmd in commands {
        if Some(cmd.pipeline_id) != cur_pipeline {
            pass.set_pipeline(pipeline_cache.get_render_pipeline(cmd.pipeline_id));
            cur_pipeline = Some(cmd.pipeline_id);
        }

        if let Some(slot) = cmd.bind_group_1
            && Some(slot) != cur_bg1
        {
            let bind_group = submissions.bind_group(slot);
            pass.set_bind_group(1, &**bind_group, &[]);
            cur_bg1 = Some(slot);
        }

        {
            let (slot, offset) = cmd.bind_group_2;
            let bind_group = submissions.bind_group(slot);
            // Object bind groups are often shared across instances with
            // different dynamic offsets, so this remains unconditional.
            pass.set_bind_group(2, &**bind_group, &[offset]);
        }

        if cmd.stencil_reference != cur_stencil {
            if let Some(value) = cmd.stencil_reference {
                pass.set_stencil_reference(value);
            }
            cur_stencil = cmd.stencil_reference;
        }

        for (binding, &slot) in submissions.vertex_buffers(cmd).iter().enumerate() {
            if cur_vertex_b[binding] != Some(slot) {
                let buffer = submissions.buffer(slot);
                pass.set_vertex_buffer(binding as u32, buffer.slice(..));
                cur_vertex_b[binding] = Some(slot);
            }
        }

        if let Some((slot, format, count)) = cmd.index_buffer {
            if Some((slot, format)) != cur_index_b {
                let buffer = submissions.buffer(slot);
                pass.set_index_buffer(buffer.slice(..), format);
                cur_index_b = Some((slot, format));
            }
            pass.draw_indexed(0..count, 0, cmd.instance_range.clone());
        } else {
            pass.draw(cmd.vertex_range.clone(), cmd.instance_range.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_buffer_slot() -> DrawBufferSlot {
        // SAFETY: this opaque slot is an integer-only value and is never
        // resolved through a DrawPlanStore in the state-transition unit test.
        unsafe { std::mem::zeroed() }
    }

    fn test_bind_group_slot() -> DrawBindGroupSlot {
        // SAFETY: this opaque slot contains one integer with no invalid bit
        // pattern and is compared only as copied identity metadata.
        unsafe { std::mem::zeroed() }
    }

    fn test_prepared_bind_group() -> PreparedBindGroupId {
        // SAFETY: this opaque ID contains integer identity fields only and is
        // never resolved by the test.
        unsafe { std::mem::zeroed() }
    }

    fn batch<'a>(pipeline: u32, vertices: &'a [DrawBufferSlot]) -> GpuIndexedIndirectBatch<'a> {
        GpuIndexedIndirectBatch {
            pipeline: RenderPipelineId(pipeline),
            vertex_buffers: vertices,
            index_buffer: test_buffer_slot(),
            index_format: wgpu::IndexFormat::Uint32,
            material_bind_group: test_bind_group_slot(),
            object_bind_group: test_prepared_bind_group(),
            indirect_arguments: test_buffer_slot(),
            indirect_offset: 0,
            // SAFETY: this integer/newtype identity token is compared only by
            // the transition test and never reaches execute or the registry.
            instance_table_physical: unsafe { std::mem::zeroed() },
        }
    }

    #[test]
    fn gpu_indirect_state_elides_identical_adjacent_setters_only() {
        let vertices = [test_buffer_slot()];
        let changed_vertices = [test_buffer_slot(), test_buffer_slot()];
        let mut state = GpuIndirectSubmissionState::default();

        assert_eq!(
            state.transition(batch(1, &vertices)),
            GpuIndirectStateDelta {
                pipeline: true,
                material_bind_group: true,
                object_bind_group: true,
                vertex_buffers: true,
                index_buffer: true,
            }
        );
        assert_eq!(
            state.transition(batch(1, &vertices)),
            GpuIndirectStateDelta {
                pipeline: false,
                material_bind_group: false,
                object_bind_group: false,
                vertex_buffers: false,
                index_buffer: false,
            }
        );
        assert_eq!(
            state.transition(batch(2, &changed_vertices)),
            GpuIndirectStateDelta {
                pipeline: true,
                material_bind_group: false,
                object_bind_group: false,
                vertex_buffers: true,
                index_buffer: false,
            }
        );
    }
}
