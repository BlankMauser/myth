//! Unified Draw Command Submission
//!
//! Provides [`submit_draw_commands`], the single function responsible for
//! translating a pre-baked draw plan into GPU render-pass calls. All
//! scene-drawing passes invoke this function, ensuring consistent state
//! management and eliminating duplicated draw-loop boilerplate.
//!
//! [`DrawCommand`] stores compact slots into a graph-attempt-owned resource
//! table. Stable physical IDs carried by that table provide identity without
//! depending on WGPU wrapper addresses.

use crate::graph::frame::{
    DrawBindGroupSlot, DrawBufferSlot, DrawCommand, DrawPlanId, DrawPlanResolver,
};
use crate::pipeline::{PipelineCache, RenderPipelineId};

/// Submit a pre-baked draw plan through a raw render pass.
///
/// The pipeline cache is append-only for the lifetime of a graph attempt.
/// Its deprecated `clear()` API must not run between bake finalization and
/// execute; the composer's immutable cache borrow enforces that contract on
/// the current serial path. Pipelines are resolved only when their stable ID
/// changes.
///
/// # Bind Group Convention
///
/// * **Group 0** (frame / camera / lights) — set by the caller.
/// * **Group 1** (material) — selected by the draw command.
/// * **Group 2** (object / transform) — selected by the draw command with its
///   dynamic offset.
/// * **Group 3** (screen / transient + clustered lighting) — set by the
///   caller at pass level.
pub fn submit_draw_commands<'pass>(
    pass: &mut wgpu::RenderPass<'pass>,
    draw_plans: DrawPlanResolver<'pass>,
    plan: DrawPlanId,
    pipeline_cache: &'pass PipelineCache,
) {
    let commands: &'pass [DrawCommand] = draw_plans.commands(plan);
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
            let bind_group = draw_plans.bind_group(slot);
            pass.set_bind_group(1, &**bind_group, &[]);
            cur_bg1 = Some(slot);
        }

        {
            let (slot, offset) = cmd.bind_group_2;
            let bind_group = draw_plans.bind_group(slot);
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

        for (binding, &slot) in cmd.vertex_buffers.iter().enumerate() {
            if cur_vertex_b[binding] != Some(slot) {
                let buffer = draw_plans.buffer(slot);
                pass.set_vertex_buffer(binding as u32, buffer.slice(..));
                cur_vertex_b[binding] = Some(slot);
            }
        }

        if let Some((slot, format, count)) = cmd.index_buffer {
            if Some((slot, format)) != cur_index_b {
                let buffer = draw_plans.buffer(slot);
                pass.set_index_buffer(buffer.slice(..), format);
                cur_index_b = Some((slot, format));
            }
            pass.draw_indexed(0..count, 0, cmd.instance_range.clone());
        } else {
            pass.draw(cmd.vertex_range.clone(), cmd.instance_range.clone());
        }
    }
}
