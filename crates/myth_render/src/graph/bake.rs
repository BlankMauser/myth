//! Render Command Baking
//!
//! Converts handle-based [`RenderCommand`] / [`ShadowRenderCommand`] lists
//! into physical [`DrawCommand`] lists by resolving every asset handle to an
//! owned clone of its exact `wgpu` resource.
//!
//! This baking step bridges the **cull** phase (which works with lightweight
//! handles) and the **execute** phase (which should be "blind" — touching only
//! pre-resolved GPU state).
//!
//! # Call Site
//!
//! [`finalize_live_draw_plans`] is invoked once per frame inside
//! [`FrameComposer::render()`](super::composer::FrameComposer::render) after
//! the RDG prepare phase and before the execute loop.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::core::ResourceManager;
use crate::core::gpu::BufferUploadContext;
use crate::pipeline::{PipelineCache, RenderPipelineId};

use super::frame::{
    DrawCommand, DrawPlanId, DrawPlanKey, DrawPlanStore, RenderCommand, RenderLists,
    ShadowRenderCommand,
};

/// Finalize only draw plans demanded by passes that survived DPE.
///
/// Reservations are created while the graph is built and marked live after
/// topology compilation. A demanded plan is finalized even when its source
/// list is empty or every command is skipped, keeping "empty" distinct from
/// "not prepared". Dead plans perform no asset lookup, pipeline validation, or
/// WGPU handle cloning.
pub(crate) fn finalize_live_draw_plans(
    draw_plans: &mut DrawPlanStore,
    render_lists: &RenderLists,
    resource_manager: &ResourceManager,
    pipeline_cache: &PipelineCache,
) {
    for plan_index in 0..draw_plans.plan_count() {
        let meta = draw_plans
            .plan_meta(plan_index)
            .expect("draw-plan reservation table changed during finalization");
        if !meta.is_live || meta.is_finalized {
            continue;
        }

        match meta.key {
            DrawPlanKey::Opaque => bake_main_commands(
                meta.id,
                &render_lists.opaque,
                draw_plans,
                resource_manager,
                pipeline_cache,
            ),
            DrawPlanKey::Transparent => bake_main_commands(
                meta.id,
                &render_lists.transparent,
                draw_plans,
                resource_manager,
                pipeline_cache,
            ),
            DrawPlanKey::Prepass {
                needs_normal: _,
                needs_feature_id: _,
                needs_velocity: _,
            } => panic!("live prepass draw plan was not finalized by PrepassPassNode::prepare"),
            DrawPlanKey::Shadow {
                light_id,
                layer_index,
            } => {
                let commands = render_lists
                    .shadow_queues
                    .get(&(light_id, layer_index))
                    .map_or(&[][..], Vec::as_slice);
                bake_shadow_commands(
                    meta.id,
                    commands,
                    draw_plans,
                    resource_manager,
                    pipeline_cache,
                );
            }
        }
    }

    draw_plans.assert_all_live_finalized();
}

// ============================================================================
// Internals
// ============================================================================

/// Bake a slice of main-camera [`RenderCommand`]s into [`DrawCommand`]s.
///
/// The output preserves the input order (already sorted by
/// [`RenderLists::sort`]).
fn bake_main_commands(
    plan: DrawPlanId,
    commands: &[RenderCommand],
    draw_plans: &mut DrawPlanStore,
    rm: &ResourceManager,
    pc: &PipelineCache,
) {
    draw_plans.reserve_commands(commands.len());
    let start = draw_plans.begin_finalize(plan);

    for cmd in commands {
        // Preserve the old bake-time validation boundary even though execute
        // now resolves the already-selected pipeline by stable ID.
        let _ = pc.get_render_pipeline(cmd.pipeline_id);

        let Some(gpu_mat) = rm.get_material(cmd.material_handle) else {
            continue;
        };
        let Some(gpu_geom) = rm.get_geometry(cmd.geometry_handle) else {
            continue;
        };

        assert_eq!(
            gpu_geom.vertex_buffers.len(),
            gpu_geom.vertex_buffer_ids.len(),
            "geometry buffer handles and physical IDs must stay aligned"
        );
        let mut vertex_buffers = SmallVec::with_capacity(gpu_geom.vertex_buffers.len());
        for (buffer, &id) in gpu_geom
            .vertex_buffers
            .iter()
            .zip(&gpu_geom.vertex_buffer_ids)
        {
            vertex_buffers.push(draw_plans.intern_buffer(buffer, id));
        }

        let index_buffer = gpu_geom
            .index_buffer
            .as_ref()
            .map(|(buffer, format, count, id)| {
                (draw_plans.intern_buffer(buffer, *id), *format, *count)
            });
        let material_bind_group =
            draw_plans.intern_bind_group(&gpu_mat.bind_group, gpu_mat.bind_group_id);
        let object_bind_group = draw_plans.intern_bind_group(
            &cmd.object_bind_group.bind_group,
            cmd.object_bind_group.bind_group_id,
        );

        draw_plans.push_command(DrawCommand {
            sort_key: cmd.sort_key.bits(),
            pipeline_id: cmd.pipeline_id,
            vertex_buffers,
            index_buffer,
            bind_group_1: Some(material_bind_group),
            bind_group_2: (object_bind_group, cmd.dynamic_offset),
            stencil_reference: None,
            vertex_range: gpu_geom.draw_range.clone(),
            instance_range: gpu_geom.instance_range.clone(),
        });
    }

    draw_plans.finish_finalize(plan, start);
}

/// Bake prepass draw commands from the opaque list with pipeline remapping.
///
/// For each opaque command that has a corresponding prepass pipeline in
/// `local_cache`, a [`DrawCommand`] is emitted using the prepass
/// pipeline. Feature-ID output selection is encoded in that pipeline variant;
/// this central draw path does not override the render-pass stencil reference.
pub(crate) fn finalize_prepass_draw_plan(
    plan: DrawPlanId,
    opaque_commands: &[RenderCommand],
    draw_plans: &mut DrawPlanStore,
    uploads: &BufferUploadContext<'_>,
    pc: &PipelineCache,
    local_cache: &FxHashMap<(RenderPipelineId, bool, bool, bool), RenderPipelineId>,
    needs_normal: bool,
    needs_feature_id: bool,
    needs_velocity: bool,
) {
    draw_plans.reserve_commands(opaque_commands.len());
    let start = draw_plans.begin_finalize(plan);

    for cmd in opaque_commands {
        let Some(&prepass_pipeline_id) = local_cache.get(&(
            cmd.pipeline_id,
            needs_normal,
            needs_feature_id,
            needs_velocity,
        )) else {
            continue;
        };

        let _ = pc.get_render_pipeline(prepass_pipeline_id);

        let Some(gpu_mat) = uploads.get_material(cmd.material_handle) else {
            continue;
        };
        let Some(gpu_geom) = uploads.get_geometry(cmd.geometry_handle) else {
            continue;
        };

        assert_eq!(
            gpu_geom.vertex_buffers.len(),
            gpu_geom.vertex_buffer_ids.len(),
            "geometry buffer handles and physical IDs must stay aligned"
        );
        let mut vertex_buffers = SmallVec::with_capacity(gpu_geom.vertex_buffers.len());
        for (buffer, &id) in gpu_geom
            .vertex_buffers
            .iter()
            .zip(&gpu_geom.vertex_buffer_ids)
        {
            vertex_buffers.push(draw_plans.intern_buffer(buffer, id));
        }

        let index_buffer = gpu_geom
            .index_buffer
            .as_ref()
            .map(|(buffer, format, count, id)| {
                (draw_plans.intern_buffer(buffer, *id), *format, *count)
            });
        let material_bind_group =
            draw_plans.intern_bind_group(&gpu_mat.bind_group, gpu_mat.bind_group_id);
        let object_bind_group = draw_plans.intern_bind_group(
            &cmd.object_bind_group.bind_group,
            cmd.object_bind_group.bind_group_id,
        );

        draw_plans.push_command(DrawCommand {
            sort_key: cmd.sort_key.bits(),
            pipeline_id: prepass_pipeline_id,
            vertex_buffers,
            index_buffer,
            bind_group_1: Some(material_bind_group),
            bind_group_2: (object_bind_group, cmd.dynamic_offset),
            stencil_reference: None,
            vertex_range: gpu_geom.draw_range.clone(),
            instance_range: gpu_geom.instance_range.clone(),
        });
    }

    draw_plans.finish_finalize(plan, start);
}

/// Bake one demanded shadow-view command queue and sort it by pipeline ID.
fn bake_shadow_commands(
    plan: DrawPlanId,
    commands: &[ShadowRenderCommand],
    draw_plans: &mut DrawPlanStore,
    rm: &ResourceManager,
    pc: &PipelineCache,
) {
    draw_plans.reserve_commands(commands.len());
    let start = draw_plans.begin_finalize(plan);

    for cmd in commands {
        let _ = pc.get_render_pipeline(cmd.pipeline_id);

        let Some(gpu_mat) = rm.get_material(cmd.material_handle) else {
            continue;
        };
        let Some(gpu_geom) = rm.get_geometry(cmd.geometry_handle) else {
            continue;
        };

        assert_eq!(
            gpu_geom.vertex_buffers.len(),
            gpu_geom.vertex_buffer_ids.len(),
            "geometry buffer handles and physical IDs must stay aligned"
        );
        let mut vertex_buffers = SmallVec::with_capacity(gpu_geom.vertex_buffers.len());
        for (buffer, &id) in gpu_geom
            .vertex_buffers
            .iter()
            .zip(&gpu_geom.vertex_buffer_ids)
        {
            vertex_buffers.push(draw_plans.intern_buffer(buffer, id));
        }

        let index_buffer = gpu_geom
            .index_buffer
            .as_ref()
            .map(|(buffer, format, count, id)| {
                (draw_plans.intern_buffer(buffer, *id), *format, *count)
            });
        let material_bind_group =
            draw_plans.intern_bind_group(&gpu_mat.bind_group, gpu_mat.bind_group_id);
        let object_bind_group = draw_plans.intern_bind_group(
            &cmd.object_bind_group.bind_group,
            cmd.object_bind_group.bind_group_id,
        );

        // Shadow sort key: pipeline ID in high bits, material bind group
        // in low bits — minimises both pipeline and bind-group switches.
        let sort_key = u64::from(cmd.pipeline_id.0) << 32 | (gpu_mat.bind_group_id & 0xFFFF_FFFF);

        draw_plans.push_command(DrawCommand {
            sort_key,
            pipeline_id: cmd.pipeline_id,
            vertex_buffers,
            index_buffer,
            bind_group_1: Some(material_bind_group),
            bind_group_2: (object_bind_group, cmd.dynamic_offset),
            stencil_reference: None,
            vertex_range: gpu_geom.draw_range.clone(),
            instance_range: gpu_geom.instance_range.clone(),
        });
    }

    // Sort shadow commands to minimise state switches.
    draw_plans.sort_commands_from(start);
    draw_plans.finish_finalize(plan, start);
}
