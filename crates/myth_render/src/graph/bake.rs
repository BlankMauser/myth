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
//! [`bake_render_lists`] is invoked once per frame inside
//! [`FrameComposer::render()`](super::composer::FrameComposer::render) after
//! the RDG prepare phase and before the execute loop.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::core::ResourceManager;
use crate::pipeline::{PipelineCache, RenderPipelineId};

use super::frame::{
    DrawCommand, DrawPlanId, DrawPlanStore, PreparedGlobalBindGroup, RenderCommand, RenderLists,
    ShadowRenderCommand,
};

// ============================================================================
// Public API
// ============================================================================

/// Configuration for baking prepass draw commands.
///
/// Passed as `Some(...)` in the HighFidelity render path and `None` when
/// the prepass is disabled (BasicForward).
pub struct PrepassBakeConfig<'a> {
    /// Pipeline remapping table: `(main_pipeline, needs_normal, needs_feature_id, needs_velocity) → prepass_pipeline`.
    pub local_cache: &'a FxHashMap<(RenderPipelineId, bool, bool, bool), RenderPipelineId>,
    /// Whether the prepass outputs view-space normals.
    pub needs_normal: bool,
    /// Whether the prepass outputs a feature-ID stencil mask.
    pub needs_feature_id: bool,
    /// Whether the prepass outputs velocity vectors for TAA.
    pub needs_velocity: bool,
}

/// Bake all render lists into the graph-attempt-owned [`DrawPlanStore`].
///
/// Every handle lookup (`get_geometry`, `get_material`,
/// `get_render_pipeline`) is performed exactly once here. Buffer and bind-group
/// clones pin the selected physical resources until the graph attempt ends;
/// pipelines remain addressed by stable [`RenderPipelineId`] values.
pub(crate) fn bake_render_lists(
    draw_plans: &mut DrawPlanStore,
    global_bind_group: &mut PreparedGlobalBindGroup,
    render_lists: &RenderLists,
    resource_manager: &ResourceManager,
    pipeline_cache: &PipelineCache,
    prepass_config: Option<&PrepassBakeConfig<'_>>,
) {
    draw_plans.clear();
    global_bind_group.clear();

    let opaque_plan = bake_main_commands(
        &render_lists.opaque,
        draw_plans,
        resource_manager,
        pipeline_cache,
    );
    draw_plans.set_opaque_plan(opaque_plan);

    let transparent_plan = bake_main_commands(
        &render_lists.transparent,
        draw_plans,
        resource_manager,
        pipeline_cache,
    );
    draw_plans.set_transparent_plan(transparent_plan);

    let prepass_plan = if let Some(cfg) = prepass_config {
        bake_prepass_commands(
            &render_lists.opaque,
            draw_plans,
            resource_manager,
            pipeline_cache,
            cfg,
        )
    } else {
        let start = draw_plans.begin_plan();
        draw_plans.finish_plan(start)
    };
    draw_plans.set_prepass_plan(prepass_plan);

    bake_shadow_queues(
        &render_lists.shadow_queues,
        draw_plans,
        resource_manager,
        pipeline_cache,
    );

    global_bind_group.stage(
        render_lists
            .gpu_global_bind_group
            .as_ref()
            .expect("Global bind group not built!"),
    );
}

// ============================================================================
// Internals
// ============================================================================

/// Bake a slice of main-camera [`RenderCommand`]s into [`DrawCommand`]s.
///
/// The output preserves the input order (already sorted by
/// [`RenderLists::sort`]).
fn bake_main_commands(
    commands: &[RenderCommand],
    draw_plans: &mut DrawPlanStore,
    rm: &ResourceManager,
    pc: &PipelineCache,
) -> DrawPlanId {
    draw_plans.reserve_commands(commands.len());
    let start = draw_plans.begin_plan();

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

    draw_plans.finish_plan(start)
}

/// Bake prepass draw commands from the opaque list with pipeline remapping.
///
/// For each opaque command that has a corresponding prepass pipeline in
/// `cfg.local_cache`, a [`DrawCommand`] is emitted using the prepass
/// pipeline. Feature-ID output selection is encoded in that pipeline variant;
/// this central draw path does not override the render-pass stencil reference.
fn bake_prepass_commands(
    opaque_commands: &[RenderCommand],
    draw_plans: &mut DrawPlanStore,
    rm: &ResourceManager,
    pc: &PipelineCache,
    cfg: &PrepassBakeConfig<'_>,
) -> DrawPlanId {
    draw_plans.reserve_commands(opaque_commands.len());
    let start = draw_plans.begin_plan();

    for cmd in opaque_commands {
        let Some(&prepass_pipeline_id) = cfg.local_cache.get(&(
            cmd.pipeline_id,
            cfg.needs_normal,
            cfg.needs_feature_id,
            cfg.needs_velocity,
        )) else {
            continue;
        };

        let _ = pc.get_render_pipeline(prepass_pipeline_id);

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

    draw_plans.finish_plan(start)
}

/// Bake all per-view shadow command queues and sort them by pipeline ID.
fn bake_shadow_queues(
    shadow_queues: &FxHashMap<(u64, u32), Vec<ShadowRenderCommand>>,
    draw_plans: &mut DrawPlanStore,
    rm: &ResourceManager,
    pc: &PipelineCache,
) {
    for (&key, commands) in shadow_queues {
        draw_plans.reserve_commands(commands.len());
        let start = draw_plans.begin_plan();

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
            let sort_key =
                u64::from(cmd.pipeline_id.0) << 32 | (gpu_mat.bind_group_id & 0xFFFF_FFFF);

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
        let plan = draw_plans.finish_plan(start);
        draw_plans.insert_shadow_plan(key, plan);
    }
}
