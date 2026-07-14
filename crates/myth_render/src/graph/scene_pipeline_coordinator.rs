//! Serial post-DPE scene-pipeline coordination and direct-stream finalization.
//!
//! Live pass nodes gather only immutable, Copy recipes. This module is the
//! single physical owner that restores canonical recipe order, deduplicates
//! exact pipeline requests, materializes cold WGPU pipelines, and finally
//! writes contiguous [`DrawCommand`] ranges for the CPU-direct backend.

use myth_resources::{
    AntiAliasingMode,
    material::{AlphaMode, Side},
};
use myth_scene::camera::RenderCamera;

use crate::{
    HDR_TEXTURE_FORMAT,
    core::{
        WgpuContext,
        gpu::{GpuGlobalState, SystemTextures},
    },
    pipeline::{
        BlendStateKey, ColorTargetKey, DepthStencilKey, FastPipelineKey, FastShadowPipelineKey,
        GraphicsPipelineKey, PipelineCache, RenderPipelineId, ShaderCompilationOptions,
        ShaderManager, ShaderSource, SimpleGeometryPipelineKey, pipeline_key::PipelineFlags,
    },
    settings::RenderPath,
};

use super::{
    asset_frame::{
        GeometryFramePhysical, GeometryFrameRecord, GeometryFrameResolver, ObjectInterfaceRecord,
        ObjectInterfaceResolver,
    },
    extracted::{ExtractedRenderItem, ExtractedScene, SceneFeatures},
    frame::{
        DrawCommand, DrawPlanId, DrawPlanKey, DrawPlanStore, PreparedDrawSource, PreparedDrawStore,
        RenderIntent, RenderKey, RenderLists, ShadowRenderIntent,
    },
    material_plan::{MaterialPlanResolver, PreparedMaterialFrame},
    passes::{
        prepass::{FEATURE_ID_FORMAT, NORMAL_FORMAT, VELOCITY_FORMAT},
        shadow::SHADOW_BINDING_WGSL,
    },
    scene_pipeline_prepare::{
        AcceptedPreparedSource, ScenePipelinePrepareStore, SceneRecipeOrdinal,
    },
};

/// Gather main-view and shadow recipes for every live CPU-direct plan.
///
/// Prepass recipes are gathered by the live Prepass node because its output
/// mask and L1 cache are Feature-owned. This function performs no WGPU object
/// creation and does not mutate either pipeline cache level.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_live_pipeline_recipes(
    store: &mut ScenePipelinePrepareStore,
    draw_plans: &DrawPlanStore,
    prepared_draws: &PreparedDrawStore,
    geometry_frames: GeometryFrameResolver<'_>,
    material_frames: MaterialPlanResolver<'_>,
    object_interfaces: ObjectInterfaceResolver<'_>,
    render_lists: &RenderLists,
    extracted_scene: &ExtractedScene,
    gpu_world: Option<&GpuGlobalState>,
    wgpu_ctx: &WgpuContext,
    shader_manager: &ShaderManager,
    camera: &RenderCamera,
) {
    // Attempt invariants are derived at most once per relevant pipeline
    // family. They are deliberately recomputed at the later coordination
    // barrier so mutations between Gather and Coordinate still fail closed.
    let mut scene_defines_hash = None;
    let mut depth_shader_identity = None;
    for plan_index in 0..draw_plans.plan_count() {
        let meta = draw_plans
            .plan_meta(plan_index)
            .expect("draw-plan reservation table changed during recipe gathering");
        if !meta.is_live || meta.is_finalized {
            continue;
        }

        match meta.key {
            DrawPlanKey::Opaque | DrawPlanKey::Transparent => {
                let Some(gpu_world) = gpu_world else {
                    continue;
                };
                let transparent = matches!(meta.key, DrawPlanKey::Transparent);
                let intents = if transparent {
                    render_lists.transparent.as_slice()
                } else {
                    render_lists.opaque.as_slice()
                };
                gather_main_plan(
                    store,
                    meta.id,
                    intents,
                    transparent,
                    prepared_draws,
                    geometry_frames,
                    material_frames,
                    object_interfaces,
                    render_lists,
                    extracted_scene,
                    gpu_world,
                    wgpu_ctx,
                    shader_manager,
                    camera,
                    *scene_defines_hash
                        .get_or_insert_with(|| extracted_scene.scene_defines.compute_hash()),
                );
            }
            DrawPlanKey::Prepass { .. } => {}
            DrawPlanKey::Shadow {
                light_id,
                layer_index,
            } => {
                let layout = store.registered_shadow_layout().expect(
                    "a live Shadow plan reached pipeline gathering before its Feature layout was registered",
                );
                let intents = render_lists
                    .shadow_queues
                    .get(&(light_id, layer_index))
                    .map_or(&[][..], Vec::as_slice);
                gather_shadow_plan(
                    store,
                    meta.id,
                    intents,
                    layout,
                    prepared_draws,
                    geometry_frames,
                    material_frames,
                    object_interfaces,
                    render_lists,
                    extracted_scene,
                    wgpu_ctx,
                    *depth_shader_identity.get_or_insert_with(|| {
                        shader_manager
                            .source_identity_hash(ShaderSource::File("entry/utility/depth_prepass"))
                    }),
                );
            }
        }
    }
}

/// Restore canonical order, coordinate every unique exact pipeline request,
/// and publish accepted main/shadow L1 entries.
///
/// Prepass L1 publications stay pending until the second live-node barrier so
/// the owning Feature can accept them through `finalize_prepare`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn coordinate_scene_pipelines(
    store: &mut ScenePipelinePrepareStore,
    plan_count: usize,
    prepared_draws: &PreparedDrawStore,
    geometry_frames: GeometryFrameResolver<'_>,
    material_frames: MaterialPlanResolver<'_>,
    object_interfaces: ObjectInterfaceResolver<'_>,
    render_lists: &RenderLists,
    extracted_scene: &ExtractedScene,
    gpu_world: Option<&GpuGlobalState>,
    wgpu_ctx: &WgpuContext,
    system_textures: &SystemTextures,
    pipeline_cache: &mut PipelineCache,
    shader_manager: &mut ShaderManager,
    camera: &RenderCamera,
) {
    store.plan(plan_count);
    coordinate_main_requests(
        store,
        prepared_draws,
        geometry_frames,
        material_frames,
        object_interfaces,
        render_lists,
        extracted_scene,
        gpu_world,
        wgpu_ctx,
        system_textures,
        pipeline_cache,
        shader_manager,
        camera,
    );
    coordinate_prepass_requests(
        store,
        prepared_draws,
        geometry_frames,
        material_frames,
        object_interfaces,
        render_lists,
        extracted_scene,
        gpu_world,
        wgpu_ctx,
        pipeline_cache,
        shader_manager,
    );
    coordinate_shadow_requests(
        store,
        prepared_draws,
        geometry_frames,
        material_frames,
        object_interfaces,
        render_lists,
        extracted_scene,
        wgpu_ctx,
        pipeline_cache,
        shader_manager,
    );

    finalize_source_epoch_gate(
        store,
        prepared_draws,
        geometry_frames,
        material_frames,
        object_interfaces,
        render_lists,
        extracted_scene,
    );

    // The gate above is this attempt's source linearization point. Every L1
    // publication and the later immutable Direct stream must derive only from
    // its accepted source set.
    store.publish_main_l1(pipeline_cache);
    store.publish_shadow_l1(pipeline_cache);
}

#[allow(clippy::too_many_arguments)]
fn finalize_source_epoch_gate(
    store: &mut ScenePipelinePrepareStore,
    prepared_draws: &PreparedDrawStore,
    geometry_frames: GeometryFrameResolver<'_>,
    material_frames: MaterialPlanResolver<'_>,
    object_interfaces: ObjectInterfaceResolver<'_>,
    render_lists: &RenderLists,
    extracted_scene: &ExtractedScene,
) {
    #[cfg(not(debug_assertions))]
    let _ = (object_interfaces, render_lists, extracted_scene);

    for index in 0..store.participating_source_count() {
        let prepared_source = store.participating_source_at(index);
        let Some(source) = prepared_draws.source(prepared_source) else {
            continue;
        };
        let Some(material) = material_frames.material(source.material) else {
            continue;
        };
        #[cfg(debug_assertions)]
        debug_assert!(
            resolve_prepared_source(
                source,
                geometry_frames,
                material_frames,
                object_interfaces,
                render_lists,
                extracted_scene,
            )
            .is_some(),
            "published prepared-source joins must remain immutable until the epoch gate"
        );
        // This is the only live interior-epoch validation after source
        // publication. Once accepted, the graph attempt owns an immutable
        // physical stream and must not reopen these epochs in Finalize.
        if !geometry_frames.pipeline_state_is_current(source.geometry)
            || !material.pipeline_state_is_current()
        {
            continue;
        }
        store.accept_source(
            prepared_source,
            AcceptedPreparedSource::from_prepared(source, material.interface().bind_group_id),
        );
    }
    store.finish_source_gate();
}

/// Finalize every live plan from immutable recipe/result tables.
///
/// No asset, shader, or pipeline-cache query occurs here. The only owned data
/// copied into commands are compact physical slots and small draw ranges.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_live_draw_plans(
    store: &ScenePipelinePrepareStore,
    draw_plans: &mut DrawPlanStore,
    geometry_frames: GeometryFrameResolver<'_>,
) {
    for plan_index in 0..draw_plans.plan_count() {
        let meta = draw_plans
            .plan_meta(plan_index)
            .expect("draw-plan reservation table changed during finalization");
        if !meta.is_live || meta.is_finalized {
            continue;
        }

        match meta.key {
            DrawPlanKey::Opaque | DrawPlanKey::Transparent => finalize_main_plan(
                store,
                plan_index,
                meta.id,
                matches!(meta.key, DrawPlanKey::Transparent),
                draw_plans,
                geometry_frames,
            ),
            DrawPlanKey::Prepass {
                needs_normal,
                needs_feature_id,
                needs_velocity,
            } => finalize_prepass_plan(
                store,
                plan_index,
                meta.id,
                (needs_normal, needs_feature_id, needs_velocity),
                draw_plans,
                geometry_frames,
            ),
            DrawPlanKey::Shadow { .. } => {
                finalize_shadow_plan(store, plan_index, meta.id, draw_plans, geometry_frames);
            }
        }
    }

    draw_plans.assert_all_live_finalized();
}

#[allow(clippy::too_many_arguments)]
fn gather_main_plan(
    store: &mut ScenePipelinePrepareStore,
    plan: DrawPlanId,
    intents: &[RenderIntent],
    transparent_plan: bool,
    prepared_draws: &PreparedDrawStore,
    geometry_frames: GeometryFrameResolver<'_>,
    material_frames: MaterialPlanResolver<'_>,
    object_interfaces: ObjectInterfaceResolver<'_>,
    render_lists: &RenderLists,
    extracted_scene: &ExtractedScene,
    gpu_world: &GpuGlobalState,
    wgpu_ctx: &WgpuContext,
    shader_manager: &ShaderManager,
    camera: &RenderCamera,
    scene_defines_hash: u64,
) {
    let taa_enabled = matches!(camera.aa_mode, AntiAliasingMode::TAA { .. });
    for (intent_index, intent) in intents.iter().enumerate() {
        let Some((prepared_source, resolved)) = resolve_prepared_source_for_frame(
            intent.source,
            prepared_draws,
            geometry_frames,
            material_frames,
            object_interfaces,
            render_lists,
            extracted_scene,
        ) else {
            continue;
        };
        if (resolved.item.geometry, resolved.item.material)
            != (intent.geometry_handle, intent.material_handle)
        {
            continue;
        }

        let material_pipeline = resolved.material.pipeline();
        let material_settings = material_pipeline.settings();
        let is_transparent = matches!(
            material_settings.alpha_mode,
            AlphaMode::Blend | AlphaMode::BlendMask
        ) || material_pipeline.use_transmission();
        if is_transparent != transparent_plan {
            continue;
        }
        let shader_source = material_shader_source(material_pipeline);
        let shader_source_identity = store.material_shader_source_identity(
            intent.material_handle,
            resolved.material.asset_version(),
            resolved.material.asset_identity(),
            resolved.material.material_version(),
            || shader_manager.source_identity_hash(shader_source),
        );
        let key = FastPipelineKey {
            material_handle: intent.material_handle,
            material_asset_version: resolved.material.asset_version(),
            material_asset_identity: resolved.material.asset_identity(),
            material_version: resolved.material.material_version(),
            geometry_handle: intent.geometry_handle,
            geometry_asset_version: resolved.geometry.key.asset_version,
            geometry_asset_identity: resolved.geometry.key.asset_identity,
            geometry_version: resolved.geometry.key.layout_version,
            vertex_layout_id: resolved.geometry_physical.layout_id,
            instance_variants: resolved.item.item_variant_flags,
            item_shader_features: resolved.item.item_shader_features,
            shader_source_identity,
            global_layout_id: gpu_world.layout_id,
            global_binding_interface_id: gpu_world.binding_interface_id,
            material_layout_id: resolved.material.interface().layout_id,
            material_binding_interface_id: resolved.material.interface().binding_interface_id,
            object_layout_id: resolved.object_interface.layout_id,
            object_binding_interface_id: resolved.object_interface.binding_interface_id,
            scene_variants: extracted_scene.scene_variants,
            scene_defines_hash,
            taa_enabled,
            pipeline_settings_version: wgpu_ctx.pipeline_settings_version,
            #[cfg(feature = "debug_view")]
            debug_view_mode: camera.debug_view.mode,
        };
        store.push_main(
            plan,
            SceneRecipeOrdinal::new(plan.index(), intent_index),
            prepared_source,
            key,
            intent.material_index,
            intent.distance_sq,
            transparent_plan,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_shadow_plan(
    store: &mut ScenePipelinePrepareStore,
    plan: DrawPlanId,
    intents: &[ShadowRenderIntent],
    layout: super::scene_pipeline_prepare::ShadowPipelineLayoutId,
    prepared_draws: &PreparedDrawStore,
    geometry_frames: GeometryFrameResolver<'_>,
    material_frames: MaterialPlanResolver<'_>,
    object_interfaces: ObjectInterfaceResolver<'_>,
    render_lists: &RenderLists,
    extracted_scene: &ExtractedScene,
    wgpu_ctx: &WgpuContext,
    shader_source_identity: u64,
) {
    let shadow_global_layout_id = store.shadow_layout(layout).id();
    for (intent_index, intent) in intents.iter().enumerate() {
        let Some((prepared_source, resolved)) = resolve_prepared_source_for_frame(
            intent.source,
            prepared_draws,
            geometry_frames,
            material_frames,
            object_interfaces,
            render_lists,
            extracted_scene,
        ) else {
            continue;
        };
        if (resolved.item.geometry, resolved.item.material)
            != (intent.geometry_handle, intent.material_handle)
            || resolved.material.pipeline().settings().alpha_mode == AlphaMode::Blend
        {
            continue;
        }
        let material_interface = resolved.material.interface();
        let key = FastShadowPipelineKey {
            material_handle: intent.material_handle,
            material_asset_version: resolved.material.asset_version(),
            material_asset_identity: resolved.material.asset_identity(),
            material_version: resolved.material.material_version(),
            geometry_handle: intent.geometry_handle,
            geometry_asset_version: resolved.geometry.key.asset_version,
            geometry_asset_identity: resolved.geometry.key.asset_identity,
            geometry_version: resolved.geometry.key.layout_version,
            vertex_layout_id: resolved.geometry_physical.layout_id,
            instance_variants: resolved.item.item_variant_flags,
            item_shader_features: resolved.item.item_shader_features,
            shader_source_identity,
            shadow_global_layout_id,
            material_layout_id: material_interface.layout_id,
            material_binding_interface_id: material_interface.binding_interface_id,
            object_layout_id: resolved.object_interface.layout_id,
            object_binding_interface_id: resolved.object_interface.binding_interface_id,
            pipeline_settings_version: wgpu_ctx.pipeline_settings_version,
        };
        store.push_shadow(
            plan,
            SceneRecipeOrdinal::new(plan.index(), intent_index),
            prepared_source,
            key,
            layout,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn coordinate_main_requests(
    store: &mut ScenePipelinePrepareStore,
    prepared_draws: &PreparedDrawStore,
    geometry_frames: GeometryFrameResolver<'_>,
    material_frames: MaterialPlanResolver<'_>,
    object_interfaces: ObjectInterfaceResolver<'_>,
    render_lists: &RenderLists,
    extracted_scene: &ExtractedScene,
    gpu_world: Option<&GpuGlobalState>,
    wgpu_ctx: &WgpuContext,
    system_textures: &SystemTextures,
    pipeline_cache: &mut PipelineCache,
    shader_manager: &mut ShaderManager,
    camera: &RenderCamera,
) {
    let Some(gpu_world) = gpu_world else {
        return;
    };
    if store.main_request_count() == 0 {
        return;
    }
    let pipeline_settings_version = wgpu_ctx.pipeline_settings_version;
    let mut scene_defines_hash = None;
    for order_index in 0..store.main_request_count() {
        let request = store.main_request_at(order_index);
        let (fast_key, owner, result) = store.main_request(request);
        debug_assert!(result.is_none());
        if let Some(pipeline) = pipeline_cache.get_pipeline_fast(fast_key) {
            store.resolve_main_request(request, pipeline, false);
            continue;
        }
        let scene_defines_hash =
            *scene_defines_hash.get_or_insert_with(|| extracted_scene.scene_defines.compute_hash());
        let Some(source) = prepared_draws.source(owner) else {
            continue;
        };
        let Some(resolved) = resolve_prepared_source(
            source,
            geometry_frames,
            material_frames,
            object_interfaces,
            render_lists,
            extracted_scene,
        ) else {
            continue;
        };
        if !main_request_matches_source(
            fast_key,
            resolved,
            gpu_world,
            extracted_scene,
            shader_manager,
            camera,
            scene_defines_hash,
            pipeline_settings_version,
        ) {
            continue;
        }
        let material_pipeline = resolved.material.pipeline();
        let material_settings = material_pipeline.settings();
        let is_transparent = matches!(
            material_settings.alpha_mode,
            AlphaMode::Blend | AlphaMode::BlendMask
        ) || material_pipeline.use_transmission();
        let use_clustered_shading = extracted_scene
            .scene_variants
            .contains(SceneFeatures::USE_CLUSTERED_SHADING);
        let final_a2c_enable = matches!(material_settings.alpha_mode, AlphaMode::Mask)
            && material_settings.alpha_to_coverage
            && wgpu_ctx.msaa_samples > 1;
        let mut flags = PipelineFlags::empty();
        let mut options = ShaderCompilationOptions::from_merged(
            material_pipeline.shader_defines(),
            resolved.geometry.shader_defines(),
            &extracted_scene.scene_defines,
            resolved.item.item_shader_defines(),
        );
        if use_clustered_shading {
            options.add_define("USE_CLUSTERED_SHADING", "1");
        }
        if final_a2c_enable {
            options.add_define("ALPHA_TO_COVERAGE", "1");
            flags |= PipelineFlags::ALPHA_TO_COVERAGE;
        }
        if wgpu_ctx.render_path.supports_post_processing() {
            options.add_define("HDR", "1");
        }
        let is_opaque = !is_transparent;
        if !is_opaque {
            options.add_define("IN_TRANSPARENT_PASS", "1");
        }
        let has_specular_data_mrt = match wgpu_ctx.render_path {
            RenderPath::HighFidelity => {
                is_opaque
                    && extracted_scene
                        .scene_variants
                        .intersects(SceneFeatures::USE_SSS | SceneFeatures::USE_SSR)
            }
            RenderPath::BasicForward => false,
        };
        let has_material_data_mrt = match wgpu_ctx.render_path {
            RenderPath::HighFidelity => {
                is_opaque
                    && extracted_scene
                        .scene_variants
                        .intersects(SceneFeatures::USE_SSGI | SceneFeatures::USE_SSR)
            }
            RenderPath::BasicForward => false,
        };
        if has_specular_data_mrt {
            options.add_define("HAS_MRT_SPECULAR_DATA", "1");
        }
        if has_material_data_mrt {
            options.add_define("HAS_MRT_MATERIAL_DATA", "1");
        }

        let shader_source = material_shader_source(material_pipeline);
        debug_assert_eq!(
            shader_manager.source_identity_hash(shader_source),
            fast_key.shader_source_identity
        );
        let shader_hash = shader_manager.pipeline_shader_hash(shader_source, &options);
        let use_depth_pre = wgpu_ctx.msaa_samples == 1 && wgpu_ctx.render_path.requires_z_prepass();
        let depth_write = if is_opaque && use_depth_pre {
            false
        } else {
            material_settings.depth_write
        };
        if depth_write {
            flags |= PipelineFlags::DEPTH_WRITE;
        }
        let depth_compare = if material_settings.depth_test {
            if is_opaque && use_depth_pre {
                wgpu::CompareFunction::Equal
            } else {
                wgpu::CompareFunction::Greater
            }
        } else {
            wgpu::CompareFunction::Always
        };
        let color_format = wgpu_ctx
            .render_path
            .main_color_format(wgpu_ctx.surface_view_format);
        let mut color_targets = smallvec::smallvec![ColorTargetKey::from(wgpu::ColorTargetState {
            format: color_format,
            blend: matches!(
                material_settings.alpha_mode,
                AlphaMode::Blend | AlphaMode::BlendMask
            )
            .then_some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        if has_specular_data_mrt {
            color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                format: HDR_TEXTURE_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }));
        }
        if has_material_data_mrt {
            color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }));
        }
        let screen_layout = if use_clustered_shading {
            &system_textures.screen_layout_clustered
        } else {
            &system_textures.screen_layout
        };
        let canonical_key = GraphicsPipelineKey {
            shader_hash,
            vertex_layout_id: resolved.geometry_physical.layout_id,
            bind_group_layout_ids: [
                gpu_world.layout_id,
                resolved.material.interface().layout_id,
                resolved.object_interface.layout_id,
                screen_layout.id(),
            ],
            topology: resolved.geometry.topology,
            cull_mode: side_cull_mode(material_settings.side),
            depth_compare,
            blend_state: matches!(
                material_settings.alpha_mode,
                AlphaMode::Blend | AlphaMode::BlendMask
            )
            .then(|| BlendStateKey::from(wgpu::BlendState::ALPHA_BLENDING)),
            color_targets,
            depth_format: wgpu_ctx.depth_format,
            sample_count: wgpu_ctx.msaa_samples,
            front_face: item_front_face(resolved.item),
            flags,
        };
        let pipeline = pipeline_cache.get_or_create_graphics(
            &wgpu_ctx.device,
            shader_manager,
            shader_source,
            &canonical_key,
            &options,
            &resolved.geometry_physical.layout_info,
            &resolved.material.interface().layout,
            &resolved.material.interface().binding_wgsl,
            &resolved.object_interface.layout,
            &resolved.object_interface.binding_wgsl,
            gpu_world,
            screen_layout,
        );
        store.resolve_main_request(request, pipeline, true);
    }
}

#[allow(clippy::too_many_arguments)]
fn coordinate_prepass_requests(
    store: &mut ScenePipelinePrepareStore,
    prepared_draws: &PreparedDrawStore,
    geometry_frames: GeometryFrameResolver<'_>,
    material_frames: MaterialPlanResolver<'_>,
    object_interfaces: ObjectInterfaceResolver<'_>,
    render_lists: &RenderLists,
    extracted_scene: &ExtractedScene,
    gpu_world: Option<&GpuGlobalState>,
    wgpu_ctx: &WgpuContext,
    pipeline_cache: &mut PipelineCache,
    shader_manager: &mut ShaderManager,
) {
    let Some(gpu_world) = gpu_world else {
        return;
    };
    let shader_source = ShaderSource::File("entry/utility/depth_prepass");
    let mut request_invariants = None;
    for order_index in 0..store.prepass_request_count() {
        let request = store.prepass_request_at(order_index);
        if store.prepass_result(request).is_some() {
            continue;
        }
        let (key, owner, cached) = store.prepass_request(request);
        debug_assert!(cached.is_none());
        let (scene_defines_hash, shader_source_identity) =
            *request_invariants.get_or_insert_with(|| {
                (
                    extracted_scene.scene_defines.compute_hash(),
                    shader_manager.source_identity_hash(shader_source),
                )
            });
        let Some(source) = prepared_draws.source(owner) else {
            continue;
        };
        let Some(resolved) = resolve_prepared_source(
            source,
            geometry_frames,
            material_frames,
            object_interfaces,
            render_lists,
            extracted_scene,
        ) else {
            continue;
        };
        if !prepass_request_matches_source(
            key,
            resolved,
            gpu_world,
            wgpu_ctx,
            scene_defines_hash,
            wgpu_ctx.pipeline_settings_version,
            shader_source_identity,
        ) {
            continue;
        }
        let material_pipeline = resolved.material.pipeline();
        if matches!(
            material_pipeline.settings().alpha_mode,
            AlphaMode::Blend | AlphaMode::BlendMask
        ) || material_pipeline.use_transmission()
        {
            continue;
        }
        let mut options = ShaderCompilationOptions::from_merged(
            material_pipeline.shader_defines(),
            resolved.geometry.shader_defines(),
            &extracted_scene.scene_defines,
            resolved.item.item_shader_defines(),
        );
        options.add_define("IS_PREPASS", "1");
        if key.needs_normal {
            options.add_define("OUTPUT_NORMAL", "1");
        }
        if key.needs_velocity {
            options.add_define("HAS_VELOCITY_TARGET", "1");
        }
        if key.needs_feature_id {
            options.add_define("OUTPUT_FEATURE_ID", "1");
        }
        options.inject_code(
            "vertex_input_code",
            &resolved.geometry_physical.layout_info.vertex_input_code,
        );
        options.inject_code(
            "binding_code",
            format!(
                "{}\n{}\n{}",
                gpu_world.binding_wgsl.as_str(),
                &resolved.material.interface().binding_wgsl,
                &resolved.object_interface.binding_wgsl
            ),
        );
        options.inject_code(
            "scene_lighting_structs",
            myth_resources::uniforms::scene_lighting_structs_wgsl(),
        );
        let (shader_module, shader_hash) =
            shader_manager.get_or_compile(&wgpu_ctx.device, shader_source, &options);
        let layout = wgpu_ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Prepass Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&gpu_world.layout),
                    Some(&resolved.material.interface().layout),
                    Some(&resolved.object_interface.layout),
                ],
                immediate_size: 0,
            });
        let vertex_buffers_layout: Vec<_> = resolved
            .geometry_physical
            .layout_info
            .buffers
            .iter()
            .map(|layout| layout.as_wgpu())
            .collect();
        let mut color_targets = smallvec::smallvec![];
        if key.needs_normal {
            color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                format: NORMAL_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }));
        }
        if key.needs_feature_id {
            color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                format: FEATURE_ID_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }));
        }
        if key.needs_velocity {
            color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                format: VELOCITY_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }));
        }
        let canonical_key = SimpleGeometryPipelineKey {
            shader_hash,
            vertex_layout_id: resolved.geometry_physical.layout_id,
            bind_group_layout_ids: key.bind_group_layout_ids,
            color_targets,
            depth_stencil: DepthStencilKey::from(wgpu::DepthStencilState {
                format: key.depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            topology: key.topology,
            cull_mode: key.cull_mode,
            front_face: key.front_face,
            sample_count: 1,
        };
        let pipeline = pipeline_cache.get_or_create_simple_geometry(
            &wgpu_ctx.device,
            shader_module,
            &layout,
            &canonical_key,
            "Prepass Pipeline",
            &vertex_buffers_layout,
        );
        store.resolve_prepass_request(request, pipeline, true);
    }
}

#[allow(clippy::too_many_arguments)]
fn coordinate_shadow_requests(
    store: &mut ScenePipelinePrepareStore,
    prepared_draws: &PreparedDrawStore,
    geometry_frames: GeometryFrameResolver<'_>,
    material_frames: MaterialPlanResolver<'_>,
    object_interfaces: ObjectInterfaceResolver<'_>,
    render_lists: &RenderLists,
    extracted_scene: &ExtractedScene,
    wgpu_ctx: &WgpuContext,
    pipeline_cache: &mut PipelineCache,
    shader_manager: &mut ShaderManager,
) {
    let shader_source = ShaderSource::File("entry/utility/depth_prepass");
    if store.shadow_request_count() == 0 {
        return;
    }
    let pipeline_settings_version = wgpu_ctx.pipeline_settings_version;
    let mut shader_source_identity = None;
    for order_index in 0..store.shadow_request_count() {
        let request = store.shadow_request_at(order_index);
        let (fast_key, owner, layout_id, result) = store.shadow_request(request);
        debug_assert!(result.is_none());
        if let Some(pipeline) = pipeline_cache.get_shadow_pipeline_fast(fast_key) {
            store.resolve_shadow_request(request, pipeline, false);
            continue;
        }
        let shader_source_identity = *shader_source_identity
            .get_or_insert_with(|| shader_manager.source_identity_hash(shader_source));
        let Some(source) = prepared_draws.source(owner) else {
            continue;
        };
        let Some(resolved) = resolve_prepared_source(
            source,
            geometry_frames,
            material_frames,
            object_interfaces,
            render_lists,
            extracted_scene,
        ) else {
            continue;
        };
        let shadow_layout = store.shadow_layout(layout_id);
        if !shadow_request_matches_source(
            fast_key,
            resolved,
            shadow_layout.id(),
            pipeline_settings_version,
            shader_source_identity,
        ) {
            continue;
        }
        if resolved.material.pipeline().settings().alpha_mode == AlphaMode::Blend {
            continue;
        }
        let mut options = ShaderCompilationOptions::from_merged(
            resolved.material.pipeline().shader_defines(),
            resolved.geometry.shader_defines(),
            &myth_resources::shader_defines::ShaderDefines::new(),
            resolved.item.item_shader_defines(),
        );
        options.add_define("SHADOW_PASS", "1");
        options.inject_code(
            "vertex_input_code",
            &resolved.geometry_physical.layout_info.vertex_input_code,
        );
        options.inject_code(
            "binding_code",
            format!(
                "{}\n{}\n{}",
                SHADOW_BINDING_WGSL,
                &resolved.material.interface().binding_wgsl,
                &resolved.object_interface.binding_wgsl
            ),
        );
        options.inject_code(
            "scene_lighting_structs",
            myth_resources::uniforms::scene_lighting_structs_wgsl(),
        );
        let (shader_module, shader_hash) =
            shader_manager.get_or_compile(&wgpu_ctx.device, shader_source, &options);
        let layout = wgpu_ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Pipeline Layout"),
                bind_group_layouts: &[
                    Some(shadow_layout),
                    Some(&resolved.material.interface().layout),
                    Some(&resolved.object_interface.layout),
                ],
                immediate_size: 0,
            });
        let vertex_buffers_layout: Vec<_> = resolved
            .geometry_physical
            .layout_info
            .buffers
            .iter()
            .map(|layout| layout.as_wgpu())
            .collect();
        let canonical_key = SimpleGeometryPipelineKey {
            shader_hash,
            vertex_layout_id: resolved.geometry_physical.layout_id,
            bind_group_layout_ids: [
                shadow_layout.id(),
                resolved.material.interface().layout_id,
                resolved.object_interface.layout_id,
            ],
            color_targets: smallvec::smallvec![],
            depth_stencil: DepthStencilKey::from(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            topology: resolved.geometry.topology,
            cull_mode: Some(wgpu::Face::Front),
            front_face: item_front_face(resolved.item),
            sample_count: 1,
        };
        let pipeline = pipeline_cache.get_or_create_simple_geometry(
            &wgpu_ctx.device,
            shader_module,
            &layout,
            &canonical_key,
            "Shadow Pipeline",
            &vertex_buffers_layout,
        );
        store.resolve_shadow_request(request, pipeline, true);
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_main_plan(
    store: &ScenePipelinePrepareStore,
    plan_index: usize,
    plan: DrawPlanId,
    transparent: bool,
    draw_plans: &mut DrawPlanStore,
    geometry_frames: GeometryFrameResolver<'_>,
) {
    let recipes = store.main_draws(plan_index);
    draw_plans.reserve_commands(recipes.len());
    let start = draw_plans.begin_finalize(plan);
    for &recipe in recipes {
        assert_recipe_plan(recipe.draw_plan, recipe.ordinal, plan, plan_index);
        debug_assert_eq!(recipe.transparent, transparent);
        let Some(pipeline) = store.main_result(recipe.pipeline) else {
            continue;
        };
        let Some(source) = store.accepted_source(recipe.prepared_source) else {
            continue;
        };
        let geometry = geometry_frames
            .physical(source.geometry)
            .expect("accepted main source lost its immutable geometry physical");
        let sort_key = RenderKey::new(
            pipeline,
            recipe.material_index,
            recipe.distance_sq,
            transparent,
        );
        push_resolved_command(draw_plans, source, geometry, pipeline, sort_key.bits());
    }
    draw_plans.sort_commands_from(start);
    draw_plans.finish_finalize(plan, start);
}

#[allow(clippy::too_many_arguments)]
fn finalize_prepass_plan(
    store: &ScenePipelinePrepareStore,
    plan_index: usize,
    plan: DrawPlanId,
    output_mask: (bool, bool, bool),
    draw_plans: &mut DrawPlanStore,
    geometry_frames: GeometryFrameResolver<'_>,
) {
    let recipes = store.prepass_draws(plan_index);
    draw_plans.reserve_commands(recipes.len());
    let start = draw_plans.begin_finalize(plan);
    for &recipe in recipes {
        assert_recipe_plan(recipe.draw_plan, recipe.ordinal, plan, plan_index);
        let (pipeline_key, _, _) = store.prepass_request(recipe.pipeline);
        assert_eq!(
            (
                pipeline_key.needs_normal,
                pipeline_key.needs_feature_id,
                pipeline_key.needs_velocity,
            ),
            output_mask,
            "Prepass recipe output mask does not match its draw plan"
        );
        let Some(pipeline) = store.prepass_result(recipe.pipeline) else {
            continue;
        };
        let Some(source) = store.accepted_source(recipe.prepared_source) else {
            continue;
        };
        let geometry = geometry_frames
            .physical(source.geometry)
            .expect("accepted Prepass source lost its immutable geometry physical");
        let sort_key = RenderKey::new(pipeline, recipe.material_index, recipe.distance_sq, false);
        push_resolved_command(draw_plans, source, geometry, pipeline, sort_key.bits());
    }
    draw_plans.sort_commands_from(start);
    draw_plans.finish_finalize(plan, start);
}

#[allow(clippy::too_many_arguments)]
fn finalize_shadow_plan(
    store: &ScenePipelinePrepareStore,
    plan_index: usize,
    plan: DrawPlanId,
    draw_plans: &mut DrawPlanStore,
    geometry_frames: GeometryFrameResolver<'_>,
) {
    let recipes = store.shadow_draws(plan_index);
    draw_plans.reserve_commands(recipes.len());
    let start = draw_plans.begin_finalize(plan);
    for &recipe in recipes {
        assert_recipe_plan(recipe.draw_plan, recipe.ordinal, plan, plan_index);
        let Some(pipeline) = store.shadow_result(recipe.pipeline) else {
            continue;
        };
        let Some(source) = store.accepted_source(recipe.prepared_source) else {
            continue;
        };
        let geometry = geometry_frames
            .physical(source.geometry)
            .expect("accepted Shadow source lost its immutable geometry physical");
        let sort_key = u64::from(pipeline.0) << 32 | (source.material_bind_group_id & 0xffff_ffff);
        push_resolved_command(draw_plans, source, geometry, pipeline, sort_key);
    }
    draw_plans.sort_commands_from(start);
    draw_plans.finish_finalize(plan, start);
}

fn assert_recipe_plan(
    recipe_plan: DrawPlanId,
    ordinal: SceneRecipeOrdinal,
    plan: DrawPlanId,
    plan_index: usize,
) {
    assert_eq!(
        recipe_plan, plan,
        "recipe was routed to a foreign draw plan"
    );
    assert_eq!(
        ordinal.plan_index as usize, plan_index,
        "recipe ordinal does not match its draw plan"
    );
}

fn push_resolved_command(
    draw_plans: &mut DrawPlanStore,
    source: &AcceptedPreparedSource,
    geometry: &GeometryFramePhysical,
    pipeline_id: RenderPipelineId,
    sort_key: u64,
) {
    if source.commits_morph_history {
        draw_plans.record_emitted_morph_history_source(source.frame_source);
    }
    draw_plans.push_command(
        DrawCommand {
            #[cfg(test)]
            frame_source: source.frame_source,
            sort_key,
            pipeline_id,
            vertex_buffer_range: 0..0,
            index_buffer: geometry.index_buffer,
            bind_group_1: Some(source.material_bind_group),
            bind_group_2: source.object_bind_group,
            stencil_reference: None,
            vertex_range: geometry.draw_range.clone(),
            instance_range: geometry.instance_range.clone(),
        },
        &geometry.vertex_buffers,
    );
}

#[derive(Clone, Copy)]
pub(crate) struct ResolvedPipelineSource<'a> {
    pub(crate) item: &'a ExtractedRenderItem,
    pub(crate) geometry: &'a GeometryFrameRecord,
    pub(crate) geometry_physical: &'a GeometryFramePhysical,
    pub(crate) material: PreparedMaterialFrame<'a>,
    pub(crate) object_interface: &'a ObjectInterfaceRecord,
}

/// Resolve immutable graph-attempt records without reopening live asset
/// epochs. Scene-source publication already established this exact join; the
/// one later epoch check belongs exclusively to `finalize_source_epoch_gate`.
#[allow(clippy::too_many_arguments)]
fn resolve_prepared_source<'a>(
    source: &'a PreparedDrawSource,
    geometry_frames: GeometryFrameResolver<'a>,
    material_frames: MaterialPlanResolver<'a>,
    object_interfaces: ObjectInterfaceResolver<'a>,
    render_lists: &'a RenderLists,
    extracted_scene: &'a ExtractedScene,
) -> Option<ResolvedPipelineSource<'a>> {
    if render_lists.extracted_source(source.frame_source) != Some(source.extracted_source) {
        return None;
    }
    let item = extracted_scene
        .render_items
        .get(source.extracted_source.index())?;
    let geometry = geometry_frames.geometry(source.geometry)?;
    let geometry_physical = geometry_frames.physical(source.geometry)?;
    let material = material_frames.material(source.material)?;
    let object_interface = object_interfaces.interface(source.object_interface)?;
    if geometry.key.handle != item.geometry
        || material.handle() != item.material
        || (geometry.key.asset_version, geometry.key.asset_identity)
            != (item.geometry_asset_version, item.geometry_asset_identity)
        || (material.asset_version(), material.asset_identity())
            != (item.material_asset_version, item.material_asset_identity)
        || material.material_version() != item.material_version
    {
        return None;
    }
    Some(ResolvedPipelineSource {
        item,
        geometry,
        geometry_physical,
        material,
        object_interface,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_prepared_source_for_frame<'a>(
    frame_source: super::frame::FrameSourceId,
    prepared_draws: &'a PreparedDrawStore,
    geometry_frames: GeometryFrameResolver<'a>,
    material_frames: MaterialPlanResolver<'a>,
    object_interfaces: ObjectInterfaceResolver<'a>,
    render_lists: &'a RenderLists,
    extracted_scene: &'a ExtractedScene,
) -> Option<(super::frame::PreparedSourceId, ResolvedPipelineSource<'a>)> {
    let (prepared_source, source) = prepared_draws.source_for_frame(frame_source)?;
    let resolved = resolve_prepared_source(
        source,
        geometry_frames,
        material_frames,
        object_interfaces,
        render_lists,
        extracted_scene,
    )?;
    Some((prepared_source, resolved))
}

fn main_request_matches_source(
    key: FastPipelineKey,
    resolved: ResolvedPipelineSource<'_>,
    gpu_world: &GpuGlobalState,
    extracted_scene: &ExtractedScene,
    shader_manager: &ShaderManager,
    camera: &RenderCamera,
    scene_defines_hash: u64,
    pipeline_settings_version: u64,
) -> bool {
    let matches = key.material_handle == resolved.material.handle()
        && key.material_asset_version == resolved.material.asset_version()
        && key.material_asset_identity == resolved.material.asset_identity()
        && key.material_version == resolved.material.material_version()
        && key.geometry_handle == resolved.geometry.key.handle
        && key.geometry_asset_version == resolved.geometry.key.asset_version
        && key.geometry_asset_identity == resolved.geometry.key.asset_identity
        && key.geometry_version == resolved.geometry.key.layout_version
        && key.vertex_layout_id == resolved.geometry_physical.layout_id
        && key.instance_variants == resolved.item.item_variant_flags
        && key.item_shader_features == resolved.item.item_shader_features
        && key.global_layout_id == gpu_world.layout_id
        && key.global_binding_interface_id == gpu_world.binding_interface_id
        && key.material_layout_id == resolved.material.interface().layout_id
        && key.material_binding_interface_id == resolved.material.interface().binding_interface_id
        && key.object_layout_id == resolved.object_interface.layout_id
        && key.object_binding_interface_id == resolved.object_interface.binding_interface_id
        && key.scene_variants == extracted_scene.scene_variants
        && key.scene_defines_hash == scene_defines_hash
        && key.pipeline_settings_version == pipeline_settings_version
        && key.shader_source_identity
            == shader_manager
                .source_identity_hash(material_shader_source(resolved.material.pipeline()))
        && key.taa_enabled == matches!(camera.aa_mode, AntiAliasingMode::TAA { .. });
    #[cfg(feature = "debug_view")]
    let matches = matches && key.debug_view_mode == camera.debug_view.mode;
    matches
}

fn prepass_request_matches_source(
    key: super::scene_pipeline_prepare::PrepassPipelineL1Key,
    resolved: ResolvedPipelineSource<'_>,
    gpu_world: &GpuGlobalState,
    wgpu_ctx: &WgpuContext,
    scene_defines_hash: u64,
    pipeline_settings_version: u64,
    shader_source_identity: u64,
) -> bool {
    key.geometry_handle == resolved.geometry.key.handle
        && key.geometry_asset_version == resolved.geometry.key.asset_version
        && key.geometry_asset_identity == resolved.geometry.key.asset_identity
        && key.geometry_layout_version == resolved.geometry.key.layout_version
        && key.material_handle == resolved.material.handle()
        && key.material_asset_version == resolved.material.asset_version()
        && key.material_asset_identity == resolved.material.asset_identity()
        && key.material_version == resolved.material.material_version()
        && key.item_variants == resolved.item.item_variant_flags
        && key.item_shader_features == resolved.item.item_shader_features
        && key.scene_defines_hash == scene_defines_hash
        && key.pipeline_settings_version == pipeline_settings_version
        && key.global_binding_interface_id == gpu_world.binding_interface_id
        && key.material_binding_interface_id == resolved.material.interface().binding_interface_id
        && key.object_binding_interface_id == resolved.object_interface.binding_interface_id
        && key.vertex_layout_id == resolved.geometry_physical.layout_id
        && key.bind_group_layout_ids
            == [
                gpu_world.layout_id,
                resolved.material.interface().layout_id,
                resolved.object_interface.layout_id,
            ]
        && key.depth_format == wgpu_ctx.depth_format
        && key.topology == resolved.geometry.topology
        && key.cull_mode == side_cull_mode(resolved.material.pipeline().settings().side)
        && key.front_face == item_front_face(resolved.item)
        && key.shader_source_identity == shader_source_identity
}

fn shadow_request_matches_source(
    key: FastShadowPipelineKey,
    resolved: ResolvedPipelineSource<'_>,
    shadow_layout_id: u64,
    pipeline_settings_version: u64,
    shader_source_identity: u64,
) -> bool {
    key.material_handle == resolved.material.handle()
        && key.material_asset_version == resolved.material.asset_version()
        && key.material_asset_identity == resolved.material.asset_identity()
        && key.material_version == resolved.material.material_version()
        && key.geometry_handle == resolved.geometry.key.handle
        && key.geometry_asset_version == resolved.geometry.key.asset_version
        && key.geometry_asset_identity == resolved.geometry.key.asset_identity
        && key.geometry_version == resolved.geometry.key.layout_version
        && key.vertex_layout_id == resolved.geometry_physical.layout_id
        && key.instance_variants == resolved.item.item_variant_flags
        && key.item_shader_features == resolved.item.item_shader_features
        && key.shadow_global_layout_id == shadow_layout_id
        && key.material_layout_id == resolved.material.interface().layout_id
        && key.material_binding_interface_id == resolved.material.interface().binding_interface_id
        && key.object_layout_id == resolved.object_interface.layout_id
        && key.object_binding_interface_id == resolved.object_interface.binding_interface_id
        && key.pipeline_settings_version == pipeline_settings_version
        && key.shader_source_identity == shader_source_identity
}

fn material_shader_source(
    pipeline: &crate::core::gpu::MaterialPipelineSnapshot,
) -> ShaderSource<'_> {
    if let Some(template_source) = pipeline.shader_template() {
        ShaderSource::Inline {
            name: pipeline.shader_name(),
            source: template_source,
            mode: pipeline.shader_template_mode(),
        }
    } else {
        ShaderSource::File(pipeline.shader_name())
    }
}

#[inline]
fn side_cull_mode(side: Side) -> Option<wgpu::Face> {
    match side {
        Side::Front => Some(wgpu::Face::Back),
        Side::Back => Some(wgpu::Face::Front),
        Side::Double => None,
    }
}

#[inline]
fn item_front_face(item: &ExtractedRenderItem) -> wgpu::FrontFace {
    if item.item_variant_flags & 0x1 != 0 {
        wgpu::FrontFace::Cw
    } else {
        wgpu::FrontFace::Ccw
    }
}
