//! Post-DPE scene draw preparation and command baking.
//!
//! CPU-only [`RenderIntent`](super::frame::RenderIntent) lists are collected
//! before graph construction. After compilation has projected live
//! [`DrawPlanKey`] demands and the exact source union has been materialized,
//! this module derives pipelines only for those live plans and writes blind,
//! graph-attempt-owned [`DrawCommand`] ranges.

use log::error;
use rustc_hash::FxHashMap;

use crate::HDR_TEXTURE_FORMAT;
use crate::core::{ResourceManager, WgpuContext};
use crate::pipeline::pipeline_key::PipelineFlags;
use crate::pipeline::shader_gen::ShaderCompilationOptions;
use crate::pipeline::shader_manager::{ShaderManager, ShaderSource};
use crate::pipeline::{
    BlendStateKey, ColorTargetKey, DepthStencilKey, FastPipelineKey, FastShadowPipelineKey,
    GraphicsPipelineKey, PipelineCache, RenderPipelineId, SimpleGeometryPipelineKey,
};
use crate::settings::RenderPath;
use myth_resources::material::{AlphaMode, Side};
use myth_resources::{AntiAliasingMode, RenderableMaterialTrait};
use myth_scene::camera::RenderCamera;

use super::extracted::{ExtractedScene, SceneFeatures};
use super::frame::{
    DrawCommand, DrawPlanId, DrawPlanKey, DrawPlanStore, PreparedDrawSource, PreparedDrawStore,
    RenderIntent, RenderKey, RenderLists, ShadowRenderIntent,
};
use super::render_state::RenderState;

/// Shadow-only WGSL binding declaration, injected into shadow depth shaders.
const SHADOW_BINDING_WGSL: &str = "
struct Struct_shadow_light {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u_shadow_light: Struct_shadow_light;
";

/// Finalize every draw plan demanded by a pass that survived DPE.
///
/// Prepass is finalized by its live PassNode because its output mask and L1
/// cache are Feature-owned. Opaque, Transparent, BasicForward, and exact
/// shadow-view plans are finalized here after all live-node preparation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_live_draw_plans(
    draw_plans: &mut DrawPlanStore,
    prepared_draws: &PreparedDrawStore,
    render_lists: &RenderLists,
    extracted_scene: &ExtractedScene,
    render_state: &RenderState,
    wgpu_ctx: &WgpuContext,
    resource_manager: &mut ResourceManager,
    pipeline_cache: &mut PipelineCache,
    shader_manager: &mut ShaderManager,
    camera: &RenderCamera,
) {
    let mut shadow_global_layout = None;

    for plan_index in 0..draw_plans.plan_count() {
        let meta = draw_plans
            .plan_meta(plan_index)
            .expect("draw-plan reservation table changed during finalization");
        if !meta.is_live || meta.is_finalized {
            continue;
        }

        match meta.key {
            DrawPlanKey::Opaque => bake_main_intents(
                meta.id,
                &render_lists.opaque,
                false,
                draw_plans,
                prepared_draws,
                extracted_scene,
                render_state,
                wgpu_ctx,
                resource_manager,
                pipeline_cache,
                shader_manager,
                camera,
            ),
            DrawPlanKey::Transparent => bake_main_intents(
                meta.id,
                &render_lists.transparent,
                true,
                draw_plans,
                prepared_draws,
                extracted_scene,
                render_state,
                wgpu_ctx,
                resource_manager,
                pipeline_cache,
                shader_manager,
                camera,
            ),
            DrawPlanKey::Prepass { .. } => {
                panic!("live prepass draw plan was not finalized by PrepassPassNode::prepare")
            }
            DrawPlanKey::Shadow {
                light_id,
                layer_index,
            } => {
                let (layout, layout_id) = shadow_global_layout.get_or_insert_with(|| {
                    let entries = [wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<glam::Mat4>() as u64,
                            ),
                        },
                        count: None,
                    }];
                    resource_manager.get_or_create_layout(&entries)
                });
                let intents = render_lists
                    .shadow_queues
                    .get(&(light_id, layer_index))
                    .map_or(&[][..], Vec::as_slice);
                bake_shadow_intents(
                    meta.id,
                    intents,
                    draw_plans,
                    prepared_draws,
                    extracted_scene,
                    wgpu_ctx,
                    layout,
                    *layout_id,
                    pipeline_cache,
                    shader_manager,
                );
            }
        }
    }

    draw_plans.assert_all_live_finalized();
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn bake_main_intents(
    plan: DrawPlanId,
    intents: &[RenderIntent],
    transparent_plan: bool,
    draw_plans: &mut DrawPlanStore,
    prepared_draws: &PreparedDrawStore,
    extracted_scene: &ExtractedScene,
    render_state: &RenderState,
    wgpu_ctx: &WgpuContext,
    resource_manager: &ResourceManager,
    pipeline_cache: &mut PipelineCache,
    shader_manager: &mut ShaderManager,
    camera: &RenderCamera,
) {
    draw_plans.reserve_commands(intents.len());
    let start = draw_plans.begin_finalize(plan);

    let Some(gpu_world) =
        resource_manager.get_global_state(render_state.id, extracted_scene.scene_id)
    else {
        error!(
            "Render Environment missing for render_state_id {}, scene_id {}",
            render_state.id, extracted_scene.scene_id
        );
        draw_plans.finish_finalize(plan, start);
        return;
    };

    let color_format = wgpu_ctx
        .render_path
        .main_color_format(wgpu_ctx.surface_view_format);
    let depth_format = wgpu_ctx.depth_format;
    let sample_count = wgpu_ctx.msaa_samples;
    let pipeline_settings_version = wgpu_ctx.pipeline_settings_version;
    let use_clustered_shading = extracted_scene
        .scene_variants
        .contains(SceneFeatures::USE_CLUSTERED_SHADING);
    let scene_defines_hash = extracted_scene.scene_defines.compute_hash();
    let taa_enabled = matches!(camera.aa_mode, AntiAliasingMode::TAA { .. });
    let use_depth_pre = sample_count == 1 && wgpu_ctx.render_path.requires_z_prepass();
    let mut shader_source_identities = FxHashMap::default();
    for intent in intents {
        let Some(source) = prepared_draws.source(intent.source_item_index) else {
            continue;
        };
        let Some(item) = extracted_scene
            .render_items
            .get(intent.source_item_index as usize)
        else {
            continue;
        };
        if !source.matches_extracted(item, intent.geometry_handle, intent.material_handle) {
            continue;
        }
        if !source.pipeline_state_is_current() {
            continue;
        }
        let geometry = &source.geometry_asset;
        let material = &source.material_asset;
        let is_transparent = material.is_transparent() || material.use_transmission();
        if is_transparent != transparent_plan {
            continue;
        }
        let shader_source = if let Some(template_source) = material.shader_template() {
            ShaderSource::Inline {
                name: material.shader_name(),
                source: template_source,
                mode: material.shader_template_mode(),
            }
        } else {
            ShaderSource::File(material.shader_name())
        };
        let shader_source_identity = *shader_source_identities
            .entry((
                intent.material_handle,
                source.material_asset_version,
                source.material_asset_identity,
                source.material_version,
            ))
            .or_insert_with(|| shader_manager.source_identity_hash(shader_source));

        let fast_key = FastPipelineKey {
            material_handle: intent.material_handle,
            material_asset_version: source.material_asset_version,
            material_asset_identity: source.material_asset_identity,
            material_version: source.material_version,
            geometry_handle: intent.geometry_handle,
            geometry_asset_version: source.geometry_asset_version,
            geometry_asset_identity: source.geometry_asset_identity,
            geometry_version: source.geometry_layout_version,
            vertex_layout_id: source.geometry_layout_id,
            instance_variants: item.item_variant_flags,
            item_shader_defines_hash: item.item_shader_defines_hash,
            shader_source_identity,
            global_layout_id: gpu_world.layout_id,
            global_binding_wgsl_hash: gpu_world.binding_wgsl_hash,
            material_layout_id: source.material_layout_id,
            material_binding_wgsl_hash: source.material_binding_wgsl_hash,
            object_layout_id: source.object_layout_id,
            object_binding_wgsl_hash: source.object_binding_wgsl_hash,
            scene_variants: extracted_scene.scene_variants,
            scene_defines_hash,
            taa_enabled,
            pipeline_settings_version,
            #[cfg(feature = "debug_view")]
            debug_view_mode: camera.debug_view.mode,
        };

        let (pipeline_id, publish_l1) = if let Some(id) = pipeline_cache.get_pipeline_fast(fast_key)
        {
            (id, false)
        } else {
            let geo_defines = geometry.shader_defines();
            let mat_defines = material.shader_defines();
            let mut flags = PipelineFlags::empty();
            let final_a2c_enable = matches!(material.alpha_mode(), AlphaMode::Mask)
                && material.alpha_to_coverage()
                && sample_count > 1;
            let mut options = ShaderCompilationOptions::from_merged(
                &mat_defines,
                geo_defines,
                &extracted_scene.scene_defines,
                &item.item_shader_defines,
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

            let is_opaque_item = !is_transparent;
            if !is_opaque_item {
                options.add_define("IN_TRANSPARENT_PASS", "1");
            }
            let has_specular_data_mrt = match wgpu_ctx.render_path {
                RenderPath::HighFidelity => {
                    is_opaque_item
                        && extracted_scene
                            .scene_variants
                            .intersects(SceneFeatures::USE_SSS | SceneFeatures::USE_SSR)
                }
                RenderPath::BasicForward => false,
            };
            let has_material_data_mrt = match wgpu_ctx.render_path {
                RenderPath::HighFidelity => {
                    is_opaque_item
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

            let shader_hash = shader_manager.pipeline_shader_hash(shader_source, &options);
            let depth_write = if is_opaque_item && use_depth_pre {
                false
            } else {
                material.depth_write()
            };
            if depth_write {
                flags |= PipelineFlags::DEPTH_WRITE;
            }
            let depth_compare = if material.depth_test() {
                if is_opaque_item && use_depth_pre {
                    wgpu::CompareFunction::Equal
                } else {
                    wgpu::CompareFunction::Greater
                }
            } else {
                wgpu::CompareFunction::Always
            };

            let main_color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: color_format,
                blend: material
                    .is_transparent()
                    .then_some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            });
            let mut color_targets = smallvec::smallvec![main_color_target];
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
                &resource_manager.system_textures().screen_layout_clustered
            } else {
                &resource_manager.system_textures().screen_layout
            };
            let canonical_key = GraphicsPipelineKey {
                shader_hash,
                vertex_layout_id: source.geometry_layout_id,
                bind_group_layout_ids: [
                    gpu_world.layout_id,
                    source.material_layout_id,
                    source.object_layout_id,
                    screen_layout.id(),
                ],
                topology: geometry.topology,
                cull_mode: match material.side() {
                    Side::Front => Some(wgpu::Face::Back),
                    Side::Back => Some(wgpu::Face::Front),
                    Side::Double => None,
                },
                depth_compare,
                blend_state: material
                    .is_transparent()
                    .then(|| BlendStateKey::from(wgpu::BlendState::ALPHA_BLENDING)),
                color_targets,
                depth_format,
                sample_count,
                front_face: if item.item_variant_flags & 0x1 != 0 {
                    wgpu::FrontFace::Cw
                } else {
                    wgpu::FrontFace::Ccw
                },
                flags,
            };

            let id = pipeline_cache.get_or_create_graphics(
                &wgpu_ctx.device,
                shader_manager,
                shader_source,
                &canonical_key,
                &options,
                &source.geometry_layout_info,
                &source.material_layout,
                &source.material_binding_wgsl,
                &source.object_layout,
                &source.object_binding_wgsl,
                gpu_world,
                screen_layout,
            );
            (id, true)
        };

        let sort_key = RenderKey::new(
            pipeline_id,
            intent.material_index,
            intent.distance_sq,
            transparent_plan,
        );
        if !source.pipeline_state_is_current() {
            continue;
        }
        if publish_l1 {
            pipeline_cache.insert_pipeline_fast(fast_key, pipeline_id);
        }
        push_prepared_source_command(
            draw_plans,
            intent.source_item_index,
            source,
            pipeline_id,
            sort_key.bits(),
            None,
        );
    }

    draw_plans.sort_commands_from(start);
    draw_plans.finish_finalize(plan, start);
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn bake_shadow_intents(
    plan: DrawPlanId,
    intents: &[ShadowRenderIntent],
    draw_plans: &mut DrawPlanStore,
    prepared_draws: &PreparedDrawStore,
    extracted_scene: &ExtractedScene,
    wgpu_ctx: &WgpuContext,
    shadow_global_layout: &wgpu::BindGroupLayout,
    shadow_global_layout_id: u64,
    pipeline_cache: &mut PipelineCache,
    shader_manager: &mut ShaderManager,
) {
    draw_plans.reserve_commands(intents.len());
    let start = draw_plans.begin_finalize(plan);
    let shader_source = ShaderSource::File("entry/utility/depth_prepass");
    let shader_source_identity = shader_manager.source_identity_hash(shader_source);
    for intent in intents {
        let Some(source) = prepared_draws.source(intent.source_item_index) else {
            continue;
        };
        let Some(item) = extracted_scene
            .render_items
            .get(intent.source_item_index as usize)
        else {
            continue;
        };
        if !source.matches_extracted(item, intent.geometry_handle, intent.material_handle) {
            continue;
        }
        if !source.pipeline_state_is_current() {
            continue;
        }
        let geometry = &source.geometry_asset;
        let material = &source.material_asset;
        if material.alpha_mode() == AlphaMode::Blend {
            continue;
        }

        let fast_key = FastShadowPipelineKey {
            material_handle: intent.material_handle,
            material_asset_version: source.material_asset_version,
            material_asset_identity: source.material_asset_identity,
            material_version: source.material_version,
            geometry_handle: intent.geometry_handle,
            geometry_asset_version: source.geometry_asset_version,
            geometry_asset_identity: source.geometry_asset_identity,
            geometry_version: source.geometry_layout_version,
            vertex_layout_id: source.geometry_layout_id,
            instance_variants: item.item_variant_flags,
            item_shader_defines_hash: item.item_shader_defines_hash,
            shader_source_identity,
            shadow_global_layout_id,
            material_layout_id: source.material_layout_id,
            material_binding_wgsl_hash: source.material_binding_wgsl_hash,
            object_layout_id: source.object_layout_id,
            object_binding_wgsl_hash: source.object_binding_wgsl_hash,
            pipeline_settings_version: wgpu_ctx.pipeline_settings_version,
        };
        let (pipeline_id, publish_l1) = if let Some(id) =
            pipeline_cache.get_shadow_pipeline_fast(fast_key)
        {
            (id, false)
        } else {
            let geo_defines = geometry.shader_defines();
            let mat_defines = material.shader_defines();
            let mut options = ShaderCompilationOptions::from_merged(
                &mat_defines,
                geo_defines,
                &myth_resources::shader_defines::ShaderDefines::new(),
                &item.item_shader_defines,
            );
            options.add_define("SHADOW_PASS", "1");
            options.inject_code(
                "vertex_input_code",
                &source.geometry_layout_info.vertex_input_code,
            );
            options.inject_code(
                "binding_code",
                format!(
                    "{}\n{}\n{}",
                    SHADOW_BINDING_WGSL, source.material_binding_wgsl, source.object_binding_wgsl
                ),
            );
            options.inject_code(
                "scene_lighting_structs",
                myth_resources::uniforms::scene_lighting_structs_wgsl(),
            );
            let (shader_module, code_hash) =
                shader_manager.get_or_compile(&wgpu_ctx.device, shader_source, &options);
            let layout = wgpu_ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Shadow Pipeline Layout"),
                    bind_group_layouts: &[
                        Some(shadow_global_layout),
                        Some(&source.material_layout),
                        Some(&source.object_layout),
                    ],
                    immediate_size: 0,
                });
            let vertex_buffers_layout: Vec<_> = source
                .geometry_layout_info
                .buffers
                .iter()
                .map(|layout| layout.as_wgpu())
                .collect();
            let canonical_key = SimpleGeometryPipelineKey {
                shader_hash: code_hash,
                vertex_layout_id: source.geometry_layout_id,
                bind_group_layout_ids: [
                    shadow_global_layout_id,
                    source.material_layout_id,
                    source.object_layout_id,
                ],
                color_targets: smallvec::smallvec![],
                depth_stencil: DepthStencilKey::from(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                topology: geometry.topology,
                cull_mode: Some(wgpu::Face::Front),
                front_face: if item.item_variant_flags & 0x1 != 0 {
                    wgpu::FrontFace::Cw
                } else {
                    wgpu::FrontFace::Ccw
                },
                sample_count: 1,
            };
            let id = pipeline_cache.get_or_create_simple_geometry(
                &wgpu_ctx.device,
                shader_module,
                &layout,
                &canonical_key,
                "Shadow Pipeline",
                &vertex_buffers_layout,
            );
            (id, true)
        };

        let material_bits = source.material_bind_group_id & 0xffff_ffff;
        let sort_key = u64::from(pipeline_id.0) << 32 | material_bits;
        if !source.pipeline_state_is_current() {
            continue;
        }
        if publish_l1 {
            pipeline_cache.insert_shadow_pipeline_fast(fast_key, pipeline_id);
        }
        push_prepared_source_command(
            draw_plans,
            intent.source_item_index,
            source,
            pipeline_id,
            sort_key,
            None,
        );
    }

    draw_plans.sort_commands_from(start);
    draw_plans.finish_finalize(plan, start);
}

/// Append one already-resolved physical source to the currently open plan.
///
/// Prepass reuses this helper after deriving its Feature-owned pipeline.
pub(crate) fn push_prepared_source_command(
    draw_plans: &mut DrawPlanStore,
    source_item_index: u32,
    source: &PreparedDrawSource,
    pipeline_id: RenderPipelineId,
    sort_key: u64,
    stencil_reference: Option<u32>,
) {
    draw_plans.push_command(DrawCommand {
        source_item_index,
        sort_key,
        pipeline_id,
        vertex_buffers: source.vertex_buffers.clone(),
        index_buffer: source.index_buffer,
        bind_group_1: Some(source.material_bind_group),
        bind_group_2: source.object_bind_group,
        stencil_reference,
        vertex_range: source.draw_range.clone(),
        instance_range: source.instance_range.clone(),
    });
}
