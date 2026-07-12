//! Scene Culling and Render Command Generation
//!
//! Standalone functions that perform unified view-based culling, render command
//! generation, and sorting. Extracted from the old `SceneCullPass` as part of
//! the RDG migration. Composer runs this derivation only after successfully
//! acquiring a render target; it is not part of `Renderer::begin_frame`.
//!
//! # Architecture
//!
//! ```text
//! RenderFrame::extract_and_prepare()
//!     │
//!     ├── extract scene data
//!     ├── build_shadow_views()        ← shadow view generation (pure math)
//!     ├── update_shadow_metadata()    ← light storage buffer update
//!     └── prepare_global()            ← global bind group creation
//!
//! FrameComposer::render()
//!     │
//!     ├── acquire target
//!     ├── prepare reachable draw resources
//!     └── culling::cull_and_sort()    ← THIS MODULE
//!     │
//!     ├── prepare_main_camera_commands()  → opaque + transparent lists
//!     └── prepare_shadow_commands()       → per-view shadow command queues
//! ```

use glam::Vec3A;
use log::{error, warn};
use slotmap::Key;

use crate::HDR_TEXTURE_FORMAT;
use crate::RenderPath;
use crate::core::view::ViewTarget;
use crate::core::{ResourceManager, WgpuContext};
use crate::graph::extracted::{ExtractedScene, SceneFeatures};
use crate::graph::frame::{RenderCommand, RenderKey, RenderLists, ShadowRenderCommand};
use crate::graph::render_state::RenderState;
use crate::pipeline::pipeline_key::PipelineFlags;
use crate::pipeline::shader_gen::ShaderCompilationOptions;
use crate::pipeline::shader_manager::ShaderSource;
use crate::pipeline::{
    BlendStateKey, ColorTargetKey, DepthStencilKey, FastPipelineKey, FastShadowPipelineKey,
    GraphicsPipelineKey, PipelineCache, ShaderManager, SimpleGeometryPipelineKey,
};
use myth_assets::AssetServer;
use myth_resources::AntiAliasingMode;
use myth_resources::RenderableMaterialTrait;
use myth_resources::material::{AlphaMode, Side};
use myth_scene::camera::RenderCamera;

/// Shadow-only WGSL binding declaration, injected into shadow depth shaders.
const SHADOW_BINDING_WGSL: &str = "
struct Struct_shadow_light {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u_shadow_light: Struct_shadow_light;
";

/// Top-level entry point: performs culling, command generation, and sorting.
///
/// Call after `extract_and_prepare` has populated the CPU scene snapshot and
/// active views, and after `prepare_draw_resources` has frozen the shared
/// model buffer and materialized each reachable source item.
///
/// # Phases
///
/// 1. **Main camera commands** — frustum cull render items against the main
///    camera, look up / create pipelines, produce sorted opaque + transparent
///    command lists.
/// 2. **Shadow commands** — per-shadow-view frustum cull + shadow pipeline
///    lookup → per-view `ShadowRenderCommand` queues.
///
/// Model uniforms and object bind groups have already been prepared once per
/// source item by `ExtractedScene`; every generated view command reuses them.
#[allow(clippy::too_many_arguments)]
pub fn cull_and_sort(
    extracted_scene: &ExtractedScene,
    render_state: &RenderState,
    wgpu_ctx: &WgpuContext,
    resource_manager: &mut ResourceManager,
    pipeline_cache: &mut PipelineCache,
    shader_manager: &mut ShaderManager,
    render_lists: &mut RenderLists,
    camera: &RenderCamera,
    assets: &AssetServer,
) {
    prepare_main_camera_commands(
        extracted_scene,
        render_state,
        wgpu_ctx,
        resource_manager,
        pipeline_cache,
        shader_manager,
        render_lists,
        camera,
        assets,
    );

    prepare_shadow_commands(
        extracted_scene,
        wgpu_ctx,
        resource_manager,
        pipeline_cache,
        shader_manager,
        render_lists,
        assets,
    );
}

// ============================================================================
// Main Camera Culling + Command Generation
// ============================================================================

/// Cull render items against the main camera frustum and generate sorted
/// opaque / transparent command lists.
///
/// # Performance
///
/// - L1 pipeline cache avoids repeated shader compilation for hot items.
/// - `sort_unstable_by` avoids extra allocation.
/// - Pre-computed `world_aabb` in `ExtractedRenderItem` avoids geometry
///   lookups during culling.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn prepare_main_camera_commands(
    extracted_scene: &ExtractedScene,
    render_state: &RenderState,
    wgpu_ctx: &WgpuContext,
    resource_manager: &mut ResourceManager,
    pipeline_cache: &mut PipelineCache,
    shader_manager: &mut ShaderManager,
    render_lists: &mut RenderLists,
    camera: &RenderCamera,
    assets: &AssetServer,
) {
    let color_format = wgpu_ctx
        .render_path
        .main_color_format(wgpu_ctx.surface_view_format);
    let depth_format = wgpu_ctx.depth_format;
    let sample_count = wgpu_ctx.msaa_samples;
    let render_state_id = render_state.id;
    let scene_id = extracted_scene.scene_id;
    let pipeline_settings_version = wgpu_ctx.pipeline_settings_version;
    let use_clustered_shading = extracted_scene
        .scene_variants
        .contains(SceneFeatures::USE_CLUSTERED_SHADING);
    let taa_enabled = matches!(camera.aa_mode, AntiAliasingMode::TAA { .. });
    let camera_frustum = camera.frustum;
    let camera_pos = camera.position;

    let use_depth_pre = sample_count == 1 && wgpu_ctx.render_path.requires_z_prepass();

    // blackboard.clear();
    {
        let Some(gpu_world) = resource_manager.get_global_state(render_state_id, scene_id) else {
            error!(
                "Render Environment missing for render_state_id {render_state_id}, scene_id {scene_id}"
            );
            return;
        };

        render_lists.gpu_global_bind_group = Some(gpu_world.bind_group.clone());
    }

    let mut has_transmission = false;
    {
        let geo_guard = assets.geometries.read_lock();
        let mat_guard = assets.materials.read_lock();

        for item_idx in 0..extracted_scene.render_items.len() {
            let item = &extracted_scene.render_items[item_idx];

            // ========== Frustum Culling ==========
            let aabb = item.world_aabb;
            if aabb.is_finite() && !camera_frustum.intersects_aabb(&aabb) {
                continue;
            }

            let Some(prepared_item) = extracted_scene.prepared_render_item(item_idx) else {
                continue;
            };

            let Some(gpu_world) = resource_manager.get_global_state(render_state_id, scene_id)
            else {
                error!("CRITICAL: GpuWorld missing during iteration");
                continue;
            };

            let Some(geometry) = geo_guard.get_loaded(item.geometry) else {
                warn!("Geometry {:?} missing during render prepare", item.geometry);
                continue;
            };
            let Some(material) = mat_guard.get_loaded(item.material) else {
                warn!("Material {:?} missing during render prepare", item.material);
                continue;
            };

            let object_bind_group = &prepared_item.object_bind_group;

            let Some(gpu_geometry) = resource_manager.get_geometry(item.geometry) else {
                error!("CRITICAL: GpuGeometry missing for {:?}", item.geometry);
                continue;
            };
            let Some(gpu_material) = resource_manager.get_material(item.material) else {
                error!("CRITICAL: GpuMaterial missing for {:?}", item.material);
                continue;
            };

            let fast_key = FastPipelineKey {
                material_handle: item.material,
                material_version: gpu_material.version,
                geometry_handle: item.geometry,
                geometry_version: geometry.layout_version(),
                instance_variants: item.item_variant_flags,
                global_state_id: gpu_world.id,
                scene_variants: extracted_scene.scene_variants,
                taa_enabled,
                pipeline_settings_version,
                #[cfg(feature = "debug_view")]
                debug_view_mode: camera.debug_view.mode,
            };

            // ========== Hot-Path: L1 cache first ==========
            let pipeline_id = if let Some(id) = pipeline_cache.get_pipeline_fast(fast_key) {
                id
            } else {
                let shader_source = if let Some(template_source) = material.shader_template() {
                    ShaderSource::Inline {
                        name: material.shader_name(),
                        source: template_source,
                        mode: material.shader_template_mode(),
                    }
                } else {
                    ShaderSource::File(material.shader_name())
                };

                let geo_defines = geometry.shader_defines();
                let mat_defines = material.shader_defines();

                let mut flags = PipelineFlags::empty();

                let final_a2c_enable = match material.alpha_mode() {
                    AlphaMode::Mask => material.alpha_to_coverage(), // A2C can be enabled for Masked materials to achieve smoother edges
                    _ => false,
                } && sample_count > 1; // A2C only makes sense with MSAA

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

                let is_opaque_item = !material.is_transparent() && !material.use_transmission();

                if !is_opaque_item {
                    options.add_define("IN_TRANSPARENT_PASS", "1");
                }

                // Shared material-bus determination: inject MRT defines before the shader hash
                // so different attachment layouts produce distinct shader variants.
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
                    blend: if material.is_transparent() {
                        Some(wgpu::BlendState::ALPHA_BLENDING)
                    } else {
                        None
                    },
                    write_mask: wgpu::ColorWrites::ALL,
                });
                let mut color_targets: smallvec::SmallVec<[ColorTargetKey; 3]> =
                    smallvec::smallvec![main_color_target];

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

                let canonical_key = GraphicsPipelineKey {
                    shader_hash,
                    vertex_layout_id: gpu_geometry.layout_id,
                    bind_group_layout_ids: [
                        gpu_world.layout_id,
                        gpu_material.layout_id,
                        object_bind_group.layout_id,
                        if use_clustered_shading {
                            resource_manager
                                .system_textures()
                                .screen_layout_clustered
                                .id()
                        } else {
                            resource_manager.system_textures().screen_layout.id()
                        },
                    ],
                    topology: geometry.topology,
                    cull_mode: match material.side() {
                        Side::Front => Some(wgpu::Face::Back),
                        Side::Back => Some(wgpu::Face::Front),
                        Side::Double => None,
                    },
                    depth_compare,
                    blend_state: if material.is_transparent() {
                        Some(BlendStateKey::from(wgpu::BlendState::ALPHA_BLENDING))
                    } else {
                        None
                    },
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
                    &gpu_geometry.layout_info,
                    gpu_material,
                    object_bind_group,
                    gpu_world,
                    if use_clustered_shading {
                        &resource_manager.system_textures().screen_layout_clustered
                    } else {
                        &resource_manager.system_textures().screen_layout
                    },
                );

                pipeline_cache.insert_pipeline_fast(fast_key, id);
                id
            };

            let mat_id = item.material.data().as_ffi() as u32;

            let use_transmission = material.use_transmission();
            if use_transmission {
                has_transmission = true;
            }

            let is_transparent = material.is_transparent() || use_transmission;

            let item_pos = Vec3A::from(item.world_matrix.w_axis.truncate());
            let distance_sq = camera_pos.distance_squared(item_pos);
            let sort_key = RenderKey::new(pipeline_id, mat_id, distance_sq, is_transparent);

            let cmd = RenderCommand {
                source_item_index: checked_source_item_index(item_idx),
                object_bind_group: object_bind_group.clone(),
                geometry_handle: item.geometry,
                material_handle: item.material,
                pipeline_id,
                sort_key,
                dynamic_offset: prepared_item.dynamic_offset,
            };

            if is_transparent {
                render_lists.insert_transparent(cmd);
            } else {
                render_lists.insert_opaque(cmd);
            }
        }
    }

    render_lists.use_transmission = has_transmission;
    render_lists.sort();
}

#[inline]
fn checked_source_item_index(item_idx: usize) -> u32 {
    u32::try_from(item_idx).expect("one frame cannot extract more than u32::MAX render items")
}

// ============================================================================
// Shadow Culling + Command Generation
// ============================================================================

/// Per-shadow-view frustum culling and shadow render command generation.
///
/// Iterates shadow views already stored in `render_lists.active_views` (built
/// during frame extraction), culls `extracted_scene.render_items` per view,
/// and produces per-view `ShadowRenderCommand` queues in `render_lists.shadow_queues`.
#[allow(clippy::too_many_arguments)]
fn prepare_shadow_commands(
    extracted_scene: &ExtractedScene,
    wgpu_ctx: &WgpuContext,
    resource_manager: &mut ResourceManager,
    pipeline_cache: &mut PipelineCache,
    shader_manager: &mut ShaderManager,
    render_lists: &mut RenderLists,
    assets: &AssetServer,
) {
    let depth_format = wgpu::TextureFormat::Depth32Float;
    let pipeline_settings_version = wgpu_ctx.pipeline_settings_version;

    let shadow_layout_entries = [wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<glam::Mat4>() as u64),
        },
        count: None,
    }];
    let (shadow_global_layout, _) = resource_manager.get_or_create_layout(&shadow_layout_entries);

    // Collect shadow views (indices) from active_views
    let shadow_view_indices: Vec<usize> = render_lists
        .active_views
        .iter()
        .enumerate()
        .filter_map(|(i, v)| if v.is_shadow() { Some(i) } else { None })
        .collect();

    if shadow_view_indices.is_empty() {
        return;
    }

    let geo_guard = assets.geometries.read_lock();
    let mat_guard = assets.materials.read_lock();

    for &view_idx in &shadow_view_indices {
        let view = &render_lists.active_views[view_idx];
        let ViewTarget::ShadowLight {
            light_id,
            layer_index,
        } = view.target
        else {
            continue;
        };

        let view_frustum = view.frustum;

        let queue = render_lists
            .shadow_queues
            .entry((light_id, layer_index))
            .or_default();

        for (item_idx, item) in extracted_scene.render_items.iter().enumerate() {
            if !item.cast_shadows {
                continue;
            }

            let aabb = item.world_aabb;
            if aabb.is_finite() && !view_frustum.intersects_aabb(&aabb) {
                continue;
            }

            let Some(prepared_item) = extracted_scene.prepared_render_item(item_idx) else {
                continue;
            };

            let Some(geometry) = geo_guard.get_loaded(item.geometry) else {
                continue;
            };
            let Some(material) = mat_guard.get_loaded(item.material) else {
                continue;
            };

            if material.alpha_mode() == AlphaMode::Blend {
                continue;
            }

            let Some(gpu_geometry) = resource_manager.get_geometry(item.geometry) else {
                continue;
            };
            let Some(gpu_material) = resource_manager.get_material(item.material) else {
                continue;
            };

            let fast_key = FastShadowPipelineKey {
                material_handle: item.material,
                material_version: gpu_material.version,
                geometry_handle: item.geometry,
                geometry_version: geometry.layout_version(),
                instance_variants: item.item_variant_flags,
                pipeline_settings_version,
            };

            let pipeline_id = if let Some(id) = pipeline_cache.get_shadow_pipeline_fast(fast_key) {
                id
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

                let binding_code = format!(
                    "{}\n{}\n{}",
                    SHADOW_BINDING_WGSL,
                    gpu_material.binding_wgsl,
                    prepared_item.object_bind_group.binding_wgsl
                );

                options.inject_code(
                    "vertex_input_code",
                    &gpu_geometry.layout_info.vertex_input_code,
                );
                options.inject_code("binding_code", binding_code);
                options.inject_code(
                    "scene_lighting_structs",
                    myth_resources::uniforms::scene_lighting_structs_wgsl(),
                );

                let (shader_module, code_hash) = shader_manager.get_or_compile(
                    &wgpu_ctx.device,
                    ShaderSource::File("entry/utility/depth_prepass"),
                    &options,
                );

                let layout =
                    wgpu_ctx
                        .device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("Shadow Pipeline Layout"),
                            bind_group_layouts: &[
                                Some(&shadow_global_layout),
                                Some(&gpu_material.layout),
                                Some(&prepared_item.object_bind_group.layout),
                            ],
                            immediate_size: 0,
                        });

                let vertex_buffers_layout: Vec<_> = gpu_geometry
                    .layout_info
                    .buffers
                    .iter()
                    .map(|l| l.as_wgpu())
                    .collect();

                // For shadow pipelines, we always cull front faces to avoid self-shadowing artifacts.
                // Todo: consider making this configurable per-material or per-item if we encounter cases where back-face shadows are desirable.
                let cull_mode = Some(wgpu::Face::Front);

                let front_face = if item.item_variant_flags & 0x1 != 0 {
                    wgpu::FrontFace::Cw
                } else {
                    wgpu::FrontFace::Ccw
                };

                let canonical_key = SimpleGeometryPipelineKey {
                    shader_hash: code_hash,
                    vertex_layout_id: gpu_geometry.layout_id,
                    color_targets: smallvec::smallvec![],
                    depth_stencil: DepthStencilKey::from(wgpu::DepthStencilState {
                        format: depth_format,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        // Todo: expose depth bias settings in LightShadow for finer control over shadow acne / peter-panning
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    topology: geometry.topology,
                    cull_mode,
                    front_face,
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

                pipeline_cache.insert_shadow_pipeline_fast(fast_key, id);
                id
            };

            queue.push(ShadowRenderCommand {
                source_item_index: checked_source_item_index(item_idx),
                object_bind_group: prepared_item.object_bind_group.clone(),
                geometry_handle: item.geometry,
                material_handle: item.material,
                pipeline_id,
                dynamic_offset: prepared_item.dynamic_offset,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::checked_source_item_index;

    #[test]
    fn shared_draw_assets_keep_distinct_source_item_indices() {
        // Geometry/material identity can be identical for two instances; the
        // extraction position remains the discriminator carried by the draw
        // command into later per-item pipeline derivation.
        let shared_geometry = 17_u64;
        let shared_material = 23_u64;
        let first = (
            shared_geometry,
            shared_material,
            checked_source_item_index(4),
        );
        let second = (
            shared_geometry,
            shared_material,
            checked_source_item_index(9),
        );

        assert_eq!((first.0, first.1), (second.0, second.1));
        assert_ne!(first.2, second.2);
    }
}
