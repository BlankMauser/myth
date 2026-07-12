//! Prepass Feature + Ephemeral PassNode
//!
//! - **`PrepassFeature`** (long-lived): owns its cross-frame pipeline L1.
//! - **`PrepassSnapshot`**: carries only scene identity and output flags.
//! - **`PrepassPassNode`** (ephemeral per-frame): derives missing pipelines
//!   and finalizes its draw plan only after surviving DPE.
//!
//! # RDG Slots (explicit wiring)
//!
//! - `scene_depth`: Scene depth buffer (created & returned by `add_to_graph`)
//! - `scene_normals`: Optional normal buffer (created & returned by `add_to_graph`)
//! - `feature_id`: Optional feature-ID buffer (created & returned by `add_to_graph`)
//! - `velocity_buffer`: Optional velocity buffer for TAA (created & returned by `add_to_graph`)

use rustc_hash::FxHashMap;

use crate::core::gpu::GlobalStateKey;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, PassNode, PrepareContext, RenderTargetOps, TextureDesc, TextureNodeId,
};
use crate::graph::frame::{DrawPlanId, DrawPlanKey, RenderKey};
use crate::graph::passes::draw::submit_draw_commands;
use crate::pipeline::{
    ColorTargetKey, DepthStencilKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
    SimpleGeometryPipelineKey,
};
use myth_assets::{GeometryHandle, MaterialHandle};
use myth_resources::material::Side;

/// Normal texture format — Rgba8Unorm ([-1,1] → [0,1] mapping).
pub(crate) const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Feature ID texture format — Rg8Uint.
pub(crate) const FEATURE_ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Uint;

/// Velocity buffer format — Rg16Float (2-component half-precision motion vectors).
pub(crate) const VELOCITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Lightweight graph-build snapshot for one Prepass variant.
///
/// Physical layouts, generated WGSL, assets, and pipelines are deliberately
/// resolved only by a live [`PrepassPassNode`] during prepare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrepassSnapshot {
    global_state_key: GlobalStateKey,
    needs_normal: bool,
    needs_feature_id: bool,
    needs_velocity: bool,
}

/// Outputs produced by the Prepass, returned to the Composer for
/// explicit downstream wiring.
#[must_use = "SSA Graph: You must use the outputs of prepass to wire downstream passes!"]
pub struct PrepassOutputs {
    /// Single-sample scene depth (written by the prepass).
    pub scene_depth: TextureNodeId,
    /// View-space normals (if screen-space effects require them).
    pub scene_normals: Option<TextureNodeId>,
    /// Feature-ID colour attachment (if SSS/SSR require it).
    pub feature_id: Option<TextureNodeId>,
    /// Screen-space velocity buffer for TAA reprojection (`None` when
    /// TAA is not active).  Format: `Rg16Float`.
    pub velocity_buffer: Option<TextureNodeId>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RenderFeatures)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Long-lived prepass feature.
///
/// The Feature owns only cross-frame derived pipeline IDs. Missing variants
/// are compiled by a live [`PrepassPassNode`] after DPE.
pub struct PrepassFeature {
    /// Feature-local L1 over canonical Prepass inputs. It is intentionally
    /// independent from a main-pass pipeline ID, so a Prepass-only live graph
    /// never has to create an otherwise dead Opaque pipeline.
    local_cache: FxHashMap<PrepassPipelineL1Key, RenderPipelineId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PrepassPipelineL1Key {
    geometry_handle: GeometryHandle,
    geometry_asset_version: u32,
    geometry_asset_identity: u64,
    geometry_layout_version: u64,
    material_handle: MaterialHandle,
    material_asset_version: u32,
    material_asset_identity: u64,
    material_version: u64,
    item_variants: u32,
    item_shader_defines_hash: u64,
    shader_source_identity: u64,
    scene_defines_hash: u64,
    pipeline_settings_version: u64,
    global_binding_wgsl_hash: u64,
    material_binding_wgsl_hash: u64,
    object_binding_wgsl_hash: u64,
    vertex_layout_id: u64,
    bind_group_layout_ids: [u64; 3],
    depth_format: wgpu::TextureFormat,
    topology: wgpu::PrimitiveTopology,
    cull_mode: Option<wgpu::Face>,
    front_face: wgpu::FrontFace,
    needs_normal: bool,
    needs_feature_id: bool,
    needs_velocity: bool,
}

impl Default for PrepassFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl PrepassFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            local_cache: FxHashMap::default(),
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn cached_pipeline_count(&self) -> usize {
        self.local_cache.len()
    }

    /// Capture the scene/output identity without touching physical resources.
    #[must_use]
    pub const fn extract(
        &self,
        global_state_key: GlobalStateKey,
        needs_normal: bool,
        needs_feature_id: bool,
        needs_velocity: bool,
    ) -> PrepassSnapshot {
        PrepassSnapshot {
            global_state_key,
            needs_normal,
            needs_feature_id,
            needs_velocity,
        }
    }

    /// Derive missing Prepass variants and bake the live plan directly from
    /// CPU intents plus graph-attempt source records.
    #[allow(clippy::too_many_lines)]
    fn prepare_draw_plan(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        snapshot: PrepassSnapshot,
        plan: DrawPlanId,
    ) {
        if ctx.scene_draws.draw_plans.resolve_plan(plan).is_some() {
            return;
        }

        let start = ctx.scene_draws.draw_plans.begin_finalize(plan);
        let Some(gpu_world) = ctx.global_states.get(snapshot.global_state_key) else {
            ctx.scene_draws.draw_plans.finish_finalize(plan, start);
            return;
        };
        let global_layout_id = gpu_world.layout_id;
        let global_binding_wgsl_hash = gpu_world.binding_wgsl_hash;
        let depth_format = ctx.scene_draws.depth_format;
        let scene_defines_hash = ctx.scene_draws.extracted_scene.scene_defines.compute_hash();
        let pipeline_settings_version = ctx.scene_draws.pipeline_settings_version;
        let shader_source = ShaderSource::File("entry/utility/depth_prepass");
        let shader_source_identity = ctx.shader_manager.source_identity_hash(shader_source);

        for intent in &ctx.scene_draws.render_lists.opaque {
            let Some(source) = ctx
                .scene_draws
                .prepared_draws
                .source(intent.source_item_index)
            else {
                continue;
            };
            let Some(extracted_item) = ctx
                .scene_draws
                .extracted_scene
                .render_items
                .get(intent.source_item_index as usize)
            else {
                continue;
            };
            if !source.matches_extracted(
                extracted_item,
                intent.geometry_handle,
                intent.material_handle,
            ) {
                continue;
            }
            if !source.pipeline_state_is_current() {
                continue;
            }
            let geometry = &source.geometry_asset;
            let material = &source.material_asset;
            if material.is_transparent() || material.use_transmission() {
                continue;
            }

            let cull_mode = match material.side() {
                Side::Front => Some(wgpu::Face::Back),
                Side::Back => Some(wgpu::Face::Front),
                Side::Double => None,
            };
            let front_face = if extracted_item.item_variant_flags & 0x1 != 0 {
                wgpu::FrontFace::Cw
            } else {
                wgpu::FrontFace::Ccw
            };
            let l1_key = PrepassPipelineL1Key {
                geometry_handle: intent.geometry_handle,
                geometry_asset_version: source.geometry_asset_version,
                geometry_asset_identity: source.geometry_asset_identity,
                geometry_layout_version: source.geometry_layout_version,
                material_handle: intent.material_handle,
                material_asset_version: source.material_asset_version,
                material_asset_identity: source.material_asset_identity,
                material_version: source.material_version,
                item_variants: extracted_item.item_variant_flags,
                item_shader_defines_hash: extracted_item.item_shader_defines_hash,
                shader_source_identity,
                scene_defines_hash,
                pipeline_settings_version,
                global_binding_wgsl_hash,
                material_binding_wgsl_hash: source.material_binding_wgsl_hash,
                object_binding_wgsl_hash: source.object_binding_wgsl_hash,
                vertex_layout_id: source.geometry_layout_id,
                bind_group_layout_ids: [
                    global_layout_id,
                    source.material_layout_id,
                    source.object_layout_id,
                ],
                depth_format,
                topology: geometry.topology,
                cull_mode,
                front_face,
                needs_normal: snapshot.needs_normal,
                needs_feature_id: snapshot.needs_feature_id,
                needs_velocity: snapshot.needs_velocity,
            };

            let (pipeline_id, publish_l1) =
                if let Some(&pipeline_id) = self.local_cache.get(&l1_key) {
                    (pipeline_id, false)
                } else {
                    let geo_defines = geometry.shader_defines();
                    let mat_defines = material.shader_defines();
                    let mut options = ShaderCompilationOptions::from_merged(
                        &mat_defines,
                        geo_defines,
                        &ctx.scene_draws.extracted_scene.scene_defines,
                        &extracted_item.item_shader_defines,
                    );
                    options.add_define("IS_PREPASS", "1");
                    if snapshot.needs_normal {
                        options.add_define("OUTPUT_NORMAL", "1");
                    }
                    if snapshot.needs_velocity {
                        options.add_define("HAS_VELOCITY_TARGET", "1");
                    }
                    if snapshot.needs_feature_id {
                        options.add_define("OUTPUT_FEATURE_ID", "1");
                    }
                    options.inject_code(
                        "vertex_input_code",
                        &source.geometry_layout_info.vertex_input_code,
                    );
                    options.inject_code(
                        "binding_code",
                        format!(
                            "{}\n{}\n{}",
                            gpu_world.binding_wgsl,
                            source.material_binding_wgsl,
                            source.object_binding_wgsl
                        ),
                    );
                    options.inject_code(
                        "scene_lighting_structs",
                        myth_resources::uniforms::scene_lighting_structs_wgsl(),
                    );
                    let (shader_module, shader_hash) =
                        ctx.shader_manager
                            .get_or_compile(ctx.device, shader_source, &options);
                    let layout =
                        ctx.device
                            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some("Prepass Pipeline Layout"),
                                bind_group_layouts: &[
                                    Some(&gpu_world.layout),
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
                    let mut color_targets = smallvec::smallvec![];
                    if snapshot.needs_normal {
                        color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                            format: NORMAL_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }));
                    }
                    if snapshot.needs_feature_id {
                        color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                            format: FEATURE_ID_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }));
                    }
                    if snapshot.needs_velocity {
                        color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                            format: VELOCITY_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }));
                    }
                    let canonical_key = SimpleGeometryPipelineKey {
                        shader_hash,
                        vertex_layout_id: source.geometry_layout_id,
                        bind_group_layout_ids: [
                            global_layout_id,
                            source.material_layout_id,
                            source.object_layout_id,
                        ],
                        color_targets,
                        depth_stencil: DepthStencilKey::from(wgpu::DepthStencilState {
                            format: depth_format,
                            depth_write_enabled: Some(true),
                            depth_compare: Some(wgpu::CompareFunction::Greater),
                            stencil: wgpu::StencilState::default(),
                            bias: wgpu::DepthBiasState::default(),
                        }),
                        topology: geometry.topology,
                        cull_mode,
                        front_face,
                        sample_count: 1,
                    };
                    let pipeline_id = ctx.pipeline_cache.get_or_create_simple_geometry(
                        ctx.device,
                        shader_module,
                        &layout,
                        &canonical_key,
                        "Prepass Pipeline",
                        &vertex_buffers_layout,
                    );
                    (pipeline_id, true)
                };

            if !source.pipeline_state_is_current() {
                continue;
            }
            if publish_l1 {
                self.local_cache.insert(l1_key, pipeline_id);
            }

            let sort_key = RenderKey::new(
                pipeline_id,
                intent.material_index,
                intent.distance_sq,
                false,
            );
            crate::graph::bake::push_prepared_source_command(
                ctx.scene_draws.draw_plans,
                intent.source_item_index,
                source,
                pipeline_id,
                sort_key.bits(),
                None,
            );
        }

        ctx.scene_draws.draw_plans.sort_commands_from(start);
        ctx.scene_draws.draw_plans.finish_finalize(plan, start);
    }

    /// Build the ephemeral pass node, register resources, and insert it
    /// into the graph.
    ///
    /// All shared resources (`Scene_Depth`, `Scene_Normals`, `Feature_ID`)
    /// are created here so the Composer can wire them to downstream passes
    /// via explicit [`TextureNodeId`] connections.
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: PrepassSnapshot,
    ) -> PrepassOutputs {
        let fc = ctx.frame_config;

        // Single-sample scene depth (always created).
        // COPY_SRC is required so TAA can archive scene depth to history.
        let depth_desc = TextureDesc::new(
            fc.width,
            fc.height,
            1,
            1,
            1,
            wgpu::TextureDimension::D2,
            fc.depth_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let scene_depth = ctx.graph.register_texture("Scene_Depth", depth_desc, false);

        let scene_normals = if snapshot.needs_normal {
            let desc = TextureDesc::new_2d(
                fc.width,
                fc.height,
                NORMAL_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            Some(ctx.graph.register_texture("Scene_Normals", desc, false))
        } else {
            None
        };

        let feature_id = if snapshot.needs_feature_id {
            let desc = TextureDesc::new_2d(
                fc.width,
                fc.height,
                FEATURE_ID_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            Some(ctx.graph.register_texture("Feature_ID", desc, false))
        } else {
            None
        };

        let velocity_buffer = if snapshot.needs_velocity {
            let desc = TextureDesc::new_2d(
                fc.width,
                fc.height,
                VELOCITY_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            Some(ctx.graph.register_texture("Velocity_Buffer", desc, false))
        } else {
            None
        };
        let draw_plan = ctx.graph.reserve_draw_plan(DrawPlanKey::Prepass {
            needs_normal: snapshot.needs_normal,
            needs_feature_id: snapshot.needs_feature_id,
            needs_velocity: snapshot.needs_velocity,
        });

        let node = PrepassPassNode {
            feature: self,
            snapshot,
            scene_depth,
            scene_normals: scene_normals.unwrap_or(TextureNodeId::from_index(0)),
            feature_id: feature_id.unwrap_or(TextureNodeId::from_index(0)),
            velocity_buffer: velocity_buffer.unwrap_or(TextureNodeId::from_index(0)),
            draw_plan,
        };
        ctx.graph.add_pass("Pre_Pass", |builder| {
            builder.demand_draw_plan(draw_plan);
            builder.write_texture(scene_depth);
            if let Some(n) = scene_normals {
                builder.write_texture(n);
            }
            if let Some(f) = feature_id {
                builder.write_texture(f);
            }
            if let Some(v) = velocity_buffer {
                builder.write_texture(v);
            }
            (node, ())
        });

        PrepassOutputs {
            scene_depth,
            scene_normals,
            feature_id,
            velocity_buffer,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral per-frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral per-frame prepass node.
///
/// Carries a lightweight snapshot and RDG IDs. A live node lazily derives any
/// missing pipeline variants and finalizes its own graph-attempt draw plan.
pub struct PrepassPassNode<'a> {
    feature: &'a mut PrepassFeature,
    snapshot: PrepassSnapshot,

    // ─── RDG Resource Slots (filled in setup) ──────────────────────
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    feature_id: TextureNodeId,
    velocity_buffer: TextureNodeId,

    draw_plan: DrawPlanId,
}

impl<'a> PassNode<'a> for PrepassPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature
            .prepare_draw_plan(ctx, self.snapshot, self.draw_plan);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let gpu_global_bind_group = ctx.global_bind_group;

        let mut color_attachments: smallvec::SmallVec<
            [Option<wgpu::RenderPassColorAttachment>; 3],
        > = smallvec::SmallVec::with_capacity(3);

        let normal_clear = wgpu::Color {
            r: 0.5,
            g: 0.5,
            b: 1.0,
            a: 0.0,
        };
        if self.snapshot.needs_normal
            && let Some(att) = ctx.get_color_attachment(
                self.scene_normals,
                RenderTargetOps::Clear(normal_clear),
                None,
            )
        {
            color_attachments.push(Some(att));
        }
        if self.snapshot.needs_feature_id
            && let Some(att) = ctx.get_color_attachment(
                self.feature_id,
                RenderTargetOps::Clear(wgpu::Color::TRANSPARENT),
                None,
            )
        {
            color_attachments.push(Some(att));
        }
        if self.snapshot.needs_velocity
            && let Some(att) = ctx.get_color_attachment(
                self.velocity_buffer,
                RenderTargetOps::Clear(wgpu::Color::TRANSPARENT),
                None,
            )
        {
            color_attachments.push(Some(att));
        }

        let dtt = ctx.get_depth_stencil_attachment(self.scene_depth, 0.0);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Depth-Normal-Velocity Prepass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: dtt,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_bind_group(0, gpu_global_bind_group, &[]);

        submit_draw_commands(
            &mut pass,
            ctx.draw_plans,
            self.draw_plan,
            ctx.pipeline_cache,
        );
    }
}
