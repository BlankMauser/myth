//! Prepass Feature + Ephemeral PassNode
//!
//! - **`PrepassFeature`** (long-lived): owns its cross-frame pipeline L1.
//! - **`PrepassSnapshot`**: carries only scene identity and output flags.
//! - **`PrepassPassNode`** (ephemeral per-frame): gathers exact CPU recipes
//!   after surviving DPE; the scene coordinator derives pipelines and commands.
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
use crate::graph::core::scene_submission::{SceneSubmissionKey, SceneSubmissionPlanId};
use crate::graph::core::{
    ExecuteContext, PassNode, PrepareContext, RenderTargetOps, TextureDesc, TextureNodeId,
};
use crate::graph::passes::draw::submit_scene_submission;
use crate::graph::scene_pipeline_prepare::PrepassPipelineL1Key;
use crate::pipeline::{RenderPipelineId, ShaderSource};
use myth_resources::material::{AlphaMode, Side};

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
/// The Feature owns only cross-frame derived pipeline IDs. A live
/// [`PrepassPassNode`] gathers exact recipes after DPE; the scene coordinator
/// compiles missing variants and publishes accepted IDs back into this L1.
pub struct PrepassFeature {
    /// Feature-local L1 over canonical Prepass inputs. It is intentionally
    /// independent from a main-pass pipeline ID, so a Prepass-only live graph
    /// never has to create an otherwise dead Opaque pipeline.
    local_cache: FxHashMap<PrepassPipelineL1Key, RenderPipelineId>,
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

    /// Gather exact Prepass recipes from CPU intents and graph-attempt source
    /// records. This phase performs no physical resource creation or L1 write.
    fn gather_submission(
        &self,
        ctx: &mut PrepareContext<'_>,
        snapshot: PrepassSnapshot,
        submission: SceneSubmissionPlanId,
    ) {
        let render_lists = ctx.scene_draws.render_lists;
        let extracted_scene = ctx.scene_draws.extracted_scene;
        let prepared_draws = ctx.scene_draws.prepared_draws;
        let geometry_frames = ctx.scene_draws.geometry_frames;
        let material_frames = ctx.scene_draws.material_frames;
        let object_interfaces = ctx.scene_draws.object_interfaces;
        let depth_format = ctx.scene_draws.depth_format;
        let pipeline_settings_version = ctx.scene_draws.pipeline_settings_version;

        let Some(gpu_world) = ctx.global_states.get(snapshot.global_state_key) else {
            return;
        };
        let global_layout_id = gpu_world.layout_id;
        let global_binding_interface_id = gpu_world.binding_interface_id;
        let scene_defines_hash = extracted_scene.scene_defines.compute_hash();
        let shader_source = ShaderSource::File("entry/utility/depth_prepass");
        let shader_source_identity = ctx.shader_manager.source_identity_hash(shader_source);

        for (intent_index, intent) in render_lists.opaque.iter().enumerate() {
            let Some(prepared_source) = prepared_draws.prepared_source_id(intent.source) else {
                continue;
            };
            let Some(source) = prepared_draws.source(prepared_source) else {
                continue;
            };
            let Some(extracted_source) = render_lists.extracted_source(intent.source) else {
                continue;
            };
            let Some(extracted_item) = extracted_scene.render_items.get(extracted_source.index())
            else {
                continue;
            };
            let Some(geometry) = geometry_frames.geometry(source.geometry) else {
                continue;
            };
            let Some(geometry_physical) = geometry_frames.physical(source.geometry) else {
                continue;
            };
            let Some(material) = material_frames.material(source.material) else {
                continue;
            };
            let Some(object_interface) = object_interfaces.interface(source.object_interface)
            else {
                continue;
            };
            if (extracted_item.geometry, extracted_item.material)
                != (intent.geometry_handle, intent.material_handle)
                || geometry.key.handle != intent.geometry_handle
                || material.handle() != intent.material_handle
                || (geometry.key.asset_version, geometry.key.asset_identity)
                    != (
                        extracted_item.geometry_asset_version,
                        extracted_item.geometry_asset_identity,
                    )
                || (material.asset_version(), material.asset_identity())
                    != (
                        extracted_item.material_asset_version,
                        extracted_item.material_asset_identity,
                    )
                || !geometry_frames.pipeline_state_is_current(source.geometry)
                || !material.pipeline_state_is_current()
            {
                continue;
            }
            let material_pipeline = material.pipeline();
            let material_settings = material_pipeline.settings();
            let material_interface = material.interface();
            if matches!(
                material_settings.alpha_mode,
                AlphaMode::Blend | AlphaMode::BlendMask
            ) || material_pipeline.use_transmission()
            {
                continue;
            }

            let cull_mode = match material_settings.side {
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
                geometry_asset_version: geometry.key.asset_version,
                geometry_asset_identity: geometry.key.asset_identity,
                geometry_layout_version: geometry.key.layout_version,
                material_handle: intent.material_handle,
                material_asset_version: material.asset_version(),
                material_asset_identity: material.asset_identity(),
                material_version: material.material_version(),
                item_variants: extracted_item.item_variant_flags,
                item_shader_defines_hash: extracted_item.item_shader_defines_hash,
                shader_source_identity,
                scene_defines_hash,
                pipeline_settings_version,
                global_binding_interface_id,
                material_binding_interface_id: material_interface.binding_interface_id,
                object_binding_interface_id: object_interface.binding_interface_id,
                vertex_layout_id: geometry_physical.layout_id,
                bind_group_layout_ids: [
                    global_layout_id,
                    material_interface.layout_id,
                    object_interface.layout_id,
                ],
                depth_format,
                topology: geometry.topology,
                cull_mode,
                front_face,
                needs_normal: snapshot.needs_normal,
                needs_feature_id: snapshot.needs_feature_id,
                needs_velocity: snapshot.needs_velocity,
            };

            if !geometry_frames.pipeline_state_is_current(source.geometry)
                || !material.pipeline_state_is_current()
            {
                continue;
            }
            let cached = self.local_cache.get(&l1_key).copied();
            ctx.scene_draws.push_prepass_recipe(
                submission,
                intent_index,
                intent.source,
                prepared_source,
                l1_key,
                cached,
                intent.material_index,
                intent.distance_sq,
            );
        }
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
        let scene_submission = ctx
            .graph
            .reserve_scene_submission(SceneSubmissionKey::Prepass {
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
            scene_submission,
        };
        ctx.graph.add_pass("Pre_Pass", |builder| {
            builder.demand_global_state(snapshot.global_state_key);
            builder.demand_scene_submission(scene_submission);
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
/// Carries a lightweight snapshot and RDG IDs. A live node gathers exact
/// recipes, the scene coordinator resolves them, and `finalize_prepare`
/// publishes the accepted pipeline IDs before immutable submission executes.
pub struct PrepassPassNode<'a> {
    feature: &'a mut PrepassFeature,
    snapshot: PrepassSnapshot,

    // ─── RDG Resource Slots (filled in setup) ──────────────────────
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    feature_id: TextureNodeId,
    velocity_buffer: TextureNodeId,

    scene_submission: SceneSubmissionPlanId,
}

impl<'a> PassNode<'a> for PrepassPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature
            .gather_submission(ctx, self.snapshot, self.scene_submission);
    }

    fn finalize_prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        ctx.scene_draws
            .publish_prepass_pipeline_l1(&mut self.feature.local_cache);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let gpu_global_bind_group = ctx.global_bind_group.expect(
            "Pre_Pass scene Global Group 0 is missing; its demand was not declared or finalized",
        );

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

        submit_scene_submission(
            &mut pass,
            ctx.scene_submissions,
            self.scene_submission,
            ctx.pipeline_cache,
        );
    }
}
