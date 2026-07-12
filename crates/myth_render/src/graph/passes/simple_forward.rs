//! RDG Simple Forward Render Pass
//!
//! Single-pass LDR rendering for the [`BasicForward`] path. Combines opaque,
//! skybox, and transparent drawing into one `wgpu::RenderPass` with optional
//! hardware MSAA.
//!
//! # RDG Slots (explicit wiring)
//!
//! - `surface_out`: LDR colour output (input, from Composer)
//! - `scene_depth`: Depth buffer (created internally)
//!
//! # Push Parameters
//!
//! - `clear_color`: Background clear colour (from scene settings)
//!
//! # Rendering Order
//!
//! 1. **Clear** colour and depth
//! 2. **Opaque** objects (front-to-back)
//! 3. **Skybox** (drawn behind opaque geometry via Reverse-Z)
//! 4. **Transparent** objects (back-to-front)
//!
//! [`BasicForward`]: crate::settings::RenderPath::BasicForward

use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::GlobalStateKey;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ClusteredScreenBindings, ExecuteContext, PassNode, PrepareContext, RenderTargetOps,
    TextureDesc, TextureNodeId, build_screen_bind_group,
};
use crate::graph::frame::{DrawPlanId, DrawPlanKey};
use crate::graph::passes::draw::submit_draw_commands;
use crate::graph::passes::skybox::{SkyboxDrawRequest, SkyboxDrawState};

// ─── Feature ───────────────────────────────────────────────────────────

pub struct SimpleForwardFeature;

impl Default for SimpleForwardFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleForwardFeature {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)] // Keep the uniform Feature::add_to_graph API.
    pub(crate) fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        global_state_key: GlobalStateKey,
        // surface_out: TextureNodeId,
        clear_color: wgpu::Color,
        skybox: Option<SkyboxDrawRequest<'a>>,
        shadow_tex: Option<TextureNodeId>,
        shadow_cube_tex: Option<TextureNodeId>,
        // env_map_tex: Option<TextureNodeId>,
        pmrem_tex: Option<TextureNodeId>,
        scene_lighting: ClusteredScreenBindings,
    ) -> TextureNodeId {
        let fc = ctx.frame_config;

        // let depth_desc = TextureDesc::new(
        //     fc.width,
        //     fc.height,
        //     1,
        //     1,
        //     fc.msaa_samples,
        //     wgpu::TextureDimension::D2,
        //     fc.depth_format,
        //     wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        // );

        let out_desc = fc.create_surface_desc(wgpu::TextureUsages::TEXTURE_BINDING);
        let depth_desc = fc.create_depth_desc(wgpu::TextureUsages::TEXTURE_BINDING);
        let opaque_draw_plan = ctx.graph.reserve_draw_plan(DrawPlanKey::Opaque);
        let transparent_draw_plan = ctx.graph.reserve_draw_plan(DrawPlanKey::Transparent);

        ctx.graph.add_pass("SimpleForward_Pass", |builder| {
            builder.demand_global_state(global_state_key);
            builder.demand_draw_plan(opaque_draw_plan);
            builder.demand_draw_plan(transparent_draw_plan);
            let out = builder.create_texture("Scene_Color", out_desc);
            let scene_depth = builder.create_texture("Scene_Depth", depth_desc);

            if let Some(shadow) = shadow_tex {
                builder.read_texture(shadow);
            }
            if let Some(shadow_cube) = shadow_cube_tex {
                builder.read_texture(shadow_cube);
            }

            if let Some(pmrem) = pmrem_tex {
                builder.read_texture(pmrem);
            }
            if let Some(light_metadata) = scene_lighting.light_metadata {
                builder.read_buffer(light_metadata);
            }
            if let Some(lights) = scene_lighting.lights {
                builder.read_buffer(lights);
            }
            if let Some(params) = scene_lighting.params {
                builder.read_buffer(params);
            }
            if let Some(records) = scene_lighting.records {
                builder.read_buffer(records);
            }
            if let Some(indices) = scene_lighting.light_indices {
                builder.read_buffer(indices);
            }
            if let Some(transmittance) = scene_lighting.atmosphere_transmittance {
                builder.read_texture(transmittance);
            }
            if let Some(bake_params) = scene_lighting.atmosphere_bake_params {
                builder.read_buffer(bake_params);
            }
            let skybox = skybox.map(|request| request.build(builder));

            let msaa_view = if fc.msaa_samples > 1 {
                let desc = TextureDesc::new(
                    fc.width,
                    fc.height,
                    1,
                    1,
                    fc.msaa_samples,
                    wgpu::TextureDimension::D2,
                    fc.surface_format,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                );
                Some(builder.create_texture("Scene_Msaa", desc))
            } else {
                None
            };

            let node = SimpleForwardPassNode {
                surface_out: out,
                scene_depth,
                msaa_view,
                clear_color,
                skybox,
                shadow_input: shadow_tex,
                shadow_cube_input: shadow_cube_tex,
                scene_lighting,
                opaque_draw_plan,
                transparent_draw_plan,
                screen_bind_group: None,
            };
            (node, out)
        })
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

/// RDG Simple Forward Render Pass.
///
/// Draws the entire scene in a single LDR render pass with optional MSAA,
/// intended for the [`BasicForward`] rendering path. The pass writes
/// directly to the swap-chain surface via `surface_out`.
///
/// [`BasicForward`]: crate::settings::RenderPath::BasicForward
pub struct SimpleForwardPassNode<'a> {
    pub surface_out: TextureNodeId,
    pub scene_depth: TextureNodeId,
    pub msaa_view: Option<TextureNodeId>,
    pub clear_color: wgpu::Color,
    skybox: Option<SkyboxDrawState<'a>>,
    pub shadow_input: Option<TextureNodeId>,
    pub shadow_cube_input: Option<TextureNodeId>,
    pub scene_lighting: ClusteredScreenBindings,
    opaque_draw_plan: DrawPlanId,
    transparent_draw_plan: DrawPlanId,
    screen_bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for SimpleForwardPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let bg = build_screen_bind_group(
            ctx,
            None,
            None,
            self.shadow_input,
            self.shadow_cube_input,
            self.scene_lighting,
        );
        self.screen_bind_group = Some(bg);
        if let Some(skybox) = &mut self.skybox {
            skybox.prepare(ctx);
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let gpu_global_bind_group = ctx.global_bind_group.expect(
            "SimpleForward_Pass scene Global Group 0 is missing; its demand was not declared or finalized",
        );

        let (color_view, resolve_target) = if let Some(msaa_view) = self.msaa_view {
            (msaa_view, Some(self.surface_out))
        } else {
            (self.surface_out, None)
        };

        let depth_att = ctx.get_depth_stencil_attachment(self.scene_depth, 0.0);
        let color_att = ctx.get_color_attachment(
            color_view,
            RenderTargetOps::Clear(self.clear_color),
            resolve_target,
        );

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("RDG Simple Forward Pass"),
            color_attachments: &[color_att],
            depth_stencil_attachment: depth_att,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let raw_pass = encoder.begin_render_pass(&pass_desc);
        let mut pass = raw_pass;

        pass.set_bind_group(0, gpu_global_bind_group, &[]);

        let screen_bg = ctx.resolve_prepared_bind_group(self.screen_bind_group.unwrap());
        pass.set_bind_group(3, screen_bg, &[]);

        // 1. Opaque (front-to-back)
        submit_draw_commands(
            &mut pass,
            ctx.draw_plans,
            self.opaque_draw_plan,
            ctx.pipeline_cache,
        );

        // 2. Skybox (between opaque and transparent)
        if let Some(skybox) = &self.skybox {
            skybox.draw(&mut pass, ctx);

            pass.set_bind_group(0, gpu_global_bind_group, &[]);
            pass.set_bind_group(3, screen_bg, &[]);
        }

        // 3. Transparent (back-to-front)
        submit_draw_commands(
            &mut pass,
            ctx.draw_plans,
            self.transparent_draw_plan,
            ctx.pipeline_cache,
        );
    }
}
