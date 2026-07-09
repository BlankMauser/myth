//! FXAA Feature + Ephemeral PassNode
//!
//! - **`FxaaFeature`** (long-lived): owns pipeline cache and bind group layout.
//!   `extract()` returns the quality/format snapshot; pipeline creation is
//!   deferred until the pass survives graph compilation.
//! - **`FxaaPassNode`** (ephemeral per-frame): carries lightweight IDs and
//!   a transient bind-group slot.  Created by `FxaaFeature::add_to_graph()`.

use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, PassNode, PrepareContext, RenderTargetOps, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::FxaaQuality;
use wgpu::CommandEncoder;

type FxaaL1CacheKey = (FxaaQuality, wgpu::TextureFormat);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RenderFeatures)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone, Copy)]
pub struct FxaaSnapshot {
    quality: FxaaQuality,
    output_format: wgpu::TextureFormat,
}

pub struct FxaaFeature {
    // ─── Persistent Cache ──────────────────────────────────────────
    l1_cache_key: Option<FxaaL1CacheKey>,
    pipeline_id: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl Default for FxaaFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl FxaaFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            l1_cache_key: None,
            pipeline_id: None,
            bind_group_layout: None,
        }
    }

    #[must_use]
    pub const fn extract(
        &self,
        quality: FxaaQuality,
        output_format: wgpu::TextureFormat,
    ) -> FxaaSnapshot {
        FxaaSnapshot {
            quality,
            output_format,
        }
    }

    /// Materialize persistent resources after the node survives DPE.
    fn prepare_resources(&mut self, ctx: &mut PrepareContext<'_>, snapshot: FxaaSnapshot) {
        // ── 1. Lazy-create BindGroupLayout (once) ──────────────────
        if self.bind_group_layout.is_none() {
            let layout = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("FXAA BindGroup Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
            self.bind_group_layout = Some(Tracked::new(layout));
        }

        // ── 2. L1 Cache: compile pipeline on quality/format change ─
        let current_key = (snapshot.quality, snapshot.output_format);

        if self.l1_cache_key != Some(current_key) {
            let mut options = ShaderCompilationOptions::default();
            if snapshot.quality != FxaaQuality::Medium {
                options.add_define(snapshot.quality.define_key(), "1");
            }

            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/post_process/fxaa"),
                &options,
            );

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: snapshot.output_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key = FullscreenPipelineKey::fullscreen(
                shader_hash,
                smallvec::smallvec![color_target],
                None,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("FXAA Pipeline Layout"),
                        bind_group_layouts: &[self.bind_group_layout.as_deref()],
                        immediate_size: 0,
                    });

            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                shader_module,
                &pipeline_layout,
                &key,
                &format!("FXAA Pipeline {:?}", snapshot.quality),
            );
            self.pipeline_id = Some(id);
            self.l1_cache_key = Some(current_key);
        }
    }

    /// Build the ephemeral pass node and insert it into the graph.
    ///
    /// Accepts the LDR input and the target surface, performs an SSA relay
    /// on `target_surface` (via `write_texture`), and returns the
    /// updated surface handle. This enforces a pure dataflow chain where
    /// every Feature explicitly produces a new resource version.
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: FxaaSnapshot,
        input_ldr: TextureNodeId,
        // target_surface: TextureNodeId,
    ) -> TextureNodeId {
        ctx.graph.add_pass("FXAA_Pass", |builder| {
            builder.read_texture(input_ldr);
            // let output = builder.write_texture(target_surface);

            let out_desc = ctx
                .frame_config
                .create_surface_desc(wgpu::TextureUsages::TEXTURE_BINDING);
            let output = builder.create_texture("LDR_Antialiased", out_desc);

            let node = FxaaPassNode {
                feature: self,
                snapshot,
                input_tex: input_ldr,
                output_tex: output,
                transient_bg: None,
            };
            (node, output)
        })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct FxaaPassNode<'a> {
    feature: &'a mut FxaaFeature,
    snapshot: FxaaSnapshot,
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for FxaaPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.prepare_resources(ctx, self.snapshot);
        let layout = self
            .feature
            .bind_group_layout
            .as_ref()
            .expect("FXAA layout must exist after prepare");
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, layout, Some("FXAA BindGroup"), [
                0 => self.input_tex,
                1 => CommonSampler::LinearClamp,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut CommandEncoder) {
        let bind_group = self.transient_bg.expect("FXAA BG not prepared!");
        let pipeline_id = self
            .feature
            .pipeline_id
            .expect("FXAA pipeline must exist after prepare");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("FXAA Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(ctx.pipeline_cache.get_render_pipeline(pipeline_id));
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
