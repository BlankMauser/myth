//! Present Feature + Ephemeral PassNode
//!
//! Provides the single, unified entry point through which any pipeline
//! configuration delivers its final image to the swap-chain surface (or the
//! headless render target).  Every post-process stage renders into a transient
//! LDR target; the Composer then appends one `PresentPass` that forwards that
//! target onto the external surface.
//!
//! # Why a dedicated present stage?
//!
//! Centralising surface ownership decouples the Composer from the question of
//! *which* pass happens to run last (ToneMapping, FXAA, DebugView, …).  Passes
//! no longer need to know whether they are the final stage — they always write
//! an intermediate, and `PresentPass` is the sole writer of the surface.
//!
//! # Zero runtime cost
//!
//! The pass is flagged [`PassBuilder::mark_pure_forwarding`].  Because the
//! intermediate source always shares the surface's format and size, the graph
//! compiler folds the pass away during
//! [`RenderGraph::fold_simple_passes`](crate::graph::core::RenderGraph) —
//! rewiring the upstream producer to target the surface directly.  The blit
//! pipeline below therefore acts purely as a robust fallback for the rare case
//! where folding is impossible (e.g. a format mismatch).

use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, RenderTargetOps, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use wgpu::CommandEncoder;

/// L1 cache key: the present pipeline depends only on the surface format.
type PresentL1CacheKey = wgpu::TextureFormat;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RendererState)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Long-lived present feature — owns the fullscreen blit pipeline and bind
/// group layout, recompiling only when the surface format changes.
pub struct PresentFeature {
    l1_cache_key: Option<PresentL1CacheKey>,
    pipeline_id: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl Default for PresentFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl PresentFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            l1_cache_key: None,
            pipeline_id: None,
            bind_group_layout: None,
        }
    }

    /// Pre-RDG resource preparation: create the bind group layout and compile
    /// the blit pipeline for the active surface format.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        surface_format: wgpu::TextureFormat,
    ) {
        // ── 1. Lazy-create BindGroupLayout (once) ──────────────────
        if self.bind_group_layout.is_none() {
            let layout = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Present BindGroup Layout"),
                    entries: &[
                        // binding 0: LDR source texture
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
                        // binding 1: sampler
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

        // ── 2. L1 Cache: recompile pipeline on surface-format change ──
        if self.l1_cache_key != Some(surface_format) {
            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/utility/blit.wgsl"),
                &ShaderCompilationOptions::default(),
            );

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: surface_format,
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
                        label: Some("Present Pipeline Layout"),
                        bind_group_layouts: &[self.bind_group_layout.as_deref()],
                        immediate_size: 0,
                    });

            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                shader_module,
                &pipeline_layout,
                &key,
                "Present Pipeline",
            );
            self.pipeline_id = Some(id);
            self.l1_cache_key = Some(surface_format);
        }
    }

    /// Append the unified present pass.
    ///
    /// Reads `source` (the final transient LDR target) and writes a versioned
    /// alias of `target_surface` (the external swap-chain / headless view).  The
    /// pass is marked pure-forwarding so the graph compiler can collapse it into
    /// the upstream producer.  Returns the surface-alias handle for downstream
    /// wiring (e.g. UI overlay hooks), which remains valid whether or not the
    /// pass is folded away.
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        source: TextureNodeId,
        target_surface: TextureNodeId,
    ) -> TextureNodeId {
        let pipeline_id = self.pipeline_id.expect("PresentFeature not prepared");
        let layout = self.bind_group_layout.as_ref().unwrap();

        ctx.graph.add_pass("Present_Pass", |builder| {
            builder.read_texture(source);
            // Write a versioned alias of the surface rather than the surface
            // itself.  This gives the graph compiler a distinct "latest
            // version" it can reroute the upstream producer onto when folding
            // the pass away, while a non-folded blit still resolves the alias
            // to the real swap-chain view.
            let output = builder.replace_texture(target_surface, "Surface_Present");
            builder.mark_pure_forwarding();

            let node = PresentPassNode {
                source_tex: source,
                output_tex: output,
                pipeline_id,
                layout,
                transient_bg: None,
            };
            (node, output)
        })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral present pass node — zero-drop POD.
///
/// Only reached when compile-time folding could not eliminate the pass; in the
/// folded fast path neither `prepare` nor `execute` runs.
struct PresentPassNode<'a> {
    source_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline_id: RenderPipelineId,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for PresentPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, self.layout, Some("Present BindGroup"), [
                0 => self.source_tex,
                1 => CommonSampler::LinearClamp,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut CommandEncoder) {
        let bind_group = self.transient_bg.expect("Present BG not prepared!");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Present Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(ctx.pipeline_cache.get_render_pipeline(self.pipeline_id));
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
