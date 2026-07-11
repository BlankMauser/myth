//! MSAA Sync Feature + Ephemeral PassNode
//!
//! Solves the **stale multi-sample context** problem that occurs when
//! single-sample screen-space effects (SSSS, etc.) modify the resolved
//! `Scene_Color_HDR` after the initial MSAA Opaque resolve.  Without this
//! pass, subsequent multi-sampled draws (Skybox, Transparent) would bind
//! an outdated `Scene_Color_MSAA` surface, and the final resolve would
//! overwrite the screen-space results.
//!
//! # Strategy
//!
//! A single fullscreen blit copies the **correct** single-sample HDR data
//! back into every sample of the multi-sampled color target.  The depth
//! buffer is intentionally **not bound**, preserving the pixel-accurate
//! MSAA geometry depth written by the Opaque pass for downstream
//! depth-tested draws.
//!
//! # Data Flow
//!
//! ```text
//!  Scene_Color_HDR ──(read)──> MsaaSyncPassNode ──(write)──> Scene_Color_MSAA
//! ```
//!
//! # Conditional Insertion
//!
//! The Composer inserts this pass **only** when all three conditions hold:
//! 1. `HighFidelity` render path.
//! 2. MSAA is active (`msaa_samples > 1`).
//! 3. A single-sample pass (e.g. SSSS) has modified `Scene_Color_HDR`
//!    after the Opaque resolve.

use crate::HDR_TEXTURE_FORMAT;
use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, PassNode, PrepareContext, RenderTargetOps, TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, MultisampleKey, RenderPipelineId,
    ShaderCompilationOptions, ShaderSource,
};
use wgpu::CommandEncoder;

/// L1 cache key: pipeline depends on destination MSAA sample count.
type MsaaSyncCacheKey = u32;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone, Copy)]
pub struct MsaaSyncSnapshot {
    msaa_samples: u32,
}

/// Long-lived MSAA sync feature — owns the blit pipeline and bind group
/// layout.  Produces an ephemeral [`MsaaSyncPassNode`] each frame via
/// [`Self::add_to_graph`].
pub struct MsaaSyncFeature {
    l1_cache_key: Option<MsaaSyncCacheKey>,
    pipeline_id: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl Default for MsaaSyncFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl MsaaSyncFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            l1_cache_key: None,
            pipeline_id: None,
            bind_group_layout: None,
        }
    }

    #[must_use]
    pub fn extract(&self, msaa_samples: u32) -> MsaaSyncSnapshot {
        debug_assert!(msaa_samples > 1, "MsaaSyncFeature requires MSAA");
        MsaaSyncSnapshot { msaa_samples }
    }

    /// Materialize the MSAA-aware blit pipeline after the node survives DPE.
    fn prepare_resources(&mut self, ctx: &mut PrepareContext<'_>, snapshot: MsaaSyncSnapshot) {
        let msaa_samples = snapshot.msaa_samples;

        let device = ctx.device;

        // ── 1. Lazy-create BindGroupLayout (once) ──────────────────
        if self.bind_group_layout.is_none() {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("MSAA Sync BindGroup Layout"),
                entries: &[
                    // binding 0: single-sample HDR source texture
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

        // ── 2. L1 Cache: recompile pipeline when sample count changes ──
        let current_key = msaa_samples;
        if self.l1_cache_key != Some(current_key) {
            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/utility/blit.wgsl"),
                &ShaderCompilationOptions::default(),
            );

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MSAA Sync Pipeline Layout"),
                bind_group_layouts: &[self.bind_group_layout.as_deref()],
                immediate_size: 0,
            });

            let key = FullscreenPipelineKey {
                shader_hash,
                compilation_hash: 0,
                color_targets: smallvec::smallvec![ColorTargetKey::from(wgpu::ColorTargetState {
                    format: HDR_TEXTURE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                depth_stencil: None,
                multisample: MultisampleKey::from(wgpu::MultisampleState {
                    count: msaa_samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                }),
                topology: wgpu::PrimitiveTopology::TriangleList,
            };

            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                shader_module,
                &pipeline_layout,
                &key,
                "MSAA Sync Pipeline",
            );
            self.pipeline_id = Some(id);
            self.l1_cache_key = Some(current_key);
        }
    }

    /// Build the ephemeral pass node and insert it into the graph.
    ///
    /// - `src_hdr`: single-sample `Scene_Color_HDR` (read).
    /// - `dst_msaa`: multi-sampled `Scene_Color_MSAA` (write).
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: MsaaSyncSnapshot,
        src_hdr: TextureNodeId,
    ) -> TextureNodeId {
        let msaa_color_desc = TextureDesc::new(
            ctx.frame_config.width,
            ctx.frame_config.height,
            1,
            1,
            snapshot.msaa_samples,
            wgpu::TextureDimension::D2,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );

        let dst_msaa = ctx
            .graph
            .register_texture("Scene_Color_MSAA", msaa_color_desc, false);

        let node = MsaaSyncPassNode {
            feature: self,
            snapshot,
            src_hdr,
            dst_msaa,
            transient_bg: None,
        };
        ctx.graph.add_pass("Msaa_Sync_Pass", |builder| {
            builder.read_texture(src_hdr);
            builder.write_texture(dst_msaa);
            (node, ())
        });
        dst_msaa
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct MsaaSyncPassNode<'a> {
    feature: &'a mut MsaaSyncFeature,
    snapshot: MsaaSyncSnapshot,
    src_hdr: TextureNodeId,
    dst_msaa: TextureNodeId,
    transient_bg: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for MsaaSyncPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.prepare_resources(ctx, self.snapshot);
        let layout = self
            .feature
            .bind_group_layout
            .as_ref()
            .expect("MSAA sync layout must exist after prepare");
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, layout, Some("MSAA Sync BindGroup"), [
                0 => self.src_hdr,
                1 => CommonSampler::LinearClamp,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut CommandEncoder) {
        let bind_group = self.transient_bg.expect("MSAA Sync BG not prepared");
        let bind_group = ctx.resolve_prepared_bind_group(bind_group);
        let pipeline_id = self
            .feature
            .pipeline_id
            .expect("MSAA sync pipeline must exist after prepare");

        let rtt = ctx.get_color_attachment(self.dst_msaa, RenderTargetOps::DontCare, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("MSAA Sync Pass"),
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
