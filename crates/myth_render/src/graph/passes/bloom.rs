//! RDG Bloom Post-Processing — Flattened Micro-Pass Architecture.
//!
//! Implements the Call of Duty: Advanced Warfare physically-based bloom
//! technique within the RDG framework using fully flattened per-mip passes
//! and a strict two-layer binding separation:
//!
//! - **Group 0 (Parameters)**: Feature-owned sampler plus, where needed, an
//!   RDG transient uniform buffer. Karis is a pipeline override, not a buffer.
//! - **Group 1 (Transient)**: RDG texture views — built by each micro-pass
//!   node during `prepare`, cached via [`GlobalBindGroupCache`].
//!
//! # Architecture
//!
//! - **`BloomFeature`** – persistent owner of lazily-created GPU pipelines,
//!   bind-group layouts, and the downsample sampler bind group. The first live
//!   node of each micro-pass family initializes its shared state after DPE.
//!
//! - **`BloomSnapshot`** – lightweight copied settings captured while the
//!   Composer constructs the graph. It contains no WGPU objects or handles.
//!
//! - **Micro-Passes** – each downsample, upsample, and composite step is
//!   an independent [`PassNode`] with its own RDG resource declarations.
//!   This exposes fine-grained dependencies to the RDG compiler, enabling
//!   optimal barrier insertion and memory aliasing across mip levels.
//!
//! # RDG Topology (Flattened)
//!
//! ```text
//! Bloom_Extract (Scene HDR → Mip_0)
//!   → Bloom_Downsample_1 (Mip_0 → Mip_1)
//!   → Bloom_Downsample_2 (Mip_1 → Mip_2)
//!   → ...
//!   → Bloom_Upsample_N-2 (Mip_N-1 + Mip_N-2 → Up_N-2)
//!   → ...
//!   → Bloom_Upsample_0 (Up_1 + Mip_0 → Up_0)
//!   → Bloom_Composite (Scene HDR + Up_0 → Output)
//! ```

use std::mem::size_of;
use std::sync::OnceLock;

use crate::HDR_TEXTURE_FORMAT;
use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::binding_layout::{sampler, texture_2d, uniform_buffer};
use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, PassNode, PrepareContext, RenderTargetOps,
    TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::bloom::{BloomSettings, CompositeUniforms, UpsampleUniforms};
use myth_resources::uniforms::WgslStruct;

// =============================================================================
// BloomFeature — persistent GPU resource owner
// =============================================================================

const MAX_BLOOM_MIPS: usize = 16;

/// Lightweight per-frame bloom data captured during graph construction.
#[derive(Clone, Copy)]
pub struct BloomSnapshot {
    upsample: UpsampleUniforms,
    composite: CompositeUniforms,
    karis_average: bool,
    max_mip_levels: u32,
}

/// Persistent bloom feature. Static resources are partitioned by micro-pass
/// family so absent upsample work does not initialize an unused pipeline.
pub struct BloomFeature {
    downsample: OnceLock<BloomDownsampleResources>,
    upsample: OnceLock<BloomPipelineResources>,
    composite: OnceLock<BloomPipelineResources>,
}

struct BloomDownsampleResources {
    karis_on_pipeline_id: RenderPipelineId,
    karis_off_pipeline_id: RenderPipelineId,
    textures_layout: Tracked<wgpu::BindGroupLayout>,
    params_bind_group: wgpu::BindGroup,
}

struct BloomPipelineResources {
    pipeline_id: RenderPipelineId,
    params_layout: Tracked<wgpu::BindGroupLayout>,
    textures_layout: Tracked<wgpu::BindGroupLayout>,
}

impl Default for BloomFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl BloomFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            downsample: OnceLock::new(),
            upsample: OnceLock::new(),
            composite: OnceLock::new(),
        }
    }

    /// Capture CPU-only settings. No GPU upload, shader compilation, layout
    /// creation, or bind-group construction occurs here.
    #[must_use]
    pub fn extract(&self, settings: &BloomSettings) -> BloomSnapshot {
        BloomSnapshot {
            upsample: *settings.upsample_uniforms.read(),
            composite: *settings.composite_uniforms.read(),
            karis_average: settings.karis_average,
            max_mip_levels: settings.max_mip_levels(),
        }
    }

    fn ensure_downsample_resources(
        &self,
        ctx: &mut PrepareContext<'_>,
    ) -> &BloomDownsampleResources {
        self.downsample
            .get_or_init(|| BloomDownsampleResources::create(ctx))
    }

    fn downsample_resources(&self) -> &BloomDownsampleResources {
        self.downsample
            .get()
            .expect("Bloom downsample resources must be initialized by Bloom_Extract")
    }

    fn ensure_upsample_resources(&self, ctx: &mut PrepareContext<'_>) -> &BloomPipelineResources {
        self.upsample.get_or_init(|| create_upsample_resources(ctx))
    }

    fn upsample_resources(&self) -> &BloomPipelineResources {
        self.upsample
            .get()
            .expect("Bloom upsample resources must be initialized by the coarsest upsample node")
    }

    fn ensure_composite_resources(&self, ctx: &mut PrepareContext<'_>) -> &BloomPipelineResources {
        self.composite
            .get_or_init(|| create_composite_resources(ctx))
    }

    /// Emit the full bloom pipeline as a chain of independent RDG passes.
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: BloomSnapshot,
        input_color: TextureNodeId,
    ) -> TextureNodeId {
        let fc = ctx.frame_config;
        debug_assert_eq!(fc.hdr_format, HDR_TEXTURE_FORMAT);
        let hdr_format = HDR_TEXTURE_FORMAT;
        let bloom_w = (fc.width / 2).max(1);
        let bloom_h = (fc.height / 2).max(1);
        let mip_count = bloom_mip_count(bloom_w, bloom_h, snapshot.max_mip_levels);

        ctx.with_group("Bloom_System", |ctx| {
            // ─── 1. Extract: Scene HDR → Bloom Mip 0 ──────────────
            let mip0_desc = TextureDesc::new_2d(
                bloom_w,
                bloom_h,
                hdr_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let mut current_mip: TextureNodeId = ctx.graph.add_pass("Bloom_Extract", |builder| {
                builder.read_texture(input_color);
                let out = builder.create_texture("Bloom_Mip_0", mip0_desc);
                let node = BloomDownsampleNode {
                    feature: self,
                    initialize_resources: true,
                    karis_average: snapshot.karis_average,
                    input_tex: input_color,
                    output_tex: out,
                    pipeline_id: None,
                    transient_bg: None,
                };
                (node, out)
            });

            // Collect downsample chain for upsample phase references.
            let mut downsample_chain = Vec::with_capacity(mip_count);
            downsample_chain.push(current_mip);

            // ─── 2. Downsample chain: Mip N → Mip N+1 ─────────────
            let mut w = bloom_w;
            let mut h = bloom_h;
            for i in 1..mip_count {
                w = (w / 2).max(1);
                h = (h / 2).max(1);

                let desc = TextureDesc::new_2d(
                    w,
                    h,
                    hdr_format,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                );

                let input = current_mip;
                let label: &'static str = DOWNSAMPLE_PASS_NAMES[i];
                let out_label: &'static str = DOWNSAMPLE_MIP_NAMES[i];

                current_mip = ctx.graph.add_pass(label, |builder| {
                    builder.read_texture(input);
                    let out = builder.create_texture(out_label, desc);
                    let node = BloomDownsampleNode {
                        feature: self,
                        initialize_resources: false,
                        karis_average: false,
                        input_tex: input,
                        output_tex: out,
                        pipeline_id: None,
                        transient_bg: None,
                    };
                    (node, out)
                });

                downsample_chain.push(current_mip);
            }

            // ─── 3. Upsample chain: coarse → fine with additive blend ─
            let mut upsample_params_buffer = None;
            for i in (0..(mip_count - 1)).rev() {
                let coarser = current_mip;
                let finer = downsample_chain[i];

                let label: &'static str = UPSAMPLE_PASS_NAMES[i];
                let out_label: &'static str = UPSAMPLE_MIP_NAMES[i];

                current_mip = ctx.graph.add_pass(label, |builder| {
                    builder.read_texture(coarser);
                    let (params_buffer, upload_params) =
                        if let Some(buffer) = upsample_params_buffer {
                            (builder.read_buffer(buffer), false)
                        } else {
                            let buffer = create_uniform_buffer::<UpsampleUniforms>(
                                builder,
                                "Bloom_Upsample_Params",
                            );
                            upsample_params_buffer = Some(buffer);
                            (buffer, true)
                        };
                    let out = builder.mutate_texture(finer, out_label);
                    let node = BloomUpsampleNode {
                        feature: self,
                        initialize_resources: upload_params,
                        uniforms: snapshot.upsample,
                        params_buffer,
                        upload_params,
                        coarser_tex: coarser,
                        output_tex: out,
                        pipeline_id: None,
                        params_bg: None,
                        textures_bg: None,
                    };
                    (node, out)
                });
            }

            // ─── 4. Composite: Scene HDR + Bloom → Output ─────────
            let bloom_result = current_mip;
            ctx.graph.add_pass("Bloom_Composite", |builder| {
                builder.read_texture(input_color);
                builder.read_texture(bloom_result);

                let out_desc = TextureDesc::new_2d(
                    fc.width,
                    fc.height,
                    hdr_format,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                );

                let out = builder.create_texture("Scene_Color_Bloom", out_desc);
                let params_buffer =
                    create_uniform_buffer::<CompositeUniforms>(builder, "Bloom_Composite_Params");

                let node = BloomCompositeNode {
                    feature: self,
                    uniforms: snapshot.composite,
                    params_buffer,
                    original_tex: input_color,
                    bloom_tex: bloom_result,
                    output_tex: out,
                    pipeline_id: None,
                    params_bg: None,
                    textures_bg: None,
                };
                (node, out)
            })
        })
    }
}

fn bloom_mip_count(width: u32, height: u32, requested: u32) -> usize {
    let max_possible = width.max(height).max(1).ilog2() + 1;
    requested
        .min(max_possible)
        .min(MAX_BLOOM_MIPS as u32)
        .max(1) as usize
}

fn create_upsample_resources(ctx: &mut PrepareContext<'_>) -> BloomPipelineResources {
    let mut options = ShaderCompilationOptions::default();
    options.add_define(
        "struct_definitions",
        UpsampleUniforms::wgsl_struct_def("UpsampleUniforms").as_str(),
    );
    prepare_uniform_pipeline(
        ctx,
        "entry/post_process/bloom/upsample",
        "Bloom US Pipeline",
        "Bloom US Pipeline Layout",
        "Bloom US Params Layout (G0)",
        "Bloom US Textures Layout (G1)",
        1,
        &options,
        additive_blend(),
    )
}

fn create_composite_resources(ctx: &mut PrepareContext<'_>) -> BloomPipelineResources {
    let mut options = ShaderCompilationOptions::default();
    options.add_define(
        "struct_definitions",
        CompositeUniforms::wgsl_struct_def("CompositeUniforms").as_str(),
    );
    prepare_uniform_pipeline(
        ctx,
        "entry/post_process/bloom/composite",
        "Bloom Comp Pipeline",
        "Bloom Comp Pipeline Layout",
        "Bloom Comp Params Layout (G0)",
        "Bloom Comp Textures Layout (G1)",
        2,
        &options,
        wgpu::BlendState::REPLACE,
    )
}

impl BloomDownsampleResources {
    fn create(ctx: &mut PrepareContext<'_>) -> Self {
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/post_process/bloom/downsample"),
            &ShaderCompilationOptions::default(),
        );
        let target = color_target(wgpu::BlendState::REPLACE);

        let karis_off_constants = [("USE_KARIS_AVERAGE", 0.0)];
        let karis_off_options = wgpu::PipelineCompilationOptions {
            constants: &karis_off_constants,
            ..Default::default()
        };
        let karis_off_key = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![target.clone()],
            None,
        )
        .with_compilation_options(&karis_off_options);

        let karis_on_constants = [("USE_KARIS_AVERAGE", 1.0)];
        let karis_on_options = wgpu::PipelineCompilationOptions {
            constants: &karis_on_constants,
            ..Default::default()
        };
        let karis_on_key =
            FullscreenPipelineKey::fullscreen(shader_hash, smallvec::smallvec![target], None)
                .with_compilation_options(&karis_on_options);

        let cached_off = ctx
            .pipeline_cache
            .get_fullscreen_pipeline_id(&karis_off_key);
        let cached_on = ctx.pipeline_cache.get_fullscreen_pipeline_id(&karis_on_key);

        let (params_layout, textures_layout) = if let Some(id) = cached_off.or(cached_on) {
            (
                ctx.pipeline_cache.get_tracked_layout(id, 0).clone(),
                ctx.pipeline_cache.get_tracked_layout(id, 1).clone(),
            )
        } else {
            (
                create_sampler_layout(ctx.device, "Bloom DS Params Layout (G0)"),
                create_textures_layout(ctx.device, "Bloom DS Textures Layout (G1)", 1),
            )
        };

        let pipeline_layout = (cached_off.is_none() || cached_on.is_none()).then(|| {
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Bloom DS Pipeline Layout"),
                    bind_group_layouts: &[Some(&params_layout), Some(&textures_layout)],
                    immediate_size: 0,
                })
        });

        let karis_off_pipeline_id = cached_off.unwrap_or_else(|| {
            let id = ctx.pipeline_cache.get_or_create_fullscreen_with_options(
                ctx.device,
                module,
                pipeline_layout
                    .as_ref()
                    .expect("missing Bloom DS pipeline layout"),
                &karis_off_key,
                &karis_off_options,
                "Bloom DS Pipeline (Karis Off)",
            );
            ctx.pipeline_cache.register_render_layouts(
                id,
                smallvec::smallvec![params_layout.clone(), textures_layout.clone()],
            );
            id
        });

        let karis_on_pipeline_id = cached_on.unwrap_or_else(|| {
            let id = ctx.pipeline_cache.get_or_create_fullscreen_with_options(
                ctx.device,
                module,
                pipeline_layout
                    .as_ref()
                    .expect("missing Bloom DS pipeline layout"),
                &karis_on_key,
                &karis_on_options,
                "Bloom DS Pipeline (Karis On)",
            );
            ctx.pipeline_cache.register_render_layouts(
                id,
                smallvec::smallvec![params_layout.clone(), textures_layout.clone()],
            );
            id
        });

        let params_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom DS G0"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(
                    ctx.sampler_registry.get_common(CommonSampler::LinearClamp),
                ),
            }],
        });

        Self {
            karis_on_pipeline_id,
            karis_off_pipeline_id,
            textures_layout,
            params_bind_group,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_uniform_pipeline(
    ctx: &mut PrepareContext<'_>,
    shader_path: &'static str,
    pipeline_label: &'static str,
    pipeline_layout_label: &'static str,
    params_layout_label: &'static str,
    textures_layout_label: &'static str,
    texture_count: usize,
    options: &ShaderCompilationOptions,
    blend: wgpu::BlendState,
) -> BloomPipelineResources {
    let (module, shader_hash) =
        ctx.shader_manager
            .get_or_compile(ctx.device, ShaderSource::File(shader_path), options);
    let key = FullscreenPipelineKey::fullscreen(
        shader_hash,
        smallvec::smallvec![color_target(blend)],
        None,
    );

    if let Some(pipeline_id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
        return BloomPipelineResources {
            pipeline_id,
            params_layout: ctx
                .pipeline_cache
                .get_tracked_layout(pipeline_id, 0)
                .clone(),
            textures_layout: ctx
                .pipeline_cache
                .get_tracked_layout(pipeline_id, 1)
                .clone(),
        };
    }

    let params_layout = create_sampler_uniform_layout(ctx.device, params_layout_label);
    let textures_layout = create_textures_layout(ctx.device, textures_layout_label, texture_count);
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(pipeline_layout_label),
            bind_group_layouts: &[Some(&params_layout), Some(&textures_layout)],
            immediate_size: 0,
        });
    let pipeline_id = ctx.pipeline_cache.get_or_create_fullscreen(
        ctx.device,
        module,
        &pipeline_layout,
        &key,
        pipeline_label,
    );
    ctx.pipeline_cache.register_render_layouts(
        pipeline_id,
        smallvec::smallvec![params_layout.clone(), textures_layout.clone()],
    );

    BloomPipelineResources {
        pipeline_id,
        params_layout,
        textures_layout,
    }
}

fn color_target(blend: wgpu::BlendState) -> ColorTargetKey {
    ColorTargetKey::from(wgpu::ColorTargetState {
        format: HDR_TEXTURE_FORMAT,
        blend: Some(blend),
        write_mask: wgpu::ColorWrites::ALL,
    })
}

fn additive_blend() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent::OVER,
    }
}

fn create_sampler_layout(
    device: &wgpu::Device,
    label: &'static str,
) -> Tracked<wgpu::BindGroupLayout> {
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[sampler(
                0,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            )],
        }),
    )
}

fn create_sampler_uniform_layout(
    device: &wgpu::Device,
    label: &'static str,
) -> Tracked<wgpu::BindGroupLayout> {
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                sampler(
                    0,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                ),
                uniform_buffer(1, wgpu::ShaderStages::FRAGMENT),
            ],
        }),
    )
}

fn create_textures_layout(
    device: &wgpu::Device,
    label: &'static str,
    texture_count: usize,
) -> Tracked<wgpu::BindGroupLayout> {
    let entries = [
        texture_2d(
            0,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::TextureSampleType::Float { filterable: true },
        ),
        texture_2d(
            1,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::TextureSampleType::Float { filterable: true },
        ),
    ];
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries[..texture_count],
        }),
    )
}

fn create_uniform_buffer<T>(
    builder: &mut crate::graph::core::PassBuilder<'_, '_>,
    name: &'static str,
) -> BufferNodeId {
    builder.create_buffer(
        name,
        BufferDesc::new(
            size_of::<T>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        ),
    )
}

// =============================================================================
// Pre-allocated static pass/resource name tables
// =============================================================================
//
// RDG pass and resource names must be `&'static str`. We pre-allocate up to
// 16 mip levels (covers 65536×65536 max resolution) as compile-time constants
// to avoid runtime string formatting and heap allocation.

/// Pass names for each downsample step (index 0 is the extract step).
static DOWNSAMPLE_PASS_NAMES: [&str; MAX_BLOOM_MIPS] = [
    "Bloom_Extract",
    "Bloom_Downsample_1",
    "Bloom_Downsample_2",
    "Bloom_Downsample_3",
    "Bloom_Downsample_4",
    "Bloom_Downsample_5",
    "Bloom_Downsample_6",
    "Bloom_Downsample_7",
    "Bloom_Downsample_8",
    "Bloom_Downsample_9",
    "Bloom_Downsample_10",
    "Bloom_Downsample_11",
    "Bloom_Downsample_12",
    "Bloom_Downsample_13",
    "Bloom_Downsample_14",
    "Bloom_Downsample_15",
];

/// Resource names for each downsample mip output.
static DOWNSAMPLE_MIP_NAMES: [&str; MAX_BLOOM_MIPS] = [
    "Bloom_Mip_0",
    "Bloom_Mip_1",
    "Bloom_Mip_2",
    "Bloom_Mip_3",
    "Bloom_Mip_4",
    "Bloom_Mip_5",
    "Bloom_Mip_6",
    "Bloom_Mip_7",
    "Bloom_Mip_8",
    "Bloom_Mip_9",
    "Bloom_Mip_10",
    "Bloom_Mip_11",
    "Bloom_Mip_12",
    "Bloom_Mip_13",
    "Bloom_Mip_14",
    "Bloom_Mip_15",
];

/// Pass names for each upsample step.
static UPSAMPLE_PASS_NAMES: [&str; MAX_BLOOM_MIPS] = [
    "Bloom_Upsample_0",
    "Bloom_Upsample_1",
    "Bloom_Upsample_2",
    "Bloom_Upsample_3",
    "Bloom_Upsample_4",
    "Bloom_Upsample_5",
    "Bloom_Upsample_6",
    "Bloom_Upsample_7",
    "Bloom_Upsample_8",
    "Bloom_Upsample_9",
    "Bloom_Upsample_10",
    "Bloom_Upsample_11",
    "Bloom_Upsample_12",
    "Bloom_Upsample_13",
    "Bloom_Upsample_14",
    "Bloom_Upsample_15",
];

/// Resource names for each upsample output.
static UPSAMPLE_MIP_NAMES: [&str; MAX_BLOOM_MIPS] = [
    "Bloom_Up_0",
    "Bloom_Up_1",
    "Bloom_Up_2",
    "Bloom_Up_3",
    "Bloom_Up_4",
    "Bloom_Up_5",
    "Bloom_Up_6",
    "Bloom_Up_7",
    "Bloom_Up_8",
    "Bloom_Up_9",
    "Bloom_Up_10",
    "Bloom_Up_11",
    "Bloom_Up_12",
    "Bloom_Up_13",
    "Bloom_Up_14",
    "Bloom_Up_15",
];

// =============================================================================
// Micro-Pass Nodes
// =============================================================================

/// Downsample micro-pass: reads one input texture, writes one output texture
/// at half resolution using a 13-tap box filter.
///
/// Used for both the initial extract (scene HDR → mip 0) and subsequent
/// mip-chain downsamples. Karis averaging is controlled by the selected
/// override-specialized pipeline.
struct BloomDownsampleNode<'a> {
    feature: &'a BloomFeature,
    initialize_resources: bool,
    karis_average: bool,
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    transient_bg: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for BloomDownsampleNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let resources = if self.initialize_resources {
            self.feature.ensure_downsample_resources(ctx)
        } else {
            self.feature.downsample_resources()
        };
        self.pipeline_id = Some(if self.karis_average {
            resources.karis_on_pipeline_id
        } else {
            resources.karis_off_pipeline_id
        });
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, &resources.textures_layout, Some("Bloom DS G1"), [
                0 => self.input_tex,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let transient_bg = self
            .transient_bg
            .expect("Bloom DS transient BG not prepared");
        let transient_bg = ctx.resolve_prepared_bind_group(transient_bg);

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Downsample"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(
            ctx.pipeline_cache
                .get_render_pipeline(self.pipeline_id.expect("Bloom DS pipeline not prepared")),
        );
        rpass.set_bind_group(
            0,
            &self.feature.downsample_resources().params_bind_group,
            &[],
        );
        rpass.set_bind_group(1, transient_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

struct BloomUpsampleNode<'a> {
    feature: &'a BloomFeature,
    initialize_resources: bool,
    uniforms: UpsampleUniforms,
    params_buffer: BufferNodeId,
    upload_params: bool,
    coarser_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    params_bg: Option<PreparedBindGroupId>,
    textures_bg: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for BloomUpsampleNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let resources = if self.initialize_resources {
            self.feature.ensure_upsample_resources(ctx)
        } else {
            self.feature.upsample_resources()
        };
        self.pipeline_id = Some(resources.pipeline_id);
        if self.upload_params {
            ctx.queue.write_buffer(
                ctx.views.get_buffer(self.params_buffer),
                0,
                bytemuck::bytes_of(&self.uniforms),
            );
        }
        self.params_bg = Some(
            crate::myth_bind_group!(ctx, &resources.params_layout, Some("Bloom US G0"), [
                0 => CommonSampler::LinearClamp,
                1 => self.params_buffer,
            ]),
        );
        self.textures_bg = Some(
            crate::myth_bind_group!(ctx, &resources.textures_layout, Some("Bloom US G1"), [
                0 => self.coarser_tex,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let params_bg = self.params_bg.expect("Bloom US G0 not prepared");
        let textures_bg = self.textures_bg.expect("Bloom US G1 not prepared");
        let params_bg = ctx.resolve_prepared_bind_group(params_bg);
        let textures_bg = ctx.resolve_prepared_bind_group(textures_bg);

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::Load, None);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Upsample"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(
            ctx.pipeline_cache
                .get_render_pipeline(self.pipeline_id.expect("Bloom US pipeline not prepared")),
        );
        rpass.set_bind_group(0, params_bg, &[]);
        rpass.set_bind_group(1, textures_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

struct BloomCompositeNode<'a> {
    feature: &'a BloomFeature,
    uniforms: CompositeUniforms,
    params_buffer: BufferNodeId,
    original_tex: TextureNodeId,
    bloom_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    params_bg: Option<PreparedBindGroupId>,
    textures_bg: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for BloomCompositeNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let resources = self.feature.ensure_composite_resources(ctx);
        self.pipeline_id = Some(resources.pipeline_id);
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.params_buffer),
            0,
            bytemuck::bytes_of(&self.uniforms),
        );
        self.params_bg = Some(
            crate::myth_bind_group!(ctx, &resources.params_layout, Some("Bloom Comp G0"), [
                0 => CommonSampler::LinearClamp,
                1 => self.params_buffer,
            ]),
        );
        self.textures_bg = Some(
            crate::myth_bind_group!(ctx, &resources.textures_layout, Some("Bloom Comp G1"), [
                0 => self.original_tex,
                1 => self.bloom_tex,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let params_bg = self.params_bg.expect("Bloom Comp G0 not prepared");
        let textures_bg = self.textures_bg.expect("Bloom Comp G1 not prepared");
        let params_bg = ctx.resolve_prepared_bind_group(params_bg);
        let textures_bg = ctx.resolve_prepared_bind_group(textures_bg);

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Composite"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(
            ctx.pipeline_cache.get_render_pipeline(
                self.pipeline_id
                    .expect("Bloom composite pipeline not prepared"),
            ),
        );
        rpass.set_bind_group(0, params_bg, &[]);
        rpass.set_bind_group(1, textures_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_BLOOM_MIPS, bloom_mip_count};

    #[test]
    fn bloom_mip_count_respects_size_and_requested_budget() {
        assert_eq!(bloom_mip_count(960, 540, 6), 6);
        assert_eq!(bloom_mip_count(1, 1, 16), 1);
        assert_eq!(bloom_mip_count(0, 0, 0), 1);
        assert_eq!(
            bloom_mip_count(u32::MAX, u32::MAX, u32::MAX),
            MAX_BLOOM_MIPS
        );
    }
}
