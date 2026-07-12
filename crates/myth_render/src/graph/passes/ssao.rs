//! SSAO Feature + Ephemeral PassNodes (Flattened)
//!
//! - **`SsaoFeature`** (long-lived): owns physically partitioned raw/blur
//!   pipeline state and tracked layouts.
//! - **`SsaoSnapshot`** (per-frame): contains only copied uniforms and a
//!   lightweight global-state key.
//! - **`SsaoRawNode`** / **`SsaoBlurNode`** (ephemeral): ensure persistent
//!   state only after surviving DPE, then build transient bind groups.
//!
//! The two RDG nodes form a raw SSAO → cross-bilateral blur chain. Parameters
//! are uploaded through an RDG transient uniform buffer only when the raw node
//! is live.

use std::mem::size_of;

use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::binding_layout::{
    depth_texture_2d, sampler, texture, texture_2d, uniform_buffer,
};
use crate::core::gpu::{CommonSampler, GlobalStateKey, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, PassNode, PrepareContext, RenderTargetOps,
    TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::buffer::CpuBuffer;
use myth_resources::ssao::SsaoUniforms;
use myth_resources::uniforms::WgslStruct;

/// The SSAO output texture format: single-channel unsigned normalized.
const SSAO_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Lightweight per-frame scene snapshot. No WGPU object or generated shader
/// interface crosses the graph-construction boundary.
#[derive(Clone, Copy)]
pub struct SsaoSnapshot {
    uniforms: SsaoUniforms,
    global_state_key: GlobalStateKey,
}

/// Long-lived SSAO feature. Raw and blur state are physically partitioned so
/// both graph nodes can independently borrow their own persistent resources.
#[derive(Default)]
pub struct SsaoFeature {
    raw: SsaoRawFeature,
    blur: SsaoBlurFeature,
}

#[derive(Default)]
struct SsaoRawFeature {
    pipeline_id: Option<RenderPipelineId>,
    raw_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    uniforms_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

#[derive(Default)]
struct SsaoBlurFeature {
    pipeline_id: Option<RenderPipelineId>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl SsaoFeature {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Capture CPU-only values during graph construction. The global binding
    /// state remains represented by a stable logical key and is resolved only
    /// if the live raw node finds its Feature-owned static resources missing.
    pub fn extract(
        &self,
        ssao_uniforms: &CpuBuffer<SsaoUniforms>,
        viewport: glam::Vec2,
        frame_index: u32,
        global_state_key: GlobalStateKey,
    ) -> SsaoSnapshot {
        let mut uniforms = *ssao_uniforms.read();
        uniforms.noise_scale =
            glam::Vec2::new((viewport.x * 0.5).max(1.0), (viewport.y * 0.5).max(1.0));
        uniforms.frame_index = frame_index;

        SsaoSnapshot {
            uniforms,
            global_state_key,
        }
    }

    /// Emit the raw SSAO and cross-bilateral blur nodes.
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: SsaoSnapshot,
        scene_depth: TextureNodeId,
        scene_normals: TextureNodeId,
    ) -> TextureNodeId {
        let raw_feature = &mut self.raw;
        let blur_feature = &mut self.blur;
        let half_w = (ctx.frame_config.width / 2).max(1);
        let half_h = (ctx.frame_config.height / 2).max(1);

        ctx.with_group("SSAO_System", |ctx| {
            let raw_desc = TextureDesc::new_2d(
                half_w,
                half_h,
                SSAO_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let raw_tex = ctx.graph.add_pass("SSAO_Raw", |builder| {
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                let uniforms_buffer = builder.create_buffer(
                    "SSAO_Uniforms",
                    BufferDesc::new(
                        size_of::<SsaoUniforms>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    ),
                );
                let output_tex = builder.create_texture("SSAO_Raw_Tex", raw_desc);
                let node = SsaoRawNode {
                    feature: raw_feature,
                    snapshot,
                    depth_tex: scene_depth,
                    normal_tex: scene_normals,
                    uniforms_buffer,
                    output_tex,
                    raw_bind_group: None,
                    uniforms_bind_group: None,
                };
                (node, output_tex)
            });

            let output_desc = TextureDesc::new_2d(
                half_w,
                half_h,
                SSAO_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            ctx.graph.add_pass("SSAO_Blur", |builder| {
                builder.read_texture(raw_tex);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                let output_tex = builder.create_texture("SSAO_Output", output_desc);
                let node = SsaoBlurNode {
                    feature: blur_feature,
                    raw_tex,
                    depth_tex: scene_depth,
                    normal_tex: scene_normals,
                    output_tex,
                    bind_group: None,
                };
                (node, output_tex)
            })
        })
    }
}

impl SsaoRawFeature {
    /// Ensure raw-pass static resources exist.
    /// Stable frames return before global-state resolution, shader compilation,
    /// layout creation, or L2 pipeline-cache access.
    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>, global_state_key: GlobalStateKey) {
        if self.pipeline_id.is_some() && self.raw_layout.is_some() && self.uniforms_layout.is_some()
        {
            return;
        }

        let global_state = ctx
            .global_states
            .get(global_state_key)
            .expect("SSAO: global binding state must exist before RDG prepare");

        let mut options = ShaderCompilationOptions::default();
        options.add_define(
            "struct_definitions",
            SsaoUniforms::wgsl_struct_def("SsaoUniforms").as_str(),
        );
        options.inject_code("binding_code", &global_state.binding_wgsl);
        options.inject_code(
            "scene_lighting_structs",
            myth_resources::uniforms::scene_lighting_structs_wgsl(),
        );

        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/features/ssao/raw"),
            &options,
        );
        let key = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![color_target(SSAO_TEXTURE_FORMAT)],
            None,
        );

        let pipeline_id = if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
            self.raw_layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 1).clone());
            self.uniforms_layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 2).clone());
            id
        } else {
            let global_layout =
                Tracked::with_id(global_state.layout.clone(), global_state.layout_id);
            let raw_layout = create_raw_layout(ctx.device);
            let uniforms_layout = create_uniforms_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSAO Raw Pipeline Layout"),
                        bind_group_layouts: &[
                            Some(&global_state.layout),
                            Some(&raw_layout),
                            Some(&uniforms_layout),
                        ],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSAO Raw Pipeline",
            );

            self.raw_layout = Some(raw_layout.clone());
            self.uniforms_layout = Some(uniforms_layout.clone());
            ctx.pipeline_cache.register_render_layouts(
                id,
                smallvec::smallvec![global_layout, raw_layout, uniforms_layout],
            );
            id
        };

        self.pipeline_id = Some(pipeline_id);
    }

    fn pipeline_id(&self) -> RenderPipelineId {
        self.pipeline_id
            .expect("SSAO raw pipeline must be prepared by the live raw node")
    }

    fn raw_layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.raw_layout
            .as_ref()
            .expect("SSAO raw layout must be prepared by the live raw node")
    }

    fn uniforms_layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.uniforms_layout
            .as_ref()
            .expect("SSAO uniforms layout must be prepared by the live raw node")
    }
}

impl SsaoBlurFeature {
    /// Ensure blur-pass static resources exist.
    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>) {
        if self.pipeline_id.is_some() && self.layout.is_some() {
            return;
        }

        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/features/ssao/blur"),
            &ShaderCompilationOptions::default(),
        );
        let key = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![color_target(SSAO_TEXTURE_FORMAT)],
            None,
        );

        let pipeline_id = if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
            self.layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 0).clone());
            id
        } else {
            let layout = create_blur_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSAO Blur Pipeline Layout"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSAO Blur Pipeline",
            );

            self.layout = Some(layout.clone());
            ctx.pipeline_cache
                .register_render_layouts(id, smallvec::smallvec![layout]);
            id
        };

        self.pipeline_id = Some(pipeline_id);
    }

    fn pipeline_id(&self) -> RenderPipelineId {
        self.pipeline_id
            .expect("SSAO blur pipeline must be prepared by the live blur node")
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("SSAO blur layout must be prepared by the live blur node")
    }
}

fn color_target(format: wgpu::TextureFormat) -> ColorTargetKey {
    ColorTargetKey::from(wgpu::ColorTargetState {
        format,
        blend: Some(wgpu::BlendState::REPLACE),
        write_mask: wgpu::ColorWrites::ALL,
    })
}

fn create_raw_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Raw Layout"),
            entries: &[
                depth_texture_2d(0, fragment),
                texture_2d(
                    1,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture(
                    2,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                    blue_noise_view_dimension(),
                    false,
                ),
                sampler(3, fragment, wgpu::SamplerBindingType::Filtering),
                sampler(4, fragment, wgpu::SamplerBindingType::Filtering),
                sampler(5, fragment, wgpu::SamplerBindingType::NonFiltering),
            ],
        }),
    )
}

fn create_uniforms_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Uniforms Layout"),
            entries: &[uniform_buffer(0, wgpu::ShaderStages::FRAGMENT)],
        }),
    )
}

fn create_blur_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Blur Layout"),
            entries: &[
                texture_2d(
                    0,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                depth_texture_2d(1, fragment),
                texture_2d(
                    2,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                sampler(3, fragment, wgpu::SamplerBindingType::Filtering),
                sampler(4, fragment, wgpu::SamplerBindingType::NonFiltering),
            ],
        }),
    )
}

struct SsaoRawNode<'a> {
    feature: &'a mut SsaoRawFeature,
    snapshot: SsaoSnapshot,
    depth_tex: TextureNodeId,
    normal_tex: TextureNodeId,
    uniforms_buffer: BufferNodeId,
    output_tex: TextureNodeId,
    raw_bind_group: Option<PreparedBindGroupId>,
    uniforms_bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for SsaoRawNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature
            .ensure_resources(ctx, self.snapshot.global_state_key);

        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.uniforms_buffer),
            0,
            bytemuck::bytes_of(&self.snapshot.uniforms),
        );

        let blue_noise = ctx.system_textures.blue_noise.clone();
        let blue_noise_sampler = ctx.system_textures.blue_noise_sampler.clone();
        self.raw_bind_group = Some(
            ctx.build_bind_group(self.feature.raw_layout(), Some("SSAO Raw BG (G1)"))
                .bind_texture(0, self.depth_tex)
                .bind_texture(1, self.normal_tex)
                .bind_tracked_texture_view(2, &blue_noise)
                .bind_common_sampler(3, CommonSampler::LinearClamp)
                .bind_tracked_sampler(4, &blue_noise_sampler)
                .bind_common_sampler(5, CommonSampler::NearestClamp)
                .build(),
        );
        self.uniforms_bind_group = Some(
            ctx.build_bind_group(
                self.feature.uniforms_layout(),
                Some("SSAO Uniforms BG (G2)"),
            )
            .bind_buffer(0, self.uniforms_buffer)
            .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let target = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSAO Raw Pass"),
            color_attachments: &[target],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_render_pipeline(self.feature.pipeline_id()),
        );
        pass.set_bind_group(0, ctx.global_bind_group, &[]);
        pass.set_bind_group(
            1,
            ctx.resolve_prepared_bind_group(self.raw_bind_group.expect("SSAO raw BG not prepared")),
            &[],
        );
        pass.set_bind_group(
            2,
            ctx.resolve_prepared_bind_group(
                self.uniforms_bind_group
                    .expect("SSAO uniforms BG not prepared"),
            ),
            &[],
        );
        pass.draw(0..3, 0..1);
    }
}

struct SsaoBlurNode<'a> {
    feature: &'a mut SsaoBlurFeature,
    raw_tex: TextureNodeId,
    depth_tex: TextureNodeId,
    normal_tex: TextureNodeId,
    output_tex: TextureNodeId,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for SsaoBlurNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(ctx);
        self.bind_group = Some(
            ctx.build_bind_group(self.feature.layout(), Some("SSAO Blur BG (G0)"))
                .bind_texture(0, self.raw_tex)
                .bind_texture(1, self.depth_tex)
                .bind_texture(2, self.normal_tex)
                .bind_common_sampler(3, CommonSampler::LinearClamp)
                .bind_common_sampler(4, CommonSampler::NearestClamp)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let target = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSAO Blur Pass"),
            color_attachments: &[target],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_render_pipeline(self.feature.pipeline_id()),
        );
        pass.set_bind_group(
            0,
            ctx.resolve_prepared_bind_group(self.bind_group.expect("SSAO blur BG not prepared")),
            &[],
        );
        pass.draw(0..3, 0..1);
    }
}

fn blue_noise_view_dimension() -> wgpu::TextureViewDimension {
    if cfg!(feature = "advanced_noise") {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}
