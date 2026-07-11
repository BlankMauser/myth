//! Demand-driven RDG tone mapping.
//!
//! Composer captures only POD settings and an optional logical 3D-LUT
//! identity. Layout/pipeline creation, LUT materialization, uniform upload, and
//! bind-group assembly happen in `PassNode::prepare` after DPE.

use rustc_hash::FxHashMap;

use crate::core::ResourceManager;
use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::{
    CommonSampler, GlobalStateKey, PreparedTextureSource, Tracked, binding_layout,
};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, PassNode, PrepareContext, RenderTargetOps,
    TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_assets::{AssetServer, TextureHandle};
use myth_resources::ShaderDefines;
use myth_resources::buffer::CpuBuffer;
use myth_resources::texture::TextureSource;
use myth_resources::tone_mapping::{ToneMappingMode, ToneMappingUniforms};
use myth_resources::uniforms::WgslStruct;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ToneMappingPipelineKey {
    mode: ToneMappingMode,
    output_format: wgpu::TextureFormat,
    has_lut: bool,
    global_layout_id: u64,
    shader_source_identity: u64,
}

#[derive(Debug, Clone, Copy)]
struct ToneMappingLutSnapshot {
    source: TextureSource,
    logical_version: u64,
}

/// Lightweight per-frame state captured while Composer builds the graph.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ToneMappingSnapshot {
    mode: ToneMappingMode,
    output_format: wgpu::TextureFormat,
    global_state_key: GlobalStateKey,
    uniforms: ToneMappingUniforms,
    lut: Option<ToneMappingLutSnapshot>,
}

/// View dimension of the system blue-noise texture: a temporally-layered array
/// when `advanced_noise` is enabled, otherwise a single 64×64 slice.
fn blue_noise_view_dimension() -> wgpu::TextureViewDimension {
    if cfg!(feature = "advanced_noise") {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

/// Cross-frame static state. No current-frame pipeline, bind group, uniform, or
/// physical LUT view is retained here.
pub struct ToneMappingFeature {
    static_layout_base: Option<Tracked<wgpu::BindGroupLayout>>,
    static_layout_lut: Option<Tracked<wgpu::BindGroupLayout>>,
    transient_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    local_cache: FxHashMap<ToneMappingPipelineKey, RenderPipelineId>,
}

impl Default for ToneMappingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl ToneMappingFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            static_layout_base: None,
            static_layout_lut: None,
            transient_layout: None,
            local_cache: FxHashMap::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn has_lut_pipeline(&self) -> bool {
        self.local_cache.keys().any(|key| key.has_lut)
    }

    #[cfg(test)]
    pub(crate) fn has_base_pipeline(&self) -> bool {
        self.local_cache.keys().any(|key| !key.has_lut)
    }

    /// Capture scene data and logical LUT readiness without touching WGPU.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::unused_self)] // Keep the uniform Feature::extract API.
    pub(crate) fn extract(
        &self,
        resource_manager: &ResourceManager,
        assets: &AssetServer,
        mode: ToneMappingMode,
        output_format: wgpu::TextureFormat,
        global_state_key: GlobalStateKey,
        uniforms: &CpuBuffer<ToneMappingUniforms>,
        lut_handle: Option<TextureHandle>,
    ) -> ToneMappingSnapshot {
        let lut = lut_handle.and_then(|handle| {
            let source = TextureSource::Asset(handle);
            let logical = resource_manager.snapshot_texture_source(assets, source);
            (logical.ready && logical.view_dimension == wgpu::TextureViewDimension::D3).then_some(
                ToneMappingLutSnapshot {
                    source,
                    logical_version: logical.version,
                },
            )
        });

        ToneMappingSnapshot {
            mode,
            output_format,
            global_state_key,
            uniforms: *uniforms.read(),
            lut,
        }
    }

    fn ensure_layouts(&mut self, device: &wgpu::Device, has_lut: bool) {
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let filterable = wgpu::TextureSampleType::Float { filterable: true };

        if self.transient_layout.is_none() {
            self.transient_layout = Some(Tracked::new(device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("ToneMap Transient Layout (G2)"),
                    entries: &[binding_layout::texture_2d(0, fragment, filterable)],
                },
            )));
        }

        let sampler_entry =
            binding_layout::sampler(0, fragment, wgpu::SamplerBindingType::Filtering);
        let uniform_entry = binding_layout::uniform_buffer(1, fragment);
        let blue_noise_tex_entry =
            binding_layout::texture(4, fragment, filterable, blue_noise_view_dimension(), false);
        let blue_noise_sampler_entry =
            binding_layout::sampler(5, fragment, wgpu::SamplerBindingType::Filtering);

        if has_lut {
            if self.static_layout_lut.is_none() {
                self.static_layout_lut = Some(Tracked::new(device.create_bind_group_layout(
                    &wgpu::BindGroupLayoutDescriptor {
                        label: Some("ToneMap Static Layout (LUT, G1)"),
                        entries: &[
                            sampler_entry,
                            uniform_entry,
                            binding_layout::texture(
                                2,
                                fragment,
                                filterable,
                                wgpu::TextureViewDimension::D3,
                                false,
                            ),
                            binding_layout::sampler(
                                3,
                                fragment,
                                wgpu::SamplerBindingType::Filtering,
                            ),
                            blue_noise_tex_entry,
                            blue_noise_sampler_entry,
                        ],
                    },
                )));
            }
        } else if self.static_layout_base.is_none() {
            self.static_layout_base = Some(Tracked::new(device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("ToneMap Static Layout (base, G1)"),
                    entries: &[
                        sampler_entry,
                        uniform_entry,
                        blue_noise_tex_entry,
                        blue_noise_sampler_entry,
                    ],
                },
            )));
        }
    }

    fn static_layout(&self, has_lut: bool) -> &Tracked<wgpu::BindGroupLayout> {
        if has_lut {
            self.static_layout_lut
                .as_ref()
                .expect("ToneMap LUT layout missing after prepare")
        } else {
            self.static_layout_base
                .as_ref()
                .expect("ToneMap base layout missing after prepare")
        }
    }

    fn transient_layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.transient_layout
            .as_ref()
            .expect("ToneMap transient layout missing after prepare")
    }

    fn get_or_create_pipeline(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        snapshot: ToneMappingSnapshot,
        has_lut: bool,
    ) -> RenderPipelineId {
        self.ensure_layouts(ctx.device, has_lut);

        let global_state = ctx
            .global_states
            .get(snapshot.global_state_key)
            .expect("ToneMap global state must exist before RDG prepare");
        let mut defines = ShaderDefines::new();
        snapshot.mode.apply_to_defines(&mut defines);
        if has_lut {
            defines.set("USE_LUT", "1");
        }
        defines.set("USE_BLUE_NOISE", "1");

        let mut options = ShaderCompilationOptions {
            defines,
            ..Default::default()
        };
        options.add_define(
            "struct_definitions",
            ToneMappingUniforms::wgsl_struct_def("Uniforms").as_str(),
        );
        options.inject_code("binding_code", &global_state.binding_wgsl);
        options.inject_code(
            "scene_lighting_structs",
            myth_resources::uniforms::scene_lighting_structs_wgsl(),
        );

        let source = ShaderSource::File("entry/post_process/tone_mapping");
        let shader_source_identity = ctx.shader_manager.pipeline_shader_hash(source, &options);
        let cache_key = ToneMappingPipelineKey {
            mode: snapshot.mode,
            output_format: snapshot.output_format,
            has_lut,
            global_layout_id: global_state.layout_id,
            shader_source_identity,
        };
        if let Some(&pipeline_id) = self.local_cache.get(&cache_key) {
            return pipeline_id;
        }

        let (shader_module, shader_hash) = ctx
            .shader_manager
            .get_or_compile(ctx.device, source, &options);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ToneMap Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&global_state.layout),
                    Some(self.static_layout(has_lut)),
                    Some(self.transient_layout()),
                ],
                immediate_size: 0,
            });
        let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
            format: snapshot.output_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let key =
            FullscreenPipelineKey::fullscreen(shader_hash, smallvec::smallvec![color_target], None);
        let pipeline_id = ctx.pipeline_cache.get_or_create_fullscreen(
            ctx.device,
            shader_module,
            &pipeline_layout,
            &key,
            "ToneMap Pipeline",
        );
        self.local_cache.insert(cache_key, pipeline_id);
        pipeline_id
    }

    pub(crate) fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: ToneMappingSnapshot,
        input_hdr: TextureNodeId,
    ) -> TextureNodeId {
        let output_desc = TextureDesc::new_2d(
            ctx.frame_config.width,
            ctx.frame_config.height,
            snapshot.output_format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );

        ctx.graph.add_pass("ToneMap_Pass", |builder| {
            builder.read_texture(input_hdr);
            let params_buffer = builder.create_buffer(
                "ToneMap_Params",
                BufferDesc::new(
                    std::mem::size_of::<ToneMappingUniforms>() as u64,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
            );
            let output = builder.create_texture("LDR_ToneMapped", output_desc);
            (
                ToneMapPassNode {
                    feature: self,
                    snapshot,
                    input_tex: input_hdr,
                    output_tex: output,
                    params_buffer,
                    pipeline_id: None,
                    static_bg: None,
                    transient_bg: None,
                },
                output,
            )
        })
    }
}

struct ToneMapPassNode<'a> {
    feature: &'a mut ToneMappingFeature,
    snapshot: ToneMappingSnapshot,
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    params_buffer: BufferNodeId,
    pipeline_id: Option<RenderPipelineId>,
    static_bg: Option<PreparedBindGroupId>,
    transient_bg: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for ToneMapPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let lut: Option<PreparedTextureSource> = self.snapshot.lut.and_then(|lut| {
            let prepared =
                ctx.uploads
                    .resolve_texture_source(ctx.assets, lut.source, lut.logical_version)?;
            (prepared.view_dimension == wgpu::TextureViewDimension::D3).then_some(prepared)
        });
        let has_lut = lut.is_some();
        self.pipeline_id = Some(
            self.feature
                .get_or_create_pipeline(ctx, self.snapshot, has_lut),
        );

        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.params_buffer),
            0,
            bytemuck::bytes_of(&self.snapshot.uniforms),
        );

        let blue_noise = ctx.system_textures.blue_noise.clone();
        let blue_noise_sampler = ctx.system_textures.blue_noise_sampler.clone();
        let static_layout = self.feature.static_layout(has_lut);
        let builder = ctx
            .build_bind_group(static_layout, Some("ToneMap Static BG (G1)"))
            .bind_common_sampler(0, CommonSampler::LinearClamp)
            .bind_buffer(1, self.params_buffer);
        let builder = if let Some(lut) = lut.as_ref() {
            builder
                .bind_texture_view_with_id(2, &lut.view, lut.view_id)
                .bind_common_sampler(3, CommonSampler::LinearClamp)
        } else {
            builder
        };
        self.static_bg = Some(
            builder
                .bind_tracked_texture_view(4, &blue_noise)
                .bind_tracked_sampler(5, &blue_noise_sampler)
                .build(),
        );

        self.transient_bg = Some(
            ctx.build_bind_group(
                self.feature.transient_layout(),
                Some("ToneMap Transient BG (G2)"),
            )
            .bind_texture(0, self.input_tex)
            .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let static_bg = ctx.resolve_prepared_bind_group(
            self.static_bg
                .expect("ToneMap static bind group missing after prepare"),
        );
        let transient_bg = ctx.resolve_prepared_bind_group(
            self.transient_bg
                .expect("ToneMap transient bind group missing after prepare"),
        );
        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ToneMap Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(
            ctx.pipeline_cache.get_render_pipeline(
                self.pipeline_id
                    .expect("ToneMap pipeline missing after prepare"),
            ),
        );
        rpass.set_bind_group(0, ctx.baked_lists.global_bind_group, &[]);
        rpass.set_bind_group(1, static_bg, &[]);
        rpass.set_bind_group(2, transient_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::graph::FrameConfig;
    use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};

    struct TestNode;

    impl PassNode<'_> for TestNode {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    fn frame_config() -> FrameConfig {
        FrameConfig {
            width: 320,
            height: 180,
            depth_format: wgpu::TextureFormat::Depth32Float,
            msaa_samples: 1,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            hdr_format: crate::HDR_TEXTURE_FORMAT,
        }
    }

    fn snapshot() -> ToneMappingSnapshot {
        ToneMappingSnapshot {
            mode: ToneMappingMode::Neutral,
            output_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            global_state_key: GlobalStateKey::new(1, 2),
            uniforms: ToneMappingUniforms::default(),
            lut: None,
        }
    }

    fn live_passes(root_output: bool) -> Vec<&'static str> {
        let mut feature = ToneMappingFeature::new();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let input = graph.add_pass("ToneMap_Input", |builder| {
            let output = builder.create_texture(
                "ToneMap_Input_HDR",
                TextureDesc::new_2d(
                    320,
                    180,
                    crate::HDR_TEXTURE_FORMAT,
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
            );
            (TestNode, output)
        });
        let config = frame_config();
        let output = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            feature.add_to_graph(&mut ctx, snapshot(), input)
        };
        if root_output {
            graph.add_pass("ToneMap_Root", |builder| {
                builder.read_texture(output);
                builder.mark_side_effect();
                (TestNode, ())
            });
        }
        graph.compile_topology();
        graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect()
    }

    #[test]
    fn dead_tone_mapping_culls_its_input_chain() {
        assert!(live_passes(false).is_empty());
    }

    #[test]
    fn live_tone_mapping_keeps_only_its_real_input() {
        let live = live_passes(true);
        assert!(live.contains(&"ToneMap_Input"));
        assert!(live.contains(&"ToneMap_Pass"));
        assert!(live.contains(&"ToneMap_Root"));
    }

    #[test]
    fn graph_build_does_not_initialize_static_resources() {
        let mut feature = ToneMappingFeature::new();
        {
            let mut storage = GraphStorage::new();
            let arena = FrameArena::new();
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let input = graph.add_pass("ToneMap_Dead_Input", |builder| {
                let output = builder.create_texture(
                    "ToneMap_Dead_HDR",
                    TextureDesc::new_2d(
                        16,
                        16,
                        crate::HDR_TEXTURE_FORMAT,
                        wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    ),
                );
                (TestNode, output)
            });
            let config = frame_config();
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            let _ = feature.add_to_graph(&mut ctx, snapshot(), input);
        }
        assert!(feature.static_layout_base.is_none());
        assert!(feature.static_layout_lut.is_none());
        assert!(feature.transient_layout.is_none());
        assert!(feature.local_cache.is_empty());
    }
}
