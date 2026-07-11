//! Screen Space Global Illumination feature.
//!
//! SSGI is implemented as a scene-level screen-space system with four stages:
//! raw Hi-Z ray march, temporal accumulation, multi-stage A-Trous cleanup,
//! and material-modulated merge back into the HDR scene color.

use std::mem::size_of;

use rustc_hash::FxHashMap;

use crate::HDR_TEXTURE_FORMAT;
use crate::core::gpu::binding_layout::{
    depth_texture_2d, sampler, texture, texture_2d, uniform_buffer,
};
use crate::core::gpu::{CommonSampler, GlobalStateKey, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::persistent::TEMPORAL_DOMAIN_TTL_FRAMES;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, HistoryPlan, HistoryRegistry, HistorySetId,
    HistoryTextureRef, PassNode, PrepareContext, RenderTargetOps, TemporalDomainKey, TextureDesc,
    TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::buffer::CpuBuffer;
use myth_resources::ssgi::SsgiUniforms;
use myth_resources::uniforms::WgslStruct;
use myth_scene::camera::RenderCamera;

const SSGI_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MAX_ATROUS_PASSES: usize = 4;
const HISTORY_FLAG_INDIRECT_VALID: u32 = 1 << 0;
const HISTORY_FLAG_SOURCE_VALID: u32 = 1 << 1;
const HISTORY_FLAG_CAMERA_CUT: u32 = 1 << 2;
const HISTORY_VALID_MASK: u32 = HISTORY_FLAG_INDIRECT_VALID | HISTORY_FLAG_SOURCE_VALID;
const HISTORY_RUNTIME_MASK: u32 = HISTORY_VALID_MASK | HISTORY_FLAG_CAMERA_CUT;
const SSGI_ATROUS_PASS_NAMES: [&str; MAX_ATROUS_PASSES] = [
    "SSGI_ATrous_Step_1",
    "SSGI_ATrous_Step_2",
    "SSGI_ATrous_Step_4",
    "SSGI_ATrous_Step_8",
];
const SSGI_ATROUS_OUTPUT_NAMES: [&str; MAX_ATROUS_PASSES] = [
    "SSGI_ATrous_Step_1_Output",
    "SSGI_ATrous_Step_2_Output",
    "SSGI_ATrous_Step_4_Output",
    "SSGI_ATrous_Step_8_Output",
];

fn blue_noise_view_dimension() -> wgpu::TextureViewDimension {
    if cfg!(feature = "advanced_noise") {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

#[must_use = "SSA Graph: consume the merged color output from SSGI"]
pub struct SsgiOutputs {
    pub merged_color: TextureNodeId,
    pub raw_indirect: TextureNodeId,
    pub clean_indirect: TextureNodeId,
}

/// Per-frame, CPU-only SSGI data captured while Composer builds the graph.
/// Physical WGPU objects remain feature-owned and are materialized only from
/// live nodes during deferred prepare.
#[derive(Clone, Copy)]
pub struct SsgiSnapshot {
    uniforms: SsgiUniforms,
    indirect_history: HistoryPlan<2>,
    source_history: HistoryPlan<1>,
    full_resolution: (u32, u32),
    half_resolution: (u32, u32),
    max_steps: u32,
    atrous_pass_count: usize,
    taa_history: Option<HistoryTextureRef>,
    global_state_key: GlobalStateKey,
}

#[derive(Default)]
struct SsgiRawFeature {
    pipelines: FxHashMap<u32, RenderPipelineId>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

#[derive(Default)]
struct SsgiTemporalFeature {
    pipeline_id: Option<RenderPipelineId>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

#[derive(Default)]
struct SsgiAtrousFeature {
    pipeline_id: Option<RenderPipelineId>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

#[derive(Default)]
struct SsgiMergeFeature {
    pipeline_id: Option<RenderPipelineId>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

#[derive(Debug, Clone, Copy)]
struct SsgiHistoryDomain {
    indirect_set_id: HistorySetId,
    source_set_id: HistorySetId,
    full_resolution: (u32, u32),
    last_planned_frame: u64,
}

pub struct SsgiFeature {
    raw: SsgiRawFeature,
    temporal: SsgiTemporalFeature,
    atrous: [SsgiAtrousFeature; MAX_ATROUS_PASSES],
    merge: SsgiMergeFeature,
    histories: FxHashMap<TemporalDomainKey, SsgiHistoryDomain>,
}

impl Default for SsgiFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SsgiFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            raw: SsgiRawFeature::default(),
            temporal: SsgiTemporalFeature::default(),
            atrous: std::array::from_fn(|_| SsgiAtrousFeature::default()),
            merge: SsgiMergeFeature::default(),
            histories: FxHashMap::default(),
        }
    }

    /// Capture a lightweight snapshot and plan both independently-published
    /// history sets. No WGPU object is created and the scene-owned CPU buffer
    /// is never mutated.
    #[allow(clippy::too_many_arguments)]
    pub fn extract(
        &mut self,
        histories: &mut HistoryRegistry,
        scene_id: u32,
        camera: &RenderCamera,
        temporal_serial: u64,
        renderer_frame: u64,
        source_uniforms: &CpuBuffer<SsgiUniforms>,
        size: (u32, u32),
        _frame_index: u32,
        taa_history: Option<HistoryTextureRef>,
        global_state_key: GlobalStateKey,
    ) -> SsgiSnapshot {
        let key = TemporalDomainKey::new(scene_id, camera.camera_id);
        if self.histories.get(&key).is_some_and(|domain| {
            renderer_frame.saturating_sub(domain.last_planned_frame) > TEMPORAL_DOMAIN_TTL_FRAMES
        }) {
            let expired = self
                .histories
                .remove(&key)
                .expect("expired SSGI history domain must exist");
            let indirect_removed = histories.remove_set(expired.indirect_set_id);
            let source_removed = histories.remove_set(expired.source_set_id);
            debug_assert!(
                indirect_removed && source_removed,
                "SSGI history domain lost a registry set"
            );
        }

        let (indirect_set_id, source_set_id, full_resolution_changed) =
            if let Some(domain) = self.histories.get_mut(&key) {
                let changed = domain.full_resolution != size;
                domain.full_resolution = size;
                domain.last_planned_frame = renderer_frame;
                (domain.indirect_set_id, domain.source_set_id, changed)
            } else {
                let indirect_set_id = histories.create_set();
                let source_set_id = histories.create_set();
                self.histories.insert(
                    key,
                    SsgiHistoryDomain {
                        indirect_set_id,
                        source_set_id,
                        full_resolution: size,
                        last_planned_frame: renderer_frame,
                    },
                );
                (indirect_set_id, source_set_id, false)
            };

        let full_resolution = (size.0.max(1), size.1.max(1));
        let half_resolution = (
            (full_resolution.0 / 2).max(1),
            (full_resolution.1 / 2).max(1),
        );
        let indirect_desc = TextureDesc::new_2d(
            half_resolution.0,
            half_resolution.1,
            SSGI_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let source_desc = TextureDesc::new_2d(
            full_resolution.0,
            full_resolution.1,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let reset = camera.camera_cut != 0 || full_resolution_changed;
        let indirect_history = histories.plan_set(
            indirect_set_id,
            [indirect_desc, indirect_desc],
            temporal_serial,
            reset,
        );
        let source_history =
            histories.plan_set(source_set_id, [source_desc], temporal_serial, reset);

        let mut uniforms = *source_uniforms.read();
        uniforms.full_resolution = glam::Vec4::new(
            full_resolution.0 as f32,
            full_resolution.1 as f32,
            1.0 / full_resolution.0 as f32,
            1.0 / full_resolution.1 as f32,
        );
        uniforms.half_resolution = glam::Vec4::new(
            half_resolution.0 as f32,
            half_resolution.1 as f32,
            1.0 / half_resolution.0 as f32,
            1.0 / half_resolution.1 as f32,
        );
        // Sampling advances with this history domain, even when cameras alternate.
        uniforms.frame_params.x = temporal_serial as u32;
        uniforms.temporal_params.z = camera.near.max(0.0001);
        uniforms.frame_params.w &= !HISTORY_RUNTIME_MASK;
        if indirect_history.valid {
            uniforms.frame_params.w |= HISTORY_FLAG_INDIRECT_VALID;
        }
        if source_history.valid {
            uniforms.frame_params.w |= HISTORY_FLAG_SOURCE_VALID;
        }
        if camera.camera_cut != 0 {
            uniforms.frame_params.w |= HISTORY_FLAG_CAMERA_CUT;
        }

        SsgiSnapshot {
            uniforms,
            indirect_history,
            source_history,
            full_resolution,
            half_resolution,
            max_steps: uniforms.frame_params.y,
            atrous_pass_count: uniforms.denoise_params.x.clamp(1, MAX_ATROUS_PASSES as u32)
                as usize,
            taa_history: source_history.valid.then_some(taa_history).flatten(),
            global_state_key,
        }
    }

    pub(crate) fn garbage_collect(
        &mut self,
        histories: &mut HistoryRegistry,
        current_frame: u64,
        ttl: u64,
    ) -> usize {
        let before = self.histories.len();
        self.histories.retain(|_, domain| {
            let live = current_frame.saturating_sub(domain.last_planned_frame) <= ttl;
            if !live {
                let indirect_removed = histories.remove_set(domain.indirect_set_id);
                let source_removed = histories.remove_set(domain.source_set_id);
                debug_assert!(
                    indirect_removed && source_removed,
                    "SSGI history domain lost a registry set"
                );
            }
            live
        });
        before - self.histories.len()
    }

    /// Emit Raw → Temporal → A-Trous → Merge. Temporal renders directly into
    /// the next indirect/meta bank and Merge writes both its isolated transient
    /// output and the next source-history bank as identical MRT attachments.
    #[allow(clippy::too_many_arguments)]
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: SsgiSnapshot,
        current_color: TextureNodeId,
        scene_depth: TextureNodeId,
        scene_hiz: TextureNodeId,
        scene_normals: TextureNodeId,
        velocity: TextureNodeId,
        material_mrt: TextureNodeId,
        pmrem_texture: Option<TextureNodeId>,
    ) -> SsgiOutputs {
        let raw_feature = &mut self.raw;
        let temporal_feature = &mut self.temporal;
        let atrous_features = &mut self.atrous;
        let merge_feature = &mut self.merge;

        ctx.with_group("SSGI_System", |ctx| {
            let previous_indirect = ctx.graph.import_persistent_texture(
                "SSGI_History_Indirect_Previous",
                snapshot.indirect_history.previous[0],
            );
            let previous_meta = ctx.graph.import_persistent_texture(
                "SSGI_History_Meta_Previous",
                snapshot.indirect_history.previous[1],
            );
            let next_indirect = ctx.graph.import_persistent_texture(
                "SSGI_History_Indirect_Next",
                snapshot.indirect_history.next[0],
            );
            let next_meta = ctx.graph.import_persistent_texture(
                "SSGI_History_Meta_Next",
                snapshot.indirect_history.next[1],
            );
            let previous_source = if let Some(taa_history) = snapshot.taa_history {
                ctx.graph
                    .import_persistent_texture("TAA_History_Color_Previous", taa_history)
            } else {
                ctx.graph.import_persistent_texture(
                    "SSGI_History_Source_Previous",
                    snapshot.source_history.previous[0],
                )
            };
            let next_source = ctx.graph.import_persistent_texture(
                "SSGI_History_Source_Next",
                snapshot.source_history.next[0],
            );

            let uniforms_desc = BufferDesc::new(
                size_of::<SsgiUniforms>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let half_desc = TextureDesc::new_2d(
                snapshot.half_resolution.0,
                snapshot.half_resolution.1,
                SSGI_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let (raw_indirect, uniforms_buffer) = ctx.graph.add_pass("SSGI_Raw", |builder| {
                builder.read_texture(scene_depth);
                builder.read_texture(scene_hiz);
                builder.read_texture(scene_normals);
                builder.read_texture(previous_source);
                if let Some(pmrem_texture) = pmrem_texture {
                    builder.read_texture(pmrem_texture);
                }

                let uniforms_buffer = builder.create_buffer("SSGI_Uniforms", uniforms_desc);
                let output = builder.create_texture("SSGI_Raw_Indirect", half_desc);
                let node = SsgiRawNode {
                    feature: raw_feature,
                    global_state_key: snapshot.global_state_key,
                    max_steps: snapshot.max_steps,
                    uniforms_data: snapshot.uniforms,
                    scene_depth,
                    scene_hiz,
                    scene_normals,
                    source_history: previous_source,
                    pmrem_texture,
                    uniforms: uniforms_buffer,
                    output_tex: output,
                    pipeline_id: None,
                    bind_group: None,
                };
                (node, (output, uniforms_buffer))
            });

            ctx.graph.add_pass("SSGI_Temporal", |builder| {
                builder.read_texture(raw_indirect);
                builder.read_texture(previous_indirect);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                builder.read_texture(previous_meta);
                builder.read_texture(velocity);
                builder.read_buffer(uniforms_buffer);
                builder.write_texture(next_indirect);
                builder.write_texture(next_meta);

                let node = SsgiTemporalNode {
                    feature: temporal_feature,
                    raw_indirect,
                    indirect_history: previous_indirect,
                    scene_depth,
                    scene_normals,
                    history_meta: previous_meta,
                    velocity,
                    uniforms: uniforms_buffer,
                    output_indirect: next_indirect,
                    output_meta: next_meta,
                    pipeline_id: None,
                    bind_group: None,
                };
                (node, ())
            });
            ctx.graph.register_history_commit(
                snapshot.indirect_history.commit,
                &[next_indirect, next_meta],
            );

            let mut clean_indirect = next_indirect;
            for (pass_index, feature) in atrous_features
                .iter_mut()
                .take(snapshot.atrous_pass_count)
                .enumerate()
            {
                let input = clean_indirect;
                clean_indirect =
                    ctx.graph
                        .add_pass(SSGI_ATROUS_PASS_NAMES[pass_index], |builder| {
                            builder.read_texture(input);
                            builder.read_texture(scene_depth);
                            builder.read_texture(scene_normals);
                            builder.read_texture(next_meta);
                            builder.read_buffer(uniforms_buffer);

                            let output = builder
                                .create_texture(SSGI_ATROUS_OUTPUT_NAMES[pass_index], half_desc);
                            let node = SsgiAtrousNode {
                                feature,
                                step_size: 1 << pass_index,
                                input_indirect: input,
                                scene_depth,
                                scene_normals,
                                variance_meta: next_meta,
                                uniforms: uniforms_buffer,
                                output_tex: output,
                                pipeline_id: None,
                                bind_group: None,
                            };
                            (node, output)
                        });
            }

            let merged_desc = TextureDesc::new_2d(
                snapshot.full_resolution.0,
                snapshot.full_resolution.1,
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let merged_color = ctx.graph.add_pass("SSGI_Merge", |builder| {
                builder.read_texture(current_color);
                builder.read_texture(clean_indirect);
                builder.read_texture(material_mrt);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                builder.read_buffer(uniforms_buffer);
                builder.write_texture(next_source);

                let output = builder.create_texture("SSGI_Merged_Color", merged_desc);
                let node = SsgiMergeNode {
                    feature: merge_feature,
                    current_color,
                    clean_indirect,
                    material_mrt,
                    scene_depth,
                    scene_normals,
                    uniforms: uniforms_buffer,
                    output_tex: output,
                    output_source: next_source,
                    pipeline_id: None,
                    bind_group: None,
                };
                (node, output)
            });
            ctx.graph
                .register_history_commit(snapshot.source_history.commit, &[next_source]);

            SsgiOutputs {
                merged_color,
                raw_indirect,
                clean_indirect,
            }
        })
    }
}

impl SsgiRawFeature {
    fn ensure_resources(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        global_state_key: GlobalStateKey,
        max_steps: u32,
    ) {
        if self.pipelines.contains_key(&max_steps) && self.layout.is_some() {
            return;
        }

        let global_state = ctx
            .global_states
            .get(global_state_key)
            .expect("SSGI raw: global binding state must exist before RDG prepare");
        let mut options = global_shader_options(global_state.binding_wgsl.as_str());
        options.inject_code("ssgi_max_steps", max_steps.to_string());
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/features/ssgi/raw"),
            &options,
        );
        let key = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![color_target(SSGI_TEXTURE_FORMAT)],
            None,
        );

        let pipeline_id = if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
            self.layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 1).clone());
            id
        } else {
            let local_layout = self
                .layout
                .clone()
                .unwrap_or_else(|| create_raw_layout(ctx.device));
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSGI Raw Pipeline Layout"),
                        bind_group_layouts: &[Some(&global_state.layout), Some(&local_layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSGI Raw Pipeline",
            );
            register_global_and_local_layouts(ctx, id, global_state, local_layout.clone());
            self.layout = Some(local_layout);
            id
        };
        self.pipelines.insert(max_steps, pipeline_id);
    }

    fn pipeline_id(&self, max_steps: u32) -> RenderPipelineId {
        self.pipelines
            .get(&max_steps)
            .copied()
            .expect("SSGI raw pipeline must be prepared by the live node")
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("SSGI raw layout must be prepared by the live node")
    }
}

impl SsgiTemporalFeature {
    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>) {
        if self.pipeline_id.is_some() && self.layout.is_some() {
            return;
        }

        let options = local_shader_options();
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/features/ssgi/temporal"),
            &options,
        );
        let key = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![
                color_target(SSGI_TEXTURE_FORMAT),
                color_target(SSGI_TEXTURE_FORMAT),
            ],
            None,
        );
        let pipeline_id = if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
            self.layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 0).clone());
            id
        } else {
            let layout = create_temporal_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSGI Temporal Pipeline Layout"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSGI Temporal Pipeline",
            );
            ctx.pipeline_cache
                .register_render_layouts(id, smallvec::smallvec![layout.clone()]);
            self.layout = Some(layout);
            id
        };
        self.pipeline_id = Some(pipeline_id);
    }

    fn pipeline_id(&self) -> RenderPipelineId {
        self.pipeline_id
            .expect("SSGI temporal pipeline must be prepared by the live node")
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("SSGI temporal layout must be prepared by the live node")
    }
}

impl SsgiAtrousFeature {
    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>, step_size: u32) {
        if self.pipeline_id.is_some() && self.layout.is_some() {
            return;
        }

        let options = local_shader_options();
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/features/ssgi/atrous_blur"),
            &options,
        );
        let constants = [("SSGI_ATROUS_STEP_SIZE", f64::from(step_size))];
        let compilation_options = wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        };
        let key = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![color_target(SSGI_TEXTURE_FORMAT)],
            None,
        )
        .with_compilation_options(&compilation_options);
        let pipeline_id = if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
            self.layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 0).clone());
            id
        } else {
            let layout = create_atrous_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSGI A-Trous Pipeline Layout"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_fullscreen_with_options(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                &compilation_options,
                "SSGI A-Trous Pipeline",
            );
            ctx.pipeline_cache
                .register_render_layouts(id, smallvec::smallvec![layout.clone()]);
            self.layout = Some(layout);
            id
        };
        self.pipeline_id = Some(pipeline_id);
    }

    fn pipeline_id(&self) -> RenderPipelineId {
        self.pipeline_id
            .expect("SSGI A-Trous pipeline must be prepared by the live node")
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("SSGI A-Trous layout must be prepared by the live node")
    }
}

impl SsgiMergeFeature {
    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>) {
        if self.pipeline_id.is_some() && self.layout.is_some() {
            return;
        }

        let options = local_shader_options();
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/features/ssgi/merge"),
            &options,
        );
        let key = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![
                color_target(HDR_TEXTURE_FORMAT),
                color_target(HDR_TEXTURE_FORMAT),
            ],
            None,
        );
        let pipeline_id = if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
            self.layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 0).clone());
            id
        } else {
            let layout = create_merge_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSGI Merge Pipeline Layout"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSGI Merge Pipeline",
            );
            ctx.pipeline_cache
                .register_render_layouts(id, smallvec::smallvec![layout.clone()]);
            self.layout = Some(layout);
            id
        };
        self.pipeline_id = Some(pipeline_id);
    }

    fn pipeline_id(&self) -> RenderPipelineId {
        self.pipeline_id
            .expect("SSGI merge pipeline must be prepared by the live node")
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("SSGI merge layout must be prepared by the live node")
    }
}

fn local_shader_options() -> ShaderCompilationOptions {
    let mut options = ShaderCompilationOptions::default();
    options.add_define(
        "struct_definitions",
        SsgiUniforms::wgsl_struct_def("SsgiUniforms").as_str(),
    );
    options
}

fn global_shader_options(binding_wgsl: &str) -> ShaderCompilationOptions {
    let mut options = local_shader_options();
    options.inject_code("binding_code", binding_wgsl);
    options.inject_code(
        "scene_lighting_structs",
        myth_resources::uniforms::scene_lighting_structs_wgsl(),
    );
    options
}

fn register_global_and_local_layouts(
    ctx: &mut PrepareContext<'_>,
    pipeline_id: RenderPipelineId,
    global_state: &crate::core::gpu::GpuGlobalState,
    local_layout: Tracked<wgpu::BindGroupLayout>,
) {
    let global_layout = Tracked::with_id(global_state.layout.clone(), global_state.layout_id);
    ctx.pipeline_cache.register_render_layouts(
        pipeline_id,
        smallvec::smallvec![global_layout, local_layout],
    );
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
            label: Some("SSGI Raw Layout"),
            entries: &[
                depth_texture_2d(0, fragment),
                texture_2d(
                    1,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(
                    2,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(
                    3,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: false },
                ),
                texture(
                    4,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                    wgpu::TextureViewDimension::Cube,
                    false,
                ),
                sampler(5, fragment, wgpu::SamplerBindingType::Filtering),
                sampler(6, fragment, wgpu::SamplerBindingType::NonFiltering),
                uniform_buffer(7, fragment),
                texture(
                    8,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                    blue_noise_view_dimension(),
                    false,
                ),
                sampler(9, fragment, wgpu::SamplerBindingType::Filtering),
            ],
        }),
    )
}

fn create_temporal_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSGI Temporal Layout"),
            entries: &[
                texture_2d(
                    0,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(
                    1,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                depth_texture_2d(2, fragment),
                texture_2d(
                    3,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(
                    4,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(
                    5,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: false },
                ),
                sampler(6, fragment, wgpu::SamplerBindingType::Filtering),
                sampler(7, fragment, wgpu::SamplerBindingType::NonFiltering),
                uniform_buffer(8, fragment),
            ],
        }),
    )
}

fn create_atrous_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSGI A-Trous Layout"),
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
                texture_2d(
                    3,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                sampler(4, fragment, wgpu::SamplerBindingType::Filtering),
                sampler(5, fragment, wgpu::SamplerBindingType::NonFiltering),
                uniform_buffer(6, fragment),
            ],
        }),
    )
}

fn create_merge_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSGI Merge Layout"),
            entries: &[
                texture_2d(
                    0,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(
                    1,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(
                    2,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                depth_texture_2d(3, fragment),
                texture_2d(
                    4,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                sampler(5, fragment, wgpu::SamplerBindingType::Filtering),
                sampler(6, fragment, wgpu::SamplerBindingType::NonFiltering),
                uniform_buffer(7, fragment),
            ],
        }),
    )
}

struct SsgiRawNode<'a> {
    feature: &'a mut SsgiRawFeature,
    global_state_key: GlobalStateKey,
    max_steps: u32,
    uniforms_data: SsgiUniforms,
    scene_depth: TextureNodeId,
    scene_hiz: TextureNodeId,
    scene_normals: TextureNodeId,
    source_history: TextureNodeId,
    pmrem_texture: Option<TextureNodeId>,
    uniforms: BufferNodeId,
    output_tex: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsgiRawNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature
            .ensure_resources(ctx, self.global_state_key, self.max_steps);
        self.pipeline_id = Some(self.feature.pipeline_id(self.max_steps));
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.uniforms),
            0,
            bytemuck::bytes_of(&self.uniforms_data),
        );

        let layout = self.feature.layout().clone();
        let blue_noise_view = ctx.system_textures.blue_noise.clone();
        let blue_noise_sampler = ctx.system_textures.blue_noise_sampler.clone();
        let black_cube = ctx.system_textures.black_cube.clone();
        let builder = ctx
            .build_bind_group(&layout, Some("SSGI Raw BG"))
            .bind_texture(0, self.scene_depth)
            .bind_texture(1, self.scene_normals)
            .bind_texture(2, self.source_history)
            .bind_texture(3, self.scene_hiz);
        let builder = if let Some(pmrem_texture) = self.pmrem_texture {
            builder.bind_texture(4, pmrem_texture)
        } else {
            builder.bind_tracked_texture_view(4, &black_cube)
        };
        self.bind_group = Some(
            builder
                .bind_common_sampler(5, CommonSampler::LinearClamp)
                .bind_common_sampler(6, CommonSampler::NearestClamp)
                .bind_buffer(7, self.uniforms)
                .bind_tracked_texture_view(8, &blue_noise_view)
                .bind_tracked_sampler(9, &blue_noise_sampler)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let output = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI Raw Pass"),
            color_attachments: &[output],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_render_pipeline(self.pipeline_id.expect("SSGI raw pipeline not prepared")),
        );
        pass.set_bind_group(0, ctx.baked_lists.global_bind_group, &[]);
        pass.set_bind_group(1, self.bind_group.expect("SSGI raw BG missing"), &[]);
        pass.draw(0..3, 0..1);
    }
}

struct SsgiTemporalNode<'a> {
    feature: &'a mut SsgiTemporalFeature,
    raw_indirect: TextureNodeId,
    indirect_history: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    history_meta: TextureNodeId,
    velocity: TextureNodeId,
    uniforms: BufferNodeId,
    output_indirect: TextureNodeId,
    output_meta: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsgiTemporalNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(ctx);
        self.pipeline_id = Some(self.feature.pipeline_id());
        let layout = self.feature.layout().clone();
        self.bind_group = Some(
            crate::myth_bind_group!(ctx, &layout, Some("SSGI Temporal BG"), [
                0 => self.raw_indirect,
                1 => self.indirect_history,
                2 => self.scene_depth,
                3 => self.scene_normals,
                4 => self.history_meta,
                5 => self.velocity,
                6 => CommonSampler::LinearClamp,
                7 => CommonSampler::NearestClamp,
                8 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let attachments = [
            ctx.get_color_attachment(self.output_indirect, RenderTargetOps::DontCare, None),
            ctx.get_color_attachment(self.output_meta, RenderTargetOps::DontCare, None),
        ];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI Temporal Pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(
            ctx.pipeline_cache.get_render_pipeline(
                self.pipeline_id
                    .expect("SSGI temporal pipeline not prepared"),
            ),
        );
        pass.set_bind_group(0, self.bind_group.expect("SSGI temporal BG missing"), &[]);
        pass.draw(0..3, 0..1);
    }
}

struct SsgiAtrousNode<'a> {
    feature: &'a mut SsgiAtrousFeature,
    step_size: u32,
    input_indirect: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    variance_meta: TextureNodeId,
    uniforms: BufferNodeId,
    output_tex: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsgiAtrousNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(ctx, self.step_size);
        self.pipeline_id = Some(self.feature.pipeline_id());
        let layout = self.feature.layout().clone();
        self.bind_group = Some(
            crate::myth_bind_group!(ctx, &layout, Some("SSGI A-Trous BG"), [
                0 => self.input_indirect,
                1 => self.scene_depth,
                2 => self.scene_normals,
                3 => self.variance_meta,
                4 => CommonSampler::LinearClamp,
                5 => CommonSampler::NearestClamp,
                6 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let output = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI A-Trous Pass"),
            color_attachments: &[output],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(
            ctx.pipeline_cache.get_render_pipeline(
                self.pipeline_id
                    .expect("SSGI A-Trous pipeline not prepared"),
            ),
        );
        pass.set_bind_group(0, self.bind_group.expect("SSGI A-Trous BG missing"), &[]);
        pass.draw(0..3, 0..1);
    }
}

struct SsgiMergeNode<'a> {
    feature: &'a mut SsgiMergeFeature,
    current_color: TextureNodeId,
    clean_indirect: TextureNodeId,
    material_mrt: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    uniforms: BufferNodeId,
    output_tex: TextureNodeId,
    output_source: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsgiMergeNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(ctx);
        self.pipeline_id = Some(self.feature.pipeline_id());
        let layout = self.feature.layout().clone();
        self.bind_group = Some(
            crate::myth_bind_group!(ctx, &layout, Some("SSGI Merge BG"), [
                0 => self.current_color,
                1 => self.clean_indirect,
                2 => self.material_mrt,
                3 => self.scene_depth,
                4 => self.scene_normals,
                5 => CommonSampler::LinearClamp,
                6 => CommonSampler::NearestClamp,
                7 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let attachments = [
            ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None),
            ctx.get_color_attachment(self.output_source, RenderTargetOps::DontCare, None),
        ];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI Merge Pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_render_pipeline(self.pipeline_id.expect("SSGI merge pipeline not prepared")),
        );
        pass.set_bind_group(0, self.bind_group.expect("SSGI merge BG missing"), &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::graph::FrameConfig;
    use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};
    use myth_resources::ssgi::SsgiSettings;
    use myth_scene::camera::Camera;

    const CUSTOM_RUNTIME_BIT: u32 = 1 << 7;

    fn camera(camera_id: u128) -> RenderCamera {
        let mut camera = Camera::new_perspective(45.0, 1.0, 0.1);
        let mut render_camera = camera.extract_render_camera();
        render_camera.camera_id = camera_id;
        render_camera.camera_cut = 0;
        render_camera
    }

    fn extract(
        feature: &mut SsgiFeature,
        histories: &mut HistoryRegistry,
        settings: &SsgiSettings,
        camera: &RenderCamera,
        temporal_serial: u64,
        size: (u32, u32),
    ) -> SsgiSnapshot {
        feature.extract(
            histories,
            1,
            camera,
            temporal_serial,
            temporal_serial,
            &settings.uniforms,
            size,
            temporal_serial as u32,
            None,
            GlobalStateKey::new(1, 1),
        )
    }

    fn commit_indirect(histories: &mut HistoryRegistry, snapshot: SsgiSnapshot) -> bool {
        histories.commit(
            snapshot.indirect_history.commit,
            &snapshot.indirect_history.next.map(|texture| texture.id),
        )
    }

    fn commit_source(histories: &mut HistoryRegistry, snapshot: SsgiSnapshot) -> bool {
        histories.commit(
            snapshot.source_history.commit,
            &snapshot.source_history.next.map(|texture| texture.id),
        )
    }

    struct TestNode;

    impl<'a> PassNode<'a> for TestNode {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    fn frame_config() -> FrameConfig {
        FrameConfig {
            width: 1280,
            height: 720,
            depth_format: wgpu::TextureFormat::Depth32Float,
            msaa_samples: 1,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            hdr_format: HDR_TEXTURE_FORMAT,
        }
    }

    fn register_inputs(
        graph: &mut RenderGraph<'_>,
    ) -> (
        TextureNodeId,
        TextureNodeId,
        TextureNodeId,
        TextureNodeId,
        TextureNodeId,
        TextureNodeId,
    ) {
        let sampled = wgpu::TextureUsages::TEXTURE_BINDING;
        let color = graph.register_texture(
            "SSGI_Test_Color",
            TextureDesc::new_2d(1280, 720, HDR_TEXTURE_FORMAT, sampled),
            true,
        );
        let depth = graph.register_texture(
            "SSGI_Test_Depth",
            TextureDesc::new_2d(1280, 720, wgpu::TextureFormat::Depth32Float, sampled),
            true,
        );
        let hiz = graph.register_texture(
            "SSGI_Test_HiZ",
            TextureDesc::new_2d(1280, 720, wgpu::TextureFormat::R32Float, sampled),
            true,
        );
        let data = graph.register_texture(
            "SSGI_Test_Data",
            TextureDesc::new_2d(1280, 720, SSGI_TEXTURE_FORMAT, sampled),
            true,
        );
        (color, depth, hiz, data, data, data)
    }

    fn add_graph<'a>(
        feature: &'a mut SsgiFeature,
        graph: &mut RenderGraph<'a>,
        snapshot: SsgiSnapshot,
    ) -> SsgiOutputs {
        let config = frame_config();
        let (color, depth, hiz, normals, velocity, material) = register_inputs(graph);
        let mut ctx = GraphBuilderContext::new(graph, &config, 0);
        feature.add_to_graph(
            &mut ctx, snapshot, color, depth, hiz, normals, velocity, material, None,
        )
    }

    fn add_root(graph: &mut RenderGraph<'_>, source: TextureNodeId) {
        let output = graph.register_texture(
            "SSGI_Test_Output",
            TextureDesc::new_2d(
                1280,
                720,
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
            true,
        );
        graph.add_pass("SSGI_Test_Root", |builder| {
            builder.read_texture(source);
            builder.write_texture(output);
            (TestNode, ())
        });
    }

    fn followup_validity(histories: &mut HistoryRegistry, snapshot: SsgiSnapshot) -> (bool, bool) {
        let indirect = histories.plan_set(
            snapshot.indirect_history.commit.set_id,
            snapshot.indirect_history.next.map(|texture| texture.desc),
            2,
            false,
        );
        let source = histories.plan_set(
            snapshot.source_history.commit.set_id,
            snapshot.source_history.next.map(|texture| texture.desc),
            2,
            false,
        );
        (indirect.valid, source.valid)
    }

    #[test]
    fn extract_does_not_mutate_scene_uniforms_and_validity_requires_commit() {
        let mut feature = SsgiFeature::new();
        let mut histories = HistoryRegistry::new();
        let mut settings = SsgiSettings::default();
        settings.set_history_flags(CUSTOM_RUNTIME_BIT);
        let camera = camera(7);
        let before = *settings.uniforms.read();

        let first = extract(
            &mut feature,
            &mut histories,
            &settings,
            &camera,
            1,
            (1920, 1080),
        );
        assert!(!first.indirect_history.valid);
        assert!(!first.source_history.valid);
        assert_eq!(first.uniforms.frame_params.w, CUSTOM_RUNTIME_BIT);
        assert_eq!(
            bytemuck::bytes_of(&*settings.uniforms.read()),
            bytemuck::bytes_of(&before)
        );

        assert!(commit_indirect(&mut histories, first));
        assert!(commit_source(&mut histories, first));
        let second = extract(
            &mut feature,
            &mut histories,
            &settings,
            &camera,
            2,
            (1920, 1080),
        );
        assert!(second.indirect_history.valid);
        assert!(second.source_history.valid);
        assert_eq!(
            second.uniforms.frame_params.w,
            CUSTOM_RUNTIME_BIT | HISTORY_FLAG_INDIRECT_VALID | HISTORY_FLAG_SOURCE_VALID
        );
    }

    #[test]
    fn taa_source_is_gated_by_committed_ssgi_source_history() {
        let mut feature = SsgiFeature::new();
        let mut histories = HistoryRegistry::new();
        let settings = SsgiSettings::default();
        let camera = camera(70);
        let taa_set = histories.create_set();
        let taa_desc = TextureDesc::new_2d(
            1280,
            720,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let taa_plan = histories.plan_set(taa_set, [taa_desc], 1, false);
        let taa_previous = taa_plan.previous[0];

        let first = feature.extract(
            &mut histories,
            1,
            &camera,
            1,
            1,
            &settings.uniforms,
            (1280, 720),
            1,
            Some(taa_previous),
            GlobalStateKey::new(1, 1),
        );
        assert!(first.taa_history.is_none());
        assert!(commit_indirect(&mut histories, first));
        assert!(commit_source(&mut histories, first));

        let second = feature.extract(
            &mut histories,
            1,
            &camera,
            2,
            2,
            &settings.uniforms,
            (1280, 720),
            2,
            Some(taa_previous),
            GlobalStateKey::new(1, 1),
        );
        assert_eq!(second.taa_history, Some(taa_previous));
    }

    #[test]
    fn alternating_scene_camera_domains_keep_independent_continuity() {
        let mut feature = SsgiFeature::new();
        let mut histories = HistoryRegistry::new();
        let mut settings = SsgiSettings::default();
        settings.set_checkerboard_enabled(true);
        let camera_a = camera(71);
        let camera_b = camera(72);

        let mut previous = [None, None];
        let mut sampled_phases = [0_u32; 2];
        for serial in 1..=4 {
            for (index, camera) in [&camera_a, &camera_b].into_iter().enumerate() {
                let scene_id = index as u32 + 1;
                let global_frame = 100 + serial * 2 + index as u64;
                let snapshot = feature.extract(
                    &mut histories,
                    scene_id,
                    camera,
                    serial,
                    global_frame,
                    &settings.uniforms,
                    (1280, 720),
                    global_frame as u32,
                    None,
                    GlobalStateKey::new(1, scene_id),
                );
                assert_eq!(snapshot.uniforms.frame_params.x, serial as u32);
                sampled_phases[index] |= 1 << (snapshot.uniforms.frame_params.x & 3);
                assert_eq!(snapshot.indirect_history.valid, serial > 1);
                assert_eq!(snapshot.source_history.valid, serial > 1);
                if let Some(previous) = previous[index] {
                    assert_eq!(snapshot.indirect_history.previous[0].id, previous);
                }
                previous[index] = Some(snapshot.indirect_history.next[0].id);
                assert!(commit_indirect(&mut histories, snapshot));
                assert!(commit_source(&mut histories, snapshot));
            }
            assert_ne!(previous[0], previous[1]);
        }
        assert_eq!(sampled_phases, [0b1111, 0b1111]);
    }

    #[test]
    fn odd_full_resize_and_camera_cut_reset_both_sets() {
        let mut feature = SsgiFeature::new();
        let mut histories = HistoryRegistry::new();
        let settings = SsgiSettings::default();
        let mut camera = camera(8);
        let first = extract(
            &mut feature,
            &mut histories,
            &settings,
            &camera,
            1,
            (1280, 720),
        );
        assert!(commit_indirect(&mut histories, first));
        assert!(commit_source(&mut histories, first));

        let resized = extract(
            &mut feature,
            &mut histories,
            &settings,
            &camera,
            2,
            (1280, 721),
        );
        assert_eq!(first.half_resolution, resized.half_resolution);
        assert!(!resized.indirect_history.valid);
        assert!(!resized.source_history.valid);
        assert!(commit_indirect(&mut histories, resized));
        assert!(commit_source(&mut histories, resized));

        camera.camera_cut = 1;
        let cut = extract(
            &mut feature,
            &mut histories,
            &settings,
            &camera,
            3,
            (1280, 721),
        );
        assert!(!cut.indirect_history.valid);
        assert!(!cut.source_history.valid);
        assert_ne!(cut.uniforms.frame_params.w & HISTORY_FLAG_CAMERA_CUT, 0);
    }

    #[test]
    fn abandoned_domain_gc_recreates_both_sets_as_invalid() {
        let mut feature = SsgiFeature::new();
        let mut histories = HistoryRegistry::new();
        let settings = SsgiSettings::default();
        let camera = camera(9);
        let first = feature.extract(
            &mut histories,
            1,
            &camera,
            1,
            10,
            &settings.uniforms,
            (1280, 720),
            1,
            None,
            GlobalStateKey::new(1, 1),
        );
        assert!(commit_indirect(&mut histories, first));
        assert!(commit_source(&mut histories, first));
        assert_eq!(feature.garbage_collect(&mut histories, 12, 2), 0);
        assert_eq!(feature.garbage_collect(&mut histories, 13, 2), 1);

        let replacement = feature.extract(
            &mut histories,
            1,
            &camera,
            2,
            13,
            &settings.uniforms,
            (1280, 720),
            2,
            None,
            GlobalStateKey::new(1, 1),
        );
        assert!(!replacement.indirect_history.valid);
        assert!(!replacement.source_history.valid);
        assert_ne!(
            replacement.indirect_history.commit.set_id,
            first.indirect_history.commit.set_id
        );
        assert_ne!(
            replacement.source_history.commit.set_id,
            first.source_history.commit.set_id
        );
    }

    #[test]
    fn reentry_expires_stale_domain_even_without_periodic_gc() {
        let mut feature = SsgiFeature::new();
        let mut histories = HistoryRegistry::new();
        let settings = SsgiSettings::default();
        let camera = camera(73);
        let first = feature.extract(
            &mut histories,
            1,
            &camera,
            1,
            1,
            &settings.uniforms,
            (1280, 720),
            1,
            None,
            GlobalStateKey::new(1, 1),
        );
        assert!(commit_indirect(&mut histories, first));
        assert!(commit_source(&mut histories, first));

        let replacement = feature.extract(
            &mut histories,
            1,
            &camera,
            2,
            TEMPORAL_DOMAIN_TTL_FRAMES + 2,
            &settings.uniforms,
            (1280, 720),
            2,
            None,
            GlobalStateKey::new(1, 1),
        );
        assert!(!replacement.indirect_history.valid);
        assert!(!replacement.source_history.valid);
        assert_ne!(
            replacement.indirect_history.commit.set_id,
            first.indirect_history.commit.set_id
        );
        assert_ne!(
            replacement.source_history.commit.set_id,
            first.source_history.commit.set_id
        );
    }

    #[test]
    fn raw_root_commits_neither_history_set() {
        let mut feature = SsgiFeature::new();
        let mut histories = HistoryRegistry::new();
        let settings = SsgiSettings::default();
        let camera = camera(11);
        let snapshot = extract(
            &mut feature,
            &mut histories,
            &settings,
            &camera,
            1,
            (1280, 720),
        );
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let outputs = add_graph(&mut feature, &mut graph, snapshot);
        add_root(&mut graph, outputs.raw_indirect);
        graph.compile_topology();
        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(live, ["SSGI_Raw", "SSGI_Test_Root"]);
        graph.commit_submitted_histories(&mut histories);
        assert_eq!(followup_validity(&mut histories, snapshot), (false, false));
    }

    #[test]
    fn clean_root_commits_only_indirect_history() {
        let mut feature = SsgiFeature::new();
        let mut histories = HistoryRegistry::new();
        let settings = SsgiSettings::default();
        let camera = camera(12);
        let snapshot = extract(
            &mut feature,
            &mut histories,
            &settings,
            &camera,
            1,
            (1280, 720),
        );
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let outputs = add_graph(&mut feature, &mut graph, snapshot);
        add_root(&mut graph, outputs.clean_indirect);
        graph.compile_topology();
        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(
            live,
            [
                "SSGI_Raw",
                "SSGI_Temporal",
                "SSGI_ATrous_Step_1",
                "SSGI_ATrous_Step_2",
                "SSGI_ATrous_Step_4",
                "SSGI_Test_Root",
            ]
        );
        graph.commit_submitted_histories(&mut histories);
        assert_eq!(followup_validity(&mut histories, snapshot), (true, false));
    }

    #[test]
    fn merged_root_commits_both_history_sets() {
        let mut feature = SsgiFeature::new();
        let mut histories = HistoryRegistry::new();
        let settings = SsgiSettings::default();
        let camera = camera(13);
        let snapshot = extract(
            &mut feature,
            &mut histories,
            &settings,
            &camera,
            1,
            (1280, 720),
        );
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let outputs = add_graph(&mut feature, &mut graph, snapshot);
        add_root(&mut graph, outputs.merged_color);
        graph.compile_topology();
        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(
            live,
            [
                "SSGI_Raw",
                "SSGI_Temporal",
                "SSGI_ATrous_Step_1",
                "SSGI_ATrous_Step_2",
                "SSGI_ATrous_Step_4",
                "SSGI_Merge",
                "SSGI_Test_Root",
            ]
        );
        graph.commit_submitted_histories(&mut histories);
        assert_eq!(followup_validity(&mut histories, snapshot), (true, true));

        graph.finish();

        // Building and compiling a live graph must not eagerly create GPU state.
        assert!(feature.raw.pipelines.is_empty());
        assert!(feature.temporal.pipeline_id.is_none());
        assert!(
            feature
                .atrous
                .iter()
                .all(|stage| stage.pipeline_id.is_none())
        );
        assert!(feature.merge.pipeline_id.is_none());
    }
}
