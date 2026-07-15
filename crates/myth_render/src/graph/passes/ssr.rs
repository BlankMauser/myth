//! Screen Space Reflections with graph-owned temporal history.
//!
//! SSR is a four-stage Trace → Temporal → Spatial → Merge chain. Graph build
//! captures a lightweight [`SsrSnapshot`], while every stage lazily prepares
//! only its own static GPU state after DPE. Reflection and validation metadata
//! form one logical two-bank history set and are published atomically only
//! after successful queue submission.

use std::mem::size_of;

use rustc_hash::FxHashMap;

use crate::HDR_TEXTURE_FORMAT;
use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::binding_layout::{sampler, texture, texture_2d, uniform_buffer};
use crate::core::gpu::{CommonSampler, GlobalStateKey, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::persistent::TEMPORAL_DOMAIN_TTL_FRAMES;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, HistoryPlan, HistoryRegistry, HistorySetId, PassNode,
    PrepareContext, RenderTargetOps, TemporalDomainKey, TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::buffer::CpuBuffer;
use myth_resources::ssr::SsrUniforms;
use myth_resources::uniforms::WgslStruct;
use myth_scene::camera::RenderCamera;

const SSR_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const SSR_HISTORY_FLAG_VALID: u32 = 1 << 0;
const SSR_HISTORY_FLAG_CAMERA_CUT: u32 = 1 << 1;
const SSR_RUNTIME_HISTORY_MASK: u32 = SSR_HISTORY_FLAG_VALID | SSR_HISTORY_FLAG_CAMERA_CUT;

fn ssr_trace_resolution(width: u32, height: u32, half_resolution: bool) -> (u32, u32) {
    if half_resolution {
        ((width / 2).max(1), (height / 2).max(1))
    } else {
        (width.max(1), height.max(1))
    }
}

fn blue_noise_view_dimension() -> wgpu::TextureViewDimension {
    if cfg!(feature = "advanced_noise") {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

#[must_use = "SSA Graph: consume the merged color output from SSR"]
pub struct SsrOutputs {
    pub merged_color: TextureNodeId,
    #[cfg(feature = "debug_view")]
    pub raw_reflection: TextureNodeId,
    #[cfg(feature = "debug_view")]
    pub clean_reflection: TextureNodeId,
    #[cfg(feature = "debug_view")]
    pub raw_diagnostic: Option<TextureNodeId>,
}

/// Per-frame CPU-only data captured while Composer builds the graph.
#[derive(Clone, Copy)]
pub struct SsrSnapshot {
    uniforms: SsrUniforms,
    history: HistoryPlan<2>,
    trace_resolution: (u32, u32),
    max_steps: u32,
    trace_diagnostics: bool,
    global_state_key: GlobalStateKey,
}

/// Long-lived SSR owner. Static resources are physically partitioned by stage
/// so a debug root can keep only a prefix live without preparing the suffix.
pub struct SsrFeature {
    raw: SsrRawFeature,
    temporal: SsrTemporalFeature,
    spatial: SsrSpatialFeature,
    merge: SsrMergeFeature,
    histories: FxHashMap<TemporalDomainKey, SsrHistoryDomain>,
}

#[derive(Debug, Clone, Copy)]
struct SsrHistoryDomain {
    set_id: HistorySetId,
    full_resolution: (u32, u32),
    last_planned_frame: u64,
}

#[derive(Default)]
struct SsrRawFeature {
    pipelines: FxHashMap<(u32, bool), RenderPipelineId>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

#[derive(Default)]
struct SsrTemporalFeature {
    pipeline_id: Option<RenderPipelineId>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

#[derive(Default)]
struct SsrSpatialFeature {
    pipeline_id: Option<RenderPipelineId>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

#[derive(Default)]
struct SsrMergeFeature {
    pipeline_id: Option<RenderPipelineId>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl Default for SsrFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SsrFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            raw: SsrRawFeature::default(),
            temporal: SsrTemporalFeature::default(),
            spatial: SsrSpatialFeature::default(),
            merge: SsrMergeFeature::default(),
            histories: FxHashMap::default(),
        }
    }

    /// Capture POD settings and plan logical reflection/meta history. No WGPU
    /// object is created, no buffer is uploaded, and the scene-owned
    /// [`CpuBuffer`] remains unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn extract(
        &mut self,
        histories: &mut HistoryRegistry,
        scene_id: u32,
        camera: &RenderCamera,
        temporal_serial: u64,
        source_uniforms: &CpuBuffer<SsrUniforms>,
        size: (u32, u32),
        renderer_frame: u64,
        _frame_index: u32,
        trace_diagnostics: bool,
        global_state_key: GlobalStateKey,
    ) -> SsrSnapshot {
        let history_key = TemporalDomainKey::new(scene_id, camera.camera_id);
        if self.histories.get(&history_key).is_some_and(|domain| {
            renderer_frame.saturating_sub(domain.last_planned_frame) > TEMPORAL_DOMAIN_TTL_FRAMES
        }) {
            let expired = self
                .histories
                .remove(&history_key)
                .expect("expired SSR history domain must exist");
            let removed = histories.remove_set(expired.set_id);
            debug_assert!(removed, "SSR history domain lost its registry set");
        }
        let (set_id, full_resolution_changed) =
            if let Some(domain) = self.histories.get_mut(&history_key) {
                let changed = domain.full_resolution != size;
                domain.full_resolution = size;
                domain.last_planned_frame = renderer_frame;
                (domain.set_id, changed)
            } else {
                let id = histories.create_set();
                self.histories.insert(
                    history_key,
                    SsrHistoryDomain {
                        set_id: id,
                        full_resolution: size,
                        last_planned_frame: renderer_frame,
                    },
                );
                (id, false)
            };

        let mut uniforms = *source_uniforms.read();
        let trace_resolution = ssr_trace_resolution(size.0, size.1, uniforms.denoise_params.y != 0);
        uniforms.full_resolution = glam::Vec4::new(
            size.0 as f32,
            size.1 as f32,
            1.0 / size.0.max(1) as f32,
            1.0 / size.1.max(1) as f32,
        );
        // Sampling advances with this history domain, even when cameras alternate.
        uniforms.frame_params.x = temporal_serial as u32;
        uniforms.temporal_params.z = camera.near.max(0.0001);

        let history_desc = TextureDesc::new_2d(
            trace_resolution.0,
            trace_resolution.1,
            SSR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let history = histories.plan_set(
            set_id,
            [history_desc, history_desc],
            temporal_serial,
            camera.camera_cut != 0 || full_resolution_changed,
        );

        uniforms.frame_params.w &= !SSR_RUNTIME_HISTORY_MASK;
        if history.valid {
            uniforms.frame_params.w |= SSR_HISTORY_FLAG_VALID;
        }
        if camera.camera_cut != 0 {
            uniforms.frame_params.w |= SSR_HISTORY_FLAG_CAMERA_CUT;
        }

        SsrSnapshot {
            uniforms,
            history,
            trace_resolution,
            max_steps: uniforms.frame_params.y,
            trace_diagnostics,
            global_state_key,
        }
    }

    /// Retire history for scene/camera domains that have not planned an SSR
    /// frame within `ttl` renderer frames.
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
                let removed = histories.remove_set(domain.set_id);
                debug_assert!(removed, "SSR history domain lost its registry set");
            }
            live
        });
        before - self.histories.len()
    }

    /// Emit Trace → Temporal → Spatial → Merge. Temporal renders directly into
    /// the next persistent reflection/meta bank, eliminating both archive-copy
    /// passes while keeping history publication conditional on DPE liveness.
    #[allow(clippy::too_many_arguments)]
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: SsrSnapshot,
        current_color: TextureNodeId,
        scene_depth: TextureNodeId,
        scene_hiz: TextureNodeId,
        scene_normals: TextureNodeId,
        velocity: TextureNodeId,
        material_mrt: TextureNodeId,
        specular_mrt: TextureNodeId,
    ) -> SsrOutputs {
        let raw_feature = &mut self.raw;
        let temporal_feature = &mut self.temporal;
        let spatial_feature = &mut self.spatial;
        let merge_feature = &mut self.merge;

        ctx.with_group("SSR_System", |ctx| {
            let previous_reflection = ctx.graph.import_persistent_texture(
                "SSR_History_Reflection_Previous",
                snapshot.history.previous[0],
            );
            let previous_meta = ctx.graph.import_persistent_texture(
                "SSR_History_Meta_Previous",
                snapshot.history.previous[1],
            );
            let next_reflection = ctx
                .graph
                .import_persistent_texture("SSR_History_Reflection_Next", snapshot.history.next[0]);
            let next_meta = ctx
                .graph
                .import_persistent_texture("SSR_History_Meta_Next", snapshot.history.next[1]);

            let raw_desc = TextureDesc::new_2d(
                snapshot.trace_resolution.0,
                snapshot.trace_resolution.1,
                SSR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            let uniforms_desc = BufferDesc::new(
                size_of::<SsrUniforms>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );

            let (raw_reflection, raw_diagnostic, uniforms_buffer) =
                ctx.graph.add_pass("SSR_Trace", |builder| {
                    builder.demand_global_state(snapshot.global_state_key);
                    builder.read_texture(scene_depth);
                    builder.read_texture(scene_normals);
                    builder.read_texture(current_color);
                    builder.read_texture(scene_hiz);
                    builder.read_texture(material_mrt);
                    builder.read_texture(specular_mrt);

                    let uniforms_buffer = builder.create_buffer("SSR_Uniforms", uniforms_desc);
                    let output_reflection = builder.create_texture("SSR_Raw_Reflection", raw_desc);
                    let output_diagnostic = snapshot
                        .trace_diagnostics
                        .then(|| builder.create_texture("SSR_Raw_Diagnostic", raw_desc));
                    let node = SsrTraceNode {
                        feature: raw_feature,
                        global_state_key: snapshot.global_state_key,
                        max_steps: snapshot.max_steps,
                        trace_diagnostics: snapshot.trace_diagnostics,
                        uniforms_data: snapshot.uniforms,
                        scene_depth,
                        scene_normals,
                        source_color: current_color,
                        scene_hiz,
                        material_mrt,
                        specular_mrt,
                        uniforms: uniforms_buffer,
                        output_reflection,
                        output_diagnostic,
                        pipeline_id: None,
                        bind_group: None,
                    };

                    (
                        node,
                        (output_reflection, output_diagnostic, uniforms_buffer),
                    )
                });

            ctx.graph.add_pass("SSR_Temporal", |builder| {
                builder.demand_global_state(snapshot.global_state_key);
                builder.read_texture(raw_reflection);
                builder.read_texture(previous_reflection);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                builder.read_texture(previous_meta);
                builder.read_texture(velocity);
                builder.read_texture(material_mrt);
                builder.read_buffer(uniforms_buffer);
                builder.write_texture(next_reflection);
                builder.write_texture(next_meta);

                let node = SsrTemporalNode {
                    feature: temporal_feature,
                    global_state_key: snapshot.global_state_key,
                    raw_reflection,
                    reflection_history: previous_reflection,
                    scene_depth,
                    scene_normals,
                    history_meta: previous_meta,
                    velocity,
                    material_mrt,
                    uniforms: uniforms_buffer,
                    output_reflection: next_reflection,
                    output_meta: next_meta,
                    pipeline_id: None,
                    bind_group: None,
                };

                (node, ())
            });
            ctx.graph
                .register_history_commit(snapshot.history.commit, &[next_reflection, next_meta]);

            let clean_desc = TextureDesc::new_2d(
                snapshot.trace_resolution.0,
                snapshot.trace_resolution.1,
                SSR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let clean_reflection = ctx.graph.add_pass("SSR_Resolve", |builder| {
                builder.read_texture(next_reflection);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                builder.read_texture(material_mrt);
                builder.read_buffer(uniforms_buffer);

                let output = builder.create_texture("SSR_Clean_Reflection", clean_desc);
                let node = SsrSpatialNode {
                    feature: spatial_feature,
                    input_reflection: next_reflection,
                    scene_depth,
                    scene_normals,
                    material_mrt,
                    uniforms: uniforms_buffer,
                    output,
                    pipeline_id: None,
                    bind_group: None,
                };
                (node, output)
            });

            let merged_desc = TextureDesc::new_2d(
                ctx.frame_config.width,
                ctx.frame_config.height,
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let merged_color = ctx.graph.add_pass("SSR_Merge", |builder| {
                builder.demand_global_state(snapshot.global_state_key);
                builder.read_texture(current_color);
                builder.read_texture(clean_reflection);
                builder.read_texture(material_mrt);
                builder.read_texture(specular_mrt);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                builder.read_buffer(uniforms_buffer);

                let output = builder.create_texture("SSR_Merged_Color", merged_desc);
                let node = SsrMergeNode {
                    feature: merge_feature,
                    global_state_key: snapshot.global_state_key,
                    current_color,
                    clean_reflection,
                    material_mrt,
                    specular_mrt,
                    scene_depth,
                    scene_normals,
                    uniforms: uniforms_buffer,
                    output,
                    pipeline_id: None,
                    bind_group: None,
                };
                (node, output)
            });

            #[cfg(not(feature = "debug_view"))]
            let _ = raw_diagnostic;

            SsrOutputs {
                merged_color,
                #[cfg(feature = "debug_view")]
                raw_reflection,
                #[cfg(feature = "debug_view")]
                clean_reflection,
                #[cfg(feature = "debug_view")]
                raw_diagnostic,
            }
        })
    }
}

impl SsrRawFeature {
    fn ensure_resources(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        global_state_key: GlobalStateKey,
        max_steps: u32,
        trace_diagnostics: bool,
    ) {
        let variant = (max_steps, trace_diagnostics);
        if self.pipelines.contains_key(&variant) && self.layout.is_some() {
            return;
        }

        let global_state = ctx
            .global_states
            .get(global_state_key)
            .expect("SSR trace: global binding state must exist before RDG prepare");
        let mut options = global_shader_options::<SsrUniforms>(global_state.binding_wgsl.as_str());
        let max_steps_define = max_steps.to_string();
        options.inject_code("ssr_max_steps", &max_steps_define);
        if trace_diagnostics {
            options.add_define("SSR_TRACE_DIAGNOSTICS", "1");
        }

        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/features/ssr/trace"),
            &options,
        );
        let reflection_target = color_target(SSR_TEXTURE_FORMAT);
        let key = if trace_diagnostics {
            FullscreenPipelineKey::fullscreen(
                shader_hash,
                smallvec::smallvec![reflection_target, color_target(SSR_TEXTURE_FORMAT)],
                None,
            )
        } else {
            FullscreenPipelineKey::fullscreen(
                shader_hash,
                smallvec::smallvec![reflection_target],
                None,
            )
        };

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
                        label: Some("SSR Trace Pipeline Layout"),
                        bind_group_layouts: &[Some(&global_state.layout), Some(&local_layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSR Trace Pipeline",
            );
            register_global_and_local_layouts(ctx, id, global_state, local_layout.clone());
            self.layout = Some(local_layout);
            id
        };
        self.pipelines.insert(variant, pipeline_id);
    }

    fn pipeline_id(&self, max_steps: u32, trace_diagnostics: bool) -> RenderPipelineId {
        self.pipelines
            .get(&(max_steps, trace_diagnostics))
            .copied()
            .expect("SSR trace pipeline must be prepared by the live trace node")
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("SSR trace layout must be prepared by the live trace node")
    }
}

impl SsrTemporalFeature {
    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>, global_state_key: GlobalStateKey) {
        if self.pipeline_id.is_some() && self.layout.is_some() {
            return;
        }

        let global_state = ctx
            .global_states
            .get(global_state_key)
            .expect("SSR temporal: global binding state must exist before RDG prepare");
        let options = global_shader_options::<SsrUniforms>(global_state.binding_wgsl.as_str());
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/features/ssr/temporal"),
            &options,
        );
        let key = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![
                color_target(SSR_TEXTURE_FORMAT),
                color_target(SSR_TEXTURE_FORMAT),
            ],
            None,
        );

        let pipeline_id = if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
            self.layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 1).clone());
            id
        } else {
            let local_layout = create_temporal_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSR Temporal Pipeline Layout"),
                        bind_group_layouts: &[Some(&global_state.layout), Some(&local_layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSR Temporal Pipeline",
            );
            register_global_and_local_layouts(ctx, id, global_state, local_layout.clone());
            self.layout = Some(local_layout);
            id
        };
        self.pipeline_id = Some(pipeline_id);
    }

    fn pipeline_id(&self) -> RenderPipelineId {
        self.pipeline_id
            .expect("SSR temporal pipeline must be prepared by the live temporal node")
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("SSR temporal layout must be prepared by the live temporal node")
    }
}

impl SsrSpatialFeature {
    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>) {
        if self.pipeline_id.is_some() && self.layout.is_some() {
            return;
        }

        let mut options = ShaderCompilationOptions::default();
        options.add_define(
            "struct_definitions",
            SsrUniforms::wgsl_struct_def("SsrUniforms").as_str(),
        );
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/features/ssr/spatial_filter"),
            &options,
        );
        let key = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![color_target(SSR_TEXTURE_FORMAT)],
            None,
        );

        let pipeline_id = if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
            self.layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 0).clone());
            id
        } else {
            let layout = create_spatial_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSR Spatial Pipeline Layout"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSR Spatial Pipeline",
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
            .expect("SSR spatial pipeline must be prepared by the live spatial node")
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("SSR spatial layout must be prepared by the live spatial node")
    }
}

impl SsrMergeFeature {
    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>, global_state_key: GlobalStateKey) {
        if self.pipeline_id.is_some() && self.layout.is_some() {
            return;
        }

        let global_state = ctx
            .global_states
            .get(global_state_key)
            .expect("SSR merge: global binding state must exist before RDG prepare");
        let options = global_shader_options::<SsrUniforms>(global_state.binding_wgsl.as_str());
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/features/ssr/merge"),
            &options,
        );
        let key = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![color_target(HDR_TEXTURE_FORMAT)],
            None,
        );

        let pipeline_id = if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
            self.layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 1).clone());
            id
        } else {
            let local_layout = create_merge_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSR Merge Pipeline Layout"),
                        bind_group_layouts: &[Some(&global_state.layout), Some(&local_layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSR Merge Pipeline",
            );
            register_global_and_local_layouts(ctx, id, global_state, local_layout.clone());
            self.layout = Some(local_layout);
            id
        };
        self.pipeline_id = Some(pipeline_id);
    }

    fn pipeline_id(&self) -> RenderPipelineId {
        self.pipeline_id
            .expect("SSR merge pipeline must be prepared by the live merge node")
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("SSR merge layout must be prepared by the live merge node")
    }
}

fn global_shader_options<T: WgslStruct>(binding_wgsl: &str) -> ShaderCompilationOptions {
    let mut options = ShaderCompilationOptions::default();
    options.add_define(
        "struct_definitions",
        T::wgsl_struct_def("SsrUniforms").as_str(),
    );
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
            label: Some("SSR Trace Layout"),
            entries: &[
                texture_2d(0, fragment, wgpu::TextureSampleType::Depth),
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
                texture_2d(
                    4,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(
                    5,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                sampler(6, fragment, wgpu::SamplerBindingType::Filtering),
                sampler(7, fragment, wgpu::SamplerBindingType::NonFiltering),
                uniform_buffer(8, fragment),
                texture(
                    9,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                    blue_noise_view_dimension(),
                    false,
                ),
                sampler(10, fragment, wgpu::SamplerBindingType::Filtering),
            ],
        }),
    )
}

fn create_temporal_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Temporal Layout"),
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
                texture_2d(2, fragment, wgpu::TextureSampleType::Depth),
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
                texture_2d(
                    6,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                sampler(7, fragment, wgpu::SamplerBindingType::Filtering),
                sampler(8, fragment, wgpu::SamplerBindingType::NonFiltering),
                uniform_buffer(9, fragment),
            ],
        }),
    )
}

fn create_spatial_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Spatial Layout"),
            entries: &[
                texture_2d(
                    0,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(1, fragment, wgpu::TextureSampleType::Depth),
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
            label: Some("SSR Merge Layout"),
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
                texture_2d(
                    3,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(4, fragment, wgpu::TextureSampleType::Depth),
                texture_2d(
                    5,
                    fragment,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                sampler(6, fragment, wgpu::SamplerBindingType::Filtering),
                sampler(7, fragment, wgpu::SamplerBindingType::NonFiltering),
                uniform_buffer(8, fragment),
            ],
        }),
    )
}

struct SsrTraceNode<'a> {
    feature: &'a mut SsrRawFeature,
    global_state_key: GlobalStateKey,
    max_steps: u32,
    trace_diagnostics: bool,
    uniforms_data: SsrUniforms,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    source_color: TextureNodeId,
    scene_hiz: TextureNodeId,
    material_mrt: TextureNodeId,
    specular_mrt: TextureNodeId,
    uniforms: BufferNodeId,
    output_reflection: TextureNodeId,
    output_diagnostic: Option<TextureNodeId>,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for SsrTraceNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(
            ctx,
            self.global_state_key,
            self.max_steps,
            self.trace_diagnostics,
        );
        self.pipeline_id = Some(
            self.feature
                .pipeline_id(self.max_steps, self.trace_diagnostics),
        );
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.uniforms),
            0,
            bytemuck::bytes_of(&self.uniforms_data),
        );

        let layout = self.feature.layout().clone();
        let blue_noise_view = ctx.system_textures.blue_noise.clone();
        let blue_noise_sampler = ctx.system_textures.blue_noise_sampler.clone();
        self.bind_group = Some(
            ctx.build_bind_group(&layout, Some("SSR Trace BG"))
                .bind_texture(0, self.scene_depth)
                .bind_texture(1, self.scene_normals)
                .bind_texture(2, self.source_color)
                .bind_texture(3, self.scene_hiz)
                .bind_texture(4, self.material_mrt)
                .bind_texture(5, self.specular_mrt)
                .bind_common_sampler(6, CommonSampler::LinearClamp)
                .bind_common_sampler(7, CommonSampler::NearestClamp)
                .bind_buffer(8, self.uniforms)
                .bind_tracked_texture_view(9, &blue_noise_view)
                .bind_tracked_sampler(10, &blue_noise_sampler)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let attachments = if let Some(output_diagnostic) = self.output_diagnostic {
            vec![
                ctx.get_color_attachment(self.output_reflection, RenderTargetOps::DontCare, None),
                ctx.get_color_attachment(output_diagnostic, RenderTargetOps::DontCare, None),
            ]
        } else {
            vec![ctx.get_color_attachment(self.output_reflection, RenderTargetOps::DontCare, None)]
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Trace Pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_render_pipeline(self.pipeline_id.expect("SSR trace pipeline not prepared")),
        );
        pass.set_bind_group(
            0,
            ctx.global_bind_group.expect(
                "SSR_Trace scene Global Group 0 is missing; its demand was not declared or finalized",
            ),
            &[],
        );
        pass.set_bind_group(
            1,
            ctx.resolve_prepared_bind_group(self.bind_group.expect("SSR trace BG missing")),
            &[],
        );
        pass.draw(0..3, 0..1);
    }
}

struct SsrTemporalNode<'a> {
    feature: &'a mut SsrTemporalFeature,
    global_state_key: GlobalStateKey,
    raw_reflection: TextureNodeId,
    reflection_history: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    history_meta: TextureNodeId,
    velocity: TextureNodeId,
    material_mrt: TextureNodeId,
    uniforms: BufferNodeId,
    output_reflection: TextureNodeId,
    output_meta: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for SsrTemporalNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(ctx, self.global_state_key);
        self.pipeline_id = Some(self.feature.pipeline_id());
        let layout = self.feature.layout().clone();
        self.bind_group = Some(
            crate::myth_bind_group!(ctx, &layout, Some("SSR Temporal BG"), [
                0 => self.raw_reflection,
                1 => self.reflection_history,
                2 => self.scene_depth,
                3 => self.scene_normals,
                4 => self.history_meta,
                5 => self.velocity,
                6 => self.material_mrt,
                7 => CommonSampler::LinearClamp,
                8 => CommonSampler::NearestClamp,
                9 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let attachments = [
            ctx.get_color_attachment(self.output_reflection, RenderTargetOps::DontCare, None),
            ctx.get_color_attachment(self.output_meta, RenderTargetOps::DontCare, None),
        ];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Temporal Pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(
            ctx.pipeline_cache.get_render_pipeline(
                self.pipeline_id
                    .expect("SSR temporal pipeline not prepared"),
            ),
        );
        pass.set_bind_group(
            0,
            ctx.global_bind_group.expect(
                "SSR_Temporal scene Global Group 0 is missing; its demand was not declared or finalized",
            ),
            &[],
        );
        pass.set_bind_group(
            1,
            ctx.resolve_prepared_bind_group(self.bind_group.expect("SSR temporal BG missing")),
            &[],
        );
        pass.draw(0..3, 0..1);
    }
}

struct SsrSpatialNode<'a> {
    feature: &'a mut SsrSpatialFeature,
    input_reflection: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    material_mrt: TextureNodeId,
    uniforms: BufferNodeId,
    output: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for SsrSpatialNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(ctx);
        self.pipeline_id = Some(self.feature.pipeline_id());
        let layout = self.feature.layout().clone();
        self.bind_group = Some(
            crate::myth_bind_group!(ctx, &layout, Some("SSR Spatial BG"), [
                0 => self.input_reflection,
                1 => self.scene_depth,
                2 => self.scene_normals,
                3 => self.material_mrt,
                4 => CommonSampler::LinearClamp,
                5 => CommonSampler::NearestClamp,
                6 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let output = ctx.get_color_attachment(self.output, RenderTargetOps::DontCare, None);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Spatial Pass"),
            color_attachments: &[output],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_render_pipeline(self.pipeline_id.expect("SSR spatial pipeline not prepared")),
        );
        pass.set_bind_group(
            0,
            ctx.resolve_prepared_bind_group(self.bind_group.expect("SSR spatial BG missing")),
            &[],
        );
        pass.draw(0..3, 0..1);
    }
}

struct SsrMergeNode<'a> {
    feature: &'a mut SsrMergeFeature,
    global_state_key: GlobalStateKey,
    current_color: TextureNodeId,
    clean_reflection: TextureNodeId,
    material_mrt: TextureNodeId,
    specular_mrt: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    uniforms: BufferNodeId,
    output: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for SsrMergeNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(ctx, self.global_state_key);
        self.pipeline_id = Some(self.feature.pipeline_id());
        let layout = self.feature.layout().clone();
        self.bind_group = Some(
            crate::myth_bind_group!(ctx, &layout, Some("SSR Merge BG"), [
                0 => self.current_color,
                1 => self.clean_reflection,
                2 => self.material_mrt,
                3 => self.specular_mrt,
                4 => self.scene_depth,
                5 => self.scene_normals,
                6 => CommonSampler::LinearClamp,
                7 => CommonSampler::NearestClamp,
                8 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let output = ctx.get_color_attachment(self.output, RenderTargetOps::DontCare, None);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Merge Pass"),
            color_attachments: &[output],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_render_pipeline(self.pipeline_id.expect("SSR merge pipeline not prepared")),
        );
        pass.set_bind_group(
            0,
            ctx.global_bind_group.expect(
                "SSR_Merge scene Global Group 0 is missing; its demand was not declared or finalized",
            ),
            &[],
        );
        pass.set_bind_group(
            1,
            ctx.resolve_prepared_bind_group(self.bind_group.expect("SSR merge BG missing")),
            &[],
        );
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests;
