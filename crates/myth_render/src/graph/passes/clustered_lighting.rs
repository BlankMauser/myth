//! Clustered lighting feature.
//!
//! Graph construction captures only a lightweight CPU snapshot. Persistent
//! pipeline state is physically partitioned by pass and initialized by live
//! nodes during deferred prepare; per-frame parameters use an RDG transient
//! uniform buffer so dead-pass elimination suppresses their upload entirely.

use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::Tracked;
use crate::core::gpu::binding_layout::{storage_buffer, uniform_buffer};
use crate::graph::composer::GpuLightBuffers;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{BufferDesc, BufferNodeId, ExecuteContext, PassNode, PrepareContext};
use crate::graph::extracted::ExtractedLight;
use crate::graph::passes::light_merge::LightMergeFeature;
use crate::graph::{ExtractedScene, RenderState};
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::uniforms::{
    ClusterRecord, ClusteredLightingParams, GpuLightStorage, clustered_lighting_structs_wgsl,
    scene_lighting_structs_wgsl,
};
use myth_scene::RenderCamera;
use myth_scene::light::LightKind;

/// Default cluster depth slices for forward clustered lighting.
pub const DEFAULT_CLUSTER_Z_SLICES: u32 = 24;
/// Approximate screen-space tile size in pixels.
pub const DEFAULT_CLUSTER_TILE_SIZE: u32 = 120;
/// Soft per-cluster light budget used for heatmap normalization and global
/// light-index capacity sizing.
pub const DEFAULT_MAX_LIGHTS_PER_CLUSTER: u32 = 256;
/// Workgroup width shared by the clustered light view transform pass.
pub const CLUSTER_LIGHT_VIEW_TRANSFORM_WG_SIZE: u32 = 64;
/// Fallback finite far depth when using an infinite reverse-Z camera.
pub const DEFAULT_CLUSTER_FAR_DEPTH_FALLBACK: f32 = 64.0;
/// Bit flag stored in `ClusteredLightingParams::budget.z`.
pub const CLUSTERED_LIGHTING_FLAG_ENABLED: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusteredLightingConfig {
    pub tile_size_x: u32,
    pub tile_size_y: u32,
    pub slices_z: u32,
    pub max_lights_per_cluster: u32,
}

impl Default for ClusteredLightingConfig {
    fn default() -> Self {
        Self {
            tile_size_x: DEFAULT_CLUSTER_TILE_SIZE,
            tile_size_y: DEFAULT_CLUSTER_TILE_SIZE,
            slices_z: DEFAULT_CLUSTER_Z_SLICES,
            max_lights_per_cluster: DEFAULT_MAX_LIGHTS_PER_CLUSTER,
        }
    }
}

pub struct ClusteredLightingFeature {
    pub config: ClusteredLightingConfig,
    light_merge: LightMergeFeature,
    dispatch_args: ClusterDispatchArgsFeature,
    light_view: ClusterLightViewFeature,
    cull: ClusterCullFeature,
}

/// Lightweight per-frame input captured while the Composer builds the graph.
/// No WGPU object, physical layout, or generated WGSL interface crosses the
/// graph-construction boundary.
#[derive(Clone, Copy)]
pub struct ClusteredLightingSnapshot<'a> {
    params: ClusteredLightingParams,
    global_state_key: crate::core::gpu::GlobalStateKey,
    cull_max_lights_per_cluster: u32,
    lights: &'a [ExtractedLight],
    view_matrix: glam::Mat4,
    camera_near: f32,
    camera_far: f32,
}

#[derive(Clone, Copy)]
pub struct ClusteredLightingInputs {
    pub cpu_light_metadata_buffer: BufferNodeId,
    pub cpu_light_data_buffer: BufferNodeId,
    pub injected_gpu_lights: Option<GpuLightBuffers>,
}

pub struct ClusteredLightingOutputs {
    pub final_light_metadata_buffer: BufferNodeId,
    pub final_light_data_buffer: BufferNodeId,
    pub final_indirect_count_buffer: Option<BufferNodeId>,
    pub params_buffer: BufferNodeId,
    pub cluster_records: Option<BufferNodeId>,
    pub light_indices: Option<BufferNodeId>,
}

impl Default for ClusteredLightingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusteredLightingFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ClusteredLightingConfig::default(),
            light_merge: LightMergeFeature::new(),
            dispatch_args: ClusterDispatchArgsFeature::default(),
            light_view: ClusterLightViewFeature::default(),
            cull: ClusterCullFeature::default(),
        }
    }

    #[inline]
    #[must_use]
    pub fn params_desc() -> BufferDesc {
        BufferDesc::new(
            std::mem::size_of::<ClusteredLightingParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )
    }

    #[inline]
    #[must_use]
    pub fn cluster_record_desc(params: ClusteredLightingParams) -> BufferDesc {
        BufferDesc::new(
            u64::from(params.grid_dimensions.y) * std::mem::size_of::<ClusterRecord>() as u64,
            wgpu::BufferUsages::STORAGE,
        )
    }

    #[inline]
    #[must_use]
    pub fn light_index_desc(params: ClusteredLightingParams) -> BufferDesc {
        BufferDesc::new(
            u64::from(params.budget.y) * std::mem::size_of::<u32>() as u64,
            wgpu::BufferUsages::STORAGE,
        )
    }

    #[inline]
    #[must_use]
    pub fn light_index_allocator_desc() -> BufferDesc {
        BufferDesc::new(
            std::mem::size_of::<u32>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
    }

    #[inline]
    #[must_use]
    pub fn light_view_positions_desc(light_slot_count: u32) -> BufferDesc {
        BufferDesc::new(
            u64::from(light_slot_count.max(1)) * std::mem::size_of::<[f32; 4]>() as u64,
            wgpu::BufferUsages::STORAGE,
        )
    }

    #[inline]
    #[must_use]
    pub fn dispatch_args_desc() -> BufferDesc {
        BufferDesc::new(
            std::mem::size_of::<[u32; 4]>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        )
    }

    pub fn extract<'a>(
        &self,
        render_state: &RenderState,
        render_camera: &RenderCamera,
        extracted_scene: &'a ExtractedScene,
        max_storage_buffer_binding_size: u64,
        global_state_key: crate::core::gpu::GlobalStateKey,
        enabled: bool,
        active_local_light_count: u32,
    ) -> ClusteredLightingSnapshot<'a> {
        let render_uniforms = render_state.uniforms().read();
        let width = render_uniforms.viewport.x.max(1.0) as u32;
        let height = render_uniforms.viewport.y.max(1.0) as u32;
        let near = render_uniforms.camera_near.max(0.001);
        let camera_far = render_uniforms.camera_far;
        drop(render_uniforms);

        let cluster_x = width.div_ceil(self.config.tile_size_x.max(1));
        let cluster_y = height.div_ceil(self.config.tile_size_y.max(1));
        let cluster_z = self.config.slices_z.max(1);
        let total_clusters = cluster_x
            .saturating_mul(cluster_y)
            .saturating_mul(cluster_z)
            .max(1);
        let max_indices_by_limit = (max_storage_buffer_binding_size
            / std::mem::size_of::<u32>() as u64)
            .max(u64::from(total_clusters));
        let requested_light_indices =
            u64::from(total_clusters) * u64::from(self.config.max_lights_per_cluster.max(1));
        let effective_light_indices = requested_light_indices.min(max_indices_by_limit);
        let effective_max_lights =
            (effective_light_indices / u64::from(total_clusters)).max(1) as u32;

        if effective_max_lights < self.config.max_lights_per_cluster {
            log::warn!(
                "Clustered lighting compact light-index capacity reduced from an average of {} to {} lights per cluster by maxStorageBufferBindingSize; extremely dense scenes may still overflow the global index list",
                self.config.max_lights_per_cluster,
                effective_max_lights,
            );
        }

        let params = ClusteredLightingParams {
            screen_dimensions: glam::UVec4::new(width, height, cluster_x, cluster_y),
            grid_dimensions: glam::UVec4::new(
                cluster_z,
                total_clusters,
                self.config.tile_size_x.max(1),
                self.config.tile_size_y.max(1),
            ),
            budget: glam::UVec4::new(
                effective_max_lights,
                effective_light_indices.max(1) as u32,
                if enabled {
                    CLUSTERED_LIGHTING_FLAG_ENABLED
                } else {
                    0
                },
                active_local_light_count,
            ),
            // The depth range requires an O(light_count) derivation for
            // infinite-far cameras. Keep it unresolved until a live params
            // upload node reaches deferred prepare.
            depth_params: glam::Vec4::ZERO,
        };

        ClusteredLightingSnapshot {
            params,
            global_state_key,
            cull_max_lights_per_cluster: self.config.max_lights_per_cluster.max(1),
            lights: &extracted_scene.lights,
            view_matrix: render_camera.view_matrix,
            camera_near: near,
            camera_far,
        }
    }

    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: ClusteredLightingSnapshot<'a>,
        inputs: ClusteredLightingInputs,
    ) -> ClusteredLightingOutputs {
        let params_buffer = ctx.graph.add_pass("Cluster_Params_Upload", |builder| {
            let params_buffer = builder.create_buffer("Clustered_Params", Self::params_desc());
            (
                ClusterParamsPassNode {
                    params_buffer,
                    snapshot,
                },
                params_buffer,
            )
        });

        let Self {
            light_merge: light_merge_feature,
            dispatch_args: dispatch_args_feature,
            light_view: light_view_feature,
            cull: cull_feature,
            ..
        } = self;

        let cpu_local_light_count = ctx.scene_local_light_count();
        let cpu_light_capacity =
            light_capacity_from_desc(ctx.buffer_desc(inputs.cpu_light_data_buffer))
                .max(cpu_local_light_count);
        let final_lights = match inputs.injected_gpu_lights {
            Some(gpu_lights) if cpu_local_light_count == 0 => ClusteredLightingOutputs {
                final_light_metadata_buffer: gpu_lights.light_metadata,
                final_light_data_buffer: gpu_lights.light_storage,
                final_indirect_count_buffer: gpu_lights.indirect_count_buffer,
                params_buffer,
                cluster_records: None,
                light_indices: None,
            },
            Some(gpu_lights) => {
                let merged = light_merge_feature.add_to_graph(
                    ctx,
                    inputs.cpu_light_metadata_buffer,
                    inputs.cpu_light_data_buffer,
                    cpu_light_capacity,
                    gpu_lights,
                );
                ClusteredLightingOutputs {
                    final_light_metadata_buffer: merged.light_metadata,
                    final_light_data_buffer: merged.light_storage,
                    final_indirect_count_buffer: Some(merged.indirect_count_buffer),
                    params_buffer,
                    cluster_records: None,
                    light_indices: None,
                }
            }
            None => ClusteredLightingOutputs {
                final_light_metadata_buffer: inputs.cpu_light_metadata_buffer,
                final_light_data_buffer: inputs.cpu_light_data_buffer,
                final_indirect_count_buffer: None,
                params_buffer,
                cluster_records: None,
                light_indices: None,
            },
        };

        if snapshot.params.budget.z & CLUSTERED_LIGHTING_FLAG_ENABLED == 0 {
            return final_lights;
        }

        let light_slots =
            light_capacity_from_desc(ctx.buffer_desc(final_lights.final_light_data_buffer));

        let dispatch_args_buffer = final_lights
            .final_indirect_count_buffer
            .map(|count_buffer| {
                ctx.graph
                    .add_pass("Cluster_Light_Dispatch_Args", |builder| {
                        let count_buffer = builder.read_buffer(count_buffer);
                        let dispatch_args = builder.create_buffer(
                            "Cluster_Light_Dispatch_Args",
                            Self::dispatch_args_desc(),
                        );

                        let node = ClusterDispatchArgsPassNode {
                            feature: dispatch_args_feature,
                            count_buffer,
                            dispatch_args,
                            bind_group: None,
                        };

                        (node, dispatch_args)
                    })
            });

        let light_view_positions = ctx.graph.add_pass("Cluster_Light_View_Pass", |builder| {
            builder.demand_global_state(snapshot.global_state_key);
            let light_metadata_buffer =
                builder.read_buffer(final_lights.final_light_metadata_buffer);
            let light_data_buffer = builder.read_buffer(final_lights.final_light_data_buffer);
            if let Some(dispatch_args) = dispatch_args_buffer {
                builder.read_buffer(dispatch_args);
            }
            let light_view_positions = builder.create_buffer(
                "Cluster_Light_View_Positions",
                Self::light_view_positions_desc(light_slots),
            );

            let node = ClusterLightViewPassNode {
                feature: light_view_feature,
                global_state_key: snapshot.global_state_key,
                light_metadata_buffer,
                light_data_buffer,
                light_view_positions,
                indirect_dispatch_buffer: dispatch_args_buffer,
                bind_group: None,
                total_light_slots: light_slots,
            };

            (node, light_view_positions)
        });

        ctx.graph.add_pass("Cluster_Cull_Pass", |builder| {
            builder.demand_global_state(snapshot.global_state_key);
            let params_buffer = builder.read_buffer(params_buffer);
            let light_metadata_buffer =
                builder.read_buffer(final_lights.final_light_metadata_buffer);
            let light_view_positions = builder.read_buffer(light_view_positions);
            let cluster_records = builder.create_buffer(
                "Cluster_Records",
                Self::cluster_record_desc(snapshot.params),
            );
            let light_indices = builder.create_buffer(
                "Cluster_Light_Indices",
                Self::light_index_desc(snapshot.params),
            );
            let light_index_allocator = builder.create_buffer(
                "Cluster_Light_Index_Allocator",
                Self::light_index_allocator_desc(),
            );

            let node = ClusterCullPassNode {
                feature: cull_feature,
                global_state_key: snapshot.global_state_key,
                resource_key: ClusterCullResourceKey::new(snapshot.cull_max_lights_per_cluster),
                params_buffer,
                light_metadata_buffer,
                light_view_positions,
                cluster_records,
                light_indices,
                light_index_allocator,
                bind_group: None,
                dispatch_grid_x: snapshot.params.screen_dimensions.z.max(1),
                dispatch_grid_y: snapshot.params.screen_dimensions.w.max(1),
                dispatch_grid_z: snapshot.params.grid_dimensions.x.max(1),
            };
            (
                node,
                ClusteredLightingOutputs {
                    final_light_metadata_buffer: final_lights.final_light_metadata_buffer,
                    final_light_data_buffer: final_lights.final_light_data_buffer,
                    final_indirect_count_buffer: final_lights.final_indirect_count_buffer,
                    params_buffer,
                    cluster_records: Some(cluster_records),
                    light_indices: Some(light_indices),
                },
            )
        })
    }
}

fn estimate_cluster_light_max_depth(
    lights: &[ExtractedLight],
    view_matrix: glam::Mat4,
) -> Option<f32> {
    let mut max_depth = 0.0_f32;

    for light in lights {
        let range = match &light.kind {
            LightKind::Point(point) => point.range,
            LightKind::Spot(spot) => spot.range,
            LightKind::Directional(_) => 0.0,
        };

        if range <= 0.0 {
            continue;
        }

        let view_center = view_matrix * light.position.extend(1.0);
        let far_depth = -view_center.z + range;
        if far_depth.is_finite() {
            max_depth = max_depth.max(far_depth);
        }
    }

    (max_depth > 0.0).then_some(max_depth)
}

fn resolve_cluster_params_for_upload(
    mut params: ClusteredLightingParams,
    lights: &[ExtractedLight],
    view_matrix: glam::Mat4,
    camera_near: f32,
    camera_far: f32,
) -> ClusteredLightingParams {
    let near = camera_near.max(0.001);
    let estimated_light_depth = if camera_far.is_finite() {
        None
    } else {
        estimate_cluster_light_max_depth(lights, view_matrix)
    };
    let far = resolve_cluster_far_depth(near, camera_far, estimated_light_depth);
    let cluster_z = params.grid_dimensions.x.max(1);
    let log_ratio = (far / near).ln().max(0.0001);
    let slice_scale = cluster_z as f32 / log_ratio;
    let slice_bias = -(near.ln() * slice_scale);
    params.depth_params = glam::Vec4::new(near, far, slice_scale, slice_bias);
    params
}

fn resolve_cluster_far_depth(
    near: f32,
    camera_far: f32,
    estimated_light_depth: Option<f32>,
) -> f32 {
    if camera_far.is_finite() {
        return camera_far.max(near + 0.001);
    }

    estimated_light_depth
        .unwrap_or(near + DEFAULT_CLUSTER_FAR_DEPTH_FALLBACK)
        .max(near + DEFAULT_CLUSTER_FAR_DEPTH_FALLBACK)
}

#[derive(Default)]
struct ClusterDispatchArgsFeature {
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
    pipeline_id: Option<ComputePipelineId>,
}

impl ClusterDispatchArgsFeature {
    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>) {
        if self.layout.is_some() && self.pipeline_id.is_some() {
            return;
        }

        let options = ShaderCompilationOptions::default();
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/clustered/cluster_dispatch_args"),
            &options,
        );
        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let pipeline_key =
            ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options);

        let (pipeline_id, layout) =
            if let Some(id) = ctx.pipeline_cache.get_compute_pipeline_id(&pipeline_key) {
                (
                    id,
                    ctx.pipeline_cache.get_tracked_compute_layout(id, 0).clone(),
                )
            } else {
                let layout = create_dispatch_args_layout(ctx.device);
                let pipeline_layout =
                    ctx.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("Cluster Dispatch Args Pipeline Layout"),
                            bind_group_layouts: &[Some(&layout)],
                            immediate_size: 0,
                        });
                let id = ctx.pipeline_cache.get_or_create_compute(
                    ctx.device,
                    module,
                    &pipeline_layout,
                    &pipeline_key,
                    &compilation_options,
                    "Cluster Dispatch Args Pipeline",
                );
                ctx.pipeline_cache
                    .register_compute_layouts(id, smallvec::smallvec![layout.clone()]);
                (id, layout)
            };

        self.layout = Some(layout);
        self.pipeline_id = Some(pipeline_id);
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("cluster dispatch-args layout must be prepared by the live node")
    }

    fn pipeline_id(&self) -> ComputePipelineId {
        self.pipeline_id
            .expect("cluster dispatch-args pipeline must be prepared by the live node")
    }
}

#[derive(Default)]
struct ClusterLightViewFeature {
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
    pipeline_id: Option<ComputePipelineId>,
}

impl ClusterLightViewFeature {
    fn ensure_resources(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        global_state_key: crate::core::gpu::GlobalStateKey,
    ) {
        if self.layout.is_some() && self.pipeline_id.is_some() {
            return;
        }

        let global_state = ctx
            .global_states
            .get(global_state_key)
            .expect("cluster light-view requires prepared global binding state");
        let mut options = ShaderCompilationOptions::default();
        options.inject_code("binding_code", global_state.binding_wgsl.as_str());
        options.inject_code("scene_lighting_structs", scene_lighting_structs_wgsl());
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/clustered/cluster_light_view_transform"),
            &options,
        );
        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let pipeline_key =
            ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options);

        let (pipeline_id, layout) = if let Some(id) =
            ctx.pipeline_cache.get_compute_pipeline_id(&pipeline_key)
        {
            (
                id,
                ctx.pipeline_cache.get_tracked_compute_layout(id, 1).clone(),
            )
        } else {
            let layout = create_light_view_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Cluster Light View Pipeline Layout"),
                        bind_group_layouts: &[Some(&global_state.layout), Some(&layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_compute(
                ctx.device,
                module,
                &pipeline_layout,
                &pipeline_key,
                &compilation_options,
                "Cluster Light View Pipeline",
            );
            let global_layout =
                Tracked::with_id(global_state.layout.clone(), global_state.layout_id);
            ctx.pipeline_cache
                .register_compute_layouts(id, smallvec::smallvec![global_layout, layout.clone()]);
            (id, layout)
        };

        self.layout = Some(layout);
        self.pipeline_id = Some(pipeline_id);
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("cluster light-view layout must be prepared by the live node")
    }

    fn pipeline_id(&self) -> ComputePipelineId {
        self.pipeline_id
            .expect("cluster light-view pipeline must be prepared by the live node")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClusterCullResourceKey {
    max_lights_per_cluster: u32,
}

impl ClusterCullResourceKey {
    const fn new(max_lights_per_cluster: u32) -> Self {
        Self {
            max_lights_per_cluster,
        }
    }
}

#[derive(Default)]
struct ClusterCullFeature {
    resource_key: Option<ClusterCullResourceKey>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
    pipeline_id: Option<ComputePipelineId>,
}

impl ClusterCullFeature {
    fn ensure_resources(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        global_state_key: crate::core::gpu::GlobalStateKey,
        resource_key: ClusterCullResourceKey,
    ) {
        if self.resource_key == Some(resource_key)
            && self.layout.is_some()
            && self.pipeline_id.is_some()
        {
            return;
        }

        let global_state = ctx
            .global_states
            .get(global_state_key)
            .expect("cluster cull requires prepared global binding state");
        let mut options = ShaderCompilationOptions::default();
        options.inject_code("binding_code", global_state.binding_wgsl.as_str());
        options.inject_code("scene_lighting_structs", scene_lighting_structs_wgsl());
        options.inject_code(
            "clustered_lighting_structs",
            clustered_lighting_structs_wgsl(),
        );
        options.inject_code(
            "clustered_max_local_lights",
            format!("{}u", resource_key.max_lights_per_cluster),
        );
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/clustered/cluster_cull"),
            &options,
        );
        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let pipeline_key =
            ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options);

        let (pipeline_id, layout) = if let Some(id) =
            ctx.pipeline_cache.get_compute_pipeline_id(&pipeline_key)
        {
            (
                id,
                ctx.pipeline_cache.get_tracked_compute_layout(id, 1).clone(),
            )
        } else {
            let layout = create_cull_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Cluster Cull Pipeline Layout"),
                        bind_group_layouts: &[Some(&global_state.layout), Some(&layout)],
                        immediate_size: 0,
                    });
            let id = ctx.pipeline_cache.get_or_create_compute(
                ctx.device,
                module,
                &pipeline_layout,
                &pipeline_key,
                &compilation_options,
                "Cluster Cull Pipeline",
            );
            let global_layout =
                Tracked::with_id(global_state.layout.clone(), global_state.layout_id);
            ctx.pipeline_cache
                .register_compute_layouts(id, smallvec::smallvec![global_layout, layout.clone()]);
            (id, layout)
        };

        self.resource_key = Some(resource_key);
        self.layout = Some(layout);
        self.pipeline_id = Some(pipeline_id);
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("cluster cull layout must be prepared by the live node")
    }

    fn pipeline_id(&self) -> ComputePipelineId {
        self.pipeline_id
            .expect("cluster cull pipeline must be prepared by the live node")
    }
}

fn create_dispatch_args_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cluster Dispatch Args Layout"),
            entries: &[
                storage_buffer(0, wgpu::ShaderStages::COMPUTE, true),
                storage_buffer(1, wgpu::ShaderStages::COMPUTE, false),
            ],
        }),
    )
}

fn create_light_view_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cluster Light View Layout"),
            entries: &[
                uniform_buffer(0, wgpu::ShaderStages::COMPUTE),
                storage_buffer(1, wgpu::ShaderStages::COMPUTE, true),
                storage_buffer(2, wgpu::ShaderStages::COMPUTE, false),
            ],
        }),
    )
}

fn create_cull_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cluster Cull Layout"),
            entries: &[
                uniform_buffer(0, wgpu::ShaderStages::COMPUTE),
                storage_buffer(1, wgpu::ShaderStages::COMPUTE, false),
                storage_buffer(2, wgpu::ShaderStages::COMPUTE, false),
                storage_buffer(3, wgpu::ShaderStages::COMPUTE, false),
                uniform_buffer(4, wgpu::ShaderStages::COMPUTE),
                storage_buffer(5, wgpu::ShaderStages::COMPUTE, true),
            ],
        }),
    )
}

struct ClusterParamsPassNode<'a> {
    params_buffer: BufferNodeId,
    snapshot: ClusteredLightingSnapshot<'a>,
}

impl<'a> PassNode<'a> for ClusterParamsPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let params = resolve_cluster_params_for_upload(
            self.snapshot.params,
            self.snapshot.lights,
            self.snapshot.view_matrix,
            self.snapshot.camera_near,
            self.snapshot.camera_far,
        );
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.params_buffer),
            0,
            bytemuck::bytes_of(&params),
        );
    }

    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
}

struct ClusterDispatchArgsPassNode<'a> {
    feature: &'a mut ClusterDispatchArgsFeature,
    count_buffer: BufferNodeId,
    dispatch_args: BufferNodeId,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for ClusterDispatchArgsPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(ctx);
        self.bind_group = Some(
            ctx.build_bind_group(self.feature.layout(), Some("Cluster Dispatch Args BG"))
                .bind_buffer(0, self.count_buffer)
                .bind_buffer(1, self.dispatch_args)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cluster Dispatch Args Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_compute_pipeline(self.feature.pipeline_id()),
        );
        pass.set_bind_group(
            0,
            ctx.resolve_prepared_bind_group(
                self.bind_group.expect("Cluster dispatch args BG missing"),
            ),
            &[],
        );
        pass.dispatch_workgroups(1, 1, 1);
    }
}

struct ClusterLightViewPassNode<'a> {
    feature: &'a mut ClusterLightViewFeature,
    global_state_key: crate::core::gpu::GlobalStateKey,
    light_metadata_buffer: BufferNodeId,
    light_data_buffer: BufferNodeId,
    light_view_positions: BufferNodeId,
    indirect_dispatch_buffer: Option<BufferNodeId>,
    bind_group: Option<PreparedBindGroupId>,
    total_light_slots: u32,
}

impl<'a> PassNode<'a> for ClusterLightViewPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(ctx, self.global_state_key);
        self.bind_group = Some(
            ctx.build_bind_group(self.feature.layout(), Some("Cluster Light View BG"))
                .bind_buffer(0, self.light_metadata_buffer)
                .bind_buffer(1, self.light_data_buffer)
                .bind_buffer(2, self.light_view_positions)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cluster Light View Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_compute_pipeline(self.feature.pipeline_id()),
        );
        pass.set_bind_group(
            0,
            ctx.global_bind_group.expect(
                "Cluster_Light_View_Pass scene Global Group 0 is missing; its demand was not declared or finalized",
            ),
            &[],
        );
        pass.set_bind_group(
            1,
            ctx.resolve_prepared_bind_group(
                self.bind_group.expect("Cluster light view BG missing"),
            ),
            &[],
        );
        if let Some(indirect_dispatch_buffer) = self.indirect_dispatch_buffer {
            pass.dispatch_workgroups_indirect(ctx.get_buffer(indirect_dispatch_buffer), 0);
        } else {
            pass.dispatch_workgroups(
                self.total_light_slots
                    .div_ceil(CLUSTER_LIGHT_VIEW_TRANSFORM_WG_SIZE),
                1,
                1,
            );
        }
    }
}

struct ClusterCullPassNode<'a> {
    feature: &'a mut ClusterCullFeature,
    global_state_key: crate::core::gpu::GlobalStateKey,
    resource_key: ClusterCullResourceKey,
    params_buffer: BufferNodeId,
    light_metadata_buffer: BufferNodeId,
    light_view_positions: BufferNodeId,
    cluster_records: BufferNodeId,
    light_indices: BufferNodeId,
    light_index_allocator: BufferNodeId,
    bind_group: Option<PreparedBindGroupId>,
    dispatch_grid_x: u32,
    dispatch_grid_y: u32,
    dispatch_grid_z: u32,
}

impl<'a> PassNode<'a> for ClusterCullPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature
            .ensure_resources(ctx, self.global_state_key, self.resource_key);

        self.bind_group = Some(
            ctx.build_bind_group(self.feature.layout(), Some("Cluster Cull BG"))
                .bind_buffer(0, self.params_buffer)
                .bind_buffer(1, self.cluster_records)
                .bind_buffer(2, self.light_indices)
                .bind_buffer(3, self.light_index_allocator)
                .bind_buffer(4, self.light_metadata_buffer)
                .bind_buffer(5, self.light_view_positions)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        // Clear the light index allocator to zero before culling so that the first cluster starts with a clean slate;
        encoder.clear_buffer(ctx.get_buffer(self.light_index_allocator), 0, None);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cluster Cull Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_compute_pipeline(self.feature.pipeline_id()),
        );
        pass.set_bind_group(
            0,
            ctx.global_bind_group.expect(
                "Cluster_Cull_Pass scene Global Group 0 is missing; its demand was not declared or finalized",
            ),
            &[],
        );
        pass.set_bind_group(
            1,
            ctx.resolve_prepared_bind_group(self.bind_group.expect("Cluster cull BG missing")),
            &[],
        );
        pass.dispatch_workgroups(
            self.dispatch_grid_x,
            self.dispatch_grid_y,
            self.dispatch_grid_z,
        );
    }
}

fn light_capacity_from_desc(desc: BufferDesc) -> u32 {
    let bytes_per_light = std::mem::size_of::<GpuLightStorage>() as u64;
    ((desc.logical_size / bytes_per_light).max(1)).min(u64::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::GlobalStateKey;
    use crate::graph::core::graph::FrameConfig;
    use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};
    use myth_resources::uniforms::LightBufferMetadata;

    struct TestNode;

    impl PassNode<'_> for TestNode {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    fn frame_config() -> FrameConfig {
        FrameConfig {
            width: 1280,
            height: 720,
            depth_format: wgpu::TextureFormat::Depth32Float,
            msaa_samples: 1,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            hdr_format: crate::HDR_TEXTURE_FORMAT,
        }
    }

    fn snapshot(
        enabled: bool,
        active_local_light_count: u32,
    ) -> ClusteredLightingSnapshot<'static> {
        let cluster_x = 11;
        let cluster_y = 6;
        let cluster_z = DEFAULT_CLUSTER_Z_SLICES;
        let total_clusters = cluster_x * cluster_y * cluster_z;
        ClusteredLightingSnapshot {
            params: ClusteredLightingParams {
                screen_dimensions: glam::UVec4::new(1280, 720, cluster_x, cluster_y),
                grid_dimensions: glam::UVec4::new(
                    cluster_z,
                    total_clusters,
                    DEFAULT_CLUSTER_TILE_SIZE,
                    DEFAULT_CLUSTER_TILE_SIZE,
                ),
                budget: glam::UVec4::new(
                    DEFAULT_MAX_LIGHTS_PER_CLUSTER,
                    total_clusters * DEFAULT_MAX_LIGHTS_PER_CLUSTER,
                    u32::from(enabled) * CLUSTERED_LIGHTING_FLAG_ENABLED,
                    active_local_light_count,
                ),
                depth_params: glam::Vec4::ZERO,
            },
            global_state_key: GlobalStateKey::new(1, 1),
            cull_max_lights_per_cluster: DEFAULT_MAX_LIGHTS_PER_CLUSTER,
            lights: &[],
            view_matrix: glam::Mat4::IDENTITY,
            camera_near: 0.1,
            camera_far: f32::INFINITY,
        }
    }

    fn metadata_desc() -> BufferDesc {
        BufferDesc::new(
            std::mem::size_of::<LightBufferMetadata>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
        )
    }

    fn light_storage_desc(capacity: u32) -> BufferDesc {
        BufferDesc::new(
            u64::from(capacity) * std::mem::size_of::<GpuLightStorage>() as u64,
            wgpu::BufferUsages::STORAGE,
        )
    }

    fn indirect_count_desc() -> BufferDesc {
        BufferDesc::new(
            std::mem::size_of::<u32>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        )
    }

    fn final_lights_live_passes(clustered_enabled: bool) -> Vec<&'static str> {
        let mut feature = ClusteredLightingFeature::new();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let cpu_metadata = graph.register_buffer("CPU_Metadata", metadata_desc(), true);
        let cpu_lights = graph.register_buffer("CPU_Lights", light_storage_desc(4), true);
        let gpu_metadata = graph.register_buffer("GPU_Metadata", metadata_desc(), true);
        let gpu_lights = graph.register_buffer("GPU_Lights", light_storage_desc(8), true);
        let gpu_count = graph.register_buffer("GPU_Count", indirect_count_desc(), true);
        let config = frame_config();
        let outputs = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 1);
            feature.add_to_graph(
                &mut ctx,
                snapshot(clustered_enabled, 1),
                ClusteredLightingInputs {
                    cpu_light_metadata_buffer: cpu_metadata,
                    cpu_light_data_buffer: cpu_lights,
                    injected_gpu_lights: Some(GpuLightBuffers {
                        light_metadata: gpu_metadata,
                        light_storage: gpu_lights,
                        indirect_count_buffer: Some(gpu_count),
                    }),
                },
            )
        };

        graph.add_pass("Cluster_Test_Root", |builder| {
            builder.read_buffer(outputs.final_light_metadata_buffer);
            builder.read_buffer(outputs.final_light_data_buffer);
            builder.mark_side_effect();
            (TestNode, ())
        });
        graph.compile_topology();

        graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect()
    }

    #[test]
    fn infinite_camera_far_resolves_to_finite_cluster_depth() {
        let far = resolve_cluster_far_depth(0.1, f32::INFINITY, Some(48.0));
        assert!(far.is_finite());
        assert!(far > 48.0);
    }

    #[test]
    fn infinite_camera_far_uses_fallback_when_scene_depth_missing() {
        let near = 0.25;
        let far = resolve_cluster_far_depth(near, f32::INFINITY, None);
        assert!(far.is_finite());
        assert!(far >= near + DEFAULT_CLUSTER_FAR_DEPTH_FALLBACK);
    }

    #[test]
    fn upload_derivation_resolves_depth_without_changing_topology_fields() {
        let snapshot = snapshot(true, 3);

        let resolved = resolve_cluster_params_for_upload(
            snapshot.params,
            snapshot.lights,
            snapshot.view_matrix,
            snapshot.camera_near,
            snapshot.camera_far,
        );

        assert_eq!(
            resolved.screen_dimensions,
            snapshot.params.screen_dimensions
        );
        assert_eq!(resolved.grid_dimensions, snapshot.params.grid_dimensions);
        assert_eq!(resolved.budget, snapshot.params.budget);
        assert_eq!(
            ClusteredLightingFeature::cluster_record_desc(resolved),
            ClusteredLightingFeature::cluster_record_desc(snapshot.params)
        );
        assert_eq!(
            ClusteredLightingFeature::light_index_desc(resolved),
            ClusteredLightingFeature::light_index_desc(snapshot.params)
        );
        assert_eq!(resolved.depth_params.x, snapshot.camera_near);
        assert!(resolved.depth_params.y.is_finite());
        assert!(resolved.depth_params.y > resolved.depth_params.x);
        assert!(resolved.depth_params.z.is_finite());
        assert!(resolved.depth_params.w.is_finite());
    }

    #[test]
    fn cluster_outputs_keep_params_upload_light_view_and_cull_live() {
        let mut feature = ClusteredLightingFeature::new();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let cpu_metadata = graph.register_buffer("CPU_Metadata", metadata_desc(), true);
        let cpu_lights = graph.register_buffer("CPU_Lights", light_storage_desc(4), true);
        let config = frame_config();
        let outputs = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 1);
            feature.add_to_graph(
                &mut ctx,
                snapshot(true, 1),
                ClusteredLightingInputs {
                    cpu_light_metadata_buffer: cpu_metadata,
                    cpu_light_data_buffer: cpu_lights,
                    injected_gpu_lights: None,
                },
            )
        };

        graph.add_pass("Cluster_Test_Root", |builder| {
            builder.read_buffer(outputs.params_buffer);
            builder.read_buffer(outputs.cluster_records.expect("cluster records missing"));
            builder.read_buffer(outputs.light_indices.expect("light indices missing"));
            builder.mark_side_effect();
            (TestNode, ())
        });
        graph.compile_topology();

        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert!(live.contains(&"Cluster_Params_Upload"));
        assert!(live.contains(&"Cluster_Light_View_Pass"));
        assert!(live.contains(&"Cluster_Cull_Pass"));

        let params_resource = graph
            .storage
            .resources
            .iter()
            .find(|resource| resource.name == "Clustered_Params")
            .expect("cluster params resource missing");
        let params_producer = params_resource
            .producer
            .map(|index| graph.storage.passes[index].name);
        assert_eq!(params_producer, Some("Cluster_Params_Upload"));
    }

    #[test]
    fn final_lights_only_culls_cluster_suffix() {
        let live = final_lights_live_passes(true);
        assert_eq!(live, ["Merge_Local_Lights", "Cluster_Test_Root"]);
    }

    #[test]
    fn merge_stays_live_when_clustering_is_disabled() {
        assert_eq!(
            final_lights_live_passes(false),
            ["Merge_Local_Lights", "Cluster_Test_Root"]
        );
    }
}
