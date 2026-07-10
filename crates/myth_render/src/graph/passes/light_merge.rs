use crate::core::gpu::Tracked;
use crate::core::gpu::binding_layout::{storage_buffer, uniform_buffer};
use crate::graph::composer::{GpuLightBuffers, GraphBuilderContext};
use crate::graph::core::{BufferDesc, BufferNodeId, ExecuteContext, PassNode, PrepareContext};
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::material::ShaderTemplateMode;
use myth_resources::uniforms::{GpuLightStorage, LightBufferMetadata, scene_lighting_structs_wgsl};

pub const LIGHT_MERGE_WG_SIZE: u32 = 64;

const LIGHT_MERGE_SHADER_TEMPLATE: &str = r"
{{ scene_lighting_structs }}

@group(0) @binding(0) var<uniform> u_cpu_light_metadata: LightBufferMetadata;
@group(0) @binding(1) var<storage, read> st_cpu_lights: array<Struct_lights>;
@group(0) @binding(2) var<uniform> u_gpu_light_metadata: LightBufferMetadata;
@group(0) @binding(3) var<storage, read> st_gpu_lights: array<Struct_lights>;
@group(0) @binding(4) var<storage, read_write> st_merged_light_metadata: LightBufferMetadata;
@group(0) @binding(5) var<storage, read_write> st_merged_lights: array<Struct_lights>;
@group(0) @binding(6) var<storage, read_write> st_merged_indirect_count: array<u32>;

@compute @workgroup_size({{ light_merge_workgroup_size }})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let raw_cpu_count = u_cpu_light_metadata.total_light_count;
    let cpu_count = min(raw_cpu_count, arrayLength(&st_cpu_lights));
    let raw_gpu_count = u_gpu_light_metadata.total_light_count;
    let gpu_count = min(raw_gpu_count, arrayLength(&st_gpu_lights));
    let total_count = min(cpu_count + gpu_count, arrayLength(&st_merged_lights));

    if (index == 0u) {
        st_merged_light_metadata.total_light_count = total_count;
        st_merged_light_metadata.active_local_light_count = total_count;
        st_merged_light_metadata.reserved_0 = 0u;
        st_merged_light_metadata.reserved_1 = 0u;
        st_merged_indirect_count[0] = total_count;
    }

    if (index >= total_count) {
        return;
    }

    if (index < cpu_count) {
        st_merged_lights[index] = st_cpu_lights[index];
        return;
    }

    let gpu_index = index - cpu_count;
    st_merged_lights[index] = st_gpu_lights[gpu_index];
}
";

#[derive(Clone, Copy)]
pub struct LightMergePassOutputs {
    pub light_metadata: BufferNodeId,
    pub light_storage: BufferNodeId,
    pub indirect_count_buffer: BufferNodeId,
}

pub struct LightMergeFeature {
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
    pipeline: Option<ComputePipelineId>,
}

impl Default for LightMergeFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl LightMergeFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            layout: None,
            pipeline: None,
        }
    }

    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>) {
        if self.layout.is_some() && self.pipeline.is_some() {
            return;
        }

        let mut options = ShaderCompilationOptions::default();
        options.inject_code("scene_lighting_structs", scene_lighting_structs_wgsl());
        options.inject_code(
            "light_merge_workgroup_size",
            LIGHT_MERGE_WG_SIZE.to_string(),
        );

        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::Inline {
                name: "entry/utility/clustered/light_merge",
                source: LIGHT_MERGE_SHADER_TEMPLATE,
                mode: ShaderTemplateMode::Template,
            },
            &options,
        );

        let pipeline_key =
            ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options);
        let (pipeline_id, layout) =
            if let Some(id) = ctx.pipeline_cache.get_compute_pipeline_id(&pipeline_key) {
                (
                    id,
                    ctx.pipeline_cache.get_tracked_compute_layout(id, 0).clone(),
                )
            } else {
                let layout = create_light_merge_layout(ctx.device);
                let pipeline_layout =
                    ctx.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("Light Merge Pipeline Layout"),
                            bind_group_layouts: &[Some(&layout)],
                            immediate_size: 0,
                        });
                let id = ctx.pipeline_cache.get_or_create_compute(
                    ctx.device,
                    module,
                    &pipeline_layout,
                    &pipeline_key,
                    &compilation_options,
                    "Light Merge Pipeline",
                );
                ctx.pipeline_cache
                    .register_compute_layouts(id, smallvec::smallvec![layout.clone()]);
                (id, layout)
            };

        self.layout = Some(layout);
        self.pipeline = Some(pipeline_id);
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("light merge layout must be prepared by the live merge node")
    }

    fn pipeline_id(&self) -> ComputePipelineId {
        self.pipeline
            .expect("light merge pipeline must be prepared by the live merge node")
    }

    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        cpu_light_metadata: BufferNodeId,
        cpu_light_storage: BufferNodeId,
        cpu_light_capacity: u32,
        gpu_lights: GpuLightBuffers,
    ) -> LightMergePassOutputs {
        let bytes_per_light = std::mem::size_of::<GpuLightStorage>() as u64;
        let gpu_capacity = light_capacity_from_desc(ctx.buffer_desc(gpu_lights.light_storage));
        let merged_capacity = cpu_light_capacity.saturating_add(gpu_capacity).max(1);
        let merged_storage_size = u64::from(merged_capacity).saturating_mul(bytes_per_light);

        ctx.with_group("Merge_Local_Lights", |ctx| {
            ctx.graph.add_pass("Merge_Local_Lights", |builder| {
                builder.read_buffer(cpu_light_metadata);
                builder.read_buffer(cpu_light_storage);
                builder.read_buffer(gpu_lights.light_metadata);
                builder.read_buffer(gpu_lights.light_storage);

                let merged_light_metadata = builder.create_buffer(
                    "Merged_Local_Light_Metadata",
                    BufferDesc::new(
                        std::mem::size_of::<LightBufferMetadata>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
                    ),
                );
                let merged_light_storage = builder.create_buffer(
                    "Merged_Local_Lights",
                    BufferDesc::new(merged_storage_size, wgpu::BufferUsages::STORAGE),
                );
                let merged_indirect_count = builder.create_buffer(
                    "Merged_Local_Light_Count",
                    BufferDesc::new(4, wgpu::BufferUsages::STORAGE),
                );

                (
                    LightMergePassNode {
                        feature: self,
                        cpu_light_metadata,
                        cpu_light_storage,
                        gpu_light_metadata: gpu_lights.light_metadata,
                        gpu_light_storage: gpu_lights.light_storage,
                        merged_light_metadata,
                        merged_light_storage,
                        merged_indirect_count,
                        merged_capacity,
                        bind_group: None,
                    },
                    LightMergePassOutputs {
                        light_metadata: merged_light_metadata,
                        light_storage: merged_light_storage,
                        indirect_count_buffer: merged_indirect_count,
                    },
                )
            })
        })
    }
}

fn create_light_merge_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Light Merge Layout"),
            entries: &[
                uniform_buffer(0, wgpu::ShaderStages::COMPUTE),
                storage_buffer(1, wgpu::ShaderStages::COMPUTE, true),
                uniform_buffer(2, wgpu::ShaderStages::COMPUTE),
                storage_buffer(3, wgpu::ShaderStages::COMPUTE, true),
                storage_buffer(4, wgpu::ShaderStages::COMPUTE, false),
                storage_buffer(5, wgpu::ShaderStages::COMPUTE, false),
                storage_buffer(6, wgpu::ShaderStages::COMPUTE, false),
            ],
        }),
    )
}

struct LightMergePassNode<'a> {
    feature: &'a mut LightMergeFeature,
    cpu_light_metadata: BufferNodeId,
    cpu_light_storage: BufferNodeId,
    gpu_light_metadata: BufferNodeId,
    gpu_light_storage: BufferNodeId,
    merged_light_metadata: BufferNodeId,
    merged_light_storage: BufferNodeId,
    merged_indirect_count: BufferNodeId,
    merged_capacity: u32,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for LightMergePassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(ctx);
        self.bind_group = Some(
            ctx.build_bind_group(self.feature.layout(), Some("Light Merge BG"))
                .bind_buffer(0, self.cpu_light_metadata)
                .bind_buffer(1, self.cpu_light_storage)
                .bind_buffer(2, self.gpu_light_metadata)
                .bind_buffer(3, self.gpu_light_storage)
                .bind_buffer(4, self.merged_light_metadata)
                .bind_buffer(5, self.merged_light_storage)
                .bind_buffer(6, self.merged_indirect_count)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Light Merge Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_compute_pipeline(self.feature.pipeline_id()),
        );
        pass.set_bind_group(0, self.bind_group.expect("Light merge BG missing"), &[]);
        pass.dispatch_workgroups(self.merged_capacity.div_ceil(LIGHT_MERGE_WG_SIZE), 1, 1);
    }
}

fn light_capacity_from_desc(desc: BufferDesc) -> u32 {
    let bytes_per_light = std::mem::size_of::<GpuLightStorage>() as u64;
    ((desc.logical_size / bytes_per_light).max(1)).min(u64::from(u32::MAX)) as u32
}
