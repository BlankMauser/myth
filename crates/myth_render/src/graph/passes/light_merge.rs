use crate::core::gpu::Tracked;
use crate::graph::composer::{GpuLightBuffers, GraphBuilderContext};
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, ExtractContext, PassNode, PrepareContext,
};
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

    fn ensure_pipeline(&mut self, ctx: &mut ExtractContext) {
        let layout = self.layout.get_or_insert_with(|| {
            Tracked::new(
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Light Merge Layout"),
                        entries: &[
                            uniform_entry(0),
                            storage_entry(1, true),
                            uniform_entry(2),
                            storage_entry(3, true),
                            storage_entry(4, false),
                            storage_entry(5, false),
                            storage_entry(6, false),
                        ],
                    }),
            )
        });

        if self.pipeline.is_some() {
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

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Merge Pipeline Layout"),
                bind_group_layouts: &[Some(&**layout)],
                immediate_size: 0,
            });

        self.pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            module,
            &pipeline_layout,
            &ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            "Light Merge Pipeline",
        ));
    }

    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.ensure_pipeline(ctx);
    }

    pub fn add_to_graph<'a>(
        &'a self,
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

        let layout = self
            .layout
            .as_ref()
            .expect("Light merge layout must exist before graph build");
        let pipeline_id = self
            .pipeline
            .expect("Light merge pipeline must exist before graph build");

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
                        cpu_light_metadata,
                        cpu_light_storage,
                        gpu_light_metadata: gpu_lights.light_metadata,
                        gpu_light_storage: gpu_lights.light_storage,
                        merged_light_metadata,
                        merged_light_storage,
                        merged_indirect_count,
                        merged_capacity,
                        layout,
                        pipeline_id,
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

struct LightMergePassNode<'a> {
    cpu_light_metadata: BufferNodeId,
    cpu_light_storage: BufferNodeId,
    gpu_light_metadata: BufferNodeId,
    gpu_light_storage: BufferNodeId,
    merged_light_metadata: BufferNodeId,
    merged_light_storage: BufferNodeId,
    merged_indirect_count: BufferNodeId,
    merged_capacity: u32,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    pipeline_id: ComputePipelineId,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for LightMergePassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.bind_group = Some(
            ctx.build_bind_group(self.layout, Some("Light Merge BG"))
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
        pass.set_pipeline(ctx.pipeline_cache.get_compute_pipeline(self.pipeline_id));
        pass.set_bind_group(0, self.bind_group.expect("Light merge BG missing"), &[]);
        pass.dispatch_workgroups(self.merged_capacity.div_ceil(LIGHT_MERGE_WG_SIZE), 1, 1);
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn light_capacity_from_desc(desc: BufferDesc) -> u32 {
    let bytes_per_light = std::mem::size_of::<GpuLightStorage>() as u64;
    ((desc.logical_size / bytes_per_light).max(1)).min(u64::from(u32::MAX)) as u32
}
