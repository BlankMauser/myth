//! Shared Hi-Z depth pyramid infrastructure.
//!
//! The hierarchy stores reverse-Z maximum depth in an `R32Float` mip chain so
//! screen-space ray features can conservatively skip empty space.

use crate::core::gpu::Tracked;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, SubViewKey, TextureDesc,
    TextureNodeId,
};
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};

const HIZ_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const HIZ_WORKGROUP_SIZE: u32 = 8;
const HIZ_MIN_MIP_SIZE: u32 = 16;

pub struct HiZFeature {
    init_pipeline: Option<ComputePipelineId>,
    downsample_pipeline: Option<ComputePipelineId>,
    init_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    downsample_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl Default for HiZFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl HiZFeature {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            init_pipeline: None,
            downsample_pipeline: None,
            init_layout: None,
            downsample_layout: None,
        }
    }

    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.ensure_layouts(ctx.device);
        self.ensure_pipelines(ctx);
    }

    #[must_use]
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        scene_depth: TextureNodeId,
    ) -> TextureNodeId {
        let mip_count = hiz_mip_count(ctx.frame_config.width, ctx.frame_config.height);
        let desc = TextureDesc::new(
            ctx.frame_config.width,
            ctx.frame_config.height,
            1,
            mip_count,
            1,
            wgpu::TextureDimension::D2,
            HIZ_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );

        let init_pipeline_id = self.init_pipeline.expect("Hi-Z init pipeline missing");
        let downsample_pipeline_id = self
            .downsample_pipeline
            .expect("Hi-Z downsample pipeline missing");
        let init_layout = self.init_layout.as_ref().expect("Hi-Z init layout missing");
        let downsample_layout = self
            .downsample_layout
            .as_ref()
            .expect("Hi-Z downsample layout missing");

        ctx.graph.add_pass("HiZ_Build", |builder| {
            builder.read_texture(scene_depth);
            let hiz_texture = builder.create_texture("Scene_HiZ", desc);

            let mut width = ctx.frame_config.width;
            let mut height = ctx.frame_config.height;
            let mut mip_states = Vec::with_capacity(mip_count.saturating_sub(1) as usize);
            for mip_level in 1..mip_count {
                width = width.div_ceil(2);
                height = height.div_ceil(2);
                mip_states.push(HiZMipState {
                    mip_level,
                    target_size: (width.max(1), height.max(1)),
                    bind_group: None,
                });
            }
            let mip_states = builder.graph.alloc_slice_mut(&mip_states);

            let node = HiZBuildNode {
                scene_depth,
                hiz_texture,
                init_pipeline_id,
                downsample_pipeline_id,
                init_layout,
                downsample_layout,
                init_bg: None,
                mips: mip_states,
            };

            (node, hiz_texture)
        })
    }

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.init_layout.is_some() {
            return;
        }

        let init_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Hi-Z Init Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HIZ_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let downsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Hi-Z Downsample Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HIZ_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        self.init_layout = Some(Tracked::new(init_layout));
        self.downsample_layout = Some(Tracked::new(downsample_layout));
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.init_pipeline.is_none() {
            let (module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/utility/hiz/init"),
                &ShaderCompilationOptions::default(),
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Hi-Z Init Pipeline Layout"),
                        bind_group_layouts: &[self.init_layout.as_deref()],
                        immediate_size: 0,
                    });

            self.init_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                ctx.device,
                module,
                &pipeline_layout,
                &ComputePipelineKey::new(shader_hash),
                &wgpu::PipelineCompilationOptions::default(),
                "Hi-Z Init Pipeline",
            ));
        }

        if self.downsample_pipeline.is_none() {
            let (module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/utility/hiz/downsample"),
                &ShaderCompilationOptions::default(),
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Hi-Z Downsample Pipeline Layout"),
                        bind_group_layouts: &[self.downsample_layout.as_deref()],
                        immediate_size: 0,
                    });

            self.downsample_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                ctx.device,
                module,
                &pipeline_layout,
                &ComputePipelineKey::new(shader_hash),
                &wgpu::PipelineCompilationOptions::default(),
                "Hi-Z Downsample Pipeline",
            ));
        }
    }
}

#[derive(Clone, Copy)]
struct HiZMipState<'a> {
    mip_level: u32,
    target_size: (u32, u32),
    bind_group: Option<&'a wgpu::BindGroup>,
}

struct HiZBuildNode<'a> {
    scene_depth: TextureNodeId,
    hiz_texture: TextureNodeId,
    init_pipeline_id: ComputePipelineId,
    downsample_pipeline_id: ComputePipelineId,
    init_layout: &'a Tracked<wgpu::BindGroupLayout>,
    downsample_layout: &'a Tracked<wgpu::BindGroupLayout>,
    init_bg: Option<&'a wgpu::BindGroup>,
    mips: &'a mut [HiZMipState<'a>],
}

impl<'a> PassNode<'a> for HiZBuildNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let mip0 = ctx
            .views
            .get_or_create_sub_view(
                self.hiz_texture,
                &SubViewKey {
                    base_mip: 0,
                    mip_count: Some(1),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    ..Default::default()
                },
            )
            .clone();

        self.init_bg = Some(
            ctx.build_bind_group(self.init_layout, Some("Hi-Z Init BG"))
                .bind_texture(0, self.scene_depth)
                .bind_tracked_texture_view(1, &mip0)
                .build(),
        );

        for mip in self.mips.iter_mut() {
            let src_view = ctx
                .views
                .get_or_create_sub_view(
                    self.hiz_texture,
                    &SubViewKey {
                        base_mip: mip.mip_level - 1,
                        mip_count: Some(1),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        ..Default::default()
                    },
                )
                .clone();
            let dst_view = ctx
                .views
                .get_or_create_sub_view(
                    self.hiz_texture,
                    &SubViewKey {
                        base_mip: mip.mip_level,
                        mip_count: Some(1),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        ..Default::default()
                    },
                )
                .clone();

            mip.bind_group = Some(
                ctx.build_bind_group(self.downsample_layout, Some("Hi-Z Downsample BG"))
                    .bind_tracked_texture_view(0, &src_view)
                    .bind_tracked_texture_view(1, &dst_view)
                    .build(),
            );
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let hiz_texture = ctx.get_texture(self.hiz_texture);
        let full_w = hiz_texture.width();
        let full_h = hiz_texture.height();

        let mut init_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Hi-Z Init Compute"),
            timestamp_writes: None,
        });

        init_pass.set_pipeline(
            ctx.pipeline_cache
                .get_compute_pipeline(self.init_pipeline_id),
        );
        init_pass.set_bind_group(0, self.init_bg.expect("Hi-Z init BG missing"), &[]);
        init_pass.dispatch_workgroups(
            full_w.div_ceil(HIZ_WORKGROUP_SIZE),
            full_h.div_ceil(HIZ_WORKGROUP_SIZE),
            1,
        );
        drop(init_pass);

        for mip in self.mips.iter() {
            let mut downsample_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hi-Z Downsample Compute"),
                timestamp_writes: None,
            });
            downsample_pass.set_pipeline(
                ctx.pipeline_cache
                    .get_compute_pipeline(self.downsample_pipeline_id),
            );
            downsample_pass.set_bind_group(0, mip.bind_group.expect("Hi-Z mip BG missing"), &[]);
            downsample_pass.dispatch_workgroups(
                mip.target_size.0.div_ceil(HIZ_WORKGROUP_SIZE),
                mip.target_size.1.div_ceil(HIZ_WORKGROUP_SIZE),
                1,
            );
        }
    }
}

#[inline]
fn hiz_mip_count(width: u32, height: u32) -> u32 {
    let mut mip_count = 1;
    let mut max_dim = width.max(height).max(1);

    while max_dim > HIZ_MIN_MIP_SIZE {
        max_dim = max_dim.div_ceil(2);
        mip_count += 1;
    }

    mip_count
}

#[cfg(test)]
mod tests {
    use super::hiz_mip_count;

    #[test]
    fn extends_mip_chain_until_minimum_size() {
        assert_eq!(hiz_mip_count(1, 1), 1);
        assert_eq!(hiz_mip_count(17, 17), 2);
        assert_eq!(hiz_mip_count(64, 32), 3);
        assert_eq!(hiz_mip_count(1920, 1080), 8);
        assert_eq!(hiz_mip_count(3840, 2160), 9);
    }
}
