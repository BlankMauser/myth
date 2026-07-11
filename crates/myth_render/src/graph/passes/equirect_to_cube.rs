//! Environment source conversion pass.
//!
//! The feature extracts only CPU-side source identity. Pipeline compilation,
//! texture upload/resolution, custom-view creation, sampler creation, and bind
//! group assembly are deferred until the node survives DPE.

use crate::core::gpu::{
    CommonSampler, CubeSourceType, GpuEnvironment, PreparedTextureSource, Tracked, binding_layout,
};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{ExecuteContext, PassNode, PrepareContext, TextureNodeId};
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::texture::TextureSource;

/// CPU-only description of one source-to-base-cube conversion.
#[derive(Debug, Clone, Copy)]
pub struct EquirectToCubeSnapshot {
    pub(crate) source: TextureSource,
    pub(crate) source_type: CubeSourceType,
    pub(crate) source_version: u64,
    /// L1 identity used by the logical base-cache key for this conversion.
    pub(crate) producer_shader_identity: u64,
}

/// Cross-frame static state for environment source conversion.
pub struct EquirectToCubeFeature {
    equirect_pipeline_id: Option<ComputePipelineId>,
    equirect_pipeline_shader_identity: Option<u64>,
    cubemap_pipeline_id: Option<ComputePipelineId>,
    cubemap_pipeline_shader_identity: Option<u64>,
    equirect_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    cubemap_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    equirect_sampler: Option<Tracked<wgpu::Sampler>>,
}

impl Default for EquirectToCubeFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl EquirectToCubeFeature {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            equirect_pipeline_id: None,
            equirect_pipeline_shader_identity: None,
            cubemap_pipeline_id: None,
            cubemap_pipeline_shader_identity: None,
            equirect_layout: None,
            cubemap_layout: None,
            equirect_sampler: None,
        }
    }

    /// Extract a lightweight source snapshot without touching the GPU.
    #[must_use]
    #[allow(clippy::unused_self)] // Keep the uniform Feature::extract API.
    pub(crate) fn extract(
        &self,
        gpu_env: &GpuEnvironment,
        producer_shader_identity: u64,
    ) -> Option<EquirectToCubeSnapshot> {
        if !gpu_env.source_ready || gpu_env.source_type == CubeSourceType::Procedural {
            return None;
        }

        Some(EquirectToCubeSnapshot {
            source: gpu_env.source_key?,
            source_type: gpu_env.source_type,
            source_version: gpu_env.source_version,
            producer_shader_identity,
        })
    }

    /// Insert a conversion writer for an invalid base-cube cache plan.
    ///
    /// The caller owns cache validity decisions. Reaching this method means
    /// the conversion must run if the node remains live.
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: EquirectToCubeSnapshot,
        base_cube: TextureNodeId,
    ) -> TextureNodeId {
        ctx.graph.add_pass("Environment_Source_Convert", |builder| {
            builder.write_texture(base_cube);
            let node = EquirectToCubePassNode {
                feature: self,
                snapshot,
                base_cube,
                bind_group: None,
            };
            (node, ())
        });

        base_cube
    }

    /// Ensure only the variant selected by this live node.
    fn prepare_resources(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        source_type: CubeSourceType,
        producer_shader_identity: u64,
    ) {
        match source_type {
            CubeSourceType::Equirectangular => {
                self.prepare_equirect_resources(ctx, producer_shader_identity);
            }
            CubeSourceType::Cubemap => {
                self.prepare_cubemap_resources(ctx, producer_shader_identity);
            }
            CubeSourceType::Procedural => {
                panic!("procedural atmosphere must not use source conversion")
            }
        }
    }

    fn prepare_equirect_resources(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        expected_shader_identity: u64,
    ) {
        if self.equirect_layout.is_none() {
            self.equirect_layout = Some(Self::create_layout(
                ctx.device,
                "Environment EquirectToCube BGL",
                wgpu::TextureViewDimension::D2,
            ));
        }

        if self.equirect_sampler.is_none() {
            self.equirect_sampler = Some(Tracked::new(ctx.device.create_sampler(
                &wgpu::SamplerDescriptor {
                    label: Some("Environment Equirect Sampler"),
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    lod_min_clamp: 0.0,
                    lod_max_clamp: 32.0,
                    compare: None,
                    anisotropy_clamp: 1,
                    border_color: None,
                },
            )));
        }

        if self.equirect_pipeline_id.is_none()
            || self.equirect_pipeline_shader_identity != Some(expected_shader_identity)
        {
            assert_eq!(
                ctx.shader_manager.pipeline_shader_hash(
                    ShaderSource::File("entry/utility/equirect_to_cube"),
                    &ShaderCompilationOptions::default(),
                ),
                expected_shader_identity,
                "environment cache key and equirect pipeline shader identity diverged",
            );
            let pipeline_id = Self::create_pipeline(
                ctx,
                self.equirect_layout
                    .as_ref()
                    .expect("equirect layout must exist"),
                ShaderSource::File("entry/utility/equirect_to_cube"),
                "Environment EquirectToCube PL",
                "Environment EquirectToCube Pipeline",
            );
            self.equirect_pipeline_id = Some(pipeline_id);
            self.equirect_pipeline_shader_identity = Some(expected_shader_identity);
        }
    }

    fn prepare_cubemap_resources(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        expected_shader_identity: u64,
    ) {
        if self.cubemap_layout.is_none() {
            self.cubemap_layout = Some(Self::create_layout(
                ctx.device,
                "Environment CubeToCube BGL",
                wgpu::TextureViewDimension::Cube,
            ));
        }

        if self.cubemap_pipeline_id.is_none()
            || self.cubemap_pipeline_shader_identity != Some(expected_shader_identity)
        {
            assert_eq!(
                ctx.shader_manager.pipeline_shader_hash(
                    ShaderSource::File("entry/utility/cube_to_cube"),
                    &ShaderCompilationOptions::default(),
                ),
                expected_shader_identity,
                "environment cache key and cube pipeline shader identity diverged",
            );
            let pipeline_id = Self::create_pipeline(
                ctx,
                self.cubemap_layout
                    .as_ref()
                    .expect("cubemap layout must exist"),
                ShaderSource::File("entry/utility/cube_to_cube"),
                "Environment CubeToCube PL",
                "Environment CubeToCube Pipeline",
            );
            self.cubemap_pipeline_id = Some(pipeline_id);
            self.cubemap_pipeline_shader_identity = Some(expected_shader_identity);
        }
    }

    fn create_layout(
        device: &wgpu::Device,
        label: &'static str,
        source_dimension: wgpu::TextureViewDimension,
    ) -> Tracked<wgpu::BindGroupLayout> {
        Tracked::new(
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    binding_layout::texture(
                        0,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::TextureSampleType::Float { filterable: true },
                        source_dimension,
                        false,
                    ),
                    binding_layout::sampler(
                        1,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::SamplerBindingType::Filtering,
                    ),
                    binding_layout::storage_texture(
                        2,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::StorageTextureAccess::WriteOnly,
                        wgpu::TextureFormat::Rgba16Float,
                        wgpu::TextureViewDimension::D2Array,
                    ),
                ],
            }),
        )
    }

    fn create_pipeline(
        ctx: &mut PrepareContext<'_>,
        layout: &Tracked<wgpu::BindGroupLayout>,
        shader: ShaderSource<'static>,
        layout_label: &'static str,
        pipeline_label: &'static str,
    ) -> ComputePipelineId {
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            shader,
            &ShaderCompilationOptions::default(),
        );
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(layout_label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
        let compilation_options = wgpu::PipelineCompilationOptions::default();

        ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            module,
            &pipeline_layout,
            &ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            pipeline_label,
        )
    }

    fn layout(&self, source_type: CubeSourceType) -> &Tracked<wgpu::BindGroupLayout> {
        match source_type {
            CubeSourceType::Equirectangular => self
                .equirect_layout
                .as_ref()
                .expect("equirect layout must exist after prepare"),
            CubeSourceType::Cubemap => self
                .cubemap_layout
                .as_ref()
                .expect("cubemap layout must exist after prepare"),
            CubeSourceType::Procedural => {
                panic!("procedural atmosphere must not use source conversion")
            }
        }
    }

    fn pipeline_id(&self, source_type: CubeSourceType) -> ComputePipelineId {
        match source_type {
            CubeSourceType::Equirectangular => self
                .equirect_pipeline_id
                .expect("equirect pipeline must exist after prepare"),
            CubeSourceType::Cubemap => self
                .cubemap_pipeline_id
                .expect("cubemap pipeline must exist after prepare"),
            CubeSourceType::Procedural => {
                panic!("procedural atmosphere must not use source conversion")
            }
        }
    }
}

struct EquirectToCubePassNode<'a> {
    feature: &'a mut EquirectToCubeFeature,
    snapshot: EquirectToCubeSnapshot,
    base_cube: TextureNodeId,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for EquirectToCubePassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.prepare_resources(
            ctx,
            self.snapshot.source_type,
            self.snapshot.producer_shader_identity,
        );
        ctx.mipmap_generator
            .ensure_pipeline(ctx.device, wgpu::TextureFormat::Rgba16Float);

        let PreparedTextureSource {
            view: source_view,
            view_id: source_view_id,
            view_dimension: source_view_dimension,
            logical_version: source_version,
        } = ctx
            .uploads
            .resolve_texture_source(
                ctx.assets,
                self.snapshot.source,
                self.snapshot.source_version,
            )
            .expect("environment source must keep its snapshot version through prepare");
        let expected_view_dimension = match self.snapshot.source_type {
            CubeSourceType::Equirectangular => wgpu::TextureViewDimension::D2,
            CubeSourceType::Cubemap => wgpu::TextureViewDimension::Cube,
            CubeSourceType::Procedural => unreachable!(),
        };
        assert_eq!(
            source_view_dimension, expected_view_dimension,
            "prepared environment source dimension diverged from its CPU snapshot"
        );
        assert_eq!(
            source_version, self.snapshot.source_version,
            "prepared environment source version diverged from its CPU snapshot"
        );
        let base_cube_uid = ctx.views.get_physical_texture_uid(self.base_cube);
        let storage_view =
            ctx.views
                .get_texture(self.base_cube)
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Environment Base Cube Mip 0 Storage View"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                });

        let builder = ctx
            .build_bind_group(
                self.feature.layout(self.snapshot.source_type),
                Some("Environment Source Convert BG"),
            )
            .bind_texture_view_with_id(0, &source_view, source_view_id);
        let builder = match self.snapshot.source_type {
            CubeSourceType::Equirectangular => builder.bind_tracked_sampler(
                1,
                self.feature
                    .equirect_sampler
                    .as_ref()
                    .expect("equirect sampler must exist after prepare"),
            ),
            CubeSourceType::Cubemap => builder.bind_common_sampler(1, CommonSampler::LinearClamp),
            CubeSourceType::Procedural => unreachable!(),
        };
        self.bind_group = Some(
            builder
                .bind_texture_view_with_id(2, &storage_view, base_cube_uid)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let pipeline = ctx
            .pipeline_cache
            .get_compute_pipeline(self.feature.pipeline_id(self.snapshot.source_type));
        let group_count = ctx.get_texture(self.base_cube).width().div_ceil(8);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Environment Source Convert"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(
                0,
                self.bind_group
                    .expect("environment source bind group must be prepared"),
                &[],
            );
            cpass.dispatch_workgroups(group_count, group_count, 6);
        }

        ctx.mipmap_generator
            .generate(ctx.device, encoder, ctx.get_texture(self.base_cube));
    }
}
