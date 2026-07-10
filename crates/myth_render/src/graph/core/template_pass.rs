use smallvec::SmallVec;

use crate::core::gpu::binding_layout;
use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::core::{
    BufferNodeId, ExecuteContext, ExtractContext, GraphBinding, PassBuilder, PassNode,
    PrepareContext, RawBufferBinding, RawSamplerBinding, RawTextureViewBinding, RenderTargetOps,
    TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, ComputePipelineId, ComputePipelineKey, DepthStencilKey, FullscreenPipelineKey,
    MultisampleKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use crate::renderer::Renderer;
use myth_resources::material::ShaderTemplateMode;

const MAX_TEMPLATE_BIND_GROUPS: usize = 4;

#[derive(Clone, Copy)]
pub enum TemplateShaderSource {
    File(&'static str),
    Inline {
        name: &'static str,
        source: &'static str,
    },
}

impl TemplateShaderSource {
    fn as_shader_source(self) -> ShaderSource<'static> {
        match self {
            Self::File(path) => ShaderSource::File(path),
            Self::Inline { name, source } => ShaderSource::Inline {
                name,
                source,
                mode: ShaderTemplateMode::Template,
            },
        }
    }
}

#[derive(Clone)]
pub struct TemplateBindingLayoutDesc {
    pub group: u32,
    pub binding: u32,
    pub visibility: wgpu::ShaderStages,
    pub binding_type: wgpu::BindingType,
}

impl TemplateBindingLayoutDesc {
    #[must_use]
    pub fn texture(
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Self {
        Self {
            group,
            binding,
            visibility,
            binding_type: binding_layout::texture_type(
                wgpu::TextureSampleType::Float { filterable },
                view_dimension,
                false,
            ),
        }
    }

    #[must_use]
    pub fn texture_2d(
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) -> Self {
        Self::texture(
            group,
            binding,
            visibility,
            filterable,
            wgpu::TextureViewDimension::D2,
        )
    }

    #[must_use]
    pub fn texture_cube(
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) -> Self {
        Self::texture(
            group,
            binding,
            visibility,
            filterable,
            wgpu::TextureViewDimension::Cube,
        )
    }

    #[must_use]
    pub fn depth_texture_2d(group: u32, binding: u32, visibility: wgpu::ShaderStages) -> Self {
        Self {
            group,
            binding,
            visibility,
            binding_type: binding_layout::texture_type(
                wgpu::TextureSampleType::Depth,
                wgpu::TextureViewDimension::D2,
                false,
            ),
        }
    }

    #[must_use]
    pub fn sampler(
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        sampler_type: wgpu::SamplerBindingType,
    ) -> Self {
        Self {
            group,
            binding,
            visibility,
            binding_type: binding_layout::sampler_type(sampler_type),
        }
    }

    #[must_use]
    pub fn uniform_buffer(group: u32, binding: u32, visibility: wgpu::ShaderStages) -> Self {
        Self {
            group,
            binding,
            visibility,
            binding_type: binding_layout::uniform_buffer_type(),
        }
    }

    #[must_use]
    pub fn storage_buffer(
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> Self {
        Self {
            group,
            binding,
            visibility,
            binding_type: binding_layout::storage_buffer_type(read_only),
        }
    }

    #[must_use]
    pub fn storage_texture(
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        access: wgpu::StorageTextureAccess,
        format: wgpu::TextureFormat,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Self {
        Self {
            group,
            binding,
            visibility,
            binding_type: binding_layout::storage_texture_type(access, format, view_dimension),
        }
    }
}

pub struct TemplatePassDescriptor {
    shader_source: TemplateShaderSource,
    binding_layouts: SmallVec<[TemplateBindingLayoutDesc; 8]>,
    tracked_layouts: SmallVec<[Tracked<wgpu::BindGroupLayout>; MAX_TEMPLATE_BIND_GROUPS]>,
}

impl TemplatePassDescriptor {
    #[must_use]
    pub fn new(shader_source: TemplateShaderSource) -> Self {
        Self {
            shader_source,
            binding_layouts: SmallVec::new(),
            tracked_layouts: SmallVec::new(),
        }
    }

    pub fn add_binding_layout(&mut self, desc: TemplateBindingLayoutDesc) {
        self.binding_layouts.push(desc);
    }

    pub fn add_texture_2d(
        &mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) {
        self.add_binding_layout(TemplateBindingLayoutDesc::texture_2d(
            group, binding, visibility, filterable,
        ));
    }

    pub fn add_texture_cube(
        &mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) {
        self.add_binding_layout(TemplateBindingLayoutDesc::texture_cube(
            group, binding, visibility, filterable,
        ));
    }

    pub fn add_depth_texture_2d(
        &mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) {
        self.add_binding_layout(TemplateBindingLayoutDesc::depth_texture_2d(
            group, binding, visibility,
        ));
    }

    pub fn add_sampler(
        &mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        sampler_type: wgpu::SamplerBindingType,
    ) {
        self.add_binding_layout(TemplateBindingLayoutDesc::sampler(
            group,
            binding,
            visibility,
            sampler_type,
        ));
    }

    pub fn add_uniform_buffer(&mut self, group: u32, binding: u32, visibility: wgpu::ShaderStages) {
        self.add_binding_layout(TemplateBindingLayoutDesc::uniform_buffer(
            group, binding, visibility,
        ));
    }

    pub fn add_storage_buffer(
        &mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) {
        self.add_binding_layout(TemplateBindingLayoutDesc::storage_buffer(
            group, binding, visibility, read_only,
        ));
    }

    pub fn add_storage_texture(
        &mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        access: wgpu::StorageTextureAccess,
        format: wgpu::TextureFormat,
        view_dimension: wgpu::TextureViewDimension,
    ) {
        self.add_binding_layout(TemplateBindingLayoutDesc::storage_texture(
            group,
            binding,
            visibility,
            access,
            format,
            view_dimension,
        ));
    }

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if !self.tracked_layouts.is_empty() {
            return;
        }

        let mut sorted = self.binding_layouts.clone();
        sorted.sort_by_key(|entry| (entry.group, entry.binding));

        let mut next_group = 0u32;
        let mut cursor = 0usize;
        while cursor < sorted.len() {
            let group = sorted[cursor].group;
            assert!(
                group == next_group,
                "Template pass bind groups must be dense and start at group 0"
            );
            assert!(
                self.tracked_layouts.len() < MAX_TEMPLATE_BIND_GROUPS,
                "Template pass exceeds the supported bind-group limit of {MAX_TEMPLATE_BIND_GROUPS}"
            );

            let start = cursor;
            while cursor < sorted.len() && sorted[cursor].group == group {
                cursor += 1;
            }

            let entries = sorted[start..cursor]
                .iter()
                .map(|entry| wgpu::BindGroupLayoutEntry {
                    binding: entry.binding,
                    visibility: entry.visibility,
                    ty: entry.binding_type,
                    count: None,
                })
                .collect::<Vec<_>>();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Template Pass Layout"),
                entries: &entries,
            });

            self.tracked_layouts.push(Tracked::new(layout));
            next_group += 1;
        }
    }

    #[must_use]
    pub fn bind_group_count(&self) -> usize {
        self.tracked_layouts.len().max(
            self.binding_layouts
                .iter()
                .map(|entry| entry.group as usize + 1)
                .max()
                .unwrap_or(0),
        )
    }

    pub fn compile_compute_with_extract(
        &mut self,
        ctx: &mut ExtractContext,
        shader_options: &ShaderCompilationOptions,
        label: &str,
    ) -> ComputePipelineId {
        self.ensure_layouts(ctx.device);

        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            self.shader_source.as_shader_source(),
            shader_options,
        );
        let raw_layouts = self
            .tracked_layouts
            .iter()
            .map(|layout| {
                let layout_ref: &wgpu::BindGroupLayout = layout;
                Some(layout_ref)
            })
            .collect::<Vec<_>>();
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &raw_layouts,
                immediate_size: 0,
            });
        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let pipeline_id = ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            module,
            &pipeline_layout,
            &ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            label,
        );
        ctx.pipeline_cache
            .register_compute_layouts(pipeline_id, self.tracked_layouts.clone());
        pipeline_id
    }

    pub fn compile_fullscreen_with_extract(
        &mut self,
        ctx: &mut ExtractContext,
        shader_options: &ShaderCompilationOptions,
        color_targets: &[wgpu::ColorTargetState],
        depth_stencil: Option<wgpu::DepthStencilState>,
        multisample: wgpu::MultisampleState,
        label: &str,
    ) -> RenderPipelineId {
        self.ensure_layouts(ctx.device);

        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            self.shader_source.as_shader_source(),
            shader_options,
        );
        let raw_layouts = self
            .tracked_layouts
            .iter()
            .map(|layout| {
                let layout_ref: &wgpu::BindGroupLayout = layout;
                Some(layout_ref)
            })
            .collect::<Vec<_>>();
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &raw_layouts,
                immediate_size: 0,
            });
        let key = FullscreenPipelineKey {
            shader_hash,
            color_targets: color_targets
                .iter()
                .cloned()
                .map(ColorTargetKey::from)
                .collect(),
            depth_stencil: depth_stencil.map(DepthStencilKey::from),
            multisample: MultisampleKey::from(multisample),
        };
        let pipeline_id = ctx.pipeline_cache.get_or_create_fullscreen(
            ctx.device,
            module,
            &pipeline_layout,
            &key,
            label,
        );
        ctx.pipeline_cache
            .register_render_layouts(pipeline_id, self.tracked_layouts.clone());
        pipeline_id
    }

    fn compile_compute_with_renderer(
        &mut self,
        renderer: &mut Renderer,
        shader_options: &ShaderCompilationOptions,
        label: &str,
    ) -> ComputePipelineId {
        {
            let wgpu_ctx = renderer
                .wgpu_ctx()
                .expect("Renderer must be initialized before preparing template passes");
            self.ensure_layouts(&wgpu_ctx.device);
        }
        let layout_refs = self
            .tracked_layouts
            .iter()
            .collect::<SmallVec<[&Tracked<wgpu::BindGroupLayout>; MAX_TEMPLATE_BIND_GROUPS]>>();

        renderer.get_or_create_compute_pipeline(
            self.shader_source.as_shader_source(),
            shader_options,
            &layout_refs,
            label,
        )
    }

    fn compile_fullscreen_with_renderer(
        &mut self,
        renderer: &mut Renderer,
        shader_options: &ShaderCompilationOptions,
        color_targets: &[wgpu::ColorTargetState],
        depth_stencil: Option<wgpu::DepthStencilState>,
        multisample: wgpu::MultisampleState,
        label: &str,
    ) -> RenderPipelineId {
        {
            let wgpu_ctx = renderer
                .wgpu_ctx()
                .expect("Renderer must be initialized before preparing template passes");
            self.ensure_layouts(&wgpu_ctx.device);
        }
        let layout_refs = self
            .tracked_layouts
            .iter()
            .collect::<SmallVec<[&Tracked<wgpu::BindGroupLayout>; MAX_TEMPLATE_BIND_GROUPS]>>();

        renderer.get_or_create_fullscreen_pipeline(
            self.shader_source.as_shader_source(),
            shader_options,
            &layout_refs,
            color_targets,
            depth_stencil,
            multisample,
            label,
        )
    }

    fn build_bind_groups<'a>(
        &self,
        builder: &PassBuilder<'_, 'a>,
        bindings: SmallVec<[TemplatePassBinding<'a>; 8]>,
        bind_group_label: Option<&'static str>,
    ) -> ([RuntimeBindGroupSlot<'a>; MAX_TEMPLATE_BIND_GROUPS], usize) {
        let mut slots = [RuntimeBindGroupSlot::default(); MAX_TEMPLATE_BIND_GROUPS];
        if bindings.is_empty() {
            return (slots, 0);
        }

        let mut sorted = bindings;
        sorted.sort_by_key(|binding| (binding.group, binding.binding));

        let mut cursor = 0usize;
        let mut count = 0usize;
        while cursor < sorted.len() {
            let group = sorted[cursor].group;
            assert!(
                (group as usize) < self.bind_group_count(),
                "Runtime binding targets undeclared bind group {group}"
            );
            assert!(
                count < MAX_TEMPLATE_BIND_GROUPS,
                "Runtime bind groups exceed the supported limit of {MAX_TEMPLATE_BIND_GROUPS}"
            );

            let start = cursor;
            while cursor < sorted.len() && sorted[cursor].group == group {
                cursor += 1;
            }

            let binding_slice = builder.graph.alloc_slice(&sorted[start..cursor]);
            slots[count] = RuntimeBindGroupSlot {
                group,
                bindings: binding_slice,
                label: bind_group_label,
                bind_group: None,
            };
            count += 1;
        }

        (slots, count)
    }

    pub fn build_fullscreen_node<'a, F>(
        &'a self,
        builder: &mut PassBuilder<'_, 'a>,
        pipeline_id: RenderPipelineId,
        label: &'static str,
        output_tex: TextureNodeId,
        output_ops: RenderTargetOps,
        bind_group_label: Option<&'static str>,
        configure: F,
    ) -> StandardFullscreenNode<'a>
    where
        F: FnOnce(&mut TemplatePassBindingsBuilder<'a>),
    {
        let mut binding_builder = TemplatePassBindingsBuilder::new();
        configure(&mut binding_builder);
        let (bind_groups, bind_group_count) =
            self.build_bind_groups(builder, binding_builder.finish(), bind_group_label);

        StandardFullscreenNode {
            label,
            pipeline_id,
            output_tex,
            output_ops,
            bind_groups,
            bind_group_count,
        }
    }

    pub fn build_compute_node<'a, F>(
        &'a self,
        builder: &mut PassBuilder<'_, 'a>,
        pipeline_id: ComputePipelineId,
        label: &'static str,
        dispatch: [u32; 3],
        bind_group_label: Option<&'static str>,
        configure: F,
    ) -> StandardComputeNode<'a>
    where
        F: FnOnce(&mut TemplatePassBindingsBuilder<'a>),
    {
        let mut binding_builder = TemplatePassBindingsBuilder::new();
        configure(&mut binding_builder);
        let (bind_groups, bind_group_count) =
            self.build_bind_groups(builder, binding_builder.finish(), bind_group_label);

        StandardComputeNode {
            label,
            pipeline_id,
            dispatch,
            bind_groups,
            bind_group_count,
        }
    }
}

pub struct TemplateComputePass {
    descriptor: TemplatePassDescriptor,
    shader_options: ShaderCompilationOptions,
    pipeline_label: &'static str,
    pipeline_id: Option<ComputePipelineId>,
}

impl TemplateComputePass {
    #[must_use]
    pub fn new(
        descriptor: TemplatePassDescriptor,
        shader_options: ShaderCompilationOptions,
        pipeline_label: &'static str,
    ) -> Self {
        Self {
            descriptor,
            shader_options,
            pipeline_label,
            pipeline_id: None,
        }
    }

    pub fn set_shader_options(&mut self, shader_options: ShaderCompilationOptions) {
        self.shader_options = shader_options;
    }

    pub fn prepare_with_renderer(&mut self, renderer: &mut Renderer) -> ComputePipelineId {
        let pipeline_id = self.descriptor.compile_compute_with_renderer(
            renderer,
            &self.shader_options,
            self.pipeline_label,
        );
        self.pipeline_id = Some(pipeline_id);
        pipeline_id
    }

    pub fn prepare_with_extract(&mut self, ctx: &mut ExtractContext) -> ComputePipelineId {
        let pipeline_id = self.descriptor.compile_compute_with_extract(
            ctx,
            &self.shader_options,
            self.pipeline_label,
        );
        self.pipeline_id = Some(pipeline_id);
        pipeline_id
    }

    pub fn build_node<'a, F>(
        &'a self,
        builder: &mut PassBuilder<'_, 'a>,
        label: &'static str,
        dispatch: [u32; 3],
        bind_group_label: Option<&'static str>,
        configure: F,
    ) -> StandardComputeNode<'a>
    where
        F: FnOnce(&mut TemplatePassBindingsBuilder<'a>),
    {
        self.descriptor.build_compute_node(
            builder,
            self.pipeline_id
                .expect("Template compute pass must be prepared before graph build"),
            label,
            dispatch,
            bind_group_label,
            configure,
        )
    }
}

pub struct TemplateFullscreenPass {
    descriptor: TemplatePassDescriptor,
    shader_options: ShaderCompilationOptions,
    color_targets: SmallVec<[wgpu::ColorTargetState; 2]>,
    depth_stencil: Option<wgpu::DepthStencilState>,
    multisample: wgpu::MultisampleState,
    pipeline_label: &'static str,
    pipeline_id: Option<RenderPipelineId>,
}

impl TemplateFullscreenPass {
    #[must_use]
    pub fn new(
        descriptor: TemplatePassDescriptor,
        shader_options: ShaderCompilationOptions,
        pipeline_label: &'static str,
        color_targets: SmallVec<[wgpu::ColorTargetState; 2]>,
        depth_stencil: Option<wgpu::DepthStencilState>,
        multisample: wgpu::MultisampleState,
    ) -> Self {
        Self {
            descriptor,
            shader_options,
            color_targets,
            depth_stencil,
            multisample,
            pipeline_label,
            pipeline_id: None,
        }
    }

    pub fn set_shader_options(&mut self, shader_options: ShaderCompilationOptions) {
        self.shader_options = shader_options;
    }

    pub fn set_color_targets(&mut self, color_targets: &[wgpu::ColorTargetState]) {
        self.color_targets = color_targets.iter().cloned().collect();
    }

    pub fn set_depth_stencil(&mut self, depth_stencil: Option<wgpu::DepthStencilState>) {
        self.depth_stencil = depth_stencil;
    }

    pub fn set_multisample(&mut self, multisample: wgpu::MultisampleState) {
        self.multisample = multisample;
    }

    pub fn set_pipeline_label(&mut self, label: &'static str) {
        self.pipeline_label = label;
    }

    pub fn prepare_with_renderer(&mut self, renderer: &mut Renderer) -> RenderPipelineId {
        let pipeline_id = self.descriptor.compile_fullscreen_with_renderer(
            renderer,
            &self.shader_options,
            &self.color_targets,
            self.depth_stencil.clone(),
            self.multisample,
            self.pipeline_label,
        );
        self.pipeline_id = Some(pipeline_id);
        pipeline_id
    }

    pub fn prepare_with_extract(&mut self, ctx: &mut ExtractContext) -> RenderPipelineId {
        let pipeline_id = self.descriptor.compile_fullscreen_with_extract(
            ctx,
            &self.shader_options,
            &self.color_targets,
            self.depth_stencil.clone(),
            self.multisample,
            self.pipeline_label,
        );
        self.pipeline_id = Some(pipeline_id);
        pipeline_id
    }

    pub fn build_node<'a, F>(
        &'a self,
        builder: &mut PassBuilder<'_, 'a>,
        label: &'static str,
        output_tex: TextureNodeId,
        output_ops: RenderTargetOps,
        bind_group_label: Option<&'static str>,
        configure: F,
    ) -> StandardFullscreenNode<'a>
    where
        F: FnOnce(&mut TemplatePassBindingsBuilder<'a>),
    {
        self.descriptor.build_fullscreen_node(
            builder,
            self.pipeline_id
                .expect("Template fullscreen pass must be prepared before graph build"),
            label,
            output_tex,
            output_ops,
            bind_group_label,
            configure,
        )
    }
}

#[must_use]
pub struct RenderPassBuilder {
    inner: FullscreenPassTemplateBuilder,
}

impl RenderPassBuilder {
    pub fn fullscreen(pipeline_label: &'static str) -> Self {
        Self {
            inner: FullscreenPassTemplateBuilder::new(pipeline_label),
        }
    }

    pub fn shader_template(mut self, name: &'static str) -> Self {
        self.inner = self.inner.shader_template(name);
        self
    }

    pub fn inline_shader_template(mut self, name: &'static str, source: &'static str) -> Self {
        self.inner = self.inner.inline_shader_template(name, source);
        self
    }

    pub fn shader_options(mut self, shader_options: ShaderCompilationOptions) -> Self {
        self.inner = self.inner.shader_options(shader_options);
        self
    }

    pub fn bind_texture_2d(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) -> Self {
        self.inner = self
            .inner
            .bind_texture_2d(group, binding, visibility, filterable);
        self
    }

    pub fn bind_texture_cube(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) -> Self {
        self.inner = self
            .inner
            .bind_texture_cube(group, binding, visibility, filterable);
        self
    }

    pub fn bind_depth_texture_2d(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> Self {
        self.inner = self.inner.bind_depth_texture_2d(group, binding, visibility);
        self
    }

    pub fn bind_sampler(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        sampler_type: wgpu::SamplerBindingType,
    ) -> Self {
        self.inner = self
            .inner
            .bind_sampler(group, binding, visibility, sampler_type);
        self
    }

    pub fn bind_uniform_buffer(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> Self {
        self.inner = self.inner.bind_uniform_buffer(group, binding, visibility);
        self
    }

    pub fn bind_storage_buffer(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> Self {
        self.inner = self
            .inner
            .bind_storage_buffer(group, binding, visibility, read_only);
        self
    }

    pub fn bind_storage_texture(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        access: wgpu::StorageTextureAccess,
        format: wgpu::TextureFormat,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Self {
        self.inner = self.inner.bind_storage_texture(
            group,
            binding,
            visibility,
            access,
            format,
            view_dimension,
        );
        self
    }

    pub fn color_target(mut self, color_target: wgpu::ColorTargetState) -> Self {
        self.inner = self.inner.color_target(color_target);
        self
    }

    pub fn depth_stencil(mut self, depth_stencil: wgpu::DepthStencilState) -> Self {
        self.inner = self.inner.depth_stencil(depth_stencil);
        self
    }

    pub fn multisample(mut self, multisample: wgpu::MultisampleState) -> Self {
        self.inner = self.inner.multisample(multisample);
        self
    }

    #[must_use]
    pub fn finish(self) -> TemplateFullscreenPass {
        self.inner.finish()
    }

    #[must_use]
    pub fn build(self, renderer: &mut Renderer) -> TemplateFullscreenPass {
        self.inner.build(renderer)
    }
}

#[must_use]
pub struct ComputePassBuilder {
    inner: ComputePassTemplateBuilder,
}

impl ComputePassBuilder {
    pub fn new(pipeline_label: &'static str) -> Self {
        Self {
            inner: ComputePassTemplateBuilder::new(pipeline_label),
        }
    }

    pub fn shader_template(mut self, name: &'static str) -> Self {
        self.inner = self.inner.shader_template(name);
        self
    }

    pub fn inline_shader_template(mut self, name: &'static str, source: &'static str) -> Self {
        self.inner = self.inner.inline_shader_template(name, source);
        self
    }

    pub fn shader_options(mut self, shader_options: ShaderCompilationOptions) -> Self {
        self.inner = self.inner.shader_options(shader_options);
        self
    }

    pub fn bind_texture_2d(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) -> Self {
        self.inner = self
            .inner
            .bind_texture_2d(group, binding, visibility, filterable);
        self
    }

    pub fn bind_texture_cube(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) -> Self {
        self.inner = self
            .inner
            .bind_texture_cube(group, binding, visibility, filterable);
        self
    }

    pub fn bind_depth_texture_2d(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> Self {
        self.inner = self.inner.bind_depth_texture_2d(group, binding, visibility);
        self
    }

    pub fn bind_sampler(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        sampler_type: wgpu::SamplerBindingType,
    ) -> Self {
        self.inner = self
            .inner
            .bind_sampler(group, binding, visibility, sampler_type);
        self
    }

    pub fn bind_uniform_buffer(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> Self {
        self.inner = self.inner.bind_uniform_buffer(group, binding, visibility);
        self
    }

    pub fn bind_storage_buffer(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> Self {
        self.inner = self
            .inner
            .bind_storage_buffer(group, binding, visibility, read_only);
        self
    }

    pub fn bind_storage_texture(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        access: wgpu::StorageTextureAccess,
        format: wgpu::TextureFormat,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Self {
        self.inner = self.inner.bind_storage_texture(
            group,
            binding,
            visibility,
            access,
            format,
            view_dimension,
        );
        self
    }

    #[must_use]
    pub fn finish(self) -> TemplateComputePass {
        self.inner.finish()
    }

    #[must_use]
    pub fn build(self, renderer: &mut Renderer) -> TemplateComputePass {
        self.inner.build(renderer)
    }
}

struct FullscreenPassTemplateBuilder {
    shader_source: Option<TemplateShaderSource>,
    binding_layouts: SmallVec<[TemplateBindingLayoutDesc; 8]>,
    shader_options: ShaderCompilationOptions,
    color_targets: SmallVec<[wgpu::ColorTargetState; 2]>,
    depth_stencil: Option<wgpu::DepthStencilState>,
    multisample: wgpu::MultisampleState,
    pipeline_label: &'static str,
}

impl FullscreenPassTemplateBuilder {
    fn new(pipeline_label: &'static str) -> Self {
        Self {
            shader_source: None,
            binding_layouts: SmallVec::new(),
            shader_options: ShaderCompilationOptions::default(),
            color_targets: SmallVec::new(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            pipeline_label,
        }
    }

    fn shader_template(mut self, name: &'static str) -> Self {
        self.shader_source = Some(TemplateShaderSource::File(name));
        self
    }

    fn inline_shader_template(mut self, name: &'static str, source: &'static str) -> Self {
        self.shader_source = Some(TemplateShaderSource::Inline { name, source });
        self
    }

    fn shader_options(mut self, shader_options: ShaderCompilationOptions) -> Self {
        self.shader_options = shader_options;
        self
    }

    fn bind_texture_2d(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::texture_2d(
                group, binding, visibility, filterable,
            ));
        self
    }

    fn bind_texture_cube(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::texture_cube(
                group, binding, visibility, filterable,
            ));
        self
    }

    fn bind_depth_texture_2d(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::depth_texture_2d(
                group, binding, visibility,
            ));
        self
    }

    fn bind_sampler(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        sampler_type: wgpu::SamplerBindingType,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::sampler(
                group,
                binding,
                visibility,
                sampler_type,
            ));
        self
    }

    fn bind_uniform_buffer(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::uniform_buffer(
                group, binding, visibility,
            ));
        self
    }

    fn bind_storage_buffer(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::storage_buffer(
                group, binding, visibility, read_only,
            ));
        self
    }

    fn bind_storage_texture(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        access: wgpu::StorageTextureAccess,
        format: wgpu::TextureFormat,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::storage_texture(
                group,
                binding,
                visibility,
                access,
                format,
                view_dimension,
            ));
        self
    }

    fn color_target(mut self, color_target: wgpu::ColorTargetState) -> Self {
        self.color_targets.push(color_target);
        self
    }

    fn depth_stencil(mut self, depth_stencil: wgpu::DepthStencilState) -> Self {
        self.depth_stencil = Some(depth_stencil);
        self
    }

    fn multisample(mut self, multisample: wgpu::MultisampleState) -> Self {
        self.multisample = multisample;
        self
    }

    fn finish(self) -> TemplateFullscreenPass {
        let mut descriptor = TemplatePassDescriptor::new(
            self.shader_source
                .expect("Template fullscreen pass requires a shader template"),
        );
        for binding_layout in self.binding_layouts {
            descriptor.add_binding_layout(binding_layout);
        }

        TemplateFullscreenPass::new(
            descriptor,
            self.shader_options,
            self.pipeline_label,
            self.color_targets,
            self.depth_stencil,
            self.multisample,
        )
    }

    fn build(self, renderer: &mut Renderer) -> TemplateFullscreenPass {
        let mut pass = self.finish();
        pass.prepare_with_renderer(renderer);
        pass
    }
}

struct ComputePassTemplateBuilder {
    shader_source: Option<TemplateShaderSource>,
    binding_layouts: SmallVec<[TemplateBindingLayoutDesc; 8]>,
    shader_options: ShaderCompilationOptions,
    pipeline_label: &'static str,
}

impl ComputePassTemplateBuilder {
    fn new(pipeline_label: &'static str) -> Self {
        Self {
            shader_source: None,
            binding_layouts: SmallVec::new(),
            shader_options: ShaderCompilationOptions::default(),
            pipeline_label,
        }
    }

    fn shader_template(mut self, name: &'static str) -> Self {
        self.shader_source = Some(TemplateShaderSource::File(name));
        self
    }

    fn inline_shader_template(mut self, name: &'static str, source: &'static str) -> Self {
        self.shader_source = Some(TemplateShaderSource::Inline { name, source });
        self
    }

    fn shader_options(mut self, shader_options: ShaderCompilationOptions) -> Self {
        self.shader_options = shader_options;
        self
    }

    fn bind_texture_2d(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::texture_2d(
                group, binding, visibility, filterable,
            ));
        self
    }

    fn bind_texture_cube(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        filterable: bool,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::texture_cube(
                group, binding, visibility, filterable,
            ));
        self
    }

    fn bind_depth_texture_2d(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::depth_texture_2d(
                group, binding, visibility,
            ));
        self
    }

    fn bind_sampler(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        sampler_type: wgpu::SamplerBindingType,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::sampler(
                group,
                binding,
                visibility,
                sampler_type,
            ));
        self
    }

    fn bind_uniform_buffer(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::uniform_buffer(
                group, binding, visibility,
            ));
        self
    }

    fn bind_storage_buffer(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::storage_buffer(
                group, binding, visibility, read_only,
            ));
        self
    }

    fn bind_storage_texture(
        mut self,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        access: wgpu::StorageTextureAccess,
        format: wgpu::TextureFormat,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Self {
        self.binding_layouts
            .push(TemplateBindingLayoutDesc::storage_texture(
                group,
                binding,
                visibility,
                access,
                format,
                view_dimension,
            ));
        self
    }

    fn finish(self) -> TemplateComputePass {
        let mut descriptor = TemplatePassDescriptor::new(
            self.shader_source
                .expect("Template compute pass requires a shader template"),
        );
        for binding_layout in self.binding_layouts {
            descriptor.add_binding_layout(binding_layout);
        }

        TemplateComputePass::new(descriptor, self.shader_options, self.pipeline_label)
    }

    fn build(self, renderer: &mut Renderer) -> TemplateComputePass {
        let mut pass = self.finish();
        pass.prepare_with_renderer(renderer);
        pass
    }
}

#[derive(Clone, Copy)]
pub struct TemplatePassBinding<'a> {
    group: u32,
    binding: u32,
    resource: GraphBinding<'a>,
}

pub struct TemplatePassBindingsBuilder<'a> {
    bindings: SmallVec<[TemplatePassBinding<'a>; 8]>,
}

impl<'a> TemplatePassBindingsBuilder<'a> {
    fn new() -> Self {
        Self {
            bindings: SmallVec::new(),
        }
    }

    fn push(&mut self, group: u32, binding: u32, resource: GraphBinding<'a>) -> &mut Self {
        self.bindings.push(TemplatePassBinding {
            group,
            binding,
            resource,
        });
        self
    }

    pub fn bind_buffer(&mut self, group: u32, binding: u32, id: BufferNodeId) -> &mut Self {
        self.push(group, binding, GraphBinding::Buffer(id))
    }

    pub fn bind_texture(&mut self, group: u32, binding: u32, id: TextureNodeId) -> &mut Self {
        self.push(group, binding, GraphBinding::Texture(id))
    }

    pub fn bind_tracked_buffer(
        &mut self,
        group: u32,
        binding: u32,
        buffer: &'a Tracked<wgpu::Buffer>,
    ) -> &mut Self {
        self.push(group, binding, GraphBinding::TrackedBuffer(buffer))
    }

    pub fn bind_tracked_texture_view(
        &mut self,
        group: u32,
        binding: u32,
        view: &'a Tracked<wgpu::TextureView>,
    ) -> &mut Self {
        self.push(group, binding, GraphBinding::TrackedTextureView(view))
    }

    pub fn bind_tracked_sampler(
        &mut self,
        group: u32,
        binding: u32,
        sampler: &'a Tracked<wgpu::Sampler>,
    ) -> &mut Self {
        self.push(group, binding, GraphBinding::TrackedSampler(sampler))
    }

    pub fn bind_raw_buffer(
        &mut self,
        group: u32,
        binding: u32,
        buffer: RawBufferBinding<'a>,
    ) -> &mut Self {
        self.push(group, binding, GraphBinding::RawBuffer(buffer))
    }

    pub fn bind_raw_texture_view(
        &mut self,
        group: u32,
        binding: u32,
        view: RawTextureViewBinding<'a>,
    ) -> &mut Self {
        self.push(group, binding, GraphBinding::RawTextureView(view))
    }

    pub fn bind_raw_sampler(
        &mut self,
        group: u32,
        binding: u32,
        sampler: RawSamplerBinding<'a>,
    ) -> &mut Self {
        self.push(group, binding, GraphBinding::Sampler(sampler))
    }

    pub fn bind_common_sampler(
        &mut self,
        group: u32,
        binding: u32,
        sampler: CommonSampler,
    ) -> &mut Self {
        self.push(group, binding, GraphBinding::CommonSampler(sampler))
    }

    fn finish(self) -> SmallVec<[TemplatePassBinding<'a>; 8]> {
        self.bindings
    }
}

#[derive(Clone, Copy, Default)]
struct RuntimeBindGroupSlot<'a> {
    group: u32,
    bindings: &'a [TemplatePassBinding<'a>],
    label: Option<&'static str>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

pub struct StandardFullscreenNode<'a> {
    label: &'static str,
    pipeline_id: RenderPipelineId,
    output_tex: TextureNodeId,
    output_ops: RenderTargetOps,
    bind_groups: [RuntimeBindGroupSlot<'a>; MAX_TEMPLATE_BIND_GROUPS],
    bind_group_count: usize,
}

impl<'a> PassNode<'a> for StandardFullscreenNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        for slot in &mut self.bind_groups[..self.bind_group_count] {
            let mut bind_group_builder = ctx.build_render_pipeline_bind_group(
                self.pipeline_id,
                slot.group as usize,
                slot.label,
            );
            for binding in slot.bindings {
                bind_group_builder =
                    bind_group_builder.bind_graph_binding(binding.binding, binding.resource);
            }
            slot.bind_group = Some(bind_group_builder.build());
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let color_attachment = ctx.get_color_attachment(self.output_tex, self.output_ops, None);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(self.label),
            color_attachments: &[color_attachment],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(ctx.pipeline_cache.get_render_pipeline(self.pipeline_id));
        for slot in &self.bind_groups[..self.bind_group_count] {
            pass.set_bind_group(
                slot.group,
                slot.bind_group
                    .expect("Template fullscreen bind group was not prepared"),
                &[],
            );
        }
        pass.draw(0..3, 0..1);
    }
}

pub struct StandardComputeNode<'a> {
    label: &'static str,
    pipeline_id: ComputePipelineId,
    dispatch: [u32; 3],
    bind_groups: [RuntimeBindGroupSlot<'a>; MAX_TEMPLATE_BIND_GROUPS],
    bind_group_count: usize,
}

impl<'a> PassNode<'a> for StandardComputeNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        for slot in &mut self.bind_groups[..self.bind_group_count] {
            let mut bind_group_builder = ctx.build_compute_pipeline_bind_group(
                self.pipeline_id,
                slot.group as usize,
                slot.label,
            );
            for binding in slot.bindings {
                bind_group_builder =
                    bind_group_builder.bind_graph_binding(binding.binding, binding.resource);
            }
            slot.bind_group = Some(bind_group_builder.build());
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.label),
            timestamp_writes: None,
        });

        pass.set_pipeline(ctx.pipeline_cache.get_compute_pipeline(self.pipeline_id));
        for slot in &self.bind_groups[..self.bind_group_count] {
            pass.set_bind_group(
                slot.group,
                slot.bind_group
                    .expect("Template compute bind group was not prepared"),
                &[],
            );
        }
        pass.dispatch_workgroups(self.dispatch[0], self.dispatch[1], self.dispatch[2]);
    }
}
