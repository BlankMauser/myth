//! Debug View Feature + Ephemeral PassNode
//!
//! Provides runtime visualisation of intermediate RDG textures (depth,
//! normals, velocity, SSAO, bloom, etc.) without invasive pipeline changes.
//!
//! - **`DebugViewFeature`** (long-lived): owns the pipeline, bind-group
//!   layouts, and a small uniform buffer for the `view_mode` selector.
//!   `extract()` captures only the per-frame configuration. Pipeline creation,
//!   buffer upload, and bind-group assembly are deferred until the node
//!   survives graph compilation.
//!
//! - **`DebugViewPassNode`** (ephemeral per-frame): carries the source
//!   texture ID, target surface ID, and borrowed references to the
//!   Feature's persistent resources.  Created by
//!   `DebugViewFeature::add_to_graph()` each frame.
//!
//! # Bind Group Model
//!
//! - **Group 0 (static)**: sampler + uniform buffer — Feature-owned,
//!   rebuilt only when the uniform buffer identity changes.
//! - **Group 1 (transient)**: source texture — PassNode-owned, rebuilt
//!   each frame during the RDG prepare phase.
//!
//! # Safety
//!
//! The entire module is gated behind `#[cfg(feature = "debug_view")]`.
//! When the feature is disabled, no code is compiled and the engine has
//! zero overhead from this system.

#![cfg(feature = "debug_view")]

use bytemuck::{Pod, Zeroable};
use wgpu::CommandEncoder;

use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ClusteredScreenBindings, ExecuteContext, PassNode, PrepareContext, RenderTargetOps,
    TextureNodeId,
};
use crate::graph::render_state::DebugViewTarget;
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::buffer::CpuBuffer;
use myth_resources::uniforms::clustered_lighting_structs_wgsl;
use myth_scene::camera::RenderCamera;

// ─── GPU Uniform Layout ─────────────────────────────────────────────────────

/// Maps to the `DebugUniforms` struct in `debug_view.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DebugViewUniforms {
    pub view_mode: u32,
    pub custom_scale: f32,
    pub z_near: f32,
    pub z_far: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct DebugViewSnapshot {
    output_format: wgpu::TextureFormat,
    params: DebugViewUniforms,
    is_depth: bool,
}

impl Default for DebugViewUniforms {
    fn default() -> Self {
        Self {
            view_mode: 0,
            custom_scale: 100.0,
            z_near: 0.1,
            z_far: 100.0,
        }
    }
}

impl myth_resources::buffer::GpuData for DebugViewUniforms {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
    fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RendererState)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Persistent resources for the debug-view overlay pass.
pub struct DebugViewFeature {
    /// L1 cache key: surface format the pipeline was compiled for.
    l1_cache_format: Option<wgpu::TextureFormat>,
    l1_cache_is_depth: Option<bool>,
    pipeline_id: Option<RenderPipelineId>,

    /// Group 0 (static): sampler + uniform buffer.
    static_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Group 1 (transient): source texture.
    transient_layout_color: Option<Tracked<wgpu::BindGroupLayout>>,
    transient_layout_depth: Option<Tracked<wgpu::BindGroupLayout>>,

    /// Feature-owned static bind group (Group 0).
    static_bg: Option<wgpu::BindGroup>,
    /// Staleness tracking for uniform buffer identity.
    last_uniforms_buffer_id: u64,

    uniforms: CpuBuffer<DebugViewUniforms>,
}

impl Default for DebugViewFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugViewFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            l1_cache_format: None,
            l1_cache_is_depth: None,
            pipeline_id: None,
            static_layout: None,
            transient_layout_color: None,
            transient_layout_depth: None,
            static_bg: None,
            last_uniforms_buffer_id: 0,
            uniforms: CpuBuffer::new(
                DebugViewUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("DebugView Uniforms"),
            ),
        }
    }

    /// Lazily create the two bind group layouts.
    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.static_layout.is_some() {
            return;
        }

        // Group 0 (static): sampler + uniforms
        self.static_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("DebugView Static Layout (G0)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        )));

        // Group 1 (transient): source texture
        self.transient_layout_color = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("DebugView Transient Layout Color (G1)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        )));

        self.transient_layout_depth = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("DebugView Transient Layout Depth (G1)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        )));
    }

    /// Capture the lightweight, topology-independent state needed by the
    /// debug-view node this frame. This phase performs no GPU interaction.
    pub fn extract(
        &self,
        camera: &RenderCamera,
        output_format: wgpu::TextureFormat,
        target: DebugViewTarget,
    ) -> Option<DebugViewSnapshot> {
        let debug_view = camera.debug_view;
        let view_mode = debug_view.mode.post_process_view_mode()?;

        Some(DebugViewSnapshot {
            output_format,
            params: DebugViewUniforms {
                view_mode,
                custom_scale: debug_view.custom_scale,
                z_near: camera.near,
                z_far: if camera.far.is_infinite() {
                    10_000.0
                } else {
                    camera.far
                },
            },
            is_depth: target.is_depth(),
        })
    }

    /// Materialize persistent resources after the node survives DPE.
    fn prepare_resources(&mut self, ctx: &mut PrepareContext<'_>, snapshot: DebugViewSnapshot) {
        let DebugViewSnapshot {
            output_format,
            params,
            is_depth,
        } = snapshot;
        self.ensure_layouts(ctx.device);

        // Update the CPU-side uniform and flush to GPU.
        {
            let mut guard = self.uniforms.write();
            *guard = params;
        }
        ctx.uploads.ensure_buffer(&self.uniforms);

        // ── Pipeline (re)creation on format change ─────────────────
        if self.l1_cache_format != Some(output_format) || self.l1_cache_is_depth != Some(is_depth) {
            let mut options = ShaderCompilationOptions::default();
            options.inject_code(
                "clustered_lighting_structs",
                clustered_lighting_structs_wgsl(),
            );

            if is_depth {
                options.add_define("IS_DEPTH", "1");
            }

            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/post_process/debug_view"),
                &options,
            );

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: output_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key = FullscreenPipelineKey::fullscreen(
                shader_hash,
                smallvec::smallvec![color_target],
                None,
            );

            let transient_layout = if is_depth {
                self.transient_layout_depth.as_deref()
            } else {
                self.transient_layout_color.as_deref()
            };

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("DebugView Pipeline Layout"),
                        bind_group_layouts: &[self.static_layout.as_deref(), transient_layout],
                        immediate_size: 0,
                    });

            self.pipeline_id = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                shader_module,
                &pipeline_layout,
                &key,
                "DebugView Pipeline",
            ));
            self.l1_cache_format = Some(output_format);
            self.l1_cache_is_depth = Some(is_depth);
        }

        // ── Static bind group (Group 0) — rebuild on buffer identity change
        if let Some(handle) = self.uniforms.gpu_handle()
            && let Some(gpu_buf) = ctx.uploads.get_gpu_buffer(handle)
            && (self.static_bg.is_none() || self.last_uniforms_buffer_id != gpu_buf.id)
        {
            let sampler = ctx.sampler_registry.get_common(CommonSampler::NearestClamp);
            let layout = self.static_layout.as_ref().unwrap();

            self.static_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DebugView Static BG (G0)"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gpu_buf.buffer.as_entire_binding(),
                    },
                ],
            }));
            self.last_uniforms_buffer_id = gpu_buf.id;
        }
    }

    /// Inject the debug-view pass into the render graph.
    ///
    /// Reads `source_tex`, writes to `target_surface` via SSA relay, and
    /// returns the updated surface handle.
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: DebugViewSnapshot,
        source_tex: TextureNodeId,
        target_surface: TextureNodeId,
        clustered: ClusteredScreenBindings,
    ) -> TextureNodeId {
        ctx.graph.add_pass("DebugView_Pass", |builder| {
            builder.read_texture(source_tex);
            if let Some(params) = clustered.params {
                builder.read_buffer(params);
            }
            if let Some(records) = clustered.records {
                builder.read_buffer(records);
            }
            let output = builder.replace_texture(target_surface, "DebugView_Output");

            let node = DebugViewPassNode {
                feature: self,
                snapshot,
                source_tex,
                output_tex: output,
                clustered_params: clustered.params,
                clustered_records: clustered.records,
                transient_bg: None,
            };
            (node, output)
        })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct DebugViewPassNode<'a> {
    feature: &'a mut DebugViewFeature,
    snapshot: DebugViewSnapshot,
    source_tex: TextureNodeId,
    output_tex: TextureNodeId,
    clustered_params: Option<crate::graph::core::BufferNodeId>,
    clustered_records: Option<crate::graph::core::BufferNodeId>,
    /// Transient bind group built in `prepare()`.
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for DebugViewPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.prepare_resources(ctx, self.snapshot);

        let transient_layout = if self.snapshot.is_depth {
            self.feature.transient_layout_depth.as_ref().unwrap()
        } else {
            self.feature.transient_layout_color.as_ref().unwrap()
        };
        let fallback_params = &ctx.system_textures.clustered_params;
        let fallback_records = &ctx.system_textures.clustered_records;

        let mut builder = ctx
            .build_bind_group(transient_layout, Some("DebugView Transient BG (G1)"))
            .bind_texture(0, self.source_tex);

        builder = if let Some(params) = self.clustered_params {
            builder.bind_buffer(1, params)
        } else {
            builder.bind_tracked_buffer(1, fallback_params)
        };

        builder = if let Some(records) = self.clustered_records {
            builder.bind_buffer(2, records)
        } else {
            builder.bind_tracked_buffer(2, fallback_records)
        };

        self.transient_bg = Some(builder.build());
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut CommandEncoder) {
        let transient_bg = self
            .transient_bg
            .expect("DebugView transient BG not prepared!");
        let pipeline_id = self
            .feature
            .pipeline_id
            .expect("DebugView pipeline must exist after prepare");
        let pipeline = ctx.pipeline_cache.get_render_pipeline(pipeline_id);
        let static_bg = self
            .feature
            .static_bg
            .as_ref()
            .expect("DebugView static BG must exist after prepare");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("DebugView Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, static_bg, &[]);
        rpass.set_bind_group(1, transient_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}
