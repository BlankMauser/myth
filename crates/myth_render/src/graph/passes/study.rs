//! Myth vs LunarFenrir study harness support (feature `study_post_chain`).
//!
//! Replaces the HighFidelity scene group with a deterministic input pass
//! (`Input_Gen`, the protocol's `gen_input.wgsl`) feeding Myth's own TAA,
//! CAS, bloom, tone-mapping and FXAA features unchanged, and optionally
//! copies every stage's output to readback buffers (`Study_Capture`).
//! See `/mnt/data/scratch/captain/myth-study/PROTOCOL.md`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use parking_lot::Mutex;

use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::Tracked;
use crate::core::gpu::binding_layout::uniform_buffer;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, PassNode, PrepareContext, RenderTargetOps,
    TextureDesc, TextureNodeId,
};

/// The generator WGSL, byte-identical to the protocol's `gen_input.wgsl`.
pub const GEN_INPUT_WGSL: &str = include_str!("study_gen_input.wgsl");

/// Protocol frame index `f` written into `GenParams.frame` (set by the harness).
pub static GEN_FRAME: AtomicU32 = AtomicU32::new(0);
/// Capture runs: adds COPY_SRC to the stage textures that lack it.
pub static CAPTURE_MODE: AtomicBool = AtomicBool::new(false);
/// Capture the stages of the next built frame (set by the harness).
pub static CAPTURE_THIS_FRAME: AtomicBool = AtomicBool::new(false);
/// Bit mask of stages to capture (bit i = i-th entry of the stage list in
/// `add_capture`: in_color, in_velocity, in_depth, taa, cas, bloom, tonemap,
/// fxaa). Default: all.
pub static CAPTURE_STAGES: AtomicU32 = AtomicU32::new(0xFF);
/// Readback buffers filled by the last captured frame.
pub static CAPTURED: Mutex<Vec<CaptureBuffer>> = Mutex::new(Vec::new());

/// Extra usage for stage textures (only in capture runs).
#[must_use]
pub fn capture_usage() -> wgpu::TextureUsages {
    if CAPTURE_MODE.load(Ordering::Relaxed) {
        wgpu::TextureUsages::COPY_SRC
    } else {
        wgpu::TextureUsages::empty()
    }
}

/// One captured stage, still in a mapped-on-demand GPU buffer.
pub struct CaptureBuffer {
    pub stage: &'static str,
    pub tag: &'static str,
    pub buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub padded_bytes_per_row: u32,
}

/// Texture node ids of every protocol stage in the current frame.
#[derive(Default, Clone, Copy)]
pub struct StudyStageIds {
    pub in_color: Option<TextureNodeId>,
    pub in_velocity: Option<TextureNodeId>,
    pub in_depth: Option<TextureNodeId>,
    pub taa: Option<TextureNodeId>,
    pub cas: Option<TextureNodeId>,
    pub bloom: Option<TextureNodeId>,
    pub tonemap: Option<TextureNodeId>,
    pub fxaa: Option<TextureNodeId>,
}

pub struct InputGenOutput {
    pub color: TextureNodeId,
    pub velocity: TextureNodeId,
    pub depth: TextureNodeId,
}

struct GenResources {
    layout: Tracked<wgpu::BindGroupLayout>,
    pipeline: wgpu::RenderPipeline,
}

static GEN_RESOURCES: OnceLock<GenResources> = OnceLock::new();

const GEN_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const GEN_VELOCITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

fn gen_resources(device: &wgpu::Device, depth_format: wgpu::TextureFormat) -> &'static GenResources {
    GEN_RESOURCES.get_or_init(|| {
        let layout = Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Study Input_Gen Layout"),
                entries: &[uniform_buffer(0, wgpu::ShaderStages::FRAGMENT)],
            },
        ));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Study gen_input.wgsl"),
            source: wgpu::ShaderSource::Wgsl(GEN_INPUT_WGSL.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Study Input_Gen Pipeline Layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Study Input_Gen Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: GEN_COLOR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GEN_VELOCITY_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        GenResources { layout, pipeline }
    })
}

/// Adds the `Input_Gen` pass (colour, velocity, depth for TAA).
pub fn add_input_gen(ctx: &mut GraphBuilderContext<'_, '_>) -> InputGenOutput {
    let fc = ctx.frame_config;
    let (w, h, depth_format) = (fc.width, fc.height, fc.depth_format);
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC;
    ctx.graph.add_pass("Input_Gen", |builder| {
        let color = builder.create_texture(
            "Study_Input_Color",
            TextureDesc::new_2d(w, h, GEN_COLOR_FORMAT, usage),
        );
        let velocity = builder.create_texture(
            "Study_Input_Velocity",
            TextureDesc::new_2d(w, h, GEN_VELOCITY_FORMAT, usage),
        );
        let depth = builder.create_texture(
            "Study_Input_Depth",
            TextureDesc::new_2d(w, h, depth_format, usage),
        );
        let params = builder.create_buffer(
            "Study_Gen_Params",
            BufferDesc::new(16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
        );
        let node = InputGenNode {
            width: w,
            height: h,
            depth_format,
            color,
            velocity,
            depth,
            params,
            bind_group: None,
        };
        (
            node,
            InputGenOutput {
                color,
                velocity,
                depth,
            },
        )
    })
}

struct InputGenNode {
    width: u32,
    height: u32,
    depth_format: wgpu::TextureFormat,
    color: TextureNodeId,
    velocity: TextureNodeId,
    depth: TextureNodeId,
    params: BufferNodeId,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for InputGenNode {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let res = gen_resources(ctx.device, self.depth_format);
        let frame = GEN_FRAME.load(Ordering::Relaxed);
        let params: [i32; 4] = [
            i32::try_from(self.width).expect("width fits i32"),
            i32::try_from(self.height).expect("height fits i32"),
            i32::try_from(frame).expect("frame fits i32"),
            0,
        ];
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.params),
            0,
            bytemuck::cast_slice(&params),
        );
        self.bind_group = Some(crate::myth_bind_group!(
            ctx,
            &res.layout,
            Some("Study Input_Gen BG"),
            [0 => self.params]
        ));
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let res = GEN_RESOURCES.get().expect("Input_Gen resources prepared");
        let clear = RenderTargetOps::Clear(wgpu::Color::TRANSPARENT);
        let color = ctx.get_color_attachment(self.color, clear, None);
        let velocity = ctx.get_color_attachment(self.velocity, clear, None);
        let depth = ctx.get_depth_stencil_attachment(self.depth, 0.0);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Input_Gen"),
            color_attachments: &[color, velocity],
            depth_stencil_attachment: depth,
            ..Default::default()
        });
        pass.set_pipeline(&res.pipeline);
        pass.set_bind_group(
            0,
            ctx.resolve_prepared_bind_group(self.bind_group.expect("Input_Gen BG prepared")),
            &[],
        );
        pass.draw(0..3, 0..1);
    }
}

// ─── Capture ───────────────────────────────────────────────────────────────

const MAX_STAGES: usize = 8;

#[derive(Clone, Copy)]
struct StageSpec {
    stage: &'static str,
    tag: &'static str,
    id: TextureNodeId,
    bytes_per_pixel: u32,
    depth: bool,
}

/// Adds a side-effect pass copying every known stage to readback buffers,
/// only when the harness requested a capture of this frame.
pub fn add_capture(ctx: &mut GraphBuilderContext<'_, '_>, ids: &StudyStageIds) {
    if !CAPTURE_THIS_FRAME.swap(false, Ordering::Relaxed) {
        return;
    }
    let list: [(&'static str, &'static str, Option<TextureNodeId>, u32, bool); MAX_STAGES] = [
        ("in_color", "rgba16f", ids.in_color, 8, false),
        ("in_velocity", "rg16f", ids.in_velocity, 4, false),
        ("in_depth", "r32f", ids.in_depth, 4, true),
        ("taa", "rgba16f", ids.taa, 8, false),
        ("cas", "rgba16f", ids.cas, 8, false),
        ("bloom", "rgba16f", ids.bloom, 8, false),
        ("tonemap", "rgba8", ids.tonemap, 4, false),
        ("fxaa", "rgba8", ids.fxaa, 4, false),
    ];
    let (w, h) = (ctx.frame_config.width, ctx.frame_config.height);
    ctx.graph.add_pass("Study_Capture", |builder| {
        let mut stages = [None; MAX_STAGES];
        let mask = CAPTURE_STAGES.load(Ordering::Relaxed);
        for (bit, (slot, (stage, tag, id, bpp, depth))) in
            stages.iter_mut().zip(list).enumerate()
        {
            if mask & (1 << bit) == 0 {
                continue;
            }
            if let Some(id) = id {
                builder.read_texture(id);
                *slot = Some(StageSpec {
                    stage,
                    tag,
                    id,
                    bytes_per_pixel: bpp,
                    depth,
                });
            }
        }
        builder.mark_side_effect();
        (
            CaptureNode {
                width: w,
                height: h,
                stages,
                first_buffer: 0,
            },
            (),
        )
    });
}

struct CaptureNode {
    width: u32,
    height: u32,
    stages: [Option<StageSpec>; MAX_STAGES],
    first_buffer: usize,
}

fn padded_row(width: u32, bpp: u32) -> u32 {
    (width * bpp).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

impl<'a> PassNode<'a> for CaptureNode {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let mut out = CAPTURED.lock();
        self.first_buffer = out.len();
        for spec in self.stages.iter().flatten() {
            let row = padded_row(self.width, spec.bytes_per_pixel);
            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Study Capture Readback"),
                size: u64::from(row) * u64::from(self.height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            out.push(CaptureBuffer {
                stage: spec.stage,
                tag: spec.tag,
                buffer,
                width: self.width,
                height: self.height,
                bytes_per_pixel: spec.bytes_per_pixel,
                padded_bytes_per_row: row,
            });
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let out = CAPTURED.lock();
        for (index, spec) in self.stages.iter().flatten().enumerate() {
            let target = &out[self.first_buffer + index];
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: ctx.get_texture(spec.id),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: if spec.depth {
                        wgpu::TextureAspect::DepthOnly
                    } else {
                        wgpu::TextureAspect::All
                    },
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &target.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(target.padded_bytes_per_row),
                        rows_per_image: Some(self.height),
                    },
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}
