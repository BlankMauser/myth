//! SSSS Feature + Ephemeral PassNodes (Flattened)
//!
//! - **`SsssFeature`** (long-lived): owns the lazily-created shared layout and
//!   profiles storage buffer.
//! - **`SsssHorizontalNode`** / **`SsssVerticalNode`** (ephemeral per-frame):
//!   two independent RDG passes created by `SsssFeature::add_to_graph()`.
//!   Each live node resolves only its own pipeline during deferred prepare.
//!
//! Implements a separable Gaussian blur for SSS materials identified via
//! a **data-driven soft mask**: the `Feature_ID` colour attachment written
//! by the Prepass carries per-pixel `sss_id` / `ssr_id`.  The shader reads
//! this texture and performs an early-out for non-SSS pixels, replacing the
//! former hardware stencil test with zero cross-platform compatibility cost.
//!
//! # Data Flow (explicit wiring)
//!
//! ```text
//!  Opaque                            SsssPassNode
//!       |                 +-----------------------------------+
//! color_in  ------------>|  H Sub-Pass: Horizontal blur      |---> temp_blur
//! normal_in ----+------->|                                   |         |
//! depth_in  ----|        |  V Sub-Pass: Vertical blur        |<--------+
//! feature_id ---|        |  (writes back to color_in         |
//! specular_tex -+        |   in-place)                       |---> color_in
//!                        +-----------------------------------+
//! ```
//!
//! # Integration
//!
//! Must come **after** `OpaquePass` and **before** the Skybox/MSAA-Sync
//! stage in the `HighFidelity` render path.  The Feature only calls
//! `add_to_graph` when SSS is enabled — zero cost when disabled.
//!
//! # GPU Resources
//!
//! - **Profiles StorageBuffer**: 256 x `SssProfileData` (12 KB).
//!   Uploaded from `AssetServer.sss_registry`.
//! - **temp_blur TransientTexture**: `Rgba16Float`, same size as scene colour.

use crate::HDR_TEXTURE_FORMAT;
use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, FrameArena, PassNode, PrepareContext, RenderTargetOps, TextureDesc,
    TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::ssss::{SssProfileData, SssRegistry};
use std::mem::size_of;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RenderFeatures)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Clone, Copy)]
pub struct SsssSnapshot<'a> {
    registry_version: u64,
    profiles: &'a [SssProfileData; 256],
}

/// Long-lived SSSS feature. CPU extraction remains lightweight; persistent GPU
/// resources are initialized only by a live horizontal node after DPE.
pub struct SsssFeature {
    bind_group_layout: OnceLock<Tracked<wgpu::BindGroupLayout>>,
    profiles_buffer: OnceLock<Tracked<wgpu::Buffer>>,
    last_uploaded_registry_version: AtomicU64,
}

impl Default for SsssFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SsssFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            bind_group_layout: OnceLock::new(),
            profiles_buffer: OnceLock::new(),
            last_uploaded_registry_version: AtomicU64::new(u64::MAX),
        }
    }

    /// Capture the current registry into frame-local POD storage. This performs
    /// no WGPU work and does not retain per-frame snapshot state on the feature.
    pub fn extract<'a>(&self, arena: &'a FrameArena, registry: &SssRegistry) -> SsssSnapshot<'a> {
        SsssSnapshot {
            registry_version: registry.version,
            profiles: arena.alloc(registry.buffer_data),
        }
    }

    fn prepare_shared_resources(&self, ctx: &PrepareContext<'_>, snapshot: SsssSnapshot<'_>) {
        self.bind_group_layout
            .get_or_init(|| Tracked::new(create_bind_group_layout(ctx.device)));

        let profiles_buffer = self.profiles_buffer.get_or_init(|| {
            Tracked::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SSS Profiles Buffer"),
                size: (256 * size_of::<SssProfileData>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });

        if self.last_uploaded_registry_version.load(Ordering::Acquire) != snapshot.registry_version
        {
            ctx.queue
                .write_buffer(profiles_buffer, 0, bytemuck::cast_slice(snapshot.profiles));
            self.last_uploaded_registry_version
                .store(snapshot.registry_version, Ordering::Release);
        }
    }

    fn bind_group_layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.bind_group_layout
            .get()
            .expect("SSSS shared layout must be prepared by the horizontal node")
    }

    fn profiles_buffer(&self) -> &Tracked<wgpu::Buffer> {
        self.profiles_buffer
            .get()
            .expect("SSSS profiles buffer must be prepared by the horizontal node")
    }

    /// Build the ephemeral pass nodes and insert them into the graph as
    /// two independent RDG passes within an `"SSSS_System"` group.
    ///
    /// All inputs are explicitly wired — no blackboard lookups.
    /// The pass modifies `scene_color` in-place (read + write via alias).
    ///
    /// # Flattened Pass Chain
    ///
    /// 1. **SSSS_Blur_H** — horizontal scatter: `scene_color` → `temp_blur`
    /// 2. **SSSS_Blur_V** — vertical scatter: `temp_blur` → `scene_color` alias
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: SsssSnapshot<'a>,
        scene_color: TextureNodeId,
        scene_depth: TextureNodeId,
        scene_normals: TextureNodeId,
        feature_id: TextureNodeId,
        specular_tex: TextureNodeId,
    ) -> TextureNodeId {
        ctx.with_group("SSSS_System", |ctx| {
            // ─── Pass 1: Horizontal blur ───────────────────────────
            let fc = ctx.frame_config;
            let (w, h) = (fc.width, fc.height);
            let hdr_format = fc.hdr_format;
            let temp_desc = TextureDesc::new_2d(
                w,
                h,
                hdr_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let temp_blur: TextureNodeId = ctx.graph.add_pass("SSSS_Blur_H", |builder| {
                builder.read_texture(scene_color);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                builder.read_texture(feature_id);
                builder.read_texture(specular_tex);
                let out = builder.create_texture("SSSS_Temp", temp_desc);
                let node = SsssHorizontalNode {
                    feature: self,
                    snapshot,
                    scene_color_in: scene_color,
                    temp_blur: out,
                    depth_in: scene_depth,
                    normal_in: scene_normals,
                    feature_id,
                    specular_tex,
                    pipeline_id: None,
                    bind_group: None,
                };
                (node, out)
            });

            // ─── Pass 2: Vertical blur ─────────────────────────────
            let ssss_out: TextureNodeId = ctx.graph.add_pass("SSSS_Blur_V", |builder| {
                builder.read_texture(temp_blur);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                builder.read_texture(feature_id);
                builder.read_texture(specular_tex);
                let out = builder.mutate_texture(scene_color, "Scene_Color_SSSS");
                let node = SsssVerticalNode {
                    feature: self,
                    scene_color_out: out,
                    temp_blur,
                    depth_in: scene_depth,
                    normal_in: scene_normals,
                    feature_id,
                    specular_tex,
                    pipeline_id: None,
                    bind_group: None,
                };
                (node, out)
            });

            ssss_out
        })
    }
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SSSS Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn prepare_pipeline(
    ctx: &mut PrepareContext<'_>,
    bind_group_layout: &Tracked<wgpu::BindGroupLayout>,
    vertical: bool,
) -> RenderPipelineId {
    let mut options = ShaderCompilationOptions::default();
    if vertical {
        options.add_define("SSSS_VERTICAL_PASS", "1");
    }

    let (shader, shader_hash) = ctx.shader_manager.get_or_compile(
        ctx.device,
        ShaderSource::File("entry/features/ssss/blur"),
        &options,
    );

    let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
        format: HDR_TEXTURE_FORMAT,
        blend: Some(wgpu::BlendState::REPLACE),
        write_mask: wgpu::ColorWrites::ALL,
    });
    let key =
        FullscreenPipelineKey::fullscreen(shader_hash, smallvec::smallvec![color_target], None);

    if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
        return id;
    }

    let raw_layout: &wgpu::BindGroupLayout = bind_group_layout;
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(if vertical {
                "SSSS Vertical Pipeline Layout"
            } else {
                "SSSS Horizontal Pipeline Layout"
            }),
            bind_group_layouts: &[Some(raw_layout)],
            immediate_size: 0,
        });
    let label = if vertical {
        "SSSS Vertical Pipeline"
    } else {
        "SSSS Horizontal Pipeline"
    };
    let id = ctx.pipeline_cache.get_or_create_fullscreen(
        ctx.device,
        shader,
        &pipeline_layout,
        &key,
        label,
    );
    ctx.pipeline_cache
        .register_render_layouts(id, smallvec::smallvec![bind_group_layout.clone()]);
    id
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pass 1: SsssHorizontalNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral per-frame node for the horizontal SSS scatter pass.
///
/// Reads scene colour and writes to the scratch `temp_blur` texture.
/// Non-SSS pixels are skipped via shader early-out on `Feature_ID`.
struct SsssHorizontalNode<'a> {
    feature: &'a SsssFeature,
    snapshot: SsssSnapshot<'a>,
    scene_color_in: TextureNodeId,
    temp_blur: TextureNodeId,
    depth_in: TextureNodeId,
    normal_in: TextureNodeId,
    feature_id: TextureNodeId,
    specular_tex: TextureNodeId,

    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsssHorizontalNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.prepare_shared_resources(ctx, self.snapshot);
        let bind_group_layout = self.feature.bind_group_layout();
        let profiles_buffer = self.feature.profiles_buffer();
        self.pipeline_id = Some(prepare_pipeline(ctx, bind_group_layout, false));
        self.bind_group = Some(
            crate::myth_bind_group!(ctx, bind_group_layout, Some("SSSS Horizontal Bind Group"), [
                0 => self.scene_color_in,
                1 => self.normal_in,
                2 => self.depth_in,
                3 => profiles_buffer,
                4 => CommonSampler::LinearClamp,
                5 => self.feature_id,
                6 => self.specular_tex,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let rtt = ctx.get_color_attachment(
            self.temp_blur,
            RenderTargetOps::Clear(wgpu::Color::TRANSPARENT),
            None,
        );

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSSS Horizontal"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(
            ctx.pipeline_cache.get_render_pipeline(
                self.pipeline_id
                    .expect("SSSS horizontal pipeline not prepared"),
            ),
        );
        rpass.set_bind_group(0, self.bind_group.unwrap(), &[]);
        rpass.draw(0..3, 0..1);
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pass 2: SsssVerticalNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral per-frame node for the vertical SSS scatter pass.
///
/// Reads the `temp_blur` scratch texture and writes back to the scene colour
/// alias (via `mutate_texture`).  Non-SSS pixels pass through unchanged
/// thanks to the shader soft-mask on `Feature_ID`.
struct SsssVerticalNode<'a> {
    feature: &'a SsssFeature,
    scene_color_out: TextureNodeId,
    temp_blur: TextureNodeId,
    depth_in: TextureNodeId,
    normal_in: TextureNodeId,
    feature_id: TextureNodeId,
    specular_tex: TextureNodeId,

    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsssVerticalNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        // The vertical pass reads the horizontal pass's transient output, so
        // RDG topological order guarantees shared preparation has completed.
        let bind_group_layout = self.feature.bind_group_layout();
        let profiles_buffer = self.feature.profiles_buffer();
        self.pipeline_id = Some(prepare_pipeline(ctx, bind_group_layout, true));
        self.bind_group = Some(
            crate::myth_bind_group!(ctx, bind_group_layout, Some("SSSS Vertical Bind Group"), [
                0 => self.temp_blur,
                1 => self.normal_in,
                2 => self.depth_in,
                3 => profiles_buffer,
                4 => CommonSampler::LinearClamp,
                5 => self.feature_id,
                6 => self.specular_tex,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let rtt = ctx.get_color_attachment(self.scene_color_out, RenderTargetOps::Load, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSSS Vertical"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(
            ctx.pipeline_cache.get_render_pipeline(
                self.pipeline_id
                    .expect("SSSS vertical pipeline not prepared"),
            ),
        );
        rpass.set_bind_group(0, self.bind_group.unwrap(), &[]);
        rpass.draw(0..3, 0..1);
    }
}
