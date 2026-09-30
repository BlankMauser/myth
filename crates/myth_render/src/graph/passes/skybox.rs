//! RDG Skybox / Background Render Pass
//!
//! Renders the scene background (gradient, cubemap, equirectangular, or planar
//! texture) as a fullscreen triangle at the far depth plane. Uses Reverse-Z
//! depth testing (`GreaterEqual`) so opaque geometry masks the background.
//!
//! # RDG Slots
//!
//! - `scene_color`: HDR color buffer (read + write, LoadOp::Load)
//! - `scene_depth`: Depth buffer (read, LoadOp::Load)
//!
//! # Push Parameters (set by Composer)
//!
//! All scene-level configuration is pushed into the public fields by the
//! Composer before the RDG prepare loop. The pass itself never accesses
//! `Scene` directly.
//!
//! - `background_mode`: Background rendering mode (gradient, texture, etc.)
//! - `bg_uniforms_cpu_id`: CPU buffer ID for `CpuBuffer<SkyboxParamsUniforms>`
//! - `bg_uniforms_gpu_id`: GPU buffer ID (from `ensure_buffer_id`)
//! - `scene_id`: Scene unique ID for global state lookup

use rustc_hash::FxHashMap;

use crate::core::gpu::{
    CommonSampler, GlobalStateKey, ProceduralEnvironmentSnapshot, ResourceState, Tracked,
};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, ExtractContext, PassBuilder, PassNode,
    PrepareContext, RawBufferBinding, RenderTargetOps, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, DepthStencilKey, FullscreenPipelineKey, MultisampleKey, PipelineCache,
    RenderPipelineId, ShaderCompilationOptions, ShaderManager, ShaderSource,
};
use myth_resources::buffer::CpuBuffer;
use myth_resources::shader_defines::ShaderDefines;
use myth_resources::texture::TextureSampler;
use myth_resources::texture::TextureSource;
use myth_resources::uniforms::WgslStruct;
use myth_scene::background::{
    BackgroundMapping, BackgroundMode, ProceduralSkyParams, SkyboxParamsUniforms,
};

/// Sampler key for the skybox environment map: trilinear filtering with
/// horizontal repeat (seamless panorama wrap) and vertical/depth clamp.
const SKYBOX_SAMPLER_KEY: TextureSampler = TextureSampler {
    address_mode_u: wgpu::AddressMode::Repeat,
    address_mode_v: wgpu::AddressMode::ClampToEdge,
    address_mode_w: wgpu::AddressMode::ClampToEdge,
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    mipmap_filter: wgpu::MipmapFilterMode::Linear,
    lod_min_clamp: 0.0,
    lod_max_clamp: 32.0,
    compare: None,
    anisotropy_clamp: Some(1),
    border_color: None,
};

// ─── Pipeline Variant Key ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SkyboxPipelineKey {
    variant: SkyboxVariant,
    procedural_starbox: ProceduralStarboxKind,
    procedural_moon_texture: bool,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
    global_layout_id: u64,
    shader_source_identity: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreparedTextureTargetKey {
    variant: SkyboxVariant,
    global_state_key: (u32, u32),
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ProceduralStarboxKind {
    None,
    Equirectangular,
    Cube,
}

impl ProceduralStarboxKind {
    fn from_view_dimension(view_dimension: wgpu::TextureViewDimension) -> Option<Self> {
        match view_dimension {
            wgpu::TextureViewDimension::D2 => Some(Self::Equirectangular),
            wgpu::TextureViewDimension::Cube => Some(Self::Cube),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SkyboxVariant {
    Gradient,
    Cube,
    Equirectangular,
    Planar,
    Procedural,
}

impl SkyboxVariant {
    fn from_background(mode: &BackgroundMode) -> Option<Self> {
        match mode {
            BackgroundMode::Color(_) => None,
            BackgroundMode::Gradient { .. } => Some(Self::Gradient),
            BackgroundMode::Texture { mapping, .. } => match mapping {
                BackgroundMapping::Cube => Some(Self::Cube),
                BackgroundMapping::Equirectangular => Some(Self::Equirectangular),
                BackgroundMapping::Planar => Some(Self::Planar),
            },
            BackgroundMode::Procedural(_) => Some(Self::Procedural),
        }
    }

    fn shader_define_key(self) -> &'static str {
        match self {
            Self::Gradient => "SKYBOX_GRADIENT",
            Self::Cube => "SKYBOX_CUBE",
            Self::Equirectangular => "SKYBOX_EQUIRECT",
            Self::Planar => "SKYBOX_PLANAR",
            Self::Procedural => "SKYBOX_PROCEDURAL",
        }
    }

    fn needs_texture(self) -> bool {
        !matches!(self, Self::Gradient)
    }

    fn apply_shader_defines(
        self,
        defines: &mut ShaderDefines,
        starbox_kind: ProceduralStarboxKind,
        moon_texture: bool,
    ) {
        defines.set(self.shader_define_key(), "1");
        if self == Self::Gradient {
            // Wire blue-noise dithering for the gradient banding fix; the shader
            // keeps a procedural-hash fallback when the texture is unavailable.
            defines.set("USE_BLUE_NOISE", "1");
        }
        if self == Self::Procedural {
            match starbox_kind {
                ProceduralStarboxKind::None => {}
                ProceduralStarboxKind::Equirectangular => {
                    defines.set("CELESTIAL_STARBOX_EQUIRECT", "1");
                }
                ProceduralStarboxKind::Cube => {
                    defines.set("CELESTIAL_STARBOX_CUBE", "1");
                }
            }
            if moon_texture {
                defines.set("USE_MOON_TEXTURE", "1");
            }
        }
    }
}

#[derive(Clone, Copy)]
struct ResolvedTextureView<'a> {
    view: &'a wgpu::TextureView,
    view_dimension: wgpu::TextureViewDimension,
    resource_key: u64,
}

/// Logical atmosphere outputs sampled by a procedural skybox draw.
///
/// These IDs are the actual bind-group inputs, not artificial ordering edges.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProceduralSkyboxInputs {
    pub sky_view: TextureNodeId,
    pub transmittance: TextureNodeId,
    pub bake_params: BufferNodeId,
}

#[derive(Debug, Clone, Copy)]
struct ProceduralStarboxSnapshot {
    source: TextureSource,
    kind: ProceduralStarboxKind,
    logical_version: u64,
}

#[derive(Debug, Clone, Copy)]
struct ProceduralTextureSnapshot {
    source: TextureSource,
    logical_version: u64,
}

/// Lightweight CPU-only state captured by Composer for a procedural skybox.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProceduralSkyboxSnapshot {
    global_state_key: GlobalStateKey,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
    starbox: Option<ProceduralStarboxSnapshot>,
    moon_texture: Option<ProceduralTextureSnapshot>,
}

/// Lightweight scene data for the texture-free gradient variant.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GradientSkyboxSnapshot {
    params: SkyboxParamsUniforms,
    global_state_key: GlobalStateKey,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
}

/// Build-time request shared by the standalone and inline skybox paths.
pub(crate) enum SkyboxDrawRequest<'a> {
    /// Compatibility path for Texture backgrounds. Their asset mipmap
    /// upload still occurs before graph build until RDG asset imports can carry
    /// deferred mip generation.
    Prepared {
        pipeline_id: RenderPipelineId,
        bind_group: &'a wgpu::BindGroup,
    },
    Gradient {
        feature: &'a mut SkyboxFeature,
        snapshot: GradientSkyboxSnapshot,
    },
    /// Fully deferred procedural path. Physical resources are resolved only if
    /// the containing render node survives DPE.
    Procedural {
        feature: &'a mut SkyboxFeature,
        snapshot: ProceduralSkyboxSnapshot,
        inputs: ProceduralSkyboxInputs,
    },
}

pub(crate) enum SkyboxDrawState<'a> {
    Prepared {
        pipeline_id: RenderPipelineId,
        bind_group: &'a wgpu::BindGroup,
    },
    Gradient(GradientSkyboxDrawState<'a>),
    Procedural(ProceduralSkyboxDrawState<'a>),
}

pub(crate) struct GradientSkyboxDrawState<'a> {
    feature: &'a mut SkyboxFeature,
    snapshot: GradientSkyboxSnapshot,
    params_buffer: BufferNodeId,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

pub(crate) struct ProceduralSkyboxDrawState<'a> {
    feature: &'a mut SkyboxFeature,
    snapshot: ProceduralSkyboxSnapshot,
    inputs: ProceduralSkyboxInputs,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

// ─── Layout Helpers ───────────────────────────────────────────────────────────

fn create_uniform_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Skybox Layout (Gradient)"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Blue-noise dithering source for the gradient banding fix.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: blue_noise_view_dimension(),
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// View dimension of the system blue-noise texture: a temporally-layered array
/// when `advanced_noise` is enabled, otherwise a single 64×64 slice.
fn blue_noise_view_dimension() -> wgpu::TextureViewDimension {
    if cfg!(feature = "advanced_noise") {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

fn create_texture_layout(
    device: &wgpu::Device,
    view_dimension: wgpu::TextureViewDimension,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn create_procedural_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Skybox Layout (Procedural)"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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

fn create_procedural_texture_layout(
    device: &wgpu::Device,
    view_dimension: wgpu::TextureViewDimension,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

// ─── RDG Skybox Pass ──────────────────────────────────────────────────────────

/// Skybox / Background rendering feature.
///
/// Owns persistent GPU state (layouts, pipeline cache, bind groups) and
/// produces an ephemeral [`SkyboxPassNode`] each frame via [`Self::add_to_graph`].
pub struct SkyboxFeature {
    // ─── RDG Resource Slots ────────────────────────────────────────
    pub scene_color: TextureNodeId,
    pub scene_depth: TextureNodeId,

    // ─── Bind Group Layouts (Group 1) ──────────────────────────────
    layout_gradient: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_cube: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_2d: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_procedural: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_procedural_eq: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_procedural_cube: Option<Tracked<wgpu::BindGroupLayout>>,

    // procedural_sampler: Option<Tracked<wgpu::Sampler>>,

    // ─── Pipeline Cache ────────────────────────────────────────────
    local_cache: FxHashMap<SkyboxPipelineKey, RenderPipelineId>,

    // ─── Runtime State ─────────────────────────────────────────────
    pub(crate) current_bind_group: Option<wgpu::BindGroup>,
    pub(crate) current_pipeline: Option<RenderPipelineId>,
    current_texture_target: Option<PreparedTextureTargetKey>,
}

impl Default for SkyboxFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SkyboxFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene_color: TextureNodeId::from_index(0),
            scene_depth: TextureNodeId::from_index(0),
            layout_gradient: None,
            layout_cube: None,
            layout_2d: None,
            layout_procedural: None,
            layout_procedural_eq: None,
            layout_procedural_cube: None,
            // procedural_sampler: None,
            local_cache: FxHashMap::default(),
            current_bind_group: None,
            current_pipeline: None,
            current_texture_target: None,
        }
    }

    fn ensure_layout_for_variant(
        &mut self,
        device: &wgpu::Device,
        variant: SkyboxVariant,
        procedural_starbox: ProceduralStarboxKind,
    ) {
        match variant {
            SkyboxVariant::Gradient => {
                if self.layout_gradient.is_none() {
                    self.layout_gradient = Some(Tracked::new(create_uniform_layout(device)));
                }
            }
            SkyboxVariant::Cube => {
                if self.layout_cube.is_none() {
                    self.layout_cube = Some(Tracked::new(create_texture_layout(
                        device,
                        wgpu::TextureViewDimension::Cube,
                        "Skybox Layout (Cube)",
                    )));
                }
            }
            SkyboxVariant::Equirectangular | SkyboxVariant::Planar => {
                if self.layout_2d.is_none() {
                    self.layout_2d = Some(Tracked::new(create_texture_layout(
                        device,
                        wgpu::TextureViewDimension::D2,
                        "Skybox Layout (2D)",
                    )));
                }
            }
            SkyboxVariant::Procedural => match procedural_starbox {
                ProceduralStarboxKind::None => {
                    if self.layout_procedural.is_none() {
                        self.layout_procedural =
                            Some(Tracked::new(create_procedural_layout(device)));
                    }
                }
                ProceduralStarboxKind::Equirectangular => {
                    if self.layout_procedural_eq.is_none() {
                        self.layout_procedural_eq =
                            Some(Tracked::new(create_procedural_texture_layout(
                                device,
                                wgpu::TextureViewDimension::D2,
                                "Skybox Layout (Procedural Eq)",
                            )));
                    }
                }
                ProceduralStarboxKind::Cube => {
                    if self.layout_procedural_cube.is_none() {
                        self.layout_procedural_cube =
                            Some(Tracked::new(create_procedural_texture_layout(
                                device,
                                wgpu::TextureViewDimension::Cube,
                                "Skybox Layout (Procedural Cube)",
                            )));
                    }
                }
            },
        }
    }

    fn layout_for_variant(
        &self,
        variant: SkyboxVariant,
        procedural_starbox: ProceduralStarboxKind,
    ) -> &Tracked<wgpu::BindGroupLayout> {
        match variant {
            SkyboxVariant::Gradient => self.layout_gradient.as_ref().unwrap(),
            SkyboxVariant::Cube => self.layout_cube.as_ref().unwrap(),
            SkyboxVariant::Equirectangular | SkyboxVariant::Planar => {
                self.layout_2d.as_ref().unwrap()
            }
            SkyboxVariant::Procedural => match procedural_starbox {
                ProceduralStarboxKind::None => self.layout_procedural.as_ref().unwrap(),
                ProceduralStarboxKind::Equirectangular => {
                    self.layout_procedural_eq.as_ref().unwrap()
                }
                ProceduralStarboxKind::Cube => self.layout_procedural_cube.as_ref().unwrap(),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn get_or_create_pipeline(
        &mut self,
        device: &wgpu::Device,
        pipeline_cache: &mut PipelineCache,
        shader_manager: &mut ShaderManager,
        global_layout: &wgpu::BindGroupLayout,
        global_layout_id: u64,
        global_binding_wgsl: &str,
        variant: SkyboxVariant,
        procedural_starbox: ProceduralStarboxKind,
        procedural_moon_texture: bool,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> RenderPipelineId {
        let shader_source = ShaderSource::File("entry/utility/skybox");
        let key = SkyboxPipelineKey {
            variant,
            procedural_starbox,
            procedural_moon_texture,
            color_format,
            depth_format,
            msaa_samples,
            global_layout_id,
            shader_source_identity: shader_manager.source_identity_hash(shader_source),
        };
        if let Some(&pipeline_id) = self.local_cache.get(&key) {
            return pipeline_id;
        }

        let mut defines = ShaderDefines::new();
        key.variant.apply_shader_defines(
            &mut defines,
            key.procedural_starbox,
            key.procedural_moon_texture,
        );

        let mut options = ShaderCompilationOptions {
            defines,
            ..Default::default()
        };
        options.add_define(
            "struct_definitions",
            SkyboxParamsUniforms::wgsl_struct_def("SkyboxParams").as_str(),
        );
        options.inject_code("binding_code", global_binding_wgsl);
        options.inject_code(
            "scene_lighting_structs",
            myth_resources::uniforms::scene_lighting_structs_wgsl(),
        );

        let (shader_module, shader_hash) =
            shader_manager.get_or_compile(device, shader_source, &options);

        let layout = self.layout_for_variant(key.variant, key.procedural_starbox);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Pipeline Layout"),
            bind_group_layouts: &[Some(global_layout), Some(layout)],
            immediate_size: 0,
        });

        let fullscreen_key = FullscreenPipelineKey {
            shader_hash,
            compilation_hash: 0,
            color_targets: smallvec::smallvec![ColorTargetKey::from(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            depth_stencil: Some(DepthStencilKey::from(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })),
            multisample: MultisampleKey::from(wgpu::MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            }),
            topology: wgpu::PrimitiveTopology::TriangleList,
        };

        let pipeline_id = pipeline_cache.get_or_create_fullscreen(
            device,
            shader_module,
            &pipeline_layout,
            &fullscreen_key,
            "Skybox Pipeline",
        );

        self.local_cache.insert(key, pipeline_id);
        pipeline_id
    }

    fn resolve_texture_view_with_dimension<'a>(
        resource_manager: &'a crate::core::ResourceManager,
        source: &TextureSource,
    ) -> Option<ResolvedTextureView<'a>> {
        match source {
            TextureSource::Asset(handle) => {
                let binding = resource_manager.get_texture_binding(*handle)?;
                let img = resource_manager.get_image(binding.image_handle)?;
                Some(ResolvedTextureView {
                    view: &img.default_view,
                    view_dimension: img.default_view_dimension,
                    resource_key: binding.view_id,
                })
            }
            TextureSource::Attachment(id, dim) => {
                let view = resource_manager.get_internal_texture(*id)?;
                let view_id = resource_manager.get_internal_texture_view_id(*id)?;
                Some(ResolvedTextureView {
                    view,
                    view_dimension: *dim,
                    resource_key: view_id,
                })
            }
        }
    }

    fn resolve_texture_view<'a>(
        resource_manager: &'a crate::core::ResourceManager,
        source: &TextureSource,
        mapping: BackgroundMapping,
    ) -> Option<ResolvedTextureView<'a>> {
        let resolved = Self::resolve_texture_view_with_dimension(resource_manager, source)?;
        match mapping {
            BackgroundMapping::Cube
                if resolved.view_dimension == wgpu::TextureViewDimension::Cube =>
            {
                Some(resolved)
            }
            BackgroundMapping::Equirectangular | BackgroundMapping::Planar
                if resolved.view_dimension == wgpu::TextureViewDimension::D2 =>
            {
                Some(resolved)
            }
            _ => None,
        }
    }

    /// Compatibility preparation for Texture backgrounds.
    ///
    /// Texture backgrounds remain here temporarily because their asset path
    /// owns mip-chain allocation and queues same-frame RDG mip generation. The
    /// procedural path is deliberately rejected and is prepared only by a live
    /// graph node.
    pub(crate) fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        background_mode: &BackgroundMode,
        bg_uniforms: &CpuBuffer<SkyboxParamsUniforms>,
        global_state_key: (u32, u32),
        color_format: wgpu::TextureFormat,
    ) {
        let Some(variant) = SkyboxVariant::from_background(background_mode) else {
            self.deactivate();
            return;
        };
        if !variant.needs_texture() || variant == SkyboxVariant::Procedural {
            self.deactivate();
            return;
        }

        self.ensure_layout_for_variant(ctx.device, variant, ProceduralStarboxKind::None);
        let target_key = PreparedTextureTargetKey {
            variant,
            global_state_key,
            color_format,
            depth_format: ctx.wgpu_ctx.depth_format,
            msaa_samples: ctx.wgpu_ctx.msaa_samples,
        };

        if let BackgroundMode::Texture {
            source: TextureSource::Asset(handle),
            ..
        } = background_mode
        {
            let state = ctx.resource_manager.prepare_texture(ctx.assets, *handle);
            if matches!(state, ResourceState::Pending)
                && self.current_texture_target == Some(target_key)
                && self.current_bind_group.is_some()
                && self.current_pipeline.is_some()
            {
                // Preserve only the same scene's last prepared texture. The old
                // feature-global fallback could leak a different scene's skybox.
                return;
            }
            if matches!(state, ResourceState::Pending) {
                self.deactivate();
                return;
            }
        }

        let sampler_id = ctx
            .resource_manager
            .get_or_create_sampler(SKYBOX_SAMPLER_KEY);

        let texture_view = match background_mode {
            BackgroundMode::Texture {
                source, mapping, ..
            } => Self::resolve_texture_view(ctx.resource_manager, source, *mapping)
                .map(|resolved| (resolved.view.clone(), resolved.resource_key)),
            _ => None,
        };
        let layout = self.layout_for_variant(variant, ProceduralStarboxKind::None);
        let bg_uniforms_resource_id = ctx.resource_manager.ensure_buffer_id(bg_uniforms);

        let Some((tex_view, tex_view_resource_key)) = texture_view else {
            self.deactivate();
            return;
        };
        let params_buffer = {
            let params_gpu = bg_uniforms
                .gpu_handle()
                .and_then(|h| ctx.resource_manager.get_gpu_buffer(h))
                .expect("Skybox params GPU buffer must exist");
            params_gpu.buffer.clone()
        };
        let bind_group = ctx
            .build_bind_group(layout, Some("Skybox BG (Texture)"))
            .bind_raw_buffer(
                0,
                RawBufferBinding::new(&params_buffer, bg_uniforms_resource_id, None),
            )
            .bind_texture_view_with_id(1, &tex_view, tex_view_resource_key)
            .bind_sampler_by_id(2, sampler_id)
            .build();

        self.current_bind_group = Some(bind_group.clone());
        let gpu_world = ctx
            .resource_manager
            .get_global_state(global_state_key.0, global_state_key.1)
            .expect("Skybox global state must exist");
        self.current_pipeline = Some(self.get_or_create_pipeline(
            ctx.device,
            ctx.pipeline_cache,
            ctx.shader_manager,
            &gpu_world.layout,
            gpu_world.layout_id,
            &gpu_world.binding_wgsl,
            variant,
            ProceduralStarboxKind::None,
            false,
            color_format,
            ctx.wgpu_ctx.depth_format,
            ctx.wgpu_ctx.msaa_samples,
        ));
        self.current_texture_target = Some(target_key);
    }

    /// Capture only POD state for a gradient draw.
    #[must_use]
    #[allow(clippy::unused_self)] // Keep the uniform Feature::extract API.
    pub(crate) fn extract_gradient(
        &self,
        params: &CpuBuffer<SkyboxParamsUniforms>,
        global_state_key: GlobalStateKey,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> GradientSkyboxSnapshot {
        GradientSkyboxSnapshot {
            params: *params.read(),
            global_state_key,
            color_format,
            depth_format,
            msaa_samples,
        }
    }

    /// Capture only logical/POD state for a procedural draw.
    #[must_use]
    #[allow(clippy::unused_self)] // Keep the uniform Feature::extract API.
    pub(crate) fn extract_procedural(
        &self,
        params: &ProceduralSkyParams,
        environment: ProceduralEnvironmentSnapshot,
        global_state_key: GlobalStateKey,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> ProceduralSkyboxSnapshot {
        let starbox = params
            .starbox_texture
            .zip(environment.starbox_texture)
            .and_then(|(source, logical)| {
                if !logical.ready {
                    return None;
                }
                Some(ProceduralStarboxSnapshot {
                    source,
                    kind: ProceduralStarboxKind::from_view_dimension(logical.view_dimension)?,
                    logical_version: logical.version,
                })
            });
        let moon_texture = params
            .moon_albedo_texture
            .zip(environment.moon_texture)
            .and_then(|(source, logical)| {
                (logical.ready && logical.view_dimension == wgpu::TextureViewDimension::D2)
                    .then_some(ProceduralTextureSnapshot {
                        source,
                        logical_version: logical.version,
                    })
            });

        ProceduralSkyboxSnapshot {
            global_state_key,
            color_format,
            depth_format,
            msaa_samples,
            starbox,
            moon_texture,
        }
    }

    #[must_use]
    pub(crate) fn prepared_draw_request(&self) -> Option<SkyboxDrawRequest<'_>> {
        Some(SkyboxDrawRequest::Prepared {
            pipeline_id: self.current_pipeline?,
            bind_group: self.current_bind_group.as_ref()?,
        })
    }

    #[must_use]
    pub(crate) fn procedural_draw_request(
        &mut self,
        snapshot: ProceduralSkyboxSnapshot,
        inputs: ProceduralSkyboxInputs,
    ) -> SkyboxDrawRequest<'_> {
        SkyboxDrawRequest::Procedural {
            feature: self,
            snapshot,
            inputs,
        }
    }

    #[must_use]
    pub(crate) fn gradient_draw_request(
        &mut self,
        snapshot: GradientSkyboxSnapshot,
    ) -> SkyboxDrawRequest<'_> {
        SkyboxDrawRequest::Gradient {
            feature: self,
            snapshot,
        }
    }

    fn prepare_gradient_draw<'frame>(
        &mut self,
        ctx: &mut PrepareContext<'frame>,
        snapshot: GradientSkyboxSnapshot,
        params_buffer: BufferNodeId,
    ) -> (RenderPipelineId, &'frame wgpu::BindGroup) {
        self.ensure_layout_for_variant(
            ctx.device,
            SkyboxVariant::Gradient,
            ProceduralStarboxKind::None,
        );
        ctx.queue.write_buffer(
            ctx.views.get_buffer(params_buffer),
            0,
            bytemuck::bytes_of(&snapshot.params),
        );

        let global_state = ctx
            .global_states
            .get(snapshot.global_state_key)
            .expect("gradient Skybox global state must exist before RDG prepare");
        let pipeline_id = self.get_or_create_pipeline(
            ctx.device,
            ctx.pipeline_cache,
            ctx.shader_manager,
            &global_state.layout,
            global_state.layout_id,
            &global_state.binding_wgsl,
            SkyboxVariant::Gradient,
            ProceduralStarboxKind::None,
            false,
            snapshot.color_format,
            snapshot.depth_format,
            snapshot.msaa_samples,
        );

        let blue_noise = ctx.system_textures.blue_noise.clone();
        let blue_noise_sampler = ctx.system_textures.blue_noise_sampler.clone();
        let layout = self.layout_for_variant(SkyboxVariant::Gradient, ProceduralStarboxKind::None);
        let bind_group = ctx
            .build_bind_group(layout, Some("Skybox BG (Gradient)"))
            .bind_buffer(0, params_buffer)
            .bind_tracked_texture_view(1, &blue_noise)
            .bind_tracked_sampler(2, &blue_noise_sampler)
            .build();

        (pipeline_id, bind_group)
    }

    fn prepare_procedural_draw<'frame>(
        &mut self,
        ctx: &mut PrepareContext<'frame>,
        snapshot: ProceduralSkyboxSnapshot,
        inputs: ProceduralSkyboxInputs,
    ) -> (RenderPipelineId, &'frame wgpu::BindGroup) {
        let starbox_kind = snapshot
            .starbox
            .map_or(ProceduralStarboxKind::None, |starbox| starbox.kind);
        self.ensure_layout_for_variant(ctx.device, SkyboxVariant::Procedural, starbox_kind);

        let starbox = snapshot.starbox.map(|starbox| {
            let prepared = ctx
                .uploads
                .resolve_texture_source(ctx.assets, starbox.source, starbox.logical_version)
                .expect("procedural starbox must keep its snapshot version through prepare");
            assert_eq!(
                ProceduralStarboxKind::from_view_dimension(prepared.view_dimension),
                Some(starbox.kind),
                "procedural starbox view dimension changed after extraction",
            );
            assert_eq!(
                prepared.logical_version, starbox.logical_version,
                "procedural starbox version changed after extraction",
            );
            prepared
        });
        let moon_texture = snapshot.moon_texture.map(|source| {
            let prepared = ctx
                .uploads
                .resolve_texture_source(ctx.assets, source.source, source.logical_version)
                .expect("procedural moon texture must keep its snapshot version through prepare");
            assert_eq!(
                prepared.view_dimension,
                wgpu::TextureViewDimension::D2,
                "procedural moon texture view dimension changed after extraction",
            );
            assert_eq!(
                prepared.logical_version, source.logical_version,
                "procedural moon texture version changed after extraction",
            );
            prepared
        });

        let global_state = ctx
            .global_states
            .get(snapshot.global_state_key)
            .expect("procedural Skybox global state must exist before RDG prepare");
        let pipeline_id = self.get_or_create_pipeline(
            ctx.device,
            ctx.pipeline_cache,
            ctx.shader_manager,
            &global_state.layout,
            global_state.layout_id,
            &global_state.binding_wgsl,
            SkyboxVariant::Procedural,
            starbox_kind,
            moon_texture.is_some(),
            snapshot.color_format,
            snapshot.depth_format,
            snapshot.msaa_samples,
        );

        let white_moon = ctx.system_textures.white_2d.clone();
        let layout = self.layout_for_variant(SkyboxVariant::Procedural, starbox_kind);
        let label = if starbox.is_some() {
            Some("Skybox BG (Procedural+Starbox)")
        } else {
            Some("Skybox BG (Procedural)")
        };
        let builder = ctx
            .build_bind_group(layout, label)
            .bind_buffer(0, inputs.bake_params)
            .bind_texture(1, inputs.sky_view)
            .bind_common_sampler(2, CommonSampler::LinearClamp)
            .bind_texture(3, inputs.transmittance);
        let builder = if let Some(moon) = moon_texture.as_ref() {
            builder.bind_texture_view_with_id(4, &moon.view, moon.view_id)
        } else {
            builder.bind_tracked_texture_view(4, &white_moon)
        };
        let bind_group = if let Some(starbox) = starbox.as_ref() {
            builder
                .bind_texture_view_with_id(5, &starbox.view, starbox.view_id)
                .build()
        } else {
            builder.build()
        };

        (pipeline_id, bind_group)
    }

    /// Release the compatibility path's frame-selected physical draw state.
    pub(crate) fn deactivate(&mut self) {
        self.current_bind_group = None;
        self.current_pipeline = None;
        self.current_texture_target = None;
    }
}

impl<'a> SkyboxDrawRequest<'a> {
    /// Declare the real resources sampled by this draw and turn the build-time
    /// request into node-local prepare state.
    pub(crate) fn build(self, builder: &mut PassBuilder<'_, 'a>) -> SkyboxDrawState<'a> {
        match self {
            Self::Prepared {
                pipeline_id,
                bind_group,
            } => SkyboxDrawState::Prepared {
                pipeline_id,
                bind_group,
            },
            Self::Gradient { feature, snapshot } => {
                let params_buffer = builder.create_buffer(
                    "Skybox_Params",
                    BufferDesc::new(
                        std::mem::size_of::<SkyboxParamsUniforms>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    ),
                );
                SkyboxDrawState::Gradient(GradientSkyboxDrawState {
                    feature,
                    snapshot,
                    params_buffer,
                    pipeline_id: None,
                    bind_group: None,
                })
            }
            Self::Procedural {
                feature,
                snapshot,
                inputs,
            } => {
                builder.read_texture(inputs.sky_view);
                builder.read_texture(inputs.transmittance);
                builder.read_buffer(inputs.bake_params);
                SkyboxDrawState::Procedural(ProceduralSkyboxDrawState {
                    feature,
                    snapshot,
                    inputs,
                    pipeline_id: None,
                    bind_group: None,
                })
            }
        }
    }

    /// Add the standalone HighFidelity skybox pass.
    pub(crate) fn add_to_graph(
        self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        scene_color: TextureNodeId,
        scene_depth: TextureNodeId,
    ) -> TextureNodeId {
        ctx.graph.add_pass("Skybox_Pass", |builder| {
            let out_color = builder.mutate_texture(scene_color, "Scene_Color_Skybox");
            builder.read_texture(scene_depth);
            let draw = self.build(builder);
            let node = SkyboxPassNode {
                out_color,
                scene_depth,
                draw,
            };
            (node, out_color)
        })
    }
}

impl<'a> SkyboxDrawState<'a> {
    pub(crate) fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        match self {
            Self::Prepared { .. } => {}
            Self::Gradient(state) => {
                let (pipeline_id, bind_group) =
                    state
                        .feature
                        .prepare_gradient_draw(ctx, state.snapshot, state.params_buffer);
                state.pipeline_id = Some(pipeline_id);
                state.bind_group = Some(bind_group);
            }
            Self::Procedural(state) => {
                let (pipeline_id, bind_group) =
                    state
                        .feature
                        .prepare_procedural_draw(ctx, state.snapshot, state.inputs);
                state.pipeline_id = Some(pipeline_id);
                state.bind_group = Some(bind_group);
            }
        }
    }

    pub(crate) fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'a>,
        global_bind_group: &'a wgpu::BindGroup,
        pipeline_cache: &'a PipelineCache,
    ) {
        let (pipeline_id, bind_group) = match self {
            Self::Prepared {
                pipeline_id,
                bind_group,
            } => (*pipeline_id, *bind_group),
            Self::Gradient(state) => (
                state
                    .pipeline_id
                    .expect("gradient Skybox pipeline must exist after prepare"),
                state
                    .bind_group
                    .expect("gradient Skybox bind group must exist after prepare"),
            ),
            Self::Procedural(state) => (
                state
                    .pipeline_id
                    .expect("procedural Skybox pipeline must exist after prepare"),
                state
                    .bind_group
                    .expect("procedural Skybox bind group must exist after prepare"),
            ),
        };
        pass.set_pipeline(pipeline_cache.get_render_pipeline(pipeline_id));
        pass.set_bind_group(0, global_bind_group, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

// ─── Skybox Pass Node ─────────────────────────────────────────────────────────

/// Ephemeral per-frame skybox render pass node.
pub struct SkyboxPassNode<'a> {
    out_color: TextureNodeId,
    scene_depth: TextureNodeId,
    draw: SkyboxDrawState<'a>,
}

impl<'a> PassNode<'a> for SkyboxPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.draw.prepare(ctx);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let gpu_global_bind_group = ctx.baked_lists.global_bind_group;

        let color_att = ctx.get_color_attachment(self.out_color, RenderTargetOps::Load, None);
        let depth_att = ctx.get_depth_stencil_attachment(self.scene_depth, 0.0);

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Skybox Pass"),
            color_attachments: &[color_att],
            depth_stencil_attachment: depth_att,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let mut pass = encoder.begin_render_pass(&pass_desc);
        self.draw
            .draw(&mut pass, gpu_global_bind_group, ctx.pipeline_cache);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::graph::FrameConfig;
    use crate::graph::core::{
        BufferDesc, ClusteredScreenBindings, FrameArena, GraphStorage, RenderGraph, TextureDesc,
    };
    use crate::graph::passes::SimpleForwardFeature;

    struct TestNode;

    impl PassNode<'_> for TestNode {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    fn frame_config() -> FrameConfig {
        FrameConfig {
            width: 320,
            height: 180,
            depth_format: wgpu::TextureFormat::Depth32Float,
            msaa_samples: 1,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            hdr_format: crate::HDR_TEXTURE_FORMAT,
        }
    }

    fn texture_desc(format: wgpu::TextureFormat) -> TextureDesc {
        TextureDesc::new_2d(
            16,
            16,
            format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }

    fn procedural_graph_live_passes(root_skybox: bool) -> Vec<&'static str> {
        let mut feature = SkyboxFeature::new();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let scene_color = graph.add_pass("Scene_Color_Producer", |builder| {
            let output =
                builder.create_texture("Scene_Color", texture_desc(crate::HDR_TEXTURE_FORMAT));
            (TestNode, output)
        });
        let scene_depth = graph.add_pass("Scene_Depth_Producer", |builder| {
            let output = builder.create_texture(
                "Scene_Depth",
                texture_desc(wgpu::TextureFormat::Depth32Float),
            );
            (TestNode, output)
        });
        let sky_view = graph.add_pass("SkyView_Producer", |builder| {
            let output =
                builder.create_texture("Sky_View", texture_desc(wgpu::TextureFormat::Rgba16Float));
            (TestNode, output)
        });
        let transmittance = graph.add_pass("Transmittance_Producer", |builder| {
            let output = builder.create_texture(
                "Transmittance",
                texture_desc(wgpu::TextureFormat::Rgba16Float),
            );
            (TestNode, output)
        });
        let bake_params = graph.add_pass("Bake_Params_Producer", |builder| {
            let output = builder.create_buffer(
                "Bake_Params",
                BufferDesc::new(
                    64,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
            );
            (TestNode, output)
        });
        let config = frame_config();
        let output = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            feature
                .procedural_draw_request(
                    ProceduralSkyboxSnapshot {
                        global_state_key: GlobalStateKey::new(1, 7),
                        color_format: crate::HDR_TEXTURE_FORMAT,
                        depth_format: wgpu::TextureFormat::Depth32Float,
                        msaa_samples: 1,
                        starbox: None,
                        moon_texture: None,
                    },
                    ProceduralSkyboxInputs {
                        sky_view,
                        transmittance,
                        bake_params,
                    },
                )
                .add_to_graph(&mut ctx, scene_color, scene_depth)
        };
        if root_skybox {
            graph.add_pass("Skybox_Test_Root", |builder| {
                builder.read_texture(output);
                builder.mark_side_effect();
                (TestNode, ())
            });
        }

        graph.compile_topology();
        graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect()
    }

    #[test]
    fn procedural_skybox_declares_all_physical_inputs() {
        let live = procedural_graph_live_passes(true);
        assert!(live.contains(&"SkyView_Producer"));
        assert!(live.contains(&"Transmittance_Producer"));
        assert!(live.contains(&"Bake_Params_Producer"));
        assert!(live.contains(&"Skybox_Pass"));
    }

    #[test]
    fn basic_forward_inline_skybox_keeps_all_inputs_live() {
        let mut skybox_feature = SkyboxFeature::new();
        let simple_forward = SimpleForwardFeature::new();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let sky_view = graph.add_pass("Inline_SkyView_Producer", |builder| {
            let output = builder.create_texture(
                "Inline_Sky_View",
                texture_desc(wgpu::TextureFormat::Rgba16Float),
            );
            (TestNode, output)
        });
        let transmittance = graph.add_pass("Inline_Transmittance_Producer", |builder| {
            let output = builder.create_texture(
                "Inline_Transmittance",
                texture_desc(wgpu::TextureFormat::Rgba16Float),
            );
            (TestNode, output)
        });
        let bake_params = graph.add_pass("Inline_Bake_Params_Producer", |builder| {
            let output = builder.create_buffer(
                "Inline_Bake_Params",
                BufferDesc::new(
                    64,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
            );
            (TestNode, output)
        });
        let config = frame_config();
        let output = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            let skybox = skybox_feature.procedural_draw_request(
                ProceduralSkyboxSnapshot {
                    global_state_key: GlobalStateKey::new(2, 9),
                    color_format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    depth_format: wgpu::TextureFormat::Depth32Float,
                    msaa_samples: 1,
                    starbox: None,
                    moon_texture: None,
                },
                ProceduralSkyboxInputs {
                    sky_view,
                    transmittance,
                    bake_params,
                },
            );
            simple_forward.add_to_graph(
                &mut ctx,
                wgpu::Color::BLACK,
                Some(skybox),
                None,
                None,
                None,
                ClusteredScreenBindings::default(),
            )
        };
        graph.add_pass("Inline_Skybox_Test_Root", |builder| {
            builder.read_texture(output);
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
        assert!(live.contains(&"Inline_SkyView_Producer"));
        assert!(live.contains(&"Inline_Transmittance_Producer"));
        assert!(live.contains(&"Inline_Bake_Params_Producer"));
        assert!(live.contains(&"SimpleForward_Pass"));
        assert!(!live.contains(&"Skybox_Pass"));
    }

    #[test]
    fn dead_procedural_skybox_culls_its_entire_input_chain() {
        assert!(procedural_graph_live_passes(false).is_empty());
    }
}
