//! Procedural sky atmosphere bake passes.
//!
//! The system is decomposed into four demand-driven RDG nodes:
//!
//! - Transmittance LUT (independent persistent cache)
//! - Multi-scatter LUT (independent persistent cache)
//! - Sky-view LUT (transient, varies with the current sun)
//! - Sky-view -> persistent scene base cubemap bake
//!
//! Graph construction carries only POD snapshots and logical resource IDs.
//! Static layouts/pipelines, source texture upload, bind groups, and uniform
//! writes happen only in live-node `prepare` after DPE.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher};

use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::{CommonSampler, ProceduralEnvironmentSnapshot, Tracked, binding_layout};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, CacheId, CacheKey, CachePlan, ExecuteContext, HistoryRegistry,
    PassNode, PersistentTextureSpec, PrepareContext, TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderManager, ShaderSource,
};
use myth_resources::shader_defines::ShaderDefines;
use myth_resources::texture::TextureSource;
use myth_scene::background::ProceduralSkyParams;

const TRANSMITTANCE_WIDTH: u32 = 256;
const TRANSMITTANCE_HEIGHT: u32 = 64;
const MULTI_SCATTER_SIZE: u32 = 32;
const SKY_VIEW_WIDTH: u32 = 192;
const SKY_VIEW_HEIGHT: u32 = 108;
const SCENE_CACHE_TTL: u64 = 120;
const TRANSMITTANCE_CACHE_SCHEMA: u64 = 1;
const MULTI_SCATTER_CACHE_SCHEMA: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AtmosphereStarboxKind {
    None,
    Equirectangular,
    Cube,
}

impl AtmosphereStarboxKind {
    fn from_view_dimension(view_dimension: wgpu::TextureViewDimension) -> Option<Self> {
        match view_dimension {
            wgpu::TextureViewDimension::D2 => Some(Self::Equirectangular),
            wgpu::TextureViewDimension::Cube => Some(Self::Cube),
            _ => None,
        }
    }

    fn apply_shader_defines(self, defines: &mut ShaderDefines) {
        match self {
            Self::None => {}
            Self::Equirectangular => defines.set("CELESTIAL_STARBOX_EQUIRECT", "1"),
            Self::Cube => defines.set("CELESTIAL_STARBOX_CUBE", "1"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AtmosphereSkyToCubePipelineKey {
    starbox_kind: AtmosphereStarboxKind,
    moon_texture: bool,
}

impl AtmosphereSkyToCubePipelineKey {
    fn apply_shader_defines(self, defines: &mut ShaderDefines) {
        self.starbox_kind.apply_shader_defines(defines);
        if self.moon_texture {
            defines.set("USE_MOON_TEXTURE", "1");
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct AtmosphereComputePipelineState {
    id: ComputePipelineId,
    shader_identity: u64,
}

#[derive(Debug, Clone, Copy)]
struct AtmosphereStarboxSnapshot {
    source: TextureSource,
    kind: AtmosphereStarboxKind,
    logical_version: u64,
}

#[derive(Debug, Clone, Copy)]
struct AtmosphereTextureSnapshot {
    source: TextureSource,
    logical_version: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuAtmosphereParams {
    rayleigh_scattering: [f32; 3],
    rayleigh_scale_height: f32,
    mie_scattering: f32,
    mie_absorption: f32,
    mie_scale_height: f32,
    mie_anisotropy: f32,
    ozone_absorption: [f32; 3],
    _pad0: f32,
    planet_radius: f32,
    atmosphere_radius: f32,
    sun_intensity: f32,
    sun_cos_angle: f32,
    sun_direction: [f32; 3],
    _pad1: f32,
}

impl GpuAtmosphereParams {
    fn from_scene(params: &ProceduralSkyParams) -> Self {
        Self {
            rayleigh_scattering: params.rayleigh_scattering.into(),
            rayleigh_scale_height: params.rayleigh_scale_height,
            mie_scattering: params.mie_scattering,
            mie_absorption: params.mie_absorption,
            mie_scale_height: params.mie_scale_height,
            mie_anisotropy: params.mie_anisotropy,
            ozone_absorption: params.ozone_absorption.into(),
            _pad0: 0.0,
            planet_radius: params.planet_radius,
            atmosphere_radius: params.atmosphere_radius,
            sun_intensity: params.sun_intensity,
            sun_cos_angle: params.sun_direction.y,
            sun_direction: params.sun_direction.into(),
            _pad1: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBakeParams {
    sun_direction: [f32; 3],
    sun_intensity: f32,
    moon_direction: [f32; 3],
    moon_intensity: f32,
    star_axis: [f32; 3],
    sun_disk_size: f32,
    moon_disk_size: f32,
    planet_radius: f32,
    atmosphere_radius: f32,
    star_intensity: f32,
    star_rotation: f32,
    _pad3: [f32; 3],
}

impl GpuBakeParams {
    fn from_scene(params: &ProceduralSkyParams) -> Self {
        Self {
            sun_direction: params.sun_direction.into(),
            sun_intensity: params.sun_intensity,
            moon_direction: params.moon_direction.into(),
            moon_intensity: params.moon_intensity,
            star_axis: params.star_axis.into(),
            sun_disk_size: params.sun_disk_size,
            moon_disk_size: params.moon_disk_size,
            planet_radius: params.planet_radius,
            atmosphere_radius: params.atmosphere_radius,
            star_intensity: params.star_intensity,
            star_rotation: params.star_rotation,
            _pad3: [0.0; 3],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AtmosphereSnapshot {
    scene_id: u32,
    gpu_params: GpuAtmosphereParams,
    bake_params: GpuBakeParams,
    physics_hash: u64,
    transmittance_shader_identity: u64,
    multi_scatter_shader_identity: u64,
    sky_view_shader_identity: u64,
    sky_to_cube_shader_identity: u64,
    sky_to_cube_pipeline_key: AtmosphereSkyToCubePipelineKey,
    starbox: Option<AtmosphereStarboxSnapshot>,
    moon_texture: Option<AtmosphereTextureSnapshot>,
}

impl AtmosphereSnapshot {
    #[must_use]
    pub(crate) fn procedural_base_shader_identity(self) -> u64 {
        let mut hasher = FxHasher::default();
        self.transmittance_shader_identity.hash(&mut hasher);
        self.multi_scatter_shader_identity.hash(&mut hasher);
        self.sky_view_shader_identity.hash(&mut hasher);
        self.sky_to_cube_pipeline_key.hash(&mut hasher);
        self.sky_to_cube_shader_identity.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AtmosphereCacheSnapshot {
    transmittance_plan: CachePlan<1>,
    multi_scatter_plan: CachePlan<1>,
}

struct AtmosphereCacheDomain {
    transmittance_cache: CacheId,
    multi_scatter_cache: CacheId,
    last_used_frame: u64,
}

pub(crate) struct AtmosphereGraphOutput {
    pub sky_view: TextureNodeId,
    pub transmittance: TextureNodeId,
    pub bake_params: BufferNodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtmosphereComputeKind {
    Transmittance,
    MultiScatter,
    SkyView,
}

impl AtmosphereComputeKind {
    fn shader_source(self) -> ShaderSource<'static> {
        match self {
            Self::Transmittance => ShaderSource::File("entry/utility/atmosphere/transmittance_lut"),
            Self::MultiScatter => ShaderSource::File("entry/utility/atmosphere/multi_scatter_lut"),
            Self::SkyView => ShaderSource::File("entry/utility/atmosphere/sky_view_lut"),
        }
    }

    fn pipeline_layout_label(self) -> &'static str {
        match self {
            Self::Transmittance => "Atmo Transmittance PL",
            Self::MultiScatter => "Atmo Multi-Scatter PL",
            Self::SkyView => "Atmo Sky-View PL",
        }
    }

    fn pipeline_label(self) -> &'static str {
        match self {
            Self::Transmittance => "Atmo Transmittance Pipeline",
            Self::MultiScatter => "Atmo Multi-Scatter Pipeline",
            Self::SkyView => "Atmo Sky-View Pipeline",
        }
    }
}

struct AtmosphereComputeFeature {
    kind: AtmosphereComputeKind,
    pipeline: Option<AtmosphereComputePipelineState>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl AtmosphereComputeFeature {
    const fn new(kind: AtmosphereComputeKind) -> Self {
        Self {
            kind,
            pipeline: None,
            layout: None,
        }
    }

    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>, expected_shader_identity: u64) {
        if self.layout.is_none() {
            self.layout = Some(create_compute_layout(ctx.device, self.kind));
        }
        if self
            .pipeline
            .is_some_and(|state| state.shader_identity == expected_shader_identity)
        {
            return;
        }

        let source = self.kind.shader_source();
        let options = ShaderCompilationOptions::default();
        assert_eq!(
            ctx.shader_manager.pipeline_shader_hash(source, &options),
            expected_shader_identity,
            "atmosphere cache key and live compute shader identity diverged",
        );
        let (module, shader_hash) = ctx
            .shader_manager
            .get_or_compile(ctx.device, source, &options);
        let layout = self.layout.as_deref().expect("atmosphere layout missing");
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(self.kind.pipeline_layout_label()),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let id = ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            module,
            &pipeline_layout,
            &ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            self.kind.pipeline_label(),
        );
        self.pipeline = Some(AtmosphereComputePipelineState {
            id,
            shader_identity: expected_shader_identity,
        });
    }

    fn pipeline_id(&self) -> ComputePipelineId {
        self.pipeline
            .expect("atmosphere pipeline must exist after prepare")
            .id
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("atmosphere layout must exist after prepare")
    }
}

struct AtmosphereSkyToCubeFeature {
    pipelines: FxHashMap<AtmosphereSkyToCubePipelineKey, AtmosphereComputePipelineState>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
    equirect_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    cube_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl AtmosphereSkyToCubeFeature {
    fn new() -> Self {
        Self {
            pipelines: FxHashMap::default(),
            layout: None,
            equirect_layout: None,
            cube_layout: None,
        }
    }

    fn ensure_layout(&mut self, device: &wgpu::Device, kind: AtmosphereStarboxKind) {
        match kind {
            AtmosphereStarboxKind::None => {
                if self.layout.is_none() {
                    self.layout = Some(Tracked::new(create_sky_to_cube_layout(
                        device,
                        None,
                        "Atmo Sky-to-Cube BGL",
                    )));
                }
            }
            AtmosphereStarboxKind::Equirectangular => {
                if self.equirect_layout.is_none() {
                    self.equirect_layout = Some(Tracked::new(create_sky_to_cube_layout(
                        device,
                        Some(wgpu::TextureViewDimension::D2),
                        "Atmo Sky-to-Cube Eq BGL",
                    )));
                }
            }
            AtmosphereStarboxKind::Cube => {
                if self.cube_layout.is_none() {
                    self.cube_layout = Some(Tracked::new(create_sky_to_cube_layout(
                        device,
                        Some(wgpu::TextureViewDimension::Cube),
                        "Atmo Sky-to-Cube Cube BGL",
                    )));
                }
            }
        }
    }

    fn layout(&self, kind: AtmosphereStarboxKind) -> &Tracked<wgpu::BindGroupLayout> {
        match kind {
            AtmosphereStarboxKind::None => self.layout.as_ref(),
            AtmosphereStarboxKind::Equirectangular => self.equirect_layout.as_ref(),
            AtmosphereStarboxKind::Cube => self.cube_layout.as_ref(),
        }
        .expect("selected sky-to-cube layout must exist after prepare")
    }

    fn ensure_resources(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        key: AtmosphereSkyToCubePipelineKey,
        expected_shader_identity: u64,
    ) {
        self.ensure_layout(ctx.device, key.starbox_kind);
        if self
            .pipelines
            .get(&key)
            .is_some_and(|state| state.shader_identity == expected_shader_identity)
        {
            return;
        }

        let source = ShaderSource::File("entry/utility/atmosphere/sky_to_cube");
        let mut options = ShaderCompilationOptions::default();
        key.apply_shader_defines(&mut options.defines);
        assert_eq!(
            ctx.shader_manager.pipeline_shader_hash(source, &options),
            expected_shader_identity,
            "procedural base key and live sky-to-cube shader identity diverged",
        );
        let (module, shader_hash) = ctx
            .shader_manager
            .get_or_compile(ctx.device, source, &options);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atmo Sky-to-Cube PL"),
                bind_group_layouts: &[Some(self.layout(key.starbox_kind))],
                immediate_size: 0,
            });
        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let id = ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            module,
            &pipeline_layout,
            &ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            "Atmo Sky-to-Cube Pipeline",
        );
        self.pipelines.insert(
            key,
            AtmosphereComputePipelineState {
                id,
                shader_identity: expected_shader_identity,
            },
        );
    }

    fn pipeline_id(&self, key: AtmosphereSkyToCubePipelineKey) -> ComputePipelineId {
        self.pipelines
            .get(&key)
            .expect("sky-to-cube pipeline must exist after prepare")
            .id
    }
}

pub struct AtmosphereFeature {
    domains: FxHashMap<u32, AtmosphereCacheDomain>,
    transmittance: AtmosphereComputeFeature,
    multi_scatter: AtmosphereComputeFeature,
    sky_view: AtmosphereComputeFeature,
    sky_to_cube: AtmosphereSkyToCubeFeature,
}

impl Default for AtmosphereFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl AtmosphereFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            domains: FxHashMap::default(),
            transmittance: AtmosphereComputeFeature::new(AtmosphereComputeKind::Transmittance),
            multi_scatter: AtmosphereComputeFeature::new(AtmosphereComputeKind::MultiScatter),
            sky_view: AtmosphereComputeFeature::new(AtmosphereComputeKind::SkyView),
            sky_to_cube: AtmosphereSkyToCubeFeature::new(),
        }
    }

    /// Capture the current atmosphere without touching WGPU state.
    #[must_use]
    pub(crate) fn extract(
        &self,
        shader_manager: &ShaderManager,
        scene_id: u32,
        params: &ProceduralSkyParams,
        environment: ProceduralEnvironmentSnapshot,
    ) -> AtmosphereSnapshot {
        let starbox = params
            .starbox_texture
            .zip(environment.starbox_texture)
            .and_then(|(source, logical)| {
                if !logical.ready {
                    return None;
                }
                Some(AtmosphereStarboxSnapshot {
                    source,
                    kind: AtmosphereStarboxKind::from_view_dimension(logical.view_dimension)?,
                    logical_version: logical.version,
                })
            });
        let moon_texture = params
            .moon_albedo_texture
            .zip(environment.moon_texture)
            .and_then(|(source, logical)| {
                (logical.ready && logical.view_dimension == wgpu::TextureViewDimension::D2)
                    .then_some(AtmosphereTextureSnapshot {
                        source,
                        logical_version: logical.version,
                    })
            });

        let options = ShaderCompilationOptions::default();
        let transmittance_shader_identity =
            shader_manager.pipeline_shader_hash(self.transmittance.kind.shader_source(), &options);
        let multi_scatter_shader_identity =
            shader_manager.pipeline_shader_hash(self.multi_scatter.kind.shader_source(), &options);
        let sky_view_shader_identity =
            shader_manager.pipeline_shader_hash(self.sky_view.kind.shader_source(), &options);
        let sky_to_cube_pipeline_key = AtmosphereSkyToCubePipelineKey {
            starbox_kind: starbox.map_or(AtmosphereStarboxKind::None, |source| source.kind),
            moon_texture: moon_texture.is_some(),
        };
        let mut sky_to_cube_options = ShaderCompilationOptions::default();
        sky_to_cube_pipeline_key.apply_shader_defines(&mut sky_to_cube_options.defines);
        let sky_to_cube_shader_identity = shader_manager.pipeline_shader_hash(
            ShaderSource::File("entry/utility/atmosphere/sky_to_cube"),
            &sky_to_cube_options,
        );

        AtmosphereSnapshot {
            scene_id,
            gpu_params: GpuAtmosphereParams::from_scene(params),
            bake_params: GpuBakeParams::from_scene(params),
            physics_hash: environment.physics_hash,
            transmittance_shader_identity,
            multi_scatter_shader_identity,
            sky_view_shader_identity,
            sky_to_cube_shader_identity,
            sky_to_cube_pipeline_key,
            starbox,
            moon_texture,
        }
    }

    pub(crate) fn plan_luts(
        &mut self,
        registry: &mut HistoryRegistry,
        snapshot: &AtmosphereSnapshot,
        attempt_serial: u64,
    ) -> AtmosphereCacheSnapshot {
        let domain =
            self.domains
                .entry(snapshot.scene_id)
                .or_insert_with(|| AtmosphereCacheDomain {
                    transmittance_cache: registry.create_cache(),
                    multi_scatter_cache: registry.create_cache(),
                    last_used_frame: attempt_serial,
                });
        domain.last_used_frame = attempt_serial;

        let transmittance_key = transmittance_cache_key(
            snapshot.physics_hash,
            snapshot.transmittance_shader_identity,
        );
        let transmittance_plan = registry.plan_cache(
            domain.transmittance_cache,
            transmittance_key,
            [transmittance_spec()],
            attempt_serial,
        );
        let multi_scatter_key = multi_scatter_cache_key(
            transmittance_key,
            transmittance_plan.epoch,
            snapshot.multi_scatter_shader_identity,
        );
        let multi_scatter_plan = registry.plan_cache(
            domain.multi_scatter_cache,
            multi_scatter_key,
            [multi_scatter_spec()],
            attempt_serial,
        );

        AtmosphereCacheSnapshot {
            transmittance_plan,
            multi_scatter_plan,
        }
    }

    /// Retire both independent LUT cache identities for inactive scenes.
    pub(crate) fn maintain(&mut self, registry: &mut HistoryRegistry, current_frame: u64) {
        let retired: Vec<u32> = self
            .domains
            .iter()
            .filter_map(|(&scene_id, domain)| {
                (current_frame.saturating_sub(domain.last_used_frame) > SCENE_CACHE_TTL)
                    .then_some(scene_id)
            })
            .collect();
        for scene_id in retired {
            let domain = self
                .domains
                .remove(&scene_id)
                .expect("retired atmosphere cache domain disappeared");
            let removed_transmittance = registry.remove_cache(domain.transmittance_cache);
            let removed_multi_scatter = registry.remove_cache(domain.multi_scatter_cache);
            debug_assert!(removed_transmittance);
            debug_assert!(removed_multi_scatter);
        }
    }

    #[cfg(test)]
    pub(crate) fn has_scene_state(&self, scene_id: u32) -> bool {
        self.domains.contains_key(&scene_id)
    }

    /// Add logical LUT writers and the transient SkyView producer.
    ///
    /// `environment_base` is present only when EnvironmentCache requests a new
    /// procedural base publication. Visual SkyView remains independent, so a
    /// pending optional star/moon texture cannot suppress the sky itself.
    pub(crate) fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: &AtmosphereSnapshot,
        cache: &AtmosphereCacheSnapshot,
        environment_base: Option<TextureNodeId>,
    ) -> AtmosphereGraphOutput {
        let snapshot = *snapshot;
        let atmosphere_params = ctx.graph.add_pass("Atmosphere_Params_Upload", |builder| {
            let output = builder.create_buffer("Atmosphere_Params", atmosphere_params_desc());
            (
                AtmosphereParamsUploadNode {
                    output,
                    params: snapshot.gpu_params,
                },
                output,
            )
        });
        let bake_params = ctx
            .graph
            .add_pass("Atmosphere_Bake_Params_Upload", |builder| {
                let output = builder.create_buffer("Atmosphere_Bake_Params", bake_params_desc());
                (
                    AtmosphereBakeParamsUploadNode {
                        output,
                        params: snapshot.bake_params,
                    },
                    output,
                )
            });

        let transmittance = ctx.graph.import_persistent_texture(
            "Atmosphere_Transmittance",
            cache.transmittance_plan.textures[0],
        );
        if let Some(token) = cache.transmittance_plan.commit {
            let feature = &mut self.transmittance;
            ctx.graph.add_pass("Atmosphere_Transmittance", |builder| {
                builder.read_buffer(atmosphere_params);
                builder.write_texture(transmittance);
                (
                    AtmosphereTransmittanceNode {
                        feature,
                        expected_shader_identity: snapshot.transmittance_shader_identity,
                        output_tex: transmittance,
                        atmosphere_params,
                        bind_group: None,
                    },
                    (),
                )
            });
            ctx.graph.register_cache_commit(token, &[transmittance]);
        }

        let multi_scatter = ctx.graph.import_persistent_texture(
            "Atmosphere_MultiScatter",
            cache.multi_scatter_plan.textures[0],
        );
        if let Some(token) = cache.multi_scatter_plan.commit {
            let feature = &mut self.multi_scatter;
            ctx.graph.add_pass("Atmosphere_MultiScatter", |builder| {
                builder.read_buffer(atmosphere_params);
                builder.read_texture(transmittance);
                builder.write_texture(multi_scatter);
                (
                    AtmosphereMultiScatterNode {
                        feature,
                        expected_shader_identity: snapshot.multi_scatter_shader_identity,
                        transmittance,
                        output_tex: multi_scatter,
                        atmosphere_params,
                        bind_group: None,
                    },
                    (),
                )
            });
            ctx.graph.register_cache_commit(token, &[multi_scatter]);
        }

        let sky_view_feature = &mut self.sky_view;
        let sky_view = ctx.graph.add_pass("Atmosphere_SkyView", |builder| {
            builder.read_buffer(atmosphere_params);
            builder.read_texture(transmittance);
            builder.read_texture(multi_scatter);
            let output = builder.create_texture("Atmosphere_SkyView", sky_view_desc());
            (
                AtmosphereSkyViewNode {
                    feature: sky_view_feature,
                    expected_shader_identity: snapshot.sky_view_shader_identity,
                    transmittance,
                    multi_scatter,
                    output_tex: output,
                    atmosphere_params,
                    bind_group: None,
                },
                output,
            )
        });

        if let Some(base_cube) = environment_base {
            let feature = &mut self.sky_to_cube;
            ctx.graph.add_pass("Atmosphere_SkyToCube", |builder| {
                builder.read_texture(transmittance);
                builder.read_texture(sky_view);
                builder.write_texture(base_cube);
                let canonical_bake_params =
                    builder.create_buffer("Atmosphere_Environment_Bake_Params", bake_params_desc());
                (
                    AtmosphereSkyToCubeNode {
                        feature,
                        snapshot,
                        base_cube,
                        transmittance,
                        sky_view,
                        canonical_bake_params,
                        bind_group: None,
                    },
                    (),
                )
            });
        }

        AtmosphereGraphOutput {
            sky_view,
            transmittance,
            bake_params,
        }
    }
}

fn transmittance_cache_key(physics_hash: u64, shader_identity: u64) -> CacheKey {
    let mut low = FxHasher::default();
    TRANSMITTANCE_CACHE_SCHEMA.hash(&mut low);
    physics_hash.hash(&mut low);
    shader_identity.hash(&mut low);
    CacheKey::new((u128::from(physics_hash) << 64) | u128::from(low.finish()))
}

fn multi_scatter_cache_key(
    transmittance_key: CacheKey,
    transmittance_epoch: u64,
    shader_identity: u64,
) -> CacheKey {
    let mut high = FxHasher::default();
    MULTI_SCATTER_CACHE_SCHEMA.hash(&mut high);
    transmittance_key.get().hash(&mut high);
    transmittance_epoch.hash(&mut high);
    let mut low = FxHasher::default();
    transmittance_key.get().hash(&mut low);
    transmittance_epoch.hash(&mut low);
    shader_identity.hash(&mut low);
    CacheKey::new((u128::from(high.finish()) << 64) | u128::from(low.finish()))
}

fn atmosphere_params_desc() -> BufferDesc {
    BufferDesc::new(
        std::mem::size_of::<GpuAtmosphereParams>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    )
}

fn bake_params_desc() -> BufferDesc {
    BufferDesc::new(
        std::mem::size_of::<GpuBakeParams>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    )
}

fn transmittance_desc() -> TextureDesc {
    TextureDesc::new_2d(
        TRANSMITTANCE_WIDTH,
        TRANSMITTANCE_HEIGHT,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
    )
}

fn multi_scatter_desc() -> TextureDesc {
    TextureDesc::new_2d(
        MULTI_SCATTER_SIZE,
        MULTI_SCATTER_SIZE,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
    )
}

fn sky_view_desc() -> TextureDesc {
    TextureDesc::new_2d(
        SKY_VIEW_WIDTH,
        SKY_VIEW_HEIGHT,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
    )
}

fn transmittance_spec() -> PersistentTextureSpec {
    PersistentTextureSpec::new(transmittance_desc(), Some(wgpu::TextureViewDimension::D2))
}

fn multi_scatter_spec() -> PersistentTextureSpec {
    PersistentTextureSpec::new(multi_scatter_desc(), Some(wgpu::TextureViewDimension::D2))
}

fn create_compute_layout(
    device: &wgpu::Device,
    kind: AtmosphereComputeKind,
) -> Tracked<wgpu::BindGroupLayout> {
    let visibility = wgpu::ShaderStages::COMPUTE;
    let sampled = wgpu::TextureSampleType::Float { filterable: true };
    let storage = |binding| {
        binding_layout::storage_texture(
            binding,
            visibility,
            wgpu::StorageTextureAccess::WriteOnly,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureViewDimension::D2,
        )
    };
    let (label, entries): (&str, Vec<wgpu::BindGroupLayoutEntry>) = match kind {
        AtmosphereComputeKind::Transmittance => (
            "Atmo Transmittance BGL",
            vec![binding_layout::uniform_buffer(0, visibility), storage(1)],
        ),
        AtmosphereComputeKind::MultiScatter => (
            "Atmo Multi-Scatter BGL",
            vec![
                binding_layout::uniform_buffer(0, visibility),
                binding_layout::texture_2d(1, visibility, sampled),
                binding_layout::sampler(2, visibility, wgpu::SamplerBindingType::Filtering),
                storage(3),
            ],
        ),
        AtmosphereComputeKind::SkyView => (
            "Atmo Sky-View BGL",
            vec![
                binding_layout::uniform_buffer(0, visibility),
                binding_layout::texture_2d(1, visibility, sampled),
                binding_layout::texture_2d(2, visibility, sampled),
                binding_layout::sampler(3, visibility, wgpu::SamplerBindingType::Filtering),
                storage(4),
            ],
        ),
    };
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        }),
    )
}

fn create_sky_to_cube_layout(
    device: &wgpu::Device,
    starbox_view_dimension: Option<wgpu::TextureViewDimension>,
    label: &str,
) -> wgpu::BindGroupLayout {
    let visibility = wgpu::ShaderStages::COMPUTE;
    let sampled = wgpu::TextureSampleType::Float { filterable: true };
    let mut entries = vec![
        binding_layout::texture_2d(0, visibility, sampled),
        binding_layout::sampler(1, visibility, wgpu::SamplerBindingType::Filtering),
        binding_layout::uniform_buffer(2, visibility),
        binding_layout::storage_texture(
            3,
            visibility,
            wgpu::StorageTextureAccess::WriteOnly,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureViewDimension::D2Array,
        ),
        binding_layout::texture_2d(4, visibility, sampled),
        binding_layout::texture_2d(5, visibility, sampled),
    ];
    if let Some(view_dimension) = starbox_view_dimension {
        entries.push(binding_layout::texture(
            6,
            visibility,
            sampled,
            view_dimension,
            false,
        ));
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

struct AtmosphereParamsUploadNode {
    output: BufferNodeId,
    params: GpuAtmosphereParams,
}

impl PassNode<'_> for AtmosphereParamsUploadNode {
    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.output),
            0,
            bytemuck::bytes_of(&self.params),
        );
    }

    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
}

struct AtmosphereBakeParamsUploadNode {
    output: BufferNodeId,
    params: GpuBakeParams,
}

impl PassNode<'_> for AtmosphereBakeParamsUploadNode {
    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.output),
            0,
            bytemuck::bytes_of(&self.params),
        );
    }

    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
}

struct AtmosphereTransmittanceNode<'a> {
    feature: &'a mut AtmosphereComputeFeature,
    expected_shader_identity: u64,
    output_tex: TextureNodeId,
    atmosphere_params: BufferNodeId,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for AtmosphereTransmittanceNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature
            .ensure_resources(ctx, self.expected_shader_identity);
        self.bind_group = Some(
            ctx.build_bind_group(self.feature.layout(), Some("Atmo Transmittance BG"))
                .bind_buffer(0, self.atmosphere_params)
                .bind_texture(1, self.output_tex)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Atmosphere Transmittance LUT"),
            timestamp_writes: None,
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_compute_pipeline(self.feature.pipeline_id()),
        );
        pass.set_bind_group(
            0,
            ctx.resolve_prepared_bind_group(
                self.bind_group.expect("atmo transmittance bg missing"),
            ),
            &[],
        );
        pass.dispatch_workgroups(
            TRANSMITTANCE_WIDTH.div_ceil(8),
            TRANSMITTANCE_HEIGHT.div_ceil(8),
            1,
        );
    }
}

struct AtmosphereMultiScatterNode<'a> {
    feature: &'a mut AtmosphereComputeFeature,
    expected_shader_identity: u64,
    transmittance: TextureNodeId,
    output_tex: TextureNodeId,
    atmosphere_params: BufferNodeId,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for AtmosphereMultiScatterNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature
            .ensure_resources(ctx, self.expected_shader_identity);
        self.bind_group = Some(
            ctx.build_bind_group(self.feature.layout(), Some("Atmo Multi-Scatter BG"))
                .bind_buffer(0, self.atmosphere_params)
                .bind_texture(1, self.transmittance)
                .bind_common_sampler(2, CommonSampler::LinearClamp)
                .bind_texture(3, self.output_tex)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Atmosphere Multi-Scatter LUT"),
            timestamp_writes: None,
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_compute_pipeline(self.feature.pipeline_id()),
        );
        pass.set_bind_group(
            0,
            ctx.resolve_prepared_bind_group(
                self.bind_group.expect("atmo multi-scatter bg missing"),
            ),
            &[],
        );
        pass.dispatch_workgroups(
            MULTI_SCATTER_SIZE.div_ceil(8),
            MULTI_SCATTER_SIZE.div_ceil(8),
            1,
        );
    }
}

struct AtmosphereSkyViewNode<'a> {
    feature: &'a mut AtmosphereComputeFeature,
    expected_shader_identity: u64,
    transmittance: TextureNodeId,
    multi_scatter: TextureNodeId,
    output_tex: TextureNodeId,
    atmosphere_params: BufferNodeId,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for AtmosphereSkyViewNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature
            .ensure_resources(ctx, self.expected_shader_identity);
        self.bind_group = Some(
            ctx.build_bind_group(self.feature.layout(), Some("Atmo Sky-View BG"))
                .bind_buffer(0, self.atmosphere_params)
                .bind_texture(1, self.transmittance)
                .bind_texture(2, self.multi_scatter)
                .bind_common_sampler(3, CommonSampler::LinearClamp)
                .bind_texture(4, self.output_tex)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Atmosphere Sky-View LUT"),
            timestamp_writes: None,
        });
        pass.set_pipeline(
            ctx.pipeline_cache
                .get_compute_pipeline(self.feature.pipeline_id()),
        );
        pass.set_bind_group(
            0,
            ctx.resolve_prepared_bind_group(self.bind_group.expect("atmo sky-view bg missing")),
            &[],
        );
        pass.dispatch_workgroups(SKY_VIEW_WIDTH.div_ceil(8), SKY_VIEW_HEIGHT.div_ceil(8), 1);
    }
}

struct AtmosphereSkyToCubeNode<'a> {
    feature: &'a mut AtmosphereSkyToCubeFeature,
    snapshot: AtmosphereSnapshot,
    base_cube: TextureNodeId,
    transmittance: TextureNodeId,
    sky_view: TextureNodeId,
    canonical_bake_params: BufferNodeId,
    bind_group: Option<PreparedBindGroupId>,
}

impl<'a> PassNode<'a> for AtmosphereSkyToCubeNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(
            ctx,
            self.snapshot.sky_to_cube_pipeline_key,
            self.snapshot.sky_to_cube_shader_identity,
        );
        ctx.mipmap_generator
            .ensure_pipeline(ctx.device, wgpu::TextureFormat::Rgba16Float);
        let canonical = GpuBakeParams {
            star_rotation: 0.0,
            ..self.snapshot.bake_params
        };
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.canonical_bake_params),
            0,
            bytemuck::bytes_of(&canonical),
        );

        let starbox = self.snapshot.starbox.map(|source| {
            let prepared = ctx
                .uploads
                .resolve_texture_source(ctx.assets, source.source, source.logical_version)
                .expect("ready atmosphere starbox must keep its snapshot version through prepare");
            assert_eq!(
                AtmosphereStarboxKind::from_view_dimension(prepared.view_dimension),
                Some(source.kind),
                "atmosphere starbox dimension changed after snapshot extraction"
            );
            assert_eq!(
                prepared.logical_version, source.logical_version,
                "atmosphere starbox version changed after snapshot extraction"
            );
            prepared
        });
        let moon = self.snapshot.moon_texture.map(|source| {
            let prepared = ctx
                .uploads
                .resolve_texture_source(ctx.assets, source.source, source.logical_version)
                .expect(
                    "ready atmosphere moon texture must keep its snapshot version through prepare",
                );
            assert_eq!(
                prepared.view_dimension,
                wgpu::TextureViewDimension::D2,
                "atmosphere moon texture dimension changed after snapshot extraction"
            );
            assert_eq!(
                prepared.logical_version, source.logical_version,
                "atmosphere moon texture version changed after snapshot extraction"
            );
            prepared
        });
        let white_moon = ctx.system_textures.white_2d.clone();
        let base_cube_storage_view =
            Tracked::new(ctx.views.get_texture(self.base_cube).create_view(
                &wgpu::TextureViewDescriptor {
                    label: Some("Scene Environment Base Cube Mip0"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                },
            ));

        let key = self.snapshot.sky_to_cube_pipeline_key;
        let layout = self.feature.layout(key.starbox_kind);
        let builder = ctx
            .build_bind_group(layout, Some("Atmo Sky-to-Cube BG"))
            .bind_texture(0, self.sky_view)
            .bind_common_sampler(1, CommonSampler::LinearClamp)
            .bind_buffer(2, self.canonical_bake_params)
            .bind_tracked_texture_view(3, &base_cube_storage_view)
            .bind_texture(4, self.transmittance);
        let builder = if let Some(moon) = moon.as_ref() {
            builder.bind_texture_view_with_id(5, &moon.view, moon.view_id)
        } else {
            builder.bind_tracked_texture_view(5, &white_moon)
        };
        self.bind_group = Some(if let Some(starbox) = starbox.as_ref() {
            builder
                .bind_texture_view_with_id(6, &starbox.view, starbox.view_id)
                .build()
        } else {
            builder.build()
        });
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let dispatch = ctx.get_texture(self.base_cube).width().div_ceil(8);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Atmosphere Sky-to-Cube Bake"),
                timestamp_writes: None,
            });
            pass.set_pipeline(
                ctx.pipeline_cache.get_compute_pipeline(
                    self.feature
                        .pipeline_id(self.snapshot.sky_to_cube_pipeline_key),
                ),
            );
            pass.set_bind_group(
                0,
                ctx.resolve_prepared_bind_group(
                    self.bind_group.expect("atmo sky-to-cube bg missing"),
                ),
                &[],
            );
            pass.dispatch_workgroups(dispatch, dispatch, 6);
        }
        ctx.mipmap_generator
            .generate(ctx.device, encoder, ctx.get_texture(self.base_cube));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::procedural_physics_hash;
    use crate::graph::core::graph::FrameConfig;
    use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};

    struct TestNode;

    impl PassNode<'_> for TestNode {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    #[derive(Clone, Copy)]
    enum TestRoot {
        None,
        Transmittance,
        SkyView,
    }

    fn snapshot(scene_id: u32, physics_hash: u64) -> AtmosphereSnapshot {
        AtmosphereSnapshot {
            scene_id,
            gpu_params: bytemuck::Zeroable::zeroed(),
            bake_params: bytemuck::Zeroable::zeroed(),
            physics_hash,
            transmittance_shader_identity: 101,
            multi_scatter_shader_identity: 202,
            sky_view_shader_identity: 303,
            sky_to_cube_shader_identity: 404,
            sky_to_cube_pipeline_key: AtmosphereSkyToCubePipelineKey {
                starbox_kind: AtmosphereStarboxKind::None,
                moon_texture: false,
            },
            starbox: None,
            moon_texture: None,
        }
    }

    fn live_passes(root: TestRoot) -> Vec<&'static str> {
        let mut registry = HistoryRegistry::new();
        let mut feature = AtmosphereFeature::new();
        let snapshot = snapshot(5, 17);
        let cache = feature.plan_luts(&mut registry, &snapshot, 1);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let config = FrameConfig {
            width: 320,
            height: 180,
            depth_format: wgpu::TextureFormat::Depth32Float,
            msaa_samples: 1,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            hdr_format: crate::HDR_TEXTURE_FORMAT,
        };
        let output = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            feature.add_to_graph(&mut ctx, &snapshot, &cache, None)
        };
        match root {
            TestRoot::None => {}
            TestRoot::Transmittance => {
                graph.add_pass("Atmosphere_Test_Root", |builder| {
                    builder.read_texture(output.transmittance);
                    builder.mark_side_effect();
                    (TestNode, ())
                });
            }
            TestRoot::SkyView => {
                graph.add_pass("Atmosphere_Test_Root", |builder| {
                    builder.read_texture(output.sky_view);
                    builder.mark_side_effect();
                    (TestNode, ())
                });
            }
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
    fn physics_mutation_rekeys_both_lut_plans() {
        let mut registry = HistoryRegistry::new();
        let mut feature = AtmosphereFeature::new();
        let first = feature.plan_luts(&mut registry, &snapshot(1, 11), 1);
        registry
            .commit_cache(
                first
                    .transmittance_plan
                    .commit
                    .expect("cold transmittance plan must publish"),
                &[first.transmittance_plan.textures[0].id],
            )
            .expect("transmittance publication must succeed");
        registry
            .commit_cache(
                first
                    .multi_scatter_plan
                    .commit
                    .expect("cold multi-scatter plan must publish"),
                &[first.multi_scatter_plan.textures[0].id],
            )
            .expect("multi-scatter publication must succeed");

        let steady = feature.plan_luts(&mut registry, &snapshot(1, 11), 2);
        assert!(steady.transmittance_plan.valid);
        assert!(steady.multi_scatter_plan.valid);
        let changed = feature.plan_luts(&mut registry, &snapshot(1, 12), 3);
        assert!(!changed.transmittance_plan.valid);
        assert!(!changed.multi_scatter_plan.valid);
    }

    #[test]
    fn only_multi_shader_change_keeps_transmittance_valid() {
        let mut registry = HistoryRegistry::new();
        let mut feature = AtmosphereFeature::new();
        let initial = feature.plan_luts(&mut registry, &snapshot(2, 7), 1);
        registry
            .commit_cache(
                initial.transmittance_plan.commit.unwrap(),
                &[initial.transmittance_plan.textures[0].id],
            )
            .unwrap();
        registry
            .commit_cache(
                initial.multi_scatter_plan.commit.unwrap(),
                &[initial.multi_scatter_plan.textures[0].id],
            )
            .unwrap();

        let mut changed = snapshot(2, 7);
        changed.multi_scatter_shader_identity += 1;
        let next = feature.plan_luts(&mut registry, &changed, 2);
        assert!(next.transmittance_plan.valid);
        assert!(!next.multi_scatter_plan.valid);
    }

    #[test]
    fn dynamic_sun_does_not_change_static_physics_hash() {
        let mut params = ProceduralSkyParams::golden_hour();
        let initial = procedural_physics_hash(&params);
        params.sun_direction = glam::Vec3::X;
        params.sun_intensity *= 2.0;
        assert_eq!(initial, procedural_physics_hash(&params));
        params.rayleigh_scale_height *= 1.1;
        assert_ne!(initial, procedural_physics_hash(&params));
    }

    #[test]
    fn cache_commit_registration_does_not_keep_dead_luts_alive() {
        assert!(live_passes(TestRoot::None).is_empty());
    }

    #[test]
    fn transmittance_only_demand_does_not_publish_multi_scatter() {
        let live = live_passes(TestRoot::Transmittance);
        assert!(live.contains(&"Atmosphere_Params_Upload"));
        assert!(live.contains(&"Atmosphere_Transmittance"));
        assert!(!live.contains(&"Atmosphere_MultiScatter"));
        assert!(!live.contains(&"Atmosphere_SkyView"));
        assert!(!live.contains(&"Atmosphere_Bake_Params_Upload"));
    }

    #[test]
    fn sky_view_demand_keeps_both_lut_writers_live() {
        let live = live_passes(TestRoot::SkyView);
        assert!(live.contains(&"Atmosphere_Params_Upload"));
        assert!(live.contains(&"Atmosphere_Transmittance"));
        assert!(live.contains(&"Atmosphere_MultiScatter"));
        assert!(live.contains(&"Atmosphere_SkyView"));
        assert!(!live.contains(&"Atmosphere_Bake_Params_Upload"));
    }

    #[test]
    fn ttl_retires_both_scene_cache_identities() {
        let mut registry = HistoryRegistry::new();
        let mut feature = AtmosphereFeature::new();
        let _ = feature.plan_luts(&mut registry, &snapshot(9, 1), 10);
        let (transmittance_cache, multi_scatter_cache) = {
            let domain = feature
                .domains
                .get(&9)
                .expect("atmosphere cache domain missing");
            (domain.transmittance_cache, domain.multi_scatter_cache)
        };
        feature.maintain(&mut registry, 10 + SCENE_CACHE_TTL);
        assert!(feature.has_scene_state(9));
        feature.maintain(&mut registry, 11 + SCENE_CACHE_TTL);
        assert!(!feature.has_scene_state(9));
        assert!(!registry.remove_cache(transmittance_cache));
        assert!(!registry.remove_cache(multi_scatter_cache));
    }
}
