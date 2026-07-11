//! Logical scene-environment caches and deferred PMREM generation.
//!
//! [`EnvironmentCacheFeature`] owns only logical, per-scene cache domains and
//! publication metadata. [`IblComputeFeature`] owns only cross-frame static
//! pipeline resources. Keeping the two lifetimes separate lets the composer
//! observe post-submit cache receipts while a live graph still borrows the IBL
//! writer feature.

use std::hash::{Hash, Hasher};

use glam::Vec3;
use rustc_hash::{FxHashMap, FxHasher};

use crate::core::gpu::{CommonSampler, CubeSourceType, GpuEnvironment, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, CacheCommitToken, CacheId, CacheKey, CachePlan,
    CachePublicationReceipt, ExecuteContext, HistoryRegistry, PassNode, PersistentTextureRef,
    PersistentTextureSpec, PrepareContext, TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};

const SCENE_CACHE_TTL: u64 = 120;
const STATIC_PMREM_SAMPLE_COUNT: u32 = 4096;
const DYNAMIC_PMREM_SAMPLE_COUNT: u32 = 64;
const PROCEDURAL_IBL_CELESTIAL_UPDATE_THRESHOLD_DEGREES: f32 = 0.5;
const ENVIRONMENT_BASE_CACHE_SCHEMA: u64 = 1;
const ENVIRONMENT_PMREM_CACHE_SCHEMA: u64 = 1;
const PMREM_PARAMS_SIZE: u64 = std::mem::size_of::<[f32; 4]>() as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IblPipelineVariant {
    Static,
    Dynamic,
}

impl IblPipelineVariant {
    #[inline]
    #[must_use]
    const fn from_source_type(source_type: CubeSourceType) -> Self {
        match source_type {
            CubeSourceType::Procedural => Self::Dynamic,
            CubeSourceType::Equirectangular | CubeSourceType::Cubemap => Self::Static,
        }
    }

    #[inline]
    #[must_use]
    const fn sample_count(self) -> u32 {
        match self {
            Self::Static => STATIC_PMREM_SAMPLE_COUNT,
            Self::Dynamic => DYNAMIC_PMREM_SAMPLE_COUNT,
        }
    }
}

/// Lightweight graph-build result for one scene environment.
///
/// Ready sources carry both cache plans. A pending source never changes the
/// registry's desired key; it may instead carry synthetic read-only base and
/// PMREM plans for the last successfully published contents.
#[derive(Debug, Clone, Copy)]
pub struct EnvironmentCacheSnapshot {
    pub scene_id: u32,
    pub source_type: CubeSourceType,
    pub base_key: Option<CacheKey>,
    pub base_plan: Option<CachePlan<1>>,
    pub pmrem_key: Option<CacheKey>,
    pub pmrem_plan: Option<CachePlan<1>>,
    pub ibl_variant: IblPipelineVariant,
    /// L1 shader identity used by the logical PMREM cache key.
    pub ibl_shader_identity: u64,
    pub pmrem_size: u32,
    pub pmrem_mip_level_count: u32,
    pub params_stride: u64,
}

impl EnvironmentCacheSnapshot {
    /// Whether this snapshot requires a live PMREM writer.
    #[inline]
    #[must_use]
    pub fn needs_pmrem_write(self) -> bool {
        self.pmrem_plan.and_then(|plan| plan.commit).is_some()
    }

    /// Published or planned PMREM texture, if the scene currently has one.
    #[inline]
    #[must_use]
    pub fn pmrem_texture(self) -> Option<PersistentTextureRef> {
        self.pmrem_plan.map(|plan| plan.textures[0])
    }

    /// Published or planned base cube. A pending replacement source may expose
    /// the last valid published base as a read-only fallback.
    #[inline]
    #[must_use]
    pub fn base_texture(self) -> Option<PersistentTextureRef> {
        self.base_plan.map(|plan| plan.textures[0])
    }
}

#[derive(Debug, Clone, Copy)]
struct PublishedBaseMetadata {
    key: CacheKey,
    epoch: u64,
    texture: PersistentTextureRef,
    procedural_physics_hash: u64,
    procedural_static_bake_hash: u64,
    producer_shader_identity: u64,
    sun_direction: Option<Vec3>,
    moon_direction: Option<Vec3>,
}

#[derive(Debug, Clone, Copy)]
struct PublishedPmremMetadata {
    key: CacheKey,
    epoch: u64,
    texture: PersistentTextureRef,
    size: u32,
    mip_level_count: u32,
}

#[derive(Debug, Clone, Copy)]
struct PendingBasePublication {
    token: CacheCommitToken,
    metadata: PublishedBaseMetadata,
}

#[derive(Debug, Clone, Copy)]
struct PendingPmremPublication {
    token: CacheCommitToken,
    metadata: PublishedPmremMetadata,
}

struct EnvironmentCacheDomain {
    base_cache: CacheId,
    pmrem_cache: CacheId,
    published_base: Option<PublishedBaseMetadata>,
    published_pmrem: Option<PublishedPmremMetadata>,
    pending_base: Option<PendingBasePublication>,
    pending_pmrem: Option<PendingPmremPublication>,
    last_used_frame: u64,
}

/// Scene-local logical cache coordinator.
///
/// The feature never owns WGPU resources. [`HistoryRegistry`] remains the
/// sole authority for physical allocation and validity; the metadata kept here
/// exists only to construct future keys, retain the last published procedural
/// sun/moon directions, and correlate typed post-submit receipts.
#[derive(Default)]
pub struct EnvironmentCacheFeature {
    domains: FxHashMap<u32, EnvironmentCacheDomain>,
}

impl EnvironmentCacheFeature {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Plan the base-cube and PMREM caches for one scene without performing
    /// any CPU-heavy derivation or WGPU work.
    #[allow(clippy::too_many_arguments)]
    pub fn extract(
        &mut self,
        registry: &mut HistoryRegistry,
        scene_id: u32,
        environment: &GpuEnvironment,
        attempt_serial: u64,
        base_producer_shader_identity: u64,
        ibl_shader_identity: u64,
        procedural_actual_sun: Option<Vec3>,
        procedural_actual_moon: Option<Vec3>,
        min_uniform_buffer_offset_alignment: u32,
    ) -> EnvironmentCacheSnapshot {
        let variant = IblPipelineVariant::from_source_type(environment.source_type);
        let params_stride = align_up(
            PMREM_PARAMS_SIZE,
            u64::from(min_uniform_buffer_offset_alignment).max(PMREM_PARAMS_SIZE),
        );
        let pmrem_mip_level_count = full_mip_count(environment.pmrem_size);

        let domain = self
            .domains
            .entry(scene_id)
            .or_insert_with(|| EnvironmentCacheDomain {
                base_cache: registry.create_cache(),
                pmrem_cache: registry.create_cache(),
                published_base: None,
                published_pmrem: None,
                pending_base: None,
                pending_pmrem: None,
                last_used_frame: environment.last_used_frame,
            });
        domain.last_used_frame = environment.last_used_frame;

        if !environment.source_ready {
            // Readiness is deliberately checked before planning: a transiently
            // missing asset must not re-key and destroy a still-usable PMREM.
            domain.pending_base = None;
            domain.pending_pmrem = None;

            let published_base = domain
                .published_base
                .filter(|published| registry.is_cache_valid(domain.base_cache, published.key));
            let base_plan = published_base.map(|published| CachePlan {
                textures: [published.texture],
                valid: true,
                epoch: published.epoch,
                commit: None,
            });
            let published_pmrem = domain
                .published_pmrem
                .filter(|published| registry.is_cache_valid(domain.pmrem_cache, published.key));
            let pmrem_plan = published_pmrem.map(|published| CachePlan {
                textures: [published.texture],
                valid: true,
                epoch: published.epoch,
                commit: None,
            });

            return EnvironmentCacheSnapshot {
                scene_id,
                source_type: environment.source_type,
                base_key: published_base.map(|published| published.key),
                base_plan,
                pmrem_key: published_pmrem.map(|published| published.key),
                pmrem_plan,
                ibl_variant: variant,
                ibl_shader_identity,
                pmrem_size: published_pmrem
                    .map_or(environment.pmrem_size, |published| published.size),
                pmrem_mip_level_count: published_pmrem
                    .map_or(pmrem_mip_level_count, |published| published.mip_level_count),
                params_stride,
            };
        }

        let base_key = if environment.source_type == CubeSourceType::Procedural {
            let actual_sun = procedural_actual_sun
                .expect("procedural environment cache planning requires the actual sun direction");
            let actual_moon = procedural_actual_moon
                .expect("procedural environment cache planning requires the actual moon direction");
            domain
                .published_base
                .filter(|published| {
                    published.procedural_physics_hash == environment.procedural_physics_hash
                        && published.procedural_static_bake_hash
                            == environment.procedural_static_bake_hash
                        && published.producer_shader_identity == base_producer_shader_identity
                        && published
                            .sun_direction
                            .is_some_and(|sun| directions_within_update_threshold(sun, actual_sun))
                        && published.moon_direction.is_some_and(|moon| {
                            directions_within_update_threshold(moon, actual_moon)
                        })
                })
                .map_or_else(
                    || procedural_base_key(environment, base_producer_shader_identity),
                    |published| published.key,
                )
        } else {
            texture_base_key(environment, base_producer_shader_identity)
        };
        let base_spec = environment_base_spec(environment.base_cube_size);
        let base_plan =
            registry.plan_cache(domain.base_cache, base_key, [base_spec], attempt_serial);

        let pmrem_key = environment_pmrem_key(
            base_key,
            base_plan.epoch,
            environment.pmrem_size,
            variant,
            ibl_shader_identity,
        );
        let pmrem_spec = environment_pmrem_spec(environment.pmrem_size);
        let pmrem_plan =
            registry.plan_cache(domain.pmrem_cache, pmrem_key, [pmrem_spec], attempt_serial);

        domain.pending_base = base_plan.commit.map(|token| PendingBasePublication {
            token,
            metadata: PublishedBaseMetadata {
                key: base_key,
                epoch: base_plan.epoch,
                texture: base_plan.textures[0],
                procedural_physics_hash: environment.procedural_physics_hash,
                procedural_static_bake_hash: environment.procedural_static_bake_hash,
                producer_shader_identity: base_producer_shader_identity,
                sun_direction: (environment.source_type == CubeSourceType::Procedural)
                    .then_some(procedural_actual_sun)
                    .flatten(),
                moon_direction: (environment.source_type == CubeSourceType::Procedural)
                    .then_some(procedural_actual_moon)
                    .flatten(),
            },
        });
        domain.pending_pmrem = pmrem_plan.commit.map(|token| PendingPmremPublication {
            token,
            metadata: PublishedPmremMetadata {
                key: pmrem_key,
                epoch: pmrem_plan.epoch,
                texture: pmrem_plan.textures[0],
                size: environment.pmrem_size,
                mip_level_count: pmrem_mip_level_count,
            },
        });

        EnvironmentCacheSnapshot {
            scene_id,
            source_type: environment.source_type,
            base_key: Some(base_key),
            base_plan: Some(base_plan),
            pmrem_key: Some(pmrem_key),
            pmrem_plan: Some(pmrem_plan),
            ibl_variant: variant,
            ibl_shader_identity,
            pmrem_size: environment.pmrem_size,
            pmrem_mip_level_count,
            params_stride,
        }
    }

    /// Promote only metadata acknowledged by the registry after successful
    /// queue submission. Dead or failed attempts therefore cannot advance the
    /// procedural sun/moon baselines or advertise a new PMREM.
    pub fn observe_publications(&mut self, receipts: &[CachePublicationReceipt]) {
        for receipt in receipts {
            for domain in self.domains.values_mut() {
                if receipt.cache_id == domain.base_cache {
                    if let Some(pending) = domain.pending_base
                        && receipt_matches(pending.token, *receipt)
                    {
                        domain.published_base = Some(pending.metadata);
                        domain.pending_base = None;
                    }
                } else if receipt.cache_id == domain.pmrem_cache
                    && let Some(pending) = domain.pending_pmrem
                    && receipt_matches(pending.token, *receipt)
                {
                    domain.published_pmrem = Some(pending.metadata);
                    domain.pending_pmrem = None;
                }
            }
        }
    }

    #[inline]
    #[must_use]
    pub fn last_published_sun_direction(&self, scene_id: u32) -> Option<Vec3> {
        self.domains
            .get(&scene_id)
            .and_then(|domain| domain.published_base)
            .and_then(|published| published.sun_direction)
    }

    #[must_use]
    pub fn published_pmrem_view<'a>(
        &self,
        scene_id: u32,
        registry: &'a HistoryRegistry,
    ) -> Option<&'a Tracked<wgpu::TextureView>> {
        let domain = self.domains.get(&scene_id)?;
        let published = domain.published_pmrem?;
        registry.get_valid_cache_texture_view(domain.pmrem_cache, published.key, 0)
    }

    /// Number of mip levels in the currently published PMREM, if it is still
    /// valid in the registry.
    #[must_use]
    pub fn published_pmrem_mip_level_count(
        &self,
        scene_id: u32,
        registry: &HistoryRegistry,
    ) -> Option<u32> {
        let domain = self.domains.get(&scene_id)?;
        let published = domain.published_pmrem?;
        registry
            .is_cache_valid(domain.pmrem_cache, published.key)
            .then_some(published.mip_level_count)
    }

    #[inline]
    #[must_use]
    pub fn has_scene_state(&self, scene_id: u32) -> bool {
        self.domains.contains_key(&scene_id)
    }

    /// Retire inactive cache domains atomically and return the associated
    /// scene IDs so the renderer can evict global bind groups that retain the
    /// old PMREM views.
    pub fn maintain(&mut self, registry: &mut HistoryRegistry, current_frame: u64) -> Vec<u32> {
        let retired: Vec<u32> = self
            .domains
            .iter()
            .filter_map(|(&scene_id, domain)| {
                (current_frame.saturating_sub(domain.last_used_frame) > SCENE_CACHE_TTL)
                    .then_some(scene_id)
            })
            .collect();

        for scene_id in &retired {
            let domain = self
                .domains
                .remove(scene_id)
                .expect("retired environment cache domain disappeared");
            let base_removed = registry.remove_cache(domain.base_cache);
            let pmrem_removed = registry.remove_cache(domain.pmrem_cache);
            debug_assert!(base_removed && pmrem_removed);
        }

        retired
    }
}

#[inline]
fn receipt_matches(token: CacheCommitToken, receipt: CachePublicationReceipt) -> bool {
    token.cache_id == receipt.cache_id
        && token.key == receipt.key
        && token.epoch == receipt.epoch
        && token.generation == receipt.generation
        && token.attempt_serial == receipt.attempt_serial
}

#[inline]
fn full_mip_count(size: u32) -> u32 {
    size.max(1).ilog2() + 1
}

#[inline]
fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn environment_base_spec(size: u32) -> PersistentTextureSpec {
    PersistentTextureSpec::new(
        TextureDesc::new(
            size.max(1),
            size.max(1),
            6,
            full_mip_count(size),
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
        Some(wgpu::TextureViewDimension::Cube),
    )
}

fn environment_pmrem_spec(size: u32) -> PersistentTextureSpec {
    PersistentTextureSpec::new(
        TextureDesc::new(
            size.max(1),
            size.max(1),
            6,
            full_mip_count(size),
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        ),
        Some(wgpu::TextureViewDimension::Cube),
    )
}

fn texture_base_key(environment: &GpuEnvironment, producer_shader_identity: u64) -> CacheKey {
    let source_kind = match environment.source_type {
        CubeSourceType::Equirectangular => 1_u8,
        CubeSourceType::Cubemap => 2_u8,
        CubeSourceType::Procedural => {
            unreachable!("procedural environments require an explicit sun-aware cache key")
        }
    };
    let mut dependencies = FxHasher::default();
    ENVIRONMENT_BASE_CACHE_SCHEMA.hash(&mut dependencies);
    source_kind.hash(&mut dependencies);
    environment.source_version.hash(&mut dependencies);
    environment.base_cube_size.max(1).hash(&mut dependencies);
    producer_shader_identity.hash(&mut dependencies);

    // Preserve the complete logical source version in one half instead of
    // collapsing every dependency into a single 64-bit digest.
    CacheKey::new(
        (u128::from(environment.source_version) << 64) | u128::from(dependencies.finish()),
    )
}

fn procedural_base_key(environment: &GpuEnvironment, producer_shader_identity: u64) -> CacheKey {
    let mut dependencies = FxHasher::default();
    ENVIRONMENT_BASE_CACHE_SCHEMA.hash(&mut dependencies);
    0_u8.hash(&mut dependencies);
    // The procedural source version is computed from the current static
    // atmosphere hashes and actual sun/moon directions. It is used only when
    // the published-key throttle above decides that a new bake is required.
    environment.source_version.hash(&mut dependencies);
    environment.base_cube_size.max(1).hash(&mut dependencies);
    producer_shader_identity.hash(&mut dependencies);

    CacheKey::new(
        (u128::from(environment.source_version) << 64) | u128::from(dependencies.finish()),
    )
}

fn directions_within_update_threshold(previous: Vec3, current: Vec3) -> bool {
    if previous == current {
        return true;
    }
    let previous = previous.normalize_or_zero();
    let current = current.normalize_or_zero();
    previous != Vec3::ZERO
        && current != Vec3::ZERO
        && previous.dot(current)
            > PROCEDURAL_IBL_CELESTIAL_UPDATE_THRESHOLD_DEGREES
                .to_radians()
                .cos()
}

fn environment_pmrem_key(
    base_key: CacheKey,
    base_epoch: u64,
    pmrem_size: u32,
    variant: IblPipelineVariant,
    ibl_shader_identity: u64,
) -> CacheKey {
    let mut high = FxHasher::default();
    ENVIRONMENT_PMREM_CACHE_SCHEMA.hash(&mut high);
    base_key.get().hash(&mut high);
    base_epoch.hash(&mut high);

    let mut low = FxHasher::default();
    base_key.get().hash(&mut low);
    pmrem_size.max(1).hash(&mut low);
    full_mip_count(pmrem_size).hash(&mut low);
    variant.hash(&mut low);
    variant.sample_count().hash(&mut low);
    ibl_shader_identity.hash(&mut low);

    CacheKey::new((u128::from(high.finish()) << 64) | u128::from(low.finish()))
}

pub struct IblGraphOutput {
    pub updated_pmrem: TextureNodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IblComputePipelineL1Key {
    shader_identity: u64,
    sample_count: u32,
}

#[derive(Debug, Clone, Copy)]
struct IblComputePipelineState {
    id: ComputePipelineId,
    l1_key: IblComputePipelineL1Key,
}

/// Cross-frame static resources for the PMREM writer.
///
/// Every field is lazily initialized by a live node's `prepare`; constructing
/// this feature performs no WGPU work.
pub struct IblComputeFeature {
    static_pipeline: Option<IblComputePipelineState>,
    dynamic_pipeline: Option<IblComputePipelineState>,
    source_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    dest_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl Default for IblComputeFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl IblComputeFeature {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            static_pipeline: None,
            dynamic_pipeline: None,
            source_layout: None,
            dest_layout: None,
        }
    }

    fn ensure_resources(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        variant: IblPipelineVariant,
        expected_shader_identity: u64,
    ) {
        if self.source_layout.is_none() {
            self.source_layout = Some(Tracked::new(ctx.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("IBL Source BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::Cube,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: true,
                                min_binding_size: wgpu::BufferSize::new(PMREM_PARAMS_SIZE),
                            },
                            count: None,
                        },
                    ],
                },
            )));
        }

        if self.dest_layout.is_none() {
            self.dest_layout = Some(Tracked::new(ctx.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("IBL Dest BGL"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                        },
                        count: None,
                    }],
                },
            )));
        }

        let expected_l1_key = IblComputePipelineL1Key {
            shader_identity: expected_shader_identity,
            sample_count: variant.sample_count(),
        };
        let pipeline_is_current = match variant {
            IblPipelineVariant::Static => self
                .static_pipeline
                .is_some_and(|state| state.l1_key == expected_l1_key),
            IblPipelineVariant::Dynamic => self
                .dynamic_pipeline
                .is_some_and(|state| state.l1_key == expected_l1_key),
        };
        if pipeline_is_current {
            return;
        }

        let shader_options = ShaderCompilationOptions::default();
        assert_eq!(
            ctx.shader_manager
                .pipeline_shader_hash(ShaderSource::File("entry/utility/ibl"), &shader_options,),
            expected_shader_identity,
            "environment cache key and IBL pipeline shader identity diverged",
        );
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/ibl"),
            &shader_options,
        );
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("IBL Compute PL"),
                bind_group_layouts: &[self.source_layout.as_deref(), self.dest_layout.as_deref()],
                immediate_size: 0,
            });
        let constants = [("SAMPLE_COUNT", f64::from(variant.sample_count()))];
        let compilation_options = wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        };
        let pipeline = ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            module,
            &pipeline_layout,
            &ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            match variant {
                IblPipelineVariant::Static => "IBL Compute Pipeline (Static)",
                IblPipelineVariant::Dynamic => "IBL Compute Pipeline (Dynamic)",
            },
        );

        let state = IblComputePipelineState {
            id: pipeline,
            l1_key: expected_l1_key,
        };
        match variant {
            IblPipelineVariant::Static => self.static_pipeline = Some(state),
            IblPipelineVariant::Dynamic => self.dynamic_pipeline = Some(state),
        }
    }

    #[inline]
    #[must_use]
    fn pipeline(&self, variant: IblPipelineVariant) -> Option<ComputePipelineId> {
        match variant {
            IblPipelineVariant::Static => self.static_pipeline.map(|state| state.id),
            IblPipelineVariant::Dynamic => self.dynamic_pipeline.map(|state| state.id),
        }
    }

    /// Add the PMREM writer. Callers must invoke this only for an invalid plan
    /// that carries a commit token and register that token separately.
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: EnvironmentCacheSnapshot,
        base_cube: TextureNodeId,
        pmrem: TextureNodeId,
    ) -> IblGraphOutput {
        assert!(
            snapshot.needs_pmrem_write(),
            "IBL writer requires an invalid PMREM cache plan"
        );
        let params_size = snapshot
            .params_stride
            .checked_mul(u64::from(snapshot.pmrem_mip_level_count))
            .expect("IBL packed parameter buffer size overflow");

        ctx.graph.add_pass("IBL_Compute", |builder| {
            builder.read_texture(base_cube);
            builder.write_texture(pmrem);
            let params_buffer = builder.create_buffer(
                "IBL_Params_Packed",
                BufferDesc::new(
                    params_size,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
            );

            let mip_states: Vec<IblMipState<'a>> = (0..snapshot.pmrem_mip_level_count)
                .map(|mip_level| {
                    let dynamic_offset = snapshot
                        .params_stride
                        .checked_mul(u64::from(mip_level))
                        .and_then(|offset| u32::try_from(offset).ok())
                        .expect("IBL dynamic uniform offset exceeds u32");
                    IblMipState {
                        mip_level,
                        dynamic_offset,
                        dest_bg: None,
                    }
                })
                .collect();
            let mip_states = builder.graph.alloc_slice_mut(&mip_states);

            let node = IblComputePassNode {
                feature: self,
                snapshot,
                base_cube,
                pmrem,
                params_buffer,
                source_bg: None,
                mips: mip_states,
            };
            (node, ())
        });

        IblGraphOutput {
            updated_pmrem: pmrem,
        }
    }
}

#[derive(Clone, Copy)]
struct IblMipState<'a> {
    mip_level: u32,
    dynamic_offset: u32,
    dest_bg: Option<&'a wgpu::BindGroup>,
}

struct IblComputePassNode<'a> {
    feature: &'a mut IblComputeFeature,
    snapshot: EnvironmentCacheSnapshot,
    base_cube: TextureNodeId,
    pmrem: TextureNodeId,
    params_buffer: BufferNodeId,
    source_bg: Option<&'a wgpu::BindGroup>,
    mips: &'a mut [IblMipState<'a>],
}

impl<'a> PassNode<'a> for IblComputePassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.feature.ensure_resources(
            ctx,
            self.snapshot.ibl_variant,
            self.snapshot.ibl_shader_identity,
        );

        let params_size = self
            .snapshot
            .params_stride
            .checked_mul(self.mips.len() as u64)
            .expect("IBL packed parameter buffer size overflow");
        let params_size = usize::try_from(params_size)
            .expect("IBL packed parameter buffer does not fit address space");
        let mut packed_params = vec![0_u8; params_size];
        let roughness_denominator = self.mips.len().saturating_sub(1).max(1) as f32;
        for mip in self.mips.iter() {
            let mip_size = (self.snapshot.pmrem_size >> mip.mip_level).max(1);
            let params = [
                mip.mip_level as f32 / roughness_denominator,
                mip_size as f32,
                0.0,
                0.0,
            ];
            let start = mip.dynamic_offset as usize;
            let end = start + PMREM_PARAMS_SIZE as usize;
            packed_params[start..end].copy_from_slice(bytemuck::bytes_of(&params));
        }
        ctx.queue
            .write_buffer(ctx.views.get_buffer(self.params_buffer), 0, &packed_params);

        let source_layout = self
            .feature
            .source_layout
            .as_ref()
            .expect("IBL source layout must exist after prepare");
        self.source_bg = Some(
            ctx.build_bind_group(source_layout, Some("IBL Source BG"))
                .bind_texture(0, self.base_cube)
                .bind_common_sampler(1, CommonSampler::LinearClamp)
                .bind_buffer_range(
                    2,
                    self.params_buffer,
                    0,
                    wgpu::BufferSize::new(PMREM_PARAMS_SIZE)
                        .expect("IBL params binding size must be non-zero"),
                )
                .build(),
        );

        let dest_layout = self
            .feature
            .dest_layout
            .as_ref()
            .expect("IBL destination layout must exist after prepare");
        for mip in self.mips.iter_mut() {
            let mip_view = Tracked::new(ctx.views.get_texture(self.pmrem).create_view(
                &wgpu::TextureViewDescriptor {
                    label: Some("IBL PMREM Storage Mip"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: mip.mip_level,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                },
            ));
            mip.dest_bg = Some(
                ctx.build_bind_group(dest_layout, Some("IBL Dest BG"))
                    .bind_tracked_texture_view(0, &mip_view)
                    .build(),
            );
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let pipeline_id = self
            .feature
            .pipeline(self.snapshot.ibl_variant)
            .expect("IBL pipeline must exist after prepare");
        let pipeline = ctx.pipeline_cache.get_compute_pipeline(pipeline_id);
        let source_bg = self.source_bg.expect("IBL source BG missing");
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("IBL Compute"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);

        for mip in self.mips.iter() {
            cpass.set_bind_group(0, source_bg, &[mip.dynamic_offset]);
            cpass.set_bind_group(1, mip.dest_bg.expect("IBL destination BG missing"), &[]);
            let mip_size = (self.snapshot.pmrem_size >> mip.mip_level).max(1);
            let group_count = mip_size.div_ceil(8);
            cpass.dispatch_workgroups(group_count, group_count, 6);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn procedural_environment(
        base_size: u32,
        pmrem_size: u32,
        source_version: u64,
    ) -> GpuEnvironment {
        GpuEnvironment {
            base_cube_size: base_size,
            pmrem_size,
            source_version,
            source_type: CubeSourceType::Procedural,
            source_key: None,
            source_ready: true,
            procedural_physics_hash: 23,
            procedural_static_bake_hash: 29,
            last_used_frame: 1,
        }
    }

    fn texture_environment(source_version: u64, source_ready: bool) -> GpuEnvironment {
        GpuEnvironment {
            base_cube_size: 256,
            pmrem_size: 128,
            source_version,
            source_type: CubeSourceType::Equirectangular,
            source_key: None,
            source_ready,
            procedural_physics_hash: 0,
            procedural_static_bake_hash: 0,
            last_used_frame: 1,
        }
    }

    fn publish(registry: &mut HistoryRegistry, plan: CachePlan<1>) -> CachePublicationReceipt {
        registry
            .commit_cache(
                plan.commit
                    .expect("test cache plan must require publication"),
                &[plan.textures[0].id],
            )
            .expect("test cache publication must succeed")
    }

    #[test]
    fn pmrem_resize_does_not_change_base_key() {
        let mut registry = HistoryRegistry::new();
        let mut feature = EnvironmentCacheFeature::new();
        let first = feature.extract(
            &mut registry,
            1,
            &procedural_environment(256, 128, 17),
            1,
            101,
            202,
            Some(Vec3::Y),
            Some(-Vec3::Y),
            256,
        );
        let second = feature.extract(
            &mut registry,
            1,
            &procedural_environment(256, 64, 17),
            2,
            101,
            202,
            Some(Vec3::Y),
            Some(-Vec3::Y),
            256,
        );

        assert_eq!(first.base_key, second.base_key);
        assert_ne!(first.pmrem_key, second.pmrem_key);
    }

    #[test]
    fn procedural_celestial_throttle_uses_only_matching_published_static_state() {
        let mut registry = HistoryRegistry::new();
        let mut feature = EnvironmentCacheFeature::new();
        let initial = feature.extract(
            &mut registry,
            7,
            &procedural_environment(256, 128, 100),
            1,
            101,
            202,
            Some(Vec3::Z),
            Some(-Vec3::Z),
            256,
        );
        let base_receipt = publish(&mut registry, initial.base_plan.unwrap());
        feature.observe_publications(&[base_receipt]);

        let near_sun = glam::Quat::from_rotation_y(0.25_f32.to_radians()) * Vec3::Z;
        let throttled = feature.extract(
            &mut registry,
            7,
            &procedural_environment(256, 128, 101),
            2,
            101,
            202,
            Some(near_sun),
            Some(-Vec3::Z),
            256,
        );
        assert_eq!(initial.base_key, throttled.base_key);
        assert!(throttled.base_plan.unwrap().valid);

        let mut changed_static = procedural_environment(256, 128, 102);
        changed_static.procedural_physics_hash += 1;
        let invalidated = feature.extract(
            &mut registry,
            7,
            &changed_static,
            3,
            101,
            202,
            Some(near_sun),
            Some(-Vec3::Z),
            256,
        );
        assert_ne!(throttled.base_key, invalidated.base_key);
        assert!(invalidated.base_plan.unwrap().commit.is_some());
    }

    #[test]
    fn procedural_day_night_motion_throttles_both_sun_and_moon() {
        let mut registry = HistoryRegistry::new();
        let mut feature = EnvironmentCacheFeature::new();
        let initial_sun = Vec3::Z;
        let initial_moon = -Vec3::Z;
        let initial = feature.extract(
            &mut registry,
            8,
            &procedural_environment(256, 128, 100),
            1,
            101,
            202,
            Some(initial_sun),
            Some(initial_moon),
            256,
        );
        let base_receipt = publish(&mut registry, initial.base_plan.unwrap());
        feature.observe_publications(&[base_receipt]);

        let first_small_step = glam::Quat::from_rotation_y(0.4_f32.to_radians());
        let throttled = feature.extract(
            &mut registry,
            8,
            &procedural_environment(256, 128, 101),
            2,
            101,
            202,
            Some(first_small_step * initial_sun),
            Some(first_small_step * initial_moon),
            256,
        );
        assert_eq!(initial.base_key, throttled.base_key);
        assert!(throttled.base_plan.unwrap().valid);

        // Each incremental step is only 0.4 degrees, but the second state is
        // 0.8 degrees from the last successfully published baseline. This
        // catches an accidental rolling baseline that would never accumulate.
        let second_small_step = glam::Quat::from_rotation_y(0.8_f32.to_radians());
        let invalidated = feature.extract(
            &mut registry,
            8,
            &procedural_environment(256, 128, 102),
            3,
            101,
            202,
            Some(second_small_step * initial_sun),
            Some(second_small_step * initial_moon),
            256,
        );
        assert_ne!(throttled.base_key, invalidated.base_key);
        assert!(invalidated.base_plan.unwrap().commit.is_some());
    }

    #[test]
    fn culled_celestial_replacement_does_not_advance_the_published_baseline() {
        let mut registry = HistoryRegistry::new();
        let mut feature = EnvironmentCacheFeature::new();
        let initial_sun = Vec3::Z;
        let initial_moon = -Vec3::Z;
        let initial = feature.extract(
            &mut registry,
            10,
            &procedural_environment(256, 128, 100),
            1,
            101,
            202,
            Some(initial_sun),
            Some(initial_moon),
            256,
        );
        let base_receipt = publish(&mut registry, initial.base_plan.unwrap());
        feature.observe_publications(&[base_receipt]);

        let culled_step = glam::Quat::from_rotation_y(0.75_f32.to_radians());
        let culled = feature.extract(
            &mut registry,
            10,
            &procedural_environment(256, 128, 101),
            2,
            101,
            202,
            Some(culled_step * initial_sun),
            Some(culled_step * initial_moon),
            256,
        );
        let culled_token = culled
            .base_plan
            .and_then(|plan| plan.commit)
            .expect("over-threshold replacement needs a base token");
        assert!(registry.rollback_cache(culled_token));

        // This is only 0.25 degrees past the culled attempt, but a full degree
        // from the last submitted publication. Without a receipt for the
        // culled attempt, it must still schedule a replacement.
        let next_step = glam::Quat::from_rotation_y(1.0_f32.to_radians());
        let next = feature.extract(
            &mut registry,
            10,
            &procedural_environment(256, 128, 102),
            3,
            101,
            202,
            Some(next_step * initial_sun),
            Some(next_step * initial_moon),
            256,
        );
        assert_ne!(initial.base_key, next.base_key);
        assert!(next.base_plan.unwrap().commit.is_some());
    }

    #[test]
    fn pending_source_keeps_still_valid_published_environment_textures() {
        let mut registry = HistoryRegistry::new();
        let mut feature = EnvironmentCacheFeature::new();
        let ready = feature.extract(
            &mut registry,
            9,
            &procedural_environment(256, 128, 100),
            1,
            101,
            202,
            Some(Vec3::Y),
            Some(-Vec3::Y),
            256,
        );
        let receipts = [
            publish(&mut registry, ready.base_plan.unwrap()),
            publish(&mut registry, ready.pmrem_plan.unwrap()),
        ];
        feature.observe_publications(&receipts);

        let mut pending = procedural_environment(256, 64, 200);
        pending.source_type = CubeSourceType::Equirectangular;
        pending.source_ready = false;
        let fallback = feature.extract(&mut registry, 9, &pending, 2, 303, 202, None, None, 256);

        assert!(fallback.base_plan.is_some_and(|plan| plan.valid));
        assert!(fallback.pmrem_plan.is_some_and(|plan| plan.valid));
        assert!(!fallback.needs_pmrem_write());
        assert_eq!(fallback.base_texture(), ready.base_texture());
        assert_eq!(fallback.pmrem_texture(), ready.pmrem_texture());
    }

    #[test]
    fn culled_replacement_does_not_erase_pending_fallback() {
        let mut registry = HistoryRegistry::new();
        let mut feature = EnvironmentCacheFeature::new();
        let published = feature.extract(
            &mut registry,
            11,
            &texture_environment(100, true),
            1,
            101,
            202,
            None,
            None,
            256,
        );
        let receipts = [
            publish(&mut registry, published.base_plan.unwrap()),
            publish(&mut registry, published.pmrem_plan.unwrap()),
        ];
        feature.observe_publications(&receipts);

        let replacement = feature.extract(
            &mut registry,
            11,
            &texture_environment(200, true),
            2,
            101,
            202,
            None,
            None,
            256,
        );
        assert!(
            registry.rollback_cache(
                replacement
                    .base_plan
                    .and_then(|plan| plan.commit)
                    .expect("replacement base needs a token")
            )
        );
        assert!(
            registry.rollback_cache(
                replacement
                    .pmrem_plan
                    .and_then(|plan| plan.commit)
                    .expect("replacement PMREM needs a token")
            )
        );

        let fallback = feature.extract(
            &mut registry,
            11,
            &texture_environment(200, false),
            3,
            101,
            202,
            None,
            None,
            256,
        );
        assert_eq!(fallback.base_texture(), published.base_texture());
        assert_eq!(fallback.pmrem_texture(), published.pmrem_texture());
        assert!(fallback.base_plan.is_some_and(|plan| plan.valid));
        assert!(fallback.pmrem_plan.is_some_and(|plan| plan.valid));
    }

    #[test]
    fn packed_params_stride_respects_device_alignment() {
        assert_eq!(align_up(PMREM_PARAMS_SIZE, 256), 256);
        assert_eq!(align_up(PMREM_PARAMS_SIZE, PMREM_PARAMS_SIZE), 16);
    }
}
