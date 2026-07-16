//! Frame Composer
//!
//! `FrameComposer` orchestrates the entire rendering pipeline for a single
//! frame using the Declarative Render Graph (RDG). All GPU work — compute
//! pre-processing, shadow mapping, scene rendering, post-processing, and
//! custom user hooks — flows through a single unified RDG.
//!
//! # Resource Ownership (Explicit Wiring)
//!
//! The Composer registers only the **external** `Surface_Out` resource and
//! the routing-level `LDR_Intermediate`.  All scene-level transient resources
//! (`Scene_Color_HDR`, `Scene_Depth`, MSAA intermediates, specular MRT, etc.)
//! are created by their **producer passes** inside `add_to_graph()`.  Each
//! pass returns typed output structs (`PrepassOutputs`, `OpaqueOutputs`, …)
//! carrying `TextureNodeId` values that the Composer threads to downstream
//! consumers — no blackboard lookups remain for mutable resources.
//!
//! # Rendering Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                    Unified RDG Pipeline                        │
//! │                                                                │
//! │  HighFidelity:                                                 │
//! │  IBL → Shadow → Prepass → SSAO → Opaque →                    │
//! │  SSSS → Skybox → TransmissionCopy → Transparent →             │
//! │  [Bloom_System: Extract → DS_1..N → US_N..0 → Composite] →   │
//! │  ToneMap → FXAA → [User Hooks] → Surface                     │
//! │                                                                │
//! │  BasicForward:                                                 │
//! │  IBL → Shadow → Skybox(prepare) →                             │
//! │  SimpleForward → [User Hooks] → Surface                       │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! renderer.begin_frame(scene, &camera, assets, time)?
//!     .add_custom_pass(HookStage::AfterPostProcess, |rdg, bb| {
//!         let new_surface = rdg.add_pass("UI_Pass", |builder| {
//!             let out = builder.mutate_texture(bb.surface_out, "Surface_With_UI");
//!             (ui_node, out)
//!         });
//!         GraphBlackboard { surface_out: new_surface, ..bb }
//!     })
//!     .render();
//! ```

use crate::core::binding::GlobalBindGroupCache;
use crate::core::gpu::{
    AssetTextureSnapshot, BufferUploadContext, CubeSourceType, GlobalStateKey,
    GlobalStatePhysicalFailure, GlobalStateResolver, MipmapRequestResolver, Tracked,
};
use crate::core::view::ViewTarget;
use crate::core::{GpuDrivenCapabilities, ResourceManager, WgpuContext};
use crate::graph::ExtractedScene;
use crate::graph::RdgSurfaceAcquireFailure;
use crate::graph::RenderState;
use crate::graph::core::ClusteredScreenBindings;
use crate::graph::core::GraphStorage;
#[cfg(feature = "rdg_bench_support")]
use crate::graph::core::RdgDetailPhase;
#[cfg(feature = "3dgs")]
use crate::graph::core::gaussian::GaussianFrameResolver;
use crate::graph::core::graph::{
    FrameConfig, PersistentBufferGraphMaterializationFailure, PersistentBufferUploadPreparer,
};
use crate::graph::core::scene_submission::{
    SceneSubmissionKey, SceneSubmissionPlanId, SceneSubmissionResolver,
};
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, FrameArena, GlobalStateMaterializationFailure,
    GlobalStateSources, GraphBlackboard, HistoryRegistry, HookStage, PersistentBufferRegistry,
    PersistentBufferResolver, PrepareContext, RdgCommandEncodingFailure, RdgComposerAttemptGuard,
    RdgDiagnosticsRecorder, RdgFramePhase, RdgGpuDrivenFailure, RdgGpuDrivenPolicySnapshot,
    RdgPostSubmitPublicationFailure, RdgRenderTargetKind, RdgRenderWorldRebuildReason,
    RdgResourceMaterializationFailure, RdgSceneSubmissionBackend, RdgSceneSubmissionFamily,
    RdgSceneSubmissionSelectionReason, RenderGraph, SceneBufferMaterializationFailure,
    SceneBufferSources, SceneDrawPrepareContext, TextureDesc, TransientPool, ViewResolver,
};
use crate::graph::extracted::SceneFeatures;
use crate::graph::frame::{RenderLists, ShadowLightInstance};
use crate::graph::gpu_driven_policy::{
    GpuDrivenAutoDecision, GpuDrivenAutoDecisionReason, GpuDrivenAutoPolicy, GpuDrivenCoarseGate,
    GpuDrivenExactWorkload, GpuDrivenFamilyMask, GpuDrivenPolicyDomainKey,
};
use crate::graph::gpu_scene::{
    GPU_SCENE_INSTANCE_CASTS_SHADOW, GPU_SCENE_INSTANCE_MAIN_PREPASS, GPU_SCENE_INSTANCE_SHADOW_2D,
    GPU_SCENE_INSTANCE_SHADOW_CUBE, GpuScene, GpuSceneMultiBinCommitFailure,
    GpuSceneMultiBinPreflightFailure, GpuSceneMultiBinProjection, GpuSceneSelectionRejection,
};
use crate::graph::gpu_scene_submission::{
    GpuSceneMultiBinFamilyPreparePlan, GpuSceneRenderFeature, GpuSceneSubmissionFinalizeFailure,
    GpuSceneSubmissionTarget, finalize_gpu_scene_multi_bin_family,
};
#[cfg(feature = "3dgs")]
use crate::graph::passes::GaussianSplattingFeature;
use crate::graph::passes::gpu_scene::{
    GPU_SCENE_CULL_WORKGROUP_SIZE, GPU_SCENE_INDEXED_ARGUMENT_SIZE,
    GPU_SCENE_VISIBLE_INSTANCE_SIZE, GpuSceneCullFeature, GpuSceneCullViewDescriptor,
    GpuSceneMultiBinCullDescriptor, GpuSceneTableSetPatchOutput, add_gpu_scene_table_set_patch,
};
use crate::graph::passes::ibl_compute::EnvironmentCacheSnapshot;
use crate::graph::passes::shadow::ShadowSubmissions;
use crate::graph::passes::skybox::{ProceduralSkyboxInputs, TextureSkyboxInput};
use crate::graph::passes::utils::add_msaa_resolve_pass;
use crate::graph::passes::{
    AtmosphereFeature, BloomFeature, CasFeature, ClusteredLightingFeature, ClusteredLightingInputs,
    EnvironmentCacheFeature, EquirectToCubeFeature, FxaaFeature, HiZFeature, IblComputeFeature,
    MipmapFeature, MsaaSyncFeature, OpaqueFeature, PrepassFeature, PresentFeature, ShadowFeature,
    SimpleForwardFeature, SkyboxFeature, SsaoFeature, SsgiFeature, SsrFeature, SsssFeature,
    TaaFeature, ToneMappingFeature, TransmissionCopyFeature, TransparentFeature,
};
use crate::pipeline::{PipelineCache, ShaderCompilationOptions, ShaderManager, ShaderSource};
use crate::renderer::FrameTime;
use crate::settings::{GpuDrivenMode, RendererSettings};
use myth_assets::AssetServer;
use myth_resources::texture::TextureSource;
use myth_scene::Scene;
use myth_scene::background::BackgroundMapping;
use myth_scene::camera::RenderCamera;
use myth_scene::light::LightKind;
use smallvec::SmallVec;

fn present_then_reconfigure_if_needed<T>(
    output: T,
    reconfigure_after_present: bool,
    present: impl FnOnce(T),
    reconfigure: impl FnOnce(),
) {
    present(output);
    if reconfigure_after_present {
        reconfigure();
    }
}

pub struct ComposerContext<'a> {
    pub wgpu_ctx: &'a mut WgpuContext,
    pub resource_manager: &'a mut ResourceManager,
    pub pipeline_cache: &'a mut PipelineCache,
    pub shader_manager: &'a mut ShaderManager,

    pub extracted_scene: &'a mut ExtractedScene,
    pub render_state: &'a RenderState,
    pub renderer_settings: &'a RendererSettings,
    pub clustered_lighting_enabled: bool,

    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,

    /// CPU-only per-view draw intents and active render views.
    pub render_lists: &'a mut RenderLists,

    // External scene data
    pub scene: &'a mut Scene,
    pub camera: RenderCamera,
    pub assets: &'a AssetServer,
    pub frame_time: FrameTime,

    pub graph_storage: &'a mut GraphStorage,
    pub(crate) rdg_diagnostics: &'a RdgDiagnosticsRecorder,
    #[cfg(feature = "rdg_bench_support")]
    pub(crate) rdg_gpu_timings: &'a mut crate::graph::core::RdgGpuTimingRecorder,
    pub transient_pool: &'a mut TransientPool,
    pub history_registry: &'a mut HistoryRegistry,
    pub(crate) persistent_buffer_registry: &'a mut PersistentBufferRegistry,
    pub(crate) gpu_scene: &'a mut GpuScene,
    pub(crate) gpu_scene_cull: &'a mut GpuSceneCullFeature,
    pub(crate) gpu_scene_render: &'a mut GpuSceneRenderFeature,
    pub(crate) gpu_driven_policy: &'a mut GpuDrivenAutoPolicy,
    // pub sampler_registry: &'a mut SamplerRegistry,
    pub frame_arena: &'a FrameArena,

    // ─── RDG Features ────────────────────────────────────────────────────
    // Post-processing
    pub fxaa_pass: &'a mut FxaaFeature,
    pub taa_pass: &'a mut TaaFeature,
    pub cas_pass: &'a mut CasFeature,
    pub tone_map_pass: &'a mut ToneMappingFeature,
    pub present_pass: &'a mut PresentFeature,
    pub bloom_pass: &'a mut BloomFeature,
    pub ssao_pass: &'a mut SsaoFeature,
    pub hiz_pass: &'a mut HiZFeature,
    pub ssgi_pass: &'a mut SsgiFeature,
    pub ssr_pass: &'a mut SsrFeature,
    // Scene rendering
    pub prepass: &'a mut PrepassFeature,
    pub opaque_pass: &'a mut OpaqueFeature,
    pub skybox_pass: &'a mut SkyboxFeature,
    pub transparent_pass: &'a mut TransparentFeature,
    pub transmission_copy_pass: &'a mut TransmissionCopyFeature,
    pub simple_forward_pass: &'a mut SimpleForwardFeature,
    pub ssss_pass: &'a mut SsssFeature,
    pub msaa_sync_pass: &'a mut MsaaSyncFeature,

    // Shadow + Compute
    pub shadow_pass: &'a mut ShadowFeature,
    pub environment_cache: &'a mut EnvironmentCacheFeature,
    pub equirect_to_cube_pass: &'a mut EquirectToCubeFeature,
    pub ibl_pass: &'a mut IblComputeFeature,
    pub atmosphere_pass: &'a mut AtmosphereFeature,
    pub clustered_lighting_pass: &'a mut ClusteredLightingFeature,
    pub mipmap_pass: &'a mut MipmapFeature,

    #[cfg(feature = "3dgs")]
    // Gaussian Splatting
    pub gaussian_splatting_pass: &'a mut GaussianSplattingFeature,

    // Debug view (compile-time gated)
    #[cfg(feature = "debug_view")]
    pub debug_view_pass: &'a mut crate::graph::passes::DebugViewFeature,
}

pub struct GraphBuilderContext<'a, 'g> {
    pub graph: &'g mut RenderGraph<'a>,
    pub frame_config: &'g FrameConfig,
    scene_local_light_count: u32,
}

impl GraphBuilderContext<'_, '_> {
    #[inline]
    pub(crate) fn new<'a, 'g>(
        graph: &'g mut RenderGraph<'a>,
        frame_config: &'g FrameConfig,
        scene_local_light_count: u32,
    ) -> GraphBuilderContext<'a, 'g> {
        GraphBuilderContext {
            graph,
            frame_config,
            scene_local_light_count,
        }
    }

    #[cfg(feature = "rdg_inspector")]
    pub fn with_group<F, R>(&mut self, group_name: &'static str, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.graph.push_group(group_name);
        let result = f(self);
        self.graph.pop_group();
        result
    }

    /// Zero-cost fallback when the inspector is disabled.
    #[cfg(not(feature = "rdg_inspector"))]
    #[inline]
    pub fn with_group<F, R>(&mut self, _group_name: &'static str, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        f(self)
    }

    /// Returns the logical descriptor of an RDG buffer visible to the current
    /// graph build. User hooks can use this to size derived transient buffers
    /// without reaching into renderer internals.
    #[inline]
    #[must_use]
    pub fn buffer_desc(&self, id: BufferNodeId) -> BufferDesc {
        self.graph.storage.resources[id.index() as usize].buffer_desc()
    }

    /// Returns the extracted CPU local-light count for the current frame.
    /// This is the authoritative emptiness check for local-light hooks; the
    /// imported storage buffer may still be a one-slot dummy allocation.
    #[inline]
    #[must_use]
    pub const fn scene_local_light_count(&self) -> u32 {
        self.scene_local_light_count
    }
}

#[derive(Debug, Clone, Copy)]
enum TextureSkyboxSourcePlan {
    Asset {
        owner_plan: crate::graph::asset_texture_batch::AssetTextureBatchId,
        snapshot: AssetTextureSnapshot,
    },
    Attachment {
        source: TextureSource,
        logical_version: u64,
    },
}

#[derive(Debug, Clone, Copy)]
enum TextureSkyboxSourceRequest {
    Asset(crate::graph::asset_texture_batch::AssetTextureBatchId),
    Attachment {
        source: TextureSource,
        logical_version: u64,
    },
}

#[inline]
fn skybox_mapping_accepts(
    mapping: BackgroundMapping,
    view_dimension: wgpu::TextureViewDimension,
) -> bool {
    matches!(
        (mapping, view_dimension),
        (BackgroundMapping::Cube, wgpu::TextureViewDimension::Cube)
            | (
                BackgroundMapping::Equirectangular | BackgroundMapping::Planar,
                wgpu::TextureViewDimension::D2
            )
    )
}

#[inline]
fn asset_skybox_format_supported(device: &wgpu::Device, snapshot: AssetTextureSnapshot) -> bool {
    let features = snapshot
        .format
        .guaranteed_format_features(device.features());
    features
        .flags
        .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
        && features.allowed_usages.contains(snapshot.usage)
}

#[derive(Clone, Copy)]
pub struct GpuLightBuffers {
    pub light_metadata: BufferNodeId,
    pub light_storage: BufferNodeId,
    pub indirect_count_buffer: Option<BufferNodeId>,
}

fn import_scene_lighting(
    ctx: &mut GraphBuilderContext<'_, '_>,
    extracted_scene: &ExtractedScene,
) -> GpuLightBuffers {
    GpuLightBuffers {
        light_metadata: ctx.graph.import_scene_light_metadata_buffer(
            "Scene_Local_Light_Metadata",
            &extracted_scene.local_light_metadata_buffer,
        ),
        light_storage: ctx.graph.import_scene_light_storage_buffer(
            "Scene_Local_Lights",
            &extracted_scene.local_light_storage_buffer,
        ),
        indirect_count_buffer: None,
    }
}

/// Frame Composer
///
/// Holds all context references needed to render a single frame and provides
/// a fluent API for injecting custom RDG passes via hooks.
///
pub struct FrameComposer<'a> {
    ctx: ComposerContext<'a>,
    frame_config: FrameConfig,
    #[allow(clippy::type_complexity)]
    gpu_local_light_hook: Option<
        Box<dyn for<'g> FnMut(&mut GraphBuilderContext<'a, 'g>) -> Option<GpuLightBuffers> + 'a>,
    >,
    #[allow(clippy::type_complexity)]
    hooks: smallvec::SmallVec<
        [(
            HookStage,
            Option<Box<dyn FnOnce(&mut RenderGraph<'a>, GraphBlackboard) -> GraphBlackboard + 'a>>,
        ); 4],
    >,
    #[cfg(test)]
    forced_surface_acquisition_failure: Option<RdgSurfaceAcquireFailure>,
    // Declared last so hook captures and every other Composer field are
    // released before the terminal attempt snapshot is published.
    _attempt_guard: RdgComposerAttemptGuard<'a>,
}

fn classify_surface_acquisition_failure(
    result: &wgpu::CurrentSurfaceTexture,
) -> Option<RdgSurfaceAcquireFailure> {
    match result {
        wgpu::CurrentSurfaceTexture::Timeout => Some(RdgSurfaceAcquireFailure::Timeout),
        wgpu::CurrentSurfaceTexture::Occluded => Some(RdgSurfaceAcquireFailure::Occluded),
        wgpu::CurrentSurfaceTexture::Outdated => Some(RdgSurfaceAcquireFailure::Outdated),
        wgpu::CurrentSurfaceTexture::Lost => Some(RdgSurfaceAcquireFailure::Lost),
        wgpu::CurrentSurfaceTexture::Validation => Some(RdgSurfaceAcquireFailure::Validation),
        wgpu::CurrentSurfaceTexture::Success(_) | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
            None
        }
    }
}

#[derive(Clone, Copy)]
enum RenderTargetFormatContract {
    /// Swap-chain textures may expose the configured sRGB view over the
    /// corresponding linear physical format.
    SurfaceViewFamily,
    /// Headless textures declare no alternate view formats, so their physical
    /// format must exactly match the logical graph descriptor.
    Exact,
}

#[inline]
fn render_target_matches_desc(
    texture: &wgpu::Texture,
    desc: TextureDesc,
    format_contract: RenderTargetFormatContract,
) -> bool {
    render_target_metadata_matches_desc(
        wgpu::Extent3d {
            width: texture.width(),
            height: texture.height(),
            depth_or_array_layers: texture.depth_or_array_layers(),
        },
        texture.mip_level_count(),
        texture.sample_count(),
        texture.dimension(),
        texture.format(),
        desc,
        format_contract,
    )
}

#[inline]
fn render_target_metadata_matches_desc(
    size: wgpu::Extent3d,
    mip_level_count: u32,
    sample_count: u32,
    dimension: wgpu::TextureDimension,
    format: wgpu::TextureFormat,
    desc: TextureDesc,
    format_contract: RenderTargetFormatContract,
) -> bool {
    let format_matches = match format_contract {
        RenderTargetFormatContract::SurfaceViewFamily => {
            format.add_srgb_suffix() == desc.format.add_srgb_suffix()
        }
        RenderTargetFormatContract::Exact => format == desc.format,
    };
    size == desc.size
        && mip_level_count == desc.mip_level_count
        && sample_count == desc.sample_count
        && dimension == desc.dimension
        && format_matches
}

#[inline]
fn gpu_scene_binding_sized_buffer_supported(
    limits: &wgpu::Limits,
    size: Option<u64>,
    max_binding_size: u64,
) -> bool {
    size.is_some_and(|size| size != 0 && size <= limits.max_buffer_size && size <= max_binding_size)
}

#[inline]
fn checked_align_up_u64(value: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

/// Device-independent workload facts for one M7.4 multi-bin view family.
///
/// The four table sizes are the exact logical images requested from the
/// persistent-buffer registry. `visible_view_stride_bytes` is the authority's
/// already storage-aligned packed partition span for one view. Keeping this
/// hand-off `Copy` lets capability selection stay a pure preflight: it neither
/// mutates the GPU Scene authority nor allocates graph resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSceneMultiBinWorkloadFacts {
    pub(crate) instance_table_bytes: u64,
    pub(crate) geometry_table_bytes: u64,
    pub(crate) material_table_bytes: u64,
    pub(crate) batch_table_bytes: u64,
    pub(crate) instance_slot_span: u32,
    pub(crate) batch_slot_span: u32,
    pub(crate) active_bin_count: u32,
    pub(crate) visible_view_stride_bytes: u64,
    pub(crate) view_count: u32,
}

// These are the fixed CPU/WGSL records consumed by the M7.4 Cull layout.
// Keep the constants beside the capability contract until the production
// composer owns the descriptor and can source them from the shared ABI types.
const GPU_SCENE_MULTI_BIN_ACTIVE_BATCH_SIZE: u64 = 16;
const GPU_SCENE_MULTI_BIN_VIEW_PARAM_SIZE: u64 = 96;
const GPU_SCENE_MULTI_BIN_DOMAIN_UNIFORM_SIZE: u64 = 32;
const GPU_SCENE_MULTI_BIN_BINDING_COUNT: u32 = 7;
const GPU_SCENE_MULTI_BIN_STORAGE_BINDING_COUNT: u32 = 6;

/// Validate the complete portable M7.4 Cull/Draw resource envelope without
/// creating a buffer, node, pipeline, or authority revision.
///
/// All four persistent tables are storage-table ABIs, so each member must be
/// legal both as a physical buffer and as one storage binding. The current Cull
/// shader binds Instance and Batch directly; applying the same contract to
/// Geometry and Material preserves a coherent four-table capability tier for
/// their downstream consumers.
#[inline]
pub(crate) fn gpu_scene_multi_bin_workload_supported(
    capabilities: &GpuDrivenCapabilities,
    facts: GpuSceneMultiBinWorkloadFacts,
) -> bool {
    if !capabilities.supports_indirect_primitives()
        || facts.instance_slot_span == 0
        || facts.batch_slot_span == 0
        || facts.active_bin_count == 0
        || facts.active_bin_count > facts.batch_slot_span
        || facts.view_count == 0
        || facts.visible_view_stride_bytes == 0
        || facts.visible_view_stride_bytes > u64::from(u32::MAX)
    {
        return false;
    }

    let limits = capabilities.device_limits();
    let storage_alignment = u64::from(limits.min_storage_buffer_offset_alignment);
    if storage_alignment == 0
        || !facts
            .visible_view_stride_bytes
            .is_multiple_of(storage_alignment)
        || !facts
            .visible_view_stride_bytes
            .is_multiple_of(GPU_SCENE_VISIBLE_INSTANCE_SIZE)
    {
        return false;
    }

    // Use the same u32 index domain as WGSL before widening to bytes. This
    // rejects a view-major sparse argument index that would wrap even if its
    // eventual u64 allocation happened to fit the device limit.
    let Some(argument_record_count) = facts.batch_slot_span.checked_mul(facts.view_count) else {
        return false;
    };
    let Some(visible_bytes) = facts
        .visible_view_stride_bytes
        .checked_mul(u64::from(facts.view_count))
    else {
        return false;
    };
    let Some(argument_bytes) =
        u64::from(argument_record_count).checked_mul(GPU_SCENE_INDEXED_ARGUMENT_SIZE)
    else {
        return false;
    };
    let Some(active_batch_bytes) =
        u64::from(facts.active_bin_count).checked_mul(GPU_SCENE_MULTI_BIN_ACTIVE_BATCH_SIZE)
    else {
        return false;
    };
    let Some(view_param_bytes) =
        u64::from(facts.view_count).checked_mul(GPU_SCENE_MULTI_BIN_VIEW_PARAM_SIZE)
    else {
        return false;
    };
    let workgroups_x = facts
        .instance_slot_span
        .max(facts.active_bin_count)
        .div_ceil(GPU_SCENE_CULL_WORKGROUP_SIZE);

    let max_storage_binding = limits.max_storage_buffer_binding_size;
    let max_uniform_binding = limits.max_uniform_buffer_binding_size;
    let uniform_alignment = u64::from(limits.min_uniform_buffer_offset_alignment);
    let Some(view_param_offset) = checked_align_up_u64(active_batch_bytes, storage_alignment)
    else {
        return false;
    };
    let Some(domain_uniform_offset) = view_param_offset
        .checked_add(view_param_bytes)
        .and_then(|end| checked_align_up_u64(end, uniform_alignment))
    else {
        return false;
    };
    let Some(cull_parameter_bytes) =
        domain_uniform_offset.checked_add(GPU_SCENE_MULTI_BIN_DOMAIN_UNIFORM_SIZE)
    else {
        return false;
    };
    if cull_parameter_bytes > limits.max_buffer_size {
        return false;
    }
    let persistent_table_bytes = [
        facts.instance_table_bytes,
        facts.geometry_table_bytes,
        facts.material_table_bytes,
        facts.batch_table_bytes,
    ];
    // The coherent Patch transaction uses one concatenated mapped staging
    // buffer so its four member copies share one publication gate. Validate
    // that physical allocation here; checking each destination alone would
    // allow prepare to exceed `max_buffer_size` on a cold/full replay.
    let Some(coherent_patch_bytes) = persistent_table_bytes
        .iter()
        .try_fold(0u64, |total, &size| total.checked_add(size))
    else {
        return false;
    };
    if coherent_patch_bytes > limits.max_buffer_size {
        return false;
    }
    let transient_storage_bytes = [
        visible_bytes,
        argument_bytes,
        active_batch_bytes,
        view_param_bytes,
    ];

    // Portable multi-bin indirect records always keep `first_instance = 0`.
    // Exact visible subranges are selected through Group 2 instead, so this
    // tier deliberately has no `INDIRECT_FIRST_INSTANCE` feature dependency.
    persistent_table_bytes.into_iter().all(|size| {
        gpu_scene_binding_sized_buffer_supported(limits, Some(size), max_storage_binding)
    }) && transient_storage_bytes.into_iter().all(|size| {
        gpu_scene_binding_sized_buffer_supported(limits, Some(size), max_storage_binding)
    }) && gpu_scene_binding_sized_buffer_supported(
        limits,
        Some(GPU_SCENE_MULTI_BIN_DOMAIN_UNIFORM_SIZE),
        max_uniform_binding,
    ) && limits.max_bind_groups >= 4
        && limits.max_bindings_per_bind_group >= GPU_SCENE_MULTI_BIN_BINDING_COUNT
        && limits.max_storage_buffers_per_shader_stage >= GPU_SCENE_MULTI_BIN_STORAGE_BINDING_COUNT
        && limits.max_uniform_buffers_per_shader_stage >= 1
        && limits.max_compute_invocations_per_workgroup >= GPU_SCENE_CULL_WORKGROUP_SIZE
        && limits.max_compute_workgroup_size_x >= GPU_SCENE_CULL_WORKGROUP_SIZE
        && limits.max_compute_workgroup_size_y >= 1
        && limits.max_compute_workgroup_size_z >= 1
        && workgroups_x <= limits.max_compute_workgroups_per_dimension
        && facts.view_count <= limits.max_compute_workgroups_per_dimension
}

#[inline]
fn gpu_scene_multi_bin_family_supported(
    capabilities: &GpuDrivenCapabilities,
    projection: GpuSceneMultiBinProjection,
    view_count: usize,
) -> bool {
    if view_count == 0 {
        return true;
    }
    let Ok(view_count) = u32::try_from(view_count) else {
        return false;
    };
    let tables = projection.table_set;
    gpu_scene_multi_bin_workload_supported(
        capabilities,
        GpuSceneMultiBinWorkloadFacts {
            instance_table_bytes: tables.instance_logical_bytes,
            geometry_table_bytes: tables.geometry_logical_bytes,
            material_table_bytes: tables.material_logical_bytes,
            batch_table_bytes: tables.batch_logical_bytes,
            instance_slot_span: tables.instance_slot_span,
            batch_slot_span: tables.batch_slot_span,
            active_bin_count: projection.bin_count,
            visible_view_stride_bytes: tables.visible_view_stride,
            view_count,
        },
    )
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct GpuSceneShadowFamilyWorkload {
    d2_view_count: usize,
    cube_view_count: usize,
}

/// Count the exact packed Cull families without allocating a Shadow snapshot.
/// The point-light classification intentionally mirrors `ShadowFeature::extract`.
fn gpu_scene_shadow_family_workload(
    render_lists: &RenderLists,
    extracted_scene: &ExtractedScene,
) -> GpuSceneShadowFamilyWorkload {
    let mut workload = GpuSceneShadowFamilyWorkload::default();
    for view in &render_lists.active_views {
        let ViewTarget::ShadowLight { light_id, .. } = view.target else {
            continue;
        };
        let is_point = extracted_scene
            .lights
            .get(view.light_buffer_index)
            .is_some_and(|light| light.id == light_id && matches!(light.kind, LightKind::Point(_)));
        if is_point {
            workload.cube_view_count += 1;
        } else {
            workload.d2_view_count += 1;
        }
    }
    workload
}

#[inline]
const fn auto_selection_reason(
    reason: GpuDrivenAutoDecisionReason,
) -> RdgSceneSubmissionSelectionReason {
    match reason {
        GpuDrivenAutoDecisionReason::SmallWorkload => {
            RdgSceneSubmissionSelectionReason::AutoSmallWorkload
        }
        GpuDrivenAutoDecisionReason::CostModel => RdgSceneSubmissionSelectionReason::AutoCostModel,
        GpuDrivenAutoDecisionReason::Hysteresis => {
            RdgSceneSubmissionSelectionReason::AutoHysteresis
        }
        GpuDrivenAutoDecisionReason::LowConfidence => {
            RdgSceneSubmissionSelectionReason::AutoLowConfidence
        }
    }
}

#[inline]
fn selected_family_backend(
    selected: bool,
    selected_members: u32,
    total_members: u32,
    has_work: bool,
) -> RdgSceneSubmissionBackend {
    if !has_work {
        return RdgSceneSubmissionBackend::Inactive;
    }
    if !selected {
        return RdgSceneSubmissionBackend::CpuDirect;
    }
    if selected_members < total_members {
        RdgSceneSubmissionBackend::Hybrid
    } else {
        RdgSceneSubmissionBackend::GpuDriven
    }
}

/// Families whose complete semantic CPU domain is represented by the exact
/// retained GPU recipe. This is stronger than merely selecting a family: Auto
/// may retain only high-reuse cohorts and must keep the complement on CPU.
#[inline]
fn complete_gpu_family_ownership(
    selected_families: GpuDrivenFamilyMask,
    projection: Option<GpuSceneMultiBinProjection>,
    synopsis: crate::graph::gpu_driven_policy::GpuDrivenWorkloadSynopsis,
) -> GpuDrivenFamilyMask {
    let Some(projection) = projection else {
        return GpuDrivenFamilyMask::empty();
    };
    let mut complete = GpuDrivenFamilyMask::empty();
    if selected_families.contains(GpuDrivenFamilyMask::MAIN_PREPASS)
        && projection.candidate_count == synopsis.opaque_item_count
    {
        complete.insert(GpuDrivenFamilyMask::MAIN_PREPASS);
    }
    if projection.covers_complete_shadow_domain {
        complete |=
            selected_families & (GpuDrivenFamilyMask::SHADOW_2D | GpuDrivenFamilyMask::SHADOW_CUBE);
    }
    complete
}

fn gpu_scene_capability_families(
    capabilities: &GpuDrivenCapabilities,
    projection: GpuSceneMultiBinProjection,
    shadows: GpuSceneShadowFamilyWorkload,
) -> GpuDrivenFamilyMask {
    let mut families = GpuDrivenFamilyMask::empty();
    if gpu_scene_multi_bin_family_supported(capabilities, projection, 1) {
        families.insert(GpuDrivenFamilyMask::MAIN_PREPASS);
    }
    if shadows.d2_view_count != 0
        && gpu_scene_multi_bin_family_supported(capabilities, projection, shadows.d2_view_count)
    {
        families.insert(GpuDrivenFamilyMask::SHADOW_2D);
    }
    if shadows.cube_view_count != 0
        && gpu_scene_multi_bin_family_supported(capabilities, projection, shadows.cube_view_count)
    {
        families.insert(GpuDrivenFamilyMask::SHADOW_CUBE);
    }
    families
}

#[inline]
#[cfg(test)]
fn resolve_gpu_scene_shadow_ownership(
    mode: GpuDrivenMode,
    has_shadow_work: bool,
    covers_complete_domain: bool,
    workload_supported: bool,
) -> Result<bool, RdgGpuDrivenFailure> {
    if mode == GpuDrivenMode::Off || !has_shadow_work {
        return Ok(false);
    }
    if !covers_complete_domain {
        return if mode == GpuDrivenMode::Required {
            Err(RdgGpuDrivenFailure::UnsupportedShadowDomain)
        } else {
            Ok(false)
        };
    }
    if !workload_supported {
        return if mode == GpuDrivenMode::Required {
            Err(RdgGpuDrivenFailure::WorkloadExceedsDeviceLimits)
        } else {
            Ok(false)
        };
    }
    Ok(true)
}

const fn selection_failure_reason(reason: GpuSceneSelectionRejection) -> RdgGpuDrivenFailure {
    match reason {
        GpuSceneSelectionRejection::Skinned => RdgGpuDrivenFailure::SkinnedGeometry,
        GpuSceneSelectionRejection::MorphTargets => RdgGpuDrivenFailure::MorphTargets,
        GpuSceneSelectionRejection::InstancedVertexAttributes => {
            RdgGpuDrivenFailure::InstancedVertexAttributes
        }
        GpuSceneSelectionRejection::IncompatibleMaterial => {
            RdgGpuDrivenFailure::IncompatibleMaterial
        }
        GpuSceneSelectionRejection::NonIndexed => RdgGpuDrivenFailure::NonIndexedGeometry,
        GpuSceneSelectionRejection::EmptyIndexBuffer => RdgGpuDrivenFailure::EmptyIndexBuffer,
        GpuSceneSelectionRejection::EmptyVertexBuffers => RdgGpuDrivenFailure::EmptyVertexBuffers,
        GpuSceneSelectionRejection::UnsupportedTopology => RdgGpuDrivenFailure::UnsupportedTopology,
        GpuSceneSelectionRejection::NonFiniteTransform => RdgGpuDrivenFailure::NonFiniteTransform,
        GpuSceneSelectionRejection::NonInvertibleTransform => {
            RdgGpuDrivenFailure::NonInvertibleTransform
        }
        GpuSceneSelectionRejection::NonFiniteBounds => RdgGpuDrivenFailure::NonFiniteBounds,
        GpuSceneSelectionRejection::ConflictingGeometryEpoch => {
            RdgGpuDrivenFailure::ConflictingGeometryEpoch
        }
        GpuSceneSelectionRejection::ConflictingMaterialEpoch => {
            RdgGpuDrivenFailure::ConflictingMaterialEpoch
        }
        GpuSceneSelectionRejection::GenerationExhausted => RdgGpuDrivenFailure::GenerationExhausted,
        GpuSceneSelectionRejection::DuplicateInstance => RdgGpuDrivenFailure::DuplicateInstance,
        GpuSceneSelectionRejection::TooManyCandidates => RdgGpuDrivenFailure::TooManyCandidates,
    }
}

#[inline]
const fn multi_bin_preflight_failure_reason(
    reason: GpuSceneMultiBinPreflightFailure,
) -> RdgGpuDrivenFailure {
    match reason {
        GpuSceneMultiBinPreflightFailure::Domain(reason) => selection_failure_reason(reason),
        GpuSceneMultiBinPreflightFailure::RenderWorldDrift => {
            RdgGpuDrivenFailure::SubmissionDescriptorDrift
        }
        GpuSceneMultiBinPreflightFailure::InvalidVisibleStride
        | GpuSceneMultiBinPreflightFailure::InvalidStorageAlignment
        | GpuSceneMultiBinPreflightFailure::LayoutOverflow => {
            RdgGpuDrivenFailure::WorkloadExceedsDeviceLimits
        }
    }
}

#[inline]
fn record_gpu_scene_recipe_diagnostics(diagnostics: &RdgDiagnosticsRecorder, gpu_scene: &GpuScene) {
    let stats = gpu_scene.recipe_stats();
    diagnostics.update_render_world(|snapshot| {
        snapshot.gpu_members_reused = stats.members_reused;
        snapshot.gpu_members_examined = stats.members_examined;
        snapshot.gpu_members_updated = stats.members_updated;
        snapshot.gpu_members_rebuilt = stats.members_rebuilt;
        snapshot.retained_bytes = snapshot
            .retained_bytes
            .saturating_add(u64::try_from(stats.retained_bytes).unwrap_or(u64::MAX));
        if stats.full_rebuild_reason.is_some() && snapshot.rebuild_reason.is_none() {
            snapshot.rebuild_reason = Some(RdgRenderWorldRebuildReason::ConsumerRevisionGap);
        }
    });
}

#[inline]
const fn multi_bin_commit_failure_reason(
    reason: GpuSceneMultiBinCommitFailure,
) -> RdgGpuDrivenFailure {
    match reason {
        GpuSceneMultiBinCommitFailure::StalePreflight
        | GpuSceneMultiBinCommitFailure::SceneDrift => {
            RdgGpuDrivenFailure::SubmissionDescriptorDrift
        }
    }
}

const fn finalize_failure_reason(reason: GpuSceneSubmissionFinalizeFailure) -> RdgGpuDrivenFailure {
    match reason {
        GpuSceneSubmissionFinalizeFailure::DescriptorDrift => {
            RdgGpuDrivenFailure::SubmissionDescriptorDrift
        }
        GpuSceneSubmissionFinalizeFailure::RepresentativeDrift => {
            RdgGpuDrivenFailure::RepresentativeDrift
        }
        GpuSceneSubmissionFinalizeFailure::GeometryUnavailable => {
            RdgGpuDrivenFailure::GeometryUnavailable
        }
        GpuSceneSubmissionFinalizeFailure::GeometryDrift => RdgGpuDrivenFailure::GeometryDrift,
        GpuSceneSubmissionFinalizeFailure::MaterialUnavailable => {
            RdgGpuDrivenFailure::MaterialUnavailable
        }
        GpuSceneSubmissionFinalizeFailure::MaterialDrift => RdgGpuDrivenFailure::MaterialDrift,
        GpuSceneSubmissionFinalizeFailure::GlobalStateUnavailable => {
            RdgGpuDrivenFailure::GlobalStateUnavailable
        }
        GpuSceneSubmissionFinalizeFailure::UnsupportedMaterialState => {
            RdgGpuDrivenFailure::UnsupportedMaterialState
        }
        GpuSceneSubmissionFinalizeFailure::BatchPublicationRejected => {
            RdgGpuDrivenFailure::ImmutableBatchPublicationRejected
        }
    }
}

impl<'a> FrameComposer<'a> {
    /// Creates a new frame composer.
    pub(crate) fn new(ctx: ComposerContext<'a>, size: (u32, u32)) -> Self {
        let frame_config = FrameConfig {
            width: size.0,
            height: size.1,
            depth_format: ctx.wgpu_ctx.depth_format,
            msaa_samples: ctx.wgpu_ctx.msaa_samples,
            surface_format: ctx.wgpu_ctx.surface_view_format,
            hdr_format: crate::HDR_TEXTURE_FORMAT,
        };

        Self {
            _attempt_guard: RdgComposerAttemptGuard::new(ctx.rdg_diagnostics),
            ctx,
            frame_config,
            gpu_local_light_hook: None,
            hooks: smallvec::SmallVec::new(),
            #[cfg(test)]
            forced_surface_acquisition_failure: None,
        }
    }

    /// Test-only acquisition seam. It exercises the exact early-return path
    /// without requiring a platform window surface to fail nondeterministically.
    #[cfg(test)]
    pub(crate) fn with_forced_surface_acquisition_failure(
        mut self,
        reason: RdgSurfaceAcquireFailure,
    ) -> Self {
        self.forced_surface_acquisition_failure = Some(reason);
        self
    }

    /// Returns a reference to the wgpu device.
    #[inline]
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.ctx.wgpu_ctx.device
    }

    /// Returns a reference to the resource manager.
    ///
    /// Useful for user-land passes that need to resolve engine resources
    /// (e.g. texture handles) before the RDG prepare phase.
    #[inline]
    #[must_use]
    pub fn resource_manager(&self) -> &ResourceManager {
        self.ctx.resource_manager
    }

    /// Registers a custom pass hook that will be invoked during graph building.
    ///
    /// The closure receives a mutable reference to the [`RenderGraph`] and
    /// the [`GraphBlackboard`] containing the frame's well-known resource slots.
    /// It must return an (optionally updated) [`GraphBlackboard`] — the Rust
    /// type system enforces that every hook path returns a valid blackboard.
    ///
    /// Hooks registered with [`HookStage::AfterPostProcess`] run after all
    /// built-in post-processing (Bloom, ToneMap, FXAA) and are typically used
    /// for UI rendering.
    ///
    /// # Example
    ///
    /// ```ignore
    /// composer
    ///     .add_custom_pass(HookStage::AfterPostProcess, |rdg, bb| {
    ///         let new_surface = rdg.add_pass("MyPass", |builder| {
    ///             let out = builder.mutate_texture(bb.surface_out, "Surface_MyPass");
    ///             (MyPassNode { target: out }, out)
    ///         });
    ///         GraphBlackboard { surface_out: new_surface, ..bb }
    ///     })
    ///     .render();
    /// ```
    #[inline]
    #[must_use]
    pub fn add_custom_pass<F>(mut self, stage: HookStage, hook: F) -> Self
    where
        F: FnOnce(&mut RenderGraph<'a>, GraphBlackboard) -> GraphBlackboard + 'a,
    {
        self.hooks.push((stage, Some(Box::new(hook))));
        self
    }

    /// Registers a hook that can inject an optional GPU-generated local-light
    /// track before clustered lighting and forward shading are wired.
    #[inline]
    #[must_use]
    pub fn inject_gpu_local_lights<F>(mut self, hook: F) -> Self
    where
        F: for<'g> FnMut(&mut GraphBuilderContext<'a, 'g>) -> Option<GpuLightBuffers> + 'a,
    {
        self.gpu_local_light_hook = Some(Box::new(hook));
        self
    }

    /// Executes the full rendering pipeline.
    ///
    /// In **windowed mode** the result is presented to the swap-chain surface.
    /// In **headless mode** the result is written to the offscreen render target
    /// (available for readback via [`Renderer::readback_pixels`]).
    ///
    /// # Architecture
    ///
    /// 1. **Build Logical RDG** — collect CPU draw intents, register resources,
    ///    wire passes, and run DPE/topological compilation
    /// 2. **Acquire Render Target** — wait for the swap-chain only after the
    ///    CPU graph is ready, then late-bind its physical view
    /// 3. **Materialize & Execute** — allocate transients, prepare, record, and
    ///    submit only when target acquisition succeeded
    /// 4. **Present** — present swap-chain (windowed only)
    ///
    /// Consumes `self`; the composer cannot be reused after render.
    pub fn render(mut self) {
        let rdg_diagnostics = self.ctx.rdg_diagnostics;
        let view_format = self.ctx.wgpu_ctx.surface_view_format;

        // A missing target is known without waiting and should not spend CPU
        // building a graph. Surface availability itself is intentionally not
        // queried until logical compilation has completed.
        if self.ctx.wgpu_ctx.surface.is_none() && self.ctx.wgpu_ctx.headless_texture.is_none() {
            rdg_diagnostics.enter_phase(RdgFramePhase::AcquireTarget);
            if rdg_diagnostics.mark_no_render_target() {
                log::error!("No render target available (neither surface nor headless texture)");
            }
            return;
        }

        // ━━━ 1. Build CPU-only Logical RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        rdg_diagnostics.enter_phase(RdgFramePhase::Cull);

        // ━━━ 2a. Select backend ownership, then collect CPU intents ━━━━━━
        //
        // Reset retained intent/interner state before the pre-frustum
        // selector publishes its fixed-state representative. A selected GPU
        // batch then skips duplicate CPU opaque culling and source capture;
        // transparent and independently-fallback Shadow intents remain CPU-owned.
        self.ctx.render_lists.begin_intent_collection();

        // M7.5 keeps Off as a strict zero-planning baseline, preserves
        // Required as the all-or-nothing correctness oracle, and makes Auto a
        // submission-transactional family/cohort policy. Auto's cheap synopsis
        // may reject tiny/stable CPU workloads before O(N) preflight.
        let mode = self.ctx.renderer_settings.gpu_driven;
        let mut synopsis = self.ctx.extracted_scene.gpu_driven_workload_synopsis();
        let mut gpu_scene_preflight = None;
        let mut selection_projection = None;
        let mut selected_families = GpuDrivenFamilyMask::empty();
        let mut shadow_workload = GpuSceneShadowFamilyWorkload::default();
        let mut selection_reason = match mode {
            GpuDrivenMode::Off => RdgSceneSubmissionSelectionReason::ModeOff,
            GpuDrivenMode::Required => RdgSceneSubmissionSelectionReason::Required,
            GpuDrivenMode::Auto => RdgSceneSubmissionSelectionReason::NotEvaluated,
        };
        let mut gpu_policy_submission_token = None;
        let mut policy_minimum_cohort_members = 0;
        let mut policy_exact_evaluated = false;
        let mut gpu_recipe_evaluated = false;

        #[cfg(feature = "rdg_bench_support")]
        let backend_selection_timing =
            rdg_diagnostics.detail_timer(RdgDetailPhase::GpuBackendSelection);
        match mode {
            // No capability query, shadow scan, policy lookup, preflight, or
            // GPU Scene state access is allowed in the CpuDirect baseline.
            GpuDrivenMode::Off => {}
            GpuDrivenMode::Required => {
                debug_assert!(self.ctx.extracted_scene.gpu_driven_facts_captured());
                policy_minimum_cohort_members = 1;
                policy_exact_evaluated = true;
                let capabilities = self.ctx.wgpu_ctx.gpu_driven_capabilities();
                if !capabilities.supports_indirect_primitives() {
                    rdg_diagnostics.mark_gpu_driven_failed(RdgGpuDrivenFailure::UnsupportedDevice);
                    return;
                }
                let storage_alignment = u64::from(
                    capabilities
                        .device_limits()
                        .min_storage_buffer_offset_alignment,
                );
                gpu_recipe_evaluated = true;
                let preflight = match self.ctx.gpu_scene.preflight_multi_bin_scene(
                    self.ctx.extracted_scene,
                    GPU_SCENE_VISIBLE_INSTANCE_SIZE,
                    storage_alignment,
                ) {
                    Ok(preflight) => preflight,
                    Err(reason) => {
                        record_gpu_scene_recipe_diagnostics(rdg_diagnostics, self.ctx.gpu_scene);
                        rdg_diagnostics
                            .mark_gpu_driven_failed(multi_bin_preflight_failure_reason(reason));
                        return;
                    }
                };
                let has_shadow_casters = self.ctx.gpu_scene.preflight_has_shadow_casters();
                if has_shadow_casters {
                    shadow_workload = gpu_scene_shadow_family_workload(
                        self.ctx.render_lists,
                        self.ctx.extracted_scene,
                    );
                    synopsis.shadow_2d_view_count =
                        u32::try_from(shadow_workload.d2_view_count).unwrap_or(u32::MAX);
                    synopsis.shadow_cube_view_count =
                        u32::try_from(shadow_workload.cube_view_count).unwrap_or(u32::MAX);
                }
                let has_shadow_work =
                    shadow_workload.d2_view_count != 0 || shadow_workload.cube_view_count != 0;
                if let Some(preflight) = preflight {
                    let capability_families = gpu_scene_capability_families(
                        capabilities,
                        preflight.projection,
                        shadow_workload,
                    );
                    if !capability_families.contains(GpuDrivenFamilyMask::MAIN_PREPASS) {
                        record_gpu_scene_recipe_diagnostics(rdg_diagnostics, self.ctx.gpu_scene);
                        rdg_diagnostics.mark_gpu_driven_failed(
                            RdgGpuDrivenFailure::WorkloadExceedsDeviceLimits,
                        );
                        return;
                    }
                    selected_families.insert(GpuDrivenFamilyMask::MAIN_PREPASS);
                    if has_shadow_work && !preflight.projection.covers_complete_shadow_domain {
                        record_gpu_scene_recipe_diagnostics(rdg_diagnostics, self.ctx.gpu_scene);
                        rdg_diagnostics
                            .mark_gpu_driven_failed(RdgGpuDrivenFailure::UnsupportedShadowDomain);
                        return;
                    }
                    if shadow_workload.d2_view_count != 0 {
                        if !capability_families.contains(GpuDrivenFamilyMask::SHADOW_2D) {
                            record_gpu_scene_recipe_diagnostics(
                                rdg_diagnostics,
                                self.ctx.gpu_scene,
                            );
                            rdg_diagnostics.mark_gpu_driven_failed(
                                RdgGpuDrivenFailure::WorkloadExceedsDeviceLimits,
                            );
                            return;
                        }
                        selected_families.insert(GpuDrivenFamilyMask::SHADOW_2D);
                    }
                    if shadow_workload.cube_view_count != 0 {
                        if !capability_families.contains(GpuDrivenFamilyMask::SHADOW_CUBE) {
                            record_gpu_scene_recipe_diagnostics(
                                rdg_diagnostics,
                                self.ctx.gpu_scene,
                            );
                            rdg_diagnostics.mark_gpu_driven_failed(
                                RdgGpuDrivenFailure::WorkloadExceedsDeviceLimits,
                            );
                            return;
                        }
                        selected_families.insert(GpuDrivenFamilyMask::SHADOW_CUBE);
                    }
                    selection_projection = Some(preflight.projection);
                    gpu_scene_preflight = Some(preflight);
                } else {
                    if has_shadow_work {
                        record_gpu_scene_recipe_diagnostics(rdg_diagnostics, self.ctx.gpu_scene);
                        rdg_diagnostics
                            .mark_gpu_driven_failed(RdgGpuDrivenFailure::UnsupportedShadowDomain);
                        return;
                    }
                    // A transparent-only scene has no opaque semantic domain.
                    selection_reason = RdgSceneSubmissionSelectionReason::NoWork;
                }
            }
            GpuDrivenMode::Auto => {
                debug_assert!(self.ctx.extracted_scene.gpu_driven_facts_captured());
                let capabilities = self.ctx.wgpu_ctx.gpu_driven_capabilities();
                if capabilities.supports_indirect_primitives() {
                    if synopsis.shadow_caster_count != 0 {
                        shadow_workload = gpu_scene_shadow_family_workload(
                            self.ctx.render_lists,
                            self.ctx.extracted_scene,
                        );
                        synopsis.shadow_2d_view_count =
                            u32::try_from(shadow_workload.d2_view_count).unwrap_or(u32::MAX);
                        synopsis.shadow_cube_view_count =
                            u32::try_from(shadow_workload.cube_view_count).unwrap_or(u32::MAX);
                    }
                    let domain = GpuDrivenPolicyDomainKey {
                        scene_id: self.ctx.extracted_scene.scene_id,
                        camera_id: self.ctx.camera.camera_id,
                        render_path: self.ctx.wgpu_ctx.render_path,
                        msaa_samples: self.ctx.wgpu_ctx.msaa_samples,
                        viewport_width: self.frame_config.width,
                        viewport_height: self.frame_config.height,
                        pipeline_epoch: self.ctx.wgpu_ctx.pipeline_settings_version,
                    };
                    match self.ctx.gpu_driven_policy.coarse_gate(domain, synopsis) {
                        GpuDrivenCoarseGate::Decide(decision) => {
                            debug_assert!(decision.families.is_empty());
                            debug_assert!(decision.minimum_cohort_members >= 1);
                            selected_families = decision.families;
                            selection_reason = auto_selection_reason(decision.reason);
                            gpu_policy_submission_token = Some(decision.token);
                            policy_minimum_cohort_members = decision.minimum_cohort_members;
                        }
                        GpuDrivenCoarseGate::EvaluateExact {
                            minimum_cohort_members,
                        } => {
                            policy_exact_evaluated = true;
                            policy_minimum_cohort_members = minimum_cohort_members;
                            let storage_alignment = u64::from(
                                capabilities
                                    .device_limits()
                                    .min_storage_buffer_offset_alignment,
                            );
                            gpu_recipe_evaluated = true;
                            let (preflight, preflight_failure) =
                                match self.ctx.gpu_scene.preflight_multi_bin_cohorts(
                                    self.ctx.extracted_scene,
                                    GPU_SCENE_VISIBLE_INSTANCE_SIZE,
                                    storage_alignment,
                                    minimum_cohort_members,
                                ) {
                                    Ok(preflight) => (preflight, None),
                                    Err(reason) => {
                                        (None, Some(multi_bin_preflight_failure_reason(reason)))
                                    }
                                };
                            let exact = if let Some(preflight) = preflight {
                                let projection = preflight.projection;
                                selection_projection = Some(projection);
                                let capability_families = gpu_scene_capability_families(
                                    capabilities,
                                    projection,
                                    shadow_workload,
                                );
                                let selected_index_work = if synopsis.supported_opaque_count == 0 {
                                    0
                                } else {
                                    synopsis
                                        .indexed_primitive_count
                                        .saturating_mul(u64::from(projection.candidate_count))
                                        / u64::from(synopsis.supported_opaque_count)
                                };
                                GpuDrivenExactWorkload {
                                    candidate_count: projection.candidate_count,
                                    bin_count: projection.bin_count,
                                    unique_geometry_count: projection.unique_geometry_count,
                                    unique_material_count: projection.unique_material_count,
                                    shadow_candidate_count: projection.shadow_candidate_count,
                                    instance_slot_span: projection.table_set.instance_slot_span,
                                    batch_slot_span: projection.table_set.batch_slot_span,
                                    dirty_candidate_count: synopsis
                                        .transform_changed_count
                                        .min(projection.candidate_count),
                                    indexed_primitive_count: selected_index_work,
                                    main_submission_count: if self
                                        .ctx
                                        .wgpu_ctx
                                        .render_path
                                        .supports_post_processing()
                                    {
                                        2
                                    } else {
                                        1
                                    },
                                    shadow_2d_view_count: u32::try_from(
                                        shadow_workload.d2_view_count,
                                    )
                                    .unwrap_or(u32::MAX),
                                    shadow_cube_view_count: u32::try_from(
                                        shadow_workload.cube_view_count,
                                    )
                                    .unwrap_or(u32::MAX),
                                    viewport_pixels: u64::from(self.frame_config.width)
                                        .saturating_mul(u64::from(self.frame_config.height)),
                                    capability_families,
                                }
                            } else {
                                // No retained cohort or a typed preflight
                                // rejection has no safe GPU family. Still
                                // issue a transactional policy decision so a
                                // previously selected family is disabled only
                                // after this CPU fallback frame is submitted.
                                GpuDrivenExactWorkload::default()
                            };
                            let decision: GpuDrivenAutoDecision = self
                                .ctx
                                .gpu_driven_policy
                                .evaluate_exact(domain, synopsis, exact, minimum_cohort_members);
                            debug_assert_eq!(
                                decision.minimum_cohort_members,
                                minimum_cohort_members
                            );
                            selected_families = decision.families;
                            selection_reason = if let Some(reason) = preflight_failure {
                                debug_assert!(selected_families.is_empty());
                                RdgSceneSubmissionSelectionReason::Unsupported(reason)
                            } else {
                                auto_selection_reason(decision.reason)
                            };
                            gpu_policy_submission_token = Some(decision.token);
                            if !selected_families.is_empty() {
                                gpu_scene_preflight = preflight;
                                debug_assert!(gpu_scene_preflight.is_some());
                            }
                        }
                    }
                } else {
                    selection_reason = RdgSceneSubmissionSelectionReason::Unsupported(
                        RdgGpuDrivenFailure::UnsupportedDevice,
                    );
                }
            }
        }
        #[cfg(feature = "rdg_bench_support")]
        drop(backend_selection_timing);

        // This is the first persistent GPU Scene mutation. Main and Shadow
        // capability gates have already passed, and the retained preflight is
        // consumed exactly once into the shared four-table revision.
        let gpu_scene_projection = gpu_scene_preflight.map(|preflight| preflight.projection);
        let gpu_scene_selected = if let Some(preflight) = gpu_scene_preflight {
            #[cfg(feature = "rdg_bench_support")]
            let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::GpuSceneCommit);
            if let Err(reason) = self.ctx.gpu_scene.commit_multi_bin_selection_with_families(
                self.ctx.extracted_scene,
                preflight,
                selected_families,
                f64::from(self.ctx.frame_time.time),
                self.ctx.render_lists,
            ) {
                record_gpu_scene_recipe_diagnostics(rdg_diagnostics, self.ctx.gpu_scene);
                rdg_diagnostics.mark_gpu_driven_failed(multi_bin_commit_failure_reason(reason));
                return;
            }
            true
        } else {
            false
        };
        if gpu_recipe_evaluated {
            record_gpu_scene_recipe_diagnostics(rdg_diagnostics, self.ctx.gpu_scene);
        }
        let selected_candidate_count =
            selection_projection.map_or(0, |projection| projection.candidate_count);
        let selected_shadow_candidate_count =
            selection_projection.map_or(0, |projection| projection.shadow_candidate_count);

        #[cfg(feature = "rdg_bench_support")]
        let cpu_intent_timing = rdg_diagnostics.detail_timer(RdgDetailPhase::CpuIntentCollection);
        if mode == GpuDrivenMode::Off || selected_families.is_empty() {
            crate::graph::culling::collect_draw_intents(
                self.ctx.extracted_scene,
                self.ctx.render_lists,
                &self.ctx.camera,
                false,
                false,
            );
        } else {
            let complete_families =
                complete_gpu_family_ownership(selected_families, selection_projection, synopsis);
            let complete_main_item_domain = complete_families
                .contains(GpuDrivenFamilyMask::MAIN_PREPASS)
                && selected_candidate_count == synopsis.extracted_item_count;
            crate::graph::culling::collect_draw_intents_with_gpu_complement(
                self.ctx.extracted_scene,
                self.ctx.render_lists,
                &self.ctx.camera,
                selected_families,
                complete_families,
                complete_main_item_domain,
            );
        }
        #[cfg(feature = "rdg_bench_support")]
        drop(cpu_intent_timing);

        rdg_diagnostics.record_gpu_driven_policy(RdgGpuDrivenPolicySnapshot {
            extracted_item_count: synopsis.extracted_item_count,
            supported_opaque_count: synopsis.supported_opaque_count,
            retained_candidate_count: selected_candidate_count,
            retained_bin_count: selection_projection.map_or(0, |projection| projection.bin_count),
            dirty_candidate_count: synopsis
                .transform_changed_count
                .min(selected_candidate_count),
            shadow_2d_view_count: synopsis.shadow_2d_view_count,
            shadow_cube_view_count: synopsis.shadow_cube_view_count,
            instance_slot_span: selection_projection
                .map_or(0, |projection| projection.table_set.instance_slot_span),
            batch_slot_span: selection_projection
                .map_or(0, |projection| projection.table_set.batch_slot_span),
            minimum_cohort_members: policy_minimum_cohort_members,
            indexed_primitive_count: synopsis.indexed_primitive_count,
            exact_evaluated: policy_exact_evaluated,
        });
        let diagnostic_reason = selection_reason;
        let main_backend = if mode == GpuDrivenMode::Off {
            RdgSceneSubmissionBackend::CpuDirect
        } else {
            selected_family_backend(
                selected_families.contains(GpuDrivenFamilyMask::MAIN_PREPASS),
                selected_candidate_count,
                synopsis.opaque_item_count,
                synopsis.opaque_item_count != 0,
            )
        };
        rdg_diagnostics.record_scene_submission_selection(
            RdgSceneSubmissionFamily::MainPrepass,
            main_backend,
            diagnostic_reason,
        );
        for (family, policy_family, view_count) in [
            (
                RdgSceneSubmissionFamily::Shadow2D,
                GpuDrivenFamilyMask::SHADOW_2D,
                shadow_workload.d2_view_count,
            ),
            (
                RdgSceneSubmissionFamily::ShadowCube,
                GpuDrivenFamilyMask::SHADOW_CUBE,
                shadow_workload.cube_view_count,
            ),
        ] {
            let backend = if mode == GpuDrivenMode::Off {
                RdgSceneSubmissionBackend::CpuDirect
            } else {
                selected_family_backend(
                    selected_families.contains(policy_family),
                    selected_shadow_candidate_count,
                    synopsis.shadow_caster_count,
                    view_count != 0 && synopsis.shadow_caster_count != 0,
                )
            };
            rdg_diagnostics.record_scene_submission_selection(family, backend, diagnostic_reason);
        }

        rdg_diagnostics.enter_phase(RdgFramePhase::BuildGraph);

        // ━━━ 2b. Build Unified RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        // Declared before `graph` so every early-return path drops the graph
        // (and invalidates its stored raw external-view pointer) before the
        // physical view. The value is initialized only after acquisition.
        let surface_view_tracked;
        let mut graph = RenderGraph::from_started_renderer_attempt(
            self.ctx.graph_storage,
            self.ctx.frame_arena,
            rdg_diagnostics,
        );
        let global_state_key =
            GlobalStateKey::new(self.ctx.render_state.id, self.ctx.extracted_scene.scene_id);
        graph.capture_global_state_plan(
            global_state_key,
            GlobalStateSources {
                camera: self.ctx.render_state.uniforms(),
                environment: &self.ctx.extracted_scene.environment_uniforms_buffer,
                directional_lights: &self.ctx.extracted_scene.directional_light_storage_buffer,
            },
        );
        graph.storage.geometry_frames.capture_visible(
            self.ctx.assets,
            self.ctx.extracted_scene,
            self.ctx.render_lists,
        );
        let geometry_eligibility = graph
            .storage
            .geometry_frames
            .eligibility(self.ctx.extracted_scene, self.ctx.render_lists);
        graph.storage.material_plans.capture(
            self.ctx.assets,
            self.ctx.extracted_scene,
            self.ctx.render_lists,
            geometry_eligibility,
            &mut graph.storage.asset_texture_plans,
        );

        let mut gpu_scene_publication_probe = None;
        let mut gpu_scene_publication_acknowledger = None;
        let gpu_scene_topology = if gpu_scene_selected {
            let Ok((tables, acknowledger)) = self.ctx.gpu_scene.plan_table_set_with_acknowledger(
                self.ctx.persistent_buffer_registry,
                self.ctx.resource_manager.frame_index(),
            ) else {
                rdg_diagnostics.mark_gpu_driven_failed(RdgGpuDrivenFailure::PersistentPlanRejected);
                return;
            };
            gpu_scene_publication_acknowledger = Some(acknowledger);
            let (patched, publication_probe) = add_gpu_scene_table_set_patch(&mut graph, &tables);
            gpu_scene_publication_probe = Some(publication_probe);
            let projection =
                gpu_scene_projection.expect("selected GPU Scene lost its mutation-free projection");
            debug_assert_eq!(
                patched.instance_slot_span,
                projection.table_set.instance_slot_span
            );
            debug_assert_eq!(
                patched.batch_slot_span,
                projection.table_set.batch_slot_span
            );
            let main_culled = selected_families
                .contains(GpuDrivenFamilyMask::MAIN_PREPASS)
                .then(|| {
                    let frustum_planes = std::array::from_fn(|index| {
                        self.ctx.camera.frustum.planes()[index].to_array()
                    });
                    let views =
                        graph.alloc_slice(&[GpuSceneCullViewDescriptor::new(frustum_planes)]);
                    self.ctx.gpu_scene_cull.add_multi_bin_to_graph(
                        &mut graph,
                        patched.instances,
                        patched.batches,
                        "GpuScene_Cull",
                        GpuSceneMultiBinCullDescriptor::new(
                            self.ctx.extracted_scene.scene_id,
                            patched.instance_slot_span,
                            patched.batch_slot_span,
                            projection.storage_alignment,
                            projection.table_set.visible_view_stride,
                            views,
                            patched.active_batches,
                        )
                        .with_required_instance_flags(GPU_SCENE_INSTANCE_MAIN_PREPASS),
                    )
                });
            Some((patched, main_culled))
        } else {
            None
        };

        let surface_desc = TextureDesc::new_2d(
            self.frame_config.width,
            self.frame_config.height,
            view_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let surface_out = graph.import_late_external_texture("Surface_View", surface_desc);

        let mut graph_ctx = GraphBuilderContext::new(
            &mut graph,
            &self.frame_config,
            self.ctx.extracted_scene.local_light_count() as u32,
        );

        // ── 2c. Register Resources ──────────────────────────────────────
        // Only the swapchain surface is truly external.
        // Scene colour and depth are transient — owned and aliased by the RDG.

        let is_high_fidelity = self.ctx.wgpu_ctx.render_path.supports_post_processing();
        let msaa_samples = self.ctx.wgpu_ctx.msaa_samples;
        let is_msaa = msaa_samples > 1;

        // ── 2d. Scene Configuration ────────────────────────────────────
        let ssao_enabled = self.ctx.scene.ssao.enabled && is_high_fidelity;
        let ssgi_enabled = is_high_fidelity
            && self
                .ctx
                .extracted_scene
                .scene_variants
                .contains(SceneFeatures::USE_SSGI);
        let ssr_enabled = is_high_fidelity
            && self
                .ctx
                .extracted_scene
                .scene_variants
                .contains(SceneFeatures::USE_SSR);
        let needs_scene_hiz = ssgi_enabled || ssr_enabled;

        let needs_feature_id = is_high_fidelity && (self.ctx.scene.ssss.enabled || ssr_enabled);

        #[cfg(feature = "debug_view")]
        let (dbg_needs_normal, dbg_needs_velocity, ssr_trace_diagnostics) = {
            use crate::graph::render_state::DebugViewTarget;
            let target = DebugViewTarget::from_mode(self.ctx.render_state.debug_view_mode);
            (
                target == DebugViewTarget::SceneNormal,
                target == DebugViewTarget::Velocity,
                target == DebugViewTarget::SsrTraceDebug,
            )
        };

        #[cfg(not(feature = "debug_view"))]
        let (dbg_needs_normal, dbg_needs_velocity, ssr_trace_diagnostics) = (false, false, false);

        let taa_enabled = self.ctx.camera.aa_mode.is_taa();
        let needs_normal = ssao_enabled || ssgi_enabled || needs_feature_id || dbg_needs_normal;
        let needs_velocity = taa_enabled || ssgi_enabled || ssr_enabled || dbg_needs_velocity;
        let prepass_snapshot = is_high_fidelity.then(|| {
            self.ctx.prepass.extract(
                GlobalStateKey::new(self.ctx.render_state.id, self.ctx.extracted_scene.scene_id),
                needs_normal,
                needs_feature_id,
                needs_velocity,
            )
        });

        #[cfg(feature = "3dgs")]
        let gaussian_snapshot = if is_high_fidelity {
            self.ctx
                .gaussian_splatting_pass
                .extract(
                    self.ctx.scene,
                    self.ctx.assets,
                    &self.ctx.camera,
                    &self.ctx.wgpu_ctx.device,
                )
                .map(|extract| {
                    GaussianSplattingFeature::stage_snapshot(
                        &mut graph_ctx,
                        extract,
                        GlobalStateKey::new(
                            self.ctx.render_state.id,
                            self.ctx.extracted_scene.scene_id,
                        ),
                        self.ctx.resource_manager.frame_index(),
                    )
                })
        } else {
            None
        };

        let taa_snapshot = if taa_enabled {
            let settings = self
                .ctx
                .camera
                .aa_mode
                .taa_settings()
                .expect("TAA settings must exist when TAA is enabled");
            Some(self.ctx.taa_pass.extract(
                self.ctx.history_registry,
                self.ctx.scene.id(),
                &self.ctx.camera,
                settings.feedback_weight,
                self.ctx.render_state.temporal_serial(),
                self.ctx.resource_manager.frame_index(),
                (self.frame_config.width, self.frame_config.height),
                self.frame_config.depth_format,
            ))
        } else {
            None
        };

        let taa_history =
            taa_snapshot.and_then(crate::graph::passes::taa::TaaSnapshot::previous_color);
        let ssgi_snapshot = ssgi_enabled.then(|| {
            self.ctx.ssgi_pass.extract(
                self.ctx.history_registry,
                self.ctx.scene.id(),
                &self.ctx.camera,
                self.ctx.render_state.temporal_serial(),
                self.ctx.resource_manager.frame_index(),
                &self.ctx.scene.ssgi.uniforms,
                (self.frame_config.width, self.frame_config.height),
                self.ctx.frame_time.frame_count as u32,
                taa_history,
                GlobalStateKey::new(self.ctx.render_state.id, self.ctx.extracted_scene.scene_id),
            )
        });

        let ssr_snapshot = ssr_enabled.then(|| {
            self.ctx.ssr_pass.extract(
                self.ctx.history_registry,
                self.ctx.scene.id(),
                &self.ctx.camera,
                self.ctx.render_state.temporal_serial(),
                &self.ctx.scene.ssr.uniforms,
                (self.frame_config.width, self.frame_config.height),
                self.ctx.resource_manager.frame_index(),
                self.ctx.frame_time.frame_count as u32,
                ssr_trace_diagnostics,
                GlobalStateKey::new(self.ctx.render_state.id, self.ctx.extracted_scene.scene_id),
            )
        });

        let clustered_snapshot = self.ctx.clustered_lighting_pass.extract(
            self.ctx.render_state,
            &self.ctx.camera,
            self.ctx.extracted_scene,
            self.ctx
                .wgpu_ctx
                .device
                .limits()
                .max_storage_buffer_binding_size,
            GlobalStateKey::new(self.ctx.render_state.id, self.ctx.extracted_scene.scene_id),
            self.ctx.clustered_lighting_enabled,
            self.ctx.extracted_scene.local_light_count() as u32,
        );

        // let needs_normal = ssao_enabled || needs_feature_id;
        let needs_skybox = self.ctx.scene.background.needs_skybox_pass();
        let skybox_global_state_key =
            GlobalStateKey::new(self.ctx.render_state.id, self.ctx.extracted_scene.scene_id);
        let skybox_color_format = if is_high_fidelity {
            crate::HDR_TEXTURE_FORMAT
        } else {
            self.frame_config.surface_format
        };
        let texture_skybox_request = if needs_skybox
            && let myth_scene::background::BackgroundMode::Texture {
                source, mapping, ..
            } = &self.ctx.scene.background.mode
        {
            match *source {
                TextureSource::Asset(handle) => Some((
                    *mapping,
                    TextureSkyboxSourceRequest::Asset(
                        graph_ctx.graph.storage.asset_texture_plans.request(handle),
                    ),
                )),
                TextureSource::Attachment(_, _) => {
                    let logical = self
                        .ctx
                        .resource_manager
                        .snapshot_texture_source(self.ctx.assets, *source);
                    (logical.ready && skybox_mapping_accepts(*mapping, logical.view_dimension))
                        .then_some((
                            *mapping,
                            TextureSkyboxSourceRequest::Attachment {
                                source: *source,
                                logical_version: logical.version,
                            },
                        ))
                }
            }
        } else {
            None
        };

        // Material and Skybox requests are now complete. Seal the one
        // graph-attempt Texture -> Image batch before either publishes an RDG
        // consumer edge, then resolve all Copy topology snapshots from it.
        graph_ctx
            .graph
            .storage
            .asset_texture_plans
            .capture_requested(self.ctx.assets);
        {
            let storage = &mut *graph_ctx.graph.storage;
            storage.material_plans.finalize_texture_snapshots(
                self.ctx.assets,
                self.ctx.extracted_scene,
                self.ctx.render_lists,
                &storage.asset_texture_plans,
            );
        }
        let draw_texture_dependencies = graph_ctx
            .graph
            .storage
            .material_plans
            .take_topology_recipes();
        let texture_skybox_plan = texture_skybox_request.and_then(|(mapping, request)| {
            let (source_plan, logical_version, view_dimension) = match request {
                TextureSkyboxSourceRequest::Asset(id) => {
                    let snapshot = graph_ctx
                        .graph
                        .storage
                        .asset_texture_plans
                        .state(id)?
                        .ready?;
                    if !asset_skybox_format_supported(&self.ctx.wgpu_ctx.device, snapshot) {
                        return None;
                    }
                    (
                        TextureSkyboxSourcePlan::Asset {
                            owner_plan: id,
                            snapshot,
                        },
                        snapshot.logical_version,
                        snapshot.view_dimension,
                    )
                }
                TextureSkyboxSourceRequest::Attachment {
                    source,
                    logical_version,
                } => {
                    let logical = self
                        .ctx
                        .resource_manager
                        .snapshot_texture_source(self.ctx.assets, source);
                    if !logical.ready || logical.version != logical_version {
                        return None;
                    }
                    (
                        TextureSkyboxSourcePlan::Attachment {
                            source,
                            logical_version,
                        },
                        logical_version,
                        logical.view_dimension,
                    )
                }
            };
            if !skybox_mapping_accepts(mapping, view_dimension) {
                return None;
            }
            Some((
                self.ctx.skybox_pass.extract_texture(
                    &self.ctx.scene.background.uniforms,
                    mapping,
                    logical_version,
                    skybox_global_state_key,
                    skybox_color_format,
                    self.frame_config.depth_format,
                    self.frame_config.msaa_samples,
                ),
                source_plan,
            ))
        });
        let gradient_skybox_snapshot = matches!(
            &self.ctx.scene.background.mode,
            myth_scene::background::BackgroundMode::Gradient { .. }
        )
        .then(|| {
            self.ctx.skybox_pass.extract_gradient(
                &self.ctx.scene.background.uniforms,
                skybox_global_state_key,
                skybox_color_format,
                self.frame_config.depth_format,
                self.frame_config.msaa_samples,
            )
        });
        let procedural_environment_snapshot =
            if let myth_scene::background::BackgroundMode::Procedural(params) =
                &self.ctx.scene.background.mode
            {
                Some(
                    self.ctx
                        .resource_manager
                        .snapshot_procedural_environment(self.ctx.assets, params),
                )
            } else {
                None
            };
        let procedural_skybox_snapshot = if let (
            myth_scene::background::BackgroundMode::Procedural(params),
            Some(environment),
        ) = (
            &self.ctx.scene.background.mode,
            procedural_environment_snapshot,
        ) {
            Some(self.ctx.skybox_pass.extract_procedural(
                params,
                environment,
                skybox_global_state_key,
                skybox_color_format,
                self.frame_config.depth_format,
                self.frame_config.msaa_samples,
            ))
        } else {
            None
        };
        let atmosphere_snapshot = if let (
            myth_scene::background::BackgroundMode::Procedural(params),
            Some(environment),
        ) = (
            &self.ctx.scene.background.mode,
            procedural_environment_snapshot,
        ) {
            Some(self.ctx.atmosphere_pass.extract(
                self.ctx.shader_manager,
                self.ctx.scene.id(),
                params,
                environment,
            ))
        } else {
            None
        };
        let atmosphere_cache_snapshot = atmosphere_snapshot.map(|snapshot| {
            self.ctx.atmosphere_pass.plan_luts(
                self.ctx.history_registry,
                &snapshot,
                self.ctx.resource_manager.frame_index(),
            )
        });
        let ssss_enabled = self.ctx.scene.ssss.enabled;
        let has_transmission = self.ctx.render_lists.use_transmission;
        let bloom_enabled = self.ctx.scene.bloom.enabled && is_high_fidelity;
        let has_active_environment = matches!(
            self.ctx.scene.background.mode,
            myth_scene::background::BackgroundMode::Procedural(_)
        ) || self.ctx.scene.environment.has_env_map();
        // let fxaa_enabled = self.ctx.wgpu_ctx.fxaa_enabled && is_high_fidelity;
        // let taa_enabled = self.ctx.wgpu_ctx.taa_enabled && is_high_fidelity;

        // Assigned at the post-compile acquisition barrier. The presentation
        // owner remains outside the execution scope and is consumed only after
        // both the graph and its late-bound view have been explicitly dropped.
        let surface_output;
        let mut reconfigure_after_present = false;
        // Exactly four semantic families can exist in one frame: Main,
        // Prepass, Shadow2D, and ShadowCube. Variable per-view submissions
        // live in the FrameArena rather than spilling this fixed family list.
        let mut gpu_scene_family_prepare_plans =
            SmallVec::<[GpuSceneMultiBinFamilyPreparePlan<'a>; 4]>::new();
        {
            let scene_id = self.ctx.scene.id();
            let scene_gpu_environment = has_active_environment
                .then(|| self.ctx.resource_manager.gpu_environment(scene_id).copied())
                .flatten()
                .map(|mut environment| {
                    let source_matches = match environment.source_type {
                        CubeSourceType::Procedural => procedural_environment_snapshot
                            .is_some_and(|snapshot| snapshot.matches_gpu_environment(&environment)),
                        CubeSourceType::Equirectangular | CubeSourceType::Cubemap => {
                            environment.source_key.is_some_and(|source| {
                                let logical = self
                                    .ctx
                                    .resource_manager
                                    .snapshot_texture_source(self.ctx.assets, source);
                                let dimension_matches = match environment.source_type {
                                    CubeSourceType::Equirectangular => {
                                        logical.view_dimension == wgpu::TextureViewDimension::D2
                                    }
                                    CubeSourceType::Cubemap => {
                                        logical.view_dimension == wgpu::TextureViewDimension::Cube
                                    }
                                    CubeSourceType::Procedural => unreachable!(),
                                };
                                logical.ready == environment.source_ready
                                    && logical.version == environment.source_version
                                    && dimension_matches
                            })
                        }
                    };
                    if !source_matches {
                        // Asset readiness/version changed after begin-frame.
                        // Treat this attempt as pending so the cache coordinator
                        // preserves the last submitted Base/PMREM publication.
                        environment.source_ready = false;
                    }
                    environment
                });
            let environment_shader_identities = scene_gpu_environment.map(|environment| {
                let shader_options = ShaderCompilationOptions::default();
                let base_shader_identity = match environment.source_type {
                    CubeSourceType::Procedural => atmosphere_snapshot
                        .expect("procedural environment requires an atmosphere snapshot")
                        .procedural_base_shader_identity(),
                    CubeSourceType::Equirectangular => {
                        self.ctx.shader_manager.pipeline_shader_hash(
                            ShaderSource::File("entry/utility/equirect_to_cube"),
                            &shader_options,
                        )
                    }
                    CubeSourceType::Cubemap => self.ctx.shader_manager.pipeline_shader_hash(
                        ShaderSource::File("entry/utility/cube_to_cube"),
                        &shader_options,
                    ),
                };
                let ibl_shader_identity = self
                    .ctx
                    .shader_manager
                    .pipeline_shader_hash(ShaderSource::File("entry/utility/ibl"), &shader_options);

                (base_shader_identity, ibl_shader_identity)
            });
            let source_conversion_snapshot = scene_gpu_environment
                .zip(environment_shader_identities)
                .and_then(|(environment, (base_shader_identity, _))| {
                    self.ctx
                        .equirect_to_cube_pass
                        .extract(&environment, base_shader_identity)
                });
            let environment_cache_snapshot = scene_gpu_environment
                .zip(environment_shader_identities)
                .map(
                    |(environment, (base_shader_identity, ibl_shader_identity))| {
                        let (procedural_sun, procedural_moon) =
                            match &self.ctx.scene.background.mode {
                                myth_scene::background::BackgroundMode::Procedural(params) => {
                                    (Some(params.sun_direction), Some(params.moon_direction))
                                }
                                _ => (None, None),
                            };

                        self.ctx.environment_cache.extract(
                            self.ctx.history_registry,
                            scene_id,
                            &environment,
                            self.ctx.resource_manager.frame_index(),
                            base_shader_identity,
                            ibl_shader_identity,
                            procedural_sun,
                            procedural_moon,
                            self.ctx
                                .wgpu_ctx
                                .device
                                .limits()
                                .min_uniform_buffer_offset_alignment,
                        )
                    },
                );

            let environment_base = environment_cache_snapshot
                .and_then(EnvironmentCacheSnapshot::base_texture)
                .map(|texture| {
                    graph_ctx
                        .graph
                        .import_persistent_texture("Scene_Environment_BaseCube", texture)
                });
            let environment_pmrem = environment_cache_snapshot
                .and_then(EnvironmentCacheSnapshot::pmrem_texture)
                .map(|texture| {
                    let node = graph_ctx
                        .graph
                        .import_persistent_texture("Scene_Environment_PMREM", texture);
                    let valid = environment_cache_snapshot
                        .and_then(|snapshot| snapshot.pmrem_plan)
                        .is_some_and(|plan| plan.valid);
                    (node, texture, valid)
                });

            let env_dependency_pmrem = environment_pmrem.map(|(node, _, _)| node);
            let environment_pmrem_max_mip_level = environment_cache_snapshot
                .filter(|snapshot| snapshot.pmrem_plan.is_some())
                .map(|snapshot| snapshot.pmrem_mip_level_count.saturating_sub(1) as f32);
            let mut procedural_skybox_inputs = None;
            let mut texture_skybox_input = None;
            let mut atmosphere_transmittance = None;
            let mut atmosphere_bake_params = None;

            let mipmap_snapshot = self.ctx.mipmap_pass.extract(self.ctx.resource_manager);
            // Register every lazy material import before its draw consumers.
            // The import store coordinates repeated exact snapshots into one
            // optional Mipmap SSA chain; each canonical submission receives only
            // the roots or aliases sampled by its own source list.
            for plan in draw_texture_dependencies.iter() {
                graph_ctx
                    .graph
                    .begin_scene_submission_texture_dependencies(plan.key.into());
                for &planned in draw_texture_dependencies.textures(plan) {
                    let snapshot = planned.snapshot;
                    let imported = graph_ctx.graph.import_planned_asset_texture(
                        "Material_Asset_Texture",
                        planned.owner_plan,
                        snapshot,
                    );
                    let ready = if let Some(ready) = graph_ctx
                        .graph
                        .planned_asset_texture_ready(planned.owner_plan)
                    {
                        ready
                    } else {
                        let needs_mipmap_generation = snapshot.mip_count > 1
                            && !self
                                .ctx
                                .resource_manager
                                .asset_snapshot_mipmaps_submitted(snapshot);
                        let ready = MipmapFeature::add_asset_texture_to_graph(
                            &mut graph_ctx,
                            imported,
                            needs_mipmap_generation,
                        );
                        graph_ctx
                            .graph
                            .set_planned_asset_texture_ready(planned.owner_plan, ready);
                        ready
                    };
                    graph_ctx
                        .graph
                        .push_scene_submission_texture_dependency(ready);
                }
                graph_ctx
                    .graph
                    .finish_scene_submission_texture_dependencies(plan.key.into());
            }
            graph_ctx
                .graph
                .storage
                .material_plans
                .recycle_topology_recipes(draw_texture_dependencies);

            // Texture producers are registered before the GPU submission so
            // its logical descriptor inherits the exact normalized Opaque
            // material dependency set, just like the CpuDirect backend.
            let mut diagnostic_main_submissions = [None; 2];
            let gpu_opaque_submission =
                gpu_scene_topology.as_ref().and_then(|(patched, culled)| {
                    let culled = culled.as_ref()?;
                    let first = culled
                        .view_batch(0, 0)
                        .expect("selected GPU Scene Cull lost its first batch");
                    let indirect_span = u64::from(patched.batch_slot_span)
                        .checked_mul(GPU_SCENE_INDEXED_ARGUMENT_SIZE)
                        .expect("GPU Scene main argument window overflowed");
                    let reserve = if self.ctx.render_lists.opaque.is_empty() {
                        RenderGraph::reserve_gpu_scene_multi_bin_submission
                    } else {
                        RenderGraph::reserve_hybrid_gpu_scene_multi_bin_submission
                    };
                    let submission = reserve(
                        graph_ctx.graph,
                        SceneSubmissionKey::Opaque,
                        patched.candidates[0].representative,
                        patched.instances,
                        patched.instances,
                        culled.visible_instances,
                        culled.indirect_arguments,
                        first.indirect_offset,
                        0,
                        indirect_span,
                    );
                    let submissions = graph_ctx.graph.alloc_slice(&[submission]);
                    gpu_scene_family_prepare_plans.push(GpuSceneMultiBinFamilyPreparePlan {
                        submissions,
                        candidates: patched.candidates,
                        instance_table: patched.instances,
                        culled: *culled,
                        target: GpuSceneSubmissionTarget::Main { global_state_key },
                    });
                    Some(submission)
                });
            let gpu_prepass_submission = if is_high_fidelity {
                gpu_scene_topology.as_ref().and_then(|(patched, culled)| {
                    let culled = culled.as_ref()?;
                    let key = SceneSubmissionKey::Prepass {
                        needs_normal,
                        needs_feature_id,
                        needs_velocity,
                    };
                    let first = culled
                        .view_batch(0, 0)
                        .expect("selected GPU Scene Cull lost its first batch");
                    let indirect_span = u64::from(patched.batch_slot_span)
                        .checked_mul(GPU_SCENE_INDEXED_ARGUMENT_SIZE)
                        .expect("GPU Scene prepass argument window overflowed");
                    let reserve = if self.ctx.render_lists.opaque.is_empty() {
                        RenderGraph::reserve_gpu_scene_multi_bin_submission
                    } else {
                        RenderGraph::reserve_hybrid_gpu_scene_multi_bin_submission
                    };
                    let submission = reserve(
                        graph_ctx.graph,
                        key,
                        patched.candidates[0].representative,
                        patched.instances,
                        patched.instances,
                        culled.visible_instances,
                        culled.indirect_arguments,
                        first.indirect_offset,
                        0,
                        indirect_span,
                    );
                    let submissions = graph_ctx.graph.alloc_slice(&[submission]);
                    gpu_scene_family_prepare_plans.push(GpuSceneMultiBinFamilyPreparePlan {
                        submissions,
                        candidates: patched.candidates,
                        instance_table: patched.instances,
                        culled: *culled,
                        target: GpuSceneSubmissionTarget::Prepass {
                            global_state_key,
                            needs_normal,
                            needs_feature_id,
                            needs_velocity,
                        },
                    });
                    Some(submission)
                })
            } else {
                None
            };

            if let Some((_, source)) = texture_skybox_plan {
                texture_skybox_input = Some(match source {
                    TextureSkyboxSourcePlan::Asset {
                        owner_plan,
                        snapshot,
                    } => {
                        let imported = graph_ctx.graph.import_planned_asset_texture(
                            "Skybox_Asset_Texture",
                            owner_plan,
                            snapshot,
                        );
                        let mip_ready = if let Some(ready) =
                            graph_ctx.graph.planned_asset_texture_ready(owner_plan)
                        {
                            ready
                        } else {
                            let needs_mipmap_generation = snapshot.mip_count > 1
                                && !self
                                    .ctx
                                    .resource_manager
                                    .asset_snapshot_mipmaps_submitted(snapshot);
                            let ready = MipmapFeature::add_asset_texture_to_graph(
                                &mut graph_ctx,
                                imported,
                                needs_mipmap_generation,
                            );
                            graph_ctx
                                .graph
                                .set_planned_asset_texture_ready(owner_plan, ready);
                            ready
                        };
                        TextureSkyboxInput::Graph(mip_ready)
                    }
                    TextureSkyboxSourcePlan::Attachment {
                        source,
                        logical_version,
                    } => TextureSkyboxInput::Attachment {
                        source,
                        logical_version,
                    },
                });
            }

            let mipmap_batch = graph_ctx
                .graph
                .stage_mipmap_requests(mipmap_snapshot.into_requests());
            let mipmap_generator = self.ctx.mipmap_pass.generator_mut();

            // ── 2c. Wire Compute + Shadow Passes ───────────────────────────
            graph_ctx.with_group("Compute", |c| {
                if let Some(batch) = mipmap_batch {
                    MipmapFeature::add_to_graph(c, batch);
                }

                // Visual atmosphere generation is independent from Environment
                // cache readiness. Pending optional star/moon assets therefore
                // suppress only Base/PMREM publication, not the live sky.
                if let (Some(snapshot), Some(cache)) =
                    (atmosphere_snapshot, atmosphere_cache_snapshot)
                {
                    let base_writer = environment_cache_snapshot
                        .filter(|environment| environment.source_type == CubeSourceType::Procedural)
                        .and_then(|environment| environment.base_plan)
                        .and_then(|plan| plan.commit.map(|token| (token, environment_base)))
                        .and_then(|(token, base)| base.map(|base| (token, base)));
                    let atmosphere_output = self.ctx.atmosphere_pass.add_to_graph(
                        c,
                        &snapshot,
                        &cache,
                        base_writer.map(|(_, base)| base),
                    );
                    procedural_skybox_inputs = Some(ProceduralSkyboxInputs {
                        sky_view: atmosphere_output.sky_view,
                        transmittance: atmosphere_output.transmittance,
                        bake_params: atmosphere_output.bake_params,
                    });
                    atmosphere_transmittance = Some(atmosphere_output.transmittance);
                    atmosphere_bake_params = Some(atmosphere_output.bake_params);
                    if let Some((token, base)) = base_writer {
                        c.graph.register_cache_commit(token, &[base]);
                    }
                }

                if let (Some(snapshot), Some(base_cube)) =
                    (environment_cache_snapshot, environment_base)
                {
                    let base_plan = snapshot
                        .base_plan
                        .expect("ready environment cache snapshot must plan a base cube");
                    match snapshot.source_type {
                        CubeSourceType::Procedural => {}
                        CubeSourceType::Equirectangular | CubeSourceType::Cubemap => {
                            if let Some(token) = base_plan.commit {
                                self.ctx.equirect_to_cube_pass.add_to_graph(
                                    c,
                                    source_conversion_snapshot.expect(
                                        "ready texture environment must have a source snapshot",
                                    ),
                                    base_cube,
                                );
                                c.graph.register_cache_commit(token, &[base_cube]);
                            }
                        }
                    }

                    if let (Some(pmrem_plan), Some((pmrem, _, _))) =
                        (snapshot.pmrem_plan, environment_pmrem)
                        && let Some(token) = pmrem_plan.commit
                    {
                        self.ctx
                            .ibl_pass
                            .add_to_graph(c, snapshot, base_cube, pmrem);
                        c.graph.register_cache_commit(token, &[pmrem]);
                    }
                }
            });

            // SSGI samples the actual scene PMREM. If IBL recomputes it this
            // frame, reading the same graph resource creates the dependency;
            // without an environment, Raw prepare binds the system black cube.
            let ssgi_pmrem = env_dependency_pmrem;

            let mut diagnostic_shadow_submissions = None;
            let shadow_output = if self.ctx.extracted_scene.has_shadow_casters() {
                let snapshot = self.ctx.shadow_pass.extract(
                    self.ctx.render_lists,
                    self.ctx.extracted_scene,
                    self.ctx.frame_arena,
                );
                let shadow_submissions = if selected_families
                    .intersects(GpuDrivenFamilyMask::SHADOW_2D | GpuDrivenFamilyMask::SHADOW_CUBE)
                {
                    let (patched, _) = gpu_scene_topology
                        .as_ref()
                        .expect("GPU Shadow ownership lost the selected GPU Scene topology");
                    let projection = gpu_scene_projection
                        .expect("GPU Shadow ownership lost its workload projection");
                    let d2 = if selected_families.contains(GpuDrivenFamilyMask::SHADOW_2D) {
                        add_gpu_scene_shadow_family(
                            graph_ctx.graph,
                            self.ctx.gpu_scene_cull,
                            self.ctx.render_lists,
                            snapshot.d2_lights,
                            patched,
                            projection,
                            self.ctx.extracted_scene.scene_id,
                            GPU_SCENE_INSTANCE_CASTS_SHADOW | GPU_SCENE_INSTANCE_SHADOW_2D,
                            "GpuScene_Shadow_2D_Cull",
                            &mut gpu_scene_family_prepare_plans,
                        )
                    } else {
                        reserve_cpu_shadow_family(graph_ctx.graph, snapshot.d2_lights)
                    };
                    let cube = if selected_families.contains(GpuDrivenFamilyMask::SHADOW_CUBE) {
                        add_gpu_scene_shadow_family(
                            graph_ctx.graph,
                            self.ctx.gpu_scene_cull,
                            self.ctx.render_lists,
                            snapshot.cube_lights,
                            patched,
                            projection,
                            self.ctx.extracted_scene.scene_id,
                            GPU_SCENE_INSTANCE_CASTS_SHADOW | GPU_SCENE_INSTANCE_SHADOW_CUBE,
                            "GpuScene_Shadow_Cube_Cull",
                            &mut gpu_scene_family_prepare_plans,
                        )
                    } else {
                        reserve_cpu_shadow_family(graph_ctx.graph, snapshot.cube_lights)
                    };
                    ShadowSubmissions { d2, cube }
                } else {
                    ShadowSubmissions {
                        d2: reserve_cpu_shadow_family(graph_ctx.graph, snapshot.d2_lights),
                        cube: reserve_cpu_shadow_family(graph_ctx.graph, snapshot.cube_lights),
                    }
                };
                diagnostic_shadow_submissions = Some(shadow_submissions);
                graph_ctx.with_group("Shadow", |c| {
                    self.ctx
                        .shadow_pass
                        .add_to_graph(c, snapshot, Some(shadow_submissions))
                })
            } else {
                crate::graph::passes::shadow::ShadowOutput {
                    shadow_2d: None,
                    shadow_cube: None,
                }
            };

            // ── 2d. Wire Scene Rendering Passes (explicit data-flow) ──────
            //
            // Each pass's `add_to_graph` creates its own transient resources
            // internally and returns typed output structs.  The Composer
            // threads `TextureNodeId` values from producer to consumer —
            // no blackboard lookups remain for mutable resources.

            // Track scene_color / scene_depth for the GraphBlackboard (hooks).
            let mut bb_scene_color = None;
            let mut bb_scene_depth = None;
            let mut bb_scene_hiz = None;

            // Debug view: capture intermediate texture IDs for safe resolution.
            #[cfg(feature = "debug_view")]
            let mut dbg_normals: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_velocity: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssao: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssgi_raw: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssgi_denoised: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssr_raw: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssr_resolved: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssr_trace_diagnostic: Option<
                crate::graph::core::TextureNodeId,
            > = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_clustered_params: Option<crate::graph::core::BufferNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_clustered_records: Option<crate::graph::core::BufferNodeId> = None;

            // Final surface handle, written by the unified PresentPass at the
            // tail of whichever pipeline path runs below.
            let current_surface;

            if is_high_fidelity {
                // ────────────────────────────────────────────────────────────
                // HighFidelity pipeline: separate passes, explicit wiring.
                // ────────────────────────────────────────────────────────────

                // ── Scene Rendering Group ──────────────────────────────────

                // let taa_enabled = self.ctx.camera.aa_mode.is_taa();

                let cas_enabled = if let Some(s) = self.ctx.camera.aa_mode.taa_settings() {
                    s.sharpen_intensity > 0.0
                } else {
                    false
                };

                let fxaa_enabled = self.ctx.camera.aa_mode.is_fxaa();

                let (mut active_color, mut scene_depth, mut scene_hiz) =
                    graph_ctx.with_group("Scene", |c| {
                        let scene_lights = c.with_group("Scene_Lighting", |c| {
                            import_scene_lighting(c, self.ctx.extracted_scene)
                        });
                        let injected_gpu_lights = self
                            .gpu_local_light_hook
                            .as_mut()
                            .and_then(|hook| c.with_group("Inject_GPU_Local_Lights", |c| hook(c)));
                        let clustered_out = self.ctx.clustered_lighting_pass.add_to_graph(
                            c,
                            clustered_snapshot,
                            ClusteredLightingInputs {
                                cpu_light_metadata_buffer: scene_lights.light_metadata,
                                cpu_light_data_buffer: scene_lights.light_storage,
                                injected_gpu_lights,
                            },
                        );
                        let scene_lighting = ClusteredScreenBindings {
                            light_metadata: Some(clustered_out.final_light_metadata_buffer),
                            lights: Some(clustered_out.final_light_data_buffer),
                            params: self
                                .ctx
                                .clustered_lighting_enabled
                                .then_some(clustered_out.params_buffer),
                            records: if self.ctx.clustered_lighting_enabled {
                                clustered_out.cluster_records
                            } else {
                                None
                            },
                            light_indices: if self.ctx.clustered_lighting_enabled {
                                clustered_out.light_indices
                            } else {
                                None
                            },
                            atmosphere_transmittance,
                            atmosphere_bake_params,
                        };

                        #[cfg(feature = "debug_view")]
                        {
                            dbg_clustered_params = Some(clustered_out.params_buffer);
                            dbg_clustered_records = clustered_out.cluster_records;
                        }

                        // 1. Prepass
                        let prepass_submission = gpu_prepass_submission.unwrap_or_else(|| {
                            c.graph
                                .reserve_scene_submission(SceneSubmissionKey::Prepass {
                                    needs_normal,
                                    needs_feature_id,
                                    needs_velocity,
                                })
                        });
                        diagnostic_main_submissions[0] = Some(prepass_submission);
                        let prepass_out = self.ctx.prepass.add_to_graph(
                            c,
                            prepass_submission,
                            prepass_snapshot
                                .expect("HighFidelity path must capture a Prepass snapshot"),
                        );

                        let scene_depth = prepass_out.scene_depth;

                        // 2. SSAO
                        let ssao_output = if ssao_enabled {
                            let viewport = self.ctx.render_state.uniforms().read().viewport;
                            let snapshot = self.ctx.ssao_pass.extract(
                                &self.ctx.scene.ssao.uniforms,
                                viewport,
                                self.ctx.resource_manager.frame_index() as u32,
                                GlobalStateKey::new(
                                    self.ctx.render_state.id,
                                    self.ctx.extracted_scene.scene_id,
                                ),
                            );
                            Some(
                                self.ctx.ssao_pass.add_to_graph(
                                    c,
                                    snapshot,
                                    scene_depth,
                                    prepass_out
                                        .scene_normals
                                        .expect("SSAO requires scene normals from Prepass"),
                                ),
                            )
                        } else {
                            None
                        };

                        // 3. Opaque
                        let opaque_submission = gpu_opaque_submission.unwrap_or_else(|| {
                            c.graph.reserve_scene_submission(SceneSubmissionKey::Opaque)
                        });
                        diagnostic_main_submissions[1] = Some(opaque_submission);
                        let opaque_out = self.ctx.opaque_pass.add_to_graph(
                            c,
                            opaque_submission,
                            global_state_key,
                            scene_depth,
                            self.ctx.extracted_scene.background.clear_color(),
                            ssss_enabled || ssr_enabled,
                            ssgi_enabled || ssr_enabled,
                            ssao_output,
                            shadow_output.shadow_2d,
                            shadow_output.shadow_cube,
                            // env_dependency_base,
                            env_dependency_pmrem,
                            scene_lighting,
                        );

                        let mut active_color = opaque_out.active_color;

                        let scene_hiz =
                            needs_scene_hiz.then(|| self.ctx.hiz_pass.add_to_graph(c, scene_depth));

                        // 4. SSSS
                        if ssss_enabled {
                            let ssss_snapshot = {
                                let registry = self.ctx.assets.sss_registry.read();
                                self.ctx.ssss_pass.extract(self.ctx.frame_arena, &registry)
                            };

                            if is_msaa {
                                let hdr_desc = TextureDesc::new_2d(
                                    c.frame_config.width,
                                    c.frame_config.height,
                                    crate::HDR_TEXTURE_FORMAT,
                                    wgpu::TextureUsages::RENDER_ATTACHMENT
                                        | wgpu::TextureUsages::TEXTURE_BINDING
                                        | wgpu::TextureUsages::COPY_SRC,
                                );
                                // If MSAA is enabled, resolve the opaque output to an intermediate non-MSAA texture for SSSS input.
                                active_color = add_msaa_resolve_pass(c, active_color, hdr_desc);
                            }

                            active_color = self.ctx.ssss_pass.add_to_graph(
                                c,
                                ssss_snapshot,
                                active_color,
                                prepass_out.scene_depth,
                                prepass_out.scene_normals.unwrap(),
                                prepass_out.feature_id.unwrap(),
                                opaque_out.specular_mrt.unwrap(),
                            );

                            if is_msaa {
                                // If MSAA is enabled, synchronize the SSSS output back to an MSAA texture for downstream passes (Skybox, Transparent) that expect MSAA input.
                                // This avoids redundant MSAA resolve + re-multisample operations.
                                let snapshot = self
                                    .ctx
                                    .msaa_sync_pass
                                    .extract(self.frame_config.msaa_samples);
                                active_color =
                                    self.ctx
                                        .msaa_sync_pass
                                        .add_to_graph(c, snapshot, active_color);
                            }
                        }

                        // 5. Skybox
                        if needs_skybox {
                            let draw = if let Some((snapshot, _)) = texture_skybox_plan {
                                texture_skybox_input.map(|input| {
                                    self.ctx.skybox_pass.texture_draw_request(snapshot, input)
                                })
                            } else if let Some(snapshot) = procedural_skybox_snapshot {
                                procedural_skybox_inputs.map(|inputs| {
                                    self.ctx
                                        .skybox_pass
                                        .procedural_draw_request(snapshot, inputs)
                                })
                            } else if let Some(snapshot) = gradient_skybox_snapshot {
                                Some(self.ctx.skybox_pass.gradient_draw_request(snapshot))
                            } else {
                                None
                            };
                            if let Some(draw) = draw {
                                active_color =
                                    draw.add_to_graph(c, active_color, opaque_out.active_depth);
                            }
                        }

                        if ssgi_enabled {
                            let ssgi_out = self.ctx.ssgi_pass.add_to_graph(
                                c,
                                ssgi_snapshot.expect("SSGI snapshot missing"),
                                active_color,
                                scene_depth,
                                scene_hiz.expect("SSGI requires Hi-Z pyramid"),
                                prepass_out
                                    .scene_normals
                                    .expect("SSGI requires scene normals from Prepass"),
                                prepass_out
                                    .velocity_buffer
                                    .expect("SSGI requires motion vectors from Prepass"),
                                opaque_out
                                    .material_mrt
                                    .expect("SSGI requires opaque material MRT"),
                                ssgi_pmrem,
                            );

                            #[cfg(feature = "debug_view")]
                            {
                                dbg_ssgi_raw = Some(ssgi_out.raw_indirect);
                                dbg_ssgi_denoised = Some(ssgi_out.clean_indirect);
                            }

                            active_color = ssgi_out.merged_color;
                        }

                        if ssr_enabled {
                            let ssr_out = self.ctx.ssr_pass.add_to_graph(
                                c,
                                ssr_snapshot.expect("SSR snapshot missing"),
                                active_color,
                                scene_depth,
                                scene_hiz.expect("SSR requires Hi-Z pyramid"),
                                prepass_out
                                    .scene_normals
                                    .expect("SSR requires scene normals from Prepass"),
                                prepass_out
                                    .velocity_buffer
                                    .expect("SSR requires motion vectors from Prepass"),
                                opaque_out
                                    .material_mrt
                                    .expect("SSR requires opaque material MRT"),
                                opaque_out
                                    .specular_mrt
                                    .expect("SSR requires opaque specular MRT"),
                            );

                            #[cfg(feature = "debug_view")]
                            {
                                dbg_ssr_raw = Some(ssr_out.raw_reflection);
                                dbg_ssr_resolved = Some(ssr_out.clean_reflection);
                                dbg_ssr_trace_diagnostic = ssr_out.raw_diagnostic;
                            }

                            active_color = ssr_out.merged_color;
                        }

                        // ── 6. TAA Resolve ────────────────────────────────────────────
                        // Resolve temporal anti-aliasing before bloom/tone-mapping.
                        // The resolved colour replaces post_transparent_color for
                        // downstream post-processing.
                        if taa_enabled && let Some(velocity) = prepass_out.velocity_buffer {
                            c.with_group("TAA_System", |c| {
                                active_color = self.ctx.taa_pass.add_to_graph(
                                    c,
                                    taa_snapshot.expect("TAA snapshot missing"),
                                    active_color,
                                    velocity,
                                    scene_depth,
                                );

                                // ── 6b. CAS (Contrast Adaptive Sharpening) ────────────
                                // Recover fine detail lost to temporal filtering.
                                if cas_enabled {
                                    let taa_settings = self
                                        .ctx
                                        .camera
                                        .aa_mode
                                        .taa_settings()
                                        .expect("CAS requires TAA settings");
                                    let snapshot = self.ctx.cas_pass.extract(
                                        taa_settings.sharpen_intensity,
                                        self.frame_config.hdr_format,
                                    );
                                    active_color =
                                        self.ctx.cas_pass.add_to_graph(c, snapshot, active_color);
                                }
                            });
                        }

                        #[cfg(feature = "3dgs")]
                        if let Some(snapshot) = gaussian_snapshot {
                            // 7a. Gaussian Splatting
                            c.with_group("3D_Gaussian_Splatting", |c| {
                                active_color = self.ctx.gaussian_splatting_pass.add_to_graph(
                                    c,
                                    snapshot,
                                    active_color,
                                    opaque_out.active_depth,
                                );
                            });
                        }

                        // 7. Transmission Copy
                        let transmission_tex = if has_transmission {
                            Some(
                                self.ctx
                                    .transmission_copy_pass
                                    .add_to_graph(c, active_color),
                            )
                        } else {
                            None
                        };

                        // 8. Transparent
                        let active_color = self.ctx.transparent_pass.add_to_graph(
                            c,
                            global_state_key,
                            active_color,
                            opaque_out.active_depth,
                            transmission_tex,
                            ssao_output,
                            shadow_output.shadow_2d,
                            shadow_output.shadow_cube,
                            scene_lighting,
                        );

                        // Capture intermediate IDs for debug view resolution.
                        #[cfg(feature = "debug_view")]
                        {
                            dbg_normals = prepass_out.scene_normals;
                            dbg_velocity = prepass_out.velocity_buffer;
                            dbg_ssao = ssao_output;
                        }

                        (active_color, scene_depth, scene_hiz)
                    });

                // ── Before-Post-Process Hooks ──────────────────────────────
                {
                    let mut blackboard = GraphBlackboard {
                        scene_color: Some(active_color),
                        scene_depth: Some(scene_depth),
                        scene_hiz,
                        environment_base_cube: environment_base,
                        environment_pmrem: env_dependency_pmrem,
                        environment_pmrem_max_mip_level,
                        atmosphere_transmittance,
                        atmosphere_bake_params,
                        surface_out,
                    };
                    for (stage, hook_opt) in &mut self.hooks {
                        if *stage == HookStage::BeforePostProcess
                            && let Some(hook) = hook_opt.take()
                        {
                            blackboard = hook(graph_ctx.graph, blackboard);
                        }
                    }

                    active_color = blackboard.scene_color.unwrap_or(active_color);
                    scene_depth = blackboard.scene_depth.unwrap_or(scene_depth);
                    scene_hiz = blackboard.scene_hiz;
                }

                // ── Post-Processing Group ──────────────────────────────────
                // Every stage renders into a transient LDR target.  The unified
                // PresentPass appended below is the sole writer of the surface,
                // so no post-process pass needs to know whether it happens to be
                // the final stage.
                #[cfg_attr(not(feature = "debug_view"), allow(unused_mut))]
                let mut current_target = graph_ctx.with_group("PostProcess", |ctx| {
                    // Bloom (internally flattened into Bloom_System subgroup)
                    if bloom_enabled {
                        let snapshot = self.ctx.bloom_pass.extract(&self.ctx.scene.bloom);
                        active_color =
                            self.ctx
                                .bloom_pass
                                .add_to_graph(ctx, snapshot, active_color);
                    }

                    // ToneMapping: HDR → LDR, into a transient target.
                    let snapshot = self.ctx.tone_map_pass.extract(
                        self.ctx.resource_manager,
                        self.ctx.assets,
                        self.ctx.scene.tone_mapping.mode,
                        self.frame_config.surface_format,
                        GlobalStateKey::new(
                            self.ctx.render_state.id,
                            self.ctx.extracted_scene.scene_id,
                        ),
                        &self.ctx.scene.tone_mapping.uniforms,
                        self.ctx.scene.tone_mapping.lut_texture,
                    );
                    active_color = self
                        .ctx
                        .tone_map_pass
                        .add_to_graph(ctx, snapshot, active_color);

                    // FXAA: anti-alias the LDR result into a second transient
                    // target.
                    if fxaa_enabled {
                        let fxaa_settings = self
                            .ctx
                            .camera
                            .aa_mode
                            .fxaa_settings()
                            .expect("FXAA settings must exist when FXAA is enabled");
                        let snapshot = self
                            .ctx
                            .fxaa_pass
                            .extract(fxaa_settings.quality(), self.frame_config.surface_format);
                        active_color = self.ctx.fxaa_pass.add_to_graph(ctx, snapshot, active_color);
                    }

                    bb_scene_color = Some(active_color);
                    bb_scene_depth = Some(scene_depth);
                    bb_scene_hiz = scene_hiz;

                    active_color
                });

                // ── Debug View Override ────────────────────────────────────
                // Resolve the semantic DebugViewMode to a concrete
                // TextureNodeId, then blit it onto the surface.  Targets
                // whose producer was disabled (e.g. SSAO off) safely
                // resolve to None — no pass is injected.
                // Material-override modes are handled separately via shader
                // defines and do not use this post-process path.
                #[cfg(feature = "debug_view")]
                {
                    use crate::graph::render_state::DebugViewTarget;
                    let target = DebugViewTarget::from_mode(self.ctx.render_state.debug_view_mode);
                    let source: Option<crate::graph::core::TextureNodeId> = match target {
                        DebugViewTarget::SceneNormal => dbg_normals,
                        DebugViewTarget::Velocity => dbg_velocity,
                        DebugViewTarget::SsaoRaw => dbg_ssao,
                        DebugViewTarget::SceneDepth | DebugViewTarget::ClusterHeatmap => {
                            Some(scene_depth)
                        }
                        DebugViewTarget::SsgiRaw => dbg_ssgi_raw,
                        DebugViewTarget::SsgiDenoised => dbg_ssgi_denoised,
                        DebugViewTarget::SsrRaw => dbg_ssr_raw,
                        DebugViewTarget::SsrResolved => dbg_ssr_resolved,
                        DebugViewTarget::SsrTraceDebug => dbg_ssr_trace_diagnostic,
                        DebugViewTarget::None => None,
                    };

                    if let Some(src) = source
                        && let Some(snapshot) = self.ctx.debug_view_pass.extract(
                            &self.ctx.camera,
                            self.frame_config.surface_format,
                            target,
                        )
                    {
                        let clustered = if target == DebugViewTarget::ClusterHeatmap {
                            ClusteredScreenBindings {
                                light_metadata: None,
                                lights: None,
                                params: dbg_clustered_params,
                                records: dbg_clustered_records,
                                light_indices: None,
                                atmosphere_transmittance: None,
                                atmosphere_bake_params: None,
                            }
                        } else {
                            ClusteredScreenBindings::default()
                        };

                        current_target = self.ctx.debug_view_pass.add_to_graph(
                            &mut graph_ctx,
                            snapshot,
                            src,
                            current_target,
                            clustered,
                        );
                    }
                }

                // ── Present ────────────────────────────────────────────────
                // Forward the final LDR target onto the swap-chain surface.  In
                // the common case (target already matches the surface format and
                // size) the graph compiler folds this blit away, rewiring the
                // last real pass to target the surface directly.
                current_surface =
                    self.ctx
                        .present_pass
                        .add_to_graph(&mut graph_ctx, current_target, surface_out);
            } else {
                // BasicForward pipeline: single-pass LDR rendering.

                let scene_color = graph_ctx.with_group("BasicForward", |c| {
                    let scene_lights = c.with_group("Scene_Lighting", |c| {
                        import_scene_lighting(c, self.ctx.extracted_scene)
                    });
                    let injected_gpu_lights = self
                        .gpu_local_light_hook
                        .as_mut()
                        .and_then(|hook| c.with_group("Inject_GPU_Local_Lights", |c| hook(c)));
                    let clustered_out = self.ctx.clustered_lighting_pass.add_to_graph(
                        c,
                        clustered_snapshot,
                        ClusteredLightingInputs {
                            cpu_light_metadata_buffer: scene_lights.light_metadata,
                            cpu_light_data_buffer: scene_lights.light_storage,
                            injected_gpu_lights,
                        },
                    );
                    let scene_lighting = ClusteredScreenBindings {
                        light_metadata: Some(clustered_out.final_light_metadata_buffer),
                        lights: Some(clustered_out.final_light_data_buffer),
                        params: self
                            .ctx
                            .clustered_lighting_enabled
                            .then_some(clustered_out.params_buffer),
                        records: if self.ctx.clustered_lighting_enabled {
                            clustered_out.cluster_records
                        } else {
                            None
                        },
                        light_indices: if self.ctx.clustered_lighting_enabled {
                            clustered_out.light_indices
                        } else {
                            None
                        },
                        atmosphere_transmittance,
                        atmosphere_bake_params,
                    };
                    let skybox = if needs_skybox {
                        if let Some((snapshot, _)) = texture_skybox_plan {
                            texture_skybox_input.map(|input| {
                                self.ctx.skybox_pass.texture_draw_request(snapshot, input)
                            })
                        } else if let Some(snapshot) = procedural_skybox_snapshot {
                            procedural_skybox_inputs.map(|inputs| {
                                self.ctx
                                    .skybox_pass
                                    .procedural_draw_request(snapshot, inputs)
                            })
                        } else if let Some(snapshot) = gradient_skybox_snapshot {
                            Some(self.ctx.skybox_pass.gradient_draw_request(snapshot))
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let opaque_submission = gpu_opaque_submission.unwrap_or_else(|| {
                        c.graph.reserve_scene_submission(SceneSubmissionKey::Opaque)
                    });
                    diagnostic_main_submissions[0] = Some(opaque_submission);
                    self.ctx.simple_forward_pass.add_to_graph(
                        c,
                        opaque_submission,
                        global_state_key,
                        self.ctx.extracted_scene.background.clear_color(),
                        skybox,
                        shadow_output.shadow_2d,
                        shadow_output.shadow_cube,
                        env_dependency_pmrem,
                        scene_lighting,
                    )
                });

                // ── Present ────────────────────────────────────────────────
                // Forward the rendered LDR image onto the swap-chain surface
                // (folded away at compile time in the common case).
                current_surface =
                    self.ctx
                        .present_pass
                        .add_to_graph(&mut graph_ctx, scene_color, surface_out);
            }

            // drop(graph_ctx);

            // ── After-Post-Process Hooks (UI, debug overlays) ──────────────
            {
                let mut blackboard = GraphBlackboard {
                    scene_color: bb_scene_color,
                    scene_depth: bb_scene_depth,
                    scene_hiz: bb_scene_hiz,
                    environment_base_cube: environment_base,
                    environment_pmrem: env_dependency_pmrem,
                    environment_pmrem_max_mip_level,
                    atmosphere_transmittance,
                    atmosphere_bake_params,
                    surface_out: current_surface,
                };
                for (stage, hook_opt) in &mut self.hooks {
                    if *stage == HookStage::AfterPostProcess
                        && let Some(hook) = hook_opt.take()
                    {
                        blackboard = hook(&mut graph, blackboard);
                    }
                }
            }

            // ━━━ 3. Compile & Execute RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            // Logical topology, DPE, lifetimes, and submission demand are all
            // CPU-only and intentionally complete before waiting for a
            // swap-chain image.
            graph.compile_topology();

            let mut gpu_scene_live_submission = false;
            for submission in diagnostic_main_submissions.into_iter().flatten() {
                let (cpu, gpu) = graph
                    .storage
                    .scene_submissions
                    .live_backend_presence(submission)
                    .expect("main diagnostics retained a stale scene submission");
                gpu_scene_live_submission |= gpu;
                rdg_diagnostics.record_live_scene_submissions(
                    RdgSceneSubmissionFamily::MainPrepass,
                    RdgSceneSubmissionBackend::CpuDirect,
                    u32::from(cpu),
                );
                rdg_diagnostics.record_live_scene_submissions(
                    RdgSceneSubmissionFamily::MainPrepass,
                    RdgSceneSubmissionBackend::GpuDriven,
                    u32::from(gpu),
                );
            }
            if let Some(submissions) = diagnostic_shadow_submissions {
                for (family, family_submissions) in [
                    (RdgSceneSubmissionFamily::Shadow2D, submissions.d2),
                    (RdgSceneSubmissionFamily::ShadowCube, submissions.cube),
                ] {
                    for &submission in family_submissions {
                        let (cpu, gpu) = graph
                            .storage
                            .scene_submissions
                            .live_backend_presence(submission)
                            .expect("shadow diagnostics retained a stale scene submission");
                        gpu_scene_live_submission |= gpu;
                        rdg_diagnostics.record_live_scene_submissions(
                            family,
                            RdgSceneSubmissionBackend::CpuDirect,
                            u32::from(cpu),
                        );
                        rdg_diagnostics.record_live_scene_submissions(
                            family,
                            RdgSceneSubmissionBackend::GpuDriven,
                            u32::from(gpu),
                        );
                    }
                }
            }

            // ━━━ 3a. Acquire and Late-bind Render Target ━━━━━━━━━━━━━━━
            rdg_diagnostics.enter_phase(RdgFramePhase::AcquireTarget);

            #[cfg(test)]
            if let Some(reason) = self.forced_surface_acquisition_failure {
                rdg_diagnostics.mark_surface_acquisition_failed(reason);
                return;
            }

            if let Some(surface) = &self.ctx.wgpu_ctx.surface {
                let (output, suboptimal) = match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame) => (frame, false),
                    wgpu::CurrentSurfaceTexture::Suboptimal(frame) => (frame, true),
                    failure => {
                        let reason = classify_surface_acquisition_failure(&failure)
                            .expect("successful surface acquisition entered the failure branch");
                        if matches!(reason, RdgSurfaceAcquireFailure::Outdated)
                            && let Some(config) = &self.ctx.wgpu_ctx.config
                        {
                            surface.configure(&self.ctx.wgpu_ctx.device, config);
                        }
                        if rdg_diagnostics.mark_surface_acquisition_failed(reason) {
                            match reason {
                                RdgSurfaceAcquireFailure::Timeout => {
                                    log::trace!(
                                        "Skipped frame: timed out acquiring swap-chain surface"
                                    );
                                }
                                RdgSurfaceAcquireFailure::Occluded => {
                                    log::debug!("Skipped frame: swap-chain surface is occluded");
                                }
                                RdgSurfaceAcquireFailure::Outdated => {
                                    log::debug!(
                                        "Skipped frame: reconfigured outdated swap-chain surface"
                                    );
                                }
                                RdgSurfaceAcquireFailure::Lost => {
                                    log::warn!(
                                        "Skipped frame: lost swap-chain surface must be recreated by the application"
                                    );
                                }
                                RdgSurfaceAcquireFailure::Validation => {
                                    log::error!(
                                        "Validation error while acquiring swap-chain surface"
                                    );
                                }
                            }
                        }
                        return;
                    }
                };

                if !render_target_matches_desc(
                    &output.texture,
                    surface_desc,
                    RenderTargetFormatContract::SurfaceViewFamily,
                ) {
                    drop(output);
                    if let Some(config) = &self.ctx.wgpu_ctx.config {
                        surface.configure(&self.ctx.wgpu_ctx.device, config);
                    }
                    if rdg_diagnostics
                        .mark_surface_acquisition_failed(RdgSurfaceAcquireFailure::Outdated)
                    {
                        log::warn!(
                            "Acquired swap-chain target no longer matches the compiled RDG descriptor"
                        );
                    }
                    return;
                }

                rdg_diagnostics.mark_target_acquired(RdgRenderTargetKind::Surface, suboptimal);
                if suboptimal {
                    reconfigure_after_present = true;
                    log::debug!(
                        "Acquired suboptimal swap-chain surface; reconfiguring after present"
                    );
                }
                surface_view_tracked = Tracked::with_id(
                    output.texture.create_view(&wgpu::TextureViewDescriptor {
                        format: Some(view_format),
                        ..Default::default()
                    }),
                    0,
                );
                surface_output = Some(output);
            } else if let Some(texture) = &self.ctx.wgpu_ctx.headless_texture {
                if !render_target_matches_desc(
                    texture,
                    surface_desc,
                    RenderTargetFormatContract::Exact,
                ) {
                    rdg_diagnostics.mark_resource_materialization_failed(
                        RdgResourceMaterializationFailure::RenderTargetDescriptorDrift,
                    );
                    return;
                }
                rdg_diagnostics.mark_target_acquired(RdgRenderTargetKind::Headless, false);
                surface_view_tracked = Tracked::with_id(
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        format: Some(view_format),
                        ..Default::default()
                    }),
                    0,
                );
                surface_output = None;
            } else {
                unreachable!("render-target presence was checked before graph construction");
            }

            graph.bind_late_external_texture(surface_out, &surface_view_tracked);
            graph.materialize_transient_resources(
                self.ctx.transient_pool,
                &self.ctx.wgpu_ctx.device,
            );
            let (scene_buffers, global_buffers) = {
                let resource_manager = &mut *self.ctx.resource_manager;
                let mut uploads = BufferUploadContext::new(
                    &self.ctx.wgpu_ctx.device,
                    &self.ctx.wgpu_ctx.queue,
                    resource_manager.frame_index,
                    &mut resource_manager.resources,
                    &resource_manager.internal_textures,
                );
                let scene_buffers = {
                    #[cfg(feature = "rdg_bench_support")]
                    let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::SceneBuffers);
                    graph.materialize_live_scene_buffers(
                        SceneBufferSources {
                            light_metadata: &self.ctx.extracted_scene.local_light_metadata_buffer,
                            light_storage: &self.ctx.extracted_scene.local_light_storage_buffer,
                        },
                        &mut uploads,
                    )
                };
                let global_buffers = if scene_buffers.is_ok() {
                    #[cfg(feature = "rdg_bench_support")]
                    let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::GlobalBuffers);
                    graph.materialize_live_global_state(
                        GlobalStateSources {
                            camera: self.ctx.render_state.uniforms(),
                            environment: &self.ctx.extracted_scene.environment_uniforms_buffer,
                            directional_lights: &self
                                .ctx
                                .extracted_scene
                                .directional_light_storage_buffer,
                        },
                        &mut uploads,
                    )
                } else {
                    Ok(())
                };
                (scene_buffers, global_buffers)
            };
            if let Err(reason) = scene_buffers {
                let reason = match reason {
                    SceneBufferMaterializationFailure::SnapshotDrift => {
                        RdgResourceMaterializationFailure::SceneBufferSnapshotDrift
                    }
                    SceneBufferMaterializationFailure::ConflictingPhysicalState => {
                        RdgResourceMaterializationFailure::IncompatibleSceneBufferPhysicalState
                    }
                    SceneBufferMaterializationFailure::Unavailable => {
                        RdgResourceMaterializationFailure::IncompleteSceneBufferBatch
                    }
                };
                rdg_diagnostics.mark_resource_materialization_failed(reason);
                return;
            }
            if let Err(reason) = global_buffers {
                let reason = match reason {
                    GlobalStateMaterializationFailure::SnapshotDrift => {
                        RdgResourceMaterializationFailure::GlobalStateSnapshotDrift
                    }
                    GlobalStateMaterializationFailure::ConflictingPhysicalState => {
                        RdgResourceMaterializationFailure::IncompatibleGlobalStatePhysicalState
                    }
                    GlobalStateMaterializationFailure::Unavailable => {
                        RdgResourceMaterializationFailure::IncompleteGlobalStateBatch
                    }
                };
                rdg_diagnostics.mark_resource_materialization_failed(reason);
                return;
            }
            {
                #[cfg(feature = "rdg_bench_support")]
                let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::PersistentTextures);
                graph.materialize_live_persistent_textures(
                    self.ctx.history_registry,
                    &self.ctx.wgpu_ctx.device,
                );
            }
            let persistent_buffers = if graph.has_persistent_buffer_imports() {
                #[cfg(feature = "rdg_bench_support")]
                let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::PersistentBuffers);
                graph.materialize_live_persistent_buffers(
                    self.ctx.persistent_buffer_registry,
                    &self.ctx.wgpu_ctx.device,
                )
            } else {
                Ok(())
            };
            if let Err(reason) = persistent_buffers {
                let reason = match reason {
                    PersistentBufferGraphMaterializationFailure::Registry(
                        crate::graph::core::PersistentBufferMaterializationFailure::CapacityOverflow,
                    ) => RdgResourceMaterializationFailure::PersistentBufferCapacityOverflow,
                    PersistentBufferGraphMaterializationFailure::Registry(
                        crate::graph::core::PersistentBufferMaterializationFailure::ExceedsDeviceLimit,
                    ) => RdgResourceMaterializationFailure::PersistentBufferExceedsDeviceLimit,
                    PersistentBufferGraphMaterializationFailure::PartialCommitGroup
                    | PersistentBufferGraphMaterializationFailure::InvalidLiveBufferHasNoCommit => {
                        RdgResourceMaterializationFailure::PersistentBufferIncompleteWriteBatch
                    }
                    PersistentBufferGraphMaterializationFailure::Registry(
                        crate::graph::core::PersistentBufferMaterializationFailure::UnknownOrRetired
                        | crate::graph::core::PersistentBufferMaterializationFailure::StalePlan,
                    )
                    | PersistentBufferGraphMaterializationFailure::PublishedBufferHasLiveWriter => {
                        RdgResourceMaterializationFailure::PersistentBufferStateDrift
                    }
                };
                rdg_diagnostics.mark_resource_materialization_failed(reason);
                return;
            }
            {
                #[cfg(feature = "rdg_bench_support")]
                let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::AssetTexture);
                graph.materialize_live_asset_textures(self.ctx.assets, self.ctx.resource_manager);
            }

            // Group 0 keeps a stable layout/WGSL interface, but its PMREM
            // physical view is selected only after DPE has decided whether a
            // same-frame cache writer is live and materialized. The graph read
            // edge on `env_dependency_pmrem` orders that writer before every
            // scene pass using this refreshed bind group.
            {
                #[cfg(feature = "rdg_bench_support")]
                let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::GlobalBindGroup);
                let pmrem_override = environment_pmrem.and_then(|(node, texture, valid)| {
                    let live =
                        graph.storage.resources[node.index() as usize].first_use != usize::MAX;
                    if !(valid || live)
                        || !self
                            .ctx
                            .history_registry
                            .is_texture_materialized(texture.id)
                    {
                        return None;
                    }
                    Some(self.ctx.history_registry.get_texture_view(texture.id))
                });
                if let Some(key) = graph.global_state_allowed_key() {
                    let [camera, environment, directional_lights] = graph
                        .materialized_global_buffer_tokens()
                        .expect("DPE-live global state lacks its atomic buffer batch");
                    let finalize_result: Result<u32, GlobalStatePhysicalFailure> =
                        self.ctx.resource_manager.prepare_global_from_materialized(
                            key,
                            &camera,
                            &environment,
                            &directional_lights,
                            pmrem_override,
                        );
                    if finalize_result.is_err() {
                        rdg_diagnostics.mark_resource_materialization_failed(
                            RdgResourceMaterializationFailure::IncompatibleGlobalStatePhysicalState,
                        );
                        return;
                    }
                    let bind_group = self
                        .ctx
                        .resource_manager
                        .get_global_state(key.render_state_id(), key.scene_id())
                        .expect("finalized global state was not published")
                        .bind_group
                        .clone();
                    graph.stage_finalized_global_bind_group(key, &bind_group);
                }
            }

            // ─── 3a. Materialize the DPE-live central source union ───────
            //
            // Draw-plan demand marks are the sole liveness authority. Main,
            // Prepass, and any number of shadow views share one source slot;
            // dead plan branches never allocate model data, geometry, object
            // bindings, or pipelines.
            {
                #[cfg(feature = "rdg_bench_support")]
                let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::LiveSourceUnion);
                let storage = &mut *graph.storage;
                let any_gpu_submission_live = storage.scene_submissions.has_live_gpu_plan();
                storage.scene_source_prepare.select_live_sources_with_gpu(
                    self.ctx.render_lists,
                    storage.draw_plans.live_keys(),
                    self.ctx
                        .render_lists
                        .gpu_opaque_representatives()
                        .filter(|_| any_gpu_submission_live),
                );
            }
            {
                #[cfg(feature = "rdg_bench_support")]
                let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::MaterialPlan);
                let storage = &mut *graph.storage;
                storage.material_plans.prepare_live(
                    self.ctx.assets,
                    &mut storage.asset_texture_plans,
                    self.ctx.resource_manager,
                    storage.scene_source_prepare.live_sources(),
                );
            }
            {
                #[cfg(feature = "rdg_bench_support")]
                let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::Geometry);
                graph.storage.geometry_frames.capture_live(
                    self.ctx.assets,
                    self.ctx.extracted_scene,
                    self.ctx.render_lists,
                    graph.storage.scene_source_prepare.live_sources(),
                );
                graph.storage.geometry_frames.prepare_live(
                    self.ctx.assets,
                    self.ctx.resource_manager,
                    &mut graph.storage.draw_plans,
                );
            }
            {
                let storage = &mut *graph.storage;
                {
                    #[cfg(feature = "rdg_bench_support")]
                    let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::SourceGather);
                    storage.scene_source_prepare.gather(
                        self.ctx.extracted_scene,
                        self.ctx.render_lists,
                        storage.geometry_frames.resolver(),
                        storage.material_plans.resolver(),
                    );
                }
                {
                    #[cfg(feature = "rdg_bench_support")]
                    let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::SourcePlan);
                    storage.scene_source_prepare.plan();
                }
                {
                    #[cfg(feature = "rdg_bench_support")]
                    let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::SourceMaterialize);
                    storage.scene_source_prepare.materialize(
                        self.ctx.scene,
                        self.ctx.resource_manager,
                        storage.geometry_frames.resolver(),
                        storage.material_plans.resolver(),
                    );
                }
                {
                    #[cfg(feature = "rdg_bench_support")]
                    let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::SourceFinalize);
                    storage.scene_source_prepare.finalize(
                        storage.material_plans.resolver(),
                        self.ctx.resource_manager,
                        &mut storage.object_interfaces,
                        &mut storage.prepared_draws,
                        &mut storage.draw_plans,
                    );
                }
            }

            rdg_diagnostics.enter_phase(RdgFramePhase::PreparePasses);

            // ─── 3b. RDG Prepare: live-pass deferred preparation ───────────
            //
            // Only the swapchain surface is truly external — all other textures
            // (scene_color, scene_depth, etc.) are RDG transient resources.

            {
                let finalized_global_state_key = graph.finalized_global_state_key();
                let mut prepare_ctx = PrepareContext {
                    views: ViewResolver {
                        resources: &graph.storage.resources,
                        pool: self.ctx.transient_pool,
                        persistent_resources: self.ctx.history_registry,
                        persistent_buffers: PersistentBufferResolver::new(
                            self.ctx.persistent_buffer_registry,
                            &graph.storage.persistent_buffer_materializations,
                        ),
                        asset_textures: graph.storage.asset_texture_imports.resolver(),
                        scene_buffers: graph.storage.scene_buffer_imports.resolver(),
                    },
                    device: &self.ctx.wgpu_ctx.device,
                    queue: &self.ctx.wgpu_ctx.queue,
                    pipeline_cache: self.ctx.pipeline_cache,
                    shader_manager: self.ctx.shader_manager,
                    assets: self.ctx.assets,
                    uploads: crate::core::gpu::BufferUploadContext::new(
                        &self.ctx.wgpu_ctx.device,
                        &self.ctx.wgpu_ctx.queue,
                        self.ctx.resource_manager.frame_index,
                        &mut self.ctx.resource_manager.resources,
                        &self.ctx.resource_manager.internal_textures,
                    ),
                    scene_draws: SceneDrawPrepareContext::new(
                        self.ctx.render_lists,
                        self.ctx.extracted_scene,
                        &graph.storage.prepared_draws,
                        graph.storage.geometry_frames.resolver(),
                        graph.storage.material_plans.resolver(),
                        graph.storage.object_interfaces.resolver(),
                        self.ctx.wgpu_ctx.depth_format,
                        self.ctx.wgpu_ctx.pipeline_settings_version,
                        &mut graph.storage.scene_submissions,
                        &mut graph.storage.scene_pipeline_prepare,
                        #[cfg(feature = "rdg_bench_support")]
                        rdg_diagnostics,
                    ),
                    mipmap_generator,
                    mipmap_requests: MipmapRequestResolver::new(&graph.storage.mipmap_batches),
                    #[cfg(feature = "3dgs")]
                    gaussian_frames: GaussianFrameResolver::new(
                        &graph.storage.gaussian_frame_batches,
                    ),
                    global_states: GlobalStateResolver::new(
                        &self.ctx.resource_manager.bind_groups,
                        finalized_global_state_key,
                    ),
                    sampler_registry: &self.ctx.resource_manager.sampler_registry,
                    system_textures: &self.ctx.resource_manager.system_textures,
                    global_bind_group_cache: self.ctx.global_bind_group_cache,
                    prepared_bind_groups: &mut graph.storage.prepared_bind_groups,
                    persistent_buffer_uploads: PersistentBufferUploadPreparer::new(
                        &mut graph.storage.persistent_buffer_commit_groups,
                    ),
                };

                for &pass_idx in &graph.storage.execution_queue {
                    // SAFETY: RenderGraph still owns the live FrameArena and
                    // the execution queue is traversed sequentially, so this
                    // pass has no competing mutable reference.
                    let pass = unsafe { graph.storage.passes[pass_idx].get_pass_mut() };
                    pass.prepare(&mut prepare_ctx);
                    rdg_diagnostics.record_prepared_pass();
                }

                // ─── 3c. Gather barrier + typed scene coordinators ─────────
                //
                // All live nodes have now emitted immutable recipes. The
                // serial owner restores semantic order, exact-deduplicates
                // canonical requests, creates only cold physical pipelines,
                // and assigns Copy IDs before Feature L1 publication.
                rdg_diagnostics.enter_phase(RdgFramePhase::FinalizeDrawPlans);
                prepare_ctx.coordinate_scene_draws(
                    &graph.storage.draw_plans,
                    finalized_global_state_key,
                    self.ctx.wgpu_ctx,
                    &self.ctx.camera,
                );
                for plan in gpu_scene_family_prepare_plans {
                    if let Err(reason) = finalize_gpu_scene_multi_bin_family(
                        &mut prepare_ctx,
                        &mut graph.storage.draw_plans,
                        self.ctx.gpu_scene_render,
                        plan,
                        self.ctx.wgpu_ctx,
                        &self.ctx.camera,
                    ) {
                        rdg_diagnostics.mark_gpu_driven_failed(finalize_failure_reason(reason));
                        return;
                    }
                }

                {
                    #[cfg(feature = "rdg_bench_support")]
                    let _timing =
                        rdg_diagnostics.detail_timer(RdgDetailPhase::PipelineFeatureFinalize);
                    for &pass_idx in &graph.storage.execution_queue {
                        // SAFETY: This is the second serial barrier over the same
                        // live arena nodes; no competing mutable reference exists.
                        let pass = unsafe { graph.storage.passes[pass_idx].get_pass_mut() };
                        pass.finalize_prepare(&mut prepare_ctx);
                    }
                }

                // Execute will consume only these immutable, contiguous
                // Direct streams. No shader/cache/asset lookup occurs here.
                {
                    #[cfg(feature = "rdg_bench_support")]
                    let _timing = rdg_diagnostics.detail_timer(RdgDetailPhase::PipelineFinalize);
                    prepare_ctx.finalize_scene_draws(&mut graph.storage.draw_plans);
                }
            }

            graph
                .storage
                .scene_submissions
                .assert_all_live_finalized(&graph.storage.draw_plans);

            #[cfg(feature = "3dgs")]
            if let Some(snapshot) = gaussian_snapshot {
                graph.release_gaussian_frame_batch(snapshot.batch_id());
            }

            // Execute consumes only finalized DrawCommands. Release exact CPU
            // asset/interface owners as soon as every live plan is baked;
            // DrawPlanStore continues to own the physical WGPU handles.
            graph.storage.sample_retained_capacity();
            graph.storage.release_prepare_owners();

            rdg_diagnostics.enter_phase(RdgFramePhase::RecordCommands);

            // ─── 3d. Execute ───────────────────────────────────────────────

            let mut encoder =
                self.ctx
                    .wgpu_ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Unified Encoder"),
                    });
            #[cfg(feature = "rdg_bench_support")]
            let mut gpu_timing_sample = self.ctx.rdg_gpu_timings.begin_sample(
                self.ctx.frame_time.frame_count,
                graph.storage.execution_queue.len(),
            );
            #[cfg(feature = "rdg_bench_support")]
            if let Some(sample) = gpu_timing_sample.as_mut() {
                sample.write_start(&mut encoder);
            }

            {
                // Pass storage is frozen from topology compilation through
                // execution. Keep a direct pointer so the immutable execution
                // facades may borrow their disjoint graph-owned stores while
                // each arena-backed PassNode is visited mutably exactly once.
                let passes = graph.storage.passes.as_mut_ptr();
                let finalized_global_bind_group = graph.finalized_global_bind_group().cloned();
                let mut execute_ctx = ExecuteContext {
                    resources: &graph.storage.resources,
                    pool: self.ctx.transient_pool,
                    persistent_resources: self.ctx.history_registry,
                    persistent_buffers: PersistentBufferResolver::new(
                        self.ctx.persistent_buffer_registry,
                        &graph.storage.persistent_buffer_materializations,
                    ),
                    persistent_buffer_commits: graph.persistent_buffer_commit_tracker(),
                    asset_textures: graph.storage.asset_texture_imports.resolver(),
                    scene_buffers: graph.storage.scene_buffer_imports.resolver(),
                    device: &self.ctx.wgpu_ctx.device,
                    queue: &self.ctx.wgpu_ctx.queue,
                    pipeline_cache: self.ctx.pipeline_cache,
                    prepared_bind_groups: &graph.storage.prepared_bind_groups,
                    mipmap_generator: &*mipmap_generator,
                    mipmap_requests: MipmapRequestResolver::new(&graph.storage.mipmap_batches),
                    mipmap_encode_coordinator: &graph.storage.mipmap_encode_coordinator,
                    scene_submissions: SceneSubmissionResolver::new(
                        &graph.storage.scene_submissions,
                        &graph.storage.draw_plans,
                    )
                    .with_prepared_bind_groups(&graph.storage.prepared_bind_groups),
                    global_bind_group: finalized_global_bind_group.as_ref(),
                    wgpu_ctx: &*self.ctx.wgpu_ctx,
                    current_timeline_index: 0,
                };

                for (timeline_index, &pass_idx) in graph.storage.execution_queue.iter().enumerate()
                {
                    execute_ctx.current_timeline_index = timeline_index;
                    // SAFETY: topology compilation freezes `passes`; every
                    // execution-queue index is valid and visited serially.
                    let pass = unsafe { &mut *passes.add(pass_idx) };
                    #[cfg(debug_assertions)]
                    encoder.push_debug_group(pass.name);
                    // SAFETY: The graph/arena remain alive for the complete
                    // execute phase and each queue entry is visited serially.
                    unsafe { pass.get_pass_mut() }.execute(&execute_ctx, &mut encoder);
                    rdg_diagnostics.record_executed_pass();
                    #[cfg(feature = "rdg_bench_support")]
                    if let Some(sample) = gpu_timing_sample.as_mut() {
                        sample.write_pass_end(&mut encoder, pass.name);
                    }
                    #[cfg(debug_assertions)]
                    encoder.pop_debug_group();
                }
            }
            #[cfg(feature = "rdg_bench_support")]
            if let Some(sample) = gpu_timing_sample.as_ref() {
                sample.encode_resolve(&mut encoder);
            }

            // ━━━ 4. Submit & Present ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            rdg_diagnostics.enter_phase(RdgFramePhase::Submit);
            let Ok(submission_receipt) =
                graph.submit_command_buffer(&self.ctx.wgpu_ctx.queue, encoder.finish())
            else {
                rdg_diagnostics.mark_command_encoding_failed(
                    RdgCommandEncodingFailure::PersistentBufferWriteBatchIncomplete,
                );
                return;
            };
            // Queue acceptance is the transactional boundary for retained
            // world-transform and emitted morph history. Publish/commit both
            // immediately after submit, before diagnostics, timing, policy or
            // any other fallible/debug-only post-submit bookkeeping. The
            // actual RenderWorld settle remains deferred until the next
            // extraction because graph payloads still borrow its records.
            self.ctx.extracted_scene.mark_world_history_submitted();
            self.ctx.extracted_scene.commit_prepared_draw_history(
                self.ctx.scene,
                self.ctx.render_lists,
                &graph.storage.draw_plans,
            );
            rdg_diagnostics.mark_queue_submitted();
            #[cfg(feature = "rdg_bench_support")]
            if let Some(sample) = gpu_timing_sample.take() {
                sample.map_after_submit();
            }
            if let Some(token) = gpu_policy_submission_token {
                let committed = self.ctx.gpu_driven_policy.commit_submitted(token);
                debug_assert!(
                    committed,
                    "submitted GPU-driven Auto decision lost its policy epoch/state"
                );
            }
            rdg_diagnostics.enter_phase(RdgFramePhase::Publish);
            graph.commit_submitted_asset_texture_mipmaps();
            let persistent_buffers_published = if graph.has_persistent_buffer_commit_groups() {
                graph.commit_submitted_persistent_buffers(
                    self.ctx.persistent_buffer_registry,
                    submission_receipt,
                )
            } else {
                Ok(())
            };
            if persistent_buffers_published.is_err() {
                rdg_diagnostics.mark_post_submit_publication_failed(
                    RdgPostSubmitPublicationFailure::PersistentBufferStateDrift,
                );
            }
            if let Some(batch) = mipmap_batch {
                let committed = graph.commit_submitted_mipmap_batch(batch);
                debug_assert!(
                    committed,
                    "live asset mipmap batch was not encoded before submit"
                );
            }
            let cache_publications =
                graph.commit_submitted_persistent_resources(self.ctx.history_registry);
            self.ctx
                .environment_cache
                .observe_publications(&cache_publications);
            graph.run_after_submit_hooks();

            // `ResourceKind::External` stores a raw pointer to this view.
            // Make the unsafe lifetime contract explicit instead of relying on
            // incidental local drop order, then release the view before present.
            drop(graph);
            if persistent_buffers_published.is_ok()
                && let Some(probe) = gpu_scene_publication_probe
            {
                // Generic publication success means every *live* commit group
                // was published; a fully DPE-dead group is intentionally
                // skipped. The exact four-table probe is the authority for
                // whether GPU Scene may drain its dirty/quarantine journals.
                let _ = gpu_scene_publication_acknowledger
                    .as_mut()
                    .expect("GPU Scene publication probe lost its acknowledger")
                    .acknowledge_if_published(
                        self.ctx.persistent_buffer_registry,
                        &probe,
                        gpu_scene_live_submission,
                    );
            }
            drop(surface_view_tracked);
        };

        if let Some(output) = surface_output {
            rdg_diagnostics.enter_phase(RdgFramePhase::Present);
            present_then_reconfigure_if_needed(
                output,
                reconfigure_after_present,
                |output| {
                    output.present();
                    rdg_diagnostics.mark_presented();
                },
                || {
                    if let (Some(surface), Some(config)) =
                        (&self.ctx.wgpu_ctx.surface, &self.ctx.wgpu_ctx.config)
                    {
                        surface.configure(&self.ctx.wgpu_ctx.device, config);
                    }
                },
            );
        }
        rdg_diagnostics.mark_completed();
    }
}

fn add_gpu_scene_shadow_family<'a>(
    graph: &mut RenderGraph<'a>,
    cull_feature: &'a GpuSceneCullFeature,
    render_lists: &RenderLists,
    lights: &'a [ShadowLightInstance],
    patched: &GpuSceneTableSetPatchOutput<'a>,
    projection: GpuSceneMultiBinProjection,
    active_scene_id: u32,
    required_instance_flags: u32,
    pass_name: &'static str,
    prepare_plans: &mut SmallVec<[GpuSceneMultiBinFamilyPreparePlan<'a>; 4]>,
) -> &'a [SceneSubmissionPlanId] {
    if lights.is_empty() {
        return graph.alloc_slice(&[]);
    }

    let views = graph.alloc_slice_fill_with(lights.len(), |light_index| {
        let light = &lights[light_index];
        let frustum_planes = std::array::from_fn(|index| light.frustum.planes()[index].to_array());
        GpuSceneCullViewDescriptor::new(frustum_planes)
    });

    let culled = cull_feature.add_multi_bin_to_graph(
        graph,
        patched.instances,
        patched.batches,
        pass_name,
        GpuSceneMultiBinCullDescriptor::new(
            active_scene_id,
            patched.instance_slot_span,
            patched.batch_slot_span,
            projection.storage_alignment,
            projection.table_set.visible_view_stride,
            views,
            patched.active_batches,
        )
        .with_required_instance_flags(required_instance_flags),
    );
    let indirect_span = u64::from(patched.batch_slot_span)
        .checked_mul(GPU_SCENE_INDEXED_ARGUMENT_SIZE)
        .expect("GPU Scene Shadow argument window overflowed");
    let reserve_submission =
        |graph: &mut RenderGraph<'a>, view_index: usize| -> SceneSubmissionPlanId {
            let light = &lights[view_index];
            let view_index = u32::try_from(view_index)
                .expect("validated GPU Scene Shadow view index exceeds u32");
            let first = culled
                .view_batch(view_index, 0)
                .expect("packed GPU Scene Shadow Cull lost its first batch");
            let indirect_window_start = u64::from(view_index)
                .checked_mul(indirect_span)
                .expect("GPU Scene Shadow argument window base overflowed");
            let key = SceneSubmissionKey::Shadow {
                light_id: light.light_id,
                layer_index: light.view_layer_index,
            };
            let has_cpu_cohort = render_lists
                .shadow_queues
                .get(&(light.light_id, light.view_layer_index))
                .is_some_and(|queue| !queue.is_empty());
            if has_cpu_cohort {
                graph.reserve_hybrid_gpu_scene_multi_bin_submission(
                    key,
                    patched.candidates[0].representative,
                    patched.instances,
                    patched.instances,
                    culled.visible_instances,
                    culled.indirect_arguments,
                    first.indirect_offset,
                    indirect_window_start,
                    indirect_span,
                )
            } else {
                graph.reserve_gpu_scene_multi_bin_submission(
                    key,
                    patched.candidates[0].representative,
                    patched.instances,
                    patched.instances,
                    culled.visible_instances,
                    culled.indirect_arguments,
                    first.indirect_offset,
                    indirect_window_start,
                    indirect_span,
                )
            }
        };

    // Reserve view zero first, then initialize the exact-size arena slice
    // with that valid ID before overwriting the remaining slots in place.
    // This keeps arbitrary light counts off the heap while exposing the same
    // contiguous submission slice expected by ShadowPass.
    let first_submission = reserve_submission(graph, 0);
    let submissions = graph.alloc_slice_fill_copy_mut(lights.len(), first_submission);
    for (view_index, submission) in submissions.iter_mut().enumerate().skip(1) {
        *submission = reserve_submission(graph, view_index);
    }
    let submissions: &'a [SceneSubmissionPlanId] = submissions;
    prepare_plans.push(GpuSceneMultiBinFamilyPreparePlan {
        submissions,
        candidates: patched.candidates,
        instance_table: patched.instances,
        culled,
        target: GpuSceneSubmissionTarget::Shadow,
    });
    submissions
}

fn reserve_cpu_shadow_family<'a>(
    graph: &mut RenderGraph<'a>,
    lights: &'a [ShadowLightInstance],
) -> &'a [SceneSubmissionPlanId] {
    if lights.is_empty() {
        return graph.alloc_slice(&[]);
    }
    let reserve = |graph: &mut RenderGraph<'a>, index: usize| {
        let light = &lights[index];
        graph.reserve_scene_submission(SceneSubmissionKey::Shadow {
            light_id: light.light_id,
            layer_index: light.view_layer_index,
        })
    };
    let first = reserve(graph, 0);
    let submissions = graph.alloc_slice_fill_copy_mut(lights.len(), first);
    for (index, submission) in submissions.iter_mut().enumerate().skip(1) {
        *submission = reserve(graph, index);
    }
    submissions
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    fn gpu_driven_capabilities(
        limits: wgpu::Limits,
        indirect_first_instance: bool,
    ) -> GpuDrivenCapabilities {
        let features = if indirect_first_instance {
            wgpu::Features::INDIRECT_FIRST_INSTANCE
        } else {
            wgpu::Features::empty()
        };
        let downlevel = wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::COMPUTE_SHADERS
                | wgpu::DownlevelFlags::INDIRECT_EXECUTION
                | wgpu::DownlevelFlags::VERTEX_STORAGE,
            ..Default::default()
        };
        GpuDrivenCapabilities::from_parts(features, features, downlevel, limits.clone(), limits)
    }

    fn one_multi_bin_family() -> GpuSceneMultiBinWorkloadFacts {
        GpuSceneMultiBinWorkloadFacts {
            instance_table_bytes: 256,
            geometry_table_bytes: 256,
            material_table_bytes: 256,
            batch_table_bytes: 256,
            instance_slot_span: 1,
            batch_slot_span: 1,
            active_bin_count: 1,
            visible_view_stride_bytes: 256,
            view_count: 1,
        }
    }

    #[test]
    fn late_target_descriptor_enforces_shape_and_backend_format_contract() {
        let desc = TextureDesc::new_2d(
            1280,
            720,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );

        assert!(render_target_metadata_matches_desc(
            desc.size,
            1,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Bgra8Unorm,
            desc,
            RenderTargetFormatContract::SurfaceViewFamily,
        ));
        assert!(!render_target_metadata_matches_desc(
            desc.size,
            1,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Bgra8Unorm,
            desc,
            RenderTargetFormatContract::Exact,
        ));
        assert!(!render_target_metadata_matches_desc(
            wgpu::Extent3d {
                width: 1920,
                height: 1080,
                depth_or_array_layers: 1,
            },
            1,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Bgra8Unorm,
            desc,
            RenderTargetFormatContract::SurfaceViewFamily,
        ));
        assert!(!render_target_metadata_matches_desc(
            desc.size,
            2,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Bgra8Unorm,
            desc,
            RenderTargetFormatContract::SurfaceViewFamily,
        ));
        assert!(!render_target_metadata_matches_desc(
            desc.size,
            1,
            4,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Bgra8Unorm,
            desc,
            RenderTargetFormatContract::SurfaceViewFamily,
        ));
        assert!(!render_target_metadata_matches_desc(
            desc.size,
            1,
            1,
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::Bgra8Unorm,
            desc,
            RenderTargetFormatContract::SurfaceViewFamily,
        ));
        assert!(!render_target_metadata_matches_desc(
            desc.size,
            1,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba8Unorm,
            desc,
            RenderTargetFormatContract::SurfaceViewFamily,
        ));
    }

    #[test]
    fn multi_bin_preflight_accepts_boundaries_and_rejects_invalid_families() {
        let capabilities = gpu_driven_capabilities(wgpu::Limits::default(), false);
        assert!(gpu_scene_multi_bin_workload_supported(
            &capabilities,
            one_multi_bin_family(),
        ));

        let facts = GpuSceneMultiBinWorkloadFacts {
            instance_table_bytes: 512,
            geometry_table_bytes: 256,
            material_table_bytes: 256,
            batch_table_bytes: 512,
            instance_slot_span: 65,
            batch_slot_span: 3,
            active_bin_count: 2,
            visible_view_stride_bytes: 512,
            view_count: 2,
        };
        let exact_limits = wgpu::Limits {
            max_bind_groups: 4,
            max_bindings_per_bind_group: 7,
            max_storage_buffers_per_shader_stage: 6,
            max_uniform_buffers_per_shader_stage: 1,
            max_compute_invocations_per_workgroup: 64,
            max_compute_workgroup_size_x: 64,
            max_compute_workgroup_size_y: 1,
            max_compute_workgroup_size_z: 1,
            max_compute_workgroups_per_dimension: 2,
            max_buffer_size: 1536,
            max_storage_buffer_binding_size: 1024,
            max_uniform_buffer_binding_size: 32,
            min_storage_buffer_offset_alignment: 256,
            ..wgpu::Limits::default()
        };
        assert!(gpu_scene_multi_bin_workload_supported(
            &gpu_driven_capabilities(exact_limits.clone(), false),
            facts,
        ));

        let below_buffer = wgpu::Limits {
            max_buffer_size: 1535,
            ..exact_limits.clone()
        };
        assert!(!gpu_scene_multi_bin_workload_supported(
            &gpu_driven_capabilities(below_buffer, false),
            facts,
        ));
        let below_storage_binding = wgpu::Limits {
            max_storage_buffer_binding_size: 1023,
            ..exact_limits.clone()
        };
        assert!(!gpu_scene_multi_bin_workload_supported(
            &gpu_driven_capabilities(below_storage_binding, false),
            facts,
        ));
        let below_uniform_binding = wgpu::Limits {
            max_uniform_buffer_binding_size: 31,
            ..exact_limits.clone()
        };
        assert!(!gpu_scene_multi_bin_workload_supported(
            &gpu_driven_capabilities(below_uniform_binding, false),
            facts,
        ));
        let below_layout = wgpu::Limits {
            max_bindings_per_bind_group: 6,
            ..exact_limits.clone()
        };
        assert!(!gpu_scene_multi_bin_workload_supported(
            &gpu_driven_capabilities(below_layout, false),
            facts,
        ));
        let below_storage_slots = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 5,
            ..exact_limits.clone()
        };
        assert!(!gpu_scene_multi_bin_workload_supported(
            &gpu_driven_capabilities(below_storage_slots, false),
            facts,
        ));
        let below_group_2_contract = wgpu::Limits {
            max_bind_groups: 3,
            ..exact_limits.clone()
        };
        assert!(!gpu_scene_multi_bin_workload_supported(
            &gpu_driven_capabilities(below_group_2_contract, false),
            facts,
        ));
        let below_dispatch = wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..exact_limits
        };
        assert!(!gpu_scene_multi_bin_workload_supported(
            &gpu_driven_capabilities(below_dispatch, false),
            facts,
        ));

        let limits = wgpu::Limits {
            max_buffer_size: 4096,
            max_storage_buffer_binding_size: 4096,
            ..wgpu::Limits::default()
        };
        let bounded_capabilities = gpu_driven_capabilities(limits, false);

        let mut overflow = one_multi_bin_family();
        overflow.batch_slot_span = u32::MAX;
        overflow.view_count = 2;
        assert!(!gpu_scene_multi_bin_workload_supported(
            &bounded_capabilities,
            overflow,
        ));

        for member in 0..4 {
            let mut oversized = one_multi_bin_family();
            match member {
                0 => oversized.instance_table_bytes = 4097,
                1 => oversized.geometry_table_bytes = 4097,
                2 => oversized.material_table_bytes = 4097,
                3 => oversized.batch_table_bytes = 4097,
                _ => unreachable!(),
            }
            assert!(!gpu_scene_multi_bin_workload_supported(
                &bounded_capabilities,
                oversized,
            ));
        }

        let mut aggregate_oversized = one_multi_bin_family();
        aggregate_oversized.instance_table_bytes = 1025;
        aggregate_oversized.geometry_table_bytes = 1024;
        aggregate_oversized.material_table_bytes = 1024;
        aggregate_oversized.batch_table_bytes = 1024;
        assert!(
            [
                aggregate_oversized.instance_table_bytes,
                aggregate_oversized.geometry_table_bytes,
                aggregate_oversized.material_table_bytes,
                aggregate_oversized.batch_table_bytes,
            ]
            .into_iter()
            .all(|size| size <= 4096),
            "the aggregate regression must keep every destination individually legal"
        );
        assert!(!gpu_scene_multi_bin_workload_supported(
            &bounded_capabilities,
            aggregate_oversized,
        ));
    }

    #[test]
    fn portable_multi_view_shadow_does_not_require_indirect_first_instance() {
        let mut facts = one_multi_bin_family();
        facts.instance_slot_span = 1024;
        facts.batch_slot_span = 8;
        facts.active_bin_count = 6;
        facts.visible_view_stride_bytes = 8192;
        facts.view_count = 6;

        let portable = gpu_driven_capabilities(wgpu::Limits::default(), false);
        assert!(!portable.supports_indirect_first_instance());
        assert!(gpu_scene_multi_bin_workload_supported(&portable, facts));
    }

    #[test]
    fn shadow_preflight_fallback_is_scoped_and_required_is_typed() {
        assert_eq!(
            resolve_gpu_scene_shadow_ownership(GpuDrivenMode::Auto, true, true, false),
            Ok(false),
        );
        assert_eq!(
            resolve_gpu_scene_shadow_ownership(GpuDrivenMode::Required, true, true, false),
            Err(RdgGpuDrivenFailure::WorkloadExceedsDeviceLimits),
        );
        assert_eq!(
            resolve_gpu_scene_shadow_ownership(GpuDrivenMode::Auto, true, false, false),
            Ok(false),
        );
        assert_eq!(
            resolve_gpu_scene_shadow_ownership(GpuDrivenMode::Required, true, false, false),
            Err(RdgGpuDrivenFailure::UnsupportedShadowDomain),
        );
        assert_eq!(
            resolve_gpu_scene_shadow_ownership(GpuDrivenMode::Required, false, false, false),
            Ok(false),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::present_then_reconfigure_if_needed;

    #[test]
    fn suboptimal_surface_reconfigures_only_after_present() {
        use std::cell::RefCell;

        let events = RefCell::new(Vec::new());
        present_then_reconfigure_if_needed(
            (),
            true,
            |()| events.borrow_mut().push("present"),
            || events.borrow_mut().push("reconfigure"),
        );
        assert_eq!(*events.borrow(), ["present", "reconfigure"]);

        events.borrow_mut().clear();
        present_then_reconfigure_if_needed(
            (),
            false,
            |()| events.borrow_mut().push("present"),
            || events.borrow_mut().push("reconfigure"),
        );
        assert_eq!(*events.borrow(), ["present"]);
    }
}
