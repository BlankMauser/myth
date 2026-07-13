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
use crate::core::{ResourceManager, WgpuContext};
use crate::graph::ExtractedScene;
use crate::graph::RdgSurfaceAcquireFailure;
use crate::graph::RenderState;
use crate::graph::core::ClusteredScreenBindings;
use crate::graph::core::GraphStorage;
#[cfg(feature = "3dgs")]
use crate::graph::core::gaussian::GaussianFrameResolver;
use crate::graph::core::graph::FrameConfig;
use crate::graph::core::scene_submission::SceneSubmissionResolver;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, FrameArena, GlobalStateMaterializationFailure,
    GlobalStateSources, GraphBlackboard, HistoryRegistry, HookStage, PrepareContext,
    RdgComposerAttemptGuard, RdgDiagnosticsRecorder, RdgFramePhase, RdgRenderTargetKind,
    RdgResourceMaterializationFailure, RenderGraph, SceneBufferMaterializationFailure,
    SceneBufferSources, SceneDrawPrepareContext, TextureDesc, TransientPool, ViewResolver,
};
use crate::graph::extracted::SceneFeatures;
use crate::graph::frame::RenderLists;
#[cfg(feature = "3dgs")]
use crate::graph::passes::GaussianSplattingFeature;
use crate::graph::passes::ibl_compute::EnvironmentCacheSnapshot;
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
use crate::settings::RendererSettings;
use myth_assets::AssetServer;
use myth_resources::texture::TextureSource;
use myth_scene::Scene;
use myth_scene::background::BackgroundMapping;
use myth_scene::camera::RenderCamera;

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
    pub transient_pool: &'a mut TransientPool,
    pub history_registry: &'a mut HistoryRegistry,
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
    /// 1. **Acquire Render Target** — swap-chain back buffer or headless texture
    /// 2. **Build RDG** — register resources, wire passes (compute, shadow,
    ///    scene, post-processing), invoke user hooks
    /// 3. **Compile & Execute** — topological sort, allocate transients,
    ///    prepare, execute, submit
    /// 4. **Present** — present swap-chain (windowed only)
    ///
    /// Consumes `self`; the composer cannot be reused after render.
    pub fn render(mut self) {
        // ━━━ 1. Acquire Render Target ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        let rdg_diagnostics = self.ctx.rdg_diagnostics;
        rdg_diagnostics.enter_phase(RdgFramePhase::AcquireTarget);

        #[cfg(test)]
        if let Some(reason) = self.forced_surface_acquisition_failure {
            rdg_diagnostics.mark_surface_acquisition_failed(reason);
            return;
        }

        let view_format = self.ctx.wgpu_ctx.surface_view_format;

        // Acquire either the swap-chain back buffer or the headless texture view.
        // `surface_output` is `Some` only in windowed mode and holds the
        // `SurfaceTexture` that must be `.present()`ed after submission.
        let (surface_view, width, height, surface_output);
        let mut reconfigure_after_present = false;

        if let Some(surface) = &self.ctx.wgpu_ctx.surface {
            let output = match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(frame) => {
                    rdg_diagnostics.mark_target_acquired(RdgRenderTargetKind::Surface, false);
                    frame
                }
                wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                    rdg_diagnostics.mark_target_acquired(RdgRenderTargetKind::Surface, true);
                    reconfigure_after_present = true;
                    log::debug!(
                        "Acquired suboptimal swap-chain surface; reconfiguring after present"
                    );
                    frame
                }
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
                                log::error!("Validation error while acquiring swap-chain surface");
                            }
                        }
                    }
                    return;
                }
            };

            surface_view = output.texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(view_format),
                ..Default::default()
            });
            width = output.texture.width();
            height = output.texture.height();
            surface_output = Some(output);
        } else if let Some(tex) = &self.ctx.wgpu_ctx.headless_texture {
            rdg_diagnostics.mark_target_acquired(RdgRenderTargetKind::Headless, false);
            surface_view = tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(view_format),
                ..Default::default()
            });
            width = tex.width();
            height = tex.height();
            surface_output = None;
        } else {
            if rdg_diagnostics.mark_no_render_target() {
                log::error!("No render target available (neither surface nor headless texture)");
            }
            return;
        }

        rdg_diagnostics.enter_phase(RdgFramePhase::Cull);

        // ━━━ 2a. Collect CPU-only central draw intents ━━━━━━━━━━━━━━━━━━
        //
        // Intent collection determines topology-affecting transmission and
        // per-view source membership without touching WGPU resources.
        crate::graph::culling::collect_draw_intents(
            self.ctx.extracted_scene,
            self.ctx.render_lists,
            &self.ctx.camera,
        );

        rdg_diagnostics.enter_phase(RdgFramePhase::BuildGraph);

        // ━━━ 2b. Build Unified RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        let surface_desc = TextureDesc::new_2d(
            width,
            height,
            view_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let surface_view_tracked = Tracked::with_id(surface_view, 0);

        // Declared after the tracked external view so reverse drop order keeps
        // the raw external-view pointer valid for the graph's whole lifetime.
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

        let surface_out =
            graph.import_external_resource("Surface_View", surface_desc, &surface_view_tracked);

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
            for plan in &draw_texture_dependencies {
                let mut dependencies = Vec::with_capacity(plan.textures.len());
                for &planned in &plan.textures {
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
                    dependencies.push(ready);
                }
                graph_ctx
                    .graph
                    .set_scene_submission_texture_dependencies(plan.key.into(), dependencies);
            }

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

            let shadow_output = if self.ctx.extracted_scene.has_shadow_casters() {
                let snapshot = self.ctx.shadow_pass.extract(
                    self.ctx.render_lists,
                    self.ctx.extracted_scene,
                    self.ctx.frame_arena,
                );
                graph_ctx.with_group("Shadow", |c| self.ctx.shadow_pass.add_to_graph(c, snapshot))
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
                        let prepass_out = self.ctx.prepass.add_to_graph(
                            c,
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
                        let opaque_out = self.ctx.opaque_pass.add_to_graph(
                            c,
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
                    self.ctx.simple_forward_pass.add_to_graph(
                        c,
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

            graph.compile(self.ctx.transient_pool, &self.ctx.wgpu_ctx.device);
            let (scene_buffers, global_buffers) = {
                let resource_manager = &mut *self.ctx.resource_manager;
                let mut uploads = BufferUploadContext::new(
                    &self.ctx.wgpu_ctx.device,
                    &self.ctx.wgpu_ctx.queue,
                    resource_manager.frame_index,
                    &mut resource_manager.resources,
                    &resource_manager.internal_textures,
                );
                let scene_buffers = graph.materialize_live_scene_buffers(
                    SceneBufferSources {
                        light_metadata: &self.ctx.extracted_scene.local_light_metadata_buffer,
                        light_storage: &self.ctx.extracted_scene.local_light_storage_buffer,
                    },
                    &mut uploads,
                );
                let global_buffers = if scene_buffers.is_ok() {
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
            graph.materialize_live_persistent_textures(
                self.ctx.history_registry,
                &self.ctx.wgpu_ctx.device,
            );
            graph.materialize_live_asset_textures(self.ctx.assets, self.ctx.resource_manager);

            // Group 0 keeps a stable layout/WGSL interface, but its PMREM
            // physical view is selected only after DPE has decided whether a
            // same-frame cache writer is live and materialized. The graph read
            // edge on `env_dependency_pmrem` orders that writer before every
            // scene pass using this refreshed bind group.
            let pmrem_override = environment_pmrem.and_then(|(node, texture, valid)| {
                let live = graph.storage.resources[node.index() as usize].first_use != usize::MAX;
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
                    .materialized_global_buffer_refs()
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

            // ─── 3a. Materialize the DPE-live central source union ───────
            //
            // Draw-plan demand marks are the sole liveness authority. Main,
            // Prepass, and any number of shadow views share one source slot;
            // dead plan branches never allocate model data, geometry, object
            // bindings, or pipelines.
            let live_frame_sources = self
                .ctx
                .render_lists
                .live_frame_sources(graph.storage.draw_plans.live_keys());
            graph.storage.material_plans.prepare_live(
                self.ctx.assets,
                self.ctx.resource_manager,
                &live_frame_sources,
            );
            graph.storage.geometry_frames.capture_live(
                self.ctx.assets,
                self.ctx.extracted_scene,
                self.ctx.render_lists,
                &live_frame_sources,
            );
            graph.storage.geometry_frames.prepare_live(
                self.ctx.assets,
                self.ctx.resource_manager,
                &mut graph.storage.draw_plans,
            );
            let material_plans = graph.storage.material_plans.resolver();
            let geometry_frames = graph.storage.geometry_frames.resolver();
            self.ctx.extracted_scene.prepare_draw_resources(
                self.ctx.scene,
                self.ctx.assets,
                self.ctx.resource_manager,
                &live_frame_sources,
                self.ctx.render_lists,
                geometry_frames,
                material_plans,
                &mut graph.storage.object_interfaces,
                &mut graph.storage.prepared_draws,
                &mut graph.storage.draw_plans,
            );

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
                        &graph.storage.scene_submissions,
                        &mut graph.storage.draw_plans,
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
                };

                for &pass_idx in &graph.storage.execution_queue {
                    // SAFETY: RenderGraph still owns the live FrameArena and
                    // the execution queue is traversed sequentially, so this
                    // pass has no competing mutable reference.
                    let pass = unsafe { graph.storage.passes[pass_idx].get_pass_mut() };
                    pass.prepare(&mut prepare_ctx);
                    rdg_diagnostics.record_prepared_pass();
                }
            }

            rdg_diagnostics.enter_phase(RdgFramePhase::FinalizeDrawPlans);

            #[cfg(feature = "3dgs")]
            if let Some(snapshot) = gaussian_snapshot {
                graph.release_gaussian_frame_batch(snapshot.batch_id());
            }

            // ─── 3c. Finalize live draw plans ──────────────────────────────
            //
            // DPE-selected PassRecords have already marked their reserved
            // plan IDs live. Resolve only those plans into graph-attempt-owned
            // physical resource tables and compact command ranges.
            crate::graph::bake::finalize_live_draw_plans(
                &mut graph.storage.draw_plans,
                &graph.storage.prepared_draws,
                graph.storage.geometry_frames.resolver(),
                graph.storage.material_plans.resolver(),
                graph.storage.object_interfaces.resolver(),
                self.ctx.render_lists,
                self.ctx.extracted_scene,
                self.ctx.render_state,
                self.ctx.wgpu_ctx,
                self.ctx.resource_manager,
                self.ctx.pipeline_cache,
                self.ctx.shader_manager,
                &self.ctx.camera,
            );
            // Execute consumes only finalized DrawCommands. Release exact CPU
            // asset/interface owners as soon as every live plan is baked;
            // DrawPlanStore continues to own the physical WGPU handles.
            graph.storage.sample_retained_capacity();
            graph.storage.prepared_draws.clear();
            graph.storage.object_interfaces.clear();
            graph.storage.geometry_frames.clear();
            graph.storage.material_plans.clear();
            graph.storage.asset_texture_plans.clear();

            rdg_diagnostics.enter_phase(RdgFramePhase::RecordCommands);

            // ─── 3d. Execute ───────────────────────────────────────────────

            let mut encoder =
                self.ctx
                    .wgpu_ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Unified Encoder"),
                    });

            {
                let finalized_global_bind_group = graph.finalized_global_bind_group().cloned();
                let mut execute_ctx = ExecuteContext {
                    resources: &graph.storage.resources,
                    pool: self.ctx.transient_pool,
                    persistent_resources: self.ctx.history_registry,
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
                    ),
                    global_bind_group: finalized_global_bind_group.as_ref(),
                    wgpu_ctx: &*self.ctx.wgpu_ctx,
                    current_timeline_index: 0,
                };

                for (timeline_index, &pass_idx) in graph.storage.execution_queue.iter().enumerate()
                {
                    execute_ctx.current_timeline_index = timeline_index;
                    #[cfg(debug_assertions)]
                    encoder.push_debug_group(graph.storage.passes[pass_idx].name);
                    // SAFETY: The graph/arena remain alive for the complete
                    // execute phase and each queue entry is visited serially.
                    unsafe { graph.storage.passes[pass_idx].get_pass_mut() }
                        .execute(&execute_ctx, &mut encoder);
                    rdg_diagnostics.record_executed_pass();
                    #[cfg(debug_assertions)]
                    encoder.pop_debug_group();
                }
            }

            // ━━━ 4. Submit & Present ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            rdg_diagnostics.enter_phase(RdgFramePhase::Submit);
            self.ctx.wgpu_ctx.queue.submit(Some(encoder.finish()));
            rdg_diagnostics.mark_queue_submitted();
            rdg_diagnostics.enter_phase(RdgFramePhase::Publish);
            // Morph history is a submission result, not a begin-frame side
            // effect. Once the queue accepts these commands, advance the
            // previous-weight baseline even if later publication bookkeeping
            // reports an internal error.
            self.ctx.extracted_scene.commit_prepared_draw_history(
                self.ctx.scene,
                self.ctx.render_lists,
                &graph.storage.draw_plans,
            );
            graph.commit_submitted_asset_texture_mipmaps();
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
        };

        // Dropping the graph invalidates its arena-backed node slots, releases
        // frame-attempt owners, and ends the raw external-view borrow before
        // the tracked surface view is released.
        drop(graph);
        drop(surface_view_tracked);

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
