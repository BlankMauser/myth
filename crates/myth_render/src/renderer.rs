//! Rendering System
//!
//! The main [`Renderer`] struct orchestrating GPU rendering operations.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use smallvec::SmallVec;

use crate::core::binding::GlobalBindGroupCache;
use crate::core::gpu::Tracked;
use crate::graph::composer::ComposerContext;
use crate::graph::core::allocator::TransientPool;
use crate::graph::core::arena::FrameArena;
use crate::graph::core::graph::{GraphStorage, RdgRetainedCapacityStats};
use crate::graph::core::persistent::TEMPORAL_DOMAIN_TTL_FRAMES;
use crate::graph::core::{
    HistoryRegistry, PersistentBufferRegistry, RdgDiagnosticsRecorder, RdgFrameDiagnostics,
    RdgFramePhase, RdgRenderWorldRebuildReason, RdgRenderWorldSnapshot, RdgRendererAttemptGuard,
};
#[cfg(feature = "rdg_bench_support")]
use crate::graph::core::{RdgGpuFrameTimings, RdgGpuTimingRecorder};
use crate::graph::frame::{RenderFrameFeatureConfig, RenderLists};
use crate::graph::gpu_driven_policy::GpuDrivenAutoPolicy;
use crate::graph::gpu_scene::GpuScene;
use crate::graph::gpu_scene_capacity::GpuSceneCapacityLedger;
use crate::graph::gpu_scene_submission::GpuSceneRenderFeature;
#[cfg(feature = "debug_view")]
use crate::graph::passes::DebugViewFeature;
#[cfg(feature = "3dgs")]
use crate::graph::passes::GaussianSplattingFeature;
use crate::graph::passes::gpu_scene::GpuSceneCullFeature;
use crate::graph::passes::{
    AtmosphereFeature, BloomFeature, CasFeature, ClusteredLightingFeature, EnvironmentCacheFeature,
    EquirectToCubeFeature, FxaaFeature, HiZFeature, IblComputeFeature, MipmapFeature,
    MsaaSyncFeature, OpaqueFeature, PrepassFeature, PresentFeature, ShadowFeature,
    SimpleForwardFeature, SkyboxFeature, SsaoFeature, SsgiFeature, SsrFeature, SsssFeature,
    TaaFeature, ToneMappingFeature, TransmissionCopyFeature, TransparentFeature,
};
use crate::graph::render_world::RenderWorldFullRebuildReason;
use myth_assets::AssetServer;
use myth_core::Result;
use myth_resources::uniforms::GpuLightStorage;
use myth_scene::Scene;
use myth_scene::background::BackgroundMode;
use myth_scene::camera::RenderCamera;

use crate::core::{GpuDrivenCapabilities, ResourceManager, WgpuContext};
use crate::graph::extracted::SceneFeatures;
use crate::graph::{FrameComposer, RenderFrame};
use crate::pipeline::{
    ColorTargetKey, ComputePipelineId, ComputePipelineKey, DepthStencilKey, FullscreenPipelineKey,
    MultisampleKey, PipelineCache, RenderPipelineId, ShaderCompilationOptions, ShaderManager,
    ShaderSource,
};
use crate::settings::{
    ClusteredShadingMode, GpuDrivenMode, RenderPath, RendererInitConfig, RendererSettings,
};

#[inline]
fn rdg_render_world_rebuild_reason(
    reason: RenderWorldFullRebuildReason,
) -> Option<RdgRenderWorldRebuildReason> {
    Some(match reason {
        RenderWorldFullRebuildReason::None => return None,
        RenderWorldFullRebuildReason::Cold | RenderWorldFullRebuildReason::SceneInitial => {
            RdgRenderWorldRebuildReason::InitialMaterialization
        }
        RenderWorldFullRebuildReason::SceneChanged => {
            RdgRenderWorldRebuildReason::SceneIdentityDrift
        }
        RenderWorldFullRebuildReason::SceneExplicit => {
            RdgRenderWorldRebuildReason::ExplicitInvalidation
        }
        RenderWorldFullRebuildReason::SceneOverflow => {
            RdgRenderWorldRebuildReason::MutationOverflow
        }
        RenderWorldFullRebuildReason::SceneUntrackedStorage => {
            RdgRenderWorldRebuildReason::UntrackedSceneStorage
        }
        RenderWorldFullRebuildReason::SceneCursorOrIdentityDrift => {
            RdgRenderWorldRebuildReason::MutationCursorDrift
        }
        RenderWorldFullRebuildReason::SceneUntrackedHierarchy => {
            RdgRenderWorldRebuildReason::UntrackedHierarchy
        }
        RenderWorldFullRebuildReason::TopologyChanged => {
            RdgRenderWorldRebuildReason::TopologyChanged
        }
        RenderWorldFullRebuildReason::AssetAuthorityDrift => {
            RdgRenderWorldRebuildReason::AssetAuthorityDrift
        }
        RenderWorldFullRebuildReason::AssetEpochFallback => {
            RdgRenderWorldRebuildReason::AssetEpochChanged
        }
        RenderWorldFullRebuildReason::GpuFactsUpgrade => {
            RdgRenderWorldRebuildReason::GpuFactProfileChanged
        }
        RenderWorldFullRebuildReason::JournalOverflow => {
            RdgRenderWorldRebuildReason::ConsumerJournalOverflow
        }
    })
}

fn build_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layouts: &[&Tracked<wgpu::BindGroupLayout>],
    label: &str,
) -> (
    wgpu::PipelineLayout,
    SmallVec<[Tracked<wgpu::BindGroupLayout>; 4]>,
) {
    let raw_layouts = bind_group_layouts
        .iter()
        .map(|layout| {
            let raw_layout: &wgpu::BindGroupLayout = layout;
            Some(raw_layout)
        })
        .collect::<Vec<_>>();
    let tracked_layouts = bind_group_layouts
        .iter()
        .map(|layout| (*layout).clone())
        .collect::<SmallVec<[Tracked<wgpu::BindGroupLayout>; 4]>>();
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &raw_layouts,
        immediate_size: 0,
    });

    (pipeline_layout, tracked_layouts)
}

/// The main renderer responsible for GPU rendering operations.
///
/// The renderer manages the complete rendering pipeline including:
/// - GPU context (device, queue, surface)
/// - Resource management (buffers, textures, bind groups)
/// - Pipeline caching (shader compilation, PSO creation)
/// - Frame rendering (scene extraction, command submission)
///
/// # Lifecycle
///
/// 1. Create with [`Renderer::new`] (no GPU resources allocated)
/// 2. Initialize GPU with [`Renderer::init`]
/// 3. Render frames with [`Renderer::begin_frame`]
/// 4. Clean up with [`Renderer::maybe_prune`]
pub struct Renderer {
    size: (u32, u32),
    init_config: RendererInitConfig,
    settings: RendererSettings,
    context: Option<RendererState>,
}

/// Internal renderer state
struct RendererState {
    wgpu_ctx: WgpuContext,
    resource_manager: ResourceManager,
    pipeline_cache: PipelineCache,
    shader_manager: ShaderManager,

    render_frame: RenderFrame,
    /// Render lists (separated from `render_frame` to avoid borrow conflicts)
    render_lists: RenderLists,
    // /// Frame blackboard (cross-pass transient data communication, cleared each frame)
    // blackboard: FrameBlackboard,
    global_bind_group_cache: GlobalBindGroupCache,

    // ===== RDG (Declarative Render Graph) =====
    /// Renderer-attempt lifecycle lives beside GraphStorage so Composer can
    /// retain a shared, non-atomic guard while RenderGraph exclusively borrows
    /// topology storage for the frame lifetime.
    pub(crate) rdg_diagnostics: RdgDiagnosticsRecorder,
    #[cfg(feature = "rdg_bench_support")]
    pub(crate) rdg_gpu_timings: RdgGpuTimingRecorder,
    pub(crate) graph_storage: GraphStorage,
    // pub(crate) sampler_registry: SamplerRegistry,
    pub(crate) transient_pool: TransientPool,
    pub(crate) history_registry: HistoryRegistry,
    pub(crate) persistent_buffer_registry: PersistentBufferRegistry,
    pub(crate) gpu_scene_capacity: GpuSceneCapacityLedger,
    /// Persistent CPU authority and static backend state for the first
    /// executable GPU-driven scene-submission slice.
    pub(crate) gpu_scene: GpuScene,
    pub(crate) gpu_scene_cull: GpuSceneCullFeature,
    pub(crate) gpu_scene_render: GpuSceneRenderFeature,
    /// Fixed-capacity, submission-transactional Auto backend policy.
    pub(crate) gpu_driven_policy: GpuDrivenAutoPolicy,
    pub(crate) frame_arena: FrameArena,

    // Post-processing passes
    pub(crate) fxaa_pass: FxaaFeature,
    pub(crate) taa_pass: TaaFeature,
    pub(crate) cas_pass: CasFeature,
    pub(crate) tone_map_pass: ToneMappingFeature,
    pub(crate) present_pass: PresentFeature,
    pub(crate) bloom_pass: BloomFeature,
    pub(crate) ssao_pass: SsaoFeature,
    pub(crate) hiz_pass: HiZFeature,
    pub(crate) ssgi_pass: SsgiFeature,
    pub(crate) ssr_pass: SsrFeature,

    // Scene rendering passes
    pub(crate) prepass: PrepassFeature,
    pub(crate) opaque_pass: OpaqueFeature,
    pub(crate) skybox_pass: SkyboxFeature,
    pub(crate) transparent_pass: TransparentFeature,
    pub(crate) transmission_copy_pass: TransmissionCopyFeature,
    pub(crate) simple_forward_pass: SimpleForwardFeature,
    pub(crate) ssss_pass: SsssFeature,
    pub(crate) msaa_sync_pass: MsaaSyncFeature,

    // Shadow + Compute passes (migrated from old system)
    pub(crate) shadow_pass: ShadowFeature,
    pub(crate) environment_cache: EnvironmentCacheFeature,
    pub(crate) equirect_to_cube_pass: EquirectToCubeFeature,
    pub(crate) ibl_pass: IblComputeFeature,
    pub(crate) atmosphere_pass: AtmosphereFeature,
    pub(crate) clustered_lighting_pass: ClusteredLightingFeature,
    pub(crate) mipmap_pass: MipmapFeature,

    #[cfg(feature = "3dgs")]
    // Gaussian Splatting
    pub(crate) gaussian_splatting_pass: GaussianSplattingFeature,

    // Debug view (compile-time gated)
    #[cfg(feature = "debug_view")]
    pub(crate) debug_view_pass: DebugViewFeature,

    /// Cached staging buffer for synchronous `readback_pixels()`.
    /// Re-used across calls when the required size has not changed.
    cached_readback_buffer: Option<wgpu::Buffer>,
    /// Size (in bytes) of the cached readback buffer.
    cached_readback_buffer_size: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FrameTime {
    pub time: f32,
    pub delta_time: f32,
    pub frame_count: u64,
}

fn max_extracted_light_count(device: &wgpu::Device) -> usize {
    let max_storage_binding_size = device.limits().max_storage_buffer_binding_size as usize;
    let max_gpu_light_count = max_storage_binding_size / std::mem::size_of::<GpuLightStorage>();
    let max_view_position_count = max_storage_binding_size / std::mem::size_of::<[f32; 4]>();
    max_gpu_light_count.min(max_view_position_count)
}

impl Renderer {
    /// Phase 1: Create configuration (no GPU resources yet).
    ///
    /// This only stores the render settings. GPU resources are
    /// allocated when [`init`](Self::init) is called.
    ///
    /// # Arguments
    ///
    /// * `init_config` - Static GPU/device configuration (consumed at init time)
    /// * `settings` - Runtime rendering settings (can be changed later via [`update_settings`](Self::update_settings))
    #[must_use]
    pub fn new(init_config: RendererInitConfig, settings: RendererSettings) -> Self {
        Self {
            init_config,
            settings,
            context: None,
            size: (0, 0),
        }
    }

    /// Returns the current surface size in pixels as `(width, height)`.
    #[inline]
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Phase 2: Initialize GPU context with window handle.
    ///
    /// This method:
    /// 1. Creates the wgpu instance and adapter
    /// 2. Requests a device with required features/limits
    /// 3. Configures the surface for presentation
    /// 4. Initializes resource manager and pipeline cache
    pub async fn init<W>(&mut self, window: W, width: u32, height: u32) -> Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        if self.context.is_some() {
            return Ok(());
        }

        self.size = (width, height);

        let wgpu_ctx =
            WgpuContext::new(window, &self.init_config, &self.settings, width, height).await?;

        self.assemble_state(wgpu_ctx);
        log::info!("Renderer initialized (windowed)");
        Ok(())
    }

    /// Phase 2 (headless): Initialize GPU context without a window.
    ///
    /// Creates an offscreen render target of the specified dimensions. No
    /// window surface is created, making this suitable for server-side
    /// rendering, automated testing, and GPU readback workflows.
    ///
    /// # Arguments
    ///
    /// * `width` — Render target width in pixels.
    /// * `height` — Render target height in pixels.
    /// * `format` — Desired pixel format. Pass `None` for the default
    ///   `Rgba8Unorm` (sRGB). Use `Some(Rgba16Float)` for HDR readback.
    pub async fn init_headless(
        &mut self,
        width: u32,
        height: u32,
        format: Option<myth_resources::PixelFormat>,
    ) -> Result<()> {
        if self.context.is_some() {
            return Ok(());
        }

        self.size = (width, height);

        let wgpu_format = format.map(|f| f.to_wgpu(myth_resources::ColorSpace::Srgb));

        let wgpu_ctx = WgpuContext::new_headless(
            &self.init_config,
            &self.settings,
            width,
            height,
            wgpu_format,
        )
        .await?;

        self.assemble_state(wgpu_ctx);
        log::info!("Renderer initialized (headless {width}×{height})");
        Ok(())
    }

    /// Assembles the internal renderer state from a fully initialised GPU context.
    fn assemble_state(&mut self, wgpu_ctx: WgpuContext) {
        let resource_manager = ResourceManager::new(
            wgpu_ctx.device.clone(),
            wgpu_ctx.queue.clone(),
            self.settings.anisotropy_clamp,
        );

        let render_frame = RenderFrame::new();
        let global_bind_group_cache = GlobalBindGroupCache::new();

        let shadow_pass = ShadowFeature::new(&wgpu_ctx.device);
        let environment_cache = EnvironmentCacheFeature::new();
        let equirect_to_cube_pass = EquirectToCubeFeature::new();
        let ibl_pass = IblComputeFeature::new();
        let mipmap_pass = MipmapFeature::new(&wgpu_ctx.device);
        #[cfg(feature = "rdg_bench_support")]
        let rdg_gpu_timings = RdgGpuTimingRecorder::new(&wgpu_ctx.device, &wgpu_ctx.queue);

        self.context = Some(RendererState {
            wgpu_ctx,
            resource_manager,
            pipeline_cache: PipelineCache::new(),
            shader_manager: ShaderManager::new(),

            render_frame,
            render_lists: RenderLists::new(),
            global_bind_group_cache,

            rdg_diagnostics: RdgDiagnosticsRecorder::default(),
            #[cfg(feature = "rdg_bench_support")]
            rdg_gpu_timings,
            graph_storage: GraphStorage::new(),
            transient_pool: TransientPool::new(),
            history_registry: HistoryRegistry::new(),
            persistent_buffer_registry: PersistentBufferRegistry::new(),
            gpu_scene_capacity: GpuSceneCapacityLedger::new(),
            gpu_scene: GpuScene::new(),
            gpu_scene_cull: GpuSceneCullFeature::new(),
            gpu_scene_render: GpuSceneRenderFeature::new(),
            gpu_driven_policy: GpuDrivenAutoPolicy::new(),
            frame_arena: FrameArena::new(),
            fxaa_pass: FxaaFeature::new(),
            taa_pass: TaaFeature::new(),
            cas_pass: CasFeature::new(),
            tone_map_pass: ToneMappingFeature::new(),
            present_pass: PresentFeature::new(),
            bloom_pass: BloomFeature::new(),
            ssao_pass: SsaoFeature::new(),
            hiz_pass: HiZFeature::new(),
            ssgi_pass: SsgiFeature::new(),
            ssr_pass: SsrFeature::new(),

            prepass: PrepassFeature::new(),
            opaque_pass: OpaqueFeature::new(),
            skybox_pass: SkyboxFeature::new(),
            transparent_pass: TransparentFeature::new(),
            transmission_copy_pass: TransmissionCopyFeature::new(),
            simple_forward_pass: SimpleForwardFeature::new(),
            ssss_pass: SsssFeature::new(),
            msaa_sync_pass: MsaaSyncFeature::new(),

            shadow_pass,
            environment_cache,
            equirect_to_cube_pass,
            ibl_pass,
            atmosphere_pass: AtmosphereFeature::new(),
            clustered_lighting_pass: ClusteredLightingFeature::new(),
            mipmap_pass,

            #[cfg(feature = "3dgs")]
            gaussian_splatting_pass: GaussianSplattingFeature::new(),

            #[cfg(feature = "debug_view")]
            debug_view_pass: DebugViewFeature::new(),

            cached_readback_buffer: None,
            cached_readback_buffer_size: 0,
        });
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.size = (width, height);
        if let Some(state) = &mut self.context {
            state.wgpu_ctx.resize(width, height);
            // Invalidate all cached bind groups — texture views are now stale.
            state.global_bind_group_cache.clear();
        }
    }

    /// Begins building a new frame for rendering.
    ///
    /// Returns a [`FrameComposer`] that provides a chainable API for
    /// configuring the render pipeline via custom pass hooks.
    ///
    /// # Usage
    ///
    /// ```rust,ignore
    /// // Method 1: Use default built-in passes
    /// if let Some(composer) = renderer.begin_frame(scene, camera, assets, time) {
    ///     composer.render();
    /// }
    ///
    /// // Method 2: With custom hooks (e.g., UI overlay)
    /// if let Some(composer) = renderer.begin_frame(scene, camera, assets, time) {
    ///     composer
    ///         .add_custom_pass(HookStage::AfterPostProcess, |graph, bb| {
    ///             ui_pass.target_tex = bb.surface_out;
    ///             graph.add_pass(&mut ui_pass);
    ///         })
    ///         .render();
    /// }
    /// ```
    ///
    /// # Returns
    ///
    /// Returns `Some(FrameComposer)` if frame preparation succeeds,
    /// or `None` if rendering should be skipped (e.g., window size is 0).
    pub fn begin_frame<'a>(
        &'a mut self,
        scene: &'a mut Scene,
        camera: RenderCamera,
        assets: &'a AssetServer,
        frame_time: FrameTime,
    ) -> Option<FrameComposer<'a>> {
        if self.size.0 == 0 || self.size.1 == 0 {
            return None;
        }

        let state = self.context.as_mut()?;
        #[cfg(feature = "rdg_bench_support")]
        state
            .rdg_gpu_timings
            .poll(&state.wgpu_ctx.device, frame_time.frame_count);

        // ── Frame Arena Lifecycle ───────────────────────────────────────
        // Invalidate every raw NodeSlot before resetting the arena that owns
        // its pointee. This also makes an abandoned Composer or a later surface
        // acquisition failure leave an empty, safe graph inspection state.
        state
            .rdg_diagnostics
            .start_renderer_attempt(frame_time.frame_count);
        let frame_attempt = RdgRendererAttemptGuard::new(&state.rdg_diagnostics);
        // Invalidate every arena node before GPU Scene maintenance can clear
        // or shrink slices borrowed by the previous graph's pass payloads.
        state.graph_storage.clear();
        state.gpu_scene.maintain_residency(
            &mut state.persistent_buffer_registry,
            f64::from(frame_time.time),
        );
        state
            .gpu_scene_capacity
            .maintain(&mut state.gpu_scene, &mut state.gpu_scene_render);
        // Reset the arena in O(1) — all previous PassNodes are trivially
        // forgotten (no Drop needed).
        state.frame_arena.reset();

        // Advance the bind-group cache's frame counter for TTL tracking.
        state.global_bind_group_cache.begin_frame();

        // ── Phase 1: Extract a CPU-only frame snapshot ─────────────────

        frame_attempt.enter_phase(RdgFramePhase::Extract);

        let surface_size = state.wgpu_ctx.size();
        let max_extracted_lights = max_extracted_light_count(&state.wgpu_ctx.device);
        let requested_msaa = camera.aa_mode.msaa_sample_count();
        if state.wgpu_ctx.msaa_samples != requested_msaa {
            state.wgpu_ctx.msaa_samples = requested_msaa;
            state.wgpu_ctx.pipeline_settings_version += 1;
        }

        let feature_config = RenderFrameFeatureConfig {
            clustered_shading: self.settings.clustered_shading,
            gpu_driven: self.settings.gpu_driven,
            render_path: state.wgpu_ctx.render_path,
            msaa_samples: state.wgpu_ctx.msaa_samples,
        };
        let has_active_environment =
            matches!(&scene.background.mode, BackgroundMode::Procedural(_))
                || scene.environment.has_env_map();
        let published_pmrem_max_mip = has_active_environment
            .then(|| {
                state
                    .environment_cache
                    .published_pmrem_mip_level_count(scene.id(), &state.history_registry)
                    .map(|mip_count| mip_count.saturating_sub(1) as f32)
            })
            .flatten();
        state.render_frame.extract(
            &mut state.resource_manager,
            scene,
            &camera,
            assets,
            frame_time,
            &mut state.render_lists,
            surface_size,
            max_extracted_lights,
            feature_config,
            published_pmrem_max_mip,
        );
        let render_world_stats = state.render_frame.last_render_world_sync_stats();
        let hierarchy_stats = scene.last_transform_update_stats();
        let mutation_journal = scene.render_mutation_journal_stats();
        let retained_bytes = render_world_stats
            .retained_bytes
            .saturating_add(hierarchy_stats.retained_bytes)
            .saturating_add(mutation_journal.estimated_retained_bytes);
        state
            .rdg_diagnostics
            .record_render_world(RdgRenderWorldSnapshot {
                scene_id: Some(render_world_stats.scene_id),
                revision: render_world_stats.revision,
                resident_scene_count: render_world_stats.resident_domain_count,
                mutation_count: render_world_stats.mutation_count,
                hierarchy_dirty_roots_examined: u32::try_from(hierarchy_stats.dirty_roots_examined)
                    .unwrap_or(u32::MAX),
                hierarchy_dirty_parent_links_examined: u32::try_from(
                    hierarchy_stats.dirty_root_parent_links_examined,
                )
                .unwrap_or(u32::MAX),
                hierarchy_nodes_visited: u32::try_from(hierarchy_stats.visited_nodes)
                    .unwrap_or(u32::MAX),
                hierarchy_nodes_updated: u32::try_from(hierarchy_stats.updated_world_matrices)
                    .unwrap_or(u32::MAX),
                hierarchy_full_traversal: hierarchy_stats.full_traversal,
                render_item_count: render_world_stats.item_count,
                render_items_reused: render_world_stats.items_reused,
                render_items_examined: render_world_stats.items_examined,
                render_items_added: render_world_stats.items_added,
                render_items_updated: render_world_stats.items_updated,
                render_items_removed: render_world_stats.items_removed,
                render_items_rebuilt: render_world_stats.items_rebuilt,
                shadow_casters_examined: render_world_stats.shadow_casters_examined,
                shadow_aggregate_rebuilt: render_world_stats.shadow_union_rebuilt,
                submitted_history_settled: render_world_stats.submitted_history_settled,
                retained_bytes: u64::try_from(retained_bytes).unwrap_or(u64::MAX),
                rebuild_reason: rdg_render_world_rebuild_reason(
                    render_world_stats.full_rebuild_reason,
                ),
                ..RdgRenderWorldSnapshot::default()
            });

        frame_attempt.enter_phase(RdgFramePhase::Maintenance);

        let clustered_lighting_enabled = state
            .render_frame
            .extracted_scene
            .scene_variants
            .contains(SceneFeatures::USE_CLUSTERED_SHADING);

        // ── Phase 2: Cross-frame feature maintenance ───────────────────
        //
        // No Feature performs physical preparation here. Composer snapshots
        // drive live-node prepare after DPE; this block only advances cache
        // maintenance and synchronizes lightweight debug state.
        {
            // Sync camera debug settings before Composer snapshots RenderState.
            #[cfg(feature = "debug_view")]
            {
                let dv = camera.debug_view;
                state.render_frame.render_state.debug_view_mode = dv.mode;
                state.render_frame.render_state.debug_view_scale = dv.custom_scale;
            }

            let environment_cache_frame = state.resource_manager.frame_index();
            let retired_environment_scenes = state
                .environment_cache
                .maintain(&mut state.history_registry, environment_cache_frame);
            for retired_scene_id in retired_environment_scenes {
                state
                    .resource_manager
                    .remove_scene_global_states(retired_scene_id);
            }
            state
                .atmosphere_pass
                .maintain(&mut state.history_registry, environment_cache_frame);
        }

        // ── Phase 3: Build ComposerContext ──────────────────────────────
        frame_attempt.enter_phase(RdgFramePhase::AwaitingRender);
        let ctx = ComposerContext {
            wgpu_ctx: &mut state.wgpu_ctx,
            resource_manager: &mut state.resource_manager,
            pipeline_cache: &mut state.pipeline_cache,
            shader_manager: &mut state.shader_manager,

            extracted_scene: &mut state.render_frame.extracted_scene,
            render_state: &state.render_frame.render_state,
            renderer_settings: &self.settings,
            clustered_lighting_enabled,

            global_bind_group_cache: &mut state.global_bind_group_cache,

            render_lists: &mut state.render_lists,

            // blackboard: &mut state.blackboard,
            scene,
            camera,
            assets,
            frame_time,

            graph_storage: &mut state.graph_storage,
            rdg_diagnostics: &state.rdg_diagnostics,
            #[cfg(feature = "rdg_bench_support")]
            rdg_gpu_timings: &mut state.rdg_gpu_timings,
            transient_pool: &mut state.transient_pool,
            history_registry: &mut state.history_registry,
            persistent_buffer_registry: &mut state.persistent_buffer_registry,
            gpu_scene: &mut state.gpu_scene,
            gpu_scene_cull: &mut state.gpu_scene_cull,
            gpu_scene_render: &mut state.gpu_scene_render,
            gpu_driven_policy: &mut state.gpu_driven_policy,
            // sampler_registry: &mut state.sampler_registry,
            frame_arena: &state.frame_arena,
            fxaa_pass: &mut state.fxaa_pass,
            taa_pass: &mut state.taa_pass,
            cas_pass: &mut state.cas_pass,
            tone_map_pass: &mut state.tone_map_pass,
            present_pass: &mut state.present_pass,
            bloom_pass: &mut state.bloom_pass,
            ssao_pass: &mut state.ssao_pass,
            hiz_pass: &mut state.hiz_pass,
            ssgi_pass: &mut state.ssgi_pass,
            ssr_pass: &mut state.ssr_pass,

            prepass: &mut state.prepass,
            opaque_pass: &mut state.opaque_pass,
            skybox_pass: &mut state.skybox_pass,
            transparent_pass: &mut state.transparent_pass,
            transmission_copy_pass: &mut state.transmission_copy_pass,
            simple_forward_pass: &mut state.simple_forward_pass,
            ssss_pass: &mut state.ssss_pass,
            msaa_sync_pass: &mut state.msaa_sync_pass,

            shadow_pass: &mut state.shadow_pass,
            environment_cache: &mut state.environment_cache,
            equirect_to_cube_pass: &mut state.equirect_to_cube_pass,
            ibl_pass: &mut state.ibl_pass,
            atmosphere_pass: &mut state.atmosphere_pass,
            clustered_lighting_pass: &mut state.clustered_lighting_pass,
            mipmap_pass: &mut state.mipmap_pass,

            #[cfg(feature = "3dgs")]
            gaussian_splatting_pass: &mut state.gaussian_splatting_pass,

            #[cfg(feature = "debug_view")]
            debug_view_pass: &mut state.debug_view_pass,
        };

        // Return FrameComposer, defer Surface acquisition to render() call
        let composer = FrameComposer::new(ctx, self.size);
        frame_attempt.hand_off();
        Some(composer)
    }

    /// Performs periodic resource cleanup.
    ///
    /// Should be called after each frame to release unused GPU resources.
    /// Uses internal heuristics to avoid expensive cleanup every frame.
    pub fn maybe_prune(&mut self) {
        if let Some(state) = &mut self.context {
            state.render_frame.maybe_prune(&mut state.resource_manager);
            // Evict stale bind groups that haven't been touched recently.
            state.global_bind_group_cache.garbage_collect();

            // Feature maps own logical history-domain liveness; retiring one
            // atomically removes its set and both physical banks. RenderState
            // is touched for every active camera, so with the same TTL it can
            // never expire while an associated feature history remains live.
            let current_frame = state.resource_manager.frame_index();
            #[cfg(feature = "3dgs")]
            let gaussian_clouds = state
                .gaussian_splatting_pass
                .garbage_collect(current_frame, TEMPORAL_DOMAIN_TTL_FRAMES);
            let taa_domains = state.taa_pass.garbage_collect(
                &mut state.history_registry,
                current_frame,
                TEMPORAL_DOMAIN_TTL_FRAMES,
            );
            let ssr_domains = state.ssr_pass.garbage_collect(
                &mut state.history_registry,
                current_frame,
                TEMPORAL_DOMAIN_TTL_FRAMES,
            );
            let ssgi_domains = state.ssgi_pass.garbage_collect(
                &mut state.history_registry,
                current_frame,
                TEMPORAL_DOMAIN_TTL_FRAMES,
            );
            let view_domains = state
                .render_frame
                .render_state
                .garbage_collect_temporal_views(current_frame, TEMPORAL_DOMAIN_TTL_FRAMES);

            if taa_domains + ssr_domains + ssgi_domains + view_domains > 0 {
                log::debug!(
                    "Temporal GC: retired {taa_domains} TAA, {ssr_domains} SSR, {ssgi_domains} SSGI, and {view_domains} previous-view domains"
                );
            }
            #[cfg(feature = "3dgs")]
            if gaussian_clouds > 0 {
                log::debug!("Gaussian GC: retired {gaussian_clouds} persistent cloud uploads");
            }
        }
    }

    // === Runtime Settings API ===

    /// Returns the current [`RenderPath`].
    #[inline]
    pub fn render_path(&self) -> &RenderPath {
        &self.settings.path
    }

    /// Returns a reference to the current runtime renderer settings.
    #[inline]
    pub fn settings(&self) -> &RendererSettings {
        &self.settings
    }

    /// Returns a reference to the init-time configuration.
    #[inline]
    pub fn init_config(&self) -> &RendererInitConfig {
        &self.init_config
    }

    /// Applies new runtime settings, performing an internal diff to update
    /// only the parts that actually changed.
    ///
    /// This is the **single entry point** for all runtime configuration
    /// changes. Callers (UI panels, scripting layers, etc.) should maintain
    /// their own [`RendererSettings`] instance, mutate it, and pass it here.
    pub fn update_settings(&mut self, new_settings: RendererSettings) {
        if self.settings == new_settings {
            return;
        }

        let old = std::mem::replace(&mut self.settings, new_settings);

        if let Some(state) = &mut self.context {
            // VSync
            if old.vsync != self.settings.vsync {
                state.wgpu_ctx.set_vsync(self.settings.vsync);
            }

            // Render path
            if old.path != self.settings.path {
                state.wgpu_ctx.render_path = self.settings.path;
                state.wgpu_ctx.pipeline_settings_version += 1;
                state.gpu_driven_policy.bump_epoch();
                log::info!("RenderPath changed to {:?}", self.settings.path);
            }

            // Anisotropy
            if old.anisotropy_clamp != self.settings.anisotropy_clamp {
                state
                    .resource_manager
                    .set_global_anisotropy(self.settings.anisotropy_clamp);
                log::info!(
                    "Anisotropy clamp changed to {}",
                    self.settings.anisotropy_clamp
                );
            }

            if old.clustered_shading != self.settings.clustered_shading {
                state.wgpu_ctx.pipeline_settings_version += 1;
                log::info!(
                    "Clustered shading mode changed to {:?}",
                    self.settings.clustered_shading
                );
            }

            if old.gpu_driven != self.settings.gpu_driven {
                state.gpu_driven_policy.bump_epoch();
                if self.settings.gpu_driven == GpuDrivenMode::Off {
                    // Runtime disable is the one transition that drains the
                    // previous GPU attempt scratch immediately. Regular frame
                    // maintenance continues to age retained authority and the
                    // capacity ledger while the mode remains Off.
                    state.gpu_scene.clear_frame_scratch();
                    state.gpu_scene_render.clear_finalization_scratch();
                }
                log::info!(
                    "GPU-driven scene-submission policy changed to {:?}",
                    self.settings.gpu_driven
                );
            }
        }
    }

    /// Sets the runtime clustered-lighting routing mode.
    pub fn set_clustered_shading_mode(&mut self, mode: ClusteredShadingMode) {
        if self.settings.clustered_shading != mode {
            let mut new = self.settings.clone();
            new.clustered_shading = mode;
            self.update_settings(new);
        }
    }

    /// Sets the GPU-driven scene-submission policy.
    pub fn set_gpu_driven_mode(&mut self, mode: GpuDrivenMode) {
        if self.settings.gpu_driven != mode {
            let mut new = self.settings.clone();
            new.gpu_driven = mode;
            self.update_settings(new);
        }
    }

    /// Switches the active render path at runtime.
    ///
    /// Convenience wrapper around [`update_settings`](Self::update_settings)
    /// for changing only the render path.
    pub fn set_render_path(&mut self, path: RenderPath) {
        if self.settings.path != path {
            let mut new = self.settings.clone();
            new.path = path;
            self.update_settings(new);
        }
    }

    /// Sets the active debug view mode.
    ///
    /// When set to anything other than `None`, the FrameComposer will
    /// either replace the post-process output with a fullscreen
    /// visualisation of the selected intermediate texture (post-process
    /// modes), or inject shader defines to short-circuit PBR lighting
    /// and output raw material attributes (material-override modes).
    #[cfg(feature = "debug_view")]
    pub fn set_debug_view_mode(&mut self, mode: myth_scene::camera::DebugViewMode) {
        if let Some(state) = &mut self.context {
            state.render_frame.render_state.debug_view_mode = mode;
        }
    }

    /// Returns the current debug view mode.
    #[cfg(feature = "debug_view")]
    pub fn debug_view_mode(&self) -> myth_scene::camera::DebugViewMode {
        self.context
            .as_ref()
            .map(|s| s.render_frame.render_state.debug_view_mode)
            .unwrap_or_default()
    }

    // === Public Methods: For External Plugins (e.g., UI Pass) ===

    /// Returns a reference to the wgpu Device.
    ///
    /// Useful for external plugins to initialize GPU resources.
    pub fn device(&self) -> Option<&wgpu::Device> {
        self.context.as_ref().map(|s| &s.wgpu_ctx.device)
    }

    /// Returns a reference to the wgpu Queue.
    ///
    /// Useful for external plugins to submit commands.
    pub fn queue(&self) -> Option<&wgpu::Queue> {
        self.context.as_ref().map(|s| &s.wgpu_ctx.queue)
    }

    /// Returns the surface/render-target texture format.
    ///
    /// In windowed mode this is the swap-chain format; in headless mode it
    /// is the offscreen texture format. Returns `None` before initialisation.
    pub fn surface_format(&self) -> Option<wgpu::TextureFormat> {
        self.context
            .as_ref()
            .map(|s| s.wgpu_ctx.surface_view_format)
    }

    /// Returns a reference to the `WgpuContext`.
    ///
    /// For external plugins that need access to low-level GPU resources.
    /// Only available after renderer initialization.
    pub fn wgpu_ctx(&self) -> Option<&WgpuContext> {
        self.context.as_ref().map(|s| &s.wgpu_ctx)
    }

    /// Returns the immutable GPU-driven capability snapshot captured during
    /// device initialization, or `None` before initialization.
    #[inline]
    #[must_use]
    pub fn gpu_driven_capabilities(&self) -> Option<&GpuDrivenCapabilities> {
        self.context
            .as_ref()
            .map(|state| state.wgpu_ctx.gpu_driven_capabilities())
    }

    /// Returns a reference to the GPU resource manager.
    pub fn resource_manager(&self) -> Option<&ResourceManager> {
        self.context.as_ref().map(|s| &s.resource_manager)
    }

    /// Returns the maximum mip level of the scene's currently published
    /// environment PMREM.
    ///
    /// This reports persistent-cache publication truth rather than the
    /// desired size of a pending replacement source. External graph passes
    /// can therefore keep roughness-to-LOD mapping consistent with the
    /// logical PMREM resource exposed through
    /// [`GraphBlackboard`](crate::graph::core::GraphBlackboard).
    #[must_use]
    pub fn published_environment_max_mip_level(&self, scene_id: u32) -> Option<f32> {
        self.context.as_ref().and_then(|state| {
            state
                .environment_cache
                .published_pmrem_mip_level_count(scene_id, &state.history_registry)
                .map(|mip_count| mip_count.saturating_sub(1) as f32)
        })
    }

    /// Returns the latest terminal Render Graph attempt diagnostics.
    ///
    /// This includes submitted frames, target-acquisition failures, and
    /// abandoned composers. Zero-sized or uninitialized Renderer calls do not
    /// create an RDG attempt and therefore leave this snapshot unchanged.
    #[must_use]
    pub fn latest_rdg_diagnostics(&self) -> Option<RdgFrameDiagnostics> {
        self.context
            .as_ref()
            .and_then(|state| state.rdg_diagnostics.latest())
    }

    /// Returns the latest asynchronously completed RDG GPU timestamp sample.
    ///
    /// The sample normally trails the current renderer frame by several frames;
    /// [`RdgGpuFrameTimings::sample_lag`] exposes that delay explicitly.
    #[cfg(feature = "rdg_bench_support")]
    #[must_use]
    pub fn latest_rdg_gpu_timings(&self) -> Option<&RdgGpuFrameTimings> {
        self.context
            .as_ref()
            .and_then(|state| state.rdg_gpu_timings.latest())
    }

    /// Changes the non-blocking GPU timestamp sampling cadence. Zero disables
    /// sampling, which is also the default. Returns false before renderer
    /// initialization.
    #[cfg(feature = "rdg_bench_support")]
    pub fn set_rdg_gpu_timing_sample_interval(&mut self, interval: u64) -> bool {
        let Some(state) = self.context.as_mut() else {
            return false;
        };
        state.rdg_gpu_timings.set_sample_interval(interval);
        true
    }

    /// Whether the active adapter/device enabled the optional timestamp tier.
    #[cfg(feature = "rdg_bench_support")]
    #[must_use]
    pub fn rdg_gpu_timings_supported(&self) -> bool {
        self.context
            .as_ref()
            .is_some_and(|state| state.rdg_gpu_timings.supported())
    }

    /// Returns the accounted CPU backing capacity retained by the RDG between
    /// attempts, including topology, attempt stores, and renderer-owned GPU
    /// Scene authority/finalization storage. Hash-table bytes are stable
    /// estimates rather than allocator-exact data.
    #[must_use]
    pub fn rdg_retained_capacity_stats(&self) -> Option<RdgRetainedCapacityStats> {
        self.context.as_ref().map(|state| {
            state
                .graph_storage
                .retained_capacity_stats()
                .with_gpu_scene_capacity(
                    GpuSceneCapacityLedger::authority_usage(&state.gpu_scene),
                    GpuSceneCapacityLedger::scratch_usage(
                        &state.gpu_scene,
                        &state.gpu_scene_render,
                    ),
                    state.gpu_scene_capacity.stats(),
                )
        })
    }

    /// Requests pressure-driven retained-capacity maintenance. GraphStorage
    /// and renderer-owned GPU Scene storage each select at most one eligible
    /// family under the same hysteretic policy; deallocation is deferred to
    /// their shared next-frame maintenance boundary.
    pub fn request_rdg_retained_capacity_trim(&mut self) {
        if let Some(state) = &mut self.context {
            state.graph_storage.request_retained_capacity_trim();
            state.gpu_scene_capacity.request_trim();
        }
    }

    /// Dumps a compact, human-oriented Mermaid view of the latest Render Graph.
    ///
    /// Lazy Asset Texture inputs are aggregated by consumer. Use
    /// [`dump_graph_mermaid_with_mode`](Self::dump_graph_mermaid_with_mode)
    /// with [`MermaidDumpMode::Full`](crate::graph::MermaidDumpMode::Full) for
    /// exact per-resource topology.
    pub fn dump_graph_mermaid(&self) -> Option<String> {
        self.context
            .as_ref()
            .map(|s| s.graph_storage.dump_mermaid())
    }

    /// Dumps the current Render Graph using the requested level of detail.
    ///
    /// Compact mode aggregates high-cardinality lazy Asset Texture inputs;
    /// Full mode preserves every logical resource edge for exact inspection.
    pub fn dump_graph_mermaid_with_mode(
        &self,
        mode: crate::graph::MermaidDumpMode,
    ) -> Option<String> {
        self.context
            .as_ref()
            .map(|s| s.graph_storage.dump_mermaid_with_mode(mode))
    }

    // === Custom Shader Registration API ===

    /// Registers a custom WGSL shader template with the given name.
    ///
    /// The source string is pre-processed by the minijinja template engine at
    /// compile time, so `{$ include "chunks/camera_uniforms.wgsl" $}` and
    /// similar directives are fully supported.
    ///
    /// # Usage
    ///
    /// ```rust,ignore
    /// renderer.register_shader_template(
    ///     "custom_unlit",
    ///     include_str!("shaders/custom_unlit.wgsl"),
    /// );
    /// ```
    ///
    /// After registration, any named template lookup that resolves
    /// `custom_unlit` through the standard shader template system will use this
    /// exact template source.
    ///
    /// This API is the low-level raw-template escape hatch. For typical
    /// custom materials, prefer `#[myth_material(shader = "...", shader_src = ...)]`,
    /// which lets the engine supply the standard material prelude automatically.
    ///
    /// # Panics
    ///
    /// Panics if the renderer has not been initialized via [`init`](Self::init).
    pub fn register_shader_template(&mut self, name: &str, source: &str) {
        let state = self
            .context
            .as_mut()
            .expect("Renderer must be initialized before registering shader templates");
        state.shader_manager.register_template(name, source);
    }

    /// Compiles or retrieves a cached compute pipeline from the renderer's
    /// standard shader template system.
    ///
    /// Use [`register_shader_template`](Self::register_shader_template) first
    /// when `source` is [`ShaderSource::File`] and you want that named lookup
    /// to resolve to a registered custom template.
    /// This keeps standalone examples on the same shader compilation and cache
    /// path as engine-owned passes. The provided tracked bind-group layouts are
    /// also registered into [`PipelineCache`] so later RDG code can look them
    /// up by pipeline ID instead of carrying layouts through user state.
    pub(crate) fn get_or_create_compute_pipeline(
        &mut self,
        source: ShaderSource<'_>,
        shader_options: &ShaderCompilationOptions,
        bind_group_layouts: &[&Tracked<wgpu::BindGroupLayout>],
        label: &str,
    ) -> ComputePipelineId {
        let state = self
            .context
            .as_mut()
            .expect("Renderer must be initialized before creating compute pipelines");

        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let layout_label = format!("{label} Layout");
        let (module, shader_hash) =
            state
                .shader_manager
                .get_or_compile(&state.wgpu_ctx.device, source, shader_options);
        let (pipeline_layout, tracked_layouts) =
            build_pipeline_layout(&state.wgpu_ctx.device, bind_group_layouts, &layout_label);

        let pipeline_id = state.pipeline_cache.get_or_create_compute(
            &state.wgpu_ctx.device,
            module,
            &pipeline_layout,
            &ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            label,
        );
        state
            .pipeline_cache
            .register_compute_layouts(pipeline_id, tracked_layouts);
        pipeline_id
    }

    /// Compiles or retrieves a cached fullscreen render pipeline from the
    /// renderer's standard shader template system.
    ///
    /// This helper is intended for post-processing and fullscreen user passes.
    /// It hides shader compilation, pipeline-layout creation, and pipeline-key
    /// construction behind the same template system used by engine-owned
    /// passes, while also registering tracked bind-group layouts in
    /// [`PipelineCache`] for later lookup by pipeline ID.
    pub(crate) fn get_or_create_fullscreen_pipeline(
        &mut self,
        source: ShaderSource<'_>,
        shader_options: &ShaderCompilationOptions,
        bind_group_layouts: &[&Tracked<wgpu::BindGroupLayout>],
        color_targets: &[wgpu::ColorTargetState],
        depth_stencil: Option<wgpu::DepthStencilState>,
        multisample: wgpu::MultisampleState,
        label: &str,
    ) -> RenderPipelineId {
        let state = self
            .context
            .as_mut()
            .expect("Renderer must be initialized before creating render pipelines");

        let layout_label = format!("{label} Layout");
        let (module, shader_hash) =
            state
                .shader_manager
                .get_or_compile(&state.wgpu_ctx.device, source, shader_options);
        let (pipeline_layout, tracked_layouts) =
            build_pipeline_layout(&state.wgpu_ctx.device, bind_group_layouts, &layout_label);
        let key = FullscreenPipelineKey {
            shader_hash,
            compilation_hash: 0,
            color_targets: color_targets
                .iter()
                .cloned()
                .map(ColorTargetKey::from)
                .collect(),
            depth_stencil: depth_stencil.map(DepthStencilKey::from),
            multisample: MultisampleKey::from(multisample),
            topology: wgpu::PrimitiveTopology::TriangleList,
        };

        let pipeline_id = state.pipeline_cache.get_or_create_fullscreen(
            &state.wgpu_ctx.device,
            module,
            &pipeline_layout,
            &key,
            label,
        );
        state
            .pipeline_cache
            .register_render_layouts(pipeline_id, tracked_layouts);
        pipeline_id
    }

    /// Returns `true` if the renderer is in headless (offscreen) mode.
    #[inline]
    #[must_use]
    pub fn is_headless(&self) -> bool {
        self.context
            .as_ref()
            .is_some_and(|s| s.wgpu_ctx.is_headless())
    }

    /// Reads back the current headless render target as raw pixel data.
    ///
    /// The returned `Vec<u8>` contains tightly-packed pixel data whose per-pixel
    /// byte count matches the headless texture format (e.g. 4 bytes for RGBA8,
    /// 8 bytes for RGBA16Float). Row ordering is top-to-bottom.
    ///
    /// A staging buffer is cached internally and re-used across calls as long
    /// as the required size has not changed, eliminating per-frame allocation.
    ///
    /// This method submits a GPU copy command and blocks the calling thread
    /// until the transfer completes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The renderer has not been initialised
    /// - No headless render target exists (windowed mode)
    /// - The GPU buffer mapping fails
    pub fn readback_pixels(&mut self) -> Result<Vec<u8>> {
        let state = self
            .context
            .as_mut()
            .ok_or(myth_core::RenderError::NotInitialized)?;

        let texture = state
            .wgpu_ctx
            .headless_texture
            .as_ref()
            .ok_or(myth_core::RenderError::NoHeadlessTarget)?;

        let width = state.wgpu_ctx.target_width;
        let height = state.wgpu_ctx.target_height;
        let format = state.wgpu_ctx.surface_view_format;

        let bytes_per_pixel = format.block_copy_size(None).ok_or_else(|| {
            myth_core::RenderError::ReadbackFailed(format!(
                "unsupported readback format: {format:?}"
            ))
        })?;

        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = u64::from(padded_bytes_per_row) * u64::from(height);

        // Re-use the cached buffer when the required capacity matches.
        if state.cached_readback_buffer_size != buffer_size {
            state.cached_readback_buffer = Some(state.wgpu_ctx.device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Readback Buffer"),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            ));
            state.cached_readback_buffer_size = buffer_size;
        }

        let readback_buffer = state.cached_readback_buffer.as_ref().unwrap();

        let mut encoder =
            state
                .wgpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Readback Encoder"),
                });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        state
            .wgpu_ctx
            .queue
            .submit(std::iter::once(encoder.finish()));

        // Block until the GPU finishes the copy and the buffer is mappable.
        let buffer_slice = readback_buffer.slice(..);

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        state
            .wgpu_ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| myth_core::RenderError::ReadbackFailed(e.to_string()))?;

        rx.recv()
            .map_err(|e| myth_core::RenderError::ReadbackFailed(e.to_string()))?
            .map_err(|e| myth_core::RenderError::ReadbackFailed(e.to_string()))?;

        // Strip per-row padding and produce a tightly-packed pixel buffer.
        let mapped = buffer_slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        readback_buffer.unmap();

        Ok(pixels)
    }

    /// Creates a [`ReadbackStream`] backed by the headless render target.
    ///
    /// The stream pre-allocates `buffer_count` staging buffers that rotate in
    /// a ring, enabling fully non-blocking GPU→CPU readback suitable for
    /// video recording and AI training-data pipelines.
    ///
    /// # Errors
    ///
    /// Returns an error if the renderer is not initialised or not in headless
    /// mode, or if the texture format does not support readback.
    pub fn create_readback_stream(
        &self,
        buffer_count: usize,
        max_stash_size: usize,
    ) -> Result<crate::core::ReadbackStream> {
        let state = self
            .context
            .as_ref()
            .ok_or(myth_core::RenderError::NotInitialized)?;

        if state.wgpu_ctx.headless_texture.is_none() {
            return Err(myth_core::RenderError::NoHeadlessTarget.into());
        }

        let width = state.wgpu_ctx.target_width;
        let height = state.wgpu_ctx.target_height;
        let format = state.wgpu_ctx.surface_view_format;

        let stream = crate::core::ReadbackStream::new(
            &state.wgpu_ctx.device,
            width,
            height,
            format,
            buffer_count,
            max_stash_size,
        )?;

        Ok(stream)
    }

    /// Drives pending GPU callbacks without blocking.
    ///
    /// Call this once per frame in a readback-stream loop so that `map_async`
    /// callbacks fire and frames become available via
    /// [`ReadbackStream::try_recv`].
    pub fn poll_device(&self) {
        if let Some(state) = &self.context {
            let _ = state.wgpu_ctx.device.poll(wgpu::PollType::Poll);
        }
    }

    /// Returns a reference to the headless render target texture, if present.
    #[must_use]
    pub fn headless_texture(&self) -> Option<&wgpu::Texture> {
        self.context
            .as_ref()
            .and_then(|s| s.wgpu_ctx.headless_texture.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graph::core::scene_submission::SceneSubmissionPlanId;
    use crate::graph::core::{
        ExecuteContext, GraphBlackboard, HookStage, PassNode, RdgAttemptOutcome, RdgFramePhase,
        RdgGpuDrivenFailure, RdgResourceMaterializationFailure, RdgSceneSubmissionBackend,
        RdgSceneSubmissionFamily, RdgSceneSubmissionSelectionReason, RdgSurfaceAcquireFailure,
        TextureDesc,
    };
    use crate::test_support::SharedTestContext;
    use glam::{Affine3A, Mat4, Quat, Vec3, Vec4};
    use myth_assets::{AssetServer, MaterialHandle, SceneExt, TextureHandle};
    use myth_core::NodeHandle;
    #[cfg(feature = "3dgs")]
    use myth_resources::gaussian_splat::{GaussianCloud, GaussianSHCoefficients, GaussianSplat};
    use myth_resources::geometry::{Attribute, Geometry, VertexFormat};
    use myth_resources::image::{ColorSpace, Image, ImageDimension, PixelFormat};
    use myth_resources::material::{PhysicalFeatures, PhysicalMaterial, UnlitMaterial};
    use myth_resources::texture::{Texture, TextureSource};
    use myth_scene::background::BackgroundMode;
    use myth_scene::camera::Camera;
    use myth_scene::skeleton::Skeleton;
    use myth_scene::{BindMode, Light, Scene};

    struct DepthOnlyTestRoot;

    impl PassNode<'_> for DepthOnlyTestRoot {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    struct MaterialEpochGateTestRoot<'a> {
        drift_material: Option<MaterialHandle>,
        opaque_submission: SceneSubmissionPlanId,
        prepass_submission: SceneSubmissionPlanId,
        opaque_command_count: &'a std::sync::atomic::AtomicUsize,
        prepass_command_count: &'a std::sync::atomic::AtomicUsize,
    }

    impl<'a> PassNode<'a> for MaterialEpochGateTestRoot<'a> {
        fn prepare(&mut self, ctx: &mut crate::graph::core::PrepareContext<'_>) {
            let Some(material_handle) = self.drift_material else {
                return;
            };
            let material = ctx
                .assets
                .materials
                .get(material_handle)
                .expect("test material missing during live prepare");
            let physical = material
                .as_physical()
                .expect("epoch-drift test requires a physical material");
            let previous_epoch = material.as_renderable().version();
            physical.enable_feature(PhysicalFeatures::CLEARCOAT);
            assert_ne!(
                material.as_renderable().version(),
                previous_epoch,
                "the test seam must mutate the physical material interior epoch"
            );
        }

        fn execute(&self, ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {
            let opaque_stream = ctx.scene_submissions.stream(self.opaque_submission);
            let prepass_stream = ctx.scene_submissions.stream(self.prepass_submission);
            self.opaque_command_count.store(
                ctx.scene_submissions.commands(opaque_stream).len(),
                std::sync::atomic::Ordering::Relaxed,
            );
            self.prepass_command_count.store(
                ctx.scene_submissions.commands(prepass_stream).len(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }

    fn scene_submission_for_test_pass(
        graph: &crate::graph::core::RenderGraph<'_>,
        pass_name: &str,
    ) -> SceneSubmissionPlanId {
        let pass_index = graph
            .storage
            .passes
            .iter()
            .position(|pass| pass.name == pass_name)
            .unwrap_or_else(|| panic!("{pass_name} missing during graph construction"));
        *graph
            .storage
            .pass_scene_submission_demands(pass_index)
            .first()
            .unwrap_or_else(|| panic!("{pass_name} has no scene submission demand"))
    }

    fn init_shared_headless_renderer(
        renderer: &mut Renderer,
        width: u32,
        height: u32,
        format: Option<PixelFormat>,
    ) {
        if renderer.context.is_some() {
            return;
        }

        renderer.size = (width, height);
        let target_format = format.map(|format| format.to_wgpu(ColorSpace::Srgb));
        let wgpu_ctx = pollster::block_on(SharedTestContext::new_headless(
            &renderer.init_config,
            &renderer.settings,
            width,
            height,
            target_format,
        ))
        .expect("headless renderer init failed");
        renderer.assemble_state(wgpu_ctx);
    }

    fn init_headless_renderer() -> Renderer {
        let mut renderer =
            Renderer::new(RendererInitConfig::default(), RendererSettings::default());
        init_shared_headless_renderer(&mut renderer, 128, 128, None);
        renderer
    }

    fn init_headless_gpu_renderer(path: RenderPath, mode: GpuDrivenMode) -> Renderer {
        init_headless_gpu_renderer_with_features(path, mode, wgpu::Features::empty())
    }

    fn init_headless_gpu_renderer_with_features(
        path: RenderPath,
        mode: GpuDrivenMode,
        required_features: wgpu::Features,
    ) -> Renderer {
        let mut renderer = Renderer::new(
            RendererInitConfig {
                required_features,
                ..RendererInitConfig::default()
            },
            RendererSettings {
                path,
                gpu_driven: mode,
                ..RendererSettings::default()
            },
        );
        pollster::block_on(renderer.init_headless(128, 128, None))
            .expect("headless GPU-driven renderer init failed");
        renderer
    }

    fn init_headless_basic_forward_gpu_renderer(mode: GpuDrivenMode) -> Renderer {
        init_headless_gpu_renderer(RenderPath::BasicForward, mode)
    }
    fn init_headless_renderer_with_model_capacity(initial_capacity: usize) -> Renderer {
        let mut renderer = init_headless_renderer();
        let anisotropy_clamp = renderer.settings.anisotropy_clamp;
        let state = renderer.context.as_mut().expect("renderer state missing");
        state.resource_manager = ResourceManager::new_with_model_capacity(
            state.wgpu_ctx.device.clone(),
            state.wgpu_ctx.queue.clone(),
            anisotropy_clamp,
            initial_capacity,
        );
        renderer
    }

    fn make_camera() -> RenderCamera {
        let mut camera = Camera::new_perspective(45.0, 1.0, 0.1);
        camera.update_view_projection(&Affine3A::from_translation(Vec3::new(0.0, 0.0, 5.0)));
        camera.extract_render_camera()
    }

    fn make_shadow_camera() -> RenderCamera {
        let eye = Vec3::new(4.5, 3.5, 6.5);
        let target = Vec3::new(0.0, -0.35, 0.0);
        let world = Mat4::look_at_rh(eye, target, Vec3::Y).inverse();
        let mut camera = Camera::new_perspective(45.0, 1.0, 0.1);
        camera.update_view_projection(&Affine3A::from_mat4(world));
        camera.extract_render_camera()
    }

    /// Build a two-bin shadowed scene. Receiver and caster deliberately use
    /// distinct Geometry/Material identities so Main, Prepass, 2D Shadow and
    /// Cube Shadow all exercise the M7.4 multi-bin stream.
    fn make_gpu_scene_shadow_fixture(cast_shadows: bool) -> (AssetServer, Scene) {
        let assets = AssetServer::new();
        let receiver_geometry = assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
        let receiver_material = assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.72, 0.64, 0.55, 1.0)));
        let caster_geometry = assets.geometries.add(Geometry::new_box(0.9, 1.45, 0.9));
        let caster_material = assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.42, 0.55, 0.78, 1.0)));
        let mut scene = Scene::new();

        let receiver = scene.spawn(receiver_geometry, receiver_material, &assets);
        scene
            .edit_node(receiver)
            .set_position(0.0, -1.0, 0.0)
            .set_scale_xyz(6.0, 0.12, 6.0)
            .set_cast_shadows(false)
            .set_receive_shadows(true);

        let caster = scene.spawn(caster_geometry, caster_material, &assets);
        scene
            .edit_node(caster)
            .set_position(-0.35, -0.15, 0.0)
            .set_scale_xyz(1.0, 1.0, 1.0)
            .set_cast_shadows(cast_shadows)
            .set_receive_shadows(true);

        let mut directional = Light::new_directional(Vec3::new(1.0, 0.94, 0.86), 1.0);
        directional.cast_shadows = cast_shadows;
        if let Some(shadow) = directional.shadow.as_mut() {
            shadow.map_size = 256;
            shadow.normal_bias = 0.0;
        }
        let directional = scene.add_light(directional);
        scene
            .edit_node(directional)
            .set_position(-4.0, 6.0, -4.0)
            .look_at(Vec3::new(-0.2, -0.8, 0.0));

        // Keep a real cube-family producer in the graph without filling the
        // receiver's directional shadow with a second direct-light term.
        let mut point = Light::new_point(Vec3::new(0.55, 0.72, 1.0), 0.0, 9.0);
        point.cast_shadows = cast_shadows;
        if let Some(shadow) = point.shadow.as_mut() {
            shadow.map_size = 32;
            shadow.normal_bias = 0.0;
        }
        let point = scene.add_light(point);
        scene.edit_node(point).set_position(-2.0, 1.8, 2.4);
        scene.update_matrix_world();

        (assets, scene)
    }

    /// Two distinct textured bins whose solid texels make both material
    /// bindings observable in a headless readback.
    fn make_gpu_scene_textured_fixture() -> (AssetServer, Scene, [TextureHandle; 2]) {
        let assets = AssetServer::new();
        let mut textures = Vec::with_capacity(2);
        for (label, rgba) in [
            ("GPU Scene Red", [224, 24, 16, 255]),
            ("GPU Scene Blue", [16, 48, 224, 255]),
        ] {
            let image = assets.images.add(Image::new(
                8,
                8,
                1,
                ImageDimension::D2,
                PixelFormat::Rgba8Unorm,
                Some(rgba.repeat(8 * 8)),
            ));
            let mut texture = Texture::new_2d(Some(label), image);
            texture.color_space = ColorSpace::Linear;
            texture.generate_mipmaps = true;
            textures.push(assets.textures.add(texture));
        }
        let textures: [TextureHandle; 2] = textures
            .try_into()
            .expect("textured GPU Scene fixture must create two textures");

        let mut scene = Scene::new();
        let left = scene.spawn_box(
            1.0,
            1.0,
            1.0,
            UnlitMaterial::new(Vec4::ONE).with_map(textures[0]),
            &assets,
        );
        scene.edit_node(left).set_position(-0.7, 0.0, 0.0);
        let right = scene.spawn_box(
            0.75,
            1.25,
            0.8,
            UnlitMaterial::new(Vec4::ONE).with_map(textures[1]),
            &assets,
        );
        scene.edit_node(right).set_position(0.7, 0.0, 0.0);
        scene.update_matrix_world();

        (assets, scene, textures)
    }

    fn spawn_test_morph_box(scene: &mut Scene, assets: &AssetServer) -> NodeHandle {
        let mut geometry = Geometry::new_box(1.0, 1.0, 1.0);
        let vertex_count = geometry
            .get_attribute("position")
            .expect("box position attribute missing")
            .count as usize;
        geometry.add_morph_attribute(
            "position",
            Attribute::new_planar(&vec![Vec3::ZERO; vertex_count], VertexFormat::Float32x3),
        );
        geometry.build_morph_storage_buffers();
        assert!(geometry.has_morph_targets());
        let mesh_node = scene.spawn(geometry, PhysicalMaterial::new(Vec4::ONE), assets);
        scene
            .meshes
            .get_mut(mesh_node)
            .expect("morph mesh missing")
            .init_morph_targets(1, vertex_count as u32);
        mesh_node
    }

    fn assert_local_light_buffers_unmaterialized(state: &RendererState) {
        let scene = &state.render_frame.extracted_scene;
        for (id, handle) in [
            (
                scene.local_light_metadata_buffer.id(),
                scene.local_light_metadata_buffer.gpu_handle(),
            ),
            (
                scene.local_light_storage_buffer.id(),
                scene.local_light_storage_buffer.gpu_handle(),
            ),
        ] {
            assert!(
                handle.is_none(),
                "pre-graph frame exit must not assign a local-light GPU handle"
            );
            assert!(
                state
                    .resource_manager
                    .get_gpu_buffer_by_cpu_id(id)
                    .is_none(),
                "pre-graph frame exit must not create a local-light physical buffer"
            );
        }
    }

    fn global_buffer_ids(state: &RendererState) -> [u64; 3] {
        [
            state.render_frame.render_state.uniforms().id(),
            state
                .render_frame
                .extracted_scene
                .environment_uniforms_buffer
                .id(),
            state
                .render_frame
                .extracted_scene
                .directional_light_storage_buffer
                .id(),
        ]
    }

    fn assert_global_state_unmaterialized(state: &RendererState, scene_id: u32) {
        for cpu_id in global_buffer_ids(state) {
            assert!(
                state
                    .resource_manager
                    .get_gpu_buffer_by_cpu_id(cpu_id)
                    .is_none(),
                "pre-graph frame exit must not create a Group 0 physical buffer"
            );
        }
        assert!(
            state
                .resource_manager
                .get_global_state(state.render_frame.render_state.id, scene_id)
                .is_none(),
            "pre-graph frame exit must not create a persistent Group 0 state"
        );
    }

    #[cfg(feature = "3dgs")]
    fn pack_half2(left: f32, right: f32) -> u32 {
        u32::from(half::f16::from_f32(left).to_bits())
            | (u32::from(half::f16::from_f32(right).to_bits()) << 16)
    }

    #[cfg(feature = "3dgs")]
    fn tiny_gaussian_cloud() -> GaussianCloud {
        GaussianCloud {
            gaussians: vec![GaussianSplat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                opacity: pack_half2(0.8, 0.0),
                sh_idx: 0,
                cov: [
                    pack_half2(0.01, 0.0),
                    pack_half2(0.0, 0.01),
                    pack_half2(0.0, 0.01),
                ],
            }],
            sh_coefficients: vec![GaussianSHCoefficients { data: [0; 24] }],
            sh_degree: 0,
            num_points: 1,
            aabb_min: Vec3::splat(-0.5),
            aabb_max: Vec3::splat(0.5),
            center: Vec3::ZERO,
            mip_splatting: false,
            kernel_size: 0.3,
            color_space: ColorSpace::Linear,
            opacity_compensation: 1.0,
        }
    }

    fn render_frame(
        renderer: &mut Renderer,
        scene: &mut Scene,
        camera: RenderCamera,
        assets: &AssetServer,
        frame_index: u64,
    ) {
        renderer
            .begin_frame(
                scene,
                camera,
                assets,
                FrameTime {
                    time: frame_index as f32 / 60.0,
                    delta_time: 1.0 / 60.0,
                    frame_count: frame_index,
                },
            )
            .expect("frame composer must exist")
            .render();
    }

    #[test]
    fn abandoned_composer_is_transactional_across_diagnostics_assets_and_history() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let image_handle = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![192; 8 * 8 * 4]),
        ));
        let mut texture = Texture::new_2d(Some("Abandoned Frame Texture"), image_handle);
        texture.generate_mipmaps = true;
        let texture_handle = assets.textures.add(texture);
        let mesh_node = spawn_test_morph_box(&mut scene, &assets);
        let (geometry_handle, material_handle, morph_buffer_id) = {
            let mesh = scene.meshes.get_mut(mesh_node).expect("test mesh missing");
            mesh.set_morph_target_influence(0, 1.0);
            (mesh.geometry, mesh.material, mesh.morph_uniforms.id())
        };
        assets
            .materials
            .get(material_handle)
            .expect("test material missing")
            .as_physical()
            .expect("test material must be physical")
            .set_map(Some(texture_handle));
        let initial_pipeline_count = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .pipeline_cache
            .render_pipeline_count();

        let composer = renderer
            .begin_frame(
                &mut scene,
                make_camera(),
                &assets,
                FrameTime {
                    frame_count: 23,
                    ..FrameTime::default()
                },
            )
            .expect("frame composer missing");
        drop(composer);

        let diagnostics = renderer
            .latest_rdg_diagnostics()
            .expect("abandoned composer diagnostics missing");
        assert_eq!(diagnostics.renderer_frame(), Some(23));
        assert_eq!(diagnostics.outcome(), RdgAttemptOutcome::Abandoned);
        assert_eq!(diagnostics.phase(), RdgFramePhase::AwaitingRender);
        assert_eq!(diagnostics.declared_passes(), 0);
        assert_eq!(diagnostics.live_passes(), None);
        assert_eq!(diagnostics.prepared_passes(), 0);
        assert_eq!(diagnostics.executed_passes(), 0);
        assert!(!diagnostics.queue_submitted());
        assert_local_light_buffers_unmaterialized(
            renderer.context.as_ref().expect("renderer state missing"),
        );
        assert_global_state_unmaterialized(
            renderer.context.as_ref().expect("renderer state missing"),
            scene.id(),
        );
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(state.render_lists.opaque.is_empty());
        assert!(
            state
                .resource_manager
                .get_geometry(geometry_handle)
                .is_none()
        );
        assert!(
            state
                .resource_manager
                .get_material(material_handle)
                .is_none()
        );
        assert!(
            state
                .resource_manager
                .get_texture_image(texture_handle)
                .is_none()
        );
        assert!(
            state
                .resource_manager
                .get_gpu_buffer_by_cpu_id(morph_buffer_id)
                .is_none()
        );
        assert!(!state.mipmap_pass.generator_initialized());
        assert_eq!(state.prepass.cached_pipeline_count(), 0);
        assert_eq!(
            state.pipeline_cache.render_pipeline_count(),
            initial_pipeline_count
        );
        let unsubmitted = scene
            .meshes
            .get(mesh_node)
            .expect("test mesh missing")
            .morph_uniforms();
        assert!(unsubmitted.weights[0].x.abs() <= f32::EPSILON);
        drop(unsubmitted);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 24);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(
            state
                .resource_manager
                .get_geometry(geometry_handle)
                .is_some()
        );
        assert!(
            state
                .resource_manager
                .get_material(material_handle)
                .is_some()
        );
        assert!(
            state
                .resource_manager
                .get_texture_image(texture_handle)
                .expect("submitted texture missing")
                .mipmaps_submitted()
        );
        let submitted = scene
            .meshes
            .get(mesh_node)
            .expect("test mesh missing")
            .morph_uniforms();
        assert!((submitted.weights[0].x - 1.0).abs() <= f32::EPSILON);
        assert!(submitted.prev_weights[0].x.abs() <= f32::EPSILON);
    }

    #[test]
    fn surface_acquisition_failure_does_not_advance_render_world_transform_history() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let mesh_node = scene.spawn_box(1.0, 1.0, 1.0, PhysicalMaterial::new(Vec4::ONE), &assets);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);
        let baseline_world = {
            let state = renderer.context.as_ref().expect("renderer state missing");
            let item = state
                .render_frame
                .extracted_scene
                .render_items
                .iter()
                .find(|item| item.node_handle == mesh_node)
                .expect("baseline render-world item missing");
            assert_eq!(item.prev_world_matrix, item.world_matrix);
            item.world_matrix
        };
        assert!(
            renderer
                .latest_rdg_diagnostics()
                .expect("baseline diagnostics missing")
                .queue_submitted()
        );

        scene.edit_node(mesh_node).set_position(2.0, 0.0, 0.0);
        scene.update_matrix_world();
        renderer
            .begin_frame(
                &mut scene,
                make_camera(),
                &assets,
                FrameTime {
                    frame_count: 1,
                    ..FrameTime::default()
                },
            )
            .expect("surface-failure frame composer missing")
            .with_forced_surface_acquisition_failure(RdgSurfaceAcquireFailure::Timeout)
            .render();

        let failed = renderer
            .latest_rdg_diagnostics()
            .expect("surface-failure diagnostics missing");
        assert_eq!(
            failed.outcome(),
            RdgAttemptOutcome::SurfaceAcquisitionFailed(RdgSurfaceAcquireFailure::Timeout)
        );
        assert!(!failed.queue_submitted());
        assert_eq!(failed.render_world().submitted_history_settled(), 0);
        {
            let state = renderer.context.as_ref().expect("renderer state missing");
            let item = state
                .render_frame
                .extracted_scene
                .render_items
                .iter()
                .find(|item| item.node_handle == mesh_node)
                .expect("failed-frame render-world item missing");
            assert_eq!(item.prev_world_matrix, baseline_world);
            assert_ne!(item.world_matrix, baseline_world);
        }

        // The acquisition failure must not publish the moved transform. The
        // next extraction therefore still renders motion from the last frame
        // that actually reached queue submission.
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 2);
        let submitted = renderer
            .latest_rdg_diagnostics()
            .expect("recovery diagnostics missing");
        assert!(submitted.queue_submitted());
        assert_eq!(submitted.render_world().submitted_history_settled(), 0);
        {
            let state = renderer.context.as_ref().expect("renderer state missing");
            let item = state
                .render_frame
                .extracted_scene
                .render_items
                .iter()
                .find(|item| item.node_handle == mesh_node)
                .expect("recovery render-world item missing");
            assert_eq!(item.prev_world_matrix, baseline_world);
            assert_ne!(item.world_matrix, baseline_world);
        }

        // Settlement is deliberately replayed at the next extraction, after
        // the submitted graph no longer borrows the render-world records.
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 3);
        let settled = renderer
            .latest_rdg_diagnostics()
            .expect("settlement diagnostics missing");
        assert!(settled.queue_submitted());
        assert_eq!(settled.render_world().submitted_history_settled(), 1);
        let state = renderer.context.as_ref().expect("renderer state missing");
        let item = state
            .render_frame
            .extracted_scene
            .render_items
            .iter()
            .find(|item| item.node_handle == mesh_node)
            .expect("settled render-world item missing");
        assert_eq!(item.prev_world_matrix, item.world_matrix);
        assert_ne!(item.world_matrix, baseline_world);

        let settled_world = item.world_matrix;
        scene.edit_node(mesh_node).set_position(4.0, 0.0, 0.0);
        scene.update_matrix_world();
        let abandoned = renderer
            .begin_frame(
                &mut scene,
                make_camera(),
                &assets,
                FrameTime {
                    time: 4.0 / 60.0,
                    delta_time: 1.0 / 60.0,
                    frame_count: 4,
                },
            )
            .expect("abandoned transform frame composer missing");
        drop(abandoned);

        let diagnostics = renderer
            .latest_rdg_diagnostics()
            .expect("abandoned transform diagnostics missing");
        assert_eq!(diagnostics.outcome(), RdgAttemptOutcome::Abandoned);
        assert!(!diagnostics.queue_submitted());
        {
            let state = renderer.context.as_ref().expect("renderer state missing");
            let item = state
                .render_frame
                .extracted_scene
                .render_items
                .iter()
                .find(|item| item.node_handle == mesh_node)
                .expect("abandoned-frame render-world item missing");
            assert_eq!(item.prev_world_matrix, settled_world);
            assert_ne!(item.world_matrix, settled_world);
        }

        // Dropping a fully composed CPU graph is the other unsubmitted-frame
        // boundary. It must preserve the same baseline and retained delta as
        // target acquisition failure.
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 5);
        let state = renderer.context.as_ref().expect("renderer state missing");
        let item = state
            .render_frame
            .extracted_scene
            .render_items
            .iter()
            .find(|item| item.node_handle == mesh_node)
            .expect("post-abandon render-world item missing");
        assert_eq!(item.prev_world_matrix, settled_world);
        assert_ne!(item.world_matrix, settled_world);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 6);
        let diagnostics = renderer
            .latest_rdg_diagnostics()
            .expect("post-abandon settlement diagnostics missing");
        assert_eq!(diagnostics.render_world().submitted_history_settled(), 1);
        let state = renderer.context.as_ref().expect("renderer state missing");
        let item = state
            .render_frame
            .extracted_scene
            .render_items
            .iter()
            .find(|item| item.node_handle == mesh_node)
            .expect("post-abandon settled render-world item missing");
        assert_eq!(item.prev_world_matrix, item.world_matrix);
        assert_ne!(item.world_matrix, settled_world);
    }

    #[test]
    fn pre_materialization_exits_publish_zero_work_and_preserve_warm_state() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();

        renderer
            .begin_frame(
                &mut scene,
                make_camera(),
                &assets,
                FrameTime {
                    frame_count: 31,
                    ..FrameTime::default()
                },
            )
            .expect("frame composer missing")
            .with_forced_surface_acquisition_failure(RdgSurfaceAcquireFailure::Timeout)
            .render();

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_global_state_unmaterialized(state, scene.id());
        assert_local_light_buffers_unmaterialized(state);
        let diagnostics = renderer
            .latest_rdg_diagnostics()
            .expect("surface-failure diagnostics missing");
        assert_eq!(
            diagnostics.outcome(),
            RdgAttemptOutcome::SurfaceAcquisitionFailed(RdgSurfaceAcquireFailure::Timeout)
        );
        assert_eq!(diagnostics.phase(), RdgFramePhase::AcquireTarget);
        assert!(diagnostics.declared_passes() > 0);
        assert!(diagnostics.live_passes().is_some());
        assert_eq!(diagnostics.prepared_passes(), 0);
        assert_eq!(diagnostics.executed_passes(), 0);
        assert_eq!(diagnostics.target(), None);
        assert!(!diagnostics.queue_submitted());

        let mut renderer = init_headless_renderer();
        let mut scene = Scene::new();
        renderer.size.0 += 1;
        renderer
            .begin_frame(
                &mut scene,
                make_camera(),
                &assets,
                FrameTime {
                    frame_count: 32,
                    ..FrameTime::default()
                },
            )
            .expect("frame composer missing")
            .render();
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_global_state_unmaterialized(state, scene.id());
        assert_local_light_buffers_unmaterialized(state);
        let diagnostics = renderer.latest_rdg_diagnostics().unwrap();
        assert_eq!(
            diagnostics.outcome(),
            RdgAttemptOutcome::ResourceMaterializationFailed(
                RdgResourceMaterializationFailure::RenderTargetDescriptorDrift,
            )
        );
        assert_eq!(diagnostics.prepared_passes(), 0);
        assert_eq!(diagnostics.executed_passes(), 0);
        assert!(!diagnostics.queue_submitted());

        let mut renderer = init_headless_renderer();
        let mut scene = Scene::new();
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);
        let before = global_buffer_ids(renderer.context.as_ref().unwrap()).map(|cpu_id| {
            let gpu = renderer
                .context
                .as_ref()
                .unwrap()
                .resource_manager
                .get_gpu_buffer_by_cpu_id(cpu_id)
                .expect("warm global buffer missing");
            (gpu.id, gpu.last_uploaded_version, gpu.last_used_frame)
        });
        renderer.context.as_mut().unwrap().wgpu_ctx.headless_texture = None;
        renderer
            .begin_frame(
                &mut scene,
                make_camera(),
                &assets,
                FrameTime {
                    frame_count: 1,
                    ..FrameTime::default()
                },
            )
            .expect("frame composer missing")
            .render();
        let after = global_buffer_ids(renderer.context.as_ref().unwrap()).map(|cpu_id| {
            let gpu = renderer
                .context
                .as_ref()
                .unwrap()
                .resource_manager
                .get_gpu_buffer_by_cpu_id(cpu_id)
                .expect("warm global buffer missing after no-target attempt");
            (gpu.id, gpu.last_uploaded_version, gpu.last_used_frame)
        });
        assert_eq!(after, before);
        let diagnostics = renderer.latest_rdg_diagnostics().unwrap();
        assert_eq!(diagnostics.outcome(), RdgAttemptOutcome::NoRenderTarget);
        assert_eq!(diagnostics.declared_passes(), 0);
        assert_eq!(diagnostics.live_passes(), None);
        assert_eq!(diagnostics.prepared_passes(), 0);
        assert_eq!(diagnostics.executed_passes(), 0);
        assert!(!diagnostics.queue_submitted());
    }

    fn local_light_storage_state(state: &RendererState) -> (u64, u64, u64, u64) {
        let cpu = &state
            .render_frame
            .extracted_scene
            .local_light_storage_buffer;
        let stamp = cpu.stamp();
        let gpu = state
            .resource_manager
            .get_gpu_buffer_by_cpu_id(stamp.id)
            .expect("local-light storage physical buffer missing");
        let logical_size = state
            .graph_storage
            .resources
            .iter()
            .find(|resource| resource.name == "Scene_Local_Lights")
            .expect("logical scene-light storage import missing")
            .buffer_desc()
            .logical_size;
        assert_eq!(logical_size, stamp.size as u64);
        assert_eq!(gpu.last_uploaded_version, stamp.version);
        (gpu.id, gpu.size, logical_size, stamp.version)
    }

    #[test]
    fn local_light_resize_reuses_physical_high_water_with_exact_logical_ranges() {
        let mut renderer = Renderer::new(
            RendererInitConfig::default(),
            RendererSettings {
                path: RenderPath::BasicForward,
                ..RendererSettings::default()
            },
        );
        init_shared_headless_renderer(&mut renderer, 128, 128, None);
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let mut lights = (0..4)
            .map(|_| scene.add_light(Light::new_point(Vec3::ONE, 1.0, 100.0)))
            .collect::<Vec<_>>();

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);
        let four =
            local_light_storage_state(renderer.context.as_ref().expect("renderer state missing"));
        let element_size = std::mem::size_of::<myth_resources::uniforms::GpuLightStorage>() as u64;
        assert_eq!(four.2, 4 * element_size);

        for light in lights.drain(1..) {
            scene.remove_node(light);
        }
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);
        let one =
            local_light_storage_state(renderer.context.as_ref().expect("renderer state missing"));
        assert_eq!(one.2, element_size);
        assert_eq!(one.0, four.0, "shrinking must retain physical identity");
        assert_eq!(one.1, four.1, "shrinking must retain high-water capacity");

        lights.extend((0..3).map(|_| scene.add_light(Light::new_point(Vec3::ONE, 1.0, 100.0))));
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 2);
        let four_again =
            local_light_storage_state(renderer.context.as_ref().expect("renderer state missing"));
        assert_eq!(four_again.2, four.2);
        assert_eq!(four_again.0, four.0);
        assert_eq!(four_again.1, four.1);
        assert!(one.3 > four.3);
        assert!(four_again.3 > one.3);
    }

    fn published_pmrem_view_id(state: &RendererState, scene_id: u32) -> Option<u64> {
        state
            .environment_cache
            .published_pmrem_view(scene_id, &state.history_registry)
            .map(Tracked::id)
    }

    fn graph_contains_pass(state: &RendererState, name: &str) -> bool {
        state
            .graph_storage
            .passes
            .iter()
            .any(|pass| pass.name == name)
    }

    fn execution_contains_pass(state: &RendererState, name: &str) -> bool {
        state
            .graph_storage
            .execution_queue
            .iter()
            .any(|&index| state.graph_storage.passes[index].name == name)
    }

    #[cfg(feature = "3dgs")]
    #[test]
    fn gaussian_extract_and_dead_path_keep_all_gpu_caches_cold() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let cloud = assets.gaussian_clouds.add(tiny_gaussian_cloud());
        let mut scene = Scene::new();
        scene.add_gaussian_cloud("extract-only-a", cloud);
        scene.add_gaussian_cloud("extract-only-b", cloud);
        let camera = make_camera();

        let state = renderer.context.as_mut().expect("renderer state missing");
        assert!(state.gaussian_splatting_pass.cache_is_empty());
        let extract =
            state
                .gaussian_splatting_pass
                .extract(&scene, &assets, &camera, &state.wgpu_ctx.device);

        let extract = extract.expect("the visible loaded cloud must be extracted");
        assert_eq!(
            extract.asset_count(),
            1,
            "two instances of one exact asset version share one frame owner"
        );
        assert_eq!(extract.plan_asset_indices(), [0, 0]);
        assert_eq!(extract.asset_version(0), 1);
        assert!(
            state.gaussian_splatting_pass.cache_is_empty(),
            "extract must not create layouts, pipelines, or cloud GPU buffers"
        );
        drop(extract);

        let mut basic_forward = Renderer::new(
            RendererInitConfig::default(),
            RendererSettings {
                path: RenderPath::BasicForward,
                ..RendererSettings::default()
            },
        );
        init_shared_headless_renderer(&mut basic_forward, 128, 128, None);
        render_frame(&mut basic_forward, &mut scene, camera, &assets, 0);
        let state = basic_forward
            .context
            .as_ref()
            .expect("renderer state missing");
        assert!(state.gaussian_splatting_pass.cache_is_empty());
        assert!(!graph_contains_pass(state, "GS_Compute"));
        assert!(!graph_contains_pass(state, "GS_Render"));
        assert!(!graph_contains_pass(state, "GS_Composite"));
    }

    #[cfg(feature = "3dgs")]
    #[test]
    fn gaussian_live_cache_reuses_replaces_and_retires_exact_resources() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let cloud = assets.gaussian_clouds.add(tiny_gaussian_cloud());
        let mut scene = Scene::new();
        scene.add_gaussian_cloud("live-cache", cloud);
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let (first_static, first_cloud, first_last_used) = {
            let state = renderer.context.as_ref().expect("renderer state missing");
            assert!(graph_contains_pass(state, "GS_Compute"));
            assert!(graph_contains_pass(state, "GS_Render"));
            assert!(graph_contains_pass(state, "GS_Composite"));
            assert!(state.gaussian_splatting_pass.static_cache_is_initialized());
            assert_eq!(state.gaussian_splatting_pass.cloud_cache_len(), 1);
            assert!(
                state
                    .graph_storage
                    .gaussian_frame_batches
                    .iter()
                    .all(Option::is_none),
                "exact-version asset owners must be released after live prepare"
            );
            (
                state.gaussian_splatting_pass.static_cache_identity(),
                state
                    .gaussian_splatting_pass
                    .cloud_cache_identity(cloud)
                    .expect("live prepare must upload the cloud"),
                state
                    .gaussian_splatting_pass
                    .cloud_last_used_frame(cloud)
                    .expect("live cloud must record its frame"),
            )
        };

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let second_last_used = {
            let state = renderer.context.as_ref().expect("renderer state missing");
            assert_eq!(
                state.gaussian_splatting_pass.static_cache_identity(),
                first_static,
                "a stable second frame must reuse every Gaussian layout and pipeline"
            );
            assert_eq!(
                state.gaussian_splatting_pass.cloud_cache_identity(cloud),
                Some(first_cloud),
                "a stable asset version must reuse both persistent cloud buffers"
            );
            state
                .gaussian_splatting_pass
                .cloud_last_used_frame(cloud)
                .expect("second live prepare must touch the cloud")
        };
        assert!(second_last_used > first_last_used);

        assets
            .gaussian_clouds
            .insert_ready(cloud, tiny_gaussian_cloud());
        assert_eq!(assets.gaussian_clouds.get_version(cloud), Some(1));
        render_frame(&mut renderer, &mut scene, camera, &assets, 2);
        let state = renderer.context.as_mut().expect("renderer state missing");
        let replacement = state
            .gaussian_splatting_pass
            .cloud_cache_identity(cloud)
            .expect("replacement cloud must be uploaded");
        assert_eq!(
            state.gaussian_splatting_pass.static_cache_identity(),
            first_static
        );
        assert_ne!(
            (replacement.1, replacement.2),
            (first_cloud.1, first_cloud.2),
            "same handle/version with a new exact asset identity must rebuild cloud buffers"
        );
        let replacement_last_used = state
            .gaussian_splatting_pass
            .cloud_last_used_frame(cloud)
            .expect("replacement cloud must record its frame");

        const TEST_TTL: u64 = 3;
        assert_eq!(
            state
                .gaussian_splatting_pass
                .garbage_collect(replacement_last_used + TEST_TTL, TEST_TTL),
            0,
            "age == TTL remains live"
        );
        assert_eq!(state.gaussian_splatting_pass.cloud_cache_len(), 1);
        assert_eq!(
            state
                .gaussian_splatting_pass
                .garbage_collect(replacement_last_used + TEST_TTL + 1, TEST_TTL),
            1,
            "age == TTL + 1 retires the persistent upload"
        );
        assert_eq!(state.gaussian_splatting_pass.cloud_cache_len(), 0);
        assert!(
            state.gaussian_splatting_pass.static_cache_is_initialized(),
            "cloud TTL must not evict cross-cloud static pipelines"
        );
    }

    #[test]
    fn inactive_and_pending_environment_sources_preserve_fallback_or_last_publication() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let scene_id = scene.id();
        assert!(
            state.resource_manager.gpu_environment(scene_id).is_none(),
            "a color background without IBL must not create scene environment metadata"
        );
        assert!(
            !state.environment_cache.has_scene_state(scene_id),
            "an inactive environment must not create a logical cache domain"
        );
        assert!(
            published_pmrem_view_id(state, scene_id).is_none(),
            "an inactive environment must not materialize a persistent PMREM"
        );

        let global_state = state
            .resource_manager
            .get_global_state(state.render_frame.render_state.id, scene_id)
            .expect("global state missing after render");
        let resource_ids = global_state.resource_ids.as_slice();
        let black_cube_id = state.resource_manager.system_textures.black_cube.id();
        assert_eq!(black_cube_id, resource_ids[3]);

        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.1, 0.2, 0.3, 1.0),
            Vec4::new(0.01, 0.02, 0.03, 1.0),
        ));
        render_frame(&mut renderer, &mut scene, camera, &assets, 1);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(state.resource_manager.gpu_environment(scene_id).is_none());
        assert!(!state.environment_cache.has_scene_state(scene_id));
        assert!(published_pmrem_view_id(state, scene_id).is_none());

        let background_texture = assets.checkerboard(16, 2);
        scene
            .background
            .set_mode(BackgroundMode::planar(background_texture, 1.0));
        render_frame(&mut renderer, &mut scene, camera, &assets, 2);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(state.resource_manager.gpu_environment(scene_id).is_none());
        assert!(!state.environment_cache.has_scene_state(scene_id));
        assert!(published_pmrem_view_id(state, scene_id).is_none());

        scene.environment.set_base_cube_size(16);
        scene.environment.set_pmrem_size(8);
        scene
            .environment
            .set_env_map(Some(TextureSource::Attachment(
                u64::MAX,
                wgpu::TextureViewDimension::D2,
            )));
        render_frame(&mut renderer, &mut scene, camera, &assets, 3);
        let state = renderer.context.as_ref().expect("renderer state missing");
        let pending = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("configured pending environment metadata missing");
        assert!(!pending.source_ready);
        assert!(state.environment_cache.has_scene_state(scene_id));
        assert!(published_pmrem_view_id(state, scene_id).is_none());
        let global = state
            .resource_manager
            .get_global_state(state.render_frame.render_state.id, scene_id)
            .expect("global state missing for pending source");
        assert_eq!(
            global.resource_ids.as_slice()[3],
            state.resource_manager.system_textures.black_cube.id()
        );

        scene.environment.set_env_map(None::<TextureSource>);
        scene.background.set_mode(BackgroundMode::procedural());
        render_frame(&mut renderer, &mut scene, camera, &assets, 4);
        let published = published_pmrem_view_id(
            renderer.context.as_ref().expect("renderer state missing"),
            scene_id,
        )
        .expect("procedural environment must publish a PMREM");
        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            params.set_starbox_texture(Some(TextureSource::Attachment(
                u64::MAX,
                wgpu::TextureViewDimension::D2,
            )));
        }
        render_frame(&mut renderer, &mut scene, camera, &assets, 5);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(
            !state
                .resource_manager
                .gpu_environment(scene_id)
                .expect("procedural metadata missing")
                .source_ready
        );
        assert_eq!(published_pmrem_view_id(state, scene_id), Some(published));
        assert!(graph_contains_pass(state, "Atmosphere_SkyView"));
        assert!(!graph_contains_pass(state, "Atmosphere_SkyToCube"));
        assert!(!graph_contains_pass(state, "IBL_Compute"));
    }

    #[test]
    fn one_pixel_pmrem_is_published_even_though_its_max_mip_is_zero() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.environment.set_base_cube_size(4);
        scene.environment.set_pmrem_size(1);
        scene.background.set_mode(BackgroundMode::procedural());

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        assert_eq!(
            renderer.published_environment_max_mip_level(scene.id()),
            Some(0.0),
            "max mip zero is valid metadata for a published 1×1 PMREM"
        );
        let state = renderer.context.as_ref().expect("renderer state missing");
        let global = state
            .resource_manager
            .get_global_state(state.render_frame.render_state.id, scene.id())
            .expect("global state missing");
        assert_ne!(
            global.resource_ids.as_slice()[3],
            state.resource_manager.system_textures.black_cube.id(),
            "PMREM availability must not be inferred from max_mip > 0"
        );
    }

    #[test]
    fn environment_domain_reuses_before_ttl_then_retires_and_recreates_atomically() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.environment.set_base_cube_size(16);
        scene.environment.set_pmrem_size(8);
        scene.background.set_mode(BackgroundMode::procedural());
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let scene_id = scene.id();
        let state = renderer.context.as_ref().expect("renderer state missing");
        let active_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("active scene environment metadata missing");
        assert_eq!(16, active_env.base_cube_size);
        assert_eq!(8, active_env.pmrem_size);
        assert!(active_env.source_ready);
        let last_active_frame = active_env.last_used_frame;
        let active_source_version = active_env.source_version;
        let pmrem_id = published_pmrem_view_id(state, scene_id)
            .expect("active environment must publish a PMREM");

        scene
            .background
            .set_mode(BackgroundMode::color(0.02, 0.03, 0.04));
        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let dormant_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("environment should remain available until TTL pruning");
        assert_eq!(
            last_active_frame, dormant_env.last_used_frame,
            "an inactive environment must not refresh its cache lifetime"
        );

        let global_state = state
            .resource_manager
            .get_global_state(state.render_frame.render_state.id, scene_id)
            .expect("global state missing after disabling environment");
        let black_cube_id = state.resource_manager.system_textures.black_cube.id();
        assert_eq!(black_cube_id, global_state.resource_ids.as_slice()[3]);
        assert!(!graph_contains_pass(state, "Skybox_Pass"));
        assert_eq!(Some(pmrem_id), published_pmrem_view_id(state, scene_id));
        assert!(state.environment_cache.has_scene_state(scene_id));
        assert!(
            state.atmosphere_pass.has_scene_state(scene_id),
            "procedural LUTs should remain reusable until their feature TTL expires"
        );

        scene.background.set_mode(BackgroundMode::procedural());
        render_frame(&mut renderer, &mut scene, camera, &assets, 2);
        let state = renderer.context.as_ref().expect("renderer state missing");
        let reenabled = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("re-enabled environment metadata missing");
        assert!(reenabled.last_used_frame > last_active_frame);
        assert_eq!(published_pmrem_view_id(state, scene_id), Some(pmrem_id));

        scene
            .background
            .set_mode(BackgroundMode::color(0.02, 0.03, 0.04));
        render_frame(&mut renderer, &mut scene, camera, &assets, 3);

        let state = renderer.context.as_mut().expect("renderer state missing");
        for _ in 0..121 {
            state.resource_manager.next_frame();
        }
        let current_frame = state.resource_manager.frame_index();
        let retired = state
            .environment_cache
            .maintain(&mut state.history_registry, current_frame);
        state
            .atmosphere_pass
            .maintain(&mut state.history_registry, current_frame);
        assert_eq!(vec![scene_id], retired);
        state.resource_manager.remove_scene_global_states(scene_id);
        state.resource_manager.prune(1);
        assert!(state.resource_manager.gpu_environment(scene_id).is_none());
        assert!(!state.environment_cache.has_scene_state(scene_id));
        assert!(!state.atmosphere_pass.has_scene_state(scene_id));
        assert!(published_pmrem_view_id(state, scene_id).is_none());
        assert_ne!(0, active_source_version);

        scene.environment.set_base_cube_size(32);
        scene.environment.set_pmrem_size(16);
        scene.background.set_mode(BackgroundMode::procedural());
        render_frame(&mut renderer, &mut scene, camera, &assets, 4);
        let state = renderer.context.as_ref().expect("renderer state missing");
        let recreated = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("retired environment must be recreated");
        assert_eq!((recreated.base_cube_size, recreated.pmrem_size), (32, 16));
        assert!(recreated.source_ready);
        assert_ne!(published_pmrem_view_id(state, scene_id), Some(pmrem_id));
    }

    #[test]
    fn environment_content_changes_reuse_views_while_spec_changes_reallocate() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.environment.set_base_cube_size(16);
        scene.environment.set_pmrem_size(8);
        let attachment_id = 0xCAFE_BABE;

        {
            let state = renderer.context.as_mut().expect("renderer state missing");
            let texture = state
                .wgpu_ctx
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("Environment Attachment Revision Test"),
                    size: wgpu::Extent3d {
                        width: 4,
                        height: 4,
                        depth_or_array_layers: 6,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Environment Attachment Revision Test Cube View"),
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });
            state
                .resource_manager
                .register_internal_texture_direct(attachment_id, view);
        }
        scene
            .environment
            .set_env_map(Some(TextureSource::Attachment(
                attachment_id,
                wgpu::TextureViewDimension::Cube,
            )));
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let scene_id = scene.id();
        let state = renderer.context.as_ref().expect("renderer state missing");
        let initial_source_version = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("environment metadata missing")
            .source_version;
        let initial_revision = state
            .resource_manager
            .get_internal_texture_revision(attachment_id)
            .expect("attachment revision missing");
        let initial_pmrem_id =
            published_pmrem_view_id(state, scene_id).expect("initial PMREM must be published");

        let state = renderer.context.as_mut().expect("renderer state missing");
        assert!(
            state
                .resource_manager
                .mark_internal_texture_dirty(attachment_id)
        );
        let dirty_revision = state
            .resource_manager
            .get_internal_texture_revision(attachment_id)
            .expect("dirty attachment revision missing");
        assert_ne!(initial_revision, dirty_revision);

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_ne!(
            initial_source_version,
            state
                .resource_manager
                .gpu_environment(scene_id)
                .expect("updated environment metadata missing")
                .source_version,
            "an in-place attachment update must invalidate derived environment contents"
        );
        assert_eq!(
            Some(initial_pmrem_id),
            published_pmrem_view_id(state, scene_id),
            "content-only attachment updates should reuse the PMREM allocation"
        );

        let after_dirty_version = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("updated environment metadata missing")
            .source_version;
        scene
            .environment
            .set_env_map(Some(assets.checkerboard(32, 4)));
        render_frame(&mut renderer, &mut scene, camera, &assets, 2);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_ne!(
            state
                .resource_manager
                .gpu_environment(scene_id)
                .expect("switched environment metadata missing")
                .source_version,
            after_dirty_version
        );
        assert_eq!(
            published_pmrem_view_id(state, scene_id),
            Some(initial_pmrem_id)
        );

        scene.environment.set_base_cube_size(32);
        scene.environment.set_pmrem_size(16);
        render_frame(&mut renderer, &mut scene, camera, &assets, 3);
        let state = renderer.context.as_ref().expect("renderer state missing");
        let resized = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("resized environment metadata missing");
        assert_eq!((resized.base_cube_size, resized.pmrem_size), (32, 16));
        assert_ne!(
            published_pmrem_view_id(state, scene_id),
            Some(initial_pmrem_id)
        );
    }

    #[test]
    fn procedural_environment_rebake_policy_distinguishes_small_large_and_source_changes() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.environment.set_base_cube_size(16);
        scene.environment.set_pmrem_size(8);
        scene.background.set_mode(BackgroundMode::procedural());
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let scene_id = scene.id();
        let state = renderer.context.as_ref().expect("renderer state missing");
        let initial_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene environment metadata missing after first render");
        let initial_source_version = initial_env.source_version;
        let initial_last_used_frame = initial_env.last_used_frame;
        let initial_pmrem_id = published_pmrem_view_id(state, scene_id)
            .expect("initial persistent PMREM must be published");
        let render_state_id = state.render_frame.render_state.id;
        let initial_bind_group_id = state
            .resource_manager
            .get_global_state(render_state_id, scene_id)
            .expect("global state missing after first render")
            .bind_group_id;

        let (initial_sun_direction, initial_moon_direction) =
            if let BackgroundMode::Procedural(params) = &scene.background.mode {
                (params.sun_direction, params.moon_direction)
            } else {
                panic!("expected procedural background");
            };

        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            let slight_rotation = Quat::from_rotation_x(0.1_f32.to_radians());
            params.set_sun_direction((slight_rotation * initial_sun_direction).normalize());
            params.set_moon_direction((slight_rotation * initial_moon_direction).normalize());
        }

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let after_small_rotation = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene environment metadata missing after small sun rotation");
        let after_small_source_version = after_small_rotation.source_version;
        let after_small_bind_group_id = state
            .resource_manager
            .get_global_state(render_state_id, scene_id)
            .expect("global state missing after small sun rotation")
            .bind_group_id;

        assert_ne!(
            initial_source_version, after_small_source_version,
            "the CPU snapshot must still describe the current celestial directions"
        );
        assert!(after_small_rotation.last_used_frame > initial_last_used_frame);
        assert_eq!(
            Some(initial_pmrem_id),
            published_pmrem_view_id(state, scene_id)
        );
        assert_eq!(initial_bind_group_id, after_small_bind_group_id);
        assert!(
            !graph_contains_pass(state, "Atmosphere_SkyToCube")
                && !graph_contains_pass(state, "IBL_Compute"),
            "sub-threshold sun/moon motion must not schedule base or PMREM rewrites"
        );

        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            let larger_rotation = Quat::from_rotation_x(1.0_f32.to_radians());
            params.set_moon_direction((larger_rotation * initial_moon_direction).normalize());
        }

        render_frame(&mut renderer, &mut scene, camera, &assets, 2);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let after_large_rotation = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene environment metadata missing after large sun rotation");
        assert_ne!(
            after_small_source_version,
            after_large_rotation.source_version
        );
        assert_eq!(
            Some(initial_pmrem_id),
            published_pmrem_view_id(state, scene_id)
        );
        assert!(
            graph_contains_pass(state, "Atmosphere_SkyToCube")
                && graph_contains_pass(state, "IBL_Compute"),
            "moon motion beyond the threshold should rewrite base and PMREM in the same physical views"
        );

        let after_large_source_version = after_large_rotation.source_version;
        let starbox = assets.checkerboard(32, 8);
        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            params.set_starbox_texture(Some(starbox.into()));
            params.set_star_intensity(2.0);
        }
        render_frame(&mut renderer, &mut scene, camera, &assets, 3);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_ne!(
            state
                .resource_manager
                .gpu_environment(scene_id)
                .expect("updated environment metadata missing")
                .source_version,
            after_large_source_version
        );
        assert_eq!(
            published_pmrem_view_id(state, scene_id),
            Some(initial_pmrem_id)
        );
        assert!(graph_contains_pass(state, "Atmosphere_SkyToCube"));
        assert!(graph_contains_pass(state, "IBL_Compute"));
    }

    #[test]
    fn tone_mapping_rejects_non_3d_lut_without_stale_state() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let image = Image::new(
            2,
            2,
            2,
            ImageDimension::D3,
            PixelFormat::Rgba8Unorm,
            Some(vec![64; 2 * 2 * 2 * 4]),
        );
        let image_handle = assets.images.add(image);
        let mut texture = Texture::new_3d(Some("ToneMap Previous LUT"), image_handle);
        texture.color_space = ColorSpace::Linear;
        let valid_lut = assets.textures.add(texture);
        let invalid_lut = assets.checkerboard(8, 2);
        let mut scene = Scene::new();
        scene.tone_mapping.set_lut_texture(Some(valid_lut));
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(state.tone_map_pass.has_lut_pipeline());

        scene.tone_mapping.set_lut_texture(Some(invalid_lut));
        scene
            .tone_mapping
            .set_mode(myth_resources::tone_mapping::ToneMappingMode::Linear);
        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(graph_contains_pass(state, "ToneMap_Pass"));
        assert!(
            state.tone_map_pass.has_base_pipeline(),
            "an invalid replacement LUT must select the current no-LUT mode instead of reusing stale state"
        );
    }

    #[test]
    fn visible_material_texture_routes_first_frame_mipmaps_before_opaque_draw() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let image_handle = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![144; 8 * 8 * 4]),
        ));
        let mut texture = Texture::new_2d(Some("Visible Material Mip Test"), image_handle);
        texture.color_space = ColorSpace::Linear;
        texture.generate_mipmaps = true;
        let texture_handle = assets.textures.add(texture);
        let material = PhysicalMaterial::new(Vec4::ONE);
        material.set_map(Some(texture_handle));
        let mut scene = Scene::new();
        scene.spawn_box(1.0, 1.0, 1.0, material, &assets);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let gpu_image = state
            .resource_manager
            .get_texture_image(texture_handle)
            .expect("the visible material must materialize its texture");
        let physical_image_id = gpu_image.id;
        assert_eq!(gpu_image.mip_level_count, 4);
        assert!(gpu_image.mipmaps_submitted());
        assert!(state.mipmap_pass.generator_initialized());
        assert_eq!(
            state
                .graph_storage
                .passes
                .iter()
                .filter(|pass| pass.name == "Asset_Mipmap_Generation")
                .count(),
            0,
            "planned material mip work must be owned directly by its optional mip chain"
        );
        assert_eq!(
            state
                .graph_storage
                .passes
                .iter()
                .filter(|pass| pass.name == "Asset_Texture_Upload")
                .count(),
            0,
            "lazy Asset Texture imports must not allocate proxy upload passes"
        );
        assert_eq!(
            state
                .graph_storage
                .passes
                .iter()
                .filter(|pass| pass.name == "Asset_Texture_Mipmap_Generation")
                .count(),
            1
        );
        assert!(!graph_contains_pass(state, "Skybox_Pass"));

        let execution_names: Vec<_> = state
            .graph_storage
            .execution_queue
            .iter()
            .map(|&index| state.graph_storage.passes[index].name)
            .collect();
        let mipmap = execution_names
            .iter()
            .position(|&name| name == "Asset_Texture_Mipmap_Generation")
            .expect("logical material mipmap producer missing from execution queue");
        let opaque = execution_names
            .iter()
            .position(|&name| name == "Opaque_Pass")
            .expect("opaque consumer missing from execution queue");
        assert!(mipmap < opaque);
        assert!(
            state
                .graph_storage
                .mipmap_batches
                .iter()
                .all(Option::is_none),
            "submitted mipmap batches must release their physical owners"
        );

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(!graph_contains_pass(state, "Asset_Mipmap_Generation"));
        assert!(!graph_contains_pass(
            state,
            "Asset_Texture_Mipmap_Generation"
        ));
        assert_eq!(
            state
                .graph_storage
                .passes
                .iter()
                .filter(|pass| pass.name == "Asset_Texture_Upload")
                .count(),
            0,
            "the stable exact import must remain a resource, not a proxy pass"
        );
        assert_eq!(
            state
                .resource_manager
                .get_texture_image(texture_handle)
                .expect("material texture must remain resident")
                .id,
            physical_image_id
        );
    }

    #[test]
    fn prepass_only_live_graph_does_not_derive_a_dead_main_pipeline() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let image = assets.images.add(Image::new(
            4,
            4,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![192; 4 * 4 * 4]),
        ));
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("Prepass DrawPlan Texture"), image));
        let material = PhysicalMaterial::new(Vec4::ONE);
        material.set_map(Some(texture));
        let mut scene = Scene::new();
        scene.spawn_box(1.0, 1.0, 1.0, material, &assets);
        let camera = make_camera();
        let initial_main_l1_count = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .pipeline_cache
            .fast_pipeline_count();

        renderer
            .begin_frame(&mut scene, camera, &assets, FrameTime::default())
            .expect("frame composer must exist")
            .add_custom_pass(HookStage::BeforePostProcess, |graph, blackboard| {
                let depth = blackboard
                    .scene_depth
                    .expect("HighFidelity graph must publish Prepass depth");
                let output = graph.add_pass("Depth_Only_Test_Root", |builder| {
                    builder.read_texture(depth);
                    let output = builder.create_texture(
                        "Depth_Only_Test_Output",
                        TextureDesc::new_2d(
                            128,
                            128,
                            crate::HDR_TEXTURE_FORMAT,
                            wgpu::TextureUsages::RENDER_ATTACHMENT
                                | wgpu::TextureUsages::TEXTURE_BINDING,
                        ),
                    );
                    (DepthOnlyTestRoot, output)
                });
                GraphBlackboard {
                    scene_color: Some(output),
                    ..blackboard
                }
            })
            .render();

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(execution_contains_pass(state, "Pre_Pass"));
        assert!(execution_contains_pass(state, "Depth_Only_Test_Root"));
        assert!(!execution_contains_pass(state, "Opaque_Pass"));
        assert!(!execution_contains_pass(state, "Asset_Texture_Upload"));
        assert!(
            state.resource_manager.get_texture_image(texture).is_some(),
            "the live Prepass must materialize its lazy material texture import"
        );
        assert!(state.prepass.cached_pipeline_count() > 0);
        assert_eq!(
            state.pipeline_cache.fast_pipeline_count(),
            initial_main_l1_count,
            "a dead Opaque DrawPlan must not derive or cache a main pipeline"
        );
        assert_eq!(
            state.resource_manager.model_allocator.len(),
            1,
            "the visible source must still be materialized for the live Prepass plan"
        );
    }

    #[test]
    fn same_handle_asset_reloads_refresh_gpu_state_and_l1_without_duplicate_l2_pipelines() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let mesh_node = scene.spawn_box(
            1.0,
            1.0,
            1.0,
            PhysicalMaterial::new(Vec4::new(0.7, 0.7, 0.7, 1.0)),
            &assets,
        );
        let (geometry_handle, material_handle) = {
            let mesh = scene.meshes.get(mesh_node).expect("test mesh missing");
            (mesh.geometry, mesh.material)
        };
        let camera = make_camera();

        // Warm every steady-state L1/L2 path before recording the baseline.
        render_frame(&mut renderer, &mut scene, camera, &assets, 0);
        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let (
            initial_geometry_identity,
            initial_geometry_layout,
            initial_vertex_buffers,
            initial_material_identity,
            initial_material_layout,
            initial_material_bind_group,
            initial_prepass_l1_count,
            initial_l2_count,
        ) = {
            let state = renderer.context.as_ref().expect("renderer state missing");
            let geometry = state
                .resource_manager
                .get_geometry(geometry_handle)
                .expect("GPU geometry missing");
            let material = state
                .resource_manager
                .get_material(material_handle)
                .expect("GPU material missing");
            (
                geometry.asset_identity,
                geometry.layout_id,
                geometry.vertex_buffer_ids.clone(),
                material.asset_identity,
                material.layout_id,
                material.bind_group_id,
                state.prepass.cached_pipeline_count(),
                state.pipeline_cache.render_pipeline_count(),
            )
        };

        assets.geometries.insert_ready(
            geometry_handle,
            myth_resources::Geometry::new_box(1.0, 1.0, 1.0),
        );
        let (_, reloaded_geometry_version, reloaded_geometry_identity) = assets
            .geometries
            .get_entry_with_identity(geometry_handle)
            .expect("reloaded geometry missing");
        assert_eq!(reloaded_geometry_version, 1);
        assert_ne!(reloaded_geometry_identity, initial_geometry_identity);

        render_frame(&mut renderer, &mut scene, camera, &assets, 2);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let geometry = state
            .resource_manager
            .get_geometry(geometry_handle)
            .expect("reloaded GPU geometry missing");
        assert_eq!(geometry.asset_identity, reloaded_geometry_identity);
        assert_ne!(geometry.vertex_buffer_ids, initial_vertex_buffers);
        assert_eq!(geometry.layout_id, initial_geometry_layout);
        assert_eq!(
            state.prepass.cached_pipeline_count(),
            initial_prepass_l1_count + 1,
            "exact geometry identity must specialize the persistent Prepass L1"
        );
        assert_eq!(
            state.pipeline_cache.render_pipeline_count(),
            initial_l2_count,
            "an equivalent replacement should reuse canonical physical pipelines"
        );

        assets.materials.insert_ready(
            material_handle,
            myth_resources::Material::from(PhysicalMaterial::new(Vec4::new(0.7, 0.7, 0.7, 1.0))),
        );
        let (_, reloaded_material_version, reloaded_material_identity) = assets
            .materials
            .get_entry_with_identity(material_handle)
            .expect("reloaded material missing");
        assert_eq!(reloaded_material_version, 1);
        assert_ne!(reloaded_material_identity, initial_material_identity);

        render_frame(&mut renderer, &mut scene, camera, &assets, 3);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let material = state
            .resource_manager
            .get_material(material_handle)
            .expect("reloaded GPU material missing");
        assert_eq!(material.asset_identity, reloaded_material_identity);
        assert_ne!(material.bind_group_id, initial_material_bind_group);
        assert_eq!(material.layout_id, initial_material_layout);
        assert_eq!(
            state.prepass.cached_pipeline_count(),
            initial_prepass_l1_count + 2,
            "exact material identity must specialize the persistent Prepass L1"
        );
        assert_eq!(
            state.pipeline_cache.render_pipeline_count(),
            initial_l2_count,
            "the canonical L2 must still deduplicate equivalent replacement state"
        );
    }

    #[test]
    fn offscreen_draw_sources_do_not_materialize_gpu_assets() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let image = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![91; 8 * 8 * 4]),
        ));
        let mut texture_asset = Texture::new_2d(Some("offscreen material texture"), image);
        texture_asset.generate_mipmaps = true;
        let texture = assets.textures.add(texture_asset);
        let material = PhysicalMaterial::new(Vec4::new(0.6, 0.6, 0.6, 1.0));
        material.set_map(Some(texture));
        let mesh_node = scene.spawn_box(1.0, 1.0, 1.0, material, &assets);
        scene
            .edit_node(mesh_node)
            .set_position(1_000_000.0, 0.0, 0.0);
        scene.update_matrix_world();
        let (geometry_handle, material_handle) = {
            let mesh = scene.meshes.get(mesh_node).expect("test mesh missing");
            (mesh.geometry, mesh.material)
        };

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(state.render_lists.opaque.is_empty());
        assert!(state.render_lists.transparent.is_empty());
        assert_eq!(state.resource_manager.model_allocator.len(), 0);
        assert!(
            state
                .resource_manager
                .get_geometry(geometry_handle)
                .is_none()
        );
        assert!(
            state
                .resource_manager
                .get_material(material_handle)
                .is_none()
        );
        assert!(state.resource_manager.get_texture_image(texture).is_none());
        assert!(!graph_contains_pass(state, "Asset_Texture_Upload"));
        assert!(!graph_contains_pass(
            state,
            "Asset_Texture_Mipmap_Generation"
        ));
    }

    #[test]
    fn same_version_asset_replacement_after_extract_fails_the_graph_attempt_closed() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let mesh_node = scene.spawn_box(1.0, 1.0, 1.0, PhysicalMaterial::new(Vec4::ONE), &assets);
        let (geometry_handle, material_handle) = {
            let mesh = scene.meshes.get(mesh_node).expect("test mesh missing");
            (mesh.geometry, mesh.material)
        };
        let camera = make_camera();

        let composer = renderer
            .begin_frame(&mut scene, camera, &assets, FrameTime::default())
            .expect("frame composer must exist");
        assets.geometries.insert_ready(
            geometry_handle,
            myth_resources::Geometry::new_box(1.0, 1.0, 1.0),
        );
        assert_eq!(assets.geometries.get_version(geometry_handle), Some(1));
        composer.render();

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(
            state.render_lists.opaque.len(),
            1,
            "culling must retain the coherently extracted intent; the graph-build geometry qualification gate is the fail-closed asset-identity boundary"
        );
        assert_eq!(state.resource_manager.model_allocator.len(), 0);
        assert!(
            state
                .resource_manager
                .get_geometry(geometry_handle)
                .is_none()
        );
        assert!(
            state
                .resource_manager
                .get_material(material_handle)
                .is_none(),
            "a geometry generation rejected by the front-door qualification gate must suppress material preparation for that draw source"
        );

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);
        let old_gpu_material_identity = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .resource_manager
            .get_material(material_handle)
            .expect("material should recover on the next frame")
            .asset_identity;

        let composer = renderer
            .begin_frame(
                &mut scene,
                camera,
                &assets,
                FrameTime {
                    frame_count: 2,
                    ..FrameTime::default()
                },
            )
            .expect("frame composer must exist");
        assets.materials.insert_ready(
            material_handle,
            myth_resources::Material::from(PhysicalMaterial::new(Vec4::ONE)),
        );
        let (_, replacement_version, replacement_identity) = assets
            .materials
            .get_entry_with_identity(material_handle)
            .expect("replacement material missing");
        assert_eq!(replacement_version, 1);
        assert_ne!(replacement_identity, old_gpu_material_identity);
        composer.render();

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(
            state.render_lists.opaque.len(),
            1,
            "culling must retain the coherently extracted intent; MaterialPlan capture is the fail-closed asset-identity boundary"
        );
        assert_eq!(state.resource_manager.model_allocator.len(), 0);
        assert_eq!(
            state
                .resource_manager
                .get_material(material_handle)
                .expect("the previous persistent GPU material may remain cached")
                .asset_identity,
            old_gpu_material_identity,
            "the stale graph attempt must not publish the replacement into mixed old/new state"
        );

        render_frame(&mut renderer, &mut scene, camera, &assets, 3);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(state.render_lists.opaque.len(), 1);
        assert_eq!(
            state
                .resource_manager
                .get_material(material_handle)
                .expect("replacement material should recover on the next extraction")
                .asset_identity,
            replacement_identity
        );
    }

    #[test]
    fn post_prepass_material_epoch_drift_rejects_shared_direct_streams_and_l1() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let mesh = scene.spawn_box(1.0, 1.0, 1.0, PhysicalMaterial::new(Vec4::ONE), &assets);
        let material_handle = scene.meshes.get(mesh).expect("test mesh missing").material;
        let captured_epoch = assets
            .materials
            .get(material_handle)
            .expect("test material missing")
            .as_renderable()
            .version();
        let initial_prepass_l1_count = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .prepass
            .cached_pipeline_count();
        let drifted_opaque_command_count = std::sync::atomic::AtomicUsize::new(usize::MAX);
        let drifted_prepass_command_count = std::sync::atomic::AtomicUsize::new(usize::MAX);
        let drifted_opaque_command_count_ref = &drifted_opaque_command_count;
        let drifted_prepass_command_count_ref = &drifted_prepass_command_count;

        renderer
            .begin_frame(&mut scene, make_camera(), &assets, FrameTime::default())
            .expect("frame composer must exist")
            .add_custom_pass(HookStage::BeforePostProcess, move |graph, blackboard| {
                let scene_depth = blackboard
                    .scene_depth
                    .expect("HighFidelity graph must publish Prepass depth");
                let opaque_submission = scene_submission_for_test_pass(graph, "Opaque_Pass");
                let prepass_submission = scene_submission_for_test_pass(graph, "Pre_Pass");
                graph.add_pass("Material_Epoch_Drift_Test", |builder| {
                    builder.read_texture(scene_depth);
                    builder.mark_side_effect();
                    (
                        MaterialEpochGateTestRoot {
                            drift_material: Some(material_handle),
                            opaque_submission,
                            prepass_submission,
                            opaque_command_count: drifted_opaque_command_count_ref,
                            prepass_command_count: drifted_prepass_command_count_ref,
                        },
                        (),
                    )
                });
                blackboard
            })
            .render();

        let state = renderer.context.as_ref().expect("renderer state missing");
        let execution_names: Vec<_> = state
            .graph_storage
            .execution_queue
            .iter()
            .map(|&index| state.graph_storage.passes[index].name)
            .collect();
        let prepass_position = execution_names
            .iter()
            .position(|&name| name == "Pre_Pass")
            .expect("Prepass must remain DPE-live");
        let drift_position = execution_names
            .iter()
            .position(|&name| name == "Material_Epoch_Drift_Test")
            .expect("the side-effect epoch-drift pass must remain DPE-live");
        assert!(
            prepass_position < drift_position,
            "the test mutation must occur after Prepass gathered its recipe"
        );
        assert!(
            execution_names.contains(&"Opaque_Pass"),
            "the shared Opaque plan must remain DPE-live"
        );
        assert_ne!(
            assets
                .materials
                .get(material_handle)
                .expect("test material missing after drift")
                .as_renderable()
                .version(),
            captured_epoch
        );
        assert_eq!(
            state.prepass.cached_pipeline_count(),
            initial_prepass_l1_count,
            "a rejected final epoch must not publish the speculative Prepass pipeline into L1"
        );
        assert_eq!(
            drifted_opaque_command_count.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the final source gate must fail the Opaque stream closed"
        );
        assert_eq!(
            drifted_prepass_command_count.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the final source gate must fail the already-gathered Prepass stream closed"
        );

        let recovered_opaque_command_count = std::sync::atomic::AtomicUsize::new(usize::MAX);
        let recovered_prepass_command_count = std::sync::atomic::AtomicUsize::new(usize::MAX);
        let recovered_opaque_command_count_ref = &recovered_opaque_command_count;
        let recovered_prepass_command_count_ref = &recovered_prepass_command_count;
        renderer
            .begin_frame(
                &mut scene,
                make_camera(),
                &assets,
                FrameTime {
                    frame_count: 1,
                    ..FrameTime::default()
                },
            )
            .expect("recovery frame composer must exist")
            .add_custom_pass(HookStage::BeforePostProcess, move |graph, blackboard| {
                let scene_depth = blackboard
                    .scene_depth
                    .expect("HighFidelity graph must publish Prepass depth");
                let opaque_submission = scene_submission_for_test_pass(graph, "Opaque_Pass");
                let prepass_submission = scene_submission_for_test_pass(graph, "Pre_Pass");
                graph.add_pass("Material_Epoch_Recovery_Probe", |builder| {
                    builder.read_texture(scene_depth);
                    builder.mark_side_effect();
                    (
                        MaterialEpochGateTestRoot {
                            drift_material: None,
                            opaque_submission,
                            prepass_submission,
                            opaque_command_count: recovered_opaque_command_count_ref,
                            prepass_command_count: recovered_prepass_command_count_ref,
                        },
                        (),
                    )
                });
                blackboard
            })
            .render();

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(
            state.prepass.cached_pipeline_count() > initial_prepass_l1_count,
            "the next coherent attempt must publish the recovered Prepass L1"
        );
        assert_eq!(
            recovered_opaque_command_count.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            recovered_prepass_command_count.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn compact_frame_source_history_maps_back_to_the_nonzero_extracted_item() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        spawn_test_morph_box(&mut scene, &assets);
        spawn_test_morph_box(&mut scene, &assets);
        let mut nodes = scene
            .meshes
            .iter()
            .map(|(node, _)| node)
            .collect::<Vec<_>>();
        nodes.sort_unstable_by_key(|node| crate::graph::render_world::node_order(*node));
        let [culled_node, visible_node] = nodes.as_slice() else {
            panic!("expected exactly two test meshes");
        };
        let (culled_node, visible_node) = (*culled_node, *visible_node);
        scene
            .edit_node(culled_node)
            .set_position(1_000_000.0, 0.0, 0.0);
        for (node, influence) in [(culled_node, 0.25), (visible_node, 1.0)] {
            let mesh = scene.meshes.get_mut(node).expect("test mesh missing");
            mesh.set_morph_target_influence(0, influence);
        }
        scene.update_matrix_world();

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);
        {
            let state = renderer.context.as_ref().expect("renderer state missing");
            assert_eq!(state.render_frame.extracted_scene.render_items.len(), 2);
            assert_eq!(state.render_lists.frame_source_count(), 1);
            let extracted = state
                .render_lists
                .extracted_source(crate::graph::frame::FrameSourceId::from_index(0))
                .expect("visible frame source mapping missing");
            assert_eq!(extracted.index(), 1);
            assert_eq!(
                state.render_frame.extracted_scene.render_items[extracted.index()].node_handle,
                visible_node
            );
        }

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);
        let visible = scene
            .meshes
            .get(visible_node)
            .expect("visible test mesh missing")
            .morph_uniforms();
        assert!((visible.prev_weights[0].x - 1.0).abs() <= f32::EPSILON);
        drop(visible);

        let culled = scene
            .meshes
            .get(culled_node)
            .expect("culled test mesh missing")
            .morph_uniforms();
        assert!(culled.weights[0].x.abs() <= f32::EPSILON);
        assert!(culled.prev_weights[0].x.abs() <= f32::EPSILON);
    }

    #[test]
    fn group2_recipe_matrix_shares_rigid_l0_without_erasing_morph_or_skin_bindings() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let box_node = scene.spawn_box(1.0, 1.0, 1.0, PhysicalMaterial::new(Vec4::ONE), &assets);
        let material = scene
            .meshes
            .get(box_node)
            .expect("box mesh missing")
            .material;
        let sphere_node = scene.spawn_sphere(0.5, material, &assets);
        scene.edit_node(box_node).set_position(-0.75, 0.0, 0.0);
        scene.edit_node(sphere_node).set_position(0.75, 0.0, 0.0);
        scene.update_matrix_world();
        let rigid_morph_buffer_ids = [
            scene
                .meshes
                .get(box_node)
                .expect("box mesh missing")
                .morph_uniforms
                .id(),
            scene
                .meshes
                .get(sphere_node)
                .expect("sphere mesh missing")
                .morph_uniforms
                .id(),
        ];

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(state.render_lists.opaque.len(), 2);
        assert_eq!(state.resource_manager.rigid_object_bind_group_l0_len(), 1);
        assert_eq!(
            state.resource_manager.cached_object_bind_group_ids().len(),
            1
        );
        assert!(
            state.resource_manager.rigid_object_bind_group_l0_hits() > 0,
            "the second distinct rigid mesh must share the first model-only Group 2"
        );
        for morph_buffer_id in rigid_morph_buffer_ids {
            assert!(
                state
                    .resource_manager
                    .get_gpu_buffer_by_cpu_id(morph_buffer_id)
                    .is_none(),
                "a rigid mesh must not materialize its unused empty morph uniform"
            );
        }
        let hits_before = state.resource_manager.rigid_object_bind_group_l0_hits();
        let uploads_before = state.resource_manager.model_buffer_upload_count();
        let bind_groups_before = state.resource_manager.cached_object_bind_group_ids();

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(
            state.resource_manager.rigid_object_bind_group_l0_hits() > hits_before,
            "the stable rigid source must bypass ResourceBuilder on its warm frame"
        );
        assert_eq!(
            state.resource_manager.cached_object_bind_group_ids(),
            bind_groups_before,
            "an L0 hit must retain the exact persistent BindGroup owner"
        );
        assert_eq!(
            state.resource_manager.model_buffer_upload_count(),
            uploads_before,
            "an identical static model stream must not enqueue a redundant queue write"
        );

        let morph_node = spawn_test_morph_box(&mut scene, &assets);
        scene.edit_node(morph_node).set_position(-1.5, 0.0, 0.0);
        let morph_buffer_id = scene
            .meshes
            .get(morph_node)
            .expect("morph mesh missing")
            .morph_uniforms
            .id();

        let skinned_node =
            scene.spawn_box(0.5, 0.5, 0.5, PhysicalMaterial::new(Vec4::ONE), &assets);
        scene.edit_node(skinned_node).set_position(0.0, 0.0, 0.0);
        let skinned_morph_buffer_id = scene
            .meshes
            .get(skinned_node)
            .expect("skinned mesh missing")
            .morph_uniforms
            .id();
        let bone = scene.create_node();
        let skeleton = Skeleton::new(
            "Group2 Recipe Matrix Skeleton",
            vec![bone],
            vec![Affine3A::IDENTITY],
            0,
        );
        let skeleton_key = scene.add_skeleton(skeleton);
        scene.bind_skeleton(skinned_node, skeleton_key, BindMode::Attached);

        let stale_node = scene.spawn_box(0.5, 0.5, 0.5, PhysicalMaterial::new(Vec4::ONE), &assets);
        scene.edit_node(stale_node).set_position(1.5, 0.0, 0.0);
        let stale_skeleton = scene.add_skeleton(Skeleton::new(
            "Removed Group2 Recipe Skeleton",
            vec![bone],
            vec![Affine3A::IDENTITY],
            0,
        ));
        assert!(scene.skeleton_pool.remove(stale_skeleton).is_some());
        scene.bind_skeleton(stale_node, stale_skeleton, BindMode::Attached);
        scene.update_matrix_world();

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 2);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(state.resource_manager.rigid_object_bind_group_l0_len(), 1);
        for buffer_id in [morph_buffer_id, skinned_morph_buffer_id] {
            assert!(
                state
                    .resource_manager
                    .get_gpu_buffer_by_cpu_id(buffer_id)
                    .is_some(),
                "non-rigid recipes must preserve their full Group 2 bindings"
            );
        }
        let stale_item = state
            .render_frame
            .extracted_scene
            .render_items
            .iter()
            .find(|item| item.node_handle == stale_node)
            .expect("stale-skeleton item missing");
        assert!(stale_item.skeleton.is_none());
        assert_eq!(stale_item.item_variant_flags & (1 << 1), 0);
        assert!(!stale_item.item_shader_defines().contains("HAS_SKINNING"));
    }

    #[test]
    fn model_buffer_resize_and_ttl_recreation_never_leave_stale_object_bindings() {
        let mut renderer = init_headless_renderer_with_model_capacity(4);
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let mesh_node = scene.spawn_box(
            1.0,
            1.0,
            1.0,
            PhysicalMaterial::new(Vec4::new(0.5, 0.5, 0.5, 1.0)),
            &assets,
        );
        let template = scene
            .meshes
            .get(mesh_node)
            .expect("test mesh missing")
            .clone();

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(state.render_lists.opaque.len(), 1);
        let old_object_bind_group_ids = state.resource_manager.cached_object_bind_group_ids();
        assert!(
            !old_object_bind_group_ids.is_empty(),
            "the first submitted draw must populate the persistent object cache"
        );
        let model_buffer = state
            .resource_manager
            .get_gpu_buffer_by_cpu_id(state.resource_manager.model_buffer_id())
            .expect("model buffer missing");
        let old_model_physical_id = model_buffer.id;
        let old_model_size = model_buffer.size;
        let old_upload_count = state.resource_manager.model_buffer_upload_count();
        assert!(
            state
                .resource_manager
                .rigid_object_bind_group_l0_contains_model(old_model_physical_id)
        );

        for _ in 0..4 {
            scene.add_mesh(template.clone());
        }
        scene.update_matrix_world();

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(state.render_lists.opaque.len(), 5);
        let model_buffer = state
            .resource_manager
            .get_gpu_buffer_by_cpu_id(state.resource_manager.model_buffer_id())
            .expect("resized model buffer missing");
        assert!(model_buffer.size > old_model_size);
        assert_ne!(model_buffer.id, old_model_physical_id);
        assert!(
            state.resource_manager.model_buffer_upload_count() > old_upload_count,
            "a resized and expanded model stream must upload the new exact payload"
        );
        assert!(
            !state
                .resource_manager
                .rigid_object_bind_group_l0_contains_model(old_model_physical_id),
            "model-buffer resize must invalidate non-owning L0 entries"
        );
        for old_object_bind_group_id in &old_object_bind_group_ids {
            assert!(
                state
                    .resource_manager
                    .get_cached_bind_group(*old_object_bind_group_id)
                    .is_none(),
                "resizing must evict every object BindGroup that owns the old buffer"
            );
        }

        let stride = crate::core::gpu::ModelBufferAllocator::uniform_stride().get();
        assert!(
            state.render_lists.opaque.len() as u64 * stride <= model_buffer.size,
            "every live source model range must fit the final physical buffer"
        );
        let new_object_bind_group_ids = state.resource_manager.cached_object_bind_group_ids();
        assert!(!new_object_bind_group_ids.is_empty());
        assert!(
            new_object_bind_group_ids
                .iter()
                .all(|id| !old_object_bind_group_ids.contains(id))
        );
        let resized_physical_id = model_buffer.id;
        let resized_bind_group_id = new_object_bind_group_ids[0];

        let state = renderer.context.as_mut().expect("renderer state missing");
        let logical_id = state.resource_manager.model_buffer_id();
        state.resource_manager.next_frame();
        state.resource_manager.prune(0);
        assert!(
            state
                .resource_manager
                .get_gpu_buffer_by_cpu_id(logical_id)
                .is_none()
        );
        assert!(
            state
                .resource_manager
                .get_cached_bind_group(resized_bind_group_id)
                .is_none()
        );
        assert_eq!(state.resource_manager.rigid_object_bind_group_l0_len(), 0);

        state.resource_manager.flush_model_buffers();
        let recreated = state
            .resource_manager
            .get_gpu_buffer_by_cpu_id(logical_id)
            .expect("model buffer must be recreated before the next prepare");
        assert_ne!(recreated.id, resized_physical_id);
    }

    #[test]
    fn gpu_driven_unsupported_morph_reports_typed_required_failure() {
        // An unsupported morph source falls back before adding GPU topology
        // in Auto, while Required fails with a stable typed reason.
        let auto_assets = AssetServer::new();
        let mut auto_scene = Scene::new();
        spawn_test_morph_box(&mut auto_scene, &auto_assets);
        let mut auto = init_headless_basic_forward_gpu_renderer(GpuDrivenMode::Auto);
        render_frame(&mut auto, &mut auto_scene, make_camera(), &auto_assets, 0);
        let auto_state = auto.context.as_ref().expect("renderer state missing");
        assert!(!graph_contains_pass(auto_state, "GpuScene_Patch"));
        assert!(!graph_contains_pass(auto_state, "GpuScene_Cull"));
        assert_eq!(
            auto.latest_rdg_diagnostics().unwrap().outcome(),
            RdgAttemptOutcome::Submitted
        );

        let required_assets = AssetServer::new();
        let mut required_scene = Scene::new();
        spawn_test_morph_box(&mut required_scene, &required_assets);
        let mut required = init_headless_basic_forward_gpu_renderer(GpuDrivenMode::Required);
        render_frame(
            &mut required,
            &mut required_scene,
            make_camera(),
            &required_assets,
            0,
        );
        let outcome = required.latest_rdg_diagnostics().unwrap().outcome();
        if required
            .gpu_driven_capabilities()
            .is_some_and(GpuDrivenCapabilities::supports_indirect_primitives)
        {
            assert_eq!(
                outcome,
                RdgAttemptOutcome::GpuDrivenFailed(RdgGpuDrivenFailure::MorphTargets)
            );
        } else {
            assert_eq!(
                outcome,
                RdgAttemptOutcome::GpuDrivenFailed(RdgGpuDrivenFailure::UnsupportedDevice)
            );
        }
    }

    #[test]
    fn gpu_driven_required_basic_forward_closes_patch_cull_indirect_and_warm_publication() {
        let mut renderer = init_headless_basic_forward_gpu_renderer(GpuDrivenMode::Required);
        if !renderer
            .gpu_driven_capabilities()
            .is_some_and(GpuDrivenCapabilities::supports_indirect_primitives)
        {
            return;
        }
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let color = Vec4::new(0.7, 0.2, 0.1, 1.0);
        scene.spawn_box(1.0, 1.0, 1.0, UnlitMaterial::new(color), &assets);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);
        {
            let state = renderer.context.as_ref().expect("renderer state missing");
            assert_eq!(
                renderer.latest_rdg_diagnostics().unwrap().outcome(),
                RdgAttemptOutcome::Submitted
            );
            assert!(execution_contains_pass(state, "GpuScene_Patch"));
            assert!(execution_contains_pass(state, "GpuScene_Cull"));
            assert!(execution_contains_pass(state, "SimpleForward_Pass"));
            let main = renderer
                .latest_rdg_diagnostics()
                .expect("RDG diagnostics missing")
                .scene_submission_family(RdgSceneSubmissionFamily::MainPrepass);
            assert_eq!(
                main.selected_backend(),
                RdgSceneSubmissionBackend::GpuDriven
            );
            assert_eq!(
                main.selection_reason(),
                RdgSceneSubmissionSelectionReason::Required
            );
            assert_eq!(main.live_backend(), RdgSceneSubmissionBackend::GpuDriven);
            let policy = renderer
                .latest_rdg_diagnostics()
                .expect("RDG diagnostics missing")
                .gpu_driven_policy();
            assert_eq!(policy.extracted_item_count(), 1);
            assert_eq!(policy.retained_candidate_count(), 1);
            assert_eq!(policy.retained_bin_count(), 1);
            assert_eq!(policy.minimum_cohort_members(), 1);
            assert!(policy.exact_evaluated());
        }

        // The exact submitted table revision is acknowledged in the same
        // frame. A stable warm frame imports it read-only but still executes
        // Cull and the indirect draw.
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(!execution_contains_pass(state, "GpuScene_Patch"));
        assert!(execution_contains_pass(state, "GpuScene_Cull"));
    }

    #[test]
    fn gpu_driven_packed_shadow_families_match_cpu_and_change_lit_pixels() {
        // The portable tier binds each visible partition as an exact Group 2
        // range and keeps every indirect `first_instance` at zero.
        let mut gpu_renderer =
            init_headless_gpu_renderer(RenderPath::HighFidelity, GpuDrivenMode::Required);
        if !gpu_renderer
            .gpu_driven_capabilities()
            .is_some_and(GpuDrivenCapabilities::supports_indirect_primitives)
        {
            return;
        }
        let (gpu_assets, mut gpu_scene) = make_gpu_scene_shadow_fixture(true);

        render_frame(
            &mut gpu_renderer,
            &mut gpu_scene,
            make_shadow_camera(),
            &gpu_assets,
            0,
        );
        {
            let state = gpu_renderer
                .context
                .as_ref()
                .expect("renderer state missing");
            assert_eq!(
                gpu_renderer.latest_rdg_diagnostics().unwrap().outcome(),
                RdgAttemptOutcome::Submitted
            );
            assert!(execution_contains_pass(state, "GpuScene_Patch"));
            assert!(execution_contains_pass(state, "GpuScene_Cull"));
            assert!(execution_contains_pass(state, "GpuScene_Shadow_2D_Cull"));
            assert!(execution_contains_pass(state, "GpuScene_Shadow_Cube_Cull"));
            assert!(execution_contains_pass(state, "Shadow_2D_Pass"));
            assert!(execution_contains_pass(state, "Shadow_Cube_Pass"));
            assert_eq!(state.gpu_scene.active_batch_descriptors().len(), 2);
            assert_eq!(
                state
                    .graph_storage
                    .passes
                    .iter()
                    .filter(|pass| pass.name == "GpuScene_Shadow_2D_Cull")
                    .count(),
                1,
                "all directional/spot views must share one packed 2D-family Cull"
            );
            assert_eq!(
                state
                    .graph_storage
                    .passes
                    .iter()
                    .filter(|pass| pass.name == "GpuScene_Shadow_Cube_Cull")
                    .count(),
                1,
                "all point-light faces must share one packed cube-family Cull"
            );
        }
        let gpu_pixels = gpu_renderer
            .readback_pixels()
            .expect("GPU-driven shadow readback failed");

        // The retained CpuDirect renderer remains the executable oracle for
        // both shadow-map rasterization and final shadow sampling.
        let mut cpu_renderer =
            init_headless_gpu_renderer(RenderPath::HighFidelity, GpuDrivenMode::Off);
        let (cpu_assets, mut cpu_scene) = make_gpu_scene_shadow_fixture(true);
        render_frame(
            &mut cpu_renderer,
            &mut cpu_scene,
            make_shadow_camera(),
            &cpu_assets,
            0,
        );
        let cpu_pixels = cpu_renderer
            .readback_pixels()
            .expect("CpuDirect shadow readback failed");
        assert_eq!(gpu_pixels, cpu_pixels);

        // Prove that the fixture observes projected shadowing rather than
        // merely executing equivalent but visually unused shadow passes.
        let mut unshadowed_renderer =
            init_headless_gpu_renderer(RenderPath::HighFidelity, GpuDrivenMode::Off);
        let (unshadowed_assets, mut unshadowed_scene) = make_gpu_scene_shadow_fixture(false);
        render_frame(
            &mut unshadowed_renderer,
            &mut unshadowed_scene,
            make_shadow_camera(),
            &unshadowed_assets,
            0,
        );
        let unshadowed_pixels = unshadowed_renderer
            .readback_pixels()
            .expect("unshadowed oracle readback failed");
        let changed_pixels = cpu_pixels
            .chunks_exact(4)
            .zip(unshadowed_pixels.chunks_exact(4))
            .filter(|(shadowed, unshadowed)| shadowed[..3] != unshadowed[..3])
            .count();
        assert!(
            changed_pixels >= 32,
            "shadow-enabled fixture changed only {changed_pixels} pixels"
        );
    }

    #[test]
    fn gpu_driven_multi_bin_textures_retain_all_producers_and_match_cpu() {
        let mut gpu_renderer = init_headless_basic_forward_gpu_renderer(GpuDrivenMode::Required);
        if !gpu_renderer
            .gpu_driven_capabilities()
            .is_some_and(GpuDrivenCapabilities::supports_indirect_primitives)
        {
            return;
        }
        let (gpu_assets, mut gpu_scene, gpu_textures) = make_gpu_scene_textured_fixture();

        render_frame(
            &mut gpu_renderer,
            &mut gpu_scene,
            make_camera(),
            &gpu_assets,
            0,
        );
        let gpu_state = gpu_renderer
            .context
            .as_ref()
            .expect("renderer state missing");
        assert!(execution_contains_pass(gpu_state, "GpuScene_Patch"));
        assert!(execution_contains_pass(gpu_state, "GpuScene_Cull"));
        assert_eq!(gpu_state.gpu_scene.active_batch_descriptors().len(), 2);
        assert_eq!(
            gpu_state.render_lists.gpu_opaque_representatives().count(),
            2
        );
        for texture in gpu_textures {
            let image = gpu_state
                .resource_manager
                .get_texture_image(texture)
                .expect("every live GPU-driven material texture must be resident");
            assert_eq!(image.mip_level_count, 4);
            assert!(image.mipmaps_submitted());
        }
        let execution_names: Vec<_> = gpu_state
            .graph_storage
            .execution_queue
            .iter()
            .map(|&index| gpu_state.graph_storage.passes[index].name)
            .collect();
        let forward = execution_names
            .iter()
            .position(|&name| name == "SimpleForward_Pass")
            .expect("GPU-driven forward consumer missing");
        let mipmap_positions: Vec<_> = execution_names
            .iter()
            .enumerate()
            .filter_map(|(index, &name)| {
                (name == "Asset_Texture_Mipmap_Generation").then_some(index)
            })
            .collect();
        assert_eq!(mipmap_positions.len(), 2);
        assert!(mipmap_positions.into_iter().all(|index| index < forward));
        let gpu_pixels = gpu_renderer
            .readback_pixels()
            .expect("textured multi-bin GPU Scene readback failed");
        let red_pixels = gpu_pixels
            .chunks_exact(4)
            .filter(|pixel| pixel[0] > pixel[1].saturating_add(48) && pixel[0] > pixel[2])
            .count();
        let blue_pixels = gpu_pixels
            .chunks_exact(4)
            .filter(|pixel| pixel[2] > pixel[1].saturating_add(48) && pixel[2] > pixel[0])
            .count();
        assert!(
            red_pixels >= 32,
            "red texture affected only {red_pixels} pixels"
        );
        assert!(
            blue_pixels >= 32,
            "blue texture affected only {blue_pixels} pixels"
        );

        let mut cpu_renderer = init_headless_basic_forward_gpu_renderer(GpuDrivenMode::Off);
        let (cpu_assets, mut cpu_scene, _) = make_gpu_scene_textured_fixture();
        render_frame(
            &mut cpu_renderer,
            &mut cpu_scene,
            make_camera(),
            &cpu_assets,
            0,
        );
        let cpu_pixels = cpu_renderer
            .readback_pixels()
            .expect("textured multi-bin CpuDirect readback failed");
        assert_eq!(gpu_pixels, cpu_pixels);
    }
}
