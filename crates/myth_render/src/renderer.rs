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
    HistoryRegistry, RdgDiagnosticsRecorder, RdgFrameDiagnostics, RdgFramePhase,
    RdgRendererAttemptGuard,
};
use crate::graph::frame::{RenderFrameFeatureConfig, RenderLists};
#[cfg(feature = "debug_view")]
use crate::graph::passes::DebugViewFeature;
#[cfg(feature = "3dgs")]
use crate::graph::passes::GaussianSplattingFeature;
use crate::graph::passes::{
    AtmosphereFeature, BloomFeature, CasFeature, ClusteredLightingFeature, EnvironmentCacheFeature,
    EquirectToCubeFeature, FxaaFeature, HiZFeature, IblComputeFeature, MipmapFeature,
    MsaaSyncFeature, OpaqueFeature, PrepassFeature, PresentFeature, ShadowFeature,
    SimpleForwardFeature, SkyboxFeature, SsaoFeature, SsgiFeature, SsrFeature, SsssFeature,
    TaaFeature, ToneMappingFeature, TransmissionCopyFeature, TransparentFeature,
};
use myth_assets::AssetServer;
use myth_core::Result;
use myth_resources::uniforms::GpuLightStorage;
use myth_scene::Scene;
use myth_scene::background::BackgroundMode;
use myth_scene::camera::RenderCamera;

use crate::core::{ResourceManager, WgpuContext};
use crate::graph::extracted::SceneFeatures;
use crate::graph::{FrameComposer, RenderFrame};
use crate::pipeline::{
    ColorTargetKey, ComputePipelineId, ComputePipelineKey, DepthStencilKey, FullscreenPipelineKey,
    MultisampleKey, PipelineCache, RenderPipelineId, ShaderCompilationOptions, ShaderManager,
    ShaderSource,
};
use crate::settings::{ClusteredShadingMode, RenderPath, RendererInitConfig, RendererSettings};

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
    pub(crate) graph_storage: GraphStorage,
    // pub(crate) sampler_registry: SamplerRegistry,
    pub(crate) transient_pool: TransientPool,
    pub(crate) history_registry: HistoryRegistry,
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

        self.context = Some(RendererState {
            wgpu_ctx,
            resource_manager,
            pipeline_cache: PipelineCache::new(),
            shader_manager: ShaderManager::new(),

            render_frame,
            render_lists: RenderLists::new(),
            global_bind_group_cache,

            rdg_diagnostics: RdgDiagnosticsRecorder::default(),
            graph_storage: GraphStorage::new(),
            transient_pool: TransientPool::new(),
            history_registry: HistoryRegistry::new(),
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

        // ── Frame Arena Lifecycle ───────────────────────────────────────
        // Invalidate every raw NodeSlot before resetting the arena that owns
        // its pointee. This also makes an abandoned Composer or a later surface
        // acquisition failure leave an empty, safe graph inspection state.
        state
            .rdg_diagnostics
            .start_renderer_attempt(frame_time.frame_count);
        let frame_attempt = RdgRendererAttemptGuard::new(&state.rdg_diagnostics);
        state.graph_storage.clear();
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
            transient_pool: &mut state.transient_pool,
            history_registry: &mut state.history_registry,
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

    /// Returns the accounted CPU backing capacity retained by the RDG between
    /// attempts, including topology, compile scratch, and attempt stores.
    /// Hash-table bytes are stable estimates rather than allocator-exact data.
    #[must_use]
    pub fn rdg_retained_capacity_stats(&self) -> Option<RdgRetainedCapacityStats> {
        self.context
            .as_ref()
            .map(|state| state.graph_storage.retained_capacity_stats())
    }

    /// Requests one pressure-driven retained-capacity trim. The largest
    /// eligible family is selected, but deallocation is deferred until the
    /// earliest safe `GraphStorage::clear` boundary after any active global
    /// trim cooldown has elapsed.
    pub fn request_rdg_retained_capacity_trim(&mut self) {
        if let Some(state) = &mut self.context {
            state.graph_storage.request_retained_capacity_trim();
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

    use crate::graph::core::{
        ExecuteContext, GraphBlackboard, HookStage, PassNode, RdgAttemptOutcome, RdgFramePhase,
        RdgRenderTargetKind, RdgSurfaceAcquireFailure, TextureDesc,
    };
    use glam::{Affine3A, Quat, Vec3, Vec4};
    use myth_assets::{AssetServer, SceneExt};
    use myth_core::SkeletonKey;
    #[cfg(feature = "3dgs")]
    use myth_resources::gaussian_splat::{GaussianCloud, GaussianSHCoefficients, GaussianSplat};
    use myth_resources::image::{ColorSpace, Image, ImageDimension, PixelFormat};
    use myth_resources::material::PhysicalMaterial;
    use myth_resources::texture::{Texture, TextureSource};
    use myth_scene::background::BackgroundMode;
    use myth_scene::camera::Camera;
    use myth_scene::light::Light;
    use myth_scene::{BindMode, Scene};

    struct DepthOnlyTestRoot;

    impl PassNode<'_> for DepthOnlyTestRoot {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    fn init_headless_renderer() -> Renderer {
        let mut renderer =
            Renderer::new(RendererInitConfig::default(), RendererSettings::default());
        pollster::block_on(renderer.init_headless(128, 128, None))
            .expect("headless renderer init failed");
        renderer
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

    fn assert_local_light_buffers_materialized(state: &RendererState) {
        let scene = &state.render_frame.extracted_scene;
        for (id, handle, stamp) in [
            (
                scene.local_light_metadata_buffer.id(),
                scene.local_light_metadata_buffer.gpu_handle(),
                scene.local_light_metadata_buffer.stamp(),
            ),
            (
                scene.local_light_storage_buffer.id(),
                scene.local_light_storage_buffer.gpu_handle(),
                scene.local_light_storage_buffer.stamp(),
            ),
        ] {
            assert!(
                handle.is_none(),
                "exact scene-buffer imports must not publish process-wide cached handles"
            );
            let gpu_buffer = state
                .resource_manager
                .get_gpu_buffer_by_cpu_id(id)
                .expect("live scene-light import has no resolvable physical buffer");
            assert_eq!(gpu_buffer.last_uploaded_version, stamp.version);
            assert_eq!(gpu_buffer.usage, stamp.usage);
            assert!(
                gpu_buffer.size >= stamp.size as u64,
                "scene-light physical allocation is smaller than the captured snapshot"
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

    fn assert_global_state_materialized_exact(state: &RendererState, scene_id: u32) {
        let stamps = [
            state.render_frame.render_state.uniforms().stamp(),
            state
                .render_frame
                .extracted_scene
                .environment_uniforms_buffer
                .stamp(),
            state
                .render_frame
                .extracted_scene
                .directional_light_storage_buffer
                .stamp(),
        ];
        let global = state
            .resource_manager
            .get_global_state(state.render_frame.render_state.id, scene_id)
            .expect("DPE-live scene passes must finalize Group 0");
        for (binding, stamp) in stamps.into_iter().enumerate() {
            let gpu = state
                .resource_manager
                .get_gpu_buffer_by_cpu_id(stamp.id)
                .expect("DPE-live Group 0 buffer missing");
            assert_eq!(gpu.last_uploaded_version, stamp.version);
            assert_eq!(gpu.usage, stamp.usage);
            assert!(gpu.size >= stamp.size as u64);
            assert_eq!(global.resource_ids.as_slice()[binding], gpu.id);
        }
    }

    fn global_gpu_state(
        state: &RendererState,
        scene_id: u32,
    ) -> ([(u64, u64, u64, u64); 3], (u32, u64, u64, Vec<u64>, u64)) {
        let buffers = global_buffer_ids(state).map(|cpu_id| {
            let gpu = state
                .resource_manager
                .get_gpu_buffer_by_cpu_id(cpu_id)
                .expect("Group 0 physical buffer missing");
            (
                gpu.id,
                gpu.size,
                gpu.last_uploaded_version,
                gpu.last_used_frame,
            )
        });
        let global = state
            .resource_manager
            .get_global_state(state.render_frame.render_state.id, scene_id)
            .expect("persistent Group 0 state missing");
        (
            buffers,
            (
                global.id,
                global.bind_group_id,
                global.layout_id,
                global.resource_ids.as_slice().to_vec(),
                global.last_used_frame,
            ),
        )
    }

    fn local_light_gpu_state(state: &RendererState) -> [(u64, u64, u64, u64); 2] {
        let scene = &state.render_frame.extracted_scene;
        [
            scene.local_light_metadata_buffer.id(),
            scene.local_light_storage_buffer.id(),
        ]
        .map(|cpu_id| {
            let gpu = state
                .resource_manager
                .get_gpu_buffer_by_cpu_id(cpu_id)
                .expect("local-light physical buffer missing");
            (
                gpu.id,
                gpu.size,
                gpu.last_uploaded_version,
                gpu.last_used_frame,
            )
        })
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
    fn headless_renderer_publishes_pass_conservation_diagnostics() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 17);

        let diagnostics = renderer
            .latest_rdg_diagnostics()
            .expect("submitted frame diagnostics missing");
        let live = diagnostics
            .live_passes()
            .expect("submitted graph must have compiled topology");
        assert_eq!(diagnostics.renderer_frame(), Some(17));
        assert_eq!(diagnostics.outcome(), RdgAttemptOutcome::Submitted);
        assert_eq!(diagnostics.phase(), RdgFramePhase::Complete);
        assert_eq!(diagnostics.target(), Some(RdgRenderTargetKind::Headless));
        assert!(diagnostics.declared_passes() >= live);
        assert_eq!(diagnostics.prepared_passes(), live);
        assert_eq!(diagnostics.executed_passes(), live);
        assert!(diagnostics.queue_submitted());
        assert!(!diagnostics.presented());
        assert!(!diagnostics.panicked());
        assert_local_light_buffers_materialized(
            renderer.context.as_ref().expect("renderer state missing"),
        );
        assert_global_state_materialized_exact(
            renderer.context.as_ref().expect("renderer state missing"),
            scene.id(),
        );
    }

    #[test]
    fn dropped_composer_publishes_an_abandoned_pre_graph_attempt() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();

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
    }

    #[test]
    fn missing_headless_target_publishes_no_target_without_building_a_graph() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        renderer
            .context
            .as_mut()
            .expect("renderer state missing")
            .wgpu_ctx
            .headless_texture = None;

        renderer
            .begin_frame(
                &mut scene,
                make_camera(),
                &assets,
                FrameTime {
                    frame_count: 29,
                    ..FrameTime::default()
                },
            )
            .expect("frame composer missing")
            .render();

        let diagnostics = renderer
            .latest_rdg_diagnostics()
            .expect("no-target diagnostics missing");
        assert_eq!(diagnostics.renderer_frame(), Some(29));
        assert_eq!(diagnostics.outcome(), RdgAttemptOutcome::NoRenderTarget);
        assert_eq!(diagnostics.phase(), RdgFramePhase::AcquireTarget);
        assert_eq!(diagnostics.target(), None);
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
    }

    #[test]
    fn surface_acquisition_failure_does_not_materialize_global_state() {
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
        assert_eq!(diagnostics.declared_passes(), 0);
        assert_eq!(diagnostics.live_passes(), None);
        assert!(!diagnostics.queue_submitted());
    }

    #[test]
    fn missing_target_does_not_touch_warm_scene_light_buffers() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let before =
            local_light_gpu_state(renderer.context.as_ref().expect("renderer state missing"));
        let global_before = global_gpu_state(
            renderer.context.as_ref().expect("renderer state missing"),
            scene.id(),
        );
        renderer
            .context
            .as_mut()
            .expect("renderer state missing")
            .wgpu_ctx
            .headless_texture = None;

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

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(local_light_gpu_state(state), before);
        assert_eq!(global_gpu_state(state, scene.id()), global_before);
        let diagnostics = renderer
            .latest_rdg_diagnostics()
            .expect("no-target diagnostics missing");
        assert_eq!(diagnostics.outcome(), RdgAttemptOutcome::NoRenderTarget);
        assert!(!diagnostics.queue_submitted());
    }

    #[test]
    fn basic_forward_materializes_live_scene_light_imports() {
        let mut renderer = Renderer::new(
            RendererInitConfig::default(),
            RendererSettings {
                path: RenderPath::BasicForward,
                ..RendererSettings::default()
            },
        );
        pollster::block_on(renderer.init_headless(128, 128, None))
            .expect("headless renderer init failed");
        let assets = AssetServer::new();
        let mut scene = Scene::new();

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        assert_local_light_buffers_materialized(
            renderer.context.as_ref().expect("renderer state missing"),
        );
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
        pollster::block_on(renderer.init_headless(128, 128, None))
            .expect("headless renderer init failed");
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
    fn gaussian_extract_is_cpu_only_and_keeps_all_gpu_caches_cold() {
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
    }

    #[cfg(feature = "3dgs")]
    #[test]
    fn live_gaussian_prepare_reuses_static_and_cloud_caches_until_ttl_expires() {
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

        let state = renderer.context.as_mut().expect("renderer state missing");
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
        let second_last_used = state
            .gaussian_splatting_pass
            .cloud_last_used_frame(cloud)
            .expect("second live prepare must touch the cloud");
        assert!(second_last_used > first_last_used);

        const TEST_TTL: u64 = 3;
        assert_eq!(
            state
                .gaussian_splatting_pass
                .garbage_collect(second_last_used + TEST_TTL, TEST_TTL),
            0,
            "age == TTL remains live"
        );
        assert_eq!(state.gaussian_splatting_pass.cloud_cache_len(), 1);
        assert_eq!(
            state
                .gaussian_splatting_pass
                .garbage_collect(second_last_used + TEST_TTL + 1, TEST_TTL),
            1,
            "age == TTL + 1 retires the persistent upload"
        );
        assert_eq!(state.gaussian_splatting_pass.cloud_cache_len(), 0);
        assert!(
            state.gaussian_splatting_pass.static_cache_is_initialized(),
            "cloud TTL must not evict cross-cloud static pipelines"
        );
    }

    #[cfg(feature = "3dgs")]
    #[test]
    fn gaussian_same_handle_same_version_replacement_rebuilds_only_cloud_buffers() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let cloud = assets.gaussian_clouds.add(tiny_gaussian_cloud());
        let mut scene = Scene::new();
        scene.add_gaussian_cloud("exact-identity", cloud);
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let (first_static, first_cloud) = {
            let state = renderer.context.as_ref().expect("renderer state missing");
            (
                state.gaussian_splatting_pass.static_cache_identity(),
                state
                    .gaussian_splatting_pass
                    .cloud_cache_identity(cloud)
                    .expect("first live prepare must upload the cloud"),
            )
        };
        assert_eq!(assets.gaussian_clouds.get_version(cloud), Some(1));
        assert_eq!(first_cloud.0, 1);

        // `insert_ready` replaces the Arc stored behind an existing handle but
        // deliberately initializes its numeric version to 1 again. The weak
        // exact-asset identity must therefore participate in the L1 key.
        assets
            .gaussian_clouds
            .insert_ready(cloud, tiny_gaussian_cloud());
        assert_eq!(assets.gaussian_clouds.get_version(cloud), Some(1));

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let second_cloud = state
            .gaussian_splatting_pass
            .cloud_cache_identity(cloud)
            .expect("replacement asset must remain resident after prepare");
        assert_eq!(second_cloud.0, 1);
        assert_eq!(
            state.gaussian_splatting_pass.static_cache_identity(),
            first_static,
            "an asset-only replacement must preserve static layouts and pipelines"
        );
        assert_ne!(
            (second_cloud.1, second_cloud.2),
            (first_cloud.1, first_cloud.2),
            "same handle/version with a new Arc identity must rebuild both raw cloud buffers"
        );
    }

    #[cfg(feature = "3dgs")]
    #[test]
    fn basic_forward_with_gaussian_scene_does_not_initialize_gaussian_resources() {
        let mut renderer = Renderer::new(
            RendererInitConfig::default(),
            RendererSettings {
                path: RenderPath::BasicForward,
                ..RendererSettings::default()
            },
        );
        pollster::block_on(renderer.init_headless(128, 128, None))
            .expect("headless renderer init failed");
        let assets = AssetServer::new();
        let cloud = assets.gaussian_clouds.add(tiny_gaussian_cloud());
        let mut scene = Scene::new();
        scene.add_gaussian_cloud("basic-forward", cloud);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(state.gaussian_splatting_pass.cache_is_empty());
        assert!(!graph_contains_pass(state, "GS_Compute"));
        assert!(!graph_contains_pass(state, "GS_Render"));
        assert!(!graph_contains_pass(state, "GS_Composite"));
        assert!(state.graph_storage.gaussian_frame_batches.is_empty());
    }

    #[test]
    fn inactive_environment_uses_black_cube_without_allocating_scene_textures() {
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
    fn background_only_skybox_modes_do_not_allocate_ibl_textures() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let camera = make_camera();

        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.1, 0.2, 0.3, 1.0),
            Vec4::new(0.01, 0.02, 0.03, 1.0),
        ));
        render_frame(&mut renderer, &mut scene, camera, &assets, 0);
        {
            let state = renderer.context.as_ref().expect("renderer state missing");
            assert!(
                state.resource_manager.gpu_environment(scene.id()).is_none(),
                "a gradient background must not create IBL metadata"
            );
            assert!(!state.environment_cache.has_scene_state(scene.id()));
            assert!(published_pmrem_view_id(state, scene.id()).is_none());
        }

        let background_texture = assets.checkerboard(16, 2);
        scene
            .background
            .set_mode(BackgroundMode::planar(background_texture, 1.0));
        render_frame(&mut renderer, &mut scene, camera, &assets, 1);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(
            state.resource_manager.gpu_environment(scene.id()).is_none(),
            "a background-only texture must not create IBL metadata"
        );
        assert!(!state.environment_cache.has_scene_state(scene.id()));
        assert!(published_pmrem_view_id(state, scene.id()).is_none());
    }

    #[test]
    fn reenabled_environment_before_ttl_reuses_persistent_views() {
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
            .expect("initial scene environment metadata missing");
        assert_eq!(16, initial_env.base_cube_size);
        assert_eq!(8, initial_env.pmrem_size);
        assert!(initial_env.source_ready);
        let initial_source_version = initial_env.source_version;
        let initial_last_used_frame = initial_env.last_used_frame;
        let initial_pmrem_id = published_pmrem_view_id(state, scene_id)
            .expect("initial persistent PMREM must be published");
        assert!(state.environment_cache.has_scene_state(scene_id));

        scene
            .background
            .set_mode(BackgroundMode::color(0.0, 0.0, 0.0));
        render_frame(&mut renderer, &mut scene, camera, &assets, 1);
        scene.background.set_mode(BackgroundMode::procedural());
        render_frame(&mut renderer, &mut scene, camera, &assets, 2);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let reenabled_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("re-enabled scene environment metadata missing");
        assert_eq!(initial_source_version, reenabled_env.source_version);
        assert!(reenabled_env.last_used_frame > initial_last_used_frame);
        assert_eq!(
            Some(initial_pmrem_id),
            published_pmrem_view_id(state, scene_id),
            "re-enabling before TTL expiry should reuse the published PMREM view"
        );
    }

    #[test]
    fn environment_feature_maintenance_prunes_abandoned_scene_domains() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let camera = make_camera();

        let mut source_scene = Scene::new();
        source_scene.environment.set_base_cube_size(16);
        source_scene.environment.set_pmrem_size(8);
        source_scene
            .environment
            .set_env_map(Some(assets.checkerboard(16, 2)));
        render_frame(&mut renderer, &mut source_scene, camera, &assets, 0);

        let source_scene_id = source_scene.id();
        {
            let state = renderer.context.as_ref().expect("renderer state missing");
            assert!(state.environment_cache.has_scene_state(source_scene_id));
            assert!(published_pmrem_view_id(state, source_scene_id).is_some());
        }

        let mut procedural_scene = Scene::new();
        procedural_scene.environment.set_base_cube_size(16);
        procedural_scene.environment.set_pmrem_size(8);
        procedural_scene
            .background
            .set_mode(BackgroundMode::procedural());
        render_frame(&mut renderer, &mut procedural_scene, camera, &assets, 1);

        let procedural_scene_id = procedural_scene.id();
        {
            let state = renderer.context.as_ref().expect("renderer state missing");
            assert!(state.environment_cache.has_scene_state(procedural_scene_id));
            assert!(published_pmrem_view_id(state, procedural_scene_id).is_some());
            assert!(state.atmosphere_pass.has_scene_state(procedural_scene_id));
        }

        // Static converter/filter features no longer own scene state. The
        // logical coordinator retires both persistent domains atomically.
        let state = renderer.context.as_mut().expect("renderer state missing");
        let retired = state
            .environment_cache
            .maintain(&mut state.history_registry, u64::MAX);
        state
            .atmosphere_pass
            .maintain(&mut state.history_registry, u64::MAX);
        assert!(retired.contains(&source_scene_id));
        assert!(retired.contains(&procedural_scene_id));
        assert!(!state.environment_cache.has_scene_state(source_scene_id));
        assert!(!state.environment_cache.has_scene_state(procedural_scene_id));
        assert!(published_pmrem_view_id(state, source_scene_id).is_none());
        assert!(published_pmrem_view_id(state, procedural_scene_id).is_none());
        assert!(!state.atmosphere_pass.has_scene_state(procedural_scene_id));
    }

    #[test]
    fn missing_environment_attachment_stays_pending_without_graph_panic() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.environment.set_base_cube_size(16);
        scene.environment.set_pmrem_size(8);
        scene
            .environment
            .set_env_map(Some(TextureSource::Attachment(
                u64::MAX,
                wgpu::TextureViewDimension::D2,
            )));
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let gpu_env = state
            .resource_manager
            .gpu_environment(scene.id())
            .expect("configured environment should retain CPU metadata");
        assert_eq!(16, gpu_env.base_cube_size);
        assert_eq!(8, gpu_env.pmrem_size);
        assert!(!gpu_env.source_ready);
        assert_ne!(0, gpu_env.source_version);
        assert_eq!(
            state.resource_manager.frame_index(),
            gpu_env.last_used_frame
        );
        assert!(state.environment_cache.has_scene_state(scene.id()));
        assert!(published_pmrem_view_id(state, scene.id()).is_none());

        let global_state = state
            .resource_manager
            .get_global_state(state.render_frame.render_state.id, scene.id())
            .expect("global state missing for pending environment");
        assert_eq!(
            state.resource_manager.system_textures.black_cube.id(),
            global_state.resource_ids.as_slice()[3],
            "a first-time pending source must bind the black cube"
        );
    }

    #[test]
    fn disabling_environment_stops_keepalive_and_allows_ttl_pruning() {
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
    }

    #[test]
    fn reenabled_environment_recreates_using_current_configuration() {
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
            .expect("initial scene environment metadata missing");
        let initial_last_used_frame = initial_env.last_used_frame;
        let initial_pmrem_id = published_pmrem_view_id(state, scene_id)
            .expect("initial persistent PMREM must be published");

        scene
            .background
            .set_mode(BackgroundMode::color(0.0, 0.0, 0.0));
        render_frame(&mut renderer, &mut scene, camera, &assets, 1);
        {
            let state = renderer.context.as_mut().expect("renderer state missing");
            let retired = state
                .environment_cache
                .maintain(&mut state.history_registry, initial_last_used_frame + 121);
            assert_eq!(vec![scene_id], retired);
            state.resource_manager.remove_scene_global_states(scene_id);
            assert!(!state.environment_cache.has_scene_state(scene_id));
        }

        scene.environment.set_base_cube_size(32);
        scene.environment.set_pmrem_size(16);
        scene.background.set_mode(BackgroundMode::procedural());
        render_frame(&mut renderer, &mut scene, camera, &assets, 2);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let recreated_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("re-enabled scene environment metadata missing");
        assert_eq!(32, recreated_env.base_cube_size);
        assert_eq!(16, recreated_env.pmrem_size);
        assert!(recreated_env.source_ready);
        assert!(recreated_env.last_used_frame > initial_last_used_frame);
        assert_ne!(
            Some(initial_pmrem_id),
            published_pmrem_view_id(state, scene_id),
            "a retired domain must materialize a new PMREM using the current configuration"
        );
    }

    #[test]
    fn active_environment_source_switch_reuses_persistent_views() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.environment.set_base_cube_size(16);
        scene.environment.set_pmrem_size(8);
        let first_source = assets.checkerboard(16, 2);
        scene.environment.set_env_map(Some(first_source));
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let scene_id = scene.id();
        let state = renderer.context.as_ref().expect("renderer state missing");
        let initial_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("initial scene environment metadata missing");
        let initial_pmrem_id = published_pmrem_view_id(state, scene_id)
            .expect("initial persistent PMREM must be published");
        let initial_source_version = initial_env.source_version;
        let initial_last_used_frame = initial_env.last_used_frame;

        let second_source = assets.checkerboard(32, 4);
        scene.environment.set_env_map(Some(second_source));
        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let switched_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("switched scene environment metadata missing");
        assert_eq!(16, switched_env.base_cube_size);
        assert_eq!(8, switched_env.pmrem_size);
        assert!(switched_env.source_ready);
        assert_ne!(initial_source_version, switched_env.source_version);
        assert!(switched_env.last_used_frame > initial_last_used_frame);
        assert_eq!(
            Some(initial_pmrem_id),
            published_pmrem_view_id(state, scene_id),
            "a same-spec source switch should reuse the physical PMREM view"
        );
    }

    #[test]
    fn dirty_internal_environment_attachment_republishes_without_reallocating_pmrem() {
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
    }

    #[test]
    fn procedural_environment_updates_keep_global_bind_group_stable() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.environment.set_base_cube_size(16);
        scene.environment.set_pmrem_size(8);
        scene.background.set_mode(BackgroundMode::procedural());
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let scene_id = scene.id();
        let render_state_id = state.render_frame.render_state.id;
        let global_state = state
            .resource_manager
            .get_global_state(render_state_id, scene_id)
            .expect("global state missing after first render");
        let first_bind_group_id = global_state.bind_group_id;
        let first_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene environment metadata missing after first render");
        let first_source_version = first_env.source_version;
        let first_last_used_frame = first_env.last_used_frame;
        let first_pmrem_id = published_pmrem_view_id(state, scene_id)
            .expect("first persistent PMREM must be published");

        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            params.set_sun_intensity(25.0);
        } else {
            panic!("expected procedural background");
        }

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let global_state = state
            .resource_manager
            .get_global_state(render_state_id, scene_id)
            .expect("global state missing after procedural update");
        let updated_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene environment metadata missing after procedural update");

        assert_eq!(
            first_bind_group_id, global_state.bind_group_id,
            "procedural parameter changes should not rebuild the global bind group"
        );
        assert_eq!(16, updated_env.base_cube_size);
        assert_eq!(8, updated_env.pmrem_size);
        assert!(updated_env.source_ready);
        assert_ne!(first_source_version, updated_env.source_version);
        assert!(updated_env.last_used_frame > first_last_used_frame);
        assert_eq!(
            Some(first_pmrem_id),
            published_pmrem_view_id(state, scene_id),
            "procedural parameter changes should keep the persistent PMREM view stable"
        );
    }

    #[test]
    fn resizing_environment_maps_recreates_persistent_views() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.environment.set_base_cube_size(16);
        scene.environment.set_pmrem_size(8);
        scene.background.set_mode(BackgroundMode::procedural());
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let scene_id = scene.id();
        let render_state_id = state.render_frame.render_state.id;
        let global_state = state
            .resource_manager
            .get_global_state(render_state_id, scene_id)
            .expect("global state missing after first render");
        let first_bind_group_id = global_state.bind_group_id;
        let first_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene environment metadata missing after first render");
        let first_source_version = first_env.source_version;
        let first_last_used_frame = first_env.last_used_frame;
        let first_pmrem_id = published_pmrem_view_id(state, scene_id)
            .expect("first persistent PMREM must be published");

        scene.environment.set_base_cube_size(32);
        scene.environment.set_pmrem_size(16);

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let global_state = state
            .resource_manager
            .get_global_state(render_state_id, scene_id)
            .expect("global state missing after resize");
        let resized_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene environment metadata missing after resize");

        assert_ne!(
            first_bind_group_id, global_state.bind_group_id,
            "environment map resizing should rebuild the global bind group"
        );
        assert_ne!(
            Some(first_pmrem_id),
            published_pmrem_view_id(state, scene_id),
            "PMREM resize should recreate the persistent PMREM view"
        );
        assert_eq!(32, resized_env.base_cube_size);
        assert_eq!(16, resized_env.pmrem_size);
        assert!(resized_env.source_ready);
        assert_eq!(first_source_version, resized_env.source_version);
        assert!(resized_env.last_used_frame > first_last_used_frame);
    }

    #[test]
    fn small_day_night_step_does_not_rebake_procedural_environment() {
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
    }

    #[test]
    fn procedural_starbox_triggers_environment_rebake() {
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

        let starbox = assets.checkerboard(32, 8);
        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            params.set_starbox_texture(Some(starbox.into()));
            params.set_star_intensity(2.0);
        }

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let updated_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene environment metadata missing after starbox update");

        assert_ne!(
            initial_source_version, updated_env.source_version,
            "adding a procedural starbox should trigger an environment rebake"
        );
        assert!(updated_env.last_used_frame > initial_last_used_frame);
        assert_eq!(
            Some(initial_pmrem_id),
            published_pmrem_view_id(state, scene_id)
        );
        assert!(graph_contains_pass(state, "Atmosphere_SkyToCube"));
        assert!(graph_contains_pass(state, "IBL_Compute"));
    }

    #[test]
    fn tone_mapping_materializes_ready_3d_lut_only_in_live_prepare() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let image = Image::new(
            2,
            2,
            2,
            ImageDimension::D3,
            PixelFormat::Rgba8Unorm,
            Some(vec![128; 2 * 2 * 2 * 4]),
        );
        let image_handle = assets.images.add(image);
        let mut texture = Texture::new_3d(Some("ToneMap Test LUT"), image_handle);
        texture.color_space = ColorSpace::Linear;
        let lut_handle = assets.textures.add(texture);
        let mut scene = Scene::new();
        scene.tone_mapping.set_lut_texture(Some(lut_handle));
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(graph_contains_pass(state, "ToneMap_Pass"));
        assert!(state.tone_map_pass.has_lut_pipeline());
        assert_eq!(
            state
                .resource_manager
                .get_image(image_handle)
                .expect("live ToneMap must materialize the LUT image")
                .default_view_dimension,
            wgpu::TextureViewDimension::D3,
        );
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
    fn live_mixed_shadows_prepare_one_shared_vp_upload() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let caster = scene.spawn_box(
            1.0,
            1.0,
            1.0,
            PhysicalMaterial::new(Vec4::new(0.7, 0.7, 0.7, 1.0)),
            &assets,
        );
        scene.node(&caster).set_shadows(true, true);

        let mut directional = Light::new_directional(Vec3::ONE, 1.0);
        directional.cast_shadows = true;
        if let Some(shadow) = directional.shadow.as_mut() {
            shadow.map_size = 32;
        }
        let directional_node = scene.add_light(directional);
        scene
            .node(&directional_node)
            .set_position(3.0, 4.0, 2.0)
            .look_at(Vec3::ZERO);

        let mut point = Light::new_point(Vec3::ONE, 1.0, 8.0);
        point.cast_shadows = true;
        if let Some(shadow) = point.shadow.as_mut() {
            shadow.map_size = 32;
        }
        let point_node = scene.add_light(point);
        scene.node(&point_node).set_position(2.0, 2.0, 2.0);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(graph_contains_pass(state, "Shadow_VP_Upload"));
        assert!(graph_contains_pass(state, "Shadow_2D_Pass"));
        assert!(graph_contains_pass(state, "Shadow_Cube_Pass"));
        assert_eq!(
            state
                .graph_storage
                .passes
                .iter()
                .filter(|pass| pass.name == "Shadow_VP_Upload")
                .count(),
            1,
        );
        assert_eq!(
            state.resource_manager.model_allocator.len(),
            1,
            "main, Prepass, directional cascades, and point-light faces must share one model slot for the caster source"
        );
    }

    #[test]
    fn mipmap_generator_stays_uninitialized_without_requests() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(!state.mipmap_pass.generator_initialized());
        assert!(!graph_contains_pass(state, "Asset_Mipmap_Generation"));
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
    fn pending_material_texture_enters_logical_chain_only_after_becoming_ready() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let image_handle = assets.images.reserve();
        let mut texture = Texture::new_2d(Some("Pending Material Texture"), image_handle);
        texture.generate_mipmaps = true;
        let texture_handle = assets.textures.add(texture);
        let material = PhysicalMaterial::new(Vec4::ONE);
        material.set_map(Some(texture_handle));
        let mut scene = Scene::new();
        scene.spawn_box(1.0, 1.0, 1.0, material, &assets);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(!execution_contains_pass(state, "Asset_Texture_Upload"));
        assert!(!graph_contains_pass(state, "Asset_Texture_Upload"));
        assert!(!graph_contains_pass(
            state,
            "Asset_Texture_Mipmap_Generation"
        ));
        assert!(
            state
                .resource_manager
                .get_texture_image(texture_handle)
                .is_none()
        );

        assets.images.insert_ready(
            image_handle,
            Image::new(
                8,
                8,
                1,
                ImageDimension::D2,
                PixelFormat::Rgba8Unorm,
                Some(vec![73; 8 * 8 * 4]),
            ),
        );
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);
        let state = renderer.context.as_ref().expect("renderer state missing");
        let execution_names: Vec<_> = state
            .graph_storage
            .execution_queue
            .iter()
            .map(|&index| state.graph_storage.passes[index].name)
            .collect();
        let mipmap = execution_names
            .iter()
            .position(|&name| name == "Asset_Texture_Mipmap_Generation")
            .expect("newly ready material mipmap producer missing");
        let opaque = execution_names
            .iter()
            .position(|&name| name == "Opaque_Pass")
            .expect("opaque consumer missing");
        assert!(mipmap < opaque);
        assert!(
            state
                .resource_manager
                .get_texture_image(texture_handle)
                .expect("newly ready texture missing")
                .mipmaps_submitted()
        );

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 2);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(!execution_contains_pass(state, "Asset_Texture_Upload"));
        assert!(!execution_contains_pass(
            state,
            "Asset_Texture_Mipmap_Generation"
        ));
    }

    #[test]
    fn same_spec_image_refresh_reopens_logical_mipmap_generation() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let image_handle = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![21; 8 * 8 * 4]),
        ));
        let mut texture = Texture::new_2d(Some("Dynamic Mip Material"), image_handle);
        texture.generate_mipmaps = true;
        let texture_handle = assets.textures.add(texture);
        let material = PhysicalMaterial::new(Vec4::ONE);
        material.set_map(Some(texture_handle));
        let mut scene = Scene::new();
        scene.spawn_box(1.0, 1.0, 1.0, material, &assets);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);
        let initial_physical_id = renderer
            .context
            .as_ref()
            .and_then(|state| state.resource_manager.get_texture_image(texture_handle))
            .expect("initial material texture missing")
            .id;

        assert_eq!(
            assets.images.update(
                image_handle,
                Image::new(
                    8,
                    8,
                    1,
                    ImageDimension::D2,
                    PixelFormat::Rgba8Unorm,
                    Some(vec![219; 8 * 8 * 4]),
                ),
            ),
            Some(2)
        );
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(execution_contains_pass(
            state,
            "Asset_Texture_Mipmap_Generation"
        ));
        let refreshed = state
            .resource_manager
            .get_texture_image(texture_handle)
            .expect("refreshed material texture missing");
        assert_eq!(refreshed.id, initial_physical_id);
        assert!(refreshed.mipmaps_submitted());

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 2);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(!execution_contains_pass(
            state,
            "Asset_Texture_Mipmap_Generation"
        ));
    }

    #[test]
    fn abandoned_composer_does_not_materialize_texture_or_publish_mipmaps() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let image = Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![192; 8 * 8 * 4]),
        );
        let image_handle = assets.images.add(image);
        let mut texture = Texture::new_2d(Some("Mipmap Submission Test"), image_handle);
        texture.color_space = ColorSpace::Linear;
        texture.generate_mipmaps = true;
        let texture_handle = assets.textures.add(texture);
        let mut scene = Scene::new();
        scene
            .background
            .set_mode(BackgroundMode::planar(texture_handle, 1.0));
        let camera = make_camera();

        let composer = renderer
            .begin_frame(
                &mut scene,
                camera,
                &assets,
                FrameTime {
                    time: 0.0,
                    delta_time: 1.0 / 60.0,
                    frame_count: 0,
                },
            )
            .expect("frame composer must exist");
        drop(composer);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(
            state.resource_manager.get_image(image_handle).is_none(),
            "dropping Composer before graph build must not materialize the texture"
        );
        assert!(!state.mipmap_pass.generator_initialized());

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let gpu_image = state
            .resource_manager
            .get_image(image_handle)
            .expect("submitted texture must remain resident");
        assert_eq!(gpu_image.mip_level_count, 4);
        assert!(gpu_image.mipmaps_submitted());
        assert!(state.mipmap_pass.generator_initialized());
        assert!(!graph_contains_pass(state, "Asset_Texture_Upload"));
        assert!(graph_contains_pass(
            state,
            "Asset_Texture_Mipmap_Generation"
        ));
        let execution_names: Vec<_> = state
            .graph_storage
            .execution_queue
            .iter()
            .map(|&index| state.graph_storage.passes[index].name)
            .collect();
        let mipmap_position = execution_names
            .iter()
            .position(|&name| name == "Asset_Texture_Mipmap_Generation")
            .expect("mipmap pass missing from execution queue");
        let skybox_position = execution_names
            .iter()
            .position(|&name| name == "Skybox_Pass")
            .expect("texture skybox pass missing from execution queue");
        assert!(mipmap_position < skybox_position);
        assert!(
            state
                .graph_storage
                .mipmap_batches
                .iter()
                .all(Option::is_none),
            "submitted mipmap batches must release their Texture/Arc owners immediately"
        );
        assert!(
            state.graph_storage.asset_texture_imports.is_empty(),
            "submitted asset imports must release their view/request owners immediately"
        );
    }

    #[test]
    fn prepass_pipelines_are_created_only_by_live_prepare_and_reused() {
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
            let mesh = scene
                .meshes
                .get(mesh_node)
                .expect("spawned test mesh missing");
            (mesh.geometry, mesh.material)
        };
        let camera = make_camera();
        let initial_pipeline_count = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .pipeline_cache
            .render_pipeline_count();

        let composer = renderer
            .begin_frame(
                &mut scene,
                camera,
                &assets,
                FrameTime {
                    time: 0.0,
                    delta_time: 1.0 / 60.0,
                    frame_count: 0,
                },
            )
            .expect("frame composer must exist");
        drop(composer);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(
            state.prepass.cached_pipeline_count(),
            0,
            "begin_frame must not compile Prepass pipelines"
        );
        assert_eq!(
            state.pipeline_cache.render_pipeline_count(),
            initial_pipeline_count,
            "an abandoned Composer must not derive scene render pipelines"
        );
        assert!(state.render_lists.opaque.is_empty());
        assert!(state.render_lists.transparent.is_empty());
        assert!(state.render_lists.shadow_queues.is_empty());
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

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(graph_contains_pass(state, "Pre_Pass"));
        let live_pipeline_count = state.prepass.cached_pipeline_count();
        assert!(live_pipeline_count > 0);
        assert!(state.pipeline_cache.render_pipeline_count() > initial_pipeline_count);
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

        render_frame(&mut renderer, &mut scene, camera, &assets, 2);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(state.prepass.cached_pipeline_count(), live_pipeline_count);
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
        scene.node(&mesh_node).set_position(1_000_000.0, 0.0, 0.0);
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
    fn material_readiness_is_frozen_by_the_cpu_scene_snapshot() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let mesh_node = scene.spawn_box(1.0, 1.0, 1.0, PhysicalMaterial::new(Vec4::ONE), &assets);
        let pending_material = assets.materials.reserve();
        scene
            .meshes
            .get_mut(mesh_node)
            .expect("test mesh missing")
            .material = pending_material;

        let composer = renderer
            .begin_frame(&mut scene, make_camera(), &assets, FrameTime::default())
            .expect("frame composer must exist");
        assets.materials.insert_ready(
            pending_material,
            myth_resources::Material::from(PhysicalMaterial::new(Vec4::new(0.3, 0.4, 0.5, 1.0))),
        );
        composer.render();

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(state.render_frame.extracted_scene.render_items.is_empty());
        assert!(state.render_lists.opaque.is_empty());
        assert!(
            state
                .resource_manager
                .get_material(pending_material)
                .is_none()
        );

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(state.render_frame.extracted_scene.render_items.len(), 1);
        assert_eq!(state.render_lists.opaque.len(), 1);
        assert!(
            state
                .resource_manager
                .get_material(pending_material)
                .is_some()
        );
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
            "culling must retain the coherently extracted intent; GeometryFrame capture is the fail-closed asset-identity boundary"
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
                .is_some(),
            "the independently current material may be prepared and retained even though the stale geometry closes the draw source"
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
    fn blend_material_snapshot_is_not_shadow_eligible() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.spawn_box(
            1.0,
            1.0,
            1.0,
            PhysicalMaterial::new(Vec4::ONE)
                .with_alpha_mode(myth_resources::material::AlphaMode::Blend),
            &assets,
        );

        let composer = renderer
            .begin_frame(&mut scene, make_camera(), &assets, FrameTime::default())
            .expect("frame composer must exist");
        drop(composer);

        let item = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .render_frame
            .extracted_scene
            .render_items
            .first()
            .expect("render item missing");
        assert!(!item.cast_shadows);
    }

    #[test]
    fn material_classification_epoch_drift_fails_the_frame_closed() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let mesh = scene.spawn_box(1.0, 1.0, 1.0, PhysicalMaterial::new(Vec4::ONE), &assets);
        let material_handle = scene.meshes.get(mesh).expect("test mesh missing").material;

        let composer = renderer
            .begin_frame(&mut scene, make_camera(), &assets, FrameTime::default())
            .expect("frame composer must exist");
        let material = assets
            .materials
            .get(material_handle)
            .expect("test material missing");
        let myth_resources::material::MaterialType::Physical(physical) = &material.data else {
            panic!("test material must be physical");
        };
        physical.set_alpha_mode(myth_resources::material::AlphaMode::Blend);
        composer.render();

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert_eq!(
            state.render_lists.opaque.len(),
            1,
            "culling must consume the coherent extracted classification snapshot"
        );
        assert!(state.render_lists.transparent.is_empty());
        assert_eq!(
            state.resource_manager.model_allocator.len(),
            0,
            "MaterialPlan must reject the drifted material before physical draw preparation"
        );

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);
        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(state.render_lists.opaque.is_empty());
        assert_eq!(state.render_lists.transparent.len(), 1);
    }

    #[test]
    fn abandoned_composer_does_not_commit_morph_history() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let mesh_node = scene.spawn_box(1.0, 1.0, 1.0, PhysicalMaterial::new(Vec4::ONE), &assets);
        {
            let mesh = scene.meshes.get_mut(mesh_node).expect("test mesh missing");
            mesh.init_morph_targets(1, 1);
            mesh.set_morph_target_influence(0, 1.0);
        }

        let composer = renderer
            .begin_frame(&mut scene, make_camera(), &assets, FrameTime::default())
            .expect("frame composer must exist");
        drop(composer);

        let unsubmitted = scene
            .meshes
            .get(mesh_node)
            .expect("test mesh missing")
            .morph_uniforms();
        assert!(unsubmitted.weights[0].x.abs() <= f32::EPSILON);
        drop(unsubmitted);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 1);

        let submitted = scene
            .meshes
            .get(mesh_node)
            .expect("test mesh missing")
            .morph_uniforms();
        assert!((submitted.weights[0].x - 1.0).abs() <= f32::EPSILON);
        assert!(submitted.prev_weights[0].x.abs() <= f32::EPSILON);
        drop(submitted);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 2);
        let followup = scene
            .meshes
            .get(mesh_node)
            .expect("test mesh missing")
            .morph_uniforms();
        assert!((followup.prev_weights[0].x - 1.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn compact_frame_source_history_maps_back_to_the_nonzero_extracted_item() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.spawn_box(1.0, 1.0, 1.0, PhysicalMaterial::new(Vec4::ONE), &assets);
        scene.spawn_box(1.0, 1.0, 1.0, PhysicalMaterial::new(Vec4::ONE), &assets);
        let mut extraction_order = scene.meshes.iter().map(|(node, _)| node);
        let culled_node = extraction_order.next().expect("first test mesh missing");
        let visible_node = extraction_order.next().expect("second test mesh missing");
        assert!(extraction_order.next().is_none());
        scene.node(&culled_node).set_position(1_000_000.0, 0.0, 0.0);
        for (node, influence) in [(culled_node, 0.25), (visible_node, 1.0)] {
            let mesh = scene.meshes.get_mut(node).expect("test mesh missing");
            mesh.init_morph_targets(1, 1);
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
    fn stale_skeleton_binding_does_not_select_a_skinned_draw_variant() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let mesh_node = scene.spawn_box(
            1.0,
            1.0,
            1.0,
            PhysicalMaterial::new(Vec4::new(0.4, 0.4, 0.4, 1.0)),
            &assets,
        );
        let stale_skeleton = SkeletonKey::from(slotmap::KeyData::from_ffi(0x0000_0001_0000_0001));
        scene.bind_skeleton(mesh_node, stale_skeleton, BindMode::Attached);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let item = state
            .render_frame
            .extracted_scene
            .render_items
            .first()
            .expect("render item missing");
        assert!(item.skeleton.is_none());
        assert_eq!(item.item_variant_flags & (1 << 1), 0);
        assert!(!item.item_shader_defines.contains("HAS_SKINNING"));
        assert_eq!(state.render_lists.opaque.len(), 1);
    }

    #[test]
    fn model_buffer_resize_rebinds_objects_before_commands_capture_them() {
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
    }

    #[test]
    fn pruned_model_buffer_is_recreated_before_the_next_draw_prepare() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.spawn_box(1.0, 1.0, 1.0, PhysicalMaterial::new(Vec4::ONE), &assets);
        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_mut().expect("renderer state missing");
        let old_object_bind_group_id = *state
            .resource_manager
            .cached_object_bind_group_ids()
            .first()
            .expect("submitted draw did not populate the object cache");
        let logical_id = state.resource_manager.model_buffer_id();
        let initial_physical_id = state
            .resource_manager
            .get_gpu_buffer_by_cpu_id(logical_id)
            .expect("initial model buffer missing")
            .id;

        state.resource_manager.next_frame();
        state.resource_manager.prune(0);
        assert!(
            state
                .resource_manager
                .get_gpu_buffer_by_cpu_id(logical_id)
                .is_none(),
            "the test must exercise the stale allocator-handle path"
        );
        assert!(
            state
                .resource_manager
                .get_cached_bind_group(old_object_bind_group_id)
                .is_none(),
            "object cache TTL must release contexts owning stale physical buffers"
        );

        state.resource_manager.flush_model_buffers();
        let recreated = state
            .resource_manager
            .get_gpu_buffer_by_cpu_id(logical_id)
            .expect("model buffer was not recreated");
        assert_ne!(recreated.id, initial_physical_id);
    }

    #[test]
    fn basic_forward_texture_skybox_consumes_the_live_mip_ready_import() {
        let mut renderer = Renderer::new(
            RendererInitConfig::default(),
            RendererSettings {
                path: RenderPath::BasicForward,
                ..RendererSettings::default()
            },
        );
        pollster::block_on(renderer.init_headless(128, 128, None))
            .expect("headless renderer init failed");
        let assets = AssetServer::new();
        let image_handle = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![96; 8 * 8 * 4]),
        ));
        let mut texture = Texture::new_2d(Some("BasicForward Skybox"), image_handle);
        texture.color_space = ColorSpace::Linear;
        texture.generate_mipmaps = true;
        let texture_handle = assets.textures.add(texture);
        let mut scene = Scene::new();
        scene
            .background
            .set_mode(BackgroundMode::planar(texture_handle, 1.0));

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let image = state
            .resource_manager
            .get_texture_image(texture_handle)
            .expect("live BasicForward skybox must materialize its texture");
        assert_eq!(image.mip_level_count, 4);
        assert!(image.mipmaps_submitted());
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
            .expect("asset mipmap pass missing");
        let forward = execution_names
            .iter()
            .position(|&name| name == "SimpleForward_Pass")
            .expect("BasicForward pass missing");
        assert!(mipmap < forward);
        assert!(state.graph_storage.asset_texture_imports.is_empty());
    }

    #[test]
    fn basic_forward_draw_plans_retain_opaque_and_transparent_material_textures() {
        let mut renderer = Renderer::new(
            RendererInitConfig::default(),
            RendererSettings {
                path: RenderPath::BasicForward,
                ..RendererSettings::default()
            },
        );
        pollster::block_on(renderer.init_headless(128, 128, None))
            .expect("headless renderer init failed");
        let assets = AssetServer::new();
        let opaque_image = assets.images.add(Image::new(
            4,
            4,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![64; 4 * 4 * 4]),
        ));
        let transparent_image = assets.images.add(Image::new(
            4,
            4,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![192; 4 * 4 * 4]),
        ));
        let opaque_texture = assets
            .textures
            .add(Texture::new_2d(Some("BasicForward Opaque"), opaque_image));
        let transparent_texture = assets.textures.add(Texture::new_2d(
            Some("BasicForward Transparent"),
            transparent_image,
        ));
        let opaque = PhysicalMaterial::new(Vec4::ONE);
        opaque.set_map(Some(opaque_texture));
        let transparent = PhysicalMaterial::new(Vec4::ONE)
            .with_alpha_mode(myth_resources::material::AlphaMode::Blend);
        transparent.set_map(Some(transparent_texture));
        let mut scene = Scene::new();
        scene.spawn_box(1.0, 1.0, 1.0, opaque, &assets);
        scene.spawn_box(0.5, 0.5, 0.5, transparent, &assets);

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let execution_names: Vec<_> = state
            .graph_storage
            .execution_queue
            .iter()
            .map(|&index| state.graph_storage.passes[index].name)
            .collect();
        assert!(execution_names.contains(&"SimpleForward_Pass"));
        assert!(!execution_names.contains(&"Asset_Texture_Upload"));
        assert!(!execution_names.contains(&"Asset_Texture_Mipmap_Generation"));
        assert!(
            state
                .resource_manager
                .get_texture_image(opaque_texture)
                .is_some()
        );
        assert!(
            state
                .resource_manager
                .get_texture_image(transparent_texture)
                .is_some()
        );
        assert_eq!(state.render_lists.opaque.len(), 1);
        assert_eq!(state.render_lists.transparent.len(), 1);
    }

    #[test]
    fn material_and_texture_skybox_share_one_logical_asset_chain() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let image_handle = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![160; 8 * 8 * 4]),
        ));
        let mut material_texture = Texture::new_2d(Some("Shared Material"), image_handle);
        material_texture.color_space = ColorSpace::Linear;
        material_texture.generate_mipmaps = true;
        let material_texture_handle = assets.textures.add(material_texture);
        let material = PhysicalMaterial::new(Vec4::ONE);
        material.set_map(Some(material_texture_handle));
        let mut scene = Scene::new();
        scene.spawn_box(1.0, 1.0, 1.0, material, &assets);
        scene
            .background
            .set_mode(BackgroundMode::planar(material_texture_handle, 1.0));

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(
            !graph_contains_pass(state, "Asset_Mipmap_Generation"),
            "the shared logical import must not create a legacy mip request"
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
        assert_eq!(
            state
                .graph_storage
                .passes
                .iter()
                .filter(|pass| pass.name == "Asset_Texture_Upload")
                .count(),
            0
        );
        let image = state
            .resource_manager
            .get_texture_image(material_texture_handle)
            .expect("shared material/Skybox texture must remain resident");
        assert!(image.mipmaps_submitted());

        let execution_names: Vec<_> = state
            .graph_storage
            .execution_queue
            .iter()
            .map(|&index| state.graph_storage.passes[index].name)
            .collect();
        let mipmap = execution_names
            .iter()
            .position(|&name| name == "Asset_Texture_Mipmap_Generation")
            .expect("shared logical mipmap producer missing");
        let opaque = execution_names
            .iter()
            .position(|&name| name == "Opaque_Pass")
            .expect("opaque consumer missing");
        let skybox = execution_names
            .iter()
            .position(|&name| name == "Skybox_Pass")
            .expect("Skybox consumer missing");
        assert!(mipmap < opaque && mipmap < skybox);
    }

    #[test]
    fn distinct_texture_handles_share_one_physical_mipmap_generation() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let image_handle = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![160; 8 * 8 * 4]),
        ));
        let mut material_texture = Texture::new_2d(Some("Shared Material"), image_handle);
        material_texture.color_space = ColorSpace::Linear;
        material_texture.generate_mipmaps = true;
        let material_texture_handle = assets.textures.add(material_texture);
        let mut skybox_texture = Texture::new_2d(Some("Shared Skybox"), image_handle);
        skybox_texture.color_space = ColorSpace::Linear;
        skybox_texture.generate_mipmaps = true;
        skybox_texture.sampler.mag_filter = wgpu::FilterMode::Nearest;
        let skybox_texture_handle = assets.textures.add(skybox_texture);
        let material = PhysicalMaterial::new(Vec4::ONE);
        material.set_map(Some(material_texture_handle));
        let mut scene = Scene::new();
        scene.spawn_box(1.0, 1.0, 1.0, material, &assets);
        scene
            .background
            .set_mode(BackgroundMode::planar(skybox_texture_handle, 1.0));

        render_frame(&mut renderer, &mut scene, make_camera(), &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        assert!(!graph_contains_pass(state, "Asset_Mipmap_Generation"));
        assert_eq!(
            state
                .graph_storage
                .passes
                .iter()
                .filter(|pass| pass.name == "Asset_Texture_Upload")
                .count(),
            0,
            "distinct logical snapshots remain import resources, not proxy passes"
        );
        assert_eq!(
            state
                .graph_storage
                .passes
                .iter()
                .filter(|pass| pass.name == "Asset_Texture_Mipmap_Generation")
                .count(),
            2,
            "each logical import exposes readiness while physical encoding is coordinated"
        );
        let execution_names: Vec<_> = state
            .graph_storage
            .execution_queue
            .iter()
            .map(|&index| state.graph_storage.passes[index].name)
            .collect();
        assert_eq!(
            execution_names
                .iter()
                .filter(|&&name| name == "Asset_Texture_Upload")
                .count(),
            0
        );
        assert_eq!(
            execution_names
                .iter()
                .filter(|&&name| name == "Asset_Texture_Mipmap_Generation")
                .count(),
            2
        );
        let material_image = state
            .resource_manager
            .get_texture_image(material_texture_handle)
            .expect("material texture must remain resident");
        let skybox_image = state
            .resource_manager
            .get_texture_image(skybox_texture_handle)
            .expect("Skybox texture must remain resident");
        assert_eq!(material_image.id, skybox_image.id);
        assert!(material_image.mipmaps_submitted());
        assert!(skybox_image.mipmaps_submitted());
    }

    #[test]
    fn pending_procedural_optional_source_keeps_published_environment() {
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
        let initial_pmrem_id = published_pmrem_view_id(state, scene_id)
            .expect("initial persistent PMREM must be published");

        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            params.set_starbox_texture(Some(TextureSource::Attachment(
                u64::MAX,
                wgpu::TextureViewDimension::D2,
            )));
        }

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let pending_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("procedural environment metadata missing");

        assert!(!pending_env.source_ready);
        assert_eq!(
            Some(initial_pmrem_id),
            published_pmrem_view_id(state, scene_id),
            "a pending optional source must not overwrite the last published PMREM"
        );
        assert!(
            graph_contains_pass(state, "Atmosphere_SkyView"),
            "the visual atmosphere path must remain available while an optional source is pending"
        );
        assert!(!graph_contains_pass(state, "Atmosphere_SkyToCube"));
        assert!(!graph_contains_pass(state, "IBL_Compute"));
    }
}
