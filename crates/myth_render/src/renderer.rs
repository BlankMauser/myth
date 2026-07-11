//! Rendering System
//!
//! The main [`Renderer`] struct orchestrating GPU rendering operations.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use smallvec::SmallVec;

use crate::core::binding::GlobalBindGroupCache;
use crate::core::gpu::Tracked;
use crate::graph::composer::ComposerContext;
use crate::graph::core::HistoryRegistry;
use crate::graph::core::allocator::TransientPool;
use crate::graph::core::arena::FrameArena;
use crate::graph::core::graph::GraphStorage;
use crate::graph::core::persistent::TEMPORAL_DOMAIN_TTL_FRAMES;
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
        // Reset the arena in O(1) — all previous PassNodes are trivially
        // forgotten (no Drop needed).
        state.frame_arena.reset();

        // Advance the bind-group cache's frame counter for TTL tracking.
        state.global_bind_group_cache.begin_frame();

        // ── Phase 1: Extract scene, build shadow views, prepare global ──

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
        let published_pmrem_view = has_active_environment
            .then(|| {
                state
                    .environment_cache
                    .published_pmrem_view(scene.id(), &state.history_registry)
                    .cloned()
            })
            .flatten();
        let published_pmrem_max_mip = has_active_environment
            .then(|| {
                state
                    .environment_cache
                    .published_pmrem_mip_level_count(scene.id(), &state.history_registry)
                    .map(|mip_count| mip_count.saturating_sub(1) as f32)
            })
            .flatten();
        state.render_frame.extract_and_prepare(
            &mut state.resource_manager,
            scene,
            &camera,
            assets,
            frame_time,
            &mut state.render_lists,
            surface_size,
            max_extracted_lights,
            feature_config,
            published_pmrem_view.as_ref(),
            published_pmrem_max_mip,
        );

        // ── Phase 2: Cull + sort + command generation ───────────────────
        crate::graph::culling::cull_and_sort(
            &state.render_frame.extracted_scene,
            &state.render_frame.render_state,
            &state.wgpu_ctx,
            &mut state.resource_manager,
            &mut state.pipeline_cache,
            &mut state.shader_manager,
            &mut state.render_lists,
            &camera,
            assets,
        );

        let clustered_lighting_enabled = state
            .render_frame
            .extracted_scene
            .scene_variants
            .contains(SceneFeatures::USE_CLUSTERED_SHADING);

        // ── Phase 2.5: Remaining eager feature preparation ─────────────
        //
        // Features not yet migrated to deferred RDG preparation still resolve
        // their persistent resources here. Migrated features capture snapshots
        // in FrameComposer and materialize resources only in live PassNodes.
        {
            use crate::HDR_TEXTURE_FORMAT;
            use crate::graph::core::context::ExtractContext;

            let view_format = state.wgpu_ctx.surface_view_format;
            let is_hf = state.wgpu_ctx.render_path.supports_post_processing();
            let scene_id_val = scene.id();
            let render_state_id = state.render_frame.render_state.id;
            let global_state_key = (render_state_id, scene_id_val);

            let ssao_enabled = scene.ssao.enabled && is_hf;
            let ssgi_enabled = state
                .render_frame
                .extracted_scene
                .scene_variants
                .contains(SceneFeatures::USE_SSGI);
            let ssr_enabled = state
                .render_frame
                .extracted_scene
                .scene_variants
                .contains(SceneFeatures::USE_SSR);
            let needs_feature_id = is_hf && (scene.ssss.enabled || ssr_enabled);

            // Sync camera debug settings → RenderState before borrowing it.
            #[cfg(feature = "debug_view")]
            {
                let dv = camera.debug_view;
                state.render_frame.render_state.debug_view_mode = dv.mode;
                state.render_frame.render_state.debug_view_scale = dv.custom_scale;
            }

            #[cfg(feature = "debug_view")]
            let (dbg_needs_normal, dbg_needs_velocity) = {
                use crate::graph::render_state::DebugViewTarget;
                let target =
                    DebugViewTarget::from_mode(state.render_frame.render_state.debug_view_mode);
                (
                    target == DebugViewTarget::SceneNormal,
                    target == DebugViewTarget::Velocity,
                )
            };

            #[cfg(not(feature = "debug_view"))]
            let (dbg_needs_normal, dbg_needs_velocity) = (false, false);

            let needs_normal = ssao_enabled || ssgi_enabled || needs_feature_id || dbg_needs_normal;
            let needs_velocity =
                camera.aa_mode.is_taa() || ssgi_enabled || ssr_enabled || dbg_needs_velocity;

            // let needs_normal = ssao_enabled || needs_feature_id;
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

            let mut extract_ctx = ExtractContext {
                device: &state.wgpu_ctx.device,
                queue: &state.wgpu_ctx.queue,
                pipeline_cache: &mut state.pipeline_cache,
                shader_manager: &mut state.shader_manager,
                global_bind_group_cache: &mut state.global_bind_group_cache,
                resource_manager: &mut state.resource_manager,
                wgpu_ctx: &state.wgpu_ctx,
                render_lists: &mut state.render_lists,
                extracted_scene: &state.render_frame.extracted_scene,
                render_state: &state.render_frame.render_state,
                render_camera: &camera,
                assets,
            };
            // Always: shadow extraction. Environment source/PMREM extraction
            // now happens inside Composer after render-target acquisition.
            state.shadow_pass.extract_and_prepare(&mut extract_ctx);

            // Texture backgrounds still use the compatibility preparation path
            // for same-frame asset mip generation. Gradient and Procedural
            // Skybox preparation are deferred to their live RDG consumer.
            if matches!(&scene.background.mode, BackgroundMode::Texture { .. }) {
                let color_format = if is_hf {
                    HDR_TEXTURE_FORMAT
                } else {
                    view_format
                };
                state.skybox_pass.extract_and_prepare(
                    &mut extract_ctx,
                    &scene.background.mode,
                    &scene.background.uniforms,
                    global_state_key,
                    color_format,
                );
            } else {
                state.skybox_pass.deactivate();
            }

            // Gaussian Splatting
            #[cfg(feature = "3dgs")]
            {
                let mut cloud_entries = Vec::new();
                for (node_handle, cloud_handle) in &scene.gaussian_clouds {
                    let Some(node) = scene.get_node(node_handle) else {
                        continue;
                    };
                    if !node.visible {
                        continue;
                    }
                    if let Some(cloud) = assets.gaussian_clouds.get(*cloud_handle) {
                        let world_matrix = glam::Mat4::from(*node.world_matrix());
                        cloud_entries.push((*cloud_handle, cloud, world_matrix));
                    }
                }
                state
                    .gaussian_splatting_pass
                    .extract_and_prepare(&mut extract_ctx, &cloud_entries);
            }

            if is_hf {
                state.prepass.extract_and_prepare(
                    &mut extract_ctx,
                    needs_normal,
                    needs_feature_id,
                    needs_velocity,
                );

                state.tone_map_pass.extract_and_prepare(
                    &mut extract_ctx,
                    scene.tone_mapping.mode,
                    view_format,
                    global_state_key,
                    &scene.tone_mapping.uniforms,
                    scene.tone_mapping.lut_texture,
                );
            }

            state.mipmap_pass.extract_and_prepare(&mut extract_ctx);
        }

        // ── Phase 3: Build ComposerContext ──────────────────────────────
        let ctx = ComposerContext {
            wgpu_ctx: &mut state.wgpu_ctx,
            resource_manager: &mut state.resource_manager,
            pipeline_cache: &mut state.pipeline_cache,
            shader_manager: &mut state.shader_manager,

            extracted_scene: &state.render_frame.extracted_scene,
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
        Some(FrameComposer::new(ctx, self.size))
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

    pub fn dump_graph_mermaid(&self) -> Option<String> {
        self.context
            .as_ref()
            .map(|s| s.graph_storage.dump_mermaid())
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

    use glam::{Affine3A, Quat, Vec3, Vec4};
    use myth_assets::AssetServer;
    use myth_resources::texture::TextureSource;
    use myth_scene::Scene;
    use myth_scene::background::BackgroundMode;
    use myth_scene::camera::Camera;

    fn init_headless_renderer() -> Renderer {
        let mut renderer =
            Renderer::new(RendererInitConfig::default(), RendererSettings::default());
        pollster::block_on(renderer.init_headless(128, 128, None))
            .expect("headless renderer init failed");
        renderer
    }

    fn make_camera() -> RenderCamera {
        let mut camera = Camera::new_perspective(45.0, 1.0, 0.1);
        camera.update_view_projection(&Affine3A::from_translation(Vec3::new(0.0, 0.0, 5.0)));
        camera.extract_render_camera()
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
    fn global_state_and_render_graph_exclude_the_unused_brdf_lut() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        let camera = make_camera();

        {
            let state = renderer.context.as_ref().expect("renderer state missing");
            assert!(
                !state
                    .resource_manager
                    .internal_textures
                    .contains_name("BRDF_LUT"),
                "renderer initialization must not allocate the unused LUT"
            );
        }

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let global_state = state
            .resource_manager
            .get_global_state(state.render_frame.render_state.id, scene.id())
            .expect("global state missing after render");
        assert_eq!(
            global_state.resource_ids.as_slice().len(),
            4,
            "global fingerprint should contain only camera, environment, lights, and the bound PMREM identity"
        );
        assert_eq!(
            global_state.binding_wgsl.matches("@group(0)").count(),
            5,
            "global layout should contain three buffers plus PMREM texture/sampler"
        );
        assert!(
            !global_state.binding_wgsl.contains("brdf_lut"),
            "the generated global shader interface must not expose a dead LUT binding"
        );
        assert!(
            !state
                .resource_manager
                .internal_textures
                .contains_name("BRDF_LUT"),
            "rendering a frame must not allocate the unused LUT"
        );
        assert!(
            state
                .graph_storage
                .passes
                .iter()
                .all(|pass| pass.name != "BRDF_LUT"),
            "the graph must not schedule a dead BRDF compute pass"
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
        assert!(state.skybox_pass.current_bind_group.is_none());
        assert!(state.skybox_pass.current_pipeline.is_none());
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
