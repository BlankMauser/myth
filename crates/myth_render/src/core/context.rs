//! wgpu Context
//!
//! The [`WgpuContext`] holds core GPU handles: device, queue, and render target state.
//! It supports two modes of operation:
//!
//! - **Windowed mode**: A `wgpu::Surface` is created from a window handle for
//!   on-screen presentation. The surface owns the swap-chain back buffers.
//! - **Headless mode**: No surface is created. The engine maintains an offscreen
//!   `wgpu::Texture` as the render target, suitable for server-side rendering,
//!   GPU readback, or CI/CD pipelines.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::settings::{RenderPath, RendererInitConfig, RendererSettings};
use myth_core::{Error, PlatformError, Result};

/// GPU-driven primitive capability tier of the active logical device.
///
/// The tier reflects capabilities that are safe to use immediately. Optional
/// adapter features do not raise the tier unless they were explicitly enabled
/// when the device was requested. It does not claim that every Myth GPU Scene
/// workload fits the device limits; the GPU Scene preflight validates each
/// complete view-family ABI against [`GpuDrivenCapabilities::device_limits`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum GpuDrivenTier {
    /// Required compute/indirect/vertex-storage primitives are unavailable.
    Unavailable,
    /// Compute-generated indexed indirect drawing and vertex-stage scene-table
    /// reads are available.
    Indirect,
    /// The baseline is available and the device has enabled indirect-count
    /// multi-draw support.
    MultiDrawIndirectCount,
}

/// Immutable adapter/device capability snapshot used by the GPU-driven path.
///
/// Adapter features describe what could be enabled by recreating the device;
/// device features describe what may be used by the current device. Keeping
/// both sets prevents an adapter-only optional feature from leaking into live
/// backend selection. Optional tiers are never inferred from adapter support:
/// the corresponding feature must be enabled on the logical device.
#[derive(Debug, Clone)]
pub struct GpuDrivenCapabilities {
    adapter_features: wgpu::Features,
    device_features: wgpu::Features,
    downlevel: wgpu::DownlevelCapabilities,
    adapter_limits: wgpu::Limits,
    device_limits: wgpu::Limits,
    tier: GpuDrivenTier,
}

impl GpuDrivenCapabilities {
    const BASELINE_DOWNLEVEL_FLAGS: wgpu::DownlevelFlags = wgpu::DownlevelFlags::COMPUTE_SHADERS
        .union(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
        .union(wgpu::DownlevelFlags::VERTEX_STORAGE);

    fn capture(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Self {
        Self::from_parts(
            adapter.features(),
            device.features(),
            adapter.get_downlevel_capabilities(),
            adapter.limits(),
            device.limits(),
        )
    }

    pub(crate) fn from_parts(
        adapter_features: wgpu::Features,
        device_features: wgpu::Features,
        downlevel: wgpu::DownlevelCapabilities,
        adapter_limits: wgpu::Limits,
        device_limits: wgpu::Limits,
    ) -> Self {
        let has_baseline_flags = downlevel.flags.contains(Self::BASELINE_DOWNLEVEL_FLAGS);
        let has_baseline_limits = device_limits.max_storage_buffers_per_shader_stage > 0
            && device_limits.max_storage_buffer_binding_size > 0
            && device_limits.max_buffer_size > 0
            && device_limits.max_compute_workgroups_per_dimension > 0
            && device_limits.max_compute_invocations_per_workgroup > 0
            && device_limits.max_compute_workgroup_size_x > 0;

        let tier = if !has_baseline_flags || !has_baseline_limits {
            GpuDrivenTier::Unavailable
        } else if device_features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
            GpuDrivenTier::MultiDrawIndirectCount
        } else {
            GpuDrivenTier::Indirect
        };

        Self {
            adapter_features,
            device_features,
            downlevel,
            adapter_limits,
            device_limits,
            tier,
        }
    }

    /// Returns the GPU-driven primitive tier of the current logical device.
    #[inline]
    #[must_use]
    pub const fn tier(&self) -> GpuDrivenTier {
        self.tier
    }

    /// Returns all optional features advertised by the physical adapter.
    #[inline]
    #[must_use]
    pub const fn adapter_features(&self) -> wgpu::Features {
        self.adapter_features
    }

    /// Returns the optional features enabled on the current logical device.
    #[inline]
    #[must_use]
    pub const fn device_features(&self) -> wgpu::Features {
        self.device_features
    }

    /// Returns the adapter's downlevel capability snapshot.
    #[inline]
    #[must_use]
    pub const fn downlevel(&self) -> &wgpu::DownlevelCapabilities {
        &self.downlevel
    }

    /// Returns the limits advertised by the physical adapter.
    #[inline]
    #[must_use]
    pub const fn adapter_limits(&self) -> &wgpu::Limits {
        &self.adapter_limits
    }

    /// Returns the limits enabled on the current logical device.
    #[inline]
    #[must_use]
    pub const fn device_limits(&self) -> &wgpu::Limits {
        &self.device_limits
    }

    /// Returns whether the primitives needed to build a compute-generated
    /// indirect workload are available.
    #[inline]
    #[must_use]
    pub const fn supports_indirect_primitives(&self) -> bool {
        !matches!(self.tier, GpuDrivenTier::Unavailable)
    }

    /// Returns whether non-zero `first_instance` indirect arguments are
    /// enabled on the current logical device.
    #[inline]
    #[must_use]
    pub const fn supports_indirect_first_instance(&self) -> bool {
        self.supports_indirect_primitives()
            && self
                .device_features
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    }

    /// Returns whether indirect-count multi-draw is enabled on the current
    /// logical device.
    #[inline]
    #[must_use]
    pub const fn supports_multi_draw_indirect_count(&self) -> bool {
        matches!(self.tier, GpuDrivenTier::MultiDrawIndirectCount)
    }

    /// Returns whether storage buffers may also be used as index buffers.
    ///
    /// This is not required by the M7 baseline, but it determines whether a
    /// later geometry mega-buffer can combine `STORAGE` and `INDEX` usages.
    #[inline]
    #[must_use]
    pub const fn supports_storage_index_buffers(&self) -> bool {
        self.downlevel
            .flags
            .contains(wgpu::DownlevelFlags::UNRESTRICTED_INDEX_BUFFER)
    }
}

/// Core wgpu context holding GPU handles and render target state.
///
/// This struct owns the fundamental wgpu resources needed for rendering:
/// - `device`: GPU device for resource creation
/// - `queue`: Command submission queue
/// - `surface` (optional): Window surface for on-screen presentation
/// - `headless_texture` (optional): Offscreen texture for headless rendering
///
/// The render target dimensions and format are always available via
/// [`target_width`], [`target_height`], and [`surface_view_format`],
/// regardless of the active mode.
pub struct WgpuContext {
    /// The wgpu device for GPU operations.
    pub device: wgpu::Device,
    /// The command queue for submitting work.
    pub queue: wgpu::Queue,

    /// Window surface for on-screen presentation (`None` in headless mode).
    pub surface: Option<wgpu::Surface<'static>>,
    /// Surface configuration (`None` in headless mode).
    pub config: Option<wgpu::SurfaceConfiguration>,

    /// Offscreen render target for headless mode (`None` in windowed mode).
    ///
    /// Created with `RENDER_ATTACHMENT | COPY_SRC` usage to support both
    /// rendering and GPU-to-CPU readback.
    pub headless_texture: Option<wgpu::Texture>,

    /// Render target width in pixels.
    pub target_width: u32,
    /// Render target height in pixels.
    pub target_height: u32,

    /// Depth buffer format.
    pub depth_format: wgpu::TextureFormat,

    /// The view format used for render target views (always sRGB-qualified).
    pub surface_view_format: wgpu::TextureFormat,

    pub msaa_samples: u32,

    pub anisotropy_clamp: u16,

    /// The active render path. Stored for runtime branching in the frame graph.
    pub render_path: RenderPath,

    /// Version counter for pipeline-affecting settings (HDR, MSAA, RenderPath).
    /// Incremented when these settings change, used to invalidate L1 pipeline cache.
    pub pipeline_settings_version: u64,

    gpu_driven_capabilities: GpuDrivenCapabilities,
}

impl WgpuContext {
    pub async fn new<W>(
        window: W,
        init_config: &RendererInitConfig,
        settings: &RendererSettings,
        width: u32,
        height: u32,
    ) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let instance_desc = match init_config.backends {
            Some(backends) => wgpu::InstanceDescriptor {
                backends,
                ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
            },
            None => wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        };
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance
            .create_surface(window)
            .map_err(|e| Error::Platform(PlatformError::SurfaceConfigFailed(e.to_string())))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: init_config.power_preference,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| Error::Platform(PlatformError::AdapterNotFound(e.to_string())))?;

        let info = adapter.get_info();

        log::debug!("Backend: {:?}", info.backend);
        log::debug!("Device: {}", info.name);
        log::debug!("Vendor: {:x}", info.vendor);

        // === Query Surface-supported formats ===
        let caps = surface.get_capabilities(&adapter);

        // Print debug info showing formats supported on the current platform
        log::debug!("Surface Supported Formats: {:?}", caps.formats);

        // Prefer sRGB format (Native); if unavailable (Web), select the first available format (usually Linear)
        // Note: On the Web, an sRGB format will definitely not be found here, falling back to caps.formats[0]
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);

        log::debug!("Selected Surface Format: {surface_format:?}");

        // let required_limits = Self::requested_limits_for_adapter(init_config, &adapter);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: init_config.required_features,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|e| {
                Error::Render(myth_core::RenderError::RequestDeviceFailed(e.to_string()))
            })?;

        let gpu_driven_capabilities = GpuDrivenCapabilities::capture(&adapter, &device);

        let view_format = surface_format.add_srgb_suffix();

        let present_mode = if settings.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            desired_maximum_frame_latency: 2,
            present_mode,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![view_format],
        };

        surface.configure(&device, &config);

        Ok(Self {
            device,
            queue,
            surface: Some(surface),
            config: Some(config),
            headless_texture: None,
            target_width: width,
            target_height: height,
            depth_format: init_config.depth_format,
            surface_view_format: view_format,
            msaa_samples: 1,
            anisotropy_clamp: settings.anisotropy_clamp,
            render_path: settings.path,
            pipeline_settings_version: 0,
            gpu_driven_capabilities,
        })
    }

    /// Creates a headless (offscreen) GPU context without a window surface.
    ///
    /// This constructor skips surface creation and instead allocates an
    /// offscreen `wgpu::Texture` as the render target. The texture is created
    /// with `RENDER_ATTACHMENT | COPY_SRC` usage, enabling both rendering and
    /// GPU-to-CPU readback.
    ///
    /// # Arguments
    ///
    /// * `target_format` — Desired render target format. Pass `None` to use
    ///   the default `Rgba8UnormSrgb`, which provides broad compatibility and
    ///   straightforward CPU readback. Use `Rgba16Float` for HDR workflows or
    ///   `Rgba32Float` for maximum precision.
    pub async fn new_headless(
        init_config: &RendererInitConfig,
        settings: &RendererSettings,
        width: u32,
        height: u32,
        target_format: Option<wgpu::TextureFormat>,
    ) -> Result<Self> {
        let instance_desc = match init_config.backends {
            Some(backends) => wgpu::InstanceDescriptor {
                backends,
                ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
            },
            None => wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        };
        let instance = wgpu::Instance::new(instance_desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: init_config.power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| Error::Platform(PlatformError::AdapterNotFound(e.to_string())))?;

        let info = adapter.get_info();
        log::debug!("Backend (headless): {:?}", info.backend);
        log::debug!("Device: {}", info.name);

        // let required_limits = Self::requested_limits_for_adapter(init_config, &adapter);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Headless Device"),
                required_features: init_config.required_features,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|e| {
                Error::Render(myth_core::RenderError::RequestDeviceFailed(e.to_string()))
            })?;

        let gpu_driven_capabilities = GpuDrivenCapabilities::capture(&adapter, &device);

        let view_format = target_format.unwrap_or(wgpu::TextureFormat::Rgba8UnormSrgb);

        let headless_texture = Self::create_headless_texture(&device, width, height, view_format);

        Ok(Self {
            device,
            queue,
            surface: None,
            config: None,
            headless_texture: Some(headless_texture),
            target_width: width,
            target_height: height,
            depth_format: init_config.depth_format,
            surface_view_format: view_format,
            msaa_samples: 1,
            anisotropy_clamp: settings.anisotropy_clamp,
            render_path: settings.path,
            pipeline_settings_version: 0,
            gpu_driven_capabilities,
        })
    }

    /// Builds an isolated headless context around a process-shared test
    /// device. Each caller still owns its target texture and all higher-level
    /// renderer state; only the immutable wgpu device/queue handles are shared.
    #[cfg(test)]
    pub(crate) fn from_shared_test_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        gpu_driven_capabilities: GpuDrivenCapabilities,
        init_config: &RendererInitConfig,
        settings: &RendererSettings,
        width: u32,
        height: u32,
        target_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        assert!(
            device.features().contains(init_config.required_features),
            "shared test device does not satisfy the requested feature set"
        );

        let view_format = target_format.unwrap_or(wgpu::TextureFormat::Rgba8UnormSrgb);
        let headless_texture = Self::create_headless_texture(&device, width, height, view_format);

        Self {
            device,
            queue,
            surface: None,
            config: None,
            headless_texture: Some(headless_texture),
            target_width: width,
            target_height: height,
            depth_format: init_config.depth_format,
            surface_view_format: view_format,
            msaa_samples: 1,
            anisotropy_clamp: settings.anisotropy_clamp,
            render_path: settings.path,
            pipeline_settings_version: 0,
            gpu_driven_capabilities,
        }
    }

    /// Creates the offscreen render target texture for headless mode.
    fn create_headless_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Headless Render Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.target_width = width;
            self.target_height = height;

            if let Some(config) = &mut self.config {
                config.width = width;
                config.height = height;
                if let Some(surface) = &self.surface {
                    surface.configure(&self.device, config);
                }
            }

            if self.headless_texture.is_some() {
                self.headless_texture = Some(Self::create_headless_texture(
                    &self.device,
                    width,
                    height,
                    self.surface_view_format,
                ));
            }
        }
    }

    /// Dynamically reconfigure the surface present mode for VSync toggling.
    ///
    /// This is a no-op in headless mode where no surface exists.
    pub fn set_vsync(&mut self, vsync: bool) {
        let Some(config) = &mut self.config else {
            return;
        };

        let present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        if config.present_mode != present_mode {
            config.present_mode = present_mode;
            if let Some(surface) = &self.surface {
                surface.configure(&self.device, config);
            }
            log::info!("Surface reconfigured — VSync: {vsync}");
        }
    }

    #[must_use]
    pub fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Returns the current render target dimensions.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }

    /// Returns `true` if the context is running in headless (offscreen) mode.
    #[inline]
    #[must_use]
    pub fn is_headless(&self) -> bool {
        self.surface.is_none()
    }

    /// Returns the immutable GPU-driven capability snapshot captured during
    /// device initialization.
    #[inline]
    #[must_use]
    pub const fn gpu_driven_capabilities(&self) -> &GpuDrivenCapabilities {
        &self.gpu_driven_capabilities
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuDrivenCapabilities, GpuDrivenTier};

    fn baseline_downlevel() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::COMPUTE_SHADERS
                | wgpu::DownlevelFlags::INDIRECT_EXECUTION
                | wgpu::DownlevelFlags::VERTEX_STORAGE,
            ..Default::default()
        }
    }

    #[test]
    fn baseline_capabilities_select_indirect_tier() {
        let capabilities = GpuDrivenCapabilities::from_parts(
            wgpu::Features::empty(),
            wgpu::Features::empty(),
            baseline_downlevel(),
            wgpu::Limits::default(),
            wgpu::Limits::default(),
        );

        assert_eq!(capabilities.tier(), GpuDrivenTier::Indirect);
        assert!(capabilities.supports_indirect_primitives());
        assert!(!capabilities.supports_multi_draw_indirect_count());
    }

    #[test]
    fn adapter_only_optional_features_are_not_reported_as_enabled() {
        let optional =
            wgpu::Features::INDIRECT_FIRST_INSTANCE | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        let capabilities = GpuDrivenCapabilities::from_parts(
            optional,
            wgpu::Features::empty(),
            baseline_downlevel(),
            wgpu::Limits::default(),
            wgpu::Limits::default(),
        );

        assert!(capabilities.adapter_features().contains(optional));
        assert!(!capabilities.device_features().intersects(optional));
        assert_eq!(capabilities.tier(), GpuDrivenTier::Indirect);
        assert!(!capabilities.supports_indirect_first_instance());
        assert!(!capabilities.supports_multi_draw_indirect_count());
    }

    #[test]
    fn enabled_indirect_count_promotes_the_device_tier() {
        let capabilities = GpuDrivenCapabilities::from_parts(
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
            baseline_downlevel(),
            wgpu::Limits::default(),
            wgpu::Limits::default(),
        );

        assert_eq!(capabilities.tier(), GpuDrivenTier::MultiDrawIndirectCount);
        assert!(capabilities.supports_multi_draw_indirect_count());
    }

    #[test]
    fn missing_downlevel_requirement_disables_gpu_driven_baseline() {
        let mut downlevel = baseline_downlevel();
        downlevel
            .flags
            .remove(wgpu::DownlevelFlags::INDIRECT_EXECUTION);
        let capabilities = GpuDrivenCapabilities::from_parts(
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
            downlevel,
            wgpu::Limits::default(),
            wgpu::Limits::default(),
        );

        assert_eq!(capabilities.tier(), GpuDrivenTier::Unavailable);
        assert!(!capabilities.supports_indirect_primitives());
        assert!(!capabilities.supports_multi_draw_indirect_count());
    }

    #[test]
    fn zero_storage_limit_disables_gpu_driven_baseline() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 0,
            ..wgpu::Limits::default()
        };
        let capabilities = GpuDrivenCapabilities::from_parts(
            wgpu::Features::empty(),
            wgpu::Features::empty(),
            baseline_downlevel(),
            wgpu::Limits::default(),
            limits,
        );

        assert_eq!(capabilities.tier(), GpuDrivenTier::Unavailable);
    }
}
