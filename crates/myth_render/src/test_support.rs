//! Shared infrastructure for crate-local GPU contract tests.
//!
//! A wgpu device is expensive to request and is safe to clone across isolated
//! test fixtures. Renderer state, caches, resource managers, graph storage,
//! and render targets deliberately remain per-test.

use std::future::{Ready, ready};
use std::sync::OnceLock;

use myth_core::Result;

use crate::core::{GpuDrivenCapabilities, WgpuContext};
use crate::settings::{RendererInitConfig, RendererSettings};

struct SharedTestGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gpu_driven_capabilities: GpuDrivenCapabilities,
}

static SHARED_TEST_GPU: OnceLock<SharedTestGpu> = OnceLock::new();

fn shared_test_gpu() -> &'static SharedTestGpu {
    SHARED_TEST_GPU.get_or_init(|| {
        let context = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            1,
            1,
            None,
        ))
        .expect("shared headless GPU context required");
        let gpu_driven_capabilities = context.gpu_driven_capabilities().clone();

        let gpu_driven_capabilities = context.gpu_driven_capabilities().clone();
        SharedTestGpu {
            device: context.device,
            queue: context.queue,
            gpu_driven_capabilities,
        }
    })
}

/// Test-only constructor with the same call shape as
/// [`WgpuContext::new_headless`]. Returning a ready future lets existing test
/// fixtures keep using `pollster::block_on` while the one-time device request
/// happens before that outer executor starts.
pub(crate) struct SharedTestContext;

impl SharedTestContext {
    pub(crate) fn new_headless(
        init_config: &RendererInitConfig,
        settings: &RendererSettings,
        width: u32,
        height: u32,
        target_format: Option<wgpu::TextureFormat>,
    ) -> Ready<Result<WgpuContext>> {
        let shared = shared_test_gpu();
        ready(Ok(WgpuContext::from_shared_test_device(
            shared.device.clone(),
            shared.queue.clone(),
            shared.gpu_driven_capabilities.clone(),
            init_config,
            settings,
            width,
            height,
            target_format,
        )))
    }
}
