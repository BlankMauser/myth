//! RDG Gaussian Splatting Pass — 3D Gaussian Splatting Rendering
//!
//! Implements Myth's 3DGS path in three explicit stages:
//!
//! 1. Preprocess — project 3D Gaussians to 2D screen-space splats, evaluate
//!    SH colour, cull invisible points, and emit reverse-Z depth keys.
//! 2. Portable GPU radix sort — build per-workgroup histograms, scan them with
//!    a hierarchical Blelloch prefix sum, then stably scatter front-to-back.
//! 3. Render — draw storage-buffer-pulled triangle strips into an isolated
//!    non-linear accumulation target, then composite that result back into
//!    Myth's linear HDR scene colour.
//!
//! Multiple Gaussian clouds are supported simultaneously. Each cloud owns its
//! own preprocess buffers, sort buffers, and indirect draw buffer.

use std::sync::{Arc, Weak};

use glam::{Mat4, Vec3, Vec3A};
use rustc_hash::FxHashMap;

use crate::core::gpu::{CommonSampler, GlobalStateKey, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::gaussian::{GaussianFrameBatchId, GaussianFrameCloud};
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, PassNode, PrepareContext, RawBufferBinding,
    RenderTargetOps, TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, ComputePipelineId, ComputePipelineKey, DepthStencilKey, FullscreenPipelineKey,
    MultisampleKey, PipelineCache, RenderPipelineId, ShaderCompilationOptions, ShaderManager,
    ShaderSource,
};
use myth_assets::AssetServer;
use myth_resources::GaussianCloudHandle;
use myth_resources::gaussian_splat::{
    GaussianCloud, GaussianSHCoefficients, GaussianSplat, Splat2D,
};
use myth_resources::image::ColorSpace;
use myth_scene::{RenderCamera, Scene};

const PREPROCESS_WG_SIZE: u32 = 256;

const SORT_WG_SIZE_X: u32 = 16;
const SORT_WG_SIZE_Y: u32 = 16;
const SORT_THREADS_PER_WG: usize = (SORT_WG_SIZE_X * SORT_WG_SIZE_Y) as usize;
const SORT_ITEMS_PER_THREAD: usize = 8;
const SORT_KEYS_PER_WG: usize = SORT_THREADS_PER_WG * SORT_ITEMS_PER_THREAD;
const SORT_RADIX_BITS: u32 = 4;
const SORT_RADIX_SIZE: usize = 1 << SORT_RADIX_BITS;
const SORT_PASSES: usize = 32 / SORT_RADIX_BITS as usize;
const SORT_SCAN_ITEMS_PER_WG: usize = SORT_THREADS_PER_WG * 2;
const SORT_MAX_SCAN_LEVELS: usize = 4;
#[cfg(test)]
const SORT_PIPELINE_COUNT: usize =
    SORT_PASSES * 2 + SORT_MAX_SCAN_LEVELS + (SORT_MAX_SCAN_LEVELS - 1);

const _: () = {
    assert!(SORT_SCAN_ITEMS_PER_WG == SORT_THREADS_PER_WG * 2);
    assert!(SORT_SCAN_ITEMS_PER_WG.is_power_of_two());
    assert!(SORT_THREADS_PER_WG.is_multiple_of(32));
    assert!(SORT_RADIX_SIZE <= SORT_THREADS_PER_WG);
    assert!(SORT_RADIX_SIZE.is_power_of_two());
    assert!(SORT_PASSES.is_multiple_of(2));
};

const SPLAT_VERTEX_COUNT: u32 = 4;
const SORT_DISPATCH_INDIRECT_OFFSET: u64 = std::mem::size_of::<[u32; 3]>() as u64;
const GS_ACCUMULATION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCompositeSettings {
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRenderSettings {
    gaussian_scaling: f32,
    max_sh_deg: u32,
    mip_splatting: u32,
    kernel_size: f32,
    scene_extent: f32,
    color_space_flag: u32,
    opacity_compensation: f32,
    _pad0: u32,
    model_matrix: [f32; 16],
    model_inv_matrix: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSortInfos {
    keys_size: u32,
    max_workgroups: u32,
    scan_levels: u32,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuDrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GaussianPreprocessPipelineKey {
    global_layout_id: u64,
    shader_identity: u64,
    sort_config: GaussianSortConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GaussianSortPipelineKey {
    shader_identity: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GaussianRenderPipelineKey {
    shader_identity: u64,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GaussianCompositePipelineKey {
    shader_identity: u64,
    color_format: wgpu::TextureFormat,
    msaa_samples: u32,
}

#[derive(Clone, Copy, Debug)]
struct SortBufferLayout {
    key_capacity: usize,
    max_workgroups: usize,
    histogram_words: usize,
    internal_buffer_words: usize,
    scan_level_count: usize,
    scan_workgroups: [u32; SORT_MAX_SCAN_LEVELS],
}

impl SortBufferLayout {
    fn for_key_count(key_count: usize) -> Self {
        let key_capacity = key_count.max(1);
        let max_workgroups = key_capacity.div_ceil(SORT_KEYS_PER_WG);
        let histogram_words = SORT_RADIX_SIZE
            .checked_mul(max_workgroups)
            .expect("Gaussian sort histogram size overflow");

        let mut scan_workgroups = [0; SORT_MAX_SCAN_LEVELS];
        let mut scan_level_count = 0;
        let mut level_words = histogram_words;
        let mut internal_buffer_words = histogram_words;

        loop {
            assert!(
                scan_level_count < SORT_MAX_SCAN_LEVELS,
                "Gaussian sort prefix scan exceeds supported hierarchy depth"
            );
            let workgroups = level_words.div_ceil(SORT_SCAN_ITEMS_PER_WG);
            scan_workgroups[scan_level_count] =
                u32::try_from(workgroups).expect("Gaussian sort scan dispatch exceeds u32");
            scan_level_count += 1;
            internal_buffer_words = internal_buffer_words
                .checked_add(workgroups)
                .expect("Gaussian sort scratch size overflow");

            if workgroups == 1 {
                break;
            }
            level_words = workgroups;
        }

        Self {
            key_capacity,
            max_workgroups,
            histogram_words,
            internal_buffer_words,
            scan_level_count,
            scan_workgroups,
        }
    }
}

/// Compatibility shell carried through the deferred Gaussian plan. The
/// portable sorter deliberately has one device-independent configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct GaussianSortConfig;

impl GaussianSortConfig {
    fn for_device(_device: &wgpu::Device) -> Self {
        Self
    }

    const fn keys_per_wg_u32() -> u32 {
        SORT_KEYS_PER_WG as u32
    }

    fn buffer_layout(key_count: usize) -> SortBufferLayout {
        SortBufferLayout::for_key_count(key_count)
    }
}

fn build_sort_shader_options() -> ShaderCompilationOptions {
    let mut options = ShaderCompilationOptions::default();
    options.add_define("SORT_WG_SIZE_X", &SORT_WG_SIZE_X.to_string());
    options.add_define("SORT_WG_SIZE_Y", &SORT_WG_SIZE_Y.to_string());
    options.add_define("SORT_THREADS_PER_WG", &SORT_THREADS_PER_WG.to_string());
    options.add_define("SORT_ITEMS_PER_THREAD", &SORT_ITEMS_PER_THREAD.to_string());
    options.add_define("SORT_RADIX_BITS", &SORT_RADIX_BITS.to_string());
    options.add_define("SORT_RADIX_SIZE", &SORT_RADIX_SIZE.to_string());
    options.add_define(
        "SORT_SCAN_ITEMS_PER_WG",
        &SORT_SCAN_ITEMS_PER_WG.to_string(),
    );
    options
}

fn create_sort_pipelines(
    device: &wgpu::Device,
    shader_manager: &mut ShaderManager,
    pipeline_cache: &mut PipelineCache,
    sort_layout: &wgpu::BindGroupLayout,
) -> GaussianSortPipelines {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("GS Sort Pipeline Layout"),
        bind_group_layouts: &[Some(sort_layout)],
        immediate_size: 0,
    });
    let make_pipeline = |pipeline_cache: &mut PipelineCache,
                         module: &wgpu::ShaderModule,
                         shader_hash: u128,
                         entry_point: &'static str,
                         label: &str| {
        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let canonical = ComputePipelineKey::new(shader_hash).with_entry_point(entry_point);
        pipeline_cache.get_or_create_compute(
            device,
            module,
            &pipeline_layout,
            &canonical,
            &compilation_options,
            label,
        )
    };

    // Generate pass constants directly into distinct WGSL modules. In
    // particular, do not use pipeline override values here: Safari/WebKit can
    // incorrectly reuse a specialization across pipelines from one module.
    let mut radix_passes = Vec::with_capacity(SORT_PASSES);
    for pass in 0..SORT_PASSES {
        let mut options = build_sort_shader_options();
        options.add_define(
            "SORT_RADIX_SHIFT",
            &(u32::try_from(pass).expect("Gaussian radix pass exceeds u32") * SORT_RADIX_BITS)
                .to_string(),
        );
        options.add_define(
            "SORT_WRITE_KEYS",
            &u32::from(pass + 1 != SORT_PASSES).to_string(),
        );
        options.add_define("SORT_SCAN_LEVEL", "0");
        let (module, shader_hash) = shader_manager.get_or_compile(
            device,
            ShaderSource::File("entry/utility/3dgs/gs_radix_sort"),
            &options,
        );
        radix_passes.push(GaussianRadixPassPipelines {
            histogram: make_pipeline(
                pipeline_cache,
                module,
                shader_hash,
                "block_histogram",
                &format!("GS Sort Histogram {pass}"),
            ),
            scatter: make_pipeline(
                pipeline_cache,
                module,
                shader_hash,
                "ranked_scatter",
                &format!("GS Sort Scatter {pass}"),
            ),
        });
    }

    let mut prefix_scan = Vec::with_capacity(SORT_MAX_SCAN_LEVELS);
    let mut prefix_add = Vec::with_capacity(SORT_MAX_SCAN_LEVELS.saturating_sub(1));
    for level in 0..SORT_MAX_SCAN_LEVELS {
        let mut options = build_sort_shader_options();
        options.add_define("SORT_RADIX_SHIFT", "0");
        options.add_define("SORT_WRITE_KEYS", "1");
        options.add_define(
            "SORT_SCAN_LEVEL",
            &u32::try_from(level)
                .expect("Gaussian scan level exceeds u32")
                .to_string(),
        );
        let (module, shader_hash) = shader_manager.get_or_compile(
            device,
            ShaderSource::File("entry/utility/3dgs/gs_radix_sort"),
            &options,
        );
        prefix_scan.push(make_pipeline(
            pipeline_cache,
            module,
            shader_hash,
            "prefix_scan",
            &format!("GS Sort Prefix Scan {level}"),
        ));
        if level + 1 < SORT_MAX_SCAN_LEVELS {
            prefix_add.push(make_pipeline(
                pipeline_cache,
                module,
                shader_hash,
                "prefix_add",
                &format!("GS Sort Prefix Add {level}"),
            ));
        }
    }

    GaussianSortPipelines {
        radix_passes: radix_passes
            .try_into()
            .unwrap_or_else(|_| unreachable!("Gaussian radix pass count is fixed")),
        prefix_scan: prefix_scan
            .try_into()
            .unwrap_or_else(|_| unreachable!("Gaussian prefix scan level count is fixed")),
        prefix_add: prefix_add
            .try_into()
            .unwrap_or_else(|_| unreachable!("Gaussian prefix add level count is fixed")),
    }
}

#[derive(Clone, Copy)]
struct GaussianRadixPassPipelines {
    histogram: ComputePipelineId,
    scatter: ComputePipelineId,
}

#[derive(Clone, Copy)]
struct GaussianSortPipelines {
    radix_passes: [GaussianRadixPassPipelines; SORT_PASSES],
    prefix_scan: [ComputePipelineId; SORT_MAX_SCAN_LEVELS],
    prefix_add: [ComputePipelineId; SORT_MAX_SCAN_LEVELS - 1],
}

struct CloudGpuData {
    gaussian_buf: Tracked<wgpu::Buffer>,
    sh_buf: Tracked<wgpu::Buffer>,
}

struct CachedCloudGpuData {
    version: u32,
    asset_identity: Weak<GaussianCloud>,
    last_used_frame: u64,
    gpu: CloudGpuData,
}

#[derive(Clone, Copy)]
struct GaussianCloudPlan {
    asset_index: u32,
    handle: GaussianCloudHandle,
    version: u32,
    num_points: u32,
    num_sh_coefficients: u32,
    sort_layout: SortBufferLayout,
    sh_degree: u32,
    mip_splatting: bool,
    kernel_size: f32,
    scene_extent: f32,
    color_space_flag: u32,
    opacity_compensation: f32,
    model_matrix: [f32; 16],
}

pub(crate) struct GaussianExtract {
    assets: Vec<GaussianFrameCloud>,
    plans: Vec<GaussianCloudPlan>,
    composite_settings: GpuCompositeSettings,
    sort_config: GaussianSortConfig,
}

#[cfg(test)]
impl GaussianExtract {
    pub(crate) fn asset_count(&self) -> usize {
        self.assets.len()
    }

    pub(crate) fn plan_asset_indices(&self) -> Vec<u32> {
        self.plans.iter().map(|plan| plan.asset_index).collect()
    }

    pub(crate) fn asset_version(&self, index: usize) -> u32 {
        self.assets[index].version
    }
}

#[derive(Clone, Copy)]
pub(crate) struct GaussianSnapshot<'a> {
    batch: GaussianFrameBatchId,
    plans: &'a [GaussianCloudPlan],
    composite_settings: GpuCompositeSettings,
    sort_config: GaussianSortConfig,
    global_state_key: GlobalStateKey,
    frame_index: u64,
}

impl GaussianSnapshot<'_> {
    #[must_use]
    pub(crate) const fn batch_id(self) -> GaussianFrameBatchId {
        self.batch
    }
}

#[derive(Clone, Copy)]
struct CloudGraphBuffers {
    splat_buf: BufferNodeId,
    sort_infos_buf: BufferNodeId,
    sort_dispatch_buf: BufferNodeId,
    sort_internal_buf: BufferNodeId,
    sort_depths_a_buf: BufferNodeId,
    sort_depths_b_buf: BufferNodeId,
    sort_indices_a_buf: BufferNodeId,
    sort_indices_b_buf: BufferNodeId,
    draw_indirect_buf: BufferNodeId,
    render_settings_buf: BufferNodeId,
    num_points: u32,
    sort_layout: SortBufferLayout,
    sort_infos_init: GpuSortInfos,
    draw_indirect_init: GpuDrawIndirect,
}

#[derive(Clone, Copy)]
struct CloudComputeState<'a> {
    buffers: CloudGraphBuffers,
    plan: GaussianCloudPlan,
    preprocess_bg1: Option<&'a wgpu::BindGroup>,
    preprocess_bg2: Option<&'a wgpu::BindGroup>,
    preprocess_bg3: Option<&'a wgpu::BindGroup>,
    sort_bg_a_to_b: Option<&'a wgpu::BindGroup>,
    sort_bg_b_to_a: Option<&'a wgpu::BindGroup>,
}

#[derive(Clone, Copy)]
struct CloudRenderState<'a> {
    buffers: CloudGraphBuffers,
    render_bg: Option<&'a wgpu::BindGroup>,
}

struct GaussianComputeResources {
    preprocess_pipeline: Option<ComputePipelineId>,
    preprocess_pipeline_key: Option<GaussianPreprocessPipelineKey>,
    sort_pipelines: Option<GaussianSortPipelines>,
    sort_pipeline_key: Option<GaussianSortPipelineKey>,
    preprocess_layout_g1: Option<Tracked<wgpu::BindGroupLayout>>,
    preprocess_layout_g2: Option<Tracked<wgpu::BindGroupLayout>>,
    preprocess_layout_g3: Option<Tracked<wgpu::BindGroupLayout>>,
    sort_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    clouds: FxHashMap<GaussianCloudHandle, CachedCloudGpuData>,
}

struct GaussianRenderResources {
    pipeline: Option<RenderPipelineId>,
    pipeline_key: Option<GaussianRenderPipelineKey>,
    render_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

struct GaussianCompositeResources {
    pipeline: Option<RenderPipelineId>,
    pipeline_key: Option<GaussianCompositePipelineKey>,
    composite_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

pub struct GaussianSplattingFeature {
    compute: GaussianComputeResources,
    render: GaussianRenderResources,
    composite: GaussianCompositeResources,
    mixed_color_space_warned: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GaussianStaticCacheIdentity {
    preprocess_pipeline: Option<usize>,
    sort_pipelines: Option<[usize; SORT_PIPELINE_COUNT]>,
    render_pipeline: Option<usize>,
    composite_pipeline: Option<usize>,
    layout_ids: [Option<u64>; 6],
}

impl Default for GaussianSplattingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussianSplattingFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            compute: GaussianComputeResources {
                preprocess_pipeline: None,
                preprocess_pipeline_key: None,
                sort_pipelines: None,
                sort_pipeline_key: None,
                preprocess_layout_g1: None,
                preprocess_layout_g2: None,
                preprocess_layout_g3: None,
                sort_layout: None,
                clouds: FxHashMap::default(),
            },
            render: GaussianRenderResources {
                pipeline: None,
                pipeline_key: None,
                render_layout: None,
            },
            composite: GaussianCompositeResources {
                pipeline: None,
                pipeline_key: None,
                composite_layout: None,
            },
            mixed_color_space_warned: false,
        }
    }

    /// Captures only graph-shaping metadata plus exact-version Arc owners.
    /// No WGPU resource is created or uploaded here.
    #[must_use]
    pub(crate) fn extract(
        &mut self,
        scene: &Scene,
        assets: &AssetServer,
        camera: &RenderCamera,
        device: &wgpu::Device,
    ) -> Option<GaussianExtract> {
        let sort_config = GaussianSortConfig::for_device(device);
        let limits = device.limits();
        let mut frame_assets = Vec::new();
        let mut asset_indices = FxHashMap::default();
        let mut ordered_plans = Vec::new();

        for (node_handle, cloud_handle) in &scene.gaussian_clouds {
            let Some(node) = scene.get_node(node_handle) else {
                continue;
            };
            if !node.visible {
                continue;
            }
            // Capture one exact asset version per handle for the whole graph.
            // A concurrent replacement between two instances must not make one
            // frame contain two versions that compete for the same cache slot.
            let (asset_index, version, cloud) = if let Some(&index) =
                asset_indices.get(cloud_handle)
            {
                let captured: &GaussianFrameCloud = &frame_assets[index as usize];
                (index, captured.version, captured.asset.clone())
            } else {
                let Some((cloud, version)) = assets.gaussian_clouds.get_entry(*cloud_handle) else {
                    continue;
                };
                let index = u32::try_from(frame_assets.len())
                    .expect("Gaussian frame asset count exceeds u32");
                frame_assets.push(GaussianFrameCloud {
                    handle: *cloud_handle,
                    version,
                    asset: cloud.clone(),
                });
                asset_indices.insert(*cloud_handle, index);
                (index, version, cloud)
            };

            let num_points =
                u32::try_from(cloud.num_points).expect("Gaussian cloud exceeds u32 capacity");
            let num_sh_coefficients = u32::try_from(cloud.sh_coefficients.len())
                .expect("Gaussian SH coefficient table exceeds u32 capacity");
            let sort_layout = GaussianSortConfig::buffer_layout(cloud.num_points);
            let preprocess_workgroups = num_points.div_ceil(PREPROCESS_WG_SIZE);
            assert!(
                preprocess_workgroups <= limits.max_compute_workgroups_per_dimension,
                "Gaussian cloud requires {preprocess_workgroups} preprocess workgroups, exceeding the device limit of {}",
                limits.max_compute_workgroups_per_dimension
            );
            assert!(
                sort_layout.max_workgroups <= limits.max_compute_workgroups_per_dimension as usize,
                "Gaussian cloud requires {} sort workgroups, exceeding the device limit of {}",
                sort_layout.max_workgroups,
                limits.max_compute_workgroups_per_dimension
            );
            let key_buffer_size = sort_layout
                .key_capacity
                .checked_mul(std::mem::size_of::<u32>())
                .expect("Gaussian sort key buffer size overflow");
            let internal_buffer_size = sort_layout
                .internal_buffer_words
                .checked_mul(std::mem::size_of::<u32>())
                .expect("Gaussian sort scratch buffer size overflow");
            assert!(
                key_buffer_size <= limits.max_storage_buffer_binding_size as usize,
                "Gaussian sort key buffer ({key_buffer_size} bytes) exceeds the device storage binding limit ({})",
                limits.max_storage_buffer_binding_size
            );
            assert!(
                internal_buffer_size <= limits.max_storage_buffer_binding_size as usize,
                "Gaussian sort scratch buffer ({internal_buffer_size} bytes) exceeds the device storage binding limit ({})",
                limits.max_storage_buffer_binding_size
            );
            let model_matrix = Mat4::from(*node.world_matrix());
            let local_center = Vec3::new(cloud.center.x, cloud.center.y, cloud.center.z);
            let world_center = Vec3A::from(model_matrix.transform_point3(local_center));
            let distance_sq = camera.position.distance_squared(world_center);

            ordered_plans.push((
                GaussianCloudPlan {
                    asset_index,
                    handle: *cloud_handle,
                    version,
                    num_points,
                    num_sh_coefficients,
                    sort_layout,
                    sh_degree: cloud.sh_degree,
                    mip_splatting: cloud.mip_splatting,
                    kernel_size: cloud.kernel_size,
                    scene_extent: cloud.scene_extent().max(1e-5),
                    color_space_flag: match cloud.color_space {
                        ColorSpace::Linear => 0,
                        ColorSpace::Srgb => 1,
                    },
                    opacity_compensation: cloud.opacity_compensation,
                    model_matrix: model_matrix.to_cols_array(),
                },
                distance_sq,
            ));
        }

        if ordered_plans.is_empty() {
            return None;
        }

        ordered_plans.sort_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let plans: Vec<_> = ordered_plans.into_iter().map(|(plan, _)| plan).collect();

        let first_color_space = plans[0].color_space_flag;
        let mixed_color_space = plans
            .iter()
            .any(|plan| plan.color_space_flag != first_color_space);
        if mixed_color_space && !self.mixed_color_space_warned {
            log::warn!(
                "Gaussian splatting accumulation expects a consistent cloud color space; mixed clouds will use the first visible cloud"
            );
            self.mixed_color_space_warned = true;
        }

        Some(GaussianExtract {
            assets: frame_assets,
            plans,
            composite_settings: GpuCompositeSettings {
                flags: [first_color_space, 0, 0, 0],
            },
            sort_config,
        })
    }

    pub(crate) fn stage_snapshot<'a>(
        ctx: &mut GraphBuilderContext<'a, '_>,
        extract: GaussianExtract,
        global_state_key: GlobalStateKey,
        frame_index: u64,
    ) -> GaussianSnapshot<'a> {
        let batch = ctx
            .graph
            .stage_gaussian_frame_batch(extract.assets)
            .expect("non-empty Gaussian extract must own at least one asset");
        let plans = ctx.graph.alloc_slice(&extract.plans);
        GaussianSnapshot {
            batch,
            plans,
            composite_settings: extract.composite_settings,
            sort_config: extract.sort_config,
            global_state_key,
            frame_index,
        }
    }

    /// Retains cloud GPU uploads across alternating scenes and retires them
    /// only after they have not participated in a live Gaussian node for the
    /// requested number of renderer frames.
    pub(crate) fn garbage_collect(&mut self, current_frame: u64, ttl_frames: u64) -> usize {
        let before = self.compute.clouds.len();
        self.compute
            .clouds
            .retain(|_, cached| current_frame.saturating_sub(cached.last_used_frame) <= ttl_frames);
        before - self.compute.clouds.len()
    }

    #[cfg(test)]
    pub(crate) fn cache_is_empty(&self) -> bool {
        self.compute.preprocess_pipeline.is_none()
            && self.compute.preprocess_pipeline_key.is_none()
            && self.compute.sort_pipelines.is_none()
            && self.compute.sort_pipeline_key.is_none()
            && self.compute.preprocess_layout_g1.is_none()
            && self.compute.preprocess_layout_g2.is_none()
            && self.compute.preprocess_layout_g3.is_none()
            && self.compute.sort_layout.is_none()
            && self.compute.clouds.is_empty()
            && self.render.pipeline.is_none()
            && self.render.pipeline_key.is_none()
            && self.render.render_layout.is_none()
            && self.composite.pipeline.is_none()
            && self.composite.pipeline_key.is_none()
            && self.composite.composite_layout.is_none()
    }

    #[cfg(test)]
    pub(crate) fn static_cache_identity(&self) -> GaussianStaticCacheIdentity {
        let sort_pipelines = self.compute.sort_pipelines.map(|pipelines| {
            let mut identities = [0; SORT_PIPELINE_COUNT];
            let mut cursor = 0;
            for pass in pipelines.radix_passes {
                identities[cursor] = pass.histogram.index();
                identities[cursor + 1] = pass.scatter.index();
                cursor += 2;
            }
            for pipeline in pipelines.prefix_scan {
                identities[cursor] = pipeline.index();
                cursor += 1;
            }
            for pipeline in pipelines.prefix_add {
                identities[cursor] = pipeline.index();
                cursor += 1;
            }
            identities
        });
        GaussianStaticCacheIdentity {
            preprocess_pipeline: self
                .compute
                .preprocess_pipeline
                .map(ComputePipelineId::index),
            sort_pipelines,
            render_pipeline: self.render.pipeline.map(RenderPipelineId::index),
            composite_pipeline: self.composite.pipeline.map(RenderPipelineId::index),
            layout_ids: [
                self.compute.preprocess_layout_g1.as_ref().map(Tracked::id),
                self.compute.preprocess_layout_g2.as_ref().map(Tracked::id),
                self.compute.preprocess_layout_g3.as_ref().map(Tracked::id),
                self.compute.sort_layout.as_ref().map(Tracked::id),
                self.render.render_layout.as_ref().map(Tracked::id),
                self.composite.composite_layout.as_ref().map(Tracked::id),
            ],
        }
    }

    #[cfg(test)]
    pub(crate) fn static_cache_is_initialized(&self) -> bool {
        let identity = self.static_cache_identity();
        identity.preprocess_pipeline.is_some()
            && identity.sort_pipelines.is_some()
            && identity.render_pipeline.is_some()
            && identity.composite_pipeline.is_some()
            && identity.layout_ids.iter().all(Option::is_some)
    }

    #[cfg(test)]
    pub(crate) fn cloud_cache_identity(
        &self,
        handle: GaussianCloudHandle,
    ) -> Option<(u32, u64, u64)> {
        self.compute.clouds.get(&handle).map(|cached| {
            (
                cached.version,
                cached.gpu.gaussian_buf.id(),
                cached.gpu.sh_buf.id(),
            )
        })
    }

    #[cfg(test)]
    pub(crate) fn cloud_last_used_frame(&self, handle: GaussianCloudHandle) -> Option<u64> {
        self.compute
            .clouds
            .get(&handle)
            .map(|cached| cached.last_used_frame)
    }

    #[cfg(test)]
    pub(crate) fn cloud_cache_len(&self) -> usize {
        self.compute.clouds.len()
    }
}

impl GaussianComputeResources {
    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        use crate::core::gpu::binding_layout::{storage_buffer, uniform_buffer};

        if self.preprocess_layout_g1.is_some() {
            return;
        }

        let cs = wgpu::ShaderStages::COMPUTE;

        self.preprocess_layout_g1 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G1 (Packed Gaussians + SH + Packed Splats)"),
                entries: &[
                    storage_buffer(0, cs, true),
                    storage_buffer(1, cs, true),
                    storage_buffer(2, cs, false),
                ],
            },
        )));

        self.preprocess_layout_g2 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G2 (Sort Front Buffers)"),
                entries: &[
                    storage_buffer(0, cs, false),
                    storage_buffer(1, cs, false),
                    storage_buffer(2, cs, false),
                ],
            },
        )));

        self.preprocess_layout_g3 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G3 (Render Settings)"),
                entries: &[uniform_buffer(0, cs)],
            },
        )));

        self.sort_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Sort"),
                entries: &[
                    storage_buffer(0, cs, true),
                    storage_buffer(1, cs, false),
                    storage_buffer(2, cs, true),
                    storage_buffer(3, cs, false),
                    storage_buffer(4, cs, true),
                    storage_buffer(5, cs, false),
                ],
            },
        )));
    }

    fn ensure_pipelines(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        global_state_key: GlobalStateKey,
        sort_config: GaussianSortConfig,
    ) -> (ComputePipelineId, GaussianSortPipelines) {
        self.ensure_layouts(ctx.device);
        let (global_layout, global_layout_id, global_binding_wgsl) = {
            let global = ctx
                .global_states
                .get(global_state_key)
                .expect("GS preprocess requires a global render-state bind group");
            (
                global.layout.clone(),
                global.layout_id,
                global.binding_wgsl.clone(),
            )
        };

        let mut preprocess_options = ShaderCompilationOptions::default();
        preprocess_options.add_define(
            "GS_SORT_KEYS_PER_WG",
            &GaussianSortConfig::keys_per_wg_u32().to_string(),
        );
        preprocess_options.inject_code("binding_code", &global_binding_wgsl);
        preprocess_options.inject_code(
            "scene_lighting_structs",
            myth_resources::uniforms::scene_lighting_structs_wgsl(),
        );
        let preprocess_shader_identity = ctx.shader_manager.pipeline_shader_hash(
            ShaderSource::File("entry/utility/3dgs/gaussian_preprocess"),
            &preprocess_options,
        );
        let preprocess_key = GaussianPreprocessPipelineKey {
            global_layout_id,
            shader_identity: preprocess_shader_identity,
            sort_config,
        };

        if self.preprocess_pipeline.is_none()
            || self.preprocess_pipeline_key != Some(preprocess_key)
        {
            let (module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/utility/3dgs/gaussian_preprocess"),
                &preprocess_options,
            );

            let layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GS Preprocess Pipeline Layout"),
                    bind_group_layouts: &[
                        Some(&global_layout),
                        self.preprocess_layout_g1.as_deref(),
                        self.preprocess_layout_g2.as_deref(),
                        self.preprocess_layout_g3.as_deref(),
                    ],
                    immediate_size: 0,
                });

            let compilation_options = wgpu::PipelineCompilationOptions::default();
            let canonical_key = ComputePipelineKey::new(shader_hash);
            let pipeline_id = ctx.pipeline_cache.get_or_create_compute(
                ctx.device,
                module,
                &layout,
                &canonical_key,
                &compilation_options,
                "GS Preprocess Pipeline",
            );
            self.preprocess_pipeline = Some(pipeline_id);
            self.preprocess_pipeline_key = Some(preprocess_key);
        }

        let sort_options = build_sort_shader_options();
        let sort_key = GaussianSortPipelineKey {
            shader_identity: ctx.shader_manager.pipeline_shader_hash(
                ShaderSource::File("entry/utility/3dgs/gs_radix_sort"),
                &sort_options,
            ),
        };

        if self.sort_pipelines.is_none() || self.sort_pipeline_key != Some(sort_key) {
            let sort_layout = self
                .sort_layout
                .clone()
                .expect("GS sort layout missing after prepare");
            self.sort_pipelines = Some(create_sort_pipelines(
                ctx.device,
                ctx.shader_manager,
                ctx.pipeline_cache,
                &sort_layout,
            ));
            self.sort_pipeline_key = Some(sort_key);
        }

        (
            self.preprocess_pipeline
                .expect("GS preprocess pipeline missing after prepare"),
            self.sort_pipelines
                .expect("GS sort pipelines missing after prepare"),
        )
    }

    fn create_cloud_gpu_data(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cloud: &GaussianCloud,
    ) -> CloudGpuData {
        let upload_count = cloud.num_points.max(cloud.gaussians.len()).max(1);
        let sh_upload_count = cloud.sh_coefficients.len().max(1);

        let gaussian_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Gaussian Data"),
            size: (upload_count * std::mem::size_of::<GaussianSplat>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&gaussian_buf, 0, bytemuck::cast_slice(&cloud.gaussians));

        let sh_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS SH Coefficients"),
            size: (sh_upload_count * std::mem::size_of::<GaussianSHCoefficients>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&sh_buf, 0, bytemuck::cast_slice(&cloud.sh_coefficients));

        CloudGpuData {
            gaussian_buf,
            sh_buf,
        }
    }

    fn ensure_cloud(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_index: u64,
        frame_cloud: &GaussianFrameCloud,
    ) {
        let asset_identity = Arc::downgrade(&frame_cloud.asset);
        let needs_rebuild = self.clouds.get(&frame_cloud.handle).is_none_or(|cached| {
            cached.version != frame_cloud.version
                || !Weak::ptr_eq(&cached.asset_identity, &asset_identity)
        });
        if needs_rebuild {
            let gpu = Self::create_cloud_gpu_data(device, queue, &frame_cloud.asset);
            self.clouds.insert(
                frame_cloud.handle,
                CachedCloudGpuData {
                    version: frame_cloud.version,
                    asset_identity,
                    last_used_frame: frame_index,
                    gpu,
                },
            );
        } else {
            self.clouds
                .get_mut(&frame_cloud.handle)
                .expect("checked Gaussian cache entry missing")
                .last_used_frame = frame_index;
        }
    }

    #[must_use]
    fn cloud(&self, handle: GaussianCloudHandle, version: u32) -> &CloudGpuData {
        let cached = self
            .clouds
            .get(&handle)
            .expect("Gaussian cloud missing after live prepare");
        assert_eq!(
            cached.version, version,
            "Gaussian cloud cache version changed during one prepare"
        );
        &cached.gpu
    }

    #[must_use]
    fn render_settings(plan: GaussianCloudPlan) -> GpuRenderSettings {
        let model_matrix = Mat4::from_cols_array(&plan.model_matrix);
        GpuRenderSettings {
            gaussian_scaling: 1.0,
            max_sh_deg: plan.sh_degree,
            mip_splatting: u32::from(plan.mip_splatting),
            kernel_size: plan.kernel_size,
            scene_extent: plan.scene_extent,
            color_space_flag: plan.color_space_flag,
            opacity_compensation: plan.opacity_compensation,
            _pad0: 0,
            model_matrix: plan.model_matrix,
            model_inv_matrix: model_matrix.inverse().to_cols_array(),
        }
    }
}

impl GaussianRenderResources {
    fn ensure_pipeline(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        depth_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> RenderPipelineId {
        use crate::core::gpu::binding_layout::storage_buffer;

        let shader_options = ShaderCompilationOptions::default();
        let source = ShaderSource::File("entry/utility/3dgs/gaussian_render");
        let shader_identity = ctx
            .shader_manager
            .pipeline_shader_hash(source, &shader_options);
        let key = GaussianRenderPipelineKey {
            shader_identity,
            depth_format,
            msaa_samples,
        };
        if self.pipeline_key == Some(key)
            && let Some(pipeline) = self.pipeline
        {
            return pipeline;
        }

        let layout = self.render_layout.get_or_insert_with(|| {
            Tracked::new(
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("GS Render"),
                        entries: &[
                            storage_buffer(0, wgpu::ShaderStages::VERTEX, true),
                            storage_buffer(1, wgpu::ShaderStages::VERTEX, true),
                        ],
                    }),
            )
        });
        let (module, shader_hash) =
            ctx.shader_manager
                .get_or_compile(ctx.device, source, &shader_options);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Render Pipeline Layout"),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
        let target = ColorTargetKey::from(wgpu::ColorTargetState {
            format: GS_ACCUMULATION_FORMAT,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let mut canonical = FullscreenPipelineKey::fullscreen(
            shader_hash,
            smallvec::smallvec![target],
            Some(DepthStencilKey::from(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })),
        )
        .with_topology(wgpu::PrimitiveTopology::TriangleStrip);
        canonical.multisample = MultisampleKey::from(wgpu::MultisampleState {
            count: msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        });
        let pipeline = ctx.pipeline_cache.get_or_create_fullscreen(
            ctx.device,
            module,
            &pipeline_layout,
            &canonical,
            "GS Render Pipeline",
        );
        ctx.pipeline_cache
            .register_render_layouts(pipeline, smallvec::smallvec![layout.clone()]);
        self.pipeline = Some(pipeline);
        self.pipeline_key = Some(key);
        pipeline
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.render_layout
            .as_ref()
            .expect("GS render layout missing after prepare")
    }
}

impl GaussianCompositeResources {
    fn ensure_pipeline(
        &mut self,
        ctx: &mut PrepareContext<'_>,
        color_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> RenderPipelineId {
        use crate::core::gpu::binding_layout::{sampler, texture_2d, uniform_buffer};

        let shader_options = ShaderCompilationOptions::default();
        let source = ShaderSource::File("entry/utility/3dgs/gs_composite");
        let shader_identity = ctx
            .shader_manager
            .pipeline_shader_hash(source, &shader_options);
        let key = GaussianCompositePipelineKey {
            shader_identity,
            color_format,
            msaa_samples,
        };
        if self.pipeline_key == Some(key)
            && let Some(pipeline) = self.pipeline
        {
            return pipeline;
        }

        let layout = self.composite_layout.get_or_insert_with(|| {
            Tracked::new(
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("GS Composite"),
                        entries: &[
                            texture_2d(
                                0,
                                wgpu::ShaderStages::FRAGMENT,
                                wgpu::TextureSampleType::Float { filterable: true },
                            ),
                            sampler(
                                1,
                                wgpu::ShaderStages::FRAGMENT,
                                wgpu::SamplerBindingType::Filtering,
                            ),
                            uniform_buffer(2, wgpu::ShaderStages::FRAGMENT),
                        ],
                    }),
            )
        });
        let (module, shader_hash) =
            ctx.shader_manager
                .get_or_compile(ctx.device, source, &shader_options);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Composite Pipeline Layout"),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
        let target = ColorTargetKey::from(wgpu::ColorTargetState {
            format: color_format,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let mut canonical =
            FullscreenPipelineKey::fullscreen(shader_hash, smallvec::smallvec![target], None);
        canonical.multisample = MultisampleKey::from(wgpu::MultisampleState {
            count: msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        });
        let pipeline = ctx.pipeline_cache.get_or_create_fullscreen(
            ctx.device,
            module,
            &pipeline_layout,
            &canonical,
            "GS Composite Pipeline",
        );
        ctx.pipeline_cache
            .register_render_layouts(pipeline, smallvec::smallvec![layout.clone()]);
        self.pipeline = Some(pipeline);
        self.pipeline_key = Some(key);
        pipeline
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.composite_layout
            .as_ref()
            .expect("GS composite layout missing after prepare")
    }
}

impl GaussianSplattingFeature {
    pub(crate) fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: GaussianSnapshot<'a>,
        active_color: TextureNodeId,
        active_depth: TextureNodeId,
    ) -> TextureNodeId {
        let GaussianSplattingFeature {
            compute,
            render,
            composite,
            ..
        } = self;

        let cloud_buffers = ctx.graph.add_pass("GS_Compute", |builder| {
            let mut graph_buffers = Vec::with_capacity(snapshot.plans.len());
            let mut compute_states = Vec::with_capacity(snapshot.plans.len());

            for &plan in snapshot.plans {
                let upload_count = usize::try_from(plan.num_points.max(1))
                    .expect("Gaussian point count exceeds usize capacity");
                let sort_key_buffer_size =
                    (plan.sort_layout.key_capacity * std::mem::size_of::<u32>()) as u64;

                let splat_buf = builder.create_buffer(
                    "GS_Splats",
                    BufferDesc::new(
                        (upload_count * std::mem::size_of::<Splat2D>()) as u64,
                        wgpu::BufferUsages::STORAGE,
                    ),
                );
                let sort_infos_buf = builder.create_buffer(
                    "GS_Sort_Infos",
                    BufferDesc::new(
                        std::mem::size_of::<GpuSortInfos>() as u64,
                        wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    ),
                );
                let sort_dispatch_buf = builder.create_buffer(
                    "GS_Sort_Dispatch",
                    BufferDesc::new(
                        std::mem::size_of::<[u32; 3]>() as u64,
                        wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::INDIRECT,
                    ),
                );
                let sort_internal_buf = builder.create_buffer(
                    "GS_Sort_Internal",
                    BufferDesc::new(
                        (plan.sort_layout.internal_buffer_words * std::mem::size_of::<u32>())
                            as u64,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    ),
                );
                let sort_depths_a_buf = builder.create_buffer(
                    "GS_Sort_Depths_A",
                    BufferDesc::new(sort_key_buffer_size, wgpu::BufferUsages::STORAGE),
                );
                let alternate_depths = builder.create_buffer(
                    "GS_Sort_Depths_B",
                    BufferDesc::new(sort_key_buffer_size, wgpu::BufferUsages::STORAGE),
                );
                let sort_indices_a_buf = builder.create_buffer(
                    "GS_Sort_Indices_A",
                    BufferDesc::new(sort_key_buffer_size, wgpu::BufferUsages::STORAGE),
                );
                let alternate_indices = builder.create_buffer(
                    "GS_Sort_Indices_B",
                    BufferDesc::new(sort_key_buffer_size, wgpu::BufferUsages::STORAGE),
                );
                let draw_indirect_buf = builder.create_buffer(
                    "GS_Draw_Indirect",
                    BufferDesc::new(
                        std::mem::size_of::<GpuDrawIndirect>() as u64,
                        wgpu::BufferUsages::INDIRECT
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    ),
                );
                let render_settings_buf = builder.create_buffer(
                    "GS_Render_Settings",
                    BufferDesc::new(
                        std::mem::size_of::<GpuRenderSettings>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    ),
                );

                let buffers = CloudGraphBuffers {
                    splat_buf,
                    sort_infos_buf,
                    sort_dispatch_buf,
                    sort_internal_buf,
                    sort_depths_a_buf,
                    sort_depths_b_buf: alternate_depths,
                    sort_indices_a_buf,
                    sort_indices_b_buf: alternate_indices,
                    draw_indirect_buf,
                    render_settings_buf,
                    num_points: plan.num_points,
                    sort_layout: plan.sort_layout,
                    sort_infos_init: GpuSortInfos {
                        keys_size: 0,
                        max_workgroups: u32::try_from(plan.sort_layout.max_workgroups)
                            .expect("Gaussian sort workgroup count exceeds u32"),
                        scan_levels: u32::try_from(plan.sort_layout.scan_level_count)
                            .expect("Gaussian sort scan level count exceeds u32"),
                        dispatch_x: 0,
                        dispatch_y: 1,
                        dispatch_z: 1,
                    },
                    draw_indirect_init: GpuDrawIndirect {
                        vertex_count: SPLAT_VERTEX_COUNT,
                        instance_count: 0,
                        base_vertex: 0,
                        base_instance: 0,
                    },
                };
                graph_buffers.push(buffers);
                compute_states.push(CloudComputeState {
                    buffers,
                    plan,
                    preprocess_bg1: None,
                    preprocess_bg2: None,
                    preprocess_bg3: None,
                    sort_bg_a_to_b: None,
                    sort_bg_b_to_a: None,
                });
            }

            let graph_buffers = builder.graph.alloc_slice(&graph_buffers);
            let compute_states = builder.graph.alloc_slice_mut(&compute_states);
            (
                GaussianComputePassNode {
                    resources: compute,
                    batch: snapshot.batch,
                    global_state_key: snapshot.global_state_key,
                    sort_config: snapshot.sort_config,
                    frame_index: snapshot.frame_index,
                    preprocess_pipeline: None,
                    sort_pipelines: None,
                    clouds: compute_states,
                },
                graph_buffers,
            )
        });

        let gs_accumulation = ctx.graph.add_pass("GS_Render", |builder| {
            for &cloud in cloud_buffers {
                builder.read_buffer(cloud.splat_buf);
                builder.read_buffer(cloud.sort_indices_a_buf);
                builder.read_buffer(cloud.draw_indirect_buf);
            }
            builder.read_texture(active_depth);

            let accumulation_desc = TextureDesc::new_2d(
                ctx.frame_config.width,
                ctx.frame_config.height,
                GS_ACCUMULATION_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let accumulation_resolved =
                builder.create_texture("GS_Accumulation", accumulation_desc);
            let accumulation_target = if ctx.frame_config.msaa_samples > 1 {
                builder.create_texture(
                    "GS_Accumulation_MSAA",
                    TextureDesc::new(
                        ctx.frame_config.width,
                        ctx.frame_config.height,
                        1,
                        1,
                        ctx.frame_config.msaa_samples,
                        wgpu::TextureDimension::D2,
                        GS_ACCUMULATION_FORMAT,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                    ),
                )
            } else {
                accumulation_resolved
            };
            let render_states: Vec<_> = cloud_buffers
                .iter()
                .copied()
                .map(|buffers| CloudRenderState {
                    buffers,
                    render_bg: None,
                })
                .collect();
            let render_states = builder.graph.alloc_slice_mut(&render_states);
            (
                GaussianRenderPassNode {
                    resources: render,
                    pipeline: None,
                    msaa_samples: ctx.frame_config.msaa_samples,
                    depth_format: ctx.frame_config.depth_format,
                    clouds: render_states,
                    color_target: accumulation_target,
                    resolve_target: (ctx.frame_config.msaa_samples > 1)
                        .then_some(accumulation_resolved),
                    depth_target: active_depth,
                },
                accumulation_resolved,
            )
        });

        ctx.graph.add_pass("GS_Composite", |builder| {
            builder.read_texture(gs_accumulation);
            let composite_settings = builder.create_buffer(
                "GS_Composite_Settings",
                BufferDesc::new(
                    std::mem::size_of::<GpuCompositeSettings>() as u64,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
            );
            let color_out = builder.mutate_texture(active_color, "GS_Composite_Color");
            (
                GaussianCompositePassNode {
                    resources: composite,
                    pipeline: None,
                    msaa_samples: ctx.frame_config.msaa_samples,
                    color_format: ctx.frame_config.hdr_format,
                    settings: snapshot.composite_settings,
                    accumulation_tex: gs_accumulation,
                    composite_settings_buf: composite_settings,
                    color_target: color_out,
                    composite_bg: None,
                },
                color_out,
            )
        })
    }
}

struct GaussianComputePassNode<'a> {
    resources: &'a mut GaussianComputeResources,
    batch: GaussianFrameBatchId,
    global_state_key: GlobalStateKey,
    sort_config: GaussianSortConfig,
    frame_index: u64,
    preprocess_pipeline: Option<ComputePipelineId>,
    sort_pipelines: Option<GaussianSortPipelines>,
    clouds: &'a mut [CloudComputeState<'a>],
}

impl<'a> PassNode<'a> for GaussianComputePassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let (preprocess_pipeline, sort_pipelines) =
            self.resources
                .ensure_pipelines(ctx, self.global_state_key, self.sort_config);
        self.preprocess_pipeline = Some(preprocess_pipeline);
        self.sort_pipelines = Some(sort_pipelines);

        for cloud in self.clouds.iter() {
            let frame_cloud = ctx
                .gaussian_frames
                .cloud(self.batch, cloud.plan.asset_index);
            assert_eq!(frame_cloud.handle, cloud.plan.handle);
            assert_eq!(frame_cloud.version, cloud.plan.version);
            self.resources
                .ensure_cloud(ctx.device, ctx.queue, self.frame_index, frame_cloud);
        }

        let preprocess_layout_g1 = self.resources.preprocess_layout_g1.clone().unwrap();
        let preprocess_layout_g2 = self.resources.preprocess_layout_g2.clone().unwrap();
        let preprocess_layout_g3 = self.resources.preprocess_layout_g3.clone().unwrap();
        let sort_layout = self.resources.sort_layout.clone().unwrap();

        for cloud in self.clouds.iter_mut() {
            ctx.queue.write_buffer(
                ctx.views.get_buffer(cloud.buffers.sort_infos_buf),
                0,
                bytemuck::bytes_of(&cloud.buffers.sort_infos_init),
            );
            ctx.queue.write_buffer(
                ctx.views.get_buffer(cloud.buffers.draw_indirect_buf),
                0,
                bytemuck::bytes_of(&cloud.buffers.draw_indirect_init),
            );
            let render_settings = GaussianComputeResources::render_settings(cloud.plan);
            ctx.queue.write_buffer(
                ctx.views.get_buffer(cloud.buffers.render_settings_buf),
                0,
                bytemuck::bytes_of(&render_settings),
            );

            let gpu = self.resources.cloud(cloud.plan.handle, cloud.plan.version);
            let gaussian_size = wgpu::BufferSize::new(
                u64::from(cloud.plan.num_points.max(1))
                    * std::mem::size_of::<GaussianSplat>() as u64,
            );
            let sh_size = wgpu::BufferSize::new(
                u64::from(cloud.plan.num_sh_coefficients.max(1))
                    * std::mem::size_of::<GaussianSHCoefficients>() as u64,
            );

            let preprocess_bg1 = ctx
                .build_bind_group(&preprocess_layout_g1, Some("GS Preprocess BG1"))
                .bind_raw_buffer(
                    0,
                    RawBufferBinding::new(&gpu.gaussian_buf, gpu.gaussian_buf.id(), gaussian_size),
                )
                .bind_raw_buffer(
                    1,
                    RawBufferBinding::new(&gpu.sh_buf, gpu.sh_buf.id(), sh_size),
                )
                .bind_buffer(2, cloud.buffers.splat_buf)
                .build();

            let preprocess_bg2 = ctx
                .build_bind_group(&preprocess_layout_g2, Some("GS Preprocess BG2"))
                .bind_buffer(0, cloud.buffers.sort_infos_buf)
                .bind_buffer(1, cloud.buffers.sort_depths_a_buf)
                .bind_buffer(2, cloud.buffers.sort_indices_a_buf)
                .build();

            let preprocess_bg3 = ctx
                .build_bind_group(&preprocess_layout_g3, Some("GS Preprocess BG3"))
                .bind_buffer(0, cloud.buffers.render_settings_buf)
                .build();

            let sort_bg_a_to_b = ctx
                .build_bind_group(&sort_layout, Some("GS Sort BG A to B"))
                .bind_buffer(0, cloud.buffers.sort_infos_buf)
                .bind_buffer(1, cloud.buffers.sort_internal_buf)
                .bind_buffer(2, cloud.buffers.sort_depths_a_buf)
                .bind_buffer(3, cloud.buffers.sort_depths_b_buf)
                .bind_buffer(4, cloud.buffers.sort_indices_a_buf)
                .bind_buffer(5, cloud.buffers.sort_indices_b_buf)
                .build();
            let sort_bg_b_to_a = ctx
                .build_bind_group(&sort_layout, Some("GS Sort BG B to A"))
                .bind_buffer(0, cloud.buffers.sort_infos_buf)
                .bind_buffer(1, cloud.buffers.sort_internal_buf)
                .bind_buffer(2, cloud.buffers.sort_depths_b_buf)
                .bind_buffer(3, cloud.buffers.sort_depths_a_buf)
                .bind_buffer(4, cloud.buffers.sort_indices_b_buf)
                .bind_buffer(5, cloud.buffers.sort_indices_a_buf)
                .build();

            cloud.preprocess_bg1 = Some(preprocess_bg1);
            cloud.preprocess_bg2 = Some(preprocess_bg2);
            cloud.preprocess_bg3 = Some(preprocess_bg3);
            cloud.sort_bg_a_to_b = Some(sort_bg_a_to_b);
            cloud.sort_bg_b_to_a = Some(sort_bg_b_to_a);
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let preprocess_pipeline_id = self
            .preprocess_pipeline
            .expect("GS preprocess pipeline missing");
        let sort_pipelines = self.sort_pipelines.expect("GS sort pipelines missing");
        let preprocess_pipeline = ctx
            .pipeline_cache
            .get_compute_pipeline(preprocess_pipeline_id);
        let global_bind_group = ctx.baked_lists.global_bind_group;

        for cloud in self.clouds.iter() {
            let preprocess_workgroups = cloud.buffers.num_points.div_ceil(PREPROCESS_WG_SIZE);

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Preprocess"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(preprocess_pipeline);
                cpass.set_bind_group(0, global_bind_group, &[]);
                cpass.set_bind_group(
                    1,
                    cloud.preprocess_bg1.expect("GS preprocess BG1 missing"),
                    &[],
                );
                cpass.set_bind_group(
                    2,
                    cloud.preprocess_bg2.expect("GS preprocess BG2 missing"),
                    &[],
                );
                cpass.set_bind_group(
                    3,
                    cloud.preprocess_bg3.expect("GS preprocess BG3 missing"),
                    &[],
                );
                cpass.dispatch_workgroups(preprocess_workgroups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(
                ctx.get_buffer(cloud.buffers.sort_infos_buf),
                SORT_DISPATCH_INDIRECT_OFFSET,
                ctx.get_buffer(cloud.buffers.sort_dispatch_buf),
                0,
                std::mem::size_of::<[u32; 3]>() as u64,
            );

            let sort_bg_a_to_b = cloud
                .sort_bg_a_to_b
                .expect("GS sort A-to-B bind group missing");
            let sort_bg_b_to_a = cloud
                .sort_bg_b_to_a
                .expect("GS sort B-to-A bind group missing");
            let sort_dispatch = ctx.get_buffer(cloud.buffers.sort_dispatch_buf);
            let sort_internal = ctx.get_buffer(cloud.buffers.sort_internal_buf);
            let histogram_bytes = u64::try_from(
                cloud
                    .buffers
                    .sort_layout
                    .histogram_words
                    .checked_mul(std::mem::size_of::<u32>())
                    .expect("Gaussian sort histogram byte size overflow"),
            )
            .expect("Gaussian sort histogram byte size exceeds u64");

            {
                for (pass_index, radix_pass) in sort_pipelines.radix_passes.iter().enumerate() {
                    let sort_bg = if pass_index % 2 == 0 {
                        sort_bg_a_to_b
                    } else {
                        sort_bg_b_to_a
                    };

                    // The visible count is GPU-generated and can shrink between
                    // frames. Clear the whole fixed-capacity histogram so
                    // inactive workgroups never contribute stale counts.
                    encoder.clear_buffer(sort_internal, 0, Some(histogram_bytes));

                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GS Sort Histogram"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(
                            ctx.pipeline_cache
                                .get_compute_pipeline(radix_pass.histogram),
                        );
                        cpass.set_bind_group(0, sort_bg, &[]);
                        cpass.dispatch_workgroups_indirect(sort_dispatch, 0);
                    }

                    for level in 0..cloud.buffers.sort_layout.scan_level_count {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GS Sort Prefix Scan"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(
                            ctx.pipeline_cache
                                .get_compute_pipeline(sort_pipelines.prefix_scan[level]),
                        );
                        cpass.set_bind_group(0, sort_bg_a_to_b, &[]);
                        cpass.dispatch_workgroups(
                            cloud.buffers.sort_layout.scan_workgroups[level],
                            1,
                            1,
                        );
                    }

                    for level in
                        (0..cloud.buffers.sort_layout.scan_level_count.saturating_sub(1)).rev()
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GS Sort Prefix Add"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(
                            ctx.pipeline_cache
                                .get_compute_pipeline(sort_pipelines.prefix_add[level]),
                        );
                        cpass.set_bind_group(0, sort_bg_a_to_b, &[]);
                        cpass.dispatch_workgroups(
                            cloud.buffers.sort_layout.scan_workgroups[level],
                            1,
                            1,
                        );
                    }

                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GS Sort Scatter"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(
                            ctx.pipeline_cache.get_compute_pipeline(radix_pass.scatter),
                        );
                        cpass.set_bind_group(0, sort_bg, &[]);
                        cpass.dispatch_workgroups_indirect(sort_dispatch, 0);
                    }
                }
            }

            encoder.copy_buffer_to_buffer(
                ctx.get_buffer(cloud.buffers.sort_infos_buf),
                0,
                ctx.get_buffer(cloud.buffers.draw_indirect_buf),
                4,
                4,
            );
        }
    }
}

struct GaussianRenderPassNode<'a> {
    resources: &'a mut GaussianRenderResources,
    pipeline: Option<RenderPipelineId>,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
    clouds: &'a mut [CloudRenderState<'a>],
    color_target: TextureNodeId,
    resolve_target: Option<TextureNodeId>,
    depth_target: TextureNodeId,
}

impl<'a> PassNode<'a> for GaussianRenderPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.pipeline = Some(self.resources.ensure_pipeline(
            ctx,
            self.depth_format,
            self.msaa_samples,
        ));
        let layout = self.resources.layout().clone();
        for cloud in self.clouds.iter_mut() {
            cloud.render_bg = Some(
                ctx.build_bind_group(&layout, Some("GS Render BG"))
                    .bind_buffer(0, cloud.buffers.splat_buf)
                    .bind_buffer(1, cloud.buffers.sort_indices_a_buf)
                    .build(),
            );
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_pipeline = ctx.pipeline_cache.get_render_pipeline(
            self.pipeline
                .expect("GS render pipeline missing after prepare"),
        );

        let color_attachment = ctx
            .get_color_attachment(
                self.color_target,
                RenderTargetOps::Clear(wgpu::Color::TRANSPARENT),
                self.resolve_target,
            )
            .expect("GS color target missing");
        let depth_attachment = ctx.get_depth_stencil_attachment(self.depth_target, 0.0);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GS Render"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(render_pipeline);

        for cloud in self.clouds.iter() {
            rpass.set_bind_group(0, cloud.render_bg.expect("GS render BG missing"), &[]);
            rpass.draw_indirect(ctx.get_buffer(cloud.buffers.draw_indirect_buf), 0);
        }
    }
}

struct GaussianCompositePassNode<'a> {
    resources: &'a mut GaussianCompositeResources,
    pipeline: Option<RenderPipelineId>,
    color_format: wgpu::TextureFormat,
    msaa_samples: u32,
    settings: GpuCompositeSettings,
    accumulation_tex: TextureNodeId,
    composite_settings_buf: BufferNodeId,
    color_target: TextureNodeId,
    composite_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for GaussianCompositePassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.pipeline = Some(self.resources.ensure_pipeline(
            ctx,
            self.color_format,
            self.msaa_samples,
        ));
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.composite_settings_buf),
            0,
            bytemuck::bytes_of(&self.settings),
        );
        let layout = self.resources.layout().clone();
        self.composite_bg = Some(
            ctx.build_bind_group(&layout, Some("GS Composite BG"))
                .bind_texture(0, self.accumulation_tex)
                .bind_common_sampler(1, CommonSampler::NearestClamp)
                .bind_buffer(2, self.composite_settings_buf)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let composite_pipeline = ctx.pipeline_cache.get_render_pipeline(
            self.pipeline
                .expect("GS composite pipeline missing after prepare"),
        );
        let color_attachment = ctx
            .get_color_attachment(self.color_target, RenderTargetOps::Load, None)
            .expect("GS composite color target missing");

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GS Composite"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(composite_pipeline);
        rpass.set_bind_group(0, self.composite_bg.expect("GS composite BG missing"), &[]);
        rpass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HDR_TEXTURE_FORMAT;
    use crate::graph::core::graph::FrameConfig;
    use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};

    fn test_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn test_buffer(
        device: &wgpu::Device,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    async fn request_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(error) => {
                eprintln!("skipping Gaussian sort GPU test: no adapter available ({error})");
                return None;
            }
        };
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GS Sort Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|error| {
                eprintln!("skipping Gaussian sort GPU test: device request failed ({error})");
            })
            .ok()
    }

    #[test]
    fn sort_layout_handles_empty_and_workgroup_boundaries() {
        let empty = SortBufferLayout::for_key_count(0);
        assert_eq!(empty.key_capacity, 1);
        assert_eq!(empty.max_workgroups, 1);
        assert_eq!(empty.histogram_words, SORT_RADIX_SIZE);
        assert_eq!(empty.internal_buffer_words, SORT_RADIX_SIZE + 1);
        assert_eq!(empty.scan_level_count, 1);
        assert_eq!(empty.scan_workgroups[0], 1);

        let exact = SortBufferLayout::for_key_count(SORT_KEYS_PER_WG);
        assert_eq!(exact.key_capacity, SORT_KEYS_PER_WG);
        assert_eq!(exact.max_workgroups, 1);

        let overflow = SortBufferLayout::for_key_count(SORT_KEYS_PER_WG + 1);
        assert_eq!(overflow.max_workgroups, 2);
        assert_eq!(overflow.histogram_words, SORT_RADIX_SIZE * 2);
        assert_eq!(overflow.scan_workgroups[0], 1);
    }

    #[test]
    fn sort_layout_builds_hierarchical_scan_scratch() {
        let layout = SortBufferLayout::for_key_count(310_920);
        assert_eq!(layout.key_capacity, 310_920);
        assert_eq!(layout.max_workgroups, 152);
        assert_eq!(layout.histogram_words, 2_432);
        assert_eq!(layout.scan_level_count, 2);
        assert_eq!(&layout.scan_workgroups[..2], &[5, 1]);
        assert_eq!(layout.internal_buffer_words, 2_438);

        let three_levels = SortBufferLayout::for_key_count((16_384 + 1) * SORT_KEYS_PER_WG);
        assert_eq!(three_levels.scan_level_count, 3);
        assert_eq!(&three_levels.scan_workgroups[..3], &[513, 2, 1]);
    }

    #[test]
    fn even_radix_pass_count_leaves_payload_in_front_buffer() {
        assert_eq!(SORT_PASSES, 8);
        assert_eq!(SORT_PASSES % 2, 0);
    }

    #[test]
    fn portable_shaders_do_not_use_pipeline_overrides() {
        let sort_shader =
            include_str!("../../pipeline/shaders/entry/utility/3dgs/gs_radix_sort.wgsl");
        let preprocess_shader =
            include_str!("../../pipeline/shaders/entry/utility/3dgs/gaussian_preprocess.wgsl");
        for (name, source) in [
            ("portable sort", sort_shader),
            ("Gaussian preprocess", preprocess_shader),
        ] {
            assert!(
                !source
                    .lines()
                    .any(|line| line.trim_start().starts_with("override ")),
                "{name} must not depend on WebGPU pipeline override constants"
            );
        }
    }

    #[test]
    fn portable_gpu_sort_matches_stable_cpu_reference() {
        pollster::block_on(async {
            // More than 32 sort workgroups forces a two-level prefix scan.
            // Repeated buckets are deliberately separated in the input to
            // verify that the scatter remains stable.
            let key_count = 70_001usize;
            let buffer_capacity = 75_003usize;
            let keys: Vec<u32> = (0..key_count)
                .map(|index| {
                    let bucket = u32::try_from(index % 4_096).unwrap();
                    bucket.wrapping_mul(0x9e37_79b9).rotate_left(bucket & 31)
                })
                .collect();
            let payloads: Vec<u32> = (0..u32::try_from(key_count).unwrap()).collect();
            let mut expected = payloads.clone();
            expected.sort_by_key(|&index| keys[index as usize]);

            let layout = SortBufferLayout::for_key_count(buffer_capacity);
            assert_eq!(layout.scan_level_count, 2);

            let Some((device, queue)) = request_test_device().await else {
                return;
            };
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GS Sort Test Layout"),
                    entries: &[
                        test_storage_entry(0, true),
                        test_storage_entry(1, false),
                        test_storage_entry(2, true),
                        test_storage_entry(3, false),
                        test_storage_entry(4, true),
                        test_storage_entry(5, false),
                    ],
                });
            let mut shader_manager = ShaderManager::new();
            let mut pipeline_cache = PipelineCache::new();
            let pipelines = create_sort_pipelines(
                &device,
                &mut shader_manager,
                &mut pipeline_cache,
                &bind_group_layout,
            );

            let infos = GpuSortInfos {
                keys_size: u32::try_from(key_count).unwrap(),
                max_workgroups: u32::try_from(layout.max_workgroups).unwrap(),
                scan_levels: u32::try_from(layout.scan_level_count).unwrap(),
                dispatch_x: u32::try_from(key_count.div_ceil(SORT_KEYS_PER_WG)).unwrap(),
                dispatch_y: 1,
                dispatch_z: 1,
            };
            let infos_buffer = test_buffer(
                &device,
                "GS Sort Test Infos",
                std::mem::size_of::<GpuSortInfos>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let internal_buffer = test_buffer(
                &device,
                "GS Sort Test Scratch",
                (layout.internal_buffer_words * std::mem::size_of::<u32>()) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let data_size = (buffer_capacity * std::mem::size_of::<u32>()) as u64;
            let active_data_size = (key_count * std::mem::size_of::<u32>()) as u64;
            let keys_a = test_buffer(
                &device,
                "GS Sort Test Keys A",
                data_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let keys_b = test_buffer(
                &device,
                "GS Sort Test Keys B",
                data_size,
                wgpu::BufferUsages::STORAGE,
            );
            let payload_a = test_buffer(
                &device,
                "GS Sort Test Payload A",
                data_size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            let payload_b = test_buffer(
                &device,
                "GS Sort Test Payload B",
                data_size,
                wgpu::BufferUsages::STORAGE,
            );
            let readback = test_buffer(
                &device,
                "GS Sort Test Readback",
                active_data_size,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            );

            queue.write_buffer(&infos_buffer, 0, bytemuck::bytes_of(&infos));
            queue.write_buffer(&keys_a, 0, bytemuck::cast_slice(&keys));
            queue.write_buffer(&payload_a, 0, bytemuck::cast_slice(&payloads));

            let bind_group_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GS Sort Test Bind Group A to B"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: infos_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: internal_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: keys_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: payload_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: payload_b.as_entire_binding(),
                    },
                ],
            });
            let bind_group_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GS Sort Test Bind Group B to A"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: infos_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: internal_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: keys_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: payload_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: payload_a.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GS Sort Test Encoder"),
            });
            let histogram_bytes = (layout.histogram_words * std::mem::size_of::<u32>()) as u64;
            let sort_workgroups = u32::try_from(key_count.div_ceil(SORT_KEYS_PER_WG)).unwrap();

            for (pass_index, radix_pass) in pipelines.radix_passes.iter().enumerate() {
                let bind_group = if pass_index % 2 == 0 {
                    &bind_group_a_to_b
                } else {
                    &bind_group_b_to_a
                };
                encoder.clear_buffer(&internal_buffer, 0, Some(histogram_bytes));
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(pipeline_cache.get_compute_pipeline(radix_pass.histogram));
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(sort_workgroups, 1, 1);
                }
                for level in 0..layout.scan_level_count {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(
                        pipeline_cache.get_compute_pipeline(pipelines.prefix_scan[level]),
                    );
                    pass.set_bind_group(0, &bind_group_a_to_b, &[]);
                    pass.dispatch_workgroups(layout.scan_workgroups[level], 1, 1);
                }
                for level in (0..layout.scan_level_count.saturating_sub(1)).rev() {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(
                        pipeline_cache.get_compute_pipeline(pipelines.prefix_add[level]),
                    );
                    pass.set_bind_group(0, &bind_group_a_to_b, &[]);
                    pass.dispatch_workgroups(layout.scan_workgroups[level], 1, 1);
                }
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(pipeline_cache.get_compute_pipeline(radix_pass.scatter));
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(sort_workgroups, 1, 1);
                }
            }
            encoder.copy_buffer_to_buffer(&payload_a, 0, &readback, 0, active_data_size);
            queue.submit([encoder.finish()]);

            let slice = readback.slice(..);
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).ok();
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("Gaussian sort GPU poll failed");
            receiver
                .recv()
                .expect("Gaussian sort readback callback dropped")
                .expect("Gaussian sort readback mapping failed");

            let mapped = slice.get_mapped_range();
            let actual: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
            drop(mapped);
            readback.unmap();

            if let Some((position, (&actual_index, &expected_index))) = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .find(|(_, (actual_index, expected_index))| actual_index != expected_index)
            {
                panic!(
                    "portable GPU sort mismatch at {position}: actual index {actual_index}, expected {expected_index}, actual key {}, expected key {}",
                    keys[actual_index as usize], keys[expected_index as usize]
                );
            }
        });
    }

    struct TestRootNode;

    impl PassNode<'_> for TestRootNode {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    fn tiny_cloud() -> GaussianCloud {
        GaussianCloud {
            gaussians: vec![GaussianSplat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                opacity: 0,
                sh_idx: 0,
                cov: [0; 3],
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

    fn frame_config() -> FrameConfig {
        FrameConfig {
            width: 320,
            height: 180,
            depth_format: wgpu::TextureFormat::Depth32Float,
            msaa_samples: 1,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            hdr_format: HDR_TEXTURE_FORMAT,
        }
    }

    #[test]
    fn new_feature_has_no_pipeline_layout_or_cloud_cache() {
        let feature = GaussianSplattingFeature::new();

        assert!(feature.cache_is_empty());
        assert_eq!(feature.cloud_cache_len(), 0);
        assert_eq!(
            feature.static_cache_identity(),
            GaussianStaticCacheIdentity {
                preprocess_pipeline: None,
                sort_pipelines: None,
                render_pipeline: None,
                composite_pipeline: None,
                layout_ids: [None; 6],
            }
        );
    }

    #[test]
    fn gaussian_pass_nodes_are_no_drop_and_form_three_stage_chain() {
        assert!(!std::mem::needs_drop::<GaussianComputePassNode<'static>>());
        assert!(!std::mem::needs_drop::<GaussianRenderPassNode<'static>>());
        assert!(!std::mem::needs_drop::<GaussianCompositePassNode<'static>>());

        let assets = AssetServer::new();
        let cloud_handle = assets.gaussian_clouds.add(tiny_cloud());
        let sort_config = GaussianSortConfig;
        let plan = GaussianCloudPlan {
            asset_index: 0,
            handle: cloud_handle,
            version: 1,
            num_points: 1,
            num_sh_coefficients: 1,
            sort_layout: GaussianSortConfig::buffer_layout(1),
            sh_degree: 0,
            mip_splatting: false,
            kernel_size: 0.3,
            scene_extent: 1.0,
            color_space_flag: 0,
            opacity_compensation: 1.0,
            model_matrix: Mat4::IDENTITY.to_cols_array(),
        };
        let mut feature = GaussianSplattingFeature::new();

        let (live_gaussian_passes, splat_edge, accumulation_edge) = {
            let mut storage = GraphStorage::new();
            let arena = FrameArena::new();
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let config = frame_config();
            let active_color = graph.register_texture(
                "GS_Test_Color",
                TextureDesc::new_2d(
                    config.width,
                    config.height,
                    HDR_TEXTURE_FORMAT,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                ),
                true,
            );
            let active_depth = graph.register_texture(
                "GS_Test_Depth",
                TextureDesc::new_2d(
                    config.width,
                    config.height,
                    config.depth_format,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
                true,
            );
            let plans = arena.alloc_slice_copy(&[plan]);
            let snapshot = GaussianSnapshot {
                batch: GaussianFrameBatchId::from_index(0),
                plans,
                composite_settings: GpuCompositeSettings { flags: [0; 4] },
                sort_config,
                global_state_key: GlobalStateKey::new(1, 1),
                frame_index: 1,
            };
            let output = {
                let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
                feature.add_to_graph(&mut ctx, snapshot, active_color, active_depth)
            };
            graph.add_pass("GS_Test_Root", |builder| {
                builder.read_texture(output);
                builder.mark_side_effect();
                (TestRootNode, ())
            });
            graph.compile_topology();

            let live_gaussian_passes = graph
                .storage
                .execution_queue
                .iter()
                .map(|&index| graph.storage.passes[index].name)
                .filter(|name| name.starts_with("GS_"))
                .filter(|&name| name != "GS_Test_Root")
                .collect::<Vec<_>>();
            let edge = |resource_name: &str| {
                let resource = graph
                    .storage
                    .resources
                    .iter()
                    .find(|resource| resource.name == resource_name)
                    .unwrap_or_else(|| panic!("Gaussian resource '{resource_name}' missing"));
                let producer = resource
                    .producer
                    .map(|index| graph.storage.passes[index].name);
                let consumers = resource
                    .consumers
                    .iter()
                    .map(|&index| graph.storage.passes[index].name)
                    .collect::<Vec<_>>();
                (producer, consumers)
            };
            (
                live_gaussian_passes,
                edge("GS_Splats"),
                edge("GS_Accumulation"),
            )
        };

        assert_eq!(
            live_gaussian_passes,
            ["GS_Compute", "GS_Render", "GS_Composite"]
        );
        assert_eq!(splat_edge.0, Some("GS_Compute"));
        assert_eq!(splat_edge.1, ["GS_Render"]);
        assert_eq!(accumulation_edge.0, Some("GS_Render"));
        assert_eq!(accumulation_edge.1, ["GS_Composite"]);
        assert!(
            feature.cache_is_empty(),
            "graph construction must not initialize any physical Gaussian cache"
        );
    }
}
