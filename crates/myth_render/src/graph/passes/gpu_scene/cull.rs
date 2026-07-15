//! DPE-live GPU Scene frustum culling and indexed-indirect argument emission.
//!
//! Graph construction declares only the persistent instance-table read and the
//! two immutable outputs consumed by a later scene submission. A live node
//! creates its pipeline and bind group during `prepare`; `execute` first clears
//! the complete indirect argument record and then records only the compute
//! dispatch. This makes zero-visible and zero-slot domains valid no-op draws
//! without a CPU readback or a cross-workgroup initialization race.

use std::{
    mem::{size_of, size_of_val},
    sync::OnceLock,
};

use bytemuck::{Pod, Zeroable};
#[cfg(test)]
use smallvec::SmallVec;

use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::Tracked;
use crate::core::gpu::binding_layout::{storage_buffer, uniform_buffer};
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, PassNode, PrepareContext, RenderGraph,
};
use crate::graph::gpu_scene::{GpuSceneActiveBatchDescriptor, GpuSceneActiveBatchSet};
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};

pub(crate) const GPU_SCENE_CULL_WORKGROUP_SIZE: u32 = 64;
pub(crate) const GPU_SCENE_INDEXED_ARGUMENT_SIZE: u64 = 5 * size_of::<u32>() as u64;
pub(crate) const GPU_SCENE_VISIBLE_INSTANCE_SIZE: u64 = size_of::<GpuSceneVisibleInstance>() as u64;

/// Frozen M7.4 persistent batch-table ABI. The authority owns the Rust POD;
/// Cull only needs its byte stride when validating logical table capacity.
#[cfg(test)]
pub(crate) const GPU_SCENE_BATCH_RECORD_SIZE: u64 = 80;

const GPU_SCENE_CULL_SHADER: &str = "entry/utility/gpu_scene/cull";

/// One generation-stamped entry in the compact visible-instance stream.
///
/// Retaining the generation beside the stable slot lets the later vertex path
/// fail closed if a future candidate indirection outlives slot retirement.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct GpuSceneVisibleInstance {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
}

/// Exact 128-byte CPU/WGSL ABI uploaded only for a DPE-live Cull node.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[cfg(test)]
struct GpuSceneCullParams {
    frustum_planes: [[f32; 4]; 6],
    /// `(active_scene_id, slot_span, index_count, first_index)`.
    domain: [u32; 4],
    /// `(base_vertex, required_instance_flags bits, visible_base bits, reserved)`.
    draw: [i32; 4],
}

#[cfg(test)]
const _: () = assert!(size_of::<GpuSceneCullParams>() == 128);
const _: () = assert!(size_of::<GpuSceneVisibleInstance>() == 8);
const _: () = assert!(GPU_SCENE_INDEXED_ARGUMENT_SIZE == 20);

/// One compact, live batch identity supplied by the CPU authority.
///
/// Partition offsets are relative to one view's visible stream. Both offset
/// and size obey the device storage-buffer offset alignment, allowing the draw
/// backend to bind the exact partition while keeping `first_instance == 0`.
/// Compatibility name for the authority-owned retained descriptor. This is a
/// true alias, not a conversion layer: Composer passes the authority slice
/// directly into Cull without an `O(bin)` temporary allocation or copy.
pub(crate) type GpuSceneCullBinDescriptor = GpuSceneActiveBatchDescriptor;

/// One immutable view family consumed by a single multi-bin dispatch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuSceneCullViewDescriptor {
    pub(crate) frustum_planes: [[f32; 4]; 6],
}

impl GpuSceneCullViewDescriptor {
    #[must_use]
    pub(crate) const fn new(frustum_planes: [[f32; 4]; 6]) -> Self {
        Self { frustum_planes }
    }
}

/// Logical M7.4 Cull recipe. No physical WGPU owner crosses graph build.
///
/// `visible_view_stride` is the packed size of all batch partitions for one
/// view, including alignment padding. Consequently allocation is
/// `O(sum(partition capacities) * views)`, never `slot_span * bin_count`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuSceneMultiBinCullDescriptor<'a> {
    pub(crate) active_scene_id: u32,
    pub(crate) instance_slot_span: u32,
    pub(crate) batch_slot_span: u32,
    pub(crate) required_instance_flags: u32,
    pub(crate) storage_alignment: u64,
    pub(crate) visible_view_stride: u64,
    pub(crate) views: &'a [GpuSceneCullViewDescriptor],
    pub(crate) bins: GpuSceneActiveBatchSet<'a>,
}

impl<'a> GpuSceneMultiBinCullDescriptor<'a> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub(crate) const fn new(
        active_scene_id: u32,
        instance_slot_span: u32,
        batch_slot_span: u32,
        storage_alignment: u64,
        visible_view_stride: u64,
        views: &'a [GpuSceneCullViewDescriptor],
        bins: GpuSceneActiveBatchSet<'a>,
    ) -> Self {
        Self {
            active_scene_id,
            instance_slot_span,
            batch_slot_span,
            required_instance_flags: 0,
            storage_alignment,
            visible_view_stride,
            views,
            bins,
        }
    }

    /// Build an intentionally untrusted descriptor for validation and shader
    /// contract tests. Production callers must supply an authority-certified
    /// batch set through [`Self::new`].
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub(crate) const fn new_untrusted(
        active_scene_id: u32,
        instance_slot_span: u32,
        batch_slot_span: u32,
        storage_alignment: u64,
        visible_view_stride: u64,
        views: &'a [GpuSceneCullViewDescriptor],
        bins: &'a [GpuSceneCullBinDescriptor],
    ) -> Self {
        Self::new(
            active_scene_id,
            instance_slot_span,
            batch_slot_span,
            storage_alignment,
            visible_view_stride,
            views,
            GpuSceneActiveBatchSet::untrusted(bins),
        )
    }

    /// Require every emitted instance to contain all supplied metadata flags.
    #[must_use]
    pub(crate) const fn with_required_instance_flags(mut self, flags: u32) -> Self {
        self.required_instance_flags = flags;
        self
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuSceneMultiBinCullDomain {
    /// `(scene_id, instance_slot_span, batch_slot_span, active_batch_count)`.
    domain: [u32; 4],
    /// `(view_count, visible_view_stride_records, required_flags, dispatch_items)`.
    layout: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuSceneMultiBinCullView {
    frustum_planes: [[f32; 4]; 6],
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuSceneMultiBinCullActiveBatch {
    /// `(stable slot, generation, reserved, reserved)`.
    id: [u32; 4],
}

const _: () = assert!(size_of::<GpuSceneMultiBinCullDomain>() == 32);
const _: () = assert!(size_of::<GpuSceneMultiBinCullView>() == 96);
const _: () = assert!(size_of::<GpuSceneMultiBinCullActiveBatch>() == 16);

/// Copy-only graph-build recipe for one constrained rigid indexed batch.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuSceneCullDescriptor {
    active_scene_id: u32,
    slot_span: u32,
    frustum_planes: [[f32; 4]; 6],
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    required_instance_flags: u32,
}

#[cfg(test)]
impl GpuSceneCullDescriptor {
    #[must_use]
    pub(crate) const fn new(
        active_scene_id: u32,
        slot_span: u32,
        frustum_planes: [[f32; 4]; 6],
        index_count: u32,
        first_index: u32,
        base_vertex: i32,
    ) -> Self {
        Self {
            active_scene_id,
            slot_span,
            frustum_planes,
            index_count,
            first_index,
            base_vertex,
            required_instance_flags: 0,
        }
    }

    /// Require every emitted instance to contain all supplied metadata flags.
    /// A zero mask preserves the main-view single-batch behavior.
    #[inline]
    #[must_use]
    pub(crate) const fn with_required_instance_flags(mut self, flags: u32) -> Self {
        self.required_instance_flags = flags;
        self
    }

    #[inline]
    #[must_use]
    const fn params(self, visible_base: u32) -> GpuSceneCullParams {
        GpuSceneCullParams {
            frustum_planes: self.frustum_planes,
            domain: [
                self.active_scene_id,
                self.slot_span,
                self.index_count,
                self.first_index,
            ],
            draw: [
                self.base_vertex,
                self.required_instance_flags.cast_signed(),
                visible_base.cast_signed(),
                0,
            ],
        }
    }
}

/// Ordinary RDG outputs consumed by the immutable indirect submission.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuSceneCullOutput {
    pub(crate) visible_instances: BufferNodeId,
    pub(crate) indirect_arguments: BufferNodeId,
}

/// One immutable slice within a packed multi-view Cull output.
///
/// `first_instance` is written into the indexed-indirect argument record, so
/// the vertex shader's `@builtin(instance_index)` directly addresses this
/// view's compact visible partition without another per-view binding.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSceneCullViewOutput {
    pub(crate) visible_instances: BufferNodeId,
    pub(crate) indirect_arguments: BufferNodeId,
    pub(crate) indirect_offset: u64,
    pub(crate) first_instance: u32,
}

/// Shared Buffer outputs for one packed family of Cull views.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuSceneCullBatchOutput {
    pub(crate) visible_instances: BufferNodeId,
    pub(crate) indirect_arguments: BufferNodeId,
    view_count: u32,
    visible_stride: u32,
}

/// Exact immutable draw slice produced for one `(view, batch)` pair.
///
/// The visible buffer is intentionally shared, but consumers bind only this
/// byte range. `first_instance` is always zero: the binding offset, rather
/// than `INDIRECT_FIRST_INSTANCE`, selects the batch partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSceneMultiBinDrawOutput {
    pub(crate) visible_instances: BufferNodeId,
    pub(crate) visible_offset: u64,
    pub(crate) visible_size: u64,
    pub(crate) indirect_arguments: BufferNodeId,
    pub(crate) indirect_offset: u64,
    pub(crate) first_instance: u32,
    pub(crate) batch_slot: u32,
    pub(crate) batch_generation: u32,
}

/// Shared outputs of one single-dispatch, multi-view, multi-bin Cull family.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuSceneMultiBinCullOutput<'a> {
    pub(crate) visible_instances: BufferNodeId,
    pub(crate) indirect_arguments: BufferNodeId,
    view_count: u32,
    batch_slot_span: u32,
    visible_view_stride: u64,
    bins: &'a [GpuSceneCullBinDescriptor],
}

impl GpuSceneMultiBinCullOutput<'_> {
    #[must_use]
    pub(crate) const fn view_count(self) -> u32 {
        self.view_count
    }

    #[must_use]
    pub(crate) fn batch_count(self) -> u32 {
        u32::try_from(self.bins.len()).expect("validated GPU Scene bin count exceeds u32")
    }

    /// Resolve an exact partition without allocating a per-view descriptor
    /// table. The caller iterates the compact batch list, never sparse slots.
    #[must_use]
    pub(crate) fn view_batch(
        self,
        view_index: u32,
        compact_batch_index: u32,
    ) -> Option<GpuSceneMultiBinDrawOutput> {
        if view_index >= self.view_count {
            return None;
        }
        let bin = *self.bins.get(compact_batch_index as usize)?;
        let visible_offset = u64::from(view_index)
            .checked_mul(self.visible_view_stride)?
            .checked_add(bin.partition_base)?;
        let argument_index = u64::from(view_index)
            .checked_mul(u64::from(self.batch_slot_span))?
            .checked_add(u64::from(bin.id.slot()))?;
        Some(GpuSceneMultiBinDrawOutput {
            visible_instances: self.visible_instances,
            visible_offset,
            visible_size: bin.partition_size,
            indirect_arguments: self.indirect_arguments,
            indirect_offset: argument_index.checked_mul(GPU_SCENE_INDEXED_ARGUMENT_SIZE)?,
            first_instance: 0,
            batch_slot: bin.id.slot(),
            batch_generation: bin.id.generation(),
        })
    }
}

#[cfg(test)]
impl GpuSceneCullBatchOutput {
    const fn view_count(&self) -> u32 {
        self.view_count
    }
}

#[cfg(test)]
impl GpuSceneCullBatchOutput {
    /// Resolve one per-view immutable indirect slice.
    #[must_use]
    pub(crate) fn view(self, index: u32) -> Option<GpuSceneCullViewOutput> {
        if index >= self.view_count {
            return None;
        }
        Some(GpuSceneCullViewOutput {
            visible_instances: self.visible_instances,
            indirect_arguments: self.indirect_arguments,
            indirect_offset: u64::from(index) * GPU_SCENE_INDEXED_ARGUMENT_SIZE,
            first_instance: index.checked_mul(self.visible_stride)?,
        })
    }
}

/// Cross-frame static Cull resources. No scene identity enters this cache.
#[derive(Default)]
pub(crate) struct GpuSceneCullFeature {
    #[cfg(test)]
    single_view: OnceLock<GpuSceneCullStaticResources>,
    #[cfg(test)]
    multi_view: OnceLock<GpuSceneCullStaticResources>,
    multi_bin: OnceLock<GpuSceneCullStaticResources>,
}

struct GpuSceneCullStaticResources {
    layout: Tracked<wgpu::BindGroupLayout>,
    pipeline: ComputePipelineId,
}

impl GpuSceneCullFeature {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add the Cull producer. Neither output roots the branch by itself.
    #[cfg(test)]
    pub(crate) fn add_to_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        instances: BufferNodeId,
        descriptor: GpuSceneCullDescriptor,
    ) -> GpuSceneCullOutput {
        let params = graph.alloc_slice(&[descriptor.params(0)]);
        let batch = self.add_cull_pass(
            graph,
            instances,
            "GpuScene_Cull",
            params,
            descriptor.slot_span.max(1),
            GpuSceneCullMode::SingleView,
        );
        GpuSceneCullOutput {
            visible_instances: batch.visible_instances,
            indirect_arguments: batch.indirect_arguments,
        }
    }

    /// Add one packed family Cull. Descriptors may differ only by frustum;
    /// every view addresses the same scene/table and constrained draw state.
    /// Neither the shared outputs nor their per-view slices root the branch.
    #[cfg(test)]
    pub(crate) fn add_batch_to_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        instances: BufferNodeId,
        pass_name: &'static str,
        descriptors: &[GpuSceneCullDescriptor],
    ) -> GpuSceneCullBatchOutput {
        let first = *descriptors
            .first()
            .expect("packed GPU Scene Cull requires at least one view");
        assert!(
            descriptors.iter().all(|descriptor| {
                descriptor.active_scene_id == first.active_scene_id
                    && descriptor.slot_span == first.slot_span
                    && descriptor.index_count == first.index_count
                    && descriptor.first_index == first.first_index
                    && descriptor.base_vertex == first.base_vertex
                    && descriptor.required_instance_flags == first.required_instance_flags
            }),
            "one packed GPU Scene Cull family used conflicting fixed state"
        );

        let view_count = u32::try_from(descriptors.len())
            .expect("one packed GPU Scene Cull cannot contain more than u32::MAX views");
        let visible_stride = first.slot_span.max(1);
        view_count
            .checked_mul(visible_stride)
            .expect("packed GPU Scene visible-instance index space overflowed");

        let mut params = SmallVec::<[GpuSceneCullParams; 16]>::new();
        params.reserve(descriptors.len());
        for (view_index, descriptor) in descriptors.iter().copied().enumerate() {
            let view_index = u32::try_from(view_index)
                .expect("validated packed GPU Scene view index exceeds u32");
            let visible_base = view_index
                .checked_mul(visible_stride)
                .expect("packed GPU Scene visible-instance base overflowed");
            params.push(descriptor.params(visible_base));
        }
        let params = graph.alloc_slice(&params);
        self.add_cull_pass(
            graph,
            instances,
            pass_name,
            params,
            visible_stride,
            GpuSceneCullMode::MultiView { view_count },
        )
    }

    /// Add one M7.4 multi-bin Cull producer for a complete view family.
    ///
    /// Exactly one RDG node and one compute dispatch are emitted. Instance and
    /// batch identities remain logical persistent reads; all small parameters
    /// are uploaded only if DPE keeps the node alive.
    pub(crate) fn add_multi_bin_to_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        instances: BufferNodeId,
        batches: BufferNodeId,
        pass_name: &'static str,
        descriptor: GpuSceneMultiBinCullDescriptor<'a>,
    ) -> GpuSceneMultiBinCullOutput<'a> {
        validate_multi_bin_descriptor(descriptor);

        let view_count = u32::try_from(descriptor.views.len())
            .expect("validated GPU Scene view count exceeds u32");
        let active_batch_count = u32::try_from(descriptor.bins.len())
            .expect("validated GPU Scene batch count exceeds u32");
        let visible_view_stride_records =
            u32::try_from(descriptor.visible_view_stride / GPU_SCENE_VISIBLE_INSTANCE_SIZE)
                .expect("GPU Scene visible view stride exceeds WGSL u32 indexing");
        let dispatch_items = descriptor.instance_slot_span.max(active_batch_count);
        let domain = [GpuSceneMultiBinCullDomain {
            domain: [
                descriptor.active_scene_id,
                descriptor.instance_slot_span,
                descriptor.batch_slot_span,
                active_batch_count,
            ],
            layout: [
                view_count,
                visible_view_stride_records,
                descriptor.required_instance_flags,
                dispatch_items,
            ],
        }];
        let domain = graph.alloc_slice(&domain);

        let views =
            graph.alloc_slice_fill_with(descriptor.views.len(), |index| GpuSceneMultiBinCullView {
                frustum_planes: descriptor.views[index].frustum_planes,
            });

        let active_batches = graph.alloc_slice_fill_with(descriptor.bins.len(), |index| {
            let bin = descriptor.bins[index];
            GpuSceneMultiBinCullActiveBatch {
                id: [bin.id.slot(), bin.id.generation(), 0, 0],
            }
        });

        let visible_size = u64::from(view_count)
            .checked_mul(descriptor.visible_view_stride)
            .expect("GPU Scene multi-bin visible buffer size overflowed");
        let argument_records = u64::from(view_count)
            .checked_mul(u64::from(descriptor.batch_slot_span))
            .expect("GPU Scene multi-bin argument record count overflowed");
        let argument_size = argument_records
            .checked_mul(GPU_SCENE_INDEXED_ARGUMENT_SIZE)
            .expect("GPU Scene multi-bin argument buffer size overflowed");

        graph.add_pass(pass_name, |builder| {
            builder.read_buffer(instances);
            builder.read_buffer(batches);
            let visible_instances = builder.create_buffer(
                "GpuScene_MultiBin_Visible_Instances",
                BufferDesc::new(visible_size, wgpu::BufferUsages::STORAGE),
            );
            let indirect_arguments = builder.create_buffer(
                "GpuScene_MultiBin_Indexed_Indirect_Arguments",
                BufferDesc::new(
                    argument_size,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST,
                ),
            );
            let active_batch_buffer = builder.create_buffer(
                "GpuScene_MultiBin_Active_Batches",
                BufferDesc::new(
                    u64::try_from(size_of_val(active_batches))
                        .expect("GPU Scene active-batch bytes exceed u64"),
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                ),
            );
            let view_buffer = builder.create_buffer(
                "GpuScene_MultiBin_Cull_Views",
                BufferDesc::new(
                    u64::try_from(size_of_val(views))
                        .expect("GPU Scene Cull view bytes exceed u64"),
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                ),
            );
            let domain_buffer = builder.create_buffer(
                "GpuScene_MultiBin_Cull_Domain",
                BufferDesc::new(
                    size_of::<GpuSceneMultiBinCullDomain>() as u64,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
            );

            (
                GpuSceneMultiBinCullNode {
                    feature: self,
                    instances,
                    batches,
                    active_batch_buffer,
                    visible_instances,
                    indirect_arguments,
                    view_buffer,
                    domain_buffer,
                    active_batches,
                    views,
                    domain,
                    argument_size,
                    dispatch_items,
                    view_count,
                    prepared: None,
                },
                GpuSceneMultiBinCullOutput {
                    visible_instances,
                    indirect_arguments,
                    view_count,
                    batch_slot_span: descriptor.batch_slot_span,
                    visible_view_stride: descriptor.visible_view_stride,
                    bins: descriptor.bins.as_slice(),
                },
            )
        })
    }

    #[cfg(test)]
    fn add_cull_pass<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        instances: BufferNodeId,
        pass_name: &'static str,
        params: &'a [GpuSceneCullParams],
        visible_stride: u32,
        mode: GpuSceneCullMode,
    ) -> GpuSceneCullBatchOutput {
        let view_count = mode.view_count();
        debug_assert_eq!(params.len(), view_count as usize);
        let visible_records = u64::from(view_count)
            .checked_mul(u64::from(visible_stride))
            .expect("packed GPU Scene visible-instance count overflowed");
        let visible_size = visible_records
            .checked_mul(GPU_SCENE_VISIBLE_INSTANCE_SIZE)
            .expect("packed GPU Scene visible-instance buffer size overflowed");
        let argument_size = u64::from(view_count)
            .checked_mul(GPU_SCENE_INDEXED_ARGUMENT_SIZE)
            .expect("packed GPU Scene argument buffer size overflowed");
        let params_size = u64::try_from(params.len())
            .expect("packed GPU Scene parameter count exceeds u64")
            .checked_mul(size_of::<GpuSceneCullParams>() as u64)
            .expect("packed GPU Scene parameter buffer size overflowed");
        let params_usage = if mode.is_multi_view() {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        } else {
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
        };

        graph.add_pass(pass_name, |builder| {
            builder.read_buffer(instances);
            let visible_instances = builder.create_buffer(
                "GpuScene_Visible_Instances",
                BufferDesc::new(visible_size, wgpu::BufferUsages::STORAGE),
            );
            let indirect_arguments = builder.create_buffer(
                "GpuScene_Indexed_Indirect_Arguments",
                BufferDesc::new(
                    argument_size,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST,
                ),
            );
            let params_buffer = builder.create_buffer(
                "GpuScene_Cull_Params",
                BufferDesc::new(params_size, params_usage),
            );

            (
                GpuSceneCullNode {
                    feature: self,
                    instances,
                    visible_instances,
                    indirect_arguments,
                    params_buffer,
                    params,
                    argument_size,
                    mode,
                    prepared: None,
                },
                GpuSceneCullBatchOutput {
                    visible_instances,
                    indirect_arguments,
                    view_count,
                    visible_stride,
                },
            )
        })
    }

    #[cfg(test)]
    fn resources<'a>(
        &'a self,
        ctx: &mut PrepareContext<'_>,
        mode: GpuSceneCullMode,
    ) -> &'a GpuSceneCullStaticResources {
        let cell = if mode.is_multi_view() {
            &self.multi_view
        } else {
            &self.single_view
        };
        cell.get_or_init(|| Self::create_resources(ctx, mode.is_multi_view()))
    }

    #[cfg(test)]
    fn create_resources(
        ctx: &mut PrepareContext<'_>,
        multi_view: bool,
    ) -> GpuSceneCullStaticResources {
        let mut shader_options = ShaderCompilationOptions::default();
        if multi_view {
            shader_options.add_define("GPU_SCENE_MULTI_VIEW", "1");
        }

        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File(GPU_SCENE_CULL_SHADER),
            &shader_options,
        );
        let key =
            ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options);

        let (pipeline, layout) =
            if let Some(pipeline) = ctx.pipeline_cache.get_compute_pipeline_id(&key) {
                (
                    pipeline,
                    ctx.pipeline_cache
                        .get_tracked_compute_layout(pipeline, 0)
                        .clone(),
                )
            } else {
                let layout = create_gpu_scene_cull_layout(ctx.device, multi_view);
                let pipeline_layout =
                    ctx.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("GpuScene Cull Pipeline Layout"),
                            bind_group_layouts: &[Some(&layout)],
                            immediate_size: 0,
                        });
                let pipeline = ctx.pipeline_cache.get_or_create_compute(
                    ctx.device,
                    module,
                    &pipeline_layout,
                    &key,
                    &compilation_options,
                    "GpuScene Cull Pipeline",
                );
                ctx.pipeline_cache
                    .register_compute_layouts(pipeline, smallvec::smallvec![layout.clone()]);
                (pipeline, layout)
            };

        GpuSceneCullStaticResources { layout, pipeline }
    }

    fn multi_bin_resources<'a>(
        &'a self,
        ctx: &mut PrepareContext<'_>,
    ) -> &'a GpuSceneCullStaticResources {
        self.multi_bin
            .get_or_init(|| Self::create_multi_bin_resources(ctx))
    }

    fn create_multi_bin_resources(ctx: &mut PrepareContext<'_>) -> GpuSceneCullStaticResources {
        let mut shader_options = ShaderCompilationOptions::default();
        shader_options.add_define("GPU_SCENE_MULTI_BIN", "1");
        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File(GPU_SCENE_CULL_SHADER),
            &shader_options,
        );
        let key =
            ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options);
        let (pipeline, layout) =
            if let Some(pipeline) = ctx.pipeline_cache.get_compute_pipeline_id(&key) {
                (
                    pipeline,
                    ctx.pipeline_cache
                        .get_tracked_compute_layout(pipeline, 0)
                        .clone(),
                )
            } else {
                let layout = create_gpu_scene_multi_bin_cull_layout(ctx.device);
                let pipeline_layout =
                    ctx.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("GpuScene Multi-Bin Cull Pipeline Layout"),
                            bind_group_layouts: &[Some(&layout)],
                            immediate_size: 0,
                        });
                let pipeline = ctx.pipeline_cache.get_or_create_compute(
                    ctx.device,
                    module,
                    &pipeline_layout,
                    &key,
                    &compilation_options,
                    "GpuScene Multi-Bin Cull Pipeline",
                );
                ctx.pipeline_cache
                    .register_compute_layouts(pipeline, smallvec::smallvec![layout.clone()]);
                (pipeline, layout)
            };
        GpuSceneCullStaticResources { layout, pipeline }
    }
}

fn validate_multi_bin_descriptor(descriptor: GpuSceneMultiBinCullDescriptor<'_>) {
    assert!(
        !descriptor.views.is_empty(),
        "GPU Scene Cull requires a view"
    );
    assert!(
        !descriptor.bins.is_empty(),
        "GPU Scene Cull requires a live batch"
    );
    assert_ne!(
        descriptor.batch_slot_span, 0,
        "GPU Scene batch span is zero"
    );
    let view_count = u32::try_from(descriptor.views.len())
        .expect("GPU Scene Cull view count exceeds WGSL u32 indexing");
    u32::try_from(descriptor.bins.len())
        .expect("GPU Scene Cull active-batch count exceeds WGSL u32 indexing");
    assert!(
        descriptor.storage_alignment >= GPU_SCENE_VISIBLE_INSTANCE_SIZE
            && descriptor.storage_alignment.is_power_of_two()
            && descriptor
                .storage_alignment
                .is_multiple_of(GPU_SCENE_VISIBLE_INSTANCE_SIZE),
        "GPU Scene visible alignment is incompatible with its record ABI"
    );
    assert!(
        descriptor.visible_view_stride != 0
            && descriptor
                .visible_view_stride
                .is_multiple_of(descriptor.storage_alignment),
        "GPU Scene visible view stride is not storage-aligned"
    );
    assert!(
        u32::try_from(descriptor.visible_view_stride).is_ok(),
        "GPU Scene visible view stride exceeds portable WGSL byte metadata"
    );
    let visible_view_stride_records =
        u32::try_from(descriptor.visible_view_stride / GPU_SCENE_VISIBLE_INSTANCE_SIZE)
            .expect("GPU Scene visible view stride exceeds WGSL u32 record indexing");
    view_count
        .checked_mul(visible_view_stride_records)
        .expect("GPU Scene visible view-major index space exceeds WGSL u32 indexing");
    view_count
        .checked_mul(descriptor.batch_slot_span)
        .expect("GPU Scene argument view-major index space exceeds WGSL u32 indexing");

    for bin in descriptor.bins.iter() {
        assert_ne!(bin.id.generation(), 0, "GPU Scene batch generation is zero");
        assert!(
            bin.id.slot() < descriptor.batch_slot_span,
            "GPU Scene active batch lies outside the sparse slot span"
        );
        assert!(
            bin.partition_base % descriptor.storage_alignment == 0
                && bin.partition_size != 0
                && bin.partition_size % descriptor.storage_alignment == 0,
            "GPU Scene visible partition is not storage-aligned"
        );
        let end = bin
            .partition_base
            .checked_add(bin.partition_size)
            .expect("GPU Scene visible partition overflowed");
        assert!(
            end <= descriptor.visible_view_stride,
            "GPU Scene visible partition exceeds one view stride"
        );
        assert!(
            u32::try_from(bin.partition_base).is_ok() && u32::try_from(bin.partition_size).is_ok(),
            "GPU Scene visible partition exceeds portable WGSL byte metadata"
        );
    }

    #[cfg(test)]
    if !descriptor.bins.is_authority_certified() {
        validate_untrusted_batch_set(descriptor.bins.as_slice());
    }
}

/// Keep adversarial duplicate/overlap coverage without charging the certified
/// production path an `O(batch_count^2)` scan for every view family.
#[cfg(test)]
fn validate_untrusted_batch_set(bins: &[GpuSceneCullBinDescriptor]) {
    for (index, bin) in bins.iter().enumerate() {
        for other in &bins[index + 1..] {
            assert_ne!(
                bin.id.slot(),
                other.id.slot(),
                "GPU Scene active batch slot was duplicated"
            );
            let bin_end = bin.partition_base + bin.partition_size;
            let other_end = other.partition_base + other.partition_size;
            assert!(
                bin_end <= other.partition_base || other_end <= bin.partition_base,
                "GPU Scene visible partitions overlap"
            );
        }
    }
}

#[cfg(test)]
fn create_gpu_scene_cull_layout(
    device: &wgpu::Device,
    multi_view: bool,
) -> Tracked<wgpu::BindGroupLayout> {
    let params = if multi_view {
        storage_buffer(3, wgpu::ShaderStages::COMPUTE, true)
    } else {
        uniform_buffer(3, wgpu::ShaderStages::COMPUTE)
    };
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuScene Cull Bind Group Layout"),
            entries: &[
                storage_buffer(0, wgpu::ShaderStages::COMPUTE, true),
                storage_buffer(1, wgpu::ShaderStages::COMPUTE, false),
                storage_buffer(2, wgpu::ShaderStages::COMPUTE, false),
                params,
            ],
        }),
    )
}

fn create_gpu_scene_multi_bin_cull_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuScene Multi-Bin Cull Bind Group Layout"),
            entries: &[
                storage_buffer(0, wgpu::ShaderStages::COMPUTE, true),
                storage_buffer(1, wgpu::ShaderStages::COMPUTE, true),
                storage_buffer(2, wgpu::ShaderStages::COMPUTE, true),
                storage_buffer(3, wgpu::ShaderStages::COMPUTE, false),
                storage_buffer(4, wgpu::ShaderStages::COMPUTE, false),
                storage_buffer(5, wgpu::ShaderStages::COMPUTE, true),
                uniform_buffer(6, wgpu::ShaderStages::COMPUTE),
            ],
        }),
    )
}

#[derive(Debug, Clone, Copy)]
#[cfg(test)]
enum GpuSceneCullMode {
    SingleView,
    MultiView { view_count: u32 },
}

#[cfg(test)]
impl GpuSceneCullMode {
    #[inline]
    const fn view_count(self) -> u32 {
        match self {
            Self::SingleView => 1,
            Self::MultiView { view_count } => view_count,
        }
    }

    #[inline]
    const fn is_multi_view(self) -> bool {
        matches!(self, Self::MultiView { .. })
    }
}

#[cfg(test)]
struct GpuSceneCullNode<'a> {
    feature: &'a GpuSceneCullFeature,
    instances: BufferNodeId,
    visible_instances: BufferNodeId,
    indirect_arguments: BufferNodeId,
    params_buffer: BufferNodeId,
    params: &'a [GpuSceneCullParams],
    argument_size: u64,
    mode: GpuSceneCullMode,
    prepared: Option<PreparedGpuSceneCull>,
}

struct GpuSceneMultiBinCullNode<'a> {
    feature: &'a GpuSceneCullFeature,
    instances: BufferNodeId,
    batches: BufferNodeId,
    active_batch_buffer: BufferNodeId,
    visible_instances: BufferNodeId,
    indirect_arguments: BufferNodeId,
    view_buffer: BufferNodeId,
    domain_buffer: BufferNodeId,
    active_batches: &'a [GpuSceneMultiBinCullActiveBatch],
    views: &'a [GpuSceneMultiBinCullView],
    domain: &'a [GpuSceneMultiBinCullDomain],
    argument_size: u64,
    dispatch_items: u32,
    view_count: u32,
    prepared: Option<PreparedGpuSceneCull>,
}

#[derive(Debug, Clone, Copy)]
enum PreparedGpuSceneCull {
    #[allow(dead_code, reason = "legacy single-bin test path")]
    Empty,
    Dispatch {
        pipeline: ComputePipelineId,
        bind_group: PreparedBindGroupId,
        workgroups_x: u32,
        workgroups_y: u32,
    },
}

#[cfg(test)]
impl<'a> PassNode<'a> for GpuSceneCullNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        assert!(
            self.prepared.is_none(),
            "GpuScene Cull was prepared more than once"
        );

        // An empty table still has a live indirect-output contract. Execute
        // clears the argument record, but no shader, pipeline, BG, or upload is
        // required to prove the resulting zero-instance draw.
        let slot_span = self.params[0].domain[1];
        if slot_span == 0 {
            self.prepared = Some(PreparedGpuSceneCull::Empty);
            return;
        }

        let workgroups_x = slot_span.div_ceil(GPU_SCENE_CULL_WORKGROUP_SIZE);
        let workgroups_y = self.mode.view_count();
        assert!(
            workgroups_x <= ctx.device.limits().max_compute_workgroups_per_dimension
                && workgroups_y <= ctx.device.limits().max_compute_workgroups_per_dimension,
            "GpuScene Cull slot span exceeds max_compute_workgroups_per_dimension"
        );

        let resources = self.feature.resources(ctx, self.mode);
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.params_buffer),
            0,
            bytemuck::cast_slice(self.params),
        );
        let bind_group = ctx
            .build_bind_group(&resources.layout, Some("GpuScene Cull Bind Group"))
            .bind_buffer(0, self.instances)
            .bind_buffer(1, self.visible_instances)
            .bind_buffer(2, self.indirect_arguments)
            .bind_buffer(3, self.params_buffer)
            .build();
        self.prepared = Some(PreparedGpuSceneCull::Dispatch {
            pipeline: resources.pipeline,
            bind_group,
            workgroups_x,
            workgroups_y,
        });
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let prepared = self
            .prepared
            .expect("live GpuScene Cull reached execute without preparation");

        // This clear is deliberately encoded before the dispatch. Initializing
        // instance_count from invocation zero would race atomic increments from
        // every other workgroup because WGSL has no device-wide workgroup barrier.
        encoder.clear_buffer(
            ctx.get_buffer(self.indirect_arguments),
            0,
            Some(self.argument_size),
        );

        let PreparedGpuSceneCull::Dispatch {
            pipeline,
            bind_group,
            workgroups_x,
            workgroups_y,
        } = prepared
        else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GpuScene Cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ctx.pipeline_cache.get_compute_pipeline(pipeline));
        pass.set_bind_group(0, ctx.resolve_prepared_bind_group(bind_group), &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}

impl<'a> PassNode<'a> for GpuSceneMultiBinCullNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        assert!(
            self.prepared.is_none(),
            "GPU Scene multi-bin Cull was prepared more than once"
        );
        let workgroups_x = self.dispatch_items.div_ceil(GPU_SCENE_CULL_WORKGROUP_SIZE);
        assert_ne!(workgroups_x, 0, "live multi-bin Cull has no work");
        assert!(
            workgroups_x <= ctx.device.limits().max_compute_workgroups_per_dimension
                && self.view_count <= ctx.device.limits().max_compute_workgroups_per_dimension,
            "GPU Scene multi-bin Cull exceeds compute dispatch limits"
        );

        let resources = self.feature.multi_bin_resources(ctx);
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.active_batch_buffer),
            0,
            bytemuck::cast_slice(self.active_batches),
        );
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.view_buffer),
            0,
            bytemuck::cast_slice(self.views),
        );
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.domain_buffer),
            0,
            bytemuck::cast_slice(self.domain),
        );
        let bind_group = ctx
            .build_bind_group(
                &resources.layout,
                Some("GpuScene Multi-Bin Cull Bind Group"),
            )
            .bind_buffer(0, self.instances)
            .bind_buffer(1, self.batches)
            .bind_buffer(2, self.active_batch_buffer)
            .bind_buffer(3, self.visible_instances)
            .bind_buffer(4, self.indirect_arguments)
            .bind_buffer(5, self.view_buffer)
            .bind_buffer(6, self.domain_buffer)
            .build();
        self.prepared = Some(PreparedGpuSceneCull::Dispatch {
            pipeline: resources.pipeline,
            bind_group,
            workgroups_x,
            workgroups_y: self.view_count,
        });
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let PreparedGpuSceneCull::Dispatch {
            pipeline,
            bind_group,
            workgroups_x,
            workgroups_y,
        } = self
            .prepared
            .expect("live GPU Scene multi-bin Cull reached execute without preparation")
        else {
            unreachable!("multi-bin Cull never uses the empty legacy state")
        };

        // One clear plus one dispatch for the complete family. Static argument
        // fields and bounded instance counts are populated by disjoint shader
        // fields, avoiding cross-workgroup initialization barriers.
        encoder.clear_buffer(
            ctx.get_buffer(self.indirect_arguments),
            0,
            Some(self.argument_size),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GpuScene Multi-Bin Cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ctx.pipeline_cache.get_compute_pipeline(pipeline));
        pass.set_bind_group(0, ctx.resolve_prepared_bind_group(bind_group), &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WgpuContext;
    use crate::graph::core::{FrameArena, GraphStorage, PassNode};
    use crate::graph::gpu_scene::{
        GPU_SCENE_INSTANCE_CASTS_SHADOW, GpuScene, GpuSceneBatchId, GpuSceneInstanceInput,
        GpuSceneInstanceRecord,
    };
    use crate::settings::{RendererInitConfig, RendererSettings};
    use glam::{Mat4, Vec3};
    use myth_resources::BoundingBox;
    use myth_scene::NodeHandle;
    use slotmap::KeyData;
    use wgpu::util::DeviceExt;

    fn gpu() -> WgpuContext {
        pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required")
    }

    fn input(node: u64) -> GpuSceneInstanceInput {
        GpuSceneInstanceInput::new(
            NodeHandle::from(KeyData::from_ffi(node)),
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            BoundingBox {
                min: Vec3::splat(-0.25),
                max: Vec3::splat(0.25),
            },
        )
    }

    fn cube_planes() -> [[f32; 4]; 6] {
        [
            [1.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    }

    fn cull_bin(
        slot: u32,
        generation: u32,
        partition_base: u64,
        partition_size: u64,
    ) -> GpuSceneCullBinDescriptor {
        GpuSceneActiveBatchDescriptor {
            id: GpuSceneBatchId::from_parts(slot, generation),
            partition_base,
            partition_size,
        }
    }

    #[derive(Debug)]
    struct CullReadback {
        args: [u32; 5],
        visible: Vec<GpuSceneVisibleInstance>,
    }

    fn run_cull(
        gpu: &WgpuContext,
        records: &[GpuSceneInstanceRecord],
        descriptor: GpuSceneCullDescriptor,
        repetitions: u32,
    ) -> CullReadback {
        let empty_record = GpuSceneInstanceRecord::zeroed();
        let records = if records.is_empty() {
            std::slice::from_ref(&empty_record)
        } else {
            records
        };
        let instances = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene Cull test instances"),
                contents: bytemuck::cast_slice(records),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let visible_size = u64::from(descriptor.slot_span.max(1)) * GPU_SCENE_VISIBLE_INSTANCE_SIZE;
        let visible = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuScene Cull test visible"),
            size: visible_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let args_seed = [u32::MAX; 5];
        let args = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene Cull test args"),
                contents: bytemuck::cast_slice(&args_seed),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let params = descriptor.params(0);
        let params_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene Cull test params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout = create_gpu_scene_cull_layout(&gpu.device, false);
        let mut shader_manager = crate::pipeline::ShaderManager::new();
        let (shader, _) = shader_manager.get_or_compile(
            &gpu.device,
            ShaderSource::File(GPU_SCENE_CULL_SHADER),
            &ShaderCompilationOptions::default(),
        );
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuScene Cull test pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GpuScene Cull test pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuScene Cull test bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let readback_offset = 24u64;
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuScene Cull test readback"),
            size: readback_offset + visible_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuScene Cull test encoder"),
            });
        for _ in 0..repetitions.max(1) {
            encoder.clear_buffer(&args, 0, Some(GPU_SCENE_INDEXED_ARGUMENT_SIZE));
            if descriptor.slot_span != 0 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GpuScene Cull test pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    descriptor.slot_span.div_ceil(GPU_SCENE_CULL_WORKGROUP_SIZE),
                    1,
                    1,
                );
            }
        }
        encoder.copy_buffer_to_buffer(&args, 0, &readback, 0, GPU_SCENE_INDEXED_ARGUMENT_SIZE);
        encoder.copy_buffer_to_buffer(&visible, 0, &readback, readback_offset, visible_size);
        gpu.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GpuScene Cull readback poll failed");
        rx.recv()
            .expect("GpuScene Cull map callback was dropped")
            .expect("GpuScene Cull readback map failed");
        let mapped = slice.get_mapped_range();
        let args =
            bytemuck::pod_read_unaligned(&mapped[..GPU_SCENE_INDEXED_ARGUMENT_SIZE as usize]);
        let visible = bytemuck::cast_slice(
            &mapped[readback_offset as usize..(readback_offset + visible_size) as usize],
        )
        .to_vec();
        drop(mapped);
        readback.unmap();

        CullReadback { args, visible }
    }

    fn run_cull_batch(
        gpu: &WgpuContext,
        records: &[GpuSceneInstanceRecord],
        descriptors: &[GpuSceneCullDescriptor],
        repetitions: u32,
    ) -> (Vec<[u32; 5]>, Vec<GpuSceneVisibleInstance>) {
        let first = descriptors[0];
        let view_count = descriptors.len() as u32;
        let visible_stride = first.slot_span.max(1);
        let empty_record = GpuSceneInstanceRecord::zeroed();
        let records = if records.is_empty() {
            std::slice::from_ref(&empty_record)
        } else {
            records
        };
        let instances = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene packed Cull test instances"),
                contents: bytemuck::cast_slice(records),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let visible_size =
            u64::from(view_count) * u64::from(visible_stride) * GPU_SCENE_VISIBLE_INSTANCE_SIZE;
        let visible = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuScene packed Cull test visible"),
            size: visible_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let argument_size = u64::from(view_count) * GPU_SCENE_INDEXED_ARGUMENT_SIZE;
        let args_seed = vec![u32::MAX; descriptors.len() * 5];
        let args = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene packed Cull test args"),
                contents: bytemuck::cast_slice(&args_seed),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let params: Vec<_> = descriptors
            .iter()
            .copied()
            .enumerate()
            .map(|(index, descriptor)| descriptor.params(index as u32 * visible_stride))
            .collect();
        let params_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene packed Cull test params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let layout = create_gpu_scene_cull_layout(&gpu.device, true);
        let mut shader_options = ShaderCompilationOptions::default();
        shader_options.add_define("GPU_SCENE_MULTI_VIEW", "1");
        let mut shader_manager = crate::pipeline::ShaderManager::new();
        let (shader, _) = shader_manager.get_or_compile(
            &gpu.device,
            ShaderSource::File(GPU_SCENE_CULL_SHADER),
            &shader_options,
        );
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuScene packed Cull test pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GpuScene packed Cull test pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuScene packed Cull test bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuScene packed Cull test readback"),
            size: argument_size + visible_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuScene packed Cull test encoder"),
            });
        for _ in 0..repetitions.max(1) {
            encoder.clear_buffer(&args, 0, Some(argument_size));
            if first.slot_span != 0 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GpuScene packed Cull test pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    first.slot_span.div_ceil(GPU_SCENE_CULL_WORKGROUP_SIZE),
                    view_count,
                    1,
                );
            }
        }
        encoder.copy_buffer_to_buffer(&args, 0, &readback, 0, argument_size);
        encoder.copy_buffer_to_buffer(&visible, 0, &readback, argument_size, visible_size);
        gpu.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GpuScene packed Cull readback poll failed");
        rx.recv()
            .expect("GpuScene packed Cull map callback was dropped")
            .expect("GpuScene packed Cull readback map failed");
        let mapped = slice.get_mapped_range();
        let args = mapped[..argument_size as usize]
            .chunks_exact(GPU_SCENE_INDEXED_ARGUMENT_SIZE as usize)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let visible = bytemuck::cast_slice(&mapped[argument_size as usize..]).to_vec();
        drop(mapped);
        readback.unmap();
        (args, visible)
    }

    #[repr(C, align(16))]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct TestMultiBinInstance {
        world_matrix: [[f32; 4]; 4],
        world_matrix_inverse: [[f32; 4]; 4],
        previous_world_matrix: [[f32; 4]; 4],
        bounds_min: [f32; 4],
        bounds_max: [f32; 4],
        batch_id: [u32; 4],
        metadata: [u32; 4],
    }

    #[repr(C, align(16))]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct TestMultiBinBatch {
        metadata: [u32; 4],
        geometry_id: [u32; 2],
        material_id: [u32; 2],
        partition_bytes: [u64; 2],
        draw_state: [u32; 8],
    }

    const _: () = assert!(size_of::<TestMultiBinInstance>() == 256);
    const _: () = assert!(size_of::<TestMultiBinBatch>() == GPU_SCENE_BATCH_RECORD_SIZE as usize);

    fn test_multi_bin_instance(
        instance_generation: u32,
        scene_id: u32,
        flags: u32,
        batch_slot: u32,
        batch_generation: u32,
        center_x: f32,
    ) -> TestMultiBinInstance {
        TestMultiBinInstance {
            world_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            world_matrix_inverse: Mat4::IDENTITY.to_cols_array_2d(),
            previous_world_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            bounds_min: [center_x - 0.25, -0.25, -0.25, 0.0],
            bounds_max: [center_x + 0.25, 0.25, 0.25, 0.0],
            batch_id: [batch_slot, batch_generation, 0, 0],
            metadata: [instance_generation, 1, scene_id, flags],
        }
    }

    fn test_multi_bin_batch(
        generation: u32,
        scene_id: u32,
        member_count: u32,
        partition_offset: u64,
        partition_size: u64,
        index_count: u32,
    ) -> TestMultiBinBatch {
        TestMultiBinBatch {
            metadata: [generation, 1, scene_id, member_count],
            geometry_id: [generation, 1],
            material_id: [generation, 1],
            partition_bytes: [partition_offset, partition_size],
            draw_state: [
                u32::try_from(partition_offset / GPU_SCENE_VISIBLE_INSTANCE_SIZE).unwrap(),
                u32::try_from(partition_size / GPU_SCENE_VISIBLE_INSTANCE_SIZE).unwrap(),
                index_count,
                0,
                0,
                0,
                0,
                0,
            ],
        }
    }

    fn run_multi_bin_cull(
        gpu: &WgpuContext,
        instances: &[TestMultiBinInstance],
        batches: &[TestMultiBinBatch],
        descriptor: GpuSceneMultiBinCullDescriptor<'_>,
    ) -> (Vec<[u32; 5]>, Vec<GpuSceneVisibleInstance>) {
        validate_multi_bin_descriptor(descriptor);
        let view_count = descriptor.views.len() as u32;
        let active_batch_count = descriptor.bins.len() as u32;
        let domain = GpuSceneMultiBinCullDomain {
            domain: [
                descriptor.active_scene_id,
                descriptor.instance_slot_span,
                descriptor.batch_slot_span,
                active_batch_count,
            ],
            layout: [
                view_count,
                (descriptor.visible_view_stride / GPU_SCENE_VISIBLE_INSTANCE_SIZE) as u32,
                descriptor.required_instance_flags,
                descriptor.instance_slot_span.max(active_batch_count),
            ],
        };
        let views = descriptor
            .views
            .iter()
            .map(|view| GpuSceneMultiBinCullView {
                frustum_planes: view.frustum_planes,
            })
            .collect::<Vec<_>>();
        let active = descriptor
            .bins
            .iter()
            .map(|bin| GpuSceneMultiBinCullActiveBatch {
                id: [bin.id.slot(), bin.id.generation(), 0, 0],
            })
            .collect::<Vec<_>>();

        let instances = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene multi-bin test instances"),
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let batches = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene multi-bin test batches"),
                contents: bytemuck::cast_slice(batches),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let active = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene multi-bin test active batches"),
                contents: bytemuck::cast_slice(&active),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let views = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene multi-bin test views"),
                contents: bytemuck::cast_slice(&views),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let domain_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuScene multi-bin test domain"),
                contents: bytemuck::bytes_of(&domain),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let visible_size = u64::from(view_count) * descriptor.visible_view_stride;
        let visible = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuScene multi-bin test visible"),
            size: visible_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let argument_size = u64::from(view_count)
            * u64::from(descriptor.batch_slot_span)
            * GPU_SCENE_INDEXED_ARGUMENT_SIZE;
        let arguments = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuScene multi-bin test arguments"),
            size: argument_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let layout = create_gpu_scene_multi_bin_cull_layout(&gpu.device);
        let mut shader_options = ShaderCompilationOptions::default();
        shader_options.add_define("GPU_SCENE_MULTI_BIN", "1");
        let mut shader_manager = crate::pipeline::ShaderManager::new();
        let (shader, _) = shader_manager.get_or_compile(
            &gpu.device,
            ShaderSource::File(GPU_SCENE_CULL_SHADER),
            &shader_options,
        );
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuScene multi-bin test pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GpuScene multi-bin test pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuScene multi-bin test bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: batches.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: active.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: arguments.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: views.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: domain_buffer.as_entire_binding(),
                },
            ],
        });

        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuScene multi-bin test readback"),
            size: argument_size + visible_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuScene multi-bin test encoder"),
            });
        encoder.clear_buffer(&arguments, 0, Some(argument_size));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GpuScene multi-bin single dispatch test"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                domain.layout[3].div_ceil(GPU_SCENE_CULL_WORKGROUP_SIZE),
                view_count,
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&arguments, 0, &readback, 0, argument_size);
        encoder.copy_buffer_to_buffer(&visible, 0, &readback, argument_size, visible_size);
        gpu.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GpuScene multi-bin readback poll failed");
        rx.recv()
            .expect("GpuScene multi-bin map callback was dropped")
            .expect("GpuScene multi-bin readback map failed");
        let mapped = slice.get_mapped_range();
        let args = mapped[..argument_size as usize]
            .chunks_exact(GPU_SCENE_INDEXED_ARGUMENT_SIZE as usize)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let visible = bytemuck::cast_slice(&mapped[argument_size as usize..]).to_vec();
        drop(mapped);
        readback.unmap();
        (args, visible)
    }

    struct TestConsumer;

    impl PassNode<'_> for TestConsumer {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    #[test]
    fn cull_abi_is_stable() {
        assert_eq!(size_of::<GpuSceneCullParams>(), 128);
        assert_eq!(std::mem::align_of::<GpuSceneCullParams>(), 16);
        assert_eq!(size_of::<GpuSceneVisibleInstance>(), 8);
        assert_eq!(GPU_SCENE_INDEXED_ARGUMENT_SIZE, 20);
    }

    #[test]
    #[should_panic(expected = "GPU Scene visible view stride is not storage-aligned")]
    fn multi_bin_descriptor_rejects_wrong_visible_record_stride() {
        let views = [GpuSceneCullViewDescriptor::new([[0.0; 4]; 6])];
        let bins = [cull_bin(0, 1, 0, GPU_SCENE_VISIBLE_INSTANCE_SIZE)];
        validate_multi_bin_descriptor(GpuSceneMultiBinCullDescriptor::new_untrusted(
            7, 1, 1, 8, 12, &views, &bins,
        ));
    }

    #[test]
    #[should_panic(expected = "visible view-major index space exceeds WGSL u32 indexing")]
    fn multi_bin_descriptor_rejects_visible_view_major_index_overflow() {
        let views = [GpuSceneCullViewDescriptor::new([[0.0; 4]; 6]); 9];
        let bins = [cull_bin(0, 1, 0, GPU_SCENE_VISIBLE_INSTANCE_SIZE)];
        let visible_view_stride = u64::from(u32::MAX) & !(GPU_SCENE_VISIBLE_INSTANCE_SIZE - 1);
        validate_multi_bin_descriptor(GpuSceneMultiBinCullDescriptor::new_untrusted(
            7,
            1,
            1,
            GPU_SCENE_VISIBLE_INSTANCE_SIZE,
            visible_view_stride,
            &views,
            &bins,
        ));
    }

    #[test]
    #[should_panic(expected = "argument view-major index space exceeds WGSL u32 indexing")]
    fn multi_bin_descriptor_rejects_argument_view_major_index_overflow() {
        let views = [GpuSceneCullViewDescriptor::new([[0.0; 4]; 6]); 2];
        let bins = [cull_bin(0, 1, 0, GPU_SCENE_VISIBLE_INSTANCE_SIZE)];
        validate_multi_bin_descriptor(GpuSceneMultiBinCullDescriptor::new_untrusted(
            7,
            1,
            u32::MAX,
            GPU_SCENE_VISIBLE_INSTANCE_SIZE,
            GPU_SCENE_VISIBLE_INSTANCE_SIZE,
            &views,
            &bins,
        ));
    }

    #[test]
    #[should_panic(expected = "GPU Scene active batch slot was duplicated")]
    fn multi_bin_descriptor_rejects_duplicate_active_slot() {
        let views = [GpuSceneCullViewDescriptor::new([[0.0; 4]; 6])];
        let bins = [
            cull_bin(0, 1, 0, GPU_SCENE_VISIBLE_INSTANCE_SIZE),
            cull_bin(
                0,
                2,
                GPU_SCENE_VISIBLE_INSTANCE_SIZE,
                GPU_SCENE_VISIBLE_INSTANCE_SIZE,
            ),
        ];
        validate_multi_bin_descriptor(GpuSceneMultiBinCullDescriptor::new_untrusted(
            7,
            1,
            1,
            GPU_SCENE_VISIBLE_INSTANCE_SIZE,
            2 * GPU_SCENE_VISIBLE_INSTANCE_SIZE,
            &views,
            &bins,
        ));
    }

    #[test]
    #[should_panic(expected = "GPU Scene visible partitions overlap")]
    fn multi_bin_descriptor_rejects_overlapping_partitions() {
        let views = [GpuSceneCullViewDescriptor::new([[0.0; 4]; 6])];
        let bins = [
            cull_bin(0, 1, 0, 2 * GPU_SCENE_VISIBLE_INSTANCE_SIZE),
            cull_bin(
                1,
                1,
                GPU_SCENE_VISIBLE_INSTANCE_SIZE,
                2 * GPU_SCENE_VISIBLE_INSTANCE_SIZE,
            ),
        ];
        validate_multi_bin_descriptor(GpuSceneMultiBinCullDescriptor::new_untrusted(
            7,
            1,
            2,
            GPU_SCENE_VISIBLE_INSTANCE_SIZE,
            3 * GPU_SCENE_VISIBLE_INSTANCE_SIZE,
            &views,
            &bins,
        ));
    }

    #[test]
    fn multi_bin_cull_shader_and_layout_compile() {
        let gpu = gpu();
        let layout = create_gpu_scene_multi_bin_cull_layout(&gpu.device);
        let mut shader_options = ShaderCompilationOptions::default();
        shader_options.add_define("GPU_SCENE_MULTI_BIN", "1");
        let mut shader_manager = crate::pipeline::ShaderManager::new();
        let (shader, _) = shader_manager.get_or_compile(
            &gpu.device,
            ShaderSource::File(GPU_SCENE_CULL_SHADER),
            &shader_options,
        );
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuScene multi-bin Cull compile test layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let _pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GpuScene multi-bin Cull compile test pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
    }

    #[test]
    fn multi_bin_single_dispatch_handles_views_holes_stale_ids_flags_and_alignment() {
        let gpu = gpu();
        let alignment = u64::from(gpu.device.limits().min_storage_buffer_offset_alignment)
            .max(GPU_SCENE_VISIBLE_INSTANCE_SIZE);
        let partition_size = alignment;
        let visible_view_stride = 2 * alignment;
        let views = [
            GpuSceneCullViewDescriptor::new(cube_planes()),
            GpuSceneCullViewDescriptor::new([[0.0; 4]; 6]),
        ];
        let bins = [
            cull_bin(0, 11, 0, partition_size),
            cull_bin(2, 5, alignment, partition_size),
        ];
        let descriptor = GpuSceneMultiBinCullDescriptor::new_untrusted(
            7,
            7,
            4,
            alignment,
            visible_view_stride,
            &views,
            &bins,
        )
        .with_required_instance_flags(GPU_SCENE_INSTANCE_CASTS_SHADOW);

        let instances = [
            test_multi_bin_instance(1, 7, GPU_SCENE_INSTANCE_CASTS_SHADOW, 0, 11, 0.0),
            test_multi_bin_instance(2, 7, GPU_SCENE_INSTANCE_CASTS_SHADOW, 2, 5, 10.0),
            // Stale generation for a currently live stable batch slot.
            test_multi_bin_instance(3, 7, GPU_SCENE_INSTANCE_CASTS_SHADOW, 2, 4, 0.0),
            // Sparse hole in the BatchTable.
            test_multi_bin_instance(4, 7, GPU_SCENE_INSTANCE_CASTS_SHADOW, 1, 9, 0.0),
            // Missing the required semantic membership flag.
            test_multi_bin_instance(5, 7, 0, 0, 11, 0.0),
            // A different retained scene domain must fail closed.
            test_multi_bin_instance(6, 8, GPU_SCENE_INSTANCE_CASTS_SHADOW, 2, 5, 10.0),
            test_multi_bin_instance(7, 7, GPU_SCENE_INSTANCE_CASTS_SHADOW, 2, 5, 10.0),
        ];
        let mut batches = [TestMultiBinBatch::zeroed(); 4];
        batches[0] = test_multi_bin_batch(11, 7, 1, 0, partition_size, 6);
        batches[2] = test_multi_bin_batch(5, 7, 2, alignment, partition_size, 36);

        let (args, visible) = run_multi_bin_cull(&gpu, &instances, &batches, descriptor);
        assert_eq!(args.len(), 8);
        assert_eq!(args[0], [6, 1, 0, 0, 0]);
        assert_eq!(args[1], [0; 5]);
        assert_eq!(args[2], [36, 0, 0, 0, 0]);
        assert_eq!(args[3], [0; 5]);
        assert_eq!(args[4], [6, 1, 0, 0, 0]);
        assert_eq!(args[5], [0; 5]);
        assert_eq!(args[6], [36, 2, 0, 0, 0]);
        assert_eq!(args[7], [0; 5]);
        assert!(args.iter().all(|argument| argument[4] == 0));

        let records_per_alignment = (alignment / GPU_SCENE_VISIBLE_INSTANCE_SIZE) as usize;
        assert_eq!(
            visible[0],
            GpuSceneVisibleInstance {
                slot: 0,
                generation: 1,
            }
        );
        assert_eq!(
            visible[2 * records_per_alignment],
            GpuSceneVisibleInstance {
                slot: 0,
                generation: 1,
            }
        );
        let mut far_batch = visible[3 * records_per_alignment..][..args[6][1] as usize].to_vec();
        far_batch.sort_unstable_by_key(|entry| entry.slot);
        assert_eq!(
            far_batch,
            [
                GpuSceneVisibleInstance {
                    slot: 1,
                    generation: 2,
                },
                GpuSceneVisibleInstance {
                    slot: 6,
                    generation: 7,
                },
            ]
        );
    }

    #[test]
    fn multi_bin_member_count_truncates_ghost_instances_before_partition_padding() {
        let gpu = gpu();
        let alignment = u64::from(gpu.device.limits().min_storage_buffer_offset_alignment)
            .max(GPU_SCENE_VISIBLE_INSTANCE_SIZE);
        let views = [GpuSceneCullViewDescriptor::new([[0.0; 4]; 6])];
        let bins = [cull_bin(0, 11, 0, alignment)];
        let descriptor = GpuSceneMultiBinCullDescriptor::new_untrusted(
            7, 4, 1, alignment, alignment, &views, &bins,
        );
        let instances = [
            test_multi_bin_instance(1, 7, 0, 0, 11, 0.0),
            test_multi_bin_instance(2, 7, 0, 0, 11, 0.0),
            test_multi_bin_instance(3, 7, 0, 0, 11, 0.0),
            test_multi_bin_instance(4, 7, 0, 0, 11, 0.0),
        ];
        let batches = [test_multi_bin_batch(11, 7, 1, 0, alignment, 12)];

        let (args, visible) = run_multi_bin_cull(&gpu, &instances, &batches, descriptor);
        assert_eq!(args, [[12, 1, 0, 0, 0]]);
        assert!(visible[0].slot < instances.len() as u32);
        assert_eq!(visible[0].generation, visible[0].slot + 1);
    }

    #[test]
    fn multi_bin_wrong_visible_record_stride_fails_closed() {
        let gpu = gpu();
        let alignment = u64::from(gpu.device.limits().min_storage_buffer_offset_alignment)
            .max(GPU_SCENE_VISIBLE_INSTANCE_SIZE);
        let views = [GpuSceneCullViewDescriptor::new([[0.0; 4]; 6])];
        let bins = [cull_bin(0, 11, 0, alignment)];
        let descriptor = GpuSceneMultiBinCullDescriptor::new_untrusted(
            7, 1, 1, alignment, alignment, &views, &bins,
        );
        let instances = [test_multi_bin_instance(1, 7, 0, 0, 11, 0.0)];
        let mut batches = [test_multi_bin_batch(11, 7, 1, 0, alignment, 12)];
        batches[0].draw_state[1] =
            u32::try_from(alignment / (2 * GPU_SCENE_VISIBLE_INSTANCE_SIZE)).unwrap();

        let (args, _) = run_multi_bin_cull(&gpu, &instances, &batches, descriptor);
        assert_eq!(args, [[0; 5]]);
    }

    #[test]
    fn multi_bin_member_count_exceeding_capacity_fails_closed() {
        let gpu = gpu();
        let alignment = u64::from(gpu.device.limits().min_storage_buffer_offset_alignment)
            .max(GPU_SCENE_VISIBLE_INSTANCE_SIZE);
        let capacity = u32::try_from(alignment / GPU_SCENE_VISIBLE_INSTANCE_SIZE).unwrap();
        let views = [GpuSceneCullViewDescriptor::new([[0.0; 4]; 6])];
        let bins = [cull_bin(0, 11, 0, alignment)];
        let descriptor = GpuSceneMultiBinCullDescriptor::new_untrusted(
            7, 1, 1, alignment, alignment, &views, &bins,
        );
        let instances = [test_multi_bin_instance(1, 7, 0, 0, 11, 0.0)];
        let batches = [test_multi_bin_batch(11, 7, capacity + 1, 0, alignment, 12)];

        let (args, _) = run_multi_bin_cull(&gpu, &instances, &batches, descriptor);
        assert_eq!(args, [[0; 5]]);
    }

    #[test]
    fn multi_bin_output_resolves_exact_portable_draw_ranges() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let views = [
            GpuSceneCullViewDescriptor::new(cube_planes()),
            GpuSceneCullViewDescriptor::new([[0.0; 4]; 6]),
        ];
        let bins = [cull_bin(0, 3, 0, 256), cull_bin(3, 9, 256, 256)];
        let descriptor =
            GpuSceneMultiBinCullDescriptor::new_untrusted(7, 10, 5, 256, 512, &views, &bins);
        let feature = GpuSceneCullFeature::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let (instances, batches) = graph.add_pass("GpuScene_MultiBin_Test_Tables", |builder| {
            let instances = builder.create_buffer(
                "GpuScene test instances",
                BufferDesc::new(10 * 256, wgpu::BufferUsages::STORAGE),
            );
            let batches = builder.create_buffer(
                "GpuScene test batches",
                BufferDesc::new(5 * GPU_SCENE_BATCH_RECORD_SIZE, wgpu::BufferUsages::STORAGE),
            );
            (TestConsumer, (instances, batches))
        });
        let output = feature.add_multi_bin_to_graph(
            &mut graph,
            instances,
            batches,
            "GpuScene_MultiBin_Cull",
            descriptor,
        );
        assert_eq!(
            output.bins.as_ptr(),
            bins.as_ptr(),
            "Cull must retain the authority descriptor slice without copying"
        );
        assert_eq!(output.view_count(), 2);
        assert_eq!(output.batch_count(), 2);
        let view0_batch1 = output.view_batch(0, 1).unwrap();
        assert_eq!(view0_batch1.visible_offset, 256);
        assert_eq!(view0_batch1.visible_size, 256);
        assert_eq!(view0_batch1.indirect_offset, 3 * 20);
        assert_eq!(view0_batch1.first_instance, 0);
        let view1_batch1 = output.view_batch(1, 1).unwrap();
        assert_eq!(view1_batch1.visible_offset, 768);
        assert_eq!(view1_batch1.indirect_offset, (5 + 3) * 20);
        assert_eq!(view1_batch1.first_instance, 0);
        assert!(output.view_batch(2, 0).is_none());
        assert!(output.view_batch(0, 2).is_none());

        graph.add_pass("GpuScene_MultiBin_Test_Consumer", |builder| {
            builder.read_buffer(output.visible_instances);
            builder.read_buffer(output.indirect_arguments);
            builder.mark_side_effect();
            (TestConsumer, ())
        });
        graph.compile_topology();
        let live = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect::<Vec<_>>();
        assert_eq!(
            live,
            [
                "GpuScene_MultiBin_Test_Tables",
                "GpuScene_MultiBin_Cull",
                "GpuScene_MultiBin_Test_Consumer",
            ]
        );
    }

    #[test]
    fn packed_batch_outputs_exact_slices_and_one_feature_can_feed_multiple_nodes() {
        let mut registry = crate::graph::core::PersistentBufferRegistry::new();
        let mut gpu_scene = GpuScene::new();
        gpu_scene.synchronize_scene(7, [input(1), input(2)]);
        let plan = gpu_scene.plan_instance_table(&mut registry, 1).unwrap();
        let feature = GpuSceneCullFeature::new();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let patched = super::super::add_gpu_scene_patch(&mut graph, plan);
        let descriptor =
            GpuSceneCullDescriptor::new(7, patched.slot_span, cube_planes(), 36, 4, -2)
                .with_required_instance_flags(GPU_SCENE_INSTANCE_CASTS_SHADOW);
        let main = feature.add_to_graph(
            &mut graph,
            patched.instances,
            GpuSceneCullDescriptor::new(7, patched.slot_span, cube_planes(), 36, 4, -2),
        );
        let d2 = feature.add_batch_to_graph(
            &mut graph,
            patched.instances,
            "GpuScene_Shadow_2D_Cull",
            &[descriptor, descriptor],
        );
        let cube = feature.add_batch_to_graph(
            &mut graph,
            patched.instances,
            "GpuScene_Shadow_Cube_Cull",
            &[descriptor],
        );

        assert_eq!(d2.view_count(), 2);
        assert_eq!(d2.view(0).unwrap().indirect_offset, 0);
        assert_eq!(d2.view(0).unwrap().first_instance, 0);
        assert_eq!(d2.view(1).unwrap().indirect_offset, 20);
        assert_eq!(d2.view(1).unwrap().first_instance, patched.slot_span.max(1));
        assert_eq!(d2.view(0).unwrap().visible_instances, d2.visible_instances);
        assert_eq!(
            d2.view(1).unwrap().indirect_arguments,
            d2.indirect_arguments
        );
        assert!(d2.view(2).is_none());
        assert_ne!(d2.visible_instances, cube.visible_instances);

        graph.add_pass("GpuScene_Packed_Cull_Test_Consumer", |builder| {
            builder.read_buffer(main.indirect_arguments);
            builder.read_buffer(d2.visible_instances);
            builder.read_buffer(d2.indirect_arguments);
            builder.read_buffer(cube.visible_instances);
            builder.read_buffer(cube.indirect_arguments);
            builder.mark_side_effect();
            (TestConsumer, ())
        });
        graph.compile_topology();
        let live = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect::<Vec<_>>();
        assert_eq!(
            live,
            [
                "GpuScene_Patch",
                "GpuScene_Cull",
                "GpuScene_Shadow_2D_Cull",
                "GpuScene_Shadow_Cube_Cull",
                "GpuScene_Packed_Cull_Test_Consumer",
            ]
        );
    }

    #[test]
    fn packed_multi_view_gpu_cull_filters_flags_and_partitions_visible_ranges() {
        let gpu = gpu();
        let mut records = vec![GpuSceneInstanceRecord::zeroed(); 3];
        records[0].metadata = [1, 1, 7, GPU_SCENE_INSTANCE_CASTS_SHADOW];
        records[0].bounds_min = [-0.25, -0.25, -0.25, 0.0];
        records[0].bounds_max = [0.25, 0.25, 0.25, 0.0];
        records[1].metadata = [2, 1, 7, 0];
        records[1].bounds_min = [-0.25, -0.25, -0.25, 0.0];
        records[1].bounds_max = [0.25, 0.25, 0.25, 0.0];
        records[2].metadata = [3, 1, 7, GPU_SCENE_INSTANCE_CASTS_SHADOW];
        records[2].bounds_min = [9.75, -0.25, -0.25, 0.0];
        records[2].bounds_max = [10.25, 0.25, 0.25, 0.0];

        let constrained = GpuSceneCullDescriptor::new(7, 3, cube_planes(), 36, 4, -2)
            .with_required_instance_flags(GPU_SCENE_INSTANCE_CASTS_SHADOW);
        let all_planes = GpuSceneCullDescriptor::new(7, 3, [[0.0; 4]; 6], 36, 4, -2)
            .with_required_instance_flags(GPU_SCENE_INSTANCE_CASTS_SHADOW);
        let (args, visible) = run_cull_batch(&gpu, &records, &[constrained, all_planes], 3);

        assert_eq!(args[0], [36, 1, 4, (-2i32) as u32, 0]);
        assert_eq!(args[1], [36, 2, 4, (-2i32) as u32, 3]);
        assert_eq!(
            visible[0],
            GpuSceneVisibleInstance {
                slot: 0,
                generation: 1,
            }
        );
        let mut second_view = visible[3..3 + args[1][1] as usize].to_vec();
        second_view.sort_unstable_by_key(|entry| entry.slot);
        assert_eq!(
            second_view,
            [
                GpuSceneVisibleInstance {
                    slot: 0,
                    generation: 1,
                },
                GpuSceneVisibleInstance {
                    slot: 2,
                    generation: 3,
                },
            ]
        );
    }

    #[test]
    fn cull_is_an_ordered_non_rooting_patch_consumer() {
        let mut registry = crate::graph::core::PersistentBufferRegistry::new();
        let mut gpu_scene = GpuScene::new();
        gpu_scene.synchronize_scene(7, [input(1)]);
        let plan = gpu_scene.plan_instance_table(&mut registry, 1).unwrap();
        let feature = GpuSceneCullFeature::new();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let patched = super::super::add_gpu_scene_patch(&mut graph, plan);
        let culled = feature.add_to_graph(
            &mut graph,
            patched.instances,
            GpuSceneCullDescriptor::new(7, patched.slot_span, cube_planes(), 36, 4, -2),
        );
        graph.add_pass("GpuScene_Cull_Test_Consumer", |builder| {
            builder.read_buffer(culled.visible_instances);
            builder.read_buffer(culled.indirect_arguments);
            builder.mark_side_effect();
            (TestConsumer, ())
        });

        graph.compile_topology();
        let live = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect::<Vec<_>>();
        assert_eq!(
            live,
            [
                "GpuScene_Patch",
                "GpuScene_Cull",
                "GpuScene_Cull_Test_Consumer"
            ]
        );
        let args_desc =
            graph.storage.resources[culled.indirect_arguments.index() as usize].buffer_desc();
        assert!(args_desc.usage.contains(wgpu::BufferUsages::COPY_DST));
        assert!(args_desc.usage.contains(wgpu::BufferUsages::INDIRECT));
        assert!(args_desc.usage.contains(wgpu::BufferUsages::STORAGE));
    }

    #[test]
    fn dead_cull_branch_does_not_root_patch_or_cull() {
        let mut registry = crate::graph::core::PersistentBufferRegistry::new();
        let mut gpu_scene = GpuScene::new();
        gpu_scene.synchronize_scene(7, [input(1)]);
        let plan = gpu_scene.plan_instance_table(&mut registry, 1).unwrap();
        let feature = GpuSceneCullFeature::new();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let patched = super::super::add_gpu_scene_patch(&mut graph, plan);
        let _ = feature.add_to_graph(
            &mut graph,
            patched.instances,
            GpuSceneCullDescriptor::new(7, patched.slot_span, cube_planes(), 36, 0, 0),
        );

        graph.compile_topology();
        assert!(graph.storage.execution_queue.is_empty());
    }

    #[test]
    fn sparse_high_slot_and_all_domain_filters_are_honored() {
        let gpu = gpu();
        let mut records = vec![GpuSceneInstanceRecord::zeroed(); 130];
        records[0].metadata = [1, 0, 7, 0];
        records[1].metadata = [1, 1, 8, 0];
        records[2].metadata = [0, 1, 7, 0];
        records[3].metadata = [1, 1, 7, 0];
        records[3].bounds_min = [2.0, 2.0, 2.0, 0.0];
        records[3].bounds_max = [3.0, 3.0, 3.0, 0.0];
        records[129].metadata = [9, 1, 7, 0];
        records[129].bounds_min = [-0.25, -0.25, -0.25, 0.0];
        records[129].bounds_max = [0.25, 0.25, 0.25, 0.0];

        let result = run_cull(
            &gpu,
            &records,
            GpuSceneCullDescriptor::new(7, 130, cube_planes(), 36, 4, -2),
            1,
        );
        assert_eq!(result.args, [36, 1, 4, (-2i32) as u32, 0]);
        assert_eq!(
            &result.visible[..result.args[1] as usize],
            &[GpuSceneVisibleInstance {
                slot: 129,
                generation: 9,
            }]
        );
    }

    #[test]
    fn zero_visible_is_valid_and_clear_prevents_count_accumulation() {
        let gpu = gpu();
        let mut record = GpuSceneInstanceRecord::zeroed();
        record.metadata = [4, 1, 99, 0];
        let result = run_cull(
            &gpu,
            &[record],
            GpuSceneCullDescriptor::new(7, 1, cube_planes(), 12, 3, 5),
            3,
        );
        assert_eq!(result.args, [12, 0, 3, 5, 0]);
    }

    #[test]
    fn empty_slot_span_clears_the_entire_argument_record_without_dispatch() {
        let gpu = gpu();
        let result = run_cull(
            &gpu,
            &[],
            GpuSceneCullDescriptor::new(7, 0, cube_planes(), 12, 3, 5),
            1,
        );
        assert_eq!(result.args, [0; 5]);
    }

    #[test]
    fn non_finite_bounds_are_conservatively_retained() {
        let gpu = gpu();
        let mut record = GpuSceneInstanceRecord::zeroed();
        record.metadata = [5, 1, 7, 0];
        record.bounds_min = [f32::NAN, 1000.0, 1000.0, 0.0];
        record.bounds_max = [f32::INFINITY, 1001.0, 1001.0, 0.0];
        let result = run_cull(
            &gpu,
            &[record],
            GpuSceneCullDescriptor::new(7, 1, cube_planes(), 6, 0, 0),
            1,
        );
        assert_eq!(result.args[1], 1);
        assert_eq!(
            result.visible[0],
            GpuSceneVisibleInstance {
                slot: 0,
                generation: 5
            }
        );
    }
}
