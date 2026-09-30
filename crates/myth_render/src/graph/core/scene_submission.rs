//! Backend-neutral scene-submission identities and logical dependencies.
//!
//! This module is the contract boundary between render Features and the
//! current CPU-direct command backend. Features receive opaque attempt-scoped
//! IDs; backend tags and `DrawPlanId` values remain private to the core
//! coordinator. Built-in scene Features depend only on this facade.

use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;
#[cfg(test)]
use smallvec::SmallVec;

use crate::core::binding::{PreparedBindGroupId, PreparedBindGroupStore};
use crate::core::gpu::Tracked;
use crate::graph::capacity::{
    CapacityUsage, estimated_hash_map_footprint, shrink_hash_map_by_family_ratio,
    shrink_vec_by_family_ratio, vec_footprint,
};
use crate::graph::frame::{
    DrawBindGroupSlot, DrawBufferSlot, DrawCommand, DrawPlanId, DrawPlanKey, DrawPlanResolver,
    DrawPlanStore, FrameSourceId,
};
use crate::pipeline::RenderPipelineId;

use super::persistent_buffer::PersistentBufferPhysicalToken;
use super::types::{BufferNodeId, ErasedResourceNodeId, TextureNodeId};

static NEXT_SCENE_SUBMISSION_ATTEMPT: AtomicU64 = AtomicU64::new(1);

fn next_attempt_token() -> u64 {
    NEXT_SCENE_SUBMISSION_ATTEMPT
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("scene-submission attempt token space exhausted")
}

/// Opaque graph-build/DPE identity for one scene-submission plan.
///
/// The private fields prevent Features from inspecting the selected backend.
/// The process-unique attempt token rejects IDs retained across GraphStorage
/// clear boundaries or originating in another renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SceneSubmissionPlanId {
    attempt_token: u64,
    index: u32,
}

/// Opaque identity for one GPU-indirect backend plan in a graph attempt.
///
/// The plan identity is deliberately distinct from [`SceneSubmissionPlanId`]:
/// Features retain only the backend-neutral identity while the coordinator
/// uses this generation-stamped handle to materialize the selected backend.
/// Clearing the attempt advances the token and makes every retained handle
/// fail closed before its dense index can be reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GpuIndirectPlanId {
    attempt_token: u64,
    index: u32,
}

/// Attempt-scoped contiguous range of coordinator-published GPU batches.
///
/// The range is private to the backend and becomes observable only after all
/// constituent batches and vertex-binding slices have been appended. Keeping
/// the attempt token beside the dense range prevents a stale finalized plan
/// from aliasing a later graph attempt after retained storage is cleared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GpuIndexedIndirectStreamRange {
    attempt_token: u64,
    start: u32,
    end: u32,
}

/// Opaque post-finalize identity for an executable scene draw stream.
///
/// A plan becomes a stream only after DPE selected it and its backend reports
/// successful finalization. Empty CPU-direct command ranges are valid streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DrawStreamId {
    attempt_token: u64,
    index: u32,
}

/// Canonical backend-neutral recipe selected by scene Features.
///
/// This key describes semantic draw intent only. Mapping it to the current
/// CPU-direct backend's private [`DrawPlanKey`] is owned by the coordinator,
/// so Features cannot couple themselves to command-range implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SceneSubmissionKey {
    Opaque,
    Transparent,
    Prepass {
        needs_normal: bool,
        needs_feature_id: bool,
        needs_velocity: bool,
    },
    Shadow {
        light_id: u64,
        layer_index: u32,
    },
}

impl SceneSubmissionKey {
    #[must_use]
    pub(super) const fn cpu_direct_key(self) -> DrawPlanKey {
        match self {
            Self::Opaque => DrawPlanKey::Opaque,
            Self::Transparent => DrawPlanKey::Transparent,
            Self::Prepass {
                needs_normal,
                needs_feature_id,
                needs_velocity,
            } => DrawPlanKey::Prepass {
                needs_normal,
                needs_feature_id,
                needs_velocity,
            },
            Self::Shadow {
                light_id,
                layer_index,
            } => DrawPlanKey::Shadow {
                light_id,
                layer_index,
            },
        }
    }
}

impl From<DrawPlanKey> for SceneSubmissionKey {
    fn from(key: DrawPlanKey) -> Self {
        match key {
            DrawPlanKey::Opaque => Self::Opaque,
            DrawPlanKey::Transparent => Self::Transparent,
            DrawPlanKey::Prepass {
                needs_normal,
                needs_feature_id,
                needs_velocity,
            } => Self::Prepass {
                needs_normal,
                needs_feature_id,
                needs_velocity,
            },
            DrawPlanKey::Shadow {
                light_id,
                layer_index,
            } => Self::Shadow {
                light_id,
                layer_index,
            },
        }
    }
}

/// Logical dependency fixture for scene-submission behavior tests.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct SceneSubmissionDependencies {
    reads: SmallVec<[ErasedResourceNodeId; 8]>,
}

#[cfg(test)]
impl SceneSubmissionDependencies {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            reads: SmallVec::new(),
        }
    }

    #[must_use]
    pub(crate) fn as_slice(&self) -> &[ErasedResourceNodeId] {
        &self.reads
    }
}

/// Immutable logical recipe for one GPU-indirect scene submission.
///
/// These are graph resources, not WGPU owners. The draw consumer declares the
/// visible and argument buffers as ordinary RDG reads, which roots the cull
/// producer and its candidate input transitively. `candidate_instances` is
/// retained as coordinator topology metadata but is deliberately not a direct
/// draw read: adding that redundant edge would extend its lifetime and hide a
/// missing cull-pass dependency. The candidate and visible buffers remain
/// distinct even when an early backend aliases them.
///
/// The descriptor owns one checked byte window in the indexed-indirect
/// argument buffer. Generic contract tests may expose one 20-byte record;
/// production multi-bin producers expose the whole immutable sparse window
/// addressed by the finalized batch stream. Keeping the window on the backend
/// plan lets the graph prove the complete physical range before any live-node
/// preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GpuIndirectPlanDescriptor {
    key: SceneSubmissionKey,
    representative: FrameSourceId,
    instance_table: BufferNodeId,
    candidate_instances: BufferNodeId,
    visible_instances: BufferNodeId,
    indirect_arguments: BufferNodeId,
    /// Exact record addressed by the first active candidate. This remains
    /// descriptor identity for finalizer drift checks; it is not necessarily
    /// the beginning of a sparse multi-bin window.
    indirect_offset: u64,
    indirect_window_start: u64,
    indirect_span: u64,
}

impl GpuIndirectPlanDescriptor {
    pub(crate) const INDEXED_ARGUMENT_SIZE: u64 = 5 * std::mem::size_of::<u32>() as u64;

    #[must_use]
    pub(crate) const fn key(self) -> SceneSubmissionKey {
        self.key
    }

    #[must_use]
    #[cfg(test)]
    pub(super) fn new(
        key: SceneSubmissionKey,
        candidate_instances: BufferNodeId,
        visible_instances: BufferNodeId,
        indirect_arguments: BufferNodeId,
    ) -> Self {
        Self {
            key,
            representative: FrameSourceId::from_index(0),
            instance_table: candidate_instances,
            candidate_instances,
            visible_instances,
            indirect_arguments,
            indirect_offset: 0,
            indirect_window_start: 0,
            indirect_span: Self::INDEXED_ARGUMENT_SIZE,
        }
    }

    #[must_use]
    pub(super) const fn new_multi_bin(
        key: SceneSubmissionKey,
        representative: FrameSourceId,
        instance_table: BufferNodeId,
        candidate_instances: BufferNodeId,
        visible_instances: BufferNodeId,
        indirect_arguments: BufferNodeId,
        indirect_offset: u64,
        indirect_window_start: u64,
        indirect_span: u64,
    ) -> Self {
        Self {
            key,
            representative,
            instance_table,
            candidate_instances,
            visible_instances,
            indirect_arguments,
            indirect_offset,
            indirect_window_start,
            indirect_span,
        }
    }

    pub(super) fn dependencies(self) -> impl Iterator<Item = ErasedResourceNodeId> + Clone {
        [
            self.instance_table.erase(),
            self.visible_instances.erase(),
            self.indirect_arguments.erase(),
        ]
        .into_iter()
    }

    #[must_use]
    pub(crate) const fn representative(self) -> FrameSourceId {
        self.representative
    }

    #[must_use]
    pub(crate) const fn instance_table(self) -> BufferNodeId {
        self.instance_table
    }

    #[must_use]
    pub(crate) const fn candidate_instances(self) -> BufferNodeId {
        self.candidate_instances
    }

    #[must_use]
    pub(crate) const fn visible_instances(self) -> BufferNodeId {
        self.visible_instances
    }

    #[must_use]
    pub(crate) const fn indirect_arguments(self) -> BufferNodeId {
        self.indirect_arguments
    }

    #[must_use]
    pub(crate) const fn indirect_offset(self) -> u64 {
        self.indirect_offset
    }

    #[must_use]
    pub(crate) const fn indirect_span(self) -> u64 {
        self.indirect_span
    }

    #[must_use]
    pub(crate) const fn indirect_window_start(self) -> u64 {
        self.indirect_window_start
    }

    /// Checked exclusive end of the immutable argument-buffer window.
    #[must_use]
    pub(crate) const fn checked_indirect_window_end(self) -> Option<u64> {
        self.indirect_window_start.checked_add(self.indirect_span)
    }

    /// Whether one indexed argument record is wholly contained in this plan's
    /// window and starts on its canonical 20-byte record grid.
    #[must_use]
    pub(super) fn contains_indexed_argument(self, offset: u64) -> bool {
        let Some(relative) = offset.checked_sub(self.indirect_window_start) else {
            return false;
        };
        if relative % Self::INDEXED_ARGUMENT_SIZE != 0 {
            return false;
        }
        offset
            .checked_add(Self::INDEXED_ARGUMENT_SIZE)
            .zip(self.checked_indirect_window_end())
            .is_some_and(|(record_end, window_end)| record_end <= window_end)
    }

    /// Every logical buffer mentioned by the descriptor, including topology
    /// metadata that is intentionally not a direct draw dependency.
    pub(super) fn referenced_resources(self) -> impl Iterator<Item = ErasedResourceNodeId> + Clone {
        [
            self.instance_table.erase(),
            self.candidate_instances.erase(),
            self.visible_instances.erase(),
            self.indirect_arguments.erase(),
        ]
        .into_iter()
    }
}

/// Backend selection is deliberately private. Features cannot match or branch
/// on it; only the graph coordinator may project a plan to backend work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SceneSubmissionBackend {
    CpuDirect(DrawPlanId),
    GpuIndirect(GpuIndirectPlanId),
    /// One semantic submission whose disjoint cohorts are materialized by
    /// both physical backends. The pair is one liveness/finalization unit:
    /// execute may observe it only after both streams have been published.
    Hybrid {
        cpu_direct: DrawPlanId,
        gpu_indirect: GpuIndirectPlanId,
    },
}

impl SceneSubmissionBackend {
    #[inline]
    const fn cpu_direct(self) -> Option<DrawPlanId> {
        match self {
            Self::CpuDirect(plan)
            | Self::Hybrid {
                cpu_direct: plan, ..
            } => Some(plan),
            Self::GpuIndirect(_) => None,
        }
    }

    #[inline]
    const fn gpu_indirect(self) -> Option<GpuIndirectPlanId> {
        match self {
            Self::GpuIndirect(plan)
            | Self::Hybrid {
                gpu_indirect: plan, ..
            } => Some(plan),
            Self::CpuDirect(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SceneSubmissionRecord {
    backend: SceneSubmissionBackend,
    dependency_start: u32,
    dependency_end: u32,
    live: bool,
}

#[derive(Debug, Clone, Copy)]
struct GpuIndirectPlanRecord {
    descriptor: GpuIndirectPlanDescriptor,
    /// Exactly one semantic submission backend may own a GPU plan. Without
    /// this claim, separately reserving pure-GPU and Hybrid wrappers for the
    /// same key could execute the identical indirect stream twice.
    owner_backend: Option<SceneSubmissionBackend>,
    finalized_stream: Option<GpuIndexedIndirectStreamRange>,
}

/// Copy-only physical command packet published as the final linearization
/// point of GPU submission preparation. All WGPU owners remain in the graph's
/// attempt stores and are addressed only through checked IDs.
#[derive(Debug, Clone, Copy)]
struct GpuIndexedIndirectBatchRecord {
    pipeline: RenderPipelineId,
    vertex_start: u32,
    vertex_end: u32,
    index_buffer: DrawBufferSlot,
    index_format: wgpu::IndexFormat,
    material_bind_group: DrawBindGroupSlot,
    object_bind_group: PreparedBindGroupId,
    indirect_arguments: DrawBufferSlot,
    indirect_offset: u64,
    instance_table_physical: PersistentBufferPhysicalToken,
}

/// Coordinator input. Publication copies only IDs and appends the vertex-slot
/// slice into retained flat storage; no asset or WGPU owner crosses the gate.
#[derive(Clone, Copy)]
pub(crate) struct GpuIndexedIndirectBatchInput<'a> {
    pub(crate) pipeline: RenderPipelineId,
    pub(crate) vertex_buffers: &'a [DrawBufferSlot],
    pub(crate) index_buffer: DrawBufferSlot,
    pub(crate) index_format: wgpu::IndexFormat,
    pub(crate) material_bind_group: DrawBindGroupSlot,
    pub(crate) object_bind_group: PreparedBindGroupId,
    pub(crate) indirect_arguments: DrawBufferSlot,
    pub(crate) indirect_offset: u64,
    pub(crate) instance_table_physical: PersistentBufferPhysicalToken,
}

/// Borrowed execute packet returned only for a live, finalized plan.
#[derive(Clone, Copy)]
pub(crate) struct GpuIndexedIndirectBatch<'a> {
    pub(crate) pipeline: RenderPipelineId,
    pub(crate) vertex_buffers: &'a [DrawBufferSlot],
    pub(crate) index_buffer: DrawBufferSlot,
    pub(crate) index_format: wgpu::IndexFormat,
    pub(crate) material_bind_group: DrawBindGroupSlot,
    pub(crate) object_bind_group: PreparedBindGroupId,
    pub(crate) indirect_arguments: DrawBufferSlot,
    pub(crate) indirect_offset: u64,
    pub(crate) instance_table_physical: PersistentBufferPhysicalToken,
}

/// Borrowed execute-time view of one immutable compact GPU batch stream.
///
/// Records stay in graph-attempt-owned flat storage. Iteration projects their
/// vertex ranges by reference, so resolving a multi-batch stream copies no
/// WGPU owners and allocates no temporary command list.
#[derive(Clone, Copy)]
pub(crate) struct GpuIndexedIndirectStream<'a> {
    batches: &'a [GpuIndexedIndirectBatchRecord],
    vertex_buffers: &'a [DrawBufferSlot],
}

impl<'a> GpuIndexedIndirectStream<'a> {
    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) fn len(self) -> usize {
        self.batches.len()
    }

    #[inline]
    #[must_use]
    pub(crate) fn is_empty(self) -> bool {
        self.batches.is_empty()
    }

    #[inline]
    pub(crate) fn batches(self) -> impl ExactSizeIterator<Item = GpuIndexedIndirectBatch<'a>> + 'a {
        self.batches.iter().map(move |record| {
            let vertices =
                &self.vertex_buffers[record.vertex_start as usize..record.vertex_end as usize];
            GpuIndexedIndirectBatch {
                pipeline: record.pipeline,
                vertex_buffers: vertices,
                index_buffer: record.index_buffer,
                index_format: record.index_format,
                material_bind_group: record.material_bind_group,
                object_bind_group: record.object_bind_group,
                indirect_arguments: record.indirect_arguments,
                indirect_offset: record.indirect_offset,
                instance_table_physical: record.instance_table_physical,
            }
        })
    }
}

/// Backend tag is exposed only at the common execute facade, after both
/// implementations have become immutable streams.
pub(crate) enum ResolvedSceneSubmission<'a> {
    CpuDirect(&'a [DrawCommand]),
    GpuIndexedIndirect(GpuIndexedIndirectStream<'a>),
    /// Two disjoint cohorts sharing one semantic render-pass contract.
    /// Keeping both borrowed streams in one resolved value preserves the
    /// zero-copy execute path while preventing partial hybrid execution.
    Hybrid {
        cpu_direct: &'a [DrawCommand],
        gpu_indirect: GpuIndexedIndirectStream<'a>,
    },
}

/// Core-only projection of one live CPU-direct plan.
///
/// This type is visible only inside `graph::core`; Feature modules continue to
/// see opaque scene-submission IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LiveCpuDirectPlan {
    pub(super) submission: SceneSubmissionPlanId,
    pub(super) draw_plan: DrawPlanId,
}

/// Core-only projection of one DPE-live GPU-indirect plan.
///
/// Its fields are not visible to Feature modules. The coordinator turns this
/// logical projection into an immutable batch only after exact physical
/// materialization has completed successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LiveGpuIndirectPlan {
    pub(crate) submission: SceneSubmissionPlanId,
    pub(crate) plan: GpuIndirectPlanId,
    pub(crate) descriptor: GpuIndirectPlanDescriptor,
}

/// Test projection of the backend lifecycle without adding counters to the
/// CpuDirect production path before a GPU backend can actually be selected.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct SceneSubmissionBackendCounts {
    pub(crate) reserved_cpu_direct: u32,
    pub(crate) reserved_gpu_indirect: u32,
    pub(crate) reserved_hybrid: u32,
    pub(crate) live_cpu_direct: u32,
    pub(crate) live_gpu_indirect: u32,
    pub(crate) live_hybrid: u32,
}

/// Capacity-retaining plan registry for one GraphStorage attempt.
pub(crate) struct SceneSubmissionStore {
    attempt_token: u64,
    plans: Vec<SceneSubmissionRecord>,
    lookup: FxHashMap<SceneSubmissionBackend, SceneSubmissionPlanId>,
    /// Dense one-owner claim for each attempt-local DrawPlanId. This keeps a
    /// CpuDirect stream from being wrapped once as pure CPU and again as one
    /// side of a Hybrid submission.
    cpu_direct_owners: Vec<Option<SceneSubmissionBackend>>,
    dependencies: Vec<ErasedResourceNodeId>,
    gpu_indirect_plans: Vec<GpuIndirectPlanRecord>,
    gpu_indirect_lookup: FxHashMap<SceneSubmissionKey, GpuIndirectPlanId>,
    gpu_indirect_batches: Vec<GpuIndexedIndirectBatchRecord>,
    gpu_indirect_vertex_buffers: Vec<DrawBufferSlot>,
}

impl Default for SceneSubmissionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneSubmissionStore {
    #[must_use]
    pub(super) fn new() -> Self {
        Self {
            attempt_token: next_attempt_token(),
            plans: Vec::with_capacity(16),
            lookup: FxHashMap::default(),
            cpu_direct_owners: Vec::with_capacity(16),
            dependencies: Vec::with_capacity(64),
            // Keep the default CpuDirect path allocation-neutral. The first
            // GPU reservation owns the cold growth of this retained table.
            gpu_indirect_plans: Vec::new(),
            gpu_indirect_lookup: FxHashMap::default(),
            gpu_indirect_batches: Vec::new(),
            gpu_indirect_vertex_buffers: Vec::new(),
        }
    }

    #[must_use]
    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.plans, 16)
            + estimated_hash_map_footprint(&self.lookup, 16)
            + vec_footprint(&self.cpu_direct_owners, 16)
            + vec_footprint(&self.dependencies, 64)
            + vec_footprint(&self.gpu_indirect_plans, 4)
            + estimated_hash_map_footprint(&self.gpu_indirect_lookup, 4)
            + vec_footprint(&self.gpu_indirect_batches, 4)
            + vec_footprint(&self.gpu_indirect_vertex_buffers, 8)
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        shrink_vec_by_family_ratio(&mut self.plans, 16, target_bytes, retained_bytes);
        shrink_hash_map_by_family_ratio(&mut self.lookup, 16, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(
            &mut self.cpu_direct_owners,
            16,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(&mut self.dependencies, 64, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(
            &mut self.gpu_indirect_plans,
            4,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.gpu_indirect_lookup,
            4,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.gpu_indirect_batches,
            4,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.gpu_indirect_vertex_buffers,
            8,
            target_bytes,
            retained_bytes,
        );
    }

    /// Release attempt records while retaining every backing allocation.
    pub(super) fn clear(&mut self) {
        self.plans.clear();
        self.lookup.clear();
        self.cpu_direct_owners.clear();
        self.dependencies.clear();
        self.gpu_indirect_plans.clear();
        self.gpu_indirect_lookup.clear();
        self.gpu_indirect_batches.clear();
        self.gpu_indirect_vertex_buffers.clear();
        self.attempt_token = next_attempt_token();
    }

    /// Reserve or recover the wrapper for one valid CPU-direct DrawPlan.
    #[cfg(test)]
    pub(super) fn reserve_cpu_direct(
        &mut self,
        draw_plans: &DrawPlanStore,
        draw_plan: DrawPlanId,
        dependencies: &SceneSubmissionDependencies,
    ) -> SceneSubmissionPlanId {
        assert!(
            draw_plans.validate_reserved(draw_plan),
            "scene submission cannot wrap a stale or foreign DrawPlanId"
        );
        let backend = SceneSubmissionBackend::CpuDirect(draw_plan);
        self.claim_cpu_direct_backend(draw_plan, backend);
        self.reserve_backend(backend, dependencies.as_slice().iter().copied())
    }

    /// Zero-temporary-allocation adapter for the current CpuDirect backend.
    /// DrawPlan material dependencies are already stored as typed texture IDs;
    /// copy them directly into the retained submission dependency table.
    pub(super) fn reserve_cpu_direct_textures(
        &mut self,
        draw_plans: &DrawPlanStore,
        draw_plan: DrawPlanId,
        dependencies: &[TextureNodeId],
    ) -> SceneSubmissionPlanId {
        assert!(
            draw_plans.validate_reserved(draw_plan),
            "scene submission cannot wrap a stale or foreign DrawPlanId"
        );
        let backend = SceneSubmissionBackend::CpuDirect(draw_plan);
        self.claim_cpu_direct_backend(draw_plan, backend);
        self.reserve_backend(
            backend,
            dependencies.iter().map(|dependency| dependency.erase()),
        )
    }

    /// Reserve or recover one production GPU-indirect backend plan.
    ///
    /// Backend selection remains coordinator-owned. Feature code receives the
    /// returned [`SceneSubmissionPlanId`] and cannot inspect whether it wraps
    /// this plan or the current CpuDirect adapter.
    #[cfg(test)]
    pub(super) fn reserve_gpu_indirect(
        &mut self,
        descriptor: GpuIndirectPlanDescriptor,
    ) -> SceneSubmissionPlanId {
        self.reserve_gpu_indirect_with_textures(descriptor, &[])
    }

    pub(super) fn reserve_gpu_indirect_with_textures(
        &mut self,
        descriptor: GpuIndirectPlanDescriptor,
        material_dependencies: &[TextureNodeId],
    ) -> SceneSubmissionPlanId {
        let plan = self.reserve_gpu_indirect_plan(descriptor);
        let backend = SceneSubmissionBackend::GpuIndirect(plan);
        self.claim_gpu_indirect_backend(plan, backend);
        let dependencies = || {
            descriptor.dependencies().chain(
                material_dependencies
                    .iter()
                    .map(|dependency| dependency.erase()),
            )
        };
        self.reserve_backend(backend, dependencies())
    }

    /// Reserve one atomic CpuDirect + GPU-indirect semantic submission.
    ///
    /// The material dependency slice is shared by both disjoint cohorts. GPU
    /// buffer dependencies are appended exactly once, and the returned plan
    /// becomes live/finalized only as one unit. This is the core ownership
    /// primitive used by Auto-mode hybrid bin selection; Feature code still
    /// receives only the backend-neutral [`SceneSubmissionPlanId`].
    pub(super) fn reserve_hybrid_with_textures(
        &mut self,
        draw_plans: &DrawPlanStore,
        draw_plan: DrawPlanId,
        descriptor: GpuIndirectPlanDescriptor,
        material_dependencies: &[TextureNodeId],
    ) -> SceneSubmissionPlanId {
        assert!(
            draw_plans.validate_reserved(draw_plan),
            "hybrid scene submission cannot wrap a stale or foreign DrawPlanId"
        );
        let gpu_plan = self.reserve_gpu_indirect_plan(descriptor);
        let backend = SceneSubmissionBackend::Hybrid {
            cpu_direct: draw_plan,
            gpu_indirect: gpu_plan,
        };
        self.claim_cpu_direct_backend(draw_plan, backend);
        self.claim_gpu_indirect_backend(gpu_plan, backend);
        let dependencies = descriptor.dependencies().chain(
            material_dependencies
                .iter()
                .map(|dependency| dependency.erase()),
        );
        self.reserve_backend(backend, dependencies)
    }

    fn reserve_gpu_indirect_plan(
        &mut self,
        descriptor: GpuIndirectPlanDescriptor,
    ) -> GpuIndirectPlanId {
        if let Some(&plan) = self.gpu_indirect_lookup.get(&descriptor.key) {
            assert!(
                self.validate_gpu_indirect_plan(plan),
                "GPU-indirect lookup retained a stale plan identity"
            );
            assert_eq!(
                self.gpu_indirect_record(plan)
                    .expect("validated GPU-indirect plan disappeared")
                    .descriptor,
                descriptor,
                "one semantic GPU-indirect plan was registered with a conflicting descriptor"
            );
            return plan;
        }

        let index = u32::try_from(self.gpu_indirect_plans.len())
            .expect("one graph attempt cannot reserve more than u32::MAX GPU-indirect plans");
        let plan = GpuIndirectPlanId {
            attempt_token: self.attempt_token,
            index,
        };
        self.gpu_indirect_plans.push(GpuIndirectPlanRecord {
            descriptor,
            owner_backend: None,
            finalized_stream: None,
        });
        assert!(
            self.gpu_indirect_lookup
                .insert(descriptor.key, plan)
                .is_none()
        );
        plan
    }

    fn claim_gpu_indirect_backend(
        &mut self,
        plan: GpuIndirectPlanId,
        backend: SceneSubmissionBackend,
    ) {
        let record = self
            .validate_gpu_indirect_plan(plan)
            .then(|| &mut self.gpu_indirect_plans[plan.index as usize])
            .expect("GPU-indirect backend claim retained a stale plan identity");
        if let Some(current) = record.owner_backend {
            assert_eq!(
                current, backend,
                "one semantic GPU-indirect plan cannot be wrapped by two submission backends"
            );
        } else {
            record.owner_backend = Some(backend);
        }
    }

    fn claim_cpu_direct_backend(&mut self, plan: DrawPlanId, backend: SceneSubmissionBackend) {
        let required = plan
            .index()
            .checked_add(1)
            .expect("CpuDirect owner index overflowed");
        if self.cpu_direct_owners.len() < required {
            self.cpu_direct_owners.resize(required, None);
        }
        let owner = &mut self.cpu_direct_owners[plan.index()];
        if let Some(current) = *owner {
            assert_eq!(
                current, backend,
                "one CpuDirect plan cannot be wrapped by two submission backends"
            );
        } else {
            *owner = Some(backend);
        }
    }

    fn reserve_backend<I>(
        &mut self,
        backend: SceneSubmissionBackend,
        dependencies: I,
    ) -> SceneSubmissionPlanId
    where
        I: Iterator<Item = ErasedResourceNodeId> + Clone,
    {
        if let Some(&id) = self.lookup.get(&backend) {
            assert!(
                self.dependencies(id)
                    .is_some_and(|current| current.iter().copied().eq(dependencies.clone())),
                "one scene-submission backend plan was registered with conflicting dependencies"
            );
            return id;
        }

        let start = u32::try_from(self.dependencies.len())
            .expect("scene-submission dependency table exceeds u32");
        self.dependencies.extend(dependencies);
        let end = u32::try_from(self.dependencies.len())
            .expect("scene-submission dependency table exceeds u32");
        let index = u32::try_from(self.plans.len())
            .expect("one graph attempt cannot reserve more than u32::MAX scene submissions");
        let id = SceneSubmissionPlanId {
            attempt_token: self.attempt_token,
            index,
        };
        self.plans.push(SceneSubmissionRecord {
            backend,
            dependency_start: start,
            dependency_end: end,
            live: false,
        });
        assert!(self.lookup.insert(backend, id).is_none());
        id
    }

    #[must_use]
    pub(super) fn validate_reserved(&self, id: SceneSubmissionPlanId) -> bool {
        id.attempt_token == self.attempt_token && (id.index as usize) < self.plans.len()
    }

    #[must_use]
    pub(super) fn reserved_index(&self, id: SceneSubmissionPlanId) -> Option<usize> {
        self.validate_reserved(id).then_some(id.index as usize)
    }

    #[must_use]
    pub(super) fn plan_count(&self) -> usize {
        self.plans.len()
    }

    #[must_use]
    pub(super) fn dependencies(
        &self,
        id: SceneSubmissionPlanId,
    ) -> Option<&[ErasedResourceNodeId]> {
        let record = self.record(id)?;
        Some(&self.dependencies[record.dependency_start as usize..record.dependency_end as usize])
    }

    pub(super) fn reset_live(&mut self) {
        for plan in &mut self.plans {
            plan.live = false;
        }
    }

    pub(super) fn mark_live(&mut self, id: SceneSubmissionPlanId) {
        self.record_mut(id)
            .expect("scene-submission demand belongs to another graph attempt")
            .live = true;
    }

    #[must_use]
    pub(super) fn is_live(&self, id: SceneSubmissionPlanId) -> bool {
        self.record(id)
            .expect("scene-submission ID belongs to another graph attempt")
            .live
    }

    /// Allocation-free diagnostics projection for one known semantic plan.
    /// Dead plans report no live backend; stale IDs remain distinguishable.
    #[must_use]
    pub(crate) fn live_backend_presence(&self, id: SceneSubmissionPlanId) -> Option<(bool, bool)> {
        let record = self.record(id)?;
        if !record.live {
            return Some((false, false));
        }
        Some((
            record.backend.cpu_direct().is_some(),
            record.backend.gpu_indirect().is_some(),
        ))
    }

    /// Project only the current production backend to the private finalizer.
    pub(super) fn live_cpu_direct_plans(&self) -> impl Iterator<Item = LiveCpuDirectPlan> + '_ {
        let attempt_token = self.attempt_token;
        self.plans
            .iter()
            .enumerate()
            .filter_map(move |(index, record)| {
                if !record.live {
                    return None;
                }
                record
                    .backend
                    .cpu_direct()
                    .map(|draw_plan| LiveCpuDirectPlan {
                        submission: SceneSubmissionPlanId {
                            attempt_token,
                            index: u32::try_from(index).expect("validated plan index exceeds u32"),
                        },
                        draw_plan,
                    })
            })
    }

    /// Project only live GPU-indirect work to its private coordinator.
    ///
    /// The descriptor remains immutable and Copy. No cache, queue, or WGPU
    /// owner is borrowed while the caller gathers materialization recipes.
    #[cfg(test)]
    pub(super) fn live_gpu_indirect_plans(&self) -> impl Iterator<Item = LiveGpuIndirectPlan> + '_ {
        let attempt_token = self.attempt_token;
        self.plans
            .iter()
            .enumerate()
            .filter_map(move |(index, record)| {
                if !record.live {
                    return None;
                }
                let plan = record.backend.gpu_indirect()?;
                let descriptor = self
                    .gpu_indirect_record(plan)
                    .expect("live scene submission retained a stale GPU-indirect plan")
                    .descriptor;
                Some(LiveGpuIndirectPlan {
                    submission: SceneSubmissionPlanId {
                        attempt_token,
                        index: u32::try_from(index).expect("validated plan index exceeds u32"),
                    },
                    plan,
                    descriptor,
                })
            })
    }

    /// Whether DPE retained at least one GPU-indirect semantic plan.
    ///
    /// The coordinator uses this single bit to append the RenderLists-owned
    /// retained representative union exactly once. Representatives are not
    /// copied into per-view or per-bin submission metadata.
    #[must_use]
    pub(crate) fn has_live_gpu_plan(&self) -> bool {
        self.plans
            .iter()
            .any(|record| record.live && record.backend.gpu_indirect().is_some())
    }

    /// Publish one compact immutable physical batch stream transactionally.
    ///
    /// Every deterministic rejection and every fallible capacity reservation
    /// happens before the first flat-storage append. Setting
    /// `finalized_stream` after all records and vertex slots are present is the
    /// sole linearization point that can make the plan executable. Therefore a
    /// rejected member can never expose a prefix of the requested stream. A
    /// cloneable exact-size iterator lets a coordinator project retained batch
    /// and vertex-range scratch directly, without constructing a temporary Vec
    /// containing references back into that scratch. The paired submission ID
    /// addresses its live backend record directly, so publishing many shadow
    /// view plans never rescans the whole plan table.
    pub(crate) fn publish_gpu_indexed_indirect_stream<'a, I>(
        &mut self,
        submission: SceneSubmissionPlanId,
        plan: GpuIndirectPlanId,
        inputs: I,
    ) -> Option<GpuIndexedIndirectStreamRange>
    where
        I: IntoIterator<Item = GpuIndexedIndirectBatchInput<'a>>,
        I::IntoIter: ExactSizeIterator + Clone,
    {
        let inputs = inputs.into_iter();
        if !self.validate_gpu_indirect_plan(plan)
            || !self
                .record(submission)
                .is_some_and(|record| record.live && record.backend.gpu_indirect() == Some(plan))
            || inputs.len() == 0
            || self.gpu_indirect_record(plan)?.finalized_stream.is_some()
        {
            return None;
        }

        let descriptor = self.gpu_indirect_record(plan)?.descriptor;
        let batch_start = u32::try_from(self.gpu_indirect_batches.len()).ok()?;
        let batch_end = self
            .gpu_indirect_batches
            .len()
            .checked_add(inputs.len())
            .and_then(|end| u32::try_from(end).ok())?;
        let mut final_vertex_len = self.gpu_indirect_vertex_buffers.len();
        for input in inputs.clone() {
            if input.vertex_buffers.is_empty()
                || !descriptor.contains_indexed_argument(input.indirect_offset)
            {
                return None;
            }
            final_vertex_len = final_vertex_len.checked_add(input.vertex_buffers.len())?;
            u32::try_from(final_vertex_len).ok()?;
        }

        // Reservation can change retained capacity on an ultimately rejected
        // OOM attempt, but it cannot append command-visible contents. Once
        // both reservations succeed, the mutation pass below is infallible.
        self.gpu_indirect_batches
            .try_reserve_exact(inputs.len())
            .ok()?;
        self.gpu_indirect_vertex_buffers
            .try_reserve_exact(final_vertex_len - self.gpu_indirect_vertex_buffers.len())
            .ok()?;

        for input in inputs {
            let vertex_start = u32::try_from(self.gpu_indirect_vertex_buffers.len())
                .expect("GPU-indirect stream vertex preflight drifted");
            self.gpu_indirect_vertex_buffers
                .extend_from_slice(input.vertex_buffers);
            let vertex_end = u32::try_from(self.gpu_indirect_vertex_buffers.len())
                .expect("GPU-indirect stream vertex preflight drifted");
            self.gpu_indirect_batches
                .push(GpuIndexedIndirectBatchRecord {
                    pipeline: input.pipeline,
                    vertex_start,
                    vertex_end,
                    index_buffer: input.index_buffer,
                    index_format: input.index_format,
                    material_bind_group: input.material_bind_group,
                    object_bind_group: input.object_bind_group,
                    indirect_arguments: input.indirect_arguments,
                    indirect_offset: input.indirect_offset,
                    instance_table_physical: input.instance_table_physical,
                });
        }
        debug_assert_eq!(self.gpu_indirect_batches.len(), batch_end as usize);
        debug_assert_eq!(self.gpu_indirect_vertex_buffers.len(), final_vertex_len);

        let range = GpuIndexedIndirectStreamRange {
            attempt_token: self.attempt_token,
            start: batch_start,
            end: batch_end,
        };
        self.gpu_indirect_plans[plan.index as usize].finalized_stream = Some(range);
        Some(range)
    }

    #[must_use]
    pub(crate) fn live_gpu_plan_for_submission(
        &self,
        submission: SceneSubmissionPlanId,
    ) -> Option<LiveGpuIndirectPlan> {
        let record = self.record(submission)?;
        if !record.live {
            return None;
        }
        let plan = record.backend.gpu_indirect()?;
        Some(LiveGpuIndirectPlan {
            submission,
            plan,
            descriptor: self.gpu_indirect_record(plan)?.descriptor,
        })
    }

    pub(crate) fn assert_all_live_finalized(&self, draw_plans: &DrawPlanStore) {
        for record in &self.plans {
            if !record.live {
                continue;
            }
            let cpu_finalized = record
                .backend
                .cpu_direct()
                .is_none_or(|plan| draw_plans.resolve_plan(plan).is_some());
            let gpu_finalized = record.backend.gpu_indirect().is_none_or(|plan| {
                self.gpu_indirect_record(plan)
                    .and_then(|record| record.finalized_stream)
                    .is_some_and(|stream| self.validate_gpu_indirect_stream(stream))
            });
            assert!(
                cpu_finalized && gpu_finalized,
                "live scene submission was not finalized before execute"
            );
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn backend_counts(&self) -> SceneSubmissionBackendCounts {
        let mut counts = SceneSubmissionBackendCounts::default();
        for record in &self.plans {
            match record.backend {
                SceneSubmissionBackend::CpuDirect(_) => {
                    counts.reserved_cpu_direct += 1;
                    counts.live_cpu_direct += u32::from(record.live);
                }
                SceneSubmissionBackend::GpuIndirect(_) => {
                    counts.reserved_gpu_indirect += 1;
                    counts.live_gpu_indirect += u32::from(record.live);
                }
                SceneSubmissionBackend::Hybrid { .. } => {
                    counts.reserved_hybrid += 1;
                    counts.live_hybrid += u32::from(record.live);
                }
            }
        }
        counts
    }

    /// Resolve a DPE-live, backend-finalized plan to an opaque execute stream.
    #[must_use]
    pub(super) fn resolve_stream(
        &self,
        id: SceneSubmissionPlanId,
        draw_plans: &DrawPlanStore,
    ) -> Option<DrawStreamId> {
        let record = self.record(id)?;
        if !record.live {
            return None;
        }
        let cpu_finalized = record
            .backend
            .cpu_direct()
            .is_none_or(|draw_plan| draw_plans.resolve_plan(draw_plan).is_some());
        let gpu_finalized = record.backend.gpu_indirect().is_none_or(|plan| {
            self.gpu_indirect_record(plan)
                .and_then(|record| record.finalized_stream)
                .is_some_and(|stream| self.validate_gpu_indirect_stream(stream))
        });
        let finalized = cpu_finalized && gpu_finalized;
        finalized.then_some(DrawStreamId {
            attempt_token: self.attempt_token,
            index: id.index,
        })
    }

    #[must_use]
    pub(super) fn validate_stream(&self, id: DrawStreamId) -> bool {
        id.attempt_token == self.attempt_token
            && self
                .plans
                .get(id.index as usize)
                .is_some_and(|plan| plan.live)
    }

    /// Core-only backend resolution. Feature code never receives DrawPlanId.
    #[must_use]
    pub(super) fn cpu_direct_stream(&self, stream: DrawStreamId) -> Option<DrawPlanId> {
        if !self.validate_stream(stream) {
            return None;
        }
        self.plans[stream.index as usize].backend.cpu_direct()
    }

    fn gpu_indirect_stream(&self, stream: DrawStreamId) -> Option<GpuIndexedIndirectStreamRange> {
        if !self.validate_stream(stream) {
            return None;
        }
        let plan = self.plans[stream.index as usize].backend.gpu_indirect()?;
        let range = self.gpu_indirect_record(plan)?.finalized_stream?;
        self.validate_gpu_indirect_stream(range).then_some(range)
    }

    /// Resolve a plan to its private CpuDirect reservation without exposing
    /// the backend tag to Feature code.
    pub(super) fn cpu_direct_plan(&self, plan: SceneSubmissionPlanId) -> Option<DrawPlanId> {
        self.record(plan)?.backend.cpu_direct()
    }

    fn record(&self, id: SceneSubmissionPlanId) -> Option<&SceneSubmissionRecord> {
        self.validate_reserved(id)
            .then(|| self.plans.get(id.index as usize))
            .flatten()
    }

    fn record_mut(&mut self, id: SceneSubmissionPlanId) -> Option<&mut SceneSubmissionRecord> {
        if !self.validate_reserved(id) {
            return None;
        }
        self.plans.get_mut(id.index as usize)
    }

    fn validate_gpu_indirect_plan(&self, id: GpuIndirectPlanId) -> bool {
        id.attempt_token == self.attempt_token
            && (id.index as usize) < self.gpu_indirect_plans.len()
    }

    fn gpu_indirect_record(&self, id: GpuIndirectPlanId) -> Option<&GpuIndirectPlanRecord> {
        self.validate_gpu_indirect_plan(id)
            .then(|| self.gpu_indirect_plans.get(id.index as usize))
            .flatten()
    }

    #[inline]
    fn validate_gpu_indirect_stream(&self, range: GpuIndexedIndirectStreamRange) -> bool {
        range.attempt_token == self.attempt_token
            && range.start < range.end
            && (range.end as usize) <= self.gpu_indirect_batches.len()
    }

    #[cfg(test)]
    #[must_use]
    fn is_empty(&self) -> bool {
        self.plans.is_empty()
            && self.lookup.is_empty()
            && self.cpu_direct_owners.is_empty()
            && self.dependencies.is_empty()
            && self.gpu_indirect_plans.is_empty()
            && self.gpu_indirect_lookup.is_empty()
            && self.gpu_indirect_batches.is_empty()
            && self.gpu_indirect_vertex_buffers.is_empty()
    }
}

/// Immutable execute-time facade over a finalized scene-submission attempt.
///
/// Resolution adds one plan-level tag dispatch, then returns the original
/// DrawPlanStore slice and physical owner slots by reference. No DrawCommand
/// or WGPU owner is copied.
#[derive(Clone, Copy)]
pub(crate) struct SceneSubmissionResolver<'a> {
    submissions: &'a SceneSubmissionStore,
    draw_plan_store: &'a DrawPlanStore,
    draw_plans: DrawPlanResolver<'a>,
    prepared_bind_groups: Option<&'a PreparedBindGroupStore>,
}

impl<'a> SceneSubmissionResolver<'a> {
    #[must_use]
    pub(crate) fn new(
        submissions: &'a SceneSubmissionStore,
        draw_plans: &'a DrawPlanStore,
    ) -> Self {
        Self {
            submissions,
            draw_plan_store: draw_plans,
            draw_plans: draw_plans.resolver(),
            prepared_bind_groups: None,
        }
    }

    #[must_use]
    pub(crate) const fn with_prepared_bind_groups(
        mut self,
        prepared_bind_groups: &'a PreparedBindGroupStore,
    ) -> Self {
        self.prepared_bind_groups = Some(prepared_bind_groups);
        self
    }

    #[inline]
    #[must_use]
    pub(crate) fn stream(self, plan: SceneSubmissionPlanId) -> DrawStreamId {
        self.submissions
            .resolve_stream(plan, self.draw_plan_store)
            .expect("scene-submission plan is dead, stale, or was not finalized")
    }

    #[inline]
    #[must_use]
    pub(crate) fn resolve(self, plan: SceneSubmissionPlanId) -> ResolvedSceneSubmission<'a> {
        let stream = self.stream(plan);
        let cpu_direct = self
            .submissions
            .cpu_direct_stream(stream)
            .map(|draw_plan| self.draw_plans.commands(draw_plan));
        let gpu_indirect =
            self.submissions
                .gpu_indirect_stream(stream)
                .map(|range| GpuIndexedIndirectStream {
                    batches: &self.submissions.gpu_indirect_batches
                        [range.start as usize..range.end as usize],
                    vertex_buffers: &self.submissions.gpu_indirect_vertex_buffers,
                });
        match (cpu_direct, gpu_indirect) {
            (Some(commands), None) => ResolvedSceneSubmission::CpuDirect(commands),
            (None, Some(stream)) => ResolvedSceneSubmission::GpuIndexedIndirect(stream),
            (Some(cpu_direct), Some(gpu_indirect)) => ResolvedSceneSubmission::Hybrid {
                cpu_direct,
                gpu_indirect,
            },
            (None, None) => unreachable!("finalized scene submission has no physical backend"),
        }
    }

    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) fn commands(self, stream: DrawStreamId) -> &'a [DrawCommand] {
        let draw_plan = self
            .submissions
            .cpu_direct_stream(stream)
            .expect("scene-submission stream is not a finalized CpuDirect backend");
        self.draw_plans.commands(draw_plan)
    }

    #[inline]
    #[must_use]
    pub(crate) fn vertex_buffers(self, command: &DrawCommand) -> &'a [DrawBufferSlot] {
        self.draw_plan_store.command_vertex_buffers(command)
    }

    #[inline]
    #[must_use]
    pub(crate) fn buffer(self, slot: DrawBufferSlot) -> &'a Tracked<wgpu::Buffer> {
        self.draw_plans.buffer(slot)
    }

    #[inline]
    #[must_use]
    pub(crate) fn bind_group(self, slot: DrawBindGroupSlot) -> &'a Tracked<wgpu::BindGroup> {
        self.draw_plans.bind_group(slot)
    }

    #[inline]
    #[must_use]
    pub(crate) fn prepared_bind_group(self, id: PreparedBindGroupId) -> &'a wgpu::BindGroup {
        self.prepared_bind_groups
            .and_then(|store| store.resolve(id))
            .expect("GPU scene submission retained a stale prepared bind group")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::frame::DrawPlanKey;

    fn finalized_direct(draw_plans: &mut DrawPlanStore, command: Option<u64>) -> DrawPlanId {
        let draw_plan = draw_plans.reserve_plan(DrawPlanKey::Opaque);
        draw_plans.mark_live(draw_plan);
        let start = draw_plans.begin_finalize(draw_plan);
        if let Some(sort_key) = command {
            draw_plans.push_test_command(sort_key);
        }
        draw_plans.finish_finalize(draw_plan, start);
        draw_plan
    }

    fn test_buffer_slot() -> DrawBufferSlot {
        // SAFETY: `DrawBufferSlot` is an integer-only opaque ID with no invalid
        // bit pattern. Tests never resolve this synthetic slot through a
        // `DrawPlanStore`.
        unsafe { std::mem::zeroed() }
    }

    fn test_bind_group_slot() -> DrawBindGroupSlot {
        // SAFETY: `DrawBindGroupSlot` is an integer-only opaque ID with no
        // invalid bit pattern and is inspected only as copied metadata here.
        unsafe { std::mem::zeroed() }
    }

    fn test_prepared_bind_group() -> PreparedBindGroupId {
        // SAFETY: `PreparedBindGroupId` contains only integer identity fields.
        // Resolver tests do not dereference this synthetic attempt-local ID.
        unsafe { std::mem::zeroed() }
    }

    fn test_physical_token() -> PersistentBufferPhysicalToken {
        // SAFETY: the token contains integer/newtype identity fields only. It
        // is carried through immutable records but never presented to the
        // persistent registry by these unit tests.
        let mut token: PersistentBufferPhysicalToken = unsafe { std::mem::zeroed() };
        token.physical_id = 17;
        token
    }

    fn gpu_batch_input<'a>(
        pipeline: u32,
        indirect_offset: u64,
        vertex_buffers: &'a [DrawBufferSlot],
    ) -> GpuIndexedIndirectBatchInput<'a> {
        GpuIndexedIndirectBatchInput {
            pipeline: RenderPipelineId(pipeline),
            vertex_buffers,
            index_buffer: test_buffer_slot(),
            index_format: wgpu::IndexFormat::Uint32,
            material_bind_group: test_bind_group_slot(),
            object_bind_group: test_prepared_bind_group(),
            indirect_arguments: test_buffer_slot(),
            indirect_offset,
            instance_table_physical: test_physical_token(),
        }
    }

    fn live_gpu_plan(
        store: &mut SceneSubmissionStore,
    ) -> (SceneSubmissionPlanId, GpuIndirectPlanId) {
        let descriptor = test_gpu_descriptor();
        let submission = store.reserve_gpu_indirect(descriptor);
        store.mark_live(submission);
        let plan = store
            .live_gpu_indirect_plans()
            .next()
            .expect("live GPU-indirect projection missing")
            .plan;
        (submission, plan)
    }

    fn test_gpu_descriptor() -> GpuIndirectPlanDescriptor {
        GpuIndirectPlanDescriptor::new_multi_bin(
            SceneSubmissionKey::Opaque,
            FrameSourceId::from_index(0),
            BufferNodeId::new(3, 0),
            BufferNodeId::new(3, 0),
            BufferNodeId::new(7, 0),
            BufferNodeId::new(11, 0),
            0,
            0,
            4 * GpuIndirectPlanDescriptor::INDEXED_ARGUMENT_SIZE,
        )
    }

    #[test]
    fn cpu_direct_state_matrix_preserves_empty_finalization_and_zero_copy() {
        let dependencies = SceneSubmissionDependencies::new();

        let mut draw_plans = DrawPlanStore::new();
        let first_draw = draw_plans.reserve_plan(DrawPlanKey::Opaque);
        let second_draw = draw_plans.reserve_plan(DrawPlanKey::Transparent);
        let mut store = SceneSubmissionStore::new();
        let dead = store.reserve_cpu_direct(&draw_plans, first_draw, &dependencies);
        let live = store.reserve_cpu_direct(&draw_plans, second_draw, &dependencies);

        store.reset_live();
        store.mark_live(live);
        assert!(!store.is_live(dead));
        assert!(store.is_live(live));
        assert_eq!(store.backend_counts().live_cpu_direct, 1);
        assert_eq!(
            store.live_cpu_direct_plans().collect::<Vec<_>>(),
            vec![LiveCpuDirectPlan {
                submission: live,
                draw_plan: second_draw,
            }]
        );
        assert!(store.resolve_stream(dead, &draw_plans).is_none());
        assert!(store.resolve_stream(live, &draw_plans).is_none());

        // A live backend becomes resolvable even when finalization produced an
        // empty direct command slice; deadness and emptiness are independent.
        draw_plans.mark_live(second_draw);
        let start = draw_plans.begin_finalize(second_draw);
        draw_plans.finish_finalize(second_draw, start);
        let stream = store
            .resolve_stream(live, &draw_plans)
            .expect("live empty direct plan must still be finalized");
        assert!(store.validate_stream(stream));
        assert_eq!(store.cpu_direct_stream(stream), Some(second_draw));
        assert!(draw_plans.resolve_plan(second_draw).unwrap().is_empty());

        // Re-reservation is idempotent and does not duplicate the record.
        assert_eq!(
            store.reserve_cpu_direct(&draw_plans, second_draw, &dependencies),
            live
        );
        assert_eq!(store.plan_count(), 2);
        assert_eq!(store.reserved_index(live), Some(1));

        draw_plans.mark_live(first_draw);
        let start = draw_plans.begin_finalize(first_draw);
        draw_plans.push_test_command(73);
        draw_plans.finish_finalize(first_draw, start);
        let original = draw_plans
            .resolve_plan(first_draw)
            .expect("direct plan was finalized");
        let original_ptr = original.as_ptr();
        store.mark_live(dead);
        let resolver = SceneSubmissionResolver::new(&store, &draw_plans);
        let commands = resolver.commands(resolver.stream(dead));
        assert_eq!(commands.as_ptr(), original_ptr);
        assert_eq!(commands.len(), original.len());
        assert_eq!(commands[0].sort_key, 73);
    }

    #[test]
    fn plan_stream_and_gpu_ids_are_attempt_local_and_clear_retains_capacity() {
        let mut draw_plans = DrawPlanStore::new();
        let direct = finalized_direct(&mut draw_plans, None);
        let descriptor = test_gpu_descriptor();
        let vertices = [test_buffer_slot()];
        let mut first = SceneSubmissionStore::new();

        let direct_submission =
            first.reserve_cpu_direct(&draw_plans, direct, &SceneSubmissionDependencies::new());
        first.mark_live(direct_submission);
        let direct_stream = first
            .resolve_stream(direct_submission, &draw_plans)
            .expect("finalized direct stream missing");

        let gpu_submission = first.reserve_gpu_indirect(descriptor);
        assert!(!first.has_live_gpu_plan());
        first.mark_live(gpu_submission);
        assert!(first.has_live_gpu_plan());
        first.reset_live();
        assert!(!first.has_live_gpu_plan());
        first.mark_live(gpu_submission);
        let gpu_plan = first
            .live_gpu_plan_for_submission(gpu_submission)
            .expect("live GPU plan missing")
            .plan;
        let gpu_range = first
            .publish_gpu_indexed_indirect_stream(
                gpu_submission,
                gpu_plan,
                std::iter::once(gpu_batch_input(1, 0, &vertices)),
            )
            .expect("valid GPU stream was rejected");
        let gpu_stream = first
            .resolve_stream(gpu_submission, &draw_plans)
            .expect("published GPU stream did not resolve");

        let capacities = (
            first.plans.capacity(),
            first.dependencies.capacity(),
            first.lookup.capacity(),
            first.gpu_indirect_plans.capacity(),
            first.gpu_indirect_batches.capacity(),
        );
        let second = SceneSubmissionStore::new();
        for submission in [direct_submission, gpu_submission] {
            assert!(!second.validate_reserved(submission));
        }
        for stream in [direct_stream, gpu_stream] {
            assert!(!second.validate_stream(stream));
        }
        assert!(!second.validate_gpu_indirect_plan(gpu_plan));
        assert!(!second.validate_gpu_indirect_stream(gpu_range));

        first.clear();
        assert!(first.is_empty());
        assert_eq!(
            (
                first.plans.capacity(),
                first.dependencies.capacity(),
                first.lookup.capacity(),
                first.gpu_indirect_plans.capacity(),
                first.gpu_indirect_batches.capacity(),
            ),
            capacities
        );
        for submission in [direct_submission, gpu_submission] {
            assert!(!first.validate_reserved(submission));
        }
        for stream in [direct_stream, gpu_stream] {
            assert!(!first.validate_stream(stream));
        }
        assert!(first.cpu_direct_stream(direct_stream).is_none());
        assert!(!first.validate_gpu_indirect_plan(gpu_plan));
        assert!(!first.validate_gpu_indirect_stream(gpu_range));
        assert!(
            first
                .publish_gpu_indexed_indirect_stream(
                    gpu_submission,
                    gpu_plan,
                    std::iter::once(gpu_batch_input(1, 0, &vertices)),
                )
                .is_none()
        );

        let replacement = first.reserve_gpu_indirect(descriptor);
        first.mark_live(replacement);
        let replacement_plan = first
            .live_gpu_plan_for_submission(replacement)
            .expect("replacement GPU plan missing")
            .plan;
        assert_ne!(replacement, gpu_submission);
        assert_ne!(replacement_plan, gpu_plan);
    }

    #[test]
    fn hybrid_reservation_projects_both_backends_from_one_live_semantic_plan() {
        let mut draw_plans = DrawPlanStore::new();
        let direct = draw_plans.reserve_plan(DrawPlanKey::Opaque);
        let descriptor = test_gpu_descriptor();
        let texture = TextureNodeId::new(19, 0);
        let mut store = SceneSubmissionStore::new();
        let hybrid =
            store.reserve_hybrid_with_textures(&draw_plans, direct, descriptor, &[texture]);

        assert_eq!(
            store.dependencies(hybrid),
            Some(
                &[
                    descriptor.instance_table().erase(),
                    descriptor.visible_instances().erase(),
                    descriptor.indirect_arguments().erase(),
                    texture.erase(),
                ][..]
            )
        );
        assert!(!store.has_live_gpu_plan());
        assert!(store.live_cpu_direct_plans().next().is_none());
        assert!(store.live_gpu_indirect_plans().next().is_none());

        store.mark_live(hybrid);
        assert!(store.has_live_gpu_plan());
        assert_eq!(
            store.live_cpu_direct_plans().collect::<Vec<_>>(),
            vec![LiveCpuDirectPlan {
                submission: hybrid,
                draw_plan: direct,
            }]
        );
        let gpu = store
            .live_gpu_indirect_plans()
            .next()
            .expect("live hybrid lost its GPU projection");
        assert_eq!(gpu.submission, hybrid);
        assert_eq!(gpu.descriptor, descriptor);
        let counts = store.backend_counts();
        assert_eq!(counts.reserved_hybrid, 1);
        assert_eq!(counts.live_hybrid, 1);
        assert_eq!(counts.reserved_cpu_direct, 0);
        assert_eq!(counts.reserved_gpu_indirect, 0);
    }

    #[test]
    fn one_physical_plan_cannot_be_wrapped_by_pure_and_hybrid_backends() {
        for claim_gpu_first in [false, true] {
            let mut draw_plans = DrawPlanStore::new();
            let direct = draw_plans.reserve_plan(DrawPlanKey::Opaque);
            let descriptor = test_gpu_descriptor();
            let mut store = SceneSubmissionStore::new();
            if claim_gpu_first {
                store.reserve_gpu_indirect(descriptor);
            } else {
                store.reserve_cpu_direct(&draw_plans, direct, &SceneSubmissionDependencies::new());
            }

            let conflict = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                store.reserve_hybrid_with_textures(&draw_plans, direct, descriptor, &[]);
            }));
            assert!(
                conflict.is_err(),
                "{} physical plan accepted a second backend owner",
                if claim_gpu_first { "GPU" } else { "CPU" }
            );
        }
    }

    #[test]
    fn hybrid_stream_is_fail_closed_until_both_sides_finalize_then_resolves_zero_copy() {
        let mut draw_plans = DrawPlanStore::new();
        let direct = draw_plans.reserve_plan(DrawPlanKey::Opaque);
        let descriptor = test_gpu_descriptor();
        let mut store = SceneSubmissionStore::new();
        let hybrid = store.reserve_hybrid_with_textures(&draw_plans, direct, descriptor, &[]);
        store.mark_live(hybrid);
        draw_plans.mark_live(direct);

        let gpu = store
            .live_gpu_plan_for_submission(hybrid)
            .expect("live hybrid lost its GPU plan");
        let vertices = [test_buffer_slot()];
        store
            .publish_gpu_indexed_indirect_stream(
                hybrid,
                gpu.plan,
                std::iter::once(gpu_batch_input(7, 0, &vertices)),
            )
            .expect("hybrid GPU stream publication failed");
        assert!(
            store.resolve_stream(hybrid, &draw_plans).is_none(),
            "a hybrid GPU prefix must not become executable before CpuDirect finalization"
        );

        let start = draw_plans.begin_finalize(direct);
        draw_plans.push_test_command(91);
        draw_plans.finish_finalize(direct, start);
        let direct_commands = draw_plans.resolve_plan(direct).unwrap();
        let direct_ptr = direct_commands.as_ptr();
        let stream = store
            .resolve_stream(hybrid, &draw_plans)
            .expect("fully finalized hybrid stream did not resolve");
        assert!(store.validate_stream(stream));

        let resolver = SceneSubmissionResolver::new(&store, &draw_plans);
        let ResolvedSceneSubmission::Hybrid {
            cpu_direct,
            gpu_indirect,
        } = resolver.resolve(hybrid)
        else {
            panic!("hybrid semantic plan resolved as one physical backend");
        };
        assert_eq!(cpu_direct.as_ptr(), direct_ptr);
        assert_eq!(cpu_direct.len(), 1);
        assert_eq!(cpu_direct[0].sort_key, 91);
        assert_eq!(gpu_indirect.len(), 1);
        assert_eq!(
            gpu_indirect.batches().next().unwrap().pipeline,
            RenderPipelineId(7)
        );

        // The inverse order must obey the same atomic state machine: a
        // finalized CPU suffix cannot expose a stream before GPU publication.
        let mut draw_plans = DrawPlanStore::new();
        let direct = draw_plans.reserve_plan(DrawPlanKey::Opaque);
        let mut store = SceneSubmissionStore::new();
        let hybrid = store.reserve_hybrid_with_textures(&draw_plans, direct, descriptor, &[]);
        store.mark_live(hybrid);
        draw_plans.mark_live(direct);
        let start = draw_plans.begin_finalize(direct);
        draw_plans.finish_finalize(direct, start);
        assert!(store.resolve_stream(hybrid, &draw_plans).is_none());

        let gpu = store
            .live_gpu_plan_for_submission(hybrid)
            .expect("live hybrid lost its GPU plan");
        store
            .publish_gpu_indexed_indirect_stream(
                hybrid,
                gpu.plan,
                std::iter::once(gpu_batch_input(8, 0, &vertices)),
            )
            .expect("hybrid GPU stream publication failed");
        assert!(store.resolve_stream(hybrid, &draw_plans).is_some());
    }

    #[test]
    fn hybrid_stream_rejects_gpu_publication_while_dpe_dead() {
        let mut draw_plans = DrawPlanStore::new();
        let direct = draw_plans.reserve_plan(DrawPlanKey::Opaque);
        let descriptor = test_gpu_descriptor();
        let mut store = SceneSubmissionStore::new();
        let hybrid = store.reserve_hybrid_with_textures(&draw_plans, direct, descriptor, &[]);
        let gpu_plan = store
            .record(hybrid)
            .and_then(|record| record.backend.gpu_indirect())
            .expect("hybrid lost its private GPU plan");
        let vertices = [test_buffer_slot()];
        let input = [gpu_batch_input(3, 0, &vertices)];

        assert!(
            store
                .publish_gpu_indexed_indirect_stream(hybrid, gpu_plan, input.into_iter())
                .is_none(),
            "DPE-dead hybrid GPU work must reject publication"
        );
        assert!(store.gpu_indirect_batches.is_empty());
        assert!(store.live_cpu_direct_plans().next().is_none());
        assert!(store.live_gpu_indirect_plans().next().is_none());
    }

    #[test]
    fn gpu_descriptor_is_idempotent_but_conflicting_identity_is_rejected() {
        let candidates = BufferNodeId::new(3, 0);
        let visible = BufferNodeId::new(7, 1);
        let arguments = BufferNodeId::new(11, 2);
        let descriptor = GpuIndirectPlanDescriptor::new(
            SceneSubmissionKey::Opaque,
            candidates,
            visible,
            arguments,
        );
        let draw_plans = DrawPlanStore::new();
        let mut store = SceneSubmissionStore::new();
        let indirect = store.reserve_gpu_indirect(descriptor);

        assert_eq!(
            store.dependencies(indirect),
            Some(&[candidates.erase(), visible.erase(), arguments.erase()][..])
        );
        assert_eq!(store.reserve_gpu_indirect(descriptor), indirect);
        assert_eq!(store.plan_count(), 1);

        store.mark_live(indirect);
        assert_eq!(store.backend_counts().live_gpu_indirect, 1);
        let live = store
            .live_gpu_indirect_plans()
            .next()
            .expect("live GPU-indirect projection missing");
        assert_eq!(live.submission, indirect);
        assert_eq!(live.descriptor, descriptor);
        assert_eq!(
            live.descriptor.indirect_span(),
            GpuIndirectPlanDescriptor::INDEXED_ARGUMENT_SIZE,
            "the compatibility descriptor must address exactly one record"
        );
        assert!(
            store.resolve_stream(indirect, &draw_plans).is_none(),
            "M7.0 must not expose a GPU-indirect plan as a Direct draw stream"
        );

        let conflict = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store.reserve_gpu_indirect(GpuIndirectPlanDescriptor::new(
                SceneSubmissionKey::Opaque,
                candidates,
                visible,
                BufferNodeId::new(13, 2),
            ));
        }));
        assert!(conflict.is_err(), "conflicting descriptor was accepted");
    }

    #[test]
    fn gpu_publication_validation_matrix_is_atomic_and_retryable() {
        #[derive(Clone, Copy, Debug)]
        enum InvalidPublication {
            Empty,
            MismatchedPlan,
            InvalidMember,
            OutsideReservedWindow,
        }

        for case in [
            InvalidPublication::Empty,
            InvalidPublication::MismatchedPlan,
            InvalidPublication::InvalidMember,
            InvalidPublication::OutsideReservedWindow,
        ] {
            let mut store = SceneSubmissionStore::new();
            let (submission, plan) = live_gpu_plan(&mut store);
            let mismatched_plan = if matches!(case, InvalidPublication::MismatchedPlan) {
                let other = store.reserve_gpu_indirect(GpuIndirectPlanDescriptor::new(
                    SceneSubmissionKey::Shadow {
                        light_id: 7,
                        layer_index: 0,
                    },
                    BufferNodeId::new(13, 0),
                    BufferNodeId::new(17, 0),
                    BufferNodeId::new(19, 0),
                ));
                store.mark_live(other);
                Some(
                    store
                        .live_gpu_plan_for_submission(other)
                        .expect("second live GPU plan missing")
                        .plan,
                )
            } else {
                None
            };
            let vertices = [test_buffer_slot()];
            let capacities = (
                store.gpu_indirect_batches.capacity(),
                store.gpu_indirect_vertex_buffers.capacity(),
            );

            let rejected = match case {
                InvalidPublication::Empty => store.publish_gpu_indexed_indirect_stream(
                    submission,
                    plan,
                    std::iter::empty::<GpuIndexedIndirectBatchInput<'_>>(),
                ),
                InvalidPublication::MismatchedPlan => store.publish_gpu_indexed_indirect_stream(
                    submission,
                    mismatched_plan.expect("mismatched plan missing"),
                    std::iter::once(gpu_batch_input(1, 0, &vertices)),
                ),
                InvalidPublication::InvalidMember => store.publish_gpu_indexed_indirect_stream(
                    submission,
                    plan,
                    [
                        gpu_batch_input(1, 0, &vertices),
                        gpu_batch_input(2, 2, &vertices),
                    ]
                    .into_iter(),
                ),
                InvalidPublication::OutsideReservedWindow => store
                    .publish_gpu_indexed_indirect_stream(
                        submission,
                        plan,
                        std::iter::once(gpu_batch_input(
                            1,
                            4 * GpuIndirectPlanDescriptor::INDEXED_ARGUMENT_SIZE,
                            &vertices,
                        )),
                    ),
            };

            assert!(rejected.is_none(), "{case:?} publication was accepted");
            assert!(store.gpu_indirect_batches.is_empty(), "{case:?}");
            assert!(store.gpu_indirect_vertex_buffers.is_empty(), "{case:?}");
            assert_eq!(
                (
                    store.gpu_indirect_batches.capacity(),
                    store.gpu_indirect_vertex_buffers.capacity(),
                ),
                capacities,
                "{case:?} changed retained capacity"
            );
            for rejected_plan in [Some(plan), mismatched_plan].into_iter().flatten() {
                assert!(
                    store
                        .gpu_indirect_record(rejected_plan)
                        .expect("validated plan disappeared")
                        .finalized_stream
                        .is_none(),
                    "{case:?} exposed a partial stream"
                );
            }

            let valid = [
                gpu_batch_input(1, 0, &vertices),
                gpu_batch_input(
                    2,
                    GpuIndirectPlanDescriptor::INDEXED_ARGUMENT_SIZE,
                    &vertices,
                ),
            ];
            assert!(
                store
                    .publish_gpu_indexed_indirect_stream(submission, plan, valid.into_iter())
                    .is_some(),
                "{case:?} made the valid plan non-retryable"
            );
            assert_eq!(store.gpu_indirect_batches.len(), 2, "{case:?}");
            assert_eq!(store.gpu_indirect_vertex_buffers.len(), 2, "{case:?}");
        }
    }

    #[test]
    fn gpu_multibatch_mapped_flat_scratch_preserves_order_without_a_temporary_vec() {
        #[derive(Clone, Copy)]
        struct ScratchBatch {
            pipeline: u32,
            indirect_offset: u64,
            vertex_start: usize,
            vertex_end: usize,
        }

        let mut store = SceneSubmissionStore::new();
        let (submission, plan) = live_gpu_plan(&mut store);
        let flat_vertices = [test_buffer_slot(), test_buffer_slot(), test_buffer_slot()];
        let batches = [
            ScratchBatch {
                pipeline: 3,
                indirect_offset: 0,
                vertex_start: 0,
                vertex_end: 1,
            },
            ScratchBatch {
                pipeline: 8,
                indirect_offset: 40,
                vertex_start: 1,
                vertex_end: 3,
            },
        ];
        let inputs = batches.iter().map(|batch| {
            gpu_batch_input(
                batch.pipeline,
                batch.indirect_offset,
                &flat_vertices[batch.vertex_start..batch.vertex_end],
            )
        });
        let range = store
            .publish_gpu_indexed_indirect_stream(submission, plan, inputs)
            .expect("mapped retained scratch stream was rejected");
        assert_eq!(range.end - range.start, 2);
        assert_eq!(store.gpu_indirect_batches.len(), 2);
        assert_eq!(store.gpu_indirect_vertex_buffers.len(), 3);

        let draw_plans = DrawPlanStore::new();
        let resolver = SceneSubmissionResolver::new(&store, &draw_plans);
        let ResolvedSceneSubmission::GpuIndexedIndirect(stream) = resolver.resolve(submission)
        else {
            panic!("GPU plan resolved as CPU direct");
        };
        let resolved = stream.batches().collect::<Vec<_>>();
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].pipeline, RenderPipelineId(3));
        assert_eq!(resolved[0].indirect_offset, 0);
        assert_eq!(resolved[0].vertex_buffers.len(), 1);
        assert_eq!(resolved[1].pipeline, RenderPipelineId(8));
        assert_eq!(resolved[1].indirect_offset, 40);
        assert_eq!(resolved[1].vertex_buffers.len(), 2);
    }

    #[test]
    fn gpu_indirect_tables_retain_then_trim_their_high_water_capacity() {
        let mut store = SceneSubmissionStore::new();
        for index in 0..128_u32 {
            store.reserve_gpu_indirect(GpuIndirectPlanDescriptor::new(
                SceneSubmissionKey::Shadow {
                    light_id: u64::from(index),
                    layer_index: index,
                },
                BufferNodeId::new(index * 3, 0),
                BufferNodeId::new(index * 3 + 1, 0),
                BufferNodeId::new(index * 3 + 2, 0),
            ));
        }
        let plan_capacity = store.gpu_indirect_plans.capacity();
        let lookup_capacity = store.gpu_indirect_lookup.capacity();
        let retained_bytes = store.capacity_usage().retained_bytes;
        assert!(plan_capacity >= 128);
        assert!(lookup_capacity >= 128);

        store.clear();
        assert_eq!(store.gpu_indirect_plans.capacity(), plan_capacity);
        assert_eq!(store.gpu_indirect_lookup.capacity(), lookup_capacity);

        store.trim_capacity(0, retained_bytes);
        assert!(store.gpu_indirect_plans.capacity() < plan_capacity);
        assert!(store.gpu_indirect_lookup.capacity() < lookup_capacity);
    }
}
