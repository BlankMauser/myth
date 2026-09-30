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

use crate::core::gpu::Tracked;
use crate::graph::frame::{
    DrawBindGroupSlot, DrawBufferSlot, DrawCommand, DrawPlanId, DrawPlanKey, DrawPlanResolver,
    DrawPlanStore, PreparedDrawSource,
};
use crate::pipeline::RenderPipelineId;

#[cfg(test)]
use super::types::BufferNodeId;
use super::types::{ErasedResourceNodeId, TextureNodeId};

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

/// Typed logical dependency builder for every scene-submission backend.
///
/// Texture and Buffer handles are erased only after their resource class was
/// checked by the typed method. Registration order and duplicates are
/// preserved; GraphStorage's one-shot demand finalizer owns deduplication.
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
    pub(crate) fn with_texture(mut self, texture: TextureNodeId) -> Self {
        self.read_texture(texture);
        self
    }

    #[must_use]
    pub(crate) fn with_buffer(mut self, buffer: BufferNodeId) -> Self {
        self.read_buffer(buffer);
        self
    }

    pub(crate) fn read_texture(&mut self, texture: TextureNodeId) {
        self.reads.push(texture.erase());
    }

    pub(crate) fn read_buffer(&mut self, buffer: BufferNodeId) {
        self.reads.push(buffer.erase());
    }

    #[must_use]
    pub(crate) fn as_slice(&self) -> &[ErasedResourceNodeId] {
        &self.reads
    }
}

/// Backend selection is deliberately private. Features cannot match or branch
/// on it; only the graph coordinator may project a plan to backend work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SceneSubmissionBackend {
    CpuDirect(DrawPlanId),
    #[cfg(test)]
    IndirectStub(u64),
}

#[derive(Debug, Clone, Copy)]
struct SceneSubmissionRecord {
    backend: SceneSubmissionBackend,
    dependency_start: u32,
    dependency_end: u32,
    live: bool,
    #[cfg(test)]
    indirect_finalized: bool,
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) struct LiveSceneSubmissionCounts {
    pub(super) cpu_direct: usize,
    #[cfg(test)]
    pub(super) indirect_stub: usize,
}

/// Capacity-retaining plan registry for one GraphStorage attempt.
pub(crate) struct SceneSubmissionStore {
    attempt_token: u64,
    plans: Vec<SceneSubmissionRecord>,
    lookup: FxHashMap<SceneSubmissionBackend, SceneSubmissionPlanId>,
    dependencies: Vec<ErasedResourceNodeId>,
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
            dependencies: Vec::with_capacity(64),
        }
    }

    /// Release attempt records while retaining every backing allocation.
    pub(super) fn clear(&mut self) {
        self.plans.clear();
        self.lookup.clear();
        self.dependencies.clear();
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
        self.reserve_backend(
            SceneSubmissionBackend::CpuDirect(draw_plan),
            dependencies.as_slice().iter().copied(),
        )
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
        self.reserve_backend(
            SceneSubmissionBackend::CpuDirect(draw_plan),
            dependencies.iter().map(|dependency| dependency.erase()),
        )
    }

    #[cfg(test)]
    pub(super) fn reserve_indirect_stub(
        &mut self,
        stub_key: u64,
        dependencies: &SceneSubmissionDependencies,
    ) -> SceneSubmissionPlanId {
        self.reserve_backend(
            SceneSubmissionBackend::IndirectStub(stub_key),
            dependencies.as_slice().iter().copied(),
        )
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
            #[cfg(test)]
            indirect_finalized: false,
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
                match record.backend {
                    SceneSubmissionBackend::CpuDirect(draw_plan) => Some(LiveCpuDirectPlan {
                        submission: SceneSubmissionPlanId {
                            attempt_token,
                            index: u32::try_from(index).expect("validated plan index exceeds u32"),
                        },
                        draw_plan,
                    }),
                    #[cfg(test)]
                    SceneSubmissionBackend::IndirectStub(_) => None,
                }
            })
    }

    #[must_use]
    #[cfg(test)]
    pub(super) fn live_backend_counts(&self) -> LiveSceneSubmissionCounts {
        let mut counts = LiveSceneSubmissionCounts::default();
        for record in self.plans.iter().filter(|record| record.live) {
            match record.backend {
                SceneSubmissionBackend::CpuDirect(_) => counts.cpu_direct += 1,
                #[cfg(test)]
                SceneSubmissionBackend::IndirectStub(_) => counts.indirect_stub += 1,
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
        let finalized = match record.backend {
            SceneSubmissionBackend::CpuDirect(draw_plan) => {
                draw_plans.resolve_plan(draw_plan).is_some()
            }
            #[cfg(test)]
            SceneSubmissionBackend::IndirectStub(_) => record.indirect_finalized,
        };
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
        match self.plans[stream.index as usize].backend {
            SceneSubmissionBackend::CpuDirect(draw_plan) => Some(draw_plan),
            #[cfg(test)]
            SceneSubmissionBackend::IndirectStub(_) => None,
        }
    }

    /// Resolve a plan to its private CpuDirect reservation without exposing
    /// the backend tag to Feature code.
    fn cpu_direct_plan(&self, plan: SceneSubmissionPlanId) -> Option<DrawPlanId> {
        match self.record(plan)?.backend {
            SceneSubmissionBackend::CpuDirect(draw_plan) => Some(draw_plan),
            #[cfg(test)]
            SceneSubmissionBackend::IndirectStub(_) => None,
        }
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

    #[cfg(test)]
    #[must_use]
    fn is_empty(&self) -> bool {
        self.plans.is_empty() && self.lookup.is_empty() && self.dependencies.is_empty()
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
        }
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
    pub(crate) fn commands(self, stream: DrawStreamId) -> &'a [DrawCommand] {
        let draw_plan = self
            .submissions
            .cpu_direct_stream(stream)
            .expect("scene-submission stream is not a finalized CpuDirect backend");
        self.draw_plans.commands(draw_plan)
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
}

/// Exclusive prepare-time writer for the current CPU-direct backend.
///
/// Feature nodes never receive the underlying [`DrawPlanId`], command range,
/// or physical resource slots. They append already-prepared sources through
/// this narrow facade and finish one contiguous stream. This preserves the
/// existing zero-copy execute path while leaving backend selection private to
/// the scene-submission coordinator.
pub(crate) struct SceneSubmissionWriter<'a> {
    draw_plans: &'a mut DrawPlanStore,
    draw_plan: DrawPlanId,
    start: u32,
}

impl<'a> SceneSubmissionWriter<'a> {
    /// Open an unfinalized live submission. `None` means another prepare path
    /// already finalized the canonical plan and no duplicate work is needed.
    #[must_use]
    pub(super) fn begin(
        submissions: &SceneSubmissionStore,
        draw_plans: &'a mut DrawPlanStore,
        plan: SceneSubmissionPlanId,
        expected_commands: usize,
    ) -> Option<Self> {
        let draw_plan = submissions
            .cpu_direct_plan(plan)
            .expect("scene-submission writer requires a live CpuDirect plan");
        if draw_plans.resolve_plan(draw_plan).is_some() {
            return None;
        }
        draw_plans.reserve_commands(expected_commands);
        let start = draw_plans.begin_finalize(draw_plan);
        Some(Self {
            draw_plans,
            draw_plan,
            start,
        })
    }

    /// Append one prepared source without exposing command/slot internals to
    /// the Feature. No physical owner or command payload is cloned here beyond
    /// the exact compact fields required by the direct backend.
    pub(crate) fn push_prepared_source(
        &mut self,
        source_item_index: u32,
        source: &PreparedDrawSource,
        pipeline_id: RenderPipelineId,
        sort_key: u64,
        stencil_reference: Option<u32>,
    ) {
        crate::graph::bake::push_prepared_source_command(
            self.draw_plans,
            source_item_index,
            source,
            pipeline_id,
            sort_key,
            stencil_reference,
        );
    }

    /// Sort the direct stream by its precomputed key and publish its exact
    /// contiguous command range. An empty range is a valid finalized stream.
    pub(crate) fn finish_sorted(self) {
        self.draw_plans.sort_commands_from(self.start);
        self.draw_plans.finish_finalize(self.draw_plan, self.start);
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

    #[test]
    fn direct_resolver_returns_the_original_command_slice() {
        let mut draw_plans = DrawPlanStore::new();
        let draw_plan = finalized_direct(&mut draw_plans, Some(73));
        let original = draw_plans
            .resolve_plan(draw_plan)
            .expect("direct plan was finalized");
        let original_ptr = original.as_ptr();

        let mut store = SceneSubmissionStore::new();
        let plan =
            store.reserve_cpu_direct(&draw_plans, draw_plan, &SceneSubmissionDependencies::new());
        store.mark_live(plan);
        let resolver = SceneSubmissionResolver::new(&store, &draw_plans);
        let stream = resolver.stream(plan);
        let resolved = resolver.commands(stream);

        assert_eq!(resolved.as_ptr(), original_ptr);
        assert_eq!(resolved.len(), original.len());
        assert_eq!(resolved[0].sort_key, 73);
    }

    #[test]
    fn stale_and_foreign_plan_and_stream_ids_fail_closed() {
        let mut draw_plans = DrawPlanStore::new();
        let draw_plan = finalized_direct(&mut draw_plans, None);
        let mut first = SceneSubmissionStore::new();
        let dependencies = SceneSubmissionDependencies::new();
        let plan = first.reserve_cpu_direct(&draw_plans, draw_plan, &dependencies);
        first.mark_live(plan);
        let stream = first
            .resolve_stream(plan, &draw_plans)
            .expect("finalized direct stream missing");
        let plan_capacity = first.plans.capacity();
        let dependency_capacity = first.dependencies.capacity();
        let lookup_capacity = first.lookup.capacity();

        let second = SceneSubmissionStore::new();
        assert!(!second.validate_reserved(plan));
        assert!(!second.validate_stream(stream));

        first.clear();
        assert!(first.is_empty());
        assert_eq!(first.plans.capacity(), plan_capacity);
        assert_eq!(first.dependencies.capacity(), dependency_capacity);
        assert_eq!(first.lookup.capacity(), lookup_capacity);
        assert!(!first.validate_reserved(plan));
        assert!(!first.validate_stream(stream));
        assert!(first.cpu_direct_stream(stream).is_none());
    }

    #[test]
    fn dead_plan_has_no_stream_and_live_projection_is_exact() {
        let mut draw_plans = DrawPlanStore::new();
        let first_draw = draw_plans.reserve_plan(DrawPlanKey::Opaque);
        let second_draw = draw_plans.reserve_plan(DrawPlanKey::Transparent);
        let mut store = SceneSubmissionStore::new();
        let dependencies = SceneSubmissionDependencies::new();
        let dead = store.reserve_cpu_direct(&draw_plans, first_draw, &dependencies);
        let live = store.reserve_cpu_direct(&draw_plans, second_draw, &dependencies);

        store.reset_live();
        store.mark_live(live);
        assert!(!store.is_live(dead));
        assert!(store.is_live(live));
        assert_eq!(store.live_backend_counts().cpu_direct, 1);
        assert_eq!(
            store.live_cpu_direct_plans().collect::<Vec<_>>(),
            vec![LiveCpuDirectPlan {
                submission: live,
                draw_plan: second_draw,
            }]
        );
        assert!(store.resolve_stream(dead, &draw_plans).is_none());
        assert!(store.resolve_stream(live, &draw_plans).is_none());
    }

    #[test]
    fn finalized_empty_cpu_direct_plan_resolves_opaque_stream() {
        let mut draw_plans = DrawPlanStore::new();
        let draw_plan = finalized_direct(&mut draw_plans, None);
        let mut store = SceneSubmissionStore::new();
        let plan =
            store.reserve_cpu_direct(&draw_plans, draw_plan, &SceneSubmissionDependencies::new());
        store.mark_live(plan);

        let stream = store
            .resolve_stream(plan, &draw_plans)
            .expect("live empty direct plan must still be finalized");
        assert!(store.validate_stream(stream));
        assert_eq!(store.cpu_direct_stream(stream), Some(draw_plan));

        // Re-reservation is idempotent and does not duplicate the record.
        assert_eq!(
            store.reserve_cpu_direct(&draw_plans, draw_plan, &SceneSubmissionDependencies::new(),),
            plan
        );
        assert_eq!(store.plan_count(), 1);
        assert_eq!(store.reserved_index(plan), Some(0));
    }
}
