//! Transactional cross-frame persistent buffers.
//!
//! This registry deliberately separates a logical request from its physical
//! WGPU allocation and from publication of newly written contents:
//!
//! 1. [`PersistentBufferRegistry::plan_buffer`] records the exact desired
//!    specification and revision without performing GPU work.
//! 2. [`PersistentBufferRegistry::materialize`] creates or reuses a physical
//!    allocation only for a live plan.
//! 3. [`PersistentBufferRegistry::commit_batch`] publishes a successfully
//!    submitted batch atomically after validating every logical and physical
//!    token.
//!
//! A changed revision or specification invalidates the previous publication
//! immediately. There is intentionally no rollback path: an aborted refresh
//! remains invalid and must be produced by a later live attempt. Physical
//! allocations additionally retain whether complete contents were ever
//! published for the current exact specification, so an aborted create-and-
//! swap cannot be mistaken for an initialized delta base on retry.

use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::core::gpu::Tracked;

use super::types::BufferDesc;

static NEXT_PERSISTENT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// Stable identity of one logical persistent buffer.
///
/// IDs are process-wide monotonic and are never reused by the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PersistentBufferId(u64);

impl PersistentBufferId {
    fn new() -> Self {
        let value = NEXT_PERSISTENT_BUFFER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("persistent-buffer ID space exhausted");
        Self(value)
    }

    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

/// Complete logical and physical contract for one persistent buffer.
///
/// `schema_id` identifies the record ABI, while `stride` makes the exact
/// element layout compile-visible even when two schemas happen to request the
/// same byte size and usages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PersistentBufferSpec {
    pub(crate) desc: BufferDesc,
    pub(crate) schema_id: u64,
    pub(crate) stride: u64,
}

impl PersistentBufferSpec {
    #[inline]
    #[must_use]
    pub(crate) const fn new(desc: BufferDesc, schema_id: u64, stride: u64) -> Self {
        assert!(
            stride != 0,
            "persistent Buffer record stride must be non-zero"
        );
        assert!(
            desc.logical_size.is_multiple_of(stride),
            "persistent Buffer logical size must be an exact record multiple"
        );
        Self {
            desc,
            schema_id,
            stride,
        }
    }
}

/// Exact logical target selected by one planning attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PersistentBufferRef {
    pub(crate) id: PersistentBufferId,
    pub(crate) spec: PersistentBufferSpec,
    pub(crate) epoch: u64,
    pub(crate) plan_generation: u64,
    pub(crate) revision: u64,
    pub(crate) attempt_serial: u64,
}

/// Capability to publish one invalid logical target after successful submit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PersistentBufferCommitToken {
    pub(crate) buffer_id: PersistentBufferId,
    pub(crate) epoch: u64,
    pub(crate) plan_generation: u64,
    pub(crate) revision: u64,
    pub(crate) attempt_serial: u64,
}

/// One attempt's immutable logical plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PersistentBufferPlan {
    pub(crate) buffer: PersistentBufferRef,
    /// Whether contents for this exact specification and revision were
    /// successfully submitted before this planning attempt began.
    pub(crate) valid: bool,
    /// Present only when the plan needs a new publication.
    pub(crate) commit: Option<PersistentBufferCommitToken>,
}

/// One member requested from an atomic coherent planning batch.
///
/// The attempt serial and logical revision are supplied once by the batch so
/// members cannot accidentally diverge while still appearing structurally
/// related to their owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PersistentBufferPlanRequest {
    pub(crate) id: PersistentBufferId,
    pub(crate) spec: PersistentBufferSpec,
}

impl PersistentBufferPlanRequest {
    #[inline]
    #[must_use]
    pub(crate) const fn new(id: PersistentBufferId, spec: PersistentBufferSpec) -> Self {
        Self { id, spec }
    }
}

/// Exact identity of a materialized physical allocation.
///
/// The logical coordinates prevent an allocation observed for one plan from
/// being reused as authority after another attempt, revision, or specification
/// supersedes it. `physical_id` additionally rejects create-and-swap ABA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PersistentBufferPhysicalToken {
    pub(crate) buffer_id: PersistentBufferId,
    pub(crate) epoch: u64,
    pub(crate) plan_generation: u64,
    pub(crate) revision: u64,
    pub(crate) attempt_serial: u64,
    pub(crate) physical_id: u64,
}

/// Result of materializing one live persistent-buffer plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PersistentBufferMaterialization {
    pub(crate) token: PersistentBufferPhysicalToken,
    pub(crate) recreated: bool,
    /// The current physical allocation has never published a complete image
    /// of this exact schema/stride/description.
    ///
    /// Unlike `recreated`, this bit is retained by the registry across graph
    /// attempts. A patch owner must therefore rebuild the complete logical
    /// contents while it is set, even when an aborted create-and-swap is
    /// reused by a later attempt.
    pub(crate) requires_full_initialization: bool,
    pub(crate) allocation_size: u64,
}

/// Deterministic allocation recipe validated before a graph materializes its
/// first live persistent Buffer.
///
/// The fields stay private so callers cannot bypass exact logical identity or
/// the device's maximum Buffer size. Materialization still revalidates the
/// logical token, but does not recompute capacity or query device limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PersistentBufferMaterializationPreflight {
    buffer: PersistentBufferRef,
    allocation_size: u64,
}

/// One member of an atomic publication batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PersistentBufferCommitRequest {
    pub(crate) commit: PersistentBufferCommitToken,
    pub(crate) physical: PersistentBufferPhysicalToken,
}

impl PersistentBufferCommitRequest {
    #[inline]
    #[must_use]
    pub(crate) const fn new(
        commit: PersistentBufferCommitToken,
        physical: PersistentBufferPhysicalToken,
    ) -> Self {
        Self { commit, physical }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PersistentBufferPlanFailure {
    UnknownOrRetired,
    AttemptRegressed,
    RevisionRegressed,
    SameAttemptConflict,
    EpochOverflow,
    PlanGenerationOverflow,
}

/// Failure to plan an all-valid or all-invalid persistent-Buffer batch.
///
/// Every variant is detected by a read-only preflight. The registry is
/// therefore byte-for-byte logically unchanged when this error is returned,
/// including the physical allocation's full-initialization latch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PersistentBufferBatchPlanFailure {
    EmptyBatch,
    DuplicateBuffer(PersistentBufferId),
    Member {
        buffer_id: PersistentBufferId,
        failure: PersistentBufferPlanFailure,
    },
    MixedPublicationState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PersistentBufferMaterializationFailure {
    UnknownOrRetired,
    StalePlan,
    CapacityOverflow,
    ExceedsDeviceLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PersistentBufferCommitFailure {
    EmptyBatch,
    DuplicateBuffer,
    InvalidSubmissionReceipt,
    PartialLiveGroup,
    LiveGroupNotEncoded,
    FullInitializationRequired,
    UnknownOrRetired,
    StaleLogicalToken,
    LogicalPhysicalMismatch,
    StalePhysicalToken,
    AlreadyPublished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum PersistentBufferRetireFailure {
    UnknownOrRetired,
    StalePhysicalToken,
}

struct PersistentBufferPhysical {
    usage: wgpu::BufferUsages,
    allocation_size: u64,
    /// Whether an exact submitted commit has established complete contents
    /// for the current logical specification.
    contents_initialized: bool,
    buffer: Tracked<wgpu::Buffer>,
}

#[derive(Default)]
struct PersistentBufferRecord {
    epoch: u64,
    plan_generation: u64,
    desired_spec: Option<PersistentBufferSpec>,
    desired_revision: Option<u64>,
    published_revision: Option<u64>,
    last_planned_attempt: Option<u64>,
    last_plan_valid: bool,
    physical: Option<PersistentBufferPhysical>,
}

/// Read-only recipe for one logical planning mutation.
///
/// `apply` is false for an idempotent exact same-attempt query. All fallible
/// arithmetic and state validation has completed before this recipe exists.
#[derive(Debug, Clone, Copy)]
struct PersistentBufferPlanTransition {
    plan: PersistentBufferPlan,
    apply: bool,
    target_changed: bool,
    spec_changed: bool,
}

/// Renderer-owned state machine for logical and physical persistent buffers.
#[derive(Default)]
pub(crate) struct PersistentBufferRegistry {
    records: FxHashMap<PersistentBufferId, PersistentBufferRecord>,
}

/// Frame-scoped exact resolver for persistent-buffer imports.
///
/// Graph storage owns the token map. Consequently a graph can resolve only
/// the physical generation that its live materialization phase recorded; an
/// ID alone can never observe a later create-and-swap from the registry.
#[derive(Clone, Copy)]
pub(crate) struct PersistentBufferResolver<'a> {
    registry: &'a PersistentBufferRegistry,
    materializations: &'a FxHashMap<PersistentBufferId, PersistentBufferMaterialization>,
}

impl<'a> PersistentBufferResolver<'a> {
    #[inline]
    #[must_use]
    pub(crate) const fn new(
        registry: &'a PersistentBufferRegistry,
        materializations: &'a FxHashMap<PersistentBufferId, PersistentBufferMaterialization>,
    ) -> Self {
        Self {
            registry,
            materializations,
        }
    }

    /// Returns the exact attempt-scoped materialization after rejecting a
    /// token invalidated by a later plan or physical create-and-swap.
    #[inline]
    #[must_use]
    pub(crate) fn materialization(
        self,
        id: PersistentBufferId,
    ) -> Option<PersistentBufferMaterialization> {
        let materialization = *self.materializations.get(&id)?;
        self.registry
            .resolve_physical(materialization.token)
            .map(|_| materialization)
    }

    #[inline]
    #[must_use]
    pub(crate) fn buffer(self, id: PersistentBufferId) -> Option<&'a Tracked<wgpu::Buffer>> {
        let materialization = self.materialization(id)?;
        self.registry.resolve_physical(materialization.token)
    }

    #[inline]
    #[must_use]
    pub(crate) fn is_materialized(self, id: PersistentBufferId) -> bool {
        self.buffer(id).is_some()
    }
}

impl PersistentBufferRegistry {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Allocate a stable logical identity without performing WGPU work.
    pub(crate) fn create_buffer(&mut self) -> PersistentBufferId {
        let id = PersistentBufferId::new();
        let previous = self.records.insert(id, PersistentBufferRecord::default());
        debug_assert!(previous.is_none(), "persistent-buffer ID was reused");
        id
    }

    /// Owner-only retirement that also covers a buffer which was never live
    /// enough to materialize. IDs are never reused, so every outstanding plan,
    /// commit, and physical token becomes permanently stale.
    #[cfg(test)]
    pub(crate) fn remove_buffer(&mut self, id: PersistentBufferId) -> bool {
        self.records.remove(&id).is_some()
    }

    /// Plan one exact revision for an attempt without performing WGPU work.
    ///
    /// Repeating the exact request in the same attempt returns the same plan.
    /// A conflicting same-attempt request, a decreasing attempt, or a
    /// decreasing revision is rejected without mutating registry state.
    #[cfg(test)]
    pub(crate) fn plan_buffer(
        &mut self,
        id: PersistentBufferId,
        attempt_serial: u64,
        revision: u64,
        spec: PersistentBufferSpec,
    ) -> Result<PersistentBufferPlan, PersistentBufferPlanFailure> {
        let transition = self.preflight_plan_transition(id, attempt_serial, revision, spec)?;
        let plan = transition.plan;
        self.apply_plan_transition(transition);
        Ok(plan)
    }

    /// Plan one atomic all-valid or all-invalid logical Buffer family.
    ///
    /// This is a two-phase operation. Every member and all checked arithmetic
    /// are inspected first, then the predicted publication states are proven
    /// coherent. Only after the output storage is also ready do we apply the
    /// infallible transitions. Thus duplicate, stale, conflicting, overflow,
    /// and mixed-state failures leave every member untouched.
    ///
    /// Four members stay inline, matching the GPU Scene
    /// Instance/Geometry/Material/Batch transaction, while the slice form
    /// keeps the primitive reusable by other coherent owners.
    pub(crate) fn plan_coherent_batch(
        &mut self,
        attempt_serial: u64,
        revision: u64,
        requests: &[PersistentBufferPlanRequest],
    ) -> Result<SmallVec<[PersistentBufferPlan; 4]>, PersistentBufferBatchPlanFailure> {
        if requests.is_empty() {
            return Err(PersistentBufferBatchPlanFailure::EmptyBatch);
        }

        for (index, request) in requests.iter().enumerate() {
            if requests[..index]
                .iter()
                .any(|previous| previous.id == request.id)
            {
                return Err(PersistentBufferBatchPlanFailure::DuplicateBuffer(
                    request.id,
                ));
            }
        }

        let mut transitions: SmallVec<[PersistentBufferPlanTransition; 4]> = SmallVec::new();
        for request in requests {
            let transition = self
                .preflight_plan_transition(request.id, attempt_serial, revision, request.spec)
                .map_err(|failure| PersistentBufferBatchPlanFailure::Member {
                    buffer_id: request.id,
                    failure,
                })?;
            transitions.push(transition);
        }

        let valid = transitions[0].plan.valid;
        if transitions
            .iter()
            .any(|transition| transition.plan.valid != valid)
        {
            return Err(PersistentBufferBatchPlanFailure::MixedPublicationState);
        }

        // Populate every potentially allocating output before touching the
        // registry. For the M7 four-table path both SmallVecs remain inline.
        let plans: SmallVec<[PersistentBufferPlan; 4]> = transitions
            .iter()
            .map(|transition| transition.plan)
            .collect();
        for transition in transitions {
            self.apply_plan_transition(transition);
        }
        Ok(plans)
    }

    /// Preflight one exact logical reference without allocating. Graph-level
    /// atomic barriers use this for every live import before materializing the
    /// first member.
    pub(crate) fn validate_ref(
        &self,
        buffer_ref: PersistentBufferRef,
    ) -> Result<(), PersistentBufferMaterializationFailure> {
        let record = self
            .records
            .get(&buffer_ref.id)
            .ok_or(PersistentBufferMaterializationFailure::UnknownOrRetired)?;
        if !Self::buffer_ref_matches(record, buffer_ref) {
            return Err(PersistentBufferMaterializationFailure::StalePlan);
        }
        Ok(())
    }

    /// Validate every deterministic condition needed to allocate one live
    /// persistent Buffer without performing GPU work.
    pub(crate) fn preflight_materialization(
        &self,
        buffer_ref: PersistentBufferRef,
        max_buffer_size: u64,
    ) -> Result<PersistentBufferMaterializationPreflight, PersistentBufferMaterializationFailure>
    {
        self.validate_ref(buffer_ref)?;
        let allocation_size = checked_allocation_size(buffer_ref.spec.desc)
            .ok_or(PersistentBufferMaterializationFailure::CapacityOverflow)?;
        if allocation_size > max_buffer_size {
            return Err(PersistentBufferMaterializationFailure::ExceedsDeviceLimit);
        }
        Ok(PersistentBufferMaterializationPreflight {
            buffer: buffer_ref,
            allocation_size,
        })
    }

    /// Materialize a live plan, reusing capacity when usage remains exact.
    ///
    /// A larger logical size only reallocates when it exceeds retained
    /// capacity. Any usage change forces create-and-swap, since WGPU usages
    /// cannot be added to an existing allocation.
    #[cfg(test)]
    pub(crate) fn materialize(
        &mut self,
        device: &wgpu::Device,
        buffer_ref: PersistentBufferRef,
        label: Option<&str>,
    ) -> Result<PersistentBufferMaterialization, PersistentBufferMaterializationFailure> {
        let preflight =
            self.preflight_materialization(buffer_ref, device.limits().max_buffer_size)?;
        self.materialize_prevalidated(device, preflight, label)
    }

    /// Materialize a recipe whose deterministic size/device checks were
    /// completed by a graph-wide preflight barrier.
    pub(crate) fn materialize_prevalidated(
        &mut self,
        device: &wgpu::Device,
        preflight: PersistentBufferMaterializationPreflight,
        label: Option<&str>,
    ) -> Result<PersistentBufferMaterialization, PersistentBufferMaterializationFailure> {
        let buffer_ref = preflight.buffer;
        self.validate_ref(buffer_ref)?;
        let record = self
            .records
            .get_mut(&buffer_ref.id)
            .expect("materialization preflight guaranteed a live record");

        let required_capacity = preflight.allocation_size;
        let can_reuse = record.physical.as_ref().is_some_and(|physical| {
            physical.usage == buffer_ref.spec.desc.usage
                && physical.allocation_size >= required_capacity
        });

        if !can_reuse {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label,
                size: required_capacity,
                usage: buffer_ref.spec.desc.usage,
                mapped_at_creation: false,
            });
            record.physical = Some(PersistentBufferPhysical {
                usage: buffer_ref.spec.desc.usage,
                allocation_size: required_capacity,
                contents_initialized: false,
                buffer: Tracked::new(buffer),
            });
        }

        let physical = record
            .physical
            .as_ref()
            .expect("materialization must leave a physical buffer");
        Ok(PersistentBufferMaterialization {
            token: PersistentBufferPhysicalToken {
                buffer_id: buffer_ref.id,
                epoch: buffer_ref.epoch,
                plan_generation: buffer_ref.plan_generation,
                revision: buffer_ref.revision,
                attempt_serial: buffer_ref.attempt_serial,
                physical_id: physical.buffer.id(),
            },
            recreated: !can_reuse,
            requires_full_initialization: !physical.contents_initialized,
            allocation_size: physical.allocation_size,
        })
    }

    /// Publish a submitted batch atomically.
    ///
    /// Every logical and physical token is preflighted before any record is
    /// changed. Thus one stale member leaves the entire batch unpublished.
    #[cfg(test)]
    pub(crate) fn commit_batch(
        &mut self,
        requests: &[PersistentBufferCommitRequest],
    ) -> Result<(), PersistentBufferCommitFailure> {
        self.preflight_commit_batch(requests.iter().copied())?;
        self.publish_preflighted_batch(requests.iter().copied());
        Ok(())
    }

    /// Validate an atomic publication batch without allocating or mutating
    /// registry state. The iterator is replayed only for duplicate detection;
    /// graph commit groups already own the canonical request records.
    pub(crate) fn preflight_commit_batch<I>(
        &self,
        requests: I,
    ) -> Result<(), PersistentBufferCommitFailure>
    where
        I: Clone + ExactSizeIterator<Item = PersistentBufferCommitRequest>,
    {
        if requests.len() == 0 {
            return Err(PersistentBufferCommitFailure::EmptyBatch);
        }

        let duplicate_scan = requests.clone();
        for (index, request) in requests.enumerate() {
            if duplicate_scan
                .clone()
                .take(index)
                .any(|previous| previous.commit.buffer_id == request.commit.buffer_id)
            {
                return Err(PersistentBufferCommitFailure::DuplicateBuffer);
            }
            self.preflight_commit(request)?;
        }
        Ok(())
    }

    /// Publish a batch already validated by [`Self::preflight_commit_batch`].
    /// No allocation or fallible work occurs in this mutation pass.
    pub(crate) fn publish_preflighted_batch<I>(&mut self, requests: I)
    where
        I: Iterator<Item = PersistentBufferCommitRequest>,
    {
        for request in requests {
            debug_assert!(self.preflight_commit(request).is_ok());
            let record = self
                .records
                .get_mut(&request.commit.buffer_id)
                .expect("commit preflight guaranteed a live record");
            record.published_revision = Some(request.commit.revision);
            record
                .physical
                .as_mut()
                .expect("commit preflight guaranteed a physical allocation")
                .contents_initialized = true;
        }
    }

    /// Resolve an exact physical token while it remains current.
    #[must_use]
    pub(crate) fn resolve_physical(
        &self,
        token: PersistentBufferPhysicalToken,
    ) -> Option<&Tracked<wgpu::Buffer>> {
        let record = self.records.get(&token.buffer_id)?;
        if !Self::physical_token_matches(record, token) {
            return None;
        }
        record.physical.as_ref().map(|physical| &physical.buffer)
    }

    /// Whether the exact logical reference currently has published contents.
    #[must_use]
    pub(crate) fn is_published(&self, buffer_ref: PersistentBufferRef) -> bool {
        self.records.get(&buffer_ref.id).is_some_and(|record| {
            Self::buffer_ref_matches(record, buffer_ref)
                && record.published_revision == Some(buffer_ref.revision)
                && record.physical.is_some()
        })
    }

    /// Retire only the exact current physical/logical generation.
    ///
    /// A stale token cannot destroy a replacement allocation or a newer
    /// logical plan. Successful retirement permanently removes the ID.
    #[cfg(test)]
    pub(crate) fn retire(
        &mut self,
        token: PersistentBufferPhysicalToken,
    ) -> Result<(), PersistentBufferRetireFailure> {
        let record = self
            .records
            .get(&token.buffer_id)
            .ok_or(PersistentBufferRetireFailure::UnknownOrRetired)?;
        if !Self::physical_token_matches(record, token) {
            return Err(PersistentBufferRetireFailure::StalePhysicalToken);
        }
        self.records
            .remove(&token.buffer_id)
            .expect("retirement preflight guaranteed a live record");
        Ok(())
    }

    /// Validate and compute one planning transition without mutation.
    fn preflight_plan_transition(
        &self,
        id: PersistentBufferId,
        attempt_serial: u64,
        revision: u64,
        spec: PersistentBufferSpec,
    ) -> Result<PersistentBufferPlanTransition, PersistentBufferPlanFailure> {
        let record = self
            .records
            .get(&id)
            .ok_or(PersistentBufferPlanFailure::UnknownOrRetired)?;

        if let Some(previous_attempt) = record.last_planned_attempt {
            if attempt_serial < previous_attempt {
                return Err(PersistentBufferPlanFailure::AttemptRegressed);
            }
            if attempt_serial == previous_attempt {
                if record.desired_revision != Some(revision) || record.desired_spec != Some(spec) {
                    return Err(PersistentBufferPlanFailure::SameAttemptConflict);
                }
                return Ok(PersistentBufferPlanTransition {
                    plan: Self::plan_from_record(id, record),
                    apply: false,
                    target_changed: false,
                    spec_changed: false,
                });
            }
        }

        if record
            .desired_revision
            .is_some_and(|previous| revision < previous)
        {
            return Err(PersistentBufferPlanFailure::RevisionRegressed);
        }

        let spec_changed = record.desired_spec != Some(spec);
        let target_changed = record.desired_revision != Some(revision) || spec_changed;
        let epoch = if target_changed {
            record
                .epoch
                .checked_add(1)
                .ok_or(PersistentBufferPlanFailure::EpochOverflow)?
        } else {
            record.epoch
        };
        let plan_generation = record
            .plan_generation
            .checked_add(1)
            .ok_or(PersistentBufferPlanFailure::PlanGenerationOverflow)?;
        let valid = !target_changed
            && record.published_revision == Some(revision)
            && record.physical.is_some();
        let buffer = PersistentBufferRef {
            id,
            spec,
            epoch,
            plan_generation,
            revision,
            attempt_serial,
        };
        let commit = (!valid).then_some(PersistentBufferCommitToken {
            buffer_id: id,
            epoch,
            plan_generation,
            revision,
            attempt_serial,
        });
        Ok(PersistentBufferPlanTransition {
            plan: PersistentBufferPlan {
                buffer,
                valid,
                commit,
            },
            apply: true,
            target_changed,
            spec_changed,
        })
    }

    /// Apply a fully validated transition. No checked arithmetic, lookup, or
    /// allocation occurs here, making the batch mutation phase infallible.
    fn apply_plan_transition(&mut self, transition: PersistentBufferPlanTransition) {
        if !transition.apply {
            return;
        }
        let plan = transition.plan;
        let record = self
            .records
            .get_mut(&plan.buffer.id)
            .expect("persistent Buffer planning preflight lost a live record");

        if transition.target_changed {
            record.epoch = plan.buffer.epoch;
            if transition.spec_changed {
                // Even when retained capacity is sufficient, a changed exact
                // ABI/extent has no initialized delta base until a submitted
                // complete image establishes it.
                if let Some(physical) = record.physical.as_mut() {
                    physical.contents_initialized = false;
                }
            }
            record.desired_spec = Some(plan.buffer.spec);
            record.desired_revision = Some(plan.buffer.revision);
            // Superseding a target is immediately fail-closed. An abandoned
            // graph cannot resurrect the previous publication.
            record.published_revision = None;
        }
        record.plan_generation = plan.buffer.plan_generation;
        record.last_planned_attempt = Some(plan.buffer.attempt_serial);
        record.last_plan_valid = plan.valid;
        debug_assert_eq!(Self::plan_from_record(plan.buffer.id, record), plan);
    }

    #[inline]
    fn plan_from_record(
        id: PersistentBufferId,
        record: &PersistentBufferRecord,
    ) -> PersistentBufferPlan {
        let spec = record
            .desired_spec
            .expect("a planned buffer must have a desired specification");
        let revision = record
            .desired_revision
            .expect("a planned buffer must have a desired revision");
        let attempt_serial = record
            .last_planned_attempt
            .expect("a planned buffer must have an attempt serial");
        let buffer = PersistentBufferRef {
            id,
            spec,
            epoch: record.epoch,
            plan_generation: record.plan_generation,
            revision,
            attempt_serial,
        };
        let commit = (!record.last_plan_valid).then_some(PersistentBufferCommitToken {
            buffer_id: id,
            epoch: record.epoch,
            plan_generation: record.plan_generation,
            revision,
            attempt_serial,
        });
        PersistentBufferPlan {
            buffer,
            valid: record.last_plan_valid,
            commit,
        }
    }

    #[inline]
    fn buffer_ref_matches(
        record: &PersistentBufferRecord,
        buffer_ref: PersistentBufferRef,
    ) -> bool {
        record.epoch == buffer_ref.epoch
            && record.plan_generation == buffer_ref.plan_generation
            && record.desired_revision == Some(buffer_ref.revision)
            && record.desired_spec == Some(buffer_ref.spec)
            && record.last_planned_attempt == Some(buffer_ref.attempt_serial)
    }

    #[inline]
    fn physical_token_matches(
        record: &PersistentBufferRecord,
        token: PersistentBufferPhysicalToken,
    ) -> bool {
        record.epoch == token.epoch
            && record.plan_generation == token.plan_generation
            && record.desired_revision == Some(token.revision)
            && record.last_planned_attempt == Some(token.attempt_serial)
            && record
                .physical
                .as_ref()
                .is_some_and(|physical| physical.buffer.id() == token.physical_id)
    }

    fn preflight_commit(
        &self,
        request: PersistentBufferCommitRequest,
    ) -> Result<(), PersistentBufferCommitFailure> {
        if request.commit.buffer_id != request.physical.buffer_id
            || request.commit.epoch != request.physical.epoch
            || request.commit.plan_generation != request.physical.plan_generation
            || request.commit.revision != request.physical.revision
            || request.commit.attempt_serial != request.physical.attempt_serial
        {
            return Err(PersistentBufferCommitFailure::LogicalPhysicalMismatch);
        }

        let record = self
            .records
            .get(&request.commit.buffer_id)
            .ok_or(PersistentBufferCommitFailure::UnknownOrRetired)?;
        if record.epoch != request.commit.epoch
            || record.plan_generation != request.commit.plan_generation
            || record.desired_revision != Some(request.commit.revision)
            || record.last_planned_attempt != Some(request.commit.attempt_serial)
        {
            return Err(PersistentBufferCommitFailure::StaleLogicalToken);
        }
        if !Self::physical_token_matches(record, request.physical) {
            return Err(PersistentBufferCommitFailure::StalePhysicalToken);
        }
        if record.published_revision == Some(request.commit.revision) {
            return Err(PersistentBufferCommitFailure::AlreadyPublished);
        }
        Ok(())
    }
}

#[inline]
fn checked_allocation_size(desc: BufferDesc) -> Option<u64> {
    desc.logical_size
        .max(1)
        .checked_next_power_of_two()
        .map(|size| size.max(BufferDesc::MIN_ALLOCATION_SIZE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WgpuContext;
    use crate::graph::core::graph::{
        PersistentBufferGraphMaterializationFailure, PersistentBufferWriteCoverage,
        SubmittedGraphReceipt,
    };
    use crate::graph::core::{ExecuteContext, FrameArena, GraphStorage, PassNode, RenderGraph};
    use crate::settings::{RendererInitConfig, RendererSettings};
    use wgpu::util::DeviceExt;

    struct MockPass;

    impl PassNode<'_> for MockPass {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

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

    fn spec(
        size: u64,
        usage: wgpu::BufferUsages,
        schema_id: u64,
        stride: u64,
    ) -> PersistentBufferSpec {
        PersistentBufferSpec::new(BufferDesc::new(size, usage), schema_id, stride)
    }

    fn request(
        plan: PersistentBufferPlan,
        materialization: PersistentBufferMaterialization,
    ) -> PersistentBufferCommitRequest {
        PersistentBufferCommitRequest::new(
            plan.commit.expect("test plan must require publication"),
            materialization.token,
        )
    }

    fn four_buffer_ids(registry: &mut PersistentBufferRegistry) -> [PersistentBufferId; 4] {
        std::array::from_fn(|_| registry.create_buffer())
    }

    fn four_specs(usage: wgpu::BufferUsages) -> [PersistentBufferSpec; 4] {
        std::array::from_fn(|index| spec(64, usage, 100 + index as u64, 16))
    }

    fn coherent_requests(
        ids: [PersistentBufferId; 4],
        specs: [PersistentBufferSpec; 4],
    ) -> [PersistentBufferPlanRequest; 4] {
        std::array::from_fn(|index| PersistentBufferPlanRequest::new(ids[index], specs[index]))
    }

    fn publish_plans(
        gpu: &WgpuContext,
        registry: &mut PersistentBufferRegistry,
        plans: &[PersistentBufferPlan],
    ) {
        let mut commits: SmallVec<[PersistentBufferCommitRequest; 4]> = SmallVec::new();
        for &plan in plans {
            let physical = registry
                .materialize(&gpu.device, plan.buffer, Some("coherent batch test"))
                .unwrap();
            commits.push(request(plan, physical));
        }
        registry.commit_batch(&commits).unwrap();
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct RecordSnapshot {
        epoch: u64,
        plan_generation: u64,
        desired_spec: Option<PersistentBufferSpec>,
        desired_revision: Option<u64>,
        published_revision: Option<u64>,
        last_planned_attempt: Option<u64>,
        last_plan_valid: bool,
        physical: Option<(wgpu::BufferUsages, u64, bool, u64)>,
    }

    fn record_snapshot(
        registry: &PersistentBufferRegistry,
        id: PersistentBufferId,
    ) -> RecordSnapshot {
        let record = registry.records.get(&id).expect("test Buffer must exist");
        RecordSnapshot {
            epoch: record.epoch,
            plan_generation: record.plan_generation,
            desired_spec: record.desired_spec,
            desired_revision: record.desired_revision,
            published_revision: record.published_revision,
            last_planned_attempt: record.last_planned_attempt,
            last_plan_valid: record.last_plan_valid,
            physical: record.physical.as_ref().map(|physical| {
                (
                    physical.usage,
                    physical.allocation_size,
                    physical.contents_initialized,
                    physical.buffer.id(),
                )
            }),
        }
    }

    fn four_snapshots(
        registry: &PersistentBufferRegistry,
        ids: [PersistentBufferId; 4],
    ) -> [RecordSnapshot; 4] {
        std::array::from_fn(|index| record_snapshot(registry, ids[index]))
    }

    fn mark_group_encoded(
        graph: &RenderGraph<'_>,
        group: crate::graph::core::graph::PersistentBufferCommitGroupId,
    ) {
        graph
            .persistent_buffer_commit_tracker()
            .mark_encoded_with_coverage(group, PersistentBufferWriteCoverage::Full);
    }

    fn submit_empty_graph(
        graph: &mut RenderGraph<'_>,
        gpu: &WgpuContext,
    ) -> Result<SubmittedGraphReceipt, PersistentBufferCommitFailure> {
        let encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("persistent Buffer test submit"),
            });
        graph.submit_command_buffer(&gpu.queue, encoder.finish())
    }

    #[test]
    fn ids_are_stable_monotonic_and_never_reused() {
        let mut registry = PersistentBufferRegistry::new();
        let first = registry.create_buffer();
        let second = registry.create_buffer();
        assert!(second.get() > first.get());
    }

    #[test]
    fn owner_can_retire_a_never_materialized_buffer() {
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let target = spec(256, wgpu::BufferUsages::STORAGE, 1, 16);
        let plan = registry.plan_buffer(id, 1, 1, target).unwrap();

        assert!(registry.remove_buffer(id));
        assert!(!registry.remove_buffer(id));
        assert_eq!(
            registry.plan_buffer(id, 2, 1, target),
            Err(PersistentBufferPlanFailure::UnknownOrRetired)
        );
        assert_eq!(
            registry.commit_batch(&[PersistentBufferCommitRequest::new(
                plan.commit.expect("initial plan needs publication"),
                PersistentBufferPhysicalToken {
                    buffer_id: id,
                    epoch: plan.buffer.epoch,
                    plan_generation: plan.buffer.plan_generation,
                    revision: plan.buffer.revision,
                    attempt_serial: plan.buffer.attempt_serial,
                    physical_id: 1,
                },
            )]),
            Err(PersistentBufferCommitFailure::UnknownOrRetired)
        );
    }

    #[test]
    fn same_attempt_is_idempotent_and_rejects_conflicts() {
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let base = spec(64, wgpu::BufferUsages::STORAGE, 7, 16);
        let first = registry.plan_buffer(id, 10, 3, base).unwrap();
        assert_eq!(registry.plan_buffer(id, 10, 3, base), Ok(first));
        assert_eq!(
            registry.plan_buffer(id, 10, 4, base),
            Err(PersistentBufferPlanFailure::SameAttemptConflict)
        );
        assert_eq!(
            registry.plan_buffer(id, 10, 3, spec(64, wgpu::BufferUsages::STORAGE, 8, 16),),
            Err(PersistentBufferPlanFailure::SameAttemptConflict)
        );
        assert_eq!(registry.plan_buffer(id, 10, 3, base), Ok(first));
    }

    #[test]
    fn coherent_batch_is_fresh_then_warm_and_same_attempt_idempotent() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let ids = four_buffer_ids(&mut registry);
        let specs = four_specs(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let requests = coherent_requests(ids, specs);

        let fresh = registry.plan_coherent_batch(1, 7, &requests).unwrap();
        assert_eq!(fresh.len(), 4);
        assert!(
            fresh
                .iter()
                .all(|plan| !plan.valid && plan.commit.is_some())
        );
        assert_eq!(
            registry.plan_coherent_batch(1, 7, &requests),
            Ok(fresh.clone()),
            "an exact same-attempt batch must be idempotent"
        );

        publish_plans(&gpu, &mut registry, &fresh);
        let warm = registry.plan_coherent_batch(2, 7, &requests).unwrap();
        assert!(warm.iter().all(|plan| plan.valid && plan.commit.is_none()));
        assert!(warm.iter().all(|plan| registry.is_published(plan.buffer)));
        assert_eq!(
            registry.plan_coherent_batch(2, 7, &requests),
            Ok(warm),
            "a warm exact same-attempt batch must remain idempotent"
        );
    }

    #[test]
    fn coherent_batch_rejects_mixed_spec_change_without_mutation() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let ids = four_buffer_ids(&mut registry);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let specs = four_specs(usage);
        let requests = coherent_requests(ids, specs);
        let fresh = registry.plan_coherent_batch(1, 3, &requests).unwrap();
        publish_plans(&gpu, &mut registry, &fresh);
        let warm = registry.plan_coherent_batch(2, 3, &requests).unwrap();
        let before = four_snapshots(&registry, ids);

        let replacement = registry.create_buffer();
        let replacement_before = record_snapshot(&registry, replacement);
        let mut replacement_requests = requests;
        replacement_requests[3] = PersistentBufferPlanRequest::new(replacement, specs[3]);
        assert_eq!(
            registry.plan_coherent_batch(3, 3, &replacement_requests),
            Err(PersistentBufferBatchPlanFailure::MixedPublicationState)
        );
        assert_eq!(four_snapshots(&registry, ids), before);
        assert_eq!(
            record_snapshot(&registry, replacement),
            replacement_before,
            "a fresh mixed member must remain completely unplanned"
        );

        let mut changed_specs = specs;
        changed_specs[0] = spec(128, usage, specs[0].schema_id, 16);
        let changed_requests = coherent_requests(ids, changed_specs);
        assert_eq!(
            registry.plan_coherent_batch(3, 3, &changed_requests),
            Err(PersistentBufferBatchPlanFailure::MixedPublicationState)
        );
        assert_eq!(four_snapshots(&registry, ids), before);
        assert!(
            warm.iter().all(|plan| registry.is_published(plan.buffer)),
            "a rejected spec transition must not clear any publication or initialization latch"
        );

        // The owner resolves a legitimate table-spec change by advancing the
        // shared logical revision. All four members then become invalid as one
        // coherent transaction.
        let advanced = registry
            .plan_coherent_batch(3, 4, &changed_requests)
            .unwrap();
        assert!(
            advanced
                .iter()
                .all(|plan| !plan.valid && plan.commit.is_some())
        );
    }

    #[test]
    fn coherent_batch_preflight_failures_leave_every_member_untouched() {
        let mut registry = PersistentBufferRegistry::new();
        let ids = four_buffer_ids(&mut registry);
        let specs = four_specs(wgpu::BufferUsages::STORAGE);
        let requests = coherent_requests(ids, specs);
        registry.plan_coherent_batch(10, 5, &requests).unwrap();

        let before = four_snapshots(&registry, ids);
        assert_eq!(
            registry.plan_coherent_batch(11, 5, &[]),
            Err(PersistentBufferBatchPlanFailure::EmptyBatch)
        );
        assert_eq!(four_snapshots(&registry, ids), before);

        let duplicate = [requests[0], requests[1], requests[0], requests[3]];
        assert_eq!(
            registry.plan_coherent_batch(11, 5, &duplicate),
            Err(PersistentBufferBatchPlanFailure::DuplicateBuffer(ids[0]))
        );
        assert_eq!(four_snapshots(&registry, ids), before);

        assert_eq!(
            registry.plan_coherent_batch(9, 5, &requests),
            Err(PersistentBufferBatchPlanFailure::Member {
                buffer_id: ids[0],
                failure: PersistentBufferPlanFailure::AttemptRegressed,
            })
        );
        assert_eq!(four_snapshots(&registry, ids), before);

        assert_eq!(
            registry.plan_coherent_batch(11, 4, &requests),
            Err(PersistentBufferBatchPlanFailure::Member {
                buffer_id: ids[0],
                failure: PersistentBufferPlanFailure::RevisionRegressed,
            })
        );
        assert_eq!(four_snapshots(&registry, ids), before);

        assert_eq!(
            registry.plan_coherent_batch(10, 6, &requests),
            Err(PersistentBufferBatchPlanFailure::Member {
                buffer_id: ids[0],
                failure: PersistentBufferPlanFailure::SameAttemptConflict,
            })
        );
        assert_eq!(four_snapshots(&registry, ids), before);

        let retired = registry.create_buffer();
        assert!(registry.remove_buffer(retired));
        let mut unknown = requests;
        unknown[2] = PersistentBufferPlanRequest::new(retired, specs[2]);
        let before_known = [
            record_snapshot(&registry, ids[0]),
            record_snapshot(&registry, ids[1]),
            record_snapshot(&registry, ids[3]),
        ];
        assert_eq!(
            registry.plan_coherent_batch(11, 5, &unknown),
            Err(PersistentBufferBatchPlanFailure::Member {
                buffer_id: retired,
                failure: PersistentBufferPlanFailure::UnknownOrRetired,
            })
        );
        assert_eq!(
            [
                record_snapshot(&registry, ids[0]),
                record_snapshot(&registry, ids[1]),
                record_snapshot(&registry, ids[3]),
            ],
            before_known
        );
    }

    #[test]
    fn coherent_batch_overflow_is_detected_before_any_transition_applies() {
        let mut registry = PersistentBufferRegistry::new();
        let ids = four_buffer_ids(&mut registry);
        let specs = four_specs(wgpu::BufferUsages::STORAGE);
        let requests = coherent_requests(ids, specs);
        registry.plan_coherent_batch(1, 1, &requests).unwrap();

        let original_epoch = registry.records[&ids[2]].epoch;
        registry.records.get_mut(&ids[2]).unwrap().epoch = u64::MAX;
        let before_epoch = four_snapshots(&registry, ids);
        assert_eq!(
            registry.plan_coherent_batch(2, 2, &requests),
            Err(PersistentBufferBatchPlanFailure::Member {
                buffer_id: ids[2],
                failure: PersistentBufferPlanFailure::EpochOverflow,
            })
        );
        assert_eq!(four_snapshots(&registry, ids), before_epoch);

        registry.records.get_mut(&ids[2]).unwrap().epoch = original_epoch;
        registry.records.get_mut(&ids[3]).unwrap().plan_generation = u64::MAX;
        let before_generation = four_snapshots(&registry, ids);
        assert_eq!(
            registry.plan_coherent_batch(2, 1, &requests),
            Err(PersistentBufferBatchPlanFailure::Member {
                buffer_id: ids[3],
                failure: PersistentBufferPlanFailure::PlanGenerationOverflow,
            })
        );
        assert_eq!(four_snapshots(&registry, ids), before_generation);
    }

    #[test]
    #[should_panic(expected = "record stride must be non-zero")]
    fn spec_rejects_zero_stride() {
        let _ = PersistentBufferSpec::new(BufferDesc::new(64, wgpu::BufferUsages::STORAGE), 1, 0);
    }

    #[test]
    #[should_panic(expected = "exact record multiple")]
    fn spec_rejects_partial_records() {
        let _ = PersistentBufferSpec::new(BufferDesc::new(65, wgpu::BufferUsages::STORAGE), 1, 16);
    }

    #[test]
    fn attempts_and_revisions_never_regress() {
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let base = spec(64, wgpu::BufferUsages::STORAGE, 1, 16);
        registry.plan_buffer(id, 8, 5, base).unwrap();
        assert_eq!(
            registry.plan_buffer(id, 7, 5, base),
            Err(PersistentBufferPlanFailure::AttemptRegressed)
        );
        assert_eq!(
            registry.plan_buffer(id, 9, 4, base),
            Err(PersistentBufferPlanFailure::RevisionRegressed)
        );
        assert!(registry.plan_buffer(id, 9, 5, base).is_ok());
    }

    #[test]
    fn counter_growth_is_checked_and_failure_does_not_partially_mutate() {
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let base = spec(64, wgpu::BufferUsages::STORAGE, 1, 16);
        let changed = spec(128, wgpu::BufferUsages::STORAGE, 1, 16);
        let first = registry.plan_buffer(id, 1, 1, base).unwrap();

        registry.records.get_mut(&id).unwrap().epoch = u64::MAX;
        assert_eq!(
            registry.plan_buffer(id, 2, 2, changed),
            Err(PersistentBufferPlanFailure::EpochOverflow)
        );
        let record = registry.records.get(&id).unwrap();
        assert_eq!(record.desired_spec, Some(base));
        assert_eq!(record.desired_revision, Some(1));
        assert_eq!(record.last_planned_attempt, Some(1));

        let record = registry.records.get_mut(&id).unwrap();
        record.epoch = first.buffer.epoch;
        record.plan_generation = u64::MAX;
        assert_eq!(
            registry.plan_buffer(id, 2, 1, base),
            Err(PersistentBufferPlanFailure::PlanGenerationOverflow)
        );
        assert_eq!(
            registry.records.get(&id).unwrap().last_planned_attempt,
            Some(1)
        );
    }

    #[test]
    fn changing_revision_or_exact_spec_invalidates_without_rollback() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let base = spec(
            64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            1,
            16,
        );
        let first = registry.plan_buffer(id, 1, 1, base).unwrap();
        let first_physical = registry
            .materialize(&gpu.device, first.buffer, Some("persistent first"))
            .unwrap();
        registry
            .commit_batch(&[request(first, first_physical)])
            .unwrap();
        assert!(registry.is_published(first.buffer));

        let second = registry.plan_buffer(id, 2, 2, base).unwrap();
        assert!(!second.valid);
        assert!(!registry.is_published(first.buffer));
        assert!(!registry.is_published(second.buffer));
        assert_eq!(
            registry.commit_batch(&[request(first, first_physical)]),
            Err(PersistentBufferCommitFailure::StaleLogicalToken),
            "the superseded revision must not be publishable again"
        );
        assert_eq!(
            registry.plan_buffer(id, 3, 1, base),
            Err(PersistentBufferPlanFailure::RevisionRegressed)
        );

        // Advancing attempts cannot resurrect the superseded publication.
        let retry = registry.plan_buffer(id, 3, 2, base).unwrap();
        assert!(!retry.valid);
        assert!(retry.commit.is_some());

        // Schema and stride participate in exact specification identity even
        // when the BufferDesc and revision are unchanged.
        let changed_schema = spec(base.desc.logical_size, base.desc.usage, 2, 32);
        let changed = registry.plan_buffer(id, 4, 2, changed_schema).unwrap();
        assert!(!changed.valid);
        assert_ne!(changed.buffer.epoch, retry.buffer.epoch);
    }

    #[test]
    fn materialization_reuses_capacity_but_requires_initialization_for_spec_changes() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let base = spec(300, usage, 1, 1);

        let first = registry.plan_buffer(id, 1, 1, base).unwrap();
        let first_physical = registry
            .materialize(&gpu.device, first.buffer, Some("persistent capacity"))
            .unwrap();
        assert!(first_physical.recreated);
        assert!(first_physical.requires_full_initialization);
        assert_eq!(first_physical.allocation_size, 512);
        registry
            .commit_batch(&[request(first, first_physical)])
            .unwrap();

        let revision_only = registry.plan_buffer(id, 2, 2, base).unwrap();
        let revision_only_physical = registry
            .materialize(&gpu.device, revision_only.buffer, None)
            .unwrap();
        assert!(!revision_only_physical.recreated);
        assert!(
            !revision_only_physical.requires_full_initialization,
            "a revision-only delta may retain the complete submitted base"
        );
        registry
            .commit_batch(&[request(revision_only, revision_only_physical)])
            .unwrap();

        let within_capacity = registry
            .plan_buffer(id, 3, 3, spec(400, usage, 1, 1))
            .unwrap();
        let reused = registry
            .materialize(&gpu.device, within_capacity.buffer, None)
            .unwrap();
        assert!(!reused.recreated);
        assert!(
            reused.requires_full_initialization,
            "a changed logical extent must rebuild even within retained capacity"
        );
        assert_eq!(reused.token.physical_id, first_physical.token.physical_id);
        assert_eq!(reused.allocation_size, 512);
        registry
            .commit_batch(&[request(within_capacity, reused)])
            .unwrap();

        let growth = registry
            .plan_buffer(id, 4, 4, spec(513, usage, 1, 1))
            .unwrap();
        let grown = registry
            .materialize(&gpu.device, growth.buffer, None)
            .unwrap();
        assert!(grown.recreated);
        assert!(grown.requires_full_initialization);
        assert_ne!(grown.token.physical_id, reused.token.physical_id);
        assert_eq!(grown.allocation_size, 1024);

        let changed_usage = registry
            .plan_buffer(
                id,
                5,
                5,
                spec(
                    513,
                    wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                    1,
                    1,
                ),
            )
            .unwrap();
        let swapped = registry
            .materialize(&gpu.device, changed_usage.buffer, None)
            .unwrap();
        assert!(swapped.recreated);
        assert!(swapped.requires_full_initialization);
        assert_ne!(swapped.token.physical_id, grown.token.physical_id);
        assert!(
            registry
                .resolve_physical(swapped.token)
                .is_some_and(|buffer| buffer.id() == swapped.token.physical_id)
        );
        assert!(registry.resolve_physical(grown.token).is_none());
    }

    #[test]
    fn aborted_materialization_remains_invalid_and_is_reused_by_retry() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let target = spec(
            256,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            4,
            16,
        );
        let attempt = registry.plan_buffer(id, 1, 1, target).unwrap();
        let staged = registry
            .materialize(&gpu.device, attempt.buffer, None)
            .unwrap();
        assert!(staged.requires_full_initialization);
        assert!(!registry.is_published(attempt.buffer));
        let mut materializations = FxHashMap::default();
        materializations.insert(id, staged);
        assert!(PersistentBufferResolver::new(&registry, &materializations).is_materialized(id));

        let retry = registry.plan_buffer(id, 2, 1, target).unwrap();
        assert!(!retry.valid);
        assert!(
            !PersistentBufferResolver::new(&registry, &materializations).is_materialized(id),
            "the resolver must reject the previous attempt's exact token"
        );
        let retried = registry
            .materialize(&gpu.device, retry.buffer, None)
            .unwrap();
        assert!(!retried.recreated);
        assert!(
            retried.requires_full_initialization,
            "reusing an aborted allocation must not lose its initialization requirement"
        );
        assert_eq!(retried.token.physical_id, staged.token.physical_id);
        assert!(registry.resolve_physical(staged.token).is_none());
        materializations.insert(id, retried);
        assert_eq!(
            PersistentBufferResolver::new(&registry, &materializations)
                .buffer(id)
                .map(Tracked::id),
            Some(retried.token.physical_id)
        );
    }

    #[test]
    fn grow_abort_retry_latches_full_initialization_until_exact_commit() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let base = spec(256, usage, 9, 16);
        let grown_spec = spec(1024, usage, 9, 16);

        let base_plan = registry.plan_buffer(id, 1, 1, base).unwrap();
        let base_physical = registry
            .materialize(&gpu.device, base_plan.buffer, None)
            .unwrap();
        registry
            .commit_batch(&[request(base_plan, base_physical)])
            .unwrap();

        let aborted_plan = registry.plan_buffer(id, 2, 2, grown_spec).unwrap();
        let aborted_physical = registry
            .materialize(&gpu.device, aborted_plan.buffer, None)
            .unwrap();
        assert!(aborted_physical.recreated);
        assert!(aborted_physical.requires_full_initialization);
        assert_ne!(
            aborted_physical.token.physical_id,
            base_physical.token.physical_id
        );
        // Deliberately skip publication: the replacement allocation contains
        // no authoritative complete image.

        let retry_plan = registry.plan_buffer(id, 3, 2, grown_spec).unwrap();
        let retried_physical = registry
            .materialize(&gpu.device, retry_plan.buffer, None)
            .unwrap();
        assert!(!retried_physical.recreated);
        assert_eq!(
            retried_physical.token.physical_id,
            aborted_physical.token.physical_id
        );
        assert!(
            retried_physical.requires_full_initialization,
            "an aborted create-and-swap must force a complete retry"
        );

        registry
            .commit_batch(&[request(retry_plan, retried_physical)])
            .unwrap();
        let published_retry = registry.plan_buffer(id, 4, 2, grown_spec).unwrap();
        assert!(published_retry.valid);
        assert!(published_retry.commit.is_none());
        let warm_physical = registry
            .materialize(&gpu.device, published_retry.buffer, None)
            .unwrap();
        assert!(!warm_physical.recreated);
        assert!(
            !warm_physical.requires_full_initialization,
            "the exact submitted commit releases the full-rebuild latch"
        );
    }

    #[test]
    fn schema_and_stride_changes_invalidate_initialized_contents() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        let base = registry
            .plan_buffer(id, 1, 1, spec(256, usage, 1, 16))
            .unwrap();
        let base_physical = registry
            .materialize(&gpu.device, base.buffer, None)
            .unwrap();
        registry
            .commit_batch(&[request(base, base_physical)])
            .unwrap();

        let changed_schema = registry
            .plan_buffer(id, 2, 1, spec(256, usage, 2, 16))
            .unwrap();
        let schema_physical = registry
            .materialize(&gpu.device, changed_schema.buffer, None)
            .unwrap();
        assert!(!schema_physical.recreated);
        assert!(schema_physical.requires_full_initialization);
        registry
            .commit_batch(&[request(changed_schema, schema_physical)])
            .unwrap();

        let changed_stride = registry
            .plan_buffer(id, 3, 1, spec(256, usage, 2, 32))
            .unwrap();
        let stride_physical = registry
            .materialize(&gpu.device, changed_stride.buffer, None)
            .unwrap();
        assert!(!stride_physical.recreated);
        assert!(stride_physical.requires_full_initialization);
    }

    #[test]
    fn commit_batch_is_atomic_when_one_member_is_stale() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let first_id = registry.create_buffer();
        let second_id = registry.create_buffer();
        let target = spec(
            256,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            1,
            16,
        );
        let first = registry.plan_buffer(first_id, 1, 1, target).unwrap();
        let second = registry.plan_buffer(second_id, 1, 1, target).unwrap();
        let first_physical = registry
            .materialize(&gpu.device, first.buffer, None)
            .unwrap();
        let stale_second_physical = registry
            .materialize(&gpu.device, second.buffer, None)
            .unwrap();
        let current_second = registry.plan_buffer(second_id, 2, 1, target).unwrap();

        let failed = registry.commit_batch(&[
            request(first, first_physical),
            request(second, stale_second_physical),
        ]);
        assert_eq!(
            failed,
            Err(PersistentBufferCommitFailure::StaleLogicalToken)
        );
        assert!(!registry.is_published(first.buffer));
        assert!(!registry.is_published(current_second.buffer));

        let current_second_physical = registry
            .materialize(&gpu.device, current_second.buffer, None)
            .unwrap();
        registry
            .commit_batch(&[
                request(first, first_physical),
                request(current_second, current_second_physical),
            ])
            .unwrap();
        assert!(registry.is_published(first.buffer));
        assert!(registry.is_published(current_second.buffer));
    }

    #[test]
    fn commit_batch_rejects_empty_duplicate_and_republication() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        assert_eq!(
            registry.commit_batch(&[]),
            Err(PersistentBufferCommitFailure::EmptyBatch)
        );

        let id = registry.create_buffer();
        let target = spec(
            256,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            1,
            16,
        );
        let plan = registry.plan_buffer(id, 1, 1, target).unwrap();
        let physical = registry
            .materialize(&gpu.device, plan.buffer, None)
            .unwrap();
        let commit = request(plan, physical);
        assert_eq!(
            registry.commit_batch(&[commit, commit]),
            Err(PersistentBufferCommitFailure::DuplicateBuffer)
        );
        assert!(!registry.is_published(plan.buffer));
        registry.commit_batch(&[commit]).unwrap();
        assert_eq!(
            registry.commit_batch(&[commit]),
            Err(PersistentBufferCommitFailure::AlreadyPublished)
        );
    }

    #[test]
    fn retirement_rejects_stale_physical_authority() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let target = spec(
            256,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            1,
            16,
        );
        let first = registry.plan_buffer(id, 1, 1, target).unwrap();
        let stale = registry
            .materialize(&gpu.device, first.buffer, None)
            .unwrap();
        let current = registry.plan_buffer(id, 2, 1, target).unwrap();
        assert_eq!(
            registry.retire(stale.token),
            Err(PersistentBufferRetireFailure::StalePhysicalToken)
        );

        let current_physical = registry
            .materialize(&gpu.device, current.buffer, None)
            .unwrap();
        registry.retire(current_physical.token).unwrap();
        assert_eq!(
            registry.plan_buffer(id, 3, 1, target),
            Err(PersistentBufferPlanFailure::UnknownOrRetired)
        );
        assert_eq!(
            registry.retire(current_physical.token),
            Err(PersistentBufferRetireFailure::UnknownOrRetired)
        );
    }

    #[test]
    fn oversized_capacity_fails_without_creating_a_physical_buffer() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let target = spec((1_u64 << 63) + 1, wgpu::BufferUsages::STORAGE, 1, 1);
        let plan = registry.plan_buffer(id, 1, 1, target).unwrap();
        assert_eq!(
            registry.materialize(&gpu.device, plan.buffer, None),
            Err(PersistentBufferMaterializationFailure::CapacityOverflow)
        );
        assert!(registry.records.get(&id).unwrap().physical.is_none());
    }

    #[test]
    fn device_limit_is_rejected_before_wgpu_allocation() {
        let gpu = gpu();
        let max_buffer_size = gpu.device.limits().max_buffer_size;
        let logical_size = max_buffer_size
            .checked_add(1)
            .expect("test device reported an unrepresentable max buffer size");
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let plan = registry
            .plan_buffer(
                id,
                1,
                1,
                spec(logical_size, wgpu::BufferUsages::STORAGE, 2, 1),
            )
            .unwrap();

        assert_eq!(
            registry.materialize(&gpu.device, plan.buffer, None),
            Err(PersistentBufferMaterializationFailure::ExceedsDeviceLimit)
        );
        assert!(registry.records.get(&id).unwrap().physical.is_none());
    }

    #[test]
    fn dead_graph_import_deduplicates_without_materialization_or_publication() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let target = spec(
            256,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            11,
            16,
        );
        let plan = registry.plan_buffer(id, 1, 1, target).unwrap();
        let commit = plan
            .commit
            .expect("new persistent buffer needs publication");
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);

        let first = graph.import_persistent_buffer("Dead_Persistent", plan);
        let duplicate = graph.import_persistent_buffer("Dead_Persistent_Again", plan);
        assert_eq!(first, duplicate, "same exact import must be deduplicated");
        graph.add_pass("Dead_Persistent_Writer", |builder| {
            builder.write_buffer(first);
            (MockPass, ())
        });
        graph.register_persistent_buffer_commit_group(&[(commit, first)]);

        graph.compile_topology();
        assert!(
            graph.storage.execution_queue.is_empty(),
            "a writer and commit registration without a live consumer are not DPE roots"
        );
        graph
            .materialize_live_persistent_buffers(&mut registry, &gpu.device)
            .unwrap();
        assert!(graph.storage.persistent_buffer_materializations.is_empty());
        assert!(registry.records.get(&id).unwrap().physical.is_none());
        let submitted = submit_empty_graph(&mut graph, &gpu).unwrap();
        graph
            .commit_submitted_persistent_buffers(&mut registry, submitted)
            .unwrap();
        assert!(!registry.is_published(plan.buffer));
    }

    #[test]
    fn live_graph_materializes_resolves_and_publishes_exact_buffer() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let target = spec(
            256,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            12,
            16,
        );
        let plan = registry.plan_buffer(id, 1, 1, target).unwrap();
        let commit = plan
            .commit
            .expect("new persistent buffer needs publication");
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);

        let buffer = graph.import_persistent_buffer("Live_Persistent", plan);
        graph.add_pass("Live_Persistent_Writer", |builder| {
            builder.write_buffer(buffer);
            (MockPass, ())
        });
        graph.add_pass("Live_Persistent_Consumer", |builder| {
            builder.read_buffer(buffer);
            builder.mark_side_effect();
            (MockPass, ())
        });
        let group = graph.register_persistent_buffer_commit_group(&[(commit, buffer)]);

        graph.compile_topology();
        assert_eq!(graph.storage.execution_queue.len(), 2);
        graph
            .materialize_live_persistent_buffers(&mut registry, &gpu.device)
            .unwrap();
        let physical = *graph
            .storage
            .persistent_buffer_materializations
            .get(&id)
            .expect("live import must have an exact physical token");
        {
            let resolver = PersistentBufferResolver::new(
                &registry,
                &graph.storage.persistent_buffer_materializations,
            );
            assert_eq!(
                resolver.buffer(id).map(Tracked::id),
                Some(physical.token.physical_id)
            );
            assert!(physical.recreated);
            assert_eq!(
                resolver.materialization(id),
                Some(physical),
                "prepare/execute must observe create-and-swap metadata"
            );
        }

        graph
            .materialize_live_persistent_buffers(&mut registry, &gpu.device)
            .unwrap();
        assert!(
            graph.storage.persistent_buffer_materializations[&id].recreated,
            "an idempotent second barrier must preserve the first create-and-swap fact"
        );

        let rejected_encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("unencoded persistent Buffer submit"),
            });
        assert!(matches!(
            graph.submit_command_buffer(&gpu.queue, rejected_encoder.finish()),
            Err(PersistentBufferCommitFailure::LiveGroupNotEncoded)
        ));
        assert!(!registry.is_published(plan.buffer));

        let sentinel = 0x7A11_C0DE_u32;
        let staging = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("persistent Buffer sentinel staging"),
                contents: &sentinel.to_ne_bytes(),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("persistent Buffer sentinel readback"),
            size: size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("persistent Buffer sentinel copy"),
            });
        {
            let resolver = PersistentBufferResolver::new(
                &registry,
                &graph.storage.persistent_buffer_materializations,
            );
            let persistent = resolver
                .buffer(id)
                .expect("live persistent Buffer must resolve");
            encoder.copy_buffer_to_buffer(&staging, 0, persistent, 0, size_of::<u32>() as u64);
            encoder.copy_buffer_to_buffer(persistent, 0, &readback, 0, size_of::<u32>() as u64);
        }
        mark_group_encoded(&graph, group);
        let submitted = graph
            .submit_command_buffer(&gpu.queue, encoder.finish())
            .unwrap();
        graph
            .commit_submitted_persistent_buffers(&mut registry, submitted)
            .unwrap();
        assert!(registry.is_published(plan.buffer));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("persistent Buffer readback poll failed");
        rx.recv()
            .expect("persistent Buffer map callback was dropped")
            .expect("persistent Buffer readback map failed");
        let mapped = slice.get_mapped_range();
        assert_eq!(
            u32::from_ne_bytes(mapped[..size_of::<u32>()].try_into().unwrap()),
            sentinel
        );
        drop(mapped);
        readback.unmap();
    }

    #[test]
    fn aborted_live_graph_retries_same_revision_with_reused_physical_and_fresh_token() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let target = spec(
            256,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            13,
            16,
        );
        let first_plan = registry.plan_buffer(id, 1, 7, target).unwrap();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        let stale_physical = {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let buffer = graph.import_persistent_buffer("Aborted_Persistent", first_plan);
            graph.add_pass("Aborted_Persistent_Writer", |builder| {
                builder.write_buffer(buffer);
                (MockPass, ())
            });
            graph.add_pass("Aborted_Persistent_Consumer", |builder| {
                builder.read_buffer(buffer);
                builder.mark_side_effect();
                (MockPass, ())
            });
            let group = graph.register_persistent_buffer_commit_group(&[(
                first_plan.commit.expect("first attempt needs publication"),
                buffer,
            )]);
            graph.compile_topology();
            graph
                .materialize_live_persistent_buffers(&mut registry, &gpu.device)
                .unwrap();
            let physical = graph
                .storage
                .persistent_buffer_materializations
                .get(&id)
                .expect("first live attempt must materialize")
                .token;
            // Encoding acknowledgement without queue submission is not
            // publication authority. Drop the encoder and graph deliberately.
            mark_group_encoded(&graph, group);
            let encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("aborted persistent Buffer encoder"),
                });
            drop(encoder);
            // Deliberately skip submit and post-submit commit.
            physical
        };
        assert!(!registry.is_published(first_plan.buffer));

        let retry_plan = registry.plan_buffer(id, 2, 7, target).unwrap();
        assert!(!retry_plan.valid);
        assert!(retry_plan.commit.is_some());
        assert!(
            registry.resolve_physical(stale_physical).is_none(),
            "an attempt-scoped token must become stale before retry materialization"
        );

        let mut graph = RenderGraph::new(&mut storage, &arena);
        let buffer = graph.import_persistent_buffer("Retried_Persistent", retry_plan);
        graph.add_pass("Retried_Persistent_Writer", |builder| {
            builder.write_buffer(buffer);
            (MockPass, ())
        });
        graph.add_pass("Retried_Persistent_Consumer", |builder| {
            builder.read_buffer(buffer);
            builder.mark_side_effect();
            (MockPass, ())
        });
        let retry_group = graph.register_persistent_buffer_commit_group(&[(
            retry_plan
                .commit
                .expect("aborted revision still needs publication"),
            buffer,
        )]);
        graph.compile_topology();
        graph
            .materialize_live_persistent_buffers(&mut registry, &gpu.device)
            .unwrap();
        let retried_physical = *graph
            .storage
            .persistent_buffer_materializations
            .get(&id)
            .expect("retry must materialize");
        assert_eq!(
            retried_physical.token.physical_id, stale_physical.physical_id,
            "same spec must reuse the retained physical allocation"
        );
        assert_ne!(retried_physical.token, stale_physical);
        assert!(!retried_physical.recreated);
        mark_group_encoded(&graph, retry_group);
        let submitted = submit_empty_graph(&mut graph, &gpu).unwrap();
        graph
            .commit_submitted_persistent_buffers(&mut registry, submitted)
            .unwrap();
        assert!(registry.is_published(retry_plan.buffer));
    }

    #[test]
    fn submitted_receipt_is_rejected_by_a_different_graph_attempt() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let target = spec(
            256,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            18,
            16,
        );
        let first_plan = registry.plan_buffer(id, 1, 1, target).unwrap();
        let mut first_storage = GraphStorage::new();
        let first_arena = FrameArena::new();
        let foreign_receipt = {
            let mut graph = RenderGraph::new(&mut first_storage, &first_arena);
            let buffer = graph.import_persistent_buffer("Foreign_First", first_plan);
            graph.add_pass("Foreign_First_Writer", |builder| {
                builder.write_buffer(buffer);
                (MockPass, ())
            });
            graph.add_pass("Foreign_First_Consumer", |builder| {
                builder.read_buffer(buffer);
                builder.mark_side_effect();
                (MockPass, ())
            });
            let group = graph.register_persistent_buffer_commit_group(&[(
                first_plan.commit.expect("first graph needs publication"),
                buffer,
            )]);
            graph.compile_topology();
            graph
                .materialize_live_persistent_buffers(&mut registry, &gpu.device)
                .unwrap();
            mark_group_encoded(&graph, group);
            submit_empty_graph(&mut graph, &gpu).unwrap()
        };

        let second_plan = registry.plan_buffer(id, 2, 1, target).unwrap();
        let mut second_storage = GraphStorage::new();
        let second_arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut second_storage, &second_arena);
        let buffer = graph.import_persistent_buffer("Foreign_Second", second_plan);
        graph.add_pass("Foreign_Second_Writer", |builder| {
            builder.write_buffer(buffer);
            (MockPass, ())
        });
        graph.add_pass("Foreign_Second_Consumer", |builder| {
            builder.read_buffer(buffer);
            builder.mark_side_effect();
            (MockPass, ())
        });
        let group = graph.register_persistent_buffer_commit_group(&[(
            second_plan.commit.expect("second graph needs publication"),
            buffer,
        )]);
        graph.compile_topology();
        graph
            .materialize_live_persistent_buffers(&mut registry, &gpu.device)
            .unwrap();
        mark_group_encoded(&graph, group);
        let current_receipt = submit_empty_graph(&mut graph, &gpu).unwrap();

        assert_eq!(
            graph.commit_submitted_persistent_buffers(&mut registry, foreign_receipt),
            Err(PersistentBufferCommitFailure::InvalidSubmissionReceipt)
        );
        assert!(!registry.is_published(second_plan.buffer));
        graph
            .commit_submitted_persistent_buffers(&mut registry, current_receipt)
            .unwrap();
        assert!(registry.is_published(second_plan.buffer));
    }

    #[test]
    fn stale_later_commit_group_prevents_every_group_publication() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let first_id = registry.create_buffer();
        let second_id = registry.create_buffer();
        let target = spec(
            256,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            19,
            16,
        );
        let first_plan = registry.plan_buffer(first_id, 1, 1, target).unwrap();
        let second_plan = registry.plan_buffer(second_id, 1, 1, target).unwrap();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let first = graph.import_persistent_buffer("Atomic_First", first_plan);
        let second = graph.import_persistent_buffer("Atomic_Second", second_plan);
        graph.add_pass("Atomic_First_Writer", |builder| {
            builder.write_buffer(first);
            (MockPass, ())
        });
        graph.add_pass("Atomic_First_Consumer", |builder| {
            builder.read_buffer(first);
            builder.mark_side_effect();
            (MockPass, ())
        });
        graph.add_pass("Atomic_Second_Writer", |builder| {
            builder.write_buffer(second);
            (MockPass, ())
        });
        graph.add_pass("Atomic_Second_Consumer", |builder| {
            builder.read_buffer(second);
            builder.mark_side_effect();
            (MockPass, ())
        });
        let first_group = graph.register_persistent_buffer_commit_group(&[(
            first_plan.commit.expect("first buffer needs publication"),
            first,
        )]);
        let second_group = graph.register_persistent_buffer_commit_group(&[(
            second_plan.commit.expect("second buffer needs publication"),
            second,
        )]);
        graph.compile_topology();
        graph
            .materialize_live_persistent_buffers(&mut registry, &gpu.device)
            .unwrap();
        mark_group_encoded(&graph, first_group);
        mark_group_encoded(&graph, second_group);
        let submitted = submit_empty_graph(&mut graph, &gpu).unwrap();

        let superseding_second = registry.plan_buffer(second_id, 2, 1, target).unwrap();
        assert_eq!(
            graph.commit_submitted_persistent_buffers(&mut registry, submitted),
            Err(PersistentBufferCommitFailure::StaleLogicalToken)
        );
        assert!(
            !registry.is_published(first_plan.buffer),
            "a valid earlier group must remain unpublished when a later group is stale"
        );
        assert!(!registry.is_published(superseding_second.buffer));
    }

    #[test]
    fn persistent_buffer_alias_and_builder_mutation_are_rejected() {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let mut registry = PersistentBufferRegistry::new();
        let id = registry.create_buffer();
        let plan = registry
            .plan_buffer(id, 1, 1, spec(256, wgpu::BufferUsages::STORAGE, 14, 16))
            .unwrap();

        let mut direct_storage = GraphStorage::new();
        let direct_arena = FrameArena::new();
        let mut direct_graph = RenderGraph::new(&mut direct_storage, &direct_arena);
        let direct = direct_graph.import_persistent_buffer("Persistent_Direct_Alias", plan);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                direct_graph.create_alias(direct, "Illegal_Persistent_Alias");
            }))
            .is_err()
        );
        drop(direct_graph);

        let mut builder_storage = GraphStorage::new();
        let builder_arena = FrameArena::new();
        let mut builder_graph = RenderGraph::new(&mut builder_storage, &builder_arena);
        let builder_buffer =
            builder_graph.import_persistent_buffer("Persistent_Builder_Mutation", plan);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                builder_graph.add_pass("Illegal_Persistent_Mutator", |builder| {
                    let _ = builder.mutate_buffer(builder_buffer, "Illegal_Persistent_Version");
                    (MockPass, ())
                });
            }))
            .is_err()
        );
    }

    #[test]
    fn partial_live_commit_group_is_rejected_before_any_physical_creation() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let first_id = registry.create_buffer();
        let second_id = registry.create_buffer();
        let target = spec(
            256,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            15,
            16,
        );
        let first_plan = registry.plan_buffer(first_id, 1, 1, target).unwrap();
        let second_plan = registry.plan_buffer(second_id, 1, 1, target).unwrap();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let first = graph.import_persistent_buffer("Partial_First", first_plan);
        let second = graph.import_persistent_buffer("Partial_Second", second_plan);

        graph.add_pass("Live_First_Writer", |builder| {
            builder.write_buffer(first);
            (MockPass, ())
        });
        graph.add_pass("Live_First_Consumer", |builder| {
            builder.read_buffer(first);
            builder.mark_side_effect();
            (MockPass, ())
        });
        graph.add_pass("Dead_Second_Writer", |builder| {
            builder.write_buffer(second);
            (MockPass, ())
        });
        graph.register_persistent_buffer_commit_group(&[
            (
                first_plan.commit.expect("first buffer needs publication"),
                first,
            ),
            (
                second_plan.commit.expect("second buffer needs publication"),
                second,
            ),
        ]);

        graph.compile_topology();
        assert_eq!(
            graph.materialize_live_persistent_buffers(&mut registry, &gpu.device),
            Err(PersistentBufferGraphMaterializationFailure::PartialCommitGroup)
        );
        assert!(graph.storage.persistent_buffer_materializations.is_empty());
        assert!(registry.records.get(&first_id).unwrap().physical.is_none());
        assert!(registry.records.get(&second_id).unwrap().physical.is_none());
    }

    #[test]
    fn deterministic_batch_failure_precedes_every_physical_creation() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let ordinary_id = registry.create_buffer();
        let oversized_id = registry.create_buffer();
        let ordinary = registry
            .plan_buffer(
                ordinary_id,
                1,
                1,
                spec(
                    256,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    16,
                    16,
                ),
            )
            .unwrap();
        let oversized = registry
            .plan_buffer(
                oversized_id,
                1,
                1,
                spec(
                    gpu.device
                        .limits()
                        .max_buffer_size
                        .checked_add(1)
                        .expect("test device reported an unrepresentable max Buffer size"),
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    17,
                    1,
                ),
            )
            .unwrap();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let ordinary_node = graph.import_persistent_buffer("Preflight_Ordinary", ordinary);
        let oversized_node = graph.import_persistent_buffer("Preflight_Oversized", oversized);

        graph.add_pass("Preflight_Ordinary_Writer", |builder| {
            builder.write_buffer(ordinary_node);
            (MockPass, ())
        });
        graph.add_pass("Preflight_Ordinary_Consumer", |builder| {
            builder.read_buffer(ordinary_node);
            builder.mark_side_effect();
            (MockPass, ())
        });
        graph.add_pass("Preflight_Oversized_Writer", |builder| {
            builder.write_buffer(oversized_node);
            (MockPass, ())
        });
        graph.add_pass("Preflight_Oversized_Consumer", |builder| {
            builder.read_buffer(oversized_node);
            builder.mark_side_effect();
            (MockPass, ())
        });
        graph.register_persistent_buffer_commit_group(&[
            (
                ordinary.commit.expect("ordinary buffer needs publication"),
                ordinary_node,
            ),
            (
                oversized
                    .commit
                    .expect("oversized buffer needs publication"),
                oversized_node,
            ),
        ]);

        graph.compile_topology();
        assert_eq!(
            graph.materialize_live_persistent_buffers(&mut registry, &gpu.device),
            Err(PersistentBufferGraphMaterializationFailure::Registry(
                PersistentBufferMaterializationFailure::ExceedsDeviceLimit
            ))
        );
        assert!(graph.storage.persistent_buffer_materializations.is_empty());
        assert!(
            registry
                .records
                .get(&ordinary_id)
                .unwrap()
                .physical
                .is_none(),
            "a valid sibling must not allocate before the complete batch is preflighted"
        );
        assert!(
            registry
                .records
                .get(&oversized_id)
                .unwrap()
                .physical
                .is_none()
        );
    }
}
