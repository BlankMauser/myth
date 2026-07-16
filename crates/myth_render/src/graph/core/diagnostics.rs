//! Attempt-scoped Render Graph diagnostics.
//!
//! The public snapshot is deliberately plain, immutable data. The mutable
//! state remains owned either by [`GraphStorage`](super::GraphStorage) or the
//! Renderer attempt guard. The current serial implementation updates counters
//! without atomics, locks, logging, or per-event allocation.

use std::cell::Cell;
#[cfg(feature = "rdg_bench_support")]
use std::time::Instant;

const FLAG_SURFACE_SUBOPTIMAL: u8 = 1 << 0;
const FLAG_QUEUE_SUBMITTED: u8 = 1 << 1;
const FLAG_PRESENTED: u8 = 1 << 2;
const FLAG_PANICKED: u8 = 1 << 3;

/// State of one Render Graph attempt.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgAttemptOutcome {
    /// The attempt is still owned by a Renderer or standalone RenderGraph.
    #[default]
    InProgress,
    /// Commands were submitted and all post-submit frame bookkeeping finished.
    Submitted,
    /// A standalone graph was explicitly finished without queue submission.
    CompletedWithoutSubmission,
    /// A window surface did not provide a usable texture.
    SurfaceAcquisitionFailed(RdgSurfaceAcquireFailure),
    /// Neither a window surface nor a headless target was configured.
    NoRenderTarget,
    /// GPU-driven submission was required, or an already selected live GPU
    /// topology could no longer be finalized coherently. No commands were
    /// submitted and the renderer never silently substitutes missing draws.
    GpuDrivenFailed(RdgGpuDrivenFailure),
    /// A DPE-live logical resource could not be materialized from the exact
    /// CPU/GPU state captured for this graph attempt. No commands were
    /// prepared, submitted, or presented.
    ResourceMaterializationFailed(RdgResourceMaterializationFailure),
    /// Pass execution finished, but the command stream failed an explicit
    /// pre-submit completeness contract. No commands were submitted or
    /// presented.
    CommandEncodingFailed(RdgCommandEncodingFailure),
    /// Queue submission succeeded, but exact publication bookkeeping rejected
    /// a stale or conflicting post-submit token. GPU work is not rolled back;
    /// the affected logical state remains invalid for deterministic retry.
    PostSubmitPublicationFailed(RdgPostSubmitPublicationFailure),
    /// The attempt ended before queue submission.
    Abandoned,
    /// Queue submission succeeded, but a later frame stage did not finish.
    SubmittedThenAbandoned,
}

/// Stable classification of failures at the post-DPE resource barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgResourceMaterializationFailure {
    /// The render target changed size or format after the CPU graph captured
    /// its logical external descriptor. The attempt is abandoned before any
    /// transient allocation, upload, or pass preparation.
    RenderTargetDescriptorDrift,
    /// A scene buffer changed after graph construction captured its stamp.
    SceneBufferSnapshotDrift,
    /// ResourceManager already contains incompatible or newer physical state
    /// for the exact scene-buffer snapshot requested by this attempt.
    IncompatibleSceneBufferPhysicalState,
    /// The live scene-buffer batch had already failed or could not publish a
    /// complete set of graph-owned handles.
    IncompleteSceneBufferBatch,
    /// A Group 0 CPU source changed after graph construction captured its
    /// coherent three-buffer snapshot.
    GlobalStateSnapshotDrift,
    /// ResourceManager contains physical buffer state incompatible with the
    /// exact DPE-live Group 0 snapshot.
    IncompatibleGlobalStatePhysicalState,
    /// The live Group 0 batch had already failed or could not publish a
    /// complete graph-attempt-owned physical set.
    IncompleteGlobalStateBatch,
    /// A DPE-live persistent Buffer plan, writer group, or exact physical
    /// generation drifted before the post-DPE materialization barrier.
    PersistentBufferStateDrift,
    /// A DPE-live persistent Buffer write batch was incomplete or did not
    /// match the imported publication contract.
    PersistentBufferIncompleteWriteBatch,
    /// Rounding a persistent Buffer's logical size to its retained physical
    /// capacity overflowed `u64` before any member was allocated.
    PersistentBufferCapacityOverflow,
    /// A persistent Buffer's required retained capacity exceeds the selected
    /// device's maximum Buffer size.
    PersistentBufferExceedsDeviceLimit,
}

/// Stable, allocation-free classification of GPU-driven selection and
/// finalization failures. Unsupported `Auto` candidates fall back before GPU
/// topology and therefore do not terminate an attempt; these reasons describe
/// `Required` failures and post-DPE drift after a backend was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgGpuDrivenFailure {
    UnsupportedRenderPath,
    UnsupportedDevice,
    WorkloadExceedsDeviceLimits,
    SkinnedGeometry,
    MorphTargets,
    InstancedVertexAttributes,
    IncompatibleMaterial,
    NonIndexedGeometry,
    EmptyIndexBuffer,
    EmptyVertexBuffers,
    UnsupportedTopology,
    NonFiniteTransform,
    NonInvertibleTransform,
    NonFiniteBounds,
    ConflictingGeometryEpoch,
    ConflictingMaterialEpoch,
    GenerationExhausted,
    DuplicateInstance,
    TooManyCandidates,
    UnsupportedShadowDomain,
    PersistentPlanRejected,
    SubmissionDescriptorDrift,
    RepresentativeDrift,
    GeometryUnavailable,
    GeometryDrift,
    MaterialUnavailable,
    MaterialDrift,
    GlobalStateUnavailable,
    UnsupportedMaterialState,
    ImmutableBatchPublicationRejected,
}

/// Stable classification of failures detected after execute but before the
/// queue accepts the graph's command buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgCommandEncodingFailure {
    /// At least one DPE-live persistent Buffer patch group did not explicitly
    /// acknowledge that every member write was encoded.
    PersistentBufferWriteBatchIncomplete,
}

/// Stable classification of publication failures discovered only after the
/// queue has accepted a command buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgPostSubmitPublicationFailure {
    PersistentBufferStateDrift,
}

/// Stable engine-level classification of `wgpu` surface acquisition failures.
///
/// Keeping this type independent of `wgpu::CurrentSurfaceTexture` prevents a
/// backend upgrade from changing the diagnostics API or persisted profiler
/// data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgSurfaceAcquireFailure {
    Timeout,
    Occluded,
    Outdated,
    Lost,
    Validation,
}

/// Kind of target acquired by a Renderer attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgRenderTargetKind {
    Surface,
    Headless,
}

/// Stable scene-submission families used by backend-selection diagnostics.
///
/// Main color and depth-prepass submission intentionally share one family:
/// the current GPU Scene policy selects them as one ownership domain and one
/// main-view Cull result may feed both consumers. Shadow families remain
/// separate because either family may have no live views or independently
/// retain CPU ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgSceneSubmissionFamily {
    MainPrepass,
    Shadow2D,
    ShadowCube,
}

impl RdgSceneSubmissionFamily {
    pub const COUNT: usize = 3;

    pub const ALL: [Self; Self::COUNT] = [Self::MainPrepass, Self::Shadow2D, Self::ShadowCube];

    #[inline]
    const fn diagnostics_index(self) -> usize {
        match self {
            Self::MainPrepass => 0,
            Self::Shadow2D => 1,
            Self::ShadowCube => 2,
        }
    }
}

/// Backend ownership selected or observed for one scene-submission family.
///
/// `Inactive` is meaningful rather than an alias for CPU submission. For
/// example, a family may select GPU ownership before graph compilation and
/// still be inactive after DPE removes all of its semantic submissions.
/// `Hybrid` is reserved for a family whose live cohorts are split between
/// CPU-direct and GPU-driven submission.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgSceneSubmissionBackend {
    #[default]
    Inactive,
    CpuDirect,
    GpuDriven,
    Hybrid,
}

/// Stable, coarse reason for one backend-selection decision.
///
/// Policy implementation details such as calibrated thresholds and EWMA
/// coefficients deliberately do not enter this public contract. They may
/// evolve without invalidating profiler data, while these reasons continue
/// to explain whether policy, capability, or lack of work decided the frame.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgSceneSubmissionSelectionReason {
    /// Selection did not run for this family in the graph attempt.
    #[default]
    NotEvaluated,
    /// The family had no semantic workload to own.
    NoWork,
    /// [`crate::settings::GpuDrivenMode::Off`] forced CPU ownership.
    ModeOff,
    /// [`crate::settings::GpuDrivenMode::Required`] forced GPU ownership.
    Required,
    /// `Auto` rejected the GPU path at its constant-time tiny-workload gate,
    /// before paying for exact retained-table preflight.
    AutoSmallWorkload,
    /// `Auto` chose the backend indicated by the family diagnostics from its
    /// deterministic cost model.
    AutoCostModel,
    /// `Auto` retained the preceding backend to satisfy confirmation, dwell,
    /// or enter/exit hysteresis policy.
    AutoHysteresis,
    /// `Auto` preferred the safe current backend because its estimate did not
    /// have enough confidence to justify a switch.
    AutoLowConfidence,
    /// GPU submission could not represent the exact device or workload. Auto
    /// may select CPU; Required exposes the same typed reason as a terminal
    /// attempt outcome.
    Unsupported(RdgGpuDrivenFailure),
}

/// Allocation-free backend facts for one scene-submission family.
///
/// `selected_backend` records the pre-DPE policy result. `live_backend()` is
/// derived from the post-DPE semantic submissions that actually reached the
/// backend finalizers. Keeping both facts prevents a dead GPU topology from
/// being reported as executed work and supports a future hybrid cohort split
/// without a variable-size diagnostics payload.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RdgSceneSubmissionFamilySnapshot {
    selected_backend: RdgSceneSubmissionBackend,
    selection_reason: RdgSceneSubmissionSelectionReason,
    live_cpu_submissions: u32,
    live_gpu_submissions: u32,
}

impl RdgSceneSubmissionFamilySnapshot {
    #[must_use]
    pub const fn selected_backend(self) -> RdgSceneSubmissionBackend {
        self.selected_backend
    }

    #[must_use]
    pub const fn selection_reason(self) -> RdgSceneSubmissionSelectionReason {
        self.selection_reason
    }

    #[must_use]
    pub const fn live_cpu_submissions(self) -> u32 {
        self.live_cpu_submissions
    }

    #[must_use]
    pub const fn live_gpu_submissions(self) -> u32 {
        self.live_gpu_submissions
    }

    /// Backend composition of post-DPE live semantic submissions.
    #[must_use]
    pub const fn live_backend(self) -> RdgSceneSubmissionBackend {
        match (
            self.live_cpu_submissions != 0,
            self.live_gpu_submissions != 0,
        ) {
            (false, false) => RdgSceneSubmissionBackend::Inactive,
            (true, false) => RdgSceneSubmissionBackend::CpuDirect,
            (false, true) => RdgSceneSubmissionBackend::GpuDriven,
            (true, true) => RdgSceneSubmissionBackend::Hybrid,
        }
    }
}

/// Allocation-free inputs and exact projection used by the GPU-driven backend
/// policy for one Renderer attempt.
///
/// A zero snapshot means the strict [`crate::settings::GpuDrivenMode::Off`]
/// path did not capture or evaluate GPU-specific facts. `exact_evaluated()`
/// distinguishes a tiny/coarse Auto decision from an exact preflight whose
/// retained cohort happened to be empty.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RdgGpuDrivenPolicySnapshot {
    pub(crate) extracted_item_count: u32,
    pub(crate) supported_opaque_count: u32,
    pub(crate) retained_candidate_count: u32,
    pub(crate) retained_bin_count: u32,
    pub(crate) dirty_candidate_count: u32,
    pub(crate) shadow_2d_view_count: u32,
    pub(crate) shadow_cube_view_count: u32,
    pub(crate) instance_slot_span: u32,
    pub(crate) batch_slot_span: u32,
    pub(crate) minimum_cohort_members: u32,
    pub(crate) indexed_primitive_count: u64,
    pub(crate) exact_evaluated: bool,
}

impl RdgGpuDrivenPolicySnapshot {
    #[must_use]
    pub const fn extracted_item_count(self) -> u32 {
        self.extracted_item_count
    }

    #[must_use]
    pub const fn supported_opaque_count(self) -> u32 {
        self.supported_opaque_count
    }

    #[must_use]
    pub const fn retained_candidate_count(self) -> u32 {
        self.retained_candidate_count
    }

    #[must_use]
    pub const fn retained_bin_count(self) -> u32 {
        self.retained_bin_count
    }

    #[must_use]
    pub const fn dirty_candidate_count(self) -> u32 {
        self.dirty_candidate_count
    }

    #[must_use]
    pub const fn shadow_2d_view_count(self) -> u32 {
        self.shadow_2d_view_count
    }

    #[must_use]
    pub const fn shadow_cube_view_count(self) -> u32 {
        self.shadow_cube_view_count
    }

    #[must_use]
    pub const fn instance_slot_span(self) -> u32 {
        self.instance_slot_span
    }

    #[must_use]
    pub const fn batch_slot_span(self) -> u32 {
        self.batch_slot_span
    }

    #[must_use]
    pub const fn minimum_cohort_members(self) -> u32 {
        self.minimum_cohort_members
    }

    #[must_use]
    pub const fn indexed_primitive_count(self) -> u64 {
        self.indexed_primitive_count
    }

    #[must_use]
    pub const fn exact_evaluated(self) -> bool {
        self.exact_evaluated
    }
}

/// Typed reason why the persistent Render World rejected its incremental
/// fast path for this attempt.
///
/// The reasons are deliberately coarse and allocation-free. Detailed mutable
/// journal state remains private to the Scene/Render World boundary while this
/// stable taxonomy makes an unexpected O(N) fallback visible to profilers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgRenderWorldRebuildReason {
    InitialMaterialization,
    ExplicitInvalidation,
    MutationOverflow,
    UntrackedSceneStorage,
    UntrackedHierarchy,
    SceneIdentityDrift,
    MutationCursorDrift,
    TopologyChanged,
    AssetAuthorityDrift,
    AssetEpochChanged,
    GpuFactProfileChanged,
    ConsumerRevisionGap,
    ConsumerJournalOverflow,
    InvariantDrift,
}

/// Fixed-width observability for the incremental Scene -> Render World -> GPU
/// Scene frontend of one Renderer attempt.
///
/// Shipping builds update this snapshot only from facts already produced by
/// the synchronization transactions. No timer reads, allocation, hashing, or
/// additional scene traversal is performed for diagnostics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RdgRenderWorldSnapshot {
    pub(crate) scene_id: Option<u32>,
    pub(crate) revision: u64,
    pub(crate) resident_scene_count: u32,
    pub(crate) mutation_count: u32,
    pub(crate) hierarchy_dirty_roots_examined: u32,
    pub(crate) hierarchy_dirty_parent_links_examined: u32,
    pub(crate) hierarchy_nodes_visited: u32,
    pub(crate) hierarchy_nodes_updated: u32,
    pub(crate) hierarchy_full_traversal: bool,
    pub(crate) render_item_count: u32,
    pub(crate) render_items_reused: u32,
    pub(crate) render_items_examined: u32,
    pub(crate) render_items_added: u32,
    pub(crate) render_items_updated: u32,
    pub(crate) render_items_removed: u32,
    pub(crate) render_items_rebuilt: u32,
    pub(crate) gpu_members_reused: u32,
    pub(crate) gpu_members_examined: u32,
    pub(crate) gpu_members_updated: u32,
    pub(crate) gpu_members_rebuilt: u32,
    pub(crate) shadow_casters_examined: u32,
    pub(crate) shadow_aggregate_rebuilt: bool,
    pub(crate) submitted_history_settled: u32,
    pub(crate) retained_bytes: u64,
    pub(crate) rebuild_reason: Option<RdgRenderWorldRebuildReason>,
}

impl RdgRenderWorldSnapshot {
    #[must_use]
    pub const fn scene_id(self) -> Option<u32> {
        self.scene_id
    }

    #[must_use]
    pub const fn revision(self) -> u64 {
        self.revision
    }

    #[must_use]
    pub const fn resident_scene_count(self) -> u32 {
        self.resident_scene_count
    }

    #[must_use]
    pub const fn mutation_count(self) -> u32 {
        self.mutation_count
    }

    /// Dirty transform candidates considered while coalescing overlapping
    /// hierarchy updates into disjoint roots.
    #[must_use]
    pub const fn hierarchy_dirty_roots_examined(self) -> u32 {
        self.hierarchy_dirty_roots_examined
    }

    /// Parent links followed by the dirty-root coalescer. Together with
    /// [`Self::hierarchy_dirty_roots_examined`], this exposes accidental
    /// quadratic hierarchy work without requiring profiler-only traversal.
    #[must_use]
    pub const fn hierarchy_dirty_parent_links_examined(self) -> u32 {
        self.hierarchy_dirty_parent_links_examined
    }

    #[must_use]
    pub const fn hierarchy_nodes_visited(self) -> u32 {
        self.hierarchy_nodes_visited
    }

    #[must_use]
    pub const fn hierarchy_nodes_updated(self) -> u32 {
        self.hierarchy_nodes_updated
    }

    #[must_use]
    pub const fn hierarchy_full_traversal(self) -> bool {
        self.hierarchy_full_traversal
    }

    #[must_use]
    pub const fn render_item_count(self) -> u32 {
        self.render_item_count
    }

    #[must_use]
    pub const fn render_items_reused(self) -> u32 {
        self.render_items_reused
    }

    #[must_use]
    pub const fn render_items_examined(self) -> u32 {
        self.render_items_examined
    }

    #[must_use]
    pub const fn render_items_added(self) -> u32 {
        self.render_items_added
    }

    #[must_use]
    pub const fn render_items_updated(self) -> u32 {
        self.render_items_updated
    }

    #[must_use]
    pub const fn render_items_removed(self) -> u32 {
        self.render_items_removed
    }

    #[must_use]
    pub const fn render_items_rebuilt(self) -> u32 {
        self.render_items_rebuilt
    }

    #[must_use]
    pub const fn gpu_members_reused(self) -> u32 {
        self.gpu_members_reused
    }

    #[must_use]
    pub const fn gpu_members_examined(self) -> u32 {
        self.gpu_members_examined
    }

    #[must_use]
    pub const fn gpu_members_updated(self) -> u32 {
        self.gpu_members_updated
    }

    #[must_use]
    pub const fn gpu_members_rebuilt(self) -> u32 {
        self.gpu_members_rebuilt
    }

    #[must_use]
    pub const fn shadow_casters_examined(self) -> u32 {
        self.shadow_casters_examined
    }

    #[must_use]
    pub const fn shadow_aggregate_rebuilt(self) -> bool {
        self.shadow_aggregate_rebuilt
    }

    #[must_use]
    pub const fn submitted_history_settled(self) -> u32 {
        self.submitted_history_settled
    }

    #[must_use]
    pub const fn retained_bytes(self) -> u64 {
        self.retained_bytes
    }

    #[must_use]
    pub const fn rebuild_reason(self) -> Option<RdgRenderWorldRebuildReason> {
        self.rebuild_reason
    }
}

/// Last phase entered by a Render Graph attempt.
///
/// The phase is useful for abandoned attempts: counters from later phases are
/// intentionally left untouched instead of being reported as completed zero
/// work. Detailed timing is layered onto these stable phase boundaries rather
/// than timing individual passes on the shipping hot path.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgFramePhase {
    #[default]
    Begin,
    Extract,
    Maintenance,
    AwaitingRender,
    Cull,
    BuildGraph,
    CompileGraph,
    AcquireTarget,
    MaterializeResources,
    PreparePasses,
    FinalizeDrawPlans,
    RecordCommands,
    Submit,
    Publish,
    Present,
    Complete,
}

#[cfg(feature = "rdg_bench_support")]
impl RdgFramePhase {
    /// Number of stable RDG attempt phases exposed by benchmark diagnostics.
    pub const COUNT: usize = 16;

    /// Every RDG attempt phase in execution order.
    ///
    /// This is benchmark support rather than a shipping profiler contract.
    /// Consumers can use it with [`RdgFrameDiagnostics::phase_elapsed_ns`]
    /// without allocating a temporary phase list.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Begin,
        Self::Extract,
        Self::Maintenance,
        Self::AwaitingRender,
        Self::Cull,
        Self::BuildGraph,
        Self::CompileGraph,
        Self::AcquireTarget,
        Self::MaterializeResources,
        Self::PreparePasses,
        Self::FinalizeDrawPlans,
        Self::RecordCommands,
        Self::Submit,
        Self::Publish,
        Self::Present,
        Self::Complete,
    ];

    const fn timing_index(self) -> usize {
        match self {
            Self::Begin => 0,
            Self::Extract => 1,
            Self::Maintenance => 2,
            Self::AwaitingRender => 3,
            Self::Cull => 4,
            Self::BuildGraph => 5,
            Self::CompileGraph => 6,
            Self::AcquireTarget => 7,
            Self::MaterializeResources => 8,
            Self::PreparePasses => 9,
            Self::FinalizeDrawPlans => 10,
            Self::RecordCommands => 11,
            Self::Submit => 12,
            Self::Publish => 13,
            Self::Present => 14,
            Self::Complete => 15,
        }
    }
}

/// Benchmark-only subdivisions of the post-DPE scene preparation pipeline.
///
/// Unlike [`RdgFramePhase`], these intervals may be nested inside a macro
/// phase. They are deliberately absent from shipping builds so the production
/// hot path carries neither timer reads nor telemetry storage.
#[cfg(feature = "rdg_bench_support")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdgDetailPhase {
    GpuBackendSelection,
    GpuSceneCommit,
    CpuIntentCollection,
    TransientResources,
    SceneBuffers,
    GlobalBuffers,
    PersistentTextures,
    PersistentBuffers,
    AssetTexture,
    GlobalBindGroup,
    LiveSourceUnion,
    MaterialPlan,
    Geometry,
    SourceGather,
    SourcePlan,
    SourceMaterialize,
    SourceFinalize,
    PipelineGather,
    PipelineCoordinate,
    PipelineFeatureFinalize,
    PipelineFinalize,
}

#[cfg(feature = "rdg_bench_support")]
impl RdgDetailPhase {
    pub const COUNT: usize = 21;

    pub const ALL: [Self; Self::COUNT] = [
        Self::GpuBackendSelection,
        Self::GpuSceneCommit,
        Self::CpuIntentCollection,
        Self::TransientResources,
        Self::SceneBuffers,
        Self::GlobalBuffers,
        Self::PersistentTextures,
        Self::PersistentBuffers,
        Self::AssetTexture,
        Self::GlobalBindGroup,
        Self::LiveSourceUnion,
        Self::MaterialPlan,
        Self::Geometry,
        Self::SourceGather,
        Self::SourcePlan,
        Self::SourceMaterialize,
        Self::SourceFinalize,
        Self::PipelineGather,
        Self::PipelineCoordinate,
        Self::PipelineFeatureFinalize,
        Self::PipelineFinalize,
    ];

    const fn timing_index(self) -> usize {
        match self {
            Self::GpuBackendSelection => 0,
            Self::GpuSceneCommit => 1,
            Self::CpuIntentCollection => 2,
            Self::TransientResources => 3,
            Self::SceneBuffers => 4,
            Self::GlobalBuffers => 5,
            Self::PersistentTextures => 6,
            Self::PersistentBuffers => 7,
            Self::AssetTexture => 8,
            Self::GlobalBindGroup => 9,
            Self::LiveSourceUnion => 10,
            Self::MaterialPlan => 11,
            Self::Geometry => 12,
            Self::SourceGather => 13,
            Self::SourcePlan => 14,
            Self::SourceMaterialize => 15,
            Self::SourceFinalize => 16,
            Self::PipelineGather => 17,
            Self::PipelineCoordinate => 18,
            Self::PipelineFeatureFinalize => 19,
            Self::PipelineFinalize => 20,
        }
    }
}

/// Immutable diagnostics for one Renderer or standalone Render Graph attempt.
///
/// `live_passes()` is optional by design. `None` means topology compilation
/// never completed; `Some(0)` means compilation completed and DPE removed every
/// pass. `executed_passes()` counts completed `PassNode::execute` callbacks —
/// commands recorded on the CPU — and does not claim that the GPU has finished
/// executing them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RdgFrameDiagnostics {
    attempt_id: u64,
    renderer_frame: Option<u64>,
    outcome: RdgAttemptOutcome,
    phase: RdgFramePhase,
    declared_passes: usize,
    folded_passes: usize,
    live_passes: Option<usize>,
    prepared_passes: usize,
    executed_passes: usize,
    scene_submission_families: [RdgSceneSubmissionFamilySnapshot; RdgSceneSubmissionFamily::COUNT],
    gpu_driven_policy: RdgGpuDrivenPolicySnapshot,
    render_world: RdgRenderWorldSnapshot,
    target: Option<RdgRenderTargetKind>,
    flags: u8,
    #[cfg(feature = "rdg_bench_support")]
    phase_elapsed_ns: [u64; RdgFramePhase::COUNT],
    #[cfg(feature = "rdg_bench_support")]
    detail_elapsed_ns: [u64; RdgDetailPhase::COUNT],
}

impl RdgFrameDiagnostics {
    fn new(attempt_id: u64, renderer_frame: Option<u64>, phase: RdgFramePhase) -> Self {
        Self {
            attempt_id,
            renderer_frame,
            outcome: RdgAttemptOutcome::InProgress,
            phase,
            declared_passes: 0,
            folded_passes: 0,
            live_passes: None,
            prepared_passes: 0,
            executed_passes: 0,
            scene_submission_families: [RdgSceneSubmissionFamilySnapshot::default();
                RdgSceneSubmissionFamily::COUNT],
            gpu_driven_policy: RdgGpuDrivenPolicySnapshot::default(),
            render_world: RdgRenderWorldSnapshot::default(),
            target: None,
            flags: 0,
            #[cfg(feature = "rdg_bench_support")]
            phase_elapsed_ns: [0; RdgFramePhase::COUNT],
            #[cfg(feature = "rdg_bench_support")]
            detail_elapsed_ns: [0; RdgDetailPhase::COUNT],
        }
    }

    /// Monotonic serial assigned by the owning `GraphStorage`.
    #[must_use]
    pub const fn attempt_id(&self) -> u64 {
        self.attempt_id
    }

    /// Caller-provided Renderer frame number, or `None` for standalone graphs.
    #[must_use]
    pub const fn renderer_frame(&self) -> Option<u64> {
        self.renderer_frame
    }

    #[must_use]
    pub const fn outcome(&self) -> RdgAttemptOutcome {
        self.outcome
    }

    #[must_use]
    pub const fn phase(&self) -> RdgFramePhase {
        self.phase
    }

    /// Passes whose setup closure completed and installed a valid node.
    #[must_use]
    pub const fn declared_passes(&self) -> usize {
        self.declared_passes
    }

    /// Pure forwarding passes contracted during successful fold operations.
    #[must_use]
    pub const fn folded_passes(&self) -> usize {
        self.folded_passes
    }

    /// DPE-live passes, or `None` when topology compilation did not complete.
    #[must_use]
    pub const fn live_passes(&self) -> Option<usize> {
        self.live_passes
    }

    /// Live pass `prepare` callbacks that returned normally.
    #[must_use]
    pub const fn prepared_passes(&self) -> usize {
        self.prepared_passes
    }

    /// Live pass `execute` callbacks that returned after recording commands.
    #[must_use]
    pub const fn executed_passes(&self) -> usize {
        self.executed_passes
    }

    /// Pre-DPE selection and post-DPE live-backend facts for one scene family.
    #[must_use]
    pub const fn scene_submission_family(
        &self,
        family: RdgSceneSubmissionFamily,
    ) -> RdgSceneSubmissionFamilySnapshot {
        self.scene_submission_families[family.diagnostics_index()]
    }

    /// GPU-driven policy inputs/projection for this attempt. The strict Off
    /// baseline returns the all-zero default without GPU-specific planning.
    #[must_use]
    pub const fn gpu_driven_policy(&self) -> RdgGpuDrivenPolicySnapshot {
        self.gpu_driven_policy
    }

    /// Incremental frontend facts for this attempt. Standalone RenderGraphs
    /// and attempts which never reached Scene extraction return the all-zero
    /// default snapshot.
    #[must_use]
    pub const fn render_world(&self) -> RdgRenderWorldSnapshot {
        self.render_world
    }

    #[must_use]
    pub const fn target(&self) -> Option<RdgRenderTargetKind> {
        self.target
    }

    #[must_use]
    pub const fn surface_was_suboptimal(&self) -> bool {
        self.flags & FLAG_SURFACE_SUBOPTIMAL != 0
    }

    /// Whether `wgpu::Queue::submit` returned for this attempt.
    #[must_use]
    pub const fn queue_submitted(&self) -> bool {
        self.flags & FLAG_QUEUE_SUBMITTED != 0
    }

    /// Whether a window surface texture was presented.
    #[must_use]
    pub const fn presented(&self) -> bool {
        self.flags & FLAG_PRESENTED != 0
    }

    /// Whether the terminal snapshot was published while unwinding a panic.
    #[must_use]
    pub const fn panicked(&self) -> bool {
        self.flags & FLAG_PANICKED != 0
    }

    /// CPU wall-clock nanoseconds attributed to one RDG phase.
    ///
    /// Time is accumulated when the attempt crosses a phase boundary. A
    /// terminal snapshot also seals its last active phase, including RAII
    /// abandonment and panic paths. An in-progress snapshot therefore reports
    /// completed phase intervals but not the still-running interval.
    #[cfg(feature = "rdg_bench_support")]
    #[must_use]
    pub const fn phase_elapsed_ns(&self, phase: RdgFramePhase) -> u64 {
        self.phase_elapsed_ns[phase.timing_index()]
    }

    /// CPU wall-clock nanoseconds attributed to one benchmark-only scene
    /// preparation subdivision.
    #[cfg(feature = "rdg_bench_support")]
    #[must_use]
    pub const fn detail_elapsed_ns(&self, phase: RdgDetailPhase) -> u64 {
        self.detail_elapsed_ns[phase.timing_index()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RdgAttemptOwner {
    Renderer,
    #[cfg(test)]
    StandaloneGraph,
}

/// Mutable attempt recorder owned by `RendererState` for Renderer attempts.
///
/// Every field uses single-threaded interior mutability. This lets the
/// Renderer guard and the graph view share the recorder without extending an
/// exclusive borrow of `GraphStorage`; it is still lock-free, non-atomic, and
/// allocation-free. Future worker telemetry is reduced into this serial owner
/// rather than mutating it concurrently.
#[derive(Debug)]
pub(crate) struct RdgDiagnosticsRecorder {
    next_attempt_id: Cell<u64>,
    active_owner: Cell<Option<RdgAttemptOwner>>,
    active_attempt_id: Cell<u64>,
    active_renderer_frame: Cell<Option<u64>>,
    requested_outcome: Cell<Option<RdgAttemptOutcome>>,
    phase: Cell<RdgFramePhase>,
    declared_passes: Cell<usize>,
    folded_passes: Cell<usize>,
    live_passes: Cell<Option<usize>>,
    prepared_passes: Cell<usize>,
    executed_passes: Cell<usize>,
    scene_submission_families:
        [Cell<RdgSceneSubmissionFamilySnapshot>; RdgSceneSubmissionFamily::COUNT],
    gpu_driven_policy: Cell<RdgGpuDrivenPolicySnapshot>,
    render_world: Cell<RdgRenderWorldSnapshot>,
    target: Cell<Option<RdgRenderTargetKind>>,
    flags: Cell<u8>,
    latest: Cell<Option<RdgFrameDiagnostics>>,
    #[cfg(feature = "rdg_bench_support")]
    phase_started_at: Cell<Option<Instant>>,
    #[cfg(feature = "rdg_bench_support")]
    phase_elapsed_ns: [Cell<u64>; RdgFramePhase::COUNT],
    #[cfg(feature = "rdg_bench_support")]
    detail_elapsed_ns: [Cell<u64>; RdgDetailPhase::COUNT],
}

impl Default for RdgDiagnosticsRecorder {
    fn default() -> Self {
        Self {
            next_attempt_id: Cell::new(0),
            active_owner: Cell::new(None),
            active_attempt_id: Cell::new(0),
            active_renderer_frame: Cell::new(None),
            requested_outcome: Cell::new(None),
            phase: Cell::new(RdgFramePhase::Begin),
            declared_passes: Cell::new(0),
            folded_passes: Cell::new(0),
            live_passes: Cell::new(None),
            prepared_passes: Cell::new(0),
            executed_passes: Cell::new(0),
            scene_submission_families: std::array::from_fn(|_| {
                Cell::new(RdgSceneSubmissionFamilySnapshot::default())
            }),
            gpu_driven_policy: Cell::new(RdgGpuDrivenPolicySnapshot::default()),
            render_world: Cell::new(RdgRenderWorldSnapshot::default()),
            target: Cell::new(None),
            flags: Cell::new(0),
            latest: Cell::new(None),
            #[cfg(feature = "rdg_bench_support")]
            phase_started_at: Cell::new(None),
            #[cfg(feature = "rdg_bench_support")]
            phase_elapsed_ns: std::array::from_fn(|_| Cell::new(0)),
            #[cfg(feature = "rdg_bench_support")]
            detail_elapsed_ns: std::array::from_fn(|_| Cell::new(0)),
        }
    }
}

impl RdgDiagnosticsRecorder {
    pub(crate) fn start_renderer_attempt(&self, renderer_frame: u64) {
        self.start(RdgAttemptOwner::Renderer, Some(renderer_frame));
    }

    #[cfg(test)]
    pub(crate) fn start_standalone_attempt(&self) {
        self.start(RdgAttemptOwner::StandaloneGraph, None);
        self.enter_phase(RdgFramePhase::BuildGraph);
    }

    fn start(&self, owner: RdgAttemptOwner, renderer_frame: Option<u64>) {
        if self.active_owner.get().is_some() {
            self.finish_active();
        }

        let attempt_id = self
            .next_attempt_id
            .get()
            .checked_add(1)
            .expect("RDG attempt serial overflowed");
        self.next_attempt_id.set(attempt_id);
        self.active_owner.set(Some(owner));
        self.active_attempt_id.set(attempt_id);
        self.active_renderer_frame.set(renderer_frame);
        self.requested_outcome.set(None);
        self.phase.set(RdgFramePhase::Begin);
        self.declared_passes.set(0);
        self.folded_passes.set(0);
        self.live_passes.set(None);
        self.prepared_passes.set(0);
        self.executed_passes.set(0);
        for family in &self.scene_submission_families {
            family.set(RdgSceneSubmissionFamilySnapshot::default());
        }
        self.gpu_driven_policy
            .set(RdgGpuDrivenPolicySnapshot::default());
        self.render_world.set(RdgRenderWorldSnapshot::default());
        self.target.set(None);
        self.flags.set(0);
        #[cfg(feature = "rdg_bench_support")]
        {
            for elapsed in &self.phase_elapsed_ns {
                elapsed.set(0);
            }
            for elapsed in &self.detail_elapsed_ns {
                elapsed.set(0);
            }
            self.phase_started_at.set(Some(Instant::now()));
        }
    }

    #[inline]
    pub(crate) fn enter_phase(&self, phase: RdgFramePhase) {
        self.assert_active();
        #[cfg(feature = "rdg_bench_support")]
        self.seal_phase_interval(Instant::now(), true);
        self.phase.set(phase);
    }

    #[inline]
    pub(crate) fn record_declared_pass(&self) {
        self.assert_active();
        self.declared_passes
            .set(self.declared_passes.get().saturating_add(1));
    }

    #[inline]
    pub(crate) fn record_folded_pass(&self) {
        self.assert_active();
        let folded = self.folded_passes.get().saturating_add(1);
        self.folded_passes.set(folded);
        debug_assert!(folded <= self.declared_passes.get());
    }

    pub(crate) fn record_compiled_live_passes(&self, live_passes: usize) {
        self.assert_active();
        debug_assert!(self.live_passes.get().is_none());
        let declared = self.declared_passes.get();
        debug_assert!(live_passes <= declared);
        debug_assert!(self.folded_passes.get() <= declared - live_passes);
        self.live_passes.set(Some(live_passes));
    }

    #[inline]
    pub(crate) fn record_prepared_pass(&self) {
        self.assert_active();
        let prepared = self.prepared_passes.get().saturating_add(1);
        self.prepared_passes.set(prepared);
        debug_assert!(self.live_passes.get().is_some_and(|live| prepared <= live));
    }

    #[inline]
    pub(crate) fn record_executed_pass(&self) {
        self.assert_active();
        let executed = self.executed_passes.get().saturating_add(1);
        self.executed_passes.set(executed);
        debug_assert!(executed <= self.prepared_passes.get());
    }

    /// Overwrite the pre-DPE policy decision for one fixed scene family.
    ///
    /// The caller supplies facts it already computed for backend selection;
    /// diagnostics never inspect scene items or graph topology. The fixed
    /// family array makes this operation O(1), allocation-free, and suitable
    /// for every Renderer attempt.
    #[inline]
    pub(crate) fn record_scene_submission_selection(
        &self,
        family: RdgSceneSubmissionFamily,
        backend: RdgSceneSubmissionBackend,
        reason: RdgSceneSubmissionSelectionReason,
    ) {
        self.assert_active();
        let cell = &self.scene_submission_families[family.diagnostics_index()];
        let mut snapshot = cell.get();
        snapshot.selected_backend = backend;
        snapshot.selection_reason = reason;
        cell.set(snapshot);
    }

    /// Publish the already-computed policy facts for observability. This is
    /// one fixed-size Cell write: no scene scan, allocation, clock read, or
    /// synchronization is introduced by diagnostics.
    #[inline]
    pub(crate) fn record_gpu_driven_policy(&self, snapshot: RdgGpuDrivenPolicySnapshot) {
        self.assert_active();
        self.gpu_driven_policy.set(snapshot);
    }

    /// Overwrite the fixed-width incremental frontend snapshot with facts
    /// already produced by the current transaction.
    #[inline]
    pub(crate) fn record_render_world(&self, snapshot: RdgRenderWorldSnapshot) {
        self.assert_active();
        self.render_world.set(snapshot);
    }

    /// Mutate the current fixed-width snapshot without exposing the recorder's
    /// interior storage. Callers must only merge counters they already own;
    /// diagnostics never request additional Scene/GPU Scene work.
    #[inline]
    pub(crate) fn update_render_world(&self, update: impl FnOnce(&mut RdgRenderWorldSnapshot)) {
        self.assert_active();
        let mut snapshot = self.render_world.get();
        update(&mut snapshot);
        self.render_world.set(snapshot);
    }

    /// Add post-DPE semantic submissions that reached one concrete backend.
    ///
    /// Calls may arrive from multiple live views or cohorts. Separate CPU and
    /// GPU counters let [`RdgSceneSubmissionFamilySnapshot::live_backend`]
    /// derive hybrid ownership without storing a variable-size event stream.
    #[inline]
    pub(crate) fn record_live_scene_submissions(
        &self,
        family: RdgSceneSubmissionFamily,
        backend: RdgSceneSubmissionBackend,
        count: u32,
    ) {
        self.assert_active();
        debug_assert!(
            matches!(
                backend,
                RdgSceneSubmissionBackend::CpuDirect | RdgSceneSubmissionBackend::GpuDriven
            ),
            "live scene submissions require one concrete backend"
        );
        if count == 0 {
            return;
        }
        let cell = &self.scene_submission_families[family.diagnostics_index()];
        let mut snapshot = cell.get();
        match backend {
            RdgSceneSubmissionBackend::CpuDirect => {
                snapshot.live_cpu_submissions = snapshot.live_cpu_submissions.saturating_add(count);
            }
            RdgSceneSubmissionBackend::GpuDriven => {
                snapshot.live_gpu_submissions = snapshot.live_gpu_submissions.saturating_add(count);
            }
            RdgSceneSubmissionBackend::Inactive | RdgSceneSubmissionBackend::Hybrid => return,
        }
        cell.set(snapshot);
    }

    pub(crate) fn mark_target_acquired(
        &self,
        target: RdgRenderTargetKind,
        surface_suboptimal: bool,
    ) {
        self.assert_active();
        self.target.set(Some(target));
        self.set_flag(FLAG_SURFACE_SUBOPTIMAL, surface_suboptimal);
    }

    /// Records a surface failure and suppresses only consecutive failures with
    /// the same reason. A changed reason must be reported so a timeout streak
    /// cannot hide a later lost surface or validation error.
    pub(crate) fn mark_surface_acquisition_failed(&self, reason: RdgSurfaceAcquireFailure) -> bool {
        let should_report = !self.latest.get().is_some_and(|latest| {
            latest.outcome() == RdgAttemptOutcome::SurfaceAcquisitionFailed(reason)
        });
        self.request_outcome(RdgAttemptOutcome::SurfaceAcquisitionFailed(reason));
        should_report
    }

    /// Records a missing target and suppresses repeated identical reports.
    pub(crate) fn mark_no_render_target(&self) -> bool {
        let should_report = !self
            .latest
            .get()
            .is_some_and(|latest| latest.outcome() == RdgAttemptOutcome::NoRenderTarget);
        self.request_outcome(RdgAttemptOutcome::NoRenderTarget);
        should_report
    }

    /// Terminates a GPU-driven attempt at either its required-policy selection
    /// gate or its immutable post-DPE batch publication gate.
    pub(crate) fn mark_gpu_driven_failed(&self, reason: RdgGpuDrivenFailure) {
        self.request_outcome(RdgAttemptOutcome::GpuDrivenFailed(reason));
    }

    /// Terminates an attempt cleanly before prepare when an exact post-DPE
    /// resource recipe can no longer be satisfied.
    pub(crate) fn mark_resource_materialization_failed(
        &self,
        reason: RdgResourceMaterializationFailure,
    ) {
        self.request_outcome(RdgAttemptOutcome::ResourceMaterializationFailed(reason));
    }

    pub(crate) fn mark_command_encoding_failed(&self, reason: RdgCommandEncodingFailure) {
        self.assert_active();
        debug_assert!(!self.flag(FLAG_QUEUE_SUBMITTED));
        self.request_outcome(RdgAttemptOutcome::CommandEncodingFailed(reason));
    }

    pub(crate) fn mark_queue_submitted(&self) {
        self.assert_active();
        // Submission is an irreversible external fact. Publish it before any
        // diagnostic consistency check that may panic in a debug build.
        self.set_flag(FLAG_QUEUE_SUBMITTED, true);
        let Some(live) = self.live_passes.get() else {
            panic!("RDG queue submission recorded before topology compilation completed");
        };
        debug_assert_eq!(self.prepared_passes.get(), live);
        debug_assert_eq!(self.executed_passes.get(), live);
    }

    pub(crate) fn mark_post_submit_publication_failed(
        &self,
        reason: RdgPostSubmitPublicationFailure,
    ) {
        self.assert_active();
        debug_assert!(self.flag(FLAG_QUEUE_SUBMITTED));
        self.request_outcome(RdgAttemptOutcome::PostSubmitPublicationFailed(reason));
    }

    pub(crate) fn mark_presented(&self) {
        self.assert_active();
        // Presentation has already returned at this call site; retain that
        // truth even if a following debug invariant detects recorder drift.
        self.set_flag(FLAG_PRESENTED, true);
        debug_assert!(self.flag(FLAG_QUEUE_SUBMITTED));
        debug_assert_eq!(self.target.get(), Some(RdgRenderTargetKind::Surface));
    }

    pub(crate) fn mark_completed(&self) {
        self.assert_active();
        debug_assert!(self.flag(FLAG_QUEUE_SUBMITTED));
        #[cfg(feature = "rdg_bench_support")]
        self.enter_phase(RdgFramePhase::Complete);
        #[cfg(not(feature = "rdg_bench_support"))]
        self.phase.set(RdgFramePhase::Complete);
        self.request_outcome(RdgAttemptOutcome::Submitted);
    }

    fn request_outcome(&self, outcome: RdgAttemptOutcome) {
        self.assert_active();
        if self.requested_outcome.get().is_none() {
            self.requested_outcome.set(Some(outcome));
        }
    }

    #[must_use]
    pub(crate) fn active_owner(&self) -> Option<RdgAttemptOwner> {
        self.active_owner.get()
    }

    #[must_use]
    pub(crate) fn current(&self) -> Option<RdgFrameDiagnostics> {
        self.active_owner
            .get()
            .map(|_| self.snapshot(RdgAttemptOutcome::InProgress, false))
    }

    #[must_use]
    pub(crate) fn latest(&self) -> Option<RdgFrameDiagnostics> {
        self.latest.get()
    }

    pub(crate) fn finish_renderer_attempt(&self, attempt_id: u64) {
        if self.active_owner() == Some(RdgAttemptOwner::Renderer)
            && self.active_attempt_id.get() == attempt_id
        {
            self.finish_active();
        }
    }

    #[cfg(test)]
    pub(crate) fn finish_standalone_attempt(&self) {
        if self.active_owner() == Some(RdgAttemptOwner::StandaloneGraph) {
            self.finish_active();
        }
    }

    fn finish_active(&self) {
        if self.active_owner.get().is_none() {
            return;
        }
        let fallback = if self.flag(FLAG_QUEUE_SUBMITTED) {
            RdgAttemptOutcome::SubmittedThenAbandoned
        } else {
            RdgAttemptOutcome::Abandoned
        };
        let outcome = self.requested_outcome.get().unwrap_or(fallback);
        let panicking = std::thread::panicking();
        #[cfg(feature = "rdg_bench_support")]
        self.seal_phase_interval(Instant::now(), false);
        let snapshot = self.snapshot(outcome, panicking);
        // Never introduce a second panic from RAII cleanup while unwinding the
        // original frame failure. The preserved snapshot remains inspectable.
        if !panicking {
            debug_validate_snapshot(&snapshot);
        }
        self.latest.set(Some(snapshot));
        self.active_owner.set(None);
    }

    fn snapshot(&self, outcome: RdgAttemptOutcome, panicked: bool) -> RdgFrameDiagnostics {
        RdgFrameDiagnostics {
            attempt_id: self.active_attempt_id.get(),
            renderer_frame: self.active_renderer_frame.get(),
            outcome,
            phase: self.phase.get(),
            declared_passes: self.declared_passes.get(),
            folded_passes: self.folded_passes.get(),
            live_passes: self.live_passes.get(),
            prepared_passes: self.prepared_passes.get(),
            executed_passes: self.executed_passes.get(),
            scene_submission_families: std::array::from_fn(|index| {
                self.scene_submission_families[index].get()
            }),
            gpu_driven_policy: self.gpu_driven_policy.get(),
            render_world: self.render_world.get(),
            target: self.target.get(),
            flags: self.flags.get() | if panicked { FLAG_PANICKED } else { 0 },
            #[cfg(feature = "rdg_bench_support")]
            phase_elapsed_ns: std::array::from_fn(|index| self.phase_elapsed_ns[index].get()),
            #[cfg(feature = "rdg_bench_support")]
            detail_elapsed_ns: std::array::from_fn(|index| self.detail_elapsed_ns[index].get()),
        }
    }

    #[cfg(feature = "rdg_bench_support")]
    pub(crate) fn detail_timer(&self, phase: RdgDetailPhase) -> RdgDetailTimer<'_> {
        self.assert_active();
        RdgDetailTimer {
            recorder: self,
            phase,
            started_at: Instant::now(),
        }
    }

    #[cfg(feature = "rdg_bench_support")]
    fn record_detail_elapsed(&self, phase: RdgDetailPhase, elapsed_ns: u64) {
        let slot = &self.detail_elapsed_ns[phase.timing_index()];
        slot.set(slot.get().saturating_add(elapsed_ns));
    }

    #[cfg(feature = "rdg_bench_support")]
    fn seal_phase_interval(&self, now: Instant, continue_timing: bool) {
        if let Some(started_at) = self.phase_started_at.get() {
            let phase_elapsed = duration_ns(now.saturating_duration_since(started_at));
            let slot = &self.phase_elapsed_ns[self.phase.get().timing_index()];
            slot.set(slot.get().saturating_add(phase_elapsed));
        }
        self.phase_started_at.set(continue_timing.then_some(now));
    }

    #[inline]
    fn flag(&self, flag: u8) -> bool {
        self.flags.get() & flag != 0
    }

    #[inline]
    fn set_flag(&self, flag: u8, enabled: bool) {
        let current = self.flags.get();
        self.flags.set(if enabled {
            current | flag
        } else {
            current & !flag
        });
    }

    fn assert_active(&self) {
        assert!(
            self.active_owner.get().is_some(),
            "RDG diagnostics event recorded without an active attempt"
        );
    }
}

#[cfg(feature = "rdg_bench_support")]
pub(crate) struct RdgDetailTimer<'a> {
    recorder: &'a RdgDiagnosticsRecorder,
    phase: RdgDetailPhase,
    started_at: Instant,
}

#[cfg(feature = "rdg_bench_support")]
impl Drop for RdgDetailTimer<'_> {
    fn drop(&mut self) {
        self.recorder.record_detail_elapsed(
            self.phase,
            duration_ns(Instant::now().saturating_duration_since(self.started_at)),
        );
    }
}

fn debug_validate_snapshot(snapshot: &RdgFrameDiagnostics) {
    debug_assert!(snapshot.folded_passes <= snapshot.declared_passes);
    debug_assert!(snapshot.executed_passes <= snapshot.prepared_passes);
    if let Some(live) = snapshot.live_passes {
        debug_assert!(live <= snapshot.declared_passes);
        debug_assert!(snapshot.folded_passes <= snapshot.declared_passes - live);
        debug_assert!(snapshot.prepared_passes <= live);
        if snapshot.queue_submitted() {
            debug_assert_eq!(snapshot.prepared_passes, live);
            debug_assert_eq!(snapshot.executed_passes, live);
        }
    } else {
        debug_assert_eq!(snapshot.prepared_passes, 0);
        debug_assert_eq!(snapshot.executed_passes, 0);
        debug_assert!(!snapshot.queue_submitted());
    }
}

/// Plain, `Sync` persistent state for public standalone `GraphStorage`.
///
/// Its active attempt is moved into `RenderGraph`, where exclusive graph
/// ownership is already guaranteed. Renderer attempts use the Cell-based
/// recorder above so their outer RAII guard can coexist with the graph's
/// exclusive topology borrow.
#[derive(Debug, Default)]
pub(crate) struct RdgStandaloneDiagnosticsStore {
    next_attempt_id: u64,
    latest: Option<RdgFrameDiagnostics>,
}

impl RdgStandaloneDiagnosticsStore {
    pub(crate) fn begin_attempt(&mut self) -> RdgStandaloneAttempt {
        self.next_attempt_id = self
            .next_attempt_id
            .checked_add(1)
            .expect("standalone RDG attempt serial overflowed");
        RdgStandaloneAttempt {
            snapshot: RdgFrameDiagnostics::new(
                self.next_attempt_id,
                None,
                RdgFramePhase::BuildGraph,
            ),
            requested_outcome: None,
            #[cfg(feature = "rdg_bench_support")]
            phase_started_at: Instant::now(),
        }
    }

    pub(crate) fn publish(&mut self, attempt: RdgStandaloneAttempt) {
        self.latest = Some(attempt.finish(std::thread::panicking()));
    }

    #[must_use]
    pub(crate) const fn latest(&self) -> Option<RdgFrameDiagnostics> {
        self.latest
    }
}

#[derive(Debug)]
pub(crate) struct RdgStandaloneAttempt {
    snapshot: RdgFrameDiagnostics,
    requested_outcome: Option<RdgAttemptOutcome>,
    #[cfg(feature = "rdg_bench_support")]
    phase_started_at: Instant,
}

impl RdgStandaloneAttempt {
    #[must_use]
    pub(crate) const fn current(&self) -> RdgFrameDiagnostics {
        self.snapshot
    }

    pub(crate) fn enter_phase(&mut self, phase: RdgFramePhase) {
        #[cfg(feature = "rdg_bench_support")]
        self.seal_phase_interval(Instant::now());
        self.snapshot.phase = phase;
    }

    pub(crate) fn record_declared_pass(&mut self) {
        self.snapshot.declared_passes = self.snapshot.declared_passes.saturating_add(1);
    }

    pub(crate) fn record_folded_pass(&mut self) {
        self.snapshot.folded_passes = self.snapshot.folded_passes.saturating_add(1);
        debug_assert!(self.snapshot.folded_passes <= self.snapshot.declared_passes);
    }

    pub(crate) fn record_compiled_live_passes(&mut self, live_passes: usize) {
        debug_assert!(self.snapshot.live_passes.is_none());
        debug_assert!(live_passes <= self.snapshot.declared_passes);
        debug_assert!(self.snapshot.folded_passes <= self.snapshot.declared_passes - live_passes);
        self.snapshot.live_passes = Some(live_passes);
    }

    pub(crate) fn mark_completed_without_submission(&mut self) {
        #[cfg(feature = "rdg_bench_support")]
        self.enter_phase(RdgFramePhase::Complete);
        #[cfg(not(feature = "rdg_bench_support"))]
        {
            self.snapshot.phase = RdgFramePhase::Complete;
        }
        self.requested_outcome = Some(RdgAttemptOutcome::CompletedWithoutSubmission);
    }

    fn finish(mut self, panicking: bool) -> RdgFrameDiagnostics {
        #[cfg(feature = "rdg_bench_support")]
        self.seal_phase_interval(Instant::now());
        self.snapshot.outcome = self
            .requested_outcome
            .unwrap_or(RdgAttemptOutcome::Abandoned);
        if panicking {
            self.snapshot.flags |= FLAG_PANICKED;
        } else {
            debug_validate_snapshot(&self.snapshot);
        }
        self.snapshot
    }

    #[cfg(feature = "rdg_bench_support")]
    fn seal_phase_interval(&mut self, now: Instant) {
        let elapsed = duration_ns(now.saturating_duration_since(self.phase_started_at));
        let slot = &mut self.snapshot.phase_elapsed_ns[self.snapshot.phase.timing_index()];
        *slot = slot.saturating_add(elapsed);
        self.phase_started_at = now;
    }
}

#[cfg(feature = "rdg_bench_support")]
fn duration_ns(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Covers the pre-Composer Renderer work and hands the active attempt off to
/// `FrameComposer` only after extraction and maintenance complete.
pub(crate) struct RdgRendererAttemptGuard<'a> {
    recorder: &'a RdgDiagnosticsRecorder,
    attempt_id: u64,
    handed_off: bool,
}

impl<'a> RdgRendererAttemptGuard<'a> {
    pub(crate) fn new(recorder: &'a RdgDiagnosticsRecorder) -> Self {
        Self {
            recorder,
            attempt_id: recorder
                .current()
                .expect("Renderer attempt guard created without an active attempt")
                .attempt_id(),
            handed_off: false,
        }
    }

    #[inline]
    pub(crate) fn enter_phase(&self, phase: RdgFramePhase) {
        self.recorder.enter_phase(phase);
    }

    pub(crate) fn hand_off(mut self) {
        self.handed_off = true;
    }
}

impl Drop for RdgRendererAttemptGuard<'_> {
    fn drop(&mut self) {
        if !self.handed_off {
            self.recorder.finish_renderer_attempt(self.attempt_id);
        }
    }
}

/// Terminal guard stored as a field of `FrameComposer`.
///
/// This is intentionally field-level rather than a custom `FrameComposer`
/// destructor: RenderGraph borrows topology storage for the frame lifetime,
/// while this guard touches only the separate diagnostics recorder.
pub(crate) struct RdgComposerAttemptGuard<'a> {
    recorder: &'a RdgDiagnosticsRecorder,
    attempt_id: u64,
}

impl<'a> RdgComposerAttemptGuard<'a> {
    pub(crate) fn new(recorder: &'a RdgDiagnosticsRecorder) -> Self {
        Self {
            recorder,
            attempt_id: recorder
                .current()
                .expect("Composer attempt guard created without an active attempt")
                .attempt_id(),
        }
    }
}

impl Drop for RdgComposerAttemptGuard<'_> {
    fn drop(&mut self) {
        self.recorder.finish_renderer_attempt(self.attempt_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_submission_diagnostics_distinguish_selection_dpe_and_hybrid_liveness() {
        let default_family = RdgSceneSubmissionFamilySnapshot::default();
        assert_eq!(
            default_family.selected_backend(),
            RdgSceneSubmissionBackend::Inactive
        );
        assert_eq!(
            default_family.selection_reason(),
            RdgSceneSubmissionSelectionReason::NotEvaluated
        );
        assert_eq!(
            default_family.live_backend(),
            RdgSceneSubmissionBackend::Inactive
        );
        assert_eq!(default_family.live_cpu_submissions(), 0);
        assert_eq!(default_family.live_gpu_submissions(), 0);

        let recorder = RdgDiagnosticsRecorder::default();
        recorder.start_renderer_attempt(50);

        recorder.record_scene_submission_selection(
            RdgSceneSubmissionFamily::MainPrepass,
            RdgSceneSubmissionBackend::Hybrid,
            RdgSceneSubmissionSelectionReason::AutoCostModel,
        );
        recorder.record_live_scene_submissions(
            RdgSceneSubmissionFamily::MainPrepass,
            RdgSceneSubmissionBackend::CpuDirect,
            2,
        );
        recorder.record_live_scene_submissions(
            RdgSceneSubmissionFamily::MainPrepass,
            RdgSceneSubmissionBackend::GpuDriven,
            3,
        );

        // Selection and liveness are intentionally independent: graph DPE may
        // remove every semantic submission after GPU ownership was selected.
        recorder.record_scene_submission_selection(
            RdgSceneSubmissionFamily::Shadow2D,
            RdgSceneSubmissionBackend::GpuDriven,
            RdgSceneSubmissionSelectionReason::AutoHysteresis,
        );

        recorder.record_scene_submission_selection(
            RdgSceneSubmissionFamily::ShadowCube,
            RdgSceneSubmissionBackend::CpuDirect,
            RdgSceneSubmissionSelectionReason::ModeOff,
        );
        recorder.record_live_scene_submissions(
            RdgSceneSubmissionFamily::ShadowCube,
            RdgSceneSubmissionBackend::CpuDirect,
            4,
        );

        recorder.record_compiled_live_passes(0);
        recorder.mark_queue_submitted();
        recorder.mark_completed();
        recorder.finish_renderer_attempt(1);

        let diagnostics = recorder.latest().expect("terminal snapshot missing");
        let main = diagnostics.scene_submission_family(RdgSceneSubmissionFamily::MainPrepass);
        assert_eq!(main.selected_backend(), RdgSceneSubmissionBackend::Hybrid);
        assert_eq!(
            main.selection_reason(),
            RdgSceneSubmissionSelectionReason::AutoCostModel
        );
        assert_eq!(main.live_cpu_submissions(), 2);
        assert_eq!(main.live_gpu_submissions(), 3);
        assert_eq!(main.live_backend(), RdgSceneSubmissionBackend::Hybrid);

        let shadow_2d = diagnostics.scene_submission_family(RdgSceneSubmissionFamily::Shadow2D);
        assert_eq!(
            shadow_2d.selected_backend(),
            RdgSceneSubmissionBackend::GpuDriven
        );
        assert_eq!(
            shadow_2d.live_backend(),
            RdgSceneSubmissionBackend::Inactive
        );

        let shadow_cube = diagnostics.scene_submission_family(RdgSceneSubmissionFamily::ShadowCube);
        assert_eq!(
            shadow_cube.selected_backend(),
            RdgSceneSubmissionBackend::CpuDirect
        );
        assert_eq!(
            shadow_cube.live_backend(),
            RdgSceneSubmissionBackend::CpuDirect
        );
        assert_eq!(shadow_cube.live_cpu_submissions(), 4);
    }

    #[test]
    fn scene_submission_diagnostics_reset_and_saturate_without_allocation() {
        let recorder = RdgDiagnosticsRecorder::default();
        recorder.start_renderer_attempt(60);
        recorder.record_scene_submission_selection(
            RdgSceneSubmissionFamily::MainPrepass,
            RdgSceneSubmissionBackend::GpuDriven,
            RdgSceneSubmissionSelectionReason::Required,
        );
        recorder.record_live_scene_submissions(
            RdgSceneSubmissionFamily::MainPrepass,
            RdgSceneSubmissionBackend::GpuDriven,
            u32::MAX,
        );
        recorder.record_live_scene_submissions(
            RdgSceneSubmissionFamily::MainPrepass,
            RdgSceneSubmissionBackend::GpuDriven,
            1,
        );
        let policy = RdgGpuDrivenPolicySnapshot {
            extracted_item_count: 10_000,
            supported_opaque_count: 9_000,
            retained_candidate_count: 8_000,
            retained_bin_count: 64,
            dirty_candidate_count: 80,
            shadow_2d_view_count: 4,
            shadow_cube_view_count: 6,
            instance_slot_span: 16_000,
            batch_slot_span: 128,
            minimum_cohort_members: 4,
            indexed_primitive_count: 1_000_000,
            exact_evaluated: true,
        };
        recorder.record_gpu_driven_policy(policy);
        let render_world = RdgRenderWorldSnapshot {
            scene_id: Some(7),
            revision: 12,
            mutation_count: 2,
            hierarchy_dirty_roots_examined: 2,
            hierarchy_dirty_parent_links_examined: 1,
            hierarchy_nodes_visited: 3,
            hierarchy_nodes_updated: 2,
            render_item_count: 10_000,
            render_items_reused: 9_998,
            render_items_examined: 2,
            render_items_updated: 2,
            gpu_members_reused: 8_000,
            gpu_members_examined: 2,
            gpu_members_updated: 2,
            retained_bytes: 4 * 1024 * 1024,
            ..RdgRenderWorldSnapshot::default()
        };
        recorder.record_render_world(render_world);
        recorder.update_render_world(|snapshot| {
            snapshot.shadow_casters_examined = 1;
            snapshot.shadow_aggregate_rebuilt = true;
            snapshot.rebuild_reason = Some(RdgRenderWorldRebuildReason::AssetEpochChanged);
        });
        let active = recorder.current().expect("active snapshot missing");
        assert_eq!(
            active
                .scene_submission_family(RdgSceneSubmissionFamily::MainPrepass)
                .live_gpu_submissions(),
            u32::MAX
        );
        assert_eq!(active.gpu_driven_policy(), policy);
        assert_eq!(active.render_world().scene_id(), Some(7));
        assert_eq!(active.render_world().hierarchy_dirty_roots_examined(), 2);
        assert_eq!(
            active
                .render_world()
                .hierarchy_dirty_parent_links_examined(),
            1
        );
        assert_eq!(active.render_world().render_items_examined(), 2);
        assert_eq!(active.render_world().gpu_members_updated(), 2);
        assert_eq!(active.render_world().shadow_casters_examined(), 1);
        assert!(active.render_world().shadow_aggregate_rebuilt());
        assert_eq!(
            active.render_world().rebuild_reason(),
            Some(RdgRenderWorldRebuildReason::AssetEpochChanged)
        );

        recorder.start_renderer_attempt(61);
        assert_eq!(
            recorder
                .current()
                .expect("replacement snapshot missing")
                .gpu_driven_policy(),
            RdgGpuDrivenPolicySnapshot::default()
        );
        assert_eq!(
            recorder
                .current()
                .expect("replacement snapshot missing")
                .render_world(),
            RdgRenderWorldSnapshot::default()
        );
        for family in RdgSceneSubmissionFamily::ALL {
            assert_eq!(
                recorder
                    .current()
                    .expect("replacement snapshot missing")
                    .scene_submission_family(family),
                RdgSceneSubmissionFamilySnapshot::default()
            );
        }
    }

    #[test]
    fn terminal_outcome_matrix_preserves_compile_and_submission_boundaries() {
        let recorder = RdgDiagnosticsRecorder::default();

        // A graph abandoned before compilation must not be confused with a
        // successfully compiled graph whose complete topology was culled.
        recorder.start_standalone_attempt();
        recorder.record_declared_pass();
        recorder.finish_standalone_attempt();
        let uncompiled = recorder.latest().unwrap();
        assert_eq!(uncompiled.outcome(), RdgAttemptOutcome::Abandoned);
        assert_eq!(uncompiled.live_passes(), None);

        recorder.start_standalone_attempt();
        recorder.record_declared_pass();
        recorder.record_compiled_live_passes(0);
        recorder.mark_queue_submitted();
        recorder.mark_completed();
        recorder.finish_standalone_attempt();

        let all_dead = recorder.latest().unwrap();
        assert_eq!(all_dead.outcome(), RdgAttemptOutcome::Submitted);
        assert_eq!(all_dead.live_passes(), Some(0));
        assert_eq!(all_dead.prepared_passes(), 0);
        assert_eq!(all_dead.executed_passes(), 0);

        // Submission is already externally visible even when publication is
        // abandoned later in the attempt.
        recorder.start_standalone_attempt();
        recorder.record_declared_pass();
        recorder.record_compiled_live_passes(1);
        recorder.record_prepared_pass();
        recorder.record_executed_pass();
        recorder.mark_queue_submitted();
        recorder.enter_phase(RdgFramePhase::Publish);
        recorder.finish_standalone_attempt();

        let post_submit = recorder.latest().unwrap();
        assert_eq!(
            post_submit.outcome(),
            RdgAttemptOutcome::SubmittedThenAbandoned
        );
        assert!(post_submit.queue_submitted());
        assert_eq!(post_submit.phase(), RdgFramePhase::Publish);

        // A failed post-submit publication remains typed even after the
        // independent frame work reaches completion.
        recorder.start_standalone_attempt();
        recorder.record_compiled_live_passes(0);
        recorder.mark_queue_submitted();
        recorder.enter_phase(RdgFramePhase::Publish);
        recorder.mark_post_submit_publication_failed(
            RdgPostSubmitPublicationFailure::PersistentBufferStateDrift,
        );
        recorder.mark_completed();
        recorder.finish_standalone_attempt();

        let publication_failure = recorder.latest().unwrap();
        assert_eq!(
            publication_failure.outcome(),
            RdgAttemptOutcome::PostSubmitPublicationFailed(
                RdgPostSubmitPublicationFailure::PersistentBufferStateDrift
            )
        );
        assert!(publication_failure.queue_submitted());
        assert_eq!(publication_failure.phase(), RdgFramePhase::Complete);

        // Materialization failure is a clean pre-prepare terminal state.
        recorder.start_renderer_attempt(3);
        let guard = RdgComposerAttemptGuard::new(&recorder);
        recorder.enter_phase(RdgFramePhase::MaterializeResources);
        recorder.record_declared_pass();
        recorder.record_compiled_live_passes(1);
        recorder.mark_resource_materialization_failed(
            RdgResourceMaterializationFailure::SceneBufferSnapshotDrift,
        );
        drop(guard);

        let snapshot = recorder.latest().expect("terminal snapshot missing");
        assert_eq!(
            snapshot.outcome(),
            RdgAttemptOutcome::ResourceMaterializationFailed(
                RdgResourceMaterializationFailure::SceneBufferSnapshotDrift
            )
        );
        assert_eq!(snapshot.phase(), RdgFramePhase::MaterializeResources);
        assert_eq!(snapshot.prepared_passes(), 0);
        assert_eq!(snapshot.executed_passes(), 0);
        assert!(!snapshot.queue_submitted());
        assert!(!snapshot.presented());
        assert!(!snapshot.panicked());

        // Encoding may fail after prepare/execute bookkeeping but before the
        // command buffer becomes externally visible to the queue.
        recorder.start_renderer_attempt(4);
        let guard = RdgComposerAttemptGuard::new(&recorder);
        recorder.record_declared_pass();
        recorder.record_compiled_live_passes(1);
        recorder.record_prepared_pass();
        recorder.record_executed_pass();
        recorder.enter_phase(RdgFramePhase::Submit);
        recorder.mark_command_encoding_failed(
            RdgCommandEncodingFailure::PersistentBufferWriteBatchIncomplete,
        );
        drop(guard);

        let encoding_failure = recorder.latest().expect("terminal snapshot missing");
        assert_eq!(
            encoding_failure.outcome(),
            RdgAttemptOutcome::CommandEncodingFailed(
                RdgCommandEncodingFailure::PersistentBufferWriteBatchIncomplete
            )
        );
        assert_eq!(encoding_failure.phase(), RdgFramePhase::Submit);
        assert_eq!(encoding_failure.prepared_passes(), 1);
        assert_eq!(encoding_failure.executed_passes(), 1);
        assert!(!encoding_failure.queue_submitted());
        assert!(!encoding_failure.presented());
    }

    #[test]
    fn guards_publish_or_handoff_once_and_stale_guards_are_inert() {
        let recorder = RdgDiagnosticsRecorder::default();

        // Starting a newer attempt first publishes its unfinished predecessor.
        recorder.start_renderer_attempt(41);
        let first_id = recorder.current().unwrap().attempt_id();
        recorder.start_renderer_attempt(42);

        let abandoned = recorder.latest().unwrap();
        assert_eq!(abandoned.attempt_id(), first_id);
        assert_eq!(abandoned.renderer_frame(), Some(41));
        assert_eq!(abandoned.outcome(), RdgAttemptOutcome::Abandoned);
        assert_eq!(recorder.current().unwrap().attempt_id(), first_id + 1);

        // A pre-composer guard owns abandonment until it explicitly hands off.
        recorder.start_renderer_attempt(7);
        {
            let guard = RdgRendererAttemptGuard::new(&recorder);
            guard.enter_phase(RdgFramePhase::Extract);
        }
        assert_eq!(
            recorder.latest().unwrap().outcome(),
            RdgAttemptOutcome::Abandoned
        );

        recorder.start_renderer_attempt(8);
        RdgRendererAttemptGuard::new(&recorder).hand_off();
        assert!(recorder.current().is_some());
        assert_eq!(recorder.latest().unwrap().renderer_frame(), Some(7));

        // Once a newer attempt supersedes the handed-off attempt, its stale
        // composer guard cannot terminate the new owner.
        let stale = RdgComposerAttemptGuard::new(&recorder);
        recorder.start_renderer_attempt(9);
        let current_id = recorder.current().unwrap().attempt_id();

        drop(stale);

        let current = recorder.current().expect("newer attempt was terminated");
        assert_eq!(current.attempt_id(), current_id);
        assert_eq!(current.renderer_frame(), Some(9));
        drop(RdgComposerAttemptGuard::new(&recorder));
    }

    #[test]
    fn irreversible_external_facts_survive_diagnostic_panics() {
        let recorder = RdgDiagnosticsRecorder::default();

        let submission_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recorder.start_renderer_attempt(12);
            let _guard = RdgComposerAttemptGuard::new(&recorder);
            // Deliberately omit compiled-live state. The recorder must set the
            // irreversible submission flag before rejecting this drift.
            recorder.mark_queue_submitted();
        }));
        assert!(submission_result.is_err());

        let submitted = recorder
            .latest()
            .expect("submission panic snapshot missing");
        assert_eq!(
            submitted.outcome(),
            RdgAttemptOutcome::SubmittedThenAbandoned
        );
        assert!(submitted.queue_submitted());
        assert!(submitted.panicked());
    }

    #[test]
    fn repeated_target_failures_request_only_one_log_report() {
        let recorder = RdgDiagnosticsRecorder::default();

        let reasons = [
            (RdgSurfaceAcquireFailure::Timeout, true),
            (RdgSurfaceAcquireFailure::Timeout, false),
            (RdgSurfaceAcquireFailure::Lost, true),
            (RdgSurfaceAcquireFailure::Lost, false),
            (RdgSurfaceAcquireFailure::Validation, true),
            (RdgSurfaceAcquireFailure::Validation, false),
        ];
        for (index, (reason, should_report)) in reasons.into_iter().enumerate() {
            recorder.start_renderer_attempt(20 + index as u64);
            let guard = RdgComposerAttemptGuard::new(&recorder);
            assert_eq!(
                recorder.mark_surface_acquisition_failed(reason),
                should_report
            );
            drop(guard);
        }

        recorder.start_renderer_attempt(26);
        let third = RdgComposerAttemptGuard::new(&recorder);
        assert!(recorder.mark_no_render_target());
        drop(third);

        recorder.start_renderer_attempt(27);
        let fourth = RdgComposerAttemptGuard::new(&recorder);
        assert!(!recorder.mark_no_render_target());
        drop(fourth);
    }

    #[cfg(feature = "rdg_bench_support")]
    #[test]
    fn phase_timing_is_sealed_by_renderer_and_standalone_raii() {
        use std::time::Duration;

        let recorder = RdgDiagnosticsRecorder::default();
        recorder.start_renderer_attempt(31);
        let renderer_guard = RdgRendererAttemptGuard::new(&recorder);
        std::thread::sleep(Duration::from_millis(1));
        renderer_guard.enter_phase(RdgFramePhase::Extract);
        let detail_timer = recorder.detail_timer(RdgDetailPhase::SourceGather);
        std::thread::sleep(Duration::from_millis(1));
        drop(detail_timer);
        drop(renderer_guard);

        let renderer_snapshot = recorder.latest().expect("Renderer snapshot missing");
        assert!(renderer_snapshot.phase_elapsed_ns(RdgFramePhase::Begin) > 0);
        assert!(renderer_snapshot.phase_elapsed_ns(RdgFramePhase::Extract) > 0);
        assert_eq!(
            renderer_snapshot.phase_elapsed_ns(RdgFramePhase::Maintenance),
            0
        );
        assert!(renderer_snapshot.detail_elapsed_ns(RdgDetailPhase::SourceGather) > 0);
        assert_eq!(
            renderer_snapshot.detail_elapsed_ns(RdgDetailPhase::PipelineFinalize),
            0
        );

        let mut standalone = RdgStandaloneDiagnosticsStore::default();
        let mut attempt = standalone.begin_attempt();
        std::thread::sleep(Duration::from_millis(1));
        attempt.enter_phase(RdgFramePhase::CompileGraph);
        std::thread::sleep(Duration::from_millis(1));
        standalone.publish(attempt);

        let standalone_snapshot = standalone.latest().expect("standalone snapshot missing");
        assert!(standalone_snapshot.phase_elapsed_ns(RdgFramePhase::BuildGraph) > 0);
        assert!(standalone_snapshot.phase_elapsed_ns(RdgFramePhase::CompileGraph) > 0);
        assert_eq!(
            standalone_snapshot.phase_elapsed_ns(RdgFramePhase::RecordCommands),
            0
        );
    }
}
