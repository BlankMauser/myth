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
    /// A DPE-live logical resource could not be materialized from the exact
    /// CPU/GPU state captured for this graph attempt. No commands were
    /// prepared, submitted, or presented.
    ResourceMaterializationFailed(RdgResourceMaterializationFailure),
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
    TransientResources,
    SceneBuffers,
    GlobalBuffers,
    PersistentTextures,
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
    pub const COUNT: usize = 17;

    pub const ALL: [Self; Self::COUNT] = [
        Self::TransientResources,
        Self::SceneBuffers,
        Self::GlobalBuffers,
        Self::PersistentTextures,
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
            Self::TransientResources => 0,
            Self::SceneBuffers => 1,
            Self::GlobalBuffers => 2,
            Self::PersistentTextures => 3,
            Self::AssetTexture => 4,
            Self::GlobalBindGroup => 5,
            Self::LiveSourceUnion => 6,
            Self::MaterialPlan => 7,
            Self::Geometry => 8,
            Self::SourceGather => 9,
            Self::SourcePlan => 10,
            Self::SourceMaterialize => 11,
            Self::SourceFinalize => 12,
            Self::PipelineGather => 13,
            Self::PipelineCoordinate => 14,
            Self::PipelineFeatureFinalize => 15,
            Self::PipelineFinalize => 16,
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

    /// Terminates an attempt cleanly before prepare when an exact post-DPE
    /// resource recipe can no longer be satisfied.
    pub(crate) fn mark_resource_materialization_failed(
        &self,
        reason: RdgResourceMaterializationFailure,
    ) {
        self.request_outcome(RdgAttemptOutcome::ResourceMaterializationFailed(reason));
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
    fn uncompiled_and_all_dead_attempts_are_distinct() {
        let recorder = RdgDiagnosticsRecorder::default();
        recorder.start_standalone_attempt();
        recorder.record_declared_pass();
        recorder.finish_standalone_attempt();
        assert_eq!(recorder.latest().unwrap().live_passes(), None);

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
    }

    #[test]
    fn post_submit_abandonment_preserves_submission_truth() {
        let recorder = RdgDiagnosticsRecorder::default();
        recorder.start_standalone_attempt();
        recorder.record_declared_pass();
        recorder.record_compiled_live_passes(1);
        recorder.record_prepared_pass();
        recorder.record_executed_pass();
        recorder.mark_queue_submitted();
        recorder.enter_phase(RdgFramePhase::Publish);
        recorder.finish_standalone_attempt();

        let snapshot = recorder.latest().unwrap();
        assert_eq!(
            snapshot.outcome(),
            RdgAttemptOutcome::SubmittedThenAbandoned
        );
        assert!(snapshot.queue_submitted());
        assert_eq!(snapshot.phase(), RdgFramePhase::Publish);
    }

    #[test]
    fn resource_materialization_failure_is_terminal_without_submission() {
        let recorder = RdgDiagnosticsRecorder::default();
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
    }

    #[test]
    fn a_new_attempt_publishes_an_unfinished_predecessor_once() {
        let recorder = RdgDiagnosticsRecorder::default();
        recorder.start_renderer_attempt(41);
        let first_id = recorder.current().unwrap().attempt_id();
        recorder.start_renderer_attempt(42);

        let abandoned = recorder.latest().unwrap();
        assert_eq!(abandoned.attempt_id(), first_id);
        assert_eq!(abandoned.renderer_frame(), Some(41));
        assert_eq!(abandoned.outcome(), RdgAttemptOutcome::Abandoned);
        assert_eq!(recorder.current().unwrap().attempt_id(), first_id + 1);
    }

    #[test]
    fn pre_composer_guard_publishes_abandonment_unless_handed_off() {
        let recorder = RdgDiagnosticsRecorder::default();
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
    }

    #[test]
    fn stale_guard_cannot_finish_a_newer_attempt() {
        let recorder = RdgDiagnosticsRecorder::default();
        recorder.start_renderer_attempt(10);
        let stale = RdgComposerAttemptGuard::new(&recorder);
        recorder.start_renderer_attempt(11);
        let current_id = recorder.current().unwrap().attempt_id();

        drop(stale);

        let current = recorder.current().expect("newer attempt was terminated");
        assert_eq!(current.attempt_id(), current_id);
        assert_eq!(current.renderer_frame(), Some(11));
        RdgComposerAttemptGuard::new(&recorder);
    }

    #[test]
    fn submission_truth_survives_a_diagnostic_panic() {
        let recorder = RdgDiagnosticsRecorder::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recorder.start_renderer_attempt(12);
            let _guard = RdgComposerAttemptGuard::new(&recorder);
            // Deliberately omit compiled-live state. The recorder must set the
            // irreversible submission flag before rejecting this drift.
            recorder.mark_queue_submitted();
        }));
        assert!(result.is_err());

        let snapshot = recorder.latest().expect("panic snapshot missing");
        assert_eq!(
            snapshot.outcome(),
            RdgAttemptOutcome::SubmittedThenAbandoned
        );
        assert!(snapshot.queue_submitted());
        assert!(snapshot.panicked());
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
