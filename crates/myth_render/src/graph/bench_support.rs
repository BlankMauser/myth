//! Feature-gated harnesses for exercising crate-private RDG hot paths.
//!
//! This module is present in library tests and explicit benchmark-support
//! builds, but absent from default shipping builds. Its narrow semantic API
//! avoids exposing backend command-plan internals or timing an approximate public topology.

use crate::graph::core::scene_submission::{SceneSubmissionKey, SceneSubmissionPlanId};
use crate::graph::core::types::ErasedResourceNodeId;
use crate::graph::core::{
    BufferDesc, ExecuteContext, FrameArena, GraphStorage, PassBuilder, PassNode,
    RdgDiagnosticsRecorder, RdgFramePhase, RenderGraph, TextureDesc, TextureNodeId,
};

const CUBE_LIGHT_ID_BASE: u64 = 1 << 32;

/// Benchmark-only diagnostics route for same-binary A/B measurements.
///
/// Both modes execute identical RDG topology code. `Disabled` skips only the
/// attempt lifecycle and counter updates that are absent from the control arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdgBenchDiagnosticsMode {
    Enabled,
    Disabled,
}

impl RdgBenchDiagnosticsMode {
    /// Stable label used by Criterion and the allocation probe.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
        }
    }

    const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// O(1) correctness token for the synthetic Renderer-recorder lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RdgRecorderLifecycleToken {
    pub declared_passes: usize,
    pub live_passes: usize,
    pub prepared_passes: usize,
    pub executed_passes: usize,
    pub queue_submitted: bool,
}

/// Same-binary control harness for per-pass diagnostics recorder updates.
///
/// Both modes traverse the same `black_box`-guarded loop skeleton. `Enabled`
/// additionally calls the real Renderer-owned Cell recorder, isolating its
/// incremental cost without involving GPU work or graph topology setup.
pub struct RdgRecorderLifecycleBenchHarness {
    diagnostics_mode: RdgBenchDiagnosticsMode,
    recorder: RdgDiagnosticsRecorder,
    renderer_frame: u64,
}

impl RdgRecorderLifecycleBenchHarness {
    #[must_use]
    pub fn new(diagnostics_mode: RdgBenchDiagnosticsMode) -> Self {
        Self {
            diagnostics_mode,
            recorder: RdgDiagnosticsRecorder::default(),
            renderer_frame: 0,
        }
    }

    /// Run a complete synthetic Renderer diagnostics lifecycle.
    #[must_use]
    pub fn run_iteration(&mut self, live_passes: usize) -> RdgRecorderLifecycleToken {
        self.renderer_frame = self
            .renderer_frame
            .checked_add(1)
            .expect("diagnostics lifecycle benchmark frame overflowed");
        let enabled = std::hint::black_box(self.diagnostics_mode.is_enabled());

        let attempt_id = if enabled {
            self.recorder.start_renderer_attempt(self.renderer_frame);
            self.recorder.enter_phase(RdgFramePhase::BuildGraph);
            self.recorder
                .current()
                .expect("benchmark recorder attempt missing")
                .attempt_id()
        } else {
            0
        };

        for pass_index in 0..live_passes {
            std::hint::black_box(pass_index);
            if enabled {
                self.recorder.record_declared_pass();
            }
        }
        if enabled {
            self.recorder.enter_phase(RdgFramePhase::CompileGraph);
            self.recorder.record_compiled_live_passes(live_passes);
            self.recorder.enter_phase(RdgFramePhase::PreparePasses);
        }
        for pass_index in 0..live_passes {
            std::hint::black_box(pass_index);
            if enabled {
                self.recorder.record_prepared_pass();
            }
        }
        if enabled {
            self.recorder.enter_phase(RdgFramePhase::RecordCommands);
        }
        for pass_index in 0..live_passes {
            std::hint::black_box(pass_index);
            if enabled {
                self.recorder.record_executed_pass();
            }
        }
        if enabled {
            self.recorder.enter_phase(RdgFramePhase::Submit);
            self.recorder.mark_queue_submitted();
            self.recorder.mark_completed();
            self.recorder.finish_renderer_attempt(attempt_id);
        }

        RdgRecorderLifecycleToken {
            declared_passes: live_passes,
            live_passes,
            prepared_passes: live_passes,
            executed_passes: live_passes,
            queue_submitted: true,
        }
    }
}

/// How logical texture dependencies are distributed across benchmark plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawDemandDependencyPattern {
    /// Every plan references the same texture set.
    Shared,
    /// Every plan references its own texture set.
    Disjoint,
}

impl DrawDemandDependencyPattern {
    /// Stable label used by Criterion and the allocation probe.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Disjoint => "disjoint",
        }
    }
}

/// Whether benchmark texture inputs are already resident or have real RDG producers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawDemandTextureTopology {
    /// Producer-less, non-root inputs matching steady resident lazy imports.
    ResidentInputs,
    /// One real RDG producer pass for every unique logical texture input.
    PerTextureProducer,
}

impl DrawDemandTextureTopology {
    /// Stable label used by Criterion and the allocation probe.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ResidentInputs => "resident",
            Self::PerTextureProducer => "produced",
        }
    }
}

/// Which Shadow branch is retained by the benchmark's external output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawDemandRoot {
    /// No output root; every producer and plan must be culled.
    None,
    /// Retain only the 2D Shadow branch.
    D2,
    /// Retain only the Cube Shadow branch.
    Cube,
    /// Retain both branches.
    Both,
}

/// Exact workload passed through the production SceneSubmission demand path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawDemandBenchSpec {
    /// Total distinct Shadow plans. One quarter is D2 and three quarters Cube.
    pub plan_count: usize,
    /// Number of logical material texture dependencies per plan.
    pub textures_per_plan: usize,
    /// Whether plans share or own disjoint dependency sets.
    pub dependency_pattern: DrawDemandDependencyPattern,
    /// Whether logical texture inputs are resident or produced by real passes.
    pub texture_topology: DrawDemandTextureTopology,
    /// Number of times each plan is demanded by its Shadow pass.
    pub demand_repetitions: usize,
    /// Whether the graph is compiled after construction.
    pub compile_topology: bool,
    /// Output branch used to seed DPE.
    pub root: DrawDemandRoot,
}

/// Stable benchmark projection of one semantic Shadow submission key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DrawDemandPlanKey {
    /// Stable light identity.
    pub light_id: u64,
    /// D2 layer or Cube face index.
    pub layer_index: u32,
}

/// O(1) token returned inside Criterion and allocation-counting scopes.
///
/// Full graph observations are intentionally kept out of timed paths because
/// scanning resources or plans would distort the scaling curve under test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawDemandRunToken {
    /// Pass records registered by the workload.
    pub registered_pass_count: usize,
    /// Resource records registered by the workload.
    pub registered_resource_count: usize,
    /// Unique reads on the D2 consumer.
    pub d2_read_count: usize,
    /// Unique reads on the Cube consumer.
    pub cube_read_count: usize,
    /// Unique demands on the D2 consumer.
    pub d2_demand_count: usize,
    /// Unique demands on the Cube consumer.
    pub cube_demand_count: usize,
    /// Direct producer-pass dependencies on the D2 consumer.
    pub d2_physical_dependency_count: usize,
    /// Direct producer-pass dependencies on the Cube consumer.
    pub cube_physical_dependency_count: usize,
    /// Direct producer-pass dependencies on the optional output root.
    pub root_physical_dependency_count: usize,
    /// Passes retained by compilation, or zero for build-only runs.
    pub compiled_pass_count: usize,
}

/// Exact per-consumer identities used only by untimed golden observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawDemandPassObservation {
    /// Ordered logical resource versions retained by the pass.
    pub logical_reads: Vec<(u32, u32)>,
    /// Ordered canonical Shadow plans retained by the pass.
    pub demanded_plans: Vec<DrawDemandPlanKey>,
    /// Ordered producer-pass indices derived during compilation.
    pub physical_dependencies: Vec<usize>,
}

/// Exact correctness fingerprint constructed outside benchmark timing scopes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawDemandBenchObservation {
    /// Facts for the D2 Shadow consumer.
    pub d2: DrawDemandPassObservation,
    /// Facts for the Cube Shadow consumer.
    pub cube: DrawDemandPassObservation,
    /// Registered pass records before DPE.
    pub registered_pass_count: usize,
    /// Registered logical resources before DPE.
    pub registered_resource_count: usize,
    /// Resource-side consumer entries created during graph build.
    pub resource_consumer_edge_count: usize,
    /// Exact plans selected by compiled live consumers.
    pub live_plans: Vec<DrawDemandPlanKey>,
    /// Passes retained by compilation, or zero for build-only runs.
    pub compiled_pass_count: usize,
    /// Exact producer dependencies of the optional output-root pass.
    pub root_physical_dependencies: Vec<usize>,
    /// Texture producer passes registered by this workload.
    pub texture_producer_pass_count: usize,
    /// Physical dependency edges retained by texture producer passes.
    ///
    /// Per-texture producers are intentionally flat leaves, so this must be zero.
    pub texture_producer_dependency_edge_count: usize,
}

/// Reusable owner for exact scene-submission demand benchmark iterations.
///
/// GraphStorage's flat topology collections, FrameArena, and harness scratch
/// retain their high-water-mark capacity. The allocation probe verifies that
/// adding real producer fan-in does not reintroduce per-record heap churn.
pub struct DrawDemandBenchHarness {
    storage: GraphStorage,
    arena: FrameArena,
    plan_ids: Vec<SceneSubmissionPlanId>,
    dependency_scratch: Vec<TextureNodeId>,
    diagnostics_mode: RdgBenchDiagnosticsMode,
}

impl DrawDemandBenchHarness {
    /// Construct an empty reusable harness.
    #[must_use]
    pub fn new() -> Self {
        Self::new_with_diagnostics(RdgBenchDiagnosticsMode::Enabled)
    }

    /// Construct a harness with an explicit diagnostics A/B route.
    #[must_use]
    pub fn new_with_diagnostics(diagnostics_mode: RdgBenchDiagnosticsMode) -> Self {
        Self {
            storage: GraphStorage::new(),
            arena: FrameArena::new(),
            plan_ids: Vec::new(),
            dependency_scratch: Vec::new(),
            diagnostics_mode,
        }
    }

    /// Run one production-path iteration with only an O(1) timed result.
    #[must_use]
    pub fn run_iteration(&mut self, spec: DrawDemandBenchSpec) -> DrawDemandRunToken {
        self.run_iteration_with(spec, |graph, indices, _| DrawDemandRunToken {
            registered_pass_count: graph.storage.passes.len(),
            registered_resource_count: graph.storage.resources.len(),
            d2_read_count: graph.storage.pass_reads(indices.d2).len(),
            cube_read_count: graph.storage.pass_reads(indices.cube).len(),
            d2_demand_count: graph
                .storage
                .pass_scene_submission_demands(indices.d2)
                .len(),
            cube_demand_count: graph
                .storage
                .pass_scene_submission_demands(indices.cube)
                .len(),
            d2_physical_dependency_count: graph
                .storage
                .pass_physical_dependencies(indices.d2)
                .len(),
            cube_physical_dependency_count: graph
                .storage
                .pass_physical_dependencies(indices.cube)
                .len(),
            root_physical_dependency_count: indices.root.map_or(0, |index| {
                graph.storage.pass_physical_dependencies(index).len()
            }),
            compiled_pass_count: if spec.compile_topology {
                graph.storage.execution_queue.len()
            } else {
                0
            },
        })
    }

    /// Run one iteration and construct exact identities for correctness tests.
    ///
    /// This performs O(resources + plans) observation after the production
    /// operation and must never be called inside Criterion or allocator scopes.
    #[must_use]
    pub fn run_observed_iteration(
        &mut self,
        spec: DrawDemandBenchSpec,
    ) -> DrawDemandBenchObservation {
        self.run_iteration_with(spec, |graph, indices, plan_ids| {
            let d2_plan_count = spec.plan_count / 4;
            DrawDemandBenchObservation {
                d2: observe_pass(
                    graph.storage.pass_reads(indices.d2),
                    graph.storage.pass_physical_dependencies(indices.d2),
                    graph.storage.pass_scene_submission_demands(indices.d2),
                    plan_ids,
                    d2_plan_count,
                ),
                cube: observe_pass(
                    graph.storage.pass_reads(indices.cube),
                    graph.storage.pass_physical_dependencies(indices.cube),
                    graph.storage.pass_scene_submission_demands(indices.cube),
                    plan_ids,
                    d2_plan_count,
                ),
                registered_pass_count: graph.storage.passes.len(),
                registered_resource_count: graph.storage.resources.len(),
                resource_consumer_edge_count: graph
                    .storage
                    .resources
                    .iter()
                    .map(crate::graph::core::ResourceRecord::consumer_edge_count)
                    .sum(),
                live_plans: if spec.compile_topology {
                    plan_ids
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &id)| {
                            graph
                                .storage
                                .scene_submission_is_live(id)
                                .then(|| project_plan_key(shadow_plan_key(index, d2_plan_count)))
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                compiled_pass_count: if spec.compile_topology {
                    graph.storage.execution_queue.len()
                } else {
                    0
                },
                root_physical_dependencies: indices.root.map_or_else(Vec::new, |index| {
                    graph.storage.pass_physical_dependencies(index).to_vec()
                }),
                texture_producer_pass_count: indices.texture_producer_end,
                texture_producer_dependency_edge_count: (0..indices.texture_producer_end)
                    .map(|index| graph.storage.pass_physical_dependencies(index).len())
                    .sum(),
            }
        })
    }

    fn run_iteration_with<R>(
        &mut self,
        spec: DrawDemandBenchSpec,
        observe: impl FnOnce(&RenderGraph<'_>, DrawDemandRunIndices, &[SceneSubmissionPlanId]) -> R,
    ) -> R {
        validate_spec(spec);

        let Self {
            storage,
            arena,
            plan_ids,
            dependency_scratch,
            diagnostics_mode,
        } = self;
        arena.reset();
        plan_ids.clear();
        dependency_scratch.clear();

        let mut graph =
            RenderGraph::new_for_benchmark(storage, arena, diagnostics_mode.is_enabled());
        let texture_desc = dependency_desc();

        if spec.dependency_pattern == DrawDemandDependencyPattern::Shared {
            dependency_scratch.reserve(spec.textures_per_plan);
            for _ in 0..spec.textures_per_plan {
                dependency_scratch.push(register_texture_input(
                    &mut graph,
                    texture_desc,
                    spec.texture_topology,
                ));
            }
        }

        let d2_plan_count = spec.plan_count / 4;
        plan_ids.reserve(spec.plan_count);
        for plan_index in 0..spec.plan_count {
            let key = shadow_plan_key(plan_index, d2_plan_count);

            match spec.dependency_pattern {
                DrawDemandDependencyPattern::Shared => {
                    graph.set_scene_submission_texture_dependencies(
                        key,
                        dependency_scratch.iter().copied(),
                    );
                }
                DrawDemandDependencyPattern::Disjoint => {
                    dependency_scratch.clear();
                    dependency_scratch.reserve(spec.textures_per_plan);
                    for _ in 0..spec.textures_per_plan {
                        dependency_scratch.push(register_texture_input(
                            &mut graph,
                            texture_desc,
                            spec.texture_topology,
                        ));
                    }
                    graph.set_scene_submission_texture_dependencies(
                        key,
                        dependency_scratch.iter().copied(),
                    );
                }
            }
            let plan = graph.reserve_scene_submission(key);
            plan_ids.push(plan);
        }

        let texture_producer_end = graph.storage.passes.len();

        let vp_buffer = graph.add_pass("Shadow_VP_Upload", |builder: &mut PassBuilder<'_, '_>| {
            let buffer = builder.create_buffer(
                "Shadow_VP_Buffer",
                BufferDesc::new(
                    256,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
            );
            (DrawDemandBenchNode, buffer)
        });

        let d2_pass = graph.storage.passes.len();
        let d2_output = graph.add_pass("Shadow_2D_Pass", |builder: &mut PassBuilder<'_, '_>| {
            builder.read_buffer(vp_buffer);
            demand_plans(builder, &plan_ids[..d2_plan_count], spec.demand_repetitions);
            let output = builder.create_texture("Shadow_2D_Output", texture_desc);
            (DrawDemandBenchNode, output)
        });

        let cube_pass = graph.storage.passes.len();
        let cube_output =
            graph.add_pass("Shadow_Cube_Pass", |builder: &mut PassBuilder<'_, '_>| {
                builder.read_buffer(vp_buffer);
                demand_plans(builder, &plan_ids[d2_plan_count..], spec.demand_repetitions);
                let output = builder.create_texture("Shadow_Cube_Output", texture_desc);
                (DrawDemandBenchNode, output)
            });

        let root_pass = add_output_root(&mut graph, spec.root, d2_output, cube_output);
        if spec.compile_topology {
            graph.compile_topology();
        }

        observe(
            &graph,
            DrawDemandRunIndices {
                d2: d2_pass,
                cube: cube_pass,
                root: root_pass,
                texture_producer_end,
            },
            plan_ids,
        )
    }
}

impl Default for DrawDemandBenchHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
struct DrawDemandRunIndices {
    d2: usize,
    cube: usize,
    root: Option<usize>,
    texture_producer_end: usize,
}

fn validate_spec(spec: DrawDemandBenchSpec) {
    assert!(
        spec.plan_count >= 8 && spec.plan_count.is_multiple_of(8),
        "plan count must be at least eight and divisible by eight"
    );
    assert!(
        spec.textures_per_plan > 0,
        "benchmark requires at least one texture per plan"
    );
    assert!(
        spec.demand_repetitions > 0,
        "benchmark requires at least one demand per plan"
    );
}

fn register_texture_input(
    graph: &mut RenderGraph<'_>,
    desc: TextureDesc,
    topology: DrawDemandTextureTopology,
) -> TextureNodeId {
    match topology {
        DrawDemandTextureTopology::ResidentInputs => {
            // Producer-less/non-root matches the steady resident lazy-import topology
            // without invoking Asset-specific lookup and validation.
            graph.register_texture("Shadow_Material_Texture", desc, false)
        }
        DrawDemandTextureTopology::PerTextureProducer => graph.add_pass(
            "Shadow_Material_Texture_Producer",
            |builder: &mut PassBuilder<'_, '_>| {
                let texture = builder.create_texture("Shadow_Material_Texture", desc);
                (DrawDemandBenchNode, texture)
            },
        ),
    }
}

fn shadow_plan_key(plan_index: usize, d2_plan_count: usize) -> SceneSubmissionKey {
    if plan_index < d2_plan_count {
        SceneSubmissionKey::Shadow {
            light_id: plan_index as u64,
            layer_index: 0,
        }
    } else {
        let cube_face_index = plan_index - d2_plan_count;
        SceneSubmissionKey::Shadow {
            light_id: CUBE_LIGHT_ID_BASE + (cube_face_index / 6) as u64,
            layer_index: (cube_face_index % 6) as u32,
        }
    }
}

fn demand_plans(
    builder: &mut PassBuilder<'_, '_>,
    plans: &[SceneSubmissionPlanId],
    repetitions: usize,
) {
    for &plan in plans {
        for _ in 0..repetitions {
            builder.demand_scene_submission(plan);
        }
    }
}

fn add_output_root(
    graph: &mut RenderGraph<'_>,
    root: DrawDemandRoot,
    d2_output: TextureNodeId,
    cube_output: TextureNodeId,
) -> Option<usize> {
    if root == DrawDemandRoot::None {
        return None;
    }

    let backbuffer = graph.register_texture("Backbuffer", dependency_desc(), true);
    let pass_index = graph.storage.passes.len();
    graph.add_pass("Shadow_Output_Root", |builder: &mut PassBuilder<'_, '_>| {
        if matches!(root, DrawDemandRoot::D2 | DrawDemandRoot::Both) {
            builder.read_texture(d2_output);
        }
        if matches!(root, DrawDemandRoot::Cube | DrawDemandRoot::Both) {
            builder.read_texture(cube_output);
        }
        builder.write_texture(backbuffer);
        (DrawDemandBenchNode, ())
    });
    Some(pass_index)
}

fn observe_pass(
    reads: &[ErasedResourceNodeId],
    physical_dependencies: &[usize],
    demands: &[SceneSubmissionPlanId],
    plan_ids: &[SceneSubmissionPlanId],
    d2_plan_count: usize,
) -> DrawDemandPassObservation {
    DrawDemandPassObservation {
        logical_reads: reads.iter().map(|id| (id.index(), id.version())).collect(),
        demanded_plans: demands
            .iter()
            .map(|&id| resolve_plan_key(plan_ids, d2_plan_count, id))
            .collect(),
        physical_dependencies: physical_dependencies.to_vec(),
    }
}

fn resolve_plan_key(
    plan_ids: &[SceneSubmissionPlanId],
    d2_plan_count: usize,
    id: SceneSubmissionPlanId,
) -> DrawDemandPlanKey {
    plan_ids
        .iter()
        .position(|&candidate| candidate == id)
        .map(|index| project_plan_key(shadow_plan_key(index, d2_plan_count)))
        .expect("benchmark pass retained an unknown scene-submission ID")
}

fn project_plan_key(key: SceneSubmissionKey) -> DrawDemandPlanKey {
    let SceneSubmissionKey::Shadow {
        light_id,
        layer_index,
    } = key
    else {
        panic!("Draw demand benchmark observed a non-Shadow plan")
    };
    DrawDemandPlanKey {
        light_id,
        layer_index,
    }
}

struct DrawDemandBenchNode;

impl PassNode<'_> for DrawDemandBenchNode {
    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
}

fn dependency_desc() -> TextureDesc {
    TextureDesc::new_2d(
        1,
        1,
        wgpu::TextureFormat::R8Unorm,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
    )
}

/// Whether the high-fan-out workload is rooted or entirely removed by DPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighFanOutRoot {
    /// No output root; the producer and every consumer are dead.
    None,
    /// One output root reads every consumer output.
    AllConsumers,
}

impl HighFanOutRoot {
    /// Stable label used by the allocation probe.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "all_dead",
            Self::AllConsumers => "live",
        }
    }
}

/// Exact workload for one producer resource with many consumer passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HighFanOutBenchSpec {
    /// Number of passes that directly read the producer texture.
    pub consumer_count: usize,
    /// DPE root mode.
    pub root: HighFanOutRoot,
}

/// O(1) result suitable for allocation-counting scopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HighFanOutRunToken {
    pub registered_pass_count: usize,
    pub registered_resource_count: usize,
    pub logical_consumer_count: usize,
    pub producer_reverse_consumer_count: usize,
    pub root_physical_dependency_count: usize,
    pub compiled_pass_count: usize,
}

/// Untimed exact topology facts for the high-fan-out golden tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HighFanOutBenchObservation {
    pub registered_pass_count: usize,
    pub registered_resource_count: usize,
    pub logical_consumer_order: Vec<usize>,
    pub producer_physical_dependencies: Vec<usize>,
    pub consumer_physical_dependencies: Vec<Vec<usize>>,
    pub producer_reverse_consumers: Vec<usize>,
    pub consumer_reverse_consumers: Vec<Vec<usize>>,
    pub root_physical_dependencies: Vec<usize>,
    pub execution_queue: Vec<usize>,
}

/// Capacity-reusing owner for the real high-fan-out topology.
pub struct HighFanOutBenchHarness {
    storage: GraphStorage,
    arena: FrameArena,
    consumer_outputs: Vec<TextureNodeId>,
}

impl HighFanOutBenchHarness {
    #[must_use]
    pub fn new() -> Self {
        Self {
            storage: GraphStorage::new(),
            arena: FrameArena::new(),
            consumer_outputs: Vec::new(),
        }
    }

    /// Run one compiled topology iteration with an O(1) result.
    #[must_use]
    pub fn run_iteration(&mut self, spec: HighFanOutBenchSpec) -> HighFanOutRunToken {
        self.run_iteration_with(spec, |graph, indices| HighFanOutRunToken {
            registered_pass_count: graph.storage.passes.len(),
            registered_resource_count: graph.storage.resources.len(),
            logical_consumer_count: graph.storage.resources
                [indices.producer_texture.index() as usize]
                .consumer_edge_count(),
            producer_reverse_consumer_count: graph
                .storage
                .compiled_reverse_consumers(indices.producer_pass)
                .len(),
            root_physical_dependency_count: indices.root.map_or(0, |root| {
                graph.storage.pass_physical_dependencies(root).len()
            }),
            compiled_pass_count: graph.storage.execution_queue.len(),
        })
    }

    /// Run one iteration and copy exact identities outside timed scopes.
    #[must_use]
    pub fn run_observed_iteration(
        &mut self,
        spec: HighFanOutBenchSpec,
    ) -> HighFanOutBenchObservation {
        self.run_iteration_with(spec, |graph, indices| HighFanOutBenchObservation {
            registered_pass_count: graph.storage.passes.len(),
            registered_resource_count: graph.storage.resources.len(),
            logical_consumer_order: graph
                .storage
                .resource_consumers(indices.producer_texture.index() as usize)
                .collect(),
            producer_physical_dependencies: graph
                .storage
                .pass_physical_dependencies(indices.producer_pass)
                .to_vec(),
            consumer_physical_dependencies: (indices.consumer_start..indices.consumer_end)
                .map(|consumer| graph.storage.pass_physical_dependencies(consumer).to_vec())
                .collect(),
            producer_reverse_consumers: graph
                .storage
                .compiled_reverse_consumers(indices.producer_pass)
                .to_vec(),
            consumer_reverse_consumers: (indices.consumer_start..indices.consumer_end)
                .map(|consumer| graph.storage.compiled_reverse_consumers(consumer).to_vec())
                .collect(),
            root_physical_dependencies: indices.root.map_or_else(Vec::new, |root| {
                graph.storage.pass_physical_dependencies(root).to_vec()
            }),
            execution_queue: graph.storage.execution_queue.clone(),
        })
    }

    fn run_iteration_with<R>(
        &mut self,
        spec: HighFanOutBenchSpec,
        observe: impl FnOnce(&RenderGraph<'_>, HighFanOutRunIndices) -> R,
    ) -> R {
        assert!(
            spec.consumer_count > 0,
            "high-fan-out benchmark requires at least one consumer"
        );

        self.arena.reset();
        self.consumer_outputs.clear();
        self.consumer_outputs.reserve(spec.consumer_count);

        let mut graph = RenderGraph::new_for_benchmark(&mut self.storage, &self.arena, true);
        let desc = dependency_desc();
        let producer_pass = graph.storage.passes.len();
        let producer_texture = graph.add_pass(
            "HighFanOut_Producer",
            |builder: &mut PassBuilder<'_, '_>| {
                let texture = builder.create_texture("HighFanOut_Source", desc);
                (DrawDemandBenchNode, texture)
            },
        );

        let consumer_start = graph.storage.passes.len();
        for _ in 0..spec.consumer_count {
            let output = graph.add_pass(
                "HighFanOut_Consumer",
                |builder: &mut PassBuilder<'_, '_>| {
                    builder.read_texture(producer_texture);
                    let output = builder.create_texture("HighFanOut_Consumer_Output", desc);
                    (DrawDemandBenchNode, output)
                },
            );
            self.consumer_outputs.push(output);
        }
        let consumer_end = graph.storage.passes.len();

        let root = if spec.root == HighFanOutRoot::AllConsumers {
            let backbuffer = graph.register_texture("HighFanOut_Backbuffer", desc, true);
            let root = graph.storage.passes.len();
            graph.add_pass("HighFanOut_Root", |builder: &mut PassBuilder<'_, '_>| {
                for &output in &self.consumer_outputs {
                    builder.read_texture(output);
                }
                builder.write_texture(backbuffer);
                (DrawDemandBenchNode, ())
            });
            Some(root)
        } else {
            None
        };

        graph.compile_topology();
        observe(
            &graph,
            HighFanOutRunIndices {
                producer_pass,
                producer_texture,
                consumer_start,
                consumer_end,
                root,
            },
        )
    }
}

impl Default for HighFanOutBenchHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
struct HighFanOutRunIndices {
    producer_pass: usize,
    producer_texture: TextureNodeId,
    consumer_start: usize,
    consumer_end: usize,
    root: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::{
        DrawDemandBenchHarness, DrawDemandBenchSpec, DrawDemandDependencyPattern,
        DrawDemandPlanKey, DrawDemandRoot, DrawDemandTextureTopology, HighFanOutBenchHarness,
        HighFanOutBenchSpec, HighFanOutRoot, RdgBenchDiagnosticsMode,
        RdgRecorderLifecycleBenchHarness, project_plan_key, shadow_plan_key,
    };

    fn spec(
        pattern: DrawDemandDependencyPattern,
        repetitions: usize,
        compile_topology: bool,
        root: DrawDemandRoot,
    ) -> DrawDemandBenchSpec {
        spec_with_topology(
            pattern,
            DrawDemandTextureTopology::ResidentInputs,
            repetitions,
            compile_topology,
            root,
        )
    }

    fn spec_with_topology(
        pattern: DrawDemandDependencyPattern,
        texture_topology: DrawDemandTextureTopology,
        repetitions: usize,
        compile_topology: bool,
        root: DrawDemandRoot,
    ) -> DrawDemandBenchSpec {
        DrawDemandBenchSpec {
            plan_count: 32,
            textures_per_plan: 8,
            dependency_pattern: pattern,
            texture_topology,
            demand_repetitions: repetitions,
            compile_topology,
            root,
        }
    }

    fn unique_texture_count(pattern: DrawDemandDependencyPattern) -> usize {
        match pattern {
            DrawDemandDependencyPattern::Shared => 8,
            DrawDemandDependencyPattern::Disjoint => 32 * 8,
        }
    }

    fn branch_texture_count(pattern: DrawDemandDependencyPattern, plan_count: usize) -> usize {
        match pattern {
            DrawDemandDependencyPattern::Shared => 8,
            DrawDemandDependencyPattern::Disjoint => plan_count * 8,
        }
    }

    fn expected_producer_dependencies(
        pattern: DrawDemandDependencyPattern,
    ) -> (Vec<usize>, Vec<usize>) {
        let producer_count = unique_texture_count(pattern);
        let mut d2 = vec![producer_count];
        let mut cube = vec![producer_count];
        match pattern {
            DrawDemandDependencyPattern::Shared => {
                d2.extend(0..8);
                cube.extend(0..8);
            }
            DrawDemandDependencyPattern::Disjoint => {
                d2.extend(0..8 * 8);
                cube.extend(8 * 8..producer_count);
            }
        }
        (d2, cube)
    }

    fn expected_plan_keys(plan_count: usize) -> Vec<DrawDemandPlanKey> {
        let d2_plan_count = plan_count / 4;
        (0..plan_count)
            .map(|index| project_plan_key(shadow_plan_key(index, d2_plan_count)))
            .collect()
    }

    #[test]
    fn shadow_fan_in_build_facts_match_shared_and_disjoint_formulas() {
        let expected_plans = expected_plan_keys(32);
        for pattern in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ] {
            let mut harness = DrawDemandBenchHarness::new();
            let facts =
                harness.run_observed_iteration(spec(pattern, 1, false, DrawDemandRoot::Both));
            let unique_textures = match pattern {
                DrawDemandDependencyPattern::Shared => 8,
                DrawDemandDependencyPattern::Disjoint => 32 * 8,
            };
            let d2_texture_reads = match pattern {
                DrawDemandDependencyPattern::Shared => 8,
                DrawDemandDependencyPattern::Disjoint => 8 * 8,
            };
            let cube_texture_reads = match pattern {
                DrawDemandDependencyPattern::Shared => 8,
                DrawDemandDependencyPattern::Disjoint => 24 * 8,
            };
            let consumer_edges = match pattern {
                DrawDemandDependencyPattern::Shared => 2 * 8 + 4,
                DrawDemandDependencyPattern::Disjoint => 32 * 8 + 4,
            };

            assert_eq!(facts.d2.logical_reads.len(), d2_texture_reads + 1);
            assert_eq!(facts.cube.logical_reads.len(), cube_texture_reads + 1);
            assert_eq!(facts.d2.demanded_plans, expected_plans[..8]);
            assert_eq!(facts.cube.demanded_plans, expected_plans[8..]);
            assert_eq!(facts.registered_pass_count, 4);
            assert_eq!(facts.registered_resource_count, unique_textures + 4);
            assert_eq!(facts.resource_consumer_edge_count, consumer_edges);
            assert!(facts.live_plans.is_empty());
            assert_eq!(facts.compiled_pass_count, 0);
            assert!(facts.d2.physical_dependencies.is_empty());
            assert!(facts.cube.physical_dependencies.is_empty());
        }
    }

    #[test]
    fn duplicate_shadow_demands_are_topology_idempotent() {
        for pattern in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ] {
            for texture_topology in [
                DrawDemandTextureTopology::ResidentInputs,
                DrawDemandTextureTopology::PerTextureProducer,
            ] {
                let mut harness = DrawDemandBenchHarness::new();
                let single = harness.run_observed_iteration(spec_with_topology(
                    pattern,
                    texture_topology,
                    1,
                    true,
                    DrawDemandRoot::Both,
                ));
                let duplicate = harness.run_observed_iteration(spec_with_topology(
                    pattern,
                    texture_topology,
                    2,
                    true,
                    DrawDemandRoot::Both,
                ));
                assert_eq!(single, duplicate);
            }
        }
    }

    #[test]
    fn producer_heavy_build_facts_match_exact_formulas() {
        let expected_plans = expected_plan_keys(32);
        for pattern in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ] {
            let mut harness = DrawDemandBenchHarness::new();
            let facts = harness.run_observed_iteration(spec_with_topology(
                pattern,
                DrawDemandTextureTopology::PerTextureProducer,
                1,
                false,
                DrawDemandRoot::Both,
            ));
            let producer_count = unique_texture_count(pattern);
            let d2_texture_reads = branch_texture_count(pattern, 8);
            let cube_texture_reads = branch_texture_count(pattern, 24);
            let texture_consumer_edges = match pattern {
                DrawDemandDependencyPattern::Shared => 2 * 8,
                DrawDemandDependencyPattern::Disjoint => 32 * 8,
            };

            assert_eq!(facts.d2.logical_reads.len(), d2_texture_reads + 1);
            assert_eq!(facts.cube.logical_reads.len(), cube_texture_reads + 1);
            assert_eq!(facts.d2.demanded_plans, expected_plans[..8]);
            assert_eq!(facts.cube.demanded_plans, expected_plans[8..]);
            assert_eq!(facts.registered_pass_count, producer_count + 4);
            assert_eq!(facts.registered_resource_count, producer_count + 4);
            assert_eq!(
                facts.resource_consumer_edge_count,
                texture_consumer_edges + 4
            );
            assert_eq!(facts.texture_producer_pass_count, producer_count);
            assert_eq!(facts.texture_producer_dependency_edge_count, 0);
            assert!(facts.live_plans.is_empty());
            assert_eq!(facts.compiled_pass_count, 0);
            assert!(facts.d2.physical_dependencies.is_empty());
            assert!(facts.cube.physical_dependencies.is_empty());
            assert!(facts.root_physical_dependencies.is_empty());
        }
    }

    #[test]
    fn producer_heavy_compile_keeps_ordered_flat_dependencies_for_every_root() {
        let expected_plans = expected_plan_keys(32);
        for pattern in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ] {
            let producer_count = unique_texture_count(pattern);
            let (expected_d2_dependencies, expected_cube_dependencies) =
                expected_producer_dependencies(pattern);

            for root in [
                DrawDemandRoot::None,
                DrawDemandRoot::D2,
                DrawDemandRoot::Cube,
                DrawDemandRoot::Both,
            ] {
                let mut harness = DrawDemandBenchHarness::new();
                let facts = harness.run_observed_iteration(spec_with_topology(
                    pattern,
                    DrawDemandTextureTopology::PerTextureProducer,
                    1,
                    true,
                    root,
                ));

                assert_eq!(facts.d2.physical_dependencies, expected_d2_dependencies);
                assert_eq!(facts.cube.physical_dependencies, expected_cube_dependencies);
                assert_eq!(facts.texture_producer_pass_count, producer_count);
                assert_eq!(facts.texture_producer_dependency_edge_count, 0);

                let (expected_live_plans, expected_live_passes, expected_root_dependencies) =
                    match root {
                        DrawDemandRoot::None => (&expected_plans[0..0], 0, vec![]),
                        DrawDemandRoot::D2 => (
                            &expected_plans[..8],
                            3 + branch_texture_count(pattern, 8),
                            vec![producer_count + 1],
                        ),
                        DrawDemandRoot::Cube => (
                            &expected_plans[8..],
                            3 + branch_texture_count(pattern, 24),
                            vec![producer_count + 2],
                        ),
                        DrawDemandRoot::Both => (
                            expected_plans.as_slice(),
                            4 + producer_count,
                            vec![producer_count + 1, producer_count + 2],
                        ),
                    };
                assert_eq!(facts.live_plans, expected_live_plans);
                assert_eq!(facts.compiled_pass_count, expected_live_passes);
                assert_eq!(facts.root_physical_dependencies, expected_root_dependencies);
            }
        }
    }

    #[test]
    fn resident_and_produced_inputs_preserve_logical_draw_demand_facts() {
        for pattern in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ] {
            for root in [
                DrawDemandRoot::None,
                DrawDemandRoot::D2,
                DrawDemandRoot::Cube,
                DrawDemandRoot::Both,
            ] {
                let mut resident_harness = DrawDemandBenchHarness::new();
                let resident = resident_harness.run_observed_iteration(spec_with_topology(
                    pattern,
                    DrawDemandTextureTopology::ResidentInputs,
                    1,
                    true,
                    root,
                ));
                let mut produced_harness = DrawDemandBenchHarness::new();
                let produced = produced_harness.run_observed_iteration(spec_with_topology(
                    pattern,
                    DrawDemandTextureTopology::PerTextureProducer,
                    1,
                    true,
                    root,
                ));

                assert_eq!(resident.d2.logical_reads, produced.d2.logical_reads);
                assert_eq!(resident.cube.logical_reads, produced.cube.logical_reads);
                assert_eq!(resident.d2.demanded_plans, produced.d2.demanded_plans);
                assert_eq!(resident.cube.demanded_plans, produced.cube.demanded_plans);
                assert_eq!(
                    resident.registered_resource_count,
                    produced.registered_resource_count
                );
                assert_eq!(
                    resident.resource_consumer_edge_count,
                    produced.resource_consumer_edge_count
                );
                assert_eq!(resident.live_plans, produced.live_plans);
            }
        }
    }

    #[test]
    fn shadow_fan_in_dpe_projects_exact_branch() {
        let expected_plans = expected_plan_keys(32);
        for pattern in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ] {
            let mut harness = DrawDemandBenchHarness::new();

            let d2 = harness.run_observed_iteration(spec(pattern, 1, true, DrawDemandRoot::D2));
            assert_eq!(d2.live_plans, expected_plans[..8]);
            assert_eq!(d2.compiled_pass_count, 3);
            assert_eq!(d2.d2.physical_dependencies, [0]);
            assert_eq!(d2.root_physical_dependencies, [1]);

            let cube = harness.run_observed_iteration(spec(pattern, 1, true, DrawDemandRoot::Cube));
            assert_eq!(cube.live_plans, expected_plans[8..]);
            assert_eq!(cube.compiled_pass_count, 3);
            assert_eq!(cube.cube.physical_dependencies, [0]);
            assert_eq!(cube.root_physical_dependencies, [2]);

            let both = harness.run_observed_iteration(spec(pattern, 1, true, DrawDemandRoot::Both));
            assert_eq!(both.live_plans, expected_plans);
            assert_eq!(both.compiled_pass_count, 4);
            assert_eq!(both.d2.physical_dependencies, [0]);
            assert_eq!(both.cube.physical_dependencies, [0]);
            assert_eq!(both.root_physical_dependencies, [1, 2]);

            let none = harness.run_observed_iteration(spec(pattern, 1, true, DrawDemandRoot::None));
            assert!(none.live_plans.is_empty());
            assert_eq!(none.compiled_pass_count, 0);
            assert!(none.root_physical_dependencies.is_empty());
        }
    }

    #[test]
    fn diagnostics_ab_routes_preserve_exact_topology() {
        for pattern in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ] {
            for root in [DrawDemandRoot::Both, DrawDemandRoot::None] {
                for texture_topology in [
                    DrawDemandTextureTopology::ResidentInputs,
                    DrawDemandTextureTopology::PerTextureProducer,
                ] {
                    let workload = spec_with_topology(pattern, texture_topology, 1, true, root);
                    let mut enabled = DrawDemandBenchHarness::new_with_diagnostics(
                        RdgBenchDiagnosticsMode::Enabled,
                    );
                    let mut disabled = DrawDemandBenchHarness::new_with_diagnostics(
                        RdgBenchDiagnosticsMode::Disabled,
                    );

                    assert_eq!(
                        enabled.run_observed_iteration(workload),
                        disabled.run_observed_iteration(workload),
                        "diagnostics routing changed the benchmark topology"
                    );
                }
            }
        }
    }

    #[test]
    fn recorder_lifecycle_ab_routes_preserve_work_tokens() {
        for live_passes in [0, 32, 64, 128] {
            let mut enabled =
                RdgRecorderLifecycleBenchHarness::new(RdgBenchDiagnosticsMode::Enabled);
            let mut disabled =
                RdgRecorderLifecycleBenchHarness::new(RdgBenchDiagnosticsMode::Disabled);
            assert_eq!(
                enabled.run_iteration(live_passes),
                disabled.run_iteration(live_passes)
            );

            let snapshot = enabled
                .recorder
                .latest()
                .expect("enabled recorder did not publish a terminal snapshot");
            assert_eq!(snapshot.declared_passes(), live_passes);
            assert_eq!(snapshot.live_passes(), Some(live_passes));
            assert_eq!(snapshot.prepared_passes(), live_passes);
            assert_eq!(snapshot.executed_passes(), live_passes);
            assert!(snapshot.queue_submitted());
        }
    }

    #[test]
    fn high_fan_out_preserves_consumer_order_dpe_and_live_csr() {
        for consumer_count in [4, 8, 16, 64, 128] {
            let mut harness = HighFanOutBenchHarness::new();
            let facts = harness.run_observed_iteration(HighFanOutBenchSpec {
                consumer_count,
                root: HighFanOutRoot::AllConsumers,
            });
            let consumers: Vec<_> = (1..=consumer_count).collect();
            let root = consumer_count + 1;
            let mut expected_execution = vec![0];
            expected_execution.extend(consumers.iter().copied());
            expected_execution.push(root);

            assert_eq!(facts.registered_pass_count, consumer_count + 2);
            assert_eq!(facts.registered_resource_count, consumer_count + 2);
            assert_eq!(facts.logical_consumer_order, consumers);
            assert!(facts.producer_physical_dependencies.is_empty());
            assert_eq!(
                facts.consumer_physical_dependencies,
                vec![vec![0]; consumer_count]
            );
            assert_eq!(facts.producer_reverse_consumers, consumers);
            assert_eq!(
                facts.consumer_reverse_consumers,
                vec![vec![root]; consumer_count]
            );
            assert_eq!(facts.root_physical_dependencies, consumers);
            assert_eq!(facts.execution_queue, expected_execution);
        }
    }

    #[test]
    fn high_fan_out_all_dead_keeps_physical_facts_but_builds_empty_csr() {
        for consumer_count in [4, 8, 16, 64, 128] {
            let mut harness = HighFanOutBenchHarness::new();
            let facts = harness.run_observed_iteration(HighFanOutBenchSpec {
                consumer_count,
                root: HighFanOutRoot::None,
            });
            let consumers: Vec<_> = (1..=consumer_count).collect();

            assert_eq!(facts.registered_pass_count, consumer_count + 1);
            assert_eq!(facts.registered_resource_count, consumer_count + 1);
            assert_eq!(facts.logical_consumer_order, consumers);
            assert!(facts.producer_physical_dependencies.is_empty());
            assert_eq!(
                facts.consumer_physical_dependencies,
                vec![vec![0]; consumer_count]
            );
            assert!(facts.producer_reverse_consumers.is_empty());
            assert_eq!(
                facts.consumer_reverse_consumers,
                vec![Vec::<usize>::new(); consumer_count]
            );
            assert!(facts.root_physical_dependencies.is_empty());
            assert!(facts.execution_queue.is_empty());
        }
    }

    #[test]
    fn high_fan_out_timed_token_matches_exact_live_and_dead_formulas() {
        for consumer_count in [4, 8, 16, 64, 128] {
            for root in [HighFanOutRoot::None, HighFanOutRoot::AllConsumers] {
                let mut harness = HighFanOutBenchHarness::new();
                let token = harness.run_iteration(HighFanOutBenchSpec {
                    consumer_count,
                    root,
                });
                let rooted = usize::from(root == HighFanOutRoot::AllConsumers);
                assert_eq!(token.registered_pass_count, consumer_count + 1 + rooted);
                assert_eq!(token.registered_resource_count, consumer_count + 1 + rooted);
                assert_eq!(token.logical_consumer_count, consumer_count);
                assert_eq!(
                    token.producer_reverse_consumer_count,
                    consumer_count * rooted
                );
                assert_eq!(
                    token.root_physical_dependency_count,
                    consumer_count * rooted
                );
                assert_eq!(token.compiled_pass_count, (consumer_count + 2) * rooted);
            }
        }
    }
}
