//! Feature-gated harnesses for exercising crate-private RDG hot paths.
//!
//! This module is present in library tests and explicit benchmark-support
//! builds, but absent from default shipping builds. Its narrow semantic API
//! avoids exposing backend command-plan internals or timing an approximate public topology.

use myth_assets::{AssetServer, GeometryHandle, MaterialHandle, TextureHandle};
use myth_resources::{
    BoundingBox, BufferRef, MaterialTrait, RenderableMaterialTrait, ResourceBuilder,
    geometry::Geometry,
    image::{Image, ImageDimension, PixelFormat},
    material::{Material, MaterialSettings, PhysicalMaterial},
    shader_defines::ShaderDefines,
    texture::{Texture, TextureSource},
};
use slotmap::{Key, KeyData};

use crate::core::BindingInterfaceId;
use crate::graph::core::scene_submission::{SceneSubmissionKey, SceneSubmissionPlanId};
use crate::graph::core::types::ErasedResourceNodeId;
use crate::graph::core::{
    BufferDesc, ExecuteContext, FrameArena, GraphStorage, PassBuilder, PassNode,
    RdgDiagnosticsRecorder, RdgFramePhase, RenderGraph, TextureDesc, TextureNodeId,
};
use crate::graph::extracted::ItemShaderFeatures;
use crate::graph::frame::{
    DrawPlanId, DrawPlanKey, DrawPlanStore, ExtractedSourceId, FrameSourceId, PreparedSourceId,
    RenderIntent, RenderLists, ShadowRenderIntent,
};
use crate::graph::scene_pipeline_prepare::{
    PrepassPipelineL1Key, ScenePipelinePrepareStore, SceneRecipeOrdinal,
};
use crate::graph::scene_source_prepare::SceneSourcePrepareStore;
use crate::graph::{
    asset_frame::GeometryFrameStore,
    asset_texture_batch::AssetTextureBatch,
    extracted::{ExtractedRenderItem, ExtractedScene, MaterialClassification},
    material_plan::MaterialPlanStore,
};
use crate::pipeline::RenderPipelineId;

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
                        .filter(|&(_, &id)| graph.storage.scene_submission_is_live(id))
                        .map(|(index, _)| project_plan_key(shadow_plan_key(index, d2_plan_count)))
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

/// Pure-CPU workload for the retained M6 ScenePipeline planner/finalizer.
///
/// `vertex_bindings_per_draw` must exceed the old two-binding inline shape so
/// the probe proves that variable geometry payloads stay in one retained flat
/// table instead of allocating once per command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScenePipelineWarmBenchSpec {
    pub plan_count: usize,
    pub draws_per_plan: usize,
    pub pipeline_variant_count: usize,
    pub vertex_bindings_per_draw: usize,
}

/// O(1) correctness token returned from allocation-counted iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScenePipelineWarmRunToken {
    pub plan_count: usize,
    pub command_count: usize,
    pub vertex_binding_count: usize,
    pub exact_pipeline_request_count: usize,
    pub checksum: u64,
}

/// One synthetic finalized command header.
///
/// This mirrors the allocation-relevant shape of the production draw stream:
/// fixed headers address variable vertex bindings through ranges into one flat
/// table. Physical WGPU owners are deliberately absent so the allocation probe
/// can isolate CPU planning without creating a device.
#[derive(Debug, Clone, Copy)]
struct ScenePipelineFlatCommand {
    ordinal: SceneRecipeOrdinal,
    pipeline: RenderPipelineId,
    sort_key: u64,
    vertex_binding_start: u32,
    vertex_binding_count: u32,
}

/// Retained M6 Gather -> Plan -> Finalize allocation harness.
///
/// Every iteration clears and rebuilds the production Prepass recipe store and
/// its exact request map, restores canonical semantic order, creates per-plan
/// ranges, and materializes a sorted flat command stream. All pipeline results
/// enter Gather as warm L1 hits, so no WGPU object is created or approximated.
pub struct ScenePipelineWarmBenchHarness {
    planner: ScenePipelinePrepareStore,
    plan_store: DrawPlanStore,
    plan_ids: Vec<DrawPlanId>,
    commands: Vec<ScenePipelineFlatCommand>,
    command_vertex_bindings: Vec<u32>,
    command_ranges: Vec<std::ops::Range<u32>>,
}

impl ScenePipelineWarmBenchHarness {
    #[must_use]
    pub fn new() -> Self {
        Self {
            planner: ScenePipelinePrepareStore::new(),
            plan_store: DrawPlanStore::new(),
            plan_ids: Vec::new(),
            commands: Vec::new(),
            command_vertex_bindings: Vec::new(),
            command_ranges: Vec::new(),
        }
    }

    /// Run one complete retained CPU planning iteration.
    #[must_use]
    pub fn run_iteration(&mut self, spec: ScenePipelineWarmBenchSpec) -> ScenePipelineWarmRunToken {
        validate_scene_pipeline_warm_spec(spec);

        self.planner.clear();
        self.plan_store.clear();
        self.plan_ids.clear();
        self.commands.clear();
        self.command_vertex_bindings.clear();
        self.command_ranges.clear();

        self.plan_ids.reserve(spec.plan_count);
        self.command_ranges.reserve(spec.plan_count);
        let command_count = spec
            .plan_count
            .checked_mul(spec.draws_per_plan)
            .expect("scene-pipeline benchmark command count overflowed");
        self.commands.reserve(command_count);
        self.command_vertex_bindings.reserve(
            command_count
                .checked_mul(spec.vertex_bindings_per_draw)
                .expect("scene-pipeline benchmark vertex binding count overflowed"),
        );

        for plan_index in 0..spec.plan_count {
            self.plan_ids
                .push(self.plan_store.reserve_plan(DrawPlanKey::Prepass {
                    needs_normal: plan_index & 1 != 0,
                    needs_feature_id: plan_index & 2 != 0,
                    needs_velocity: plan_index & 4 != 0,
                }));
        }

        // Gather each plan in reverse intent order. Planning must restore
        // `(plan reservation, intent)` order independently of gather order.
        for gather_plan in 0..spec.plan_count {
            let plan_index = spec.plan_count - gather_plan - 1;
            let plan = self.plan_ids[plan_index];
            for gather_intent in 0..spec.draws_per_plan {
                let intent_index = spec.draws_per_plan - gather_intent - 1;
                let source_index = plan_index * spec.draws_per_plan + intent_index;
                let variant = source_index % spec.pipeline_variant_count;
                self.planner.push_prepass(
                    plan,
                    SceneRecipeOrdinal::new(plan_index, intent_index),
                    FrameSourceId::from_index(source_index),
                    PreparedSourceId::from_index(source_index),
                    scene_pipeline_prepass_key(variant, plan_index),
                    Some(RenderPipelineId(
                        u32::try_from(variant).expect("pipeline variant exceeds u32"),
                    )),
                    u32::try_from(variant).expect("material index exceeds u32"),
                    intent_index as f32,
                );
            }
        }

        self.planner.plan(spec.plan_count);
        let mut checksum = 0_u64;
        for plan_index in 0..spec.plan_count {
            let command_start = u32::try_from(self.commands.len())
                .expect("scene-pipeline command table exceeds u32");
            let recipes = self.planner.prepass_draws(plan_index);
            assert_eq!(recipes.len(), spec.draws_per_plan);
            for (intent_index, &recipe) in recipes.iter().enumerate() {
                assert_eq!(
                    recipe.ordinal,
                    SceneRecipeOrdinal::new(plan_index, intent_index),
                    "planner did not restore canonical recipe order"
                );
                assert_eq!(recipe.draw_plan, self.plan_ids[plan_index]);
                let pipeline = self
                    .planner
                    .prepass_result(recipe.pipeline)
                    .expect("warm Prepass request lost its cached pipeline ID");
                let vertex_binding_start = u32::try_from(self.command_vertex_bindings.len())
                    .expect("scene-pipeline vertex table exceeds u32");
                for binding_index in 0..spec.vertex_bindings_per_draw {
                    let token = recipe
                        .prepared_source
                        .index()
                        .checked_mul(spec.vertex_bindings_per_draw)
                        .and_then(|base| base.checked_add(binding_index))
                        .and_then(|token| u32::try_from(token).ok())
                        .expect("scene-pipeline vertex token exceeds u32");
                    self.command_vertex_bindings.push(token);
                }
                let sort_key =
                    (u64::try_from(pipeline.index()).expect("pipeline index overflowed") << 32)
                        | u64::from(recipe.material_index);
                self.commands.push(ScenePipelineFlatCommand {
                    ordinal: recipe.ordinal,
                    pipeline,
                    sort_key,
                    vertex_binding_start,
                    vertex_binding_count: u32::try_from(spec.vertex_bindings_per_draw)
                        .expect("vertex binding count exceeds u32"),
                });
            }
            let command_end = u32::try_from(self.commands.len())
                .expect("scene-pipeline command table exceeds u32");
            self.commands[command_start as usize..command_end as usize]
                .sort_unstable_by_key(|command| command.sort_key);
            self.command_ranges.push(command_start..command_end);
        }

        for (plan_index, range) in self.command_ranges.iter().enumerate() {
            assert_eq!(range.end - range.start, spec.draws_per_plan as u32);
            for command in &self.commands[range.start as usize..range.end as usize] {
                let vertex_end = command
                    .vertex_binding_start
                    .checked_add(command.vertex_binding_count)
                    .expect("scene-pipeline vertex range overflowed");
                let vertices = &self.command_vertex_bindings
                    [command.vertex_binding_start as usize..vertex_end as usize];
                checksum = checksum
                    .wrapping_mul(0x9e37_79b1_85eb_ca87)
                    .wrapping_add(plan_index as u64)
                    .wrapping_add(u64::from(command.ordinal.intent_index))
                    .wrapping_add(command.pipeline.index() as u64)
                    .wrapping_add(command.sort_key)
                    .wrapping_add(u64::from(vertices[0]))
                    .wrapping_add(u64::from(vertices[vertices.len() - 1]));
            }
        }

        ScenePipelineWarmRunToken {
            plan_count: self.command_ranges.len(),
            command_count: self.commands.len(),
            vertex_binding_count: self.command_vertex_bindings.len(),
            exact_pipeline_request_count: self.planner.prepass_request_count(),
            checksum,
        }
    }
}

impl Default for ScenePipelineWarmBenchHarness {
    fn default() -> Self {
        Self::new()
    }
}

/// Allocation probe for post-finalize morph-history bookkeeping.
///
/// `morph_source_stride == 0` models a wholly rigid scene and must leave the
/// cold morph arena untouched. A nonzero stride records every Nth source once
/// while `plan_alias_count` replays Opaque/Prepass/Shadow-style aliases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MorphHistoryWarmBenchSpec {
    pub source_count: usize,
    pub plan_alias_count: usize,
    pub morph_source_stride: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MorphHistoryWarmRunToken {
    pub recorded_source_count: usize,
    pub checksum: u64,
}

pub struct MorphHistoryWarmBenchHarness {
    plans: DrawPlanStore,
}

impl MorphHistoryWarmBenchHarness {
    #[must_use]
    pub fn new() -> Self {
        Self {
            plans: DrawPlanStore::new(),
        }
    }

    #[must_use]
    pub fn run_iteration(&mut self, spec: MorphHistoryWarmBenchSpec) -> MorphHistoryWarmRunToken {
        assert!(
            spec.plan_alias_count > 0,
            "at least one live alias is required"
        );
        self.plans.clear();

        if spec.morph_source_stride != 0 {
            for _ in 0..spec.plan_alias_count {
                for source_index in (0..spec.source_count).step_by(spec.morph_source_stride) {
                    self.plans
                        .record_emitted_morph_history_source(FrameSourceId::from_index(
                            source_index,
                        ));
                }
            }
        }

        let mut checksum = 0_u64;
        for source in self.plans.emitted_morph_history_sources() {
            checksum = checksum
                .wrapping_mul(0x9e37_79b1_85eb_ca87)
                .wrapping_add(source.index() as u64 + 1);
        }
        MorphHistoryWarmRunToken {
            recorded_source_count: self.plans.emitted_morph_history_sources().len(),
            checksum,
        }
    }
}

impl Default for MorphHistoryWarmBenchHarness {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_scene_pipeline_warm_spec(spec: ScenePipelineWarmBenchSpec) {
    assert!(spec.plan_count > 0, "benchmark requires at least one plan");
    assert!(
        spec.plan_count <= 8,
        "synthetic Prepass keys encode the plan output mask in three bits"
    );
    assert!(
        spec.draws_per_plan > 0,
        "benchmark requires at least one draw per plan"
    );
    assert!(
        spec.pipeline_variant_count > 0,
        "benchmark requires at least one pipeline variant"
    );
    assert!(
        spec.vertex_bindings_per_draw > 2,
        "benchmark must cross the old two-binding inline shape"
    );
}

fn scene_pipeline_prepass_key(variant: usize, plan_index: usize) -> PrepassPipelineL1Key {
    let variant_u64 = u64::try_from(variant).expect("pipeline variant exceeds u64");
    let variant_u32 = u32::try_from(variant).expect("pipeline variant exceeds u32");
    let discriminator = variant_u64 + 1;
    PrepassPipelineL1Key {
        geometry_handle: GeometryHandle::from(KeyData::from_ffi(discriminator * 2 + 1)),
        geometry_asset_version: variant_u32,
        geometry_asset_identity: discriminator * 11 + 1,
        geometry_layout_version: discriminator * 13 + 1,
        material_handle: MaterialHandle::from(KeyData::from_ffi(discriminator * 2 + 2)),
        material_asset_version: variant_u32,
        material_asset_identity: discriminator * 17 + 1,
        material_version: discriminator * 19 + 1,
        item_variants: variant_u32,
        item_shader_features: ItemShaderFeatures::from_bits_retain(discriminator as u8 & 3),
        shader_source_identity: discriminator * 29 + 1,
        scene_defines_hash: discriminator * 31 + 1,
        pipeline_settings_version: discriminator * 37 + 1,
        global_binding_interface_id: BindingInterfaceId::from_raw(discriminator * 3 + 1),
        material_binding_interface_id: BindingInterfaceId::from_raw(discriminator * 3 + 2),
        object_binding_interface_id: BindingInterfaceId::from_raw(discriminator * 3 + 3),
        vertex_layout_id: discriminator * 41 + 1,
        bind_group_layout_ids: [
            discriminator * 43 + 1,
            discriminator * 43 + 2,
            discriminator * 43 + 3,
        ],
        depth_format: wgpu::TextureFormat::Depth32Float,
        topology: wgpu::PrimitiveTopology::TriangleList,
        cull_mode: Some(wgpu::Face::Back),
        front_face: wgpu::FrontFace::Ccw,
        needs_normal: plan_index & 1 != 0,
        needs_feature_id: plan_index & 2 != 0,
        needs_velocity: plan_index & 4 != 0,
    }
}

/// Cumulative boundary reached by the M6.5 Helmet-like CPU benchmark.
///
/// Criterion measures each prefix independently. Subtracting adjacent prefix
/// medians gives a phase estimate without adding timers or atomics to the
/// production renderer. The final stage joins the real Material/AssetTexture
/// transaction to the retained production ScenePipeline planner.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaterialSceneCpuBenchStage {
    /// Exact visible Geometry owner capture used to qualify Material topology.
    GeometryCapture,
    /// Add exact Material owner capture, texture visitation, and AssetTexture
    /// request interning. DPE-live shader derivation is deliberately later.
    MaterialCapture,
    /// Add the canonical Texture -> Image -> Texture capture/revalidation batch.
    AssetTextureCapture,
    /// Add Material owner/content revalidation and draw-plan texture recipes.
    MaterialTextureFinalize,
    /// Add warm Gather -> Plan -> immutable ScenePipeline stream finalization.
    WarmScenePipeline,
}

impl MaterialSceneCpuBenchStage {
    /// Stable Criterion label ordered by the cumulative production boundary.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::GeometryCapture => "00_geometry_capture",
            Self::MaterialCapture => "01_material_capture",
            Self::AssetTextureCapture => "02_asset_texture_capture",
            Self::MaterialTextureFinalize => "03_material_texture_finalize",
            Self::WarmScenePipeline => "04_warm_scene_pipeline",
        }
    }
}

/// O(1) correctness token for one Helmet-like cumulative CPU iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialSceneCpuRunToken {
    /// Number of ready Asset textures configured on the PhysicalMaterial.
    pub configured_texture_count: usize,
    /// Number of draw-plan topology recipes published after finalization.
    pub topology_recipe_count: usize,
    /// Unique ready Asset textures attached to the Opaque recipe.
    pub opaque_texture_count: usize,
    /// Unique ready Asset textures attached to the shadow recipe.
    pub shadow_texture_count: usize,
    /// Commands emitted by the final warm ScenePipeline stage.
    pub scene_command_count: usize,
    /// Unique source count selected from overlapping Opaque/Prepass/shadow keys.
    pub selected_source_count: usize,
    /// Black-box-friendly identity covering both material and pipeline output.
    pub checksum: u64,
}

/// Benchmark-only custom-material escape hatch for exercising dependency
/// counts beyond the built-in Physical material's eight Asset texture slots.
/// The allocation scenario stops before physical binding preparation; its
/// texture visitation still uses the exact production MaterialPlan and
/// AssetTexture transaction.
#[derive(Debug)]
struct ManyTextureBenchMaterial {
    textures: Vec<TextureSource>,
    uniform: BufferRef,
}

impl ManyTextureBenchMaterial {
    fn new(textures: &[TextureHandle]) -> Self {
        Self {
            textures: textures.iter().copied().map(TextureSource::Asset).collect(),
            uniform: BufferRef::empty(wgpu::BufferUsages::UNIFORM, Some("M6.5 bench uniform")),
        }
    }
}

impl MaterialTrait for ManyTextureBenchMaterial {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl RenderableMaterialTrait for ManyTextureBenchMaterial {
    fn shader_name(&self) -> &'static str {
        "m65_many_texture_bench"
    }

    fn version(&self) -> u64 {
        1
    }

    fn shader_defines(&self) -> ShaderDefines {
        ShaderDefines::new()
    }

    fn settings(&self) -> MaterialSettings {
        MaterialSettings::default()
    }

    fn visit_textures(&self, visitor: &mut dyn FnMut(&TextureSource)) {
        for texture in &self.textures {
            visitor(texture);
        }
    }

    fn define_bindings<'a>(&'a self, _builder: &mut ResourceBuilder<'a>) {}

    fn uniform_buffer(&self) -> BufferRef {
        self.uniform.clone()
    }

    fn with_uniform_bytes(&self, f: &mut dyn FnMut(&[u8])) {
        f(&[]);
    }
}

/// Real-asset, GPU-free M6.5 regression harness matching the Helmet hot shape.
///
/// The fixture owns one visible source, one built-in PhysicalMaterial (up to
/// eight textures) or custom Material escape hatch, and distinct ready
/// Texture/Image pairs. Every iteration reuses the production
/// graph-attempt stores while running their exact AssetStorage snapshot and
/// T -> I -> T revalidation paths. The final stage adds one warm production
/// ScenePipeline recipe. WGPU object creation is intentionally absent: this
/// isolates the CPU transaction that remains even when all physical resources
/// and pipelines are resident.
pub struct MaterialSceneCpuBenchHarness {
    assets: AssetServer,
    extracted_scene: ExtractedScene,
    render_lists: RenderLists,
    geometry_frames: GeometryFrameStore,
    material_plans: MaterialPlanStore,
    asset_textures: AssetTextureBatch,
    source_selection: SceneSourcePrepareStore,
    scene_pipeline: ScenePipelineWarmBenchHarness,
    configured_texture_count: usize,
}

impl MaterialSceneCpuBenchHarness {
    /// Create one immutable Helmet-like fixture.
    ///
    /// `texture_count = 0` is a useful PhysicalMaterial control arm; five is
    /// the representative glTF PBR shape and values above eight exercise the
    /// custom-material spill boundary.
    #[must_use]
    pub fn new(texture_count: usize) -> Self {
        let assets = AssetServer::new();
        let geometry = assets.geometries.add(Geometry::new());
        let textures: Vec<_> = (0..texture_count)
            .map(|texture_index| {
                let texel = u8::try_from(texture_index + 1)
                    .expect("benchmark texture index exceeds u8")
                    .wrapping_mul(23);
                let image = assets.images.add(Image::new(
                    8,
                    8,
                    1,
                    ImageDimension::D2,
                    PixelFormat::Rgba8Unorm,
                    Some(vec![texel; 8 * 8 * 4]),
                ));
                assets.textures.add(Texture::new_2d(
                    Some("M6.5 resident material texture"),
                    image,
                ))
            })
            .collect();

        let material = if texture_count <= 8 {
            let physical = PhysicalMaterial::new(glam::Vec4::ONE);
            configure_physical_texture_slots(&physical, &textures);
            assets.materials.add(Material::from(physical))
        } else {
            assets
                .materials
                .add(Material::new_custom(ManyTextureBenchMaterial::new(
                    &textures,
                )))
        };

        let (_, geometry_asset_version, geometry_asset_identity) = assets
            .geometries
            .get_entry_with_identity(geometry)
            .expect("benchmark Geometry missing");
        let (material_owner, material_asset_version, material_asset_identity) = assets
            .materials
            .get_entry_with_identity(material)
            .expect("benchmark PhysicalMaterial missing");
        let material_version = material_owner.data.version();
        drop(material_owner);

        let mut extracted_scene = ExtractedScene::new();
        extracted_scene.render_items.push(ExtractedRenderItem {
            node_handle: myth_scene::NodeHandle::null(),
            world_matrix: glam::Mat4::IDENTITY,
            prev_world_matrix: glam::Mat4::IDENTITY,
            geometry,
            material,
            geometry_asset_version,
            geometry_asset_identity,
            material_asset_version,
            material_asset_identity,
            material_version,
            material_index: 0,
            material_classification: MaterialClassification::empty(),
            skeleton: None,
            item_variant_flags: 0,
            item_shader_features: ItemShaderFeatures::empty(),
            cast_shadows: true,
            receive_shadows: true,
            world_aabb: BoundingBox::default(),
        });

        let mut render_lists = RenderLists::new();
        let source = render_lists.intern_frame_source(ExtractedSourceId::from_index(0));
        render_lists.insert_opaque(RenderIntent {
            source,
            geometry_handle: geometry,
            material_handle: material,
            material_index: 0,
            distance_sq: 0.0,
        });
        // The same Material dependency set is consumed by Opaque/Prepass and
        // one shadow view. Prepass aliases Opaque topology in DrawPlanStore;
        // the final warm stage exercises its retained production planner.
        for key in [(CUBE_LIGHT_ID_BASE, 0), (CUBE_LIGHT_ID_BASE + 1, 1)] {
            render_lists.shadow_queues.insert(
                key,
                vec![ShadowRenderIntent {
                    source,
                    geometry_handle: geometry,
                    material_handle: material,
                }],
            );
        }

        Self {
            assets,
            extracted_scene,
            render_lists,
            geometry_frames: GeometryFrameStore::new(),
            material_plans: MaterialPlanStore::new(),
            asset_textures: AssetTextureBatch::new(),
            source_selection: SceneSourcePrepareStore::new(),
            scene_pipeline: ScenePipelineWarmBenchHarness::new(),
            configured_texture_count: texture_count,
        }
    }

    /// Run one cumulative production-path prefix.
    ///
    /// Attempt stores are cleared first, preserving their capacities exactly
    /// like GraphStorage. A stage never fabricates the output of a prior phase.
    #[must_use]
    pub fn run_iteration(&mut self, stage: MaterialSceneCpuBenchStage) -> MaterialSceneCpuRunToken {
        self.geometry_frames.clear();
        self.material_plans.clear();
        self.asset_textures.clear();
        self.source_selection.clear();

        self.geometry_frames.capture_visible(
            &self.assets,
            &self.extracted_scene,
            &self.render_lists,
        );

        if stage == MaterialSceneCpuBenchStage::GeometryCapture {
            return self.token(0, 0, 0, 0, None);
        }

        self.material_plans.capture(
            &self.assets,
            &self.extracted_scene,
            &self.render_lists,
            self.geometry_frames
                .eligibility(&self.extracted_scene, &self.render_lists),
            &mut self.asset_textures,
        );

        if stage == MaterialSceneCpuBenchStage::MaterialCapture {
            return self.token(0, 0, 0, 0, None);
        }

        self.asset_textures.capture_requested(&self.assets);

        if stage == MaterialSceneCpuBenchStage::AssetTextureCapture {
            return self.token(0, 0, 0, 0, None);
        }

        self.material_plans.finalize_texture_snapshots(
            &self.assets,
            &self.extracted_scene,
            &self.render_lists,
            &self.asset_textures,
        );
        let recipes = self.material_plans.take_topology_recipes();
        let topology_recipe_count = recipes.len();
        let opaque_texture_count = recipes
            .iter()
            .find(|recipe| recipe.key == DrawPlanKey::Opaque)
            .map_or(0, |recipe| recipes.textures(recipe).len());
        let shadow_texture_count = recipes
            .iter()
            .find(|recipe| matches!(recipe.key, DrawPlanKey::Shadow { .. }))
            .map_or(0, |recipe| recipes.textures(recipe).len());
        let shadow_recipe_count = recipes
            .iter()
            .filter(|recipe| matches!(recipe.key, DrawPlanKey::Shadow { .. }))
            .count();

        assert_eq!(
            opaque_texture_count, self.configured_texture_count,
            "stable PhysicalMaterial lost a ready texture dependency"
        );
        assert_eq!(
            shadow_texture_count, self.configured_texture_count,
            "stable Material lost a ready shadow texture dependency"
        );
        assert_eq!(shadow_recipe_count, 2, "benchmark shadow views were lost");
        assert!(
            recipes
                .iter()
                .filter(|recipe| matches!(recipe.key, DrawPlanKey::Shadow { .. }))
                .all(|recipe| recipes.textures(recipe).len() == shadow_texture_count),
            "overlapping shadow views observed different Material dependencies"
        );

        if stage == MaterialSceneCpuBenchStage::MaterialTextureFinalize {
            let token = self.token(
                topology_recipe_count,
                opaque_texture_count,
                shadow_texture_count,
                0,
                None,
            );
            self.material_plans.recycle_topology_recipes(recipes);
            return token;
        }

        self.source_selection.select_live_sources(
            &self.render_lists,
            [
                DrawPlanKey::Opaque,
                DrawPlanKey::Prepass {
                    needs_normal: true,
                    needs_feature_id: true,
                    needs_velocity: true,
                },
                DrawPlanKey::Shadow {
                    light_id: CUBE_LIGHT_ID_BASE,
                    layer_index: 0,
                },
                DrawPlanKey::Shadow {
                    light_id: CUBE_LIGHT_ID_BASE + 1,
                    layer_index: 1,
                },
            ],
        );
        let selected_source_count = self.source_selection.live_sources().len();
        let scene = self
            .scene_pipeline
            .run_iteration(ScenePipelineWarmBenchSpec {
                plan_count: 1,
                draws_per_plan: 1,
                pipeline_variant_count: 1,
                vertex_bindings_per_draw: 3,
            });
        let token = self.token(
            topology_recipe_count,
            opaque_texture_count,
            shadow_texture_count,
            selected_source_count,
            Some(scene),
        );
        self.material_plans.recycle_topology_recipes(recipes);
        token
    }

    fn token(
        &self,
        topology_recipe_count: usize,
        opaque_texture_count: usize,
        shadow_texture_count: usize,
        selected_source_count: usize,
        scene: Option<ScenePipelineWarmRunToken>,
    ) -> MaterialSceneCpuRunToken {
        let scene_command_count = scene.map_or(0, |token| token.command_count);
        let scene_checksum = scene.map_or(0, |token| token.checksum);
        MaterialSceneCpuRunToken {
            configured_texture_count: self.configured_texture_count,
            topology_recipe_count,
            opaque_texture_count,
            shadow_texture_count,
            scene_command_count,
            selected_source_count,
            checksum: scene_checksum
                .wrapping_add(self.configured_texture_count as u64)
                .wrapping_add((topology_recipe_count as u64) << 16)
                .wrapping_add((opaque_texture_count as u64) << 32)
                .wrapping_add((shadow_texture_count as u64) << 48)
                .wrapping_add((selected_source_count as u64) << 8),
        }
    }
}

fn configure_physical_texture_slots(material: &PhysicalMaterial, textures: &[TextureHandle]) {
    for (index, &texture) in textures.iter().enumerate() {
        match index {
            0 => material.set_map(Some(texture)),
            1 => material.set_normal_map(Some(texture)),
            2 => material.set_roughness_map(Some(texture)),
            3 => material.set_metalness_map(Some(texture)),
            4 => material.set_emissive_map(Some(texture)),
            5 => material.set_ao_map(Some(texture)),
            6 => material.set_specular_map(Some(texture)),
            7 => material.set_specular_intensity_map(Some(texture)),
            _ => unreachable!("texture-count validation allows at most eight slots"),
        }
    }
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
        HighFanOutBenchSpec, HighFanOutRoot, MaterialSceneCpuBenchHarness,
        MaterialSceneCpuBenchStage, RdgBenchDiagnosticsMode, RdgRecorderLifecycleBenchHarness,
        project_plan_key, shadow_plan_key,
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
    fn helmet_like_cpu_harness_preserves_every_cumulative_boundary() {
        for texture_count in [0, 5, 12] {
            let mut harness = MaterialSceneCpuBenchHarness::new(texture_count);
            for stage in [
                MaterialSceneCpuBenchStage::GeometryCapture,
                MaterialSceneCpuBenchStage::MaterialCapture,
                MaterialSceneCpuBenchStage::AssetTextureCapture,
                MaterialSceneCpuBenchStage::MaterialTextureFinalize,
                MaterialSceneCpuBenchStage::WarmScenePipeline,
            ] {
                let token = harness.run_iteration(stage);
                assert_eq!(token.configured_texture_count, texture_count);
                if stage >= MaterialSceneCpuBenchStage::MaterialTextureFinalize {
                    assert_eq!(token.topology_recipe_count, 4);
                    assert_eq!(token.opaque_texture_count, texture_count);
                    assert_eq!(token.shadow_texture_count, texture_count);
                } else {
                    assert_eq!(token.topology_recipe_count, 0);
                    assert_eq!(token.opaque_texture_count, 0);
                    assert_eq!(token.shadow_texture_count, 0);
                }
                assert_eq!(
                    token.scene_command_count,
                    usize::from(stage == MaterialSceneCpuBenchStage::WarmScenePipeline)
                );
                assert_eq!(
                    token.selected_source_count,
                    usize::from(stage == MaterialSceneCpuBenchStage::WarmScenePipeline)
                );
            }
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
}
