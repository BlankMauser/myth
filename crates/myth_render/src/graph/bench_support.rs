//! Feature-gated harnesses for exercising crate-private RDG hot paths.
//!
//! This module is present in library tests and explicit benchmark-support
//! builds, but absent from default shipping builds. Its narrow semantic API
//! avoids exposing DrawPlan internals or timing an approximate public topology.

use crate::graph::core::node::PassRecord;
use crate::graph::core::{
    BufferDesc, ExecuteContext, FrameArena, GraphStorage, PassBuilder, PassNode, RenderGraph,
    TextureDesc, TextureNodeId,
};
use crate::graph::frame::{DrawPlanId, DrawPlanKey, DrawPlanStore};

const CUBE_LIGHT_ID_BASE: u64 = 1 << 32;

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

/// Exact workload passed through the production DrawPlan demand path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawDemandBenchSpec {
    /// Total distinct Shadow plans. One quarter is D2 and three quarters Cube.
    pub plan_count: usize,
    /// Number of logical material texture dependencies per plan.
    pub textures_per_plan: usize,
    /// Whether plans share or own disjoint dependency sets.
    pub dependency_pattern: DrawDemandDependencyPattern,
    /// Number of times each plan is demanded by its Shadow pass.
    pub demand_repetitions: usize,
    /// Whether the graph is compiled after construction.
    pub compile_topology: bool,
    /// Output branch used to seed DPE.
    pub root: DrawDemandRoot,
}

/// Stable benchmark projection of one internal Shadow DrawPlan key.
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
}

/// Reusable owner for exact DrawPlan demand benchmark iterations.
///
/// GraphStorage's top-level collections, FrameArena, and harness scratch retain
/// their high-water-mark capacity. Per-record SmallVec spills are deliberately
/// destroyed by GraphStorage::clear and remain visible to allocation probes.
pub struct DrawDemandBenchHarness {
    storage: GraphStorage,
    arena: FrameArena,
    plan_ids: Vec<DrawPlanId>,
    dependency_scratch: Vec<TextureNodeId>,
}

impl DrawDemandBenchHarness {
    /// Construct an empty reusable harness.
    #[must_use]
    pub fn new() -> Self {
        Self {
            storage: GraphStorage::new(),
            arena: FrameArena::new(),
            plan_ids: Vec::new(),
            dependency_scratch: Vec::new(),
        }
    }

    /// Run one production-path iteration with only an O(1) timed result.
    #[must_use]
    pub fn run_iteration(&mut self, spec: DrawDemandBenchSpec) -> DrawDemandRunToken {
        self.run_iteration_with(spec, |graph, indices| {
            let d2 = &graph.storage.passes[indices.d2];
            let cube = &graph.storage.passes[indices.cube];
            DrawDemandRunToken {
                registered_pass_count: graph.storage.passes.len(),
                registered_resource_count: graph.storage.resources.len(),
                d2_read_count: d2.reads.len(),
                cube_read_count: cube.reads.len(),
                d2_demand_count: d2.draw_plan_demands.len(),
                cube_demand_count: cube.draw_plan_demands.len(),
                compiled_pass_count: if spec.compile_topology {
                    graph.storage.execution_queue.len()
                } else {
                    0
                },
            }
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
        self.run_iteration_with(spec, |graph, indices| {
            let draw_plans = &graph.storage.draw_plans;
            DrawDemandBenchObservation {
                d2: observe_pass(&graph.storage.passes[indices.d2], draw_plans),
                cube: observe_pass(&graph.storage.passes[indices.cube], draw_plans),
                registered_pass_count: graph.storage.passes.len(),
                registered_resource_count: graph.storage.resources.len(),
                resource_consumer_edge_count: graph
                    .storage
                    .resources
                    .iter()
                    .map(|resource| resource.consumers.len())
                    .sum(),
                live_plans: if spec.compile_topology {
                    draw_plans.live_keys().map(project_plan_key).collect()
                } else {
                    Vec::new()
                },
                compiled_pass_count: if spec.compile_topology {
                    graph.storage.execution_queue.len()
                } else {
                    0
                },
                root_physical_dependencies: indices.root.map_or_else(Vec::new, |index| {
                    graph.storage.passes[index].physical_dependencies.to_vec()
                }),
            }
        })
    }

    fn run_iteration_with<R>(
        &mut self,
        spec: DrawDemandBenchSpec,
        observe: impl FnOnce(&RenderGraph<'_>, DrawDemandRunIndices) -> R,
    ) -> R {
        validate_spec(spec);

        let Self {
            storage,
            arena,
            plan_ids,
            dependency_scratch,
        } = self;
        arena.reset();
        plan_ids.clear();
        dependency_scratch.clear();

        let mut graph = RenderGraph::new(storage, arena);
        let texture_desc = dependency_desc();

        if spec.dependency_pattern == DrawDemandDependencyPattern::Shared {
            dependency_scratch.reserve(spec.textures_per_plan);
            for _ in 0..spec.textures_per_plan {
                dependency_scratch.push(register_texture_input(&mut graph, texture_desc));
            }
        }

        let d2_plan_count = spec.plan_count / 4;
        plan_ids.reserve(spec.plan_count);
        for plan_index in 0..spec.plan_count {
            let key = shadow_plan_key(plan_index, d2_plan_count);
            let plan = graph.reserve_draw_plan(key);

            match spec.dependency_pattern {
                DrawDemandDependencyPattern::Shared => {
                    graph.set_draw_plan_texture_dependencies(
                        key,
                        dependency_scratch.iter().copied(),
                    );
                }
                DrawDemandDependencyPattern::Disjoint => {
                    dependency_scratch.clear();
                    dependency_scratch.reserve(spec.textures_per_plan);
                    for _ in 0..spec.textures_per_plan {
                        dependency_scratch.push(register_texture_input(&mut graph, texture_desc));
                    }
                    graph.set_draw_plan_texture_dependencies(
                        key,
                        dependency_scratch.iter().copied(),
                    );
                }
            }
            plan_ids.push(plan);
        }

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
            },
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

fn register_texture_input(graph: &mut RenderGraph<'_>, desc: TextureDesc) -> TextureNodeId {
    // Producer-less/non-root matches the steady resident lazy-import topology
    // without invoking Asset-specific lookup and validation.
    graph.register_texture("Shadow_Material_Texture", desc, false)
}

fn shadow_plan_key(plan_index: usize, d2_plan_count: usize) -> DrawPlanKey {
    if plan_index < d2_plan_count {
        DrawPlanKey::Shadow {
            light_id: plan_index as u64,
            layer_index: 0,
        }
    } else {
        let cube_face_index = plan_index - d2_plan_count;
        DrawPlanKey::Shadow {
            light_id: CUBE_LIGHT_ID_BASE + (cube_face_index / 6) as u64,
            layer_index: (cube_face_index % 6) as u32,
        }
    }
}

fn demand_plans(builder: &mut PassBuilder<'_, '_>, plans: &[DrawPlanId], repetitions: usize) {
    for &plan in plans {
        for _ in 0..repetitions {
            builder.demand_draw_plan(plan);
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

fn observe_pass(pass: &PassRecord, draw_plans: &DrawPlanStore) -> DrawDemandPassObservation {
    DrawDemandPassObservation {
        logical_reads: pass
            .reads
            .iter()
            .map(|id| (id.index(), id.version()))
            .collect(),
        demanded_plans: pass
            .draw_plan_demands
            .iter()
            .map(|&id| resolve_plan_key(draw_plans, id))
            .collect(),
        physical_dependencies: pass.physical_dependencies.to_vec(),
    }
}

fn resolve_plan_key(draw_plans: &DrawPlanStore, id: DrawPlanId) -> DrawDemandPlanKey {
    (0..draw_plans.plan_count())
        .find_map(|index| {
            let meta = draw_plans
                .plan_meta(index)
                .expect("benchmark DrawPlan table changed while observing");
            (meta.id == id).then(|| project_plan_key(meta.key))
        })
        .expect("benchmark pass retained an unknown DrawPlan ID")
}

fn project_plan_key(key: DrawPlanKey) -> DrawDemandPlanKey {
    let DrawPlanKey::Shadow {
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

#[cfg(test)]
mod tests {
    use super::{
        DrawDemandBenchHarness, DrawDemandBenchSpec, DrawDemandDependencyPattern,
        DrawDemandPlanKey, DrawDemandRoot, project_plan_key, shadow_plan_key,
    };

    fn spec(
        pattern: DrawDemandDependencyPattern,
        repetitions: usize,
        compile_topology: bool,
        root: DrawDemandRoot,
    ) -> DrawDemandBenchSpec {
        DrawDemandBenchSpec {
            plan_count: 32,
            textures_per_plan: 8,
            dependency_pattern: pattern,
            demand_repetitions: repetitions,
            compile_topology,
            root,
        }
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
            let mut harness = DrawDemandBenchHarness::new();
            let single =
                harness.run_observed_iteration(spec(pattern, 1, true, DrawDemandRoot::Both));
            let duplicate =
                harness.run_observed_iteration(spec(pattern, 2, true, DrawDemandRoot::Both));
            assert_eq!(single, duplicate);
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
}
