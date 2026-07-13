//! Thread-scoped allocation baseline for exact RDG SceneSubmission workloads.
//!
//! This is deliberately separate from Criterion so its allocator wrapper
//! cannot perturb wall-time measurements. Run with:
//! `cargo bench --bench rdg_allocation_probe --features rdg_bench_support`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use myth_render::graph::bench_support::{
    DrawDemandBenchHarness, DrawDemandBenchSpec, DrawDemandDependencyPattern, DrawDemandRoot,
    DrawDemandTextureTopology, HighFanOutBenchHarness, HighFanOutBenchSpec, HighFanOutRoot,
    HighFanOutRunToken, RdgBenchDiagnosticsMode, RdgRecorderLifecycleBenchHarness,
    RdgRecorderLifecycleToken, ScenePipelineWarmBenchHarness, ScenePipelineWarmBenchSpec,
    ScenePipelineWarmRunToken,
};

#[global_allocator]
static ALLOCATOR: ThreadScopedAllocator = ThreadScopedAllocator;

struct ThreadScopedAllocator;

unsafe impl GlobalAlloc for ThreadScopedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegates the exact layout to the process System allocator.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record(|stats| {
                stats.alloc_calls += 1;
                stats.allocated_bytes += layout.size() as u64;
            });
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegates the exact layout to the process System allocator.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record(|stats| {
                stats.alloc_zeroed_calls += 1;
                stats.allocated_bytes += layout.size() as u64;
            });
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(|stats| {
            stats.dealloc_calls += 1;
            stats.deallocated_bytes += layout.size() as u64;
        });
        // SAFETY: `ptr` was allocated by System with the matching layout.
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Delegates the original pointer/layout and requested size to System.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            record(|stats| {
                stats.realloc_calls += 1;
                stats.reallocated_old_bytes += layout.size() as u64;
                stats.reallocated_new_bytes += new_size as u64;
            });
        }
        new_ptr
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AllocationStats {
    alloc_calls: u64,
    alloc_zeroed_calls: u64,
    realloc_calls: u64,
    dealloc_calls: u64,
    allocated_bytes: u64,
    deallocated_bytes: u64,
    reallocated_old_bytes: u64,
    reallocated_new_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct CounterState {
    active: bool,
    stats: AllocationStats,
}

thread_local! {
    static COUNTER: Cell<CounterState> = const { Cell::new(CounterState {
        active: false,
        stats: AllocationStats {
            alloc_calls: 0,
            alloc_zeroed_calls: 0,
            realloc_calls: 0,
            dealloc_calls: 0,
            allocated_bytes: 0,
            deallocated_bytes: 0,
            reallocated_old_bytes: 0,
            reallocated_new_bytes: 0,
        },
    }) };
}

fn record(update: impl FnOnce(&mut AllocationStats)) {
    let _ = COUNTER.try_with(|counter| {
        let mut state = counter.get();
        if state.active {
            update(&mut state.stats);
            counter.set(state);
        }
    });
}

struct CountingScope {
    active: bool,
}

impl CountingScope {
    fn begin() -> Self {
        COUNTER.with(|counter| {
            let state = counter.get();
            assert!(!state.active, "allocation measurement scopes cannot nest");
            counter.set(CounterState {
                active: true,
                stats: AllocationStats::default(),
            });
        });
        Self { active: true }
    }

    fn finish(mut self) -> AllocationStats {
        let stats = disable_counter();
        self.active = false;
        stats
    }
}

impl Drop for CountingScope {
    fn drop(&mut self) {
        if self.active {
            let _ = disable_counter();
        }
    }
}

fn disable_counter() -> AllocationStats {
    COUNTER.with(|counter| {
        let mut state = counter.get();
        assert!(state.active, "allocation counter was not active");
        state.active = false;
        counter.set(state);
        state.stats
    })
}

fn measure<T: Copy>(operation: impl FnOnce() -> T) -> (T, AllocationStats) {
    // Prime native TLS before enabling the allocator callback.
    COUNTER.with(|counter| black_box(counter.get()));
    let scope = CountingScope::begin();
    let value = operation();
    let stats = scope.finish();
    (value, stats)
}

fn main() {
    verify_counter_is_live();
    println!(
        "RDG full-frame allocation baseline (M1 removes demand-generated churn; flat topology storage is the final zero-allocation gate)"
    );
    println!(
        "dealloc columns include preceding same-shape topology retirement and current-frame temporaries; bytes are diagnostic, not a leak/net-balance metric"
    );
    println!(
        "pattern root plans repeats alloc zeroed realloc dealloc alloc_bytes realloc_old realloc_new dealloc_bytes"
    );

    for root in [DrawDemandRoot::Both, DrawDemandRoot::None] {
        for pattern in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ] {
            for plan_count in [32, 64, 128] {
                run_scenario(pattern, root, plan_count, 1);
            }
        }
    }
    for repetitions in [2, 4] {
        run_scenario(
            DrawDemandDependencyPattern::Shared,
            DrawDemandRoot::Both,
            64,
            repetitions,
        );
    }

    println!("producer_fan_in");
    println!(
        "pattern root plans alloc zeroed realloc dealloc alloc_bytes realloc_old realloc_new dealloc_bytes"
    );
    for root in [DrawDemandRoot::Both, DrawDemandRoot::None] {
        for pattern in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ] {
            for plan_count in [32, 64, 128] {
                run_producer_fan_in_scenario(pattern, root, plan_count);
            }
        }
    }

    println!("high_fan_out");
    println!(
        "root consumers alloc zeroed realloc dealloc alloc_bytes realloc_old realloc_new dealloc_bytes"
    );
    for root in [HighFanOutRoot::AllConsumers, HighFanOutRoot::None] {
        for consumer_count in [4, 8, 16, 64, 128] {
            run_high_fan_out_scenario(root, consumer_count);
        }
    }

    println!(
        "recorder_lifecycle live alloc zeroed realloc dealloc alloc_bytes realloc_old realloc_new dealloc_bytes"
    );
    for live_passes in [0, 32, 64, 128] {
        run_recorder_scenario(live_passes);
    }

    println!(
        "scene_pipeline_warm plans draws_per_plan vertex_bindings alloc zeroed realloc dealloc alloc_bytes realloc_old realloc_new dealloc_bytes"
    );
    run_scene_pipeline_warm_scenario();
}

fn verify_counter_is_live() {
    let ((), stats) = measure(|| {
        let mut values = Vec::with_capacity(1);
        values.push(1_u64);
        values.reserve_exact(1024);
        black_box(values);
    });
    assert!(
        stats.alloc_calls + stats.alloc_zeroed_calls + stats.realloc_calls > 0,
        "allocation counter self-test observed no allocation"
    );
    assert!(
        stats.dealloc_calls > 0,
        "allocation counter self-test observed no deallocation"
    );
}

fn run_scenario(
    pattern: DrawDemandDependencyPattern,
    root: DrawDemandRoot,
    plan_count: usize,
    demand_repetitions: usize,
) {
    let spec = DrawDemandBenchSpec {
        plan_count,
        textures_per_plan: 8,
        dependency_pattern: pattern,
        texture_topology: DrawDemandTextureTopology::ResidentInputs,
        demand_repetitions,
        compile_topology: true,
        root,
    };
    let mut enabled =
        DrawDemandBenchHarness::new_with_diagnostics(RdgBenchDiagnosticsMode::Enabled);
    let mut disabled =
        DrawDemandBenchHarness::new_with_diagnostics(RdgBenchDiagnosticsMode::Disabled);
    let (enabled_token, enabled_stats) = measure_plateau(&mut enabled, spec);
    let (disabled_token, disabled_stats) = measure_plateau(&mut disabled, spec);

    assert_eq!(
        enabled_token, disabled_token,
        "diagnostics A/B routes produced different RDG topology"
    );
    assert_eq!(
        enabled_stats, disabled_stats,
        "diagnostics bookkeeping changed steady-state allocator traffic"
    );
    assert_eq!(
        enabled_stats,
        AllocationStats::default(),
        "resident SceneSubmission workload regressed from its zero-allocation plateau"
    );
    let expected_passes = if root == DrawDemandRoot::None { 3 } else { 4 };
    let expected_live = match root {
        DrawDemandRoot::None => 0,
        DrawDemandRoot::D2 | DrawDemandRoot::Cube => 3,
        DrawDemandRoot::Both => 4,
    };
    assert_eq!(enabled_token.registered_pass_count, expected_passes);
    assert_eq!(enabled_token.compiled_pass_count, expected_live);
    assert_eq!(
        enabled_token.d2_demand_count + enabled_token.cube_demand_count,
        plan_count
    );

    println!(
        "{} {} {} {} {} {} {} {} {} {} {} {}",
        pattern.label(),
        root_label(root),
        plan_count,
        demand_repetitions,
        enabled_stats.alloc_calls,
        enabled_stats.alloc_zeroed_calls,
        enabled_stats.realloc_calls,
        enabled_stats.dealloc_calls,
        enabled_stats.allocated_bytes,
        enabled_stats.reallocated_old_bytes,
        enabled_stats.reallocated_new_bytes,
        enabled_stats.deallocated_bytes,
    );
}

fn run_producer_fan_in_scenario(
    pattern: DrawDemandDependencyPattern,
    root: DrawDemandRoot,
    plan_count: usize,
) {
    let resident_spec = DrawDemandBenchSpec {
        plan_count,
        textures_per_plan: 8,
        dependency_pattern: pattern,
        texture_topology: DrawDemandTextureTopology::ResidentInputs,
        demand_repetitions: 1,
        compile_topology: true,
        root,
    };
    let produced_spec = DrawDemandBenchSpec {
        texture_topology: DrawDemandTextureTopology::PerTextureProducer,
        ..resident_spec
    };

    // Each topology owns and warms an independent high-water-mark store. The
    // measured equality therefore checks steady churn, not retained capacity.
    let mut resident = DrawDemandBenchHarness::new();
    let mut produced = DrawDemandBenchHarness::new();
    let (resident_token, resident_stats) = measure_plateau(&mut resident, resident_spec);
    let (produced_token, produced_stats) = measure_plateau(&mut produced, produced_spec);

    assert_eq!(
        produced_stats, resident_stats,
        "real texture producers added steady-state allocation churn"
    );
    assert_eq!(
        produced_stats,
        AllocationStats::default(),
        "producer-heavy SceneSubmission workload regressed from its zero-allocation plateau"
    );
    assert_eq!(
        produced_token.registered_resource_count, resident_token.registered_resource_count,
        "producer topology changed the logical resource count"
    );
    assert_eq!(produced_token.d2_read_count, resident_token.d2_read_count);
    assert_eq!(
        produced_token.cube_read_count,
        resident_token.cube_read_count
    );
    assert_eq!(
        produced_token.d2_demand_count,
        resident_token.d2_demand_count
    );
    assert_eq!(
        produced_token.cube_demand_count,
        resident_token.cube_demand_count
    );

    let d2_plan_count = plan_count / 4;
    let cube_plan_count = plan_count - d2_plan_count;
    let producer_count = match pattern {
        DrawDemandDependencyPattern::Shared => 8,
        DrawDemandDependencyPattern::Disjoint => plan_count * 8,
    };
    let d2_texture_count = match pattern {
        DrawDemandDependencyPattern::Shared => 8,
        DrawDemandDependencyPattern::Disjoint => d2_plan_count * 8,
    };
    let cube_texture_count = match pattern {
        DrawDemandDependencyPattern::Shared => 8,
        DrawDemandDependencyPattern::Disjoint => cube_plan_count * 8,
    };
    let root_pass_count = usize::from(root != DrawDemandRoot::None);
    assert_eq!(
        produced_token.registered_pass_count,
        producer_count + 3 + root_pass_count
    );
    assert_eq!(
        produced_token.d2_physical_dependency_count,
        d2_texture_count + 1
    );
    assert_eq!(
        produced_token.cube_physical_dependency_count,
        cube_texture_count + 1
    );
    assert_eq!(
        produced_token.root_physical_dependency_count,
        match root {
            DrawDemandRoot::None => 0,
            DrawDemandRoot::D2 | DrawDemandRoot::Cube => 1,
            DrawDemandRoot::Both => 2,
        }
    );
    let expected_live = match root {
        DrawDemandRoot::None => 0,
        DrawDemandRoot::D2 => 3 + d2_texture_count,
        DrawDemandRoot::Cube => 3 + cube_texture_count,
        DrawDemandRoot::Both => 4 + producer_count,
    };
    assert_eq!(produced_token.compiled_pass_count, expected_live);

    println!(
        "{} {} {} {} {} {} {} {} {} {} {}",
        pattern.label(),
        root_label(root),
        plan_count,
        produced_stats.alloc_calls,
        produced_stats.alloc_zeroed_calls,
        produced_stats.realloc_calls,
        produced_stats.dealloc_calls,
        produced_stats.allocated_bytes,
        produced_stats.reallocated_old_bytes,
        produced_stats.reallocated_new_bytes,
        produced_stats.deallocated_bytes,
    );
}

fn measure_plateau(
    harness: &mut DrawDemandBenchHarness,
    spec: DrawDemandBenchSpec,
) -> (
    myth_render::graph::bench_support::DrawDemandRunToken,
    AllocationStats,
) {
    for _ in 0..3 {
        black_box(harness.run_iteration(spec));
    }

    let mut baseline = None;
    let mut token = None;
    for _ in 0..3 {
        let (current_token, stats) = measure(|| black_box(harness.run_iteration(spec)));
        if let Some(expected) = baseline {
            assert_eq!(stats, expected, "steady allocation counts did not plateau");
        } else {
            baseline = Some(stats);
        }
        token = Some(current_token);
    }

    (
        token.expect("SceneSubmission workload was not sampled"),
        baseline.expect("allocation baseline was not sampled"),
    )
}

fn run_high_fan_out_scenario(root: HighFanOutRoot, consumer_count: usize) {
    let spec = HighFanOutBenchSpec {
        consumer_count,
        root,
    };
    let mut harness = HighFanOutBenchHarness::new();
    let (token, stats) = measure_high_fan_out_plateau(&mut harness, spec);
    let rooted = usize::from(root == HighFanOutRoot::AllConsumers);

    assert_eq!(
        stats,
        AllocationStats::default(),
        "high-fan-out workload regressed from its zero-allocation plateau"
    );

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

    println!(
        "{} {} {} {} {} {} {} {} {} {}",
        root.label(),
        consumer_count,
        stats.alloc_calls,
        stats.alloc_zeroed_calls,
        stats.realloc_calls,
        stats.dealloc_calls,
        stats.allocated_bytes,
        stats.reallocated_old_bytes,
        stats.reallocated_new_bytes,
        stats.deallocated_bytes,
    );
}

fn measure_high_fan_out_plateau(
    harness: &mut HighFanOutBenchHarness,
    spec: HighFanOutBenchSpec,
) -> (HighFanOutRunToken, AllocationStats) {
    for _ in 0..3 {
        black_box(harness.run_iteration(spec));
    }

    let mut baseline = None;
    let mut token = None;
    for _ in 0..3 {
        let (current_token, stats) = measure(|| black_box(harness.run_iteration(spec)));
        if let Some(expected) = baseline {
            assert_eq!(
                stats, expected,
                "high-fan-out steady allocation counts did not plateau"
            );
        } else {
            baseline = Some(stats);
        }
        token = Some(current_token);
    }

    (
        token.expect("high-fan-out workload was not sampled"),
        baseline.expect("high-fan-out allocation baseline was not sampled"),
    )
}

const fn root_label(root: DrawDemandRoot) -> &'static str {
    match root {
        DrawDemandRoot::None => "all_dead",
        DrawDemandRoot::D2 => "d2",
        DrawDemandRoot::Cube => "cube",
        DrawDemandRoot::Both => "live",
    }
}

fn run_recorder_scenario(live_passes: usize) {
    let mut enabled = RdgRecorderLifecycleBenchHarness::new(RdgBenchDiagnosticsMode::Enabled);
    let mut disabled = RdgRecorderLifecycleBenchHarness::new(RdgBenchDiagnosticsMode::Disabled);
    let (enabled_token, enabled_stats) = measure_recorder_plateau(&mut enabled, live_passes);
    let (disabled_token, disabled_stats) = measure_recorder_plateau(&mut disabled, live_passes);

    assert_eq!(
        enabled_token, disabled_token,
        "diagnostics recorder A/B routes produced different work tokens"
    );
    assert_eq!(
        enabled_stats, disabled_stats,
        "diagnostics recorder changed steady-state allocator traffic"
    );

    println!(
        "recorder_lifecycle {} {} {} {} {} {} {} {} {}",
        live_passes,
        enabled_stats.alloc_calls,
        enabled_stats.alloc_zeroed_calls,
        enabled_stats.realloc_calls,
        enabled_stats.dealloc_calls,
        enabled_stats.allocated_bytes,
        enabled_stats.reallocated_old_bytes,
        enabled_stats.reallocated_new_bytes,
        enabled_stats.deallocated_bytes,
    );
}

fn measure_recorder_plateau(
    harness: &mut RdgRecorderLifecycleBenchHarness,
    live_passes: usize,
) -> (RdgRecorderLifecycleToken, AllocationStats) {
    for _ in 0..3 {
        black_box(harness.run_iteration(live_passes));
    }

    let mut baseline = None;
    let mut token = None;
    for _ in 0..3 {
        let (current_token, stats) = measure(|| black_box(harness.run_iteration(live_passes)));
        if let Some(expected) = baseline {
            assert_eq!(stats, expected, "steady allocation counts did not plateau");
        } else {
            baseline = Some(stats);
        }
        token = Some(current_token);
    }

    (
        token.expect("recorder lifecycle workload was not sampled"),
        baseline.expect("recorder lifecycle allocation baseline was not sampled"),
    )
}

fn run_scene_pipeline_warm_scenario() {
    let spec = ScenePipelineWarmBenchSpec {
        plan_count: 8,
        draws_per_plan: 64,
        pipeline_variant_count: 16,
        vertex_bindings_per_draw: 4,
    };
    let mut harness = ScenePipelineWarmBenchHarness::new();
    let (token, stats) = measure_scene_pipeline_warm_plateau(&mut harness, spec);

    assert_eq!(
        stats,
        AllocationStats::default(),
        "warm M6 Gather/Plan/Finalize regressed from its zero-allocation plateau"
    );
    assert_eq!(token.plan_count, spec.plan_count);
    assert_eq!(token.command_count, spec.plan_count * spec.draws_per_plan);
    assert_eq!(
        token.vertex_binding_count,
        token.command_count * spec.vertex_bindings_per_draw
    );
    assert_eq!(
        token.exact_pipeline_request_count,
        spec.plan_count * spec.pipeline_variant_count
    );
    assert_ne!(token.checksum, 0);

    println!(
        "scene_pipeline_warm {} {} {} {} {} {} {} {} {} {} {}",
        spec.plan_count,
        spec.draws_per_plan,
        spec.vertex_bindings_per_draw,
        stats.alloc_calls,
        stats.alloc_zeroed_calls,
        stats.realloc_calls,
        stats.dealloc_calls,
        stats.allocated_bytes,
        stats.reallocated_old_bytes,
        stats.reallocated_new_bytes,
        stats.deallocated_bytes,
    );
}

fn measure_scene_pipeline_warm_plateau(
    harness: &mut ScenePipelineWarmBenchHarness,
    spec: ScenePipelineWarmBenchSpec,
) -> (ScenePipelineWarmRunToken, AllocationStats) {
    for _ in 0..3 {
        black_box(harness.run_iteration(spec));
    }

    let mut baseline = None;
    let mut token = None;
    for _ in 0..3 {
        let (current_token, stats) = measure(|| black_box(harness.run_iteration(spec)));
        if let Some(expected) = baseline {
            assert_eq!(
                stats, expected,
                "warm scene-pipeline allocation counts did not plateau"
            );
        } else {
            baseline = Some(stats);
        }
        token = Some(current_token);
    }

    (
        token.expect("warm scene-pipeline workload was not sampled"),
        baseline.expect("warm scene-pipeline allocation baseline was not sampled"),
    )
}
