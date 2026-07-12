//! Same-binary A/B benchmark for RDG diagnostics bookkeeping.
//!
//! Both arms execute the exact crate-private DrawPlan demand topology. The
//! control arm skips only diagnostics attempt creation, counter updates, and
//! terminal publication.
//!
//! The full-graph arms are independent Criterion cases and therefore serve as
//! a regression/trend smoke test, not a causal nanosecond subtraction. The
//! recorder lifecycle group isolates the per-pass counter slope; allocator
//! equivalence is enforced separately by `rdg_allocation_probe`.
//!
//! Run with:
//! `cargo bench --bench rdg_diagnostics_ab --features rdg_bench_support`

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use myth_render::graph::bench_support::{
    DrawDemandBenchHarness, DrawDemandBenchSpec, DrawDemandDependencyPattern, DrawDemandRoot,
    DrawDemandTextureTopology, RdgBenchDiagnosticsMode, RdgRecorderLifecycleBenchHarness,
};

const PLAN_COUNTS: &[usize] = &[32, 64, 128];
const TEXTURES_PER_PLAN: usize = 8;

fn configure(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
}

fn bench_rdg_diagnostics_ab(c: &mut Criterion) {
    for (root, root_label) in [
        (DrawDemandRoot::Both, "live"),
        (DrawDemandRoot::None, "all_dead"),
    ] {
        let mut group = c.benchmark_group(format!("RdgDiagnostics_AB/{root_label}"));
        configure(&mut group);

        for (pattern_index, pattern) in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ]
        .into_iter()
        .enumerate()
        {
            for (plan_index, &plan_count) in PLAN_COUNTS.iter().enumerate() {
                let spec = DrawDemandBenchSpec {
                    plan_count,
                    textures_per_plan: TEXTURES_PER_PLAN,
                    dependency_pattern: pattern,
                    texture_topology: DrawDemandTextureTopology::ResidentInputs,
                    demand_repetitions: 1,
                    compile_topology: true,
                    root,
                };
                assert_ab_topology_equivalent(spec);

                group.throughput(Throughput::Elements(
                    (plan_count * TEXTURES_PER_PLAN) as u64,
                ));
                let enabled_first = (pattern_index + plan_index) % 2 == 0;
                let modes = if enabled_first {
                    [
                        RdgBenchDiagnosticsMode::Enabled,
                        RdgBenchDiagnosticsMode::Disabled,
                    ]
                } else {
                    [
                        RdgBenchDiagnosticsMode::Disabled,
                        RdgBenchDiagnosticsMode::Enabled,
                    ]
                };

                for mode in modes {
                    let id = BenchmarkId::new(
                        format!("{}/{}", pattern.label(), mode.label()),
                        plan_count,
                    );
                    group.bench_with_input(id, &(spec, mode), |b, &(spec, mode)| {
                        let mut harness = DrawDemandBenchHarness::new_with_diagnostics(mode);
                        for _ in 0..3 {
                            black_box(harness.run_iteration(spec));
                        }
                        b.iter(|| black_box(harness.run_iteration(spec)));
                    });
                }
            }
        }

        group.finish();
    }
}

fn assert_ab_topology_equivalent(spec: DrawDemandBenchSpec) {
    let mut enabled =
        DrawDemandBenchHarness::new_with_diagnostics(RdgBenchDiagnosticsMode::Enabled);
    let mut disabled =
        DrawDemandBenchHarness::new_with_diagnostics(RdgBenchDiagnosticsMode::Disabled);

    let enabled_token = enabled.run_iteration(spec);
    let disabled_token = disabled.run_iteration(spec);
    assert_eq!(
        enabled_token, disabled_token,
        "diagnostics A/B routes produced different RDG topology"
    );
}

fn bench_recorder_lifecycle_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("RdgDiagnostics_AB/recorder_lifecycle");
    configure(&mut group);

    for (index, live_passes) in [0, 32, 64, 128].into_iter().enumerate() {
        assert_recorder_tokens_equivalent(live_passes);
        let modes = if index % 2 == 0 {
            [
                RdgBenchDiagnosticsMode::Enabled,
                RdgBenchDiagnosticsMode::Disabled,
            ]
        } else {
            [
                RdgBenchDiagnosticsMode::Disabled,
                RdgBenchDiagnosticsMode::Enabled,
            ]
        };

        for mode in modes {
            group.bench_with_input(
                BenchmarkId::new(mode.label(), live_passes),
                &(live_passes, mode),
                |b, &(live_passes, mode)| {
                    let mut harness = RdgRecorderLifecycleBenchHarness::new(mode);
                    for _ in 0..3 {
                        black_box(harness.run_iteration(live_passes));
                    }
                    b.iter(|| black_box(harness.run_iteration(live_passes)));
                },
            );
        }
    }

    group.finish();
}

fn assert_recorder_tokens_equivalent(live_passes: usize) {
    let mut enabled = RdgRecorderLifecycleBenchHarness::new(RdgBenchDiagnosticsMode::Enabled);
    let mut disabled = RdgRecorderLifecycleBenchHarness::new(RdgBenchDiagnosticsMode::Disabled);
    assert_eq!(
        enabled.run_iteration(live_passes),
        disabled.run_iteration(live_passes),
        "diagnostics recorder A/B routes produced different work tokens"
    );
}

criterion_group!(
    benches,
    bench_rdg_diagnostics_ab,
    bench_recorder_lifecycle_ab
);
criterion_main!(benches);
