//! M7.9 persistent RenderWorld / MutationJournal CPU regression guard.
//!
//! The fixture uses real Scene nodes, Mesh components, AssetServer entries and
//! `ExtractedScene::extract_into`. Only the extraction-boundary synchronization
//! is timed. Scene mutation authoring, correctness validation, and retained
//! RenderWorld consumer acknowledgement occur outside the accumulated duration.
//!
//! Run with:
//! `cargo bench --bench rdg_render_world_bench --features rdg_bench_support`

use std::{hint::black_box, time::Duration, time::Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use myth_render::graph::bench_support::{RenderWorldBenchHarness, RenderWorldBenchWorkload};

const STATIC_SCENE_SIZES: &[usize] = &[1, 1_000, 10_000, 100_000];
const MUTATION_SCENE_SIZE: usize = 10_000;
const MUTATION_WORKLOADS: &[RenderWorldBenchWorkload] = &[
    RenderWorldBenchWorkload::StaticWarm,
    RenderWorldBenchWorkload::SingleTransform,
    RenderWorldBenchWorkload::OnePercentTransforms,
    RenderWorldBenchWorkload::AllTransforms,
];
const BOUNDARY_WORKLOADS: &[RenderWorldBenchWorkload] = &[
    RenderWorldBenchWorkload::TopologyRebuild,
    RenderWorldBenchWorkload::SceneAlternation,
];

fn timed_sync_iterations(
    harness: &mut RenderWorldBenchHarness,
    workload: RenderWorldBenchWorkload,
    iterations: u64,
) -> Duration {
    let mut measured = Duration::ZERO;
    for _ in 0..iterations {
        // Mutation authoring models simulation/editor output and is explicitly
        // not part of the M7.9 retained consumer cost under test.
        harness.prepare_iteration(workload);

        let started = Instant::now();
        harness.sync();
        measured += started.elapsed();

        // Exact delta validation and acknowledgement keep every sample on the
        // same steady frontier without charging bookkeeping to the sync.
        black_box(harness.finish_iteration());
    }
    measured
}

fn bench_static_warm_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("M79_RenderWorld_StaticWarm_Scaling");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for &item_count in STATIC_SCENE_SIZES {
        group.bench_with_input(
            BenchmarkId::new("scene_sources", item_count),
            &item_count,
            |b, &item_count| {
                let mut harness = RenderWorldBenchHarness::new(item_count);
                b.iter_custom(|iterations| {
                    timed_sync_iterations(
                        &mut harness,
                        RenderWorldBenchWorkload::StaticWarm,
                        iterations,
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_transform_mutation_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("M79_RenderWorld_TransformMutationDensity");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for &workload in MUTATION_WORKLOADS {
        let mutation_count = workload.mutation_count(MUTATION_SCENE_SIZE);
        group.bench_with_input(
            BenchmarkId::new(workload.label(), mutation_count),
            &workload,
            |b, &workload| {
                let mut harness = RenderWorldBenchHarness::new(MUTATION_SCENE_SIZE);
                b.iter_custom(|iterations| {
                    timed_sync_iterations(&mut harness, workload, iterations)
                });
            },
        );
    }

    group.finish();
}

fn bench_rebuild_and_residency_boundaries(c: &mut Criterion) {
    let mut group = c.benchmark_group("M79_RenderWorld_RebuildAndResidencyBoundaries");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for &workload in BOUNDARY_WORKLOADS {
        group.bench_with_input(
            BenchmarkId::new(workload.label(), MUTATION_SCENE_SIZE),
            &workload,
            |b, &workload| {
                let mut harness = if workload == RenderWorldBenchWorkload::SceneAlternation {
                    RenderWorldBenchHarness::new_with_scene_alternation(MUTATION_SCENE_SIZE)
                } else {
                    RenderWorldBenchHarness::new(MUTATION_SCENE_SIZE)
                };
                b.iter_custom(|iterations| {
                    timed_sync_iterations(&mut harness, workload, iterations)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_static_warm_scaling,
    bench_transform_mutation_density,
    bench_rebuild_and_residency_boundaries,
);
criterion_main!(benches);
