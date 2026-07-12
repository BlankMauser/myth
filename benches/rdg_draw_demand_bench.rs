//! Exact DrawPlan-demand scaling benchmarks.
//!
//! Run with:
//! `cargo bench --bench rdg_draw_demand_bench --features rdg_bench_support`

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use myth_render::graph::bench_support::{
    DrawDemandBenchHarness, DrawDemandBenchSpec, DrawDemandDependencyPattern, DrawDemandRoot,
    DrawDemandTextureTopology,
};

const PLAN_COUNTS: &[usize] = &[32, 64, 128];
const TEXTURES_PER_PLAN: usize = 8;
const DUPLICATE_PLAN_COUNT: usize = 64;

fn configure(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));
}

fn bench_draw_demand_build_only(c: &mut Criterion) {
    benchmark_scaling_group(c, "DrawDemand_BuildOnly", false);
}

fn bench_draw_demand_build_and_compile(c: &mut Criterion) {
    benchmark_scaling_group(c, "DrawDemand_BuildAndCompile", true);
}

fn benchmark_scaling_group(c: &mut Criterion, name: &str, compile_topology: bool) {
    let mut group = c.benchmark_group(name);
    configure(&mut group);

    for pattern in [
        DrawDemandDependencyPattern::Shared,
        DrawDemandDependencyPattern::Disjoint,
    ] {
        for &plan_count in PLAN_COUNTS {
            group.throughput(Throughput::Elements(
                (plan_count * TEXTURES_PER_PLAN) as u64,
            ));
            let spec = DrawDemandBenchSpec {
                plan_count,
                textures_per_plan: TEXTURES_PER_PLAN,
                dependency_pattern: pattern,
                texture_topology: DrawDemandTextureTopology::ResidentInputs,
                demand_repetitions: 1,
                compile_topology,
                root: DrawDemandRoot::Both,
            };
            group.bench_with_input(
                BenchmarkId::new(pattern.label(), plan_count),
                &spec,
                |b, &spec| {
                    let mut harness = DrawDemandBenchHarness::new();
                    b.iter(|| black_box(harness.run_iteration(spec)));
                },
            );
        }
    }

    group.finish();
}

fn bench_duplicate_draw_demands(c: &mut Criterion) {
    let mut group = c.benchmark_group("DrawDemand_DuplicateDemand");
    configure(&mut group);

    for repetitions in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements(
            (DUPLICATE_PLAN_COUNT * repetitions * TEXTURES_PER_PLAN) as u64,
        ));
        let spec = DrawDemandBenchSpec {
            plan_count: DUPLICATE_PLAN_COUNT,
            textures_per_plan: TEXTURES_PER_PLAN,
            dependency_pattern: DrawDemandDependencyPattern::Shared,
            texture_topology: DrawDemandTextureTopology::ResidentInputs,
            demand_repetitions: repetitions,
            compile_topology: true,
            root: DrawDemandRoot::Both,
        };
        group.bench_with_input(
            BenchmarkId::new("repetitions", repetitions),
            &spec,
            |b, &spec| {
                let mut harness = DrawDemandBenchHarness::new();
                b.iter(|| black_box(harness.run_iteration(spec)));
            },
        );
    }

    group.finish();
}

fn bench_producer_fan_in(c: &mut Criterion) {
    for (root, root_label) in [
        (DrawDemandRoot::Both, "live"),
        (DrawDemandRoot::None, "all_dead"),
    ] {
        let mut group = c.benchmark_group(format!(
            "DrawDemand_ProducerFanIn_BuildAndCompile/{root_label}"
        ));
        configure(&mut group);

        for pattern in [
            DrawDemandDependencyPattern::Shared,
            DrawDemandDependencyPattern::Disjoint,
        ] {
            for &plan_count in PLAN_COUNTS {
                group.throughput(Throughput::Elements(
                    (plan_count * TEXTURES_PER_PLAN) as u64,
                ));
                let spec = DrawDemandBenchSpec {
                    plan_count,
                    textures_per_plan: TEXTURES_PER_PLAN,
                    dependency_pattern: pattern,
                    texture_topology: DrawDemandTextureTopology::PerTextureProducer,
                    demand_repetitions: 1,
                    compile_topology: true,
                    root,
                };
                group.bench_with_input(
                    BenchmarkId::new(pattern.label(), plan_count),
                    &spec,
                    |b, &spec| {
                        let mut harness = DrawDemandBenchHarness::new();
                        b.iter(|| black_box(harness.run_iteration(spec)));
                    },
                );
            }
        }

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_draw_demand_build_only,
    bench_draw_demand_build_and_compile,
    bench_duplicate_draw_demands,
    bench_producer_fan_in,
);
criterion_main!(benches);
