//! Deterministic M7.5 CPU/GPU-driven policy and planning comparison.
//!
//! This benchmark is deliberately headless. It times the production Auto
//! selector and retained CPU planning proxies for the CPU-direct, adaptive,
//! and forced GPU-driven routes. It does not claim to replace adapter-specific
//! GPU timestamp queries; instead it provides a reproducible CI/control matrix
//! for policy overhead and crossover calibration.
//! Visibility percentage is intentionally absent: the production synopsis has
//! no post-occlusion visibility input, so inventing one here would calibrate a
//! fictitious policy. Visibility/cull efficiency belongs in the adapter GPU
//! timestamp suite that consumes this CPU control matrix.
//!
//! Run with:
//! `cargo bench --bench rdg_gpu_driven_policy_bench --features rdg_bench_support`

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use myth_render::graph::bench_support::{
    GpuDrivenMatrixBenchHarness, GpuDrivenMatrixBenchSpec, GpuDrivenMatrixMode,
};
use myth_render::settings::RenderPath;

const ITEM_COUNTS: &[usize] = &[1, 10, 100, 1_000, 10_000, 100_000];
const MODES: &[GpuDrivenMatrixMode] = &[
    GpuDrivenMatrixMode::Off,
    GpuDrivenMatrixMode::Auto,
    GpuDrivenMatrixMode::Required,
];

fn bench_gpu_driven_nb_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("M75_GpuDriven_NB_Matrix");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(500));

    for &item_count in ITEM_COUNTS {
        for bin_count in valid_bin_counts(item_count) {
            let spec = GpuDrivenMatrixBenchSpec::main_only(item_count, bin_count);
            group.throughput(Throughput::Elements(item_count as u64));
            for &mode in MODES {
                group.bench_with_input(
                    BenchmarkId::new(format!("N{item_count}_B{bin_count}"), mode.label()),
                    &(spec, mode),
                    |b, &(spec, mode)| {
                        let mut harness = GpuDrivenMatrixBenchHarness::new(spec, mode);
                        let stable = harness.stabilize();
                        assert_eq!(
                            stable.cpu_main_source_count + stable.gpu_main_source_count,
                            spec.item_count
                        );
                        black_box(stable);
                        b.iter(|| black_box(harness.run_iteration()));
                    },
                );
            }
        }
    }

    group.finish();
}

fn bench_gpu_driven_shadow_and_dirty_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("M75_GpuDriven_ShadowDirty_Curated");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(500));

    let cases = [
        (
            "basic_720p_tier1_span1_shadow0_dirty0",
            GpuDrivenMatrixBenchSpec {
                item_count: 10_000,
                bin_count: 64,
                render_path: RenderPath::BasicForward,
                viewport_width: 1280,
                viewport_height: 720,
                device_tier: 1,
                slot_span_multiplier: 1,
                shadow_2d_view_count: 0,
                shadow_cube_view_count: 0,
                dirty_percent: 0,
            },
        ),
        (
            "high_1440p_tier2_span1_shadow1_dirty1",
            GpuDrivenMatrixBenchSpec {
                item_count: 10_000,
                bin_count: 64,
                render_path: RenderPath::HighFidelity,
                viewport_width: 2560,
                viewport_height: 1440,
                device_tier: 2,
                slot_span_multiplier: 1,
                shadow_2d_view_count: 1,
                shadow_cube_view_count: 0,
                dirty_percent: 1,
            },
        ),
        (
            "high_4k_tier2_span4_shadow4_dirty10",
            GpuDrivenMatrixBenchSpec {
                item_count: 100_000,
                bin_count: 64,
                render_path: RenderPath::HighFidelity,
                viewport_width: 3840,
                viewport_height: 2160,
                device_tier: 2,
                slot_span_multiplier: 4,
                shadow_2d_view_count: 0,
                shadow_cube_view_count: 4,
                dirty_percent: 10,
            },
        ),
        (
            "high_4k_tier1_span16_shadow16_dirty100",
            GpuDrivenMatrixBenchSpec {
                item_count: 100_000,
                bin_count: 64,
                render_path: RenderPath::HighFidelity,
                viewport_width: 3840,
                viewport_height: 2160,
                device_tier: 1,
                slot_span_multiplier: 16,
                shadow_2d_view_count: 4,
                shadow_cube_view_count: 12,
                dirty_percent: 100,
            },
        ),
        (
            "basic_1440p_tier2_span1_shadow64_dirty0",
            GpuDrivenMatrixBenchSpec {
                item_count: 100_000,
                bin_count: 64,
                render_path: RenderPath::BasicForward,
                viewport_width: 2560,
                viewport_height: 1440,
                device_tier: 2,
                slot_span_multiplier: 1,
                shadow_2d_view_count: 16,
                shadow_cube_view_count: 48,
                dirty_percent: 0,
            },
        ),
    ];

    for (label, spec) in cases {
        group.throughput(Throughput::Elements(spec.item_count as u64));
        for &mode in MODES {
            group.bench_with_input(
                BenchmarkId::new(label, mode.label()),
                &(spec, mode),
                |b, &(spec, mode)| {
                    let mut harness = GpuDrivenMatrixBenchHarness::new(spec, mode);
                    black_box(harness.stabilize());
                    b.iter(|| black_box(harness.run_iteration()));
                },
            );
        }
    }

    group.finish();
}

fn valid_bin_counts(item_count: usize) -> impl Iterator<Item = usize> {
    let candidates = [1, 4, 64, item_count];
    candidates
        .into_iter()
        .enumerate()
        .filter_map(move |(index, bin_count)| {
            (bin_count <= item_count && !candidates[..index].contains(&bin_count))
                .then_some(bin_count)
        })
}

criterion_group!(
    benches,
    bench_gpu_driven_nb_matrix,
    bench_gpu_driven_shadow_and_dirty_matrix,
);
criterion_main!(benches);
