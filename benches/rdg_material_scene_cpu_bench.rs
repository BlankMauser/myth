//! M6.5 Helmet-like MaterialPlan / AssetTexture / ScenePipeline CPU guard.
//!
//! The stages are cumulative production prefixes. Comparing adjacent medians
//! isolates one phase without adding timer reads to the renderer hot path:
//!
//! - `01 - 00`: Material owner capture, texture visitation and request interning.
//! - `02 - 01`: exact Texture -> Image -> Texture batch capture/revalidation.
//! - `03 - 02`: Material/texture final validation and topology recipes.
//! - `04 - 03`: warm ScenePipeline Gather -> Plan -> immutable stream.
//!
//! The five-texture case approximates the DamagedHelmet PBR material. The
//! untextured PhysicalMaterial is a control arm for fixed transaction costs,
//! while twelve textures guards the flat dependency arena beyond the former
//! eight-entry inline boundary.
//!
//! Run with:
//! `cargo bench --bench rdg_material_scene_cpu_bench --features rdg_bench_support`

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use myth_render::graph::bench_support::{
    MaterialSceneCpuBenchHarness, MaterialSceneCpuBenchStage, ScenePipelineWarmBenchHarness,
    ScenePipelineWarmBenchSpec,
};

const TEXTURE_COUNTS: &[usize] = &[0, 5, 12];
const STAGES: &[MaterialSceneCpuBenchStage] = &[
    MaterialSceneCpuBenchStage::GeometryCapture,
    MaterialSceneCpuBenchStage::MaterialCapture,
    MaterialSceneCpuBenchStage::AssetTextureCapture,
    MaterialSceneCpuBenchStage::MaterialTextureFinalize,
    MaterialSceneCpuBenchStage::WarmScenePipeline,
];

#[derive(Clone, Copy)]
struct ScenePipelineScalingCase {
    label: &'static str,
    spec: ScenePipelineWarmBenchSpec,
}

/// Stable post-DPE live-stream shapes. The first case mirrors the fixed Helmet
/// probe's one visible source feeding main/prepass work. The second exercises a
/// modest material-diversity workload, while the third keeps every source live
/// to guard the positive-DPE, high-source throughput slope.
const SCENE_PIPELINE_SCALING_CASES: &[ScenePipelineScalingCase] = &[
    ScenePipelineScalingCase {
        label: "helmet_like_live",
        spec: ScenePipelineWarmBenchSpec {
            plan_count: 2,
            draws_per_plan: 1,
            pipeline_variant_count: 1,
            vertex_bindings_per_draw: 4,
        },
    },
    ScenePipelineScalingCase {
        label: "small_many_material_live",
        spec: ScenePipelineWarmBenchSpec {
            plan_count: 2,
            draws_per_plan: 128,
            pipeline_variant_count: 64,
            vertex_bindings_per_draw: 4,
        },
    },
    ScenePipelineScalingCase {
        label: "dpe_positive_high_sources",
        spec: ScenePipelineWarmBenchSpec {
            plan_count: 4,
            draws_per_plan: 2_048,
            pipeline_variant_count: 256,
            vertex_bindings_per_draw: 4,
        },
    },
];

fn bench_material_scene_cpu_prefixes(c: &mut Criterion) {
    let mut group = c.benchmark_group("M65_MaterialSceneCpu_Cumulative");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for &texture_count in TEXTURE_COUNTS {
        group.throughput(Throughput::Elements(1));
        let fixture = match texture_count {
            0 => "physical_untextured",
            5 => "helmet_like_5_textures",
            12 => "wide_12_textures",
            _ => unreachable!("unregistered M6.5 material benchmark shape"),
        };

        for &stage in STAGES {
            group.bench_with_input(
                BenchmarkId::new(format!("{fixture}/{}", stage.label()), texture_count),
                &(texture_count, stage),
                |b, &(texture_count, stage)| {
                    let mut harness = MaterialSceneCpuBenchHarness::new(texture_count);
                    // Establish retained capacities before Criterion samples.
                    black_box(harness.run_iteration(MaterialSceneCpuBenchStage::WarmScenePipeline));
                    b.iter(|| black_box(harness.run_iteration(stage)));
                },
            );
        }
    }

    group.finish();
}

fn bench_scene_pipeline_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("M65_ScenePipeline_Scaling");
    // Keep this regression guard practical for normal `cargo bench` runs. The
    // high-source case is large enough for a stable wall-time sample without
    // requiring long Criterion defaults.
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));

    for &case in SCENE_PIPELINE_SCALING_CASES {
        let command_count = case
            .spec
            .plan_count
            .checked_mul(case.spec.draws_per_plan)
            .expect("scene-pipeline benchmark command count overflowed");
        group.throughput(Throughput::Elements(
            u64::try_from(command_count).expect("scene-pipeline command count exceeds u64"),
        ));
        group.bench_with_input(
            BenchmarkId::new(case.label, command_count),
            &case.spec,
            |b, &spec| {
                // One retained fixture per benchmark case. The untimed first
                // iteration establishes every planner/stream high-water mark;
                // measured iterations only clear and reuse those allocations.
                let mut harness = ScenePipelineWarmBenchHarness::new();
                let warm = harness.run_iteration(spec);
                assert_eq!(warm.plan_count, spec.plan_count);
                assert_eq!(warm.command_count, command_count);
                assert_eq!(
                    warm.vertex_binding_count,
                    command_count * spec.vertex_bindings_per_draw
                );
                black_box(warm);

                b.iter(|| black_box(harness.run_iteration(spec)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_material_scene_cpu_prefixes,
    bench_scene_pipeline_scaling,
);
criterion_main!(benches);
