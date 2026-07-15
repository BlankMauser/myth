//! Interactive M7.5 CPU-direct / Auto / GPU-driven comparison lab.
//!
//! This is intentionally not a gallery example. It constructs deterministic,
//! indexed, low-polygon workloads without external assets so the same presets
//! can be compared across adapters and engine revisions.
//!
//! Run the release build so debug validation does not dominate the comparison:
//! `cargo run --release --example gpu_driven_lab`.
//! Add `--features rdg_bench_support` to include per-row RDG CPU phase means;
//! that build also exposes an explicit, default-off asynchronous GPU timestamp
//! toggle. The default build keeps all profiler timers and accumulators absent.
//!
//! Keyboard recovery/control shortcuts remain active if a `Required` workload
//! fails before it can render the UI:
//! - `1`..`6`: workload presets
//! - `F1`: Off, `F2`: Auto, `F3`: Required when the GPU baseline is available

use std::{any::Any, time::Instant};

use myth::prelude::*;
use myth::renderer::graph::{
    RdgAttemptOutcome, RdgFrameDiagnostics, RdgGpuDrivenFailure, RdgSceneSubmissionFamily,
};
#[cfg(feature = "rdg_bench_support")]
use myth::renderer::graph::{RdgDetailPhase, RdgFramePhase, RdgGpuFrameTimings, RdgGpuPassTiming};
use myth_dev_utils::FpsCounter;
use myth_egui::{UiPass, UiPassNode, egui};
use myth_resources::Key;
use winit::event::WindowEvent;

const TINY_OBJECTS: usize = 16;
const MASSIVE_OBJECTS: usize = 16_384;
const FRAGMENTED_OBJECTS: usize = 4_096;
const DYNAMIC_OBJECTS: usize = 16_384;
const SHADOW_OBJECTS: usize = 8_192;
const HYBRID_SHARED_OBJECTS: usize = 8_192;
const HYBRID_SINGLETON_OBJECTS: usize = 512;
const HYBRID_OBJECTS: usize = HYBRID_SHARED_OBJECTS + HYBRID_SINGLETON_OBJECTS;
const WARMUP_FRAMES: u32 = 120;
const SAMPLE_FRAMES: u32 = 600;

#[cfg(feature = "rdg_bench_support")]
const MATRIX_PHASES: [RdgFramePhase; 7] = [
    RdgFramePhase::Cull,
    RdgFramePhase::BuildGraph,
    RdgFramePhase::CompileGraph,
    RdgFramePhase::MaterializeResources,
    RdgFramePhase::PreparePasses,
    RdgFramePhase::FinalizeDrawPlans,
    RdgFramePhase::RecordCommands,
];

#[cfg(feature = "rdg_bench_support")]
const MATRIX_DETAIL_PHASES: [RdgDetailPhase; 3] = [
    RdgDetailPhase::GpuBackendSelection,
    RdgDetailPhase::GpuSceneCommit,
    RdgDetailPhase::CpuIntentCollection,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LabPreset {
    Tiny,
    MassiveReuse,
    Fragmented,
    DynamicStorm,
    ShadowAmplification,
    HybridCohorts,
}

impl LabPreset {
    const ALL: [Self; 6] = [
        Self::Tiny,
        Self::MassiveReuse,
        Self::Fragmented,
        Self::DynamicStorm,
        Self::ShadowAmplification,
        Self::HybridCohorts,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Tiny => "Tiny",
            Self::MassiveReuse => "MassiveReuse",
            Self::Fragmented => "Fragmented",
            Self::DynamicStorm => "DynamicStorm",
            Self::ShadowAmplification => "ShadowAmplification",
            Self::HybridCohorts => "HybridCohorts",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Tiny => "16 boxes, one geometry/material bin; Auto should stay CPU-direct.",
            Self::MassiveReuse => {
                "16,384 static boxes sharing one bin; the canonical GPU-driven win shape."
            }
            Self::Fragmented => {
                "4,096 boxes with 4,096 material bins; exposes batching and indirect overhead."
            }
            Self::DynamicStorm => {
                "16,384 shared-bin boxes with every transform changed every frame."
            }
            Self::ShadowAmplification => {
                "8,192 shared-bin casters, four CSM views and four point-light cube submissions (24 faces)."
            }
            Self::HybridCohorts => {
                "8,192 shared-bin objects plus 512 singleton material bins; Auto should be Hybrid."
            }
        }
    }

    const fn object_count(self) -> usize {
        match self {
            Self::Tiny => TINY_OBJECTS,
            Self::MassiveReuse => MASSIVE_OBJECTS,
            Self::Fragmented => FRAGMENTED_OBJECTS,
            Self::DynamicStorm => DYNAMIC_OBJECTS,
            Self::ShadowAmplification => SHADOW_OBJECTS,
            Self::HybridCohorts => HYBRID_OBJECTS,
        }
    }

    const fn intended_bin_count(self) -> usize {
        match self {
            Self::Fragmented => FRAGMENTED_OBJECTS,
            Self::HybridCohorts => HYBRID_SINGLETON_OBJECTS + 1,
            _ => 1,
        }
    }

    const fn animates_all_transforms(self) -> bool {
        matches!(self, Self::DynamicStorm)
    }

    const fn uses_shadow_amplification(self) -> bool {
        matches!(self, Self::ShadowAmplification)
    }
}

#[derive(Debug, Clone, Copy)]
struct DynamicObject {
    node: NodeHandle,
    base: Vec3,
    phase_sin: f32,
    phase_cos: f32,
}

#[derive(Debug, Clone, Copy)]
struct MeasurementResult {
    mean_fps: f64,
    mean_frame_ms: f64,
    p50_frame_ms: f64,
    p95_frame_ms: f64,
    p99_frame_ms: f64,
    max_frame_ms: f64,
    #[cfg(feature = "rdg_bench_support")]
    phase_mean_us: [f64; MATRIX_PHASES.len()],
    #[cfg(feature = "rdg_bench_support")]
    detail_mean_us: [f64; MATRIX_DETAIL_PHASES.len()],
    #[cfg(feature = "rdg_bench_support")]
    timing_samples: u32,
}

struct MeasurementWindow {
    warmup_remaining: u32,
    last_update: Option<Instant>,
    samples_ns: Vec<u64>,
    result: Option<MeasurementResult>,
    #[cfg(feature = "rdg_bench_support")]
    phase_elapsed_ns: [u128; MATRIX_PHASES.len()],
    #[cfg(feature = "rdg_bench_support")]
    detail_elapsed_ns: [u128; MATRIX_DETAIL_PHASES.len()],
    #[cfg(feature = "rdg_bench_support")]
    timing_samples: u32,
}

impl MeasurementWindow {
    fn new() -> Self {
        let mut window = Self {
            warmup_remaining: 0,
            last_update: None,
            samples_ns: Vec::with_capacity(SAMPLE_FRAMES as usize),
            result: None,
            #[cfg(feature = "rdg_bench_support")]
            phase_elapsed_ns: [0; MATRIX_PHASES.len()],
            #[cfg(feature = "rdg_bench_support")]
            detail_elapsed_ns: [0; MATRIX_DETAIL_PHASES.len()],
            #[cfg(feature = "rdg_bench_support")]
            timing_samples: 0,
        };
        window.reset();
        window
    }

    fn reset(&mut self) {
        self.warmup_remaining = WARMUP_FRAMES;
        self.last_update = None;
        self.samples_ns.clear();
        self.result = None;
        #[cfg(feature = "rdg_bench_support")]
        {
            self.phase_elapsed_ns.fill(0);
            self.detail_elapsed_ns.fill(0);
            self.timing_samples = 0;
        }
    }

    #[cfg(feature = "rdg_bench_support")]
    fn record_phase_timings(&mut self, diagnostics: &RdgFrameDiagnostics) {
        // `latest_rdg_diagnostics` describes the frame interval about to be
        // sampled below. Warmup and completed windows remain telemetry-free.
        if self.last_update.is_none()
            || self.warmup_remaining != 0
            || self.result.is_some()
            || !diagnostics.queue_submitted()
        {
            return;
        }
        for (index, phase) in MATRIX_PHASES.into_iter().enumerate() {
            self.phase_elapsed_ns[index] = self.phase_elapsed_ns[index]
                .saturating_add(u128::from(diagnostics.phase_elapsed_ns(phase)));
        }
        for (index, phase) in MATRIX_DETAIL_PHASES.into_iter().enumerate() {
            self.detail_elapsed_ns[index] = self.detail_elapsed_ns[index]
                .saturating_add(u128::from(diagnostics.detail_elapsed_ns(phase)));
        }
        self.timing_samples = self.timing_samples.saturating_add(1);
    }

    fn update(&mut self, now: Instant) {
        let Some(previous) = self.last_update.replace(now) else {
            return;
        };
        if self.result.is_some() {
            return;
        }
        if self.warmup_remaining != 0 {
            self.warmup_remaining -= 1;
            return;
        }

        let interval_ns = now.duration_since(previous).as_nanos();
        self.samples_ns.push(
            u64::try_from(interval_ns)
                .expect("one GPU-driven lab frame interval must fit in u64 nanoseconds"),
        );
        if self.samples_ns.len() == SAMPLE_FRAMES as usize {
            let total_ns: u128 = self.samples_ns.iter().map(|&ns| u128::from(ns)).sum();
            let mean_ns = total_ns as f64 / self.samples_ns.len() as f64;
            self.samples_ns.sort_unstable();
            #[cfg(feature = "rdg_bench_support")]
            let timing_divisor = f64::from(self.timing_samples.max(1)) * 1_000.0;
            self.result = Some(MeasurementResult {
                mean_fps: 1_000_000_000.0 / mean_ns,
                mean_frame_ms: mean_ns / 1_000_000.0,
                p50_frame_ms: percentile_ns(&self.samples_ns, 50) as f64 / 1_000_000.0,
                p95_frame_ms: percentile_ns(&self.samples_ns, 95) as f64 / 1_000_000.0,
                p99_frame_ms: percentile_ns(&self.samples_ns, 99) as f64 / 1_000_000.0,
                max_frame_ms: self.samples_ns[self.samples_ns.len() - 1] as f64 / 1_000_000.0,
                #[cfg(feature = "rdg_bench_support")]
                phase_mean_us: self.phase_elapsed_ns.map(|ns| ns as f64 / timing_divisor),
                #[cfg(feature = "rdg_bench_support")]
                detail_mean_us: self.detail_elapsed_ns.map(|ns| ns as f64 / timing_divisor),
                #[cfg(feature = "rdg_bench_support")]
                timing_samples: self.timing_samples,
            });
        }
    }
}

fn percentile_ns(sorted_samples: &[u64], percentile: usize) -> u64 {
    let rank = sorted_samples
        .len()
        .saturating_mul(percentile)
        .div_ceil(100);
    sorted_samples[rank.saturating_sub(1).min(sorted_samples.len() - 1)]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbPhase {
    Idle,
    MeasuringOff,
    MeasuringRequired,
    Complete,
    Failed,
}

struct AbExperiment {
    phase: AbPhase,
    original_mode: GpuDrivenMode,
    off: Option<MeasurementResult>,
    required: Option<MeasurementResult>,
    failure: Option<RdgGpuDrivenFailure>,
}

impl AbExperiment {
    fn new(mode: GpuDrivenMode) -> Self {
        Self {
            phase: AbPhase::Idle,
            original_mode: mode,
            off: None,
            required: None,
            failure: None,
        }
    }

    const fn is_active(&self) -> bool {
        matches!(
            self.phase,
            AbPhase::MeasuringOff | AbPhase::MeasuringRequired
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixPhase {
    Idle,
    Running,
    Complete,
    Cancelled,
}

#[derive(Debug, Clone, Copy)]
struct MatrixResult {
    preset: LabPreset,
    off: MeasurementResult,
    required: Result<MeasurementResult, RdgGpuDrivenFailure>,
}

struct MatrixExperiment {
    phase: MatrixPhase,
    original_preset: LabPreset,
    original_mode: GpuDrivenMode,
    preset_index: usize,
    results: Vec<MatrixResult>,
}

impl MatrixExperiment {
    fn new(preset: LabPreset, mode: GpuDrivenMode) -> Self {
        Self {
            phase: MatrixPhase::Idle,
            original_preset: preset,
            original_mode: mode,
            preset_index: 0,
            results: Vec::with_capacity(LabPreset::ALL.len()),
        }
    }

    const fn is_active(&self) -> bool {
        matches!(self.phase, MatrixPhase::Running)
    }
}

#[derive(Default)]
struct UiActions {
    restart_measurement: bool,
    experiment: Option<ExperimentAction>,
    #[cfg(feature = "rdg_bench_support")]
    gpu_timing_enabled: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExperimentAction {
    StartAb,
    CancelAb,
    StartMatrix,
    CancelMatrix,
}

struct GpuDrivenLab {
    ui_pass: UiPass,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    preset: LabPreset,
    mode: GpuDrivenMode,
    geometry: GeometryHandle,
    materials: Vec<MaterialHandle>,
    workload_root: Option<NodeHandle>,
    workload_lights: Vec<NodeHandle>,
    dynamic_objects: Vec<DynamicObject>,
    time: f32,
    measurement: MeasurementWindow,
    gpu_baseline_available: bool,
    #[cfg(feature = "rdg_bench_support")]
    gpu_timing_supported: bool,
    #[cfg(feature = "rdg_bench_support")]
    gpu_timing_enabled: bool,
    ab: AbExperiment,
    matrix: MatrixExperiment,
}

impl GpuDrivenLab {
    fn rebuild_workload(&mut self, engine: &mut Engine) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if let Some(root) = self.workload_root.take() {
            scene.remove_node(root);
        }
        for light in self.workload_lights.drain(..) {
            scene.remove_node(light);
        }
        self.dynamic_objects.clear();

        let root = scene.add_node(Node::new());
        let object_count = self.preset.object_count();
        let side = (object_count as f32).sqrt().ceil() as usize;
        let spacing = if side >= 100 { 0.22 } else { 0.32 };
        let half_extent = (side.saturating_sub(1) as f32) * spacing * 0.5;
        let cast_shadows = self.preset.uses_shadow_amplification();

        self.dynamic_objects
            .reserve(object_count.saturating_sub(self.dynamic_objects.capacity()));
        for index in 0..object_count {
            let x = index % side;
            let z = index / side;
            let base = Vec3::new(
                x as f32 * spacing - half_extent,
                0.0,
                z as f32 * spacing - half_extent,
            );
            let material = match self.preset {
                LabPreset::Fragmented => self.materials[index],
                LabPreset::HybridCohorts if index >= HYBRID_SHARED_OBJECTS => {
                    self.materials[index - HYBRID_SHARED_OBJECTS + 1]
                }
                _ => self.materials[0],
            };
            let node = scene.add_mesh_to_parent(Mesh::new(self.geometry, material), root);
            scene
                .node(&node)
                .set_position(base.x, base.y, base.z)
                .set_scale(0.16)
                .set_shadows(cast_shadows, cast_shadows);

            if self.preset.animates_all_transforms() {
                let phase = index as f32 * 0.013_579;
                self.dynamic_objects.push(DynamicObject {
                    node,
                    base,
                    phase_sin: phase.sin(),
                    phase_cos: phase.cos(),
                });
            }
        }

        if self.preset.uses_shadow_amplification() {
            self.add_shadow_amplification(scene);
        }

        self.workload_root = Some(root);
        self.time = 0.0;
        self.measurement.reset();
        self.ab = AbExperiment::new(self.mode);
        println!(
            "GPU_DRIVEN_LAB preset={} objects={} intended_bins={} mode={:?}",
            self.preset.label(),
            object_count,
            self.preset.intended_bin_count(),
            self.mode,
        );
    }

    fn add_shadow_amplification(&mut self, scene: &mut Scene) {
        let mut sun = Light::new_directional(Vec3::new(1.0, 0.92, 0.80), 1.8);
        sun.cast_shadows = true;
        if let Some(shadow) = sun.shadow.as_mut() {
            shadow.map_size = 512;
            shadow.cascade_count = 4;
            shadow.max_shadow_distance = 55.0;
        }
        let sun = scene.add_light(sun);
        scene
            .node(&sun)
            .set_position(14.0, 20.0, 12.0)
            .look_at(Vec3::ZERO);
        self.workload_lights.push(sun);

        for index in 0..4 {
            let angle = index as f32 * std::f32::consts::FRAC_PI_2;
            let mut point = Light::new_point(Vec3::new(0.45, 0.68, 1.0), 1.1, 35.0);
            point.cast_shadows = true;
            if let Some(shadow) = point.shadow.as_mut() {
                // View amplification, not fill-rate or VRAM pressure, is the
                // axis under test. Keep all 24 cube faces intentionally small.
                shadow.map_size = 256;
            }
            let point = scene.add_light(point);
            scene.node(&point).set_position(
                angle.cos() * 9.0,
                7.0 + index as f32 * 0.45,
                angle.sin() * 9.0,
            );
            self.workload_lights.push(point);
        }
    }

    fn animate_dynamic_storm(&mut self, scene: &mut Scene, dt: f32) {
        if !self.preset.animates_all_transforms() {
            return;
        }
        self.time += dt;
        let time_sin = (self.time * 2.4).sin();
        let time_cos = (self.time * 2.4).cos();
        for object in &self.dynamic_objects {
            let wave = time_sin * object.phase_cos + time_cos * object.phase_sin;
            scene.node(&object.node).set_position(
                object.base.x,
                object.base.y + wave * 0.32,
                object.base.z,
            );
        }
    }

    fn start_ab(&mut self, engine: &mut Engine) {
        if !self.gpu_baseline_available || self.ab.is_active() {
            return;
        }
        self.ab = AbExperiment::new(self.mode);
        self.ab.phase = AbPhase::MeasuringOff;
        self.mode = GpuDrivenMode::Off;
        engine.renderer.set_gpu_driven_mode(self.mode);
        self.time = 0.0;
        self.measurement.reset();
        println!(
            "GPU_DRIVEN_LAB_AB start preset={} original_mode={:?}",
            self.preset.label(),
            self.ab.original_mode
        );
    }

    fn cancel_ab(&mut self, engine: &mut Engine, override_mode: Option<GpuDrivenMode>) {
        if !self.ab.is_active() {
            return;
        }
        let restored = override_mode.unwrap_or(self.ab.original_mode);
        self.ab = AbExperiment::new(restored);
        self.mode = restored;
        engine.renderer.set_gpu_driven_mode(restored);
        self.measurement.reset();
        println!("GPU_DRIVEN_LAB_AB cancelled restored_mode={restored:?}");
    }

    fn fail_ab(&mut self, engine: &mut Engine, failure: RdgGpuDrivenFailure) {
        if self.ab.phase != AbPhase::MeasuringRequired {
            return;
        }
        self.ab.phase = AbPhase::Failed;
        self.ab.failure = Some(failure);
        self.mode = self.ab.original_mode;
        engine.renderer.set_gpu_driven_mode(self.mode);
        self.measurement.reset();
        println!(
            "GPU_DRIVEN_LAB_AB failed reason={failure:?} restored_mode={:?}",
            self.mode
        );
    }

    fn advance_ab_if_complete(&mut self, engine: &mut Engine) {
        let Some(result) = self.measurement.result else {
            return;
        };
        match self.ab.phase {
            AbPhase::MeasuringOff => {
                self.ab.off = Some(result);
                self.ab.phase = AbPhase::MeasuringRequired;
                self.mode = GpuDrivenMode::Required;
                engine.renderer.set_gpu_driven_mode(self.mode);
                self.time = 0.0;
                self.measurement.reset();
                println!(
                    "GPU_DRIVEN_LAB_AB off_complete mean_ms={:.4}; starting Required",
                    result.mean_frame_ms
                );
            }
            AbPhase::MeasuringRequired => {
                self.ab.required = Some(result);
                self.ab.phase = AbPhase::Complete;
                self.mode = self.ab.original_mode;
                engine.renderer.set_gpu_driven_mode(self.mode);
                self.measurement.reset();
                let off = self
                    .ab
                    .off
                    .expect("A/B Required completed without Off result");
                println!(
                    "GPU_DRIVEN_LAB_AB complete off_ms={:.4} required_ms={:.4} speedup={:.4}x restored_mode={:?}",
                    off.mean_frame_ms,
                    result.mean_frame_ms,
                    off.mean_frame_ms / result.mean_frame_ms,
                    self.mode
                );
            }
            AbPhase::Idle | AbPhase::Complete | AbPhase::Failed => {}
        }
    }

    fn start_matrix(&mut self, engine: &mut Engine) {
        if !self.gpu_baseline_available || self.ab.is_active() || self.matrix.is_active() {
            return;
        }

        self.matrix = MatrixExperiment::new(self.preset, self.mode);
        self.matrix.phase = MatrixPhase::Running;
        println!(
            "GPU_DRIVEN_LAB_MATRIX start presets={} original_preset={} original_mode={:?}",
            LabPreset::ALL.len(),
            self.matrix.original_preset.label(),
            self.matrix.original_mode,
        );
        self.start_current_matrix_preset(engine);
    }

    fn start_current_matrix_preset(&mut self, engine: &mut Engine) {
        assert!(self.matrix.is_active());
        let preset = LabPreset::ALL[self.matrix.preset_index];
        self.mode = self.matrix.original_mode;
        engine.renderer.set_gpu_driven_mode(self.mode);
        self.preset = preset;
        self.rebuild_workload(engine);
        println!(
            "GPU_DRIVEN_LAB_MATRIX preset_start index={}/{} preset={}",
            self.matrix.preset_index + 1,
            LabPreset::ALL.len(),
            preset.label(),
        );
        self.start_ab(engine);
    }

    fn cancel_matrix(&mut self, engine: &mut Engine, override_mode: Option<GpuDrivenMode>) {
        if !self.matrix.is_active() {
            return;
        }

        let restored_mode = override_mode.unwrap_or(self.matrix.original_mode);
        let restored_preset = self.matrix.original_preset;
        if self.ab.is_active() {
            self.cancel_ab(engine, Some(restored_mode));
        } else {
            self.mode = restored_mode;
            engine.renderer.set_gpu_driven_mode(restored_mode);
        }
        self.matrix.phase = MatrixPhase::Cancelled;
        self.mode = restored_mode;
        self.preset = restored_preset;
        self.rebuild_workload(engine);
        println!(
            "GPU_DRIVEN_LAB_MATRIX cancelled completed={}/{} restored_preset={} restored_mode={restored_mode:?}",
            self.matrix.results.len(),
            LabPreset::ALL.len(),
            restored_preset.label(),
        );
    }

    fn advance_matrix_if_ready(&mut self, engine: &mut Engine) {
        if !self.matrix.is_active() {
            return;
        }

        let required = match self.ab.phase {
            AbPhase::Complete => Ok(self
                .ab
                .required
                .expect("completed matrix A/B lost its Required result")),
            AbPhase::Failed => Err(self
                .ab
                .failure
                .expect("failed matrix A/B lost its typed failure")),
            AbPhase::Idle | AbPhase::MeasuringOff | AbPhase::MeasuringRequired => return,
        };
        let off = self
            .ab
            .off
            .expect("terminal matrix A/B lost its Off result");
        let preset = LabPreset::ALL[self.matrix.preset_index];
        debug_assert_eq!(self.preset, preset);
        self.matrix.results.push(MatrixResult {
            preset,
            off,
            required,
        });
        match required {
            Ok(required) => println!(
                "GPU_DRIVEN_LAB_MATRIX_RESULT preset={} off_mean_ms={:.4} required_mean_ms={:.4} speedup={:.4}x off_p95_ms={:.4} required_p95_ms={:.4} status=ok",
                preset.label(),
                off.mean_frame_ms,
                required.mean_frame_ms,
                off.mean_frame_ms / required.mean_frame_ms,
                off.p95_frame_ms,
                required.p95_frame_ms,
            ),
            Err(failure) => println!(
                "GPU_DRIVEN_LAB_MATRIX_RESULT preset={} off_mean_ms={:.4} status=required_failed reason={failure:?}",
                preset.label(),
                off.mean_frame_ms,
            ),
        }
        #[cfg(feature = "rdg_bench_support")]
        {
            print_matrix_cpu_timings(preset, "Off", off);
            if let Ok(required) = required {
                print_matrix_cpu_timings(preset, "Required", required);
            }
        }

        self.matrix.preset_index += 1;
        if self.matrix.preset_index < LabPreset::ALL.len() {
            self.start_current_matrix_preset(engine);
            return;
        }

        self.matrix.phase = MatrixPhase::Complete;
        self.print_matrix_table();
        let restored_preset = self.matrix.original_preset;
        let restored_mode = self.matrix.original_mode;
        self.mode = restored_mode;
        engine.renderer.set_gpu_driven_mode(restored_mode);
        self.preset = restored_preset;
        self.rebuild_workload(engine);
        println!(
            "GPU_DRIVEN_LAB_MATRIX complete restored_preset={} restored_mode={restored_mode:?}",
            restored_preset.label(),
        );
    }

    fn print_matrix_table(&self) {
        println!("GPU_DRIVEN_LAB_MATRIX_TABLE_BEGIN");
        println!("preset,off_mean_ms,required_mean_ms,speedup,off_p95_ms,required_p95_ms,status");
        for row in &self.matrix.results {
            match row.required {
                Ok(required) => println!(
                    "{},{:.4},{:.4},{:.4},{:.4},{:.4},ok",
                    row.preset.label(),
                    row.off.mean_frame_ms,
                    required.mean_frame_ms,
                    row.off.mean_frame_ms / required.mean_frame_ms,
                    row.off.p95_frame_ms,
                    required.p95_frame_ms,
                ),
                Err(failure) => println!(
                    "{},{:.4},,,{:.4},,required_failed:{failure:?}",
                    row.preset.label(),
                    row.off.mean_frame_ms,
                    row.off.p95_frame_ms,
                ),
            }
        }
        println!("GPU_DRIVEN_LAB_MATRIX_TABLE_END");
    }

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        diagnostics: Option<&RdgFrameDiagnostics>,
        #[cfg(feature = "rdg_bench_support")] gpu_timings: Option<&RdgGpuFrameTimings>,
    ) -> (LabPreset, GpuDrivenMode, UiActions) {
        let mut requested_preset = self.preset;
        let mut requested_mode = self.mode;
        let mut actions = UiActions::default();
        let controls_enabled = !self.ab.is_active() && !self.matrix.is_active();

        egui::Window::new("GPU-Driven M7.5 Lab")
            .default_pos([12.0, 12.0])
            .default_width(470.0)
            .show(ctx, |ui| {
                ui.label("Deterministic low-poly CPU/GPU submission comparison");
                ui.separator();

                ui.add_enabled_ui(controls_enabled, |ui| {
                    egui::ComboBox::from_label("Workload preset")
                        .selected_text(requested_preset.label())
                        .show_ui(ui, |ui| {
                            for preset in LabPreset::ALL {
                                ui.selectable_value(
                                    &mut requested_preset,
                                    preset,
                                    preset.label(),
                                );
                            }
                        });
                });
                ui.small(requested_preset.description());
                ui.label(format!(
                    "N={} | intended B={}",
                    requested_preset.object_count(),
                    requested_preset.intended_bin_count()
                ));
                if requested_preset == LabPreset::ShadowAmplification {
                    ui.small(
                        "Cube diagnostics count submissions (4); each point-light submission renders six faces (24 total).",
                    );
                }

                ui.horizontal(|ui| {
                    ui.label("Submission mode:");
                    ui.add_enabled_ui(controls_enabled, |ui| {
                        ui.selectable_value(&mut requested_mode, GpuDrivenMode::Off, "Off");
                        ui.selectable_value(&mut requested_mode, GpuDrivenMode::Auto, "Auto");
                        ui.add_enabled_ui(self.gpu_baseline_available, |ui| {
                            ui.selectable_value(
                                &mut requested_mode,
                                GpuDrivenMode::Required,
                                "Required",
                            );
                        });
                    });
                });
                if !self.gpu_baseline_available {
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "Required and automatic A/B are disabled: indirect GPU baseline unavailable.",
                    );
                }
                ui.small(
                    "Keyboard: 1-6 presets; F1 Off (always recovers), F2 Auto, F3 Required only when enabled.",
                );

                ui.separator();
                ui.strong("Fixed measurement window");
                ui.small("End-to-end update intervals; this is not a GPU timestamp query.");
                self.draw_measurement(ui);
                if ui
                    .add_enabled(
                        controls_enabled,
                        egui::Button::new("Restart warmup + measurement"),
                    )
                    .clicked()
                {
                    actions.restart_measurement = true;
                }

                ui.separator();
                ui.strong("Automatic Off → Required A/B");
                self.draw_ab(ui);
                if self.ab.is_active() && !self.matrix.is_active() {
                    if ui.button("Cancel A/B and restore original mode").clicked() {
                        actions.experiment = Some(ExperimentAction::CancelAb);
                    }
                } else if !self.matrix.is_active()
                    && ui
                    .add_enabled(
                        self.gpu_baseline_available,
                        egui::Button::new("Run complete Off → Required A/B"),
                    )
                    .clicked()
                {
                    actions.experiment = Some(ExperimentAction::StartAb);
                }

                ui.separator();
                ui.strong("Full preset Off → Required matrix");
                self.draw_matrix(ui);
                if self.matrix.is_active() {
                    if ui.button("Cancel matrix and restore original setup").clicked() {
                        actions.experiment = Some(ExperimentAction::CancelMatrix);
                    }
                } else if ui
                    .add_enabled(
                        self.gpu_baseline_available && !self.ab.is_active(),
                        egui::Button::new("Run all six preset comparisons"),
                    )
                    .clicked()
                {
                    actions.experiment = Some(ExperimentAction::StartMatrix);
                }

                ui.separator();
                ui.strong("Previous completed RDG attempt");
                draw_diagnostics(ui, diagnostics);

                #[cfg(feature = "rdg_bench_support")]
                {
                    ui.separator();
                    ui.strong("Asynchronous GPU attribution");
                    let mut enabled = self.gpu_timing_enabled;
                    let changed = ui
                        .add_enabled(
                            self.gpu_timing_supported && controls_enabled,
                            egui::Checkbox::new(
                                &mut enabled,
                                "Sample GPU timestamps every 8 frames",
                            ),
                        )
                        .changed();
                    if changed {
                        actions.gpu_timing_enabled = Some(enabled);
                    }
                    if !self.gpu_timing_supported {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            "This adapter does not expose encoder timestamp queries.",
                        );
                    } else if enabled {
                        ui.small(
                            "Attribution mode can perturb throughput; disable it for the final A/B baseline.",
                        );
                        draw_gpu_timings(ui, gpu_timings);
                    } else {
                        ui.small("Disabled by default; no timestamp commands or readbacks are recorded.");
                    }
                }
            });

        (requested_preset, requested_mode, actions)
    }

    fn draw_measurement(&self, ui: &mut egui::Ui) {
        if self.measurement.warmup_remaining != 0 {
            let completed = WARMUP_FRAMES - self.measurement.warmup_remaining;
            ui.label(format!(
                "Warmup: {completed}/{WARMUP_FRAMES} frames (scene rebuild excluded)"
            ));
            ui.add(egui::ProgressBar::new(
                completed as f32 / WARMUP_FRAMES as f32,
            ));
        } else if let Some(result) = self.measurement.result {
            ui.label(format!(
                "{SAMPLE_FRAMES} frames: {:.1} FPS | mean {:.3} ms",
                result.mean_fps, result.mean_frame_ms
            ));
            ui.label(format!(
                "p50/p95/p99/max: {:.3} / {:.3} / {:.3} / {:.3} ms",
                result.p50_frame_ms, result.p95_frame_ms, result.p99_frame_ms, result.max_frame_ms
            ));
        } else {
            ui.label(format!(
                "Sampling: {}/{SAMPLE_FRAMES} frames",
                self.measurement.samples_ns.len()
            ));
            ui.add(egui::ProgressBar::new(
                self.measurement.samples_ns.len() as f32 / SAMPLE_FRAMES as f32,
            ));
        }
    }

    fn draw_ab(&self, ui: &mut egui::Ui) {
        match self.ab.phase {
            AbPhase::Idle => {
                ui.label("Runs two independent 120-warmup + 600-sample windows.");
            }
            AbPhase::MeasuringOff => {
                ui.label("Phase 1/2: measuring Off (CPU-direct baseline)");
            }
            AbPhase::MeasuringRequired => {
                ui.label("Phase 2/2: measuring Required (forced GPU-driven)");
                if let Some(off) = self.ab.off {
                    ui.label(format!("Saved Off mean: {:.3} ms", off.mean_frame_ms));
                }
            }
            AbPhase::Complete => {
                let off = self.ab.off.expect("complete A/B lost Off result");
                let required = self.ab.required.expect("complete A/B lost Required result");
                let speedup = off.mean_frame_ms / required.mean_frame_ms;
                ui.label(format!(
                    "Off {:.3} ms / Required {:.3} ms → {:.3}x ({:+.1}%)",
                    off.mean_frame_ms,
                    required.mean_frame_ms,
                    speedup,
                    (speedup - 1.0) * 100.0
                ));
                ui.small(format!(
                    "Off p95/p99 {:.3}/{:.3} ms | Required {:.3}/{:.3} ms",
                    off.p95_frame_ms,
                    off.p99_frame_ms,
                    required.p95_frame_ms,
                    required.p99_frame_ms
                ));
            }
            AbPhase::Failed => {
                ui.colored_label(
                    egui::Color32::RED,
                    format!(
                        "Required failed with {:?}; original mode restored.",
                        self.ab.failure
                    ),
                );
            }
        }
    }

    fn draw_matrix(&self, ui: &mut egui::Ui) {
        match self.matrix.phase {
            MatrixPhase::Idle => {
                ui.label(
                    "Runs every preset through independent Off and Required measurement windows.",
                );
            }
            MatrixPhase::Running => {
                let preset_number = self.matrix.preset_index + 1;
                let completed_windows = self.matrix.results.len() * 2
                    + usize::from(matches!(
                        self.ab.phase,
                        AbPhase::MeasuringRequired | AbPhase::Complete | AbPhase::Failed
                    ));
                let current_window_progress = if self.measurement.result.is_some() {
                    1.0
                } else if self.measurement.warmup_remaining != 0 {
                    f64::from(WARMUP_FRAMES - self.measurement.warmup_remaining)
                        / f64::from(WARMUP_FRAMES + SAMPLE_FRAMES)
                } else {
                    f64::from(WARMUP_FRAMES).mul_add(1.0, self.measurement.samples_ns.len() as f64)
                        / f64::from(WARMUP_FRAMES + SAMPLE_FRAMES)
                };
                let total_windows = LabPreset::ALL.len() * 2;
                let overall =
                    (completed_windows as f64 + current_window_progress) / total_windows as f64;
                ui.label(format!(
                    "Preset {preset_number}/{}: {} | {:?}",
                    LabPreset::ALL.len(),
                    self.preset.label(),
                    self.ab.phase,
                ));
                ui.add(
                    egui::ProgressBar::new(overall.clamp(0.0, 1.0) as f32)
                        .text(format!("{:.1}%", overall * 100.0)),
                );
                ui.small(format!(
                    "Completed {completed_windows} of {total_windows} Off/Required windows"
                ));
            }
            MatrixPhase::Complete => {
                ui.colored_label(
                    egui::Color32::LIGHT_GREEN,
                    "Matrix complete; the original preset and mode were restored.",
                );
            }
            MatrixPhase::Cancelled => {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!(
                        "Matrix cancelled after {}/{} presets; partial results were retained.",
                        self.matrix.results.len(),
                        LabPreset::ALL.len()
                    ),
                );
            }
        }

        if self.matrix.results.is_empty() {
            return;
        }
        egui::Grid::new("gpu_driven_lab_matrix_results")
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Preset");
                ui.strong("Off ms");
                ui.strong("Required ms");
                ui.strong("Speedup");
                ui.end_row();
                for row in &self.matrix.results {
                    ui.label(row.preset.label());
                    ui.label(format!("{:.3}", row.off.mean_frame_ms));
                    match row.required {
                        Ok(required) => {
                            ui.label(format!("{:.3}", required.mean_frame_ms));
                            ui.label(format!(
                                "{:.3}x",
                                row.off.mean_frame_ms / required.mean_frame_ms
                            ));
                        }
                        Err(failure) => {
                            ui.colored_label(egui::Color32::RED, format!("{failure:?}"));
                            ui.label("—");
                        }
                    }
                    ui.end_row();
                }
            });
        #[cfg(feature = "rdg_bench_support")]
        ui.small(
            "Per-row RDG phase/detail means were sampled with the same window and emitted to stdout.",
        );
        #[cfg(not(feature = "rdg_bench_support"))]
        ui.small(
            "Enable `rdg_bench_support` to add per-row RDG phase/detail means; the matrix itself does not require it.",
        );
    }
}

impl AppHandler for GpuDrivenLab {
    fn init(engine: &mut Engine, window: &dyn Window) -> Self {
        let ui_pass = {
            let wgpu_ctx = engine
                .renderer
                .wgpu_ctx()
                .expect("renderer must be initialized before GPU-driven lab setup");
            let winit_window = window
                .as_any()
                .downcast_ref::<winit::window::Window>()
                .expect("GPU-driven lab requires the winit window backend");
            UiPass::new(&wgpu_ctx.device, wgpu_ctx.surface_view_format, winit_window)
        };

        let geometry = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let materials = (0..FRAGMENTED_OBJECTS)
            .map(|index| {
                let r = 0.18 + ((index * 37) % 251) as f32 / 250.0 * 0.65;
                let g = 0.18 + ((index * 73) % 241) as f32 / 240.0 * 0.65;
                let b = 0.18 + ((index * 109) % 239) as f32 / 238.0 * 0.65;
                engine.assets.materials.add(
                    PhysicalMaterial::new(Vec4::new(r, g, b, 1.0))
                        .with_roughness(0.62)
                        .with_metalness(0.08),
                )
            })
            .collect();

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.08));
        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.035, 0.045, 0.065, 1.0),
            Vec4::new(0.008, 0.010, 0.016, 1.0),
        ));
        let key_light = scene.add_light(Light::new_directional(Vec3::ONE, 2.2));
        scene
            .node(&key_light)
            .set_position(10.0, 16.0, 8.0)
            .look_at(Vec3::ZERO);

        let camera_position = Vec3::new(0.0, 22.0, 27.0);
        let camera_target = Vec3::ZERO;
        let camera = scene.add_camera(Camera::new_perspective(50.0, 16.0 / 9.0, 0.1));
        scene
            .node(&camera)
            .set_position(camera_position.x, camera_position.y, camera_position.z)
            .look_at(camera_target);
        scene.active_camera = Some(camera);

        let mode = engine.renderer.settings().gpu_driven;
        let gpu_baseline_available = engine
            .renderer
            .gpu_driven_capabilities()
            .is_some_and(myth::GpuDrivenCapabilities::supports_indirect_primitives);
        #[cfg(feature = "rdg_bench_support")]
        let gpu_timing_supported = engine.renderer.rdg_gpu_timings_supported();
        let mut lab = Self {
            ui_pass,
            controls: OrbitControls::new(camera_position, camera_target),
            fps_counter: FpsCounter::new(),
            preset: LabPreset::Tiny,
            mode,
            geometry,
            materials,
            workload_root: None,
            workload_lights: Vec::new(),
            dynamic_objects: Vec::with_capacity(DYNAMIC_OBJECTS),
            time: 0.0,
            measurement: MeasurementWindow::new(),
            gpu_baseline_available,
            #[cfg(feature = "rdg_bench_support")]
            gpu_timing_supported,
            #[cfg(feature = "rdg_bench_support")]
            gpu_timing_enabled: false,
            ab: AbExperiment::new(mode),
            matrix: MatrixExperiment::new(LabPreset::Tiny, mode),
        };
        lab.rebuild_workload(engine);

        println!("GPU-Driven M7.5 Lab: 1-6 presets, F1=Off, F2=Auto, F3=Required");
        if let Some(capabilities) = engine.renderer.gpu_driven_capabilities() {
            println!("GPU_DRIVEN_LAB capability_tier={:?}", capabilities.tier());
        }
        lab
    }

    fn on_event(&mut self, _engine: &mut Engine, window: &dyn Window, event: &dyn Any) -> bool {
        let Some(event) = event.downcast_ref::<WindowEvent>() else {
            return false;
        };
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("GPU-driven lab requires the winit window backend");
        if self.ui_pass.handle_input(winit_window, event) {
            return true;
        }
        if let WindowEvent::Resized(size) = event {
            self.ui_pass
                .resize(size.width, size.height, window.scale_factor());
        }
        false
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let diagnostics = engine.renderer.latest_rdg_diagnostics();
        if self.ab.phase == AbPhase::MeasuringRequired
            && let Some(diagnostics) = diagnostics
            && let RdgAttemptOutcome::GpuDrivenFailed(failure) = diagnostics.outcome()
        {
            self.fail_ab(engine, failure);
        }
        #[cfg(feature = "rdg_bench_support")]
        let gpu_timings = engine.renderer.latest_rdg_gpu_timings();
        let keyboard_preset = keyboard_preset(&engine.input);
        let keyboard_mode = keyboard_mode(&engine.input);

        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("GPU-driven lab requires the winit window backend");
        self.ui_pass.begin_frame(winit_window);
        let egui_ctx = self.ui_pass.context().clone();
        let (ui_preset, ui_mode, actions) = self.draw_ui(
            &egui_ctx,
            diagnostics.as_ref(),
            #[cfg(feature = "rdg_bench_support")]
            gpu_timings,
        );
        self.ui_pass.end_frame(winit_window);

        #[cfg(feature = "rdg_bench_support")]
        if let Some(enabled) = actions.gpu_timing_enabled {
            self.gpu_timing_enabled = enabled;
            let interval = if enabled { 8 } else { 0 };
            assert!(
                engine.renderer.set_rdg_gpu_timing_sample_interval(interval),
                "initialized renderer rejected GPU timing cadence"
            );
        }

        if actions.experiment == Some(ExperimentAction::CancelMatrix) {
            self.cancel_matrix(engine, None);
        } else if self.matrix.is_active() && keyboard_mode == Some(GpuDrivenMode::Off) {
            // F1 remains a hard recovery path for the complete unattended
            // matrix, including a Required attempt which cannot draw the UI.
            self.cancel_matrix(engine, Some(GpuDrivenMode::Off));
        } else if actions.experiment == Some(ExperimentAction::CancelAb) {
            self.cancel_ab(engine, None);
        } else if self.ab.is_active() && keyboard_mode == Some(GpuDrivenMode::Off) {
            // F1 is a recovery path even while the automated experiment owns
            // the normal mode UI.
            self.cancel_ab(engine, Some(GpuDrivenMode::Off));
        } else if !self.ab.is_active() && !self.matrix.is_active() {
            if actions.experiment == Some(ExperimentAction::StartMatrix) {
                self.start_matrix(engine);
            } else if actions.experiment == Some(ExperimentAction::StartAb) {
                self.start_ab(engine);
            } else {
                let requested_mode = keyboard_mode.unwrap_or(ui_mode);
                if (requested_mode != GpuDrivenMode::Required || self.gpu_baseline_available)
                    && requested_mode != self.mode
                {
                    self.mode = requested_mode;
                    engine.renderer.set_gpu_driven_mode(self.mode);
                    self.measurement.reset();
                    println!("GPU_DRIVEN_LAB mode={:?}", self.mode);
                }

                let requested_preset = keyboard_preset.unwrap_or(ui_preset);
                if requested_preset != self.preset {
                    self.preset = requested_preset;
                    self.rebuild_workload(engine);
                } else if actions.restart_measurement {
                    self.measurement.reset();
                }
            }
        }

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };
        let ab_active = self.ab.is_active();
        let animation_dt = if ab_active { 1.0 / 60.0 } else { frame.dt };
        self.animate_dynamic_storm(scene, animation_dt);
        if !ab_active && let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        #[cfg(feature = "rdg_bench_support")]
        if let Some(diagnostics) = diagnostics.as_ref() {
            self.measurement.record_phase_timings(diagnostics);
        }
        self.measurement.update(Instant::now());
        self.advance_ab_if_complete(engine);
        self.advance_matrix_if_ready(engine);
        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "GPU-Driven Lab | {} | {:?} | {:.1} FPS",
                self.preset.label(),
                self.mode,
                fps
            ));
        }
    }

    fn render(&mut self, engine: &mut Engine, _window: &dyn Window) {
        use myth::renderer::graph::core::{GraphBlackboard, HookStage};

        let Some(composer) = engine.compose_frame() else {
            return;
        };
        self.ui_pass
            .resolve_textures(composer.device(), composer.resource_manager());
        let ui_pass = &mut self.ui_pass;
        composer
            .add_custom_pass(HookStage::AfterPostProcess, move |rdg, blackboard| {
                let surface_out = rdg.add_pass("GpuDrivenLabUI", |builder| {
                    let output =
                        builder.mutate_texture(blackboard.surface_out, "Surface_With_Lab_UI");
                    (
                        UiPassNode {
                            pass: ui_pass,
                            target_tex: output,
                        },
                        output,
                    )
                });
                GraphBlackboard {
                    surface_out,
                    ..blackboard
                }
            })
            .render();
    }
}

#[cfg(feature = "rdg_bench_support")]
fn print_matrix_cpu_timings(preset: LabPreset, mode: &str, result: MeasurementResult) {
    for (index, phase) in MATRIX_PHASES.into_iter().enumerate() {
        println!(
            "GPU_DRIVEN_LAB_MATRIX_CPU preset={} mode={mode} kind=phase phase={phase:?} samples={} mean_us={:.3}",
            preset.label(),
            result.timing_samples,
            result.phase_mean_us[index],
        );
    }
    for (index, phase) in MATRIX_DETAIL_PHASES.into_iter().enumerate() {
        println!(
            "GPU_DRIVEN_LAB_MATRIX_CPU preset={} mode={mode} kind=detail phase={phase:?} samples={} mean_us={:.3}",
            preset.label(),
            result.timing_samples,
            result.detail_mean_us[index],
        );
    }
}

#[cfg(feature = "rdg_bench_support")]
fn draw_gpu_timings(ui: &mut egui::Ui, timings: Option<&RdgGpuFrameTimings>) {
    let Some(timings) = timings else {
        ui.label("Awaiting the first non-blocking GPU timing sample.");
        return;
    };

    ui.label(format!(
        "GPU total {:.3} ms | renderer frame {} | lag {} frames",
        timings.gpu_total_ms(),
        timings.renderer_frame(),
        timings.sample_lag(),
    ));
    ui.small(format!(
        "Dropped samples: {} | map errors: {}",
        timings.dropped_samples(),
        timings.map_errors(),
    ));

    // Keep attribution display allocation-free. Three insertion slots are
    // enough to identify the dominant GPU scopes without sorting every pass.
    let mut heaviest: [Option<&RdgGpuPassTiming>; 3] = [None; 3];
    for pass in timings.passes() {
        for rank in 0..heaviest.len() {
            if heaviest[rank].is_none_or(|current| pass.duration_ns() > current.duration_ns()) {
                for move_index in (rank + 1..heaviest.len()).rev() {
                    heaviest[move_index] = heaviest[move_index - 1];
                }
                heaviest[rank] = Some(pass);
                break;
            }
        }
    }
    for (rank, pass) in heaviest.into_iter().flatten().enumerate() {
        ui.small(format!(
            "#{} {}: {:.3} ms",
            rank + 1,
            pass.name(),
            pass.duration_ms(),
        ));
    }
}

fn draw_diagnostics(ui: &mut egui::Ui, diagnostics: Option<&RdgFrameDiagnostics>) {
    let Some(diagnostics) = diagnostics else {
        ui.label("No terminal RDG attempt yet.");
        return;
    };

    ui.label(format!(
        "Attempt {} | {:?} | passes declared/live/prepared/executed: {}/{:?}/{}/{}",
        diagnostics.attempt_id(),
        diagnostics.outcome(),
        diagnostics.declared_passes(),
        diagnostics.live_passes(),
        diagnostics.prepared_passes(),
        diagnostics.executed_passes(),
    ));

    let policy = diagnostics.gpu_driven_policy();
    ui.label(format!(
        "Policy N={} supported={} retained={} B={} dirty={} exact={} cohort≥{}",
        policy.extracted_item_count(),
        policy.supported_opaque_count(),
        policy.retained_candidate_count(),
        policy.retained_bin_count(),
        policy.dirty_candidate_count(),
        policy.exact_evaluated(),
        policy.minimum_cohort_members(),
    ));
    ui.label(format!(
        "Views D2/Cube={}/{} | slot span instance/batch={}/{}",
        policy.shadow_2d_view_count(),
        policy.shadow_cube_view_count(),
        policy.instance_slot_span(),
        policy.batch_slot_span(),
    ));

    egui::Grid::new("gpu_driven_lab_submission_grid")
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Family");
            ui.strong("Selected / reason");
            ui.strong("Live CPU/GPU");
            ui.end_row();
            for family in RdgSceneSubmissionFamily::ALL {
                let snapshot = diagnostics.scene_submission_family(family);
                ui.label(format!("{family:?}"));
                ui.label(format!(
                    "{:?} / {:?}",
                    snapshot.selected_backend(),
                    snapshot.selection_reason()
                ));
                ui.label(format!(
                    "{:?} ({}/{})",
                    snapshot.live_backend(),
                    snapshot.live_cpu_submissions(),
                    snapshot.live_gpu_submissions()
                ));
                ui.end_row();
            }
        });
}

fn keyboard_preset(input: &myth_resources::Input) -> Option<LabPreset> {
    if input.get_key_down(Key::Key1) {
        Some(LabPreset::Tiny)
    } else if input.get_key_down(Key::Key2) {
        Some(LabPreset::MassiveReuse)
    } else if input.get_key_down(Key::Key3) {
        Some(LabPreset::Fragmented)
    } else if input.get_key_down(Key::Key4) {
        Some(LabPreset::DynamicStorm)
    } else if input.get_key_down(Key::Key5) {
        Some(LabPreset::ShadowAmplification)
    } else if input.get_key_down(Key::Key6) {
        Some(LabPreset::HybridCohorts)
    } else {
        None
    }
}

fn keyboard_mode(input: &myth_resources::Input) -> Option<GpuDrivenMode> {
    if input.get_key_down(Key::F1) {
        Some(GpuDrivenMode::Off)
    } else if input.get_key_down(Key::F2) {
        Some(GpuDrivenMode::Auto)
    } else if input.get_key_down(Key::F3) {
        Some(GpuDrivenMode::Required)
    } else {
        None
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("GPU-Driven M7.5 Lab")
        .with_settings(RendererSettings {
            path: RenderPath::BasicForward,
            vsync: false,
            gpu_driven: GpuDrivenMode::Auto,
            ..Default::default()
        })
        .run::<GpuDrivenLab>()
}
