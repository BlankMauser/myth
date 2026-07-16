//! Fixed-frame end-to-end probe for the `helmet_gltf` scene.
//!
//! This is intentionally not a gallery example. It keeps the scene and renderer
//! settings aligned with `helmet_gltf`, waits for the helmet prefab to load, then
//! measures update-to-update frame intervals after a fixed warmup. The measured
//! hot loop writes into preallocated storage and performs no reporting work.
//!
//! Run with the defaults (10,000 warmup intervals and 200,000 measured intervals):
//!
//! ```powershell
//! cargo run --release --example helmet_gltf_fixed_probe
//! ```
//!
//! Override the fixed counts in PowerShell:
//!
//! ```powershell
//! $env:MYTH_HELMET_PROBE_WARMUP_FRAMES='10000'
//! $env:MYTH_HELMET_PROBE_MEASURE_FRAMES='200000'
//! $env:MYTH_HELMET_PROBE_GPU_DRIVEN='auto' # off | auto | required
//! cargo run --release --example helmet_gltf_fixed_probe
//! ```

use std::io::{self, Write};
use std::time::Instant;

use myth::prelude::*;
#[cfg(feature = "rdg_bench_support")]
use myth::renderer::graph::{RdgDetailPhase, RdgFramePhase};
use myth::renderer::graph::{RdgFrameDiagnostics, RdgSceneSubmissionFamily};
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

const WARMUP_FRAMES_ENV: &str = "MYTH_HELMET_PROBE_WARMUP_FRAMES";
const MEASURE_FRAMES_ENV: &str = "MYTH_HELMET_PROBE_MEASURE_FRAMES";
const DISABLE_ENV_ENV: &str = "MYTH_HELMET_PROBE_DISABLE_ENV";
const UNTEXTURED_ENV: &str = "MYTH_HELMET_PROBE_UNTEXTURED";
const NO_DRAW_ENV: &str = "MYTH_HELMET_PROBE_NO_DRAW";
const GPU_DRIVEN_ENV: &str = "MYTH_HELMET_PROBE_GPU_DRIVEN";
const DEFAULT_WARMUP_FRAMES: usize = 10_000;
const DEFAULT_MEASURE_FRAMES: usize = 200_000;

#[derive(Clone, Copy)]
struct ProbeResult {
    warmup_frames: usize,
    measured_frames: usize,
    total_ns: u128,
    median_ns: f64,
    p95_ns: u64,
    p99_ns: u64,
    max_ns: u64,
    mean_ns: f64,
    environment_enabled: bool,
    textured: bool,
    draw_model: bool,
    gpu_driven: GpuDrivenMode,
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "independent probe axes and one-shot lifecycle flags are clearer as named booleans"
)]
struct HelmetGltfFixedProbe {
    camera: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    helmet_prefab: PrefabHandle,
    helmet_loaded: bool,
    replacement_material: Option<MaterialHandle>,
    environment_enabled: bool,
    draw_model: bool,
    gpu_driven: GpuDrivenMode,
    warmup_frames: usize,
    warmup_remaining: usize,
    measure_frames: usize,
    last_update: Option<Instant>,
    frame_intervals_ns: Vec<u64>,
    #[cfg(feature = "rdg_bench_support")]
    phase_elapsed_ns: [u128; RdgFramePhase::COUNT],
    #[cfg(feature = "rdg_bench_support")]
    detail_elapsed_ns: [u128; RdgDetailPhase::COUNT],
    #[cfg(feature = "rdg_bench_support")]
    phase_timing_samples: usize,
    result: Option<ProbeResult>,
    exit_requested: bool,
}

impl HelmetGltfFixedProbe {
    fn record_interval(&mut self, now: Instant) -> bool {
        let Some(previous) = self.last_update.replace(now) else {
            return false;
        };

        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            return false;
        }

        let interval_ns = now.duration_since(previous).as_nanos();
        self.frame_intervals_ns.push(
            u64::try_from(interval_ns)
                .expect("a single frame interval must fit in u64 nanoseconds"),
        );

        if self.frame_intervals_ns.len() == self.measure_frames {
            self.result = Some(summarize(
                &mut self.frame_intervals_ns,
                self.warmup_frames,
                self.environment_enabled,
                self.replacement_material.is_none(),
                self.draw_model,
                self.gpu_driven,
            ));
        }

        true
    }

    #[cfg(feature = "rdg_bench_support")]
    fn record_phase_timings(&mut self, diagnostics: &RdgFrameDiagnostics) {
        for (index, phase) in RdgFramePhase::ALL.into_iter().enumerate() {
            let slot = &mut self.phase_elapsed_ns[index];
            *slot = slot.saturating_add(u128::from(diagnostics.phase_elapsed_ns(phase)));
        }
        for (index, phase) in RdgDetailPhase::ALL.into_iter().enumerate() {
            let slot = &mut self.detail_elapsed_ns[index];
            *slot = slot.saturating_add(u128::from(diagnostics.detail_elapsed_ns(phase)));
        }
        self.phase_timing_samples = self.phase_timing_samples.saturating_add(1);
    }
}

impl AppHandler for HelmetGltfFixedProbe {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let warmup_frames = read_frame_count(WARMUP_FRAMES_ENV, DEFAULT_WARMUP_FRAMES, true);
        let measure_frames = read_frame_count(MEASURE_FRAMES_ENV, DEFAULT_MEASURE_FRAMES, false);
        let environment_enabled = !read_bool(DISABLE_ENV_ENV);
        let untextured = read_bool(UNTEXTURED_ENV);
        let draw_model = !read_bool(NO_DRAW_ENV);
        let gpu_driven = read_gpu_driven_mode();

        let env_texture_handle = environment_enabled.then(|| {
            let map_path = format!("{ASSET_PATH}envs/royal_esplanade_2k.hdr.jpg");
            engine
                .assets
                .load_texture(map_path, ColorSpace::Srgb, false)
        });
        let replacement_material = untextured.then(|| {
            engine
                .assets
                .materials
                .add(Material::from(PhysicalMaterial::new(Vec4::ONE)))
        });

        let scene = engine.scene_manager.create_active();
        scene.environment.set_env_map(env_texture_handle);
        scene.add_light(Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0));

        let helmet_source = format!("{ASSET_PATH}DamagedHelmet/glTF/DamagedHelmet.gltf");
        println!("Loading glTF model from: {helmet_source}");
        let helmet_prefab = engine.assets.load_gltf(helmet_source);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(0.0, 0.0, 3.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(camera);

        Self {
            camera,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
            helmet_prefab,
            helmet_loaded: false,
            replacement_material,
            environment_enabled,
            draw_model,
            gpu_driven,
            warmup_frames,
            warmup_remaining: warmup_frames,
            measure_frames,
            last_update: None,
            frame_intervals_ns: Vec::with_capacity(measure_frames),
            #[cfg(feature = "rdg_bench_support")]
            phase_elapsed_ns: [0; RdgFramePhase::COUNT],
            #[cfg(feature = "rdg_bench_support")]
            detail_elapsed_ns: [0; RdgDetailPhase::COUNT],
            #[cfg(feature = "rdg_bench_support")]
            phase_timing_samples: 0,
            result: None,
            exit_requested: false,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        if self.result.is_some() {
            return;
        }

        // `helmet_gltf` clones the AssetServer façade once per update before
        // borrowing the active scene. Preserve that exact steady-state path.
        let assets = engine.assets.clone();
        if !self.helmet_loaded {
            let Some(scene) = engine.scene_manager.active_scene_mut() else {
                return;
            };

            if let Some(prefab) = assets.prefabs.get(self.helmet_prefab) {
                let root = self.draw_model.then(|| scene.instantiate(prefab.as_ref()));
                if self.draw_model
                    && let Some(material) = self.replacement_material
                {
                    for (_, mesh) in &mut scene.meshes {
                        mesh.material = material;
                    }
                }
                println!(
                    "Helmet ready: root={root:?}, warmup={}, measure={}, environment={}, textured={}, draw_model={}, gpu_driven={:?}",
                    self.warmup_frames,
                    self.measure_frames,
                    self.environment_enabled,
                    self.replacement_material.is_none(),
                    self.draw_model,
                    self.gpu_driven
                );
                self.helmet_loaded = true;
                // Establish the baseline after prefab instantiation, so loading and
                // scene construction cannot leak into the first warmup interval.
                self.last_update = Some(Instant::now());
                return;
            }

            if let Some(error) = assets.prefabs.get_error(self.helmet_prefab) {
                terminate_with_error(&format!("Failed to load helmet glTF: {error}"));
            }

            return;
        }

        // Keep the endpoint workload identical to `helmet_gltf`: reopen the
        // active scene, run OrbitControls even when input is idle, and retain
        // the one-second FPS/title cadence. These costs are common to both
        // revisions but are part of the production frame observed by users.
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };
        let _ = scene.edit_node_transform(self.camera, |transform| {
            self.controls
                .update(transform, &engine.input, 45.0, frame.dt);
        });
        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("glTF PBR Demo - FPS: {fps:.0}"));
        }

        let measured_interval = self.record_interval(Instant::now());
        #[cfg(feature = "rdg_bench_support")]
        if measured_interval && let Some(diagnostics) = engine.renderer.latest_rdg_diagnostics() {
            self.record_phase_timings(&diagnostics);
        }
        #[cfg(not(feature = "rdg_bench_support"))]
        let _ = measured_interval;
    }

    fn render(&mut self, engine: &mut Engine, window: &dyn Window) {
        let Some(result) = self.result else {
            engine.render_active_scene();
            return;
        };

        if !self.exit_requested {
            print_result(result);
            if let Some(diagnostics) = engine.renderer.latest_rdg_diagnostics() {
                print_scene_submission_diagnostics(&diagnostics);
            }
            #[cfg(feature = "rdg_bench_support")]
            print_phase_timings(
                &self.phase_elapsed_ns,
                &self.detail_elapsed_ns,
                self.phase_timing_samples,
            );
            self.exit_requested = true;

            if !request_graceful_exit(window) {
                // `process::exit` does not run destructors, so it is only the
                // fallback where the window backend offers no close request.
                std::process::exit(0);
            }
        }
    }
}

fn read_frame_count(name: &str, default: usize, allow_zero: bool) -> usize {
    let Ok(value) = std::env::var(name) else {
        return default;
    };

    let parsed = value
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("{name} must be an unsigned integer, got {value:?}"));
    assert!(allow_zero || parsed > 0, "{name} must be greater than zero");
    parsed
}

fn read_bool(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn read_gpu_driven_mode() -> GpuDrivenMode {
    let Ok(value) = std::env::var(GPU_DRIVEN_ENV) else {
        return GpuDrivenMode::Off;
    };
    match value.to_ascii_lowercase().as_str() {
        "off" => GpuDrivenMode::Off,
        "auto" => GpuDrivenMode::Auto,
        "required" => GpuDrivenMode::Required,
        _ => panic!("{GPU_DRIVEN_ENV} must be off, auto, or required; got {value:?}"),
    }
}

fn summarize(
    intervals_ns: &mut [u64],
    warmup_frames: usize,
    environment_enabled: bool,
    textured: bool,
    draw_model: bool,
    gpu_driven: GpuDrivenMode,
) -> ProbeResult {
    let measured_frames = intervals_ns.len();
    assert!(
        measured_frames > 0,
        "the probe requires at least one sample"
    );

    // Total elapsed / N is the primary endpoint because it matches the
    // production FPS counter's arithmetic-mean throughput. Sorting happens
    // only after the total is captured and outside the measured frame loop.
    let total_ns: u128 = intervals_ns.iter().map(|&value| u128::from(value)).sum();
    let mean_ns = total_ns as f64 / measured_frames as f64;

    intervals_ns.sort_unstable();
    let middle = measured_frames / 2;
    let median_ns = if measured_frames.is_multiple_of(2) {
        (intervals_ns[middle - 1] as f64 + intervals_ns[middle] as f64) * 0.5
    } else {
        intervals_ns[middle] as f64
    };
    let p95_index = measured_frames.saturating_mul(95).div_ceil(100) - 1;
    let p95_ns = intervals_ns[p95_index];
    let p99_index = measured_frames.saturating_mul(99).div_ceil(100) - 1;
    let p99_ns = intervals_ns[p99_index];
    let max_ns = intervals_ns[measured_frames - 1];

    ProbeResult {
        warmup_frames,
        measured_frames,
        total_ns,
        median_ns,
        p95_ns,
        p99_ns,
        max_ns,
        mean_ns,
        environment_enabled,
        textured,
        draw_model,
        gpu_driven,
    }
}

fn print_result(result: ProbeResult) {
    let median_us = result.median_ns / 1_000.0;
    let p95_us = result.p95_ns as f64 / 1_000.0;
    let p99_us = result.p99_ns as f64 / 1_000.0;
    let max_us = result.max_ns as f64 / 1_000.0;
    let mean_us = result.mean_ns / 1_000.0;
    let total_ms = result.total_ns as f64 / 1_000_000.0;
    let median_fps = 1_000_000_000.0 / result.median_ns;
    let p95_fps = 1_000_000_000.0 / result.p95_ns as f64;
    let p99_fps = 1_000_000_000.0 / result.p99_ns as f64;
    let mean_fps = 1_000_000_000.0 / result.mean_ns;

    let stdout = io::stdout();
    let mut output = stdout.lock();
    writeln!(
        output,
        "HELMET_FIXED_PROBE warmup={} frames={} environment={} textured={} draw_model={} gpu_driven={:?} total_ms={total_ms:.3} mean_us={mean_us:.3} median_us={median_us:.3} p95_us={p95_us:.3} p99_us={p99_us:.3} max_us={max_us:.3} mean_fps={mean_fps:.2} median_fps={median_fps:.2} p95_fps={p95_fps:.2} p99_fps={p99_fps:.2}",
        result.warmup_frames,
        result.measured_frames,
        result.environment_enabled,
        result.textured,
        result.draw_model,
        result.gpu_driven
    )
    .expect("failed to write helmet probe result");
    output.flush().expect("failed to flush helmet probe result");
}

fn print_scene_submission_diagnostics(diagnostics: &RdgFrameDiagnostics) {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let policy = diagnostics.gpu_driven_policy();
    writeln!(
        output,
        "HELMET_FIXED_PROBE_POLICY extracted={} supported={} retained={} bins={} dirty={} shadow_d2_views={} shadow_cube_views={} instance_span={} batch_span={} cohort_min={} indexed_primitives={} exact={}",
        policy.extracted_item_count(),
        policy.supported_opaque_count(),
        policy.retained_candidate_count(),
        policy.retained_bin_count(),
        policy.dirty_candidate_count(),
        policy.shadow_2d_view_count(),
        policy.shadow_cube_view_count(),
        policy.instance_slot_span(),
        policy.batch_slot_span(),
        policy.minimum_cohort_members(),
        policy.indexed_primitive_count(),
        policy.exact_evaluated(),
    )
    .expect("failed to write helmet probe policy diagnostics");
    for family in RdgSceneSubmissionFamily::ALL {
        let snapshot = diagnostics.scene_submission_family(family);
        writeln!(
            output,
            "HELMET_FIXED_PROBE_SUBMISSION family={family:?} selected={:?} reason={:?} live={:?} live_cpu={} live_gpu={}",
            snapshot.selected_backend(),
            snapshot.selection_reason(),
            snapshot.live_backend(),
            snapshot.live_cpu_submissions(),
            snapshot.live_gpu_submissions(),
        )
        .expect("failed to write helmet probe submission diagnostics");
    }
    output
        .flush()
        .expect("failed to flush helmet probe submission diagnostics");
}

#[cfg(feature = "rdg_bench_support")]
fn print_phase_timings(
    phase_elapsed_ns: &[u128; RdgFramePhase::COUNT],
    detail_elapsed_ns: &[u128; RdgDetailPhase::COUNT],
    samples: usize,
) {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for (index, phase) in RdgFramePhase::ALL.into_iter().enumerate() {
        let mean_us = if samples == 0 {
            0.0
        } else {
            phase_elapsed_ns[index] as f64 / samples as f64 / 1_000.0
        };
        writeln!(
            output,
            "HELMET_FIXED_PROBE_PHASE phase={phase:?} samples={samples} mean_us={mean_us:.3}"
        )
        .expect("failed to write helmet probe phase timing");
    }
    for (index, phase) in RdgDetailPhase::ALL.into_iter().enumerate() {
        let mean_us = if samples == 0 {
            0.0
        } else {
            detail_elapsed_ns[index] as f64 / samples as f64 / 1_000.0
        };
        writeln!(
            output,
            "HELMET_FIXED_PROBE_DETAIL phase={phase:?} samples={samples} mean_us={mean_us:.3}"
        )
        .expect("failed to write helmet probe detail timing");
    }
    output
        .flush()
        .expect("failed to flush helmet probe phase timings");
}

fn terminate_with_error(message: &str) -> ! {
    let stderr = io::stderr();
    let mut output = stderr.lock();
    let _ = writeln!(output, "HELMET_FIXED_PROBE_ERROR {message}");
    let _ = output.flush();
    std::process::exit(2);
}

#[cfg(target_os = "windows")]
fn request_graceful_exit(window: &dyn Window) -> bool {
    use std::ffi::c_void;

    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    const WM_CLOSE: u32 = 0x0010;

    #[link(name = "user32")]
    unsafe extern "system" {
        #[link_name = "PostMessageW"]
        fn post_message_w(window: *mut c_void, message: u32, w_param: usize, l_param: isize)
        -> i32;
    }

    let Some(window) = window
        .as_any()
        .downcast_ref::<myth_app::winit::WinitWindow>()
    else {
        return false;
    };
    let Ok(handle) = window.window_handle() else {
        return false;
    };
    let RawWindowHandle::Win32(handle) = handle.as_raw() else {
        return false;
    };

    // SAFETY: the HWND comes from the live winit window. Posting WM_CLOSE does
    // not retain the pointer; winit owns the window until it handles the event.
    unsafe { post_message_w(handle.hwnd.get() as *mut c_void, WM_CLOSE, 0, 0) != 0 }
}

#[cfg(not(target_os = "windows"))]
fn request_graceful_exit(_window: &dyn Window) -> bool {
    false
}

#[myth::main]
fn main() -> myth::Result<()> {
    let gpu_driven = read_gpu_driven_mode();
    App::new()
        .with_title("Myth Helmet Fixed Probe")
        .with_settings(RendererSettings {
            path: RenderPath::BasicForward,
            vsync: false,
            gpu_driven,
            ..Default::default()
        })
        .run::<HelmetGltfFixedProbe>()
}
