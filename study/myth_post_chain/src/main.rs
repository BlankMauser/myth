//! Myth side of the Myth vs LunarFenrir study (PROTOCOL.md v3).
//!
//! Runs Myth's own TAA -> CAS -> bloom -> tone-map -> FXAA chain on the
//! protocol's deterministic input (`Input_Gen`, feature `study_post_chain`
//! in myth_render), headless, one frame in flight.
//!
//! ```text
//! myth_post_chain --mode c|g|capture|latency --width W --height H --out DIR
//!                 [--warmup 100] [--frames 1000] [--capture-frames 0,1,2]
//!                 [--fxaa-template PATH]
//! ```
//! - c: per-frame cpu_ms / submit_to_done_ms / frame_ms (no timestamps).
//! - g: per-pass GPU ms from Myth's RDG timestamp recorder (needs the
//!   `gpu_timing` feature build); CPU columns are written but not headline.
//! - capture: raw stage captures (protocol section 5).
//! - latency: configuration L (frame 16, first frame, no history), reads
//!   back the final image and prints its sha256 and timings as JSON.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use myth_app::Engine;
use myth_render::graph::passes::study;
use myth_render::{RendererInitConfig, RendererSettings};
use myth_resources::{AntiAliasingMode, FxaaSettings, TaaSettings};
use myth_scene::camera::Camera;
use sha2::{Digest, Sha256};

/// CAS sharpness (Myth default 0.5). Build trial B2 edits this line.
const CAS_SHARPNESS: f32 = 0.5;

struct Args {
    mode: String,
    width: u32,
    height: u32,
    out: PathBuf,
    warmup: u32,
    frames: u32,
    capture_frames: Vec<u32>,
    fxaa_template: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args {
        mode: "c".into(),
        width: 1920,
        height: 1080,
        out: PathBuf::from("."),
        warmup: 100,
        frames: 1000,
        capture_frames: vec![0, 1, 2, 15, 16, 17, 512, 513, 1099],
        fxaa_template: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("missing value for {k}"));
        match k.as_str() {
            "--mode" => a.mode = v(),
            "--width" => a.width = v().parse().expect("width"),
            "--height" => a.height = v().parse().expect("height"),
            "--out" => a.out = PathBuf::from(v()),
            "--warmup" => a.warmup = v().parse().expect("warmup"),
            "--frames" => a.frames = v().parse().expect("frames"),
            "--capture-frames" => {
                a.capture_frames = v().split(',').map(|s| s.parse().expect("frame")).collect();
            }
            "--fxaa-template" => a.fxaa_template = Some(PathBuf::from(v())),
            other => panic!("unknown argument {other}"),
        }
    }
    a
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Engine with an empty scene and a TAA_FXAA camera; bloom on, the rest
/// Myth defaults (Neutral tone mapping, CAS 0.5, feedback 0.9, FXAA High).
fn setup(args: &Args) -> Engine {
    let mut engine = Engine::new(
        RendererInitConfig::default(),
        RendererSettings {
            vsync: false,
            ..RendererSettings::default()
        },
    );
    pollster::block_on(engine.init_headless(args.width, args.height, None))
        .expect("headless init failed");
    if let Some(path) = &args.fxaa_template {
        let src = std::fs::read_to_string(path).expect("read fxaa template");
        engine
            .renderer
            .register_shader_template("entry/post_process/fxaa", &src);
    }
    let scene = engine.scene_manager.create_active();
    scene.bloom.set_enabled(true);
    let mut camera = Camera::new_perspective(
        45.0,
        args.width as f32 / args.height as f32,
        0.1,
    );
    camera.set_aa_mode(AntiAliasingMode::TAA_FXAA(
        TaaSettings {
            // Myth's default (0.5), explicit so build-time trial B2 can edit it.
            sharpen_intensity: CAS_SHARPNESS,
            ..TaaSettings::default()
        },
        FxaaSettings::default(),
    ));
    let cam = scene.add_camera(camera);
    scene.active_camera = Some(cam);
    engine
}

fn wait_idle(engine: &Engine) {
    engine
        .renderer
        .device()
        .expect("device")
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
}

/// One protocol frame: returns (cpu_ms, submit_to_done_ms).
fn frame(engine: &mut Engine, f: u32) -> (f64, f64) {
    engine.update(1.0 / 60.0);
    study::GEN_FRAME.store(f, Ordering::Relaxed);
    let t0 = Instant::now();
    let rendered = engine.render_active_scene();
    let t1 = Instant::now();
    assert!(rendered, "Myth skipped frame {f}");
    wait_idle(engine);
    let t2 = Instant::now();
    (ms(t1 - t0), ms(t2 - t1))
}

/// Maps every buffer of the last captured frame; returns (stage, tag, tight bytes).
fn take_captures(engine: &Engine) -> Vec<(&'static str, &'static str, Vec<u8>)> {
    let device = engine.renderer.device().expect("device");
    let bufs: Vec<_> = std::mem::take(&mut *study::CAPTURED.lock());
    for b in &bufs {
        b.buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    }
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    bufs.iter()
        .map(|b| {
            let data = b.buffer.slice(..).get_mapped_range();
            let row = (b.width * b.bytes_per_pixel) as usize;
            let mut tight = Vec::with_capacity(row * b.height as usize);
            for y in 0..b.height as usize {
                let start = y * b.padded_bytes_per_row as usize;
                tight.extend_from_slice(&data[start..start + row]);
            }
            drop(data);
            b.buffer.unmap();
            (b.stage, b.tag, tight)
        })
        .collect()
}

fn sha256_hex(data: &[u8]) -> String {
    let mut s = String::new();
    for b in Sha256::digest(data) {
        write!(s, "{b:02x}").unwrap();
    }
    s
}

#[cfg(feature = "gpu_timing")]
fn latest_timings(engine: &Engine) -> Option<(u64, f64, Vec<(&'static str, f64)>)> {
    engine.renderer.latest_rdg_gpu_timings().map(|t| {
        (
            t.renderer_frame(),
            t.gpu_total_ms(),
            t.passes().iter().map(|p| (p.name(), p.duration_ms())).collect(),
        )
    })
}

#[cfg(not(feature = "gpu_timing"))]
fn latest_timings(_engine: &Engine) -> Option<(u64, f64, Vec<(&'static str, f64)>)> {
    None
}

fn enable_timings(engine: &mut Engine) -> bool {
    #[cfg(feature = "gpu_timing")]
    {
        engine.renderer.rdg_gpu_timings_supported()
            && engine.renderer.set_rdg_gpu_timing_sample_interval(1)
    }
    #[cfg(not(feature = "gpu_timing"))]
    {
        let _ = engine;
        false
    }
}

fn adapter_info() -> serde_json::Value {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapters: Vec<_> = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .map(|a| {
            let i = a.get_info();
            serde_json::json!({
                "name": i.name, "backend": format!("{:?}", i.backend),
                "device_type": format!("{:?}", i.device_type),
                "driver": i.driver, "driver_info": i.driver_info,
            })
        })
        .collect();
    serde_json::json!(adapters)
}

fn write_json(path: &Path, v: &serde_json::Value) {
    std::fs::write(path, serde_json::to_string_pretty(v).unwrap()).expect("write json");
}

fn main() {
    let args = parse_args();
    std::fs::create_dir_all(&args.out).expect("create out dir");
    let t_start = Instant::now();
    let mut engine = setup(&args);
    let init_ms = ms(t_start.elapsed());
    let (w, h) = (args.width, args.height);

    match args.mode.as_str() {
        "c" | "g" => {
            let timed = args.mode == "g";
            let timestamps = if timed { enable_timings(&mut engine) } else { false };
            assert!(!timed || timestamps, "G run needs the gpu_timing build and timestamp support");
            let total = args.warmup + args.frames;
            let mut cpu = Vec::with_capacity(total as usize);
            let mut gpu: BTreeMap<u64, (f64, Vec<(&'static str, f64)>)> = BTreeMap::new();
            let mut renderer_frame_of = Vec::with_capacity(total as usize);
            let mut first_frame_ms = 0.0;
            for f in 0..total {
                renderer_frame_of.push(engine.frame_count() + 1);
                let (c, s) = frame(&mut engine, f);
                if f == 0 {
                    first_frame_ms = c + s;
                }
                cpu.push((c, s));
                if let Some((rf, total_ms, passes)) = latest_timings(&engine) {
                    gpu.entry(rf).or_insert((total_ms, passes));
                }
            }
            // Flush: the recorder publishes frame N at frame N+1's begin.
            // Extra frames are never part of the stats; in C runs they also
            // sample the executed pass list (timestamps on only there).
            let c_pass_probe = !timed && enable_timings(&mut engine);
            for k in 0..3 {
                frame(&mut engine, total + k);
                if let Some((rf, total_ms, passes)) = latest_timings(&engine) {
                    gpu.entry(rf).or_insert((total_ms, passes));
                }
            }
            // Executed pass list (from Myth's graph, via the recorder).
            let pass_list: Vec<&str> = gpu
                .values()
                .last()
                .map(|(_, p)| p.iter().map(|(n, _)| *n).collect())
                .unwrap_or_default();
            let mut csv = String::from("frame,cpu_ms,submit_to_done_ms,frame_ms");
            if timed {
                for n in &pass_list {
                    write!(csv, ",{n}").unwrap();
                }
                csv.push_str(",chain_gpu_ms,extra_gpu_ms,span_ms");
            }
            csv.push('\n');
            let mut missing_gpu = 0u32;
            for f in args.warmup..total {
                let (c, s) = cpu[f as usize];
                write!(csv, "{f},{c:.6},{s:.6},{:.6}", c + s).unwrap();
                if timed {
                    let rf = renderer_frame_of[f as usize];
                    if let Some((span, passes)) = gpu.get(&rf) {
                        let map: BTreeMap<&str, f64> = passes.iter().copied().collect();
                        let mut chain = 0.0;
                        let mut extra = 0.0;
                        for n in &pass_list {
                            let v = map.get(n).copied().unwrap_or(f64::NAN);
                            write!(csv, ",{v:.6}").unwrap();
                            if *n == "Input_Gen" {
                            } else if is_chain_pass(n) {
                                chain += v;
                            } else {
                                extra += v;
                            }
                        }
                        write!(csv, ",{chain:.6},{extra:.6},{span:.6}").unwrap();
                    } else {
                        missing_gpu += 1;
                        for _ in &pass_list {
                            csv.push_str(",");
                        }
                        csv.push_str(",,,");
                    }
                }
                csv.push('\n');
            }
            std::fs::write(args.out.join("frames.csv"), csv).expect("write csv");
            write_json(
                &args.out.join("harness.json"),
                &serde_json::json!({
                    "side": "myth", "mode": args.mode, "width": w, "height": h,
                    "warmup": args.warmup, "frames": args.frames,
                    "init_ms": init_ms, "first_frame_ms": first_frame_ms,
                    "timestamps": timestamps, "pass_list_from_probe": c_pass_probe,
                    "executed_passes": pass_list,
                    "chain_passes": pass_list.iter().filter(|n| is_chain_pass(n)).collect::<Vec<_>>(),
                    "extra_passes": pass_list.iter().filter(|n| !is_chain_pass(n) && **n != "Input_Gen").collect::<Vec<_>>(),
                    "gpu_samples_missing": missing_gpu,
                    "adapters": adapter_info(),
                }),
            );
        }
        "capture" => {
            study::CAPTURE_MODE.store(true, Ordering::Relaxed);
            let last = *args.capture_frames.iter().max().expect("capture frames");
            let mut sums = String::new();
            for f in 0..=last {
                let want = args.capture_frames.contains(&f);
                study::CAPTURE_THIS_FRAME.store(want, Ordering::Relaxed);
                frame(&mut engine, f);
                if want {
                    for (stage, tag, data) in take_captures(&engine) {
                        let name = format!("myth_{w}x{h}_f{f:04}_{stage}.{tag}");
                        std::fs::write(args.out.join(&name), &data).expect("write capture");
                        writeln!(sums, "{}  {name}", sha256_hex(&data)).unwrap();
                    }
                }
            }
            std::fs::write(args.out.join("SHA256SUMS"), sums).expect("write sums");
        }
        "latency" => {
            // Configuration L: generator frame 16, first frame (no history).
            study::CAPTURE_MODE.store(true, Ordering::Relaxed);
            study::CAPTURE_STAGES.store(1 << 7, Ordering::Relaxed); // fxaa only
            study::CAPTURE_THIS_FRAME.store(true, Ordering::Relaxed);
            let t_frame = Instant::now();
            frame(&mut engine, 16);
            let caps = take_captures(&engine);
            let (_, _, data) = caps.into_iter().next().expect("fxaa capture");
            let hash = sha256_hex(&data);
            let t_done = Instant::now();
            let unix_done = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            std::fs::write(args.out.join(format!("myth_{w}x{h}_L_fxaa.rgba8")), &data)
                .expect("write");
            let line = serde_json::json!({
                "sha256": hash, "init_ms": init_ms,
                "frame_and_readback_ms": ms(t_done - t_frame),
                "process_ms": ms(t_done - t_start), "unix_done": unix_done,
            });
            println!("{line}");
            std::io::stdout().flush().ok();
        }
        other => panic!("unknown mode {other}"),
    }
}

/// Protocol passes 1-16 (Myth names).
fn is_chain_pass(name: &str) -> bool {
    matches!(name, "TAA_Resolve" | "CAS_Pass" | "ToneMap_Pass" | "FXAA_Pass")
        || name.starts_with("Bloom_")
}
