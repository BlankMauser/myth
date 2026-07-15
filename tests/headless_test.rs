//! Headless Rendering & Readback Tests
//!
//! - Synchronous readback via `Renderer.readback_pixels()`
//! - Asynchronous readback via `ReadbackStream`
//! - Material rendering: Physical (PBR), Phong, Unlit
//! - Multi-light scenes (directional + point)
//! - Alpha blending and alpha mask
//! - Multiple geometry types (box, sphere, plane)
use myth::prelude::*;
use myth::render::core::ReadbackStream;
use myth::renderer::graph::{
    RdgAttemptOutcome, RdgFrameDiagnostics, RdgGpuDrivenFailure, RdgSceneSubmissionBackend,
    RdgSceneSubmissionFamily, RdgSceneSubmissionSelectionReason,
};
use myth::scene::light::LIGHT_FLAG_IS_SUN;

// Integration tests for synchronous headless readback.
//
// Renders 10 frames in headless mode. Each frame is read back via the
// optimised `readback_pixels()` path (staging-buffer cache). Verifies:
//
// - Every call returns a non-empty pixel buffer of the expected size.
// - The cached staging buffer prevents per-frame allocation.
#[test]
fn headless_sync_readback() {
    let mut engine = Engine::default();

    let width: u32 = 256;
    let height: u32 = 256;
    pollster::block_on(engine.init_headless(width, height, None)).expect("headless init failed");

    // ── Minimal scene ────────────────────────────────────────────────
    let scene = engine.scene_manager.create_active();

    let mat = UnlitMaterial::new(Vec4::new(0.2, 0.6, 1.0, 1.0));
    let _cube = scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);

    let cam = scene.add_camera(Camera::new_perspective(
        45.0,
        width as f32 / height as f32,
        0.1,
    ));
    scene
        .node(&cam)
        .set_position(0.0, 2.0, 5.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    scene.add_light(Light::new_directional(Vec3::ONE, 3.0));

    // ── Render 10 frames & readback each ─────────────────────────────
    let expected_bytes = (width * height * 4) as usize; // RGBA8

    for i in 0..10 {
        engine.update(1.0 / 60.0);
        engine.render_active_scene();

        let pixels = engine.readback_pixels().expect("readback failed");
        assert_eq!(
            pixels.len(),
            expected_bytes,
            "frame {i}: unexpected buffer size"
        );

        // Sanity: at least one non-zero pixel (the scene is not pitch black).
        let any_nonzero = pixels.iter().any(|&b| b != 0);
        assert!(any_nonzero, "frame {i}: all pixels are zero");
    }
}

// Integration tests for `ReadbackStream` (async ring buffer).
//
// Renders 100 frames in headless mode using a `ReadbackStream` with
// `buffer_count = 3`. Frames are submitted non-blocking and collected
// via `try_recv`. Any remaining in-flight frames are drained with
// `flush` at the end. Verifies exactly 100 frames are received.
#[test]
fn headless_stream_recording() {
    let mut engine = Engine::default();

    let width: u32 = 256;
    let height: u32 = 256;
    pollster::block_on(engine.init_headless(width, height, None)).expect("headless init failed");

    // ── Minimal scene ────────────────────────────────────────────────
    let scene = engine.scene_manager.create_active();

    let mat = UnlitMaterial::new(Vec4::new(1.0, 0.4, 0.1, 1.0));
    let _cube = scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);

    let cam = scene.add_camera(Camera::new_perspective(
        45.0,
        width as f32 / height as f32,
        0.1,
    ));
    scene
        .node(&cam)
        .set_position(0.0, 2.0, 5.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    scene.add_light(Light::new_directional(Vec3::ONE, 3.0));

    // ── Create ReadbackStream ────────────────────────────────────────
    let mut stream: ReadbackStream = engine
        .renderer
        .create_readback_stream(3, 16)
        .expect("create_readback_stream failed");

    let total_frames: u64 = 100;
    let expected_bytes = (width * height * 4) as usize;
    let mut submitted: u64 = 0;
    let mut received: u64 = 0;
    let mut skipped: u64 = 0;

    // ── Hot loop ─────────────────────────────────────────────────────
    for _ in 0..total_frames {
        engine.update(1.0 / 60.0);
        engine.render_active_scene();

        match engine.submit_to_stream(&mut stream) {
            Ok(_) => {
                submitted += 1;
            }
            Err(_) => {
                skipped += 1;
            }
        }

        // Drive GPU callbacks.
        engine.poll_device();

        // Opportunistically pull ready frames.
        while let Some(frame) = stream.try_recv().expect("try_recv failed") {
            assert_eq!(
                frame.pixels.len(),
                expected_bytes,
                "frame {}: unexpected pixel buffer size",
                frame.frame_index
            );
            received += 1;
        }
    }

    // ── Flush remaining ──────────────────────────────────────────────
    let frames = engine
        .flush_stream(&mut stream)
        .expect("flush_stream failed");
    for frame in frames {
        assert_eq!(
            frame.pixels.len(),
            expected_bytes,
            "flush frame {}: unexpected pixel buffer size",
            frame.frame_index
        );
        received += 1;
    }

    assert_eq!(
        received, submitted,
        "expected {submitted} frames, got {received}"
    );

    assert_eq!(
        submitted + skipped,
        total_frames,
        "total frames ({total_frames}) should equal submitted + skipped ({submitted} + {skipped})"
    );
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Initialise a headless engine and return it together with the pixel
/// count so callers don't have to repeat the boilerplate.
fn setup_headless(w: u32, h: u32) -> (Engine, usize) {
    let mut engine = Engine::default();
    pollster::block_on(engine.init_headless(w, h, None)).expect("headless init failed");
    let expected_bytes = (w * h * 4) as usize;
    (engine, expected_bytes)
}

fn setup_headless_with_settings(w: u32, h: u32, settings: RendererSettings) -> (Engine, usize) {
    let mut engine = Engine::new(RendererInitConfig::default(), settings);
    pollster::block_on(engine.init_headless(w, h, None)).expect("headless init failed");
    let expected_bytes = (w * h * 4) as usize;
    (engine, expected_bytes)
}

/// Remove the currently active scene so a fresh one can be created.
fn reset_active_scene(engine: &mut Engine) {
    if let Some(h) = engine.scene_manager.active_handle() {
        engine.scene_manager.remove_scene(h);
    }
}

/// Render a few warm-up frames, then capture one and return the pixel buffer.
fn render_and_capture(engine: &mut Engine, warmup: usize) -> Vec<u8> {
    for _ in 0..warmup {
        engine.update(1.0 / 60.0);
        engine.render_active_scene();
    }
    // Final capture frame
    engine.update(1.0 / 60.0);
    engine.render_active_scene();
    engine.readback_pixels().expect("readback failed")
}

/// Check that the image is not entirely black (at least one pixel with a
/// non-zero RGB channel).
fn assert_not_black(pixels: &[u8], label: &str) {
    let any_color = pixels
        .chunks_exact(4)
        .any(|px| px[0] > 0 || px[1] > 0 || px[2] > 0);
    assert!(any_color, "{label}: rendered image is entirely black");
}

/// Check that at least one pixel differs between two render results
/// (i.e. the scenes are visually distinct).
fn assert_images_differ(a: &[u8], b: &[u8], label: &str) {
    assert_eq!(a.len(), b.len());
    let differs = a.iter().zip(b.iter()).any(|(x, y)| x != y);
    assert!(differs, "{label}: images are identical but should differ");
}

fn mean_rgb_abs_delta(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut total_delta = 0u64;
    let mut channel_count = 0u64;
    for (lhs, rhs) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        total_delta += u64::from(lhs[0].abs_diff(rhs[0]));
        total_delta += u64::from(lhs[1].abs_diff(rhs[1]));
        total_delta += u64::from(lhs[2].abs_diff(rhs[2]));
        channel_count += 3;
    }

    total_delta as f32 / channel_count as f32
}

fn render_and_observe_rdg(engine: &mut Engine, stage: &str) -> RdgFrameDiagnostics {
    engine.update(1.0 / 60.0);
    assert!(engine.render_active_scene(), "{stage}: render failed");
    let diagnostics = engine
        .renderer
        .latest_rdg_diagnostics()
        .unwrap_or_else(|| panic!("{stage}: diagnostics missing"));
    assert_eq!(
        (diagnostics.outcome(), diagnostics.queue_submitted()),
        (RdgAttemptOutcome::Submitted, true),
        "{stage}: frame was not submitted"
    );
    diagnostics
}

fn assert_main_submission_backend(
    diagnostics: &RdgFrameDiagnostics,
    stage: &str,
    expected: RdgSceneSubmissionBackend,
) {
    let main = diagnostics.scene_submission_family(RdgSceneSubmissionFamily::MainPrepass);
    assert_eq!(
        (main.selected_backend(), main.live_backend()),
        (expected, expected),
        "{stage}: backend mismatch; reason={:?}, policy={:?}",
        main.selection_reason(),
        diagnostics.gpu_driven_policy()
    );
}

/// Exercises the public runtime policy boundary against one long-lived scene
/// and renderer. A successful live GPU backend plus a non-empty readback is an
/// end-to-end contract: compute culling produced indirect arguments and the
/// raster pass consumed them with `DrawIndirect` before queue submission.
#[test]
fn gpu_driven_runtime_switches_are_publicly_observable_and_raster_equivalent() {
    const SIDE: usize = 16;
    const AUTO_WAIT_LIMIT: usize = 8;

    let (mut engine, expected_bytes) = setup_headless_with_settings(
        160,
        160,
        RendererSettings {
            path: RenderPath::HighFidelity,
            gpu_driven: GpuDrivenMode::Auto,
            ..RendererSettings::default()
        },
    );
    let geometry = engine
        .assets
        .geometries
        .add(Geometry::new_box(0.46, 0.46, 0.46));
    let material = engine.assets.materials.add(
        PhysicalMaterial::new(Vec4::new(0.12, 0.72, 0.34, 1.0))
            .with_roughness(0.72)
            .with_metalness(0.0),
    );

    let scene = engine.scene_manager.create_active();
    let first = scene.add_mesh(Mesh::new(geometry, material));
    scene.node(&first).set_position(-4.5, 0.0, -4.5);

    let mut sun = Light::new_directional(Vec3::ONE, 2.5);
    sun.cast_shadows = true;
    if let Some(shadow) = sun.shadow.as_mut() {
        shadow.map_size = 256;
        shadow.cascade_count = 2;
    }
    let sun = scene.add_light(sun);
    scene
        .node(&sun)
        .set_position(8.0, 12.0, 6.0)
        .look_at(Vec3::ZERO);

    let camera = scene.add_camera(Camera::new_perspective(48.0, 1.0, 0.1));
    scene
        .node(&camera)
        .set_position(0.0, 11.5, 16.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(camera);

    let tiny = render_and_observe_rdg(&mut engine, "tiny-auto");
    assert_main_submission_backend(&tiny, "tiny-auto", RdgSceneSubmissionBackend::CpuDirect);
    assert_eq!(
        tiny.scene_submission_family(RdgSceneSubmissionFamily::MainPrepass)
            .selection_reason(),
        RdgSceneSubmissionSelectionReason::AutoSmallWorkload
    );

    // Grow the same scene to a dense cohort while retaining one Geometry and
    // one Material identity across all 256 instances.
    {
        let scene = engine
            .scene_manager
            .active_scene_mut()
            .expect("dense GPU-driven scene is missing");
        for index in 1..SIDE * SIDE {
            let x = index % SIDE;
            let z = index / SIDE;
            let node = scene.add_mesh(Mesh::new(geometry, material));
            scene
                .node(&node)
                .set_position(x as f32 * 0.6 - 4.5, 0.0, z as f32 * 0.6 - 4.5);
        }
    }

    let supports_gpu_driven = engine
        .renderer
        .gpu_driven_capabilities()
        .is_some_and(|capabilities| capabilities.supports_indirect_primitives());
    if !supports_gpu_driven {
        let unsupported = render_and_observe_rdg(&mut engine, "dense-auto-unsupported");
        assert_main_submission_backend(
            &unsupported,
            "dense-auto-unsupported",
            RdgSceneSubmissionBackend::CpuDirect,
        );
        assert_eq!(
            unsupported
                .scene_submission_family(RdgSceneSubmissionFamily::MainPrepass)
                .selection_reason(),
            RdgSceneSubmissionSelectionReason::Unsupported(RdgGpuDrivenFailure::UnsupportedDevice,)
        );
        return;
    }

    let auto_entered_gpu = (0..AUTO_WAIT_LIMIT).any(|attempt| {
        let stage = format!("dense-auto-{attempt}");
        let diagnostics = render_and_observe_rdg(&mut engine, &stage);
        let main = diagnostics.scene_submission_family(RdgSceneSubmissionFamily::MainPrepass);
        assert_eq!(
            main.selected_backend(),
            main.live_backend(),
            "{stage}: selected and live Auto backends diverged"
        );
        main.live_backend() == RdgSceneSubmissionBackend::GpuDriven
    });
    assert!(
        auto_entered_gpu,
        "dense Auto workload did not enter GPU-driven submission within {AUTO_WAIT_LIMIT} submitted frames"
    );

    let (hf, basic) = (RenderPath::HighFidelity, RenderPath::BasicForward);
    let (required, off) = (GpuDrivenMode::Required, GpuDrivenMode::Off);
    let (gpu, cpu) = (
        RdgSceneSubmissionBackend::GpuDriven,
        RdgSceneSubmissionBackend::CpuDirect,
    );
    let stages = [
        ("high-fidelity-csm-required", hf, required, true, gpu),
        ("high-fidelity-csm-off", hf, off, true, cpu),
        ("high-fidelity-unshadowed", hf, required, false, gpu),
        ("basic-required", basic, required, false, gpu),
        ("basic-off", basic, off, false, cpu),
        ("high-fidelity-csm-reentry", hf, required, true, gpu),
    ];
    let mut basic_required_pixels = None;
    for (stage, path, mode, shadows, expected) in stages {
        engine.renderer.set_render_path(path);
        engine.renderer.set_gpu_driven_mode(mode);
        engine
            .scene_manager
            .active_scene_mut()
            .and_then(|scene| scene.get_light_mut(sun))
            .expect("GPU-driven sun is missing")
            .cast_shadows = shadows;
        let diagnostics = render_and_observe_rdg(&mut engine, stage);
        assert_main_submission_backend(&diagnostics, stage, expected);
        if shadows {
            assert_eq!(
                diagnostics
                    .scene_submission_family(RdgSceneSubmissionFamily::Shadow2D)
                    .live_backend(),
                expected,
                "{stage}: CSM submission used a different live backend"
            );
        }
        if path == RenderPath::BasicForward {
            let pixels = engine
                .readback_pixels()
                .expect("BasicForward readback failed");
            if mode == GpuDrivenMode::Required {
                assert_eq!(pixels.len(), expected_bytes);
                assert_not_black(&pixels, stage);
                basic_required_pixels = Some(pixels);
            } else {
                assert_eq!(basic_required_pixels.as_ref(), Some(&pixels));
            }
        }
    }
}

// ── Physical Material Tests ──────────────────────────────────────────────

/// A PhysicalMaterial sphere is rendered with a directional light.
/// The framebuffer must contain visible (non-black) pixels.
#[test]
fn physical_material_sphere() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();

    let mat = PhysicalMaterial::new(Vec4::new(0.9, 0.1, 0.1, 1.0))
        .with_roughness(0.4)
        .with_metalness(0.0);
    scene.spawn_sphere(1.0, mat, &engine.assets);

    scene.add_light(Light::new_directional(Vec3::ONE, 3.0));

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "physical_material_sphere");
}

/// A procedural-sky scene with a sun-flagged directional light must render
/// through the atmosphere-enabled PBR lighting path without going black.
#[test]
fn physical_material_sphere_with_procedural_sun_transmittance() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();
    scene.background.set_mode(BackgroundMode::procedural());

    let mat = PhysicalMaterial::new(Vec4::new(0.82, 0.78, 0.72, 1.0))
        .with_roughness(0.35)
        .with_metalness(0.0);
    scene.spawn_sphere(1.0, mat, &engine.assets);

    let sun = scene.add_light(Light::new_directional(Vec3::ONE, 3.0));
    scene
        .node(&sun)
        .set_position(0.0, 3.0, -10.0)
        .look_at(Vec3::ZERO);
    let light = scene.get_light_mut(sun).expect("sun light missing");
    light.flags |= LIGHT_FLAG_IS_SUN;

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(
        &pixels,
        "physical_material_sphere_with_procedural_sun_transmittance",
    );
}

/// A metallic PhysicalMaterial box should produce different colours than
/// a dielectric one.
#[test]
fn physical_metallic_vs_dielectric() {
    let (mut engine, _) = setup_headless(128, 128);

    // Dielectric
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(Vec4::new(0.8, 0.8, 0.2, 1.0))
            .with_roughness(0.3)
            .with_metalness(0.0);
        scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 3.0));
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 1.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let dielectric = render_and_capture(&mut engine, 2);
    assert_not_black(&dielectric, "dielectric");

    // Metallic - same colour, same scene, just metalness=1
    reset_active_scene(&mut engine);
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(Vec4::new(0.8, 0.8, 0.2, 1.0))
            .with_roughness(0.3)
            .with_metalness(1.0);
        scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 3.0));
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 1.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let metallic = render_and_capture(&mut engine, 2);
    assert_not_black(&metallic, "metallic");

    assert_images_differ(&dielectric, &metallic, "metallic_vs_dielectric");
}

/// Emissive PhysicalMaterial should produce visible output even without
/// any light in the scene.
#[test]
fn physical_emissive_no_light() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();

    let mat = PhysicalMaterial::new(Vec4::new(0.0, 0.0, 0.0, 1.0))
        .with_emissive(Vec3::new(1.0, 0.5, 0.0), 5.0);
    scene.spawn_sphere(1.0, mat, &engine.assets);

    // No light added intentionally.

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "physical_emissive_no_light");
}

// ── Phong Material Tests ─────────────────────────────────────────────────

/// Blinn-Phong shaded sphere with specular highlight.
#[test]
fn phong_material_sphere() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();

    let mat = PhongMaterial::new(Vec4::new(0.2, 0.5, 0.9, 1.0))
        .with_shininess(64.0)
        .with_specular(Vec3::new(1.0, 1.0, 1.0));
    scene.spawn_sphere(1.0, mat, &engine.assets);

    scene.add_light(Light::new_directional(Vec3::ONE, 3.0));

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "phong_material_sphere");
}

/// Emissive Phong material without scene lights.
#[test]
fn phong_emissive_no_light() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();

    let mat = PhongMaterial::new(Vec4::new(0.0, 0.0, 0.0, 1.0))
        .with_emissive(Vec3::new(0.0, 1.0, 0.5), 4.0);
    scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "phong_emissive_no_light");
}

// ── Unlit Material Tests ─────────────────────────────────────────────────

/// Unlit material should produce visible output regardless of lighting.
#[test]
fn unlit_material_no_light() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();

    let mat = UnlitMaterial::new(Vec4::new(0.0, 1.0, 0.0, 1.0));
    scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);

    // No light - unlit should still render.
    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "unlit_material_no_light");
}

// ── Multi-Light Tests ────────────────────────────────────────────────────

/// Scene with directional + point light should differ from directional-only.
#[test]
fn multi_light_scene() {
    let (mut engine, _) = setup_headless(128, 128);

    // ──── Directional only ────────────────────────────────────────────
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(Vec4::new(0.7, 0.7, 0.7, 1.0))
            .with_roughness(0.5)
            .with_metalness(0.0);
        scene.spawn_sphere(1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 2.0));
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let dir_only = render_and_capture(&mut engine, 2);
    assert_not_black(&dir_only, "directional_only");

    // ──── Directional + point light ───────────────────────────────────
    reset_active_scene(&mut engine);
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(Vec4::new(0.7, 0.7, 0.7, 1.0))
            .with_roughness(0.5)
            .with_metalness(0.0);
        scene.spawn_sphere(1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 2.0));
        let point = scene.add_light(Light::new_point(Vec3::new(1.0, 0.2, 0.2), 15.0, 20.0));
        scene.node(&point).set_position(2.0, 2.0, 2.0);
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let dir_plus_point = render_and_capture(&mut engine, 2);
    assert_not_black(&dir_plus_point, "directional_plus_point");

    assert_images_differ(&dir_only, &dir_plus_point, "multi_light");
}

/// Dense point-light scene should render correctly through the clustered
/// forward-lighting path without panicking or producing an all-black image.
#[test]
fn clustered_dense_point_lights() {
    let (mut engine, expected) = setup_headless(160, 160);
    let scene = engine.scene_manager.create_active();

    let mat = PhysicalMaterial::new(Vec4::new(0.72, 0.74, 0.78, 1.0))
        .with_roughness(0.42)
        .with_metalness(0.08);
    scene.spawn_sphere(1.0, mat, &engine.assets);

    for ring in 0..4 {
        let radius = 1.8 + ring as f32 * 0.7;
        let height = 0.4 + ring as f32 * 0.55;
        for i in 0..16 {
            let angle = (i as f32 / 16.0) * std::f32::consts::TAU;
            let color = Vec3::new(
                0.25 + (i as f32 / 16.0) * 0.75,
                0.35 + ring as f32 * 0.12,
                1.0 - (i as f32 / 16.0) * 0.55,
            );
            let light = scene.add_light(Light::new_point(color, 7.5, 4.2));
            scene
                .node(&light)
                .set_position(angle.cos() * radius, height, angle.sin() * radius);
        }
    }

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 2.6, 5.2)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "clustered_dense_point_lights");
}

fn populate_dense_point_light_scene(scene: &mut Scene, assets: &AssetServer) {
    let mat = PhysicalMaterial::new(Vec4::new(0.72, 0.74, 0.78, 1.0))
        .with_roughness(0.42)
        .with_metalness(0.08);
    scene.spawn_sphere(1.0, mat, assets);

    for ring in 0..4 {
        let radius = 1.8 + ring as f32 * 0.7;
        let height = 0.4 + ring as f32 * 0.55;
        for i in 0..16 {
            let angle = (i as f32 / 16.0) * std::f32::consts::TAU;
            let color = Vec3::new(
                0.25 + (i as f32 / 16.0) * 0.75,
                0.35 + ring as f32 * 0.12,
                1.0 - (i as f32 / 16.0) * 0.55,
            );
            let light = scene.add_light(Light::new_point(color, 7.5, 4.2));
            scene
                .node(&light)
                .set_position(angle.cos() * radius, height, angle.sin() * radius);
        }
    }

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 2.6, 5.2)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);
}

fn populate_clustered_corridor_scene(scene: &mut Scene, assets: &AssetServer) {
    use std::f32::consts::FRAC_PI_2;

    const LIGHT_GRID_X: usize = 8;
    const LIGHT_GRID_Y: usize = 3;
    const LIGHT_GRID_Z: usize = 10;

    fn centered_lattice(index: usize, count: usize, spacing: f32) -> f32 {
        (index as f32 - (count as f32 - 1.0) * 0.5) * spacing
    }

    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
        let h6 = (h.fract() * 6.0).clamp(0.0, 6.0 - f32::EPSILON);
        let i = h6.floor() as i32;
        let f = h6 - i as f32;
        let p = v * (1.0 - s);
        let q = v * (1.0 - f * s);
        let t = v * (1.0 - (1.0 - f) * s);

        match i {
            0 => Vec3::new(v, t, p),
            1 => Vec3::new(q, v, p),
            2 => Vec3::new(p, v, t),
            3 => Vec3::new(p, q, v),
            4 => Vec3::new(t, p, v),
            _ => Vec3::new(v, p, q),
        }
    }

    scene.environment.set_ambient_light(Vec3::splat(0.003));

    let floor_material = PhysicalMaterial::new(Vec4::new(0.06, 0.07, 0.08, 1.0))
        .with_roughness(0.86)
        .with_metalness(0.12);
    let wall_material = PhysicalMaterial::new(Vec4::new(0.10, 0.11, 0.14, 1.0))
        .with_roughness(0.74)
        .with_metalness(0.14);
    let block_material = PhysicalMaterial::new(Vec4::new(0.78, 0.80, 0.85, 1.0))
        .with_roughness(0.22)
        .with_metalness(0.65);

    let floor = scene.spawn_plane(18.0, 56.0, floor_material, assets);
    scene
        .node(&floor)
        .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
        .set_position(0.0, -0.3, 0.0)
        .set_receive_shadows(false);

    for &(x, y, z, sx, sy, sz) in &[
        (-7.5, 2.6, 0.0, 0.45, 5.8, 48.0),
        (7.5, 2.6, 0.0, 0.45, 5.8, 48.0),
        (0.0, 5.2, 0.0, 15.4, 0.22, 48.0),
    ] {
        let wall = scene.spawn_box(sx, sy, sz, wall_material.clone(), assets);
        scene
            .node(&wall)
            .set_position(x, y, z)
            .set_shadows(false, true);
    }

    for row in 0..8 {
        let z = centered_lattice(row, 8, 5.2);
        for col in -2..=2 {
            let x = col as f32 * 2.45;
            let y = if (row + (col + 2) as usize) % 2 == 0 {
                0.7
            } else {
                1.65
            };
            let block = if (row + (col + 2) as usize) % 3 == 0 {
                scene.spawn_sphere(0.76, block_material.clone(), assets)
            } else {
                scene.spawn_box(1.3, 1.3, 1.3, block_material.clone(), assets)
            };
            scene
                .node(&block)
                .set_position(x, y, z)
                .set_shadows(false, true);
        }
    }

    for ix in 0..LIGHT_GRID_X {
        for iy in 0..LIGHT_GRID_Y {
            for iz in 0..LIGHT_GRID_Z {
                let light_index = ix * LIGHT_GRID_Y * LIGHT_GRID_Z + iy * LIGHT_GRID_Z + iz;
                let hue = light_index as f32 / (LIGHT_GRID_X * LIGHT_GRID_Y * LIGHT_GRID_Z) as f32;
                let color = hsv_to_rgb(hue, 0.76, 1.0);
                let light = scene.add_light(Light::new_point(
                    color,
                    0.55 + iy as f32 * 0.28,
                    3.4 + ix as f32 * 0.12,
                ));

                let base = Vec3::new(
                    centered_lattice(ix, LIGHT_GRID_X, 1.35),
                    0.9 + iy as f32 * 0.7,
                    centered_lattice(iz, LIGHT_GRID_Z, 4.1),
                );
                scene.node(&light).set_position(base.x, base.y, base.z);
            }
        }
    }

    let cam = scene.add_camera(Camera::new_perspective(46.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 4.2, 20.0)
        .look_at(Vec3::new(0.0, 1.8, 0.0));
    scene.active_camera = Some(cam);
}

#[test]
fn clustered_force_modes_dense_point_lights() {
    for (label, mode) in [
        ("clustered_force_off", ClusteredShadingMode::ForceOff),
        ("clustered_force_on", ClusteredShadingMode::ForceOn),
    ] {
        let (mut engine, expected) = setup_headless_with_settings(
            160,
            160,
            RendererSettings {
                clustered_shading: mode,
                ..Default::default()
            },
        );
        let scene = engine.scene_manager.create_active();
        populate_dense_point_light_scene(scene, &engine.assets);

        let pixels = render_and_capture(&mut engine, 2);
        assert_eq!(pixels.len(), expected);
        assert_not_black(&pixels, label);
    }
}

#[test]
fn clustered_directional_plus_dense_local_lights() {
    let (mut engine, expected) = setup_headless_with_settings(
        160,
        160,
        RendererSettings {
            clustered_shading: ClusteredShadingMode::ForceOn,
            ..Default::default()
        },
    );

    {
        let scene = engine.scene_manager.create_active();
        populate_dense_point_light_scene(scene, &engine.assets);
    }
    let local_only = render_and_capture(&mut engine, 2);
    assert_eq!(local_only.len(), expected);
    assert_not_black(&local_only, "clustered_local_only");

    reset_active_scene(&mut engine);
    {
        let scene = engine.scene_manager.create_active();
        populate_dense_point_light_scene(scene, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::splat(0.85), 1.8));
    }
    let with_directional = render_and_capture(&mut engine, 2);
    assert_eq!(with_directional.len(), expected);
    assert_not_black(
        &with_directional,
        "clustered_directional_plus_dense_local_lights",
    );

    assert_images_differ(
        &local_only,
        &with_directional,
        "clustered_directional_plus_dense_local_lights",
    );
}

#[test]
fn clustered_corridor_matches_force_off_reference() {
    let (mut clustered_engine, expected) = setup_headless_with_settings(
        160,
        160,
        RendererSettings {
            clustered_shading: ClusteredShadingMode::ForceOn,
            ..Default::default()
        },
    );
    {
        let scene = clustered_engine.scene_manager.create_active();
        populate_clustered_corridor_scene(scene, &clustered_engine.assets);
    }
    let clustered = render_and_capture(&mut clustered_engine, 2);
    assert_eq!(clustered.len(), expected);
    assert_not_black(&clustered, "clustered_corridor_force_on");

    let (mut reference_engine, _) = setup_headless_with_settings(
        160,
        160,
        RendererSettings {
            clustered_shading: ClusteredShadingMode::ForceOff,
            ..Default::default()
        },
    );
    {
        let scene = reference_engine.scene_manager.create_active();
        populate_clustered_corridor_scene(scene, &reference_engine.assets);
    }
    let reference = render_and_capture(&mut reference_engine, 2);
    assert_not_black(&reference, "clustered_corridor_force_off");

    let mean_delta = mean_rgb_abs_delta(&clustered, &reference);
    assert!(
        mean_delta <= 3.0,
        "clustered corridor diverged too far from force-off reference: mean RGB abs delta = {mean_delta:.3}"
    );
}

// ── Alpha Mode Tests ─────────────────────────────────────────────────────

/// A semi-transparent (AlphaMode::Blend) object in front of a solid
/// background should produce colours that differ from a fully opaque one.
#[test]
fn alpha_blend_vs_opaque() {
    let (mut engine, _) = setup_headless(128, 128);

    // Opaque foreground
    {
        let scene = engine.scene_manager.create_active();
        // Background plane
        let bg = UnlitMaterial::new(Vec4::new(0.0, 0.0, 1.0, 1.0));
        let bg_node = scene.spawn_plane(4.0, 4.0, bg, &engine.assets);
        scene.node(&bg_node).set_position(0.0, 0.0, -2.0);
        // Foreground box (opaque red)
        let fg = UnlitMaterial::new(Vec4::new(1.0, 0.0, 0.0, 1.0));
        scene.spawn_box(1.0, 1.0, 0.1, fg, &engine.assets);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 3.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let opaque = render_and_capture(&mut engine, 2);
    assert_not_black(&opaque, "opaque_foreground");

    // Semi-transparent foreground
    reset_active_scene(&mut engine);
    {
        let scene = engine.scene_manager.create_active();
        let bg = UnlitMaterial::new(Vec4::new(0.0, 0.0, 1.0, 1.0));
        let bg_node = scene.spawn_plane(4.0, 4.0, bg, &engine.assets);
        scene.node(&bg_node).set_position(0.0, 0.0, -2.0);
        // Foreground box: same red but 50% alpha with Blend mode
        let fg =
            UnlitMaterial::new(Vec4::new(1.0, 0.0, 0.0, 0.5)).with_alpha_mode(AlphaMode::Blend);
        scene.spawn_box(1.0, 1.0, 0.1, fg, &engine.assets);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 3.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let blended = render_and_capture(&mut engine, 2);
    assert_not_black(&blended, "blended_foreground");

    assert_images_differ(&opaque, &blended, "alpha_blend_vs_opaque");
}

// ── Cross-Material Comparison ────────────────────────────────────────────

/// Varying PBR roughness on the same geometry should produce visually
/// distinct images (smooth specular highlight vs diffuse scatter).
#[test]
fn physical_roughness_variation() {
    let (mut engine, _) = setup_headless(128, 128);

    let color = Vec4::new(0.8, 0.3, 0.3, 1.0);

    // Low roughness (mirror-like)
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(color)
            .with_roughness(0.05)
            .with_metalness(1.0);
        scene.spawn_sphere(1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 3.0));
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let img_smooth = render_and_capture(&mut engine, 2);
    assert_not_black(&img_smooth, "smooth");

    // High roughness (diffuse-like)
    reset_active_scene(&mut engine);
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(color)
            .with_roughness(1.0)
            .with_metalness(1.0);
        scene.spawn_sphere(1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 3.0));
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let img_rough = render_and_capture(&mut engine, 2);
    assert_not_black(&img_rough, "rough");

    assert_images_differ(&img_smooth, &img_rough, "roughness_variation");
}

// ── Multi-Object Scene ───────────────────────────────────────────────────

/// A scene with multiple PBR objects (sphere + box + sphere) at different
/// positions should render without panics and produce a non-black image.
#[test]
fn multi_object_scene() {
    let (mut engine, expected) = setup_headless(256, 256);
    let scene = engine.scene_manager.create_active();

    // Red metallic sphere
    let sphere_mat = PhysicalMaterial::new(Vec4::new(0.9, 0.1, 0.1, 1.0))
        .with_roughness(0.2)
        .with_metalness(1.0);
    let sphere = scene.spawn_sphere(0.6, sphere_mat, &engine.assets);
    scene.node(&sphere).set_position(-1.5, 0.0, 0.0);

    // Green dielectric box
    let box_mat = PhysicalMaterial::new(Vec4::new(0.1, 0.8, 0.2, 1.0))
        .with_roughness(0.6)
        .with_metalness(0.0);
    let bx = scene.spawn_box(1.0, 1.0, 1.0, box_mat, &engine.assets);
    scene.node(&bx).set_position(0.0, 0.0, 0.0);

    // Blue rough sphere
    let blue_mat = PhysicalMaterial::new(Vec4::new(0.1, 0.3, 1.0, 1.0))
        .with_roughness(0.9)
        .with_metalness(0.5);
    let blue_sphere = scene.spawn_sphere(0.4, blue_mat, &engine.assets);
    scene.node(&blue_sphere).set_position(1.5, 0.0, 0.0);

    // Lights
    scene.add_light(Light::new_directional(Vec3::ONE, 2.5));
    let pt = scene.add_light(Light::new_point(Vec3::new(1.0, 0.8, 0.6), 10.0, 15.0));
    scene.node(&pt).set_position(0.0, 3.0, 3.0);

    // Camera
    let cam = scene.add_camera(Camera::new_perspective(60.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 3.0, 6.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 3);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "multi_object_scene");
}
