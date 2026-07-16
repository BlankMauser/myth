//! Headless capture tool for the documentation homepage hero animation.
//!
//! Faithfully replicates the `showcase` demo's **Cinematic** preset
//! (`demo_apps/showcase`) — same camera framing, IBL environment, three-point
//! lighting, tone mapping, bloom, SSAO and (crucially) subsurface skin
//! shading (SSSS) — but renders onto a **fully transparent background** and
//! writes an RGBA PNG sequence. `scripts/build-hero-video.ps1` then encodes
//! the frames into a transparent WebM (VP9 / `yuva420p`) so the hero blends
//! seamlessly into both the light and dark documentation themes.
//!
//! Transparency works because:
//!   * `BackgroundMode::Color` with alpha 0 clears to transparent and skips
//!     the skybox pass (the env map is still used for IBL, just not drawn).
//!   * Tone-mapping / FXAA preserve source alpha; MSAA resolve yields a
//!     correctly anti-aliased silhouette.
//!   * `readback_pixels()` returns straight (non-premultiplied) RGBA8.
//!
//! Timeline: frames advance at a fixed `dt = 1/FPS` and the capture length is
//! locked to exactly one loop of the first animation clip, so the facial
//! animation plays at its natural speed and loops seamlessly.
//!
//! ```bash
//! cargo run --release --example hero_capture
//! ```

use myth::ToneMappingMode;
use myth::prelude::*;
use myth::resources::tone_mapping::AgxLook;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(p) => p,
    None => "examples/assets/",
};

const DEFAULT_MODEL: &str = "cute_girl.glb";
const HDR_FILE: &str = "blouberg_sunrise_2_1k.hdr";
const LUT_FILE: &str = "Rec709 Fujifilm 3513DI D65.bin";

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    env_logger::init();

    // ── Configuration ────────────────────────────────────────────────
    let size = env_usize("HERO_SIZE", 1024) as u32;
    let fps = 30.0_f32;
    let dt = 1.0 / fps;
    // Optional cap on captured frames (0 = use full animation loop length).
    let frame_cap = env_usize("HERO_FRAMES", 0);
    let out_dir = std::path::Path::new("target/hero_frames");
    let _ = std::fs::remove_dir_all(out_dir);
    std::fs::create_dir_all(out_dir).expect("create output dir");

    // ── GPU (headless) ───────────────────────────────────────────────
    let mut engine = Engine::default();
    pollster::block_on(engine.init_headless(size, size, None)).expect("headless init failed");

    // ── Kick off async asset loads (mirrors showcase Cinematic) ──────
    let hdr_handle = engine
        .assets
        .load_hdr_texture(format!("{ASSET_PATH}envs/{HDR_FILE}"));
    let lut_handle = engine
        .assets
        .load_lut_texture(format!("{ASSET_PATH}luts/{LUT_FILE}"));
    let model_handle = engine
        .assets
        .load_gltf(format!("{ASSET_PATH}{DEFAULT_MODEL}"));

    // ── Scene + camera + lights (Cinematic preset, transparent bg) ───
    let cam_node_id;
    {
        let scene = engine.scene_manager.create_active();

        // Transparent clear — env map drives IBL, but no skybox is drawn so
        // pixels outside the silhouette keep alpha 0.
        scene
            .background
            .set_mode(BackgroundMode::color_with_alpha(0.0, 0.0, 0.0, 0.0));

        // ── Environment (Cinematic: dark IBL)
        scene.environment.set_env_map(Some(hdr_handle));
        scene.environment.set_intensity(0.8);
        scene.environment.set_ambient_light(Vec3::splat(0.02));

        // ── Three-point lighting (Cinematic: warm key / cool fill, rim off)
        let key = scene.add_light(Light::new_directional(Vec3::new(1.0, 0.85, 0.6), 5.0));
        scene
            .edit_node(key)
            .set_position(5.0, 5.0, 5.0)
            .look_at(Vec3::ZERO);
        let fill = scene.add_light(Light::new_directional(Vec3::new(0.3, 0.5, 1.0), 1.0));
        scene
            .edit_node(fill)
            .set_position(-5.0, 2.0, -2.0)
            .look_at(Vec3::ZERO);

        // ── Tone mapping (Cinematic look, but Studio-style "clean" post)
        // Vignette / chromatic aberration / film grain are intentionally
        // DISABLED: those passes write to every pixel (grain even adds colored
        // noise to fully-transparent areas), which would break the strict
        // "subject only" transparency we need for the homepage hero.
        scene
            .tone_mapping
            .set_mode(ToneMappingMode::AgX(AgxLook::None));
        scene.tone_mapping.set_exposure(1.0);
        scene.tone_mapping.set_contrast(1.1);
        scene.tone_mapping.set_saturation(1.05);
        scene.tone_mapping.set_vignette_intensity(0.0);
        scene.tone_mapping.set_chromatic_aberration(0.0);
        scene.tone_mapping.set_film_grain(0.0);
        scene.tone_mapping.set_lut_texture(Some(lut_handle));
        scene.tone_mapping.set_lut_contribution(1.0);

        // ── Bloom / SSAO (Cinematic)
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.06);
        scene.bloom.set_radius(0.005);
        scene.ssao.set_enabled(true);
        scene.ssao.set_radius(0.5);
        scene.ssao.set_intensity(1.5);

        // ── Subsurface skin shading
        scene.ssss.set_enabled(true);

        // ── Camera (same framing as showcase's initial OrbitControls pose)
        let mut camera = Camera::new_perspective(45.0, 1.0, 0.01);
        camera.set_aa_mode(AntiAliasingMode::MSAA_FXAA(4, FxaaSettings::default()));
        cam_node_id = scene.add_camera(camera);
        scene
            .edit_node(cam_node_id)
            .set_position(-0.1, 1.45, -0.45)
            .look_at(Vec3::new(0.0, 1.5, 0.0));
        scene.active_camera = Some(cam_node_id);
    }

    // ── Wait for model + HDR + LUT (dt=0 so nothing animates yet) ────
    println!("Loading assets...");
    let prefab = loop {
        engine.update(0.0);
        let ready_model = engine.assets.prefabs.get(model_handle);
        let ready_hdr = engine.assets.textures.is_loaded(hdr_handle);
        let ready_lut = engine.assets.textures.is_loaded(lut_handle);
        if let (Some(prefab), true, true) = (ready_model, ready_hdr, ready_lut) {
            break prefab;
        }
        std::thread::sleep(std::time::Duration::from_millis(4));
    };

    // ── Instantiate, set up skin SSS, play animation ─────────────────
    let anim_duration;
    {
        let scene = engine.scene_manager.active_scene_mut().unwrap();
        let root = scene.instantiate(&prefab);
        scene.update_subtree(root);

        // Subsurface skin profile on the body mesh (mirrors showcase).
        let skin_profile =
            myth::resources::ssss::SssProfile::new(Vec3::new(0.85, 0.25, 0.15), 0.15);
        if let Some(model_node) = scene.find_node_by_name("Object_9") {
            if let Some(mesh) = scene.meshes.get(model_node) {
                if let Some(material) = engine.assets.materials.get(mesh.material) {
                    if let Some(pbr) = material.as_physical() {
                        if let Some(new_id) = engine.assets.sss_registry.write().add(&skin_profile)
                        {
                            pbr.set_sss_id(Some(new_id));
                        }
                    }
                }
            }
        }

        // Play (and measure) the first animation clip.
        anim_duration = if let Some(mixer) = scene.animation_mixers.get_mut(root) {
            if let Some(name) = mixer.list_animations().first().cloned() {
                println!("Playing animation: {name}");
                mixer.play(&name);
                mixer
                    .get_action(&name)
                    .map(|a| a.clip().duration)
                    .unwrap_or(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };
    }

    // Frame count = exactly one animation loop (falls back to 5s if static).
    let loop_seconds = if anim_duration > 0.05 {
        anim_duration
    } else {
        5.0
    };
    let mut total_frames = (loop_seconds * fps).round() as usize;
    if frame_cap > 0 {
        total_frames = total_frames.min(frame_cap);
    }
    println!(
        "Animation loop: {:.2}s -> {} frames @ {} fps ({}x{})",
        loop_seconds, total_frames, fps as u32, size, size
    );

    // ── Warm-up (IBL bake / texture settle) without advancing time ───
    for _ in 0..12 {
        engine.update(0.0);
        engine.render_active_scene();
    }

    // ── Capture loop ─────────────────────────────────────────────────
    // Render first (t = i*dt), then advance — so frame 0 is the animation
    // start and the last frame sits just before the loop point.
    println!("Capturing {total_frames} frame(s)...");
    for i in 0..total_frames {
        engine.render_active_scene();
        let pixels = engine.readback_pixels().expect("readback failed");
        let path = out_dir.join(format!("frame_{i:04}.png"));
        image::save_buffer(&path, &pixels, size, size, image::ColorType::Rgba8)
            .expect("failed to save frame");

        if (i + 1) % 20 == 0 {
            println!("  {}/{}", i + 1, total_frames);
        }

        engine.update(dt);
    }

    println!(
        "Done. {total_frames} frames written to {}",
        out_dir.display()
    );
}
