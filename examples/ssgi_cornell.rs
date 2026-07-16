//! [gallery]
//! name = "SSGI Cornell Box"
//! category = "Lighting & GI"
//! description = "High-contrast Cornell-box style scene for validating screen-space GI, quality presets, and optional debug overlays."
//! instructions = "Press T to toggle SSGI\nPress Y to toggle TAA\nPress Q to cycle SSGI quality\nPress G to cycle SSGI debug views."
//! order = 410
//! features = ["debug_view"]
//!

use myth::prelude::*;
use myth::resources::input::Key;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

fn next_quality(current: SsgiQuality) -> SsgiQuality {
    let presets = SsgiQuality::all();
    let index = presets
        .iter()
        .position(|preset| *preset == current)
        .unwrap_or(0);
    presets[(index + 1) % presets.len()]
}

fn toggle_taa(mode: AntiAliasingMode) -> AntiAliasingMode {
    if mode.is_taa() {
        AntiAliasingMode::fxaa()
    } else {
        AntiAliasingMode::taa_fxaa()
    }
}

#[cfg(feature = "debug_view")]
fn next_debug_mode(current: DebugViewMode) -> DebugViewMode {
    match current {
        DebugViewMode::None => DebugViewMode::SsgiRaw,
        DebugViewMode::SsgiRaw => DebugViewMode::SsgiDenoised,
        _ => DebugViewMode::None,
    }
}

fn print_help() {
    println!("╔══════════════════════════════════════════╗");
    println!("║          SSGI Cornell Box Demo           ║");
    println!("╟──────────────────────────────────────────╢");
    println!("║ T: Toggle SSGI                           ║");
    println!("║ Y: Toggle TAA                            ║");
    println!("║ Q: Cycle SSGI Quality                    ║");
    #[cfg(feature = "debug_view")]
    println!("║ G: Cycle SSGI Debug Views                ║");
    println!("╚══════════════════════════════════════════╝");
}

struct SsgiCornellDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    light_node: NodeHandle,
    time: f32,
}

impl AppHandler for SsgiCornellDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        let env_texture = engine.assets.load_texture(
            format!("{}envs/royal_esplanade_2k.hdr.jpg", ASSET_PATH),
            ColorSpace::Srgb,
            false,
        );
        scene.environment.set_env_map(Some(env_texture));
        scene.environment.set_intensity(0.08);
        scene.environment.set_ambient_light(Vec3::splat(0.001));
        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.01, 0.01, 0.015, 1.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        ));

        scene.ssgi.set_enabled(true);
        scene.ssgi.set_quality(SsgiQuality::Ultra);

        let white = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.74, 0.73, 0.71, 1.0))
                .with_roughness(0.72)
                .with_metalness(0.02),
        );
        let red = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.78, 0.18, 0.14, 1.0))
                .with_roughness(0.78)
                .with_metalness(0.0),
        );
        let green = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.16, 0.58, 0.22, 1.0))
                .with_roughness(0.76)
                .with_metalness(0.0),
        );
        let hero = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.92, 0.92, 0.94, 1.0))
                .with_roughness(0.16)
                .with_metalness(0.56),
        );
        let bulb = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(1.0, 0.96, 0.88, 1.0))
                .with_emissive(Vec3::new(1.0, 0.92, 0.76), 5.5)
                .with_roughness(0.08)
                .with_metalness(0.0),
        );

        for &(x, y, z, sx, sy, sz, material) in &[
            (0.0, 0.0, 0.0, 6.2, 0.12, 6.2, white),
            (0.0, 6.0, 0.0, 6.2, 0.12, 6.2, white),
            (0.0, 3.0, -3.0, 6.2, 6.2, 0.12, white),
            (-3.0, 3.0, 0.0, 0.12, 6.2, 6.2, red),
            (3.0, 3.0, 0.0, 0.12, 6.2, 6.2, green),
        ] {
            let wall = scene.spawn_box(sx, sy, sz, material, &engine.assets);
            scene
                .edit_node(wall)
                .set_position(x, y, z)
                .set_shadows(true, true);
        }

        let short_box = scene.spawn_box(1.65, 2.1, 1.65, hero, &engine.assets);
        scene
            .edit_node(short_box)
            .set_position(-1.1, 1.05, -0.95)
            .set_rotation(Quat::from_rotation_y(0.32))
            .set_shadows(true, true);

        let tall_box = scene.spawn_box(1.35, 3.75, 1.35, hero, &engine.assets);
        scene
            .edit_node(tall_box)
            .set_position(1.15, 1.875, 0.65)
            .set_rotation(Quat::from_rotation_y(-0.24))
            .set_shadows(true, true);

        let mut point = Light::new_point(Vec3::new(1.0, 0.97, 0.92), 6.0, 7.2);
        point.cast_shadows = true;
        let light_node = scene.add_light(point);
        scene.edit_node(light_node).set_position(0.0, 5.2, 0.0);

        let bulb_mesh = scene.spawn_sphere(0.18, bulb, &engine.assets);
        scene.attach(bulb_mesh, light_node);
        scene
            .edit_node(bulb_mesh)
            .set_position(0.0, 0.0, 0.0)
            .set_shadows(false, false);

        let mut camera = Camera::new_perspective(40.0, 16.0 / 9.0, 0.1);
        camera.set_aa_mode(AntiAliasingMode::TAA_FXAA(
            TaaSettings::default(),
            FxaaSettings::default(),
        ));
        #[cfg(feature = "debug_view")]
        {
            camera.debug_view.custom_scale = 1.35;
        }
        let cam = scene.add_camera(camera);
        scene
            .edit_node(cam)
            .set_position(0.0, 2.6, 9.4)
            .look_at(Vec3::new(0.0, 2.3, 0.0));
        scene.active_camera = Some(cam);

        let mut controls = OrbitControls::new(Vec3::new(0.0, 2.6, 9.4), Vec3::new(0.0, 2.3, 0.0));
        controls.min_distance = 5.0;
        controls.max_distance = 14.0;

        print_help();

        Self {
            controls,
            fps_counter: FpsCounter::new(),
            light_node,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        self.time += frame.dt;

        if engine.input.get_key_down(Key::T) {
            scene.ssgi.set_enabled(!scene.ssgi.enabled);
        }

        if engine.input.get_key_down(Key::Q) {
            let next = next_quality(scene.ssgi.quality());
            scene.ssgi.set_quality(next);
        }

        if engine.input.get_key_down(Key::Y)
            && let Some(camera_node) = scene.active_camera
            && let Some(camera) = scene.camera_mut(camera_node)
        {
            camera.set_aa_mode(toggle_taa(camera.aa_mode));
        }

        #[cfg(feature = "debug_view")]
        if engine.input.get_key_down(Key::G)
            && let Some(camera_node) = scene.active_camera
            && let Some(camera) = scene.camera_mut(camera_node)
        {
            camera.debug_view.mode = next_debug_mode(camera.debug_view.mode);
        }

        let light_pos = Vec3::new(
            self.time.cos() * 0.55,
            5.1 + (self.time * 1.6).sin() * 0.16,
            self.time.sin() * 0.75,
        );
        scene
            .edit_node(self.light_node)
            .set_position(light_pos.x, light_pos.y, light_pos.z);

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            #[cfg(feature = "debug_view")]
            let debug_label = scene
                .main_camera_bundle()
                .map(|(_, camera)| camera.debug_view.mode.label())
                .unwrap_or("Final Image");

            #[cfg(not(feature = "debug_view"))]
            let debug_label = "Final Image";

            let enabled_label = if scene.ssgi.enabled { "On" } else { "Off" };
            let aa_label = scene
                .main_camera_bundle()
                .map(|(_, camera)| {
                    if camera.aa_mode.is_taa() {
                        "TAA+FXAA"
                    } else {
                        "FXAA"
                    }
                })
                .unwrap_or("Unknown");
            window.set_title(&format!(
                "SSGI Cornell Box | SSGI: {} | AA: {} | Quality: {} | View: {} | FPS: {:.2}",
                enabled_label,
                aa_label,
                scene.ssgi.quality().name(),
                debug_label,
                fps
            ));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<SsgiCornellDemo>()
}
