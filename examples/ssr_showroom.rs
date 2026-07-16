//! [gallery]
//! name = "SSR Showroom"
//! category = "Lighting & GI"
//! description = "Mirror-finished floor, animated emissive accents, and polished hero props for validating screen-space reflections, quality presets, and debug overlays."
//! instructions = "Press T to toggle SSR\nPress Y to toggle TAA\nPress Q to cycle SSR quality\nPress G to cycle SSR debug views"
//! order = 412
//! features = ["debug_view"]
//!

use std::f32::consts::{PI, TAU};

use myth::prelude::*;
use myth::resources::SsrQuality;
use myth::resources::input::Key;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

fn next_quality(current: SsrQuality) -> SsrQuality {
    let presets = SsrQuality::all();
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
        DebugViewMode::None => DebugViewMode::SsrRaw,
        // DebugViewMode::SsrRaw => DebugViewMode::SsrTraceDiagnostic,
        // DebugViewMode::SsrTraceDiagnostic => DebugViewMode::SsrTraceState,
        DebugViewMode::SsrRaw => DebugViewMode::SsrTraceState,
        DebugViewMode::SsrTraceState => DebugViewMode::SsrResolved,
        _ => DebugViewMode::None,
    }
}

struct AccentLight {
    node: NodeHandle,
    base: Vec3,
    phase: f32,
    radius: f32,
    height_amp: f32,
}

struct Showpiece {
    node: NodeHandle,
    base: Vec3,
    spin_axis: Vec3,
    spin_speed: f32,
    bob_amp: f32,
    bob_phase: f32,
}

struct SsrShowroomDemo {
    // controls: OrbitControls,
    fps_counter: FpsCounter,
    accent_lights: Vec<AccentLight>,
    showpieces: Vec<Showpiece>,
    time: f32,
}

impl SsrShowroomDemo {
    fn spawn_bar(
        scene: &mut Scene,
        box_geo: GeometryHandle,
        material: MaterialHandle,
        position: Vec3,
        scale: Vec3,
        casts_shadows: bool,
    ) {
        let node = scene.add_mesh(Mesh::new(box_geo, material));
        scene
            .edit_node(node)
            .set_position(position.x, position.y, position.z)
            .set_scale_xyz(scale.x, scale.y, scale.z)
            .set_shadows(casts_shadows, casts_shadows);
    }

    fn spawn_pedestal(
        scene: &mut Scene,
        box_geo: GeometryHandle,
        material: MaterialHandle,
        position: Vec3,
    ) {
        let node = scene.add_mesh(Mesh::new(box_geo, material));
        scene
            .edit_node(node)
            .set_position(position.x, 0.42, position.z)
            .set_scale_xyz(1.75, 0.84, 1.75)
            .set_shadows(true, true);
    }

    fn print_help() {
        println!("╔═══════════════════════════════════════╗");
        println!("║            SSR Demo Controls          ║");
        println!("╠═══════════════════════════════════════╣");
        println!("║ T - Toggle SSR                        ║");
        println!("║ Y - Toggle TAA                        ║");
        println!("║ Q - Cycle SSR Quality                 ║");
        #[cfg(feature = "debug_view")]
        println!("║ G - Cycle SSR Debug Views             ║");
        println!("╚═══════════════════════════════════════╝");
    }
}

impl AppHandler for SsrShowroomDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let sphere_geo = engine.assets.geometries.add(Geometry::new_sphere(1.0));

        let floor_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.03, 0.035, 0.045, 1.0))
                .with_roughness(0.045)
                .with_metalness(0.04),
        );
        let wall_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.09, 0.10, 0.12, 1.0))
                .with_roughness(0.78)
                .with_metalness(0.08),
        );
        let fin_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.13, 0.14, 0.17, 1.0))
                .with_roughness(0.56)
                .with_metalness(0.18),
        );
        let pedestal_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.16, 0.17, 0.20, 1.0))
                .with_roughness(0.42)
                .with_metalness(0.42),
        );
        let chrome_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.96, 0.97, 0.99, 1.0))
                .with_roughness(0.055)
                .with_metalness(1.0),
        );
        let copper_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.98, 0.58, 0.30, 1.0))
                .with_roughness(0.14)
                .with_metalness(1.0),
        );
        let ink_mirror_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.04, 0.05, 0.06, 1.0))
                .with_roughness(0.07)
                .with_metalness(0.96),
        );
        let champagne_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.93, 0.84, 0.64, 1.0))
                .with_roughness(0.22)
                .with_metalness(0.92),
        );

        let emissive_palette = [
            (
                Vec3::new(0.18, 0.92, 1.0),
                engine.assets.materials.add(
                    PhysicalMaterial::new(Vec4::new(0.08, 0.18, 0.24, 1.0))
                        .with_emissive(Vec3::new(0.18, 0.92, 1.0), 5.5)
                        .with_roughness(0.08),
                ),
            ),
            (
                Vec3::new(1.0, 0.34, 0.78),
                engine.assets.materials.add(
                    PhysicalMaterial::new(Vec4::new(0.24, 0.08, 0.18, 1.0))
                        .with_emissive(Vec3::new(1.0, 0.34, 0.78), 5.2)
                        .with_roughness(0.08),
                ),
            ),
            (
                Vec3::new(1.0, 0.74, 0.22),
                engine.assets.materials.add(
                    PhysicalMaterial::new(Vec4::new(0.26, 0.16, 0.05, 1.0))
                        .with_emissive(Vec3::new(1.0, 0.74, 0.22), 5.0)
                        .with_roughness(0.08),
                ),
            ),
        ];

        let scene = engine.scene_manager.create_active();
        scene.ssr.set_enabled(true);
        scene.ssr.set_quality(SsrQuality::Ultra);
        scene.ssr.set_thickness(0.01);
        scene.environment.set_ambient_light(Vec3::splat(0.008));
        scene.environment.set_intensity(0.55);
        scene
            .tone_mapping
            .set_mode(myth::ToneMappingMode::AgX(myth::AgxLook::Punchy));

        let env_texture = engine
            .assets
            .load_hdr_texture(format!("{}envs/royal_esplanade_2k.hdr.jpg", ASSET_PATH));
        scene.environment.set_env_map(Some(env_texture));
        scene
            .background
            .set_mode(BackgroundMode::equirectangular(env_texture, 1.0));

        Self::spawn_bar(
            scene,
            box_geo,
            floor_material,
            Vec3::new(0.0, -0.16, 0.0),
            Vec3::new(18.0, 0.28, 18.0),
            false,
        );
        Self::spawn_bar(
            scene,
            box_geo,
            wall_material,
            Vec3::new(0.0, 3.2, -8.2),
            Vec3::new(15.5, 6.4, 0.28),
            false,
        );
        Self::spawn_bar(
            scene,
            box_geo,
            wall_material,
            Vec3::new(0.0, 6.0, -1.2),
            Vec3::new(11.0, 0.20, 8.2),
            false,
        );
        Self::spawn_bar(
            scene,
            box_geo,
            fin_material,
            Vec3::new(-6.7, 2.6, -0.8),
            Vec3::new(0.24, 5.2, 10.0),
            false,
        );
        Self::spawn_bar(
            scene,
            box_geo,
            fin_material,
            Vec3::new(6.7, 2.6, -0.8),
            Vec3::new(0.24, 5.2, 10.0),
            false,
        );

        for (index, (_, material)) in emissive_palette.iter().enumerate() {
            let x = (index as f32 - 1.0) * 3.25;
            Self::spawn_bar(
                scene,
                box_geo,
                *material,
                Vec3::new(x, 3.15, -7.95),
                Vec3::new(0.22, 4.4, 0.14),
                false,
            );
        }

        for &position in &[
            Vec3::new(-3.4, 0.0, 1.8),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.4, 0.0, -1.6),
            Vec3::new(0.0, 0.0, -3.5),
        ] {
            Self::spawn_pedestal(scene, box_geo, pedestal_material, position);
        }

        let mut key_light = Light::new_directional(Vec3::new(1.0, 0.96, 0.92), 2.8);
        key_light.cast_shadows = true;
        let key_light = scene.add_light(key_light);
        scene
            .edit_node(key_light)
            .set_rotation(Quat::from_euler(EulerRot::YXZ, -0.55, -0.72, 0.0));

        let fill_light = scene.add_light(Light::new_directional(Vec3::new(0.26, 0.38, 0.62), 0.35));
        scene
            .edit_node(fill_light)
            .set_rotation(Quat::from_euler(EulerRot::YXZ, 1.45, -0.28, 0.0));

        let mut accent_lights = Vec::new();
        let accent_specs = [
            (Vec3::new(-2.8, 2.8, 1.6), 1.65, 0.22),
            (Vec3::new(0.0, 3.0, -0.2), 1.25, 0.18),
            (Vec3::new(2.9, 2.7, -1.7), 1.55, 0.20),
        ];
        for ((color, helper_material), (base, radius, height_amp)) in
            emissive_palette.iter().zip(accent_specs)
        {
            let mut point = Light::new_point(*color, 5.0, 7.8);
            point.cast_shadows = true;
            let light = scene.add_light(point);
            scene.edit_node(light).set_position(base.x, base.y, base.z);

            let helper = scene.add_mesh(Mesh::new(sphere_geo, *helper_material));
            scene.attach(helper, light);
            scene
                .edit_node(helper)
                .set_position(0.0, 0.0, 0.0)
                .set_scale(0.20)
                .set_shadows(false, false);

            accent_lights.push(AccentLight {
                node: light,
                base,
                phase: accent_lights.len() as f32 * TAU / 3.0,
                radius,
                height_amp,
            });
        }

        let showpiece_specs = [
            (
                Mesh::new(box_geo, copper_material),
                Vec3::new(-3.4, 1.86, 1.8),
                Vec3::new(0.35, 1.0, 0.18).normalize(),
                Vec3::new(1.15, 1.15, 1.15),
                0.42,
                0.10,
                0.0,
            ),
            (
                Mesh::new(sphere_geo, chrome_material),
                Vec3::new(0.0, 2.18, 0.0),
                Vec3::Y,
                Vec3::splat(1.22),
                0.30,
                0.12,
                PI * 0.35,
            ),
            (
                Mesh::new(sphere_geo, champagne_material),
                Vec3::new(3.4, 1.82, -1.6),
                Vec3::new(0.12, 1.0, 0.32).normalize(),
                Vec3::splat(1.05),
                0.34,
                0.08,
                PI * 0.72,
            ),
            (
                Mesh::new(box_geo, ink_mirror_material),
                Vec3::new(0.0, 2.0, -3.5),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.95, 2.8, 0.95),
                0.18,
                0.06,
                PI * 0.18,
            ),
        ];

        let mut showpieces = Vec::new();
        for (mesh, base, spin_axis, scale, spin_speed, bob_amp, bob_phase) in showpiece_specs {
            let node = scene.add_mesh(mesh);
            scene
                .edit_node(node)
                .set_position(base.x, base.y, base.z)
                .set_scale_xyz(scale.x, scale.y, scale.z)
                .set_shadows(true, true);
            showpieces.push(Showpiece {
                node,
                base,
                spin_axis,
                spin_speed,
                bob_amp,
                bob_phase,
            });
        }

        let floor_ring = scene.add_mesh(Mesh::new(sphere_geo, chrome_material));
        scene
            .edit_node(floor_ring)
            .set_position(0.0, 0.34, 4.6)
            .set_scale_xyz(0.38, 0.38, 0.38)
            .set_shadows(true, true);
        showpieces.push(Showpiece {
            node: floor_ring,
            base: Vec3::new(0.0, 0.34, 4.6),
            spin_axis: Vec3::new(0.22, 1.0, 0.0).normalize(),
            spin_speed: 0.55,
            bob_amp: 0.04,
            bob_phase: PI * 1.2,
        });

        let mut camera = Camera::new_perspective(38.0, 16.0 / 9.0, 0.1);
        camera.set_aa_mode(AntiAliasingMode::TAA_FXAA(
            TaaSettings::default(),
            FxaaSettings::default(),
        ));
        #[cfg(feature = "debug_view")]
        {
            camera.debug_view.custom_scale = 1.3;
        }
        let camera_pos = Vec3::new(0.0, 3.8, 15.2);
        let camera_target = Vec3::new(0.0, 1.7, -0.8);
        let cam = scene.add_camera(camera);
        scene
            .edit_node(cam)
            .set_position(camera_pos.x, camera_pos.y, camera_pos.z)
            .look_at(camera_target);
        scene.active_camera = Some(cam);

        // let mut controls = OrbitControls::new(camera_pos, camera_target);
        // controls.min_distance = 6.0;
        // controls.max_distance = 18.0;
        // controls.pan_speed = 0.85;
        // controls.rotate_speed = 0.001;

        Self::print_help();

        Self {
            // controls,
            fps_counter: FpsCounter::new(),
            accent_lights,
            showpieces,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        self.time += frame.dt;

        if engine.input.get_key_down(Key::T) {
            scene.ssr.set_enabled(!scene.ssr.enabled);
        }

        if engine.input.get_key_down(Key::Q) {
            let next = next_quality(scene.ssr.quality());
            scene.ssr.set_quality(next);
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

        for light in &self.accent_lights {
            let angle = self.time * 0.9 + light.phase;
            let position = Vec3::new(
                light.base.x + angle.cos() * light.radius,
                light.base.y + (self.time * 1.4 + light.phase).sin() * light.height_amp,
                light.base.z + angle.sin() * light.radius,
            );
            scene
                .edit_node(light.node)
                .set_position(position.x, position.y, position.z);
        }

        for showpiece in &self.showpieces {
            let height = showpiece.base.y
                + (self.time * 1.1 + showpiece.bob_phase).sin() * showpiece.bob_amp;
            let rotation = Quat::from_axis_angle(
                showpiece.spin_axis,
                self.time * showpiece.spin_speed + showpiece.bob_phase,
            );
            scene
                .edit_node(showpiece.node)
                .set_position(showpiece.base.x, height, showpiece.base.z)
                .set_rotation(rotation);
        }

        // let _ = scene.edit_main_camera_bundle(|transform, camera| {
        //     self.controls
        //         .update(transform, &engine.input, camera.fov(), frame.dt);
        // });

        if let Some(fps) = self.fps_counter.update() {
            #[cfg(feature = "debug_view")]
            let debug_label = scene
                .main_camera_bundle()
                .map(|(_, camera)| camera.debug_view.mode.label())
                .unwrap_or("Final Image");

            #[cfg(not(feature = "debug_view"))]
            let debug_label = "Final Image";

            let ssr_label = if scene.ssr.enabled { "On" } else { "Off" };
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
                "SSR Showroom | SSR: {} | AA: {} | Quality: {} | View: {} | FPS: {:.2}",
                ssr_label,
                aa_label,
                scene.ssr.quality().name(),
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
        .run::<SsrShowroomDemo>()
}
