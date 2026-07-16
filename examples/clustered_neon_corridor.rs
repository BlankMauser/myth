//! [gallery]
//! name = "Clustered Neon Corridor"
//! category = "Lighting & GI"
//! description = "A visually dense corridor with layered coloured point lights, glossy materials, and overlapping pools of clustered lighting."
//! order = 402
//!

use std::f32::consts::{FRAC_PI_2, TAU};

use myth::prelude::*;
use myth::render::ClusteredShadingMode;
use myth_dev_utils::FpsCounter;

const SECTION_COUNT: usize = 14;
const LIGHTS_PER_SECTION: usize = 5;

struct CorridorLight {
    node: NodeHandle,
    base: Vec3,
    amplitude: Vec3,
    phase: f32,
    speed: f32,
}

struct ClusteredNeonCorridorDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    lights: Vec<CorridorLight>,
    time: f32,
}

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

impl AppHandler for ClusteredNeonCorridorDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.006));
        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.03, 0.02, 0.05, 1.0),
            Vec4::new(0.004, 0.005, 0.01, 1.0),
        ));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.08);
        scene.bloom.set_radius(0.006);
        scene.tone_mapping.set_exposure(1.18);

        let floor_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.05, 0.06, 0.075, 1.0))
                .with_roughness(0.12)
                .with_metalness(0.82),
        );
        let wall_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.11, 0.12, 0.16, 1.0))
                .with_roughness(0.64)
                .with_metalness(0.18),
        );
        let frame_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.16, 0.17, 0.21, 1.0))
                .with_roughness(0.28)
                .with_metalness(0.66),
        );
        let hero_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.92, 0.92, 0.96, 1.0))
                .with_roughness(0.08)
                .with_metalness(1.0),
        );
        let accent_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.08, 0.08, 0.12, 1.0))
                .with_emissive(Vec3::new(0.24, 0.78, 1.0), 2.8)
                .with_roughness(0.22),
        );

        let floor = scene.spawn_plane(16.0, 76.0, floor_material, &engine.assets);
        scene
            .edit_node(floor)
            .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
            .set_position(0.0, -0.18, 0.0)
            .set_receive_shadows(false);

        for &(x, y, z, sx, sy, sz) in &[
            (-6.4, 2.35, 0.0, 0.45, 5.2, 62.0),
            (6.4, 2.35, 0.0, 0.45, 5.2, 62.0),
            (0.0, 4.9, 0.0, 13.8, 0.24, 62.0),
        ] {
            let wall = scene.spawn_box(sx, sy, sz, wall_material, &engine.assets);
            scene
                .edit_node(wall)
                .set_position(x, y, z)
                .set_shadows(false, true);
        }

        for section in 0..SECTION_COUNT {
            let z = centered_lattice(section, SECTION_COUNT, 4.4);

            for x in [-4.35, 4.35] {
                let pillar = scene.spawn_box(0.52, 4.8, 0.52, frame_material, &engine.assets);
                scene
                    .edit_node(pillar)
                    .set_position(x, 2.1, z)
                    .set_shadows(true, true);
            }

            let beam = scene.spawn_box(9.8, 0.18, 0.42, accent_material, &engine.assets);
            scene
                .edit_node(beam)
                .set_position(0.0, 4.05, z)
                .set_cast_shadows(false)
                .set_receive_shadows(false);

            let hero = if section % 2 == 0 {
                scene.spawn_sphere(0.92, hero_material, &engine.assets)
            } else {
                scene.spawn_box(1.5, 1.5, 1.5, hero_material, &engine.assets)
            };
            scene
                .edit_node(hero)
                .set_position(0.0, 0.72, z)
                .set_shadows(true, true);
        }

        let mut lights = Vec::with_capacity(SECTION_COUNT * LIGHTS_PER_SECTION);
        for section in 0..SECTION_COUNT {
            let z = centered_lattice(section, SECTION_COUNT, 4.4);
            for slot in 0..LIGHTS_PER_SECTION {
                let light_index = section * LIGHTS_PER_SECTION + slot;
                let hue = light_index as f32 / (SECTION_COUNT * LIGHTS_PER_SECTION) as f32;
                let color = hsv_to_rgb(hue, 0.82, 1.0);
                let light = scene.add_light(Light::new_point(
                    color,
                    1.4 + if slot == 2 { 0.35 } else { 0.0 },
                    5.8 + slot as f32 * 0.18,
                ));
                let helper = scene.spawn_sphere(
                    0.075,
                    PhysicalMaterial::new((color * 0.22).extend(1.0))
                        .with_emissive(color, 3.6)
                        .with_roughness(0.22)
                        .with_metalness(0.0),
                    &engine.assets,
                );
                scene.attach(helper, light);
                scene
                    .edit_node(helper)
                    .set_position(0.0, 0.0, 0.0)
                    .set_shadows(false, false);

                let (x, y, z_offset, amplitude, speed) = match slot {
                    0 => (-3.9, 3.25, -0.55, Vec3::new(0.35, 0.18, 0.22), 0.58),
                    1 => (3.9, 3.25, 0.55, Vec3::new(0.35, 0.18, 0.22), 0.62),
                    2 => (0.0, 2.15, 0.0, Vec3::new(0.0, 0.42, 0.35), 0.75),
                    3 => (-1.85, 1.1, 0.75, Vec3::new(0.28, 0.22, 0.18), 0.94),
                    _ => (1.85, 1.1, -0.75, Vec3::new(0.28, 0.22, 0.18), 1.02),
                };
                let base = Vec3::new(x, y, z + z_offset);
                scene.edit_node(light).set_position(base.x, base.y, base.z);

                lights.push(CorridorLight {
                    node: light,
                    base,
                    amplitude,
                    phase: (light_index as f32 / 9.0) * TAU,
                    speed,
                });
            }
        }

        let camera_pos = Vec3::new(0.0, 3.15, 16.5);
        let target = Vec3::new(0.0, 1.4, 0.0);
        let cam = scene.add_camera(Camera::new_perspective(48.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(cam)
            .set_position(camera_pos.x, camera_pos.y, camera_pos.z)
            .look_at(target);
        scene.active_camera = Some(cam);

        let mut controls = OrbitControls::new(camera_pos, target);
        controls.min_distance = 8.0;
        controls.max_distance = 30.0;

        Self {
            controls,
            fps_counter: FpsCounter::new(),
            lights,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        self.time += frame.dt;

        for light in &self.lights {
            let offset_x = (self.time * light.speed + light.phase).sin() * light.amplitude.x;
            let offset_y =
                (self.time * (light.speed * 1.7) + light.phase * 0.7).sin() * light.amplitude.y;
            let offset_z =
                (self.time * (light.speed * 1.15) + light.phase * 1.3).cos() * light.amplitude.z;
            scene.edit_node(light.node).set_position(
                light.base.x + offset_x,
                light.base.y + offset_y,
                light.base.z + offset_z,
            );
        }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "Clustered Neon Corridor | {} point lights | FPS: {:.2}",
                self.lights.len(),
                fps
            ));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_settings(RendererSettings {
            clustered_shading: ClusteredShadingMode::ForceOn,
            vsync: false,
            ..Default::default()
        })
        .run::<ClusteredNeonCorridorDemo>()
}
