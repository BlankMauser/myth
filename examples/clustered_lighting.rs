//! [gallery]
//! name = "Clustered Light Rings"
//! category = "Lighting & GI"
//! description = "A correctness-focused clustered-lighting scene with animated rings of coloured point lights orbiting a PBR material study."
//! order = 400
//!

use std::f32::consts::TAU;

use myth::prelude::*;
use myth::render::ClusteredShadingMode;
use myth_dev_utils::FpsCounter;

const LIGHTS_PER_RING: usize = 16;
const RING_COUNT: usize = 3;

struct AnimatedLight {
    node: NodeHandle,
    radius: f32,
    speed: f32,
    phase: f32,
    base_height: f32,
}

struct ClusteredLightingDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    lights: Vec<AnimatedLight>,
    time: f32,
}

impl ClusteredLightingDemo {
    fn light_color(index: usize, total: usize) -> Vec3 {
        let hue = index as f32 / total as f32;
        hsv_to_rgb(hue, 0.82, 1.0)
    }
}

impl AppHandler for ClusteredLightingDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.01));
        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.03, 0.04, 0.07, 1.0),
            Vec4::new(0.005, 0.006, 0.012, 1.0),
        ));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.045);
        scene.bloom.set_radius(0.005);
        scene.tone_mapping.set_exposure(1.1);

        let floor = scene.spawn_plane(
            30.0,
            30.0,
            PhysicalMaterial::new(Vec4::new(0.07, 0.08, 0.09, 1.0)).with_roughness(0.95),
            &engine.assets,
        );
        scene
            .edit_node(floor)
            .set_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .set_position(0.0, -0.55, 0.0)
            .set_receive_shadows(true);

        let wall = scene.spawn_box(
            12.0,
            5.0,
            0.25,
            PhysicalMaterial::new(Vec4::new(0.08, 0.09, 0.12, 1.0)).with_roughness(0.88),
            &engine.assets,
        );
        scene
            .edit_node(wall)
            .set_position(0.0, 2.0, -5.4)
            .set_cast_shadows(false)
            .set_receive_shadows(true);

        let palette = [
            (
                Vec3::new(-2.8, 0.0, -1.4),
                Vec4::new(0.88, 0.14, 0.12, 1.0),
                0.15,
                0.0,
            ),
            (
                Vec3::new(0.0, 0.0, -1.4),
                Vec4::new(0.18, 0.62, 0.94, 1.0),
                0.34,
                0.0,
            ),
            (
                Vec3::new(2.8, 0.0, -1.4),
                Vec4::new(0.94, 0.84, 0.28, 1.0),
                0.48,
                1.0,
            ),
            (
                Vec3::new(-1.4, 0.0, 1.8),
                Vec4::new(0.90, 0.92, 0.97, 1.0),
                0.08,
                1.0,
            ),
            (
                Vec3::new(1.4, 0.0, 1.8),
                Vec4::new(0.16, 0.82, 0.45, 1.0),
                0.72,
                0.0,
            ),
        ];

        for (position, color, roughness, metalness) in palette {
            let node = scene.spawn_sphere(
                0.78,
                PhysicalMaterial::new(color)
                    .with_roughness(roughness)
                    .with_metalness(metalness),
                &engine.assets,
            );
            scene
                .edit_node(node)
                .set_position(position.x, position.y + 0.35, position.z)
                .set_shadows(true, true);
        }

        scene.add_light(Light::new_directional(Vec3::splat(0.18), 1.2));

        let mut lights = Vec::with_capacity(LIGHTS_PER_RING * RING_COUNT);
        for ring in 0..RING_COUNT {
            let radius = 2.6 + ring as f32 * 1.35;
            let height = 1.2 + ring as f32 * 0.8;
            let speed = 0.35 + ring as f32 * 0.16;

            for i in 0..LIGHTS_PER_RING {
                let idx = ring * LIGHTS_PER_RING + i;
                let phase = (i as f32 / LIGHTS_PER_RING as f32) * TAU + ring as f32 * 0.45;
                let color = Self::light_color(idx, LIGHTS_PER_RING * RING_COUNT);
                let light = scene.add_light(Light::new_point(color, 1.0, 5.2));
                let helper = scene.spawn_sphere(
                    0.08,
                    PhysicalMaterial::new((color * 0.22).extend(1.0))
                        .with_emissive(color, 3.4)
                        .with_roughness(0.24)
                        .with_metalness(0.0),
                    &engine.assets,
                );
                scene.attach(helper, light);
                scene
                    .edit_node(helper)
                    .set_position(0.0, 0.0, 0.0)
                    .set_shadows(false, false);

                let x = phase.cos() * radius;
                let z = phase.sin() * radius;
                scene.edit_node(light).set_position(x, height, z);

                lights.push(AnimatedLight {
                    node: light,
                    radius,
                    speed,
                    phase,
                    base_height: height,
                });
            }
        }

        let camera_pos = Vec3::new(0.0, 4.4, 11.5);
        let target = Vec3::new(0.0, 1.0, 0.0);
        let camera_node = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera_node)
            .set_position(camera_pos.x, camera_pos.y, camera_pos.z)
            .look_at(target);
        scene.active_camera = Some(camera_node);

        let mut controls = OrbitControls::new(camera_pos, target);
        controls.min_distance = 4.0;
        controls.max_distance = 18.0;

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
            let angle = self.time * light.speed + light.phase;
            let wobble = (self.time * (light.speed * 2.1) + light.phase * 1.7).sin() * 0.55;
            let x = angle.cos() * light.radius;
            let z = angle.sin() * light.radius;
            let y = light.base_height + wobble;
            scene.edit_node(light.node).set_position(x, y, z);
        }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "Clustered Light Rings | {} point lights | FPS: {:.2}",
                self.lights.len(),
                fps
            ));
        }
    }
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

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_settings(RendererSettings {
            clustered_shading: ClusteredShadingMode::ForceOn,
            vsync: false,
            ..Default::default()
        })
        .run::<ClusteredLightingDemo>()
}
