//! [gallery]
//! name = "Vector Balls"
//! category = "Post Effects"
//! description = "Classic demoscene-style vector balls following layered Lissajous paths with bloom-heavy neon staging."
//! order = 456
//!

use std::f32::consts::TAU;

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

const BALL_COUNT: usize = 180;

fn next_rand(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

fn rand_f32(seed: &mut u32) -> f32 {
    next_rand(seed) as f32 / u32::MAX as f32
}

fn rand_range(seed: &mut u32, min: f32, max: f32) -> f32 {
    min + (max - min) * rand_f32(seed)
}

struct VectorBall {
    handle: NodeHandle,
    phase: Vec3,
    frequency: Vec3,
    radius: Vec3,
    size: f32,
}

struct VectorBallsDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    balls: Vec<VectorBall>,
    orbit_light: NodeHandle,
    time: f32,
}

impl AppHandler for VectorBallsDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let sphere_geo = engine.assets.geometries.add(Geometry::new_sphere(1.0));

        let floor_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.07, 0.08, 0.10, 1.0)).with_roughness(0.94));
        let palette = [
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.08, 0.10, 0.20, 1.0))
                    .with_emissive(Vec3::new(0.28, 0.82, 1.0), 4.8)
                    .with_roughness(0.12),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.16, 0.06, 0.20, 1.0))
                    .with_emissive(Vec3::new(1.0, 0.32, 0.92), 4.6)
                    .with_roughness(0.16),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.18, 0.12, 0.06, 1.0))
                    .with_emissive(Vec3::new(1.0, 0.78, 0.28), 4.4)
                    .with_roughness(0.18),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.06, 0.18, 0.12, 1.0))
                    .with_emissive(Vec3::new(0.30, 1.0, 0.62), 4.5)
                    .with_roughness(0.15),
            ),
        ];

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.012));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.11);
        scene.bloom.set_radius(0.006);
        scene.bloom.set_karis_average(true);

        let floor = scene.spawn_plane(24.0, 24.0, floor_material, &engine.assets);
        scene
            .edit_node(floor)
            .set_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .set_position(0.0, -3.0, 0.0)
            .set_cast_shadows(false)
            .set_receive_shadows(true);

        let mut key = Light::new_directional(Vec3::new(0.95, 0.96, 1.0), 1.2);
        key.cast_shadows = true;
        if let Some(shadow) = key.shadow.as_mut() {
            shadow.map_size = 2048;
            shadow.normal_bias = 0.0;
        }
        let key = scene.add_light(key);
        scene
            .edit_node(key)
            .set_position(10.0, 12.0, 8.0)
            .look_at(Vec3::new(0.0, 0.5, 0.0));

        let orbit_light = scene.add_light(Light::new_point(Vec3::new(0.55, 0.9, 1.0), 1.8, 28.0));
        scene.edit_node(orbit_light).set_position(0.0, 5.0, 8.0);
        let magenta_light =
            scene.add_light(Light::new_point(Vec3::new(1.0, 0.25, 0.85), 1.4, 28.0));
        scene.edit_node(magenta_light).set_position(0.0, -1.0, -6.0);

        let mut rng = 0x51A2_BA11;
        let mut balls = Vec::with_capacity(BALL_COUNT);
        for index in 0..BALL_COUNT {
            let material = palette[index % palette.len()];
            let handle = scene.add_mesh(Mesh::new(sphere_geo, material));
            scene
                .edit_node(handle)
                .set_cast_shadows(false)
                .set_receive_shadows(false);

            balls.push(VectorBall {
                handle,
                phase: Vec3::new(
                    rand_range(&mut rng, 0.0, TAU),
                    rand_range(&mut rng, 0.0, TAU),
                    rand_range(&mut rng, 0.0, TAU),
                ),
                frequency: Vec3::new(
                    rand_range(&mut rng, 0.8, 1.9),
                    rand_range(&mut rng, 1.1, 2.6),
                    rand_range(&mut rng, 0.7, 1.7),
                ),
                radius: Vec3::new(
                    rand_range(&mut rng, 2.8, 7.8),
                    rand_range(&mut rng, 1.6, 4.4),
                    rand_range(&mut rng, 2.6, 7.2),
                ),
                size: rand_range(&mut rng, 0.12, 0.34),
            });
        }

        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(cam)
            .set_position(0.0, 2.8, 15.0)
            .look_at(Vec3::new(0.0, 0.5, 0.0));
        scene.active_camera = Some(cam);

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 2.8, 15.0), Vec3::new(0.0, 0.5, 0.0)),
            fps_counter: FpsCounter::new(),
            balls,
            orbit_light,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        for (index, ball) in self.balls.iter().enumerate() {
            let t = self.time;
            let mut position = Vec3::new(
                (t * ball.frequency.x + ball.phase.x).sin() * ball.radius.x,
                (t * ball.frequency.y + ball.phase.y).sin() * ball.radius.y,
                (t * ball.frequency.z + ball.phase.z).cos() * ball.radius.z,
            );
            position.y += ((index as f32 * 0.21) + t * 0.8).cos() * 0.6;

            let pulse = 0.82
                + 0.38
                    * (t * (1.2 + ball.frequency.x * 0.25) + ball.phase.z)
                        .sin()
                        .abs();
            scene
                .edit_node(ball.handle)
                .set_position_vec(position)
                .set_scale(ball.size * pulse);
        }

        if let Some(node) = scene.node_mut(self.orbit_light) {
            node.transform.position = Vec3::new(
                (self.time * 0.75).cos() * 7.5,
                4.2 + (self.time * 1.4).sin() * 1.3,
                (self.time * 0.75).sin() * 7.5,
            );
        }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Vector Balls | FPS: {:.1}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Vector Balls")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<VectorBallsDemo>()
}
