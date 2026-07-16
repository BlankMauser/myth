//! [gallery]
//! name = "Hyperspace Starfield"
//! category = "Post Effects"
//! description = "Classic warp-speed starfield with a neon jump gate, bloom, and hundreds of animated streaks."
//! instructions = "Space: engage hyperdrive"
//! order = 452
//!

use std::f32::consts::TAU;

use myth::prelude::*;
use myth::resources::Key;
use myth_dev_utils::FpsCounter;

const STAR_COUNT: usize = 900;
const STAR_FAR_Z: f32 = -160.0;
const STAR_NEAR_Z: f32 = 12.0;

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

fn respawn_star(seed: &mut u32, star: &mut Star, far_only: bool) {
    let radius = rand_range(seed, 2.0, 34.0) * rand_f32(seed).sqrt();
    let angle = rand_range(seed, 0.0, TAU);
    let z_min = if far_only {
        STAR_FAR_Z
    } else {
        STAR_FAR_Z + 22.0
    };
    let z_max = if far_only {
        STAR_FAR_Z + 28.0
    } else {
        STAR_NEAR_Z - 10.0
    };

    star.position = Vec3::new(
        angle.cos() * radius,
        angle.sin() * radius * 0.62,
        rand_range(seed, z_min, z_max),
    );
    star.speed = rand_range(seed, 0.8, 1.8);
    star.size = rand_range(seed, 0.035, 0.11);
}

struct Star {
    handle: NodeHandle,
    position: Vec3,
    speed: f32,
    size: f32,
}

struct HyperspaceStarfieldDemo {
    stars: Vec<Star>,
    cam_node_id: NodeHandle,
    gate_root: NodeHandle,
    fps_counter: FpsCounter,
    rng: u32,
    time: f32,
}

impl AppHandler for HyperspaceStarfieldDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));

        let star_palette = [
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.06, 0.08, 0.12, 1.0))
                    .with_emissive(Vec3::new(0.85, 0.95, 1.0), 4.2)
                    .with_roughness(0.08),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.08, 0.10, 0.16, 1.0))
                    .with_emissive(Vec3::new(0.45, 0.78, 1.0), 5.2)
                    .with_roughness(0.12),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.12, 0.08, 0.18, 1.0))
                    .with_emissive(Vec3::new(1.0, 0.48, 0.92), 4.8)
                    .with_roughness(0.14),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.14, 0.10, 0.04, 1.0))
                    .with_emissive(Vec3::new(1.0, 0.78, 0.32), 4.4)
                    .with_roughness(0.16),
            ),
        ];
        let gate_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.08, 0.10, 0.18, 1.0))
                .with_emissive(Vec3::new(0.35, 0.85, 1.0), 4.8)
                .with_roughness(0.18),
        );
        let core_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.18, 0.10, 0.24, 1.0))
                .with_emissive(Vec3::new(1.0, 0.3, 0.9), 5.8)
                .with_roughness(0.16),
        );

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.002));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.14);
        scene.bloom.set_radius(0.006);
        scene.bloom.set_karis_average(true);

        let mut rng = 0xA1C3_5EED;
        let mut stars = Vec::with_capacity(STAR_COUNT);
        for _ in 0..STAR_COUNT {
            let material_index = (next_rand(&mut rng) as usize) % star_palette.len();
            let star_handle = scene.add_mesh(Mesh::new(box_geo, star_palette[material_index]));
            scene
                .edit_node(star_handle)
                .set_cast_shadows(false)
                .set_receive_shadows(false);

            let mut star = Star {
                handle: star_handle,
                position: Vec3::ZERO,
                speed: 1.0,
                size: 0.04,
            };
            respawn_star(&mut rng, &mut star, false);
            stars.push(star);
        }

        let gate_root = scene.create_node_with_name("JumpGate");
        scene.push_root_node(gate_root);
        scene.edit_node(gate_root).set_position(0.0, 0.0, -68.0);

        for ring_index in 0..2 {
            let radius = if ring_index == 0 { 6.0 } else { 8.0 };
            let segment_count = if ring_index == 0 { 40 } else { 24 };
            for segment in 0..segment_count {
                let angle = segment as f32 / segment_count as f32 * TAU;
                let segment_handle = scene.add_mesh_to_parent(
                    Mesh::new(
                        box_geo,
                        if ring_index == 0 {
                            gate_material
                        } else {
                            core_material
                        },
                    ),
                    gate_root,
                );
                let radial = Vec3::new(angle.cos(), angle.sin(), 0.0);
                let tangential = angle + std::f32::consts::FRAC_PI_2;
                scene
                    .edit_node(segment_handle)
                    .set_position_vec(radial * radius)
                    .set_rotation(Quat::from_rotation_z(tangential))
                    .set_scale_xyz(0.30, 1.5 - ring_index as f32 * 0.35, 0.18)
                    .set_cast_shadows(false)
                    .set_receive_shadows(false);
            }
        }

        let core = scene.add_mesh_to_parent(Mesh::new(box_geo, core_material), gate_root);
        scene
            .edit_node(core)
            .set_scale_xyz(0.45, 0.45, 0.45)
            .set_cast_shadows(false)
            .set_receive_shadows(false);

        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(cam_node_id)
            .set_position(0.0, 0.0, 8.0)
            .look_at(Vec3::new(0.0, 0.0, -50.0));
        scene.active_camera = Some(cam_node_id);

        Self {
            stars,
            cam_node_id,
            gate_root,
            fps_counter: FpsCounter::new(),
            rng,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let hyperdrive = engine.input.get_key(Key::Space);
        let warp = if hyperdrive { 4.8 } else { 1.0 };
        self.time += frame.dt * warp;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        scene
            .edit_node(self.cam_node_id)
            .set_position(
                (self.time * 0.9).sin() * 0.16,
                (self.time * 0.7).cos() * 0.12,
                8.0,
            )
            .look_at(Vec3::new(0.0, 0.0, -50.0));

        for star in &mut self.stars {
            star.position.z += frame.dt * (20.0 + star.speed * 34.0 * warp);
            if star.position.z > STAR_NEAR_Z {
                respawn_star(&mut self.rng, star, true);
            }

            let depth =
                ((star.position.z - STAR_FAR_Z) / (STAR_NEAR_Z - STAR_FAR_Z)).clamp(0.0, 1.0);
            let width = star.size * (0.8 + depth * 1.6);
            let stretch = star.size * (1.0 + depth * depth * 22.0 * warp);

            scene
                .edit_node(star.handle)
                .set_position_vec(star.position)
                .set_scale_xyz(width, width, stretch);
        }

        let gate_scale = 1.0 + (self.time * 2.3).sin() * 0.06;
        scene
            .edit_node(self.gate_root)
            .set_rotation(Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, self.time * 0.45))
            .set_scale(gate_scale);

        if let Some(fps) = self.fps_counter.update() {
            let mode = if hyperdrive { "HYPERDRIVE" } else { "Cruise" };
            window.set_title(&format!(
                "Hyperspace Starfield | {} | FPS: {:.1}",
                mode, fps
            ));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Hyperspace Starfield")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<HyperspaceStarfieldDemo>()
}
