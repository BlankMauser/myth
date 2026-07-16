//! [gallery]
//! name = "Cube Wave"
//! category = "Foundations"
//! description = "A classic wave-field of animated cubes that highlights transform updates, lighting, and scene scale."
//! order = 140
//!

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

const GRID_RADIUS: i32 = 8;
const CELL_SPACING: f32 = 0.9;

struct CubeCell {
    handle: NodeHandle,
    base: Vec2,
    phase: f32,
}

struct CubeWaveDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    cubes: Vec<CubeCell>,
    orbit_light: NodeHandle,
    time: f32,
}

impl AppHandler for CubeWaveDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let sphere_geo = engine.assets.geometries.add(Geometry::new_sphere(1.0));

        let floor_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.10, 0.12, 0.14, 1.0)).with_roughness(0.94));
        let helper_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.48, 0.86, 1.0, 1.0))
                .with_emissive(Vec3::new(0.22, 0.75, 1.0), 1.4)
                .with_roughness(0.18),
        );
        let palette = vec![
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.93, 0.36, 0.20, 1.0))
                    .with_metalness(0.24)
                    .with_roughness(0.18),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.98, 0.76, 0.24, 1.0))
                    .with_metalness(0.10)
                    .with_roughness(0.22),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.22, 0.66, 0.94, 1.0))
                    .with_metalness(0.32)
                    .with_roughness(0.16),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.12, 0.84, 0.62, 1.0))
                    .with_metalness(0.16)
                    .with_roughness(0.20),
            ),
        ];

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.02));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.06);
        scene.bloom.set_radius(0.005);

        let floor = scene.add_mesh(Mesh::new(box_geo, floor_material));
        scene
            .edit_node(floor)
            .set_position(0.0, -0.15, 0.0)
            .set_scale_xyz(24.0, 0.30, 24.0)
            .set_shadows(false, true);

        let mut cubes = Vec::new();
        for z in -GRID_RADIUS..=GRID_RADIUS {
            for x in -GRID_RADIUS..=GRID_RADIUS {
                let base = Vec2::new(x as f32 * CELL_SPACING, z as f32 * CELL_SPACING);
                let material_index = ((base.length() / (CELL_SPACING * 1.35)).floor() as usize
                    + (x + z).rem_euclid(2) as usize)
                    % palette.len();

                let cube = scene.add_mesh(Mesh::new(box_geo, palette[material_index]));
                scene
                    .edit_node(cube)
                    .set_position(base.x, 0.35, base.y)
                    .set_scale_xyz(0.72, 0.72, 0.72)
                    .set_shadows(true, true);

                cubes.push(CubeCell {
                    handle: cube,
                    base,
                    phase: base.length() * 0.7 + (x as f32 * 0.30) - (z as f32 * 0.25),
                });
            }
        }

        let mut sun = Light::new_directional(Vec3::new(1.0, 0.98, 0.94), 2.8);
        sun.cast_shadows = true;
        if let Some(shadow) = sun.shadow.as_mut() {
            shadow.map_size = 2048;
            shadow.normal_bias = 0.0;
        }
        let sun = scene.add_light(sun);
        scene
            .edit_node(sun)
            .set_position(10.0, 12.0, 5.0)
            .look_at(Vec3::new(0.0, 1.0, 0.0));

        let orbit_light = scene.add_light(Light::new_point(Vec3::new(0.32, 0.82, 1.0), 1.4, 30.0));
        scene.edit_node(orbit_light).set_position(0.0, 4.5, 6.0);
        let light_helper =
            scene.add_mesh_to_parent(Mesh::new(sphere_geo, helper_material), orbit_light);
        scene
            .edit_node(light_helper)
            .set_scale(0.15)
            .set_cast_shadows(false)
            .set_receive_shadows(false);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(cam)
            .set_position(10.0, 10.0, 10.0)
            .look_at(Vec3::new(0.0, 1.0, 0.0));
        scene.active_camera = Some(cam);

        Self {
            controls: OrbitControls::new(Vec3::new(10.0, 10.0, 10.0), Vec3::new(0.0, 1.0, 0.0)),
            fps_counter: FpsCounter::new(),
            cubes,
            orbit_light,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        let wave_time = self.time * 3.6;
        for cell in &self.cubes {
            let radial = cell.base.length();
            let ripple = (radial * 1.2 - wave_time).sin();
            let cross = ((cell.base.x - cell.base.y) * 0.55 + self.time * 2.1).cos();
            let pulse = ((ripple * 0.7 + cross * 0.3) * 0.5 + 0.5).clamp(0.0, 1.0);
            let height = 0.35 + pulse.powf(1.8) * 3.6;
            let twist = 0.12 * (self.time * 0.9 + cell.phase).sin();

            scene
                .edit_node(cell.handle)
                .set_position(cell.base.x, height * 0.5, cell.base.y)
                .set_scale_xyz(0.74, height, 0.74)
                .set_rotation(Quat::from_euler(
                    EulerRot::XYZ,
                    twist * 0.6,
                    self.time * 0.25 + cell.phase * 0.08,
                    -twist,
                ));
        }

        if let Some(node) = scene.node_mut(self.orbit_light) {
            let orbit = self.time * 0.75;
            node.transform.position = Vec3::new(
                6.0 * orbit.cos(),
                4.5 + 0.8 * (self.time * 1.4).sin(),
                6.0 * orbit.sin(),
            );
        }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Cube Wave | FPS: {:.1}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Cube Wave")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<CubeWaveDemo>()
}
