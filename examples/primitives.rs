//! [gallery]
//! name = "Primitives"
//! category = "Foundations"
//! description = "Classic showroom for SceneExt spawn_* helpers, covering box, sphere, plane, cylinder, cone, and torus primitives with focusable stations."
//! instructions = "1-6 focus a primitive\nR overview\nSpace toggle auto rotation"
//! order = 130
//!

use std::f32::consts::{FRAC_PI_2, PI};

use myth::prelude::*;
use myth::resources::Key;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

struct PrimitiveStation {
    object: NodeHandle,
    focus: Vec3,
    label: &'static str,
    spin_speed: f32,
}

struct PrimitiveSpawnShowcase {
    camera_node: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    stations: Vec<PrimitiveStation>,
    focus_index: Option<usize>,
    auto_spin: bool,
}

impl PrimitiveSpawnShowcase {
    fn print_help() {
        println!("Primitive Spawn Showcase");
        println!("  1-6  Focus a primitive station");
        println!("  R    Return to overview camera");
        println!("  Space Toggle primitive auto rotation");
        println!("  Mouse drag / scroll Orbit and zoom camera");
    }

    fn overview_camera() -> (Vec3, Vec3) {
        (Vec3::new(0.0, 4.8, 12.2), Vec3::new(0.0, 1.4, 0.2))
    }

    fn build_controls(camera_pos: Vec3, target: Vec3) -> OrbitControls {
        let mut controls = OrbitControls::new(camera_pos, target);
        controls.min_distance = 1.8;
        controls.max_distance = 18.0;
        controls.pan_speed = 0.85;
        controls.rotate_speed = 0.12;
        controls
    }

    fn set_focus(&mut self, scene: &mut Scene, focus_index: Option<usize>) {
        let (camera_pos, target) = if let Some(index) = focus_index {
            let focus = self.stations[index].focus;
            (focus + Vec3::new(0.0, 0.7, 3.35), focus)
        } else {
            Self::overview_camera()
        };

        scene
            .edit_node(self.camera_node)
            .set_position(camera_pos.x, camera_pos.y, camera_pos.z)
            .look_at(target);
        self.controls.set_target(target);
        self.controls.set_position(camera_pos);
        self.focus_index = focus_index;

        println!("Focus: {}", self.focus_name());
    }

    fn focus_name(&self) -> &'static str {
        self.focus_index
            .map(|index| self.stations[index].label)
            .unwrap_or("Overview")
    }
}

impl AppHandler for PrimitiveSpawnShowcase {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.01));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.035);
        scene.bloom.set_radius(0.005);
        scene.bloom.set_karis_average(true);
        scene
            .tone_mapping
            .set_mode(myth::ToneMappingMode::AgX(myth::AgxLook::Punchy));

        let env_texture = engine
            .assets
            .load_hdr_texture(format!("{}envs/blouberg_sunrise_2_1k.hdr", ASSET_PATH));
        scene.environment.set_env_map(Some(env_texture));
        scene.environment.set_intensity(1.0);
        scene
            .background
            .set_mode(BackgroundMode::equirectangular(env_texture, 1.0));

        let floor_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.10, 0.11, 0.14, 1.0))
                .with_roughness(0.95)
                .with_side(Side::Double),
        );
        let wall_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.12, 0.13, 0.17, 1.0))
                .with_roughness(0.86)
                .with_side(Side::Double),
        );
        let pedestal_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.20, 0.21, 0.26, 1.0)).with_roughness(0.54));

        let primitive_materials = [
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.92, 0.32, 0.22, 1.0))
                    .with_roughness(0.26)
                    .with_clearcoat(0.9, 0.12),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.28, 0.62, 0.98, 1.0))
                    .with_roughness(0.18)
                    .with_metalness(0.05),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.86, 0.78, 0.30, 1.0))
                    .with_roughness(0.24)
                    .with_metalness(0.92),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.82, 0.32, 0.94, 1.0))
                    .with_roughness(0.40)
                    .with_side(Side::Double),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.24, 0.92, 0.70, 1.0))
                    .with_roughness(0.34)
                    .with_anisotropy(0.85, PI * 0.25),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.94, 0.97, 1.0, 1.0))
                    .with_ior(1.46)
                    .with_roughness(0.06)
                    .with_transmission(0.85, 0.20, 4.0, Vec3::new(0.92, 0.97, 1.0)),
            ),
        ];

        let floor = scene.spawn_plane(18.0, 12.0, floor_material, &engine.assets);
        scene
            .edit_node(floor)
            .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
            .set_position(0.0, -0.02, 0.0)
            .set_shadows(false, true);

        let back_wall = scene.spawn_plane(15.0, 5.4, wall_material, &engine.assets);
        scene
            .edit_node(back_wall)
            .set_position(0.0, 2.4, -6.2)
            .set_shadows(false, false);

        let mut stations = Vec::new();

        let box_pedestal = scene.spawn_box(1.75, 0.60, 1.75, pedestal_material, &engine.assets);
        scene
            .edit_node(box_pedestal)
            .set_position(-4.2, 0.30, -1.9)
            .set_shadows(true, true);
        let box_node = scene.spawn_box(1.35, 1.35, 1.35, primitive_materials[0], &engine.assets);
        scene
            .edit_node(box_node)
            .set_position(-4.2, 1.28, -1.9)
            .set_rotation(Quat::from_rotation_y(0.32))
            .set_shadows(true, true);
        stations.push(PrimitiveStation {
            object: box_node,
            focus: Vec3::new(-4.2, 1.28, -1.9),
            label: "Box",
            spin_speed: 0.34,
        });

        let sphere_pedestal = scene.spawn_box(1.75, 0.60, 1.75, pedestal_material, &engine.assets);
        scene
            .edit_node(sphere_pedestal)
            .set_position(0.0, 0.30, -1.9)
            .set_shadows(true, true);
        let sphere_node = scene.spawn_sphere(0.84, primitive_materials[1], &engine.assets);
        scene
            .edit_node(sphere_node)
            .set_position(0.0, 1.44, -1.9)
            .set_shadows(true, true);
        stations.push(PrimitiveStation {
            object: sphere_node,
            focus: Vec3::new(0.0, 1.44, -1.9),
            label: "Sphere",
            spin_speed: 0.26,
        });

        let cylinder_pedestal =
            scene.spawn_box(1.75, 0.60, 1.75, pedestal_material, &engine.assets);
        scene
            .edit_node(cylinder_pedestal)
            .set_position(4.2, 0.30, -1.9)
            .set_shadows(true, true);
        let cylinder_node =
            scene.spawn_cylinder(0.70, 1.95, primitive_materials[2], &engine.assets);
        scene
            .edit_node(cylinder_node)
            .set_position(4.2, 1.58, -1.9)
            .set_rotation(Quat::from_rotation_y(0.22))
            .set_shadows(true, true);
        stations.push(PrimitiveStation {
            object: cylinder_node,
            focus: Vec3::new(4.2, 1.58, -1.9),
            label: "Cylinder",
            spin_speed: 0.30,
        });

        let plane_pedestal = scene.spawn_box(1.75, 0.60, 1.75, pedestal_material, &engine.assets);
        scene
            .edit_node(plane_pedestal)
            .set_position(-4.2, 0.30, 2.3)
            .set_shadows(true, true);
        let plane_node = scene.spawn_plane(1.85, 1.85, primitive_materials[3], &engine.assets);
        scene
            .edit_node(plane_node)
            .set_position(-4.2, 1.46, 2.3)
            .set_rotation(Quat::from_euler(EulerRot::XYZ, -0.30, 0.42, 0.0))
            .set_shadows(true, true);
        stations.push(PrimitiveStation {
            object: plane_node,
            focus: Vec3::new(-4.2, 1.46, 2.3),
            label: "Plane",
            spin_speed: 0.24,
        });

        let cone_pedestal = scene.spawn_box(1.75, 0.60, 1.75, pedestal_material, &engine.assets);
        scene
            .edit_node(cone_pedestal)
            .set_position(0.0, 0.30, 2.3)
            .set_shadows(true, true);
        let cone_node = scene.spawn_cone(0.92, 1.95, primitive_materials[4], &engine.assets);
        scene
            .edit_node(cone_node)
            .set_position(0.0, 1.58, 2.3)
            .set_rotation(Quat::from_rotation_y(0.28))
            .set_shadows(true, true);
        stations.push(PrimitiveStation {
            object: cone_node,
            focus: Vec3::new(0.0, 1.58, 2.3),
            label: "Cone",
            spin_speed: 0.32,
        });

        let torus_pedestal = scene.spawn_box(1.75, 0.60, 1.75, pedestal_material, &engine.assets);
        scene
            .edit_node(torus_pedestal)
            .set_position(4.2, 0.30, 2.3)
            .set_shadows(true, true);
        let torus_node = scene.spawn_torus(0.92, 0.28, primitive_materials[5], &engine.assets);
        scene
            .edit_node(torus_node)
            .set_position(4.2, 1.52, 2.3)
            .set_rotation(Quat::from_euler(EulerRot::XYZ, 0.58, 0.22, 0.0))
            .set_shadows(true, true);
        stations.push(PrimitiveStation {
            object: torus_node,
            focus: Vec3::new(4.2, 1.52, 2.3),
            label: "Torus",
            spin_speed: 0.38,
        });

        let mut key_light = Light::new_directional(Vec3::new(1.0, 0.97, 0.93), 3.0);
        key_light.cast_shadows = true;
        if let Some(shadow) = key_light.shadow.as_mut() {
            shadow.map_size = 2048;
            shadow.normal_bias = 0.0;
        }
        let key_light = scene.add_light(key_light);
        scene
            .edit_node(key_light)
            .set_position(8.5, 11.5, 7.0)
            .look_at(Vec3::new(0.0, 1.2, 0.0));

        let fill_light = scene.add_light(Light::new_point(Vec3::new(0.28, 0.86, 1.0), 1.0, 18.0));
        scene.edit_node(fill_light).set_position(-6.6, 3.0, 4.2);

        let rim_light = scene.add_light(Light::new_point(Vec3::new(1.0, 0.58, 0.24), 1.1, 20.0));
        scene.edit_node(rim_light).set_position(6.8, 3.6, -2.8);

        let (camera_pos, target) = Self::overview_camera();
        let camera_node = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera_node)
            .set_position(camera_pos.x, camera_pos.y, camera_pos.z)
            .look_at(target);
        scene.active_camera = Some(camera_node);

        Self::print_help();

        Self {
            camera_node,
            controls: Self::build_controls(camera_pos, target),
            fps_counter: FpsCounter::new(),
            stations,
            focus_index: None,
            auto_spin: true,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        let requested_focus = if engine.input.get_key_down(Key::Key1) {
            Some(Some(0))
        } else if engine.input.get_key_down(Key::Key2) {
            Some(Some(1))
        } else if engine.input.get_key_down(Key::Key3) {
            Some(Some(2))
        } else if engine.input.get_key_down(Key::Key4) {
            Some(Some(3))
        } else if engine.input.get_key_down(Key::Key5) {
            Some(Some(4))
        } else if engine.input.get_key_down(Key::Key6) {
            Some(Some(5))
        } else if engine.input.get_key_down(Key::R) {
            Some(None)
        } else {
            None
        };

        if let Some(focus) = requested_focus {
            self.set_focus(scene, focus);
        }

        if engine.input.get_key_down(Key::Space) {
            self.auto_spin = !self.auto_spin;
            println!(
                "Primitive auto rotation: {}",
                if self.auto_spin { "on" } else { "off" }
            );
        }

        if self.auto_spin {
            for station in &self.stations {
                if let Some(node) = scene.node_mut(station.object) {
                    node.transform.rotation *= Quat::from_rotation_y(frame.dt * station.spin_speed);
                }
            }
        }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "Primitive Spawn Showcase | Focus: {} | Auto Spin: {} | FPS: {:.1}",
                self.focus_name(),
                if self.auto_spin { "on" } else { "off" },
                fps
            ));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Primitive Spawn Showcase")
        .with_settings(RendererSettings {
            path: RenderPath::HighFidelity,
            vsync: false,
            ..Default::default()
        })
        .run::<PrimitiveSpawnShowcase>()
}
