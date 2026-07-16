//! [gallery]
//! name = "Advanced PBR Material"
//! category = "Materials"
//! description = "Classic turntable-style material study covering clearcoat, sheen, iridescence, anisotropy, transmission, and dispersion under HDR lighting."
//! instructions = "1-6 focus a material station\nR overview\nSpace toggle auto rotation"
//! order = 310
//!

use std::f32::consts::PI;

use myth::prelude::*;
use myth::resources::Key;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

struct MaterialStation {
    object: NodeHandle,
    focus: Vec3,
    label: &'static str,
    spin_speed: f32,
}

struct AdvancedPbrMaterialLab {
    camera_node: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    stations: Vec<MaterialStation>,
    orbit_light: NodeHandle,
    focus_index: Option<usize>,
    auto_spin: bool,
    time: f32,
}

impl AdvancedPbrMaterialLab {
    fn print_help() {
        println!("Advanced PBR Material Lab");
        println!("  1-6  Focus a material station");
        println!("  R    Return to overview camera");
        println!("  Space Toggle station auto rotation");
        println!("  Mouse drag / scroll Orbit and zoom camera");
    }

    fn overview_camera() -> (Vec3, Vec3) {
        (Vec3::new(0.0, 4.8, 11.5), Vec3::new(0.0, 1.2, 0.2))
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
            (focus + Vec3::new(0.0, 0.75, 3.35), focus)
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

        let label = self.focus_name();
        println!("Focus: {label}");
    }

    fn focus_name(&self) -> &'static str {
        self.focus_index
            .map(|index| self.stations[index].label)
            .unwrap_or("Overview")
    }
}

impl AppHandler for AdvancedPbrMaterialLab {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let sphere_geo = engine.assets.geometries.add(Geometry::new_sphere(1.0));

        let floor_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.07, 0.08, 0.10, 1.0)).with_roughness(0.96));
        let wall_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.10, 0.11, 0.14, 1.0)).with_roughness(0.86));
        let pedestal_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.18, 0.19, 0.23, 1.0)).with_roughness(0.58));
        let helper_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.28, 0.85, 1.0, 1.0))
                .with_emissive(Vec3::new(0.22, 0.82, 1.0), 2.2)
                .with_roughness(0.18),
        );
        let backlight_palette = [
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.08, 0.16, 0.22, 1.0))
                    .with_emissive(Vec3::new(0.25, 0.86, 1.0), 2.2)
                    .with_roughness(0.14),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.18, 0.08, 0.22, 1.0))
                    .with_emissive(Vec3::new(1.0, 0.46, 0.82), 2.1)
                    .with_roughness(0.14),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.20, 0.12, 0.06, 1.0))
                    .with_emissive(Vec3::new(1.0, 0.74, 0.28), 2.0)
                    .with_roughness(0.14),
            ),
        ];

        let material_handles = [
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.95, 0.96, 0.98, 1.0))
                    .with_metalness(1.0)
                    .with_roughness(0.10),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.78, 0.05, 0.03, 1.0))
                    .with_metalness(0.32)
                    .with_roughness(0.22)
                    .with_clearcoat(1.0, 0.04),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.18, 0.12, 0.26, 1.0))
                    .with_roughness(0.92)
                    .with_sheen(Vec3::new(0.85, 0.42, 0.96), 0.38),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.16, 0.16, 0.18, 1.0))
                    .with_roughness(0.08)
                    .with_iridescence(1.0, 1.3, 120.0, 900.0),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.98, 0.82, 0.32, 1.0))
                    .with_metalness(1.0)
                    .with_roughness(0.18)
                    .with_anisotropy(0.95, PI * 0.35),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.94, 0.97, 1.0, 1.0))
                    .with_ior(1.52)
                    .with_roughness(0.03)
                    .with_transmission(1.0, 0.55, 4.0, Vec3::new(0.96, 0.98, 1.0))
                    .with_dispersion(0.20),
            ),
        ];

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.01));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.04);
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

        let floor = scene.add_mesh(Mesh::new(box_geo, floor_material));
        scene
            .edit_node(floor)
            .set_position(0.0, -0.22, 0.0)
            .set_scale_xyz(18.0, 0.24, 12.5)
            .set_shadows(false, true);

        let wall = scene.add_mesh(Mesh::new(box_geo, wall_material));
        scene
            .edit_node(wall)
            .set_position(0.0, 2.2, -6.4)
            .set_scale_xyz(15.0, 4.4, 0.24)
            .set_shadows(false, false);

        let station_specs = [
            (
                Vec3::new(-4.2, 0.0, -1.9),
                material_handles[0],
                "Polished Metal",
                0.42,
                backlight_palette[0],
            ),
            (
                Vec3::new(0.0, 0.0, -1.9),
                material_handles[1],
                "Clearcoat Paint",
                0.34,
                backlight_palette[1],
            ),
            (
                Vec3::new(4.2, 0.0, -1.9),
                material_handles[2],
                "Velvet Sheen",
                0.28,
                backlight_palette[2],
            ),
            (
                Vec3::new(-4.2, 0.0, 2.3),
                material_handles[3],
                "Iridescent Film",
                0.58,
                backlight_palette[1],
            ),
            (
                Vec3::new(0.0, 0.0, 2.3),
                material_handles[4],
                "Brushed Metal",
                0.46,
                backlight_palette[2],
            ),
            (
                Vec3::new(4.2, 0.0, 2.3),
                material_handles[5],
                "Transmission Glass",
                0.36,
                backlight_palette[0],
            ),
        ];

        let mut stations = Vec::new();
        for (position, material, label, spin_speed, backlight_material) in station_specs {
            let pedestal = scene.add_mesh(Mesh::new(box_geo, pedestal_material));
            scene
                .edit_node(pedestal)
                .set_position(position.x, 0.40, position.z)
                .set_scale_xyz(1.75, 0.80, 1.75)
                .set_shadows(true, true);

            let backlight = scene.add_mesh(Mesh::new(box_geo, backlight_material));
            scene
                .edit_node(backlight)
                .set_position(position.x, 1.95, position.z - 1.55)
                .set_scale_xyz(0.22, 2.9, 0.22)
                .set_shadows(false, false);

            let object = scene.add_mesh(Mesh::new(sphere_geo, material));
            scene
                .edit_node(object)
                .set_position(position.x, 1.58, position.z)
                .set_scale(1.18)
                .set_shadows(true, true);

            stations.push(MaterialStation {
                object,
                focus: position + Vec3::new(0.0, 1.58, 0.0),
                label,
                spin_speed,
            });
        }

        let mut key_light = Light::new_directional(Vec3::new(1.0, 0.97, 0.93), 3.2);
        key_light.cast_shadows = true;
        if let Some(shadow) = key_light.shadow.as_mut() {
            shadow.map_size = 2048;
            shadow.normal_bias = 0.0;
        }
        let key_light = scene.add_light(key_light);
        scene
            .edit_node(key_light)
            .set_position(9.0, 12.0, 6.5)
            .look_at(Vec3::new(0.0, 1.2, 0.0));

        let fill_light = scene.add_light(Light::new_point(Vec3::new(1.0, 0.56, 0.24), 0.9, 18.0));
        scene.edit_node(fill_light).set_position(-6.5, 2.8, 4.0);

        let orbit_light = scene.add_light(Light::new_point(Vec3::new(0.32, 0.84, 1.0), 1.7, 28.0));
        scene.edit_node(orbit_light).set_position(0.0, 4.8, 6.2);
        let helper = scene.add_mesh_to_parent(Mesh::new(sphere_geo, helper_material), orbit_light);
        scene
            .edit_node(helper)
            .set_scale(0.15)
            .set_shadows(false, false);

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
            orbit_light,
            focus_index: None,
            auto_spin: true,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

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
                "Station auto rotation: {}",
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

        if let Some(node) = scene.node_mut(self.orbit_light) {
            let orbit = self.time * 0.72;
            node.transform.position = Vec3::new(
                6.2 * orbit.cos(),
                4.7 + 0.55 * (self.time * 1.9).sin(),
                6.2 * orbit.sin(),
            );
        }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "Advanced PBR Material Lab | Focus: {} | Auto Spin: {} | FPS: {:.1}",
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
        .with_title("Advanced PBR Material Lab")
        .with_settings(RendererSettings {
            path: RenderPath::HighFidelity,
            vsync: false,
            ..Default::default()
        })
        .run::<AdvancedPbrMaterialLab>()
}
