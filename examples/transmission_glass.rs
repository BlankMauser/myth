//! [gallery]
//! name = "Transmission & Dispersion"
//! category = "Post Effects"
//! description = "Classic glass showcase with clear, tinted, frosted, and prismatic transmission materials over a patterned HDR-lit stage."
//! instructions = "1-4 focus a glass preset\nR overview\nSpace toggle orbit light"
//! order = 444
//!

use myth::prelude::*;
use myth::resources::Key;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

struct GlassStation {
    object: NodeHandle,
    focus: Vec3,
    label: &'static str,
    spin_speed: f32,
}

struct TransmissionGlassLab {
    camera_node: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    stations: Vec<GlassStation>,
    orbit_light: NodeHandle,
    focus_index: Option<usize>,
    orbit_light_enabled: bool,
    time: f32,
}

impl TransmissionGlassLab {
    fn print_help() {
        println!("Transmission & Dispersion Lab");
        println!("  1-4  Focus a glass preset");
        println!("  R    Return to overview camera");
        println!("  Space Toggle orbit light motion");
        println!("  Mouse drag / scroll Orbit and zoom camera");
    }

    fn overview_camera() -> (Vec3, Vec3) {
        (Vec3::new(0.0, 4.2, 10.4), Vec3::new(0.0, 1.3, 0.2))
    }

    fn build_controls(camera_pos: Vec3, target: Vec3) -> OrbitControls {
        let mut controls = OrbitControls::new(camera_pos, target);
        controls.min_distance = 1.6;
        controls.max_distance = 16.0;
        controls.pan_speed = 0.85;
        controls.rotate_speed = 0.12;
        controls
    }

    fn set_focus(&mut self, scene: &mut Scene, focus_index: Option<usize>) {
        let (camera_pos, target) = if let Some(index) = focus_index {
            let focus = self.stations[index].focus;
            (focus + Vec3::new(0.0, 0.60, 3.05), focus)
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

impl AppHandler for TransmissionGlassLab {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let sphere_geo = engine.assets.geometries.add(Geometry::new_sphere(1.0));

        let checker_image = engine
            .assets
            .images
            .add(Image::checkerboard(1024, 1024, 128));
        let checker_texture = engine
            .assets
            .textures
            .add(Texture::new_2d(Some("glass_lab_checker"), checker_image));

        let floor_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.92, 0.92, 0.94, 1.0))
                .with_map(checker_texture)
                .with_roughness(0.94),
        );
        let pedestal_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.12, 0.13, 0.16, 1.0)).with_roughness(0.62));
        let helper_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(1.0, 0.80, 0.36, 1.0))
                .with_emissive(Vec3::new(1.0, 0.74, 0.28), 2.6)
                .with_roughness(0.18),
        );
        let bar_palette = [
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.08, 0.14, 0.18, 1.0))
                    .with_emissive(Vec3::new(0.20, 0.86, 1.0), 2.4)
                    .with_roughness(0.10),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.16, 0.08, 0.20, 1.0))
                    .with_emissive(Vec3::new(1.0, 0.42, 0.86), 2.3)
                    .with_roughness(0.10),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.18, 0.11, 0.05, 1.0))
                    .with_emissive(Vec3::new(1.0, 0.62, 0.20), 2.2)
                    .with_roughness(0.10),
            ),
        ];

        let glass_materials = [
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.96, 0.98, 1.0, 1.0))
                    .with_ior(1.52)
                    .with_roughness(0.03)
                    .with_transmission(1.0, 0.35, 6.0, Vec3::ONE),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(1.0, 0.84, 0.58, 1.0))
                    .with_ior(1.47)
                    .with_roughness(0.08)
                    .with_transmission(1.0, 1.10, 0.70, Vec3::new(1.0, 0.58, 0.18)),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.88, 0.94, 1.0, 1.0))
                    .with_ior(1.38)
                    .with_roughness(0.36)
                    .with_transmission(1.0, 0.90, 2.4, Vec3::new(0.84, 0.93, 1.0)),
            ),
            engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.96, 0.98, 1.0, 1.0))
                    .with_ior(1.65)
                    .with_roughness(0.02)
                    .with_transmission(1.0, 0.70, 3.2, Vec3::new(0.96, 0.99, 1.0))
                    .with_dispersion(0.46),
            ),
        ];

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.01));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.06);
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
            .set_position(0.0, -0.14, 0.0)
            .set_scale_xyz(16.0, 0.18, 11.0)
            .set_shadows(false, true);

        for index in 0..12 {
            let bar = scene.add_mesh(Mesh::new(box_geo, bar_palette[index % bar_palette.len()]));
            let x = -6.05 + index as f32 * 1.10;
            let height = 2.8 + (index % 3) as f32 * 0.75;
            scene
                .edit_node(bar)
                .set_position(x, height * 0.5 + 0.35, -4.8)
                .set_scale_xyz(0.46, height, 0.22)
                .set_shadows(false, false);
        }

        let station_specs = [
            (
                Vec3::new(-4.5, 0.0, 0.3),
                glass_materials[0],
                "Clear Glass",
                0.36,
            ),
            (
                Vec3::new(-1.5, 0.0, 0.0),
                glass_materials[1],
                "Amber Absorption",
                0.28,
            ),
            (
                Vec3::new(1.5, 0.0, 0.0),
                glass_materials[2],
                "Frosted Glass",
                0.22,
            ),
            (
                Vec3::new(4.5, 0.0, 0.3),
                glass_materials[3],
                "Prism Dispersion",
                0.46,
            ),
        ];

        let mut stations = Vec::new();
        for (position, material, label, spin_speed) in station_specs {
            let pedestal = scene.add_mesh(Mesh::new(box_geo, pedestal_material));
            scene
                .edit_node(pedestal)
                .set_position(position.x, 0.32, position.z)
                .set_scale_xyz(1.50, 0.64, 1.50)
                .set_shadows(true, true);

            let object = scene.add_mesh(Mesh::new(sphere_geo, material));
            scene
                .edit_node(object)
                .set_position(position.x, 1.42, position.z)
                .set_scale(1.06)
                .set_shadows(true, true);

            stations.push(GlassStation {
                object,
                focus: position + Vec3::new(0.0, 1.42, 0.0),
                label,
                spin_speed,
            });
        }

        let mut key_light = Light::new_directional(Vec3::new(1.0, 0.97, 0.93), 2.4);
        key_light.cast_shadows = true;
        if let Some(shadow) = key_light.shadow.as_mut() {
            shadow.map_size = 2048;
            shadow.normal_bias = 0.0;
        }
        let key_light = scene.add_light(key_light);
        scene
            .edit_node(key_light)
            .set_position(8.0, 12.0, 4.5)
            .look_at(Vec3::new(0.0, 1.1, 0.0));

        let fill_light = scene.add_light(Light::new_point(Vec3::new(0.30, 0.88, 1.0), 1.0, 18.0));
        scene.edit_node(fill_light).set_position(-6.4, 2.8, 3.0);

        let orbit_light = scene.add_light(Light::new_point(Vec3::new(1.0, 0.74, 0.30), 1.7, 24.0));
        scene.edit_node(orbit_light).set_position(0.0, 4.6, 6.0);
        let helper = scene.add_mesh_to_parent(Mesh::new(sphere_geo, helper_material), orbit_light);
        scene
            .edit_node(helper)
            .set_scale(0.14)
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
            orbit_light_enabled: true,
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
        } else if engine.input.get_key_down(Key::R) {
            Some(None)
        } else {
            None
        };

        if let Some(focus) = requested_focus {
            self.set_focus(scene, focus);
        }

        if engine.input.get_key_down(Key::Space) {
            self.orbit_light_enabled = !self.orbit_light_enabled;
            println!(
                "Orbit light motion: {}",
                if self.orbit_light_enabled {
                    "on"
                } else {
                    "off"
                }
            );
        }

        for station in &self.stations {
            if let Some(node) = scene.node_mut(station.object) {
                node.transform.rotation *= Quat::from_rotation_y(frame.dt * station.spin_speed);
            }
        }

        if self.orbit_light_enabled {
            if let Some(node) = scene.node_mut(self.orbit_light) {
                let orbit = self.time * 0.95;
                node.transform.position = Vec3::new(
                    5.4 * orbit.cos(),
                    4.5 + 0.45 * (self.time * 2.1).sin(),
                    6.0 + 1.6 * orbit.sin(),
                );
            }
        }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "Transmission & Dispersion Lab | Focus: {} | Orbit Light: {} | FPS: {:.1}",
                self.focus_name(),
                if self.orbit_light_enabled {
                    "on"
                } else {
                    "off"
                },
                fps
            ));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Transmission & Dispersion Lab")
        .with_settings(RendererSettings {
            path: RenderPath::HighFidelity,
            vsync: false,
            ..Default::default()
        })
        .run::<TransmissionGlassLab>()
}
