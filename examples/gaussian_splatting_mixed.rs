//! [gallery]
//! name = "3DGS Mesh Occlusion"
//! category = "Gaussian Splatting"
//! description = "Shows depth-based occlusion between a 3D Gaussian Splatting asset and regular mesh geometry in the same render graph."
//! order = 510
//! features = ["3dgs", "gaussian-npz"]
//!

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

struct RotatingProp {
    handle: NodeHandle,
    angular_speed: f32,
}

struct GaussianSplattingMixedDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    animated_orb: NodeHandle,
    rotating_props: [RotatingProp; 2],
    time: f32,
}

impl AppHandler for GaussianSplattingMixedDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geometry = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let sphere_geometry = engine.assets.geometries.add(Geometry::new_sphere(1.0));

        let floor_material = engine
            .assets
            .materials
            .add(UnlitMaterial::new(Vec4::new(0.08, 0.10, 0.14, 1.0)));
        let pedestal_material = engine
            .assets
            .materials
            .add(UnlitMaterial::new(Vec4::new(0.34, 0.44, 0.56, 1.0)));
        let accent_material = engine
            .assets
            .materials
            .add(UnlitMaterial::new(Vec4::new(0.92, 0.54, 0.22, 1.0)));
        let orb_material = engine
            .assets
            .materials
            .add(UnlitMaterial::new(Vec4::new(0.30, 0.90, 1.0, 1.0)));

        let scene = engine.scene_manager.create_active();
        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.03, 0.05, 0.10, 1.0),
            Vec4::new(0.13, 0.10, 0.08, 1.0),
        ));

        let floor = scene.add_mesh(Mesh::new(box_geometry, floor_material));
        scene
            .edit_node(floor)
            .set_position(0.0, -0.55, 0.0)
            .set_scale_xyz(14.0, 0.25, 14.0);

        let left_pedestal = scene.add_mesh(Mesh::new(box_geometry, pedestal_material));
        scene
            .edit_node(left_pedestal)
            .set_position(-2.45, 0.45, 1.10)
            .set_scale_xyz(0.95, 0.90, 0.95)
            .set_rotation_euler(0.0, 0.45, 0.0);

        let right_pedestal = scene.add_mesh(Mesh::new(box_geometry, accent_material));
        scene
            .edit_node(right_pedestal)
            .set_position(2.15, 0.72, -1.35)
            .set_scale_xyz(0.85, 1.45, 0.85)
            .set_rotation_euler(0.0, -0.35, 0.0);

        let rear_stage = scene.add_mesh(Mesh::new(box_geometry, pedestal_material));
        scene
            .edit_node(rear_stage)
            .set_position(0.0, -0.05, -2.85)
            .set_scale_xyz(2.40, 0.35, 0.95);

        let animated_orb = scene.add_mesh(Mesh::new(sphere_geometry, orb_material));
        scene
            .edit_node(animated_orb)
            .set_position(0.0, 0.95, 2.10)
            .set_scale_xyz(0.28, 0.28, 0.28);

        // Queue the compressed NPZ point cloud for background loading and place it
        // directly into the same HighFidelity scene as the regular mesh occluders.
        let cloud_handle = engine
            .assets
            .load_gaussian_npz(format!("{}3dgs/point_cloud.npz", ASSET_PATH));
        let cloud_node = scene.add_gaussian_cloud("gaussian_cloud", cloud_handle);
        scene.edit_node(cloud_node).set_rotation_euler(
            std::f32::consts::FRAC_PI_2,
            0.0,
            std::f32::consts::FRAC_PI_2,
        );

        let camera_position = Vec3::new(0.0, 2.4, 6.2);
        let camera_target = Vec3::new(0.0, 0.55, 0.0);
        let camera = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(camera_position.x, camera_position.y, camera_position.z)
            .look_at(camera_target);
        scene.active_camera = Some(camera);

        Self {
            controls: OrbitControls::new(camera_position, camera_target),
            fps_counter: FpsCounter::new(),
            animated_orb,
            rotating_props: [
                RotatingProp {
                    handle: left_pedestal,
                    angular_speed: 0.55,
                },
                RotatingProp {
                    handle: right_pedestal,
                    angular_speed: -0.42,
                },
            ],
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        for prop in &self.rotating_props {
            if let Some(node) = scene.node_mut(prop.handle) {
                node.transform.rotation *= Quat::from_rotation_y(prop.angular_speed * frame.dt);
            }
        }

        let orbit_angle = self.time * 0.9;
        let orbit_height = 0.95 + (self.time * 1.7).sin() * 0.22;
        let orbit_radius = 2.15 + (self.time * 0.6).sin() * 0.18;
        let orb_position = Vec3::new(
            orbit_radius * orbit_angle.cos(),
            orbit_height,
            orbit_radius * orbit_angle.sin(),
        );

        if let Some(node) = scene.node_mut(self.animated_orb) {
            node.transform.position = orb_position;
            node.transform.scale = Vec3::splat(0.26 + 0.04 * (self.time * 3.0).sin().abs());
        }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("3DGS Mesh Occlusion | FPS: {:.2}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Myth Engine — 3DGS Mesh Occlusion")
        .with_settings(RendererSettings {
            path: RenderPath::HighFidelity,
            vsync: false,
            ..Default::default()
        })
        .run::<GaussianSplattingMixedDemo>()
}
