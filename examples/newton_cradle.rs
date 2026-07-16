//! [gallery]
//! name = "Newton Cradle"
//! category = "Post Effects"
//! description = "Classic desk toy built from a node hierarchy, animated swings, and dynamic shadows."
//! order = 460
//!

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

const BALL_COUNT: usize = 5;
const BALL_RADIUS: f32 = 0.34;
const STRING_LENGTH: f32 = 2.0;
const BALL_SPACING: f32 = BALL_RADIUS * 2.0;
const SWING_ANGLE: f32 = 0.74;

fn add_box_to_parent(
    scene: &mut Scene,
    parent: NodeHandle,
    geometry: GeometryHandle,
    material: MaterialHandle,
    position: Vec3,
    scale: Vec3,
    cast_shadows: bool,
    receive_shadows: bool,
) -> NodeHandle {
    let node = scene.add_mesh_to_parent(Mesh::new(geometry, material), parent);
    scene
        .edit_node(node)
        .set_position_vec(position)
        .set_scale_xyz(scale.x, scale.y, scale.z)
        .set_shadows(cast_shadows, receive_shadows);
    node
}

struct NewtonCradleDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    pivots: Vec<NodeHandle>,
    fill_light: NodeHandle,
    time: f32,
}

impl AppHandler for NewtonCradleDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let sphere_geo = engine.assets.geometries.add(Geometry::new_sphere(1.0));

        let table_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.14, 0.15, 0.18, 1.0)).with_roughness(0.92));
        let frame_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.87, 0.88, 0.92, 1.0))
                .with_metalness(0.95)
                .with_roughness(0.14),
        );
        let rail_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.60, 0.63, 0.68, 1.0))
                .with_metalness(0.78)
                .with_roughness(0.28),
        );
        let string_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.18, 0.20, 0.22, 1.0)).with_roughness(0.75));
        let ball_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.94, 0.95, 0.98, 1.0))
                .with_metalness(1.0)
                .with_roughness(0.08),
        );
        let helper_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.72, 0.80, 1.0, 1.0))
                .with_emissive(Vec3::new(0.36, 0.48, 1.0), 1.0)
                .with_metalness(0.1)
                .with_roughness(0.2),
        );

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.018));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.03);
        scene.bloom.set_radius(0.004);

        let table = scene.add_mesh(Mesh::new(box_geo, table_material));
        scene
            .edit_node(table)
            .set_position(0.0, -0.15, 0.0)
            .set_scale_xyz(14.0, 0.30, 10.0)
            .set_shadows(false, true);

        let cradle_root = scene.create_node_with_name("CradleRoot");
        scene.push_root_node(cradle_root);
        scene.edit_node(cradle_root).set_position(0.0, 0.25, 0.0);

        for &x in &[-2.0, 2.0] {
            for &z in &[-0.5, 0.5] {
                add_box_to_parent(
                    scene,
                    cradle_root,
                    box_geo,
                    frame_material,
                    Vec3::new(x, 1.5, z),
                    Vec3::new(0.16, 3.0, 0.16),
                    true,
                    true,
                );
            }

            add_box_to_parent(
                scene,
                cradle_root,
                box_geo,
                rail_material,
                Vec3::new(x, 2.15, 0.0),
                Vec3::new(0.12, 0.12, 1.12),
                true,
                true,
            );
        }

        for &z in &[-0.5, 0.5] {
            add_box_to_parent(
                scene,
                cradle_root,
                box_geo,
                frame_material,
                Vec3::new(0.0, 3.0, z),
                Vec3::new(4.4, 0.12, 0.12),
                true,
                true,
            );

            add_box_to_parent(
                scene,
                cradle_root,
                box_geo,
                rail_material,
                Vec3::new(0.0, 0.08, z),
                Vec3::new(4.0, 0.10, 0.20),
                true,
                true,
            );
        }

        let mut pivots = Vec::with_capacity(BALL_COUNT);
        let span = (BALL_COUNT - 1) as f32 * BALL_SPACING * 0.5;
        for index in 0..BALL_COUNT {
            let x = (index as f32 * BALL_SPACING) - span;

            let pivot = scene.create_node_with_name(&format!("Pivot_{index}"));
            scene.push_root_node(pivot);
            scene.attach(pivot, cradle_root);
            scene.edit_node(pivot).set_position(x, 2.92, 0.0);

            add_box_to_parent(
                scene,
                pivot,
                box_geo,
                string_material,
                Vec3::new(0.0, -(STRING_LENGTH * 0.5), -0.09),
                Vec3::new(0.025, STRING_LENGTH, 0.025),
                false,
                false,
            );
            add_box_to_parent(
                scene,
                pivot,
                box_geo,
                string_material,
                Vec3::new(0.0, -(STRING_LENGTH * 0.5), 0.09),
                Vec3::new(0.025, STRING_LENGTH, 0.025),
                false,
                false,
            );

            let bob = scene.add_mesh_to_parent(Mesh::new(sphere_geo, ball_material), pivot);
            scene
                .edit_node(bob)
                .set_position(0.0, -(STRING_LENGTH + BALL_RADIUS), 0.0)
                .set_scale(BALL_RADIUS)
                .set_shadows(true, true);
            pivots.push(pivot);
        }

        let mut key_light = Light::new_directional(Vec3::new(1.0, 0.97, 0.92), 3.6);
        key_light.cast_shadows = true;
        if let Some(shadow) = key_light.shadow.as_mut() {
            shadow.map_size = 2048;
            shadow.normal_bias = 0.0;
        }
        let key_light = scene.add_light(key_light);
        scene
            .edit_node(key_light)
            .set_position(8.0, 10.0, 6.0)
            .look_at(Vec3::new(0.0, 1.6, 0.0));

        let fill_light = scene.add_light(Light::new_point(Vec3::new(0.62, 0.76, 1.0), 0.8, 18.0));
        scene.edit_node(fill_light).set_position(-3.0, 2.6, 2.0);

        let helper = scene.add_mesh_to_parent(Mesh::new(sphere_geo, helper_material), fill_light);
        scene
            .edit_node(helper)
            .set_scale(0.12)
            .set_cast_shadows(false)
            .set_receive_shadows(false);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(cam)
            .set_position(5.5, 3.6, 8.0)
            .look_at(Vec3::new(0.0, 1.5, 0.0));
        scene.active_camera = Some(cam);

        Self {
            controls: OrbitControls::new(Vec3::new(5.5, 3.6, 8.0), Vec3::new(0.0, 1.5, 0.0)),
            fps_counter: FpsCounter::new(),
            pivots,
            fill_light,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        let transfer = (self.time * 1.45).sin();
        for (index, pivot) in self.pivots.iter().enumerate() {
            let angle = match index {
                // End balls must swing away from the stack, otherwise they arc through the center.
                0 => -SWING_ANGLE * transfer.max(0.0),
                4 => SWING_ANGLE * (-transfer).max(0.0),
                1 | 2 | 3 => 0.0,
                _ => 0.0,
            };

            scene
                .edit_node(*pivot)
                .set_rotation(Quat::from_rotation_z(angle));
        }

        if let Some(node) = scene.node_mut(self.fill_light) {
            node.transform.position = Vec3::new(
                -3.0 + (self.time * 0.8).cos() * 1.6,
                2.7 + (self.time * 1.3).sin() * 0.25,
                1.8 + (self.time * 0.8).sin() * 1.5,
            );
        }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Newton Cradle | FPS: {:.1}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Newton Cradle")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<NewtonCradleDemo>()
}
