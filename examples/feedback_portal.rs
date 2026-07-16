//! [gallery]
//! name = "Feedback Portal"
//! category = "Post Effects"
//! description = "Recursive feedback portal rendered into a dynamic texture and framed by a neon gate."
//! order = 462
//!

use std::f32::consts::TAU;

use myth::prelude::*;
use myth_dev_utils::FpsCounter;
use myth_resources::TextureSampler;

const PORTAL_SIZE: u32 = 320;
const PORTAL_STEP: f32 = 1.0 / 30.0;

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

struct FeedbackPortalDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    texture: TextureHandle,
    current: Vec<u8>,
    previous: Vec<u8>,
    accumulator: f32,
    phase: f32,
    ring_root: NodeHandle,
}

impl FeedbackPortalDemo {
    fn add_pixel(buffer: &mut [u8], x: i32, y: i32, color: [u8; 3]) {
        if x < 0 || y < 0 || x >= PORTAL_SIZE as i32 || y >= PORTAL_SIZE as i32 {
            return;
        }
        let index = (y as usize * PORTAL_SIZE as usize + x as usize) * 4;
        buffer[index] = buffer[index].max(color[0]);
        buffer[index + 1] = buffer[index + 1].max(color[1]);
        buffer[index + 2] = buffer[index + 2].max(color[2]);
        buffer[index + 3] = 255;
    }

    fn stamp_glow(buffer: &mut [u8], center_x: f32, center_y: f32, radius: f32, color: [u8; 3]) {
        let min_x = (center_x - radius).floor() as i32;
        let max_x = (center_x + radius).ceil() as i32;
        let min_y = (center_y - radius).floor() as i32;
        let max_y = (center_y + radius).ceil() as i32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > radius {
                    continue;
                }
                let falloff = 1.0 - dist / radius.max(0.001);
                let pixel = [
                    (color[0] as f32 * falloff) as u8,
                    (color[1] as f32 * falloff) as u8,
                    (color[2] as f32 * falloff) as u8,
                ];
                Self::add_pixel(buffer, x, y, pixel);
            }
        }
    }

    fn step_feedback(&mut self) {
        for pixel in self.current.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[2, 5, 16, 255]);
        }

        let center = PORTAL_SIZE as f32 * 0.5;
        let scale = 0.986 + self.phase.sin() * 0.004;
        let rotation = self.phase * 0.018;
        let (s, c) = rotation.sin_cos();

        for y in 0..PORTAL_SIZE as usize {
            for x in 0..PORTAL_SIZE as usize {
                let nx = (x as f32 - center) / center;
                let ny = (y as f32 - center) / center;
                let radius = (nx * nx + ny * ny).sqrt();
                let src_x = (nx * c - ny * s) / scale;
                let src_y = (nx * s + ny * c) / scale;
                let sample_x = (src_x * center + center + self.phase.sin() * 2.0).round() as i32;
                let sample_y = (src_y * center + center + self.phase.cos() * 2.0).round() as i32;

                if (0..PORTAL_SIZE as i32).contains(&sample_x)
                    && (0..PORTAL_SIZE as i32).contains(&sample_y)
                {
                    let src_index =
                        (sample_y as usize * PORTAL_SIZE as usize + sample_x as usize) * 4;
                    let dst_index = (y * PORTAL_SIZE as usize + x) * 4;
                    self.current[dst_index] = (self.previous[src_index] as f32 * 0.92) as u8;
                    self.current[dst_index + 1] =
                        (self.previous[src_index + 1] as f32 * 0.95) as u8;
                    self.current[dst_index + 2] =
                        (self.previous[src_index + 2] as f32 * 0.97) as u8;
                    self.current[dst_index + 3] = 255;
                }

                let ring = 1.0 - ((radius - (0.36 + self.phase.sin() * 0.04)).abs() * 22.0);
                if ring > 0.0 {
                    let dst_index = (y * PORTAL_SIZE as usize + x) * 4;
                    let glow = ring.min(1.0);
                    self.current[dst_index] = self.current[dst_index].max((80.0 * glow) as u8);
                    self.current[dst_index + 1] =
                        self.current[dst_index + 1].max((180.0 * glow) as u8);
                    self.current[dst_index + 2] =
                        self.current[dst_index + 2].max((255.0 * glow) as u8);
                }
            }
        }

        for orb in 0..4 {
            let angle = self.phase * (0.7 + orb as f32 * 0.17) + orb as f32 / 4.0 * TAU;
            let radius = 66.0 + orb as f32 * 18.0;
            let color = match orb {
                0 => [110, 255, 255],
                1 => [255, 100, 220],
                2 => [255, 180, 72],
                _ => [120, 255, 160],
            };
            Self::stamp_glow(
                &mut self.current,
                center + angle.cos() * radius,
                center + angle.sin() * radius * 0.72,
                18.0 + orb as f32 * 2.0,
                color,
            );
        }

        std::mem::swap(&mut self.current, &mut self.previous);
    }
}

impl AppHandler for FeedbackPortalDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let plane_geo = engine.assets.geometries.add(Geometry::new_plane(1.0, 1.0));

        let frame_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.10, 0.11, 0.16, 1.0))
                .with_metalness(0.22)
                .with_roughness(0.52),
        );
        let floor_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.05, 0.06, 0.08, 1.0)).with_roughness(0.95));
        let neon_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.10, 0.10, 0.16, 1.0))
                .with_emissive(Vec3::new(0.38, 0.92, 1.0), 4.8)
                .with_roughness(0.16),
        );

        let pixels = vec![0; PORTAL_SIZE as usize * PORTAL_SIZE as usize * 4];
        let texture = engine
            .assets
            .create_dynamic_texture(
                "feedback-portal",
                PORTAL_SIZE,
                PORTAL_SIZE,
                pixels.clone(),
                ColorSpace::Srgb,
                false,
            )
            .expect("feedback portal texture should be created with matching RGBA8 bytes");

        if let Some(existing) = engine.assets.textures.get(texture) {
            let mut tuned_texture = Texture::new_2d(existing.name(), existing.image);
            tuned_texture.color_space = existing.color_space;
            tuned_texture.generate_mipmaps = existing.generate_mipmaps;
            tuned_texture.sampler = TextureSampler::LINEAR_CLAMP;
            engine.assets.textures.update(texture, tuned_texture);
        }

        let mut portal_material = Material::new_unlit(Vec4::new(1.0, 1.0, 1.0, 1.0));
        if let Some(unlit) = portal_material.as_unlit_mut() {
            unlit.set_map(Some(texture));
        }
        let portal_material = engine.assets.materials.add(portal_material);

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.008));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.10);
        scene.bloom.set_radius(0.006);

        let floor = scene.add_mesh(Mesh::new(box_geo, floor_material));
        scene
            .edit_node(floor)
            .set_position(0.0, -0.12, 0.0)
            .set_scale_xyz(16.0, 0.24, 12.0)
            .set_shadows(false, true);

        let ring_root = scene.create_node_with_name("FeedbackPortal");
        scene.push_root_node(ring_root);
        scene.edit_node(ring_root).set_position(0.0, 2.1, -0.2);

        let portal = scene.add_mesh_to_parent(Mesh::new(plane_geo, portal_material), ring_root);
        scene
            .edit_node(portal)
            .set_scale_xyz(4.8, 4.8, 1.0)
            .set_cast_shadows(false)
            .set_receive_shadows(false);

        for segment in 0..32 {
            let angle = segment as f32 / 32.0 * TAU;
            let radial = Vec3::new(angle.cos(), angle.sin(), 0.0);
            let segment_handle =
                scene.add_mesh_to_parent(Mesh::new(box_geo, neon_material), ring_root);
            scene
                .edit_node(segment_handle)
                .set_position_vec(radial * 2.9)
                .set_rotation(Quat::from_rotation_z(angle + std::f32::consts::FRAC_PI_2))
                .set_scale_xyz(0.18, 0.72, 0.22)
                .set_cast_shadows(false)
                .set_receive_shadows(false);
        }

        add_box_to_parent(
            scene,
            ring_root,
            box_geo,
            frame_material,
            Vec3::new(0.0, -3.1, 0.0),
            Vec3::new(1.6, 0.22, 1.1),
            true,
            true,
        );
        add_box_to_parent(
            scene,
            ring_root,
            box_geo,
            frame_material,
            Vec3::new(0.0, -2.8, -0.08),
            Vec3::new(0.24, 0.75, 0.24),
            true,
            true,
        );

        let left_light = scene.add_light(Light::new_point(Vec3::new(0.4, 0.95, 1.0), 1.5, 22.0));
        scene.edit_node(left_light).set_position(-3.0, 2.6, 2.4);
        let right_light = scene.add_light(Light::new_point(Vec3::new(1.0, 0.35, 0.9), 1.3, 22.0));
        scene.edit_node(right_light).set_position(3.0, 1.8, 2.8);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(cam)
            .set_position(0.0, 2.1, 8.2)
            .look_at(Vec3::new(0.0, 2.1, -0.2));
        scene.active_camera = Some(cam);

        let mut demo = Self {
            controls: OrbitControls::new(Vec3::new(0.0, 2.1, 8.2), Vec3::new(0.0, 2.1, -0.2)),
            fps_counter: FpsCounter::new(),
            texture,
            current: pixels.clone(),
            previous: pixels,
            accumulator: 0.0,
            phase: 0.0,
            ring_root,
        };

        for _ in 0..18 {
            demo.step_feedback();
        }
        engine
            .assets
            .update_dynamic_texture(demo.texture, &demo.previous)
            .expect("initial feedback portal upload should match the original buffer size");

        demo
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.phase += frame.dt * 2.1;
        self.accumulator += frame.dt;

        while self.accumulator >= PORTAL_STEP {
            self.step_feedback();
            engine
                .assets
                .update_dynamic_texture(self.texture, &self.previous)
                .expect("feedback portal updates should keep the same RGBA8 byte length");
            self.accumulator -= PORTAL_STEP;
        }

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        scene
            .edit_node(self.ring_root)
            .set_rotation(Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, self.phase * 0.16))
            .set_scale(1.0 + self.phase.sin() * 0.03);

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Feedback Portal | FPS: {:.1}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Feedback Portal")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<FeedbackPortalDemo>()
}
