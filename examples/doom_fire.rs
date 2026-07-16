//! [gallery]
//! name = "Doom Fire"
//! category = "Post Effects"
//! description = "Classic fire effect rendered through a Rust-side dynamic texture update loop."
//! instructions = "Left/Right: change wind\nSpace: toggle turbo burn\nEnter: reset wind"
//! order = 446
//!

use myth::prelude::*;
use myth::resources::Key;
use myth_dev_utils::FpsCounter;
use myth_resources::TextureSampler;

const FIRE_WIDTH: u32 = 160;
const FIRE_HEIGHT: u32 = 96;
const FIRE_WIDTH_USIZE: usize = FIRE_WIDTH as usize;
const FIRE_HEIGHT_USIZE: usize = FIRE_HEIGHT as usize;
const FIRE_STEP: f32 = 1.0 / 30.0;
const FIRE_PALETTE: [[u8; 4]; 37] = [
    [7, 7, 7, 255],
    [31, 7, 7, 255],
    [47, 15, 7, 255],
    [71, 15, 7, 255],
    [87, 23, 7, 255],
    [103, 31, 7, 255],
    [119, 31, 7, 255],
    [143, 39, 7, 255],
    [159, 47, 7, 255],
    [175, 63, 7, 255],
    [191, 71, 7, 255],
    [199, 71, 7, 255],
    [223, 79, 7, 255],
    [223, 87, 7, 255],
    [223, 87, 7, 255],
    [215, 95, 7, 255],
    [215, 95, 7, 255],
    [215, 103, 15, 255],
    [207, 111, 15, 255],
    [207, 119, 15, 255],
    [207, 127, 15, 255],
    [207, 135, 23, 255],
    [199, 135, 23, 255],
    [199, 143, 23, 255],
    [199, 151, 31, 255],
    [191, 159, 31, 255],
    [191, 159, 31, 255],
    [191, 167, 39, 255],
    [191, 167, 39, 255],
    [191, 175, 47, 255],
    [183, 175, 47, 255],
    [183, 183, 47, 255],
    [183, 183, 55, 255],
    [207, 207, 111, 255],
    [223, 223, 159, 255],
    [239, 239, 199, 255],
    [255, 255, 255, 255],
];

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

struct DoomFireDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    screen_root: NodeHandle,
    texture: TextureHandle,
    fire: Vec<u8>,
    rgba: Vec<u8>,
    accumulator: f32,
    phase: f32,
    wind: f32,
    turbo: bool,
}

impl DoomFireDemo {
    fn seed_source_row(&self, buffer: &mut [u8]) {
        let max_heat = (FIRE_PALETTE.len() - 1) as f32;
        let bottom = FIRE_HEIGHT_USIZE - 1;
        let base = if self.turbo { 31.0 } else { 27.0 };
        let span = if self.turbo { 5.0 } else { 7.0 };

        for x in 0..FIRE_WIDTH_USIZE {
            let pulse = ((self.phase * 7.0) + x as f32 * 0.17).sin() * 0.5 + 0.5;
            buffer[(bottom * FIRE_WIDTH_USIZE) + x] = (base + pulse * span).min(max_heat) as u8;
        }
    }

    fn update_rgba(&mut self) {
        for y in 0..FIRE_HEIGHT_USIZE {
            for x in 0..FIRE_WIDTH_USIZE {
                let src_index = y * FIRE_WIDTH_USIZE + x;
                let dst_index = ((FIRE_HEIGHT_USIZE - 1 - y) * FIRE_WIDTH_USIZE + x) * 4;
                let color = FIRE_PALETTE[self.fire[src_index] as usize];
                self.rgba[dst_index..dst_index + 4].copy_from_slice(&color);
            }
        }
    }

    fn step_fire(&mut self) {
        let mut next = vec![0; self.fire.len()];
        self.seed_source_row(&mut next);

        for y in 1..FIRE_HEIGHT_USIZE {
            let rise = y as f32 / FIRE_HEIGHT_USIZE as f32;
            for x in 0..FIRE_WIDTH_USIZE {
                let src_index = y * FIRE_WIDTH_USIZE + x;
                let heat = self.fire[src_index];
                if heat == 0 {
                    continue;
                }

                let noise_seed =
                    (self.phase * 31.0) + x as f32 * 12.9898 + y as f32 * 78.233 + self.wind * 17.0;
                let noise = (noise_seed.sin() * 43_758.547).fract().abs();
                let turbulence = if self.turbo { 2.8 } else { 1.8 };
                let drift = ((noise - 0.5) * (1.0 + rise * turbulence) + self.wind).round() as i32;
                let decay =
                    (1.0 + rise * 1.6 + noise * if self.turbo { 1.2 } else { 2.0 }).floor() as u8;
                let propagated = heat.saturating_sub(decay.min(3));
                if propagated == 0 {
                    continue;
                }

                let dst_x = (x as i32 + drift).clamp(0, FIRE_WIDTH_USIZE as i32 - 1) as usize;
                let dst_y = y - 1;
                let dst_index = dst_y * FIRE_WIDTH_USIZE + dst_x;
                next[dst_index] = next[dst_index].max(propagated);

                if propagated > 2 {
                    let shoulder_x = (dst_x + 1).min(FIRE_WIDTH_USIZE - 1);
                    let shoulder_index = dst_y * FIRE_WIDTH_USIZE + shoulder_x;
                    next[shoulder_index] = next[shoulder_index].max(propagated.saturating_sub(2));
                }
            }
        }

        self.fire = next;
        self.update_rgba();
    }
}

impl AppHandler for DoomFireDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let plane_geo = engine.assets.geometries.add(Geometry::new_plane(4.6, 2.7));
        // let sphere_geo = engine.assets.geometries.add(Geometry::new_sphere(1.0));

        let floor_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.08, 0.09, 0.10, 1.0)).with_roughness(0.95));
        let frame_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.17, 0.18, 0.20, 1.0))
                .with_metalness(0.12)
                .with_roughness(0.68),
        );
        let trim_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.28, 0.19, 0.10, 1.0))
                .with_metalness(0.18)
                .with_roughness(0.46),
        );
        // let helper_material = engine.assets.materials.add(
        //     PhysicalMaterial::new(Vec4::new(1.0, 0.65, 0.25, 1.0))
        //         .with_emissive(Vec3::new(1.0, 0.42, 0.08), 1.6)
        //         .with_roughness(0.2),
        // );

        let rgba = vec![0; FIRE_WIDTH_USIZE * FIRE_HEIGHT_USIZE * 4];
        let texture = engine
            .assets
            .create_dynamic_texture(
                "doom-fire",
                FIRE_WIDTH,
                FIRE_HEIGHT,
                rgba.clone(),
                ColorSpace::Srgb,
                false,
            )
            .expect("dynamic fire texture should be created with matching RGBA8 bytes");

        if let Some(existing) = engine.assets.textures.get(texture) {
            let mut retro_texture = Texture::new_2d(existing.name(), existing.image);
            retro_texture.color_space = existing.color_space;
            retro_texture.generate_mipmaps = existing.generate_mipmaps;
            retro_texture.sampler = TextureSampler::NEAREST_CLAMP;
            engine.assets.textures.update(texture, retro_texture);
        }

        let mut screen_material = Material::new_unlit(Vec4::ONE);
        if let Some(unlit) = screen_material.as_unlit_mut() {
            unlit.set_map(Some(texture));
        }
        let screen_material = engine.assets.materials.add(screen_material);

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.008));

        let floor = scene.add_mesh(Mesh::new(box_geo, floor_material));
        scene
            .edit_node(floor)
            .set_position(0.0, -0.12, 0.0)
            .set_scale_xyz(14.0, 0.24, 10.0)
            .set_shadows(false, true);

        let screen_root = scene.create_node_with_name("DoomFireScreen");
        scene.push_root_node(screen_root);
        scene.edit_node(screen_root).set_position(0.0, 1.75, 0.0);

        let screen = scene.add_mesh_to_parent(Mesh::new(plane_geo, screen_material), screen_root);
        scene
            .edit_node(screen)
            .set_cast_shadows(false)
            .set_receive_shadows(false);

        add_box_to_parent(
            scene,
            screen_root,
            box_geo,
            frame_material,
            Vec3::new(0.0, 0.0, -0.12),
            Vec3::new(4.95, 3.05, 0.08),
            true,
            true,
        );
        add_box_to_parent(
            scene,
            screen_root,
            box_geo,
            trim_material,
            Vec3::new(0.0, 1.43, -0.08),
            Vec3::new(5.05, 0.16, 0.24),
            true,
            true,
        );
        add_box_to_parent(
            scene,
            screen_root,
            box_geo,
            trim_material,
            Vec3::new(0.0, -1.43, -0.08),
            Vec3::new(5.05, 0.16, 0.24),
            true,
            true,
        );
        add_box_to_parent(
            scene,
            screen_root,
            box_geo,
            trim_material,
            Vec3::new(-2.47, 0.0, -0.08),
            Vec3::new(0.16, 3.02, 0.24),
            true,
            true,
        );
        add_box_to_parent(
            scene,
            screen_root,
            box_geo,
            trim_material,
            Vec3::new(2.47, 0.0, -0.08),
            Vec3::new(0.16, 3.02, 0.24),
            true,
            true,
        );
        add_box_to_parent(
            scene,
            screen_root,
            box_geo,
            frame_material,
            Vec3::new(0.0, -1.95, 0.0),
            Vec3::new(1.8, 0.24, 0.9),
            true,
            true,
        );
        add_box_to_parent(
            scene,
            screen_root,
            box_geo,
            frame_material,
            Vec3::new(0.0, -1.72, -0.12),
            Vec3::new(0.32, 0.70, 0.30),
            true,
            true,
        );

        let mut key = Light::new_directional(Vec3::new(0.72, 0.78, 0.95), 1.2);
        key.cast_shadows = true;
        if let Some(shadow) = key.shadow.as_mut() {
            shadow.map_size = 2048;
            shadow.normal_bias = 0.0;
        }
        let key = scene.add_light(key);
        scene
            .edit_node(key)
            .set_position(6.0, 8.0, 5.0)
            .look_at(Vec3::new(0.0, 1.4, 0.0));

        let warm_light = scene.add_light(Light::new_point(Vec3::new(1.0, 0.52, 0.16), 2.6, 18.0));
        scene.edit_node(warm_light).set_position(0.0, 2.1, 1.8);
        // let helper = scene.add_mesh_to_parent(Mesh::new(sphere_geo, helper_material), warm_light);
        // scene
        //     .edit_node(helper)
        //     .set_scale(0.14)
        //     .set_cast_shadows(false)
        //     .set_receive_shadows(false);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(cam)
            .set_position(0.0, 1.8, 6.4)
            .look_at(Vec3::new(0.0, 1.65, 0.0));
        scene.active_camera = Some(cam);

        let mut demo = Self {
            controls: OrbitControls::new(Vec3::new(0.0, 1.8, 6.4), Vec3::new(0.0, 1.65, 0.0)),
            fps_counter: FpsCounter::new(),
            screen_root,
            texture,
            fire: vec![0; FIRE_WIDTH_USIZE * FIRE_HEIGHT_USIZE],
            rgba,
            accumulator: 0.0,
            phase: 0.0,
            wind: 0.0,
            turbo: false,
        };

        for _ in 0..48 {
            demo.step_fire();
        }
        engine
            .assets
            .update_dynamic_texture(demo.texture, &demo.rgba)
            .expect("initial fire texture upload should match the original buffer size");

        demo
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        if engine.input.get_key(Key::ArrowLeft) {
            self.wind = (self.wind - frame.dt * 1.75).max(-2.5);
        }
        if engine.input.get_key(Key::ArrowRight) {
            self.wind = (self.wind + frame.dt * 1.75).min(2.5);
        }
        if engine.input.get_key_down(Key::Space) {
            self.turbo = !self.turbo;
        }
        if engine.input.get_key_down(Key::Enter) {
            self.wind = 0.0;
        }

        self.phase += frame.dt;
        self.accumulator += frame.dt;

        while self.accumulator >= FIRE_STEP {
            self.step_fire();
            engine
                .assets
                .update_dynamic_texture(self.texture, &self.rgba)
                .expect("fire texture updates should keep the same RGBA8 byte length");
            self.accumulator -= FIRE_STEP;
        }

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        scene
            .edit_node(self.screen_root)
            .set_rotation(Quat::from_euler(
                EulerRot::XYZ,
                -0.04,
                0.18 * (self.phase * 0.6).sin(),
                0.0,
            ));

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            let mode = if self.turbo { "turbo" } else { "steady" };
            window.set_title(&format!(
                "Doom Fire | wind {:+.1} | {} | FPS: {:.1}",
                self.wind, mode, fps
            ));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Doom Fire")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<DoomFireDemo>()
}
