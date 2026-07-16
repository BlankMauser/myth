//! [gallery]
//! name = "Digital Rain"
//! category = "Post Effects"
//! description = "Classic digital rain wall rendered through a dynamic texture, neon screens, and a dark cyber stage."
//! instructions = "Space: glitch burst"
//! order = 448
//!

use myth::prelude::*;
use myth::resources::Key;
use myth_dev_utils::FpsCounter;
use myth_resources::TextureSampler;

const TEX_WIDTH: u32 = 256;
const TEX_HEIGHT: u32 = 384;
const CELL_WIDTH: usize = 8;
const CELL_HEIGHT: usize = 12;
const GRID_WIDTH: usize = TEX_WIDTH as usize / CELL_WIDTH;
const GRID_HEIGHT: usize = TEX_HEIGHT as usize / CELL_HEIGHT;
const RAIN_STEP: f32 = 1.0 / 24.0;

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

fn mix_bits(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
}

fn glyph_pixel(bits: u32, px: usize, py: usize) -> bool {
    let top = bits & 1 != 0 && py >= 1 && py <= 2 && (1..=5).contains(&px);
    let upper_left = bits & 2 != 0 && px <= 2 && (2..=5).contains(&py);
    let upper_right = bits & 4 != 0 && px >= 5 && (2..=5).contains(&py);
    let middle = bits & 8 != 0 && (4..=5).contains(&py) && (1..=5).contains(&px);
    let lower_left = bits & 16 != 0 && px <= 2 && (6..=9).contains(&py);
    let lower_right = bits & 32 != 0 && px >= 5 && (6..=9).contains(&py);
    let bottom = bits & 64 != 0 && py >= 9 && py <= 10 && (1..=5).contains(&px);
    let spine = bits & 128 != 0 && (3..=4).contains(&px) && (2..=9).contains(&py);
    let cap = bits & 256 != 0 && py == 3 && (2..=4).contains(&px);
    let dot = bits & 512 != 0 && (3..=4).contains(&px) && py == 10;
    top || upper_left
        || upper_right
        || middle
        || lower_left
        || lower_right
        || bottom
        || spine
        || cap
        || dot
}

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

struct RainColumn {
    head: f32,
    speed: f32,
    trail: usize,
    seed: u32,
}

struct DigitalRainDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    texture: TextureHandle,
    pixels: Vec<u8>,
    columns: Vec<RainColumn>,
    accumulator: f32,
    phase: u32,
    rng: u32,
    panel_root: NodeHandle,
}

impl DigitalRainDemo {
    fn reset_column(column: &mut RainColumn, rng: &mut u32, column_index: usize) {
        column.head = -rand_range(rng, 0.0, GRID_HEIGHT as f32 * 1.5);
        column.speed = rand_range(rng, 0.35, 1.1);
        column.trail = rand_range(rng, 8.0, 22.0) as usize;
        column.seed = next_rand(rng) ^ (column_index as u32).wrapping_mul(0x9E37_79B9);
    }

    fn clear_pixels(&mut self) {
        for pixel in self.pixels.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[1, 7, 3, 255]);
        }
    }

    fn draw_cell(&mut self, column: usize, row: usize, bits: u32, color: [u8; 4]) {
        let base_x = column * CELL_WIDTH;
        let base_y = (GRID_HEIGHT - 1 - row) * CELL_HEIGHT;

        for py in 0..CELL_HEIGHT {
            for px in 0..CELL_WIDTH {
                if !glyph_pixel(bits, px, py) {
                    continue;
                }

                let x = base_x + px;
                let y = base_y + py;
                let index = (y * TEX_WIDTH as usize + x) * 4;
                self.pixels[index] = self.pixels[index].max(color[0]);
                self.pixels[index + 1] = self.pixels[index + 1].max(color[1]);
                self.pixels[index + 2] = self.pixels[index + 2].max(color[2]);
                self.pixels[index + 3] = 255;
            }
        }
    }

    fn trail_color(intensity: f32, head: bool) -> [u8; 4] {
        if head {
            let glow = (215.0 + intensity * 40.0).min(255.0) as u8;
            [190, 255, glow, 255]
        } else {
            let green = (35.0 + intensity * 185.0).min(255.0) as u8;
            let red = (green as f32 * 0.18) as u8;
            let blue = (green as f32 * 0.34) as u8;
            [red, green, blue, 255]
        }
    }

    fn step_rain(&mut self, glitch: bool) {
        self.phase = self.phase.wrapping_add(1);
        self.clear_pixels();

        for column_index in 0..self.columns.len() {
            let (head, trail, seed) = {
                let column = &mut self.columns[column_index];
                column.head += column.speed;
                if column.head - column.trail as f32 > GRID_HEIGHT as f32 + 2.0 {
                    Self::reset_column(column, &mut self.rng, column_index);
                }
                (column.head, column.trail.max(1), column.seed)
            };

            let head_row = head.floor() as i32;
            for trail_index in 0..=trail {
                let row = head_row - trail_index as i32;
                if !(0..GRID_HEIGHT as i32).contains(&row) {
                    continue;
                }

                let fade = 1.0 - trail_index as f32 / trail as f32;
                let intensity = fade.powf(1.35);
                let bits = mix_bits(
                    seed ^ (row as u32).wrapping_mul(0x45D9_F3B)
                        ^ self.phase.wrapping_mul(0x27D4_EB2D),
                ) | 0b1001;
                let color = Self::trail_color(intensity, trail_index == 0);
                self.draw_cell(column_index, row as usize, bits, color);
            }
        }

        if glitch {
            for _ in 0..18 {
                let column = (next_rand(&mut self.rng) as usize) % GRID_WIDTH;
                let row = (next_rand(&mut self.rng) as usize) % GRID_HEIGHT;
                let bits = mix_bits(next_rand(&mut self.rng)) | 0x1FF;
                self.draw_cell(column, row, bits, [220, 255, 230, 255]);
            }
        }
    }
}

impl AppHandler for DigitalRainDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let plane_geo = engine.assets.geometries.add(Geometry::new_plane(1.0, 1.0));

        let floor_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.06, 0.07, 0.08, 1.0)).with_roughness(0.95));
        let frame_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.11, 0.13, 0.16, 1.0))
                .with_metalness(0.28)
                .with_roughness(0.55),
        );
        let neon_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.08, 0.14, 0.10, 1.0))
                .with_emissive(Vec3::new(0.18, 1.0, 0.48), 3.8)
                .with_roughness(0.18),
        );

        let pixels = vec![0; TEX_WIDTH as usize * TEX_HEIGHT as usize * 4];
        let texture = engine
            .assets
            .create_dynamic_texture(
                "digital-rain",
                TEX_WIDTH,
                TEX_HEIGHT,
                pixels.clone(),
                ColorSpace::Srgb,
                false,
            )
            .expect("digital rain texture should be created with matching RGBA8 bytes");

        if let Some(existing) = engine.assets.textures.get(texture) {
            let mut retro_texture = Texture::new_2d(existing.name(), existing.image);
            retro_texture.color_space = existing.color_space;
            retro_texture.generate_mipmaps = existing.generate_mipmaps;
            retro_texture.sampler = TextureSampler::NEAREST_CLAMP;
            engine.assets.textures.update(texture, retro_texture);
        }

        let mut screen_material = Material::new_unlit(Vec4::new(0.75, 1.6, 0.82, 1.0));
        if let Some(unlit) = screen_material.as_unlit_mut() {
            unlit.set_map(Some(texture));
        }
        let screen_material = engine.assets.materials.add(screen_material);

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.01));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.08);
        scene.bloom.set_radius(0.005);

        let floor = scene.add_mesh(Mesh::new(box_geo, floor_material));
        scene
            .edit_node(floor)
            .set_position(0.0, -0.12, 0.0)
            .set_scale_xyz(14.0, 0.24, 12.0)
            .set_shadows(false, true);

        let panel_root = scene.create_node_with_name("DigitalRainPanels");
        scene.push_root_node(panel_root);
        scene.edit_node(panel_root).set_position(0.0, 2.1, -0.5);

        for &(x, z, yaw, width, height) in &[
            (0.0, 0.0, 0.0, 3.0, 4.8),
            (-3.0, -0.8, 0.42, 2.4, 4.0),
            (3.0, -0.8, -0.42, 2.4, 4.0),
        ] {
            let panel = scene.add_mesh_to_parent(Mesh::new(plane_geo, screen_material), panel_root);
            scene
                .edit_node(panel)
                .set_position(x, 0.0, z)
                .set_rotation(Quat::from_rotation_y(yaw))
                .set_scale_xyz(width, height, 1.0)
                .set_cast_shadows(false)
                .set_receive_shadows(false);

            add_box_to_parent(
                scene,
                panel_root,
                box_geo,
                frame_material,
                Vec3::new(x, height * 0.5 + 0.12, z - 0.06),
                Vec3::new(width + 0.24, 0.12, 0.16),
                true,
                true,
            );
            add_box_to_parent(
                scene,
                panel_root,
                box_geo,
                frame_material,
                Vec3::new(x, -(height * 0.5) - 0.12, z - 0.06),
                Vec3::new(width + 0.24, 0.12, 0.16),
                true,
                true,
            );
        }

        for &(x, z) in &[(-4.5, 1.6), (4.5, 1.6), (-4.5, -2.4), (4.5, -2.4)] {
            let pillar = scene.spawn_box(0.35, 4.6, 0.35, frame_material, &engine.assets);
            scene
                .edit_node(pillar)
                .set_position(x, 2.3, z)
                .set_shadows(true, true);

            let band = scene.spawn_box(0.55, 0.12, 0.55, neon_material, &engine.assets);
            scene
                .edit_node(band)
                .set_position(x, 1.4, z)
                .set_cast_shadows(false)
                .set_receive_shadows(false);
        }

        let left_light = scene.add_light(Light::new_point(Vec3::new(0.18, 1.0, 0.45), 1.8, 18.0));
        scene.edit_node(left_light).set_position(-2.8, 2.2, 2.5);
        let right_light = scene.add_light(Light::new_point(Vec3::new(0.18, 1.0, 0.45), 1.8, 18.0));
        scene.edit_node(right_light).set_position(2.8, 2.2, 2.5);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(cam)
            .set_position(0.0, 2.1, 8.2)
            .look_at(Vec3::new(0.0, 2.0, -0.2));
        scene.active_camera = Some(cam);

        let mut rng = 0xD161_A17E;
        let mut columns = Vec::with_capacity(GRID_WIDTH);
        for column_index in 0..GRID_WIDTH {
            let mut column = RainColumn {
                head: 0.0,
                speed: 0.0,
                trail: 0,
                seed: 0,
            };
            DigitalRainDemo::reset_column(&mut column, &mut rng, column_index);
            columns.push(column);
        }

        let mut demo = Self {
            controls: OrbitControls::new(Vec3::new(0.0, 2.1, 8.2), Vec3::new(0.0, 2.0, -0.2)),
            fps_counter: FpsCounter::new(),
            texture,
            pixels,
            columns,
            accumulator: 0.0,
            phase: 0,
            rng,
            panel_root,
        };

        for _ in 0..12 {
            demo.step_rain(false);
        }
        engine
            .assets
            .update_dynamic_texture(demo.texture, &demo.pixels)
            .expect("initial digital rain upload should match the original buffer size");

        demo
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let glitch = engine.input.get_key(Key::Space);
        self.accumulator += frame.dt;

        while self.accumulator >= RAIN_STEP {
            self.step_rain(glitch);
            engine
                .assets
                .update_dynamic_texture(self.texture, &self.pixels)
                .expect("digital rain texture updates should keep the same RGBA8 byte length");
            self.accumulator -= RAIN_STEP;
        }

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        scene
            .edit_node(self.panel_root)
            .set_rotation(Quat::from_euler(
                EulerRot::XYZ,
                0.0,
                (self.phase as f32 * 0.01).sin() * 0.04,
                0.0,
            ));

        if let Some(fps) = self.fps_counter.update() {
            let mode = if glitch { "GLITCH" } else { "Flow" };
            window.set_title(&format!("Digital Rain | {} | FPS: {:.1}", mode, fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Digital Rain")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<DigitalRainDemo>()
}
