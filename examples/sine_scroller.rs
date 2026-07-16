//! [gallery]
//! name = "Sine Scroller + Copper Bars"
//! category = "Post Effects"
//! description = "Classic demoscene scroller rendered into a dynamic texture with animated copper bars and CRT staging."
//! order = 454
//!

use myth::prelude::*;
use myth_dev_utils::FpsCounter;
use myth_resources::TextureSampler;

const SCREEN_WIDTH: u32 = 512;
const SCREEN_HEIGHT: u32 = 224;
const SCREEN_STEP: f32 = 1.0 / 30.0;
const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;
const GLYPH_ADVANCE: usize = 6;
const SCROLLER_MESSAGE: &str =
    "MYTH ENGINE  SINE SCROLLER  RENDERGRAPH  BLOOM  CUSTOM PASSES  DYNAMIC TEXTURES  ";

fn glyph_rows(ch: char) -> [u8; GLYPH_H] {
    match ch {
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F],
        'J' => [0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        _ => [0x00; GLYPH_H],
    }
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

struct SineScrollerDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    texture: TextureHandle,
    pixels: Vec<u8>,
    accumulator: f32,
    scroll_px: f32,
    phase: f32,
    panel_root: NodeHandle,
}

impl SineScrollerDemo {
    fn clear(&mut self) {
        for pixel in self.pixels.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[6, 10, 18, 255]);
        }
    }

    fn add_pixel(&mut self, x: i32, y: i32, color: [u8; 3]) {
        if x < 0 || y < 0 || x >= SCREEN_WIDTH as i32 || y >= SCREEN_HEIGHT as i32 {
            return;
        }
        let index = (y as usize * SCREEN_WIDTH as usize + x as usize) * 4;
        self.pixels[index] = self.pixels[index].max(color[0]);
        self.pixels[index + 1] = self.pixels[index + 1].max(color[1]);
        self.pixels[index + 2] = self.pixels[index + 2].max(color[2]);
        self.pixels[index + 3] = 255;
    }

    fn draw_glyph(&mut self, x: i32, y: i32, ch: char, color: [u8; 3], shadow: [u8; 3]) {
        let rows = glyph_rows(ch);
        for (py, row_bits) in rows.iter().enumerate() {
            for px in 0..GLYPH_W {
                let mask = 1 << (GLYPH_W - 1 - px);
                if row_bits & mask == 0 {
                    continue;
                }
                self.add_pixel(x + px as i32 + 1, y + py as i32 + 1, shadow);
                self.add_pixel(x + px as i32, y + py as i32, color);
            }
        }
    }

    fn draw_copper_bars(&mut self) {
        for y in 0..SCREEN_HEIGHT as usize {
            let yf = y as f32 / SCREEN_HEIGHT as f32;
            let bar_a = ((yf * 10.0 + self.phase * 0.06).sin() * 0.5 + 0.5).powf(1.4);
            let bar_b = ((yf * 17.0 - self.phase * 0.08).sin() * 0.5 + 0.5).powf(1.5);
            let bar_c = ((yf * 26.0 + self.phase * 0.04).sin() * 0.5 + 0.5).powf(1.8);
            let color = [
                (18.0 + bar_b * 185.0 + bar_c * 34.0).min(255.0) as u8,
                (26.0 + bar_a * 200.0).min(255.0) as u8,
                (42.0 + bar_c * 205.0).min(255.0) as u8,
            ];

            for x in 0..SCREEN_WIDTH as usize {
                let index = (y * SCREEN_WIDTH as usize + x) * 4;
                self.pixels[index] = self.pixels[index].max(color[0]);
                self.pixels[index + 1] = self.pixels[index + 1].max(color[1]);
                self.pixels[index + 2] = self.pixels[index + 2].max(color[2]);
            }
        }
    }

    fn draw_scroller(&mut self) {
        let base_line = 110.0;
        let amplitude = 28.0;
        let total_width = SCROLLER_MESSAGE.chars().count() as f32 * GLYPH_ADVANCE as f32;
        let origin_x = SCREEN_WIDTH as f32 - self.scroll_px;

        for (index, ch) in SCROLLER_MESSAGE.chars().enumerate() {
            if ch == ' ' {
                continue;
            }

            let x = origin_x + index as f32 * GLYPH_ADVANCE as f32;
            if x < -(GLYPH_ADVANCE as f32) || x > SCREEN_WIDTH as f32 + GLYPH_ADVANCE as f32 {
                continue;
            }

            let y = base_line + (self.phase * 0.05 + index as f32 * 0.33).sin() * amplitude;
            let glow = ((self.phase * 0.08 + index as f32 * 0.45).sin() * 0.5 + 0.5).powf(1.4);
            let color = [
                (120.0 + glow * 110.0).min(255.0) as u8,
                (220.0 + glow * 35.0).min(255.0) as u8,
                (80.0 + glow * 145.0).min(255.0) as u8,
            ];
            let shadow = [16, 38, 28];
            self.draw_glyph(x as i32, y as i32, ch, color, shadow);
        }

        if self.scroll_px > total_width + SCREEN_WIDTH as f32 {
            self.scroll_px = 0.0;
        }
    }

    fn draw_scanline_overlay(&mut self) {
        for y in (0..SCREEN_HEIGHT as usize).step_by(2) {
            for x in 0..SCREEN_WIDTH as usize {
                let index = (y * SCREEN_WIDTH as usize + x) * 4;
                self.pixels[index] = (self.pixels[index] as f32 * 0.86) as u8;
                self.pixels[index + 1] = (self.pixels[index + 1] as f32 * 0.86) as u8;
                self.pixels[index + 2] = (self.pixels[index + 2] as f32 * 0.90) as u8;
            }
        }
    }

    fn redraw(&mut self) {
        self.clear();
        self.draw_copper_bars();
        self.draw_scroller();
        self.draw_scanline_overlay();
    }
}

impl AppHandler for SineScrollerDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let plane_geo = engine.assets.geometries.add(Geometry::new_plane(1.0, 1.0));

        let frame_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.14, 0.15, 0.18, 1.0))
                .with_metalness(0.20)
                .with_roughness(0.55),
        );
        let floor_material = engine
            .assets
            .materials
            .add(PhysicalMaterial::new(Vec4::new(0.05, 0.05, 0.07, 1.0)).with_roughness(0.96));
        let tube_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.10, 0.10, 0.18, 1.0))
                .with_emissive(Vec3::new(0.35, 0.85, 1.0), 4.0)
                .with_roughness(0.18),
        );

        let pixels = vec![0; SCREEN_WIDTH as usize * SCREEN_HEIGHT as usize * 4];
        let texture = engine
            .assets
            .create_dynamic_texture(
                "sine-scroller",
                SCREEN_WIDTH,
                SCREEN_HEIGHT,
                pixels.clone(),
                ColorSpace::Srgb,
                false,
            )
            .expect("sine scroller texture should be created with matching RGBA8 bytes");

        if let Some(existing) = engine.assets.textures.get(texture) {
            let mut tuned_texture = Texture::new_2d(existing.name(), existing.image);
            tuned_texture.color_space = existing.color_space;
            tuned_texture.generate_mipmaps = existing.generate_mipmaps;
            tuned_texture.sampler = TextureSampler::NEAREST_CLAMP;
            engine.assets.textures.update(texture, tuned_texture);
        }

        let mut screen_material = Material::new_unlit(Vec4::new(1.0, 1.05, 1.1, 1.0));
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
            .set_scale_xyz(16.0, 0.24, 10.0)
            .set_shadows(false, true);

        let panel_root = scene.create_node_with_name("SineScrollerPanel");
        scene.push_root_node(panel_root);
        scene.edit_node(panel_root).set_position(0.0, 2.0, -0.4);

        let screen = scene.add_mesh_to_parent(Mesh::new(plane_geo, screen_material), panel_root);
        scene
            .edit_node(screen)
            .set_scale_xyz(7.4, 3.2, 1.0)
            .set_cast_shadows(false)
            .set_receive_shadows(false);

        add_box_to_parent(
            scene,
            panel_root,
            box_geo,
            frame_material,
            Vec3::new(0.0, 1.74, -0.10),
            Vec3::new(7.75, 0.16, 0.18),
            true,
            true,
        );
        add_box_to_parent(
            scene,
            panel_root,
            box_geo,
            frame_material,
            Vec3::new(0.0, -1.74, -0.10),
            Vec3::new(7.75, 0.16, 0.18),
            true,
            true,
        );
        add_box_to_parent(
            scene,
            panel_root,
            box_geo,
            frame_material,
            Vec3::new(-3.76, 0.0, -0.10),
            Vec3::new(0.16, 3.45, 0.18),
            true,
            true,
        );
        add_box_to_parent(
            scene,
            panel_root,
            box_geo,
            frame_material,
            Vec3::new(3.76, 0.0, -0.10),
            Vec3::new(0.16, 3.45, 0.18),
            true,
            true,
        );

        for offset in [-2.6, 0.0, 2.6] {
            let tube = scene.add_mesh_to_parent(Mesh::new(box_geo, tube_material), panel_root);
            scene
                .edit_node(tube)
                .set_position(offset, -2.1, 0.0)
                .set_scale_xyz(1.6, 0.10, 0.32)
                .set_cast_shadows(false)
                .set_receive_shadows(false);
        }

        let left_light = scene.add_light(Light::new_point(Vec3::new(0.30, 0.9, 1.0), 1.5, 20.0));
        scene.edit_node(left_light).set_position(-4.0, 2.2, 2.8);
        let right_light = scene.add_light(Light::new_point(Vec3::new(1.0, 0.48, 0.75), 1.2, 20.0));
        scene.edit_node(right_light).set_position(4.0, 1.4, 2.5);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(cam)
            .set_position(0.0, 2.0, 8.2)
            .look_at(Vec3::new(0.0, 2.0, -0.4));
        scene.active_camera = Some(cam);

        let mut demo = Self {
            controls: OrbitControls::new(Vec3::new(0.0, 2.0, 8.2), Vec3::new(0.0, 2.0, -0.4)),
            fps_counter: FpsCounter::new(),
            texture,
            pixels,
            accumulator: 0.0,
            scroll_px: 0.0,
            phase: 0.0,
            panel_root,
        };
        demo.redraw();
        engine
            .assets
            .update_dynamic_texture(demo.texture, &demo.pixels)
            .expect("initial sine scroller upload should match the original buffer size");
        demo
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.accumulator += frame.dt;
        self.scroll_px += frame.dt * 88.0;
        self.phase += frame.dt * 60.0;

        while self.accumulator >= SCREEN_STEP {
            self.redraw();
            engine
                .assets
                .update_dynamic_texture(self.texture, &self.pixels)
                .expect("sine scroller updates should keep the same RGBA8 byte length");
            self.accumulator -= SCREEN_STEP;
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
                (self.phase * 0.01).sin() * 0.03,
                0.0,
            ));

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Sine Scroller + Copper Bars | FPS: {:.1}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Sine Scroller + Copper Bars")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<SineScrollerDemo>()
}
