//! [gallery]
//! name = "Cluster Heatmap"
//! category = "Lighting & GI"
//! description = "A standalone clustered-lighting heatmap scene that visualises per-cluster light density across a deep multi-light corridor."
//! instructions = "1008 Point Lights\nPress H to toggle heatmap/final image\nInspect near/far cluster density bands"
//! features = ["debug_view"]
//! order = 460
//!

#[cfg(feature = "debug_view")]
mod app {
    use std::f32::consts::{FRAC_PI_2, TAU};

    use myth::prelude::*;
    use myth::render::ClusteredShadingMode;
    use myth::resources::input::Key;
    use myth_dev_utils::FpsCounter;

    const LIGHT_GRID_X: usize = 7;
    const LIGHT_GRID_Y: usize = 8;
    const LIGHT_GRID_Z: usize = 18;

    struct HeatmapLight {
        node: NodeHandle,
        base: Vec3,
        phase: f32,
        amplitude: Vec3,
    }

    struct ClusterHeatmapDemo {
        controls: OrbitControls,
        fps_counter: FpsCounter,
        lights: Vec<HeatmapLight>,
        show_heatmap: bool,
        time: f32,
    }

    fn centered_lattice(index: usize, count: usize, spacing: f32) -> f32 {
        (index as f32 - (count as f32 - 1.0) * 0.5) * spacing
    }

    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
        let h6 = (h.fract() * 6.0).clamp(0.0, 6.0 - f32::EPSILON);
        let i = h6.floor() as i32;
        let f = h6 - i as f32;
        let p = v * (1.0 - s);
        let q = v * (1.0 - f * s);
        let t = v * (1.0 - (1.0 - f) * s);

        match i {
            0 => Vec3::new(v, t, p),
            1 => Vec3::new(q, v, p),
            2 => Vec3::new(p, v, t),
            3 => Vec3::new(p, q, v),
            4 => Vec3::new(t, p, v),
            _ => Vec3::new(v, p, q),
        }
    }

    impl AppHandler for ClusterHeatmapDemo {
        fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
            let scene = engine.scene_manager.create_active();
            scene.environment.set_ambient_light(Vec3::splat(0.003));
            scene.background.set_mode(BackgroundMode::gradient(
                Vec4::new(0.02, 0.025, 0.04, 1.0),
                Vec4::new(0.003, 0.004, 0.008, 1.0),
            ));

            scene.bloom.set_enabled(true);
            scene.bloom.set_strength(0.045);

            let floor_material = engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.06, 0.07, 0.08, 1.0))
                    .with_roughness(0.86)
                    .with_metalness(0.12),
            );
            let wall_material = engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.10, 0.11, 0.14, 1.0))
                    .with_roughness(0.74)
                    .with_metalness(0.14),
            );
            let block_material = engine.assets.materials.add(
                PhysicalMaterial::new(Vec4::new(0.78, 0.80, 0.85, 1.0))
                    .with_roughness(0.22)
                    .with_metalness(0.65),
            );

            let floor = scene.spawn_plane(18.0, 84.0, floor_material, &engine.assets);
            scene
                .edit_node(floor)
                .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
                .set_position(0.0, -0.3, 0.0)
                .set_receive_shadows(false);

            for &(x, y, z, sx, sy, sz) in &[
                (-7.5, 2.6, 0.0, 0.45, 5.8, 68.0),
                (7.5, 2.6, 0.0, 0.45, 5.8, 68.0),
                (0.0, 5.2, 0.0, 15.4, 0.22, 68.0),
            ] {
                let wall = scene.spawn_box(sx, sy, sz, wall_material, &engine.assets);
                scene
                    .edit_node(wall)
                    .set_position(x, y, z)
                    .set_shadows(false, true);
            }

            for row in 0..11 {
                let z = centered_lattice(row, 11, 6.1);
                for col in -2..=2 {
                    let x = col as f32 * 2.45;
                    let y = if (row + (col + 2) as usize) % 2 == 0 {
                        0.7
                    } else {
                        1.65
                    };
                    let block = if (row + (col + 2) as usize) % 3 == 0 {
                        scene.spawn_sphere(0.76, block_material, &engine.assets)
                    } else {
                        scene.spawn_box(1.3, 1.3, 1.3, block_material, &engine.assets)
                    };
                    scene
                        .edit_node(block)
                        .set_position(x, y, z)
                        .set_shadows(false, true);
                }
            }

            let mut lights = Vec::with_capacity(LIGHT_GRID_X * LIGHT_GRID_Y * LIGHT_GRID_Z);
            for ix in 0..LIGHT_GRID_X {
                for iy in 0..LIGHT_GRID_Y {
                    for iz in 0..LIGHT_GRID_Z {
                        let light_index = ix * LIGHT_GRID_Y * LIGHT_GRID_Z + iy * LIGHT_GRID_Z + iz;
                        let hue = light_index as f32
                            / (LIGHT_GRID_X * LIGHT_GRID_Y * LIGHT_GRID_Z) as f32;
                        let color = hsv_to_rgb(hue, 0.76, 1.0);
                        let light = scene.add_light(Light::new_point(
                            color,
                            0.3 + iy as f32 * 0.22,
                            3.2 + ix as f32 * 0.22,
                        ));
                        let helper = scene.spawn_sphere(
                            0.04,
                            PhysicalMaterial::new((color * 0.22).extend(1.0))
                                .with_emissive(color, 10.0)
                                .with_roughness(0.22)
                                .with_metalness(0.0),
                            &engine.assets,
                        );
                        scene.attach(helper, light);
                        scene
                            .edit_node(helper)
                            .set_position(0.0, 0.0, 0.0)
                            .set_shadows(false, false);
                        let base = Vec3::new(
                            centered_lattice(ix, LIGHT_GRID_X, 2.45),
                            0.85 + iy as f32 * 0.5,
                            centered_lattice(iz, LIGHT_GRID_Z, 3.25),
                        );
                        scene.edit_node(light).set_position(base.x, base.y, base.z);
                        lights.push(HeatmapLight {
                            node: light,
                            base,
                            phase: (light_index as f32 / 13.0) * TAU,
                            amplitude: Vec3::new(0.24, 0.18, 0.22),
                        });
                    }
                }
            }

            let camera_pos = Vec3::new(0.0, 4.2, 20.0);
            let target = Vec3::new(0.0, 1.8, 0.0);
            let cam = scene.add_camera(Camera::new_perspective(46.0, 16.0 / 9.0, 0.1));
            if let Some(camera) = scene.camera_mut(cam) {
                camera.debug_view.mode = DebugViewMode::ClusterHeatmap;
            }
            scene
                .edit_node(cam)
                .set_position(camera_pos.x, camera_pos.y, camera_pos.z)
                .look_at(target);
            scene.active_camera = Some(cam);

            let mut controls = OrbitControls::new(camera_pos, target);
            controls.min_distance = 9.0;
            controls.max_distance = 34.0;

            Self {
                controls,
                fps_counter: FpsCounter::new(),
                lights,
                show_heatmap: true,
                time: 0.0,
            }
        }

        fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
            let Some(scene) = engine.scene_manager.active_scene_mut() else {
                return;
            };

            self.time += frame.dt;

            if engine.input.get_key_down(Key::H) {
                self.show_heatmap = !self.show_heatmap;
                if let Some(camera_node) = scene.active_camera {
                    if let Some(camera) = scene.camera_mut(camera_node) {
                        camera.debug_view.mode = if self.show_heatmap {
                            DebugViewMode::ClusterHeatmap
                        } else {
                            DebugViewMode::None
                        };
                    }
                }
            }

            for light in &self.lights {
                let offset_x = (self.time * 0.72 + light.phase).sin() * light.amplitude.x;
                let offset_y = (self.time * 1.18 + light.phase * 0.45).sin() * light.amplitude.y;
                let offset_z = (self.time * 0.94 + light.phase * 1.3).cos() * light.amplitude.z;
                scene.edit_node(light.node).set_position(
                    light.base.x + offset_x,
                    light.base.y + offset_y,
                    light.base.z + offset_z,
                );
            }

            let _ = scene.edit_main_camera_bundle(|transform, camera| {
                self.controls
                    .update(transform, &engine.input, camera.fov(), frame.dt);
            });

            if let Some(fps) = self.fps_counter.update() {
                let mode_label = if self.show_heatmap {
                    "Heatmap"
                } else {
                    "Final"
                };
                window.set_title(&format!(
                    "Cluster Heatmap ({}) | {} point lights | FPS: {:.2}",
                    mode_label,
                    self.lights.len(),
                    fps
                ));
            }
        }
    }

    pub fn run() -> myth::Result<()> {
        App::new()
            .with_settings(RendererSettings {
                clustered_shading: ClusteredShadingMode::ForceOn,
                vsync: false,
                ..Default::default()
            })
            .run::<ClusterHeatmapDemo>()
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    #[cfg(not(feature = "debug_view"))]
    {
        eprintln!("clustered_heatmap requires --features debug_view");
        Ok(())
    }

    #[cfg(feature = "debug_view")]
    {
        app::run()
    }
}
