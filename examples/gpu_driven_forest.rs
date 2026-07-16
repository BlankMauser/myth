//! [gallery]
//! name = "GPU-Driven Forest"
//! category = "Performance"
//! description = "A dense shared-asset forest that makes CPU-direct, adaptive, and GPU-driven scene submission easy to compare."
//! instructions = "Use the panel to switch CPU Off / Auto / GPU Required\nCompare full CSM, no-shadow HighFidelity, and no-shadow BasicForward workloads\nDrag to orbit and scroll to zoom"
//! order = 470
//! web = false
//! note = "Native performance showcase; the scene intentionally creates tens of thousands of rigid opaque render items."
//!
//! Run with `cargo run --release --example gpu_driven_forest`.
//!

use std::any::Any;

use myth::GpuDrivenCapabilities;
use myth::prelude::*;
use myth::renderer::graph::RdgSceneSubmissionFamily;
use myth_dev_utils::FpsCounter;
use myth_egui::{UiPass, UiPassNode, egui};
use winit::event::WindowEvent;

const FOREST_SIDE: usize = 128;
const TREE_COUNT: usize = FOREST_SIDE * FOREST_SIDE;
const FOREST_SPACING: f32 = 1.34;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForestWorkload {
    HighFidelityCsm,
    HighFidelityUnshadowed,
    BasicForwardUnshadowed,
}

impl ForestWorkload {
    const fn label(self) -> &'static str {
        match self {
            Self::HighFidelityCsm => "HighFidelity + 4 CSM",
            Self::HighFidelityUnshadowed => "HighFidelity / no shadows",
            Self::BasicForwardUnshadowed => "BasicForward / no shadows",
        }
    }

    const fn render_path(self) -> RenderPath {
        match self {
            Self::HighFidelityCsm | Self::HighFidelityUnshadowed => RenderPath::HighFidelity,
            Self::BasicForwardUnshadowed => RenderPath::BasicForward,
        }
    }

    const fn shadows_enabled(self) -> bool {
        matches!(self, Self::HighFidelityCsm)
    }
}

struct GpuDrivenForestDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    ui_pass: UiPass,
    mode: GpuDrivenMode,
    workload: ForestWorkload,
    sun: NodeHandle,
}

fn hash_unit(mut value: u32) -> f32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value as f32 / u32::MAX as f32
}

impl AppHandler for GpuDrivenForestDemo {
    fn init(engine: &mut Engine, window: &dyn Window) -> Self {
        let wgpu_ctx = engine
            .renderer
            .wgpu_ctx()
            .expect("renderer must be initialized before the forest demo");
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("GPU-driven forest requires the winit backend");
        let ui_pass = UiPass::new(&wgpu_ctx.device, wgpu_ctx.surface_view_format, winit_window);

        let trunk_geometry = engine
            .assets
            .geometries
            .add(Geometry::new_box(0.18, 1.0, 0.18));
        let canopy_geometry = engine.assets.geometries.add(Geometry::new_cone(0.62, 1.45));
        let ground_geometry = engine.assets.geometries.add(Geometry::new_plane(
            FOREST_SIDE as f32 * FOREST_SPACING + 12.0,
            FOREST_SIDE as f32 * FOREST_SPACING + 12.0,
        ));

        let trunk_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.19, 0.095, 0.045, 1.0))
                .with_roughness(0.92)
                .with_metalness(0.0),
        );
        let foliage_materials = [
            Vec4::new(0.055, 0.25, 0.075, 1.0),
            Vec4::new(0.075, 0.33, 0.095, 1.0),
            Vec4::new(0.11, 0.39, 0.10, 1.0),
            Vec4::new(0.16, 0.44, 0.11, 1.0),
        ]
        .map(|color| {
            engine.assets.materials.add(
                PhysicalMaterial::new(color)
                    .with_roughness(0.86)
                    .with_metalness(0.0),
            )
        });
        let ground_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.065, 0.095, 0.055, 1.0))
                .with_roughness(1.0)
                .with_metalness(0.0),
        );

        let scene = engine.scene_manager.create_active();
        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.28, 0.43, 0.60, 1.0),
            Vec4::new(0.72, 0.82, 0.86, 1.0),
        ));
        scene.environment.set_ambient_light(Vec3::splat(0.055));
        scene.tone_mapping.set_exposure(1.08);

        let ground = scene.add_mesh(Mesh::new(ground_geometry, ground_material));
        scene
            .edit_node(ground)
            .set_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .set_position(0.0, -0.02, 0.0)
            .set_shadows(false, true);

        let half_extent = (FOREST_SIDE as f32 - 1.0) * FOREST_SPACING * 0.5;
        for z in 0..FOREST_SIDE {
            for x in 0..FOREST_SIDE {
                let index = z * FOREST_SIDE + x;
                let seed = u32::try_from(index).expect("forest instance index exceeds u32");
                let jitter_x = (hash_unit(seed.wrapping_mul(4).wrapping_add(1)) - 0.5) * 0.58;
                let jitter_z = (hash_unit(seed.wrapping_mul(4).wrapping_add(2)) - 0.5) * 0.58;
                let scale = 0.72 + hash_unit(seed.wrapping_mul(4).wrapping_add(3)) * 0.62;
                let rotation =
                    hash_unit(seed.wrapping_mul(4).wrapping_add(4)) * std::f32::consts::TAU;
                let world_x = x as f32 * FOREST_SPACING - half_extent + jitter_x;
                let world_z = z as f32 * FOREST_SPACING - half_extent + jitter_z;

                let trunk = scene.add_mesh(Mesh::new(trunk_geometry, trunk_material));
                scene
                    .edit_node(trunk)
                    .set_position(world_x, 0.5 * scale, world_z)
                    .set_scale_xyz(scale, scale, scale)
                    .set_rotation(Quat::from_rotation_y(rotation))
                    .set_shadows(true, true);

                let foliage = scene.add_mesh(Mesh::new(
                    canopy_geometry,
                    foliage_materials[index % foliage_materials.len()],
                ));
                scene
                    .edit_node(foliage)
                    .set_position(world_x, 1.48 * scale, world_z)
                    .set_scale_xyz(scale, scale, scale)
                    .set_rotation(Quat::from_rotation_y(rotation))
                    .set_shadows(true, true);
            }
        }

        let mut sun = Light::new_directional(Vec3::new(1.0, 0.91, 0.76), 2.8);
        sun.cast_shadows = true;
        if let Some(shadow) = sun.shadow.as_mut() {
            shadow.map_size = 2048;
            shadow.cascade_count = 4;
            shadow.max_shadow_distance = 190.0;
            shadow.bias = 0.0008;
            shadow.normal_bias = 0.035;
        }
        let sun = scene.add_light(sun);
        scene
            .edit_node(sun)
            .set_position(-55.0, 82.0, 34.0)
            .look_at(Vec3::ZERO);

        let camera_position = Vec3::new(50.0, 42.0, 72.0);
        let camera_target = Vec3::new(0.0, 4.0, 0.0);
        let camera = scene.add_camera(Camera::new_perspective(48.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera)
            .set_position_vec(camera_position)
            .look_at(camera_target);
        scene.active_camera = Some(camera);

        let mut controls = OrbitControls::new(camera_position, camera_target);
        controls.min_distance = 18.0;
        controls.max_distance = 210.0;

        Self {
            controls,
            fps_counter: FpsCounter::new(),
            ui_pass,
            mode: GpuDrivenMode::Auto,
            workload: ForestWorkload::HighFidelityCsm,
            sun,
        }
    }

    fn on_event(&mut self, _engine: &mut Engine, window: &dyn Window, event: &dyn Any) -> bool {
        let Some(event) = event.downcast_ref::<WindowEvent>() else {
            return false;
        };
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("GPU-driven forest requires the winit backend");

        if self.ui_pass.handle_input(winit_window, event) {
            return true;
        }
        if let WindowEvent::Resized(size) = event {
            self.ui_pass
                .resize(size.width, size.height, window.scale_factor());
        }
        false
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("GPU-driven forest requires the winit backend");
        self.ui_pass.begin_frame(winit_window);

        let diagnostics = engine.renderer.latest_rdg_diagnostics();
        let capabilities = engine.renderer.gpu_driven_capabilities();
        let gpu_supported =
            capabilities.is_some_and(GpuDrivenCapabilities::supports_indirect_primitives);
        let tier = capabilities.map(GpuDrivenCapabilities::tier);
        let previous_mode = self.mode;
        let previous_workload = self.workload;
        let context = self.ui_pass.context().clone();
        egui::Window::new("GPU-Driven Forest")
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(18.0, 18.0))
            .resizable(false)
            .show(&context, |ui| {
                ui.heading(format!(
                    "{TREE_COUNT} trees / {} render items",
                    TREE_COUNT * 2 + 1
                ));
                ui.label(format!(
                    "Window throughput: {:.1} FPS",
                    self.fps_counter.current_fps
                ));
                ui.label(format!("GPU-driven capability tier: {tier:?}"));
                ui.separator();
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.mode, GpuDrivenMode::Off, "CPU Off");
                    ui.selectable_value(&mut self.mode, GpuDrivenMode::Auto, "Auto");
                    ui.add_enabled_ui(gpu_supported, |ui| {
                        ui.selectable_value(
                            &mut self.mode,
                            GpuDrivenMode::Required,
                            "GPU Required",
                        );
                    });
                });
                if !gpu_supported {
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "This adapter cannot execute the GPU-driven baseline.",
                    );
                }
                ui.separator();
                ui.label("Rendering workload");
                ui.selectable_value(
                    &mut self.workload,
                    ForestWorkload::HighFidelityCsm,
                    ForestWorkload::HighFidelityCsm.label(),
                );
                ui.selectable_value(
                    &mut self.workload,
                    ForestWorkload::HighFidelityUnshadowed,
                    ForestWorkload::HighFidelityUnshadowed.label(),
                );
                ui.selectable_value(
                    &mut self.workload,
                    ForestWorkload::BasicForwardUnshadowed,
                    ForestWorkload::BasicForwardUnshadowed.label(),
                );

                if let Some(diagnostics) = diagnostics {
                    let policy = diagnostics.gpu_driven_policy();
                    ui.separator();
                    ui.label(format!(
                        "Retained: {} instances / {} bins / {} dirty",
                        policy.retained_candidate_count(),
                        policy.retained_bin_count(),
                        policy.dirty_candidate_count(),
                    ));
                    for family in [
                        RdgSceneSubmissionFamily::MainPrepass,
                        RdgSceneSubmissionFamily::Shadow2D,
                    ] {
                        let state = diagnostics.scene_submission_family(family);
                        ui.monospace(format!(
                            "{family:?}: {:?} ({:?}) | live {:?}",
                            state.selected_backend(),
                            state.selection_reason(),
                            state.live_backend(),
                        ));
                    }
                }
                ui.separator();
                ui.label("Off and Required are deterministic A/B endpoints.");
                ui.label("Auto changes only after submitted-frame hysteresis.");
                ui.label("Wait for the FPS window to settle after changing workload.");
            });
        self.ui_pass.end_frame(winit_window);

        if self.mode != previous_mode || self.workload != previous_workload {
            let mut settings = engine.renderer.settings().clone();
            settings.gpu_driven = self.mode;
            settings.path = self.workload.render_path();
            engine.renderer.update_settings(settings);
            self.fps_counter = FpsCounter::new();
        }

        if self.workload != previous_workload
            && let Some(scene) = engine.scene_manager.active_scene_mut()
            && let Some(sun) = scene.light_mut(self.sun)
        {
            // Keep the complete four-cascade configuration resident and
            // toggle only whether it contributes shadow views. This makes
            // profile changes cheap and preserves the exact same scene,
            // light, and asset identities across A/B measurements.
            sun.cast_shadows = self.workload.shadows_enabled();
        }

        if let Some(scene) = engine.scene_manager.active_scene_mut() {
            let _ = scene.edit_main_camera_bundle(|transform, camera| {
                self.controls
                    .update(transform, &engine.input, camera.fov(), frame.dt);
            });
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "GPU-Driven Forest | {:?} | {} | {} items | FPS: {fps:.1}",
                self.mode,
                self.workload.label(),
                TREE_COUNT * 2 + 1,
            ));
        }
    }

    fn render(&mut self, engine: &mut Engine, _window: &dyn Window) {
        use myth::renderer::graph::core::{GraphBlackboard, HookStage};

        let Some(composer) = engine.compose_frame() else {
            return;
        };
        self.ui_pass
            .resolve_textures(composer.device(), composer.resource_manager());

        let ui_pass = &mut self.ui_pass;
        composer
            .add_custom_pass(HookStage::AfterPostProcess, move |rdg, blackboard| {
                let surface_out = rdg.add_pass("GpuDrivenForestUI", |builder| {
                    let target_tex =
                        builder.mutate_texture(blackboard.surface_out, "Surface_With_Forest_UI");
                    (
                        UiPassNode {
                            pass: ui_pass,
                            target_tex,
                        },
                        target_tex,
                    )
                });
                GraphBlackboard {
                    surface_out,
                    ..blackboard
                }
            })
            .render();
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("GPU-Driven Forest")
        .with_settings(RendererSettings {
            path: RenderPath::HighFidelity,
            vsync: false,
            gpu_driven: GpuDrivenMode::Auto,
            ..Default::default()
        })
        .run::<GpuDrivenForestDemo>()
}
