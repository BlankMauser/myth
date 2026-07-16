//! [gallery]
//! name = "Custom Material - Radar Sweep"
//! category = "Materials"
//! description = "Procedural radar plane with animated rings, grid lines and a rotating sweep beam."
//! order = 321
//!

//! Custom Material Example — Radar Sweep
//!
//! Demonstrates a custom material focused on planar procedural effects:
//! 1. UV-driven grid generation
//! 2. Time-based rotating sweep beam
//! 3. Transparent blending with animated rings and pulses

use std::f32::consts::FRAC_PI_2;

use myth::prelude::*;
use myth_resources::myth_material;

const RADAR_SHADER: &str = r#"
fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let local_position = vec3<f32>(in.position.xyz);
    let world_pos = u_model.world_matrix * vec4<f32>(local_position, 1.0);

    out.position = u_render_state.view_projection * world_pos;
    out.world_position = world_pos.xyz / world_pos.w;

    $$ if HAS_NORMAL is defined
    let local_normal = normalize(vec3<f32>(in.normal.xyz));
    out.geometry_normal = local_normal;
    out.normal = normalize(u_model.normal_matrix * local_normal);
    $$ endif

    $$ if HAS_UV is defined
    out.uv = in.uv;
    $$ endif

    {$ include 'mixins/uv_vertex' $}
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let time = u_render_state.time * u_material.sweep_speed;
    let centered = in.uv * 2.0 - vec2<f32>(1.0, 1.0);
    let radius = length(centered);
    let radial_fade = 1.0 - smoothstep(0.72, 1.02, radius);

    let sweep_dir = vec2<f32>(cos(time), sin(time));
    let pixel_dir = normalize(centered + vec2<f32>(1e-4, 0.0));
    let sweep = pow(saturate(dot(pixel_dir, sweep_dir)), u_material.sweep_power) * radial_fade;

    let grid_pos = fract(in.uv * u_material.grid_scale);
    let grid_line = min(
        min(grid_pos.x, 1.0 - grid_pos.x),
        min(grid_pos.y, 1.0 - grid_pos.y),
    );
    let grid = (1.0 - smoothstep(0.0, 0.035, grid_line)) * radial_fade;

    let rings = saturate(sin(radius * u_material.ring_density * 18.0 - time * 2.4) * 0.5 + 0.5) * radial_fade;
    let pulse = saturate(sin(time * 1.8 - radius * 14.0) * 0.5 + 0.5) * radial_fade;
    let center_core = 1.0 - smoothstep(0.0, 0.08, radius);

    let color =
        u_material.base_color.rgb * (0.16 + grid * 0.40 + pulse * 0.16) +
        u_material.ring_color.rgb * rings * 0.48 +
        u_material.sweep_color.rgb * sweep * 1.25 +
        u_material.sweep_color.rgb * center_core * 0.85;

    let alpha = saturate(
        u_material.opacity * (grid * 0.30 + rings * 0.25 + pulse * 0.20 + sweep * 0.75 + center_core * 0.80)
    );

    if (alpha < u_material.alpha_test) {
        discard;
    }

    return pack_fragment_output(vec4<f32>(color, alpha));
}
"#;

#[myth_material(shader = "custom_radar_sweep", shader_src = RADAR_SHADER)]
pub struct RadarMaterial {
    #[uniform(default = "Vec4::new(0.06, 0.22, 0.14, 1.0)")]
    pub base_color: Vec4,

    #[uniform(default = "Vec4::new(0.15, 1.10, 0.62, 1.0)")]
    pub sweep_color: Vec4,

    #[uniform(default = "Vec4::new(0.05, 0.90, 0.42, 1.0)")]
    pub ring_color: Vec4,

    #[uniform(default = "0.92")]
    pub opacity: f32,

    #[uniform(default = "0.02")]
    pub alpha_test: f32,

    #[uniform(default = "18.0")]
    pub grid_scale: f32,

    #[uniform(default = "2.0")]
    pub ring_density: f32,

    #[uniform(default = "1.3")]
    pub sweep_speed: f32,

    #[uniform(default = "22.0")]
    pub sweep_power: f32,
}

struct RadarSweepDemo {
    blip_a: NodeHandle,
    blip_b: NodeHandle,
    controls: OrbitControls,
    time: f32,
}

impl AppHandler for RadarSweepDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        let radar_plane = scene.spawn_plane(
            8.0,
            8.0,
            Material::new_custom(
                RadarMaterial::new()
                    .with_grid_scale(20.0)
                    .with_ring_density(2.2)
                    .with_sweep_speed(1.6)
                    .with_sweep_power(26.0)
                    .with_opacity(0.92)
                    .with_alpha_mode(AlphaMode::Blend)
                    .with_depth_write(false)
                    .with_side(Side::Double),
            ),
            &engine.assets,
        );
        scene
            .edit_node(radar_plane)
            .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
            .set_position(0.0, 0.0, 0.0);

        let base = scene.spawn_plane(
            10.0,
            10.0,
            PhongMaterial::new(Vec4::new(0.03, 0.04, 0.05, 1.0)).with_side(Side::Double),
            &engine.assets,
        );
        scene
            .edit_node(base)
            .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
            .set_position(0.0, -0.02, 0.0);

        let key_light = scene.add_light(Light::new_directional(Vec3::new(0.7, 1.0, 0.8), 2.4));
        scene
            .edit_node(key_light)
            .set_position(6.0, 8.0, 5.0)
            .look_at(Vec3::new(0.0, 0.6, 0.0));

        let blip_a = scene.spawn_box(
            0.22,
            0.9,
            0.22,
            PhongMaterial::new(Vec4::new(0.12, 0.95, 0.48, 1.0)),
            &engine.assets,
        );
        scene.edit_node(blip_a).set_position(1.8, 0.45, 0.6);

        let blip_b = scene.spawn_sphere(
            0.28,
            PhongMaterial::new(Vec4::new(0.18, 0.85, 0.55, 1.0)),
            &engine.assets,
        );
        scene.edit_node(blip_b).set_position(-2.1, 0.28, -0.8);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(0.0, 5.4, 6.8)
            .look_at(Vec3::new(0.0, 0.0, 0.0));
        scene.active_camera = Some(camera);

        Self {
            blip_a,
            blip_b,
            controls: OrbitControls::new(Vec3::new(0.0, 5.4, 6.8), Vec3::ZERO),
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        scene.edit_node(self.blip_a).set_position(
            (self.time * 0.9).cos() * 2.2,
            0.45 + (self.time * 4.0).sin().abs() * 0.25,
            (self.time * 0.9).sin() * 1.7,
        );

        scene.edit_node(self.blip_b).set_position(
            (self.time * -0.6).cos() * 2.8,
            0.28,
            (self.time * -0.6).sin() * 2.2,
        );

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new().run::<RadarSweepDemo>()
}
