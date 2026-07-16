//! [gallery]
//! name = "Custom Material - Dissolve"
//! category = "Materials"
//! description = "Animated dissolve material with procedural noise, glowing edges and vertex breakup."
//! order = 322
//!

//! Custom Material Example — Dissolve Energy Shell
//!
//! Demonstrates a custom material with:
//! 1. Procedural 3D value noise
//! 2. Animated dissolve thresholds
//! 3. Edge glow and soft transparency
//! 4. Vertex displacement linked to the dissolve phase

use std::f32::consts::FRAC_PI_2;

use myth::prelude::*;
use myth_resources::myth_material;

const DISSOLVE_SHADER: &str = r#"
fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn hash13(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = hash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);

    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);

    return mix(nxy0, nxy1, u.z);
}

fn sample_dissolve_noise(position: vec3<f32>, scale: f32) -> f32 {
    let p = position * scale;
    let base = value_noise(p);
    let detail = value_noise(p * 2.3 + vec3<f32>(11.3, 3.7, 5.1));
    return base * 0.72 + detail * 0.28;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var local_position = vec3<f32>(in.position.xyz);
    var local_normal = vec3<f32>(0.0, 1.0, 0.0);

    $$ if HAS_NORMAL is defined
    local_normal = normalize(vec3<f32>(in.normal.xyz));
    $$ endif

    let time = u_render_state.time * u_material.dissolve_speed + u_material.phase_offset;
    let threshold = 0.5 + 0.5 * sin(time);
    let breakup = sample_dissolve_noise(local_position + local_normal * 0.6, u_material.noise_scale);
    let offset = (1.0 - threshold) * breakup * u_material.displacement;

    $$ if HAS_NORMAL is defined
    local_position += local_normal * offset;
    $$ else
    local_position.y += offset;
    $$ endif

    let world_pos = u_model.world_matrix * vec4<f32>(local_position, 1.0);
    out.position = u_render_state.view_projection * world_pos;
    out.world_position = world_pos.xyz / world_pos.w;

    $$ if HAS_NORMAL is defined
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
    let time = u_render_state.time * u_material.dissolve_speed + u_material.phase_offset;
    let threshold = 0.5 + 0.5 * sin(time);

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    $$ if HAS_NORMAL is defined
    normal = normalize(in.normal);
    $$ endif

    let noise = sample_dissolve_noise(in.world_position + normal * 0.4, u_material.noise_scale);
    let body = smoothstep(threshold - u_material.dissolve_width, threshold + u_material.dissolve_width, noise);
    let edge = 1.0 - smoothstep(0.0, u_material.dissolve_width * 1.25, abs(noise - threshold));
    let sparkle = value_noise(in.world_position * (u_material.noise_scale * 3.2) + vec3<f32>(time * 6.0, 0.0, 0.0));

    let view = normalize(u_render_state.camera_position - in.world_position);
    let fresnel = pow(1.0 - saturate(dot(normal, view)), 3.5);

    let color =
        u_material.base_color.rgb * (0.18 + body * 0.82) +
        u_material.edge_color.rgb * (edge * 1.35 + sparkle * 0.12) +
        u_material.edge_color.rgb * fresnel * 0.35;

    let alpha = max(body * u_material.opacity, edge * 0.85);

    if (alpha < u_material.alpha_test) {
        discard;
    }

    return pack_fragment_output(vec4<f32>(color, alpha));
}
"#;

#[myth_material(shader = "custom_dissolve_energy", shader_src = DISSOLVE_SHADER)]
pub struct DissolveMaterial {
    #[uniform(default = "Vec4::new(0.16, 0.42, 0.98, 1.0)")]
    pub base_color: Vec4,

    #[uniform(default = "Vec4::new(1.20, 0.65, 0.20, 1.0)")]
    pub edge_color: Vec4,

    #[uniform(default = "0.92")]
    pub opacity: f32,

    #[uniform(default = "0.03")]
    pub alpha_test: f32,

    #[uniform(default = "2.6")]
    pub noise_scale: f32,

    #[uniform(default = "0.10")]
    pub dissolve_width: f32,

    #[uniform(default = "1.8")]
    pub dissolve_speed: f32,

    #[uniform(default = "0.18")]
    pub displacement: f32,

    #[uniform]
    pub phase_offset: f32,
}

struct DissolveDemo {
    box_node: NodeHandle,
    sphere_node: NodeHandle,
    shard_node: NodeHandle,
    controls: OrbitControls,
    time: f32,
}

impl AppHandler for DissolveDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        let ground = scene.spawn_plane(
            14.0,
            14.0,
            PhongMaterial::new(Vec4::new(0.08, 0.08, 0.10, 1.0)).with_side(Side::Double),
            &engine.assets,
        );
        scene
            .edit_node(ground)
            .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
            .set_position(0.0, -0.1, 0.0);

        let light = scene.add_light(Light::new_directional(Vec3::new(1.0, 0.95, 0.9), 3.2));
        scene
            .edit_node(light)
            .set_position(6.0, 9.0, 4.0)
            .look_at(Vec3::new(0.0, 1.2, 0.0));

        let box_node = scene.spawn_box(
            1.7,
            1.7,
            1.7,
            Material::new_custom(
                DissolveMaterial::default()
                    .with_base_color(Vec4::new(0.18, 0.55, 1.0, 1.0))
                    .with_edge_color(Vec4::new(1.20, 0.65, 0.22, 1.0))
                    .with_noise_scale(2.3)
                    .with_dissolve_width(0.10)
                    .with_dissolve_speed(1.6)
                    .with_displacement(0.20)
                    .with_phase_offset(0.0)
                    .with_alpha_mode(AlphaMode::Blend)
                    .with_depth_write(false)
                    .with_side(Side::Double),
            ),
            &engine.assets,
        );
        scene.edit_node(box_node).set_position(-2.4, 1.2, 0.0);

        let sphere_node = scene.spawn_sphere(
            1.0,
            Material::new_custom(
                DissolveMaterial::default()
                    .with_base_color(Vec4::new(0.22, 1.00, 0.74, 1.0))
                    .with_edge_color(Vec4::new(1.00, 1.25, 0.35, 1.0))
                    .with_noise_scale(3.0)
                    .with_dissolve_width(0.12)
                    .with_dissolve_speed(2.2)
                    .with_displacement(0.15)
                    .with_phase_offset(1.4)
                    .with_alpha_mode(AlphaMode::Blend)
                    .with_depth_write(false)
                    .with_side(Side::Double),
            ),
            &engine.assets,
        );
        scene.edit_node(sphere_node).set_position(0.0, 1.3, 0.0);

        let shard_node = scene.spawn_box(
            0.9,
            2.6,
            0.9,
            Material::new_custom(
                DissolveMaterial::default()
                    .with_base_color(Vec4::new(0.90, 0.22, 1.05, 1.0))
                    .with_edge_color(Vec4::new(1.25, 0.72, 1.20, 1.0))
                    .with_noise_scale(2.9)
                    .with_dissolve_width(0.08)
                    .with_dissolve_speed(1.9)
                    .with_displacement(0.12)
                    .with_phase_offset(2.6)
                    .with_alpha_mode(AlphaMode::Blend)
                    .with_depth_write(false)
                    .with_side(Side::Double),
            ),
            &engine.assets,
        );
        scene.edit_node(shard_node).set_position(2.3, 1.5, -0.2);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(0.0, 2.9, 8.0)
            .look_at(Vec3::new(0.0, 1.2, 0.0));
        scene.active_camera = Some(camera);

        Self {
            box_node,
            sphere_node,
            shard_node,
            controls: OrbitControls::new(Vec3::new(0.0, 2.9, 8.0), Vec3::new(0.0, 1.2, 0.0)),
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        scene.edit_node(self.box_node).set_rotation(
            Quat::from_rotation_y(self.time * 0.7) * Quat::from_rotation_x(self.time * 0.3),
        );

        scene
            .edit_node(self.sphere_node)
            .set_position(0.0, 1.3 + (self.time * 1.6).sin() * 0.25, 0.0)
            .set_rotation(Quat::from_rotation_y(-self.time * 0.8));

        scene.edit_node(self.shard_node).set_rotation(
            Quat::from_rotation_z(self.time * 0.9) * Quat::from_rotation_y(self.time * 1.1),
        );

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new().run::<DissolveDemo>()
}
