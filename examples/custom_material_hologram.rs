//! [gallery]
//! name = "Custom Material - Hologram"
//! category = "Materials"
//! description = "Animated hologram material with custom uniforms, vertex displacement and transparent scan effects."
//! order = 320
//!

//! Custom Material Example — Hologram Energy Field
//!
//! Demonstrates a more advanced custom material by combining:
//! 1. A custom material struct with several tweakable uniforms
//! 2. A WGSL shader template with animated vertex displacement
//! 3. View-dependent Fresnel glow, scan lines and procedural grids
//! 4. Multiple material instances sharing one shader with different parameters

use std::f32::consts::FRAC_PI_2;

use myth::prelude::*;
use myth_resources::myth_material;

const HOLOGRAM_SHADER: &str = r#"
fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var local_position = vec3<f32>(in.position.xyz);
    var local_normal = vec3<f32>(0.0, 1.0, 0.0);

    $$ if HAS_NORMAL is defined
    local_normal = normalize(vec3<f32>(in.normal.xyz));
    $$ endif

    let time = u_render_state.time * u_material.pulse_speed;
    let sweep = sin(local_position.y * u_material.scan_density - time * 2.4);
    let radial = length(local_position.xz);
    let ripple = sin(radial * (u_material.grid_scale * 2.0) - time * 3.5);
    let glitch = hash12(local_position.xz * 8.0 + vec2<f32>(floor(u_render_state.time * 12.0), 0.0)) - 0.5;
    let displacement =
        (0.35 + 0.65 * sweep) * u_material.displacement +
        ripple * u_material.displacement * 0.35 +
        glitch * u_material.displacement * 0.12;

    $$ if HAS_NORMAL is defined
    local_position += local_normal * displacement;
    $$ else
    local_position.y += displacement;
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
    let time = u_render_state.time * u_material.pulse_speed;
    var normal = vec3<f32>(0.0, 1.0, 0.0);

    $$ if HAS_NORMAL is defined
    normal = normalize(in.normal);
    $$ endif

    let view = normalize(u_render_state.camera_position - in.world_position);
    let fresnel = pow(1.0 - saturate(dot(normal, view)), u_material.fresnel_power);

    let scan = saturate(sin(in.world_position.y * u_material.scan_density - time * 6.0) * 0.5 + 0.5);
    let radial = length(in.world_position.xz);
    let rings = saturate(sin(radial * (u_material.grid_scale * 3.0) - time * 4.0) * 0.5 + 0.5);

    $$ if HAS_UV is defined
    let grid_uv = abs(fract(in.uv * u_material.grid_scale) - 0.5);
    let grid = smoothstep(0.42, 0.5, max(grid_uv.x, grid_uv.y));
    $$ else
    let grid_uv = abs(fract(in.world_position.xz * u_material.grid_scale * 0.35) - 0.5);
    let grid = smoothstep(0.42, 0.5, max(grid_uv.x, grid_uv.y));
    $$ endif

    let sparkle = hash12(
        floor(in.world_position.xz * 12.0) + vec2<f32>(floor(time * 7.0), floor(in.world_position.y * 5.0))
    );
    let flicker = 0.88 + 0.12 * sin(time * 11.0 + in.world_position.y * 9.0);

    let base = u_material.base_color.rgb * (0.18 + grid * 0.40 + rings * 0.18);
    let sweep = u_material.scan_color.rgb * (scan * 0.65 + sparkle * 0.10);
    let edge = u_material.edge_color.rgb * fresnel * u_material.fresnel_intensity;

    let color = (base + sweep + edge) * flicker;
    let alpha = saturate(
        u_material.opacity * (0.14 + grid * 0.24 + rings * 0.14 + scan * 0.20 + fresnel * 0.72)
    );

    if (alpha < u_material.alpha_test) {
        discard;
    }

    return pack_fragment_output(vec4<f32>(color, alpha));
}
"#;

#[myth_material(shader = "hologram_energy", shader_src = HOLOGRAM_SHADER)]
pub struct HologramMaterial {
    /// Main body color of the hologram.
    #[uniform(default = "Vec4::new(0.08, 0.78, 1.15, 1.0)")]
    pub base_color: Vec4,

    /// Fresnel edge glow color.
    #[uniform(default = "Vec4::new(0.85, 0.98, 1.35, 1.0)")]
    pub edge_color: Vec4,

    /// Moving scan-line accent color.
    #[uniform(default = "Vec4::new(0.10, 0.58, 1.30, 1.0)")]
    pub scan_color: Vec4,

    /// Overall transparency.
    #[uniform(default = "0.92")]
    pub opacity: f32,

    /// Optional alpha cutoff.
    #[uniform]
    pub alpha_test: f32,

    /// Density of the procedural grid pattern.
    #[uniform(default = "8.0")]
    pub grid_scale: f32,

    /// Density of the vertical scan effect.
    #[uniform(default = "13.0")]
    pub scan_density: f32,

    /// Animation speed multiplier.
    #[uniform(default = "2.4")]
    pub pulse_speed: f32,

    /// Fresnel falloff exponent.
    #[uniform(default = "4.0")]
    pub fresnel_power: f32,

    /// Fresnel glow strength.
    #[uniform(default = "1.8")]
    pub fresnel_intensity: f32,

    /// Vertex displacement amplitude.
    #[uniform(default = "0.12")]
    pub displacement: f32,
}

impl HologramMaterial {
    #[must_use]
    pub fn with_fresnel(self, power: f32, intensity: f32) -> Self {
        let mut uniforms = self.uniforms.write();
        uniforms.fresnel_power = power;
        uniforms.fresnel_intensity = intensity;
        drop(uniforms);
        self
    }
}

struct CustomMaterialDemo {
    core_node: NodeHandle,
    satellite_box: NodeHandle,
    satellite_sphere: NodeHandle,
    controls: OrbitControls,
    time: f32,
}

impl AppHandler for CustomMaterialDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        let ground = scene.spawn_plane(
            18.0,
            18.0,
            PhongMaterial::new(Vec4::new(0.05, 0.07, 0.11, 1.0)).with_side(Side::Double),
            &engine.assets,
        );
        scene
            .edit_node(ground)
            .set_position(0.0, -0.05, 0.0)
            .set_rotation(Quat::from_rotation_x(-FRAC_PI_2));

        let key_light = scene.add_light(Light::new_directional(Vec3::new(0.85, 0.92, 1.0), 3.5));
        scene
            .edit_node(key_light)
            .set_position(6.0, 10.0, 4.0)
            .look_at(Vec3::new(0.0, 1.4, 0.0));

        let fill_light = scene.add_light(Light::new_directional(Vec3::new(0.2, 0.35, 0.7), 1.1));
        scene
            .edit_node(fill_light)
            .set_position(-5.0, 4.0, -8.0)
            .look_at(Vec3::new(0.0, 1.2, 0.0));

        let core_node = scene.spawn_sphere(
            1.55,
            Material::new_custom(
                HologramMaterial::default()
                    .with_grid_scale(10.0)
                    .with_scan_density(15.0)
                    .with_pulse_speed(2.1)
                    .with_displacement(0.18)
                    .with_fresnel(3.2, 2.4)
                    .with_opacity(0.92)
                    .with_side(Side::Double)
                    .with_alpha_mode(AlphaMode::Blend)
                    .with_depth_write(false),
            ),
            &engine.assets,
        );
        scene.edit_node(core_node).set_position(0.0, 1.55, 0.0);

        let satellite_box = scene.spawn_box(
            0.85,
            2.6,
            0.85,
            Material::new_custom(
                HologramMaterial::default()
                    .with_base_color(Vec4::new(1.00, 0.18, 0.55, 1.0))
                    .with_edge_color(Vec4::new(1.30, 0.75, 1.00, 1.0))
                    .with_scan_color(Vec4::new(0.75, 0.10, 0.95, 1.0))
                    .with_grid_scale(6.0)
                    .with_scan_density(18.0)
                    .with_pulse_speed(3.4)
                    .with_displacement(0.10)
                    .with_fresnel(4.8, 1.7)
                    .with_opacity(0.78)
                    .with_side(Side::Double)
                    .with_alpha_mode(AlphaMode::Blend)
                    .with_depth_write(false),
            ),
            &engine.assets,
        );
        scene.edit_node(satellite_box).set_position(-3.2, 1.4, 0.0);

        let satellite_sphere = scene.spawn_sphere(
            0.9,
            Material::new_custom(
                HologramMaterial::default()
                    .with_base_color(Vec4::new(0.10, 1.05, 0.68, 1.0))
                    .with_edge_color(Vec4::new(0.95, 1.45, 1.10, 1.0))
                    .with_scan_color(Vec4::new(0.15, 0.95, 0.55, 1.0))
                    .with_grid_scale(9.0)
                    .with_scan_density(11.0)
                    .with_pulse_speed(2.8)
                    .with_displacement(0.14)
                    .with_fresnel(2.8, 2.1)
                    .with_opacity(0.84)
                    .with_side(Side::Double)
                    .with_alpha_mode(AlphaMode::Blend)
                    .with_depth_write(false),
            ),
            &engine.assets,
        );
        scene
            .edit_node(satellite_sphere)
            .set_position(3.0, 1.25, -0.8);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(0.0, 2.8, 8.5)
            .look_at(Vec3::new(0.0, 1.4, 0.0));
        scene.active_camera = Some(camera);

        Self {
            core_node,
            satellite_box,
            satellite_sphere,
            controls: OrbitControls::new(Vec3::new(0.0, 2.8, 8.5), Vec3::new(0.0, 1.4, 0.0)),
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        let orbit_a = self.time * 0.85;
        let orbit_b = -self.time * 1.15;

        scene
            .edit_node(self.core_node)
            .set_rotation(Quat::from_rotation_y(self.time * 0.45));

        scene
            .edit_node(self.satellite_box)
            .set_position(
                orbit_a.cos() * 3.1,
                1.35 + (self.time * 1.7).sin() * 0.35,
                orbit_a.sin() * 3.1,
            )
            .set_rotation(
                Quat::from_rotation_x(self.time * 1.4) * Quat::from_rotation_z(self.time * 0.9),
            );

        scene
            .edit_node(self.satellite_sphere)
            .set_position(
                orbit_b.cos() * 2.2,
                1.15 + (self.time * 2.3).sin() * 0.25,
                orbit_b.sin() * 2.2,
            )
            .set_rotation(
                Quat::from_rotation_y(-self.time * 1.1) * Quat::from_rotation_x(self.time * 0.7),
            );

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new().run::<CustomMaterialDemo>()
}
