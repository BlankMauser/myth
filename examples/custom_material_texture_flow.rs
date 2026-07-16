//! [gallery]
//! name = "Custom Material - Texture Flow"
//! category = "Materials"
//! description = "Custom material that samples a texture with animated UV distortion and layered flow lighting."
//! order = 323
//!

//! Custom Material Example — Texture Flow Panels
//!
//! Demonstrates a custom material with texture bindings:
//! 1. A custom `#[texture]` slot on the material
//! 2. Animated multi-layer texture sampling
//! 3. UV distortion and panel surface warping
//! 4. One shader reused across panels and boxes with different parameters

use std::f32::consts::FRAC_PI_2;

use myth::prelude::*;
use myth_resources::myth_material;
use myth_resources::uniforms::Mat3Uniform;

const TEXTURE_FLOW_SHADER: &str = r#"
fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var local_position = vec3<f32>(in.position.xyz);
    var local_normal = vec3<f32>(0.0, 0.0, 1.0);

    $$ if HAS_NORMAL is defined
    local_normal = normalize(vec3<f32>(in.normal.xyz));
    $$ endif

    let time = u_render_state.time * u_material.flow_speed;

    $$ if HAS_UV is defined
    let panel_wave = sin(in.uv.y * 10.0 + time * 2.2) * cos(in.uv.x * 8.0 - time * 1.6);
    $$ else
    let panel_wave = sin(local_position.y * 5.0 + time * 2.2) * cos(local_position.x * 4.0 - time * 1.6);
    $$ endif

    $$ if HAS_NORMAL is defined
    local_position += local_normal * panel_wave * u_material.panel_warp;
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
    let time = u_render_state.time * u_material.flow_speed;

    $$ if HAS_MAP is defined
    let base_uv = fract(in.map_uv * u_material.repeat + vec2<f32>(time * 0.12, 0.0));
    let warp = vec2<f32>(
        sin(base_uv.y * 18.0 + time * 2.6),
        cos(base_uv.x * 14.0 - time * 1.8)
    ) * u_material.distortion * 0.08;

    let uv_a = fract(base_uv + warp);
    let uv_b = fract(base_uv * 1.65 - warp + vec2<f32>(-time * 0.18, time * 0.07));
    let sample_a = textureSample(t_map, s_map, uv_a);
    let sample_b = textureSample(t_map, s_map, uv_b);
    $$ else
    let sample_a = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let sample_b = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    $$ endif

    var normal = vec3<f32>(0.0, 0.0, 1.0);
    $$ if HAS_NORMAL is defined
    normal = normalize(in.normal);
    $$ endif

    let view = normalize(u_render_state.camera_position - in.world_position);
    let fresnel = pow(1.0 - saturate(dot(normal, view)), 2.6);
    let lum_a = dot(sample_a.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let lum_b = dot(sample_b.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let stripes = saturate(sin((in.world_position.y + time * 0.9) * 24.0) * 0.5 + 0.5);

    let color =
        sample_a.rgb * u_material.tint.rgb +
        sample_b.rgb * u_material.glow_color.rgb * 0.45 +
        u_material.glow_color.rgb * stripes * 0.16 +
        u_material.glow_color.rgb * fresnel * 0.55;

    let alpha = saturate(
        u_material.opacity * (0.22 + lum_a * 0.45 + lum_b * 0.20 + stripes * 0.18 + fresnel * 0.30)
    );

    if (alpha < u_material.alpha_test) {
        discard;
    }

    return pack_fragment_output(vec4<f32>(color, alpha));
}
"#;

#[myth_material(shader = "custom_texture_flow", shader_src = TEXTURE_FLOW_SHADER)]
pub struct TextureFlowMaterial {
    #[uniform(default = "Vec4::new(0.14, 0.72, 1.20, 1.0)")]
    pub tint: Vec4,

    #[uniform(default = "Vec4::new(0.92, 1.12, 1.28, 1.0)")]
    pub glow_color: Vec4,

    #[uniform(default = "0.88")]
    pub opacity: f32,

    #[uniform(default = "0.02")]
    pub alpha_test: f32,

    #[uniform(default = "3.0")]
    pub repeat: f32,

    #[uniform(default = "1.6")]
    pub flow_speed: f32,

    #[uniform(default = "1.0")]
    pub distortion: f32,

    #[uniform(default = "0.08")]
    pub panel_warp: f32,

    #[texture]
    pub map: TextureSlot,
}

struct TextureFlowDemo {
    panel_left: NodeHandle,
    panel_right: NodeHandle,
    center_box: NodeHandle,
    controls: OrbitControls,
    time: f32,
}

impl AppHandler for TextureFlowDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();
        let checker = engine.assets.checkerboard(512, 16);

        let floor = scene.spawn_plane(
            14.0,
            14.0,
            PhongMaterial::new(Vec4::new(0.06, 0.07, 0.09, 1.0)).with_side(Side::Double),
            &engine.assets,
        );
        scene
            .edit_node(floor)
            .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
            .set_position(0.0, -0.15, 0.0);

        let light = scene.add_light(Light::new_directional(Vec3::new(0.95, 1.0, 1.0), 2.8));
        scene
            .edit_node(light)
            .set_position(5.0, 8.0, 6.0)
            .look_at(Vec3::new(0.0, 1.2, 0.0));

        let panel_left = scene.spawn_plane(
            2.8,
            3.8,
            Material::new_custom(
                TextureFlowMaterial::default()
                    .with_tint(Vec4::new(0.18, 0.85, 1.20, 1.0))
                    .with_glow_color(Vec4::new(0.95, 1.18, 1.28, 1.0))
                    .with_map(checker)
                    .with_repeat(3.6)
                    .with_flow_speed(1.8)
                    .with_distortion(1.0)
                    .with_panel_warp(0.09)
                    .with_alpha_mode(AlphaMode::Blend)
                    .with_depth_write(false)
                    .with_side(Side::Double),
            ),
            &engine.assets,
        );
        scene
            .edit_node(panel_left)
            .set_position(-2.2, 1.9, 0.0)
            .set_rotation(Quat::from_rotation_y(0.38));

        let panel_right = scene.spawn_plane(
            2.8,
            3.8,
            Material::new_custom(
                TextureFlowMaterial::default()
                    .with_tint(Vec4::new(1.05, 0.22, 0.85, 1.0))
                    .with_glow_color(Vec4::new(1.30, 0.92, 1.20, 1.0))
                    .with_map(checker)
                    .with_repeat(4.2)
                    .with_flow_speed(2.2)
                    .with_distortion(1.4)
                    .with_panel_warp(0.12)
                    .with_alpha_mode(AlphaMode::Blend)
                    .with_depth_write(false)
                    .with_side(Side::Double),
            ),
            &engine.assets,
        );
        scene
            .edit_node(panel_right)
            .set_position(2.2, 1.9, -0.3)
            .set_rotation(Quat::from_rotation_y(-0.42));

        let center_box = scene.spawn_box(
            1.4,
            1.4,
            1.4,
            Material::new_custom(
                TextureFlowMaterial::default()
                    .with_tint(Vec4::new(0.20, 1.00, 0.72, 1.0))
                    .with_glow_color(Vec4::new(1.00, 1.24, 0.94, 1.0))
                    .with_map(checker)
                    .with_repeat(2.2)
                    .with_flow_speed(1.4)
                    .with_distortion(0.8)
                    .with_panel_warp(0.05)
                    .with_alpha_mode(AlphaMode::Blend)
                    .with_depth_write(false)
                    .with_side(Side::Double),
            ),
            &engine.assets,
        );
        scene.edit_node(center_box).set_position(0.0, 1.1, 0.0);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(0.0, 2.7, 8.2)
            .look_at(Vec3::new(0.0, 1.6, 0.0));
        scene.active_camera = Some(camera);

        Self {
            panel_left,
            panel_right,
            center_box,
            controls: OrbitControls::new(Vec3::new(0.0, 2.7, 8.2), Vec3::new(0.0, 1.6, 0.0)),
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        scene
            .edit_node(self.panel_left)
            .set_rotation(Quat::from_rotation_y(0.38 + (self.time * 0.8).sin() * 0.12));

        scene
            .edit_node(self.panel_right)
            .set_rotation(Quat::from_rotation_y(
                -0.42 + (self.time * 0.9).cos() * 0.14,
            ));

        scene.edit_node(self.center_box).set_rotation(
            Quat::from_rotation_y(self.time * 0.9) * Quat::from_rotation_x(self.time * 0.35),
        );

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new().run::<TextureFlowDemo>()
}
