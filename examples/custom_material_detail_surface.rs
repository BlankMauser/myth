//! [gallery]
//! name = "Custom Material - Detail Surface"
//! category = "Materials"
//! description = "Production-style textured surface using albedo, normal and emissive mask inputs in a dark tech showroom."
//! order = 326
//!

//! Custom Material Example — Detail Surface Showroom
//!
//! Demonstrates a practical textured custom material:
//! 1. Albedo, normal and emissive mask texture slots
//! 2. Tangent-free normal reconstruction using derivatives
//! 3. Masked glow details and presentation-oriented rim light

use myth::prelude::*;
use myth_resources::myth_material;
use myth_resources::uniforms::Mat3Uniform;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

const DETAIL_SURFACE_SHADER: &str = r#"
{$ include 'core/common' $}

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
    let time = u_render_state.time * u_material.flow_speed;

    $$ if HAS_MAP is defined
    let base_uv = fract(in.map_uv * u_material.uv_scale);
    let base_tex = textureSample(t_map, s_map, base_uv).rgb;
    $$ else
    let base_tex = vec3<f32>(1.0, 1.0, 1.0);
    $$ endif

    var normal = vec3<f32>(0.0, 0.0, 1.0);
    $$ if HAS_NORMAL is defined
    normal = normalize(in.normal);
    $$ endif

    let view = normalize(u_render_state.camera_position - in.world_position);

    $$ if HAS_NORMAL_MAP is defined
    let normal_uv = fract(in.normal_map_uv * u_material.uv_scale);
    let map_n = textureSample(t_normal_map, s_normal_map, normal_uv).xyz * 2.0 - 1.0;
    let tbn = getTangentFrame(view, normal, normal_uv);
    let scaled_n = vec3<f32>(map_n.xy * u_material.normal_scale, map_n.z);
    normal = normalize(tbn * scaled_n);
    $$ endif

    $$ if HAS_EMISSIVE_MAP is defined
    let emissive_uv = fract(in.emissive_map_uv * (u_material.uv_scale * 1.15) + vec2<f32>(time * 0.04, 0.0));
    let emissive_mask = textureSample(t_emissive_map, s_emissive_map, emissive_uv).rgb;
    $$ else
    let emissive_mask = vec3<f32>(0.0, 0.0, 0.0);
    $$ endif

    let key_light = normalize(vec3<f32>(-0.34, 0.82, 0.46));
    let fill_light = normalize(vec3<f32>(0.52, 0.18, -0.82));
    let ndl = saturate(dot(normal, key_light));
    let fill = saturate(dot(normal, fill_light));
    let rim = pow(1.0 - saturate(dot(normal, view)), u_material.rim_power);
    let scan = saturate(sin(in.world_position.y * 22.0 - time * 7.0) * 0.5 + 0.5);
    let glow = dot(emissive_mask, vec3<f32>(0.299, 0.587, 0.114));

    let base = base_tex * u_material.base_tint.rgb;
    let color =
        base * (0.18 + ndl * 0.98 + fill * 0.14) +
        u_material.glow_color.rgb * glow * u_material.emissive_intensity * (0.42 + scan * 0.58) +
        u_material.glow_color.rgb * rim * 0.38;

    return pack_fragment_output(vec4<f32>(color, u_material.opacity));
}
"#;

#[myth_material(shader = "custom_detail_surface", shader_src = DETAIL_SURFACE_SHADER)]
pub struct DetailSurfaceMaterial {
    #[uniform(default = "Vec4::new(0.96, 0.96, 1.0, 1.0)")]
    pub base_tint: Vec4,

    #[uniform(default = "Vec4::new(0.18, 0.95, 1.18, 1.0)")]
    pub glow_color: Vec4,

    #[uniform(default = "1.0")]
    pub opacity: f32,

    #[uniform]
    pub alpha_test: f32,

    #[uniform(default = "Vec2::new(1.0, -1.0)")]
    pub normal_scale: Vec2,

    #[uniform(default = "2.6")]
    pub emissive_intensity: f32,

    #[uniform(default = "1.2")]
    pub flow_speed: f32,

    #[uniform(default = "3.8")]
    pub rim_power: f32,

    #[uniform(default = "1.0")]
    pub uv_scale: f32,

    #[texture]
    pub map: TextureSlot,

    #[texture]
    pub normal_map: TextureSlot,

    #[texture]
    pub emissive_map: TextureSlot,
}

struct DetailSurfaceDemo {
    panel_left: NodeHandle,
    panel_right: NodeHandle,
    center_block: NodeHandle,
    controls: OrbitControls,
    time: f32,
}

impl AppHandler for DetailSurfaceDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();
        scene.background.set_mode(BackgroundMode::Gradient {
            top: Vec4::new(0.02, 0.05, 0.11, 1.0),
            bottom: Vec4::new(0.0, 0.0, 0.01, 1.0),
        });
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.025);
        scene.bloom.set_radius(0.006);
        scene
            .tone_mapping
            .set_mode(myth::ToneMappingMode::AgX(myth::AgxLook::Punchy));

        let floor = scene.spawn_plane(
            18.0,
            18.0,
            PhongMaterial::new(Vec4::new(0.04, 0.05, 0.06, 1.0)).with_side(Side::Double),
            &engine.assets,
        );
        scene
            .edit_node(floor)
            .set_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .set_position(0.0, -0.05, 0.0);

        let light = scene.add_light(Light::new_directional(Vec3::new(0.78, 0.92, 1.0), 2.2));
        scene
            .edit_node(light)
            .set_position(5.0, 8.0, 4.0)
            .look_at(Vec3::new(0.0, 1.5, 0.0));

        let albedo = engine.assets.load_texture(
            format!("{}DamagedHelmet/glTF/Default_albedo.jpg", ASSET_PATH),
            ColorSpace::Srgb,
            true,
        );
        let normal = engine.assets.load_texture(
            format!("{}DamagedHelmet/glTF/Default_normal.jpg", ASSET_PATH),
            ColorSpace::Linear,
            true,
        );
        let emissive = engine.assets.load_texture(
            format!("{}DamagedHelmet/glTF/Default_emissive.jpg", ASSET_PATH),
            ColorSpace::Srgb,
            true,
        );

        let panel_left = scene.spawn_plane(
            2.6,
            4.2,
            Material::new_custom(
                DetailSurfaceMaterial::default()
                    .with_map(albedo)
                    .with_normal_map(normal)
                    .with_emissive_map(emissive)
                    .with_emissive_intensity(2.8)
                    .with_flow_speed(1.0)
                    .with_rim_power(3.2)
                    .with_uv_scale(1.0),
            ),
            &engine.assets,
        );
        scene
            .edit_node(panel_left)
            .set_position(-2.3, 2.0, 0.0)
            .set_rotation(Quat::from_rotation_y(0.32));

        let panel_right = scene.spawn_plane(
            2.6,
            4.2,
            Material::new_custom(
                DetailSurfaceMaterial::default()
                    .with_map(albedo)
                    .with_normal_map(normal)
                    .with_emissive_map(emissive)
                    .with_emissive_intensity(3.2)
                    .with_flow_speed(1.5)
                    .with_rim_power(4.2)
                    .with_uv_scale(1.1),
            ),
            &engine.assets,
        );
        scene
            .edit_node(panel_right)
            .set_position(2.3, 2.0, -0.2)
            .set_rotation(Quat::from_rotation_y(-0.36));

        let center_block = scene.spawn_box(
            1.6,
            1.6,
            1.6,
            Material::new_custom(
                DetailSurfaceMaterial::default()
                    .with_map(albedo)
                    .with_normal_map(normal)
                    .with_emissive_map(emissive)
                    .with_emissive_intensity(2.4)
                    .with_flow_speed(0.85)
                    .with_rim_power(3.6)
                    .with_uv_scale(0.9),
            ),
            &engine.assets,
        );
        scene.edit_node(center_block).set_position(0.0, 1.0, 0.5);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(0.0, 2.7, 8.3)
            .look_at(Vec3::new(0.0, 1.8, 0.0));
        scene.active_camera = Some(camera);

        Self {
            panel_left,
            panel_right,
            center_block,
            controls: OrbitControls::new(Vec3::new(0.0, 2.7, 8.3), Vec3::new(0.0, 1.8, 0.0)),
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
            .set_rotation(Quat::from_rotation_y(0.32 + (self.time * 0.7).sin() * 0.08));
        scene
            .edit_node(self.panel_right)
            .set_rotation(Quat::from_rotation_y(
                -0.36 + (self.time * 0.9).cos() * 0.08,
            ));
        scene.edit_node(self.center_block).set_rotation(
            Quat::from_rotation_y(self.time * 0.55) * Quat::from_rotation_x(self.time * 0.18),
        );

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new().run::<DetailSurfaceDemo>()
}
