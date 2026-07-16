//! [gallery]
//! name = "Custom Material - Slope Blend"
//! category = "Materials"
//! description = "Environment-style material with height and slope based surface accumulation under a golden-hour sky."
//! order = 327
//!

//! Custom Material Example — Slope Blend Landscape Study
//!
//! Demonstrates a practical environment material technique:
//! 1. Base texture sampling with world-aware height and slope masks
//! 2. Procedural accumulation for dust / moss style layering
//! 3. A warm procedural sky setup to make the whole set feel like a material study scene

use myth::prelude::*;
use myth_resources::myth_material;
use myth_resources::uniforms::Mat3Uniform;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

const SLOPE_BLEND_SHADER: &str = r#"
fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n00 = hash12(i + vec2<f32>(0.0, 0.0));
    let n10 = hash12(i + vec2<f32>(1.0, 0.0));
    let n01 = hash12(i + vec2<f32>(0.0, 1.0));
    let n11 = hash12(i + vec2<f32>(1.0, 1.0));

    let nx0 = mix(n00, n10, u.x);
    let nx1 = mix(n01, n11, u.x);
    return mix(nx0, nx1, u.y);
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
    $$ if HAS_MAP is defined
    let base_uv = fract(in.map_uv * u_material.repeat);
    let base_tex = textureSample(t_map, s_map, base_uv).rgb;
    $$ else
    let base_tex = vec3<f32>(1.0, 1.0, 1.0);
    $$ endif

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    $$ if HAS_NORMAL is defined
    normal = normalize(in.normal);
    $$ endif

    let noise = noise2(in.world_position.xz * u_material.noise_scale);
    let up_mask = pow(saturate(normal.y * 0.5 + 0.5), u_material.slope_power);
    let height_mask = smoothstep(u_material.height_start, u_material.height_end, in.world_position.y);
    let layer_mask = saturate(up_mask * 0.72 + height_mask * 0.48) * (0.62 + noise * 0.38);
    let dry_mask = saturate((1.0 - up_mask) * 0.75 + noise * 0.15);

    let base = base_tex * mix(u_material.base_tint.rgb, u_material.dry_tint.rgb, dry_mask * 0.35);
    let layer = u_material.layer_tint.rgb * (0.42 + noise * 0.58);

    let key_light = normalize(vec3<f32>(0.22, 0.92, 0.32));
    let fill_light = normalize(vec3<f32>(-0.52, 0.35, -0.62));
    let ndl = saturate(dot(normal, key_light));
    let fill = saturate(dot(normal, fill_light));
    let view = normalize(u_render_state.camera_position - in.world_position);
    let rim = pow(1.0 - saturate(dot(normal, view)), 2.8);

    var color = mix(base, layer, layer_mask * 0.82);
    color = color * (0.22 + ndl * 0.92 + fill * 0.14) + u_material.rim_color.rgb * rim * 0.18;

    return pack_fragment_output(vec4<f32>(color, u_material.opacity));
}
"#;

#[myth_material(shader = "custom_slope_blend", shader_src = SLOPE_BLEND_SHADER)]
pub struct SlopeBlendMaterial {
    #[uniform(default = "Vec4::new(0.72, 0.63, 0.50, 1.0)")]
    pub base_tint: Vec4,

    #[uniform(default = "Vec4::new(0.44, 0.50, 0.26, 1.0)")]
    pub layer_tint: Vec4,

    #[uniform(default = "Vec4::new(0.78, 0.52, 0.33, 1.0)")]
    pub dry_tint: Vec4,

    #[uniform(default = "Vec4::new(1.08, 0.72, 0.46, 1.0)")]
    pub rim_color: Vec4,

    #[uniform(default = "1.0")]
    pub opacity: f32,

    #[uniform]
    pub alpha_test: f32,

    #[uniform(default = "2.6")]
    pub repeat: f32,

    #[uniform(default = "0.34")]
    pub noise_scale: f32,

    #[uniform(default = "2.4")]
    pub slope_power: f32,

    #[uniform(default = "0.2")]
    pub height_start: f32,

    #[uniform(default = "2.8")]
    pub height_end: f32,

    #[texture]
    pub map: TextureSlot,
}

impl SlopeBlendMaterial {
    #[must_use]
    pub fn with_height_range(self, start: f32, end: f32) -> Self {
        let mut uniforms = self.uniforms.write();
        uniforms.height_start = start;
        uniforms.height_end = end;
        drop(uniforms);
        self
    }
}

struct SlopeBlendDemo {
    rock_a: NodeHandle,
    rock_b: NodeHandle,
    rock_c: NodeHandle,
    controls: OrbitControls,
    time: f32,
}

impl AppHandler for SlopeBlendDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();
        let mut sky = ProceduralSkyParams::golden_hour();
        let starbox = engine.assets.load_texture(
            format!("{}envs/Milky_Way_panorama.jpg", ASSET_PATH),
            ColorSpace::Srgb,
            true,
        );
        sky.set_starbox_texture(starbox);
        scene
            .background
            .set_mode(BackgroundMode::procedural_with(sky));
        scene
            .tone_mapping
            .set_mode(myth::ToneMappingMode::AgX(myth::AgxLook::Punchy));

        let mut sun = Light::new_directional(Vec3::new(1.0, 0.92, 0.82), 3.0);
        sun.cast_shadows = true;
        let sun_node = scene.add_light(sun);
        scene
            .edit_node(sun_node)
            .set_position(6.0, 9.0, 5.0)
            .look_at(Vec3::new(0.0, 1.0, 0.0));

        let ground_tex = engine.assets.load_texture(
            format!("{}planets/earth_atmos_4096.jpg", ASSET_PATH),
            ColorSpace::Srgb,
            true,
        );

        let ground = scene.spawn_plane(
            24.0,
            24.0,
            Material::new_custom(
                SlopeBlendMaterial::default()
                    .with_map(ground_tex)
                    .with_repeat(3.4)
                    .with_noise_scale(0.25)
                    .with_slope_power(2.1)
                    .with_height_range(-0.1, 0.8),
            ),
            &engine.assets,
        );
        scene
            .edit_node(ground)
            .set_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .set_position(0.0, 0.0, 0.0);

        let rock_a = scene.spawn_box(
            2.0,
            3.2,
            1.8,
            Material::new_custom(
                SlopeBlendMaterial::default()
                    .with_map(ground_tex)
                    .with_repeat(2.2)
                    .with_noise_scale(0.42)
                    .with_slope_power(2.8)
                    .with_height_range(0.4, 2.4),
            ),
            &engine.assets,
        );
        scene
            .edit_node(rock_a)
            .set_position(-2.6, 1.6, 0.0)
            .set_rotation(Quat::from_rotation_y(0.26) * Quat::from_rotation_z(-0.08));

        let rock_b = scene.spawn_box(
            1.7,
            2.4,
            2.8,
            Material::new_custom(
                SlopeBlendMaterial::default()
                    .with_map(ground_tex)
                    .with_repeat(2.6)
                    .with_noise_scale(0.36)
                    .with_slope_power(2.4)
                    .with_height_range(0.2, 1.8),
            ),
            &engine.assets,
        );
        scene
            .edit_node(rock_b)
            .set_position(0.8, 1.2, -1.1)
            .set_rotation(Quat::from_rotation_y(-0.34) * Quat::from_rotation_x(0.14));

        let rock_c = scene.spawn_sphere(
            1.25,
            Material::new_custom(
                SlopeBlendMaterial::default()
                    .with_map(ground_tex)
                    .with_repeat(2.0)
                    .with_noise_scale(0.38)
                    .with_slope_power(1.9)
                    .with_height_range(0.4, 1.6),
            ),
            &engine.assets,
        );
        scene.edit_node(rock_c).set_position(3.2, 1.25, 1.0);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(0.0, 3.4, 10.0)
            .look_at(Vec3::new(0.0, 1.4, 0.0));
        scene.active_camera = Some(camera);

        Self {
            rock_a,
            rock_b,
            rock_c,
            controls: OrbitControls::new(Vec3::new(0.0, 3.4, 10.0), Vec3::new(0.0, 1.4, 0.0)),
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        scene.edit_node(self.rock_a).set_rotation(
            Quat::from_rotation_y(0.26 + self.time * 0.10) * Quat::from_rotation_z(-0.08),
        );
        scene.edit_node(self.rock_b).set_rotation(
            Quat::from_rotation_y(-0.34 - self.time * 0.08) * Quat::from_rotation_x(0.14),
        );
        scene
            .edit_node(self.rock_c)
            .set_position(3.2, 1.25 + (self.time * 1.2).sin() * 0.08, 1.0);

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new().run::<SlopeBlendDemo>()
}
