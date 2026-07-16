//! [gallery]
//! name = "Custom Material - Triplanar"
//! category = "Materials"
//! description = "Production-style triplanar projection with studio lighting, HDR environment and sculptural forms."
//! order = 324
//!

//! Custom Material Example — Triplanar Studio Surface
//!
//! Demonstrates a more production-oriented custom material by combining:
//! 1. World-space triplanar texture projection that does not rely on mesh UVs
//! 2. Height tinting and edge fresnel accents for presentation lighting
//! 3. A cool studio scene with HDR environment lighting

use myth::prelude::*;
use myth_resources::myth_material;
use myth_resources::uniforms::Mat3Uniform;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

const TRIPLANAR_SHADER: &str = r#"
fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn sample_triplanar(position: vec3<f32>, weights: vec3<f32>, scale: f32) -> vec3<f32> {
    let uv_x = fract(position.yz * scale);
    let uv_y = fract(position.xz * scale);
    let uv_z = fract(position.xy * scale);

    let sx = textureSample(t_map, s_map, uv_x).rgb;
    let sy = textureSample(t_map, s_map, uv_y).rgb;
    let sz = textureSample(t_map, s_map, uv_z).rgb;
    return sx * weights.x + sy * weights.y + sz * weights.z;
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

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    $$ if HAS_NORMAL is defined
    normal = normalize(in.normal);
    $$ endif

    let n = abs(normal);
    var weights = pow(n, vec3<f32>(u_material.blend_sharpness));
    weights = weights / max(weights.x + weights.y + weights.z, 1e-5);

    let triplanar = sample_triplanar(in.world_position, weights, u_material.texture_scale);
    let contrast = pow(max(triplanar, vec3<f32>(0.001)), vec3<f32>(u_material.contrast));

    let height_mask = saturate(in.world_position.y * 0.18 + 0.5);
    let height_tint = mix(u_material.shadow_tint.rgb, u_material.highlight_tint.rgb, height_mask);

    let key_light = normalize(vec3<f32>(0.45, 0.85, 0.28));
    let fill_light = normalize(vec3<f32>(-0.28, 0.55, -0.62));
    let ndl = saturate(dot(normal, key_light));
    let fill = saturate(dot(normal, fill_light));
    let view = normalize(u_render_state.camera_position - in.world_position);
    let fresnel = pow(1.0 - saturate(dot(normal, view)), 3.2);

    let base = contrast * u_material.base_tint.rgb * height_tint;
    let color =
        base * (0.26 + ndl * 0.92 + fill * 0.18) +
        u_material.edge_tint.rgb * fresnel * 0.55;

    return pack_fragment_output(vec4<f32>(color, u_material.opacity));
}
"#;

#[myth_material(shader = "custom_triplanar_surface", shader_src = TRIPLANAR_SHADER)]
pub struct TriplanarMaterial {
    #[uniform(default = "Vec4::new(0.95, 0.95, 1.02, 1.0)")]
    pub base_tint: Vec4,

    #[uniform(default = "Vec4::new(0.86, 0.94, 1.18, 1.0)")]
    pub edge_tint: Vec4,

    #[uniform(default = "Vec4::new(0.36, 0.40, 0.48, 1.0)")]
    pub shadow_tint: Vec4,

    #[uniform(default = "Vec4::new(1.05, 1.02, 0.96, 1.0)")]
    pub highlight_tint: Vec4,

    #[uniform(default = "1.0")]
    pub opacity: f32,

    #[uniform]
    pub alpha_test: f32,

    #[uniform(default = "0.24")]
    pub texture_scale: f32,

    #[uniform(default = "5.0")]
    pub blend_sharpness: f32,

    #[uniform(default = "1.15")]
    pub contrast: f32,

    #[texture]
    pub map: TextureSlot,
}

struct TriplanarDemo {
    monolith_a: NodeHandle,
    monolith_b: NodeHandle,
    sphere: NodeHandle,
    controls: OrbitControls,
    time: f32,
}

impl AppHandler for TriplanarDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        let env = engine.assets.load_cube_texture(
            [
                format!("{}envs/Park2/posx.jpg", ASSET_PATH),
                format!("{}envs/Park2/negx.jpg", ASSET_PATH),
                format!("{}envs/Park2/posy.jpg", ASSET_PATH),
                format!("{}envs/Park2/negy.jpg", ASSET_PATH),
                format!("{}envs/Park2/posz.jpg", ASSET_PATH),
                format!("{}envs/Park2/negz.jpg", ASSET_PATH),
            ],
            ColorSpace::Srgb,
            true,
        );
        scene.environment.set_env_map(Some(env));
        scene.background.set_mode(BackgroundMode::Gradient {
            top: Vec4::new(0.88, 0.91, 0.96, 1.0),
            bottom: Vec4::new(0.46, 0.49, 0.55, 1.0),
        });
        scene
            .tone_mapping
            .set_mode(myth::ToneMappingMode::AgX(myth::AgxLook::Punchy));

        let key_light = scene.add_light(Light::new_directional(Vec3::new(1.0, 0.98, 0.94), 3.2));
        scene
            .edit_node(key_light)
            .set_position(6.0, 10.0, 5.0)
            .look_at(Vec3::new(0.0, 1.4, 0.0));

        let fill_light = scene.add_light(Light::new_directional(Vec3::new(0.62, 0.70, 0.85), 0.9));
        scene
            .edit_node(fill_light)
            .set_position(-7.0, 5.0, -8.0)
            .look_at(Vec3::new(0.0, 1.2, 0.0));

        let floor = scene.spawn_plane(
            18.0,
            18.0,
            PhongMaterial::new(Vec4::new(0.26, 0.28, 0.31, 1.0)).with_side(Side::Double),
            &engine.assets,
        );
        scene
            .edit_node(floor)
            .set_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .set_position(0.0, -0.05, 0.0);

        let albedo = engine.assets.load_texture(
            format!("{}DamagedHelmet/glTF/Default_albedo.jpg", ASSET_PATH),
            ColorSpace::Srgb,
            true,
        );

        let material_handle = engine.assets.materials.add(Material::new_custom(
            TriplanarMaterial::default()
                .with_map(albedo)
                .with_texture_scale(0.28)
                .with_blend_sharpness(6.0)
                .with_contrast(1.08),
        ));

        let monolith_a = scene.spawn_box(1.4, 4.8, 1.4, material_handle, &engine.assets);
        scene
            .edit_node(monolith_a)
            .set_position(-2.2, 2.35, 0.0)
            .set_rotation(Quat::from_rotation_y(0.18));

        let monolith_b = scene.spawn_box(1.9, 3.2, 1.9, material_handle, &engine.assets);
        scene
            .edit_node(monolith_b)
            .set_position(2.0, 1.55, -0.8)
            .set_rotation(Quat::from_rotation_y(-0.34) * Quat::from_rotation_x(0.12));

        let sphere = scene.spawn_sphere(1.25, material_handle, &engine.assets);
        scene.edit_node(sphere).set_position(0.0, 1.3, 2.2);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(0.0, 2.9, 9.0)
            .look_at(Vec3::new(0.0, 1.8, 0.0));
        scene.active_camera = Some(camera);

        Self {
            monolith_a,
            monolith_b,
            sphere,
            controls: OrbitControls::new(Vec3::new(0.0, 2.9, 9.0), Vec3::new(0.0, 1.8, 0.0)),
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        scene
            .edit_node(self.monolith_a)
            .set_rotation(Quat::from_rotation_y(0.18 + self.time * 0.18));
        scene.edit_node(self.monolith_b).set_rotation(
            Quat::from_rotation_y(-0.34 - self.time * 0.12) * Quat::from_rotation_x(0.12),
        );
        scene.edit_node(self.sphere).set_rotation(
            Quat::from_rotation_y(self.time * 0.48) * Quat::from_rotation_x(self.time * 0.12),
        );

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new().run::<TriplanarDemo>()
}
