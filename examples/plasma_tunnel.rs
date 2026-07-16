//! [gallery]
//! name = "Plasma Tunnel"
//! category = "Post Effects"
//! description = "Classic procedural plasma tunnel rendered with a custom shader, neon portal ring, and bloom-heavy staging."
//! order = 450
//!

use std::f32::consts::FRAC_PI_2;

use myth::prelude::*;
use myth_dev_utils::FpsCounter;
use myth_resources::myth_material;

const PLASMA_TUNNEL_SHADER: &str = r#"
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
    let time = u_render_state.time * u_material.speed;
    let centered = (in.uv * 2.0 - vec2<f32>(1.0, 1.0)) * vec2<f32>(1.25, 1.0);
    let radius = length(centered);
    let angle = atan2(centered.y, centered.x);

    let tunnel_depth = 0.18 / max(radius, 0.085);
    let tunnel_wave = sin(tunnel_depth * u_material.warp_scale - time * 6.0 + angle * 4.2);
    let plasma =
        sin(centered.x * u_material.swirl_scale + time * 1.1) +
        sin(centered.y * (u_material.swirl_scale * 1.35) - time * 1.6) +
        sin((centered.x + centered.y) * (u_material.swirl_scale * 0.7) + time * 0.8);

    let interference = tunnel_wave * 0.45 + (plasma / 3.0) * 0.55;
    let energy = saturate(interference * 0.5 + 0.5);
    let vignette = 1.0 - smoothstep(0.82, 1.35, radius);
    let outer_rim = 1.0 - smoothstep(0.64, 0.98, radius);
    let core = 1.0 - smoothstep(0.0, 0.11, radius);

    var color = mix(u_material.color_a.rgb, u_material.color_b.rgb, energy);
    color = mix(color, u_material.color_c.rgb, smoothstep(0.58, 1.0, energy));

    let shimmer = 0.4 + 1.6 * energy + outer_rim * 0.45 + core * 0.85;
    color *= vignette * shimmer * u_material.glow;
    color += u_material.color_c.rgb * outer_rim * 0.25;
    color += u_material.color_b.rgb * core * 0.55;

    return pack_fragment_output(vec4<f32>(color, 1.0));
}
"#;

#[myth_material(shader = "classic_plasma_tunnel", shader_src = PLASMA_TUNNEL_SHADER)]
pub struct PlasmaTunnelMaterial {
    #[uniform(default = "Vec4::new(0.10, 0.04, 0.24, 1.0)")]
    pub color_a: Vec4,

    #[uniform(default = "Vec4::new(0.08, 1.20, 1.05, 1.0)")]
    pub color_b: Vec4,

    #[uniform(default = "Vec4::new(1.45, 0.18, 0.86, 1.0)")]
    pub color_c: Vec4,

    #[uniform(default = "1.0")]
    pub speed: f32,

    #[uniform(default = "30.0")]
    pub warp_scale: f32,

    #[uniform(default = "6.5")]
    pub swirl_scale: f32,

    #[uniform(default = "1.3")]
    pub glow: f32,

    #[uniform(default = "1.0")]
    pub opacity: f32,

    #[uniform(default = "0.0")]
    pub alpha_test: f32,
}

struct PlasmaTunnelDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    portal_node: NodeHandle,
    ring_root: NodeHandle,
    time: f32,
}

impl AppHandler for PlasmaTunnelDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.015));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.12);
        scene.bloom.set_radius(0.006);

        let portal_node = scene.spawn_plane(
            7.4,
            7.4,
            Material::new_custom(
                PlasmaTunnelMaterial::default()
                    .with_speed(1.1)
                    .with_warp_scale(32.0)
                    .with_glow(1.4)
                    .with_side(Side::Double),
            ),
            &engine.assets,
        );
        scene.edit_node(portal_node).set_position(0.0, 1.7, -0.4);

        let floor = scene.spawn_plane(
            18.0,
            18.0,
            PhysicalMaterial::new(Vec4::new(0.08, 0.09, 0.11, 1.0))
                .with_roughness(0.92)
                .with_side(Side::Double),
            &engine.assets,
        );
        scene
            .edit_node(floor)
            .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
            .set_position(0.0, -0.02, 0.0)
            .set_shadows(false, true);

        let pillar_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.16, 0.18, 0.22, 1.0))
                .with_metalness(0.55)
                .with_roughness(0.24),
        );
        let neon_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.10, 0.10, 0.15, 1.0))
                .with_emissive(Vec3::new(0.35, 0.9, 1.0), 4.8)
                .with_roughness(0.16),
        );

        for &(x, z) in &[
            (-5.4, -2.8),
            (-5.4, 2.8),
            (5.4, -2.8),
            (5.4, 2.8),
            (-2.8, -4.8),
            (2.8, -4.8),
        ] {
            let pillar = scene.spawn_box(0.8, 5.0, 0.8, pillar_material, &engine.assets);
            scene
                .edit_node(pillar)
                .set_position(x, 2.5, z)
                .set_shadows(true, true);

            let band = scene.spawn_box(0.95, 0.12, 0.95, neon_material, &engine.assets);
            scene
                .edit_node(band)
                .set_position(x, 1.4, z)
                .set_cast_shadows(false)
                .set_receive_shadows(false);
        }

        let ring_root = scene.create_node_with_name("PlasmaRing");
        scene.push_root_node(ring_root);
        scene.edit_node(ring_root).set_position(0.0, 1.7, -0.2);

        let ring_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        for segment in 0..36 {
            let angle = segment as f32 / 36.0 * std::f32::consts::TAU;
            let radial = Vec3::new(angle.cos(), angle.sin(), 0.0);
            let segment_handle =
                scene.add_mesh_to_parent(Mesh::new(ring_geo, neon_material), ring_root);
            scene
                .edit_node(segment_handle)
                .set_position_vec(radial * 3.35)
                .set_rotation(Quat::from_rotation_z(angle + FRAC_PI_2))
                .set_scale_xyz(0.20, 0.92, 0.18)
                .set_cast_shadows(false)
                .set_receive_shadows(false);
        }

        let mut key = Light::new_directional(Vec3::new(0.85, 0.92, 1.0), 1.6);
        key.cast_shadows = true;
        if let Some(shadow) = key.shadow.as_mut() {
            shadow.map_size = 2048;
            shadow.normal_bias = 0.0;
        }
        let key = scene.add_light(key);
        scene
            .edit_node(key)
            .set_position(8.0, 10.0, 6.0)
            .look_at(Vec3::new(0.0, 1.6, -0.5));

        let fill = scene.add_light(Light::new_point(Vec3::new(0.4, 0.95, 1.0), 1.2, 20.0));
        scene.edit_node(fill).set_position(-3.5, 2.5, 3.0);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(cam)
            .set_position(0.0, 1.8, 8.0)
            .look_at(Vec3::new(0.0, 1.7, -0.2));
        scene.active_camera = Some(cam);

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 1.8, 8.0), Vec3::new(0.0, 1.7, -0.2)),
            fps_counter: FpsCounter::new(),
            portal_node,
            ring_root,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        scene
            .edit_node(self.ring_root)
            .set_rotation(Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, self.time * 0.5))
            .set_scale(1.0 + (self.time * 1.8).sin() * 0.05);

        scene.edit_node(self.portal_node).set_scale_xyz(
            1.0 + (self.time * 0.9).sin() * 0.015,
            1.0 + (self.time * 1.1).cos() * 0.02,
            1.0,
        );

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Plasma Tunnel | FPS: {:.1}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Plasma Tunnel")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<PlasmaTunnelDemo>()
}
