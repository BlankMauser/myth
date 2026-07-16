//! [gallery]
//! name = "Custom Material - Aurora Gate"
//! category = "Materials"
//! description = "A full-template material demo built around custom Geometry attributes that auto-map into VertexInput and HAS_* shader defines."
//! order = 325
//!

use std::f32::consts::{FRAC_PI_2, TAU};

use myth::prelude::*;
use myth_resources::myth_material;

const AURORA_GATE_TEMPLATE: &str = r#"
{{ vertex_input_code }}
{{ binding_code }}
{{ scene_lighting_structs }}
{$ include 'core/fragment_output' $}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location({{ loc.next() }}) world_position: vec3<f32>,
    @location({{ loc.next() }}) normal: vec3<f32>,
    $$ if HAS_UV is defined
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
    @location({{ loc.next() }}) portal_data: vec4<f32>,
    @location({{ loc.next() }}) pulse: f32,
};

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
    var local_normal = vec3<f32>(0.0, 0.0, 1.0);

    $$ if HAS_NORMAL is defined
    local_normal = normalize(vec3<f32>(in.normal.xyz));
    $$ endif

    var arc = 0.0;
    var depth = 0.0;
    var edge = 1.0;
    var plume = 0.0;

    $$ if HAS_ARC_DATA is defined
    arc = in.arc_data.x;
    depth = in.arc_data.y;
    edge = in.arc_data.z;
    plume = in.arc_data.w;
    $$ else
        $$ if HAS_UV is defined
        arc = in.uv.x;
        depth = in.uv.y;
        edge = 1.0 - abs(in.uv.y * 2.0 - 1.0);
        $$ else
        depth = clamp(local_position.z * 0.5 + 0.5, 0.0, 1.0);
        $$ endif
    $$ endif

    var phase = arc * 1.7 + depth * 0.9;
    $$ if HAS_PHASE_OFFSET is defined
    phase = in.phase_offset;
    $$ endif

    let time = u_render_state.time * u_material.pulse_speed;
    let tangent = normalize(vec3<f32>(-local_position.y, local_position.x, 0.0) + vec3<f32>(1e-4, 0.0, 0.0));
    let ribbon_wave = sin(arc * 31.4159265359 + time * 4.2 + phase * 6.28318530718);
    let curtain_wave = sin(depth * 18.8495559215 - time * 5.0 + plume * 5.2);
    let twist_wave = cos(arc * 15.7079632679 - depth * 9.4247779608 + time * 2.4 + phase * 3.1);
    let lift_wave = sin(arc * 12.5663706144 + time * 1.8 + plume * 2.6);
    let edge_gain = 0.3 + edge * 0.7;

    local_position += local_normal * ribbon_wave * u_material.ribbon_amplitude * edge_gain;
    local_position += tangent * twist_wave * u_material.twist_strength * (0.25 + plume * 0.75);
    local_position.z += curtain_wave * u_material.depth_amplitude * edge_gain + lift_wave * 0.08;

    let surface_normal = normalize(
        local_normal + tangent * twist_wave * 0.25 + vec3<f32>(0.0, 0.0, curtain_wave * 0.22)
    );
    let world_pos = u_model.world_matrix * vec4<f32>(local_position, 1.0);

    $$ if IN_TRANSPARENT_PASS is defined
    out.position = u_render_state.unjittered_view_projection * world_pos;
    $$ else
    out.position = u_render_state.view_projection * world_pos;
    $$ endif

    out.world_position = world_pos.xyz / world_pos.w;
    out.normal = normalize(u_model.normal_matrix * surface_normal);

    $$ if HAS_UV is defined
    out.uv = in.uv;
    $$ endif

    out.portal_data = vec4<f32>(arc, depth, edge, plume);
    out.pulse = 0.5 + 0.5 * ribbon_wave;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let time = u_render_state.time * u_material.pulse_speed;
    let arc = in.portal_data.x;
    let depth = in.portal_data.y;
    let edge = in.portal_data.z;
    let plume = in.portal_data.w;

    let normal = normalize(in.normal);
    let view = normalize(u_render_state.camera_position - in.world_position);
    let fresnel = pow(1.0 - saturate(dot(normal, view)), u_material.fresnel_power);

    $$ if HAS_ARC_DATA is defined
    let lattice_uv = abs(fract(vec2<f32>(arc * 24.0 + time * 0.08, depth * 10.0 - time * 0.12)) - 0.5);
    $$ else
        $$ if HAS_UV is defined
        let lattice_uv = abs(fract(in.uv * vec2<f32>(9.0, 5.0) + vec2<f32>(time * 0.04, -time * 0.07)) - 0.5);
        $$ else
        let lattice_uv = abs(fract(in.world_position.xy * 0.8) - 0.5);
        $$ endif
    $$ endif

    let lattice = smoothstep(0.40, 0.50, max(lattice_uv.x, lattice_uv.y));
    let ribbon = 0.5 + 0.5 * sin(arc * 18.8495559215 - time * 6.0 + plume * 4.0);
    let veil = 0.5 + 0.5 * sin(depth * 22.0 - time * 4.8 + arc * 8.0);
    let shock = 1.0 - smoothstep(0.0, 0.18, abs(arc - fract(time * 0.09 + plume * 0.2)));
    let sparkle = hash12(
        floor(vec2<f32>(arc, depth) * vec2<f32>(64.0, 18.0)) + vec2<f32>(floor(time * 9.0), floor(plume * 7.0))
    );

    let core = u_material.base_color.rgb * (0.12 + ribbon * 0.32 + veil * 0.28 + in.pulse * 0.10);
    let accent = u_material.accent_color.rgb * (lattice * 0.42 + shock * 1.30 + sparkle * 0.12 + in.pulse * 0.24);
    let edge_glow = u_material.edge_color.rgb * fresnel * u_material.glow_intensity;

    let color = core + accent + edge_glow;
    let alpha = saturate(
        u_material.opacity * (0.12 + veil * 0.24 + edge * 0.28 + shock * 0.38 + fresnel * 0.70)
    );

    if (alpha < u_material.alpha_test) {
        discard;
    }

    return pack_fragment_output(vec4<f32>(color, alpha));
}
"#;

#[myth_material(
    shader = "custom_material_template_aurora_gate",
    shader_template_src = AURORA_GATE_TEMPLATE
)]
pub struct AuroraGateMaterial {
    #[uniform(default = "Vec4::new(0.08, 0.42, 1.08, 1.0)")]
    pub base_color: Vec4,

    #[uniform(default = "Vec4::new(0.16, 1.08, 0.86, 1.0)")]
    pub accent_color: Vec4,

    #[uniform(default = "Vec4::new(0.96, 1.12, 1.35, 1.0)")]
    pub edge_color: Vec4,

    #[uniform(default = "0.90")]
    pub opacity: f32,

    #[uniform(default = "0.02")]
    pub alpha_test: f32,

    #[uniform(default = "2.2")]
    pub pulse_speed: f32,

    #[uniform(default = "0.18")]
    pub ribbon_amplitude: f32,

    #[uniform(default = "0.22")]
    pub depth_amplitude: f32,

    #[uniform(default = "0.16")]
    pub twist_strength: f32,

    #[uniform(default = "4.8")]
    pub fresnel_power: f32,

    #[uniform(default = "2.1")]
    pub glow_intensity: f32,
}

fn build_portal_geometry(
    radius: f32,
    depth: f32,
    radial_segments: usize,
    depth_segments: usize,
) -> Geometry {
    let stride = radial_segments + 1;
    let mut positions = Vec::with_capacity((radial_segments + 1) * (depth_segments + 1));
    let mut normals = Vec::with_capacity((radial_segments + 1) * (depth_segments + 1));
    let mut uvs = Vec::with_capacity((radial_segments + 1) * (depth_segments + 1));
    let mut phase_offsets = Vec::with_capacity((radial_segments + 1) * (depth_segments + 1));
    let mut arc_data = Vec::with_capacity((radial_segments + 1) * (depth_segments + 1));
    let mut indices = Vec::with_capacity(radial_segments * depth_segments * 6);

    for depth_index in 0..=depth_segments {
        let v = depth_index as f32 / depth_segments as f32;
        let z = (v - 0.5) * depth;
        let edge = 1.0 - (v * 2.0 - 1.0).abs();

        for radial_index in 0..=radial_segments {
            let u = radial_index as f32 / radial_segments as f32;
            let angle = u * TAU;
            let dir = Vec2::new(angle.cos(), angle.sin());
            let radius_offset = (u * TAU * 6.0).sin() * 0.06 + (v * 14.0).cos() * 0.03;
            let ring_radius = radius + radius_offset;
            let plume = (radial_index % 9) as f32 / 8.0;

            positions.push([dir.x * ring_radius, dir.y * ring_radius, z]);
            normals.push([dir.x, dir.y, 0.0]);
            uvs.push([u, v]);
            phase_offsets.push(u * 2.3 + v * 1.7 + plume * 0.9);
            arc_data.push([u, v, edge, plume]);
        }
    }

    for depth_index in 0..depth_segments {
        for radial_index in 0..radial_segments {
            let i0 = (depth_index * stride + radial_index) as u32;
            let i1 = i0 + 1;
            let i2 = ((depth_index + 1) * stride + radial_index) as u32;
            let i3 = i2 + 1;

            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    let mut geometry = Geometry::new();
    geometry.set_attribute(
        "position",
        myth::Attribute::new_planar(&positions, myth::VertexFormat::Float32x3),
    );
    geometry.set_attribute(
        "normal",
        myth::Attribute::new_planar(&normals, myth::VertexFormat::Float32x3),
    );
    geometry.set_attribute(
        "uv",
        myth::Attribute::new_planar(&uvs, myth::VertexFormat::Float32x2),
    );

    // These names intentionally become `VertexInput.phase_offset`,
    // `VertexInput.arc_data`, and the automatic `HAS_PHASE_OFFSET` /
    // `HAS_ARC_DATA` geometry defines.
    geometry.set_attribute(
        "phase_offset",
        myth::Attribute::new_planar(&phase_offsets, myth::VertexFormat::Float32),
    );
    geometry.set_attribute(
        "arc_data",
        myth::Attribute::new_planar(&arc_data, myth::VertexFormat::Float32x4),
    );
    geometry.set_indices_u32(&indices);
    geometry
}

struct AuroraGateDemo {
    controls: OrbitControls,
    outer_gate: NodeHandle,
    inner_gate: NodeHandle,
    core_node: NodeHandle,
    // accent_light: NodeHandle,
    time: f32,
}

impl AppHandler for AuroraGateDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();
        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.03, 0.05, 0.11, 1.0),
            Vec4::new(0.002, 0.006, 0.015, 1.0),
        ));
        scene.environment.set_ambient_light(Vec3::splat(0.015));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.04);
        scene.bloom.set_radius(0.007);
        // scene.tone_mapping.set_exposure(1.20);

        let floor_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.045, 0.05, 0.07, 1.0))
                .with_roughness(0.18)
                .with_metalness(0.82),
        );
        let frame_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.12, 0.14, 0.18, 1.0))
                .with_roughness(0.34)
                .with_metalness(0.76),
        );

        let floor = scene.spawn_plane(20.0, 20.0, floor_material, &engine.assets);
        scene
            .edit_node(floor)
            .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
            .set_position(0.0, -0.08, 0.0)
            .set_receive_shadows(true);

        for &(x, y, z, sx, sy, sz) in &[
            (-4.0, 2.0, -0.8, 0.45, 4.2, 0.55),
            (4.0, 2.0, -0.8, 0.45, 4.2, 0.55),
            (0.0, 4.0, -0.8, 8.4, 0.24, 0.55),
            (0.0, 0.2, -1.1, 8.8, 0.24, 0.65),
        ] {
            let strut = scene.spawn_box(sx, sy, sz, frame_material, &engine.assets);
            scene
                .edit_node(strut)
                .set_position(x, y, z)
                .set_shadows(true, true);
        }

        // let key = scene.add_light(Light::new_directional(Vec3::new(0.92, 0.96, 1.0), 2.1));
        // scene
        //     .edit_node(key)
        //     .set_position(8.0, 10.0, 6.0)
        //     .look_at(Vec3::new(0.0, 2.0, 0.0));

        // let accent_light = scene.add_light(Light::new_point(Vec3::new(0.18, 0.95, 0.88), 2.8, 18.0));
        // scene.edit_node(accent_light).set_position(0.0, 2.6, 5.8);

        let portal_geometry = engine
            .assets
            .geometries
            .add(build_portal_geometry(2.7, 4.6, 192, 28));

        let outer_material = engine.assets.materials.add(Material::new_custom(
            AuroraGateMaterial::default()
                .with_ribbon_amplitude(0.20)
                .with_depth_amplitude(0.26)
                .with_twist_strength(0.18)
                .with_glow_intensity(2.3)
                .with_alpha_mode(AlphaMode::Blend)
                .with_depth_write(false)
                .with_side(Side::Double),
        ));
        let inner_material = engine.assets.materials.add(Material::new_custom(
            AuroraGateMaterial::default()
                .with_base_color(Vec4::new(0.30, 0.18, 1.10, 1.0))
                .with_accent_color(Vec4::new(1.02, 0.22, 0.92, 1.0))
                .with_edge_color(Vec4::new(1.15, 0.92, 1.35, 1.0))
                .with_pulse_speed(2.8)
                .with_ribbon_amplitude(0.14)
                .with_depth_amplitude(0.18)
                .with_twist_strength(0.12)
                .with_glow_intensity(1.8)
                .with_opacity(0.76)
                .with_alpha_mode(AlphaMode::Blend)
                .with_depth_write(false)
                .with_side(Side::Double),
        ));
        let core_material = engine.assets.materials.add(Material::new_custom(
            AuroraGateMaterial::default()
                .with_base_color(Vec4::new(0.10, 0.84, 0.66, 1.0))
                .with_accent_color(Vec4::new(0.86, 1.24, 1.08, 1.0))
                .with_edge_color(Vec4::new(0.98, 1.30, 1.18, 1.0))
                .with_pulse_speed(1.6)
                .with_ribbon_amplitude(0.08)
                .with_depth_amplitude(0.10)
                .with_twist_strength(0.05)
                .with_glow_intensity(2.7)
                .with_alpha_mode(AlphaMode::Blend)
                .with_depth_write(false)
                .with_side(Side::Double),
        ));

        let outer_gate = scene.add_mesh(Mesh::new(portal_geometry, outer_material));
        scene
            .edit_node(outer_gate)
            .set_position(0.0, 2.0, 0.0)
            .set_shadows(false, false);

        let inner_gate = scene.add_mesh(Mesh::new(portal_geometry, inner_material));
        scene
            .edit_node(inner_gate)
            .set_position(0.0, 2.0, -0.2)
            .set_scale_xyz(0.82, 0.82, 0.92)
            .set_shadows(false, false);

        let core_node = scene.spawn_sphere(0.96, core_material, &engine.assets);
        scene
            .edit_node(core_node)
            .set_position(0.0, 2.0, 0.0)
            .set_shadows(false, false);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(0.0, 2.4, 10.4)
            .look_at(Vec3::new(0.0, 2.0, 0.0));
        scene.active_camera = Some(camera);

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 2.4, 10.4), Vec3::new(0.0, 2.0, 0.0)),
            outer_gate,
            inner_gate,
            core_node,
            // accent_light,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        scene.edit_node(self.outer_gate).set_rotation(
            Quat::from_rotation_z(self.time * 0.28) * Quat::from_rotation_y(self.time * 0.10),
        );
        scene.edit_node(self.inner_gate).set_rotation(
            Quat::from_rotation_z(-self.time * 0.42)
                * Quat::from_rotation_x((self.time * 0.6).sin() * 0.08),
        );
        scene
            .edit_node(self.core_node)
            .set_position(0.0, 2.0 + (self.time * 1.8).sin() * 0.16, 0.0)
            .set_rotation(Quat::from_rotation_y(-self.time * 0.8));

        // if let Some(node) = scene.node_mut(self.accent_light) {
        //     node.transform.position = Vec3::new(
        //         self.time.cos() * 5.4,
        //         2.8 + (self.time * 1.6).sin() * 0.6,
        //         4.8 + self.time.sin() * 0.8,
        //     );
        // }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Custom Material - Template Geometry Aurora Gate")
        .run::<AuroraGateDemo>()
}
