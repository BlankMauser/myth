//! [gallery]
//! name = "GPU-Driven Particle Lights"
//! category = "Lighting & GI"
//! description = "A dual-track lighting demo: CPU directional + CPU accent lights + a GPU swarm merged inside the RDG local-light track."
//! instructions = "4096 GPU point lights + CPU accent lights\nPress Space to toggle GPU local-light injection\nDrag to orbit"
//! order = 414
//!

use std::f32::consts::FRAC_PI_2;

use bytemuck::{Pod, Zeroable};
use myth::prelude::*;
use myth::render::ClusteredShadingMode;
use myth::renderer::{
    core::gpu::Tracked,
    graph::{
        composer::GpuLightBuffers,
        core::{BufferDesc, ComputePassBuilder, TemplateComputePass},
    },
    pipeline::ShaderCompilationOptions,
};
use myth::resources::{
    image::ColorSpace,
    input::Key,
    uniforms::{GpuLightStorage, LightBufferMetadata, scene_lighting_structs_wgsl},
};
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

const GPU_LIGHT_COUNT: u32 = 4096;
const PARTICLE_LIGHT_WG_SIZE: u32 = 64;
const GPU_PARTICLE_LIGHT_SHADER_NAME: &str =
    "examples/gpu_driven_particle_lights/gpu_particle_light";

const GPU_PARTICLE_LIGHT_SHADER_TEMPLATE: &str = r#"
{{ scene_lighting_structs }}

struct SwarmLightParams {
    time: f32,
    radius: f32,
    height: f32,
    count: u32,
    orbit_speed: f32,
    swirl_speed: f32,
    base_range: f32,
    base_intensity: f32,
    ring_count: u32,
    band_count: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> u_params: SwarmLightParams;
@group(0) @binding(1) var<storage, read_write> st_light_metadata: LightBufferMetadata;
@group(0) @binding(2) var<storage, read_write> st_lights: array<Struct_lights>;
@group(0) @binding(3) var<storage, read_write> st_indirect_count: array<u32>;

fn hash11(value: f32) -> f32 {
    return fract(sin(value) * 43758.5453123);
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> vec3<f32> {
    let h6 = clamp(fract(hue) * 6.0, 0.0, 6.0 - 0.000001);
    let i = i32(floor(h6));
    let f = h6 - f32(i);
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - f * saturation);
    let t = value * (1.0 - (1.0 - f) * saturation);

    switch i {
        case 0: {
            return vec3<f32>(value, t, p);
        }
        case 1: {
            return vec3<f32>(q, value, p);
        }
        case 2: {
            return vec3<f32>(p, value, t);
        }
        case 3: {
            return vec3<f32>(p, q, value);
        }
        case 4: {
            return vec3<f32>(t, p, value);
        }
        default: {
            return vec3<f32>(value, p, q);
        }
    }
}

fn zero_matrix() -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
    );
}

@compute @workgroup_size({{ particle_light_workgroup_size }})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= u_params.count) {
        return;
    }

    if (index == 0u) {
        st_light_metadata.total_light_count = u_params.count;
        st_light_metadata.active_local_light_count = u_params.count;
        st_light_metadata.reserved_0 = 0u;
        st_light_metadata.reserved_1 = 0u;
        st_indirect_count[0] = u_params.count;
    }

    let ring_count = max(u_params.ring_count, 1u);
    let lane = f32(index % ring_count) / f32(ring_count);
    let band = f32(index / ring_count);
    let band_count = max(u_params.band_count, 1.0);
    let band_norm = band / band_count;
    let signed_band = band_norm * 2.0 - 1.0;

    let orbit_phase = lane * 6.28318530718 + u_params.time * (u_params.orbit_speed + signed_band * 0.35);
    let radial_wave = sin(band * 0.173 + u_params.time * u_params.swirl_speed) * 1.55;
    let ribbon_wave = cos(lane * 18.8495559215 + u_params.time * 1.7 + band * 0.11) * 0.6;
    let radius = u_params.radius + radial_wave;
    let position = vec3<f32>(
        cos(orbit_phase) * radius,
        1.9 + signed_band * u_params.height + ribbon_wave,
        sin(orbit_phase) * radius + cos(band * 0.07 + u_params.time * 0.6) * 2.8,
    );

    let hue = fract(lane + band_norm * 0.37 + u_params.time * 0.03 + hash11(f32(index) * 0.013));
    let color = hsv_to_rgb(hue, 0.78, 1.0);
    let pulse = 0.5 + 0.5 * sin(u_params.time * 2.1 + f32(index) * 0.071);
    let flicker = 0.82 + hash11(f32(index) * 1.37) * 0.36;

    var light: Struct_lights;
    light.color = color;
    light.intensity = u_params.base_intensity * (0.75 + pulse * 0.9) * flicker;
    light.position = position;
    light.range = u_params.base_range * (0.85 + pulse * 0.4);
    light.direction = normalize(vec3<f32>(-position.x * 0.15, -0.8 - ribbon_wave * 0.12, -position.z * 0.15));
    light.decay = 2.0;
    light.inner_cone_cos = 0.0;
    light.outer_cone_cos = 0.0;
    light.light_type = 1u;
    light.shadow_layer_index = -1;
    light.shadow_bias = 0.0;
    light.shadow_normal_bias = 0.0;
    light.cascade_count = 0u;
    light.point_shadow_index = -1;
    light.cascade_splits = vec4<f32>(0.0);
    light.shadow_matrices = array<mat4x4<f32>, 4>(
        zero_matrix(),
        zero_matrix(),
        zero_matrix(),
        zero_matrix(),
    );

    st_lights[index] = light;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SwarmLightParams {
    time: f32,
    radius: f32,
    height: f32,
    count: u32,
    orbit_speed: f32,
    swirl_speed: f32,
    base_range: f32,
    base_intensity: f32,
    ring_count: u32,
    band_count: f32,
    pad: [u32; 2],
}

struct GpuDrivenParticleLightsDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    centerpiece: NodeHandle,
    swarm_pass: TemplateComputePass,
    swarm_params: Tracked<wgpu::Buffer>,
    swarm_enabled: bool,
    time: f32,
}

fn centered_lattice(index: usize, count: usize, spacing: f32) -> f32 {
    (index as f32 - (count as f32 - 1.0) * 0.5) * spacing
}

impl AppHandler for GpuDrivenParticleLightsDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();
        // scene.environment.set_ambient_light(Vec3::splat(0.004));
        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.035, 0.02, 0.045, 1.0),
            Vec4::new(0.003, 0.005, 0.012, 1.0),
        ));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.04);
        scene.bloom.set_radius(0.008);
        scene.tone_mapping.set_exposure(1.22);

        let env_texture = engine.assets.load_cube_texture(
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
        scene.environment.set_env_map(Some(env_texture));

        let mut sun = Light::new_directional(Vec3::new(1.0, 0.96, 0.9), 1.9);
        sun.cast_shadows = true;
        if let Some(shadow) = sun.shadow.as_mut() {
            shadow.bias = 0.0008;
            shadow.normal_bias = 0.04;
            shadow.max_shadow_distance = 48.0;
        }
        let sun = scene.add_light(sun);
        scene
            .edit_node(sun)
            .set_position(9.0, 14.0, 8.0)
            .look_at(Vec3::new(0.0, 1.5, 0.0));

        for &(color, intensity, range, x, y, z) in &[
            (Vec3::new(1.0, 0.55, 0.2), 3.0, 2.2, -4.2, 1.3, -8.0),
            (Vec3::new(0.2, 0.75, 1.0), 3.0, 2.0, 4.2, 1.4, 7.0),
        ] {
            let local = scene.add_light(Light::new_point(color, intensity, range));
            scene.edit_node(local).set_position(x, y, z);
        }

        for &(color, intensity, range, x, y, z, target_x, target_y, target_z) in &[
            (
                Vec3::new(1.0, 0.72, 0.34),
                3.0,
                3.0,
                -6.4,
                3.6,
                -2.0,
                0.0,
                1.1,
                -6.0,
            ),
            (
                Vec3::new(0.36, 0.82, 1.0),
                3.0,
                3.0,
                6.4,
                3.6,
                2.0,
                0.0,
                1.1,
                6.0,
            ),
        ] {
            let local = scene.add_light(Light::new_spot(color, intensity, range, 0.24, 0.52));
            scene
                .edit_node(local)
                .set_position(x, y, z)
                .look_at(Vec3::new(target_x, target_y, target_z));
        }

        let floor_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.04, 0.045, 0.06, 1.0))
                .with_roughness(0.16)
                .with_metalness(0.78),
        );
        let wall_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.08, 0.09, 0.12, 1.0))
                .with_roughness(0.62)
                .with_metalness(0.22),
        );
        let frame_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.17, 0.18, 0.22, 1.0))
                .with_roughness(0.24)
                .with_metalness(0.72),
        );
        let hero_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.92, 0.93, 0.97, 1.0))
                .with_roughness(0.08)
                .with_metalness(1.0),
        );
        let accent_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.08, 0.1, 0.14, 1.0))
                .with_emissive(Vec3::new(0.25, 0.82, 1.0), 2.8)
                .with_roughness(0.2)
                .with_metalness(0.08),
        );

        let floor = scene.spawn_plane(24.0, 88.0, floor_material, &engine.assets);
        scene
            .edit_node(floor)
            .set_rotation(Quat::from_rotation_x(-FRAC_PI_2))
            .set_position(0.0, -0.18, 0.0)
            .set_receive_shadows(false);

        for &(x, y, z, sx, sy, sz) in &[
            (-7.4, 2.45, 0.0, 0.5, 5.6, 74.0),
            (7.4, 2.45, 0.0, 0.5, 5.6, 74.0),
            (0.0, 5.05, 0.0, 15.6, 0.24, 74.0),
        ] {
            let wall = scene.spawn_box(sx, sy, sz, wall_material, &engine.assets);
            scene
                .edit_node(wall)
                .set_position(x, y, z)
                .set_shadows(false, true);
        }

        for section in 0..14 {
            let z = centered_lattice(section, 14, 5.1);

            for x in [-5.5, 5.5] {
                let pillar = scene.spawn_box(0.55, 4.8, 0.55, frame_material, &engine.assets);
                scene
                    .edit_node(pillar)
                    .set_position(x, 2.08, z)
                    .set_shadows(true, true);
            }

            let beam = scene.spawn_box(11.8, 0.16, 0.28, accent_material, &engine.assets);
            scene
                .edit_node(beam)
                .set_position(0.0, 4.18, z)
                .set_shadows(false, false);

            let hero = if section % 2 == 0 {
                scene.spawn_sphere(0.74, hero_material, &engine.assets)
            } else {
                scene.spawn_box(1.25, 1.25, 1.25, hero_material, &engine.assets)
            };
            scene
                .edit_node(hero)
                .set_position(0.0, 0.72, z)
                .set_shadows(true, true);
        }

        let centerpiece = scene.spawn_torus(2.7, 0.32, hero_material, &engine.assets);
        scene
            .edit_node(centerpiece)
            .set_position(0.0, 2.15, 0.0)
            .set_rotation(Quat::from_rotation_x(FRAC_PI_2 * 0.35))
            .set_shadows(true, true);

        let core = scene.spawn_sphere(0.88, accent_material, &engine.assets);
        scene.attach(core, centerpiece);
        scene
            .edit_node(core)
            .set_position(0.0, 0.0, 0.0)
            .set_shadows(false, false);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .edit_node(camera)
            .set_position(0.0, 4.8, 17.5)
            .look_at(Vec3::new(0.0, 1.8, 0.0));
        scene.active_camera = Some(camera);

        let mut swarm_shader_options = ShaderCompilationOptions::default();
        swarm_shader_options.inject_code("scene_lighting_structs", scene_lighting_structs_wgsl());
        swarm_shader_options.inject_code(
            "particle_light_workgroup_size",
            format!("{}u", PARTICLE_LIGHT_WG_SIZE),
        );
        let swarm_pass = ComputePassBuilder::new("GPU Particle Light Pipeline")
            .inline_shader_template(
                GPU_PARTICLE_LIGHT_SHADER_NAME,
                GPU_PARTICLE_LIGHT_SHADER_TEMPLATE,
            )
            .shader_options(swarm_shader_options)
            .bind_uniform_buffer(0, 0, wgpu::ShaderStages::COMPUTE)
            .bind_storage_buffer(0, 1, wgpu::ShaderStages::COMPUTE, false)
            .bind_storage_buffer(0, 2, wgpu::ShaderStages::COMPUTE, false)
            .bind_storage_buffer(0, 3, wgpu::ShaderStages::COMPUTE, false)
            .build(&mut engine.renderer);
        let swarm_params = {
            let wgpu_ctx = engine
                .renderer
                .wgpu_ctx()
                .expect("renderer must be initialized before example setup");
            Tracked::new(wgpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Particle Light Params"),
                size: std::mem::size_of::<SwarmLightParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        };

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 4.8, 17.5), Vec3::new(0.0, 1.8, 0.0)),
            fps_counter: FpsCounter::new(),
            centerpiece,
            swarm_pass,
            swarm_params,
            swarm_enabled: true,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if engine.input.get_key_down(Key::Space) {
            self.swarm_enabled = !self.swarm_enabled;
        }

        if let Some(node) = scene.node_mut(self.centerpiece) {
            node.transform.rotation *= Quat::from_rotation_y(frame.dt * 0.55);
            node.transform.rotation *= Quat::from_rotation_x(frame.dt * 0.22);
        }

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "GPU Particle Lights | {} | CPU dir + CPU/GPU local | FPS: {:.1}",
                if self.swarm_enabled {
                    "Merged"
                } else {
                    "Injection Off"
                },
                fps
            ));
        }
    }

    fn render(&mut self, engine: &mut Engine, _window: &dyn Window) {
        let Some(wgpu_ctx) = engine.renderer.wgpu_ctx() else {
            return;
        };

        let params = SwarmLightParams {
            ring_count: 128,
            band_count: ((GPU_LIGHT_COUNT as f32) / 128.0).ceil().max(1.0),
            time: self.time,
            radius: 5.8,
            height: 3.6,
            count: GPU_LIGHT_COUNT,
            orbit_speed: 0.92,
            swirl_speed: 1.45,
            base_range: 0.25,
            base_intensity: 11.5,
            pad: [0; 2],
        };
        wgpu_ctx
            .queue
            .write_buffer(&self.swarm_params, 0, bytemuck::bytes_of(&params));

        let Some(composer) = engine.compose_frame() else {
            return;
        };

        let swarm_pass = &self.swarm_pass;
        let params_buffer = &self.swarm_params;
        let swarm_enabled = self.swarm_enabled;

        composer
            .inject_gpu_local_lights(move |ctx| {
                if !swarm_enabled {
                    return None;
                }

                Some(ctx.with_group("GPU_Particle_Lights", |ctx| {
                    ctx.graph
                        .add_pass("GPU_Particle_Light_Generate", |builder| {
                            let light_metadata = builder.create_buffer(
                                "GPU_Particle_Light_Metadata",
                                BufferDesc::new(
                                    std::mem::size_of::<LightBufferMetadata>() as u64,
                                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
                                ),
                            );
                            let light_storage = builder.create_buffer(
                                "GPU_Particle_Light_Storage",
                                BufferDesc::new(
                                    GPU_LIGHT_COUNT as u64
                                        * std::mem::size_of::<GpuLightStorage>() as u64,
                                    wgpu::BufferUsages::STORAGE,
                                ),
                            );
                            let indirect_count_buffer = builder.create_buffer(
                                "GPU_Particle_Light_Count",
                                BufferDesc::new(4, wgpu::BufferUsages::STORAGE),
                            );

                            let node = swarm_pass.build_node(
                                builder,
                                "GPU Particle Light Pass",
                                [GPU_LIGHT_COUNT.div_ceil(PARTICLE_LIGHT_WG_SIZE), 1, 1],
                                Some("GPU Particle Light BG"),
                                |bindings| {
                                    bindings.bind_tracked_buffer(0, 0, params_buffer);
                                    bindings.bind_buffer(0, 1, light_metadata);
                                    bindings.bind_buffer(0, 2, light_storage);
                                    bindings.bind_buffer(0, 3, indirect_count_buffer);
                                },
                            );

                            (
                                node,
                                GpuLightBuffers {
                                    light_metadata,
                                    light_storage,
                                    indirect_count_buffer: Some(indirect_count_buffer),
                                },
                            )
                        })
                }))
            })
            .render();
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("GPU-Driven Particle Lights")
        .with_settings(RendererSettings {
            path: RenderPath::HighFidelity,
            clustered_shading: ClusteredShadingMode::ForceOn,
            vsync: false,
            ..Default::default()
        })
        .run::<GpuDrivenParticleLightsDemo>()
}
