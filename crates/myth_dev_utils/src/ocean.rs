use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Vec3};
use myth_render::{
    HDR_TEXTURE_FORMAT, Renderer,
    core::gpu::{CommonSampler, Tracked},
    graph::core::{
        BufferDesc, BufferNodeId, ExecuteContext, GraphBlackboard, PassNode, PrepareContext,
        RenderGraph, RenderPassBuilder, RenderTargetOps, StandardFullscreenNode,
        TemplateFullscreenPass, TextureDesc, TextureNodeId,
    },
    pipeline::ShaderCompilationOptions,
};
use myth_scene::{LightKind, ProceduralSkyParams, ProjectionType, Scene};

const OCEAN_SHADER_NAME: &str = "myth_dev_utils/ocean_surface";
const OCEAN_RESOLVE_SHADER_NAME: &str = "myth_dev_utils/ocean_resolve";

const OCEAN_SHADER_TEMPLATE: &str = r#"
{$ include 'core/full_screen_vertex' $}
{$ include 'entry/utility/atmosphere/transmittance_math' $}

struct OceanUniforms {
    resolution_time: vec4<f32>,
    sea_height_base: vec4<f32>,
    sea_choppy_water: vec4<f32>,
    sea_speed_freq_output: vec4<f32>,
    camera_position: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_forward: vec4<f32>,
    camera_projection: vec4<f32>,
    light_direction: vec4<f32>,
    light_color_energy: vec4<f32>,
    ambient_color: vec4<f32>,
};

struct AtmosphereBakeParams {
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    moon_direction: vec3<f32>,
    moon_intensity: f32,
    star_axis: vec3<f32>,
    sun_disk_size: f32,
    moon_disk_size: f32,
    planet_radius: f32,
    atmosphere_radius: f32,
    star_intensity: f32,
    star_rotation: f32,
};

@group(0) @binding(0) var<uniform> u_ocean: OceanUniforms;
$$ if OCEAN_COMPOSITE is defined
@group(0) @binding(1) var t_scene_color: texture_2d<f32>;
@group(0) @binding(2) var s_scene_color: sampler;
@group(0) @binding(3) var t_scene_depth: texture_depth_2d;
$$ endif
@group(0) @binding(4) var t_environment_cube: texture_cube<f32>;
@group(0) @binding(5) var s_environment_cube: sampler;
@group(0) @binding(6) var t_environment_pmrem: texture_cube<f32>;
@group(1) @binding(0) var t_atmosphere_transmittance: texture_2d<f32>;
@group(1) @binding(1) var<uniform> u_atmosphere_bake_params: AtmosphereBakeParams;

const PI: f32 = 3.141592;
const NUM_STEPS: i32 = 8;
const ITER_GEOMETRY: i32 = 3;
const ITER_FRAGMENT: i32 = 5;

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    let non_negative = max(color, vec3<f32>(0.0));
    let low = non_negative / 12.92;
    let high = pow((non_negative + 0.055) / 1.055, vec3<f32>(2.4));
    return select(low, high, non_negative > vec3<f32>(0.04045));
}

fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(1.0, 113.0));
    return fract(sin(h) * 43758.5453123);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let mix1 = mix(hash(i + vec2<f32>(0.0, 0.0)), hash(i + vec2<f32>(1.0, 0.0)), u.x);
    let mix2 = mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x);
    let mix3 = mix(mix1, mix2, u.y);
    return -1.0 + 2.0 * mix3;
}

fn diffuse(n: vec3<f32>, l: vec3<f32>, p: f32) -> f32 {
    return pow(dot(n, l) * 0.4 + 0.6, p);
}

fn specular(n: vec3<f32>, l: vec3<f32>, e: vec3<f32>, s: f32) -> f32 {
    let nrm = (s + 8.0) / (PI * 8.0);
    return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;
}

fn get_sky_color(e_in: vec3<f32>) -> vec3<f32> {
    var e = e_in;
    e.y = (max(e.y, 0.0) * 0.8 + 0.2) * 0.8;
    return vec3<f32>(pow(1.0 - e.y, 2.0), 1.0 - e.y, 0.6 + (1.0 - e.y) * 0.4) * 1.1;
}

fn environment_max_mip_level() -> f32 {
    return max(u_ocean.ambient_color.w, 0.0);
}

fn use_scene_environment_reflections() -> bool {
    return u_ocean.light_direction.w > 0.5;
}

fn sample_environment_cube(direction: vec3<f32>) -> vec3<f32> {
    let dir = normalize(direction);
    return textureSampleLevel(
        t_environment_cube,
        s_environment_cube,
        vec3<f32>(-dir.x, dir.y, dir.z),
        0.0,
    ).rgb;
}

fn sample_environment_pmrem(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
    let max_mip = environment_max_mip_level();
    let dir = normalize(direction);
    let lod = clamp(roughness * max_mip, 0.0, max_mip);
    return textureSampleLevel(
        t_environment_pmrem,
        s_environment_cube,
        vec3<f32>(-dir.x, dir.y, dir.z),
        lod,
    ).rgb;
}

fn sample_atmosphere_transmittance_uv(trans_uv: vec2<f32>) -> vec3<f32> {
    let lut_size = vec2<i32>(textureDimensions(t_atmosphere_transmittance));
    let clamped_uv = clamp(trans_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let pixel = clamp(
        vec2<i32>(clamped_uv * vec2<f32>(lut_size)),
        vec2<i32>(0),
        lut_size - vec2<i32>(1),
    );
    return textureLoad(t_atmosphere_transmittance, pixel, 0).rgb;
}

fn sample_celestial_light_transmittance(
    world_position: vec3<f32>,
    direction_to_light: vec3<f32>,
) -> vec3<f32> {
    let planet_radius = u_atmosphere_bake_params.planet_radius;
    let atmosphere_radius = max(
        u_atmosphere_bake_params.atmosphere_radius,
        planet_radius + 1.0,
    );
    if (planet_radius <= 0.0) {
        return vec3<f32>(1.0);
    }

    let max_altitude = max(atmosphere_radius - planet_radius, 1.0);
    let planet_center = vec3<f32>(0.0, -planet_radius, 0.0);
    let altitude = clamp(
        length(world_position - planet_center) - planet_radius,
        0.0,
        max_altitude,
    );
    let trans_uv = transmittance_lut_uv(
        altitude,
        clamp(direction_to_light.y, -1.0, 1.0),
        planet_radius,
        atmosphere_radius,
    );
    return sample_atmosphere_transmittance_uv(trans_uv);
}

fn ray_direction_from_frag_coord(frag_coord: vec2<f32>) -> vec3<f32> {
    var screen = frag_coord / u_ocean.resolution_time.xy;
    screen = screen * 2.0 - 1.0;

    if (u_ocean.camera_projection.w > 0.5) {
        return normalize(u_ocean.camera_forward.xyz);
    }

    screen.x *= u_ocean.resolution_time.x / u_ocean.resolution_time.y;
    return normalize(
        u_ocean.camera_right.xyz * screen.x
            + u_ocean.camera_up.xyz * screen.y
            + u_ocean.camera_forward.xyz * u_ocean.camera_projection.x
    );
}

$$ if OCEAN_COMPOSITE is defined
fn sample_scene_color_uv(scene_uv: vec2<f32>) -> vec3<f32> {
    let scene_size = vec2<i32>(textureDimensions(t_scene_color));
    let clamped_uv = clamp(scene_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let pixel = clamp(
        vec2<i32>(clamped_uv * vec2<f32>(scene_size)),
        vec2<i32>(0),
        scene_size - vec2<i32>(1),
    );
    return textureLoad(t_scene_color, pixel, 0).rgb;
}

fn scene_uv_from_frag_coord(frag_coord: vec2<f32>) -> vec2<f32> {
    let uv = frag_coord / max(u_ocean.resolution_time.xy, vec2<f32>(1.0));
    return clamp(vec2<f32>(uv.x, 1.0 - uv.y), vec2<f32>(0.0), vec2<f32>(1.0));
}

fn sample_scene_color(frag_coord: vec2<f32>) -> vec3<f32> {
    return sample_scene_color_uv(scene_uv_from_frag_coord(frag_coord));
}

fn scene_uv_from_direction(direction: vec3<f32>) -> vec2<f32> {
    if (u_ocean.camera_projection.w > 0.5) {
        return vec2<f32>(-1.0, -1.0);
    }

    let local_x = dot(direction, u_ocean.camera_right.xyz);
    let local_y = dot(direction, u_ocean.camera_up.xyz);
    let local_z = dot(direction, u_ocean.camera_forward.xyz);
    if (local_z <= 0.0001) {
        return vec2<f32>(-1.0, -1.0);
    }

    let aspect = u_ocean.resolution_time.x / max(u_ocean.resolution_time.y, 1.0);
    let focal = max(u_ocean.camera_projection.x, 0.0001);
    let ndc = vec2<f32>(
        (focal * local_x / local_z) / max(aspect, 0.0001),
        focal * local_y / local_z,
    );
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn sample_scene_horizon_color(direction: vec3<f32>, fallback_coord: vec2<f32>) -> vec3<f32> {
    let horizon_dir = normalize(vec3<f32>(direction.x, max(direction.y, 0.02), direction.z));
    let scene_uv = scene_uv_from_direction(horizon_dir);
    if (
        scene_uv.x < 0.0 || scene_uv.x > 1.0 || scene_uv.y < 0.0 || scene_uv.y > 1.0
    ) {
        return sample_scene_color(fallback_coord);
    }

    return sample_scene_color_uv(scene_uv);
}

fn sample_scene_reflection(reflected_dir: vec3<f32>, fallback_coord: vec2<f32>) -> vec3<f32> {
    if (reflected_dir.y <= 0.0) {
        return sample_scene_color(fallback_coord);
    }

    let scene_uv = scene_uv_from_direction(normalize(reflected_dir));
    if (
        scene_uv.x < 0.0 || scene_uv.x > 1.0 || scene_uv.y < 0.0 || scene_uv.y > 1.0
    ) {
        return sample_scene_color(fallback_coord);
    }

    return sample_scene_color_uv(scene_uv);
}

fn sample_ocean_reflection(
    reflected_dir: vec3<f32>,
    roughness: f32,
    fallback_coord: vec2<f32>,
) -> vec3<f32> {
    if (use_scene_environment_reflections()) {
        let sharp = sample_environment_cube(reflected_dir);
        let prefiltered = sample_environment_pmrem(reflected_dir, roughness);
        return mix(prefiltered, sharp, 1.0 - roughness);
    }

    return sample_scene_reflection(reflected_dir, fallback_coord);
}
$$ endif

fn sample_fallback_reflection(reflected_dir: vec3<f32>, fallback_coord: vec2<f32>) -> vec3<f32> {
    if (use_scene_environment_reflections()) {
        let sharp = sample_environment_cube(reflected_dir);
        let prefiltered = sample_environment_pmrem(reflected_dir, 0.18);
        return mix(prefiltered, sharp, 0.82);
    }

    $$ if OCEAN_COMPOSITE is defined
    return sample_scene_reflection(reflected_dir, fallback_coord);
    $$ else
    return get_sky_color(reflected_dir);
    $$ endif
}

fn sea_octave(uv_in: vec2<f32>, choppy: f32) -> f32 {
    let uv = uv_in + vec2<f32>(noise(uv_in));
    var wv = 1.0 - abs(sin(uv));
    let swv = abs(cos(uv));
    wv = mix(wv, swv, wv);
    return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

fn map_internal(p: vec3<f32>, iterations: i32) -> f32 {
    var freq = u_ocean.sea_speed_freq_output.y;
    var amp = u_ocean.sea_height_base.x;
    var choppy = u_ocean.sea_choppy_water.x;
    let sea_time = 1.0 + u_ocean.resolution_time.z * u_ocean.sea_speed_freq_output.x;
    var uv = p.xz;
    uv.x *= 0.75;
    var height = 0.0;

    for (var i = 0; i < iterations; i += 1) {
        var d = sea_octave((uv + vec2<f32>(sea_time)) * freq, choppy);
        d += sea_octave((uv - vec2<f32>(sea_time)) * freq, choppy);
        height += d * amp;
        uv = vec2<f32>(uv.x * 1.6 + uv.y * 1.2, -uv.x * 1.2 + uv.y * 1.6);
        freq *= 1.9;
        amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }

    return p.y - height;
}

fn map(p: vec3<f32>) -> f32 {
    return map_internal(p, ITER_GEOMETRY);
}

fn map_detailed(p: vec3<f32>) -> f32 {
    return map_internal(p, ITER_FRAGMENT);
}

fn get_sea_color(
    p: vec3<f32>,
    n: vec3<f32>,
    l: vec3<f32>,
    eye: vec3<f32>,
    dist: vec3<f32>,
    frag_coord: vec2<f32>,
) -> vec3<f32> {
    var fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);
    fresnel = pow(fresnel, 3.0) * 0.5;
    let base_direct_light_color = max(u_ocean.light_color_energy.xyz, vec3<f32>(0.0));
    let base_direct_light_energy = clamp(u_ocean.light_color_energy.w, 0.0, 1.0);
    let direct_light_transmittance = sample_celestial_light_transmittance(p, l);
    let direct_light_color = base_direct_light_color * direct_light_transmittance;
    let direct_light_energy = clamp(
        base_direct_light_energy
            * dot(direct_light_transmittance, vec3<f32>(0.2126, 0.7152, 0.0722)),
        0.0,
        1.0,
    );
    let ambient_color = max(u_ocean.ambient_color.xyz, vec3<f32>(0.0));
    let roughness = clamp(0.08 + u_ocean.sea_choppy_water.x * 0.075, 0.08, 0.55);
    let reflected_dir = reflect(eye, n);
    let atten = max(1.0 - dot(dist, dist) * 0.001, 0.0);

    $$ if OCEAN_COMPOSITE is defined
    if (use_scene_environment_reflections()) {
        let ambient_strength = clamp(
            max(max(ambient_color.x, ambient_color.y), ambient_color.z) * 8.0,
            0.0,
            1.0,
        );
        let daylight_boost = 0.82 + 0.28 * direct_light_energy + 0.08 * ambient_strength;
        let ambient_environment = sample_environment_pmrem(n, 1.0);
        let reflected = sample_ocean_reflection(reflected_dir, roughness, frag_coord);
        let refracted = u_ocean.sea_height_base.yzw
            + ambient_environment * u_ocean.sea_choppy_water.yzw
                * (0.16 + 0.26 * ambient_strength + 0.22 * direct_light_energy)
            + diffuse(n, l, 80.0) * u_ocean.sea_choppy_water.yzw * direct_light_color
                * (0.12 + 0.14 * direct_light_energy);
        var color = mix(refracted, reflected, fresnel);
        color += u_ocean.sea_choppy_water.yzw
            * (p.y - u_ocean.sea_height_base.x)
            * 0.18
            * atten
            * (0.06 + 0.45 * ambient_strength + 0.32 * direct_light_energy);
        color += direct_light_color * (specular(n, l, eye, 60.0) * (0.03 + 0.97 * direct_light_energy));
        return color * daylight_boost;
    }
    $$ endif

    let reflected = get_sky_color(reflected_dir);
    let refracted = u_ocean.sea_height_base.yzw
        + diffuse(n, l, 80.0) * u_ocean.sea_choppy_water.yzw * 0.12;
    var color = mix(refracted, reflected, fresnel);
    color += u_ocean.sea_choppy_water.yzw * (p.y - u_ocean.sea_height_base.x) * 0.18 * atten;
    color += direct_light_color * (specular(n, l, eye, 60.0) * max(direct_light_energy, 0.2));
    return color;
}

fn get_normal(p: vec3<f32>, eps: f32) -> vec3<f32> {
    var n: vec3<f32>;
    n.y = map_detailed(p);
    n.x = map_detailed(vec3<f32>(p.x + eps, p.y, p.z)) - n.y;
    n.z = map_detailed(vec3<f32>(p.x, p.y, p.z + eps)) - n.y;
    n.y = eps;
    return normalize(n);
}

struct OceanHit {
    position: vec3<f32>,
    hit: bool,
};

fn trace_distance_limit(ori: vec3<f32>, dir: vec3<f32>) -> f32 {
    if (dir.y >= -0.0001) {
        return 4000.0;
    }

    let sea_band_height = max(u_ocean.sea_height_base.x * 6.0, 1.0);
    let plane_distance = max((ori.y + sea_band_height) / -dir.y, 1.0);
    return clamp(plane_distance, 200.0, 20000.0);
}

fn height_map_tracing(ori: vec3<f32>, dir: vec3<f32>) -> OceanHit {
    var tm = 0.0;
    var tx = trace_distance_limit(ori, dir);
    var hx = map(ori + dir * tx);
    var p = ori + dir * tx;

    if (hx > 0.0) {
        return OceanHit(p, false);
    }

    var hm = map(ori + dir * tm);
    var tmid = 0.0;

    for (var i = 0; i < NUM_STEPS; i += 1) {
        tmid = mix(tm, tx, hm / (hm - hx));
        p = ori + dir * tmid;
        let hmid = map(p);
        if (hmid < 0.0) {
            tx = tmid;
            hx = hmid;
        } else {
            tm = tmid;
            hm = hmid;
        }
    }

    return OceanHit(p, true);
}

fn get_pixel(frag_coord: vec2<f32>) -> vec3<f32> {
    let projection_mode = u_ocean.camera_projection.w;
    var ori = u_ocean.camera_position.xyz;
    let mut_dir = ray_direction_from_frag_coord(frag_coord);
    var dir = mut_dir;

    var screen = frag_coord / u_ocean.resolution_time.xy;
    screen = screen * 2.0 - 1.0;

    if (projection_mode > 0.5) {
        ori += u_ocean.camera_right.xyz * screen.x * u_ocean.camera_projection.y;
        ori += u_ocean.camera_up.xyz * screen.y * u_ocean.camera_projection.z;
    }

    if (dir.y >= 0.0) {
        $$ if OCEAN_COMPOSITE is defined
        return sample_scene_color(frag_coord);
        $$ else
        return get_sky_color(dir);
        $$ endif
    }

    let hit = height_map_tracing(ori, dir);
    if (!hit.hit) {
        $$ if OCEAN_COMPOSITE is defined
        return sample_scene_color(frag_coord);
        $$ else
        return get_sky_color(dir);
        $$ endif
    }

    let p = hit.position;
    let dist = p - ori;
    let distance_to_hit = length(dist);
    let normal_eps = clamp(length(dist) * (0.002 / u_ocean.resolution_time.y), 0.0008, 0.08);
    let n = get_normal(p, normal_eps);
    let light = normalize(u_ocean.light_direction.xyz);

    $$ if OCEAN_COMPOSITE is defined
    let horizon_color = sample_scene_horizon_color(dir, frag_coord);
    // let horizon_angle = smoothstep(0.025, -0.12, dir.y);
    // let horizon_distance = smoothstep(250.0, 6000.0, distance_to_hit);
    // let horizon_sea_mix = mix(pow(horizon_angle, 0.85), pow(horizon_angle, 1.75), horizon_distance);
    $$ else
    let horizon_color = get_sky_color(dir);
    $$ endif
    let horizon_angle = smoothstep(0.0, -0.02, dir.y);
    let horizon_sea_mix = pow(horizon_angle, 0.2);

    return mix(
        horizon_color,
        get_sea_color(p, n, light, dir, dist, frag_coord),
        horizon_sea_mix,
    );
}

fn render_ocean(uv: vec2<f32>) -> vec3<f32> {
    let frag_coord = vec2<f32>(uv.x, 1.0 - uv.y) * u_ocean.resolution_time.xy;

    $$ if OCEAN_QUALITY_LOW is defined

    return get_pixel(frag_coord);

    $$ elif OCEAN_QUALITY_MEDIUM is defined

    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-0.25, -0.25),
        vec2<f32>(0.25, -0.25),
        vec2<f32>(-0.25, 0.25),
        vec2<f32>(0.25, 0.25),
    );
    var medium_color = vec3<f32>(0.0);
    for (var i = 0; i < 4; i += 1) {
        medium_color += get_pixel(frag_coord + offsets[i]);
    }
    return medium_color / 4.0;

    $$ else

    var color = vec3<f32>(0.0);
    for (var i = -1; i <= 1; i += 1) {
        for (var j = -1; j <= 1; j += 1) {
            let jittered = frag_coord + vec2<f32>(f32(i), f32(j)) / 3.0;
            color += get_pixel(jittered);
        }
    }

    return color / 9.0;

    $$ endif
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    $$ if OCEAN_COMPOSITE is defined
    let size = vec2<i32>(textureDimensions(t_scene_depth));
    let pixel = clamp(vec2<i32>(in.uv * vec2<f32>(size)), vec2<i32>(0), size - vec2<i32>(1));
    let scene_depth = textureLoad(t_scene_depth, pixel, 0);
    if (scene_depth > 0.000001) {
        return vec4<f32>(textureLoad(t_scene_color, pixel, 0).rgb, 1.0);
    }

    let frag_coord = vec2<f32>(in.uv.x, 1.0 - in.uv.y) * u_ocean.resolution_time.xy;
    let dir = ray_direction_from_frag_coord(frag_coord);
    if (dir.y >= 0.0) {
        return vec4<f32>(sample_scene_color(frag_coord), 1.0);
    }
    $$ endif

    let raw_color = max(render_ocean(in.uv), vec3<f32>(0.0));

    let display_color = pow(raw_color, vec3<f32>(u_ocean.sea_speed_freq_output.z));
    let ocean = srgb_to_linear(display_color);

    return vec4<f32>(ocean, 1.0);
}
"#;

const OCEAN_RESOLVE_SHADER_TEMPLATE: &str = r#"
{$ include 'core/full_screen_vertex' $}

@group(0) @binding(0) var t_ocean: texture_2d<f32>;
@group(0) @binding(1) var s_ocean: sampler;
$$ if OCEAN_COMPOSITE_RESOLVE is defined
@group(0) @binding(2) var t_scene_color: texture_2d<f32>;
@group(0) @binding(3) var s_scene_color: sampler;
@group(0) @binding(4) var t_scene_depth: texture_depth_2d;
$$ endif

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ocean = textureSample(t_ocean, s_ocean, in.uv).rgb;

    $$ if OCEAN_COMPOSITE_RESOLVE is defined
    let size = vec2<i32>(textureDimensions(t_scene_depth));
    let pixel = clamp(vec2<i32>(in.uv * vec2<f32>(size)), vec2<i32>(0), size - vec2<i32>(1));
    let scene_depth = textureLoad(t_scene_depth, pixel, 0);
    if (scene_depth > 0.000001) {
        let scene_color = textureLoad(t_scene_color, pixel, 0).rgb;
        return vec4<f32>(scene_color, 1.0);
    }
    $$ endif

    return vec4<f32>(ocean, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OceanUniforms {
    resolution_time: [f32; 4],
    sea_height_base: [f32; 4],
    sea_choppy_water: [f32; 4],
    sea_speed_freq_output: [f32; 4],
    camera_position: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    camera_forward: [f32; 4],
    camera_projection: [f32; 4],
    light_direction: [f32; 4],
    light_color_energy: [f32; 4],
    ambient_color: [f32; 4],
}

struct OceanSurfaceNode<'a> {
    inner: StandardFullscreenNode<'a>,
    uniforms: BufferNodeId,
    uniform_snapshot: OceanUniforms,
}

impl<'a> PassNode<'a> for OceanSurfaceNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.uniforms),
            0,
            bytemuck::bytes_of(&self.uniform_snapshot),
        );
        self.inner.prepare(ctx);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        self.inner.execute(ctx, encoder);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OceanPreset {
    Reference,
    Cinematic,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OceanQuality {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OceanCameraSource {
    Reference,
    SceneMainCamera,
}

impl OceanCameraSource {
    #[must_use]
    fn label(self) -> &'static str {
        match self {
            Self::Reference => "Reference",
            Self::SceneMainCamera => "Scene Camera",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OceanLightSource {
    Manual,
    ProceduralSky,
    SceneDirectional,
    SceneAuto,
}

impl OceanLightSource {
    #[must_use]
    fn label(self) -> &'static str {
        match self {
            Self::Manual => "Manual",
            Self::ProceduralSky => "Procedural Sky",
            Self::SceneDirectional => "Directional Light",
            Self::SceneAuto => "Scene Auto",
        }
    }
}

impl OceanQuality {
    #[must_use]
    fn label(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }

    fn apply_to_shader_options(self, options: &mut ShaderCompilationOptions) {
        match self {
            Self::Low => options.add_define("OCEAN_QUALITY_LOW", "1"),
            Self::Medium => options.add_define("OCEAN_QUALITY_MEDIUM", "1"),
            Self::High => options.add_define("OCEAN_QUALITY_HIGH", "1"),
        }
    }
}

struct OceanQualityPasses {
    low: TemplateFullscreenPass,
    medium: TemplateFullscreenPass,
    high: TemplateFullscreenPass,
}

impl OceanQualityPasses {
    fn get(&self, quality: OceanQuality) -> &TemplateFullscreenPass {
        match quality {
            OceanQuality::Low => &self.low,
            OceanQuality::Medium => &self.medium,
            OceanQuality::High => &self.high,
        }
    }
}

#[derive(Clone, Copy)]
struct OceanCameraState {
    position: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
    projection: [f32; 4],
}

impl OceanCameraState {
    fn reference(time: f32) -> Self {
        Self {
            position: [0.0, 3.5, time * 1.5],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, -1.0],
            projection: [2.0, 0.0, 0.0, 0.0],
        }
    }
}

#[derive(Clone, Copy)]
struct OceanLightingState {
    direction: [f32; 3],
    color: [f32; 3],
    energy: f32,
    ambient: [f32; 3],
}

#[derive(Clone)]
pub struct OceanSettings {
    pub sea_height: f32,
    pub sea_choppy: f32,
    pub sea_speed: f32,
    pub sea_frequency: f32,
    pub output_curve: f32,
    pub sea_base: [f32; 3],
    pub sea_water_color: [f32; 3],
    pub sun_direction: [f32; 3],
}

impl OceanSettings {
    #[must_use]
    pub fn reference() -> Self {
        Self {
            sea_height: 0.60,
            sea_choppy: 4.0,
            sea_speed: 0.80,
            sea_frequency: 0.16,
            output_curve: 0.65,
            sea_base: [0.0, 0.09, 0.18],
            sea_water_color: [0.48, 0.54, 0.36],
            sun_direction: [0.0, 1.0, 0.8],
        }
    }

    #[must_use]
    pub fn cinematic() -> Self {
        Self {
            sea_height: 0.72,
            sea_choppy: 4.8,
            sea_speed: 0.72,
            sea_frequency: 0.14,
            output_curve: 0.65,
            sea_base: [0.02, 0.07, 0.15],
            sea_water_color: [0.44, 0.52, 0.34],
            sun_direction: [0.0, 1.0, 0.8],
        }
    }

    fn to_uniforms(
        &self,
        width: u32,
        height: u32,
        time: f32,
        camera: OceanCameraState,
        lighting: OceanLightingState,
        environment_max_mip_level: f32,
        environment_reflections_enabled: f32,
    ) -> OceanUniforms {
        OceanUniforms {
            resolution_time: [width as f32, height as f32, time, 0.0],
            sea_height_base: [
                self.sea_height,
                self.sea_base[0],
                self.sea_base[1],
                self.sea_base[2],
            ],
            sea_choppy_water: [
                self.sea_choppy,
                self.sea_water_color[0],
                self.sea_water_color[1],
                self.sea_water_color[2],
            ],
            sea_speed_freq_output: [self.sea_speed, self.sea_frequency, self.output_curve, 0.0],
            camera_position: [
                camera.position[0],
                camera.position[1],
                camera.position[2],
                0.0,
            ],
            camera_right: [camera.right[0], camera.right[1], camera.right[2], 0.0],
            camera_up: [camera.up[0], camera.up[1], camera.up[2], 0.0],
            camera_forward: [camera.forward[0], camera.forward[1], camera.forward[2], 0.0],
            camera_projection: camera.projection,
            light_direction: [
                lighting.direction[0],
                lighting.direction[1],
                lighting.direction[2],
                environment_reflections_enabled,
            ],
            light_color_energy: [
                lighting.color[0],
                lighting.color[1],
                lighting.color[2],
                lighting.energy,
            ],
            ambient_color: [
                lighting.ambient[0],
                lighting.ambient[1],
                lighting.ambient[2],
                environment_max_mip_level,
            ],
        }
    }
}

pub struct OceanRenderer {
    surface_passes: OceanQualityPasses,
    composite_passes: OceanQualityPasses,
    resolve_pass: TemplateFullscreenPass,
    composite_resolve_pass: TemplateFullscreenPass,
    uniform_snapshot: OceanUniforms,
    pub settings: OceanSettings,
    time: f32,
    preset: OceanPreset,
    quality: OceanQuality,
    render_scale: f32,
    camera_source: OceanCameraSource,
    light_source: OceanLightSource,
    fallback_cube_view: Tracked<wgpu::TextureView>,
    fallback_atmosphere_transmittance_view: Tracked<wgpu::TextureView>,
    fallback_atmosphere_bake_params: Tracked<wgpu::Buffer>,
}

impl OceanRenderer {
    #[must_use]
    pub fn new(renderer: &mut Renderer) -> Self {
        let surface_passes = Self::build_quality_passes(renderer, false);
        let composite_passes = Self::build_quality_passes(renderer, true);

        let resolve_pass = RenderPassBuilder::fullscreen("Ocean Resolve Pipeline")
            .inline_shader_template(OCEAN_RESOLVE_SHADER_NAME, OCEAN_RESOLVE_SHADER_TEMPLATE)
            .bind_texture_2d(0, 0, wgpu::ShaderStages::FRAGMENT, true)
            .bind_sampler(
                0,
                1,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            )
            .color_target(wgpu::ColorTargetState {
                format: HDR_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })
            .build(renderer);

        let mut composite_resolve_shader_options = ShaderCompilationOptions::default();
        composite_resolve_shader_options.add_define("OCEAN_COMPOSITE_RESOLVE", "1");
        let composite_resolve_pass =
            RenderPassBuilder::fullscreen("Ocean Composite Resolve Pipeline")
                .inline_shader_template(OCEAN_RESOLVE_SHADER_NAME, OCEAN_RESOLVE_SHADER_TEMPLATE)
                .shader_options(composite_resolve_shader_options)
                .bind_texture_2d(0, 0, wgpu::ShaderStages::FRAGMENT, true)
                .bind_sampler(
                    0,
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                )
                .bind_texture_2d(0, 2, wgpu::ShaderStages::FRAGMENT, true)
                .bind_sampler(
                    0,
                    3,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                )
                .bind_depth_texture_2d(0, 4, wgpu::ShaderStages::FRAGMENT)
                .color_target(wgpu::ColorTargetState {
                    format: HDR_TEXTURE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })
                .build(renderer);

        renderer
            .wgpu_ctx()
            .expect("renderer must be initialized before ocean helper setup");
        let fallback_cube_view = renderer
            .resource_manager()
            .expect("renderer must expose resource manager before ocean helper setup")
            .system_textures
            .black_cube
            .clone();
        let fallback_atmosphere_transmittance_view = renderer
            .resource_manager()
            .expect("renderer must expose resource manager before ocean helper setup")
            .system_textures
            .white_2d
            .clone();
        let fallback_atmosphere_bake_params = renderer
            .resource_manager()
            .expect("renderer must expose resource manager before ocean helper setup")
            .system_textures
            .atmosphere_bake_params
            .clone();
        Self {
            surface_passes,
            composite_passes,
            resolve_pass,
            composite_resolve_pass,
            uniform_snapshot: OceanUniforms::zeroed(),
            settings: OceanSettings::reference(),
            time: 0.0,
            preset: OceanPreset::Reference,
            quality: OceanQuality::High,
            render_scale: 1.0,
            camera_source: OceanCameraSource::SceneMainCamera,
            light_source: OceanLightSource::SceneAuto,
            fallback_cube_view,
            fallback_atmosphere_transmittance_view,
            fallback_atmosphere_bake_params,
        }
    }

    #[must_use]
    pub fn preset(&self) -> OceanPreset {
        self.preset
    }

    pub fn apply_preset(&mut self, preset: OceanPreset) {
        self.settings = match preset {
            OceanPreset::Reference => OceanSettings::reference(),
            OceanPreset::Cinematic => OceanSettings::cinematic(),
        };
        self.preset = preset;
    }

    #[must_use]
    pub fn quality(&self) -> OceanQuality {
        self.quality
    }

    pub fn set_quality(&mut self, quality: OceanQuality) {
        self.quality = quality;
    }

    #[must_use]
    pub fn camera_source(&self) -> OceanCameraSource {
        self.camera_source
    }

    #[must_use]
    pub fn reference_camera_view(&self) -> (Vec3, Vec3, f32) {
        let camera = OceanCameraState::reference(self.time);
        let position = Vec3::from_array(camera.position);
        let forward = Vec3::from_array(camera.forward).normalize_or_zero();
        let focal = camera.projection[0].max(0.0001);
        let fov_radians = 2.0 * (1.0 / focal).atan();
        (position, position + forward, fov_radians)
    }

    pub fn set_camera_source(&mut self, camera_source: OceanCameraSource) {
        self.camera_source = camera_source;
    }

    #[must_use]
    pub fn light_source(&self) -> OceanLightSource {
        self.light_source
    }

    pub fn set_light_source(&mut self, light_source: OceanLightSource) {
        self.light_source = light_source;
    }

    #[must_use]
    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    pub fn set_render_scale(&mut self, render_scale: f32) {
        self.render_scale = render_scale.clamp(0.5, 1.0);
    }

    pub fn advance_time(&mut self, dt: f32) {
        self.time += dt;
    }

    pub fn sync_gpu(&mut self, renderer: &Renderer, width: u32, height: u32) {
        let camera = OceanCameraState::reference(self.time);
        let lighting = self.reference_lighting_state();
        self.sync_gpu_with_state(renderer, width, height, camera, lighting);
    }

    pub fn sync_gpu_with_scene(
        &mut self,
        renderer: &Renderer,
        scene: &mut Scene,
        width: u32,
        height: u32,
    ) {
        let camera = self.resolve_camera_state(scene);
        let lighting = self.resolve_lighting_state(scene);
        self.sync_gpu_with_state(renderer, width, height, camera, lighting);
    }

    fn sync_gpu_with_state(
        &mut self,
        renderer: &Renderer,
        width: u32,
        height: u32,
        camera: OceanCameraState,
        lighting: OceanLightingState,
    ) {
        if renderer.wgpu_ctx().is_none() {
            return;
        }

        let (render_width, render_height) = self.render_dimensions(width, height);
        // Exact environment availability and mip metadata are graph-build
        // outputs. Graph construction patches these two fields into a
        // per-node snapshot that is uploaded only by live-node prepare.
        let environment_max_mip_level = 0.0;
        let environment_reflections_enabled = 0.0;
        let uniforms = self.settings.to_uniforms(
            render_width,
            render_height,
            self.time,
            camera,
            lighting,
            environment_max_mip_level,
            environment_reflections_enabled,
        );
        self.uniform_snapshot = uniforms;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Preset");
            if ui
                .selectable_label(self.preset == OceanPreset::Reference, "Reference")
                .clicked()
            {
                self.apply_preset(OceanPreset::Reference);
            }
            if ui
                .selectable_label(self.preset == OceanPreset::Cinematic, "Cinematic")
                .clicked()
            {
                self.apply_preset(OceanPreset::Cinematic);
            }
        });
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Quality");
            for quality in [OceanQuality::Low, OceanQuality::Medium, OceanQuality::High] {
                if ui
                    .selectable_label(self.quality == quality, quality.label())
                    .clicked()
                {
                    self.quality = quality;
                }
            }
        });

        ui.horizontal(|ui| {
            ui.label("Camera");
            for source in [
                OceanCameraSource::Reference,
                OceanCameraSource::SceneMainCamera,
            ] {
                if ui
                    .selectable_label(self.camera_source == source, source.label())
                    .clicked()
                {
                    self.camera_source = source;
                }
            }
        });

        ui.horizontal(|ui| {
            ui.label("Light");
            for source in [
                OceanLightSource::SceneAuto,
                OceanLightSource::Manual,
                OceanLightSource::ProceduralSky,
                OceanLightSource::SceneDirectional,
            ] {
                if ui
                    .selectable_label(self.light_source == source, source.label())
                    .clicked()
                {
                    self.light_source = source;
                }
            }
        });

        let mut render_scale = self.render_scale;
        if ui
            .add(
                egui::Slider::new(&mut render_scale, 0.5..=1.0)
                    .step_by(0.05)
                    .text("Render Scale"),
            )
            .changed()
        {
            self.set_render_scale(render_scale);
        }

        ui.separator();

        ui.add(egui::Slider::new(&mut self.settings.sea_height, 0.08..=1.6).text("Height"));
        ui.add(egui::Slider::new(&mut self.settings.sea_choppy, 0.8..=6.4).text("Choppiness"));
        ui.add(egui::Slider::new(&mut self.settings.sea_speed, 0.1..=2.6).text("Speed"));
        ui.add(egui::Slider::new(&mut self.settings.sea_frequency, 0.04..=0.35).text("Frequency"));
        ui.add(
            egui::Slider::new(&mut self.settings.output_curve, 0.45..=0.85).text("Output Curve"),
        );

        ui.horizontal(|ui| {
            ui.label("Sea Base");
            ui.color_edit_button_rgb(&mut self.settings.sea_base);
        });
        ui.horizontal(|ui| {
            ui.label("Water");
            ui.color_edit_button_rgb(&mut self.settings.sea_water_color);
        });

        if self.light_source == OceanLightSource::Manual {
            let (mut azimuth, mut elevation) =
                Self::direction_to_angles(self.settings.sun_direction);
            let azimuth_changed = ui
                .add(egui::Slider::new(&mut azimuth, -180.0..=180.0).text("Sun Azimuth"))
                .changed();
            let elevation_changed = ui
                .add(egui::Slider::new(&mut elevation, -85.0..=85.0).text("Sun Elevation"))
                .changed();
            if azimuth_changed || elevation_changed {
                self.settings.sun_direction = Self::angles_to_direction(azimuth, elevation);
            }
        }
    }

    fn output_desc(width: u32, height: u32) -> TextureDesc {
        TextureDesc::new_2d(
            width.max(1),
            height.max(1),
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        )
    }

    fn render_dimensions(&self, width: u32, height: u32) -> (u32, u32) {
        let scale = self.render_scale.clamp(0.5, 1.0);
        let scaled_width = ((width as f32) * scale).round() as u32;
        let scaled_height = ((height as f32) * scale).round() as u32;
        (scaled_width.max(1), scaled_height.max(1))
    }

    fn uses_scaled_surface(&self) -> bool {
        self.render_scale < 0.999
    }

    fn manual_light_direction(&self) -> [f32; 3] {
        Self::normalize_direction(self.settings.sun_direction)
    }

    fn reference_lighting_state(&self) -> OceanLightingState {
        OceanLightingState {
            direction: self.manual_light_direction(),
            color: [1.0, 0.95, 0.85],
            energy: 1.0,
            ambient: [0.06, 0.08, 0.10],
        }
    }

    fn resolve_lighting_state(&self, scene: &Scene) -> OceanLightingState {
        let ambient = Self::scene_ambient_color(scene);
        let manual = OceanLightingState {
            ambient,
            ..self.reference_lighting_state()
        };

        match self.light_source {
            OceanLightSource::Manual => manual,
            OceanLightSource::ProceduralSky => self
                .procedural_sky_lighting_state(scene)
                .or_else(|| self.scene_directional_lighting_state(scene))
                .unwrap_or(manual),
            OceanLightSource::SceneDirectional => self
                .scene_directional_lighting_state(scene)
                .unwrap_or(manual),
            OceanLightSource::SceneAuto => self
                .procedural_sky_lighting_state(scene)
                .or_else(|| self.scene_directional_lighting_state(scene))
                .unwrap_or(manual),
        }
    }

    fn resolve_camera_state(&self, scene: &mut Scene) -> OceanCameraState {
        match self.camera_source {
            OceanCameraSource::Reference => OceanCameraState::reference(self.time),
            OceanCameraSource::SceneMainCamera => self
                .scene_camera_state(scene)
                .unwrap_or_else(|| OceanCameraState::reference(self.time)),
        }
    }

    fn scene_camera_state(&self, scene: &mut Scene) -> Option<OceanCameraState> {
        let (transform, camera) = scene.query_main_camera_bundle()?;
        let (right, up, basis_forward) = transform.rotation_basis();
        let forward = (-basis_forward).normalize_or_zero();

        let projection = match camera.projection_type() {
            ProjectionType::Perspective => {
                let focal = 1.0 / (camera.fov() * 0.5).tan().max(1e-4);
                [focal, 0.0, 0.0, 0.0]
            }
            ProjectionType::Orthographic => {
                let half_height = camera.ortho_size();
                let half_width = half_height * camera.aspect();
                [0.0, half_width, half_height, 1.0]
            }
        };

        Some(OceanCameraState {
            position: transform.position.to_array(),
            right: right.to_array(),
            up: up.to_array(),
            forward: forward.to_array(),
            projection,
        })
    }

    fn procedural_sky_lighting_state(&self, scene: &Scene) -> Option<OceanLightingState> {
        let params = scene.background.procedural_sky_params()?;
        let (direction, color, energy) = Self::dominant_procedural_sky_light(params);
        Some(OceanLightingState {
            direction,
            color,
            energy,
            ambient: Self::scene_ambient_color(scene),
        })
    }

    fn scene_directional_lighting_state(&self, scene: &Scene) -> Option<OceanLightingState> {
        let ambient = Self::scene_ambient_color(scene);
        let mut best_state = None;
        let mut best_energy = -1.0_f32;

        for (light, world_matrix) in scene.iter_active_lights() {
            if !matches!(light.kind, LightKind::Directional(_)) {
                continue;
            }

            let state = Self::directional_light_state(light, world_matrix, ambient);
            if state.energy > best_energy {
                best_energy = state.energy;
                best_state = Some(state);
            }
        }

        best_state
    }

    fn directional_light_state(
        light: &myth_scene::Light,
        world_matrix: &Affine3A,
        ambient: [f32; 3],
    ) -> OceanLightingState {
        let direction = -world_matrix
            .transform_vector3(Vec3::new(0.0, 0.0, -1.0))
            .normalize_or_zero();

        OceanLightingState {
            direction: Self::normalize_direction(direction.to_array()),
            color: Self::normalize_color(light.color),
            energy: Self::light_intensity_to_energy(light.intensity),
            ambient,
        }
    }

    fn normalize_direction(direction: [f32; 3]) -> [f32; 3] {
        let normalized = Vec3::from_array(direction).normalize_or_zero();
        if normalized.length_squared() > 0.0 {
            normalized.to_array()
        } else {
            [0.0, 1.0, 0.8]
        }
    }

    fn normalize_color(color: Vec3) -> [f32; 3] {
        let clamped = color.max(Vec3::ZERO);
        if clamped.length_squared() > 0.0 {
            clamped.to_array()
        } else {
            [1.0, 1.0, 1.0]
        }
    }

    fn scene_ambient_color(scene: &Scene) -> [f32; 3] {
        let ambient = scene.environment.ambient.max(Vec3::ZERO);
        if ambient.length_squared() > 0.0 {
            ambient.to_array()
        } else {
            [0.006, 0.008, 0.012]
        }
    }

    fn dominant_procedural_sky_light(params: &ProceduralSkyParams) -> ([f32; 3], [f32; 3], f32) {
        let sun_direction = params.sun_direction.normalize_or_zero();
        let moon_direction = params.moon_direction.normalize_or_zero();
        let sun_weight =
            params.sun_intensity.max(0.0) * Self::smoothstep(-0.08, 0.04, sun_direction.y);
        let moon_weight = params.moon_intensity.max(0.0)
            * Self::moon_phase_fraction(sun_direction, moon_direction)
            * Self::smoothstep(-0.08, 0.04, moon_direction.y)
            * (1.0 - Self::smoothstep(-0.12, 0.04, sun_direction.y));

        if moon_weight > sun_weight {
            (
                Self::normalize_direction(moon_direction.to_array()),
                [0.62, 0.72, 1.0],
                Self::light_intensity_to_energy(moon_weight / 10.0),
            )
        } else {
            (
                Self::normalize_direction(sun_direction.to_array()),
                [1.0, 0.95, 0.8],
                Self::light_intensity_to_energy(sun_weight / 10.0),
            )
        }
    }

    fn moon_phase_fraction(sun_direction: Vec3, moon_direction: Vec3) -> f32 {
        (1.0 - sun_direction.dot(moon_direction).clamp(-1.0, 1.0)) * 0.5
    }

    fn light_intensity_to_energy(intensity: f32) -> f32 {
        1.0 - (-intensity.max(0.0) * 2.0).exp()
    }

    fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    fn direction_to_angles(direction: [f32; 3]) -> (f32, f32) {
        let dir = Vec3::from_array(Self::normalize_direction(direction));
        let azimuth = dir.x.atan2(-dir.z).to_degrees();
        let elevation = dir.y.asin().to_degrees();
        (azimuth, elevation)
    }

    fn angles_to_direction(azimuth_degrees: f32, elevation_degrees: f32) -> [f32; 3] {
        let azimuth = azimuth_degrees.to_radians();
        let elevation = elevation_degrees.to_radians();
        let dir = Vec3::new(
            azimuth.sin() * elevation.cos(),
            elevation.sin(),
            -azimuth.cos() * elevation.cos(),
        )
        .normalize_or_zero();
        dir.to_array()
    }

    fn build_quality_passes(renderer: &mut Renderer, composite: bool) -> OceanQualityPasses {
        OceanQualityPasses {
            low: Self::build_surface_pass(renderer, OceanQuality::Low, composite),
            medium: Self::build_surface_pass(renderer, OceanQuality::Medium, composite),
            high: Self::build_surface_pass(renderer, OceanQuality::High, composite),
        }
    }

    fn build_surface_pass(
        renderer: &mut Renderer,
        quality: OceanQuality,
        composite: bool,
    ) -> TemplateFullscreenPass {
        let mut shader_options = ShaderCompilationOptions::default();
        quality.apply_to_shader_options(&mut shader_options);
        if composite {
            shader_options.add_define("OCEAN_COMPOSITE", "1");
        }

        let mut builder = RenderPassBuilder::fullscreen(if composite {
            "Ocean Composite Pipeline"
        } else {
            "Ocean Surface Pipeline"
        })
        .inline_shader_template(OCEAN_SHADER_NAME, OCEAN_SHADER_TEMPLATE)
        .shader_options(shader_options)
        .bind_uniform_buffer(0, 0, wgpu::ShaderStages::FRAGMENT)
        .bind_texture_cube(0, 4, wgpu::ShaderStages::FRAGMENT, true)
        .bind_sampler(
            0,
            5,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::SamplerBindingType::Filtering,
        )
        .bind_texture_cube(0, 6, wgpu::ShaderStages::FRAGMENT, true)
        .bind_texture_2d(1, 0, wgpu::ShaderStages::FRAGMENT, false)
        .bind_uniform_buffer(1, 1, wgpu::ShaderStages::FRAGMENT);

        if composite {
            builder = builder
                .bind_texture_2d(0, 1, wgpu::ShaderStages::FRAGMENT, true)
                .bind_sampler(
                    0,
                    2,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                )
                .bind_depth_texture_2d(0, 3, wgpu::ShaderStages::FRAGMENT);
        }

        builder
            .color_target(wgpu::ColorTargetState {
                format: HDR_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })
            .build(renderer)
    }

    fn graph_uniform_snapshot(
        &self,
        environment_base_cube: Option<TextureNodeId>,
        environment_pmrem: Option<TextureNodeId>,
        environment_pmrem_max_mip_level: Option<f32>,
    ) -> OceanUniforms {
        let mut uniforms = self.uniform_snapshot;
        let environment_available = environment_base_cube.is_some()
            && environment_pmrem.is_some()
            && environment_pmrem_max_mip_level.is_some();
        uniforms.light_direction[3] =
            if environment_available && self.light_source == OceanLightSource::ProceduralSky {
                1.0
            } else {
                0.0
            };
        uniforms.ambient_color[3] = environment_pmrem_max_mip_level.unwrap_or(0.0);
        uniforms
    }

    fn add_surface_pass<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        width: u32,
        height: u32,
        pass_name: &'static str,
        texture_name: &'static str,
        environment_base_cube: Option<TextureNodeId>,
        environment_pmrem: Option<TextureNodeId>,
        environment_pmrem_max_mip_level: Option<f32>,
        atmosphere_transmittance: Option<TextureNodeId>,
        atmosphere_bake_params: Option<BufferNodeId>,
    ) -> TextureNodeId {
        let out_desc = Self::output_desc(width, height);
        let pass = self.surface_passes.get(self.quality);
        let fallback_cube_view = &self.fallback_cube_view;
        let fallback_atmosphere_transmittance_view = &self.fallback_atmosphere_transmittance_view;
        let fallback_atmosphere_bake_params = &self.fallback_atmosphere_bake_params;
        let uniform_snapshot = self.graph_uniform_snapshot(
            environment_base_cube,
            environment_pmrem,
            environment_pmrem_max_mip_level,
        );

        graph.add_pass(pass_name, |builder| {
            let uniforms = builder.create_buffer(
                "Ocean_Surface_Uniforms",
                BufferDesc::new(
                    std::mem::size_of::<OceanUniforms>() as u64,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
            );
            if let Some(environment_base_cube) = environment_base_cube {
                builder.read_texture(environment_base_cube);
            }
            if let Some(environment_pmrem) = environment_pmrem {
                builder.read_texture(environment_pmrem);
            }
            if let Some(transmittance) = atmosphere_transmittance {
                builder.read_texture(transmittance);
            }
            if let Some(bake_params) = atmosphere_bake_params {
                builder.read_buffer(bake_params);
            }
            let out = builder.create_texture(texture_name, out_desc);
            let inner = pass.build_node(
                builder,
                pass_name,
                out,
                RenderTargetOps::DontCare,
                Some(pass_name),
                |bindings| {
                    bindings.bind_buffer(0, 0, uniforms);
                    if let Some(environment_base_cube) = environment_base_cube {
                        bindings.bind_texture(0, 4, environment_base_cube);
                    } else {
                        bindings.bind_tracked_texture_view(0, 4, fallback_cube_view);
                    }
                    bindings.bind_common_sampler(0, 5, CommonSampler::LinearClamp);
                    if let Some(environment_pmrem) = environment_pmrem {
                        bindings.bind_texture(0, 6, environment_pmrem);
                    } else {
                        bindings.bind_tracked_texture_view(0, 6, fallback_cube_view);
                    }
                    if let Some(transmittance) = atmosphere_transmittance {
                        bindings.bind_texture(1, 0, transmittance);
                    } else {
                        bindings.bind_tracked_texture_view(
                            1,
                            0,
                            fallback_atmosphere_transmittance_view,
                        );
                    }
                    if let Some(bake_params) = atmosphere_bake_params {
                        bindings.bind_buffer(1, 1, bake_params);
                    } else {
                        bindings.bind_tracked_buffer(1, 1, fallback_atmosphere_bake_params);
                    }
                },
            );
            let node = OceanSurfaceNode {
                inner,
                uniforms,
                uniform_snapshot,
            };
            (node, out)
        })
    }

    pub fn apply_standalone<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        blackboard: GraphBlackboard,
        width: u32,
        height: u32,
    ) -> GraphBlackboard {
        let environment_reflections = self.light_source == OceanLightSource::ProceduralSky
            && blackboard.environment_base_cube.is_some()
            && blackboard.environment_pmrem.is_some()
            && blackboard.environment_pmrem_max_mip_level.is_some();
        let environment_base_cube = environment_reflections
            .then_some(blackboard.environment_base_cube)
            .flatten();
        let environment_pmrem = environment_reflections
            .then_some(blackboard.environment_pmrem)
            .flatten();
        let environment_pmrem_max_mip_level = environment_reflections
            .then_some(blackboard.environment_pmrem_max_mip_level)
            .flatten();
        let ocean_color = if self.uses_scaled_surface() {
            let (render_width, render_height) = self.render_dimensions(width, height);
            let low_res_ocean = self.add_surface_pass(
                graph,
                render_width,
                render_height,
                "Ocean_Standalone_Scaled_Surface",
                "Ocean_Surface_HDR_Scaled",
                environment_base_cube,
                environment_pmrem,
                environment_pmrem_max_mip_level,
                None,
                None,
            );
            let out_desc = Self::output_desc(width, height);
            let resolve_pass = &self.resolve_pass;

            graph.add_pass("Ocean_Standalone_Resolve", |builder| {
                builder.read_texture(low_res_ocean);
                let out = builder.create_texture("Ocean_Surface_HDR", out_desc);
                let node = resolve_pass.build_node(
                    builder,
                    "Ocean Standalone Resolve",
                    out,
                    RenderTargetOps::DontCare,
                    Some("Ocean Resolve BG"),
                    |bindings| {
                        bindings.bind_texture(0, 0, low_res_ocean);
                        bindings.bind_common_sampler(0, 1, CommonSampler::LinearClamp);
                    },
                );
                (node, out)
            })
        } else {
            self.add_surface_pass(
                graph,
                width,
                height,
                "Ocean_Standalone",
                "Ocean_Surface_HDR",
                environment_base_cube,
                environment_pmrem,
                environment_pmrem_max_mip_level,
                None,
                None,
            )
        };

        GraphBlackboard {
            scene_color: Some(ocean_color),
            ..blackboard
        }
    }

    pub fn apply_composite<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        blackboard: GraphBlackboard,
        width: u32,
        height: u32,
    ) -> GraphBlackboard {
        let atmosphere_transmittance = blackboard.atmosphere_transmittance;
        let atmosphere_bake_params = blackboard.atmosphere_bake_params;
        let environment_reflections = self.light_source == OceanLightSource::ProceduralSky
            && blackboard.environment_base_cube.is_some()
            && blackboard.environment_pmrem.is_some()
            && blackboard.environment_pmrem_max_mip_level.is_some();
        let environment_base_cube = environment_reflections
            .then_some(blackboard.environment_base_cube)
            .flatten();
        let environment_pmrem = environment_reflections
            .then_some(blackboard.environment_pmrem)
            .flatten();
        let environment_pmrem_max_mip_level = environment_reflections
            .then_some(blackboard.environment_pmrem_max_mip_level)
            .flatten();
        let (Some(scene_color), Some(scene_depth)) =
            (blackboard.scene_color, blackboard.scene_depth)
        else {
            return self.apply_standalone(graph, blackboard, width, height);
        };

        let composited_color = if self.uses_scaled_surface() {
            let (render_width, render_height) = self.render_dimensions(width, height);
            let low_res_ocean = self.add_surface_pass(
                graph,
                render_width,
                render_height,
                "Ocean_Composite_Scaled_Surface",
                "Ocean_Composite_HDR_Scaled",
                environment_base_cube,
                environment_pmrem,
                environment_pmrem_max_mip_level,
                atmosphere_transmittance,
                atmosphere_bake_params,
            );
            let out_desc = Self::output_desc(width, height);
            let pass = &self.composite_resolve_pass;

            graph.add_pass("Ocean_Composite_Resolve", |builder| {
                builder.read_texture(low_res_ocean);
                builder.read_texture(scene_color);
                builder.read_texture(scene_depth);
                let out = builder.create_texture("Ocean_Composite_HDR", out_desc);
                let node = pass.build_node(
                    builder,
                    "Ocean Composite Resolve",
                    out,
                    RenderTargetOps::DontCare,
                    Some("Ocean Composite Resolve BG"),
                    |bindings| {
                        bindings.bind_texture(0, 0, low_res_ocean);
                        bindings.bind_common_sampler(0, 1, CommonSampler::LinearClamp);
                        bindings.bind_texture(0, 2, scene_color);
                        bindings.bind_common_sampler(0, 3, CommonSampler::LinearClamp);
                        bindings.bind_texture(0, 4, scene_depth);
                    },
                );
                (node, out)
            })
        } else {
            let out_desc = Self::output_desc(width, height);
            let pass = self.composite_passes.get(self.quality);
            let fallback_cube_view = &self.fallback_cube_view;
            let fallback_atmosphere_transmittance_view =
                &self.fallback_atmosphere_transmittance_view;
            let fallback_atmosphere_bake_params = &self.fallback_atmosphere_bake_params;
            let uniform_snapshot = self.graph_uniform_snapshot(
                environment_base_cube,
                environment_pmrem,
                environment_pmrem_max_mip_level,
            );

            graph.add_pass("Ocean_Composite", |builder| {
                let uniforms = builder.create_buffer(
                    "Ocean_Composite_Uniforms",
                    BufferDesc::new(
                        std::mem::size_of::<OceanUniforms>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    ),
                );
                builder.read_texture(scene_color);
                builder.read_texture(scene_depth);
                if let Some(environment_base_cube) = environment_base_cube {
                    builder.read_texture(environment_base_cube);
                }
                if let Some(environment_pmrem) = environment_pmrem {
                    builder.read_texture(environment_pmrem);
                }
                if let Some(transmittance) = atmosphere_transmittance {
                    builder.read_texture(transmittance);
                }
                if let Some(bake_params) = atmosphere_bake_params {
                    builder.read_buffer(bake_params);
                }
                let out = builder.create_texture("Ocean_Composite_HDR", out_desc);
                let inner = pass.build_node(
                    builder,
                    "Ocean Composite Pass",
                    out,
                    RenderTargetOps::DontCare,
                    Some("Ocean Composite BG"),
                    |bindings| {
                        bindings.bind_buffer(0, 0, uniforms);
                        bindings.bind_texture(0, 1, scene_color);
                        bindings.bind_common_sampler(0, 2, CommonSampler::LinearClamp);
                        bindings.bind_texture(0, 3, scene_depth);
                        if let Some(environment_base_cube) = environment_base_cube {
                            bindings.bind_texture(0, 4, environment_base_cube);
                        } else {
                            bindings.bind_tracked_texture_view(0, 4, fallback_cube_view);
                        }
                        bindings.bind_common_sampler(0, 5, CommonSampler::LinearClamp);
                        if let Some(environment_pmrem) = environment_pmrem {
                            bindings.bind_texture(0, 6, environment_pmrem);
                        } else {
                            bindings.bind_tracked_texture_view(0, 6, fallback_cube_view);
                        }
                        if let Some(transmittance) = atmosphere_transmittance {
                            bindings.bind_texture(1, 0, transmittance);
                        } else {
                            bindings.bind_tracked_texture_view(
                                1,
                                0,
                                fallback_atmosphere_transmittance_view,
                            );
                        }
                        if let Some(bake_params) = atmosphere_bake_params {
                            bindings.bind_buffer(1, 1, bake_params);
                        } else {
                            bindings.bind_tracked_buffer(1, 1, fallback_atmosphere_bake_params);
                        }
                    },
                );
                let node = OceanSurfaceNode {
                    inner,
                    uniforms,
                    uniform_snapshot,
                };
                (node, out)
            })
        };

        GraphBlackboard {
            scene_color: Some(composited_color),
            ..blackboard
        }
    }
}
