{$ include 'core/full_screen_vertex' $}

{{ struct_definitions }}

@group(0) @binding(0) var t_scene_color: texture_2d<f32>;
@group(0) @binding(1) var t_clean_indirect: texture_2d<f32>;
@group(0) @binding(2) var t_material_data: texture_2d<f32>;
@group(0) @binding(3) var t_depth: texture_depth_2d;
@group(0) @binding(4) var t_normal: texture_2d<f32>;
@group(0) @binding(5) var s_linear: sampler;
@group(0) @binding(6) var s_point: sampler;
@group(0) @binding(7) var<uniform> u_ssgi: SsgiUniforms;

struct MergeOutput {
    @location(0) merged_color: vec4<f32>,
    @location(1) source_history: vec4<f32>,
};

fn make_merge_output(color: vec4<f32>) -> MergeOutput {
    return MergeOutput(color, color);
}

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn unpack_view_normal(packed: vec4<f32>) -> vec3<f32> {
    let raw = packed.xyz * 2.0 - 1.0;
    return normalize(select(vec3<f32>(0.0, 0.0, 1.0), raw, dot(raw, raw) > 1e-5));
}

fn linearize_depth(z: f32) -> f32 {
    return u_ssgi.temporal_params.z / max(z, 0.0001);
}

fn resolve_full_uv(half_pixel: vec2<u32>) -> vec2<f32> {
    if (u_ssgi.frame_params.z == 0u) {
        return (vec2<f32>(half_pixel) + vec2<f32>(0.5, 0.5)) * u_ssgi.half_resolution.zw;
    }

    let frame = u_ssgi.frame_params.x;
    let base = vec2<f32>(half_pixel * 2u);
    let offset_x = f32((frame & 1u) ^ ((frame & 2u) >> 1u));
    let offset_y = f32((frame & 2u) >> 1u);
    let offset = vec2<f32>(offset_x + 0.5, offset_y + 0.5);
    return (base + offset) * u_ssgi.full_resolution.zw;
}

@fragment
fn fs_main(in: VertexOutput) -> MergeOutput {
    let scene_color = textureSampleLevel(t_scene_color, s_point, in.uv, 0.0);
    let material_data = textureSampleLevel(t_material_data, s_point, in.uv, 0.0);
    let albedo = material_data.rgb;

    let full_depth = textureSampleLevel(t_depth, s_point, in.uv, 0u);
    let full_normal_packed = textureSampleLevel(t_normal, s_point, in.uv, 0.0);
    if (full_depth <= 0.0 || full_normal_packed.a < 0.5) {
        return make_merge_output(scene_color);
    }

    let full_linear = linearize_depth(full_depth);
    let full_normal = unpack_view_normal(full_normal_packed);
    let half_extent = vec2<i32>(textureDimensions(t_clean_indirect));
    let max_coord = half_extent - vec2<i32>(1, 1);
    let half_pos = in.uv * u_ssgi.half_resolution.xy - vec2<f32>(0.5, 0.5);
    let base_coord = vec2<i32>(floor(half_pos));
    let frac = fract(half_pos);

    var sum_color = vec3<f32>(0.0);
    var sum_weight = 0.0;

    for (var y: i32 = 0; y <= 1; y++) {
        for (var x: i32 = 0; x <= 1; x++) {
            let sample_coord = clamp(base_coord + vec2<i32>(x, y), vec2<i32>(0, 0), max_coord);
            let sample_indirect = textureLoad(t_clean_indirect, sample_coord, 0);
            if (sample_indirect.a <= 0.0) {
                continue;
            }

            let sample_uv = resolve_full_uv(vec2<u32>(sample_coord));
            let sample_depth = textureSampleLevel(t_depth, s_point, sample_uv, 0u);
            let sample_normal_packed = textureSampleLevel(t_normal, s_point, sample_uv, 0.0);
            if (sample_depth <= 0.0 || sample_normal_packed.a < 0.5) {
                continue;
            }

            let sample_linear = linearize_depth(sample_depth);
            let sample_normal = unpack_view_normal(sample_normal_packed);
            let bilinear_weight_x = select(1.0 - frac.x, frac.x, x == 1);
            let bilinear_weight_y = select(1.0 - frac.y, frac.y, y == 1);
            let spatial_weight = bilinear_weight_x * bilinear_weight_y;
            let depth_weight = exp(-abs(full_linear - sample_linear) * u_ssgi.merge_params.x);
            let normal_weight = pow(
                saturate(dot(full_normal, sample_normal)),
                u_ssgi.merge_params.y
            );
            let weight = spatial_weight * depth_weight * normal_weight;

            sum_color += sample_indirect.rgb * weight;
            sum_weight += weight;
        }
    }

    if (sum_weight <= 1e-4) {
        let nearest_coord = clamp(
            vec2<i32>(floor(half_pos + vec2<f32>(0.5, 0.5))),
            vec2<i32>(0, 0),
            max_coord
        );
        let nearest_indirect = textureLoad(t_clean_indirect, nearest_coord, 0);
        if (nearest_indirect.a <= 0.0) {
            return make_merge_output(scene_color);
        }

        let contribution = nearest_indirect.rgb * albedo * u_ssgi.ray_params.x;
        return make_merge_output(vec4<f32>(scene_color.rgb + contribution, scene_color.a));
    }

    let indirect = sum_color / sum_weight;
    let contribution = indirect * albedo * u_ssgi.ray_params.x;
    return make_merge_output(vec4<f32>(scene_color.rgb + contribution, scene_color.a));
}
