{$ include 'core/full_screen_vertex' $}

{{ struct_definitions }}

override SSGI_ATROUS_STEP_SIZE: u32 = 1u;

@group(0) @binding(0) var t_indirect: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var t_normal: texture_2d<f32>;
@group(0) @binding(3) var t_variance_meta: texture_2d<f32>;
@group(0) @binding(4) var s_linear: sampler;
@group(0) @binding(5) var s_point: sampler;
@group(0) @binding(6) var<uniform> u_ssgi: SsgiUniforms;

const KERNEL_WEIGHTS: array<f32, 2> = array<f32, 2>(0.5, 0.25);

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

fn half_pixel_to_uv(half_pixel: vec2<u32>) -> vec2<f32> {
    return (vec2<f32>(half_pixel) + vec2<f32>(0.5, 0.5)) * u_ssgi.half_resolution.zw;
}

fn resolve_full_uv(half_pixel: vec2<u32>) -> vec2<f32> {
    if (u_ssgi.frame_params.z == 0u) {
        return half_pixel_to_uv(half_pixel);
    }

    let frame = u_ssgi.frame_params.x;
    let base = vec2<f32>(half_pixel * 2u);
    let offset_x = f32((frame & 1u) ^ ((frame & 2u) >> 1u));
    let offset_y = f32((frame & 2u) >> 1u);
    let offset = vec2<f32>(offset_x + 0.5, offset_y + 0.5);
    return (base + offset) * u_ssgi.full_resolution.zw;
}

fn perceptual_luma(color: vec3<f32>) -> f32 {
    let clamped = max(color, vec3<f32>(0.0));
    return log2(1.0 + dot(clamped, vec3<f32>(0.2126, 0.7152, 0.0722)));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_pixel = vec2<u32>(in.position.xy);
    let center_half_uv = half_pixel_to_uv(center_pixel);
    let center = textureSampleLevel(t_indirect, s_linear, center_half_uv, 0.0);
    if (center.a <= 0.0) {
        return vec4<f32>(0.0);
    }

    let center_full_uv = resolve_full_uv(center_pixel);
    let center_depth = textureSampleLevel(t_depth, s_point, center_full_uv, 0u);
    let center_normal_packed = textureSampleLevel(t_normal, s_point, center_full_uv, 0.0);
    if (center_depth <= 0.0 || center_normal_packed.a < 0.5) {
        return center;
    }

    let center_linear = linearize_depth(center_depth);
    let center_normal = unpack_view_normal(center_normal_packed);
    let center_luma = perceptual_luma(center.rgb);
    let spatial_variance = max(textureLoad(t_variance_meta, vec2<i32>(center_pixel), 0).a, 1e-4);
    let variance_sigma = sqrt(spatial_variance);
    let estimated_variance = max(variance_sigma, 1e-3);
    let step_size = i32(max(SSGI_ATROUS_STEP_SIZE, 1u));
    let half_extent = vec2<i32>(i32(u_ssgi.half_resolution.x), i32(u_ssgi.half_resolution.y));
    let depth_sigma = max(u_ssgi.reprojection_params.w * max(center_linear, 1.0), 1e-3);
    let luma_phi = max(u_ssgi.lighting_params.w * estimated_variance, 1e-3);
    let center_weight = KERNEL_WEIGHTS[0] * KERNEL_WEIGHTS[0];

    var color_sum = center.rgb * center_weight;
    var weight_sum = center_weight;

    for (var y: i32 = -1; y <= 1; y++) {
        for (var x: i32 = -1; x <= 1; x++) {
            if (x == 0 && y == 0) {
                continue;
            }

            let sample_half = vec2<i32>(center_pixel) + vec2<i32>(x, y) * step_size;
            if (sample_half.x < 0
                || sample_half.y < 0
                || sample_half.x >= half_extent.x
                || sample_half.y >= half_extent.y) {
                continue;
            }

            let sample_pixel = vec2<u32>(sample_half);
            let sample_half_uv = half_pixel_to_uv(sample_pixel);
            let sample_indirect = textureSampleLevel(t_indirect, s_linear, sample_half_uv, 0.0);
            if (sample_indirect.a <= 0.0) {
                continue;
            }

            let sample_full_uv = resolve_full_uv(sample_pixel);
            let sample_depth = textureSampleLevel(t_depth, s_point, sample_full_uv, 0u);
            let sample_normal_packed = textureSampleLevel(t_normal, s_point, sample_full_uv, 0.0);
            if (sample_depth <= 0.0 || sample_normal_packed.a < 0.5) {
                continue;
            }

            let sample_linear = linearize_depth(sample_depth);
            let sample_normal = unpack_view_normal(sample_normal_packed);
            let depth_delta = center_linear - sample_linear;
            let depth_weight = exp(-(depth_delta * depth_delta) / (2.0 * depth_sigma * depth_sigma));
            let normal_weight = pow(
                saturate(dot(center_normal, sample_normal)),
                u_ssgi.lighting_params.z
            );

            let sample_luma = perceptual_luma(sample_indirect.rgb);
            let luma_delta = abs(center_luma - sample_luma);
            let luma_weight = exp(-luma_delta / max(luma_phi, 1e-4));
            let spatial_weight = KERNEL_WEIGHTS[u32(abs(x))] * KERNEL_WEIGHTS[u32(abs(y))];
            let weight = spatial_weight * depth_weight * normal_weight * luma_weight;

            color_sum += sample_indirect.rgb * weight;
            weight_sum += weight;
        }
    }

    return vec4<f32>(color_sum / max(weight_sum, 1e-4), center.a);
}
