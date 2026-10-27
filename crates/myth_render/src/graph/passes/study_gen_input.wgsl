// Deterministic HDR input generator for the Myth vs LunarFenrir study.
// See PROTOCOL.md section 2. Exact by construction: integer arithmetic only,
// and every output value is exactly representable in f16 (colour, velocity)
// or f32 (depth), so every conforming implementation writes the same bits.
//
// One render pass, fullscreen triangle (vs_main = Myth's core/full_screen_vertex).
// Attachments:
//   location 0: Rgba16Float  HDR colour      -> TAA t_current_color
//   location 1: Rg16Float    velocity (UV)   -> TAA t_velocity (cur - prev)
//   depth:      Depth32Float reverse-Z depth -> TAA t_scene_depth
//               depth compare Always, depth write on, clear 0.0
// Bindings: @group(0) @binding(0) uniform GenParams (16 bytes).

struct GenParams {
    width: i32,
    height: i32,
    frame: i32,
    _pad: i32,
};
@group(0) @binding(0) var<uniform> u_gen: GenParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    output.uv = pos[vertex_index] * 0.5 + 0.5;
    output.uv.y = 1.0 - output.uv.y;
    return output;
}

// Sub-pixel units per pixel.
const Q: i32 = 64;

// Jitter in 1/64 px: round(64 * (halton(i+1, 2) - 0.5)), round(64 * (halton(i+1, 3) - 0.5)),
// i = frame % 16 (Myth's camera: 16-entry Halton(2,3) sequence, rounded to 1/64 px).
var<private> JITTER: array<vec2<i32>, 16> = array<vec2<i32>, 16>(
    vec2<i32>(  0, -11), vec2<i32>(-16,  11), vec2<i32>( 16, -25), vec2<i32>(-24,  -4),
    vec2<i32>(  8,  18), vec2<i32>( -8, -18), vec2<i32>( 24,   4), vec2<i32>(-28,  25),
    vec2<i32>(  4, -30), vec2<i32>(-12,  -8), vec2<i32>( 20,  13), vec2<i32>(-20, -23),
    vec2<i32>( 12,  -1), vec2<i32>( -4,  20), vec2<i32>( 28, -15), vec2<i32>(-30,   6)
);

struct GenOut {
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec2<f32>,
    @builtin(frag_depth) depth: f32,
};

fn disk_d2(sx: i32, sy: i32, cx: i32, cy: i32) -> i32 {
    // Clamp before squaring: 2 * 32767^2 < 2^31.
    let dx = clamp(sx - cx, -32767, 32767);
    let dy = clamp(sy - cy, -32767, 32767);
    return dx * dx + dy * dy;
}

@fragment
fn fs_main(in: VertexOutput) -> GenOut {
    let w = u_gen.width;
    let h = u_gen.height;
    let f = u_gen.frame;

    // Pixel index, origin top-left (framebuffer coordinates).
    let px = i32(in.position.x);
    let py = i32(in.position.y);
    let j = JITTER[f % 16];
    // Jittered sample position in 1/64 px.
    let sx = px * Q + Q / 2 + j.x;
    let sy = py * Q + Q / 2 + j.y;

    // 1. Background: horizontal gradient (unjittered), far depth, static.
    let g = f32((px * 1024) / w) * 0.0009765625;  // exact: k / 1024
    var color = vec3<f32>(g, 0.5 * g, 0.25);
    var depth = 0.125;
    var velocity = vec2<f32>(0.0, 0.0);

    // 2. Grid lines every 64 px (1 px wide, jittered), HDR grey.
    if ((sx / Q) % 64 == 0 || (sy / Q) % 64 == 0) {
        color = vec3<f32>(1.5, 1.5, 1.5);
    }

    // 3. Diagonal stripe panel (6 px period along x+y), static.
    if (sx >= (w / 16) * Q && sx < (w / 4) * Q && sy >= (h / 8) * Q && sy < ((7 * h) / 8) * Q) {
        if (((sx + sy) / (Q * 6)) % 2 == 0) {
            color = vec3<f32>(0.125, 0.125, 0.125);
        } else {
            color = vec3<f32>(0.75, 0.75, 0.75);
        }
        depth = 0.25;
    }

    // 4. Slanted bar through the centre (slope 5:2), static.
    let bx = sx - (w * Q) / 2;
    let by = sy - (h * Q) / 2;
    if (abs(5 * bx - 2 * by) < 3 * Q * 8 && abs(by) < (h * Q) / 3) {
        color = vec3<f32>(0.875, 0.875, 0.1875);
        depth = 0.25;
    }

    // 5. HDR highlights (bloom / tone-mapping drivers), static.
    let r_hi = (h * Q) / 60;
    if (disk_d2(sx, sy, (w * Q * 5) / 8, (h * Q) / 4) < r_hi * r_hi) {
        color = vec3<f32>(16.0, 12.0, 8.0);
        depth = 0.25;
    }
    if (disk_d2(sx, sy, (w * Q * 3) / 4, (h * Q * 3) / 4) < r_hi * r_hi) {
        color = vec3<f32>(2.0, 8.0, 32.0);
        depth = 0.25;
    }
    if (disk_d2(sx, sy, (w * Q * 7) / 8, h * Q / 2) < r_hi * r_hi) {
        color = vec3<f32>(64.0, 64.0, 64.0);
        depth = 0.25;
    }

    // 6. Moving disk (TAA motion): ping-pong along x, period 1024 frames,
    //    step W/1024 px per frame = 1/1024 UV per frame.
    let t = f % 1024;
    let s = select(1024 - t, t, t < 512);
    let step_q = (w * Q) / 1024;           // = W/16 (W must be a multiple of 16)
    let cx = (w * Q) / 4 + s * step_q;
    let cy = (h * Q) / 2;
    let big_r = (h * Q) / 6;
    let d2 = disk_d2(sx, sy, cx, cy);
    if (d2 < big_r * big_r) {
        let r0 = big_r / 2;
        let r1 = (big_r * 5) / 8;
        if (d2 >= r0 * r0 && d2 < r1 * r1) {
            color = vec3<f32>(0.25, 0.5, 4.0);
        } else {
            color = vec3<f32>(4.0, 1.0, 0.5);
        }
        depth = 0.5;
        // Velocity = current UV - previous UV. Direction of the step from
        // frame f-1 to f: +1 for t in [1, 512], -1 otherwise (t = 0 included).
        let dir = select(-1.0, 1.0, t >= 1 && t <= 512);
        velocity = vec2<f32>(dir * 0.0009765625, 0.0);  // exactly +-2^-10
    }

    var out: GenOut;
    out.color = vec4<f32>(color, 1.0);
    out.velocity = velocity;
    out.depth = depth;
    return out;
}
