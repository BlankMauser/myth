$$ if GPU_SCENE_MULTI_BIN is defined
// M7.4 multi-bin ABI. The stable batch identity deliberately occupies the
// former instance-tint field, preserving the 256-byte instance-table stride.
struct GpuSceneInstance {
    world_matrix: mat4x4<f32>,
    world_matrix_inverse: mat4x4<f32>,
    previous_world_matrix: mat4x4<f32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    batch_id: vec4<u32>,
    metadata: vec4<u32>,
}

// Frozen 80-byte persistent BatchTable record. `partition_bytes` is represented
// as low/high u32 pairs so the shader remains portable without shader-i64.
struct GpuSceneBatch {
    // generation, alive, scene_id, member_count
    metadata: vec4<u32>,
    geometry_id: vec2<u32>,
    material_id: vec2<u32>,
    // partition base low/high, partition size low/high
    partition_bytes: vec4<u32>,
    // base_visible_index, capacity_records, index_count, variant_flags,
    // item_features, topology, index_format, reserved
    draw_state: array<u32, 8>,
}

struct GpuSceneVisibleInstance {
    slot: u32,
    generation: u32,
}

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct GpuSceneActiveBatch {
    // stable batch slot, generation, reserved, reserved
    id: vec4<u32>,
}

struct GpuSceneCullView {
    frustum_planes: array<vec4<f32>, 6>,
}

struct GpuSceneMultiBinCullDomain {
    // scene_id, instance_slot_span, batch_slot_span, active_batch_count
    domain: vec4<u32>,
    // view_count, visible_view_stride_records, required_flags, dispatch_items
    config: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> instances: array<GpuSceneInstance>;
@group(0) @binding(1) var<storage, read> batches: array<GpuSceneBatch>;
@group(0) @binding(2) var<storage, read> active_batches: array<GpuSceneActiveBatch>;
@group(0) @binding(3) var<storage, read_write> visible_instances: array<GpuSceneVisibleInstance>;
@group(0) @binding(4) var<storage, read_write> indirect_args: array<DrawIndexedIndirectArgs>;
@group(0) @binding(5) var<storage, read> views: array<GpuSceneCullView>;
@group(0) @binding(6) var<uniform> cull_domain: GpuSceneMultiBinCullDomain;

fn multi_bin_has_non_finite_bounds(bounds_min: vec3<f32>, bounds_max: vec3<f32>) -> bool {
    let max_finite = vec3<f32>(bitcast<f32>(0x7f7fffffu));
    return any(bounds_min != bounds_min) || any(bounds_max != bounds_max) ||
        any(abs(bounds_min) > max_finite) || any(abs(bounds_max) > max_finite);
}

fn multi_bin_intersects_frustum(
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
    view: GpuSceneCullView,
) -> bool {
    if (multi_bin_has_non_finite_bounds(bounds_min, bounds_max)) {
        return true;
    }
    for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
        let plane = view.frustum_planes[plane_index];
        let p_vertex = select(bounds_min, bounds_max, plane.xyz >= vec3<f32>(0.0));
        if (dot(plane.xyz, p_vertex) + plane.w < 0.0) {
            return false;
        }
    }
    return true;
}

fn valid_batch(batch: GpuSceneBatch, generation: u32) -> bool {
    if (generation == 0u || batch.metadata.x != generation || batch.metadata.y == 0u ||
        batch.metadata.z != cull_domain.domain.x || batch.metadata.w == 0u) {
        return false;
    }

    // CPU publication stores exact byte metadata plus division-free record
    // indices. Requiring both representations to agree turns torn/coherency
    // bugs into a fail-closed batch instead of an out-of-bounds draw.
    let base_records = batch.draw_state[0];
    let capacity_records = batch.draw_state[1];
    let member_count = batch.metadata.w;
    if (capacity_records == 0u || batch.draw_state[2] == 0u ||
        member_count > capacity_records ||
        batch.partition_bytes.y != 0u || batch.partition_bytes.w != 0u ||
        batch.partition_bytes.x != base_records * 8u ||
        batch.partition_bytes.z != capacity_records * 8u ||
        base_records > cull_domain.config.y ||
        capacity_records > cull_domain.config.y - base_records) {
        return false;
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let item = gid.x;
    let view_index = gid.y;
    if (view_index >= cull_domain.config.x) {
        return;
    }

    // The compact list initializes sparse, view-major argument slots. Holes
    // remain zero from the encoder clear and are never visited by CPU draws.
    if (item < cull_domain.domain.w && item < arrayLength(&active_batches)) {
        let active_id = active_batches[item].id;
        if (active_id.x < cull_domain.domain.z && active_id.x < arrayLength(&batches)) {
            let batch = batches[active_id.x];
            if (valid_batch(batch, active_id.y)) {
                let argument_index = view_index * cull_domain.domain.z + active_id.x;
                if (argument_index < arrayLength(&indirect_args)) {
                    indirect_args[argument_index].index_count = batch.draw_state[2];
                    // Each Geometry owns a complete index buffer in M7.4.
                    indirect_args[argument_index].first_index = 0u;
                    indirect_args[argument_index].base_vertex = 0;
                    // Draw binds the exact visible partition.
                    indirect_args[argument_index].first_instance = 0u;
                }
            }
        }
    }

    if (item >= cull_domain.domain.y || item >= arrayLength(&instances)) {
        return;
    }
    let instance = instances[item];
    let instance_generation = instance.metadata.x;
    let alive = instance.metadata.y;
    let scene_id = instance.metadata.z;
    let instance_flags = instance.metadata.w;
    let required_flags = cull_domain.config.z;
    if (instance_generation == 0u || alive == 0u || scene_id != cull_domain.domain.x ||
        (instance_flags & required_flags) != required_flags) {
        return;
    }
    if (!multi_bin_intersects_frustum(
        instance.bounds_min.xyz,
        instance.bounds_max.xyz,
        views[view_index],
    )) {
        return;
    }

    let batch_slot = instance.batch_id.x;
    let batch_generation = instance.batch_id.y;
    if (batch_slot >= cull_domain.domain.z || batch_slot >= arrayLength(&batches)) {
        return;
    }
    let batch = batches[batch_slot];
    // Stable slot alone is insufficient: retirement/reuse must match both the
    // instance's captured generation and the live BatchTable generation.
    if (!valid_batch(batch, batch_generation)) {
        return;
    }

    let argument_index = view_index * cull_domain.domain.z + batch_slot;
    if (argument_index >= arrayLength(&indirect_args)) {
        return;
    }
    var local_index = 0xffffffffu;
    loop {
        let current = atomicLoad(&indirect_args[argument_index].instance_count);
        // The aligned partition capacity is a memory bound, not a semantic
        // draw count. Capping at the authority's exact member count prevents
        // a duplicate or missed tombstone from turning padding into draws.
        if (current >= batch.metadata.w) {
            break;
        }
        let result = atomicCompareExchangeWeak(
            &indirect_args[argument_index].instance_count,
            current,
            current + 1u,
        );
        if (result.exchanged) {
            local_index = current;
            break;
        }
    }
    if (local_index == 0xffffffffu) {
        return;
    }
    let visible_index = view_index * cull_domain.config.y +
        batch.draw_state[0] + local_index;
    if (visible_index >= arrayLength(&visible_instances)) {
        return;
    }
    visible_instances[visible_index] = GpuSceneVisibleInstance(item, instance_generation);
}

$$ else
// Canonical GPU Scene rigid-instance table ABI. Keep this in lockstep with
// `GpuSceneInstanceRecord` (256 bytes) and `GpuSceneCullParams` (128 bytes).
struct GpuSceneInstance {
    world_matrix: mat4x4<f32>,
    world_matrix_inverse: mat4x4<f32>,
    previous_world_matrix: mat4x4<f32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    instance_tint: vec4<f32>,
    metadata: vec4<u32>,
}

struct GpuSceneVisibleInstance {
    slot: u32,
    generation: u32,
}

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct GpuSceneCullParams {
    frustum_planes: array<vec4<f32>, 6>,
    // active_scene_id, slot_span, index_count, first_index
    domain: vec4<u32>,
    // base_vertex, required_instance_flags bits, visible_base bits, reserved
    draw: vec4<i32>,
}

@group(0) @binding(0) var<storage, read> instances: array<GpuSceneInstance>;
@group(0) @binding(1) var<storage, read_write> visible_instances: array<GpuSceneVisibleInstance>;
@group(0) @binding(2) var<storage, read_write> indirect_args: array<DrawIndexedIndirectArgs>;
$$ if GPU_SCENE_MULTI_VIEW is defined
@group(0) @binding(3) var<storage, read> params: array<GpuSceneCullParams>;
$$ else
@group(0) @binding(3) var<uniform> params: GpuSceneCullParams;
$$ endif

fn has_non_finite_bounds(bounds_min: vec3<f32>, bounds_max: vec3<f32>) -> bool {
    // Naga's WGSL frontend intentionally does not expose GLSL-style isNan /
    // isInf builtins. A NaN is the only float unequal to itself, while both
    // infinities exceed the largest finite IEEE-754 f32 after abs().
    let max_finite = vec3<f32>(bitcast<f32>(0x7f7fffffu));
    return any(bounds_min != bounds_min) || any(bounds_max != bounds_max) ||
        any(abs(bounds_min) > max_finite) || any(abs(bounds_max) > max_finite);
}

fn intersects_frustum(
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
    view_params: GpuSceneCullParams,
) -> bool {
    // Match the CPU policy: malformed/non-finite bounds are conservatively kept.
    if (has_non_finite_bounds(bounds_min, bounds_max)) {
        return true;
    }

    for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
        let plane = view_params.frustum_planes[plane_index];
        let p_vertex = select(bounds_min, bounds_max, plane.xyz >= vec3<f32>(0.0));
        if (dot(plane.xyz, p_vertex) + plane.w < 0.0) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    $$ if GPU_SCENE_MULTI_VIEW is defined
    let view_index = gid.y;
    let view_params = params[view_index];
    $$ else
    let view_index = 0u;
    let view_params = params;
    $$ endif
    let required_instance_flags = bitcast<u32>(view_params.draw.y);
    let visible_base = bitcast<u32>(view_params.draw.z);

    // `clear_buffer` initializes all five fields before this dispatch. Invocation
    // zero writes only static fields; no invocation writes instance_count except
    // through the atomic below, so multiple workgroups cannot race initialization.
    if (slot == 0u) {
        indirect_args[view_index].index_count = view_params.domain.z;
        indirect_args[view_index].first_index = view_params.domain.w;
        indirect_args[view_index].base_vertex = view_params.draw.x;
        indirect_args[view_index].first_instance = visible_base;
    }

    if (slot >= view_params.domain.y || slot >= arrayLength(&instances)) {
        return;
    }

    let instance = instances[slot];
    let generation = instance.metadata.x;
    let alive = instance.metadata.y;
    let scene_id = instance.metadata.z;
    let instance_flags = instance.metadata.w;
    if (generation == 0u || alive == 0u || scene_id != view_params.domain.x ||
        (instance_flags & required_instance_flags) != required_instance_flags) {
        return;
    }
    if (!intersects_frustum(
        instance.bounds_min.xyz,
        instance.bounds_max.xyz,
        view_params,
    )) {
        return;
    }

    let visible_index = visible_base + atomicAdd(&indirect_args[view_index].instance_count, 1u);
    visible_instances[visible_index] = GpuSceneVisibleInstance(slot, generation);
}
$$ endif
