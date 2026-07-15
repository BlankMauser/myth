// ── Unlit Material Entry Point ───────────────────────────────────────────
//
// Simple unlit rendering: base color × optional albedo map, with alpha
// test support.  No lighting calculations.

{{ vertex_input_code }} 
{{ binding_code }}      
{{ scene_lighting_structs }}
{$ include 'core/vertex_output' $}
{$ include 'core/fragment_output' $}

$$ if USE_GPU_SCENE is defined
{$ include 'modules/gpu_scene_transform' $}
$$ endif

{$ include 'modules/geometry/morphing' $}
{$ include 'modules/geometry/skinning' $}
{$ include 'core/alpha_test' $}


@vertex
$$ if USE_GPU_SCENE is defined
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
$$ else
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
$$ endif
    var out: VertexOutput;

    var local_position = vec3<f32>(in.position.xyz);

    $$ if HAS_NORMAL
    var local_normal = vec3<f32>(in.normal.xyz);
    $$ endif

    // ── Morph Target Blending ────────────────────────────────────────
    $$ if HAS_MORPH_TARGETS
        let morphed = apply_morph_targets(
            vertex_index,
            local_position,
            $$ if HAS_MORPH_NORMALS and HAS_NORMAL
            local_normal,
            $$ endif
        );
        local_position = morphed.position;
        $$ if HAS_MORPH_NORMALS and HAS_NORMAL
        local_normal = morphed.normal;
        $$ endif
    $$ endif

    var local_pos = vec4<f32>(local_position, 1.0);

    // ── Skeletal Skinning ────────────────────────────────────────────
    $$ if HAS_SKINNING and SUPPORT_SKINNING
        let skinned = compute_skinned_vertex(
            local_pos,
            $$ if HAS_NORMAL
            local_normal,
            $$ endif
            vec4<u32>(in.joints),
            in.weights,
        );
        local_pos = skinned.position;
        $$ if HAS_NORMAL
        local_normal = skinned.normal;
        $$ endif
    $$ endif

    $$ if USE_GPU_SCENE is defined
    // Cull validated the generation-stamped visible record. The instance
    // table is immutable between Cull and Draw, so no per-vertex revalidation
    // is required here.
    let gpu_scene_visible = st_gpu_scene_visible_instances[instance_index];
    let gpu_scene_slot = gpu_scene_visible.slot;
    let model_world_matrix = st_gpu_scene_instances[gpu_scene_slot].world_matrix;
        $$ if HAS_NORMAL
    let model_normal_matrix = gpu_scene_normal_matrix_from_inverse(
        st_gpu_scene_instances[gpu_scene_slot].world_matrix_inverse,
    );
        $$ endif
    $$ else
    let model_world_matrix = u_model.world_matrix;
        $$ if HAS_NORMAL
    let model_normal_matrix = u_model.normal_matrix;
        $$ endif
    $$ endif

    let world_pos = model_world_matrix * local_pos;

    $$ if IN_TRANSPARENT_PASS is defined
        out.position = u_render_state.unjittered_view_projection * world_pos;
    $$ else
        out.position = u_render_state.view_projection * world_pos;
    $$ endif

    out.world_position = world_pos.xyz / world_pos.w;

    $$ if HAS_COLOR
        out.color = in.color;
    $$ endif

    $$ if HAS_UV
    out.uv = in.uv;
    $$ endif

    $$ if HAS_NORMAL
    out.geometry_normal = local_normal;
    out.normal = normalize(model_normal_matrix * local_normal);
    $$ endif

    {$ include 'mixins/uv_vertex' $}
    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var diffuse_color = u_material.color;
    {$ if HAS_MAP $}
    let tex_color = textureSample(t_map, s_map, in.map_uv);
    diffuse_color = diffuse_color * tex_color;
    {$ endif $}

    $$ if ALPHA_MODE == "MASK" or ALPHA_MODE == "BLEND_MASK"
    var opacity = diffuse_color.a;
    apply_alpha_test(&opacity, u_material.alpha_test);
    $$ endif

    return pack_fragment_output(diffuse_color);
}
