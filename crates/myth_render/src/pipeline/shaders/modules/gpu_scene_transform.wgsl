// Reconstruct the CPU `DynamicModelUniforms::normal_matrix` value from the
// canonical GPU Scene inverse transform. WGSL matrices are column-major, so
// the upper-left inverse 3x3 must be transposed for normal transformation.
fn gpu_scene_normal_matrix_from_inverse(
    world_matrix_inverse: mat4x4<f32>,
) -> mat3x3<f32> {
    return transpose(mat3x3<f32>(
        world_matrix_inverse[0].xyz,
        world_matrix_inverse[1].xyz,
        world_matrix_inverse[2].xyz,
    ));
}
