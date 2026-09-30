//! Post-DPE materialization of rigid multi-bin GPU Scene submissions.
//!
//! Graph construction carries only logical table/Cull buffers and compact
//! fixed-state representatives. This coordinator validates the complete batch
//! image against the exact graph-attempt Geometry/Material owners before it
//! creates pipelines or Group 2 bindings, then atomically publishes one
//! immutable indexed-indirect stream as the sole execution linearization point.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::binding_layout::storage_buffer;
use crate::core::gpu::{GpuGlobalState, MaterialPipelineSnapshot, Tracked};
use crate::core::{BindingInterfaceId, WgpuContext};
use crate::graph::asset_frame::{GeometryFrameId, GeometryFramePhysical, GeometryFrameRecord};
use crate::graph::capacity::{
    CapacityUsage, estimated_hash_map_footprint, shrink_hash_map_by_family_ratio,
    shrink_vec_by_family_ratio, vec_footprint,
};
use crate::graph::core::scene_submission::{
    GpuIndexedIndirectBatchInput, SceneSubmissionKey, SceneSubmissionPlanId,
};
use crate::graph::core::{BufferNodeId, PrepareContext};
use crate::graph::extracted::{ExtractedRenderItem, SceneFeatures};
use crate::graph::frame::{DrawBindGroupSlot, DrawBufferSlot, DrawPlanStore};
use crate::graph::gpu_scene::GpuSceneMultiBinCandidate;
use crate::graph::material_plan::{MaterialFrameId, PreparedMaterialFrame};
use crate::graph::passes::gpu_scene::{
    GPU_SCENE_VISIBLE_INSTANCE_SIZE, GpuSceneMultiBinCullOutput, GpuSceneMultiBinDrawOutput,
};
use crate::graph::passes::prepass::{FEATURE_ID_FORMAT, NORMAL_FORMAT, VELOCITY_FORMAT};
use crate::graph::passes::shadow::SHADOW_BINDING_WGSL;
use crate::graph::scene_pipeline_prepare::PrepassPipelineL1Key;
use crate::pipeline::pipeline_key::PipelineFlags;
use crate::pipeline::{
    BlendStateKey, ColorTargetKey, DepthStencilKey, FastPipelineKey, FastShadowPipelineKey,
    GraphicsPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
    SimpleGeometryPipelineKey,
};
use myth_resources::AntiAliasingMode;
use myth_resources::material::{AlphaMode, Side};
use myth_scene::camera::RenderCamera;

const GPU_SCENE_FINALIZATION_SCRATCH_BASELINE: usize = 16;
const GPU_SCENE_PREPASS_PIPELINE_L1_BASELINE: usize = 16;

const GPU_SCENE_OBJECT_BINDING_WGSL: &str = r"
struct GpuSceneInstance {
    world_matrix: mat4x4<f32>,
    world_matrix_inverse: mat4x4<f32>,
    previous_world_matrix: mat4x4<f32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    batch_id: vec4<u32>,
    metadata: vec4<u32>,
};

struct GpuSceneVisibleInstance {
    slot: u32,
    generation: u32,
};

@group(2) @binding(0)
var<storage, read> st_gpu_scene_instances: array<GpuSceneInstance>;

@group(2) @binding(1)
var<storage, read> st_gpu_scene_visible_instances: array<GpuSceneVisibleInstance>;
";

/// One semantic view family whose immutable per-view execute streams contain
/// every active batch in CPU first-occurrence order.
///
/// Main and Prepass families contain one submission; Shadow2D and ShadowCube
/// contain one submission per packed Cull view. The submission slice lives in
/// the frame arena, so family cardinality never spills a per-frame heap
/// container. The Cull output retains only logical RDG IDs and a borrowed
/// compact bin projection.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuSceneMultiBinFamilyPreparePlan<'a> {
    pub(crate) submissions: &'a [SceneSubmissionPlanId],
    pub(crate) candidates: &'a [GpuSceneMultiBinCandidate],
    pub(crate) instance_table: BufferNodeId,
    pub(crate) culled: GpuSceneMultiBinCullOutput<'a>,
    pub(crate) target: GpuSceneSubmissionTarget,
}

/// Fixed render-pass contract selected at graph construction. The logical
/// scene-submission key and this physical pipeline target are validated again
/// by the post-prepare coordinator before immutable publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuSceneSubmissionTarget {
    Main {
        global_state_key: crate::core::gpu::GlobalStateKey,
    },
    Prepass {
        global_state_key: crate::core::gpu::GlobalStateKey,
        needs_normal: bool,
        needs_feature_id: bool,
        needs_velocity: bool,
    },
    Shadow,
}

/// A normal drift is reported to Composer and terminates the attempt before
/// execution. No variant can expose a partially published GPU stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuSceneSubmissionFinalizeFailure {
    DescriptorDrift,
    RepresentativeDrift,
    GeometryUnavailable,
    GeometryDrift,
    MaterialUnavailable,
    MaterialDrift,
    GlobalStateUnavailable,
    UnsupportedMaterialState,
    BatchPublicationRejected,
}

#[derive(Debug, Clone, Copy)]
struct ValidatedGpuSceneBin {
    candidate_index: u32,
    geometry: GeometryFrameId,
    material: MaterialFrameId,
    draw: GpuSceneMultiBinDrawOutput,
}

#[derive(Debug, Clone, Copy)]
struct PreparedGpuSceneBin {
    pipeline: RenderPipelineId,
    vertex_start: u32,
    vertex_end: u32,
    index_buffer: DrawBufferSlot,
    index_format: wgpu::IndexFormat,
    material_bind_group: DrawBindGroupSlot,
    object_bind_group: PreparedBindGroupId,
    indirect_offset: u64,
}

/// Renderer-owned, cross-frame static Group 2 ABI. Its identity is stable for
/// the Feature lifetime and never includes scene or physical Buffer state.
#[derive(Default)]
pub(crate) struct GpuSceneRenderFeature {
    object_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    object_binding_interface_id: Option<BindingInterfaceId>,
    prepass_pipeline_l1: FxHashMap<PrepassPipelineL1Key, RenderPipelineId>,
    /// Retained two-phase validation image. It contains only graph-attempt IDs
    /// and is cleared without releasing capacity on every live finalization.
    validated_bin_scratch: Vec<ValidatedGpuSceneBin>,
    /// Retained flat immutable-stream recipe populated only after every bin
    /// passed validation. Publication maps this storage directly by iterator.
    prepared_bin_scratch: Vec<PreparedGpuSceneBin>,
    /// Variable-size vertex payload shared by every prepared batch record.
    /// Ranges in `prepared_bin_scratch` address this one retained allocation.
    vertex_slot_scratch: Vec<DrawBufferSlot>,
}

impl GpuSceneRenderFeature {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Capacity family containing persistent Prepass L1 headroom plus the
    /// transient validation/materialization vectors used by finalization.
    ///
    /// Maintenance may shrink the L1 map allocation but must never clear its
    /// live entries. Only the three scratch vectors have frame-local content.
    pub(crate) fn finalization_capacity_usage(&self) -> CapacityUsage {
        estimated_hash_map_footprint(
            &self.prepass_pipeline_l1,
            GPU_SCENE_PREPASS_PIPELINE_L1_BASELINE,
        ) + vec_footprint(
            &self.validated_bin_scratch,
            GPU_SCENE_FINALIZATION_SCRATCH_BASELINE,
        ) + vec_footprint(
            &self.prepared_bin_scratch,
            GPU_SCENE_FINALIZATION_SCRATCH_BASELINE,
        ) + vec_footprint(
            &self.vertex_slot_scratch,
            GPU_SCENE_FINALIZATION_SCRATCH_BASELINE,
        )
    }

    pub(crate) fn clear_finalization_scratch(&mut self) {
        self.validated_bin_scratch.clear();
        self.prepared_bin_scratch.clear();
        self.vertex_slot_scratch.clear();
    }

    pub(crate) fn trim_finalization_capacity(
        &mut self,
        target_bytes: usize,
        retained_bytes: usize,
    ) {
        debug_assert_eq!(
            vec_footprint(
                &self.validated_bin_scratch,
                GPU_SCENE_FINALIZATION_SCRATCH_BASELINE,
            )
            .active_bytes
                + vec_footprint(
                    &self.prepared_bin_scratch,
                    GPU_SCENE_FINALIZATION_SCRATCH_BASELINE,
                )
                .active_bytes
                + vec_footprint(
                    &self.vertex_slot_scratch,
                    GPU_SCENE_FINALIZATION_SCRATCH_BASELINE,
                )
                .active_bytes,
            0,
            "GPU Scene finalization scratch must be empty before capacity maintenance"
        );
        shrink_hash_map_by_family_ratio(
            &mut self.prepass_pipeline_l1,
            GPU_SCENE_PREPASS_PIPELINE_L1_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.validated_bin_scratch,
            GPU_SCENE_FINALIZATION_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.prepared_bin_scratch,
            GPU_SCENE_FINALIZATION_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.vertex_slot_scratch,
            GPU_SCENE_FINALIZATION_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
    }

    #[cfg(test)]
    pub(crate) fn reserve_finalization_capacity_probe(&mut self, slots: usize) {
        self.validated_bin_scratch.reserve(slots);
        self.prepared_bin_scratch.reserve(slots);
        self.vertex_slot_scratch.reserve(slots.saturating_mul(3));
    }

    #[cfg(test)]
    pub(crate) fn seed_prepass_pipeline_l1_capacity_probe(&mut self, entries: u32) {
        for ordinal in 0..entries {
            let discriminator = u64::from(ordinal) + 1;
            self.prepass_pipeline_l1.insert(
                test_prepass_pipeline_l1_key(discriminator),
                RenderPipelineId(ordinal),
            );
        }
    }

    #[cfg(test)]
    pub(crate) fn prepass_pipeline_l1_capacity_probe_signature(&self) -> (usize, u64) {
        let checksum = self
            .prepass_pipeline_l1
            .iter()
            .fold(0u64, |checksum, (key, pipeline)| {
                checksum.wrapping_add(key.shader_source_identity.rotate_left(pipeline.0 & 63))
            });
        (self.prepass_pipeline_l1.len(), checksum)
    }

    fn ensure_static_resources(&mut self, device: &wgpu::Device) {
        if self.object_layout.is_some() {
            debug_assert!(self.object_binding_interface_id.is_some());
            return;
        }
        self.object_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GpuScene Render Group 2 Layout"),
                entries: &[
                    storage_buffer(0, wgpu::ShaderStages::VERTEX, true),
                    storage_buffer(1, wgpu::ShaderStages::VERTEX, true),
                ],
            },
        )));
        self.object_binding_interface_id = Some(BindingInterfaceId::allocate_static());
    }

    fn object_layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.object_layout
            .as_ref()
            .expect("GpuScene render layout was not initialized")
    }

    fn object_binding_interface_id(&self) -> BindingInterfaceId {
        self.object_binding_interface_id
            .expect("GpuScene render binding interface was not initialized")
    }
}

#[cfg(test)]
fn test_prepass_pipeline_l1_key(discriminator: u64) -> PrepassPipelineL1Key {
    use slotmap::KeyData;

    let asset_version = u32::try_from(discriminator).expect("test discriminator exceeds u32");
    PrepassPipelineL1Key {
        geometry_handle: myth_resources::GeometryHandle::from(KeyData::from_ffi(
            discriminator * 2 + 1,
        )),
        geometry_asset_version: asset_version,
        geometry_asset_identity: discriminator * 11 + 1,
        geometry_layout_version: discriminator * 13 + 1,
        material_handle: myth_resources::MaterialHandle::from(KeyData::from_ffi(
            discriminator * 2 + 2,
        )),
        material_asset_version: asset_version,
        material_asset_identity: discriminator * 17 + 1,
        material_version: discriminator * 19 + 1,
        item_variants: asset_version,
        item_shader_features: crate::graph::extracted::ItemShaderFeatures::from_bits_retain(
            discriminator as u8 & 3,
        ),
        shader_source_identity: discriminator * 29 + 1,
        scene_defines_hash: discriminator * 31 + 1,
        pipeline_settings_version: discriminator * 37 + 1,
        global_binding_interface_id: BindingInterfaceId::from_raw(discriminator * 3 + 1),
        material_binding_interface_id: BindingInterfaceId::from_raw(discriminator * 3 + 2),
        object_binding_interface_id: BindingInterfaceId::from_raw(discriminator * 3 + 3),
        vertex_layout_id: discriminator * 41 + 1,
        bind_group_layout_ids: [
            discriminator * 43 + 1,
            discriminator * 43 + 2,
            discriminator * 43 + 3,
        ],
        depth_format: wgpu::TextureFormat::Depth32Float,
        topology: wgpu::PrimitiveTopology::TriangleList,
        cull_mode: Some(wgpu::Face::Back),
        front_face: wgpu::FrontFace::Ccw,
        needs_normal: discriminator & 1 != 0,
        needs_feature_id: discriminator & 2 != 0,
        needs_velocity: discriminator & 4 != 0,
    }
}

/// Finalize one view family in exact submission/view order.
///
/// Each DPE-live semantic submission retains the existing one-view
/// all-or-nothing transaction. A later view failure therefore has exactly the
/// same boundary as the previous Composer loop: already-complete earlier
/// views remain published, while the failing view exposes no partial stream.
pub(crate) fn finalize_gpu_scene_multi_bin_family(
    ctx: &mut PrepareContext<'_>,
    draw_plans: &mut DrawPlanStore,
    feature: &mut GpuSceneRenderFeature,
    plan: GpuSceneMultiBinFamilyPreparePlan<'_>,
    wgpu_ctx: &WgpuContext,
    camera: &RenderCamera,
) -> Result<bool, GpuSceneSubmissionFinalizeFailure> {
    let Ok(view_count) = usize::try_from(plan.culled.view_count()) else {
        return Err(GpuSceneSubmissionFinalizeFailure::DescriptorDrift);
    };
    if plan.submissions.len() != view_count {
        // Preserve the DPE boundary for an internally malformed but fully
        // dead family: dead semantic submissions perform no descriptor work.
        let any_live = plan.submissions.iter().copied().any(|submission| {
            ctx.scene_draws
                .scene_submissions
                .live_gpu_plan_for_submission(submission)
                .is_some()
        });
        return if any_live {
            Err(GpuSceneSubmissionFinalizeFailure::DescriptorDrift)
        } else {
            Ok(false)
        };
    }

    let mut any_live = false;
    for (view_index, submission) in plan.submissions.iter().copied().enumerate() {
        any_live |= finalize_gpu_scene_multi_bin_submission(
            ctx,
            draw_plans,
            feature,
            plan,
            submission,
            u32::try_from(view_index)
                .map_err(|_| GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?,
            wgpu_ctx,
            camera,
        )?;
    }
    Ok(any_live)
}

/// Finalize one DPE-live semantic plan as an all-or-nothing multi-bin stream.
///
/// Phase one is read-only and validates every candidate, stable BatchId and
/// exact Cull slice. Only a complete validation image may enter phase two,
/// where pipelines and range-addressed Group 2 BindGroups are materialized.
/// The retained scratch is finally projected through a cloneable exact-size
/// iterator into one transactional stream publication.
fn finalize_gpu_scene_multi_bin_submission(
    ctx: &mut PrepareContext<'_>,
    draw_plans: &mut DrawPlanStore,
    feature: &mut GpuSceneRenderFeature,
    plan: GpuSceneMultiBinFamilyPreparePlan<'_>,
    submission: SceneSubmissionPlanId,
    view_index: u32,
    wgpu_ctx: &WgpuContext,
    camera: &RenderCamera,
) -> Result<bool, GpuSceneSubmissionFinalizeFailure> {
    let Some(live) = ctx
        .scene_draws
        .scene_submissions
        .live_gpu_plan_for_submission(submission)
    else {
        // A dead semantic plan must not even clear retained scratch: no static
        // resource, physical lookup, pipeline, BG, or per-bin traversal occurs.
        return Ok(false);
    };
    let candidates = plan.candidates;
    if candidates.is_empty()
        || plan.culled.batch_count() as usize != candidates.len()
        || view_index >= plan.culled.view_count()
    {
        return Err(GpuSceneSubmissionFinalizeFailure::DescriptorDrift);
    }
    let first_draw = plan
        .culled
        .view_batch(view_index, 0)
        .ok_or(GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?;
    let descriptor = live.descriptor;
    if descriptor.representative() != candidates[0].representative
        || descriptor.instance_table() != plan.instance_table
        || descriptor.candidate_instances() != plan.instance_table
        || descriptor.visible_instances() != plan.culled.visible_instances
        || descriptor.indirect_arguments() != plan.culled.indirect_arguments
        || descriptor.indirect_offset() != first_draw.indirect_offset
        || !target_matches_submission_key(plan.target, descriptor.key())
    {
        return Err(GpuSceneSubmissionFinalizeFailure::DescriptorDrift);
    }

    match plan.target {
        GpuSceneSubmissionTarget::Main { global_state_key }
        | GpuSceneSubmissionTarget::Prepass {
            global_state_key, ..
        } => {
            if ctx.global_states.get(global_state_key).is_none() {
                return Err(GpuSceneSubmissionFinalizeFailure::GlobalStateUnavailable);
            }
        }
        GpuSceneSubmissionTarget::Shadow => {
            if ctx
                .scene_draws
                .registered_shadow_pipeline_layout()
                .is_none()
            {
                return Err(GpuSceneSubmissionFinalizeFailure::GlobalStateUnavailable);
            }
        }
    }

    // ── Phase 1: complete read-only validation ──────────────────────────
    feature.validated_bin_scratch.clear();
    feature.validated_bin_scratch.reserve(candidates.len());
    let view_visible_base = first_draw
        .visible_offset
        .checked_sub(candidates[0].partition_base)
        .ok_or(GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?;
    let mut total_vertex_slots = 0usize;
    for (candidate_index, candidate) in candidates.iter().copied().enumerate() {
        let draw = plan
            .culled
            .view_batch(
                view_index,
                u32::try_from(candidate_index)
                    .map_err(|_| GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?,
            )
            .ok_or(GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?;
        if draw.visible_instances != plan.culled.visible_instances
            || draw.indirect_arguments != plan.culled.indirect_arguments
            || draw.first_instance != 0
            || draw.visible_size != candidate.partition_size
            || draw.visible_offset
                != view_visible_base
                    .checked_add(candidate.partition_base)
                    .ok_or(GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?
            || draw.indirect_offset % 4 != 0
        {
            return Err(GpuSceneSubmissionFinalizeFailure::DescriptorDrift);
        }
        if candidate.batch_id.slot() != draw.batch_slot
            || candidate.batch_id.generation() != draw.batch_generation
            || candidate.batch_id.generation() == 0
        {
            return Err(GpuSceneSubmissionFinalizeFailure::DescriptorDrift);
        }
        let expected_base = u64::from(candidate.base_visible_index)
            .checked_mul(GPU_SCENE_VISIBLE_INSTANCE_SIZE)
            .ok_or(GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?;
        let expected_size = u64::from(candidate.capacity_records)
            .checked_mul(GPU_SCENE_VISIBLE_INSTANCE_SIZE)
            .ok_or(GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?;
        if candidate.member_count == 0
            || candidate.partition_base != expected_base
            || candidate.partition_size != expected_size
        {
            return Err(GpuSceneSubmissionFinalizeFailure::DescriptorDrift);
        }

        let scene_draws = &ctx.scene_draws;
        let extracted_source = scene_draws
            .render_lists
            .extracted_source(candidate.representative)
            .ok_or(GpuSceneSubmissionFinalizeFailure::RepresentativeDrift)?;
        let item = scene_draws
            .extracted_scene
            .render_items
            .get(extracted_source.index())
            .ok_or(GpuSceneSubmissionFinalizeFailure::RepresentativeDrift)?;
        if !candidate.key.matches_item(item) {
            return Err(GpuSceneSubmissionFinalizeFailure::RepresentativeDrift);
        }

        let geometry_id = scene_draws
            .geometry_frames
            .geometry_id(candidate.representative, item)
            .ok_or(GpuSceneSubmissionFinalizeFailure::GeometryUnavailable)?;
        let geometry = scene_draws
            .geometry_frames
            .geometry(geometry_id)
            .ok_or(GpuSceneSubmissionFinalizeFailure::GeometryUnavailable)?;
        let geometry_physical = scene_draws
            .geometry_frames
            .physical(geometry_id)
            .ok_or(GpuSceneSubmissionFinalizeFailure::GeometryUnavailable)?;
        if !geometry_matches_multi_bin_candidate(
            geometry,
            geometry_physical,
            item,
            candidate,
            scene_draws
                .geometry_frames
                .pipeline_state_is_current(geometry_id),
        ) || geometry_physical.vertex_buffers.is_empty()
        {
            return Err(GpuSceneSubmissionFinalizeFailure::GeometryDrift);
        }

        let material = scene_draws
            .material_frames
            .prepared_material(candidate.representative, item)
            .ok_or(GpuSceneSubmissionFinalizeFailure::MaterialUnavailable)?;
        if !material_matches_multi_bin_candidate(material, candidate)
            || !material.pipeline_state_is_current()
        {
            return Err(GpuSceneSubmissionFinalizeFailure::MaterialDrift);
        }
        let material_pipeline = material.pipeline();
        if material_pipeline.use_transmission()
            || matches!(
                material_pipeline.settings().alpha_mode,
                AlphaMode::Blend | AlphaMode::BlendMask
            )
        {
            return Err(GpuSceneSubmissionFinalizeFailure::UnsupportedMaterialState);
        }

        total_vertex_slots = total_vertex_slots
            .checked_add(geometry_physical.vertex_buffers.len())
            .ok_or(GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?;
        feature.validated_bin_scratch.push(ValidatedGpuSceneBin {
            candidate_index: u32::try_from(candidate_index)
                .map_err(|_| GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?,
            geometry: geometry_id,
            material: material.id(),
            draw,
        });
    }
    debug_assert_eq!(feature.validated_bin_scratch.len(), candidates.len());

    let instance_table_physical = ctx
        .views
        .persistent_buffer_physical_token(plan.instance_table)
        .ok_or(GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?;

    // ── Phase 2: physical materialization after complete validation ─────
    feature.ensure_static_resources(ctx.device);
    feature.prepared_bin_scratch.clear();
    feature.vertex_slot_scratch.clear();
    feature.prepared_bin_scratch.reserve(candidates.len());
    feature.vertex_slot_scratch.reserve(total_vertex_slots);
    let indirect_arguments = draw_plans.intern_buffer(
        ctx.views.get_buffer(plan.culled.indirect_arguments),
        ctx.views
            .get_physical_buffer_uid(plan.culled.indirect_arguments),
    );

    for validated_index in 0..feature.validated_bin_scratch.len() {
        let validated = feature.validated_bin_scratch[validated_index];
        let candidate = candidates[validated.candidate_index as usize];
        let (pipeline, vertex_start, vertex_end, index_buffer, index_format, material_bind_group) = {
            let scene_draws = &ctx.scene_draws;
            let extracted_source = scene_draws
                .render_lists
                .extracted_source(candidate.representative)
                .expect("validated GPU Scene representative disappeared");
            let item = &scene_draws.extracted_scene.render_items[extracted_source.index()];
            let geometry = scene_draws
                .geometry_frames
                .geometry(validated.geometry)
                .expect("validated GPU Scene geometry disappeared");
            let geometry_physical = scene_draws
                .geometry_frames
                .physical(validated.geometry)
                .expect("validated GPU Scene physical geometry disappeared");
            let material = scene_draws
                .material_frames
                .material(validated.material)
                .expect("validated GPU Scene material disappeared");

            let pipeline = match plan.target {
                GpuSceneSubmissionTarget::Main { global_state_key } => {
                    let gpu_world = ctx
                        .global_states
                        .get(global_state_key)
                        .expect("validated GPU Scene global state disappeared");
                    prepare_gpu_scene_main_pipeline(
                        feature,
                        ctx.pipeline_cache,
                        ctx.shader_manager,
                        ctx.device,
                        wgpu_ctx,
                        ctx.system_textures,
                        scene_draws.extracted_scene,
                        item,
                        geometry,
                        geometry_physical,
                        material,
                        gpu_world,
                        camera,
                    )
                }
                GpuSceneSubmissionTarget::Prepass {
                    global_state_key,
                    needs_normal,
                    needs_feature_id,
                    needs_velocity,
                } => {
                    let gpu_world = ctx
                        .global_states
                        .get(global_state_key)
                        .expect("validated GPU Scene global state disappeared");
                    prepare_gpu_scene_prepass_pipeline(
                        feature,
                        ctx.pipeline_cache,
                        ctx.shader_manager,
                        ctx.device,
                        wgpu_ctx,
                        scene_draws.extracted_scene,
                        item,
                        geometry,
                        geometry_physical,
                        material,
                        gpu_world,
                        needs_normal,
                        needs_feature_id,
                        needs_velocity,
                    )
                }
                GpuSceneSubmissionTarget::Shadow => {
                    let shadow_layout = scene_draws
                        .registered_shadow_pipeline_layout()
                        .expect("validated GPU Scene shadow layout disappeared");
                    prepare_gpu_scene_shadow_pipeline(
                        feature,
                        ctx.pipeline_cache,
                        ctx.shader_manager,
                        ctx.device,
                        wgpu_ctx,
                        item,
                        geometry,
                        geometry_physical,
                        material,
                        shadow_layout,
                    )
                }
            };

            let vertex_start = u32::try_from(feature.vertex_slot_scratch.len())
                .expect("GPU Scene vertex-slot scratch exceeds u32");
            feature
                .vertex_slot_scratch
                .extend_from_slice(&geometry_physical.vertex_buffers);
            let vertex_end = u32::try_from(feature.vertex_slot_scratch.len())
                .expect("GPU Scene vertex-slot scratch exceeds u32");
            let (index_buffer, index_format, _) = geometry_physical
                .index_buffer
                .expect("validated GPU Scene geometry lost its index Buffer");
            let material_bind_group = draw_plans.intern_bind_group(
                &material.interface().bind_group,
                material.interface().bind_group_id,
            );
            (
                pipeline,
                vertex_start,
                vertex_end,
                index_buffer,
                index_format,
                material_bind_group,
            )
        };

        let visible_size = wgpu::BufferSize::new(validated.draw.visible_size)
            .ok_or(GpuSceneSubmissionFinalizeFailure::DescriptorDrift)?;
        let object_bind_group = ctx
            .build_bind_group(
                feature.object_layout(),
                Some("GpuScene Multi-Bin Render Group 2 Bind Group"),
            )
            .bind_buffer(0, plan.instance_table)
            .bind_buffer_range(
                1,
                validated.draw.visible_instances,
                validated.draw.visible_offset,
                visible_size,
            )
            .build();
        feature.prepared_bin_scratch.push(PreparedGpuSceneBin {
            pipeline,
            vertex_start,
            vertex_end,
            index_buffer,
            index_format,
            material_bind_group,
            object_bind_group,
            indirect_offset: validated.draw.indirect_offset,
        });
    }

    let prepared = &feature.prepared_bin_scratch;
    let vertices = &feature.vertex_slot_scratch;
    let inputs = prepared
        .iter()
        .map(move |batch| GpuIndexedIndirectBatchInput {
            pipeline: batch.pipeline,
            vertex_buffers: &vertices[batch.vertex_start as usize..batch.vertex_end as usize],
            index_buffer: batch.index_buffer,
            index_format: batch.index_format,
            material_bind_group: batch.material_bind_group,
            object_bind_group: batch.object_bind_group,
            indirect_arguments,
            indirect_offset: batch.indirect_offset,
            instance_table_physical,
        });
    ctx.scene_draws
        .scene_submissions
        .publish_gpu_indexed_indirect_stream(live.submission, live.plan, inputs)
        .map(|_| true)
        .ok_or(GpuSceneSubmissionFinalizeFailure::BatchPublicationRejected)
}

fn geometry_matches_multi_bin_candidate(
    geometry: &GeometryFrameRecord,
    physical: &GeometryFramePhysical,
    item: &ExtractedRenderItem,
    candidate: GpuSceneMultiBinCandidate,
    pipeline_state_is_current: bool,
) -> bool {
    let Some((_, index_format, index_count)) = physical.index_buffer else {
        return false;
    };
    candidate.key.matches_item(item)
        && geometry.key.handle == candidate.key.geometry
        && geometry.key.asset_version == candidate.key.geometry_asset_version
        && geometry.key.asset_identity == candidate.key.geometry_asset_identity
        && geometry.key.structure_version == candidate.key.geometry_structure_version
        && geometry.key.layout_version == candidate.key.geometry_layout_version
        && geometry.topology == candidate.key.geometry_topology
        && index_format == candidate.key.index_format
        && index_count == candidate.key.index_count
        && physical.instance_range == (0..1)
        && pipeline_state_is_current
}

fn material_matches_multi_bin_candidate(
    material: PreparedMaterialFrame<'_>,
    candidate: GpuSceneMultiBinCandidate,
) -> bool {
    material.handle() == candidate.key.material
        && material.asset_version() == candidate.key.material_asset_version
        && material.asset_identity() == candidate.key.material_asset_identity
        && material.material_version() == candidate.key.material_version
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn prepare_gpu_scene_main_pipeline(
    feature: &mut GpuSceneRenderFeature,
    pipeline_cache: &mut crate::pipeline::PipelineCache,
    shader_manager: &mut crate::pipeline::ShaderManager,
    device: &wgpu::Device,
    wgpu_ctx: &WgpuContext,
    system_textures: &crate::core::gpu::SystemTextures,
    extracted_scene: &crate::graph::ExtractedScene,
    item: &ExtractedRenderItem,
    geometry: &GeometryFrameRecord,
    geometry_physical: &GeometryFramePhysical,
    material: PreparedMaterialFrame<'_>,
    gpu_world: &GpuGlobalState,
    camera: &RenderCamera,
) -> RenderPipelineId {
    let material_pipeline = material.pipeline();
    let material_settings = material_pipeline.settings();
    let use_clustered_shading = extracted_scene
        .scene_variants
        .contains(SceneFeatures::USE_CLUSTERED_SHADING);
    let final_a2c_enable = matches!(material_settings.alpha_mode, AlphaMode::Mask)
        && material_settings.alpha_to_coverage
        && wgpu_ctx.msaa_samples > 1;
    let scene_defines_hash = extracted_scene.scene_defines.compute_hash();
    let shader_source = material_shader_source(material_pipeline);
    let shader_source_identity = shader_manager.source_identity_hash(shader_source);
    let fast_key = FastPipelineKey {
        material_handle: material.handle(),
        material_asset_version: material.asset_version(),
        material_asset_identity: material.asset_identity(),
        material_version: material.material_version(),
        geometry_handle: geometry.key.handle,
        geometry_asset_version: geometry.key.asset_version,
        geometry_asset_identity: geometry.key.asset_identity,
        geometry_version: geometry.key.layout_version,
        vertex_layout_id: geometry_physical.layout_id,
        instance_variants: item.item_variant_flags,
        item_shader_features: item.item_shader_features,
        shader_source_identity,
        global_layout_id: gpu_world.layout_id,
        global_binding_interface_id: gpu_world.binding_interface_id,
        material_layout_id: material.interface().layout_id,
        material_binding_interface_id: material.interface().binding_interface_id,
        object_layout_id: feature.object_layout().id(),
        object_binding_interface_id: feature.object_binding_interface_id(),
        scene_variants: extracted_scene.scene_variants,
        scene_defines_hash,
        taa_enabled: matches!(camera.aa_mode, AntiAliasingMode::TAA { .. }),
        pipeline_settings_version: wgpu_ctx.pipeline_settings_version,
        #[cfg(feature = "debug_view")]
        debug_view_mode: camera.debug_view.mode,
    };
    if let Some(pipeline) = pipeline_cache.get_pipeline_fast(fast_key) {
        return pipeline;
    }

    let mut flags = PipelineFlags::empty();
    let mut options = ShaderCompilationOptions::from_merged(
        material_pipeline.shader_defines(),
        geometry.shader_defines(),
        &extracted_scene.scene_defines,
        item.item_shader_defines(),
    );
    options.add_define("USE_GPU_SCENE", "1");
    if use_clustered_shading {
        options.add_define("USE_CLUSTERED_SHADING", "1");
    }
    if wgpu_ctx.render_path.supports_post_processing() {
        options.add_define("HDR", "1");
    }
    let has_specular_data_mrt = wgpu_ctx.render_path.supports_post_processing()
        && extracted_scene
            .scene_variants
            .intersects(SceneFeatures::USE_SSS | SceneFeatures::USE_SSR);
    let has_material_data_mrt = wgpu_ctx.render_path.supports_post_processing()
        && extracted_scene
            .scene_variants
            .intersects(SceneFeatures::USE_SSGI | SceneFeatures::USE_SSR);
    if has_specular_data_mrt {
        options.add_define("HAS_MRT_SPECULAR_DATA", "1");
    }
    if has_material_data_mrt {
        options.add_define("HAS_MRT_MATERIAL_DATA", "1");
    }
    if final_a2c_enable {
        options.add_define("ALPHA_TO_COVERAGE", "1");
        flags |= PipelineFlags::ALPHA_TO_COVERAGE;
    }
    let use_depth_pre = wgpu_ctx.msaa_samples == 1 && wgpu_ctx.render_path.requires_z_prepass();
    let depth_write = if use_depth_pre {
        false
    } else {
        material_settings.depth_write
    };
    if depth_write {
        flags |= PipelineFlags::DEPTH_WRITE;
    }
    let depth_compare = if material_settings.depth_test {
        if use_depth_pre {
            wgpu::CompareFunction::Equal
        } else {
            wgpu::CompareFunction::Greater
        }
    } else {
        wgpu::CompareFunction::Always
    };
    let mut color_targets: SmallVec<[ColorTargetKey; 3]> =
        smallvec::smallvec![ColorTargetKey::from(wgpu::ColorTargetState {
            format: wgpu_ctx
                .render_path
                .main_color_format(wgpu_ctx.surface_view_format),
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
    if has_specular_data_mrt {
        color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
            format: crate::HDR_TEXTURE_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }));
    }
    if has_material_data_mrt {
        color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }));
    }
    let screen_layout = if use_clustered_shading {
        &system_textures.screen_layout_clustered
    } else {
        &system_textures.screen_layout
    };
    let shader_hash = shader_manager.pipeline_shader_hash(shader_source, &options);
    let canonical_key = GraphicsPipelineKey {
        shader_hash,
        vertex_layout_id: geometry_physical.layout_id,
        bind_group_layout_ids: [
            gpu_world.layout_id,
            material.interface().layout_id,
            feature.object_layout().id(),
            screen_layout.id(),
        ],
        topology: geometry.topology,
        cull_mode: side_cull_mode(material_settings.side),
        front_face: item_front_face(item),
        depth_compare,
        blend_state: None::<BlendStateKey>,
        color_targets,
        depth_format: wgpu_ctx.depth_format,
        sample_count: wgpu_ctx.msaa_samples,
        flags,
    };
    let pipeline = pipeline_cache.get_or_create_graphics(
        device,
        shader_manager,
        shader_source,
        &canonical_key,
        &options,
        &geometry_physical.layout_info,
        &material.interface().layout,
        &material.interface().binding_wgsl,
        feature.object_layout(),
        GPU_SCENE_OBJECT_BINDING_WGSL,
        gpu_world,
        screen_layout,
    );
    pipeline_cache.insert_pipeline_fast(fast_key, pipeline);
    pipeline
}

#[inline]
fn target_matches_submission_key(
    target: GpuSceneSubmissionTarget,
    key: SceneSubmissionKey,
) -> bool {
    match (target, key) {
        (GpuSceneSubmissionTarget::Main { .. }, SceneSubmissionKey::Opaque)
        | (GpuSceneSubmissionTarget::Shadow, SceneSubmissionKey::Shadow { .. }) => true,
        (
            GpuSceneSubmissionTarget::Prepass {
                needs_normal,
                needs_feature_id,
                needs_velocity,
                ..
            },
            SceneSubmissionKey::Prepass {
                needs_normal: key_normal,
                needs_feature_id: key_feature_id,
                needs_velocity: key_velocity,
            },
        ) => {
            needs_normal == key_normal
                && needs_feature_id == key_feature_id
                && needs_velocity == key_velocity
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn prepare_gpu_scene_prepass_pipeline(
    feature: &mut GpuSceneRenderFeature,
    pipeline_cache: &mut crate::pipeline::PipelineCache,
    shader_manager: &mut crate::pipeline::ShaderManager,
    device: &wgpu::Device,
    wgpu_ctx: &WgpuContext,
    extracted_scene: &crate::graph::ExtractedScene,
    item: &ExtractedRenderItem,
    geometry: &GeometryFrameRecord,
    geometry_physical: &GeometryFramePhysical,
    material: PreparedMaterialFrame<'_>,
    gpu_world: &GpuGlobalState,
    needs_normal: bool,
    needs_feature_id: bool,
    needs_velocity: bool,
) -> RenderPipelineId {
    let shader_source = ShaderSource::File("entry/utility/depth_prepass");
    let shader_source_identity = shader_manager.source_identity_hash(shader_source);
    let object_layout = feature.object_layout().clone();
    let object_binding_interface_id = feature.object_binding_interface_id();
    let material_interface = material.interface();
    let material_settings = material.pipeline().settings();
    let key = PrepassPipelineL1Key {
        geometry_handle: geometry.key.handle,
        geometry_asset_version: geometry.key.asset_version,
        geometry_asset_identity: geometry.key.asset_identity,
        geometry_layout_version: geometry.key.layout_version,
        material_handle: material.handle(),
        material_asset_version: material.asset_version(),
        material_asset_identity: material.asset_identity(),
        material_version: material.material_version(),
        item_variants: item.item_variant_flags,
        item_shader_features: item.item_shader_features,
        shader_source_identity,
        scene_defines_hash: extracted_scene.scene_defines.compute_hash(),
        pipeline_settings_version: wgpu_ctx.pipeline_settings_version,
        global_binding_interface_id: gpu_world.binding_interface_id,
        material_binding_interface_id: material_interface.binding_interface_id,
        object_binding_interface_id,
        vertex_layout_id: geometry_physical.layout_id,
        bind_group_layout_ids: [
            gpu_world.layout_id,
            material_interface.layout_id,
            object_layout.id(),
        ],
        depth_format: wgpu_ctx.depth_format,
        topology: geometry.topology,
        cull_mode: side_cull_mode(material_settings.side),
        front_face: item_front_face(item),
        needs_normal,
        needs_feature_id,
        needs_velocity,
    };
    if let Some(&pipeline) = feature.prepass_pipeline_l1.get(&key) {
        return pipeline;
    }

    let mut options = ShaderCompilationOptions::from_merged(
        material.pipeline().shader_defines(),
        geometry.shader_defines(),
        &extracted_scene.scene_defines,
        item.item_shader_defines(),
    );
    options.add_define("USE_GPU_SCENE", "1");
    options.add_define("IS_PREPASS", "1");
    if needs_normal {
        options.add_define("OUTPUT_NORMAL", "1");
    }
    if needs_velocity {
        options.add_define("HAS_VELOCITY_TARGET", "1");
    }
    if needs_feature_id {
        options.add_define("OUTPUT_FEATURE_ID", "1");
    }
    options.inject_code(
        "vertex_input_code",
        &geometry_physical.layout_info.vertex_input_code,
    );
    options.inject_code(
        "binding_code",
        format!(
            "{}\n{}\n{}",
            gpu_world.binding_wgsl.as_str(),
            material_interface.binding_wgsl,
            GPU_SCENE_OBJECT_BINDING_WGSL,
        ),
    );
    options.inject_code(
        "scene_lighting_structs",
        myth_resources::uniforms::scene_lighting_structs_wgsl(),
    );
    let (shader_module, shader_hash) =
        shader_manager.get_or_compile(device, shader_source, &options);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("GpuScene Prepass Pipeline Layout"),
        bind_group_layouts: &[
            Some(&gpu_world.layout),
            Some(&material_interface.layout),
            Some(&object_layout),
        ],
        immediate_size: 0,
    });
    let vertex_buffers_layout: Vec<_> = geometry_physical
        .layout_info
        .buffers
        .iter()
        .map(|layout| layout.as_wgpu())
        .collect();
    let mut color_targets = smallvec::smallvec![];
    if needs_normal {
        color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
            format: NORMAL_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }));
    }
    if needs_feature_id {
        color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
            format: FEATURE_ID_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }));
    }
    if needs_velocity {
        color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
            format: VELOCITY_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }));
    }
    let canonical_key = SimpleGeometryPipelineKey {
        shader_hash,
        vertex_layout_id: geometry_physical.layout_id,
        bind_group_layout_ids: key.bind_group_layout_ids,
        color_targets,
        depth_stencil: DepthStencilKey::from(wgpu::DepthStencilState {
            format: key.depth_format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        topology: key.topology,
        cull_mode: key.cull_mode,
        front_face: key.front_face,
        sample_count: 1,
    };
    let pipeline = pipeline_cache.get_or_create_simple_geometry(
        device,
        shader_module,
        &layout,
        &canonical_key,
        "GpuScene Prepass Pipeline",
        &vertex_buffers_layout,
    );
    feature.prepass_pipeline_l1.insert(key, pipeline);
    pipeline
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn prepare_gpu_scene_shadow_pipeline(
    feature: &GpuSceneRenderFeature,
    pipeline_cache: &mut crate::pipeline::PipelineCache,
    shader_manager: &mut crate::pipeline::ShaderManager,
    device: &wgpu::Device,
    wgpu_ctx: &WgpuContext,
    item: &ExtractedRenderItem,
    geometry: &GeometryFrameRecord,
    geometry_physical: &GeometryFramePhysical,
    material: PreparedMaterialFrame<'_>,
    shadow_layout: &Tracked<wgpu::BindGroupLayout>,
) -> RenderPipelineId {
    let shader_source = ShaderSource::File("entry/utility/depth_prepass");
    let shader_source_identity = shader_manager.source_identity_hash(shader_source);
    let material_interface = material.interface();
    let fast_key = FastShadowPipelineKey {
        material_handle: material.handle(),
        material_asset_version: material.asset_version(),
        material_asset_identity: material.asset_identity(),
        material_version: material.material_version(),
        geometry_handle: geometry.key.handle,
        geometry_asset_version: geometry.key.asset_version,
        geometry_asset_identity: geometry.key.asset_identity,
        geometry_version: geometry.key.layout_version,
        vertex_layout_id: geometry_physical.layout_id,
        instance_variants: item.item_variant_flags,
        item_shader_features: item.item_shader_features,
        shader_source_identity,
        shadow_global_layout_id: shadow_layout.id(),
        material_layout_id: material_interface.layout_id,
        material_binding_interface_id: material_interface.binding_interface_id,
        object_layout_id: feature.object_layout().id(),
        object_binding_interface_id: feature.object_binding_interface_id(),
        pipeline_settings_version: wgpu_ctx.pipeline_settings_version,
    };
    if let Some(pipeline) = pipeline_cache.get_shadow_pipeline_fast(fast_key) {
        return pipeline;
    }

    let mut options = ShaderCompilationOptions::from_merged(
        material.pipeline().shader_defines(),
        geometry.shader_defines(),
        &myth_resources::shader_defines::ShaderDefines::new(),
        item.item_shader_defines(),
    );
    options.add_define("USE_GPU_SCENE", "1");
    options.add_define("SHADOW_PASS", "1");
    options.inject_code(
        "vertex_input_code",
        &geometry_physical.layout_info.vertex_input_code,
    );
    options.inject_code(
        "binding_code",
        format!(
            "{}\n{}\n{}",
            SHADOW_BINDING_WGSL, material_interface.binding_wgsl, GPU_SCENE_OBJECT_BINDING_WGSL,
        ),
    );
    options.inject_code(
        "scene_lighting_structs",
        myth_resources::uniforms::scene_lighting_structs_wgsl(),
    );
    let (shader_module, shader_hash) =
        shader_manager.get_or_compile(device, shader_source, &options);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("GpuScene Shadow Pipeline Layout"),
        bind_group_layouts: &[
            Some(shadow_layout),
            Some(&material_interface.layout),
            Some(feature.object_layout()),
        ],
        immediate_size: 0,
    });
    let vertex_buffers_layout: Vec<_> = geometry_physical
        .layout_info
        .buffers
        .iter()
        .map(|layout| layout.as_wgpu())
        .collect();
    let canonical_key = SimpleGeometryPipelineKey {
        shader_hash,
        vertex_layout_id: geometry_physical.layout_id,
        bind_group_layout_ids: [
            shadow_layout.id(),
            material_interface.layout_id,
            feature.object_layout().id(),
        ],
        color_targets: smallvec::smallvec![],
        depth_stencil: DepthStencilKey::from(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        topology: geometry.topology,
        cull_mode: Some(wgpu::Face::Front),
        front_face: item_front_face(item),
        sample_count: 1,
    };
    let pipeline = pipeline_cache.get_or_create_simple_geometry(
        device,
        shader_module,
        &layout,
        &canonical_key,
        "GpuScene Shadow Pipeline",
        &vertex_buffers_layout,
    );
    pipeline_cache.insert_shadow_pipeline_fast(fast_key, pipeline);
    pipeline
}

fn material_shader_source(pipeline: &MaterialPipelineSnapshot) -> ShaderSource<'_> {
    if let Some(template_source) = pipeline.shader_template() {
        ShaderSource::Inline {
            name: pipeline.shader_name(),
            source: template_source,
            mode: pipeline.shader_template_mode(),
        }
    } else {
        ShaderSource::File(pipeline.shader_name())
    }
}

#[inline]
fn side_cull_mode(side: Side) -> Option<wgpu::Face> {
    match side {
        Side::Front => Some(wgpu::Face::Back),
        Side::Back => Some(wgpu::Face::Front),
        Side::Double => None,
    }
}

#[inline]
fn item_front_face(item: &ExtractedRenderItem) -> wgpu::FrontFace {
    if item.item_variant_flags & 0x1 != 0 {
        wgpu::FrontFace::Cw
    } else {
        wgpu::FrontFace::Ccw
    }
}
