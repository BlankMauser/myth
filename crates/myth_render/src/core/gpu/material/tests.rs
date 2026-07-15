use super::*;
use crate::settings::{RendererInitConfig, RendererSettings};
use crate::test_support::SharedTestContext as WgpuContext;
use slotmap::KeyData;

fn test_resource_manager() -> ResourceManager {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1)
}

fn test_material_handle(index: u32) -> MaterialHandle {
    test_material_handle_generation(index, 1)
}

fn test_material_handle_generation(index: u32, version: u32) -> MaterialHandle {
    MaterialHandle::from(KeyData::from_ffi(
        (u64::from(version) << 32) | u64::from(index),
    ))
}

fn resource_ids(ids: &[u64]) -> ResourceIdSet {
    let mut result = ResourceIdSet::with_capacity(ids.len());
    for &id in ids {
        result.push(id);
    }
    result
}

fn test_pipeline_snapshot() -> Arc<MaterialPipelineSnapshot> {
    Arc::new(MaterialPipelineSnapshot::capture(&Material::new_unlit(
        glam::Vec4::ONE,
    )))
}

fn clone_gpu_material(source: &GpuMaterial, bind_group_id: u64) -> GpuMaterial {
    GpuMaterial {
        asset_version: source.asset_version,
        asset_identity: source.asset_identity,
        bind_group: source.bind_group.clone(),
        bind_group_id,
        layout: source.layout.clone(),
        layout_id: source.layout_id,
        layout_hash: source.layout_hash,
        binding_wgsl: Arc::clone(&source.binding_wgsl),
        binding_interface_id: source.binding_interface_id,
        pipeline_snapshot: Arc::clone(&source.pipeline_snapshot),
        canonical_binding_stamp: source.canonical_binding_stamp,
        resource_ids: source.resource_ids.clone(),
        version: source.version,
        last_used_frame: source.last_used_frame,
        last_verified_frame: source.last_verified_frame,
    }
}

fn stage_rebuild(
    resources: &mut ResourceManager,
    batch: &mut PlannedMaterialCandidateBatch,
    handle: MaterialHandle,
    ids: &[u64],
    observed_bind_group_id: Option<u64>,
) -> PlannedMaterialCandidateId {
    const ASSET_VERSION: u32 = 7;
    const ASSET_IDENTITY: u64 = 11;
    const MATERIAL_VERSION: u64 = 13;

    let builder = ResourceBuilder::new();
    let candidate = Box::new(resources.build_material_from_builder(
        &builder,
        resource_ids(ids),
        ASSET_VERSION,
        ASSET_IDENTITY,
        MATERIAL_VERSION,
        false,
        test_pipeline_snapshot(),
        None,
    ));
    let private_slot = batch.private_materials.len();
    batch.private_materials.push(Some(candidate));
    batch.push(
        handle,
        ASSET_VERSION,
        ASSET_IDENTITY,
        MATERIAL_VERSION,
        false,
        PlannedMaterialCandidateSource::Rebuild {
            observed_bind_group_id,
            private_slot,
        },
    )
}

#[test]
fn candidate_publication_reuses_equivalent_winner_and_rejects_conflicts() {
    let mut resources = test_resource_manager();

    let equivalent_handle = test_material_handle(1);
    let mut equivalent_batch = PlannedMaterialCandidateBatch::new();
    let equivalent_candidate = stage_rebuild(
        &mut resources,
        &mut equivalent_batch,
        equivalent_handle,
        &[101, 102],
        None,
    );
    let private = equivalent_batch.private_materials[0]
        .as_ref()
        .expect("private candidate missing");
    let winner_bind_group_id = private.bind_group_id.wrapping_add(10_000);
    let equivalent_winner = clone_gpu_material(private, winner_bind_group_id);
    resources
        .resources
        .insert_material(equivalent_handle, equivalent_winner);

    resources.publish_planned_material_candidates(&mut equivalent_batch, &[true]);

    assert!(equivalent_batch.result(equivalent_candidate));
    assert!(equivalent_batch.all_published());
    assert_eq!(
        resources
            .get_material(equivalent_handle)
            .expect("equivalent winner disappeared")
            .bind_group_id,
        winner_bind_group_id,
        "an exact cache winner must be reused instead of overwritten"
    );
    assert!(
        equivalent_batch.private_materials[0].is_some(),
        "reusing a winner must leave the unpublished owner with the batch"
    );

    let conflicting_handle = test_material_handle(2);
    let mut conflicting_batch = PlannedMaterialCandidateBatch::new();
    let conflicting_candidate = stage_rebuild(
        &mut resources,
        &mut conflicting_batch,
        conflicting_handle,
        &[201],
        None,
    );
    let private = conflicting_batch.private_materials[0]
        .as_ref()
        .expect("private candidate missing");
    let mut conflicting_winner = clone_gpu_material(private, private.bind_group_id + 10_000);
    conflicting_winner.resource_ids = resource_ids(&[202]);
    let conflicting_winner_id = conflicting_winner.bind_group_id;
    resources
        .resources
        .insert_material(conflicting_handle, conflicting_winner);

    resources.publish_planned_material_candidates(&mut conflicting_batch, &[true]);

    assert!(!conflicting_batch.result(conflicting_candidate));
    assert!(conflicting_batch.all_published());
    let retained_conflict = resources
        .get_material(conflicting_handle)
        .expect("conflicting winner disappeared");
    assert_eq!(retained_conflict.bind_group_id, conflicting_winner_id);
    assert_eq!(retained_conflict.resource_ids, resource_ids(&[202]));
    assert!(
        conflicting_batch.private_materials[0].is_some(),
        "a rejected private owner must remain batch-scoped"
    );
}

#[test]
fn stale_candidate_never_overwrites_the_observed_cache_entry() {
    let mut resources = test_resource_manager();
    let handle = test_material_handle(3);
    let builder = ResourceBuilder::new();
    let cached = resources.build_material_from_builder(
        &builder,
        resource_ids(&[301]),
        7,
        11,
        13,
        false,
        test_pipeline_snapshot(),
        None,
    );
    let cached_bind_group_id = cached.bind_group_id;
    resources.resources.insert_material(handle, cached);

    let mut batch = PlannedMaterialCandidateBatch::new();
    let candidate = stage_rebuild(
        &mut resources,
        &mut batch,
        handle,
        &[302],
        Some(cached_bind_group_id),
    );
    resources.publish_planned_material_candidates(&mut batch, &[false]);

    assert!(!batch.result(candidate));
    assert!(batch.all_published());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            resources.publish_planned_material_candidate(&mut batch, candidate, true);
        }))
        .is_err(),
        "reading a result must not make the candidate publishable again"
    );

    let mut chunked = PlannedMaterialCandidateBatch::new();
    let ids: Vec<_> = (0..65_u32)
        .map(|version| chunked.push_unavailable(handle, version, u64::from(version), 13))
        .collect();
    for (index, &id) in ids.iter().enumerate() {
        assert!(!resources.publish_planned_material_candidate(
            &mut chunked,
            id,
            index % 2 == 0,
        ));
    }
    assert!(chunked.all_published());
    assert!(ids.iter().all(|&id| !chunked.result(id)));
    let retained = resources
        .get_material(handle)
        .expect("stale publication removed the cache entry");
    assert_eq!(retained.bind_group_id, cached_bind_group_id);
    assert_eq!(retained.resource_ids, resource_ids(&[301]));
    assert!(batch.private_materials[0].is_some());
}

#[test]
fn accepted_rebuild_is_allocation_free_and_defers_the_retired_owner() {
    let mut resources = test_resource_manager();
    let handle = test_material_handle(4_096);
    let builder = ResourceBuilder::new();
    let cached = resources.build_material_from_builder(
        &builder,
        resource_ids(&[401]),
        7,
        11,
        13,
        false,
        test_pipeline_snapshot(),
        None,
    );
    let cached_bind_group_id = cached.bind_group_id;
    resources.resources.insert_material(handle, cached);

    let mut batch = PlannedMaterialCandidateBatch::new();
    let candidate = stage_rebuild(
        &mut resources,
        &mut batch,
        handle,
        &[402],
        Some(cached_bind_group_id),
    );
    let candidate_bind_group_id = batch.private_materials[0]
        .as_ref()
        .expect("private candidate missing")
        .bind_group_id;
    let slot_capacity = resources.resources.material_slot_capacity();

    resources.publish_planned_material_candidates(&mut batch, &[true]);

    assert!(batch.result(candidate));
    assert_eq!(resources.resources.material_slot_capacity(), slot_capacity);
    assert_eq!(
        resources
            .get_material(handle)
            .expect("published Material missing")
            .bind_group_id,
        candidate_bind_group_id
    );
    assert_eq!(
        batch.private_materials[0]
            .as_ref()
            .expect("retired owner must remain batch-scoped")
            .bind_group_id,
        cached_bind_group_id
    );

    batch.clear();
    assert!(batch.is_empty());
}

#[test]
fn newer_handle_generation_retires_the_obsolete_owner_before_publication() {
    let mut resources = test_resource_manager();
    let old_handle = test_material_handle_generation(8_192, 1);
    let new_handle = test_material_handle_generation(8_192, 3);
    let builder = ResourceBuilder::new();
    let obsolete = resources.build_material_from_builder(
        &builder,
        resource_ids(&[501]),
        7,
        11,
        13,
        false,
        test_pipeline_snapshot(),
        None,
    );
    resources.resources.insert_material(old_handle, obsolete);
    assert!(resources.get_material(old_handle).is_some());

    let mut batch = PlannedMaterialCandidateBatch::new();
    let candidate = stage_rebuild(&mut resources, &mut batch, new_handle, &[502], None);
    let mut current = [true];
    resources.reserve_planned_material_candidate_slots(&batch, &mut current);
    assert_eq!(current, [true]);

    assert!(resources.get_material(old_handle).is_none());
    assert!(resources.get_material(new_handle).is_none());
    let slot_capacity = resources.resources.material_slot_capacity();

    assert!(resources.publish_planned_material_candidate(&mut batch, candidate, true));

    assert!(batch.result(candidate));
    assert_eq!(resources.resources.material_slot_capacity(), slot_capacity);
    assert!(resources.get_material(new_handle).is_some());
    assert!(
        batch.private_materials[0].is_none(),
        "the obsolete generation must have been retired before the publication gate"
    );
}

#[test]
fn stale_handle_generation_cannot_displace_a_newer_cached_owner() {
    let mut resources = test_resource_manager();
    let stale_handle = test_material_handle_generation(12_288, 1);
    let current_handle = test_material_handle_generation(12_288, 3);
    let builder = ResourceBuilder::new();
    let current = resources.build_material_from_builder(
        &builder,
        resource_ids(&[551]),
        7,
        11,
        13,
        false,
        test_pipeline_snapshot(),
        None,
    );
    let current_bind_group_id = current.bind_group_id;
    resources.resources.insert_material(current_handle, current);

    let mut batch = PlannedMaterialCandidateBatch::new();
    let candidate = stage_rebuild(&mut resources, &mut batch, stale_handle, &[552], None);
    resources.publish_planned_material_candidates(&mut batch, &[true]);

    assert!(!batch.result(candidate));
    assert!(resources.get_material(stale_handle).is_none());
    assert_eq!(
        resources
            .get_material(current_handle)
            .expect("newer Material cache owner was displaced")
            .bind_group_id,
        current_bind_group_id
    );
    assert!(batch.private_materials[0].is_some());
}

#[test]
fn null_material_handle_reservation_fails_closed() {
    let mut resources = test_resource_manager();
    assert!(
        !resources
            .resources
            .reserve_material_slot(MaterialHandle::default())
    );
}

#[test]
fn out_of_order_material_slot_reservations_preserve_earlier_high_indices() {
    let mut resources = test_resource_manager();
    let high = test_material_handle(16_384);
    let low = test_material_handle(8);
    let mut batch = PlannedMaterialCandidateBatch::new();
    let high_candidate = stage_rebuild(&mut resources, &mut batch, high, &[601], None);
    let low_candidate = stage_rebuild(&mut resources, &mut batch, low, &[602], None);

    resources.publish_planned_material_candidates(&mut batch, &[true, true]);

    assert!(batch.result(high_candidate));
    assert!(batch.result(low_candidate));
    assert!(resources.get_material(high).is_some());
    assert!(resources.get_material(low).is_some());
}

#[test]
fn candidate_batch_clear_retains_then_trim_releases_excess_capacity() {
    let mut batch = PlannedMaterialCandidateBatch::new();
    let handle = test_material_handle(4);
    for version in 0..(PLANNED_MATERIAL_CANDIDATE_BASELINE * 16) {
        batch.push_unavailable(handle, version as u32, version as u64, version as u64);
    }

    assert!(batch.active_bytes() > 0);
    let peak_retained = batch.retained_bytes();
    assert!(peak_retained > PlannedMaterialCandidateBatch::baseline_bytes());

    batch.clear();

    assert!(batch.is_empty());
    assert_eq!(batch.active_bytes(), 0);
    assert_eq!(batch.retained_bytes(), peak_retained);

    batch.trim_capacity(0, peak_retained);

    assert!(batch.retained_bytes() < peak_retained);
    assert!(batch.retained_bytes() >= PlannedMaterialCandidateBatch::baseline_bytes());
    let reused = batch.push_unavailable(handle, 1, 2, 3);
    assert_eq!(reused.index(), 0);
    batch.clear();
    assert!(batch.is_empty());
}
