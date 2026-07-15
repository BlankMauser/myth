use super::*;

fn desc(width: u32) -> TextureDesc {
    TextureDesc::new_2d(
        width,
        64,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
    )
}

fn commit<const N: usize>(registry: &mut HistoryRegistry, plan: HistoryPlan<N>) -> bool {
    registry.commit(plan.commit, &plan.next.map(|texture| texture.id))
}

fn spec(width: u32) -> PersistentTextureSpec {
    PersistentTextureSpec::default_view(desc(width))
}

fn commit_cache<const N: usize>(
    registry: &mut HistoryRegistry,
    plan: CachePlan<N>,
) -> Option<CachePublicationReceipt> {
    registry.commit_cache(plan.commit?, &plan.textures.map(|texture| texture.id))
}

#[test]
fn history_plan_ping_pongs_only_after_commit() {
    let mut registry = HistoryRegistry::new();
    let set = registry.create_set();

    let first = registry.plan_set(set, [desc(64), desc(64)], 1, false);
    assert!(!first.valid);
    let repeated = registry.plan_set(set, [desc(64), desc(64)], 1, false);
    assert_eq!(first.previous, repeated.previous);
    assert_eq!(first.next, repeated.next);

    assert!(commit(&mut registry, first));
    let second = registry.plan_set(set, [desc(64), desc(64)], 2, false);
    assert!(second.valid);
    assert_eq!(second.previous, first.next);
    assert_eq!(second.next, first.previous);
}

#[test]
fn skipped_domain_attempt_invalidates_history() {
    let mut registry = HistoryRegistry::new();
    let set = registry.create_set();
    let first = registry.plan_set(set, [desc(64)], 1, false);
    assert!(commit(&mut registry, first));

    let skipped_attempt = registry.plan_set(set, [desc(64)], 3, false);
    assert!(!skipped_attempt.valid);
}

#[test]
fn descriptor_change_invalidates_without_changing_logical_ids() {
    let mut registry = HistoryRegistry::new();
    let set = registry.create_set();
    let first = registry.plan_set(set, [desc(64)], 1, false);
    assert!(commit(&mut registry, first));

    let resized = registry.plan_set(set, [desc(128)], 2, false);
    assert!(!resized.valid);
    assert_eq!(resized.previous[0].id, first.previous[0].id);
    assert_eq!(resized.next[0].id, first.next[0].id);
    assert_eq!(registry.physical_texture_count(), 0);
}

#[test]
fn stale_commit_is_rejected_after_reset() {
    let mut registry = HistoryRegistry::new();
    let set = registry.create_set();
    let stale = registry.plan_set(set, [desc(64)], 1, false);
    let reset = registry.plan_set(set, [desc(64)], 1, true);

    assert!(!commit(&mut registry, stale));
    assert!(commit(&mut registry, reset));
}

#[test]
fn repeated_reset_in_one_frame_is_idempotent() {
    let mut registry = HistoryRegistry::new();
    let set = registry.create_set();

    let first = registry.plan_set(set, [desc(64)], 1, true);
    let repeated = registry.plan_set(set, [desc(64)], 1, true);

    assert_eq!(first.previous, repeated.previous);
    assert_eq!(first.next, repeated.next);
    assert_eq!(first.commit, repeated.commit);
    assert!(commit(&mut registry, first));
}

#[test]
fn older_and_duplicate_commits_are_rejected() {
    let mut registry = HistoryRegistry::new();
    let set = registry.create_set();

    let first = registry.plan_set(set, [desc(64)], 1, false);
    let second = registry.plan_set(set, [desc(64)], 2, false);
    assert!(!commit(&mut registry, first));
    assert!(commit(&mut registry, second));
    assert!(!commit(&mut registry, second));
}

#[test]
fn commit_requires_the_complete_planned_write_bank() {
    let mut registry = HistoryRegistry::new();
    let set = registry.create_set();
    let plan = registry.plan_set(set, [desc(64), desc(64)], 1, false);

    assert!(!registry.commit(plan.commit, &[plan.next[0].id]));
    assert!(!registry.commit(plan.commit, &[plan.previous[0].id, plan.previous[1].id]));
    assert!(commit(&mut registry, plan));
}

#[test]
fn removing_a_set_rejects_pending_commits_and_never_reuses_its_identity() {
    let mut registry = HistoryRegistry::new();
    let retired_set = registry.create_set();
    let pending = registry.plan_set(retired_set, [desc(64), desc(64)], 1, false);

    assert_eq!(registry.history_set_count(), 1);
    assert!(registry.remove_set(retired_set));
    assert_eq!(registry.history_set_count(), 0);
    assert!(!commit(&mut registry, pending));
    assert!(!registry.remove_set(retired_set));

    let replacement_set = registry.create_set();
    assert_ne!(replacement_set, retired_set);
    let replacement = registry.plan_set(replacement_set, [desc(64), desc(64)], 1, false);
    assert!(!replacement.valid);
}

#[test]
fn cache_plan_is_idempotent_within_one_attempt_and_valid_after_commit() {
    let mut registry = HistoryRegistry::new();
    let cache = registry.create_cache();
    let key = CacheKey::new(7);

    let first = registry.plan_cache(cache, key, [spec(64), spec(32)], 10);
    let repeated = registry.plan_cache(cache, key, [spec(64), spec(32)], 10);
    assert!(!first.valid);
    assert_eq!(first.textures, repeated.textures);
    assert_eq!(first.commit, repeated.commit);

    let receipt = commit_cache(&mut registry, first).expect("cache should publish");
    assert_eq!(receipt.cache_id, cache);
    assert_eq!(receipt.key, key);
    assert!(commit_cache(&mut registry, repeated).is_none());

    let next = registry.plan_cache(cache, key, [spec(64), spec(32)], 11);
    assert!(next.valid);
    assert!(next.commit.is_none());
    assert_eq!(next.textures, first.textures);
    assert_ne!(next.commit, first.commit);
}

#[test]
fn a_new_cache_attempt_rejects_an_unsubmitted_token() {
    let mut registry = HistoryRegistry::new();
    let cache = registry.create_cache();
    let key = CacheKey::new(9);
    let stale = registry.plan_cache(cache, key, [spec(64)], 20);
    let current = registry.plan_cache(cache, key, [spec(64)], 21);

    assert!(commit_cache(&mut registry, stale).is_none());
    assert!(commit_cache(&mut registry, current).is_some());
}

#[test]
#[should_panic(expected = "attempt serial moved backwards")]
fn cache_attempt_serial_cannot_move_backwards() {
    let mut registry = HistoryRegistry::new();
    let cache = registry.create_cache();
    let _ = registry.plan_cache(cache, CacheKey::new(9), [spec(64)], 20);
    let _ = registry.plan_cache(cache, CacheKey::new(9), [spec(64)], 19);
}

#[test]
fn cache_key_and_spec_changes_invalidate_and_advance_epoch() {
    let mut registry = HistoryRegistry::new();
    let cache = registry.create_cache();
    let first = registry.plan_cache(cache, CacheKey::new(1), [spec(64)], 1);
    assert!(commit_cache(&mut registry, first).is_some());

    let rekeyed = registry.plan_cache(cache, CacheKey::new(2), [spec(64)], 2);
    assert!(!rekeyed.valid);
    assert_eq!(rekeyed.epoch, first.epoch.wrapping_add(1));
    assert_eq!(rekeyed.textures[0].id, first.textures[0].id);
    assert!(commit_cache(&mut registry, rekeyed).is_some());

    let cube_desc = TextureDesc::new(
        128,
        128,
        6,
        1,
        1,
        wgpu::TextureDimension::D2,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
    );
    let cube_spec = PersistentTextureSpec::new(cube_desc, Some(wgpu::TextureViewDimension::Cube));
    let resized = registry.plan_cache(cache, CacheKey::new(2), [cube_spec], 3);
    assert!(!resized.valid);
    assert_eq!(resized.epoch, rekeyed.epoch.wrapping_add(1));
    assert_eq!(resized.textures[0].id, rekeyed.textures[0].id);
    assert_eq!(
        registry.logical_specs.get(&resized.textures[0].id),
        Some(&cube_spec)
    );
}

#[test]
fn cache_member_count_change_retires_the_previous_logical_group() {
    let mut registry = HistoryRegistry::new();
    let cache = registry.create_cache();
    let first = registry.plan_cache(cache, CacheKey::new(1), [spec(64)], 1);
    let retired_id = first.textures[0].id;

    let expanded = registry.plan_cache(cache, CacheKey::new(1), [spec(64), spec(32)], 2);

    assert!(!expanded.valid);
    assert_eq!(expanded.epoch, first.epoch.wrapping_add(1));
    assert!(
        expanded
            .textures
            .iter()
            .all(|texture| texture.id != retired_id)
    );
    assert!(!registry.logical_specs.contains_key(&retired_id));
    assert!(commit_cache(&mut registry, first).is_none());
    assert!(commit_cache(&mut registry, expanded).is_some());
}

#[test]
#[should_panic(expected = "conflicting key/spec in one attempt")]
fn conflicting_cache_plans_in_one_attempt_are_rejected() {
    let mut registry = HistoryRegistry::new();
    let cache = registry.create_cache();
    let _ = registry.plan_cache(cache, CacheKey::new(1), [spec(64)], 1);
    let _ = registry.plan_cache(cache, CacheKey::new(2), [spec(64)], 1);
}

#[test]
fn cache_commit_requires_the_complete_texture_group() {
    let mut registry = HistoryRegistry::new();
    let cache = registry.create_cache();
    let plan = registry.plan_cache(cache, CacheKey::new(1), [spec(64), spec(32)], 1);

    assert!(
        registry
            .commit_cache(
                plan.commit.expect("invalid plan needs a token"),
                &[plan.textures[0].id]
            )
            .is_none()
    );
    assert!(commit_cache(&mut registry, plan).is_some());
}

#[test]
fn removing_a_cache_rejects_tokens_and_does_not_reuse_identity() {
    let mut registry = HistoryRegistry::new();
    let retired = registry.create_cache();
    let pending = registry.plan_cache(retired, CacheKey::new(1), [spec(64)], 1);
    let retired_member = pending.textures[0].id;

    assert_eq!(registry.cache_count(), 1);
    assert!(registry.remove_cache(retired));
    assert_eq!(registry.cache_count(), 0);
    assert!(commit_cache(&mut registry, pending).is_none());
    assert!(!registry.remove_cache(retired));

    let replacement = registry.create_cache();
    assert_ne!(replacement, retired);
    let replacement_plan = registry.plan_cache(replacement, CacheKey::new(1), [spec(64)], 1);
    assert_ne!(replacement_plan.textures[0].id, retired_member);
}

#[test]
#[should_panic(expected = "cache ID is retired")]
fn removed_cache_identity_cannot_be_revived() {
    let mut registry = HistoryRegistry::new();
    let retired = registry.create_cache();
    let _ = registry.plan_cache(retired, CacheKey::new(1), [spec(64)], 1);
    assert!(registry.remove_cache(retired));
    let _ = registry.plan_cache(retired, CacheKey::new(1), [spec(64)], 2);
}

#[test]
#[should_panic(expected = "unplanned or stale specification")]
fn unplanned_persistent_texture_spec_is_rejected_before_materialization() {
    let registry = HistoryRegistry::new();
    registry.assert_texture_spec_is_current(PersistentTextureId::new(), spec(64));
}

#[test]
#[should_panic(expected = "unplanned or stale specification")]
fn stale_persistent_texture_spec_is_rejected_before_materialization() {
    let mut registry = HistoryRegistry::new();
    let cache = registry.create_cache();
    let old = registry.plan_cache(cache, CacheKey::new(1), [spec(64)], 1);
    let _ = registry.plan_cache(cache, CacheKey::new(1), [spec(128)], 2);
    registry.assert_texture_spec_is_current(old.textures[0].id, old.textures[0].spec);
}

#[test]
#[should_panic(expected = "unplanned or stale specification")]
fn wrong_canonical_dimension_is_rejected_before_materialization() {
    let mut registry = HistoryRegistry::new();
    let cache = registry.create_cache();
    let plan = registry.plan_cache(cache, CacheKey::new(1), [spec(64)], 1);
    registry.assert_texture_spec_is_current(
        plan.textures[0].id,
        PersistentTextureSpec::new(desc(64), Some(wgpu::TextureViewDimension::D2Array)),
    );
}
