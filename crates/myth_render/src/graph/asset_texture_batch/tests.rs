use std::sync::Arc;

use myth_resources::{
    image::{ImageDimension, PixelFormat},
    texture::Texture,
};

use super::*;

fn image_8x8(value: u8) -> Image {
    Image::new(
        8,
        8,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(vec![value; 8 * 8 * 4]),
    )
}

#[test]
fn stable_dedup_retains_one_texture_and_image_owner() {
    let assets = AssetServer::new();
    let image = assets.images.add(image_8x8(1));
    let first = assets.textures.add(Texture::new_2d(Some("first"), image));
    let second = assets.textures.add(Texture::new_2d(Some("second"), image));

    let mut batch = AssetTextureBatch::new();
    let first_id = batch.request(first);
    assert_eq!(batch.request(first), first_id);
    let second_id = batch.request(second);
    assert_ne!(first_id, second_id);
    assert_eq!(batch.request(first), first_id);
    assert_eq!(batch.request(second), second_id);
    batch.capture_requested(&assets);

    assert_eq!(batch.texture_count(), 2);
    assert_eq!(batch.image_count(), 1);
    assert_eq!(batch.texture_handles, [first, second]);
    assert!(batch.state_for_handle(first).unwrap().ready.is_some());
    assert!(batch.state_for_handle(second).unwrap().ready.is_some());
    assert!(Arc::ptr_eq(
        batch.image_owner(batch.id(first).unwrap()).unwrap().asset(),
        batch
            .image_owner(batch.id(second).unwrap())
            .unwrap()
            .asset(),
    ));
}

#[test]
fn unavailable_texture_and_image_are_complete_current_states() {
    let assets = AssetServer::new();
    let pending_texture = assets.textures.reserve();
    let pending_image = assets.images.reserve();
    let texture_with_pending_image = assets
        .textures
        .add(Texture::new_2d(Some("pending image"), pending_image));

    let mut batch = AssetTextureBatch::new();
    batch.capture(&assets, [pending_texture, texture_with_pending_image]);

    let texture_state = batch.state_for_handle(pending_texture).unwrap();
    assert_eq!(texture_state.texture_version, None);
    assert_eq!(texture_state.image_handle, None);
    assert_eq!(texture_state.ready, None);

    let image_state = batch.state_for_handle(texture_with_pending_image).unwrap();
    assert!(image_state.texture_version.is_some());
    assert_eq!(image_state.image_handle, Some(pending_image));
    assert_eq!(image_state.image_version, None);
    assert_eq!(image_state.ready, None);
}

#[test]
fn pending_texture_becoming_ready_invalidates_only_its_captured_record() {
    let assets = AssetServer::new();
    let stable_image = assets.images.add(image_8x8(1));
    let stable_texture = assets
        .textures
        .add(Texture::new_2d(Some("stable"), stable_image));
    let pending_texture = assets.textures.reserve();

    let mut batch = AssetTextureBatch::new();
    batch.capture(&assets, [pending_texture, stable_texture]);
    let pending_id = batch.id(pending_texture).unwrap();
    let stable_id = batch.id(stable_texture).unwrap();
    assert!(batch.state(pending_id).is_some());
    assert!(batch.state(stable_id).unwrap().ready.is_some());

    assets.textures.insert_ready(
        pending_texture,
        Texture::new_2d(Some("became ready"), stable_image),
    );
    batch.revalidate(&assets);

    assert!(batch.state(pending_id).is_none());
    assert!(batch.state(stable_id).unwrap().ready.is_some());
}

#[test]
fn pending_image_becoming_ready_invalidates_only_its_captured_record() {
    let assets = AssetServer::new();
    let stable_image = assets.images.add(image_8x8(1));
    let pending_image = assets.images.reserve();
    let stable_texture = assets
        .textures
        .add(Texture::new_2d(Some("stable"), stable_image));
    let texture_with_pending_image = assets
        .textures
        .add(Texture::new_2d(Some("pending image"), pending_image));

    let mut batch = AssetTextureBatch::new();
    batch.capture(&assets, [texture_with_pending_image, stable_texture]);
    let pending_id = batch.id(texture_with_pending_image).unwrap();
    let stable_id = batch.id(stable_texture).unwrap();
    assert!(batch.state(pending_id).is_some());
    assert!(batch.state(pending_id).unwrap().ready.is_none());
    assert!(batch.state(stable_id).unwrap().ready.is_some());

    assets.images.insert_ready(pending_image, image_8x8(2));
    batch.revalidate(&assets);

    assert!(batch.state(pending_id).is_none());
    assert!(batch.state(stable_id).unwrap().ready.is_some());
}

#[test]
fn texture_redirect_between_storage_batches_fails_closed() {
    let assets = AssetServer::new();
    let first_image = assets.images.add(image_8x8(1));
    let second_image = assets.images.add(image_8x8(2));
    let texture = assets
        .textures
        .add(Texture::new_2d(Some("redirected"), first_image));

    let mut batch = AssetTextureBatch::new();
    batch.capture_with_hooks(
        &assets,
        [texture],
        || {
            assets.textures.insert_ready(
                texture,
                Texture::new_2d(Some("redirected replacement"), second_image),
            );
        },
        || {},
    );

    assert!(batch.state_for_handle(texture).is_none());
    assert!(batch.texture_owner(batch.id(texture).unwrap()).is_none());
}

#[test]
fn image_aba_between_capture_and_validation_fails_closed() {
    let assets = AssetServer::new();
    let image = assets.images.add(image_8x8(1));
    let texture = assets
        .textures
        .add(Texture::new_2d(Some("image ABA"), image));

    let mut batch = AssetTextureBatch::new();
    batch.capture_with_hooks(
        &assets,
        [texture],
        || {},
        || assets.images.insert_ready(image, image_8x8(2)),
    );

    assert!(batch.state_for_handle(texture).is_none());
    assert!(batch.image_owner(batch.id(texture).unwrap()).is_none());
}

#[test]
fn one_bad_image_does_not_poison_an_unrelated_texture() {
    let assets = AssetServer::new();
    let good_image = assets.images.add(image_8x8(1));
    let bad_image = assets.images.add(image_8x8(2));
    let good = assets
        .textures
        .add(Texture::new_2d(Some("good"), good_image));
    let bad = assets.textures.add(Texture::new_2d(Some("bad"), bad_image));

    let mut batch = AssetTextureBatch::new();
    batch.capture_with_hooks(
        &assets,
        [bad, good],
        || {},
        || assets.images.insert_ready(bad_image, image_8x8(3)),
    );

    assert!(batch.state_for_handle(bad).is_none());
    assert!(batch.state_for_handle(good).unwrap().ready.is_some());
    assert_eq!(batch.texture_handles, [bad, good]);
}

#[test]
fn selected_postflight_crosses_batch_width_and_deduplicates_shared_images() {
    const SHARED_TEXTURE_COUNT: usize = 64;
    let assets = AssetServer::new();
    let shared_image = assets.images.add(image_8x8(1));
    let drifting_image = assets.images.add(image_8x8(2));

    let mut batch = AssetTextureBatch::new();
    let mut selected = Vec::with_capacity(SHARED_TEXTURE_COUNT + 1);
    for _ in 0..SHARED_TEXTURE_COUNT {
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("shared postflight"), shared_image));
        selected.push(batch.request(texture));
    }
    let drifting_texture = assets
        .textures
        .add(Texture::new_2d(Some("drifting postflight"), drifting_image));
    selected.push(batch.request(drifting_texture));
    batch.capture_requested(&assets);

    assets.images.insert_ready(drifting_image, image_8x8(3));
    let current = batch.postflight_selected(&assets, &selected);

    assert_eq!(current.len(), SHARED_TEXTURE_COUNT + 1);
    assert!(current[..SHARED_TEXTURE_COUNT].iter().all(|&value| value));
    assert!(!current[SHARED_TEXTURE_COUNT]);
    assert_eq!(batch.selected_texture_indices.len(), 65);
    assert_eq!(batch.selected_image_indices.len(), 2);
    assert_eq!(batch.selected_image_lookup.slots.len(), batch.image_count());
}

#[test]
fn complete_state_postflight_accepts_ready_and_unavailable_states() {
    let assets = AssetServer::new();
    let ready_image = assets.images.add(image_8x8(1));
    let unavailable_image = assets.images.reserve();
    let ready_texture = assets
        .textures
        .add(Texture::new_2d(Some("ready complete state"), ready_image));
    let unavailable_texture = assets.textures.reserve();
    let texture_with_unavailable_image = assets.textures.add(Texture::new_2d(
        Some("unavailable image complete state"),
        unavailable_image,
    ));

    let mut batch = AssetTextureBatch::new();
    let ready_id = batch.request(ready_texture);
    let unavailable_texture_id = batch.request(unavailable_texture);
    let unavailable_image_id = batch.request(texture_with_unavailable_image);
    batch.capture_requested(&assets);
    let selected = [ready_id, unavailable_texture_id, unavailable_image_id];

    assert_eq!(
        batch.postflight_states_selected(&assets, &selected),
        [true, true, true]
    );
    assert!(batch.stable_storage_epochs.is_some());
    assert!(batch.selected_texture_indices.is_empty());
    assert!(batch.selected_image_indices.is_empty());
    assert!(batch.selected_image_lookup.is_active_empty());

    assert_eq!(
        batch.postflight_selected(&assets, &selected),
        [true, false, false],
        "the existing ready-only postflight must retain its narrower contract"
    );
}

#[test]
fn complete_state_postflight_rejects_transitions_and_redirects() {
    let assets = AssetServer::new();
    let first_image = assets.images.add(image_8x8(1));
    let replacement_image = assets.images.add(image_8x8(2));
    let unavailable_image = assets.images.reserve();
    let stable_unavailable_texture = assets.textures.reserve();
    let transitioning_texture = assets.textures.reserve();
    let texture_with_transitioning_image = assets.textures.add(Texture::new_2d(
        Some("transitioning unavailable image"),
        unavailable_image,
    ));
    let redirected_texture = assets.textures.add(Texture::new_2d(
        Some("redirected complete state"),
        first_image,
    ));

    let mut batch = AssetTextureBatch::new();
    let stable_id = batch.request(stable_unavailable_texture);
    let transitioning_texture_id = batch.request(transitioning_texture);
    let transitioning_image_id = batch.request(texture_with_transitioning_image);
    let redirected_id = batch.request(redirected_texture);
    batch.capture_requested(&assets);

    assets.textures.insert_ready(
        transitioning_texture,
        Texture::new_2d(Some("now ready"), first_image),
    );
    assets.images.insert_ready(unavailable_image, image_8x8(3));

    let current = batch.postflight_states_selected_with_hooks(
        &assets,
        &[
            stable_id,
            transitioning_texture_id,
            transitioning_image_id,
            redirected_id,
        ],
        || {},
        || {
            assets.textures.insert_ready(
                redirected_texture,
                Texture::new_2d(Some("redirected replacement"), replacement_image),
            );
        },
    );

    assert_eq!(current, [true, false, false, false]);
}

#[test]
fn complete_state_postflight_crosses_batch_width_with_aligned_unavailable_states() {
    const READY_TEXTURE_COUNT: usize = 63;
    let assets = AssetServer::new();
    let shared_image = assets.images.add(image_8x8(1));
    let unavailable_image = assets.images.reserve();

    let mut batch = AssetTextureBatch::new();
    let mut selected = Vec::with_capacity(READY_TEXTURE_COUNT + 2);
    for _ in 0..READY_TEXTURE_COUNT {
        let texture = assets.textures.add(Texture::new_2d(
            Some("shared complete-state boundary"),
            shared_image,
        ));
        selected.push(batch.request(texture));
    }
    let unavailable_texture = assets.textures.reserve();
    selected.push(batch.request(unavailable_texture));
    let texture_with_unavailable_image = assets.textures.add(Texture::new_2d(
        Some("unavailable-image boundary"),
        unavailable_image,
    ));
    selected.push(batch.request(texture_with_unavailable_image));
    batch.capture_requested(&assets);

    let current = batch.postflight_states_selected_with_hooks(&assets, &selected, || {}, || {});
    assert_eq!(current.len(), 65);
    assert!(current.iter().all(|&value| value));
    assert_eq!(batch.selected_texture_indices.len(), 65);
    assert_eq!(batch.selected_image_indices.len(), 2);
    assert_eq!(batch.selected_image_lookup.slots.len(), 2);

    assets.images.insert_ready(unavailable_image, image_8x8(2));
    let current = batch.postflight_states_selected(&assets, &selected);
    assert!(current[..64].iter().all(|&value| value));
    assert!(!current[64]);
}

#[test]
fn selected_image_sparse_lookup_reuses_retained_capacity_across_clear() {
    let assets = AssetServer::new();
    let first_image = assets.images.add(image_8x8(1));
    let first_texture = assets
        .textures
        .add(Texture::new_2d(Some("first attempt"), first_image));

    let mut batch = AssetTextureBatch::new();
    let first_id = batch.request(first_texture);
    batch.capture_requested(&assets);
    assert!(batch.postflight_selected(&assets, &[]).is_empty());
    assert!(batch.selected_image_lookup.is_active_empty());
    assert_eq!(
        batch.postflight_selected_with_hooks(&assets, &[first_id], || {}, || {}),
        [true]
    );
    assert_eq!(batch.selected_image_indices, [0]);
    let retained_lookup_capacity = batch.selected_image_lookup.slots.capacity();
    let retained_before_clear = batch.capacity_usage().retained_bytes;

    batch.clear();
    assert!(batch.selected_image_lookup.is_active_empty());
    assert_eq!(batch.capacity_usage().active_bytes, 0);
    assert_eq!(batch.capacity_usage().retained_bytes, retained_before_clear);

    let second_image = assets.images.add(image_8x8(2));
    let second_texture = assets
        .textures
        .add(Texture::new_2d(Some("second attempt"), second_image));
    let second_id = batch.request(second_texture);
    batch.capture_requested(&assets);
    assert_eq!(
        batch.postflight_selected_with_hooks(&assets, &[second_id], || {}, || {}),
        [true]
    );
    assert_eq!(
        batch.selected_image_lookup.slots.capacity(),
        retained_lookup_capacity
    );
    assert_eq!(batch.selected_image_indices, [0]);
}

#[test]
fn empty_and_high_index_single_selection_keep_image_lookup_sparse() {
    const CAPTURED_IMAGE_COUNT: usize = 1024;
    let assets = AssetServer::new();
    let mut batch = AssetTextureBatch::new();
    let mut last_id = None;

    for index in 0..CAPTURED_IMAGE_COUNT {
        let image = assets.images.add(image_8x8(index as u8));
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("sparse selected Image lookup"), image));
        last_id = Some(batch.request(texture));
    }
    batch.capture_requested(&assets);
    assert_eq!(batch.image_count(), CAPTURED_IMAGE_COUNT);
    assert_eq!(batch.selected_image_lookup.slots.capacity(), 0);

    assert!(batch.postflight_selected(&assets, &[]).is_empty());
    assert_eq!(batch.selected_image_lookup.slots.capacity(), 0);
    assert!(batch.selected_image_indices.is_empty());

    assert_eq!(
        batch.postflight_selected_with_hooks(&assets, &[last_id.unwrap()], || {}, || {},),
        [true]
    );
    assert_eq!(batch.selected_image_indices, [CAPTURED_IMAGE_COUNT - 1]);
    assert_eq!(batch.selected_image_lookup.slots.len(), 1);
    assert!(
        batch.selected_image_lookup.slots.capacity() < CAPTURED_IMAGE_COUNT / 16,
        "selected Image lookup retained capacity followed the captured domain"
    );
}

#[test]
fn stable_epochs_skip_scratch_and_unrelated_writes_restore_the_exact_slow_path() {
    let assets = AssetServer::new();
    let image = assets.images.add(image_8x8(1));
    let texture = assets
        .textures
        .add(Texture::new_2d(Some("stable epoch"), image));
    let mut batch = AssetTextureBatch::new();
    let id = batch.request(texture);
    batch.capture_requested(&assets);

    assert!(batch.storage_epochs_are_current(&assets));
    assert_eq!(batch.postflight_selected(&assets, &[id]), [true]);
    assert!(batch.selected_texture_indices.is_empty());
    assert!(batch.selected_image_indices.is_empty());

    assets.images.add(image_8x8(2));
    assert!(!batch.storage_epochs_are_current(&assets));
    assert_eq!(batch.postflight_selected(&assets, &[id]), [true]);
    assert_eq!(batch.selected_texture_indices, [id.index()]);
    assert_eq!(batch.selected_image_indices, [0]);
    assert!(
        !batch.storage_epochs_are_current(&assets),
        "a selected-only slow postflight must not retoken the full batch"
    );

    batch.revalidate(&assets);
    assert!(batch.storage_epochs_are_current(&assets));
    assert_eq!(batch.postflight_selected(&assets, &[id]), [true]);
    assert!(batch.selected_texture_indices.is_empty());
    assert!(batch.selected_image_indices.is_empty());
}

#[test]
fn selected_postflight_final_texture_check_closes_redirect_window() {
    let assets = AssetServer::new();
    let image = assets.images.add(image_8x8(1));
    let replacement_image = assets.images.add(image_8x8(2));
    let stable = assets.textures.add(Texture::new_2d(Some("stable"), image));
    let redirected = assets
        .textures
        .add(Texture::new_2d(Some("redirected"), image));

    let mut batch = AssetTextureBatch::new();
    let stable_id = batch.request(stable);
    let redirected_id = batch.request(redirected);
    batch.capture_requested(&assets);

    let current = batch.postflight_selected_with_hooks(
        &assets,
        &[stable_id, redirected_id],
        || {},
        || {
            assets.textures.insert_ready(
                redirected,
                Texture::new_2d(Some("replacement"), replacement_image),
            );
        },
    );

    assert_eq!(current, [true, false]);
}

#[test]
fn selected_postflight_and_ready_owners_fail_closed() {
    let assets = AssetServer::new();
    let image = assets.images.add(image_8x8(1));
    let ready = assets.textures.add(Texture::new_2d(Some("ready"), image));
    let unavailable = assets.textures.reserve();

    let mut batch = AssetTextureBatch::new();
    let ready_id = batch.request(ready);
    let unavailable_id = batch.request(unavailable);
    batch.capture_requested(&assets);
    let expected = batch.state(ready_id).unwrap().ready.unwrap();

    let (texture_owner, image_owner) = batch
        .ready_owners(ready_id, expected)
        .expect("exact ready owners missing");
    assert_eq!(texture_owner.image, image);
    assert_eq!(image_owner.width, 8);
    assert!(batch.ready_owners(unavailable_id, expected).is_none());

    let invalid_id = AssetTextureBatchId::from_index(1024);
    assert_eq!(
        batch.postflight_selected(&assets, &[ready_id, unavailable_id, invalid_id]),
        [true, false, false]
    );
}

#[test]
fn later_texture_and_image_aba_are_detected_by_revalidation() {
    let assets = AssetServer::new();
    let first_image = assets.images.add(image_8x8(1));
    let second_image = assets.images.add(image_8x8(2));
    let first = assets
        .textures
        .add(Texture::new_2d(Some("first"), first_image));
    let second = assets
        .textures
        .add(Texture::new_2d(Some("second"), second_image));

    let mut batch = AssetTextureBatch::new();
    batch.capture(&assets, [first, second]);
    assets.textures.insert_ready(
        first,
        Texture::new_2d(Some("first replacement"), first_image),
    );
    assets.images.insert_ready(second_image, image_8x8(4));
    batch.revalidate(&assets);

    assert!(batch.state_for_handle(first).is_none());
    assert!(batch.state_for_handle(second).is_none());
}

#[test]
fn clear_releases_exact_owners_and_retains_trimmable_capacity() {
    const UNIQUE_COUNT: usize = 512;
    let assets = AssetServer::new();
    let mut handles = Vec::with_capacity(UNIQUE_COUNT);
    let mut first_texture = None;
    let mut first_image = None;
    for index in 0..UNIQUE_COUNT {
        let image = assets.images.add(image_8x8(index as u8));
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("owner clear"), image));
        first_image.get_or_insert(image);
        first_texture.get_or_insert(texture);
        handles.push(texture);
    }
    let retained_texture = assets
        .textures
        .get(first_texture.unwrap())
        .expect("texture missing");
    let retained_image = assets
        .images
        .get(first_image.unwrap())
        .expect("image missing");

    let mut batch = AssetTextureBatch::new();
    batch.capture(&assets, handles);
    let selected: Vec<_> = (0..UNIQUE_COUNT)
        .map(AssetTextureBatchId::from_index)
        .collect();
    assert!(
        batch
            .postflight_selected(&assets, &selected)
            .iter()
            .all(|&value| value)
    );
    assert_eq!(Arc::strong_count(&retained_texture), 3);
    assert_eq!(Arc::strong_count(&retained_image), 3);
    let populated = batch.capacity_usage();
    assert!(populated.active_bytes > populated.baseline_bytes);

    batch.clear();
    assert_eq!(Arc::strong_count(&retained_texture), 2);
    assert_eq!(Arc::strong_count(&retained_image), 2);
    let cleared = batch.capacity_usage();
    assert_eq!(cleared.active_bytes, 0);
    assert_eq!(cleared.retained_bytes, populated.retained_bytes);

    batch.trim_capacity(cleared.baseline_bytes, cleared.retained_bytes);
    assert!(batch.capacity_usage().retained_bytes < cleared.retained_bytes);
    assert!(batch.is_empty());
}
