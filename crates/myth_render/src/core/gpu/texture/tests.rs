use super::*;
use crate::settings::{RendererInitConfig, RendererSettings};
use crate::test_support::SharedTestContext as WgpuContext;
use myth_resources::image::{ColorSpace, ImageDimension, PixelFormat};

fn image_8x8() -> Image {
    Image::new(
        8,
        8,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(vec![255; 8 * 8 * 4]),
    )
}

#[test]
fn snapshots_keep_texture_and_image_versions_separate() {
    let assets = AssetServer::new();
    let image = assets.images.add(image_8x8());
    let texture = assets
        .textures
        .add(Texture::new_2d(Some("Snapshot"), image));

    let first = snapshot_asset_texture(&assets, texture).expect("texture must be ready");
    assert_eq!(first.texture_version, 1);
    assert_eq!(first.image_version, 1);
    assert_eq!(first.mip_count, 1);
    assert_eq!(first.format, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut changed = Texture::new_2d(Some("Snapshot Linear Mipped"), image);
    changed.color_space = ColorSpace::Linear;
    changed.generate_mipmaps = true;
    assert_eq!(assets.textures.update(texture, changed), Some(2));

    let second = snapshot_asset_texture(&assets, texture).expect("texture must stay ready");
    assert_eq!(second.texture_version, 2);
    assert_eq!(second.image_version, 1);
    assert_eq!(second.mip_count, 4);
    assert_eq!(second.format, wgpu::TextureFormat::Rgba8Unorm);
    assert_ne!(first.logical_version, second.logical_version);
}

#[test]
fn exact_resident_token_is_fail_closed_across_sampler_and_image_drift() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
    let assets = AssetServer::new();
    let image = assets.images.add(image_8x8());
    let texture = assets
        .textures
        .add(Texture::new_2d(Some("resident token"), image));

    let initial_snapshot =
        snapshot_asset_texture(&assets, texture).expect("initial snapshot missing");
    assert!(
        resources
            .materialize_asset_texture_for_graph(&assets, initial_snapshot)
            .texture
            .is_some()
    );
    let initial_token = resources
        .exact_asset_texture_residency(initial_snapshot)
        .expect("initial binding must be exactly resident");
    let initial_generation = resources
        .resources
        .texture_generation_key(texture)
        .expect("initial generation missing");
    let reused = resources
        .reuse_exact_asset_texture_for_graph(initial_snapshot, initial_token)
        .expect("an unchanged exact token must be reusable");
    assert_eq!(
        reused
            .texture
            .expect("resident reuse lost its physical owner")
            .generation_key,
        initial_generation
    );

    let mut sampler_update = Texture::new_2d(Some("resident token nearest"), image);
    sampler_update.sampler.mag_filter = wgpu::FilterMode::Nearest;
    assert_eq!(assets.textures.update(texture, sampler_update), Some(2));
    let sampler_snapshot =
        snapshot_asset_texture(&assets, texture).expect("sampler snapshot missing");
    assert!(
        resources
            .materialize_asset_texture_for_graph(&assets, sampler_snapshot)
            .texture
            .is_some()
    );
    let sampler_token = resources
        .exact_asset_texture_residency(sampler_snapshot)
        .expect("sampler update was not published exactly");
    let sampler_generation = resources
        .resources
        .texture_generation_key(texture)
        .expect("sampler generation missing");
    assert_eq!(
        sampler_generation, initial_generation,
        "sampler-only drift must retain the physical image generation"
    );
    assert_ne!(
        sampler_token, initial_token,
        "the exact token must still cover the changed sampler slot"
    );
    let sampler_binding = *resources
        .get_texture_binding(texture)
        .expect("sampler binding missing");
    assert!(
        resources
            .reuse_exact_asset_texture_for_graph(initial_snapshot, initial_token)
            .is_none(),
        "an old resident token must not survive sampler/logical drift"
    );
    assert_eq!(
        *resources
            .get_texture_binding(texture)
            .expect("failed CAS removed the current binding"),
        sampler_binding,
        "a rejected old token must not overwrite the current binding"
    );

    let replacement = Image::new(
        8,
        8,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(vec![37; 8 * 8 * 4]),
    );
    assert_eq!(assets.images.update(image, replacement), Some(2));
    let image_snapshot =
        snapshot_asset_texture(&assets, texture).expect("image snapshot missing");
    assert!(
        resources
            .materialize_asset_texture_for_graph(&assets, image_snapshot)
            .texture
            .is_some()
    );
    let image_token = resources
        .exact_asset_texture_residency(image_snapshot)
        .expect("image refresh was not published exactly");
    let image_generation = resources
        .resources
        .texture_generation_key(texture)
        .expect("image generation missing");
    assert_eq!(image_generation.allocation, sampler_generation.allocation);
    assert_ne!(
        image_generation, sampler_generation,
        "content drift must advance the physical allocation generation"
    );
    assert_ne!(image_token, sampler_token);
    let image_binding = *resources
        .get_texture_binding(texture)
        .expect("refreshed image binding missing");
    assert!(
        resources
            .reuse_exact_asset_texture_for_graph(sampler_snapshot, sampler_token)
            .is_none(),
        "an old resident token must not survive image-generation drift"
    );
    assert_eq!(
        *resources
            .get_texture_binding(texture)
            .expect("generation CAS failure removed the current binding"),
        image_binding,
        "a rejected generation token must leave the exact winner untouched"
    );
    assert_eq!(
        resources
            .resources
            .texture_generation_key(texture)
            .expect("generation CAS failure removed the current image"),
        image_generation
    );
}

#[test]
fn incompatible_texture_specs_own_independent_physical_images_and_mip_requests() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
    let assets = AssetServer::new();
    let image = assets.images.add(image_8x8());

    let srgb = assets
        .textures
        .add(Texture::new_2d(Some("Shared Image sRGB"), image));
    let mut linear_asset = Texture::new_2d(Some("Shared Image Linear"), image);
    linear_asset.color_space = ColorSpace::Linear;
    linear_asset.generate_mipmaps = true;
    let linear = assets.textures.add(linear_asset);

    assert!(matches!(
        resources.prepare_texture(&assets, srgb),
        ResourceState::Ready
    ));
    assert!(matches!(
        resources.prepare_texture(&assets, linear),
        ResourceState::Ready
    ));

    let srgb_image = resources
        .get_texture_image(srgb)
        .expect("sRGB image missing");
    let linear_image = resources
        .get_texture_image(linear)
        .expect("linear image missing");
    let srgb_physical = resources
        .resources
        .texture_image_lease(srgb)
        .expect("sRGB lease missing")
        .physical_id;
    let linear_physical = resources
        .resources
        .texture_image_lease(linear)
        .expect("linear lease missing")
        .physical_id;
    assert_ne!(srgb_physical, linear_physical);
    assert_ne!(srgb_image.id, linear_image.id);
    assert_eq!(srgb_image.format, wgpu::TextureFormat::Rgba8UnormSrgb);
    assert_eq!(srgb_image.mip_level_count, 1);
    assert_eq!(linear_image.format, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(linear_image.mip_level_count, 4);

    let mut srgb_changed = Texture::new_2d(Some("sRGB now mipped"), image);
    srgb_changed.generate_mipmaps = true;
    assert_eq!(assets.textures.update(srgb, srgb_changed), Some(2));
    let old_srgb_id = srgb_image.id;
    assert!(matches!(
        resources.prepare_texture(&assets, srgb),
        ResourceState::Ready
    ));
    let rebuilt_srgb = resources
        .get_texture_image(srgb)
        .expect("rebuilt sRGB image missing");
    let mipped_srgb_physical = resources
        .resources
        .texture_image_lease(srgb)
        .expect("rebuilt sRGB lease missing")
        .physical_id;
    assert_ne!(mipped_srgb_physical, srgb_physical);
    assert_ne!(rebuilt_srgb.id, old_srgb_id);
    assert_eq!(rebuilt_srgb.mip_level_count, 4);
    let mipped_srgb_id = rebuilt_srgb.id;
    let binding = resources
        .get_texture_binding(srgb)
        .expect("sRGB binding missing");
    assert_eq!(binding.texture_version, 2);
    assert_eq!(binding.image_version, 1);

    let srgb_generation = resources
        .resources
        .texture_generation_key(srgb)
        .expect("sRGB generation missing");
    let linear_generation = resources
        .resources
        .texture_generation_key(linear)
        .expect("linear generation missing");

    let requests = resources.take_mipmap_requests();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .any(|request| request.key() == srgb_generation)
    );
    assert!(
        requests
            .iter()
            .any(|request| request.key() == linear_generation)
    );
    assert_ne!(requests[0].key(), requests[1].key());

    // Leave a fresh request pending so the true -> false transition also
    // proves that recreating the physical allocation releases the
    // obsolete generation owner.
    assert!(matches!(
        resources.prepare_texture(&assets, srgb),
        ResourceState::Ready
    ));

    let srgb_without_mips = Texture::new_2d(Some("sRGB mips disabled"), image);
    assert_eq!(assets.textures.update(srgb, srgb_without_mips), Some(3));
    assert!(matches!(
        resources.prepare_texture(&assets, srgb),
        ResourceState::Ready
    ));
    let rebuilt_without_mips = resources
        .get_texture_image(srgb)
        .expect("sRGB no-mip image missing");
    let no_mip_physical = resources
        .resources
        .texture_image_lease(srgb)
        .expect("sRGB no-mip lease missing")
        .physical_id;
    assert_ne!(no_mip_physical, mipped_srgb_physical);
    assert_ne!(rebuilt_without_mips.id, mipped_srgb_id);
    assert_eq!(rebuilt_without_mips.mip_level_count, 1);
    assert!(resources.take_mipmap_requests().is_empty());
}

#[test]
fn exact_spec_handles_share_physical_image_and_release_per_lease() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
    let assets = AssetServer::new();
    let image_handle = assets.images.add(image_8x8());

    let mut first_texture = Texture::new_2d(Some("shared first"), image_handle);
    first_texture.generate_mipmaps = true;
    let first = assets.textures.add(first_texture);
    let mut second_texture = Texture::new_2d(Some("shared nearest"), image_handle);
    second_texture.generate_mipmaps = true;
    second_texture.sampler.mag_filter = wgpu::FilterMode::Nearest;
    let second = assets.textures.add(second_texture);

    assert!(matches!(
        resources.prepare_texture(&assets, first),
        ResourceState::Ready
    ));
    assert!(matches!(
        resources.prepare_texture(&assets, second),
        ResourceState::Ready
    ));
    resources.resources.assert_texture_pool_consistent();

    let first_lease = resources
        .resources
        .texture_image_lease(first)
        .expect("first lease missing");
    let second_lease = resources
        .resources
        .texture_image_lease(second)
        .expect("second lease missing");
    assert_eq!(first_lease.physical_id, second_lease.physical_id);
    assert_eq!(
        resources
            .resources
            .physical_image_member_count(first_lease.physical_id),
        Some(2)
    );
    assert_eq!(
        resources
            .get_texture_image(first)
            .expect("first physical image missing")
            .id,
        resources
            .get_texture_image(second)
            .expect("second physical image missing")
            .id
    );
    assert_ne!(
        resources
            .get_texture_binding(first)
            .expect("first binding missing")
            .sampler_id,
        resources
            .get_texture_binding(second)
            .expect("second binding missing")
            .sampler_id,
        "sampler state must remain TextureHandle-local"
    );

    let generation = resources
        .resources
        .texture_generation_key(first)
        .expect("shared generation missing");
    let mut initial_requests = resources.take_mipmap_requests();
    assert_eq!(initial_requests.len(), 1);
    let old_graph_owner = initial_requests
        .pop()
        .expect("shared generation owner missing");
    assert_eq!(old_graph_owner.key(), generation);
    assert!(matches!(
        resources.prepare_texture(&assets, first),
        ResourceState::Ready
    ));

    assert_eq!(
        assets.textures.update(
            first,
            Texture::new_2d(Some("first migrated without mips"), image_handle),
        ),
        Some(2)
    );
    assert!(matches!(
        resources.prepare_texture(&assets, first),
        ResourceState::Ready
    ));
    let migrated_physical = resources
        .resources
        .texture_image_lease(first)
        .expect("migrated lease missing")
        .physical_id;
    assert_ne!(migrated_physical, second_lease.physical_id);
    assert_eq!(
        resources
            .resources
            .physical_image_member_count(migrated_physical),
        Some(1)
    );
    assert_eq!(
        resources
            .resources
            .physical_image_member_count(second_lease.physical_id),
        Some(1)
    );

    resources.resources.remove_texture_image(first);
    resources.resources.assert_texture_pool_consistent();
    assert!(resources.get_texture_image(first).is_none());
    assert!(resources.get_texture_binding(first).is_none());
    assert!(resources.get_texture_image(second).is_some());
    assert_eq!(
        resources
            .resources
            .physical_image_member_count(second_lease.physical_id),
        Some(1)
    );
    let pending = resources.take_mipmap_requests();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].key(), generation);

    assert!(matches!(
        resources.prepare_texture(&assets, second),
        ResourceState::Ready
    ));
    resources.resources.remove_texture_image(second);
    resources.resources.assert_texture_pool_consistent();
    assert!(
        !resources
            .resources
            .contains_physical_image(second_lease.physical_id)
    );
    assert!(resources.take_mipmap_requests().is_empty());

    let mut replacement_texture = Texture::new_2d(Some("replacement owner"), image_handle);
    replacement_texture.generate_mipmaps = true;
    let replacement = assets.textures.add(replacement_texture);
    assert!(matches!(
        resources.prepare_texture(&assets, replacement),
        ResourceState::Ready
    ));
    let replacement_physical = resources
        .resources
        .texture_image_lease(replacement)
        .expect("replacement lease missing")
        .physical_id;
    assert_ne!(replacement_physical, second_lease.physical_id);

    old_graph_owner.mark_submitted();
    assert!(
        !resources
            .get_texture_image(replacement)
            .expect("replacement physical image missing")
            .mipmaps_submitted(),
        "an old graph owner must not publish a reused physical slot"
    );
    resources.resources.queue_mipmap_request(old_graph_owner);
    let replacement_requests = resources.take_mipmap_requests();
    assert_eq!(replacement_requests.len(), 1);
    assert_eq!(replacement_requests[0].physical_id, replacement_physical);
    resources.resources.assert_texture_pool_consistent();
}

#[test]
fn shared_content_refresh_invalidates_other_lease_until_revalidated() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
    let assets = AssetServer::new();
    let image_handle = assets.images.add(image_8x8());
    let first = assets
        .textures
        .add(Texture::new_2d(Some("content first"), image_handle));
    let second = assets
        .textures
        .add(Texture::new_2d(Some("content second"), image_handle));

    assert!(matches!(
        resources.prepare_texture(&assets, first),
        ResourceState::Ready
    ));
    assert!(matches!(
        resources.prepare_texture(&assets, second),
        ResourceState::Ready
    ));
    resources.resources.assert_texture_pool_consistent();
    let physical_id = resources
        .resources
        .texture_image_lease(first)
        .expect("shared lease missing")
        .physical_id;
    let allocation = resources
        .get_texture_image(first)
        .expect("shared image missing")
        .id;

    let replacement = Image::new(
        8,
        8,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(vec![23; 8 * 8 * 4]),
    );
    assert_eq!(assets.images.update(image_handle, replacement), Some(2));
    assert!(matches!(
        resources.prepare_texture(&assets, first),
        ResourceState::Ready
    ));
    let refreshed_generation = resources
        .get_texture_image(first)
        .expect("refreshed image missing")
        .generation_id;
    assert!(
        resources.get_texture_binding(second).is_none(),
        "an unvalidated logical lease must not publish refreshed physical content"
    );

    assert!(matches!(
        resources.prepare_texture(&assets, second),
        ResourceState::Ready
    ));
    let second_image = resources
        .get_texture_image(second)
        .expect("revalidated image missing");
    assert_eq!(second_image.id, allocation);
    assert_eq!(second_image.generation_id, refreshed_generation);
    assert_eq!(
        resources
            .resources
            .texture_image_lease(second)
            .expect("second lease missing")
            .physical_id,
        physical_id
    );
    assert_eq!(
        resources.resources.physical_image_member_count(physical_id),
        Some(2)
    );
}

#[test]
fn physical_pending_queue_rejects_a_late_stale_generation() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
    let assets = AssetServer::new();
    let image_handle = assets.images.add(image_8x8());
    let mut texture = Texture::new_2d(Some("Physical pending"), image_handle);
    texture.generate_mipmaps = true;
    let texture_handle = assets.textures.add(texture);

    assert!(matches!(
        resources.prepare_texture(&assets, texture_handle),
        ResourceState::Ready
    ));
    let mut first = resources.take_mipmap_requests();
    assert_eq!(first.len(), 1);
    let stale = first.pop().expect("initial mip request missing");
    let physical_id = stale.physical_id;
    let stale_generation = stale.generation_id;
    let stale_after_drain = resources
        .resources
        .texture_mipmap_request(texture_handle)
        .expect("second stale request missing");

    let replacement = Image::new(
        8,
        8,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(vec![17; 8 * 8 * 4]),
    );
    assert_eq!(assets.images.update(image_handle, replacement), Some(2));
    assert!(matches!(
        resources.prepare_texture(&assets, texture_handle),
        ResourceState::Ready
    ));
    let current_generation = resources
        .get_texture_image(texture_handle)
        .expect("updated physical image missing")
        .generation_id;
    assert_ne!(current_generation, stale_generation);
    assert_eq!(
        resources
            .resources
            .texture_image_lease(texture_handle)
            .expect("updated lease missing")
            .physical_id,
        physical_id,
        "content updates must preserve the physical allocation"
    );

    stale.mark_submitted();
    assert!(
        !resources
            .get_texture_image(texture_handle)
            .expect("current physical image missing")
            .mipmaps_submitted(),
        "publishing an old generation must not make the current one ready"
    );
    resources.resources.queue_mipmap_request(stale);
    let pending = resources.take_mipmap_requests();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].physical_id, physical_id);
    assert_eq!(pending[0].generation_id, current_generation);
    pending[0].mark_submitted();
    assert!(
        resources
            .get_texture_image(texture_handle)
            .expect("submitted physical image missing")
            .mipmaps_submitted()
    );

    stale_after_drain.mark_submitted();
    assert!(
        resources
            .get_texture_image(texture_handle)
            .expect("current physical image missing after stale publication")
            .mipmaps_submitted(),
        "late publication of an old generation must not regress readiness"
    );
    resources.resources.queue_mipmap_request(stale_after_drain);
    assert!(
        resources.take_mipmap_requests().is_empty(),
        "a stale generation must remain rejected after the current request was drained"
    );
}

#[test]
fn same_handle_same_version_asset_replacements_refresh_gpu_identity() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
    let assets = AssetServer::new();
    let image_handle = assets.images.add(image_8x8());
    let mut texture = Texture::new_2d(Some("ABA sRGB"), image_handle);
    texture.generate_mipmaps = true;
    let texture_handle = assets.textures.add(texture);

    let first_snapshot =
        snapshot_asset_texture(&assets, texture_handle).expect("initial snapshot missing");
    assert!(matches!(
        resources.prepare_texture(&assets, texture_handle),
        ResourceState::Ready
    ));
    let first_binding = *resources
        .get_texture_binding(texture_handle)
        .expect("initial binding missing");
    let first_physical_id = resources
        .get_texture_image(texture_handle)
        .expect("initial GPU image missing")
        .id;
    let first_pool_id = resources
        .resources
        .texture_image_lease(texture_handle)
        .expect("initial texture lease missing")
        .physical_id;

    let original_texture_uuid = assets
        .textures
        .get(texture_handle)
        .expect("initial texture missing")
        .uuid;
    let mut replacement = Texture::new_2d(Some("ABA linear"), image_handle);
    replacement.uuid = original_texture_uuid;
    replacement.color_space = ColorSpace::Linear;
    replacement.generate_mipmaps = true;
    assets.textures.insert_ready(texture_handle, replacement);

    let replacement_snapshot =
        snapshot_asset_texture(&assets, texture_handle).expect("replacement snapshot missing");
    assert_eq!(replacement_snapshot.texture_version, 1);
    assert_eq!(replacement_snapshot.image_version, 1);
    assert_ne!(
        replacement_snapshot.texture_identity,
        first_snapshot.texture_identity
    );
    assert_ne!(
        replacement_snapshot.logical_version,
        first_snapshot.logical_version
    );

    assert!(matches!(
        resources.prepare_texture(&assets, texture_handle),
        ResourceState::Ready
    ));
    let replacement_binding = *resources
        .get_texture_binding(texture_handle)
        .expect("replacement binding missing");
    let replacement_physical_id = {
        let image = resources
            .get_texture_image(texture_handle)
            .expect("replacement GPU image missing");
        assert_eq!(image.format, wgpu::TextureFormat::Rgba8Unorm);
        image.id
    };
    let replacement_pool_id = resources
        .resources
        .texture_image_lease(texture_handle)
        .expect("replacement texture lease missing")
        .physical_id;
    assert_ne!(replacement_physical_id, first_physical_id);
    assert_ne!(replacement_pool_id, first_pool_id);
    assert_ne!(
        replacement_binding.texture_identity,
        first_binding.texture_identity
    );
    assert_eq!(replacement_binding.sampler_id, first_binding.sampler_id);
    let requests = resources.take_mipmap_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].key().allocation, replacement_physical_id);

    // A sampler-only same-version replacement reuses the physical view.
    // Graph-owned materialization must still publish the replacement's
    // exact sampler binding without leaking its mip request into the
    // legacy queue.
    let mut sampler_replacement = Texture::new_2d(Some("ABA nearest"), image_handle);
    sampler_replacement.uuid = original_texture_uuid;
    sampler_replacement.color_space = ColorSpace::Linear;
    sampler_replacement.generate_mipmaps = true;
    sampler_replacement.sampler.mag_filter = wgpu::FilterMode::Nearest;
    assets
        .textures
        .insert_ready(texture_handle, sampler_replacement);
    let sampler_snapshot =
        snapshot_asset_texture(&assets, texture_handle).expect("sampler snapshot missing");
    let deferred = resources.materialize_asset_texture_for_graph(&assets, sampler_snapshot);
    assert!(deferred.texture.is_some());
    assert!(deferred.mipmap_request.is_some());
    assert_eq!(
        resources
            .get_texture_image(texture_handle)
            .expect("deferred image missing")
            .id,
        replacement_physical_id
    );
    assert_eq!(
        resources
            .resources
            .texture_image_lease(texture_handle)
            .expect("deferred sampler lease missing")
            .physical_id,
        replacement_pool_id
    );
    let graph_binding = *resources
        .get_texture_binding(texture_handle)
        .expect("graph materialization did not publish its exact binding");
    assert_ne!(graph_binding.sampler_id, replacement_binding.sampler_id);
    assert_eq!(
        graph_binding.texture_identity,
        sampler_snapshot.texture_identity
    );
    assert!(
        resources.take_mipmap_requests().is_empty(),
        "graph materialization must not enqueue a legacy mip request"
    );
    drop(deferred);
    // The graph attempt above deliberately did not submit its mip request;
    // normal compatibility prepare therefore observes the same incomplete
    // generation and may queue an independently owned retry.
    assert!(matches!(
        resources.prepare_texture(&assets, texture_handle),
        ResourceState::Ready
    ));
    let sampler_binding = *resources
        .get_texture_binding(texture_handle)
        .expect("sampler replacement binding missing");
    assert_ne!(sampler_binding.sampler_id, replacement_binding.sampler_id);
    assert_eq!(
        sampler_binding.texture_identity,
        sampler_snapshot.texture_identity
    );
    let (replacement_generation, replacement_image_identity) = {
        let image = resources
            .get_texture_image(texture_handle)
            .expect("sampler replacement GPU image missing");
        assert_eq!(image.id, replacement_physical_id);
        (image.generation_id, image.image_identity)
    };
    assert_eq!(resources.take_mipmap_requests().len(), 1);

    let original_image_uuid = assets
        .images
        .get(image_handle)
        .expect("initial image missing")
        .uuid;
    let mut replacement_image = Image::new(
        8,
        8,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(vec![31; 8 * 8 * 4]),
    );
    replacement_image.uuid = original_image_uuid;
    assets.images.insert_ready(image_handle, replacement_image);

    assert!(matches!(
        resources.prepare_texture(&assets, texture_handle),
        ResourceState::Ready
    ));
    let updated_binding = *resources
        .get_texture_binding(texture_handle)
        .expect("updated image binding missing");
    let updated_image = resources
        .get_texture_image(texture_handle)
        .expect("updated GPU image missing");
    assert_eq!(updated_image.id, replacement_physical_id);
    assert_eq!(
        resources
            .resources
            .texture_image_lease(texture_handle)
            .expect("updated image lease missing")
            .physical_id,
        replacement_pool_id
    );
    assert_ne!(updated_image.image_identity, replacement_image_identity);
    assert_ne!(updated_image.generation_id, replacement_generation);
    assert_eq!(updated_binding.image_version, 1);
    assert_eq!(updated_binding.image_identity, updated_image.image_identity);
    let updated_generation_key = resources
        .resources
        .texture_generation_key(texture_handle)
        .expect("updated texture generation missing");
    let requests = resources.take_mipmap_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].key(), updated_generation_key);
}

#[test]
fn ttl_prune_releases_mipmap_requests_and_repairs_image_aliases() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
    let assets = AssetServer::new();
    let image_handle = assets.images.add(image_8x8());

    let mut stale_texture = Texture::new_2d(Some("stale mipped"), image_handle);
    stale_texture.generate_mipmaps = true;
    let stale = assets.textures.add(stale_texture);
    let mut survivor_texture = Texture::new_2d(Some("surviving alias"), image_handle);
    survivor_texture.generate_mipmaps = true;
    let survivor = assets.textures.add(survivor_texture);

    assert!(matches!(
        resources.prepare_texture(&assets, stale),
        ResourceState::Ready
    ));
    let stale_physical = resources
        .resources
        .texture_image_lease(stale)
        .expect("stale lease missing")
        .physical_id;
    resources.next_frame();
    assert!(matches!(
        resources.prepare_texture(&assets, survivor),
        ResourceState::Ready
    ));
    let survivor_id = resources
        .get_texture_image(survivor)
        .expect("survivor missing")
        .id;
    let survivor_physical = resources
        .resources
        .texture_image_lease(survivor)
        .expect("survivor lease missing")
        .physical_id;
    assert_eq!(
        stale_physical, survivor_physical,
        "identical specifications must share one physical image"
    );

    // Simulate the compatibility alias having last pointed at an older
    // interpretation of the same Image. Prune must repair it to the live
    // survivor rather than leaving the lookup on its O(n) fallback.
    resources
        .resources
        .touch_texture_image_alias(image_handle, stale);
    assert_eq!(
        resources.resources.latest_texture_alias(image_handle),
        Some(stale)
    );

    resources.next_frame();
    resources.prune(1);
    resources.resources.assert_texture_pool_consistent();

    assert!(resources.get_texture_image(stale).is_none());
    assert!(
        resources
            .resources
            .contains_physical_image(survivor_physical)
    );
    assert_eq!(
        resources
            .resources
            .physical_image_member_count(survivor_physical),
        Some(1)
    );
    assert_eq!(
        resources
            .get_texture_image(survivor)
            .expect("live alias was pruned")
            .id,
        survivor_id
    );
    assert!(resources.get_texture_binding(stale).is_none());
    assert!(matches!(
        resources.prepare_texture(&assets, survivor),
        ResourceState::Ready
    ));
    let pending = resources.take_mipmap_requests();
    assert_eq!(
        pending.len(),
        1,
        "pruning one lease must retain the shared physical generation"
    );
    assert_eq!(
        pending[0].key(),
        resources
            .resources
            .texture_generation_key(survivor)
            .expect("survivor generation missing")
    );
    assert_eq!(
        resources.resources.latest_texture_alias(image_handle),
        Some(survivor)
    );
    assert_eq!(
        resources.get_image(image_handle).expect("alias missing").id,
        survivor_id
    );

    let moved = assets
        .textures
        .add(Texture::new_2d(Some("moved alias"), image_handle));
    assert!(matches!(
        resources.prepare_texture(&assets, moved),
        ResourceState::Ready
    ));
    assert_eq!(
        resources.resources.latest_texture_alias(image_handle),
        Some(moved)
    );
    let other_image = assets.images.add(image_8x8());
    assert_eq!(
        assets.textures.update(
            moved,
            Texture::new_2d(Some("moved to another image"), other_image),
        ),
        Some(2)
    );
    assert!(matches!(
        resources.prepare_texture(&assets, moved),
        ResourceState::Ready
    ));
    assert_eq!(
        resources.resources.latest_texture_alias(image_handle),
        Some(survivor),
        "changing a TextureHandle's Image must repair the old compatibility alias immediately"
    );
    assert_eq!(
        resources.resources.latest_texture_alias(other_image),
        Some(moved)
    );
}
