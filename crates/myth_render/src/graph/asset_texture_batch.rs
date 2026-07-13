//! Exact cross-storage Asset Texture snapshots for one graph attempt.
//!
//! A Texture and its Image live in independent [`AssetStorage`] instances, so
//! no read guard can make the pair globally atomic. This store instead keeps
//! one exact owner per stable-deduplicated handle, derives the complete logical
//! state after both bounded captures, and brackets Image validation with two
//! Texture validations. Each Texture is accepted or rejected independently.

use std::num::NonZeroU32;

#[cfg(test)]
use myth_assets::LoadedAssetSnapshot;
use myth_assets::{AssetServer, AssetSnapshot, ImageHandle, TextureHandle};
use myth_resources::{image::Image, texture::Texture};
use rustc_hash::FxHashMap;

use crate::core::gpu::{AssetTextureSnapshot, AssetTextureStateSnapshot};

use super::capacity::{
    CapacityUsage, estimated_hash_map_footprint, shrink_hash_map_by_family_ratio,
    shrink_vec_by_family_ratio, vec_footprint,
};

const ASSET_TEXTURE_BATCH_BASELINE: usize = 128;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AssetTextureBatchId(NonZeroU32);

impl AssetTextureBatchId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("asset texture batch index exceeds u32::MAX - 1");
        Self(raw)
    }

    #[inline]
    const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AssetImageBatchId(NonZeroU32);

impl AssetImageBatchId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("asset image batch index exceeds u32::MAX - 1");
        Self(raw)
    }

    #[inline]
    const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

#[derive(Debug, Clone, Copy)]
struct AssetTextureBatchRecord {
    image: Option<AssetImageBatchId>,
    state: AssetTextureStateSnapshot,
}

/// One graph-attempt table of exact Texture and Image owners.
///
/// Input Texture handles are stable-deduplicated in first-occurrence order.
/// Image handles are then stable-deduplicated in the first occurrence order of
/// the loaded Texture owners that reference them. Neither hash table is ever
/// iterated to produce externally observable ordering.
pub(crate) struct AssetTextureBatch {
    texture_handles: Vec<TextureHandle>,
    texture_lookup: FxHashMap<TextureHandle, AssetTextureBatchId>,
    texture_snapshots: Vec<AssetSnapshot<TextureHandle, Texture>>,
    /// First Texture validation in the T -> I -> T bracket.
    texture_current_before_images: Vec<bool>,
    /// Final Texture validation in the T -> I -> T bracket, folded in-place
    /// to the complete per-record acceptance bit after all three batches.
    texture_current: Vec<bool>,

    image_handles: Vec<ImageHandle>,
    image_lookup: FxHashMap<ImageHandle, AssetImageBatchId>,
    image_snapshots: Vec<AssetSnapshot<ImageHandle, Image>>,
    image_current: Vec<bool>,

    records: Vec<AssetTextureBatchRecord>,
    initialized: bool,
}

impl AssetTextureBatch {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            texture_handles: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            texture_lookup: FxHashMap::default(),
            texture_snapshots: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            texture_current_before_images: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            texture_current: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            image_handles: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            image_lookup: FxHashMap::default(),
            image_snapshots: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            image_current: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            records: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            initialized: false,
        }
    }

    /// Register one logical Texture dependency before the batch is sealed.
    ///
    /// IDs are allocated in stable first-occurrence order and remain valid
    /// until [`Self::clear`]. Material and Skybox planning can therefore
    /// contribute to one canonical request table before either publishes RDG
    /// topology.
    pub(crate) fn request(&mut self, handle: TextureHandle) -> AssetTextureBatchId {
        assert!(
            !self.initialized,
            "asset texture requests cannot be added after capture"
        );
        match self.texture_lookup.entry(handle) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let id = AssetTextureBatchId::from_index(self.texture_handles.len());
                self.texture_handles.push(handle);
                entry.insert(id);
                id
            }
        }
    }

    /// Capture and validate every previously requested Texture dependency.
    ///
    /// This is the publication barrier shared by Material and Skybox planning.
    /// It creates no WGPU object and submits no upload.
    pub(crate) fn capture_requested(&mut self, assets: &AssetServer) {
        self.capture_requested_with_hooks(assets, || {}, || {});
    }

    /// Convenience entry point used by isolated consumers and tests.
    /// Production graph planning normally calls [`Self::request`] from each
    /// producer and seals the combined table with [`Self::capture_requested`].
    ///
    /// This method performs CPU-only AssetStorage reads and snapshot
    /// derivation. It creates no WGPU object and submits no upload.
    #[cfg(test)]
    pub(crate) fn capture(
        &mut self,
        assets: &AssetServer,
        handles: impl IntoIterator<Item = TextureHandle>,
    ) {
        for handle in handles {
            self.request(handle);
        }
        self.capture_requested(assets);
    }

    #[cfg(test)]
    fn capture_with_hooks(
        &mut self,
        assets: &AssetServer,
        handles: impl IntoIterator<Item = TextureHandle>,
        after_texture_capture: impl FnOnce(),
        after_image_capture: impl FnOnce(),
    ) {
        for handle in handles {
            self.request(handle);
        }
        self.capture_requested_with_hooks(assets, after_texture_capture, after_image_capture);
    }

    fn capture_requested_with_hooks(
        &mut self,
        assets: &AssetServer,
        after_texture_capture: impl FnOnce(),
        after_image_capture: impl FnOnce(),
    ) {
        assert!(
            !self.initialized,
            "asset texture batch may be captured only once per graph attempt"
        );
        debug_assert!(self.texture_snapshots.is_empty());
        debug_assert!(self.texture_current_before_images.is_empty());
        debug_assert!(self.texture_current.is_empty());
        debug_assert!(self.image_handles.is_empty());
        debug_assert!(self.image_lookup.is_empty());
        debug_assert!(self.image_snapshots.is_empty());
        debug_assert!(self.image_current.is_empty());
        debug_assert!(self.records.is_empty());

        assets
            .textures
            .snapshot_batched(&self.texture_handles, &mut self.texture_snapshots);
        after_texture_capture();

        for snapshot in &self.texture_snapshots {
            let Some(texture) = snapshot.loaded() else {
                continue;
            };
            let image_handle = texture.asset().image;
            if let std::collections::hash_map::Entry::Vacant(entry) =
                self.image_lookup.entry(image_handle)
            {
                let index = AssetImageBatchId::from_index(self.image_handles.len());
                self.image_handles.push(image_handle);
                entry.insert(index);
            }
        }

        assets
            .images
            .snapshot_batched(&self.image_handles, &mut self.image_snapshots);
        after_image_capture();

        self.records.reserve(self.texture_snapshots.len());
        for texture_snapshot in &self.texture_snapshots {
            self.records.push(self.derive_record(texture_snapshot));
        }

        self.refresh_current(assets);
        self.initialized = true;
    }

    /// Revalidate every retained owner using a bounded T -> I -> T sequence.
    ///
    /// This updates per-entry current bits only. A failed dependency does not
    /// invalidate unrelated records; callers query each Texture independently.
    #[cfg(test)]
    pub(crate) fn revalidate(&mut self, assets: &AssetServer) {
        assert!(self.initialized, "asset texture batch was not captured");
        self.refresh_current(assets);
    }

    fn refresh_current(&mut self, assets: &AssetServer) {
        assets.textures.revalidate_batched(
            &self.texture_snapshots,
            &mut self.texture_current_before_images,
        );
        assets
            .images
            .revalidate_batched(&self.image_snapshots, &mut self.image_current);
        assets
            .textures
            .revalidate_batched(&self.texture_snapshots, &mut self.texture_current);
        debug_assert_eq!(self.records.len(), self.texture_current.len());
        for index in 0..self.records.len() {
            self.texture_current[index] = self.record_matches_validation(index);
        }
    }

    fn derive_record(
        &self,
        texture_snapshot: &AssetSnapshot<TextureHandle, Texture>,
    ) -> AssetTextureBatchRecord {
        let texture_handle = texture_snapshot.handle();
        let Some(texture) = texture_snapshot.loaded() else {
            return AssetTextureBatchRecord {
                image: None,
                state: AssetTextureStateSnapshot {
                    texture_handle,
                    texture_version: None,
                    texture_identity: None,
                    image_handle: None,
                    image_version: None,
                    image_identity: None,
                    ready: None,
                },
            };
        };

        let image_handle = texture.asset().image;
        let image_id = self
            .image_lookup
            .get(&image_handle)
            .copied()
            .expect("loaded Texture image was not captured");
        let image_snapshot = &self.image_snapshots[image_id.index()];
        debug_assert_eq!(image_snapshot.handle(), image_handle);

        let Some(image) = image_snapshot.loaded() else {
            return AssetTextureBatchRecord {
                image: Some(image_id),
                state: AssetTextureStateSnapshot {
                    texture_handle,
                    texture_version: Some(texture.version()),
                    texture_identity: Some(texture.identity()),
                    image_handle: Some(image_handle),
                    image_version: None,
                    image_identity: None,
                    ready: None,
                },
            };
        };

        let ready = AssetTextureSnapshot::from_loaded(
            texture_handle,
            texture.asset(),
            texture.version(),
            texture.identity(),
            image.asset(),
            image.version(),
            image.identity(),
        );
        AssetTextureBatchRecord {
            image: Some(image_id),
            state: AssetTextureStateSnapshot {
                texture_handle,
                texture_version: Some(texture.version()),
                texture_identity: Some(texture.identity()),
                image_handle: Some(image_handle),
                image_version: Some(image.version()),
                image_identity: Some(image.identity()),
                ready,
            },
        }
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn id(&self, handle: TextureHandle) -> Option<AssetTextureBatchId> {
        self.texture_lookup.get(&handle).copied()
    }

    /// Return the complete captured state accepted by this batch's latest
    /// validation barrier. Physical/cache publication paths must still perform
    /// their own final postflight when storage may mutate concurrently
    /// afterward.
    #[must_use]
    pub(crate) fn state(&self, id: AssetTextureBatchId) -> Option<AssetTextureStateSnapshot> {
        self.record_is_accepted(id.index())
            .then_some(self.records.get(id.index())?.state)
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn state_for_handle(
        &self,
        handle: TextureHandle,
    ) -> Option<AssetTextureStateSnapshot> {
        self.state(self.id(handle)?)
    }

    /// Return the retained exact Texture owner for a currently valid loaded
    /// entry. A valid unavailable entry deliberately returns `None` here while
    /// still returning a complete unavailable value from [`Self::state`].
    #[must_use]
    #[cfg(test)]
    pub(crate) fn texture_owner(
        &self,
        id: AssetTextureBatchId,
    ) -> Option<&LoadedAssetSnapshot<Texture>> {
        self.record_is_accepted(id.index())
            .then(|| self.texture_snapshots.get(id.index())?.loaded())
            .flatten()
    }

    /// Return the retained exact Image owner selected by a currently valid
    /// loaded Texture. A stable unavailable Image returns `None`.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn image_owner(
        &self,
        id: AssetTextureBatchId,
    ) -> Option<&LoadedAssetSnapshot<Image>> {
        self.record_is_accepted(id.index()).then_some(())?;
        let image = self.records.get(id.index())?.image?;
        self.image_snapshots.get(image.index())?.loaded()
    }

    #[inline]
    fn record_is_accepted(&self, index: usize) -> bool {
        self.texture_current.get(index).copied() == Some(true)
    }

    fn record_matches_validation(&self, index: usize) -> bool {
        if self.texture_current_before_images.get(index).copied() != Some(true)
            || self.texture_current.get(index).copied() != Some(true)
        {
            return false;
        }
        let Some(texture_snapshot) = self.texture_snapshots.get(index) else {
            return false;
        };
        let Some(record) = self.records.get(index) else {
            return false;
        };
        if record.state.texture_handle != texture_snapshot.handle() {
            return false;
        }

        match texture_snapshot.loaded() {
            None => {
                record.image.is_none()
                    && record.state.texture_version.is_none()
                    && record.state.texture_identity.is_none()
                    && record.state.image_handle.is_none()
                    && record.state.image_version.is_none()
                    && record.state.image_identity.is_none()
                    && record.state.ready.is_none()
            }
            Some(texture) => {
                if record.state.texture_version != Some(texture.version())
                    || record.state.texture_identity != Some(texture.identity())
                    || record.state.image_handle != Some(texture.asset().image)
                {
                    return false;
                }
                let Some(image) = record.image else {
                    return false;
                };
                if self.image_current.get(image.index()).copied() != Some(true) {
                    return false;
                }
                let Some(image_snapshot) = self.image_snapshots.get(image.index()) else {
                    return false;
                };
                if image_snapshot.handle() != texture.asset().image {
                    return false;
                }
                match image_snapshot.loaded() {
                    None => {
                        record.state.image_version.is_none()
                            && record.state.image_identity.is_none()
                            && record.state.ready.is_none()
                    }
                    Some(image) => {
                        record.state.image_version == Some(image.version())
                            && record.state.image_identity == Some(image.identity())
                            && AssetTextureSnapshot::from_loaded(
                                texture_snapshot.handle(),
                                texture.asset(),
                                texture.version(),
                                texture.identity(),
                                image.asset(),
                                image.version(),
                                image.identity(),
                            ) == record.state.ready
                    }
                }
            }
        }
    }

    /// Release every exact owner while retaining outer allocations for reuse.
    pub(crate) fn clear(&mut self) {
        self.records.clear();
        self.image_current.clear();
        self.image_snapshots.clear();
        self.image_lookup.clear();
        self.image_handles.clear();
        self.texture_current.clear();
        self.texture_current_before_images.clear();
        self.texture_snapshots.clear();
        self.texture_lookup.clear();
        self.texture_handles.clear();
        self.initialized = false;
    }

    #[must_use]
    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.texture_handles, ASSET_TEXTURE_BATCH_BASELINE)
            + estimated_hash_map_footprint(&self.texture_lookup, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.texture_snapshots, ASSET_TEXTURE_BATCH_BASELINE)
            + bit_vec_footprint(
                &self.texture_current_before_images,
                ASSET_TEXTURE_BATCH_BASELINE,
            )
            + bit_vec_footprint(&self.texture_current, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.image_handles, ASSET_TEXTURE_BATCH_BASELINE)
            + estimated_hash_map_footprint(&self.image_lookup, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.image_snapshots, ASSET_TEXTURE_BATCH_BASELINE)
            + bit_vec_footprint(&self.image_current, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.records, ASSET_TEXTURE_BATCH_BASELINE)
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        shrink_vec_by_family_ratio(
            &mut self.texture_handles,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.texture_lookup,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.texture_snapshots,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.texture_current_before_images,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.texture_current,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.image_handles,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.image_lookup,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.image_snapshots,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.image_current,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.records,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.texture_handles.is_empty()
            && self.texture_lookup.is_empty()
            && self.texture_snapshots.is_empty()
            && self.texture_current_before_images.is_empty()
            && self.texture_current.is_empty()
            && self.image_handles.is_empty()
            && self.image_lookup.is_empty()
            && self.image_snapshots.is_empty()
            && self.image_current.is_empty()
            && self.records.is_empty()
            && !self.initialized
    }

    #[cfg(test)]
    pub(crate) fn texture_count(&self) -> usize {
        self.texture_snapshots.len()
    }

    #[cfg(test)]
    pub(crate) fn image_count(&self) -> usize {
        self.image_snapshots.len()
    }
}

impl Default for AssetTextureBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn bit_vec_footprint(value: &Vec<bool>, baseline_capacity: usize) -> CapacityUsage {
    CapacityUsage::new(
        value.len().div_ceil(8),
        value.capacity().div_ceil(8),
        baseline_capacity.div_ceil(8),
    )
}

#[cfg(test)]
mod tests {
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
    fn asset_texture_batch_id_uses_the_nonzero_option_niche() {
        assert_eq!(std::mem::size_of::<AssetTextureBatchId>(), 4);
        assert_eq!(std::mem::size_of::<Option<AssetTextureBatchId>>(), 4);
        assert_eq!(std::mem::size_of::<Option<AssetImageBatchId>>(), 4);
        assert_eq!(AssetTextureBatchId::from_index(0).index(), 0);
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
}
