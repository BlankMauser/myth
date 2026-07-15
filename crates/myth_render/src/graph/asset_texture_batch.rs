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

use crate::core::gpu::{
    AssetTexturePhysicalBinding, AssetTextureSnapshot, AssetTextureStateSnapshot,
};

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
    pub(crate) const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AssetImageSnapshotId(NonZeroU32);

impl AssetImageSnapshotId {
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

/// Retained sparse Image-snapshot-ID to selected-slot join.
///
/// The table is cleared without releasing capacity between postflights and is
/// never iterated for externally visible ordering. Its allocation therefore
/// follows the unique selected Image count, not the complete captured domain.
#[derive(Default)]
struct SelectedImageSparseLookup {
    slots: FxHashMap<AssetImageSnapshotId, usize>,
}

impl SelectedImageSparseLookup {
    #[inline]
    fn begin(&mut self) {
        self.slots.clear();
    }

    #[inline]
    fn get_or_insert(&mut self, image: AssetImageSnapshotId, vacant_slot: usize) -> (usize, bool) {
        match self.slots.entry(image) {
            std::collections::hash_map::Entry::Occupied(entry) => (*entry.get(), false),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(vacant_slot);
                (vacant_slot, true)
            }
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.slots.clear();
    }

    #[inline]
    #[cfg(test)]
    fn is_active_empty(&self) -> bool {
        self.slots.is_empty()
    }

    fn capacity_usage(&self, baseline_capacity: usize) -> CapacityUsage {
        estimated_hash_map_footprint(&self.slots, baseline_capacity)
    }

    fn trim_capacity(
        &mut self,
        baseline_capacity: usize,
        target_bytes: usize,
        retained_bytes: usize,
    ) {
        shrink_hash_map_by_family_ratio(
            &mut self.slots,
            baseline_capacity,
            target_bytes,
            retained_bytes,
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct AssetTextureBatchRecord {
    image: Option<AssetImageSnapshotId>,
    state: AssetTextureStateSnapshot,
    /// Exact graph-attempt physical result published by the canonical lazy
    /// import. `None` is also the deliberate result for unavailable or
    /// rejected materialization.
    physical_binding: Option<AssetTexturePhysicalBinding>,
    physical_published: bool,
}

#[derive(Clone, Copy)]
enum SelectedPostflightState {
    ReadyOnly,
    Complete,
}

/// Exact mutation epochs surrounding the full retained T -> I -> T capture.
///
/// This is only an invalidation token. Once either storage changes, selected
/// postflights fall back to the complete per-entry validation path; epoch
/// equality never repairs or replaces a stale retained owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AssetTextureStorageEpochs {
    textures: u64,
    images: u64,
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
    image_lookup: FxHashMap<ImageHandle, AssetImageSnapshotId>,
    image_snapshots: Vec<AssetSnapshot<ImageHandle, Image>>,
    image_current: Vec<bool>,

    records: Vec<AssetTextureBatchRecord>,

    /// Reused post-DPE T -> I -> T validation scratch. Texture positions are
    /// input-aligned, while Image positions are stable-deduplicated in the
    /// first selected Texture occurrence order.
    selected_texture_indices: Vec<usize>,
    selected_image_indices: Vec<usize>,
    selected_image_lookup: SelectedImageSparseLookup,
    selected_texture_image_slots: Vec<Option<usize>>,
    selected_texture_current_before_images: Vec<bool>,
    selected_image_current: Vec<bool>,
    selected_current: Vec<bool>,
    stable_storage_epochs: Option<AssetTextureStorageEpochs>,
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
            selected_texture_indices: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            selected_image_indices: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            selected_image_lookup: SelectedImageSparseLookup::default(),
            selected_texture_image_slots: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            selected_texture_current_before_images: Vec::with_capacity(
                ASSET_TEXTURE_BATCH_BASELINE,
            ),
            selected_image_current: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            selected_current: Vec::with_capacity(ASSET_TEXTURE_BATCH_BASELINE),
            stable_storage_epochs: None,
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
                let index = AssetImageSnapshotId::from_index(self.image_handles.len());
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
        let texture_epoch_before = assets.textures.global_version();
        let image_epoch_before = assets.images.global_version();
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

        // Read in reverse order so both storage epochs bracket the complete
        // cross-storage validation. A writer announces its mutation before
        // touching a slot, so equality proves no write started in the window.
        let image_epoch_after = assets.images.global_version();
        let texture_epoch_after = assets.textures.global_version();
        self.stable_storage_epochs = (texture_epoch_before == texture_epoch_after
            && image_epoch_before == image_epoch_after)
            .then_some(AssetTextureStorageEpochs {
                textures: texture_epoch_after,
                images: image_epoch_after,
            });
    }

    fn derive_record(
        &self,
        texture_snapshot: &AssetSnapshot<TextureHandle, Texture>,
    ) -> AssetTextureBatchRecord {
        let texture_handle = texture_snapshot.handle();
        let Some(texture) = texture_snapshot.loaded() else {
            return AssetTextureBatchRecord {
                image: None,
                physical_binding: None,
                physical_published: false,
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
                physical_binding: None,
                physical_published: false,
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
            physical_binding: None,
            physical_published: false,
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

    /// Publish the exact physical view/sampler selected by this attempt's
    /// canonical Asset Texture import.
    ///
    /// The logical ready snapshot must match the retained owner record. The
    /// import phase owns the sole write; later Material preparation reads this
    /// Copy token instead of reopening ResourceManager residency dictionaries.
    pub(crate) fn publish_physical_binding(
        &mut self,
        id: AssetTextureBatchId,
        expected: AssetTextureSnapshot,
        binding: Option<AssetTexturePhysicalBinding>,
    ) {
        let record = self
            .records
            .get_mut(id.index())
            .expect("asset texture physical result has an invalid owner plan");
        assert_eq!(
            record.state.ready,
            Some(expected),
            "asset texture physical result does not match its retained snapshot"
        );
        assert!(
            !record.physical_published,
            "asset texture physical result was published twice"
        );
        record.physical_binding = binding;
        record.physical_published = true;
    }

    #[inline]
    #[must_use]
    pub(crate) fn physical_binding(
        &self,
        id: AssetTextureBatchId,
    ) -> Option<AssetTexturePhysicalBinding> {
        self.records
            .get(id.index())
            .filter(|record| record.physical_published)?
            .physical_binding
    }

    /// Borrow the exact ready owners retained by this graph attempt.
    ///
    /// This is intentionally narrower than exposing either owner table. A
    /// candidate builder must name the exact ready snapshot it intends to
    /// materialize, and an invalid ID or unavailable record fails closed.
    /// Storage currency is established separately by
    /// [`Self::postflight_selected`] after candidate construction.
    #[must_use]
    pub(crate) fn ready_owners(
        &self,
        id: AssetTextureBatchId,
        expected: AssetTextureSnapshot,
    ) -> Option<(&Texture, &Image)> {
        let index = id.index();
        if !self.record_is_accepted(index) || self.records.get(index)?.state.ready != Some(expected)
        {
            return None;
        }

        let texture_snapshot = self.texture_snapshots.get(index)?;
        let texture = texture_snapshot.loaded()?;
        let image_id = self.records.get(index)?.image?;
        let image = self.image_snapshots.get(image_id.index())?.loaded()?;
        debug_assert_eq!(
            AssetTextureSnapshot::from_loaded(
                texture_snapshot.handle(),
                texture.asset(),
                texture.version(),
                texture.identity(),
                image.asset(),
                image.version(),
                image.identity(),
            ),
            Some(expected),
            "accepted Asset Texture record drifted from its retained owners"
        );
        Some((texture.asset().as_ref(), image.asset().as_ref()))
    }

    /// Revalidate the canonical DPE-live ready subset using one bounded
    /// T-before -> unique-I -> T-after sequence.
    ///
    /// Results remain input-aligned. Only an initially accepted ready record
    /// whose retained Texture and Image owners are still exact is `true`;
    /// invalid IDs and unavailable records fail closed. Image owners shared by
    /// multiple selected Textures are checked once in stable first-occurrence
    /// order.
    pub(crate) fn postflight_selected(
        &mut self,
        assets: &AssetServer,
        ids: &[AssetTextureBatchId],
    ) -> &[bool] {
        self.postflight_selected_impl(
            assets,
            ids,
            SelectedPostflightState::ReadyOnly,
            true,
            || {},
            || {},
        )
    }

    /// Revalidate complete captured states for a canonical selected subset.
    ///
    /// Unlike [`Self::postflight_selected`], this accepts every stable state:
    /// ready Texture/Image pairs, unavailable Textures, and loaded Textures
    /// whose Image remains unavailable. Results stay input-aligned and any
    /// Texture or Image transition still fails closed. Image owners are
    /// stable-deduplicated and checked once through the same bounded
    /// T-before -> unique-I -> T-after scratch path.
    pub(crate) fn postflight_states_selected(
        &mut self,
        assets: &AssetServer,
        ids: &[AssetTextureBatchId],
    ) -> &[bool] {
        self.postflight_selected_impl(
            assets,
            ids,
            SelectedPostflightState::Complete,
            true,
            || {},
            || {},
        )
    }

    #[cfg(test)]
    pub(crate) fn postflight_selected_with_hooks(
        &mut self,
        assets: &AssetServer,
        ids: &[AssetTextureBatchId],
        after_texture_before: impl FnOnce(),
        after_images: impl FnOnce(),
    ) -> &[bool] {
        self.postflight_selected_impl(
            assets,
            ids,
            SelectedPostflightState::ReadyOnly,
            false,
            after_texture_before,
            after_images,
        )
    }

    #[cfg(test)]
    fn postflight_states_selected_with_hooks(
        &mut self,
        assets: &AssetServer,
        ids: &[AssetTextureBatchId],
        after_texture_before: impl FnOnce(),
        after_images: impl FnOnce(),
    ) -> &[bool] {
        self.postflight_selected_impl(
            assets,
            ids,
            SelectedPostflightState::Complete,
            false,
            after_texture_before,
            after_images,
        )
    }

    fn postflight_selected_impl(
        &mut self,
        assets: &AssetServer,
        ids: &[AssetTextureBatchId],
        state: SelectedPostflightState,
        allow_epoch_fast_path: bool,
        after_texture_before: impl FnOnce(),
        after_images: impl FnOnce(),
    ) -> &[bool] {
        assert!(self.initialized, "asset texture batch was not captured");

        self.selected_texture_indices.clear();
        self.selected_image_indices.clear();
        self.selected_image_lookup.begin();
        self.selected_texture_image_slots.clear();
        self.selected_texture_current_before_images.clear();
        self.selected_image_current.clear();
        self.selected_current.clear();

        if allow_epoch_fast_path && self.storage_epochs_are_current(assets) {
            self.selected_current.reserve(ids.len());
            for &id in ids {
                let eligible = self.record_is_eligible(id.index(), state);
                self.selected_current.push(eligible);
            }
            return &self.selected_current;
        }

        self.selected_texture_indices.reserve(ids.len());
        self.selected_texture_image_slots.reserve(ids.len());
        for &id in ids {
            let index = id.index();
            self.selected_texture_indices.push(index);

            let image_slot = self
                .records
                .get(index)
                .filter(|record| {
                    self.record_is_accepted(index)
                        && match state {
                            SelectedPostflightState::ReadyOnly => record.state.ready.is_some(),
                            SelectedPostflightState::Complete => true,
                        }
                })
                .and_then(|record| record.image)
                .map(|image| {
                    let (slot, inserted) = self
                        .selected_image_lookup
                        .get_or_insert(image, self.selected_image_indices.len());
                    if inserted {
                        self.selected_image_indices.push(image.index());
                    }
                    slot
                });
            self.selected_texture_image_slots.push(image_slot);
        }

        assets.textures.revalidate_selected_batched(
            &self.texture_snapshots,
            &self.selected_texture_indices,
            &mut self.selected_texture_current_before_images,
        );
        after_texture_before();
        assets.images.revalidate_selected_batched(
            &self.image_snapshots,
            &self.selected_image_indices,
            &mut self.selected_image_current,
        );
        after_images();
        assets.textures.revalidate_selected_batched(
            &self.texture_snapshots,
            &self.selected_texture_indices,
            &mut self.selected_current,
        );

        debug_assert_eq!(ids.len(), self.selected_current.len());
        for (selected_index, &id) in ids.iter().enumerate() {
            let record = self
                .records
                .get(id.index())
                .filter(|_| self.record_is_accepted(id.index()));
            let state_is_eligible = self.record_is_eligible(id.index(), state);
            let image_is_current = record.is_some_and(|record| match record.image {
                Some(_) => {
                    self.selected_texture_image_slots[selected_index]
                        .and_then(|slot| self.selected_image_current.get(slot))
                        .copied()
                        == Some(true)
                }
                None => matches!(state, SelectedPostflightState::Complete),
            });
            self.selected_current[selected_index] = self
                .selected_texture_current_before_images
                .get(selected_index)
                .copied()
                == Some(true)
                && self.selected_current[selected_index]
                && image_is_current
                && state_is_eligible;
        }

        &self.selected_current
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

    #[inline]
    fn record_is_eligible(&self, index: usize, state: SelectedPostflightState) -> bool {
        self.records
            .get(index)
            .filter(|_| self.record_is_accepted(index))
            .is_some_and(|record| match state {
                SelectedPostflightState::ReadyOnly => record.state.ready.is_some(),
                SelectedPostflightState::Complete => true,
            })
    }

    #[inline]
    fn storage_epochs_are_current(&self, assets: &AssetServer) -> bool {
        let Some(expected) = self.stable_storage_epochs else {
            return false;
        };

        // The first acquire load is the fast-path linearization point. A
        // writer increments before changing storage, so a later Texture write
        // is ordered after this validation; an Image write that has already
        // started is observed by the second load and closes the fast path.
        assets.textures.global_version() == expected.textures
            && assets.images.global_version() == expected.images
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
                        if record.state.image_version != Some(image.version())
                            || record.state.image_identity != Some(image.identity())
                        {
                            return false;
                        }
                        debug_assert_eq!(
                            AssetTextureSnapshot::from_loaded(
                                texture_snapshot.handle(),
                                texture.asset(),
                                texture.version(),
                                texture.identity(),
                                image.asset(),
                                image.version(),
                                image.identity(),
                            ),
                            record.state.ready,
                            "retained Asset Texture owners no longer match their derived record"
                        );
                        true
                    }
                }
            }
        }
    }

    /// Release every exact owner while retaining outer allocations for reuse.
    pub(crate) fn clear(&mut self) {
        self.selected_current.clear();
        self.selected_image_current.clear();
        self.selected_texture_current_before_images.clear();
        self.selected_texture_image_slots.clear();
        self.selected_image_lookup.clear();
        self.selected_image_indices.clear();
        self.selected_texture_indices.clear();
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
        self.stable_storage_epochs = None;
        self.initialized = false;
    }

    #[must_use]
    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.texture_handles, ASSET_TEXTURE_BATCH_BASELINE)
            + estimated_hash_map_footprint(&self.texture_lookup, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.texture_snapshots, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(
                &self.texture_current_before_images,
                ASSET_TEXTURE_BATCH_BASELINE,
            )
            + vec_footprint(&self.texture_current, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.image_handles, ASSET_TEXTURE_BATCH_BASELINE)
            + estimated_hash_map_footprint(&self.image_lookup, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.image_snapshots, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.image_current, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.records, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.selected_texture_indices, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.selected_image_indices, ASSET_TEXTURE_BATCH_BASELINE)
            + self
                .selected_image_lookup
                .capacity_usage(ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(
                &self.selected_texture_image_slots,
                ASSET_TEXTURE_BATCH_BASELINE,
            )
            + vec_footprint(
                &self.selected_texture_current_before_images,
                ASSET_TEXTURE_BATCH_BASELINE,
            )
            + vec_footprint(&self.selected_image_current, ASSET_TEXTURE_BATCH_BASELINE)
            + vec_footprint(&self.selected_current, ASSET_TEXTURE_BATCH_BASELINE)
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
        shrink_vec_by_family_ratio(
            &mut self.selected_texture_indices,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.selected_image_indices,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        self.selected_image_lookup.trim_capacity(
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.selected_texture_image_slots,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.selected_texture_current_before_images,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.selected_image_current,
            ASSET_TEXTURE_BATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.selected_current,
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
            && self.selected_texture_indices.is_empty()
            && self.selected_image_indices.is_empty()
            && self.selected_image_lookup.is_active_empty()
            && self.selected_texture_image_slots.is_empty()
            && self.selected_texture_current_before_images.is_empty()
            && self.selected_image_current.is_empty()
            && self.selected_current.is_empty()
            && self.stable_storage_epochs.is_none()
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

#[cfg(test)]
mod tests;
