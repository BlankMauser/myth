//! Frame-scoped logical imports for AssetServer textures.
//!
//! Graph construction captures an exact CPU-only [`AssetTextureSnapshot`]
//! and registers an external, non-root texture. A producer node must declare a
//! write to that root and fill its frame slot during live `prepare`; consumers
//! then resolve the physical view through the Copy [`AssetTextureImportId`]
//! stored in [`TextureBacking`](super::types::TextureBacking).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use rustc_hash::FxHashMap;

use crate::core::gpu::{AssetTextureMaterialization, AssetTextureSnapshot, MipmapRequest, Tracked};

use super::types::{TextureDesc, TextureNodeId};

/// Copy index into GraphStorage's destructible asset-import table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetTextureImportId(u32);

impl AssetTextureImportId {
    #[inline]
    #[must_use]
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("asset texture import index exceeds u32"))
    }

    #[inline]
    #[must_use]
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Small logical key used only to deduplicate imports within one graph.
///
/// Asset handles and storage versions provide collision-free logical identity;
/// `logical_version` additionally fingerprints the resolved descriptor. The
/// physical [`AssetTextureGenerationKey`](crate::core::gpu::AssetTextureGenerationKey)
/// is deliberately *not* used here: it does not exist until the live upload
/// node materializes this snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AssetTextureLookupKey {
    texture_handle: myth_assets::TextureHandle,
    image_handle: myth_assets::ImageHandle,
    texture_version: u32,
    texture_identity: u64,
    image_version: u32,
    image_identity: u64,
    logical_version: u64,
}

impl From<AssetTextureSnapshot> for AssetTextureLookupKey {
    fn from(snapshot: AssetTextureSnapshot) -> Self {
        Self {
            texture_handle: snapshot.texture_handle,
            image_handle: snapshot.image_handle,
            texture_version: snapshot.texture_version,
            texture_identity: snapshot.texture_identity,
            image_version: snapshot.image_version,
            image_identity: snapshot.image_identity,
            logical_version: snapshot.logical_version,
        }
    }
}

/// One destructible frame slot owned by GraphStorage.
///
/// A claimed legacy mip request is independent from the optional materialized
/// view. This is intentional: if the CPU asset changes between snapshot and
/// live prepare, exact-version materialization returns no view, but an older
/// request already claimed from the compatibility queue must remain owned and
/// available for the legacy side-effect path.
pub(crate) struct AssetTextureImportEntry {
    pub(crate) node: TextureNodeId,
    pub(crate) snapshot: AssetTextureSnapshot,
    materialization: OnceLock<AssetTextureMaterialization>,
    claimed_mipmap_request: Option<MipmapRequest>,
    mipmaps_encoded: AtomicBool,
}

impl AssetTextureImportEntry {
    fn new(node: TextureNodeId, snapshot: AssetTextureSnapshot) -> Self {
        Self {
            node,
            snapshot,
            materialization: OnceLock::new(),
            claimed_mipmap_request: None,
            mipmaps_encoded: AtomicBool::new(false),
        }
    }
}

/// GraphStorage-owned lookup and frame slots for logical asset imports.
#[derive(Default)]
pub(crate) struct AssetTextureImportStore {
    entries: Vec<AssetTextureImportEntry>,
    lookup: FxHashMap<AssetTextureLookupKey, AssetTextureImportId>,
}

impl AssetTextureImportStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn clear(&mut self) {
        self.lookup.clear();
        // Drops tracked views and any unconsumed Drop-bearing mip requests.
        self.entries.clear();
    }

    #[must_use]
    pub(crate) fn find(&self, snapshot: AssetTextureSnapshot) -> Option<AssetTextureImportId> {
        self.lookup
            .get(&AssetTextureLookupKey::from(snapshot))
            .copied()
    }

    pub(crate) fn insert(
        &mut self,
        id: AssetTextureImportId,
        node: TextureNodeId,
        snapshot: AssetTextureSnapshot,
    ) {
        assert_eq!(
            id.index(),
            self.entries.len(),
            "asset texture import IDs must be densely allocated"
        );
        let old = self
            .lookup
            .insert(AssetTextureLookupKey::from(snapshot), id);
        assert!(old.is_none(), "asset texture import key registered twice");
        self.entries
            .push(AssetTextureImportEntry::new(node, snapshot));
    }

    #[must_use]
    pub(crate) fn next_id(&self) -> AssetTextureImportId {
        AssetTextureImportId::from_index(self.entries.len())
    }

    #[must_use]
    pub(crate) fn entry(&self, id: AssetTextureImportId) -> &AssetTextureImportEntry {
        self.entries
            .get(id.index())
            .expect("asset texture import ID is out of bounds")
    }

    pub(crate) fn claim_mipmap_request(
        &mut self,
        id: AssetTextureImportId,
        request: MipmapRequest,
    ) {
        let entry = self
            .entries
            .get_mut(id.index())
            .expect("asset texture import ID is out of bounds");
        assert_eq!(
            entry.snapshot.texture_handle, request.texture_handle,
            "claimed mip request belongs to a different texture handle"
        );
        assert!(
            entry.claimed_mipmap_request.is_none(),
            "asset texture import already claimed a legacy mip request"
        );
        entry.claimed_mipmap_request = Some(request);
    }

    #[must_use]
    pub(crate) fn resolver(&self) -> AssetTextureResolver<'_> {
        AssetTextureResolver::new(&self.entries)
    }

    /// Publish only requests encoded by live nodes, then immediately release
    /// every frame-owned request/view owner. Call after queue submission.
    pub(crate) fn commit_submitted_mipmaps(&mut self) -> usize {
        let published = {
            let resolver = self.resolver();
            let mut published = 0;
            for index in 0..self.entries.len() {
                let id = AssetTextureImportId::from_index(index);
                if !resolver.mipmaps_encoded(id) {
                    continue;
                }
                for request in resolver.mipmap_requests(id) {
                    request.mark_submitted();
                    published += 1;
                }
            }
            published
        };
        self.clear();
        published
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.lookup.is_empty()
    }
}

/// Narrow post-build capability shared by live prepare and execute contexts.
#[derive(Clone, Copy)]
pub(crate) struct AssetTextureResolver<'a> {
    entries: &'a [AssetTextureImportEntry],
}

impl<'a> AssetTextureResolver<'a> {
    #[must_use]
    pub(crate) const fn new(entries: &'a [AssetTextureImportEntry]) -> Self {
        Self { entries }
    }

    fn entry(self, id: AssetTextureImportId) -> &'a AssetTextureImportEntry {
        self.entries
            .get(id.index())
            .expect("asset texture import ID is out of bounds")
    }

    #[must_use]
    pub(crate) fn snapshot(self, id: AssetTextureImportId) -> AssetTextureSnapshot {
        self.entry(id).snapshot
    }

    /// Fill a frame slot exactly once from a live upload node's prepare step.
    ///
    /// The outcome may contain no view when asset versions drifted. Its mip
    /// request remains owned by GraphStorage independently of that outcome.
    pub(crate) fn materialize(
        self,
        id: AssetTextureImportId,
        materialization: AssetTextureMaterialization,
    ) {
        assert!(
            self.entry(id).materialization.set(materialization).is_ok(),
            "asset texture import materialized more than once"
        );
    }

    #[must_use]
    pub(crate) fn materialization(
        self,
        id: AssetTextureImportId,
    ) -> Option<&'a AssetTextureMaterialization> {
        self.entry(id).materialization.get()
    }

    #[must_use]
    pub(crate) fn view(self, id: AssetTextureImportId) -> Option<&'a Tracked<wgpu::TextureView>> {
        self.materialization(id)?.texture.as_ref().map(|v| &v.view)
    }

    #[must_use]
    pub(crate) fn claimed_mipmap_request(
        self,
        id: AssetTextureImportId,
    ) -> Option<&'a MipmapRequest> {
        self.entry(id).claimed_mipmap_request.as_ref()
    }

    #[must_use]
    pub(crate) fn materialized_mipmap_request(
        self,
        id: AssetTextureImportId,
    ) -> Option<&'a MipmapRequest> {
        self.materialization(id)?.mipmap_request.as_ref()
    }

    /// Iterate the claimed compatibility request and the current live
    /// materialization request, deduplicated by exact physical generation.
    #[must_use]
    pub(crate) fn mipmap_requests(
        self,
        id: AssetTextureImportId,
    ) -> AssetTextureMipmapRequests<'a> {
        AssetTextureMipmapRequests {
            claimed: self.claimed_mipmap_request(id),
            materialized: self.materialized_mipmap_request(id),
            next: 0,
        }
    }

    pub(crate) fn mark_mipmaps_encoded(self, id: AssetTextureImportId) {
        self.entry(id)
            .mipmaps_encoded
            .store(true, Ordering::Release);
    }

    #[must_use]
    pub(crate) fn mipmaps_encoded(self, id: AssetTextureImportId) -> bool {
        self.entry(id).mipmaps_encoded.load(Ordering::Acquire)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn node(self, id: AssetTextureImportId) -> TextureNodeId {
        self.entry(id).node
    }
}

pub(crate) struct AssetTextureMipmapRequests<'a> {
    claimed: Option<&'a MipmapRequest>,
    materialized: Option<&'a MipmapRequest>,
    next: u8,
}

impl<'a> Iterator for AssetTextureMipmapRequests<'a> {
    type Item = &'a MipmapRequest;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.next {
                0 => {
                    self.next = 1;
                    if let Some(claimed) = self.claimed {
                        return Some(claimed);
                    }
                }
                1 => {
                    self.next = 2;
                    if let Some(materialized) = self.materialized
                        && self
                            .claimed
                            .is_none_or(|claimed| claimed.key() != materialized.key())
                    {
                        return Some(materialized);
                    }
                }
                _ => return None,
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(2))
    }
}

#[inline]
#[must_use]
pub(crate) fn snapshot_texture_desc(snapshot: AssetTextureSnapshot) -> TextureDesc {
    TextureDesc::new(
        snapshot.extent.width,
        snapshot.extent.height,
        snapshot.extent.depth_or_array_layers,
        snapshot.mip_count,
        1,
        snapshot.dimension,
        snapshot.format,
        snapshot.usage,
    )
}
