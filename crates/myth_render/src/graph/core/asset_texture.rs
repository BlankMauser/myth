//! Frame-scoped logical imports for AssetServer textures.
//!
//! Graph construction captures an exact CPU-only [`AssetTextureSnapshot`]
//! and registers an external, non-root texture. The root is a lazy import: its
//! compiled `first_use` is the exact DPE demand signal. After DPE, a graph-level
//! batch fills only live frame slots before any `PassNode::prepare`; consumers
//! then resolve the physical view through the Copy [`AssetTextureImportId`] stored in
//! [`TextureBacking`](super::types::TextureBacking).

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
/// physical [`PhysicalImageGenerationKey`](crate::core::gpu::PhysicalImageGenerationKey)
/// is deliberately *not* used here: it does not exist until the post-DPE graph
/// batch materializes this snapshot.
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
/// The physical view and optional mip request are filled by the post-DPE graph
/// materialization batch. Both remain graph-attempt-owned until submission or
/// the RenderGraph RAII boundary releases an abandoned attempt.
pub(crate) struct AssetTextureImportEntry {
    pub(crate) node: TextureNodeId,
    pub(crate) snapshot: AssetTextureSnapshot,
    mip_ready: Option<TextureNodeId>,
    materialization: OnceLock<AssetTextureMaterialization>,
    mipmaps_encoded: AtomicBool,
}

impl AssetTextureImportEntry {
    fn new(node: TextureNodeId, snapshot: AssetTextureSnapshot) -> Self {
        Self {
            node,
            snapshot,
            mip_ready: None,
            materialization: OnceLock::new(),
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

    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return the unique mip-ready alias registered for this lazy import.
    ///
    /// The root itself is the base-ready version and needs no proxy producer.
    /// Keeping the optional alias in the attempt-owned entry prevents a second
    /// consumer (for example a material plus Skybox) from assigning another
    /// SSA writer to the shared root.
    #[must_use]
    pub(crate) fn mip_ready(&self, id: AssetTextureImportId) -> Option<TextureNodeId> {
        self.entry(id).mip_ready
    }

    pub(crate) fn set_mip_ready(&mut self, id: AssetTextureImportId, mip_ready: TextureNodeId) {
        let entry = self
            .entries
            .get_mut(id.index())
            .expect("asset texture import ID is out of bounds");
        assert!(
            entry.mip_ready.replace(mip_ready).is_none(),
            "asset texture import mipmap producer registered twice"
        );
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

    /// Fill a frame slot exactly once from the post-DPE graph batch.
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
    pub(crate) fn materialized_mipmap_request(
        self,
        id: AssetTextureImportId,
    ) -> Option<&'a MipmapRequest> {
        self.materialization(id)?.mipmap_request.as_ref()
    }

    /// Iterate the exact physical mip attempt produced by this live logical
    /// import, if one is required for the selected generation.
    pub(crate) fn mipmap_requests(
        self,
        id: AssetTextureImportId,
    ) -> impl Iterator<Item = &'a MipmapRequest> {
        self.materialized_mipmap_request(id).into_iter()
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
