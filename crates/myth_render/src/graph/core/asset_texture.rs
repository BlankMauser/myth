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

use myth_assets::AssetServer;
use rustc_hash::FxHashMap;

use crate::core::ResourceManager;
use crate::core::gpu::{
    AssetTextureCandidateBatch, AssetTextureCandidateId, AssetTextureMaterialization,
    AssetTextureSnapshot, ExactAssetTextureResidency, MaterializedAssetTexture, MipmapRequest,
    Tracked,
};
use crate::graph::asset_texture_batch::{AssetTextureBatch, AssetTextureBatchId};
use crate::graph::capacity::{
    CapacityUsage, estimated_hash_map_footprint, shrink_hash_map_by_family_ratio,
    shrink_vec_by_family_ratio, vec_footprint,
};

use super::types::{ResourceRecord, TextureDesc, TextureNodeId};

const ASSET_TEXTURE_IMPORT_BASELINE: usize = 32;

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

#[derive(Debug, Clone, Copy)]
struct PlannedAssetTextureImport {
    import: AssetTextureImportId,
    /// Canonical sampling-ready root or mip alias chosen by the first
    /// built-in consumer. The exact snapshot makes that policy stable for the
    /// rest of graph construction.
    ready: Option<TextureNodeId>,
}

#[derive(Clone, Copy)]
enum LiveAssetTexturePath {
    /// Exact logical lease, binding, sampler, physical generation, and full
    /// image spec observed before the selected AssetStorage postflight.
    Resident(ExactAssetTextureResidency),
    /// Cold, refreshed, or otherwise non-exact state owned by the original
    /// publication transaction.
    Candidate(AssetTextureCandidateId),
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
    /// Canonical graph-attempt owner plan. Built-in production imports always
    /// carry this ID so post-DPE preparation never reopens AssetStorage.
    pub(crate) owner_plan: Option<AssetTextureBatchId>,
    mip_ready: Option<TextureNodeId>,
    materialization: OnceLock<AssetTextureMaterialization>,
    mipmaps_encoded: AtomicBool,
}

impl AssetTextureImportEntry {
    fn new(
        node: TextureNodeId,
        snapshot: AssetTextureSnapshot,
        owner_plan: Option<AssetTextureBatchId>,
    ) -> Self {
        Self {
            node,
            snapshot,
            owner_plan,
            mip_ready: None,
            materialization: OnceLock::new(),
            mipmaps_encoded: AtomicBool::new(false),
        }
    }
}

/// GraphStorage-owned lookup and frame slots for logical asset imports.
pub(crate) struct AssetTextureImportStore {
    entries: Vec<AssetTextureImportEntry>,
    lookup: FxHashMap<AssetTextureLookupKey, AssetTextureImportId>,
    /// Dense canonical-plan join. Repeated shadow/material/skybox consumers
    /// avoid re-hashing the complete snapshot and re-querying mip residency.
    planned: Vec<Option<PlannedAssetTextureImport>>,
    /// Reused post-DPE candidate transaction and input-aligned joins.
    candidates: AssetTextureCandidateBatch,
    live_imports: Vec<AssetTextureImportId>,
    live_owner_plans: Vec<AssetTextureBatchId>,
    live_paths: Vec<LiveAssetTexturePath>,
    live_current: Vec<bool>,
    candidate_current: Vec<bool>,
}

impl Default for AssetTextureImportStore {
    fn default() -> Self {
        Self {
            entries: Vec::with_capacity(ASSET_TEXTURE_IMPORT_BASELINE),
            lookup: FxHashMap::default(),
            planned: Vec::with_capacity(ASSET_TEXTURE_IMPORT_BASELINE),
            candidates: AssetTextureCandidateBatch::new(),
            live_imports: Vec::with_capacity(ASSET_TEXTURE_IMPORT_BASELINE),
            live_owner_plans: Vec::with_capacity(ASSET_TEXTURE_IMPORT_BASELINE),
            live_paths: Vec::with_capacity(ASSET_TEXTURE_IMPORT_BASELINE),
            live_current: Vec::with_capacity(ASSET_TEXTURE_IMPORT_BASELINE),
            candidate_current: Vec::with_capacity(ASSET_TEXTURE_IMPORT_BASELINE),
        }
    }
}

impl AssetTextureImportStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.entries, ASSET_TEXTURE_IMPORT_BASELINE)
            + estimated_hash_map_footprint(&self.lookup, ASSET_TEXTURE_IMPORT_BASELINE)
            + vec_footprint(&self.planned, ASSET_TEXTURE_IMPORT_BASELINE)
            + CapacityUsage::new(
                self.candidates.active_bytes(),
                self.candidates.retained_bytes(),
                AssetTextureCandidateBatch::baseline_bytes(),
            )
            + vec_footprint(&self.live_imports, ASSET_TEXTURE_IMPORT_BASELINE)
            + vec_footprint(&self.live_owner_plans, ASSET_TEXTURE_IMPORT_BASELINE)
            + vec_footprint(&self.live_paths, ASSET_TEXTURE_IMPORT_BASELINE)
            + vec_footprint(&self.live_current, ASSET_TEXTURE_IMPORT_BASELINE)
            + vec_footprint(&self.candidate_current, ASSET_TEXTURE_IMPORT_BASELINE)
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        shrink_vec_by_family_ratio(
            &mut self.entries,
            ASSET_TEXTURE_IMPORT_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.lookup,
            ASSET_TEXTURE_IMPORT_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.planned,
            ASSET_TEXTURE_IMPORT_BASELINE,
            target_bytes,
            retained_bytes,
        );
        self.candidates.trim_capacity(target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(
            &mut self.live_imports,
            ASSET_TEXTURE_IMPORT_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.live_owner_plans,
            ASSET_TEXTURE_IMPORT_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.live_paths,
            ASSET_TEXTURE_IMPORT_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.live_current,
            ASSET_TEXTURE_IMPORT_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.candidate_current,
            ASSET_TEXTURE_IMPORT_BASELINE,
            target_bytes,
            retained_bytes,
        );
    }

    pub(crate) fn clear(&mut self) {
        self.candidate_current.clear();
        self.live_current.clear();
        self.live_paths.clear();
        self.live_owner_plans.clear();
        self.live_imports.clear();
        self.candidates.clear();
        self.planned.clear();
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

    #[must_use]
    pub(crate) fn find_planned(
        &self,
        owner_plan: AssetTextureBatchId,
    ) -> Option<AssetTextureImportId> {
        self.planned
            .get(owner_plan.index())
            .copied()
            .flatten()
            .map(|planned| planned.import)
    }

    #[must_use]
    pub(crate) fn planned_ready(&self, owner_plan: AssetTextureBatchId) -> Option<TextureNodeId> {
        self.planned
            .get(owner_plan.index())
            .copied()
            .flatten()
            .and_then(|planned| planned.ready)
    }

    pub(crate) fn set_planned_ready(
        &mut self,
        owner_plan: AssetTextureBatchId,
        ready: TextureNodeId,
    ) {
        let planned = self
            .planned
            .get_mut(owner_plan.index())
            .and_then(Option::as_mut)
            .expect("asset texture owner plan has no logical import");
        assert!(
            planned.ready.replace(ready).is_none_or(|old| old == ready),
            "asset texture owner plan resolved conflicting sampling roots"
        );
    }

    pub(crate) fn insert(
        &mut self,
        id: AssetTextureImportId,
        node: TextureNodeId,
        snapshot: AssetTextureSnapshot,
        owner_plan: Option<AssetTextureBatchId>,
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
            .push(AssetTextureImportEntry::new(node, snapshot, owner_plan));
        if let Some(owner_plan) = owner_plan {
            self.bind_owner_plan(id, owner_plan);
        }
    }

    /// Attach the canonical retained-owner plan to an import first registered
    /// through the generic API. Repeated planned imports must agree exactly.
    pub(crate) fn bind_owner_plan(
        &mut self,
        id: AssetTextureImportId,
        owner_plan: AssetTextureBatchId,
    ) {
        let entry = self
            .entries
            .get_mut(id.index())
            .expect("asset texture import ID is out of bounds");
        match entry.owner_plan {
            Some(existing) => assert_eq!(
                existing, owner_plan,
                "asset texture snapshot registered from conflicting owner plans"
            ),
            None => entry.owner_plan = Some(owner_plan),
        }
        if self.planned.len() <= owner_plan.index() {
            self.planned.resize(owner_plan.index() + 1, None);
        }
        match self.planned[owner_plan.index()] {
            Some(planned) => assert_eq!(
                planned.import, id,
                "asset texture owner plan registered to conflicting imports"
            ),
            None => {
                self.planned[owner_plan.index()] = Some(PlannedAssetTextureImport {
                    import: id,
                    ready: None,
                });
            }
        }
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

    /// Build unpublished GPU candidates from retained exact owners, perform
    /// a selected-live cross-storage postflight, and publish only members that
    /// remain current. Uploading cold or refreshed physical content adds one
    /// final postflight; an exact warm-cache frame retains the single-barrier
    /// path. Built-in imports therefore clone no asset owner in this post-DPE
    /// phase.
    pub(crate) fn materialize_live(
        &mut self,
        resources: &[ResourceRecord],
        owner_plans: &mut AssetTextureBatch,
        assets: &AssetServer,
        resource_manager: &mut ResourceManager,
    ) {
        self.materialize_live_impl(
            resources,
            owner_plans,
            assets,
            resource_manager,
            |_| {},
            |_| {},
        );
    }

    fn materialize_live_impl(
        &mut self,
        resources: &[ResourceRecord],
        owner_plans: &mut AssetTextureBatch,
        assets: &AssetServer,
        resource_manager: &mut ResourceManager,
        after_candidates: impl FnOnce(&ResourceManager),
        after_materialize: impl FnOnce(&ResourceManager),
    ) {
        debug_assert!(self.live_imports.is_empty());
        debug_assert!(self.live_owner_plans.is_empty());
        debug_assert!(self.live_paths.is_empty());
        debug_assert!(self.live_current.is_empty());
        debug_assert!(self.candidate_current.is_empty());
        debug_assert!(self.candidates.is_empty());

        for index in 0..self.entries.len() {
            let id = AssetTextureImportId::from_index(index);
            let entry = &self.entries[index];
            if entry.materialization.get().is_some()
                || resources[entry.node.index() as usize].first_use == usize::MAX
            {
                continue;
            }

            let Some(owner_plan) = entry.owner_plan else {
                // The generic import API is retained for isolated graph tests.
                // All built-in production imports carry an owner plan.
                let result =
                    resource_manager.materialize_asset_texture_for_graph(assets, entry.snapshot);
                assert!(entry.materialization.set(result).is_ok());
                continue;
            };
            let Some((texture, image)) = owner_plans.ready_owners(owner_plan, entry.snapshot)
            else {
                owner_plans.publish_physical_binding(owner_plan, entry.snapshot, None);
                assert!(
                    entry
                        .materialization
                        .set(AssetTextureMaterialization::unavailable())
                        .is_ok()
                );
                continue;
            };
            let path = if let Some(resident) =
                resource_manager.exact_asset_texture_residency(entry.snapshot)
            {
                LiveAssetTexturePath::Resident(resident)
            } else {
                let candidate = resource_manager.stage_asset_texture_candidate(
                    &mut self.candidates,
                    entry.snapshot,
                    texture,
                    image,
                );
                debug_assert_eq!(candidate.index(), self.candidate_current.len());
                self.candidate_current.push(false);
                LiveAssetTexturePath::Candidate(candidate)
            };
            self.live_imports.push(id);
            self.live_owner_plans.push(owner_plan);
            self.live_paths.push(path);
        }

        after_candidates(resource_manager);

        if !self.live_paths.is_empty() {
            self.live_current.extend(
                owner_plans
                    .postflight_selected(assets, &self.live_owner_plans)
                    .iter()
                    .copied(),
            );
            let mut uploaded_any = false;
            for index in 0..self.live_paths.len() {
                if !self.live_current[index] {
                    continue;
                }
                let LiveAssetTexturePath::Candidate(candidate) = self.live_paths[index] else {
                    continue;
                };
                let import = self.live_imports[index];
                let owner_plan = self.live_owner_plans[index];
                let snapshot = self.entries[import.index()].snapshot;
                let Some((_, image)) = owner_plans.ready_owners(owner_plan, snapshot) else {
                    self.live_current[index] = false;
                    continue;
                };
                uploaded_any |= resource_manager.materialize_asset_texture_candidate(
                    &mut self.candidates,
                    candidate,
                    image,
                );
                self.candidate_current[candidate.index()] = true;
            }
            after_materialize(resource_manager);
            if uploaded_any {
                let upload_current =
                    owner_plans.postflight_selected(assets, &self.live_owner_plans);
                for (current, upload_current) in
                    self.live_current.iter_mut().zip(upload_current.iter())
                {
                    *current &= *upload_current;
                }
            }
            for (live_index, &path) in self.live_paths.iter().enumerate() {
                if let LiveAssetTexturePath::Candidate(candidate) = path {
                    self.candidate_current[candidate.index()] &= self.live_current[live_index];
                }
            }
            if !self.candidates.is_empty() {
                resource_manager.publish_asset_texture_candidates(
                    &mut self.candidates,
                    &self.candidate_current,
                );
            }
            for (index, (&import, &path)) in
                self.live_imports.iter().zip(&self.live_paths).enumerate()
            {
                let result = match path {
                    LiveAssetTexturePath::Resident(expected) if self.live_current[index] => {
                        resource_manager
                            .reuse_exact_asset_texture_for_graph(
                                self.entries[import.index()].snapshot,
                                expected,
                            )
                            .unwrap_or_else(AssetTextureMaterialization::unavailable)
                    }
                    LiveAssetTexturePath::Resident(_) => AssetTextureMaterialization::unavailable(),
                    LiveAssetTexturePath::Candidate(candidate) => {
                        self.candidates.take_result(candidate)
                    }
                };
                owner_plans.publish_physical_binding(
                    self.live_owner_plans[index],
                    self.entries[import.index()].snapshot,
                    result
                        .texture
                        .as_ref()
                        .map(MaterializedAssetTexture::physical_binding),
                );
                assert!(
                    self.entries[import.index()]
                        .materialization
                        .set(result)
                        .is_ok(),
                    "Asset Texture import materialized more than once"
                );
            }
        }

        self.candidate_current.clear();
        self.live_current.clear();
        self.live_paths.clear();
        self.live_owner_plans.clear();
        self.live_imports.clear();
        self.candidates.clear();
    }

    #[cfg(test)]
    pub(crate) fn materialize_live_with_hook(
        &mut self,
        resources: &[ResourceRecord],
        owner_plans: &mut AssetTextureBatch,
        assets: &AssetServer,
        resource_manager: &mut ResourceManager,
        after_candidates: impl FnOnce(&ResourceManager),
    ) {
        self.materialize_live_impl(
            resources,
            owner_plans,
            assets,
            resource_manager,
            after_candidates,
            |_| {},
        );
    }

    #[cfg(test)]
    pub(crate) fn materialize_live_with_hooks(
        &mut self,
        resources: &[ResourceRecord],
        owner_plans: &mut AssetTextureBatch,
        assets: &AssetServer,
        resource_manager: &mut ResourceManager,
        after_candidates: impl FnOnce(&ResourceManager),
        after_materialize: impl FnOnce(&ResourceManager),
    ) {
        self.materialize_live_impl(
            resources,
            owner_plans,
            assets,
            resource_manager,
            after_candidates,
            after_materialize,
        );
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

    /// Whether this attempt registered any logical Asset Texture imports.
    ///
    /// Topology validation can skip its graph-wide derived-import scan when
    /// the canonical import table is empty. This query observes only length
    /// and is valid before or after post-DPE materialization.
    #[inline]
    #[must_use]
    pub(crate) fn has_imports(&self) -> bool {
        !self.entries.is_empty()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
            && self.lookup.is_empty()
            && self.planned.is_empty()
            && self.candidates.is_empty()
            && self.live_imports.is_empty()
            && self.live_owner_plans.is_empty()
            && self.live_paths.is_empty()
            && self.live_current.is_empty()
            && self.candidate_current.is_empty()
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
    #[cfg(test)]
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
