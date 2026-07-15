//! Graph-attempt ownership for exact geometry and object-interface snapshots.
//!
//! Draw sources retain only compact IDs into these stores. Each store owns one
//! physical/CPU owner per exact key for the whole graph attempt, preventing
//! per-draw `Arc` reference-count traffic while still closing asset ABA gaps.

use std::{num::NonZeroU32, sync::Arc};

use myth_assets::{AssetServer, AssetSnapshot, GeometryHandle};
use myth_resources::geometry::Geometry;
use myth_resources::shader_defines::ShaderDefines;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::core::{BindingInterfaceId, ResourceManager};

use super::capacity::{
    CapacityUsage, estimated_hash_map_footprint, shrink_hash_map_by_family_ratio,
    shrink_vec_by_family_ratio, vec_footprint,
};
use super::extracted::{ExtractedRenderItem, ExtractedScene};
#[cfg(test)]
use super::extracted::{ItemShaderFeatures, MaterialClassification};
use super::frame::{DrawBufferSlot, DrawPlanStore, FrameSourceId, RenderLists};

/// Compact graph-attempt identity of one exact geometry owner.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct GeometryFrameId(NonZeroU32);

impl GeometryFrameId {
    #[inline]
    pub(crate) fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("geometry frame record index exceeds u32::MAX - 1");
        Self(raw)
    }

    #[inline]
    const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

/// Compact index into the stable-deduplicated asset batch captured before
/// material topology is registered in the graph.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GeometryAssetSnapshotId(NonZeroU32);

impl GeometryAssetSnapshotId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("geometry asset snapshot index exceeds u32::MAX - 1");
        Self(raw)
    }

    #[inline]
    const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

/// Dense post-DPE index into the selected subset of visible snapshots.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GeometryLiveSnapshotId(NonZeroU32);

impl GeometryLiveSnapshotId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("live geometry snapshot index exceeds u32::MAX - 1");
        Self(raw)
    }

    #[inline]
    const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

/// Exact immutable key selected for one graph attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GeometryFrameKey {
    pub(crate) handle: GeometryHandle,
    pub(crate) asset_version: u32,
    pub(crate) asset_identity: u64,
    pub(crate) structure_version: u64,
    pub(crate) layout_version: u64,
}

impl GeometryFrameKey {
    #[inline]
    fn matches_item(self, item: &ExtractedRenderItem) -> bool {
        self.handle == item.geometry
            && (self.asset_version, self.asset_identity)
                == (item.geometry_asset_version, item.geometry_asset_identity)
    }
}

/// One coherent exact geometry snapshot owned by a graph attempt.
pub(crate) struct GeometryFrameRecord {
    pub(crate) key: GeometryFrameKey,
    pub(crate) asset: Arc<Geometry>,
    /// Published only after the exact persistent `GpuGeometry` is available.
    /// The Arc is generation-owned by that cache entry, so warm frames avoid
    /// cloning the defines' retained allocation.
    shader_defines: Option<Arc<ShaderDefines>>,
    pub(crate) topology: wgpu::PrimitiveTopology,
    physical: Option<GeometryFramePhysical>,
}

impl GeometryFrameRecord {
    #[inline]
    pub(crate) fn shader_defines(&self) -> &ShaderDefines {
        self.shader_defines
            .as_deref()
            .expect("prepared Geometry lost its DPE-live shader snapshot")
    }
}

/// Physical geometry interface interned once for one exact generation.
pub(crate) struct GeometryFramePhysical {
    pub(crate) vertex_buffers: SmallVec<[DrawBufferSlot; 2]>,
    pub(crate) index_buffer: Option<(DrawBufferSlot, wgpu::IndexFormat, u32)>,
    pub(crate) draw_range: std::ops::Range<u32>,
    pub(crate) instance_range: std::ops::Range<u32>,
    pub(crate) layout_info: Arc<crate::pipeline::vertex::GeneratedVertexLayout>,
    pub(crate) layout_id: u64,
}

/// Sparse post-DPE join from the sorted live-source union to its exact
/// geometry owner. Keeping only live joins avoids touching the full
/// `FrameSourceId` high-water domain on sparse frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceGeometryJoin {
    source: FrameSourceId,
    geometry: GeometryFrameId,
}

/// Compact source join and unique geometry-owner table for one graph attempt.
pub(crate) struct GeometryFrameStore {
    /// Stable first-occurrence handle order for bounded AssetStorage capture.
    asset_handles: Vec<GeometryHandle>,
    asset_handle_lookup: FxHashMap<GeometryHandle, GeometryAssetSnapshotId>,
    /// Exact CPU owners captured before material texture topology is emitted.
    /// The batch is per-entry coherent, then revalidated at each publication
    /// boundary; it is intentionally not presented as one atomic storage view.
    asset_snapshots: Vec<AssetSnapshot<GeometryHandle, Geometry>>,
    asset_current: Vec<bool>,
    /// Stable-deduplicated post-DPE selection. Values index asset_snapshots;
    /// revalidation therefore touches only unique live owners.
    live_snapshot_indices: Vec<usize>,
    live_snapshot_lookup: FxHashMap<GeometryAssetSnapshotId, GeometryLiveSnapshotId>,
    live_current: Vec<bool>,
    records: Vec<GeometryFrameRecord>,
    /// Parallel joins between the post-DPE record and its exact pregraph owner.
    record_snapshots: Vec<GeometryLiveSnapshotId>,
    snapshot_records: Vec<Option<GeometryFrameId>>,
    /// Memoizes an incoherent interior-epoch capture without allocating a
    /// failure hash entry or retrying it for every source sharing the owner.
    failed_snapshots: Vec<bool>,
    /// Sorted dense join for only the DPE-live source union. `Vec::clear` is
    /// O(1) for this Copy record, so warm attempts neither zero nor scan the
    /// potentially much larger frame-source domain.
    source_geometry_joins: Vec<SourceGeometryJoin>,
    /// Recycled spill allocations for physical geometries with more vertex
    /// streams than the inline representation. A graph attempt owns each
    /// active list through `GeometryFramePhysical`; `clear` returns the list
    /// here instead of freeing and reallocating it every frame.
    vertex_buffer_slot_scratch: Vec<SmallVec<[DrawBufferSlot; 2]>>,
    initialized: bool,
    live_captured: bool,
}

const GEOMETRY_FRAME_RECORD_BASELINE: usize = 256;
const GEOMETRY_FRAME_SOURCE_BASELINE: usize = 512;
const GEOMETRY_VERTEX_BUFFER_SCRATCH_BASELINE: usize = 16;

impl GeometryFrameStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            asset_handles: Vec::with_capacity(GEOMETRY_FRAME_RECORD_BASELINE),
            asset_handle_lookup: FxHashMap::default(),
            asset_snapshots: Vec::with_capacity(GEOMETRY_FRAME_RECORD_BASELINE),
            asset_current: Vec::with_capacity(GEOMETRY_FRAME_RECORD_BASELINE),
            live_snapshot_indices: Vec::with_capacity(GEOMETRY_FRAME_RECORD_BASELINE),
            live_snapshot_lookup: FxHashMap::default(),
            live_current: Vec::with_capacity(GEOMETRY_FRAME_RECORD_BASELINE),
            records: Vec::with_capacity(GEOMETRY_FRAME_RECORD_BASELINE),
            record_snapshots: Vec::with_capacity(GEOMETRY_FRAME_RECORD_BASELINE),
            snapshot_records: Vec::with_capacity(GEOMETRY_FRAME_RECORD_BASELINE),
            failed_snapshots: Vec::with_capacity(GEOMETRY_FRAME_RECORD_BASELINE),
            source_geometry_joins: Vec::with_capacity(GEOMETRY_FRAME_SOURCE_BASELINE),
            vertex_buffer_slot_scratch: Vec::with_capacity(GEOMETRY_VERTEX_BUFFER_SCRATCH_BASELINE),
            initialized: false,
            live_captured: false,
        }
    }

    /// Capture one exact CPU owner for every stable-deduplicated geometry
    /// handle in the CPU-visible source domain.
    ///
    /// This runs before material topology is registered so a stale geometry
    /// cannot cause texture producers or material work to enter the graph.
    /// The AssetStorage read lock is bounded to the storage batch width and is
    /// never held while deriving geometry state.
    pub(crate) fn capture_visible(
        &mut self,
        assets: &AssetServer,
        extracted_scene: &ExtractedScene,
        render_lists: &RenderLists,
    ) {
        assert!(
            !self.initialized,
            "geometry frame store may be captured only once per graph attempt"
        );
        debug_assert!(self.asset_handles.is_empty());
        debug_assert!(self.asset_handle_lookup.is_empty());
        debug_assert!(self.asset_snapshots.is_empty());
        debug_assert!(self.asset_current.is_empty());
        debug_assert!(self.live_snapshot_indices.is_empty());
        debug_assert!(self.live_snapshot_lookup.is_empty());
        debug_assert!(self.live_current.is_empty());
        debug_assert!(self.records.is_empty());
        debug_assert!(self.record_snapshots.is_empty());
        debug_assert!(self.snapshot_records.is_empty());
        debug_assert!(self.failed_snapshots.is_empty());
        debug_assert!(self.source_geometry_joins.is_empty());

        for source in render_lists.frame_source_ids() {
            let Some(extracted_source) = render_lists.extracted_source(source) else {
                continue;
            };
            let Some(item) = extracted_scene.render_items.get(extracted_source.index()) else {
                continue;
            };
            if let std::collections::hash_map::Entry::Vacant(entry) =
                self.asset_handle_lookup.entry(item.geometry)
            {
                let id = GeometryAssetSnapshotId::from_index(self.asset_handles.len());
                self.asset_handles.push(item.geometry);
                entry.insert(id);
            }
        }

        assets
            .geometries
            .snapshot_batched(&self.asset_handles, &mut self.asset_snapshots);
        assets
            .geometries
            .revalidate_batched(&self.asset_snapshots, &mut self.asset_current);
        debug_assert_eq!(self.asset_handles.len(), self.asset_snapshots.len());
        debug_assert_eq!(self.asset_snapshots.len(), self.asset_current.len());

        self.initialized = true;
    }

    #[must_use]
    pub(crate) const fn eligibility<'a>(
        &'a self,
        extracted_scene: &'a ExtractedScene,
        render_lists: &'a RenderLists,
    ) -> GeometryEligibilityResolver<'a> {
        GeometryEligibilityResolver {
            store: self,
            extracted_scene,
            render_lists,
        }
    }

    /// Derive interior geometry state only for the post-DPE source union.
    /// Post-DPE AssetStorage traffic and retained record owners are O(unique
    /// live geometries), never O(visible or draw sources); all storage critical
    /// sections use the fixed-width batch API.
    pub(crate) fn capture_live(
        &mut self,
        assets: &AssetServer,
        extracted_scene: &ExtractedScene,
        render_lists: &RenderLists,
        live_sources: &[FrameSourceId],
    ) {
        assert!(
            self.initialized,
            "visible geometry owners were not captured before graph topology"
        );
        assert!(
            !self.live_captured,
            "live geometry state may be captured only once per graph attempt"
        );
        debug_assert!(self.records.is_empty());
        debug_assert!(self.live_snapshot_indices.is_empty());
        debug_assert!(self.live_snapshot_lookup.is_empty());
        debug_assert!(self.live_current.is_empty());
        debug_assert!(self.record_snapshots.is_empty());
        debug_assert!(self.snapshot_records.is_empty());
        debug_assert!(self.failed_snapshots.is_empty());
        debug_assert!(self.source_geometry_joins.is_empty());

        debug_assert!(
            live_sources.windows(2).all(|pair| pair[0] < pair[1]),
            "the DPE-live frame-source union must be sorted and deduplicated"
        );

        // Project the source union into one compact, stable first-occurrence
        // list before touching AssetStorage again. The all-visible owner table
        // remains available for topology identity, but every post-DPE scan is
        // bounded by unique live Geometry count.
        for &source in live_sources {
            let Some(extracted_source) = render_lists.extracted_source(source) else {
                continue;
            };
            let Some(item) = extracted_scene.render_items.get(extracted_source.index()) else {
                continue;
            };
            let Some(snapshot_id) = self.snapshot_id_for_item(item) else {
                continue;
            };
            if let std::collections::hash_map::Entry::Vacant(entry) =
                self.live_snapshot_lookup.entry(snapshot_id)
            {
                let live_id = GeometryLiveSnapshotId::from_index(self.live_snapshot_indices.len());
                self.live_snapshot_indices.push(snapshot_id.index());
                entry.insert(live_id);
            }
        }
        assets.geometries.revalidate_selected_batched(
            &self.asset_snapshots,
            &self.live_snapshot_indices,
            &mut self.live_current,
        );
        self.snapshot_records
            .resize(self.live_snapshot_indices.len(), None);
        self.failed_snapshots
            .resize(self.live_snapshot_indices.len(), false);

        for &source in live_sources {
            let Some(extracted_source) = render_lists.extracted_source(source) else {
                continue;
            };
            let Some(item) = extracted_scene.render_items.get(extracted_source.index()) else {
                continue;
            };
            let Some(snapshot_id) = self.snapshot_id_for_item(item) else {
                continue;
            };
            let Some(&live_id) = self.live_snapshot_lookup.get(&snapshot_id) else {
                continue;
            };
            if self.live_current.get(live_id.index()).copied() != Some(true)
                || self.snapshot_records[live_id.index()].is_some()
                || self.failed_snapshots[live_id.index()]
            {
                continue;
            }
            let Some(loaded) = self.asset_snapshots[snapshot_id.index()].loaded() else {
                self.failed_snapshots[live_id.index()] = true;
                continue;
            };
            let asset = loaded.asset();

            let structure_version = asset.structure_version();
            let layout_version = asset.layout_version();
            let topology = asset.topology;
            // Keep the CPU interface snapshot coherent with its exact owner.
            if (structure_version, layout_version)
                != (asset.structure_version(), asset.layout_version())
            {
                self.failed_snapshots[live_id.index()] = true;
                continue;
            }

            let key = GeometryFrameKey {
                handle: item.geometry,
                asset_version: loaded.version(),
                asset_identity: loaded.identity(),
                structure_version,
                layout_version,
            };
            let id = GeometryFrameId::from_index(self.records.len());
            self.records.push(GeometryFrameRecord {
                key,
                asset: Arc::clone(asset),
                shader_defines: None,
                topology,
                physical: None,
            });
            self.record_snapshots.push(live_id);
            self.snapshot_records[live_id.index()] = Some(id);
        }

        // Revalidate after CPU derivation, then publish only coherent joins.
        assets.geometries.revalidate_selected_batched(
            &self.asset_snapshots,
            &self.live_snapshot_indices,
            &mut self.live_current,
        );
        for &source in live_sources {
            let Some(extracted_source) = render_lists.extracted_source(source) else {
                continue;
            };
            let Some(item) = extracted_scene.render_items.get(extracted_source.index()) else {
                continue;
            };
            let Some(snapshot_id) = self.snapshot_id_for_item(item) else {
                continue;
            };
            let Some(&live_id) = self.live_snapshot_lookup.get(&snapshot_id) else {
                continue;
            };
            let Some(id) = self.snapshot_records[live_id.index()] else {
                continue;
            };
            if self.record_is_current_from_batch(id) {
                self.source_geometry_joins.push(SourceGeometryJoin {
                    source,
                    geometry: id,
                });
            }
        }
        self.live_captured = true;
    }

    /// Materialize and intern each unique DPE-live geometry exactly once.
    /// Per-instance object preparation can then borrow the retained asset and
    /// address only compact physical slots.
    pub(crate) fn prepare_live(
        &mut self,
        assets: &AssetServer,
        resource_manager: &mut ResourceManager,
        draw_plans: &mut DrawPlanStore,
    ) {
        assert!(self.live_captured, "live geometry state was not captured");
        assets.geometries.revalidate_selected_batched(
            &self.asset_snapshots,
            &self.live_snapshot_indices,
            &mut self.live_current,
        );
        for index in 0..self.records.len() {
            let id = GeometryFrameId::from_index(index);
            if !self.record_is_current_from_batch(id) {
                continue;
            }
            let record = &self.records[index];
            if !resource_manager.prepare_geometry_exact(
                record.key.handle,
                &record.asset,
                record.key.asset_version,
                record.key.asset_identity,
            ) {
                continue;
            }
            let Some(gpu) = resource_manager.get_geometry(record.key.handle) else {
                continue;
            };
            if (gpu.asset_version, gpu.asset_identity)
                != (record.key.asset_version, record.key.asset_identity)
                || gpu.version != record.key.structure_version
                || record.asset.layout_version() != record.key.layout_version
            {
                continue;
            }
            assert_eq!(gpu.vertex_buffers.len(), gpu.vertex_buffer_ids.len());
            let mut vertex_buffers = self.vertex_buffer_slot_scratch.pop().unwrap_or_default();
            vertex_buffers.clear();
            vertex_buffers.reserve(gpu.vertex_buffers.len());
            for (buffer, &physical_id) in gpu.vertex_buffers.iter().zip(&gpu.vertex_buffer_ids) {
                vertex_buffers.push(draw_plans.intern_buffer(buffer, physical_id));
            }
            let index_buffer =
                gpu.index_buffer
                    .as_ref()
                    .map(|(buffer, format, count, physical_id)| {
                        (
                            draw_plans.intern_buffer(buffer, *physical_id),
                            *format,
                            *count,
                        )
                    });
            let shader_defines = Arc::clone(&gpu.shader_defines);
            self.records[index].physical = Some(GeometryFramePhysical {
                vertex_buffers,
                index_buffer,
                draw_range: gpu.draw_range.clone(),
                instance_range: gpu.instance_range.clone(),
                layout_info: gpu.layout_info.clone(),
                layout_id: gpu.layout_id,
            });
            self.records[index].shader_defines = Some(shader_defines);
        }

        // A writer may have replaced an owner while physical preparation ran.
        // Close that window in one bounded pass and retract stale publication.
        assets.geometries.revalidate_selected_batched(
            &self.asset_snapshots,
            &self.live_snapshot_indices,
            &mut self.live_current,
        );
        for index in 0..self.records.len() {
            let id = GeometryFrameId::from_index(index);
            if !self.record_is_current_from_batch(id) {
                if let Some(mut physical) = self.records[index].physical.take() {
                    physical.vertex_buffers.clear();
                    self.vertex_buffer_slot_scratch
                        .push(physical.vertex_buffers);
                }
                self.records[index].shader_defines = None;
            }
        }
    }

    #[must_use]
    pub(crate) const fn resolver(&self) -> GeometryFrameResolver<'_> {
        GeometryFrameResolver { store: self }
    }

    /// Validate both the exact AssetStorage owner and its pipeline-relevant
    /// interior epochs. This is the fail-closed boundary before publication.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_current(&self, assets: &AssetServer, id: GeometryFrameId) -> bool {
        let Some(record) = self.records.get(id.index()) else {
            return false;
        };
        assets
            .geometries
            .get_entry_with_identity(record.key.handle)
            .is_some_and(|(current, version, identity)| {
                (version, identity) == (record.key.asset_version, record.key.asset_identity)
                    && Arc::ptr_eq(&current, &record.asset)
                    && record.asset.structure_version() == record.key.structure_version
                    && record.asset.layout_version() == record.key.layout_version
            })
    }

    #[inline]
    fn snapshot_matches_item(
        &self,
        id: GeometryAssetSnapshotId,
        item: &ExtractedRenderItem,
    ) -> bool {
        self.asset_current.get(id.index()).copied() == Some(true)
            && self
                .asset_snapshots
                .get(id.index())
                .is_some_and(|snapshot| {
                    snapshot.handle() == item.geometry
                        && snapshot.loaded().is_some_and(|loaded| {
                            (loaded.version(), loaded.identity())
                                == (item.geometry_asset_version, item.geometry_asset_identity)
                        })
                })
    }

    #[inline]
    fn snapshot_id_for_item(&self, item: &ExtractedRenderItem) -> Option<GeometryAssetSnapshotId> {
        let &id = self.asset_handle_lookup.get(&item.geometry)?;
        self.snapshot_matches_item(id, item).then_some(id)
    }

    #[inline]
    fn record_is_current_from_batch(&self, id: GeometryFrameId) -> bool {
        let Some(record) = self.records.get(id.index()) else {
            return false;
        };
        let Some(&live_id) = self.record_snapshots.get(id.index()) else {
            return false;
        };
        let Some(&snapshot_index) = self.live_snapshot_indices.get(live_id.index()) else {
            return false;
        };
        self.live_current.get(live_id.index()).copied() == Some(true)
            && self
                .asset_snapshots
                .get(snapshot_index)
                .and_then(AssetSnapshot::loaded)
                .is_some_and(|loaded| {
                    loaded.version() == record.key.asset_version
                        && loaded.identity() == record.key.asset_identity
                        && Arc::ptr_eq(loaded.asset(), &record.asset)
                        && record.asset.structure_version() == record.key.structure_version
                        && record.asset.layout_version() == record.key.layout_version
                })
    }

    /// Cheap owner-local double-check used around deferred pipeline derivation.
    #[must_use]
    pub(crate) fn pipeline_state_is_current(&self, id: GeometryFrameId) -> bool {
        self.records.get(id.index()).is_some_and(|record| {
            record.asset.structure_version() == record.key.structure_version
                && record.asset.layout_version() == record.key.layout_version
        })
    }

    /// Release every graph-attempt owner while retaining allocations.
    pub(crate) fn clear(&mut self) {
        self.source_geometry_joins.clear();
        // Reverse return order pairs the next attempt's stable record order
        // with LIFO pops, preserving each exact geometry's prior stream-list
        // capacity without an auxiliary lookup.
        for record in self.records.iter_mut().rev() {
            if let Some(mut physical) = record.physical.take() {
                physical.vertex_buffers.clear();
                self.vertex_buffer_slot_scratch
                    .push(physical.vertex_buffers);
            }
        }
        self.snapshot_records.clear();
        self.failed_snapshots.clear();
        self.record_snapshots.clear();
        self.records.clear();
        self.live_current.clear();
        self.live_snapshot_lookup.clear();
        self.live_snapshot_indices.clear();
        self.asset_current.clear();
        self.asset_snapshots.clear();
        self.asset_handle_lookup.clear();
        self.asset_handles.clear();
        self.initialized = false;
        self.live_captured = false;
    }

    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.asset_handles, GEOMETRY_FRAME_RECORD_BASELINE)
            + estimated_hash_map_footprint(
                &self.asset_handle_lookup,
                GEOMETRY_FRAME_RECORD_BASELINE,
            )
            + vec_footprint(&self.asset_snapshots, GEOMETRY_FRAME_RECORD_BASELINE)
            + vec_footprint(&self.asset_current, GEOMETRY_FRAME_RECORD_BASELINE)
            + vec_footprint(&self.live_snapshot_indices, GEOMETRY_FRAME_RECORD_BASELINE)
            + estimated_hash_map_footprint(
                &self.live_snapshot_lookup,
                GEOMETRY_FRAME_RECORD_BASELINE,
            )
            + vec_footprint(&self.live_current, GEOMETRY_FRAME_RECORD_BASELINE)
            + vec_footprint(&self.records, GEOMETRY_FRAME_RECORD_BASELINE)
            + vec_footprint(&self.record_snapshots, GEOMETRY_FRAME_RECORD_BASELINE)
            + vec_footprint(&self.snapshot_records, GEOMETRY_FRAME_RECORD_BASELINE)
            + vec_footprint(&self.failed_snapshots, GEOMETRY_FRAME_RECORD_BASELINE)
            + vec_footprint(&self.source_geometry_joins, GEOMETRY_FRAME_SOURCE_BASELINE)
            + self.vertex_buffer_slot_scratch_usage()
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        shrink_vec_by_family_ratio(
            &mut self.asset_handles,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.asset_handle_lookup,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.asset_snapshots,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.asset_current,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.live_snapshot_indices,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.live_snapshot_lookup,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.live_current,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.records,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.record_snapshots,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.snapshot_records,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.failed_snapshots,
            GEOMETRY_FRAME_RECORD_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.source_geometry_joins,
            GEOMETRY_FRAME_SOURCE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        if target_bytes < retained_bytes {
            // Scratch entries are empty. Dropping them releases both their
            // optional spill allocations and lets the outer vector follow the
            // normal family trim policy.
            self.vertex_buffer_slot_scratch.clear();
        }
        shrink_vec_by_family_ratio(
            &mut self.vertex_buffer_slot_scratch,
            GEOMETRY_VERTEX_BUFFER_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn geometry_count(&self) -> usize {
        self.records.len()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn joined_source_count(&self) -> usize {
        self.source_geometry_joins.len()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.asset_handles.is_empty()
            && self.asset_handle_lookup.is_empty()
            && self.asset_snapshots.is_empty()
            && self.asset_current.is_empty()
            && self.live_snapshot_indices.is_empty()
            && self.live_snapshot_lookup.is_empty()
            && self.live_current.is_empty()
            && self.records.is_empty()
            && self.record_snapshots.is_empty()
            && self.snapshot_records.is_empty()
            && self.failed_snapshots.is_empty()
            && self.source_geometry_joins.is_empty()
            && !self.initialized
            && !self.live_captured
    }

    fn vertex_buffer_slot_scratch_usage(&self) -> CapacityUsage {
        let slot_bytes = std::mem::size_of::<DrawBufferSlot>();
        let outer_slot_bytes = std::mem::size_of::<SmallVec<[DrawBufferSlot; 2]>>();
        let outer_retained = self
            .vertex_buffer_slot_scratch
            .capacity()
            .saturating_mul(outer_slot_bytes);
        let outer_baseline =
            GEOMETRY_VERTEX_BUFFER_SCRATCH_BASELINE.saturating_mul(outer_slot_bytes);

        let mut nested_active = 0usize;
        let mut nested_retained = 0usize;
        for slots in self
            .records
            .iter()
            .filter_map(|record| record.physical.as_ref())
            .map(|physical| &physical.vertex_buffers)
        {
            if slots.spilled() {
                nested_active =
                    nested_active.saturating_add(slots.len().saturating_mul(slot_bytes));
                nested_retained =
                    nested_retained.saturating_add(slots.capacity().saturating_mul(slot_bytes));
            }
        }
        for slots in &self.vertex_buffer_slot_scratch {
            if slots.spilled() {
                nested_retained =
                    nested_retained.saturating_add(slots.capacity().saturating_mul(slot_bytes));
            }
        }

        CapacityUsage::new(
            nested_active,
            outer_retained.saturating_add(nested_retained),
            outer_baseline,
        )
    }
}

impl Default for GeometryFrameStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Read-only graph-build gate for material and texture topology.
#[derive(Clone, Copy)]
pub(crate) struct GeometryEligibilityResolver<'a> {
    store: &'a GeometryFrameStore,
    extracted_scene: &'a ExtractedScene,
    render_lists: &'a RenderLists,
}

impl GeometryEligibilityResolver<'_> {
    /// Return true only when this source still names the exact pregraph owner
    /// captured for its extracted geometry stamp.
    #[must_use]
    pub(crate) fn source_is_eligible(
        self,
        source: FrameSourceId,
        item: &ExtractedRenderItem,
    ) -> bool {
        debug_assert!(
            self.render_lists
                .extracted_source(source)
                .and_then(|source| self.extracted_scene.render_items.get(source.index()))
                .is_some_and(|resolved| std::ptr::eq(resolved, item)),
            "geometry eligibility received an item from a different frame source"
        );
        self.store.snapshot_id_for_item(item).is_some()
    }
}

/// Read-only source/record resolver handed to deferred preparation.
#[derive(Clone, Copy)]
pub(crate) struct GeometryFrameResolver<'a> {
    store: &'a GeometryFrameStore,
}

/// Stack-local cursor for linearly merging the sorted DPE-live source stream
/// with [`GeometryFrameStore`]'s compact source join.
#[derive(Debug, Default)]
pub(crate) struct GeometrySourceJoinCursor {
    next_join: usize,
    previous_source: Option<FrameSourceId>,
}

impl<'a> GeometryFrameResolver<'a> {
    /// Create a fresh cursor for one monotonic traversal of the sorted live
    /// source union. Tests retain a separate random-access resolver for
    /// validating compact-join semantics.
    #[must_use]
    pub(crate) fn monotonic_cursor(self) -> GeometrySourceJoinCursor {
        debug_assert!(
            self.store.live_captured,
            "Geometry source resolution requires a captured live-source union"
        );
        GeometrySourceJoinCursor {
            next_join: 0,
            previous_source: None,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn geometry_id(
        self,
        source: FrameSourceId,
        item: &ExtractedRenderItem,
    ) -> Option<GeometryFrameId> {
        let join_index = self
            .store
            .source_geometry_joins
            .binary_search_by_key(&source, |join| join.source)
            .ok()?;
        let id = self.store.source_geometry_joins[join_index].geometry;
        self.store
            .records
            .get(id.index())?
            .key
            .matches_item(item)
            .then_some(id)
    }

    /// Resolve the next sorted source in amortized O(1) time.
    ///
    /// Both the DPE-live source union and the compact join are strictly sorted
    /// by `FrameSourceId`, so Gather can advance one cursor rather than doing
    /// an O(log live) binary search per source. Missing/incoherent joins are
    /// skipped without consuming a later source's entry.
    #[must_use]
    pub(crate) fn geometry_id_monotonic(
        self,
        cursor: &mut GeometrySourceJoinCursor,
        source: FrameSourceId,
        item: &ExtractedRenderItem,
    ) -> Option<GeometryFrameId> {
        debug_assert!(
            cursor
                .previous_source
                .is_none_or(|previous| previous < source),
            "monotonic Geometry source resolution requires strictly increasing sources"
        );
        cursor.previous_source = Some(source);

        while self
            .store
            .source_geometry_joins
            .get(cursor.next_join)
            .is_some_and(|join| join.source < source)
        {
            cursor.next_join += 1;
        }
        let join = self.store.source_geometry_joins.get(cursor.next_join)?;
        if join.source != source {
            return None;
        }
        cursor.next_join += 1;
        self.store
            .records
            .get(join.geometry.index())?
            .key
            .matches_item(item)
            .then_some(join.geometry)
    }

    #[must_use]
    pub(crate) fn geometry(self, id: GeometryFrameId) -> Option<&'a GeometryFrameRecord> {
        self.store.records.get(id.index())
    }

    #[must_use]
    pub(crate) fn physical(self, id: GeometryFrameId) -> Option<&'a GeometryFramePhysical> {
        self.store.records.get(id.index())?.physical.as_ref()
    }

    #[must_use]
    pub(crate) fn pipeline_state_is_current(self, id: GeometryFrameId) -> bool {
        self.store.pipeline_state_is_current(id)
    }
}

/// Compact graph-attempt identity of one object binding interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ObjectInterfaceId(u32);

impl ObjectInterfaceId {
    #[inline]
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("object interface index exceeds u32::MAX"))
    }

    #[inline]
    const fn index(self) -> usize {
        self.0 as usize
    }
}

/// One exact immutable object binding interface.
pub(crate) struct ObjectInterfaceRecord {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) layout_id: u64,
    pub(crate) binding_wgsl: Arc<str>,
    pub(crate) binding_interface_id: BindingInterfaceId,
}

/// Unique object binding interfaces retained by one graph attempt.
pub(crate) struct ObjectInterfaceStore {
    records: Vec<ObjectInterfaceRecord>,
    lookup: FxHashMap<BindingInterfaceId, ObjectInterfaceId>,
}

const OBJECT_INTERFACE_BASELINE: usize = 64;

impl ObjectInterfaceStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            records: Vec::with_capacity(OBJECT_INTERFACE_BASELINE),
            lookup: FxHashMap::default(),
        }
    }

    /// Intern by the canonical physical-layout ID and WGSL content hash.
    /// WGPU/Arc owners are cloned only for a newly observed interface.
    pub(crate) fn intern(
        &mut self,
        binding_interface_id: BindingInterfaceId,
        layout_id: u64,
        layout: &wgpu::BindGroupLayout,
        binding_wgsl: &Arc<str>,
    ) -> ObjectInterfaceId {
        if let Some(&id) = self.lookup.get(&binding_interface_id) {
            debug_assert_eq!(self.records[id.index()].layout_id, layout_id);
            debug_assert_eq!(
                self.records[id.index()].binding_wgsl.as_ref(),
                binding_wgsl.as_ref(),
                "canonical binding-interface ID must preserve exact WGSL identity"
            );
            return id;
        }

        let id = ObjectInterfaceId::from_index(self.records.len());
        self.records.push(ObjectInterfaceRecord {
            layout: layout.clone(),
            layout_id,
            binding_wgsl: binding_wgsl.clone(),
            binding_interface_id,
        });
        assert!(self.lookup.insert(binding_interface_id, id).is_none());
        id
    }

    #[must_use]
    pub(crate) const fn resolver(&self) -> ObjectInterfaceResolver<'_> {
        ObjectInterfaceResolver { store: self }
    }

    pub(crate) fn clear(&mut self) {
        self.lookup.clear();
        self.records.clear();
    }

    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.records, OBJECT_INTERFACE_BASELINE)
            + estimated_hash_map_footprint(&self.lookup, OBJECT_INTERFACE_BASELINE)
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        shrink_vec_by_family_ratio(
            &mut self.records,
            OBJECT_INTERFACE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.lookup,
            OBJECT_INTERFACE_BASELINE,
            target_bytes,
            retained_bytes,
        );
    }
}

impl Default for ObjectInterfaceStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Read-only object-interface resolver handed to pipeline derivation.
#[derive(Clone, Copy)]
pub(crate) struct ObjectInterfaceResolver<'a> {
    store: &'a ObjectInterfaceStore,
}

impl<'a> ObjectInterfaceResolver<'a> {
    #[must_use]
    pub(crate) fn interface(self, id: ObjectInterfaceId) -> Option<&'a ObjectInterfaceRecord> {
        self.store.records.get(id.index())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{RendererInitConfig, RendererSettings};
    use crate::test_support::SharedTestContext as WgpuContext;
    use glam::Mat4;
    use myth_assets::{GeometryHandle, MaterialHandle};
    use myth_resources::{BoundingBox, geometry::Geometry};
    use slotmap::Key;

    use crate::graph::frame::ExtractedSourceId;

    fn headless_resource_manager() -> ResourceManager {
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

    fn extracted_item(
        geometry: GeometryHandle,
        asset_version: u32,
        asset_identity: u64,
    ) -> ExtractedRenderItem {
        ExtractedRenderItem {
            node_handle: myth_scene::NodeHandle::null(),
            world_matrix: Mat4::IDENTITY,
            prev_world_matrix: Mat4::IDENTITY,
            geometry,
            material: MaterialHandle::null(),
            geometry_asset_version: asset_version,
            geometry_asset_identity: asset_identity,
            material_asset_version: 0,
            material_asset_identity: 0,
            material_version: 0,
            material_index: 0,
            material_classification: MaterialClassification::empty(),
            skeleton: None,
            item_variant_flags: 0,
            item_shader_features: ItemShaderFeatures::empty(),
            cast_shadows: true,
            receive_shadows: true,
            world_aabb: BoundingBox::default(),
        }
    }

    #[test]
    fn duplicate_sources_share_one_exact_geometry_owner() {
        let assets = AssetServer::new();
        let handle = assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
        let (_, asset_version, asset_identity) = assets
            .geometries
            .get_entry_with_identity(handle)
            .expect("geometry missing");
        let mut extracted = ExtractedScene::new();
        extracted
            .render_items
            .push(extracted_item(handle, asset_version, asset_identity));
        extracted
            .render_items
            .push(extracted_item(handle, asset_version, asset_identity));

        let mut lists = RenderLists::new();
        let first = lists.intern_frame_source(ExtractedSourceId::from_index(0));
        let second = lists.intern_frame_source(ExtractedSourceId::from_index(1));
        let mut store = GeometryFrameStore::new();
        store.capture_visible(&assets, &extracted, &lists);
        store.capture_live(&assets, &extracted, &lists, &[first, second]);

        let resolver = store.resolver();
        let first_id = resolver
            .geometry_id(first, &extracted.render_items[0])
            .expect("first geometry join missing");
        let second_id = resolver
            .geometry_id(second, &extracted.render_items[1])
            .expect("second geometry join missing");
        assert_eq!(first_id, second_id);
        let mut cursor = resolver.monotonic_cursor();
        assert_eq!(
            resolver.geometry_id_monotonic(&mut cursor, first, &extracted.render_items[0]),
            Some(first_id)
        );
        assert_eq!(
            resolver.geometry_id_monotonic(&mut cursor, second, &extracted.render_items[1]),
            Some(second_id)
        );
        assert_eq!(store.geometry_count(), 1);
        assert_eq!(store.joined_source_count(), 2);
        assert_eq!(
            Arc::strong_count(&resolver.geometry(first_id).unwrap().asset),
            3,
            "owner count must remain storage + one unique snapshot + one live record, independent of draw count"
        );
        assert!(store.is_current(&assets, first_id));
        assert!(store.pipeline_state_is_current(first_id));

        assets.geometries.update(handle, Geometry::new());
        assert!(!store.is_current(&assets, first_id));
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn shader_snapshot_is_deferred_until_live_prepare_and_reused_from_exact_gpu_geometry() {
        let assets = AssetServer::new();
        let handle = assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
        let (_, asset_version, asset_identity) = assets
            .geometries
            .get_entry_with_identity(handle)
            .expect("geometry missing");

        let capture_attempt = || {
            let mut extracted = ExtractedScene::new();
            extracted
                .render_items
                .push(extracted_item(handle, asset_version, asset_identity));
            let mut lists = RenderLists::new();
            let source = lists.intern_frame_source(ExtractedSourceId::from_index(0));
            let mut store = GeometryFrameStore::new();
            store.capture_visible(&assets, &extracted, &lists);
            store.capture_live(&assets, &extracted, &lists, &[source]);
            (extracted, lists, source, store)
        };

        let (extracted, _lists, source, mut store) = capture_attempt();
        let id = store
            .resolver()
            .geometry_id(source, &extracted.render_items[0])
            .expect("live Geometry join missing");
        assert!(
            store.records[id.index()].shader_defines.is_none(),
            "post-DPE CPU capture must not clone or publish Geometry shader state"
        );

        let mut resource_manager = headless_resource_manager();
        assert!(
            resource_manager.get_geometry(handle).is_none(),
            "capturing a live Geometry must remain CPU-only"
        );
        let mut draw_plans = DrawPlanStore::new();
        store.prepare_live(&assets, &mut resource_manager, &mut draw_plans);

        let frame_snapshot = store.records[id.index()]
            .shader_defines
            .as_ref()
            .expect("the first live prepare did not publish shader state");
        let gpu_snapshot = &resource_manager
            .get_geometry(handle)
            .expect("the first live prepare did not publish a GPU Geometry")
            .shader_defines;
        assert!(
            Arc::ptr_eq(frame_snapshot, gpu_snapshot),
            "the frame record must borrow the exact persistent Geometry snapshot"
        );
        assert_eq!(
            store
                .resolver()
                .geometry(id)
                .unwrap()
                .shader_defines()
                .compute_hash(),
            gpu_snapshot.compute_hash()
        );
        let persistent_snapshot = Arc::clone(gpu_snapshot);

        let (next_extracted, _next_lists, next_source, mut next_store) = capture_attempt();
        let next_id = next_store
            .resolver()
            .geometry_id(next_source, &next_extracted.render_items[0])
            .expect("warm Geometry join missing");
        assert!(
            next_store.records[next_id.index()].shader_defines.is_none(),
            "a warm graph attempt must still defer shader-state publication"
        );
        let mut next_draw_plans = DrawPlanStore::new();
        next_store.prepare_live(&assets, &mut resource_manager, &mut next_draw_plans);
        let next_snapshot = next_store.records[next_id.index()]
            .shader_defines
            .as_ref()
            .expect("the exact warm Geometry lost its shader snapshot");
        assert!(
            Arc::ptr_eq(next_snapshot, &persistent_snapshot),
            "an exact persistent GPU Geometry must reuse its canonical snapshot Arc"
        );
    }

    #[test]
    fn shared_geometry_unique_tables_never_reserve_from_draw_count() {
        const SOURCE_COUNT: usize = 4096;
        let assets = AssetServer::new();
        let handle = assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
        let (_, asset_version, asset_identity) = assets
            .geometries
            .get_entry_with_identity(handle)
            .expect("geometry missing");
        let template = extracted_item(handle, asset_version, asset_identity);
        let mut extracted = ExtractedScene::new();
        let mut lists = RenderLists::new();
        let mut live = Vec::with_capacity(SOURCE_COUNT);
        for index in 0..SOURCE_COUNT {
            extracted.render_items.push(template.clone());
            live.push(lists.intern_frame_source(ExtractedSourceId::from_index(index)));
        }

        let mut store = GeometryFrameStore::new();
        store.capture_visible(&assets, &extracted, &lists);
        store.capture_live(&assets, &extracted, &lists, &live);

        assert_eq!(store.records.len(), 1);
        assert_eq!(store.asset_handles.len(), 1);
        assert_eq!(store.asset_snapshots.len(), 1);
        assert_eq!(store.live_snapshot_indices.len(), 1);
        assert_eq!(store.live_snapshot_lookup.len(), 1);
        assert_eq!(store.live_current.len(), 1);
        assert_eq!(store.record_snapshots.len(), 1);
        assert_eq!(store.snapshot_records.len(), 1);
        assert_eq!(store.failed_snapshots.len(), 1);
        assert!(
            store.records.capacity() < SOURCE_COUNT / 4,
            "unique Geometry records must retain a unique-asset high-water mark"
        );
        assert!(store.asset_handles.capacity() < SOURCE_COUNT / 4);
        assert!(store.asset_handle_lookup.capacity() < SOURCE_COUNT / 4);
        assert!(store.asset_snapshots.capacity() < SOURCE_COUNT / 4);
        assert!(store.asset_current.capacity() < SOURCE_COUNT / 4);
        assert!(store.live_snapshot_indices.capacity() < SOURCE_COUNT / 4);
        assert!(store.live_snapshot_lookup.capacity() < SOURCE_COUNT / 4);
        assert!(store.live_current.capacity() < SOURCE_COUNT / 4);
        assert!(store.record_snapshots.capacity() < SOURCE_COUNT / 4);
        assert!(store.snapshot_records.capacity() < SOURCE_COUNT / 4);
        assert!(store.failed_snapshots.capacity() < SOURCE_COUNT / 4);
        assert_eq!(store.source_geometry_joins.len(), SOURCE_COUNT);
    }

    #[test]
    fn sparse_live_geometry_join_does_not_materialize_the_frame_source_domain() {
        const SOURCE_DOMAIN: usize = 100_000;
        let assets = AssetServer::new();
        let handle = assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
        let (_, asset_version, asset_identity) = assets
            .geometries
            .get_entry_with_identity(handle)
            .expect("geometry missing");
        let template = extracted_item(handle, asset_version, asset_identity);
        let mut extracted = ExtractedScene::new();
        let mut lists = RenderLists::new();
        let mut last_source = None;
        for index in 0..SOURCE_DOMAIN {
            extracted.render_items.push(template.clone());
            last_source = Some(lists.intern_frame_source(ExtractedSourceId::from_index(index)));
        }
        let last_source = last_source.expect("source domain is empty");

        let mut store = GeometryFrameStore::new();
        store.capture_visible(&assets, &extracted, &lists);
        store.capture_live(&assets, &extracted, &lists, &[last_source]);

        assert_eq!(store.source_geometry_joins.len(), 1);
        assert!(
            store.source_geometry_joins.capacity() < SOURCE_DOMAIN / 4,
            "the post-DPE join must retain a live-source, not total-source, high-water mark"
        );
        assert!(
            store
                .resolver()
                .geometry_id(last_source, extracted.render_items.last().unwrap())
                .is_some()
        );
        let retained_join_allocation = store.source_geometry_joins.as_ptr();

        // `clear` invalidates the old join without scanning or zeroing the
        // 100k source domain, and an all-dead attempt publishes no stale slot.
        store.clear();
        store.capture_visible(&assets, &extracted, &lists);
        store.capture_live(&assets, &extracted, &lists, &[]);
        assert!(store.source_geometry_joins.is_empty());
        assert!(
            store
                .resolver()
                .geometry_id(last_source, extracted.render_items.last().unwrap())
                .is_none()
        );

        // A later warm live attempt reuses the retained compact join storage.
        store.clear();
        store.capture_visible(&assets, &extracted, &lists);
        store.capture_live(&assets, &extracted, &lists, &[last_source]);
        assert_eq!(
            store.source_geometry_joins.as_ptr(),
            retained_join_allocation
        );
        assert_eq!(store.source_geometry_joins.len(), 1);
    }

    #[test]
    fn multi_stream_geometry_reuses_its_spilled_vertex_slot_allocation() {
        let assets = AssetServer::new();
        // Box geometry has separate position, normal, and UV buffers, which
        // intentionally exceeds GeometryFramePhysical's two-slot inline form.
        let handle = assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
        let (_, asset_version, asset_identity) = assets
            .geometries
            .get_entry_with_identity(handle)
            .expect("geometry missing");
        let mut extracted = ExtractedScene::new();
        extracted
            .render_items
            .push(extracted_item(handle, asset_version, asset_identity));
        let mut lists = RenderLists::new();
        let source = lists.intern_frame_source(ExtractedSourceId::from_index(0));

        let mut resource_manager = headless_resource_manager();
        let mut draw_plans = DrawPlanStore::new();
        let mut store = GeometryFrameStore::new();
        store.capture_visible(&assets, &extracted, &lists);
        store.capture_live(&assets, &extracted, &lists, &[source]);
        store.prepare_live(&assets, &mut resource_manager, &mut draw_plans);

        let geometry = store
            .resolver()
            .geometry_id(source, &extracted.render_items[0])
            .expect("first geometry join missing");
        let first_slots = &store
            .resolver()
            .physical(geometry)
            .expect("first physical geometry missing")
            .vertex_buffers;
        assert!(first_slots.len() > 2);
        assert!(first_slots.spilled());
        let retained_slot_allocation = first_slots.as_ptr();
        let retained_slot_capacity = first_slots.capacity();

        store.clear();
        draw_plans.clear();
        assert_eq!(store.vertex_buffer_slot_scratch.len(), 1);
        assert_eq!(
            store.vertex_buffer_slot_scratch[0].as_ptr(),
            retained_slot_allocation
        );

        store.capture_visible(&assets, &extracted, &lists);
        store.capture_live(&assets, &extracted, &lists, &[source]);
        store.prepare_live(&assets, &mut resource_manager, &mut draw_plans);
        let geometry = store
            .resolver()
            .geometry_id(source, &extracted.render_items[0])
            .expect("warm geometry join missing");
        let warm_slots = &store
            .resolver()
            .physical(geometry)
            .expect("warm physical geometry missing")
            .vertex_buffers;
        assert_eq!(warm_slots.capacity(), retained_slot_capacity);
        assert_eq!(warm_slots.as_ptr(), retained_slot_allocation);
        assert!(store.vertex_buffer_slot_scratch.is_empty());

        store.clear();
        let retained = store.capacity_usage();
        assert_eq!(store.vertex_buffer_slot_scratch.len(), 1);
        store.trim_capacity(0, retained.retained_bytes);
        assert!(
            store.vertex_buffer_slot_scratch.is_empty(),
            "capacity trimming must release recycled nested spill allocations"
        );
        assert!(store.capacity_usage().retained_bytes < retained.retained_bytes);
    }

    #[test]
    fn post_dpe_geometry_revalidation_projects_only_the_nonprefix_live_subset() {
        const VISIBLE_COUNT: usize = 130;
        let assets = AssetServer::new();
        let mut extracted = ExtractedScene::new();
        let mut lists = RenderLists::new();
        let mut sources = Vec::with_capacity(VISIBLE_COUNT);
        for index in 0..VISIBLE_COUNT {
            let handle = assets.geometries.add(Geometry::new());
            let (_, version, identity) = assets
                .geometries
                .get_entry_with_identity(handle)
                .expect("geometry missing");
            extracted
                .render_items
                .push(extracted_item(handle, version, identity));
            sources.push(lists.intern_frame_source(ExtractedSourceId::from_index(index)));
        }

        let live = *sources.last().expect("visible source list is empty");
        let mut store = GeometryFrameStore::new();
        store.capture_visible(&assets, &extracted, &lists);
        store.capture_live(&assets, &extracted, &lists, &[live]);

        assert_eq!(store.asset_snapshots.len(), VISIBLE_COUNT);
        assert_eq!(store.live_snapshot_indices, [VISIBLE_COUNT - 1]);
        assert_eq!(store.live_current, [true]);
        assert_eq!(store.records.len(), 1);
        assert_eq!(
            store.records[0].key.handle,
            extracted.render_items.last().unwrap().geometry
        );
        assert!(
            store
                .resolver()
                .geometry_id(live, extracted.render_items.last().unwrap())
                .is_some()
        );
        assert!(
            store
                .resolver()
                .geometry_id(sources[0], &extracted.render_items[0])
                .is_none()
        );
        let resolver = store.resolver();
        let mut cursor = resolver.monotonic_cursor();
        assert!(
            resolver
                .geometry_id_monotonic(&mut cursor, sources[0], &extracted.render_items[0])
                .is_none()
        );
        assert!(
            resolver
                .geometry_id_monotonic(&mut cursor, live, extracted.render_items.last().unwrap(),)
                .is_some(),
            "a missing early source must not consume a later compact join"
        );
    }

    #[test]
    fn selected_live_geometry_rejects_same_version_replacement_after_pregraph_capture() {
        let assets = AssetServer::new();
        let handle = assets.geometries.add(Geometry::new());
        let (_, version, identity) = assets
            .geometries
            .get_entry_with_identity(handle)
            .expect("geometry missing");
        let mut extracted = ExtractedScene::new();
        extracted
            .render_items
            .push(extracted_item(handle, version, identity));
        let mut lists = RenderLists::new();
        let source = lists.intern_frame_source(ExtractedSourceId::from_index(0));
        let mut store = GeometryFrameStore::new();
        store.capture_visible(&assets, &extracted, &lists);

        assets.geometries.insert_ready(handle, Geometry::new());
        assert_eq!(assets.geometries.get_version(handle), Some(version));
        store.capture_live(&assets, &extracted, &lists, &[source]);

        assert_eq!(store.live_snapshot_indices, [0]);
        assert_eq!(store.live_current, [false]);
        assert!(store.records.is_empty());
        assert_eq!(store.joined_source_count(), 0);
        assert!(
            store
                .resolver()
                .geometry_id(source, &extracted.render_items[0])
                .is_none()
        );
    }

    #[test]
    fn geometry_snapshot_capacity_is_accounted_trimmed_and_releases_owners() {
        const UNIQUE_COUNT: usize = 1024;
        let assets = AssetServer::new();
        let mut extracted = ExtractedScene::new();
        let mut lists = RenderLists::new();
        let mut live = Vec::with_capacity(UNIQUE_COUNT);
        let mut first_handle = None;

        for index in 0..UNIQUE_COUNT {
            let handle = assets.geometries.add(Geometry::new());
            first_handle.get_or_insert(handle);
            let (_, asset_version, asset_identity) = assets
                .geometries
                .get_entry_with_identity(handle)
                .expect("geometry missing");
            extracted
                .render_items
                .push(extracted_item(handle, asset_version, asset_identity));
            live.push(lists.intern_frame_source(ExtractedSourceId::from_index(index)));
        }

        let mut store = GeometryFrameStore::new();
        store.capture_visible(&assets, &extracted, &lists);
        store.capture_live(&assets, &extracted, &lists, &live);

        let retained_owner = assets
            .geometries
            .get(first_handle.expect("no geometry was inserted"))
            .expect("first geometry missing");
        assert_eq!(
            Arc::strong_count(&retained_owner),
            4,
            "storage, test owner, unique snapshot, and live record are the only owners"
        );
        let populated = store.capacity_usage();
        assert!(populated.active_bytes > populated.baseline_bytes);

        store.clear();
        assert_eq!(
            Arc::strong_count(&retained_owner),
            2,
            "clearing the graph attempt must release both retained Geometry owners"
        );
        let cleared = store.capacity_usage();
        assert_eq!(cleared.active_bytes, 0);
        assert_eq!(cleared.retained_bytes, populated.retained_bytes);

        store.trim_capacity(cleared.baseline_bytes, cleared.retained_bytes);
        let trimmed = store.capacity_usage();
        assert!(trimmed.retained_bytes < cleared.retained_bytes);
        assert!(trimmed.retained_bytes >= trimmed.baseline_bytes);
        assert!(store.is_empty());
    }
}
