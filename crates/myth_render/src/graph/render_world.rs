//! Persistent CPU render-world authority.
//!
//! Scene mutations are acknowledged as soon as this CPU-derived cache has
//! captured them. GPU consumers use a second, retained revision journal: an
//! abandoned graph attempt therefore cannot lose a transform or fixed-state
//! delta. Submitted-frame transform history is advanced separately, only
//! after the queue accepted commands which consumed the source.

use bitflags::bitflags;
use myth_assets::storage::AssetEntry;
use myth_assets::{AssetServer, GeometryHandle, MaterialHandle};
use myth_resources::BoundingBox;
use myth_resources::RenderableMaterialTrait;
use myth_resources::material::{Material, MaterialType};
use myth_scene::{
    NodeHandle, SceneFullRebuildReasons, SceneMutation, SceneMutationBatch, SceneMutationCursor,
    SceneMutationFlags,
};
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::Key;

use super::extracted::{ExtractedRenderItem, MaterialClassification};
use super::frame::ExtractedSourceId;
use super::gpu_driven_policy::{GpuDrivenWorkloadSynopsis, mix64};

const MAX_RETAINED_RENDER_WORLD_CHANGES: usize = 262_144;

bitflags! {
    /// Render-world facts changed for one stable dense source.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub(crate) struct RenderWorldChangeFlags: u8 {
        const TRANSFORM = 1 << 0;
        const PREVIOUS_SETTLE = 1 << 1;
        const FIXED_STATE = 1 << 2;
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum RenderWorldFullRebuildReason {
    #[default]
    None,
    Cold,
    SceneChanged,
    SceneInitial,
    SceneExplicit,
    SceneOverflow,
    SceneUntrackedStorage,
    SceneCursorOrIdentityDrift,
    SceneUntrackedHierarchy,
    TopologyChanged,
    AssetAuthorityDrift,
    AssetEpochFallback,
    GpuFactsUpgrade,
    JournalOverflow,
}

/// Opaque progress cursor owned by one persistent render-world consumer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RenderWorldConsumerCursor {
    pub(crate) scene_id: u32,
    pub(crate) revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RenderWorldChangedSource {
    pub(crate) node: NodeHandle,
    pub(crate) source: ExtractedSourceId,
    pub(crate) flags: RenderWorldChangeFlags,
    revision: u64,
}

/// Exact old authority paired with one in-place dense-source refresh. Keeping
/// the old handles in the transaction lets the retained unique-asset tables
/// update in O(changed) without either rescanning the scene or retaining stale
/// assets forever.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderWorldSourceUpdate {
    pub(crate) node: NodeHandle,
    pub(crate) flags: RenderWorldChangeFlags,
    pub(crate) previous_geometry: GeometryHandle,
    pub(crate) previous_material: MaterialHandle,
    /// The refreshed record still carries a coherent complete GPU profile.
    pub(crate) gpu_facts_valid: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct ItemSynopsisContribution {
    opaque: u32,
    supported: u32,
    shadow_caster: u32,
    supported_shadow_caster: u32,
    transform_changed: u32,
    indexed_primitives: u64,
    fingerprint: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
struct GeometryAuthorityStamp {
    asset_version: u32,
    asset_identity: u64,
    structure_version: u64,
    layout_version: u64,
    topology: wgpu::PrimitiveTopology,
    index_format: Option<wgpu::IndexFormat>,
    index_count: u32,
    has_morph_targets: bool,
    has_vertex_attributes: bool,
    has_instanced_attributes: bool,
    gpu_facts_valid: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaterialAuthorityStamp {
    asset_version: u32,
    asset_identity: u64,
    material_version: u64,
    classification: u8,
    gpu_compatible: bool,
    gpu_facts_valid: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct ShadowCasterFact {
    bounds: Option<BoundingBox>,
}

impl ItemSynopsisContribution {
    fn from_item(item: &ExtractedRenderItem) -> Self {
        let opaque = !item.material_classification.intersects(
            MaterialClassification::TRANSPARENT | MaterialClassification::USES_TRANSMISSION,
        );
        let determinant = item.world_matrix.determinant();
        let supported = opaque
            && item.skeleton.is_none()
            && !item.geometry_has_morph_targets
            && !item.geometry_has_instanced_attributes
            && item.material_gpu_scene_compatible
            && item.geometry_index_format.is_some()
            && item.geometry_index_count != 0
            && item.geometry_has_vertex_attributes
            && item.geometry_topology == wgpu::PrimitiveTopology::TriangleList
            && item.world_matrix.is_finite()
            && item.prev_world_matrix.is_finite()
            && determinant.is_finite()
            && determinant.abs() > f32::EPSILON
            && item.world_aabb.is_finite()
            && item.world_aabb.min.cmple(item.world_aabb.max).all();
        let mut fingerprint = item.node_handle.data().as_ffi();
        for value in [
            item.geometry.data().as_ffi(),
            item.material.data().as_ffi(),
            item.geometry_asset_identity,
            u64::from(item.geometry_asset_version),
            item.geometry_structure_version,
            item.geometry_layout_version,
            item.material_asset_identity,
            u64::from(item.material_asset_version),
            item.material_version,
            u64::from(item.geometry_index_count),
            u64::from(item.material_index),
            u64::from(item.item_variant_flags),
            u64::from(item.material_classification.bits()),
            u64::from(item.cast_shadows),
        ] {
            fingerprint = mix64(fingerprint, value);
        }
        Self {
            opaque: u32::from(opaque),
            supported: u32::from(supported),
            shadow_caster: u32::from(item.cast_shadows),
            supported_shadow_caster: u32::from(supported && item.cast_shadows),
            transform_changed: u32::from(item.world_matrix != item.prev_world_matrix),
            indexed_primitives: if supported {
                u64::from(item.geometry_index_count / 3)
            } else {
                0
            },
            fingerprint,
        }
    }
}

/// Retained delta view. `changes` remains valid until the next mutable access
/// to the extracted scene.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderWorldDelta<'a> {
    pub(crate) scene_id: u32,
    pub(crate) revision: u64,
    pub(crate) topology_epoch: u64,
    pub(crate) requires_full_rebuild: bool,
    pub(crate) full_rebuild_reason: RenderWorldFullRebuildReason,
    pub(crate) changes: &'a [RenderWorldChangedSource],
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RenderWorldSyncStats {
    pub(crate) scene_id: u32,
    pub(crate) revision: u64,
    pub(crate) resident_domain_count: u32,
    pub(crate) mutation_count: u32,
    pub(crate) item_count: u32,
    pub(crate) items_reused: u32,
    pub(crate) items_examined: u32,
    pub(crate) items_added: u32,
    pub(crate) items_updated: u32,
    pub(crate) items_removed: u32,
    pub(crate) items_rebuilt: u32,
    pub(crate) shadow_casters_examined: u32,
    pub(crate) shadow_union_rebuilt: bool,
    pub(crate) submitted_history_settled: u32,
    pub(crate) asset_epoch_fallback: bool,
    pub(crate) retained_bytes: usize,
    pub(crate) full_rebuild_reason: RenderWorldFullRebuildReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderWorldSyncPlan {
    Reuse,
    UpdateTransforms,
    FullRebuild(RenderWorldFullRebuildReason),
}

/// Metadata parallel to `ExtractedScene::render_items`.
#[allow(clippy::struct_excessive_bools)]
struct RenderWorldDomain {
    scene_id: Option<u32>,
    /// Exact Scene frontier materialized by this domain. This is independent
    /// per renderer/domain even though Scene retains one compatibility
    /// acknowledgement frontier.
    scene_cursor: Option<SceneMutationCursor>,
    pending_scene_cursor: Option<SceneMutationCursor>,
    topology_epoch: u64,
    revision: u64,
    gpu_facts_complete: bool,
    geometry_asset_epoch: u64,
    material_asset_epoch: u64,
    node_to_source: FxHashMap<NodeHandle, ExtractedSourceId>,
    contributions: Vec<ItemSynopsisContribution>,
    geometry_ref_counts: FxHashMap<GeometryHandle, u32>,
    material_ref_counts: FxHashMap<MaterialHandle, u32>,
    geometry_stamps: FxHashMap<GeometryHandle, GeometryAuthorityStamp>,
    material_stamps: FxHashMap<MaterialHandle, MaterialAuthorityStamp>,
    has_unresolved_assets: bool,
    has_unstable_authority: bool,
    asset_authority_drifted: bool,
    mutation_scratch: Vec<SceneMutation>,
    changes: Vec<RenderWorldChangedSource>,
    change_indices: FxHashMap<NodeHandle, usize>,
    history_sources: Vec<ExtractedSourceId>,
    history_nodes: FxHashSet<NodeHandle>,
    acknowledged_revision: u64,
    full_rebuild_revision: Option<u64>,
    full_rebuild_reason: RenderWorldFullRebuildReason,
    shadow_caster_bounds_union: Option<BoundingBox>,
    shadow_caster_facts: Vec<ShadowCasterFact>,
    shadow_extrema_nodes: [Option<NodeHandle>; 6],
    synopsis: GpuDrivenWorkloadSynopsis,
    last_stats: RenderWorldSyncStats,
}

impl RenderWorldDomain {
    fn new() -> Self {
        Self {
            scene_id: None,
            scene_cursor: None,
            pending_scene_cursor: None,
            topology_epoch: 0,
            revision: 0,
            gpu_facts_complete: false,
            geometry_asset_epoch: 0,
            material_asset_epoch: 0,
            node_to_source: FxHashMap::default(),
            contributions: Vec::new(),
            geometry_ref_counts: FxHashMap::default(),
            material_ref_counts: FxHashMap::default(),
            geometry_stamps: FxHashMap::default(),
            material_stamps: FxHashMap::default(),
            has_unresolved_assets: false,
            has_unstable_authority: false,
            asset_authority_drifted: false,
            mutation_scratch: Vec::new(),
            changes: Vec::new(),
            change_indices: FxHashMap::default(),
            history_sources: Vec::new(),
            history_nodes: FxHashSet::default(),
            acknowledged_revision: 0,
            full_rebuild_revision: None,
            full_rebuild_reason: RenderWorldFullRebuildReason::None,
            shadow_caster_bounds_union: None,
            shadow_caster_facts: Vec::new(),
            shadow_extrema_nodes: [None; 6],
            synopsis: GpuDrivenWorkloadSynopsis::default(),
            last_stats: RenderWorldSyncStats::default(),
        }
    }

    fn validate_asset_authority(&mut self, assets: &AssetServer, capture_gpu_facts: bool) {
        if self.scene_id.is_none() || self.asset_authority_drifted || self.has_unstable_authority {
            return;
        }
        let geometry_epoch = assets.geometries.global_version();
        let material_epoch = assets.materials.global_version();
        if self.has_unresolved_assets
            && (geometry_epoch != self.geometry_asset_epoch
                || material_epoch != self.material_asset_epoch)
        {
            self.asset_authority_drifted = true;
            return;
        }

        // Geometry changes are storage transactions and therefore advance the
        // storage epoch. The overwhelmingly common warm frame must not scan
        // every unique geometry merely to prove that an unchanged epoch is
        // unchanged.
        if geometry_epoch != self.geometry_asset_epoch {
            let geometry_guard = assets.geometries.read_lock();
            for (&handle, expected) in &self.geometry_stamps {
                let Some(current) = geometry_guard.get_loaded(handle) else {
                    self.asset_authority_drifted = true;
                    return;
                };
                if current.version != expected.asset_version
                    || current.identity != expected.asset_identity
                {
                    self.asset_authority_drifted = true;
                    return;
                }
                if capture_gpu_facts && expected.gpu_facts_valid {
                    let index = current.index_attribute();
                    let actual = GeometryAuthorityStamp {
                        asset_version: current.version,
                        asset_identity: current.identity,
                        structure_version: current.structure_version(),
                        layout_version: current.layout_version(),
                        topology: current.topology,
                        index_format: index.map(|indices| indices.format),
                        index_count: index.map_or(0, |indices| indices.count),
                        has_morph_targets: current.has_morph_targets(),
                        has_vertex_attributes: !current.attributes().is_empty(),
                        has_instanced_attributes: current
                            .attributes()
                            .values()
                            .any(|attribute| attribute.step_mode == wgpu::VertexStepMode::Instance),
                        gpu_facts_valid: true,
                    };
                    if actual != *expected {
                        self.asset_authority_drifted = true;
                        return;
                    }
                }
            }
        }

        let material_guard = assets.materials.read_lock();
        for (&handle, expected) in &self.material_stamps {
            let Some(current) = material_guard.get_loaded(handle) else {
                self.asset_authority_drifted = true;
                return;
            };
            let Some(actual) = capture_material_authority_stamp(current, capture_gpu_facts, || {})
            else {
                // Never publish a stamp assembled from two material states.
                // The next sync takes the typed conservative rebuild path.
                self.asset_authority_drifted = true;
                return;
            };
            let expected_cpu = (
                expected.asset_version,
                expected.asset_identity,
                expected.material_version,
                expected.classification,
            );
            let actual_cpu = (
                actual.asset_version,
                actual.asset_identity,
                actual.material_version,
                actual.classification,
            );
            if actual_cpu != expected_cpu
                || (capture_gpu_facts
                    && expected.gpu_facts_valid
                    && actual.gpu_compatible != expected.gpu_compatible)
            {
                self.asset_authority_drifted = true;
                return;
            }
        }
        drop(material_guard);

        // Materials can mutate behind a stable storage entry, so their
        // interior versions are deliberately polled once per referenced
        // unique material. Unrelated storage transactions merely advance the
        // watermarks after all referenced authorities remain exact.
        self.geometry_asset_epoch = geometry_epoch;
        self.material_asset_epoch = material_epoch;
    }

    pub(crate) fn begin_sync(
        &mut self,
        batch: &SceneMutationBatch<'_>,
        capture_gpu_facts: bool,
        geometry_asset_epoch: u64,
        material_asset_epoch: u64,
    ) -> RenderWorldSyncPlan {
        self.mutation_scratch.clear();
        self.mutation_scratch.extend_from_slice(batch.entries);
        self.last_stats = RenderWorldSyncStats {
            scene_id: batch.scene_id,
            mutation_count: u32::try_from(batch.entries.len()).unwrap_or(u32::MAX),
            ..Default::default()
        };

        let scene_coverage_lost = self.scene_id == Some(batch.scene_id)
            && self
                .scene_cursor
                .is_none_or(|cursor| !batch.covers_cursor(cursor));
        self.pending_scene_cursor = Some(batch.cursor);

        let reason = if self.scene_id.is_none() {
            Some(RenderWorldFullRebuildReason::Cold)
        } else if self.scene_id != Some(batch.scene_id) {
            Some(RenderWorldFullRebuildReason::SceneChanged)
        } else if batch.requires_full_rebuild {
            Some(scene_rebuild_reason(batch.full_rebuild_reasons))
        } else if scene_coverage_lost {
            Some(RenderWorldFullRebuildReason::SceneCursorOrIdentityDrift)
        } else if batch.topology_epoch != self.topology_epoch
            || batch.entries.iter().any(|mutation| {
                mutation.flags.intersects(
                    SceneMutationFlags::TOPOLOGY
                        | SceneMutationFlags::ADDED
                        | SceneMutationFlags::REMOVED,
                )
            })
        {
            Some(RenderWorldFullRebuildReason::TopologyChanged)
        } else if self.asset_authority_drifted || self.has_unstable_authority {
            self.last_stats.asset_epoch_fallback = true;
            Some(RenderWorldFullRebuildReason::AssetAuthorityDrift)
        } else if geometry_asset_epoch != self.geometry_asset_epoch
            || material_asset_epoch != self.material_asset_epoch
        {
            self.last_stats.asset_epoch_fallback = true;
            Some(RenderWorldFullRebuildReason::AssetEpochFallback)
        } else if capture_gpu_facts && !self.gpu_facts_complete {
            Some(RenderWorldFullRebuildReason::GpuFactsUpgrade)
        } else {
            None
        };

        if let Some(reason) = reason {
            self.last_stats.full_rebuild_reason = reason;
            return RenderWorldSyncPlan::FullRebuild(reason);
        }

        if batch.entries.iter().any(|mutation| {
            mutation
                .flags
                .intersects(SceneMutationFlags::TRANSFORM | SceneMutationFlags::RENDERABLE)
        }) {
            RenderWorldSyncPlan::UpdateTransforms
        } else {
            RenderWorldSyncPlan::Reuse
        }
    }

    #[inline]
    pub(crate) fn mutations(&self) -> &[SceneMutation] {
        &self.mutation_scratch
    }

    #[inline]
    pub(crate) const fn owns_scene(&self, scene_id: u32) -> bool {
        matches!(self.scene_id, Some(current) if current == scene_id)
    }

    #[inline]
    pub(crate) fn source_for_node(&self, node: NodeHandle) -> Option<ExtractedSourceId> {
        self.node_to_source.get(&node).copied()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_full_rebuild(
        &mut self,
        records: &Vec<ExtractedRenderItem>,
        scene_id: u32,
        topology_epoch: u64,
        capture_gpu_facts: bool,
        geometry_asset_epoch: u64,
        material_asset_epoch: u64,
        has_unresolved_assets: bool,
        has_unstable_authority: bool,
        items_examined: usize,
    ) {
        let previous_item_count = self.node_to_source.len();
        let retained_item_count = records
            .iter()
            .filter(|record| self.node_to_source.contains_key(&record.node_handle))
            .count();
        self.scene_id = Some(scene_id);
        self.topology_epoch = topology_epoch;
        self.gpu_facts_complete = capture_gpu_facts;
        self.geometry_asset_epoch = geometry_asset_epoch;
        self.material_asset_epoch = material_asset_epoch;
        self.has_unresolved_assets = has_unresolved_assets;
        self.has_unstable_authority = has_unstable_authority;
        if self.has_unstable_authority {
            self.last_stats.asset_epoch_fallback = true;
            self.last_stats.full_rebuild_reason = RenderWorldFullRebuildReason::AssetAuthorityDrift;
        }
        self.asset_authority_drifted = false;
        if capture_gpu_facts {
            self.rebuild_synopsis(records);
        } else {
            self.contributions.clear();
            self.synopsis = GpuDrivenWorkloadSynopsis::default();
        }
        self.rebuild_indices(records);
        let conflicting_asset_stamps = self.rebuild_asset_stamps(records);
        self.has_unstable_authority |= conflicting_asset_stamps;
        if conflicting_asset_stamps {
            self.last_stats.asset_epoch_fallback = true;
            self.last_stats.full_rebuild_reason = RenderWorldFullRebuildReason::AssetAuthorityDrift;
        }
        self.rebuild_shadow_union(records);
        self.rebuild_history_sources(records);
        self.advance_revision();
        self.mark_full_rebuild();

        self.last_stats.item_count = u32::try_from(records.len()).unwrap_or(u32::MAX);
        self.last_stats.items_examined = u32::try_from(items_examined).unwrap_or(u32::MAX);
        self.last_stats.items_rebuilt = self.last_stats.item_count;
        self.last_stats.items_added =
            u32::try_from(records.len().saturating_sub(retained_item_count)).unwrap_or(u32::MAX);
        self.last_stats.items_removed =
            u32::try_from(previous_item_count.saturating_sub(retained_item_count))
                .unwrap_or(u32::MAX);
        self.last_stats.shadow_casters_examined = records
            .iter()
            .filter(|record| record.cast_shadows)
            .count()
            .try_into()
            .unwrap_or(u32::MAX);
        self.last_stats.shadow_union_rebuilt = true;
        self.finish_stats(records);
        self.commit_scene_cursor();
    }

    pub(crate) fn commit_transform_updates(
        &mut self,
        records: &Vec<ExtractedRenderItem>,
        changed_nodes: &[RenderWorldSourceUpdate],
        items_examined: usize,
        capture_gpu_facts: bool,
    ) {
        if !changed_nodes.is_empty() {
            self.advance_revision();
            for &RenderWorldSourceUpdate { node, flags, .. } in changed_nodes {
                if let Some(source) = self.source_for_node(node) {
                    self.record_change(node, source, flags);
                    if records
                        .get(source.index())
                        .is_some_and(|item| item.world_matrix != item.prev_world_matrix)
                        && self.history_nodes.insert(node)
                    {
                        self.history_sources.push(source);
                    }
                }
            }
        }
        let profile_remains_complete = capture_gpu_facts
            || (self.gpu_facts_complete
                && changed_nodes.iter().all(|update| update.gpu_facts_valid));
        for &update in changed_nodes {
            let Some(source) = self.source_for_node(update.node) else {
                continue;
            };
            let Some(item) = records.get(source.index()) else {
                continue;
            };
            if update.flags.contains(RenderWorldChangeFlags::FIXED_STATE) {
                self.refresh_record_asset_authority(update, item);
            }
            // A previously enriched domain stays enriched while Off is
            // selected. Pure transform work still updates the retained
            // synopsis from copied coherent GPU facts, avoiding an O(N)
            // profile-upgrade rebuild when Auto is selected again.
            if profile_remains_complete {
                self.replace_contribution(
                    source.index(),
                    ItemSynopsisContribution::from_item(item),
                );
            }
        }
        if self.gpu_facts_complete && !profile_remains_complete {
            self.gpu_facts_complete = false;
            for stamp in self.geometry_stamps.values_mut() {
                stamp.gpu_facts_valid = false;
            }
            for stamp in self.material_stamps.values_mut() {
                stamp.gpu_facts_valid = false;
            }
        }
        let (shadow_union_rebuilt, shadow_casters_examined) =
            self.update_shadow_union(records, changed_nodes);
        self.last_stats.item_count = u32::try_from(records.len()).unwrap_or(u32::MAX);
        self.last_stats.items_examined = u32::try_from(items_examined).unwrap_or(u32::MAX);
        self.last_stats.items_updated = u32::try_from(changed_nodes.len()).unwrap_or(u32::MAX);
        self.last_stats.items_reused = self
            .last_stats
            .item_count
            .saturating_sub(self.last_stats.items_updated);
        self.last_stats.shadow_union_rebuilt = shadow_union_rebuilt;
        self.last_stats.shadow_casters_examined =
            u32::try_from(shadow_casters_examined).unwrap_or(u32::MAX);
        self.finish_stats(records);
        self.commit_scene_cursor();
    }

    pub(crate) fn commit_reuse(&mut self, records: &Vec<ExtractedRenderItem>) {
        self.last_stats.item_count = u32::try_from(records.len()).unwrap_or(u32::MAX);
        self.last_stats.items_reused = self.last_stats.item_count;
        self.finish_stats(records);
        self.commit_scene_cursor();
    }

    /// Advance submitted previous-world history for an exact changed-source
    /// union. The caller must invoke this only after queue submission.
    pub(crate) fn commit_submitted_history(
        &mut self,
        cursor: RenderWorldConsumerCursor,
        records: &mut [ExtractedRenderItem],
    ) -> Option<usize> {
        // The retained source list is not revision-partitioned: settling a
        // newer authority for an older/stale submission would advance CPU
        // history past work the GPU actually consumed. Until M8 introduces a
        // revisioned history journal, accept only the exact active authority.
        if cursor != self.cursor() {
            return None;
        }
        let has_changes = self.history_sources.iter().any(|source| {
            records
                .get(source.index())
                .is_some_and(|item| item.prev_world_matrix != item.world_matrix)
        });
        if !has_changes {
            self.history_sources.clear();
            self.history_nodes.clear();
            return Some(0);
        }

        self.advance_revision();
        let mut settled = 0usize;
        for index in 0..self.history_sources.len() {
            let source = self.history_sources[index];
            let Some(item) = records.get_mut(source.index()) else {
                continue;
            };
            if item.prev_world_matrix == item.world_matrix {
                continue;
            }
            item.prev_world_matrix = item.world_matrix;
            let node = item.node_handle;
            let next_contribution = self
                .gpu_facts_complete
                .then(|| ItemSynopsisContribution::from_item(item));
            let _ = item;
            if let Some(next_contribution) = next_contribution {
                self.replace_contribution(source.index(), next_contribution);
            }
            self.record_change(node, source, RenderWorldChangeFlags::PREVIOUS_SETTLE);
            settled += 1;
        }
        self.history_sources.clear();
        self.history_nodes.clear();
        Some(settled)
    }

    pub(crate) fn delta_since(&self, cursor: RenderWorldConsumerCursor) -> RenderWorldDelta<'_> {
        let scene_id = self.scene_id.unwrap_or_default();
        let cursor_matches = cursor.scene_id == scene_id
            && cursor.revision == self.acknowledged_revision
            && cursor.revision <= self.revision;
        let requires_full_rebuild = !cursor_matches
            || self
                .full_rebuild_revision
                .is_some_and(|revision| revision > cursor.revision);
        RenderWorldDelta {
            scene_id,
            revision: self.revision,
            topology_epoch: self.topology_epoch,
            requires_full_rebuild,
            full_rebuild_reason: if requires_full_rebuild {
                self.full_rebuild_reason
            } else {
                RenderWorldFullRebuildReason::None
            },
            changes: if requires_full_rebuild {
                &[]
            } else {
                &self.changes
            },
        }
    }

    pub(crate) fn acknowledge_delta(&mut self, cursor: RenderWorldConsumerCursor) -> bool {
        if self.scene_id != Some(cursor.scene_id) || cursor.revision != self.revision {
            return false;
        }
        self.acknowledged_revision = cursor.revision;
        self.changes
            .retain(|change| change.revision > self.acknowledged_revision);
        self.rebuild_change_indices();
        if self
            .full_rebuild_revision
            .is_some_and(|revision| revision <= self.acknowledged_revision)
        {
            self.full_rebuild_revision = None;
            self.full_rebuild_reason = RenderWorldFullRebuildReason::None;
        }
        true
    }

    /// Fail closed when the Scene rejects the cursor paired with the captured
    /// CPU transaction. The retained bytes may be reused by the next full
    /// rebuild, but no consumer may treat them as current authority.
    pub(crate) fn invalidate_scene_authority(&mut self) {
        self.scene_id = None;
        self.scene_cursor = None;
        self.pending_scene_cursor = None;
        self.gpu_facts_complete = false;
        self.last_stats.full_rebuild_reason =
            RenderWorldFullRebuildReason::SceneCursorOrIdentityDrift;
        self.mark_full_rebuild();
    }

    #[inline]
    pub(crate) fn note_full_rebuild_reason(&mut self, reason: RenderWorldFullRebuildReason) {
        self.last_stats.full_rebuild_reason = reason;
        if self.full_rebuild_revision.is_some() {
            self.full_rebuild_reason = reason;
        }
    }

    #[inline]
    pub(crate) const fn cursor(&self) -> RenderWorldConsumerCursor {
        RenderWorldConsumerCursor {
            scene_id: match self.scene_id {
                Some(scene_id) => scene_id,
                None => 0,
            },
            revision: self.revision,
        }
    }

    #[inline]
    pub(crate) const fn synopsis(&self) -> GpuDrivenWorkloadSynopsis {
        self.synopsis
    }

    #[inline]
    pub(crate) const fn shadow_caster_bounds_union(&self) -> Option<BoundingBox> {
        self.shadow_caster_bounds_union
    }

    #[inline]
    pub(crate) const fn last_stats(&self) -> RenderWorldSyncStats {
        self.last_stats
    }

    fn advance_revision(&mut self) {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("RenderWorld revision exhausted");
    }

    #[inline]
    fn commit_scene_cursor(&mut self) {
        if let Some(cursor) = self.pending_scene_cursor.take() {
            self.scene_cursor = Some(cursor);
        }
    }

    fn mark_full_rebuild(&mut self) {
        self.changes.clear();
        self.change_indices.clear();
        self.full_rebuild_revision = Some(self.revision);
        self.full_rebuild_reason =
            if self.last_stats.full_rebuild_reason == RenderWorldFullRebuildReason::None {
                RenderWorldFullRebuildReason::SceneCursorOrIdentityDrift
            } else {
                self.last_stats.full_rebuild_reason
            };
    }

    fn record_change(
        &mut self,
        node: NodeHandle,
        source: ExtractedSourceId,
        flags: RenderWorldChangeFlags,
    ) {
        if self.full_rebuild_revision.is_some() {
            return;
        }
        if let Some(&index) = self.change_indices.get(&node) {
            let change = &mut self.changes[index];
            change.source = source;
            change.flags.insert(flags);
            change.revision = self.revision;
            return;
        }
        if self.changes.len() >= MAX_RETAINED_RENDER_WORLD_CHANGES {
            self.last_stats.full_rebuild_reason = RenderWorldFullRebuildReason::JournalOverflow;
            self.mark_full_rebuild();
            return;
        }
        let index = self.changes.len();
        self.changes.push(RenderWorldChangedSource {
            node,
            source,
            flags,
            revision: self.revision,
        });
        self.change_indices.insert(node, index);
    }

    fn rebuild_indices(&mut self, records: &[ExtractedRenderItem]) {
        self.node_to_source.clear();
        self.node_to_source.reserve(records.len());
        for (index, record) in records.iter().enumerate() {
            self.node_to_source
                .insert(record.node_handle, ExtractedSourceId::from_index(index));
        }
    }

    fn rebuild_asset_stamps(&mut self, records: &[ExtractedRenderItem]) -> bool {
        self.geometry_ref_counts.clear();
        self.material_ref_counts.clear();
        self.geometry_stamps.clear();
        self.material_stamps.clear();
        let mut conflicting_authority = false;
        for record in records {
            increment_ref_count(&mut self.geometry_ref_counts, record.geometry);
            increment_ref_count(&mut self.material_ref_counts, record.material);
            let geometry_stamp = geometry_authority_stamp(record, self.gpu_facts_complete);
            let material_stamp = material_authority_stamp(record, self.gpu_facts_complete);
            conflicting_authority |= self
                .geometry_stamps
                .get(&record.geometry)
                .is_some_and(|existing| *existing != geometry_stamp);
            conflicting_authority |= self
                .material_stamps
                .get(&record.material)
                .is_some_and(|existing| *existing != material_stamp);
            self.geometry_stamps
                .entry(record.geometry)
                .or_insert(geometry_stamp);
            self.material_stamps
                .entry(record.material)
                .or_insert(material_stamp);
        }
        conflicting_authority
    }

    fn observe_record_asset_stamps(&mut self, record: &ExtractedRenderItem) {
        self.geometry_stamps.insert(
            record.geometry,
            geometry_authority_stamp(record, self.gpu_facts_complete),
        );
        self.material_stamps.insert(
            record.material,
            material_authority_stamp(record, self.gpu_facts_complete),
        );
    }

    fn refresh_record_asset_authority(
        &mut self,
        update: RenderWorldSourceUpdate,
        record: &ExtractedRenderItem,
    ) {
        if update.previous_geometry != record.geometry {
            if decrement_ref_count(&mut self.geometry_ref_counts, update.previous_geometry) {
                self.geometry_stamps.remove(&update.previous_geometry);
            }
            increment_ref_count(&mut self.geometry_ref_counts, record.geometry);
        }
        if update.previous_material != record.material {
            if decrement_ref_count(&mut self.material_ref_counts, update.previous_material) {
                self.material_stamps.remove(&update.previous_material);
            }
            increment_ref_count(&mut self.material_ref_counts, record.material);
        }
        self.observe_record_asset_stamps(record);
    }

    fn rebuild_history_sources(&mut self, records: &[ExtractedRenderItem]) {
        self.history_sources.clear();
        self.history_nodes.clear();
        for (index, record) in records.iter().enumerate() {
            if record.world_matrix != record.prev_world_matrix {
                self.history_sources
                    .push(ExtractedSourceId::from_index(index));
                self.history_nodes.insert(record.node_handle);
            }
        }
    }

    fn rebuild_shadow_union(&mut self, records: &[ExtractedRenderItem]) {
        self.shadow_caster_facts.clear();
        self.shadow_caster_facts.reserve(records.len());
        self.shadow_caster_bounds_union = None;
        self.shadow_extrema_nodes = [None; 6];
        for record in records {
            let fact = ShadowCasterFact {
                bounds: conservative_shadow_bounds(record),
            };
            self.shadow_caster_facts.push(fact);
            if let Some(bounds) = fact.bounds {
                self.expand_shadow_union(record.node_handle, bounds);
            }
        }
    }

    fn update_shadow_union(
        &mut self,
        records: &[ExtractedRenderItem],
        changed_nodes: &[RenderWorldSourceUpdate],
    ) -> (bool, usize) {
        let mut rebuild = false;
        let mut examined = 0usize;
        for &RenderWorldSourceUpdate { node, flags, .. } in changed_nodes {
            if !flags
                .intersects(RenderWorldChangeFlags::TRANSFORM | RenderWorldChangeFlags::FIXED_STATE)
            {
                continue;
            }
            let Some(source) = self.source_for_node(node) else {
                continue;
            };
            let old = self
                .shadow_caster_facts
                .get(source.index())
                .copied()
                .unwrap_or_default();
            let new = records
                .get(source.index())
                .and_then(conservative_shadow_bounds);
            if shadow_bounds_eq(old.bounds, new) {
                continue;
            }
            if old.bounds.is_some() || new.is_some() {
                examined = examined.saturating_add(1);
            }
            if let Some(old_bounds) = old.bounds {
                for (axis, contributor) in self.shadow_extrema_nodes.iter().enumerate() {
                    if *contributor != Some(node) {
                        continue;
                    }
                    let contracted = new.is_none_or(|new_bounds| match axis {
                        0 => new_bounds.min.x > old_bounds.min.x,
                        1 => new_bounds.min.y > old_bounds.min.y,
                        2 => new_bounds.min.z > old_bounds.min.z,
                        3 => new_bounds.max.x < old_bounds.max.x,
                        4 => new_bounds.max.y < old_bounds.max.y,
                        _ => new_bounds.max.z < old_bounds.max.z,
                    });
                    rebuild |= contracted;
                }
            }
            if let Some(slot) = self.shadow_caster_facts.get_mut(source.index()) {
                slot.bounds = new;
            }
            if !rebuild && let Some(new_bounds) = new {
                self.expand_shadow_union(node, new_bounds);
            }
        }

        if rebuild {
            self.rebuild_shadow_union(records);
            (
                true,
                records.iter().filter(|item| item.cast_shadows).count(),
            )
        } else {
            (false, examined)
        }
    }

    fn expand_shadow_union(&mut self, node: NodeHandle, bounds: BoundingBox) {
        let Some(current) = self.shadow_caster_bounds_union else {
            self.shadow_caster_bounds_union = Some(bounds);
            self.shadow_extrema_nodes.fill(Some(node));
            return;
        };
        let mut next = current;
        if bounds.min.x < next.min.x {
            next.min.x = bounds.min.x;
            self.shadow_extrema_nodes[0] = Some(node);
        }
        if bounds.min.y < next.min.y {
            next.min.y = bounds.min.y;
            self.shadow_extrema_nodes[1] = Some(node);
        }
        if bounds.min.z < next.min.z {
            next.min.z = bounds.min.z;
            self.shadow_extrema_nodes[2] = Some(node);
        }
        if bounds.max.x > next.max.x {
            next.max.x = bounds.max.x;
            self.shadow_extrema_nodes[3] = Some(node);
        }
        if bounds.max.y > next.max.y {
            next.max.y = bounds.max.y;
            self.shadow_extrema_nodes[4] = Some(node);
        }
        if bounds.max.z > next.max.z {
            next.max.z = bounds.max.z;
            self.shadow_extrema_nodes[5] = Some(node);
        }
        self.shadow_caster_bounds_union = Some(next);
    }

    fn rebuild_change_indices(&mut self) {
        self.change_indices.clear();
        for (index, change) in self.changes.iter().enumerate() {
            self.change_indices.insert(change.node, index);
        }
    }

    fn rebuild_synopsis(&mut self, records: &[ExtractedRenderItem]) {
        self.contributions.clear();
        self.contributions.reserve(records.len());
        self.synopsis = GpuDrivenWorkloadSynopsis::default();
        for item in records {
            let contribution = ItemSynopsisContribution::from_item(item);
            add_contribution(&mut self.synopsis, contribution);
            self.contributions.push(contribution);
        }
    }

    fn replace_contribution(&mut self, index: usize, next: ItemSynopsisContribution) {
        let Some(previous) = self.contributions.get_mut(index) else {
            // A missing parallel contribution is a profile-authority failure,
            // never an invitation to publish a partial synopsis.
            self.gpu_facts_complete = false;
            return;
        };
        remove_contribution(&mut self.synopsis, *previous);
        add_contribution(&mut self.synopsis, next);
        *previous = next;
    }

    fn finish_stats(&mut self, records: &Vec<ExtractedRenderItem>) {
        self.last_stats.revision = self.revision;
        self.last_stats.resident_domain_count = 1;
        self.last_stats.retained_bytes = self.retained_storage_bytes(records);
    }

    fn retained_storage_bytes(&self, records: &Vec<ExtractedRenderItem>) -> usize {
        records
            .capacity()
            .saturating_mul(std::mem::size_of::<ExtractedRenderItem>())
            .saturating_add(
                self.changes
                    .capacity()
                    .saturating_mul(std::mem::size_of::<RenderWorldChangedSource>()),
            )
            .saturating_add(
                self.mutation_scratch
                    .capacity()
                    .saturating_mul(std::mem::size_of::<SceneMutation>()),
            )
            .saturating_add(
                self.contributions
                    .capacity()
                    .saturating_mul(std::mem::size_of::<ItemSynopsisContribution>()),
            )
            .saturating_add(
                self.history_sources
                    .capacity()
                    .saturating_mul(std::mem::size_of::<ExtractedSourceId>()),
            )
            .saturating_add(map_retained_bytes(&self.node_to_source))
            .saturating_add(map_retained_bytes(&self.change_indices))
            .saturating_add(map_retained_bytes::<GeometryHandle, u32>(
                &self.geometry_ref_counts,
            ))
            .saturating_add(map_retained_bytes::<MaterialHandle, u32>(
                &self.material_ref_counts,
            ))
            .saturating_add(
                map_retained_bytes::<GeometryHandle, GeometryAuthorityStamp>(&self.geometry_stamps),
            )
            .saturating_add(
                map_retained_bytes::<MaterialHandle, MaterialAuthorityStamp>(&self.material_stamps),
            )
            .saturating_add(set_retained_bytes(&self.history_nodes))
            .saturating_add(
                self.shadow_caster_facts
                    .capacity()
                    .saturating_mul(std::mem::size_of::<ShadowCasterFact>()),
            )
    }

    fn retained_bytes(&self, records: &Vec<ExtractedRenderItem>) -> usize {
        self.retained_storage_bytes(records)
    }

    fn trim_retained_storage(&mut self) {
        self.node_to_source.shrink_to_fit();
        self.contributions.shrink_to_fit();
        self.geometry_ref_counts.shrink_to_fit();
        self.material_ref_counts.shrink_to_fit();
        self.geometry_stamps.shrink_to_fit();
        self.material_stamps.shrink_to_fit();
        self.mutation_scratch.shrink_to_fit();
        // Journal contents and revisions remain unchanged: shrinking backing
        // storage is safe, clearing it here would silently lose an abandoned
        // graph attempt's retained delta.
        self.changes.shrink_to_fit();
        self.change_indices.shrink_to_fit();
        self.history_sources.shrink_to_fit();
        self.history_nodes.shrink_to_fit();
        self.shadow_caster_facts.shrink_to_fit();
    }

    fn retained_storage_underutilized(&self) -> bool {
        vec_underutilized(&self.contributions)
            || vec_underutilized(&self.mutation_scratch)
            || vec_underutilized(&self.changes)
            || vec_underutilized(&self.history_sources)
            || vec_underutilized(&self.shadow_caster_facts)
            || map_underutilized(&self.node_to_source)
            || map_underutilized(&self.geometry_ref_counts)
            || map_underutilized(&self.material_ref_counts)
            || map_underutilized(&self.geometry_stamps)
            || map_underutilized(&self.material_stamps)
            || map_underutilized(&self.change_indices)
            || set_underutilized(&self.history_nodes)
    }
}

impl Default for RenderWorldDomain {
    fn default() -> Self {
        Self::new()
    }
}

struct DormantRenderWorldDomain {
    domain: RenderWorldDomain,
    records: Vec<ExtractedRenderItem>,
    last_used_seconds: f64,
    last_used_sync: u64,
}

const MAX_DORMANT_RENDER_WORLD_SCENES: usize = 4;
const MAX_DORMANT_RENDER_WORLD_BYTES: usize = 64 * 1024 * 1024;
/// Real idle time after which an inactive CPU RenderWorld domain is retired.
/// Keeping this independent of frame rate gives 30 Hz and 240 Hz callers the
/// same residency contract.
const DORMANT_RENDER_WORLD_TTL_SECONDS: f64 = 1.5;
const ACTIVE_RENDER_WORLD_TRIM_FRAMES: u32 = 300;

/// Renderer-owned per-scene render-world residency. The active scene keeps its
/// record Vec in `ExtractedScene` for zero-copy downstream indexing; switching
/// scenes swaps that Vec into/out of the corresponding dormant domain.
pub(crate) struct PersistentRenderWorld {
    active: RenderWorldDomain,
    dormant: FxHashMap<u32, DormantRenderWorldDomain>,
    sync_serial: u64,
    residency_clock_seconds: f64,
    last_residency_sample_seconds: Option<f64>,
    active_underutilized_frames: u32,
    active_trimmed_this_frame: bool,
}

impl PersistentRenderWorld {
    pub(crate) fn new() -> Self {
        Self {
            active: RenderWorldDomain::new(),
            dormant: FxHashMap::default(),
            sync_serial: 0,
            residency_clock_seconds: 0.0,
            last_residency_sample_seconds: None,
            active_underutilized_frames: 0,
            active_trimmed_this_frame: false,
        }
    }

    /// Activate one scene by moving (not cloning) its dense record authority.
    /// Returns true when the active domain changed.
    pub(crate) fn activate_scene(
        &mut self,
        scene_id: u32,
        records: &mut Vec<ExtractedRenderItem>,
        elapsed_seconds: f64,
    ) -> bool {
        self.sync_serial = self.sync_serial.saturating_add(1);
        let now_seconds = self.observe_residency_time(elapsed_seconds);
        self.active_trimmed_this_frame = false;
        if self.active.owns_scene(scene_id) {
            self.maybe_trim_active(records);
            self.evict_dormant_domains(now_seconds);
            return false;
        }

        if let Some(old_scene_id) = self.active.scene_id {
            let old_domain = std::mem::take(&mut self.active);
            let old_records = std::mem::take(records);
            let replaced = self.dormant.insert(
                old_scene_id,
                DormantRenderWorldDomain {
                    domain: old_domain,
                    records: old_records,
                    last_used_seconds: now_seconds,
                    last_used_sync: self.sync_serial,
                },
            );
            debug_assert!(
                replaced.is_none(),
                "active RenderWorld duplicated a dormant scene"
            );
        }

        if let Some(cached) = self.dormant.remove(&scene_id) {
            self.active = cached.domain;
            *records = cached.records;
        } else {
            self.active = RenderWorldDomain::new();
            *records = Vec::with_capacity(1024);
        }
        self.active_underutilized_frames = 0;
        self.evict_dormant_domains(now_seconds);
        true
    }

    #[inline]
    pub(crate) fn validate_asset_authority(
        &mut self,
        assets: &AssetServer,
        capture_gpu_facts: bool,
    ) {
        self.active
            .validate_asset_authority(assets, capture_gpu_facts);
    }

    /// Mark the active authority incomplete when an eligible Scene source is
    /// waiting for asynchronous geometry/material publication. The Scene
    /// mutation may now be acknowledged safely: the next asset epoch advance
    /// forces a conservative rebuild and cannot forget the source.
    #[inline]
    pub(crate) fn note_unresolved_assets(&mut self) {
        self.active.has_unresolved_assets = true;
    }

    #[inline]
    pub(crate) const fn gpu_facts_complete(&self) -> bool {
        self.active.gpu_facts_complete
    }

    pub(crate) fn begin_sync(
        &mut self,
        batch: &SceneMutationBatch<'_>,
        capture_gpu_facts: bool,
        geometry_asset_epoch: u64,
        material_asset_epoch: u64,
    ) -> RenderWorldSyncPlan {
        self.active.begin_sync(
            batch,
            capture_gpu_facts,
            geometry_asset_epoch,
            material_asset_epoch,
        )
    }

    #[inline]
    pub(crate) fn mutations(&self) -> &[SceneMutation] {
        self.active.mutations()
    }

    #[inline]
    pub(crate) const fn owns_scene(&self, scene_id: u32) -> bool {
        self.active.owns_scene(scene_id)
    }

    #[inline]
    pub(crate) fn source_for_node(&self, node: NodeHandle) -> Option<ExtractedSourceId> {
        self.active.source_for_node(node)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_full_rebuild(
        &mut self,
        records: &Vec<ExtractedRenderItem>,
        scene_id: u32,
        topology_epoch: u64,
        capture_gpu_facts: bool,
        geometry_asset_epoch: u64,
        material_asset_epoch: u64,
        has_unresolved_assets: bool,
        has_unstable_authority: bool,
        items_examined: usize,
    ) {
        self.active.commit_full_rebuild(
            records,
            scene_id,
            topology_epoch,
            capture_gpu_facts,
            geometry_asset_epoch,
            material_asset_epoch,
            has_unresolved_assets,
            has_unstable_authority,
            items_examined,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_transform_updates(
        &mut self,
        records: &Vec<ExtractedRenderItem>,
        changed_nodes: &[RenderWorldSourceUpdate],
        items_examined: usize,
        capture_gpu_facts: bool,
    ) {
        self.active.commit_transform_updates(
            records,
            changed_nodes,
            items_examined,
            capture_gpu_facts,
        );
    }

    #[inline]
    pub(crate) fn commit_reuse(&mut self, records: &Vec<ExtractedRenderItem>) {
        self.active.commit_reuse(records);
    }

    #[inline]
    pub(crate) fn commit_submitted_history(
        &mut self,
        cursor: RenderWorldConsumerCursor,
        records: &mut [ExtractedRenderItem],
    ) -> Option<usize> {
        self.active.commit_submitted_history(cursor, records)
    }

    #[inline]
    pub(crate) fn delta_since(&self, cursor: RenderWorldConsumerCursor) -> RenderWorldDelta<'_> {
        self.active.delta_since(cursor)
    }

    #[inline]
    pub(crate) fn acknowledge_delta(&mut self, cursor: RenderWorldConsumerCursor) -> bool {
        self.active.acknowledge_delta(cursor)
    }

    #[inline]
    pub(crate) fn invalidate_scene_authority(&mut self) {
        self.active.invalidate_scene_authority();
    }

    #[inline]
    pub(crate) fn note_full_rebuild_reason(&mut self, reason: RenderWorldFullRebuildReason) {
        self.active.note_full_rebuild_reason(reason);
    }

    #[inline]
    pub(crate) const fn cursor(&self) -> RenderWorldConsumerCursor {
        self.active.cursor()
    }

    #[inline]
    pub(crate) const fn synopsis(&self) -> GpuDrivenWorkloadSynopsis {
        self.active.synopsis()
    }

    #[inline]
    pub(crate) const fn shadow_caster_bounds_union(&self) -> Option<BoundingBox> {
        self.active.shadow_caster_bounds_union()
    }

    pub(crate) fn last_stats(&self) -> RenderWorldSyncStats {
        let mut stats = self.active.last_stats();
        stats.revision = self.active.revision;
        stats.resident_domain_count =
            u32::try_from(self.dormant.len().saturating_add(1)).unwrap_or(u32::MAX);
        stats.retained_bytes = stats
            .retained_bytes
            .saturating_add(
                self.dormant
                    .values()
                    .map(|cached| cached.domain.retained_bytes(&cached.records))
                    .sum::<usize>(),
            )
            .saturating_add(map_retained_bytes::<u32, DormantRenderWorldDomain>(
                &self.dormant,
            ));
        stats
    }

    #[inline]
    pub(crate) fn take_active_trim_signal(&mut self) -> bool {
        std::mem::take(&mut self.active_trimmed_this_frame)
    }

    fn maybe_trim_active(&mut self, records: &mut Vec<ExtractedRenderItem>) {
        let underutilized = (records.capacity() > 1024
            && records.len().saturating_mul(4) < records.capacity())
            || self.active.retained_storage_underutilized();
        if !underutilized {
            self.active_underutilized_frames = 0;
            return;
        }
        self.active_underutilized_frames = self.active_underutilized_frames.saturating_add(1);
        if self.active_underutilized_frames >= ACTIVE_RENDER_WORLD_TRIM_FRAMES {
            records.shrink_to(records.len().saturating_mul(2).max(1024));
            self.active.trim_retained_storage();
            self.active_trimmed_this_frame = true;
            self.active_underutilized_frames = 0;
        }
    }

    /// Convert the externally supplied elapsed clock into a monotonic cache
    /// clock. A reset/regression merely establishes a new baseline; it never
    /// turns retained domains into accidentally ancient entries.
    fn observe_residency_time(&mut self, sample_seconds: f64) -> f64 {
        if !sample_seconds.is_finite() {
            return self.residency_clock_seconds;
        }
        if let Some(previous) = self.last_residency_sample_seconds {
            let elapsed = sample_seconds - previous;
            if elapsed > 0.0 {
                self.residency_clock_seconds += elapsed;
            }
        }
        self.last_residency_sample_seconds = Some(sample_seconds);
        self.residency_clock_seconds
    }

    fn evict_dormant_domains(&mut self, now_seconds: f64) {
        loop {
            let retained_bytes = self
                .dormant
                .values()
                .map(|cached| cached.domain.retained_bytes(&cached.records))
                .sum::<usize>();
            let ttl_victim = self
                .dormant
                .iter()
                .filter(|(_, cached)| {
                    now_seconds - cached.last_used_seconds >= DORMANT_RENDER_WORLD_TTL_SECONDS
                })
                .min_by(|(left_scene_id, left), (right_scene_id, right)| {
                    left.last_used_seconds
                        .total_cmp(&right.last_used_seconds)
                        .then_with(|| left.last_used_sync.cmp(&right.last_used_sync))
                        .then_with(|| left_scene_id.cmp(right_scene_id))
                })
                .map(|(scene_id, _)| *scene_id);
            let over_budget = self.dormant.len() > MAX_DORMANT_RENDER_WORLD_SCENES
                || retained_bytes > MAX_DORMANT_RENDER_WORLD_BYTES;
            let victim = ttl_victim.or_else(|| {
                over_budget.then(|| {
                    self.dormant
                        .iter()
                        .min_by(|(left_scene_id, left), (right_scene_id, right)| {
                            left.last_used_seconds
                                .total_cmp(&right.last_used_seconds)
                                .then_with(|| left.last_used_sync.cmp(&right.last_used_sync))
                                .then_with(|| left_scene_id.cmp(right_scene_id))
                        })
                        .map(|(scene_id, _)| *scene_id)
                        .expect("over-budget dormant RenderWorld has no eviction victim")
                })
            });
            let Some(victim) = victim else {
                break;
            };
            self.dormant.remove(&victim);
        }
    }
}

impl Default for PersistentRenderWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn increment_ref_count<H: Eq + std::hash::Hash + Copy>(counts: &mut FxHashMap<H, u32>, handle: H) {
    let count = counts.entry(handle).or_default();
    *count = count.saturating_add(1);
}

/// Returns true when the unique authority no longer has a resident source.
#[inline]
fn decrement_ref_count<H: Eq + std::hash::Hash + Copy>(
    counts: &mut FxHashMap<H, u32>,
    handle: H,
) -> bool {
    let Some(count) = counts.get_mut(&handle) else {
        debug_assert!(false, "RenderWorld asset ref-count authority drifted");
        return false;
    };
    if *count > 1 {
        *count -= 1;
        false
    } else {
        counts.remove(&handle);
        true
    }
}

#[inline]
fn map_retained_bytes<K, V>(map: &FxHashMap<K, V>) -> usize {
    map.capacity()
        .saturating_mul(std::mem::size_of::<(K, V)>().saturating_add(1))
}

#[inline]
fn set_retained_bytes<T>(set: &FxHashSet<T>) -> usize {
    set.capacity()
        .saturating_mul(std::mem::size_of::<T>().saturating_add(1))
}

#[inline]
fn vec_underutilized<T>(values: &Vec<T>) -> bool {
    values.capacity() > 1024 && values.len().saturating_mul(4) < values.capacity()
}

#[inline]
fn map_underutilized<K, V>(values: &FxHashMap<K, V>) -> bool {
    values.capacity() > 1024 && values.len().saturating_mul(4) < values.capacity()
}

#[inline]
fn set_underutilized<T>(values: &FxHashSet<T>) -> bool {
    values.capacity() > 1024 && values.len().saturating_mul(4) < values.capacity()
}

#[inline]
fn capture_material_authority_stamp(
    current: &AssetEntry<Material>,
    capture_gpu_facts: bool,
    between_version_reads: impl FnOnce(),
) -> Option<MaterialAuthorityStamp> {
    let version_before = current.data.version();
    // Test instrumentation uses this exact seam to prove that a concurrent
    // interior mutation cannot produce a mixed classification/version stamp.
    // The production closure is empty and inlines away.
    between_version_reads();
    let classification =
        u8::from(current.is_transparent()) | (u8::from(current.use_transmission()) << 1);
    let gpu_compatible = capture_gpu_facts && !matches!(&current.data, MaterialType::Custom(_));
    let version_after = current.data.version();
    (version_before == version_after).then_some(MaterialAuthorityStamp {
        asset_version: current.version,
        asset_identity: current.identity,
        material_version: version_after,
        classification,
        gpu_compatible,
        gpu_facts_valid: capture_gpu_facts,
    })
}

#[inline]
fn geometry_authority_stamp(
    record: &ExtractedRenderItem,
    gpu_facts_valid: bool,
) -> GeometryAuthorityStamp {
    GeometryAuthorityStamp {
        asset_version: record.geometry_asset_version,
        asset_identity: record.geometry_asset_identity,
        structure_version: record.geometry_structure_version,
        layout_version: record.geometry_layout_version,
        topology: record.geometry_topology,
        index_format: record.geometry_index_format,
        index_count: record.geometry_index_count,
        has_morph_targets: record.geometry_has_morph_targets,
        has_vertex_attributes: record.geometry_has_vertex_attributes,
        has_instanced_attributes: record.geometry_has_instanced_attributes,
        gpu_facts_valid,
    }
}

#[inline]
fn material_authority_stamp(
    record: &ExtractedRenderItem,
    gpu_facts_valid: bool,
) -> MaterialAuthorityStamp {
    MaterialAuthorityStamp {
        asset_version: record.material_asset_version,
        asset_identity: record.material_asset_identity,
        material_version: record.material_version,
        classification: record.material_classification.bits(),
        gpu_compatible: record.material_gpu_scene_compatible,
        gpu_facts_valid,
    }
}

#[inline]
pub(crate) fn node_order(node: NodeHandle) -> u64 {
    node.data().as_ffi()
}

fn add_contribution(
    synopsis: &mut GpuDrivenWorkloadSynopsis,
    contribution: ItemSynopsisContribution,
) {
    synopsis.extracted_item_count = synopsis.extracted_item_count.saturating_add(1);
    synopsis.opaque_item_count = synopsis
        .opaque_item_count
        .saturating_add(contribution.opaque);
    synopsis.supported_opaque_count = synopsis
        .supported_opaque_count
        .saturating_add(contribution.supported);
    synopsis.shadow_caster_count = synopsis
        .shadow_caster_count
        .saturating_add(contribution.shadow_caster);
    synopsis.supported_shadow_caster_count = synopsis
        .supported_shadow_caster_count
        .saturating_add(contribution.supported_shadow_caster);
    synopsis.transform_changed_count = synopsis
        .transform_changed_count
        .saturating_add(contribution.transform_changed);
    synopsis.indexed_primitive_count = synopsis
        .indexed_primitive_count
        .saturating_add(contribution.indexed_primitives);
    synopsis.topology_fingerprint ^= contribution.fingerprint;
}

fn remove_contribution(
    synopsis: &mut GpuDrivenWorkloadSynopsis,
    contribution: ItemSynopsisContribution,
) {
    synopsis.extracted_item_count = synopsis.extracted_item_count.saturating_sub(1);
    synopsis.opaque_item_count = synopsis
        .opaque_item_count
        .saturating_sub(contribution.opaque);
    synopsis.supported_opaque_count = synopsis
        .supported_opaque_count
        .saturating_sub(contribution.supported);
    synopsis.shadow_caster_count = synopsis
        .shadow_caster_count
        .saturating_sub(contribution.shadow_caster);
    synopsis.supported_shadow_caster_count = synopsis
        .supported_shadow_caster_count
        .saturating_sub(contribution.supported_shadow_caster);
    synopsis.transform_changed_count = synopsis
        .transform_changed_count
        .saturating_sub(contribution.transform_changed);
    synopsis.indexed_primitive_count = synopsis
        .indexed_primitive_count
        .saturating_sub(contribution.indexed_primitives);
    synopsis.topology_fingerprint ^= contribution.fingerprint;
}

fn scene_rebuild_reason(reasons: SceneFullRebuildReasons) -> RenderWorldFullRebuildReason {
    if reasons.contains(SceneFullRebuildReasons::CURSOR_OR_IDENTITY_DRIFT) {
        RenderWorldFullRebuildReason::SceneCursorOrIdentityDrift
    } else if reasons.contains(SceneFullRebuildReasons::OVERFLOW) {
        RenderWorldFullRebuildReason::SceneOverflow
    } else if reasons.contains(SceneFullRebuildReasons::UNTRACKED_HIERARCHY) {
        RenderWorldFullRebuildReason::SceneUntrackedHierarchy
    } else if reasons.contains(SceneFullRebuildReasons::UNTRACKED_STORAGE) {
        RenderWorldFullRebuildReason::SceneUntrackedStorage
    } else if reasons.contains(SceneFullRebuildReasons::EXPLICIT) {
        RenderWorldFullRebuildReason::SceneExplicit
    } else if reasons.contains(SceneFullRebuildReasons::INITIAL) {
        RenderWorldFullRebuildReason::SceneInitial
    } else {
        // `requires_full_rebuild` with no typed cause is itself authority
        // drift; preserve a fail-closed, observable classification.
        RenderWorldFullRebuildReason::SceneCursorOrIdentityDrift
    }
}

fn conservative_shadow_bounds(item: &ExtractedRenderItem) -> Option<BoundingBox> {
    if !item.cast_shadows || !item.world_aabb.is_finite() {
        return None;
    }
    let center = item.world_aabb.center();
    let radius = item.world_aabb.size().length() * 0.5;
    Some(BoundingBox {
        min: center - glam::Vec3::splat(radius),
        max: center + glam::Vec3::splat(radius),
    })
}

fn shadow_bounds_eq(a: Option<BoundingBox>, b: Option<BoundingBox>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.min == b.min && a.max == b.max,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};
    use myth_assets::{AssetServer, GeometryHandle, MaterialHandle};
    use myth_resources::BoundingBox;
    use myth_resources::RenderableMaterialTrait;
    use myth_resources::material::{AlphaMode, Material};
    use myth_scene::NodeHandle;
    use slotmap::KeyData;

    use super::{
        DORMANT_RENDER_WORLD_TTL_SECONDS, PersistentRenderWorld, RenderWorldChangeFlags,
        RenderWorldConsumerCursor, RenderWorldDomain, RenderWorldFullRebuildReason,
        RenderWorldSourceUpdate, capture_material_authority_stamp,
    };
    use crate::graph::extracted::{
        ExtractedRenderItem, ItemShaderFeatures, MaterialClassification,
    };

    fn test_item(node_discriminator: u64) -> ExtractedRenderItem {
        ExtractedRenderItem {
            node_handle: NodeHandle::from(KeyData::from_ffi(node_discriminator * 2 + 1)),
            world_matrix: Mat4::IDENTITY,
            prev_world_matrix: Mat4::IDENTITY,
            geometry: GeometryHandle::from(KeyData::from_ffi(101)),
            material: MaterialHandle::from(KeyData::from_ffi(201)),
            geometry_asset_version: 1,
            geometry_asset_identity: 11,
            geometry_structure_version: 1,
            geometry_layout_version: 1,
            geometry_topology: wgpu::PrimitiveTopology::TriangleList,
            geometry_index_format: Some(wgpu::IndexFormat::Uint32),
            geometry_index_count: 3,
            geometry_has_morph_targets: false,
            geometry_has_vertex_attributes: true,
            geometry_has_instanced_attributes: false,
            material_asset_version: 1,
            material_asset_identity: 21,
            material_version: 1,
            material_gpu_scene_compatible: true,
            material_index: 1,
            material_classification: MaterialClassification::empty(),
            skeleton: None,
            item_variant_flags: 0,
            item_shader_features: ItemShaderFeatures::empty(),
            cast_shadows: false,
            receive_shadows: true,
            world_aabb: BoundingBox {
                min: Vec3::splat(-0.5),
                max: Vec3::splat(0.5),
            },
        }
    }

    #[test]
    fn material_authority_bracket_rejects_mid_snapshot_version_drift() {
        let assets = AssetServer::new();
        let handle = assets.materials.add(Material::new_unlit(glam::Vec4::ONE));
        let guard = assets.materials.read_lock();
        let material = guard.get_loaded(handle).expect("material missing");
        let before = material.data.version();

        let stamp = capture_material_authority_stamp(material, true, || {
            material
                .as_unlit()
                .expect("expected unlit material")
                .set_alpha_mode(AlphaMode::Blend);
        });

        assert!(stamp.is_none());
        assert_ne!(material.data.version(), before);
    }

    #[test]
    fn duplicate_handle_with_conflicting_stamp_is_unstable_not_last_write_wins() {
        let mut domain = RenderWorldDomain::new();
        let first = test_item(1);
        let mut second = test_item(2);
        second.material_version += 1;

        assert!(domain.rebuild_asset_stamps(&[first, second]));
        assert_eq!(domain.material_stamps.len(), 1);
        assert_eq!(
            domain
                .material_stamps
                .values()
                .next()
                .expect("material stamp missing")
                .material_version,
            1,
            "the first coherent stamp remains authoritative until retry"
        );
    }

    #[test]
    fn fixed_state_replacement_releases_last_unique_asset_authority() {
        let mut domain = RenderWorldDomain::new();
        let mut records = vec![test_item(1), test_item(2)];
        domain.rebuild_indices(&records);
        assert!(!domain.rebuild_asset_stamps(&records));

        let old_material = records[0].material;
        let new_material = MaterialHandle::from(KeyData::from_ffi(401));
        records[0].material = new_material;
        records[0].material_asset_identity = 31;
        domain.commit_transform_updates(
            &records,
            &[RenderWorldSourceUpdate {
                node: records[0].node_handle,
                flags: RenderWorldChangeFlags::FIXED_STATE,
                previous_geometry: records[0].geometry,
                previous_material: old_material,
                gpu_facts_valid: false,
            }],
            1,
            false,
        );
        assert_eq!(domain.material_ref_counts.get(&old_material), Some(&1));
        assert_eq!(domain.material_ref_counts.get(&new_material), Some(&1));

        records[1].material = new_material;
        records[1].material_asset_identity = 31;
        domain.commit_transform_updates(
            &records,
            &[RenderWorldSourceUpdate {
                node: records[1].node_handle,
                flags: RenderWorldChangeFlags::FIXED_STATE,
                previous_geometry: records[1].geometry,
                previous_material: old_material,
                gpu_facts_valid: false,
            }],
            1,
            false,
        );
        assert!(!domain.material_ref_counts.contains_key(&old_material));
        assert!(!domain.material_stamps.contains_key(&old_material));
        assert_eq!(domain.material_ref_counts.get(&new_material), Some(&2));
    }

    #[test]
    fn full_rebuild_counts_same_length_node_replacement_exactly() {
        let mut domain = RenderWorldDomain::new();
        let old_records = vec![test_item(1), test_item(2)];
        domain.rebuild_indices(&old_records);
        let new_records = vec![test_item(2), test_item(3)];

        domain.commit_full_rebuild(&new_records, 7, 1, false, 0, 0, false, false, 2);

        assert_eq!(domain.last_stats.items_added, 1);
        assert_eq!(domain.last_stats.items_removed, 1);
        assert_eq!(domain.last_stats.items_examined, 2);
    }

    #[test]
    fn stale_consumer_ack_cannot_clear_a_newer_full_delta() {
        let mut domain = RenderWorldDomain::new();
        domain.scene_id = Some(7);
        domain.revision = 1;
        domain.last_stats.full_rebuild_reason = RenderWorldFullRebuildReason::Cold;
        domain.mark_full_rebuild();
        let stale = RenderWorldConsumerCursor {
            scene_id: 7,
            revision: 1,
        };

        domain.advance_revision();
        assert!(!domain.acknowledge_delta(stale));
        let delta = domain.delta_since(RenderWorldConsumerCursor {
            scene_id: 7,
            revision: 0,
        });
        assert!(delta.requires_full_rebuild);
        assert_eq!(delta.revision, 2);
        assert_eq!(
            delta.full_rebuild_reason,
            RenderWorldFullRebuildReason::Cold
        );
    }

    #[test]
    fn stale_submission_cursor_cannot_settle_newer_world_history() {
        let mut domain = RenderWorldDomain::new();
        let mut records = vec![test_item(1)];
        records[0].world_matrix = Mat4::from_translation(Vec3::X);
        domain.commit_full_rebuild(&records, 7, 1, false, 0, 0, false, false, 1);
        let stale_submission = domain.cursor();

        let previous_geometry = records[0].geometry;
        let previous_material = records[0].material;
        records[0].world_matrix = Mat4::from_translation(Vec3::splat(2.0));
        domain.commit_transform_updates(
            &records,
            &[RenderWorldSourceUpdate {
                node: records[0].node_handle,
                flags: RenderWorldChangeFlags::TRANSFORM,
                previous_geometry,
                previous_material,
                gpu_facts_valid: false,
            }],
            1,
            false,
        );
        assert_ne!(domain.cursor(), stale_submission);

        assert_eq!(
            domain.commit_submitted_history(stale_submission, &mut records),
            None
        );
        assert_eq!(records[0].prev_world_matrix, Mat4::IDENTITY);
        assert_ne!(records[0].prev_world_matrix, records[0].world_matrix);
    }

    fn activate_test_scene(
        world: &mut PersistentRenderWorld,
        records: &mut Vec<ExtractedRenderItem>,
        scene_id: u32,
        elapsed_seconds: f64,
    ) -> bool {
        let changed = world.activate_scene(scene_id, records, elapsed_seconds);
        if changed {
            // Production establishes this identity while committing the
            // extraction transaction immediately after activation.
            world.active.scene_id = Some(scene_id);
        }
        changed
    }

    fn assert_dormant_ttl_at_rate(frames_per_second: u32) {
        let mut world = PersistentRenderWorld::new();
        let mut records = Vec::new();
        assert!(activate_test_scene(&mut world, &mut records, 1, 0.0));
        assert!(activate_test_scene(&mut world, &mut records, 2, 0.0));
        assert!(world.dormant.contains_key(&1));

        let expiry_frame = (DORMANT_RENDER_WORLD_TTL_SECONDS * f64::from(frames_per_second)) as u32;
        for frame in 1..expiry_frame {
            assert!(!activate_test_scene(
                &mut world,
                &mut records,
                2,
                f64::from(frame) / f64::from(frames_per_second),
            ));
            assert!(
                world.dormant.contains_key(&1),
                "dormant scene expired early at {frames_per_second} Hz frame {frame}"
            );
        }

        assert!(!activate_test_scene(
            &mut world,
            &mut records,
            2,
            DORMANT_RENDER_WORLD_TTL_SECONDS,
        ));
        assert!(
            !world.dormant.contains_key(&1),
            "dormant scene did not expire after the same real duration at {frames_per_second} Hz"
        );
    }

    #[test]
    fn dormant_residency_ttl_is_frame_rate_independent() {
        assert_dormant_ttl_at_rate(30);
        assert_dormant_ttl_at_rate(240);
    }

    #[test]
    fn dormant_residency_clock_regression_only_rebases_the_sampler() {
        let mut world = PersistentRenderWorld::new();
        let mut records = Vec::new();
        assert!(activate_test_scene(&mut world, &mut records, 1, 100.0));
        assert!(activate_test_scene(&mut world, &mut records, 2, 100.0));

        assert!(!activate_test_scene(&mut world, &mut records, 2, 101.0));
        assert!(world.dormant.contains_key(&1));
        assert!(!activate_test_scene(&mut world, &mut records, 2, 5.0));
        assert_eq!(world.residency_clock_seconds, 1.0);
        assert!(
            world.dormant.contains_key(&1),
            "clock regression must not make retained domains accidentally ancient"
        );

        assert!(!activate_test_scene(&mut world, &mut records, 2, 5.49));
        assert!(world.dormant.contains_key(&1));
        assert!(!activate_test_scene(&mut world, &mut records, 2, 5.51));
        assert!(!world.dormant.contains_key(&1));
    }
}
