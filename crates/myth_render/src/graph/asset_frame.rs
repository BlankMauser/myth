//! Graph-attempt ownership for exact geometry and object-interface snapshots.
//!
//! Draw sources retain only compact IDs into these stores. Each store owns one
//! physical/CPU owner per exact key for the whole graph attempt, preventing
//! per-draw `Arc` reference-count traffic while still closing asset ABA gaps.

use std::sync::Arc;

use myth_assets::{AssetServer, GeometryHandle};
use myth_resources::geometry::Geometry;
use myth_resources::shader_defines::ShaderDefines;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::core::ResourceManager;

use super::extracted::{ExtractedRenderItem, ExtractedScene};
use super::frame::{DrawBufferSlot, DrawPlanStore, FrameSourceId, RenderLists};

/// Compact graph-attempt identity of one exact geometry owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct GeometryFrameId(u32);

impl GeometryFrameId {
    #[inline]
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("geometry frame record index exceeds u32::MAX"))
    }

    #[inline]
    const fn index(self) -> usize {
        self.0 as usize
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
    pub(crate) shader_defines: ShaderDefines,
    pub(crate) topology: wgpu::PrimitiveTopology,
    physical: Option<GeometryFramePhysical>,
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

/// Compact source join and unique geometry-owner table for one graph attempt.
pub(crate) struct GeometryFrameStore {
    records: Vec<GeometryFrameRecord>,
    lookup: FxHashMap<GeometryFrameKey, GeometryFrameId>,
    /// The extracted snapshot does not contain the interior geometry epochs.
    /// This secondary lookup lets repeated sources reuse the first coherent
    /// record without another AssetStorage read or `Arc` clone.
    stamp_lookup: FxHashMap<(GeometryHandle, u32, u64), GeometryFrameId>,
    failed_stamps: FxHashSet<(GeometryHandle, u32, u64)>,
    source_geometries: Vec<Option<GeometryFrameId>>,
    initialized: bool,
}

impl GeometryFrameStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            records: Vec::with_capacity(256),
            lookup: FxHashMap::default(),
            stamp_lookup: FxHashMap::default(),
            failed_stamps: FxHashSet::default(),
            source_geometries: Vec::with_capacity(512),
            initialized: false,
        }
    }

    /// Capture the unique geometry owners referenced by the post-DPE source
    /// union. AssetStorage traffic and `Arc` clones are O(unique geometries),
    /// never O(draw sources).
    pub(crate) fn capture_live(
        &mut self,
        assets: &AssetServer,
        extracted_scene: &ExtractedScene,
        render_lists: &RenderLists,
        live_sources: &[FrameSourceId],
    ) {
        assert!(
            !self.initialized,
            "geometry frame store may be captured only once per graph attempt"
        );
        debug_assert!(self.records.is_empty());
        debug_assert!(self.lookup.is_empty());
        debug_assert!(self.stamp_lookup.is_empty());
        debug_assert!(self.failed_stamps.is_empty());
        debug_assert!(self.source_geometries.is_empty());

        self.source_geometries
            .resize(render_lists.frame_source_count(), None);
        self.initialized = true;

        for &source in live_sources {
            let Some(slot) = self.source_geometries.get(source.index()) else {
                continue;
            };
            if slot.is_some() {
                continue;
            }
            let Some(extracted_source) = render_lists.extracted_source(source) else {
                continue;
            };
            let Some(item) = extracted_scene.render_items.get(extracted_source.index()) else {
                continue;
            };
            let stamp = (
                item.geometry,
                item.geometry_asset_version,
                item.geometry_asset_identity,
            );
            if let Some(&id) = self.stamp_lookup.get(&stamp) {
                self.source_geometries[source.index()] = Some(id);
                continue;
            }
            if self.failed_stamps.contains(&stamp) {
                continue;
            }

            let Some((asset, asset_version, asset_identity)) =
                assets.geometries.get_entry_with_identity(item.geometry)
            else {
                self.failed_stamps.insert(stamp);
                continue;
            };
            if (asset_version, asset_identity)
                != (item.geometry_asset_version, item.geometry_asset_identity)
            {
                self.failed_stamps.insert(stamp);
                continue;
            }

            let structure_version = asset.structure_version();
            let layout_version = asset.layout_version();
            let shader_defines = asset.shader_defines().clone();
            let topology = asset.topology;
            // Keep the CPU interface snapshot coherent with its exact owner.
            if (structure_version, layout_version)
                != (asset.structure_version(), asset.layout_version())
                || assets.geometries.get_version_and_identity(item.geometry)
                    != Some((asset_version, asset_identity))
            {
                self.failed_stamps.insert(stamp);
                continue;
            }

            let key = GeometryFrameKey {
                handle: item.geometry,
                asset_version,
                asset_identity,
                structure_version,
                layout_version,
            };
            let id = if let Some(&id) = self.lookup.get(&key) {
                id
            } else {
                let id = GeometryFrameId::from_index(self.records.len());
                self.records.push(GeometryFrameRecord {
                    key,
                    asset,
                    shader_defines,
                    topology,
                    physical: None,
                });
                assert!(self.lookup.insert(key, id).is_none());
                id
            };
            assert!(self.stamp_lookup.insert(stamp, id).is_none());
            self.source_geometries[source.index()] = Some(id);
        }
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
        assert!(self.initialized, "geometry frame store was not captured");
        for index in 0..self.records.len() {
            let id = GeometryFrameId::from_index(index);
            if !self.is_current(assets, id) {
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
            let mut vertex_buffers = SmallVec::with_capacity(gpu.vertex_buffers.len());
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
            self.records[index].physical = Some(GeometryFramePhysical {
                vertex_buffers,
                index_buffer,
                draw_range: gpu.draw_range.clone(),
                instance_range: gpu.instance_range.clone(),
                layout_info: gpu.layout_info.clone(),
                layout_id: gpu.layout_id,
            });
        }
    }

    #[must_use]
    pub(crate) const fn resolver(&self) -> GeometryFrameResolver<'_> {
        GeometryFrameResolver { store: self }
    }

    /// Validate both the exact AssetStorage owner and its pipeline-relevant
    /// interior epochs. This is the fail-closed boundary before publication.
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
        self.source_geometries.clear();
        self.stamp_lookup.clear();
        self.failed_stamps.clear();
        self.lookup.clear();
        self.records.clear();
        self.initialized = false;
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn geometry_count(&self) -> usize {
        self.records.len()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn joined_source_count(&self) -> usize {
        self.source_geometries.iter().flatten().count()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.records.is_empty()
            && self.lookup.is_empty()
            && self.stamp_lookup.is_empty()
            && self.failed_stamps.is_empty()
            && self.source_geometries.is_empty()
            && !self.initialized
    }
}

impl Default for GeometryFrameStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Read-only source/record resolver handed to deferred preparation.
#[derive(Clone, Copy)]
pub(crate) struct GeometryFrameResolver<'a> {
    store: &'a GeometryFrameStore,
}

impl<'a> GeometryFrameResolver<'a> {
    #[must_use]
    pub(crate) fn geometry_id(
        self,
        source: FrameSourceId,
        item: &ExtractedRenderItem,
    ) -> Option<GeometryFrameId> {
        let id = self
            .store
            .source_geometries
            .get(source.index())
            .copied()
            .flatten()?;
        self.store
            .records
            .get(id.index())?
            .key
            .matches_item(item)
            .then_some(id)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ObjectInterfaceKey {
    layout_id: u64,
    binding_wgsl_hash: u64,
}

/// One exact immutable object binding interface.
pub(crate) struct ObjectInterfaceRecord {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) layout_id: u64,
    pub(crate) binding_wgsl: Arc<str>,
    pub(crate) binding_wgsl_hash: u64,
}

/// Unique object binding interfaces retained by one graph attempt.
pub(crate) struct ObjectInterfaceStore {
    records: Vec<ObjectInterfaceRecord>,
    lookup: FxHashMap<ObjectInterfaceKey, ObjectInterfaceId>,
}

impl ObjectInterfaceStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            records: Vec::with_capacity(64),
            lookup: FxHashMap::default(),
        }
    }

    /// Intern by the canonical physical-layout ID and WGSL content hash.
    /// WGPU/Arc owners are cloned only for a newly observed interface.
    pub(crate) fn intern(
        &mut self,
        layout_id: u64,
        binding_wgsl_hash: u64,
        layout: &wgpu::BindGroupLayout,
        binding_wgsl: &Arc<str>,
    ) -> ObjectInterfaceId {
        let key = ObjectInterfaceKey {
            layout_id,
            binding_wgsl_hash,
        };
        if let Some(&id) = self.lookup.get(&key) {
            assert_eq!(
                self.records[id.index()].binding_wgsl.as_ref(),
                binding_wgsl.as_ref(),
                "object interface hash collision for one canonical layout"
            );
            return id;
        }

        let id = ObjectInterfaceId::from_index(self.records.len());
        self.records.push(ObjectInterfaceRecord {
            layout: layout.clone(),
            layout_id,
            binding_wgsl: binding_wgsl.clone(),
            binding_wgsl_hash,
        });
        assert!(self.lookup.insert(key, id).is_none());
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
    use glam::Mat4;
    use myth_assets::{GeometryHandle, MaterialHandle};
    use myth_resources::{BoundingBox, geometry::Geometry};
    use slotmap::Key;

    use crate::graph::frame::ExtractedSourceId;

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
            skeleton: None,
            item_variant_flags: 0,
            item_shader_defines: ShaderDefines::new(),
            item_shader_defines_hash: 0,
            cast_shadows: true,
            receive_shadows: true,
            world_aabb: BoundingBox::default(),
        }
    }

    #[test]
    fn frame_asset_ids_are_copy_sized_handles() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<GeometryFrameId>();
        assert_copy::<ObjectInterfaceId>();
        assert_eq!(std::mem::size_of::<GeometryFrameId>(), 4);
        assert_eq!(std::mem::size_of::<ObjectInterfaceId>(), 4);
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
        store.capture_live(&assets, &extracted, &lists, &[first, second]);

        let resolver = store.resolver();
        let first_id = resolver
            .geometry_id(first, &extracted.render_items[0])
            .expect("first geometry join missing");
        let second_id = resolver
            .geometry_id(second, &extracted.render_items[1])
            .expect("second geometry join missing");
        assert_eq!(first_id, second_id);
        assert_eq!(store.geometry_count(), 1);
        assert_eq!(store.joined_source_count(), 2);
        assert_eq!(
            Arc::strong_count(&resolver.geometry(first_id).unwrap().asset),
            2,
            "AssetStorage plus one frame record must be the only Geometry owners"
        );
        assert!(store.is_current(&assets, first_id));
        assert!(store.pipeline_state_is_current(first_id));

        assets.geometries.update(handle, Geometry::new());
        assert!(!store.is_current(&assets, first_id));
        store.clear();
        assert!(store.is_empty());
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
        store.capture_live(&assets, &extracted, &lists, &live);

        assert_eq!(store.records.len(), 1);
        assert!(
            store.records.capacity() < SOURCE_COUNT / 4,
            "unique Geometry records must retain a unique-asset high-water mark"
        );
        assert!(store.lookup.capacity() < SOURCE_COUNT / 4);
        assert!(store.stamp_lookup.capacity() < SOURCE_COUNT / 4);
    }
}
