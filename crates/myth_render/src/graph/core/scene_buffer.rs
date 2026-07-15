//! Attempt-scoped logical imports for persistent scene buffers.
//!
//! Graph construction captures only a coherent Copy identity/content stamp and
//! registers a read-only external, non-root buffer. After compilation and DPE,
//! the materialization barrier receives explicit typed source borrows, validates
//! the stamp, uploads only live inputs into `ResourceManager`'s persistent
//! buffer arena, and stores a graph-owned [`Tracked<wgpu::Buffer>`] clone for
//! prepare and execute resolution.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use myth_resources::buffer::{BufferRef, CpuBuffer, CpuBufferStamp, GpuData};
use myth_resources::uniforms::{GpuLightStorage, LightBufferMetadata};
use rustc_hash::FxHashMap;

use crate::core::gpu::{BufferUploadContext, Tracked};
use crate::graph::capacity::{
    CapacityUsage, estimated_hash_map_footprint, shrink_hash_map_by_family_ratio,
    shrink_vec_by_family_ratio, vec_footprint,
};

use super::types::{BufferDesc, BufferNodeId, ResourceRecord};

static NEXT_SCENE_BUFFER_STORE_ID: AtomicU64 = AtomicU64::new(1);

/// Generation-checked index into one [`GraphStorage`](super::graph::GraphStorage)
/// scene-buffer import store.
///
/// Store identity rejects IDs originating from another graph storage, while
/// the attempt generation rejects IDs retained across `clear`/`RenderGraph`
/// boundaries. Resolver APIs fail closed for either case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneBufferImportId {
    store: u64,
    attempt: u64,
    index: u32,
}

impl SceneBufferImportId {
    #[inline]
    #[must_use]
    fn new(store: u64, attempt: u64, index: usize) -> Self {
        Self {
            store,
            attempt,
            index: u32::try_from(index).expect("scene buffer import index exceeds u32"),
        }
    }

    #[inline]
    #[must_use]
    const fn index(self) -> usize {
        self.index as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SceneBufferKind {
    LightMetadata,
    LightStorage,
}

/// Exact CPU identity, descriptor, and content version captured for one import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SceneBufferSnapshot {
    stamp: CpuBufferStamp,
    kind: SceneBufferKind,
}

impl SceneBufferSnapshot {
    #[inline]
    #[must_use]
    pub(crate) const fn desc(self) -> BufferDesc {
        BufferDesc::new(self.stamp.size as u64, self.stamp.usage)
    }
}

/// Narrow type tag used to select one explicit source at the post-DPE barrier.
///
/// The recipe intentionally stores no payload, borrow, raw pointer, callback,
/// or Arc. Add another explicit variant only when a production scene-global
/// buffer is migrated to this path.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SceneBufferRecipe {
    LightMetadata,
    LightStorage,
}

impl SceneBufferRecipe {
    #[inline]
    #[must_use]
    pub(crate) fn snapshot(self, stamp: CpuBufferStamp) -> SceneBufferSnapshot {
        SceneBufferSnapshot {
            stamp,
            kind: match self {
                Self::LightMetadata => SceneBufferKind::LightMetadata,
                Self::LightStorage => SceneBufferKind::LightStorage,
            },
        }
    }
}

/// Explicit frame sources supplied only at the post-DPE materialization barrier.
#[derive(Clone, Copy)]
pub(crate) struct SceneBufferSources<'a> {
    pub(crate) light_metadata: &'a CpuBuffer<LightBufferMetadata>,
    pub(crate) light_storage: &'a CpuBuffer<Vec<GpuLightStorage>>,
}

impl<'a> SceneBufferSources<'a> {
    #[inline]
    fn metadata_snapshot(
        self,
    ) -> (
        CpuBufferStamp,
        myth_resources::buffer::BufferReadGuard<'a, LightBufferMetadata>,
    ) {
        self.light_metadata.read_stamped()
    }

    #[inline]
    fn storage_snapshot(
        self,
    ) -> (
        CpuBufferStamp,
        myth_resources::buffer::BufferReadGuard<'a, Vec<GpuLightStorage>>,
    ) {
        self.light_storage.read_stamped()
    }
}

/// Terminal outcome for the post-DPE scene-buffer materialization barrier.
///
/// Callers must abort the graph attempt on error. Every DPE-live scene import
/// is recorded as unavailable before an error is returned, so accidentally
/// continuing cannot expose a partially published batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SceneBufferMaterializationFailure {
    /// A CPU source no longer matches the graph-construction snapshot.
    SnapshotDrift,
    /// Persistent ResourceManager state conflicts with the exact snapshot.
    ConflictingPhysicalState,
    /// This attempt already recorded an unavailable live input.
    Unavailable,
}

#[derive(Default)]
struct LiveSceneBufferRequirements {
    metadata: Option<SceneBufferSnapshot>,
    storage: Option<SceneBufferSnapshot>,
}

struct SceneBufferImportEntry {
    node: BufferNodeId,
    snapshot: SceneBufferSnapshot,
    recipe: SceneBufferRecipe,
    /// `None` means the live barrier has not visited this entry. A stored None
    /// outcome means materialization deliberately failed closed.
    materialization: OnceLock<Option<Tracked<wgpu::Buffer>>>,
}

impl SceneBufferImportEntry {
    fn new(node: BufferNodeId, snapshot: SceneBufferSnapshot, recipe: SceneBufferRecipe) -> Self {
        Self {
            node,
            snapshot,
            recipe,
            materialization: OnceLock::new(),
        }
    }
}

/// GraphStorage-owned lookup, typed recipes, and destructible frame slots.
pub(crate) struct SceneBufferImportStore {
    store_id: u64,
    attempt: u64,
    entries: Vec<SceneBufferImportEntry>,
    lookup: FxHashMap<SceneBufferSnapshot, SceneBufferImportId>,
}

impl Default for SceneBufferImportStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneBufferImportStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            store_id: NEXT_SCENE_BUFFER_STORE_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .expect("scene buffer store identity space exhausted"),
            attempt: 1,
            entries: Vec::new(),
            lookup: FxHashMap::default(),
        }
    }

    #[must_use]
    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.entries, 16) + estimated_hash_map_footprint(&self.lookup, 16)
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        shrink_vec_by_family_ratio(&mut self.entries, 16, target_bytes, retained_bytes);
        shrink_hash_map_by_family_ratio(&mut self.lookup, 16, target_bytes, retained_bytes);
    }

    pub(crate) fn clear(&mut self) {
        self.lookup.clear();
        // Drops graph-owned WGPU handle clones while retaining the
        // ResourceManager's persistent physical buffers.
        self.entries.clear();
        self.attempt = self
            .attempt
            .checked_add(1)
            .expect("scene buffer attempt generation exhausted");
    }

    #[inline]
    #[must_use]
    pub(crate) fn find(&self, snapshot: SceneBufferSnapshot) -> Option<SceneBufferImportId> {
        self.lookup.get(&snapshot).copied()
    }

    #[must_use]
    pub(crate) fn next_id(&self) -> SceneBufferImportId {
        SceneBufferImportId::new(self.store_id, self.attempt, self.entries.len())
    }

    pub(crate) fn insert(
        &mut self,
        id: SceneBufferImportId,
        node: BufferNodeId,
        snapshot: SceneBufferSnapshot,
        recipe: SceneBufferRecipe,
    ) {
        assert!(
            self.owns(id) && id.index() == self.entries.len(),
            "scene buffer import IDs must be current and densely allocated"
        );
        assert!(
            self.lookup.insert(snapshot, id).is_none(),
            "scene buffer import snapshot registered twice"
        );
        self.entries
            .push(SceneBufferImportEntry::new(node, snapshot, recipe));
    }

    #[inline]
    #[must_use]
    fn owns(&self, id: SceneBufferImportId) -> bool {
        id.store == self.store_id && id.attempt == self.attempt
    }

    #[must_use]
    fn entry(&self, id: SceneBufferImportId) -> Option<&SceneBufferImportEntry> {
        self.owns(id)
            .then(|| self.entries.get(id.index()))
            .flatten()
    }

    #[must_use]
    pub(crate) fn node(&self, id: SceneBufferImportId) -> Option<BufferNodeId> {
        self.entry(id).map(|entry| entry.node)
    }

    #[must_use]
    pub(crate) fn snapshot(&self, id: SceneBufferImportId) -> Option<SceneBufferSnapshot> {
        self.entry(id).map(|entry| entry.snapshot)
    }

    #[must_use]
    pub(crate) fn resolver(&self) -> SceneBufferResolver<'_> {
        SceneBufferResolver {
            store_id: self.store_id,
            attempt: self.attempt,
            entries: &self.entries,
        }
    }

    /// Visit exactly the DPE-live, not-yet-attempted imports.
    pub(crate) fn materialize_live(
        &self,
        resources: &[ResourceRecord],
        sources: SceneBufferSources<'_>,
        uploads: &mut BufferUploadContext<'_>,
    ) -> Result<(), SceneBufferMaterializationFailure> {
        let requirements = match self.live_requirements(resources) {
            Ok(requirements) => requirements,
            Err(error) => {
                self.record_live_unavailable(resources);
                return Err(error);
            }
        };
        if requirements.metadata.is_none() && requirements.storage.is_none() {
            return Ok(());
        }

        // Acquire every relevant guard before validating or uploading either
        // source. The two fixed typed slots keep this hot path allocation-free
        // while preventing a concurrent CPU writer from splitting the batch.
        let metadata = requirements.metadata.map(|_| sources.metadata_snapshot());
        let storage = requirements.storage.map(|_| sources.storage_snapshot());

        let stamps_match = metadata.as_ref().is_none_or(|(actual, _)| {
            SceneBufferRecipe::LightMetadata.snapshot(*actual) == requirements.metadata.unwrap()
        }) && storage.as_ref().is_none_or(|(actual, _)| {
            SceneBufferRecipe::LightStorage.snapshot(*actual) == requirements.storage.unwrap()
        });
        if !stamps_match {
            self.record_live_unavailable(resources);
            return Err(SceneBufferMaterializationFailure::SnapshotDrift);
        }

        // `None` uses BufferRef's borrowed static debug label and therefore
        // avoids a debug-only String allocation in this per-frame batch.
        let metadata_ref = metadata.as_ref().map(|(stamp, _)| {
            BufferRef::with_fixed_id(stamp.id, stamp.size, stamp.usage, stamp.version, None)
        });
        let storage_ref = storage.as_ref().map(|(stamp, _)| {
            BufferRef::with_fixed_id(stamp.id, stamp.size, stamp.usage, stamp.version, None)
        });

        // All source stamps and all persistent physical states are valid before
        // the first mutation. Keep both CPU guards alive until both uploads
        // finish, then publish graph-owned clones as one logical batch.
        let metadata_input = metadata_ref.as_ref().map(|buffer_ref| {
            (
                buffer_ref,
                metadata
                    .as_ref()
                    .expect("metadata guard missing")
                    .1
                    .as_bytes(),
            )
        });
        let storage_input = storage_ref.as_ref().map(|buffer_ref| {
            (
                buffer_ref,
                storage
                    .as_ref()
                    .expect("storage guard missing")
                    .1
                    .as_bytes(),
            )
        });
        let Some([metadata_buffer, storage_buffer]) =
            uploads.materialize_buffer_snapshot_batch([metadata_input, storage_input])
        else {
            self.record_live_unavailable(resources);
            return Err(SceneBufferMaterializationFailure::ConflictingPhysicalState);
        };
        let mut metadata_buffer = metadata_buffer;
        let mut storage_buffer = storage_buffer;

        let resolver = self.resolver();
        for (index, entry) in self.entries.iter().enumerate() {
            if resources[entry.node.index() as usize].first_use == usize::MAX
                || entry.materialization.get().is_some()
            {
                continue;
            }
            let buffer = match entry.recipe {
                SceneBufferRecipe::LightMetadata => metadata_buffer.take(),
                SceneBufferRecipe::LightStorage => storage_buffer.take(),
            };
            let id = SceneBufferImportId::new(self.store_id, self.attempt, index);
            if !resolver.record(id, entry.snapshot, buffer) {
                self.record_live_unavailable(resources);
                return Err(SceneBufferMaterializationFailure::Unavailable);
            }
        }
        Ok(())
    }

    fn live_requirements(
        &self,
        resources: &[ResourceRecord],
    ) -> Result<LiveSceneBufferRequirements, SceneBufferMaterializationFailure> {
        let mut requirements = LiveSceneBufferRequirements::default();
        for entry in &self.entries {
            if resources[entry.node.index() as usize].first_use == usize::MAX {
                continue;
            }
            if let Some(materialization) = entry.materialization.get() {
                if materialization.is_none() {
                    return Err(SceneBufferMaterializationFailure::Unavailable);
                }
                continue;
            }
            let slot = match entry.recipe {
                SceneBufferRecipe::LightMetadata => &mut requirements.metadata,
                SceneBufferRecipe::LightStorage => &mut requirements.storage,
            };
            match *slot {
                None => *slot = Some(entry.snapshot),
                Some(snapshot) if snapshot == entry.snapshot => {}
                Some(_) => return Err(SceneBufferMaterializationFailure::SnapshotDrift),
            }
        }
        Ok(requirements)
    }

    fn record_live_unavailable(&self, resources: &[ResourceRecord]) {
        let resolver = self.resolver();
        for (index, entry) in self.entries.iter().enumerate() {
            if resources[entry.node.index() as usize].first_use == usize::MAX
                || entry.materialization.get().is_some()
            {
                continue;
            }
            let id = SceneBufferImportId::new(self.store_id, self.attempt, index);
            let _ = resolver.record(id, entry.snapshot, None);
        }
    }

    #[cfg(test)]
    fn materialize_live_with(
        &self,
        resources: &[ResourceRecord],
        mut materialize: impl FnMut(
            SceneBufferRecipe,
            SceneBufferSnapshot,
        ) -> Option<Tracked<wgpu::Buffer>>,
    ) -> Result<(), SceneBufferMaterializationFailure> {
        let resolver = self.resolver();
        for (index, entry) in self.entries.iter().enumerate() {
            let id = SceneBufferImportId::new(self.store_id, self.attempt, index);
            if resolver.was_materialized(id)
                || resources[entry.node.index() as usize].first_use == usize::MAX
            {
                continue;
            }
            let buffer = materialize(entry.recipe, entry.snapshot);
            let available = buffer.is_some();
            if !resolver.record(id, entry.snapshot, buffer) || !available {
                self.record_live_unavailable(resources);
                return Err(SceneBufferMaterializationFailure::Unavailable);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn materialize_live_for_test(
        &self,
        resources: &[ResourceRecord],
        materialize: impl FnMut(SceneBufferRecipe, SceneBufferSnapshot) -> Option<Tracked<wgpu::Buffer>>,
    ) -> Result<(), SceneBufferMaterializationFailure> {
        self.materialize_live_with(resources, materialize)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.lookup.is_empty()
    }
}

/// Narrow immutable resolver shared by prepare and execute.
#[derive(Clone, Copy)]
pub(crate) struct SceneBufferResolver<'a> {
    store_id: u64,
    attempt: u64,
    entries: &'a [SceneBufferImportEntry],
}

impl<'a> SceneBufferResolver<'a> {
    #[inline]
    #[must_use]
    fn entry(self, id: SceneBufferImportId) -> Option<&'a SceneBufferImportEntry> {
        (id.store == self.store_id && id.attempt == self.attempt)
            .then(|| self.entries.get(id.index()))
            .flatten()
    }

    /// Record one outcome exactly once. Foreign IDs and mismatched CPU stamps
    /// are rejected without exposing the supplied physical handle.
    pub(crate) fn record(
        self,
        id: SceneBufferImportId,
        stamp: SceneBufferSnapshot,
        buffer: Option<Tracked<wgpu::Buffer>>,
    ) -> bool {
        let Some(entry) = self.entry(id) else {
            return false;
        };
        let current = stamp == entry.snapshot;
        let outcome = current.then_some(buffer).flatten();
        entry.materialization.set(outcome).is_ok() && current
    }

    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(crate) fn was_materialized(self, id: SceneBufferImportId) -> bool {
        self.entry(id)
            .is_some_and(|entry| entry.materialization.get().is_some())
    }

    #[inline]
    #[must_use]
    pub(crate) fn buffer(self, id: SceneBufferImportId) -> Option<&'a Tracked<wgpu::Buffer>> {
        self.entry(id)?.materialization.get()?.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ResourceManager;
    use crate::core::gpu::Tracked;
    use crate::graph::core::graph::RenderGraph;
    use crate::graph::core::{ExecuteContext, FrameArena, GraphStorage, PassNode, PrepareContext};
    use crate::settings::{RendererInitConfig, RendererSettings};
    use crate::test_support::SharedTestContext as WgpuContext;

    struct ConsumerNode;

    impl PassNode<'_> for ConsumerNode {
        fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {}

        fn execute(&self, _ctx: &ExecuteContext<'_>, _encoder: &mut wgpu::CommandEncoder) {}
    }

    fn metadata_buffer() -> CpuBuffer<LightBufferMetadata> {
        CpuBuffer::new(
            LightBufferMetadata::default(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("scene buffer metadata test"),
        )
    }

    fn storage_buffer() -> CpuBuffer<Vec<GpuLightStorage>> {
        CpuBuffer::new(
            vec![GpuLightStorage::default()],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            Some("scene buffer storage test"),
        )
    }

    #[test]
    fn dead_imports_do_not_touch_the_real_resource_manager() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let metadata = metadata_buffer();
        let lights = storage_buffer();
        let metadata_id = metadata.id();
        let lights_id = lights.id();
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            graph.import_scene_light_metadata_buffer("dead metadata", &metadata);
            graph.import_scene_light_storage_buffer("dead lights", &lights);
            graph.compile_topology();

            let mut uploads = BufferUploadContext::new(
                &wgpu.device,
                &wgpu.queue,
                resources.frame_index,
                &mut resources.resources,
                &resources.internal_textures,
            );
            assert_eq!(
                graph.materialize_live_scene_buffers(
                    SceneBufferSources {
                        light_metadata: &metadata,
                        light_storage: &lights,
                    },
                    &mut uploads,
                ),
                Ok(())
            );
        }

        assert!(resources.get_gpu_buffer_by_cpu_id(metadata_id).is_none());
        assert!(resources.get_gpu_buffer_by_cpu_id(lights_id).is_none());
    }

    #[test]
    fn live_import_deduplicates_materializes_once_and_resolves() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let metadata = metadata_buffer();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut calls = 0;

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let first = graph.import_scene_light_metadata_buffer("metadata first", &metadata);
            let duplicate =
                graph.import_scene_light_metadata_buffer("metadata duplicate", &metadata);
            assert_eq!(first, duplicate);
            assert_eq!(graph.storage.scene_buffer_imports.entries.len(), 1);

            graph.add_pass("scene buffer consumer", |builder| {
                builder.read_buffer(first);
                builder.mark_side_effect();
                (ConsumerNode, ())
            });
            graph.compile_topology();

            let id = graph.storage.resources[first.index() as usize]
                .scene_buffer_import_id()
                .expect("lazy buffer backing missing");
            let first_result = graph
                .storage
                .scene_buffer_imports
                .materialize_live_for_test(&graph.storage.resources, |_, _| {
                    calls += 1;
                    let buffer = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("scene buffer materialization test"),
                        size: 16,
                        usage: wgpu::BufferUsages::UNIFORM,
                        mapped_at_creation: false,
                    });
                    Some(Tracked::new(buffer))
                });
            assert_eq!(first_result, Ok(()));
            let second_result = graph
                .storage
                .scene_buffer_imports
                .materialize_live_for_test(&graph.storage.resources, |_, _| {
                    calls += 1;
                    None
                });
            assert_eq!(second_result, Ok(()));

            let resolver = graph.storage.scene_buffer_imports.resolver();
            assert!(resolver.buffer(id).is_some());
            assert!(resolver.was_materialized(id));
        }

        assert_eq!(calls, 1);
    }

    #[test]
    fn drift_fails_the_whole_live_batch_before_any_upload() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let metadata = metadata_buffer();
        let lights = storage_buffer();
        let metadata_id = metadata.id();
        let lights_id = lights.id();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let metadata_node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
            let storage_node = graph.import_scene_light_storage_buffer("lights", &lights);
            graph.add_pass("scene buffer consumer", |builder| {
                builder.read_buffer(metadata_node);
                builder.read_buffer(storage_node);
                builder.mark_side_effect();
                (ConsumerNode, ())
            });
            graph.compile_topology();

            // Invalidate only one member after graph construction. Atomic
            // validation must prevent the still-current member from uploading.
            drop(metadata.write());
            let mut uploads = BufferUploadContext::new(
                &wgpu.device,
                &wgpu.queue,
                resources.frame_index,
                &mut resources.resources,
                &resources.internal_textures,
            );
            let result = graph.materialize_live_scene_buffers(
                SceneBufferSources {
                    light_metadata: &metadata,
                    light_storage: &lights,
                },
                &mut uploads,
            );
            assert_eq!(
                result,
                Err(SceneBufferMaterializationFailure::SnapshotDrift)
            );

            let resolver = graph.storage.scene_buffer_imports.resolver();
            for node in [metadata_node, storage_node] {
                let id = graph.storage.resources[node.index() as usize]
                    .scene_buffer_import_id()
                    .expect("lazy buffer backing missing");
                assert!(resolver.was_materialized(id));
                assert!(resolver.buffer(id).is_none());
            }
        }

        assert!(resources.get_gpu_buffer_by_cpu_id(metadata_id).is_none());
        assert!(resources.get_gpu_buffer_by_cpu_id(lights_id).is_none());

        // A failed attempt owns only its unavailable slots. Rebuilding the
        // graph from the current stamps must retry successfully without stale
        // OnceLock state leaking across attempts.
        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let metadata_node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
            let storage_node = graph.import_scene_light_storage_buffer("lights", &lights);
            graph.add_pass("scene buffer retry consumer", |builder| {
                builder.read_buffer(metadata_node);
                builder.read_buffer(storage_node);
                builder.mark_side_effect();
                (ConsumerNode, ())
            });
            graph.compile_topology();

            let mut uploads = BufferUploadContext::new(
                &wgpu.device,
                &wgpu.queue,
                resources.frame_index,
                &mut resources.resources,
                &resources.internal_textures,
            );
            assert_eq!(
                graph.materialize_live_scene_buffers(
                    SceneBufferSources {
                        light_metadata: &metadata,
                        light_storage: &lights,
                    },
                    &mut uploads,
                ),
                Ok(())
            );

            let resolver = graph.storage.scene_buffer_imports.resolver();
            for node in [metadata_node, storage_node] {
                let id = graph.storage.resources[node.index() as usize]
                    .scene_buffer_import_id()
                    .expect("retry lazy buffer backing missing");
                assert!(resolver.buffer(id).is_some());
            }
        }

        let metadata_gpu = resources
            .get_gpu_buffer_by_cpu_id(metadata_id)
            .expect("retry metadata upload missing");
        assert_eq!(metadata_gpu.last_uploaded_version, metadata.stamp().version);
        let lights_gpu = resources
            .get_gpu_buffer_by_cpu_id(lights_id)
            .expect("retry storage upload missing");
        assert_eq!(lights_gpu.last_uploaded_version, lights.stamp().version);
    }

    #[test]
    fn equal_version_undersized_physical_state_fails_before_batch_upload() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let metadata = metadata_buffer();
        let lights = storage_buffer();
        let metadata_id = metadata.id();
        let light_stamp = lights.stamp();
        assert!(light_stamp.size > 16, "test source must exceed seed buffer");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);

        // Seed a same-ID, same-version allocation that is too small for the
        // later graph snapshot. Equal versions cause write_buffer to skip, so
        // preflight must reject this state before uploading the other member.
        let seed = BufferRef::with_fixed_id(
            light_stamp.id,
            16,
            light_stamp.usage,
            light_stamp.version,
            None,
        );
        resources.ensure_buffer_ref(&seed, &[0; 16]);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let metadata_node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
            let storage_node = graph.import_scene_light_storage_buffer("lights", &lights);
            graph.add_pass("scene buffer consumer", |builder| {
                builder.read_buffer(metadata_node);
                builder.read_buffer(storage_node);
                builder.mark_side_effect();
                (ConsumerNode, ())
            });
            graph.compile_topology();

            let mut uploads = BufferUploadContext::new(
                &wgpu.device,
                &wgpu.queue,
                resources.frame_index,
                &mut resources.resources,
                &resources.internal_textures,
            );
            let result = graph.materialize_live_scene_buffers(
                SceneBufferSources {
                    light_metadata: &metadata,
                    light_storage: &lights,
                },
                &mut uploads,
            );
            assert_eq!(
                result,
                Err(SceneBufferMaterializationFailure::ConflictingPhysicalState)
            );
        }

        assert!(resources.get_gpu_buffer_by_cpu_id(metadata_id).is_none());
        let seeded = resources
            .get_gpu_buffer_by_cpu_id(light_stamp.id)
            .expect("seeded physical buffer missing");
        assert_eq!(seeded.size, 16);
        assert_eq!(seeded.last_uploaded_version, light_stamp.version);
    }

    #[test]
    fn newer_physical_version_fails_before_companion_upload() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let metadata = metadata_buffer();
        let lights = storage_buffer();
        let metadata_id = metadata.id();
        let light_stamp = lights.stamp();
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);

        let newer = BufferRef::with_fixed_id(
            light_stamp.id,
            light_stamp.size,
            light_stamp.usage,
            light_stamp.version + 1,
            None,
        );
        resources.ensure_buffer_ref(&newer, lights.read().as_bytes());

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let metadata_node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
            let storage_node = graph.import_scene_light_storage_buffer("lights", &lights);
            graph.add_pass("scene buffer consumer", |builder| {
                builder.read_buffer(metadata_node);
                builder.read_buffer(storage_node);
                builder.mark_side_effect();
                (ConsumerNode, ())
            });
            graph.compile_topology();

            let mut uploads = BufferUploadContext::new(
                &wgpu.device,
                &wgpu.queue,
                resources.frame_index,
                &mut resources.resources,
                &resources.internal_textures,
            );
            assert_eq!(
                graph.materialize_live_scene_buffers(
                    SceneBufferSources {
                        light_metadata: &metadata,
                        light_storage: &lights,
                    },
                    &mut uploads,
                ),
                Err(SceneBufferMaterializationFailure::ConflictingPhysicalState)
            );
        }

        assert!(resources.get_gpu_buffer_by_cpu_id(metadata_id).is_none());
        assert_eq!(
            resources
                .get_gpu_buffer_by_cpu_id(light_stamp.id)
                .expect("newer physical state missing")
                .last_uploaded_version,
            light_stamp.version + 1
        );
    }

    #[test]
    fn clear_and_graph_drop_release_attempt_slots_and_stale_ids() {
        let metadata = metadata_buffer();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        let stale_id = {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
            let id = graph.storage.resources[node.index() as usize]
                .scene_buffer_import_id()
                .expect("lazy buffer backing missing");
            assert!(!graph.storage.scene_buffer_imports.is_empty());
            id
        };

        assert!(storage.scene_buffer_imports.is_empty());
        assert!(
            storage
                .scene_buffer_imports
                .resolver()
                .buffer(stale_id)
                .is_none()
        );

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
            let old_attempt = graph.storage.resources[node.index() as usize]
                .scene_buffer_import_id()
                .expect("lazy buffer backing missing");
            graph.storage.clear();
            assert!(graph.storage.scene_buffer_imports.is_empty());
            assert!(
                graph
                    .storage
                    .scene_buffer_imports
                    .resolver()
                    .buffer(old_attempt)
                    .is_none()
            );
        }
        assert!(
            storage
                .scene_buffer_imports
                .resolver()
                .buffer(stale_id)
                .is_none()
        );
    }

    #[test]
    fn foreign_ids_and_mismatched_stamps_fail_closed() {
        let metadata = metadata_buffer();
        let lights = storage_buffer();
        let mut first_storage = GraphStorage::new();
        let mut second_storage = GraphStorage::new();
        let first_arena = FrameArena::new();
        let second_arena = FrameArena::new();

        let mut first_graph = RenderGraph::new(&mut first_storage, &first_arena);
        let first_node = first_graph.import_scene_light_metadata_buffer("metadata", &metadata);
        let first_id = first_graph.storage.resources[first_node.index() as usize]
            .scene_buffer_import_id()
            .expect("lazy buffer backing missing");
        let expected = first_graph
            .storage
            .scene_buffer_imports
            .snapshot(first_id)
            .expect("snapshot missing");

        let mut second_graph = RenderGraph::new(&mut second_storage, &second_arena);
        let second_node = second_graph.import_scene_light_storage_buffer("lights", &lights);
        let foreign_id = second_graph.storage.resources[second_node.index() as usize]
            .scene_buffer_import_id()
            .expect("lazy buffer backing missing");

        let resolver = first_graph.storage.scene_buffer_imports.resolver();
        assert!(!resolver.record(foreign_id, expected, None));
        assert!(!resolver.was_materialized(foreign_id));

        let stale = SceneBufferSnapshot {
            stamp: CpuBufferStamp {
                version: expected.stamp.version.wrapping_add(1),
                ..expected.stamp
            },
            ..expected
        };
        assert!(!resolver.record(first_id, stale, None));
        assert!(resolver.was_materialized(first_id));
        assert!(resolver.buffer(first_id).is_none());
    }
}
