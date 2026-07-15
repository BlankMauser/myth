//! Serial reference pipeline for post-DPE scene-source materialization.
//!
//! This store makes the source half of scene submission an explicit four-stage
//! transaction:
//!
//! 1. [`SceneSourcePrepareStore::select_live_sources`] builds the exact,
//!    retained DPE-live source union without per-frame allocation.
//! 2. [`SceneSourcePrepareStore::gather`] derives CPU-only, Copy recipes from
//!    exact graph-attempt snapshots.
//! 3. [`SceneSourcePrepareStore::plan`] assigns stable owners to shared
//!    skeleton uploads without depending on hash-table iteration order.
//! 4. [`SceneSourcePrepareStore::materialize`] is the sole mutable GPU-resource
//!    owner. It allocates model slots in source order, performs one model-buffer
//!    flush, uploads unique skeletons, and prepares object BindGroups.
//! 5. [`SceneSourcePrepareStore::finalize`] publishes compact prepared-source
//!    IDs and graph-attempt physical owners only after a final drift check.
//!
//! The implementation is intentionally serial. Its typed recipes and phase
//! barriers are the reference contract for later worker-local Gather jobs;
//! no concurrent cache access or universal coordinator lock is introduced.

use std::num::NonZeroU32;

use rustc_hash::FxHashSet;
#[cfg(test)]
use slotmap::{Key, KeyData};

use crate::core::ResourceManager;
use myth_resources::uniforms::{DynamicModelUniforms, Mat3Uniform};
use myth_scene::{NodeHandle, Scene, SkeletonKey};

use super::asset_frame::{GeometryFrameId, GeometryFrameResolver, ObjectInterfaceStore};
use super::capacity::{
    CapacityUsage, estimated_hash_set_footprint, shrink_hash_set_by_family_ratio,
    shrink_vec_by_family_ratio, vec_footprint,
};
use super::extracted::ExtractedScene;
use super::frame::{
    DrawPlanKey, DrawPlanStore, ExtractedSourceId, FrameSourceId, PreparedDrawSource,
    PreparedDrawStore, RenderLists,
};
use super::material_plan::{MaterialFrameId, MaterialPlanResolver};

const SCENE_SOURCE_RECIPE_BASELINE: usize = 512;
const SCENE_SOURCE_SKELETON_BASELINE: usize = 32;

/// Dense graph-attempt identity of one gathered source recipe.
///
/// One-based storage preserves the `Option<Id>` niche without widening source
/// joins that may use this ID later.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SceneSourceRecipeId(NonZeroU32);

impl SceneSourceRecipeId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("one graph attempt cannot gather more than u32::MAX - 1 scene sources");
        Self(raw)
    }

    #[inline]
    const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

/// Pure CPU recipe for one source selected by the DPE-live submission union.
///
/// It owns no Asset `Arc`, WGPU handle, generated layout, or WGSL string. The
/// exact owners remain in the Geometry/Material frame stores and are addressed
/// only through Copy graph-attempt IDs.
#[derive(Clone, Copy)]
pub(crate) struct SceneSourceRecipe {
    id: SceneSourceRecipeId,
    frame_source: FrameSourceId,
    extracted_source: ExtractedSourceId,
    node_handle: NodeHandle,
    geometry: GeometryFrameId,
    material: MaterialFrameId,
    skeleton: Option<SkeletonKey>,
    model_uniform: DynamicModelUniforms,
}

/// Model-arena result produced in the same stable order as gathered recipes.
#[derive(Debug, Clone, Copy)]
struct AllocatedSceneSource {
    recipe: SceneSourceRecipeId,
    dynamic_offset: u32,
}

/// Physical object result retained without cloning the cached BindGroup.
#[derive(Debug, Clone, Copy)]
struct MaterializedSceneSource {
    recipe: SceneSourceRecipeId,
    dynamic_offset: u32,
    object_bind_group_id: u64,
    commits_morph_history: bool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum SceneSourcePreparePhase {
    #[default]
    Empty,
    SourcesSelected,
    Gathered,
    Planned,
    Materialized,
    Finalized,
}

/// Retained graph-attempt store for source Gather -> Plan -> Materialize ->
/// Finalize.
pub(crate) struct SceneSourcePrepareStore {
    /// Exact sorted DPE-live source union. Membership is generation-stamped so
    /// repeated Opaque/Prepass/shadow consumers never inflate this retained
    /// vector and warm clear does not touch the high-water stamp domain.
    live_sources: Vec<FrameSourceId>,
    live_source_stamps: Vec<u32>,
    live_source_generation: u32,
    live_source_active_domain_len: usize,
    /// GPU fixed-state representatives that were absent from every live
    /// CpuDirect plan. Geometry/Material preparation includes them, while the
    /// per-object model allocator and dynamic Group 2 path skip them.
    gpu_only_sources: Vec<FrameSourceId>,
    recipes: Vec<SceneSourceRecipe>,
    source_domain_len: usize,
    skeleton_lookup: FxHashSet<SkeletonKey>,
    skeleton_plan: Vec<SkeletonKey>,
    allocations: Vec<AllocatedSceneSource>,
    materialized: Vec<MaterializedSceneSource>,
    phase: SceneSourcePreparePhase,
}

impl Default for SceneSourcePrepareStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneSourcePrepareStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            live_sources: Vec::with_capacity(SCENE_SOURCE_RECIPE_BASELINE),
            live_source_stamps: Vec::with_capacity(SCENE_SOURCE_RECIPE_BASELINE),
            live_source_generation: 1,
            live_source_active_domain_len: 0,
            gpu_only_sources: Vec::new(),
            recipes: Vec::with_capacity(SCENE_SOURCE_RECIPE_BASELINE),
            source_domain_len: 0,
            skeleton_lookup: FxHashSet::default(),
            skeleton_plan: Vec::with_capacity(SCENE_SOURCE_SKELETON_BASELINE),
            allocations: Vec::with_capacity(SCENE_SOURCE_RECIPE_BASELINE),
            materialized: Vec::with_capacity(SCENE_SOURCE_RECIPE_BASELINE),
            phase: SceneSourcePreparePhase::Empty,
        }
    }

    /// Build one exact source union from DPE-live plan keys.
    ///
    /// Membership work is O(total intents visited), output sorting is
    /// O(unique-live log unique-live), and no allocation occurs once the live
    /// shape reaches its retained high-water mark. In particular, a dead graph
    /// never sizes scratch to the pre-DPE frame-source domain.
    #[cfg_attr(
        not(any(test, feature = "rdg_bench_support")),
        allow(dead_code, reason = "CPU-only harness adapter")
    )]
    pub(crate) fn select_live_sources(
        &mut self,
        render_lists: &RenderLists,
        live_keys: impl IntoIterator<Item = DrawPlanKey>,
    ) {
        self.select_live_sources_with_gpu(
            render_lists,
            live_keys,
            std::iter::empty::<FrameSourceId>(),
        );
    }

    pub(crate) fn select_live_sources_with_gpu(
        &mut self,
        render_lists: &RenderLists,
        live_keys: impl IntoIterator<Item = DrawPlanKey>,
        gpu_representatives: impl IntoIterator<Item = FrameSourceId>,
    ) {
        assert_eq!(
            self.phase,
            SceneSourcePreparePhase::Empty,
            "DPE-live sources may be selected only once per graph attempt"
        );
        debug_assert!(self.live_sources.is_empty());
        debug_assert_eq!(self.live_source_active_domain_len, 0);

        self.source_domain_len = render_lists.frame_source_count();
        for key in live_keys {
            render_lists.for_each_source_for_key(key, |source| self.mark_live_source(source));
        }
        for source in gpu_representatives {
            let already_direct = source.index() < self.live_source_stamps.len()
                && self.live_source_stamps[source.index()] == self.live_source_generation;
            self.mark_live_source(source);
            if !already_direct {
                self.gpu_only_sources.push(source);
            }
        }
        self.live_sources.sort_unstable();
        self.gpu_only_sources.sort_unstable();
        self.phase = SceneSourcePreparePhase::SourcesSelected;
    }

    fn mark_live_source(&mut self, source: FrameSourceId) {
        let required = source.index() + 1;
        if self.live_source_stamps.len() < required {
            self.live_source_stamps.resize(required, 0);
        }
        self.live_source_active_domain_len = self.live_source_active_domain_len.max(required);
        if self.live_source_stamps[source.index()] == self.live_source_generation {
            return;
        }
        self.live_source_stamps[source.index()] = self.live_source_generation;
        self.live_sources.push(source);
    }

    #[inline]
    #[must_use]
    pub(crate) fn live_sources(&self) -> &[FrameSourceId] {
        assert_eq!(
            self.phase,
            SceneSourcePreparePhase::SourcesSelected,
            "DPE-live source union is available only between selection and Gather"
        );
        &self.live_sources
    }

    /// Gather one compact recipe for every source that still has exact
    /// Geometry and Material inputs. This phase performs no ResourceManager,
    /// Device, Queue, cache, or WGPU operation.
    pub(crate) fn gather(
        &mut self,
        extracted_scene: &ExtractedScene,
        render_lists: &RenderLists,
        geometry_frames: GeometryFrameResolver<'_>,
        material_plans: MaterialPlanResolver<'_>,
    ) {
        assert_eq!(
            self.phase,
            SceneSourcePreparePhase::SourcesSelected,
            "scene-source Gather must follow DPE-live source selection"
        );
        debug_assert!(self.recipes.is_empty());
        debug_assert!(self.skeleton_plan.is_empty());
        debug_assert!(self.allocations.is_empty());
        debug_assert!(self.materialized.is_empty());

        self.recipes.reserve(self.live_sources.len());
        let mut previous = None;
        let mut geometry_cursor = geometry_frames.monotonic_cursor();
        for live_index in 0..self.live_sources.len() {
            let frame_source = self.live_sources[live_index];
            if self.gpu_only_sources.binary_search(&frame_source).is_ok() {
                continue;
            }
            assert!(
                previous.is_none_or(|previous| previous < frame_source),
                "live frame source IDs must be sorted and deduplicated"
            );
            previous = Some(frame_source);

            let extracted_source = render_lists
                .extracted_source(frame_source)
                .expect("live frame source does not belong to these render lists");
            let item = extracted_scene
                .render_items
                .get(extracted_source.index())
                .expect("live frame source exceeds the extracted item table");
            let Some(geometry) =
                geometry_frames.geometry_id_monotonic(&mut geometry_cursor, frame_source, item)
            else {
                continue;
            };
            let Some(material) = material_plans.prepared_material(frame_source, item) else {
                continue;
            };
            if geometry_frames.physical(geometry).is_none()
                || !geometry_frames.pipeline_state_is_current(geometry)
                || !material.pipeline_state_is_current()
            {
                continue;
            }

            let world_matrix_inverse = item.world_matrix.inverse();
            let model_uniform = DynamicModelUniforms {
                world_matrix: item.world_matrix,
                world_matrix_inverse,
                normal_matrix: Mat3Uniform::from_mat4(world_matrix_inverse.transpose()),
                previous_world_matrix: item.prev_world_matrix,
                ..Default::default()
            };
            let id = SceneSourceRecipeId::from_index(self.recipes.len());
            self.recipes.push(SceneSourceRecipe {
                id,
                frame_source,
                extracted_source,
                node_handle: item.node_handle,
                geometry,
                material: material.id(),
                skeleton: item.skeleton,
                model_uniform,
            });
        }

        self.phase = SceneSourcePreparePhase::Gathered;
    }

    /// Stable first-semantic-occurrence planning for source-shared work.
    ///
    /// Hashing is used only for membership. Iteration order never determines
    /// the owner or materialization order.
    pub(crate) fn plan(&mut self) {
        assert_eq!(
            self.phase,
            SceneSourcePreparePhase::Gathered,
            "scene-source planning must follow Gather"
        );
        debug_assert!(self.skeleton_lookup.is_empty());
        debug_assert!(self.skeleton_plan.is_empty());

        self.skeleton_lookup.reserve(self.recipes.len().min(64));
        for recipe in &self.recipes {
            if let Some(skeleton) = recipe.skeleton
                && self.skeleton_lookup.insert(skeleton)
            {
                self.skeleton_plan.push(skeleton);
            }
        }
        self.phase = SceneSourcePreparePhase::Planned;
    }

    /// Materialize all source-local GPU state through one serial typed owner.
    ///
    /// Model slots are allocated in recipe order and flushed once before any
    /// object BindGroup observes the model buffer. Skeletons are uploaded once
    /// in their stable planned order. Morph packing remains here, rather than
    /// Gather, because it mutates submit-committed Scene history; it must become
    /// an immutable/COW recipe before this phase is parallelized.
    pub(crate) fn materialize(
        &mut self,
        scene: &mut Scene,
        resource_manager: &mut ResourceManager,
        geometry_frames: GeometryFrameResolver<'_>,
        material_plans: MaterialPlanResolver<'_>,
    ) {
        assert_eq!(
            self.phase,
            SceneSourcePreparePhase::Planned,
            "scene-source materialization must follow Plan"
        );
        debug_assert!(self.allocations.is_empty());
        debug_assert!(self.materialized.is_empty());
        self.allocations.reserve(self.recipes.len());
        self.materialized.reserve(self.recipes.len());

        for recipe in &self.recipes {
            // Gather already admitted only exact Geometry/Material frame
            // records. Reserving a model slot is harmless if a source drifts
            // afterward: the mutable Scene check below gates object work and
            // the pipeline epoch gate remains the sole publication point.
            let dynamic_offset = resource_manager.allocate_model_uniform(recipe.model_uniform);
            self.allocations.push(AllocatedSceneSource {
                recipe: recipe.id,
                dynamic_offset,
            });
        }

        if !self.allocations.is_empty() {
            // The single flush is a correctness barrier: object BindGroups
            // cannot retain a model buffer that is replaced by later growth.
            resource_manager.flush_model_buffers();
        }

        for &skeleton_key in &self.skeleton_plan {
            if let Some(skeleton) = scene.skeleton_pool.get(skeleton_key) {
                resource_manager.prepare_skeleton(skeleton);
            }
        }

        let skeleton_pool = &scene.skeleton_pool;
        let meshes = &mut scene.meshes;
        for allocation in &self.allocations {
            let recipe = &self.recipes[allocation.recipe.index()];
            debug_assert_eq!(recipe.id, allocation.recipe);
            let Some(geometry) = geometry_frames.geometry(recipe.geometry) else {
                continue;
            };
            let Some(material) = material_plans.material(recipe.material) else {
                continue;
            };
            let skeleton = recipe.skeleton.and_then(|key| skeleton_pool.get(key));
            let Some(mesh) = meshes.get_mut(recipe.node_handle) else {
                continue;
            };
            if (mesh.geometry, mesh.material) != (geometry.key.handle, material.handle()) {
                continue;
            }

            let has_morph_targets = geometry.asset.has_morph_targets();
            if skeleton.is_some() || has_morph_targets {
                mesh.prepare_morph_uniforms();
            }
            let commits_morph_history =
                has_morph_targets && !mesh.morph_target_influences().is_empty();
            let Some(object_bind_group_id) = resource_manager.prepare_draw_object_exact_geometry(
                mesh,
                skeleton,
                &geometry.asset,
                geometry.key.asset_version,
                geometry.key.asset_identity,
            ) else {
                continue;
            };
            self.materialized.push(MaterializedSceneSource {
                recipe: recipe.id,
                dynamic_offset: allocation.dynamic_offset,
                object_bind_group_id,
                commits_morph_history,
            });
        }

        self.phase = SceneSourcePreparePhase::Materialized;
    }

    /// Publish only still-current physical sources into the existing compact
    /// Direct-backend stores.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finalize(
        &mut self,
        material_plans: MaterialPlanResolver<'_>,
        resource_manager: &ResourceManager,
        object_interfaces: &mut ObjectInterfaceStore,
        prepared_draws: &mut PreparedDrawStore,
        draw_plans: &mut DrawPlanStore,
    ) {
        assert_eq!(
            self.phase,
            SceneSourcePreparePhase::Materialized,
            "scene-source Finalize must follow Materialize"
        );
        prepared_draws.begin_sources(self.recipes.len(), self.source_domain_len);

        for physical in &self.materialized {
            let recipe = &self.recipes[physical.recipe.index()];
            let Some(material) = material_plans.material(recipe.material) else {
                continue;
            };
            let Some(object_bind_group) =
                resource_manager.prepared_object_bind_group(physical.object_bind_group_id)
            else {
                continue;
            };

            let material_interface = material.interface();
            let material_bind_group_slot = draw_plans.intern_bind_group(
                &material_interface.bind_group,
                material_interface.bind_group_id,
            );
            let object_bind_group_slot = draw_plans.intern_bind_group(
                &object_bind_group.bind_group,
                object_bind_group.bind_group_id,
            );
            let object_interface = object_interfaces.intern(
                object_bind_group.binding_interface_id,
                object_bind_group.layout_id,
                &object_bind_group.layout,
                &object_bind_group.binding_wgsl,
            );

            prepared_draws.insert_source(PreparedDrawSource {
                frame_source: recipe.frame_source,
                extracted_source: recipe.extracted_source,
                geometry: recipe.geometry,
                material: recipe.material,
                object_interface,
                material_bind_group: material_bind_group_slot,
                object_bind_group: (object_bind_group_slot, physical.dynamic_offset),
                commits_morph_history: physical.commits_morph_history,
            });
        }

        self.phase = SceneSourcePreparePhase::Finalized;
    }

    /// Release graph-attempt records while retaining warm-path capacity.
    pub(crate) fn clear(&mut self) {
        self.materialized.clear();
        self.allocations.clear();
        self.skeleton_plan.clear();
        self.skeleton_lookup.clear();
        self.recipes.clear();
        self.live_sources.clear();
        self.gpu_only_sources.clear();
        self.source_domain_len = 0;
        self.live_source_active_domain_len = 0;
        self.advance_live_source_generation();
        self.phase = SceneSourcePreparePhase::Empty;
    }

    fn advance_live_source_generation(&mut self) {
        self.live_source_generation = self.live_source_generation.wrapping_add(1);
        if self.live_source_generation == 0 {
            // The only O(high-water) clear occurs once per u32 wrap. Ordinary
            // frames retain both capacity and untouched membership bytes.
            self.live_source_stamps.fill(0);
            self.live_source_generation = 1;
        }
    }

    #[must_use]
    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.live_sources, SCENE_SOURCE_RECIPE_BASELINE)
            + vec_footprint(&self.gpu_only_sources, 0)
            + live_source_join_footprint(
                self.live_source_active_domain_len,
                &self.live_source_stamps,
            )
            + vec_footprint(&self.recipes, SCENE_SOURCE_RECIPE_BASELINE)
            + estimated_hash_set_footprint(&self.skeleton_lookup, SCENE_SOURCE_SKELETON_BASELINE)
            + vec_footprint(&self.skeleton_plan, SCENE_SOURCE_SKELETON_BASELINE)
            + vec_footprint(&self.allocations, SCENE_SOURCE_RECIPE_BASELINE)
            + vec_footprint(&self.materialized, SCENE_SOURCE_RECIPE_BASELINE)
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        assert!(
            self.is_empty(),
            "live scene-source recipes cannot be trimmed"
        );
        shrink_vec_by_family_ratio(
            &mut self.live_sources,
            SCENE_SOURCE_RECIPE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(&mut self.gpu_only_sources, 0, target_bytes, retained_bytes);
        trim_live_source_join(&mut self.live_source_stamps, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(
            &mut self.recipes,
            SCENE_SOURCE_RECIPE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_set_by_family_ratio(
            &mut self.skeleton_lookup,
            SCENE_SOURCE_SKELETON_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.skeleton_plan,
            SCENE_SOURCE_SKELETON_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.allocations,
            SCENE_SOURCE_RECIPE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.materialized,
            SCENE_SOURCE_RECIPE_BASELINE,
            target_bytes,
            retained_bytes,
        );
    }

    #[inline]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.phase == SceneSourcePreparePhase::Empty
            && self.live_sources.is_empty()
            && self.gpu_only_sources.is_empty()
            && self.live_source_active_domain_len == 0
            && self.recipes.is_empty()
            && self.source_domain_len == 0
            && self.skeleton_lookup.is_empty()
            && self.skeleton_plan.is_empty()
            && self.allocations.is_empty()
            && self.materialized.is_empty()
    }

    #[cfg(test)]
    fn push_test_recipe(&mut self, skeleton: Option<SkeletonKey>) -> SceneSourceRecipeId {
        assert!(matches!(
            self.phase,
            SceneSourcePreparePhase::Empty | SceneSourcePreparePhase::Gathered
        ));
        let index = self.recipes.len();
        self.source_domain_len = self.source_domain_len.max(index + 1);
        let id = SceneSourceRecipeId::from_index(index);
        self.recipes.push(SceneSourceRecipe {
            id,
            frame_source: FrameSourceId::from_index(index),
            extracted_source: ExtractedSourceId::from_index(index),
            node_handle: NodeHandle::from(KeyData::from_ffi((index + 1) as u64)),
            geometry: GeometryFrameId::from_index(index),
            material: MaterialFrameId::from_index(index),
            skeleton,
            model_uniform: DynamicModelUniforms::default(),
        });
        self.phase = SceneSourcePreparePhase::Gathered;
        id
    }
}

fn live_source_join_footprint(active_slots: usize, stamps: &Vec<u32>) -> CapacityUsage {
    let slot_bytes = std::mem::size_of::<u32>();
    CapacityUsage::new(
        active_slots.saturating_mul(slot_bytes),
        stamps.capacity().saturating_mul(slot_bytes),
        SCENE_SOURCE_RECIPE_BASELINE.saturating_mul(slot_bytes),
    )
}

fn trim_live_source_join(stamps: &mut Vec<u32>, target_bytes: usize, retained_bytes: usize) {
    if retained_bytes == 0 || target_bytes >= retained_bytes {
        return;
    }
    // The initialized stamp length is deliberately retained across ordinary
    // clears, but trimming follows the allocation high-water mark and keeps
    // the same baseline floor as every other capacity family.
    let target_slots = (((stamps.capacity() as u128) * (target_bytes as u128))
        .div_ceil(retained_bytes as u128)
        .min(usize::MAX as u128) as usize)
        .max(SCENE_SOURCE_RECIPE_BASELINE)
        .min(stamps.capacity());
    stamps.truncate(target_slots);
    stamps.shrink_to(target_slots);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graph::frame::RenderIntent;

    fn skeleton(index: u64) -> SkeletonKey {
        SkeletonKey::from(KeyData::from_ffi(index))
    }

    fn render_intent(source: FrameSourceId) -> RenderIntent {
        RenderIntent {
            source,
            geometry_handle: myth_resources::GeometryHandle::null(),
            material_handle: myth_resources::MaterialHandle::null(),
            material_index: 0,
            distance_sq: 0.0,
        }
    }

    #[test]
    fn live_source_union_is_unique_retained_and_dead_domain_independent() {
        let mut lists = RenderLists::new();
        let mut last = FrameSourceId::from_index(0);
        for extracted in 0..100_000 {
            last = lists.intern_frame_source(ExtractedSourceId::from_index(extracted));
        }
        let early = FrameSourceId::from_index(3);
        lists.opaque = vec![
            render_intent(last),
            render_intent(early),
            render_intent(last),
        ];

        let mut plans = DrawPlanStore::new();
        let opaque = plans.reserve_plan(DrawPlanKey::Opaque);
        let prepass = plans.reserve_plan(DrawPlanKey::Prepass {
            needs_normal: true,
            needs_feature_id: false,
            needs_velocity: false,
        });
        plans.mark_live(opaque);
        plans.mark_live(prepass);

        let mut store = SceneSourcePrepareStore::new();
        store.select_live_sources(&lists, plans.live_keys());
        assert_eq!(store.live_sources(), [early, last]);
        assert_eq!(store.live_source_stamps.len(), last.index() + 1);
        let selected = store.capacity_usage();
        assert!(selected.active_bytes > 0);

        store.clear();
        let retained = store.capacity_usage();
        assert_eq!(retained.active_bytes, 0);
        assert_eq!(retained.retained_bytes, selected.retained_bytes);

        // A later all-dead graph keeps the retained high-water storage but
        // neither scans nor marks the 100k pre-DPE source domain.
        store.select_live_sources(&lists, std::iter::empty());
        assert!(store.live_sources().is_empty());
        assert_eq!(store.live_source_active_domain_len, 0);
        assert_eq!(store.live_source_stamps.len(), last.index() + 1);
        store.clear();

        // Generation wrap is the sole path that clears retained stamp bytes.
        store.live_source_generation = u32::MAX;
        store.select_live_sources(&lists, [DrawPlanKey::Opaque]);
        store.clear();
        assert_eq!(store.live_source_generation, 1);
        assert!(store.live_source_stamps.iter().all(|&stamp| stamp == 0));
    }

    #[test]
    fn multiple_gpu_bin_representatives_join_the_exact_live_source_union() {
        let mut lists = RenderLists::new();
        let direct = lists.append_gpu_opaque_representative(ExtractedSourceId::from_index(7));
        let gpu_only = lists.append_gpu_opaque_representative(ExtractedSourceId::from_index(19));
        assert_eq!(
            lists.append_gpu_opaque_representative(ExtractedSourceId::from_index(7)),
            direct,
            "one semantic bin representative must be idempotent"
        );
        lists.opaque.push(render_intent(direct));

        let mut store = SceneSourcePrepareStore::new();
        store.select_live_sources_with_gpu(
            &lists,
            [DrawPlanKey::Opaque],
            lists.gpu_opaque_representatives(),
        );

        assert_eq!(store.live_sources(), [direct, gpu_only]);
        assert_eq!(store.gpu_only_sources, [gpu_only]);
        assert_eq!(store.source_domain_len, 2);
    }

    #[test]
    fn skeleton_plan_is_stable_and_capacity_is_explicitly_retained_then_trimmed() {
        let first = skeleton(1);
        let second = skeleton(2);
        let mut store = SceneSourcePrepareStore::new();
        store.push_test_recipe(Some(first));
        store.push_test_recipe(Some(second));
        store.push_test_recipe(Some(first));
        store.push_test_recipe(None);

        store.plan();

        assert_eq!(store.skeleton_plan, [first, second]);
        assert_eq!(store.phase, SceneSourcePreparePhase::Planned);

        store.clear();
        for index in 1..=(SCENE_SOURCE_RECIPE_BASELINE * 16) {
            store.push_test_recipe(Some(skeleton(index as u64)));
        }
        store.plan();
        let populated = store.capacity_usage();
        assert!(populated.active_bytes > 0);
        assert!(populated.retained_bytes > populated.baseline_bytes);

        store.clear();
        let retained = store.capacity_usage();
        assert_eq!(retained.active_bytes, 0);
        assert_eq!(retained.retained_bytes, populated.retained_bytes);

        store.trim_capacity(0, retained.retained_bytes);
        let trimmed = store.capacity_usage();
        assert!(trimmed.retained_bytes < retained.retained_bytes);
        assert!(trimmed.retained_bytes >= trimmed.baseline_bytes);
        assert!(store.is_empty());
    }
}
