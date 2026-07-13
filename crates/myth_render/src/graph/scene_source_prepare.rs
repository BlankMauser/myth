//! Serial reference pipeline for post-DPE scene-source materialization.
//!
//! This store makes the source half of scene submission an explicit four-stage
//! transaction:
//!
//! 1. [`SceneSourcePrepareStore::gather`] derives CPU-only, Copy recipes from
//!    exact graph-attempt snapshots.
//! 2. [`SceneSourcePrepareStore::plan`] assigns stable owners to shared
//!    skeleton uploads without depending on hash-table iteration order.
//! 3. [`SceneSourcePrepareStore::materialize`] is the sole mutable GPU-resource
//!    owner. It allocates model slots in source order, performs one model-buffer
//!    flush, uploads unique skeletons, and prepares object BindGroups.
//! 4. [`SceneSourcePrepareStore::finalize`] publishes compact prepared-source
//!    IDs and graph-attempt physical owners only after a final drift check.
//!
//! The implementation is intentionally serial. Its typed recipes and phase
//! barriers are the reference contract for later worker-local Gather jobs;
//! no concurrent cache access or universal coordinator lock is introduced.

use std::num::NonZeroU32;

use rustc_hash::FxHashSet;
#[cfg(test)]
use slotmap::KeyData;

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
    DrawPlanStore, FrameSourceId, PreparedDrawSource, PreparedDrawStore, RenderLists,
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
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum SceneSourcePreparePhase {
    #[default]
    Empty,
    Gathered,
    Planned,
    Materialized,
    Finalized,
}

/// Retained graph-attempt store for source Gather -> Plan -> Materialize ->
/// Finalize.
pub(crate) struct SceneSourcePrepareStore {
    recipes: Vec<SceneSourceRecipe>,
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
            recipes: Vec::with_capacity(SCENE_SOURCE_RECIPE_BASELINE),
            skeleton_lookup: FxHashSet::default(),
            skeleton_plan: Vec::with_capacity(SCENE_SOURCE_SKELETON_BASELINE),
            allocations: Vec::with_capacity(SCENE_SOURCE_RECIPE_BASELINE),
            materialized: Vec::with_capacity(SCENE_SOURCE_RECIPE_BASELINE),
            phase: SceneSourcePreparePhase::Empty,
        }
    }

    /// Gather one compact recipe for every source that still has exact
    /// Geometry and Material inputs. This phase performs no ResourceManager,
    /// Device, Queue, cache, or WGPU operation.
    pub(crate) fn gather(
        &mut self,
        extracted_scene: &ExtractedScene,
        live_frame_sources: &[FrameSourceId],
        render_lists: &RenderLists,
        geometry_frames: GeometryFrameResolver<'_>,
        material_plans: MaterialPlanResolver<'_>,
    ) {
        assert_eq!(
            self.phase,
            SceneSourcePreparePhase::Empty,
            "scene sources may be gathered only once per graph attempt"
        );
        debug_assert!(self.recipes.is_empty());
        debug_assert!(self.skeleton_plan.is_empty());
        debug_assert!(self.allocations.is_empty());
        debug_assert!(self.materialized.is_empty());

        self.recipes.reserve(live_frame_sources.len());
        let mut previous = None;
        for &frame_source in live_frame_sources {
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
            let Some(geometry) = geometry_frames.geometry_id(frame_source, item) else {
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
            let Some(geometry) = geometry_frames.geometry(recipe.geometry) else {
                continue;
            };
            let Some(material) = material_plans.material(recipe.material) else {
                continue;
            };
            if geometry_frames.physical(recipe.geometry).is_none()
                || !geometry_frames.pipeline_state_is_current(recipe.geometry)
                || !material.pipeline_state_is_current()
                || !scene.meshes.get(recipe.node_handle).is_some_and(|mesh| {
                    (mesh.geometry, mesh.material) == (geometry.key.handle, material.handle())
                })
            {
                continue;
            }
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
            if geometry_frames.physical(recipe.geometry).is_none()
                || !geometry_frames.pipeline_state_is_current(recipe.geometry)
                || !material.pipeline_state_is_current()
            {
                continue;
            }
            let skeleton = recipe.skeleton.and_then(|key| skeleton_pool.get(key));
            let Some(mesh) = meshes.get_mut(recipe.node_handle) else {
                continue;
            };
            if (mesh.geometry, mesh.material) != (geometry.key.handle, material.handle()) {
                continue;
            }

            mesh.prepare_morph_uniforms();
            let Some(object_bind_group_id) = resource_manager.prepare_draw_object_exact_geometry(
                mesh,
                skeleton,
                &geometry.asset,
                geometry.key.asset_version,
                geometry.key.asset_identity,
            ) else {
                continue;
            };
            if !geometry_frames.pipeline_state_is_current(recipe.geometry)
                || !material.pipeline_state_is_current()
            {
                continue;
            }
            self.materialized.push(MaterializedSceneSource {
                recipe: recipe.id,
                dynamic_offset: allocation.dynamic_offset,
                object_bind_group_id,
            });
        }

        self.phase = SceneSourcePreparePhase::Materialized;
    }

    /// Publish only still-current physical sources into the existing compact
    /// Direct-backend stores.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finalize(
        &mut self,
        scene: &Scene,
        geometry_frames: GeometryFrameResolver<'_>,
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
        prepared_draws.begin_sources(self.recipes.len());

        for physical in &self.materialized {
            let recipe = &self.recipes[physical.recipe.index()];
            let Some(geometry) = geometry_frames.geometry(recipe.geometry) else {
                continue;
            };
            let Some(material) = material_plans.material(recipe.material) else {
                continue;
            };
            let Some(mesh) = scene.meshes.get(recipe.node_handle) else {
                continue;
            };
            if (mesh.geometry, mesh.material) != (geometry.key.handle, material.handle())
                || !geometry_frames.pipeline_state_is_current(recipe.geometry)
                || !material.pipeline_state_is_current()
            {
                continue;
            }
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

            prepared_draws.insert_source(
                recipe.frame_source,
                PreparedDrawSource {
                    geometry: recipe.geometry,
                    material: recipe.material,
                    object_interface,
                    material_bind_group: material_bind_group_slot,
                    object_bind_group: (object_bind_group_slot, physical.dynamic_offset),
                },
            );
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
        self.phase = SceneSourcePreparePhase::Empty;
    }

    #[must_use]
    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.recipes, SCENE_SOURCE_RECIPE_BASELINE)
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
            && self.recipes.is_empty()
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
        let id = SceneSourceRecipeId::from_index(index);
        self.recipes.push(SceneSourceRecipe {
            id,
            frame_source: FrameSourceId::from_index(index),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    fn skeleton(index: u64) -> SkeletonKey {
        SkeletonKey::from(KeyData::from_ffi(index))
    }

    #[test]
    fn source_recipe_id_preserves_option_niche() {
        assert_eq!(
            size_of::<Option<SceneSourceRecipeId>>(),
            size_of::<SceneSourceRecipeId>()
        );
        assert_eq!(SceneSourceRecipeId::from_index(0).index(), 0);
        assert_eq!(SceneSourceRecipeId::from_index(127).index(), 127);
    }

    #[test]
    fn source_recipe_is_worker_safe_pod_without_hidden_owners() {
        fn assert_send_sync_static<T: Send + Sync + 'static>() {}

        assert_send_sync_static::<SceneSourceRecipe>();
        assert!(!std::mem::needs_drop::<SceneSourceRecipe>());
        assert!(
            size_of::<SceneSourceRecipe>()
                <= size_of::<DynamicModelUniforms>() + size_of::<[u64; 8]>(),
            "source recipes must remain a compact uniform payload plus Copy identities"
        );
    }

    #[test]
    fn skeleton_plan_uses_stable_first_semantic_occurrence() {
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
    }

    #[test]
    fn clear_retains_then_trim_releases_excess_capacity() {
        let mut store = SceneSourcePrepareStore::new();
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
