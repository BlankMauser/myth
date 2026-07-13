//! Graph-attempt ownership and exact snapshots for central draw materials.
//!
//! The store is the CPU-side bridge between draw-intent collection and
//! post-DPE physical preparation. Its destructor-aware asset snapshots own the
//! exact `Arc<Material>` values because the frame arena deliberately skips
//! Drop, while render passes and DrawPlans retain only POD identities and real
//! RDG texture reads.

use std::{num::NonZeroU32, ops::Range, sync::Arc};

use myth_assets::{AssetServer, AssetSnapshot, MaterialHandle};
#[cfg(test)]
use myth_assets::{ImageHandle, TextureHandle};
use myth_resources::material::{Material, RenderableMaterialTrait};
use myth_resources::material::{MaterialSettings, ShaderTemplateMode};
use myth_resources::shader_defines::ShaderDefines;
use myth_resources::texture::TextureSource;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::core::gpu::{
    AssetTextureSnapshot, PlannedMaterialTexture, snapshot_asset_texture_state,
};
use crate::core::{BindingInterfaceId, ResourceManager};

use super::asset_frame::GeometryEligibilityResolver;
use super::asset_texture_batch::{AssetTextureBatch, AssetTextureBatchId};
use super::capacity::{
    CapacityUsage, estimated_hash_map_footprint, shrink_hash_map_by_family_ratio,
    shrink_vec_by_family_ratio, vec_footprint,
};
use super::extracted::{ExtractedRenderItem, ExtractedScene};
use super::frame::{DrawPlanKey, FrameSourceId, RenderLists};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MaterialFrameId(NonZeroU32);

impl MaterialFrameId {
    #[inline]
    pub(crate) fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("material frame record index exceeds u32::MAX - 1");
        Self(raw)
    }

    #[inline]
    const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

/// Compact index into the stable-deduplicated material-owner batch captured
/// before texture topology is emitted.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MaterialAssetSnapshotId(NonZeroU32);

impl MaterialAssetSnapshotId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("material asset snapshot index exceeds u32::MAX - 1");
        Self(raw)
    }

    #[inline]
    const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MaterialPlanKey {
    handle: MaterialHandle,
    asset_version: u32,
    asset_identity: u64,
    material_version: u64,
}

impl MaterialPlanKey {
    fn from_item(item: &ExtractedRenderItem) -> Self {
        Self {
            handle: item.material,
            asset_version: item.material_asset_version,
            asset_identity: item.material_asset_identity,
            material_version: item.material_version,
        }
    }

    fn matches_item(self, item: &ExtractedRenderItem) -> bool {
        self == Self::from_item(item)
    }
}

/// Immutable shader/raster state captured once for one exact material
/// generation. Pipeline derivation consumes this coherent representative
/// instead of repeatedly locking the live material for every draw source.
pub(crate) struct MaterialPipelineSnapshot {
    settings: MaterialSettings,
    use_transmission: bool,
    shader_name: &'static str,
    shader_template: Option<&'static str>,
    shader_template_mode: ShaderTemplateMode,
    shader_defines: ShaderDefines,
}

impl MaterialPipelineSnapshot {
    fn capture(material: &Material) -> Self {
        Self {
            settings: material.settings(),
            use_transmission: material.use_transmission(),
            shader_name: material.shader_name(),
            shader_template: material.shader_template(),
            shader_template_mode: material.shader_template_mode(),
            shader_defines: material.shader_defines(),
        }
    }

    #[inline]
    pub(crate) const fn settings(&self) -> MaterialSettings {
        self.settings
    }

    #[inline]
    pub(crate) const fn use_transmission(&self) -> bool {
        self.use_transmission
    }

    #[inline]
    pub(crate) const fn shader_name(&self) -> &'static str {
        self.shader_name
    }

    #[inline]
    pub(crate) const fn shader_template(&self) -> Option<&'static str> {
        self.shader_template
    }

    #[inline]
    pub(crate) const fn shader_template_mode(&self) -> ShaderTemplateMode {
        self.shader_template_mode
    }

    #[inline]
    pub(crate) const fn shader_defines(&self) -> &ShaderDefines {
        &self.shader_defines
    }
}

/// Physical Group 1 interface published only after this material's live
/// post-DPE preparation succeeds. One exact material generation owns one copy.
pub(crate) struct MaterialFrameInterface {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) layout_id: u64,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) bind_group_id: u64,
    pub(crate) binding_wgsl: Arc<str>,
    pub(crate) binding_interface_id: BindingInterfaceId,
}

struct MaterialFrameRecord {
    key: MaterialPlanKey,
    pipeline: MaterialPipelineSnapshot,
    textures: SmallVec<[PlannedMaterialTexture; 8]>,
    /// Binding-aligned range in the store's flat canonical Texture -> Image
    /// plan-ID stream. Attachments carry `None`; Asset bindings retain a slot
    /// even when their captured state is unavailable.
    texture_plan_range: Range<usize>,
    /// Success gate for this graph attempt. A persistent `GpuMaterial` with
    /// matching material stamps is insufficient when an Image changed and
    /// this attempt's exact texture materialization failed.
    prepared: bool,
    attempted: bool,
    interface: Option<MaterialFrameInterface>,
}

/// Owned topology recipe returned to Composer before draw consumers are added.
pub(crate) struct DrawMaterialTextureRecipe {
    pub(crate) key: DrawPlanKey,
    pub(crate) textures: Vec<PlannedAssetTextureSnapshot>,
}

/// Exact topology snapshot paired with its canonical retained-owner plan.
///
/// Carrying the plan ID through graph construction prevents the post-DPE
/// materializer from rediscovering Texture/Image owners through AssetStorage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlannedAssetTextureSnapshot {
    pub(crate) owner_plan: AssetTextureBatchId,
    pub(crate) snapshot: AssetTextureSnapshot,
}

/// Destructor-aware material owners and source joins for one graph attempt.
pub(crate) struct MaterialPlanStore {
    /// Stable first-occurrence handle order for bounded AssetStorage capture.
    asset_handles: Vec<MaterialHandle>,
    asset_handle_lookup: FxHashMap<MaterialHandle, MaterialAssetSnapshotId>,
    /// Exact visible CPU owners retained through the graph attempt. These are
    /// per-entry observations, not a globally atomic storage transaction.
    asset_snapshots: Vec<AssetSnapshot<MaterialHandle, Material>>,
    asset_current: Vec<bool>,
    plans: Vec<MaterialFrameRecord>,
    /// Flat binding-aligned plan IDs for every material record. Keeping this
    /// outside each record avoids one nested allocation for materials with
    /// more than the inline texture count.
    texture_plan_ids: Vec<Option<AssetTextureBatchId>>,
    /// Parallel plan-to-owner and owner-to-plan joins. At most one coherent
    /// material generation can match one captured handle snapshot.
    plan_snapshots: Vec<MaterialAssetSnapshotId>,
    snapshot_plans: Vec<Option<MaterialFrameId>>,
    failed_snapshots: Vec<bool>,
    /// Pregraph source-to-owner qualification. This keeps repeated draws from
    /// re-hashing handles while material records and recipes are published.
    source_snapshots: Vec<Option<MaterialAssetSnapshotId>>,
    /// Dense join indexed exclusively by the intent-visible frame domain.
    /// ExtractedScene may be much larger after view culling.
    source_plans: Vec<Option<MaterialFrameId>>,
    /// Reused stable-deduplicated post-DPE selection and validation scratch.
    selected_snapshot_indices: Vec<usize>,
    selected_current: Vec<bool>,
    live_plans: Vec<MaterialFrameId>,
    recipes: Vec<DrawMaterialTextureRecipe>,
    captured: bool,
    initialized: bool,
}

const MATERIAL_PLAN_BASELINE: usize = 128;
const MATERIAL_TEXTURE_PLAN_BASELINE: usize = MATERIAL_PLAN_BASELINE * 8;
const MATERIAL_SOURCE_BASELINE: usize = 512;
const MATERIAL_RECIPE_BASELINE: usize = 16;

#[inline]
fn bit_vec_footprint(value: &Vec<bool>, baseline_capacity: usize) -> CapacityUsage {
    CapacityUsage::new(
        value.len().div_ceil(8),
        value.capacity().div_ceil(8),
        baseline_capacity.div_ceil(8),
    )
}

impl MaterialPlanStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            asset_handles: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            asset_handle_lookup: FxHashMap::default(),
            asset_snapshots: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            asset_current: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            plans: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            texture_plan_ids: Vec::with_capacity(MATERIAL_TEXTURE_PLAN_BASELINE),
            plan_snapshots: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            snapshot_plans: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            failed_snapshots: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            source_snapshots: Vec::with_capacity(MATERIAL_SOURCE_BASELINE),
            source_plans: Vec::with_capacity(MATERIAL_SOURCE_BASELINE),
            selected_snapshot_indices: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            selected_current: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            live_plans: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            recipes: Vec::with_capacity(MATERIAL_RECIPE_BASELINE),
            captured: false,
            initialized: false,
        }
    }

    /// Capture exact material owners and register their Asset Texture requests
    /// for the CPU-visible draw-intent union. This performs no GPU interaction.
    pub(crate) fn capture(
        &mut self,
        assets: &AssetServer,
        extracted_scene: &ExtractedScene,
        render_lists: &RenderLists,
        geometry_eligibility: GeometryEligibilityResolver<'_>,
        asset_textures: &mut AssetTextureBatch,
    ) {
        assert!(
            !self.captured && !self.initialized,
            "material plans may be captured only once per graph attempt"
        );
        debug_assert!(self.asset_handles.is_empty());
        debug_assert!(self.asset_handle_lookup.is_empty());
        debug_assert!(self.asset_snapshots.is_empty());
        debug_assert!(self.asset_current.is_empty());
        debug_assert!(self.plans.is_empty());
        debug_assert!(self.texture_plan_ids.is_empty());
        debug_assert!(self.plan_snapshots.is_empty());
        debug_assert!(self.snapshot_plans.is_empty());
        debug_assert!(self.failed_snapshots.is_empty());
        debug_assert!(self.source_snapshots.is_empty());
        debug_assert!(self.source_plans.is_empty());
        debug_assert!(self.selected_snapshot_indices.is_empty());
        debug_assert!(self.selected_current.is_empty());
        debug_assert!(self.live_plans.is_empty());
        debug_assert!(self.recipes.is_empty());

        self.source_snapshots
            .resize(render_lists.frame_source_count(), None);
        self.source_plans
            .resize(render_lists.frame_source_count(), None);

        // Stable-deduplicate only Geometry-qualified material handles before
        // any AssetStorage owner is cloned or any material/texture state is
        // derived. Repeated draws therefore contribute only dense source IDs.
        for source in render_lists.frame_source_ids() {
            let Some(extracted_source) = render_lists.extracted_source(source) else {
                continue;
            };
            let Some(item) = extracted_scene.render_items.get(extracted_source.index()) else {
                continue;
            };
            // This check must precede handle deduplication and later plan
            // reuse: stale Geometry must not introduce Material topology.
            if !geometry_eligibility.source_is_eligible(source, item) {
                continue;
            }
            let snapshot_id = match self.asset_handle_lookup.entry(item.material) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let id = MaterialAssetSnapshotId::from_index(self.asset_handles.len());
                    self.asset_handles.push(item.material);
                    entry.insert(id);
                    id
                }
            };
            self.source_snapshots[source.index()] = Some(snapshot_id);
        }

        assets
            .materials
            .snapshot_batched(&self.asset_handles, &mut self.asset_snapshots);
        debug_assert_eq!(self.asset_handles.len(), self.asset_snapshots.len());
        self.snapshot_plans.resize(self.asset_snapshots.len(), None);
        self.failed_snapshots
            .resize(self.asset_snapshots.len(), false);

        // Derive each exact visible owner at most once. Source stamps that do
        // not match the retained allocation are removed before they can reuse
        // a record derived for another source.
        for source in render_lists.frame_source_ids() {
            let Some(snapshot_id) = self.source_snapshots[source.index()] else {
                continue;
            };
            let Some(extracted_source) = render_lists.extracted_source(source) else {
                self.source_snapshots[source.index()] = None;
                continue;
            };
            let Some(item) = extracted_scene.render_items.get(extracted_source.index()) else {
                self.source_snapshots[source.index()] = None;
                continue;
            };
            if !self.snapshot_matches_item(snapshot_id, item) {
                self.source_snapshots[source.index()] = None;
                continue;
            }
            if self.snapshot_plans[snapshot_id.index()].is_none()
                && !self.failed_snapshots[snapshot_id.index()]
            {
                self.capture_snapshot_plan(asset_textures, snapshot_id, item);
            }
        }

        self.captured = true;
    }

    /// Resolve all Material texture recipes from the one canonical
    /// Texture -> Image batch, then publish source joins and topology.
    ///
    /// Material and Skybox contribute requests before the batch is sealed, so
    /// this phase never reopens either AssetStorage per binding.
    pub(crate) fn finalize_texture_snapshots(
        &mut self,
        assets: &AssetServer,
        extracted_scene: &ExtractedScene,
        render_lists: &RenderLists,
        asset_textures: &AssetTextureBatch,
    ) {
        assert!(self.captured, "material owners were not captured");
        assert!(
            !self.initialized,
            "material texture snapshots may be finalized only once"
        );

        for record in &mut self.plans {
            let texture_plan_ids = &self.texture_plan_ids[record.texture_plan_range.clone()];
            debug_assert_eq!(record.textures.len(), texture_plan_ids.len());
            for (planned, &batch_id) in record.textures.iter_mut().zip(texture_plan_ids) {
                planned.asset = batch_id.and_then(|id| asset_textures.state(id));
            }
        }

        // Texture topology is checked once per unique plan, followed by one
        // bounded selected AssetStorage batch. The owner validation comes
        // last so a replacement during texture inspection cannot publish a
        // stale material recipe.
        debug_assert_eq!(self.plans.len(), self.selected_snapshot_indices.len());
        self.selected_current.clear();
        for index in 0..self.plans.len() {
            let id = MaterialFrameId::from_index(index);
            let record = &self.plans[index];
            let texture_plan_ids = &self.texture_plan_ids[record.texture_plan_range.clone()];
            let current = self.material_owner(id).is_some_and(|material| {
                material_record_content_matches_batch(material, record, texture_plan_ids)
            });
            self.selected_current.push(current);
        }
        assets.materials.revalidate_selected_batched(
            &self.asset_snapshots,
            &self.selected_snapshot_indices,
            &mut self.asset_current,
        );
        debug_assert_eq!(self.plans.len(), self.asset_current.len());
        for index in 0..self.plans.len() {
            self.selected_current[index] =
                self.selected_current[index] && self.asset_current[index];
        }

        for source in render_lists.frame_source_ids() {
            let Some(snapshot_id) = self.source_snapshots[source.index()] else {
                continue;
            };
            let Some(id) = self.snapshot_plans[snapshot_id.index()] else {
                continue;
            };
            let Some(extracted_source) = render_lists.extracted_source(source) else {
                continue;
            };
            let Some(item) = extracted_scene.render_items.get(extracted_source.index()) else {
                continue;
            };
            if self.selected_current.get(id.index()).copied() == Some(true)
                && self.plans[id.index()].key.matches_item(item)
            {
                self.source_plans[source.index()] = Some(id);
            }
        }

        self.push_recipe(
            DrawPlanKey::Opaque,
            render_lists.opaque.iter().map(|intent| intent.source),
        );
        self.push_recipe(
            DrawPlanKey::Transparent,
            render_lists.transparent.iter().map(|intent| intent.source),
        );
        let mut shadow_keys: Vec<_> = render_lists.shadow_queues.keys().copied().collect();
        shadow_keys.sort_unstable();
        for (light_id, layer_index) in shadow_keys {
            self.push_recipe(
                DrawPlanKey::Shadow {
                    light_id,
                    layer_index,
                },
                render_lists.shadow_queues[&(light_id, layer_index)]
                    .iter()
                    .map(|intent| intent.source),
            );
        }

        self.initialized = true;
    }

    fn capture_snapshot_plan(
        &mut self,
        asset_textures: &mut AssetTextureBatch,
        snapshot_id: MaterialAssetSnapshotId,
        item: &ExtractedRenderItem,
    ) {
        let Some(loaded) = self.asset_snapshots[snapshot_id.index()].loaded() else {
            self.failed_snapshots[snapshot_id.index()] = true;
            return;
        };
        let material = loaded.asset();
        let material_version = material.data.version();
        let key = MaterialPlanKey {
            handle: self.asset_snapshots[snapshot_id.index()].handle(),
            asset_version: loaded.version(),
            asset_identity: loaded.identity(),
            material_version,
        };
        if !key.matches_item(item) {
            self.failed_snapshots[snapshot_id.index()] = true;
            return;
        }

        let pipeline = MaterialPipelineSnapshot::capture(material);
        let mut textures = SmallVec::<[PlannedMaterialTexture; 8]>::new();
        material.data.visit_textures(&mut |source| {
            textures.push(PlannedMaterialTexture {
                source: *source,
                asset: None,
            });
        });
        // Every field above belongs to the same serial capture window. A
        // mutation observed during capture rejects this attempt instead of
        // publishing a mixed classification/shader/binding recipe or adding
        // orphan Texture requests to the shared batch.
        if material.data.version() != material_version {
            self.failed_snapshots[snapshot_id.index()] = true;
            return;
        }

        let texture_plan_start = self.texture_plan_ids.len();
        for planned in &textures {
            let asset_plan = match planned.source {
                TextureSource::Asset(handle) => Some(asset_textures.request(handle)),
                TextureSource::Attachment(_, _) => None,
            };
            self.texture_plan_ids.push(asset_plan);
        }
        let texture_plan_range = texture_plan_start..self.texture_plan_ids.len();
        let record = MaterialFrameRecord {
            key,
            pipeline,
            textures,
            texture_plan_range,
            prepared: false,
            attempted: false,
            interface: None,
        };

        let id = MaterialFrameId::from_index(self.plans.len());
        self.plans.push(record);
        self.plan_snapshots.push(snapshot_id);
        self.selected_snapshot_indices.push(snapshot_id.index());
        assert!(
            self.snapshot_plans[snapshot_id.index()]
                .replace(id)
                .is_none()
        );
    }

    #[inline]
    fn snapshot_matches_item(
        &self,
        id: MaterialAssetSnapshotId,
        item: &ExtractedRenderItem,
    ) -> bool {
        self.asset_snapshots
            .get(id.index())
            .is_some_and(|snapshot| {
                snapshot.handle() == item.material
                    && snapshot.loaded().is_some_and(|loaded| {
                        (loaded.version(), loaded.identity())
                            == (item.material_asset_version, item.material_asset_identity)
                            && loaded.asset().data.version() == item.material_version
                    })
            })
    }

    #[inline]
    fn material_owner(&self, id: MaterialFrameId) -> Option<&Arc<Material>> {
        let snapshot_id = *self.plan_snapshots.get(id.index())?;
        self.asset_snapshots
            .get(snapshot_id.index())?
            .loaded()
            .map(myth_assets::LoadedAssetSnapshot::asset)
    }

    fn push_recipe(&mut self, key: DrawPlanKey, sources: impl IntoIterator<Item = FrameSourceId>) {
        let mut visited_materials = FxHashSet::default();
        let mut visited_textures = FxHashSet::default();
        let mut textures = Vec::new();
        for source in sources {
            let Some(plan) = self.source_plans.get(source.index()).copied().flatten() else {
                continue;
            };
            if !visited_materials.insert(plan) {
                continue;
            }
            let record = &self.plans[plan.index()];
            let texture_plan_ids = &self.texture_plan_ids[record.texture_plan_range.clone()];
            debug_assert_eq!(record.textures.len(), texture_plan_ids.len());
            for (texture, &owner_plan) in record.textures.iter().zip(texture_plan_ids) {
                let Some(owner_plan) = owner_plan else {
                    continue;
                };
                let Some(snapshot) = texture.asset.and_then(|state| state.ready) else {
                    continue;
                };
                if visited_textures.insert(owner_plan) {
                    textures.push(PlannedAssetTextureSnapshot {
                        owner_plan,
                        snapshot,
                    });
                }
            }
        }
        self.recipes
            .push(DrawMaterialTextureRecipe { key, textures });
    }

    /// Move the topology recipes out once all material producers are about to
    /// be registered. Plans and source joins remain owned through post-DPE
    /// preparation; only this temporary Vec changes ownership.
    pub(crate) fn take_topology_recipes(&mut self) -> Vec<DrawMaterialTextureRecipe> {
        assert!(self.initialized, "material plans were not captured");
        std::mem::take(&mut self.recipes)
    }

    /// Prepare each unique material referenced by the DPE-live source union.
    /// Only a fully revalidated successful attempt opens the source join.
    pub(crate) fn prepare_live(
        &mut self,
        assets: &AssetServer,
        resource_manager: &mut ResourceManager,
        live_sources: &[FrameSourceId],
    ) {
        assert!(self.initialized, "material plans were not captured");

        self.selected_snapshot_indices.clear();
        self.selected_current.clear();
        self.live_plans.clear();
        for &source in live_sources {
            let Some(id) = self.source_plans.get(source.index()).copied().flatten() else {
                continue;
            };
            if self.plans[id.index()].attempted {
                continue;
            }
            self.plans[id.index()].attempted = true;
            self.live_plans.push(id);
            self.selected_snapshot_indices
                .push(self.plan_snapshots[id.index()].index());
        }

        // Only the stable-deduplicated DPE-live owner subset touches
        // AssetStorage after graph compilation.
        assets.materials.revalidate_selected_batched(
            &self.asset_snapshots,
            &self.selected_snapshot_indices,
            &mut self.selected_current,
        );
        debug_assert_eq!(self.live_plans.len(), self.selected_current.len());

        for live_index in 0..self.live_plans.len() {
            if !self.selected_current[live_index] {
                continue;
            }
            let id = self.live_plans[live_index];
            let Some(material) = self.material_owner(id) else {
                continue;
            };
            if !material_record_content_is_current(assets, material, &self.plans[id.index()]) {
                continue;
            }
            let prepared = {
                let record = &self.plans[id.index()];
                resource_manager.prepare_planned_material(
                    assets,
                    record.key.handle,
                    material,
                    record.key.asset_version,
                    record.key.asset_identity,
                    record.key.material_version,
                    &record.textures,
                )
            };
            if prepared {
                let record = &self.plans[id.index()];
                let Some(gpu_material) = resource_manager.get_material(record.key.handle) else {
                    continue;
                };
                if (gpu_material.asset_version, gpu_material.asset_identity)
                    != (record.key.asset_version, record.key.asset_identity)
                    || gpu_material.version != record.key.material_version
                {
                    continue;
                }
                let interface = MaterialFrameInterface {
                    layout: gpu_material.layout.clone(),
                    layout_id: gpu_material.layout_id,
                    bind_group: gpu_material.bind_group.clone(),
                    bind_group_id: gpu_material.bind_group_id,
                    binding_wgsl: gpu_material.binding_wgsl.clone(),
                    binding_interface_id: gpu_material.binding_interface_id,
                };
                let record = &mut self.plans[id.index()];
                record.interface = Some(interface);
                record.prepared = true;
            }
        }

        // Physical preparation may overlap an asset or texture mutation.
        // Revalidate the same dense live subset once and retract every stale
        // interface before any draw source can observe it.
        for live_index in 0..self.live_plans.len() {
            let id = self.live_plans[live_index];
            self.selected_current[live_index] = self.material_owner(id).is_some_and(|material| {
                material_record_content_is_current(assets, material, &self.plans[id.index()])
            });
        }
        assets.materials.revalidate_selected_batched(
            &self.asset_snapshots,
            &self.selected_snapshot_indices,
            &mut self.asset_current,
        );
        for live_index in 0..self.live_plans.len() {
            let id = self.live_plans[live_index];
            if !self.selected_current[live_index] || !self.asset_current[live_index] {
                let record = &mut self.plans[id.index()];
                record.prepared = false;
                record.interface = None;
            }
        }
    }

    #[must_use]
    pub(crate) const fn resolver(&self) -> MaterialPlanResolver<'_> {
        MaterialPlanResolver { store: self }
    }

    pub(crate) fn clear(&mut self) {
        self.recipes.clear();
        self.live_plans.clear();
        self.selected_current.clear();
        self.selected_snapshot_indices.clear();
        self.source_plans.clear();
        self.source_snapshots.clear();
        self.failed_snapshots.clear();
        self.snapshot_plans.clear();
        self.plan_snapshots.clear();
        self.texture_plan_ids.clear();
        self.plans.clear();
        self.asset_current.clear();
        self.asset_snapshots.clear();
        self.asset_handle_lookup.clear();
        self.asset_handles.clear();
        self.captured = false;
        self.initialized = false;
    }

    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.asset_handles, MATERIAL_PLAN_BASELINE)
            + estimated_hash_map_footprint(&self.asset_handle_lookup, MATERIAL_PLAN_BASELINE)
            + vec_footprint(&self.asset_snapshots, MATERIAL_PLAN_BASELINE)
            + bit_vec_footprint(&self.asset_current, MATERIAL_PLAN_BASELINE)
            + vec_footprint(&self.plans, MATERIAL_PLAN_BASELINE)
            + vec_footprint(&self.texture_plan_ids, MATERIAL_TEXTURE_PLAN_BASELINE)
            + vec_footprint(&self.plan_snapshots, MATERIAL_PLAN_BASELINE)
            + vec_footprint(&self.snapshot_plans, MATERIAL_PLAN_BASELINE)
            + bit_vec_footprint(&self.failed_snapshots, MATERIAL_PLAN_BASELINE)
            + vec_footprint(&self.source_snapshots, MATERIAL_SOURCE_BASELINE)
            + vec_footprint(&self.source_plans, MATERIAL_SOURCE_BASELINE)
            + vec_footprint(&self.selected_snapshot_indices, MATERIAL_PLAN_BASELINE)
            + bit_vec_footprint(&self.selected_current, MATERIAL_PLAN_BASELINE)
            + vec_footprint(&self.live_plans, MATERIAL_PLAN_BASELINE)
            + vec_footprint(&self.recipes, MATERIAL_RECIPE_BASELINE)
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        shrink_vec_by_family_ratio(
            &mut self.asset_handles,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.asset_handle_lookup,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.asset_snapshots,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.asset_current,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.plans,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.texture_plan_ids,
            MATERIAL_TEXTURE_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.plan_snapshots,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.snapshot_plans,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.failed_snapshots,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.source_snapshots,
            MATERIAL_SOURCE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.source_plans,
            MATERIAL_SOURCE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.selected_snapshot_indices,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.selected_current,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.live_plans,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.recipes,
            MATERIAL_RECIPE_BASELINE,
            target_bytes,
            retained_bytes,
        );
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.asset_handles.is_empty()
            && self.asset_handle_lookup.is_empty()
            && self.asset_snapshots.is_empty()
            && self.asset_current.is_empty()
            && self.plans.is_empty()
            && self.texture_plan_ids.is_empty()
            && self.plan_snapshots.is_empty()
            && self.snapshot_plans.is_empty()
            && self.failed_snapshots.is_empty()
            && self.source_snapshots.is_empty()
            && self.source_plans.is_empty()
            && self.selected_snapshot_indices.is_empty()
            && self.selected_current.is_empty()
            && self.live_plans.is_empty()
            && self.recipes.is_empty()
            && !self.captured
            && !self.initialized
    }
}

impl Default for MaterialPlanStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable exact source join consumed by central draw preparation.
#[derive(Clone, Copy)]
pub(crate) struct MaterialPlanResolver<'a> {
    store: &'a MaterialPlanStore,
}

impl<'a> MaterialPlanResolver<'a> {
    #[must_use]
    pub(crate) fn prepared_material(
        self,
        source: FrameSourceId,
        item: &ExtractedRenderItem,
    ) -> Option<PreparedMaterialFrame<'a>> {
        let id = self
            .store
            .source_plans
            .get(source.index())
            .copied()
            .flatten()?;
        let plan = self.store.plans.get(id.index())?;
        let material = self.store.material_owner(id)?.as_ref();
        (plan.prepared && plan.interface.is_some() && plan.key.matches_item(item)).then_some(
            PreparedMaterialFrame {
                id,
                record: plan,
                material,
            },
        )
    }

    #[must_use]
    pub(crate) fn material(self, id: MaterialFrameId) -> Option<PreparedMaterialFrame<'a>> {
        let record = self.store.plans.get(id.index())?;
        let material = self.store.material_owner(id)?.as_ref();
        (record.prepared && record.interface.is_some()).then_some(PreparedMaterialFrame {
            id,
            record,
            material,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PreparedMaterialFrame<'a> {
    id: MaterialFrameId,
    record: &'a MaterialFrameRecord,
    material: &'a Material,
}

impl<'a> PreparedMaterialFrame<'a> {
    #[inline]
    pub(crate) const fn id(self) -> MaterialFrameId {
        self.id
    }

    #[inline]
    pub(crate) const fn pipeline(self) -> &'a MaterialPipelineSnapshot {
        &self.record.pipeline
    }

    #[inline]
    pub(crate) fn interface(self) -> &'a MaterialFrameInterface {
        self.record
            .interface
            .as_ref()
            .expect("prepared material lacks its physical interface")
    }

    #[inline]
    pub(crate) const fn handle(self) -> MaterialHandle {
        self.record.key.handle
    }

    #[inline]
    pub(crate) const fn asset_version(self) -> u32 {
        self.record.key.asset_version
    }

    #[inline]
    pub(crate) const fn asset_identity(self) -> u64 {
        self.record.key.asset_identity
    }

    #[inline]
    pub(crate) const fn material_version(self) -> u64 {
        self.record.key.material_version
    }

    #[inline]
    pub(crate) fn pipeline_state_is_current(self) -> bool {
        self.material.data.version() == self.record.key.material_version
    }
}

#[cfg(test)]
fn material_plan_is_current(
    assets: &AssetServer,
    expected: &Arc<Material>,
    record: &MaterialFrameRecord,
) -> bool {
    let Some((material, asset_version, asset_identity)) =
        assets.materials.get_entry_with_identity(record.key.handle)
    else {
        return false;
    };
    if (asset_version, asset_identity) != (record.key.asset_version, record.key.asset_identity)
        || !Arc::ptr_eq(&material, expected)
        || expected.data.version() != record.key.material_version
    {
        return false;
    }

    material_record_content_is_current(assets, expected, record)
        && assets
            .materials
            .get_entry_with_identity(record.key.handle)
            .is_some_and(|(current, version, identity)| {
                version == record.key.asset_version
                    && identity == record.key.asset_identity
                    && Arc::ptr_eq(&current, expected)
            })
}

/// Validate the owner-local Material epoch and binding order against the
/// already sealed cross-storage batch. This is the graph-publication path: it
/// performs no per-binding AssetStorage query.
fn material_record_content_matches_batch(
    material: &Material,
    record: &MaterialFrameRecord,
    texture_plan_ids: &[Option<AssetTextureBatchId>],
) -> bool {
    if material.data.version() != record.key.material_version
        || record.textures.len() != texture_plan_ids.len()
    {
        return false;
    }

    let mut texture_index = 0usize;
    let mut matches = true;
    material.data.visit_textures(&mut |source| {
        if !matches {
            return;
        }
        let Some(planned) = record.textures.get(texture_index) else {
            matches = false;
            return;
        };
        let Some(&batch_id) = texture_plan_ids.get(texture_index) else {
            matches = false;
            return;
        };
        texture_index += 1;
        if planned.source != *source {
            matches = false;
            return;
        }
        matches = match (*source, batch_id) {
            (TextureSource::Asset(handle), Some(_)) => planned
                .asset
                .is_some_and(|state| state.texture_handle == handle),
            (TextureSource::Attachment(_, _), None) => planned.asset.is_none(),
            _ => false,
        };
    });

    matches
        && texture_index == record.textures.len()
        && material.data.version() == record.key.material_version
}

/// Validate owner-local Material epochs and the complete texture-source
/// recipe without acquiring the Material AssetStorage lock. Production paths
/// pair this with one bounded batch revalidation of exact retained owners.
fn material_record_content_is_current(
    assets: &AssetServer,
    material: &Material,
    record: &MaterialFrameRecord,
) -> bool {
    if material.data.version() != record.key.material_version {
        return false;
    }

    let mut texture_index = 0usize;
    let mut matches = true;
    material.data.visit_textures(&mut |source| {
        if !matches {
            return;
        }
        let Some(planned) = record.textures.get(texture_index) else {
            matches = false;
            return;
        };
        texture_index += 1;
        if planned.source != *source {
            matches = false;
            return;
        }
        match *source {
            TextureSource::Asset(handle) => {
                matches = planned.asset.is_some_and(|state| {
                    state.texture_handle == handle
                        && snapshot_asset_texture_state(assets, handle) == state
                });
            }
            TextureSource::Attachment(_, _) => matches = planned.asset.is_none(),
        }
    });

    matches
        && texture_index == record.textures.len()
        && material.data.version() == record.key.material_version
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WgpuContext;
    use crate::graph::extracted::MaterialClassification;
    use crate::settings::{RendererInitConfig, RendererSettings};
    use myth_assets::GeometryHandle;
    use myth_resources::BoundingBox;
    use myth_resources::geometry::Geometry;
    use myth_resources::image::{Image, ImageDimension, PixelFormat};
    use myth_resources::material::{AlphaMode, MaterialType, UnlitMaterial};
    use myth_resources::texture::Texture;
    use slotmap::Key;

    fn image_8x8() -> Image {
        Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![127; 8 * 8 * 4]),
        )
    }

    fn unlit_with_texture(texture: TextureHandle) -> Material {
        let unlit = UnlitMaterial::new(glam::Vec4::ONE);
        unlit.set_map(Some(texture));
        Material::new(MaterialType::Unlit(unlit))
    }

    fn extracted_item(
        geometry: GeometryHandle,
        geometry_asset_version: u32,
        geometry_asset_identity: u64,
        material: MaterialHandle,
        material_asset_version: u32,
        material_asset_identity: u64,
        material_version: u64,
    ) -> ExtractedRenderItem {
        ExtractedRenderItem {
            node_handle: myth_scene::NodeHandle::null(),
            world_matrix: glam::Mat4::IDENTITY,
            prev_world_matrix: glam::Mat4::IDENTITY,
            geometry,
            material,
            geometry_asset_version,
            geometry_asset_identity,
            material_asset_version,
            material_asset_identity,
            material_version,
            material_index: 0,
            material_classification: MaterialClassification::empty(),
            skeleton: None,
            item_variant_flags: 0,
            item_shader_defines: ShaderDefines::new(),
            item_shader_defines_hash: 0,
            cast_shadows: true,
            receive_shadows: true,
            world_aabb: BoundingBox::default(),
        }
    }

    fn capture_test_plans(
        store: &mut MaterialPlanStore,
        assets: &AssetServer,
        extracted_scene: &ExtractedScene,
        render_lists: &RenderLists,
        geometry_eligibility: GeometryEligibilityResolver<'_>,
    ) -> AssetTextureBatch {
        let mut asset_textures = AssetTextureBatch::new();
        store.capture(
            assets,
            extracted_scene,
            render_lists,
            geometry_eligibility,
            &mut asset_textures,
        );
        asset_textures.capture_requested(assets);
        store.finalize_texture_snapshots(assets, extracted_scene, render_lists, &asset_textures);
        asset_textures
    }

    fn capture_record(
        assets: &AssetServer,
        handle: MaterialHandle,
    ) -> (Arc<Material>, MaterialFrameRecord) {
        let (material, asset_version, asset_identity) = assets
            .materials
            .get_entry_with_identity(handle)
            .expect("material missing");
        let material_version = material.data.version();
        let mut textures = SmallVec::new();
        material.data.visit_textures(&mut |source| {
            textures.push(PlannedMaterialTexture {
                source: *source,
                asset: match *source {
                    TextureSource::Asset(texture) => {
                        Some(snapshot_asset_texture_state(assets, texture))
                    }
                    TextureSource::Attachment(_, _) => None,
                },
            });
        });
        let pipeline = MaterialPipelineSnapshot::capture(&material);
        let texture_plan_range = 0..textures.len();
        let record = MaterialFrameRecord {
            key: MaterialPlanKey {
                handle,
                asset_version,
                asset_identity,
                material_version,
            },
            pipeline,
            textures,
            texture_plan_range,
            prepared: false,
            attempted: false,
            interface: None,
        };
        (material, record)
    }

    #[test]
    fn material_frame_id_uses_the_nonzero_option_niche() {
        assert_eq!(std::mem::size_of::<MaterialFrameId>(), 4);
        assert_eq!(std::mem::size_of::<Option<MaterialFrameId>>(), 4);
        assert_eq!(std::mem::size_of::<MaterialAssetSnapshotId>(), 4);
        assert_eq!(std::mem::size_of::<Option<MaterialAssetSnapshotId>>(), 4);
        assert_eq!(MaterialFrameId::from_index(0).index(), 0);
        assert_eq!(
            MaterialFrameId::from_index(u32::MAX as usize - 1).index(),
            u32::MAX as usize - 1
        );
        assert!(
            std::panic::catch_unwind(|| MaterialFrameId::from_index(u32::MAX as usize)).is_err()
        );
    }

    #[test]
    fn repeated_sources_share_one_batched_material_owner() {
        const SOURCE_COUNT: usize = 1_024;

        let assets = AssetServer::new();
        let geometry = assets.geometries.add(Geometry::new());
        let (_, geometry_version, geometry_identity) = assets
            .geometries
            .get_entry_with_identity(geometry)
            .expect("geometry missing");
        let image = assets.images.add(image_8x8());
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("shared material texture"), image));
        let material = assets.materials.add(unlit_with_texture(texture));
        let (retained_material, material_version, material_identity) = assets
            .materials
            .get_entry_with_identity(material)
            .expect("material missing");
        let data_version = retained_material.data.version();

        let mut extracted_scene = ExtractedScene::new();
        let mut render_lists = RenderLists::new();
        for index in 0..SOURCE_COUNT {
            extracted_scene.render_items.push(extracted_item(
                geometry,
                geometry_version,
                geometry_identity,
                material,
                material_version,
                material_identity,
                data_version,
            ));
            render_lists
                .intern_frame_source(super::super::frame::ExtractedSourceId::from_index(index));
        }

        let mut geometry_frames = super::super::asset_frame::GeometryFrameStore::new();
        geometry_frames.capture_visible(&assets, &extracted_scene, &render_lists);
        let mut store = MaterialPlanStore::new();
        let asset_textures = capture_test_plans(
            &mut store,
            &assets,
            &extracted_scene,
            &render_lists,
            geometry_frames.eligibility(&extracted_scene, &render_lists),
        );

        assert_eq!(store.asset_handles, [material]);
        assert_eq!(store.asset_snapshots.len(), 1);
        assert_eq!(store.plans.len(), 1);
        assert_eq!(store.plan_snapshots.len(), 1);
        assert_eq!(store.snapshot_plans.len(), 1);
        assert_eq!(store.source_snapshots.len(), SOURCE_COUNT);
        assert_eq!(store.source_plans.len(), SOURCE_COUNT);
        assert!(store.source_plans.iter().all(Option::is_some));
        assert_eq!(asset_textures.texture_count(), 1);
        assert_eq!(asset_textures.image_count(), 1);
        assert_eq!(store.plans[0].textures.len(), 1);
        assert!(store.plans[0].textures[0].asset.unwrap().ready.is_some());
        assert_eq!(
            Arc::strong_count(&retained_material),
            3,
            "storage, test owner, and one visible snapshot are the only Material owners"
        );

        store.clear();
        assert_eq!(Arc::strong_count(&retained_material), 2);
        assert!(store.is_empty());
    }

    #[test]
    fn material_and_skybox_share_one_cross_storage_texture_snapshot() {
        let assets = AssetServer::new();
        let geometry = assets.geometries.add(Geometry::new());
        let (_, geometry_version, geometry_identity) = assets
            .geometries
            .get_entry_with_identity(geometry)
            .expect("geometry missing");
        let image = assets.images.add(image_8x8());
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("shared material and skybox"), image));
        let material = assets.materials.add(unlit_with_texture(texture));
        let (material_owner, material_version, material_identity) = assets
            .materials
            .get_entry_with_identity(material)
            .expect("material missing");
        let material_data_version = material_owner.data.version();
        drop(material_owner);
        let texture_owner = assets.textures.get(texture).expect("texture missing");
        let image_owner = assets.images.get(image).expect("image missing");
        let texture_owners_before_capture = Arc::strong_count(&texture_owner);
        let image_owners_before_capture = Arc::strong_count(&image_owner);

        let mut extracted_scene = ExtractedScene::new();
        extracted_scene.render_items.push(extracted_item(
            geometry,
            geometry_version,
            geometry_identity,
            material,
            material_version,
            material_identity,
            material_data_version,
        ));
        let mut render_lists = RenderLists::new();
        let source =
            render_lists.intern_frame_source(super::super::frame::ExtractedSourceId::from_index(0));
        render_lists.insert_opaque(super::super::frame::RenderIntent {
            source,
            geometry_handle: geometry,
            material_handle: material,
            material_index: 0,
            distance_sq: 0.0,
        });

        let mut geometry_frames = super::super::asset_frame::GeometryFrameStore::new();
        geometry_frames.capture_visible(&assets, &extracted_scene, &render_lists);
        let mut store = MaterialPlanStore::new();
        let mut asset_textures = AssetTextureBatch::new();
        store.capture(
            &assets,
            &extracted_scene,
            &render_lists,
            geometry_frames.eligibility(&extracted_scene, &render_lists),
            &mut asset_textures,
        );

        let texture_plan_range = store.plans[0].texture_plan_range.clone();
        assert_eq!(texture_plan_range.len(), 1);
        let material_texture_plan =
            store.texture_plan_ids[texture_plan_range.clone()][0].expect("asset texture plan");
        let skybox_texture_plan = asset_textures.request(texture);
        assert_eq!(material_texture_plan, skybox_texture_plan);
        assert_eq!(
            &store.texture_plan_ids[texture_plan_range],
            &[Some(skybox_texture_plan)]
        );
        assert_eq!(
            Arc::strong_count(&texture_owner),
            texture_owners_before_capture
        );
        assert_eq!(Arc::strong_count(&image_owner), image_owners_before_capture);

        asset_textures.capture_requested(&assets);
        assert_eq!(asset_textures.texture_count(), 1);
        assert_eq!(asset_textures.image_count(), 1);
        assert!(
            asset_textures
                .texture_owner(material_texture_plan)
                .is_some()
        );
        assert!(asset_textures.image_owner(material_texture_plan).is_some());
        assert_eq!(
            Arc::strong_count(&texture_owner),
            texture_owners_before_capture + 1,
            "the shared request must retain one exact Texture owner"
        );
        assert_eq!(
            Arc::strong_count(&image_owner),
            image_owners_before_capture + 1,
            "the shared request must retain one exact Image owner"
        );

        store.finalize_texture_snapshots(&assets, &extracted_scene, &render_lists, &asset_textures);
        assert!(store.source_plans[source.index()].is_some());
        assert!(
            store.plans[0].textures[0]
                .asset
                .is_some_and(|state| state.ready.is_some())
        );
    }

    #[test]
    fn material_aba_before_texture_finalize_publishes_no_source_or_topology() {
        let assets = AssetServer::new();
        let geometry = assets.geometries.add(Geometry::new());
        let (_, geometry_version, geometry_identity) = assets
            .geometries
            .get_entry_with_identity(geometry)
            .expect("geometry missing");
        let image = assets.images.add(image_8x8());
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("replaced material texture"), image));
        let material = assets.materials.add(unlit_with_texture(texture));
        let (material_owner, material_version, material_identity) = assets
            .materials
            .get_entry_with_identity(material)
            .expect("material missing");
        let material_data_version = material_owner.data.version();
        drop(material_owner);

        let mut extracted_scene = ExtractedScene::new();
        extracted_scene.render_items.push(extracted_item(
            geometry,
            geometry_version,
            geometry_identity,
            material,
            material_version,
            material_identity,
            material_data_version,
        ));
        let mut render_lists = RenderLists::new();
        let source =
            render_lists.intern_frame_source(super::super::frame::ExtractedSourceId::from_index(0));
        render_lists.insert_opaque(super::super::frame::RenderIntent {
            source,
            geometry_handle: geometry,
            material_handle: material,
            material_index: 0,
            distance_sq: 0.0,
        });

        let mut geometry_frames = super::super::asset_frame::GeometryFrameStore::new();
        geometry_frames.capture_visible(&assets, &extracted_scene, &render_lists);
        let mut store = MaterialPlanStore::new();
        let mut asset_textures = AssetTextureBatch::new();
        store.capture(
            &assets,
            &extracted_scene,
            &render_lists,
            geometry_frames.eligibility(&extracted_scene, &render_lists),
            &mut asset_textures,
        );
        assert_eq!(
            asset_textures.id(texture),
            Some(store.texture_plan_ids[0].unwrap())
        );

        assets
            .materials
            .insert_ready(material, Material::new_unlit(glam::Vec4::ONE));
        let (_, replacement_version, replacement_identity) = assets
            .materials
            .get_entry_with_identity(material)
            .expect("replacement material missing");
        assert_eq!(replacement_version, material_version);
        assert_ne!(replacement_identity, material_identity);

        asset_textures.capture_requested(&assets);
        store.finalize_texture_snapshots(&assets, &extracted_scene, &render_lists, &asset_textures);
        let recipes = store.take_topology_recipes();
        let opaque = recipes
            .iter()
            .find(|recipe| matches!(recipe.key, DrawPlanKey::Opaque))
            .expect("opaque topology recipe missing");

        assert!(store.source_plans[source.index()].is_none());
        assert!(
            opaque.textures.is_empty(),
            "a replaced Material must not publish topology from its retained old owner"
        );
    }

    #[test]
    fn packed_material_revalidation_bits_are_accounted_as_bits() {
        let mut current = Vec::with_capacity(129);
        current.resize(129, true);
        let usage = bit_vec_footprint(&current, 128);

        assert_eq!(usage.active_bytes, 129_usize.div_ceil(8));
        assert_eq!(usage.retained_bytes, current.capacity().div_ceil(8));
        assert_eq!(usage.baseline_bytes, 128_usize.div_ceil(8));
    }

    #[test]
    fn unique_material_capacity_releases_owners_and_trims_after_clear() {
        const MATERIAL_COUNT: usize = 512;

        let assets = AssetServer::new();
        let geometry = assets.geometries.add(Geometry::new());
        let (_, geometry_version, geometry_identity) = assets
            .geometries
            .get_entry_with_identity(geometry)
            .expect("geometry missing");
        let mut first_material_owner = None;
        let mut extracted_scene = ExtractedScene::new();
        let mut render_lists = RenderLists::new();
        for index in 0..MATERIAL_COUNT {
            let material = assets.materials.add(Material::new_unlit(glam::Vec4::ONE));
            let (owner, asset_version, asset_identity) = assets
                .materials
                .get_entry_with_identity(material)
                .expect("material missing");
            let material_version = owner.data.version();
            if index == 0 {
                first_material_owner = Some(Arc::clone(&owner));
            }
            extracted_scene.render_items.push(extracted_item(
                geometry,
                geometry_version,
                geometry_identity,
                material,
                asset_version,
                asset_identity,
                material_version,
            ));
            render_lists
                .intern_frame_source(super::super::frame::ExtractedSourceId::from_index(index));
        }
        let first_material_owner = first_material_owner.expect("first material owner missing");
        let owner_count_before_capture = Arc::strong_count(&first_material_owner);

        let mut geometry_frames = super::super::asset_frame::GeometryFrameStore::new();
        geometry_frames.capture_visible(&assets, &extracted_scene, &render_lists);
        let mut store = MaterialPlanStore::new();
        let mut asset_textures = AssetTextureBatch::new();
        store.capture(
            &assets,
            &extracted_scene,
            &render_lists,
            geometry_frames.eligibility(&extracted_scene, &render_lists),
            &mut asset_textures,
        );
        asset_textures.capture_requested(&assets);
        store.finalize_texture_snapshots(&assets, &extracted_scene, &render_lists, &asset_textures);

        let populated = store.capacity_usage();
        assert!(populated.active_bytes > 0);
        assert!(populated.retained_bytes >= populated.active_bytes);
        assert_eq!(
            Arc::strong_count(&first_material_owner),
            owner_count_before_capture + 1
        );

        store.clear();
        let released = store.capacity_usage();
        assert!(store.is_empty());
        assert_eq!(released.active_bytes, 0);
        assert_eq!(released.retained_bytes, populated.retained_bytes);
        assert_eq!(
            Arc::strong_count(&first_material_owner),
            owner_count_before_capture,
            "clearing the graph attempt must release every exact Material owner"
        );

        store.trim_capacity(released.baseline_bytes, released.retained_bytes);
        let trimmed = store.capacity_usage();
        assert!(trimmed.retained_bytes < released.retained_bytes);
        assert!(trimmed.retained_bytes >= trimmed.baseline_bytes);
    }

    #[test]
    fn source_plan_join_uses_only_the_visible_frame_domain() {
        let assets = AssetServer::new();
        let extracted_scene = ExtractedScene::new();
        let mut render_lists = RenderLists::new();
        render_lists.intern_frame_source(super::super::frame::ExtractedSourceId::from_index(7));
        render_lists.intern_frame_source(super::super::frame::ExtractedSourceId::from_index(
            1_000_000,
        ));

        let mut geometry_frames = super::super::asset_frame::GeometryFrameStore::new();
        geometry_frames.capture_visible(&assets, &extracted_scene, &render_lists);
        let mut store = MaterialPlanStore::new();
        let _asset_textures = capture_test_plans(
            &mut store,
            &assets,
            &extracted_scene,
            &render_lists,
            geometry_frames.eligibility(&extracted_scene, &render_lists),
        );

        assert_eq!(render_lists.frame_source_count(), 2);
        assert_eq!(store.source_plans.len(), 2);
        assert!(store.source_plans.iter().all(Option::is_none));
    }

    #[test]
    fn stale_geometry_cannot_inherit_a_shared_material_plan() {
        let assets = AssetServer::new();
        let valid_geometry = assets.geometries.add(Geometry::new());
        let stale_geometry = assets.geometries.add(Geometry::new());
        let (_, valid_version, valid_identity) = assets
            .geometries
            .get_entry_with_identity(valid_geometry)
            .expect("valid geometry missing");
        let (_, stale_version, stale_identity) = assets
            .geometries
            .get_entry_with_identity(stale_geometry)
            .expect("stale geometry missing");
        let material = assets.materials.add(Material::new_unlit(glam::Vec4::ONE));
        let (material_asset, material_version, material_identity) = assets
            .materials
            .get_entry_with_identity(material)
            .expect("material missing");
        let data_version = material_asset.data.version();
        drop(material_asset);

        // Preserve the extracted stamp while replacing the slot with a new
        // same-version owner before the graph-build qualification batch.
        assets
            .geometries
            .insert_ready(stale_geometry, Geometry::new());
        assert_eq!(assets.geometries.get_version(stale_geometry), Some(1));

        let mut extracted_scene = ExtractedScene::new();
        extracted_scene.render_items.push(extracted_item(
            valid_geometry,
            valid_version,
            valid_identity,
            material,
            material_version,
            material_identity,
            data_version,
        ));
        extracted_scene.render_items.push(extracted_item(
            stale_geometry,
            stale_version,
            stale_identity,
            material,
            material_version,
            material_identity,
            data_version,
        ));
        let mut render_lists = RenderLists::new();
        let valid_source =
            render_lists.intern_frame_source(super::super::frame::ExtractedSourceId::from_index(0));
        let stale_source =
            render_lists.intern_frame_source(super::super::frame::ExtractedSourceId::from_index(1));

        let mut geometry_frames = super::super::asset_frame::GeometryFrameStore::new();
        geometry_frames.capture_visible(&assets, &extracted_scene, &render_lists);
        let mut store = MaterialPlanStore::new();
        let _asset_textures = capture_test_plans(
            &mut store,
            &assets,
            &extracted_scene,
            &render_lists,
            geometry_frames.eligibility(&extracted_scene, &render_lists),
        );

        assert!(store.source_plans[valid_source.index()].is_some());
        assert!(store.source_plans[stale_source.index()].is_none());
        assert_eq!(store.plans.len(), 1);

        // The stale source must also leave no failed material-key poison when
        // it appears before the valid source in stable frame order.
        let mut reversed_lists = RenderLists::new();
        let stale_first = reversed_lists
            .intern_frame_source(super::super::frame::ExtractedSourceId::from_index(1));
        let valid_second = reversed_lists
            .intern_frame_source(super::super::frame::ExtractedSourceId::from_index(0));
        let mut reversed_geometry = super::super::asset_frame::GeometryFrameStore::new();
        reversed_geometry.capture_visible(&assets, &extracted_scene, &reversed_lists);
        let mut reversed_store = MaterialPlanStore::new();
        let _reversed_asset_textures = capture_test_plans(
            &mut reversed_store,
            &assets,
            &extracted_scene,
            &reversed_lists,
            reversed_geometry.eligibility(&extracted_scene, &reversed_lists),
        );
        assert!(reversed_store.source_plans[stale_first.index()].is_none());
        assert!(reversed_store.source_plans[valid_second.index()].is_some());
        assert_eq!(reversed_store.plans.len(), 1);
    }

    #[test]
    fn stale_only_textured_source_emits_no_opaque_texture_topology() {
        let assets = AssetServer::new();
        let geometry = assets.geometries.add(Geometry::new());
        let (_, geometry_version, geometry_identity) = assets
            .geometries
            .get_entry_with_identity(geometry)
            .expect("geometry missing");
        let image = assets.images.add(image_8x8());
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("stale geometry texture"), image));
        let material = assets.materials.add(unlit_with_texture(texture));
        let (material_asset, material_version, material_identity) = assets
            .materials
            .get_entry_with_identity(material)
            .expect("material missing");
        let data_version = material_asset.data.version();
        drop(material_asset);

        assets.geometries.insert_ready(geometry, Geometry::new());
        let mut extracted_scene = ExtractedScene::new();
        extracted_scene.render_items.push(extracted_item(
            geometry,
            geometry_version,
            geometry_identity,
            material,
            material_version,
            material_identity,
            data_version,
        ));
        let mut render_lists = RenderLists::new();
        let source =
            render_lists.intern_frame_source(super::super::frame::ExtractedSourceId::from_index(0));
        render_lists.insert_opaque(super::super::frame::RenderIntent {
            source,
            geometry_handle: geometry,
            material_handle: material,
            material_index: 0,
            distance_sq: 0.0,
        });

        let mut geometry_frames = super::super::asset_frame::GeometryFrameStore::new();
        geometry_frames.capture_visible(&assets, &extracted_scene, &render_lists);
        let mut store = MaterialPlanStore::new();
        let _asset_textures = capture_test_plans(
            &mut store,
            &assets,
            &extracted_scene,
            &render_lists,
            geometry_frames.eligibility(&extracted_scene, &render_lists),
        );
        let recipes = store.take_topology_recipes();
        let opaque = recipes
            .iter()
            .find(|recipe| matches!(recipe.key, DrawPlanKey::Opaque))
            .expect("opaque topology recipe missing");

        assert!(store.source_plans[source.index()].is_none());
        assert!(store.plans.is_empty());
        assert!(
            opaque.textures.is_empty(),
            "a stale-only source must not register Asset Texture upload or mipmap topology"
        );
    }

    #[test]
    fn pending_material_plan_rejects_same_handle_texture_aba() {
        let assets = AssetServer::new();
        let first_missing = ImageHandle::from(slotmap::KeyData::from_ffi(0x301));
        let second_missing = ImageHandle::from(slotmap::KeyData::from_ffi(0x302));
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("pending A"), first_missing));
        let material = assets.materials.add(unlit_with_texture(texture));
        let (material_owner, record) = capture_record(&assets, material);
        assert!(record.textures[0].asset.unwrap().ready.is_none());
        assert!(material_plan_is_current(&assets, &material_owner, &record));

        assets
            .textures
            .insert_ready(texture, Texture::new_2d(Some("pending B"), second_missing));
        assert!(
            !material_plan_is_current(&assets, &material_owner, &record),
            "two unavailable states with different exact identities must not compare equal"
        );
    }

    #[test]
    fn pipeline_snapshot_never_mixes_with_a_later_material_generation() {
        let assets = AssetServer::new();
        let material = assets.materials.add(Material::new_unlit(glam::Vec4::ONE));
        let (material_owner, record) = capture_record(&assets, material);
        assert_eq!(record.pipeline.settings().alpha_mode, AlphaMode::Opaque);

        let current = assets.materials.get(material).expect("material missing");
        current
            .as_unlit()
            .expect("expected unlit material")
            .set_alpha_mode(AlphaMode::Blend);

        assert_eq!(
            record.pipeline.settings().alpha_mode,
            AlphaMode::Opaque,
            "the representative is immutable even when the live asset mutates"
        );
        assert!(
            !material_plan_is_current(&assets, &material_owner, &record),
            "a mixed snapshot must be rejected instead of being published"
        );
    }

    #[test]
    fn planned_material_consumes_graph_binding_without_legacy_mip_queue() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resource_manager = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image = assets.images.add(image_8x8());
        let mut texture_asset = Texture::new_2d(Some("planned mipped texture"), image);
        texture_asset.generate_mipmaps = true;
        texture_asset.sampler.mag_filter = wgpu::FilterMode::Nearest;
        let texture = assets.textures.add(texture_asset);
        let material = assets.materials.add(unlit_with_texture(texture));
        let (material_owner, record) = capture_record(&assets, material);
        let snapshot = record.textures[0]
            .asset
            .and_then(|state| state.ready)
            .expect("ready texture snapshot missing");

        let materialization =
            resource_manager.materialize_asset_texture_for_graph(&assets, snapshot);
        assert!(materialization.texture.is_some());
        assert!(materialization.mipmap_request.is_some());
        assert!(resource_manager.take_mipmap_requests().is_empty());
        assert!(resource_manager.prepare_planned_material(
            &assets,
            record.key.handle,
            &material_owner,
            record.key.asset_version,
            record.key.asset_identity,
            record.key.material_version,
            &record.textures,
        ));
        assert!(resource_manager.get_material(material).is_some());
        assert!(
            resource_manager.take_mipmap_requests().is_empty(),
            "planned BindGroup rebuild rediscovered a legacy texture producer"
        );
    }
}
