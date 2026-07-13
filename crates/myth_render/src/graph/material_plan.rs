//! Graph-attempt ownership and exact snapshots for central draw materials.
//!
//! The store is the CPU-side bridge between draw-intent collection and
//! post-DPE physical preparation. It owns exact `Arc<Material>` values because
//! the frame arena deliberately skips Drop, while render passes and DrawPlans
//! retain only POD identities and real RDG texture reads.

use std::{num::NonZeroU32, sync::Arc};

use myth_assets::{AssetServer, ImageHandle, MaterialHandle, TextureHandle};
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

use super::capacity::{
    CapacityUsage, estimated_hash_map_footprint, estimated_hash_set_footprint,
    shrink_hash_map_by_family_ratio, shrink_hash_set_by_family_ratio, shrink_vec_by_family_ratio,
    vec_footprint,
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
    material: Arc<Material>,
    pipeline: MaterialPipelineSnapshot,
    textures: SmallVec<[PlannedMaterialTexture; 8]>,
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
    pub(crate) textures: Vec<AssetTextureSnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AssetTextureKey {
    texture_handle: TextureHandle,
    image_handle: ImageHandle,
    texture_version: u32,
    texture_identity: u64,
    image_version: u32,
    image_identity: u64,
    logical_version: u64,
}

impl From<AssetTextureSnapshot> for AssetTextureKey {
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

/// Destructor-aware material owners and source joins for one graph attempt.
pub(crate) struct MaterialPlanStore {
    plans: Vec<MaterialFrameRecord>,
    lookup: FxHashMap<MaterialPlanKey, MaterialFrameId>,
    failed: FxHashSet<MaterialPlanKey>,
    /// Dense join indexed exclusively by the intent-visible frame domain.
    /// ExtractedScene may be much larger after view culling.
    source_plans: Vec<Option<MaterialFrameId>>,
    recipes: Vec<DrawMaterialTextureRecipe>,
    initialized: bool,
}

const MATERIAL_PLAN_BASELINE: usize = 128;
const MATERIAL_SOURCE_BASELINE: usize = 512;
const MATERIAL_RECIPE_BASELINE: usize = 16;

impl MaterialPlanStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            plans: Vec::with_capacity(MATERIAL_PLAN_BASELINE),
            lookup: FxHashMap::default(),
            failed: FxHashSet::default(),
            source_plans: Vec::with_capacity(MATERIAL_SOURCE_BASELINE),
            recipes: Vec::with_capacity(MATERIAL_RECIPE_BASELINE),
            initialized: false,
        }
    }

    /// Capture exact material owners and complete texture states for the
    /// CPU-visible draw-intent union. This performs no GPU interaction.
    pub(crate) fn capture(
        &mut self,
        assets: &AssetServer,
        extracted_scene: &ExtractedScene,
        render_lists: &RenderLists,
    ) {
        assert!(
            !self.initialized,
            "material plans may be captured only once per graph attempt"
        );
        debug_assert!(self.plans.is_empty());
        debug_assert!(self.lookup.is_empty());
        debug_assert!(self.failed.is_empty());
        debug_assert!(self.source_plans.is_empty());
        debug_assert!(self.recipes.is_empty());

        self.source_plans
            .resize_with(render_lists.frame_source_count(), || None);
        for source in render_lists.frame_source_ids() {
            self.capture_source(assets, extracted_scene, render_lists, source);
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

    fn capture_source(
        &mut self,
        assets: &AssetServer,
        extracted_scene: &ExtractedScene,
        render_lists: &RenderLists,
        source: FrameSourceId,
    ) {
        let Some(extracted_source) = render_lists.extracted_source(source) else {
            return;
        };
        let Some(item) = extracted_scene.render_items.get(extracted_source.index()) else {
            return;
        };
        let key = MaterialPlanKey::from_item(item);
        if let Some(&id) = self.lookup.get(&key) {
            // The exact record was validated when first captured and is
            // revalidated again before post-DPE preparation. Rechecking every
            // source here would turn a shared material into O(draw-count ×
            // texture-count) AssetStorage traffic with no additional safety.
            self.source_plans[source.index()] = Some(id);
            return;
        }
        if self.failed.contains(&key) {
            return;
        }

        let Some((material, asset_version, asset_identity)) =
            assets.materials.get_entry_with_identity(key.handle)
        else {
            self.failed.insert(key);
            return;
        };
        if (asset_version, asset_identity) != (key.asset_version, key.asset_identity)
            || material.data.version() != key.material_version
        {
            self.failed.insert(key);
            return;
        }

        let pipeline = MaterialPipelineSnapshot::capture(&material);
        let mut textures = SmallVec::<[PlannedMaterialTexture; 8]>::new();
        material.data.visit_textures(&mut |source| {
            textures.push(PlannedMaterialTexture {
                source: *source,
                asset: match *source {
                    TextureSource::Asset(handle) => {
                        Some(snapshot_asset_texture_state(assets, handle))
                    }
                    TextureSource::Attachment(_, _) => None,
                },
            });
        });
        // Every field above belongs to the same serial capture window. A
        // mutation observed during capture rejects this attempt instead of
        // publishing a mixed classification/shader/binding recipe.
        if material.data.version() != key.material_version {
            self.failed.insert(key);
            return;
        }
        let record = MaterialFrameRecord {
            key,
            material,
            pipeline,
            textures,
            prepared: false,
            attempted: false,
            interface: None,
        };
        if !material_plan_is_current(assets, &record) {
            self.failed.insert(key);
            return;
        }

        let id = MaterialFrameId::from_index(self.plans.len());
        self.plans.push(record);
        assert!(self.lookup.insert(key, id).is_none());
        self.source_plans[source.index()] = Some(id);
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
            for texture in &self.plans[plan.index()].textures {
                let Some(snapshot) = texture.asset.and_then(|state| state.ready) else {
                    continue;
                };
                if visited_textures.insert(AssetTextureKey::from(snapshot)) {
                    textures.push(snapshot);
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
        for &source in live_sources {
            let Some(id) = self.source_plans.get(source.index()).copied().flatten() else {
                continue;
            };
            if self.plans[id.index()].attempted {
                continue;
            }
            self.plans[id.index()].attempted = true;
            let prepared = {
                let record = &self.plans[id.index()];
                resource_manager.prepare_planned_material(
                    assets,
                    record.key.handle,
                    &record.material,
                    record.key.asset_version,
                    record.key.asset_identity,
                    record.key.material_version,
                    &record.textures,
                )
            };
            if prepared && material_plan_is_current(assets, &self.plans[id.index()]) {
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
    }

    #[must_use]
    pub(crate) const fn resolver(&self) -> MaterialPlanResolver<'_> {
        MaterialPlanResolver { store: self }
    }

    pub(crate) fn clear(&mut self) {
        self.recipes.clear();
        self.source_plans.clear();
        self.lookup.clear();
        self.failed.clear();
        self.plans.clear();
        self.initialized = false;
    }

    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.plans, MATERIAL_PLAN_BASELINE)
            + estimated_hash_map_footprint(&self.lookup, MATERIAL_PLAN_BASELINE)
            + estimated_hash_set_footprint(&self.failed, MATERIAL_PLAN_BASELINE)
            + vec_footprint(&self.source_plans, MATERIAL_SOURCE_BASELINE)
            + vec_footprint(&self.recipes, MATERIAL_RECIPE_BASELINE)
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        shrink_vec_by_family_ratio(
            &mut self.plans,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.lookup,
            MATERIAL_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_set_by_family_ratio(
            &mut self.failed,
            MATERIAL_PLAN_BASELINE,
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
            &mut self.recipes,
            MATERIAL_RECIPE_BASELINE,
            target_bytes,
            retained_bytes,
        );
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.plans.is_empty()
            && self.lookup.is_empty()
            && self.failed.is_empty()
            && self.source_plans.is_empty()
            && self.recipes.is_empty()
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
        (plan.prepared && plan.interface.is_some() && plan.key.matches_item(item))
            .then_some(PreparedMaterialFrame { id, record: plan })
    }

    #[must_use]
    pub(crate) fn material(self, id: MaterialFrameId) -> Option<PreparedMaterialFrame<'a>> {
        let record = self.store.plans.get(id.index())?;
        (record.prepared && record.interface.is_some())
            .then_some(PreparedMaterialFrame { id, record })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PreparedMaterialFrame<'a> {
    id: MaterialFrameId,
    record: &'a MaterialFrameRecord,
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
        self.record.material.data.version() == self.record.key.material_version
    }
}

fn material_plan_is_current(assets: &AssetServer, record: &MaterialFrameRecord) -> bool {
    let Some((material, asset_version, asset_identity)) =
        assets.materials.get_entry_with_identity(record.key.handle)
    else {
        return false;
    };
    if (asset_version, asset_identity) != (record.key.asset_version, record.key.asset_identity)
        || !Arc::ptr_eq(&material, &record.material)
        || record.material.data.version() != record.key.material_version
    {
        return false;
    }

    let mut texture_index = 0usize;
    let mut matches = true;
    record.material.data.visit_textures(&mut |source| {
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
        && record.material.data.version() == record.key.material_version
        && assets
            .materials
            .get_entry_with_identity(record.key.handle)
            .is_some_and(|(current, version, identity)| {
                version == record.key.asset_version
                    && identity == record.key.asset_identity
                    && Arc::ptr_eq(&current, &record.material)
            })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WgpuContext;
    use crate::settings::{RendererInitConfig, RendererSettings};
    use myth_resources::image::{Image, ImageDimension, PixelFormat};
    use myth_resources::material::{AlphaMode, MaterialType, UnlitMaterial};
    use myth_resources::texture::Texture;

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

    fn capture_record(assets: &AssetServer, handle: MaterialHandle) -> MaterialFrameRecord {
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
        MaterialFrameRecord {
            key: MaterialPlanKey {
                handle,
                asset_version,
                asset_identity,
                material_version,
            },
            material,
            pipeline,
            textures,
            prepared: false,
            attempted: false,
            interface: None,
        }
    }

    #[test]
    fn material_frame_id_uses_the_nonzero_option_niche() {
        assert_eq!(std::mem::size_of::<MaterialFrameId>(), 4);
        assert_eq!(std::mem::size_of::<Option<MaterialFrameId>>(), 4);
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
    fn source_plan_join_uses_only_the_visible_frame_domain() {
        let assets = AssetServer::new();
        let extracted_scene = ExtractedScene::new();
        let mut render_lists = RenderLists::new();
        render_lists.intern_frame_source(super::super::frame::ExtractedSourceId::from_index(7));
        render_lists.intern_frame_source(super::super::frame::ExtractedSourceId::from_index(
            1_000_000,
        ));

        let mut store = MaterialPlanStore::new();
        store.capture(&assets, &extracted_scene, &render_lists);

        assert_eq!(render_lists.frame_source_count(), 2);
        assert_eq!(store.source_plans.len(), 2);
        assert!(store.source_plans.iter().all(Option::is_none));
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
        let record = capture_record(&assets, material);
        assert!(record.textures[0].asset.unwrap().ready.is_none());
        assert!(material_plan_is_current(&assets, &record));

        assets
            .textures
            .insert_ready(texture, Texture::new_2d(Some("pending B"), second_missing));
        assert!(
            !material_plan_is_current(&assets, &record),
            "two unavailable states with different exact identities must not compare equal"
        );
    }

    #[test]
    fn pipeline_snapshot_never_mixes_with_a_later_material_generation() {
        let assets = AssetServer::new();
        let material = assets.materials.add(Material::new_unlit(glam::Vec4::ONE));
        let record = capture_record(&assets, material);
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
            !material_plan_is_current(&assets, &record),
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
        let record = capture_record(&assets, material);
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
            &record.material,
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
