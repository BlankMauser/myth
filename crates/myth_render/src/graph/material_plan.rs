//! Graph-attempt ownership and exact snapshots for central draw materials.
//!
//! The store is the CPU-side bridge between draw-intent collection and
//! post-DPE physical preparation. It owns exact `Arc<Material>` values because
//! the frame arena deliberately skips Drop, while render passes and DrawPlans
//! retain only POD identities and real RDG texture reads.

use std::sync::Arc;

use myth_assets::{AssetServer, ImageHandle, MaterialHandle, TextureHandle};
use myth_resources::material::{Material, RenderableMaterialTrait};
use myth_resources::texture::TextureSource;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::core::ResourceManager;
use crate::core::gpu::{
    AssetTextureSnapshot, PlannedMaterialTexture, snapshot_asset_texture_state,
};

use super::extracted::{ExtractedRenderItem, ExtractedScene};
use super::frame::{DrawPlanKey, RenderLists};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MaterialPlanId(u32);

impl MaterialPlanId {
    #[inline]
    fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("material plan index exceeds u32"))
    }

    #[inline]
    const fn index(self) -> usize {
        self.0 as usize
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

struct MaterialPlanRecord {
    key: MaterialPlanKey,
    material: Arc<Material>,
    textures: SmallVec<[PlannedMaterialTexture; 8]>,
    /// Success gate for this graph attempt. A persistent `GpuMaterial` with
    /// matching material stamps is insufficient when an Image changed and
    /// this attempt's exact texture materialization failed.
    prepared: bool,
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
    plans: Vec<MaterialPlanRecord>,
    lookup: FxHashMap<MaterialPlanKey, MaterialPlanId>,
    source_plans: Vec<Option<MaterialPlanId>>,
    recipes: Vec<DrawMaterialTextureRecipe>,
    initialized: bool,
}

impl MaterialPlanStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            plans: Vec::with_capacity(128),
            lookup: FxHashMap::default(),
            source_plans: Vec::with_capacity(512),
            recipes: Vec::with_capacity(16),
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
        debug_assert!(self.source_plans.is_empty());
        debug_assert!(self.recipes.is_empty());

        self.source_plans
            .resize_with(extracted_scene.render_item_count(), || None);
        let visible_sources = render_lists.all_source_indices();
        for source_item_index in visible_sources {
            self.capture_source(assets, extracted_scene, source_item_index);
        }

        self.push_recipe(
            DrawPlanKey::Opaque,
            render_lists
                .opaque
                .iter()
                .map(|intent| intent.source_item_index),
        );
        self.push_recipe(
            DrawPlanKey::Transparent,
            render_lists
                .transparent
                .iter()
                .map(|intent| intent.source_item_index),
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
                    .map(|intent| intent.source_item_index),
            );
        }

        self.initialized = true;
    }

    fn capture_source(
        &mut self,
        assets: &AssetServer,
        extracted_scene: &ExtractedScene,
        source_item_index: u32,
    ) {
        let Some(item) = extracted_scene.render_items.get(source_item_index as usize) else {
            return;
        };
        let key = MaterialPlanKey::from_item(item);
        if let Some(&id) = self.lookup.get(&key) {
            // The exact record was validated when first captured and is
            // revalidated again before post-DPE preparation. Rechecking every
            // source here would turn a shared material into O(draw-count ×
            // texture-count) AssetStorage traffic with no additional safety.
            self.source_plans[source_item_index as usize] = Some(id);
            return;
        }

        let Some((material, asset_version, asset_identity)) =
            assets.materials.get_entry_with_identity(key.handle)
        else {
            return;
        };
        if (asset_version, asset_identity) != (key.asset_version, key.asset_identity)
            || material.data.version() != key.material_version
        {
            return;
        }

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
        let record = MaterialPlanRecord {
            key,
            material,
            textures,
            prepared: false,
        };
        if !material_plan_is_current(assets, &record) {
            return;
        }

        let id = MaterialPlanId::from_index(self.plans.len());
        self.plans.push(record);
        assert!(self.lookup.insert(key, id).is_none());
        self.source_plans[source_item_index as usize] = Some(id);
    }

    fn push_recipe(&mut self, key: DrawPlanKey, source_indices: impl IntoIterator<Item = u32>) {
        let mut visited_materials = FxHashSet::default();
        let mut visited_textures = FxHashSet::default();
        let mut textures = Vec::new();
        for source_item_index in source_indices {
            let Some(plan) = self
                .source_plans
                .get(source_item_index as usize)
                .copied()
                .flatten()
            else {
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
        live_source_indices: &[u32],
    ) {
        assert!(self.initialized, "material plans were not captured");
        let mut visited = FxHashSet::default();
        for &source_item_index in live_source_indices {
            let Some(id) = self
                .source_plans
                .get(source_item_index as usize)
                .copied()
                .flatten()
            else {
                continue;
            };
            if !visited.insert(id) {
                continue;
            }
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
                self.plans[id.index()].prepared = true;
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
        self.plans.clear();
        self.initialized = false;
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.plans.is_empty()
            && self.lookup.is_empty()
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
        source_item_index: u32,
        item: &ExtractedRenderItem,
    ) -> Option<&'a Arc<Material>> {
        let id = self
            .store
            .source_plans
            .get(source_item_index as usize)
            .copied()
            .flatten()?;
        let plan = self.store.plans.get(id.index())?;
        (plan.prepared && plan.key.matches_item(item)).then_some(&plan.material)
    }
}

fn material_plan_is_current(assets: &AssetServer, record: &MaterialPlanRecord) -> bool {
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
    use myth_resources::material::{MaterialType, UnlitMaterial};
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

    fn capture_record(assets: &AssetServer, handle: MaterialHandle) -> MaterialPlanRecord {
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
        MaterialPlanRecord {
            key: MaterialPlanKey {
                handle,
                asset_version,
                asset_identity,
                material_version,
            },
            material,
            textures,
            prepared: false,
        }
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
