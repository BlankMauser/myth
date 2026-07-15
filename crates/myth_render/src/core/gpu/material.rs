//! Material operations
//!
//! Uses an "Ensure -> Check -> Rebuild" pattern:
//! 1. Ensure all GPU resources exist with up-to-date data
//! 2. Compare physical resource IDs for changes (decide `BindGroup` rebuild)
//! 3. Update material version number (for Pipeline cache)
//!
//! # Three-dimensional version tracking separation
//!
//! 1. **Resource topology (`BindGroup`)**: Tracked by `ResourceIdSet`
//!    - Texture/sampler/Buffer ID changes -> rebuild `BindGroup`
//!
//! 2. **Resource content (Buffer Data)**: Tracked by `BufferRef`
//!    - Atomic version number changes -> upload Buffer
//!
//! 3. **Pipeline state (`RenderPipeline`)**: Tracked by `Material.version()`
//!    - Depth write/transparency/double-sided rendering changes -> switch Pipeline

use std::{mem::size_of, num::NonZeroU32, sync::Arc};

use crate::core::gpu::texture::ResourceState;
use crate::core::gpu::{AssetTexturePhysicalBinding, AssetTextureStateSnapshot, EnsureResult};
use myth_assets::{AssetServer, MaterialHandle};
use myth_resources::material::{
    CanonicalMaterialBindingStamp, Material, MaterialSettings, RenderableMaterialTrait,
    ShaderTemplateMode,
};
use myth_resources::shader_defines::ShaderDefines;

use myth_resources::texture::TextureSource;
use myth_resources::{BindingDesc, BindingResource, ResourceBuilder};

use super::{ResourceIdSet, ResourceManager, hash_layout_entries};

/// Ordered texture state captured for one graph-attempt material plan.
///
/// Asset entries retain both ready and unavailable CPU stamps. Attachments
/// carry no Asset snapshot but keep their exact logical source identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlannedMaterialTexture {
    pub(crate) source: TextureSource,
    pub(crate) asset: Option<AssetTextureStateSnapshot>,
    pub(crate) physical: Option<AssetTexturePhysicalBinding>,
}

/// Immutable shader/raster state for one exact Material generation.
///
/// The snapshot is derived only after DPE selects a live material. Once its
/// exact `GpuMaterial` is published, the persistent cache owns this value so
/// warm frames reuse it without rebuilding `ShaderDefines` during graph
/// construction.
pub(crate) struct MaterialPipelineSnapshot {
    settings: MaterialSettings,
    use_transmission: bool,
    shader_name: &'static str,
    shader_template: Option<&'static str>,
    shader_template_mode: ShaderTemplateMode,
    shader_defines: ShaderDefines,
}

impl MaterialPipelineSnapshot {
    #[must_use]
    pub(crate) fn capture(material: &Material) -> Self {
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

const PLANNED_MATERIAL_CANDIDATE_BASELINE: usize = 128;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PlannedMaterialCandidateId(NonZeroU32);

impl PlannedMaterialCandidateId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("planned material candidate index exceeds u32::MAX - 1");
        Self(raw)
    }

    #[inline]
    pub(crate) const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

#[derive(Clone, Copy)]
enum PlannedMaterialCandidateSource {
    Unavailable,
    Existing {
        expected_bind_group_id: u64,
    },
    Rebuild {
        observed_bind_group_id: Option<u64>,
        private_slot: usize,
    },
}

struct PlannedMaterialCandidate {
    handle: MaterialHandle,
    asset_version: u32,
    asset_identity: u64,
    material_version: u64,
    has_pending_textures: bool,
    canonical_binding_stamp: Option<CanonicalMaterialBindingStamp>,
    source: PlannedMaterialCandidateSource,
    result: Option<bool>,
    #[cfg(test)]
    canonical_fast_path: bool,
}

/// Graph-attempt owner for unpublished persistent Material candidates.
///
/// Candidate records stay compact; destructor-bearing `GpuMaterial` values
/// live in one retained side table and enter the persistent cache only after
/// the graph coordinator's final batched postflight succeeds.
pub(crate) struct PlannedMaterialCandidateBatch {
    candidates: Vec<PlannedMaterialCandidate>,
    private_materials: Vec<Option<Box<GpuMaterial>>>,
    published_count: usize,
}

impl PlannedMaterialCandidateBatch {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            candidates: Vec::with_capacity(PLANNED_MATERIAL_CANDIDATE_BASELINE),
            private_materials: Vec::with_capacity(PLANNED_MATERIAL_CANDIDATE_BASELINE),
            published_count: 0,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.candidates.is_empty() && self.private_materials.is_empty() && self.published_count == 0
    }

    #[cfg(test)]
    pub(crate) fn result(&self, id: PlannedMaterialCandidateId) -> bool {
        self.candidates
            .get(id.index())
            .and_then(|candidate| candidate.result)
            .expect("planned material candidate is unpublished or invalid")
    }

    #[inline]
    #[must_use]
    pub(crate) fn all_published(&self) -> bool {
        self.published_count == self.candidates.len()
    }

    #[inline]
    #[must_use]
    pub(crate) fn canonical_binding_stamp(
        &self,
        id: PlannedMaterialCandidateId,
    ) -> Option<CanonicalMaterialBindingStamp> {
        self.candidates
            .get(id.index())
            .and_then(|candidate| candidate.canonical_binding_stamp)
    }

    #[cfg(test)]
    pub(crate) fn reuses_existing(&self, id: PlannedMaterialCandidateId) -> bool {
        self.candidates.get(id.index()).is_some_and(|candidate| {
            matches!(
                candidate.source,
                PlannedMaterialCandidateSource::Existing { .. }
            )
        })
    }

    #[cfg(test)]
    pub(crate) fn used_canonical_fast_path(&self, id: PlannedMaterialCandidateId) -> bool {
        self.candidates
            .get(id.index())
            .is_some_and(|candidate| candidate.canonical_fast_path)
    }

    pub(crate) fn clear(&mut self) {
        self.private_materials.clear();
        self.candidates.clear();
        self.published_count = 0;
    }

    #[inline]
    #[must_use]
    pub(crate) fn active_bytes(&self) -> usize {
        self.candidates
            .len()
            .saturating_mul(size_of::<PlannedMaterialCandidate>())
            .saturating_add(
                self.private_materials
                    .len()
                    .saturating_mul(size_of::<Option<Box<GpuMaterial>>>()),
            )
            .saturating_add(self.private_owner_bytes())
    }

    #[inline]
    #[must_use]
    pub(crate) fn retained_bytes(&self) -> usize {
        self.candidates
            .capacity()
            .saturating_mul(size_of::<PlannedMaterialCandidate>())
            .saturating_add(
                self.private_materials
                    .capacity()
                    .saturating_mul(size_of::<Option<Box<GpuMaterial>>>()),
            )
            .saturating_add(self.private_owner_bytes())
    }

    #[inline]
    #[must_use]
    pub(crate) const fn baseline_bytes() -> usize {
        PLANNED_MATERIAL_CANDIDATE_BASELINE.saturating_mul(
            size_of::<PlannedMaterialCandidate>() + size_of::<Option<Box<GpuMaterial>>>(),
        )
    }

    #[inline]
    fn private_owner_bytes(&self) -> usize {
        self.private_materials
            .iter()
            .filter(|owner| owner.is_some())
            .count()
            .saturating_mul(size_of::<GpuMaterial>())
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        assert!(
            self.is_empty(),
            "live material candidates cannot be trimmed"
        );
        if retained_bytes == 0 || target_bytes >= retained_bytes {
            return;
        }
        let target = |capacity: usize| {
            let proportional = ((capacity as u128) * (target_bytes as u128))
                .div_ceil(retained_bytes as u128)
                .min(usize::MAX as u128) as usize;
            proportional
                .max(PLANNED_MATERIAL_CANDIDATE_BASELINE)
                .min(capacity)
        };
        self.candidates
            .shrink_to(target(self.candidates.capacity()));
        self.private_materials
            .shrink_to(target(self.private_materials.capacity()));
    }

    fn push(
        &mut self,
        handle: MaterialHandle,
        asset_version: u32,
        asset_identity: u64,
        material_version: u64,
        has_pending_textures: bool,
        source: PlannedMaterialCandidateSource,
    ) -> PlannedMaterialCandidateId {
        self.push_with_stamp(
            handle,
            asset_version,
            asset_identity,
            material_version,
            has_pending_textures,
            None,
            source,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn push_with_stamp(
        &mut self,
        handle: MaterialHandle,
        asset_version: u32,
        asset_identity: u64,
        material_version: u64,
        has_pending_textures: bool,
        canonical_binding_stamp: Option<CanonicalMaterialBindingStamp>,
        source: PlannedMaterialCandidateSource,
    ) -> PlannedMaterialCandidateId {
        assert!(
            self.published_count == 0,
            "planned material candidates cannot be staged after publication"
        );
        let id = PlannedMaterialCandidateId::from_index(self.candidates.len());
        self.candidates.push(PlannedMaterialCandidate {
            handle,
            asset_version,
            asset_identity,
            material_version,
            has_pending_textures,
            canonical_binding_stamp,
            source,
            result: None,
            #[cfg(test)]
            canonical_fast_path: false,
        });
        id
    }

    fn push_unavailable(
        &mut self,
        handle: MaterialHandle,
        asset_version: u32,
        asset_identity: u64,
        material_version: u64,
    ) -> PlannedMaterialCandidateId {
        self.push(
            handle,
            asset_version,
            asset_identity,
            material_version,
            false,
            PlannedMaterialCandidateSource::Unavailable,
        )
    }
}

/// GPU-side material resource
///
/// Uses resource ID tracking for automatic change detection
///
/// # Three-dimensional version tracking separation
///
/// 1. **Resource topology (`BindGroup`)**: Tracked by `resource_ids`
///    - Texture/sampler/Buffer ID changes -> rebuild `BindGroup`
///
/// 2. **Resource content (Buffer Data)**: Tracked by `BufferRef` (external)
///    - Atomic version number changes -> upload Buffer
///
/// 3. **Pipeline state (`RenderPipeline`)**: Tracked by `version`
///    - Depth write/transparency/double-sided rendering changes -> switch Pipeline
pub struct GpuMaterial {
    pub asset_version: u32,
    pub asset_identity: u64,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    /// Diagnostic hash of the canonical layout entries. Exact entry equality,
    /// not this hash, determines layout cache identity.
    pub layout_hash: u64,
    pub binding_wgsl: Arc<str>,
    pub binding_interface_id: super::BindingInterfaceId,
    /// Coherent pipeline state derived for the same exact Material epoch as
    /// this physical Group 1 entry.
    pub(crate) pipeline_snapshot: Arc<MaterialPipelineSnapshot>,
    /// Exact generated binding stamp last accepted by the cache-side CAS.
    /// `None` keeps hand-written custom materials on the generic discovery
    /// path.
    pub(crate) canonical_binding_stamp: Option<CanonicalMaterialBindingStamp>,
    /// Set of physical IDs of all dependent resources (guards `BindGroup` validity)
    pub resource_ids: ResourceIdSet,
    /// Records the Material version when this `GpuMaterial` was generated (for Pipeline cache)
    pub version: u64,
    pub last_used_frame: u64,
    pub last_verified_frame: u64,
}

impl ResourceManager {
    /// Prepare Material GPU resources
    ///
    /// Three-dimensional change detection:
    /// - Resource topology change -> rebuild BindGroup (detected by `ResourceIdSet`)
    /// - Resource content change -> upload Buffer (handled automatically by `BufferRef`)
    /// - Pipeline state change -> switch Pipeline (recorded by version, used externally)
    ///
    /// # Lagging Synchronization
    ///
    /// When any dependent texture is still in `Loading` state and a valid
    /// `BindGroup` already exists, the rebuild is **deferred** — the previous
    /// frame's GPU state is retained so that no visual flicker occurs.
    /// `last_verified_frame` is intentionally *not* updated, ensuring the
    /// material is re-checked on the next frame.
    pub(crate) fn prepare_material(&mut self, assets: &AssetServer, handle: MaterialHandle) {
        let Some((material, asset_version, asset_identity)) =
            assets.materials.get_entry_with_identity(handle)
        else {
            return;
        };

        // [Fast Path] Per-frame cache check
        if let Some(gpu_mat) = self.resources.material(handle)
            && gpu_mat.last_verified_frame == self.frame_index
            && gpu_mat.asset_version == asset_version
            && gpu_mat.asset_identity == asset_identity
            && gpu_mat.version == material.data.version()
        {
            return;
        }

        // 1. Ensure phase: ensure all resources exist with up-to-date data, collect physical resource IDs
        let (mut current_resource_ids, has_pending_textures) =
            self.ensure_material_resources(assets, &material);

        let needs_rebuild_bindgroup = if let Some(gpu_mat) = self.resources.material(handle) {
            let same_asset =
                gpu_mat.asset_version == asset_version && gpu_mat.asset_identity == asset_identity;
            let same_pipeline_state = gpu_mat.version == material.data.version();
            if has_pending_textures && same_asset && same_pipeline_state {
                // Retaining the old physical group is valid only while the
                // exact material's shader/binding topology is unchanged.
                // A version change must rebuild against fallback textures.
                false
            } else if !same_asset || !same_pipeline_state {
                true
            } else {
                let mut cached_ids = gpu_mat.resource_ids.clone();
                !current_resource_ids.matches(&mut cached_ids)
            }
        } else {
            true
        };

        if needs_rebuild_bindgroup {
            // 3. Rebuild phase: rebuild BindGroup (expensive operation)
            self.rebuild_material_bindgroup(
                assets,
                handle,
                &material,
                current_resource_ids,
                asset_version,
                asset_identity,
            );
        }

        // 4. Update version number and frame counter (very fast operation)
        if let Some(gpu_mat) = self.resources.material_mut(handle) {
            gpu_mat.version = material.data.version();
            gpu_mat.asset_version = asset_version;
            gpu_mat.asset_identity = asset_identity;
            gpu_mat.last_used_frame = self.frame_index;
            // If textures are pending (first-time creation with fallbacks),
            // do not mark as verified so we re-check next frame.
            if !has_pending_textures {
                gpu_mat.last_verified_frame = self.frame_index;
            }
        }
    }

    /// Build one unpublished exact Material candidate from retained CPU state.
    ///
    /// AssetStorage currency is owned by the graph coordinator's batched
    /// pre/postflights. This method performs no asset lookup and never inserts
    /// into the persistent Material cache. Ready Asset Textures must already
    /// have published an exact [`TextureBinding`](super::TextureBinding); an
    /// unavailable dependency retains the legacy lagging/fallback semantics.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_planned_material_candidate(
        &mut self,
        batch: &mut PlannedMaterialCandidateBatch,
        handle: MaterialHandle,
        material: &Material,
        asset_version: u32,
        asset_identity: u64,
        material_version: u64,
        pipeline_snapshot: &Arc<MaterialPipelineSnapshot>,
        textures: &[PlannedMaterialTexture],
    ) -> PlannedMaterialCandidateId {
        if material.data.version() != material_version {
            return batch.push_unavailable(handle, asset_version, asset_identity, material_version);
        }

        let canonical_binding_stamp = material
            .data
            .canonical_binding_stamp()
            .filter(|stamp| stamp.topology_epoch == material_version);
        if let Some(stamp) = canonical_binding_stamp
            && let Some(candidate) = self.try_prepare_exact_canonical_material(
                batch,
                handle,
                asset_version,
                asset_identity,
                material_version,
                stamp,
                textures,
            )
        {
            return candidate;
        }

        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);
        if !binding_recipe_matches_plan(&builder, textures)
            || material.data.version() != material_version
        {
            return batch.push_unavailable(handle, asset_version, asset_identity, material_version);
        }

        // Upload the canonical material uniform before the buffer-only recipe
        // preparation below. Custom bindings with inline bytes are handled by
        // `prepare_binding_buffers` from the same coherent builder.
        material.data.with_uniform_bytes(&mut |bytes| {
            self.ensure_buffer_ref(&material.data.uniform_buffer(), bytes);
        });
        self.prepare_binding_buffers(&builder.bindings);
        if material.data.version() != material_version {
            return batch.push_unavailable(handle, asset_version, asset_identity, material_version);
        }

        let has_pending_textures = self.planned_material_has_pending(textures);
        if has_pending_textures
            && self.resources.material(handle).is_some_and(|gpu_mat| {
                gpu_mat.asset_version == asset_version
                    && gpu_mat.asset_identity == asset_identity
                    && gpu_mat.version == material_version
            })
        {
            let expected_bind_group_id = self
                .resources
                .material(handle)
                .expect("exact lagging Material disappeared")
                .bind_group_id;
            return batch.push_with_stamp(
                handle,
                asset_version,
                asset_identity,
                material_version,
                true,
                canonical_binding_stamp,
                PlannedMaterialCandidateSource::Existing {
                    expected_bind_group_id,
                },
            );
        }

        let Some((current_resource_ids, fingerprint_has_pending)) =
            self.planned_binding_resource_ids(&builder, textures)
        else {
            // A ready plan without its exact TextureBinding indicates that a
            // producer drifted or the required post-DPE materialization order
            // was violated. Never silently bind a stale fallback.
            return batch.push_unavailable(handle, asset_version, asset_identity, material_version);
        };
        debug_assert_eq!(fingerprint_has_pending, has_pending_textures);

        let needs_rebuild_bindgroup = if let Some(gpu_mat) = self.resources.material(handle) {
            let same_asset =
                gpu_mat.asset_version == asset_version && gpu_mat.asset_identity == asset_identity;
            let same_pipeline_state = gpu_mat.version == material_version;
            if has_pending_textures && same_asset && same_pipeline_state {
                false
            } else if !same_asset || !same_pipeline_state {
                true
            } else {
                gpu_mat.resource_ids != current_resource_ids
            }
        } else {
            true
        };

        if !needs_rebuild_bindgroup {
            let expected_bind_group_id = self
                .resources
                .material(handle)
                .expect("reusable Material disappeared")
                .bind_group_id;
            return batch.push_with_stamp(
                handle,
                asset_version,
                asset_identity,
                material_version,
                has_pending_textures,
                canonical_binding_stamp,
                PlannedMaterialCandidateSource::Existing {
                    expected_bind_group_id,
                },
            );
        }

        let observed_bind_group_id = self
            .resources
            .material(handle)
            .map(|material| material.bind_group_id);
        let candidate = Box::new(self.build_material_from_builder(
            &builder,
            current_resource_ids,
            asset_version,
            asset_identity,
            material_version,
            has_pending_textures,
            Arc::clone(pipeline_snapshot),
            canonical_binding_stamp,
        ));
        if material.data.version() != material_version {
            return batch.push_unavailable(handle, asset_version, asset_identity, material_version);
        }
        let private_slot = batch.private_materials.len();
        batch.private_materials.push(Some(candidate));
        batch.push_with_stamp(
            handle,
            asset_version,
            asset_identity,
            material_version,
            has_pending_textures,
            canonical_binding_stamp,
            PlannedMaterialCandidateSource::Rebuild {
                observed_bind_group_id,
                private_slot,
            },
        )
    }

    /// Exact resident path for macro-generated material bindings.
    ///
    /// The canonical producer has already materialized every ready Asset
    /// Texture and exported its physical token into the owner batch. Matching
    /// those tokens and the exact uniform stamp against the persistent Group 1
    /// fingerprint makes rebuilding a ResourceBuilder redundant. Hand-written
    /// custom materials never enter this path.
    #[allow(clippy::too_many_arguments)]
    fn try_prepare_exact_canonical_material(
        &mut self,
        batch: &mut PlannedMaterialCandidateBatch,
        handle: MaterialHandle,
        asset_version: u32,
        asset_identity: u64,
        material_version: u64,
        stamp: CanonicalMaterialBindingStamp,
        textures: &[PlannedMaterialTexture],
    ) -> Option<PlannedMaterialCandidateId> {
        let (expected_bind_group_id, uniform_physical_id) = {
            let gpu_material = self.resources.material(handle)?;
            if (gpu_material.asset_version, gpu_material.asset_identity)
                != (asset_version, asset_identity)
                || gpu_material.version != material_version
                || gpu_material.canonical_binding_stamp != Some(stamp)
            {
                return None;
            }

            let resource_ids = gpu_material.resource_ids.as_slice();
            if resource_ids.len() != 1 + textures.len().saturating_mul(2) {
                return None;
            }
            for (index, texture) in textures.iter().enumerate() {
                let TextureSource::Asset(handle) = texture.source else {
                    return None;
                };
                let state = texture.asset?;
                if state.texture_handle != handle || state.ready.is_none() {
                    return None;
                }
                let physical = texture.physical?;
                let resource_index = 1 + index * 2;
                if resource_ids[resource_index] != physical.generation_key.allocation
                    || resource_ids[resource_index + 1] != physical.sampler_id as u64
                {
                    return None;
                }
            }
            (gpu_material.bind_group_id, resource_ids[0])
        };

        if !self.resources.touch_buffer_if_exact(
            stamp.uniform,
            uniform_physical_id,
            self.frame_index,
        ) {
            return None;
        }

        let candidate = batch.push_with_stamp(
            handle,
            asset_version,
            asset_identity,
            material_version,
            false,
            Some(stamp),
            PlannedMaterialCandidateSource::Existing {
                expected_bind_group_id,
            },
        );
        #[cfg(test)]
        {
            batch.candidates[candidate.index()].canonical_fast_path = true;
        }
        Some(candidate)
    }

    /// Reserve compact logical slots and dense fresh-owner capacity before
    /// opening the bounded AssetStorage publication gate.
    ///
    /// Only rebuild candidates that survived the owner-local postflight need
    /// reservations. Obsolete handle generations are retired here, outside
    /// the gate; final publication then performs no allocation or destruction.
    pub(crate) fn reserve_planned_material_candidate_slots(
        &mut self,
        batch: &PlannedMaterialCandidateBatch,
        current: &mut [bool],
    ) {
        assert_eq!(
            batch.candidates.len(),
            current.len(),
            "Material reservation bits must match candidate count"
        );
        let additional_slots = batch
            .candidates
            .iter()
            .zip(current.iter())
            .filter(|(candidate, current)| {
                **current
                    && matches!(
                        candidate.source,
                        PlannedMaterialCandidateSource::Rebuild { .. }
                    )
                    && !self.resources.material_slot_exists(candidate.handle)
            })
            .count();
        self.resources.reserve_material_slots(additional_slots);

        for (candidate, current) in batch.candidates.iter().zip(current) {
            if *current
                && matches!(
                    candidate.source,
                    PlannedMaterialCandidateSource::Rebuild { .. }
                )
                && !self.resources.reserve_material_slot(candidate.handle)
            {
                *current = false;
            }
        }
    }

    /// Publish only final-current Material candidates under a cache-side CAS.
    /// Equivalent exact winners are reused; a conflicting cache mutation is
    /// never overwritten by a stale graph attempt.
    #[cfg(test)]
    pub(crate) fn publish_planned_material_candidates(
        &mut self,
        batch: &mut PlannedMaterialCandidateBatch,
        current: &[bool],
    ) {
        assert!(
            batch.published_count == 0,
            "planned material candidate batch may be published only once"
        );
        assert_eq!(
            batch.candidates.len(),
            current.len(),
            "Material postflight bits must match candidate count"
        );
        let mut reservable = current.to_vec();
        self.reserve_planned_material_candidate_slots(batch, &mut reservable);
        for (index, is_current) in reservable.iter().copied().enumerate() {
            self.publish_planned_material_candidate_impl(
                batch,
                PlannedMaterialCandidateId::from_index(index),
                is_current,
            );
        }
    }

    /// Publish one candidate while the caller's exact Material snapshot is
    /// still protected by its bounded AssetStorage read guard.
    ///
    /// The first call seals the batch against additional staging. Distinct
    /// candidate IDs may then be published in stable chunk order; publishing
    /// the same ID twice is a contract violation.
    pub(crate) fn publish_planned_material_candidate(
        &mut self,
        batch: &mut PlannedMaterialCandidateBatch,
        id: PlannedMaterialCandidateId,
        is_current: bool,
    ) -> bool {
        self.publish_planned_material_candidate_impl(batch, id, is_current)
    }

    fn publish_planned_material_candidate_impl(
        &mut self,
        batch: &mut PlannedMaterialCandidateBatch,
        id: PlannedMaterialCandidateId,
        is_current: bool,
    ) -> bool {
        let index = id.index();
        let candidate = batch
            .candidates
            .get(index)
            .expect("planned Material candidate ID is invalid");
        assert!(
            candidate.result.is_none(),
            "planned Material candidate was already published"
        );
        let exact_cache = |gpu_mat: &GpuMaterial| {
            gpu_mat.asset_version == candidate.asset_version
                && gpu_mat.asset_identity == candidate.asset_identity
                && gpu_mat.version == candidate.material_version
        };
        let accepted = if is_current {
            match candidate.source {
                PlannedMaterialCandidateSource::Unavailable => false,
                PlannedMaterialCandidateSource::Existing {
                    expected_bind_group_id,
                } => self
                    .resources
                    .material(candidate.handle)
                    .is_some_and(|gpu_mat| {
                        exact_cache(gpu_mat) && gpu_mat.bind_group_id == expected_bind_group_id
                    }),
                PlannedMaterialCandidateSource::Rebuild {
                    observed_bind_group_id,
                    private_slot,
                } => {
                    let private = batch.private_materials[private_slot]
                        .as_ref()
                        .expect("Material rebuild candidate owner is missing");
                    let exact_winner =
                        self.resources
                            .material(candidate.handle)
                            .is_some_and(|gpu_mat| {
                                exact_cache(gpu_mat)
                                    && gpu_mat.resource_ids == private.resource_ids
                                    && gpu_mat.layout_id == private.layout_id
                                    && gpu_mat.binding_interface_id == private.binding_interface_id
                            });
                    if exact_winner {
                        true
                    } else if self
                        .resources
                        .material(candidate.handle)
                        .map(|gpu_mat| gpu_mat.bind_group_id)
                        == observed_bind_group_id
                    {
                        let private = batch.private_materials[private_slot]
                            .take()
                            .expect("Material rebuild candidate was already published");
                        let retired = self.resources.swap_material(candidate.handle, private);
                        // Defer old WGPU-owner destruction until the candidate
                        // batch is cleared outside the publication guard.
                        batch.private_materials[private_slot] = retired;
                        true
                    } else {
                        false
                    }
                }
            }
        } else {
            false
        };

        if accepted {
            let gpu_mat = self
                .resources
                .material_mut(candidate.handle)
                .expect("published Material candidate has no cache entry");
            debug_assert!(exact_cache(gpu_mat));
            gpu_mat.canonical_binding_stamp = candidate.canonical_binding_stamp;
            gpu_mat.last_used_frame = self.frame_index;
            if !candidate.has_pending_textures {
                gpu_mat.last_verified_frame = self.frame_index;
            }
        }
        batch.candidates[index].result = Some(accepted);
        batch.published_count += 1;
        accepted
    }

    /// Fingerprint the exact physical resources selected by a planned binding
    /// recipe. This follows the actual builder order and therefore includes
    /// custom material buffers as well as texture/sampler pairs.
    fn planned_binding_resource_ids(
        &self,
        builder: &ResourceBuilder<'_>,
        textures: &[PlannedMaterialTexture],
    ) -> Option<(ResourceIdSet, bool)> {
        let mut ids = ResourceIdSet::with_capacity(builder.bindings.len() * 2);
        let mut texture_index = 0usize;
        let mut has_pending = false;

        for binding in &builder.bindings {
            match &binding.resource {
                BindingResource::Buffer { buffer, .. } => {
                    ids.push(self.get_gpu_buffer_by_cpu_id(buffer.id())?.id);
                }
                BindingResource::Texture(source) => {
                    let planned = source.as_ref().map(|source| {
                        let planned = textures.get(texture_index)?;
                        texture_index += 1;
                        (planned.source == *source).then_some(planned)
                    });
                    let planned = match planned {
                        Some(Some(planned)) => Some(planned),
                        Some(None) => return None,
                        None => None,
                    };

                    match source {
                        Some(TextureSource::Asset(handle)) => {
                            let plan = planned?;
                            let state = plan.asset?;
                            if state.ready.is_some_and(|snapshot| {
                                !self.asset_texture_binding_matches_snapshot(snapshot)
                            }) {
                                return None;
                            }
                            if state.ready.is_none()
                                && *handle != myth_assets::TextureHandle::dummy_env_map()
                            {
                                has_pending = true;
                            }
                        }
                        Some(TextureSource::Attachment(id, _))
                            if self.internal_textures.get(*id).is_none() =>
                        {
                            has_pending = true;
                        }
                        Some(TextureSource::Attachment(_, _)) | None => {}
                    }
                    let (_, view_id) =
                        self.resolve_binding_texture_view(source.as_ref(), &binding.desc)?;
                    if let Some(TextureSource::Asset(handle)) = source
                        && let Some(snapshot) = planned.and_then(|plan| plan.asset)?.ready
                    {
                        let exact_view = self.resources.texture_binding(*handle)?.view_id;
                        debug_assert!(self.asset_texture_binding_matches_snapshot(snapshot));
                        if view_id != exact_view {
                            return None;
                        }
                    }
                    ids.push(view_id);

                    let sampler_id = match source {
                        Some(TextureSource::Asset(handle)) => self
                            .resources
                            .texture_binding(*handle)
                            .map(|binding| binding.sampler_id as u64),
                        _ => None,
                    }
                    .unwrap_or_else(|| match &binding.desc {
                        BindingDesc::Texture {
                            sampler_binding_type: wgpu::SamplerBindingType::Comparison,
                            ..
                        } => self.system_textures.shadow_compare_sampler.id(),
                        BindingDesc::Texture { .. } => {
                            self.sampler_registry.default_sampler().0 as u64
                        }
                        BindingDesc::Buffer { .. } => unreachable!(),
                    });
                    ids.push(sampler_id);
                }
                BindingResource::_Phantom(_) => unreachable!(),
            }
        }
        (texture_index == textures.len()).then_some((ids, has_pending))
    }

    fn planned_material_has_pending(&self, textures: &[PlannedMaterialTexture]) -> bool {
        textures.iter().any(|texture| match texture.source {
            TextureSource::Asset(handle) => {
                handle != myth_assets::TextureHandle::dummy_env_map()
                    && texture.asset.is_none_or(|state| state.ready.is_none())
            }
            TextureSource::Attachment(id, _) => self.internal_textures.get(id).is_none(),
        })
    }

    /// Ensure Material resources and return the resource ID set.
    ///
    /// Returns `(resource_ids, has_pending_textures)` where
    /// `has_pending_textures` is `true` when at least one texture dependency
    /// is still in `Loading` state.
    fn ensure_material_resources(
        &mut self,
        assets: &AssetServer,
        material: &Material,
    ) -> (ResourceIdSet, bool) {
        let mut uniform_result = EnsureResult::existing(0);

        // 2. Call with_uniform_bytes, passing a closure
        material.data.with_uniform_bytes(&mut |bytes| {
            (_, uniform_result) = self.ensure_buffer_ref(&material.data.uniform_buffer(), bytes);
        });

        // Collect resource IDs
        let mut resource_ids = ResourceIdSet::with_capacity(16);
        resource_ids.push(uniform_result.resource_id);

        let mut has_pending = false;

        // Use visit_textures to iterate over all texture resources
        material
            .data
            .visit_textures(&mut |tex_source| match tex_source {
                TextureSource::Asset(tex_handle) => {
                    let state = self.prepare_texture(assets, *tex_handle);
                    if matches!(state, ResourceState::Pending) {
                        has_pending = true;
                    }
                    if let Some(binding) = self.resources.texture_binding(*tex_handle) {
                        resource_ids.push(binding.view_id);
                        resource_ids.push(binding.sampler_id as u64);
                    } else {
                        resource_ids.push(self.system_textures.black_cube.id());
                        resource_ids.push(self.sampler_registry.default_sampler().0 as u64);
                    }
                }
                TextureSource::Attachment(id, _) => {
                    if let Some(view_id) = self.internal_textures.view_id(*id) {
                        resource_ids.push(view_id);
                    } else {
                        has_pending = true;
                        resource_ids.push(self.system_textures.black_2d.id());
                    }
                    resource_ids.push(self.sampler_registry.default_sampler().0 as u64);
                }
            });

        (resource_ids, has_pending)
    }

    /// Rebuild Material's BindGroup (may include Layout)
    fn rebuild_material_bindgroup(
        &mut self,
        assets: &AssetServer,
        handle: MaterialHandle,
        material: &Material,
        resource_ids: ResourceIdSet,
        asset_version: u32,
        asset_identity: u64,
    ) {
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);
        let pipeline_snapshot = Arc::new(MaterialPipelineSnapshot::capture(material));

        self.prepare_binding_resources(assets, &builder.bindings);

        let gpu_mat = self.build_material_from_builder(
            &builder,
            resource_ids,
            asset_version,
            asset_identity,
            material.data.version(),
            false,
            pipeline_snapshot,
            material.data.canonical_binding_stamp(),
        );
        self.resources.insert_material(handle, gpu_mat);
    }

    /// Build a candidate from an already prepared exact binding recipe.
    /// Callers decide when its logical snapshot is safe to publish.
    #[allow(clippy::too_many_arguments)]
    fn build_material_from_builder(
        &mut self,
        builder: &ResourceBuilder<'_>,
        resource_ids: ResourceIdSet,
        asset_version: u32,
        asset_identity: u64,
        material_version: u64,
        has_pending_textures: bool,
        pipeline_snapshot: Arc<MaterialPipelineSnapshot>,
        canonical_binding_stamp: Option<CanonicalMaterialBindingStamp>,
    ) -> GpuMaterial {
        // Compute hash of layout entries
        let layout_entries = builder.generate_layout_entries();
        let layout_hash = hash_layout_entries(&layout_entries);

        // Layout identity comes from exact entry-vector equality in the
        // canonical cache. `layout_hash` remains diagnostic metadata only and
        // is never a correctness condition for WGPU layout reuse.
        let (layout, layout_id) = self.get_or_create_layout(&layout_entries);

        let (bind_group, bg_id) = self.create_bind_group(&layout, builder);
        let binding_wgsl: Arc<str> = builder.generate_wgsl(1).into();
        let binding_interface_id = self
            .bind_groups
            .intern_binding_interface(layout_id, &binding_wgsl);

        GpuMaterial {
            asset_version,
            asset_identity,
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            layout_hash,
            binding_wgsl,
            binding_interface_id,
            pipeline_snapshot,
            canonical_binding_stamp,
            resource_ids,
            version: material_version,
            last_used_frame: self.frame_index,
            last_verified_frame: if has_pending_textures {
                self.frame_index.wrapping_sub(1)
            } else {
                self.frame_index
            },
        }
    }

    pub fn get_material(&self, handle: MaterialHandle) -> Option<&GpuMaterial> {
        self.resources.material(handle)
    }

    /// Return the persistent immutable pipeline snapshot only when the cached
    /// Material belongs to the exact graph-attempt generation.
    #[must_use]
    pub(crate) fn exact_material_pipeline_snapshot(
        &self,
        handle: MaterialHandle,
        asset_version: u32,
        asset_identity: u64,
        material_version: u64,
    ) -> Option<Arc<MaterialPipelineSnapshot>> {
        self.resources.material(handle).and_then(|material| {
            (material.asset_version == asset_version
                && material.asset_identity == asset_identity
                && material.version == material_version)
                .then(|| Arc::clone(&material.pipeline_snapshot))
        })
    }
}

fn binding_recipe_matches_plan(
    builder: &ResourceBuilder<'_>,
    textures: &[PlannedMaterialTexture],
) -> bool {
    let mut texture_index = 0usize;
    for binding in &builder.bindings {
        let BindingResource::Texture(Some(source)) = &binding.resource else {
            continue;
        };
        if textures
            .get(texture_index)
            .is_none_or(|planned| planned.source != *source)
        {
            return false;
        }
        texture_index += 1;
    }
    texture_index == textures.len()
}

#[cfg(test)]
mod tests;
