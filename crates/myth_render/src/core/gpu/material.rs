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
use crate::core::gpu::{AssetTextureStateSnapshot, EnsureResult};
use myth_assets::{AssetServer, MaterialHandle};
use myth_resources::material::{Material, RenderableMaterialTrait};

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
    source: PlannedMaterialCandidateSource,
    result: Option<bool>,
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
            source,
            result: None,
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
        textures: &[PlannedMaterialTexture],
    ) -> PlannedMaterialCandidateId {
        if material.data.version() != material_version {
            return batch.push_unavailable(handle, asset_version, asset_identity, material_version);
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
            return batch.push(
                handle,
                asset_version,
                asset_identity,
                material_version,
                true,
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
            return batch.push(
                handle,
                asset_version,
                asset_identity,
                material_version,
                has_pending_textures,
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
        ));
        if material.data.version() != material_version {
            return batch.push_unavailable(handle, asset_version, asset_identity, material_version);
        }
        let private_slot = batch.private_materials.len();
        batch.private_materials.push(Some(candidate));
        batch.push(
            handle,
            asset_version,
            asset_identity,
            material_version,
            has_pending_textures,
            PlannedMaterialCandidateSource::Rebuild {
                observed_bind_group_id,
                private_slot,
            },
        )
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

        self.prepare_binding_resources(assets, &builder.bindings);

        let gpu_mat = self.build_material_from_builder(
            &builder,
            resource_ids,
            asset_version,
            asset_identity,
            material.data.version(),
            false,
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
mod tests {
    use super::*;
    use crate::core::WgpuContext;
    use crate::settings::{RendererInitConfig, RendererSettings};
    use slotmap::KeyData;

    fn test_resource_manager() -> ResourceManager {
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

    fn test_material_handle(index: u32) -> MaterialHandle {
        test_material_handle_generation(index, 1)
    }

    fn test_material_handle_generation(index: u32, version: u32) -> MaterialHandle {
        MaterialHandle::from(KeyData::from_ffi(
            (u64::from(version) << 32) | u64::from(index),
        ))
    }

    fn resource_ids(ids: &[u64]) -> ResourceIdSet {
        let mut result = ResourceIdSet::with_capacity(ids.len());
        for &id in ids {
            result.push(id);
        }
        result
    }

    fn clone_gpu_material(source: &GpuMaterial, bind_group_id: u64) -> GpuMaterial {
        GpuMaterial {
            asset_version: source.asset_version,
            asset_identity: source.asset_identity,
            bind_group: source.bind_group.clone(),
            bind_group_id,
            layout: source.layout.clone(),
            layout_id: source.layout_id,
            layout_hash: source.layout_hash,
            binding_wgsl: Arc::clone(&source.binding_wgsl),
            binding_interface_id: source.binding_interface_id,
            resource_ids: source.resource_ids.clone(),
            version: source.version,
            last_used_frame: source.last_used_frame,
            last_verified_frame: source.last_verified_frame,
        }
    }

    fn stage_rebuild(
        resources: &mut ResourceManager,
        batch: &mut PlannedMaterialCandidateBatch,
        handle: MaterialHandle,
        ids: &[u64],
        observed_bind_group_id: Option<u64>,
    ) -> PlannedMaterialCandidateId {
        const ASSET_VERSION: u32 = 7;
        const ASSET_IDENTITY: u64 = 11;
        const MATERIAL_VERSION: u64 = 13;

        let builder = ResourceBuilder::new();
        let candidate = Box::new(resources.build_material_from_builder(
            &builder,
            resource_ids(ids),
            ASSET_VERSION,
            ASSET_IDENTITY,
            MATERIAL_VERSION,
            false,
        ));
        let private_slot = batch.private_materials.len();
        batch.private_materials.push(Some(candidate));
        batch.push(
            handle,
            ASSET_VERSION,
            ASSET_IDENTITY,
            MATERIAL_VERSION,
            false,
            PlannedMaterialCandidateSource::Rebuild {
                observed_bind_group_id,
                private_slot,
            },
        )
    }

    #[test]
    fn candidate_publication_reuses_equivalent_winner_and_rejects_conflicts() {
        let mut resources = test_resource_manager();

        let equivalent_handle = test_material_handle(1);
        let mut equivalent_batch = PlannedMaterialCandidateBatch::new();
        let equivalent_candidate = stage_rebuild(
            &mut resources,
            &mut equivalent_batch,
            equivalent_handle,
            &[101, 102],
            None,
        );
        let private = equivalent_batch.private_materials[0]
            .as_ref()
            .expect("private candidate missing");
        let winner_bind_group_id = private.bind_group_id.wrapping_add(10_000);
        let equivalent_winner = clone_gpu_material(private, winner_bind_group_id);
        resources
            .resources
            .insert_material(equivalent_handle, equivalent_winner);

        resources.publish_planned_material_candidates(&mut equivalent_batch, &[true]);

        assert!(equivalent_batch.result(equivalent_candidate));
        assert!(equivalent_batch.all_published());
        assert_eq!(
            resources
                .get_material(equivalent_handle)
                .expect("equivalent winner disappeared")
                .bind_group_id,
            winner_bind_group_id,
            "an exact cache winner must be reused instead of overwritten"
        );
        assert!(
            equivalent_batch.private_materials[0].is_some(),
            "reusing a winner must leave the unpublished owner with the batch"
        );

        let conflicting_handle = test_material_handle(2);
        let mut conflicting_batch = PlannedMaterialCandidateBatch::new();
        let conflicting_candidate = stage_rebuild(
            &mut resources,
            &mut conflicting_batch,
            conflicting_handle,
            &[201],
            None,
        );
        let private = conflicting_batch.private_materials[0]
            .as_ref()
            .expect("private candidate missing");
        let mut conflicting_winner = clone_gpu_material(private, private.bind_group_id + 10_000);
        conflicting_winner.resource_ids = resource_ids(&[202]);
        let conflicting_winner_id = conflicting_winner.bind_group_id;
        resources
            .resources
            .insert_material(conflicting_handle, conflicting_winner);

        resources.publish_planned_material_candidates(&mut conflicting_batch, &[true]);

        assert!(!conflicting_batch.result(conflicting_candidate));
        assert!(conflicting_batch.all_published());
        let retained_conflict = resources
            .get_material(conflicting_handle)
            .expect("conflicting winner disappeared");
        assert_eq!(retained_conflict.bind_group_id, conflicting_winner_id);
        assert_eq!(retained_conflict.resource_ids, resource_ids(&[202]));
        assert!(
            conflicting_batch.private_materials[0].is_some(),
            "a rejected private owner must remain batch-scoped"
        );
    }

    #[test]
    fn stale_candidate_never_overwrites_the_observed_cache_entry() {
        let mut resources = test_resource_manager();
        let handle = test_material_handle(3);
        let builder = ResourceBuilder::new();
        let cached =
            resources.build_material_from_builder(&builder, resource_ids(&[301]), 7, 11, 13, false);
        let cached_bind_group_id = cached.bind_group_id;
        resources.resources.insert_material(handle, cached);

        let mut batch = PlannedMaterialCandidateBatch::new();
        let candidate = stage_rebuild(
            &mut resources,
            &mut batch,
            handle,
            &[302],
            Some(cached_bind_group_id),
        );
        resources.publish_planned_material_candidates(&mut batch, &[false]);

        assert!(!batch.result(candidate));
        assert!(batch.all_published());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                resources.publish_planned_material_candidate(&mut batch, candidate, true);
            }))
            .is_err(),
            "reading a result must not make the candidate publishable again"
        );

        let mut chunked = PlannedMaterialCandidateBatch::new();
        let ids: Vec<_> = (0..65_u32)
            .map(|version| chunked.push_unavailable(handle, version, u64::from(version), 13))
            .collect();
        for (index, &id) in ids.iter().enumerate() {
            assert!(!resources.publish_planned_material_candidate(
                &mut chunked,
                id,
                index % 2 == 0,
            ));
        }
        assert!(chunked.all_published());
        assert!(ids.iter().all(|&id| !chunked.result(id)));
        let retained = resources
            .get_material(handle)
            .expect("stale publication removed the cache entry");
        assert_eq!(retained.bind_group_id, cached_bind_group_id);
        assert_eq!(retained.resource_ids, resource_ids(&[301]));
        assert!(batch.private_materials[0].is_some());
    }

    #[test]
    fn accepted_rebuild_is_allocation_free_and_defers_the_retired_owner() {
        let mut resources = test_resource_manager();
        let handle = test_material_handle(4_096);
        let builder = ResourceBuilder::new();
        let cached =
            resources.build_material_from_builder(&builder, resource_ids(&[401]), 7, 11, 13, false);
        let cached_bind_group_id = cached.bind_group_id;
        resources.resources.insert_material(handle, cached);

        let mut batch = PlannedMaterialCandidateBatch::new();
        let candidate = stage_rebuild(
            &mut resources,
            &mut batch,
            handle,
            &[402],
            Some(cached_bind_group_id),
        );
        let candidate_bind_group_id = batch.private_materials[0]
            .as_ref()
            .expect("private candidate missing")
            .bind_group_id;
        let slot_capacity = resources.resources.material_slot_capacity();

        resources.publish_planned_material_candidates(&mut batch, &[true]);

        assert!(batch.result(candidate));
        assert_eq!(resources.resources.material_slot_capacity(), slot_capacity);
        assert_eq!(
            resources
                .get_material(handle)
                .expect("published Material missing")
                .bind_group_id,
            candidate_bind_group_id
        );
        assert_eq!(
            batch.private_materials[0]
                .as_ref()
                .expect("retired owner must remain batch-scoped")
                .bind_group_id,
            cached_bind_group_id
        );

        batch.clear();
        assert!(batch.is_empty());
    }

    #[test]
    fn newer_handle_generation_retires_the_obsolete_owner_before_publication() {
        let mut resources = test_resource_manager();
        let old_handle = test_material_handle_generation(8_192, 1);
        let new_handle = test_material_handle_generation(8_192, 3);
        let builder = ResourceBuilder::new();
        let obsolete =
            resources.build_material_from_builder(&builder, resource_ids(&[501]), 7, 11, 13, false);
        resources.resources.insert_material(old_handle, obsolete);
        assert!(resources.get_material(old_handle).is_some());

        let mut batch = PlannedMaterialCandidateBatch::new();
        let candidate = stage_rebuild(&mut resources, &mut batch, new_handle, &[502], None);
        let mut current = [true];
        resources.reserve_planned_material_candidate_slots(&batch, &mut current);
        assert_eq!(current, [true]);

        assert!(resources.get_material(old_handle).is_none());
        assert!(resources.get_material(new_handle).is_none());
        let slot_capacity = resources.resources.material_slot_capacity();

        assert!(resources.publish_planned_material_candidate(&mut batch, candidate, true));

        assert!(batch.result(candidate));
        assert_eq!(resources.resources.material_slot_capacity(), slot_capacity);
        assert!(resources.get_material(new_handle).is_some());
        assert!(
            batch.private_materials[0].is_none(),
            "the obsolete generation must have been retired before the publication gate"
        );
    }

    #[test]
    fn stale_handle_generation_cannot_displace_a_newer_cached_owner() {
        let mut resources = test_resource_manager();
        let stale_handle = test_material_handle_generation(12_288, 1);
        let current_handle = test_material_handle_generation(12_288, 3);
        let builder = ResourceBuilder::new();
        let current =
            resources.build_material_from_builder(&builder, resource_ids(&[551]), 7, 11, 13, false);
        let current_bind_group_id = current.bind_group_id;
        resources.resources.insert_material(current_handle, current);

        let mut batch = PlannedMaterialCandidateBatch::new();
        let candidate = stage_rebuild(&mut resources, &mut batch, stale_handle, &[552], None);
        resources.publish_planned_material_candidates(&mut batch, &[true]);

        assert!(!batch.result(candidate));
        assert!(resources.get_material(stale_handle).is_none());
        assert_eq!(
            resources
                .get_material(current_handle)
                .expect("newer Material cache owner was displaced")
                .bind_group_id,
            current_bind_group_id
        );
        assert!(batch.private_materials[0].is_some());
    }

    #[test]
    fn null_material_handle_reservation_fails_closed() {
        let mut resources = test_resource_manager();
        assert!(
            !resources
                .resources
                .reserve_material_slot(MaterialHandle::default())
        );
    }

    #[test]
    fn out_of_order_material_slot_reservations_preserve_earlier_high_indices() {
        let mut resources = test_resource_manager();
        let high = test_material_handle(16_384);
        let low = test_material_handle(8);
        let mut batch = PlannedMaterialCandidateBatch::new();
        let high_candidate = stage_rebuild(&mut resources, &mut batch, high, &[601], None);
        let low_candidate = stage_rebuild(&mut resources, &mut batch, low, &[602], None);

        resources.publish_planned_material_candidates(&mut batch, &[true, true]);

        assert!(batch.result(high_candidate));
        assert!(batch.result(low_candidate));
        assert!(resources.get_material(high).is_some());
        assert!(resources.get_material(low).is_some());
    }

    #[test]
    fn candidate_batch_clear_retains_then_trim_releases_excess_capacity() {
        let mut batch = PlannedMaterialCandidateBatch::new();
        let handle = test_material_handle(4);
        for version in 0..(PLANNED_MATERIAL_CANDIDATE_BASELINE * 16) {
            batch.push_unavailable(handle, version as u32, version as u64, version as u64);
        }

        assert!(batch.active_bytes() > 0);
        let peak_retained = batch.retained_bytes();
        assert!(peak_retained > PlannedMaterialCandidateBatch::baseline_bytes());

        batch.clear();

        assert!(batch.is_empty());
        assert_eq!(batch.active_bytes(), 0);
        assert_eq!(batch.retained_bytes(), peak_retained);

        batch.trim_capacity(0, peak_retained);

        assert!(batch.retained_bytes() < peak_retained);
        assert!(batch.retained_bytes() >= PlannedMaterialCandidateBatch::baseline_bytes());
        let reused = batch.push_unavailable(handle, 1, 2, 3);
        assert_eq!(reused.index(), 0);
        batch.clear();
        assert!(batch.is_empty());
    }

    #[test]
    fn planned_material_candidate_id_preserves_the_option_niche() {
        assert_eq!(
            size_of::<Option<PlannedMaterialCandidateId>>(),
            size_of::<PlannedMaterialCandidateId>()
        );
        assert_eq!(PlannedMaterialCandidateId::from_index(0).index(), 0);
        assert_eq!(PlannedMaterialCandidateId::from_index(127).index(), 127);
    }
}
