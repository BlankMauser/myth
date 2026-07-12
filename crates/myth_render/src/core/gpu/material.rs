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

use std::sync::Arc;

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
    /// Hash of layout entries (for fast comparison to determine if Layout rebuild is needed)
    pub layout_hash: u64,
    pub binding_wgsl: Arc<str>,
    pub binding_wgsl_hash: u64,
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

    /// Prepare one exact graph-attempt material without discovering texture
    /// producers after RDG compilation.
    ///
    /// Ready Asset Textures must already have been materialized by their live
    /// logical import and published as an exact [`TextureBinding`](super::TextureBinding).
    /// Unavailable dependencies retain the legacy lagging behavior: an exact
    /// prior material group may remain in use, otherwise the binding recipe is
    /// built against the currently retained physical view or a system fallback.
    /// No path in this method calls `prepare_texture` or queues mip work.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_planned_material(
        &mut self,
        assets: &AssetServer,
        handle: MaterialHandle,
        material: &Arc<Material>,
        asset_version: u32,
        asset_identity: u64,
        material_version: u64,
        textures: &[PlannedMaterialTexture],
    ) -> bool {
        if !planned_material_is_current(
            assets,
            handle,
            material,
            asset_version,
            asset_identity,
            material_version,
            textures,
        ) {
            return false;
        }

        // Upload the canonical material uniform before the buffer-only recipe
        // preparation below. Custom bindings with inline bytes are handled by
        // `prepare_binding_buffers` from the same exact builder.
        material.data.with_uniform_bytes(&mut |bytes| {
            self.ensure_buffer_ref(&material.data.uniform_buffer(), bytes);
        });

        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);
        if !binding_recipe_matches_plan(&builder, textures) {
            return false;
        }
        self.prepare_binding_buffers(&builder.bindings);

        let has_pending_textures = self.planned_material_has_pending(textures);
        if has_pending_textures
            && self.resources.material(handle).is_some_and(|gpu_mat| {
                gpu_mat.asset_version == asset_version
                    && gpu_mat.asset_identity == asset_identity
                    && gpu_mat.version == material_version
            })
        {
            // Lag the complete exact material group while any dependency is
            // unavailable. Reusing an already validated BG is safe even when
            // no descriptor-compatible fallback exists for a custom binding.
            if !planned_material_is_current(
                assets,
                handle,
                material,
                asset_version,
                asset_identity,
                material_version,
                textures,
            ) {
                return false;
            }
            self.resources
                .material_mut(handle)
                .expect("validated lagging material disappeared")
                .last_used_frame = self.frame_index;
            return true;
        }

        let Some((mut current_resource_ids, fingerprint_has_pending)) =
            self.planned_binding_resource_ids(&builder, textures)
        else {
            // A ready plan without its exact TextureBinding indicates that a
            // producer drifted or the required post-DPE materialization order
            // was violated. Never silently bind a stale fallback.
            return false;
        };
        debug_assert_eq!(fingerprint_has_pending, has_pending_textures);
        if !planned_material_is_current(
            assets,
            handle,
            material,
            asset_version,
            asset_identity,
            material_version,
            textures,
        ) {
            return false;
        }

        let needs_rebuild_bindgroup = if let Some(gpu_mat) = self.resources.material(handle) {
            let same_asset =
                gpu_mat.asset_version == asset_version && gpu_mat.asset_identity == asset_identity;
            let same_pipeline_state = gpu_mat.version == material_version;
            if has_pending_textures && same_asset && same_pipeline_state {
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

        let candidate = needs_rebuild_bindgroup.then(|| {
            self.build_material_from_builder(
                handle,
                &builder,
                current_resource_ids,
                asset_version,
                asset_identity,
                material_version,
                has_pending_textures,
            )
        });

        // Candidate WGPU resources are local until this final exact check. A
        // concurrent Asset replacement can warm persistent buffers/layouts,
        // but it can never overwrite the material cache under stale stamps.
        if !planned_material_is_current(
            assets,
            handle,
            material,
            asset_version,
            asset_identity,
            material_version,
            textures,
        ) {
            return false;
        }
        if let Some(candidate) = candidate {
            self.resources.insert_material(handle, candidate);
        }

        let Some(gpu_mat) = self.resources.material_mut(handle) else {
            return false;
        };
        gpu_mat.version = material_version;
        gpu_mat.asset_version = asset_version;
        gpu_mat.asset_identity = asset_identity;
        gpu_mat.last_used_frame = self.frame_index;
        if !has_pending_textures {
            gpu_mat.last_verified_frame = self.frame_index;
        }
        true
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
            handle,
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
        handle: MaterialHandle,
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

        // Check if a new Layout is needed
        let (layout, layout_id) = if let Some(gpu_mat) = self.resources.material(handle) {
            if gpu_mat.layout_hash == layout_hash {
                // Layout unchanged, reuse
                (gpu_mat.layout.clone(), gpu_mat.layout_id)
            } else {
                // Layout changed, rebuild
                self.get_or_create_layout(&layout_entries)
            }
        } else {
            self.get_or_create_layout(&layout_entries)
        };

        let (bind_group, bg_id) = self.create_bind_group(&layout, builder);
        let binding_wgsl = builder.generate_wgsl(1);
        let binding_wgsl_hash = super::hash_binding_wgsl(&binding_wgsl);

        GpuMaterial {
            asset_version,
            asset_identity,
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            layout_hash,
            binding_wgsl: binding_wgsl.into(),
            binding_wgsl_hash,
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

#[allow(clippy::too_many_arguments)]
fn planned_material_is_current(
    assets: &AssetServer,
    handle: MaterialHandle,
    material: &Arc<Material>,
    asset_version: u32,
    asset_identity: u64,
    material_version: u64,
    textures: &[PlannedMaterialTexture],
) -> bool {
    let Some((current, current_version, current_identity)) =
        assets.materials.get_entry_with_identity(handle)
    else {
        return false;
    };
    if (current_version, current_identity) != (asset_version, asset_identity)
        || !Arc::ptr_eq(&current, material)
        || material.data.version() != material_version
    {
        return false;
    }

    let mut texture_index = 0usize;
    let mut matches = true;
    material.data.visit_textures(&mut |source| {
        if !matches {
            return;
        }
        let Some(planned) = textures.get(texture_index) else {
            matches = false;
            return;
        };
        texture_index += 1;
        if planned.source != *source {
            matches = false;
            return;
        }
        match *source {
            TextureSource::Asset(texture) => {
                matches = planned.asset.is_some_and(|state| {
                    state.texture_handle == texture
                        && crate::core::gpu::snapshot_asset_texture_state(assets, texture) == state
                });
            }
            TextureSource::Attachment(_, _) => {
                matches = planned.asset.is_none();
            }
        }
    });

    matches
        && texture_index == textures.len()
        && material.data.version() == material_version
        && assets.materials.get_version_and_identity(handle)
            == Some((asset_version, asset_identity))
}
