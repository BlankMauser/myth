//! `BindGroup` operations
//!
//! Includes Object `BindGroup` (Group 2), skeleton management, and global bindings (Group 0)

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use wgpu::ShaderStages;

use myth_assets::{AssetServer, TextureHandle};
use myth_resources::buffer::BufferRef;
use myth_resources::texture::TextureSource;
use myth_resources::uniforms::WgslStruct;
use myth_resources::uniforms::{DynamicModelUniforms, EnvironmentUniforms, RenderStateUniforms};
use myth_resources::{Mesh, geometry::Geometry};
use myth_scene::skeleton::Skeleton;

use crate::core::binding::Bindings;
use myth_resources::builder::{Binding, BindingDesc};
use myth_resources::{BindingResource, ResourceBuilder, WgslStructName};

use super::bind_group_store::{GlobalBindingInterface, SharedBindingWgsl};
use super::{
    BindGroupContext, CommonSampler, GlobalStateKey, GpuGlobalState, ModelBufferAllocator,
    ObjectBindGroupKey, ResourceManager, Tracked, generate_gpu_resource_id,
};

static NEXT_GLOBAL_STATE_ID: AtomicU32 = AtomicU32::new(0);

/// Fixed Group 0 buffer slot involved in exact post-DPE validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlobalStateBufferSlot {
    Camera,
    Environment,
    DirectionalLights,
}

/// A graph snapshot was not materialized exactly in this ResourceManager.
///
/// The post-DPE caller must fail the graph attempt closed. In particular, it
/// must never fall back to a persistent `GpuGlobalState` from an older frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlobalStatePhysicalFailure {
    MissingBuffer(GlobalStateBufferSlot),
    VersionMismatch(GlobalStateBufferSlot),
    UsageMismatch(GlobalStateBufferSlot),
    SizeMismatch(GlobalStateBufferSlot),
}

struct MaterializedGlobalBuffer {
    buffer: wgpu::Buffer,
    physical_id: u64,
}

impl ResourceManager {
    // ========================================================================
    // Skeleton management
    // ========================================================================

    /// Upload skeleton data (current and previous frame joints) to GPU.
    pub fn prepare_skeleton(&mut self, skeleton: &Skeleton) {
        let buffer_ref = skeleton.joint_matrices.handle();
        let buffer_guard = skeleton.joint_matrices.read();
        self.resources.write_buffer(
            &self.device,
            &self.queue,
            self.frame_index,
            &buffer_ref,
            bytemuck::cast_slice(buffer_guard.as_slice()),
        );

        let prev_buffer_ref = skeleton.prev_joint_matrices.handle();
        let prev_buffer_guard = skeleton.prev_joint_matrices.read();
        self.resources.write_buffer(
            &self.device,
            &self.queue,
            self.frame_index,
            &prev_buffer_ref,
            bytemuck::cast_slice(prev_buffer_guard.as_slice()),
        );
    }

    /// Register an internally generated texture (e.g. Render Target)
    ///
    /// These textures do not need CPU upload and their lifetime is managed by
    /// the caller. Registration advances a logical content revision so
    /// derived caches can detect rebinding under a stable ID.
    /// Typically called before a `RenderPass` executes.
    ///
    /// Suitable for: Pass-private resources where the Pass itself holds and maintains ID stability.
    /// Highest performance, no hash lookup.
    pub fn register_internal_texture_direct(&mut self, id: u64, view: wgpu::TextureView) {
        self.internal_textures.register_direct(id, view);
    }

    /// Suitable for: Cross-pass shared resources (e.g. "`SceneColor`").
    /// Internally maintains a Name -> ID mapping.
    pub fn register_internal_texture_by_name(
        &mut self,
        name: &str,
        view: wgpu::TextureView,
    ) -> u64 {
        self.internal_textures.register_by_name(name, view)
    }

    pub fn register_internal_texture(&mut self, view: wgpu::TextureView) -> u64 {
        self.internal_textures.register(view)
    }

    pub fn release_internal_texture(&mut self, id: u64) {
        self.internal_textures.remove(id);
        log::debug!("Released internal texture: {id}");
    }

    #[inline]
    #[must_use]
    pub fn get_internal_texture(&self, id: u64) -> Option<&wgpu::TextureView> {
        self.internal_textures.get(id)
    }

    /// Returns the logical content revision of an internal attachment.
    ///
    /// Re-registering the same stable attachment ID advances this revision,
    /// allowing derived persistent caches to invalidate without depending on
    /// a physical `TextureView` identity.
    #[inline]
    #[must_use]
    pub fn get_internal_texture_revision(&self, id: u64) -> Option<u64> {
        self.internal_textures.revision(id)
    }

    /// Returns the physical view identity used by bind-group fingerprints.
    ///
    /// Unlike the logical attachment ID and content revision, this changes
    /// only when the registered `TextureView` itself is replaced.
    #[inline]
    #[must_use]
    pub fn get_internal_texture_view_id(&self, id: u64) -> Option<u64> {
        self.internal_textures.view_id(id)
    }

    /// Notify derived-cache users that an attachment was modified in place.
    ///
    /// Re-registering a view advances the revision automatically. Call this
    /// only when commands overwrite the same registered physical view without
    /// replacing it, so environment conversion and other derived caches do not
    /// reuse stale contents.
    pub fn mark_internal_texture_dirty(&mut self, id: u64) -> bool {
        self.internal_textures.mark_content_dirty(id)
    }

    /// Unified helper method for retrieving `TextureView`
    ///
    /// Prioritizes Asset-converted textures, then registered internal textures, finally returns Dummy
    pub fn get_texture_view<'a>(&'a self, source: &TextureSource) -> &'a wgpu::TextureView {
        match source {
            TextureSource::Asset(handle) => {
                // Special handling for Dummy Env Map
                if *handle == TextureHandle::dummy_env_map() {
                    return &self.system_textures.black_cube;
                }

                // Look up GPU resource corresponding to the Asset
                if let Some(img) = self.resources.texture_image(*handle) {
                    return &img.default_view;
                }

                // Fallback
                &self.system_textures.black_2d
            }
            TextureSource::Attachment(id, _) => {
                // Directly look up the internal resource table
                self.internal_textures
                    .get(*id)
                    .unwrap_or(&self.system_textures.black_2d)
            }
        }
    }

    // ========================================================================
    // Unified prepare_mesh entry point
    // ========================================================================

    /// Prepare basic resources for a Mesh
    ///
    /// Uses an "Ensure -> Collect IDs -> Check Fingerprint -> Rebind" pattern.
    /// The caller must allocate every model slot and call
    /// [`ResourceManager::flush_model_buffers`] first; this guarantees the
    /// physical model buffer cannot change after the returned BindGroup is
    /// captured by a draw command.
    pub fn prepare_mesh(
        &mut self,
        assets: &AssetServer,
        mesh: &Mesh,
        skeleton: Option<&Skeleton>,
    ) -> Option<BindGroupContext> {
        self.prepare_mesh_internal(assets, mesh, skeleton, true)
    }

    /// Prepare geometry and the per-object binding after the graph-attempt
    /// material plan was prepared post-DPE.
    ///
    /// Live DrawPlan materialization uses this entry point after DPE. It must
    /// not discover a new material texture/mipmap producer after topology has
    /// already been compiled, so a missing planned material fails closed.
    /// Prepare one graph-planned draw from the exact retained Geometry asset.
    ///
    /// The caller owns the asset keep-alive and the stamps captured by the
    /// graph attempt. This path never reopens Geometry `AssetStorage` while
    /// preparing each live instance.
    pub(crate) fn prepare_draw_object_exact_geometry(
        &mut self,
        assets: &AssetServer,
        mesh: &Mesh,
        skeleton: Option<&Skeleton>,
        geometry: &Geometry,
        geometry_asset_version: u32,
        geometry_asset_identity: u64,
    ) -> Option<u64> {
        // Object bindings are valid only after every source item has reserved
        // its dynamic slot and the model buffer can no longer grow this frame.
        self.model_allocator.assert_frozen();

        self.ensure_buffer(&mesh.morph_uniforms);
        if !self.get_geometry(mesh.geometry).is_some_and(|gpu| {
            (gpu.asset_version, gpu.asset_identity)
                == (geometry_asset_version, geometry_asset_identity)
                && gpu.version == geometry.structure_version()
        }) {
            return None;
        }
        self.get_material(mesh.material)?;

        Some(self.prepare_object_bindings_id(assets, mesh, skeleton, geometry))
    }

    fn prepare_mesh_internal(
        &mut self,
        assets: &AssetServer,
        mesh: &Mesh,
        skeleton: Option<&Skeleton>,
        prepare_material: bool,
    ) -> Option<BindGroupContext> {
        // Object bindings are valid only after every source item has reserved
        // its dynamic slot and the model buffer can no longer grow this frame.
        self.model_allocator.assert_frozen();

        // === Ensure phase: ensure all resources are uploaded ===
        self.ensure_buffer(&mesh.morph_uniforms);
        if !self.prepare_geometry(assets, mesh.geometry) {
            return None;
        }
        if prepare_material {
            self.prepare_material(assets, mesh.material);
        } else if self.get_material(mesh.material).is_none() {
            return None;
        }

        let geometry = assets.geometries.get(mesh.geometry)?;
        let bind_group_id = self.prepare_object_bindings_id(assets, mesh, skeleton, &geometry);
        self.bind_groups.cached_bind_group(bind_group_id).cloned()
    }

    fn prepare_object_bindings_id(
        &mut self,
        assets: &AssetServer,
        mesh: &Mesh,
        skeleton: Option<&Skeleton>,
        geometry: &Geometry,
    ) -> u64 {
        // Build the exact binding recipe once. Resource preparation and cache
        // fingerprinting must use this same recipe so a future binding cannot
        // be created while silently missing from the cache key.
        let min_binding_size = ModelBufferAllocator::uniform_stride();
        let model_buffer_ref = self.model_allocator.buffer_handle();
        let mut builder = ResourceBuilder::new();
        builder.add_dynamic_uniform::<DynamicModelUniforms>(
            "model",
            model_buffer_ref,
            None,
            min_binding_size,
            ShaderStages::VERTEX | ShaderStages::FRAGMENT,
        );
        mesh.define_bindings(&mut builder);
        geometry.define_bindings(&mut builder);

        if let Some(skeleton) = skeleton {
            builder.add_storage_buffer(
                "skins",
                &skeleton.joint_matrices.handle(),
                None,
                true,
                ShaderStages::VERTEX,
                Some(WgslStructName::Name("mat4x4<f32>".into())),
            );
            builder.add_storage_buffer(
                "prev_skins",
                &skeleton.prev_joint_matrices.handle(),
                None,
                true,
                ShaderStages::VERTEX,
                Some(WgslStructName::Name("mat4x4<f32>".into())),
            );
        }

        assert_object_bindings_are_buffer_only(&builder.bindings);
        let layout_entries = builder.generate_layout_entries();
        self.prepare_binding_resources(assets, &builder.bindings);
        let (layout_id, newly_created_layout) =
            if let Some(layout_id) = self.bind_groups.cached_layout_id(&layout_entries) {
                (layout_id, None)
            } else {
                let (layout, layout_id) = self.get_or_create_layout(&layout_entries);
                (layout_id, Some(layout))
            };
        let cache_key = self.object_bind_group_fingerprint(&builder, layout_id);

        // Check global cache
        if let Some(bind_group_id) = self
            .bind_groups
            .touch_cached_object_bind_group(&cache_key, self.frame_index)
        {
            return bind_group_id;
        }

        // Cache miss: reuse the exact builder that was prepared and
        // fingerprinted above to create the physical BindGroup.
        let binding_wgsl: Arc<str> = builder.generate_wgsl(2).into();
        let layout = newly_created_layout.unwrap_or_else(|| {
            let (layout, resolved_layout_id) = self.get_or_create_layout(&layout_entries);
            debug_assert_eq!(layout_id, resolved_layout_id);
            layout
        });
        let (bind_group, bind_group_id) = self.create_bind_group(&layout, &builder);
        let binding_interface_id = self
            .bind_groups
            .intern_binding_interface(layout_id, &binding_wgsl);

        let data = BindGroupContext {
            layout,
            layout_id,
            bind_group,
            bind_group_id,
            binding_wgsl,
            binding_interface_id,
        };

        self.bind_groups.cache_object_bind_group(
            cache_key,
            bind_group_id,
            data.clone(),
            self.frame_index,
        );
        bind_group_id
    }

    /// Resolve the stable cache-owned object context after exact preparation.
    /// The caller may intern its physical bind group and interface without a
    /// temporary per-draw clone of the shared WGPU/Arc owners.
    #[inline]
    pub(crate) fn prepared_object_bind_group(
        &self,
        bind_group_id: u64,
    ) -> Option<&BindGroupContext> {
        self.bind_groups.cached_bind_group(bind_group_id)
    }

    fn object_bind_group_fingerprint(
        &self,
        builder: &ResourceBuilder<'_>,
        layout_id: u64,
    ) -> ObjectBindGroupKey {
        let mut key = ObjectBindGroupKey::with_capacity(builder.bindings.len(), layout_id);
        for binding in &builder.bindings {
            match &binding.resource {
                BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                    ..
                } => {
                    let gpu_buffer = self
                        .get_gpu_buffer_by_cpu_id(buffer.id())
                        .expect("object BindGroup buffer must be prepared before fingerprinting");
                    key.push_buffer(
                        gpu_buffer.id,
                        *offset,
                        *size,
                        binding.name,
                        binding.struct_name.as_ref(),
                    );
                }
                BindingResource::Texture(_) => {
                    panic!(
                        "object BindGroup texture binding '{}' requires physical view/sampler fingerprint support",
                        binding.name
                    );
                }
                BindingResource::_Phantom(_) => {
                    unreachable!("_Phantom should never be used for an object BindGroup")
                }
            }
        }

        key
    }

    // ========================================================================
    // BindGroup common operations
    // ========================================================================

    pub(crate) fn prepare_binding_resources(
        &mut self,
        assets: &AssetServer,
        bindings: &[Binding<'_>],
    ) {
        self.prepare_binding_buffers(bindings);

        for b in bindings {
            match &b.resource {
                BindingResource::Texture(Some(source)) => match source {
                    TextureSource::Asset(handle) => {
                        self.prepare_texture(assets, *handle);
                    }
                    TextureSource::Attachment(_, _) => {}
                },
                BindingResource::Buffer { .. }
                | BindingResource::Texture(None)
                | BindingResource::_Phantom(_) => {}
            }
        }
    }

    /// Prepare only buffers referenced by a binding recipe.
    ///
    /// Graph-planned materials call this after their lazy Asset Texture
    /// imports have been materialized from compiled DPE demand. Keeping texture
    /// discovery out of this helper guarantees a BindGroup rebuild cannot
    /// enqueue a late legacy mip request after graph compilation.
    pub(crate) fn prepare_binding_buffers(&mut self, bindings: &[Binding<'_>]) {
        for binding in bindings {
            let BindingResource::Buffer {
                buffer: buffer_ref,
                offset: _,
                size: _,
                data,
            } = &binding.resource
            else {
                continue;
            };
            let id = buffer_ref.id();
            if let Some(bytes) = data {
                self.resources.write_buffer(
                    &self.device,
                    &self.queue,
                    self.frame_index,
                    buffer_ref,
                    bytes,
                );
            } else if !self.resources.touch_buffer_by_cpu_id(id, self.frame_index) {
                panic!(
                    "ResourceManager: Trying to bind buffer {:?} (ID: {}) but it is not initialized!",
                    buffer_ref.label(),
                    id
                );
            }
        }
    }

    pub fn get_or_create_layout(
        &mut self,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> (wgpu::BindGroupLayout, u64) {
        self.bind_groups.get_or_create_layout(&self.device, entries)
    }

    #[allow(clippy::too_many_lines)]
    pub fn create_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        builder: &ResourceBuilder,
    ) -> (wgpu::BindGroup, u64) {
        self.create_bind_group_with_texture_view_override(layout, builder, None)
    }

    /// Creates a bind group while optionally replacing one texture binding
    /// with a directly borrowed view.
    ///
    /// The override only affects the physical bind-group entry. Layout and
    /// WGSL generation continue to use the binding description collected by
    /// [`ResourceBuilder`], so switching between a fallback/source texture and
    /// an RDG-owned persistent view does not change the shader interface.
    fn create_bind_group_with_texture_view_override(
        &self,
        layout: &wgpu::BindGroupLayout,
        builder: &ResourceBuilder,
        texture_view_override: Option<(&str, &wgpu::TextureView)>,
    ) -> (wgpu::BindGroup, u64) {
        let mut entries = Vec::new();
        let mut binding_index = 0u32;

        for b in &builder.bindings {
            match &b.resource {
                BindingResource::Buffer {
                    buffer,
                    data: _,
                    offset,
                    size,
                } => {
                    let cpu_id = buffer.id();
                    let gpu_buf = self
                        .get_gpu_buffer_by_cpu_id(cpu_id)
                        .expect("Buffer should be prepared");
                    entries.push(wgpu::BindGroupEntry {
                        binding: binding_index,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &gpu_buf.buffer,
                            offset: *offset,
                            size: size.and_then(wgpu::BufferSize::new),
                        }),
                    });
                    binding_index += 1;
                }
                BindingResource::Texture(source_opt) => {
                    // 1. Texture view entry
                    let view = texture_view_override
                        .filter(|(name, _)| *name == b.name)
                        .map_or_else(
                            || {
                                self.resolve_binding_texture_view(
                                    source_opt.as_ref(),
                                    &b.desc,
                                )
                                .map_or_else(
                                    || {
                                        panic!(
                                            "no compatible physical view or system fallback for texture binding '{}'",
                                            b.name
                                        )
                                    },
                                    |(view, _)| view,
                                )
                            },
                            |(_, view)| view,
                        );
                    entries.push(wgpu::BindGroupEntry {
                        binding: binding_index,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                    binding_index += 1;

                    // 2. Auto-paired sampler entry
                    let sampler = self.resolve_texture_sampler(source_opt.as_ref(), &b.desc);
                    entries.push(wgpu::BindGroupEntry {
                        binding: binding_index,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    });
                    binding_index += 1;
                }
                BindingResource::_Phantom(_) => unreachable!("_Phantom should never be used"),
            }
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Auto BindGroup"),
            layout,
            entries: &entries,
        });

        (bind_group, generate_gpu_resource_id())
    }

    /// Resolve the same descriptor-compatible texture view used by physical
    /// BindGroup creation together with its stable resource ID.
    ///
    /// Planned material fingerprinting calls this helper before building a
    /// candidate, so unsupported pending Cube/Array/Depth bindings fail closed
    /// instead of reaching WGPU validation with a dimension-mismatched view.
    pub(crate) fn resolve_binding_texture_view<'a>(
        &'a self,
        source: Option<&TextureSource>,
        desc: &BindingDesc,
    ) -> Option<(&'a wgpu::TextureView, u64)> {
        let BindingDesc::Texture {
            sample_type,
            view_dimension,
            ..
        } = desc
        else {
            return None;
        };

        match source {
            Some(TextureSource::Asset(handle)) if *handle != TextureHandle::dummy_env_map() => {
                if let Some(image) = self.resources.texture_image(*handle)
                    && image.default_view_dimension == *view_dimension
                    && texture_sample_type_supports(
                        image.format.sample_type(None, Some(self.device.features())),
                        *sample_type,
                    )
                {
                    return Some((&image.default_view, image.id));
                }
            }
            Some(TextureSource::Attachment(id, source_dimension)) => {
                if source_dimension == view_dimension
                    && let Some(view) = self.internal_textures.get(*id)
                {
                    return Some((
                        view,
                        self.internal_textures
                            .view_id(*id)
                            .expect("registered internal texture is missing its view ID"),
                    ));
                }
            }
            Some(TextureSource::Asset(_)) | None => {}
        }

        match (*sample_type, *view_dimension) {
            (wgpu::TextureSampleType::Float { .. }, wgpu::TextureViewDimension::D2) => Some((
                &self.system_textures.black_2d,
                self.system_textures.black_2d.id(),
            )),
            (wgpu::TextureSampleType::Float { .. }, wgpu::TextureViewDimension::Cube) => Some((
                &self.system_textures.black_cube,
                self.system_textures.black_cube.id(),
            )),
            (wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2Array) => Some((
                &self.system_textures.depth_d2array,
                self.system_textures.depth_d2array.id(),
            )),
            (wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::CubeArray) => Some((
                &self.system_textures.depth_cube_array,
                self.system_textures.depth_cube_array.id(),
            )),
            _ => None,
        }
    }

    /// Resolves the `wgpu::Sampler` for a texture binding.
    ///
    /// For asset textures, returns the sampler cached during `prepare_texture`.
    /// For attachments or missing textures, falls back to the dummy sampler
    /// (or the shadow comparison sampler for depth/comparison bindings).
    fn resolve_texture_sampler(
        &self,
        source: Option<&TextureSource>,
        desc: &BindingDesc,
    ) -> &wgpu::Sampler {
        if let Some(TextureSource::Asset(handle)) = source
            && let Some(binding) = self.resources.texture_binding(*handle)
            && let Some(sampler) = self
                .sampler_registry
                .get_sampler_by_index(binding.sampler_id)
        {
            return sampler;
        }

        // Fallback: comparison sampler for depth textures, default otherwise
        if let BindingDesc::Texture {
            sampler_binding_type: wgpu::SamplerBindingType::Comparison,
            ..
        } = desc
        {
            &self.system_textures.shadow_compare_sampler
        } else {
            self.sampler_registry.default_sampler().1
        }
    }

    // ========================================================================
    // Global bindings (Group 0)
    // ========================================================================

    /// Build or reuse Group 0 from buffers already materialized by a live RDG
    /// attempt.
    ///
    /// This entry point performs no CPU-buffer read, upload, asset lookup, or
    /// generic binding-resource preparation. All three `BufferRef`s must name
    /// the exact physical versions installed by the post-DPE materialization
    /// barrier. A mismatch fails closed before either the persistent state or
    /// its liveness timestamp is changed.
    pub(crate) fn prepare_global_from_materialized(
        &mut self,
        state_key: GlobalStateKey,
        camera: &BufferRef,
        environment: &BufferRef,
        directional_lights: &BufferRef,
        pmrem_override: Option<&Tracked<wgpu::TextureView>>,
    ) -> Result<u32, GlobalStatePhysicalFailure> {
        let camera_physical = self.resolve_materialized_global_buffer(
            camera,
            GlobalStateBufferSlot::Camera,
            wgpu::BufferUsages::UNIFORM,
        )?;
        let environment_physical = self.resolve_materialized_global_buffer(
            environment,
            GlobalStateBufferSlot::Environment,
            wgpu::BufferUsages::UNIFORM,
        )?;
        let directional_lights_physical = self.resolve_materialized_global_buffer(
            directional_lights,
            GlobalStateBufferSlot::DirectionalLights,
            wgpu::BufferUsages::STORAGE,
        )?;

        // Clone the selected view into this narrow materialization scope so
        // no HistoryRegistry or SystemTextures borrow overlaps the persistent
        // bind-group-store mutation below. The graph and resulting BindGroup
        // remain the authoritative physical owners for their respective
        // lifetimes.
        let pmrem = pmrem_override
            .cloned()
            .unwrap_or_else(|| self.system_textures.black_cube.clone());
        let current_ids = global_resource_ids(
            camera_physical.physical_id,
            environment_physical.physical_id,
            directional_lights_physical.physical_id,
            pmrem.id(),
        );

        if let Some(gpu_state) = self.bind_groups.global_state_mut(state_key)
            && gpu_state.resource_ids.matches_slice(current_ids.as_slice())
        {
            gpu_state.last_used_frame = self.frame_index;
            return Ok(gpu_state.id);
        }

        let interface =
            self.ensure_materialized_global_interface(camera, environment, directional_lights);
        let sampler = self.sampler_registry.get_common(CommonSampler::LinearClamp);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Global BindGroup"),
            layout: &interface.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_physical.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: environment_physical.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: directional_lights_physical.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&pmrem),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let id = self.bind_groups.global_state(state_key).map_or_else(
            || NEXT_GLOBAL_STATE_ID.fetch_add(1, Ordering::Relaxed),
            |existing| existing.id,
        );
        let gpu_state = GpuGlobalState {
            id,
            bind_group,
            bind_group_id: generate_gpu_resource_id(),
            layout: interface.layout,
            layout_id: interface.layout_id,
            binding_wgsl: interface.binding_wgsl,
            binding_interface_id: interface.binding_interface_id,
            resource_ids: current_ids,
            last_used_frame: self.frame_index,
        };
        self.bind_groups.insert_global_state(state_key, gpu_state);
        Ok(id)
    }

    fn resolve_materialized_global_buffer(
        &self,
        buffer_ref: &BufferRef,
        slot: GlobalStateBufferSlot,
        required_usage: wgpu::BufferUsages,
    ) -> Result<MaterializedGlobalBuffer, GlobalStatePhysicalFailure> {
        let gpu_buffer = self
            .resources
            .buffer_by_cpu_id(buffer_ref.id())
            .ok_or(GlobalStatePhysicalFailure::MissingBuffer(slot))?;
        if gpu_buffer.last_uploaded_version != buffer_ref.version {
            return Err(GlobalStatePhysicalFailure::VersionMismatch(slot));
        }
        if gpu_buffer.usage != buffer_ref.usage() || !buffer_ref.usage().contains(required_usage) {
            return Err(GlobalStatePhysicalFailure::UsageMismatch(slot));
        }
        if buffer_ref.size() == 0 || gpu_buffer.size < buffer_ref.size() as u64 {
            return Err(GlobalStatePhysicalFailure::SizeMismatch(slot));
        }
        Ok(MaterializedGlobalBuffer {
            buffer: gpu_buffer.buffer.clone(),
            physical_id: gpu_buffer.id,
        })
    }

    fn ensure_materialized_global_interface(
        &mut self,
        camera: &BufferRef,
        environment: &BufferRef,
        directional_lights: &BufferRef,
    ) -> GlobalBindingInterface {
        if let Some(interface) = self.bind_groups.global_interface() {
            return interface.clone();
        }

        let mut builder = ResourceBuilder::new();
        define_materialized_global_bindings(&mut builder, camera, environment, directional_lights);
        let layout_entries = builder.generate_layout_entries();
        let (layout, layout_id) = self.get_or_create_layout(&layout_entries);
        let binding_wgsl = SharedBindingWgsl::from(builder.generate_wgsl(0));
        let binding_interface_id = self
            .bind_groups
            .intern_binding_interface(layout_id, binding_wgsl.as_arc());
        let interface = GlobalBindingInterface {
            layout,
            layout_id,
            binding_wgsl,
            binding_interface_id,
        };
        self.bind_groups.insert_global_interface(interface.clone());
        interface
    }

    pub fn get_global_state(&self, render_state_id: u32, scene_id: u32) -> Option<&GpuGlobalState> {
        let state_id = GlobalStateKey::new(render_state_id, scene_id);
        self.bind_groups.global_state(state_id)
    }

    /// Drop every cached Group 0 state that still retains resources from one
    /// retired scene environment domain.
    pub(crate) fn remove_scene_global_states(&mut self, scene_id: u32) -> usize {
        self.bind_groups.remove_scene_global_states(scene_id)
    }
}

fn texture_sample_type_supports(
    actual: Option<wgpu::TextureSampleType>,
    required: wgpu::TextureSampleType,
) -> bool {
    match (actual, required) {
        (
            Some(wgpu::TextureSampleType::Float { filterable: actual }),
            wgpu::TextureSampleType::Float {
                filterable: required,
            },
        ) => !required || actual,
        (Some(actual), required) => actual == required,
        (None, _) => false,
    }
}

/// Builds the global bind-group fingerprint in binding dependency order.
///
/// The base environment cube is deliberately absent: it has no binding in
/// the current global shader interface and therefore must not invalidate the
/// global bind group.
fn global_resource_ids(
    camera_id: u64,
    environment_id: u64,
    directional_lights_id: u64,
    pmrem_id: u64,
) -> super::ResourceIdSet {
    let mut ids = super::ResourceIdSet::with_capacity(4);
    ids.push(camera_id);
    ids.push(environment_id);
    ids.push(directional_lights_id);
    ids.push(pmrem_id);
    ids
}

/// Define the process-static Group 0 interface without opening any CPU data or
/// resolving any physical resource. Buffer identities are carried only so the
/// existing ResourceBuilder can remain the single source of layout/WGSL truth.
fn define_materialized_global_bindings(
    builder: &mut ResourceBuilder<'_>,
    camera: &BufferRef,
    environment: &BufferRef,
    directional_lights: &BufferRef,
) {
    let visibility =
        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
    builder.add_uniform_buffer(
        "render_state",
        camera,
        None,
        visibility,
        false,
        None,
        Some(WgslStructName::Generator(
            RenderStateUniforms::wgsl_struct_def,
        )),
    );
    builder.add_uniform_buffer(
        "environment",
        environment,
        None,
        visibility,
        false,
        None,
        Some(WgslStructName::Generator(
            EnvironmentUniforms::wgsl_struct_def,
        )),
    );
    builder.add_storage_buffer(
        "directional_lights",
        directional_lights,
        None,
        true,
        visibility,
        Some(WgslStructName::Name("Struct_lights".to_string())),
    );
    builder.add_texture(
        "pmrem_map",
        None::<TextureSource>,
        wgpu::TextureSampleType::Float { filterable: true },
        wgpu::TextureViewDimension::Cube,
        wgpu::ShaderStages::FRAGMENT,
    );
}

/// Enforce the current Group 2 contract before resource preparation.
///
/// Texture bindings expand to a view plus sampler and therefore require both
/// physical identities in the cache fingerprint. Object bindings do not yet
/// implement that path, so accepting one without extending the fingerprint
/// would make cache reuse unsound.
fn assert_object_bindings_are_buffer_only(bindings: &[Binding<'_>]) {
    for binding in bindings {
        match binding.resource {
            BindingResource::Buffer { .. } => {}
            BindingResource::Texture(_) => panic!(
                "object BindGroup texture binding '{}' requires physical view/sampler fingerprint support",
                binding.name
            ),
            BindingResource::_Phantom(_) => {
                unreachable!("_Phantom should never be used for an object BindGroup")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GlobalStateBufferSlot, GlobalStatePhysicalFailure, assert_object_bindings_are_buffer_only,
        global_resource_ids,
    };
    use crate::core::gpu::{GlobalStateKey, GlobalStateResolver, Tracked};
    use crate::core::{ResourceManager, WgpuContext};
    use crate::settings::{RendererInitConfig, RendererSettings};
    use myth_assets::TextureHandle;
    use myth_resources::buffer::{BufferRef, CpuBuffer};
    use myth_resources::texture::TextureSource;
    use myth_resources::uniforms::{EnvironmentUniforms, GpuLightStorage, RenderStateUniforms};
    use myth_resources::{BindingDesc, ResourceBuilder};

    #[test]
    fn global_fingerprint_contains_only_bound_resource_identities() {
        let ids = global_resource_ids(11, 22, 33, 44);

        assert_eq!(ids.as_slice(), &[11, 22, 33, 44]);
    }

    #[test]
    #[should_panic(expected = "requires physical view/sampler fingerprint support")]
    fn object_binding_contract_rejects_unfingerprinted_textures() {
        let mut builder = ResourceBuilder::new();
        builder.add_texture(
            "unsupported",
            None::<TextureSource>,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
            wgpu::ShaderStages::VERTEX,
        );

        assert_object_bindings_are_buffer_only(&builder.bindings);
    }

    #[test]
    fn pending_texture_fallbacks_are_descriptor_compatible_or_unavailable() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let pending = TextureSource::Asset(TextureHandle::system_reserved(0xD35C));
        let float_cube = BindingDesc::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::Cube,
            sampler_binding_type: wgpu::SamplerBindingType::Filtering,
        };
        let float_cube_array = BindingDesc::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::CubeArray,
            sampler_binding_type: wgpu::SamplerBindingType::Filtering,
        };
        let depth_array = BindingDesc::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2Array,
            sampler_binding_type: wgpu::SamplerBindingType::Comparison,
        };

        assert_eq!(
            resources
                .resolve_binding_texture_view(Some(&pending), &float_cube)
                .map(|(_, id)| id),
            Some(resources.system_textures.black_cube.id())
        );
        assert!(
            resources
                .resolve_binding_texture_view(Some(&pending), &float_cube_array)
                .is_none(),
            "unsupported pending dimensions must fail before BindGroup creation"
        );
        assert_eq!(
            resources
                .resolve_binding_texture_view(Some(&pending), &depth_array)
                .map(|(_, id)| id),
            Some(resources.system_textures.depth_d2array.id())
        );
    }

    #[test]
    fn exact_global_state_reuses_complete_fingerprint_and_rebinds_changes() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let storage_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let camera = CpuBuffer::new(RenderStateUniforms::default(), uniform_usage, None);
        let environment = CpuBuffer::new(EnvironmentUniforms::default(), uniform_usage, None);
        let directional = CpuBuffer::new(vec![GpuLightStorage::default()], storage_usage, None);
        resources.ensure_buffer(&camera);
        resources.ensure_buffer(&environment);
        resources.ensure_buffer(&directional);
        let camera_ref = camera.handle();
        let environment_ref = environment.handle();
        let directional_ref = directional.handle();
        let key = GlobalStateKey::new(7, 11);

        let state_id = resources
            .prepare_global_from_materialized(
                key,
                &camera_ref,
                &environment_ref,
                &directional_ref,
                None,
            )
            .unwrap();
        let first = resources.bind_groups.global_state(key).unwrap();
        let first_bind_group = first.bind_group_id;
        let first_layout = first.layout_id;
        let first_wgsl = first.binding_wgsl.as_str().as_ptr();

        resources.next_frame();
        assert_eq!(
            resources
                .prepare_global_from_materialized(
                    key,
                    &camera_ref,
                    &environment_ref,
                    &directional_ref,
                    None,
                )
                .unwrap(),
            state_id
        );
        let reused = resources.bind_groups.global_state(key).unwrap();
        assert_eq!(reused.bind_group_id, first_bind_group);
        assert_eq!(reused.last_used_frame, resources.frame_index());

        let texture = wgpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Global PMREM Test"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let pmrem = Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        }));
        resources
            .prepare_global_from_materialized(
                key,
                &camera_ref,
                &environment_ref,
                &directional_ref,
                Some(&pmrem),
            )
            .unwrap();
        let rebound = resources.bind_groups.global_state(key).unwrap();
        assert_ne!(rebound.bind_group_id, first_bind_group);
        assert_eq!(rebound.id, state_id);
        assert_eq!(rebound.layout_id, first_layout);
        assert_eq!(rebound.binding_wgsl.as_str().as_ptr(), first_wgsl);
        assert_eq!(rebound.resource_ids.as_slice()[3], pmrem.id());

        assert!(
            GlobalStateResolver::new(&resources.bind_groups, None)
                .get(key)
                .is_none()
        );
        assert!(
            GlobalStateResolver::new(&resources.bind_groups, Some(key))
                .get(key)
                .is_some()
        );
    }

    #[test]
    fn exact_global_state_rejects_unmaterialized_or_inexact_buffers() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let storage_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let camera = CpuBuffer::new(RenderStateUniforms::default(), uniform_usage, None);
        let environment = CpuBuffer::new(EnvironmentUniforms::default(), uniform_usage, None);
        let directional = CpuBuffer::new(vec![GpuLightStorage::default()], storage_usage, None);
        resources.ensure_buffer(&environment);
        resources.ensure_buffer(&directional);
        let key = GlobalStateKey::new(3, 5);
        assert_eq!(
            resources.prepare_global_from_materialized(
                key,
                &camera.handle(),
                &environment.handle(),
                &directional.handle(),
                None,
            ),
            Err(GlobalStatePhysicalFailure::MissingBuffer(
                GlobalStateBufferSlot::Camera
            ))
        );
        assert!(resources.bind_groups.global_interface().is_none());
        assert!(resources.bind_groups.global_state(key).is_none());

        resources.ensure_buffer(&camera);
        let exact = camera.handle();
        let stale = BufferRef::with_fixed_id(
            exact.id(),
            exact.size(),
            exact.usage(),
            exact.version + 1,
            None,
        );
        assert_eq!(
            resources.prepare_global_from_materialized(
                key,
                &stale,
                &environment.handle(),
                &directional.handle(),
                None,
            ),
            Err(GlobalStatePhysicalFailure::VersionMismatch(
                GlobalStateBufferSlot::Camera
            ))
        );
    }
}
