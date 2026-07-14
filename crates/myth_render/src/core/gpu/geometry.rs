//! Geometry operations

use std::ops::Range;
use std::sync::Arc;

use crate::pipeline::vertex::GeneratedVertexLayout;
use myth_assets::{AssetServer, GeometryHandle};
use myth_resources::geometry::Geometry;
use myth_resources::shader_defines::ShaderDefines;

use super::ResourceManager;

/// GPU-side geometry resource
///
/// Vertex Buffer IDs are used for Pipeline cache validation and do not affect `BindGroup`
pub struct GpuGeometry {
    pub asset_version: u32,
    pub asset_identity: u64,
    pub layout_info: Arc<GeneratedVertexLayout>,
    pub layout_id: u64,
    /// Immutable shader interface derived for this exact layout generation.
    /// Frame-local geometry records share it instead of cloning the define
    /// vector for every DPE-live graph attempt.
    pub(crate) shader_defines: Arc<ShaderDefines>,
    pub vertex_buffers: Vec<wgpu::Buffer>,
    pub vertex_buffer_ids: Vec<u64>,
    pub index_buffer: Option<(wgpu::Buffer, wgpu::IndexFormat, u32, u64)>,
    pub draw_range: Range<u32>,
    pub instance_range: Range<u32>,
    pub version: u64,
    pub last_data_version: u64,
    pub last_used_frame: u64,
}

impl ResourceManager {
    /// Prepare Geometry GPU resources
    ///
    /// Returns `false` only when the asset is unavailable or preparation did
    /// not publish a GPU entry. Callers resolve exact physical IDs from the
    /// persistent `GpuGeometry` after all shared buffers are frozen.
    pub(crate) fn prepare_geometry(
        &mut self,
        assets: &AssetServer,
        handle: GeometryHandle,
    ) -> bool {
        let Some((geometry, asset_version, asset_identity)) =
            assets.geometries.get_entry_with_identity(handle)
        else {
            return false;
        };

        self.prepare_geometry_exact(handle, &geometry, asset_version, asset_identity)
    }

    /// Prepare one exact, already-retained Geometry asset revision.
    ///
    /// Post-DPE draw preparation uses the same retained asset for validation,
    /// GPU materialization, and object-binding construction. Keeping the
    /// borrowed asset and its stamps together avoids reopening `AssetStorage`
    /// for every live instance while preserving the handle-addressed GPU cache.
    pub(crate) fn prepare_geometry_exact(
        &mut self,
        handle: GeometryHandle,
        geometry: &Geometry,
        asset_version: u32,
        asset_identity: u64,
    ) -> bool {
        // Fast path: check if any update is needed
        if let Some(gpu_geo) = self.resources.geometry_mut(handle)
            && gpu_geo.asset_version == asset_version
            && gpu_geo.asset_identity == asset_identity
            && geometry.structure_version() == gpu_geo.version
            && geometry.data_version() == gpu_geo.last_data_version
        {
            gpu_geo.last_used_frame = self.frame_index;
            return true;
        }

        // Ensure all attribute buffers
        let mut any_buffer_recreated = false;
        let mut new_vertex_ids = Vec::new();

        for attr in geometry.attributes().values() {
            let result = self.prepare_attribute(attr);
            new_vertex_ids.push(result.resource_id);
            if result.was_recreated {
                any_buffer_recreated = true;
            }
        }

        if let Some(indices) = geometry.index_attribute() {
            let result = self.prepare_index(indices);
            if result.was_recreated {
                any_buffer_recreated = true;
            }
        }

        // Check if GpuGeometry needs to be rebuilt
        let needs_rebuild = if let Some(gpu_geo) = self.resources.geometry(handle) {
            gpu_geo.asset_version != asset_version
                || gpu_geo.asset_identity != asset_identity
                || geometry.structure_version() != gpu_geo.version
                || any_buffer_recreated
                || gpu_geo.vertex_buffer_ids != new_vertex_ids
        } else {
            true
        };

        if needs_rebuild {
            self.create_gpu_geometry(geometry, handle, asset_version, asset_identity);
        } else {
            // Only update the data version
            if let Some(gpu_geo) = self.resources.geometry_mut(handle) {
                gpu_geo.last_data_version = geometry.data_version();
                gpu_geo.asset_version = asset_version;
                gpu_geo.asset_identity = asset_identity;
            }
        }

        if let Some(gpu_geo) = self.resources.geometry_mut(handle) {
            gpu_geo.last_used_frame = self.frame_index;
            true
        } else {
            false
        }
    }

    fn create_gpu_geometry(
        &mut self,
        geometry: &Geometry,
        handle: GeometryHandle,
        asset_version: u32,
        asset_identity: u64,
    ) {
        let layout_info = crate::pipeline::vertex::generate_vertex_layout(geometry);

        let layout_id = self.get_or_create_vertex_layout_id(&layout_info);

        let mut vertex_buffers = Vec::new();
        let mut vertex_buffer_ids = Vec::new();

        for layout_desc in &layout_info.buffers {
            let gpu_buf = self
                .get_gpu_buffer_by_cpu_id(layout_desc.buffer.id())
                .expect("Vertex buffer should be prepared");
            vertex_buffers.push(gpu_buf.buffer.clone());
            vertex_buffer_ids.push(gpu_buf.id);
        }

        let index_buffer = if let Some(indices) = geometry.index_attribute() {
            let gpu_buf = self
                .get_gpu_buffer_by_cpu_id(indices.buffer.id())
                .expect("Index buffer should be prepared");
            Some((
                gpu_buf.buffer.clone(),
                indices.format,
                indices.count,
                gpu_buf.id,
            ))
        } else {
            None
        };

        let mut draw_range = geometry.draw_range.clone();
        if draw_range == (0..u32::MAX) {
            if let Some(attr) = geometry.attributes().get("position") {
                draw_range = draw_range.start..std::cmp::min(attr.count, draw_range.end);
            } else if let Some(attr) = geometry.attributes().values().next() {
                draw_range = draw_range.start..std::cmp::min(attr.count, draw_range.end);
            } else {
                draw_range = 0..0;
            }
        }

        let gpu_geo = GpuGeometry {
            asset_version,
            asset_identity,
            layout_info: Arc::new(layout_info),
            layout_id,
            shader_defines: Arc::new(geometry.shader_defines().clone()),
            vertex_buffers,
            vertex_buffer_ids,
            index_buffer,
            draw_range,
            instance_range: 0..1,
            version: geometry.structure_version(),
            last_data_version: geometry.data_version(),
            last_used_frame: self.frame_index,
        };

        self.resources.insert_geometry(handle, gpu_geo);
    }

    pub fn get_geometry(&self, handle: GeometryHandle) -> Option<&GpuGeometry> {
        self.resources.geometry(handle)
    }

    pub fn get_or_create_vertex_layout_id(&mut self, layout: &GeneratedVertexLayout) -> u64 {
        let signature = layout.to_signature();
        self.bind_groups.get_or_create_vertex_layout_id(signature)
    }
}
