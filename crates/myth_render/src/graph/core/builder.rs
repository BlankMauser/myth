use crate::core::gpu::GlobalStateKey;
use crate::core::gpu::Tracked;

use super::graph::RenderGraph;
use super::scene_submission::SceneSubmissionPlanId;
use super::types::{
    BufferDesc, BufferNodeId, GraphResourceType, ResourceClass, ResourceNodeId, TextureDesc,
    TextureNodeId,
};

/// Builder for declaring a pass's resource dependencies during eager graph
/// construction.
///
/// Obtained exclusively inside the closure passed to
/// [`RenderGraph::add_pass`].  All topology wiring — resource creation, read
/// / write declarations, and alias production — happens **immediately** and
/// is captured before the closure returns.
pub struct PassBuilder<'graph, 'a> {
    pub(crate) graph: &'graph mut RenderGraph<'a>,
    pub(crate) pass_index: usize,
}

impl PassBuilder<'_, '_> {
    pub fn create_texture(&mut self, name: &'static str, desc: TextureDesc) -> TextureNodeId {
        let id = self.graph.register_texture(name, desc, false);
        self.graph.storage.passes[self.pass_index]
            .creates
            .push(id.erase());
        self.write(id)
    }

    pub fn create_buffer(&mut self, name: &'static str, desc: BufferDesc) -> BufferNodeId {
        let id = self.graph.register_buffer(name, desc, false);
        self.graph.storage.passes[self.pass_index]
            .creates
            .push(id.erase());
        self.write(id)
    }

    pub fn read_external_texture(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
        view: &Tracked<wgpu::TextureView>,
    ) -> TextureNodeId {
        let id = self.graph.import_external_texture(name, desc, view);
        self.read(id)
    }

    pub fn write_external_texture(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
        view: &Tracked<wgpu::TextureView>,
    ) -> TextureNodeId {
        let id = self.graph.import_external_texture(name, desc, view);
        self.write(id)
    }

    pub fn read_external_buffer(
        &mut self,
        name: &'static str,
        desc: BufferDesc,
        buffer: &Tracked<wgpu::Buffer>,
    ) -> BufferNodeId {
        let id = self.graph.import_external_buffer(name, desc, buffer);
        self.read(id)
    }

    pub fn write_external_buffer(
        &mut self,
        name: &'static str,
        desc: BufferDesc,
        buffer: &Tracked<wgpu::Buffer>,
    ) -> BufferNodeId {
        let id = self.graph.import_external_buffer(name, desc, buffer);
        self.write(id)
    }

    pub fn read<T: GraphResourceType>(&mut self, id: ResourceNodeId<T>) -> ResourceNodeId<T> {
        let raw = id.erase();
        self.graph.storage.append_pass_read(self.pass_index, raw);
        id
    }

    pub fn write<T: GraphResourceType>(&mut self, id: ResourceNodeId<T>) -> ResourceNodeId<T> {
        let raw = id.erase();
        let res = &mut self.graph.storage.resources[raw.index() as usize];
        let resource_class = match res.class() {
            ResourceClass::Texture => "Texture",
            ResourceClass::Buffer => "Buffer",
        };

        assert!(
            res.asset_texture_import_id().is_none(),
            "Asset Texture import root '{}' is a read-only lazy external input; use builder.mutate_texture() to produce a derived version",
            res.name
        );
        assert!(
            res.scene_buffer_import_id().is_none(),
            "Scene Buffer import root '{}' is a read-only lazy external input",
            res.name
        );

        if let Some(existing_producer) = res.producer {
            panic!(
                "SSA Violation in Pass '{}': {} '{}' already has a producer (Pass '{}'). \
                 Use `builder.mutate()` to create a new version (alias).",
                self.graph.storage.passes[self.pass_index].name,
                resource_class,
                res.name,
                self.graph.storage.passes[existing_producer].name
            );
        }

        self.graph.storage.passes[self.pass_index].writes.push(raw);
        res.producer = Some(self.pass_index);
        id
    }

    #[inline]
    pub fn read_texture(&mut self, id: TextureNodeId) -> TextureNodeId {
        self.read(id)
    }

    #[inline]
    pub fn write_texture(&mut self, id: TextureNodeId) -> TextureNodeId {
        self.write(id)
    }

    #[inline]
    pub fn read_buffer(&mut self, id: BufferNodeId) -> BufferNodeId {
        self.read(id)
    }

    #[inline]
    pub fn write_buffer(&mut self, id: BufferNodeId) -> BufferNodeId {
        self.write(id)
    }

    #[must_use = "The returned resource handle must be used for downstream wiring"]
    pub fn mutate<T: GraphResourceType>(
        &mut self,
        input_id: ResourceNodeId<T>,
        new_name: &'static str,
    ) -> ResourceNodeId<T> {
        self.read(input_id);
        let new_id = self.graph.create_alias(input_id, new_name);
        self.write(new_id)
    }

    #[must_use = "The returned resource handle must be used for downstream wiring"]
    pub fn replace<T: GraphResourceType>(
        &mut self,
        input_id: ResourceNodeId<T>,
        new_name: &'static str,
    ) -> ResourceNodeId<T> {
        let new_id = self.graph.create_alias(input_id, new_name);
        self.write(new_id)
    }

    #[must_use = "The returned TextureNodeId must be used for downstream wiring"]
    pub fn mutate_texture(
        &mut self,
        input_id: TextureNodeId,
        new_name: &'static str,
    ) -> TextureNodeId {
        self.mutate(input_id, new_name)
    }

    #[must_use = "The returned TextureNodeId must be used for downstream wiring"]
    pub fn replace_texture(
        &mut self,
        input_id: TextureNodeId,
        new_name: &'static str,
    ) -> TextureNodeId {
        self.replace(input_id, new_name)
    }

    #[must_use = "The returned BufferNodeId must be used for downstream wiring"]
    pub fn mutate_buffer(
        &mut self,
        input_id: BufferNodeId,
        new_name: &'static str,
    ) -> BufferNodeId {
        self.mutate(input_id, new_name)
    }

    #[must_use = "The returned BufferNodeId must be used for downstream wiring"]
    pub fn replace_buffer(
        &mut self,
        input_id: BufferNodeId,
        new_name: &'static str,
    ) -> BufferNodeId {
        self.replace(input_id, new_name)
    }

    pub fn mark_side_effect(&mut self) {
        self.graph.storage.passes[self.pass_index].has_side_effect = true;
    }

    /// Declare that this pass consumes one backend-neutral scene submission.
    ///
    /// The hot path records only the opaque attempt-scoped ID. After setup,
    /// one-shot finalization deduplicates plans and attaches their ordered
    /// texture/buffer dependencies as ordinary logical reads. Demand metadata
    /// never roots an otherwise dead pass.
    #[allow(dead_code)] // Activated incrementally as Features migrate in M3.
    pub(crate) fn demand_scene_submission(&mut self, id: SceneSubmissionPlanId) {
        assert!(
            self.graph.storage.scene_submissions.validate_reserved(id),
            "pass '{}' demanded a stale or foreign scene-submission plan",
            self.graph.storage.passes[self.pass_index].name
        );
        self.graph.storage.scene_submission_raw_demands.push(id);
    }

    /// Declare that this pass consumes the scene-global Group 0 state.
    ///
    /// Demand metadata never roots a pass. Topology compilation projects the
    /// unique key only from passes already retained by DPE.
    pub(crate) fn demand_global_state(&mut self, key: GlobalStateKey) {
        let demand = &mut self.graph.storage.passes[self.pass_index].global_state_demand;
        match *demand {
            None => *demand = Some(key),
            Some(current) if current == key => {}
            Some(_) => panic!(
                "pass '{}' demanded two different global state keys",
                self.graph.storage.passes[self.pass_index].name
            ),
        }
    }

    /// Flags this pass as a pure forwarding (blit/present) node.
    ///
    /// The pass must read exactly one source and write exactly one
    /// destination of identical format and size.  At compile time the graph
    /// will attempt to fold the pass away via edge contraction — rewiring the
    /// source's producer to write the destination directly — so the forwarding
    /// copy carries no runtime cost in the common case.  See
    /// [`RenderGraph::fold_simple_passes`](super::graph::RenderGraph).
    pub fn mark_pure_forwarding(&mut self) {
        self.graph.storage.passes[self.pass_index].is_pure_forwarding = true;
    }
}
