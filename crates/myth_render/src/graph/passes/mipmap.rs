//! Persistent asset mipmap generation through RDG.
//!
//! Texture upload remains owned by `ResourceManager`, but GPU mip generation is
//! deferred into this feature so it participates in the frame command encoder
//! instead of submitting work during resource preparation.

use crate::core::ResourceManager;
use crate::core::gpu::{MipmapBatchId, MipmapGenerator, MipmapRequest};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{ExecuteContext, PassNode, PrepareContext};

/// Drop-bearing physical attempts captured at Composer time. The snapshot is
/// immediately moved into GraphStorage; arena pass nodes retain only a Copy
/// [`MipmapBatchId`].
pub(crate) struct MipmapSnapshot {
    requests: Vec<MipmapRequest>,
}

impl MipmapSnapshot {
    #[must_use]
    pub(crate) fn into_requests(self) -> Vec<MipmapRequest> {
        self.requests
    }
}

pub struct MipmapFeature {
    generator: MipmapGenerator,
}

impl MipmapFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            generator: MipmapGenerator::new(device),
        }
    }

    /// Drain the requests accumulated by actual texture preparation during
    /// this frame. This is CPU ownership transfer only; no WGPU work occurs.
    #[must_use]
    #[allow(clippy::unused_self)] // Keep the uniform Feature::extract API.
    pub(crate) fn extract(&self, resource_manager: &mut ResourceManager) -> MipmapSnapshot {
        MipmapSnapshot {
            requests: resource_manager.take_mipmap_requests(),
        }
    }

    pub(crate) fn generator_mut(&mut self) -> &mut MipmapGenerator {
        &mut self.generator
    }

    pub(crate) fn add_to_graph(ctx: &mut GraphBuilderContext<'_, '_>, batch: MipmapBatchId) {
        ctx.graph.add_pass("Asset_Mipmap_Generation", |builder| {
            // Asset textures are not logical RDG imports yet, so this producer
            // remains an explicit side effect until every sampling consumer can
            // declare a real per-texture dependency.
            builder.mark_side_effect();
            (AssetMipmapNode { batch }, ())
        });
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn generator_initialized(&self) -> bool {
        self.generator.is_initialized()
    }
}

struct AssetMipmapNode {
    batch: MipmapBatchId,
}

impl PassNode<'_> for AssetMipmapNode {
    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        for request in ctx.mipmap_requests.requests(self.batch) {
            ctx.mipmap_generator
                .ensure_pipeline(ctx.device, request.format);
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        for request in ctx.mipmap_requests.requests(self.batch) {
            ctx.mipmap_generator
                .generate(ctx.device, encoder, &request.texture);
        }
        ctx.mipmap_requests.mark_encoded(self.batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::graph::FrameConfig;
    use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};

    fn frame_config() -> FrameConfig {
        FrameConfig {
            width: 16,
            height: 16,
            depth_format: wgpu::TextureFormat::Depth32Float,
            msaa_samples: 1,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            hdr_format: crate::HDR_TEXTURE_FORMAT,
        }
    }

    #[test]
    fn asset_mipmap_node_is_zero_drop() {
        assert!(!std::mem::needs_drop::<AssetMipmapNode>());
    }

    #[test]
    fn asset_mipmap_batch_remains_an_explicit_side_effect() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let config = frame_config();
        {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            MipmapFeature::add_to_graph(&mut ctx, MipmapBatchId::from_index(0));
        }

        graph.compile_topology();

        assert_eq!(graph.storage.execution_queue.len(), 1);
        assert_eq!(
            graph.storage.passes[graph.storage.execution_queue[0]].name,
            "Asset_Mipmap_Generation"
        );
    }
}
