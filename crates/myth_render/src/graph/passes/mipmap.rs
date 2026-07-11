//! Persistent asset mipmap generation through RDG.
//!
//! Texture upload remains owned by `ResourceManager`, but GPU mip generation is
//! deferred into this feature so it participates in the frame command encoder
//! instead of submitting work during resource preparation.

use crate::core::gpu::{MipmapGenerator, MipmapRequest};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{ExecuteContext, ExtractContext, PassNode, PrepareContext};

pub struct MipmapFeature {
    generator: MipmapGenerator,
    requests: Vec<MipmapRequest>,
}

impl MipmapFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            generator: MipmapGenerator::new(device),
            requests: Vec::new(),
        }
    }

    /// Physically partition static generator state from the frame-owned
    /// request slice so graph nodes can retain the latter while live prepare
    /// mutates only the former.
    pub(crate) fn split(&mut self) -> (&mut MipmapGenerator, &[MipmapRequest]) {
        (&mut self.generator, &self.requests)
    }

    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.requests = ctx.resource_manager.take_mipmap_requests();
    }

    pub(crate) fn add_requests_to_graph<'a>(
        ctx: &mut GraphBuilderContext<'a, '_>,
        requests: &'a [MipmapRequest],
    ) {
        if requests.is_empty() {
            return;
        }

        ctx.graph.add_pass("Asset_Mipmap_Generation", |builder| {
            builder.mark_side_effect();
            (AssetMipmapNode { requests }, ())
        });
    }
}

struct AssetMipmapNode<'a> {
    requests: &'a [MipmapRequest],
}

impl PassNode<'_> for AssetMipmapNode<'_> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        for request in self.requests {
            ctx.mipmap_generator
                .ensure_pipeline(ctx.device, request.format);
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        for request in self.requests {
            ctx.mipmap_generator
                .generate(ctx.device, encoder, &request.texture);
            request.mark_complete();
        }
    }
}
