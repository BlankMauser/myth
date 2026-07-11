//! Persistent asset mipmap generation through RDG.
//!
//! Texture upload remains owned by `ResourceManager`, but GPU mip generation is
//! deferred into this feature so it participates in the frame command encoder
//! instead of submitting work during resource preparation.

use crate::core::ResourceManager;
use crate::core::gpu::{MipmapBatchId, MipmapGenerator, MipmapRequest, PhysicalImageGenerationKey};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{ExecuteContext, PassNode, PrepareContext, TextureNodeId};

/// Drop-bearing physical attempts captured at Composer time. The snapshot is
/// immediately moved into GraphStorage; arena pass nodes retain only a Copy
/// [`MipmapBatchId`].
pub(crate) struct MipmapSnapshot {
    requests: Vec<MipmapRequest>,
}

impl MipmapSnapshot {
    /// Transfer an already queued physical generation to a logical asset
    /// import. Any TextureHandle selecting the exact shared allocation may
    /// claim it; logical origin never participates in ownership.
    #[must_use]
    pub(crate) fn claim_for_generation(
        &mut self,
        generation: PhysicalImageGenerationKey,
    ) -> Option<MipmapRequest> {
        let index = self
            .requests
            .iter()
            .position(|request| request.key() == generation)?;
        Some(self.requests.swap_remove(index))
    }

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
            // Legacy material textures are not logical RDG imports yet, so
            // this compatibility producer remains an explicit side effect
            // until DrawPlan gives every sampling consumer a real edge.
            builder.mark_side_effect();
            (AssetMipmapNode { batch }, ())
        });
    }

    /// Build the first-class logical asset texture chain.
    ///
    /// `Asset_Texture_Upload` produces the imported base level in live
    /// prepare. When the source requests mips, the second node mutates that
    /// logical version into `MipReady`; a sampling consumer that reads the
    /// returned alias therefore orders upload, generation, and draw without a
    /// synthetic side-effect edge.
    ///
    /// A request claimed from the legacy material queue keeps the mip node as
    /// a side effect until DrawPlan gives those material consumers real asset
    /// texture reads. The Skybox-only path remains fully cullable.
    pub(crate) fn add_asset_texture_to_graph(
        ctx: &mut GraphBuilderContext<'_, '_>,
        imported: TextureNodeId,
        needs_mipmaps: bool,
        claimed_legacy_request: bool,
    ) -> TextureNodeId {
        let base_ready = ctx.graph.add_pass("Asset_Texture_Upload", |builder| {
            builder.write_texture(imported);
            (AssetTextureUploadNode { texture: imported }, imported)
        });

        if !needs_mipmaps && !claimed_legacy_request {
            return base_ready;
        }

        let mip_ready = ctx
            .graph
            .add_pass("Asset_Texture_Mipmap_Generation", |builder| {
                let output = builder.mutate_texture(base_ready, "Asset_Texture_MipReady");
                if claimed_legacy_request {
                    builder.mark_side_effect();
                }
                (
                    AssetTextureMipmapNode {
                        texture: base_ready,
                    },
                    output,
                )
            });

        if needs_mipmaps { mip_ready } else { base_ready }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn generator_initialized(&self) -> bool {
        self.generator.is_initialized()
    }
}

/// Live producer for one exact CPU-only asset texture snapshot.
///
/// The node is deliberately POD. Drop-bearing views and mip requests are
/// written into the GraphStorage-owned import slot.
struct AssetTextureUploadNode {
    texture: TextureNodeId,
}

impl PassNode<'_> for AssetTextureUploadNode {
    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let import = ctx
            .views
            .asset_texture_import_id(self.texture)
            .expect("asset upload node requires an Asset TextureBacking");
        let snapshot = ctx.views.asset_textures.snapshot(import);
        let materialization = ctx.uploads.materialize_asset_texture(ctx.assets, snapshot);
        ctx.views
            .asset_textures
            .materialize(import, materialization);
    }

    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
}

/// Generates every exact physical request owned by one asset import and
/// publishes those generations only after submission.
struct AssetTextureMipmapNode {
    texture: TextureNodeId,
}

impl PassNode<'_> for AssetTextureMipmapNode {
    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let import = ctx
            .views
            .asset_texture_import_id(self.texture)
            .expect("asset mipmap node requires an Asset TextureBacking");
        for request in ctx.views.asset_textures.mipmap_requests(import) {
            ctx.mipmap_generator
                .ensure_pipeline(ctx.device, request.format);
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let import = ctx
            .asset_texture_import_id(self.texture)
            .expect("asset mipmap node requires an Asset TextureBacking");
        let mut generated_any = false;
        for request in ctx.asset_textures.mipmap_requests(import) {
            ctx.mipmap_generator
                .generate(ctx.device, encoder, &request.texture);
            generated_any = true;
        }
        if generated_any {
            ctx.asset_textures.mark_mipmaps_encoded(import);
        }
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
    use crate::core::WgpuContext;
    use crate::core::gpu::snapshot_asset_texture;
    use crate::graph::core::graph::FrameConfig;
    use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};
    use crate::settings::{RendererInitConfig, RendererSettings};
    use myth_assets::AssetServer;
    use myth_resources::image::{Image, ImageDimension, PixelFormat};
    use myth_resources::texture::Texture;

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

    #[test]
    fn exact_generation_claim_is_independent_of_texture_handle() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![255; 8 * 8 * 4]),
        ));
        let mut producer_texture = Texture::new_2d(Some("mip producer"), image);
        producer_texture.generate_mipmaps = true;
        let producer = assets.textures.add(producer_texture);
        let mut consumer_texture = Texture::new_2d(Some("mip consumer"), image);
        consumer_texture.generate_mipmaps = true;
        let consumer = assets.textures.add(consumer_texture);

        resources.prepare_texture(&assets, producer);
        assert!(resources.get_texture_image(producer).is_some());
        assert!(resources.get_texture_image(consumer).is_none());
        let consumer_snapshot =
            snapshot_asset_texture(&assets, consumer).expect("consumer snapshot missing");
        let generation = resources
            .physical_generation_for_asset_snapshot(consumer_snapshot)
            .expect("exact shared generation missing");
        let mut snapshot = MipmapSnapshot {
            requests: resources.take_mipmap_requests(),
        };

        let claimed = snapshot
            .claim_for_generation(generation)
            .expect("another handle must claim the shared physical generation");
        assert_eq!(claimed.key(), generation);
        assert!(snapshot.into_requests().is_empty());
    }
}
