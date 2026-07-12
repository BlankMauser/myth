//! Persistent asset mipmap generation through RDG.
//!
//! Texture upload remains owned by `ResourceManager`, but GPU mip generation is
//! deferred into this feature so it participates in the frame command encoder
//! instead of submitting work during resource preparation.

use crate::core::ResourceManager;
use crate::core::gpu::{MipmapBatchId, MipmapGenerator, MipmapRequest};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{ExecuteContext, PassNode, PrepareContext, TextureNodeId};

/// Drop-bearing physical attempts captured at Composer time. The snapshot is
/// immediately moved into GraphStorage; arena pass nodes retain only a Copy
/// [`MipmapBatchId`].
pub(crate) struct MipmapSnapshot {
    requests: Vec<MipmapRequest>,
}

impl MipmapSnapshot {
    fn from_requests(requests: Vec<MipmapRequest>) -> Self {
        Self { requests }
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
        MipmapSnapshot::from_requests(resource_manager.take_mipmap_requests())
    }

    pub(crate) fn generator_mut(&mut self) -> &mut MipmapGenerator {
        &mut self.generator
    }

    pub(crate) fn add_to_graph(ctx: &mut GraphBuilderContext<'_, '_>, batch: MipmapBatchId) {
        ctx.graph.add_pass("Asset_Mipmap_Generation", |builder| {
            // Compatibility callers of ResourceManager::prepare_texture still
            // have no logical RDG consumer edge. Keep that isolated queue as a
            // side effect until those public/legacy entry points are migrated.
            builder.mark_side_effect();
            (AssetMipmapNode { batch }, ())
        });
    }

    /// Build the first-class logical asset texture chain.
    ///
    /// `Asset_Texture_Upload` declares the imported base-level producer; the
    /// post-DPE graph batch fills its attempt-owned physical slot before node
    /// prepare. When the source requests mips, the second node mutates that
    /// logical version into `MipReady`; a sampling consumer that reads the
    /// returned alias therefore orders upload, generation, and draw through
    /// ordinary physical dependencies.
    ///
    /// The attempt-local import entry coordinates the producer aliases as well
    /// as the logical root. Repeated calls therefore share one SSA chain even
    /// when a material and Skybox consume the same exact snapshot.
    ///
    pub(crate) fn add_asset_texture_to_graph(
        ctx: &mut GraphBuilderContext<'_, '_>,
        imported: TextureNodeId,
        needs_mipmap_generation: bool,
    ) -> TextureNodeId {
        ctx.graph.assert_topology_mutable();
        let import = ctx.graph.storage.resources[imported.index() as usize]
            .asset_texture_import_id()
            .expect("asset texture producer target is not an asset import");
        let (existing_base, existing_mip) = ctx
            .graph
            .storage
            .asset_texture_imports
            .producer_chain(import);
        let base_ready = if let Some(base_ready) = existing_base {
            base_ready
        } else {
            let base_ready = ctx.graph.add_pass("Asset_Texture_Upload", |builder| {
                builder.write_texture(imported);
                (AssetTextureUploadNode { texture: imported }, imported)
            });
            ctx.graph
                .storage
                .asset_texture_imports
                .set_base_ready(import, base_ready);
            base_ready
        };

        if !needs_mipmap_generation {
            return base_ready;
        }

        if let Some(mip_ready) = existing_mip {
            mip_ready
        } else {
            let mip_ready = ctx
                .graph
                .add_pass("Asset_Texture_Mipmap_Generation", |builder| {
                    let output = builder.mutate_texture(base_ready, "Asset_Texture_MipReady");
                    (
                        AssetTextureMipmapNode {
                            texture: base_ready,
                        },
                        output,
                    )
                });
            ctx.graph
                .storage
                .asset_texture_imports
                .set_mip_ready(import, mip_ready);
            mip_ready
        }
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
        assert!(
            ctx.views.asset_textures.materialization(import).is_some(),
            "live Asset Texture imports must be batch-materialized before PassNode::prepare"
        );
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
            if !ctx.try_claim_mipmap_encoding(request.key()) {
                continue;
            }
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
            if !ctx.try_claim_mipmap_encoding(request.key()) {
                continue;
            }
            ctx.mipmap_generator
                .generate(ctx.device, encoder, &request.texture);
        }
        ctx.mipmap_requests.mark_encoded(self.batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::snapshot_asset_texture;
    use crate::graph::core::graph::FrameConfig;
    use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};
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
    fn asset_texture_producer_chain_is_idempotent_monotonic_and_dpe_pure() {
        let assets = AssetServer::new();
        let image = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![255; 8 * 8 * 4]),
        ));
        let mut texture = Texture::new_2d(Some("coordinated asset texture"), image);
        texture.generate_mipmaps = true;
        let texture = assets.textures.add(texture);
        let snapshot = snapshot_asset_texture(&assets, texture).expect("snapshot missing");

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let config = frame_config();
        let (imported, base_ready, mip_ready) = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            let imported = ctx.graph.import_asset_texture("Asset", snapshot);
            let base_ready = MipmapFeature::add_asset_texture_to_graph(&mut ctx, imported, false);
            let mip_ready = MipmapFeature::add_asset_texture_to_graph(&mut ctx, imported, true);
            (imported, base_ready, mip_ready)
        };
        let repeated_mip_ready = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            MipmapFeature::add_asset_texture_to_graph(&mut ctx, imported, true)
        };
        let repeated_base_ready = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            MipmapFeature::add_asset_texture_to_graph(&mut ctx, imported, false)
        };

        assert_ne!(base_ready, mip_ready);
        assert_eq!(mip_ready, repeated_mip_ready);
        assert_eq!(base_ready, repeated_base_ready);
        assert_eq!(
            graph
                .storage
                .passes
                .iter()
                .filter(|pass| pass.name == "Asset_Texture_Upload")
                .count(),
            1
        );
        let mipmap_passes: Vec<_> = graph
            .storage
            .passes
            .iter()
            .filter(|pass| pass.name == "Asset_Texture_Mipmap_Generation")
            .collect();
        assert_eq!(mipmap_passes.len(), 1);
        assert!(
            !mipmap_passes[0].has_side_effect,
            "logical Asset Texture mip work must remain demand-driven"
        );
    }

    #[test]
    #[should_panic(expected = "render graph topology is frozen after compilation")]
    fn asset_texture_producer_fast_path_rejects_post_compile_reuse() {
        let assets = AssetServer::new();
        let image = assets.images.add(Image::new(
            4,
            4,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![255; 4 * 4 * 4]),
        ));
        let texture = assets
            .textures
            .add(Texture::new_2d(Some("frozen asset texture"), image));
        let snapshot = snapshot_asset_texture(&assets, texture).expect("snapshot missing");
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let config = frame_config();
        let imported = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            let imported = ctx.graph.import_asset_texture("Asset", snapshot);
            MipmapFeature::add_asset_texture_to_graph(&mut ctx, imported, false);
            imported
        };
        graph.compile_topology();

        let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
        MipmapFeature::add_asset_texture_to_graph(&mut ctx, imported, false);
    }
}
