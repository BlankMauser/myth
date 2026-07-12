//! Persistent asset mipmap generation through RDG.
//!
//! Texture upload remains owned by `ResourceManager`, but GPU mip generation is
//! deferred into this feature so it participates in the frame command encoder
//! instead of submitting work during resource preparation.

use crate::core::ResourceManager;
use crate::core::gpu::{MipmapBatchId, MipmapGenerator, MipmapRequest, PhysicalImageGenerationKey};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{ExecuteContext, PassNode, PrepareContext, TextureNodeId};
use rustc_hash::FxHashMap;

/// Drop-bearing physical attempts captured at Composer time. The snapshot is
/// immediately moved into GraphStorage; arena pass nodes retain only a Copy
/// [`MipmapBatchId`].
pub(crate) struct MipmapSnapshot {
    requests: Vec<MipmapRequest>,
    request_lookup: FxHashMap<PhysicalImageGenerationKey, usize>,
}

impl MipmapSnapshot {
    fn from_requests(requests: Vec<MipmapRequest>) -> Self {
        let mut request_lookup =
            FxHashMap::with_capacity_and_hasher(requests.len(), rustc_hash::FxBuildHasher);
        for (index, request) in requests.iter().enumerate() {
            assert!(
                request_lookup.insert(request.key(), index).is_none(),
                "mipmap snapshot contains duplicate physical generations"
            );
        }
        Self {
            requests,
            request_lookup,
        }
    }

    /// Transfer an already queued physical generation to a logical asset
    /// import. Any TextureHandle selecting the exact shared allocation may
    /// claim it; logical origin never participates in ownership.
    #[must_use]
    pub(crate) fn claim_for_generation(
        &mut self,
        generation: PhysicalImageGenerationKey,
    ) -> Option<MipmapRequest> {
        let index = self.request_lookup.remove(&generation)?;
        let request = self.requests.swap_remove(index);
        if let Some(moved) = self.requests.get(index) {
            *self
                .request_lookup
                .get_mut(&moved.key())
                .expect("moved mipmap request missing from the lookup") = index;
        }
        Some(request)
    }

    #[must_use]
    pub(crate) fn into_requests(self) -> Vec<MipmapRequest> {
        debug_assert_eq!(self.requests.len(), self.request_lookup.len());
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
    /// The attempt-local import entry coordinates the producer aliases as well
    /// as the logical root. Repeated calls therefore share one SSA chain even
    /// when a material and Skybox consume the same exact snapshot.
    ///
    /// A request claimed from the legacy eager-material queue keeps the mip
    /// node as a compatibility side effect. DrawPlan consumers also attach a
    /// real read to the returned alias; the side effect can be removed with
    /// the eager material-preparation boundary in the next migration batch.
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
        let claimed_legacy_request = ctx
            .graph
            .storage
            .asset_texture_imports
            .has_claimed_mipmap_request(import);

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

        if !needs_mipmap_generation && !claimed_legacy_request {
            return base_ready;
        }

        let mip_ready = if let Some(mip_ready) = existing_mip {
            mip_ready
        } else {
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
            ctx.graph
                .storage
                .asset_texture_imports
                .set_mip_ready(import, mip_ready);
            mip_ready
        };

        if claimed_legacy_request {
            let producer = ctx.graph.storage.resources[mip_ready.index() as usize]
                .producer
                .expect("asset mip-ready alias must have a producer");
            ctx.graph.storage.passes[producer].has_side_effect = true;
        }

        if needs_mipmap_generation {
            mip_ready
        } else {
            base_ready
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
    fn asset_texture_producer_chain_is_idempotent_and_monotonic() {
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
        let mut texture = Texture::new_2d(Some("coordinated asset texture"), image);
        texture.generate_mipmaps = true;
        let texture = assets.textures.add(texture);
        let snapshot = snapshot_asset_texture(&assets, texture).expect("snapshot missing");
        resources.prepare_texture(&assets, texture);
        let request = resources
            .take_mipmap_requests()
            .pop()
            .expect("late compatibility request missing");
        let generation = request.key();

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
        graph.claim_asset_texture_mipmap_request(imported, generation, request);
        let mip_producer = graph.storage.resources[mip_ready.index() as usize]
            .producer
            .expect("mip-ready alias missing its producer");
        assert!(
            graph.storage.passes[mip_producer].has_side_effect,
            "claiming after chain creation must atomically upgrade its producer"
        );
        let claimed_ready = {
            let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
            MipmapFeature::add_asset_texture_to_graph(&mut ctx, imported, true)
        };

        assert_ne!(base_ready, mip_ready);
        assert_eq!(mip_ready, claimed_ready);
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
            mipmap_passes[0].has_side_effect,
            "a late compatibility claim must upgrade the existing producer"
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
        let mut snapshot = MipmapSnapshot::from_requests(resources.take_mipmap_requests());

        let claimed = snapshot
            .claim_for_generation(generation)
            .expect("another handle must claim the shared physical generation");
        assert_eq!(claimed.key(), generation);
        assert!(snapshot.into_requests().is_empty());
    }

    #[test]
    fn mipmap_snapshot_repairs_the_index_after_swap_remove() {
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
        let first_image = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![31; 8 * 8 * 4]),
        ));
        let second_image = assets.images.add(Image::new(
            4,
            4,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![63; 4 * 4 * 4]),
        ));
        let mut first = Texture::new_2d(Some("first indexed mip"), first_image);
        first.generate_mipmaps = true;
        let first = assets.textures.add(first);
        let mut second = Texture::new_2d(Some("second indexed mip"), second_image);
        second.generate_mipmaps = true;
        let second = assets.textures.add(second);
        resources.prepare_texture(&assets, first);
        resources.prepare_texture(&assets, second);
        let requests = resources.take_mipmap_requests();
        assert_eq!(requests.len(), 2);
        let keys = [requests[0].key(), requests[1].key()];
        let mut snapshot = MipmapSnapshot::from_requests(requests);

        assert_eq!(
            snapshot.claim_for_generation(keys[0]).unwrap().key(),
            keys[0]
        );
        assert_eq!(
            snapshot.claim_for_generation(keys[1]).unwrap().key(),
            keys[1]
        );
        assert!(snapshot.into_requests().is_empty());
    }
}
