//! Frame-owned Gaussian asset snapshots.
//!
//! Gaussian assets are reference-counted and therefore cannot be stored in the
//! no-drop [`FrameArena`](super::FrameArena). `GraphStorage` owns the exact
//! asset versions captured while the graph is built; pass nodes retain only a
//! Copy batch/index pair and resolve the assets during live preparation.

use std::sync::Arc;

use myth_resources::GaussianCloudHandle;
use myth_resources::gaussian_splat::GaussianCloud;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GaussianFrameBatchId(u32);

impl GaussianFrameBatchId {
    #[inline]
    #[must_use]
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("Gaussian frame batch index exceeds u32"))
    }

    #[inline]
    #[must_use]
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

pub(crate) struct GaussianFrameCloud {
    pub(crate) handle: GaussianCloudHandle,
    pub(crate) version: u32,
    pub(crate) asset: Arc<GaussianCloud>,
}

pub(crate) struct GaussianFrameBatch {
    clouds: Vec<GaussianFrameCloud>,
}

impl GaussianFrameBatch {
    #[must_use]
    pub(crate) fn new(clouds: Vec<GaussianFrameCloud>) -> Self {
        Self { clouds }
    }

    #[inline]
    #[must_use]
    fn cloud(&self, index: u32) -> &GaussianFrameCloud {
        &self.clouds[index as usize]
    }
}

/// Narrow exact-version asset lookup used only by Gaussian live prepare.
#[derive(Clone, Copy)]
pub(crate) struct GaussianFrameResolver<'a> {
    batches: &'a [Option<GaussianFrameBatch>],
}

impl<'a> GaussianFrameResolver<'a> {
    #[must_use]
    pub(crate) const fn new(batches: &'a [Option<GaussianFrameBatch>]) -> Self {
        Self { batches }
    }

    #[must_use]
    pub(crate) fn cloud(self, batch: GaussianFrameBatchId, index: u32) -> &'a GaussianFrameCloud {
        self.batches[batch.index()]
            .as_ref()
            .expect("Gaussian frame batch released before live prepare")
            .cloud(index)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};
    use myth_resources::image::ColorSpace;

    fn empty_cloud() -> GaussianCloud {
        GaussianCloud {
            gaussians: Vec::new(),
            sh_coefficients: Vec::new(),
            sh_degree: 0,
            num_points: 0,
            aabb_min: Vec3::ZERO,
            aabb_max: Vec3::ZERO,
            center: Vec3::ZERO,
            mip_splatting: false,
            kernel_size: 0.0,
            color_space: ColorSpace::Linear,
            opacity_compensation: 1.0,
        }
    }

    #[test]
    fn graph_batch_owns_exact_asset_until_explicit_release() {
        let handle = GaussianCloudHandle::from(slotmap::KeyData::from_ffi(0x0000_0001_0000_0001));
        let asset = Arc::new(empty_cloud());
        let retained = asset.clone();
        let mut batches = vec![Some(GaussianFrameBatch::new(vec![GaussianFrameCloud {
            handle,
            version: 7,
            asset,
        }]))];

        {
            let resolved =
                GaussianFrameResolver::new(&batches).cloud(GaussianFrameBatchId::from_index(0), 0);
            assert_eq!(resolved.handle, handle);
            assert_eq!(resolved.version, 7);
            assert!(Arc::ptr_eq(&resolved.asset, &retained));
            assert_eq!(Arc::strong_count(&retained), 2);
        }

        batches[0].take();
        assert_eq!(Arc::strong_count(&retained), 1);
    }

    #[test]
    fn render_graph_drop_releases_exact_asset_owner() {
        let handle = GaussianCloudHandle::from(slotmap::KeyData::from_ffi(0x0000_0001_0000_0001));
        let retained = Arc::new(empty_cloud());
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let batch = graph
                .stage_gaussian_frame_batch(vec![GaussianFrameCloud {
                    handle,
                    version: 7,
                    asset: retained.clone(),
                }])
                .expect("Gaussian frame batch must be staged");
            assert_eq!(batch.index(), 0);
            assert_eq!(Arc::strong_count(&retained), 2);
        }

        assert!(storage.gaussian_frame_batches.is_empty());
        assert_eq!(Arc::strong_count(&retained), 1);
    }
}
