//! Renderer-owned retained-capacity policy for GPU Scene CPU storage.
//!
//! GPU Scene authority outlives `GraphStorage`, so it cannot borrow the
//! graph-attempt governor directly. This sibling ledger uses the same policy
//! and runs at the same next-frame maintenance boundary. Authority trimming
//! can reclaim container headroom but never changes logical lengths or stable
//! identities. Scratch contents are discarded before their capacity shrinks;
//! persistent finalization L1 entries remain intact while only map headroom
//! participates in proportional maintenance.

#[cfg(test)]
use super::capacity::CapacityPolicy;
use super::capacity::{CapacityGovernor, CapacityGovernorStats, CapacityUsage};
use super::gpu_scene::GpuScene;
use super::gpu_scene_submission::GpuSceneRenderFeature;

const GPU_SCENE_CAPACITY_FAMILY_COUNT: usize = 3;

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuSceneCapacityFamily {
    Authority = 0,
    PlanningScratch = 1,
    Finalization = 2,
}

impl GpuSceneCapacityFamily {
    const fn index(self) -> usize {
        self as usize
    }
}

pub(crate) struct GpuSceneCapacityLedger {
    governor: CapacityGovernor<GPU_SCENE_CAPACITY_FAMILY_COUNT>,
}

impl Default for GpuSceneCapacityLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuSceneCapacityLedger {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            governor: CapacityGovernor::default(),
        }
    }

    #[cfg(test)]
    fn with_policy(policy: CapacityPolicy) -> Self {
        Self {
            governor: CapacityGovernor::new(policy),
        }
    }

    fn snapshot(
        gpu_scene: &GpuScene,
        render: &GpuSceneRenderFeature,
    ) -> [CapacityUsage; GPU_SCENE_CAPACITY_FAMILY_COUNT] {
        let mut usage = [CapacityUsage::default(); GPU_SCENE_CAPACITY_FAMILY_COUNT];
        usage[GpuSceneCapacityFamily::Authority.index()] = gpu_scene.authority_capacity_usage();
        usage[GpuSceneCapacityFamily::PlanningScratch.index()] = gpu_scene.scratch_capacity_usage();
        usage[GpuSceneCapacityFamily::Finalization.index()] = render.finalization_capacity_usage();
        usage
    }

    /// Observe the just-finished frame and start a fresh scratch lifetime.
    ///
    /// This is called immediately beside `GraphStorage::clear`, after Rust's
    /// borrow rules have guaranteed that no previous Composer/RenderGraph can
    /// remain live. A selected family is cleared before shrinking so retained
    /// capacity never forces an identity-bearing element to move or disappear.
    pub(crate) fn maintain(
        &mut self,
        gpu_scene: &mut GpuScene,
        render: &mut GpuSceneRenderFeature,
    ) {
        let observed = Self::snapshot(gpu_scene, render);
        self.governor.observe(observed);

        gpu_scene.clear_frame_scratch();
        render.clear_finalization_scratch();

        let Some(pending) = self.governor.take_pending_target() else {
            return;
        };
        let retained_before = observed[pending.family].retained_bytes;
        let target_bytes = pending.target_bytes.min(retained_before);
        match pending.family {
            family if family == GpuSceneCapacityFamily::PlanningScratch.index() => {
                gpu_scene.trim_scratch_capacity(target_bytes, retained_before);
            }
            family if family == GpuSceneCapacityFamily::Finalization.index() => {
                render.trim_finalization_capacity(target_bytes, retained_before);
            }
            family if family == GpuSceneCapacityFamily::Authority.index() => {
                gpu_scene.trim_authority_capacity(target_bytes, retained_before);
            }
            _ => unreachable!("GPU Scene capacity governor selected an unknown family"),
        }
        let retained_after = Self::snapshot(gpu_scene, render)[pending.family].retained_bytes;
        self.governor.complete_trim(retained_before, retained_after);
    }

    pub(crate) fn request_trim(&mut self) {
        self.governor.request_trim();
    }

    #[must_use]
    pub(crate) fn authority_usage(gpu_scene: &GpuScene) -> CapacityUsage {
        gpu_scene.authority_capacity_usage()
    }

    #[must_use]
    pub(crate) fn scratch_usage(
        gpu_scene: &GpuScene,
        render: &GpuSceneRenderFeature,
    ) -> CapacityUsage {
        gpu_scene.scratch_capacity_usage() + render.finalization_capacity_usage()
    }

    #[must_use]
    pub(crate) const fn stats(&self) -> CapacityGovernorStats {
        self.governor.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::RdgRetainedCapacityStats;

    const fn test_policy() -> CapacityPolicy {
        CapacityPolicy {
            minimum_family_bytes: 1,
            cold_numerator: 1,
            cold_denominator: 1,
            cold_attempts: 2,
            minimum_recoverable_bytes: 1,
            cooldown_attempts: 0,
        }
    }

    #[test]
    fn cold_trim_reclaims_only_scratch_and_preserves_generation_watermark_capacity() {
        let mut gpu_scene = GpuScene::new();
        let mut render = GpuSceneRenderFeature::new();
        gpu_scene.reserve_capacity_probe(16_384, 16);
        render.reserve_finalization_capacity_probe(16_384);

        let authority_before = gpu_scene.authority_capacity_usage();
        let scratch_before =
            gpu_scene.scratch_capacity_usage() + render.finalization_capacity_usage();
        assert!(authority_before.retained_bytes > 0);
        assert!(scratch_before.retained_bytes > 0);

        let mut ledger = GpuSceneCapacityLedger::with_policy(test_policy());
        ledger.maintain(&mut gpu_scene, &mut render);
        ledger.maintain(&mut gpu_scene, &mut render);

        let authority_after = gpu_scene.authority_capacity_usage();
        let scratch_after =
            gpu_scene.scratch_capacity_usage() + render.finalization_capacity_usage();
        assert_eq!(authority_after.active_bytes, authority_before.active_bytes);
        assert!(scratch_after.retained_bytes < scratch_before.retained_bytes);
        assert_eq!(ledger.stats().trim_count, 1);
        assert!(ledger.stats().last_trimmed_bytes > 0);
    }

    #[test]
    fn finalization_trim_preserves_persistent_prepass_l1_entries() {
        let mut render = GpuSceneRenderFeature::new();
        render.seed_prepass_pipeline_l1_capacity_probe(8);
        render.reserve_finalization_capacity_probe(16_384);
        let signature_before = render.prepass_pipeline_l1_capacity_probe_signature();

        render.clear_finalization_scratch();
        let usage_before = render.finalization_capacity_usage();
        render.trim_finalization_capacity(usage_before.active_bytes, usage_before.retained_bytes);

        assert_eq!(
            render.prepass_pipeline_l1_capacity_probe_signature(),
            signature_before,
            "capacity maintenance must shrink only L1 headroom, never entries"
        );
        assert!(
            render.finalization_capacity_usage().retained_bytes < usage_before.retained_bytes,
            "the oversized transient vectors should still be reclaimed"
        );
    }

    #[test]
    fn authority_is_accounted_and_only_excess_capacity_is_recoverable() {
        let mut gpu_scene = GpuScene::new();
        let render = GpuSceneRenderFeature::new();
        gpu_scene.reserve_capacity_probe(0, 4_096);
        let authority = gpu_scene.authority_capacity_usage();
        let scratch = gpu_scene.scratch_capacity_usage() + render.finalization_capacity_usage();
        assert!(authority.retained_bytes > 0);
        assert_eq!(authority.baseline_bytes, authority.active_bytes);
        assert!(authority.retained_bytes > authority.baseline_bytes);
        let retained_before = authority.retained_bytes;
        gpu_scene.trim_authority_capacity(authority.active_bytes, retained_before);
        let authority_after = gpu_scene.authority_capacity_usage();
        assert_eq!(
            authority_after.active_bytes, authority.active_bytes,
            "authority maintenance must preserve every logical element"
        );
        assert!(authority_after.retained_bytes < retained_before);

        let ledger = GpuSceneCapacityLedger::new();
        let stats = RdgRetainedCapacityStats::default().with_gpu_scene_capacity(
            authority_after,
            scratch,
            ledger.stats(),
        );
        assert_eq!(
            stats.estimated_retained_bytes(),
            authority_after.retained_bytes + scratch.retained_bytes
        );
        assert_eq!(
            stats.estimated_renderer_owned_bytes(),
            authority_after.retained_bytes + scratch.retained_bytes
        );
        assert_eq!(
            stats.estimated_gpu_scene_authority_bytes(),
            authority_after.retained_bytes
        );
        assert_eq!(
            stats.estimated_gpu_scene_scratch_bytes(),
            scratch.retained_bytes
        );
    }
}
