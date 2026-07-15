//! Deterministic, allocation-free backend policy for GPU-driven submission.
//!
//! The policy is intentionally a pure CPU coordinator. It never queries WGPU,
//! mutates GPU Scene authority, or advances hysteresis before a frame reaches
//! the queue-submitted boundary.

use bitflags::bitflags;

use crate::settings::RenderPath;

const DOMAIN_CAPACITY: usize = 8;
const ENTER_CONFIRMATIONS: u8 = 3;
const EXIT_CONFIRMATIONS: u8 = 5;
const MINIMUM_DWELL_SUBMISSIONS: u64 = 30;
const CPU_RECHECK_INTERVAL: u64 = 30;
const MINIMUM_AUTO_CANDIDATES: u32 = 32;

bitflags! {
    /// Independently selectable semantic scene-submission families. Main and
    /// Prepass share one visibility result and therefore one policy bit.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub(crate) struct GpuDrivenFamilyMask: u8 {
        const MAIN_PREPASS = 1 << 0;
        const SHADOW_2D = 1 << 1;
        const SHADOW_CUBE = 1 << 2;
    }
}

impl GpuDrivenFamilyMask {
    pub(crate) const ALL: Self = Self::from_bits_retain(
        Self::MAIN_PREPASS.bits() | Self::SHADOW_2D.bits() | Self::SHADOW_CUBE.bits(),
    );

    pub(crate) const ORDERED: [Self; 3] = [Self::MAIN_PREPASS, Self::SHADOW_2D, Self::SHADOW_CUBE];

    #[inline]
    const fn bit_index(bit: Self) -> usize {
        match bit.bits() {
            bits if bits == Self::MAIN_PREPASS.bits() => 0,
            bits if bits == Self::SHADOW_2D.bits() => 1,
            bits if bits == Self::SHADOW_CUBE.bits() => 2,
            _ => panic!("GPU-driven policy requires one family bit"),
        }
    }
}

/// Cheap facts accumulated inside the existing extraction loop. No extra
/// scene scan, allocation, or asset-dictionary lookup is required.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GpuDrivenWorkloadSynopsis {
    pub(crate) extracted_item_count: u32,
    pub(crate) opaque_item_count: u32,
    pub(crate) supported_opaque_count: u32,
    pub(crate) shadow_caster_count: u32,
    pub(crate) supported_shadow_caster_count: u32,
    pub(crate) transform_changed_count: u32,
    pub(crate) indexed_primitive_count: u64,
    pub(crate) shadow_2d_view_count: u32,
    pub(crate) shadow_cube_view_count: u32,
    /// Stable fixed-state fingerprint. Transform matrices are deliberately
    /// excluded; `transform_changed_count` participates in `workload_key()`.
    pub(crate) topology_fingerprint: u64,
}

impl GpuDrivenWorkloadSynopsis {
    #[inline]
    pub(crate) const fn has_gpu_candidates(self) -> bool {
        self.supported_opaque_count != 0
    }

    #[inline]
    pub(crate) const fn workload_key(self) -> u64 {
        let key = mix64(
            self.topology_fingerprint,
            (self.transform_changed_count as u64) << 32 | self.supported_opaque_count as u64,
        );
        mix64(
            key,
            (self.shadow_2d_view_count as u64) << 32 | self.shadow_cube_view_count as u64,
        )
    }
}

/// Cross-frame policy identity. Pipeline epoch prevents a decision calibrated
/// for one shader/fixed-state generation from leaking into another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GpuDrivenPolicyDomainKey {
    pub(crate) scene_id: u32,
    pub(crate) camera_id: u128,
    pub(crate) render_path: RenderPath,
    pub(crate) msaa_samples: u32,
    pub(crate) viewport_width: u32,
    pub(crate) viewport_height: u32,
    pub(crate) pipeline_epoch: u64,
}

/// Exact mutation-free projection produced only after the cheap synopsis gate
/// admits a workload.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GpuDrivenExactWorkload {
    pub(crate) candidate_count: u32,
    pub(crate) bin_count: u32,
    pub(crate) unique_geometry_count: u32,
    pub(crate) unique_material_count: u32,
    pub(crate) shadow_candidate_count: u32,
    pub(crate) instance_slot_span: u32,
    pub(crate) batch_slot_span: u32,
    pub(crate) dirty_candidate_count: u32,
    pub(crate) indexed_primitive_count: u64,
    /// Main/Prepass semantic consumers sharing one visibility result: one for
    /// BasicForward, two for HighFidelity.
    pub(crate) main_submission_count: u32,
    pub(crate) shadow_2d_view_count: u32,
    pub(crate) shadow_cube_view_count: u32,
    pub(crate) viewport_pixels: u64,
    /// 0 = conservative/unknown, 1 = portable indirect, 2 = enabled
    /// indirect-count capability. This is a capability prior, not an attempt
    /// to infer integrated versus discrete hardware from feature bits.
    pub(crate) device_tier: u8,
    /// Families whose projected buffers/bindings fit the active device.
    pub(crate) capability_families: GpuDrivenFamilyMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuDrivenAutoDecisionReason {
    SmallWorkload,
    CostModel,
    Hysteresis,
    LowConfidence,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum GpuDrivenCoarseGate {
    EvaluateExact { minimum_cohort_members: u32 },
    Decide(GpuDrivenAutoDecision),
}

/// Opaque transaction capability. It can be committed exactly once against
/// the unchanged policy epoch/state and only after queue submission succeeds.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuDrivenAutoDecisionToken {
    policy_epoch: u64,
    expected_submission_serial: u64,
    expected_revision: u64,
    next: DomainState,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuDrivenAutoDecision {
    pub(crate) families: GpuDrivenFamilyMask,
    pub(crate) reason: GpuDrivenAutoDecisionReason,
    pub(crate) minimum_cohort_members: u32,
    pub(crate) token: GpuDrivenAutoDecisionToken,
}

#[derive(Debug, Clone, Copy)]
struct DomainState {
    key: GpuDrivenPolicyDomainKey,
    selected: GpuDrivenFamilyMask,
    enter_streaks: [u8; 3],
    exit_streaks: [u8; 3],
    last_switch_submissions: [u64; 3],
    switched_once: GpuDrivenFamilyMask,
    workload_key: u64,
    last_exact_submission: u64,
    last_exact_proposal: GpuDrivenFamilyMask,
    had_exact_decision: bool,
    last_seen_submission: u64,
    revision: u64,
}

impl DomainState {
    const fn new(key: GpuDrivenPolicyDomainKey) -> Self {
        Self {
            key,
            selected: GpuDrivenFamilyMask::empty(),
            enter_streaks: [0; 3],
            exit_streaks: [0; 3],
            last_switch_submissions: [0; 3],
            switched_once: GpuDrivenFamilyMask::empty(),
            workload_key: 0,
            last_exact_submission: 0,
            last_exact_proposal: GpuDrivenFamilyMask::empty(),
            had_exact_decision: false,
            last_seen_submission: 0,
            revision: 0,
        }
    }
}

/// Renderer-owned fixed-capacity adaptive state. Eight domains cover common
/// alternating camera/scene workloads without introducing a hot-path HashMap.
pub(crate) struct GpuDrivenAutoPolicy {
    domains: [Option<DomainState>; DOMAIN_CAPACITY],
    policy_epoch: u64,
    submission_serial: u64,
}

impl Default for GpuDrivenAutoPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDrivenAutoPolicy {
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            domains: [None; DOMAIN_CAPACITY],
            policy_epoch: 1,
            submission_serial: 0,
        }
    }

    /// Invalidate pending tokens and calibration state after an explicit
    /// policy/fixed-state contract change.
    pub(crate) fn bump_epoch(&mut self) {
        self.policy_epoch = self
            .policy_epoch
            .checked_add(1)
            .expect("GPU-driven policy epoch exhausted");
        self.domains.fill(None);
    }

    /// Cheap pre-preflight gate. Stable CPU decisions are periodically
    /// rechecked so Auto can adapt without paying O(N) preflight every frame.
    pub(crate) fn coarse_gate(
        &self,
        key: GpuDrivenPolicyDomainKey,
        synopsis: GpuDrivenWorkloadSynopsis,
    ) -> GpuDrivenCoarseGate {
        let minimum_cohort_members = cohort_threshold(synopsis.supported_opaque_count);
        let state = self.state_for(key);
        let no_promising_work = !synopsis.has_gpu_candidates()
            || synopsis.supported_opaque_count < MINIMUM_AUTO_CANDIDATES;
        if no_promising_work {
            if !state.selected.is_empty() {
                // A resident GPU choice still needs an exact recipe until the
                // submitted exit confirmation and dwell contract release it.
                return GpuDrivenCoarseGate::EvaluateExact {
                    minimum_cohort_members: 1,
                };
            }
            return GpuDrivenCoarseGate::Decide(self.decision_for_proposal(
                state,
                GpuDrivenFamilyMask::empty(),
                GpuDrivenFamilyMask::ALL,
                synopsis.workload_key(),
                minimum_cohort_members,
                GpuDrivenAutoDecisionReason::SmallWorkload,
                false,
            ));
        }

        if state.had_exact_decision
            && state.selected.is_empty()
            && state.last_exact_proposal.is_empty()
            && state.workload_key == synopsis.workload_key()
            && self
                .submission_serial
                .saturating_sub(state.last_exact_submission)
                < CPU_RECHECK_INTERVAL
        {
            return GpuDrivenCoarseGate::Decide(self.decision_for_proposal(
                state,
                GpuDrivenFamilyMask::empty(),
                GpuDrivenFamilyMask::ALL,
                synopsis.workload_key(),
                minimum_cohort_members,
                GpuDrivenAutoDecisionReason::Hysteresis,
                false,
            ));
        }

        GpuDrivenCoarseGate::EvaluateExact {
            minimum_cohort_members,
        }
    }

    pub(crate) fn evaluate_exact(
        &self,
        key: GpuDrivenPolicyDomainKey,
        synopsis: GpuDrivenWorkloadSynopsis,
        exact: GpuDrivenExactWorkload,
        minimum_cohort_members: u32,
    ) -> GpuDrivenAutoDecision {
        let state = self.state_for(key);
        let available = available_families(exact);
        let (proposal, reason) = select_cost_optimum(state.selected, exact);
        self.decision_for_proposal(
            state,
            proposal,
            available,
            synopsis.workload_key(),
            minimum_cohort_members,
            reason,
            true,
        )
    }

    /// Advance state only for a command buffer accepted by the queue. Surface
    /// acquisition failure, abandoned Composer, and pre-submit validation
    /// failure leave all confirmation/dwell counters unchanged.
    pub(crate) fn commit_submitted(&mut self, token: GpuDrivenAutoDecisionToken) -> bool {
        if token.policy_epoch != self.policy_epoch
            || token.expected_submission_serial != self.submission_serial.saturating_add(1)
        {
            return false;
        }

        let existing = self
            .domains
            .iter()
            .position(|state| state.is_some_and(|state| state.key == token.next.key));
        let slot = if let Some(slot) = existing {
            if self.domains[slot].is_none_or(|state| state.revision != token.expected_revision) {
                return false;
            }
            slot
        } else {
            if token.expected_revision != 0 {
                return false;
            }
            self.domains
                .iter()
                .position(Option::is_none)
                .unwrap_or_else(|| {
                    self.domains
                        .iter()
                        .enumerate()
                        .min_by_key(|(index, state)| {
                            (
                                state
                                    .expect("full policy cache lost a state")
                                    .last_seen_submission,
                                *index,
                            )
                        })
                        .map_or(0, |(index, _)| index)
                })
        };

        self.submission_serial = token.expected_submission_serial;
        let mut next = token.next;
        next.last_seen_submission = self.submission_serial;
        next.revision = token.expected_revision.saturating_add(1);
        self.domains[slot] = Some(next);
        true
    }

    #[inline]
    fn state_for(&self, key: GpuDrivenPolicyDomainKey) -> DomainState {
        self.domains
            .iter()
            .flatten()
            .find(|state| state.key == key)
            .copied()
            .unwrap_or_else(|| DomainState::new(key))
    }

    #[allow(clippy::too_many_arguments)]
    fn decision_for_proposal(
        &self,
        state: DomainState,
        proposal: GpuDrivenFamilyMask,
        available: GpuDrivenFamilyMask,
        workload_key: u64,
        minimum_cohort_members: u32,
        mut reason: GpuDrivenAutoDecisionReason,
        exact: bool,
    ) -> GpuDrivenAutoDecision {
        let expected_submission_serial = self
            .submission_serial
            .checked_add(1)
            .expect("GPU-driven policy submission serial exhausted");
        let mut next = state;
        let mut selected = state.selected;
        let proposal = proposal & available;

        for bit in GpuDrivenFamilyMask::ORDERED {
            let index = GpuDrivenFamilyMask::bit_index(bit);
            let current_gpu = selected.contains(bit);
            let proposed_gpu = proposal.contains(bit);
            if current_gpu && !available.contains(bit) {
                // Device/workload limits are a hard safety contract, not a
                // performance preference. Never let dwell or exit hysteresis
                // retain a family whose projected buffers no longer fit.
                selected.remove(bit);
                next.enter_streaks[index] = 0;
                next.exit_streaks[index] = 0;
                next.last_switch_submissions[index] = expected_submission_serial;
                next.switched_once.insert(bit);
                reason = GpuDrivenAutoDecisionReason::CostModel;
                continue;
            }
            if current_gpu == proposed_gpu {
                next.enter_streaks[index] = 0;
                next.exit_streaks[index] = 0;
                continue;
            }

            let dwell_complete = !next.switched_once.contains(bit)
                || expected_submission_serial.saturating_sub(next.last_switch_submissions[index])
                    >= MINIMUM_DWELL_SUBMISSIONS;
            if proposed_gpu {
                next.enter_streaks[index] = next.enter_streaks[index].saturating_add(1);
                next.exit_streaks[index] = 0;
                if dwell_complete && next.enter_streaks[index] >= ENTER_CONFIRMATIONS {
                    selected.insert(bit);
                    next.enter_streaks[index] = 0;
                    next.last_switch_submissions[index] = expected_submission_serial;
                    next.switched_once.insert(bit);
                } else {
                    reason = GpuDrivenAutoDecisionReason::Hysteresis;
                }
            } else {
                next.exit_streaks[index] = next.exit_streaks[index].saturating_add(1);
                next.enter_streaks[index] = 0;
                if dwell_complete && next.exit_streaks[index] >= EXIT_CONFIRMATIONS {
                    selected.remove(bit);
                    next.exit_streaks[index] = 0;
                    next.last_switch_submissions[index] = expected_submission_serial;
                    next.switched_once.insert(bit);
                } else {
                    reason = GpuDrivenAutoDecisionReason::Hysteresis;
                }
            }
        }

        next.selected = selected;
        next.workload_key = workload_key;
        if exact {
            next.had_exact_decision = true;
            next.last_exact_submission = expected_submission_serial;
            next.last_exact_proposal = proposal;
        }

        GpuDrivenAutoDecision {
            families: selected,
            reason,
            minimum_cohort_members,
            token: GpuDrivenAutoDecisionToken {
                policy_epoch: self.policy_epoch,
                expected_submission_serial,
                expected_revision: state.revision,
                next,
            },
        }
    }
}

fn select_cost_optimum(
    current: GpuDrivenFamilyMask,
    exact: GpuDrivenExactWorkload,
) -> (GpuDrivenFamilyMask, GpuDrivenAutoDecisionReason) {
    if exact.candidate_count == 0 || exact.bin_count == 0 {
        return (
            GpuDrivenFamilyMask::empty(),
            GpuDrivenAutoDecisionReason::SmallWorkload,
        );
    }

    let available = available_families(exact);
    let cpu_only = score_mask(GpuDrivenFamilyMask::empty(), available, exact);
    let mut best_mask = GpuDrivenFamilyMask::empty();
    let mut best_score = cpu_only;
    for raw in 1..=GpuDrivenFamilyMask::ALL.bits() {
        let mask = GpuDrivenFamilyMask::from_bits_retain(raw) & available;
        if mask.bits() != raw || mask.is_empty() {
            continue;
        }
        let score = score_mask(mask, available, exact);
        if score < best_score || (score == best_score && mask == current) {
            best_score = score;
            best_mask = mask;
        }
    }

    // Estimates within ten percent are deliberately treated as low
    // confidence. Retaining the current backend avoids oscillation and gives
    // benchmark calibration an explicit, observable region.
    let difference = cpu_only.abs_diff(best_score);
    let low_confidence = difference.saturating_mul(10) <= cpu_only.max(best_score);
    if low_confidence {
        return (
            current & available,
            GpuDrivenAutoDecisionReason::LowConfidence,
        );
    }
    (best_mask, GpuDrivenAutoDecisionReason::CostModel)
}

fn available_families(exact: GpuDrivenExactWorkload) -> GpuDrivenFamilyMask {
    let mut families = GpuDrivenFamilyMask::MAIN_PREPASS;
    if exact.shadow_candidate_count != 0 && exact.shadow_2d_view_count != 0 {
        families.insert(GpuDrivenFamilyMask::SHADOW_2D);
    }
    if exact.shadow_candidate_count != 0 && exact.shadow_cube_view_count != 0 {
        families.insert(GpuDrivenFamilyMask::SHADOW_CUBE);
    }
    families & exact.capability_families
}

fn score_mask(
    gpu: GpuDrivenFamilyMask,
    available: GpuDrivenFamilyMask,
    exact: GpuDrivenExactWorkload,
) -> u64 {
    let candidates = u64::from(exact.candidate_count);
    let shadows = u64::from(exact.shadow_candidate_count);
    let bins = u64::from(exact.bin_count);
    let slot_waste = u64::from(
        exact
            .instance_slot_span
            .saturating_sub(exact.candidate_count),
    );
    let device_fixed = match exact.device_tier {
        2 => 4_000,
        1 => 7_000,
        _ => 10_000,
    };
    let mut score = if gpu.is_empty() {
        0
    } else {
        device_fixed
            + u64::from(exact.dirty_candidate_count).saturating_mul(20)
            + bins.saturating_mul(180)
            + slot_waste.saturating_mul(3)
            + u64::from(
                exact
                    .unique_geometry_count
                    .saturating_add(exact.unique_material_count),
            )
            .saturating_mul(24)
    };

    for family in GpuDrivenFamilyMask::ORDERED {
        if !available.contains(family) {
            continue;
        }
        let (cpu_cost, gpu_cost) = if family == GpuDrivenFamilyMask::MAIN_PREPASS {
            let submissions = u64::from(exact.main_submission_count.max(1));
            let path_multiplier = if exact.viewport_pixels >= 8_000_000 {
                2
            } else {
                1
            };
            (
                candidates.saturating_mul(220).saturating_mul(submissions),
                4_000
                    + candidates.saturating_mul(9 * path_multiplier)
                    + bins.saturating_mul(900).saturating_mul(submissions),
            )
        } else {
            let views = if family == GpuDrivenFamilyMask::SHADOW_2D {
                u64::from(exact.shadow_2d_view_count)
            } else {
                u64::from(exact.shadow_cube_view_count)
            };
            (
                shadows.saturating_mul(views).saturating_mul(150),
                3_000
                    + shadows.saturating_mul(views).saturating_mul(7)
                    + bins.saturating_mul(views).saturating_mul(700),
            )
        };
        score = score.saturating_add(if gpu.contains(family) {
            gpu_cost
        } else {
            cpu_cost
        });
    }

    // This term is intentionally small: both backends execute the same
    // indexed vertex work, but the GPU path pays one extra visibility-table
    // indirection. It prevents tiny high-polygon hero cohorts winning merely
    // on draw-count arithmetic.
    if !gpu.is_empty() {
        score = score.saturating_add(exact.indexed_primitive_count / 256);
        score = score.saturating_add(u64::from(exact.batch_slot_span) * 4);
    }
    score
}

const fn cohort_threshold(candidate_count: u32) -> u32 {
    if candidate_count >= 4_096 {
        4
    } else if candidate_count >= 256 {
        8
    } else {
        16
    }
}

#[inline]
pub(crate) const fn mix64(seed: u64, value: u64) -> u64 {
    let mut mixed = seed ^ value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mixed ^= mixed >> 30;
    mixed = mixed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed ^= mixed >> 27;
    mixed = mixed.wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(scene_id: u32) -> GpuDrivenPolicyDomainKey {
        GpuDrivenPolicyDomainKey {
            scene_id,
            camera_id: 7,
            render_path: RenderPath::BasicForward,
            msaa_samples: 1,
            viewport_width: 1920,
            viewport_height: 1080,
            pipeline_epoch: 1,
        }
    }

    fn synopsis(candidates: u32, fingerprint: u64) -> GpuDrivenWorkloadSynopsis {
        GpuDrivenWorkloadSynopsis {
            extracted_item_count: candidates,
            opaque_item_count: candidates,
            supported_opaque_count: candidates,
            shadow_caster_count: candidates,
            supported_shadow_caster_count: candidates,
            transform_changed_count: 0,
            indexed_primitive_count: u64::from(candidates) * 12,
            shadow_2d_view_count: 4,
            shadow_cube_view_count: 0,
            topology_fingerprint: fingerprint,
        }
    }

    fn shared_exact(candidates: u32) -> GpuDrivenExactWorkload {
        GpuDrivenExactWorkload {
            candidate_count: candidates,
            bin_count: 1,
            unique_geometry_count: 1,
            unique_material_count: 1,
            shadow_candidate_count: candidates,
            instance_slot_span: candidates,
            batch_slot_span: 1,
            dirty_candidate_count: 0,
            indexed_primitive_count: u64::from(candidates) * 12,
            main_submission_count: 1,
            shadow_2d_view_count: 4,
            shadow_cube_view_count: 0,
            viewport_pixels: 1920 * 1080,
            device_tier: 2,
            capability_families: GpuDrivenFamilyMask::ALL,
        }
    }

    #[test]
    fn helmet_shape_never_enters_exact_preflight() {
        let policy = GpuDrivenAutoPolicy::new();
        let GpuDrivenCoarseGate::Decide(decision) = policy.coarse_gate(key(1), synopsis(1, 1))
        else {
            panic!("one candidate must stay on the CPU coarse path");
        };
        assert!(decision.families.is_empty());
        assert_eq!(decision.reason, GpuDrivenAutoDecisionReason::SmallWorkload);
    }

    #[test]
    fn shared_workload_enters_only_after_submitted_confirmations() {
        let mut policy = GpuDrivenAutoPolicy::new();
        for frame in 0..ENTER_CONFIRMATIONS {
            let GpuDrivenCoarseGate::EvaluateExact {
                minimum_cohort_members,
            } = policy.coarse_gate(key(1), synopsis(1_024, 2))
            else {
                panic!("shared workload unexpectedly skipped exact evaluation");
            };
            let decision = policy.evaluate_exact(
                key(1),
                synopsis(1_024, 2),
                shared_exact(1_024),
                minimum_cohort_members,
            );
            assert_eq!(
                decision
                    .families
                    .contains(GpuDrivenFamilyMask::MAIN_PREPASS),
                frame + 1 == ENTER_CONFIRMATIONS
            );
            assert!(policy.commit_submitted(decision.token));
        }
    }

    #[test]
    fn abandoned_decision_does_not_advance_confirmation() {
        let mut policy = GpuDrivenAutoPolicy::new();
        let workload = synopsis(1_024, 3);
        let GpuDrivenCoarseGate::EvaluateExact {
            minimum_cohort_members,
        } = policy.coarse_gate(key(1), workload)
        else {
            panic!();
        };
        let abandoned = policy.evaluate_exact(
            key(1),
            workload,
            shared_exact(1_024),
            minimum_cohort_members,
        );
        assert!(abandoned.families.is_empty());

        let repeated = policy.evaluate_exact(
            key(1),
            workload,
            shared_exact(1_024),
            minimum_cohort_members,
        );
        assert!(repeated.families.is_empty());
        assert!(policy.commit_submitted(repeated.token));
    }

    #[test]
    fn capability_loss_bypasses_exit_hysteresis() {
        let mut policy = GpuDrivenAutoPolicy::new();
        let workload = synopsis(1_024, 33);
        for _ in 0..ENTER_CONFIRMATIONS {
            let GpuDrivenCoarseGate::EvaluateExact {
                minimum_cohort_members,
            } = policy.coarse_gate(key(1), workload)
            else {
                panic!();
            };
            let decision = policy.evaluate_exact(
                key(1),
                workload,
                shared_exact(1_024),
                minimum_cohort_members,
            );
            assert!(policy.commit_submitted(decision.token));
        }
        assert!(
            policy
                .state_for(key(1))
                .selected
                .contains(GpuDrivenFamilyMask::MAIN_PREPASS)
        );

        let mut unavailable = shared_exact(1_024);
        unavailable.capability_families = GpuDrivenFamilyMask::empty();
        let decision = policy.evaluate_exact(key(1), workload, unavailable, 1);
        assert!(
            decision.families.is_empty(),
            "hard device limits must not wait for dwell or exit confirmations"
        );
        assert!(policy.commit_submitted(decision.token));
    }

    #[test]
    fn high_bin_diversity_stays_cpu_and_skips_repeat_preflight() {
        let mut policy = GpuDrivenAutoPolicy::new();
        let workload = synopsis(1_024, 4);
        let GpuDrivenCoarseGate::EvaluateExact {
            minimum_cohort_members,
        } = policy.coarse_gate(key(1), workload)
        else {
            panic!();
        };
        let mut exact = shared_exact(1_024);
        exact.bin_count = 1_024;
        exact.unique_geometry_count = 1_024;
        exact.unique_material_count = 1_024;
        exact.batch_slot_span = 1_024;
        let decision = policy.evaluate_exact(key(1), workload, exact, minimum_cohort_members);
        assert!(decision.families.is_empty());
        assert!(policy.commit_submitted(decision.token));
        assert!(matches!(
            policy.coarse_gate(key(1), workload),
            GpuDrivenCoarseGate::Decide(_)
        ));
    }

    #[test]
    fn shadow_view_topology_invalidates_a_cached_cpu_decision() {
        let mut policy = GpuDrivenAutoPolicy::new();
        let mut workload = synopsis(1_024, 44);
        workload.shadow_2d_view_count = 0;
        let GpuDrivenCoarseGate::EvaluateExact {
            minimum_cohort_members,
        } = policy.coarse_gate(key(1), workload)
        else {
            panic!();
        };
        let mut fragmented = shared_exact(1_024);
        fragmented.bin_count = 1_024;
        fragmented.unique_geometry_count = 1_024;
        fragmented.unique_material_count = 1_024;
        fragmented.batch_slot_span = 1_024;
        let decision = policy.evaluate_exact(key(1), workload, fragmented, minimum_cohort_members);
        assert!(decision.families.is_empty());
        assert!(policy.commit_submitted(decision.token));
        assert!(matches!(
            policy.coarse_gate(key(1), workload),
            GpuDrivenCoarseGate::Decide(_)
        ));

        workload.shadow_2d_view_count = 16;
        assert!(matches!(
            policy.coarse_gate(key(1), workload),
            GpuDrivenCoarseGate::EvaluateExact { .. }
        ));
    }

    #[test]
    fn shared_visibility_makes_high_fidelity_cross_over_before_basic_forward() {
        let mut basic = shared_exact(32);
        basic.shadow_2d_view_count = 0;
        basic.shadow_candidate_count = 0;
        basic.main_submission_count = 1;
        let (basic_proposal, _) = select_cost_optimum(GpuDrivenFamilyMask::empty(), basic);
        assert!(!basic_proposal.contains(GpuDrivenFamilyMask::MAIN_PREPASS));

        let mut high_fidelity = basic;
        high_fidelity.main_submission_count = 2;
        let (high_fidelity_proposal, _) =
            select_cost_optimum(GpuDrivenFamilyMask::empty(), high_fidelity);
        assert!(high_fidelity_proposal.contains(GpuDrivenFamilyMask::MAIN_PREPASS));
    }

    #[test]
    fn alternating_domains_keep_independent_confirmation_state() {
        let mut policy = GpuDrivenAutoPolicy::new();
        for scene in [1, 2, 1, 2, 1, 2] {
            let workload = synopsis(1_024, u64::from(scene));
            let GpuDrivenCoarseGate::EvaluateExact {
                minimum_cohort_members,
            } = policy.coarse_gate(key(scene), workload)
            else {
                panic!();
            };
            let decision = policy.evaluate_exact(
                key(scene),
                workload,
                shared_exact(1_024),
                minimum_cohort_members,
            );
            assert!(policy.commit_submitted(decision.token));
        }
        assert!(
            policy
                .state_for(key(1))
                .selected
                .contains(GpuDrivenFamilyMask::MAIN_PREPASS)
        );
        assert!(
            policy
                .state_for(key(2))
                .selected
                .contains(GpuDrivenFamilyMask::MAIN_PREPASS)
        );
    }

    #[test]
    fn stale_and_epoch_invalidated_tokens_are_rejected() {
        let mut policy = GpuDrivenAutoPolicy::new();
        let GpuDrivenCoarseGate::Decide(decision) = policy.coarse_gate(key(1), synopsis(1, 1))
        else {
            panic!();
        };
        policy.bump_epoch();
        assert!(!policy.commit_submitted(decision.token));
    }
}
