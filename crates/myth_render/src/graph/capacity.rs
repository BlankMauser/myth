//! Retained CPU-capacity accounting and hysteresis-based trimming.
//!
//! The helpers in this module deliberately account only for container backing
//! storage. Heap allocations owned by live elements are outside this ledger
//! and must have been released before a family is trimmed.

use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    hash::BuildHasher,
    mem::size_of,
    ops::{Add, AddAssign},
};

/// Accounted CPU bytes for one container or capacity family.
// Keep the unit in every field name: this type is mixed with element counts
// and capacities throughout the policy code, so explicit byte suffixes prevent
// accidental dimension errors.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapacityUsage {
    pub(crate) active_bytes: usize,
    pub(crate) retained_bytes: usize,
    pub(crate) baseline_bytes: usize,
}

impl CapacityUsage {
    #[must_use]
    pub(crate) const fn new(
        active_bytes: usize,
        retained_bytes: usize,
        baseline_bytes: usize,
    ) -> Self {
        Self {
            active_bytes,
            retained_bytes,
            baseline_bytes,
        }
    }

    #[must_use]
    pub(crate) const fn saturating_add(self, rhs: Self) -> Self {
        Self {
            active_bytes: self.active_bytes.saturating_add(rhs.active_bytes),
            retained_bytes: self.retained_bytes.saturating_add(rhs.retained_bytes),
            baseline_bytes: self.baseline_bytes.saturating_add(rhs.baseline_bytes),
        }
    }

    #[must_use]
    pub(crate) const fn recoverable_bytes(self, target_bytes: usize) -> usize {
        self.retained_bytes.saturating_sub(target_bytes)
    }
}

impl Add for CapacityUsage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.saturating_add(rhs)
    }
}

impl AddAssign for CapacityUsage {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.saturating_add(rhs);
    }
}

#[inline]
const fn bytes_for_slots(slots: usize, bytes_per_slot: usize) -> usize {
    slots.saturating_mul(bytes_per_slot)
}

/// Exact outer-allocation footprint for a `Vec`.
#[must_use]
pub(crate) fn vec_footprint<T>(value: &Vec<T>, baseline_capacity: usize) -> CapacityUsage {
    let slot = size_of::<T>();
    CapacityUsage::new(
        bytes_for_slots(value.len(), slot),
        bytes_for_slots(value.capacity(), slot),
        bytes_for_slots(baseline_capacity, slot),
    )
}

/// Exact outer-allocation footprint for a `BinaryHeap`.
#[must_use]
pub(crate) fn binary_heap_footprint<T>(
    value: &BinaryHeap<T>,
    baseline_capacity: usize,
) -> CapacityUsage {
    let slot = size_of::<T>();
    CapacityUsage::new(
        bytes_for_slots(value.len(), slot),
        bytes_for_slots(value.capacity(), slot),
        bytes_for_slots(baseline_capacity, slot),
    )
}

/// Estimated hash-table footprint.
///
/// `HashMap` does not expose its raw bucket allocation. The estimate charges
/// each reported capacity slot for its payload plus one control byte. It is a
/// stable policy metric, not an allocator-exact measurement.
#[must_use]
pub(crate) fn estimated_hash_map_footprint<K, V, S>(
    value: &HashMap<K, V, S>,
    baseline_capacity: usize,
) -> CapacityUsage {
    let slot = size_of::<(K, V)>().saturating_add(1);
    CapacityUsage::new(
        bytes_for_slots(value.len(), slot),
        bytes_for_slots(value.capacity(), slot),
        bytes_for_slots(baseline_capacity, slot),
    )
}

/// Estimated hash-set footprint; see [`estimated_hash_map_footprint`].
#[must_use]
pub(crate) fn estimated_hash_set_footprint<T, S>(
    value: &HashSet<T, S>,
    baseline_capacity: usize,
) -> CapacityUsage {
    let slot = size_of::<T>().saturating_add(1);
    CapacityUsage::new(
        bytes_for_slots(value.len(), slot),
        bytes_for_slots(value.capacity(), slot),
        bytes_for_slots(baseline_capacity, slot),
    )
}

fn proportional_capacity(
    current_capacity: usize,
    len: usize,
    baseline_capacity: usize,
    family_target_bytes: usize,
    family_retained_bytes: usize,
) -> usize {
    if family_retained_bytes == 0 || family_target_bytes >= family_retained_bytes {
        return current_capacity;
    }

    let numerator = (current_capacity as u128) * (family_target_bytes as u128);
    let denominator = family_retained_bytes as u128;
    let proportional = numerator.div_ceil(denominator).min(usize::MAX as u128) as usize;
    proportional
        .max(len)
        .max(baseline_capacity)
        .min(current_capacity)
}

/// Shrink a vector by its containing family's target/current-retained ratio.
pub(crate) fn shrink_vec_by_family_ratio<T>(
    value: &mut Vec<T>,
    baseline_capacity: usize,
    family_target_bytes: usize,
    family_retained_bytes: usize,
) -> bool {
    if size_of::<T>() == 0 {
        return false;
    }
    let old = value.capacity();
    let target = proportional_capacity(
        old,
        value.len(),
        baseline_capacity,
        family_target_bytes,
        family_retained_bytes,
    );
    value.shrink_to(target);
    value.capacity() < old
}

/// Shrink a heap by its containing family's target/current-retained ratio.
pub(crate) fn shrink_binary_heap_by_family_ratio<T>(
    value: &mut BinaryHeap<T>,
    baseline_capacity: usize,
    family_target_bytes: usize,
    family_retained_bytes: usize,
) -> bool {
    if size_of::<T>() == 0 {
        return false;
    }
    let old = value.capacity();
    let target = proportional_capacity(
        old,
        value.len(),
        baseline_capacity,
        family_target_bytes,
        family_retained_bytes,
    );
    value.shrink_to(target);
    value.capacity() < old
}

/// Shrink a map by its containing family's target/current-retained ratio.
pub(crate) fn shrink_hash_map_by_family_ratio<K: Eq + std::hash::Hash, V, S: BuildHasher>(
    value: &mut HashMap<K, V, S>,
    baseline_capacity: usize,
    family_target_bytes: usize,
    family_retained_bytes: usize,
) -> bool {
    let old = value.capacity();
    let target = proportional_capacity(
        old,
        value.len(),
        baseline_capacity,
        family_target_bytes,
        family_retained_bytes,
    );
    value.shrink_to(target);
    value.capacity() < old
}

/// Shrink a set by its containing family's target/current-retained ratio.
pub(crate) fn shrink_hash_set_by_family_ratio<T: Eq + std::hash::Hash, S: BuildHasher>(
    value: &mut HashSet<T, S>,
    baseline_capacity: usize,
    family_target_bytes: usize,
    family_retained_bytes: usize,
) -> bool {
    let old = value.capacity();
    let target = proportional_capacity(
        old,
        value.len(),
        baseline_capacity,
        family_target_bytes,
        family_retained_bytes,
    );
    value.shrink_to(target);
    value.capacity() < old
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapacityPolicy {
    pub(crate) minimum_family_bytes: usize,
    pub(crate) cold_numerator: usize,
    pub(crate) cold_denominator: usize,
    pub(crate) cold_attempts: u32,
    pub(crate) minimum_recoverable_bytes: usize,
    pub(crate) cooldown_attempts: u32,
}

impl Default for CapacityPolicy {
    fn default() -> Self {
        Self {
            minimum_family_bytes: 1024 * 1024,
            cold_numerator: 1,
            cold_denominator: 4,
            cold_attempts: 240,
            minimum_recoverable_bytes: 512 * 1024,
            cooldown_attempts: 240,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct CapacityFamilyState {
    usage: CapacityUsage,
    cold_attempts: u32,
    cold_peak_active_bytes: usize,
    recent_peak_active_bytes: usize,
    previous_recent_peak_active_bytes: usize,
    recent_attempts: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PendingCapacityTrim {
    pub(crate) family: usize,
    pub(crate) target_bytes: usize,
    pub(crate) retained_bytes: usize,
    pub(crate) recoverable_bytes: usize,
    pub(crate) explicit: bool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapacityGovernorStats {
    pub(crate) attempts: u64,
    pub(crate) usage: CapacityUsage,
    pub(crate) cooldown_remaining: u32,
    pub(crate) pending: Option<PendingCapacityTrim>,
    pub(crate) explicit_request_queued: bool,
    pub(crate) trim_count: u64,
    pub(crate) last_trimmed_bytes: usize,
}

/// Allocation-free governor for a fixed set of retained-capacity families.
pub(crate) struct CapacityGovernor<const N: usize> {
    policy: CapacityPolicy,
    families: [CapacityFamilyState; N],
    attempts: u64,
    cooldown_remaining: u32,
    pending: Option<PendingCapacityTrim>,
    explicit_request: bool,
    aggregate: CapacityUsage,
    trim_count: u64,
    last_trimmed_bytes: usize,
}

impl<const N: usize> Default for CapacityGovernor<N> {
    fn default() -> Self {
        Self::new(CapacityPolicy::default())
    }
}

impl<const N: usize> CapacityGovernor<N> {
    #[must_use]
    pub(crate) const fn new(policy: CapacityPolicy) -> Self {
        Self {
            policy,
            families: [CapacityFamilyState {
                usage: CapacityUsage::new(0, 0, 0),
                cold_attempts: 0,
                cold_peak_active_bytes: 0,
                recent_peak_active_bytes: 0,
                previous_recent_peak_active_bytes: 0,
                recent_attempts: 0,
            }; N],
            attempts: 0,
            cooldown_remaining: 0,
            pending: None,
            explicit_request: false,
            aggregate: CapacityUsage::new(0, 0, 0),
            trim_count: 0,
            last_trimmed_bytes: 0,
        }
    }

    /// Queue a maintenance trim request. It bypasses only the cold-ratio and
    /// cold-duration gates; family-size, recoverable-byte, pending, and
    /// cooldown gates remain intact.
    pub(crate) fn request_trim(&mut self) {
        if self.pending.is_none() {
            self.explicit_request = true;
        }
    }

    /// Return whether a clear boundary needs a point-in-time capacity scan to
    /// resolve an explicit maintenance request.
    ///
    /// Ordinary attempts already feed retained-capacity usage through
    /// [`Self::observe`].  Rebuilding the full multi-family snapshot at every
    /// clear boundary is only useful for an explicit request that can be
    /// resolved immediately; pending work and the global cooldown deliberately
    /// defer that resolution.
    #[must_use]
    pub(crate) const fn needs_requested_trim_resolution(&self) -> bool {
        self.explicit_request && self.pending.is_none() && self.cooldown_remaining == 0
    }

    /// Resolve an explicitly queued request against the capacity visible at a
    /// safe maintenance boundary without counting a second graph attempt.
    pub(crate) fn resolve_requested_trim(&mut self, usages: [CapacityUsage; N]) {
        if !self.explicit_request || self.pending.is_some() || self.cooldown_remaining > 0 {
            return;
        }

        let mut best = None;
        for (family, (state, usage)) in self.families.iter().zip(usages).enumerate() {
            if usage.retained_bytes < self.policy.minimum_family_bytes {
                continue;
            }
            let peak = state
                .recent_peak_active_bytes
                .max(state.previous_recent_peak_active_bytes)
                .max(state.usage.active_bytes)
                .max(usage.active_bytes);
            let target = usage
                .baseline_bytes
                .max(peak.saturating_mul(2))
                .min(usage.retained_bytes);
            let recoverable = usage.recoverable_bytes(target);
            if recoverable < self.policy.minimum_recoverable_bytes {
                continue;
            }
            let candidate = PendingCapacityTrim {
                family,
                target_bytes: target,
                retained_bytes: usage.retained_bytes,
                recoverable_bytes: recoverable,
                explicit: true,
            };
            if best.is_none_or(|current: PendingCapacityTrim| {
                candidate.recoverable_bytes > current.recoverable_bytes
            }) {
                best = Some(candidate);
            }
        }
        self.explicit_request = false;
        if best.is_some() {
            self.pending = best;
        }
    }

    /// Observe one completed attempt and, when eligible, publish at most one
    /// trim target: the family with the greatest recoverable byte count.
    pub(crate) fn observe(&mut self, usages: [CapacityUsage; N]) {
        self.attempts = self.attempts.saturating_add(1);
        self.aggregate = CapacityUsage::default();
        if self.pending.is_none() && self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
        }

        for (state, usage) in self.families.iter_mut().zip(usages) {
            debug_assert!(usage.active_bytes <= usage.retained_bytes);
            let retained_capacity_grew = usage.retained_bytes > state.usage.retained_bytes;
            state.usage = usage;
            self.aggregate += usage;

            if state.recent_attempts >= self.policy.cold_attempts.max(1) {
                state.previous_recent_peak_active_bytes = state.recent_peak_active_bytes;
                state.recent_attempts = 0;
                state.recent_peak_active_bytes = 0;
            }
            state.recent_attempts = state.recent_attempts.saturating_add(1);
            state.recent_peak_active_bytes = state.recent_peak_active_bytes.max(usage.active_bytes);

            // A family that has not reached the maintenance threshold has no
            // reclaimable high-water mark and therefore must not accumulate a
            // cold streak. Likewise, any capacity growth starts a fresh
            // observation window: a newly expanded family must remain cold
            // for the full hysteresis duration before it can be reclaimed.
            let cold = usage.retained_bytes >= self.policy.minimum_family_bytes
                && ratio_is_at_most(
                    usage.active_bytes,
                    usage.retained_bytes,
                    self.policy.cold_numerator,
                    self.policy.cold_denominator,
                );
            if retained_capacity_grew {
                state.cold_attempts = 0;
                state.cold_peak_active_bytes = 0;
            }
            if cold {
                state.cold_attempts = state.cold_attempts.saturating_add(1);
                state.cold_peak_active_bytes = state.cold_peak_active_bytes.max(usage.active_bytes);
            } else {
                state.cold_attempts = 0;
                state.cold_peak_active_bytes = 0;
            }
        }

        if self.pending.is_some() || self.cooldown_remaining > 0 {
            return;
        }

        let explicit = self.explicit_request;
        let mut best = None;
        for (family, state) in self.families.iter().enumerate() {
            if state.usage.retained_bytes < self.policy.minimum_family_bytes
                || (!explicit && state.cold_attempts < self.policy.cold_attempts.max(1))
            {
                continue;
            }
            let peak = if explicit {
                state
                    .recent_peak_active_bytes
                    .max(state.previous_recent_peak_active_bytes)
            } else {
                state.cold_peak_active_bytes
            };
            let target = state
                .usage
                .baseline_bytes
                .max(peak.saturating_mul(2))
                .min(state.usage.retained_bytes);
            let recoverable = state.usage.recoverable_bytes(target);
            if recoverable < self.policy.minimum_recoverable_bytes {
                continue;
            }
            let candidate = PendingCapacityTrim {
                family,
                target_bytes: target,
                retained_bytes: state.usage.retained_bytes,
                recoverable_bytes: recoverable,
                explicit,
            };
            if best.is_none_or(|current: PendingCapacityTrim| {
                candidate.recoverable_bytes > current.recoverable_bytes
            }) {
                best = Some(candidate);
            }
        }
        if explicit {
            self.explicit_request = false;
        }
        if best.is_some() {
            self.pending = best;
        }
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) const fn pending_target(&self) -> Option<PendingCapacityTrim> {
        self.pending
    }

    /// Consume the pending target and start the global trim cooldown.
    pub(crate) fn take_pending_target(&mut self) -> Option<PendingCapacityTrim> {
        let pending = self.pending.take()?;
        self.cooldown_remaining = self.policy.cooldown_attempts;
        let state = &mut self.families[pending.family];
        state.cold_attempts = 0;
        state.cold_peak_active_bytes = 0;
        state.recent_attempts = 0;
        state.recent_peak_active_bytes = 0;
        state.previous_recent_peak_active_bytes = 0;
        Some(pending)
    }

    /// Record the allocator-observed result after one pending family has been
    /// trimmed at a safe graph-attempt boundary.
    pub(crate) fn complete_trim(&mut self, retained_before: usize, retained_after: usize) {
        self.trim_count = self.trim_count.saturating_add(1);
        self.last_trimmed_bytes = retained_before.saturating_sub(retained_after);
    }

    #[must_use]
    pub(crate) const fn stats(&self) -> CapacityGovernorStats {
        CapacityGovernorStats {
            attempts: self.attempts,
            usage: self.aggregate,
            cooldown_remaining: self.cooldown_remaining,
            pending: self.pending,
            explicit_request_queued: self.explicit_request,
            trim_count: self.trim_count,
            last_trimmed_bytes: self.last_trimmed_bytes,
        }
    }
}

fn ratio_is_at_most(value: usize, total: usize, numerator: usize, denominator: usize) -> bool {
    if denominator == 0 {
        return false;
    }
    (value as u128) * (denominator as u128) <= (total as u128) * (numerator as u128)
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn test_policy() -> CapacityPolicy {
        CapacityPolicy {
            minimum_family_bytes: 100,
            cold_numerator: 1,
            cold_denominator: 4,
            cold_attempts: 3,
            minimum_recoverable_bytes: 40,
            cooldown_attempts: 2,
        }
    }

    const fn usage(active: usize, retained: usize, baseline: usize) -> CapacityUsage {
        CapacityUsage::new(active, retained, baseline)
    }

    #[test]
    fn cold_hysteresis_keeps_twice_the_window_peak() {
        let mut governor = CapacityGovernor::<1>::new(test_policy());
        governor.observe([usage(20, 400, 50)]);
        governor.observe([usage(30, 400, 50)]);
        assert_eq!(governor.pending_target(), None);
        governor.observe([usage(10, 400, 50)]);

        let pending = governor.pending_target().unwrap();
        assert_eq!(pending.target_bytes, 60);
        assert_eq!(pending.recoverable_bytes, 340);
        assert!(!pending.explicit);
    }

    #[test]
    fn chooses_only_the_largest_recoverable_family() {
        let mut governor = CapacityGovernor::<3>::new(test_policy());
        for _ in 0..3 {
            governor.observe([usage(10, 300, 50), usage(10, 700, 50), usage(10, 500, 50)]);
        }
        let pending = governor.pending_target().unwrap();
        assert_eq!(pending.family, 1);
        assert_eq!(pending.recoverable_bytes, 650);
    }

    #[test]
    fn consuming_a_target_enforces_global_cooldown() {
        let mut governor = CapacityGovernor::<2>::new(test_policy());
        for _ in 0..3 {
            governor.observe([usage(10, 500, 50), usage(10, 400, 50)]);
        }
        assert_eq!(governor.take_pending_target().unwrap().family, 0);
        governor.observe([usage(10, 100, 50), usage(10, 400, 50)]);
        assert_eq!(governor.pending_target(), None);
        governor.observe([usage(10, 100, 50), usage(10, 400, 50)]);
        assert_eq!(
            governor.pending_target(),
            Some(PendingCapacityTrim {
                family: 1,
                target_bytes: 50,
                retained_bytes: 400,
                recoverable_bytes: 350,
                explicit: false,
            })
        );
    }

    #[test]
    fn explicit_request_waits_through_cooldown_without_being_lost() {
        let mut governor = CapacityGovernor::<1>::new(test_policy());
        for _ in 0..3 {
            governor.observe([usage(10, 500, 50)]);
        }
        assert!(governor.take_pending_target().is_some());

        governor.request_trim();
        governor.resolve_requested_trim([usage(10, 500, 50)]);
        assert_eq!(governor.pending_target(), None);
        assert!(governor.stats().explicit_request_queued);

        governor.observe([usage(10, 500, 50)]);
        assert_eq!(governor.pending_target(), None);
        governor.observe([usage(10, 500, 50)]);

        let pending = governor.pending_target().unwrap();
        assert!(pending.explicit);
        assert!(!governor.stats().explicit_request_queued);
    }

    #[test]
    fn explicit_request_bypasses_cold_gate_but_keeps_recent_headroom() {
        let mut governor = CapacityGovernor::<1>::new(test_policy());
        governor.observe([usage(180, 400, 20)]);
        governor.request_trim();
        governor.observe([usage(120, 400, 20)]);

        let pending = governor.pending_target().unwrap();
        assert!(pending.explicit);
        assert_eq!(pending.target_bytes, 360);
        assert_eq!(pending.recoverable_bytes, 40);
    }

    #[test]
    fn explicit_request_can_resolve_at_a_safe_boundary_without_a_fake_attempt() {
        let mut governor = CapacityGovernor::<1>::new(test_policy());
        governor.observe([usage(80, 400, 20)]);
        governor.request_trim();
        governor.resolve_requested_trim([usage(40, 400, 20)]);

        let pending = governor.pending_target().unwrap();
        assert!(pending.explicit);
        assert_eq!(pending.target_bytes, 160);
        assert_eq!(governor.stats().attempts, 1);
    }

    #[test]
    fn dormant_family_serves_a_full_cold_window_after_first_growth() {
        let mut governor = CapacityGovernor::<1>::new(test_policy());
        for _ in 0..6 {
            governor.observe([usage(0, 0, 0)]);
        }

        governor.observe([usage(10, 400, 50)]);
        governor.observe([usage(10, 400, 50)]);
        assert_eq!(governor.pending_target(), None);

        governor.observe([usage(10, 400, 50)]);
        assert!(governor.pending_target().is_some());
    }

    #[test]
    fn retained_growth_restarts_an_existing_cold_window() {
        let mut governor = CapacityGovernor::<1>::new(test_policy());
        governor.observe([usage(10, 400, 50)]);
        governor.observe([usage(10, 400, 50)]);

        governor.observe([usage(20, 800, 50)]);
        governor.observe([usage(20, 800, 50)]);
        assert_eq!(governor.pending_target(), None);

        governor.observe([usage(20, 800, 50)]);
        assert!(governor.pending_target().is_some());
    }

    #[test]
    fn explicit_headroom_has_no_tumbling_window_cliff() {
        let mut governor = CapacityGovernor::<1>::new(test_policy());
        for _ in 0..3 {
            governor.observe([usage(180, 400, 20)]);
        }

        governor.request_trim();
        governor.observe([usage(20, 400, 20)]);

        let pending = governor.pending_target().unwrap();
        assert!(pending.explicit);
        assert_eq!(pending.target_bytes, 360);
    }

    #[test]
    fn proportional_shrink_respects_length_and_baseline() {
        let mut values = Vec::with_capacity(100);
        values.extend(0..30);
        assert!(shrink_vec_by_family_ratio(&mut values, 40, 250, 1000));
        assert!(values.capacity() >= 40);
        assert!(values.capacity() < 100);
        assert_eq!(values.len(), 30);
    }
}
