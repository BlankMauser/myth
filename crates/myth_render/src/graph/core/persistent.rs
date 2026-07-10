//! Logical cross-frame persistent textures.
//!
//! History and ordinary cache resources are planned during graph construction
//! using stable IDs and compile-visible texture specifications. Physical
//! textures are materialized only after DPE, and new contents become readable
//! only after the frame's GPU work has been submitted successfully.

use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::core::gpu::Tracked;

use super::types::TextureDesc;

static NEXT_PERSISTENT_TEXTURE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HISTORY_SET_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

/// Number of renderer frames an inactive scene/camera temporal domain remains
/// reusable. The frame-end sweep enforces the memory bound, while each active
/// domain also checks this age on re-entry so missed sweeps cannot revive stale
/// temporal state.
pub(crate) const TEMPORAL_DOMAIN_TTL_FRAMES: u64 = 120;

/// Stable identity for state that belongs to one scene/camera timeline.
///
/// Feature histories and the renderer's previous-view state share this key,
/// but retain independent lifetimes: disabling one feature must not reset the
/// view state used by another temporal consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemporalDomainKey {
    scene_id: u32,
    camera_id: u128,
}

impl TemporalDomainKey {
    #[must_use]
    pub const fn new(scene_id: u32, camera_id: u128) -> Self {
        Self {
            scene_id,
            camera_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PersistentTextureId(u64);

impl PersistentTextureId {
    fn new() -> Self {
        Self(NEXT_PERSISTENT_TEXTURE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Complete physical specification for a logical persistent texture.
///
/// `canonical_view_dimension` describes the view returned by the ordinary RDG
/// resolvers. The backing texture remains available through that view, so live
/// prepare code may create additional per-mip or storage views without making
/// those view shapes part of the graph's physical-resource model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PersistentTextureSpec {
    pub desc: TextureDesc,
    pub canonical_view_dimension: Option<wgpu::TextureViewDimension>,
}

impl PersistentTextureSpec {
    #[must_use]
    pub const fn new(
        desc: TextureDesc,
        canonical_view_dimension: Option<wgpu::TextureViewDimension>,
    ) -> Self {
        Self {
            desc,
            canonical_view_dimension,
        }
    }

    #[must_use]
    pub const fn default_view(desc: TextureDesc) -> Self {
        Self::new(desc, None)
    }
}

/// A graph-importable logical persistent texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistentTextureRef {
    pub id: PersistentTextureId,
    pub spec: PersistentTextureSpec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistorySetId(u64);

impl HistorySetId {
    fn new() -> Self {
        Self(NEXT_HISTORY_SET_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryTextureRef {
    pub id: PersistentTextureId,
    pub desc: TextureDesc,
}

impl From<HistoryTextureRef> for PersistentTextureRef {
    fn from(texture: HistoryTextureRef) -> Self {
        Self {
            id: texture.id,
            spec: PersistentTextureSpec::default_view(texture.desc),
        }
    }
}

/// Stable identity for an ordinary persistent cache entry.
///
/// IDs are globally monotonic and never reused. A feature should allocate one
/// ID for each independently invalidated cache entry and retain it across
/// frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheId(u64);

impl CacheId {
    fn new() -> Self {
        Self(NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Caller-defined identity of the contents expected in a cache entry.
///
/// The registry deliberately does not hash engine objects itself. Callers can
/// combine asset IDs, revisions, quality settings, shader variants, and every
/// other dependency that can change generated contents into this stable value,
/// without putting feature-specific state in graph storage. Omitting a source
/// or version dependency can incorrectly reuse stale GPU contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey(u128);

impl CacheKey {
    #[must_use]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }
}

impl From<u64> for CacheKey {
    fn from(value: u64) -> Self {
        Self(u128::from(value))
    }
}

impl From<u128> for CacheKey {
    fn from(value: u128) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheCommitToken {
    pub(crate) cache_id: CacheId,
    pub(crate) key: CacheKey,
    pub(crate) epoch: u64,
    pub(crate) generation: u64,
    pub(crate) attempt_serial: u64,
}

/// One frame's logical plan for an ordinary persistent cache entry.
#[derive(Debug, Clone, Copy)]
pub struct CachePlan<const N: usize> {
    pub textures: [PersistentTextureRef; N],
    /// Whether the textures already contain successfully submitted contents
    /// for `key` and the requested specifications.
    pub valid: bool,
    pub epoch: u64,
    /// Publication capability for an invalid entry. A valid entry is strictly
    /// read-only until [`HistoryRegistry::invalidate_cache`] is called, which
    /// prevents a partially culled refresh from corrupting an otherwise valid
    /// multi-texture cache group.
    pub commit: Option<CacheCommitToken>,
}

/// Strongly typed acknowledgement of a cache publication performed after GPU
/// command submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachePublicationReceipt {
    pub cache_id: CacheId,
    pub key: CacheKey,
    pub epoch: u64,
    pub generation: u64,
    pub attempt_serial: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryCommitToken {
    pub(crate) set_id: HistorySetId,
    pub(crate) epoch: u64,
    pub(crate) write_bank: u8,
    pub(crate) temporal_serial: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct HistoryPlan<const N: usize> {
    pub previous: [HistoryTextureRef; N],
    pub next: [HistoryTextureRef; N],
    pub valid: bool,
    pub commit: HistoryCommitToken,
}

struct HistorySetRecord {
    epoch: u64,
    valid: bool,
    committed_bank: u8,
    descriptors: Vec<TextureDesc>,
    members: Vec<[PersistentTextureId; 2]>,
    last_planned_serial: Option<u64>,
    last_plan_valid: bool,
    last_reset_serial: Option<u64>,
    last_commit_serial: Option<u64>,
}

impl HistorySetRecord {
    fn new(descriptors: &[TextureDesc]) -> Self {
        Self {
            epoch: 0,
            valid: false,
            committed_bank: 0,
            descriptors: descriptors.to_vec(),
            members: descriptors
                .iter()
                .map(|_| [PersistentTextureId::new(), PersistentTextureId::new()])
                .collect(),
            last_planned_serial: None,
            last_plan_valid: false,
            last_reset_serial: None,
            last_commit_serial: None,
        }
    }

    fn reset_for_descriptors(&mut self, descriptors: &[TextureDesc]) -> Vec<PersistentTextureId> {
        self.epoch = self.epoch.wrapping_add(1);
        self.valid = false;
        self.committed_bank = 0;
        self.descriptors.clear();
        self.descriptors.extend_from_slice(descriptors);
        let retired = if self.members.len() == descriptors.len() {
            Vec::new()
        } else {
            let retired = self.members.iter().flatten().copied().collect();
            self.members = descriptors
                .iter()
                .map(|_| [PersistentTextureId::new(), PersistentTextureId::new()])
                .collect();
            retired
        };
        self.last_planned_serial = None;
        self.last_plan_valid = false;
        self.last_commit_serial = None;
        retired
    }

    fn invalidate(&mut self, temporal_serial: Option<u64>) {
        self.epoch = self.epoch.wrapping_add(1);
        self.valid = false;
        self.last_plan_valid = false;
        self.last_reset_serial = temporal_serial;
        self.last_commit_serial = None;
    }
}

struct CacheRecord {
    epoch: u64,
    valid: bool,
    key: CacheKey,
    specs: Vec<PersistentTextureSpec>,
    members: Vec<PersistentTextureId>,
    generation: u64,
    last_planned_serial: Option<u64>,
    last_commit_generation: Option<u64>,
}

impl CacheRecord {
    fn new(key: CacheKey, specs: &[PersistentTextureSpec]) -> Self {
        Self {
            epoch: 0,
            valid: false,
            key,
            specs: specs.to_vec(),
            members: specs.iter().map(|_| PersistentTextureId::new()).collect(),
            generation: 0,
            last_planned_serial: None,
            last_commit_generation: None,
        }
    }

    fn reset(
        &mut self,
        key: CacheKey,
        specs: &[PersistentTextureSpec],
    ) -> Vec<PersistentTextureId> {
        self.epoch = self.epoch.wrapping_add(1);
        self.valid = false;
        self.key = key;
        self.specs.clear();
        self.specs.extend_from_slice(specs);
        let retired = if self.members.len() == specs.len() {
            Vec::new()
        } else {
            let retired = std::mem::take(&mut self.members);
            self.members = specs.iter().map(|_| PersistentTextureId::new()).collect();
            retired
        };
        self.last_planned_serial = None;
        self.last_commit_generation = None;
        retired
    }
}

struct PersistentTextureRecord {
    spec: PersistentTextureSpec,
    view: Tracked<wgpu::TextureView>,
}

/// Renderer-owned registry for cross-frame persistent textures.
#[derive(Default)]
pub struct HistoryRegistry {
    sets: FxHashMap<HistorySetId, HistorySetRecord>,
    active_cache_ids: FxHashSet<CacheId>,
    caches: FxHashMap<CacheId, CacheRecord>,
    logical_specs: FxHashMap<PersistentTextureId, PersistentTextureSpec>,
    textures: FxHashMap<PersistentTextureId, PersistentTextureRecord>,
}

impl HistoryRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a logical history-set identity. This performs no WGPU work.
    pub fn create_set(&mut self) -> HistorySetId {
        let id = HistorySetId::new();
        debug_assert!(!self.sets.contains_key(&id));
        id
    }

    /// Allocate a stable ordinary-cache identity. This performs no WGPU work.
    pub fn create_cache(&mut self) -> CacheId {
        let id = CacheId::new();
        let inserted = self.active_cache_ids.insert(id);
        debug_assert!(inserted && !self.caches.contains_key(&id));
        id
    }

    /// Atomically retire a logical set and every materialized texture bank it
    /// owns. Set IDs are globally monotonic and are never reused, so stale
    /// commit tokens cannot target a later domain after retirement.
    pub(crate) fn remove_set(&mut self, set_id: HistorySetId) -> bool {
        let Some(record) = self.sets.remove(&set_id) else {
            return false;
        };

        for id in record.members.into_iter().flatten() {
            self.textures.remove(&id);
            self.logical_specs.remove(&id);
        }
        true
    }

    /// Atomically retire an ordinary cache entry and all of its physical
    /// textures. Cache IDs are never reused, so every outstanding token is
    /// rejected after this call.
    pub fn remove_cache(&mut self, cache_id: CacheId) -> bool {
        if !self.active_cache_ids.remove(&cache_id) {
            return false;
        }

        if let Some(record) = self.caches.remove(&cache_id) {
            for id in record.members {
                self.textures.remove(&id);
                self.logical_specs.remove(&id);
            }
        }
        true
    }

    /// Plan one frame of a grouped ping-pong history set.
    ///
    /// Descriptor changes and explicit resets invalidate the previous bank.
    /// A planned-but-unsubmitted attempt makes the next plan invalid. Frames
    /// spent rendering other history domains do not break this set's
    /// continuity. `temporal_serial` must therefore be local to this history
    /// domain rather than the renderer's global frame counter.
    pub fn plan_set<const N: usize>(
        &mut self,
        set_id: HistorySetId,
        descriptors: [TextureDesc; N],
        temporal_serial: u64,
        reset: bool,
    ) -> HistoryPlan<N> {
        let (plan, retired_ids) = {
            let record = self
                .sets
                .entry(set_id)
                .or_insert_with(|| HistorySetRecord::new(&descriptors));

            let descriptors_changed = record.descriptors.as_slice() != descriptors;
            let reset_applied = reset && record.last_reset_serial != Some(temporal_serial);
            let retired_ids = if descriptors_changed {
                record.last_reset_serial = Some(temporal_serial);
                record.reset_for_descriptors(&descriptors)
            } else if reset_applied {
                record.invalidate(Some(temporal_serial));
                Vec::new()
            } else {
                Vec::new()
            };

            debug_assert_eq!(record.members.len(), N);
            let read_bank = usize::from(record.committed_bank);
            let write_bank = 1 - read_bank;
            let repeated_plan = !descriptors_changed
                && !reset_applied
                && record.last_planned_serial == Some(temporal_serial);
            let valid = if repeated_plan {
                record.last_plan_valid
            } else {
                record.valid
                    && record
                        .last_commit_serial
                        .is_some_and(|last| last.checked_add(1) == Some(temporal_serial))
            };

            let previous = std::array::from_fn(|index| HistoryTextureRef {
                id: record.members[index][read_bank],
                desc: descriptors[index],
            });
            let next = std::array::from_fn(|index| HistoryTextureRef {
                id: record.members[index][write_bank],
                desc: descriptors[index],
            });
            record.last_planned_serial = Some(temporal_serial);
            record.last_plan_valid = valid;

            (
                HistoryPlan {
                    previous,
                    next,
                    valid,
                    commit: HistoryCommitToken {
                        set_id,
                        epoch: record.epoch,
                        write_bank: write_bank as u8,
                        temporal_serial,
                    },
                },
                retired_ids,
            )
        };

        for id in retired_ids {
            self.textures.remove(&id);
            self.logical_specs.remove(&id);
        }
        for texture in plan.previous.into_iter().chain(plan.next) {
            self.logical_specs.insert(
                texture.id,
                PersistentTextureSpec::default_view(texture.desc),
            );
        }
        plan
    }

    /// Plan one attempt to populate or consume an ordinary persistent cache.
    ///
    /// `attempt_serial` identifies a graph-build attempt (normally the renderer
    /// frame index) and must be monotonic for this cache ID. Repeating a plan
    /// with the same serial, key, and specs is idempotent and returns the same
    /// token. Advancing the serial invalidates older unsubmitted tokens without
    /// invalidating already-published contents. A key or specification change
    /// invalidates contents, advances the entry epoch, and lazily recreates
    /// physical textures only if a live pass touches them after DPE.
    pub fn plan_cache<const N: usize>(
        &mut self,
        cache_id: CacheId,
        key: CacheKey,
        specs: [PersistentTextureSpec; N],
        attempt_serial: u64,
    ) -> CachePlan<N> {
        assert!(N > 0, "persistent cache requires at least one texture");
        assert!(
            self.active_cache_ids.contains(&cache_id),
            "persistent cache ID is retired or was not created by this registry"
        );

        let (plan, retired_ids) = {
            let record = self
                .caches
                .entry(cache_id)
                .or_insert_with(|| CacheRecord::new(key, &specs));

            assert!(
                record
                    .last_planned_serial
                    .is_none_or(|last| attempt_serial >= last),
                "persistent cache attempt serial moved backwards"
            );
            let desired_changed = record.key != key || record.specs.as_slice() != specs;
            assert!(
                !(desired_changed && record.last_planned_serial == Some(attempt_serial)),
                "persistent cache planned with conflicting key/spec in one attempt"
            );
            let retired_ids = if desired_changed {
                record.reset(key, &specs)
            } else {
                Vec::new()
            };

            let repeated_plan =
                !desired_changed && record.last_planned_serial == Some(attempt_serial);
            if !repeated_plan {
                record.generation = record.generation.wrapping_add(1);
                record.last_planned_serial = Some(attempt_serial);
            }

            debug_assert_eq!(record.members.len(), N);
            let textures = std::array::from_fn(|index| PersistentTextureRef {
                id: record.members[index],
                spec: specs[index],
            });
            (
                CachePlan {
                    textures,
                    valid: record.valid,
                    epoch: record.epoch,
                    commit: (!record.valid).then_some(CacheCommitToken {
                        cache_id,
                        key,
                        epoch: record.epoch,
                        generation: record.generation,
                        attempt_serial,
                    }),
                },
                retired_ids,
            )
        };

        for id in retired_ids {
            self.textures.remove(&id);
            self.logical_specs.remove(&id);
        }
        for texture in plan.textures {
            self.logical_specs.insert(texture.id, texture.spec);
        }
        plan
    }

    /// Explicitly invalidate a cache entry before requesting a refresh.
    ///
    /// Valid entries never expose a commit token. Requiring this transition
    /// before rewriting ensures a partially live writer group cannot damage
    /// contents that remain marked valid.
    pub fn invalidate_cache(&mut self, cache_id: CacheId) -> bool {
        let Some(record) = self.caches.get_mut(&cache_id) else {
            return false;
        };
        record.epoch = record.epoch.wrapping_add(1);
        record.valid = false;
        record.last_commit_generation = None;
        true
    }

    pub fn invalidate(&mut self, set_id: HistorySetId) {
        if let Some(record) = self.sets.get_mut(&set_id) {
            record.invalidate(None);
        }
    }

    fn assert_texture_spec_is_current(&self, id: PersistentTextureId, spec: PersistentTextureSpec) {
        assert_eq!(
            self.logical_specs.get(&id),
            Some(&spec),
            "persistent texture {id:?} imported with an unplanned or stale specification"
        );
    }

    pub(crate) fn materialize_texture(
        &mut self,
        device: &wgpu::Device,
        label: &'static str,
        id: PersistentTextureId,
        spec: PersistentTextureSpec,
    ) {
        self.assert_texture_spec_is_current(id, spec);
        let needs_recreate = self
            .textures
            .get(&id)
            .is_none_or(|texture| texture.spec != spec);
        if !needs_recreate {
            return;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: spec.desc.size,
            mip_level_count: spec.desc.mip_level_count,
            sample_count: spec.desc.sample_count,
            dimension: spec.desc.dimension,
            format: spec.desc.format,
            usage: spec.desc.usage,
            view_formats: &[],
        });
        let view = Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: spec.canonical_view_dimension,
            ..Default::default()
        }));
        self.textures
            .insert(id, PersistentTextureRecord { spec, view });
    }

    #[must_use]
    pub fn get_texture_view(&self, id: PersistentTextureId) -> &Tracked<wgpu::TextureView> {
        &self
            .textures
            .get(&id)
            .unwrap_or_else(|| panic!("persistent texture {id:?} was not materialized"))
            .view
    }

    #[must_use]
    pub(crate) fn is_texture_materialized(&self, id: PersistentTextureId) -> bool {
        self.textures.contains_key(&id)
    }

    /// Publish a grouped history bank after successful queue submission.
    pub(crate) fn commit(
        &mut self,
        token: HistoryCommitToken,
        written_textures: &[PersistentTextureId],
    ) -> bool {
        let Some(record) = self.sets.get_mut(&token.set_id) else {
            return false;
        };
        let write_bank = usize::from(token.write_bank);
        if record.epoch != token.epoch
            || record.last_planned_serial != Some(token.temporal_serial)
            || record
                .last_commit_serial
                .is_some_and(|last| last >= token.temporal_serial)
            || token.write_bank > 1
            || token.write_bank == record.committed_bank
            || written_textures.len() != record.members.len()
            || record
                .members
                .iter()
                .any(|member| !written_textures.contains(&member[write_bank]))
        {
            return false;
        }

        record.committed_bank = token.write_bank;
        record.valid = true;
        record.last_commit_serial = Some(token.temporal_serial);
        true
    }

    /// Publish an ordinary cache entry after successful queue submission.
    ///
    /// The token must belong to the latest plan, must not have been consumed,
    /// and the complete planned texture group must have been written. Failed
    /// validation leaves the cache invalid and returns no receipt.
    pub(crate) fn commit_cache(
        &mut self,
        token: CacheCommitToken,
        written_textures: &[PersistentTextureId],
    ) -> Option<CachePublicationReceipt> {
        let record = self.caches.get_mut(&token.cache_id)?;
        if record.valid
            || record.key != token.key
            || record.epoch != token.epoch
            || record.generation != token.generation
            || record.last_planned_serial != Some(token.attempt_serial)
            || record.last_commit_generation == Some(token.generation)
            || written_textures.len() != record.members.len()
            || record
                .members
                .iter()
                .any(|member| !written_textures.contains(member))
        {
            return None;
        }

        record.valid = true;
        record.last_commit_generation = Some(token.generation);
        Some(CachePublicationReceipt {
            cache_id: token.cache_id,
            key: token.key,
            epoch: token.epoch,
            generation: token.generation,
            attempt_serial: token.attempt_serial,
        })
    }

    #[cfg(test)]
    fn physical_texture_count(&self) -> usize {
        self.textures.len()
    }

    #[cfg(test)]
    fn history_set_count(&self) -> usize {
        self.sets.len()
    }

    #[cfg(test)]
    fn cache_count(&self) -> usize {
        self.active_cache_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(width: u32) -> TextureDesc {
        TextureDesc::new_2d(
            width,
            64,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }

    fn commit<const N: usize>(registry: &mut HistoryRegistry, plan: HistoryPlan<N>) -> bool {
        registry.commit(plan.commit, &plan.next.map(|texture| texture.id))
    }

    fn spec(width: u32) -> PersistentTextureSpec {
        PersistentTextureSpec::default_view(desc(width))
    }

    fn commit_cache<const N: usize>(
        registry: &mut HistoryRegistry,
        plan: CachePlan<N>,
    ) -> Option<CachePublicationReceipt> {
        registry.commit_cache(plan.commit?, &plan.textures.map(|texture| texture.id))
    }

    #[test]
    fn history_plan_ping_pongs_only_after_commit() {
        let mut registry = HistoryRegistry::new();
        let set = registry.create_set();

        let first = registry.plan_set(set, [desc(64), desc(64)], 1, false);
        assert!(!first.valid);
        let repeated = registry.plan_set(set, [desc(64), desc(64)], 1, false);
        assert_eq!(first.previous, repeated.previous);
        assert_eq!(first.next, repeated.next);

        assert!(commit(&mut registry, first));
        let second = registry.plan_set(set, [desc(64), desc(64)], 2, false);
        assert!(second.valid);
        assert_eq!(second.previous, first.next);
        assert_eq!(second.next, first.previous);
    }

    #[test]
    fn skipped_domain_attempt_invalidates_history() {
        let mut registry = HistoryRegistry::new();
        let set = registry.create_set();
        let first = registry.plan_set(set, [desc(64)], 1, false);
        assert!(commit(&mut registry, first));

        let skipped_attempt = registry.plan_set(set, [desc(64)], 3, false);
        assert!(!skipped_attempt.valid);
    }

    #[test]
    fn descriptor_change_invalidates_without_changing_logical_ids() {
        let mut registry = HistoryRegistry::new();
        let set = registry.create_set();
        let first = registry.plan_set(set, [desc(64)], 1, false);
        assert!(commit(&mut registry, first));

        let resized = registry.plan_set(set, [desc(128)], 2, false);
        assert!(!resized.valid);
        assert_eq!(resized.previous[0].id, first.previous[0].id);
        assert_eq!(resized.next[0].id, first.next[0].id);
        assert_eq!(registry.physical_texture_count(), 0);
    }

    #[test]
    fn stale_commit_is_rejected_after_reset() {
        let mut registry = HistoryRegistry::new();
        let set = registry.create_set();
        let stale = registry.plan_set(set, [desc(64)], 1, false);
        let reset = registry.plan_set(set, [desc(64)], 1, true);

        assert!(!commit(&mut registry, stale));
        assert!(commit(&mut registry, reset));
    }

    #[test]
    fn repeated_reset_in_one_frame_is_idempotent() {
        let mut registry = HistoryRegistry::new();
        let set = registry.create_set();

        let first = registry.plan_set(set, [desc(64)], 1, true);
        let repeated = registry.plan_set(set, [desc(64)], 1, true);

        assert_eq!(first.previous, repeated.previous);
        assert_eq!(first.next, repeated.next);
        assert_eq!(first.commit, repeated.commit);
        assert!(commit(&mut registry, first));
    }

    #[test]
    fn older_and_duplicate_commits_are_rejected() {
        let mut registry = HistoryRegistry::new();
        let set = registry.create_set();

        let first = registry.plan_set(set, [desc(64)], 1, false);
        let second = registry.plan_set(set, [desc(64)], 2, false);
        assert!(!commit(&mut registry, first));
        assert!(commit(&mut registry, second));
        assert!(!commit(&mut registry, second));
    }

    #[test]
    fn commit_requires_the_complete_planned_write_bank() {
        let mut registry = HistoryRegistry::new();
        let set = registry.create_set();
        let plan = registry.plan_set(set, [desc(64), desc(64)], 1, false);

        assert!(!registry.commit(plan.commit, &[plan.next[0].id]));
        assert!(!registry.commit(plan.commit, &[plan.previous[0].id, plan.previous[1].id]));
        assert!(commit(&mut registry, plan));
    }

    #[test]
    fn removing_a_set_rejects_pending_commits_and_never_reuses_its_identity() {
        let mut registry = HistoryRegistry::new();
        let retired_set = registry.create_set();
        let pending = registry.plan_set(retired_set, [desc(64), desc(64)], 1, false);

        assert_eq!(registry.history_set_count(), 1);
        assert!(registry.remove_set(retired_set));
        assert_eq!(registry.history_set_count(), 0);
        assert!(!commit(&mut registry, pending));
        assert!(!registry.remove_set(retired_set));

        let replacement_set = registry.create_set();
        assert_ne!(replacement_set, retired_set);
        let replacement = registry.plan_set(replacement_set, [desc(64), desc(64)], 1, false);
        assert!(!replacement.valid);
    }

    #[test]
    fn cache_plan_is_idempotent_within_one_attempt_and_valid_after_commit() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let key = CacheKey::new(7);

        let first = registry.plan_cache(cache, key, [spec(64), spec(32)], 10);
        let repeated = registry.plan_cache(cache, key, [spec(64), spec(32)], 10);
        assert!(!first.valid);
        assert_eq!(first.textures, repeated.textures);
        assert_eq!(first.commit, repeated.commit);

        let receipt = commit_cache(&mut registry, first).expect("cache should publish");
        assert_eq!(receipt.cache_id, cache);
        assert_eq!(receipt.key, key);
        assert!(commit_cache(&mut registry, repeated).is_none());

        let next = registry.plan_cache(cache, key, [spec(64), spec(32)], 11);
        assert!(next.valid);
        assert!(next.commit.is_none());
        assert_eq!(next.textures, first.textures);
        assert_ne!(next.commit, first.commit);
    }

    #[test]
    fn a_new_cache_attempt_rejects_an_unsubmitted_token() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let key = CacheKey::new(9);
        let stale = registry.plan_cache(cache, key, [spec(64)], 20);
        let current = registry.plan_cache(cache, key, [spec(64)], 21);

        assert!(commit_cache(&mut registry, stale).is_none());
        assert!(commit_cache(&mut registry, current).is_some());
    }

    #[test]
    #[should_panic(expected = "attempt serial moved backwards")]
    fn cache_attempt_serial_cannot_move_backwards() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let _ = registry.plan_cache(cache, CacheKey::new(9), [spec(64)], 20);
        let _ = registry.plan_cache(cache, CacheKey::new(9), [spec(64)], 19);
    }

    #[test]
    fn cache_key_and_spec_changes_invalidate_and_advance_epoch() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let first = registry.plan_cache(cache, CacheKey::new(1), [spec(64)], 1);
        assert!(commit_cache(&mut registry, first).is_some());

        let rekeyed = registry.plan_cache(cache, CacheKey::new(2), [spec(64)], 2);
        assert!(!rekeyed.valid);
        assert_eq!(rekeyed.epoch, first.epoch.wrapping_add(1));
        assert_eq!(rekeyed.textures[0].id, first.textures[0].id);
        assert!(commit_cache(&mut registry, rekeyed).is_some());

        let cube_desc = TextureDesc::new(
            128,
            128,
            6,
            1,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let cube_spec =
            PersistentTextureSpec::new(cube_desc, Some(wgpu::TextureViewDimension::Cube));
        let resized = registry.plan_cache(cache, CacheKey::new(2), [cube_spec], 3);
        assert!(!resized.valid);
        assert_eq!(resized.epoch, rekeyed.epoch.wrapping_add(1));
        assert_eq!(resized.textures[0].id, rekeyed.textures[0].id);
        assert_eq!(
            registry.logical_specs.get(&resized.textures[0].id),
            Some(&cube_spec)
        );
    }

    #[test]
    fn cache_member_count_change_retires_the_previous_logical_group() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let first = registry.plan_cache(cache, CacheKey::new(1), [spec(64)], 1);
        let retired_id = first.textures[0].id;

        let expanded = registry.plan_cache(cache, CacheKey::new(1), [spec(64), spec(32)], 2);

        assert!(!expanded.valid);
        assert_eq!(expanded.epoch, first.epoch.wrapping_add(1));
        assert!(
            expanded
                .textures
                .iter()
                .all(|texture| texture.id != retired_id)
        );
        assert!(!registry.logical_specs.contains_key(&retired_id));
        assert!(commit_cache(&mut registry, first).is_none());
        assert!(commit_cache(&mut registry, expanded).is_some());
    }

    #[test]
    #[should_panic(expected = "conflicting key/spec in one attempt")]
    fn conflicting_cache_plans_in_one_attempt_are_rejected() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let _ = registry.plan_cache(cache, CacheKey::new(1), [spec(64)], 1);
        let _ = registry.plan_cache(cache, CacheKey::new(2), [spec(64)], 1);
    }

    #[test]
    fn cache_commit_requires_the_complete_texture_group() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let plan = registry.plan_cache(cache, CacheKey::new(1), [spec(64), spec(32)], 1);

        assert!(
            registry
                .commit_cache(
                    plan.commit.expect("invalid plan needs a token"),
                    &[plan.textures[0].id]
                )
                .is_none()
        );
        assert!(commit_cache(&mut registry, plan).is_some());
    }

    #[test]
    fn removing_a_cache_rejects_tokens_and_does_not_reuse_identity() {
        let mut registry = HistoryRegistry::new();
        let retired = registry.create_cache();
        let pending = registry.plan_cache(retired, CacheKey::new(1), [spec(64)], 1);
        let retired_member = pending.textures[0].id;

        assert_eq!(registry.cache_count(), 1);
        assert!(registry.remove_cache(retired));
        assert_eq!(registry.cache_count(), 0);
        assert!(commit_cache(&mut registry, pending).is_none());
        assert!(!registry.remove_cache(retired));

        let replacement = registry.create_cache();
        assert_ne!(replacement, retired);
        let replacement_plan = registry.plan_cache(replacement, CacheKey::new(1), [spec(64)], 1);
        assert_ne!(replacement_plan.textures[0].id, retired_member);
    }

    #[test]
    #[should_panic(expected = "cache ID is retired")]
    fn removed_cache_identity_cannot_be_revived() {
        let mut registry = HistoryRegistry::new();
        let retired = registry.create_cache();
        let _ = registry.plan_cache(retired, CacheKey::new(1), [spec(64)], 1);
        assert!(registry.remove_cache(retired));
        let _ = registry.plan_cache(retired, CacheKey::new(1), [spec(64)], 2);
    }

    #[test]
    #[should_panic(expected = "unplanned or stale specification")]
    fn unplanned_persistent_texture_spec_is_rejected_before_materialization() {
        let registry = HistoryRegistry::new();
        registry.assert_texture_spec_is_current(PersistentTextureId::new(), spec(64));
    }

    #[test]
    #[should_panic(expected = "unplanned or stale specification")]
    fn stale_persistent_texture_spec_is_rejected_before_materialization() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let old = registry.plan_cache(cache, CacheKey::new(1), [spec(64)], 1);
        let _ = registry.plan_cache(cache, CacheKey::new(1), [spec(128)], 2);
        registry.assert_texture_spec_is_current(old.textures[0].id, old.textures[0].spec);
    }

    #[test]
    #[should_panic(expected = "unplanned or stale specification")]
    fn wrong_canonical_dimension_is_rejected_before_materialization() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let plan = registry.plan_cache(cache, CacheKey::new(1), [spec(64)], 1);
        registry.assert_texture_spec_is_current(
            plan.textures[0].id,
            PersistentTextureSpec::new(desc(64), Some(wgpu::TextureViewDimension::D2Array)),
        );
    }
}
