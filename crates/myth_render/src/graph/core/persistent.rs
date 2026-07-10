//! Logical cross-frame texture history.
//!
//! History resources are planned during graph construction using stable IDs
//! and compile-visible descriptors. Physical textures are materialized only
//! after DPE, and a ping-pong bank becomes readable only after the frame's GPU
//! work has been submitted successfully.

use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;

use crate::core::gpu::Tracked;

use super::types::{ResourceRecord, TextureDesc};

static NEXT_PERSISTENT_TEXTURE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HISTORY_SET_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PersistentTextureId(u64);

impl PersistentTextureId {
    fn new() -> Self {
        Self(NEXT_PERSISTENT_TEXTURE_ID.fetch_add(1, Ordering::Relaxed))
    }
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

struct PersistentTextureRecord {
    desc: TextureDesc,
    view: Tracked<wgpu::TextureView>,
}

/// Renderer-owned registry for cross-frame persistent history textures.
#[derive(Default)]
pub struct HistoryRegistry {
    sets: FxHashMap<HistorySetId, HistorySetRecord>,
    textures: FxHashMap<PersistentTextureId, PersistentTextureRecord>,
}

// TODO: add renderer-frame TTL eviction for abandoned scene/camera domains.
// Descriptor changes already reuse IDs, so resize does not grow this storage;
// domain churn still needs an explicit ownership/retirement policy.

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
        }
        plan
    }

    pub fn invalidate(&mut self, set_id: HistorySetId) {
        if let Some(record) = self.sets.get_mut(&set_id) {
            record.invalidate(None);
        }
    }

    /// Materialize only persistent textures touched by live graph passes.
    pub fn materialize_live_textures(
        &mut self,
        device: &wgpu::Device,
        resources: &[ResourceRecord],
    ) {
        for resource in resources {
            if resource.first_use == usize::MAX {
                continue;
            }
            let Some(id) = resource.persistent_texture_id() else {
                continue;
            };
            let desc = resource.texture_desc();
            let needs_recreate = self
                .textures
                .get(&id)
                .is_none_or(|texture| texture.desc != desc);
            if !needs_recreate {
                continue;
            }

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(resource.name),
                size: desc.size,
                mip_level_count: desc.mip_level_count,
                sample_count: desc.sample_count,
                dimension: desc.dimension,
                format: desc.format,
                usage: desc.usage,
                view_formats: &[],
            });
            let view = Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.textures
                .insert(id, PersistentTextureRecord { desc, view });
        }
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

    #[cfg(test)]
    fn physical_texture_count(&self) -> usize {
        self.textures.len()
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
}
