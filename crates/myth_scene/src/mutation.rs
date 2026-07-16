use std::collections::hash_map::Entry;
use std::fmt;
use std::ops::{BitOr, BitOrAssign, Deref, DerefMut};
use std::sync::atomic::{AtomicU32, Ordering};

use myth_core::NodeHandle;
use rustc_hash::FxHashMap;

const MAX_PENDING_SCENE_MUTATIONS: usize = 262_144;
static NEXT_SCENE_STORAGE_LINEAGE: AtomicU32 = AtomicU32::new(1);

/// Render-relevant domains changed by a scene mutation.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SceneMutationFlags(u8);

impl SceneMutationFlags {
    pub const TOPOLOGY: Self = Self(1 << 0);
    pub const TRANSFORM: Self = Self(1 << 1);
    pub const RENDERABLE: Self = Self(1 << 2);
    pub const ADDED: Self = Self(1 << 3);
    pub const REMOVED: Self = Self(1 << 4);

    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[inline]
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

/// Typed causes that require rebuilding the persistent render world.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SceneFullRebuildReasons(u8);

impl SceneFullRebuildReasons {
    pub const INITIAL: Self = Self(1 << 0);
    pub const EXPLICIT: Self = Self(1 << 1);
    pub const OVERFLOW: Self = Self(1 << 2);
    pub const UNTRACKED_STORAGE: Self = Self(1 << 3);
    pub const CURSOR_OR_IDENTITY_DRIFT: Self = Self(1 << 4);
    pub const UNTRACKED_HIERARCHY: Self = Self(1 << 5);

    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl BitOr for SceneFullRebuildReasons {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for SceneFullRebuildReasons {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl fmt::Debug for SceneFullRebuildReasons {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for (name, flag) in [
            ("INITIAL", Self::INITIAL),
            ("EXPLICIT", Self::EXPLICIT),
            ("OVERFLOW", Self::OVERFLOW),
            ("UNTRACKED_STORAGE", Self::UNTRACKED_STORAGE),
            ("CURSOR_OR_IDENTITY_DRIFT", Self::CURSOR_OR_IDENTITY_DRIFT),
            ("UNTRACKED_HIERARCHY", Self::UNTRACKED_HIERARCHY),
        ] {
            if self.contains(flag) {
                list.entry(&name);
            }
        }
        list.finish()
    }
}

impl BitOr for SceneMutationFlags {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for SceneMutationFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl fmt::Debug for SceneMutationFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for (name, flag) in [
            ("TOPOLOGY", Self::TOPOLOGY),
            ("TRANSFORM", Self::TRANSFORM),
            ("RENDERABLE", Self::RENDERABLE),
            ("ADDED", Self::ADDED),
            ("REMOVED", Self::REMOVED),
        ] {
            if self.contains(flag) {
                list.entry(&name);
            }
        }
        list.finish()
    }
}

/// One coalesced render-relevant scene mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneMutation {
    pub sequence: u64,
    pub node: NodeHandle,
    pub flags: SceneMutationFlags,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SceneStorageStamp {
    lineage: u32,
    owner_scene_id: u32,
    generation: u64,
}

impl SceneStorageStamp {
    #[inline]
    fn is_between(self, lower: Self, upper: Self) -> bool {
        self.lineage == lower.lineage
            && self.lineage == upper.lineage
            && self.owner_scene_id == lower.owner_scene_id
            && self.owner_scene_id == upper.owner_scene_id
            && self.generation >= lower.generation
            && self.generation <= upper.generation
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SceneStorageGenerations {
    pub nodes: SceneStorageStamp,
    pub meshes: SceneStorageStamp,
    pub skeletons: SceneStorageStamp,
    pub skins: SceneStorageStamp,
    pub morph_weights: SceneStorageStamp,
    pub split_primitive_tags: SceneStorageStamp,
    #[cfg(feature = "3dgs")]
    pub gaussian_clouds: SceneStorageStamp,
}

/// An acknowledgement token for a stable mutation snapshot.
///
/// Cursors are bound to one scene. A stale cursor can be acknowledged safely:
/// mutations recorded after it remain pending.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneMutationCursor {
    scene_id: u32,
    sequence: u64,
    storage_generations: SceneStorageGenerations,
}

impl SceneMutationCursor {
    #[inline]
    #[must_use]
    pub const fn scene_id(self) -> u32 {
        self.scene_id
    }

    #[inline]
    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
}

/// Immutable view of all render-relevant mutations not yet acknowledged.
#[derive(Debug, Clone, Copy)]
pub struct SceneMutationBatch<'a> {
    pub scene_id: u32,
    pub topology_epoch: u64,
    pub requires_full_rebuild: bool,
    pub full_rebuild_reasons: SceneFullRebuildReasons,
    pub overflowed: bool,
    pub untracked_storage_mutation: bool,
    pub entries: &'a [SceneMutation],
    pub cursor: SceneMutationCursor,
    /// Global acknowledgement frontier at which `entries` begins. Kept
    /// private so consumers must use `covers_cursor` instead of inventing
    /// partial sequence/generation comparisons.
    covered_after: SceneMutationCursor,
}

impl SceneMutationBatch<'_> {
    /// Whether this batch contains every mutation after `previous` through
    /// `cursor`.
    ///
    /// Scene keeps one compatibility acknowledgement frontier. A second
    /// renderer may therefore arrive after another consumer has drained older
    /// entries. Returning false lets that renderer fail closed to a full
    /// rebuild instead of silently reusing stale derived state. Raw-storage
    /// generations participate in the same coverage proof.
    #[inline]
    #[must_use]
    pub fn covers_cursor(&self, previous: SceneMutationCursor) -> bool {
        previous.scene_id == self.scene_id
            && previous.sequence >= self.covered_after.sequence
            && previous.sequence <= self.cursor.sequence
            && storage_generations_cover(
                previous.storage_generations,
                self.covered_after.storage_generations,
                self.cursor.storage_generations,
            )
    }
}

#[inline]
fn storage_generations_cover(
    previous: SceneStorageGenerations,
    covered_after: SceneStorageGenerations,
    current: SceneStorageGenerations,
) -> bool {
    macro_rules! covered {
        ($field:ident) => {
            previous
                .$field
                .is_between(covered_after.$field, current.$field)
        };
    }
    covered!(nodes)
        && covered!(meshes)
        && covered!(skeletons)
        && covered!(skins)
        && covered!(morph_weights)
        && covered!(split_primitive_tags)
        && {
            #[cfg(feature = "3dgs")]
            {
                covered!(gaussian_clouds)
            }
            #[cfg(not(feature = "3dgs"))]
            {
                true
            }
        }
}

#[inline]
fn storage_generations_at_or_before(
    candidate: SceneStorageGenerations,
    current: SceneStorageGenerations,
) -> bool {
    macro_rules! at_or_before {
        ($field:ident) => {
            candidate.$field.lineage == current.$field.lineage
                && candidate.$field.owner_scene_id == current.$field.owner_scene_id
                && candidate.$field.generation <= current.$field.generation
        };
    }
    at_or_before!(nodes)
        && at_or_before!(meshes)
        && at_or_before!(skeletons)
        && at_or_before!(skins)
        && at_or_before!(morph_weights)
        && at_or_before!(split_primitive_tags)
        && {
            #[cfg(feature = "3dgs")]
            {
                at_or_before!(gaussian_clouds)
            }
            #[cfg(not(feature = "3dgs"))]
            {
                true
            }
        }
}

#[inline]
fn acknowledge_storage_stamp(acknowledged: &mut SceneStorageStamp, cursor: SceneStorageStamp) {
    if acknowledged.lineage == cursor.lineage
        && acknowledged.owner_scene_id == cursor.owner_scene_id
    {
        acknowledged.generation = acknowledged.generation.max(cursor.generation);
    } else {
        *acknowledged = cursor;
    }
}

/// Capacity ledger for the retained mutation journal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SceneMutationJournalStats {
    pub pending_entries: usize,
    pub peak_pending_entries: usize,
    pub entries_capacity: usize,
    pub index_capacity: usize,
    /// Conservative payload estimate; allocator bucket/control bytes are not included.
    pub estimated_retained_bytes: usize,
}

/// Public scene storage that preserves legacy field access while detecting
/// mutable access which bypasses the scene's tracked APIs.
#[doc(hidden)]
#[derive(Debug)]
pub struct SceneStorage<S> {
    inner: S,
    lineage: u32,
    owner_scene_id: u32,
    external_mutation_generation: u64,
}

impl<S> SceneStorage<S> {
    #[inline]
    pub(crate) fn new(inner: S, owner_scene_id: u32) -> Self {
        let lineage = NEXT_SCENE_STORAGE_LINEAGE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("SceneStorage lineage exhausted");
        Self {
            inner,
            lineage,
            owner_scene_id,
            external_mutation_generation: 0,
        }
    }

    #[inline]
    pub(crate) const fn inner(&self) -> &S {
        &self.inner
    }

    #[inline]
    pub(crate) fn inner_mut_for(&mut self, owner_scene_id: u32) -> &mut S {
        if self.owner_scene_id != owner_scene_id {
            // A whole public wrapper may be moved between Scenes without
            // invoking DerefMut. Rebinding before any tracked mutation makes
            // that ownership transition part of the monotonic content stamp,
            // so moving it back cannot ABA an older RenderWorld cursor.
            self.owner_scene_id = owner_scene_id;
            self.note_external_mutation();
        }
        &mut self.inner
    }

    #[inline]
    pub(crate) const fn external_mutation_generation(&self) -> SceneStorageStamp {
        SceneStorageStamp {
            lineage: self.lineage,
            owner_scene_id: self.owner_scene_id,
            generation: self.external_mutation_generation,
        }
    }

    #[inline]
    fn note_external_mutation(&mut self) {
        self.external_mutation_generation = self
            .external_mutation_generation
            .checked_add(1)
            .expect("SceneStorage external-mutation generation exhausted");
    }
}

impl<S> Deref for SceneStorage<S> {
    type Target = S;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<S> DerefMut for SceneStorage<S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.note_external_mutation();
        &mut self.inner
    }
}

// `SceneStorage` is intentionally generic over several SlotMap/secondary-map
// families which do not share a public `iter` trait. Deref preserves their
// native `iter` APIs; these forwarding impls only preserve `for` syntax.
#[allow(clippy::into_iter_without_iter)]
impl<'a, S> IntoIterator for &'a SceneStorage<S>
where
    &'a S: IntoIterator,
{
    type Item = <&'a S as IntoIterator>::Item;
    type IntoIter = <&'a S as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        (&self.inner).into_iter()
    }
}

#[allow(clippy::into_iter_without_iter)]
impl<'a, S> IntoIterator for &'a mut SceneStorage<S>
where
    &'a mut S: IntoIterator,
{
    type Item = <&'a mut S as IntoIterator>::Item;
    type IntoIter = <&'a mut S as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.note_external_mutation();
        (&mut self.inner).into_iter()
    }
}

#[derive(Debug)]
pub(crate) struct SceneMutationJournal {
    scene_id: u32,
    sequence: u64,
    acknowledged_sequence: u64,
    topology_epoch: u64,
    full_rebuild_sequence: u64,
    pending_rebuild_reasons: SceneFullRebuildReasons,
    suppress_deltas_until_ack: bool,
    acknowledged_storage_generations: SceneStorageGenerations,
    entries: Vec<SceneMutation>,
    entry_indices: FxHashMap<NodeHandle, usize>,
    peak_pending_entries: usize,
}

impl SceneMutationJournal {
    pub(crate) fn new(scene_id: u32) -> Self {
        Self {
            scene_id,
            sequence: 1,
            acknowledged_sequence: 0,
            topology_epoch: 1,
            full_rebuild_sequence: 1,
            pending_rebuild_reasons: SceneFullRebuildReasons::INITIAL,
            suppress_deltas_until_ack: true,
            acknowledged_storage_generations: SceneStorageGenerations::default(),
            entries: Vec::new(),
            entry_indices: FxHashMap::default(),
            peak_pending_entries: 0,
        }
    }

    #[inline]
    fn next_sequence(&mut self) -> u64 {
        self.sequence = self
            .sequence
            .checked_add(1)
            .expect("Scene mutation sequence exhausted");
        self.sequence
    }

    pub(crate) fn record(&mut self, node: NodeHandle, flags: SceneMutationFlags) {
        let sequence = self.next_sequence();
        if flags.contains(SceneMutationFlags::TOPOLOGY) {
            self.topology_epoch = self
                .topology_epoch
                .checked_add(1)
                .expect("Scene topology epoch exhausted");
        }

        if self.suppress_deltas_until_ack {
            self.full_rebuild_sequence = sequence;
            return;
        }

        if let Some(&index) = self.entry_indices.get(&node) {
            let entry = &mut self.entries[index];
            entry.sequence = sequence;
            entry.flags |= flags;
            return;
        }

        if self.entries.len() >= MAX_PENDING_SCENE_MUTATIONS {
            self.entries.clear();
            self.entry_indices.clear();
            self.force_full_rebuild_at_current_sequence(SceneFullRebuildReasons::OVERFLOW);
            return;
        }

        let index = self.entries.len();
        self.entries.push(SceneMutation {
            sequence,
            node,
            flags,
        });
        self.entry_indices.insert(node, index);
        self.peak_pending_entries = self.peak_pending_entries.max(self.entries.len());
    }

    pub(crate) fn force_full_rebuild(&mut self, reason: SceneFullRebuildReasons) {
        self.next_sequence();
        self.force_full_rebuild_at_current_sequence(reason);
    }

    fn force_full_rebuild_at_current_sequence(&mut self, reason: SceneFullRebuildReasons) {
        self.full_rebuild_sequence = self.sequence;
        self.pending_rebuild_reasons |= reason;
        self.suppress_deltas_until_ack = true;
        self.entries.clear();
        self.entry_indices.clear();
    }

    pub(crate) fn batch(
        &self,
        storage_generations: SceneStorageGenerations,
    ) -> SceneMutationBatch<'_> {
        let untracked_storage_mutation =
            storage_generations != self.acknowledged_storage_generations;
        let journal_requires_rebuild = self.full_rebuild_sequence > self.acknowledged_sequence;

        let mut full_rebuild_reasons = if journal_requires_rebuild {
            self.pending_rebuild_reasons
        } else {
            SceneFullRebuildReasons::default()
        };
        if untracked_storage_mutation {
            full_rebuild_reasons |= SceneFullRebuildReasons::UNTRACKED_STORAGE;
        }

        SceneMutationBatch {
            scene_id: self.scene_id,
            topology_epoch: self.topology_epoch,
            requires_full_rebuild: !full_rebuild_reasons.is_empty(),
            full_rebuild_reasons,
            overflowed: full_rebuild_reasons.contains(SceneFullRebuildReasons::OVERFLOW),
            untracked_storage_mutation,
            entries: &self.entries,
            cursor: SceneMutationCursor {
                scene_id: self.scene_id,
                sequence: self.sequence,
                storage_generations,
            },
            covered_after: SceneMutationCursor {
                scene_id: self.scene_id,
                sequence: self.acknowledged_sequence,
                storage_generations: self.acknowledged_storage_generations,
            },
        }
    }

    pub(crate) fn acknowledge(
        &mut self,
        cursor: SceneMutationCursor,
        current_storage_generations: SceneStorageGenerations,
    ) -> bool {
        if cursor.scene_id != self.scene_id
            || cursor.sequence > self.sequence
            || !storage_generations_at_or_before(
                cursor.storage_generations,
                current_storage_generations,
            )
        {
            self.force_full_rebuild(SceneFullRebuildReasons::CURSOR_OR_IDENTITY_DRIFT);
            return false;
        }

        if cursor.sequence > self.acknowledged_sequence {
            self.acknowledged_sequence = cursor.sequence;
            self.entries
                .retain(|mutation| mutation.sequence > cursor.sequence);
            self.rebuild_entry_indices();
        }

        acknowledge_storage_stamp(
            &mut self.acknowledged_storage_generations.nodes,
            cursor.storage_generations.nodes,
        );
        acknowledge_storage_stamp(
            &mut self.acknowledged_storage_generations.meshes,
            cursor.storage_generations.meshes,
        );
        acknowledge_storage_stamp(
            &mut self.acknowledged_storage_generations.skeletons,
            cursor.storage_generations.skeletons,
        );
        acknowledge_storage_stamp(
            &mut self.acknowledged_storage_generations.skins,
            cursor.storage_generations.skins,
        );
        acknowledge_storage_stamp(
            &mut self.acknowledged_storage_generations.morph_weights,
            cursor.storage_generations.morph_weights,
        );
        acknowledge_storage_stamp(
            &mut self.acknowledged_storage_generations.split_primitive_tags,
            cursor.storage_generations.split_primitive_tags,
        );
        #[cfg(feature = "3dgs")]
        {
            acknowledge_storage_stamp(
                &mut self.acknowledged_storage_generations.gaussian_clouds,
                cursor.storage_generations.gaussian_clouds,
            );
        }

        if cursor.sequence >= self.full_rebuild_sequence {
            self.suppress_deltas_until_ack = false;
            self.pending_rebuild_reasons = SceneFullRebuildReasons::default();
        }

        true
    }

    fn rebuild_entry_indices(&mut self) {
        self.entry_indices.clear();
        for (index, mutation) in self.entries.iter().enumerate() {
            match self.entry_indices.entry(mutation.node) {
                Entry::Vacant(entry) => {
                    entry.insert(index);
                }
                Entry::Occupied(_) => {
                    debug_assert!(
                        false,
                        "scene mutation journal contains duplicate node entries"
                    );
                }
            }
        }
    }

    pub(crate) fn stats(&self) -> SceneMutationJournalStats {
        let entries_bytes = self
            .entries
            .capacity()
            .saturating_mul(std::mem::size_of::<SceneMutation>());
        let index_bytes = self
            .entry_indices
            .capacity()
            .saturating_mul(std::mem::size_of::<(NodeHandle, usize)>());
        SceneMutationJournalStats {
            pending_entries: self.entries.len(),
            peak_pending_entries: self.peak_pending_entries,
            entries_capacity: self.entries.capacity(),
            index_capacity: self.entry_indices.capacity(),
            estimated_retained_bytes: entries_bytes.saturating_add(index_bytes),
        }
    }

    pub(crate) fn trim_idle_capacity(&mut self) -> bool {
        if !self.entries.is_empty()
            || self.suppress_deltas_until_ack
            || !self.pending_rebuild_reasons.is_empty()
        {
            return false;
        }

        let before = (self.entries.capacity(), self.entry_indices.capacity());
        self.entries.shrink_to(64);
        self.entry_indices.shrink_to(64);
        self.peak_pending_entries = self.entries.len();
        before != (self.entries.capacity(), self.entry_indices.capacity())
    }
}

#[cfg(test)]
mod tests {
    use slotmap::KeyData;

    use super::*;

    fn node(index: u32) -> NodeHandle {
        NodeHandle::from(KeyData::from_ffi((1_u64 << 32) | u64::from(index)))
    }

    #[test]
    #[should_panic(expected = "Scene mutation sequence exhausted")]
    fn sequence_exhaustion_cannot_alias_an_old_cursor() {
        let mut journal = SceneMutationJournal::new(7);
        let storage = SceneStorageGenerations::default();
        let initial = journal.batch(storage).cursor;
        assert!(journal.acknowledge(initial, storage));

        journal.sequence = u64::MAX;
        journal.record(node(1), SceneMutationFlags::TRANSFORM);
    }

    #[test]
    fn drained_ack_frontier_exposes_lost_coverage_to_independent_consumers() {
        let mut journal = SceneMutationJournal::new(8);
        let storage = SceneStorageGenerations::default();
        let initial = journal.batch(storage).cursor;
        assert!(journal.acknowledge(initial, storage));

        journal.record(node(1), SceneMutationFlags::TRANSFORM);
        let pending = journal.batch(SceneStorageGenerations::default());
        assert!(pending.covers_cursor(initial));
        let current = pending.cursor;
        assert!(journal.acknowledge(current, storage));

        let drained = journal.batch(SceneStorageGenerations::default());
        assert!(!drained.covers_cursor(initial));
        assert!(drained.covers_cursor(current));

        let raw_generation = SceneStorageGenerations {
            nodes: SceneStorageStamp {
                lineage: 1,
                owner_scene_id: 1,
                generation: 1,
            },
            ..SceneStorageGenerations::default()
        };
        let raw = journal.batch(raw_generation);
        let raw_cursor = raw.cursor;
        assert!(raw.requires_full_rebuild);
        assert!(journal.acknowledge(raw_cursor, raw_generation));
        let raw_drained = journal.batch(raw_generation);
        assert!(!raw_drained.covers_cursor(current));
        assert!(raw_drained.covers_cursor(raw_cursor));
    }

    #[test]
    fn idle_trim_releases_a_journal_high_water_mark() {
        let mut journal = SceneMutationJournal::new(9);
        let storage = SceneStorageGenerations::default();
        let initial = journal.batch(storage).cursor;
        assert!(journal.acknowledge(initial, storage));

        for index in 1..=256 {
            journal.record(node(index), SceneMutationFlags::RENDERABLE);
        }
        let cursor = journal.batch(SceneStorageGenerations::default()).cursor;
        assert!(journal.acknowledge(cursor, storage));
        let before = journal.stats().estimated_retained_bytes;

        assert!(journal.trim_idle_capacity());
        assert!(journal.stats().estimated_retained_bytes < before);
    }
}
