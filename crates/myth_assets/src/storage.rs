use parking_lot::{RwLock, RwLockReadGuard};
use rustc_hash::FxHashMap;
use slotmap::{Key, SlotMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use uuid::Uuid;

use myth_resources::ImageHandle;
use myth_resources::image::{DynamicImageError, Image};

/// Maximum number of asset slots inspected while holding one storage read lock.
///
/// Batch APIs split larger inputs into chunks of this width so asset streaming
/// writers are never blocked behind an unbounded read-side scan.
pub const ASSET_READ_BATCH_WIDTH: usize = 64;

/// Process-wide exact identity for a loaded asset allocation. Keeping this
/// global (rather than per `AssetStorage`) also distinguishes two AssetServer
/// instances whose slot handles and public versions happen to coincide.
static NEXT_LOADED_ASSET_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// One loaded asset owner and the exact storage stamp observed with it.
///
/// The `Arc` keeps the observed allocation alive after the storage read lock is
/// released. It does not promise that the allocation is still current; use
/// [`AssetStorage::revalidate_batched`] before publishing derived state.
#[derive(Debug)]
pub struct LoadedAssetSnapshot<T> {
    asset: Arc<T>,
    version: u32,
    identity: u64,
}

impl<T> LoadedAssetSnapshot<T> {
    /// Returns the retained asset owner.
    #[inline]
    #[must_use]
    pub fn asset(&self) -> &Arc<T> {
        &self.asset
    }

    /// Returns the public content version observed with the asset.
    #[inline]
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the process-wide identity of the observed loaded allocation.
    #[inline]
    #[must_use]
    pub const fn identity(&self) -> u64 {
        self.identity
    }

    /// Moves the retained asset owner out of the snapshot.
    #[inline]
    #[must_use]
    pub fn into_asset(self) -> Arc<T> {
        self.asset
    }

    /// Splits the snapshot without cloning its retained asset owner.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> (Arc<T>, u32, u64) {
        (self.asset, self.version, self.identity)
    }
}

/// Input-aligned observation of one asset handle.
///
/// A missing [`loaded`](Self::loaded) value represents every state that is not
/// currently renderable (`Loading`, `Failed`, or an invalid handle). Keeping
/// the handle in the observation lets batch revalidation preserve this
/// availability state without a second join table.
#[derive(Debug)]
pub struct AssetSnapshot<H: Key, T> {
    handle: H,
    loaded: Option<LoadedAssetSnapshot<T>>,
}

impl<H: Key, T> AssetSnapshot<H, T> {
    /// Returns the handle associated with this input position.
    #[inline]
    #[must_use]
    pub const fn handle(&self) -> H {
        self.handle
    }

    /// Returns the exact loaded observation, or `None` when unavailable.
    #[inline]
    #[must_use]
    pub const fn loaded(&self) -> Option<&LoadedAssetSnapshot<T>> {
        self.loaded.as_ref()
    }

    /// Returns whether this handle was loaded when the batch was captured.
    #[inline]
    #[must_use]
    pub const fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    /// Moves the exact loaded observation out without cloning its asset owner.
    #[inline]
    #[must_use]
    pub fn into_loaded(self) -> Option<LoadedAssetSnapshot<T>> {
        self.loaded
    }

    /// Splits the input-aligned observation without cloning its asset owner.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> (H, Option<LoadedAssetSnapshot<T>>) {
        (self.handle, self.loaded)
    }
}

/// Failure modes for zero-allocation dynamic image updates in [`AssetStorage`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicImageUpdateError {
    InvalidHandle,
    NotLoaded,
    Update(DynamicImageError),
}

impl std::fmt::Display for DynamicImageUpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHandle => f.write_str("image handle is not present in storage"),
            Self::NotLoaded => f.write_str("image handle is not loaded yet"),
            Self::Update(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for DynamicImageUpdateError {}

/// Versioned wrapper around a loaded asset in [`AssetStorage`].
///
/// The `version` counter is bumped every time the asset data is replaced
/// or mutated in place. The render backend compares this against its own
/// synced version to decide whether a GPU re-upload is needed.
///
/// Implements `Deref<Target = T>` for ergonomic access to the inner asset
/// through the `Arc`.
#[derive(Debug)]
pub struct AssetEntry<T> {
    pub asset: Arc<T>,
    /// Monotonically increasing counter. Starts at 1 on first insert.
    pub version: u32,
    /// Process-wide monotonic identity of this particular loaded `Arc`.
    ///
    /// Unlike `version`, this never intentionally resets to `1` when a
    /// reserved/failed slot is populated again. Render backends can therefore
    /// distinguish two replacement assets that occupy the same handle and
    /// happen to expose the same numeric version.
    pub identity: u64,
}

impl<T> std::ops::Deref for AssetEntry<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.asset
    }
}

/// Lifecycle state of an asset slot in [`AssetStorage`].
///
/// Binds resource data and its lifecycle tag together in a single enum,
/// making it impossible to observe a `Loaded` state without the
/// corresponding data, or vice-versa.
#[derive(Debug)]
pub enum AssetSlot<T> {
    /// Handle has been allocated; a background task is producing the data.
    Loading,
    /// Data is fully available and ready for use.
    Loaded(AssetEntry<T>),
    /// The loading attempt failed. The message is kept for diagnostics.
    Failed(String),
}

impl<T> AssetSlot<T> {
    /// Returns `true` if the slot contains loaded data.
    #[inline]
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::Loaded(_))
    }

    /// Returns `true` if the slot is still waiting for data.
    #[inline]
    #[must_use]
    pub fn is_loading(&self) -> bool {
        matches!(self, Self::Loading)
    }

    /// Extracts a reference to the loaded entry, if available.
    #[inline]
    #[must_use]
    pub fn as_loaded(&self) -> Option<&AssetEntry<T>> {
        match self {
            Self::Loaded(entry) => Some(entry),
            _ => None,
        }
    }

    /// Extracts a mutable reference to the loaded entry, if available.
    #[inline]
    pub fn as_loaded_mut(&mut self) -> Option<&mut AssetEntry<T>> {
        match self {
            Self::Loaded(entry) => Some(entry),
            _ => None,
        }
    }
}

/// Internal data structure, protected by a lock.
pub struct StorageInner<H: Key, T> {
    pub(crate) map: SlotMap<H, AssetSlot<T>>,
    pub lookup: FxHashMap<Uuid, H>,
}

impl<H: Key, T> StorageInner<H, T> {
    /// Gets a reference to the loaded [`AssetEntry`] at `handle`.
    ///
    /// Returns `None` if the handle is invalid or the asset is not yet loaded.
    #[inline]
    pub fn get_loaded(&self, handle: H) -> Option<&AssetEntry<T>> {
        self.map.get(handle).and_then(AssetSlot::as_loaded)
    }
}

impl<H: Key, T> Default for StorageInner<H, T> {
    fn default() -> Self {
        Self {
            map: SlotMap::default(),
            lookup: FxHashMap::default(),
        }
    }
}

/// Thread-safe, version-tracked asset container with lifecycle awareness.
///
/// Each slot in the storage is an [`AssetSlot`] — either `Loading`, `Loaded`,
/// or `Failed` — making it impossible to access data that hasn't arrived yet.
/// Loaded assets carry a monotonically increasing version counter that the
/// render backend compares against its last-synced snapshot to detect stale
/// GPU resources.
pub struct AssetStorage<H: Key, T> {
    inner: RwLock<StorageInner<H, T>>,
    /// Global mutation epoch — bumped on every write, enabling O(1) "anything
    /// changed?" checks by the render loop without iterating entries.
    global_version: AtomicU32,
}

impl<H: Key, T> Default for AssetStorage<H, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Key, T> AssetStorage<H, T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RwLock::default(),
            global_version: AtomicU32::new(0),
        }
    }

    #[inline]
    fn allocate_identity() -> u64 {
        NEXT_LOADED_ASSET_IDENTITY.fetch_add(1, Ordering::Relaxed)
    }

    /// Returns the global mutation epoch.
    #[inline]
    pub fn global_version(&self) -> u32 {
        self.global_version.load(Ordering::Relaxed)
    }

    // ── Immediate insertion (asset data available now) ───────────────────

    /// Inserts a fully-loaded resource and returns its handle.
    pub fn add(&self, asset: impl Into<T>) -> H {
        let mut guard = self.inner.write();
        let slot = AssetSlot::Loaded(AssetEntry {
            asset: Arc::new(asset.into()),
            version: 1,
            identity: Self::allocate_identity(),
        });
        self.global_version.fetch_add(1, Ordering::Relaxed);
        guard.map.insert(slot)
    }

    /// Inserts a fully-loaded resource keyed by UUID for deduplication.
    ///
    /// If an entry with the same UUID already exists **and** is in `Loading`
    /// state (i.e. a prior async task reserved it), the slot is promoted to
    /// `Loaded` so all holders of the handle see the data immediately.
    /// If the slot is already `Loaded`, the existing handle is returned as-is.
    pub fn add_with_uuid(&self, uuid: Uuid, asset: impl Into<T>) -> H {
        let mut guard = self.inner.write();
        if let Some(&handle) = guard.lookup.get(&uuid) {
            if let Some(slot) = guard.map.get_mut(handle)
                && !slot.is_loaded()
            {
                // includes Loading and Failed states
                *slot = AssetSlot::Loaded(AssetEntry {
                    asset: Arc::new(asset.into()),
                    version: 1,
                    identity: Self::allocate_identity(),
                });
                self.global_version.fetch_add(1, Ordering::Relaxed);
            }

            return handle;
        }
        let slot = AssetSlot::Loaded(AssetEntry {
            asset: Arc::new(asset.into()),
            version: 1,
            identity: Self::allocate_identity(),
        });
        let handle = guard.map.insert(slot);
        guard.lookup.insert(uuid, handle);
        self.global_version.fetch_add(1, Ordering::Relaxed);
        handle
    }

    // ── Deferred insertion (reserve now, fill later) ────────────────────

    /// Pre-allocates a handle in `Loading` state.
    ///
    /// The returned handle is immediately usable as a placeholder (e.g. to
    /// bind into a material). When the background task finishes, call
    /// [`insert_ready`](Self::insert_ready) or [`mark_failed`](Self::mark_failed).
    pub fn reserve(&self) -> H {
        let mut guard = self.inner.write();
        guard.map.insert(AssetSlot::Loading)
    }

    /// Pre-allocates a handle keyed by UUID, with built-in deduplication.
    ///
    /// Returns `(handle, is_new)`: when `is_new` is `true` the caller must
    /// spawn a background task to fill the slot; when `false` the resource
    /// is already loading or loaded and no new work is needed.
    pub fn reserve_with_uuid(&self, uuid: Uuid) -> (H, bool) {
        let mut guard = self.inner.write();
        if let Some(&handle) = guard.lookup.get(&uuid) {
            return (handle, false);
        }
        let handle = guard.map.insert(AssetSlot::Loading);
        guard.lookup.insert(uuid, handle);
        (handle, true)
    }

    /// Fills a previously-reserved handle with loaded data.
    ///
    /// This is an atomic state + data transition: the slot moves from
    /// `Loading` to `Loaded` in a single write, so no observer can ever
    /// see a half-initialised entry.
    pub fn insert_ready(&self, handle: H, asset: impl Into<T>) {
        let mut guard = self.inner.write();
        if let Some(slot) = guard.map.get_mut(handle) {
            *slot = AssetSlot::Loaded(AssetEntry {
                asset: Arc::new(asset.into()),
                version: 1,
                identity: Self::allocate_identity(),
            });
            self.global_version.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Marks a previously-reserved handle as failed.
    pub fn mark_failed(&self, handle: H, error: String) {
        let mut guard = self.inner.write();
        if let Some(slot) = guard.map.get_mut(handle) {
            *slot = AssetSlot::Failed(error);
        }
    }

    // ── Mutation ────────────────────────────────────────────────────────

    /// Replaces the asset data at `handle`, incrementing its version.
    ///
    /// Returns the new version, or `None` if the handle is invalid or not
    /// in the `Loaded` state.
    pub fn update(&self, handle: H, asset: impl Into<T>) -> Option<u32> {
        let mut guard = self.inner.write();
        if let Some(AssetSlot::Loaded(entry)) = guard.map.get_mut(handle) {
            entry.asset = Arc::new(asset.into());
            entry.version += 1;
            entry.identity = Self::allocate_identity();
            self.global_version.fetch_add(1, Ordering::Relaxed);
            Some(entry.version)
        } else {
            None
        }
    }

    /// Removes the slot (any state) and returns the previous value.
    pub fn remove(&self, handle: H) -> Option<AssetSlot<T>> {
        let mut guard = self.inner.write();
        guard.map.remove(handle)
    }

    // ── Read accessors ─────────────────────────────────────────────────

    /// Gets the loaded resource data, or `None` if the handle is invalid,
    /// still loading, or failed.
    pub fn get(&self, handle: H) -> Option<Arc<T>> {
        let guard = self.inner.read();
        match guard.map.get(handle) {
            Some(AssetSlot::Loaded(entry)) => Some(entry.asset.clone()),
            _ => None,
        }
    }

    /// Gets the loaded entry with its version, or `None` if unavailable.
    pub fn get_entry(&self, handle: H) -> Option<(Arc<T>, u32)> {
        let guard = self.inner.read();
        match guard.map.get(handle) {
            Some(AssetSlot::Loaded(entry)) => Some((entry.asset.clone(), entry.version)),
            _ => None,
        }
    }

    /// Gets the loaded resource, its public content version, and the exact
    /// identity of the currently stored `Arc`.
    ///
    /// The identity closes the ABA gap for systems that may observe an asset
    /// being repopulated under the same handle with a version reset to `1`.
    pub fn get_entry_with_identity(&self, handle: H) -> Option<(Arc<T>, u32, u64)> {
        let guard = self.inner.read();
        match guard.map.get(handle) {
            Some(AssetSlot::Loaded(entry)) => {
                Some((entry.asset.clone(), entry.version, entry.identity))
            }
            _ => None,
        }
    }

    /// Gets just the version of a loaded resource.
    pub fn get_version(&self, handle: H) -> Option<u32> {
        let guard = self.inner.read();
        match guard.map.get(handle) {
            Some(AssetSlot::Loaded(entry)) => Some(entry.version),
            _ => None,
        }
    }

    /// Gets the public content version and exact loaded identity without
    /// cloning the asset `Arc`.
    #[inline]
    pub fn get_version_and_identity(&self, handle: H) -> Option<(u32, u64)> {
        let guard = self.inner.read();
        match guard.map.get(handle) {
            Some(AssetSlot::Loaded(entry)) => Some((entry.version, entry.identity)),
            _ => None,
        }
    }

    /// Captures input-aligned loaded owners and exact storage stamps in bounded
    /// read-side critical sections.
    ///
    /// `out` is caller-owned and reused across calls. Existing snapshots are
    /// dropped and any required capacity is reserved before a storage lock is
    /// acquired. Inputs longer than [`ASSET_READ_BATCH_WIDTH`] are deliberately
    /// split across independent lock acquisitions; this is therefore a set of
    /// exact per-entry observations, not one globally atomic storage snapshot.
    /// Callers that need one retained owner per unique asset should
    /// stable-deduplicate repeated handles before calling this method; input
    /// order and duplicates are otherwise preserved exactly.
    pub fn snapshot_batched(&self, handles: &[H], out: &mut Vec<AssetSnapshot<H, T>>) {
        // Dropping retained Arcs or growing the output may invoke arbitrary
        // destructors/allocator code, so both operations must precede locking.
        out.clear();
        out.reserve(handles.len());

        for chunk in handles.chunks(ASSET_READ_BATCH_WIDTH) {
            self.append_snapshot_chunk(chunk, out);
        }
    }

    /// Revalidates input-aligned snapshots in bounded read-side critical
    /// sections without cloning any asset owner.
    ///
    /// A loaded observation remains current only when its version, process-wide
    /// identity, and exact `Arc` allocation all match. An unavailable
    /// observation remains current while the handle is still not loaded.
    pub fn revalidate_batched(&self, snapshots: &[AssetSnapshot<H, T>], out: &mut Vec<bool>) {
        out.clear();
        out.reserve(snapshots.len());

        for chunk in snapshots.chunks(ASSET_READ_BATCH_WIDTH) {
            self.append_revalidation_chunk(chunk, out);
        }
    }

    /// Revalidates only selected snapshot positions in bounded read-side
    /// critical sections.
    ///
    /// `indices` order and duplicates are preserved in `out`; an out-of-range
    /// index fails closed as `false`. This lets a demand-driven consumer keep a
    /// CPU-visible owner table while scanning only its stable-deduplicated live
    /// subset after graph compilation.
    pub fn revalidate_selected_batched(
        &self,
        snapshots: &[AssetSnapshot<H, T>],
        indices: &[usize],
        out: &mut Vec<bool>,
    ) {
        out.clear();
        out.reserve(indices.len());

        for chunk in indices.chunks(ASSET_READ_BATCH_WIDTH) {
            self.append_selected_revalidation_chunk(snapshots, chunk, out);
        }
    }

    /// Revalidates selected snapshots and opens a bounded publication gate
    /// while each storage read guard is still held.
    ///
    /// `indices` are processed in chunks no wider than
    /// [`ASSET_READ_BATCH_WIDTH`]. For each chunk, `publish_chunk` receives its
    /// starting position in the complete selected input, the original indices,
    /// and an equally sized, position-aligned current mask. Order and duplicates
    /// are preserved; an out-of-range index fails closed as `false`. The same
    /// masks are appended to `out`, whose allocation is retained and reused
    /// across calls. Revalidation only compares stamps and `Arc` allocation
    /// identity; it never clones an asset owner.
    ///
    /// The callback runs **before** that chunk's storage read guard is released.
    /// It may therefore publish derived state to an external cache for mask
    /// entries that are `true`, knowing the corresponding exact snapshots
    /// cannot be replaced concurrently. The guarantee is deliberately local
    /// to one bounded chunk; writers may run between successive callbacks.
    ///
    /// # Reentrancy
    ///
    /// `publish_chunk` must not call any method on this same `AssetStorage`.
    /// In particular, attempting to acquire its write lock would deadlock, and
    /// recursively acquiring a read lock can deadlock when a writer is queued
    /// because `parking_lot::RwLock` uses eventual fairness. Keep the callback
    /// bounded and restrict it to external cache publication.
    pub fn revalidate_selected_batched_at_publish(
        &self,
        snapshots: &[AssetSnapshot<H, T>],
        indices: &[usize],
        out: &mut Vec<bool>,
        mut publish_chunk: impl FnMut(usize, &[usize], &[bool]),
    ) {
        // Clear/grow outside the lock so dropping old contents or allocator
        // work can never lengthen the read-side critical section.
        out.clear();
        out.reserve(indices.len());

        for chunk in indices.chunks(ASSET_READ_BATCH_WIDTH) {
            let output_start = out.len();
            let guard = self.inner.read();
            Self::append_selected_revalidation_chunk_locked(&guard, snapshots, chunk, out);
            publish_chunk(output_start, chunk, &out[output_start..]);
            // Keep this explicit: publication must finish before the exact
            // snapshot is allowed to change.
            drop(guard);
        }
    }

    #[inline]
    fn append_snapshot_chunk(&self, handles: &[H], out: &mut Vec<AssetSnapshot<H, T>>) {
        debug_assert!(handles.len() <= ASSET_READ_BATCH_WIDTH);
        debug_assert!(out.capacity() - out.len() >= handles.len());

        let guard = self.inner.read();
        for &handle in handles {
            let loaded = guard.get_loaded(handle).map(|entry| LoadedAssetSnapshot {
                asset: Arc::clone(&entry.asset),
                version: entry.version,
                identity: entry.identity,
            });
            out.push(AssetSnapshot { handle, loaded });
        }
    }

    #[inline]
    fn append_revalidation_chunk(&self, snapshots: &[AssetSnapshot<H, T>], out: &mut Vec<bool>) {
        debug_assert!(snapshots.len() <= ASSET_READ_BATCH_WIDTH);
        debug_assert!(out.capacity() - out.len() >= snapshots.len());

        let guard = self.inner.read();
        for snapshot in snapshots {
            out.push(Self::snapshot_is_current(&guard, snapshot));
        }
    }

    #[inline]
    fn append_selected_revalidation_chunk(
        &self,
        snapshots: &[AssetSnapshot<H, T>],
        indices: &[usize],
        out: &mut Vec<bool>,
    ) {
        debug_assert!(indices.len() <= ASSET_READ_BATCH_WIDTH);
        debug_assert!(out.capacity() - out.len() >= indices.len());

        let guard = self.inner.read();
        Self::append_selected_revalidation_chunk_locked(&guard, snapshots, indices, out);
    }

    #[inline]
    fn append_selected_revalidation_chunk_locked(
        guard: &StorageInner<H, T>,
        snapshots: &[AssetSnapshot<H, T>],
        indices: &[usize],
        out: &mut Vec<bool>,
    ) {
        debug_assert!(indices.len() <= ASSET_READ_BATCH_WIDTH);
        debug_assert!(out.capacity() - out.len() >= indices.len());

        for &index in indices {
            out.push(
                snapshots
                    .get(index)
                    .is_some_and(|snapshot| Self::snapshot_is_current(guard, snapshot)),
            );
        }
    }

    #[inline]
    fn snapshot_is_current(guard: &StorageInner<H, T>, snapshot: &AssetSnapshot<H, T>) -> bool {
        let current = guard.get_loaded(snapshot.handle);
        match (&snapshot.loaded, current) {
            (None, None) => true,
            (Some(expected), Some(current)) => {
                current.version == expected.version
                    && current.identity == expected.identity
                    && Arc::ptr_eq(&current.asset, &expected.asset)
            }
            _ => false,
        }
    }

    #[cfg(test)]
    fn snapshot_batched_with_chunk_hook(
        &self,
        handles: &[H],
        out: &mut Vec<AssetSnapshot<H, T>>,
        mut after_chunk: impl FnMut(usize),
    ) {
        out.clear();
        out.reserve(handles.len());

        let mut completed_chunks = 0;
        for chunk in handles.chunks(ASSET_READ_BATCH_WIDTH) {
            self.append_snapshot_chunk(chunk, out);
            completed_chunks += 1;
            if out.len() < handles.len() {
                // append_snapshot_chunk's guard has been dropped before this
                // test-only observation point.
                after_chunk(completed_chunks);
            }
        }
    }

    /// Queries the lifecycle state of a slot.
    pub fn get_state(&self, handle: H) -> Option<&'static str> {
        let guard = self.inner.read();
        guard.map.get(handle).map(|slot| match slot {
            AssetSlot::Loading => "loading",
            AssetSlot::Loaded(_) => "loaded",
            AssetSlot::Failed(_) => "failed",
        })
    }

    /// Returns `true` if the handle points to a `Loaded` slot.
    pub fn is_loaded(&self, handle: H) -> bool {
        let guard = self.inner.read();
        matches!(guard.map.get(handle), Some(AssetSlot::Loaded(_)))
    }

    /// Returns `true` if the handle points to a `Loading` slot.
    pub fn is_loading(&self, handle: H) -> bool {
        let guard = self.inner.read();
        if let Some(slot) = guard.map.get(handle) {
            slot.is_loading()
        } else {
            false
        }
    }

    /// Returns `true` if the handle points to a `Failed` slot.
    pub fn is_failed(&self, handle: H) -> bool {
        let guard = self.inner.read();
        if let Some(slot) = guard.map.get(handle) {
            matches!(slot, AssetSlot::Failed(_))
        } else {
            false
        }
    }

    /// Returns the error message if the handle points to a `Failed` slot.
    pub fn get_error(&self, handle: H) -> Option<String> {
        let guard = self.inner.read();
        match guard.map.get(handle) {
            Some(AssetSlot::Failed(msg)) => Some(msg.clone()),
            _ => None,
        }
    }

    pub fn get_by_uuid(&self, uuid: &Uuid) -> Option<Arc<T>> {
        let guard = self.inner.read();
        let handle = guard.lookup.get(uuid)?;
        match guard.map.get(*handle) {
            Some(AssetSlot::Loaded(entry)) => Some(entry.asset.clone()),
            _ => None,
        }
    }

    /// Gets a Handle by UUID (when only the UUID is known).
    pub fn get_handle_by_uuid(&self, uuid: &Uuid) -> Option<H> {
        let guard = self.inner.read();
        guard.lookup.get(uuid).copied()
    }

    /// Acquires a read-lock guard for batch access.
    ///
    /// Use [`StorageInner::get_loaded`] on the returned guard to access
    /// individual entries without re-acquiring the lock on each lookup.
    pub fn read_lock(&self) -> RwLockReadGuard<'_, StorageInner<H, T>> {
        self.inner.read()
    }

    // ── Cache invalidation ─────────────────────────────────────────────

    /// Removes the UUID association for a specific handle, resetting it to
    /// `Loading` so a fresh background task can re-populate the slot.
    ///
    /// Returns `true` if the UUID was found and removed.
    pub fn invalidate_uuid(&self, uuid: &Uuid) -> bool {
        let mut guard = self.inner.write();
        if let Some(handle) = guard.lookup.remove(uuid) {
            if let Some(slot) = guard.map.get_mut(handle) {
                *slot = AssetSlot::Loading;
            }
            true
        } else {
            false
        }
    }

    /// Evicts **all** UUID mappings, resetting every tracked slot to
    /// `Loading`. Existing handles remain valid but will need to be
    /// re-populated by fresh background tasks.
    pub fn invalidate_all_uuids(&self) {
        let mut guard = self.inner.write();
        let handles: Vec<_> = guard.lookup.values().copied().collect();
        for handle in handles {
            if let Some(slot) = guard.map.get_mut(handle) {
                *slot = AssetSlot::Loading;
            }
        }
        guard.lookup.clear();
        self.global_version.fetch_add(1, Ordering::Relaxed);
    }
}

impl AssetStorage<ImageHandle, Image> {
    /// Overwrites a dynamic image buffer in place and bumps the asset version.
    ///
    /// This is the zero-allocation update path for streaming images. The image
    /// must have been created with [`Image::new_dynamic`] and `new_bytes` must
    /// match the preallocated buffer length exactly.
    pub fn update_dynamic_data(
        &self,
        handle: ImageHandle,
        new_bytes: &[u8],
    ) -> Result<u32, DynamicImageUpdateError> {
        let mut guard = self.inner.write();
        let Some(slot) = guard.map.get_mut(handle) else {
            return Err(DynamicImageUpdateError::InvalidHandle);
        };
        let Some(entry) = slot.as_loaded_mut() else {
            return Err(DynamicImageUpdateError::NotLoaded);
        };

        entry
            .asset
            .update_dynamic_data(new_bytes)
            .map_err(DynamicImageUpdateError::Update)?;

        entry.version = entry.version.wrapping_add(1);
        if entry.version == 0 {
            // Dynamic updates intentionally keep the same Arc allocation. On
            // the extremely rare public-version wrap, advance exact identity
            // so `(version, identity)` still cannot ABA against old GPU state.
            entry.identity = Self::allocate_identity();
        }
        self.global_version.fetch_add(1, Ordering::Relaxed);
        Ok(entry.version)
    }
}

#[cfg(test)]
mod tests {
    use myth_resources::image::{ImageDimension, PixelFormat};

    use super::*;

    fn loaded_value(snapshot: &AssetSnapshot<ImageHandle, u32>) -> Option<u32> {
        snapshot.loaded().map(|loaded| **loaded.asset())
    }

    #[test]
    fn batch_snapshot_preserves_input_order_and_unavailable_slots() {
        let storage = AssetStorage::<ImageHandle, u32>::new();
        let first = storage.add(10_u32);
        let loading = storage.reserve();
        let failed = storage.reserve();
        storage.mark_failed(failed, "expected failure".to_string());
        let last = storage.add(40_u32);
        let handles = [last, loading, first, failed];

        let mut snapshots = Vec::new();
        storage.snapshot_batched(&handles, &mut snapshots);

        assert_eq!(snapshots.len(), handles.len());
        for (snapshot, handle) in snapshots.iter().zip(handles) {
            assert_eq!(snapshot.handle(), handle);
        }
        assert_eq!(loaded_value(&snapshots[0]), Some(40));
        assert!(!snapshots[1].is_loaded());
        assert_eq!(loaded_value(&snapshots[2]), Some(10));
        assert!(!snapshots[3].is_loaded());

        let mut current = Vec::new();
        storage.revalidate_batched(&snapshots, &mut current);
        assert_eq!(current, [true, true, true, true]);
        storage.revalidate_selected_batched(&snapshots, &[2, 0, usize::MAX, 2], &mut current);
        assert_eq!(current, [true, true, false, true]);

        storage.insert_ready(loading, 20_u32);
        storage.revalidate_batched(&snapshots, &mut current);
        assert_eq!(current, [true, false, true, true]);
    }

    #[test]
    fn batch_width_bounds_each_read_lock_critical_section() {
        let storage = AssetStorage::<ImageHandle, u32>::new();
        let handles: Vec<_> = (0..=(ASSET_READ_BATCH_WIDTH * 2))
            .map(|value| storage.add(value as u32))
            .collect();
        let mut snapshots = Vec::new();

        for len in [
            0,
            ASSET_READ_BATCH_WIDTH - 1,
            ASSET_READ_BATCH_WIDTH,
            ASSET_READ_BATCH_WIDTH + 1,
            ASSET_READ_BATCH_WIDTH * 2,
            ASSET_READ_BATCH_WIDTH * 2 + 1,
        ] {
            let mut hooks = 0;
            storage
                .snapshot_batched_with_chunk_hook(&handles[..len], &mut snapshots, |_| hooks += 1);
            assert_eq!(hooks, len.saturating_sub(1) / ASSET_READ_BATCH_WIDTH);
            assert_eq!(snapshots.len(), len);
        }

        let mut cross_boundary_hooks = 0;
        storage.snapshot_batched_with_chunk_hook(
            &handles[..=ASSET_READ_BATCH_WIDTH],
            &mut snapshots,
            |completed| {
                cross_boundary_hooks += 1;
                assert_eq!(completed, 1);
                assert!(
                    storage.inner.try_write().is_some(),
                    "the read guard must be released between chunks"
                );
                assert_eq!(
                    storage.update(handles[ASSET_READ_BATCH_WIDTH], 999_u32),
                    Some(2)
                );
            },
        );

        assert_eq!(cross_boundary_hooks, 1);
        assert_eq!(snapshots.len(), ASSET_READ_BATCH_WIDTH + 1);
        let last = snapshots.last().unwrap().loaded().unwrap();
        assert_eq!(**last.asset(), 999);
        assert_eq!(last.version(), 2);
    }

    #[test]
    fn replacement_repopulation_and_slot_reuse_fail_exact_revalidation() {
        let storage = AssetStorage::<ImageHandle, u32>::new();
        let handle = storage.add(1_u32);
        let mut snapshots = Vec::new();
        let mut current = Vec::new();

        storage.snapshot_batched(&[handle], &mut snapshots);
        let first_identity = snapshots[0].loaded().unwrap().identity();
        assert_eq!(storage.update(handle, 2_u32), Some(2));
        storage.revalidate_batched(&snapshots, &mut current);
        assert_eq!(current, [false]);

        storage.snapshot_batched(&[handle], &mut snapshots);
        let replacement_identity = snapshots[0].loaded().unwrap().identity();
        assert_ne!(replacement_identity, first_identity);
        storage.insert_ready(handle, 3_u32);
        let (_, repopulated_version, repopulated_identity) = storage
            .get_entry_with_identity(handle)
            .expect("repopulated asset missing");
        assert_eq!(repopulated_version, 1);
        assert_ne!(repopulated_identity, replacement_identity);
        storage.revalidate_batched(&snapshots, &mut current);
        assert_eq!(current, [false]);

        storage.snapshot_batched(&[handle], &mut snapshots);
        drop(storage.remove(handle).expect("asset slot missing"));
        let reused = storage.add(4_u32);
        assert_ne!(reused, handle, "SlotMap generation must change on reuse");
        storage.revalidate_batched(&snapshots, &mut current);
        assert_eq!(current, [false]);
    }

    #[test]
    fn batch_output_capacity_is_reused() {
        let storage = AssetStorage::<ImageHandle, u32>::new();
        let handles: Vec<_> = (0..(ASSET_READ_BATCH_WIDTH * 2))
            .map(|value| storage.add(value as u32))
            .collect();
        let mut snapshots = Vec::new();

        storage.snapshot_batched(&handles, &mut snapshots);
        let allocation = snapshots.as_ptr();
        let capacity = snapshots.capacity();
        assert!(capacity >= handles.len());

        storage.snapshot_batched(&handles[..1], &mut snapshots);
        assert_eq!(snapshots.as_ptr(), allocation);
        assert_eq!(snapshots.capacity(), capacity);
        assert_eq!(snapshots.len(), 1);
    }

    #[test]
    fn batch_revalidation_does_not_clone_retained_asset_owners() {
        let storage = AssetStorage::<ImageHandle, u32>::new();
        let handle = storage.add(7_u32);
        let mut snapshots = Vec::new();
        storage.snapshot_batched(&[handle], &mut snapshots);

        let retained = snapshots[0].loaded().unwrap().asset();
        let owners_before = Arc::strong_count(retained);
        let mut current = Vec::new();
        storage.revalidate_batched(&snapshots, &mut current);

        assert_eq!(current, [true]);
        assert_eq!(Arc::strong_count(retained), owners_before);
        storage.revalidate_selected_batched(&snapshots, &[0, 0], &mut current);
        assert_eq!(current, [true, true]);
        assert_eq!(Arc::strong_count(retained), owners_before);
    }

    #[test]
    fn selected_publish_gate_preserves_chunk_and_output_alignment() {
        let storage = AssetStorage::<ImageHandle, u32>::new();
        let handles: Vec<_> = (0..=ASSET_READ_BATCH_WIDTH)
            .map(|value| storage.add(value as u32))
            .collect();
        let mut snapshots = Vec::new();
        storage.snapshot_batched(&handles, &mut snapshots);
        assert_eq!(storage.update(handles[1], 999_u32), Some(2));

        let mut indices: Vec<_> = (0..handles.len()).collect();
        indices.extend([usize::MAX, 1]);
        let mut current = Vec::new();
        let mut observed_indices = Vec::new();
        let mut observed_current = Vec::new();
        let mut chunk_lengths = Vec::new();
        let mut chunk_starts = Vec::new();

        storage.revalidate_selected_batched_at_publish(
            &snapshots,
            &indices,
            &mut current,
            |start, chunk, chunk_current| {
                assert_eq!(start, observed_indices.len());
                assert_eq!(chunk.len(), chunk_current.len());
                assert!(chunk.len() <= ASSET_READ_BATCH_WIDTH);
                assert!(
                    storage.inner.try_write().is_none(),
                    "publication must run while the storage read guard is held"
                );
                chunk_starts.push(start);
                chunk_lengths.push(chunk.len());
                observed_indices.extend_from_slice(chunk);
                observed_current.extend_from_slice(chunk_current);
            },
        );

        assert_eq!(chunk_starts, [0, ASSET_READ_BATCH_WIDTH]);
        assert_eq!(chunk_lengths, [ASSET_READ_BATCH_WIDTH, 3]);
        assert_eq!(observed_indices, indices);
        assert_eq!(observed_current, current);
        assert_eq!(current.len(), indices.len());
        for (position, &index) in indices.iter().enumerate() {
            let expected = index != usize::MAX && index != 1;
            assert_eq!(current[position], expected, "position {position}");
        }
        assert!(
            storage.inner.try_write().is_some(),
            "the final batch read guard must be released after publication"
        );
    }

    #[test]
    fn selected_publish_gate_reuses_capacity_without_cloning_owners() {
        let storage = AssetStorage::<ImageHandle, u32>::new();
        let handles: Vec<_> = (0..=ASSET_READ_BATCH_WIDTH)
            .map(|value| storage.add(value as u32))
            .collect();
        let mut snapshots = Vec::new();
        storage.snapshot_batched(&handles, &mut snapshots);
        let retained = snapshots[0].loaded().unwrap().asset();
        let owners_before = Arc::strong_count(retained);
        let all_indices: Vec<_> = (0..handles.len()).collect();
        let mut current = Vec::new();

        storage.revalidate_selected_batched_at_publish(
            &snapshots,
            &all_indices,
            &mut current,
            |_, _, _| {},
        );
        let allocation = current.as_ptr();
        let capacity = current.capacity();
        assert!(capacity >= all_indices.len());
        assert_eq!(Arc::strong_count(retained), owners_before);

        storage.revalidate_selected_batched_at_publish(
            &snapshots,
            &[0, 0],
            &mut current,
            |start, indices, mask| {
                assert_eq!(start, 0);
                assert_eq!(indices, [0, 0]);
                assert_eq!(mask, [true, true]);
            },
        );
        assert_eq!(current.as_ptr(), allocation);
        assert_eq!(current.capacity(), capacity);
        assert_eq!(current, [true, true]);
        assert_eq!(Arc::strong_count(retained), owners_before);
    }

    fn dynamic_image(value: u8) -> Image {
        Image::new_dynamic(
            2,
            2,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            vec![value; 16],
        )
    }

    #[test]
    fn loaded_identity_distinguishes_replacements_from_in_place_updates() {
        let storage = AssetStorage::<ImageHandle, Image>::new();
        let handle = storage.add(dynamic_image(1));
        let (_, first_version, first_identity) = storage
            .get_entry_with_identity(handle)
            .expect("initial image missing");
        assert_eq!(first_version, 1);

        assert_eq!(storage.update(handle, dynamic_image(2)), Some(2));
        let (_, updated_version, updated_identity) = storage
            .get_entry_with_identity(handle)
            .expect("updated image missing");
        assert_eq!(updated_version, 2);
        assert_ne!(updated_identity, first_identity);

        storage.insert_ready(handle, dynamic_image(3));
        let (_, reloaded_version, reloaded_identity) = storage
            .get_entry_with_identity(handle)
            .expect("reloaded image missing");
        assert_eq!(reloaded_version, 1);
        assert_ne!(reloaded_identity, updated_identity);

        storage
            .update_dynamic_data(handle, &[4; 16])
            .expect("dynamic update failed");
        let (_, dynamic_version, dynamic_identity) = storage
            .get_entry_with_identity(handle)
            .expect("dynamic image missing");
        assert_eq!(dynamic_version, 2);
        assert_eq!(dynamic_identity, reloaded_identity);

        {
            let mut guard = storage.inner.write();
            guard
                .map
                .get_mut(handle)
                .and_then(AssetSlot::as_loaded_mut)
                .expect("dynamic image slot missing")
                .version = u32::MAX;
        }
        storage
            .update_dynamic_data(handle, &[5; 16])
            .expect("wrapping dynamic update failed");
        let (_, wrapped_version, wrapped_identity) = storage
            .get_entry_with_identity(handle)
            .expect("wrapped dynamic image missing");
        assert_eq!(wrapped_version, 0);
        assert_ne!(wrapped_identity, dynamic_identity);

        let other_storage = AssetStorage::<ImageHandle, Image>::new();
        let other_handle = other_storage.add(dynamic_image(5));
        let (_, other_version, other_identity) = other_storage
            .get_entry_with_identity(other_handle)
            .expect("other storage image missing");
        assert_eq!(other_version, 1);
        assert_ne!(other_identity, dynamic_identity);
    }
}
