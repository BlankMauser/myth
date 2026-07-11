use parking_lot::{RwLock, RwLockReadGuard};
use rustc_hash::FxHashMap;
use slotmap::{Key, SlotMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use uuid::Uuid;

use myth_resources::ImageHandle;
use myth_resources::image::{DynamicImageError, Image};

/// Process-wide exact identity for a loaded asset allocation. Keeping this
/// global (rather than per `AssetStorage`) also distinguishes two AssetServer
/// instances whose slot handles and public versions happen to coincide.
static NEXT_LOADED_ASSET_IDENTITY: AtomicU64 = AtomicU64::new(1);

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
