//! Registry for internally generated texture views.

use rustc_hash::FxHashMap;

use super::generate_gpu_resource_id;

struct InternalTextureRecord {
    view: wgpu::TextureView,
    /// Monotonic content revision for one stable logical attachment ID.
    revision: u64,
    /// Physical identity used by bind-group fingerprints.
    view_id: u64,
}

/// Internally generated texture views such as render targets and utility LUTs.
#[derive(Default)]
pub(crate) struct InternalTextureRegistry {
    /// Key: stable logical resource ID, value: current physical view record.
    resources: FxHashMap<u64, InternalTextureRecord>,
    /// Mapping from stable internal names to resource IDs.
    name_lookup: FxHashMap<String, u64>,
}

impl InternalTextureRegistry {
    pub(super) fn register_direct(&mut self, id: u64, view: wgpu::TextureView) {
        // Use the renderer-wide monotonic identity rather than a per-record
        // counter: removing and later re-registering the same logical ID must
        // never recycle an old content revision.
        let revision = generate_gpu_resource_id();
        self.resources.insert(
            id,
            InternalTextureRecord {
                view,
                revision,
                view_id: revision,
            },
        );
    }

    pub(super) fn register_by_name(&mut self, name: &str, view: wgpu::TextureView) -> u64 {
        let id = *self
            .name_lookup
            .entry(name.to_string())
            .or_insert_with(generate_gpu_resource_id);
        self.register_direct(id, view);
        id
    }

    pub(super) fn register(&mut self, view: wgpu::TextureView) -> u64 {
        let id = generate_gpu_resource_id();
        self.register_direct(id, view);
        id
    }

    #[inline]
    #[must_use]
    pub(super) fn get(&self, id: u64) -> Option<&wgpu::TextureView> {
        self.resources.get(&id).map(|record| &record.view)
    }

    #[inline]
    #[must_use]
    pub(super) fn revision(&self, id: u64) -> Option<u64> {
        self.resources.get(&id).map(|record| record.revision)
    }

    #[inline]
    #[must_use]
    pub(super) fn view_id(&self, id: u64) -> Option<u64> {
        self.resources.get(&id).map(|record| record.view_id)
    }

    pub(super) fn mark_content_dirty(&mut self, id: u64) -> bool {
        let Some(record) = self.resources.get_mut(&id) else {
            return false;
        };
        record.revision = generate_gpu_resource_id();
        true
    }

    pub(super) fn remove(&mut self, id: u64) {
        self.resources.remove(&id);
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn contains_name(&self, name: &str) -> bool {
        self.name_lookup.contains_key(name)
    }
}
