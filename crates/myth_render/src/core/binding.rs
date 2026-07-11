//! Bind-group cache and the [`Bindings`] trait.
//!
//! The [`myth_resources::BindingResource`] and [`ResourceBuilder`] data types live in
//! `myth_resources`; this module provides the **trait** that connects
//! domain objects (Material, Geometry, Mesh, etc.) to the builder, plus
//! a frame-based bind-group cache with TTL eviction.

use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::graph::RenderState;
use myth_resources::Mesh;
use myth_resources::ResourceBuilder;
use myth_resources::geometry::Geometry;
use myth_resources::material::{Material, RenderableMaterialTrait};
use myth_resources::uniforms::{MorphUniforms, RenderStateUniforms};
use myth_scene::Scene;

/// Binding resource Trait
pub trait Bindings {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>);
}

impl Bindings for Material {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        self.data.define_bindings(builder);
    }
}

impl Bindings for Geometry {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        // Morph Target Storage Buffers
        if self.has_morph_targets() {
            // Position morph storage
            if let (Some(buffer), Some(data)) =
                (&self.morph_position_buffer, self.morph_position_bytes())
            {
                builder.add_storage_buffer(
                    "morph_positions",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(myth_resources::WgslStructName::Name("f32".into())),
                );
            }

            // Normal morph storage (optional)
            if let (Some(buffer), Some(data)) =
                (&self.morph_normal_buffer, self.morph_normal_bytes())
            {
                builder.add_storage_buffer(
                    "morph_normals",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(myth_resources::WgslStructName::Name("f32".into())),
                );
            }

            // Tangent morph storage (optional)
            if let (Some(buffer), Some(data)) =
                (&self.morph_tangent_buffer, self.morph_tangent_bytes())
            {
                builder.add_storage_buffer(
                    "morph_tangents",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(myth_resources::WgslStructName::Name("f32".into())),
                );
            }
        }
    }
}

impl Bindings for Mesh {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        // todo: should we check if geometry features contain USE_MORPHING?
        builder.add_uniform::<MorphUniforms>(
            "morph_targets",
            &self.morph_uniforms,
            wgpu::ShaderStages::VERTEX,
        );
    }
}

impl Bindings for RenderState {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        builder.add_uniform::<RenderStateUniforms>(
            "render_state",
            self.uniforms(),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
        );
    }
}

impl Bindings for Scene {
    fn define_bindings<'a>(&'a self, _builder: &mut ResourceBuilder<'a>) {
        // Scene-level global bindings are now built by
        // `ResourceManager::define_global_scene_bindings` which resolves
        // environment textures from the GPU cache instead of from Environment.
        // This impl is kept empty for trait coherence; the actual bindings
        // are constructed in `ResourceManager::create_global_state`.
    }
}

/// Number of idle frames before a cached bind group is evicted.
const BIND_GROUP_EVICTION_THRESHOLD: u64 = 10;

/// Process-unique token source for frame-owned prepared-bind-group stores.
/// Relaxed ordering is sufficient: uniqueness, not synchronization, is the
/// contract.
static NEXT_PREPARED_BIND_GROUP_ATTEMPT: AtomicU64 = AtomicU64::new(1);

fn next_prepared_bind_group_attempt() -> u64 {
    NEXT_PREPARED_BIND_GROUP_ATTEMPT
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .expect("prepared bind-group graph-attempt token space exhausted")
}

/// A cached bind group entry with TTL (Time-To-Live) tracking.
///
/// Records the last frame the bind group was accessed so that stale
/// entries can be garbage-collected when the associated physical
/// textures are released by the transient pool.
struct CachedBindGroup {
    bg: wgpu::BindGroup,
    last_accessed_frame: AtomicU64,
}

/// Global bind group cache with frame-based TTL eviction.
///
/// Caches `wgpu::BindGroup` instances keyed by their resource composition.
/// Each entry tracks the last frame it was accessed; entries idle for more
/// than [`BIND_GROUP_EVICTION_THRESHOLD`] consecutive frames are evicted
/// by [`garbage_collect`](Self::garbage_collect), ensuring that VRAM held
/// by stale bind groups (e.g. after SSAO is disabled or a resize occurs)
/// is promptly released.
///
/// RDG nodes never borrow entries from this cache across phases. Live prepare
/// clones each exact handle into its graph attempt's
/// [`PreparedBindGroupStore`], which provides Copy IDs to nodes and owns the
/// handles until execute or abort completes.
pub struct GlobalBindGroupCache {
    cache: FxHashMap<BindGroupKey, CachedBindGroup>,
    current_frame: u64,
}

impl Default for GlobalBindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalBindGroupCache {
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
            current_frame: 0,
        }
    }

    /// Advances the internal frame counter.
    ///
    /// Must be called once per frame **before** any `get` / `insert` /
    /// `get_or_create` calls so that TTL timestamps are correct.
    #[inline]
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Returns the current frame number.
    #[inline]
    #[must_use]
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Looks up a cached bind group, updating its TTL timestamp on hit.
    #[must_use]
    pub fn get(&self, key: &BindGroupKey) -> Option<&wgpu::BindGroup> {
        if let Some(entry) = self.cache.get(key) {
            entry
                .last_accessed_frame
                .store(self.current_frame, Ordering::Relaxed);
            Some(&entry.bg)
        } else {
            None
        }
    }

    /// Checks whether the cache contains an entry for the given key
    /// without updating its TTL.
    #[must_use]
    pub fn contains(&self, key: &BindGroupKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Inserts a bind group into the cache with the current frame timestamp.
    pub fn insert(&mut self, key: BindGroupKey, bind_group: wgpu::BindGroup) {
        self.cache.insert(
            key,
            CachedBindGroup {
                bg: bind_group,
                last_accessed_frame: AtomicU64::new(self.current_frame),
            },
        );
    }

    /// Returns an existing entry or creates one via `factory`, updating TTL.
    pub fn get_or_create(
        &mut self,
        key: BindGroupKey,
        factory: impl FnOnce() -> wgpu::BindGroup,
    ) -> &wgpu::BindGroup {
        let frame = self.current_frame;
        let entry = self.cache.entry(key).or_insert_with(|| CachedBindGroup {
            bg: factory(),
            last_accessed_frame: AtomicU64::new(frame),
        });
        entry.last_accessed_frame.store(frame, Ordering::Relaxed);
        &entry.bg
    }

    /// Forcibly clears all entries.
    ///
    /// Call on window resize to drop every bind group that references
    /// now-stale texture views.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Evicts entries that have not been accessed for
    /// [`BIND_GROUP_EVICTION_THRESHOLD`] frames.
    ///
    /// Should be called once at the **end** of each frame, after all
    /// passes have executed.  This keeps the cache in sync with the
    /// transient pool's own idle-eviction policy, preventing VRAM leaks
    /// from orphaned bind groups.
    pub fn garbage_collect(&mut self) {
        let threshold = self
            .current_frame
            .saturating_sub(BIND_GROUP_EVICTION_THRESHOLD);
        let before = self.cache.len();
        self.cache
            .retain(|_, entry| entry.last_accessed_frame.load(Ordering::Relaxed) >= threshold);
        let evicted = before - self.cache.len();
        if evicted > 0 {
            log::debug!(
                "GlobalBindGroupCache: evicted {evicted} stale entries, {} remaining",
                self.cache.len()
            );
        }
    }
}

/// Copy handle to a bind group owned by one graph attempt.
///
/// The process-unique 64-bit attempt token prevents a stale ID from aliasing a
/// later frame or a different renderer even when the vector index is reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedBindGroupId {
    attempt_token: u64,
    index: u32,
}

/// Destructor-aware, frame-scoped ownership for bind groups used by live RDG
/// nodes.
///
/// The cross-frame [`GlobalBindGroupCache`] remains the reuse layer, while
/// this store clones the exact WGPU handles needed by the current graph
/// attempt. Pass nodes retain only [`PreparedBindGroupId`] values, so clearing
/// or collecting the persistent cache can never invalidate an in-flight
/// graph. The attempt token turns stale-ID reuse into a checked lookup miss.
pub(crate) struct PreparedBindGroupStore {
    bind_groups: Vec<wgpu::BindGroup>,
    lookup: FxHashMap<BindGroupKey, PreparedBindGroupId>,
    attempt_token: u64,
}

impl Default for PreparedBindGroupStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PreparedBindGroupStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            bind_groups: Vec::new(),
            lookup: FxHashMap::default(),
            attempt_token: next_prepared_bind_group_attempt(),
        }
    }

    /// Returns the frame-local ID for `key`, cloning a cache-owned WGPU handle
    /// only on the first use in this graph attempt.
    pub(crate) fn get_or_create(
        &mut self,
        cache: &mut GlobalBindGroupCache,
        key: BindGroupKey,
        factory: impl FnOnce() -> wgpu::BindGroup,
    ) -> PreparedBindGroupId {
        if let Some(&id) = self.lookup.get(&key) {
            debug_assert!(self.resolve(id).is_some());
            return id;
        }

        let bind_group = cache.get_or_create(key.clone(), factory).clone();
        let index = u32::try_from(self.bind_groups.len())
            .expect("one graph attempt cannot prepare more than u32::MAX bind groups");
        let id = PreparedBindGroupId {
            attempt_token: self.attempt_token,
            index,
        };
        self.bind_groups.push(bind_group);
        let previous = self.lookup.insert(key, id);
        debug_assert!(previous.is_none());
        id
    }

    #[must_use]
    pub(crate) fn resolve(&self, id: PreparedBindGroupId) -> Option<&wgpu::BindGroup> {
        if id.attempt_token != self.attempt_token {
            return None;
        }
        self.bind_groups.get(id.index as usize)
    }

    pub(crate) fn clear(&mut self) {
        self.lookup.clear();
        self.bind_groups.clear();
        self.attempt_token = next_prepared_bind_group_attempt();
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.bind_groups.len()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn attempt_token(&self) -> u64 {
        self.attempt_token
    }
}

/// Global `BindGroup` cache key
/// Contains the unique identifier of the Layout + unique identifiers of all binding resources
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupKey {
    layout_id: u64,
    resources: SmallVec<[u64; 8]>,
}

impl BindGroupKey {
    #[must_use]
    pub fn new(layout_id: u64) -> Self {
        Self {
            layout_id,
            resources: SmallVec::with_capacity(8), // Estimated common size
        }
    }

    #[must_use]
    pub fn with_resource(mut self, id: u64) -> Self {
        self.resources.push(id);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WgpuContext;
    use crate::settings::{RendererInitConfig, RendererSettings};

    fn empty_bind_group(device: &wgpu::Device, label: &'static str) -> wgpu::BindGroup {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[],
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &[],
        })
    }

    #[test]
    fn frame_store_deduplicates_isolates_attempts_and_survives_cache_clear() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            1,
            1,
            None,
        ))
        .expect("headless GPU context required");

        let mut cache = GlobalBindGroupCache::new();
        cache.begin_frame();
        let mut prepared = PreparedBindGroupStore::new();
        let key = BindGroupKey::new(1).with_resource(7);
        let id = prepared.get_or_create(&mut cache, key.clone(), || {
            empty_bind_group(&wgpu.device, "Prepared BindGroup A")
        });
        let same = prepared.get_or_create(&mut cache, key.clone(), || {
            panic!("same-frame prepared bind group must be deduplicated")
        });
        assert_eq!(id, same);
        assert_eq!(prepared.len(), 1);

        let mut other_attempt = PreparedBindGroupStore::new();
        let other_id = other_attempt.get_or_create(&mut cache, key, || {
            panic!("the cross-frame cache already owns this bind group")
        });
        assert_ne!(id, other_id);
        assert!(other_attempt.resolve(id).is_none());
        assert!(prepared.resolve(other_id).is_none());

        let frame_owned = prepared
            .resolve(id)
            .expect("prepared bind group must resolve")
            .clone();
        cache.clear();
        assert_eq!(prepared.resolve(id), Some(&frame_owned));
        assert!(other_attempt.resolve(other_id).is_some());

        prepared.clear();
        assert!(prepared.resolve(id).is_none());

        let next = prepared.get_or_create(&mut cache, BindGroupKey::new(2), || {
            empty_bind_group(&wgpu.device, "Prepared BindGroup B")
        });
        assert_ne!(id, next);
        assert!(prepared.resolve(id).is_none());
        assert!(prepared.resolve(next).is_some());
    }
}
