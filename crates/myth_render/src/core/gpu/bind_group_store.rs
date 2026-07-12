//! Persistent bind-group, layout, and shader-interface caches.

use std::num::NonZeroU64;
use std::sync::Arc;

use myth_core::utils::interner::{self, Symbol};
use myth_resources::WgslStructName;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::pipeline::vertex::VertexLayoutSignature;

use super::{ResourceIdSet, generate_gpu_resource_id};

/// Shared source text for one generated binding interface.
///
/// The wrapper preserves the existing `String`-like call sites while making
/// clones across persistent states allocation-free.
#[derive(Clone)]
pub struct SharedBindingWgsl(Arc<str>);

impl SharedBindingWgsl {
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for SharedBindingWgsl {
    fn from(value: String) -> Self {
        Self(value.into())
    }
}

impl From<&SharedBindingWgsl> for String {
    fn from(value: &SharedBindingWgsl) -> Self {
        value.as_str().to_owned()
    }
}

impl std::fmt::Display for SharedBindingWgsl {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl AsRef<str> for SharedBindingWgsl {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::ops::Deref for SharedBindingWgsl {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Stable logical key for the scene-level global binding state.
///
/// The key is lightweight snapshot data. It deliberately identifies the
/// state without exposing its physical bind group, layout, or generated WGSL
/// interface to graph construction.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalStateKey(u64);

impl GlobalStateKey {
    #[must_use]
    pub const fn new(render_state_id: u32, scene_id: u32) -> Self {
        Self((scene_id as u64) << 32 | render_state_id as u64)
    }

    #[must_use]
    pub const fn render_state_id(self) -> u32 {
        self.0 as u32
    }

    #[must_use]
    pub const fn scene_id(self) -> u32 {
        (self.0 >> 32) as u32
    }
}

/// GPU global state (Group 0)
///
/// Contains Camera Uniforms, Light Storage Buffer, Environment Maps, etc.
///
/// Uses an "Ensure -> Collect IDs -> Check Fingerprint -> Rebind" pattern
pub struct GpuGlobalState {
    pub id: u32,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    pub binding_wgsl: SharedBindingWgsl,
    pub binding_wgsl_hash: u64,
    /// Set of physical IDs of all dependent resources (used for automatic change detection)
    pub resource_ids: ResourceIdSet,
    pub last_used_frame: u64,
}

/// Exact physical recipe for one object BindGroup.
///
/// The layout ID comes from the layout cache's exact entry-vector lookup, and
/// each buffer dependency retains binding order, physical identity, offset,
/// size, binding name, and WGSL struct identity. Hash collisions therefore
/// fall through to structural equality instead of aliasing either a different
/// WGPU binding or different generated shader interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ObjectWgslStructKey {
    Generator(usize),
    Name(Symbol),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ObjectBufferBindingKey {
    physical_id: u64,
    offset: u64,
    size: Option<NonZeroU64>,
    name: Symbol,
    struct_name: Option<ObjectWgslStructKey>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ObjectBindGroupKey {
    // Model + morph uniforms are the common two-binding case and therefore
    // remain allocation-free on object-cache hits.
    buffers: SmallVec<[ObjectBufferBindingKey; 2]>,
    layout_id: u64,
}

impl ObjectBindGroupKey {
    pub(crate) fn with_capacity(binding_count: usize, layout_id: u64) -> Self {
        Self {
            buffers: SmallVec::with_capacity(binding_count),
            layout_id,
        }
    }

    pub(crate) fn push_buffer(
        &mut self,
        physical_id: u64,
        offset: u64,
        size: Option<u64>,
        name: &str,
        struct_name: Option<&WgslStructName>,
    ) {
        let struct_name = struct_name.map(|struct_name| match struct_name {
            WgslStructName::Generator(generator) => {
                ObjectWgslStructKey::Generator(*generator as usize)
            }
            WgslStructName::Name(name) => ObjectWgslStructKey::Name(interner::intern(name)),
        });
        self.buffers.push(ObjectBufferBindingKey {
            physical_id,
            offset,
            size: size.map(|size| {
                NonZeroU64::new(size).expect("object buffer binding size must be non-zero")
            }),
            name: interner::intern(name),
            struct_name,
        });
    }
}

#[derive(Clone)]
pub struct BindGroupContext {
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub binding_wgsl: Arc<str>,
    pub binding_wgsl_hash: u64,
}

struct CachedObjectBindGroup {
    context: BindGroupContext,
    last_used_frame: u64,
}

/// Narrow, immutable post-compile capability for resolving global binding
/// state from a lightweight [`GlobalStateKey`].
#[derive(Clone, Copy)]
pub struct GlobalStateResolver<'a> {
    store: &'a BindGroupStore,
    allowed_key: Option<GlobalStateKey>,
}

impl<'a> GlobalStateResolver<'a> {
    pub(crate) const fn new(
        store: &'a BindGroupStore,
        allowed_key: Option<GlobalStateKey>,
    ) -> Self {
        Self { store, allowed_key }
    }

    #[inline]
    #[must_use]
    pub fn get(self, key: GlobalStateKey) -> Option<&'a GpuGlobalState> {
        (self.allowed_key == Some(key))
            .then(|| self.store.global_state(key))
            .flatten()
    }
}

/// Process-lifetime Group 0 shader interface, created by the first live
/// post-DPE global-state materialization.
#[derive(Clone)]
pub(super) struct GlobalBindingInterface {
    pub(super) layout: wgpu::BindGroupLayout,
    pub(super) layout_id: u64,
    pub(super) binding_wgsl: SharedBindingWgsl,
    pub(super) binding_wgsl_hash: u64,
}

/// Persistent bind-group, layout, and shader-interface caches.
#[derive(Default)]
pub(crate) struct BindGroupStore {
    global_states: FxHashMap<GlobalStateKey, GpuGlobalState>,
    global_interface: Option<GlobalBindingInterface>,
    layout_cache: FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,
    vertex_layout_cache: FxHashMap<VertexLayoutSignature, u64>,
    object_bind_group_cache: FxHashMap<ObjectBindGroupKey, CachedObjectBindGroup>,
    bind_group_id_lookup: FxHashMap<u64, BindGroupContext>,
}

impl BindGroupStore {
    #[inline]
    #[must_use]
    pub(super) fn global_interface(&self) -> Option<&GlobalBindingInterface> {
        self.global_interface.as_ref()
    }

    pub(super) fn insert_global_interface(&mut self, interface: GlobalBindingInterface) {
        assert!(
            self.global_interface.replace(interface).is_none(),
            "global binding interface initialized twice"
        );
    }

    pub(super) fn prune(&mut self, cutoff: u64) {
        self.global_states
            .retain(|_, v| v.last_used_frame >= cutoff);
        let mut stale_object_ids = Vec::new();
        self.object_bind_group_cache.retain(|_, entry| {
            let retain = entry.last_used_frame >= cutoff;
            if !retain {
                stale_object_ids.push(entry.context.bind_group_id);
            }
            retain
        });
        for bind_group_id in stale_object_ids {
            self.bind_group_id_lookup.remove(&bind_group_id);
        }
    }

    pub(super) fn remove_scene_global_states(&mut self, scene_id: u32) -> usize {
        let before = self.global_states.len();
        self.global_states
            .retain(|key, _| key.scene_id() != scene_id);
        before - self.global_states.len()
    }

    #[inline]
    pub(super) fn clear_object_bind_group_caches(&mut self) {
        self.object_bind_group_cache.clear();
        self.bind_group_id_lookup.clear();
    }

    #[inline]
    #[must_use]
    pub(super) fn cached_object_bind_group(
        &mut self,
        key: &ObjectBindGroupKey,
        frame_index: u64,
    ) -> Option<BindGroupContext> {
        let entry = self.object_bind_group_cache.get_mut(key)?;
        entry.last_used_frame = frame_index;
        Some(entry.context.clone())
    }

    #[inline]
    pub(super) fn cache_object_bind_group(
        &mut self,
        key: ObjectBindGroupKey,
        bind_group_id: u64,
        context: BindGroupContext,
        frame_index: u64,
    ) {
        self.object_bind_group_cache.insert(
            key,
            CachedObjectBindGroup {
                context: context.clone(),
                last_used_frame: frame_index,
            },
        );
        self.bind_group_id_lookup.insert(bind_group_id, context);
    }

    #[inline]
    #[must_use]
    pub(super) fn cached_bind_group(&self, bind_group_id: u64) -> Option<&BindGroupContext> {
        self.bind_group_id_lookup.get(&bind_group_id)
    }

    #[cfg(test)]
    pub(super) fn cached_object_bind_group_ids(&self) -> Vec<u64> {
        self.bind_group_id_lookup.keys().copied().collect()
    }

    pub(super) fn get_or_create_layout(
        &mut self,
        device: &wgpu::Device,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> (wgpu::BindGroupLayout, u64) {
        if let Some(layout) = self.layout_cache.get(entries) {
            return layout.clone();
        }

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cached BindGroupLayout"),
            entries,
        });

        let id = generate_gpu_resource_id();
        self.layout_cache
            .insert(entries.to_vec(), (layout.clone(), id));
        (layout, id)
    }

    #[inline]
    #[must_use]
    pub(super) fn cached_layout_id(&self, entries: &[wgpu::BindGroupLayoutEntry]) -> Option<u64> {
        self.layout_cache.get(entries).map(|(_, id)| *id)
    }

    pub(super) fn get_or_create_vertex_layout_id(
        &mut self,
        signature: VertexLayoutSignature,
    ) -> u64 {
        if let Some(&id) = self.vertex_layout_cache.get(&signature) {
            return id;
        }

        let id = generate_gpu_resource_id();
        self.vertex_layout_cache.insert(signature, id);
        id
    }

    #[inline]
    #[must_use]
    pub(super) fn global_state(&self, state_id: GlobalStateKey) -> Option<&GpuGlobalState> {
        self.global_states.get(&state_id)
    }

    #[inline]
    pub(super) fn global_state_mut(
        &mut self,
        state_id: GlobalStateKey,
    ) -> Option<&mut GpuGlobalState> {
        self.global_states.get_mut(&state_id)
    }

    #[inline]
    pub(super) fn insert_global_state(
        &mut self,
        state_id: GlobalStateKey,
        gpu_state: GpuGlobalState,
    ) {
        self.global_states.insert(state_id, gpu_state);
    }
}

#[cfg(test)]
mod tests {
    use super::{GlobalStateKey, ObjectBindGroupKey};
    use myth_resources::WgslStructName;

    fn object_key(bindings: &[(u64, u64, Option<u64>)], layout_id: u64) -> ObjectBindGroupKey {
        let mut key = ObjectBindGroupKey::with_capacity(bindings.len(), layout_id);
        for (index, &(physical_id, offset, size)) in bindings.iter().enumerate() {
            let name = match index {
                0 => "model",
                1 => "morph_targets",
                _ => "extra",
            };
            key.push_buffer(physical_id, offset, size, name, None);
        }
        key
    }

    #[test]
    fn global_state_key_round_trips_both_ids() {
        let key = GlobalStateKey::new(0x89ab_cdef, 0x0123_4567);

        assert_eq!(key.render_state_id(), 0x89ab_cdef);
        assert_eq!(key.scene_id(), 0x0123_4567);
    }

    #[test]
    fn object_bind_group_key_tracks_layout_and_binding_order() {
        let baseline = object_key(&[(11, 0, None), (22, 0, Some(64)), (33, 16, None)], 101);

        assert_eq!(
            baseline,
            object_key(&[(11, 0, None), (22, 0, Some(64)), (33, 16, None)], 101)
        );
        assert_ne!(
            baseline,
            object_key(&[(11, 0, None), (22, 0, Some(64)), (33, 16, None)], 202),
            "a different object layout must not reuse the cached BindGroup"
        );
        assert_ne!(
            baseline,
            object_key(&[(22, 0, Some(64)), (11, 0, None), (33, 16, None)], 101),
            "physical resource IDs are positional BindGroup dependencies"
        );
        assert_ne!(
            baseline,
            object_key(&[(11, 0, None), (22, 0, Some(64)), (44, 16, None)], 101),
            "recreating one bound buffer must invalidate the object BindGroup"
        );
        assert_ne!(
            baseline,
            object_key(&[(11, 0, None), (22, 32, Some(64)), (33, 16, None)], 101),
            "binding offsets are part of the physical recipe"
        );
        assert_ne!(
            baseline,
            object_key(&[(11, 0, None), (22, 0, Some(128)), (33, 16, None)], 101),
            "binding sizes are part of the physical recipe"
        );

        let mut different_interface = ObjectBindGroupKey::with_capacity(3, baseline.layout_id);
        different_interface.push_buffer(11, 0, None, "model", None);
        different_interface.push_buffer(22, 0, Some(64), "morph_targets", None);
        different_interface.push_buffer(33, 16, None, "different_semantic", None);
        assert_ne!(
            baseline, different_interface,
            "WGSL binding names are part of the cached context contract"
        );

        let mut different_struct = ObjectBindGroupKey::with_capacity(3, baseline.layout_id);
        different_struct.push_buffer(11, 0, None, "model", None);
        different_struct.push_buffer(22, 0, Some(64), "morph_targets", None);
        let named_struct = WgslStructName::Name("DifferentStruct".to_owned());
        different_struct.push_buffer(33, 16, None, "extra", Some(&named_struct));
        assert_ne!(
            baseline, different_struct,
            "WGSL struct identity is part of the cached context contract"
        );
    }
}
