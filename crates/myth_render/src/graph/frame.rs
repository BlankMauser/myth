//! Render frame management.
//!
//! `RenderFrame` owns the frame's CPU scene snapshot, render state, light/view
//! metadata, and the small set of scene-global inputs that must exist before
//! graph construction. Feature-local physical preparation is deferred until
//! after DPE by [`FrameComposer`].
//!
//! # Three-Phase Rendering Architecture
//!
//! The rendering pipeline is divided into three distinct phases:
//!
//! 1. **Snapshot**: `extract_and_prepare()` captures scene/light/view state and
//!    refreshes the shared global interface.
//! 2. **Compose + Compile**: [`FrameComposer`] collects CPU draw intents,
//!    declares topology, and runs DPE.
//! 3. **Deferred prepare**: only live passes and their exact draw-source union
//!    materialize physical resources and pipelines.
//! 4. **Execute + Commit**: blind commands are recorded; temporal/producer
//!    state is published only after submission.
//!

use glam::Mat4;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::gpu::Tracked;
use crate::core::{RenderView, ResourceManager};
use crate::pipeline::RenderPipelineId;
use crate::pipeline::vertex::GeneratedVertexLayout;
use crate::renderer::FrameTime;
use crate::settings::{ClusteredShadingMode, RenderPath};
use myth_assets::{AssetServer, GeometryHandle, MaterialHandle};
use myth_resources::RenderableMaterialTrait;
use myth_resources::geometry::Geometry;
use myth_resources::material::Material;
use myth_scene::Scene;
use myth_scene::camera::RenderCamera;

use super::core::TextureNodeId;
use super::extracted::{ExtractedRenderItem, ExtractedScene, SceneFeatures};
use super::render_state::RenderState;
use super::shadow_utils;

// ============================================================================
// Render intents, physical commands, and render lists
// ============================================================================

/// CPU-only main-view draw intent.
///
/// This is produced before RDG compilation and contains no WGPU owner or
/// pipeline ID. `source_item_index` is the exact identity used to join the
/// intent with graph-attempt physical preparation after DPE.
#[derive(Debug, Clone, Copy)]
pub struct RenderIntent {
    pub source_item_index: u32,
    pub geometry_handle: GeometryHandle,
    pub material_handle: MaterialHandle,
    pub material_index: u32,
    pub distance_sq: f32,
}

/// CPU-only per-shadow-view draw intent.
#[derive(Debug, Clone, Copy)]
pub struct ShadowRenderIntent {
    pub source_item_index: u32,
    pub geometry_handle: GeometryHandle,
    pub material_handle: MaterialHandle,
}

/// Frame-scoped feature inputs that affect scene shader variants.
#[derive(Debug, Clone, Copy)]
pub struct RenderFrameFeatureConfig {
    pub clustered_shading: ClusteredShadingMode,
    pub render_path: RenderPath,
    pub msaa_samples: u32,
}

// ============================================================================
// DrawCommand — Baked Physical Draw Command
// ============================================================================

/// Frame-local slot for one exact physical buffer owned by [`DrawPlanStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DrawBufferSlot(u32);

/// Frame-local slot for one exact physical bind group owned by
/// [`DrawPlanStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DrawBindGroupSlot(u32);

/// Copy handle to one contiguous command range in a [`DrawPlanStore`].
///
/// The attempt token prevents an ID retained by a future pass-node API from
/// aliasing a plan at the same vector index in a later graph attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DrawPlanId {
    attempt_token: u64,
    index: u32,
}

/// Canonical frame-local recipe for one draw-plan command list.
///
/// Recipes contain only graph-build-time POD data. Repeated reservation of the
/// same key returns the same [`DrawPlanId`], allowing multiple passes to share
/// one finalized command range without borrowing physical WGPU resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DrawPlanKey {
    Opaque,
    Transparent,
    Prepass {
        needs_normal: bool,
        needs_feature_id: bool,
        needs_velocity: bool,
    },
    Shadow {
        light_id: u64,
        layer_index: u32,
    },
}

struct DrawPlanRecord {
    key: DrawPlanKey,
    commands: Option<std::ops::Range<u32>>,
    live: bool,
    topology_demanded: bool,
}

/// Copy metadata used by the post-compile draw-plan finalizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DrawPlanMeta {
    pub(crate) id: DrawPlanId,
    pub(crate) key: DrawPlanKey,
    pub(crate) is_live: bool,
    pub(crate) is_finalized: bool,
}

/// A single pre-baked draw command with all GPU resources resolved to
/// frame-local physical-resource slots.
///
/// Produced by [`bake::finalize_live_draw_plans`] after DPE, consumed by the
/// unified [`submit_draw_commands`] helper during the execute phase.  Contains
/// **zero** asset handles. Each unique WGPU buffer or bind group is cloned
/// exactly once into the graph-attempt-owned [`DrawPlanStore`]; commands carry
/// compact slots plus a stable [`RenderPipelineId`]. The execute phase never
/// looks resources up through asset stores and cannot outlive the physical
/// resources selected during baking.
///
/// # Design Philosophy — "The Blind Execute Phase"
///
/// All semantic decisions (pipeline variant selection, alpha-test detection,
/// material/geometry resolution) are completed during the earlier **cull /
/// finalize** phase. The execute phase sees only hardware-level GPU state and
/// can be trivially parallelised in the future.
///
/// [`bake::finalize_live_draw_plans`]: super::bake::finalize_live_draw_plans
/// [`submit_draw_commands`]: super::passes::draw::submit_draw_commands
pub struct DrawCommand {
    /// Exact extracted source used only by the post-submit history commit.
    /// Execute remains blind to this logical identity.
    pub(crate) source_item_index: u32,

    /// Composite sort key for minimising GPU state switches.
    ///
    /// * Opaque — Pipeline › Material › Depth (front-to-back).
    /// * Transparent — Depth (back-to-front) › Pipeline › Material.
    /// * Shadow — Pipeline (state-switch minimisation).
    pub sort_key: u64,

    /// Stable handle for the pre-selected render pipeline.
    pub pipeline_id: RenderPipelineId,

    /// Pre-resolved vertex buffer bindings.
    ///
    /// Typically 1–2 entries, interned by stable physical resource ID.
    pub(crate) vertex_buffers: SmallVec<[DrawBufferSlot; 2]>,

    /// Pre-resolved index buffer, or `None` for non-indexed draws.
    ///
    /// Tuple: `(buffer, format, index_count)`.
    pub(crate) index_buffer: Option<(DrawBufferSlot, wgpu::IndexFormat, u32)>,

    /// Material bind group (Group 1), or `None` when unused.
    pub(crate) bind_group_1: Option<DrawBindGroupSlot>,

    /// Object / transform bind group (Group 2) with dynamic uniform offset.
    ///
    /// Tuple: `(bind_group, dynamic_offset)`.
    pub(crate) bind_group_2: (DrawBindGroupSlot, u32),

    /// Stencil reference value for feature-ID writing.  `None` when unused.
    pub stencil_reference: Option<u32>,

    /// Vertex range for non-indexed draws.
    pub vertex_range: std::ops::Range<u32>,

    /// Instance range.
    pub instance_range: std::ops::Range<u32>,
}

/// Exact graph-attempt physical state for one extracted draw source.
///
/// Buffer and bind-group slots are owned by the sibling [`DrawPlanStore`].
/// This record retains only compact slots plus the cloned layout/interface
/// metadata required to derive live pipelines after DPE. It deliberately owns
/// no vertex buffer or material/object BindGroup directly.
pub(crate) struct PreparedDrawSource {
    /// Exact CPU assets selected for this graph attempt. These Arc owners close
    /// the same-handle/same-version ABA gap during deferred pipeline creation.
    pub(crate) geometry_asset: Arc<Geometry>,
    pub(crate) material_asset: Arc<Material>,
    pub(crate) geometry_asset_version: u32,
    pub(crate) geometry_asset_identity: u64,
    pub(crate) material_asset_version: u32,
    pub(crate) material_asset_identity: u64,

    pub(crate) vertex_buffers: SmallVec<[DrawBufferSlot; 2]>,
    pub(crate) index_buffer: Option<(DrawBufferSlot, wgpu::IndexFormat, u32)>,
    pub(crate) material_bind_group: DrawBindGroupSlot,
    pub(crate) material_bind_group_id: u64,
    pub(crate) object_bind_group: (DrawBindGroupSlot, u32),
    pub(crate) draw_range: Range<u32>,
    pub(crate) instance_range: Range<u32>,

    /// Shared immutable vertex interface. Sources that reuse one geometry
    /// clone only this Arc rather than reallocating the generated layout.
    pub(crate) geometry_layout_info: Arc<GeneratedVertexLayout>,
    pub(crate) geometry_layout_id: u64,
    pub(crate) geometry_version: u64,
    pub(crate) geometry_layout_version: u64,
    pub(crate) material_layout: wgpu::BindGroupLayout,
    pub(crate) material_layout_id: u64,
    pub(crate) material_binding_wgsl: Arc<str>,
    pub(crate) material_binding_wgsl_hash: u64,
    pub(crate) material_version: u64,
    pub(crate) object_layout: wgpu::BindGroupLayout,
    pub(crate) object_layout_id: u64,
    pub(crate) object_binding_wgsl: Arc<str>,
    pub(crate) object_binding_wgsl_hash: u64,
}

impl PreparedDrawSource {
    #[inline]
    pub(crate) fn matches_extracted(
        &self,
        item: &ExtractedRenderItem,
        geometry_handle: GeometryHandle,
        material_handle: MaterialHandle,
    ) -> bool {
        (item.geometry, item.material) == (geometry_handle, material_handle)
            && (item.geometry_asset_version, item.geometry_asset_identity)
                == (self.geometry_asset_version, self.geometry_asset_identity)
            && (item.material_asset_version, item.material_asset_identity)
                == (self.material_asset_version, self.material_asset_identity)
    }

    /// Validate shader/raster-relevant interior versions around deferred
    /// derivation. Exact Arc identity closes storage ABA; these counters close
    /// concurrent in-place mutation of the same allocation.
    #[inline]
    pub(crate) fn pipeline_state_is_current(&self) -> bool {
        self.geometry_asset.structure_version() == self.geometry_version
            && self.geometry_asset.layout_version() == self.geometry_layout_version
            && self.material_asset.data.version() == self.material_version
    }
}

/// Graph-attempt-scoped source preparation indexed by `source_item_index`.
///
/// The parallel `Option` array preserves exact extracted-item identity while
/// allowing unavailable or DPE-dead sources to remain unprepared. All physical
/// Buffer/BindGroup owners live in [`DrawPlanStore`], so clearing both stores
/// deterministically releases one graph attempt.
pub(crate) struct PreparedDrawStore {
    sources: Vec<Option<PreparedDrawSource>>,
    sources_initialized: bool,
}

impl PreparedDrawStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            sources: Vec::with_capacity(512),
            sources_initialized: false,
        }
    }

    /// Release graph-attempt metadata while retaining the parallel array's
    /// allocation for the next frame.
    pub(crate) fn clear(&mut self) {
        self.sources.clear();
        self.sources_initialized = false;
    }

    /// Initialize the exact parallel source table once for this graph attempt.
    pub(crate) fn begin_sources(&mut self, source_count: usize) {
        assert!(
            !self.sources_initialized,
            "prepared draw sources may be initialized only once per graph attempt"
        );
        debug_assert!(self.sources.is_empty());
        self.sources.resize_with(source_count, || None);
        self.sources_initialized = true;
    }

    /// Publish one exact physical source after its resources have been interned
    /// into the sibling [`DrawPlanStore`].
    pub(crate) fn insert_source(&mut self, source_item_index: u32, source: PreparedDrawSource) {
        assert!(
            self.sources_initialized,
            "prepared draw source table must be initialized before insertion"
        );
        let index = source_item_index as usize;
        let slot = self
            .sources
            .get_mut(index)
            .expect("prepared draw source index exceeds the extracted item table");
        assert!(slot.is_none(), "prepared draw source was inserted twice");
        *slot = Some(source);
    }

    #[inline]
    #[must_use]
    pub(crate) fn source(&self, source_item_index: u32) -> Option<&PreparedDrawSource> {
        self.sources
            .get(source_item_index as usize)
            .and_then(Option::as_ref)
    }
}

impl Default for PreparedDrawStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph-attempt-scoped pre-baked draw plans.
///
/// All GPU handle lookups have been resolved to owned physical WGPU clones.
/// [`GraphStorage`](super::core::GraphStorage) owns this store and clears it
/// on both successful completion and abandoned graph attempts.
pub struct DrawPlanStore {
    commands: Vec<DrawCommand>,
    plans: Vec<DrawPlanRecord>,
    plan_lookup: FxHashMap<DrawPlanKey, DrawPlanId>,
    texture_dependencies: FxHashMap<DrawPlanKey, SmallVec<[TextureNodeId; 8]>>,
    open_plan: Option<(DrawPlanId, u32)>,
    buffers: Vec<Tracked<wgpu::Buffer>>,
    buffer_lookup: FxHashMap<u64, DrawBufferSlot>,
    bind_groups: Vec<Tracked<wgpu::BindGroup>>,
    bind_group_lookup: FxHashMap<u64, DrawBindGroupSlot>,
    attempt_token: u64,
}

impl Default for DrawPlanStore {
    fn default() -> Self {
        Self::new()
    }
}

impl DrawPlanStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            commands: Vec::with_capacity(1024),
            plans: Vec::with_capacity(16),
            plan_lookup: FxHashMap::default(),
            texture_dependencies: FxHashMap::default(),
            open_plan: None,
            buffers: Vec::with_capacity(256),
            buffer_lookup: FxHashMap::default(),
            bind_groups: Vec::with_capacity(256),
            bind_group_lookup: FxHashMap::default(),
            attempt_token: next_draw_plan_attempt(),
        }
    }

    /// Releases all graph-attempt WGPU owners while retaining store capacity.
    pub(crate) fn clear(&mut self) {
        self.commands.clear();
        self.plans.clear();
        self.plan_lookup.clear();
        self.texture_dependencies.clear();
        self.open_plan = None;
        self.buffer_lookup.clear();
        self.buffers.clear();
        self.bind_group_lookup.clear();
        self.bind_groups.clear();
        self.attempt_token = next_draw_plan_attempt();
    }

    pub(crate) fn reserve_commands(&mut self, additional: usize) {
        self.commands.reserve(additional);
    }

    /// Reserve a stable plan ID during graph construction.
    ///
    /// Reservation is idempotent for one canonical key within the graph
    /// attempt. It performs no physical-resource lookup or WGPU cloning.
    pub(crate) fn reserve_plan(&mut self, key: DrawPlanKey) -> DrawPlanId {
        if let Some(&id) = self.plan_lookup.get(&key) {
            debug_assert!(self.validate_reserved(id));
            return id;
        }

        let index = u32::try_from(self.plans.len())
            .expect("one graph attempt cannot reserve more than u32::MAX draw plans");
        let id = DrawPlanId {
            attempt_token: self.attempt_token,
            index,
        };
        self.plans.push(DrawPlanRecord {
            key,
            commands: None,
            live: false,
            topology_demanded: false,
        });
        let previous = self.plan_lookup.insert(key, id);
        debug_assert!(previous.is_none());
        id
    }

    /// Register the Asset Texture aliases sampled by one canonical plan.
    ///
    /// Prepass variants share the opaque material set, so their dependency
    /// key is normalized to [`DrawPlanKey::Opaque`]. Registration must happen
    /// before any pass demands that plan; otherwise its real RDG reads would
    /// be incomplete and producer ordering could not be recovered later.
    pub(crate) fn assert_texture_dependencies_registerable(&self, key: DrawPlanKey) {
        let key = draw_texture_dependency_key(key);
        assert!(
            self.plans.iter().all(|plan| {
                !plan.topology_demanded || draw_texture_dependency_key(plan.key) != key
            }),
            "draw-plan texture dependencies must be registered before the first consumer"
        );
    }

    pub(crate) fn set_texture_dependencies(
        &mut self,
        key: DrawPlanKey,
        dependencies: SmallVec<[TextureNodeId; 8]>,
    ) {
        let key = draw_texture_dependency_key(key);
        match self.texture_dependencies.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(dependencies);
            }
            Entry::Occupied(_) => {
                panic!("draw-plan texture dependencies registered twice for {key:?}");
            }
        }
    }

    /// Marks one canonical plan as topology-demanded and returns its normalized
    /// texture-dependency key. The caller may then borrow the registered slice
    /// without cloning it.
    pub(crate) fn mark_topology_demanded(&mut self, id: DrawPlanId) -> DrawPlanKey {
        let plan = self
            .plan_mut(id)
            .expect("draw plan demand does not belong to this graph attempt");
        plan.topology_demanded = true;
        draw_texture_dependency_key(plan.key)
    }

    /// Borrow the logical texture dependencies registered for a normalized
    /// draw-plan key.
    pub(crate) fn texture_dependencies(&self, normalized_key: DrawPlanKey) -> &[TextureNodeId] {
        self.texture_dependencies
            .get(&normalized_key)
            .map_or(&[], SmallVec::as_slice)
    }

    /// Return whether an ID addresses a reserved plan in this graph attempt.
    #[must_use]
    pub(crate) fn validate_reserved(&self, id: DrawPlanId) -> bool {
        id.attempt_token == self.attempt_token && (id.index as usize) < self.plans.len()
    }

    /// Return the dense attempt-local slot addressed by a valid plan ID.
    #[must_use]
    pub(crate) fn reserved_index(&self, id: DrawPlanId) -> Option<usize> {
        self.validate_reserved(id).then_some(id.index as usize)
    }

    /// Reset compile-time demand marks before collecting the current DPE
    /// result. Reservations and already-owned resources are left intact.
    pub(crate) fn reset_live_demands(&mut self) {
        for plan in &mut self.plans {
            plan.live = false;
        }
    }

    /// Mark a reserved plan as demanded by at least one live pass.
    pub(crate) fn mark_live(&mut self, id: DrawPlanId) {
        let plan = self
            .plan_mut(id)
            .expect("draw plan demand does not belong to this graph attempt");
        plan.live = true;
    }

    #[must_use]
    pub(crate) fn is_live(&self, id: DrawPlanId) -> bool {
        self.plan(id)
            .expect("draw plan ID does not belong to this graph attempt")
            .live
    }

    /// Iterate canonical keys demanded by passes that survived DPE.
    ///
    /// The iterator copies POD keys only; it does not expose mutable plan state
    /// or make source metadata participate in graph liveness.
    pub(crate) fn live_keys(&self) -> impl Iterator<Item = DrawPlanKey> + '_ {
        self.plans
            .iter()
            .filter(|plan| plan.live)
            .map(|plan| plan.key)
    }

    /// Return Copy metadata for a reservation-table slot.
    ///
    /// Iterating by index and copying this value lets the finalizer mutate the
    /// store without retaining an immutable borrow of the plan table.
    #[must_use]
    pub(crate) fn plan_meta(&self, index: usize) -> Option<DrawPlanMeta> {
        let plan = self.plans.get(index)?;
        Some(DrawPlanMeta {
            id: DrawPlanId {
                attempt_token: self.attempt_token,
                index: u32::try_from(index).ok()?,
            },
            key: plan.key,
            is_live: plan.live,
            is_finalized: plan.commands.is_some(),
        })
    }

    #[must_use]
    pub(crate) fn plan_count(&self) -> usize {
        self.plans.len()
    }

    /// Verify the prepare/finalize barrier before command execution.
    ///
    /// This catches a newly added draw-plan key whose live-node handler forgot
    /// to materialize its range; an empty range is fully valid.
    pub(crate) fn assert_all_live_finalized(&self) {
        assert!(
            self.open_plan.is_none(),
            "draw-plan finalization remained open at the execute barrier"
        );
        for plan in &self.plans {
            assert!(
                !plan.live || plan.commands.is_some(),
                "live draw plan {:?} was not finalized before execute",
                plan.key
            );
        }
    }

    /// Begin materializing one live reserved plan into the shared command
    /// array. Only one plan may be open because command ranges are contiguous.
    pub(crate) fn begin_finalize(&mut self, id: DrawPlanId) -> u32 {
        let plan = self
            .plan(id)
            .expect("draw plan ID does not belong to this graph attempt");
        assert!(
            plan.live,
            "cannot finalize a draw plan with no live consumer"
        );
        assert!(
            plan.commands.is_none(),
            "draw plan was finalized more than once"
        );
        assert!(
            self.open_plan.is_none(),
            "draw plan finalization cannot be interleaved"
        );

        let start = u32::try_from(self.commands.len())
            .expect("one graph attempt cannot bake more than u32::MAX draw commands");
        self.open_plan = Some((id, start));
        start
    }

    pub(crate) fn push_command(&mut self, command: DrawCommand) {
        assert!(
            self.open_plan.is_some(),
            "draw commands must be appended inside plan finalization"
        );
        self.commands.push(command);
    }

    pub(crate) fn sort_commands_from(&mut self, start: u32) {
        assert!(
            self.open_plan
                .is_some_and(|(_, open_start)| open_start == start),
            "only the currently open draw plan may be sorted"
        );
        self.commands[start as usize..].sort_unstable_by_key(|command| command.sort_key);
    }

    /// Finish one reserved plan. `start == end` is a valid demanded empty
    /// plan and stays distinguishable from an unfinalized dead plan.
    pub(crate) fn finish_finalize(&mut self, id: DrawPlanId, start: u32) {
        let end = u32::try_from(self.commands.len())
            .expect("one graph attempt cannot bake more than u32::MAX draw commands");
        assert_eq!(
            self.open_plan.take(),
            Some((id, start)),
            "draw plan finish does not match the currently open plan"
        );
        let plan = self
            .plan_mut(id)
            .expect("draw plan ID does not belong to this graph attempt");
        assert!(plan.live, "cannot finish a draw plan with no live consumer");
        assert!(
            plan.commands.is_none(),
            "draw plan was finalized more than once"
        );
        plan.commands = Some(start..end);
    }

    #[inline]
    pub(crate) fn intern_buffer(&mut self, buffer: &wgpu::Buffer, id: u64) -> DrawBufferSlot {
        match self.buffer_lookup.entry(id) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let slot = DrawBufferSlot(
                    u32::try_from(self.buffers.len())
                        .expect("one graph attempt cannot intern more than u32::MAX draw buffers"),
                );
                self.buffers.push(Tracked::with_id(buffer.clone(), id));
                entry.insert(slot);
                slot
            }
        }
    }

    #[inline]
    pub(crate) fn intern_bind_group(
        &mut self,
        bind_group: &wgpu::BindGroup,
        id: u64,
    ) -> DrawBindGroupSlot {
        match self.bind_group_lookup.entry(id) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let slot =
                    DrawBindGroupSlot(u32::try_from(self.bind_groups.len()).expect(
                        "one graph attempt cannot intern more than u32::MAX draw bind groups",
                    ));
                self.bind_groups
                    .push(Tracked::with_id(bind_group.clone(), id));
                entry.insert(slot);
                slot
            }
        }
    }

    pub(crate) fn resolve_plan(&self, id: DrawPlanId) -> Option<&[DrawCommand]> {
        let range = self.plan(id)?.commands.as_ref()?;
        Some(&self.commands[range.start as usize..range.end as usize])
    }

    /// Source identities of commands that were actually emitted into at least
    /// one live plan. The caller deduplicates shared main/prepass/shadow uses.
    pub(crate) fn emitted_source_indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.commands
            .iter()
            .map(|command| command.source_item_index)
    }

    fn plan(&self, id: DrawPlanId) -> Option<&DrawPlanRecord> {
        if id.attempt_token != self.attempt_token {
            return None;
        }
        self.plans.get(id.index as usize)
    }

    fn plan_mut(&mut self, id: DrawPlanId) -> Option<&mut DrawPlanRecord> {
        if id.attempt_token != self.attempt_token {
            return None;
        }
        self.plans.get_mut(id.index as usize)
    }

    fn resolve_buffer(&self, slot: DrawBufferSlot) -> Option<&Tracked<wgpu::Buffer>> {
        self.buffers.get(slot.0 as usize)
    }

    fn resolve_bind_group(&self, slot: DrawBindGroupSlot) -> Option<&Tracked<wgpu::BindGroup>> {
        self.bind_groups.get(slot.0 as usize)
    }

    #[must_use]
    pub(crate) const fn resolver(&self) -> DrawPlanResolver<'_> {
        DrawPlanResolver { store: self }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.commands.is_empty()
            && self.plans.is_empty()
            && self.plan_lookup.is_empty()
            && self.texture_dependencies.is_empty()
            && self.open_plan.is_none()
            && self.buffers.is_empty()
            && self.buffer_lookup.is_empty()
            && self.bind_groups.is_empty()
            && self.bind_group_lookup.is_empty()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn buffer_owner_count(&self) -> usize {
        self.buffers.len()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn bind_group_owner_count(&self) -> usize {
        self.bind_groups.len()
    }
}

#[inline]
fn draw_texture_dependency_key(key: DrawPlanKey) -> DrawPlanKey {
    match key {
        DrawPlanKey::Prepass { .. } => DrawPlanKey::Opaque,
        key => key,
    }
}

/// Narrow immutable execute-time view over one graph attempt's draw plans.
#[derive(Clone, Copy)]
pub struct DrawPlanResolver<'a> {
    store: &'a DrawPlanStore,
}

impl<'a> DrawPlanResolver<'a> {
    #[must_use]
    #[inline]
    pub(crate) fn commands(self, id: DrawPlanId) -> &'a [DrawCommand] {
        self.store
            .resolve_plan(id)
            .expect("draw plan ID does not belong to this graph attempt")
    }

    #[must_use]
    pub(crate) fn is_empty(self, id: DrawPlanId) -> bool {
        self.commands(id).is_empty()
    }

    #[must_use]
    #[inline]
    pub(crate) fn buffer(self, slot: DrawBufferSlot) -> &'a Tracked<wgpu::Buffer> {
        self.store
            .resolve_buffer(slot)
            .expect("draw buffer slot does not belong to this graph attempt")
    }

    #[must_use]
    #[inline]
    pub(crate) fn bind_group(self, slot: DrawBindGroupSlot) -> &'a Tracked<wgpu::BindGroup> {
        self.store
            .resolve_bind_group(slot)
            .expect("draw bind-group slot does not belong to this graph attempt")
    }
}

/// Attempt-scoped owner for the scene-global bind group used by render passes.
///
/// This is intentionally separate from [`DrawPlanStore`]: the global binding
/// is pass-level state, not a draw-plan resource slot.
#[derive(Default)]
pub(crate) struct PreparedGlobalBindGroup {
    bind_group: Option<wgpu::BindGroup>,
}

impl PreparedGlobalBindGroup {
    pub(crate) fn stage(&mut self, bind_group: &wgpu::BindGroup) {
        self.bind_group = Some(bind_group.clone());
    }

    #[must_use]
    pub(crate) fn resolve(&self) -> &wgpu::BindGroup {
        self.bind_group
            .as_ref()
            .expect("global bind group was not staged for this graph attempt")
    }

    pub(crate) fn clear(&mut self) {
        self.bind_group = None;
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.bind_group.is_none()
    }
}

static NEXT_DRAW_PLAN_ATTEMPT: AtomicU64 = AtomicU64::new(1);

fn next_draw_plan_attempt() -> u64 {
    NEXT_DRAW_PLAN_ATTEMPT
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("draw-plan attempt token space exhausted")
}

#[derive(Clone, Copy)]
pub struct ShadowLightInstance {
    pub light_id: u64,
    /// Original layer index from the `RenderView`, used as the shadow queue key.
    pub view_layer_index: u32,
    /// Layer index within the target texture (2D array or cube array).
    pub texture_layer_index: u32,
    pub light_buffer_index: usize,
    pub light_view_projection: Mat4,
    /// `true` if this layer belongs to a point light (cube array target).
    pub is_point: bool,
}

/// CPU-only render intent lists plus shared frame metadata.
///
/// Intent lists are populated before RDG compilation. After DPE, live
/// [`DrawPlanKey`] values select an exact, deduplicated source-item union for
/// physical preparation. Physical slots and finalized commands belong to the
/// graph-attempt [`PreparedDrawStore`] and [`DrawPlanStore`].
///
/// # Design Principles
/// - **Data-logic separation**: stores data only, contains no rendering logic
/// - **Inter-frame reuse**: pre-allocated memory, cleared via `clear()` each frame
/// - **Extensible**: can add `alpha_test`, `shadow_casters`, etc. in the future
pub struct RenderLists {
    /// Main-view opaque intents.
    pub opaque: Vec<RenderIntent>,
    /// Main-view transparent intents.
    pub transparent: Vec<RenderIntent>,
    /// Shadow intent queues, keyed by `(light_id, layer_index)` for exact
    /// per-view demand projection.
    ///
    /// Each cascade of a directional light (or each spot light) gets its own queue.
    pub shadow_queues: FxHashMap<(u64, u32), Vec<ShadowRenderIntent>>,

    /// All active render views for the current frame.
    ///
    /// Populated during CPU extraction, consumed by intent collection and
    /// shadow graph construction.
    /// Contains main camera view + all shadow views.
    pub active_views: Vec<RenderView>,

    /// Whether a transmission copy is needed this frame
    pub use_transmission: bool,
}

impl RenderLists {
    /// Creates empty render lists with pre-allocated default capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            opaque: Vec::with_capacity(512),
            transparent: Vec::with_capacity(128),
            shadow_queues: FxHashMap::default(),
            active_views: Vec::with_capacity(16),
            use_transmission: false,
        }
    }

    /// Clears all lists (retains capacity for memory reuse).
    #[inline]
    pub fn clear(&mut self) {
        self.opaque.clear();
        self.transparent.clear();
        self.shadow_queues.clear();
        self.active_views.clear();
        self.use_transmission = false;
    }

    /// Inserts an opaque render intent.
    #[inline]
    pub fn insert_opaque(&mut self, intent: RenderIntent) {
        self.opaque.push(intent);
    }

    /// Inserts a transparent render intent.
    #[inline]
    pub fn insert_transparent(&mut self, intent: RenderIntent) {
        self.transparent.push(intent);
    }

    /// Return the exact sorted and deduplicated source-item union selected by
    /// DPE-live draw-plan keys.
    #[must_use]
    pub(crate) fn live_source_indices(
        &self,
        live_keys: impl IntoIterator<Item = DrawPlanKey>,
    ) -> Vec<u32> {
        let mut sources = Vec::new();
        for key in live_keys {
            self.extend_sources_for_key(key, &mut sources);
        }
        sort_and_dedup_sources(&mut sources);
        sources
    }

    /// Return every intent's source index, independent of DPE liveness.
    #[must_use]
    pub(crate) fn all_source_indices(&self) -> Vec<u32> {
        let mut sources = Vec::with_capacity(
            self.opaque.len()
                + self.transparent.len()
                + self.shadow_queues.values().map(Vec::len).sum::<usize>(),
        );
        sources.extend(self.opaque.iter().map(|intent| intent.source_item_index));
        sources.extend(
            self.transparent
                .iter()
                .map(|intent| intent.source_item_index),
        );
        for queue in self.shadow_queues.values() {
            sources.extend(queue.iter().map(|intent| intent.source_item_index));
        }
        sort_and_dedup_sources(&mut sources);
        sources
    }

    fn extend_sources_for_key(&self, key: DrawPlanKey, sources: &mut Vec<u32>) {
        match key {
            DrawPlanKey::Opaque | DrawPlanKey::Prepass { .. } => {
                sources.extend(self.opaque.iter().map(|intent| intent.source_item_index));
            }
            DrawPlanKey::Transparent => {
                sources.extend(
                    self.transparent
                        .iter()
                        .map(|intent| intent.source_item_index),
                );
            }
            DrawPlanKey::Shadow {
                light_id,
                layer_index,
            } => {
                if let Some(queue) = self.shadow_queues.get(&(light_id, layer_index)) {
                    sources.extend(queue.iter().map(|intent| intent.source_item_index));
                }
            }
        }
    }

    /// Returns `true` if both lists are empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.transparent.is_empty()
    }
}

impl Default for RenderLists {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn sort_and_dedup_sources(sources: &mut Vec<u32>) {
    sources.sort_unstable();
    sources.dedup();
}

// ============================================================================
// RenderKey — Sort Key
// ============================================================================

/// Render sort key (Pipeline ID + Material ID + Depth).
///
/// Encodes sorting information in a 64-bit integer for efficient radix sorting.
///
/// # Sorting Strategy
/// - **Opaque objects**: Pipeline > Material > Depth (front-to-back)
///   - Minimizes pipeline state switches
///   - Front-to-back leverages Early-Z for performance
/// - **Transparent objects**: Depth (back-to-front) > Pipeline > Material
///   - Ensures correct alpha blending order
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderKey(u64);

impl RenderKey {
    /// Returns the raw 64-bit sort key value.
    #[inline]
    #[must_use]
    pub fn bits(self) -> u64 {
        self.0
    }

    /// Constructs a sort key.
    ///
    /// # Parameters
    /// - `pipeline_id`: Pipeline handle
    /// - `material_index`: Material index (20 bits)
    /// - `depth`: Squared distance to the camera
    /// - `transparent`: Whether the object is transparent
    #[must_use]
    pub fn new(
        pipeline_id: RenderPipelineId,
        material_index: u32,
        depth: f32,
        transparent: bool,
    ) -> Self {
        // 1. Compress depth into 30 bits.
        // Note: assumes depth >= 0.0. Clamping negative values to 0 is safe.
        let d_u32 = if depth.is_sign_negative() {
            0
        } else {
            depth.to_bits() >> 2
        };
        let raw_d_bits = u64::from(d_u32) & 0x3FFF_FFFF;

        // 2. Prepare other fields
        let p_bits = u64::from(pipeline_id.0 & 0x3FFF); // 14 bits
        let m_bits = u64::from(material_index & 0xFFFFF); // 20 bits

        if transparent {
            // [Transparent]: Sort by Depth (far→near) > Pipeline > Material

            // 1. Invert depth so farther objects (larger depth) get smaller values, sorting first.
            let d_bits = raw_d_bits ^ 0x3FFF_FFFF;

            // 2. Bit layout: Depth (30) << 34 | Pipeline (14) << 20 | Material (20)
            Self((d_bits << 34) | (p_bits << 20) | m_bits)
        } else {
            // [Opaque]: Sort by Pipeline > Material > Depth (near→far)

            // Depth in ascending order (smaller depth first = front-to-back)
            let d_bits = raw_d_bits;

            // Pipeline (14) << 50 | Material (20) << 30 | Depth (30)
            Self((p_bits << 50) | (m_bits << 30) | d_bits)
        }
    }
}

// ============================================================================
// RenderFrame
// ============================================================================

/// Render frame manager.
///
/// Uses a render graph architecture:
/// 1. **Extract**: Pull rendering data from the scene
/// 2. **Prepare**: Prepare GPU resources
/// 3. **Execute**: Run render passes via [`FrameComposer`]
///
/// # Performance Notes
/// - `ExtractedScene` is persistent to reuse memory across frames
/// - `FrameComposer` is created per-frame but extremely cheap (just pointer ops)
///
/// # Note
/// `RenderLists` is stored in `RendererState` rather than here
/// to avoid borrow-checker limitations.
pub struct RenderFrame {
    pub(crate) render_state: RenderState,
    pub(crate) extracted_scene: ExtractedScene,
}

impl Default for RenderFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFrame {
    #[must_use]
    pub fn new() -> Self {
        Self {
            render_state: RenderState::new(),
            extracted_scene: ExtractedScene::with_capacity(1024),
        }
    }

    /// Returns a reference to the render state.
    #[inline]
    pub fn render_state(&self) -> &RenderState {
        &self.render_state
    }

    /// Returns a reference to the extracted scene data.
    #[inline]
    pub fn extracted_scene(&self) -> &ExtractedScene {
        &self.extracted_scene
    }

    /// Extract scene data, build shadow views, and prepare global GPU resources.
    ///
    /// # Phases
    ///
    /// 1. **Extract** — Pull rendering data from the [`Scene`] into
    ///    [`ExtractedScene`].
    /// 2. **Feature Variants** — Resolve frame-dependent scene shader
    ///    variants such as clustered lighting and screen-space effects.
    /// 3. **Light Ordering** — Keep local lights packed at the front of the
    ///    extracted list and pre-sort dense local-light sets by camera-
    ///    weighted importance so they can be short-circuited directly into
    ///    the clustered local-light track when no GPU merge is needed.
    /// 4. **Shadow View Generation** — Build [`RenderView`]s for all
    ///    shadow-casting lights (pure math, no GPU work).
    /// 5. **Shadow Metadata** — Write per-light shadow layer indices,
    ///    cascade matrices and split distances into the directional / local
    ///    light tracks so both the global bind group and the RDG local-light
    ///    path already contain correct data.
    /// 6. **Global Prepare** — Upload camera / scene / light uniforms and
    ///    create the global bind group (Group 0).
    ///
    /// # Note
    ///
    /// Surface acquisition is deferred to `FrameComposer::render()` to
    /// minimise swap-chain buffer hold time.
    pub fn extract_and_prepare(
        &mut self,
        resource_manager: &mut ResourceManager,
        scene: &mut Scene,
        camera: &RenderCamera,
        assets: &AssetServer,
        frame_time: FrameTime,
        render_lists: &mut RenderLists,
        surface_size: (u32, u32),
        max_extracted_lights: usize,
        feature_config: RenderFrameFeatureConfig,
        published_pmrem_view: Option<&Tracked<wgpu::TextureView>>,
        published_pmrem_max_mip: Option<f32>,
    ) {
        use crate::core::view::RenderView;

        resource_manager.next_frame();

        // ── 1. Extract ─────────────────────────────────────────────────
        self.extracted_scene
            .extract_into(scene, camera, assets, max_extracted_lights);

        // ── 2. Resolve frame-dependent shader variants ────────────────
        self.resolve_scene_feature_variants(scene, feature_config);

        // ── 3. Resolve GPU environment ─────────────────────────────────
        let desired_env_max_mip = resource_manager.resolve_gpu_environment(
            scene.id(),
            assets,
            &scene.environment,
            &scene.background.mode,
        );
        let env_max_mip = if resource_manager.scene_environment_source_ready(scene.id()) {
            desired_env_max_mip
        } else {
            published_pmrem_max_mip.unwrap_or(desired_env_max_mip)
        };
        self.extracted_scene.prepare_gpu_scene_inputs(env_max_mip);

        // ── 4. Build shadow views (pure math) ──────────────────────────
        render_lists.clear();
        render_lists.active_views.push(RenderView::new_main_camera(
            camera.view_projection_matrix,
            camera.frustum,
            surface_size,
        ));

        let shadow_views = Self::build_shadow_views(
            &self.extracted_scene,
            camera,
            render_lists.active_views.len(),
        );

        // ── 5. Shadow map allocation is deferred to RDG ─────────────────
        // The physical shadow texture is now a transient RDG resource,
        // allocated by the graph compiler after topology compilation.
        // No `ensure_shadow_maps()` call is needed here.

        render_lists.active_views.extend(shadow_views);

        // ── 6. Update light storage buffer with shadow metadata ────────
        Self::update_light_shadow_metadata(&render_lists.active_views, &mut self.extracted_scene);

        // ── 7. Global GPU resources ────────────────────────────────────
        self.render_state.update(
            scene.id(),
            camera,
            frame_time,
            resource_manager.frame_index(),
            surface_size,
        );
        resource_manager.prepare_global_with_pmrem_override(
            assets,
            &self.extracted_scene,
            &self.render_state,
            published_pmrem_view,
        );
    }

    fn resolve_scene_feature_variants(
        &mut self,
        scene: &Scene,
        feature_config: RenderFrameFeatureConfig,
    ) {
        let active_local_light_count = self.extracted_scene.local_light_count() as u32;
        let clustered_lighting_enabled = feature_config
            .clustered_shading
            .is_enabled(active_local_light_count)
            && active_local_light_count > 0;

        if clustered_lighting_enabled {
            self.extracted_scene
                .scene_variants
                .insert(SceneFeatures::USE_CLUSTERED_SHADING);
        }

        let screen_space_supported = feature_config.render_path.supports_post_processing()
            && feature_config.msaa_samples <= 1;

        if !(screen_space_supported && scene.ssgi.enabled) {
            self.extracted_scene
                .scene_variants
                .remove(SceneFeatures::USE_SSGI);
            self.extracted_scene.scene_defines.remove("USE_SSGI");
        }

        let ssr_enabled = screen_space_supported && scene.ssr.enabled;
        if !ssr_enabled {
            self.extracted_scene
                .scene_variants
                .remove(SceneFeatures::USE_SSR);
            self.extracted_scene.scene_defines.remove("USE_SSR");
        }

        if scene.ssss.enabled || ssr_enabled {
            self.extracted_scene
                .scene_defines
                .set("USE_SCREEN_SPACE_FEATURES", "1");
        } else {
            self.extracted_scene
                .scene_defines
                .remove("USE_SCREEN_SPACE_FEATURES");
        }
    }

    /// Build [`RenderView`]s for all shadow-casting lights.
    ///
    /// Returns a `Vec` of shadow views. Each directional light may produce
    /// multiple cascade views; spot lights produce a single view; point
    /// lights produce 6 cube map face views for omnidirectional shadows.
    fn build_shadow_views(
        extracted_scene: &ExtractedScene,
        camera: &RenderCamera,
        _existing_view_count: usize,
    ) -> Vec<crate::core::view::RenderView> {
        use myth_scene::light::LightKind;

        // Compute scene caster extent (for CSM Z extension)
        let camera_pos: glam::Vec3 = camera.position.to_array().into();
        let mut max_distance = 0.0f32;
        for item in &extracted_scene.render_items {
            if !item.cast_shadows {
                continue;
            }
            let aabb = item.world_aabb;
            let effective_radius = if aabb.is_finite() {
                aabb.size().length() * 0.5
            } else {
                0.0
            };
            let center_ws = aabb.center();
            let distance = camera_pos.distance(center_ws) + effective_radius;
            max_distance = max_distance.max(distance);
        }
        let scene_caster_extent = max_distance.max(50.0);

        let mut shadow_views = Vec::with_capacity(16);

        for (light_buffer_index, light) in extracted_scene.lights.iter().enumerate() {
            if !light.cast_shadows {
                continue;
            }
            let shadow_cfg = light.shadow.clone().unwrap_or_default();

            match &light.kind {
                LightKind::Directional(_) => {
                    let cam_far = if camera.far.is_finite() {
                        camera.far
                    } else {
                        shadow_cfg.max_shadow_distance
                    };
                    let shadow_far = shadow_cfg.max_shadow_distance.min(cam_far);
                    let caster_extension = scene_caster_extent.max(shadow_cfg.max_shadow_distance);
                    let base_layer = shadow_views.len() as u32;

                    let (views, _splits) = shadow_utils::build_directional_views(
                        light.id,
                        light.direction,
                        light_buffer_index,
                        camera,
                        &shadow_cfg,
                        shadow_far,
                        caster_extension,
                        base_layer,
                    );
                    shadow_views.extend(views);
                }
                LightKind::Spot(spot) => {
                    let base_layer = shadow_views.len() as u32;
                    shadow_views.push(shadow_utils::build_spot_view(
                        light.id,
                        light_buffer_index,
                        light.position,
                        light.direction,
                        spot,
                        &shadow_cfg,
                        base_layer,
                    ));
                }
                LightKind::Point(point) => {
                    // Coarse culling: skip point lights whose bounding sphere
                    // does not intersect the main camera frustum.
                    if !camera
                        .frustum
                        .intersects_sphere(light.position, point.range)
                    {
                        continue;
                    }
                    let base_layer = shadow_views.len() as u32;
                    shadow_views.extend(shadow_utils::build_point_views(
                        light.id,
                        light_buffer_index,
                        light.position,
                        point,
                        &shadow_cfg,
                        base_layer,
                    ));
                }
            }
        }

        shadow_views
    }

    /// Write per-light shadow metadata (layer indices, cascade matrices,
    /// bias values) into the scene's light storage buffer.
    ///
    fn update_light_shadow_metadata(
        active_views: &[crate::core::view::RenderView],
        extracted_scene: &mut ExtractedScene,
    ) {
        use crate::core::view::ViewTarget;
        use crate::graph::shadow_utils::MAX_CASCADES;
        use glam::{Mat4, Vec4};
        use myth_scene::light::LightKind;

        // The extracted light payload already carries the complete no-shadow
        // state. Avoid taking write guards here: `CpuBuffer::write()` advances
        // the version even when the assigned values are unchanged, which used
        // to force two redundant Queue uploads on every shadow-free frame.
        let has_shadow_views = active_views
            .iter()
            .any(crate::core::view::RenderView::is_shadow);
        if !has_shadow_views {
            return;
        }

        // Reset shadow fields
        {
            let mut local_light_storage = extracted_scene.local_light_storage_buffer.write();
            for light in local_light_storage.iter_mut() {
                light.shadow_layer_index = -1;
                light.point_shadow_index = -1;
                light.shadow_matrices.0 = [Mat4::IDENTITY; 4];
                light.cascade_count = 0;
                light.cascade_splits = Vec4::ZERO;
            }

            let mut directional_light_storage =
                extracted_scene.directional_light_storage_buffer.write();
            for light in directional_light_storage.iter_mut() {
                light.shadow_layer_index = -1;
                light.point_shadow_index = -1;
                light.shadow_matrices.0 = [Mat4::IDENTITY; 4];
                light.cascade_count = 0;
                light.cascade_splits = Vec4::ZERO;
            }
        }

        // Aggregate per-light shadow metadata
        {
            let mut local_light_storage = extracted_scene.local_light_storage_buffer.write();
            let mut directional_light_storage =
                extracted_scene.directional_light_storage_buffer.write();
            let local_light_count = extracted_scene.local_light_count();
            let mut point_shadow_counter = 0i32;
            let mut d2_layer_counter = 0u32;

            for (light_buffer_index, light) in extracted_scene.lights.iter().enumerate() {
                if !light.cast_shadows {
                    continue;
                }
                let shadow_cfg = light.shadow.clone().unwrap_or_default();
                let is_point = matches!(light.kind, LightKind::Point(_));

                let mut base_layer = u32::MAX;
                let mut view_count = 0u32;
                let mut cascade_matrices = [Mat4::IDENTITY; MAX_CASCADES as usize];
                let mut cascade_splits_arr = [0.0f32; MAX_CASCADES as usize];

                for view in active_views {
                    let ViewTarget::ShadowLight {
                        light_id,
                        layer_index,
                    } = view.target
                    else {
                        continue;
                    };
                    if light_id != light.id {
                        continue;
                    }
                    if layer_index < base_layer {
                        base_layer = layer_index;
                    }
                    view_count += 1;
                }

                if view_count == 0 {
                    continue;
                }

                // Point lights: store cube index, skip VP matrices (shader
                // uses direction vector + depth comparison).
                if is_point {
                    if let Some(gpu_light) = local_light_storage.get_mut(light_buffer_index) {
                        // point_shadow_index = sequential cube index (0, 1, 2, ...)
                        gpu_light.point_shadow_index = point_shadow_counter;
                        gpu_light.shadow_bias = shadow_cfg.bias;
                        gpu_light.shadow_normal_bias = shadow_cfg.normal_bias;
                    }
                    point_shadow_counter += 1;
                    continue;
                }

                // Directional / Spot: collect VP matrices
                for view in active_views {
                    let ViewTarget::ShadowLight {
                        light_id,
                        layer_index,
                    } = view.target
                    else {
                        continue;
                    };
                    if light_id != light.id {
                        continue;
                    }
                    let cascade_idx = (layer_index - base_layer) as usize;
                    if cascade_idx < MAX_CASCADES as usize {
                        cascade_matrices[cascade_idx] = view.view_projection;
                        if let Some(split) = view.csm_split {
                            cascade_splits_arr[cascade_idx] = split;
                        }
                    }
                }

                let storage = if light.is_directional() {
                    directional_light_storage.get_mut(light_buffer_index - local_light_count)
                } else {
                    local_light_storage.get_mut(light_buffer_index)
                };

                if let Some(gpu_light) = storage {
                    gpu_light.shadow_layer_index = d2_layer_counter.cast_signed();
                    gpu_light.shadow_matrices.0 = cascade_matrices;
                    gpu_light.cascade_count = view_count;
                    gpu_light.cascade_splits = Vec4::new(
                        cascade_splits_arr[0],
                        cascade_splits_arr[1.min(view_count as usize - 1)],
                        cascade_splits_arr[2.min(view_count as usize - 1)],
                        cascade_splits_arr[3.min(view_count as usize - 1)],
                    );
                    gpu_light.shadow_bias = shadow_cfg.bias;
                    gpu_light.shadow_normal_bias = shadow_cfg.normal_bias;
                }
                d2_layer_counter += view_count;
            }
        }
    }

    /// Periodically prune stale resources.
    pub fn maybe_prune(&self, resource_manager: &mut ResourceManager) {
        // Periodic cleanup (TODO: LRU eviction strategy)
        if resource_manager.frame_index().is_multiple_of(600) {
            resource_manager.prune(6000);
        }
    }
}

#[cfg(test)]
mod draw_plan_tests {
    use super::*;
    use slotmap::Key;

    fn test_command(sort_key: u64) -> DrawCommand {
        DrawCommand {
            source_item_index: 0,
            sort_key,
            pipeline_id: RenderPipelineId(0),
            vertex_buffers: SmallVec::new(),
            index_buffer: None,
            bind_group_1: None,
            bind_group_2: (DrawBindGroupSlot(0), 0),
            stencil_reference: None,
            vertex_range: 0..3,
            instance_range: 0..1,
        }
    }

    fn render_intent(source_item_index: u32) -> RenderIntent {
        RenderIntent {
            source_item_index,
            geometry_handle: GeometryHandle::null(),
            material_handle: MaterialHandle::null(),
            material_index: source_item_index,
            distance_sq: source_item_index as f32,
        }
    }

    fn shadow_intent(source_item_index: u32) -> ShadowRenderIntent {
        ShadowRenderIntent {
            source_item_index,
            geometry_handle: GeometryHandle::null(),
            material_handle: MaterialHandle::null(),
        }
    }

    #[test]
    fn clearing_store_invalidates_attempt_scoped_plan_ids() {
        let mut store = DrawPlanStore::new();
        let plan = store.reserve_plan(DrawPlanKey::Opaque);
        store.mark_live(plan);
        let start = store.begin_finalize(plan);
        store.finish_finalize(plan, start);

        assert!(store.resolve_plan(plan).is_some());
        store.clear();
        assert!(store.resolve_plan(plan).is_none());
        assert!(store.is_empty());
    }

    #[test]
    fn reservations_are_keyed_and_demanded_empty_plans_finalize() {
        let mut store = DrawPlanStore::new();
        let opaque = store.reserve_plan(DrawPlanKey::Opaque);
        let repeated = store.reserve_plan(DrawPlanKey::Opaque);
        let transparent = store.reserve_plan(DrawPlanKey::Transparent);

        assert_eq!(opaque, repeated);
        assert_ne!(opaque, transparent);
        assert!(!store.is_live(opaque));
        assert!(store.resolve_plan(opaque).is_none());

        store.mark_live(opaque);
        let start = store.begin_finalize(opaque);
        store.finish_finalize(opaque, start);

        assert!(store.is_live(opaque));
        assert!(
            store
                .resolve_plan(opaque)
                .is_some_and(<[DrawCommand]>::is_empty)
        );
        assert!(store.resolve_plan(transparent).is_none());
        assert_eq!(
            store.plan_meta(0),
            Some(DrawPlanMeta {
                id: opaque,
                key: DrawPlanKey::Opaque,
                is_live: true,
                is_finalized: true,
            })
        );
    }

    #[test]
    fn non_empty_plans_finalize_and_sort_out_of_reservation_order() {
        let mut store = DrawPlanStore::new();
        let first = store.reserve_plan(DrawPlanKey::Opaque);
        let second = store.reserve_plan(DrawPlanKey::Transparent);
        store.mark_live(first);
        store.mark_live(second);

        let second_start = store.begin_finalize(second);
        store.push_command(test_command(30));
        store.push_command(test_command(10));
        store.sort_commands_from(second_start);
        store.finish_finalize(second, second_start);
        let first_start = store.begin_finalize(first);
        store.push_command(test_command(20));
        store.sort_commands_from(first_start);
        store.finish_finalize(first, first_start);

        assert_eq!(
            store
                .resolve_plan(first)
                .unwrap()
                .iter()
                .map(|command| command.sort_key)
                .collect::<Vec<_>>(),
            vec![20]
        );
        assert_eq!(
            store
                .resolve_plan(second)
                .unwrap()
                .iter()
                .map(|command| command.sort_key)
                .collect::<Vec<_>>(),
            vec![10, 30]
        );
        assert_eq!(store.plan_count(), 2);
    }

    #[test]
    fn live_prepass_and_exact_shadow_select_only_their_deduplicated_sources() {
        let mut lists = RenderLists::new();
        lists.opaque = vec![render_intent(2), render_intent(1)];
        lists.transparent = vec![render_intent(3)];
        lists
            .shadow_queues
            .insert((10, 0), vec![shadow_intent(4), shadow_intent(2)]);
        lists.shadow_queues.insert((20, 3), vec![shadow_intent(5)]);

        let mut plans = DrawPlanStore::new();
        let dead_transparent = plans.reserve_plan(DrawPlanKey::Transparent);
        let prepass = plans.reserve_plan(DrawPlanKey::Prepass {
            needs_normal: true,
            needs_feature_id: false,
            needs_velocity: true,
        });
        let live_shadow = plans.reserve_plan(DrawPlanKey::Shadow {
            light_id: 10,
            layer_index: 0,
        });
        let dead_shadow = plans.reserve_plan(DrawPlanKey::Shadow {
            light_id: 20,
            layer_index: 3,
        });

        plans.mark_live(prepass);
        plans.mark_live(live_shadow);

        assert!(!plans.is_live(dead_transparent));
        assert!(!plans.is_live(dead_shadow));
        assert_eq!(
            lists.live_source_indices(plans.live_keys()),
            vec![1, 2, 4],
            "Prepass must select opaque sources, the exact live shadow must add only its queue, and shared source 2 must be deduplicated"
        );
        assert_eq!(lists.all_source_indices(), vec![1, 2, 3, 4, 5]);
    }
}
