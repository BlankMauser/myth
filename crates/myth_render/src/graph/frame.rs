//! Render frame management.
//!
//! `RenderFrame` owns the frame's CPU scene snapshot, render state, light/view
//! metadata, and scene-global CPU inputs required for graph construction.
//! All physical preparation is deferred until after DPE by [`FrameComposer`].
//!
//! # Four-Phase Rendering Architecture
//!
//! The rendering pipeline is divided into four distinct phases:
//!
//! 1. **Snapshot**: `extract()` captures scene/light/view state without GPU
//!    uploads or bind-group creation.
//! 2. **Compose + Compile**: [`FrameComposer`] collects CPU draw intents,
//!    declares topology, and runs DPE.
//! 3. **Deferred prepare**: only live passes and their exact draw-source union
//!    materialize physical resources and pipelines.
//! 4. **Execute + Commit**: blind commands are recorded; temporal/producer
//!    state is published only after submission.
//!

use glam::Mat4;
use rustc_hash::FxHashMap;
#[cfg(test)]
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::gpu::Tracked;
use crate::core::{RenderView, ResourceManager};
use crate::pipeline::RenderPipelineId;
use crate::renderer::FrameTime;
use crate::settings::{ClusteredShadingMode, GpuDrivenMode, RenderPath};
use myth_assets::{AssetServer, GeometryHandle, MaterialHandle};
use myth_scene::Scene;
use myth_scene::camera::{Frustum, RenderCamera};

use super::asset_frame::{GeometryFrameId, ObjectInterfaceId};
use super::capacity::{
    CapacityUsage, estimated_hash_map_footprint, shrink_hash_map_by_family_ratio,
    shrink_vec_by_family_ratio, vec_footprint,
};
use super::core::TextureNodeId;
use super::extracted::{ExtractedScene, SceneFeatures};
use super::gpu_driven_policy::GpuDrivenFamilyMask;
use super::material_plan::MaterialFrameId;
use super::render_state::RenderState;
use super::render_world::RenderWorldSyncStats;
use super::shadow_utils;

// ============================================================================
// Render intents, physical commands, and render lists
// ============================================================================

/// Stable identity of one item in the immutable frame extraction snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ExtractedSourceId(u32);

impl ExtractedSourceId {
    #[inline]
    #[must_use]
    pub(crate) fn from_index(index: usize) -> Self {
        Self(
            u32::try_from(index)
                .expect("one extracted scene cannot contain more than u32::MAX draw sources"),
        )
    }

    #[must_use]
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Dense frame-local identity assigned only to sources referenced by intents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct FrameSourceId(u32);

impl FrameSourceId {
    #[inline]
    #[must_use]
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("one frame cannot retain more than u32::MAX draw sources"))
    }

    #[must_use]
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Dense graph-attempt identity of one physically prepared draw source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PreparedSourceId(u32);

impl PreparedSourceId {
    #[inline]
    #[must_use]
    pub(crate) fn from_index(index: usize) -> Self {
        Self(
            u32::try_from(index)
                .expect("one graph attempt cannot prepare more than u32::MAX draw sources"),
        )
    }

    #[must_use]
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

/// CPU-only main-view draw intent.
///
/// This is produced before RDG compilation and contains no WGPU owner or
/// pipeline ID. `source` is the compact frame-local identity used to join the
/// intent with graph-attempt physical preparation after DPE.
#[derive(Debug, Clone, Copy)]
pub struct RenderIntent {
    pub(crate) source: FrameSourceId,
    pub geometry_handle: GeometryHandle,
    pub material_handle: MaterialHandle,
    pub material_index: u32,
    pub distance_sq: f32,
}

/// CPU-only per-shadow-view draw intent.
#[derive(Debug, Clone, Copy)]
pub struct ShadowRenderIntent {
    pub(crate) source: FrameSourceId,
    pub geometry_handle: GeometryHandle,
    pub material_handle: MaterialHandle,
}

/// Frame-scoped feature inputs that affect scene shader variants.
#[derive(Debug, Clone, Copy)]
pub struct RenderFrameFeatureConfig {
    pub clustered_shading: ClusteredShadingMode,
    /// Controls whether extraction captures the additional coherent asset
    /// facts consumed exclusively by GPU-driven preflight. `Off` keeps the
    /// CpuDirect baseline free of those version/topology queries.
    pub gpu_driven: GpuDrivenMode,
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
pub(crate) struct DrawPlanId {
    attempt_token: u64,
    index: u32,
}

impl DrawPlanId {
    #[must_use]
    pub(crate) const fn index(self) -> usize {
        self.index as usize
    }
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
/// Produced by [`scene_pipeline_coordinator::finalize_live_draw_plans`] after
/// DPE, consumed by the
/// unified [`submit_scene_submission`] helper during the execute phase. Contains
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
/// [`scene_pipeline_coordinator::finalize_live_draw_plans`]: super::scene_pipeline_coordinator::finalize_live_draw_plans
/// [`submit_scene_submission`]: super::passes::draw::submit_scene_submission
pub(crate) struct DrawCommand {
    /// Test-only semantic identity used by Direct-stream equivalence checks.
    /// Production commands need no source identity after morph-history
    /// candidates have been recorded at finalization.
    #[cfg(test)]
    pub(crate) frame_source: FrameSourceId,

    /// Composite sort key for minimising GPU state switches.
    ///
    /// * Opaque — Pipeline › Material › Depth (front-to-back).
    /// * Transparent — Depth (back-to-front) › Pipeline › Material.
    /// * Shadow — Pipeline (state-switch minimisation).
    pub sort_key: u64,

    /// Stable handle for the pre-selected render pipeline.
    pub pipeline_id: RenderPipelineId,

    /// Range into the DrawPlanStore's retained flat vertex-binding table.
    ///
    /// Variable-length bindings stay out of the command object, so arbitrary
    /// supported geometry layouts cannot trigger one heap allocation per draw.
    pub(crate) vertex_buffer_range: std::ops::Range<u32>,

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

/// Test-only, backend-semantic observation of one finalized direct draw
/// stream.
///
/// Attempt tokens, command-array ranges, raw pointers, and frame-local
/// physical-owner slots are deliberately excluded. Buffer and bind-group
/// slots are resolved to their stable [`Tracked`] IDs so two implementations
/// can be compared even when their private interning order differs.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DrawStreamObservation {
    /// Reservation-table order of the canonical plan within this attempt.
    pub(crate) plan_order: u32,
    pub(crate) plan_key: DrawPlanKey,
    /// Vector order is the exact execute order for this stream.
    pub(crate) commands: Vec<DrawCommandObservation>,
}

/// Fully normalized POD payload of one direct draw command.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DrawCommandObservation {
    pub(crate) frame_source: FrameSourceId,
    pub(crate) sort_key: u64,
    pub(crate) pipeline_id: RenderPipelineId,
    /// Physical buffer IDs in vertex binding order.
    pub(crate) vertex_buffer_ids: SmallVec<[u64; 2]>,
    pub(crate) index_buffer: Option<DrawIndexBufferObservation>,
    pub(crate) material_bind_group_id: Option<u64>,
    pub(crate) object_bind_group_id: u64,
    pub(crate) object_dynamic_offset: u32,
    pub(crate) stencil_reference: Option<u32>,
    pub(crate) vertex_range: std::ops::Range<u32>,
    pub(crate) instance_range: std::ops::Range<u32>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DrawIndexBufferObservation {
    pub(crate) buffer_id: u64,
    pub(crate) format: wgpu::IndexFormat,
    pub(crate) index_count: u32,
}

/// Exact graph-attempt physical state unique to one prepared draw instance.
///
/// Shared Geometry, Material, and object-interface owners are interned in
/// sibling attempt stores. This dense record contains only Copy IDs plus the
/// genuinely per-instance object binding and dynamic offset.
pub(crate) struct PreparedDrawSource {
    /// Stable frame-local identity selected by the source Gather transaction.
    ///
    /// Keeping the join identity beside the dense prepared record lets every
    /// later phase address this source through [`PreparedSourceId`] without
    /// repeating the frame-source hash lookup.
    pub(crate) frame_source: FrameSourceId,
    /// Stable index into the immutable extraction snapshot. Later phases still
    /// validate the RenderLists join before dereferencing this index, but do
    /// not have to rediscover it from the frame source.
    pub(crate) extracted_source: ExtractedSourceId,
    pub(crate) geometry: GeometryFrameId,
    pub(crate) material: MaterialFrameId,
    pub(crate) object_interface: ObjectInterfaceId,
    pub(crate) material_bind_group: DrawBindGroupSlot,
    pub(crate) object_bind_group: (DrawBindGroupSlot, u32),
    /// Whether a submitted command using this source must advance the Mesh's
    /// CPU morph-history window. Rigid and skin-only sources keep this false.
    pub(crate) commits_morph_history: bool,
}

/// Compact graph-attempt source preparation addressed through frame-local IDs.
///
/// Only DPE-live sources occupy the dense source vector. The lookup preserves
/// their frame identity without forcing preparation to mirror the potentially
/// sparse extracted-scene table. All physical Buffer/BindGroup owners live in
/// [`DrawPlanStore`], so clearing both stores deterministically releases one
/// graph attempt.
pub(crate) struct PreparedDrawStore {
    sources: Vec<PreparedDrawSource>,
    /// Dense `FrameSourceId -> PreparedSourceId` join. Slots are valid only
    /// when their stamp equals `source_join_generation`; warm attempts neither
    /// clear nor fill the retained high-water arrays.
    source_join_indices: Vec<u32>,
    source_join_stamps: Vec<u32>,
    source_join_generation: u32,
    source_domain_len: usize,
    sources_initialized: bool,
}

const PREPARED_DRAW_SOURCE_BASELINE: usize = 512;

impl PreparedDrawStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            sources: Vec::with_capacity(PREPARED_DRAW_SOURCE_BASELINE),
            source_join_indices: Vec::with_capacity(PREPARED_DRAW_SOURCE_BASELINE),
            source_join_stamps: Vec::with_capacity(PREPARED_DRAW_SOURCE_BASELINE),
            source_join_generation: 1,
            source_domain_len: 0,
            sources_initialized: false,
        }
    }

    /// Release graph-attempt metadata while retaining dense source/index
    /// allocations for the next frame.
    pub(crate) fn clear(&mut self) {
        self.sources.clear();
        self.source_domain_len = 0;
        self.sources_initialized = false;
        self.advance_source_join_generation();
    }

    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        let join_slot_bytes = std::mem::size_of::<u32>();
        let join_active_bytes = self
            .source_domain_len
            .saturating_mul(join_slot_bytes)
            .saturating_mul(2);
        let join_retained_bytes = self
            .source_join_indices
            .capacity()
            .saturating_add(self.source_join_stamps.capacity())
            .saturating_mul(join_slot_bytes);
        let join_baseline_bytes = PREPARED_DRAW_SOURCE_BASELINE
            .saturating_mul(join_slot_bytes)
            .saturating_mul(2);
        vec_footprint(&self.sources, PREPARED_DRAW_SOURCE_BASELINE)
            + CapacityUsage::new(join_active_bytes, join_retained_bytes, join_baseline_bytes)
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        assert!(
            !self.sources_initialized && self.sources.is_empty(),
            "live prepared-source joins cannot be trimmed"
        );
        shrink_vec_by_family_ratio(
            &mut self.sources,
            PREPARED_DRAW_SOURCE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        let current_slots = self
            .source_join_indices
            .len()
            .max(self.source_join_stamps.len());
        let target_slots = if retained_bytes == 0 || target_bytes >= retained_bytes {
            current_slots
        } else {
            (((current_slots as u128) * (target_bytes as u128))
                .div_ceil(retained_bytes as u128)
                .min(usize::MAX as u128) as usize)
                .max(PREPARED_DRAW_SOURCE_BASELINE)
                .min(current_slots)
        };
        self.source_join_indices.truncate(target_slots);
        self.source_join_stamps.truncate(target_slots);
        self.source_join_indices.shrink_to(target_slots);
        self.source_join_stamps.shrink_to(target_slots);
    }

    /// Reserve the expected DPE-live source count and dense frame-source domain
    /// once for this graph attempt.
    pub(crate) fn begin_sources(&mut self, expected_live: usize, source_domain_len: usize) {
        assert!(
            !self.sources_initialized,
            "prepared draw sources may be initialized only once per graph attempt"
        );
        debug_assert!(self.sources.is_empty());
        self.sources.reserve(expected_live);
        if expected_live == 0 {
            // An all-dead graph still completes the attempt transaction, but
            // has no valid frame-source join. In particular, do not size this
            // retained scratch to a large pre-DPE frame domain that cannot be
            // queried successfully in this attempt.
            self.source_domain_len = 0;
            self.sources_initialized = true;
            return;
        }
        if self.source_join_indices.len() < source_domain_len {
            self.source_join_indices.resize(source_domain_len, 0);
            self.source_join_stamps.resize(source_domain_len, 0);
        }
        self.source_domain_len = source_domain_len;
        self.sources_initialized = true;
    }

    /// Publish one exact physical source after its resources have been interned
    /// into the sibling [`DrawPlanStore`].
    pub(crate) fn insert_source(&mut self, source: PreparedDrawSource) -> PreparedSourceId {
        assert!(
            self.sources_initialized,
            "prepared draw source table must be initialized before insertion"
        );
        let frame_index = source.frame_source.index();
        assert!(
            frame_index < self.source_domain_len,
            "prepared draw source exceeds the declared frame-source domain"
        );
        assert_ne!(
            self.source_join_stamps[frame_index], self.source_join_generation,
            "prepared draw source was inserted twice"
        );
        let prepared = PreparedSourceId::from_index(self.sources.len());
        self.sources.push(source);
        self.source_join_indices[frame_index] = prepared.0;
        self.source_join_stamps[frame_index] = self.source_join_generation;
        prepared
    }

    fn advance_source_join_generation(&mut self) {
        self.source_join_generation = self.source_join_generation.wrapping_add(1);
        if self.source_join_generation == 0 {
            // The only O(high-water) clear occurs once per u32 generation
            // wrap, preventing an ancient stamp from becoming current again.
            self.source_join_stamps.fill(0);
            self.source_join_generation = 1;
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn prepared_source_id(
        &self,
        frame_source: FrameSourceId,
    ) -> Option<PreparedSourceId> {
        let index = frame_source.index();
        if index >= self.source_domain_len
            || self.source_join_stamps.get(index).copied() != Some(self.source_join_generation)
        {
            return None;
        }
        Some(PreparedSourceId(self.source_join_indices[index]))
    }

    /// Resolve the only frame-domain join in the scene pipeline transaction.
    ///
    /// Gather starts from a [`FrameSourceId`], so it performs this
    /// generation-stamped dense lookup once and carries the returned
    /// [`PreparedSourceId`] through Plan, Materialize, and Finalize. All later
    /// phases directly index `sources`.
    #[inline]
    #[must_use]
    pub(crate) fn source_for_frame(
        &self,
        frame_source: FrameSourceId,
    ) -> Option<(PreparedSourceId, &PreparedDrawSource)> {
        let prepared = self.prepared_source_id(frame_source)?;
        let source = self.sources.get(prepared.index())?;
        (source.frame_source == frame_source).then_some((prepared, source))
    }

    #[inline]
    #[must_use]
    pub(crate) fn source(&self, prepared: PreparedSourceId) -> Option<&PreparedDrawSource> {
        self.sources.get(prepared.index())
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
pub(crate) struct DrawPlanStore {
    commands: Vec<DrawCommand>,
    /// Flat variable-length payload addressed by DrawCommand ranges. This is
    /// retained across attempts and therefore allocation-free at warm shape.
    command_vertex_buffers: Vec<DrawBufferSlot>,
    plans: Vec<DrawPlanRecord>,
    plan_lookup: FxHashMap<DrawPlanKey, DrawPlanId>,
    /// Flat variable-length logical texture dependencies. The map owns only
    /// ranges, so clearing a graph retains one shared high-water allocation
    /// even when a custom material spills past eight textures.
    texture_dependencies: Vec<TextureNodeId>,
    texture_dependency_ranges: FxHashMap<DrawPlanKey, std::ops::Range<usize>>,
    open_texture_dependencies: Option<(DrawPlanKey, usize)>,
    open_plan: Option<(DrawPlanId, u32)>,
    buffers: Vec<Tracked<wgpu::Buffer>>,
    buffer_lookup: FxHashMap<u64, DrawBufferSlot>,
    bind_groups: Vec<Tracked<wgpu::BindGroup>>,
    bind_group_lookup: FxHashMap<u64, DrawBindGroupSlot>,
    /// Unique sources whose morph payload reached at least one finalized live
    /// command. A generation-stamped dense membership table removes the old
    /// per-submit HashSet allocation while keeping main/prepass/shadow aliases
    /// idempotent.
    morph_history_sources: Vec<FrameSourceId>,
    morph_history_source_stamps: Vec<u32>,
    morph_history_source_generation: u32,
    morph_history_active_domain_len: usize,
    attempt_token: u64,
}

const DRAW_COMMAND_BASELINE: usize = 1024;
const DRAW_COMMAND_VERTEX_BUFFER_BASELINE: usize = DRAW_COMMAND_BASELINE * 2;
const DRAW_PLAN_BASELINE: usize = 16;
const DRAW_PHYSICAL_OWNER_BASELINE: usize = 256;
const MAX_DRAW_VERTEX_BUFFERS: usize = 8;

impl Default for DrawPlanStore {
    fn default() -> Self {
        Self::new()
    }
}

impl DrawPlanStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            commands: Vec::with_capacity(DRAW_COMMAND_BASELINE),
            command_vertex_buffers: Vec::with_capacity(DRAW_COMMAND_VERTEX_BUFFER_BASELINE),
            plans: Vec::with_capacity(DRAW_PLAN_BASELINE),
            plan_lookup: FxHashMap::default(),
            texture_dependencies: Vec::with_capacity(DRAW_PLAN_BASELINE * 8),
            texture_dependency_ranges: FxHashMap::default(),
            open_texture_dependencies: None,
            open_plan: None,
            buffers: Vec::with_capacity(DRAW_PHYSICAL_OWNER_BASELINE),
            buffer_lookup: FxHashMap::default(),
            bind_groups: Vec::with_capacity(DRAW_PHYSICAL_OWNER_BASELINE),
            bind_group_lookup: FxHashMap::default(),
            // Morphing is intentionally cold: a wholly rigid application
            // should not pay even one allocation for history bookkeeping.
            morph_history_sources: Vec::new(),
            morph_history_source_stamps: Vec::new(),
            morph_history_source_generation: 1,
            morph_history_active_domain_len: 0,
            attempt_token: next_draw_plan_attempt(),
        }
    }

    /// Releases all graph-attempt WGPU owners while retaining store capacity.
    pub(crate) fn clear(&mut self) {
        self.commands.clear();
        self.command_vertex_buffers.clear();
        self.plans.clear();
        self.plan_lookup.clear();
        self.texture_dependencies.clear();
        self.texture_dependency_ranges.clear();
        self.open_texture_dependencies = None;
        self.open_plan = None;
        self.buffer_lookup.clear();
        self.buffers.clear();
        self.bind_group_lookup.clear();
        self.bind_groups.clear();
        self.morph_history_sources.clear();
        self.morph_history_active_domain_len = 0;
        self.advance_morph_history_generation();
        self.attempt_token = next_draw_plan_attempt();
    }

    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.commands, DRAW_COMMAND_BASELINE)
            + vec_footprint(
                &self.command_vertex_buffers,
                DRAW_COMMAND_VERTEX_BUFFER_BASELINE,
            )
            + vec_footprint(&self.plans, DRAW_PLAN_BASELINE)
            + estimated_hash_map_footprint(&self.plan_lookup, DRAW_PLAN_BASELINE)
            + vec_footprint(&self.texture_dependencies, DRAW_PLAN_BASELINE * 8)
            + estimated_hash_map_footprint(&self.texture_dependency_ranges, DRAW_PLAN_BASELINE)
            + vec_footprint(&self.buffers, DRAW_PHYSICAL_OWNER_BASELINE)
            + estimated_hash_map_footprint(&self.buffer_lookup, DRAW_PHYSICAL_OWNER_BASELINE)
            + vec_footprint(&self.bind_groups, DRAW_PHYSICAL_OWNER_BASELINE)
            + estimated_hash_map_footprint(&self.bind_group_lookup, DRAW_PHYSICAL_OWNER_BASELINE)
            + vec_footprint(&self.morph_history_sources, 0)
            + CapacityUsage::new(
                self.morph_history_active_domain_len
                    .saturating_mul(std::mem::size_of::<u32>()),
                self.morph_history_source_stamps
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
                0,
            )
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        shrink_vec_by_family_ratio(
            &mut self.commands,
            DRAW_COMMAND_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.command_vertex_buffers,
            DRAW_COMMAND_VERTEX_BUFFER_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.plans,
            DRAW_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.plan_lookup,
            DRAW_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.texture_dependencies,
            DRAW_PLAN_BASELINE * 8,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.texture_dependency_ranges,
            DRAW_PLAN_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.buffers,
            DRAW_PHYSICAL_OWNER_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.buffer_lookup,
            DRAW_PHYSICAL_OWNER_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.bind_groups,
            DRAW_PHYSICAL_OWNER_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.bind_group_lookup,
            DRAW_PHYSICAL_OWNER_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.morph_history_sources,
            0,
            target_bytes,
            retained_bytes,
        );
        trim_zero_baseline_u32_join(
            &mut self.morph_history_source_stamps,
            target_bytes,
            retained_bytes,
        );
    }

    /// Record one source whose morph state is represented by an emitted live
    /// command. The first occurrence establishes stable commit order; aliases
    /// in other live plans are O(1) stamped no-ops.
    #[inline]
    pub(crate) fn record_emitted_morph_history_source(&mut self, source: FrameSourceId) {
        let required = source.index() + 1;
        if self.morph_history_source_stamps.len() < required {
            self.morph_history_source_stamps.resize(required, 0);
        }
        self.morph_history_active_domain_len = self.morph_history_active_domain_len.max(required);
        if self.morph_history_source_stamps[source.index()] == self.morph_history_source_generation
        {
            return;
        }
        self.morph_history_source_stamps[source.index()] = self.morph_history_source_generation;
        self.morph_history_sources.push(source);
    }

    #[inline]
    #[must_use]
    pub(crate) fn emitted_morph_history_sources(&self) -> &[FrameSourceId] {
        &self.morph_history_sources
    }

    fn advance_morph_history_generation(&mut self) {
        self.morph_history_source_generation = self.morph_history_source_generation.wrapping_add(1);
        if self.morph_history_source_generation == 0 {
            // The only O(high-water) stamp clear occurs once per u32 wrap.
            self.morph_history_source_stamps.fill(0);
            self.morph_history_source_generation = 1;
        }
    }

    pub(crate) fn reserve_commands(&mut self, additional: usize) {
        self.commands.reserve(additional);
        self.command_vertex_buffers
            .reserve(additional.saturating_mul(2));
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
        let topology_demanded = |candidate| {
            self.plan_lookup.get(&candidate).is_some_and(|id| {
                self.plans
                    .get(id.index())
                    .is_some_and(|plan| plan.topology_demanded)
            })
        };
        let dependency_was_demanded = match key {
            DrawPlanKey::Opaque => OPAQUE_TEXTURE_DEPENDENCY_ALIASES
                .iter()
                .copied()
                .any(topology_demanded),
            // Normalization above guarantees that Prepass cannot reach this
            // match. Transparent and Shadow each have one exact canonical key.
            DrawPlanKey::Transparent | DrawPlanKey::Shadow { .. } => topology_demanded(key),
            DrawPlanKey::Prepass { .. } => {
                unreachable!("prepass dependency key was not normalized")
            }
        };
        assert!(
            !dependency_was_demanded,
            "scene-submission texture dependencies must be registered before reservation"
        );
    }

    pub(crate) fn begin_texture_dependencies(&mut self, key: DrawPlanKey) {
        let key = draw_texture_dependency_key(key);
        assert!(
            self.open_texture_dependencies.is_none(),
            "draw-plan texture dependency registration is already active"
        );
        assert!(
            !self.texture_dependency_ranges.contains_key(&key),
            "draw-plan texture dependencies registered twice for {key:?}"
        );
        self.open_texture_dependencies = Some((key, self.texture_dependencies.len()));
    }

    #[inline]
    pub(crate) fn push_texture_dependency(&mut self, dependency: TextureNodeId) {
        assert!(
            self.open_texture_dependencies.is_some(),
            "draw-plan texture dependency registration is not active"
        );
        self.texture_dependencies.push(dependency);
    }

    pub(crate) fn finish_texture_dependencies(&mut self, key: DrawPlanKey) {
        let key = draw_texture_dependency_key(key);
        let (open_key, start) = self
            .open_texture_dependencies
            .take()
            .expect("draw-plan texture dependency registration is not active");
        assert_eq!(
            open_key, key,
            "draw-plan texture dependency registration ended with another key"
        );
        let previous = self
            .texture_dependency_ranges
            .insert(key, start..self.texture_dependencies.len());
        debug_assert!(previous.is_none());
    }

    /// Roll back an open flat dependency recipe after graph-construction
    /// unwinding. No range is published until `finish`, so restoring the
    /// pre-registration tail and closing the transaction is sufficient.
    #[cfg(any(test, feature = "rdg_bench_support"))]
    pub(crate) fn abort_texture_dependencies(&mut self) {
        let Some((_key, start)) = self.open_texture_dependencies.take() else {
            return;
        };
        self.texture_dependencies.truncate(start);
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
        self.texture_dependency_ranges
            .get(&normalized_key)
            .map_or(&[], |range| &self.texture_dependencies[range.clone()])
    }

    /// Return whether an ID addresses a reserved plan in this graph attempt.
    #[must_use]
    pub(crate) fn validate_reserved(&self, id: DrawPlanId) -> bool {
        id.attempt_token == self.attempt_token && (id.index as usize) < self.plans.len()
    }

    /// Return the dense attempt-local slot addressed by a valid plan ID.
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

    pub(crate) fn push_command(
        &mut self,
        mut command: DrawCommand,
        vertex_buffers: &[DrawBufferSlot],
    ) {
        assert!(
            self.open_plan.is_some(),
            "draw commands must be appended inside plan finalization"
        );
        assert!(
            vertex_buffers.len() <= MAX_DRAW_VERTEX_BUFFERS,
            "draw command exceeds the Direct backend vertex-buffer limit"
        );
        let vertex_start = u32::try_from(self.command_vertex_buffers.len())
            .expect("one graph attempt cannot address more than u32::MAX vertex bindings");
        let vertex_end = self
            .command_vertex_buffers
            .len()
            .checked_add(vertex_buffers.len())
            .and_then(|end| u32::try_from(end).ok())
            .expect("one graph attempt cannot address more than u32::MAX vertex bindings");
        self.command_vertex_buffers
            .extend_from_slice(vertex_buffers);
        command.vertex_buffer_range = vertex_start..vertex_end;
        self.commands.push(command);
    }

    #[cfg(test)]
    pub(crate) fn push_test_command(&mut self, sort_key: u64) {
        self.push_command(
            DrawCommand {
                frame_source: FrameSourceId::from_index(0),
                sort_key,
                pipeline_id: RenderPipelineId(0),
                vertex_buffer_range: 0..0,
                index_buffer: None,
                bind_group_1: None,
                bind_group_2: (DrawBindGroupSlot(0), 0),
                stencil_reference: None,
                vertex_range: 0..3,
                instance_range: 0..1,
            },
            &[],
        );
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

    #[inline]
    #[must_use]
    pub(crate) fn command_vertex_buffers(&self, command: &DrawCommand) -> &[DrawBufferSlot] {
        &self.command_vertex_buffers
            [command.vertex_buffer_range.start as usize..command.vertex_buffer_range.end as usize]
    }

    /// Capture exact, field-level semantics for every live finalized direct
    /// stream in canonical plan-reservation order.
    ///
    /// This is intentionally test-only: building the owned observation is not
    /// part of the shipping prepare or execute hot path.
    #[cfg(test)]
    pub(crate) fn observe_live_streams(&self) -> Vec<DrawStreamObservation> {
        self.assert_all_live_finalized();
        self.plans
            .iter()
            .enumerate()
            .filter(|(_, plan)| plan.live)
            .map(|(plan_order, plan)| {
                let range = plan
                    .commands
                    .as_ref()
                    .expect("live draw plan passed the finalized-stream barrier");
                DrawStreamObservation {
                    plan_order: u32::try_from(plan_order)
                        .expect("validated draw-plan reservation order exceeds u32"),
                    plan_key: plan.key,
                    commands: self.commands[range.start as usize..range.end as usize]
                        .iter()
                        .map(|command| self.observe_command(command))
                        .collect(),
                }
            })
            .collect()
    }

    #[cfg(test)]
    fn observe_command(&self, command: &DrawCommand) -> DrawCommandObservation {
        let vertex_buffer_ids = self
            .command_vertex_buffers(command)
            .iter()
            .map(|&slot| {
                self.resolve_buffer(slot)
                    .expect("draw command references a missing vertex-buffer owner")
                    .id()
            })
            .collect();
        let index_buffer =
            command
                .index_buffer
                .map(|(slot, format, index_count)| DrawIndexBufferObservation {
                    buffer_id: self
                        .resolve_buffer(slot)
                        .expect("draw command references a missing index-buffer owner")
                        .id(),
                    format,
                    index_count,
                });
        let material_bind_group_id = command.bind_group_1.map(|slot| {
            self.resolve_bind_group(slot)
                .expect("draw command references a missing material bind-group owner")
                .id()
        });
        let (object_bind_group_slot, object_dynamic_offset) = command.bind_group_2;
        let object_bind_group_id = self
            .resolve_bind_group(object_bind_group_slot)
            .expect("draw command references a missing object bind-group owner")
            .id();

        DrawCommandObservation {
            frame_source: command.frame_source,
            sort_key: command.sort_key,
            pipeline_id: command.pipeline_id,
            vertex_buffer_ids,
            index_buffer,
            material_bind_group_id,
            object_bind_group_id,
            object_dynamic_offset,
            stencil_reference: command.stencil_reference,
            vertex_range: command.vertex_range.clone(),
            instance_range: command.instance_range.clone(),
        }
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
            && self.texture_dependency_ranges.is_empty()
            && self.open_texture_dependencies.is_none()
            && self.open_plan.is_none()
            && self.buffers.is_empty()
            && self.buffer_lookup.is_empty()
            && self.bind_groups.is_empty()
            && self.bind_group_lookup.is_empty()
            && self.morph_history_sources.is_empty()
            && self.morph_history_active_domain_len == 0
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

fn trim_zero_baseline_u32_join(stamps: &mut Vec<u32>, target_bytes: usize, retained_bytes: usize) {
    if retained_bytes == 0 || target_bytes >= retained_bytes {
        return;
    }
    let current_slots = stamps.len();
    let target_slots = (((current_slots as u128) * (target_bytes as u128))
        .div_ceil(retained_bytes as u128)
        .min(usize::MAX as u128)) as usize;
    stamps.truncate(target_slots);
    stamps.shrink_to(target_slots);
}

#[inline]
fn draw_texture_dependency_key(key: DrawPlanKey) -> DrawPlanKey {
    match key {
        DrawPlanKey::Prepass { .. } => DrawPlanKey::Opaque,
        key => key,
    }
}

/// Every draw-plan key whose material texture dependencies are owned by the
/// canonical opaque dependency set. Keeping this finite alias list beside the
/// normalization function turns registration validation into at most nine
/// direct retained-map probes instead of scanning every shadow plan already
/// reserved in the attempt.
const OPAQUE_TEXTURE_DEPENDENCY_ALIASES: [DrawPlanKey; 9] = [
    DrawPlanKey::Opaque,
    DrawPlanKey::Prepass {
        needs_normal: false,
        needs_feature_id: false,
        needs_velocity: false,
    },
    DrawPlanKey::Prepass {
        needs_normal: false,
        needs_feature_id: false,
        needs_velocity: true,
    },
    DrawPlanKey::Prepass {
        needs_normal: false,
        needs_feature_id: true,
        needs_velocity: false,
    },
    DrawPlanKey::Prepass {
        needs_normal: false,
        needs_feature_id: true,
        needs_velocity: true,
    },
    DrawPlanKey::Prepass {
        needs_normal: true,
        needs_feature_id: false,
        needs_velocity: false,
    },
    DrawPlanKey::Prepass {
        needs_normal: true,
        needs_feature_id: false,
        needs_velocity: true,
    },
    DrawPlanKey::Prepass {
        needs_normal: true,
        needs_feature_id: true,
        needs_velocity: false,
    },
    DrawPlanKey::Prepass {
        needs_normal: true,
        needs_feature_id: true,
        needs_velocity: true,
    },
];

/// Narrow immutable execute-time view over one graph attempt's draw plans.
#[derive(Clone, Copy)]
pub(crate) struct DrawPlanResolver<'a> {
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
    /// Exact CPU culling volume captured from the active RenderView. GPU
    /// Shadow culling consumes the same six planes as the CpuDirect oracle.
    pub frustum: Frustum,
    /// `true` if this layer belongs to a point light (cube array target).
    pub is_point: bool,
}

/// CPU-only render intent lists plus shared frame metadata.
///
/// Intent lists are populated before RDG compilation. After DPE, live
/// [`DrawPlanKey`] values select an exact, deduplicated frame-source union for
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
    /// Deterministic pre-frustum representatives of opaque GPU bins. Each bin
    /// contributes at most one source to asset/topology capture; physical
    /// preparation still selects only representatives whose GPU submissions
    /// survive DPE. The retained vector scales with unique bins rather than
    /// visible instances.
    gpu_opaque_representatives: Vec<FrameSourceId>,
    /// Generation-stamped membership indexed directly by dense
    /// [`FrameSourceId`]. Ordinary frame resets retain both allocation and
    /// bytes; only generation wrap clears the table.
    gpu_opaque_representative_stamps: Vec<u32>,
    gpu_opaque_representative_generation: u32,
    /// Shadow intent queues, keyed by `(light_id, layer_index)` for exact
    /// per-view demand projection.
    ///
    /// Each cascade of a directional light (or each spot light) gets its own queue.
    pub shadow_queues: FxHashMap<(u64, u32), Vec<ShadowRenderIntent>>,

    /// Dense mapping from frame-local source IDs to immutable extraction IDs.
    frame_sources: Vec<ExtractedSourceId>,
    /// Generation-stamped inverse join indexed directly by the dense
    /// `ExtractedSourceId`. Warm frames never clear the retained high-water
    /// arrays; only newly observed extraction indices initialize slots.
    frame_source_indices: Vec<u32>,
    frame_source_stamps: Vec<u32>,
    frame_source_generation: u32,

    /// Debug-only exact GPU ownership proof for each immutable extraction
    /// source. Masks are generation-stamped, so an ordinary debug frame reset
    /// remains O(1). Release collection consumes the dense complements below
    /// and carries no scene-width mask/stamp storage or publication scan.
    #[cfg(test)]
    gpu_family_ownership_masks: Vec<GpuDrivenFamilyMask>,
    #[cfg(test)]
    gpu_family_ownership_stamps: Vec<u32>,
    #[cfg(test)]
    gpu_family_ownership_generation: u32,

    /// Dense CPU complement of the current committed GPU Main/Prepass recipe.
    /// This is published only after the authority transaction succeeds, so
    /// Hybrid intent collection never has to rescan GPU-owned sources.
    gpu_cpu_main_complement: Vec<ExtractedSourceId>,
    /// Dense CPU caster complement shared by selected Shadow2D/ShadowCube
    /// families. Unselected families still consume the complete scene domain.
    gpu_cpu_shadow_complement: Vec<ExtractedSourceId>,
    gpu_cpu_complement_families: GpuDrivenFamilyMask,

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
            gpu_opaque_representatives: Vec::with_capacity(4),
            gpu_opaque_representative_stamps: Vec::with_capacity(4),
            gpu_opaque_representative_generation: 1,
            shadow_queues: FxHashMap::default(),
            frame_sources: Vec::with_capacity(512),
            frame_source_indices: Vec::with_capacity(512),
            frame_source_stamps: Vec::with_capacity(512),
            frame_source_generation: 1,
            // GPU ownership is cold when the default mode is Off. Let the
            // first Auto/Required selection own this allocation.
            #[cfg(test)]
            gpu_family_ownership_masks: Vec::new(),
            #[cfg(test)]
            gpu_family_ownership_stamps: Vec::new(),
            #[cfg(test)]
            gpu_family_ownership_generation: 1,
            gpu_cpu_main_complement: Vec::new(),
            gpu_cpu_shadow_complement: Vec::new(),
            gpu_cpu_complement_families: GpuDrivenFamilyMask::empty(),
            active_views: Vec::with_capacity(16),
            use_transmission: false,
        }
    }

    /// Clears all lists (retains capacity for memory reuse).
    #[inline]
    pub fn clear(&mut self) {
        self.begin_intent_collection();
        self.active_views.clear();
    }

    /// Begin collecting a new frame's intents while preserving extracted views.
    ///
    /// This is used after extraction has already populated `active_views`.
    /// Intent arrays and the dense source interner retain their allocations.
    #[inline]
    pub(crate) fn begin_intent_collection(&mut self) {
        self.opaque.clear();
        self.transparent.clear();
        self.shadow_queues.clear();
        self.frame_sources.clear();
        self.gpu_opaque_representatives.clear();
        self.advance_gpu_opaque_representative_generation();
        self.advance_frame_source_generation();
        #[cfg(test)]
        self.advance_gpu_family_ownership_generation();
        self.use_transmission = false;
    }

    /// Intern one immutable extraction source into this frame's compact domain.
    pub(crate) fn intern_frame_source(&mut self, extracted: ExtractedSourceId) -> FrameSourceId {
        let extracted_index = extracted.index();
        if self.frame_source_indices.len() <= extracted_index {
            let new_len = extracted_index
                .checked_add(1)
                .expect("extracted source join length overflowed");
            self.frame_source_indices.resize(new_len, 0);
            self.frame_source_stamps.resize(new_len, 0);
        }
        if self.frame_source_stamps[extracted_index] == self.frame_source_generation {
            return FrameSourceId(self.frame_source_indices[extracted_index]);
        }

        let source = FrameSourceId::from_index(self.frame_sources.len());
        self.frame_sources.push(extracted);
        self.frame_source_indices[extracted_index] = source.0;
        self.frame_source_stamps[extracted_index] = self.frame_source_generation;
        source
    }

    fn advance_frame_source_generation(&mut self) {
        self.frame_source_generation = self.frame_source_generation.wrapping_add(1);
        if self.frame_source_generation == 0 {
            self.frame_source_stamps.fill(0);
            self.frame_source_generation = 1;
        }
    }

    #[cfg(test)]
    fn advance_gpu_family_ownership_generation(&mut self) {
        self.gpu_family_ownership_generation = self.gpu_family_ownership_generation.wrapping_add(1);
        if self.gpu_family_ownership_generation == 0 {
            self.gpu_family_ownership_stamps.fill(0);
            self.gpu_family_ownership_generation = 1;
        }
    }

    /// Assign one or more disjoint scene-submission families to GPU Scene for
    /// an exact extracted source. Repeating an already-owned family is a
    /// construction bug: it would make two GPU submissions render one cohort.
    #[cfg(test)]
    pub(crate) fn claim_gpu_family_ownership(
        &mut self,
        extracted: ExtractedSourceId,
        families: GpuDrivenFamilyMask,
    ) {
        assert!(
            !families.is_empty(),
            "GPU family ownership claim cannot be empty"
        );
        let index = extracted.index();
        if self.gpu_family_ownership_masks.len() <= index {
            let required = index
                .checked_add(1)
                .expect("GPU family ownership length overflowed");
            self.gpu_family_ownership_masks
                .resize(required, GpuDrivenFamilyMask::empty());
            self.gpu_family_ownership_stamps.resize(required, 0);
        }
        if self.gpu_family_ownership_stamps[index] != self.gpu_family_ownership_generation {
            self.gpu_family_ownership_masks[index] = families;
            self.gpu_family_ownership_stamps[index] = self.gpu_family_ownership_generation;
            return;
        }

        let current = &mut self.gpu_family_ownership_masks[index];
        assert!(
            !current.intersects(families),
            "one extracted source family cannot have two GPU owners"
        );
        current.insert(families);
    }

    /// Return the exact current-frame GPU ownership mask for one extraction
    /// source. A stale generation or unseen high index is CPU-owned.
    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) fn gpu_family_ownership(&self, extracted: ExtractedSourceId) -> GpuDrivenFamilyMask {
        let index = extracted.index();
        if self.gpu_family_ownership_stamps.get(index).copied()
            == Some(self.gpu_family_ownership_generation)
        {
            self.gpu_family_ownership_masks[index]
        } else {
            GpuDrivenFamilyMask::empty()
        }
    }

    /// Begin publishing the exact CPU complement of one successfully committed
    /// GPU Scene recipe. The default Off route never calls this method and does
    /// not touch the retained vectors.
    pub(crate) fn begin_gpu_cpu_complement(&mut self, selected_families: GpuDrivenFamilyMask) {
        assert!(!selected_families.is_empty());
        self.gpu_cpu_main_complement.clear();
        self.gpu_cpu_shadow_complement.clear();
        self.gpu_cpu_complement_families = selected_families;
    }

    /// Append one extraction source which is not part of the retained GPU
    /// recipe. Callers scan extraction order, preserving deterministic CPU draw
    /// and shadow ordering without a sort or hash table.
    pub(crate) fn append_gpu_cpu_complement(
        &mut self,
        extracted: ExtractedSourceId,
        cast_shadows: bool,
        incomplete_families: GpuDrivenFamilyMask,
    ) {
        debug_assert!(
            incomplete_families
                .difference(self.gpu_cpu_complement_families)
                .is_empty()
        );
        if incomplete_families.contains(GpuDrivenFamilyMask::MAIN_PREPASS) {
            self.gpu_cpu_main_complement.push(extracted);
        }
        if cast_shadows
            && incomplete_families
                .intersects(GpuDrivenFamilyMask::SHADOW_2D | GpuDrivenFamilyMask::SHADOW_CUBE)
        {
            self.gpu_cpu_shadow_complement.push(extracted);
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn gpu_cpu_main_complement_len(&self) -> usize {
        self.gpu_cpu_main_complement.len()
    }

    #[inline]
    #[must_use]
    pub(crate) fn gpu_cpu_main_complement(&self, index: usize) -> ExtractedSourceId {
        self.gpu_cpu_main_complement[index]
    }

    #[inline]
    #[must_use]
    pub(crate) fn gpu_cpu_shadow_complement_len(&self) -> usize {
        self.gpu_cpu_shadow_complement.len()
    }

    #[inline]
    #[must_use]
    pub(crate) fn gpu_cpu_shadow_complement(&self, index: usize) -> ExtractedSourceId {
        self.gpu_cpu_shadow_complement[index]
    }

    #[inline]
    #[must_use]
    pub(crate) const fn gpu_cpu_complement_families(&self) -> GpuDrivenFamilyMask {
        self.gpu_cpu_complement_families
    }

    fn advance_gpu_opaque_representative_generation(&mut self) {
        self.gpu_opaque_representative_generation =
            self.gpu_opaque_representative_generation.wrapping_add(1);
        if self.gpu_opaque_representative_generation == 0 {
            self.gpu_opaque_representative_stamps.fill(0);
            self.gpu_opaque_representative_generation = 1;
        }
    }

    /// Resolve a compact frame source back to the immutable extraction table.
    #[must_use]
    pub(crate) fn extracted_source(&self, source: FrameSourceId) -> Option<ExtractedSourceId> {
        self.frame_sources.get(source.index()).copied()
    }

    #[must_use]
    pub(crate) fn frame_source_count(&self) -> usize {
        self.frame_sources.len()
    }

    /// Iterate all valid frame-local source IDs in dense order.
    pub(crate) fn frame_source_ids(&self) -> impl ExactSizeIterator<Item = FrameSourceId> + '_ {
        let count = u32::try_from(self.frame_sources.len())
            .expect("validated frame source count exceeds u32::MAX");
        (0..count).map(FrameSourceId)
    }

    /// Append one pre-frustum source used to prepare an opaque GPU bin's fixed
    /// state. Repeated semantic sources are idempotent in O(1) through the
    /// retained generation-stamped dense join and preserve stable
    /// first-occurrence order. This never inserts a CPU draw intent and
    /// therefore cannot cause duplicate opaque rendering.
    pub(crate) fn append_gpu_opaque_representative(
        &mut self,
        extracted: ExtractedSourceId,
    ) -> FrameSourceId {
        let source = self.intern_frame_source(extracted);
        let required = source.index() + 1;
        if self.gpu_opaque_representative_stamps.len() < required {
            self.gpu_opaque_representative_stamps.resize(required, 0);
        }
        let stamp = &mut self.gpu_opaque_representative_stamps[source.index()];
        if *stamp != self.gpu_opaque_representative_generation {
            *stamp = self.gpu_opaque_representative_generation;
            self.gpu_opaque_representatives.push(source);
        }
        source
    }

    /// Iterate retained GPU-bin representatives in deterministic insertion
    /// order.
    pub(crate) fn gpu_opaque_representatives(
        &self,
    ) -> impl ExactSizeIterator<Item = FrameSourceId> + '_ {
        self.gpu_opaque_representatives.iter().copied()
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

    /// Visit the frame sources selected by one DPE-live draw-plan key.
    ///
    /// The caller owns deduplication scratch. Keeping that scratch in the
    /// graph-attempt source coordinator avoids allocating and copying one
    /// temporary `Vec` for every warm frame while still letting several plan
    /// families share the same source union.
    pub(crate) fn for_each_source_for_key(
        &self,
        key: DrawPlanKey,
        mut visit: impl FnMut(FrameSourceId),
    ) {
        match key {
            DrawPlanKey::Opaque | DrawPlanKey::Prepass { .. } => {
                for intent in &self.opaque {
                    visit(intent.source);
                }
            }
            DrawPlanKey::Transparent => {
                for intent in &self.transparent {
                    visit(intent.source);
                }
            }
            DrawPlanKey::Shadow {
                light_id,
                layer_index,
            } => {
                if let Some(queue) = self.shadow_queues.get(&(light_id, layer_index)) {
                    for intent in queue {
                        visit(intent.source);
                    }
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
    last_render_world_sync_stats: RenderWorldSyncStats,
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
            last_render_world_sync_stats: RenderWorldSyncStats::default(),
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

    #[inline]
    pub(crate) const fn last_render_world_sync_stats(&self) -> RenderWorldSyncStats {
        self.last_render_world_sync_stats
    }

    /// Extract scene data and build shadow views without physical GPU work.
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
    /// 6. **Global Snapshot** — Update the CPU camera state consumed by the
    ///    post-DPE `GlobalStatePlan`.
    ///
    /// # Note
    ///
    /// Surface acquisition is deferred to `FrameComposer::render()` to
    /// minimise swap-chain buffer hold time.
    pub fn extract(
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
        published_pmrem_max_mip: Option<f32>,
    ) {
        use crate::core::view::RenderView;

        resource_manager.next_frame();

        // ── 1. Extract ─────────────────────────────────────────────────
        let render_world_stats = self.extracted_scene.extract_into(
            scene,
            camera,
            assets,
            f64::from(frame_time.time),
            max_extracted_lights,
            feature_config.gpu_driven != GpuDrivenMode::Off,
        );

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

        Self::append_shadow_views(
            &self.extracted_scene,
            camera,
            &mut render_lists.active_views,
        );

        // ── 5. Shadow map allocation is deferred to RDG ─────────────────
        // The physical shadow texture is now a transient RDG resource,
        // allocated by the graph compiler after topology compilation.
        // No `ensure_shadow_maps()` call is needed here.

        // ── 6. Update light storage buffer with shadow metadata ────────
        Self::update_light_shadow_metadata(&render_lists.active_views, &mut self.extracted_scene);

        // ── 7. Global CPU snapshot ─────────────────────────────────────
        self.render_state.update(
            scene.id(),
            camera,
            frame_time,
            resource_manager.frame_index(),
            surface_size,
        );

        self.last_render_world_sync_stats = render_world_stats;
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
    /// Append shadow views directly into caller-retained frame storage. Each
    /// directional light may produce multiple cascade views; spot lights
    /// produce one and point lights six cube faces.
    fn append_shadow_views(
        extracted_scene: &ExtractedScene,
        camera: &RenderCamera,
        active_views: &mut Vec<crate::core::view::RenderView>,
    ) {
        use myth_scene::light::LightKind;

        // Compute scene caster extent (for CSM Z extension) from the
        // persistent conservative caster union. RenderWorld expands every
        // caster AABB to its enclosing sphere box, so the farthest union
        // corner cannot underestimate the previous per-caster
        // `distance(center) + half_diagonal` bound.
        let camera_pos: glam::Vec3 = camera.position.to_array().into();
        let max_distance = extracted_scene
            .shadow_caster_bounds_union()
            .filter(myth_resources::BoundingBox::is_finite)
            .map_or(0.0, |bounds| {
                let farthest_delta = (bounds.min - camera_pos)
                    .abs()
                    .max((bounds.max - camera_pos).abs());
                farthest_delta.length()
            });
        let scene_caster_extent = max_distance.max(50.0);

        let shadow_view_start = active_views.len();

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
                    let base_layer =
                        u32::try_from(active_views.len() - shadow_view_start).unwrap_or(u32::MAX);

                    let _splits = shadow_utils::append_directional_views(
                        light.id,
                        light.direction,
                        light_buffer_index,
                        camera,
                        &shadow_cfg,
                        shadow_far,
                        caster_extension,
                        base_layer,
                        active_views,
                    );
                }
                LightKind::Spot(spot) => {
                    let base_layer =
                        u32::try_from(active_views.len() - shadow_view_start).unwrap_or(u32::MAX);
                    active_views.push(shadow_utils::build_spot_view(
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
                    let base_layer =
                        u32::try_from(active_views.len() - shadow_view_start).unwrap_or(u32::MAX);
                    shadow_utils::append_point_views(
                        light.id,
                        light_buffer_index,
                        light.position,
                        point,
                        &shadow_cfg,
                        base_layer,
                        active_views,
                    );
                }
            }
        }
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
mod shadow_metadata_tests {
    use super::*;
    use glam::{Vec3, Vec4};
    use myth_resources::{buffer::CpuBuffer, uniforms::GpuLightStorage};

    fn seed_prior_shadow_metadata(buffer: &CpuBuffer<Vec<GpuLightStorage>>) {
        let mut lights = buffer.write();
        for light in lights.iter_mut() {
            light.shadow_layer_index = 7;
            light.point_shadow_index = 3;
            light.shadow_bias = 0.125;
            light.shadow_normal_bias = 0.25;
            light.cascade_count = 2;
            light.cascade_splits = Vec4::new(2.0, 8.0, 0.0, 0.0);
            light.shadow_matrices.0 = [Mat4::from_translation(Vec3::ONE); 4];
        }
    }

    fn assert_default_light_payload(buffer: &CpuBuffer<Vec<GpuLightStorage>>) {
        assert!(
            buffer
                .read()
                .iter()
                .all(|light| *light == GpuLightStorage::default()),
            "no-shadow input synchronization must erase prior shadow metadata"
        );
    }

    #[test]
    fn empty_shadow_metadata_is_a_noop_after_current_light_payload_synchronization() {
        let mut scene = ExtractedScene::new();
        scene.prepare_gpu_scene_inputs(0.0);
        let local_before = scene.local_light_storage_buffer.stamp();
        let directional_before = scene.directional_light_storage_buffer.stamp();

        RenderFrame::update_light_shadow_metadata(&[], &mut scene);

        assert_eq!(scene.local_light_storage_buffer.stamp(), local_before);
        assert_eq!(
            scene.directional_light_storage_buffer.stamp(),
            directional_before
        );

        // On a shadow-to-no-shadow transition extraction first synchronizes
        // the current light payload. The empty metadata update must preserve
        // that synchronized stamp and must not resurrect stale shadow fields.
        seed_prior_shadow_metadata(&scene.local_light_storage_buffer);
        seed_prior_shadow_metadata(&scene.directional_light_storage_buffer);
        let stale_local = scene.local_light_storage_buffer.stamp();
        let stale_directional = scene.directional_light_storage_buffer.stamp();

        // This is the production order: extraction synchronizes the current
        // light payload before shadow views and their metadata are processed.
        scene.prepare_gpu_scene_inputs(0.0);
        assert_default_light_payload(&scene.local_light_storage_buffer);
        assert_default_light_payload(&scene.directional_light_storage_buffer);
        let synchronized_local = scene.local_light_storage_buffer.stamp();
        let synchronized_directional = scene.directional_light_storage_buffer.stamp();
        assert!(synchronized_local.version > stale_local.version);
        assert!(synchronized_directional.version > stale_directional.version);

        RenderFrame::update_light_shadow_metadata(&[], &mut scene);

        assert_eq!(scene.local_light_storage_buffer.stamp(), synchronized_local);
        assert_eq!(
            scene.directional_light_storage_buffer.stamp(),
            synchronized_directional
        );
        assert_default_light_payload(&scene.local_light_storage_buffer);
        assert_default_light_payload(&scene.directional_light_storage_buffer);
    }
}

#[cfg(test)]
mod draw_plan_tests {
    use super::*;
    use crate::settings::{RendererInitConfig, RendererSettings};
    use crate::test_support::SharedTestContext as WgpuContext;
    use slotmap::Key;

    fn test_command(sort_key: u64) -> DrawCommand {
        DrawCommand {
            frame_source: FrameSourceId::from_index(0),
            sort_key,
            pipeline_id: RenderPipelineId(0),
            vertex_buffer_range: 0..0,
            index_buffer: None,
            bind_group_1: None,
            bind_group_2: (DrawBindGroupSlot(0), 0),
            stencil_reference: None,
            vertex_range: 0..3,
            instance_range: 0..1,
        }
    }

    fn render_intent(source: usize) -> RenderIntent {
        RenderIntent {
            source: FrameSourceId::from_index(source),
            geometry_handle: GeometryHandle::null(),
            material_handle: MaterialHandle::null(),
            material_index: source as u32,
            distance_sq: source as f32,
        }
    }

    fn shadow_intent(source: usize) -> ShadowRenderIntent {
        ShadowRenderIntent {
            source: FrameSourceId::from_index(source),
            geometry_handle: GeometryHandle::null(),
            material_handle: MaterialHandle::null(),
        }
    }

    #[test]
    fn plan_ids_are_deduplicated_and_attempt_scoped() {
        let mut store = DrawPlanStore::new();
        let plan = store.reserve_plan(DrawPlanKey::Opaque);
        assert_eq!(store.reserve_plan(DrawPlanKey::Opaque), plan);
        store.mark_live(plan);
        let start = store.begin_finalize(plan);
        store.finish_finalize(plan, start);
        assert!(store.resolve_plan(plan).is_some());

        store.clear();
        assert!(store.resolve_plan(plan).is_none());
        assert!(!store.validate_reserved(plan));
        assert!(store.is_empty());
    }

    #[test]
    fn prepared_source_join_is_lazy_generation_scoped_and_trimmable() {
        let frame_source = FrameSourceId::from_index(37);
        let extracted_source = ExtractedSourceId::from_index(17);
        let mut prepared = PreparedDrawStore::new();

        let retained_before = prepared.capacity_usage().retained_bytes;
        let indices_len_before = prepared.source_join_indices.len();
        let stamps_len_before = prepared.source_join_stamps.len();
        prepared.begin_sources(0, 4096);
        assert_eq!(prepared.source_domain_len, 0);
        assert_eq!(prepared.source_join_indices.len(), indices_len_before);
        assert_eq!(prepared.source_join_stamps.len(), stamps_len_before);
        assert_eq!(prepared.capacity_usage().active_bytes, 0);
        assert_eq!(prepared.capacity_usage().retained_bytes, retained_before);
        assert_eq!(prepared.prepared_source_id(frame_source), None);
        prepared.clear();

        prepared.begin_sources(1, frame_source.index() + 1);
        let prepared_source = prepared.insert_source(PreparedDrawSource {
            frame_source,
            extracted_source,
            geometry: GeometryFrameId::from_index(3),
            material: MaterialFrameId::from_index(5),
            object_interface: ObjectInterfaceId::from_index(7),
            material_bind_group: DrawBindGroupSlot(11),
            object_bind_group: (DrawBindGroupSlot(13), 256),
            commits_morph_history: false,
        });

        let (joined_id, joined) = prepared
            .source_for_frame(frame_source)
            .expect("inserted frame source must join to its dense prepared record");
        assert_eq!(joined_id, prepared_source);
        assert_eq!(joined.frame_source, frame_source);
        assert_eq!(joined.extracted_source, extracted_source);
        assert!(
            std::ptr::eq(
                joined,
                prepared
                    .source(prepared_source)
                    .expect("dense prepared ID must remain directly addressable"),
            ),
            "Gather and later direct-index phases must observe one compact record"
        );
        assert!(
            prepared
                .source_for_frame(FrameSourceId::from_index(36))
                .is_none(),
            "a foreign frame identity must fail closed"
        );

        let retained_join_len = prepared.source_join_stamps.len();
        prepared.clear();
        assert_eq!(prepared.source_join_stamps.len(), retained_join_len);
        assert_eq!(prepared.prepared_source_id(frame_source), None);
        prepared.begin_sources(1, frame_source.index() + 1);
        let next = prepared.insert_source(PreparedDrawSource {
            frame_source,
            extracted_source,
            geometry: GeometryFrameId::from_index(3),
            material: MaterialFrameId::from_index(5),
            object_interface: ObjectInterfaceId::from_index(7),
            material_bind_group: DrawBindGroupSlot(11),
            object_bind_group: (DrawBindGroupSlot(13), 256),
            commits_morph_history: false,
        });
        assert_eq!(next, PreparedSourceId::from_index(0));
        assert_eq!(prepared.prepared_source_id(frame_source), Some(next));

        prepared.source_join_stamps.fill(1);
        prepared.source_join_generation = u32::MAX;
        prepared.clear();
        assert_eq!(prepared.source_join_generation, 1);
        assert!(
            prepared.source_join_stamps.iter().all(|&stamp| stamp == 0),
            "generation wrap must erase ancient prepared-source stamps once"
        );

        prepared.begin_sources(1, 4096);
        assert!(prepared.capacity_usage().active_bytes > 0);
        prepared.clear();

        let before = prepared.capacity_usage();
        assert_eq!(before.active_bytes, 0);
        assert!(prepared.source_join_stamps.len() >= 4096);
        prepared.trim_capacity(0, before.retained_bytes);

        assert!(prepared.source_join_stamps.len() <= PREPARED_DRAW_SOURCE_BASELINE);
        assert!(prepared.source_join_indices.len() <= PREPARED_DRAW_SOURCE_BASELINE);
        assert!(prepared.capacity_usage().retained_bytes < before.retained_bytes);
    }

    #[test]
    fn frame_source_interner_is_dense_generation_scoped_and_resettable() {
        let mut lists = RenderLists::new();
        let extracted_9 = ExtractedSourceId::from_index(9);
        let extracted_3 = ExtractedSourceId::from_index(3);

        let first = lists.intern_frame_source(extracted_9);
        let second = lists.intern_frame_source(extracted_3);
        let repeated = lists.intern_frame_source(extracted_9);

        assert_eq!(first, FrameSourceId::from_index(0));
        assert_eq!(second, FrameSourceId::from_index(1));
        assert_eq!(repeated, first);
        assert_eq!(lists.frame_source_count(), 2);
        assert_eq!(lists.extracted_source(first), Some(extracted_9));
        assert_eq!(lists.extracted_source(second), Some(extracted_3));
        assert_eq!(
            lists.frame_source_ids().collect::<Vec<_>>(),
            vec![first, second]
        );

        lists.begin_intent_collection();
        assert_eq!(lists.frame_source_count(), 0);
        assert_eq!(lists.extracted_source(first), None);
        assert_eq!(lists.frame_source_stamps.len(), 10);
        assert_ne!(
            lists.frame_source_stamps[extracted_9.index()],
            lists.frame_source_generation,
            "generation advance invalidates old slots without clearing them"
        );

        let reinterned = lists.intern_frame_source(extracted_9);
        assert_eq!(reinterned, FrameSourceId::from_index(0));
        assert_eq!(lists.extracted_source(reinterned), Some(extracted_9));

        // A generation wrap is the one cold-path full clear. Ancient stamps
        // must not alias the first generation after wrapping.
        let extracted = ExtractedSourceId::from_index(7);
        lists.frame_source_indices.resize(8, 0);
        lists.frame_source_stamps.resize(8, 1);
        lists.frame_source_generation = u32::MAX;

        lists.begin_intent_collection();

        assert_eq!(lists.frame_source_generation, 1);
        assert!(lists.frame_source_stamps.iter().all(|&stamp| stamp == 0));
        assert_eq!(
            lists.intern_frame_source(extracted),
            FrameSourceId::from_index(0),
            "a pre-wrap stamp must not alias the new generation"
        );
    }

    #[test]
    fn gpu_family_ownership_is_exact_generation_stamped_and_composable() {
        let mut lists = RenderLists::new();
        let selected = ExtractedSourceId::from_index(37);
        let cpu_only = ExtractedSourceId::from_index(5);

        lists.claim_gpu_family_ownership(selected, GpuDrivenFamilyMask::MAIN_PREPASS);
        lists.claim_gpu_family_ownership(selected, GpuDrivenFamilyMask::SHADOW_2D);
        assert_eq!(
            lists.gpu_family_ownership(selected),
            GpuDrivenFamilyMask::MAIN_PREPASS | GpuDrivenFamilyMask::SHADOW_2D
        );
        assert!(lists.gpu_family_ownership(cpu_only).is_empty());

        let mask_capacity = lists.gpu_family_ownership_masks.capacity();
        let stamp_capacity = lists.gpu_family_ownership_stamps.capacity();
        lists.begin_intent_collection();
        assert!(
            lists.gpu_family_ownership(selected).is_empty(),
            "a prior frame's ownership must fail closed to CpuDirect"
        );
        assert_eq!(lists.gpu_family_ownership_masks.capacity(), mask_capacity);
        assert_eq!(lists.gpu_family_ownership_stamps.capacity(), stamp_capacity);

        lists.gpu_family_ownership_stamps.fill(u32::MAX);
        lists.gpu_family_ownership_generation = u32::MAX;
        lists.begin_intent_collection();
        assert_eq!(lists.gpu_family_ownership_generation, 1);
        assert!(
            lists
                .gpu_family_ownership_stamps
                .iter()
                .all(|&stamp| stamp == 0),
            "only generation wrap may zero-fill the retained ownership table"
        );

        let mut duplicate_lists = RenderLists::new();
        let selected = ExtractedSourceId::from_index(3);
        duplicate_lists.claim_gpu_family_ownership(selected, GpuDrivenFamilyMask::SHADOW_CUBE);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            duplicate_lists.claim_gpu_family_ownership(selected, GpuDrivenFamilyMask::SHADOW_CUBE);
        }))
        .expect_err("one family cannot have two GPU owners");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic>");
        assert!(
            message.contains("cannot have two GPU owners"),
            "unexpected duplicate-owner failure: {message}"
        );
    }

    #[test]
    fn gpu_opaque_representatives_append_deduplicate_and_retain_capacity() {
        let mut lists = RenderLists::new();
        let initial_capacity = lists.gpu_opaque_representatives.capacity();
        let extracted_9 = ExtractedSourceId::from_index(9);
        let extracted_3 = ExtractedSourceId::from_index(3);

        let first = lists.append_gpu_opaque_representative(extracted_9);
        let second = lists.append_gpu_opaque_representative(extracted_3);
        let repeated = lists.append_gpu_opaque_representative(extracted_9);

        assert_eq!(repeated, first);
        assert_eq!(lists.gpu_opaque_representatives().next(), Some(first));
        assert_eq!(
            lists.gpu_opaque_representatives().collect::<Vec<_>>(),
            [first, second]
        );
        assert_eq!(lists.frame_source_count(), 2);
        let populated_capacity = lists.gpu_opaque_representatives.capacity();
        assert!(populated_capacity >= initial_capacity);
        let stamps_before_reset = lists.gpu_opaque_representative_stamps.clone();
        let generation_before_reset = lists.gpu_opaque_representative_generation;

        lists.begin_intent_collection();
        assert!(lists.gpu_opaque_representatives().next().is_none());
        assert_eq!(
            lists.gpu_opaque_representative_stamps, stamps_before_reset,
            "ordinary warm reset must not fill retained representative stamps"
        );
        assert_eq!(
            lists.gpu_opaque_representative_generation,
            generation_before_reset + 1
        );
        assert_eq!(
            lists.gpu_opaque_representatives.capacity(),
            populated_capacity,
            "frame reset must retain representative-bin allocation"
        );

        let mut wrap_lists = RenderLists::new();
        wrap_lists.gpu_opaque_representative_stamps = vec![u32::MAX; 32];
        wrap_lists.gpu_opaque_representative_generation = u32::MAX;

        wrap_lists.begin_intent_collection();

        assert_eq!(wrap_lists.gpu_opaque_representative_generation, 1);
        assert!(
            wrap_lists
                .gpu_opaque_representative_stamps
                .iter()
                .all(|&stamp| stamp == 0)
        );
        let source = wrap_lists.append_gpu_opaque_representative(ExtractedSourceId::from_index(31));
        assert_eq!(source, FrameSourceId::from_index(0));
        assert_eq!(wrap_lists.gpu_opaque_representative_stamps[0], 1);
    }

    #[test]
    fn morph_history_storage_is_cold_until_needed_and_generation_deduplicated() {
        let mut store = DrawPlanStore::new();
        assert!(store.emitted_morph_history_sources().is_empty());
        assert_eq!(store.morph_history_sources.capacity(), 0);
        assert_eq!(store.morph_history_source_stamps.capacity(), 0);

        let plan = store.reserve_plan(DrawPlanKey::Opaque);
        store.mark_live(plan);
        let start = store.begin_finalize(plan);
        store.push_command(test_command(0), &[]);
        store.finish_finalize(plan, start);

        assert!(store.emitted_morph_history_sources().is_empty());
        assert_eq!(store.morph_history_sources.capacity(), 0);
        assert_eq!(store.morph_history_source_stamps.capacity(), 0);

        let source_37 = FrameSourceId::from_index(37);
        let source_5 = FrameSourceId::from_index(5);

        store.record_emitted_morph_history_source(source_37);
        store.record_emitted_morph_history_source(source_5);
        store.record_emitted_morph_history_source(source_37);
        assert_eq!(
            store.emitted_morph_history_sources(),
            &[source_37, source_5],
            "main/prepass/shadow aliases must commit a source only once"
        );
        let retained_stamp_len = store.morph_history_source_stamps.len();

        store.clear();
        assert!(store.emitted_morph_history_sources().is_empty());
        assert_eq!(store.morph_history_source_stamps.len(), retained_stamp_len);
        store.record_emitted_morph_history_source(source_37);
        assert_eq!(store.emitted_morph_history_sources(), &[source_37]);

        store.morph_history_source_stamps.fill(1);
        store.morph_history_source_generation = u32::MAX;
        store.clear();
        assert_eq!(store.morph_history_source_generation, 1);
        assert!(
            store
                .morph_history_source_stamps
                .iter()
                .all(|&stamp| stamp == 0),
            "generation wrap must erase ancient morph-history stamps once"
        );
        assert_eq!(store.morph_history_active_domain_len, 0);
    }

    #[test]
    fn structured_stream_observation_normalizes_every_field_in_plan_reservation_order() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let vertex_a = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("observation vertex A"),
            size: 32,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let vertex_b_and_index = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("observation vertex B/index"),
            size: 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });
        let layout = wgpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("observation bind-group layout"),
                entries: &[],
            });
        let material_bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("observation material bind group"),
            layout: &layout,
            entries: &[],
        });
        let object_bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("observation object bind group"),
            layout: &layout,
            entries: &[],
        });

        let mut store = DrawPlanStore::new();
        // Deliberately intern owners in an order unrelated to binding order.
        // The observation must expose stable physical IDs, never these slots.
        let secondary_buffer_slot = store.intern_buffer(&vertex_b_and_index, 202);
        let primary_buffer_slot = store.intern_buffer(&vertex_a, 101);
        let object_bind_group_slot = store.intern_bind_group(&object_bind_group, 404);
        let material_bind_group_slot = store.intern_bind_group(&material_bind_group, 303);

        let transparent = store.reserve_plan(DrawPlanKey::Transparent);
        assert_eq!(store.reserve_plan(DrawPlanKey::Transparent), transparent);
        let dead = store.reserve_plan(DrawPlanKey::Shadow {
            light_id: 9,
            layer_index: 2,
        });
        let opaque = store.reserve_plan(DrawPlanKey::Opaque);
        store.mark_live(transparent);
        store.mark_live(opaque);
        assert_eq!(store.plan_count(), 3);
        assert!(!store.is_live(dead));

        // Finalize out of reservation order. The observation remains keyed by
        // canonical plan order, and a finalized empty stream remains visible.
        let opaque_start = store.begin_finalize(opaque);
        store.finish_finalize(opaque, opaque_start);

        let transparent_start = store.begin_finalize(transparent);
        store.push_command(
            DrawCommand {
                frame_source: FrameSourceId::from_index(8),
                sort_key: 90,
                pipeline_id: RenderPipelineId(19),
                vertex_buffer_range: 0..0,
                index_buffer: Some((secondary_buffer_slot, wgpu::IndexFormat::Uint32, 36)),
                bind_group_1: Some(material_bind_group_slot),
                bind_group_2: (object_bind_group_slot, 256),
                stencil_reference: Some(7),
                vertex_range: 2..9,
                instance_range: 3..5,
            },
            &[
                primary_buffer_slot,
                secondary_buffer_slot,
                primary_buffer_slot,
            ],
        );
        store.push_command(
            DrawCommand {
                frame_source: FrameSourceId::from_index(2),
                sort_key: 10,
                pipeline_id: RenderPipelineId(3),
                vertex_buffer_range: 0..0,
                index_buffer: None,
                bind_group_1: None,
                bind_group_2: (object_bind_group_slot, 512),
                stencil_reference: None,
                vertex_range: 4..12,
                instance_range: 0..2,
            },
            &[secondary_buffer_slot],
        );
        store.sort_commands_from(transparent_start);
        store.finish_finalize(transparent, transparent_start);

        assert_eq!(store.buffer_owner_count(), 2);
        assert_eq!(store.bind_group_owner_count(), 2);
        assert!(
            store
                .resolve_plan(opaque)
                .is_some_and(<[DrawCommand]>::is_empty)
        );
        assert!(store.resolve_plan(dead).is_none());
        assert_eq!(
            store.plan_meta(2),
            Some(DrawPlanMeta {
                id: opaque,
                key: DrawPlanKey::Opaque,
                is_live: true,
                is_finalized: true,
            })
        );
        assert_eq!(
            store.observe_live_streams(),
            vec![
                DrawStreamObservation {
                    plan_order: 0,
                    plan_key: DrawPlanKey::Transparent,
                    commands: vec![
                        DrawCommandObservation {
                            frame_source: FrameSourceId::from_index(2),
                            sort_key: 10,
                            pipeline_id: RenderPipelineId(3),
                            vertex_buffer_ids: smallvec::smallvec![202],
                            index_buffer: None,
                            material_bind_group_id: None,
                            object_bind_group_id: 404,
                            object_dynamic_offset: 512,
                            stencil_reference: None,
                            vertex_range: 4..12,
                            instance_range: 0..2,
                        },
                        DrawCommandObservation {
                            frame_source: FrameSourceId::from_index(8),
                            sort_key: 90,
                            pipeline_id: RenderPipelineId(19),
                            vertex_buffer_ids: smallvec::smallvec![101, 202, 101],
                            index_buffer: Some(DrawIndexBufferObservation {
                                buffer_id: 202,
                                format: wgpu::IndexFormat::Uint32,
                                index_count: 36,
                            }),
                            material_bind_group_id: Some(303),
                            object_bind_group_id: 404,
                            object_dynamic_offset: 256,
                            stencil_reference: Some(7),
                            vertex_range: 2..9,
                            instance_range: 3..5,
                        },
                    ],
                },
                DrawStreamObservation {
                    plan_order: 2,
                    plan_key: DrawPlanKey::Opaque,
                    commands: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn live_plan_source_union_is_exact_and_deduplicated() {
        let mut lists = RenderLists::new();
        for extracted in 0..=5 {
            lists.intern_frame_source(ExtractedSourceId::from_index(extracted));
        }
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
        let mut source_prepare = crate::graph::scene_source_prepare::SceneSourcePrepareStore::new();
        source_prepare.select_live_sources(&lists, plans.live_keys());
        assert_eq!(
            source_prepare.live_sources(),
            vec![
                FrameSourceId::from_index(1),
                FrameSourceId::from_index(2),
                FrameSourceId::from_index(4)
            ],
            "Prepass must select opaque sources, the exact live shadow must add only its queue, and shared source 2 must be deduplicated"
        );
    }
}
