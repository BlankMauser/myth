//! Render frame management.
//!
//! `RenderFrame` is responsible for:
//! - Holding extracted scene data ([`ExtractedScene`]) and render state ([`RenderState`])
//! - Running the Extract and Prepare phases of the frame pipeline
//!
//! # Three-Phase Rendering Architecture
//!
//! The rendering pipeline is divided into three distinct phases:
//!
//! 1. **Prepare**: `extract_and_prepare()` — extract scene data and prepare GPU resources
//! 2. **Compose**: Chain render nodes via [`FrameComposer`]
//! 3. **Execute**: `FrameComposer::render()` — acquire the surface and submit GPU commands
//!

use glam::Mat4;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::gpu::Tracked;
use crate::core::{BindGroupContext, RenderView, ResourceManager};
use crate::pipeline::RenderPipelineId;
use crate::renderer::FrameTime;
use crate::settings::{ClusteredShadingMode, RenderPath};
use myth_assets::{AssetServer, GeometryHandle, MaterialHandle};
use myth_resources::buffer::CpuBuffer;
use myth_scene::Scene;
use myth_scene::camera::RenderCamera;

use super::extracted::{ExtractedScene, SceneFeatures};
use super::render_state::RenderState;
use super::shadow_utils;

// ============================================================================
// RenderCommand & RenderLists
// ============================================================================

/// A single render command.
///
/// Contains all information needed to draw one object. Produced by `CullPass`,
/// consumed by `OpaquePass` / `TransparentPass`.
///
/// # Performance Notes
/// - Pipeline is obtained via `clone` (`wgpu::RenderPipeline` is internally `Arc`)
/// - `dynamic_offset` supports dynamic uniform buffering
/// - `sort_key` enables efficient sorting (front-to-back / back-to-front)
pub struct RenderCommand {
    /// Per-object bind group (model matrix, skeleton, etc.)
    pub object_bind_group: BindGroupContext,
    /// Geometry handle
    pub geometry_handle: GeometryHandle,
    /// Material handle
    pub material_handle: MaterialHandle,
    /// Pipeline handle (index into [`PipelineCache`] storage).
    ///
    /// Resolve to a `&wgpu::RenderPipeline` via
    /// [`PipelineCache::get_render_pipeline`] during the execute phase.
    pub pipeline_id: RenderPipelineId,
    /// Sort key
    pub sort_key: RenderKey,
    /// Dynamic uniform offset
    pub dynamic_offset: u32,
}

pub struct ShadowRenderCommand {
    pub object_bind_group: BindGroupContext,
    pub geometry_handle: GeometryHandle,
    pub material_handle: MaterialHandle,
    /// Pipeline handle (index into [`PipelineCache`] storage).
    pub pipeline_id: RenderPipelineId,
    pub dynamic_offset: u32,
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

/// A single pre-baked draw command with all GPU resources resolved to
/// frame-local physical-resource slots.
///
/// Produced by [`bake::bake_render_lists`] after culling, consumed by the
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
/// bake** phase.  The execute phase sees only hardware-level GPU state and
/// can be trivially parallelised in the future.
///
/// [`bake::bake_render_lists`]: super::bake::bake_render_lists
/// [`submit_draw_commands`]: super::rdg::draw::submit_draw_commands
pub struct DrawCommand {
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

/// Graph-attempt-scoped pre-baked draw plans.
///
/// All GPU handle lookups have been resolved to owned physical WGPU clones.
/// [`GraphStorage`](super::core::GraphStorage) owns this store and clears it
/// on both successful completion and abandoned graph attempts.
pub struct DrawPlanStore {
    commands: Vec<DrawCommand>,
    plans: Vec<std::ops::Range<u32>>,
    opaque_plan: Option<DrawPlanId>,
    transparent_plan: Option<DrawPlanId>,
    prepass_plan: Option<DrawPlanId>,
    shadow_plans: FxHashMap<(u64, u32), DrawPlanId>,
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
            opaque_plan: None,
            transparent_plan: None,
            prepass_plan: None,
            shadow_plans: FxHashMap::default(),
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
        self.opaque_plan = None;
        self.transparent_plan = None;
        self.prepass_plan = None;
        self.shadow_plans.clear();
        self.buffer_lookup.clear();
        self.buffers.clear();
        self.bind_group_lookup.clear();
        self.bind_groups.clear();
        self.attempt_token = next_draw_plan_attempt();
    }

    pub(crate) fn reserve_commands(&mut self, additional: usize) {
        self.commands.reserve(additional);
    }

    pub(crate) fn begin_plan(&self) -> u32 {
        u32::try_from(self.commands.len())
            .expect("one graph attempt cannot bake more than u32::MAX draw commands")
    }

    pub(crate) fn push_command(&mut self, command: DrawCommand) {
        self.commands.push(command);
    }

    pub(crate) fn sort_commands_from(&mut self, start: u32) {
        self.commands[start as usize..].sort_unstable_by_key(|command| command.sort_key);
    }

    pub(crate) fn finish_plan(&mut self, start: u32) -> DrawPlanId {
        let end = u32::try_from(self.commands.len())
            .expect("one graph attempt cannot bake more than u32::MAX draw commands");
        let index = u32::try_from(self.plans.len())
            .expect("one graph attempt cannot bake more than u32::MAX draw plans");
        self.plans.push(start..end);
        DrawPlanId {
            attempt_token: self.attempt_token,
            index,
        }
    }

    pub(crate) fn set_opaque_plan(&mut self, plan: DrawPlanId) {
        debug_assert!(self.resolve_plan(plan).is_some());
        self.opaque_plan = Some(plan);
    }

    pub(crate) fn set_transparent_plan(&mut self, plan: DrawPlanId) {
        debug_assert!(self.resolve_plan(plan).is_some());
        self.transparent_plan = Some(plan);
    }

    pub(crate) fn set_prepass_plan(&mut self, plan: DrawPlanId) {
        debug_assert!(self.resolve_plan(plan).is_some());
        self.prepass_plan = Some(plan);
    }

    pub(crate) fn insert_shadow_plan(&mut self, key: (u64, u32), plan: DrawPlanId) {
        debug_assert!(self.resolve_plan(plan).is_some());
        let previous = self.shadow_plans.insert(key, plan);
        debug_assert!(previous.is_none(), "shadow draw plan staged twice");
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
        if id.attempt_token != self.attempt_token {
            return None;
        }
        let range = self.plans.get(id.index as usize)?;
        Some(&self.commands[range.start as usize..range.end as usize])
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
            && self.opaque_plan.is_none()
            && self.transparent_plan.is_none()
            && self.prepass_plan.is_none()
            && self.shadow_plans.is_empty()
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

/// Narrow immutable execute-time view over one graph attempt's draw plans.
#[derive(Clone, Copy)]
pub struct DrawPlanResolver<'a> {
    store: &'a DrawPlanStore,
}

impl<'a> DrawPlanResolver<'a> {
    #[must_use]
    pub(crate) fn opaque_plan(self) -> DrawPlanId {
        self.store
            .opaque_plan
            .expect("opaque draw plan was not baked")
    }

    #[must_use]
    pub(crate) fn transparent_plan(self) -> DrawPlanId {
        self.store
            .transparent_plan
            .expect("transparent draw plan was not baked")
    }

    #[must_use]
    pub(crate) fn prepass_plan(self) -> DrawPlanId {
        self.store
            .prepass_plan
            .expect("prepass draw plan was not baked")
    }

    #[must_use]
    pub(crate) fn shadow_plan(self, key: (u64, u32)) -> Option<DrawPlanId> {
        self.store.shadow_plans.get(&key).copied()
    }

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

/// Render lists.
///
/// Stores culled and sorted render commands. Populated by `SceneCullPass`,
/// consumed by `OpaquePass`, `TransparentPass`, and `SimpleForwardPass`.
///
/// # Design Principles
/// - **Data-logic separation**: stores data only, contains no rendering logic
/// - **Inter-frame reuse**: pre-allocated memory, cleared via `clear()` each frame
/// - **Extensible**: can add `alpha_test`, `shadow_casters`, etc. in the future
pub struct RenderLists {
    /// Opaque command list (front-to-back sorted)
    pub opaque: Vec<RenderCommand>,
    /// Transparent command list (back-to-front sorted)
    pub transparent: Vec<RenderCommand>,
    /// Shadow command queues, keyed by `(light_id, layer_index)` for per-view culling.
    ///
    /// Each cascade of a directional light (or each spot light) gets its own queue.
    pub shadow_queues: FxHashMap<(u64, u32), Vec<ShadowRenderCommand>>,

    /// All active render views for the current frame.
    ///
    /// Populated by `SceneCullPass`, consumed by `ShadowPass` and other passes.
    /// Contains main camera view + all shadow views.
    pub active_views: Vec<RenderView>,

    /// Global bind group (camera, lighting, environment, etc.)
    pub gpu_global_bind_group: Option<wgpu::BindGroup>,

    /// Stable per-frame scene-light metadata buffer for RDG imports.
    pub gpu_local_light_metadata_buffer: Option<Tracked<wgpu::Buffer>>,

    /// Stable per-frame scene-light storage buffer for RDG imports.
    pub gpu_local_light_storage_buffer: Option<Tracked<wgpu::Buffer>>,

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
            gpu_global_bind_group: None,
            gpu_local_light_metadata_buffer: None,
            gpu_local_light_storage_buffer: None,
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
        self.gpu_global_bind_group = None;
        self.gpu_local_light_metadata_buffer = None;
        self.gpu_local_light_storage_buffer = None;
        self.use_transmission = false;
    }

    /// Inserts an opaque render command.
    #[inline]
    pub fn insert_opaque(&mut self, cmd: RenderCommand) {
        self.opaque.push(cmd);
    }

    /// Inserts a transparent render command.
    #[inline]
    pub fn insert_transparent(&mut self, cmd: RenderCommand) {
        self.transparent.push(cmd);
    }

    /// Sorts command lists.
    ///
    /// - Opaque: by Pipeline > Material > Depth (front-to-back)
    /// - Transparent: by Depth (back-to-front) > Pipeline > Material
    pub fn sort(&mut self) {
        self.opaque.sort_unstable_by_key(|a| a.sort_key);
        self.transparent.sort_unstable_by_key(|a| a.sort_key);
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
        self.extracted_scene.extract_into(
            scene,
            camera,
            assets,
            resource_manager,
            max_extracted_lights,
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
        Self::update_light_shadow_metadata(
            &render_lists.active_views,
            &mut self.extracted_scene,
            resource_manager,
        );

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
        resource_manager.ensure_buffer(&self.extracted_scene.local_light_metadata_buffer);
        resource_manager.ensure_buffer(&self.extracted_scene.local_light_storage_buffer);
        render_lists.gpu_local_light_metadata_buffer = tracked_cpu_buffer(
            resource_manager,
            &self.extracted_scene.local_light_metadata_buffer,
        );
        render_lists.gpu_local_light_storage_buffer = tracked_cpu_buffer(
            resource_manager,
            &self.extracted_scene.local_light_storage_buffer,
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
        resource_manager: &mut ResourceManager,
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
            // Keep the normal ensure path for first-frame materialization,
            // resource residency and stable external RDG imports.
            resource_manager.ensure_buffer(&extracted_scene.local_light_storage_buffer);
            resource_manager.ensure_buffer(&extracted_scene.directional_light_storage_buffer);
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

        resource_manager.ensure_buffer(&extracted_scene.local_light_storage_buffer);
        resource_manager.ensure_buffer(&extracted_scene.directional_light_storage_buffer);
    }

    /// Periodically prune stale resources.
    pub fn maybe_prune(&self, resource_manager: &mut ResourceManager) {
        // Periodic cleanup (TODO: LRU eviction strategy)
        if resource_manager.frame_index().is_multiple_of(600) {
            resource_manager.prune(6000);
        }
    }
}

fn tracked_cpu_buffer<T: myth_resources::buffer::GpuData>(
    resource_manager: &ResourceManager,
    cpu_buffer: &CpuBuffer<T>,
) -> Option<Tracked<wgpu::Buffer>> {
    resource_manager
        .get_gpu_buffer_by_cpu_id(cpu_buffer.id())
        .map(|gpu_buffer| Tracked::with_id(gpu_buffer.buffer.clone(), gpu_buffer.id))
}

#[cfg(test)]
mod draw_plan_tests {
    use super::*;

    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    fn assert_copy<T: Copy>() {}

    #[test]
    fn draw_plan_types_do_not_borrow_frame_resources() {
        assert_send_sync_static::<DrawCommand>();
        assert_send_sync_static::<DrawPlanId>();
        assert_send_sync_static::<DrawPlanStore>();
        assert_copy::<DrawPlanId>();
        assert!(std::mem::needs_drop::<DrawPlanStore>());
    }

    #[test]
    fn clearing_store_invalidates_attempt_scoped_plan_ids() {
        let mut store = DrawPlanStore::new();
        let start = store.begin_plan();
        let plan = store.finish_plan(start);

        assert!(store.resolve_plan(plan).is_some());
        store.clear();
        assert!(store.resolve_plan(plan).is_none());
        assert!(store.is_empty());
    }
}
