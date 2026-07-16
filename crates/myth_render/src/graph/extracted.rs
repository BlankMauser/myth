//! Render Extract Phase
//!
//! Before rendering begins, extract the minimal CPU data needed for the current
//! frame. Physical draw preparation is a later explicit step while Composer
//! still owns the Scene borrow.
//!
//! # Design Principles
//! - Only copy "minimal data" needed for rendering, don't copy actual Mesh/Material resources
//! - Extract ALL active meshes without frustum culling (culling is deferred to the Cull phase)
//! - Use Copy types to minimize overhead
//! - Carry cache IDs to avoid repeated lookups each frame
//! - Single source of truth: one `render_items` list consumed by all `RenderView`s

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, Ordering};

use glam::{Mat4, Vec3};

use crate::graph::frame::{DrawPlanStore, RenderLists};
use crate::graph::gpu_driven_policy::GpuDrivenWorkloadSynopsis;
use crate::graph::render_world::{
    PersistentRenderWorld, RenderWorldChangeFlags, RenderWorldConsumerCursor, RenderWorldDelta,
    RenderWorldFullRebuildReason, RenderWorldSourceUpdate, RenderWorldSyncPlan,
    RenderWorldSyncStats, node_order,
};
use bitflags::{Flags, bitflags};
use myth_assets::storage::AssetEntry;
use myth_assets::{AssetServer, GeometryHandle, MaterialHandle};
use myth_resources::buffer::CpuBuffer;
use myth_resources::geometry::Geometry;
use myth_resources::material::Material;
use myth_resources::material::{AlphaMode, MaterialType};
use myth_resources::shader_defines::ShaderDefines;
use myth_resources::uniforms::{EnvironmentUniforms, GpuLightStorage, LightBufferMetadata};
use myth_resources::{BoundingBox, RenderableMaterialTrait};
use myth_scene::background::BackgroundMode;
use myth_scene::camera::RenderCamera;
use myth_scene::environment::Environment;
use myth_scene::light::{LightKind, ShadowConfig, SpotLight};
use myth_scene::{NodeHandle, Scene, SkeletonKey};
use rustc_hash::FxHashMap;
use slotmap::Key;

const SPOT_TIGHT_SPHERE_COS_THRESHOLD: f32 = std::f32::consts::FRAC_1_SQRT_2; // ~ 0.70710678, corresponds to a 90-degree outer cone angle
const LIGHT_SCORE_DISTANCE_FLOOR: f32 = 1.0;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub(crate) struct MaterialClassification: u8 {
        const TRANSPARENT = 1 << 0;
        const USES_TRANSMISSION = 1 << 1;
    }
}

bitflags! {
    /// Canonical two-bit object shader variant captured during extraction.
    ///
    /// These are the only item-local defines consumed by scene pipelines.
    /// Keeping this POD selector on every item avoids allocating, hashing, and
    /// later dropping one `ShaderDefines` Vec per extracted object.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct ItemShaderFeatures: u8 {
        const HAS_SKINNING = 1 << 0;
        const RECEIVE_SHADOWS = 1 << 1;
    }
}

struct ItemShaderVariant {
    defines: ShaderDefines,
}

static ITEM_SHADER_VARIANTS: LazyLock<[ItemShaderVariant; 4]> = LazyLock::new(|| {
    std::array::from_fn(|index| {
        let features = ItemShaderFeatures::from_bits_retain(index as u8);
        let mut defines = ShaderDefines::with_capacity(features.bits().count_ones() as usize);
        if features.contains(ItemShaderFeatures::HAS_SKINNING) {
            defines.set("HAS_SKINNING", "1");
        }
        if features.contains(ItemShaderFeatures::RECEIVE_SHADOWS) {
            defines.set("RECEIVE_SHADOWS", "1");
        }
        ItemShaderVariant { defines }
    })
});

impl ItemShaderFeatures {
    #[inline]
    #[must_use]
    pub fn shader_defines(self) -> &'static ShaderDefines {
        // Only the two shader-semantic bits participate in the canonical
        // variant index. `bitflags` deliberately permits callers to retain
        // unknown bits, and those must not turn this public selector into an
        // out-of-bounds table access.
        let index = usize::from(self.contains(Self::HAS_SKINNING))
            | (usize::from(self.contains(Self::RECEIVE_SHADOWS)) << 1);
        &ITEM_SHADER_VARIANTS[index].defines
    }
}

/// Minimal CPU-only render item.
///
/// Contains all per-object attributes needed for view-independent filtering and culling.
#[derive(Clone)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the CPU snapshot preserves independent exact scene/asset eligibility facts"
)]
pub struct ExtractedRenderItem {
    /// Node handle (for debugging and cache write-back)
    pub node_handle: NodeHandle,
    /// World transform matrix (64 bytes)
    pub world_matrix: Mat4,

    /// Previous frame's world transform matrix (64 bytes)
    pub prev_world_matrix: Mat4,

    /// Geometry handle (8 bytes)
    pub geometry: GeometryHandle,
    /// Material handle (8 bytes)
    pub material: MaterialHandle,
    /// Exact loaded asset stamps captured with the CPU snapshot. Public
    /// versions alone can ABA when a reserved slot is repopulated.
    pub(crate) geometry_asset_version: u32,
    pub(crate) geometry_asset_identity: u64,
    /// Coherent geometry facts consumed by the pre-frustum GPU-driven
    /// selector. Backend selection must not reopen the live asset dictionary.
    pub(crate) geometry_structure_version: u64,
    pub(crate) geometry_layout_version: u64,
    pub(crate) geometry_topology: wgpu::PrimitiveTopology,
    pub(crate) geometry_index_format: Option<wgpu::IndexFormat>,
    pub(crate) geometry_index_count: u32,
    pub(crate) geometry_has_morph_targets: bool,
    pub(crate) geometry_has_vertex_attributes: bool,
    pub(crate) geometry_has_instanced_attributes: bool,
    pub(crate) material_asset_version: u32,
    pub(crate) material_asset_identity: u64,
    /// Interior material epoch captured with classification-relevant state.
    pub(crate) material_version: u64,
    pub(crate) material_gpu_scene_compatible: bool,

    /// Copy-only draw classification captured coherently with
    /// `material_version`. Culling consumes these fields without reopening
    /// the live asset dictionaries; MaterialPlan capture remains the
    /// fail-closed authority for a material that drifts after extraction.
    pub(crate) material_index: u32,
    pub(crate) material_classification: MaterialClassification,

    /// Exact skeleton selected while the Scene is still borrowed.
    pub(crate) skeleton: Option<SkeletonKey>,

    pub item_variant_flags: u32,

    pub item_shader_features: ItemShaderFeatures,

    pub cast_shadows: bool,
    pub receive_shadows: bool,

    /// World-space axis-aligned bounding box.
    pub world_aabb: BoundingBox,
}

impl ExtractedRenderItem {
    #[inline]
    #[must_use]
    pub fn item_shader_defines(&self) -> &'static ShaderDefines {
        self.item_shader_features.shader_defines()
    }
}

#[derive(Clone)]
pub struct ExtractedLight {
    pub id: u64,
    pub color: Vec3,
    pub intensity: f32,
    pub flags: u32,
    pub cast_shadows: bool,
    pub kind: LightKind,
    pub position: Vec3,
    pub direction: Vec3,
    pub shadow: Option<ShadowConfig>,
}

impl ExtractedLight {
    #[inline]
    #[must_use]
    pub fn is_directional(&self) -> bool {
        matches!(self.kind, LightKind::Directional(_))
    }

    #[inline]
    #[must_use]
    pub fn local_range(&self) -> f32 {
        match &self.kind {
            LightKind::Point(point) => point.range,
            LightKind::Spot(spot) => spot.range,
            LightKind::Directional(_) => 0.0,
        }
    }
}

/// Extracted skeleton data
#[derive(Clone)]
pub struct ExtractedSkeleton {
    pub skeleton_key: SkeletonKey,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SceneFeatures: u32 {
        const HAS_SHADOWS = 1 << 0;
        const USE_SSAO = 1 << 1;
        const USE_SSS = 1 << 2;
        const USE_SSR = 1 << 3;
        const USE_CLUSTERED_SHADING = 1 << 4;
        const USE_SSGI = 1 << 5;


        const USE_SCREEN_SPACE_FEATURES = Self::USE_SSS.bits() | Self::USE_SSR.bits();
    }
}

/// Extracted scene data
///
/// This is a lightweight structure containing only the minimal dataset needed
/// for current frame rendering. Physical draw state is kept in a separate,
/// parallel store and populated only when Composer proceeds with rendering.
///
/// # "Single Source of Truth" Design
///
/// `render_items` is the **single, unified list** of all active renderables.
/// No frustum culling is performed during extraction — that is deferred to the
/// Cull phase where each `RenderView` (main camera, shadow cascades, etc.)
/// performs its own culling against this shared list.
pub struct ExtractedScene {
    /// All active render items (NOT frustum-culled).
    /// Each `RenderView` in the Cull phase filters and culls from this list.
    pub render_items: Vec<ExtractedRenderItem>,
    /// Cross-frame CPU authority for the dense render-item table. Scene
    /// journal acknowledgement and GPU-consumer acknowledgement are separate
    /// transactions; abandoned graph attempts therefore cannot lose deltas.
    render_world: PersistentRenderWorld,
    render_world_pending: FxHashMap<NodeHandle, myth_scene::SceneMutationFlags>,
    render_world_hierarchy_stack: Vec<NodeHandle>,
    render_world_ordered_changes: Vec<(NodeHandle, myth_scene::SceneMutationFlags)>,
    render_world_changed_sources: Vec<RenderWorldSourceUpdate>,
    submitted_previous_scratch: FxHashMap<NodeHandle, Mat4>,
    submitted_world_history: SubmittedWorldHistoryMarker,
    render_world_scratch_underutilized_frames: u32,
    /// Whether the current snapshot carries the extra coherent topology and
    /// resource facts needed by GPU-driven preflight. The default CpuDirect
    /// path deliberately leaves these fields inert.
    gpu_driven_facts_captured: bool,
    gpu_driven_synopsis: GpuDrivenWorkloadSynopsis,
    /// Scene's shader macro definitions
    pub scene_id: u32,
    pub scene_variants: SceneFeatures,
    pub scene_defines: ShaderDefines,
    pub background: BackgroundMode,
    pub envvironment: Environment,
    pub has_transmission: bool,
    pub lights: Vec<ExtractedLight>,
    /// Number of local point / spot lights occupying the head segment of `lights`.
    local_light_count: usize,
    /// GPU-visible directional-light payload for the current frame.
    pub directional_light_storage_buffer: CpuBuffer<Vec<GpuLightStorage>>,
    /// GPU-visible local-light payload for the current frame.
    pub local_light_storage_buffer: CpuBuffer<Vec<GpuLightStorage>>,
    /// GPU-visible local-light metadata used by RDG scene-light bindings.
    pub local_light_metadata_buffer: CpuBuffer<LightBufferMetadata>,
    /// GPU-visible environment uniforms for the current frame.
    pub environment_uniforms_buffer: CpuBuffer<EnvironmentUniforms>,

    local_light_upload_cache: Vec<GpuLightStorage>,
    directional_light_upload_cache: Vec<GpuLightStorage>,
}

#[derive(Debug, Clone, Copy, Default)]
struct RenderItemExtractionStatus {
    has_unresolved_assets: bool,
    has_unstable_authority: bool,
    items_examined: usize,
}

const SUBMITTED_HISTORY_EMPTY: u32 = 0;
const SUBMITTED_HISTORY_WRITING: u32 = 1;
const SUBMITTED_HISTORY_READY: u32 = 2;

/// Wasm-compatible single-slot publication for the post-submit history token.
/// The state word protects the split revision fields; AtomicU64 is avoided so
/// this remains available on wasm32 targets without native 64-bit atomics.
#[derive(Default)]
struct SubmittedWorldHistoryMarker {
    state: AtomicU32,
    scene_id: AtomicU32,
    revision_low: AtomicU32,
    revision_high: AtomicU32,
}

impl SubmittedWorldHistoryMarker {
    fn publish(&self, cursor: RenderWorldConsumerCursor) {
        loop {
            match self.state.load(Ordering::Acquire) {
                SUBMITTED_HISTORY_EMPTY => {
                    if self
                        .state
                        .compare_exchange(
                            SUBMITTED_HISTORY_EMPTY,
                            SUBMITTED_HISTORY_WRITING,
                            Ordering::Acquire,
                            Ordering::Relaxed,
                        )
                        .is_err()
                    {
                        continue;
                    }
                    self.store_token(cursor);
                    self.state.store(SUBMITTED_HISTORY_READY, Ordering::Release);
                    return;
                }
                SUBMITTED_HISTORY_READY => {
                    if self
                        .state
                        .compare_exchange(
                            SUBMITTED_HISTORY_READY,
                            SUBMITTED_HISTORY_WRITING,
                            Ordering::Acquire,
                            Ordering::Relaxed,
                        )
                        .is_err()
                    {
                        continue;
                    }
                    // Serial M7.9 drains before another frame can submit. If
                    // a future executor nevertheless publishes twice, retain
                    // the queue-latest token without a post-submit panic; the
                    // exact revision gate at drain either settles it or
                    // invalidates the derived authority fail-closed.
                    self.store_token(cursor);
                    self.state.store(SUBMITTED_HISTORY_READY, Ordering::Release);
                    return;
                }
                SUBMITTED_HISTORY_WRITING => std::hint::spin_loop(),
                _ => unreachable!("invalid submitted RenderWorld marker state"),
            }
        }
    }

    fn take(&self) -> Option<RenderWorldConsumerCursor> {
        loop {
            match self.state.load(Ordering::Acquire) {
                SUBMITTED_HISTORY_EMPTY => return None,
                SUBMITTED_HISTORY_READY => {
                    if self
                        .state
                        .compare_exchange(
                            SUBMITTED_HISTORY_READY,
                            SUBMITTED_HISTORY_WRITING,
                            Ordering::Acquire,
                            Ordering::Relaxed,
                        )
                        .is_err()
                    {
                        continue;
                    }
                    let cursor = self.load_token();
                    self.state.store(SUBMITTED_HISTORY_EMPTY, Ordering::Release);
                    return Some(cursor);
                }
                SUBMITTED_HISTORY_WRITING => std::hint::spin_loop(),
                _ => unreachable!("invalid submitted RenderWorld marker state"),
            }
        }
    }

    fn load_token(&self) -> RenderWorldConsumerCursor {
        let low = self.revision_low.load(Ordering::Relaxed);
        let high = self.revision_high.load(Ordering::Relaxed);
        RenderWorldConsumerCursor {
            scene_id: self.scene_id.load(Ordering::Relaxed),
            revision: u64::from(low) | (u64::from(high) << 32),
        }
    }

    fn store_token(&self, cursor: RenderWorldConsumerCursor) {
        self.scene_id.store(cursor.scene_id, Ordering::Relaxed);
        self.revision_low
            .store(cursor.revision as u32, Ordering::Relaxed);
        self.revision_high
            .store((cursor.revision >> 32) as u32, Ordering::Relaxed);
    }
}

const _: fn() = || {
    fn assert_sync<T: Sync>() {}
    assert_sync::<ExtractedScene>();
};

impl ExtractedScene {
    /// Creates an empty extracted scene
    #[must_use]
    pub fn new() -> Self {
        Self {
            render_items: Vec::new(),
            render_world: PersistentRenderWorld::new(),
            render_world_pending: FxHashMap::default(),
            render_world_hierarchy_stack: Vec::new(),
            render_world_ordered_changes: Vec::new(),
            render_world_changed_sources: Vec::new(),
            submitted_previous_scratch: FxHashMap::default(),
            submitted_world_history: SubmittedWorldHistoryMarker::default(),
            render_world_scratch_underutilized_frames: 0,
            gpu_driven_facts_captured: false,
            gpu_driven_synopsis: GpuDrivenWorkloadSynopsis::default(),
            scene_id: 0,
            scene_variants: SceneFeatures::empty(),
            scene_defines: ShaderDefines::new(),
            background: BackgroundMode::default(),
            envvironment: Environment::default(),
            has_transmission: false,
            lights: Vec::new(),
            local_light_count: 0,
            directional_light_storage_buffer: CpuBuffer::new(
                vec![GpuLightStorage::default()],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneDirectionalLightStorageBuffer"),
            ),
            local_light_storage_buffer: CpuBuffer::new(
                vec![GpuLightStorage::default()],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneLocalLightStorageBuffer"),
            ),
            local_light_metadata_buffer: CpuBuffer::new(
                LightBufferMetadata::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneLocalLightMetadata"),
            ),
            environment_uniforms_buffer: CpuBuffer::new(
                EnvironmentUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneEnvironmentUniforms"),
            ),

            local_light_upload_cache: Vec::with_capacity(16),
            directional_light_upload_cache: Vec::with_capacity(4),
        }
    }

    /// Pre-allocates capacity
    #[must_use]
    pub fn with_capacity(item_capacity: usize) -> Self {
        Self {
            render_items: Vec::with_capacity(item_capacity),
            render_world: PersistentRenderWorld::new(),
            render_world_pending: FxHashMap::default(),
            render_world_hierarchy_stack: Vec::new(),
            render_world_ordered_changes: Vec::new(),
            render_world_changed_sources: Vec::new(),
            submitted_previous_scratch: FxHashMap::default(),
            submitted_world_history: SubmittedWorldHistoryMarker::default(),
            render_world_scratch_underutilized_frames: 0,
            gpu_driven_facts_captured: false,
            gpu_driven_synopsis: GpuDrivenWorkloadSynopsis::default(),
            scene_id: 0,
            scene_variants: SceneFeatures::empty(),
            scene_defines: ShaderDefines::new(),
            background: BackgroundMode::default(),
            envvironment: Environment::default(),
            has_transmission: false,
            lights: Vec::with_capacity(16),
            local_light_count: 0,
            directional_light_storage_buffer: CpuBuffer::new(
                vec![GpuLightStorage::default(); 4],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneDirectionalLightStorageBuffer"),
            ),
            local_light_storage_buffer: CpuBuffer::new(
                vec![GpuLightStorage::default(); 16],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneLocalLightStorageBuffer"),
            ),
            local_light_metadata_buffer: CpuBuffer::new(
                LightBufferMetadata::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneLocalLightMetadata"),
            ),
            environment_uniforms_buffer: CpuBuffer::new(
                EnvironmentUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneEnvironmentUniforms"),
            ),

            local_light_upload_cache: Vec::with_capacity(16),
            directional_light_upload_cache: Vec::with_capacity(4),
        }
    }

    /// Clear data for reuse
    pub fn clear(&mut self) {
        // `render_items` is the persistent render-world payload. Its length
        // and bytes remain untouched on a static warm frame.
        self.gpu_driven_facts_captured = false;
        self.gpu_driven_synopsis = GpuDrivenWorkloadSynopsis::default();
        self.scene_defines.clear();
        self.scene_id = 0;
        self.lights.clear();
        self.local_light_count = 0;
    }

    #[must_use]
    pub fn local_light_count(&self) -> usize {
        self.local_light_count
    }

    #[must_use]
    pub fn directional_light_count(&self) -> usize {
        self.lights.len().saturating_sub(self.local_light_count)
    }

    #[must_use]
    pub fn local_lights(&self) -> &[ExtractedLight] {
        &self.lights[..self.local_light_count]
    }

    #[must_use]
    pub fn directional_lights(&self) -> &[ExtractedLight] {
        &self.lights[self.local_light_count..]
    }

    /// Reuse current instance memory, extract data from Scene.
    ///
    /// Extracts ALL active meshes into `render_items` without frustum culling.
    /// Frustum culling is deferred to the Cull phase where each `RenderView`
    /// independently culls from this unified list.
    pub(crate) fn extract_into(
        &mut self,
        scene: &mut Scene,
        camera: &RenderCamera,
        assets: &AssetServer,
        elapsed_seconds: f64,
        max_light_count: usize,
        capture_gpu_driven_facts: bool,
    ) -> RenderWorldSyncStats {
        // Extraction is the final authority boundary for derived matrices.
        // Direct Renderer users are not required to remember a separate Scene
        // update call; the warm path is O(1), while a late mutation updates
        // only its retained dirty subtrees. Do not settle Scene's legacy
        // previous matrices here -- RenderWorld owns submitted-frame history.
        scene.flush_dirty_world_matrices();
        // Queue submission is marked through a shared reference because the
        // graph arena still owns immutable snapshot borrows. Materialize that
        // result only now, after the previous graph/arena has been cleared and
        // before another Scene domain can become active.
        let submitted_history_settled = self.commit_marked_submitted_world_history();
        self.clear();
        self.gpu_driven_facts_captured = capture_gpu_driven_facts;
        self.extract_lights(scene, camera, max_light_count);

        let mutation_batch = scene.render_mutations();
        let mutation_cursor = mutation_batch.cursor;
        let scene_id = mutation_batch.scene_id;
        let topology_epoch = mutation_batch.topology_epoch;
        self.render_world
            .activate_scene(scene_id, &mut self.render_items, elapsed_seconds);
        if self.render_world.take_active_trim_signal() {
            self.trim_render_world_scratch();
        }
        self.render_world_pending.clear();
        self.render_world_hierarchy_stack.clear();
        self.render_world_ordered_changes.clear();
        self.render_world_changed_sources.clear();
        self.render_world
            .validate_asset_authority(assets, capture_gpu_driven_facts);
        let retained_gpu_facts_complete = self.render_world.gpu_facts_complete();
        let geometry_asset_epoch = assets.geometries.global_version();
        let material_asset_epoch = assets.materials.global_version();
        let render_world_plan = self.render_world.begin_sync(
            &mutation_batch,
            capture_gpu_driven_facts,
            geometry_asset_epoch,
            material_asset_epoch,
        );

        match render_world_plan {
            RenderWorldSyncPlan::FullRebuild(_) => {
                self.submitted_previous_scratch.clear();
                if self.render_world.owns_scene(scene_id) {
                    self.submitted_previous_scratch.extend(
                        self.render_items
                            .iter()
                            .map(|item| (item.node_handle, item.prev_world_matrix)),
                    );
                }
                self.render_items.clear();
                let extraction = Self::extract_render_items(
                    &mut self.render_items,
                    capture_gpu_driven_facts,
                    scene,
                    camera,
                    assets,
                    &self.submitted_previous_scratch,
                );
                self.render_items
                    .sort_unstable_by_key(|item| node_order(item.node_handle));
                self.render_world.commit_full_rebuild(
                    &self.render_items,
                    scene_id,
                    topology_epoch,
                    capture_gpu_driven_facts,
                    geometry_asset_epoch,
                    material_asset_epoch,
                    extraction.has_unresolved_assets,
                    extraction.has_unstable_authority,
                    extraction.items_examined,
                );
            }
            RenderWorldSyncPlan::UpdateTransforms => {
                for &mutation in self.render_world.mutations() {
                    let entry = self.render_world_pending.entry(mutation.node).or_default();
                    let propagate_transform = mutation
                        .flags
                        .contains(myth_scene::SceneMutationFlags::TRANSFORM)
                        && !entry.contains(myth_scene::SceneMutationFlags::TRANSFORM);
                    *entry |= mutation.flags;
                    if propagate_transform {
                        self.render_world_hierarchy_stack.push(mutation.node);
                    }
                    while let Some(parent) = self.render_world_hierarchy_stack.pop() {
                        let Some(parent_node) = scene.nodes.get(parent) else {
                            continue;
                        };
                        for &child in parent_node.children() {
                            let child_flags = self.render_world_pending.entry(child).or_default();
                            if !child_flags.contains(myth_scene::SceneMutationFlags::TRANSFORM) {
                                *child_flags |= myth_scene::SceneMutationFlags::TRANSFORM;
                                self.render_world_hierarchy_stack.push(child);
                            }
                        }
                    }
                }
                self.render_world_ordered_changes.extend(
                    self.render_world_pending
                        .iter()
                        .map(|(&node, &flags)| (node, flags)),
                );
                self.render_world_ordered_changes
                    .sort_unstable_by_key(|(node, _)| node_order(*node));

                let mut structural_fallback = false;
                let geo_guard = assets.geometries.read_lock();
                let mat_guard = assets.materials.read_lock();
                for &(node_handle, _mutation_flags) in &self.render_world_ordered_changes {
                    let source = self.render_world.source_for_node(node_handle);
                    let node = scene.nodes.get(node_handle);
                    let mesh = scene.meshes.get(node_handle);
                    let active = node.is_some_and(|node| node.visible)
                        && mesh.is_some_and(|mesh| mesh.visible);

                    let Some(source) = source else {
                        if active {
                            let assets_ready = mesh.is_some_and(|mesh| {
                                geo_guard.get_loaded(mesh.geometry).is_some()
                                    && mat_guard.get_loaded(mesh.material).is_some()
                            });
                            if assets_ready {
                                // The source became eligible while absent from
                                // the dense table (visibility/material change,
                                // or an async publication observed in this
                                // transaction). Rebuild to assign stable dense
                                // identities deterministically.
                                structural_fallback = true;
                            } else {
                                // The Scene mutation can still be acknowledged:
                                // this retained latch makes the next asset epoch
                                // advance a mandatory full rebuild, so an
                                // unavailable source can never be forgotten.
                                self.render_world.note_unresolved_assets();
                            }
                        }
                        continue;
                    };
                    let Some(previous_item) = self.render_items.get(source.index()).cloned() else {
                        structural_fallback = true;
                        continue;
                    };
                    let (Some(node), Some(mesh)) = (node, mesh) else {
                        structural_fallback = true;
                        continue;
                    };
                    let Some(geometry) = geo_guard.get_loaded(mesh.geometry) else {
                        structural_fallback = true;
                        continue;
                    };
                    let Some(material) = mat_guard.get_loaded(mesh.material) else {
                        structural_fallback = true;
                        continue;
                    };
                    let Some(mut next_item) = extract_render_item(
                        scene,
                        node_handle,
                        node,
                        mesh,
                        geometry,
                        material,
                        capture_gpu_driven_facts,
                        Some(previous_item.prev_world_matrix),
                    ) else {
                        structural_fallback = true;
                        continue;
                    };
                    let gpu_facts_valid = capture_gpu_driven_facts
                        || (retained_gpu_facts_complete
                            && preserve_gpu_facts(&previous_item, &mut next_item));

                    let mut change_flags = RenderWorldChangeFlags::empty();
                    if previous_item.world_matrix != next_item.world_matrix
                        || previous_item.world_aabb.min != next_item.world_aabb.min
                        || previous_item.world_aabb.max != next_item.world_aabb.max
                    {
                        change_flags.insert(RenderWorldChangeFlags::TRANSFORM);
                    }
                    if !render_item_fixed_state_eq(&previous_item, &next_item) {
                        change_flags.insert(RenderWorldChangeFlags::FIXED_STATE);
                    }
                    self.render_items[source.index()] = next_item;
                    if !change_flags.is_empty() {
                        self.render_world_changed_sources
                            .push(RenderWorldSourceUpdate {
                                node: node_handle,
                                flags: change_flags,
                                previous_geometry: previous_item.geometry,
                                previous_material: previous_item.material,
                                gpu_facts_valid,
                            });
                    }
                }
                drop(geo_guard);
                drop(mat_guard);

                if structural_fallback {
                    self.submitted_previous_scratch.clear();
                    self.submitted_previous_scratch.extend(
                        self.render_items
                            .iter()
                            .map(|item| (item.node_handle, item.prev_world_matrix)),
                    );
                    self.render_items.clear();
                    let extraction = Self::extract_render_items(
                        &mut self.render_items,
                        capture_gpu_driven_facts,
                        scene,
                        camera,
                        assets,
                        &self.submitted_previous_scratch,
                    );
                    self.render_items
                        .sort_unstable_by_key(|item| node_order(item.node_handle));
                    self.render_world
                        .note_full_rebuild_reason(RenderWorldFullRebuildReason::TopologyChanged);
                    self.render_world.commit_full_rebuild(
                        &self.render_items,
                        scene_id,
                        topology_epoch,
                        capture_gpu_driven_facts,
                        geometry_asset_epoch,
                        material_asset_epoch,
                        extraction.has_unresolved_assets,
                        extraction.has_unstable_authority,
                        extraction
                            .items_examined
                            .saturating_add(self.render_world_ordered_changes.len()),
                    );
                    // The conservative rebuild has already completed this
                    // branch's transaction.
                    self.gpu_driven_synopsis = self.render_world.synopsis();
                } else {
                    self.render_world.commit_transform_updates(
                        &self.render_items,
                        &self.render_world_changed_sources,
                        self.render_world_ordered_changes.len(),
                        capture_gpu_driven_facts,
                    );
                }
            }
            RenderWorldSyncPlan::Reuse => {
                self.render_world.commit_reuse(&self.render_items);
            }
        }
        let acknowledged = scene.acknowledge_render_mutations(mutation_cursor);
        debug_assert!(
            acknowledged,
            "persistent RenderWorld rejected its exact Scene mutation cursor"
        );
        if !acknowledged {
            self.render_world.invalidate_scene_authority();
        }
        self.gpu_driven_synopsis = if capture_gpu_driven_facts {
            self.render_world.synopsis()
        } else {
            GpuDrivenWorkloadSynopsis::default()
        };
        self.extract_environment(scene);
        self.maybe_trim_render_world_scratch();

        self.scene_variants.clear();

        if self.lights.iter().any(|light| light.cast_shadows) {
            self.scene_defines.set("HAS_SHADOWS", "1");
            self.scene_variants.insert(SceneFeatures::HAS_SHADOWS);
        }

        if scene.ssao.enabled {
            self.scene_defines.set("USE_SSAO", "1");
            self.scene_variants.insert(SceneFeatures::USE_SSAO);
        }

        if scene.ssgi.enabled {
            self.scene_defines.set("USE_SSGI", "1");
            self.scene_variants.insert(SceneFeatures::USE_SSGI);
        }

        if scene.ssss.enabled {
            self.scene_defines.set("USE_SCREEN_SPACE_FEATURES", "1");
            self.scene_defines.set("USE_SSS", "1");
            self.scene_variants.insert(SceneFeatures::USE_SSS);
        }

        if scene.ssr.enabled {
            self.scene_defines.set("USE_SCREEN_SPACE_FEATURES", "1");
            self.scene_defines.set("USE_SSR", "1");
            self.scene_variants.insert(SceneFeatures::USE_SSR);
        }

        if matches!(scene.background.mode, BackgroundMode::Procedural(_)) {
            self.scene_defines.set("USE_ATMOSPHERE_TRANSMITTANCE", "1");
        }

        // Material-override debug view — inject shader defines so the PBR
        // fragment shader short-circuits lighting and outputs raw attributes.
        #[cfg(feature = "debug_view")]
        {
            use myth_scene::camera::DebugViewMode;
            match camera.debug_view.mode {
                DebugViewMode::Albedo => self.scene_defines.set("DEBUG_VIEW_ALBEDO", "1"),
                DebugViewMode::Roughness => self.scene_defines.set("DEBUG_VIEW_ROUGHNESS", "1"),
                DebugViewMode::Metalness => self.scene_defines.set("DEBUG_VIEW_METALNESS", "1"),
                _ => {}
            }
        }

        let mut stats = self.render_world.last_stats();
        stats.submitted_history_settled =
            u32::try_from(submitted_history_settled).unwrap_or(u32::MAX);
        stats.retained_bytes =
            stats
                .retained_bytes
                .saturating_add(self.render_world_pending.capacity().saturating_mul(
                    std::mem::size_of::<NodeHandle>()
                        + std::mem::size_of::<myth_scene::SceneMutationFlags>(),
                ))
                .saturating_add(
                    self.render_world_hierarchy_stack
                        .capacity()
                        .saturating_mul(std::mem::size_of::<NodeHandle>()),
                )
                .saturating_add(self.render_world_ordered_changes.capacity().saturating_mul(
                    std::mem::size_of::<(NodeHandle, myth_scene::SceneMutationFlags)>(),
                ))
                .saturating_add(
                    self.render_world_changed_sources
                        .capacity()
                        .saturating_mul(std::mem::size_of::<RenderWorldSourceUpdate>()),
                )
                .saturating_add(self.submitted_previous_scratch.capacity().saturating_mul(
                    std::mem::size_of::<NodeHandle>() + std::mem::size_of::<Mat4>(),
                ));
        stats
    }

    /// Refresh the extracted scene's GPU-facing light and environment buffers.
    pub fn prepare_gpu_scene_inputs(&mut self, env_map_max_mip_level: f32) {
        self.sync_directional_light_storage_buffer();
        self.sync_local_light_storage_buffer();
        self.sync_local_light_metadata_buffer();
        self.sync_environment_uniforms(env_map_max_mip_level);
    }

    fn extract_lights(&mut self, scene: &Scene, camera: &RenderCamera, max_light_count: usize) {
        self.lights.reserve(scene.lights.len());

        for (light, world_matrix) in scene.iter_active_lights() {
            let position = world_matrix.translation.to_vec3();
            let direction = world_matrix
                .transform_vector3(-glam::Vec3::Z)
                .normalize_or_zero();

            let extracted_light = ExtractedLight {
                id: light.id(),
                color: light.color,
                intensity: light.intensity,
                flags: light.flags,
                cast_shadows: light.cast_shadows,
                kind: light.kind.clone(),
                position,
                direction,
                shadow: light.shadow.clone(),
            };

            if !light_is_visible_for_camera(&extracted_light, camera) {
                continue;
            }

            self.lights.push(extracted_light);
        }

        let visible_light_count = self.lights.len();
        let camera_position = camera.position.to_array().into();
        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut self.lights, camera_position, max_light_count);
        if truncated_count > 0 {
            log::warn!(
                "Extracted light list truncated from {} to {} entries to satisfy GPU storage buffer limits",
                visible_light_count,
                self.lights.len(),
            );
        }

        self.local_light_count = local_light_count;
    }

    fn sync_directional_light_storage_buffer(&mut self) {
        let mut upload_cache = std::mem::take(&mut self.directional_light_upload_cache);
        upload_cache.clear();
        upload_cache.reserve(self.directional_light_count().max(1));

        for light in self.directional_lights() {
            upload_cache.push(extracted_light_to_gpu(light));
        }

        if upload_cache.is_empty() {
            upload_cache.push(GpuLightStorage::default());
        }

        let needs_update =
            self.directional_light_storage_buffer.read().as_slice() != upload_cache.as_slice();
        if needs_update {
            self.directional_light_storage_buffer
                .write()
                .clone_from(&upload_cache);
        }

        self.directional_light_upload_cache = upload_cache;
    }

    fn sync_local_light_storage_buffer(&mut self) {
        let mut upload_cache = std::mem::take(&mut self.local_light_upload_cache);
        upload_cache.clear();
        upload_cache.reserve(self.local_light_count.max(1));

        for light in self.local_lights() {
            upload_cache.push(extracted_light_to_gpu(light));
        }

        if upload_cache.is_empty() {
            upload_cache.push(GpuLightStorage::default());
        }

        let needs_update =
            self.local_light_storage_buffer.read().as_slice() != upload_cache.as_slice();
        if needs_update {
            self.local_light_storage_buffer
                .write()
                .clone_from(&upload_cache);
        }

        self.local_light_upload_cache = upload_cache;
    }

    fn sync_local_light_metadata_buffer(&mut self) {
        let new_metadata = LightBufferMetadata {
            total_light_count: self.local_light_count as u32,
            active_local_light_count: self.local_light_count as u32,
            ..Default::default()
        };

        let needs_update = *self.local_light_metadata_buffer.read() != new_metadata;
        if needs_update {
            *self.local_light_metadata_buffer.write() = new_metadata;
        }
    }

    fn sync_environment_uniforms(&mut self, env_map_max_mip_level: f32) {
        let env = &self.envvironment;
        let new_uniforms = EnvironmentUniforms {
            ambient_light: env.ambient,
            directional_light_count: self.directional_light_count() as u32,
            env_map_intensity: env.intensity,
            env_map_rotation: env.rotation,
            env_map_max_mip_level,
            ..Default::default()
        };

        let needs_update = *self.environment_uniforms_buffer.read() != new_uniforms;
        if needs_update {
            *self.environment_uniforms_buffer.write() = new_uniforms;
        }
    }

    #[must_use]
    pub fn has_shadow_casters(&self) -> bool {
        self.scene_variants.contains(SceneFeatures::HAS_SHADOWS)
    }

    #[inline]
    #[must_use]
    pub(crate) const fn gpu_driven_facts_captured(&self) -> bool {
        self.gpu_driven_facts_captured
    }

    #[inline]
    #[must_use]
    pub(crate) const fn gpu_driven_workload_synopsis(&self) -> GpuDrivenWorkloadSynopsis {
        self.gpu_driven_synopsis
    }

    #[inline]
    #[must_use]
    pub(crate) fn render_world_delta(
        &self,
        cursor: RenderWorldConsumerCursor,
    ) -> RenderWorldDelta<'_> {
        self.render_world.delta_since(cursor)
    }

    #[inline]
    pub(crate) fn acknowledge_render_world_delta(
        &mut self,
        cursor: RenderWorldConsumerCursor,
    ) -> bool {
        self.render_world.acknowledge_delta(cursor)
    }

    /// Record that the queue accepted this RenderWorld authority. This shared
    /// operation is intentionally atomic-marker-only: graph snapshots may
    /// still borrow the extracted scene until the frame arena is reset. The next mutable
    /// extraction commits the submitted transform baseline before activating
    /// another per-scene domain.
    pub(crate) fn mark_world_history_submitted(&self) {
        let cursor = self.render_world.cursor();
        self.submitted_world_history.publish(cursor);
    }

    fn commit_marked_submitted_world_history(&mut self) -> usize {
        let Some(cursor) = self.submitted_world_history.take() else {
            return 0;
        };
        let Some(settled) = self
            .render_world
            .commit_submitted_history(cursor, &mut self.render_items)
        else {
            // A stale or cross-domain marker must never settle the current
            // authority. Drop the derived cache so the following extraction
            // rebuilds from Scene instead of silently advancing history past
            // work which was not part of this submission.
            self.render_world.invalidate_scene_authority();
            return 0;
        };
        self.gpu_driven_synopsis = self.render_world.synopsis();
        settled
    }

    #[inline]
    #[must_use]
    pub(crate) const fn shadow_caster_bounds_union(&self) -> Option<BoundingBox> {
        self.render_world.shadow_caster_bounds_union()
    }

    #[cfg(test)]
    pub(crate) fn initialize_test_render_world(
        &mut self,
        scene_id: u32,
        capture_gpu_facts: bool,
    ) -> RenderWorldConsumerCursor {
        self.scene_id = scene_id;
        self.render_items
            .sort_unstable_by_key(|item| node_order(item.node_handle));
        let test_items = std::mem::take(&mut self.render_items);
        self.render_world
            .activate_scene(scene_id, &mut self.render_items, 0.0);
        self.render_items = test_items;
        self.render_world
            .note_full_rebuild_reason(RenderWorldFullRebuildReason::Cold);
        self.render_world.commit_full_rebuild(
            &self.render_items,
            scene_id,
            1,
            capture_gpu_facts,
            0,
            0,
            false,
            false,
            self.render_items.len(),
        );
        self.gpu_driven_facts_captured = capture_gpu_facts;
        self.gpu_driven_synopsis = self.render_world.synopsis();
        RenderWorldConsumerCursor {
            scene_id,
            revision: 0,
        }
    }

    #[cfg(test)]
    pub(crate) const fn current_test_render_world_cursor(&self) -> RenderWorldConsumerCursor {
        self.render_world.cursor()
    }

    #[cfg(test)]
    pub(crate) fn mark_test_source_change(
        &mut self,
        source: crate::graph::frame::ExtractedSourceId,
        flags: RenderWorldChangeFlags,
    ) {
        let item = self
            .render_items
            .get(source.index())
            .expect("test RenderWorld source must exist");
        let update = RenderWorldSourceUpdate {
            node: item.node_handle,
            flags,
            previous_geometry: item.geometry,
            previous_material: item.material,
            gpu_facts_valid: self.render_world.gpu_facts_complete(),
        };
        self.render_world.commit_transform_updates(
            &self.render_items,
            std::slice::from_ref(&update),
            1,
            self.gpu_driven_facts_captured,
        );
        self.gpu_driven_synopsis = self.render_world.synopsis();
    }

    fn trim_render_world_scratch(&mut self) {
        self.render_world_pending.shrink_to_fit();
        self.render_world_hierarchy_stack.shrink_to_fit();
        self.render_world_ordered_changes.shrink_to_fit();
        self.render_world_changed_sources.shrink_to_fit();
        self.submitted_previous_scratch.shrink_to_fit();
        self.render_world_scratch_underutilized_frames = 0;
    }

    fn maybe_trim_render_world_scratch(&mut self) {
        let underutilized = map_underutilized(&self.render_world_pending)
            || vec_underutilized(&self.render_world_hierarchy_stack)
            || vec_underutilized(&self.render_world_ordered_changes)
            || vec_underutilized(&self.render_world_changed_sources)
            || map_underutilized(&self.submitted_previous_scratch);
        if !underutilized {
            self.render_world_scratch_underutilized_frames = 0;
            return;
        }
        self.render_world_scratch_underutilized_frames = self
            .render_world_scratch_underutilized_frames
            .saturating_add(1);
        if self.render_world_scratch_underutilized_frames >= 300 {
            self.trim_render_world_scratch();
        }
    }

    /// Extract all active render items (no frustum culling).
    ///
    /// Only performs lightweight validity checks:
    /// - `mesh.visible` and `node.visible` flags
    /// - Geometry and material assets exist
    ///
    /// World-space bounding spheres are pre-computed here so the Cull phase
    /// can test each item against multiple `RenderView` frustums without
    /// re-acquiring the geometry read lock.
    #[allow(clippy::too_many_lines)]
    fn extract_render_items(
        render_items: &mut Vec<ExtractedRenderItem>,
        capture_gpu_driven_facts: bool,
        scene: &Scene,
        _camera: &RenderCamera,
        assets: &AssetServer,
        submitted_previous: &FxHashMap<NodeHandle, Mat4>,
    ) -> RenderItemExtractionStatus {
        let geo_guard = assets.geometries.read_lock();
        let mat_guard = assets.materials.read_lock();
        let mut status = RenderItemExtractionStatus::default();
        for (node_handle, mesh) in &scene.meshes {
            status.items_examined = status.items_examined.saturating_add(1);
            let Some(node) = scene.nodes.get(node_handle) else {
                continue;
            };
            if !mesh.visible || !node.visible {
                continue;
            }
            let Some(geometry) = geo_guard.get_loaded(mesh.geometry) else {
                status.has_unresolved_assets = true;
                continue;
            };
            let Some(material) = mat_guard.get_loaded(mesh.material) else {
                status.has_unresolved_assets = true;
                continue;
            };
            let previous = submitted_previous.get(&node_handle).copied();
            if let Some(item) = extract_render_item(
                scene,
                node_handle,
                node,
                mesh,
                geometry,
                material,
                capture_gpu_driven_facts,
                previous,
            ) {
                render_items.push(item);
            } else {
                // Visibility was checked above, so failure here is a material
                // coherence race rather than an ineligible source. Retain an
                // unconditional retry latch before acknowledging the Scene.
                status.has_unstable_authority = true;
            }
        }
        status
    }

    /// Advance CPU morph history for draw resources consumed by a submitted
    /// frame. Abandoned composers and failed target acquisition never call
    /// this commit boundary.
    pub(crate) fn commit_prepared_draw_history(
        &self,
        scene: &mut Scene,
        render_lists: &RenderLists,
        draw_plans: &DrawPlanStore,
    ) {
        for &frame_source in draw_plans.emitted_morph_history_sources() {
            let Some(extracted_source) = render_lists.extracted_source(frame_source) else {
                continue;
            };
            let Some(item) = self.render_items.get(extracted_source.index()) else {
                continue;
            };
            scene.commit_submitted_morph_history(item.node_handle);
        }
    }

    /// Extract environment data
    fn extract_environment(&mut self, scene: &Scene) {
        self.background.clone_from(&scene.background.mode);
        self.scene_defines.clone_from(scene.shader_defines());
        self.scene_id = scene.id();
        self.envvironment.clone_from(&scene.environment);
    }

    /// Get render item count
    #[inline]
    #[must_use]
    pub fn render_item_count(&self) -> usize {
        self.render_items.len()
    }

    /// Check if empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.render_items.is_empty()
    }
}

impl Default for ExtractedScene {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_render_item(
    scene: &Scene,
    node_handle: NodeHandle,
    node: &myth_scene::node::Node,
    mesh: &myth_resources::mesh::Mesh,
    geometry: &AssetEntry<Geometry>,
    material: &AssetEntry<Material>,
    capture_gpu_driven_facts: bool,
    submitted_previous: Option<Mat4>,
) -> Option<ExtractedRenderItem> {
    if !mesh.visible || !node.visible {
        return None;
    }

    let node_world = node.transform.world_matrix;
    let world_matrix = Mat4::from(node_world);
    // A newly observed source has no submitted render history. Starting it at
    // current avoids a spurious first-frame velocity. Existing sources retain
    // the RenderWorld-owned baseline even across a conservative full rebuild.
    let prev_world_matrix = submitted_previous.unwrap_or(world_matrix);
    let skeleton_key = scene
        .skins
        .get(node_handle)
        .map(|binding| binding.skeleton)
        .filter(|key| scene.skeleton_pool.get(*key).is_some());

    let world_determinant = world_matrix.determinant();
    let item_variant_flags =
        u32::from(world_determinant < 0.0) | (u32::from(skeleton_key.is_some()) << 1);
    let world_aabb = if let Some(key) = skeleton_key
        && let Some(skeleton) = scene.skeleton_pool.get(key)
        && let Some(local_bounds) = skeleton.local_bounds()
    {
        local_bounds.transform(&node_world)
    } else {
        geometry.bounding_box.transform(&node_world)
    };

    let material_version = material.data.version();
    let material_index = mesh.material.data().as_ffi() as u32;
    let mut material_classification = MaterialClassification::empty();
    material_classification.set(
        MaterialClassification::TRANSPARENT,
        material.is_transparent(),
    );
    material_classification.set(
        MaterialClassification::USES_TRANSMISSION,
        material.use_transmission(),
    );
    let cast_shadows = mesh.cast_shadows && material.alpha_mode() != AlphaMode::Blend;

    let mut item_shader_features = ItemShaderFeatures::empty();
    if skeleton_key.is_some() {
        item_shader_features.insert(ItemShaderFeatures::HAS_SKINNING);
    }
    if mesh.receive_shadows {
        item_shader_features.insert(ItemShaderFeatures::RECEIVE_SHADOWS);
    }

    let (
        geometry_structure_version,
        geometry_layout_version,
        geometry_topology,
        geometry_index_format,
        geometry_index_count,
        geometry_has_morph_targets,
        geometry_has_vertex_attributes,
        geometry_has_instanced_attributes,
        material_gpu_scene_compatible,
    ) = if capture_gpu_driven_facts {
        let index_attribute = geometry.index_attribute();
        (
            geometry.structure_version(),
            geometry.layout_version(),
            geometry.topology,
            index_attribute.map(|indices| indices.format),
            index_attribute.map_or(0, |indices| indices.count),
            geometry.has_morph_targets(),
            !geometry.attributes().is_empty(),
            geometry
                .attributes()
                .values()
                .any(|attribute| attribute.step_mode == wgpu::VertexStepMode::Instance),
            !matches!(&material.data, MaterialType::Custom(_)),
        )
    } else {
        (
            0,
            0,
            wgpu::PrimitiveTopology::TriangleList,
            None,
            0,
            false,
            false,
            false,
            false,
        )
    };
    if material.data.version() != material_version {
        return None;
    }

    Some(ExtractedRenderItem {
        node_handle,
        world_matrix,
        prev_world_matrix,
        geometry: mesh.geometry,
        material: mesh.material,
        geometry_asset_version: geometry.version,
        geometry_asset_identity: geometry.identity,
        geometry_structure_version,
        geometry_layout_version,
        geometry_topology,
        geometry_index_format,
        geometry_index_count,
        geometry_has_morph_targets,
        geometry_has_vertex_attributes,
        geometry_has_instanced_attributes,
        material_asset_version: material.version,
        material_asset_identity: material.identity,
        material_version,
        material_gpu_scene_compatible,
        material_index,
        material_classification,
        skeleton: skeleton_key,
        item_variant_flags,
        item_shader_features,
        cast_shadows,
        receive_shadows: mesh.receive_shadows,
        world_aabb,
    })
}

/// Preserve an already enriched GPU profile while the public mode is Off.
/// Only authority-stable halves are copied; a changed geometry/material
/// leaves the record deliberately incomplete so RenderWorld schedules one
/// typed profile upgrade when GPU-driven rendering is selected again.
fn preserve_gpu_facts(previous: &ExtractedRenderItem, next: &mut ExtractedRenderItem) -> bool {
    let geometry_stable = previous.geometry == next.geometry
        && previous.geometry_asset_version == next.geometry_asset_version
        && previous.geometry_asset_identity == next.geometry_asset_identity;
    if geometry_stable {
        next.geometry_structure_version = previous.geometry_structure_version;
        next.geometry_layout_version = previous.geometry_layout_version;
        next.geometry_topology = previous.geometry_topology;
        next.geometry_index_format = previous.geometry_index_format;
        next.geometry_index_count = previous.geometry_index_count;
        next.geometry_has_morph_targets = previous.geometry_has_morph_targets;
        next.geometry_has_vertex_attributes = previous.geometry_has_vertex_attributes;
        next.geometry_has_instanced_attributes = previous.geometry_has_instanced_attributes;
    }

    let material_stable = previous.material == next.material
        && previous.material_asset_version == next.material_asset_version
        && previous.material_asset_identity == next.material_asset_identity
        && previous.material_version == next.material_version;
    if material_stable {
        next.material_gpu_scene_compatible = previous.material_gpu_scene_compatible;
    }
    geometry_stable && material_stable
}

fn render_item_fixed_state_eq(a: &ExtractedRenderItem, b: &ExtractedRenderItem) -> bool {
    a.geometry == b.geometry
        && a.material == b.material
        && a.geometry_asset_version == b.geometry_asset_version
        && a.geometry_asset_identity == b.geometry_asset_identity
        && a.geometry_structure_version == b.geometry_structure_version
        && a.geometry_layout_version == b.geometry_layout_version
        && a.geometry_topology == b.geometry_topology
        && a.geometry_index_format == b.geometry_index_format
        && a.geometry_index_count == b.geometry_index_count
        && a.geometry_has_morph_targets == b.geometry_has_morph_targets
        && a.geometry_has_vertex_attributes == b.geometry_has_vertex_attributes
        && a.geometry_has_instanced_attributes == b.geometry_has_instanced_attributes
        && a.material_asset_version == b.material_asset_version
        && a.material_asset_identity == b.material_asset_identity
        && a.material_version == b.material_version
        && a.material_gpu_scene_compatible == b.material_gpu_scene_compatible
        && a.material_index == b.material_index
        && a.material_classification == b.material_classification
        && a.skeleton == b.skeleton
        && a.item_variant_flags == b.item_variant_flags
        && a.item_shader_features == b.item_shader_features
        && a.cast_shadows == b.cast_shadows
        && a.receive_shadows == b.receive_shadows
}

#[inline]
fn vec_underutilized<T>(values: &Vec<T>) -> bool {
    values.capacity() > 1024 && values.len().saturating_mul(4) < values.capacity()
}

#[inline]
fn map_underutilized<K, V>(values: &FxHashMap<K, V>) -> bool {
    values.capacity() > 1024 && values.len().saturating_mul(4) < values.capacity()
}

fn light_is_visible_for_camera(light: &ExtractedLight, camera: &RenderCamera) -> bool {
    match &light.kind {
        LightKind::Directional(_) => true,
        LightKind::Point(point) => {
            point.range > 0.0
                && camera
                    .frustum
                    .intersects_sphere(light.position, point.range)
        }
        LightKind::Spot(spot) => {
            if spot.range <= 0.0 {
                return false;
            }

            let (sphere_center, sphere_radius) =
                spot_bounding_sphere(light.position, light.direction, spot);
            camera
                .frustum
                .intersects_sphere(sphere_center, sphere_radius)
        }
    }
}

fn extracted_light_to_gpu(light: &ExtractedLight) -> GpuLightStorage {
    let mut gpu_light = GpuLightStorage {
        color: light.color,
        intensity: light.intensity,
        position: light.position,
        direction: light.direction,
        flags: light.flags,
        shadow_layer_index: -1,
        ..Default::default()
    };

    match &light.kind {
        LightKind::Directional(_) => {
            gpu_light.light_type = 0;
        }
        LightKind::Point(point) => {
            gpu_light.light_type = 1;
            gpu_light.range = point.range;
        }
        LightKind::Spot(spot) => {
            gpu_light.light_type = 2;
            gpu_light.range = spot.range;
            gpu_light.inner_cone_cos = spot.inner_cone.cos();
            gpu_light.outer_cone_cos = spot.outer_cone.cos();
        }
    }

    gpu_light
}

fn extracted_light_score(light: &ExtractedLight, camera_position: Vec3) -> f32 {
    let radius = extracted_light_radius(light);
    if radius <= 0.0 || light.intensity <= 0.0 {
        return 0.0;
    }

    let distance = (light.position - camera_position)
        .length()
        .max(LIGHT_SCORE_DISTANCE_FLOOR);
    let score = light.intensity * (radius / distance);
    if score.is_finite() { score } else { 0.0 }
}

fn sort_and_truncate_lights(
    lights: &mut Vec<ExtractedLight>,
    camera_position: Vec3,
    max_light_count: usize,
) -> (usize, usize) {
    lights.sort_unstable_by(
        |lhs, rhs| match (lhs.is_directional(), rhs.is_directional()) {
            (false, false) => {
                let lhs_score = extracted_light_score(lhs, camera_position);
                let rhs_score = extracted_light_score(rhs, camera_position);
                rhs_score
                    .total_cmp(&lhs_score)
                    .then_with(|| lhs.id.cmp(&rhs.id))
            }
            (true, true) => lhs.id.cmp(&rhs.id),
            (false, true) => std::cmp::Ordering::Less,
            (true, false) => std::cmp::Ordering::Greater,
        },
    );

    let visible_light_count = lights.len();
    let mut local_light_count = lights.partition_point(|light| !light.is_directional());
    let directional_light_count = visible_light_count - local_light_count;
    if visible_light_count > max_light_count {
        if directional_light_count >= max_light_count {
            lights.drain(..local_light_count);
            lights.truncate(max_light_count);
            local_light_count = 0;
        } else {
            let retained_local_count = max_light_count - directional_light_count;
            lights.drain(retained_local_count..local_light_count);
            local_light_count = retained_local_count;
        }
    }

    let truncated_count = visible_light_count.saturating_sub(lights.len());
    (local_light_count, truncated_count)
}

fn extracted_light_radius(light: &ExtractedLight) -> f32 {
    match &light.kind {
        LightKind::Directional(_) => 0.0,
        LightKind::Point(point) => point.range,
        LightKind::Spot(spot) => spot_bounding_sphere(light.position, light.direction, spot).1,
    }
}

fn spot_bounding_sphere(position: Vec3, direction: Vec3, spot: &SpotLight) -> (Vec3, f32) {
    let outer_cone_cos = spot.outer_cone.cos();
    if outer_cone_cos < SPOT_TIGHT_SPHERE_COS_THRESHOLD {
        return (position, spot.range);
    }

    let cos_sq = (outer_cone_cos * outer_cone_cos).max(1e-4);
    let radius = 0.5 * spot.range / cos_sq;
    (position + direction * radius, radius)
}

#[cfg(test)]
mod tests {
    use super::{
        ExtractedScene, ItemShaderFeatures, RenderWorldFullRebuildReason,
        SubmittedWorldHistoryMarker, sort_and_truncate_lights, spot_bounding_sphere,
    };
    use glam::{Affine3A, Vec3, Vec4};
    use myth_assets::AssetServer;
    use myth_resources::geometry::Geometry;
    use myth_resources::material::Material;
    use myth_resources::mesh::Mesh;
    use myth_scene::Scene;
    use myth_scene::camera::{Camera, RenderCamera};
    use myth_scene::light::{DirectionalLight, LightKind, PointLight, SpotLight};

    use crate::graph::extracted::ExtractedLight;
    use crate::graph::render_world::RenderWorldConsumerCursor;

    fn test_camera() -> RenderCamera {
        let mut camera = Camera::new_perspective(45.0, 1.0, 0.1);
        camera.update_view_projection(&Affine3A::from_translation(Vec3::new(0.0, 0.0, 5.0)));
        camera.extract_render_camera()
    }

    #[test]
    fn submitted_history_marker_round_trips_split_u64_revision() {
        let marker = SubmittedWorldHistoryMarker::default();
        let cursor = RenderWorldConsumerCursor {
            scene_id: 17,
            revision: 0xfedc_ba98_7654_3210,
        };
        marker.publish(cursor);
        assert_eq!(marker.take(), Some(cursor));
        assert_eq!(marker.take(), None);

        let first = RenderWorldConsumerCursor {
            scene_id: 18,
            revision: 1,
        };
        let queue_latest = RenderWorldConsumerCursor {
            scene_id: 18,
            revision: 2,
        };
        marker.publish(first);
        marker.publish(queue_latest);
        assert_eq!(marker.take(), Some(queue_latest));
    }

    fn extract(
        extracted: &mut ExtractedScene,
        scene: &mut Scene,
        assets: &AssetServer,
    ) -> super::RenderWorldSyncStats {
        scene.update_matrix_world();
        extracted.extract_into(scene, &test_camera(), assets, 0.0, 16, false)
    }

    fn loaded_mesh_fixture() -> (AssetServer, Scene, myth_scene::NodeHandle) {
        let assets = AssetServer::new();
        let geometry = assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
        let material = assets.materials.add(Material::new_unlit(Vec4::ONE));
        let mut scene = Scene::new();
        let node = scene.add_mesh(Mesh::new(geometry, material));
        (assets, scene, node)
    }

    #[test]
    fn static_warm_scene_reuses_dense_items_without_rebuild_or_update() {
        let (assets, mut scene, _) = loaded_mesh_fixture();
        let mut extracted = ExtractedScene::new();

        let cold = extract(&mut extracted, &mut scene, &assets);
        assert_eq!(cold.items_rebuilt, 1);
        let warm = extract(&mut extracted, &mut scene, &assets);

        assert_eq!(warm.item_count, 1);
        assert_eq!(warm.items_reused, 1);
        assert_eq!(warm.items_updated, 0);
        assert_eq!(warm.items_rebuilt, 0);
        assert_eq!(warm.shadow_casters_examined, 0);
    }

    #[test]
    fn independent_render_world_consumers_fail_closed_after_scene_ack_moves_past_them() {
        let (assets, mut scene, node) = loaded_mesh_fixture();
        let mut first = ExtractedScene::new();
        let mut second = ExtractedScene::new();

        extract(&mut first, &mut scene, &assets);
        extract(&mut second, &mut scene, &assets);
        assert_eq!(second.render_items[0].world_matrix.w_axis.x, 0.0);

        scene.edit_node(node).set_position(3.0, 0.0, 0.0);
        let first_update = extract(&mut first, &mut scene, &assets);
        assert_eq!(first_update.items_updated, 1);

        // `first` has advanced Scene's compatibility acknowledgement
        // frontier. `second` cannot see the drained per-node entry, so its
        // own cursor coverage proof must force an exact rebuild rather than
        // silently reuse the old transform.
        let second_update = extract(&mut second, &mut scene, &assets);
        assert_eq!(
            second_update.full_rebuild_reason,
            RenderWorldFullRebuildReason::SceneCursorOrIdentityDrift
        );
        assert_eq!(second_update.items_rebuilt, 1);
        assert_eq!(second.render_items[0].world_matrix.w_axis.x, 3.0);
    }

    #[test]
    fn extraction_flushes_late_transform_mutations_before_acknowledging_them() {
        let (assets, mut scene, node) = loaded_mesh_fixture();
        let mut extracted = ExtractedScene::new();

        // Deliberately bypass Scene::update_matrix_world: Renderer::begin_frame
        // is a complete public boundary and extraction must not acknowledge a
        // TRS mutation while copying its stale derived matrix.
        extracted.extract_into(&mut scene, &test_camera(), &assets, 0.0, 16, false);
        scene.edit_node(node).set_position(7.0, 0.0, 0.0);
        let changed = extracted.extract_into(&mut scene, &test_camera(), &assets, 0.1, 16, false);

        assert_eq!(changed.items_updated, 1);
        assert_eq!(extracted.render_items[0].world_matrix.w_axis.x, 7.0);
        let warm = extracted.extract_into(&mut scene, &test_camera(), &assets, 0.2, 16, false);
        assert_eq!(warm.items_reused, 1);
        assert_eq!(warm.items_updated, 0);
        assert_eq!(warm.items_rebuilt, 0);
    }

    #[test]
    fn enriched_auto_profile_survives_off_mode_transform_without_upgrade_rebuild() {
        let (assets, mut scene, node) = loaded_mesh_fixture();
        let mut extracted = ExtractedScene::new();
        scene.update_matrix_world();
        extracted.extract_into(&mut scene, &test_camera(), &assets, 0.0, 16, true);

        scene.edit_node(node).set_position(1.0, 0.0, 0.0);
        scene.update_matrix_world();
        let off = extracted.extract_into(&mut scene, &test_camera(), &assets, 0.0, 16, false);
        assert_eq!(off.items_updated, 1);
        assert_eq!(off.items_rebuilt, 0);

        scene.update_matrix_world();
        let auto = extracted.extract_into(&mut scene, &test_camera(), &assets, 0.0, 16, true);
        assert_eq!(auto.items_reused, 1);
        assert_eq!(auto.items_rebuilt, 0);
        assert_eq!(auto.full_rebuild_reason, RenderWorldFullRebuildReason::None);
        assert!(extracted.gpu_driven_facts_captured());
    }

    #[test]
    fn parent_transform_mutation_updates_descendant_mesh_source() {
        let assets = AssetServer::new();
        let geometry = assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
        let material = assets.materials.add(Material::new_unlit(Vec4::ONE));
        let mut scene = Scene::new();
        let parent = scene.create_node();
        let child = scene.add_mesh_to_parent(Mesh::new(geometry, material), parent);
        let mut extracted = ExtractedScene::new();
        extract(&mut extracted, &mut scene, &assets);

        scene.edit_node(parent).set_position(3.0, 0.0, 0.0);
        let stats = extract(&mut extracted, &mut scene, &assets);

        assert_eq!(stats.items_updated, 1);
        let item = extracted
            .render_items
            .iter()
            .find(|item| item.node_handle == child)
            .expect("descendant mesh source missing");
        assert_eq!(item.world_matrix.w_axis.x, 3.0);
        assert_eq!(item.prev_world_matrix.w_axis.x, 0.0);
    }

    #[test]
    fn renderable_visibility_changes_remove_and_restore_dense_source() {
        let (assets, mut scene, node) = loaded_mesh_fixture();
        let mut extracted = ExtractedScene::new();
        extract(&mut extracted, &mut scene, &assets);

        scene.mesh_mut(node).expect("mesh missing").visible = false;
        let hidden = extract(&mut extracted, &mut scene, &assets);
        assert_eq!(hidden.item_count, 0);
        assert!(extracted.render_items.is_empty());

        scene.mesh_mut(node).expect("mesh missing").visible = true;
        let visible = extract(&mut extracted, &mut scene, &assets);
        assert_eq!(visible.item_count, 1);
        assert_eq!(extracted.render_items[0].node_handle, node);
    }

    #[test]
    fn unavailable_source_is_rebuilt_when_async_asset_becomes_ready() {
        let assets = AssetServer::new();
        let geometry = assets.geometries.reserve();
        let material = assets.materials.add(Material::new_unlit(Vec4::ONE));
        let mut scene = Scene::new();
        let node = scene.add_mesh(Mesh::new(geometry, material));
        let mut extracted = ExtractedScene::new();

        let unavailable = extract(&mut extracted, &mut scene, &assets);
        assert_eq!(unavailable.item_count, 0);
        let warm_unavailable = extract(&mut extracted, &mut scene, &assets);
        assert_eq!(warm_unavailable.item_count, 0);
        assert_eq!(warm_unavailable.items_rebuilt, 0);

        assets
            .geometries
            .insert_ready(geometry, Geometry::new_box(1.0, 1.0, 1.0));
        let ready = extract(&mut extracted, &mut scene, &assets);

        assert_eq!(ready.item_count, 1);
        assert_eq!(ready.items_rebuilt, 1);
        assert_eq!(
            ready.full_rebuild_reason,
            RenderWorldFullRebuildReason::AssetAuthorityDrift
        );
        assert_eq!(extracted.render_items[0].node_handle, node);
    }

    #[test]
    fn conservative_asset_rebuild_preserves_last_submitted_transform_history() {
        let (assets, mut scene, node) = loaded_mesh_fixture();
        let geometry = scene.mesh(node).expect("mesh missing").geometry;
        let mut extracted = ExtractedScene::new();
        extract(&mut extracted, &mut scene, &assets);

        scene.edit_node(node).set_position(5.0, 0.0, 0.0);
        extract(&mut extracted, &mut scene, &assets);
        assert_eq!(extracted.render_items[0].prev_world_matrix.w_axis.x, 0.0);
        assert_eq!(extracted.render_items[0].world_matrix.w_axis.x, 5.0);

        // This graph attempt is intentionally abandoned. An unrelated
        // conservative full rebuild must not promote current to previous.
        assets
            .geometries
            .insert_ready(geometry, Geometry::new_box(2.0, 2.0, 2.0));
        let rebuilt = extract(&mut extracted, &mut scene, &assets);
        assert_eq!(
            rebuilt.full_rebuild_reason,
            RenderWorldFullRebuildReason::AssetAuthorityDrift
        );
        assert_eq!(extracted.render_items[0].prev_world_matrix.w_axis.x, 0.0);
        assert_eq!(extracted.render_items[0].world_matrix.w_axis.x, 5.0);

        extracted.mark_world_history_submitted();
        let submitted = extract(&mut extracted, &mut scene, &assets);
        assert_eq!(submitted.submitted_history_settled, 1);
        assert_eq!(extracted.render_items[0].prev_world_matrix.w_axis.x, 5.0);
    }

    #[test]
    fn alternating_scenes_restore_independent_resident_history_domains() {
        let (assets_a, mut scene_a, node_a) = loaded_mesh_fixture();
        let (assets_b, mut scene_b, node_b) = loaded_mesh_fixture();
        scene_a.edit_node(node_a).set_position(2.0, 0.0, 0.0);
        scene_b.edit_node(node_b).set_position(9.0, 0.0, 0.0);
        let mut extracted = ExtractedScene::new();

        extract(&mut extracted, &mut scene_a, &assets_a);
        assert_eq!(extracted.render_items[0].prev_world_matrix.w_axis.x, 2.0);
        extract(&mut extracted, &mut scene_b, &assets_b);
        assert_eq!(extracted.render_items[0].prev_world_matrix.w_axis.x, 9.0);
        let restored = extract(&mut extracted, &mut scene_a, &assets_a);

        assert_eq!(restored.items_reused, 1);
        assert_eq!(restored.resident_domain_count, 2);
        assert_eq!(extracted.render_items[0].prev_world_matrix.w_axis.x, 2.0);
    }

    #[test]
    fn evicted_scene_cursor_cannot_aba_a_same_revision_cold_domain() {
        let assets = AssetServer::new();
        let geometry = assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
        let material = assets.materials.add(Material::new_unlit(Vec4::ONE));
        let mut scenes = (0..6)
            .map(|_| {
                let mut scene = Scene::new();
                scene.add_mesh(Mesh::new(geometry, material));
                scene
            })
            .collect::<Vec<_>>();
        let mut extracted = ExtractedScene::new();

        extract(&mut extracted, &mut scenes[0], &assets);
        let old_cursor = extracted.current_test_render_world_cursor();
        for scene in &mut scenes[1..] {
            extract(&mut extracted, scene, &assets);
        }

        let cold = extract(&mut extracted, &mut scenes[0], &assets);
        assert_eq!(cold.full_rebuild_reason, RenderWorldFullRebuildReason::Cold);
        let stale_delta = extracted.render_world_delta(old_cursor);
        assert!(stale_delta.requires_full_rebuild);

        let new_cursor = extracted.current_test_render_world_cursor();
        assert!(extracted.acknowledge_render_world_delta(new_cursor));
        assert!(
            !extracted
                .render_world_delta(new_cursor)
                .requires_full_rebuild
        );
    }

    #[test]
    fn unchanged_renderable_mutation_is_examined_but_not_updated() {
        let (assets, mut scene, node) = loaded_mesh_fixture();
        let mut extracted = ExtractedScene::new();
        extract(&mut extracted, &mut scene, &assets);

        // Mutable access records a conservative RENDERABLE mutation even
        // though this assignment preserves the exact source state.
        scene.mesh_mut(node).expect("mesh missing").visible = true;
        let stats = extract(&mut extracted, &mut scene, &assets);

        assert_eq!(stats.items_examined, 1);
        assert_eq!(stats.items_updated, 0);
        assert_eq!(stats.items_reused, 1);
    }

    #[test]
    fn item_shader_features_resolve_to_four_shared_canonical_variants() {
        assert_eq!(std::mem::size_of::<ItemShaderFeatures>(), 1);

        for bits in 0..4 {
            let features = ItemShaderFeatures::from_bits_retain(bits);
            let defines = features.shader_defines();
            assert_eq!(
                defines.contains("HAS_SKINNING"),
                features.contains(ItemShaderFeatures::HAS_SKINNING)
            );
            assert_eq!(
                defines.contains("RECEIVE_SHADOWS"),
                features.contains(ItemShaderFeatures::RECEIVE_SHADOWS)
            );
            assert!(std::ptr::eq(defines, features.shader_defines()));
        }

        let retained_unknown =
            ItemShaderFeatures::from_bits_retain(0x80 | ItemShaderFeatures::RECEIVE_SHADOWS.bits());
        let canonical = ItemShaderFeatures::RECEIVE_SHADOWS.shader_defines();
        assert!(std::ptr::eq(retained_unknown.shader_defines(), canonical));
    }

    fn directional_light(id: u64) -> ExtractedLight {
        ExtractedLight {
            id,
            color: Vec3::ONE,
            intensity: 1.0,
            flags: 0,
            cast_shadows: false,
            kind: LightKind::Directional(DirectionalLight {}),
            position: Vec3::ZERO,
            direction: -Vec3::Z,
            shadow: None,
        }
    }

    fn point_light(id: u64, position: Vec3, intensity: f32, range: f32) -> ExtractedLight {
        ExtractedLight {
            id,
            color: Vec3::ONE,
            intensity,
            flags: 0,
            cast_shadows: false,
            kind: LightKind::Point(PointLight { range }),
            position,
            direction: -Vec3::Z,
            shadow: None,
        }
    }

    fn spot_light(id: u64, position: Vec3, intensity: f32, range: f32) -> ExtractedLight {
        ExtractedLight {
            id,
            color: Vec3::ONE,
            intensity,
            flags: 0,
            cast_shadows: false,
            kind: LightKind::Spot(SpotLight {
                range,
                inner_cone: 0.25,
                outer_cone: 0.5,
            }),
            position,
            direction: -Vec3::Z,
            shadow: None,
        }
    }

    #[test]
    fn partition_sort_and_truncate_keeps_directional_lights_in_tail_segment() {
        let mut lights = vec![
            point_light(1, Vec3::new(0.0, 0.0, -4.0), 4.0, 6.0),
            directional_light(2),
            spot_light(3, Vec3::new(0.0, 0.0, -2.0), 2.0, 5.0),
        ];

        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut lights, Vec3::ZERO, usize::MAX);

        assert_eq!(local_light_count, 2);
        assert_eq!(truncated_count, 0);
        assert_eq!(
            lights.iter().map(|light| light.id).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
    }

    #[test]
    fn partition_sort_and_truncate_preserves_directional_lights_when_capping_locals() {
        let mut lights = vec![
            point_light(1, Vec3::new(0.0, 0.0, -12.0), 2.0, 3.0),
            point_light(2, Vec3::new(0.0, 0.0, -3.0), 4.0, 10.0),
            spot_light(3, Vec3::new(0.0, 0.0, -5.0), 3.0, 9.0),
            directional_light(4),
        ];

        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut lights, Vec3::ZERO, 2);

        assert_eq!(local_light_count, 1);
        assert_eq!(truncated_count, 2);
        assert_eq!(
            lights.iter().map(|light| light.id).collect::<Vec<_>>(),
            vec![2, 4]
        );
    }

    #[test]
    fn sort_and_truncate_reuses_the_input_allocation() {
        let mut lights = Vec::with_capacity(16);
        lights.extend([
            directional_light(9),
            point_light(2, Vec3::new(0.0, 0.0, -3.0), 4.0, 10.0),
            directional_light(4),
            point_light(1, Vec3::new(0.0, 0.0, -12.0), 2.0, 3.0),
        ]);
        let allocation = lights.as_ptr();
        let capacity = lights.capacity();

        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut lights, Vec3::ZERO, 3);

        assert_eq!(lights.as_ptr(), allocation);
        assert_eq!(lights.capacity(), capacity);
        assert_eq!(local_light_count, 1);
        assert_eq!(truncated_count, 1);
        assert_eq!(
            lights.iter().map(|light| light.id).collect::<Vec<_>>(),
            vec![2, 4, 9]
        );
    }

    #[test]
    fn sort_and_truncate_prefers_lowest_directional_ids_at_tight_capacity() {
        let mut lights = vec![
            point_light(1, Vec3::new(0.0, 0.0, -3.0), 4.0, 10.0),
            directional_light(9),
            directional_light(4),
            directional_light(7),
        ];

        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut lights, Vec3::ZERO, 2);

        assert_eq!(local_light_count, 0);
        assert_eq!(truncated_count, 2);
        assert_eq!(
            lights.iter().map(|light| light.id).collect::<Vec<_>>(),
            vec![4, 7]
        );
    }

    #[test]
    fn sort_and_truncate_preserves_local_id_tie_breaking() {
        let mut lights = vec![
            point_light(9, Vec3::new(0.0, 0.0, -5.0), 2.0, 10.0),
            point_light(4, Vec3::new(0.0, 0.0, -5.0), 2.0, 10.0),
            directional_light(7),
        ];

        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut lights, Vec3::ZERO, usize::MAX);

        assert_eq!(local_light_count, 2);
        assert_eq!(truncated_count, 0);
        assert_eq!(
            lights.iter().map(|light| light.id).collect::<Vec<_>>(),
            vec![4, 9, 7]
        );
    }

    #[test]
    fn narrow_spot_uses_tighter_bounding_sphere() {
        let spot = SpotLight {
            range: 12.0,
            inner_cone: 0.1,
            outer_cone: 0.3,
        };

        let (center, radius) = spot_bounding_sphere(Vec3::ZERO, -Vec3::Z, &spot);

        assert!(radius > spot.range * 0.5);
        assert!(center.z < 0.0);
    }
}
