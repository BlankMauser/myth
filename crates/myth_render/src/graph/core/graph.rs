use std::cmp::Ordering;
use std::collections::BinaryHeap;

use myth_assets::AssetServer;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::core::ResourceManager;
use crate::core::binding::PreparedBindGroupStore;
use crate::core::gpu::{
    AssetTextureSnapshot, BufferUploadContext, GlobalStateKey, MipmapBatchId, MipmapRequest,
    MipmapRequestBatch, PhysicalImageGenerationKey, Tracked,
};
use crate::graph::core::allocator::TransientPool;
use crate::graph::core::arena::FrameArena;
#[cfg(feature = "3dgs")]
use crate::graph::core::gaussian::{GaussianFrameBatch, GaussianFrameBatchId, GaussianFrameCloud};
use crate::graph::frame::{DrawPlanId, DrawPlanKey, DrawPlanStore, PreparedDrawStore};
use crate::graph::material_plan::MaterialPlanStore;

use super::asset_texture::{AssetTextureImportId, AssetTextureImportStore, snapshot_texture_desc};
use super::builder::PassBuilder;
use super::diagnostics::{
    RdgAttemptOwner, RdgDiagnosticsRecorder, RdgFrameDiagnostics, RdgFramePhase,
    RdgStandaloneAttempt, RdgStandaloneDiagnosticsStore,
};
use super::global_state::{
    GlobalStateAttempt, GlobalStateMaterializationFailure, GlobalStatePlan, GlobalStateSources,
};
use super::node::{NodeSlot, PassNode, PassRecord};
use super::persistent::{
    CacheCommitToken, CachePublicationReceipt, HistoryCommitToken, HistoryRegistry,
    PersistentTextureId, PersistentTextureRef,
};
use super::scene_buffer::{
    SceneBufferImportStore, SceneBufferMaterializationFailure, SceneBufferRecipe,
    SceneBufferSources,
};
use super::scene_submission::{
    SceneSubmissionDependencies, SceneSubmissionPlanId, SceneSubmissionStore,
};
use super::types::{
    BufferDesc, BufferNodeId, ErasedResourceNodeId, GraphResourceType, ResourceClass, ResourceKind,
    ResourceNodeId, ResourceRecord, TextureDesc, TextureNodeId,
};
use smallvec::SmallVec;
use wgpu::Device;

/// Per-frame rendering configuration stored in the [`RenderGraph`].
/// during graph construction through [`PassBuilder`] accessors.  This
/// eliminates the need for passes to receive screen-size or format
/// information through push parameters — they can derive `TextureDesc`s
/// directly.
#[derive(Debug, Clone, Copy)]
pub struct FrameConfig {
    /// Framebuffer width in pixels.
    pub width: u32,
    /// Framebuffer height in pixels.
    pub height: u32,
    /// Device depth format (e.g. `Depth32Float`).
    pub depth_format: wgpu::TextureFormat,
    /// MSAA sample count (1 = disabled).
    pub msaa_samples: u32,
    /// Swap-chain surface format (e.g. `Bgra8UnormSrgb`).
    pub surface_format: wgpu::TextureFormat,
    /// HDR render-target format (e.g. `Rgba16Float`).
    pub hdr_format: wgpu::TextureFormat,
}

impl FrameConfig {
    /// Returns a `RdgTextureDesc` for a 2D render target matching the frame
    /// configuration's resolution and HDR format, with the given usage flags.
    #[must_use]
    pub fn create_render_target_desc(&self, usage: wgpu::TextureUsages) -> TextureDesc {
        TextureDesc::new_2d(
            self.width,
            self.height,
            self.hdr_format,
            usage | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }

    #[must_use]
    pub fn create_surface_desc(&self, usage: wgpu::TextureUsages) -> TextureDesc {
        TextureDesc::new_2d(
            self.width,
            self.height,
            self.surface_format,
            usage | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }

    #[must_use]
    pub fn create_depth_desc(&self, usage: wgpu::TextureUsages) -> TextureDesc {
        TextureDesc::new_2d(
            self.width,
            self.height,
            self.depth_format,
            usage | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }
}

/// Controls the level of detail emitted by Render Graph Mermaid dumps.
///
/// Compact mode preserves the executable pass topology while aggregating
/// high-cardinality lazy Asset Texture inputs. Full mode emits every logical
/// resource edge for exact forensic inspection.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MermaidDumpMode {
    #[default]
    Compact,
    Full,
}

// ReadyNode is a helper struct used during topological sorting of the render graph. It represents a pass that is ready to be executed, along with its priority score for scheduling.
#[derive(Eq, PartialEq)]
struct ReadyNode {
    /// The priority score of the node, which can be based on various heuristics such as the number of downstream dependencies, resource usage, or custom user-defined metrics. Higher scores indicate higher priority for execution.
    priority: i32,
    pass_idx: usize,
}

struct HistoryCommitRequest {
    token: HistoryCommitToken,
    required_writes: SmallVec<[ErasedResourceNodeId; 4]>,
}

struct CacheCommitRequest {
    token: CacheCommitToken,
    required_writes: SmallVec<[ErasedResourceNodeId; 4]>,
}

#[derive(Debug, Clone, Copy)]
struct PersistentTextureImport {
    node: TextureNodeId,
    spec: super::persistent::PersistentTextureSpec,
}

// Rust's BinaryHeap is a max-heap, so we define the ordering such that higher priority scores come first.
impl Ord for ReadyNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.pass_idx.cmp(&self.pass_idx))
    }
}

impl PartialOrd for ReadyNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Attempt-scoped ownership of physical mipmap encoding.
///
/// Logical imports and the legacy compatibility batch may independently carry
/// requests for the same physical generation. The first node reached by the
/// compiled serial execution queue records the commands; later nodes observe
/// the claim and skip duplicate GPU work. Publication remains owned by the
/// request stores and occurs only after queue submission.
#[derive(Default)]
pub(crate) struct MipmapEncodeCoordinator {
    claimed: Mutex<FxHashSet<PhysicalImageGenerationKey>>,
}

impl MipmapEncodeCoordinator {
    #[must_use]
    pub(crate) fn try_claim(&self, generation: PhysicalImageGenerationKey) -> bool {
        self.claimed.lock().insert(generation)
    }

    pub(crate) fn clear(&mut self) {
        self.claimed.get_mut().clear();
    }

}

/// Reusable dense set for attempt-local IDs.
///
/// Starting a new operation advances one generation instead of clearing every
/// slot. The backing vector remains at its high-water mark across frames. On
/// generation wrap, one O(high-water) fill clears stale stamps before
/// generation one is safely reused; long-running editor sessions may reach it.
#[derive(Default)]
struct DenseGenerationScratch {
    stamps: Vec<u32>,
    generation: u32,
}

impl DenseGenerationScratch {
    fn begin(&mut self, domain_len: usize) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamps.fill(0);
            self.generation = 1;
        }
        if self.stamps.len() < domain_len {
            self.stamps.resize(domain_len, 0);
        }
    }

    #[inline]
    fn insert(&mut self, index: usize) -> bool {
        let stamp = self
            .stamps
            .get_mut(index)
            .expect("dense generation scratch index is outside its domain");
        if *stamp == self.generation {
            return false;
        }
        *stamp = self.generation;
        true
    }
}

fn append_logical_read(
    logical_reads: &mut Vec<ErasedResourceNodeId>,
    pass: &mut PassRecord,
    resources: &mut [ResourceRecord],
    read: ErasedResourceNodeId,
) {
    let resource_index = read.index() as usize;
    assert!(
        resource_index < resources.len(),
        "pass '{}' read a resource from another graph",
        pass.name
    );
    debug_assert_eq!(
        pass.logical_read_range.end,
        logical_reads.len(),
        "pass '{}' logical reads are no longer the open flat-table tail",
        pass.name
    );
    logical_reads.push(read);
    pass.logical_read_range.end += 1;
    resources[resource_index].add_consumer_edge();
}

/// Declarative Render Graph — persistent capacity storage.
///
/// Holds the backing `Vec`s whose heap capacity is reused across frames.
/// This struct never appears in the public API — it is hidden inside
/// [`RendererState`] and lent to [`RenderGraph`] each frame via a
/// mutable reference.
///
/// # Why separate?
///
/// A [`RenderGraph`] has a per-frame lifetime `'a` tied to the
/// [`FrameArena`].  Storing `Vec`s directly on that view would
/// require reconstructing them every frame, wasting the capacity
/// that `Vec::clear()` preserves for free.
pub struct GraphStorage {
    /// Attempt lifecycle and zero-allocation health counters. This is not
    /// cleared with topology or frame-owned physical resources.
    pub(crate) diagnostics: RdgStandaloneDiagnosticsStore,
    /// All pass records for the current frame.
    pub passes: Vec<PassRecord>,
    /// All resource records for the current frame.
    pub resources: Vec<ResourceRecord>,
    /// Compiled execution queue (topologically sorted pass indices).
    pub execution_queue: Vec<usize>,
    /// Capacity-retaining flat logical-read table. Pass records address this
    /// table by range.
    logical_reads: Vec<ErasedResourceNodeId>,
    /// Capacity-retaining flat physical producer table. Pass records address
    /// this table by range after topology compilation.
    physical_dependencies: Vec<usize>,
    /// Whether topology compilation has started for the current graph attempt.
    ///
    /// Compilation derives mutable state in-place (reference counts, physical
    /// dependencies, execution order, and draw-plan liveness), so it is a
    /// deliberately one-shot phase rather than an idempotent query.
    topology_compiled: bool,

    /// Per-frame logical import deduplication for persistent textures.
    persistent_texture_lookup: FxHashMap<PersistentTextureId, PersistentTextureImport>,
    /// Logical asset roots and destructible live-prepare materialization slots.
    pub(crate) asset_texture_imports: AssetTextureImportStore,
    /// Logical scene-buffer roots and destructible live materialization slots.
    pub(crate) scene_buffer_imports: SceneBufferImportStore,
    /// Grouped history publications evaluated after DPE and applied post-submit.
    history_commit_requests: Vec<HistoryCommitRequest>,
    /// Ordinary cache publications evaluated after DPE and applied post-submit.
    cache_commit_requests: Vec<CacheCommitRequest>,
    /// `None` until canonical ordinary-cache rollback eligibility has been
    /// finalized after DPE; `Some` records whether any participating invalid
    /// cache member was touched by a live pass. The phase is idempotent for one
    /// graph attempt.
    persistent_cache_finalization: Option<bool>,

    /// Drop-bearing physical mip-generation attempts staged for this frame.
    /// Pass nodes retain only [`MipmapBatchId`] values into these arrays.
    pub(crate) mipmap_batches: Vec<Option<MipmapRequestBatch>>,
    /// Physical-generation encode claims for this graph attempt.
    pub(crate) mipmap_encode_coordinator: MipmapEncodeCoordinator,

    /// Destructor-aware ownership for cached bind groups used by this graph
    /// attempt. Pass nodes carry only generational IDs into this store.
    pub(crate) prepared_bind_groups: PreparedBindGroupStore,

    /// Destructor-aware physical resources and compact commands for scene
    /// draws selected during post-prepare baking.
    pub(crate) draw_plans: DrawPlanStore,
    /// Canonical per-pass plan demands in one capacity-retaining flat table.
    /// Pass records address this table by range.
    pub(crate) draw_plan_demands: Vec<DrawPlanId>,
    /// Backend-neutral scene-submission plans reserved for this attempt.
    pub(super) scene_submissions: SceneSubmissionStore,
    /// Canonical per-pass scene-submission demands in first-occurrence order.
    scene_submission_demands: Vec<SceneSubmissionPlanId>,
    /// Graph-attempt source records materialized only for the source-item
    /// union selected by DPE-live draw plans.
    pub(crate) prepared_draws: PreparedDrawStore,
    /// Exact CPU material owners and texture snapshots captured before graph
    /// construction, then opened only for the DPE-live source union.
    pub(crate) material_plans: MaterialPlanStore,
    /// Single logical Group 0 plan, projected DPE demand, and destructible
    /// attempt-owned physical results.
    pub(crate) global_state_attempt: GlobalStateAttempt,

    /// Exact Gaussian asset versions captured for this graph attempt. Arc
    /// owners live here because the frame arena intentionally skips Drop.
    #[cfg(feature = "3dgs")]
    pub(crate) gaussian_frame_batches: Vec<Option<GaussianFrameBatch>>,

    // /// Name-based resource registry for self-wiring passes.
    // resource_registry: FxHashMap<&'static str, TextureNodeId>,

    // --- Compile-time scratch buffers (zero-alloc across frames) ---
    compile_stack: Vec<usize>,
    compile_in_degrees: Vec<usize>,
    compile_ready_heap: BinaryHeap<ReadyNode>,
    /// CSR reverse adjacency from producer pass to DPE-live consumers.
    compile_reverse_offsets: Vec<usize>,
    compile_reverse_edges: Vec<usize>,
    compile_reverse_cursors: Vec<usize>,

    compile_max_ext_indices: Vec<usize>,

    /// One-shot draw-demand union scratch, retained at the plan/resource
    /// high-water marks and generation-cleared between operations.
    draw_plan_demand_scratch: DenseGenerationScratch,
    draw_plan_resource_scratch: DenseGenerationScratch,
    scene_submission_demand_scratch: DenseGenerationScratch,
    scene_submission_resource_scratch: DenseGenerationScratch,
    physical_dependency_scratch: DenseGenerationScratch,
    /// Raw demand IDs for the pass whose setup closure is currently running.
    /// Cleared without releasing capacity before each `add_pass` call.
    pub(crate) draw_plan_raw_demands: Vec<DrawPlanId>,
    /// Raw opaque submission IDs for the pass whose setup is in progress.
    pub(crate) scene_submission_raw_demands: Vec<SceneSubmissionPlanId>,
    /// Setup is deliberately non-reentrant. External callers cannot reach the
    /// graph through PassBuilder, and internal nested construction would make
    /// one-shot finalization and consumer-edge order ambiguous.
    pass_setup_active: bool,

    /// Group-name stack for logical subgraph tagging.
    #[cfg(feature = "rdg_inspector")]
    current_group_stack: Vec<&'static str>,

    #[cfg(debug_assertions)]
    prev_execution_names: Vec<&'static str>,
}

impl Default for GraphStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphStorage {
    #[must_use]
    pub fn new() -> Self {
        Self {
            diagnostics: RdgStandaloneDiagnosticsStore::default(),
            passes: Vec::new(),
            resources: Vec::new(),
            execution_queue: Vec::new(),
            logical_reads: Vec::with_capacity(256),
            physical_dependencies: Vec::with_capacity(256),
            topology_compiled: false,
            persistent_texture_lookup: FxHashMap::default(),
            asset_texture_imports: AssetTextureImportStore::new(),
            scene_buffer_imports: SceneBufferImportStore::new(),
            history_commit_requests: Vec::new(),
            cache_commit_requests: Vec::new(),
            persistent_cache_finalization: None,
            mipmap_batches: Vec::new(),
            mipmap_encode_coordinator: MipmapEncodeCoordinator::default(),
            prepared_bind_groups: PreparedBindGroupStore::new(),
            draw_plans: DrawPlanStore::new(),
            draw_plan_demands: Vec::with_capacity(64),
            scene_submissions: SceneSubmissionStore::new(),
            scene_submission_demands: Vec::with_capacity(64),
            prepared_draws: PreparedDrawStore::new(),
            material_plans: MaterialPlanStore::new(),
            global_state_attempt: GlobalStateAttempt::default(),
            #[cfg(feature = "3dgs")]
            gaussian_frame_batches: Vec::new(),
            // resource_registry: FxHashMap::default(),
            compile_stack: Vec::new(),
            compile_in_degrees: Vec::new(),
            compile_ready_heap: BinaryHeap::new(),
            compile_reverse_offsets: Vec::new(),
            compile_reverse_edges: Vec::new(),
            compile_reverse_cursors: Vec::new(),
            compile_max_ext_indices: Vec::new(),
            draw_plan_demand_scratch: DenseGenerationScratch::default(),
            draw_plan_resource_scratch: DenseGenerationScratch::default(),
            scene_submission_demand_scratch: DenseGenerationScratch::default(),
            scene_submission_resource_scratch: DenseGenerationScratch::default(),
            physical_dependency_scratch: DenseGenerationScratch::default(),
            draw_plan_raw_demands: Vec::with_capacity(64),
            scene_submission_raw_demands: Vec::with_capacity(64),
            pass_setup_active: false,
            #[cfg(feature = "rdg_inspector")]
            current_group_stack: Vec::new(),
            #[cfg(debug_assertions)]
            prev_execution_names: Vec::new(),
        }
    }

    /// Clears all per-frame data while retaining heap capacity.
    pub fn clear(&mut self) {
        self.passes.clear();
        self.resources.clear();
        self.execution_queue.clear();
        self.logical_reads.clear();
        self.physical_dependencies.clear();
        self.compile_reverse_offsets.clear();
        self.compile_reverse_edges.clear();
        self.compile_reverse_cursors.clear();
        self.draw_plan_demands.clear();
        self.draw_plan_raw_demands.clear();
        self.scene_submission_demands.clear();
        self.scene_submission_raw_demands.clear();
        self.pass_setup_active = false;
        self.topology_compiled = false;
        self.persistent_texture_lookup.clear();
        self.history_commit_requests.clear();
        self.cache_commit_requests.clear();
        self.persistent_cache_finalization = None;
        self.release_frame_owners();
        // self.resource_registry.clear();
        #[cfg(feature = "rdg_inspector")]
        self.current_group_stack.clear();
    }

    /// Latest terminal standalone attempt, retained independently from graph
    /// topology. Renderer-owned attempts are exposed by
    /// `Renderer::latest_rdg_diagnostics`.
    #[must_use]
    pub fn latest_diagnostics(&self) -> Option<RdgFrameDiagnostics> {
        self.diagnostics.latest()
    }

    /// Logical resources read by one pass, in declaration order.
    #[must_use]
    pub fn pass_reads(&self, pass_index: usize) -> &[ErasedResourceNodeId] {
        &self.logical_reads[self.passes[pass_index].logical_read_range.clone()]
    }

    /// Enumerates consumer pass indices for one resource in pass/read
    /// declaration order.
    ///
    /// This is a linear topology-inspection view derived from the canonical
    /// flat pass-read table. Runtime compilation should use pass-local reads or
    /// compiled physical adjacency instead of repeatedly scanning this view.
    pub fn resource_consumers(&self, resource_index: usize) -> impl Iterator<Item = usize> + '_ {
        assert!(
            resource_index < self.resources.len(),
            "resource consumer query is outside this graph"
        );
        self.passes
            .iter()
            .enumerate()
            .flat_map(move |(pass_index, _)| {
                let edge_count = self
                    .pass_reads(pass_index)
                    .iter()
                    .filter(|read| read.index() as usize == resource_index)
                    .count();
                std::iter::repeat_n(pass_index, edge_count)
            })
    }

    /// Unique physical producer passes required by one pass, in the order
    /// their first logical read declared them.
    ///
    /// The slice is empty before topology compilation. If compilation aborts,
    /// already-derived passes retain a valid prefix for post-mortem inspection;
    /// the next graph attempt clears every range and reuses the flat capacity.
    #[must_use]
    pub fn pass_physical_dependencies(&self, pass_index: usize) -> &[usize] {
        &self.physical_dependencies[self.passes[pass_index].physical_dependency_range.clone()]
    }

    #[cfg(any(test, feature = "rdg_bench_support"))]
    pub(crate) fn compiled_reverse_consumers(&self, producer_index: usize) -> &[usize] {
        let start = self.compile_reverse_offsets[producer_index];
        let end = self.compile_reverse_offsets[producer_index + 1];
        &self.compile_reverse_edges[start..end]
    }

    /// Append one logical read to the currently open pass range.
    pub(crate) fn append_pass_read(&mut self, pass_index: usize, read: ErasedResourceNodeId) {
        assert!(
            self.pass_setup_active,
            "logical reads may only be appended during pass setup"
        );
        assert_eq!(
            pass_index + 1,
            self.passes.len(),
            "logical reads must target the currently open final pass"
        );
        append_logical_read(
            &mut self.logical_reads,
            &mut self.passes[pass_index],
            &mut self.resources,
            read,
        );
    }

    /// Make one pass's logical-read view empty and remove its resource-side
    /// consumer edges without shifting later pass ranges. The retained flat
    /// bytes are discarded on the next clear.
    fn clear_pass_reads(&mut self, pass_index: usize) {
        let range = self.passes[pass_index].logical_read_range.clone();
        for read_index in range.clone() {
            let resource_index = self.logical_reads[read_index].index() as usize;
            if let Some(resource) = self.resources.get_mut(resource_index) {
                resource.remove_consumer_edge();
            }
        }
        let start = range.start;
        self.passes[pass_index].logical_read_range = start..start;
    }

    /// Roll back only logical-read staging when a setup closure unwinds.
    /// Other attempt topology remains forensic-only until the next clear, but
    /// no trailing reads or consumer edges may leak into that next attempt.
    fn rollback_active_pass_reads(&mut self) {
        if !self.pass_setup_active {
            return;
        }
        let Some(pass_index) = self.passes.len().checked_sub(1) else {
            return;
        };
        let range = self.passes[pass_index].logical_read_range.clone();
        let logical_read_count = self.logical_reads.len();
        if range.start <= logical_read_count {
            let valid_end = range.end.min(logical_read_count);
            for read_index in range.start..valid_end {
                let resource_index = self.logical_reads[read_index].index() as usize;
                if let Some(resource) = self.resources.get_mut(resource_index) {
                    resource.remove_consumer_edge();
                }
            }
            self.logical_reads.truncate(range.start);
        }
        let cleared_start = range.start.min(self.logical_reads.len());
        self.passes[pass_index].logical_read_range = cleared_start..cleared_start;
    }

    /// Canonicalize raw draw-plan demands and attach their logical texture
    /// reads after one pass setup closure has completed.
    ///
    /// Both dense sets are generation-stamped and storage-owned, so this does
    /// no temporary hashing or dependency cloning. First-demand and
    /// first-resource order are preserved.
    fn finalize_pass_draw_plan_demands(&mut self, pass_index: usize) {
        let canonical_start = self.draw_plan_demands.len();
        if self.draw_plan_raw_demands.is_empty() {
            self.passes[pass_index].draw_plan_demand_range = canonical_start..canonical_start;
            return;
        }

        self.draw_plan_demand_scratch
            .begin(self.draw_plans.plan_count());
        let raw_demand_count = self.draw_plan_raw_demands.len();
        let mut unique_demand_count = 0;
        for read_index in 0..raw_demand_count {
            let id = self.draw_plan_raw_demands[read_index];
            let dense_index = self
                .draw_plans
                .reserved_index(id)
                .expect("draw plan demand does not belong to this graph attempt");
            if self.draw_plan_demand_scratch.insert(dense_index) {
                self.draw_plan_raw_demands[unique_demand_count] = id;
                unique_demand_count += 1;
            }
        }
        self.draw_plan_raw_demands.truncate(unique_demand_count);
        self.draw_plan_demands
            .extend_from_slice(&self.draw_plan_raw_demands);
        let canonical_end = self.draw_plan_demands.len();
        self.passes[pass_index].draw_plan_demand_range = canonical_start..canonical_end;

        self.draw_plan_resource_scratch.begin(self.resources.len());
        let existing_read_count = self.pass_reads(pass_index).len();
        for read_index in 0..existing_read_count {
            let read = self.pass_reads(pass_index)[read_index];
            self.draw_plan_resource_scratch
                .insert(read.index() as usize);
        }

        for demand_index in canonical_start..canonical_end {
            let id = self.draw_plan_demands[demand_index];
            let dependency_key = self.draw_plans.mark_topology_demanded(id);
            let dependencies = self.draw_plans.texture_dependencies(dependency_key);
            let logical_reads = &mut self.logical_reads;
            let pass = &mut self.passes[pass_index];
            let resources = &mut self.resources;
            let resource_scratch = &mut self.draw_plan_resource_scratch;
            for &dependency in dependencies {
                let resource_index = dependency.index() as usize;
                assert!(
                    resource_index < resources.len(),
                    "pass '{}' demanded a draw-plan texture from another graph",
                    pass.name
                );
                if !resource_scratch.insert(resource_index) {
                    continue;
                }

                append_logical_read(logical_reads, pass, resources, dependency.erase());
            }
        }
    }

    /// Canonicalize backend-neutral submission demands and attach their mixed
    /// texture/buffer dependencies once after pass setup.
    ///
    /// Both dense sets retain capacity across attempts. Plan order follows
    /// first demand; resource order follows the first occurrence across direct
    /// reads, legacy DrawPlan dependencies, and submission dependencies.
    fn finalize_pass_scene_submission_demands(&mut self, pass_index: usize) {
        let canonical_start = self.scene_submission_demands.len();
        if self.scene_submission_raw_demands.is_empty() {
            self.passes[pass_index].scene_submission_demand_range =
                canonical_start..canonical_start;
            return;
        }

        self.scene_submission_demand_scratch
            .begin(self.scene_submissions.plan_count());
        let raw_demand_count = self.scene_submission_raw_demands.len();
        let mut unique_demand_count = 0;
        for raw_index in 0..raw_demand_count {
            let id = self.scene_submission_raw_demands[raw_index];
            let dense_index = self
                .scene_submissions
                .reserved_index(id)
                .expect("scene-submission demand does not belong to this graph attempt");
            if self.scene_submission_demand_scratch.insert(dense_index) {
                self.scene_submission_raw_demands[unique_demand_count] = id;
                unique_demand_count += 1;
            }
        }
        self.scene_submission_raw_demands
            .truncate(unique_demand_count);
        self.scene_submission_demands
            .extend_from_slice(&self.scene_submission_raw_demands);
        let canonical_end = self.scene_submission_demands.len();
        self.passes[pass_index].scene_submission_demand_range = canonical_start..canonical_end;

        self.scene_submission_resource_scratch
            .begin(self.resources.len());
        let existing_read_count = self.pass_reads(pass_index).len();
        for read_index in 0..existing_read_count {
            let read = self.pass_reads(pass_index)[read_index];
            self.scene_submission_resource_scratch
                .insert(read.index() as usize);
        }

        for demand_index in canonical_start..canonical_end {
            let id = self.scene_submission_demands[demand_index];
            let dependencies = self
                .scene_submissions
                .dependencies(id)
                .expect("canonical scene-submission demand became stale");
            for &dependency in dependencies {
                let resource_index = dependency.index() as usize;
                assert!(
                    resource_index < self.resources.len(),
                    "pass '{}' demanded a scene-submission resource from another graph",
                    self.passes[pass_index].name
                );
                if !self
                    .scene_submission_resource_scratch
                    .insert(resource_index)
                {
                    continue;
                }
                append_logical_read(
                    &mut self.logical_reads,
                    &mut self.passes[pass_index],
                    &mut self.resources,
                    dependency,
                );
            }
        }
    }

    #[inline]
    #[cfg(any(test, feature = "rdg_bench_support"))]
    pub(crate) fn pass_draw_plan_demands(&self, pass_index: usize) -> &[DrawPlanId] {
        &self.draw_plan_demands[self.passes[pass_index].draw_plan_demand_range.clone()]
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn pass_scene_submission_demands(
        &self,
        pass_index: usize,
    ) -> &[SceneSubmissionPlanId] {
        &self.scene_submission_demands[self.passes[pass_index]
            .scene_submission_demand_range
            .clone()]
    }

    /// Immediately releases Drop-bearing owners captured by one graph attempt.
    ///
    /// Topology and execution records intentionally remain available after a
    /// successful frame for inspection. Keeping owner cleanup separate from
    /// [`Self::clear`] also lets the engine's frame-attempt guard release
    /// abandoned WGPU and asset owners during unwinding instead of retaining
    /// them until the next graph is constructed.
    pub(crate) fn release_frame_owners(&mut self) {
        self.asset_texture_imports.clear();
        self.scene_buffer_imports.clear();
        self.mipmap_batches.clear();
        self.mipmap_encode_coordinator.clear();
        self.prepared_bind_groups.clear();
        // Source records address resource slots owned by draw_plans, so drop
        // those records before resetting the sibling slot tables/token.
        self.prepared_draws.clear();
        self.material_plans.clear();
        self.scene_submissions.clear();
        self.draw_plans.clear();
        self.global_state_attempt.clear();
        #[cfg(feature = "3dgs")]
        self.gaussian_frame_batches.clear();
    }

    /// Invalidates arena-backed runtime nodes while retaining pure topology
    /// metadata for inspection after the frame finishes.
    fn invalidate_pass_nodes(&mut self) {
        for pass in &mut self.passes {
            pass.node = None;
        }
    }

    fn stage_mipmap_requests(&mut self, requests: Vec<MipmapRequest>) -> Option<MipmapBatchId> {
        if requests.is_empty() {
            return None;
        }

        let batch = MipmapBatchId::from_index(self.mipmap_batches.len());
        self.mipmap_batches
            .push(Some(MipmapRequestBatch::new(requests)));
        Some(batch)
    }

    fn commit_submitted_mipmap_batch(&mut self, batch: MipmapBatchId) -> bool {
        let attempt = self.mipmap_batches[batch.index()]
            .take()
            .expect("mipmap request batch already released");
        if !attempt.was_encoded() {
            return false;
        }
        attempt.publish_submitted();
        true
    }

    #[cfg(feature = "3dgs")]
    fn stage_gaussian_frame_batch(
        &mut self,
        clouds: Vec<GaussianFrameCloud>,
    ) -> Option<GaussianFrameBatchId> {
        if clouds.is_empty() {
            return None;
        }

        let batch = GaussianFrameBatchId::from_index(self.gaussian_frame_batches.len());
        self.gaussian_frame_batches
            .push(Some(GaussianFrameBatch::new(clouds)));
        Some(batch)
    }

    #[cfg(feature = "3dgs")]
    fn release_gaussian_frame_batch(&mut self, batch: GaussianFrameBatchId) {
        self.gaussian_frame_batches[batch.index()]
            .take()
            .expect("Gaussian frame batch already released");
    }

    /// Dumps a compact, human-oriented Mermaid view of the current topology.
    #[must_use]
    pub fn dump_mermaid(&self) -> String {
        self.dump_mermaid_with_mode(MermaidDumpMode::Compact)
    }

    /// Dumps the exact pass/resource topology as a Mermaid flowchart.
    #[must_use]
    pub fn dump_mermaid_full(&self) -> String {
        self.dump_mermaid_with_mode(MermaidDumpMode::Full)
    }

    /// Dumps the current Render Graph topology using the requested detail.
    #[must_use]
    pub fn dump_mermaid_with_mode(&self, mode: MermaidDumpMode) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        writeln!(&mut out, "flowchart TD").unwrap();

        writeln!(
            &mut out,
            "    classDef alive fill:#2b3c5a,stroke:#4a6f9f,stroke-width:2px,color:#fff,rx:5,ry:5;"
        )
        .unwrap();

        let compact_asset_inputs = matches!(mode, MermaidDumpMode::Compact);
        let mut asset_input_edge_counts =
            compact_asset_inputs.then(|| vec![0usize; self.passes.len()]);
        let mut asset_input_edges = FxHashSet::default();
        let mut declared_asset_inputs = 0usize;
        let mut live_asset_resources =
            compact_asset_inputs.then(|| vec![false; self.resources.len()]);
        let mut read_edges = String::new();

        if compact_asset_inputs {
            for res in &self.resources {
                if res.asset_texture_import_id().is_none() {
                    continue;
                }
                if !res.is_external || res.producer.is_some() || res.alias_of.is_some() {
                    continue;
                }
                declared_asset_inputs += 1;
            }
        }

        // Logical reads are the canonical consumer topology. Scan them exactly
        // once to derive both human-facing edges and compact Asset Texture
        // aggregation; ResourceRecord intentionally stores only an edge count.
        for consumer_index in 0..self.passes.len() {
            for &read_id in self.pass_reads(consumer_index) {
                let resource_index = read_id.index() as usize;
                let res = &self.resources[resource_index];
                let asset_import = res.asset_texture_import_id();
                let compact_asset_root = compact_asset_inputs
                    && asset_import.is_some()
                    && res.is_external
                    && res.producer.is_none()
                    && res.alias_of.is_none();

                if compact_asset_root {
                    if self.passes[consumer_index].reference_count > 0 {
                        live_asset_resources
                            .as_mut()
                            .expect("compact Asset Texture live set missing")[resource_index] =
                            true;
                    }
                    let import = asset_import.expect("compact Asset Texture import missing");
                    if asset_input_edges.insert((import, consumer_index)) {
                        asset_input_edge_counts
                            .as_mut()
                            .expect("compact Asset Texture counts missing")[consumer_index] += 1;
                    }
                    continue;
                }

                if res.is_external
                    && res.producer.is_none()
                    && !(compact_asset_inputs && asset_import.is_some())
                {
                    writeln!(
                        &mut read_edges,
                        "    IN_{resource_index} -.-> P{consumer_index};",
                    )
                    .unwrap();
                }

                if let Some(producer_index) = res.producer {
                    let edge_style = if res.alias_of.is_some() { "==>" } else { "-->" };
                    let name = Self::escape_mermaid_label(res.name);
                    writeln!(
                        &mut read_edges,
                        "    P{producer_index} {edge_style}|\"{name}\"| P{consumer_index};"
                    )
                    .unwrap();
                }
            }
        }
        let live_asset_inputs = live_asset_resources
            .as_deref()
            .map_or(0, |live| live.iter().filter(|&&is_live| is_live).count());
        writeln!(
            &mut out,
            "    classDef dead fill:#222,stroke:#555,stroke-width:2px,stroke-dasharray: 5 5,color:#777,rx:5,ry:5;"
        )
        .unwrap();
        writeln!(
            &mut out,
            "    classDef external_out fill:#5a2b3c,stroke:#9f4a6f,stroke-width:2px,color:#fff;"
        )
        .unwrap();
        writeln!(
            &mut out,
            "    classDef external_in fill:#3c5a2b,stroke:#6f9f4a,stroke-width:2px,color:#fff;"
        )
        .unwrap();

        if compact_asset_inputs && declared_asset_inputs > 0 {
            if self.topology_compiled {
                writeln!(
                    &mut out,
                    "    ASSET_TEXTURES_IN[\"Asset Textures<br/>declared {declared_asset_inputs} · live {live_asset_inputs}\"]:::external_in"
                )
                .unwrap();
            } else {
                writeln!(
                    &mut out,
                    "    ASSET_TEXTURES_IN[\"Asset Textures<br/>declared {declared_asset_inputs} · live pending\"]:::external_in"
                )
                .unwrap();
            }
        }

        #[cfg(feature = "rdg_inspector")]
        {
            #[derive(Default)]
            struct GroupNode {
                name: &'static str,
                passes: Vec<usize>,
                children: Vec<GroupNode>,
            }

            impl GroupNode {
                fn insert(&mut self, path: &[&'static str], pass_idx: usize) {
                    if path.is_empty() {
                        self.passes.push(pass_idx);
                    } else {
                        let next_name = path[0];
                        if let Some(child) = self.children.iter_mut().find(|c| c.name == next_name)
                        {
                            child.insert(&path[1..], pass_idx);
                        } else {
                            let mut new_child = GroupNode {
                                name: next_name,
                                ..Default::default()
                            };
                            new_child.insert(&path[1..], pass_idx);
                            self.children.push(new_child);
                        }
                    }
                }

                fn write_mermaid(
                    &self,
                    out: &mut String,
                    storage: &GraphStorage,
                    depth: usize,
                    next_group_id: &mut usize,
                ) {
                    use std::fmt::Write;
                    const STYLES: &[(&str, &str)] = &[
                        ("#3b82f614", "#3b82f6"),
                        ("#10b98114", "#10b981"),
                        ("#8b5cf614", "#8b5cf6"),
                        ("#f59e0b14", "#f59e0b"),
                        ("#ef444414", "#ef4444"),
                        ("#ec489914", "#ec4899"),
                        ("#06b6d414", "#06b6d4"),
                    ];

                    let indent = "    ".repeat(depth);
                    let group_id = if self.name.is_empty() {
                        None
                    } else {
                        let id = format!("G{}", *next_group_id);
                        *next_group_id += 1;
                        Some(id)
                    };

                    if let Some(group_id) = &group_id {
                        let label = GraphStorage::escape_mermaid_label(self.name);
                        writeln!(out, "{indent}subgraph {group_id} [\"{label}\"]").unwrap();
                        writeln!(out, "{indent}    direction TB").unwrap();
                    }

                    let inner_indent = if self.name.is_empty() {
                        indent.clone()
                    } else {
                        format!("{indent}    ")
                    };

                    for &pass_idx in &self.passes {
                        GraphStorage::write_pass_node(
                            out,
                            pass_idx,
                            &storage.passes[pass_idx],
                            &inner_indent,
                        );
                    }

                    for child in &self.children {
                        child.write_mermaid(
                            out,
                            storage,
                            if self.name.is_empty() {
                                depth
                            } else {
                                depth + 1
                            },
                            next_group_id,
                        );
                    }

                    if let Some(group_id) = &group_id {
                        writeln!(out, "{indent}end").unwrap();

                        let hash = self
                            .name
                            .bytes()
                            .fold(0usize, |acc, b| acc.wrapping_add(b as usize));

                        let (bg, stroke) = STYLES[hash % STYLES.len()];

                        writeln!(
                            out,
                            "{indent}style {group_id} fill:{bg},stroke:{stroke},stroke-width:2px,stroke-dasharray: 5 5,color:#fff,rx:10,ry:10"
                        ).unwrap();
                    }
                }
            }

            let mut root = GroupNode::default();
            for (i, pass) in self.passes.iter().enumerate() {
                root.insert(&pass.groups, i);
            }

            let mut next_group_id = 0;
            root.write_mermaid(&mut out, self, 1, &mut next_group_id);
        }

        #[cfg(not(feature = "rdg_inspector"))]
        {
            writeln!(&mut out, "\n    %% --- Passes ---").unwrap();
            for (i, pass) in self.passes.iter().enumerate() {
                Self::write_pass_node(&mut out, i, pass, "    ");
            }
        }

        writeln!(&mut out, "\n    %% --- Data Flow (Edges) ---").unwrap();

        for (i, res) in self.resources.iter().enumerate() {
            if res.is_external {
                if compact_asset_inputs && res.asset_texture_import_id().is_some() {
                    continue;
                }
                if res.producer.is_none() {
                    let name = Self::escape_mermaid_label(res.name);
                    writeln!(&mut out, "    IN_{i}[\"{name}\"]:::external_in").unwrap();
                }
                if self.is_declared_terminal_output(i) {
                    let name = Self::escape_mermaid_label(res.name);
                    writeln!(&mut out, "    OUT_{i}[/\"{name}\"/]:::external_out").unwrap();
                }
            }
        }

        out.push_str(&read_edges);

        // Preserve the declared-output contract after edge contraction: an
        // aliased source ResourceRecord can inherit output-root metadata, but
        // only IDs still present in a pass's writes are graph outputs.
        for (producer_index, pass) in self.passes.iter().enumerate() {
            for &write_id in &pass.writes {
                let res = &self.resources[write_id.index() as usize];
                if Self::is_terminal_external_output(res) {
                    writeln!(
                        &mut out,
                        "    P{producer_index} --> OUT_{};",
                        write_id.index()
                    )
                    .unwrap();
                }
            }
        }

        if let Some(asset_input_edge_counts) = asset_input_edge_counts {
            for (pass_idx, count) in asset_input_edge_counts.into_iter().enumerate() {
                if count == 0 {
                    continue;
                }
                writeln!(
                    &mut out,
                    "    ASSET_TEXTURES_IN -.->|\"{count} texture{}\"| P{pass_idx};",
                    if count == 1 { "" } else { "s" }
                )
                .unwrap();
            }
        }

        out
    }

    fn write_pass_node(out: &mut String, index: usize, pass: &PassRecord, indent: &str) {
        use std::fmt::Write;
        let class = if pass.reference_count > 0 {
            "alive"
        } else {
            "dead"
        };
        let name = Self::escape_mermaid_label(pass.name);
        writeln!(out, "{indent}P{index}([\"{name}\"]):::{class}").unwrap();
    }

    fn escape_mermaid_label(label: &str) -> String {
        let mut escaped = String::with_capacity(label.len());
        for character in label.chars() {
            match character {
                '&' => escaped.push_str("&amp;"),
                '"' => escaped.push_str("&quot;"),
                '<' => escaped.push_str("&lt;"),
                '>' => escaped.push_str("&gt;"),
                '\n' => escaped.push_str("<br/>"),
                '\r' => {}
                _ => escaped.push(character),
            }
        }
        escaped
    }

    #[inline]
    fn is_terminal_external_output(res: &ResourceRecord) -> bool {
        res.is_external
            && res.is_output_root
            && res.producer.is_some()
            && res.consumer_edge_count() == 0
    }

    #[inline]
    fn is_declared_terminal_output(&self, resource_index: usize) -> bool {
        let res = &self.resources[resource_index];
        Self::is_terminal_external_output(res)
            && res.producer.is_some_and(|producer_index| {
                self.passes[producer_index]
                    .writes
                    .iter()
                    .any(|write| write.index() as usize == resource_index)
            })
    }
}

/// Zero-drop marker trait for PassNodes.
trait AssertNoDrop {
    const VALID: ();
}

// Any PassNode that requires Drop will fail to compile due to the static assertion in AssertNoDrop::VALID.
impl<T> AssertNoDrop for T {
    // This constant will fail to compile if T needs Drop, which enforces that PassNodes cannot have destructors or own heap data.
    const VALID: () = assert!(
        !std::mem::needs_drop::<Self>(),
        "FATAL ERROR: PassNode MUST NOT implement Drop or contain heap allocations (like Arc, Vec, String). It must be a POD type or only hold references."
    );
}

/// Declarative Render Graph — per-frame view with lifetime `'a`.
///
/// Provides the full graph construction, compilation, and execution API.
/// Created each frame by [`FrameComposer`] from a [`GraphStorage`]
/// reference and a [`FrameArena`] reference, ensuring that:
///
/// - All `PassNode<'a>` values can borrow data with frame scope.
/// - The underlying `Vec` capacity is silently reused across frames.
/// - The `RenderGraph` is *not* `'static`, accurately reflecting its
///   single-frame semantic lifetime.
pub struct RenderGraph<'a> {
    pub(crate) storage: &'a mut GraphStorage,
    arena: &'a FrameArena,
    renderer_diagnostics: Option<&'a RdgDiagnosticsRecorder>,
    standalone_diagnostics: Option<RdgStandaloneAttempt>,
}

impl Drop for RenderGraph<'_> {
    fn drop(&mut self) {
        // Raw demand staging is meaningful only while one setup closure is
        // running. In particular, do not let a setup panic leave stale IDs for
        // retained-topology inspection or a later attempt.
        self.storage.rollback_active_pass_reads();
        self.storage.draw_plan_raw_demands.clear();
        self.storage.scene_submission_raw_demands.clear();
        self.storage.pass_setup_active = false;
        // NodeSlot erases the arena lifetime behind a raw trait-object pointer.
        // Remove every slot at the graph boundary so resetting FrameArena can
        // never leave a callable dangling node in retained topology metadata.
        self.storage.invalidate_pass_nodes();
        // Successful submission paths have already consumed/published their
        // producer attempts; prepared bind-group clones intentionally remain
        // owned through execute and are released here. On panic or early abort
        // this is also the frame-attempt RAII boundary that prevents
        // GraphStorage from pinning physical textures, mip requests, Gaussian
        // asset Arcs, or WGPU bind groups until a later graph is constructed.
        self.storage.release_frame_owners();
        // Renderer attempts remain owned by FrameComposer so target/present
        // paths can publish one terminal snapshot after graph owners drop.
        // A standalone graph owns its plain active snapshot directly and
        // publishes it here without adding interior mutability to public
        // GraphStorage.
        if let Some(attempt) = self.standalone_diagnostics.take() {
            self.storage.diagnostics.publish(attempt);
        }
    }
}

impl<'a> RenderGraph<'a> {
    /// Creates a new per-frame render graph view.
    ///
    /// Clears the storage's per-frame data and records the frame config
    /// for pass-level access via [`PassBuilder::frame_config`].
    pub fn new(storage: &'a mut GraphStorage, arena: &'a FrameArena) -> Self {
        storage.clear();
        let standalone_diagnostics = storage.diagnostics.begin_attempt();
        Self {
            storage,
            arena,
            renderer_diagnostics: None,
            standalone_diagnostics: Some(standalone_diagnostics),
        }
    }

    /// Benchmark-only constructor for same-binary diagnostics A/B measurements.
    ///
    /// The disabled branch preserves the exact graph topology path while
    /// omitting attempt creation, counter updates, and terminal publication.
    /// This API is crate-private and absent from shipping builds.
    #[cfg(any(test, feature = "rdg_bench_support"))]
    pub(crate) fn new_for_benchmark(
        storage: &'a mut GraphStorage,
        arena: &'a FrameArena,
        diagnostics_enabled: bool,
    ) -> Self {
        storage.clear();
        let standalone_diagnostics =
            diagnostics_enabled.then(|| storage.diagnostics.begin_attempt());
        Self {
            storage,
            arena,
            renderer_diagnostics: None,
            standalone_diagnostics,
        }
    }

    /// Creates the graph view for an attempt already started by `Renderer`.
    pub(crate) fn from_started_renderer_attempt(
        storage: &'a mut GraphStorage,
        arena: &'a FrameArena,
        diagnostics: &'a RdgDiagnosticsRecorder,
    ) -> Self {
        assert_eq!(
            diagnostics.active_owner(),
            Some(RdgAttemptOwner::Renderer),
            "FrameComposer constructed a RenderGraph without a Renderer attempt"
        );
        debug_assert!(storage.passes.is_empty());
        debug_assert!(storage.resources.is_empty());
        debug_assert!(storage.execution_queue.is_empty());
        debug_assert!(!storage.topology_compiled);
        diagnostics.enter_phase(RdgFramePhase::BuildGraph);
        Self {
            storage,
            arena,
            renderer_diagnostics: Some(diagnostics),
            standalone_diagnostics: None,
        }
    }

    /// Current in-progress diagnostics for this graph attempt.
    #[must_use]
    pub fn diagnostics(&self) -> RdgFrameDiagnostics {
        if let Some(diagnostics) = self.renderer_diagnostics {
            diagnostics
                .current()
                .expect("live RenderGraph lost its Renderer diagnostics attempt")
        } else {
            self.standalone_diagnostics
                .as_ref()
                .expect("live standalone RenderGraph lost its diagnostics attempt")
                .current()
        }
    }

    #[inline]
    fn enter_diagnostics_phase(&mut self, phase: RdgFramePhase) {
        if let Some(diagnostics) = self.renderer_diagnostics {
            diagnostics.enter_phase(phase);
        } else {
            #[cfg(any(test, feature = "rdg_bench_support"))]
            if self.standalone_diagnostics.is_none() {
                return;
            }
            self.standalone_diagnostics
                .as_mut()
                .expect("standalone diagnostics attempt missing")
                .enter_phase(phase);
        }
    }

    #[inline]
    fn record_declared_pass(&mut self) {
        if let Some(diagnostics) = self.renderer_diagnostics {
            diagnostics.record_declared_pass();
        } else {
            #[cfg(any(test, feature = "rdg_bench_support"))]
            if self.standalone_diagnostics.is_none() {
                return;
            }
            self.standalone_diagnostics
                .as_mut()
                .expect("standalone diagnostics attempt missing")
                .record_declared_pass();
        }
    }

    #[inline]
    fn record_folded_pass(&mut self) {
        if let Some(diagnostics) = self.renderer_diagnostics {
            diagnostics.record_folded_pass();
        } else {
            #[cfg(any(test, feature = "rdg_bench_support"))]
            if self.standalone_diagnostics.is_none() {
                return;
            }
            self.standalone_diagnostics
                .as_mut()
                .expect("standalone diagnostics attempt missing")
                .record_folded_pass();
        }
    }

    #[inline]
    fn record_compiled_live_passes(&mut self, live_passes: usize) {
        if let Some(diagnostics) = self.renderer_diagnostics {
            diagnostics.record_compiled_live_passes(live_passes);
        } else {
            #[cfg(any(test, feature = "rdg_bench_support"))]
            if self.standalone_diagnostics.is_none() {
                return;
            }
            self.standalone_diagnostics
                .as_mut()
                .expect("standalone diagnostics attempt missing")
                .record_compiled_live_passes(live_passes);
        }
    }

    /// Explicitly ends this graph attempt and releases its arena-backed nodes
    /// and Drop-bearing frame owners.
    ///
    /// A standalone graph is reported as
    /// [`RdgAttemptOutcome::CompletedWithoutSubmission`](crate::graph::RdgAttemptOutcome::CompletedWithoutSubmission).
    /// Calling `finish` also makes the lifetime boundary clear when code needs
    /// to borrow a Feature again after building the graph. This method does not
    /// submit GPU commands.
    pub fn finish(mut self) {
        if let Some(diagnostics) = self.standalone_diagnostics.as_mut() {
            diagnostics.mark_completed_without_submission();
        }
    }

    #[inline]
    pub(crate) fn assert_topology_mutable(&self) {
        assert!(
            self.storage.persistent_cache_finalization.is_none(),
            "render graph topology is frozen after persistent cache finalization"
        );
        assert!(
            !self.storage.topology_compiled,
            "render graph topology is frozen after compilation"
        );
    }

    /// Allocates a copy of `src` on the per-frame arena.
    ///
    /// Returns a slice reference valid for the frame lifetime `'a`.
    /// Requires `T: Copy` to ensure no `Drop` glue.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn alloc_slice<T: Copy>(&self, src: &[T]) -> &'a [T] {
        self.arena.alloc_slice_copy(src)
    }

    /// Allocates a mutable copy of `src` on the per-frame arena.
    #[inline]
    pub(crate) fn alloc_slice_mut<T: Copy>(&self, src: &[T]) -> &'a mut [T] {
        self.arena.alloc_slice_copy(src)
    }

    /// Allocates frame-scoped shared state used to hand prepared physical
    /// resources from a producer node to its consumers.
    ///
    /// The value must not require explicit teardown: the frame arena is reset
    /// as a whole after graph execution. Pass nodes should store only a
    /// reference to the returned value, keeping their own payloads POD.
    #[inline]
    pub(crate) fn alloc_shared_state<T>(&self, value: T) -> &'a T {
        self.arena.alloc(value)
    }

    /// Reserve a stable frame-local draw-plan ID during graph construction.
    ///
    /// The reservation owns no physical resources. Multiple passes reserving
    /// the same canonical key receive the same ID and may independently
    /// declare it through [`PassBuilder::demand_draw_plan`].
    pub(crate) fn reserve_draw_plan(&mut self, key: DrawPlanKey) -> DrawPlanId {
        self.assert_topology_mutable();
        self.storage.draw_plans.reserve_plan(key)
    }

    /// Wrap a valid legacy CPU-direct DrawPlan behind the backend-neutral
    /// scene-submission contract.
    ///
    /// This is the M3 transition bridge: Features can migrate to opaque
    /// submission IDs while the existing CPU command backend remains the
    /// private adapter selected by the core coordinator.
    #[allow(dead_code)] // Activated incrementally as Features migrate in M3.
    pub(crate) fn reserve_cpu_direct_scene_submission(
        &mut self,
        draw_plan: DrawPlanId,
        dependencies: &SceneSubmissionDependencies,
    ) -> SceneSubmissionPlanId {
        self.assert_topology_mutable();
        for &dependency in dependencies.as_slice() {
            assert!(
                (dependency.index() as usize) < self.storage.resources.len(),
                "scene-submission dependency belongs to another graph"
            );
        }
        let draw_dependency_key = self.storage.draw_plans.mark_topology_demanded(draw_plan);
        for &texture in self
            .storage
            .draw_plans
            .texture_dependencies(draw_dependency_key)
        {
            assert!(
                dependencies
                    .as_slice()
                    .iter()
                    .any(|&dependency| dependency == texture.erase()),
                "CpuDirect scene submission omitted a registered DrawPlan texture dependency"
            );
        }
        self.storage.scene_submissions.reserve_cpu_direct(
            &self.storage.draw_plans,
            draw_plan,
            dependencies,
        )
    }

    /// Test-only topology backend proving that indirect instance/argument/
    /// count buffers can participate as ordinary RDG dependencies without a
    /// Feature knowing the selected submission backend.
    #[cfg(test)]
    fn reserve_indirect_scene_submission_for_test(
        &mut self,
        stub_key: u64,
        dependencies: &SceneSubmissionDependencies,
    ) -> SceneSubmissionPlanId {
        self.assert_topology_mutable();
        for &dependency in dependencies.as_slice() {
            assert!(
                (dependency.index() as usize) < self.storage.resources.len(),
                "scene-submission dependency belongs to another graph"
            );
        }
        self.storage
            .scene_submissions
            .reserve_indirect_stub(stub_key, dependencies)
    }

    /// Capture the sole logical Group 0 plan for this graph attempt.
    /// Repeating the identical capture is idempotent; a different plan is an
    /// invalid multi-domain graph construction.
    pub(crate) fn capture_global_state_plan(
        &mut self,
        key: GlobalStateKey,
        sources: GlobalStateSources<'_>,
    ) {
        self.assert_topology_mutable();
        self.storage
            .global_state_attempt
            .set_plan(GlobalStatePlan::capture(key, sources));
    }

    /// Bind exact logical Asset Texture aliases to a canonical draw recipe.
    /// Passes that later demand the plan inherit these as normal RDG reads.
    pub(crate) fn set_draw_plan_texture_dependencies(
        &mut self,
        key: DrawPlanKey,
        dependencies: impl IntoIterator<Item = TextureNodeId>,
    ) {
        self.assert_topology_mutable();
        self.storage
            .draw_plans
            .assert_texture_dependencies_registerable(key);
        self.storage
            .draw_plan_resource_scratch
            .begin(self.storage.resources.len());
        let mut unique = SmallVec::<[TextureNodeId; 8]>::new();
        for dependency in dependencies {
            let resource_index = dependency.index() as usize;
            assert!(
                resource_index < self.storage.resources.len(),
                "draw-plan texture dependency belongs to another graph"
            );
            if self
                .storage
                .draw_plan_resource_scratch
                .insert(resource_index)
            {
                unique.push(dependency);
            }
        }
        self.storage
            .draw_plans
            .set_texture_dependencies(key, unique);
    }

    /// Moves Drop-bearing WGPU request owners into persistent GraphStorage and
    /// returns a Copy ID suitable for a no-drop pass node.
    pub(crate) fn stage_mipmap_requests(
        &mut self,
        requests: Vec<MipmapRequest>,
    ) -> Option<MipmapBatchId> {
        self.storage.stage_mipmap_requests(requests)
    }

    /// Publishes generation tokens only after the batch was encoded by a live
    /// node and the containing command buffer was submitted.
    pub(crate) fn commit_submitted_mipmap_batch(&mut self, batch: MipmapBatchId) -> bool {
        self.storage.commit_submitted_mipmap_batch(batch)
    }

    /// Moves exact-version Gaussian asset owners into destructible graph
    /// storage. Pass nodes retain only the returned Copy ID.
    #[cfg(feature = "3dgs")]
    pub(crate) fn stage_gaussian_frame_batch(
        &mut self,
        clouds: Vec<GaussianFrameCloud>,
    ) -> Option<GaussianFrameBatchId> {
        self.storage.stage_gaussian_frame_batch(clouds)
    }

    /// Releases the CPU asset Arc owners as soon as all live preparations have
    /// consumed them; execute uses only prepared GPU resources and bind groups.
    #[cfg(feature = "3dgs")]
    pub(crate) fn release_gaussian_frame_batch(&mut self, batch: GaussianFrameBatchId) {
        self.storage.release_gaussian_frame_batch(batch);
    }

    // ─── Logical Grouping (Inspector) ────────────────────────────────

    #[cfg(feature = "rdg_inspector")]
    #[inline]
    pub(crate) fn push_group(&mut self, name: &'static str) {
        self.storage.current_group_stack.push(name);
    }

    #[cfg(feature = "rdg_inspector")]
    #[inline]
    pub(crate) fn pop_group(&mut self) {
        self.storage.current_group_stack.pop();
    }

    fn register_resource(
        &mut self,
        name: &'static str,
        kind: ResourceKind,
        is_external: bool,
        version: u32,
    ) -> ErasedResourceNodeId {
        self.register_resource_with_root(name, kind, is_external, is_external, version)
    }

    fn register_resource_with_root(
        &mut self,
        name: &'static str,
        kind: ResourceKind,
        is_external: bool,
        is_output_root: bool,
        version: u32,
    ) -> ErasedResourceNodeId {
        self.assert_topology_mutable();
        let id = ErasedResourceNodeId::new(self.storage.resources.len() as u32, version);
        self.storage.resources.push(ResourceRecord {
            name,
            is_external,
            is_output_root,
            producer: None,
            consumer_edge_count: 0,
            first_use: usize::MAX,
            last_use: 0,
            physical_index: None,
            kind,
            alias_of: None,
        });
        id
    }

    /// Registers a named texture resource.
    pub fn register_texture(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
        is_external: bool,
    ) -> TextureNodeId {
        ResourceNodeId::from_erased(self.register_resource(
            name,
            ResourceKind::texture(desc),
            is_external,
            0,
        ))
    }

    pub fn register_buffer(
        &mut self,
        name: &'static str,
        desc: BufferDesc,
        is_external: bool,
    ) -> BufferNodeId {
        ResourceNodeId::from_erased(self.register_resource(
            name,
            ResourceKind::buffer(desc),
            is_external,
            0,
        ))
    }

    pub fn import_external_texture(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
        view: &Tracked<wgpu::TextureView>,
    ) -> TextureNodeId {
        ResourceNodeId::from_erased(self.register_resource(
            name,
            ResourceKind::external_texture(desc, std::ptr::from_ref(view)),
            true,
            0,
        ))
    }

    pub fn import_external_resource(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
        view: &Tracked<wgpu::TextureView>,
    ) -> TextureNodeId {
        self.import_external_texture(name, desc, view)
    }

    pub fn import_external_buffer(
        &mut self,
        name: &'static str,
        desc: BufferDesc,
        buffer: &Tracked<wgpu::Buffer>,
    ) -> BufferNodeId {
        ResourceNodeId::from_erased(self.register_resource(
            name,
            ResourceKind::external_buffer(desc, std::ptr::from_ref(buffer)),
            true,
            0,
        ))
    }

    fn import_scene_buffer(
        &mut self,
        name: &'static str,
        recipe: SceneBufferRecipe,
        stamp: myth_resources::buffer::CpuBufferStamp,
    ) -> BufferNodeId {
        self.assert_topology_mutable();
        let snapshot = recipe.snapshot(stamp);
        let desc = snapshot.desc();
        if let Some(id) = self.storage.scene_buffer_imports.find(snapshot) {
            assert_eq!(
                self.storage.scene_buffer_imports.snapshot(id),
                Some(snapshot),
                "scene buffer import key collided with a different snapshot"
            );
            let node = self
                .storage
                .scene_buffer_imports
                .node(id)
                .expect("current scene buffer import lost its node");
            assert_eq!(
                self.storage.resources[node.index() as usize].buffer_desc(),
                desc,
                "scene buffer imported with conflicting graph descriptors"
            );
            return node;
        }

        let import_id = self.storage.scene_buffer_imports.next_id();
        let node = ResourceNodeId::from_erased(self.register_resource_with_root(
            name,
            ResourceKind::scene_buffer(desc, import_id),
            true,
            false,
            0,
        ));
        self.storage
            .scene_buffer_imports
            .insert(import_id, node, snapshot, recipe);
        node
    }

    /// Capture a CPU-only logical import of the local-light metadata buffer.
    /// No upload or physical lookup occurs until the post-DPE barrier.
    pub(crate) fn import_scene_light_metadata_buffer(
        &mut self,
        name: &'static str,
        buffer: &myth_resources::buffer::CpuBuffer<myth_resources::uniforms::LightBufferMetadata>,
    ) -> BufferNodeId {
        self.import_scene_buffer(name, SceneBufferRecipe::LightMetadata, buffer.stamp())
    }

    /// Capture a CPU-only logical import of the local-light storage buffer.
    /// No payload clone, upload, or physical lookup occurs here.
    pub(crate) fn import_scene_light_storage_buffer(
        &mut self,
        name: &'static str,
        buffer: &myth_resources::buffer::CpuBuffer<Vec<myth_resources::uniforms::GpuLightStorage>>,
    ) -> BufferNodeId {
        self.import_scene_buffer(name, SceneBufferRecipe::LightStorage, buffer.stamp())
    }

    /// Import one exact CPU-only asset snapshot as a lazy external, non-root
    /// texture. A live read of this root is the materialization demand; the
    /// post-DPE graph batch fills its frame slot before any PassNode prepare
    /// step. Only derived SSA versions, such as `MipReady`, require a producer.
    ///
    /// Repeated imports of the same logical snapshot return the same graph
    /// root. No asset lookup, upload, or WGPU allocation happens here.
    pub(crate) fn import_asset_texture(
        &mut self,
        name: &'static str,
        snapshot: AssetTextureSnapshot,
    ) -> TextureNodeId {
        self.assert_topology_mutable();
        let desc = snapshot_texture_desc(snapshot);
        if let Some(id) = self.storage.asset_texture_imports.find(snapshot) {
            let entry = self.storage.asset_texture_imports.entry(id);
            assert_eq!(
                entry.snapshot, snapshot,
                "asset texture import key collided with a different snapshot"
            );
            assert_eq!(
                self.storage.resources[entry.node.index() as usize].texture_desc(),
                desc,
                "asset texture imported with conflicting graph descriptors"
            );
            return entry.node;
        }

        let import_id = self.storage.asset_texture_imports.next_id();
        let node = ResourceNodeId::from_erased(self.register_resource_with_root(
            name,
            ResourceKind::asset_texture(desc, import_id),
            true,
            false,
            0,
        ));
        self.storage
            .asset_texture_imports
            .insert(import_id, node, snapshot);
        node
    }

    /// Publish exact mip generations encoded by live logical asset producers.
    /// Call only after submitting the containing command buffer.
    pub(crate) fn commit_submitted_asset_texture_mipmaps(&mut self) -> usize {
        self.storage
            .asset_texture_imports
            .commit_submitted_mipmaps()
    }

    /// Import a logical persistent texture without requiring a physical view.
    ///
    /// Repeated imports of the same ID in one frame return the same graph root.
    /// Persistent writes are not DPE roots by default; they become live only
    /// through normal downstream demand or an explicit side effect.
    pub fn import_persistent_texture<T>(&mut self, name: &'static str, texture: T) -> TextureNodeId
    where
        T: Into<PersistentTextureRef>,
    {
        self.assert_topology_mutable();
        let texture = texture.into();
        if let Some(&existing) = self.storage.persistent_texture_lookup.get(&texture.id) {
            assert_eq!(
                existing.spec, texture.spec,
                "persistent texture {:?} imported with conflicting specifications",
                texture.id
            );
            return existing.node;
        }

        let id = ResourceNodeId::from_erased(self.register_resource_with_root(
            name,
            ResourceKind::persistent_texture(texture.spec.desc, texture.id),
            true,
            false,
            0,
        ));
        self.storage.persistent_texture_lookup.insert(
            texture.id,
            PersistentTextureImport {
                node: id,
                spec: texture.spec,
            },
        );
        id
    }

    /// Finalize ordinary-cache rollback eligibility after DPE and before any
    /// persistent texture can be materialized for this attempt.
    ///
    /// An entirely untouched replacement is the only case where the previous
    /// submitted publication is still known to match the physical backing. As
    /// soon as any member is touched by a live reader or writer, prepare may
    /// recreate or overwrite that backing. Rollback is therefore forfeited
    /// before entering prepare, even if the attempt later aborts before queue
    /// submission.
    pub(crate) fn finalize_persistent_cache_attempts_after_dpe(
        &mut self,
        registry: &mut HistoryRegistry,
    ) -> bool {
        assert!(
            self.storage.topology_compiled,
            "persistent texture materialization requires a compiled graph"
        );
        if let Some(any_cache_member_touched) = self.storage.persistent_cache_finalization {
            return any_cache_member_touched;
        }

        let imported_touches: SmallVec<[(PersistentTextureId, bool); 8]> = self
            .storage
            .persistent_texture_lookup
            .iter()
            .map(|(&id, import)| {
                (
                    id,
                    self.storage.resources[import.node.index() as usize].first_use != usize::MAX,
                )
            })
            .collect();

        let registered_requests: SmallVec<
            [(CacheCommitToken, SmallVec<[PersistentTextureId; 4]>); 4],
        > = self
            .storage
            .cache_commit_requests
            .iter()
            .map(|request| {
                let required_writes = request
                    .required_writes
                    .iter()
                    .map(|id| {
                        self.storage.resources[id.index() as usize]
                            .persistent_texture_id()
                            .expect("validated cache writer lost persistent backing")
                    })
                    .collect();
                (request.token, required_writes)
            })
            .collect();
        let any_cache_member_touched = registry.finalize_cache_attempts_after_dpe(
            &imported_touches,
            registered_requests
                .iter()
                .map(|(token, writes)| (*token, writes.as_slice())),
        );
        self.storage.persistent_cache_finalization = Some(any_cache_member_touched);
        any_cache_member_touched
    }

    /// Materialize only logical persistent imports touched by live passes.
    ///
    /// The per-frame import side table retains the complete requested view
    /// specification without inflating every [`ResourceRecord`]. Both the graph
    /// descriptor and registry's current logical specification are validated
    /// before physical creation. Ordinary-cache rollback is finalized first so
    /// an abort after any allocation can never revive stale metadata around a
    /// replacement physical texture.
    ///
    /// Call this after [`Self::compile`] (or [`Self::compile_topology`]) and
    /// before prepare/execute/submission. Compilation already freezes graph
    /// topology; repeated calls in the same attempt are idempotent.
    ///
    /// # Panics
    ///
    /// Panics before modifying graph or registry state when topology has not
    /// been compiled.
    pub fn materialize_live_persistent_textures(
        &mut self,
        registry: &mut HistoryRegistry,
        device: &Device,
    ) {
        self.finalize_persistent_cache_attempts_after_dpe(registry);

        for (&id, import) in &self.storage.persistent_texture_lookup {
            let resource = &self.storage.resources[import.node.index() as usize];
            if resource.first_use == usize::MAX {
                continue;
            }
            assert_eq!(
                resource.texture_desc(),
                import.spec.desc,
                "persistent texture {id:?} graph descriptor disagrees with its import spec"
            );
            registry.materialize_texture(device, resource.name, id, import.spec);
        }
    }

    /// Materialize only logical Asset Texture imports touched by the compiled
    /// execution queue, then publish their exact sampler/view bindings before
    /// central material preparation.
    ///
    /// The attempt-owned import slot records unavailable outcomes too, making
    /// repeated calls idempotent. Physical base-level uploads may safely warm
    /// the persistent cache on an aborted frame; mip completion remains owned
    /// by the graph and is published only after submission.
    pub(crate) fn materialize_live_asset_textures(
        &mut self,
        assets: &AssetServer,
        resource_manager: &mut ResourceManager,
    ) {
        assert!(
            self.storage.topology_compiled,
            "Asset Texture materialization requires a compiled graph"
        );
        let resolver = self.storage.asset_texture_imports.resolver();
        for index in 0..self.storage.asset_texture_imports.len() {
            let id = AssetTextureImportId::from_index(index);
            if resolver.materialization(id).is_some() {
                continue;
            }
            let entry = self.storage.asset_texture_imports.entry(id);
            let live = self.storage.resources[entry.node.index() as usize].first_use != usize::MAX;
            if !live {
                continue;
            }
            let materialization =
                resource_manager.materialize_asset_texture_for_graph(assets, entry.snapshot);
            resolver.materialize(id, materialization);
            assert!(
                resolver.materialization(id).is_some(),
                "live Asset Texture import must record a materialization outcome"
            );
        }
    }

    /// Materialize only logical scene buffers touched by DPE-live passes.
    ///
    /// The explicit typed sources are borrowed only for this barrier; neither
    /// GraphStorage nor PassNodes retain them. A typed failure requires the
    /// caller to terminate the attempt before PassNode prepare begins.
    pub(crate) fn materialize_live_scene_buffers(
        &mut self,
        sources: SceneBufferSources<'_>,
        uploads: &mut BufferUploadContext<'_>,
    ) -> Result<(), SceneBufferMaterializationFailure> {
        assert!(
            self.storage.topology_compiled,
            "Scene Buffer materialization requires a compiled graph"
        );
        self.storage.scene_buffer_imports.materialize_live(
            &self.storage.resources,
            sources,
            uploads,
        )
    }

    /// Materialize the fixed Group 0 buffer batch only when a DPE-live pass
    /// declared the captured plan. Final bind-group creation is deliberately a
    /// separate ResourceManager step performed by the Composer.
    pub(crate) fn materialize_live_global_state(
        &mut self,
        sources: GlobalStateSources<'_>,
        uploads: &mut BufferUploadContext<'_>,
    ) -> Result<(), GlobalStateMaterializationFailure> {
        assert!(
            self.storage.topology_compiled,
            "Global State materialization requires a compiled graph"
        );
        self.storage
            .global_state_attempt
            .materialize_live(sources, uploads)
    }

    /// Key authorized for ResourceManager Group 0 finalization after exact
    /// DPE-live buffer materialization.
    #[inline]
    #[must_use]
    pub(crate) fn global_state_allowed_key(&self) -> Option<GlobalStateKey> {
        self.storage.global_state_attempt.allowed_key()
    }

    /// Physical IDs in Group 0 buffer binding order: camera, environment,
    /// directional lights.
    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) fn materialized_global_buffer_ids(&self) -> Option<[u64; 3]> {
        self.storage.global_state_attempt.prepared.resource_ids()
    }

    /// Owned exact BufferRefs in Group 0 binding order: camera, environment,
    /// directional lights. They remain hidden until the whole captured batch
    /// has been successfully materialized.
    #[inline]
    #[must_use]
    pub(crate) fn materialized_global_buffer_refs(
        &self,
    ) -> Option<[myth_resources::buffer::BufferRef; 3]> {
        self.storage.global_state_attempt.materialized_buffer_refs()
    }

    /// Retain the finalized persistent Group 0 bind group for this attempt.
    pub(crate) fn stage_finalized_global_bind_group(
        &mut self,
        key: GlobalStateKey,
        bind_group: &wgpu::BindGroup,
    ) {
        let allowed_key = self.global_state_allowed_key();
        self.storage
            .global_state_attempt
            .prepared
            .stage_finalized(allowed_key, key, bind_group);
    }

    #[inline]
    #[must_use]
    pub(crate) fn finalized_global_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.storage.global_state_attempt.prepared.bind_group()
    }

    /// Exact resolver whitelist key, visible only after its BindGroup has been
    /// staged into the graph-attempt owner.
    #[inline]
    #[must_use]
    pub(crate) fn finalized_global_state_key(&self) -> Option<GlobalStateKey> {
        self.storage.global_state_attempt.prepared.finalized_key()
    }

    /// Register an atomic history publication. The bank is committed only if
    /// every required writer survives DPE and the command buffer is submitted.
    pub fn register_history_commit(
        &mut self,
        token: HistoryCommitToken,
        required_writes: &[TextureNodeId],
    ) {
        self.assert_topology_mutable();
        assert!(
            !required_writes.is_empty(),
            "history commit requires writers"
        );
        for &id in required_writes {
            assert!(
                self.storage.resources[id.index() as usize]
                    .persistent_texture_id()
                    .is_some(),
                "history commit output must be a persistent texture"
            );
        }
        self.storage
            .history_commit_requests
            .push(HistoryCommitRequest {
                token,
                required_writes: required_writes.iter().map(|id| id.erase()).collect(),
            });
    }

    /// Register an atomic ordinary-cache publication.
    ///
    /// Registration adds no dependency and never seeds DPE liveness. The cache
    /// is published only by the post-submit step when every required resource
    /// has a producer and every such producer survived DPE.
    pub fn register_cache_commit(
        &mut self,
        token: CacheCommitToken,
        required_writes: &[TextureNodeId],
    ) {
        self.assert_topology_mutable();
        assert!(!required_writes.is_empty(), "cache commit requires writers");
        assert!(
            required_writes
                .iter()
                .enumerate()
                .all(|(index, id)| !required_writes[..index].contains(id)),
            "cache commit writer set contains duplicates"
        );
        for &id in required_writes {
            assert!(
                self.storage.resources[id.index() as usize]
                    .persistent_texture_id()
                    .is_some(),
                "cache commit output must be a persistent texture"
            );
        }
        let required_writes: SmallVec<[ErasedResourceNodeId; 4]> =
            required_writes.iter().map(|id| id.erase()).collect();
        if let Some(existing) = self
            .storage
            .cache_commit_requests
            .iter()
            .find(|request| request.token.cache_id == token.cache_id)
        {
            assert_eq!(
                existing.token, token,
                "cache ID registered with conflicting commit tokens"
            );
            let same_set = existing.required_writes.len() == required_writes.len()
                && existing
                    .required_writes
                    .iter()
                    .all(|id| required_writes.contains(id));
            assert!(
                same_set,
                "cache commit token registered with conflicting writer sets"
            );
            return;
        }
        self.storage.cache_commit_requests.push(CacheCommitRequest {
            token,
            required_writes,
        });
    }

    /// Creates a versioned alias of `input_id` that shares the same physical
    /// GPU memory.
    pub fn create_alias<T: GraphResourceType>(
        &mut self,
        input_id: ResourceNodeId<T>,
        name: &'static str,
    ) -> ResourceNodeId<T> {
        let root_idx = self.resolve_alias_root(input_id.index() as usize);
        let root_id = ErasedResourceNodeId::new(root_idx as u32, 0);

        let root_res = &self.storage.resources[root_idx];
        assert!(
            root_res.persistent_texture_id().is_none(),
            "persistent textures cannot be versioned or mutated in place"
        );
        assert!(
            root_res.scene_buffer_import_id().is_none(),
            "scene buffer imports cannot be versioned or mutated in place"
        );
        let kind = root_res.kind.without_external_binding();
        let is_external = root_res.is_external;
        let is_output_root = root_res.is_output_root;

        let new_id = self.register_resource_with_root(
            name,
            kind,
            is_external,
            is_output_root,
            input_id.version().saturating_add(1),
        );
        self.storage.resources[new_id.index() as usize].alias_of = Some(root_id);
        ResourceNodeId::from_erased(new_id)
    }

    // pub fn create_alias(&mut self, input_id: TextureNodeId, name: &'static str) -> TextureNodeId {
    //     self.create_alias_typed(input_id, name)
    // }

    /// Chases the `alias_of` chain to find the root (non-alias) resource.
    #[inline]
    fn resolve_alias_root(&self, idx: usize) -> usize {
        if let Some(root_id) = self.storage.resources[idx].alias_of {
            root_id.index() as usize
        } else {
            idx
        }
    }

    /// Adds a pass to the graph using eager setup.
    ///
    /// The closure receives a [`PassBuilder`] and must return a tuple of
    /// `(PassNode, Out)`.  All resource creation and dependency wiring
    /// happens immediately inside the closure; the returned `Out` (usually
    /// a [`TextureNodeId`] or typed output struct) is forwarded to the
    /// caller for downstream wiring.
    ///
    /// Setup closures are non-reentrant: they must not recursively call
    /// `add_pass`. One-shot demand finalization runs after each closure, and a
    /// nested setup would make both staging ownership and consumer-edge order
    /// ambiguous. Such an internal call fails before mutating graph state.
    ///
    /// # Arena Allocation
    ///
    /// The returned `PassNode` is allocated on the current frame's
    /// [`FrameArena`] — a simple $O(1)$ pointer bump with no system
    /// allocator overhead.  All nodes reside in contiguous memory,
    /// maximising CPU cache utilisation during the execute phase.
    ///
    /// # Zero Drop
    ///
    /// Arena-allocated nodes are **never** dropped.  [`FrameArena::reset()`]
    /// reclaims all memory in O(1) without running destructors.  Nodes
    /// must therefore hold only borrowed references (`&'a T`) or `Copy`
    /// types — never owned heap data.
    pub fn add_pass<N, Out, F>(&mut self, name: &'static str, setup_fn: F) -> Out
    where
        N: PassNode<'a> + 'a,
        F: FnOnce(&mut PassBuilder<'_, 'a>) -> (N, Out),
    {
        self.assert_topology_mutable();
        // Static assertion to enforce that N does not implement Drop and does not contain heap data.
        // This ensures that PassNodes are safe to allocate on the FrameArena without needing to run destructors.
        let () = <N as AssertNoDrop>::VALID;

        assert!(
            !self.storage.pass_setup_active,
            "nested RenderGraph::add_pass is unsupported; pass setup must complete before another pass is added"
        );
        self.storage.pass_setup_active = true;
        let pass_index = self.storage.passes.len();
        let logical_read_start = self.storage.logical_reads.len();
        self.storage.draw_plan_raw_demands.clear();
        self.storage.scene_submission_raw_demands.clear();

        // Phase 1: placeholder record (node = None) so the PassBuilder can
        // reference the correct index.
        self.storage
            .passes
            .push(PassRecord::new_empty(name, logical_read_start));

        #[cfg(feature = "rdg_inspector")]
        {
            self.storage.passes[pass_index].groups =
                self.storage.current_group_stack.iter().copied().collect();
        }

        // Phase 2: execute the closure — all topology wiring happens here.
        let (node, output) = {
            let mut builder = PassBuilder {
                graph: self,
                pass_index,
            };
            setup_fn(&mut builder)
        };
        debug_assert_eq!(
            self.storage.passes[pass_index].logical_read_range.end,
            self.storage.logical_reads.len()
        );
        self.storage.pass_setup_active = false;

        // Phase 3: one-shot plan-demand canonicalization. The setup hot path
        // records raw IDs only; mixed logical reads and consumer edges are
        // attached once, in deterministic first-occurrence order, after the
        // closure no longer borrows the graph. The legacy DrawPlan bridge runs
        // first so already-migrated submission dependencies deduplicate against
        // every compatibility edge too.
        self.storage.finalize_pass_draw_plan_demands(pass_index);
        self.storage
            .finalize_pass_scene_submission_demands(pass_index);

        // Phase 4: allocate the node on the frame arena (O(1) pointer bump).
        // let node_ref: &mut N = self.arena.alloc(node);
        let node_ref: &'a mut N = self.arena.alloc(node);
        let ptr = node_ref as *mut N;
        self.storage.passes[pass_index].node = Some(NodeSlot::new(ptr));
        self.record_declared_pass();
        output
    }

    pub fn compile_topology(&mut self) {
        self.assert_topology_mutable();
        self.enter_diagnostics_phase(RdgFramePhase::CompileGraph);
        self.storage.topology_compiled = true;
        self.fold_simple_passes();
        self.build_physical_dependencies();
        self.cull_dead_passes();
        self.validate_live_asset_texture_dependencies();
        self.topological_sort();
        self.collect_live_global_state_demand();
        self.collect_live_draw_plan_demands();
        self.collect_live_scene_submission_demands();

        self.compute_resource_lifetimes();
        let live_passes = self.storage.execution_queue.len();
        self.record_compiled_live_passes(live_passes);
    }

    pub fn compile(&mut self, pool: &mut TransientPool, device: &Device) {
        self.compile_topology();

        self.enter_diagnostics_phase(RdgFramePhase::MaterializeResources);
        self.allocate_physical_resources(pool, device);
    }

    /// Publish history banks after the frame's command buffer was submitted.
    /// Only the renderer's submission path may authorize this publication.
    pub(crate) fn commit_submitted_histories(&self, registry: &mut HistoryRegistry) {
        for request in &self.storage.history_commit_requests {
            let live_writers = request
                .required_writes
                .iter()
                .filter(|id| {
                    self.storage.resources[id.index() as usize]
                        .producer
                        .is_some_and(|producer| self.storage.passes[producer].reference_count > 0)
                })
                .count();

            if live_writers == request.required_writes.len() {
                let written_textures: SmallVec<[PersistentTextureId; 4]> = request
                    .required_writes
                    .iter()
                    .map(|id| {
                        self.storage.resources[id.index() as usize]
                            .persistent_texture_id()
                            .expect("validated history writer lost persistent backing")
                    })
                    .collect();
                let committed = registry.commit(request.token, &written_textures);
                debug_assert!(committed, "stale history commit token");
            } else {
                debug_assert_eq!(
                    live_writers, 0,
                    "history set was only partially written by live passes"
                );
            }
        }
    }

    /// Publish every eligible persistent resource after command submission.
    ///
    /// History behavior is intentionally delegated to the existing API. Cache
    /// requests return typed receipts instead of storing callbacks or feature
    /// pointers in graph storage; callers may use those receipts to update
    /// feature-local bookkeeping after submission.
    pub(crate) fn commit_submitted_persistent_resources(
        &mut self,
        registry: &mut HistoryRegistry,
    ) -> SmallVec<[CachePublicationReceipt; 4]> {
        if self.storage.persistent_cache_finalization.is_none() {
            let any_cache_member_touched =
                self.finalize_persistent_cache_attempts_after_dpe(registry);
            assert!(
                !any_cache_member_touched,
                "live persistent cache members must be finalized before post-submit publication"
            );
        }

        self.commit_submitted_histories(registry);

        let mut receipts = SmallVec::new();
        for request in &self.storage.cache_commit_requests {
            let live_writers = request
                .required_writes
                .iter()
                .filter(|id| {
                    self.storage.resources[id.index() as usize]
                        .producer
                        .is_some_and(|producer| self.storage.passes[producer].reference_count > 0)
                })
                .count();
            if live_writers != request.required_writes.len() {
                // DPE-time finalization already resolved rollback before any
                // persistent materialization. Submission only publishes a
                // complete live writer group; every other shape stays invalid.
                continue;
            }

            let written_textures: SmallVec<[PersistentTextureId; 4]> = request
                .required_writes
                .iter()
                .map(|id| {
                    self.storage.resources[id.index() as usize]
                        .persistent_texture_id()
                        .expect("validated cache writer lost persistent backing")
                })
                .collect();
            let receipt = registry
                .commit_cache(request.token, &written_textures)
                .expect("live persistent cache commit token or writer group is invalid");
            receipts.push(receipt);
        }
        receipts
    }

    /// Compile-time **edge contraction** for pure-forwarding (blit / present)
    /// passes.
    ///
    /// A pass flagged via [`PassBuilder::mark_pure_forwarding`] merely copies a
    /// single source resource into a single destination of identical format and
    /// size — the canonical example being the unified
    /// [`PresentFeature`](crate::graph::passes::PresentFeature), which forwards
    /// the final post-process image onto the swap-chain surface.  When the copy
    /// is provably redundant we eliminate it so the upstream producer renders
    /// straight into the surface, saving a full-screen blit every frame.
    ///
    /// # Mechanism
    ///
    /// PassNodes capture their render target by resource id at build time and
    /// are type-erased, so the producer's node cannot be rewritten directly.
    /// The fold therefore leans on the graph's existing aliasing machinery:
    ///
    /// 1. **Redirect** the producer's declared writes from the source onto the
    ///    forwarding pass's destination, so the dependency graph and dead-pass
    ///    cull see the producer as the surface's final writer and anchor it.
    /// 2. **Alias** the source onto the destination's physical memory
    ///    (`alias_of` + inherited `is_external`).  The producer's node still
    ///    references the source id, but it now resolves to the surface view at
    ///    execute time — and because the alias inherits the surface's external
    ///    status, the auto-deduced `StoreOp` keeps the result instead of
    ///    discarding it as a transient.
    /// 3. **Strip** the forwarding pass into an island for
    ///    [`cull_dead_passes`](Self::cull_dead_passes) to remove.
    ///
    /// # Safety conditions
    ///
    /// Folding is applied only when it cannot change observable behaviour:
    ///
    /// 1. **Layout match** — identical class and (for textures) format/size, so
    ///    sharing physical memory is valid; a mismatch keeps the real blit.
    /// 2. **Unique consumer** — the source feeds only this forwarding pass; if
    ///    anything else samples it, the intermediate must survive.
    /// 3. **Known producer** — the source has an upstream producer to redirect.
    ///
    /// Runs as the first compilation stage so the rewired relationships feed
    /// into the subsequently built physical dependency graph.
    fn fold_simple_passes(&mut self) {
        for i in 0..self.storage.passes.len() {
            // ── Gate: only fold flagged single-in / single-out passes ──
            let pass = &self.storage.passes[i];
            if !pass.is_pure_forwarding
                || self.storage.pass_reads(i).is_empty()
                || pass.writes.is_empty()
            {
                continue;
            }

            let src_idx = self.storage.pass_reads(i)[0].index() as usize;
            let dst_id = pass.writes[0];
            let dst_idx = dst_id.index() as usize;
            let src_root_idx = self.resolve_alias_root(src_idx);
            let dst_root_idx = self.resolve_alias_root(dst_idx);

            // Persistent and asset resources have logical identity and
            // materialization/publication semantics. Contracting either
            // endpoint would reroute the physical write without updating its
            // import side table.
            if self.storage.resources[src_root_idx]
                .persistent_texture_id()
                .is_some()
                || self.storage.resources[dst_root_idx]
                    .persistent_texture_id()
                    .is_some()
                || self.storage.resources[src_root_idx]
                    .asset_texture_import_id()
                    .is_some()
                || self.storage.resources[dst_root_idx]
                    .asset_texture_import_id()
                    .is_some()
            {
                continue;
            }

            // ── Condition 1: identical physical layout ──
            let src_res = &self.storage.resources[src_idx];
            let dst_res = &self.storage.resources[dst_idx];
            if src_res.class() != dst_res.class() {
                continue;
            }
            let layout_match = match src_res.class() {
                ResourceClass::Texture => {
                    let s = src_res.texture_desc();
                    let d = dst_res.texture_desc();
                    s.format == d.format && s.size == d.size
                }
                ResourceClass::Buffer => {
                    src_res.buffer_desc().logical_size == dst_res.buffer_desc().logical_size
                }
            };
            if !layout_match {
                continue;
            }

            // ── Condition 2: the source feeds only this forwarding pass ──
            if src_res.consumer_edge_count() != 1 {
                continue;
            }

            // ── Condition 3: a known producer we can redirect ──
            let Some(producer_idx) = src_res.producer else {
                continue;
            };

            // ── Edge contraction ──
            // Resolve the destination's physical root (the surface) that the
            // producer must ultimately render into.
            let dst_root_id = ErasedResourceNodeId::new(dst_root_idx as u32, 0);
            let dst_root_external = self.storage.resources[dst_root_idx].is_external;
            let dst_root_output = self.storage.resources[dst_root_idx].is_output_root;

            // (1) Redirect the producer's declared outputs from `src` onto `dst`
            //     so the dead-pass cull anchors the producer as the final writer.
            let producer = &mut self.storage.passes[producer_idx];
            for id in producer
                .writes
                .iter_mut()
                .chain(producer.creates.iter_mut())
            {
                if id.index() as usize == src_idx {
                    *id = dst_id;
                }
            }

            // (2) Alias `src` onto the destination's physical memory.  The
            //     producer's (type-erased) PassNode still targets `src` at
            //     execute time; aliasing reroutes that target to the surface and
            //     inherits its external status so the write is stored, not
            //     discarded.
            let src_res = &mut self.storage.resources[src_idx];
            src_res.alias_of = Some(dst_root_id);
            src_res.is_external = dst_root_external;
            src_res.is_output_root = dst_root_output;

            // (3) Transfer destination ownership to the producer.
            self.storage.resources[dst_idx].producer = Some(producer_idx);

            // (4) Strip the forwarding pass into an island for the dead-pass
            //     cull to remove.
            self.storage.clear_pass_reads(i);
            let folded = &mut self.storage.passes[i];
            folded.writes.clear();
            folded.creates.clear();
            folded.has_side_effect = false;
            self.record_folded_pass();
        }
    }

    fn build_physical_dependencies(&mut self) {
        self.storage.physical_dependencies.clear();
        let pass_count = self.storage.passes.len();
        for pass_idx in 0..pass_count {
            let dependency_start = self.storage.physical_dependencies.len();
            self.storage.passes[pass_idx].physical_dependency_range =
                dependency_start..dependency_start;
            self.storage.physical_dependency_scratch.begin(pass_count);

            let read_count = self.storage.pass_reads(pass_idx).len();
            for read_index in 0..read_count {
                let read = self.storage.pass_reads(pass_idx)[read_index];
                let Some(producer_index) = self.storage.resources[read.index() as usize].producer
                else {
                    continue;
                };
                if producer_index >= pass_idx
                    || !self
                        .storage
                        .physical_dependency_scratch
                        .insert(producer_index)
                {
                    continue;
                }
                self.storage.physical_dependencies.push(producer_index);
                self.storage.passes[pass_idx].physical_dependency_range.end =
                    self.storage.physical_dependencies.len();
            }
        }
    }

    /// A logical asset root is a lazy external input materialized after DPE.
    /// Derived SSA versions still require an earlier live producer. Catch a
    /// missing or late derived producer during topology compilation instead of
    /// allowing a resolver panic deep in prepare/execute.
    fn validate_live_asset_texture_dependencies(&self) {
        for pass_idx in 0..self.storage.passes.len() {
            let pass = &self.storage.passes[pass_idx];
            if pass.reference_count == 0 {
                continue;
            }
            for write in &pass.writes {
                let write_idx = write.index() as usize;
                let root_idx = self.resolve_alias_root(write_idx);
                if self.storage.resources[root_idx]
                    .asset_texture_import_id()
                    .is_none()
                {
                    continue;
                }
                assert_ne!(
                    write_idx, root_idx,
                    "live pass '{}' cannot write a read-only Asset Texture import root",
                    pass.name
                );
                assert!(
                    self.storage
                        .pass_reads(pass_idx)
                        .iter()
                        .any(|read| { self.resolve_alias_root(read.index() as usize) == root_idx }),
                    "derived asset texture producer '{}' must read the lazy import chain it mutates",
                    pass.name
                );
            }
            for read in self.storage.pass_reads(pass_idx) {
                let read_idx = read.index() as usize;
                let root_idx = self.resolve_alias_root(read_idx);
                if self.storage.resources[root_idx]
                    .asset_texture_import_id()
                    .is_none()
                {
                    continue;
                }
                if read_idx == root_idx {
                    continue;
                }
                let producer = self.storage.resources[read_idx]
                    .producer
                    .unwrap_or_else(|| {
                        panic!(
                            "live pass '{}' reads derived asset texture '{}' without a producer",
                            pass.name, self.storage.resources[read_idx].name
                        )
                    });
                assert!(
                    producer < pass_idx && self.storage.passes[producer].reference_count > 0,
                    "derived asset texture producer '{}' must be registered before live consumer '{}'",
                    self.storage.passes[producer].name,
                    pass.name
                );
                assert!(
                    self.storage
                        .pass_reads(producer)
                        .iter()
                        .any(|read| { self.resolve_alias_root(read.index() as usize) == root_idx }),
                    "derived asset texture producer '{}' must read the lazy import chain it mutates",
                    self.storage.passes[producer].name
                );
            }
        }
    }

    fn cull_dead_passes(&mut self) {
        self.storage.compile_stack.clear();

        let res_count = self.storage.resources.len();

        self.storage.compile_max_ext_indices.clear();
        self.storage.compile_max_ext_indices.resize(res_count, 0);

        for i in 0..res_count {
            if self.storage.resources[i].is_external && self.storage.resources[i].is_output_root {
                let root_idx = self.resolve_alias_root(i);
                self.storage.compile_max_ext_indices[root_idx] = i;
            }
        }

        for (i, pass) in self.storage.passes.iter_mut().enumerate() {
            pass.reference_count = 0;
            if pass.has_side_effect {
                self.storage.compile_stack.push(i);
                pass.reference_count += 1;
                continue;
            }
            for write_id in &pass.writes {
                let write_idx = write_id.index() as usize;
                let res = &self.storage.resources[write_idx];

                if res.is_external && res.is_output_root {
                    let root_idx = res.alias_of.map_or(write_idx, |id| id.index() as usize);

                    if write_idx == self.storage.compile_max_ext_indices[root_idx] {
                        self.storage.compile_stack.push(i);
                        pass.reference_count += 1;
                        break;
                    }
                }
            }
        }

        while let Some(pass_idx) = self.storage.compile_stack.pop() {
            let num_deps = self.storage.pass_physical_dependencies(pass_idx).len();
            for dep_i in 0..num_deps {
                let dep_idx = self.storage.pass_physical_dependencies(pass_idx)[dep_i];
                if self.storage.passes[dep_idx].reference_count == 0 {
                    self.storage.compile_stack.push(dep_idx);
                }
                self.storage.passes[dep_idx].reference_count += 1;
            }
        }
    }

    /// Project DPE liveness onto frame-local draw-plan reservations.
    ///
    /// Plan demands never participate in pass rooting or dependency
    /// construction. Only records already selected into the compiled execution
    /// queue can make a plan live, and repeated demands are idempotent.
    fn collect_live_draw_plan_demands(&mut self) {
        self.storage.draw_plans.reset_live_demands();

        for queue_index in 0..self.storage.execution_queue.len() {
            let pass_index = self.storage.execution_queue[queue_index];
            let demand_range = self.storage.passes[pass_index]
                .draw_plan_demand_range
                .clone();
            for demand_index in demand_range {
                let id = self.storage.draw_plan_demands[demand_index];
                assert!(
                    self.storage.draw_plans.validate_reserved(id),
                    "live pass '{}' retained a draw plan from another graph attempt",
                    self.storage.passes[pass_index].name
                );
                self.storage.draw_plans.mark_live(id);
                debug_assert!(self.storage.draw_plans.is_live(id));
            }
        }
    }

    /// Project backend-neutral submission liveness from the compiled queue,
    /// then synchronize the transitional CPU-direct adapter with its private
    /// legacy DrawPlan records.
    fn collect_live_scene_submission_demands(&mut self) {
        if self.storage.scene_submissions.plan_count() == 0 {
            return;
        }
        self.storage.scene_submissions.reset_live();

        for queue_index in 0..self.storage.execution_queue.len() {
            let pass_index = self.storage.execution_queue[queue_index];
            let demand_range = self.storage.passes[pass_index]
                .scene_submission_demand_range
                .clone();
            for demand_index in demand_range {
                let id = self.storage.scene_submission_demands[demand_index];
                assert!(
                    self.storage.scene_submissions.validate_reserved(id),
                    "live pass '{}' retained a stale or foreign scene submission",
                    self.storage.passes[pass_index].name
                );
                self.storage.scene_submissions.mark_live(id);
                debug_assert!(self.storage.scene_submissions.is_live(id));
            }
        }

        let scene_submissions = &self.storage.scene_submissions;
        let draw_plans = &mut self.storage.draw_plans;
        for direct in scene_submissions.live_cpu_direct_plans() {
            draw_plans.mark_live(direct.draw_plan);
            debug_assert!(draw_plans.is_live(direct.draw_plan));
        }
    }

    /// Project the unique Group 0 demand from passes already selected by DPE.
    /// Demand metadata does not participate in rooting or dependency creation.
    fn collect_live_global_state_demand(&mut self) {
        let mut live_key = None;
        for &pass_index in &self.storage.execution_queue {
            let Some(key) = self.storage.passes[pass_index].global_state_demand else {
                continue;
            };
            match live_key {
                None => live_key = Some(key),
                Some(current) if current == key => {}
                Some(_) => panic!(
                    "DPE-live passes demanded different global state keys (latest pass '{}')",
                    self.storage.passes[pass_index].name
                ),
            }
        }
        self.storage
            .global_state_attempt
            .project_live_key(live_key, self.renderer_diagnostics.is_some());
    }

    fn topological_sort(&mut self) {
        let pass_count = self.storage.passes.len();

        self.storage.execution_queue.clear();
        self.storage.compile_in_degrees.clear();
        self.storage.compile_in_degrees.resize(pass_count, 0);
        self.storage.compile_ready_heap.clear();
        self.storage.compile_reverse_offsets.clear();
        self.storage
            .compile_reverse_offsets
            .resize(pass_count + 1, 0);
        self.storage.compile_reverse_edges.clear();
        self.storage.compile_reverse_cursors.clear();
        self.storage.compile_reverse_cursors.resize(pass_count, 0);

        // First CSR pass: count only edges whose consumer survived DPE.
        for consumer_index in 0..pass_count {
            if self.storage.passes[consumer_index].reference_count == 0 {
                continue;
            }
            let dependency_count = self
                .storage
                .pass_physical_dependencies(consumer_index)
                .len();
            self.storage.compile_in_degrees[consumer_index] = dependency_count;
            if dependency_count == 0 {
                self.storage.compile_ready_heap.push(ReadyNode {
                    priority: 0,
                    pass_idx: consumer_index,
                });
            }
            for dependency_index in 0..dependency_count {
                let producer_index =
                    self.storage.pass_physical_dependencies(consumer_index)[dependency_index];
                debug_assert!(self.storage.passes[producer_index].reference_count > 0);
                self.storage.compile_reverse_offsets[producer_index + 1] += 1;
            }
        }

        for producer_index in 0..pass_count {
            self.storage.compile_reverse_offsets[producer_index + 1] +=
                self.storage.compile_reverse_offsets[producer_index];
        }
        let reverse_edge_count = self.storage.compile_reverse_offsets[pass_count];
        self.storage
            .compile_reverse_edges
            .resize(reverse_edge_count, usize::MAX);
        self.storage
            .compile_reverse_cursors
            .copy_from_slice(&self.storage.compile_reverse_offsets[..pass_count]);

        // Second CSR pass: consumers are visited in registration order, so
        // every producer's adjacency slice preserves that deterministic order.
        for consumer_index in 0..pass_count {
            if self.storage.passes[consumer_index].reference_count == 0 {
                continue;
            }
            let dependency_count = self
                .storage
                .pass_physical_dependencies(consumer_index)
                .len();
            for dependency_index in 0..dependency_count {
                let producer_index =
                    self.storage.pass_physical_dependencies(consumer_index)[dependency_index];
                let cursor = self.storage.compile_reverse_cursors[producer_index];
                self.storage.compile_reverse_edges[cursor] = consumer_index;
                self.storage.compile_reverse_cursors[producer_index] += 1;
            }
        }
        debug_assert!((0..pass_count).all(|producer_index| {
            self.storage.compile_reverse_cursors[producer_index]
                == self.storage.compile_reverse_offsets[producer_index + 1]
        }));

        let mut sequence_counter = 0;

        while let Some(ReadyNode { pass_idx: node, .. }) = self.storage.compile_ready_heap.pop() {
            self.storage.execution_queue.push(node);
            sequence_counter += 1;

            let edge_start = self.storage.compile_reverse_offsets[node];
            let edge_end = self.storage.compile_reverse_offsets[node + 1];
            for edge_index in edge_start..edge_end {
                let downstream = self.storage.compile_reverse_edges[edge_index];
                self.storage.compile_in_degrees[downstream] -= 1;
                if self.storage.compile_in_degrees[downstream] == 0 {
                    let heuristic_score = sequence_counter;

                    self.storage.compile_ready_heap.push(ReadyNode {
                        priority: heuristic_score,
                        pass_idx: downstream,
                    });
                }
            }
        }

        let alive_count = self
            .storage
            .passes
            .iter()
            .filter(|p| p.reference_count > 0)
            .count();
        assert_eq!(
            self.storage.execution_queue.len(),
            alive_count,
            "Render Graph Detected Circular Dependency!"
        );

        #[cfg(debug_assertions)]
        self.debug_print_topology_changes();
    }

    fn compute_resource_lifetimes(&mut self) {
        for res in &mut self.storage.resources {
            res.first_use = usize::MAX;
            res.last_use = 0;
        }

        for (timeline_index, &pass_idx) in self.storage.execution_queue.iter().enumerate() {
            let read_count = self.storage.pass_reads(pass_idx).len();
            for read_index in 0..read_count {
                let id = self.storage.pass_reads(pass_idx)[read_index];
                let res = &mut self.storage.resources[id.index() as usize];
                res.first_use = res.first_use.min(timeline_index);
                res.last_use = res.last_use.max(timeline_index);
            }

            let pass = &self.storage.passes[pass_idx];
            for &id in &pass.writes {
                let res = &mut self.storage.resources[id.index() as usize];
                res.first_use = res.first_use.min(timeline_index);
                res.last_use = res.last_use.max(timeline_index);
            }
            for &id in &pass.creates {
                let res = &mut self.storage.resources[id.index() as usize];
                res.first_use = res.first_use.min(timeline_index);
                res.last_use = res.last_use.max(timeline_index);
            }
        }
    }

    /// Allocates physical GPU textures for transient resources, with
    /// alias-aware sharing.
    ///
    /// **Phase 1** — Calculate the unified lifetime for each alias group by propagating the
    /// lifetimes from aliases to their root resources.This ensures that the root resource's lifetime encompasses all of its aliases, preventing premature deallocation of shared physical resources.
    ///
    /// **Phase 2** — Propagate the root's unified lifetime back to every alias.\
    /// This ensures that the execute-phase `StoreOp` deduction sees the correct final lifetime and never discards data prematurely for intermediate versions.
    ///
    /// **Phase 3** — Request physical allocations from the pool for root resources.
    /// Aliases will be skipped since they share the same physical memory.
    ///
    /// **Phase 4** — Propagate the root's unified `last_use` back to every alias.\
    /// This ensures that the execute-phase `LoadOp` deduction sees the correct lifetime and keeps the physical memory alive for the entire duration of all aliases.    
    fn allocate_physical_resources(&mut self, pool: &mut TransientPool, device: &Device) {
        pool.begin_frame();

        for i in 0..self.storage.resources.len() {
            if self.storage.resources[i].alias_of.is_some() {
                let root_idx = self.resolve_alias_root(i);
                let alias_first = self.storage.resources[i].first_use;
                let alias_last = self.storage.resources[i].last_use;
                self.storage.resources[root_idx].first_use =
                    self.storage.resources[root_idx].first_use.min(alias_first);
                self.storage.resources[root_idx].last_use =
                    self.storage.resources[root_idx].last_use.max(alias_last);
            }
        }

        for i in 0..self.storage.resources.len() {
            if self.storage.resources[i].alias_of.is_some() {
                let root_idx = self.resolve_alias_root(i);
                self.storage.resources[i].first_use = self.storage.resources[root_idx].first_use;
                self.storage.resources[i].last_use = self.storage.resources[root_idx].last_use;
            }
        }

        for i in 0..self.storage.resources.len() {
            let res = &mut self.storage.resources[i];
            if res.is_external || res.first_use == usize::MAX || res.alias_of.is_some() {
                continue;
            }
            res.physical_index = Some(match res.kind {
                ResourceKind::Texture { desc, .. } => {
                    pool.acquire(device, &desc, res.first_use, res.last_use)
                }
                ResourceKind::Buffer { desc, .. } => {
                    pool.acquire_buffer(device, &desc, res.first_use, res.last_use)
                }
            });
        }

        for i in 0..self.storage.resources.len() {
            if self.storage.resources[i].alias_of.is_some() {
                let root_idx = self.resolve_alias_root(i);
                self.storage.resources[i].physical_index =
                    self.storage.resources[root_idx].physical_index;
            }
        }
    }

    #[cfg(debug_assertions)]
    fn debug_print_topology_changes(&mut self) {
        let current_names: Vec<&'static str> = self
            .storage
            .execution_queue
            .iter()
            .map(|&idx| self.storage.passes[idx].name)
            .collect();

        if current_names != self.storage.prev_execution_names {
            if log::log_enabled!(log::Level::Info) {
                log::info!(
                    "🌈 RDG Topology Changed! New Execution Order ({} passes): {current_names:?}",
                    current_names.len()
                );
                let dump = self.dump_mermaid();
                log::info!("\n🌈 RDG Compact Topology Mermaid Dump:\n{dump}");
            }
            self.storage.prev_execution_names = current_names;
        }
    }

    /// Dumps the current Render Graph topology as a Mermaid flowchart.
    ///
    /// When the `rdg_inspector` feature is enabled, passes that share the
    /// same [`with_group`](Self::with_group) scope are wrapped in Mermaid
    /// `subgraph` blocks for high-level readability.  Without the feature
    /// the output is a flat graph (still fully functional for debugging).
    #[must_use]
    pub fn dump_mermaid(&self) -> String {
        self.storage.dump_mermaid()
    }

    /// Notify every DPE-live pass after the frame has crossed the queue-submit
    /// boundary.
    ///
    /// The composer calls this exactly once after a successful submission.
    /// Keeping the traversal on `RenderGraph` centralizes access to the
    /// arena-backed pass nodes.
    pub(crate) fn run_after_submit_hooks(&mut self) {
        for &pass_idx in &self.storage.execution_queue {
            // SAFETY: The graph still owns the live FrameArena and this loop
            // visits each DPE-live pass sequentially after submission.
            unsafe { self.storage.passes[pass_idx].get_pass_mut() }.after_submit();
        }
    }

    /// Dumps the exact, unaggregated pass/resource topology.
    #[must_use]
    pub fn dump_mermaid_full(&self) -> String {
        self.storage.dump_mermaid_full()
    }

    /// Dumps the topology using the requested detail level.
    #[must_use]
    pub fn dump_mermaid_with_mode(&self, mode: MermaidDumpMode) -> String {
        self.storage.dump_mermaid_with_mode(mode)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::core::gpu::snapshot_asset_texture;
    use crate::core::{ResourceManager, WgpuContext};
    use crate::graph::RdgAttemptOutcome;
    use crate::graph::core::persistent::{CacheKey, PersistentTextureSpec};
    use crate::graph::core::types::ResourceClass;
    use crate::graph::{composer::GraphBuilderContext, core::context::ExecuteContext};
    use crate::pipeline::vertex::GeneratedVertexLayout;
    use crate::settings::{RendererInitConfig, RendererSettings};
    use myth_assets::AssetServer;
    use myth_resources::image::{Image, ImageDimension, PixelFormat};
    use myth_resources::texture::Texture;

    fn dummy_config() -> FrameConfig {
        FrameConfig {
            width: 1920,
            height: 1080,
            depth_format: wgpu::TextureFormat::Depth32Float,
            msaa_samples: 1,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            hdr_format: wgpu::TextureFormat::Rgba16Float,
        }
    }

    fn dummy_desc() -> TextureDesc {
        TextureDesc::new_2d(
            1,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }

    #[test]
    fn release_frame_owners_clears_interned_draw_plan_resources() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let buffer = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw-plan intern test"),
            size: 16,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let layout = wgpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("draw-plan intern test"),
                entries: &[],
            });
        let bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw-plan intern test"),
            layout: &layout,
            entries: &[],
        });

        let mut storage = GraphStorage::new();
        let buffer_slot = storage.draw_plans.intern_buffer(&buffer, 41);
        let duplicate_buffer_slot = storage.draw_plans.intern_buffer(&buffer, 41);
        let bind_group_slot = storage.draw_plans.intern_bind_group(&bind_group, 73);
        let duplicate_bind_group_slot = storage.draw_plans.intern_bind_group(&bind_group, 73);
        assert_eq!(buffer_slot, duplicate_buffer_slot);
        assert_eq!(bind_group_slot, duplicate_bind_group_slot);
        assert_eq!(storage.draw_plans.buffer_owner_count(), 1);
        assert_eq!(storage.draw_plans.bind_group_owner_count(), 1);

        storage.prepared_draws.begin_sources(1);
        storage.prepared_draws.insert_source(
            0,
            crate::graph::frame::PreparedDrawSource {
                geometry_asset: Arc::new(myth_resources::geometry::Geometry::default()),
                material_asset: Arc::new(myth_resources::material::Material::new_unlit(
                    glam::Vec4::ONE,
                )),
                geometry_asset_version: 0,
                geometry_asset_identity: 0,
                material_asset_version: 0,
                material_asset_identity: 0,
                vertex_buffers: smallvec::smallvec![buffer_slot],
                index_buffer: None,
                material_bind_group: bind_group_slot,
                material_bind_group_id: 73,
                object_bind_group: (bind_group_slot, 0),
                draw_range: 0..3,
                instance_range: 0..1,
                geometry_layout_info: Arc::new(GeneratedVertexLayout {
                    buffers: Vec::new(),
                    vertex_input_code: String::new(),
                    attribute_locations: rustc_hash::FxHashMap::default(),
                }),
                geometry_layout_id: 11,
                geometry_version: 12,
                geometry_layout_version: 0,
                material_layout: layout.clone(),
                material_layout_id: 13,
                material_binding_wgsl: Arc::from(""),
                material_binding_wgsl_hash: 0,
                material_version: 14,
                object_layout: layout.clone(),
                object_layout_id: 15,
                object_binding_wgsl: Arc::from(""),
                object_binding_wgsl_hash: 0,
            },
        );
        assert!(storage.prepared_draws.source(0).is_some());

        let plan = storage.draw_plans.reserve_plan(DrawPlanKey::Opaque);
        storage.draw_plans.mark_live(plan);
        let start = storage.draw_plans.begin_finalize(plan);
        storage.draw_plans.push_command(crate::graph::DrawCommand {
            source_item_index: 0,
            sort_key: 0,
            pipeline_id: crate::pipeline::RenderPipelineId(0),
            vertex_buffers: smallvec::smallvec![buffer_slot],
            index_buffer: None,
            bind_group_1: Some(bind_group_slot),
            bind_group_2: (bind_group_slot, 0),
            stencil_reference: None,
            vertex_range: 0..3,
            instance_range: 0..1,
        });
        storage.draw_plans.finish_finalize(plan, start);
        assert_eq!(storage.draw_plans.resolve_plan(plan).unwrap().len(), 1);

        drop(buffer);
        drop(bind_group);
        drop(layout);
        assert_eq!(
            storage.draw_plans.resolver().buffer(buffer_slot).size(),
            16,
            "the frame store must own the exact physical buffer through execute"
        );
        let _ = storage.draw_plans.resolver().bind_group(bind_group_slot);
        storage.release_frame_owners();
        assert!(storage.prepared_draws.source(0).is_none());
        assert!(storage.material_plans.is_empty());
        // clear() resets the one-shot initialization barrier for a new graph
        // attempt while retaining capacity.
        storage.prepared_draws.begin_sources(1);
        assert!(storage.draw_plans.is_empty());
        assert!(storage.global_state_attempt.prepared.is_empty());
        assert!(storage.draw_plans.resolve_plan(plan).is_none());
    }

    #[test]
    fn standalone_graph_drop_publishes_exact_abandoned_attempt() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            graph.add_pass("Declared_Only", |_builder| (MockExec, ()));

            let current = graph.diagnostics();
            assert_eq!(current.outcome(), RdgAttemptOutcome::InProgress);
            assert_eq!(current.declared_passes(), 1);
            assert_eq!(current.live_passes(), None);
        }

        let abandoned = storage.latest_diagnostics().unwrap();
        assert_eq!(abandoned.outcome(), RdgAttemptOutcome::Abandoned);
        assert_eq!(abandoned.declared_passes(), 1);
        assert_eq!(abandoned.live_passes(), None);
        assert_eq!(abandoned.prepared_passes(), 0);
        assert_eq!(abandoned.executed_passes(), 0);
        assert!(!abandoned.queue_submitted());
    }

    #[test]
    fn all_dead_graph_records_compiled_zero_and_survives_topology_clear() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            graph.add_pass("Dead", |_builder| (MockExec, ()));
            graph.compile_topology();

            let current = graph.diagnostics();
            assert_eq!(current.declared_passes(), 1);
            assert_eq!(current.live_passes(), Some(0));
            graph.finish();
        }

        let submitted = storage.latest_diagnostics().unwrap();
        assert_eq!(
            submitted.outcome(),
            RdgAttemptOutcome::CompletedWithoutSubmission
        );
        assert!(!submitted.queue_submitted());
        assert_eq!(submitted.live_passes(), Some(0));
        assert_eq!(submitted.prepared_passes(), 0);
        assert_eq!(submitted.executed_passes(), 0);

        storage.clear();
        storage.clear();
        assert_eq!(storage.latest_diagnostics(), Some(submitted));
    }

    #[test]
    fn pass_setup_panic_does_not_count_a_placeholder_as_declared() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            graph.add_pass("Panicking_Setup", |_builder| -> (MockExec, ()) {
                panic!("intentional setup failure")
            });
        }));
        assert!(result.is_err());

        let abandoned = storage.latest_diagnostics().unwrap();
        assert_eq!(abandoned.outcome(), RdgAttemptOutcome::Abandoned);
        assert_eq!(abandoned.declared_passes(), 0);
        assert_eq!(abandoned.live_passes(), None);
        assert!(abandoned.panicked());
    }

    #[test]
    fn next_attempt_clears_raw_draw_staging_after_setup_panic() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut graph = begin_test_frame(&mut storage, &arena);
            let texture = graph.register_texture("Abandoned", dummy_desc(), false);
            graph.set_draw_plan_texture_dependencies(DrawPlanKey::Opaque, [texture]);
            let plan = graph.reserve_draw_plan(DrawPlanKey::Opaque);
            graph.add_pass("Panicking_Draw_Setup", |builder| -> (MockExec, ()) {
                builder.read_texture(texture);
                builder.read_texture(texture);
                builder.demand_draw_plan(plan);
                panic!("intentional draw setup failure")
            });
        }));
        assert!(result.is_err());
        assert!(storage.draw_plan_raw_demands.is_empty());
        assert!(storage.logical_reads.is_empty());
        assert!(storage.pass_reads(0).is_empty());
        assert_eq!(
            storage.resources[0].consumer_edge_count(),
            0,
            "setup unwind must remove its logical-read consumer edge"
        );

        let mut graph = begin_test_frame(&mut storage, &arena);
        assert!(graph.storage.passes.is_empty());
        assert!(graph.storage.logical_reads.is_empty());
        assert!(graph.storage.draw_plan_raw_demands.is_empty());
        assert!(graph.storage.draw_plan_demands.is_empty());

        let texture = graph.register_texture("Recovered", dummy_desc(), false);
        graph.set_draw_plan_texture_dependencies(DrawPlanKey::Transparent, [texture]);
        let plan = graph.reserve_draw_plan(DrawPlanKey::Transparent);
        let pass_index = graph.storage.passes.len();
        graph.add_pass("Recovered_Draw_Setup", |builder| {
            builder.demand_draw_plan(plan);
            (MockExec, ())
        });
        assert_eq!(graph.storage.pass_draw_plan_demands(pass_index), &[plan]);
        assert_eq!(graph.storage.pass_reads(pass_index), &[texture.erase()]);
    }

    #[test]
    fn second_setup_panic_rolls_back_only_its_read_tail() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut graph = begin_test_frame(&mut storage, &arena);
            let first_input = graph.register_texture("First_Input", dummy_desc(), false);
            let second_input = graph.register_texture("Second_Input", dummy_desc(), false);
            graph.add_pass("Completed_First", |builder| {
                builder.read_texture(first_input);
                builder.mark_side_effect();
                (MockExec, ())
            });
            graph.add_pass("Panicking_Second", |builder| -> (MockExec, ()) {
                builder.read_texture(second_input);
                panic!("intentional second setup failure")
            });
        }));
        assert!(result.is_err());

        assert_eq!(
            storage.pass_reads(0),
            &[ErasedResourceNodeId::new(0, 0)],
            "completed pass prefix must survive later setup unwind"
        );
        assert!(storage.pass_reads(1).is_empty());
        assert_eq!(storage.logical_reads.len(), 1);
        assert_eq!(storage.resource_consumers(0).collect::<Vec<_>>(), [0]);
        assert_eq!(storage.resources[0].consumer_edge_count(), 1);
        assert_eq!(storage.resources[1].consumer_edge_count(), 0);
    }

    #[test]
    fn repeated_reads_are_distinct_consumer_edges_derived_from_pass_reads() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let input = graph.register_texture("Repeated_Input", dummy_desc(), false);
        let pass_index = graph.storage.passes.len();

        graph.add_pass("Repeated_Reader", |builder| {
            builder.read_texture(input);
            builder.read_texture(input);
            builder.mark_side_effect();
            (MockExec, ())
        });

        assert_eq!(
            graph.storage.pass_reads(pass_index),
            &[input.erase(), input.erase()]
        );
        assert_eq!(
            graph.storage.resources[input.index() as usize].consumer_edge_count(),
            2
        );
        assert_eq!(
            graph
                .storage
                .resource_consumers(input.index() as usize)
                .collect::<Vec<_>>(),
            [pass_index, pass_index]
        );
    }

    #[test]
    fn duplicate_request_owners_share_publication_and_retry_after_attempt_release() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![255; 8 * 8 * 4]),
        ));
        let mut texture = Texture::new_2d(Some("duplicate mip owners"), image);
        texture.generate_mipmaps = true;
        let texture = assets.textures.add(texture);

        resources.prepare_texture(&assets, texture);
        let first = resources
            .take_mipmap_requests()
            .pop()
            .expect("first request owner missing");
        resources.prepare_texture(&assets, texture);
        let second = resources
            .take_mipmap_requests()
            .pop()
            .expect("second request owner missing");
        assert_eq!(first.key(), second.key());
        assert!(first.shares_submission_token(&second));

        let mut storage = GraphStorage::new();
        assert!(storage.mipmap_encode_coordinator.try_claim(first.key()));
        assert!(!storage.mipmap_encode_coordinator.try_claim(second.key()));
        first.mark_submitted();
        assert!(
            resources
                .get_texture_image(texture)
                .expect("physical texture missing")
                .mipmaps_submitted()
        );

        storage.release_frame_owners();
        assert!(storage.mipmap_encode_coordinator.try_claim(second.key()));
    }

    fn dummy_asset_snapshot() -> AssetTextureSnapshot {
        AssetTextureSnapshot {
            texture_handle: myth_assets::TextureHandle::system_reserved(7),
            image_handle: myth_assets::ImageHandle::default(),
            texture_version: 3,
            texture_identity: 33,
            image_version: 5,
            image_identity: 55,
            logical_version: 0xA55E_7001,
            extent: wgpu::Extent3d {
                width: 8,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Rgba8Unorm,
            dimension: wgpu::TextureDimension::D2,
            view_dimension: wgpu::TextureViewDimension::D2,
            mip_count: 4,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            generate_mipmaps: true,
        }
    }

    // ─── Shared Mock Pass Type ───────────────────────────────────────

    struct MockExec;
    impl PassNode<'_> for MockExec {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    #[test]
    fn draw_plan_demands_follow_dpe_and_deduplicate_shared_consumers() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let dead_texture = graph.register_texture("Dead_Draw_Texture", dummy_desc(), false);
        graph.add_pass("Dead_Texture_Producer", |builder| {
            builder.write_texture(dead_texture);
            (MockExec, ())
        });
        let shared_texture = graph.register_texture("Shared_Draw_Texture", dummy_desc(), false);
        graph.add_pass("Shared_Texture_Producer", |builder| {
            builder.write_texture(shared_texture);
            (MockExec, ())
        });
        graph.set_draw_plan_texture_dependencies(DrawPlanKey::Transparent, [dead_texture]);
        graph.set_draw_plan_texture_dependencies(
            DrawPlanKey::Opaque,
            [shared_texture, shared_texture],
        );

        let dead = graph.reserve_draw_plan(DrawPlanKey::Transparent);
        let shared = graph.reserve_draw_plan(DrawPlanKey::Opaque);
        let second = graph.reserve_draw_plan(DrawPlanKey::Prepass {
            needs_normal: true,
            needs_feature_id: false,
            needs_velocity: true,
        });

        graph.add_pass("Dead_Draw_Consumer", |builder| {
            builder.demand_draw_plan(dead);
            (MockExec, ())
        });
        graph.add_pass("Live_Shared_Consumer", |builder| {
            builder.demand_draw_plan(shared);
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph.add_pass("Live_Two_Plan_Consumer", |builder| {
            builder.demand_draw_plan(shared);
            builder.demand_draw_plan(second);
            builder.demand_draw_plan(second);
            builder.mark_side_effect();
            (MockExec, ())
        });

        assert_eq!(graph.storage.pass_draw_plan_demands(4).len(), 2);
        assert_eq!(
            graph.storage.pass_reads(4),
            &[shared_texture.erase()],
            "repeated demands and Prepass/Opaque sharing must add one physical read"
        );
        graph.compile_topology();

        assert_eq!(graph.storage.passes[0].reference_count, 0);
        assert!(!graph.storage.draw_plans.is_live(dead));
        assert!(graph.storage.draw_plans.is_live(shared));
        assert!(graph.storage.draw_plans.is_live(second));
        assert!(graph.storage.draw_plans.resolve_plan(shared).is_none());
        assert!(graph.storage.draw_plans.resolve_plan(second).is_none());
        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(
            live,
            [
                "Shared_Texture_Producer",
                "Live_Shared_Consumer",
                "Live_Two_Plan_Consumer"
            ]
        );
    }

    #[test]
    fn draw_plan_demands_are_raw_during_setup_and_canonical_afterward() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let texture = graph.register_texture("Draw_Texture", dummy_desc(), false);
        graph.set_draw_plan_texture_dependencies(DrawPlanKey::Opaque, [texture, texture]);
        let plan = graph.reserve_draw_plan(DrawPlanKey::Opaque);
        let pass_index = graph.storage.passes.len();

        graph.add_pass("Repeated_Draw_Consumer", |builder| {
            builder.demand_draw_plan(plan);
            builder.demand_draw_plan(plan);
            builder.demand_draw_plan(plan);
            assert_eq!(
                builder.graph.storage.draw_plan_raw_demands.len(),
                3,
                "setup must append raw IDs without accumulated-demand scans"
            );
            assert!(
                builder
                    .graph
                    .storage
                    .pass_reads(builder.pass_index)
                    .is_empty(),
                "draw-plan dependency union must wait until setup returns"
            );
            (MockExec, ())
        });

        assert_eq!(graph.storage.pass_draw_plan_demands(pass_index), &[plan]);
        assert_eq!(graph.storage.pass_reads(pass_index), &[texture.erase()]);
        assert_eq!(
            graph
                .storage
                .resource_consumers(texture.index() as usize)
                .collect::<Vec<_>>(),
            &[pass_index]
        );
    }

    #[test]
    fn repeated_scene_submission_demand_is_canonical_and_projects_cpu_direct_live() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let dependency = graph.add_pass("Submission_Texture_Producer", |builder| {
            let texture = builder.create_texture("Submission_Texture", dummy_desc());
            (MockExec, texture)
        });
        let draw_plan = graph.reserve_draw_plan(DrawPlanKey::Opaque);
        let submission = graph.reserve_cpu_direct_scene_submission(
            draw_plan,
            &SceneSubmissionDependencies::new().with_texture(dependency),
        );
        let pass_index = graph.storage.passes.len();

        graph.add_pass("Repeated_Submission_Consumer", |builder| {
            builder.demand_scene_submission(submission);
            builder.demand_scene_submission(submission);
            builder.demand_scene_submission(submission);
            assert_eq!(builder.graph.storage.scene_submission_raw_demands.len(), 3);
            assert!(
                builder
                    .graph
                    .storage
                    .pass_reads(builder.pass_index)
                    .is_empty(),
                "submission dependencies must be finalized only after setup"
            );
            builder.mark_side_effect();
            (MockExec, ())
        });

        assert_eq!(
            graph.storage.pass_scene_submission_demands(pass_index),
            &[submission]
        );
        assert_eq!(graph.storage.pass_reads(pass_index), &[dependency.erase()]);
        assert_eq!(
            graph
                .storage
                .resource_consumers(dependency.index() as usize)
                .collect::<Vec<_>>(),
            &[pass_index]
        );

        graph.compile_topology();
        assert!(graph.storage.scene_submissions.is_live(submission));
        assert!(
            graph.storage.draw_plans.is_live(draw_plan),
            "the CpuDirect adapter must synchronize live submission state"
        );
    }

    #[test]
    fn cpu_direct_scene_submission_matches_legacy_draw_projection() {
        fn observe(
            use_submission: bool,
        ) -> (
            Vec<ErasedResourceNodeId>,
            Vec<&'static str>,
            Vec<DrawPlanKey>,
        ) {
            let mut storage = GraphStorage::new();
            let arena = FrameArena::new();
            let mut graph = begin_test_frame(&mut storage, &arena);
            let dependency = graph.add_pass("Direct_Projection_Texture", |builder| {
                let texture = builder.create_texture("Direct_Projection_Input", dummy_desc());
                (MockExec, texture)
            });
            graph.set_draw_plan_texture_dependencies(DrawPlanKey::Opaque, [dependency]);
            let draw_plan = graph.reserve_draw_plan(DrawPlanKey::Opaque);
            let submission = use_submission.then(|| {
                graph.reserve_cpu_direct_scene_submission(
                    draw_plan,
                    &SceneSubmissionDependencies::new().with_texture(dependency),
                )
            });
            let consumer_index = graph.storage.passes.len();
            graph.add_pass("Direct_Projection_Consumer", |builder| {
                if let Some(submission) = submission {
                    builder.demand_scene_submission(submission);
                } else {
                    builder.demand_draw_plan(draw_plan);
                }
                builder.mark_side_effect();
                (MockExec, ())
            });
            let reads = graph.storage.pass_reads(consumer_index).to_vec();
            graph.compile_topology();
            let execution = graph
                .storage
                .execution_queue
                .iter()
                .map(|&index| graph.storage.passes[index].name)
                .collect();
            let live_keys = graph.storage.draw_plans.live_keys().collect();
            (reads, execution, live_keys)
        }

        assert_eq!(observe(true), observe(false));
    }

    #[test]
    fn scene_submission_mixed_dependencies_preserve_first_occurrence_order() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let texture_a = graph.register_texture("Submission_Texture_A", dummy_desc(), false);
        let buffer = graph.register_buffer(
            "Submission_Buffer",
            BufferDesc::new(64, wgpu::BufferUsages::STORAGE),
            false,
        );
        let texture_b = graph.register_texture("Submission_Texture_B", dummy_desc(), false);

        let first_draw = graph.reserve_draw_plan(DrawPlanKey::Opaque);
        let first_submission = graph.reserve_cpu_direct_scene_submission(
            first_draw,
            &SceneSubmissionDependencies::new()
                .with_texture(texture_a)
                .with_buffer(buffer),
        );
        let second_draw = graph.reserve_draw_plan(DrawPlanKey::Transparent);
        let mut second_dependencies = SceneSubmissionDependencies::new()
            .with_texture(texture_b)
            .with_buffer(buffer);
        second_dependencies.read_texture(texture_b);
        let second_submission =
            graph.reserve_cpu_direct_scene_submission(second_draw, &second_dependencies);
        let pass_index = graph.storage.passes.len();

        graph.add_pass("Mixed_Submission_Consumer", |builder| {
            builder.demand_scene_submission(second_submission);
            builder.demand_scene_submission(first_submission);
            builder.demand_scene_submission(second_submission);
            (MockExec, ())
        });

        assert_eq!(
            graph.storage.pass_scene_submission_demands(pass_index),
            &[second_submission, first_submission]
        );
        assert_eq!(
            graph.storage.pass_reads(pass_index),
            &[texture_b.erase(), buffer.erase(), texture_a.erase()],
            "texture and buffer dependencies must share one ordered dedup domain"
        );
    }

    #[test]
    fn all_dead_scene_submission_demand_does_not_root_or_mark_backends_live() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let dependency = graph.add_pass("Dead_Submission_Producer", |builder| {
            let texture = builder.create_texture("Dead_Submission_Texture", dummy_desc());
            (MockExec, texture)
        });
        let draw_plan = graph.reserve_draw_plan(DrawPlanKey::Opaque);
        let submission = graph.reserve_cpu_direct_scene_submission(
            draw_plan,
            &SceneSubmissionDependencies::new().with_texture(dependency),
        );
        let indirect_dependency = graph.add_pass("Dead_Indirect_Args_Producer", |builder| {
            let buffer = builder.create_buffer(
                "Dead_Indirect_Args",
                BufferDesc::new(
                    16,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                ),
            );
            (MockExec, buffer)
        });
        let indirect = graph.reserve_indirect_scene_submission_for_test(
            1,
            &SceneSubmissionDependencies::new().with_buffer(indirect_dependency),
        );
        graph.add_pass("Dead_Submission_Consumer", |builder| {
            builder.demand_scene_submission(submission);
            builder.demand_scene_submission(indirect);
            (MockExec, ())
        });

        graph.compile_topology();

        assert!(graph.storage.execution_queue.is_empty());
        assert!(!graph.storage.scene_submissions.is_live(submission));
        assert!(!graph.storage.scene_submissions.is_live(indirect));
        assert_eq!(
            graph.storage.scene_submissions.live_backend_counts(),
            super::super::scene_submission::LiveSceneSubmissionCounts::default()
        );
        assert!(!graph.storage.draw_plans.is_live(draw_plan));
    }

    #[test]
    fn indirect_stub_buffers_are_ordinary_ordered_dpe_dependencies() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let make_buffer = |graph: &mut RenderGraph<'_>, pass_name, resource_name, usage| {
            graph.add_pass(pass_name, |builder| {
                let buffer = builder.create_buffer(resource_name, BufferDesc::new(64, usage));
                (MockExec, buffer)
            })
        };
        let instance = make_buffer(
            &mut graph,
            "Indirect_Instance_Producer",
            "Indirect_Instances",
            wgpu::BufferUsages::STORAGE,
        );
        let args = make_buffer(
            &mut graph,
            "Indirect_Args_Producer",
            "Indirect_Args",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        let count = make_buffer(
            &mut graph,
            "Indirect_Count_Producer",
            "Indirect_Count",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        let submission = graph.reserve_indirect_scene_submission_for_test(
            7,
            &SceneSubmissionDependencies::new()
                .with_buffer(instance)
                .with_buffer(args)
                .with_buffer(count),
        );
        let consumer_index = graph.storage.passes.len();
        graph.add_pass("Indirect_Submission_Consumer", |builder| {
            builder.demand_scene_submission(submission);
            builder.demand_scene_submission(submission);
            builder.mark_side_effect();
            (MockExec, ())
        });

        assert_eq!(
            graph.storage.pass_reads(consumer_index),
            &[instance.erase(), args.erase(), count.erase()]
        );
        graph.compile_topology();

        assert!(graph.storage.scene_submissions.is_live(submission));
        let counts = graph.storage.scene_submissions.live_backend_counts();
        assert_eq!(counts.cpu_direct, 0);
        assert_eq!(counts.indirect_stub, 1);
        assert!(
            graph
                .storage
                .scene_submissions
                .resolve_stream(submission, &graph.storage.draw_plans)
                .is_none(),
            "a topology-only indirect recipe must not invent a finalized stream"
        );
        let execution_names = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect::<Vec<_>>();
        assert_eq!(
            execution_names,
            [
                "Indirect_Instance_Producer",
                "Indirect_Args_Producer",
                "Indirect_Count_Producer",
                "Indirect_Submission_Consumer",
            ]
        );
    }

    #[test]
    #[should_panic(expected = "stale or foreign scene-submission plan")]
    fn foreign_scene_submission_demand_is_rejected_during_setup() {
        let mut first_storage = GraphStorage::new();
        let first_arena = FrameArena::new();
        let mut first_graph = begin_test_frame(&mut first_storage, &first_arena);
        let draw_plan = first_graph.reserve_draw_plan(DrawPlanKey::Opaque);
        let foreign = first_graph
            .reserve_cpu_direct_scene_submission(draw_plan, &SceneSubmissionDependencies::new());

        let mut second_storage = GraphStorage::new();
        let second_arena = FrameArena::new();
        let mut second_graph = begin_test_frame(&mut second_storage, &second_arena);
        second_graph.add_pass("Foreign_Submission_Consumer", |builder| {
            builder.demand_scene_submission(foreign);
            (MockExec, ())
        });
    }

    #[test]
    #[should_panic(expected = "stale or foreign scene-submission plan")]
    fn stale_scene_submission_demand_is_rejected_after_attempt_release() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let stale = {
            let mut graph = begin_test_frame(&mut storage, &arena);
            let draw_plan = graph.reserve_draw_plan(DrawPlanKey::Opaque);
            graph
                .reserve_cpu_direct_scene_submission(draw_plan, &SceneSubmissionDependencies::new())
        };

        let mut graph = begin_test_frame(&mut storage, &arena);
        graph.add_pass("Stale_Submission_Consumer", |builder| {
            builder.demand_scene_submission(stale);
            (MockExec, ())
        });
    }

    #[test]
    #[should_panic(expected = "nested RenderGraph::add_pass is unsupported")]
    fn nested_add_pass_is_rejected_before_mutating_graph_state() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        graph.add_pass("Outer", |builder| {
            builder.graph.add_pass("Nested", |_nested| (MockExec, ()));
            (MockExec, ())
        });
    }

    #[test]
    fn caught_nested_add_pass_panic_preserves_outer_raw_demands() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let texture = graph.register_texture("Outer_Texture", dummy_desc(), false);
        graph.set_draw_plan_texture_dependencies(DrawPlanKey::Opaque, [texture]);
        let plan = graph.reserve_draw_plan(DrawPlanKey::Opaque);
        let outer_index = graph.storage.passes.len();

        graph.add_pass("Outer", |builder| {
            builder.demand_draw_plan(plan);
            let nested = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                builder.graph.add_pass("Nested", |_nested| (MockExec, ()));
            }));
            assert!(nested.is_err());
            builder.demand_draw_plan(plan);
            (MockExec, ())
        });

        assert_eq!(graph.storage.passes.len(), 1);
        assert_eq!(graph.storage.pass_draw_plan_demands(outer_index), &[plan]);
        assert_eq!(graph.storage.pass_reads(outer_index), &[texture.erase()]);
        assert_eq!(
            graph
                .storage
                .resource_consumers(texture.index() as usize)
                .collect::<Vec<_>>(),
            &[outer_index]
        );
    }

    #[test]
    fn disjoint_draw_plans_union_dependencies_in_first_occurrence_order() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let direct = graph.register_texture("Direct", dummy_desc(), false);
        let shared = graph.register_texture("Shared", dummy_desc(), false);
        let opaque_only = graph.register_texture("Opaque", dummy_desc(), false);
        let transparent_only = graph.register_texture("Transparent", dummy_desc(), false);

        graph
            .set_draw_plan_texture_dependencies(DrawPlanKey::Opaque, [shared, opaque_only, shared]);
        graph.set_draw_plan_texture_dependencies(
            DrawPlanKey::Transparent,
            [transparent_only, shared, transparent_only],
        );
        let opaque = graph.reserve_draw_plan(DrawPlanKey::Opaque);
        let transparent = graph.reserve_draw_plan(DrawPlanKey::Transparent);
        let pass_index = graph.storage.passes.len();

        graph.add_pass("Disjoint_Draw_Consumer", |builder| {
            builder.read_texture(direct);
            builder.read_texture(shared);
            builder.demand_draw_plan(transparent);
            builder.demand_draw_plan(opaque);
            builder.demand_draw_plan(transparent);
            builder.mark_side_effect();
            (MockExec, ())
        });

        assert_eq!(
            graph.storage.pass_draw_plan_demands(pass_index),
            &[transparent, opaque]
        );
        assert_eq!(
            graph.storage.pass_reads(pass_index),
            &[
                direct.erase(),
                shared.erase(),
                transparent_only.erase(),
                opaque_only.erase()
            ]
        );
        for texture in [direct, shared, opaque_only, transparent_only] {
            assert_eq!(
                graph
                    .storage
                    .resource_consumers(texture.index() as usize)
                    .collect::<Vec<_>>(),
                &[pass_index],
                "the union must install exactly one consumer edge per resource"
            );
        }

        graph.compile_topology();
        assert!(graph.storage.draw_plans.is_live(opaque));
        assert!(graph.storage.draw_plans.is_live(transparent));
    }

    #[test]
    fn draw_plan_resource_scratch_recovers_after_iterator_panic() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let abandoned = graph.register_texture("Abandoned", dummy_desc(), false);
        let retained = graph.register_texture("Retained", dummy_desc(), false);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            graph.set_draw_plan_texture_dependencies(
                DrawPlanKey::Opaque,
                std::iter::once(abandoned)
                    .chain(std::iter::once_with(|| panic!("dependency iterator panic"))),
            );
        }));
        assert!(result.is_err());

        graph.set_draw_plan_texture_dependencies(DrawPlanKey::Transparent, [retained, retained]);
        let plan = graph.reserve_draw_plan(DrawPlanKey::Transparent);
        let pass_index = graph.storage.passes.len();
        graph.add_pass("Post_Panic_Consumer", |builder| {
            builder.demand_draw_plan(plan);
            (MockExec, ())
        });

        assert_eq!(graph.storage.pass_reads(pass_index), &[retained.erase()]);
        assert_eq!(
            graph.storage.resources[abandoned.index() as usize].consumer_edge_count(),
            0
        );
    }

    #[test]
    fn dense_generation_scratch_retains_capacity_and_clears_on_wrap() {
        let mut scratch = DenseGenerationScratch::default();
        scratch.begin(64);
        assert!(scratch.insert(7));
        assert!(!scratch.insert(7));
        let capacity = scratch.stamps.capacity();

        scratch.begin(8);
        assert_eq!(scratch.stamps.len(), 64, "the dense domain must not shrink");
        assert_eq!(scratch.stamps.capacity(), capacity);
        assert!(scratch.insert(7));

        scratch.generation = u32::MAX;
        scratch.stamps[7] = 1;
        scratch.begin(8);
        assert_eq!(scratch.generation, 1);
        assert!(
            scratch.insert(7),
            "wrap must clear stale generation-one marks"
        );
        assert!(!scratch.insert(7));
    }

    #[test]
    #[should_panic(
        expected = "draw-plan texture dependencies must be registered before the first consumer"
    )]
    fn draw_plan_texture_dependencies_reject_late_registration() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let plan = graph.reserve_draw_plan(DrawPlanKey::Opaque);
        graph.add_pass("Consumer", |builder| {
            builder.demand_draw_plan(plan);
            (MockExec, ())
        });
        let texture = graph.register_texture("Late_Texture", dummy_desc(), false);
        graph.set_draw_plan_texture_dependencies(DrawPlanKey::Opaque, [texture]);
    }

    #[test]
    #[should_panic(
        expected = "draw-plan texture dependencies must be registered before the first consumer"
    )]
    fn prepass_demand_rejects_late_opaque_dependency_registration() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let prepass = graph.reserve_draw_plan(DrawPlanKey::Prepass {
            needs_normal: true,
            needs_feature_id: false,
            needs_velocity: true,
        });
        graph.add_pass("Prepass_Consumer", |builder| {
            builder.demand_draw_plan(prepass);
            (MockExec, ())
        });

        let texture = graph.register_texture("Late_Opaque_Texture", dummy_desc(), false);
        graph.set_draw_plan_texture_dependencies(DrawPlanKey::Opaque, [texture]);
    }

    #[test]
    fn retained_demand_and_read_ranges_restart_at_zero_after_attempt_clear() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        let first_plan;
        let first_texture;
        {
            let mut graph = begin_test_frame(&mut storage, &arena);
            first_texture = graph.register_texture("First_Texture", dummy_desc(), false);
            graph.set_draw_plan_texture_dependencies(DrawPlanKey::Opaque, [first_texture]);
            first_plan = graph.reserve_draw_plan(DrawPlanKey::Opaque);
            graph.add_pass("First_Consumer", |builder| {
                builder.demand_draw_plan(first_plan);
                (MockExec, ())
            });
        }
        assert_eq!(storage.pass_draw_plan_demands(0), &[first_plan]);
        assert_eq!(storage.passes[0].draw_plan_demand_range, 0..1);
        assert_eq!(storage.pass_reads(0), &[first_texture.erase()]);
        assert_eq!(storage.passes[0].logical_read_range, 0..1);
        let retained_read_capacity = storage.logical_reads.capacity();

        let mut graph = begin_test_frame(&mut storage, &arena);
        assert_eq!(
            graph.storage.logical_reads.capacity(),
            retained_read_capacity
        );
        let second_texture = graph.register_texture("Second_Texture", dummy_desc(), false);
        graph.set_draw_plan_texture_dependencies(DrawPlanKey::Transparent, [second_texture]);
        let second_plan = graph.reserve_draw_plan(DrawPlanKey::Transparent);
        graph.add_pass("Second_Consumer", |builder| {
            builder.demand_draw_plan(second_plan);
            (MockExec, ())
        });
        assert_eq!(graph.storage.pass_draw_plan_demands(0), &[second_plan]);
        assert_eq!(graph.storage.passes[0].draw_plan_demand_range, 0..1);
        assert_eq!(graph.storage.pass_reads(0), &[second_texture.erase()]);
        assert_eq!(graph.storage.passes[0].logical_read_range, 0..1);
    }

    #[test]
    fn physical_dependency_ranges_survive_drop_and_restart_after_clear() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        {
            let mut graph = begin_test_frame(&mut storage, &arena);
            let output = graph.add_pass("First_Producer", |builder| {
                let output = builder.create_texture("First_Output", dummy_desc());
                (MockExec, output)
            });
            graph.add_pass("First_Consumer", |builder| {
                builder.read_texture(output);
                builder.mark_side_effect();
                (MockExec, ())
            });
            graph.compile_topology();
            assert_eq!(graph.storage.pass_physical_dependencies(1), &[0]);
            assert_eq!(graph.storage.passes[1].physical_dependency_range, 0..1);
        }

        assert_eq!(storage.pass_physical_dependencies(1), &[0]);
        let retained_capacity = storage.physical_dependencies.capacity();

        let mut graph = begin_test_frame(&mut storage, &arena);
        assert!(graph.storage.physical_dependencies.is_empty());
        assert!(graph.storage.compile_reverse_offsets.is_empty());
        assert_eq!(
            graph.storage.physical_dependencies.capacity(),
            retained_capacity
        );
        let output = graph.add_pass("Second_Producer", |builder| {
            let output = builder.create_texture("Second_Output", dummy_desc());
            (MockExec, output)
        });
        graph.add_pass("Second_Consumer", |builder| {
            builder.read_texture(output);
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph.compile_topology();
        assert_eq!(graph.storage.pass_physical_dependencies(1), &[0]);
        assert_eq!(graph.storage.passes[1].physical_dependency_range, 0..1);
    }

    #[test]
    fn physical_dependency_scratch_wrap_clears_stale_producer_marks() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let output = graph.add_pass("Wrap_Producer", |builder| {
            let output = builder.create_texture("Wrap_Output", dummy_desc());
            (MockExec, output)
        });
        graph.add_pass("Wrap_Consumer", |builder| {
            builder.read_texture(output);
            builder.read_texture(output);
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph
            .storage
            .physical_dependency_scratch
            .stamps
            .resize(2, 1);
        graph.storage.physical_dependency_scratch.generation = u32::MAX - 1;

        graph.compile_topology();

        assert_eq!(graph.storage.pass_physical_dependencies(1), &[0]);
        assert_eq!(
            graph.storage.physical_dependency_scratch.generation, 1,
            "the consumer pass must trigger the wrap clear"
        );
    }

    #[test]
    fn compile_panic_partial_physical_tail_is_cleared_by_next_attempt() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let first = graph.add_pass("Panic_Producer_A", |builder| {
            let output = builder.create_texture("Panic_A", dummy_desc());
            (MockExec, output)
        });
        let second = graph.add_pass("Panic_Producer_B", |builder| {
            let output = builder.create_texture("Panic_B", dummy_desc());
            (MockExec, output)
        });
        graph.add_pass("Panic_Consumer", |builder| {
            builder.read_texture(first);
            builder.read_texture(second);
            builder.mark_side_effect();
            (MockExec, ())
        });
        let second_read = graph.storage.passes[2].logical_read_range.start + 1;
        graph.storage.logical_reads[second_read] = ErasedResourceNodeId::new(u32::MAX, 0);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            graph.compile_topology();
        }));
        assert!(result.is_err());
        drop(graph);

        assert_eq!(storage.pass_physical_dependencies(2), &[0]);
        let retained_capacity = storage.physical_dependencies.capacity();

        let mut graph = begin_test_frame(&mut storage, &arena);
        assert!(graph.storage.physical_dependencies.is_empty());
        assert_eq!(
            graph.storage.physical_dependencies.capacity(),
            retained_capacity
        );
        let output = graph.add_pass("Recovered_Producer", |builder| {
            let output = builder.create_texture("Recovered_Output", dummy_desc());
            (MockExec, output)
        });
        graph.add_pass("Recovered_Consumer", |builder| {
            builder.read_texture(output);
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph.compile_topology();
        assert_eq!(graph.storage.pass_physical_dependencies(1), &[0]);
        assert_eq!(graph.storage.passes[1].physical_dependency_range, 0..1);
    }

    #[test]
    fn large_draw_plan_texture_sets_use_deduplicated_physical_edges() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let mut textures = Vec::new();
        for _ in 0..24 {
            let texture = graph.register_texture("Draw_Texture", dummy_desc(), false);
            graph.add_pass("Draw_Texture_Producer", |builder| {
                builder.write_texture(texture);
                (MockExec, ())
            });
            textures.push(texture);
        }
        graph.set_draw_plan_texture_dependencies(
            DrawPlanKey::Opaque,
            textures.iter().chain(&textures).copied(),
        );
        let plan = graph.reserve_draw_plan(DrawPlanKey::Opaque);
        let consumer = graph.storage.passes.len();
        graph.add_pass("Large_Draw_Consumer", |builder| {
            builder.demand_draw_plan(plan);
            builder.demand_draw_plan(plan);
            builder.mark_side_effect();
            (MockExec, ())
        });

        assert_eq!(graph.storage.pass_reads(consumer).len(), textures.len());
        graph.compile_topology();
        let expected_producers: Vec<_> = (0..textures.len()).collect();
        assert_eq!(
            graph.storage.pass_physical_dependencies(consumer),
            expected_producers,
            ">16 fan-in must remain linear and preserve first producer order"
        );
        assert_eq!(graph.storage.execution_queue.len(), textures.len() + 1);
    }

    #[test]
    fn physical_dependencies_deduplicate_in_first_read_producer_order() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let a = graph.add_pass("Producer_A", |builder| {
            let output = builder.create_texture("A", dummy_desc());
            (MockExec, output)
        });
        let b = graph.add_pass("Producer_B", |builder| {
            let output = builder.create_texture("B", dummy_desc());
            (MockExec, output)
        });
        let c = graph.add_pass("Producer_C", |builder| {
            let output = builder.create_texture("C", dummy_desc());
            (MockExec, output)
        });
        let consumer = graph.storage.passes.len();
        graph.add_pass("Ordered_Consumer", |builder| {
            builder.read_texture(b);
            builder.read_texture(a);
            builder.read_texture(b);
            builder.read_texture(c);
            builder.read_texture(a);
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(
            graph.storage.pass_physical_dependencies(consumer),
            &[1, 0, 2]
        );
        for producer in 0..3 {
            assert_eq!(
                graph.storage.compiled_reverse_consumers(producer),
                &[consumer]
            );
        }
    }

    #[test]
    fn csr_topology_preserves_diamond_and_independent_ready_order() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let a = graph.add_pass("Diamond_A", |builder| {
            let output = builder.create_texture("Diamond_A_Output", dummy_desc());
            (MockExec, output)
        });
        let b = graph.add_pass("Diamond_B", |builder| {
            builder.read_texture(a);
            let output = builder.create_texture("Diamond_B_Output", dummy_desc());
            (MockExec, output)
        });
        let c = graph.add_pass("Diamond_C", |builder| {
            builder.read_texture(a);
            let output = builder.create_texture("Diamond_C_Output", dummy_desc());
            (MockExec, output)
        });
        graph.add_pass("Diamond_D", |builder| {
            builder.read_texture(b);
            builder.read_texture(c);
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph.add_pass("Independent", |builder| {
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.storage.compiled_reverse_consumers(0), &[1, 2]);
        assert_eq!(graph.storage.compiled_reverse_consumers(1), &[3]);
        assert_eq!(graph.storage.compiled_reverse_consumers(2), &[3]);
        assert_eq!(graph.storage.pass_physical_dependencies(3), &[1, 2]);
        let order: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(
            order,
            [
                "Diamond_A",
                "Diamond_B",
                "Diamond_C",
                "Diamond_D",
                "Independent"
            ]
        );
    }

    #[test]
    fn csr_reverse_adjacency_handles_more_than_eight_consumers_without_reordering() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let output = graph.add_pass("Fanout_Producer", |builder| {
            let output = builder.create_texture("Fanout_Output", dummy_desc());
            (MockExec, output)
        });
        for _ in 0..12 {
            graph.add_pass("Fanout_Consumer", |builder| {
                builder.read_texture(output);
                builder.mark_side_effect();
                (MockExec, ())
            });
        }

        graph.compile_topology();

        let expected_consumers: Vec<_> = (1..=12).collect();
        assert_eq!(
            graph.storage.compiled_reverse_consumers(0),
            expected_consumers
        );
        assert_eq!(graph.storage.execution_queue, (0..=12).collect::<Vec<_>>());
    }

    #[test]
    fn csr_contains_only_live_consumer_edges() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let dead_output = graph.add_pass("Dead_Producer", |builder| {
            let output = builder.create_texture("Dead_Output", dummy_desc());
            (MockExec, output)
        });
        graph.add_pass("Dead_Consumer", |builder| {
            builder.read_texture(dead_output);
            (MockExec, ())
        });
        let live_output = graph.add_pass("Live_Producer", |builder| {
            let output = builder.create_texture("Live_Output", dummy_desc());
            (MockExec, output)
        });
        graph.add_pass("Live_Consumer", |builder| {
            builder.read_texture(live_output);
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.storage.pass_physical_dependencies(1), &[0]);
        assert_eq!(graph.storage.pass_physical_dependencies(3), &[2]);
        assert!(graph.storage.compiled_reverse_consumers(0).is_empty());
        assert_eq!(graph.storage.compiled_reverse_consumers(2), &[3]);
        assert_eq!(graph.storage.execution_queue, [2, 3]);
    }

    #[test]
    #[should_panic(expected = "demanded a draw plan from another graph attempt")]
    fn pass_builder_rejects_a_foreign_draw_plan_id() {
        let mut foreign = DrawPlanStore::new();
        let foreign_id = foreign.reserve_plan(DrawPlanKey::Opaque);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        graph.add_pass("Foreign_Draw_Consumer", |builder| {
            builder.demand_draw_plan(foreign_id);
            (MockExec, ())
        });
    }

    #[test]
    #[should_panic(expected = "render graph topology is frozen after compilation")]
    fn topology_compilation_is_a_one_shot_phase() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        graph.compile_topology();
        graph.compile_topology();
    }

    #[test]
    #[should_panic(expected = "is a read-only lazy external input")]
    fn asset_texture_import_root_rejects_a_writer() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());

        graph.add_pass("Invalid_Asset_Writer", |builder| {
            builder.write_texture(asset);
            (MockExec, ())
        });
    }

    #[test]
    fn dead_lazy_asset_import_does_not_root_the_graph() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());

        graph.compile_topology();

        assert!(graph.storage.execution_queue.is_empty());
        assert_eq!(
            graph.storage.resources[asset.index() as usize].first_use,
            usize::MAX
        );
    }

    #[test]
    fn live_lazy_asset_import_requires_no_proxy_producer() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
        graph.add_pass("Asset_Consumer", |builder| {
            builder.read_texture(asset);
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.storage.execution_queue, [0]);
        assert_eq!(graph.storage.resources[asset.index() as usize].first_use, 0);
    }

    #[test]
    #[should_panic(expected = "reads derived asset texture 'Asset_Mip_Ready' without a producer")]
    fn live_derived_asset_version_still_requires_a_producer() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
        let mip_ready = graph.add_pass("Broken_Mipmap", |builder| {
            let output = builder.mutate_texture(asset, "Asset_Mip_Ready");
            (MockExec, output)
        });
        graph.storage.resources[mip_ready.index() as usize].producer = None;
        graph.add_pass("Asset_Consumer", |builder| {
            builder.read_texture(mip_ready);
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();
    }

    #[test]
    #[should_panic(expected = "must be registered before live consumer")]
    fn derived_asset_producer_must_precede_its_consumer() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
        let mip_ready = graph.create_alias(asset, "Asset_Mip_Ready");
        graph.add_pass("Asset_Consumer", |builder| {
            builder.read_texture(mip_ready);
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph.add_pass("Late_Mipmap", |builder| {
            builder.read_texture(asset);
            builder.write_texture(mip_ready);
            (MockExec, ())
        });

        graph.compile_topology();
    }

    #[test]
    #[should_panic(expected = "must read the lazy import chain it mutates")]
    fn derived_asset_producer_cannot_replace_without_demanding_its_import() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
        let replaced = graph.add_pass("Invalid_Replacement", |builder| {
            let output = builder.replace_texture(asset, "Asset_Replaced");
            (MockExec, output)
        });
        graph.add_pass("Asset_Consumer", |builder| {
            builder.read_texture(replaced);
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();
    }

    #[test]
    #[should_panic(expected = "must read the lazy import chain it mutates")]
    fn side_effect_asset_replacement_cannot_bypass_lazy_import_demand() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
        graph.add_pass("Invalid_Side_Effect_Replacement", |builder| {
            let _output = builder.replace_texture(asset, "Asset_Replaced");
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();
    }

    #[test]
    fn high_cardinality_lazy_asset_inputs_add_no_proxy_passes_or_dependencies() {
        const TEXTURE_COUNT: usize = 1_024;

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let mut imports = Vec::with_capacity(TEXTURE_COUNT);
        for index in 0..TEXTURE_COUNT {
            let mut snapshot = dummy_asset_snapshot();
            snapshot.texture_identity = 1_000 + index as u64;
            snapshot.logical_version = 0xA55E_8000 + index as u64;
            imports.push(graph.import_asset_texture("Large_Asset_Set", snapshot));
        }
        graph.add_pass("Large_Asset_Consumer", |builder| {
            for &import in &imports {
                builder.read_texture(import);
            }
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.storage.passes.len(), 1);
        assert!(graph.storage.pass_physical_dependencies(0).is_empty());
        assert_eq!(graph.storage.execution_queue, [0]);
        let compact = graph.dump_mermaid();
        assert!(compact.contains("declared 1024 · live 1024"));
        assert!(compact.contains("1024 textures"));
        assert!(
            compact.len() < 4_096,
            "Compact dump size must not scale with individual lazy imports"
        );
    }

    #[test]
    fn dropping_graph_releases_unsubmitted_frame_owners_but_keeps_topology() {
        let mut storage = GraphStorage::new();
        let prepared_attempt;
        let mut arena = FrameArena::new();
        {
            let mut graph = begin_test_frame(&mut storage, &arena);
            prepared_attempt = graph.storage.prepared_bind_groups.attempt_token();
            let _asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
            graph
                .storage
                .mipmap_batches
                .push(Some(MipmapRequestBatch::new(Vec::new())));
            graph.add_pass("Inspectable", |_builder| (MockExec, ()));

            assert!(!graph.storage.asset_texture_imports.is_empty());
            assert_eq!(graph.storage.mipmap_batches.len(), 1);
        }

        assert!(storage.asset_texture_imports.is_empty());
        assert!(storage.mipmap_batches.is_empty());
        assert_ne!(
            storage.prepared_bind_groups.attempt_token(),
            prepared_attempt
        );
        assert_eq!(storage.passes.len(), 1);
        assert_eq!(storage.passes[0].name, "Inspectable");
        assert!(storage.passes[0].node.is_none());
        arena.reset();
        assert!(storage.passes[0].node.is_none());
    }

    #[test]
    fn live_asset_consumer_marks_only_the_lazy_import_live() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
        let repeated = graph.import_asset_texture("AssetAgain", dummy_asset_snapshot());
        assert_eq!(asset, repeated);
        graph.add_pass("Asset_Consumer", |builder| {
            builder.read_texture(asset);
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();

        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(live, ["Asset_Consumer"]);
        assert_eq!(graph.storage.resources[asset.index() as usize].first_use, 0);
    }

    #[test]
    fn asset_texture_batch_materializes_only_dpe_live_imports() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resource_manager = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let live_image = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![11; 8 * 8 * 4]),
        ));
        let dead_image = assets.images.add(Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![22; 8 * 8 * 4]),
        ));
        let mut live_texture = Texture::new_2d(Some("live graph asset"), live_image);
        live_texture.generate_mipmaps = true;
        let live_handle = assets.textures.add(live_texture);
        let dead_handle = assets
            .textures
            .add(Texture::new_2d(Some("dead graph asset"), dead_image));
        let live_snapshot =
            snapshot_asset_texture(&assets, live_handle).expect("live snapshot missing");
        let dead_snapshot =
            snapshot_asset_texture(&assets, dead_handle).expect("dead snapshot missing");

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        {
            let mut graph = begin_test_frame(&mut storage, &arena);
            let live = graph.import_asset_texture("Live_Asset", live_snapshot);
            let dead = graph.import_asset_texture("Dead_Asset", dead_snapshot);
            graph.add_pass("Live_Asset_Consumer", |builder| {
                builder.read_texture(live);
                builder.mark_side_effect();
                (MockExec, ())
            });

            graph.compile_topology();
            graph.materialize_live_asset_textures(&assets, &mut resource_manager);

            assert!(resource_manager.get_texture_binding(live_handle).is_some());
            assert!(resource_manager.get_texture_binding(dead_handle).is_none());
            assert!(resource_manager.take_mipmap_requests().is_empty());

            let live_id = graph.storage.resources[live.index() as usize]
                .asset_texture_import_id()
                .unwrap();
            let dead_id = graph.storage.resources[dead.index() as usize]
                .asset_texture_import_id()
                .unwrap();
            let resolver = graph.storage.asset_texture_imports.resolver();
            assert!(resolver.materialization(live_id).is_some());
            assert!(resolver.materialization(dead_id).is_none());
            assert_eq!(resolver.mipmap_requests(live_id).count(), 1);
        }
        assert!(storage.asset_texture_imports.is_empty());
        assert!(
            !resource_manager
                .get_texture_image(live_handle)
                .expect("aborted live base upload should remain safely cached")
                .mipmaps_submitted(),
            "dropping an unsubmitted graph must not publish mip completion"
        );
        assert!(
            !resource_manager.asset_snapshot_mipmaps_submitted(live_snapshot),
            "the next graph must retry the unsubmitted logical mip producer"
        );
    }

    #[test]
    fn mip_ready_consumer_retains_only_the_real_mipmap_producer() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
        let mip_ready = graph.add_pass("Asset_Mipmap", |builder| {
            let output = builder.mutate_texture(asset, "Asset_Mip_Ready");
            (MockExec, output)
        });
        graph.add_pass("Asset_Consumer", |builder| {
            builder.read_texture(mip_ready);
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();

        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(live, ["Asset_Mipmap", "Asset_Consumer"]);
        assert_eq!(
            graph.storage.resources[asset.index() as usize].first_use,
            0,
            "the live mip producer must demand materialization of its lazy root"
        );
    }

    /// Helper: create a fresh RenderGraph for a test frame.
    fn begin_test_frame<'a>(
        storage: &'a mut GraphStorage,
        arena: &'a FrameArena,
    ) -> RenderGraph<'a> {
        RenderGraph::new(storage, arena)
    }

    #[test]
    fn unencoded_mipmap_batch_is_released_without_publication() {
        let mut storage = GraphStorage::new();
        let batch = MipmapBatchId::from_index(0);
        storage
            .mipmap_batches
            .push(Some(MipmapRequestBatch::new(Vec::new())));

        assert!(!storage.commit_submitted_mipmap_batch(batch));
        assert!(storage.mipmap_batches[batch.index()].is_none());
    }

    #[test]
    fn encoded_mipmap_batch_is_publishable_and_released() {
        let mut storage = GraphStorage::new();
        let batch = MipmapBatchId::from_index(0);
        let attempt = MipmapRequestBatch::new(Vec::new());
        attempt.mark_encoded();
        storage.mipmap_batches.push(Some(attempt));

        assert!(storage.commit_submitted_mipmap_batch(batch));
        assert!(storage.mipmap_batches[batch.index()].is_none());
    }

    #[test]
    fn test_zero_alloc_graph() {
        let mut storage = GraphStorage::new();
        let mut arena = FrameArena::new();

        // Run two frames to verify capacity reuse.
        for frame in 0..2 {
            arena.reset();
            let mut graph = begin_test_frame(&mut storage, &arena);

            let backbuffer = graph.register_texture("Backbuffer", dummy_desc(), true);

            let scene_color = graph.add_pass("Opaque", |builder| {
                let out = builder.create_texture("SceneColor", dummy_desc());
                (MockExec, out)
            });

            let bloom_tex = graph.add_pass("Bloom", |builder| {
                builder.read_texture(scene_color);
                let out = builder.create_texture("BloomTex", dummy_desc());
                (MockExec, out)
            });

            graph.add_pass("ToneMapping", |builder| {
                builder.read_texture(scene_color);
                builder.read_texture(bloom_tex);
                builder.write_texture(backbuffer);
                (MockExec, ())
            });

            graph.compile_topology();

            assert_eq!(graph.storage.execution_queue.len(), 3);
            assert_eq!(
                graph.storage.passes[graph.storage.execution_queue[0]].name,
                "Opaque"
            );
            assert_eq!(
                graph.storage.passes[graph.storage.execution_queue[1]].name,
                "Bloom"
            );
            assert_eq!(
                graph.storage.passes[graph.storage.execution_queue[2]].name,
                "ToneMapping"
            );

            println!(
                "Frame {} executed: {:?}",
                frame,
                graph
                    .storage
                    .execution_queue
                    .iter()
                    .map(|&i| graph.storage.passes[i].name)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn side_effect_passes_keep_registration_order_when_unconstrained() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        graph.add_pass("First", |builder| {
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph.add_pass("Second", |builder| {
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph.add_pass("Third", |builder| {
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.compile_topology();

        let queue_names: Vec<&str> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&i| graph.storage.passes[i].name)
            .collect();

        assert_eq!(queue_names, ["First", "Second", "Third"]);
    }

    #[test]
    fn test_dead_resource_culling() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_texture("Backbuffer", dummy_desc(), true);

        let (color, motion) = graph.add_pass("GBuffer", |builder| {
            let color = builder.create_texture("Color", dummy_desc());
            let motion = builder.create_texture("MotionVec", dummy_desc());
            (MockExec, (color, motion))
        });

        graph.add_pass("ToneMap", |builder| {
            builder.read_texture(color);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.storage.execution_queue.len(), 2);
        assert_ne!(
            graph.storage.resources[color.index() as usize].consumer_edge_count(),
            0
        );
        assert_eq!(
            graph.storage.resources[motion.index() as usize].consumer_edge_count(),
            0
        );
        assert_ne!(
            graph.storage.resources[motion.index() as usize].first_use,
            usize::MAX
        );
    }

    #[test]
    fn test_fold_pure_forwarding_present() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        // External surface (swap-chain stand-in).
        let surface = graph.register_texture("Surface", dummy_desc(), true);

        // Producer renders into a transient intermediate of identical layout.
        let intermediate = graph.add_pass("Producer", |builder| {
            let out = builder.create_texture("LDR_Intermediate", dummy_desc());
            (MockExec, out)
        });

        // Unified present: forward the intermediate onto the surface.
        let present_out = graph.add_pass("Present", |builder| {
            builder.read_texture(intermediate);
            let out = builder.replace_texture(surface, "Surface_Present");
            builder.mark_pure_forwarding();
            (MockExec, out)
        });

        assert_eq!(
            graph.storage.resources[intermediate.index() as usize].consumer_edge_count(),
            1,
            "the fold gate must observe the canonical logical-read edge"
        );

        graph.compile_topology();

        assert!(graph.storage.pass_reads(1).is_empty());
        assert_eq!(
            graph.storage.passes[1].logical_read_range.start,
            graph.storage.passes[1].logical_read_range.end,
            "folding must clear the pass view without shifting flat storage"
        );
        assert_eq!(
            graph.storage.resources[intermediate.index() as usize].consumer_edge_count(),
            0,
            "folding must remove the eliminated pass's consumer edge"
        );
        assert!(graph.storage.pass_physical_dependencies(1).is_empty());
        assert!(graph.storage.compiled_reverse_consumers(0).is_empty());

        let queue_names: Vec<&str> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&i| graph.storage.passes[i].name)
            .collect();

        // The present pass is edge-contracted into the producer and removed.
        assert!(
            !queue_names.contains(&"Present"),
            "Present pass should be folded away, got {queue_names:?}"
        );
        assert!(
            queue_names.contains(&"Producer"),
            "Producer must survive as the surface writer, got {queue_names:?}"
        );

        // The intermediate now aliases the surface so the producer's node
        // resolves to the swap-chain view (and stores, not discards) at execute.
        let inter_res = &graph.storage.resources[intermediate.index() as usize];
        assert_eq!(
            inter_res.alias_of,
            Some(surface.erase()),
            "intermediate must alias the surface root after folding"
        );
        assert!(
            inter_res.is_external,
            "intermediate must inherit the surface's external status"
        );

        let dump = graph.dump_mermaid_full();
        assert!(dump.contains(&format!("P0 --> OUT_{};", present_out.index())));
        assert!(
            !dump.contains(&format!("P0 --> OUT_{};", intermediate.index())),
            "folded source aliases must not invent an additional terminal edge"
        );
        assert!(
            !dump.contains(&format!("OUT_{}[", intermediate.index())),
            "folded source aliases must not invent an orphan terminal node"
        );

        // The present output (a surface alias) is now owned by the producer, so
        // any downstream reader depends on the producer rather than the
        // eliminated present pass.
        let producer_idx = graph
            .storage
            .execution_queue
            .iter()
            .copied()
            .find(|&i| graph.storage.passes[i].name == "Producer")
            .expect("Producer in execution queue");
        assert_eq!(
            graph.storage.resources[present_out.index() as usize].producer,
            Some(producer_idx),
            "the surface alias must be owned by the producer after rerouting"
        );

        let diagnostics = graph.diagnostics();
        assert_eq!(diagnostics.declared_passes(), 2);
        assert_eq!(diagnostics.folded_passes(), 1);
        assert_eq!(diagnostics.live_passes(), Some(1));
    }

    #[test]
    fn folding_middle_pass_does_not_shift_later_flat_ranges() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let surface = graph.register_texture("Surface", dummy_desc(), true);
        let later_input = graph.register_texture("Later_Input", dummy_desc(), false);
        let intermediate = graph.add_pass("Producer", |builder| {
            let output = builder.create_texture("Intermediate", dummy_desc());
            (MockExec, output)
        });
        graph.add_pass("Present", |builder| {
            builder.read_texture(intermediate);
            let output = builder.replace_texture(surface, "Surface_Present");
            builder.mark_pure_forwarding();
            (MockExec, output)
        });
        let later_index = graph.storage.passes.len();
        graph.add_pass("Later", |builder| {
            builder.read_texture(later_input);
            builder.mark_side_effect();
            (MockExec, ())
        });
        let later_range = graph.storage.passes[later_index].logical_read_range.clone();
        let later_reads = graph.storage.pass_reads(later_index).to_vec();

        graph.compile_topology();

        assert!(graph.storage.pass_reads(1).is_empty());
        assert_eq!(
            graph.storage.passes[later_index].logical_read_range,
            later_range
        );
        assert_eq!(graph.storage.pass_reads(later_index), later_reads);
        assert!(graph.storage.pass_physical_dependencies(1).is_empty());
        assert!(graph.storage.compiled_reverse_consumers(0).is_empty());
        assert_eq!(
            graph
                .storage
                .resource_consumers(later_input.index() as usize)
                .collect::<Vec<_>>(),
            &[later_index]
        );
    }

    #[test]
    fn folded_count_survives_a_later_compile_failure_without_inventing_live_count() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut graph = begin_test_frame(&mut storage, &arena);
            let surface = graph.register_texture("Surface", dummy_desc(), true);
            let intermediate = graph.add_pass("Producer", |builder| {
                let output = builder.create_texture("Intermediate", dummy_desc());
                (MockExec, output)
            });
            graph.add_pass("Present", |builder| {
                builder.read_texture(intermediate);
                let output = builder.replace_texture(surface, "Surface_Present");
                builder.mark_pure_forwarding();
                (MockExec, output)
            });

            let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
            let mip_ready = graph.add_pass("Broken_Mipmap", |builder| {
                let output = builder.mutate_texture(asset, "Asset_Mip_Ready");
                (MockExec, output)
            });
            graph.storage.resources[mip_ready.index() as usize].producer = None;
            graph.add_pass("Asset_Consumer", |builder| {
                builder.read_texture(mip_ready);
                builder.mark_side_effect();
                (MockExec, ())
            });

            graph.compile_topology();
        }));
        assert!(result.is_err());

        let diagnostics = storage.latest_diagnostics().unwrap();
        assert_eq!(diagnostics.declared_passes(), 4);
        assert_eq!(diagnostics.folded_passes(), 1);
        assert_eq!(diagnostics.live_passes(), None);
        assert_eq!(diagnostics.phase(), RdgFramePhase::CompileGraph);
        assert!(diagnostics.panicked());
    }

    #[test]
    fn test_fold_skipped_with_extra_consumer() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let surface = graph.register_texture("Surface", dummy_desc(), true);

        let intermediate = graph.add_pass("Producer", |builder| {
            let out = builder.create_texture("LDR_Intermediate", dummy_desc());
            (MockExec, out)
        });

        // A second consumer of the intermediate must keep it materialised, so
        // the forwarding pass cannot be folded.
        graph.add_pass("Histogram", |builder| {
            builder.read_texture(intermediate);
            builder.mark_side_effect();
            (MockExec, ())
        });

        graph.add_pass("Present", |builder| {
            builder.read_texture(intermediate);
            let out = builder.replace_texture(surface, "Surface_Present");
            builder.mark_pure_forwarding();
            (MockExec, out)
        });

        graph.compile_topology();

        let queue_names: Vec<&str> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&i| graph.storage.passes[i].name)
            .collect();

        assert!(
            queue_names.contains(&"Present"),
            "Present must survive when the source has another consumer, got {queue_names:?}"
        );
        assert!(
            graph.storage.resources[intermediate.index() as usize]
                .alias_of
                .is_none(),
            "intermediate must not be aliased onto the surface when fold is skipped"
        );
    }

    #[test]
    fn test_self_read_prevents_culling() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_texture("Backbuffer", dummy_desc(), true);

        let internal = graph.add_pass("MacroNode", |builder| {
            let internal = builder.create_texture("Internal", dummy_desc());
            builder.read_texture(internal);
            builder.write_texture(backbuffer);
            (MockExec, internal)
        });

        graph.compile_topology();
        assert_ne!(
            graph.storage.resources[internal.index() as usize].consumer_edge_count(),
            0
        );
    }

    #[test]
    fn test_resource_lifetime_deduction() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_texture("Backbuffer", dummy_desc(), true);

        let color = graph.add_pass("Opaque", |builder| {
            let out = builder.create_texture("Color", dummy_desc());
            (MockExec, out)
        });

        let bloom = graph.add_pass("Bloom", |builder| {
            builder.read_texture(color);
            let out = builder.create_texture("Bloom", dummy_desc());
            (MockExec, out)
        });

        graph.add_pass("ToneMapping", |builder| {
            builder.read_texture(color);
            builder.read_texture(bloom);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        let color_res = &graph.storage.resources[color.index() as usize];
        assert_eq!(
            color_res.first_use, 0,
            "Color first written by Opaque at timeline 0"
        );
        assert_eq!(
            color_res.last_use, 2,
            "Color last read by ToneMapping at timeline 2"
        );

        let bloom_res = &graph.storage.resources[bloom.index() as usize];
        assert_eq!(
            bloom_res.first_use, 1,
            "Bloom first written by BloomPass at timeline 1"
        );
        assert_eq!(
            bloom_res.last_use, 2,
            "Bloom last read by ToneMapping at timeline 2"
        );
    }

    #[test]
    fn test_ssa_alias_relay_passes() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_texture("Backbuffer", dummy_desc(), true);

        let color_v0 = graph.add_pass("Opaque", |builder| {
            let out = builder.create_texture("SceneColor_v0", dummy_desc());
            (MockExec, out)
        });

        assert!(
            graph.storage.resources[color_v0.index() as usize]
                .alias_of
                .is_none(),
            "v0 is a root resource"
        );

        let color_v1 = graph.add_pass("Skybox", |builder| {
            let out = builder.mutate_texture(color_v0, "SceneColor_v1");
            (MockExec, out)
        });

        assert_eq!(
            graph.storage.resources[color_v1.index() as usize].alias_of,
            Some(color_v0.erase()),
            "v1 aliases v0"
        );

        let color_v2 = graph.add_pass("Transparent", |builder| {
            let out = builder.mutate_texture(color_v1, "SceneColor_v2");
            (MockExec, out)
        });

        assert_eq!(
            graph.storage.resources[color_v2.index() as usize].alias_of,
            Some(color_v0.erase()),
            "v2 aliases v0 (root)"
        );

        graph.add_pass("ToneMap", |builder| {
            builder.read_texture(color_v2);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.storage.execution_queue.len(), 4);
        let names: Vec<&str> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&i| graph.storage.passes[i].name)
            .collect();
        assert_eq!(names, vec!["Opaque", "Skybox", "Transparent", "ToneMap"]);

        assert_eq!(
            graph.resolve_alias_root(color_v2.index() as usize),
            color_v0.index() as usize
        );
        assert_eq!(
            graph.resolve_alias_root(color_v1.index() as usize),
            color_v0.index() as usize
        );
        assert_eq!(
            graph.resolve_alias_root(color_v0.index() as usize),
            color_v0.index() as usize
        );
    }

    #[test]
    fn test_mutate_texture_api() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_texture("Backbuffer", dummy_desc(), true);

        let color = graph.add_pass("Writer", |builder| {
            let out = builder.create_texture("Color", dummy_desc());
            (MockExec, out)
        });

        let mutated_id = graph.add_pass("Mutator", |builder| {
            let out = builder.mutate_texture(color, "Color_Mutated");
            (MockExec, out)
        });

        assert!(
            graph.storage.resources[mutated_id.index() as usize]
                .alias_of
                .is_some()
        );

        graph.add_pass("Reader", |builder| {
            builder.read_texture(mutated_id);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.storage.execution_queue.len(), 3);
        let names: Vec<&str> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&i| graph.storage.passes[i].name)
            .collect();
        assert_eq!(names, vec!["Writer", "Mutator", "Reader"]);
    }

    #[test]
    fn after_submit_hooks_run_once_for_live_passes_only() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct SubmitProbe<'a>(&'a AtomicUsize);

        impl<'a> PassNode<'a> for SubmitProbe<'a> {
            fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}

            fn after_submit(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let live_submissions = AtomicUsize::new(0);
        let dead_submissions = AtomicUsize::new(0);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        graph.add_pass("Dead_Submit_Probe", |_builder| {
            (SubmitProbe(&dead_submissions), ())
        });
        graph.add_pass("Live_Submit_Probe", |builder| {
            builder.mark_side_effect();
            (SubmitProbe(&live_submissions), ())
        });

        graph.compile_topology();
        graph.run_after_submit_hooks();

        assert_eq!(live_submissions.load(Ordering::Relaxed), 1);
        assert_eq!(dead_submissions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_buffer_resource_lifetime_deduction() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_texture("Backbuffer", dummy_desc(), true);

        let visible_instances = graph.add_pass("Cull", |builder| {
            let out = builder.create_buffer(
                "VisibleInstances",
                BufferDesc::new(4097, wgpu::BufferUsages::STORAGE),
            );
            (MockExec, out)
        });

        let draw_indirect = graph.add_pass("BuildIndirect", |builder| {
            builder.read_buffer(visible_instances);
            let out = builder.create_buffer(
                "DrawIndirect",
                BufferDesc::new(16, wgpu::BufferUsages::STORAGE),
            );
            (MockExec, out)
        });

        graph.add_pass("Draw", |builder| {
            builder.read_buffer(visible_instances);
            builder.read_buffer(draw_indirect);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        let visible_res = &graph.storage.resources[visible_instances.index() as usize];
        assert_eq!(visible_res.class(), ResourceClass::Buffer);
        assert_eq!(visible_res.buffer_desc().logical_size, 4097);
        assert_eq!(visible_res.first_use, 0);
        assert_eq!(visible_res.last_use, 2);

        let indirect_res = &graph.storage.resources[draw_indirect.index() as usize];
        assert_eq!(indirect_res.class(), ResourceClass::Buffer);
        assert_eq!(indirect_res.first_use, 1);
        assert_eq!(indirect_res.last_use, 2);
    }

    #[test]
    fn test_with_group_preserves_topology() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let frame_config = dummy_config();

        let backbuffer = graph.register_texture("Backbuffer", dummy_desc(), true);

        let mut ctx = GraphBuilderContext::new(&mut graph, &frame_config, 0);

        let scene_color = ctx.with_group("Scene", |ctx| {
            let opaque_out = ctx.graph.add_pass("Opaque", |builder| {
                let out = builder.create_texture("SceneColor", dummy_desc());
                (MockExec, out)
            });

            let skybox_out = ctx.graph.add_pass("Skybox", |builder| {
                let out = builder.mutate_texture(opaque_out, "SceneColor_Sky");
                (MockExec, out)
            });

            skybox_out
        });

        ctx.with_group("PostProcess", |ctx| {
            ctx.graph.add_pass("ToneMap", |builder| {
                builder.read_texture(scene_color);
                builder.write_texture(backbuffer);
                (MockExec, ())
            });
        });

        graph.compile_topology();

        let names: Vec<&str> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&i| graph.storage.passes[i].name)
            .collect();
        assert_eq!(names, vec!["Opaque", "Skybox", "ToneMap"]);
    }

    #[test]
    fn persistent_imports_deduplicate_without_becoming_dpe_roots() {
        let mut histories = HistoryRegistry::new();
        let set = histories.create_set();
        let plan = histories.plan_set(set, [dummy_desc()], 1, false);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let first = graph.import_persistent_texture("History", plan.next[0]);
        let repeated = graph.import_persistent_texture("HistoryAgain", plan.next[0]);
        assert_eq!(first, repeated);

        graph.add_pass("HistoryOnly", |builder| {
            builder.write_texture(first);
            (MockExec, ())
        });
        graph.register_history_commit(plan.commit, &[first]);
        graph.compile_topology();

        assert!(graph.storage.execution_queue.is_empty());
        graph.commit_submitted_histories(&mut histories);
        assert!(!histories.plan_set(set, [dummy_desc()], 2, false).valid);
    }

    #[test]
    fn demanded_persistent_writer_commits_as_one_history_bank() {
        let mut histories = HistoryRegistry::new();
        let set = histories.create_set();
        let plan = histories.plan_set(set, [dummy_desc(), dummy_desc()], 1, false);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let history_a = graph.import_persistent_texture("HistoryA", plan.next[0]);
        let history_b = graph.import_persistent_texture("HistoryB", plan.next[1]);
        let output = graph.register_texture("Output", dummy_desc(), true);

        graph.add_pass("Temporal", |builder| {
            builder.write_texture(history_a);
            builder.write_texture(history_b);
            (MockExec, ())
        });
        graph.add_pass("Consume", |builder| {
            builder.read_texture(history_a);
            builder.write_texture(output);
            (MockExec, ())
        });
        graph.register_history_commit(plan.commit, &[history_a, history_b]);
        graph.compile_topology();

        assert_eq!(graph.storage.execution_queue.len(), 2);
        assert!(
            graph
                .commit_submitted_persistent_resources(&mut histories)
                .is_empty(),
            "history publication must not manufacture cache receipts"
        );
        let next = histories.plan_set(set, [dummy_desc(), dummy_desc()], 2, false);
        assert!(next.valid);
        assert_eq!(next.previous, plan.next);
    }

    #[test]
    fn cache_publication_registration_does_not_seed_dpe_liveness() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let plan = registry.plan_cache(
            cache,
            CacheKey::new(1),
            [PersistentTextureSpec::default_view(dummy_desc())],
            1,
        );

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let texture = graph.import_persistent_texture("Cache", plan.textures[0]);
        graph.add_pass("CacheOnly", |builder| {
            builder.write_texture(texture);
            (MockExec, ())
        });
        graph.register_cache_commit(
            plan.commit.expect("invalid cache needs a token"),
            &[texture],
        );
        graph.compile_topology();

        assert!(graph.storage.execution_queue.is_empty());
        assert!(
            graph
                .commit_submitted_persistent_resources(&mut registry)
                .is_empty()
        );
        assert!(
            !registry
                .plan_cache(
                    cache,
                    CacheKey::new(1),
                    [PersistentTextureSpec::default_view(dummy_desc())],
                    2,
                )
                .valid
        );
    }

    #[test]
    fn dead_cache_replacement_restores_the_last_publication() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let published_key = CacheKey::new(1);
        let replacement_key = CacheKey::new(2);
        let spec = PersistentTextureSpec::default_view(dummy_desc());
        let published = registry.plan_cache(cache, published_key, [spec], 1);
        assert!(
            registry
                .commit_cache(
                    published.commit.expect("initial cache needs a token"),
                    &[published.textures[0].id],
                )
                .is_some()
        );
        let replacement = registry.plan_cache(cache, replacement_key, [spec], 2);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let texture = graph.import_persistent_texture("Replacement", replacement.textures[0]);
        graph.add_pass("DeadReplacement", |builder| {
            builder.write_texture(texture);
            (MockExec, ())
        });
        graph.register_cache_commit(
            replacement.commit.expect("replacement cache needs a token"),
            &[texture],
        );
        graph.compile_topology();

        assert!(graph.storage.execution_queue.is_empty());
        // Intentionally exercise the all-untouched post-submit compatibility
        // path: it may finalize and restore because no cache member was live.
        assert!(
            graph
                .commit_submitted_persistent_resources(&mut registry)
                .is_empty()
        );
        assert!(registry.is_cache_valid(cache, published_key));
        let restored = registry.plan_cache(cache, published_key, [spec], 3);
        assert!(restored.valid);
        assert_eq!(restored.textures, published.textures);
        assert!(
            registry
                .commit_cache(
                    replacement.commit.expect("replacement cache needs a token"),
                    &[replacement.textures[0].id],
                )
                .is_none(),
            "the rolled-back replacement token must remain stale"
        );
    }

    #[test]
    fn partial_cache_replacement_cannot_be_rolled_back_by_a_later_dead_attempt() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let published_key = CacheKey::new(1);
        let replacement_key = CacheKey::new(2);
        let spec = PersistentTextureSpec::default_view(dummy_desc());
        let published = registry.plan_cache(cache, published_key, [spec, spec], 1);
        assert!(
            registry
                .commit_cache(
                    published.commit.expect("initial cache needs a token"),
                    &published.textures.map(|texture| texture.id),
                )
                .is_some()
        );

        let replacement = registry.plan_cache(cache, replacement_key, [spec, spec], 2);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        {
            let mut graph = begin_test_frame(&mut storage, &arena);
            let replacement_a =
                graph.import_persistent_texture("ReplacementA", replacement.textures[0]);
            let replacement_b =
                graph.import_persistent_texture("ReplacementB", replacement.textures[1]);
            let output = graph.register_texture("Output", dummy_desc(), true);
            graph.add_pass("LiveReplacementWriter", |builder| {
                builder.write_texture(replacement_a);
                (MockExec, ())
            });
            graph.add_pass("DeadReplacementWriter", |builder| {
                builder.write_texture(replacement_b);
                (MockExec, ())
            });
            graph.add_pass("ConsumeReplacement", |builder| {
                builder.read_texture(replacement_a);
                builder.write_texture(output);
                (MockExec, ())
            });
            graph.register_cache_commit(
                replacement.commit.expect("replacement needs a token"),
                &[replacement_a, replacement_b],
            );
            graph.compile_topology();
            graph.finalize_persistent_cache_attempts_after_dpe(&mut registry);
            assert!(
                graph
                    .commit_submitted_persistent_resources(&mut registry)
                    .is_empty()
            );
        }

        let dead_attempt = registry.plan_cache(cache, replacement_key, [spec, spec], 3);
        {
            let mut graph = begin_test_frame(&mut storage, &arena);
            let replacement_a =
                graph.import_persistent_texture("ReplacementA", dead_attempt.textures[0]);
            let replacement_b =
                graph.import_persistent_texture("ReplacementB", dead_attempt.textures[1]);
            graph.add_pass("DeadReplacementWriter", |builder| {
                builder.write_texture(replacement_a);
                builder.write_texture(replacement_b);
                (MockExec, ())
            });
            graph.register_cache_commit(
                dead_attempt.commit.expect("invalid cache needs a token"),
                &[replacement_a, replacement_b],
            );
            graph.compile_topology();
            graph.finalize_persistent_cache_attempts_after_dpe(&mut registry);
            assert!(graph.storage.execution_queue.is_empty());
            assert!(
                graph
                    .commit_submitted_persistent_resources(&mut registry)
                    .is_empty()
            );
        }

        assert!(!registry.is_cache_valid(cache, published_key));
        assert!(
            !registry
                .plan_cache(cache, replacement_key, [spec, spec], 4)
                .valid
        );
    }

    #[test]
    fn live_reader_of_an_invalid_replacement_forfeits_rollback() {
        let context = pollster::block_on(crate::core::WgpuContext::new_headless(
            &crate::settings::RendererInitConfig::default(),
            &crate::settings::RendererSettings::default(),
            2,
            2,
            None,
        ))
        .expect("headless context init failed");

        for register_publication in [false, true] {
            let mut registry = HistoryRegistry::new();
            let cache = registry.create_cache();
            let published_key = CacheKey::new(1);
            let replacement_key = CacheKey::new(2);
            let published_spec = PersistentTextureSpec::default_view(dummy_desc());
            let published = registry.plan_cache(cache, published_key, [published_spec], 1);
            registry.materialize_texture(
                &context.device,
                "Published",
                published.textures[0].id,
                published_spec,
            );
            let published_view = registry.get_texture_view(published.textures[0].id).id();
            assert!(
                registry
                    .commit_cache(
                        published.commit.expect("initial cache needs a token"),
                        &[published.textures[0].id],
                    )
                    .is_some()
            );

            let replacement_spec = PersistentTextureSpec::default_view(TextureDesc::new_2d(
                2,
                2,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            ));
            let replacement = registry.plan_cache(cache, replacement_key, [replacement_spec], 2);
            let mut storage = GraphStorage::new();
            let arena = FrameArena::new();
            {
                let mut graph = begin_test_frame(&mut storage, &arena);
                let texture =
                    graph.import_persistent_texture("Replacement", replacement.textures[0]);
                let output = graph.register_texture("Output", dummy_desc(), true);
                graph.add_pass("ReadInvalidReplacement", |builder| {
                    builder.read_texture(texture);
                    builder.write_texture(output);
                    (MockExec, ())
                });
                if register_publication {
                    graph.register_cache_commit(
                        replacement.commit.expect("replacement needs a token"),
                        &[texture],
                    );
                }
                let premature_materialization =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        graph.materialize_live_persistent_textures(&mut registry, &context.device);
                    }));
                assert!(
                    premature_materialization.is_err(),
                    "uncompiled materialization must reject before restoring cache metadata or freezing topology"
                );
                assert!(!registry.is_cache_valid(cache, published_key));
                assert!(!registry.is_cache_valid(cache, replacement_key));
                graph.compile_topology();
                graph.materialize_live_persistent_textures(&mut registry, &context.device);
                graph.materialize_live_persistent_textures(&mut registry, &context.device);
                assert!(!registry.is_cache_valid(cache, published_key));
                assert!(!registry.is_cache_valid(cache, replacement_key));
                assert_ne!(
                    published_view,
                    registry.get_texture_view(replacement.textures[0].id).id(),
                    "the live reader replaced the original physical texture"
                );
                // Abandon this attempt before submit, with or without a registered
                // publication. Physical materialization has already happened.
            }

            let dead_attempt = registry.plan_cache(cache, replacement_key, [replacement_spec], 3);
            let mut graph = begin_test_frame(&mut storage, &arena);
            let texture =
                graph.import_persistent_texture("DeadReplacement", dead_attempt.textures[0]);
            graph.register_cache_commit(
                dead_attempt.commit.expect("replacement is still invalid"),
                &[texture],
            );
            graph.compile_topology();
            assert!(
                graph
                    .commit_submitted_persistent_resources(&mut registry)
                    .is_empty()
            );
            assert!(
                !registry.is_cache_valid(cache, published_key),
                "an abandoned live reader must not revive metadata for the destroyed texture (registered: {register_publication})"
            );
        }
    }

    #[test]
    #[should_panic(expected = "writer set does not match canonical members")]
    fn cache_finalization_rejects_a_writer_set_that_omits_a_live_member() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let published_key = CacheKey::new(1);
        let replacement_key = CacheKey::new(2);
        let spec = PersistentTextureSpec::default_view(dummy_desc());
        let published = registry.plan_cache(cache, published_key, [spec, spec], 1);
        assert!(
            registry
                .commit_cache(
                    published.commit.expect("initial cache needs a token"),
                    &published.textures.map(|texture| texture.id),
                )
                .is_some()
        );
        let replacement = registry.plan_cache(cache, replacement_key, [spec, spec], 2);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let reported_writer =
            graph.import_persistent_texture("ReportedWriter", replacement.textures[0]);
        let omitted_live_member =
            graph.import_persistent_texture("OmittedLiveMember", replacement.textures[1]);
        let output = graph.register_texture("Output", dummy_desc(), true);
        graph.add_pass("DeadReportedWriter", |builder| {
            builder.write_texture(reported_writer);
            (MockExec, ())
        });
        graph.add_pass("ReadOmittedMember", |builder| {
            builder.read_texture(omitted_live_member);
            builder.write_texture(output);
            (MockExec, ())
        });
        graph.register_cache_commit(
            replacement.commit.expect("replacement needs a token"),
            &[reported_writer],
        );
        graph.compile_topology();

        assert_eq!(
            graph.storage.resources[reported_writer.index() as usize].first_use,
            usize::MAX
        );
        assert_ne!(
            graph.storage.resources[omitted_live_member.index() as usize].first_use,
            usize::MAX
        );
        graph.finalize_persistent_cache_attempts_after_dpe(&mut registry);
    }

    #[test]
    #[should_panic(expected = "topology is frozen after persistent cache finalization")]
    fn topology_mutation_after_cache_finalization_is_rejected() {
        let mut registry = HistoryRegistry::new();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        graph.compile_topology();
        assert!(!graph.finalize_persistent_cache_attempts_after_dpe(&mut registry));

        graph.add_pass("TooLate", |_builder| (MockExec, ()));
    }

    #[test]
    #[should_panic(expected = "topology is frozen after persistent cache finalization")]
    fn cache_commit_registration_after_finalization_is_rejected() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let spec = PersistentTextureSpec::default_view(dummy_desc());
        let plan = registry.plan_cache(cache, CacheKey::new(1), [spec], 1);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let texture = graph.import_persistent_texture("Cache", plan.textures[0]);
        graph.compile_topology();
        assert!(!graph.finalize_persistent_cache_attempts_after_dpe(&mut registry));

        graph.register_cache_commit(
            plan.commit.expect("invalid cache needs a token"),
            &[texture],
        );
    }

    #[test]
    #[should_panic(
        expected = "live persistent cache members must be finalized before post-submit publication"
    )]
    fn post_submit_rejects_a_live_cache_that_skipped_pre_materialization_finalization() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let spec = PersistentTextureSpec::default_view(dummy_desc());
        let plan = registry.plan_cache(cache, CacheKey::new(1), [spec], 1);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let texture = graph.import_persistent_texture("Cache", plan.textures[0]);
        let output = graph.register_texture("Output", dummy_desc(), true);
        graph.add_pass("Populate", |builder| {
            builder.write_texture(texture);
            (MockExec, ())
        });
        graph.add_pass("Consume", |builder| {
            builder.read_texture(texture);
            builder.write_texture(output);
            (MockExec, ())
        });
        graph.register_cache_commit(
            plan.commit.expect("invalid cache needs a token"),
            &[texture],
        );
        graph.compile_topology();

        graph.commit_submitted_persistent_resources(&mut registry);
    }

    #[test]
    fn demanded_cache_writer_publishes_after_submit_with_typed_receipt() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let key = CacheKey::new(42);
        let specs = [
            PersistentTextureSpec::default_view(dummy_desc()),
            PersistentTextureSpec::default_view(dummy_desc()),
        ];
        let plan = registry.plan_cache(cache, key, specs, 1);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let cache_a = graph.import_persistent_texture("CacheA", plan.textures[0]);
        let cache_b = graph.import_persistent_texture("CacheB", plan.textures[1]);
        let output = graph.register_texture("Output", dummy_desc(), true);
        graph.add_pass("PopulateCache", |builder| {
            builder.write_texture(cache_a);
            builder.write_texture(cache_b);
            (MockExec, ())
        });
        graph.add_pass("ConsumeCache", |builder| {
            builder.read_texture(cache_a);
            builder.write_texture(output);
            (MockExec, ())
        });
        let token = plan.commit.expect("invalid cache needs a token");
        graph.register_cache_commit(token, &[cache_a, cache_b]);
        graph.register_cache_commit(token, &[cache_b, cache_a]);
        graph.compile_topology();
        graph.finalize_persistent_cache_attempts_after_dpe(&mut registry);

        let receipts = graph.commit_submitted_persistent_resources(&mut registry);
        assert_eq!(
            receipts.as_slice(),
            &[CachePublicationReceipt {
                cache_id: cache,
                key,
                epoch: plan.epoch,
                generation: token.generation,
                attempt_serial: 1,
            }]
        );
        assert!(registry.plan_cache(cache, key, specs, 2).valid);
    }

    #[test]
    fn partial_or_missing_cache_writers_do_not_publish() {
        let mut registry = HistoryRegistry::new();
        let partial_cache = registry.create_cache();
        let missing_cache = registry.create_cache();
        let spec = PersistentTextureSpec::default_view(dummy_desc());
        let partial = registry.plan_cache(partial_cache, CacheKey::new(1), [spec, spec], 1);
        let missing = registry.plan_cache(missing_cache, CacheKey::new(2), [spec], 1);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let partial_a = graph.import_persistent_texture("PartialA", partial.textures[0]);
        let partial_b = graph.import_persistent_texture("PartialB", partial.textures[1]);
        let missing_texture = graph.import_persistent_texture("Missing", missing.textures[0]);
        let output = graph.register_texture("Output", dummy_desc(), true);
        graph.add_pass("LivePartialWriter", |builder| {
            builder.write_texture(partial_a);
            (MockExec, ())
        });
        graph.add_pass("DeadPartialWriter", |builder| {
            builder.write_texture(partial_b);
            (MockExec, ())
        });
        graph.add_pass("ConsumePartial", |builder| {
            builder.read_texture(partial_a);
            builder.write_texture(output);
            (MockExec, ())
        });
        graph.register_cache_commit(
            partial.commit.expect("invalid cache needs a token"),
            &[partial_a, partial_b],
        );
        graph.register_cache_commit(
            missing.commit.expect("invalid cache needs a token"),
            &[missing_texture],
        );
        graph.compile_topology();
        graph.finalize_persistent_cache_attempts_after_dpe(&mut registry);

        assert!(
            graph
                .commit_submitted_persistent_resources(&mut registry)
                .is_empty()
        );
        assert!(
            !registry
                .plan_cache(partial_cache, CacheKey::new(1), [spec, spec], 2)
                .valid
        );
        assert!(
            !registry
                .plan_cache(missing_cache, CacheKey::new(2), [spec], 2)
                .valid
        );
    }

    #[test]
    fn valid_cache_is_read_only_and_partial_refresh_stays_invalid() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let key = CacheKey::new(8);
        let spec = PersistentTextureSpec::default_view(dummy_desc());
        let initial = registry.plan_cache(cache, key, [spec, spec], 1);
        assert!(
            registry
                .commit_cache(
                    initial.commit.expect("invalid cache needs a token"),
                    &initial.textures.map(|texture| texture.id),
                )
                .is_some()
        );

        let valid = registry.plan_cache(cache, key, [spec, spec], 2);
        assert!(valid.valid);
        assert!(valid.commit.is_none(), "valid caches must be read-only");
        assert!(registry.invalidate_cache(cache));
        let refresh = registry.plan_cache(cache, key, [spec, spec], 2);
        assert!(!refresh.valid);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let refresh_a = graph.import_persistent_texture("RefreshA", refresh.textures[0]);
        let refresh_b = graph.import_persistent_texture("RefreshB", refresh.textures[1]);
        let output = graph.register_texture("Output", dummy_desc(), true);
        graph.add_pass("LiveRefreshWriter", |builder| {
            builder.write_texture(refresh_a);
            (MockExec, ())
        });
        graph.add_pass("DeadRefreshWriter", |builder| {
            builder.write_texture(refresh_b);
            (MockExec, ())
        });
        graph.add_pass("ConsumeRefresh", |builder| {
            builder.read_texture(refresh_a);
            builder.write_texture(output);
            (MockExec, ())
        });
        graph.register_cache_commit(
            refresh.commit.expect("invalid cache needs a token"),
            &[refresh_a, refresh_b],
        );
        graph.compile_topology();
        graph.finalize_persistent_cache_attempts_after_dpe(&mut registry);

        assert!(
            graph
                .commit_submitted_persistent_resources(&mut registry)
                .is_empty()
        );
        let next = registry.plan_cache(cache, key, [spec, spec], 3);
        assert!(!next.valid);
        assert!(next.commit.is_some());
    }

    #[test]
    #[should_panic(expected = "conflicting writer sets")]
    fn duplicate_cache_token_rejects_a_different_writer_set() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let spec = PersistentTextureSpec::default_view(dummy_desc());
        let plan = registry.plan_cache(cache, CacheKey::new(1), [spec, spec], 1);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let cache_a = graph.import_persistent_texture("CacheA", plan.textures[0]);
        let cache_b = graph.import_persistent_texture("CacheB", plan.textures[1]);
        let token = plan.commit.expect("invalid cache needs a token");
        graph.register_cache_commit(token, &[cache_a, cache_b]);
        graph.register_cache_commit(token, &[cache_a]);
    }

    #[test]
    #[should_panic(expected = "cache ID registered with conflicting commit tokens")]
    fn one_graph_rejects_multiple_attempt_tokens_for_the_same_cache() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let spec = PersistentTextureSpec::default_view(dummy_desc());
        let first = registry.plan_cache(cache, CacheKey::new(1), [spec], 1);
        let second = registry.plan_cache(cache, CacheKey::new(1), [spec], 2);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let texture = graph.import_persistent_texture("Cache", second.textures[0]);
        graph.register_cache_commit(
            first.commit.expect("first attempt needs a token"),
            &[texture],
        );
        graph.register_cache_commit(
            second.commit.expect("second attempt needs a token"),
            &[texture],
        );
    }

    #[test]
    fn cache_does_not_publish_when_post_submit_step_is_not_called() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let key = CacheKey::new(3);
        let spec = PersistentTextureSpec::default_view(dummy_desc());
        let first = registry.plan_cache(cache, key, [spec], 1);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let texture = graph.import_persistent_texture("Cache", first.textures[0]);
        let output = graph.register_texture("Output", dummy_desc(), true);
        graph.add_pass("Populate", |builder| {
            builder.write_texture(texture);
            (MockExec, ())
        });
        graph.add_pass("Consume", |builder| {
            builder.read_texture(texture);
            builder.write_texture(output);
            (MockExec, ())
        });
        graph.register_cache_commit(
            first.commit.expect("invalid cache needs a token"),
            &[texture],
        );
        graph.compile_topology();

        let second = registry.plan_cache(cache, key, [spec], 2);
        assert!(!second.valid);
        assert!(second.commit.is_some());
    }

    #[test]
    #[should_panic(expected = "conflicting specifications")]
    fn persistent_import_dedup_rejects_a_different_canonical_view() {
        let mut registry = HistoryRegistry::new();
        let cache = registry.create_cache();
        let plan = registry.plan_cache(
            cache,
            CacheKey::new(1),
            [PersistentTextureSpec::default_view(dummy_desc())],
            1,
        );
        let conflicting = PersistentTextureRef {
            id: plan.textures[0].id,
            spec: PersistentTextureSpec::new(
                dummy_desc(),
                Some(wgpu::TextureViewDimension::D2Array),
            ),
        };

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let _ = graph.import_persistent_texture("Default", plan.textures[0]);
        let _ = graph.import_persistent_texture("Conflicting", conflicting);
    }

    #[test]
    fn pure_forwarding_does_not_contract_a_persistent_history_source() {
        let mut histories = HistoryRegistry::new();
        let set = histories.create_set();
        let plan = histories.plan_set(set, [dummy_desc()], 1, false);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let history = graph.import_persistent_texture("History", plan.next[0]);
        let surface = graph.register_texture("Surface", dummy_desc(), true);

        graph.add_pass("Temporal", |builder| {
            builder.write_texture(history);
            (MockExec, ())
        });
        graph.add_pass("Present", |builder| {
            builder.read_texture(history);
            let output = builder.replace_texture(surface, "Surface_Present");
            builder.mark_pure_forwarding();
            (MockExec, output)
        });
        graph.register_history_commit(plan.commit, &[history]);
        graph.compile_topology();

        let names: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(names, ["Temporal", "Present"]);
        assert!(
            graph.storage.resources[history.index() as usize]
                .alias_of
                .is_none()
        );

        graph.commit_submitted_histories(&mut histories);
        assert!(histories.plan_set(set, [dummy_desc()], 2, false).valid);
    }

    #[test]
    fn pure_forwarding_does_not_contract_an_asset_texture_source() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let snapshot = dummy_asset_snapshot();
        let asset = graph.import_asset_texture("Asset", snapshot);
        let surface = graph.register_texture("Surface", snapshot_texture_desc(snapshot), true);

        graph.add_pass("Present", |builder| {
            builder.read_texture(asset);
            let output = builder.replace_texture(surface, "Surface_Present");
            builder.mark_pure_forwarding();
            (MockExec, output)
        });
        graph.compile_topology();

        let names: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(names, ["Present"]);
        assert!(
            graph.storage.resources[asset.index() as usize]
                .alias_of
                .is_none()
        );
    }

    #[test]
    #[should_panic(expected = "persistent textures cannot be versioned")]
    fn persistent_history_cannot_be_mutated_in_place() {
        let mut histories = HistoryRegistry::new();
        let set = histories.create_set();
        let plan = histories.plan_set(set, [dummy_desc()], 1, false);

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let history = graph.import_persistent_texture("History", plan.next[0]);
        graph.create_alias(history, "Mutated_History");
    }

    #[test]
    fn compact_mermaid_aggregates_lazy_asset_inputs_and_full_keeps_details() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let first = graph.import_asset_texture("Arbitrary_First", dummy_asset_snapshot());
        let mut second_snapshot = dummy_asset_snapshot();
        second_snapshot.texture_handle = myth_assets::TextureHandle::system_reserved(8);
        second_snapshot.texture_identity = 34;
        second_snapshot.logical_version = 0xA55E_7002;
        let second = graph.import_asset_texture("Arbitrary_Second", second_snapshot);
        graph.add_pass("Asset_Consumer", |builder| {
            builder.read_texture(first);
            builder.read_texture(second);
            builder.mark_side_effect();
            (MockExec, ())
        });
        let precompile = graph.dump_mermaid();
        assert!(precompile.contains("declared 2 · live pending"));
        assert!(precompile.contains("ASSET_TEXTURES_IN -.->|\"2 textures\"| P0;"));
        graph.compile_topology();

        let compact = graph.dump_mermaid();
        assert!(compact.contains("Asset Textures<br/>declared 2 · live 2"));
        assert!(compact.contains("ASSET_TEXTURES_IN -.->|\"2 textures\"| P0;"));
        assert!(!compact.contains(&format!("IN_{}[", first.index())));
        assert!(!compact.contains(&format!("IN_{}[", second.index())));

        let full = graph.dump_mermaid_with_mode(MermaidDumpMode::Full);
        assert!(!full.contains("ASSET_TEXTURES_IN"));
        assert!(full.contains(&format!("IN_{}[\"Arbitrary_First\"]", first.index())));
        assert!(full.contains(&format!("IN_{}[\"Arbitrary_Second\"]", second.index())));
    }

    #[test]
    fn mermaid_derives_deterministic_consumer_edges_before_and_after_compile() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let output = graph.add_pass("Shared_Producer", |builder| {
            let output = builder.create_texture("Shared_Output", dummy_desc());
            (MockExec, output)
        });
        for name in ["First_Consumer", "Second_Consumer"] {
            graph.add_pass(name, |builder| {
                builder.read_texture(output);
                builder.mark_side_effect();
                (MockExec, ())
            });
        }

        let consumer_edges = |dump: &str| {
            dump.lines()
                .filter(|line| line.contains("-->|\"Shared_Output\"|"))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        let expected = [
            "    P0 -->|\"Shared_Output\"| P1;".to_owned(),
            "    P0 -->|\"Shared_Output\"| P2;".to_owned(),
        ];
        let precompile = consumer_edges(&graph.dump_mermaid_full());
        assert_eq!(precompile, expected);

        graph.compile_topology();

        let compiled = consumer_edges(&graph.dump_mermaid_full());
        assert_eq!(compiled, expected);
        assert_eq!(compiled, precompile);
    }

    #[test]
    fn compact_mermaid_deduplicates_repeated_asset_reads_and_summarizes_dead_imports() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let live = graph.import_asset_texture("Live_Asset", dummy_asset_snapshot());
        let mut dead_snapshot = dummy_asset_snapshot();
        dead_snapshot.texture_handle = myth_assets::TextureHandle::system_reserved(8);
        dead_snapshot.texture_identity = 34;
        dead_snapshot.logical_version = 0xA55E_7002;
        graph.import_asset_texture("Dead_Asset", dead_snapshot);
        graph.add_pass("Repeated_Consumer", |builder| {
            builder.read_texture(live);
            builder.read_texture(live);
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph.compile_topology();

        let compact = graph.dump_mermaid();
        assert!(compact.contains("Asset Textures<br/>declared 2 · live 1"));
        assert!(compact.contains("ASSET_TEXTURES_IN -.->|\"1 texture\"| P0;"));
    }

    #[test]
    fn compact_mermaid_preserves_real_mipmap_producer_edges() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
        let mip_ready = graph.add_pass("Real_Mipmap", |builder| {
            let output = builder.mutate_texture(asset, "Asset_Mip_Ready");
            (MockExec, output)
        });
        graph.add_pass("Asset_Consumer", |builder| {
            builder.read_texture(mip_ready);
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph.compile_topology();

        let compact = graph.dump_mermaid();
        assert!(compact.contains("ASSET_TEXTURES_IN -.->|\"1 texture\"| P0;"));
        assert!(!compact.contains("ASSET_TEXTURES_IN -.->|\"1 texture\"| P1;"));
        assert!(compact.contains("P0 ==>|\"Asset_Mip_Ready\"| P1;"));
    }

    #[test]
    fn mermaid_does_not_invent_an_output_for_an_unconsumed_asset_alias() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let asset = graph.import_asset_texture("Asset", dummy_asset_snapshot());
        let mip_ready = graph.add_pass("Side_Effect_Mipmap", |builder| {
            let output = builder.mutate_texture(asset, "Unconsumed_Mip_Ready");
            builder.mark_side_effect();
            (MockExec, output)
        });
        graph.compile_topology();

        for dump in [graph.dump_mermaid(), graph.dump_mermaid_full()] {
            assert!(!dump.contains(&format!("OUT_{}", mip_ready.index())));
            assert!(!dump.contains(&format!("P0 --> OUT_{}", mip_ready.index())));
        }
    }

    #[test]
    fn mermaid_escapes_custom_pass_and_resource_labels() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let output = graph.add_pass("Pass \"Quoted\"\nNext", |builder| {
            let output = builder.create_texture("Color <HDR> & More", dummy_desc());
            (MockExec, output)
        });
        graph.add_pass("Label_Consumer", |builder| {
            builder.read_texture(output);
            builder.mark_side_effect();
            (MockExec, ())
        });
        graph.compile_topology();

        let dump = graph.dump_mermaid_full();
        assert!(dump.contains("Pass &quot;Quoted&quot;<br/>Next"));
        assert!(dump.contains("Color &lt;HDR&gt; &amp; More"));
    }

    #[cfg(feature = "rdg_inspector")]
    #[test]
    fn test_dump_mermaid_subgraphs() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);
        let frame_config = dummy_config();

        let backbuffer = graph.register_texture("Backbuffer", dummy_desc(), true);

        let mut ctx = GraphBuilderContext::new(&mut graph, &frame_config, 0);

        let bloom_out = ctx.with_group("Bloom_System", |ctx| {
            let extract_out = ctx.graph.add_pass("Bloom_Extract", |builder| {
                let out = builder.create_texture("Bloom_Mip0", dummy_desc());
                (MockExec, out)
            });

            ctx.graph.add_pass("Bloom_Composite", |builder| {
                builder.read_texture(extract_out);
                let out = builder.create_texture("Bloom_Final", dummy_desc());
                (MockExec, out)
            })
        });

        ctx.graph.add_pass("ToneMap", |builder| {
            builder.read_texture(bloom_out);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        let mermaid = graph.dump_mermaid();

        assert!(
            mermaid.contains("subgraph G0 [\"Bloom_System\"]"),
            "Mermaid output should contain the Bloom_System subgraph"
        );
        assert!(
            mermaid.contains("Bloom_Extract"),
            "Subgraph should contain Bloom_Extract pass"
        );
        assert!(
            mermaid.contains("Bloom_Composite"),
            "Subgraph should contain Bloom_Composite pass"
        );
        assert!(
            mermaid.contains("ToneMap"),
            "Mermaid should have an ungrouped section for ToneMap"
        );
        assert_eq!(graph.storage.passes[0].groups.as_slice(), &["Bloom_System"]);
        assert_eq!(graph.storage.passes[1].groups.as_slice(), &["Bloom_System"]);
        assert!(graph.storage.passes[2].groups.is_empty());
    }
}
