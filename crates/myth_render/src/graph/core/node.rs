use crate::core::gpu::GlobalStateKey;
use crate::graph::core::context::{ExecuteContext, PrepareContext};
use std::ops::Range;

use super::types::ErasedResourceNodeId;
use smallvec::SmallVec;
use wgpu::CommandEncoder;

/// Pure GPU command recorder for a single render or compute pass.
///
/// `PassNode` is intentionally minimal — it carries lightweight IDs and
/// borrowed POD/frame data. Destructor-bearing prepared GPU resources live in
/// GraphStorage-owned stores, while persistent layouts, pipelines, and buffers
/// live in the owning `Feature`.
///
/// # Lifecycle
///
/// Resource topology and naming are declared **outside** the `PassNode`,
/// inside the closure passed to [`RenderGraph::add_pass`].  The node
/// itself participates in four runtime phases:
///
/// 1. **`prepare`** — called after graph compilation and transient memory
///    allocation, and only for passes that survived dead-pass elimination.
///    Materialize pipelines, uploads, and bind groups without changing graph
///    topology or resource descriptors.
/// 2. **`finalize_prepare`** — after every live node has gathered recipes and
///    the serial coordinators have assigned physical IDs and closed their
///    final epoch gates, publish Feature-owned L1 state without creating
///    resources or mutating scene-source/asset epochs.
/// 3. **`execute`** — record GPU commands into the shared encoder.
/// 4. **`after_submit`** — retire pass-owned resources only after the shared
///    command buffer has been handed to the queue.
///
/// # Lifetime Model
///
/// The lifetime parameter `'a` represents the duration of a single frame.
/// Concrete implementations may carry frame-scoped borrowed references
/// (e.g. `&'a [ShadowLightInstance]`) and Copy IDs for destructor-bearing
/// prepared GPU objects. Their
/// lifetimes are tied to either:
///
/// - The [`FrameArena`](super::arena::FrameArena) that allocates the node.
/// - Persistent `Feature` state that outlives the frame.
///
/// Because all nodes are allocated on the arena and the arena is reset
/// at frame boundaries, **no `Drop` glue is required** — the arena
/// reclaims all memory in O(1) without running destructors.
pub trait PassNode<'a>: Send + Sync {
    /// Materialize the execution state after physical resource allocation.
    ///
    /// Heavy CPU derivation and GPU interaction should live here when they do
    /// not affect graph topology, resource descriptors, or dependency edges.
    #[allow(unused_variables)]
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {}

    /// Publish coordinator results after the global Gather -> Plan ->
    /// Coordinate barrier. The default is intentionally empty; only nodes
    /// that own a cross-frame L1 derived from a shared recipe need this hook.
    /// The coordinator's accepted streams are already linearized here, so
    /// implementations must not mutate scene-source or asset epochs.
    #[allow(unused_variables)]
    fn finalize_prepare(&mut self, ctx: &mut PrepareContext<'a>) {}

    /// Record GPU commands into the shared encoder.
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut CommandEncoder);

    /// Retire resources whose last encoded use has been handed to the queue.
    ///
    /// This is called synchronously once for each live pass after
    /// [`wgpu::Queue::submit`] returns. It is not called when target
    /// acquisition, preparation, or command encoding does not reach
    /// submission. The callback observes queue submission, not GPU completion;
    /// wgpu retains submitted resources for as long as the device needs them.
    fn after_submit(&mut self) {}
}

// ─── NodeSlot ──────────────────────────────────────────────────────────────

/// Type-erased handle to a [`PassNode`] allocated on the [`FrameArena`].
///
/// Stores a fat pointer (data + vtable) to a `dyn PassNode` trait object.
/// No ownership or drop semantics — the arena reclaims all memory in O(1)
/// via [`FrameArena::reset()`](super::arena::FrameArena::reset) at frame
/// boundaries.
///
/// # Safety Contract
///
/// The arena allocation must outlive this handle.  In practice this is
/// guaranteed by the frame lifecycle: `arena.reset()` is called only
/// after execute and all after-submit callbacks complete.
pub(crate) struct NodeSlot {
    /// Fat pointer to the `dyn PassNode` trait object.
    pub(crate) ptr: *mut dyn PassNode<'static>,
}

// SAFETY: `NodeSlot` wraps a pointer to a `dyn PassNode` which itself
// requires `Send + Sync`.  The engine's single-threaded frame model
// guarantees no concurrent access to the pointed-to data.
unsafe impl Send for NodeSlot {}
unsafe impl Sync for NodeSlot {}

impl NodeSlot {
    /// Creates a slot for an arena-allocated or externally-owned node.
    #[inline]
    #[allow(clippy::transmute_ptr_to_ptr)]
    pub(crate) fn new<'a, N: PassNode<'a> + 'a>(ptr: *mut N) -> Self {
        let erased_ptr = unsafe {
            std::mem::transmute::<*mut dyn PassNode<'a>, *mut dyn PassNode<'static>>(
                ptr as *mut dyn PassNode<'a>,
            )
        };

        Self { ptr: erased_ptr }
    }
}

// ─── PassRecord ────────────────────────────────────────────────────────────

/// Per-pass metadata stored in the [`RenderGraph`].
///
/// Holds per-frame topology information (reads, writes, dependencies) and
/// a type-erased handle to the [`PassNode`] allocated on the
/// [`FrameArena`](super::arena::FrameArena).
pub struct PassRecord {
    /// Human-readable name for debug labels, topology dumps, and GPU
    /// debug groups.  Set once by [`RenderGraph::add_pass`].
    pub name: &'static str,

    /// Logical group this pass belongs to (e.g. `"Bloom_System"`).
    ///
    /// Populated by [`RenderGraph::with_group`] when the `rdg_inspector`
    /// feature is enabled; otherwise always `None`.  Used exclusively
    /// by [`RenderGraph::dump_mermaid`] to emit Mermaid `subgraph` blocks.
    #[cfg(feature = "rdg_inspector")]
    pub groups: smallvec::SmallVec<[&'static str; 4]>,

    /// Type-erased handle to the arena-allocated pass node.
    pub(crate) node: Option<NodeSlot>,

    /// Range into GraphStorage's capacity-retaining flat logical-read table.
    /// Direct setup reads and deferred submission dependencies remain
    /// contiguous because pass setup is deliberately non-reentrant.
    pub(crate) logical_read_range: Range<usize>,
    pub writes: SmallVec<[ErasedResourceNodeId; 4]>,
    pub creates: SmallVec<[ErasedResourceNodeId; 4]>,
    /// Canonical backend-neutral scene-submission plans consumed by this
    /// pass, addressed through GraphStorage's retained flat demand table.
    /// Demand metadata never roots an otherwise dead pass.
    pub(crate) scene_submission_demand_range: Range<usize>,

    /// Optional Group 0 demand projected only from DPE-live passes.
    /// Repeated declarations in one setup closure are idempotent; conflicting
    /// keys are rejected immediately by [`PassBuilder`](super::builder::PassBuilder).
    pub(crate) global_state_demand: Option<GlobalStateKey>,

    // Compile-time state
    /// Range into GraphStorage's capacity-retaining flat producer table.
    /// Entries preserve first-read producer order and are unique per pass.
    pub(crate) physical_dependency_range: Range<usize>,
    pub has_side_effect: bool,
    pub reference_count: u32,

    /// Marks the pass as a pure forwarding (blit/present) node whose sole
    /// purpose is to copy a single source resource into a single destination
    /// of identical format and size.
    ///
    /// Such passes are candidates for compile-time **edge contraction** in
    /// [`RenderGraph::fold_simple_passes`](super::graph::RenderGraph): the
    /// upstream producer is rewired to write the destination directly and the
    /// forwarding pass is stripped to an island for the subsequent dead-pass
    /// cull, eliminating the redundant copy at zero runtime cost.
    pub is_pure_forwarding: bool,
}

impl PassRecord {
    /// Creates a placeholder record without a node.
    ///
    /// Used internally by [`RenderGraph::add_pass`] during the two-phase
    /// insertion: the record is pushed first so the [`PassBuilder`] can
    /// reference it, then the node is stored after the closure returns.
    #[must_use]
    pub(crate) fn new_empty(name: &'static str, logical_read_start: usize) -> Self {
        Self {
            name,
            #[cfg(feature = "rdg_inspector")]
            groups: SmallVec::new(),
            node: None,
            logical_read_range: logical_read_start..logical_read_start,
            writes: SmallVec::new(),
            creates: SmallVec::new(),
            scene_submission_demand_range: 0..0,
            global_state_demand: None,
            physical_dependency_range: 0..0,
            has_side_effect: false,
            reference_count: 0,
            is_pure_forwarding: false,
        }
    }

    /// Returns a mutable reference to the pass node.
    ///
    /// # Panics
    /// Panics if the node has not been inserted yet.
    ///
    /// # Safety
    ///
    /// The returned reference is derived from a raw pointer stored in
    /// [`NodeSlot`]. Callers must ensure the owning [`RenderGraph`] and
    /// [`FrameArena`](super::arena::FrameArena) remain alive, the returned
    /// reference does not escape that graph attempt, and no aliasing mutable
    /// references exist. In practice, the sequential
    /// prepare→finalize-prepare→execute→after-submit pipeline guarantees these
    /// conditions.
    #[inline]
    #[allow(clippy::transmute_ptr_to_ptr)]
    pub(crate) unsafe fn get_pass_mut<'a>(&mut self) -> &mut (dyn PassNode<'a> + 'a) {
        let slot = self.node.as_ref().expect("PassRecord node not set");
        // SAFETY: The pointer was set by `add_pass`
        // and remains valid until `arena.reset()`.
        // `&mut self` guarantees exclusive access to this record.
        // unsafe { std::mem::transmute(&mut *slot.ptr) }
        unsafe {
            let ptr =
                std::mem::transmute::<*mut dyn PassNode<'static>, *mut dyn PassNode<'a>>(slot.ptr);
            &mut *ptr
        }
    }
}
