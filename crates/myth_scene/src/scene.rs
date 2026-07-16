use std::borrow::Cow;
use std::sync::atomic::{AtomicU32, Ordering};

use myth_animation::{AnimationMixer, AnimationTarget};
use myth_core::{NodeHandle, SkeletonKey, Transform};
#[cfg(feature = "3dgs")]
use myth_resources::GaussianCloudHandle;
use myth_resources::Input;
use myth_resources::bloom::BloomSettings;
use myth_resources::mesh::Mesh;
use myth_resources::shader_defines::ShaderDefines;
use myth_resources::ssao::SsaoSettings;
use myth_resources::ssgi::SsgiSettings;
use myth_resources::ssr::SsrSettings;
use myth_resources::ssss::SsssSettings;
use myth_resources::tone_mapping::ToneMappingSettings;

use crate::background::{BackgroundMode, BackgroundSettings};
use crate::camera::Camera;
use crate::environment::Environment;
use crate::light::Light;
use crate::mutation::{
    SceneFullRebuildReasons, SceneMutationBatch, SceneMutationCursor, SceneMutationFlags,
    SceneMutationJournal, SceneMutationJournalStats, SceneStorage, SceneStorageGenerations,
    SceneStorageStamp,
};
use crate::node::Node;
use crate::node_editor::NodeEditor;
use crate::skeleton::{BindMode, Skeleton, SkinBinding};
use crate::transform_system;
use glam::{Affine3A, Quat, Vec3};
use rustc_hash::FxHashSet;
use slotmap::{SecondaryMap, SlotMap, SparseSecondaryMap};

static NEXT_SCENE_ID: AtomicU32 = AtomicU32::new(1);
const SCENE_FRONTEND_AUTO_TRIM_IDLE_ACKNOWLEDGEMENTS: u16 = 300;

/// Trait for scene update logic.
///
/// Allows users to define custom behavior scripts that update
/// along with the scene lifecycle each frame.
///
/// # Example
///
/// ```rust,ignore
/// struct RotateScript {
///     target: NodeHandle,
///     speed: f32,
/// }
///
/// impl SceneLogic for RotateScript {
///     fn update(&mut self, scene: &mut Scene, input: &Input, dt: f32) {
///         if let Some(node) = scene.node_mut(self.target) {
///             node.transform.rotation *= Quat::from_rotation_y(self.speed * dt);
///         }
///     }
/// }
/// ```
pub trait SceneLogic: Send + Sync + 'static {
    /// Called each frame to update scene state.
    fn update(&mut self, scene: &mut Scene, input: &Input, dt: f32);
}

/// Syntactic sugar: allows using closures directly as scene logic.
pub struct CallbackLogic<F>(pub F);
impl<F> SceneLogic for CallbackLogic<F>
where
    F: FnMut(&mut Scene, &Input, f32) + Send + Sync + 'static,
{
    fn update(&mut self, scene: &mut Scene, input: &Input, dt: f32) {
        (self.0)(scene, input, dt);
    }
}

/// Tag component indicating a split primitive node.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitPrimitiveTag;

/// Panic-safe exact-edit boundary for public Transform callbacks. Cached
/// matrices and dirty bits are restored unconditionally; a successful commit
/// writes back only local TRS.
struct TransformEditTransaction<'a> {
    transform: &'a mut Transform,
    before: Transform,
    committed: bool,
}

impl<'a> TransformEditTransaction<'a> {
    fn new(transform: &'a mut Transform) -> Self {
        Self {
            before: *transform,
            transform,
            committed: false,
        }
    }

    fn transform(&mut self) -> &mut Transform {
        self.transform
    }

    fn commit(mut self) -> bool {
        let after = (
            self.transform.position,
            self.transform.rotation,
            self.transform.scale,
        );
        let before = (
            self.before.position,
            self.before.rotation,
            self.before.scale,
        );
        *self.transform = self.before;
        if after != before {
            self.transform.position = after.0;
            self.transform.rotation = after.1;
            self.transform.scale = after.2;
        }
        self.committed = true;
        after != before
    }
}

impl Drop for TransformEditTransaction<'_> {
    fn drop(&mut self) {
        if !self.committed {
            *self.transform = self.before;
        }
    }
}

#[derive(Debug)]
struct TransformUpdateState {
    full_traversal_required: bool,
    observed_nodes_storage_generation: SceneStorageStamp,
    dirty_roots: Vec<NodeHandle>,
    dirty_root_set: FxHashSet<NodeHandle>,
    settle_roots: Vec<NodeHandle>,
    settle_full_traversal: bool,
    update_stack: Vec<(NodeHandle, Affine3A, bool)>,
    settle_stack: Vec<NodeHandle>,
}

impl TransformUpdateState {
    fn new() -> Self {
        Self {
            full_traversal_required: true,
            observed_nodes_storage_generation: SceneStorageStamp::default(),
            dirty_roots: Vec::new(),
            dirty_root_set: FxHashSet::default(),
            settle_roots: Vec::new(),
            settle_full_traversal: false,
            update_stack: Vec::with_capacity(64),
            settle_stack: Vec::with_capacity(64),
        }
    }

    fn mark_dirty_root(&mut self, nodes: &SlotMap<NodeHandle, Node>, handle: NodeHandle) {
        if self.full_traversal_required || !nodes.contains_key(handle) {
            return;
        }
        if self.dirty_root_set.insert(handle) {
            self.dirty_roots.push(handle);
        }
    }

    /// Reduce O(1)-collected exact dirty nodes to disjoint subtree roots.
    ///
    /// Maintaining the minimum set during every mutation makes k independent
    /// roots perform k² pairwise ancestry checks.  This single retained pass
    /// follows only parent links, making flat animation O(k) and general work
    /// O(k * hierarchy depth).
    fn coalesce_dirty_roots(&mut self, nodes: &SlotMap<NodeHandle, Node>) -> (usize, usize) {
        let examined = self.dirty_roots.len();
        if examined == 0 || self.full_traversal_required {
            return (examined, 0);
        }

        let mut parent_links_examined = 0usize;
        let mut write = 0usize;
        let mut hierarchy_drifted = false;
        for read in 0..self.dirty_roots.len() {
            let handle = self.dirty_roots[read];
            let mut ancestor = nodes.get(handle).and_then(Node::parent);
            let mut remaining = nodes.len().saturating_add(1);
            let mut covered = false;
            while let Some(parent) = ancestor {
                if remaining == 0 {
                    hierarchy_drifted = true;
                    break;
                }
                remaining -= 1;
                parent_links_examined = parent_links_examined.saturating_add(1);
                if self.dirty_root_set.contains(&parent) {
                    covered = true;
                    break;
                }
                ancestor = nodes.get(parent).and_then(Node::parent);
            }
            if hierarchy_drifted {
                break;
            }
            if !covered {
                self.dirty_roots[write] = handle;
                write += 1;
            }
        }

        if hierarchy_drifted {
            self.full_traversal_required = true;
            return (examined, parent_links_examined);
        }
        self.dirty_roots.truncate(write);

        // A new dirty root settles its own previous value while updating.
        // Remove only settle subtrees covered by a dirty node; a settle parent
        // must remain when just one of its descendants is dirty.
        self.settle_roots.retain(|&settle_root| {
            let mut current = Some(settle_root);
            let mut remaining = nodes.len().saturating_add(1);
            while let Some(handle) = current {
                if self.dirty_root_set.contains(&handle) {
                    return false;
                }
                if remaining == 0 {
                    return true;
                }
                remaining -= 1;
                current = nodes.get(handle).and_then(Node::parent);
            }
            true
        });

        self.dirty_root_set.clear();
        self.dirty_root_set.extend(self.dirty_roots.iter().copied());
        (examined, parent_links_examined)
    }

    fn remove_handle(&mut self, handle: NodeHandle) {
        if self.dirty_root_set.remove(&handle)
            && let Some(index) = self.dirty_roots.iter().position(|&root| root == handle)
        {
            self.dirty_roots.swap_remove(index);
        }
        self.settle_roots.retain(|&root| root != handle);
    }

    fn clear_dirty_roots(&mut self) {
        self.dirty_roots.clear();
        self.dirty_root_set.clear();
    }

    fn finish_dirty_update(&mut self, schedule_settle: bool) {
        self.settle_roots.clear();
        if schedule_settle {
            self.settle_roots.extend_from_slice(&self.dirty_roots);
        }
        self.clear_dirty_roots();
    }

    /// Finish an extraction-boundary update without consuming settlement
    /// owed by an earlier simulation update in the same renderer frame.
    fn finish_dirty_update_preserving_settles(
        &mut self,
        nodes: &SlotMap<NodeHandle, Node>,
        schedule_settle: bool,
    ) {
        if schedule_settle {
            // Both inputs are already disjoint internally. Build their union
            // once, then discard every root whose ancestor is also present.
            // The former incremental merge compared every dirty root against
            // every retained settle root, making a flat animation frame O(k²).
            // Reusing the retained dirty-root index keeps this O(k * depth)
            // without a frame allocation.
            self.dirty_root_set.clear();
            let mut write = 0usize;
            for read in 0..self.settle_roots.len() {
                let root = self.settle_roots[read];
                if self.dirty_root_set.insert(root) {
                    self.settle_roots[write] = root;
                    write += 1;
                }
            }
            self.settle_roots.truncate(write);
            for &dirty_root in &self.dirty_roots {
                if self.dirty_root_set.insert(dirty_root) {
                    self.settle_roots.push(dirty_root);
                }
            }

            let mut write = 0usize;
            let mut hierarchy_drifted = false;
            for read in 0..self.settle_roots.len() {
                let root = self.settle_roots[read];
                let mut ancestor = nodes.get(root).and_then(Node::parent);
                let mut remaining = nodes.len().saturating_add(1);
                let mut covered = false;
                while let Some(parent) = ancestor {
                    if remaining == 0 {
                        hierarchy_drifted = true;
                        break;
                    }
                    remaining -= 1;
                    if self.dirty_root_set.contains(&parent) {
                        covered = true;
                        break;
                    }
                    ancestor = nodes.get(parent).and_then(Node::parent);
                }
                if hierarchy_drifted {
                    break;
                }
                if !covered {
                    self.settle_roots[write] = root;
                    write += 1;
                }
            }

            if hierarchy_drifted {
                self.settle_roots.clear();
                self.settle_full_traversal = true;
            } else {
                self.settle_roots.truncate(write);
            }
        }
        self.clear_dirty_roots();
    }

    fn estimated_retained_bytes(&self) -> usize {
        let handles = self
            .dirty_roots
            .capacity()
            .saturating_add(self.settle_roots.capacity())
            .saturating_add(self.settle_stack.capacity())
            .saturating_mul(std::mem::size_of::<NodeHandle>());
        let dirty_index = self
            .dirty_root_set
            .capacity()
            .saturating_mul(std::mem::size_of::<NodeHandle>());
        let update_stack = self
            .update_stack
            .capacity()
            .saturating_mul(std::mem::size_of::<(NodeHandle, Affine3A, bool)>());
        handles
            .saturating_add(dirty_index)
            .saturating_add(update_stack)
    }

    fn trim_idle_capacity(&mut self) -> bool {
        if self.full_traversal_required
            || self.settle_full_traversal
            || !self.dirty_roots.is_empty()
            || !self.settle_roots.is_empty()
        {
            return false;
        }
        let before = self.estimated_retained_bytes();
        self.dirty_roots.shrink_to(64);
        self.dirty_root_set.shrink_to(64);
        self.settle_roots.shrink_to(64);
        self.update_stack.shrink_to(64);
        self.settle_stack.shrink_to(64);
        self.estimated_retained_bytes() < before
    }
}

/// The scene graph container.
///
/// Scene is the pure data layer that stores scene graph hierarchy and component data.
/// Uses `SlotMap` + `SecondaryMap` for high-performance component-based storage.
///
/// # Storage Layout
///
/// - `nodes`: Core node data (hierarchy and transforms) using `SlotMap`
/// - Dense components (names, meshes): Use `SecondaryMap`
/// - Sparse components (cameras, lights, skins): Use `SparseSecondaryMap`
///
/// # Access semantics
///
/// - noun accessors such as [`Scene::node`] and [`Scene::mesh`] are read-only;
/// - `*_mut` accessors return escaping mutable references and conservatively
///   invalidate any persistent render domain owned by that component;
/// - `edit_*` APIs provide narrow, change-detected transactions and should be
///   preferred in per-frame hot paths.
///
/// # Example
///
/// ```rust,ignore
/// let mut scene = Scene::new();
///
/// // Create nodes
/// let root = scene.create_node_with_name("Root");
/// let child = scene.create_node_with_name("Child");
/// scene.attach(child, root);
///
/// // Add mesh component
/// scene.set_mesh(child, Mesh::new(geometry, material));
/// ```
pub struct Scene {
    /// Unique scene identifier (assigned automatically, read-only)
    id: u32,

    // === Core Node Storage ===
    /// Low-level node storage retained as a fail-closed extension boundary.
    ///
    /// Prefer [`Scene::node`], [`Scene::node_mut`], or [`Scene::edit_node`].
    /// Direct mutable access is detected and forces a conservative RenderWorld
    /// rebuild.
    #[doc(hidden)]
    pub nodes: SceneStorage<SlotMap<NodeHandle, Node>>,
    /// Root-level nodes (no parent)
    root_nodes: Vec<NodeHandle>,

    // === Dense Components (most nodes have these) ===
    /// Node names - almost all nodes have a name
    pub names: SecondaryMap<NodeHandle, Cow<'static, str>>,

    // === Sparse Components (only some nodes have these) ===
    /// Low-level mesh storage. Prefer [`Scene::mesh`] and [`Scene::mesh_mut`].
    /// Direct mutable access forces a conservative RenderWorld rebuild.
    pub meshes: SceneStorage<SparseSecondaryMap<NodeHandle, Mesh>>,
    /// Camera components stored directly on nodes
    pub cameras: SparseSecondaryMap<NodeHandle, Camera>,
    /// Light components stored directly on nodes
    pub lights: SparseSecondaryMap<NodeHandle, Light>,
    /// Low-level skeletal skin bindings. Direct mutable access forces a
    /// conservative RenderWorld rebuild.
    pub skins: SceneStorage<SparseSecondaryMap<NodeHandle, SkinBinding>>,
    /// Low-level morph target weights. Prefer [`Scene::morph_weights`] and
    /// [`Scene::morph_weights_mut`]. Direct mutable access forces a conservative
    /// RenderWorld rebuild.
    pub morph_weights: SceneStorage<SparseSecondaryMap<NodeHandle, Vec<f32>>>,
    /// Animation mixer components (sparse, only character roots have animations)
    pub animation_mixers: SparseSecondaryMap<NodeHandle, AnimationMixer>,
    /// Rest pose transforms recorded before animation takes over.
    /// Used to restore nodes when animations stop or blend with weight < 1.0.
    pub rest_transforms: SparseSecondaryMap<NodeHandle, Transform>,
    /// Low-level split-primitive tags. Direct mutable access forces a
    /// conservative RenderWorld rebuild.
    pub split_primitive_tags: SceneStorage<SparseSecondaryMap<NodeHandle, SplitPrimitiveTag>>,
    #[cfg(feature = "3dgs")]
    /// Low-level Gaussian splatting point-cloud handles. Direct mutable access
    /// forces a conservative RenderWorld rebuild.
    pub gaussian_clouds: SceneStorage<SparseSecondaryMap<NodeHandle, GaussianCloudHandle>>,

    // === Resource Pools (only truly shared resources) ===
    /// Low-level shared skeleton pool. Direct mutable access forces a
    /// conservative RenderWorld rebuild.
    pub skeleton_pool: SceneStorage<SlotMap<SkeletonKey, Skeleton>>,

    // === Environment and Global Settings ===
    /// Scene environment settings (skybox, IBL)
    pub environment: Environment,
    /// Tone mapping settings (exposure, mode)
    pub tone_mapping: ToneMappingSettings,
    /// Bloom post-processing settings
    pub bloom: BloomSettings,
    /// SSAO (Screen Space Ambient Occlusion) settings
    pub ssao: SsaoSettings,
    /// SSGI (Screen Space Global Illumination) settings
    pub ssgi: SsgiSettings,
    /// SSR (Screen Space Reflections) tuning settings
    pub ssr: SsrSettings,
    /// SSSS (Screen Space Subsurface Scattering) settings
    pub ssss: SsssSettings,
    /// Background rendering settings (mode + skybox uniform buffer)
    pub background: BackgroundSettings,
    /// Currently active camera for rendering
    pub active_camera: Option<NodeHandle>,

    shader_defines: ShaderDefines,

    last_env_version: u64,

    mutation_journal: SceneMutationJournal,
    render_frontend_idle_acknowledgements: u16,

    transform_update_state: TransformUpdateState,
    pending_render_transform_update_stats: transform_system::TransformUpdateStats,
    last_transform_update_stats: transform_system::TransformUpdateStats,

    // === Scene Logic System ===
    pub(crate) logics: Vec<Box<dyn SceneLogic>>,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        let id = NEXT_SCENE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("Scene identity space exhausted");
        Self {
            id,

            nodes: SceneStorage::new(SlotMap::with_key(), id),
            root_nodes: Vec::new(),

            // Dense components
            names: SecondaryMap::new(),

            // Sparse components (direct storage)
            meshes: SceneStorage::new(SparseSecondaryMap::new(), id),
            cameras: SparseSecondaryMap::new(),
            lights: SparseSecondaryMap::new(),
            skins: SceneStorage::new(SparseSecondaryMap::new(), id),
            morph_weights: SceneStorage::new(SparseSecondaryMap::new(), id),
            animation_mixers: SparseSecondaryMap::new(),
            rest_transforms: SparseSecondaryMap::new(),

            split_primitive_tags: SceneStorage::new(SparseSecondaryMap::new(), id),
            #[cfg(feature = "3dgs")]
            gaussian_clouds: SceneStorage::new(SparseSecondaryMap::new(), id),

            // Resource pools (only truly shared resources)
            skeleton_pool: SceneStorage::new(SlotMap::with_key(), id),

            environment: Environment::new(),
            tone_mapping: ToneMappingSettings::default(),
            bloom: BloomSettings::default(),
            ssao: SsaoSettings::default(),
            ssgi: SsgiSettings::default(),
            ssr: SsrSettings::default(),
            ssss: SsssSettings::default(),
            background: BackgroundSettings::default(),

            active_camera: None,

            shader_defines: ShaderDefines::default(),
            last_env_version: 0,

            mutation_journal: SceneMutationJournal::new(id),
            render_frontend_idle_acknowledgements: 0,
            transform_update_state: TransformUpdateState::new(),
            pending_render_transform_update_stats: transform_system::TransformUpdateStats::default(
            ),
            last_transform_update_stats: transform_system::TransformUpdateStats::default(),

            logics: Vec::new(),
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Returns the unique scene identifier.
    #[inline]
    #[must_use]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the render-relevant mutations not yet acknowledged by the
    /// persistent render-world consumer.
    ///
    /// A batch remains valid until the next mutable access to the scene. The
    /// cursor may be acknowledged after the batch borrow ends; mutations newer
    /// than that cursor remain pending.
    #[inline]
    #[must_use]
    pub fn render_mutations(&self) -> SceneMutationBatch<'_> {
        self.mutation_journal.batch(self.storage_generations())
    }

    /// Acknowledges the CPU-derived state captured through `cursor`.
    ///
    /// Returns `false` for a cursor from another scene or from the future.
    /// This acknowledgement is intentionally independent of submitted-frame
    /// temporal history.
    #[inline]
    pub fn acknowledge_render_mutations(&mut self, cursor: SceneMutationCursor) -> bool {
        let current_storage_generations = self.storage_generations();
        let was_non_idle = {
            let batch = self.mutation_journal.batch(current_storage_generations);
            batch.requires_full_rebuild || !batch.entries.is_empty()
        };
        let acknowledged = self
            .mutation_journal
            .acknowledge(cursor, current_storage_generations);
        if !acknowledged || was_non_idle {
            self.render_frontend_idle_acknowledgements = 0;
            return acknowledged;
        }

        self.render_frontend_idle_acknowledgements =
            self.render_frontend_idle_acknowledgements.saturating_add(1);
        if self.render_frontend_idle_acknowledgements
            >= SCENE_FRONTEND_AUTO_TRIM_IDLE_ACKNOWLEDGEMENTS
        {
            self.mutation_journal.trim_idle_capacity();
            self.transform_update_state.trim_idle_capacity();
            self.render_frontend_idle_acknowledgements = 0;
        }
        true
    }

    /// Conservatively invalidates the incremental render-world authority.
    /// The next consumer must perform a full rebuild.
    #[inline]
    pub fn invalidate_render_mutation_authority(&mut self) {
        self.render_frontend_idle_acknowledgements = 0;
        self.mutation_journal
            .force_full_rebuild(SceneFullRebuildReasons::EXPLICIT);
    }

    /// Returns the retained-capacity ledger for the scene mutation journal.
    #[inline]
    #[must_use]
    pub fn render_mutation_journal_stats(&self) -> SceneMutationJournalStats {
        self.mutation_journal.stats()
    }

    /// Explicit non-hot-path maintenance hook for releasing a historical
    /// mutation high-water mark. Trimming is refused while work is pending.
    #[inline]
    pub fn trim_render_mutation_journal(&mut self) -> bool {
        let batch = self.render_mutations();
        if batch.requires_full_rebuild || !batch.entries.is_empty() {
            return false;
        }
        self.mutation_journal.trim_idle_capacity()
    }

    fn storage_generations(&self) -> SceneStorageGenerations {
        SceneStorageGenerations {
            nodes: self.nodes.external_mutation_generation(),
            meshes: self.meshes.external_mutation_generation(),
            skeletons: self.skeleton_pool.external_mutation_generation(),
            skins: self.skins.external_mutation_generation(),
            morph_weights: self.morph_weights.external_mutation_generation(),
            split_primitive_tags: self.split_primitive_tags.external_mutation_generation(),
            #[cfg(feature = "3dgs")]
            gaussian_clouds: self.gaussian_clouds.external_mutation_generation(),
        }
    }

    #[inline]
    fn record_render_mutation(&mut self, node: NodeHandle, flags: SceneMutationFlags) {
        self.render_frontend_idle_acknowledgements = 0;
        if flags.intersects(SceneMutationFlags::TRANSFORM | SceneMutationFlags::TOPOLOGY) {
            self.transform_update_state
                .mark_dirty_root(self.nodes.inner(), node);
        }
        self.mutation_journal.record(node, flags);
    }

    /// Returns a read-only slice of root-level node handles.
    #[inline]
    #[must_use]
    pub fn root_nodes(&self) -> &[NodeHandle] {
        &self.root_nodes
    }

    /// Registers a node as a root-level node (no parent).
    pub fn push_root_node(&mut self, handle: NodeHandle) {
        self.root_nodes.push(handle);
        self.record_render_mutation(
            handle,
            SceneMutationFlags::TOPOLOGY
                | SceneMutationFlags::TRANSFORM
                | SceneMutationFlags::RENDERABLE,
        );
    }

    /// Returns a read-only reference to the node storage.
    #[inline]
    #[must_use]
    pub fn nodes(&self) -> &SlotMap<NodeHandle, Node> {
        self.nodes.inner()
    }

    // ========================================================================
    // Node Management API
    // ========================================================================

    /// Creates a new node and returns its handle.
    pub fn create_node(&mut self) -> NodeHandle {
        let handle = self.nodes.inner_mut_for(self.id).insert(Node::new());
        self.record_render_mutation(
            handle,
            SceneMutationFlags::TOPOLOGY | SceneMutationFlags::ADDED,
        );
        handle
    }

    /// Creates a new node with a name.
    pub fn create_node_with_name(&mut self, name: &str) -> NodeHandle {
        let handle = self.nodes.inner_mut_for(self.id).insert(Node::new());
        self.names.insert(handle, Cow::Owned(name.to_string()));
        self.record_render_mutation(
            handle,
            SceneMutationFlags::TOPOLOGY | SceneMutationFlags::ADDED,
        );
        handle
    }

    /// Adds a node to the scene (defaults to root level).
    pub fn add_node(&mut self, mut node: Node) -> NodeHandle {
        node.parent = None;
        node.take_untracked_hierarchy_mutation();
        let handle = self.nodes.inner_mut_for(self.id).insert(node);
        self.root_nodes.push(handle);
        self.record_render_mutation(
            handle,
            SceneMutationFlags::TOPOLOGY
                | SceneMutationFlags::TRANSFORM
                | SceneMutationFlags::RENDERABLE
                | SceneMutationFlags::ADDED,
        );
        handle
    }

    /// Adds a node as a child of the specified parent.
    pub fn add_to_parent(&mut self, mut child: Node, parent_handle: NodeHandle) -> NodeHandle {
        child.parent = Some(parent_handle);
        child.take_untracked_hierarchy_mutation();
        let handle = self.nodes.inner_mut_for(self.id).insert(child);

        // Establish parent-child relationship
        if let Some(parent) = self.nodes.inner_mut_for(self.id).get_mut(parent_handle) {
            parent.children.push(handle);
        }
        self.record_render_mutation(
            handle,
            SceneMutationFlags::TOPOLOGY
                | SceneMutationFlags::TRANSFORM
                | SceneMutationFlags::RENDERABLE
                | SceneMutationFlags::ADDED,
        );

        handle
    }

    /// Removes a node and all its descendants recursively.
    pub fn remove_node(&mut self, handle: NodeHandle) {
        // 1. Collect all nodes to remove (depth-first)
        let mut to_remove = Vec::new();
        self.collect_subtree(handle, &mut to_remove);

        // 2. Handle parent relationship
        if let Some(node) = self.nodes.get(handle) {
            if let Some(parent_handle) = node.parent {
                if let Some(parent) = self.nodes.inner_mut_for(self.id).get_mut(parent_handle) {
                    parent.children.retain(|&h| h != handle);
                }
            } else {
                self.root_nodes.retain(|&h| h != handle);
            }
        }

        // 3. Remove all nodes and their components
        for node_handle in to_remove {
            self.transform_update_state.remove_handle(node_handle);
            self.meshes.inner_mut_for(self.id).remove(node_handle);
            self.cameras.remove(node_handle);
            self.lights.remove(node_handle);
            self.skins.inner_mut_for(self.id).remove(node_handle);
            self.morph_weights
                .inner_mut_for(self.id)
                .remove(node_handle);
            self.names.remove(node_handle);
            self.animation_mixers.remove(node_handle);
            self.rest_transforms.remove(node_handle);
            #[cfg(feature = "3dgs")]
            self.gaussian_clouds
                .inner_mut_for(self.id)
                .remove(node_handle);

            self.nodes.inner_mut_for(self.id).remove(node_handle);
            self.record_render_mutation(
                node_handle,
                SceneMutationFlags::TOPOLOGY
                    | SceneMutationFlags::RENDERABLE
                    | SceneMutationFlags::REMOVED,
            );
        }
    }

    /// Collects all nodes in a subtree (depth-first).
    fn collect_subtree(&self, handle: NodeHandle, result: &mut Vec<NodeHandle>) {
        result.push(handle);
        if let Some(node) = self.nodes.get(handle) {
            for &child in &node.children {
                self.collect_subtree(child, result);
            }
        }
    }

    /// Attaches a node as a child of another (establishes parent-child relationship).
    pub fn attach(&mut self, child_handle: NodeHandle, parent_handle: NodeHandle) {
        if child_handle == parent_handle {
            log::warn!("Cannot attach node to itself!");
            return;
        }
        let Some(child_parent) = self.nodes.get(child_handle).map(Node::parent) else {
            log::error!("Child node not found during attach!");
            return;
        };
        if !self.nodes.contains_key(parent_handle) {
            log::error!("Parent node not found during attach!");
            return;
        }
        if child_parent == Some(parent_handle) {
            return;
        }

        // The new parent must not already live below the child. Validate the
        // complete parent chain before mutating either side so rejected or
        // already-corrupt topology cannot detach a valid subtree halfway.
        let mut ancestor = Some(parent_handle);
        let mut remaining = self.nodes.len().saturating_add(1);
        while let Some(handle) = ancestor {
            if handle == child_handle {
                log::warn!("Cannot attach a node below its own descendant!");
                return;
            }
            if remaining == 0 {
                log::error!("Cannot attach through a cyclic parent chain!");
                return;
            }
            remaining -= 1;
            ancestor = self.nodes.get(handle).and_then(Node::parent);
        }

        // 1. Detach from old parent
        if let Some(old_parent) = child_parent {
            if let Some(parent) = self.nodes.inner_mut_for(self.id).get_mut(old_parent) {
                parent.children.retain(|&h| h != child_handle);
            }
        } else {
            self.root_nodes.retain(|&h| h != child_handle);
        }

        // 2. Attach to new parent
        self.nodes
            .inner_mut_for(self.id)
            .get_mut(parent_handle)
            .expect("validated parent node disappeared")
            .children
            .push(child_handle);

        // 3. Update child
        let child = self
            .nodes
            .inner_mut_for(self.id)
            .get_mut(child_handle)
            .expect("validated child node disappeared");
        child.parent = Some(parent_handle);
        child.transform.mark_dirty();
        self.record_render_mutation(
            child_handle,
            SceneMutationFlags::TOPOLOGY
                | SceneMutationFlags::TRANSFORM
                | SceneMutationFlags::RENDERABLE,
        );
    }

    /// Returns a read-only reference to a node.
    #[inline]
    pub fn node(&self, handle: NodeHandle) -> Option<&Node> {
        self.nodes.get(handle)
    }

    /// Edits a node transform while preserving exact mutation authority.
    ///
    /// Only local TRS participates in change detection. Cached local/world
    /// matrices and compatibility dirty bits remain transform-system details
    /// and cannot create false persistent render-world work.
    pub fn edit_node_transform<R>(
        &mut self,
        handle: NodeHandle,
        edit: impl FnOnce(&mut Transform) -> R,
    ) -> Option<R> {
        let (result, changed) = {
            let node = self.nodes.inner_mut_for(self.id).get_mut(handle)?;
            let mut transaction = TransformEditTransaction::new(&mut node.transform);
            let result = edit(transaction.transform());
            let changed = transaction.commit();
            (result, changed)
        };
        if changed {
            self.record_render_mutation(handle, SceneMutationFlags::TRANSFORM);
        }
        Some(result)
    }

    /// Returns a conservatively tracked mutable reference to a node.
    ///
    /// Because the reference can escape this call, the scene records both a
    /// transform and renderable mutation before returning it. Prefer
    /// [`Scene::edit_node_transform`] or [`Scene::edit_node`] when the edit can
    /// be expressed through their exact mutation boundaries.
    #[inline]
    pub fn node_mut(&mut self, handle: NodeHandle) -> Option<&mut Node> {
        if !self.nodes.contains_key(handle) {
            return None;
        }
        self.record_render_mutation(
            handle,
            SceneMutationFlags::TRANSFORM | SceneMutationFlags::RENDERABLE,
        );
        self.nodes.inner_mut_for(self.id).get_mut(handle)
    }

    /// Writes one transform component through the tracked scene authority.
    ///
    /// These narrow setters deliberately keep the public `node_mut`
    /// compatibility boundary conservative while allowing engine-owned APIs
    /// to avoid manufacturing transform/renderable work for equal values.
    #[inline]
    pub(crate) fn set_node_position_tracked(&mut self, handle: NodeHandle, position: Vec3) -> bool {
        let Some(node) = self.nodes.get(handle) else {
            return false;
        };
        if node.transform.position == position {
            return false;
        }

        self.nodes
            .inner_mut_for(self.id)
            .get_mut(handle)
            .expect("validated scene node disappeared")
            .transform
            .position = position;
        self.record_render_mutation(handle, SceneMutationFlags::TRANSFORM);
        true
    }

    #[inline]
    pub(crate) fn set_node_rotation_tracked(&mut self, handle: NodeHandle, rotation: Quat) -> bool {
        let Some(node) = self.nodes.get(handle) else {
            return false;
        };
        if node.transform.rotation == rotation {
            return false;
        }

        self.nodes
            .inner_mut_for(self.id)
            .get_mut(handle)
            .expect("validated scene node disappeared")
            .transform
            .rotation = rotation;
        self.record_render_mutation(handle, SceneMutationFlags::TRANSFORM);
        true
    }

    #[inline]
    pub(crate) fn set_node_scale_tracked(&mut self, handle: NodeHandle, scale: Vec3) -> bool {
        let Some(node) = self.nodes.get(handle) else {
            return false;
        };
        if node.transform.scale == scale {
            return false;
        }

        self.nodes
            .inner_mut_for(self.id)
            .get_mut(handle)
            .expect("validated scene node disappeared")
            .transform
            .scale = scale;
        self.record_render_mutation(handle, SceneMutationFlags::TRANSFORM);
        true
    }

    #[inline]
    pub(crate) fn set_node_visible_tracked(&mut self, handle: NodeHandle, visible: bool) -> bool {
        let Some(node) = self.nodes.get(handle) else {
            return false;
        };
        if node.visible == visible {
            return false;
        }

        self.nodes
            .inner_mut_for(self.id)
            .get_mut(handle)
            .expect("validated scene node disappeared")
            .visible = visible;
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
        true
    }

    // ========================================================================
    // Component Management API (ECS-style)
    // ========================================================================

    /// Sets the name for a node.
    pub fn set_name(&mut self, handle: NodeHandle, name: &str) {
        self.names.insert(handle, Cow::Owned(name.to_string()));
    }

    /// Returns the name of a node.
    pub fn name(&self, handle: NodeHandle) -> Option<&str> {
        self.names.get(handle).map(std::convert::AsRef::as_ref)
    }

    /// Sets the mesh component for a node.
    pub fn set_mesh(&mut self, handle: NodeHandle, mesh: Mesh) {
        self.meshes.inner_mut_for(self.id).insert(handle, mesh);
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
    }

    /// Gets a reference to the node's Mesh component
    pub fn mesh(&self, handle: NodeHandle) -> Option<&Mesh> {
        self.meshes.get(handle)
    }

    /// Gets a conservatively tracked mutable reference to the node's mesh.
    ///
    /// Prefer narrow setters or [`Scene::edit_node`] when possible; an escaping
    /// mutable mesh reference necessarily records a renderable mutation.
    pub fn mesh_mut(&mut self, handle: NodeHandle) -> Option<&mut Mesh> {
        if !self.meshes.contains_key(handle) {
            return None;
        }
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
        self.meshes.inner_mut_for(self.id).get_mut(handle)
    }

    #[inline]
    pub(crate) fn set_mesh_shadow_flags_tracked(
        &mut self,
        handle: NodeHandle,
        cast_shadows: Option<bool>,
        receive_shadows: Option<bool>,
    ) -> bool {
        let Some(mesh) = self.meshes.get(handle) else {
            return false;
        };
        let cast_changed = cast_shadows.is_some_and(|cast| mesh.cast_shadows != cast);
        let receive_changed =
            receive_shadows.is_some_and(|receive| mesh.receive_shadows != receive);
        if !cast_changed && !receive_changed {
            return false;
        }

        let mesh = self
            .meshes
            .inner_mut_for(self.id)
            .get_mut(handle)
            .expect("validated scene mesh disappeared");
        if let Some(cast) = cast_shadows {
            mesh.cast_shadows = cast;
        }
        if let Some(receive) = receive_shadows {
            mesh.receive_shadows = receive;
        }
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
        true
    }

    /// Commits renderer-owned morph history after queue submission without
    /// reporting a renderable mutation. This changes only the temporal CPU
    /// baseline and GPU-uniform update countdown; geometry, material,
    /// visibility, bounds, and shader classification remain unchanged.
    #[doc(hidden)]
    pub fn commit_submitted_morph_history(&mut self, handle: NodeHandle) -> bool {
        let Some(mesh) = self.meshes.inner_mut_for(self.id).get_mut(handle) else {
            return false;
        };
        mesh.commit_morph_history();
        true
    }

    #[cfg(feature = "3dgs")]
    /// Attaches a Gaussian splatting point cloud handle to a node.
    pub fn set_gaussian_cloud(&mut self, handle: NodeHandle, cloud: GaussianCloudHandle) {
        self.gaussian_clouds
            .inner_mut_for(self.id)
            .insert(handle, cloud);
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
    }

    #[cfg(feature = "3dgs")]
    /// Gets the Gaussian cloud handle attached to a node.
    pub fn gaussian_cloud(&self, handle: NodeHandle) -> Option<GaussianCloudHandle> {
        self.gaussian_clouds.get(handle).copied()
    }

    #[cfg(feature = "3dgs")]
    /// Creates a named node with a Gaussian splatting point cloud and adds it as a root.
    pub fn add_gaussian_cloud(
        &mut self,
        name: &str,
        cloud_handle: GaussianCloudHandle,
    ) -> NodeHandle {
        let handle = self.create_node_with_name(name);
        self.gaussian_clouds
            .inner_mut_for(self.id)
            .insert(handle, cloud_handle);
        self.root_nodes.push(handle);
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
        handle
    }

    #[cfg(feature = "3dgs")]
    /// Returns `true` if the scene contains any Gaussian splatting clouds.
    #[inline]
    pub fn has_gaussian_clouds(&self) -> bool {
        !self.gaussian_clouds.is_empty()
    }

    /// Sets the Camera component for a node
    pub fn set_camera(&mut self, handle: NodeHandle, camera: Camera) {
        self.cameras.insert(handle, camera);
    }

    /// Gets a reference to the node's Camera component
    pub fn camera(&self, handle: NodeHandle) -> Option<&Camera> {
        self.cameras.get(handle)
    }

    /// Gets a mutable reference to the node's camera component.
    ///
    /// Camera components are extracted every frame and are not part of the
    /// persistent RenderWorld mutation domain. Use [`Scene::edit_camera_bundle`]
    /// when the node transform must be edited alongside the camera.
    pub fn camera_mut(&mut self, handle: NodeHandle) -> Option<&mut Camera> {
        self.cameras.get_mut(handle)
    }

    /// Sets the Light component for a node
    pub fn set_light(&mut self, handle: NodeHandle, light: Light) {
        self.lights.insert(handle, light);
    }

    /// Gets a reference to the node's Light component
    pub fn light(&self, handle: NodeHandle) -> Option<&Light> {
        self.lights.get(handle)
    }

    /// Gets a mutable reference to the node's light component.
    ///
    /// Light components are extracted every frame and are not part of the
    /// persistent RenderWorld mutation domain. Use [`Scene::edit_light_bundle`]
    /// when the node transform must be edited alongside the light.
    pub fn light_mut(&mut self, handle: NodeHandle) -> Option<&mut Light> {
        self.lights.get_mut(handle)
    }

    /// Binds a skeleton to a node
    pub fn bind_skeleton(
        &mut self,
        handle: NodeHandle,
        skeleton_key: SkeletonKey,
        bind_mode: BindMode,
    ) {
        if let Some(node) = self.nodes.get(handle) {
            let bind_matrix_inv = node.transform.world_matrix.inverse();
            self.skins.inner_mut_for(self.id).insert(
                handle,
                SkinBinding {
                    skeleton: skeleton_key,
                    bind_mode,
                    bind_matrix_inv,
                },
            );
            self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
        }
    }

    /// Gets the node's skin binding
    pub fn skin_binding(&self, handle: NodeHandle) -> Option<&SkinBinding> {
        self.skins.get(handle)
    }

    /// Sets morph weights
    pub fn set_morph_weights(&mut self, handle: NodeHandle, weights: Vec<f32>) {
        self.morph_weights
            .inner_mut_for(self.id)
            .insert(handle, weights);
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
    }

    /// Gets morph weights
    pub fn morph_weights(&self, handle: NodeHandle) -> Option<&[f32]> {
        self.morph_weights.get(handle).map(Vec::as_slice)
    }

    /// Gets a conservatively tracked mutable slice of morph weights.
    pub fn morph_weights_mut(&mut self, handle: NodeHandle) -> Option<&mut [f32]> {
        if !self.morph_weights.contains_key(handle) {
            return None;
        }
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
        self.morph_weights
            .inner_mut_for(self.id)
            .get_mut(handle)
            .map(Vec::as_mut_slice)
    }

    /// Sets morph weights for a node (from POD data)
    pub fn set_morph_weights_from_pod(
        &mut self,
        handle: NodeHandle,
        data: &myth_animation::values::MorphWeightData,
    ) {
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
        let weights = self
            .morph_weights
            .inner_mut_for(self.id)
            .entry(handle)
            .unwrap()
            .or_default();

        if weights.len() != data.weights.len() {
            weights.resize(data.weights.len(), 0.0);
        }
        weights.copy_from_slice(&data.weights);
    }

    // ========================================================================
    // Iterate over all active lights in the scene
    // ========================================================================

    pub fn iter_active_lights(&self) -> impl Iterator<Item = (&Light, &Affine3A)> {
        self.lights.iter().filter_map(move |(node_handle, light)| {
            let node = self.nodes.get(node_handle)?;
            if node.visible {
                Some((light, &node.transform.world_matrix))
            } else {
                None
            }
        })
    }

    // ========================================================================
    // Component Bundle API
    // ========================================================================

    /// Gets a read-only `(Transform, Camera)` bundle for the main camera.
    pub fn main_camera_bundle(&self) -> Option<(&Transform, &Camera)> {
        let node_handle = self.active_camera?;
        self.camera_bundle(node_handle)
    }

    /// Gets a read-only `(Transform, Camera)` bundle for a camera node.
    pub fn camera_bundle(&self, node_handle: NodeHandle) -> Option<(&Transform, &Camera)> {
        let transform = &self.nodes.get(node_handle)?.transform;
        let camera = self.cameras.get(node_handle)?;
        Some((transform, camera))
    }

    /// Edits the active camera and its node transform through an exact tracked
    /// callback. Camera-only changes do not affect the persistent RenderWorld.
    pub fn edit_main_camera_bundle<R>(
        &mut self,
        edit: impl FnOnce(&mut Transform, &mut Camera) -> R,
    ) -> Option<R> {
        let node_handle = self.active_camera?;
        self.edit_camera_bundle(node_handle, edit)
    }

    /// Edits a camera and its node transform through an exact tracked callback.
    ///
    /// This is the preferred narrow API for per-frame camera control.
    pub fn edit_camera_bundle<R>(
        &mut self,
        node_handle: NodeHandle,
        edit: impl FnOnce(&mut Transform, &mut Camera) -> R,
    ) -> Option<R> {
        if !self.cameras.contains_key(node_handle) {
            return None;
        }

        let (result, transform_changed) = {
            let node = self.nodes.inner_mut_for(self.id).get_mut(node_handle)?;
            let camera = self.cameras.get_mut(node_handle)?;
            let mut transaction = TransformEditTransaction::new(&mut node.transform);
            let result = edit(transaction.transform(), camera);
            let changed = transaction.commit();
            (result, changed)
        };
        if transform_changed {
            self.record_render_mutation(node_handle, SceneMutationFlags::TRANSFORM);
        }
        Some(result)
    }

    /// Gets a read-only `(Transform, Light)` bundle for a light node.
    pub fn light_bundle(&self, node_handle: NodeHandle) -> Option<(&Transform, &Light)> {
        let transform = &self.nodes.get(node_handle)?.transform;
        let light = self.lights.get(node_handle)?;
        Some((transform, light))
    }

    /// Edits a light and its node transform through an exact tracked callback.
    ///
    /// Light-only changes do not affect the persistent RenderWorld. Local TRS
    /// changes are published only when the callback actually changes them.
    pub fn edit_light_bundle<R>(
        &mut self,
        node_handle: NodeHandle,
        edit: impl FnOnce(&mut Transform, &mut Light) -> R,
    ) -> Option<R> {
        if !self.lights.contains_key(node_handle) {
            return None;
        }

        let (result, transform_changed) = {
            let node = self.nodes.inner_mut_for(self.id).get_mut(node_handle)?;
            let light = self.lights.get_mut(node_handle)?;
            let mut transaction = TransformEditTransaction::new(&mut node.transform);
            let result = edit(transaction.transform(), light);
            let changed = transaction.commit();
            (result, changed)
        };
        if transform_changed {
            self.record_render_mutation(node_handle, SceneMutationFlags::TRANSFORM);
        }
        Some(result)
    }

    // ========================================================================
    // Matrix Update Pipeline
    // ========================================================================

    /// Returns the most recent transform-update sample. After an extraction
    /// flush this is the exact aggregate since the preceding extraction
    /// boundary, including simulation-side updates from the same frame.
    #[inline]
    #[must_use]
    pub const fn last_transform_update_stats(&self) -> transform_system::TransformUpdateStats {
        self.last_transform_update_stats
    }

    /// Explicit non-hot-path maintenance hook for releasing transform-system
    /// scratch capacity after a historical workload spike.
    #[inline]
    pub fn trim_transform_update_capacity(&mut self) -> bool {
        self.transform_update_state.trim_idle_capacity()
    }

    /// Updates only dirty transform subtrees, falling back to a full traversal
    /// after untracked raw node-storage mutation.
    pub fn update_matrix_world(&mut self) -> transform_system::TransformUpdateStats {
        let stats = self.update_matrix_world_impl(true);
        self.pending_render_transform_update_stats.merge(stats);
        self.pending_render_transform_update_stats.retained_bytes = stats.retained_bytes;
        self.last_transform_update_stats = stats;
        stats
    }

    /// Materializes transform mutations before a renderer extracts the Scene.
    ///
    /// Unlike the simulation update, this does not consume compatibility
    /// `previous_world_matrix` settlement left by an earlier update in the
    /// same frame. A completely clean Scene is O(1); its returned and published
    /// work sample describes only this flush. `last_transform_update_stats`
    /// publishes the aggregate since the preceding extraction boundary so RDG
    /// diagnostics also include simulation-side hierarchy work from this frame.
    pub fn flush_dirty_world_matrices(&mut self) -> transform_system::TransformUpdateStats {
        let stats = self.update_matrix_world_impl(false);
        let mut aggregate = std::mem::take(&mut self.pending_render_transform_update_stats);
        aggregate.merge(stats);
        aggregate.retained_bytes = stats.retained_bytes;
        self.last_transform_update_stats = aggregate;
        stats
    }

    fn update_matrix_world_impl(
        &mut self,
        settle_previous: bool,
    ) -> transform_system::TransformUpdateStats {
        let nodes_storage_generation = self.nodes.external_mutation_generation();
        if nodes_storage_generation
            != self
                .transform_update_state
                .observed_nodes_storage_generation
        {
            self.transform_update_state.full_traversal_required = true;
        }

        let mut untracked_hierarchy = false;
        for &handle in &self.transform_update_state.dirty_roots {
            if self
                .nodes
                .inner_mut_for(self.id)
                .get_mut(handle)
                .is_some_and(Node::take_untracked_hierarchy_mutation)
            {
                self.mutation_journal.record(
                    handle,
                    SceneMutationFlags::TOPOLOGY
                        | SceneMutationFlags::TRANSFORM
                        | SceneMutationFlags::RENDERABLE,
                );
                untracked_hierarchy = true;
            }
        }
        if untracked_hierarchy {
            self.mutation_journal
                .force_full_rebuild(SceneFullRebuildReasons::UNTRACKED_HIERARCHY);
            self.transform_update_state.full_traversal_required = true;
        }

        let (dirty_roots_examined, dirty_root_parent_links_examined) = self
            .transform_update_state
            .coalesce_dirty_roots(self.nodes.inner());
        let mut stats = transform_system::TransformUpdateStats {
            dirty_roots_examined,
            dirty_root_parent_links_examined,
            ..Default::default()
        };
        if !settle_previous
            && !self.transform_update_state.full_traversal_required
            && self.transform_update_state.dirty_roots.is_empty()
        {
            stats.retained_bytes = self.transform_update_state.estimated_retained_bytes();
            return stats;
        }
        if self.transform_update_state.full_traversal_required {
            let update = transform_system::update_hierarchy_iterative_with_scratch(
                self.nodes.inner_mut_for(self.id),
                &mut self.cameras,
                &self.root_nodes,
                &mut self.transform_update_state.update_stack,
            );
            stats.merge(update);
            stats.full_traversal = true;
            if stats.untracked_hierarchy_mutations != 0 {
                self.mutation_journal
                    .force_full_rebuild(SceneFullRebuildReasons::UNTRACKED_HIERARCHY);
            }
            self.transform_update_state.full_traversal_required = false;
            self.transform_update_state
                .observed_nodes_storage_generation = self.nodes.external_mutation_generation();
            self.transform_update_state.clear_dirty_roots();
            self.transform_update_state.settle_roots.clear();
            self.transform_update_state.settle_full_traversal = stats.changed_world_matrices != 0;
            stats.retained_bytes = self.transform_update_state.estimated_retained_bytes();
            return stats;
        }

        if settle_previous && self.transform_update_state.settle_full_traversal {
            let settle = transform_system::settle_previous_subtrees_with_scratch(
                self.nodes.inner_mut_for(self.id),
                &self.root_nodes,
                &mut self.transform_update_state.settle_stack,
            );
            stats.merge(settle);
            self.transform_update_state.settle_full_traversal = false;
        } else if settle_previous && !self.transform_update_state.settle_roots.is_empty() {
            let settle = transform_system::settle_previous_subtrees_with_scratch(
                self.nodes.inner_mut_for(self.id),
                &self.transform_update_state.settle_roots,
                &mut self.transform_update_state.settle_stack,
            );
            stats.merge(settle);
            self.transform_update_state.settle_roots.clear();
        }

        if !self.transform_update_state.dirty_roots.is_empty() {
            let update = transform_system::update_subtrees_iterative_with_scratch(
                self.nodes.inner_mut_for(self.id),
                &mut self.cameras,
                &self.transform_update_state.dirty_roots,
                &mut self.transform_update_state.update_stack,
            );
            stats.merge(update);
            if settle_previous {
                self.transform_update_state
                    .finish_dirty_update(update.changed_world_matrices != 0);
            } else {
                self.transform_update_state
                    .finish_dirty_update_preserving_settles(
                        self.nodes.inner(),
                        update.changed_world_matrices != 0,
                    );
            }
        }

        stats.retained_bytes = self.transform_update_state.estimated_retained_bytes();
        stats
    }

    /// Updates world matrices for a specified subtree
    pub fn update_subtree(
        &mut self,
        root_handle: NodeHandle,
    ) -> transform_system::TransformUpdateStats {
        self.transform_update_state
            .mark_dirty_root(self.nodes.inner(), root_handle);
        self.update_matrix_world()
    }

    // ========================================================================
    // Resource Management API
    // ========================================================================

    pub fn add_mesh(&mut self, mesh: Mesh) -> NodeHandle {
        let node_handle = self.create_node_with_name(&mesh.name);
        self.meshes.inner_mut_for(self.id).insert(node_handle, mesh);
        self.root_nodes.push(node_handle);
        self.record_render_mutation(node_handle, SceneMutationFlags::RENDERABLE);
        node_handle
    }

    pub fn add_mesh_to_parent(&mut self, mesh: Mesh, parent: NodeHandle) -> NodeHandle {
        let node_handle = self.create_node_with_name(&mesh.name);
        self.meshes.inner_mut_for(self.id).insert(node_handle, mesh);
        self.record_render_mutation(node_handle, SceneMutationFlags::RENDERABLE);
        self.attach(node_handle, parent);
        node_handle
    }

    pub fn add_skeleton(&mut self, skeleton: Skeleton) -> SkeletonKey {
        self.skeleton_pool.inner_mut_for(self.id).insert(skeleton)
    }

    pub fn add_camera(&mut self, camera: Camera) -> NodeHandle {
        let node_handle = self.create_node_with_name("Camera");
        self.cameras.insert(node_handle, camera);
        self.root_nodes.push(node_handle);
        node_handle
    }

    pub fn add_camera_to_parent(&mut self, camera: Camera, parent: NodeHandle) -> NodeHandle {
        let node_handle = self.create_node_with_name("Camera");
        self.cameras.insert(node_handle, camera);
        self.attach(node_handle, parent);
        node_handle
    }

    pub fn add_light(&mut self, light: Light) -> NodeHandle {
        let node_handle = self.create_node_with_name("Light");
        self.lights.insert(node_handle, light);
        self.root_nodes.push(node_handle);
        node_handle
    }

    pub fn add_light_to_parent(&mut self, light: Light, parent: NodeHandle) -> NodeHandle {
        let node_handle = self.create_node_with_name("Light");
        self.lights.insert(node_handle, light);
        self.attach(node_handle, parent);
        node_handle
    }

    pub fn mark_as_split_primitive(&mut self, handle: NodeHandle) {
        self.split_primitive_tags
            .inner_mut_for(self.id)
            .insert(handle, SplitPrimitiveTag);
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
    }

    /// Synchronizes shader macro definitions based on the current scene state.
    fn sync_shader_defines(&mut self) {
        let current_env_version = self.environment.version();

        // Only recompute if the environment version has changed since the last computation
        if self.last_env_version != current_env_version {
            let mut defines = ShaderDefines::new();

            // Recompute logic
            if self.environment.has_env_map() {
                defines.set("HAS_ENV_MAP", "1");
            }
            // ... additional defines based on scene state can be added here ...

            self.shader_defines = defines;
            self.last_env_version = current_env_version;
        }
    }

    /// Computes the scene's shader macro definitions
    ///
    /// Uses internal caching mechanism, only recalculates when Environment version changes.
    pub fn shader_defines(&self) -> &ShaderDefines {
        &self.shader_defines
    }

    // ========================================================================
    // Scene Update and Logic System
    // ========================================================================

    pub fn add_logic<L: SceneLogic>(&mut self, logic: L) {
        self.logics.push(Box::new(logic));
    }

    /// Shortcut method: Add closure logic (for quick prototyping)
    pub fn on_update<F>(&mut self, f: F)
    where
        F: FnMut(&mut Scene, &Input, f32) + Send + Sync + 'static,
    {
        self.add_logic(CallbackLogic(f));
    }

    /// Updates scene state (called every frame)
    pub fn update(&mut self, input: &Input, dt: f32) {
        // 1. Execute user scripts (Gameplay)
        let mut logics = std::mem::take(&mut self.logics);
        for logic in &mut logics {
            logic.update(self, input, dt);
        }
        self.logics.append(&mut logics);

        // 2. Animation system update (modifies node Transform)
        {
            let mut mixers = std::mem::take(&mut self.animation_mixers);
            for (_handle, mixer) in &mut mixers {
                mixer.update(dt, self);
            }
            self.animation_mixers = mixers;
        }

        // 3. Execute internal engine systems (Transform, Skeleton, Morph)
        self.update_matrix_world();
        self.update_skeletons();
        self.sync_morph_weights();
        self.sync_shader_defines();
    }

    pub fn update_skeletons(&mut self) {
        let mut tasks = Vec::new();

        for (node_handle, binding) in &self.skins {
            if let Some(node) = self.nodes.get(node_handle) {
                let root_inv = match binding.bind_mode {
                    BindMode::Attached => node.transform.world_matrix.inverse(),
                    BindMode::Detached => binding.bind_matrix_inv,
                };
                tasks.push((binding.skeleton, root_inv));
            }
        }

        for (skeleton_id, root_inv) in tasks {
            if let Some(skeleton) = self
                .skeleton_pool
                .inner_mut_for(self.id)
                .get_mut(skeleton_id)
            {
                skeleton.compute_joint_matrices(&self.nodes, root_inv);
                // Lazy compute bounding box (only computed when first needed)
                if skeleton.local_bounds.is_none() {
                    skeleton.compute_local_bounds(&self.nodes);
                }
            }
        }
    }

    /// Copies scene-level morph weights into their target meshes.
    ///
    /// Packing the GPU-facing morph uniform is deferred until a renderer owns
    /// a viable frame target, and history is committed only after submission.
    /// A scene may run multiple simulation updates before one render without
    /// consuming the two-frame velocity update window twice.
    pub fn sync_morph_weights(&mut self) {
        for (handle, weights) in &self.morph_weights {
            if weights.is_empty() {
                continue;
            }

            let weights_slice = weights.as_slice();

            if let Some(mesh) = self.meshes.inner_mut_for(self.id).get_mut(handle) {
                mesh.set_morph_target_influences(weights_slice);
            } else if let Some(node) = self.nodes.get(handle) {
                for &child_handle in &node.children {
                    // Broadcast to child nodes that have SplitPrimitiveTag
                    if self.split_primitive_tags.contains_key(child_handle)
                        && let Some(child_mesh) =
                            self.meshes.inner_mut_for(self.id).get_mut(child_handle)
                    {
                        child_mesh.set_morph_target_influences(weights_slice);
                    }
                }
            }
        }
    }

    pub fn main_camera_node(&self) -> Option<&Node> {
        let handle = self.active_camera?;
        self.node(handle)
    }

    // ========================================================================
    // Background API
    // ========================================================================

    /// Sets the background to a solid color.
    pub fn set_background_color(&mut self, r: f32, g: f32, b: f32) {
        self.background.set_mode(BackgroundMode::color(r, g, b));
    }

    // ========================================================================
    // High-Level Helpers (exact node editor, builder)
    // ========================================================================

    /// Returns an exact, chainable editor for the given node.
    ///
    /// Silently no-ops if the handle is stale, avoiding `unwrap()`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// scene.edit_node(handle)
    ///     .set_position(0.0, 3.0, 0.0)
    ///     .set_scale(2.0)
    ///     .look_at(Vec3::ZERO);
    /// ```
    pub fn edit_node(&mut self, handle: NodeHandle) -> NodeEditor<'_> {
        NodeEditor::new(self, handle)
    }

    /// Starts building a node
    pub fn build_node(&mut self, name: &str) -> NodeBuilder<'_> {
        NodeBuilder::new(self, name)
    }

    /// Finds a node by name
    pub fn find_node_by_name(&self, name: &str) -> Option<NodeHandle> {
        for (handle, node_name) in &self.names {
            if node_name.as_ref() == name {
                return Some(handle);
            }
        }
        None
    }

    /// Returns the world transform of a node.
    pub fn world_transform(&self, handle: NodeHandle) -> Option<Affine3A> {
        self.nodes
            .get(handle)
            .map(|node| node.transform.world_matrix)
    }

    /// Plays a specific animation clip on the node (if an AnimationMixer is present)
    pub fn play_animation(&mut self, node_handle: NodeHandle, clip_name: &str) {
        if let Some(mixer) = self.animation_mixers.get_mut(node_handle) {
            mixer.play(clip_name);
        } else {
            log::warn!("No animation mixer found for node {node_handle:?}");
        }
    }

    /// Plays any animation on the node (used for simple cases where clip name is not important)
    pub fn play_if_any_animation(&mut self, node_handle: NodeHandle) {
        if let Some(mixer) = self.animation_mixers.get_mut(node_handle) {
            mixer
                .any_action()
                .map(myth_animation::mixer::ActionControl::play);
        } else {
            log::info!("No animation mixer found for node {node_handle:?}");
        }
    }

    // ========================================================================
    // Bounding Box Queries
    // ========================================================================

    /// Computes the world-space bounding box of a single node (not including children).
    ///
    /// `query` implements [`GeometryQuery`] to map geometry handles to local-space bounding boxes.
    fn node_world_bounds(
        &self,
        node_handle: NodeHandle,
        query: &impl crate::GeometryQuery,
    ) -> Option<myth_resources::BoundingBox> {
        let node = self.node(node_handle)?;
        if !node.visible {
            return None;
        }
        let mesh = self.meshes.get(node_handle)?;
        if !mesh.visible {
            return None;
        }

        // When there's a skeleton binding, use the skeleton's bounding box
        if let Some(skeleton_binding) = self.skins.get(node_handle)
            && let Some(skeleton) = self.skeleton_pool.get(skeleton_binding.skeleton)
        {
            return skeleton.compute_tight_world_bounds(&self.nodes);
        }

        // Otherwise compute from the geometry's static bounding box
        let local_bbox = query.get_geometry_bbox(mesh.geometry)?;
        Some(local_bbox.transform(&node.transform.world_matrix))
    }

    /// Recursively computes the world-space bounding box enclosing a node and all its descendants.
    ///
    /// `query` implements [`crate::GeometryQuery`] to map geometry handles to local-space bounding boxes.
    pub fn subtree_world_bounds(
        &self,
        node_handle: NodeHandle,
        query: &impl crate::GeometryQuery,
    ) -> Option<myth_resources::BoundingBox> {
        let mut combined_bbox = self.node_world_bounds(node_handle, query);
        let node = self.node(node_handle)?;
        for &child_handle in &node.children {
            if let Some(child_bbox) = self.subtree_world_bounds(child_handle, query) {
                combined_bbox = match combined_bbox {
                    Some(existing) => Some(existing.union(&child_bbox)),
                    None => Some(child_bbox),
                };
            }
        }

        combined_bbox
    }
}

// ============================================================================
// NodeBuilder
// ============================================================================

pub struct NodeBuilder<'a> {
    scene: &'a mut Scene,
    handle: NodeHandle,
    parent: Option<NodeHandle>,
    mesh: Option<Mesh>,
}

impl<'a> NodeBuilder<'a> {
    pub fn new(scene: &'a mut Scene, name: &str) -> Self {
        let handle = scene.create_node_with_name(name);
        Self {
            scene,
            handle,
            parent: None,
            mesh: None,
        }
    }

    #[must_use]
    pub fn with_position(self, x: f32, y: f32, z: f32) -> Self {
        if let Some(node) = self.scene.node_mut(self.handle) {
            node.transform.position = glam::Vec3::new(x, y, z);
        }
        self
    }

    #[must_use]
    pub fn with_scale(self, s: f32) -> Self {
        if let Some(node) = self.scene.node_mut(self.handle) {
            node.transform.scale = glam::Vec3::splat(s);
        }
        self
    }

    #[must_use]
    pub fn with_parent(mut self, parent: NodeHandle) -> Self {
        self.parent = Some(parent);
        self
    }

    #[must_use]
    pub fn with_mesh(mut self, mesh: Mesh) -> Self {
        self.mesh = Some(mesh);
        self
    }

    pub fn build(self) -> NodeHandle {
        let handle = self.handle;

        // Set Mesh component
        if let Some(mesh) = self.mesh {
            self.scene.set_mesh(handle, mesh);
        }

        // Handle parent-child relationship
        if let Some(parent) = self.parent {
            self.scene.attach(handle, parent);
        } else {
            self.scene.root_nodes.push(handle);
        }

        handle
    }
}

// ============================================================================
// AnimationTarget Implementation
// ============================================================================

impl AnimationTarget for Scene {
    fn node_children(&self, handle: NodeHandle) -> Option<Vec<NodeHandle>> {
        self.nodes.get(handle).map(|n| n.children().to_vec())
    }

    fn node_name(&self, handle: NodeHandle) -> Option<String> {
        self.names.get(handle).map(|n| n.as_ref().to_string())
    }

    fn node_transform(&self, handle: NodeHandle) -> Option<Transform> {
        self.nodes.get(handle).map(|n| n.transform)
    }

    fn set_node_position(&mut self, handle: NodeHandle, position: Vec3) {
        self.set_node_position_tracked(handle, position);
    }

    fn set_node_rotation(&mut self, handle: NodeHandle, rotation: Quat) {
        self.set_node_rotation_tracked(handle, rotation);
    }

    fn set_node_scale(&mut self, handle: NodeHandle, scale: Vec3) {
        self.set_node_scale_tracked(handle, scale);
    }

    fn mark_node_dirty(&mut self, _handle: NodeHandle) {
        // AnimationMixer invokes this compatibility hook unconditionally
        // after its setters. The narrow setters above already schedule an
        // exact dirty root when TRS really changes, so recording/scheduling
        // here would turn paused or constant animation into false per-frame
        // RenderWorld work. It must be a no-op for Scene: even setting the
        // legacy Transform force bit would defer false work into a later full
        // or independently scheduled traversal.
    }

    fn has_rest_transform(&self, handle: NodeHandle) -> bool {
        self.rest_transforms.contains_key(handle)
    }

    fn rest_transform(&self, handle: NodeHandle) -> Option<Transform> {
        self.rest_transforms.get(handle).copied()
    }

    fn store_rest_transform(&mut self, handle: NodeHandle, transform: Transform) {
        self.rest_transforms.insert(handle, transform);
    }

    fn morph_weights_mut(&mut self, handle: NodeHandle) -> &mut Vec<f32> {
        self.record_render_mutation(handle, SceneMutationFlags::RENDERABLE);
        self.morph_weights
            .inner_mut_for(self.id)
            .entry(handle)
            .unwrap()
            .or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use myth_resources::{GeometryHandle, MaterialHandle};
    use slotmap::KeyData;

    #[test]
    fn simulation_sync_does_not_consume_render_morph_window() {
        let mut scene = Scene::new();
        let geometry = GeometryHandle::from(KeyData::from_ffi(0x0000_0001_0000_0001));
        let material = MaterialHandle::from(KeyData::from_ffi(0x0000_0001_0000_0002));
        let mut mesh = Mesh::new(geometry, material);
        mesh.init_morph_targets(1, 1);
        let node = scene.add_mesh(mesh);
        scene.set_morph_weights(node, vec![1.0]);

        // Multiple simulation ticks before one render only synchronize the
        // target value; they must not latch the render-frame history.
        scene.sync_morph_weights();
        scene.sync_morph_weights();

        let mesh = scene.meshes.get_mut(node).expect("test mesh missing");
        mesh.update_morph_uniforms();
        let (first_current, first_previous) = {
            let uniforms = mesh.morph_uniforms();
            (uniforms.weights[0].x, uniforms.prev_weights[0].x)
        };
        assert!((first_current - 1.0).abs() <= f32::EPSILON);
        assert!(first_previous.abs() <= f32::EPSILON);

        scene.sync_morph_weights();
        let mesh = scene.meshes.get_mut(node).expect("test mesh missing");
        mesh.update_morph_uniforms();
        let second_previous = mesh.morph_uniforms().prev_weights[0].x;
        assert!((second_previous - 1.0).abs() <= f32::EPSILON);
    }

    #[cfg(feature = "3dgs")]
    #[test]
    fn remove_node_clears_gaussian_cloud_component() {
        let mut scene = Scene::new();
        let cloud = GaussianCloudHandle::from(slotmap::KeyData::from_ffi(0x0000_0001_0000_0001));
        let node = scene.add_gaussian_cloud("cloud", cloud);

        assert_eq!(scene.gaussian_cloud(node), Some(cloud));

        scene.remove_node(node);

        assert!(scene.gaussian_cloud(node).is_none());
        assert!(!scene.gaussian_clouds.contains_key(node));
    }
}
