use glam::Affine3A;
use myth_core::{NodeHandle, Transform};

/// A minimal scene node containing only essential hot data.
///
/// # Design Principles
///
/// - Only keeps data that must be traversed every frame (hierarchy and transform)
/// - Other attributes (Mesh, Camera, Light, Skin, etc.) are stored in Scene's component maps
/// - Improves CPU cache hit rate by keeping nodes small and contiguous
///
/// # Hierarchy
///
/// Nodes form a tree structure through parent-child relationships:
/// - `parent`: Optional handle to parent node (None for root nodes)
/// - `children`: List of child node handles
///
/// # Transform
///
/// Each node has a [`Transform`] component that manages:
/// - Local position, rotation, and scale
/// - Cached local and world matrices
/// - Dirty flag for efficient updates
#[derive(Debug, Clone)]
pub struct Node {
    // === Core Hierarchy ===
    /// Parent node handle (None for root nodes)
    pub(crate) parent: Option<NodeHandle>,
    /// Child node handles
    pub(crate) children: Vec<NodeHandle>,

    // === Core Spatial Data ===
    /// Transform component (hot data accessed every frame)
    pub transform: Transform,

    // === Core State ===
    /// Visibility flag for culling
    pub visible: bool,

    // Low-level hierarchy setters cannot update Scene's topology journal.
    // Scene audits this bit for nodes returned through node_mut().
    pub(crate) untracked_hierarchy_mutation: bool,
}

impl Node {
    /// Creates a new node with default transform and visibility.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            transform: Transform::new(),
            visible: true,
            untracked_hierarchy_mutation: false,
        }
    }

    /// Returns the parent node handle, if any.
    #[inline]
    #[must_use]
    pub fn parent(&self) -> Option<NodeHandle> {
        self.parent
    }

    /// Returns a read-only slice of child node handles.
    #[inline]
    #[must_use]
    pub fn children(&self) -> &[NodeHandle] {
        &self.children
    }

    /// Sets the parent of this node. Prefer using [`crate::Scene::attach`] which
    /// keeps both parent and child in sync. This is exposed for low-level
    /// construction (e.g., building hierarchies outside of a `Scene`).
    #[inline]
    pub fn set_parent(&mut self, parent: Option<NodeHandle>) {
        self.parent = parent;
        self.untracked_hierarchy_mutation = true;
    }

    /// Appends a child handle. Prefer using [`crate::Scene::attach`] which keeps
    /// both parent and child in sync. This is exposed for low-level
    /// construction (e.g., building hierarchies outside of a `Scene`).
    #[inline]
    pub fn push_child(&mut self, child: NodeHandle) {
        self.children.push(child);
        self.untracked_hierarchy_mutation = true;
    }

    #[inline]
    pub(crate) fn take_untracked_hierarchy_mutation(&mut self) -> bool {
        std::mem::take(&mut self.untracked_hierarchy_mutation)
    }

    /// Returns a reference to the world transformation matrix.
    ///
    /// This matrix transforms local coordinates to world coordinates.
    /// It is automatically updated by the transform system each frame.
    #[inline]
    #[must_use]
    pub fn world_matrix(&self) -> &Affine3A {
        &self.transform.world_matrix
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}
