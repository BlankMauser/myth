//! Transform System
//!
//! Handles scene graph matrix hierarchy updates, decoupled from Scene to avoid
//! borrow conflicts. This is an independent system that only needs to borrow
//! the nodes `SlotMap` and `root_nodes` list.
//!
//! # Parallelization Strategy
//!
//! Scene graph updates can be parallelized in batches by level (BFS order):
//! 1. Level 0: All root nodes (no dependencies, can run in parallel)
//! 2. Level 1: Direct children of root nodes (depend on level 0, parallelizable within level)
//! 3. ...and so on
//!
//! Use `build_level_order_batches` to precompute level batches, then you can:
//! - Execute batches sequentially on a single thread
//! - Use rayon to parallelize nodes within each batch

use glam::Affine3A;
use slotmap::{SlotMap, SparseSecondaryMap};

use crate::camera::Camera;
use crate::node::Node;
use myth_core::NodeHandle;

/// Work performed by one transform-system update.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransformUpdateStats {
    pub visited_nodes: usize,
    pub updated_world_matrices: usize,
    pub changed_world_matrices: usize,
    pub settled_previous_matrices: usize,
    /// Exact dirty candidates considered by the retained root coalescer.
    pub dirty_roots_examined: usize,
    /// Parent links followed while reducing dirty candidates to disjoint roots.
    pub dirty_root_parent_links_examined: usize,
    pub untracked_hierarchy_mutations: usize,
    pub full_traversal: bool,
    pub retained_bytes: usize,
}

impl TransformUpdateStats {
    #[inline]
    pub(crate) fn merge(&mut self, other: Self) {
        self.visited_nodes = self.visited_nodes.saturating_add(other.visited_nodes);
        self.updated_world_matrices = self
            .updated_world_matrices
            .saturating_add(other.updated_world_matrices);
        self.changed_world_matrices = self
            .changed_world_matrices
            .saturating_add(other.changed_world_matrices);
        self.settled_previous_matrices = self
            .settled_previous_matrices
            .saturating_add(other.settled_previous_matrices);
        self.dirty_roots_examined = self
            .dirty_roots_examined
            .saturating_add(other.dirty_roots_examined);
        self.dirty_root_parent_links_examined = self
            .dirty_root_parent_links_examined
            .saturating_add(other.dirty_root_parent_links_examined);
        self.untracked_hierarchy_mutations = self
            .untracked_hierarchy_mutations
            .saturating_add(other.untracked_hierarchy_mutations);
        self.full_traversal |= other.full_traversal;
    }
}

/// Level-order batch information for parallelization.
#[derive(Debug, Default)]
pub struct LevelOrderBatches {
    /// Node lists per level. batches\[0\] contains root nodes, batches\[1\] contains first-level children, etc.
    pub batches: Vec<Vec<NodeHandle>>,
    /// Parent node handle for each node (used to lookup parent world matrix).
    pub parent_indices: Vec<Option<NodeHandle>>,
}

impl LevelOrderBatches {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears and reuses memory.
    pub fn clear(&mut self) {
        for batch in &mut self.batches {
            batch.clear();
        }
        self.parent_indices.clear();
    }

    /// Gets the total number of nodes.
    #[must_use]
    pub fn total_nodes(&self) -> usize {
        self.batches.iter().map(std::vec::Vec::len).sum()
    }

    /// Gets the hierarchy depth.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.batches.len()
    }
}

/// Builds level-ordered node batches (BFS order).
///
/// This function flattens the scene graph by level so that:
/// - Nodes at the same level are in the same batch and can be processed in parallel
/// - Different levels are processed sequentially to ensure parents are updated before children
///
/// # Parallelization Strategy
/// ```ignore
/// for batch in batches.iter() {
///     // Nodes within each batch can be processed in parallel
///     batch.par_iter().for_each(|node_handle| {
///         // Safe: nodes at the same level don't depend on each other
///         update_single_node(...);
///     });
/// }
/// ```
pub fn build_level_order_batches(
    nodes: &SlotMap<NodeHandle, Node>,
    roots: &[NodeHandle],
    output: &mut LevelOrderBatches,
) {
    output.clear();

    if roots.is_empty() {
        return;
    }

    // Level 0: root nodes
    let mut current_level: Vec<NodeHandle> = roots.to_vec();

    while !current_level.is_empty() {
        // Collect next level nodes
        let mut next_level = Vec::new();

        for &node_handle in &current_level {
            if let Some(node) = nodes.get(node_handle) {
                for &child_handle in &node.children {
                    next_level.push(child_handle);
                }
            }
        }

        // Save current level
        if output.batches.len() <= output.depth() {
            output.batches.push(current_level);
        } else {
            let idx = output.depth();
            output.batches[idx] = current_level;
        }

        current_level = next_level;
    }

    // Remove empty batches
    output.batches.retain(|b| !b.is_empty());
}

/// Uses precomputed batches for hierarchy updates (prepared for parallelization).
///
/// This is currently a single-threaded version, but the structure is ready for rayon parallelization.
/// To enable parallelization, simply change the inner loop to `par_iter`.
pub fn update_hierarchy_batched(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SparseSecondaryMap<NodeHandle, Camera>,
    batches: &LevelOrderBatches,
) {
    for (level, batch) in batches.batches.iter().enumerate() {
        for &node_handle in batch {
            // Get parent world matrix
            let parent_world = if let Some(node) = nodes.get(node_handle) {
                if let Some(parent_handle) = node.parent {
                    nodes
                        .get(parent_handle)
                        .map_or(Affine3A::IDENTITY, |p| p.transform.world_matrix)
                } else {
                    Affine3A::IDENTITY
                }
            } else {
                continue;
            };

            // Update current node
            if let Some(node) = nodes.get_mut(node_handle) {
                node.transform.previous_world_matrix = node.transform.world_matrix;

                let local_changed = node.transform.update_local_matrix();
                let parent_changed = level > 0;

                if local_changed || parent_changed {
                    let new_world = parent_world * *node.transform.local_matrix();
                    node.transform.set_world_matrix(new_world);

                    // Synchronously update camera
                    if let Some(camera) = cameras.get_mut(node_handle) {
                        camera.update_view_projection(&new_world);
                    }
                }
            }
        }
    }
}

/// Updates world matrices for the entire scene hierarchy.
///
/// # Arguments
///
/// * `nodes` - Mutable reference to the nodes `SlotMap`
/// * `cameras` - Mutable reference to the cameras `SlotMap` (for synchronous view-projection matrix updates)
/// * `camera_components` - Component mapping from nodes to cameras
/// * `roots` - List of root node handles
///
pub fn update_hierarchy(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SparseSecondaryMap<NodeHandle, Camera>,
    roots: &[NodeHandle],
) {
    for &root_handle in roots {
        update_transform_recursive(nodes, cameras, root_handle, Affine3A::IDENTITY, false);
    }
}

/// Recursively updates node transforms (iterative optimized version).
///
/// Uses an explicit stack instead of recursive calls to avoid stack overflow risks
/// in deeply nested scenes, while also reducing repeated borrow overhead.
pub fn update_hierarchy_iterative(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SparseSecondaryMap<NodeHandle, Camera>,
    roots: &[NodeHandle],
) -> TransformUpdateStats {
    let mut stack: Vec<(NodeHandle, Affine3A, bool)> = Vec::with_capacity(64);

    update_hierarchy_iterative_with_scratch(nodes, cameras, roots, &mut stack)
}

pub(crate) fn update_hierarchy_iterative_with_scratch(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SparseSecondaryMap<NodeHandle, Camera>,
    roots: &[NodeHandle],
    stack: &mut Vec<(NodeHandle, Affine3A, bool)>,
) -> TransformUpdateStats {
    stack.clear();

    for &root_handle in roots.iter().rev() {
        stack.push((root_handle, Affine3A::IDENTITY, false));
    }

    update_from_stack(nodes, cameras, stack)
}

fn update_from_stack(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SparseSecondaryMap<NodeHandle, Camera>,
    stack: &mut Vec<(NodeHandle, Affine3A, bool)>,
) -> TransformUpdateStats {
    let mut stats = TransformUpdateStats::default();
    while let Some((node_handle, parent_world_matrix, parent_changed)) = stack.pop() {
        stats.visited_nodes += 1;
        // --- Phase 1: Mutable borrow, handle update logic ---
        let (current_world, world_needs_update) = {
            let Some(node) = nodes.get_mut(node_handle) else {
                continue;
            };

            if node.take_untracked_hierarchy_mutation() {
                stats.untracked_hierarchy_mutations += 1;
            }

            node.transform.previous_world_matrix = node.transform.world_matrix;

            let local_changed = node.transform.update_local_matrix();
            let world_needs_update = local_changed || parent_changed;

            if world_needs_update {
                stats.updated_world_matrices += 1;
                let new_world = parent_world_matrix * *node.transform.local_matrix();
                if new_world != node.transform.world_matrix {
                    stats.changed_world_matrices += 1;
                }
                node.transform.set_world_matrix(new_world);

                if let Some(camera) = cameras.get_mut(node_handle) {
                    camera.update_view_projection(&new_world);
                }
            }

            (node.transform.world_matrix, world_needs_update)
        };
        // Closure/scope ends here, `node`'s mutable borrow lifetime ends

        // --- Phase 2: Immutable borrow, efficiently collect child nodes ---
        // [Fix] Move lookup outside the loop, only perform one SlotMap lookup
        if let Some(node) = nodes.get(node_handle) {
            // Directly iterate over slice, no need for repeated get
            for &child_handle in node.children.iter().rev() {
                stack.push((child_handle, current_world, world_needs_update));
            }
        }
    }
    stats
}

/// Updates a coalesced set of dirty roots and all of their descendants.
pub fn update_subtrees_iterative(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SparseSecondaryMap<NodeHandle, Camera>,
    roots: &[NodeHandle],
) -> TransformUpdateStats {
    let mut stack = Vec::with_capacity(64.max(roots.len()));
    update_subtrees_iterative_with_scratch(nodes, cameras, roots, &mut stack)
}

pub(crate) fn update_subtrees_iterative_with_scratch(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SparseSecondaryMap<NodeHandle, Camera>,
    roots: &[NodeHandle],
    stack: &mut Vec<(NodeHandle, Affine3A, bool)>,
) -> TransformUpdateStats {
    stack.clear();
    for &root_handle in roots.iter().rev() {
        let parent_world = nodes
            .get(root_handle)
            .and_then(|node| node.parent)
            .and_then(|parent| nodes.get(parent))
            .map_or(Affine3A::IDENTITY, |parent| parent.transform.world_matrix);
        stack.push((root_handle, parent_world, false));
    }
    update_from_stack(nodes, cameras, stack)
}

/// Latches current world matrices into the legacy previous-world cache for a
/// coalesced set of roots without recomputing world transforms.
pub fn settle_previous_subtrees(
    nodes: &mut SlotMap<NodeHandle, Node>,
    roots: &[NodeHandle],
) -> TransformUpdateStats {
    let mut stack = Vec::with_capacity(64.max(roots.len()));
    settle_previous_subtrees_with_scratch(nodes, roots, &mut stack)
}

pub(crate) fn settle_previous_subtrees_with_scratch(
    nodes: &mut SlotMap<NodeHandle, Node>,
    roots: &[NodeHandle],
    stack: &mut Vec<NodeHandle>,
) -> TransformUpdateStats {
    stack.clear();
    stack.extend(roots.iter().rev().copied());
    let mut stats = TransformUpdateStats::default();

    while let Some(handle) = stack.pop() {
        {
            let Some(node) = nodes.get_mut(handle) else {
                continue;
            };
            stats.visited_nodes += 1;
            if node.transform.previous_world_matrix != node.transform.world_matrix {
                node.transform.previous_world_matrix = node.transform.world_matrix;
                stats.settled_previous_matrices += 1;
            }
        }
        if let Some(node) = nodes.get(handle) {
            stack.extend(node.children.iter().rev().copied());
        }
    }

    stats
}

/// Recursively updates a single node and its subtree (original recursive version kept for reference)
fn update_transform_recursive(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SparseSecondaryMap<NodeHandle, Camera>,
    node_handle: NodeHandle,
    parent_world_matrix: Affine3A,
    parent_changed: bool,
) {
    // Phase 1: Process current node
    let (current_world_matrix, children_handles, world_needs_update) = {
        let Some(node) = nodes.get_mut(node_handle) else {
            return;
        };

        node.transform.previous_world_matrix = node.transform.world_matrix;

        // 1. Smartly update local matrix
        let local_changed = node.transform.update_local_matrix();

        // 2. Decide whether to update world matrix
        let world_needs_update = local_changed || parent_changed;

        if world_needs_update {
            let new_world = parent_world_matrix * *node.transform.local_matrix();
            node.transform.set_world_matrix(new_world);

            // Synchronously update camera
            if let Some(camera) = cameras.get_mut(node_handle) {
                camera.update_view_projection(&new_world);
            }
        }

        // Collect necessary info to avoid later borrow conflicts
        let world = node.transform.world_matrix;
        let children: Vec<NodeHandle> = node.children.clone();

        (world, children, world_needs_update)
    };

    // Phase 2: Recursively process child nodes
    for child_handle in children_handles {
        update_transform_recursive(
            nodes,
            cameras,
            child_handle,
            current_world_matrix,
            world_needs_update,
        );
    }
}

/// Updates only a single node's local matrix (non-recursive)
/// Used for scenarios that require immediate refresh of a single node
#[inline]
pub fn update_single_node_local(node: &mut Node) -> bool {
    node.transform.update_local_matrix()
}

/// Updates the subtree downward starting from the specified node
/// Used for partial updates of a portion of the scene graph
pub fn update_subtree(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SparseSecondaryMap<NodeHandle, Camera>,
    root_handle: NodeHandle,
) -> TransformUpdateStats {
    update_subtrees_iterative(nodes, cameras, &[root_handle])
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_hierarchy_update() {
        let mut nodes: SlotMap<NodeHandle, Node> = SlotMap::with_key();
        let mut cameras: SparseSecondaryMap<NodeHandle, Camera> = SparseSecondaryMap::new();

        // Create a simple parent-child hierarchy
        let mut parent = Node::new();
        parent.transform.position = Vec3::new(1.0, 0.0, 0.0);
        let parent_handle = nodes.insert(parent);

        let mut child = Node::new();
        child.transform.position = Vec3::new(0.0, 1.0, 0.0);
        child.parent = Some(parent_handle);
        let child_handle = nodes.insert(child);

        // Establish parent-child relationship
        nodes
            .get_mut(parent_handle)
            .unwrap()
            .children
            .push(child_handle);

        let roots = vec![parent_handle];

        // Execute update
        update_hierarchy(&mut nodes, &mut cameras, &roots);

        // Verify child node's world position
        let child_world_pos = nodes
            .get(child_handle)
            .unwrap()
            .transform
            .world_matrix
            .translation;
        assert!((child_world_pos.x - 1.0).abs() < 1e-5);
        assert!((child_world_pos.y - 1.0).abs() < 1e-5);
    }
}
