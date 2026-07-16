//! Transform and TransformSystem tests
//!
//! Tests for:
//! - Transform TRS operations and dirty checking
//! - Euler angle round-trip conversions
//! - look_at orientation
//! - apply_local_matrix decomposition
//! - Hierarchical matrix propagation (iterative, batched, subtree)
//! - Level-order batching (BFS)

use glam::{Affine3A, EulerRot, Mat4, Quat, Vec3};
use myth::scene::NodeHandle;
use myth::scene::Scene;
use myth::scene::Transform;
use myth::scene::camera::Camera;
use myth::scene::node::Node;
use myth::scene::transform_system::*;
use slotmap::{SlotMap, SparseSecondaryMap};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

// ============================================================================
// Helper
// ============================================================================

const EPSILON: f32 = 1e-5;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

fn vec3_approx(a: Vec3, b: Vec3) -> bool {
    approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
}

#[test]
fn scene_transform_updates_are_incremental_and_coalesce_dirty_roots() {
    let mut scene = Scene::new();
    let parent = scene.add_node(Node::new());
    let child = scene.add_to_parent(Node::new(), parent);

    let initial = scene.update_matrix_world();
    assert!(initial.full_traversal);
    assert_eq!(initial.visited_nodes, 2);

    // Identity transforms do not need a compatibility settle pass.
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);

    scene.node_mut(child).unwrap().transform.position = Vec3::Y;
    scene.node_mut(parent).unwrap().transform.position = Vec3::X;
    scene.node_mut(parent).unwrap().transform.scale = Vec3::splat(2.0);

    // Parent supersedes the already-dirty child, so each node is visited once.
    let changed = scene.update_matrix_world();
    assert!(!changed.full_traversal);
    assert_eq!(changed.visited_nodes, 2);
    assert_eq!(changed.updated_world_matrices, 2);
    let child_world = scene
        .node(child)
        .unwrap()
        .transform
        .world_matrix
        .translation;
    assert!(vec3_approx(child_world.into(), Vec3::new(1.0, 2.0, 0.0)));

    scene.node_mut(parent).unwrap().transform.position = Vec3::Y;
    let continuous = scene.update_matrix_world();
    assert_eq!(
        continuous.visited_nodes, 2,
        "continuous animation must update a subtree once, not settle then update it"
    );
}

#[test]
fn raw_node_storage_mutation_triggers_one_full_transform_fallback() {
    let mut scene = Scene::new();
    let root = scene.add_node(Node::new());
    scene.update_matrix_world();
    scene.update_matrix_world();
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);

    scene.nodes.get_mut(root).unwrap().transform.position = Vec3::X;
    let fallback = scene.update_matrix_world();
    assert!(fallback.full_traversal);
    assert_eq!(fallback.visited_nodes, 1);
}

#[test]
fn scene_attach_rejects_invalid_hierarchy_without_partial_mutation() {
    let mut scene = Scene::new();
    let root = scene.add_node(Node::new());
    let child = scene.add_to_parent(Node::new(), root);
    scene.update_matrix_world();

    scene.attach(root, child);

    assert_eq!(scene.node(root).unwrap().parent(), None);
    assert_eq!(scene.node(child).unwrap().parent(), Some(root));
    let stats = scene.update_matrix_world();
    assert_eq!(stats.visited_nodes, 0);
    assert_eq!(scene.root_nodes(), &[root]);

    let missing_parent = scene.create_node();
    scene.remove_node(missing_parent);
    scene.attach(child, missing_parent);

    assert_eq!(scene.node(root).unwrap().parent(), None);
    assert_eq!(scene.node(child).unwrap().parent(), Some(root));
    assert_eq!(scene.root_nodes(), &[root]);
}

#[test]
fn render_flush_updates_late_dirty_roots_without_settling_earlier_motion() {
    let mut scene = Scene::new();
    let earlier = scene.add_node(Node::new());
    let late = scene.add_node(Node::new());
    scene.update_matrix_world();

    scene.edit_node(earlier).set_position_vec(Vec3::X);
    scene.update_matrix_world();
    let earlier_before_flush = scene.node(earlier).unwrap().transform;
    assert_ne!(
        earlier_before_flush.previous_world_matrix,
        earlier_before_flush.world_matrix
    );

    scene.edit_node(late).set_position_vec(Vec3::Y);
    let flush = scene.flush_dirty_world_matrices();
    assert_eq!(flush.visited_nodes, 1);
    let earlier_after_flush = scene.node(earlier).unwrap().transform;
    assert_eq!(
        earlier_after_flush.previous_world_matrix,
        earlier_before_flush.previous_world_matrix
    );
    assert_eq!(
        earlier_after_flush.world_matrix,
        earlier_before_flush.world_matrix
    );
    assert_eq!(
        scene.node(late).unwrap().transform.world_matrix.translation,
        Vec3::Y.into()
    );
}

#[test]
fn overlapping_settle_roots_stay_disjoint_across_a_render_flush() {
    let mut scene = Scene::new();
    let parent = scene.add_node(Node::new());
    let child = scene.add_to_parent(Node::new(), parent);
    scene.update_matrix_world();

    // A simulation-side parent update owes previous-matrix settlement for the
    // whole subtree. A later child-only extraction flush must not append a
    // redundant child settle root below it.
    scene.edit_node(parent).set_position_vec(Vec3::X);
    assert_eq!(scene.update_matrix_world().visited_nodes, 2);
    scene.edit_node(child).set_position_vec(Vec3::Y);
    assert_eq!(scene.flush_dirty_world_matrices().visited_nodes, 1);
    assert_eq!(scene.update_matrix_world().visited_nodes, 2);
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);

    // The inverse overlap is equivalent: a late parent update subsumes an
    // existing child settle root instead of retaining both roots.
    scene.edit_node(child).set_position_vec(Vec3::Z);
    assert_eq!(scene.update_matrix_world().visited_nodes, 1);
    scene.edit_node(parent).set_position_vec(Vec3::Y);
    assert_eq!(scene.flush_dirty_world_matrices().visited_nodes, 2);
    assert_eq!(scene.update_matrix_world().visited_nodes, 2);
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);
}

#[test]
fn render_flush_merges_many_independent_settle_roots_without_losing_work() {
    const ROOT_COUNT: usize = 256;
    const COHORT_SIZE: usize = ROOT_COUNT / 2;

    let mut scene = Scene::new();
    let roots = (0..ROOT_COUNT)
        .map(|_| scene.add_node(Node::new()))
        .collect::<Vec<_>>();
    scene.update_matrix_world();

    for &root in &roots[..COHORT_SIZE] {
        scene.edit_node(root).set_position_vec(Vec3::Y);
    }
    assert_eq!(
        scene.flush_dirty_world_matrices().visited_nodes,
        COHORT_SIZE
    );

    // A disjoint second extraction-boundary update must union its roots with
    // the pending first cohort, then settle every root exactly once at the
    // next simulation boundary.
    for &root in &roots[COHORT_SIZE..] {
        scene.edit_node(root).set_position_vec(Vec3::Z);
    }
    assert_eq!(
        scene.flush_dirty_world_matrices().visited_nodes,
        COHORT_SIZE
    );
    assert_eq!(scene.update_matrix_world().visited_nodes, ROOT_COUNT);
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);
}

#[test]
fn pending_full_settlement_preserves_late_unattached_render_flush_roots() {
    let mut scene = Scene::new();
    let rooted = scene.add_node(Node::new());
    scene.edit_node(rooted).set_position_vec(Vec3::X);

    let initial = scene.update_matrix_world();
    assert!(initial.full_traversal);
    assert_eq!(initial.visited_nodes, 1);

    // `create_node` deliberately creates an unattached node which is absent
    // from `root_nodes` until a caller attaches it. A late render flush must
    // retain its explicit settle root because the pending full settlement
    // cannot reach it.
    let unattached = scene.create_node();
    scene.edit_node(unattached).set_position_vec(Vec3::Y);
    assert_eq!(scene.flush_dirty_world_matrices().visited_nodes, 1);

    assert_eq!(scene.update_matrix_world().visited_nodes, 1);
    assert_eq!(scene.update_matrix_world().visited_nodes, 1);
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);
}

#[test]
fn many_independent_dirty_roots_coalesce_with_linear_parent_work() {
    const ROOT_COUNT: usize = 4_096;

    let mut scene = Scene::new();
    let roots = (0..ROOT_COUNT)
        .map(|_| scene.add_node(Node::new()))
        .collect::<Vec<_>>();
    scene.update_matrix_world();
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);

    for (index, &root) in roots.iter().enumerate() {
        scene.node_mut(root).unwrap().transform.position.x = index as f32;
    }
    let stats = scene.update_matrix_world();

    assert!(!stats.full_traversal);
    assert_eq!(stats.dirty_roots_examined, ROOT_COUNT);
    assert_eq!(stats.dirty_root_parent_links_examined, 0);
    assert_eq!(stats.visited_nodes, ROOT_COUNT);
}

// ============================================================================
// Transform Unit Tests
// ============================================================================

#[test]
fn transform_default_is_identity() {
    let t = Transform::new();
    assert_eq!(t.position, Vec3::ZERO);
    assert_eq!(t.rotation, Quat::IDENTITY);
    assert_eq!(t.scale, Vec3::ONE);
}

#[test]
fn transform_update_local_matrix_dirty_check() {
    let mut t = Transform::new();

    // First call should always return true (force_update starts true)
    assert!(t.update_local_matrix());

    // Second call without changes should return false
    assert!(!t.update_local_matrix());

    // Changing position should trigger a new update
    t.position = Vec3::new(1.0, 2.0, 3.0);
    assert!(t.update_local_matrix());

    // No change again
    assert!(!t.update_local_matrix());

    // Changing rotation
    t.rotation = Quat::from_rotation_y(FRAC_PI_2);
    assert!(t.update_local_matrix());
    assert!(!t.update_local_matrix());

    // Changing scale
    t.scale = Vec3::splat(2.0);
    assert!(t.update_local_matrix());
    assert!(!t.update_local_matrix());
}

#[test]
fn transform_local_matrix_reflects_trs() {
    let mut t = Transform::new();
    t.position = Vec3::new(10.0, 20.0, 30.0);
    t.scale = Vec3::splat(2.0);
    t.update_local_matrix();

    let mat = Mat4::from(*t.local_matrix());
    // The translation column should reflect position
    let translation = mat.w_axis.truncate();
    assert!(vec3_approx(translation, Vec3::new(10.0, 20.0, 30.0)));
}

#[test]
fn transform_euler_roundtrip() {
    let mut t = Transform::new();
    let (x, y, z) = (0.3, 0.7, 1.2);
    t.set_rotation_euler(x, y, z);

    let euler = t.rotation_euler();
    assert!(approx_eq(euler.x, x));
    assert!(approx_eq(euler.y, y));
    assert!(approx_eq(euler.z, z));
}

#[test]
fn transform_euler_with_order() {
    let mut t = Transform::new();
    t.set_rotation_euler_with_order(0.5, 0.3, 0.1, EulerRot::YXZ);

    // Verify rotation is not identity (was actually set)
    let q = t.rotation;
    assert!((q.length() - 1.0).abs() < 1e-4);
    assert_ne!(q, Quat::IDENTITY);
}

#[test]
fn transform_look_at_basic() {
    let mut t = Transform::new();
    t.position = Vec3::ZERO;
    t.look_at(Vec3::new(0.0, 0.0, -10.0), Vec3::Y);

    // After looking at -Z from origin, the rotation should produce forward = -Z
    t.update_local_matrix();
    let mat = Mat4::from(*t.local_matrix());
    // Z-axis column (negated for right-hand) should point toward target
    let forward = -mat.z_axis.truncate().normalize();
    assert!(vec3_approx(forward, Vec3::new(0.0, 0.0, -1.0)));
}

#[test]
fn transform_look_at_collinear_up_noop() {
    let mut t = Transform::new();
    let original_rotation = t.rotation;
    // Target is directly above, up is also Vec3::Y → collinear, should be no-op
    t.look_at(Vec3::new(0.0, 10.0, 0.0), Vec3::Y);
    assert_eq!(t.rotation, original_rotation);
}

#[test]
fn transform_apply_local_matrix_decomposition() {
    let original_pos = Vec3::new(5.0, -3.0, 7.0);
    let original_rot = Quat::from_rotation_y(FRAC_PI_4);
    let original_scale = Vec3::new(2.0, 3.0, 1.5);

    let mat = Affine3A::from_scale_rotation_translation(original_scale, original_rot, original_pos);

    let mut t = Transform::new();
    t.apply_local_matrix(mat);

    assert!(vec3_approx(t.position, original_pos));
    assert!(vec3_approx(t.scale, original_scale));
    // Quaternion may differ in sign, but represent the same rotation
    let angle = t.rotation.angle_between(original_rot);
    assert!(angle < 1e-4);
}

#[test]
fn transform_mark_dirty_forces_update() {
    let mut t = Transform::new();
    t.update_local_matrix();

    // No changes, should not update
    assert!(!t.update_local_matrix());

    // Mark dirty explicitly
    t.mark_dirty();
    assert!(t.update_local_matrix());
}

#[test]
fn transform_set_position_marks_dirty() {
    let mut t = Transform::new();
    t.update_local_matrix();
    assert!(!t.update_local_matrix());

    t.set_position(Vec3::new(1.0, 0.0, 0.0));
    assert!(t.update_local_matrix());
}

// ============================================================================
// Hierarchy Setup Helpers
// ============================================================================

struct HierarchySetup {
    nodes: SlotMap<NodeHandle, Node>,
    cameras: SparseSecondaryMap<NodeHandle, Camera>,
    roots: Vec<NodeHandle>,
}

fn create_chain(length: usize) -> (HierarchySetup, Vec<NodeHandle>) {
    let mut nodes: SlotMap<NodeHandle, Node> = SlotMap::with_key();
    let cameras = SparseSecondaryMap::new();

    let mut handles = Vec::new();
    for i in 0..length {
        let mut node = Node::new();
        node.transform.position = Vec3::new(1.0, 0.0, 0.0); // Each translates +1 in X
        if i > 0 {
            node.set_parent(Some(handles[i - 1]));
        }
        let handle = nodes.insert(node);
        if i > 0 {
            nodes.get_mut(handles[i - 1]).unwrap().push_child(handle);
        }
        handles.push(handle);
    }

    let roots = vec![handles[0]];
    (
        HierarchySetup {
            nodes,
            cameras,
            roots,
        },
        handles,
    )
}

// ============================================================================
// TransformSystem Hierarchy Tests
// ============================================================================

#[test]
fn hierarchy_chain_world_positions() {
    let (mut setup, handles) = create_chain(5);

    update_hierarchy_iterative(&mut setup.nodes, &mut setup.cameras, &setup.roots);

    // Node[i] should have world X = i+1 (cumulative translations)
    for (i, &handle) in handles.iter().enumerate() {
        let world_pos = setup
            .nodes
            .get(handle)
            .unwrap()
            .transform
            .world_matrix()
            .translation;
        let expected_x = (i + 1) as f32;
        assert!(
            approx_eq(world_pos.x, expected_x),
            "Node {i}: expected x={expected_x}, got x={}",
            world_pos.x
        );
    }
}

#[test]
fn hierarchy_recursive_matches_iterative() {
    let (mut setup_iter, _) = create_chain(4);
    let (mut setup_rec, _) = create_chain(4);

    update_hierarchy_iterative(
        &mut setup_iter.nodes,
        &mut setup_iter.cameras,
        &setup_iter.roots,
    );
    update_hierarchy(
        &mut setup_rec.nodes,
        &mut setup_rec.cameras,
        &setup_rec.roots,
    );

    // Both should produce the same world matrices
    for (key, node_iter) in &setup_iter.nodes {
        let node_rec = setup_rec.nodes.get(key).unwrap();
        let pos_iter = node_iter.transform.world_matrix().translation;
        let pos_rec = node_rec.transform.world_matrix().translation;
        assert!(
            vec3_approx(pos_iter.into(), pos_rec.into()),
            "Mismatch for node {:?}",
            key
        );
    }
}

#[test]
fn hierarchy_batched_matches_iterative() {
    let (mut setup_iter, _) = create_chain(4);
    let (mut setup_batch, handles) = create_chain(4);

    update_hierarchy_iterative(
        &mut setup_iter.nodes,
        &mut setup_iter.cameras,
        &setup_iter.roots,
    );

    let mut batches = LevelOrderBatches::new();
    build_level_order_batches(&setup_batch.nodes, &setup_batch.roots, &mut batches);
    update_hierarchy_batched(&mut setup_batch.nodes, &mut setup_batch.cameras, &batches);

    for &handle in &handles {
        let pos_iter = setup_iter
            .nodes
            .get(handle)
            .unwrap()
            .transform
            .world_matrix()
            .translation;
        let pos_batch = setup_batch
            .nodes
            .get(handle)
            .unwrap()
            .transform
            .world_matrix()
            .translation;
        assert!(
            vec3_approx(pos_iter.into(), pos_batch.into()),
            "Mismatch for batched update"
        );
    }
}

#[test]
fn hierarchy_with_rotation_and_scale() {
    let mut nodes: SlotMap<NodeHandle, Node> = SlotMap::with_key();
    let mut cameras = SparseSecondaryMap::new();

    // Parent: translate (5,0,0), rotate 90° around Y, scale 2x
    let mut parent = Node::new();
    parent.transform.position = Vec3::new(5.0, 0.0, 0.0);
    parent.transform.rotation = Quat::from_rotation_y(FRAC_PI_2);
    parent.transform.scale = Vec3::splat(2.0);
    let parent_h = nodes.insert(parent);

    // Child: translate (1,0,0) in local space
    let mut child = Node::new();
    child.transform.position = Vec3::new(1.0, 0.0, 0.0);
    child.set_parent(Some(parent_h));
    let child_h = nodes.insert(child);
    nodes.get_mut(parent_h).unwrap().push_child(child_h);

    let roots = vec![parent_h];
    update_hierarchy_iterative(&mut nodes, &mut cameras, &roots);

    // Parent world: position = (5,0,0), scale = 2, rotation = 90° Y
    // Child local (1,0,0) in parent space:
    //   After parent's rotation (90° Y): (1,0,0) → (0,0,-1)
    //   After parent's scale (2x): (0,0,-2)
    //   After parent's translation: (5,0,-2)
    let child_world = nodes
        .get(child_h)
        .unwrap()
        .transform
        .world_matrix()
        .translation;
    assert!(
        approx_eq(child_world.x, 5.0),
        "child world x: expected 5.0, got {}",
        child_world.x
    );
    assert!(
        approx_eq(child_world.z, -2.0),
        "child world z: expected -2.0, got {}",
        child_world.z
    );
}

#[test]
fn hierarchy_subtree_update() {
    let (mut setup, handles) = create_chain(5);

    // First do a full update
    update_hierarchy_iterative(&mut setup.nodes, &mut setup.cameras, &setup.roots);

    // Move node[2] to a different position
    setup.nodes.get_mut(handles[2]).unwrap().transform.position = Vec3::new(10.0, 0.0, 0.0);

    // Only update the subtree starting from node[2]
    update_subtree(&mut setup.nodes, &mut setup.cameras, handles[2]);

    // Node[2] world X = parent(2) + 10 = 2 + 10 = 12
    let node2_world = setup
        .nodes
        .get(handles[2])
        .unwrap()
        .transform
        .world_matrix()
        .translation;
    assert!(
        approx_eq(node2_world.x, 12.0),
        "expected 12.0, got {}",
        node2_world.x
    );

    // Node[3] world X = node2(12) + 1 = 13
    let node3_world = setup
        .nodes
        .get(handles[3])
        .unwrap()
        .transform
        .world_matrix()
        .translation;
    assert!(
        approx_eq(node3_world.x, 13.0),
        "expected 13.0, got {}",
        node3_world.x
    );
}

// ============================================================================
// Level-Order Batching Tests
// ============================================================================

#[test]
fn bfs_batches_depth_and_count() {
    let (setup, _handles) = create_chain(5);
    let mut batches = LevelOrderBatches::new();

    build_level_order_batches(&setup.nodes, &setup.roots, &mut batches);

    assert_eq!(batches.depth(), 5, "Chain of 5 should have depth 5");
    assert_eq!(batches.total_nodes(), 5);

    // Each level should have exactly 1 node
    for (level, batch) in batches.batches.iter().enumerate() {
        assert_eq!(batch.len(), 1, "Level {level} should have 1 node");
    }
}

#[test]
fn bfs_batches_wide_tree() {
    let mut nodes: SlotMap<NodeHandle, Node> = SlotMap::with_key();

    // Root with 10 children (depth 2, level 0 = 1 node, level 1 = 10 nodes)
    let root = nodes.insert(Node::new());
    for _ in 0..10 {
        let mut child = Node::new();
        child.set_parent(Some(root));
        let child_h = nodes.insert(child);
        nodes.get_mut(root).unwrap().push_child(child_h);
    }

    let roots = vec![root];
    let mut batches = LevelOrderBatches::new();
    build_level_order_batches(&nodes, &roots, &mut batches);

    assert_eq!(batches.depth(), 2);
    assert_eq!(batches.batches[0].len(), 1);
    assert_eq!(batches.batches[1].len(), 10);
    assert_eq!(batches.total_nodes(), 11);
}

#[test]
fn bfs_batches_empty_scene() {
    let nodes: SlotMap<NodeHandle, Node> = SlotMap::with_key();
    let roots: Vec<NodeHandle> = vec![];
    let mut batches = LevelOrderBatches::new();

    build_level_order_batches(&nodes, &roots, &mut batches);

    assert_eq!(batches.depth(), 0);
    assert_eq!(batches.total_nodes(), 0);
}

#[test]
fn bfs_batches_multiple_roots() {
    let mut nodes: SlotMap<NodeHandle, Node> = SlotMap::with_key();
    let root1 = nodes.insert(Node::new());
    let root2 = nodes.insert(Node::new());

    // root1 has 2 children
    for _ in 0..2 {
        let mut child = Node::new();
        child.set_parent(Some(root1));
        let ch = nodes.insert(child);
        nodes.get_mut(root1).unwrap().push_child(ch);
    }

    let roots = vec![root1, root2];
    let mut batches = LevelOrderBatches::new();
    build_level_order_batches(&nodes, &roots, &mut batches);

    assert_eq!(batches.depth(), 2);
    assert_eq!(batches.batches[0].len(), 2); // 2 roots
    assert_eq!(batches.batches[1].len(), 2); // 2 children of root1
}

#[test]
fn hierarchy_camera_sync_on_update() {
    let mut nodes: SlotMap<NodeHandle, Node> = SlotMap::with_key();
    let mut cameras: SparseSecondaryMap<NodeHandle, Camera> = SparseSecondaryMap::new();

    let mut node = Node::new();
    node.transform.position = Vec3::new(0.0, 5.0, 10.0);
    let handle = nodes.insert(node);

    let camera = Camera::new_perspective(60.0, 1.0, 0.1);
    cameras.insert(handle, camera);

    let roots = vec![handle];
    update_hierarchy_iterative(&mut nodes, &mut cameras, &roots);

    // Camera's view projection should reflect the node's world position
    let cam = cameras.get_mut(handle).unwrap();
    let cam_render = cam.extract_render_camera();
    assert!(
        approx_eq(cam_render.position.y, 5.0),
        "Camera should be at Y=5, got {}",
        cam_render.position.y
    );
}

// ============================================================================
// Transform: Identity Hierarchy
// ============================================================================

#[test]
fn identity_hierarchy_produces_identity_world() {
    let mut nodes: SlotMap<NodeHandle, Node> = SlotMap::with_key();
    let mut cameras = SparseSecondaryMap::new();

    let root = nodes.insert(Node::new());
    let mut child = Node::new();
    child.set_parent(Some(root));
    let child_h = nodes.insert(child);
    nodes.get_mut(root).unwrap().push_child(child_h);

    let roots = vec![root];
    update_hierarchy_iterative(&mut nodes, &mut cameras, &roots);

    let world = *nodes.get(child_h).unwrap().transform.world_matrix();
    // With all identity transforms, world should be identity
    assert!(vec3_approx(world.translation.into(), Vec3::ZERO));
}

#[test]
fn deeply_nested_hierarchy_no_stack_overflow() {
    let depth = 500; // Recursive version would stack overflow; iterative should handle this
    let (mut setup, handles) = create_chain(depth);

    update_hierarchy_iterative(&mut setup.nodes, &mut setup.cameras, &setup.roots);

    // Last node should have world X = depth
    let last = setup.nodes.get(*handles.last().unwrap()).unwrap();
    let expected = depth as f32;
    assert!(
        approx_eq(last.transform.world_matrix().translation.x, expected),
        "expected {expected}, got {}",
        last.transform.world_matrix().translation.x
    );
}
