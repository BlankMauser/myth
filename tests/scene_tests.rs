//! Scene Integration Tests
//!
//! Tests for:
//! - Scene: create/remove nodes, attach/detach hierarchy
//! - Component management: set/get mesh, camera, light, morph weights
//! - Node query: names, root_nodes, subtree collection
//! - NodeEditor exact fluent-edit API

use std::sync::Arc;

use glam::Vec3;
use myth::Mesh;
use myth::animation::action::AnimationAction;
use myth::animation::binding::{ClipBinding, Rig, TargetPath, TrackBinding};
use myth::animation::clip::{AnimationClip, Track, TrackData, TrackMeta};
use myth::animation::tracks::{InterpolationMode, KeyframeTrack};
use myth::animation::{AnimationMixer, AnimationTarget};
use myth::resources::{GeometryHandle, MaterialHandle};
use myth::scene::camera::Camera;
use myth::scene::light::{Light, LightKind};
use myth::scene::mutation::{SceneFullRebuildReasons, SceneMutationFlags};
use myth::scene::node::Node;
use myth::scene::{Scene, Skeleton};
use slotmap::KeyData;

fn new_scene() -> Scene {
    Scene::new()
}

fn acknowledge_initial_render_world(scene: &mut Scene) {
    let batch = scene.render_mutations();
    assert!(batch.requires_full_rebuild);
    assert!(
        batch
            .full_rebuild_reasons
            .contains(SceneFullRebuildReasons::INITIAL)
    );
    let cursor = batch.cursor;
    assert!(scene.acknowledge_render_mutations(cursor));
}

#[test]
fn idle_render_acknowledgements_release_scene_frontend_high_water_marks() {
    const ROOT_COUNT: usize = 256;

    let mut scene = new_scene();
    let roots = (0..ROOT_COUNT)
        .map(|_| scene.add_node(Node::new()))
        .collect::<Vec<_>>();
    scene.update_matrix_world();
    acknowledge_initial_render_world(&mut scene);

    for (index, &root) in roots.iter().enumerate() {
        scene
            .edit_node(root)
            .set_position_vec(Vec3::new(index as f32 + 1.0, 0.0, 0.0));
    }
    let update = scene.update_matrix_world();
    assert_eq!(update.dirty_roots_examined, ROOT_COUNT);
    let settled = scene.update_matrix_world();
    assert_eq!(settled.visited_nodes, ROOT_COUNT);

    let cursor = scene.render_mutations().cursor;
    let journal_before = scene.render_mutation_journal_stats();
    assert_eq!(journal_before.pending_entries, ROOT_COUNT);
    assert!(scene.acknowledge_render_mutations(cursor));

    // Cross the retained frontend's idle maintenance threshold using the same
    // clean cursor, as a static renderer does on consecutive frames.
    for _ in 0..512 {
        let cursor = scene.render_mutations().cursor;
        assert!(scene.acknowledge_render_mutations(cursor));
    }

    let journal_after = scene.render_mutation_journal_stats();
    assert!(journal_after.estimated_retained_bytes < journal_before.estimated_retained_bytes);
    let transform_after = scene.update_matrix_world();
    assert!(transform_after.retained_bytes < settled.retained_bytes);
    assert!(!scene.trim_transform_update_capacity());
}

// ============================================================================
// Node Creation & Removal
// ============================================================================

#[test]
fn scene_create_node() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    assert!(scene.node(handle).is_some());
}

#[test]
fn scene_create_node_with_name() {
    let mut scene = new_scene();
    let handle = scene.create_node_with_name("TestNode");
    assert_eq!(scene.name(handle), Some("TestNode"));
}

#[test]
fn scene_set_name() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    scene.set_name(handle, "Renamed");
    assert_eq!(scene.name(handle), Some("Renamed"));
}

#[test]
fn scene_add_node_to_root() {
    let mut scene = new_scene();
    let handle = scene.add_node(Node::new());
    assert!(scene.root_nodes().contains(&handle));
}

#[test]
fn scene_remove_node_removes_from_root() {
    let mut scene = new_scene();
    let handle = scene.add_node(Node::new());
    assert!(scene.root_nodes().contains(&handle));

    scene.remove_node(handle);
    assert!(!scene.root_nodes().contains(&handle));
    assert!(scene.node(handle).is_none());
}

#[test]
fn scene_remove_node_removes_subtree() {
    let mut scene = new_scene();
    let parent = scene.add_node(Node::new());
    let child = scene.create_node();
    let grandchild = scene.create_node();

    scene.attach(child, parent);
    scene.attach(grandchild, child);

    scene.remove_node(parent);

    assert!(scene.node(parent).is_none());
    assert!(scene.node(child).is_none());
    assert!(scene.node(grandchild).is_none());
}

// ============================================================================
// Hierarchy: Attach / Detach
// ============================================================================

#[test]
fn scene_attach_sets_parent_child() {
    let mut scene = new_scene();
    let parent = scene.create_node();
    let child = scene.create_node();

    scene.attach(child, parent);

    assert_eq!(scene.node(child).unwrap().parent(), Some(parent));
    assert!(scene.node(parent).unwrap().children().contains(&child));
}

#[test]
fn scene_attach_removes_from_old_parent() {
    let mut scene = new_scene();
    let parent1 = scene.create_node();
    let parent2 = scene.create_node();
    let child = scene.create_node();

    scene.attach(child, parent1);
    assert!(scene.node(parent1).unwrap().children().contains(&child));

    // Re-attach to parent2
    scene.attach(child, parent2);
    assert!(
        !scene.node(parent1).unwrap().children().contains(&child),
        "Child should be removed from old parent"
    );
    assert!(
        scene.node(parent2).unwrap().children().contains(&child),
        "Child should be in new parent"
    );
}

#[test]
fn scene_attach_to_self_is_noop() {
    let mut scene = new_scene();
    let node = scene.create_node();

    // attach to self should not crash
    scene.attach(node, node);

    assert_eq!(scene.node(node).unwrap().parent(), None);
}

#[test]
fn scene_add_to_parent() {
    let mut scene = new_scene();
    let parent = scene.add_node(Node::new());
    let child = scene.add_to_parent(Node::new(), parent);

    assert_eq!(scene.node(child).unwrap().parent(), Some(parent));
    assert!(scene.node(parent).unwrap().children().contains(&child));
}

// ============================================================================
// Component Management: Camera, Light
// ============================================================================

#[test]
fn scene_set_and_read_camera() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    let camera = Camera::new_perspective(60.0, 16.0 / 9.0, 0.1);
    scene.set_camera(handle, camera);

    assert!(scene.camera(handle).is_some());
}

#[test]
fn scene_active_camera() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    scene.set_camera(handle, Camera::new_perspective(60.0, 1.0, 0.1));
    scene.active_camera = Some(handle);

    assert_eq!(scene.active_camera, Some(handle));
}

#[test]
fn scene_set_and_read_light() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    let light = Light::new_directional(Vec3::ONE, 1.0);
    scene.set_light(handle, light);

    let l = scene.light(handle).unwrap();
    assert!(matches!(l.kind, LightKind::Directional(_)));
}

#[test]
fn scene_set_morph_weights() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    scene.set_morph_weights(handle, vec![0.5, 0.3, 0.1]);

    let weights = scene.morph_weights(handle).unwrap();
    assert_eq!(weights.len(), 3);
    assert!((weights[0] - 0.5).abs() < 1e-5);
}

// ============================================================================
// Hierarchy + Transform Dirty Propagation
// ============================================================================

#[test]
fn scene_attach_marks_child_dirty() {
    let mut scene = new_scene();
    let parent = scene.create_node();
    let child = scene.create_node();

    // Consume dirty flag by calling update_local_matrix
    scene
        .node_mut(child)
        .unwrap()
        .transform
        .update_local_matrix();

    scene.attach(child, parent);

    // attach should mark child transform as dirty, so update_local_matrix returns true
    let child_node = scene.node_mut(child).unwrap();
    assert!(
        child_node.transform.update_local_matrix(),
        "Attach should mark child transform dirty"
    );
}

// ============================================================================
// Multiple Nodes & Iteration
// ============================================================================

#[test]
fn scene_iterate_active_lights() {
    let mut scene = new_scene();
    let h1 = scene.create_node();
    scene.set_light(h1, Light::new_directional(Vec3::ONE, 1.0));

    let h2 = scene.create_node();
    scene.set_light(h2, Light::new_point(Vec3::ONE, 1.0, 10.0));

    // Make h2 invisible
    scene.node_mut(h2).unwrap().visible = false;

    let active_lights: Vec<_> = scene.iter_active_lights().collect();
    assert_eq!(
        active_lights.len(),
        1,
        "Only visible lights should be iterated"
    );
}

#[test]
fn scene_unique_ids() {
    let s1 = new_scene();
    let s2 = new_scene();
    assert_ne!(s1.id(), s2.id(), "Each scene should have a unique ID");
}

// ============================================================================
// Persistent render-world mutation contract
// ============================================================================

#[test]
fn render_mutations_are_static_after_initial_authority_capture() {
    let mut scene = new_scene();
    scene.add_node(Node::new());
    acknowledge_initial_render_world(&mut scene);

    let batch = scene.render_mutations();
    assert!(!batch.requires_full_rebuild);
    assert!(batch.entries.is_empty());
}

#[test]
fn tracked_transform_and_mesh_changes_coalesce_by_node() {
    let mut scene = new_scene();
    let geometry = GeometryHandle::from(KeyData::from_ffi(0x0000_0001_0000_0001));
    let material = MaterialHandle::from(KeyData::from_ffi(0x0000_0001_0000_0002));
    let replacement = MaterialHandle::from(KeyData::from_ffi(0x0000_0001_0000_0003));
    let node = scene.add_mesh(Mesh::new(geometry, material));
    acknowledge_initial_render_world(&mut scene);

    scene.node_mut(node).unwrap().transform.position = Vec3::X;
    scene.mesh_mut(node).unwrap().material = replacement;

    let batch = scene.render_mutations();
    assert!(!batch.requires_full_rebuild);
    assert_eq!(batch.entries.len(), 1);
    assert_eq!(batch.entries[0].node, node);
    assert!(
        batch.entries[0]
            .flags
            .contains(SceneMutationFlags::TRANSFORM)
    );
    assert!(
        batch.entries[0]
            .flags
            .contains(SceneMutationFlags::RENDERABLE)
    );
    assert!(
        !batch.entries[0]
            .flags
            .contains(SceneMutationFlags::TOPOLOGY)
    );
}

#[test]
fn removed_subtree_emits_explicit_tombstones() {
    let mut scene = new_scene();
    let parent = scene.add_node(Node::new());
    let child = scene.add_to_parent(Node::new(), parent);
    acknowledge_initial_render_world(&mut scene);

    scene.remove_node(parent);

    let batch = scene.render_mutations();
    assert_eq!(batch.entries.len(), 2);
    for handle in [parent, child] {
        let mutation = batch
            .entries
            .iter()
            .find(|mutation| mutation.node == handle)
            .expect("removed node must have a tombstone");
        assert!(mutation.flags.contains(SceneMutationFlags::REMOVED));
        assert!(mutation.flags.contains(SceneMutationFlags::TOPOLOGY));
    }
}

#[test]
fn raw_storage_mutation_fails_safe_with_a_typed_full_rebuild() {
    let mut scene = new_scene();
    let node = scene.add_node(Node::new());
    acknowledge_initial_render_world(&mut scene);

    scene.nodes.get_mut(node).unwrap().visible = false;

    let batch = scene.render_mutations();
    assert!(batch.requires_full_rebuild);
    assert!(batch.untracked_storage_mutation);
    assert!(
        batch
            .full_rebuild_reasons
            .contains(SceneFullRebuildReasons::UNTRACKED_STORAGE)
    );

    // Skeleton identity and bounds are part of a retained RenderItem too.
    // Mutating the legacy public pool must therefore cross the same
    // conservative compatibility boundary instead of leaving a stale cache.
    let mut scene = new_scene();
    let skeleton = scene.add_skeleton(Skeleton::new("raw", Vec::new(), Vec::new(), 0));
    acknowledge_initial_render_world(&mut scene);
    assert!(scene.skeleton_pool.remove(skeleton).is_some());

    let batch = scene.render_mutations();
    assert!(batch.requires_full_rebuild);
    assert!(batch.untracked_storage_mutation);
    assert!(
        batch
            .full_rebuild_reasons
            .contains(SceneFullRebuildReasons::UNTRACKED_STORAGE)
    );
}

#[test]
fn swapping_public_storage_wrappers_between_scenes_fails_safe() {
    let mut first = new_scene();
    let mut second = new_scene();
    first.add_node(Node::new());
    second.add_node(Node::new());
    acknowledge_initial_render_world(&mut first);
    acknowledge_initial_render_world(&mut second);

    // `mem::swap` does not call `DerefMut`, so each wrapper carries a stable
    // lineage in addition to its mutation counter. Moving a whole public
    // storage between scenes must still invalidate both retained consumers.
    std::mem::swap(&mut first.meshes, &mut second.meshes);

    for scene in [&first, &second] {
        let batch = scene.render_mutations();
        assert!(batch.requires_full_rebuild);
        assert!(batch.untracked_storage_mutation);
        assert!(
            batch
                .full_rebuild_reasons
                .contains(SceneFullRebuildReasons::UNTRACKED_STORAGE)
        );
    }
}

#[test]
fn foreign_tracked_storage_mutation_cannot_aba_after_a_round_trip_swap() {
    let mut first = new_scene();
    let mut second = new_scene();
    let first_node = first.add_node(Node::new());
    let second_node = second.add_node(Node::new());
    acknowledge_initial_render_world(&mut first);
    acknowledge_initial_render_world(&mut second);

    std::mem::swap(&mut first.nodes, &mut second.nodes);
    // The handles intentionally occupy the same SlotMap position. The tracked
    // edit must nevertheless rebind and advance the foreign wrapper's stamp.
    second.edit_node(second_node).set_visible(false);
    std::mem::swap(&mut first.nodes, &mut second.nodes);

    let cursor = {
        let batch = first.render_mutations();
        assert!(batch.requires_full_rebuild);
        assert!(batch.untracked_storage_mutation);
        batch.cursor
    };
    assert!(first.acknowledge_render_mutations(cursor));
    assert!(!first.render_mutations().requires_full_rebuild);

    // The test's original node identity remains valid; this also prevents an
    // optimizer from treating the first handle as irrelevant setup.
    assert_eq!(first.node(first_node).unwrap().visible, false);
}

#[test]
fn controlled_insertion_consumes_detached_node_hierarchy_markers() {
    let mut scene = new_scene();
    let parent = scene.add_node(Node::new());
    let mut detached_child = Node::new();
    detached_child.set_parent(Some(parent));
    let child = scene.add_to_parent(detached_child, parent);
    scene.update_matrix_world();
    acknowledge_initial_render_world(&mut scene);
    scene.update_matrix_world();

    scene.node_mut(child).unwrap().transform.position = Vec3::Y;
    scene.update_matrix_world();

    let batch = scene.render_mutations();
    assert!(!batch.requires_full_rebuild);
    assert_eq!(batch.entries.len(), 1);
    assert!(
        batch.entries[0]
            .flags
            .contains(SceneMutationFlags::TRANSFORM)
    );
}

#[test]
fn low_level_hierarchy_mutation_is_a_typed_fail_safe_boundary() {
    let mut scene = new_scene();
    let node = scene.add_node(Node::new());
    scene.update_matrix_world();
    acknowledge_initial_render_world(&mut scene);
    scene.update_matrix_world();

    let parent = scene.node(node).unwrap().parent();
    scene.node_mut(node).unwrap().set_parent(parent);
    let stats = scene.update_matrix_world();

    assert!(stats.full_traversal);
    let batch = scene.render_mutations();
    assert!(batch.requires_full_rebuild);
    assert!(
        batch
            .full_rebuild_reasons
            .contains(SceneFullRebuildReasons::UNTRACKED_HIERARCHY)
    );
}

#[test]
fn precise_scene_node_setters_emit_only_real_domain_changes() {
    let mut scene = new_scene();
    let geometry = GeometryHandle::from(KeyData::from_ffi(0x0000_0001_0000_0011));
    let material = MaterialHandle::from(KeyData::from_ffi(0x0000_0001_0000_0012));
    let root = scene.add_mesh(Mesh::new(geometry, material));
    let child = scene.add_to_parent(Node::new(), root);
    scene.update_matrix_world();
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);
    acknowledge_initial_render_world(&mut scene);

    scene
        .edit_node(root)
        .set_position_vec(Vec3::ZERO)
        .set_scale(1.0)
        .set_scale_xyz(1.0, 1.0, 1.0)
        .set_rotation(glam::Quat::IDENTITY)
        .set_rotation_euler(0.0, 0.0, 0.0)
        .rotate_x(0.0)
        .rotate_y(0.0)
        .look_at(Vec3::NEG_Z)
        .set_visible(true)
        .set_shadows(true, true);

    assert!(scene.render_mutations().entries.is_empty());
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);

    scene.edit_node(root).set_visible(false);
    let visibility_cursor = {
        let batch = scene.render_mutations();
        assert_eq!(batch.entries.len(), 1);
        assert!(
            batch.entries[0]
                .flags
                .contains(SceneMutationFlags::RENDERABLE)
        );
        assert!(
            !batch.entries[0]
                .flags
                .contains(SceneMutationFlags::TRANSFORM)
        );
        batch.cursor
    };
    assert_eq!(
        scene.update_matrix_world().visited_nodes,
        0,
        "visibility must not dirty the transform subtree"
    );
    assert!(scene.acknowledge_render_mutations(visibility_cursor));
    scene.edit_node(root).set_visible(false);
    assert!(scene.render_mutations().entries.is_empty());

    scene.edit_node(root).set_shadows(false, false);
    let shadow_cursor = {
        let batch = scene.render_mutations();
        assert_eq!(batch.entries.len(), 1);
        assert!(
            batch.entries[0]
                .flags
                .contains(SceneMutationFlags::RENDERABLE)
        );
        assert!(
            !batch.entries[0]
                .flags
                .contains(SceneMutationFlags::TRANSFORM)
        );
        batch.cursor
    };
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);
    assert!(scene.acknowledge_render_mutations(shadow_cursor));
    scene.edit_node(root).set_shadows(false, false);
    assert!(scene.render_mutations().entries.is_empty());

    scene
        .edit_node(root)
        .set_position_vec(Vec3::X)
        .set_position(1.0, 0.0, 0.0);
    let batch = scene.render_mutations();
    assert_eq!(batch.entries.len(), 1);
    assert_eq!(batch.entries[0].node, root);
    assert!(
        batch.entries[0]
            .flags
            .contains(SceneMutationFlags::TRANSFORM)
    );
    assert!(
        !batch.entries[0]
            .flags
            .contains(SceneMutationFlags::RENDERABLE)
    );
    let changed = scene.update_matrix_world();
    assert_eq!(changed.visited_nodes, 2);
    assert_eq!(
        scene
            .node(child)
            .unwrap()
            .transform
            .world_matrix
            .translation,
        Vec3::X.into()
    );
}

#[test]
fn exact_transform_camera_and_light_callbacks_keep_idle_frames_static() {
    let mut scene = new_scene();
    let camera_node = scene.add_node(Node::new());
    scene.set_camera(camera_node, Camera::new_perspective(60.0, 16.0 / 9.0, 0.1));
    let light_node = scene.add_light(Light::new_directional(Vec3::ONE, 1.0));
    scene.active_camera = Some(camera_node);
    scene.update_matrix_world();
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);
    acknowledge_initial_render_world(&mut scene);

    assert_eq!(
        scene.edit_node_transform(camera_node, |transform| transform.position),
        Some(Vec3::ZERO)
    );
    scene
        .edit_camera_bundle(camera_node, |transform, camera| {
            transform.position = Vec3::ZERO;
            camera.name = "Tracked Camera".into();
        })
        .unwrap();
    scene
        .edit_main_camera_bundle(|transform, _camera| {
            transform.rotation = glam::Quat::IDENTITY;
        })
        .unwrap();
    let (_, light) = scene
        .light_bundle(light_node)
        .expect("light bundle missing");
    assert_eq!(light.intensity, 1.0);
    scene
        .edit_light_bundle(light_node, |transform, light| {
            transform.position = Vec3::ZERO;
            light.intensity = 2.0;
        })
        .unwrap();

    assert!(scene.render_mutations().entries.is_empty());
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);

    let before = scene.node(camera_node).unwrap().transform;
    scene
        .edit_node_transform(camera_node, |transform| {
            transform.local_matrix = glam::Affine3A::from_translation(Vec3::X);
            transform.world_matrix = glam::Affine3A::from_translation(Vec3::Y);
            transform.previous_world_matrix = glam::Affine3A::from_translation(Vec3::Z);
            transform.mark_dirty();
        })
        .unwrap();
    let after = scene.node(camera_node).unwrap().transform;
    assert_eq!(after.position, before.position);
    assert_eq!(after.rotation, before.rotation);
    assert_eq!(after.scale, before.scale);
    assert_eq!(after.local_matrix, before.local_matrix);
    assert_eq!(after.world_matrix, before.world_matrix);
    assert_eq!(after.previous_world_matrix, before.previous_world_matrix);
    assert!(scene.render_mutations().entries.is_empty());
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);

    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Option<()> = scene.edit_node_transform(camera_node, |transform| {
            transform.position = Vec3::Z;
            panic!("probe exact transform edit rollback");
        });
    }));
    assert!(unwound.is_err());
    assert_eq!(
        scene.node(camera_node).unwrap().transform.position,
        before.position
    );
    assert!(scene.render_mutations().entries.is_empty());

    scene
        .edit_node_transform(camera_node, |transform| {
            transform.position = Vec3::Y;
        })
        .unwrap();
    let batch = scene.render_mutations();
    assert_eq!(batch.entries.len(), 1);
    assert!(
        batch.entries[0]
            .flags
            .contains(SceneMutationFlags::TRANSFORM)
    );
    assert!(
        !batch.entries[0]
            .flags
            .contains(SceneMutationFlags::RENDERABLE)
    );
}

#[test]
fn animation_target_trs_setters_do_not_imply_renderable_changes() {
    let mut scene = new_scene();
    let node = scene.add_node(Node::new());
    scene.update_matrix_world();
    acknowledge_initial_render_world(&mut scene);

    AnimationTarget::set_node_position(&mut scene, node, Vec3::ZERO);
    AnimationTarget::set_node_rotation(&mut scene, node, glam::Quat::IDENTITY);
    AnimationTarget::set_node_scale(&mut scene, node, Vec3::ONE);
    // AnimationMixer calls this even when the sampled pose is unchanged.
    AnimationTarget::mark_node_dirty(&mut scene, node);
    assert!(scene.render_mutations().entries.is_empty());
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);

    AnimationTarget::set_node_position(&mut scene, node, Vec3::Z);
    AnimationTarget::mark_node_dirty(&mut scene, node);
    let batch = scene.render_mutations();
    assert_eq!(batch.entries.len(), 1);
    assert!(
        batch.entries[0]
            .flags
            .contains(SceneMutationFlags::TRANSFORM)
    );
    assert!(
        !batch.entries[0]
            .flags
            .contains(SceneMutationFlags::RENDERABLE)
    );
}

#[test]
fn constant_and_paused_animation_only_publish_real_pose_transitions() {
    let mut scene = new_scene();
    let node = scene.add_node(Node::new());
    scene.update_matrix_world();
    acknowledge_initial_render_world(&mut scene);
    let rest = scene.node(node).unwrap().transform;
    AnimationTarget::store_rest_transform(&mut scene, node, rest);

    let clip = Arc::new(AnimationClip::new(
        "constant".into(),
        vec![Track {
            meta: TrackMeta {
                path: vec!["root".into()],
                target: TargetPath::Translation,
            },
            data: TrackData::Vector3(KeyframeTrack::new(
                vec![0.0, 1.0],
                vec![Vec3::X, Vec3::X],
                InterpolationMode::Linear,
            )),
        }],
    ));
    let mut action = AnimationAction::new(clip);
    action.clip_binding = ClipBinding {
        bindings: vec![TrackBinding {
            track_index: 0,
            bone_index: 0,
            target: TargetPath::Translation,
        }],
    };
    let mut mixer = AnimationMixer::new();
    mixer.set_rig(Rig {
        bones: vec![node],
        bone_paths: vec![vec!["root".into()]],
    });
    mixer.add_action(action);
    mixer.play("constant");

    mixer.update(0.25, &mut scene);
    assert_eq!(scene.render_mutations().entries.len(), 1);
    assert_eq!(scene.update_matrix_world().visited_nodes, 1);
    let cursor = scene.render_mutations().cursor;
    assert!(scene.acknowledge_render_mutations(cursor));

    // Re-sampling the same final pose must not restore-to-rest and write it
    // back again. The one walk below only settles previous-matrix history
    // owed by the real transition from the preceding update.
    mixer.update(0.25, &mut scene);
    assert!(scene.render_mutations().entries.is_empty());
    let settle = scene.update_matrix_world();
    assert_eq!(settle.updated_world_matrices, 0);
    assert_eq!(settle.settled_previous_matrices, 1);

    mixer.update(0.25, &mut scene);
    assert!(scene.render_mutations().entries.is_empty());
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);

    mixer.action("constant").unwrap().pause();
    mixer.update(0.25, &mut scene);
    assert_eq!(scene.render_mutations().entries.len(), 1);
    assert_eq!(scene.update_matrix_world().visited_nodes, 1);
    assert_eq!(scene.node(node).unwrap().transform.position, Vec3::ZERO);
    let cursor = scene.render_mutations().cursor;
    assert!(scene.acknowledge_render_mutations(cursor));

    // A paused action restores once, settles that real transition once, then
    // becomes completely static.
    mixer.update(0.25, &mut scene);
    assert!(scene.render_mutations().entries.is_empty());
    let settle = scene.update_matrix_world();
    assert_eq!(settle.updated_world_matrices, 0);
    assert_eq!(settle.settled_previous_matrices, 1);

    mixer.update(0.25, &mut scene);
    assert!(scene.render_mutations().entries.is_empty());
    assert_eq!(scene.update_matrix_world().visited_nodes, 0);
}
