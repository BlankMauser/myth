//! CPU-only scene draw-intent collection.
//!
//! This module is the boundary between the extracted scene snapshot and the
//! post-DPE draw preparation path. It performs only view culling and consumes
//! classification facts captured by extraction. Live asset dictionaries,
//! physical resources, layouts, bind groups, and pipelines are deliberately
//! absent; later plan capture rejects any asset-generation drift fail-closed.

use glam::Vec3A;

use crate::core::view::ViewTarget;
use crate::graph::extracted::{ExtractedScene, MaterialClassification};
use crate::graph::frame::{ExtractedSourceId, RenderIntent, RenderLists, ShadowRenderIntent};
use crate::graph::gpu_driven_policy::GpuDrivenFamilyMask;
use myth_scene::camera::RenderCamera;
use myth_scene::light::LightKind;

/// Collect CPU-only main-view and per-shadow-view draw intents.
///
/// The caller must have populated [`RenderLists::active_views`]. Existing
/// intent lists are cleared while active views and shared light/global frame
/// metadata are preserved. Every active shadow view receives a queue entry,
/// including views with no surviving casters.
pub(crate) fn collect_draw_intents(
    extracted_scene: &ExtractedScene,
    render_lists: &mut RenderLists,
    camera: &RenderCamera,
    gpu_owns_opaque: bool,
    gpu_owns_shadows: bool,
) {
    collect_draw_intents_impl::<false>(
        extracted_scene,
        render_lists,
        camera,
        gpu_owns_opaque,
        gpu_owns_shadows,
    );
}

/// Collect only cohorts which were not assigned to GPU Scene by the current
/// per-source family ownership transaction. This is the M7.5 Auto path:
/// unsupported and low-reuse items remain CPU-owned without duplicating a
/// retained GPU cohort in Main, Prepass, Shadow2D, or ShadowCube.
pub(crate) fn collect_draw_intents_with_family_ownership(
    extracted_scene: &ExtractedScene,
    render_lists: &mut RenderLists,
    camera: &RenderCamera,
) {
    collect_draw_intents_impl::<true>(extracted_scene, render_lists, camera, false, false);
}

fn collect_draw_intents_impl<const PER_SOURCE_FAMILIES: bool>(
    extracted_scene: &ExtractedScene,
    render_lists: &mut RenderLists,
    camera: &RenderCamera,
    gpu_owns_opaque: bool,
    gpu_owns_shadows: bool,
) {
    let camera_frustum = camera.frustum;
    let camera_pos = camera.position;

    // Main-camera intents. Pipeline-dependent ordering is intentionally left
    // to the post-DPE plan finalizer; these records contain only CPU facts.
    for (item_idx, item) in extracted_scene.render_items.iter().enumerate() {
        let use_transmission = item
            .material_classification
            .contains(MaterialClassification::USES_TRANSMISSION);
        let is_transparent = item
            .material_classification
            .contains(MaterialClassification::TRANSPARENT);
        render_lists.use_transmission |= use_transmission;

        // A selected GPU Scene batch owns the complete pre-frustum opaque
        // domain. Do not repeat CPU visibility, source interning, or intent
        // writes for objects that will be compacted by the Cull node.
        let extracted_source = ExtractedSourceId::from_index(item_idx);
        let gpu_owns_main = if PER_SOURCE_FAMILIES {
            render_lists
                .gpu_family_ownership(extracted_source)
                .contains(GpuDrivenFamilyMask::MAIN_PREPASS)
        } else {
            gpu_owns_opaque
        };
        if gpu_owns_main && !is_transparent && !use_transmission {
            continue;
        }

        let aabb = item.world_aabb;
        if aabb.is_finite() && !camera_frustum.intersects_aabb(&aabb) {
            continue;
        }

        let item_pos = Vec3A::from(item.world_matrix.w_axis.truncate());
        let source = render_lists.intern_frame_source(extracted_source);
        let intent = RenderIntent {
            source,
            geometry_handle: item.geometry,
            material_handle: item.material,
            material_index: item.material_index,
            distance_sq: camera_pos.distance_squared(item_pos),
        };

        if is_transparent || use_transmission {
            render_lists.insert_transparent(intent);
        } else {
            render_lists.insert_opaque(intent);
        }
    }

    // Shadow queues retain the exact active-view topology even when a view
    // has no casters, keeping an empty live plan distinct from no plan.
    if !PER_SOURCE_FAMILIES && gpu_owns_shadows {
        return;
    }

    for view_index in 0..render_lists.active_views.len() {
        let view = &render_lists.active_views[view_index];
        let ViewTarget::ShadowLight {
            light_id,
            layer_index,
        } = view.target
        else {
            continue;
        };
        let view_frustum = view.frustum;
        let gpu_shadow_family = if PER_SOURCE_FAMILIES {
            gpu_shadow_family_for_view(extracted_scene, view.light_buffer_index, light_id)
        } else {
            GpuDrivenFamilyMask::empty()
        };

        let shadow_key = (light_id, layer_index);
        let mut queue = render_lists
            .shadow_queues
            .remove(&shadow_key)
            .unwrap_or_default();

        for (item_idx, item) in extracted_scene.render_items.iter().enumerate() {
            if !item.cast_shadows {
                continue;
            }

            let extracted_source = ExtractedSourceId::from_index(item_idx);
            if PER_SOURCE_FAMILIES
                && render_lists
                    .gpu_family_ownership(extracted_source)
                    .contains(gpu_shadow_family)
            {
                continue;
            }

            let aabb = item.world_aabb;
            if aabb.is_finite() && !view_frustum.intersects_aabb(&aabb) {
                continue;
            }

            let source = render_lists.intern_frame_source(extracted_source);
            queue.push(ShadowRenderIntent {
                source,
                geometry_handle: item.geometry,
                material_handle: item.material,
            });
        }
        assert!(
            render_lists
                .shadow_queues
                .insert(shadow_key, queue)
                .is_none(),
            "one active shadow view was inserted twice without ownership transfer"
        );
    }
}

/// Map one active shadow view onto its independently selectable GPU family.
/// A stale light/view join conservatively falls back to the 2D family, exactly
/// matching Composer's existing packed-family workload classification.
#[inline]
fn gpu_shadow_family_for_view(
    extracted_scene: &ExtractedScene,
    light_buffer_index: usize,
    light_id: u64,
) -> GpuDrivenFamilyMask {
    if extracted_scene
        .lights
        .get(light_buffer_index)
        .is_some_and(|light| light.id == light_id && matches!(light.kind, LightKind::Point(_)))
    {
        GpuDrivenFamilyMask::SHADOW_CUBE
    } else {
        GpuDrivenFamilyMask::SHADOW_2D
    }
}
