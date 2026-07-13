//! CPU-only scene draw-intent collection.
//!
//! This module is the boundary between the extracted scene snapshot and the
//! post-DPE draw preparation path. It performs only logical asset-readiness
//! checks, view culling, and material classification. Physical resources,
//! layouts, bind groups, and pipelines are deliberately absent.

use glam::Vec3A;
use slotmap::Key;

use crate::core::view::ViewTarget;
use crate::graph::extracted::ExtractedScene;
use crate::graph::frame::{ExtractedSourceId, RenderIntent, RenderLists, ShadowRenderIntent};
use myth_assets::AssetServer;
use myth_resources::RenderableMaterialTrait;
use myth_resources::material::AlphaMode;
use myth_scene::camera::RenderCamera;

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
    assets: &AssetServer,
) {
    render_lists.begin_intent_collection();

    let camera_frustum = camera.frustum;
    let camera_pos = camera.position;
    let geo_guard = assets.geometries.read_lock();
    let mat_guard = assets.materials.read_lock();

    // Main-camera intents. Pipeline-dependent ordering is intentionally left
    // to the post-DPE plan finalizer; these records contain only CPU facts.
    for (item_idx, item) in extracted_scene.render_items.iter().enumerate() {
        let aabb = item.world_aabb;
        if aabb.is_finite() && !camera_frustum.intersects_aabb(&aabb) {
            continue;
        }

        let Some(geometry) = geo_guard.get_loaded(item.geometry) else {
            continue;
        };
        let Some(material) = mat_guard.get_loaded(item.material) else {
            continue;
        };
        if (geometry.version, geometry.identity)
            != (item.geometry_asset_version, item.geometry_asset_identity)
            || (material.version, material.identity)
                != (item.material_asset_version, item.material_asset_identity)
            || material.data.version() != item.material_version
        {
            continue;
        }

        let use_transmission = material.use_transmission();
        let is_transparent = material.is_transparent();
        if material.data.version() != item.material_version {
            continue;
        }
        render_lists.use_transmission |= use_transmission;

        let item_pos = Vec3A::from(item.world_matrix.w_axis.truncate());
        let source = render_lists.intern_frame_source(ExtractedSourceId::from_index(item_idx));
        let intent = RenderIntent {
            source,
            geometry_handle: item.geometry,
            material_handle: item.material,
            material_index: item.material.data().as_ffi() as u32,
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

        let shadow_key = (light_id, layer_index);
        let mut queue = render_lists
            .shadow_queues
            .remove(&shadow_key)
            .unwrap_or_default();

        for (item_idx, item) in extracted_scene.render_items.iter().enumerate() {
            if !item.cast_shadows {
                continue;
            }

            let aabb = item.world_aabb;
            if aabb.is_finite() && !view_frustum.intersects_aabb(&aabb) {
                continue;
            }

            let Some(geometry) = geo_guard.get_loaded(item.geometry) else {
                continue;
            };
            let Some(material) = mat_guard.get_loaded(item.material) else {
                continue;
            };
            if (geometry.version, geometry.identity)
                != (item.geometry_asset_version, item.geometry_asset_identity)
                || (material.version, material.identity)
                    != (item.material_asset_version, item.material_asset_identity)
                || material.data.version() != item.material_version
            {
                continue;
            }
            let alpha_mode = material.alpha_mode();
            if material.data.version() != item.material_version || alpha_mode == AlphaMode::Blend {
                continue;
            }

            let source = render_lists.intern_frame_source(ExtractedSourceId::from_index(item_idx));
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
