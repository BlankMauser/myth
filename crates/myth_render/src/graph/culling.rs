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
use crate::graph::frame::{RenderIntent, RenderLists, ShadowRenderIntent};
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
    render_lists.opaque.clear();
    render_lists.transparent.clear();
    render_lists.shadow_queues.clear();
    render_lists.use_transmission = false;

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
        let intent = RenderIntent {
            source_item_index: checked_source_item_index(item_idx),
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
    for view in &render_lists.active_views {
        let ViewTarget::ShadowLight {
            light_id,
            layer_index,
        } = view.target
        else {
            continue;
        };

        let queue = render_lists
            .shadow_queues
            .entry((light_id, layer_index))
            .or_default();

        for (item_idx, item) in extracted_scene.render_items.iter().enumerate() {
            if !item.cast_shadows {
                continue;
            }

            let aabb = item.world_aabb;
            if aabb.is_finite() && !view.frustum.intersects_aabb(&aabb) {
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

            queue.push(ShadowRenderIntent {
                source_item_index: checked_source_item_index(item_idx),
                geometry_handle: item.geometry,
                material_handle: item.material,
            });
        }
    }
}

#[inline]
fn checked_source_item_index(item_idx: usize) -> u32 {
    u32::try_from(item_idx).expect("one frame cannot extract more than u32::MAX render items")
}
