//! Exact chainable node editor.
//!
//! [`NodeEditor`] borrows a [`Scene`] mutably and provides a fluent API for
//! narrow, change-detected edits without exposing a broad `&mut Node`.
//!
//! All methods silently no-op when the handle is stale, so users never
//! encounter panics from dangling handles.
//!
//! # Example
//!
//! ```rust,ignore
//! scene.edit_node(handle)
//!     .set_position(0.0, 3.0, 0.0)
//!     .set_scale(2.0)
//!     .look_at(Vec3::ZERO)
//!     .set_visible(false);
//! ```
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::must_use_candidate)]
use glam::{Quat, Vec3};

use crate::scene::Scene;
use myth_core::NodeHandle;

/// Temporary mutable borrow of a scene node for chainable operations.
pub struct NodeEditor<'a> {
    scene: &'a mut Scene,
    handle: NodeHandle,
}

impl<'a> NodeEditor<'a> {
    #[inline]
    pub(crate) fn new(scene: &'a mut Scene, handle: NodeHandle) -> Self {
        Self { scene, handle }
    }

    /// Returns the underlying handle.
    #[inline]
    #[must_use]
    pub fn handle(&self) -> NodeHandle {
        self.handle
    }

    // -- Transform setters (chainable) --

    /// Sets the node's local position.
    #[inline]
    pub fn set_position(self, x: f32, y: f32, z: f32) -> Self {
        self.scene
            .set_node_position_tracked(self.handle, Vec3::new(x, y, z));
        self
    }

    /// Sets the node's local position from a Vec3.
    #[inline]
    pub fn set_position_vec(self, pos: Vec3) -> Self {
        self.scene.set_node_position_tracked(self.handle, pos);
        self
    }

    /// Sets uniform scale.
    #[inline]
    pub fn set_scale(self, s: f32) -> Self {
        self.scene
            .set_node_scale_tracked(self.handle, Vec3::splat(s));
        self
    }

    /// Sets non-uniform scale.
    #[inline]
    pub fn set_scale_xyz(self, x: f32, y: f32, z: f32) -> Self {
        self.scene
            .set_node_scale_tracked(self.handle, Vec3::new(x, y, z));
        self
    }

    /// Sets rotation from a quaternion.
    #[inline]
    pub fn set_rotation(self, quat: Quat) -> Self {
        self.scene.set_node_rotation_tracked(self.handle, quat);
        self
    }

    /// Sets rotation from Euler angles (XYZ intrinsic order, radians).
    #[inline]
    pub fn set_rotation_euler(self, x: f32, y: f32, z: f32) -> Self {
        let rotation = Quat::from_euler(glam::EulerRot::XYZ, x, y, z);
        self.scene.set_node_rotation_tracked(self.handle, rotation);
        self
    }

    /// Rotates around the Y axis by `angle` radians (cumulative).
    #[inline]
    pub fn rotate_y(self, angle: f32) -> Self {
        if let Some(node) = self.scene.node(self.handle) {
            let rotation = node.transform.rotation * Quat::from_rotation_y(angle);
            self.scene.set_node_rotation_tracked(self.handle, rotation);
        }
        self
    }

    /// Rotates around the X axis by `angle` radians (cumulative).
    #[inline]
    pub fn rotate_x(self, angle: f32) -> Self {
        if let Some(node) = self.scene.node(self.handle) {
            let rotation = node.transform.rotation * Quat::from_rotation_x(angle);
            self.scene.set_node_rotation_tracked(self.handle, rotation);
        }
        self
    }

    /// Orients the node to face `target` (in parent-local space).
    #[inline]
    pub fn look_at(self, target: Vec3) -> Self {
        if let Some(node) = self.scene.node(self.handle) {
            let mut transform = node.transform;
            transform.look_at(target, Vec3::Y);
            self.scene
                .set_node_rotation_tracked(self.handle, transform.rotation);
        }
        self
    }

    /// Sets node visibility.
    #[inline]
    pub fn set_visible(self, visible: bool) -> Self {
        self.scene.set_node_visible_tracked(self.handle, visible);
        self
    }

    /// Sets the mesh cast_shadows flag (no-op if node has no mesh).
    #[inline]
    pub fn set_cast_shadows(self, cast: bool) -> Self {
        self.scene
            .set_mesh_shadow_flags_tracked(self.handle, Some(cast), None);
        self
    }

    /// Sets the mesh receive_shadows flag (no-op if node has no mesh).
    #[inline]
    pub fn set_receive_shadows(self, receive: bool) -> Self {
        self.scene
            .set_mesh_shadow_flags_tracked(self.handle, None, Some(receive));
        self
    }

    /// Convenience: enable/disable both cast and receive shadows.
    #[inline]
    pub fn set_shadows(self, cast: bool, receive: bool) -> Self {
        self.scene
            .set_mesh_shadow_flags_tracked(self.handle, Some(cast), Some(receive));
        self
    }
}
