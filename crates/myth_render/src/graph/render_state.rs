//! Render State
//!
//! Manages per-frame render state (camera, time, and other global uniforms).

use std::sync::atomic::{AtomicU32, Ordering};

use myth_resources::buffer::CpuBuffer;
use myth_resources::uniforms::RenderStateUniforms;
use myth_scene::camera::RenderCamera;
use rustc_hash::FxHashMap;

use crate::renderer::FrameTime;

// ─── Debug View Target (compile-time gated) ─────────────────────────────────

/// Semantic identifier for an intermediate render texture to visualise.
///
/// This enum lives in the **render state layer** — it carries no frame-specific
/// physical IDs (`TextureNodeId`).  The [`FrameComposer`] resolves it each
/// frame into a concrete RDG resource, safely handling cases where the
/// target texture was not produced (e.g. SSAO disabled).
///
/// Derived from [`DebugViewMode`](myth_scene::camera::DebugViewMode) during
/// the extract phase.  Material-override modes (Albedo, Roughness, Metalness)
/// do not use this target resolution — they are handled via shader defines.
#[cfg(feature = "debug_view")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DebugViewTarget {
    #[default]
    None,
    SceneDepth,
    SceneNormal,
    Velocity,
    SsaoRaw,
    ClusterHeatmap,
    SsgiRaw,
    SsgiDenoised,
    SsrRaw,
    SsrResolved,
    SsrTraceDebug,
}

#[cfg(feature = "debug_view")]
impl DebugViewTarget {
    /// Maps a logical [`DebugViewMode`] to the render-layer target.
    ///
    /// Material-override modes return `None` since they bypass the
    /// post-process debug overlay entirely.
    #[must_use]
    pub fn from_mode(mode: myth_scene::camera::DebugViewMode) -> Self {
        use myth_scene::camera::DebugViewMode;
        match mode {
            DebugViewMode::SSAO => Self::SsaoRaw,
            DebugViewMode::Normal => Self::SceneNormal,
            DebugViewMode::Velocity => Self::Velocity,
            DebugViewMode::Depth => Self::SceneDepth,
            DebugViewMode::ClusterHeatmap => Self::ClusterHeatmap,
            DebugViewMode::SsgiRaw => Self::SsgiRaw,
            DebugViewMode::SsgiDenoised => Self::SsgiDenoised,
            DebugViewMode::SsrRaw => Self::SsrRaw,
            DebugViewMode::SsrResolved => Self::SsrResolved,
            DebugViewMode::SsrTraceDiagnostic | DebugViewMode::SsrTraceState => Self::SsrTraceDebug,
            _ => Self::None,
        }
    }

    /// Returns `true` when the source texture is a depth attachment.
    #[must_use]
    pub const fn is_depth(self) -> bool {
        matches!(self, Self::SceneDepth | Self::ClusterHeatmap)
    }
}

static NEXT_RENDER_STATE_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TemporalViewKey {
    scene_id: u32,
    camera_id: u128,
}

#[derive(Debug, Clone, Copy)]
struct TemporalViewState {
    view_projection: glam::Mat4,
    jitter: glam::Vec2,
    unjittered_view_projection: glam::Mat4,
    serial: u64,
}

pub struct RenderState {
    pub id: u32,
    uniforms: CpuBuffer<RenderStateUniforms>,
    /// Previous matrices and local frame sequence, partitioned by temporal
    /// render domain so interleaved scene/camera rendering cannot cross-talk.
    temporal_views: FxHashMap<TemporalViewKey, TemporalViewState>,
    current_temporal_serial: u64,
    /// Active debug-view mode (from camera settings, resolved per-frame).
    #[cfg(feature = "debug_view")]
    pub debug_view_mode: myth_scene::camera::DebugViewMode,
    /// Scale factor for debug view (e.g. velocity amplification).
    #[cfg(feature = "debug_view")]
    pub debug_view_scale: f32,
}

impl Default for RenderState {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderState {
    pub fn new() -> Self {
        Self {
            id: NEXT_RENDER_STATE_ID.fetch_add(1, Ordering::Relaxed),
            uniforms: CpuBuffer::new(
                RenderStateUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("RenderState Uniforms"),
            ),
            temporal_views: FxHashMap::default(),
            current_temporal_serial: 0,
            #[cfg(feature = "debug_view")]
            debug_view_mode: myth_scene::camera::DebugViewMode::None,
            #[cfg(feature = "debug_view")]
            debug_view_scale: 100.0,
        }
    }

    pub fn uniforms(&self) -> &CpuBuffer<RenderStateUniforms> {
        &self.uniforms
    }

    pub fn uniforms_mut(&mut self) -> myth_resources::buffer::BufferGuard<'_, RenderStateUniforms> {
        self.uniforms.write()
    }

    /// Sequence number of the current `(scene, camera)` temporal domain.
    /// It advances on every extract attempt for that domain, independently of
    /// frames spent rendering other domains.
    #[must_use]
    pub const fn temporal_serial(&self) -> u64 {
        self.current_temporal_serial
    }

    pub fn update(
        &mut self,
        scene_id: u32,
        camera: &RenderCamera,
        frame_time: FrameTime,
        viewport_size: (u32, u32),
    ) {
        let key = TemporalViewKey {
            scene_id,
            camera_id: camera.camera_id,
        };
        let previous = self.temporal_views.get(&key).copied();
        let prev_vp = previous.map_or(glam::Mat4::IDENTITY, |state| state.view_projection);
        let prev_j = previous.map_or(glam::Vec2::ZERO, |state| state.jitter);
        let prev_unjittered_vp = previous.map_or(glam::Mat4::IDENTITY, |state| {
            state.unjittered_view_projection
        });
        let temporal_serial = previous.map_or(1, |state| {
            state
                .serial
                .checked_add(1)
                .expect("temporal view serial overflow")
        });

        let unjittered_projection_inverse = camera.unjittered_projection.inverse();
        let unjittered_vp = camera.unjittered_projection * camera.view_matrix;
        let focal_x = (camera.projection_matrix.x_axis.x * viewport_size.0 as f32 * 0.5).abs();
        let focal_y = (camera.projection_matrix.y_axis.y * viewport_size.1 as f32 * 0.5).abs();

        let mut u = self.uniforms_mut();
        u.view_projection = camera.view_projection_matrix;
        u.view_projection_inverse = camera.view_projection_matrix.inverse();
        u.projection_matrix = camera.projection_matrix;
        u.projection_inverse = camera.projection_matrix.inverse();
        u.unjittered_projection_matrix = camera.unjittered_projection;
        u.unjittered_projection_inverse = unjittered_projection_inverse;
        u.view_matrix = camera.view_matrix;
        u.prev_view_projection = prev_vp;
        u.unjittered_view_projection = unjittered_vp;
        u.unjittered_view_projection_inverse = unjittered_vp.inverse();
        u.prev_unjittered_view_projection = prev_unjittered_vp;
        u.camera_position = camera.position.into();
        u.viewport = glam::Vec2::new(viewport_size.0 as f32, viewport_size.1 as f32);
        u.focal = glam::Vec2::new(focal_x, focal_y);
        u.time = frame_time.time % 7200.0; // Wrap time to avoid precision issues in shaders
        u.time_cycle_2pi = frame_time.time % std::f32::consts::PI * 2.0;
        u.delta_time = frame_time.delta_time;
        u.jitter = camera.jitter;
        u.prev_jitter = prev_j;
        u.camera_near = camera.near;
        u.camera_far = camera.far;
        drop(u);

        // Latch current values only for this scene/camera domain.
        self.temporal_views.insert(
            key,
            TemporalViewState {
                view_projection: camera.view_projection_matrix,
                jitter: camera.jitter,
                unjittered_view_projection: unjittered_vp,
                serial: temporal_serial,
            },
        );
        self.current_temporal_serial = temporal_serial;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use myth_scene::camera::Camera;

    fn camera(camera_id: u128) -> RenderCamera {
        let mut camera = Camera::new_perspective(45.0, 1.0, 0.1);
        let mut render_camera = camera.extract_render_camera();
        render_camera.camera_id = camera_id;
        render_camera
    }

    fn frame_time() -> FrameTime {
        FrameTime {
            time: 0.0,
            delta_time: 1.0 / 60.0,
            frame_count: 0,
        }
    }

    #[test]
    fn temporal_serial_and_previous_state_are_partitioned_by_scene_and_camera() {
        let mut state = RenderState::new();
        let camera_a = camera(1);
        let camera_b = camera(2);

        state.update(10, &camera_a, frame_time(), (64, 64));
        assert_eq!(state.temporal_serial(), 1);
        state.update(10, &camera_b, frame_time(), (64, 64));
        assert_eq!(state.temporal_serial(), 1);
        state.update(10, &camera_a, frame_time(), (64, 64));
        assert_eq!(state.temporal_serial(), 2);
        state.update(11, &camera_a, frame_time(), (64, 64));
        assert_eq!(state.temporal_serial(), 1);
        assert_eq!(state.temporal_views.len(), 3);
    }
}
