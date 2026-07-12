//! Render Graph and Frame Management
//!
//! This module handles per-frame rendering organization:
//!
//! - [`RenderFrame`]: Per-frame render logic management
//! - [`RenderState`]: Render state (camera, time, etc.)
//! - [`ExtractedScene`]: Scene data extracted for GPU rendering
//! - [`FrameComposer`]: Chainable API for frame composition (hook-based RDG)
//! - [`RenderLists`]: CPU-only main/shadow draw-intent lists

pub mod bake;
pub mod composer;
pub mod core;
pub mod culling;
pub mod extracted;
pub mod frame;
pub mod passes;
pub mod render_state;
pub mod shadow_utils;

pub use composer::FrameComposer;
pub use extracted::{ExtractedRenderItem, ExtractedScene, ExtractedSkeleton};
pub use frame::{
    DrawCommand, DrawPlanId, RenderFrame, RenderIntent, RenderKey, RenderLists, ShadowRenderIntent,
};
#[cfg(feature = "debug_view")]
pub use render_state::DebugViewTarget;
pub use render_state::RenderState;
