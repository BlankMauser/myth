//! Render Graph and Frame Management
//!
//! This module handles per-frame rendering organization:
//!
//! - [`RenderFrame`]: Per-frame render logic management
//! - [`RenderState`]: Render state (camera, time, etc.)
//! - [`ExtractedScene`]: Scene data extracted for GPU rendering
//! - [`FrameComposer`]: Chainable API for frame composition (hook-based RDG)
//! - [`RenderLists`]: CPU-only main/shadow draw-intent lists

pub(crate) mod asset_frame;
pub(crate) mod asset_texture_batch;
pub mod bake;
#[cfg(any(test, feature = "rdg_bench_support"))]
#[doc(hidden)]
pub mod bench_support;
pub(crate) mod capacity;
pub mod composer;
pub mod core;
pub mod culling;
pub mod extracted;
pub mod frame;
pub(crate) mod material_plan;
pub mod passes;
pub mod render_state;
pub mod shadow_utils;

pub use composer::FrameComposer;
pub use core::{
    MermaidDumpMode, RdgAttemptOutcome, RdgFrameDiagnostics, RdgFramePhase, RdgRenderTargetKind,
    RdgResourceMaterializationFailure, RdgRetainedCapacityStats, RdgSurfaceAcquireFailure,
};
pub use extracted::{ExtractedRenderItem, ExtractedScene, ExtractedSkeleton};
pub use frame::{RenderFrame, RenderIntent, RenderKey, RenderLists, ShadowRenderIntent};
#[cfg(feature = "debug_view")]
pub use render_state::DebugViewTarget;
pub use render_state::RenderState;
