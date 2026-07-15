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
#[cfg(any(test, feature = "rdg_bench_support"))]
#[doc(hidden)]
pub mod bench_support;
pub(crate) mod capacity;
pub mod composer;
pub mod core;
pub mod culling;
pub mod extracted;
pub mod frame;
pub(crate) mod gpu_driven_policy;
pub(crate) mod gpu_scene;
pub(crate) mod gpu_scene_capacity;
pub(crate) mod gpu_scene_submission;
pub(crate) mod material_plan;
pub mod passes;
pub mod render_state;
pub(crate) mod scene_pipeline_coordinator;
pub(crate) mod scene_pipeline_prepare;
pub(crate) mod scene_source_prepare;
pub mod shadow_utils;

pub use composer::FrameComposer;
pub use core::{
    MermaidDumpMode, RdgAttemptOutcome, RdgCommandEncodingFailure, RdgFrameDiagnostics,
    RdgFramePhase, RdgGpuDrivenFailure, RdgGpuDrivenPolicySnapshot,
    RdgPostSubmitPublicationFailure, RdgRenderTargetKind, RdgResourceMaterializationFailure,
    RdgRetainedCapacityStats, RdgSceneSubmissionBackend, RdgSceneSubmissionFamily,
    RdgSceneSubmissionFamilySnapshot, RdgSceneSubmissionSelectionReason, RdgSurfaceAcquireFailure,
};
#[cfg(feature = "rdg_bench_support")]
pub use core::{RdgDetailPhase, RdgGpuFrameTimings, RdgGpuPassTiming};
pub use extracted::{ExtractedRenderItem, ExtractedScene, ExtractedSkeleton, ItemShaderFeatures};
pub use frame::{RenderFrame, RenderIntent, RenderKey, RenderLists, ShadowRenderIntent};
#[cfg(feature = "debug_view")]
pub use render_state::DebugViewTarget;
pub use render_state::RenderState;
