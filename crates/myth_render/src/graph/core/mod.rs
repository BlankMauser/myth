pub mod allocator;
pub mod arena;
mod asset_texture;
pub mod blackboard;
pub mod builder;
pub mod context;
mod diagnostics;
#[cfg(feature = "3dgs")]
pub(crate) mod gaussian;
mod global_state;
pub mod graph;
pub mod node;
pub mod persistent;
mod scene_buffer;
pub(crate) mod scene_submission;
pub mod template_pass;
pub mod types;

pub use allocator::{SubViewKey, TransientPool};
pub use arena::FrameArena;
pub use asset_texture::AssetTextureImportId;
pub(crate) use asset_texture::AssetTextureResolver;
pub use blackboard::{CustomPassHook, GraphBlackboard, HookStage};
pub use builder::PassBuilder;
pub(crate) use context::SceneDrawPrepareContext;
pub use context::{
    BindGroupBuilder, BindableResource, ClusteredScreenBindings, ExecuteContext, ExtractContext,
    GraphBinding, PrepareContext, RawBufferBinding, RawSamplerBinding, RawTextureViewBinding,
    ViewResolver, build_screen_bind_group,
};
pub use diagnostics::{
    RdgAttemptOutcome, RdgFrameDiagnostics, RdgFramePhase, RdgRenderTargetKind,
    RdgResourceMaterializationFailure, RdgSurfaceAcquireFailure,
};
pub(crate) use diagnostics::{
    RdgComposerAttemptGuard, RdgDiagnosticsRecorder, RdgRendererAttemptGuard,
};
pub(crate) use global_state::{GlobalStateMaterializationFailure, GlobalStateSources};
pub use graph::{GraphStorage, MermaidDumpMode, RdgRetainedCapacityStats, RenderGraph};
pub use node::PassNode;
pub use persistent::{
    CacheCommitToken, CacheId, CacheKey, CachePlan, CachePublicationReceipt, HistoryCommitToken,
    HistoryPlan, HistoryRegistry, HistorySetId, HistoryTextureRef, PersistentTextureId,
    PersistentTextureRef, PersistentTextureSpec, TemporalDomainKey,
};
pub use scene_buffer::SceneBufferImportId;
pub(crate) use scene_buffer::{
    SceneBufferMaterializationFailure, SceneBufferResolver, SceneBufferSources,
};
pub use template_pass::{
    ComputePassBuilder, RenderPassBuilder, StandardComputeNode, StandardFullscreenNode,
    TemplateBindingLayoutDesc, TemplateComputePass, TemplateFullscreenPass,
    TemplatePassBindingsBuilder, TemplatePassDescriptor, TemplateShaderSource,
};
pub use types::{
    Buffer, BufferDesc, BufferNodeId, GraphResourceType, RenderTargetOps, ResourceKind,
    ResourceNodeId, ResourceRecord, Texture, TextureDesc, TextureNodeId,
};
