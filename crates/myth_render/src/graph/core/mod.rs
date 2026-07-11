pub mod allocator;
pub mod arena;
pub mod blackboard;
pub mod builder;
pub mod context;
#[cfg(feature = "3dgs")]
pub(crate) mod gaussian;
pub mod graph;
pub mod node;
pub mod persistent;
pub mod template_pass;
pub mod types;

pub use allocator::{SubViewKey, TransientPool};
pub use arena::FrameArena;
pub use blackboard::{CustomPassHook, GraphBlackboard, HookStage};
pub use builder::PassBuilder;
pub use context::{
    BindGroupBuilder, BindableResource, ClusteredScreenBindings, ExecuteContext, ExtractContext,
    GraphBinding, PrepareContext, RawBufferBinding, RawSamplerBinding, RawTextureViewBinding,
    ViewResolver, build_screen_bind_group,
};
pub use graph::{GraphStorage, RenderGraph};
pub use node::PassNode;
pub use persistent::{
    CacheCommitToken, CacheId, CacheKey, CachePlan, CachePublicationReceipt, HistoryCommitToken,
    HistoryPlan, HistoryRegistry, HistorySetId, HistoryTextureRef, PersistentTextureId,
    PersistentTextureRef, PersistentTextureSpec, TemporalDomainKey,
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
