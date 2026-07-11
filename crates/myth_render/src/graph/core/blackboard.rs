//! Graph Blackboard & Custom Pass Hooks
//!
//! The [`GraphBlackboard`] is a lightweight, pass-by-value **cursor** over
//! well-known resource slots from the current frame's render graph.  External
//! code (e.g. UI plugins) receives a copy, wires its passes, and returns an
//! updated copy — there is no shared mutable state.
//!
//! [`CustomPassHook`] is a builder-time callback that lets external code
//! inject arbitrary [`PassNode`]s into the graph at a chosen stage.

use super::types::{BufferNodeId, TextureNodeId};

/// Well-known resource slots published by the engine's graph builder.
///
/// This is a **value type** (`Copy + Clone`).  Hooks receive the blackboard
/// by value, modify the slots they care about, and return a fresh copy.
/// The Rust compiler enforces that every hook path returns a valid
/// blackboard — forgetting to do so is a compile-time error.
///
/// # Fields
///
/// | Slot | Semantic | Typical consumer |
/// |------|----------|------------------|
/// | `scene_color` | HDR scene colour buffer | Custom post-FX |
/// | `scene_depth` | Main depth buffer (reverse-Z) | Depth-aware FX |
/// | `scene_hiz` | Reverse-Z Hi-Z pyramid (max-depth mip chain) | Hierarchical screen-space ray tracing |
/// | `environment_base_cube` | Scene environment cube before prefiltering | Custom reflections / sky lighting |
/// | `environment_pmrem` | Prefiltered scene environment cube | Custom image-based lighting |
/// | `environment_pmrem_max_mip_level` | Max LOD of `environment_pmrem` | Roughness-to-LOD mapping |
/// | `atmosphere_transmittance` | Atmosphere transmittance LUT | Custom lit composites |
/// | `atmosphere_bake_params` | Atmosphere bake uniform buffer | Custom lit composites |
/// | `surface_out` | Final swap-chain output | UI overlay |
#[derive(Clone, Copy)]
pub struct GraphBlackboard {
    /// HDR scene colour render target (written by Opaque / Skybox / Transparent).
    pub scene_color: Option<TextureNodeId>,
    /// Main depth buffer (reverse-Z, written by scene passes).
    pub scene_depth: Option<TextureNodeId>,
    /// Reverse-Z Hi-Z pyramid storing the maximum depth per mip texel.
    pub scene_hiz: Option<TextureNodeId>,
    /// Optional scene environment cube before PMREM prefiltering.
    ///
    /// This is the logical RDG resource, not a retained physical view. Reading
    /// it from a custom pass keeps same-frame environment producers ordered
    /// and lets DPE remove them when no live pass consumes the result.
    pub environment_base_cube: Option<TextureNodeId>,
    /// Optional prefiltered scene environment cube used for image-based
    /// lighting. This may refer to either a previously published cache entry
    /// or a same-frame persistent-cache writer.
    pub environment_pmrem: Option<TextureNodeId>,
    /// Maximum valid mip level of `environment_pmrem`. `Some(0.0)` describes
    /// a valid 1×1 PMREM; availability must not be inferred from the value.
    pub environment_pmrem_max_mip_level: Option<f32>,
    /// Optional atmosphere transmittance LUT for custom passes that need the
    /// same celestial-light attenuation as the main lighting path.
    pub atmosphere_transmittance: Option<TextureNodeId>,
    /// Optional procedural-sky bake parameters paired with the transmittance
    /// LUT. Falls back to the renderer default when unavailable.
    pub atmosphere_bake_params: Option<BufferNodeId>,
    /// Final swap-chain output target.  UI and overlays should write here.
    pub surface_out: TextureNodeId,
}

/// Injection stage for custom pass hooks.
///
/// Determines **when** in the pipeline the hook's passes are wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookStage {
    /// After scene rendering, before post-processing (Bloom, ToneMap, FXAA).
    BeforePostProcess,
    /// After all post-processing, before surface presentation.
    /// This is the typical stage for UI overlays.
    AfterPostProcess,
}

/// A builder-time callback that injects passes into the render graph.
///
/// The closure **consumes** the current [`GraphBlackboard`] and **returns**
/// a potentially updated copy.  This pure-functional signature ensures that
/// the caller cannot silently forget to propagate modified resource slots
/// — the Rust type system makes omission a compile-time error.
///
/// # Example
///
/// ```rust,ignore
/// renderer.add_custom_pass_hook(HookStage::AfterPostProcess, |graph, cursor| {
///     let new_surface = graph.add_pass("UI_Pass", |builder| {
///         let exported = builder.mutate_texture(cursor.surface_out, "Surface_With_UI");
///         let ui_node = UiPassNode { target_tex: exported, .. };
///         (ui_node, exported)
///     });
///     GraphBlackboard { surface_out: new_surface, ..cursor }
/// });
/// ```
pub type CustomPassHook<'a> = Box<
    dyn for<'g> FnMut(&mut super::graph::RenderGraph<'g>, GraphBlackboard) -> GraphBlackboard + 'a,
>;
