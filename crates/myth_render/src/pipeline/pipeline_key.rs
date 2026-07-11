//! Strongly-typed pipeline cache keys.
//!
//! `wgpu` descriptor types (`ColorTargetState`, `DepthStencilState`, …) do not
//! implement `Hash` / `Eq`. This module defines *mirror* types that extract the
//! fields relevant for pipeline identity and derive the correct trait impls.
//!
//! Three key families are provided:
//!
//! - [`GraphicsPipelineKey`] — material-driven scene geometry pipelines
//!   (opaque, transparent, shadow).
//! - [`FullscreenPipelineKey`] — post-processing / fullscreen passes
//!   (bloom, SSAO, FXAA, tone map, SSSS, skybox…).
//! - [`SimpleGeometryPipelineKey`] — simplified geometry passes (prepass, shadow).
//! - [`ComputePipelineKey`] — compute shader pipelines (IBL, clustered lighting).

use bitflags::bitflags;
use std::hash::{Hash, Hasher};

// ─── Hashable Mirror Types ────────────────────────────────────────────────────

/// Hashable mirror of `wgpu::BlendComponent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendComponentKey {
    pub src_factor: wgpu::BlendFactor,
    pub dst_factor: wgpu::BlendFactor,
    pub operation: wgpu::BlendOperation,
}

impl From<wgpu::BlendComponent> for BlendComponentKey {
    fn from(b: wgpu::BlendComponent) -> Self {
        Self {
            src_factor: b.src_factor,
            dst_factor: b.dst_factor,
            operation: b.operation,
        }
    }
}

/// Hashable mirror of `wgpu::BlendState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendStateKey {
    pub color: BlendComponentKey,
    pub alpha: BlendComponentKey,
}

impl From<wgpu::BlendState> for BlendStateKey {
    fn from(b: wgpu::BlendState) -> Self {
        Self {
            color: b.color.into(),
            alpha: b.alpha.into(),
        }
    }
}

/// Hashable mirror of `wgpu::ColorTargetState`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColorTargetKey {
    pub format: wgpu::TextureFormat,
    pub blend: Option<BlendStateKey>,
    pub write_mask: u32, // wgpu::ColorWrites bits
}

impl From<wgpu::ColorTargetState> for ColorTargetKey {
    fn from(c: wgpu::ColorTargetState) -> Self {
        Self {
            format: c.format,
            blend: c.blend.map(Into::into),
            write_mask: c.write_mask.bits(),
        }
    }
}

/// Hashable mirror of `wgpu::StencilFaceState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilFaceKey {
    pub compare: wgpu::CompareFunction,
    pub fail_op: wgpu::StencilOperation,
    pub depth_fail_op: wgpu::StencilOperation,
    pub pass_op: wgpu::StencilOperation,
}

impl From<wgpu::StencilFaceState> for StencilFaceKey {
    fn from(s: wgpu::StencilFaceState) -> Self {
        Self {
            compare: s.compare,
            fail_op: s.fail_op,
            depth_fail_op: s.depth_fail_op,
            pass_op: s.pass_op,
        }
    }
}

/// Hashable mirror of `wgpu::StencilState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilStateKey {
    pub front: StencilFaceKey,
    pub back: StencilFaceKey,
    pub read_mask: u32,
    pub write_mask: u32,
}

impl From<wgpu::StencilState> for StencilStateKey {
    fn from(s: wgpu::StencilState) -> Self {
        Self {
            front: s.front.into(),
            back: s.back.into(),
            read_mask: s.read_mask,
            write_mask: s.write_mask,
        }
    }
}

/// Hashable mirror of `wgpu::DepthBiasState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthBiasKey {
    pub constant: i32,
    pub slope_scale_bits: u32,
    pub clamp_bits: u32,
}

impl From<wgpu::DepthBiasState> for DepthBiasKey {
    fn from(b: wgpu::DepthBiasState) -> Self {
        Self {
            constant: b.constant,
            slope_scale_bits: b.slope_scale.to_bits(),
            clamp_bits: b.clamp.to_bits(),
        }
    }
}

/// Hashable mirror of `wgpu::DepthStencilState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthStencilKey {
    pub format: wgpu::TextureFormat,
    pub depth_write_enabled: Option<bool>,
    pub depth_compare: Option<wgpu::CompareFunction>,
    pub stencil: StencilStateKey,
    pub bias: DepthBiasKey,
}

impl From<wgpu::DepthStencilState> for DepthStencilKey {
    fn from(d: wgpu::DepthStencilState) -> Self {
        Self {
            format: d.format,
            depth_write_enabled: d.depth_write_enabled,
            depth_compare: d.depth_compare,
            stencil: d.stencil.into(),
            bias: d.bias.into(),
        }
    }
}

/// Hashable mirror of `wgpu::MultisampleState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultisampleKey {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}

impl From<wgpu::MultisampleState> for MultisampleKey {
    fn from(m: wgpu::MultisampleState) -> Self {
        Self {
            count: m.count,
            mask: m.mask,
            alpha_to_coverage_enabled: m.alpha_to_coverage_enabled,
        }
    }
}

// ─── Pipeline Keys ────────────────────────────────────────────────────────────

bitflags! {
    /// Bitmask representation of boolean graphics pipeline states.
    /// This ensures efficient hashing and avoids excessive boolean fields.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PipelineFlags: u32 {
        /// Enables depth writing.
        const DEPTH_WRITE         = 1 << 0;
        /// Enables Alpha-to-Coverage for multisampling.
        const ALPHA_TO_COVERAGE   = 1 << 1;
    }
}

/// L2 cache key for material-driven scene geometry pipelines.
///
/// This is the successor to the old `PipelineKey`. It fully describes all
/// wgpu pipeline state that is relevant for deduplication. The `shader_hash`
/// collapses the (shader source identity + compilation options) tuple into a
/// single `u64`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphicsPipelineKey {
    pub shader_hash: u64,
    pub vertex_layout_id: u64,
    /// `[Global, Material, Object, Screen]`
    pub bind_group_layout_ids: [u64; 4],
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub front_face: wgpu::FrontFace,
    pub depth_compare: wgpu::CompareFunction,
    pub blend_state: Option<BlendStateKey>,
    /// Explicit render-target layout for the scene pass.
    ///
    /// Opaque pipelines may append shared material/specular MRTs after the
    /// main scene color target, and those auxiliary targets may use formats
    /// that differ from the main HDR surface.
    pub color_targets: smallvec::SmallVec<[ColorTargetKey; 3]>,
    pub depth_format: wgpu::TextureFormat,
    pub sample_count: u32,

    pub flags: PipelineFlags,
}

/// L2 cache key for non-material render pipelines.
///
/// Covers fullscreen / post-processing passes **and** other no-vertex-buffer
/// utility passes such as Skybox. The standard constructor uses a triangle
/// list, while callers that emit strips can specialize the primitive topology.
///
/// The `shader_hash` is an xxh3-128 of the final WGSL source code.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FullscreenPipelineKey {
    /// xxh3-128 hash of the final WGSL source code.
    pub shader_hash: u128,
    /// Hash of fragment-stage compilation options such as WGSL overrides.
    pub compilation_hash: u64,
    /// Color render targets (usually 1).
    pub color_targets: smallvec::SmallVec<[ColorTargetKey; 2]>,
    /// Depth/stencil configuration (optional).
    pub depth_stencil: Option<DepthStencilKey>,
    /// Multisample configuration.
    /// Skybox needs MSAA-aware pipelines; pure post-process passes pass 1×.
    pub multisample: MultisampleKey,
    /// Primitive topology for no-vertex-buffer draws.
    pub topology: wgpu::PrimitiveTopology,
}

impl FullscreenPipelineKey {
    /// Convenience constructor for a standard fullscreen-triangle pipeline.
    ///
    /// Sets `multisample` to 1× (default).
    #[must_use]
    pub fn fullscreen(
        shader_hash: u128,
        color_targets: smallvec::SmallVec<[ColorTargetKey; 2]>,
        depth_stencil: Option<DepthStencilKey>,
    ) -> Self {
        Self {
            shader_hash,
            compilation_hash: 0,
            color_targets,
            depth_stencil,
            multisample: MultisampleKey::from(wgpu::MultisampleState::default()),
            topology: wgpu::PrimitiveTopology::TriangleList,
        }
    }

    #[must_use]
    pub fn with_compilation_options(
        mut self,
        options: &wgpu::PipelineCompilationOptions<'_>,
    ) -> Self {
        self.compilation_hash = hash_pipeline_compilation_options(options);
        self
    }

    /// Specialize the primitive topology for a no-vertex-buffer utility pass.
    #[must_use]
    pub const fn with_topology(mut self, topology: wgpu::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }
}

/// L2 cache key for compute pipelines.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputePipelineKey {
    /// xxh3-128 hash of the final WGSL source code.
    pub shader_hash: u128,
    /// Hash of pipeline compilation options such as overridable constants.
    pub compilation_hash: u64,
    /// Static WGSL compute entry point selected by this pipeline.
    pub entry_point: &'static str,
}

impl ComputePipelineKey {
    #[must_use]
    pub fn new(shader_hash: u128) -> Self {
        Self {
            shader_hash,
            compilation_hash: 0,
            entry_point: "main",
        }
    }

    #[must_use]
    pub fn with_compilation_options(
        mut self,
        options: &wgpu::PipelineCompilationOptions<'_>,
    ) -> Self {
        self.compilation_hash = hash_pipeline_compilation_options(options);
        self
    }

    /// Select a static WGSL compute entry point.
    #[must_use]
    pub const fn with_entry_point(mut self, entry_point: &'static str) -> Self {
        self.entry_point = entry_point;
        self
    }
}

// ─── Simple Geometry Pipeline Key ─────────────────────────────────────────────

/// L2 cache key for simplified geometry passes (Depth Prepass, Shadow Pass).
///
/// These passes render actual meshes with vertex input but skip complex PBR/Phong
/// material state. The `vertex_layout_id` is critical for correct deduplication —
/// two meshes with different vertex buffer layouts (e.g. static vs skinned) must
/// produce distinct pipeline entries even if their shader defines are identical.
///
/// - **Prepass**: `color_targets` = normal/feature-id; `depth_stencil` = main camera depth.
/// - **Shadow**: `color_targets` = empty (depth-only); `depth_stencil` = shadow map depth.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimpleGeometryPipelineKey {
    /// xxh3-128 hash of the final WGSL source code.
    pub shader_hash: u128,
    /// Distinguishes different vertex buffer layouts (static, skinned, morphed…).
    pub vertex_layout_id: u64,
    pub color_targets: smallvec::SmallVec<[ColorTargetKey; 3]>,
    pub depth_stencil: DepthStencilKey,
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub front_face: wgpu::FrontFace,

    pub sample_count: u32,
}

// ─── Convenience helpers ──────────────────────────────────────────────────────

/// Compute a `u64` hash of any `Hash`-able value using `FxBuildHasher`.
#[inline]
pub fn fx_hash_key<K: Hash>(key: &K) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Compute a stable hash for `wgpu` pipeline compilation options.
#[inline]
#[must_use]
pub fn hash_pipeline_compilation_options(options: &wgpu::PipelineCompilationOptions<'_>) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    options.zero_initialize_workgroup_memory.hash(&mut hasher);
    for (name, value) in options.constants {
        name.hash(&mut hasher);
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::{ComputePipelineKey, FullscreenPipelineKey};

    #[test]
    fn compute_key_distinguishes_entry_points() {
        let default = ComputePipelineKey::new(7);

        assert_eq!(default.entry_point, "main");
        assert_ne!(
            default.clone().with_entry_point("calculate_histogram"),
            default.with_entry_point("prefix_histogram"),
        );
    }

    #[test]
    fn fullscreen_key_distinguishes_override_constants() {
        let disabled_constants = [("USE_KARIS_AVERAGE", 0.0)];
        let enabled_constants = [("USE_KARIS_AVERAGE", 1.0)];
        let disabled_options = wgpu::PipelineCompilationOptions {
            constants: &disabled_constants,
            ..Default::default()
        };
        let enabled_options = wgpu::PipelineCompilationOptions {
            constants: &enabled_constants,
            ..Default::default()
        };
        let base = FullscreenPipelineKey::fullscreen(7, smallvec::smallvec![], None);

        assert_ne!(
            base.clone().with_compilation_options(&disabled_options),
            base.with_compilation_options(&enabled_options),
        );
    }

    #[test]
    fn fullscreen_key_distinguishes_primitive_topology() {
        let base = FullscreenPipelineKey::fullscreen(7, smallvec::smallvec![], None);

        assert_eq!(base.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_ne!(
            base.clone()
                .with_topology(wgpu::PrimitiveTopology::TriangleList),
            base.with_topology(wgpu::PrimitiveTopology::TriangleStrip),
        );
    }
}
