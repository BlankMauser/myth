//! Persistent CPU authority for the GPU Scene table set.
//!
//! The store owns stable generational Instance/Geometry/Material/Batch
//! identities, canonical POD records, and replayable dirty journals. It
//! performs no WGPU work. After mutation-free capability preflight and an
//! executable multi-bin selection, RDG imports one coherent
//! [`GpuSceneTableSetPlan`] whose four tables are published atomically by the
//! batched `GpuScene_Patch` node.

use std::{hash::Hash, mem::size_of};

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use myth_assets::{GeometryHandle, MaterialHandle};
use myth_resources::BoundingBox;
use myth_scene::NodeHandle;
use rustc_hash::{FxHashMap, FxHashSet};

use super::capacity::{
    CapacityUsage, estimated_hash_map_footprint, estimated_hash_set_footprint,
    shrink_hash_map_by_family_ratio, shrink_hash_set_by_family_ratio, shrink_vec_by_family_ratio,
    vec_footprint,
};
#[cfg(test)]
use super::core::PersistentBufferPlanFailure;
use super::core::{
    BufferDesc, PersistentBufferBatchPlanFailure, PersistentBufferId, PersistentBufferPlan,
    PersistentBufferPlanRequest, PersistentBufferRef, PersistentBufferRegistry,
    PersistentBufferSpec,
};
use super::extracted::{
    ExtractedRenderItem, ExtractedScene, ItemShaderFeatures, MaterialClassification,
};
use super::frame::{ExtractedSourceId, FrameSourceId, RenderLists};
use super::gpu_driven_policy::GpuDrivenFamilyMask;

const GPU_SCENE_INSTANCE_SCHEMA_ID: u64 = u64::from_be_bytes(*b"MYTHINST");
const GPU_SCENE_GEOMETRY_SCHEMA_ID: u64 = u64::from_be_bytes(*b"MYTHGEOM");
const GPU_SCENE_MATERIAL_SCHEMA_ID: u64 = u64::from_be_bytes(*b"MYTHMATL");
const GPU_SCENE_BATCH_SCHEMA_ID: u64 = u64::from_be_bytes(*b"MYTHBTCH");
const GPU_SCENE_INSTANCE_BUFFER_USAGE: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);
/// Real idle time after which an inactive scene domain stops occupying the
/// shared instance table. This is deliberately time-based: a 30 Hz editor and
/// a 4000 Hz benchmark should retain a dormant scene for the same duration.
const GPU_SCENE_DOMAIN_IDLE_TTL_SECONDS: f64 = 1.5;
/// Bound the number of scene domains whose stable slots participate in one
/// shared table scan. The current scene consumes one entry; the remaining
/// three entries form a small deterministic LRU of dormant domains.
const GPU_SCENE_DOMAIN_BUDGET: usize = 4;
const GPU_SCENE_SCRATCH_BASELINE: usize = 16;
/// Instance participates in shadow-caster culling.
///
/// This lives in `GpuSceneInstanceRecord::metadata.w`. Keeping semantic
/// membership beside the stable transform record lets one packed multi-view
/// Cull scan the persistent table without rebuilding per-view CPU candidate
/// buffers.
pub(crate) const GPU_SCENE_INSTANCE_CASTS_SHADOW: u32 = 1 << 0;
/// Instance belongs to the GPU-owned Main + Prepass cohort.
pub(crate) const GPU_SCENE_INSTANCE_MAIN_PREPASS: u32 = 1 << 1;
/// Instance belongs to the GPU-owned directional/spot Shadow2D cohort.
pub(crate) const GPU_SCENE_INSTANCE_SHADOW_2D: u32 = 1 << 2;
/// Instance belongs to the GPU-owned point-light ShadowCube cohort.
pub(crate) const GPU_SCENE_INSTANCE_SHADOW_CUBE: u32 = 1 << 3;

#[inline]
fn gpu_scene_instance_family_flags(families: GpuDrivenFamilyMask) -> u32 {
    let mut flags = 0;
    if families.contains(GpuDrivenFamilyMask::MAIN_PREPASS) {
        flags |= GPU_SCENE_INSTANCE_MAIN_PREPASS;
    }
    if families.contains(GpuDrivenFamilyMask::SHADOW_2D) {
        flags |= GPU_SCENE_INSTANCE_SHADOW_2D;
    }
    if families.contains(GpuDrivenFamilyMask::SHADOW_CUBE) {
        flags |= GPU_SCENE_INSTANCE_SHADOW_CUBE;
    }
    flags
}
const EMPTY_GPU_SCENE_INSTANCE_RECORD: GpuSceneInstanceRecord = GpuSceneInstanceRecord {
    world_matrix: [[0.0; 4]; 4],
    world_matrix_inverse: [[0.0; 4]; 4],
    previous_world_matrix: [[0.0; 4]; 4],
    bounds_min: [0.0; 4],
    bounds_max: [0.0; 4],
    batch_id: [0; 4],
    metadata: [0; 4],
};

/// Stable cross-frame identity of one scene instance.
///
/// `slot` addresses the persistent table while `generation` prevents a stale
/// candidate or history record from aliasing a later occupant of that slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct GpuSceneInstanceId {
    slot: u32,
    generation: u32,
}

impl GpuSceneInstanceId {
    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn slot(self) -> u32 {
        self.slot
    }

    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn generation(self) -> u32 {
        self.generation
    }
}

macro_rules! define_gpu_scene_id {
    ($name:ident) => {
        #[repr(C)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Pod, Zeroable)]
        pub(crate) struct $name {
            slot: u32,
            generation: u32,
        }

        impl $name {
            #[inline]
            #[must_use]
            pub(crate) const fn slot(self) -> u32 {
                self.slot
            }

            #[inline]
            #[must_use]
            pub(crate) const fn generation(self) -> u32 {
                self.generation
            }

            #[inline]
            #[allow(
                dead_code,
                reason = "the shared ID macro also serves legacy batch identity tests"
            )]
            const fn words(self) -> [u32; 2] {
                [self.slot, self.generation]
            }
        }
    };
}

define_gpu_scene_id!(GpuSceneGeometryId);
define_gpu_scene_id!(GpuSceneMaterialId);
define_gpu_scene_id!(GpuSceneBatchId);

#[cfg(test)]
impl GpuSceneBatchId {
    #[inline]
    #[must_use]
    pub(crate) const fn from_parts(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

/// Stable geometry-table ABI reserved by the M7.4 CPU authority.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuSceneGeometryRecord {
    /// `(generation, alive, asset_version, index_count)`.
    pub(crate) metadata: [u32; 4],
    /// `(asset_identity, structure_version, layout_version)`.
    pub(crate) epochs: [u64; 3],
    /// `(primitive_topology, index_format)`.
    pub(crate) fixed_state: [u32; 2],
}

/// Stable material-table ABI reserved by the M7.4 CPU authority.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuSceneMaterialRecord {
    /// `(generation, alive, asset_version, material_index)`.
    pub(crate) metadata: [u32; 4],
    /// `(asset_identity, material_epoch)`.
    pub(crate) epochs: [u64; 2],
}

/// Stable batch-table ABI. Byte partitions are the CPU binding authority;
/// `draw_state[0..2]` mirrors the same range in visible-record coordinates so
/// WGSL never needs 64-bit division.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuSceneBatchRecord {
    /// `(generation, alive, scene_id, member_count)`.
    pub(crate) metadata: [u32; 4],
    pub(crate) geometry_id: [u32; 2],
    pub(crate) material_id: [u32; 2],
    /// `(binding_base_bytes, binding_size_bytes)`.
    pub(crate) partition_bytes: [u64; 2],
    /// `(base_visible_index, capacity_records, index_count,
    /// item_variant_flags, item_shader_features, topology, index_format, 0)`.
    pub(crate) draw_state: [u32; 8],
}

const EMPTY_GPU_SCENE_GEOMETRY_RECORD: GpuSceneGeometryRecord = GpuSceneGeometryRecord {
    metadata: [0; 4],
    epochs: [0; 3],
    fixed_state: [0; 2],
};
const EMPTY_GPU_SCENE_MATERIAL_RECORD: GpuSceneMaterialRecord = GpuSceneMaterialRecord {
    metadata: [0; 4],
    epochs: [0; 2],
};
const EMPTY_GPU_SCENE_BATCH_RECORD: GpuSceneBatchRecord = GpuSceneBatchRecord {
    metadata: [0; 4],
    geometry_id: [0; 2],
    material_id: [0; 2],
    partition_bytes: [0; 2],
    draw_state: [0; 8],
};

const _: () = assert!(size_of::<GpuSceneGeometryRecord>() == 48);
const _: () = assert!(size_of::<GpuSceneMaterialRecord>() == 32);
const _: () = assert!(size_of::<GpuSceneBatchRecord>() == 80);

impl GpuSceneGeometryRecord {
    fn live(key: GpuSceneRigidBatchKey, generation: u32) -> Self {
        Self {
            metadata: [generation, 1, key.geometry_asset_version, key.index_count],
            epochs: [
                key.geometry_asset_identity,
                key.geometry_structure_version,
                key.geometry_layout_version,
            ],
            fixed_state: [key.geometry_topology as u32, key.index_format as u32],
        }
    }
}

impl GpuSceneMaterialRecord {
    fn live(key: GpuSceneRigidBatchKey, generation: u32) -> Self {
        Self {
            metadata: [
                generation,
                1,
                key.material_asset_version,
                key.material_index,
            ],
            epochs: [key.material_asset_identity, key.material_version],
        }
    }
}

impl GpuSceneBatchRecord {
    fn live(
        scene_id: u32,
        bin: GpuScenePreflightBin,
        generation: u32,
        geometry_id: GpuSceneGeometryId,
        material_id: GpuSceneMaterialId,
    ) -> Self {
        Self {
            metadata: [generation, 1, scene_id, bin.member_count],
            geometry_id: geometry_id.words(),
            material_id: material_id.words(),
            partition_bytes: [bin.partition_base, bin.partition_size],
            draw_state: [
                bin.base_visible_index,
                bin.capacity_records,
                bin.key.index_count,
                bin.key.item_variant_flags,
                u32::from(bin.key.item_shader_features.bits()),
                bin.key.geometry_topology as u32,
                bin.key.index_format as u32,
                0,
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GpuSceneInstanceKey {
    scene_id: u32,
    node: NodeHandle,
}

/// Canonical GPU ABI for one rigid instance.
///
/// Matrices are column-major, matching WGSL. Bounds are kept as a world-space
/// AABB so the first cull shader can conservatively test every transformed
/// object without reopening Geometry assets. `metadata = (generation, alive,
/// scene_id, batch_flags)` makes slot reuse and retirement explicit while
/// preserving the owning scene across multi-scene domains. `batch_id` is a
/// stable `(slot, generation)` pair plus two reserved words. It occupies the
/// former tint lane, so the record remains exactly 256 bytes.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuSceneInstanceRecord {
    pub(crate) world_matrix: [[f32; 4]; 4],
    pub(crate) world_matrix_inverse: [[f32; 4]; 4],
    pub(crate) previous_world_matrix: [[f32; 4]; 4],
    pub(crate) bounds_min: [f32; 4],
    pub(crate) bounds_max: [f32; 4],
    pub(crate) batch_id: [u32; 4],
    pub(crate) metadata: [u32; 4],
}

impl GpuSceneInstanceRecord {
    #[cfg(test)]
    fn live(input: GpuSceneInstanceInput, generation: u32, scene_id: u32) -> Self {
        let inverse = input.world_matrix.inverse().to_cols_array_2d();
        Self::live_with_batch(input, inverse, generation, scene_id, None)
    }

    fn live_with_batch(
        input: GpuSceneInstanceInput,
        world_matrix_inverse: [[f32; 4]; 4],
        generation: u32,
        scene_id: u32,
        batch_id: Option<GpuSceneBatchId>,
    ) -> Self {
        Self {
            world_matrix: input.world_matrix.to_cols_array_2d(),
            world_matrix_inverse,
            previous_world_matrix: input.previous_world_matrix.to_cols_array_2d(),
            bounds_min: [
                input.world_bounds.min.x,
                input.world_bounds.min.y,
                input.world_bounds.min.z,
                0.0,
            ],
            bounds_max: [
                input.world_bounds.max.x,
                input.world_bounds.max.y,
                input.world_bounds.max.z,
                0.0,
            ],
            batch_id: batch_id.map_or([u32::MAX, 0, 0, 0], |id| [id.slot(), id.generation(), 0, 0]),
            metadata: [generation, 1, scene_id, input.flags],
        }
    }

    fn tombstone(generation: u32) -> Self {
        Self {
            metadata: [generation, 0, 0, 0],
            ..Self::zeroed()
        }
    }

    #[inline]
    fn bitwise_eq(&self, other: &Self) -> bool {
        bytemuck::bytes_of(self) == bytemuck::bytes_of(other)
    }
}

const _: () = assert!(size_of::<GpuSceneInstanceRecord>() == 256);

/// Copy-only extraction input accepted by the persistent instance authority.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuSceneInstanceInput {
    #[cfg(test)]
    pub(crate) node: NodeHandle,
    pub(crate) world_matrix: Mat4,
    pub(crate) previous_world_matrix: Mat4,
    pub(crate) world_bounds: BoundingBox,
    pub(crate) flags: u32,
}

/// Complete fixed-state identity of one GPU-driven opaque batch. Every member
/// of a bin shares this key, while one pre-frustum opaque domain may contain
/// any number of distinct keys in deterministic first-occurrence order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GpuSceneRigidBatchKey {
    pub(crate) geometry: GeometryHandle,
    pub(crate) geometry_asset_version: u32,
    pub(crate) geometry_asset_identity: u64,
    pub(crate) geometry_structure_version: u64,
    pub(crate) geometry_layout_version: u64,
    pub(crate) geometry_topology: wgpu::PrimitiveTopology,
    pub(crate) index_format: wgpu::IndexFormat,
    pub(crate) index_count: u32,
    pub(crate) material: MaterialHandle,
    pub(crate) material_asset_version: u32,
    pub(crate) material_asset_identity: u64,
    pub(crate) material_version: u64,
    pub(crate) material_index: u32,
    pub(crate) item_variant_flags: u32,
    pub(crate) item_shader_features: ItemShaderFeatures,
}

impl GpuSceneRigidBatchKey {
    pub(crate) fn from_item(item: &ExtractedRenderItem) -> Option<Self> {
        Some(Self {
            geometry: item.geometry,
            geometry_asset_version: item.geometry_asset_version,
            geometry_asset_identity: item.geometry_asset_identity,
            geometry_structure_version: item.geometry_structure_version,
            geometry_layout_version: item.geometry_layout_version,
            geometry_topology: item.geometry_topology,
            index_format: item.geometry_index_format?,
            index_count: item.geometry_index_count,
            material: item.material,
            material_asset_version: item.material_asset_version,
            material_asset_identity: item.material_asset_identity,
            material_version: item.material_version,
            material_index: item.material_index,
            item_variant_flags: item.item_variant_flags,
            item_shader_features: item.item_shader_features,
        })
    }

    #[inline]
    #[must_use]
    pub(crate) fn matches_item(self, item: &ExtractedRenderItem) -> bool {
        Self::from_item(item) == Some(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSceneMultiBinProjection {
    pub(crate) candidate_count: u32,
    pub(crate) bin_count: u32,
    pub(crate) unique_geometry_count: u32,
    pub(crate) unique_material_count: u32,
    /// Selected candidates which may contribute to a shadow family. This is
    /// distinct from the complete scene caster count when Auto retains only
    /// high-reuse cohorts and leaves the rest on the CPU backend.
    pub(crate) shadow_candidate_count: u32,
    /// True when every shadow caster belongs to the same complete opaque
    /// domain represented by this recipe. This is accumulated during the
    /// mandatory preflight scan so Shadow ownership never rescans the scene.
    pub(crate) covers_complete_shadow_domain: bool,
    pub(crate) visible_stride: u64,
    pub(crate) storage_alignment: u64,
    pub(crate) visible_partition_bytes: u64,
    /// Mutation-free capability facts for the exact retained recipe.
    pub(crate) table_set: GpuSceneProjectedTableSet,
}

/// Exact logical capacities needed if the associated preflight token is
/// committed. Computing this value never mutates a stable table, its dirty
/// journal, or a frame-local render list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSceneProjectedTableSet {
    pub(crate) instance_slot_span: u32,
    pub(crate) geometry_slot_span: u32,
    pub(crate) material_slot_span: u32,
    pub(crate) batch_slot_span: u32,
    pub(crate) instance_logical_bytes: u64,
    pub(crate) geometry_logical_bytes: u64,
    pub(crate) material_logical_bytes: u64,
    pub(crate) batch_logical_bytes: u64,
    pub(crate) visible_view_stride: u64,
    pub(crate) visible_partition_bytes: u64,
    pub(crate) indirect_args_span: u64,
}

/// Copy-only capability produced by a retained preflight scratch image. A
/// later preflight invalidates it; no persistent table or RenderLists state is
/// mutated until the exact token is committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSceneMultiBinPreflight {
    serial: u64,
    scene_id: u32,
    /// Frame-local identity proof. Commit compares only the address and never
    /// dereferences it; this prevents an internal caller from pairing retained
    /// recipe indices with a different `ExtractedScene` that happens to reuse
    /// the same semantic scene id.
    scene_identity: usize,
    pub(crate) projection: GpuSceneMultiBinProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSceneMultiBinCandidate {
    pub(crate) batch_id: GpuSceneBatchId,
    pub(crate) key: GpuSceneRigidBatchKey,
    pub(crate) representative: FrameSourceId,
    pub(crate) member_count: u32,
    pub(crate) partition_base: u64,
    pub(crate) partition_size: u64,
    pub(crate) base_visible_index: u32,
    pub(crate) capacity_records: u32,
}

/// Canonical, compact Cull input retained in the same first-occurrence order
/// as [`GpuSceneMultiBinCandidate`]. Consumers do not need to rebuild a
/// temporary bin list or depend on Pass-layer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSceneActiveBatchDescriptor {
    pub(crate) id: GpuSceneBatchId,
    pub(crate) partition_base: u64,
    pub(crate) partition_size: u64,
}

/// Authority-certified compact batch set consumed by GPU Scene Cull.
///
/// The private production constructor is the proof boundary: only the CPU
/// authority can certify that stable batch slots are unique and visible
/// partitions do not overlap. Cull can therefore retain linear ABI/range
/// validation without repeating an `O(batch_count^2)` pairwise scan for every
/// view family.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuSceneActiveBatchSet<'a> {
    batches: &'a [GpuSceneActiveBatchDescriptor],
    #[cfg(test)]
    authority_certified: bool,
}

impl<'a> GpuSceneActiveBatchSet<'a> {
    #[inline]
    #[must_use]
    fn from_authority(batches: &'a [GpuSceneActiveBatchDescriptor]) -> Self {
        Self {
            batches,
            #[cfg(test)]
            authority_certified: true,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn as_slice(self) -> &'a [GpuSceneActiveBatchDescriptor] {
        self.batches
    }

    /// Construct an intentionally untrusted set for adversarial Cull tests.
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(crate) const fn untrusted(batches: &'a [GpuSceneActiveBatchDescriptor]) -> Self {
        Self {
            batches,
            authority_certified: false,
        }
    }

    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(crate) const fn is_authority_certified(self) -> bool {
        self.authority_certified
    }
}

impl std::ops::Deref for GpuSceneActiveBatchSet<'_> {
    type Target = [GpuSceneActiveBatchDescriptor];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.batches
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuSceneMultiBinPreflightFailure {
    Domain(GpuSceneSelectionRejection),
    InvalidVisibleStride,
    InvalidStorageAlignment,
    LayoutOverflow,
}

/// Mutation-free selection contract used while constructing a retained
/// multi-bin recipe. `CompleteDomain` preserves the strict `Required` oracle:
/// one unsupported opaque item rejects the entire GPU backend. `Cohorts`
/// keeps unsupported and low-reuse bins CPU-owned while retaining only bins
/// large enough to amortize an indirect submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuSceneMultiBinSelectionPolicy {
    CompleteDomain,
    Cohorts { minimum_member_count: u32 },
}

impl GpuSceneMultiBinSelectionPolicy {
    #[inline]
    const fn rejects_unsupported(self) -> bool {
        matches!(self, Self::CompleteDomain)
    }

    #[inline]
    const fn minimum_member_count(self) -> u32 {
        match self {
            Self::CompleteDomain => 1,
            Self::Cohorts {
                minimum_member_count,
            } => {
                if minimum_member_count == 0 {
                    1
                } else {
                    minimum_member_count
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuSceneMultiBinCommitFailure {
    StalePreflight,
    SceneDrift,
}

#[derive(Debug, Clone, Copy)]
struct GpuScenePreflightBin {
    key: GpuSceneRigidBatchKey,
    representative: ExtractedSourceId,
    member_count: u32,
    partition_base: u64,
    partition_size: u64,
    base_visible_index: u32,
    capacity_records: u32,
}

#[derive(Debug, Clone, Copy)]
struct GpuScenePreflightMember {
    item: u32,
    bin: u32,
    /// Index into the sparse inverse-update scratch, or `u32::MAX` when the
    /// retained instance record already owns the exact current inverse.
    inverse_update: u32,
}

#[derive(Debug, Clone, Copy)]
struct GpuSceneItemPreflight {
    key: GpuSceneRigidBatchKey,
    inverse_update: Option<[[f32; 4]; 4]>,
}

#[derive(Debug, Clone, Copy)]
#[allow(
    clippy::struct_field_names,
    reason = "the ID suffix distinguishes stable table identities from source stamps and records"
)]
struct GpuSceneResolvedBin {
    geometry_id: GpuSceneGeometryId,
    material_id: GpuSceneMaterialId,
    batch_id: GpuSceneBatchId,
}

/// Stable semantic identity of a batch. Asset epochs and physical partition
/// coordinates deliberately do not participate: both update the record in
/// place, while dependent Geometry/Material generation drift bumps the Batch
/// generation at the same address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GpuSceneBatchKey {
    scene_id: u32,
    geometry: GeometryHandle,
    material: MaterialHandle,
    item_variant_flags: u32,
    item_shader_features: ItemShaderFeatures,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GpuSceneGeometryStamp {
    asset_version: u32,
    asset_identity: u64,
    structure_version: u64,
    layout_version: u64,
    topology: wgpu::PrimitiveTopology,
    index_format: wgpu::IndexFormat,
    index_count: u32,
}

impl GpuSceneGeometryStamp {
    fn from_key(key: GpuSceneRigidBatchKey) -> Self {
        Self {
            asset_version: key.geometry_asset_version,
            asset_identity: key.geometry_asset_identity,
            structure_version: key.geometry_structure_version,
            layout_version: key.geometry_layout_version,
            topology: key.geometry_topology,
            index_format: key.index_format,
            index_count: key.index_count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GpuSceneMaterialStamp {
    asset_version: u32,
    asset_identity: u64,
    material_version: u64,
    material_index: u32,
}

impl GpuSceneMaterialStamp {
    fn from_key(key: GpuSceneRigidBatchKey) -> Self {
        Self {
            asset_version: key.material_asset_version,
            asset_identity: key.material_asset_identity,
            material_version: key.material_version,
            material_index: key.material_index,
        }
    }
}

impl GpuSceneBatchKey {
    #[inline]
    fn new(scene_id: u32, key: GpuSceneRigidBatchKey) -> Self {
        Self {
            scene_id,
            geometry: key.geometry,
            material: key.material,
            item_variant_flags: key.item_variant_flags,
            item_shader_features: key.item_shader_features,
        }
    }
}

/// Typed reasons why the first constrained GPU backend cannot represent the
/// complete opaque domain. `Auto` falls back before emitting GPU topology;
/// `Required` turns the same value into a terminal frame outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuSceneSelectionRejection {
    Skinned,
    MorphTargets,
    InstancedVertexAttributes,
    IncompatibleMaterial,
    NonIndexed,
    EmptyIndexBuffer,
    EmptyVertexBuffers,
    UnsupportedTopology,
    NonFiniteTransform,
    NonInvertibleTransform,
    NonFiniteBounds,
    TooManyCandidates,
    DuplicateInstance,
    ConflictingGeometryEpoch,
    ConflictingMaterialEpoch,
    /// One retained live identity reached the last representable generation.
    /// The production backend fails over before a later transaction could
    /// relocate it and invalidate the mutation-free slot-span projection.
    GenerationExhausted,
}

#[inline]
fn is_gpu_scene_prefrustum_opaque(item: &ExtractedRenderItem) -> bool {
    !item
        .material_classification
        .intersects(MaterialClassification::TRANSPARENT | MaterialClassification::USES_TRANSMISSION)
}

fn preflight_gpu_scene_item(
    item: &ExtractedRenderItem,
    world_matrix_is_current: bool,
) -> Result<Option<GpuSceneItemPreflight>, GpuSceneSelectionRejection> {
    if !is_gpu_scene_prefrustum_opaque(item) {
        return Ok(None);
    }
    if item.skeleton.is_some()
        || item
            .item_shader_features
            .contains(ItemShaderFeatures::HAS_SKINNING)
    {
        return Err(GpuSceneSelectionRejection::Skinned);
    }
    if item.geometry_has_morph_targets {
        return Err(GpuSceneSelectionRejection::MorphTargets);
    }
    if item.geometry_has_instanced_attributes {
        return Err(GpuSceneSelectionRejection::InstancedVertexAttributes);
    }
    if !item.material_gpu_scene_compatible {
        return Err(GpuSceneSelectionRejection::IncompatibleMaterial);
    }
    if item.geometry_index_format.is_none() {
        return Err(GpuSceneSelectionRejection::NonIndexed);
    }
    if item.geometry_index_count == 0 {
        return Err(GpuSceneSelectionRejection::EmptyIndexBuffer);
    }
    if !item.geometry_has_vertex_attributes {
        return Err(GpuSceneSelectionRejection::EmptyVertexBuffers);
    }
    if item.geometry_topology != wgpu::PrimitiveTopology::TriangleList {
        return Err(GpuSceneSelectionRejection::UnsupportedTopology);
    }
    if !item.world_matrix.is_finite() || !item.prev_world_matrix.is_finite() {
        return Err(GpuSceneSelectionRejection::NonFiniteTransform);
    }
    let inverse_update = if world_matrix_is_current {
        None
    } else {
        let determinant = item.world_matrix.determinant();
        if !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
            return Err(GpuSceneSelectionRejection::NonInvertibleTransform);
        }
        let inverse = item.world_matrix.inverse();
        if !inverse.is_finite() {
            return Err(GpuSceneSelectionRejection::NonInvertibleTransform);
        }
        Some(inverse.to_cols_array_2d())
    };
    if !item.world_aabb.is_finite() || !item.world_aabb.min.cmple(item.world_aabb.max).all() {
        return Err(GpuSceneSelectionRejection::NonFiniteBounds);
    }
    Ok(Some(GpuSceneItemPreflight {
        key: GpuSceneRigidBatchKey::from_item(item)
            .expect("indexed GPU Scene candidate lost its index format"),
        inverse_update,
    }))
}

fn checked_gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn checked_lcm(left: u64, right: u64) -> Option<u64> {
    let gcd = checked_gcd(left, right);
    (left / gcd).checked_mul(right)
}

fn checked_align_up(value: u64, alignment: u64) -> Option<u64> {
    debug_assert_ne!(alignment, 0);
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

impl GpuSceneInstanceInput {
    #[inline]
    #[must_use]
    pub(crate) const fn new(
        node: NodeHandle,
        world_matrix: Mat4,
        previous_world_matrix: Mat4,
        world_bounds: BoundingBox,
    ) -> Self {
        #[cfg(not(test))]
        let _ = node;
        Self {
            #[cfg(test)]
            node,
            world_matrix,
            previous_world_matrix,
            world_bounds,
            flags: 0,
        }
    }

    /// Attach stable instance-membership flags without widening the canonical
    /// persistent record ABI.
    #[inline]
    #[must_use]
    pub(crate) const fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }
}

#[derive(Debug)]
struct InstanceSlot {
    generation: u32,
    key: Option<GpuSceneInstanceKey>,
    geometry_id: Option<GpuSceneGeometryId>,
    material_id: Option<GpuSceneMaterialId>,
    batch_id: Option<GpuSceneBatchId>,
    seen_sync: u64,
    dirty: bool,
    reusable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AuthorityId {
    slot: u32,
    generation: u32,
}

#[derive(Debug)]
struct AuthoritySlot<K> {
    generation: u32,
    key: Option<K>,
    seen_sync: u64,
    ref_count: u32,
    dirty: bool,
    reusable: bool,
}

trait AuthorityRecord: Copy + Pod {
    fn tombstone(generation: u32) -> Self;

    #[inline]
    fn bitwise_eq(&self, other: &Self) -> bool {
        bytemuck::bytes_of(self) == bytemuck::bytes_of(other)
    }
}

impl AuthorityRecord for GpuSceneGeometryRecord {
    fn tombstone(generation: u32) -> Self {
        Self {
            metadata: [generation, 0, 0, 0],
            ..Self::zeroed()
        }
    }
}

impl AuthorityRecord for GpuSceneMaterialRecord {
    fn tombstone(generation: u32) -> Self {
        Self {
            metadata: [generation, 0, 0, 0],
            ..Self::zeroed()
        }
    }
}

impl AuthorityRecord for GpuSceneBatchRecord {
    fn tombstone(generation: u32) -> Self {
        Self {
            metadata: [generation, 0, 0, 0],
            ..Self::zeroed()
        }
    }
}

#[derive(Debug)]
struct AuthorityTable<K, R> {
    slots: Vec<AuthoritySlot<K>>,
    records: Vec<R>,
    generation_watermarks: Vec<u32>,
    lookup: FxHashMap<K, AuthorityId>,
    reusable_slots: Vec<u32>,
    quarantined_slots: Vec<QuarantinedSlot>,
    dirty_slots: Vec<u32>,
    dirty_ranges: Vec<GpuSceneDirtyRange>,
}

impl<K, R> Default for AuthorityTable<K, R> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            records: Vec::new(),
            generation_watermarks: Vec::new(),
            lookup: FxHashMap::default(),
            reusable_slots: Vec::new(),
            quarantined_slots: Vec::new(),
            dirty_slots: Vec::new(),
            dirty_ranges: Vec::new(),
        }
    }
}

impl<K, R> AuthorityTable<K, R>
where
    K: Copy + Eq + Hash + std::fmt::Debug,
    R: AuthorityRecord,
{
    #[inline]
    fn id(&self, key: K) -> Option<AuthorityId> {
        self.lookup.get(&key).copied()
    }

    #[inline]
    fn slot_span(&self) -> u32 {
        u32::try_from(self.records.len()).expect("GPU Scene authority table exceeds u32 slots")
    }

    fn projected_slot_span<I>(&self, keys: I) -> Option<u32>
    where
        I: IntoIterator<Item = K>,
    {
        let new_keys = keys
            .into_iter()
            .filter(|key| !self.lookup.contains_key(key))
            .count();
        let retained = self.acknowledged_tail_retained_len();
        let interior_reusable = self.slots[..retained]
            .iter()
            .filter(|slot| slot.reusable)
            .count();
        retained
            .checked_add(new_keys.saturating_sub(interior_reusable))
            .and_then(|span| u32::try_from(span).ok())
    }

    fn allocate_with(&mut self, key: K, make_record: impl FnOnce(u32) -> R) -> AuthorityId {
        let slot_id = if let Some(slot_id) = self.reusable_slots.pop() {
            let slot = &self.slots[slot_id as usize];
            assert!(slot.reusable && slot.key.is_none() && !slot.dirty);
            slot_id
        } else {
            let slot_id = u32::try_from(self.slots.len())
                .expect("GPU Scene cannot allocate more than u32::MAX authority slots");
            assert_ne!(slot_id, u32::MAX);
            let index = slot_id as usize;
            let generation = if let Some(&generation) = self.generation_watermarks.get(index) {
                assert_ne!(generation, u32::MAX, "permanently retired slot regrew");
                generation
            } else {
                assert_eq!(index, self.generation_watermarks.len());
                self.generation_watermarks.push(1);
                1
            };
            self.slots.push(AuthoritySlot {
                generation,
                key: None,
                seen_sync: 0,
                ref_count: 0,
                dirty: false,
                reusable: false,
            });
            self.records.push(R::tombstone(generation));
            slot_id
        };

        let slot = &mut self.slots[slot_id as usize];
        assert!(slot.key.is_none() && !slot.dirty);
        slot.reusable = false;
        slot.key = Some(key);
        slot.ref_count = 0;
        let id = AuthorityId {
            slot: slot_id,
            generation: slot.generation,
        };
        self.records[slot_id as usize] = make_record(id.generation);
        assert!(self.lookup.insert(key, id).is_none());
        self.mark_dirty(slot_id);
        id
    }

    fn bump_generation_with(
        &mut self,
        id: AuthorityId,
        retired_revision: u64,
        make_record: impl FnOnce(u32) -> R,
    ) -> AuthorityId {
        let index = id.slot as usize;
        assert_eq!(self.slots[index].generation, id.generation);
        let key = self.slots[index]
            .key
            .expect("cannot bump an empty authority slot");
        if id.generation == u32::MAX {
            let ref_count = self.slots[index].ref_count;
            self.slots[index].key = None;
            self.slots[index].ref_count = 0;
            let removed = self.lookup.remove(&key).expect("authority lookup drift");
            assert_eq!(removed, id);
            self.records[index] = R::tombstone(u32::MAX);
            self.mark_dirty(id.slot);
            self.quarantined_slots.push(QuarantinedSlot {
                slot: id.slot,
                retired_revision,
            });
            let replacement = self.allocate_with(key, make_record);
            self.slots[replacement.slot as usize].ref_count = ref_count;
            return replacement;
        }
        let slot = &mut self.slots[index];
        let generation = slot.generation + 1;
        slot.generation = generation;
        self.generation_watermarks[index] = generation;
        let next = AuthorityId {
            slot: id.slot,
            generation,
        };
        *self.lookup.get_mut(&key).expect("authority lookup drift") = next;
        self.records[index] = make_record(generation);
        self.mark_dirty(id.slot);
        next
    }

    fn update_with(&mut self, id: AuthorityId, make_record: impl FnOnce(u32) -> R) -> bool {
        let index = id.slot as usize;
        assert_eq!(self.slots[index].generation, id.generation);
        let next = make_record(id.generation);
        if self.records[index].bitwise_eq(&next) {
            return false;
        }
        self.records[index] = next;
        self.mark_dirty(id.slot);
        true
    }

    fn add_ref(&mut self, id: AuthorityId) {
        let slot = &mut self.slots[id.slot as usize];
        assert_eq!(slot.generation, id.generation);
        assert!(slot.key.is_some());
        slot.ref_count = slot
            .ref_count
            .checked_add(1)
            .expect("GPU Scene authority reference count overflowed");
    }

    /// Returns true only when the final reference retired the slot.
    fn release(&mut self, id: AuthorityId, retired_revision: u64) -> bool {
        let index = id.slot as usize;
        let slot = &mut self.slots[index];
        // A dependency generation may have advanced while the owning Batch
        // retained its old pair. Slot identity, not the stale generation word,
        // owns the reference count.
        assert!(slot.key.is_some());
        assert_ne!(slot.ref_count, 0);
        slot.ref_count -= 1;
        if slot.ref_count != 0 {
            return false;
        }
        let key = slot.key.take().expect("authority slot lost its owner");
        let removed = self.lookup.remove(&key).expect("authority lookup drift");
        assert_eq!(removed.slot, id.slot);
        if let Some(next) = slot.generation.checked_add(1) {
            slot.generation = next;
            self.generation_watermarks[index] = next;
        }
        self.records[index] = R::tombstone(slot.generation);
        self.mark_dirty(id.slot);
        self.quarantined_slots.push(QuarantinedSlot {
            slot: id.slot,
            retired_revision,
        });
        true
    }

    fn mark_seen(&mut self, id: AuthorityId, serial: u64) {
        let slot = &mut self.slots[id.slot as usize];
        assert!(slot.key.is_some());
        slot.seen_sync = serial;
    }

    fn mark_dirty(&mut self, slot_id: u32) {
        let slot = &mut self.slots[slot_id as usize];
        if slot.dirty {
            return;
        }
        slot.dirty = true;
        self.dirty_slots.push(slot_id);
    }

    fn rebuild_dirty_ranges(&mut self) {
        rebuild_dirty_ranges(&mut self.dirty_slots, &mut self.dirty_ranges);
    }

    fn acknowledge(&mut self, current_revision: u64, submitted_revision: u64) -> bool {
        let acknowledged = acknowledge_authority_revision_parts(
            &mut self.slots,
            &mut self.dirty_slots,
            &mut self.quarantined_slots,
            &mut self.reusable_slots,
            current_revision,
            submitted_revision,
        );
        if acknowledged {
            self.dirty_ranges.clear();
        }
        acknowledged
    }

    fn trim_acknowledged_trailing_slots(&mut self) -> bool {
        let retained = self.acknowledged_tail_retained_len();
        if retained == self.slots.len() {
            return false;
        }
        debug_assert!(
            self.slots[retained..]
                .iter()
                .all(|slot| slot.key.is_none() && !slot.dirty && slot.reusable)
        );
        self.reusable_slots
            .retain(|slot| (*slot as usize) < retained);
        self.slots.truncate(retained);
        self.records.truncate(retained);
        self.dirty_ranges.clear();
        true
    }

    #[inline]
    fn acknowledged_tail_retained_len(&self) -> usize {
        self.slots
            .iter()
            .rposition(|slot| !slot.reusable)
            .map_or(0, |index| index + 1)
    }

    fn capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.slots, 0)
            + vec_footprint(&self.records, 0)
            + vec_footprint(&self.generation_watermarks, 0)
            + estimated_hash_map_footprint(&self.lookup, 0)
            + vec_footprint(&self.reusable_slots, 0)
            + vec_footprint(&self.quarantined_slots, 0)
            + vec_footprint(&self.dirty_slots, 0)
            + vec_footprint(&self.dirty_ranges, 0)
    }

    fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        shrink_vec_by_family_ratio(&mut self.slots, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(&mut self.records, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(
            &mut self.generation_watermarks,
            0,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(&mut self.lookup, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(&mut self.reusable_slots, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(&mut self.quarantined_slots, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(&mut self.dirty_slots, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(&mut self.dirty_ranges, 0, target_bytes, retained_bytes);
    }
}

#[derive(Debug, Default)]
struct SceneDomain {
    slots: Vec<u32>,
    batch_slots: Vec<u32>,
    last_used_seconds: f64,
    last_used_sync: u64,
}

#[derive(Debug, Clone, Copy)]
struct DomainSweepCandidate {
    scene_id: u32,
    last_used_seconds: f64,
    last_used_sync: u64,
}

#[derive(Debug, Clone, Copy)]
struct QuarantinedSlot {
    slot: u32,
    retired_revision: u64,
}

/// One contiguous dirty interval in instance-table record coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSceneDirtyRange {
    pub(crate) first_slot: u32,
    pub(crate) slot_count: u32,
}

impl GpuSceneDirtyRange {
    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn byte_offset(self) -> u64 {
        self.first_slot as u64 * size_of::<GpuSceneInstanceRecord>() as u64
    }

    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn byte_size(self) -> u64 {
        self.slot_count as u64 * size_of::<GpuSceneInstanceRecord>() as u64
    }
}

/// Summary of one CPU synchronization transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct GpuSceneSyncStats {
    pub(crate) revision: u64,
    pub(crate) inserted: u32,
    pub(crate) updated: u32,
    pub(crate) retired: u32,
    pub(crate) alive: u32,
}

/// Exact persistent-table target whose successful publication may retire the
/// matching CPU dirty journal.
///
/// This is deliberately a probe rather than publication authority: the owner
/// must still ask the registry whether this complete attempt/epoch/generation/
/// revision identity was accepted after the real queue submission.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuScenePublicationProbe {
    buffer: PersistentBufferRef,
    revision: u64,
}

/// Immutable hand-off from the CPU authority to one RDG graph attempt.
#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) struct GpuSceneTablePlan<'a> {
    pub(crate) persistent: PersistentBufferPlan,
    pub(crate) publication_probe: GpuScenePublicationProbe,
    pub(crate) revision: u64,
    pub(crate) records: &'a [GpuSceneInstanceRecord],
    pub(crate) dirty_ranges: &'a [GpuSceneDirtyRange],
    /// Allocated table span, including tombstone holes but excluding the
    /// bindable empty-table sentinel. Cull dispatch must use this rather than
    /// `alive_count`, because live instances can occupy high sparse slots.
    pub(crate) slot_span: u32,
    pub(crate) alive_count: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct GpuSceneTableMemberPlan<'a, R> {
    pub(crate) persistent: PersistentBufferPlan,
    pub(crate) records: &'a [R],
    pub(crate) dirty_ranges: &'a [GpuSceneDirtyRange],
    pub(crate) slot_span: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSceneTableSetPublicationProbe {
    buffers: [PersistentBufferRef; 4],
    revision: u64,
}

/// One coherent four-table logical image. All persistent plans are either
/// valid together or require publication together at the shared revision.
#[derive(Clone, Copy)]
pub(crate) struct GpuSceneTableSetPlan<'a> {
    #[cfg(test)]
    pub(crate) revision: u64,
    pub(crate) publication_probe: GpuSceneTableSetPublicationProbe,
    pub(crate) instances: GpuSceneTableMemberPlan<'a, GpuSceneInstanceRecord>,
    pub(crate) geometries: GpuSceneTableMemberPlan<'a, GpuSceneGeometryRecord>,
    pub(crate) materials: GpuSceneTableMemberPlan<'a, GpuSceneMaterialRecord>,
    pub(crate) batches: GpuSceneTableMemberPlan<'a, GpuSceneBatchRecord>,
    pub(crate) candidates: &'a [GpuSceneMultiBinCandidate],
    pub(crate) active_batches: GpuSceneActiveBatchSet<'a>,
}

pub(crate) struct GpuSceneTableSetPublicationAcknowledger<'a> {
    persistent_buffers: [Option<PersistentBufferId>; 4],
    revision: u64,
    instance_slots: &'a mut [InstanceSlot],
    instance_dirty_slots: &'a mut Vec<u32>,
    instance_quarantine: &'a mut Vec<QuarantinedSlot>,
    instance_reusable: &'a mut Vec<u32>,
    geometry_slots: &'a mut [AuthoritySlot<GeometryHandle>],
    geometry_dirty_slots: &'a mut Vec<u32>,
    geometry_quarantine: &'a mut Vec<QuarantinedSlot>,
    geometry_reusable: &'a mut Vec<u32>,
    material_slots: &'a mut [AuthoritySlot<MaterialHandle>],
    material_dirty_slots: &'a mut Vec<u32>,
    material_quarantine: &'a mut Vec<QuarantinedSlot>,
    material_reusable: &'a mut Vec<u32>,
    batch_slots: &'a mut [AuthoritySlot<GpuSceneBatchKey>],
    batch_dirty_slots: &'a mut Vec<u32>,
    batch_quarantine: &'a mut Vec<QuarantinedSlot>,
    batch_reusable: &'a mut Vec<u32>,
}

impl GpuSceneTableSetPublicationAcknowledger<'_> {
    pub(crate) fn acknowledge_if_published(
        &mut self,
        registry: &PersistentBufferRegistry,
        probe: &GpuSceneTableSetPublicationProbe,
    ) -> bool {
        if probe.revision != self.revision
            || probe
                .buffers
                .iter()
                .zip(self.persistent_buffers)
                .any(|(buffer, id)| {
                    id != Some(buffer.id)
                        || buffer.revision != probe.revision
                        || !registry.is_published(*buffer)
                })
        {
            return false;
        }
        let instance = acknowledge_submitted_revision_parts(
            self.instance_slots,
            self.instance_dirty_slots,
            self.instance_quarantine,
            self.instance_reusable,
            self.revision,
            probe.revision,
        );
        let geometry = acknowledge_authority_revision_parts(
            self.geometry_slots,
            self.geometry_dirty_slots,
            self.geometry_quarantine,
            self.geometry_reusable,
            self.revision,
            probe.revision,
        );
        let material = acknowledge_authority_revision_parts(
            self.material_slots,
            self.material_dirty_slots,
            self.material_quarantine,
            self.material_reusable,
            self.revision,
            probe.revision,
        );
        let batch = acknowledge_authority_revision_parts(
            self.batch_slots,
            self.batch_dirty_slots,
            self.batch_quarantine,
            self.batch_reusable,
            self.revision,
            probe.revision,
        );
        instance && geometry && material && batch
    }
}

/// Renderer-owned persistent GPU Scene CPU state.
///
/// Slots retired by revision `R` remain quarantined until the persistent
/// registry proves that exact-or-newer table contents were submitted. This
/// avoids recycling an address while an aborted graph still owes the GPU its
/// tombstone. IDs and slots are never derived from frame-local extraction
/// indices.
#[derive(Default)]
pub(crate) struct GpuScene {
    persistent_instance_buffer: Option<PersistentBufferId>,
    persistent_geometry_buffer: Option<PersistentBufferId>,
    persistent_material_buffer: Option<PersistentBufferId>,
    persistent_batch_buffer: Option<PersistentBufferId>,
    slots: Vec<InstanceSlot>,
    records: Vec<GpuSceneInstanceRecord>,
    /// Per-index generation survives logical tail trimming. Retaining four
    /// bytes per historical slot is the minimum state needed to prevent an
    /// old `(slot, generation)` from aliasing a later regrowth of that index.
    slot_generation_watermarks: Vec<u32>,
    lookup: FxHashMap<GpuSceneInstanceKey, GpuSceneInstanceId>,
    domains: FxHashMap<u32, SceneDomain>,
    domain_sweep_scratch: Vec<DomainSweepCandidate>,
    multi_bin_lookup: FxHashMap<GpuSceneRigidBatchKey, u32>,
    multi_bin_geometry_uniques: FxHashMap<GeometryHandle, GpuSceneGeometryStamp>,
    multi_bin_material_uniques: FxHashMap<MaterialHandle, GpuSceneMaterialStamp>,
    multi_bin_instance_uniques: FxHashSet<NodeHandle>,
    /// Complete item-domain fact accumulated by the retained preflight scan.
    /// Composer reads it only after an `Ok` outcome, avoiding a second O(N)
    /// scan while preserving the distinction between shadow lights and actual
    /// shadow-casting render items.
    multi_bin_has_shadow_casters: bool,
    multi_bin_preflight: Vec<GpuScenePreflightBin>,
    multi_bin_members: Vec<GpuScenePreflightMember>,
    /// Old-bin -> retained-bin remap used to compact Auto cohorts in place.
    /// A retained high-water vector avoids an allocation in warm preflight.
    multi_bin_bin_remap: Vec<u32>,
    /// Sparse matrix inverses for new or transform-dirty members. Stable
    /// members reuse the inverse already stored in their persistent record.
    multi_bin_inverse_updates: Vec<[[f32; 4]; 4]>,
    multi_bin_resolved: Vec<GpuSceneResolvedBin>,
    multi_bin_committed: Vec<GpuSceneMultiBinCandidate>,
    active_batch_descriptors: Vec<GpuSceneActiveBatchDescriptor>,
    multi_bin_serial: u64,
    geometries: AuthorityTable<GeometryHandle, GpuSceneGeometryRecord>,
    materials: AuthorityTable<MaterialHandle, GpuSceneMaterialRecord>,
    batches: AuthorityTable<GpuSceneBatchKey, GpuSceneBatchRecord>,
    scene_batch_scratch: Vec<u32>,
    cascade_scratch: Vec<u32>,
    reusable_slots: Vec<u32>,
    quarantined_slots: Vec<QuarantinedSlot>,
    dirty_slots: Vec<u32>,
    dirty_ranges: Vec<GpuSceneDirtyRange>,
    revision: u64,
    sync_serial: u64,
    alive_count: u32,
    /// Permanent, astronomically rare safety fuse. Once a live authority ID
    /// reaches `u32::MAX`, later GPU transactions are rejected before mutation
    /// instead of making preflight predict a relocation that commit alone can
    /// discover. CPU submission remains the exact `Auto` fallback.
    generation_exhausted: bool,
    residency_clock_seconds: f64,
    last_residency_sample_seconds: Option<f64>,
}

impl GpuScene {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// CPU backing retained by the stable cross-frame authority.
    ///
    /// Active bytes are the minimum safe target. A maintenance trim may reduce
    /// allocation headroom to each container's current length, but never
    /// truncates a slot, journal, domain member, or generation watermark.
    pub(crate) fn authority_capacity_usage(&self) -> CapacityUsage {
        let mut usage = vec_footprint(&self.slots, 0)
            + vec_footprint(&self.records, 0)
            + vec_footprint(&self.slot_generation_watermarks, 0)
            + estimated_hash_map_footprint(&self.lookup, 0)
            + estimated_hash_map_footprint(&self.domains, GPU_SCENE_DOMAIN_BUDGET)
            + vec_footprint(&self.reusable_slots, 0)
            + vec_footprint(&self.quarantined_slots, 0)
            + vec_footprint(&self.dirty_slots, 0)
            + vec_footprint(&self.dirty_ranges, 0)
            + self.geometries.capacity_usage()
            + self.materials.capacity_usage()
            + self.batches.capacity_usage();
        for domain in self.domains.values() {
            usage += vec_footprint(&domain.slots, 0);
            usage += vec_footprint(&domain.batch_slots, 0);
        }
        usage.baseline_bytes = usage.active_bytes;
        usage
    }

    /// Reclaim only allocation headroom from stable authority containers.
    /// Logical lengths and contents remain exact, so index-based identities
    /// and generation watermarks survive even if a Vec reallocates.
    pub(crate) fn trim_authority_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        let active_before = self.authority_capacity_usage().active_bytes;
        shrink_vec_by_family_ratio(&mut self.slots, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(&mut self.records, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(
            &mut self.slot_generation_watermarks,
            0,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(&mut self.lookup, 0, target_bytes, retained_bytes);
        shrink_hash_map_by_family_ratio(&mut self.domains, 0, target_bytes, retained_bytes);
        for domain in self.domains.values_mut() {
            shrink_vec_by_family_ratio(&mut domain.slots, 0, target_bytes, retained_bytes);
            shrink_vec_by_family_ratio(&mut domain.batch_slots, 0, target_bytes, retained_bytes);
        }
        shrink_vec_by_family_ratio(&mut self.reusable_slots, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(&mut self.quarantined_slots, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(&mut self.dirty_slots, 0, target_bytes, retained_bytes);
        shrink_vec_by_family_ratio(&mut self.dirty_ranges, 0, target_bytes, retained_bytes);
        self.geometries.trim_capacity(target_bytes, retained_bytes);
        self.materials.trim_capacity(target_bytes, retained_bytes);
        self.batches.trim_capacity(target_bytes, retained_bytes);
        debug_assert_eq!(self.authority_capacity_usage().active_bytes, active_before);
    }

    /// Retained CPU workspace whose contents have no cross-frame identity.
    pub(crate) fn scratch_capacity_usage(&self) -> CapacityUsage {
        vec_footprint(&self.domain_sweep_scratch, GPU_SCENE_DOMAIN_BUDGET)
            + estimated_hash_map_footprint(&self.multi_bin_lookup, GPU_SCENE_SCRATCH_BASELINE)
            + estimated_hash_map_footprint(
                &self.multi_bin_geometry_uniques,
                GPU_SCENE_SCRATCH_BASELINE,
            )
            + estimated_hash_map_footprint(
                &self.multi_bin_material_uniques,
                GPU_SCENE_SCRATCH_BASELINE,
            )
            + estimated_hash_set_footprint(
                &self.multi_bin_instance_uniques,
                GPU_SCENE_SCRATCH_BASELINE,
            )
            + vec_footprint(&self.multi_bin_preflight, GPU_SCENE_SCRATCH_BASELINE)
            + vec_footprint(&self.multi_bin_members, GPU_SCENE_SCRATCH_BASELINE)
            + vec_footprint(&self.multi_bin_bin_remap, GPU_SCENE_SCRATCH_BASELINE)
            + vec_footprint(&self.multi_bin_inverse_updates, GPU_SCENE_SCRATCH_BASELINE)
            + vec_footprint(&self.multi_bin_resolved, GPU_SCENE_SCRATCH_BASELINE)
            + vec_footprint(&self.multi_bin_committed, GPU_SCENE_SCRATCH_BASELINE)
            + vec_footprint(&self.active_batch_descriptors, GPU_SCENE_SCRATCH_BASELINE)
            + vec_footprint(&self.scene_batch_scratch, GPU_SCENE_SCRATCH_BASELINE)
            + vec_footprint(&self.cascade_scratch, GPU_SCENE_SCRATCH_BASELINE)
    }

    /// End the previous frame's scratch lifetime at the same safe boundary as
    /// `GraphStorage::clear`. No stable slot, journal, domain, or generation
    /// watermark is touched.
    pub(crate) fn clear_frame_scratch(&mut self) {
        self.domain_sweep_scratch.clear();
        self.multi_bin_lookup.clear();
        self.multi_bin_geometry_uniques.clear();
        self.multi_bin_material_uniques.clear();
        self.multi_bin_instance_uniques.clear();
        self.multi_bin_has_shadow_casters = false;
        self.multi_bin_preflight.clear();
        self.multi_bin_members.clear();
        self.multi_bin_bin_remap.clear();
        self.multi_bin_inverse_updates.clear();
        self.multi_bin_resolved.clear();
        self.multi_bin_committed.clear();
        self.active_batch_descriptors.clear();
        self.scene_batch_scratch.clear();
        self.cascade_scratch.clear();
    }

    pub(crate) fn trim_scratch_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        debug_assert_eq!(self.scratch_capacity_usage().active_bytes, 0);
        shrink_vec_by_family_ratio(
            &mut self.domain_sweep_scratch,
            GPU_SCENE_DOMAIN_BUDGET,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.multi_bin_lookup,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.multi_bin_geometry_uniques,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.multi_bin_material_uniques,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_set_by_family_ratio(
            &mut self.multi_bin_instance_uniques,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.multi_bin_preflight,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.multi_bin_members,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.multi_bin_bin_remap,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.multi_bin_inverse_updates,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.multi_bin_resolved,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.multi_bin_committed,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.active_batch_descriptors,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.scene_batch_scratch,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.cascade_scratch,
            GPU_SCENE_SCRATCH_BASELINE,
            target_bytes,
            retained_bytes,
        );
    }

    #[cfg(test)]
    pub(crate) fn reserve_capacity_probe(&mut self, scratch_slots: usize, authority_slots: usize) {
        self.multi_bin_preflight.reserve(scratch_slots);
        self.multi_bin_members.reserve(scratch_slots);
        self.multi_bin_bin_remap.reserve(scratch_slots);
        self.multi_bin_committed.reserve(scratch_slots);
        self.slots.reserve(authority_slots);
        self.records.reserve(authority_slots);
        self.slot_generation_watermarks.reserve(authority_slots);
        self.lookup.reserve(authority_slots);
        self.domains.reserve(authority_slots);
    }

    fn project_table_set(
        &self,
        scene_id: u32,
        visible_view_stride: u64,
        visible_partition_bytes: u64,
        bin_count: u32,
    ) -> Option<GpuSceneProjectedTableSet> {
        let instance_slot_span = projected_instance_span_for_keys(
            &self.slots,
            &self.lookup,
            self.acknowledged_tail_retained_len(),
            self.multi_bin_instance_uniques
                .iter()
                .copied()
                .map(|node| GpuSceneInstanceKey { scene_id, node }),
        )?;
        let geometry_slot_span = self
            .geometries
            .projected_slot_span(self.multi_bin_geometry_uniques.keys().copied())?;
        let material_slot_span = self
            .materials
            .projected_slot_span(self.multi_bin_material_uniques.keys().copied())?;
        let batch_slot_span = self.batches.projected_slot_span(
            self.multi_bin_preflight
                .iter()
                .map(|bin| GpuSceneBatchKey::new(scene_id, bin.key)),
        )?;
        Some(GpuSceneProjectedTableSet {
            instance_slot_span,
            geometry_slot_span,
            material_slot_span,
            batch_slot_span,
            instance_logical_bytes: logical_table_bytes::<GpuSceneInstanceRecord>(
                instance_slot_span,
            )?,
            geometry_logical_bytes: logical_table_bytes::<GpuSceneGeometryRecord>(
                geometry_slot_span,
            )?,
            material_logical_bytes: logical_table_bytes::<GpuSceneMaterialRecord>(
                material_slot_span,
            )?,
            batch_logical_bytes: logical_table_bytes::<GpuSceneBatchRecord>(batch_slot_span)?,
            visible_view_stride,
            visible_partition_bytes,
            indirect_args_span: u64::from(bin_count).checked_mul(5 * size_of::<u32>() as u64)?,
        })
    }

    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) fn active_batch_descriptors(&self) -> &[GpuSceneActiveBatchDescriptor] {
        &self.active_batch_descriptors
    }

    #[inline]
    fn instance_world_matrix_is_current(&self, scene_id: u32, item: &ExtractedRenderItem) -> bool {
        let key = GpuSceneInstanceKey {
            scene_id,
            node: item.node_handle,
        };
        self.lookup.get(&key).is_some_and(|id| {
            let record = &self.records[id.slot as usize];
            record.metadata[1] != 0
                && record.metadata[2] == scene_id
                && record.world_matrix == item.world_matrix.to_cols_array_2d()
        })
    }

    /// Build a retained, deterministic multi-bin recipe without touching any
    /// persistent table or frame-local representative. Bins appear in first
    /// occurrence order; the hash table is lookup-only and never defines
    /// output order.
    pub(crate) fn preflight_multi_bin_scene(
        &mut self,
        extracted_scene: &ExtractedScene,
        visible_stride: u64,
        storage_alignment: u64,
    ) -> Result<Option<GpuSceneMultiBinPreflight>, GpuSceneMultiBinPreflightFailure> {
        self.preflight_multi_bin_scene_with_policy(
            extracted_scene,
            visible_stride,
            storage_alignment,
            GpuSceneMultiBinSelectionPolicy::CompleteDomain,
        )
    }

    /// Build the permissive Auto recipe. Unsupported opaque objects and bins
    /// below `minimum_member_count` remain CPU-owned; stable authority is not
    /// touched until the returned token is explicitly committed.
    pub(crate) fn preflight_multi_bin_cohorts(
        &mut self,
        extracted_scene: &ExtractedScene,
        visible_stride: u64,
        storage_alignment: u64,
        minimum_member_count: u32,
    ) -> Result<Option<GpuSceneMultiBinPreflight>, GpuSceneMultiBinPreflightFailure> {
        self.preflight_multi_bin_scene_with_policy(
            extracted_scene,
            visible_stride,
            storage_alignment,
            GpuSceneMultiBinSelectionPolicy::Cohorts {
                minimum_member_count,
            },
        )
    }

    fn preflight_multi_bin_scene_with_policy(
        &mut self,
        extracted_scene: &ExtractedScene,
        visible_stride: u64,
        storage_alignment: u64,
        selection: GpuSceneMultiBinSelectionPolicy,
    ) -> Result<Option<GpuSceneMultiBinPreflight>, GpuSceneMultiBinPreflightFailure> {
        if self.generation_exhausted {
            return Err(GpuSceneMultiBinPreflightFailure::Domain(
                GpuSceneSelectionRejection::GenerationExhausted,
            ));
        }
        if visible_stride == 0 {
            return Err(GpuSceneMultiBinPreflightFailure::InvalidVisibleStride);
        }
        if storage_alignment < 8 || !storage_alignment.is_power_of_two() {
            return Err(GpuSceneMultiBinPreflightFailure::InvalidStorageAlignment);
        }
        let partition_quantum = checked_lcm(visible_stride, storage_alignment)
            .ok_or(GpuSceneMultiBinPreflightFailure::LayoutOverflow)?;

        self.multi_bin_serial = self
            .multi_bin_serial
            .checked_add(1)
            .expect("GPU Scene multi-bin preflight serial exhausted");
        self.multi_bin_lookup.clear();
        self.multi_bin_geometry_uniques.clear();
        self.multi_bin_material_uniques.clear();
        self.multi_bin_instance_uniques.clear();
        self.multi_bin_has_shadow_casters = false;
        self.multi_bin_preflight.clear();
        self.multi_bin_members.clear();
        self.multi_bin_bin_remap.clear();
        self.multi_bin_inverse_updates.clear();
        self.multi_bin_committed.clear();
        self.active_batch_descriptors.clear();
        let mut total_shadow_caster_count = 0u32;

        for (item_index, item) in extracted_scene.render_items.iter().enumerate() {
            self.multi_bin_has_shadow_casters |= item.cast_shadows;
            if item.cast_shadows {
                total_shadow_caster_count = total_shadow_caster_count.checked_add(1).ok_or(
                    GpuSceneMultiBinPreflightFailure::Domain(
                        GpuSceneSelectionRejection::TooManyCandidates,
                    ),
                )?;
            }
            let world_matrix_is_current =
                self.instance_world_matrix_is_current(extracted_scene.scene_id, item);
            let item_preflight = preflight_gpu_scene_item(item, world_matrix_is_current);
            if let Err(reason) = item_preflight
                && selection.rejects_unsupported()
            {
                return Err(GpuSceneMultiBinPreflightFailure::Domain(reason));
            }
            // Auto is deliberately permissive: transparent, unsupported, and
            // later low-reuse items remain under the CPU cohort authority.
            let Some(item_preflight) = item_preflight.ok().flatten() else {
                continue;
            };
            let key = item_preflight.key;
            if self.multi_bin_members.len() == u32::MAX as usize {
                return Err(GpuSceneMultiBinPreflightFailure::Domain(
                    GpuSceneSelectionRejection::TooManyCandidates,
                ));
            }
            if !self.multi_bin_instance_uniques.insert(item.node_handle) {
                // Duplicate live identities cannot be split safely between
                // CPU and GPU ownership; keep this a typed whole-attempt
                // rejection in both modes.
                return Err(GpuSceneMultiBinPreflightFailure::Domain(
                    GpuSceneSelectionRejection::DuplicateInstance,
                ));
            }
            let geometry_stamp = GpuSceneGeometryStamp::from_key(key);
            if self
                .multi_bin_geometry_uniques
                .insert(key.geometry, geometry_stamp)
                .is_some_and(|existing| existing != geometry_stamp)
            {
                return Err(GpuSceneMultiBinPreflightFailure::Domain(
                    GpuSceneSelectionRejection::ConflictingGeometryEpoch,
                ));
            }
            let material_stamp = GpuSceneMaterialStamp::from_key(key);
            if self
                .multi_bin_material_uniques
                .insert(key.material, material_stamp)
                .is_some_and(|existing| existing != material_stamp)
            {
                return Err(GpuSceneMultiBinPreflightFailure::Domain(
                    GpuSceneSelectionRejection::ConflictingMaterialEpoch,
                ));
            }
            let bin_index = if let Some(&bin_index) = self.multi_bin_lookup.get(&key) {
                let bin = &mut self.multi_bin_preflight[bin_index as usize];
                bin.member_count = bin.member_count.checked_add(1).ok_or(
                    GpuSceneMultiBinPreflightFailure::Domain(
                        GpuSceneSelectionRejection::TooManyCandidates,
                    ),
                )?;
                bin_index
            } else {
                let bin_index = u32::try_from(self.multi_bin_preflight.len()).map_err(|_| {
                    GpuSceneMultiBinPreflightFailure::Domain(
                        GpuSceneSelectionRejection::TooManyCandidates,
                    )
                })?;
                self.multi_bin_lookup.insert(key, bin_index);
                self.multi_bin_preflight.push(GpuScenePreflightBin {
                    key,
                    representative: ExtractedSourceId::from_index(item_index),
                    member_count: 1,
                    partition_base: 0,
                    partition_size: 0,
                    base_visible_index: 0,
                    capacity_records: 0,
                });
                bin_index
            };
            let inverse_update_index = if let Some(inverse) = item_preflight.inverse_update {
                let index = u32::try_from(self.multi_bin_inverse_updates.len()).map_err(|_| {
                    GpuSceneMultiBinPreflightFailure::Domain(
                        GpuSceneSelectionRejection::TooManyCandidates,
                    )
                })?;
                self.multi_bin_inverse_updates.push(inverse);
                index
            } else {
                u32::MAX
            };
            self.multi_bin_members.push(GpuScenePreflightMember {
                item: u32::try_from(item_index).map_err(|_| {
                    GpuSceneMultiBinPreflightFailure::Domain(
                        GpuSceneSelectionRejection::TooManyCandidates,
                    )
                })?,
                bin: bin_index,
                inverse_update: inverse_update_index,
            });
        }

        let minimum_member_count = selection.minimum_member_count();
        if minimum_member_count > 1 && !self.multi_bin_preflight.is_empty() {
            self.multi_bin_bin_remap
                .resize(self.multi_bin_preflight.len(), u32::MAX);
            let mut retained_len = 0usize;
            for read_index in 0..self.multi_bin_preflight.len() {
                let bin = self.multi_bin_preflight[read_index];
                if bin.member_count < minimum_member_count {
                    continue;
                }
                let retained_index = u32::try_from(retained_len).map_err(|_| {
                    GpuSceneMultiBinPreflightFailure::Domain(
                        GpuSceneSelectionRejection::TooManyCandidates,
                    )
                })?;
                self.multi_bin_bin_remap[read_index] = retained_index;
                if retained_len != read_index {
                    self.multi_bin_preflight[retained_len] = bin;
                }
                retained_len += 1;
            }
            self.multi_bin_preflight.truncate(retained_len);
            let bin_remap = &self.multi_bin_bin_remap;
            self.multi_bin_members.retain_mut(|member| {
                let retained = bin_remap[member.bin as usize];
                if retained == u32::MAX {
                    false
                } else {
                    member.bin = retained;
                    true
                }
            });

            // Projection facts must describe only retained cohorts. The maps
            // are scratch lookup tables, so rebuilding them is mutation-free
            // with respect to the persistent four-table authority.
            self.multi_bin_geometry_uniques.clear();
            self.multi_bin_material_uniques.clear();
            self.multi_bin_instance_uniques.clear();
            for member in &self.multi_bin_members {
                self.multi_bin_instance_uniques
                    .insert(extracted_scene.render_items[member.item as usize].node_handle);
            }
            for bin in &self.multi_bin_preflight {
                self.multi_bin_geometry_uniques
                    .insert(bin.key.geometry, GpuSceneGeometryStamp::from_key(bin.key));
                self.multi_bin_material_uniques
                    .insert(bin.key.material, GpuSceneMaterialStamp::from_key(bin.key));
            }
        }

        if self.multi_bin_preflight.is_empty() {
            return Ok(None);
        }

        let candidate_count = u32::try_from(self.multi_bin_members.len()).map_err(|_| {
            GpuSceneMultiBinPreflightFailure::Domain(GpuSceneSelectionRejection::TooManyCandidates)
        })?;
        let shadow_candidate_count = self
            .multi_bin_members
            .iter()
            .try_fold(0u32, |count, member| {
                if extracted_scene.render_items[member.item as usize].cast_shadows {
                    count.checked_add(1)
                } else {
                    Some(count)
                }
            })
            .ok_or(GpuSceneMultiBinPreflightFailure::Domain(
                GpuSceneSelectionRejection::TooManyCandidates,
            ))?;
        let covers_complete_shadow_domain = shadow_candidate_count == total_shadow_caster_count;

        let mut partition_end = 0u64;
        for bin in &mut self.multi_bin_preflight {
            let raw_size = u64::from(bin.member_count)
                .checked_mul(visible_stride)
                .ok_or(GpuSceneMultiBinPreflightFailure::LayoutOverflow)?;
            let partition_size = checked_align_up(raw_size, partition_quantum)
                .ok_or(GpuSceneMultiBinPreflightFailure::LayoutOverflow)?;
            let partition_base = partition_end;
            partition_end = partition_end
                .checked_add(partition_size)
                .ok_or(GpuSceneMultiBinPreflightFailure::LayoutOverflow)?;
            bin.partition_base = partition_base;
            bin.partition_size = partition_size;
            bin.base_visible_index = u32::try_from(partition_base / visible_stride)
                .map_err(|_| GpuSceneMultiBinPreflightFailure::LayoutOverflow)?;
            bin.capacity_records = u32::try_from(partition_size / visible_stride)
                .map_err(|_| GpuSceneMultiBinPreflightFailure::LayoutOverflow)?;
        }

        let bin_count = u32::try_from(self.multi_bin_preflight.len()).map_err(|_| {
            GpuSceneMultiBinPreflightFailure::Domain(GpuSceneSelectionRejection::TooManyCandidates)
        })?;
        let unique_geometry_count = u32::try_from(self.multi_bin_geometry_uniques.len())
            .map_err(|_| GpuSceneMultiBinPreflightFailure::LayoutOverflow)?;
        let unique_material_count = u32::try_from(self.multi_bin_material_uniques.len())
            .map_err(|_| GpuSceneMultiBinPreflightFailure::LayoutOverflow)?;
        let table_set = self
            .project_table_set(
                extracted_scene.scene_id,
                partition_end,
                partition_end,
                bin_count,
            )
            .ok_or(GpuSceneMultiBinPreflightFailure::LayoutOverflow)?;
        Ok(Some(GpuSceneMultiBinPreflight {
            serial: self.multi_bin_serial,
            scene_id: extracted_scene.scene_id,
            scene_identity: extracted_scene as *const ExtractedScene as usize,
            projection: GpuSceneMultiBinProjection {
                candidate_count,
                bin_count,
                unique_geometry_count,
                unique_material_count,
                shadow_candidate_count,
                covers_complete_shadow_domain,
                visible_stride,
                storage_alignment,
                visible_partition_bytes: partition_end,
                table_set,
            },
        }))
    }

    /// Exact shadow-caster item fact from the latest successful complete
    /// preflight scan. Callers must not consume this after an error outcome.
    #[inline]
    #[must_use]
    pub(crate) const fn preflight_has_shadow_casters(&self) -> bool {
        self.multi_bin_has_shadow_casters
    }

    /// Commit the retained recipe as one four-table CPU transaction, then
    /// publish frame-local representatives. No fallible capability check is
    /// performed after this boundary: callers must validate `preflight`'s
    /// mutation-free projection first.
    #[cfg(test)]
    pub(crate) fn commit_multi_bin_selection(
        &mut self,
        extracted_scene: &ExtractedScene,
        preflight: GpuSceneMultiBinPreflight,
        elapsed_seconds: f64,
        render_lists: &mut RenderLists,
    ) -> Result<&[GpuSceneMultiBinCandidate], GpuSceneMultiBinCommitFailure> {
        self.commit_multi_bin_selection_with_families(
            extracted_scene,
            preflight,
            GpuDrivenFamilyMask::MAIN_PREPASS,
            elapsed_seconds,
            render_lists,
        )
    }

    /// Commit one retained Auto recipe and atomically assign its exact
    /// per-source scene-submission families. Families become visible to CPU
    /// intent collection only after the authority transaction has completed;
    /// every retained member is therefore owned by exactly one backend for
    /// each selected family.
    pub(crate) fn commit_multi_bin_selection_with_families(
        &mut self,
        extracted_scene: &ExtractedScene,
        preflight: GpuSceneMultiBinPreflight,
        selected_families: GpuDrivenFamilyMask,
        elapsed_seconds: f64,
        render_lists: &mut RenderLists,
    ) -> Result<&[GpuSceneMultiBinCandidate], GpuSceneMultiBinCommitFailure> {
        assert!(
            !selected_families.is_empty(),
            "GPU Scene commit requires at least one selected family"
        );
        if preflight.serial != self.multi_bin_serial {
            return Err(GpuSceneMultiBinCommitFailure::StalePreflight);
        }
        if preflight.scene_id != extracted_scene.scene_id
            || preflight.scene_identity != extracted_scene as *const ExtractedScene as usize
        {
            return Err(GpuSceneMultiBinCommitFailure::SceneDrift);
        }
        let mut changed = self.trim_all_acknowledged_tails();
        let sync_serial = self.next_sync_serial();
        let next_revision = self
            .revision
            .checked_add(1)
            .expect("GPU Scene table-set revision exhausted");
        let now = self.observe_residency_time(elapsed_seconds);
        let swept = self.sweep_inactive_domains(extracted_scene.scene_id, now, next_revision);
        changed |= swept != 0;

        self.multi_bin_resolved.clear();
        let mut asset_tables_changed = false;
        // Resolve handle-addressed asset tables first. Epoch drift changes the
        // generation in place; no replacement identity is inserted into the
        // semantic lookup.
        for bin_index in 0..self.multi_bin_preflight.len() {
            let key = self.multi_bin_preflight[bin_index].key;
            let (geometry_id, geometry_changed) = self.resolve_geometry(key, next_revision);
            let (material_id, material_changed) = self.resolve_material(key, next_revision);
            asset_tables_changed |= geometry_changed | material_changed;
            changed |= geometry_changed | material_changed;
            self.multi_bin_resolved.push(GpuSceneResolvedBin {
                geometry_id,
                material_id,
                batch_id: GpuSceneBatchId {
                    slot: u32::MAX,
                    generation: 0,
                },
            });
        }
        if asset_tables_changed {
            // Stable frames must not scan every Batch retained by dormant
            // scene domains. A cascade can only be required after an asset
            // table allocation or generation change in this transaction.
            changed |= self.cascade_asset_generations(next_revision);
        }

        let mut domain = self
            .domains
            .remove(&extracted_scene.scene_id)
            .unwrap_or_default();
        domain.last_used_seconds = now;
        domain.last_used_sync = sync_serial;
        self.scene_batch_scratch.clear();
        std::mem::swap(&mut domain.batch_slots, &mut self.scene_batch_scratch);

        // Resolve stable scene+handle+fixed-state batches. Partition and member
        // count changes patch the record without changing its generation.
        for bin_index in 0..self.multi_bin_preflight.len() {
            let bin = self.multi_bin_preflight[bin_index];
            let resolved = self.multi_bin_resolved[bin_index];
            let (batch_id, batch_changed, inserted) = self.resolve_batch(
                extracted_scene.scene_id,
                bin,
                resolved.geometry_id,
                resolved.material_id,
                next_revision,
            );
            changed |= batch_changed;
            self.multi_bin_resolved[bin_index].batch_id = batch_id;
            self.batches.mark_seen(authority_id(batch_id), sync_serial);
            domain.batch_slots.push(batch_id.slot());
            if inserted {
                debug_assert_eq!(
                    self.batches.slots[batch_id.slot() as usize].ref_count,
                    0,
                    "new GPU Scene Batch unexpectedly retained a domain reference"
                );
                self.batches.add_ref(authority_id(batch_id));
                self.geometries.add_ref(authority_id(resolved.geometry_id));
                self.materials.add_ref(authority_id(resolved.material_id));
            } else {
                // Batch identity contains `scene_id`, so an existing lookup is
                // necessarily the one reference already owned by this domain.
                // `inserted` is therefore the O(1) membership proof; scanning
                // the old slot list here would turn a stable B-bin frame into
                // O(B²) work.
                let slot = &self.batches.slots[batch_id.slot() as usize];
                debug_assert_eq!(slot.ref_count, 1);
                debug_assert_eq!(
                    slot.key.map(|key| key.scene_id),
                    Some(extracted_scene.scene_id)
                );
            }
        }

        let mut old_batch_index = 0;
        while old_batch_index < self.scene_batch_scratch.len() {
            let slot_id = self.scene_batch_scratch[old_batch_index];
            if self.batches.slots[slot_id as usize].seen_sync == sync_serial {
                old_batch_index += 1;
                continue;
            }
            self.release_batch_slot(slot_id, next_revision);
            changed = true;
            self.scene_batch_scratch.swap_remove(old_batch_index);
        }
        self.scene_batch_scratch.clear();

        let selected_instance_flags = gpu_scene_instance_family_flags(selected_families);
        let mut inserted_instances = 0u32;
        let mut retired_instances = 0u32;
        for member_index in 0..self.multi_bin_members.len() {
            let member = self.multi_bin_members[member_index];
            let item = &extracted_scene.render_items[member.item as usize];
            let resolved = self.multi_bin_resolved[member.bin as usize];
            let input = GpuSceneInstanceInput::new(
                item.node_handle,
                item.world_matrix,
                item.prev_world_matrix,
                item.world_aabb,
            )
            .with_flags(
                selected_instance_flags
                    | if item.cast_shadows {
                        GPU_SCENE_INSTANCE_CASTS_SHADOW
                    } else {
                        0
                    },
            );
            let key = GpuSceneInstanceKey {
                scene_id: extracted_scene.scene_id,
                node: item.node_handle,
            };
            let existing = self.lookup.get(&key).copied();
            let world_matrix_inverse = if member.inverse_update == u32::MAX {
                let id = existing
                    .expect("GPU Scene stable-inverse recipe lost its retained instance identity");
                self.records[id.slot as usize].world_matrix_inverse
            } else {
                self.multi_bin_inverse_updates[member.inverse_update as usize]
            };
            if let Some(id) = existing {
                let slot_index = id.slot as usize;
                assert_ne!(self.slots[slot_index].seen_sync, sync_serial);
                self.slots[slot_index].seen_sync = sync_serial;
                let dependency_drift = self.slots[slot_index].batch_id != Some(resolved.batch_id);
                let current = if dependency_drift {
                    changed = true;
                    let current = self.bump_instance_generation(id, resolved, next_revision);
                    if current.slot != id.slot {
                        for domain_slot in &mut domain.slots {
                            if *domain_slot == id.slot {
                                *domain_slot = current.slot;
                            }
                        }
                    }
                    current
                } else {
                    id
                };
                let next = GpuSceneInstanceRecord::live_with_batch(
                    input,
                    world_matrix_inverse,
                    current.generation,
                    extracted_scene.scene_id,
                    Some(resolved.batch_id),
                );
                let current_index = current.slot as usize;
                if !self.records[current_index].bitwise_eq(&next) {
                    self.records[current_index] = next;
                    self.mark_dirty(current.slot);
                    changed = true;
                }
                self.slots[current_index].geometry_id = Some(resolved.geometry_id);
                self.slots[current_index].material_id = Some(resolved.material_id);
                self.slots[current_index].batch_id = Some(resolved.batch_id);
            } else {
                let id = self.allocate_slot_with_dependencies(
                    key,
                    input,
                    world_matrix_inverse,
                    resolved,
                    sync_serial,
                );
                domain.slots.push(id.slot);
                inserted_instances += 1;
                changed = true;
            }
        }

        let mut domain_index = 0;
        while domain_index < domain.slots.len() {
            let slot_id = domain.slots[domain_index];
            if self.slots[slot_id as usize].seen_sync == sync_serial {
                domain_index += 1;
                continue;
            }
            self.retire_slot(slot_id, next_revision);
            domain.slots.swap_remove(domain_index);
            retired_instances += 1;
            changed = true;
        }
        if !domain.slots.is_empty() || !domain.batch_slots.is_empty() {
            assert!(
                self.domains
                    .insert(extracted_scene.scene_id, domain)
                    .is_none()
            );
        }
        self.alive_count = self
            .alive_count
            .checked_add(inserted_instances)
            .and_then(|count| count.checked_sub(retired_instances + swept))
            .expect("GPU Scene instance alive count drift");
        if changed {
            self.revision = next_revision;
        }

        // Publish a dense CPU complement with one ordered linear merge. The
        // retained members preserve extraction order, so Hybrid collection can
        // consume O(CpuComplement) sources instead of rescanning O(Scene) once
        // for Main and again for every shadow view.
        render_lists.begin_gpu_cpu_complement(selected_families);
        let mut incomplete_families = GpuDrivenFamilyMask::empty();
        if selected_families.contains(GpuDrivenFamilyMask::MAIN_PREPASS)
            && preflight.projection.candidate_count as usize != extracted_scene.render_items.len()
        {
            incomplete_families.insert(GpuDrivenFamilyMask::MAIN_PREPASS);
        }
        if !preflight.projection.covers_complete_shadow_domain {
            incomplete_families |= selected_families
                & (GpuDrivenFamilyMask::SHADOW_2D | GpuDrivenFamilyMask::SHADOW_CUBE);
        }
        if !incomplete_families.is_empty() {
            let mut retained_index = 0usize;
            for item_index in 0..extracted_scene.render_items.len() {
                let retained = self
                    .multi_bin_members
                    .get(retained_index)
                    .is_some_and(|member| member.item as usize == item_index);
                if retained {
                    retained_index += 1;
                } else {
                    let item = &extracted_scene.render_items[item_index];
                    render_lists.append_gpu_cpu_complement(
                        ExtractedSourceId::from_index(item_index),
                        item.cast_shadows,
                        incomplete_families,
                    );
                }
            }
            debug_assert_eq!(
                retained_index,
                self.multi_bin_members.len(),
                "retained GPU Scene members must remain extraction-order sorted"
            );
        }

        // Publish ownership only after the persistent authority transaction
        // has no remaining fallible boundary. These attempt-local claims are
        // a debug-only CPU xor GPU proof. Release collection consumes the
        // dense complement directly and performs no scene-width stamp writes.
        #[cfg(any(debug_assertions, test))]
        for member in &self.multi_bin_members {
            render_lists.claim_gpu_family_ownership(
                ExtractedSourceId::from_index(member.item as usize),
                selected_families,
            );
        }

        self.multi_bin_committed.clear();
        self.active_batch_descriptors.clear();
        for bin_index in 0..self.multi_bin_preflight.len() {
            let bin = self.multi_bin_preflight[bin_index];
            let batch_id = self.multi_bin_resolved[bin_index].batch_id;
            let representative = render_lists.append_gpu_opaque_representative(bin.representative);
            self.multi_bin_committed.push(GpuSceneMultiBinCandidate {
                batch_id,
                key: bin.key,
                representative,
                member_count: bin.member_count,
                partition_base: bin.partition_base,
                partition_size: bin.partition_size,
                base_visible_index: bin.base_visible_index,
                capacity_records: bin.capacity_records,
            });
            self.active_batch_descriptors
                .push(GpuSceneActiveBatchDescriptor {
                    id: batch_id,
                    partition_base: bin.partition_base,
                    partition_size: bin.partition_size,
                });
        }
        Ok(&self.multi_bin_committed)
    }

    fn resolve_geometry(
        &mut self,
        key: GpuSceneRigidBatchKey,
        revision: u64,
    ) -> (GpuSceneGeometryId, bool) {
        if let Some(id) = self.geometries.id(key.geometry) {
            let desired = GpuSceneGeometryRecord::live(key, id.generation);
            if self.geometries.records[id.slot as usize].bitwise_eq(&desired) {
                return (geometry_id(id), false);
            }
            let next = self
                .geometries
                .bump_generation_with(id, revision, |generation| {
                    GpuSceneGeometryRecord::live(key, generation)
                });
            self.generation_exhausted |= next.generation == u32::MAX;
            return (geometry_id(next), true);
        }
        let id = self.geometries.allocate_with(key.geometry, |generation| {
            GpuSceneGeometryRecord::live(key, generation)
        });
        (geometry_id(id), true)
    }

    fn resolve_material(
        &mut self,
        key: GpuSceneRigidBatchKey,
        revision: u64,
    ) -> (GpuSceneMaterialId, bool) {
        if let Some(id) = self.materials.id(key.material) {
            let desired = GpuSceneMaterialRecord::live(key, id.generation);
            if self.materials.records[id.slot as usize].bitwise_eq(&desired) {
                return (material_id(id), false);
            }
            let next = self
                .materials
                .bump_generation_with(id, revision, |generation| {
                    GpuSceneMaterialRecord::live(key, generation)
                });
            self.generation_exhausted |= next.generation == u32::MAX;
            return (material_id(next), true);
        }
        let id = self.materials.allocate_with(key.material, |generation| {
            GpuSceneMaterialRecord::live(key, generation)
        });
        (material_id(id), true)
    }

    /// Refresh every retained Batch at most once after all asset records have
    /// reached their new generation. This avoids a Geometry+Material drift
    /// causing two transient Batch/Instance generations in one transaction.
    fn cascade_asset_generations(&mut self, revision: u64) -> bool {
        self.cascade_scratch.clear();
        for (index, slot) in self.batches.slots.iter().enumerate() {
            let Some(key) = slot.key else { continue };
            let geometry = self
                .geometries
                .id(key.geometry)
                .expect("live Batch lost its Geometry dependency");
            let material = self
                .materials
                .id(key.material)
                .expect("live Batch lost its Material dependency");
            let record = self.batches.records[index];
            if record.geometry_id != [geometry.slot, geometry.generation]
                || record.material_id != [material.slot, material.generation]
            {
                self.cascade_scratch.push(index as u32);
            }
        }
        let mut changed = false;
        while let Some(slot_id) = self.cascade_scratch.pop() {
            let index = slot_id as usize;
            let key = self.batches.slots[index]
                .key
                .expect("cascade retained an empty Batch slot");
            let old = AuthorityId {
                slot: slot_id,
                generation: self.batches.slots[index].generation,
            };
            let geometry = self.geometries.id(key.geometry).unwrap();
            let material = self.materials.id(key.material).unwrap();
            let geometry_record = self.geometries.records[geometry.slot as usize];
            let mut next_record = self.batches.records[index];
            next_record.geometry_id = [geometry.slot, geometry.generation];
            next_record.material_id = [material.slot, material.generation];
            next_record.draw_state[2] = geometry_record.metadata[3];
            next_record.draw_state[5] = geometry_record.fixed_state[0];
            next_record.draw_state[6] = geometry_record.fixed_state[1];
            let next = self
                .batches
                .bump_generation_with(old, revision, |generation| {
                    next_record.metadata[0] = generation;
                    next_record
                });
            self.generation_exhausted |= next.generation == u32::MAX;
            if next.slot != old.slot {
                for domain in self.domains.values_mut() {
                    for domain_slot in &mut domain.batch_slots {
                        if *domain_slot == old.slot {
                            *domain_slot = next.slot;
                        }
                    }
                }
            }
            // Instances in the current scene are repaired by the one member
            // scan below. Dormant domains deliberately retain the stale Batch
            // generation and therefore fail closed until their next sync,
            // avoiding O(changed_batches * total_instances) eager cascades.
            changed = true;
        }
        changed
    }

    fn resolve_batch(
        &mut self,
        scene_id: u32,
        bin: GpuScenePreflightBin,
        geometry: GpuSceneGeometryId,
        material: GpuSceneMaterialId,
        revision: u64,
    ) -> (GpuSceneBatchId, bool, bool) {
        let key = GpuSceneBatchKey::new(scene_id, bin.key);
        if let Some(id) = self.batches.id(key) {
            let desired =
                GpuSceneBatchRecord::live(scene_id, bin, id.generation, geometry, material);
            let current = self.batches.records[id.slot as usize];
            if current.geometry_id != geometry.words() || current.material_id != material.words() {
                let next = self
                    .batches
                    .bump_generation_with(id, revision, |generation| {
                        GpuSceneBatchRecord::live(scene_id, bin, generation, geometry, material)
                    });
                self.generation_exhausted |= next.generation == u32::MAX;
                return (batch_id(next), true, false);
            }
            let changed = !current.bitwise_eq(&desired);
            if changed {
                self.batches.update_with(id, |_| desired);
            }
            return (batch_id(id), changed, false);
        }
        let id = self.batches.allocate_with(key, |generation| {
            GpuSceneBatchRecord::live(scene_id, bin, generation, geometry, material)
        });
        (batch_id(id), true, true)
    }

    fn release_batch_slot(&mut self, slot_id: u32, revision: u64) {
        let index = slot_id as usize;
        let id = AuthorityId {
            slot: slot_id,
            generation: self.batches.slots[index].generation,
        };
        let record = self.batches.records[index];
        if self.batches.release(id, revision) {
            self.geometries.release(
                AuthorityId {
                    slot: record.geometry_id[0],
                    generation: record.geometry_id[1],
                },
                revision,
            );
            self.materials.release(
                AuthorityId {
                    slot: record.material_id[0],
                    generation: record.material_id[1],
                },
                revision,
            );
        }
    }

    /// Compute the exact table span a complete opaque-domain synchronization
    /// would require without mutating the persistent authority. Backend
    /// capability selection uses this before `Auto` commits to GPU topology,
    /// so an unsupported workload falls back without touching the dirty
    /// journal or retiring any stable slot.
    #[cfg(test)]
    pub(crate) fn projected_slot_span_for_extracted_scene(
        &self,
        extracted_scene: &ExtractedScene,
    ) -> Option<u32> {
        let scene_id = extracted_scene.scene_id;
        let mut new_instances = 0usize;
        for item in extracted_scene
            .render_items
            .iter()
            .filter(|item| is_gpu_scene_prefrustum_opaque(item))
        {
            let key = GpuSceneInstanceKey {
                scene_id,
                node: item.node_handle,
            };
            if !self.lookup.contains_key(&key) {
                new_instances = new_instances.checked_add(1)?;
            }
        }
        // Mirror the next safe-boundary trim without mutating authority during
        // capability preflight. Only already-acknowledged reusable slots count:
        // domains that this frame's later transaction may sweep remain live or
        // quarantined and therefore still contribute to the conservative span.
        let effective_len = self.acknowledged_tail_retained_len();
        let interior_reusable = self.slots[..effective_len]
            .iter()
            .filter(|slot| slot.reusable)
            .count();
        let appended = new_instances.saturating_sub(interior_reusable);
        effective_len
            .checked_add(appended)
            .and_then(|span| u32::try_from(span).ok())
    }

    /// Synchronize the complete pre-frustum opaque domain. Backend selection
    /// chooses only one fixed-state representative, but stable slot retirement
    /// must observe every managed node rather than mistaking omitted instances
    /// for deletion. Transparent/transmission objects belong to the independent
    /// CPU stream and never occupy this first GPU Scene table. `elapsed_seconds`
    /// is the renderer's real elapsed clock, not a frame counter or simulation
    /// tick; it drives dormant-domain residency only after backend preflight has
    /// selected this production path.
    #[cfg(test)]
    pub(crate) fn synchronize_extracted_scene(
        &mut self,
        extracted_scene: &ExtractedScene,
        elapsed_seconds: f64,
    ) -> GpuSceneSyncStats {
        self.synchronize_scene_transaction(
            extracted_scene.scene_id,
            extracted_scene
                .render_items
                .iter()
                .filter(|item| is_gpu_scene_prefrustum_opaque(item))
                .map(|item| {
                    GpuSceneInstanceInput::new(
                        item.node_handle,
                        item.world_matrix,
                        item.prev_world_matrix,
                        item.world_aabb,
                    )
                    .with_flags(if item.cast_shadows {
                        GPU_SCENE_INSTANCE_CASTS_SHADOW
                    } else {
                        0
                    })
                }),
            Some(elapsed_seconds),
        )
    }

    /// Synchronize one complete scene domain as one logical revision.
    ///
    /// Every supplied node must be unique. Missing nodes are retired. Any
    /// number of inserts/updates/retires advances the table revision exactly
    /// once and every changed slot appears exactly once in the replay journal.
    #[cfg(test)]
    pub(crate) fn synchronize_scene(
        &mut self,
        scene_id: u32,
        instances: impl IntoIterator<Item = GpuSceneInstanceInput>,
    ) -> GpuSceneSyncStats {
        self.synchronize_scene_transaction(scene_id, instances, None)
    }

    #[cfg(test)]
    fn synchronize_scene_transaction(
        &mut self,
        scene_id: u32,
        instances: impl IntoIterator<Item = GpuSceneInstanceInput>,
        residency_sample_seconds: Option<f64>,
    ) -> GpuSceneSyncStats {
        // Only an exact acknowledgement makes an empty slot reusable. Trimming
        // here is therefore a safe boundary, and it happens before any new
        // allocation can regrow the compact suffix.
        self.trim_acknowledged_trailing_slots();
        let sync_serial = self.next_sync_serial();
        let next_revision = self
            .revision
            .checked_add(1)
            .expect("GPU Scene table revision exhausted");
        let residency_now =
            residency_sample_seconds.map(|sample| self.observe_residency_time(sample));
        let mut retired = residency_now.map_or(0, |now| {
            self.sweep_inactive_domains(scene_id, now, next_revision)
        });
        let mut domain = self.domains.remove(&scene_id).unwrap_or_default();
        if let Some(now) = residency_now {
            domain.last_used_seconds = now;
        }
        domain.last_used_sync = sync_serial;
        let mut inserted = 0u32;
        let mut updated = 0u32;

        for input in instances {
            let key = GpuSceneInstanceKey {
                scene_id,
                node: input.node,
            };
            if let Some(id) = self.lookup.get(&key).copied() {
                let slot_index = id.slot as usize;
                let slot = self
                    .slots
                    .get(slot_index)
                    .expect("GPU Scene lookup retained an out-of-range slot");
                assert_eq!(
                    slot.key,
                    Some(key),
                    "GPU Scene lookup retained a slot owned by another instance"
                );
                assert_eq!(
                    slot.generation, id.generation,
                    "GPU Scene lookup retained a stale generation"
                );
                assert_ne!(
                    slot.seen_sync, sync_serial,
                    "one scene synchronization contained a duplicate NodeHandle"
                );

                self.slots[slot_index].seen_sync = sync_serial;
                let next = GpuSceneInstanceRecord::live(input, id.generation, scene_id);
                if !self.records[slot_index].bitwise_eq(&next) {
                    self.records[slot_index] = next;
                    self.mark_dirty(id.slot);
                    updated = updated
                        .checked_add(1)
                        .expect("GPU Scene update count overflowed");
                }
                continue;
            }

            let id = self.allocate_slot(key, input, sync_serial);
            domain.slots.push(id.slot);
            inserted = inserted
                .checked_add(1)
                .expect("GPU Scene insertion count overflowed");
        }

        let mut domain_index = 0;
        while domain_index < domain.slots.len() {
            let slot_id = domain.slots[domain_index];
            let slot_index = slot_id as usize;
            if self.slots[slot_index].seen_sync == sync_serial {
                domain_index += 1;
                continue;
            }

            self.retire_slot(slot_id, next_revision);
            domain.slots.swap_remove(domain_index);
            retired = retired
                .checked_add(1)
                .expect("GPU Scene retirement count overflowed");
        }

        let retired_batches = !domain.batch_slots.is_empty();
        for slot_id in domain.batch_slots.drain(..) {
            self.release_batch_slot(slot_id, next_revision);
        }

        if !domain.slots.is_empty() || !domain.batch_slots.is_empty() {
            let previous = self.domains.insert(scene_id, domain);
            debug_assert!(previous.is_none());
        }

        if inserted != 0 || updated != 0 || retired != 0 || retired_batches {
            self.revision = next_revision;
        }
        self.alive_count = self
            .alive_count
            .checked_add(inserted)
            .and_then(|count| count.checked_sub(retired))
            .expect("GPU Scene alive count overflowed or underflowed");

        GpuSceneSyncStats {
            revision: self.revision,
            inserted,
            updated,
            retired,
            alive: self.alive_count,
        }
    }

    /// Retire every instance owned by one scene domain as a normal replayable
    /// table revision. Call this when a scene leaves the renderer permanently;
    /// simply ceasing to synchronize a domain intentionally does not guess at
    /// ownership loss or reclaim its stable IDs.
    #[cfg(test)]
    pub(crate) fn retire_scene(&mut self, scene_id: u32) -> GpuSceneSyncStats {
        self.synchronize_scene(scene_id, std::iter::empty::<GpuSceneInstanceInput>())
    }

    /// Plan the exact persistent instance-table revision without allocating a
    /// physical buffer. A previously submitted exact revision acknowledges the
    /// dirty journal and releases tombstoned slots from quarantine.
    #[cfg(test)]
    pub(crate) fn plan_instance_table(
        &mut self,
        registry: &mut PersistentBufferRegistry,
        attempt_serial: u64,
    ) -> Result<GpuSceneTablePlan<'_>, PersistentBufferPlanFailure> {
        self.trim_acknowledged_trailing_slots();
        let id = *self
            .persistent_instance_buffer
            .get_or_insert_with(|| registry.create_buffer());
        let logical_records = self.records.len().max(1);
        let stride = size_of::<GpuSceneInstanceRecord>() as u64;
        let logical_size = u64::try_from(logical_records)
            .expect("GPU Scene record count exceeds u64")
            .checked_mul(stride)
            .expect("GPU Scene instance-table byte size overflowed");
        let spec = PersistentBufferSpec::new(
            BufferDesc::new(logical_size, GPU_SCENE_INSTANCE_BUFFER_USAGE),
            GPU_SCENE_INSTANCE_SCHEMA_ID,
            stride,
        );
        let persistent = registry.plan_buffer(id, attempt_serial, self.revision, spec)?;

        let publication_probe = GpuScenePublicationProbe {
            buffer: persistent.buffer,
            revision: self.revision,
        };
        if persistent.valid {
            assert!(
                persistent.commit.is_none(),
                "published GPU Scene table unexpectedly requested a commit"
            );
            assert!(
                self.acknowledge_if_published(registry, publication_probe),
                "valid GPU Scene table plan lacked exact registry publication proof"
            );
        }
        self.rebuild_dirty_ranges();
        let records = if self.records.is_empty() {
            std::slice::from_ref(&EMPTY_GPU_SCENE_INSTANCE_RECORD)
        } else {
            &self.records
        };
        let slot_span = self.slot_span();

        Ok(GpuSceneTablePlan {
            persistent,
            publication_probe,
            revision: self.revision,
            records,
            dirty_ranges: &self.dirty_ranges,
            slot_span,
            alive_count: self.alive_count,
        })
    }

    #[cfg(test)]
    pub(crate) fn plan_table_set(
        &mut self,
        registry: &mut PersistentBufferRegistry,
        attempt_serial: u64,
    ) -> Result<GpuSceneTableSetPlan<'_>, PersistentBufferBatchPlanFailure> {
        let (plan, _acknowledger) =
            self.plan_table_set_with_acknowledger(registry, attempt_serial)?;
        Ok(plan)
    }

    /// Atomically plan Instance/Geometry/Material/Batch at one shared logical
    /// revision. A tail-spec change advances that revision even when no POD
    /// field changed, preventing a warm subset from mixing with a resized
    /// member at the same semantic revision.
    pub(crate) fn plan_table_set_with_acknowledger(
        &mut self,
        registry: &mut PersistentBufferRegistry,
        attempt_serial: u64,
    ) -> Result<
        (
            GpuSceneTableSetPlan<'_>,
            GpuSceneTableSetPublicationAcknowledger<'_>,
        ),
        PersistentBufferBatchPlanFailure,
    > {
        if self.trim_all_acknowledged_tails() {
            self.revision = self
                .revision
                .checked_add(1)
                .expect("GPU Scene table-set revision exhausted on tail trim");
        }
        let ids = [
            *self
                .persistent_instance_buffer
                .get_or_insert_with(|| registry.create_buffer()),
            *self
                .persistent_geometry_buffer
                .get_or_insert_with(|| registry.create_buffer()),
            *self
                .persistent_material_buffer
                .get_or_insert_with(|| registry.create_buffer()),
            *self
                .persistent_batch_buffer
                .get_or_insert_with(|| registry.create_buffer()),
        ];
        let specs = [
            table_spec::<GpuSceneInstanceRecord>(self.records.len(), GPU_SCENE_INSTANCE_SCHEMA_ID),
            table_spec::<GpuSceneGeometryRecord>(
                self.geometries.records.len(),
                GPU_SCENE_GEOMETRY_SCHEMA_ID,
            ),
            table_spec::<GpuSceneMaterialRecord>(
                self.materials.records.len(),
                GPU_SCENE_MATERIAL_SCHEMA_ID,
            ),
            table_spec::<GpuSceneBatchRecord>(
                self.batches.records.len(),
                GPU_SCENE_BATCH_SCHEMA_ID,
            ),
        ];
        let requests: [PersistentBufferPlanRequest; 4] =
            std::array::from_fn(|index| PersistentBufferPlanRequest::new(ids[index], specs[index]));
        let plans = registry.plan_coherent_batch(attempt_serial, self.revision, &requests)?;
        debug_assert_eq!(plans.len(), 4);
        let publication_probe = GpuSceneTableSetPublicationProbe {
            buffers: std::array::from_fn(|index| plans[index].buffer),
            revision: self.revision,
        };
        if plans[0].valid {
            debug_assert!(plans.iter().all(|plan| plan.valid && plan.commit.is_none()));
            assert!(
                self.acknowledge_table_set_if_published(registry, &publication_probe),
                "valid GPU Scene table set lacked coherent publication proof"
            );
        } else {
            debug_assert!(
                plans
                    .iter()
                    .all(|plan| !plan.valid && plan.commit.is_some())
            );
        }

        self.rebuild_dirty_ranges();
        self.geometries.rebuild_dirty_ranges();
        self.materials.rebuild_dirty_ranges();
        self.batches.rebuild_dirty_ranges();

        let instance_records = if self.records.is_empty() {
            std::slice::from_ref(&EMPTY_GPU_SCENE_INSTANCE_RECORD)
        } else {
            &self.records
        };
        let geometry_records = if self.geometries.records.is_empty() {
            std::slice::from_ref(&EMPTY_GPU_SCENE_GEOMETRY_RECORD)
        } else {
            &self.geometries.records
        };
        let material_records = if self.materials.records.is_empty() {
            std::slice::from_ref(&EMPTY_GPU_SCENE_MATERIAL_RECORD)
        } else {
            &self.materials.records
        };
        let batch_records = if self.batches.records.is_empty() {
            std::slice::from_ref(&EMPTY_GPU_SCENE_BATCH_RECORD)
        } else {
            &self.batches.records
        };
        let plan = GpuSceneTableSetPlan {
            #[cfg(test)]
            revision: self.revision,
            publication_probe,
            instances: GpuSceneTableMemberPlan {
                persistent: plans[0],
                records: instance_records,
                dirty_ranges: &self.dirty_ranges,
                slot_span: self.slot_span(),
            },
            geometries: GpuSceneTableMemberPlan {
                persistent: plans[1],
                records: geometry_records,
                dirty_ranges: &self.geometries.dirty_ranges,
                slot_span: self.geometries.slot_span(),
            },
            materials: GpuSceneTableMemberPlan {
                persistent: plans[2],
                records: material_records,
                dirty_ranges: &self.materials.dirty_ranges,
                slot_span: self.materials.slot_span(),
            },
            batches: GpuSceneTableMemberPlan {
                persistent: plans[3],
                records: batch_records,
                dirty_ranges: &self.batches.dirty_ranges,
                slot_span: self.batches.slot_span(),
            },
            candidates: &self.multi_bin_committed,
            active_batches: GpuSceneActiveBatchSet::from_authority(&self.active_batch_descriptors),
        };
        let acknowledger = GpuSceneTableSetPublicationAcknowledger {
            persistent_buffers: [
                self.persistent_instance_buffer,
                self.persistent_geometry_buffer,
                self.persistent_material_buffer,
                self.persistent_batch_buffer,
            ],
            revision: self.revision,
            instance_slots: &mut self.slots,
            instance_dirty_slots: &mut self.dirty_slots,
            instance_quarantine: &mut self.quarantined_slots,
            instance_reusable: &mut self.reusable_slots,
            geometry_slots: &mut self.geometries.slots,
            geometry_dirty_slots: &mut self.geometries.dirty_slots,
            geometry_quarantine: &mut self.geometries.quarantined_slots,
            geometry_reusable: &mut self.geometries.reusable_slots,
            material_slots: &mut self.materials.slots,
            material_dirty_slots: &mut self.materials.dirty_slots,
            material_quarantine: &mut self.materials.quarantined_slots,
            material_reusable: &mut self.materials.reusable_slots,
            batch_slots: &mut self.batches.slots,
            batch_dirty_slots: &mut self.batches.dirty_slots,
            batch_quarantine: &mut self.batches.quarantined_slots,
            batch_reusable: &mut self.batches.reusable_slots,
        };
        Ok((plan, acknowledger))
    }

    pub(crate) fn acknowledge_table_set_if_published(
        &mut self,
        registry: &PersistentBufferRegistry,
        probe: &GpuSceneTableSetPublicationProbe,
    ) -> bool {
        let ids = [
            self.persistent_instance_buffer,
            self.persistent_geometry_buffer,
            self.persistent_material_buffer,
            self.persistent_batch_buffer,
        ];
        if probe.revision != self.revision
            || probe.buffers.iter().zip(ids).any(|(buffer, id)| {
                id != Some(buffer.id)
                    || buffer.revision != probe.revision
                    || !registry.is_published(*buffer)
            })
        {
            return false;
        }
        self.acknowledge_submitted_revision(probe.revision)
            && self.geometries.acknowledge(self.revision, probe.revision)
            && self.materials.acknowledge(self.revision, probe.revision)
            && self.batches.acknowledge(self.revision, probe.revision)
    }

    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) fn instance_id(
        &self,
        scene_id: u32,
        node: NodeHandle,
    ) -> Option<GpuSceneInstanceId> {
        self.lookup
            .get(&GpuSceneInstanceKey { scene_id, node })
            .copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn slot_span(&self) -> u32 {
        u32::try_from(self.records.len()).expect("GPU Scene instance-table slot span exceeds u32")
    }

    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(crate) fn record(&self, id: GpuSceneInstanceId) -> Option<&GpuSceneInstanceRecord> {
        let slot = self.slots.get(id.slot as usize)?;
        (slot.key.is_some() && slot.generation == id.generation)
            .then(|| &self.records[id.slot as usize])
    }

    /// Release only the journal proven to have reached the queue and registry.
    ///
    /// Callers retain the copy-only probe outside the graph, publish the graph's
    /// persistent commit groups, drop the graph (which releases borrowed record
    /// slices), and then call this method in the same frame. A stale attempt,
    /// foreign owner, superseded revision, or failed publication is fail-closed.
    #[cfg(test)]
    pub(crate) fn acknowledge_if_published(
        &mut self,
        registry: &PersistentBufferRegistry,
        probe: GpuScenePublicationProbe,
    ) -> bool {
        if self.persistent_instance_buffer != Some(probe.buffer.id)
            || probe.buffer.revision != probe.revision
            || probe.revision != self.revision
            || !registry.is_published(probe.buffer)
        {
            return false;
        }
        self.acknowledge_submitted_revision(probe.revision)
    }

    #[cfg(test)]
    fn allocate_slot(
        &mut self,
        key: GpuSceneInstanceKey,
        input: GpuSceneInstanceInput,
        sync_serial: u64,
    ) -> GpuSceneInstanceId {
        self.allocate_instance_with(key, sync_serial, None, |generation| {
            GpuSceneInstanceRecord::live(input, generation, key.scene_id)
        })
    }

    fn allocate_slot_with_dependencies(
        &mut self,
        key: GpuSceneInstanceKey,
        input: GpuSceneInstanceInput,
        world_matrix_inverse: [[f32; 4]; 4],
        resolved: GpuSceneResolvedBin,
        sync_serial: u64,
    ) -> GpuSceneInstanceId {
        self.allocate_instance_with(key, sync_serial, Some(resolved), |generation| {
            GpuSceneInstanceRecord::live_with_batch(
                input,
                world_matrix_inverse,
                generation,
                key.scene_id,
                Some(resolved.batch_id),
            )
        })
    }

    fn allocate_instance_with(
        &mut self,
        key: GpuSceneInstanceKey,
        sync_serial: u64,
        resolved: Option<GpuSceneResolvedBin>,
        make_record: impl FnOnce(u32) -> GpuSceneInstanceRecord,
    ) -> GpuSceneInstanceId {
        let slot_id = if let Some(slot) = self.reusable_slots.pop() {
            let reusable = &self.slots[slot as usize];
            assert!(
                reusable.reusable,
                "GPU Scene free list retained a busy slot"
            );
            slot
        } else {
            let slot = u32::try_from(self.slots.len())
                .expect("GPU Scene cannot allocate more than u32::MAX instance slots");
            assert_ne!(
                slot,
                u32::MAX,
                "GPU Scene cannot represent a u32::MAX + 1 slot span"
            );
            let slot_index = slot as usize;
            let generation =
                if let Some(&generation) = self.slot_generation_watermarks.get(slot_index) {
                    assert_ne!(
                        generation,
                        u32::MAX,
                        "GPU Scene attempted to regrow a permanently retired slot"
                    );
                    generation
                } else {
                    assert_eq!(
                        slot_index,
                        self.slot_generation_watermarks.len(),
                        "GPU Scene generation watermark developed a sparse tail"
                    );
                    self.slot_generation_watermarks.push(1);
                    1
                };
            self.slots.push(InstanceSlot {
                generation,
                key: None,
                geometry_id: None,
                material_id: None,
                batch_id: None,
                seen_sync: 0,
                dirty: false,
                reusable: false,
            });
            self.records
                .push(GpuSceneInstanceRecord::tombstone(generation));
            slot
        };
        let slot = &mut self.slots[slot_id as usize];
        assert!(slot.key.is_none(), "GPU Scene reused a live slot");
        assert!(!slot.dirty, "GPU Scene reused an unsubmitted dirty slot");
        slot.reusable = false;
        let id = GpuSceneInstanceId {
            slot: slot_id,
            generation: slot.generation,
        };
        slot.key = Some(key);
        slot.geometry_id = resolved.map(|resolved| resolved.geometry_id);
        slot.material_id = resolved.map(|resolved| resolved.material_id);
        slot.batch_id = resolved.map(|resolved| resolved.batch_id);
        slot.seen_sync = sync_serial;
        self.records[slot_id as usize] = make_record(id.generation);
        assert!(self.lookup.insert(key, id).is_none());
        self.mark_dirty(slot_id);
        id
    }

    fn bump_instance_generation(
        &mut self,
        id: GpuSceneInstanceId,
        resolved: GpuSceneResolvedBin,
        revision: u64,
    ) -> GpuSceneInstanceId {
        let index = id.slot as usize;
        assert_eq!(self.slots[index].generation, id.generation);
        let key = self.slots[index]
            .key
            .expect("cannot bump an empty instance slot");
        let seen_sync = self.slots[index].seen_sync;
        let mut record = self.records[index];
        if id.generation != u32::MAX {
            let generation = id.generation + 1;
            self.slots[index].generation = generation;
            self.slots[index].geometry_id = Some(resolved.geometry_id);
            self.slots[index].material_id = Some(resolved.material_id);
            self.slots[index].batch_id = Some(resolved.batch_id);
            self.slot_generation_watermarks[index] = generation;
            let next = GpuSceneInstanceId {
                slot: id.slot,
                generation,
            };
            self.generation_exhausted |= generation == u32::MAX;
            *self.lookup.get_mut(&key).expect("instance lookup drift") = next;
            record.metadata[0] = generation;
            record.batch_id = [
                resolved.batch_id.slot(),
                resolved.batch_id.generation(),
                0,
                0,
            ];
            self.records[index] = record;
            self.mark_dirty(id.slot);
            return next;
        }

        self.slots[index].key = None;
        self.slots[index].geometry_id = None;
        self.slots[index].material_id = None;
        self.slots[index].batch_id = None;
        let removed = self.lookup.remove(&key).expect("instance lookup drift");
        assert_eq!(removed, id);
        self.records[index] = GpuSceneInstanceRecord::tombstone(u32::MAX);
        self.mark_dirty(id.slot);
        self.quarantined_slots.push(QuarantinedSlot {
            slot: id.slot,
            retired_revision: revision,
        });
        let next = self.allocate_instance_with(key, seen_sync, Some(resolved), |generation| {
            record.metadata[0] = generation;
            record.batch_id = [
                resolved.batch_id.slot(),
                resolved.batch_id.generation(),
                0,
                0,
            ];
            record
        });
        if let Some(domain) = self.domains.get_mut(&key.scene_id) {
            for slot_id in &mut domain.slots {
                if *slot_id == id.slot {
                    *slot_id = next.slot;
                }
            }
        }
        next
    }

    fn retire_slot(&mut self, slot_id: u32, retired_revision: u64) {
        let slot_index = slot_id as usize;
        let slot = &mut self.slots[slot_index];
        assert!(!slot.reusable, "GPU Scene attempted to retire a free slot");
        let key = slot
            .key
            .take()
            .expect("GPU Scene attempted to retire an empty slot");
        slot.geometry_id = None;
        slot.material_id = None;
        slot.batch_id = None;
        let removed = self
            .lookup
            .remove(&key)
            .expect("GPU Scene slot owner was absent from the lookup");
        assert_eq!(removed.slot, slot_id);
        assert_eq!(removed.generation, slot.generation);

        if let Some(next_generation) = slot.generation.checked_add(1) {
            slot.generation = next_generation;
            self.slot_generation_watermarks[slot_index] = next_generation;
        }
        self.records[slot_index] = GpuSceneInstanceRecord::tombstone(slot.generation);
        self.mark_dirty(slot_id);
        self.quarantined_slots.push(QuarantinedSlot {
            slot: slot_id,
            retired_revision,
        });
    }

    fn mark_dirty(&mut self, slot_id: u32) {
        let slot = &mut self.slots[slot_id as usize];
        if slot.dirty {
            return;
        }
        slot.dirty = true;
        self.dirty_slots.push(slot_id);
    }

    fn acknowledge_submitted_revision(&mut self, revision: u64) -> bool {
        let acknowledged = acknowledge_submitted_revision_parts(
            &mut self.slots,
            &mut self.dirty_slots,
            &mut self.quarantined_slots,
            &mut self.reusable_slots,
            self.revision,
            revision,
        );
        if acknowledged {
            self.dirty_ranges.clear();
        }
        acknowledged
    }

    fn rebuild_dirty_ranges(&mut self) {
        self.dirty_ranges.clear();
        self.dirty_slots.sort_unstable();
        let Some(&first) = self.dirty_slots.first() else {
            return;
        };

        let mut range_start = first;
        let mut range_end = first;
        for &slot in &self.dirty_slots[1..] {
            debug_assert!(
                slot > range_end,
                "GPU Scene dirty journal contains duplicates"
            );
            if slot == range_end + 1 {
                range_end = slot;
                continue;
            }
            self.dirty_ranges.push(GpuSceneDirtyRange {
                first_slot: range_start,
                slot_count: range_end - range_start + 1,
            });
            range_start = slot;
            range_end = slot;
        }
        self.dirty_ranges.push(GpuSceneDirtyRange {
            first_slot: range_start,
            slot_count: range_end - range_start + 1,
        });
    }

    /// Convert the externally supplied elapsed clock into a monotonic cache
    /// clock. A reset/regression merely establishes a new baseline; it never
    /// turns every retained domain into an accidentally ancient entry.
    fn observe_residency_time(&mut self, sample_seconds: f64) -> f64 {
        if !sample_seconds.is_finite() {
            return self.residency_clock_seconds;
        }
        if let Some(previous) = self.last_residency_sample_seconds {
            let elapsed = sample_seconds - previous;
            if elapsed > 0.0 {
                self.residency_clock_seconds += elapsed;
            }
        }
        self.last_residency_sample_seconds = Some(sample_seconds);
        self.residency_clock_seconds
    }

    /// Retire inactive domains as part of the selected scene's one logical
    /// transaction. The current scene is never a victim. Candidate order is
    /// deterministic even when several frames share one floating timestamp.
    fn sweep_inactive_domains(
        &mut self,
        current_scene_id: u32,
        now_seconds: f64,
        retired_revision: u64,
    ) -> u32 {
        self.domain_sweep_scratch.clear();
        self.domain_sweep_scratch.extend(
            self.domains
                .iter()
                .filter(|(scene_id, _)| **scene_id != current_scene_id)
                .map(|(&scene_id, domain)| DomainSweepCandidate {
                    scene_id,
                    last_used_seconds: domain.last_used_seconds,
                    last_used_sync: domain.last_used_sync,
                }),
        );
        self.domain_sweep_scratch.sort_unstable_by(|left, right| {
            left.last_used_seconds
                .total_cmp(&right.last_used_seconds)
                .then_with(|| left.last_used_sync.cmp(&right.last_used_sync))
                .then_with(|| left.scene_id.cmp(&right.scene_id))
        });

        let expired_count = self
            .domain_sweep_scratch
            .iter()
            .take_while(|candidate| {
                now_seconds - candidate.last_used_seconds >= GPU_SCENE_DOMAIN_IDLE_TTL_SECONDS
            })
            .count();
        let dormant_budget = GPU_SCENE_DOMAIN_BUDGET.saturating_sub(1);
        let budget_eviction_count = self
            .domain_sweep_scratch
            .len()
            .saturating_sub(dormant_budget);
        let eviction_count = expired_count.max(budget_eviction_count);

        let mut retired = 0u32;
        for candidate_index in 0..eviction_count {
            let scene_id = self.domain_sweep_scratch[candidate_index].scene_id;
            let domain = self
                .domains
                .remove(&scene_id)
                .expect("GPU Scene sweep candidate disappeared during its transaction");
            for slot_id in domain.slots {
                self.retire_slot(slot_id, retired_revision);
                retired = retired
                    .checked_add(1)
                    .expect("GPU Scene sweep retirement count overflowed");
            }
            for slot_id in domain.batch_slots {
                self.release_batch_slot(slot_id, retired_revision);
            }
        }
        self.domain_sweep_scratch.clear();
        retired
    }

    /// Remove only the contiguous suffix whose tombstones have already been
    /// proven submitted. Generation watermarks intentionally retain four
    /// bytes per historical index so later regrowth cannot produce an ABA ID.
    fn trim_acknowledged_trailing_slots(&mut self) -> bool {
        debug_assert_eq!(self.slots.len(), self.records.len());
        debug_assert!(self.slot_generation_watermarks.len() >= self.slots.len());

        let retained_len = self.acknowledged_tail_retained_len();
        for (index, slot) in self.slots[retained_len..].iter().enumerate() {
            debug_assert!(slot.key.is_none());
            debug_assert!(!slot.dirty);
            debug_assert_ne!(slot.generation, u32::MAX);
            debug_assert_eq!(
                self.slot_generation_watermarks[retained_len + index],
                slot.generation
            );
        }

        if retained_len == self.slots.len() {
            return false;
        }
        debug_assert!(
            self.dirty_slots
                .iter()
                .all(|slot| (*slot as usize) < retained_len)
        );
        debug_assert!(
            self.quarantined_slots
                .iter()
                .all(|slot| (slot.slot as usize) < retained_len)
        );
        debug_assert!(self.domains.values().all(|domain| {
            domain
                .slots
                .iter()
                .all(|slot| (*slot as usize) < retained_len)
        }));

        self.reusable_slots
            .retain(|slot| (*slot as usize) < retained_len);
        self.slots.truncate(retained_len);
        self.records.truncate(retained_len);
        self.dirty_ranges.clear();
        true
    }

    fn trim_all_acknowledged_tails(&mut self) -> bool {
        let instances = self.trim_acknowledged_trailing_slots();
        let geometries = self.geometries.trim_acknowledged_trailing_slots();
        let materials = self.materials.trim_acknowledged_trailing_slots();
        let batches = self.batches.trim_acknowledged_trailing_slots();
        instances || geometries || materials || batches
    }

    #[inline]
    fn acknowledged_tail_retained_len(&self) -> usize {
        self.slots
            .iter()
            .rposition(|slot| !slot.reusable)
            .map_or(0, |index| index + 1)
    }

    fn next_sync_serial(&mut self) -> u64 {
        self.sync_serial = self
            .sync_serial
            .checked_add(1)
            .expect("GPU Scene synchronization serial exhausted");
        self.sync_serial
    }
}

#[inline]
fn geometry_id(id: AuthorityId) -> GpuSceneGeometryId {
    GpuSceneGeometryId {
        slot: id.slot,
        generation: id.generation,
    }
}

#[inline]
fn material_id(id: AuthorityId) -> GpuSceneMaterialId {
    GpuSceneMaterialId {
        slot: id.slot,
        generation: id.generation,
    }
}

#[inline]
fn batch_id(id: AuthorityId) -> GpuSceneBatchId {
    GpuSceneBatchId {
        slot: id.slot,
        generation: id.generation,
    }
}

trait IntoAuthorityId {
    fn into_authority_id(self) -> AuthorityId;
}

macro_rules! impl_into_authority_id {
    ($name:ident) => {
        impl IntoAuthorityId for $name {
            fn into_authority_id(self) -> AuthorityId {
                AuthorityId {
                    slot: self.slot(),
                    generation: self.generation(),
                }
            }
        }
    };
}

impl_into_authority_id!(GpuSceneGeometryId);
impl_into_authority_id!(GpuSceneMaterialId);
impl_into_authority_id!(GpuSceneBatchId);

#[inline]
fn authority_id(id: impl IntoAuthorityId) -> AuthorityId {
    id.into_authority_id()
}

fn logical_table_bytes<R>(slot_span: u32) -> Option<u64> {
    u64::from(slot_span.max(1)).checked_mul(size_of::<R>() as u64)
}

fn table_spec<R>(record_count: usize, schema_id: u64) -> PersistentBufferSpec {
    let stride = size_of::<R>() as u64;
    let logical_size = u64::try_from(record_count.max(1))
        .expect("GPU Scene table record count exceeds u64")
        .checked_mul(stride)
        .expect("GPU Scene table byte size overflowed");
    PersistentBufferSpec::new(
        BufferDesc::new(logical_size, GPU_SCENE_INSTANCE_BUFFER_USAGE),
        schema_id,
        stride,
    )
}

fn projected_instance_span_for_keys(
    slots: &[InstanceSlot],
    lookup: &FxHashMap<GpuSceneInstanceKey, GpuSceneInstanceId>,
    retained_len: usize,
    keys: impl IntoIterator<Item = GpuSceneInstanceKey>,
) -> Option<u32> {
    let new_keys = keys
        .into_iter()
        .filter(|key| !lookup.contains_key(key))
        .count();
    let interior_reusable = slots[..retained_len]
        .iter()
        .filter(|slot| slot.reusable)
        .count();
    retained_len
        .checked_add(new_keys.saturating_sub(interior_reusable))
        .and_then(|span| u32::try_from(span).ok())
}

fn rebuild_dirty_ranges(dirty_slots: &mut [u32], dirty_ranges: &mut Vec<GpuSceneDirtyRange>) {
    dirty_ranges.clear();
    dirty_slots.sort_unstable();
    let Some(&first) = dirty_slots.first() else {
        return;
    };
    let mut start = first;
    let mut end = first;
    for &slot in &dirty_slots[1..] {
        debug_assert!(slot > end, "GPU Scene dirty journal contains duplicates");
        if slot == end + 1 {
            end = slot;
        } else {
            dirty_ranges.push(GpuSceneDirtyRange {
                first_slot: start,
                slot_count: end - start + 1,
            });
            start = slot;
            end = slot;
        }
    }
    dirty_ranges.push(GpuSceneDirtyRange {
        first_slot: start,
        slot_count: end - start + 1,
    });
}

fn acknowledge_authority_revision_parts<K>(
    slots: &mut [AuthoritySlot<K>],
    dirty_slots: &mut Vec<u32>,
    quarantined_slots: &mut Vec<QuarantinedSlot>,
    reusable_slots: &mut Vec<u32>,
    current_revision: u64,
    submitted_revision: u64,
) -> bool {
    if submitted_revision != current_revision {
        return false;
    }
    for slot in dirty_slots.drain(..) {
        slots[slot as usize].dirty = false;
    }
    let mut index = 0;
    while index < quarantined_slots.len() {
        if quarantined_slots[index].retired_revision > submitted_revision {
            index += 1;
            continue;
        }
        let retired = quarantined_slots.swap_remove(index);
        let slot = &mut slots[retired.slot as usize];
        debug_assert!(slot.key.is_none());
        if slot.generation != u32::MAX {
            debug_assert!(!slot.reusable);
            slot.reusable = true;
            reusable_slots.push(retired.slot);
        }
    }
    true
}

fn acknowledge_submitted_revision_parts(
    slots: &mut [InstanceSlot],
    dirty_slots: &mut Vec<u32>,
    quarantined_slots: &mut Vec<QuarantinedSlot>,
    reusable_slots: &mut Vec<u32>,
    current_revision: u64,
    submitted_revision: u64,
) -> bool {
    if submitted_revision != current_revision {
        return false;
    }
    for slot in dirty_slots.drain(..) {
        slots[slot as usize].dirty = false;
    }

    let mut quarantine_index = 0;
    while quarantine_index < quarantined_slots.len() {
        if quarantined_slots[quarantine_index].retired_revision > submitted_revision {
            quarantine_index += 1;
            continue;
        }
        let retired = quarantined_slots.swap_remove(quarantine_index);
        let slot = &mut slots[retired.slot as usize];
        debug_assert!(slot.key.is_none());
        if slot.generation != u32::MAX {
            debug_assert!(!slot.reusable);
            slot.reusable = true;
            reusable_slots.push(retired.slot);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{RenderView, WgpuContext};
    use crate::graph::core::PersistentBufferCommitRequest;
    use crate::graph::culling::collect_draw_intents_with_gpu_complement;
    use crate::graph::extracted::ExtractedLight;
    use crate::settings::{RendererInitConfig, RendererSettings};
    use glam::{Affine3A, Vec3};
    use myth_scene::camera::Camera;
    use myth_scene::light::{LightKind, PointLight, SpotLight};
    use slotmap::KeyData;

    fn gpu() -> WgpuContext {
        pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required")
    }

    fn node(value: u64) -> NodeHandle {
        NodeHandle::from(KeyData::from_ffi(value))
    }

    fn input(node_value: u64, translation: f32) -> GpuSceneInstanceInput {
        let world = Mat4::from_translation(glam::Vec3::new(translation, 0.0, 0.0));
        GpuSceneInstanceInput::new(
            node(node_value),
            world,
            world,
            BoundingBox {
                min: glam::Vec3::splat(translation - 1.0),
                max: glam::Vec3::splat(translation + 1.0),
            },
        )
    }

    fn geometry(value: u64) -> GeometryHandle {
        GeometryHandle::from(KeyData::from_ffi(value))
    }

    fn material(value: u64) -> MaterialHandle {
        MaterialHandle::from(KeyData::from_ffi(value))
    }

    fn rigid_item(node_value: u64, translation: f32) -> ExtractedRenderItem {
        let world = Mat4::from_translation(glam::Vec3::new(translation, 0.0, 0.0));
        let center = glam::Vec3::new(translation, 0.0, 0.0);
        ExtractedRenderItem {
            node_handle: node(node_value),
            world_matrix: world,
            prev_world_matrix: world,
            geometry: geometry(101),
            material: material(201),
            geometry_asset_version: 11,
            geometry_asset_identity: 12,
            geometry_structure_version: 13,
            geometry_layout_version: 14,
            geometry_topology: wgpu::PrimitiveTopology::TriangleList,
            geometry_index_format: Some(wgpu::IndexFormat::Uint32),
            geometry_index_count: 36,
            geometry_has_morph_targets: false,
            geometry_has_vertex_attributes: true,
            geometry_has_instanced_attributes: false,
            material_asset_version: 21,
            material_asset_identity: 22,
            material_version: 23,
            material_gpu_scene_compatible: true,
            material_index: 201,
            material_classification: MaterialClassification::empty(),
            skeleton: None,
            item_variant_flags: 0,
            item_shader_features: ItemShaderFeatures::RECEIVE_SHADOWS,
            cast_shadows: true,
            receive_shadows: true,
            world_aabb: BoundingBox {
                min: center - glam::Vec3::ONE,
                max: center + glam::Vec3::ONE,
            },
        }
    }

    fn rigid_item_with_assets(
        node_value: u64,
        geometry_value: u64,
        material_value: u64,
    ) -> ExtractedRenderItem {
        let mut item = rigid_item(node_value, node_value as f32);
        item.geometry = geometry(geometry_value);
        item.geometry_asset_identity = geometry_value + 10_000;
        item.material = material(material_value);
        item.material_asset_identity = material_value + 20_000;
        item.material_index = material_value as u32;
        item
    }

    fn extracted_scene(items: impl IntoIterator<Item = ExtractedRenderItem>) -> ExtractedScene {
        let mut scene = ExtractedScene::new();
        scene.scene_id = 17;
        scene.render_items.extend(items);
        scene
    }

    fn test_render_camera() -> myth_scene::camera::RenderCamera {
        let mut camera = Camera::new_perspective(60.0, 1.0, 0.1);
        camera.update_view_projection(&Affine3A::from_translation(Vec3::new(0.0, 0.0, 20.0)));
        camera.extract_render_camera()
    }

    fn shadow_light(id: u64, kind: LightKind) -> ExtractedLight {
        ExtractedLight {
            id,
            color: Vec3::ONE,
            intensity: 1.0,
            flags: 0,
            cast_shadows: true,
            kind,
            position: Vec3::ZERO,
            direction: -Vec3::Z,
            shadow: None,
        }
    }

    fn shadow_queue_sources(lists: &RenderLists, key: (u64, u32)) -> Vec<usize> {
        lists.shadow_queues[&key]
            .iter()
            .map(|intent| {
                lists
                    .extracted_source(intent.source)
                    .expect("shadow intent lost its extracted source")
                    .index()
            })
            .collect()
    }

    fn preflight_and_commit(
        authority: &mut GpuScene,
        scene: &ExtractedScene,
        elapsed_seconds: f64,
        lists: &mut RenderLists,
    ) -> GpuSceneMultiBinProjection {
        let preflight = authority
            .preflight_multi_bin_scene(scene, 16, 256)
            .unwrap()
            .expect("test scene must contain a GPU Scene domain");
        let projection = preflight.projection;
        authority
            .commit_multi_bin_selection(scene, preflight, elapsed_seconds, lists)
            .unwrap();
        projection
    }

    fn publish_table_set(
        authority: &mut GpuScene,
        registry: &mut PersistentBufferRegistry,
        gpu: &WgpuContext,
        attempt_serial: u64,
    ) -> GpuSceneTableSetPublicationProbe {
        let (plan, mut acknowledger) = authority
            .plan_table_set_with_acknowledger(registry, attempt_serial)
            .expect("coherent table-set plan");
        let persistent = [
            plan.instances.persistent,
            plan.geometries.persistent,
            plan.materials.persistent,
            plan.batches.persistent,
        ];
        let probe = plan.publication_probe;
        let mut commits = Vec::with_capacity(4);
        for plan in persistent {
            let physical = registry
                .materialize(&gpu.device, plan.buffer, Some("GPU Scene table-set test"))
                .unwrap();
            commits.push(PersistentBufferCommitRequest::new(
                plan.commit.expect("unpublished table-set member"),
                physical.token,
            ));
        }
        registry.commit_batch(&commits).unwrap();
        assert!(acknowledger.acknowledge_if_published(registry, &probe));
        probe
    }

    #[test]
    fn four_table_projection_is_read_only_and_two_by_two_ids_are_stable() {
        let scene = extracted_scene([
            rigid_item_with_assets(1, 10, 20),
            rigid_item_with_assets(2, 11, 20),
            rigid_item_with_assets(3, 10, 21),
            rigid_item_with_assets(4, 11, 21),
        ]);
        let mut authority = GpuScene::new();
        let mut lists = RenderLists::new();

        let preflight = authority
            .preflight_multi_bin_scene(&scene, 16, 256)
            .unwrap()
            .unwrap();
        assert_eq!(
            preflight.projection.table_set,
            GpuSceneProjectedTableSet {
                instance_slot_span: 4,
                geometry_slot_span: 2,
                material_slot_span: 2,
                batch_slot_span: 4,
                instance_logical_bytes: 4 * 256,
                geometry_logical_bytes: 2 * 48,
                material_logical_bytes: 2 * 32,
                batch_logical_bytes: 4 * 80,
                visible_view_stride: 4 * 256,
                visible_partition_bytes: 4 * 256,
                indirect_args_span: 4 * 20,
            }
        );
        assert_eq!(authority.revision, 0);
        assert!(authority.records.is_empty());
        assert!(authority.geometries.records.is_empty());
        assert!(authority.materials.records.is_empty());
        assert!(authority.batches.records.is_empty());
        assert!(authority.dirty_slots.is_empty());

        authority
            .commit_multi_bin_selection(&scene, preflight, 0.0, &mut lists)
            .unwrap();
        let geometry_ids = [
            geometry_id(authority.geometries.id(geometry(10)).unwrap()),
            geometry_id(authority.geometries.id(geometry(11)).unwrap()),
        ];
        let material_ids = [
            material_id(authority.materials.id(material(20)).unwrap()),
            material_id(authority.materials.id(material(21)).unwrap()),
        ];
        let batch_ids = authority
            .multi_bin_committed
            .iter()
            .map(|candidate| candidate.batch_id)
            .collect::<Vec<_>>();
        let revision = authority.revision;

        lists.begin_intent_collection();
        preflight_and_commit(&mut authority, &scene, 0.1, &mut lists);
        assert_eq!(authority.revision, revision);
        assert_eq!(
            geometry_ids,
            [
                geometry_id(authority.geometries.id(geometry(10)).unwrap()),
                geometry_id(authority.geometries.id(geometry(11)).unwrap()),
            ]
        );
        assert_eq!(
            material_ids,
            [
                material_id(authority.materials.id(material(20)).unwrap()),
                material_id(authority.materials.id(material(21)).unwrap()),
            ]
        );
        assert_eq!(
            batch_ids,
            authority
                .multi_bin_committed
                .iter()
                .map(|candidate| candidate.batch_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(authority.active_batch_descriptors.len(), 4);
        assert_eq!(
            authority.active_batch_descriptors[0].id,
            authority.multi_bin_committed[0].batch_id
        );
    }

    #[test]
    fn auto_cohort_projection_and_cpu_complement_partition_sources_exactly_once() {
        let shared = [
            rigid_item_with_assets(1, 10, 20),
            rigid_item_with_assets(2, 10, 20),
            rigid_item_with_assets(3, 10, 20),
        ];
        let singleton = rigid_item_with_assets(4, 11, 21);
        let mut unsupported = rigid_item_with_assets(5, 12, 22);
        unsupported.geometry_has_morph_targets = true;
        let scene = extracted_scene(shared.into_iter().chain([singleton, unsupported]));
        let mut authority = GpuScene::new();
        let mut lists = RenderLists::new();
        let preflight = authority
            .preflight_multi_bin_cohorts(&scene, 16, 256, 2)
            .unwrap()
            .unwrap();
        assert_eq!(preflight.projection.candidate_count, 3);
        assert_eq!(preflight.projection.bin_count, 1);
        assert_eq!(preflight.projection.unique_geometry_count, 1);
        assert_eq!(preflight.projection.unique_material_count, 1);
        assert_eq!(preflight.projection.table_set.instance_slot_span, 3);
        assert_eq!(preflight.projection.shadow_candidate_count, 3);
        assert!(!preflight.projection.covers_complete_shadow_domain);
        assert_eq!(authority.multi_bin_members.len(), 3);
        assert_eq!(authority.multi_bin_preflight[0].member_count, 3);
        assert_eq!(authority.revision, 0, "preflight must remain read-only");
        authority
            .commit_multi_bin_selection_with_families(
                &scene,
                preflight,
                GpuDrivenFamilyMask::MAIN_PREPASS,
                0.0,
                &mut lists,
            )
            .unwrap();

        collect_draw_intents_with_gpu_complement(
            &scene,
            &mut lists,
            &test_render_camera(),
            GpuDrivenFamilyMask::MAIN_PREPASS,
            GpuDrivenFamilyMask::empty(),
            false,
        );

        let cpu_sources = lists
            .opaque
            .iter()
            .map(|intent| lists.extracted_source(intent.source).unwrap().index())
            .collect::<Vec<_>>();
        assert_eq!(cpu_sources, [3, 4]);
        for index in 0..scene.render_items.len() {
            let owned = lists
                .gpu_family_ownership(ExtractedSourceId::from_index(index))
                .contains(GpuDrivenFamilyMask::MAIN_PREPASS);
            let cpu = cpu_sources.contains(&index);
            assert_ne!(
                owned, cpu,
                "each main opaque source must be owned by exactly one backend"
            );
        }
        for node_value in 1..=3 {
            let id = authority
                .instance_id(scene.scene_id, node(node_value))
                .unwrap();
            assert_eq!(
                authority.record(id).unwrap().metadata[3],
                GPU_SCENE_INSTANCE_CASTS_SHADOW | GPU_SCENE_INSTANCE_MAIN_PREPASS
            );
        }
        assert!(authority.instance_id(scene.scene_id, node(4)).is_none());
        assert!(authority.instance_id(scene.scene_id, node(5)).is_none());
    }

    #[test]
    fn shadow_2d_and_cube_family_ownership_filter_only_the_selected_family() {
        let mut supported_a = rigid_item_with_assets(1, 10, 20);
        let mut supported_b = rigid_item_with_assets(2, 10, 20);
        let mut unsupported = rigid_item_with_assets(3, 11, 21);
        unsupported.geometry_has_morph_targets = true;
        for item in [&mut supported_a, &mut supported_b, &mut unsupported] {
            item.world_matrix = Mat4::IDENTITY;
            item.prev_world_matrix = Mat4::IDENTITY;
            item.world_aabb = BoundingBox {
                min: -Vec3::ONE,
                max: Vec3::ONE,
            };
        }
        let mut scene = extracted_scene([supported_a, supported_b, unsupported]);
        scene.lights.push(shadow_light(
            101,
            LightKind::Spot(SpotLight {
                range: 20.0,
                inner_cone: 0.4,
                outer_cone: 0.8,
            }),
        ));
        scene.lights.push(shadow_light(
            202,
            LightKind::Point(PointLight { range: 20.0 }),
        ));

        let shadow_frustum = myth_scene::camera::Frustum::from_matrix_standard_z(Mat4::IDENTITY);
        let mut lists = RenderLists::new();
        lists.active_views.push(RenderView::new_shadow(
            101,
            0,
            0,
            "Shadow2D".into(),
            Mat4::IDENTITY,
            shadow_frustum,
            (64, 64),
            None,
        ));
        lists.active_views.push(RenderView::new_shadow(
            202,
            0,
            1,
            "ShadowCube".into(),
            Mat4::IDENTITY,
            shadow_frustum,
            (64, 64),
            None,
        ));

        let mut authority = GpuScene::new();
        let d2 = authority
            .preflight_multi_bin_cohorts(&scene, 16, 256, 2)
            .unwrap()
            .unwrap();
        authority
            .commit_multi_bin_selection_with_families(
                &scene,
                d2,
                GpuDrivenFamilyMask::SHADOW_2D,
                0.0,
                &mut lists,
            )
            .unwrap();
        collect_draw_intents_with_gpu_complement(
            &scene,
            &mut lists,
            &test_render_camera(),
            GpuDrivenFamilyMask::SHADOW_2D,
            GpuDrivenFamilyMask::empty(),
            false,
        );
        assert_eq!(shadow_queue_sources(&lists, (101, 0)), [2]);
        assert_eq!(shadow_queue_sources(&lists, (202, 0)), [0, 1, 2]);
        let first = authority.instance_id(scene.scene_id, node(1)).unwrap();
        assert_eq!(
            authority.record(first).unwrap().metadata[3],
            GPU_SCENE_INSTANCE_CASTS_SHADOW | GPU_SCENE_INSTANCE_SHADOW_2D
        );

        lists.begin_intent_collection();
        let cube = authority
            .preflight_multi_bin_cohorts(&scene, 16, 256, 2)
            .unwrap()
            .unwrap();
        authority
            .commit_multi_bin_selection_with_families(
                &scene,
                cube,
                GpuDrivenFamilyMask::SHADOW_CUBE,
                0.1,
                &mut lists,
            )
            .unwrap();
        collect_draw_intents_with_gpu_complement(
            &scene,
            &mut lists,
            &test_render_camera(),
            GpuDrivenFamilyMask::SHADOW_CUBE,
            GpuDrivenFamilyMask::empty(),
            false,
        );
        assert_eq!(shadow_queue_sources(&lists, (101, 0)), [0, 1, 2]);
        assert_eq!(shadow_queue_sources(&lists, (202, 0)), [2]);
        let second = authority.instance_id(scene.scene_id, node(1)).unwrap();
        assert_eq!(
            authority.record(second).unwrap().metadata[3],
            GPU_SCENE_INSTANCE_CASTS_SHADOW | GPU_SCENE_INSTANCE_SHADOW_CUBE
        );
    }

    #[test]
    fn asset_epoch_bump_cascades_batch_and_instance_generation_in_place() {
        let scene = extracted_scene([rigid_item_with_assets(1, 10, 20)]);
        let mut authority = GpuScene::new();
        let mut lists = RenderLists::new();
        preflight_and_commit(&mut authority, &scene, 0.0, &mut lists);
        let geometry_before = geometry_id(authority.geometries.id(geometry(10)).unwrap());
        let batch_before = authority.multi_bin_committed[0].batch_id;
        let instance_before = authority.instance_id(17, node(1)).unwrap();

        let mut changed = scene.render_items[0].clone();
        changed.geometry_asset_identity += 1;
        changed.geometry_asset_version += 1;
        changed.geometry_structure_version += 1;
        let changed_scene = extracted_scene([changed]);
        lists.begin_intent_collection();
        preflight_and_commit(&mut authority, &changed_scene, 0.1, &mut lists);

        let geometry_after = geometry_id(authority.geometries.id(geometry(10)).unwrap());
        let batch_after = authority.multi_bin_committed[0].batch_id;
        let instance_after = authority.instance_id(17, node(1)).unwrap();
        assert_eq!(geometry_after.slot(), geometry_before.slot());
        assert_eq!(
            geometry_after.generation(),
            geometry_before.generation() + 1
        );
        assert_eq!(batch_after.slot(), batch_before.slot());
        assert_eq!(batch_after.generation(), batch_before.generation() + 1);
        assert_eq!(instance_after.slot(), instance_before.slot());
        assert_eq!(
            instance_after.generation(),
            instance_before.generation() + 1
        );
        assert_eq!(
            authority.record(instance_after).unwrap().batch_id[..2],
            batch_after.words()
        );
        assert_eq!(authority.revision, 2, "all four tables share one revision");
    }

    #[test]
    fn same_handle_stamp_conflicts_fail_before_any_authority_mutation() {
        let cases: &[(
            &str,
            GpuSceneSelectionRejection,
            fn(&mut ExtractedRenderItem),
        )] = &[
            (
                "geometry asset version",
                GpuSceneSelectionRejection::ConflictingGeometryEpoch,
                |item| item.geometry_asset_version += 1,
            ),
            (
                "geometry asset identity",
                GpuSceneSelectionRejection::ConflictingGeometryEpoch,
                |item| item.geometry_asset_identity += 1,
            ),
            (
                "geometry structure version",
                GpuSceneSelectionRejection::ConflictingGeometryEpoch,
                |item| item.geometry_structure_version += 1,
            ),
            (
                "geometry layout version",
                GpuSceneSelectionRejection::ConflictingGeometryEpoch,
                |item| item.geometry_layout_version += 1,
            ),
            (
                "index format",
                GpuSceneSelectionRejection::ConflictingGeometryEpoch,
                |item| item.geometry_index_format = Some(wgpu::IndexFormat::Uint16),
            ),
            (
                "index count",
                GpuSceneSelectionRejection::ConflictingGeometryEpoch,
                |item| item.geometry_index_count += 3,
            ),
            (
                "material asset version",
                GpuSceneSelectionRejection::ConflictingMaterialEpoch,
                |item| item.material_asset_version += 1,
            ),
            (
                "material asset identity",
                GpuSceneSelectionRejection::ConflictingMaterialEpoch,
                |item| item.material_asset_identity += 1,
            ),
            (
                "material version",
                GpuSceneSelectionRejection::ConflictingMaterialEpoch,
                |item| item.material_version += 1,
            ),
            (
                "material index",
                GpuSceneSelectionRejection::ConflictingMaterialEpoch,
                |item| item.material_index += 1,
            ),
        ];

        for &(label, expected, mutate) in cases {
            let first = rigid_item_with_assets(1, 10, 20);
            let mut second = rigid_item_with_assets(2, 10, 20);
            mutate(&mut second);
            let scene = extracted_scene([first, second]);
            let mut authority = GpuScene::new();

            assert_eq!(
                authority.preflight_multi_bin_scene(&scene, 16, 256),
                Err(GpuSceneMultiBinPreflightFailure::Domain(expected)),
                "same-handle stamp conflict was misclassified: {label}"
            );
            assert_eq!(authority.revision, 0, "{label}");
            assert_eq!(authority.slot_span(), 0, "{label}");
            assert!(authority.records.is_empty(), "{label}");
            assert!(authority.geometries.records.is_empty(), "{label}");
            assert!(authority.materials.records.is_empty(), "{label}");
            assert!(authority.batches.records.is_empty(), "{label}");
            assert!(authority.multi_bin_committed.is_empty(), "{label}");
            assert!(authority.active_batch_descriptors.is_empty(), "{label}");
        }
    }

    #[test]
    fn coherent_table_set_is_all_invalid_or_all_valid_and_trim_advances_revision() {
        let gpu = gpu();
        let scene = extracted_scene([rigid_item_with_assets(1, 10, 20)]);
        let mut authority = GpuScene::new();
        let mut lists = RenderLists::new();
        let mut registry = PersistentBufferRegistry::new();
        preflight_and_commit(&mut authority, &scene, 0.0, &mut lists);

        {
            let plan = authority.plan_table_set(&mut registry, 1).unwrap();
            let members = [
                plan.instances.persistent,
                plan.geometries.persistent,
                plan.materials.persistent,
                plan.batches.persistent,
            ];
            assert!(members.iter().all(|member| !member.valid));
            assert!(members.iter().all(|member| member.commit.is_some()));
            assert_eq!(plan.instances.persistent.buffer.revision, plan.revision);
            assert_eq!(plan.geometries.persistent.buffer.revision, plan.revision);
            assert_eq!(plan.materials.persistent.buffer.revision, plan.revision);
            assert_eq!(plan.batches.persistent.buffer.revision, plan.revision);
            assert_eq!(plan.candidates.len(), 1);
            assert_eq!(plan.active_batches.len(), 1);
        }
        publish_table_set(&mut authority, &mut registry, &gpu, 1);
        {
            let plan = authority.plan_table_set(&mut registry, 2).unwrap();
            assert!(
                [
                    plan.instances.persistent,
                    plan.geometries.persistent,
                    plan.materials.persistent,
                    plan.batches.persistent,
                ]
                .iter()
                .all(|member| member.valid && member.commit.is_none())
            );
            assert!(plan.instances.dirty_ranges.is_empty());
            assert!(plan.geometries.dirty_ranges.is_empty());
            assert!(plan.materials.dirty_ranges.is_empty());
            assert!(plan.batches.dirty_ranges.is_empty());
        }

        authority.retire_scene(17);
        publish_table_set(&mut authority, &mut registry, &gpu, 3);
        let semantic_revision = authority.revision;
        let plan = authority.plan_table_set(&mut registry, 4).unwrap();
        assert_eq!(plan.revision, semantic_revision + 1);
        assert_eq!(plan.instances.slot_span, 0);
        assert_eq!(plan.geometries.slot_span, 0);
        assert_eq!(plan.materials.slot_span, 0);
        assert_eq!(plan.batches.slot_span, 0);
        assert_eq!(plan.instances.records.len(), 1);
        assert_eq!(plan.instances.records[0].metadata[1], 0);
        assert_eq!(plan.geometries.records.len(), 1);
        assert_eq!(plan.materials.records.len(), 1);
        assert_eq!(plan.batches.records.len(), 1);
        assert!(
            [
                plan.instances.persistent,
                plan.geometries.persistent,
                plan.materials.persistent,
                plan.batches.persistent,
            ]
            .iter()
            .all(|member| !member.valid && member.commit.is_some())
        );
    }

    #[test]
    fn four_table_quarantine_replays_then_reuses_without_aba() {
        let gpu = gpu();
        let first_scene = extracted_scene([rigid_item_with_assets(1, 10, 20)]);
        let mut authority = GpuScene::new();
        let mut lists = RenderLists::new();
        let mut registry = PersistentBufferRegistry::new();
        preflight_and_commit(&mut authority, &first_scene, 0.0, &mut lists);
        let first_instance = authority.instance_id(17, node(1)).unwrap();
        let first_geometry = geometry_id(authority.geometries.id(geometry(10)).unwrap());
        let first_material = material_id(authority.materials.id(material(20)).unwrap());
        let first_batch = authority.multi_bin_committed[0].batch_id;
        publish_table_set(&mut authority, &mut registry, &gpu, 1);

        authority.retire_scene(17);
        assert_eq!(authority.quarantined_slots.len(), 1);
        assert_eq!(authority.geometries.quarantined_slots.len(), 1);
        assert_eq!(authority.materials.quarantined_slots.len(), 1);
        assert_eq!(authority.batches.quarantined_slots.len(), 1);

        let second_scene = extracted_scene([rigid_item_with_assets(2, 11, 21)]);
        lists.begin_intent_collection();
        preflight_and_commit(&mut authority, &second_scene, 0.1, &mut lists);
        assert_ne!(
            authority.instance_id(17, node(2)).unwrap().slot(),
            first_instance.slot()
        );
        assert_ne!(
            authority.geometries.id(geometry(11)).unwrap().slot,
            first_geometry.slot()
        );
        assert_ne!(
            authority.materials.id(material(21)).unwrap().slot,
            first_material.slot()
        );
        assert_ne!(
            authority.multi_bin_committed[0].batch_id.slot(),
            first_batch.slot()
        );
        publish_table_set(&mut authority, &mut registry, &gpu, 2);

        let third = rigid_item_with_assets(3, 12, 22);
        let expanded = extracted_scene([rigid_item_with_assets(2, 11, 21), third]);
        lists.begin_intent_collection();
        preflight_and_commit(&mut authority, &expanded, 0.2, &mut lists);
        let reused_instance = authority.instance_id(17, node(3)).unwrap();
        let reused_geometry = geometry_id(authority.geometries.id(geometry(12)).unwrap());
        let reused_material = material_id(authority.materials.id(material(22)).unwrap());
        let reused_batch = authority.multi_bin_committed[1].batch_id;
        assert_eq!(reused_instance.slot(), first_instance.slot());
        assert_eq!(
            reused_instance.generation(),
            first_instance.generation() + 1
        );
        assert_eq!(reused_geometry.slot(), first_geometry.slot());
        assert_eq!(
            reused_geometry.generation(),
            first_geometry.generation() + 1
        );
        assert_eq!(reused_material.slot(), first_material.slot());
        assert_eq!(
            reused_material.generation(),
            first_material.generation() + 1
        );
        assert_eq!(reused_batch.slot(), first_batch.slot());
        assert_eq!(reused_batch.generation(), first_batch.generation() + 1);
        assert!(authority.record(first_instance).is_none());
    }

    #[test]
    fn saturated_generations_fail_closed_and_never_wrap_or_alias() {
        let scene = extracted_scene([rigid_item_with_assets(1, 10, 20)]);
        let mut authority = GpuScene::new();
        let mut lists = RenderLists::new();
        preflight_and_commit(&mut authority, &scene, 0.0, &mut lists);
        let original = authority.geometries.id(geometry(10)).unwrap();
        let index = original.slot as usize;
        authority.geometries.slots[index].generation = u32::MAX;
        authority.geometries.generation_watermarks[index] = u32::MAX;
        authority.geometries.records[index].metadata[0] = u32::MAX;
        *authority.geometries.lookup.get_mut(&geometry(10)).unwrap() = AuthorityId {
            slot: original.slot,
            generation: u32::MAX,
        };
        authority.generation_exhausted = true;

        let mut changed = scene.render_items[0].clone();
        changed.geometry_asset_identity += 1;
        let changed_scene = extracted_scene([changed]);
        lists.begin_intent_collection();
        let revision = authority.revision;
        assert_eq!(
            authority.preflight_multi_bin_scene(&changed_scene, 16, 256),
            Err(GpuSceneMultiBinPreflightFailure::Domain(
                GpuSceneSelectionRejection::GenerationExhausted
            ))
        );
        assert_eq!(authority.revision, revision);
        assert_eq!(
            authority.geometries.id(geometry(10)),
            Some(AuthorityId {
                slot: original.slot,
                generation: u32::MAX,
            })
        );
        assert_eq!(authority.geometries.slots[index].generation, u32::MAX);
        assert!(authority.geometries.quarantined_slots.is_empty());
        assert_eq!(lists.frame_source_count(), 0);

        let mut retiring = GpuScene::new();
        retiring.synchronize_scene(6, [input(1, 0.0)]);
        let first = retiring.instance_id(6, node(1)).unwrap();
        assert!(retiring.acknowledge_submitted_revision(1));
        retiring.retire_scene(6);
        assert!(retiring.acknowledge_submitted_revision(2));
        retiring.slots[first.slot() as usize].generation = u32::MAX - 1;
        retiring.slot_generation_watermarks[first.slot() as usize] = u32::MAX - 1;
        retiring.synchronize_scene(6, [input(2, 0.0)]);
        let penultimate = retiring.instance_id(6, node(2)).unwrap();
        assert_eq!(
            (penultimate.slot(), penultimate.generation()),
            (first.slot(), u32::MAX - 1)
        );
        assert!(retiring.acknowledge_submitted_revision(3));
        retiring.retire_scene(6);
        assert!(retiring.acknowledge_submitted_revision(4));
        assert!(!retiring.reusable_slots.contains(&first.slot()));
        retiring.synchronize_scene(6, [input(3, 0.0)]);
        assert_ne!(
            retiring.instance_id(6, node(3)).unwrap().slot(),
            first.slot()
        );
        assert_eq!(retiring.slots[first.slot() as usize].generation, u32::MAX);
    }

    #[test]
    fn multi_bin_preflight_groups_exact_keys_in_first_occurrence_order() {
        let scene = extracted_scene([
            rigid_item_with_assets(1, 10, 20),
            rigid_item_with_assets(2, 11, 20),
            rigid_item_with_assets(3, 10, 21),
            rigid_item_with_assets(4, 11, 21),
            rigid_item_with_assets(5, 10, 20),
        ]);
        let mut authority = GpuScene::new();
        let mut lists = RenderLists::new();

        let preflight = authority
            .preflight_multi_bin_scene(&scene, 16, 256)
            .unwrap()
            .unwrap();
        assert_eq!(preflight.projection.candidate_count, 5);
        assert_eq!(preflight.projection.bin_count, 4);
        assert_eq!(preflight.projection.unique_geometry_count, 2);
        assert_eq!(preflight.projection.unique_material_count, 2);
        assert_eq!(preflight.projection.visible_partition_bytes, 4 * 256);
        assert_eq!(authority.revision, 0);
        assert_eq!(authority.slot_span(), 0);
        assert_eq!(lists.frame_source_count(), 0);
        assert!(lists.gpu_opaque_representatives().next().is_none());

        let committed = authority
            .commit_multi_bin_selection(&scene, preflight, 0.0, &mut lists)
            .unwrap();
        assert_eq!(
            committed
                .iter()
                .map(|bin| (bin.key.geometry, bin.key.material, bin.member_count))
                .collect::<Vec<_>>(),
            [
                (geometry(10), material(20), 2),
                (geometry(11), material(20), 1),
                (geometry(10), material(21), 1),
                (geometry(11), material(21), 1),
            ]
        );
        assert_eq!(
            committed
                .iter()
                .map(|bin| (bin.partition_base, bin.partition_size))
                .collect::<Vec<_>>(),
            [(0, 256), (256, 256), (512, 256), (768, 256)]
        );
        assert_eq!(
            lists
                .gpu_opaque_representatives()
                .map(|source| lists.extracted_source(source).unwrap().index())
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }

    #[test]
    fn supported_batch_identity_drift_partitions_into_distinct_bins() {
        let cases: &[(&str, u32, u32, fn(&mut ExtractedRenderItem))] = &[
            ("geometry handle", 2, 1, |item| item.geometry = geometry(11)),
            ("material handle", 1, 2, |item| item.material = material(21)),
            ("item variant flags", 1, 1, |item| {
                item.item_variant_flags ^= 1
            }),
            ("item shader features", 1, 1, |item| {
                item.item_shader_features = ItemShaderFeatures::empty()
            }),
        ];

        for &(label, geometry_count, material_count, mutate) in cases {
            let first = rigid_item_with_assets(1, 10, 20);
            let mut second = rigid_item_with_assets(2, 10, 20);
            mutate(&mut second);
            let scene = extracted_scene([first, second]);
            let mut authority = GpuScene::new();

            let projection = authority
                .preflight_multi_bin_scene(&scene, 16, 256)
                .unwrap_or_else(|error| panic!("supported drift was rejected ({label}): {error:?}"))
                .expect("two supported candidates must produce a projection")
                .projection;
            assert_eq!(projection.candidate_count, 2, "{label}");
            assert_eq!(projection.bin_count, 2, "{label}");
            assert_eq!(projection.unique_geometry_count, geometry_count, "{label}");
            assert_eq!(projection.unique_material_count, material_count, "{label}");
            assert_eq!(
                authority
                    .multi_bin_preflight
                    .iter()
                    .map(|bin| bin.representative.index())
                    .collect::<Vec<_>>(),
                [0, 1],
                "{label}"
            );
            assert_eq!(authority.revision, 0, "{label}");
            assert_eq!(authority.slot_span(), 0, "{label}");
            assert!(authority.records.is_empty(), "{label}");
            assert!(authority.geometries.records.is_empty(), "{label}");
            assert!(authority.materials.records.is_empty(), "{label}");
            assert!(authority.batches.records.is_empty(), "{label}");
        }
    }

    #[test]
    fn multi_bin_commit_rejects_a_different_extracted_scene_with_the_same_id() {
        let item = rigid_item_with_assets(1, 10, 20);
        let scene = extracted_scene([item.clone()]);
        let impostor = extracted_scene([item]);
        let mut authority = GpuScene::new();
        let mut lists = RenderLists::new();
        let preflight = authority
            .preflight_multi_bin_scene(&scene, 16, 256)
            .unwrap()
            .unwrap();

        assert_eq!(
            authority.commit_multi_bin_selection(&impostor, preflight, 0.0, &mut lists),
            Err(GpuSceneMultiBinCommitFailure::SceneDrift)
        );
        assert_eq!(authority.revision, 0);
        assert_eq!(authority.slot_span(), 0);
        assert_eq!(lists.frame_source_count(), 0);
    }

    #[test]
    fn multi_bin_inverse_recipe_is_sparse_and_only_tracks_world_matrix_changes() {
        let initial = extracted_scene([
            rigid_item_with_assets(1, 10, 20),
            rigid_item_with_assets(2, 11, 21),
        ]);
        let mut authority = GpuScene::new();
        let mut lists = RenderLists::new();
        let cold = authority
            .preflight_multi_bin_scene(&initial, 16, 256)
            .unwrap()
            .unwrap();
        assert_eq!(authority.multi_bin_inverse_updates.len(), 2);
        authority
            .commit_multi_bin_selection(&initial, cold, 0.0, &mut lists)
            .unwrap();

        let mut motion_only = rigid_item_with_assets(1, 10, 20);
        motion_only.prev_world_matrix = Mat4::from_translation(glam::Vec3::splat(-7.0));
        let motion_scene = extracted_scene([motion_only, rigid_item_with_assets(2, 11, 21)]);
        lists.begin_intent_collection();
        let motion = authority
            .preflight_multi_bin_scene(&motion_scene, 16, 256)
            .unwrap()
            .unwrap();
        assert!(
            authority.multi_bin_inverse_updates.is_empty(),
            "previous-transform changes do not invalidate the current inverse"
        );
        authority
            .commit_multi_bin_selection(&motion_scene, motion, 0.1, &mut lists)
            .unwrap();

        let mut moved = rigid_item_with_assets(1, 10, 20);
        moved.world_matrix = Mat4::from_scale_rotation_translation(
            glam::Vec3::new(2.0, 3.0, 4.0),
            glam::Quat::from_rotation_y(0.4),
            glam::Vec3::new(9.0, 2.0, -3.0),
        );
        let expected_inverse = moved.world_matrix.inverse().to_cols_array_2d();
        let moved_scene = extracted_scene([moved, rigid_item_with_assets(2, 11, 21)]);
        lists.begin_intent_collection();
        let moved_preflight = authority
            .preflight_multi_bin_scene(&moved_scene, 16, 256)
            .unwrap()
            .unwrap();
        assert_eq!(
            authority.multi_bin_inverse_updates,
            [expected_inverse],
            "only the one transform-dirty instance derives an inverse"
        );
        authority
            .commit_multi_bin_selection(&moved_scene, moved_preflight, 0.2, &mut lists)
            .unwrap();
        let moved_id = authority.instance_id(17, node(1)).unwrap();
        assert_eq!(
            authority.records[moved_id.slot() as usize].world_matrix_inverse,
            expected_inverse
        );
    }

    #[test]
    fn multi_bin_partition_layout_is_checked_aligned_and_linear() {
        let mut items = Vec::new();
        let counts = [1u32, 3, 17, 33];
        let mut node_id = 1u64;
        for (bin, count) in counts.into_iter().enumerate() {
            for _ in 0..count {
                items.push(rigid_item_with_assets(
                    node_id,
                    100 + bin as u64,
                    200 + bin as u64,
                ));
                node_id += 1;
            }
        }
        let scene = extracted_scene(items);
        let mut authority = GpuScene::new();
        let preflight = authority
            .preflight_multi_bin_scene(&scene, 24, 64)
            .unwrap()
            .unwrap();
        let bins = &authority.multi_bin_preflight;
        assert_eq!(bins.len(), counts.len());
        let quantum = checked_lcm(24, 64).unwrap();
        let mut previous_end = 0u64;
        for bin in bins {
            assert_eq!(bin.partition_base, previous_end);
            assert_eq!(bin.partition_base % 64, 0);
            assert_eq!(bin.partition_size % 64, 0);
            assert_eq!(bin.partition_base % 24, 0);
            assert_eq!(bin.partition_size % 24, 0);
            assert!(bin.partition_size >= u64::from(bin.member_count) * 24);
            previous_end = bin.partition_base + bin.partition_size;
        }
        let raw_bytes = u64::from(preflight.projection.candidate_count) * 24;
        assert_eq!(previous_end, preflight.projection.visible_partition_bytes);
        assert!(previous_end < raw_bytes + bins.len() as u64 * quantum);

        assert_eq!(
            authority.preflight_multi_bin_scene(&scene, 24, 4),
            Err(GpuSceneMultiBinPreflightFailure::InvalidStorageAlignment)
        );
        assert_eq!(
            authority.preflight_multi_bin_scene(&scene, u64::MAX, 8),
            Err(GpuSceneMultiBinPreflightFailure::LayoutOverflow)
        );
    }

    #[test]
    fn strict_multi_bin_preflight_rejects_every_unsupported_shape_before_publication() {
        let cases: &[(
            &str,
            GpuSceneSelectionRejection,
            fn(&mut ExtractedRenderItem),
        )] = &[
            ("skinned", GpuSceneSelectionRejection::Skinned, |item| {
                item.skeleton = Some(myth_scene::SkeletonKey::from(KeyData::from_ffi(91)));
                item.item_shader_features
                    .insert(ItemShaderFeatures::HAS_SKINNING);
            }),
            (
                "morph targets",
                GpuSceneSelectionRejection::MorphTargets,
                |item| {
                    item.geometry_has_morph_targets = true;
                },
            ),
            (
                "native instance attributes",
                GpuSceneSelectionRejection::InstancedVertexAttributes,
                |item| item.geometry_has_instanced_attributes = true,
            ),
            (
                "custom material",
                GpuSceneSelectionRejection::IncompatibleMaterial,
                |item| item.material_gpu_scene_compatible = false,
            ),
            (
                "non-indexed",
                GpuSceneSelectionRejection::NonIndexed,
                |item| {
                    item.geometry_index_format = None;
                },
            ),
            (
                "empty index buffer",
                GpuSceneSelectionRejection::EmptyIndexBuffer,
                |item| item.geometry_index_count = 0,
            ),
            (
                "empty vertex buffers",
                GpuSceneSelectionRejection::EmptyVertexBuffers,
                |item| item.geometry_has_vertex_attributes = false,
            ),
            (
                "unsupported topology",
                GpuSceneSelectionRejection::UnsupportedTopology,
                |item| item.geometry_topology = wgpu::PrimitiveTopology::LineList,
            ),
            (
                "non-finite current transform",
                GpuSceneSelectionRejection::NonFiniteTransform,
                |item| {
                    item.world_matrix = Mat4::from_translation(glam::Vec3::new(f32::NAN, 0.0, 0.0));
                },
            ),
            (
                "non-finite previous transform",
                GpuSceneSelectionRejection::NonFiniteTransform,
                |item| {
                    item.prev_world_matrix =
                        Mat4::from_translation(glam::Vec3::new(0.0, f32::INFINITY, 0.0));
                },
            ),
            (
                "non-invertible transform",
                GpuSceneSelectionRejection::NonInvertibleTransform,
                |item| item.world_matrix = Mat4::from_scale(glam::Vec3::new(0.0, 1.0, 1.0)),
            ),
            (
                "non-finite bounds",
                GpuSceneSelectionRejection::NonFiniteBounds,
                |item| item.world_aabb = BoundingBox::infinite(),
            ),
            (
                "inverted bounds",
                GpuSceneSelectionRejection::NonFiniteBounds,
                |item| {
                    item.world_aabb = BoundingBox {
                        min: glam::Vec3::ONE,
                        max: -glam::Vec3::ONE,
                    };
                },
            ),
        ];

        for &(label, expected, mutate) in cases {
            let mut unsupported = rigid_item_with_assets(2, 11, 21);
            mutate(&mut unsupported);
            let scene = extracted_scene([rigid_item_with_assets(1, 10, 20), unsupported]);
            let mut authority = GpuScene::new();

            assert_eq!(
                authority.preflight_multi_bin_scene(&scene, 16, 256),
                Err(GpuSceneMultiBinPreflightFailure::Domain(expected)),
                "unsupported shape produced the wrong rejection: {label}"
            );
            assert_eq!(authority.revision, 0, "{label}");
            assert_eq!(authority.slot_span(), 0, "{label}");
            assert!(authority.records.is_empty(), "{label}");
            assert!(authority.geometries.records.is_empty(), "{label}");
            assert!(authority.materials.records.is_empty(), "{label}");
            assert!(authority.batches.records.is_empty(), "{label}");
        }
    }

    #[test]
    fn multi_bin_preflight_selects_opaque_domain_and_reports_shadow_completeness() {
        let mut transparent = rigid_item(3, 30.0);
        transparent
            .material_classification
            .insert(MaterialClassification::TRANSPARENT);
        let mut transmission = rigid_item(4, 40.0);
        transmission
            .material_classification
            .insert(MaterialClassification::USES_TRANSMISSION);
        let scene = extracted_scene([
            rigid_item(1, -1000.0),
            rigid_item(2, 1000.0),
            transparent,
            transmission,
        ]);
        let mut authority = GpuScene::new();

        let projection = authority
            .preflight_multi_bin_scene(&scene, 16, 256)
            .unwrap()
            .expect("the opaque pre-frustum domain must not be empty")
            .projection;
        assert_eq!(projection.candidate_count, 2);
        assert_eq!(projection.bin_count, 1);
        assert_eq!(authority.multi_bin_preflight[0].member_count, 2);
        assert_eq!(
            authority
                .multi_bin_members
                .iter()
                .map(|member| member.item)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(authority.revision, 0);
        assert_eq!(authority.slot_span(), 0);

        let mut transparent = rigid_item(2, 2.0);
        transparent.cast_shadows = true;
        transparent
            .material_classification
            .insert(MaterialClassification::TRANSPARENT);
        let scene = extracted_scene([rigid_item(1, 1.0), transparent]);
        let mut shadow_authority = GpuScene::new();

        let projection = shadow_authority
            .preflight_multi_bin_scene(&scene, 16, 256)
            .unwrap()
            .unwrap()
            .projection;
        assert_eq!(projection.candidate_count, 1);
        assert!(!projection.covers_complete_shadow_domain);
        assert!(shadow_authority.preflight_has_shadow_casters());

        let only_transparent = extracted_scene([scene.render_items[1].clone()]);
        assert!(
            shadow_authority
                .preflight_multi_bin_scene(&only_transparent, 16, 256)
                .unwrap()
                .is_none()
        );
        assert!(shadow_authority.preflight_has_shadow_casters());

        let mut non_caster = scene.render_items[1].clone();
        non_caster.cast_shadows = false;
        let complete_scene = extracted_scene([scene.render_items[0].clone(), non_caster]);
        let projection = shadow_authority
            .preflight_multi_bin_scene(&complete_scene, 16, 256)
            .unwrap()
            .unwrap()
            .projection;
        assert!(projection.covers_complete_shadow_domain);
    }

    #[test]
    fn cpu_only_table_planning_tracks_specs_revision_and_dirty_ranges() {
        let mut scene = GpuScene::new();
        let mut registry = PersistentBufferRegistry::new();
        let empty = scene.plan_instance_table(&mut registry, 1).unwrap();
        assert_eq!(empty.records.len(), 1);
        assert_eq!(empty.slot_span, 0);
        assert_eq!(empty.records[0].metadata[1], 0);
        assert_eq!(empty.persistent.buffer.spec.desc.logical_size, 256);

        scene.synchronize_scene(8, [input(1, 0.0), input(2, 0.0)]);
        let plan = scene.plan_instance_table(&mut registry, 2).unwrap();
        assert!(!plan.persistent.valid);
        assert!(plan.persistent.commit.is_some());
        assert_eq!(plan.revision, 1);
        assert_eq!(plan.publication_probe.revision, 1);
        assert_eq!(plan.publication_probe.buffer, plan.persistent.buffer);
        assert_eq!(
            (plan.records.len(), plan.slot_span, plan.alive_count),
            (2, 2, 2)
        );
        assert_eq!(plan.dirty_ranges.len(), 1);
        assert_eq!(
            plan.persistent.buffer.spec.schema_id,
            GPU_SCENE_INSTANCE_SCHEMA_ID
        );
        assert_eq!(plan.persistent.buffer.spec.stride, 256);
        assert_eq!(plan.persistent.buffer.spec.desc.logical_size, 512);
        assert_eq!(
            plan.persistent.buffer.spec.desc.usage,
            GPU_SCENE_INSTANCE_BUFFER_USAGE
        );
    }

    #[test]
    fn instance_sync_preserves_identity_flags_domains_and_transaction_atomicity() {
        let mut scene = GpuScene::new();
        let first = scene.synchronize_scene(7, [input(1, 3.0)]);
        let id = scene.instance_id(7, node(1)).unwrap();
        assert_eq!(first.revision, 1);
        assert_eq!((first.inserted, first.updated, first.retired), (1, 0, 0));

        let second = scene.synchronize_scene(7, [input(1, 3.0)]);
        assert_eq!(second.revision, 1);
        assert_eq!((second.inserted, second.updated, second.retired), (0, 0, 0));
        assert_eq!(scene.instance_id(7, node(1)), Some(id));
        assert!(scene.record(id).is_some());
        assert_eq!(scene.dirty_slots, [id.slot()]);

        let mut updated = GpuScene::new();
        updated.synchronize_scene(3, [input(1, 0.0), input(2, 0.0)]);
        assert!(updated.acknowledge_submitted_revision(1));
        let stats = updated.synchronize_scene(3, [input(1, 4.0), input(2, 5.0)]);
        assert_eq!(stats.revision, 2);
        assert_eq!(stats.updated, 2);
        assert_eq!(updated.dirty_slots.len(), 2);
        let retry = updated.synchronize_scene(3, [input(1, 4.0), input(2, 5.0)]);
        assert_eq!(retry.revision, 2);
        assert_eq!(updated.dirty_slots.len(), 2);

        let mut extracted = GpuScene::new();
        let caster = rigid_item(1, 0.0);
        extracted.synchronize_extracted_scene(&extracted_scene([caster.clone()]), 0.0);
        let caster_id = extracted.instance_id(17, node(1)).unwrap();
        assert_eq!(
            extracted.record(caster_id).unwrap().metadata[3],
            GPU_SCENE_INSTANCE_CASTS_SHADOW
        );
        let mut non_caster = caster;
        non_caster.cast_shadows = false;
        let changed = extracted.synchronize_extracted_scene(&extracted_scene([non_caster]), 0.1);
        assert_eq!((changed.revision, changed.updated), (2, 1));
        assert_eq!(extracted.record(caster_id).unwrap().metadata[3], 0);

        let mut domains = GpuScene::new();
        domains.synchronize_scene(1, [input(9, 1.0)]);
        domains.synchronize_scene(2, [input(9, 2.0)]);
        let first = domains.instance_id(1, node(9)).unwrap();
        let second = domains.instance_id(2, node(9)).unwrap();
        assert_ne!(first, second);
        assert_eq!(domains.record(first).unwrap().metadata[2], 1);
        assert_eq!(domains.record(second).unwrap().metadata[2], 2);
        domains.synchronize_scene(1, []);
        assert!(domains.instance_id(1, node(9)).is_none());
        assert_eq!(domains.instance_id(2, node(9)), Some(second));

        let duplicate = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut invalid = GpuScene::new();
            invalid.synchronize_scene(1, [input(4, 0.0), input(4, 1.0)]);
        }));
        assert!(
            duplicate.is_err(),
            "duplicate nodes must fail the transaction"
        );
    }

    #[test]
    fn domain_residency_uses_time_and_deterministic_budget_for_all_authorities() {
        let mut alternating = GpuScene::new();
        alternating.synchronize_scene_transaction(1, [input(1, 0.0)], Some(0.0));
        alternating.synchronize_scene_transaction(2, [input(2, 0.0)], Some(0.001));
        let first = alternating.instance_id(1, node(1)).unwrap();
        let second = alternating.instance_id(2, node(2)).unwrap();
        for step in 2..2_050 {
            let (scene_id, node_id) = if step % 2 == 0 { (1, 1) } else { (2, 2) };
            alternating.synchronize_scene_transaction(
                scene_id,
                [input(node_id, 0.0)],
                Some(step as f64 * 0.001),
            );
        }
        assert_eq!(alternating.instance_id(1, node(1)), Some(first));
        assert_eq!(alternating.instance_id(2, node(2)), Some(second));
        assert_eq!(
            alternating.revision, 2,
            "cache touches are not table changes"
        );

        let mut expiry = GpuScene::new();
        expiry.synchronize_scene_transaction(1, [input(1, 0.0)], Some(0.0));
        expiry.synchronize_scene_transaction(2, [input(2, 0.0)], Some(0.1));
        let expiring = expiry.instance_id(1, node(1)).unwrap();
        for _ in 0..2_000 {
            expiry.synchronize_scene_transaction(2, [input(2, 0.0)], Some(0.1));
        }
        expiry.synchronize_scene_transaction(2, [input(2, 0.0)], Some(1.499));
        assert_eq!(expiry.instance_id(1, node(1)), Some(expiring));
        let expired = expiry.synchronize_scene_transaction(2, [input(2, 0.0)], Some(1.5));
        assert_eq!(expired.retired, 1);
        assert!(expiry.instance_id(1, node(1)).is_none());

        let mut budgeted = GpuScene::new();
        for scene_id in 1..=4 {
            budgeted.synchronize_scene_transaction(
                scene_id,
                [input(u64::from(scene_id), 0.0)],
                Some(0.0),
            );
        }
        let revision_before = budgeted.revision;
        let fifth = budgeted.synchronize_scene_transaction(5, [input(5, 0.0)], Some(0.0));
        assert_eq!(fifth.revision, revision_before + 1);
        assert_eq!((fifth.inserted, fifth.retired), (1, 1));
        assert_eq!(budgeted.domains.len(), GPU_SCENE_DOMAIN_BUDGET);
        assert!(budgeted.instance_id(1, node(1)).is_none());

        let mut authority = GpuScene::new();
        let mut lists = RenderLists::new();
        for scene_id in 1..=5u32 {
            let mut scene = extracted_scene([rigid_item_with_assets(
                u64::from(scene_id),
                100 + u64::from(scene_id),
                200 + u64::from(scene_id),
            )]);
            scene.scene_id = scene_id;
            lists.begin_intent_collection();
            preflight_and_commit(&mut authority, &scene, 0.0, &mut lists);
        }
        assert!(authority.instance_id(1, node(1)).is_none());
        assert!(authority.geometries.id(geometry(101)).is_none());
        assert!(authority.materials.id(material(201)).is_none());
        assert!(authority.batches.lookup.keys().all(|key| key.scene_id != 1));
        assert_eq!(authority.quarantined_slots.len(), 1);
        assert_eq!(authority.geometries.quarantined_slots.len(), 1);
        assert_eq!(authority.materials.quarantined_slots.len(), 1);
        assert_eq!(authority.batches.quarantined_slots.len(), 1);
    }

    #[test]
    fn stale_ack_cannot_trim_but_exact_ack_regrowth_rejects_aba_ids() {
        let mut scene = GpuScene::new();
        scene.synchronize_scene_transaction(1, [input(1, 0.0)], Some(0.0));
        scene.synchronize_scene_transaction(2, [input(2, 0.0)], Some(0.1));
        let retired_id = scene.instance_id(2, node(2)).unwrap();
        scene.synchronize_scene_transaction(1, [input(1, 0.0)], Some(0.2));

        let swept = scene.synchronize_scene_transaction(1, [input(1, 0.0)], Some(1.7));
        assert_eq!(swept.revision, 3);
        assert_eq!(swept.retired, 1);
        assert_eq!(scene.slot_span(), 2);
        assert!(!scene.acknowledge_submitted_revision(2));
        scene.synchronize_scene_transaction(1, [input(1, 0.0)], Some(1.8));
        assert_eq!(scene.slot_span(), 2, "quarantine must block tail trim");

        assert!(scene.acknowledge_submitted_revision(3));
        assert_eq!(
            scene.slot_span(),
            2,
            "ack itself cannot alias graph borrows"
        );
        let mut registry = PersistentBufferRegistry::new();
        {
            let plan = scene.plan_instance_table(&mut registry, 1).unwrap();
            assert_eq!(plan.slot_span, 1);
            assert_eq!(plan.persistent.buffer.spec.desc.logical_size, 256);
        }

        scene.synchronize_scene_transaction(3, [input(3, 0.0)], Some(1.9));
        let replacement = scene.instance_id(3, node(3)).unwrap();
        assert_eq!(replacement.slot(), retired_id.slot());
        assert_eq!(replacement.generation(), retired_id.generation() + 1);
        assert!(scene.record(retired_id).is_none());
        assert!(scene.record(replacement).is_some());

        let mut projected = GpuScene::new();
        projected.synchronize_extracted_scene(
            &extracted_scene([
                rigid_item(1, 0.0),
                rigid_item(2, 0.0),
                rigid_item(3, 0.0),
                rigid_item(4, 0.0),
            ]),
            0.0,
        );
        assert!(projected.acknowledge_submitted_revision(1));
        projected.synchronize_extracted_scene(
            &extracted_scene([rigid_item(1, 0.0), rigid_item(2, 0.0)]),
            0.1,
        );
        let regrown = extracted_scene([rigid_item(1, 0.0), rigid_item(2, 0.0), rigid_item(5, 0.0)]);
        assert_eq!(
            projected.projected_slot_span_for_extracted_scene(&regrown),
            Some(5)
        );
        assert!(projected.acknowledge_submitted_revision(2));
        assert_eq!(projected.slot_span(), 4, "projection must remain read-only");
        assert_eq!(
            projected.projected_slot_span_for_extracted_scene(&regrown),
            Some(3)
        );
    }

    #[test]
    fn dirty_journal_coalesces_retries_and_survives_stale_acknowledgements() {
        let mut scene = GpuScene::new();
        scene.synchronize_scene(
            1,
            [input(1, 0.0), input(2, 0.0), input(3, 0.0), input(4, 0.0)],
        );
        assert!(scene.acknowledge_submitted_revision(1));
        scene.synchronize_scene(
            1,
            [input(1, 1.0), input(2, 2.0), input(3, 0.0), input(4, 4.0)],
        );
        scene.rebuild_dirty_ranges();
        assert_eq!(
            scene.dirty_ranges,
            [
                GpuSceneDirtyRange {
                    first_slot: 0,
                    slot_count: 2,
                },
                GpuSceneDirtyRange {
                    first_slot: 3,
                    slot_count: 1,
                },
            ]
        );
        assert_eq!(scene.dirty_ranges[0].byte_offset(), 0);
        assert_eq!(scene.dirty_ranges[0].byte_size(), 512);

        let mut replay = GpuScene::new();
        replay.synchronize_scene(5, [input(1, 0.0)]);
        assert!(replay.acknowledge_submitted_revision(1));
        replay.synchronize_scene(5, []);
        assert!(!replay.acknowledge_submitted_revision(1));
        assert_eq!(replay.dirty_slots.len(), 1);
        assert_eq!(replay.quarantined_slots.len(), 1);
        assert!(replay.reusable_slots.is_empty());

        let mut retries = GpuScene::new();
        retries.synchronize_scene(5, [input(1, 0.0), input(2, 0.0)]);
        assert!(retries.acknowledge_submitted_revision(1));
        let first_retry = retries.synchronize_scene(5, [input(1, 1.0), input(2, 0.0)]);
        assert_eq!(first_retry.revision, 2);
        let second_retry = retries.synchronize_scene(5, [input(1, 1.0), input(2, 2.0)]);
        assert_eq!(second_retry.revision, 3);
        assert_eq!(retries.dirty_slots.len(), 2);
        retries.rebuild_dirty_ranges();
        assert_eq!(
            retries.dirty_ranges,
            [GpuSceneDirtyRange {
                first_slot: 0,
                slot_count: 2,
            }]
        );
    }

    #[test]
    fn submitted_tail_trim_shrinks_logical_spec_and_requires_full_replay() {
        let gpu = gpu();
        let mut scene = GpuScene::new();
        let mut registry = PersistentBufferRegistry::new();
        scene.synchronize_scene(8, [input(1, 0.0), input(2, 0.0)]);

        let (first_persistent, first_probe) = {
            let plan = scene.plan_instance_table(&mut registry, 1).unwrap();
            assert_eq!(plan.persistent.buffer.spec.desc.logical_size, 512);
            (plan.persistent, plan.publication_probe)
        };
        let first_physical = registry
            .materialize(
                &gpu.device,
                first_persistent.buffer,
                Some("GPU Scene initial table"),
            )
            .unwrap();
        assert!(first_physical.requires_full_initialization);
        registry
            .commit_batch(&[PersistentBufferCommitRequest::new(
                first_persistent.commit.unwrap(),
                first_physical.token,
            )])
            .unwrap();
        assert!(scene.acknowledge_if_published(&registry, first_probe));

        scene.retire_scene(8);
        let (tombstone_persistent, tombstone_probe) = {
            let plan = scene.plan_instance_table(&mut registry, 2).unwrap();
            assert_eq!(
                plan.slot_span, 2,
                "unsubmitted tombstones remain addressable"
            );
            assert_eq!(plan.persistent.buffer.spec.desc.logical_size, 512);
            (plan.persistent, plan.publication_probe)
        };
        let tombstone_physical = registry
            .materialize(&gpu.device, tombstone_persistent.buffer, None)
            .unwrap();
        assert!(!tombstone_physical.requires_full_initialization);
        assert_eq!(
            tombstone_physical.token.physical_id,
            first_physical.token.physical_id
        );
        registry
            .commit_batch(&[PersistentBufferCommitRequest::new(
                tombstone_persistent.commit.unwrap(),
                tombstone_physical.token,
            )])
            .unwrap();
        assert!(scene.acknowledge_if_published(&registry, tombstone_probe));
        assert_eq!(
            scene.slot_span(),
            2,
            "same-frame acknowledgement only releases slots"
        );

        let compact_persistent = {
            let plan = scene.plan_instance_table(&mut registry, 3).unwrap();
            assert_eq!(plan.slot_span, 0);
            assert_eq!(plan.records.len(), 1);
            assert_eq!(plan.persistent.buffer.spec.desc.logical_size, 256);
            plan.persistent
        };
        let compact_physical = registry
            .materialize(&gpu.device, compact_persistent.buffer, None)
            .unwrap();
        assert!(
            compact_physical.requires_full_initialization,
            "logical-size changes must not apply a delta to the old table meaning"
        );
        assert_eq!(
            compact_physical.token.physical_id, first_physical.token.physical_id,
            "logical shrink should reuse retained physical capacity"
        );
        assert_eq!(compact_physical.allocation_size, 512);
    }

    #[test]
    fn authority_capacity_trim_preserves_live_ids_domains_records_and_watermarks() {
        let mut scene = GpuScene::new();
        scene.synchronize_scene(17, [input(1, 0.0), input(2, 2.0)]);
        let ids = [
            scene.instance_id(17, node(1)).unwrap(),
            scene.instance_id(17, node(2)).unwrap(),
        ];
        let records = [
            *scene.record(ids[0]).unwrap(),
            *scene.record(ids[1]).unwrap(),
        ];
        let domain_slots = scene.domains.get(&17).unwrap().slots.clone();
        let watermarks = scene.slot_generation_watermarks.clone();
        scene.reserve_capacity_probe(0, 16_384);
        scene.domains.get_mut(&17).unwrap().slots.reserve(16_384);

        let before = scene.authority_capacity_usage();
        assert!(before.retained_bytes > before.active_bytes);
        scene.trim_authority_capacity(before.active_bytes, before.retained_bytes);
        let after = scene.authority_capacity_usage();

        assert_eq!(after.active_bytes, before.active_bytes);
        assert!(after.retained_bytes < before.retained_bytes);
        assert_eq!(scene.slot_generation_watermarks, watermarks);
        assert_eq!(scene.domains.get(&17).unwrap().slots, domain_slots);
        for ((id, node), record) in ids.into_iter().zip([node(1), node(2)]).zip(records) {
            assert_eq!(scene.instance_id(17, node), Some(id));
            assert!(scene.record(id).unwrap().bitwise_eq(&record));
        }
    }
}
