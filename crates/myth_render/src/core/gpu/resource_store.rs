//! GPU-side asset arenas and reverse indices.

use std::{mem::size_of, num::NonZeroU32};

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use slotmap::{SecondaryMap, SparseSecondaryMap};
use smallvec::SmallVec;

use myth_assets::{GeometryHandle, ImageHandle, MaterialHandle, TextureHandle};
use myth_resources::buffer::BufferRef;
use myth_resources::image::Image;
use myth_resources::texture::{Texture, TextureSampler};

use super::texture::MaterializedAssetTexture;
use super::{
    AssetTextureMaterialization, AssetTextureSnapshot, EnsureResult, GpuBuffer, GpuBufferHandle,
    GpuGeometry, GpuImage, GpuImageSpec, GpuMaterial, MipmapRequest, PhysicalImageGenerationKey,
    PhysicalImageId, ResourceManager, TextureBinding, TextureImageLease, Tracked,
};

const ASSET_TEXTURE_CANDIDATE_BASELINE: usize = 128;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AssetTextureCandidateId(NonZeroU32);

impl AssetTextureCandidateId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("asset texture candidate index exceeds u32::MAX - 1");
        Self(raw)
    }

    #[inline]
    const fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PhysicalAssetTextureCandidateKey {
    spec: GpuImageSpec,
    image_version: u32,
    image_identity: u64,
}

/// Complete cache-side content state used by a Fresh candidate's CAS.
/// Version/identity are retained explicitly so correctness does not depend on
/// any one allocation-generation implementation detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PhysicalAssetTextureObservedState {
    generation: PhysicalImageGenerationKey,
    image_version: u32,
    image_identity: u64,
}

enum PhysicalAssetTextureCandidateSource {
    /// An exact persistent allocation observed while the candidate was built.
    Existing {
        expected: PhysicalImageGenerationKey,
    },
    /// A deferred same-spec refresh. It preserves the existing allocation and
    /// is bracketed by selected-live postflights around the upload.
    Refresh {
        observed: PhysicalAssetTextureObservedState,
    },
    /// No physical allocation existed at staging. The private image is built
    /// after the first postflight and enters the persistent pool only after
    /// the upload-side postflight also succeeds.
    New { image_slot: Option<usize> },
}

struct PhysicalAssetTextureCandidate {
    key: PhysicalAssetTextureCandidateKey,
    source: PhysicalAssetTextureCandidateSource,
    /// First accepted logical member. This lets a new physical entry be born
    /// with a member and preserves the no-zero-member pool invariant.
    first_live_member: Option<TextureHandle>,
    /// Only a newly inserted absent-cache target is born with this member.
    /// Existing targets use the lease table for O(1) attachment checks.
    pre_attached_member: Option<TextureHandle>,
    resolved: bool,
    target: Option<PhysicalImageId>,
    target_state: Option<PhysicalAssetTextureObservedState>,
}

struct LogicalAssetTextureCandidate {
    snapshot: AssetTextureSnapshot,
    sampler: TextureSampler,
    physical: Option<usize>,
    sampler_id: Option<usize>,
    result: Option<AssetTextureMaterialization>,
}

/// Graph-attempt owner for unpublished Asset Texture GPU candidates.
///
/// Physical work is stable-deduplicated in first-occurrence order. Candidate
/// staging is CPU-only; uploads and persistent cache mutation begin only after
/// the selected-live T -> I -> T postflight succeeds.
pub(crate) struct AssetTextureCandidateBatch {
    physical: Vec<PhysicalAssetTextureCandidate>,
    physical_lookup: FxHashMap<PhysicalAssetTextureCandidateKey, usize>,
    /// Cold-path GPU owners live out of line so every warm physical candidate
    /// does not inherit the size of `GpuImage`. Slots are retained as one
    /// batch allocation and never allocate once capacity is warm.
    private_images: Vec<Option<GpuImage>>,
    logical: Vec<LogicalAssetTextureCandidate>,
    published: bool,
}

impl AssetTextureCandidateBatch {
    #[must_use]
    pub(crate) fn new() -> Self {
        let mut physical_lookup = FxHashMap::default();
        physical_lookup.reserve(ASSET_TEXTURE_CANDIDATE_BASELINE);
        Self {
            physical: Vec::with_capacity(ASSET_TEXTURE_CANDIDATE_BASELINE),
            physical_lookup,
            private_images: Vec::with_capacity(ASSET_TEXTURE_CANDIDATE_BASELINE),
            logical: Vec::with_capacity(ASSET_TEXTURE_CANDIDATE_BASELINE),
            published: false,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.logical.is_empty()
            && self.physical.is_empty()
            && self.physical_lookup.is_empty()
            && self.private_images.is_empty()
    }

    /// Move one publication result out exactly once. An unavailable result is
    /// still a completed result and is therefore represented by `Some` here.
    pub(crate) fn take_result(
        &mut self,
        id: AssetTextureCandidateId,
    ) -> AssetTextureMaterialization {
        assert!(
            self.published,
            "asset texture candidates were not published"
        );
        self.logical
            .get_mut(id.index())
            .and_then(|candidate| candidate.result.take())
            .expect("asset texture candidate result was already taken or is invalid")
    }

    /// Release all private GPU owners and materialization attempts while
    /// retaining the outer allocations for the next graph attempt.
    pub(crate) fn clear(&mut self) {
        self.logical.clear();
        self.private_images.clear();
        self.physical_lookup.clear();
        self.physical.clear();
        self.published = false;
    }

    #[inline]
    #[must_use]
    pub(crate) fn active_bytes(&self) -> usize {
        self.physical
            .len()
            .saturating_mul(size_of::<PhysicalAssetTextureCandidate>())
            .saturating_add(
                self.logical
                    .len()
                    .saturating_mul(size_of::<LogicalAssetTextureCandidate>()),
            )
            .saturating_add(self.physical_lookup.len().saturating_mul(
                size_of::<(PhysicalAssetTextureCandidateKey, usize)>().saturating_add(1),
            ))
            .saturating_add(
                self.private_images
                    .len()
                    .saturating_mul(size_of::<Option<GpuImage>>()),
            )
    }

    #[inline]
    #[must_use]
    pub(crate) fn retained_bytes(&self) -> usize {
        self.physical
            .capacity()
            .saturating_mul(size_of::<PhysicalAssetTextureCandidate>())
            .saturating_add(
                self.logical
                    .capacity()
                    .saturating_mul(size_of::<LogicalAssetTextureCandidate>()),
            )
            .saturating_add(self.physical_lookup.capacity().saturating_mul(
                size_of::<(PhysicalAssetTextureCandidateKey, usize)>().saturating_add(1),
            ))
            .saturating_add(
                self.private_images
                    .capacity()
                    .saturating_mul(size_of::<Option<GpuImage>>()),
            )
    }

    #[inline]
    #[must_use]
    pub(crate) const fn baseline_bytes() -> usize {
        ASSET_TEXTURE_CANDIDATE_BASELINE.saturating_mul(
            size_of::<PhysicalAssetTextureCandidate>()
                + size_of::<LogicalAssetTextureCandidate>()
                + size_of::<(PhysicalAssetTextureCandidateKey, usize)>()
                + size_of::<Option<GpuImage>>()
                + 1,
        )
    }

    /// Apply the graph capacity governor's family target/current ratio.
    /// Private GPU owners must have been released before retained CPU storage
    /// is trimmed.
    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        assert!(
            self.is_empty(),
            "live asset texture candidates cannot be trimmed"
        );
        if retained_bytes == 0 || target_bytes >= retained_bytes {
            return;
        }
        let target = |capacity: usize| {
            let proportional = ((capacity as u128) * (target_bytes as u128))
                .div_ceil(retained_bytes as u128)
                .min(usize::MAX as u128) as usize;
            proportional
                .max(ASSET_TEXTURE_CANDIDATE_BASELINE)
                .min(capacity)
        };
        self.physical.shrink_to(target(self.physical.capacity()));
        self.logical.shrink_to(target(self.logical.capacity()));
        self.private_images
            .shrink_to(target(self.private_images.capacity()));
        self.physical_lookup
            .shrink_to(target(self.physical_lookup.capacity()));
    }
}

struct PhysicalImageEntry {
    spec: GpuImageSpec,
    image: GpuImage,
    /// Texture assets currently leasing this exact physical allocation.
    /// The member list is the single source of truth for reference count.
    members: SmallVec<[TextureHandle; 2]>,
}

/// GPU-side asset arenas and reverse indices.
///
/// This is the part of `ResourceManager` that mirrors CPU asset handles to
/// persistent GPU allocations. Keeping it separate lets callers borrow asset
/// storage independently from bind-group and environment state.
pub(crate) struct GpuResourceStore {
    gpu_geometries: SecondaryMap<GeometryHandle, GpuGeometry>,
    /// Slotmap-compatible generation ordering with pointer-sized sparse slots.
    /// `GpuMaterial` owners are boxed during cold candidate construction so a
    /// high Material handle never allocates an array of large owner values.
    gpu_materials: SparseSecondaryMap<MaterialHandle, Option<Box<GpuMaterial>>, FxBuildHasher>,
    /// Destructor-bearing exact physical image arena.
    physical_images: slotmap::SlotMap<PhysicalImageId, PhysicalImageEntry>,
    /// Referenced-only physical pool. Zero-member entries are released
    /// immediately rather than retained as an implicit VRAM cache.
    physical_image_index: FxHashMap<GpuImageSpec, PhysicalImageId>,
    /// Logical TextureHandle residency pointing into the physical arena.
    texture_image_leases: SecondaryMap<TextureHandle, TextureImageLease>,

    /// Compatibility lookup for callers that still only carry an ImageHandle.
    /// Exact render paths must use `texture_image(TextureHandle)` instead.
    latest_texture_for_image: SecondaryMap<ImageHandle, TextureHandle>,

    /// Mapping from `TextureHandle` to (`ImageId`, `SamplerId`)
    texture_bindings: SecondaryMap<TextureHandle, TextureBinding>,

    /// All GPU buffers stored in a contiguous arena for O(1) handle-based access.
    gpu_buffers: slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
    /// Reverse index: CPU-side buffer ID -> SlotMap handle.
    buffer_index: FxHashMap<u64, GpuBufferHandle>,

    pending_mipmap_requests: Vec<MipmapRequest>,
    /// At most the newest content generation for one physical allocation is
    /// retained before ownership moves into GraphStorage.
    pending_mipmap_indices: FxHashMap<PhysicalImageId, usize>,
}

impl GpuResourceStore {
    #[must_use]
    pub(super) fn new(
        gpu_buffers: slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
        buffer_index: FxHashMap<u64, GpuBufferHandle>,
    ) -> Self {
        Self {
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SparseSecondaryMap::with_hasher(FxBuildHasher),
            physical_images: slotmap::SlotMap::with_key(),
            physical_image_index: FxHashMap::default(),
            texture_image_leases: SecondaryMap::new(),
            latest_texture_for_image: SecondaryMap::new(),
            texture_bindings: SecondaryMap::new(),
            gpu_buffers,
            buffer_index,
            pending_mipmap_requests: Vec::new(),
            pending_mipmap_indices: FxHashMap::default(),
        }
    }

    pub(super) fn queue_mipmap_request(&mut self, request: MipmapRequest) {
        let is_current_generation = self
            .physical_images
            .get(request.physical_id)
            .is_some_and(|entry| entry.image.generation_key(request.physical_id) == request.key());
        if !is_current_generation {
            // Reject requests detached by content replacement, TTL/removal,
            // or a SlotMap generation reuse even when the pending table has
            // already been drained into GraphStorage.
            return;
        }

        // Only the newest content generation of one physical allocation can
        // matter. Logical TextureHandles never participate in ownership or
        // claim routing.
        // This also prevents repeated begin_frame calls without render from
        // retaining an unbounded chain of obsolete WGPU texture owners.
        if let Some(&index) = self.pending_mipmap_indices.get(&request.physical_id) {
            let queued_generation = self.pending_mipmap_requests[index].generation_id;
            if queued_generation > request.generation_id {
                // A stale request may arrive after a newer coordinator result;
                // it must never replace the newest physical generation.
                return;
            }
            self.pending_mipmap_requests[index] = request;
            return;
        }

        let index = self.pending_mipmap_requests.len();
        self.pending_mipmap_indices
            .insert(request.physical_id, index);
        self.pending_mipmap_requests.push(request);
    }

    fn remove_pending_mipmap_at(&mut self, index: usize) -> MipmapRequest {
        let stale = self.pending_mipmap_requests.swap_remove(index);
        self.pending_mipmap_indices.remove(&stale.physical_id);
        if let Some(moved) = self.pending_mipmap_requests.get(index) {
            self.pending_mipmap_indices.insert(moved.physical_id, index);
        }
        stale
    }

    fn discard_mipmap_request_for_physical(&mut self, physical_id: PhysicalImageId) {
        if let Some(index) = self.pending_mipmap_indices.get(&physical_id).copied() {
            self.remove_pending_mipmap_at(index);
        }
    }

    #[must_use]
    pub(super) fn take_mipmap_requests(&mut self) -> Vec<MipmapRequest> {
        self.pending_mipmap_indices.clear();
        std::mem::take(&mut self.pending_mipmap_requests)
    }

    pub(super) fn buffer_storage_mut(
        &mut self,
    ) -> (
        &mut slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
        &mut FxHashMap<u64, GpuBufferHandle>,
    ) {
        (&mut self.gpu_buffers, &mut self.buffer_index)
    }

    #[inline]
    #[must_use]
    pub(super) fn geometry(&self, handle: GeometryHandle) -> Option<&GpuGeometry> {
        self.gpu_geometries.get(handle)
    }

    #[inline]
    pub(super) fn geometry_mut(&mut self, handle: GeometryHandle) -> Option<&mut GpuGeometry> {
        self.gpu_geometries.get_mut(handle)
    }

    #[inline]
    pub(super) fn insert_geometry(&mut self, handle: GeometryHandle, geometry: GpuGeometry) {
        self.gpu_geometries.insert(handle, geometry);
    }

    #[inline]
    #[must_use]
    pub(super) fn material(&self, handle: MaterialHandle) -> Option<&GpuMaterial> {
        self.gpu_materials.get(handle).and_then(Option::as_deref)
    }

    #[inline]
    pub(super) fn material_mut(&mut self, handle: MaterialHandle) -> Option<&mut GpuMaterial> {
        self.gpu_materials
            .get_mut(handle)
            .and_then(Option::as_deref_mut)
    }

    #[inline]
    pub(super) fn insert_material(&mut self, handle: MaterialHandle, material: GpuMaterial) {
        assert!(
            self.reserve_material_slot(handle),
            "cannot insert an obsolete or null Material handle"
        );
        drop(self.swap_material(handle, Box::new(material)));
    }

    /// Reserve the exact generational cache slot before entering a bounded
    /// AssetStorage publication guard.
    ///
    /// Reusing a slot index with a newer `MaterialHandle` retires the obsolete
    /// owner here, where destructor work cannot extend the publication gate.
    #[inline]
    pub(super) fn reserve_material_slot(&mut self, handle: MaterialHandle) -> bool {
        let Some(entry) = self.gpu_materials.entry(handle) else {
            // Null and older wrapped generations cannot displace a current
            // cache owner. Their candidates fail closed at publication.
            return false;
        };
        entry.or_insert(None);
        true
    }

    #[inline]
    #[must_use]
    pub(super) fn material_slot_exists(&self, handle: MaterialHandle) -> bool {
        self.gpu_materials.contains_key(handle)
    }

    #[inline]
    pub(super) fn reserve_material_slots(&mut self, additional: usize) {
        self.gpu_materials.reserve(additional);
    }

    /// Replace a pre-reserved Material owner without allocation or destruction.
    /// The caller retains the returned owner until after its publication guard.
    #[inline]
    pub(super) fn swap_material(
        &mut self,
        handle: MaterialHandle,
        material: Box<GpuMaterial>,
    ) -> Option<Box<GpuMaterial>> {
        self.gpu_materials
            .get_mut(handle)
            .expect("Material cache slot must be reserved before publication")
            .replace(material)
    }

    #[cfg(test)]
    #[must_use]
    pub(super) fn material_slot_capacity(&self) -> usize {
        self.gpu_materials.capacity()
    }

    #[inline]
    #[must_use]
    pub(super) fn texture_image(&self, handle: TextureHandle) -> Option<&GpuImage> {
        let lease = self.texture_image_leases.get(handle)?;
        self.physical_images
            .get(lease.physical_id)
            .map(|entry| &entry.image)
    }

    #[inline]
    #[must_use]
    #[cfg(test)]
    pub(super) fn texture_image_lease(&self, handle: TextureHandle) -> Option<TextureImageLease> {
        self.texture_image_leases.get(handle).copied()
    }

    #[cfg(test)]
    #[must_use]
    pub(super) fn contains_physical_image(&self, id: PhysicalImageId) -> bool {
        self.physical_images.contains_key(id)
    }

    #[cfg(test)]
    #[must_use]
    pub(super) fn physical_image_member_count(&self, id: PhysicalImageId) -> Option<usize> {
        self.physical_images
            .get(id)
            .map(|entry| entry.members.len())
    }

    #[cfg(test)]
    pub(super) fn assert_texture_pool_consistent(&self) {
        let mut member_count = 0;
        for (physical_id, entry) in &self.physical_images {
            assert!(!entry.members.is_empty());
            assert_eq!(
                self.physical_image_index.get(&entry.spec),
                Some(&physical_id)
            );
            for handle in &entry.members {
                assert_eq!(
                    self.texture_image_leases
                        .get(*handle)
                        .map(|lease| lease.physical_id),
                    Some(physical_id)
                );
                member_count += 1;
            }
        }
        assert_eq!(member_count, self.texture_image_leases.len());
        for (spec, physical_id) in &self.physical_image_index {
            assert_eq!(
                self.physical_images
                    .get(*physical_id)
                    .map(|entry| entry.spec),
                Some(*spec)
            );
        }
        for request in &self.pending_mipmap_requests {
            assert!(
                self.physical_images
                    .get(request.physical_id)
                    .is_some_and(|entry| {
                        entry.image.generation_key(request.physical_id) == request.key()
                    })
            );
        }
    }

    #[inline]
    fn physical_state_for_spec(
        &self,
        spec: GpuImageSpec,
    ) -> Option<PhysicalAssetTextureObservedState> {
        let physical_id = *self.physical_image_index.get(&spec)?;
        let entry = self.physical_images.get(physical_id)?;
        (entry.spec == spec && !entry.members.is_empty()).then(|| {
            PhysicalAssetTextureObservedState {
                generation: entry.image.generation_key(physical_id),
                image_version: entry.image.image_version,
                image_identity: entry.image.image_identity,
            }
        })
    }

    #[inline]
    fn exact_physical_generation(
        &self,
        key: PhysicalAssetTextureCandidateKey,
    ) -> Option<PhysicalImageGenerationKey> {
        let physical_id = *self.physical_image_index.get(&key.spec)?;
        let entry = self.physical_images.get(physical_id)?;
        (entry.spec == key.spec
            && !entry.members.is_empty()
            && entry.image.image_version == key.image_version
            && entry.image.image_identity == key.image_identity)
            .then(|| entry.image.generation_key(physical_id))
    }

    fn stage_asset_texture_candidate(
        &self,
        batch: &mut AssetTextureCandidateBatch,
        snapshot: AssetTextureSnapshot,
        texture: &Texture,
        image: &Image,
    ) -> AssetTextureCandidateId {
        assert!(
            !batch.published,
            "asset texture candidates cannot be staged after publication"
        );

        // The retained batch owners should reproduce the exact plan snapshot.
        // Treat a coordinator misuse as unavailable instead of uploading bytes
        // under a mismatched logical identity.
        let retained_snapshot = AssetTextureSnapshot::from_loaded(
            snapshot.texture_handle,
            texture,
            snapshot.texture_version,
            snapshot.texture_identity,
            image,
            snapshot.image_version,
            snapshot.image_identity,
        );
        let physical = if retained_snapshot == Some(snapshot) {
            let key = PhysicalAssetTextureCandidateKey {
                spec: snapshot.physical_spec(),
                image_version: snapshot.image_version,
                image_identity: snapshot.image_identity,
            };
            if let Some(&index) = batch.physical_lookup.get(&key) {
                Some(index)
            } else {
                let source = if let Some(expected) = self.exact_physical_generation(key) {
                    PhysicalAssetTextureCandidateSource::Existing { expected }
                } else {
                    match self.physical_state_for_spec(key.spec) {
                        Some(observed) => PhysicalAssetTextureCandidateSource::Refresh { observed },
                        None => PhysicalAssetTextureCandidateSource::New { image_slot: None },
                    }
                };
                let index = batch.physical.len();
                batch.physical.push(PhysicalAssetTextureCandidate {
                    key,
                    source,
                    first_live_member: None,
                    pre_attached_member: None,
                    resolved: false,
                    target: None,
                    target_state: None,
                });
                batch.physical_lookup.insert(key, index);
                Some(index)
            }
        } else {
            None
        };

        let id = AssetTextureCandidateId::from_index(batch.logical.len());
        batch.logical.push(LogicalAssetTextureCandidate {
            snapshot,
            sampler: texture.sampler,
            physical,
            sampler_id: None,
            result: None,
        });
        id
    }

    /// Resolve one accepted logical candidate into a physical target after
    /// AssetStorage postflight. Physical work is deduplicated, so only the
    /// first accepted logical member for a shared image performs this work.
    fn materialize_asset_texture_candidate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &mut AssetTextureCandidateBatch,
        id: AssetTextureCandidateId,
        image: &Image,
        frame_index: u64,
    ) -> bool {
        let Some(logical) = batch.logical.get(id.index()) else {
            return false;
        };
        let Some(physical_index) = logical.physical else {
            return false;
        };
        let first_member = logical.snapshot.texture_handle;
        let candidate = &mut batch.physical[physical_index];
        if candidate.resolved {
            return false;
        }
        candidate.resolved = true;
        candidate.first_live_member = Some(first_member);

        let (target, uploaded) = match &mut candidate.source {
            PhysicalAssetTextureCandidateSource::Existing { expected } => (
                self.exact_physical_generation(candidate.key)
                    .filter(|current| *current == *expected)
                    .map(|current| current.physical),
                false,
            ),
            PhysicalAssetTextureCandidateSource::Refresh { observed } => {
                let observed = *observed;
                if let Some(exact) = self.exact_physical_generation(candidate.key) {
                    (Some(exact.physical), false)
                } else if self.physical_state_for_spec(candidate.key.spec) != Some(observed) {
                    (None, false)
                } else {
                    let physical_id = observed.generation.physical;
                    self.discard_mipmap_request_for_physical(physical_id);
                    let (physical_images, texture_bindings) =
                        (&mut self.physical_images, &mut self.texture_bindings);
                    let entry = physical_images
                        .get_mut(physical_id)
                        .expect("observed physical image disappeared after CAS");
                    debug_assert_eq!(
                        PhysicalAssetTextureObservedState {
                            generation: entry.image.generation_key(physical_id),
                            image_version: entry.image.image_version,
                            image_identity: entry.image.image_identity,
                        },
                        observed
                    );
                    for member in &entry.members {
                        texture_bindings.remove(*member);
                    }
                    entry.image.update_content(
                        queue,
                        image,
                        candidate.key.image_version,
                        candidate.key.image_identity,
                    );
                    entry.image.last_used_frame = frame_index;
                    (Some(physical_id), true)
                }
            }
            PhysicalAssetTextureCandidateSource::New { image_slot } => {
                if let Some(exact) = self.exact_physical_generation(candidate.key) {
                    (Some(exact.physical), false)
                } else if self.physical_state_for_spec(candidate.key.spec).is_some() {
                    (None, false)
                } else {
                    let mut unpublished = GpuImage::new(
                        device,
                        queue,
                        image,
                        candidate.key.spec,
                        candidate.key.image_version,
                        candidate.key.image_identity,
                    );
                    unpublished.last_used_frame = frame_index;
                    let private_index = batch.private_images.len();
                    batch.private_images.push(Some(unpublished));
                    *image_slot = Some(private_index);
                    (None, true)
                }
            }
        };
        candidate.target = target;
        candidate.target_state =
            target.and_then(|_| self.physical_state_for_spec(candidate.key.spec));
        uploaded
    }

    /// Publish postflight-current candidates after their physical targets were
    /// resolved. A final cache-side CAS protects the short coordinator window
    /// between physical materialization and logical lease publication.
    fn publish_asset_texture_candidates(
        &mut self,
        batch: &mut AssetTextureCandidateBatch,
        current: &[bool],
        frame_index: u64,
    ) {
        debug_assert_eq!(batch.logical.len(), current.len());

        // Publish private new allocations only after the upload-side
        // postflight. Same-content winners are reused; any other cache change
        // rejects this physical candidate instead of overwriting it.
        for candidate in &mut batch.physical {
            if candidate.target_state.is_some() {
                continue;
            }
            let Some(first_member) = candidate.first_live_member else {
                continue;
            };
            let PhysicalAssetTextureCandidateSource::New { image_slot } = &mut candidate.source
            else {
                continue;
            };
            let target = if let Some(exact) = self.exact_physical_generation(candidate.key) {
                Some(exact.physical)
            } else if self.physical_state_for_spec(candidate.key.spec).is_some() {
                None
            } else {
                let Some(unpublished) = image_slot
                    .take()
                    .and_then(|slot| batch.private_images.get_mut(slot))
                    .and_then(Option::take)
                else {
                    continue;
                };
                // A stale dangling index is not a valid observed allocation.
                if let Some(stale_id) = self.physical_image_index.remove(&candidate.key.spec)
                    && self
                        .physical_images
                        .get(stale_id)
                        .is_some_and(|entry| entry.members.is_empty())
                {
                    self.discard_mipmap_request_for_physical(stale_id);
                    self.physical_images.remove(stale_id);
                }
                let physical_id = self.physical_images.insert(PhysicalImageEntry {
                    spec: candidate.key.spec,
                    image: unpublished,
                    members: SmallVec::from_slice(&[first_member]),
                });
                let replaced = self
                    .physical_image_index
                    .insert(candidate.key.spec, physical_id);
                debug_assert!(replaced.is_none());
                candidate.pre_attached_member = Some(first_member);
                Some(physical_id)
            };
            candidate.target = target;
            candidate.target_state =
                target.and_then(|_| self.physical_state_for_spec(candidate.key.spec));
        }

        // Validate all targets before migrating logical leases. New
        // allocations were born with their first accepted member, so the pool
        // never exposes a zero-member entry even transiently.
        for candidate in &mut batch.physical {
            let Some(expected) = candidate.target_state else {
                continue;
            };
            if self.physical_state_for_spec(candidate.key.spec) != Some(expected) {
                candidate.target = None;
                candidate.target_state = None;
            }
        }

        // Attach every accepted target before detaching any previous lease.
        // This makes swaps safe: moving the final old member cannot destroy a
        // physical allocation that another logical candidate will acquire.
        for (index, candidate) in batch.logical.iter().enumerate() {
            if current.get(index).copied() != Some(true) || candidate.sampler_id.is_none() {
                continue;
            }
            let Some(physical_index) = candidate.physical else {
                continue;
            };
            let Some(target) = batch.physical[physical_index].target else {
                continue;
            };
            let handle = candidate.snapshot.texture_handle;
            let already_attached = batch.physical[physical_index].pre_attached_member
                == Some(handle)
                || self
                    .texture_image_leases
                    .get(handle)
                    .is_some_and(|lease| lease.physical_id == target);
            if already_attached {
                continue;
            }
            let entry = self
                .physical_images
                .get_mut(target)
                .expect("published asset texture target disappeared before attachment");
            debug_assert!(!entry.members.contains(&handle));
            entry.members.push(handle);
        }

        for (index, candidate) in batch.logical.iter_mut().enumerate() {
            let unavailable = || Some(AssetTextureMaterialization::unavailable());
            if current.get(index).copied() != Some(true) {
                candidate.result = unavailable();
                continue;
            }
            let (Some(physical_index), Some(sampler_id)) =
                (candidate.physical, candidate.sampler_id)
            else {
                candidate.result = unavailable();
                continue;
            };
            let Some(physical_id) = batch.physical[physical_index].target else {
                candidate.result = unavailable();
                continue;
            };
            let snapshot = candidate.snapshot;
            let previous_lease = self
                .texture_image_leases
                .get(snapshot.texture_handle)
                .copied();

            self.texture_image_leases.insert(
                snapshot.texture_handle,
                TextureImageLease {
                    physical_id,
                    image_handle: snapshot.image_handle,
                    texture_version: snapshot.texture_version,
                    texture_identity: snapshot.texture_identity,
                    image_version: snapshot.image_version,
                    image_identity: snapshot.image_identity,
                    last_used_frame: frame_index,
                },
            );
            if let Some(previous) = previous_lease
                && previous.physical_id != physical_id
            {
                self.detach_physical_member(previous.physical_id, snapshot.texture_handle);
            }

            let (view, generation_key, mipmap_request) = {
                let entry = self
                    .physical_images
                    .get_mut(physical_id)
                    .expect("asset texture target disappeared during publication");
                debug_assert!(entry.members.contains(&snapshot.texture_handle));
                debug_assert_eq!(entry.spec, snapshot.physical_spec());
                debug_assert_eq!(entry.image.image_version, snapshot.image_version);
                debug_assert_eq!(entry.image.image_identity, snapshot.image_identity);
                entry.image.last_used_frame = frame_index;
                (
                    Tracked::with_id(entry.image.default_view.clone(), entry.image.id),
                    entry.image.generation_key(physical_id),
                    snapshot
                        .generate_mipmaps
                        .then(|| entry.image.mipmap_request(physical_id))
                        .flatten(),
                )
            };

            self.latest_texture_for_image
                .insert(snapshot.image_handle, snapshot.texture_handle);
            if let Some(previous_image_handle) = previous_lease.map(|lease| lease.image_handle)
                && previous_image_handle != snapshot.image_handle
            {
                self.repair_texture_image_alias(previous_image_handle);
            }
            self.insert_texture_binding(
                snapshot.texture_handle,
                TextureBinding {
                    view_id: generation_key.allocation,
                    image_handle: snapshot.image_handle,
                    sampler_id,
                    texture_version: snapshot.texture_version,
                    texture_identity: snapshot.texture_identity,
                    image_version: snapshot.image_version,
                    image_identity: snapshot.image_identity,
                },
            );
            candidate.result = Some(AssetTextureMaterialization {
                texture: Some(MaterializedAssetTexture {
                    view,
                    generation_key,
                }),
                mipmap_request,
            });
        }
    }

    /// Acquire or update one TextureHandle's exact physical residency.
    ///
    /// Sampler and logical asset stamps remain per-handle. Only the immutable
    /// allocation descriptor and ImageHandle select a shared physical entry.
    pub(super) fn materialize_texture_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_index: u64,
        handle: TextureHandle,
        image: &Image,
        spec: GpuImageSpec,
        texture_version: u32,
        texture_identity: u64,
        image_version: u32,
        image_identity: u64,
    ) -> PhysicalImageId {
        let previous_lease = self.texture_image_leases.get(handle).copied();
        let current_physical = previous_lease.and_then(|lease| {
            self.physical_images
                .get(lease.physical_id)
                .filter(|entry| entry.spec == spec)
                .map(|entry| {
                    debug_assert!(entry.members.contains(&handle));
                    lease.physical_id
                })
        });

        let (physical_id, attach_target) = if let Some(physical_id) = current_physical {
            let replaced = self.physical_image_index.insert(spec, physical_id);
            debug_assert!(replaced.is_none_or(|old| old == physical_id));
            (physical_id, false)
        } else {
            let indexed = self.physical_image_index.get(&spec).copied();
            let pooled = indexed.filter(|physical_id| {
                self.physical_images
                    .get(*physical_id)
                    .is_some_and(|entry| entry.spec == spec && !entry.members.is_empty())
            });
            if indexed.is_some() && pooled.is_none() {
                self.physical_image_index.remove(&spec);
            }

            let physical_id = if let Some(physical_id) = pooled {
                physical_id
            } else {
                let mut gpu_image =
                    GpuImage::new(device, queue, image, spec, image_version, image_identity);
                gpu_image.last_used_frame = frame_index;
                let physical_id = self.physical_images.insert(PhysicalImageEntry {
                    spec,
                    image: gpu_image,
                    members: SmallVec::new(),
                });
                let replaced = self.physical_image_index.insert(spec, physical_id);
                debug_assert!(replaced.is_none());
                physical_id
            };
            (physical_id, true)
        };

        let generation_changed = {
            let entry = self
                .physical_images
                .get_mut(physical_id)
                .expect("texture lease target is missing its physical image");
            let generation_before = entry.image.generation_id;
            entry
                .image
                .update_content(queue, image, image_version, image_identity);
            entry.image.last_used_frame = frame_index;
            entry.image.generation_id != generation_before
        };
        if generation_changed {
            self.discard_mipmap_request_for_physical(physical_id);
        }

        if attach_target {
            let entry = self
                .physical_images
                .get_mut(physical_id)
                .expect("texture lease target disappeared before publication");
            debug_assert!(!entry.members.contains(&handle));
            entry.members.push(handle);
            self.texture_bindings.remove(handle);
            if let Some(previous) = previous_lease {
                self.detach_physical_member(previous.physical_id, handle);
            }
        }

        self.texture_image_leases.insert(
            handle,
            TextureImageLease {
                physical_id,
                image_handle: spec.image_handle,
                texture_version,
                texture_identity,
                image_version,
                image_identity,
                last_used_frame: frame_index,
            },
        );
        self.latest_texture_for_image
            .insert(spec.image_handle, handle);
        if let Some(previous_image_handle) = previous_lease.map(|lease| lease.image_handle)
            && previous_image_handle != spec.image_handle
        {
            self.repair_texture_image_alias(previous_image_handle);
        }
        physical_id
    }

    /// Release one logical member without disturbing work still required by
    /// another lease. The allocation and pending mip generation are removed
    /// only when the member list becomes empty.
    fn detach_physical_member(&mut self, physical_id: PhysicalImageId, handle: TextureHandle) {
        let Some(entry) = self.physical_images.get_mut(physical_id) else {
            self.physical_image_index
                .retain(|_, indexed| *indexed != physical_id);
            self.discard_mipmap_request_for_physical(physical_id);
            return;
        };
        let Some(member_index) = entry.members.iter().position(|member| *member == handle) else {
            debug_assert!(false, "texture lease missing from physical member list");
            return;
        };
        entry.members.swap_remove(member_index);
        if !entry.members.is_empty() {
            return;
        }

        let spec = entry.spec;
        self.discard_mipmap_request_for_physical(physical_id);
        if self.physical_image_index.get(&spec) == Some(&physical_id) {
            self.physical_image_index.remove(&spec);
        }
        let removed = self
            .physical_images
            .remove(physical_id)
            .expect("empty physical image disappeared during detach");
        debug_assert!(removed.members.is_empty());
    }

    pub(super) fn touch_texture_image(&mut self, handle: TextureHandle, frame_index: u64) -> bool {
        let Some(lease) = self.texture_image_leases.get_mut(handle) else {
            return false;
        };
        lease.last_used_frame = frame_index;
        if let Some(entry) = self.physical_images.get_mut(lease.physical_id) {
            entry.image.last_used_frame = frame_index;
            true
        } else {
            false
        }
    }

    #[must_use]
    pub(super) fn texture_mipmap_request(&self, handle: TextureHandle) -> Option<MipmapRequest> {
        let lease = self.texture_image_leases.get(handle)?;
        self.physical_images
            .get(lease.physical_id)?
            .image
            .mipmap_request(lease.physical_id)
    }

    #[must_use]
    pub(super) fn texture_generation_key(
        &self,
        handle: TextureHandle,
    ) -> Option<PhysicalImageGenerationKey> {
        let lease = self.texture_image_leases.get(handle)?;
        Some(
            self.physical_images
                .get(lease.physical_id)?
                .image
                .generation_key(lease.physical_id),
        )
    }

    /// Return whether the exact shared physical content selected by a CPU
    /// snapshot already has a submitted mip chain.
    #[must_use]
    pub(super) fn physical_mipmaps_submitted_for_spec(
        &self,
        spec: GpuImageSpec,
        image_version: u32,
        image_identity: u64,
    ) -> bool {
        self.physical_image_index
            .get(&spec)
            .and_then(|&physical_id| self.physical_images.get(physical_id))
            .is_some_and(|entry| {
                entry.spec == spec
                    && !entry.members.is_empty()
                    && entry.image.image_version == image_version
                    && entry.image.image_identity == image_identity
                    && entry.image.mipmaps_submitted()
            })
    }

    #[inline]
    pub(super) fn remove_texture_image(&mut self, handle: TextureHandle) {
        self.texture_bindings.remove(handle);
        if let Some(lease) = self.texture_image_leases.remove(handle) {
            self.detach_physical_member(lease.physical_id, handle);
            self.repair_texture_image_alias(lease.image_handle);
        }
    }

    fn repair_texture_image_alias(&mut self, image_handle: ImageHandle) {
        if self
            .latest_texture_for_image
            .get(image_handle)
            .and_then(|texture| self.texture_image(*texture))
            .is_some_and(|image| image.image_handle == image_handle)
        {
            return;
        }

        self.latest_texture_for_image.remove(image_handle);
        if let Some((texture_handle, _)) = self
            .texture_image_leases
            .iter()
            .filter(|(texture, _)| {
                self.texture_image(*texture)
                    .is_some_and(|image| image.image_handle == image_handle)
            })
            .max_by_key(|(_, lease)| lease.last_used_frame)
        {
            self.latest_texture_for_image
                .insert(image_handle, texture_handle);
        }
    }

    /// Compatibility lookup for legacy call sites that only know the CPU image.
    #[inline]
    #[must_use]
    pub(super) fn latest_image(&self, handle: ImageHandle) -> Option<&GpuImage> {
        self.latest_texture_for_image
            .get(handle)
            .and_then(|texture| self.texture_image(*texture))
            .filter(|image| image.image_handle == handle)
            .or_else(|| {
                self.texture_image_leases.iter().find_map(|(texture, _)| {
                    self.texture_image(texture)
                        .filter(|image| image.image_handle == handle)
                })
            })
    }

    #[inline]
    #[must_use]
    pub(super) fn texture_binding(&self, handle: TextureHandle) -> Option<&TextureBinding> {
        let binding = self.texture_bindings.get(handle)?;
        let lease = self.texture_image_leases.get(handle).copied()?;
        self.physical_images
            .get(lease.physical_id)
            .filter(|entry| binding.matches_residency(lease, &entry.image))
            .map(|_| binding)
    }

    #[inline]
    pub(super) fn insert_texture_binding(
        &mut self,
        handle: TextureHandle,
        binding: TextureBinding,
    ) {
        self.latest_texture_for_image
            .insert(binding.image_handle, handle);
        self.texture_bindings.insert(handle, binding);
    }

    #[inline]
    pub(super) fn touch_texture_image_alias(
        &mut self,
        image_handle: ImageHandle,
        texture_handle: TextureHandle,
    ) {
        self.latest_texture_for_image
            .insert(image_handle, texture_handle);
    }

    #[cfg(test)]
    #[must_use]
    pub(super) fn latest_texture_alias(&self, image_handle: ImageHandle) -> Option<TextureHandle> {
        self.latest_texture_for_image.get(image_handle).copied()
    }

    pub(super) fn prune(&mut self, cutoff: u64) {
        self.gpu_geometries
            .retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_materials.retain(|_, material| {
            material
                .as_deref()
                .is_some_and(|material| material.last_used_frame >= cutoff)
        });
        self.gpu_buffers.retain(|_, v| v.last_used_frame >= cutoff);
        self.buffer_index
            .retain(|_, h| self.gpu_buffers.contains_key(*h));
        let stale_textures: Vec<_> = self
            .texture_image_leases
            .iter()
            .filter_map(|(handle, lease)| (lease.last_used_frame < cutoff).then_some(handle))
            .collect();
        for handle in stale_textures {
            self.remove_texture_image(handle);
        }
        let texture_image_leases = &self.texture_image_leases;
        let physical_images = &self.physical_images;
        self.texture_bindings.retain(|handle, binding| {
            texture_image_leases
                .get(handle)
                .and_then(|lease| {
                    physical_images
                        .get(lease.physical_id)
                        .map(|entry| (*lease, &entry.image))
                })
                .is_some_and(|(lease, image)| binding.matches_residency(lease, image))
        });
        self.pending_mipmap_requests.retain(|request| {
            physical_images
                .get(request.physical_id)
                .is_some_and(|entry| {
                    entry.image.generation_key(request.physical_id) == request.key()
                })
        });
        self.pending_mipmap_indices.clear();
        self.pending_mipmap_indices.extend(
            self.pending_mipmap_requests
                .iter()
                .enumerate()
                .map(|(index, request)| (request.physical_id, index)),
        );
        self.latest_texture_for_image.retain(|image, texture| {
            texture_image_leases
                .get(*texture)
                .and_then(|lease| physical_images.get(lease.physical_id))
                .is_some_and(|entry| entry.image.image_handle == image)
        });

        // Preserve the compatibility lookup when its most-recent alias was
        // evicted but another TextureHandle for the same Image remains live.
        let live_images: FxHashSet<_> = self
            .texture_image_leases
            .keys()
            .filter_map(|texture| self.texture_image(texture).map(|image| image.image_handle))
            .collect();
        for image_handle in live_images {
            if self.latest_texture_for_image.contains_key(image_handle) {
                continue;
            }
            if let Some((texture_handle, _)) = self
                .texture_image_leases
                .iter()
                .filter(|(texture, _)| {
                    self.texture_image(*texture)
                        .is_some_and(|image| image.image_handle == image_handle)
                })
                .max_by_key(|(_, lease)| lease.last_used_frame)
            {
                self.latest_texture_for_image
                    .insert(image_handle, texture_handle);
            }
        }
    }

    /// Upload `data` for the buffer identified by `buffer_ref`, creating or
    /// resizing the GPU-side buffer as needed.
    pub(super) fn write_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_index: u64,
        buffer_ref: &BufferRef,
        data: &[u8],
    ) -> (GpuBufferHandle, EnsureResult) {
        let cpu_id = buffer_ref.id();

        if let Some(&handle) = self.buffer_index.get(&cpu_id) {
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
                let mut was_recreated = false;

                if buffer_ref.version > gpu_buf.last_uploaded_version {
                    let old_id = gpu_buf.id;
                    was_recreated = gpu_buf.write_to_gpu(device, queue, data);
                    if !was_recreated && gpu_buf.id != old_id {
                        was_recreated = true;
                    }
                    gpu_buf.last_uploaded_version = buffer_ref.version;
                }
                gpu_buf.last_used_frame = frame_index;
                return (handle, EnsureResult::new(gpu_buf.id, was_recreated));
            }
            self.buffer_index.remove(&cpu_id);
        }

        let mut buf = GpuBuffer::new(device, data, buffer_ref.usage, buffer_ref.label());
        buf.last_uploaded_version = buffer_ref.version;
        buf.last_used_frame = frame_index;
        let phys_id = buf.id;
        let handle = self.gpu_buffers.insert(buf);
        self.buffer_index.insert(cpu_id, handle);
        (handle, EnsureResult::created(phys_id))
    }

    pub(super) fn write_uniform_slot(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_index: u64,
        slot_id: u64,
        data: &[u8],
        label: &str,
    ) -> EnsureResult {
        if let Some(&handle) = self.buffer_index.get(&slot_id) {
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
                let was_recreated = gpu_buf.write_to_gpu(device, queue, data);
                gpu_buf.last_used_frame = frame_index;
                return EnsureResult::new(gpu_buf.id, was_recreated);
            }
            self.buffer_index.remove(&slot_id);
        }

        let mut buf = GpuBuffer::new(
            device,
            data,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some(label),
        );
        buf.last_used_frame = frame_index;
        let phys_id = buf.id;
        let handle = self.gpu_buffers.insert(buf);
        self.buffer_index.insert(slot_id, handle);
        EnsureResult::created(phys_id)
    }

    pub(super) fn touch_buffer_by_cpu_id(&mut self, cpu_id: u64, frame_index: u64) -> bool {
        let Some(&handle) = self.buffer_index.get(&cpu_id) else {
            return false;
        };

        if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
            gpu_buf.last_used_frame = frame_index;
            return true;
        }

        self.buffer_index.remove(&cpu_id);
        false
    }

    #[inline]
    pub(super) fn buffer_mut(&mut self, handle: GpuBufferHandle) -> Option<&mut GpuBuffer> {
        self.gpu_buffers.get_mut(handle)
    }

    #[inline]
    pub(super) fn buffer(&self, handle: GpuBufferHandle) -> Option<&GpuBuffer> {
        self.gpu_buffers.get(handle)
    }

    #[inline]
    pub(super) fn buffer_by_cpu_id(&self, cpu_id: u64) -> Option<&GpuBuffer> {
        self.buffer_index
            .get(&cpu_id)
            .and_then(|&handle| self.gpu_buffers.get(handle))
    }
}

impl ResourceManager {
    /// Build one graph-attempt-private physical/logical Asset Texture
    /// candidate from retained exact CPU owners.
    ///
    /// This is CPU-only and does not mutate the persistent physical pool,
    /// leases, bindings, aliases, residency timestamps, mip state, or sampler
    /// cache.
    pub(crate) fn stage_asset_texture_candidate(
        &mut self,
        batch: &mut AssetTextureCandidateBatch,
        snapshot: AssetTextureSnapshot,
        texture: &Texture,
        image: &Image,
    ) -> AssetTextureCandidateId {
        self.resources
            .stage_asset_texture_candidate(batch, snapshot, texture, image)
    }

    /// Materialize one postflight-current candidate from the retained Image
    /// owner. Same-spec content refreshes update the existing allocation in
    /// place; missing specs allocate only after postflight succeeds.
    pub(crate) fn materialize_asset_texture_candidate(
        &mut self,
        batch: &mut AssetTextureCandidateBatch,
        id: AssetTextureCandidateId,
        image: &Image,
    ) -> bool {
        self.resources.materialize_asset_texture_candidate(
            &self.device,
            &self.queue,
            batch,
            id,
            image,
            self.frame_index,
        )
    }

    /// Publish only selected-live candidates that passed the final batched
    /// asset postflight. Results remain in candidate slots and must be moved
    /// into graph import owners with [`AssetTextureCandidateBatch::take_result`].
    pub(crate) fn publish_asset_texture_candidates(
        &mut self,
        batch: &mut AssetTextureCandidateBatch,
        current: &[bool],
    ) {
        assert!(
            !batch.published,
            "asset texture candidate batch may be published only once"
        );
        assert_eq!(
            batch.logical.len(),
            current.len(),
            "asset texture postflight bits must match staged logical candidates"
        );

        for physical in &mut batch.physical {
            physical.first_live_member = None;
            physical.pre_attached_member = None;
        }

        // Sampler cache publication is deliberately delayed until after the
        // T -> I -> T postflight. Resolve all accepted descriptors before any
        // physical pool mutation so a creation failure leaves the pool intact.
        for (index, candidate) in batch.logical.iter_mut().enumerate() {
            if !current[index] {
                continue;
            }
            let Some(physical) = candidate.physical else {
                continue;
            };
            let ready = batch.physical[physical].target_state.is_some()
                || matches!(
                    &batch.physical[physical].source,
                    PhysicalAssetTextureCandidateSource::New {
                        image_slot: Some(_)
                    }
                );
            if !ready {
                continue;
            }
            candidate.sampler_id = Some(
                self.sampler_registry
                    .get_custom(&self.device, &candidate.sampler)
                    .0,
            );
            batch.physical[physical].first_live_member = Some(candidate.snapshot.texture_handle);
        }

        self.resources
            .publish_asset_texture_candidates(batch, current, self.frame_index);
        batch.published = true;
    }
}

#[cfg(test)]
mod candidate_tests {
    use super::*;
    use crate::core::WgpuContext;
    use crate::core::gpu::texture::snapshot_asset_texture;
    use crate::settings::{RendererInitConfig, RendererSettings};
    use myth_assets::AssetServer;
    use myth_resources::image::{ImageDimension, PixelFormat};

    fn image_8x8(byte: u8) -> Image {
        Image::new(
            8,
            8,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(vec![byte; 8 * 8 * 4]),
        )
    }

    #[test]
    fn empty_candidate_capacity_is_reusable_and_trimmable() {
        let mut batch = AssetTextureCandidateBatch::new();
        batch.physical.reserve(ASSET_TEXTURE_CANDIDATE_BASELINE * 8);
        batch.logical.reserve(ASSET_TEXTURE_CANDIDATE_BASELINE * 8);
        batch
            .private_images
            .reserve(ASSET_TEXTURE_CANDIDATE_BASELINE * 8);
        batch
            .physical_lookup
            .reserve(ASSET_TEXTURE_CANDIDATE_BASELINE * 8);

        assert!(batch.is_empty());
        assert_eq!(batch.active_bytes(), 0);
        let high_watermark = batch.retained_bytes();
        assert!(high_watermark > AssetTextureCandidateBatch::baseline_bytes());

        batch.trim_capacity(AssetTextureCandidateBatch::baseline_bytes(), high_watermark);

        assert!(batch.is_empty());
        assert_eq!(batch.active_bytes(), 0);
        assert!(batch.retained_bytes() < high_watermark);
        assert!(batch.retained_bytes() >= AssetTextureCandidateBatch::baseline_bytes());
    }

    #[test]
    fn single_mip_candidate_cannot_overwrite_a_newer_cache_generation() {
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image_handle = assets.images.add(image_8x8(1));
        let texture_handle = assets
            .textures
            .add(Texture::new_2d(Some("single mip CAS"), image_handle));

        let first = snapshot_asset_texture(&assets, texture_handle).unwrap();
        assert_eq!(first.mip_count, 1);
        let first_materialized = resources.materialize_asset_texture_for_graph(&assets, first);
        assert!(first_materialized.texture.is_some());
        let first_generation = resources
            .resources
            .texture_generation_key(texture_handle)
            .unwrap();

        assets.images.insert_ready(image_handle, image_8x8(2));
        let (texture_owner, texture_version, texture_identity) = assets
            .textures
            .get_entry_with_identity(texture_handle)
            .unwrap();
        let (image_owner, image_version, image_identity) =
            assets.images.get_entry_with_identity(image_handle).unwrap();
        let second = AssetTextureSnapshot::from_loaded(
            texture_handle,
            &texture_owner,
            texture_version,
            texture_identity,
            &image_owner,
            image_version,
            image_identity,
        )
        .unwrap();
        let mut candidates = AssetTextureCandidateBatch::new();
        let staged = resources.stage_asset_texture_candidate(
            &mut candidates,
            second,
            &texture_owner,
            &image_owner,
        );

        assets.images.insert_ready(image_handle, image_8x8(3));
        let third = snapshot_asset_texture(&assets, texture_handle).unwrap();
        assert_ne!(second.image_identity, third.image_identity);
        let third_materialized = resources.materialize_asset_texture_for_graph(&assets, third);
        assert!(third_materialized.texture.is_some());
        let third_generation = resources
            .resources
            .texture_generation_key(texture_handle)
            .unwrap();
        assert_ne!(first_generation, third_generation);

        resources.materialize_asset_texture_candidate(&mut candidates, staged, &image_owner);
        resources.publish_asset_texture_candidates(&mut candidates, &[true]);
        let stale_result = candidates.take_result(staged);
        assert!(stale_result.texture.is_none());
        assert!(stale_result.mipmap_request.is_none());
        assert!(resources.asset_texture_binding_matches_snapshot(third));
        assert!(!resources.asset_texture_binding_matches_snapshot(second));
        assert_eq!(
            resources.resources.texture_generation_key(texture_handle),
            Some(third_generation)
        );
    }

    #[test]
    fn shared_physical_candidate_skips_a_dead_first_member() {
        const TEXTURE_COUNT: usize = 130;
        let wgpu = pollster::block_on(WgpuContext::new_headless(
            &RendererInitConfig::default(),
            &RendererSettings::default(),
            8,
            8,
            None,
        ))
        .expect("headless GPU context required");
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let assets = AssetServer::new();
        let image_handle = assets.images.add(image_8x8(7));
        let (image_owner, image_version, image_identity) =
            assets.images.get_entry_with_identity(image_handle).unwrap();
        let mut candidates = AssetTextureCandidateBatch::new();
        let mut staged = Vec::with_capacity(TEXTURE_COUNT);
        let mut handles = Vec::with_capacity(TEXTURE_COUNT);

        for _ in 0..TEXTURE_COUNT {
            let handle = assets
                .textures
                .add(Texture::new_2d(Some("shared candidate"), image_handle));
            let (texture_owner, texture_version, texture_identity) =
                assets.textures.get_entry_with_identity(handle).unwrap();
            let snapshot = AssetTextureSnapshot::from_loaded(
                handle,
                &texture_owner,
                texture_version,
                texture_identity,
                &image_owner,
                image_version,
                image_identity,
            )
            .unwrap();
            staged.push(resources.stage_asset_texture_candidate(
                &mut candidates,
                snapshot,
                &texture_owner,
                &image_owner,
            ));
            handles.push(handle);
        }
        assert_eq!(candidates.physical.len(), 1);

        let mut current = vec![true; TEXTURE_COUNT];
        current[0] = false;
        for (&candidate, &is_current) in staged.iter().zip(&current) {
            if is_current {
                resources.materialize_asset_texture_candidate(
                    &mut candidates,
                    candidate,
                    &image_owner,
                );
            }
        }
        resources.publish_asset_texture_candidates(&mut candidates, &current);
        for (index, candidate) in staged.into_iter().enumerate() {
            assert_eq!(
                candidates.take_result(candidate).texture.is_some(),
                index != 0
            );
        }

        let physical = resources
            .resources
            .texture_image_lease(handles[1])
            .unwrap()
            .physical_id;
        assert_eq!(
            resources.resources.physical_image_member_count(physical),
            Some(TEXTURE_COUNT - 1)
        );
        assert!(resources.get_texture_binding(handles[0]).is_none());
        assert!(
            handles
                .iter()
                .skip(1)
                .all(|&handle| resources.get_texture_binding(handle).is_some())
        );
        resources.resources.assert_texture_pool_consistent();
    }
}
