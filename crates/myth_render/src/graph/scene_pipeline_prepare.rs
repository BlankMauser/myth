//! Retained serial planning for CPU-direct scene pipeline requests.
//!
//! Gather writes only Copy recipes and exact cache keys. Planning restores the
//! canonical semantic order and assigns one deterministic owner to each exact
//! request. Typed coordinators fill pipeline results later; finalization reads
//! those immutable results and emits the backend command stream.

use std::{num::NonZeroU32, ops::Range};

use myth_assets::{GeometryHandle, MaterialHandle};
use rustc_hash::FxHashMap;

use crate::{
    core::{BindingInterfaceId, gpu::Tracked},
    pipeline::{FastPipelineKey, FastShadowPipelineKey, RenderPipelineId},
};

use super::{
    asset_frame::GeometryFrameId,
    capacity::{
        CapacityUsage, estimated_hash_map_footprint, shrink_hash_map_by_family_ratio,
        shrink_vec_by_family_ratio, vec_footprint,
    },
    frame::{DrawBindGroupSlot, DrawPlanId, FrameSourceId, PreparedDrawSource, PreparedSourceId},
};

const PIPELINE_REQUEST_BASELINE: usize = 256;
const SHADER_IDENTITY_BASELINE: usize = 256;
const DRAW_RECIPE_BASELINE: usize = 1024;
const DRAW_PLAN_RANGE_BASELINE: usize = 16;
const SHADOW_LAYOUT_BASELINE: usize = 1;
const PIPELINE_SOURCE_BASELINE: usize = DRAW_RECIPE_BASELINE;

macro_rules! request_id {
    ($name:ident, $message:literal) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(crate) struct $name(NonZeroU32);

        impl $name {
            #[inline]
            fn from_index(index: usize) -> Self {
                let raw = u32::try_from(index)
                    .ok()
                    .and_then(|index| index.checked_add(1))
                    .and_then(NonZeroU32::new)
                    .expect($message);
                Self(raw)
            }

            #[inline]
            const fn index(self) -> usize {
                self.0.get() as usize - 1
            }
        }
    };
}

request_id!(
    MainPipelineRequestId,
    "main pipeline request index exceeds u32::MAX - 1"
);
request_id!(
    PrepassPipelineRequestId,
    "prepass pipeline request index exceeds u32::MAX - 1"
);
request_id!(
    ShadowPipelineRequestId,
    "shadow pipeline request index exceeds u32::MAX - 1"
);
request_id!(
    ShadowPipelineLayoutId,
    "shadow pipeline layout index exceeds u32::MAX - 1"
);

/// Canonical ordering carried through worker-local permutations in M8.
///
/// `plan_index` is the DrawPlan reservation ordinal; `intent_index` is the
/// source's position in that plan's immutable RenderIntent list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SceneRecipeOrdinal {
    pub(crate) plan_index: u32,
    pub(crate) intent_index: u32,
}

impl SceneRecipeOrdinal {
    #[must_use]
    pub(crate) fn new(plan_index: usize, intent_index: usize) -> Self {
        Self {
            plan_index: u32::try_from(plan_index).expect("draw-plan index exceeds u32"),
            intent_index: u32::try_from(intent_index).expect("draw-intent index exceeds u32"),
        }
    }
}

/// Feature-owned Prepass L1 key. It is Copy and contains no WGPU owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PrepassPipelineL1Key {
    pub(crate) geometry_handle: GeometryHandle,
    pub(crate) geometry_asset_version: u32,
    pub(crate) geometry_asset_identity: u64,
    pub(crate) geometry_layout_version: u64,
    pub(crate) material_handle: MaterialHandle,
    pub(crate) material_asset_version: u32,
    pub(crate) material_asset_identity: u64,
    pub(crate) material_version: u64,
    pub(crate) item_variants: u32,
    pub(crate) item_shader_features: super::extracted::ItemShaderFeatures,
    pub(crate) shader_source_identity: u64,
    pub(crate) scene_defines_hash: u64,
    pub(crate) pipeline_settings_version: u64,
    pub(crate) global_binding_interface_id: BindingInterfaceId,
    pub(crate) material_binding_interface_id: BindingInterfaceId,
    pub(crate) object_binding_interface_id: BindingInterfaceId,
    pub(crate) vertex_layout_id: u64,
    pub(crate) bind_group_layout_ids: [u64; 3],
    pub(crate) depth_format: wgpu::TextureFormat,
    pub(crate) topology: wgpu::PrimitiveTopology,
    pub(crate) cull_mode: Option<wgpu::Face>,
    pub(crate) front_face: wgpu::FrontFace,
    pub(crate) needs_normal: bool,
    pub(crate) needs_feature_id: bool,
    pub(crate) needs_velocity: bool,
}

/// Exact material generation used to amortize shader-source identity hashing.
///
/// Inline templates may be large, so hashing them once per draw would turn a
/// shared material into `O(draws * WGSL bytes)` work. This retained scratch is
/// attempt-local and stores no asset or WGPU owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MaterialShaderIdentityKey {
    handle: MaterialHandle,
    asset_version: u32,
    asset_identity: u64,
    material_version: u64,
}

#[derive(Debug, Clone, Copy)]
struct MainPipelineRequest {
    key: FastPipelineKey,
    owner: PreparedSourceId,
    owner_ordinal: SceneRecipeOrdinal,
    result: Option<RenderPipelineId>,
    publish_l1: bool,
}

#[derive(Debug, Clone, Copy)]
struct PrepassPipelineRequest {
    key: PrepassPipelineL1Key,
    owner: PreparedSourceId,
    owner_ordinal: SceneRecipeOrdinal,
    result: Option<RenderPipelineId>,
    publish_l1: bool,
}

#[derive(Debug, Clone, Copy)]
struct ShadowPipelineRequest {
    key: FastShadowPipelineKey,
    owner: PreparedSourceId,
    owner_ordinal: SceneRecipeOrdinal,
    layout: ShadowPipelineLayoutId,
    result: Option<RenderPipelineId>,
    publish_l1: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MainDrawRecipe {
    pub(crate) draw_plan: DrawPlanId,
    pub(crate) ordinal: SceneRecipeOrdinal,
    pub(crate) prepared_source: PreparedSourceId,
    pub(crate) pipeline: MainPipelineRequestId,
    pub(crate) material_index: u32,
    pub(crate) distance_sq: f32,
    pub(crate) transparent: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrepassDrawRecipe {
    pub(crate) draw_plan: DrawPlanId,
    pub(crate) ordinal: SceneRecipeOrdinal,
    pub(crate) prepared_source: PreparedSourceId,
    pub(crate) pipeline: PrepassPipelineRequestId,
    pub(crate) material_index: u32,
    pub(crate) distance_sq: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ShadowDrawRecipe {
    pub(crate) draw_plan: DrawPlanId,
    pub(crate) ordinal: SceneRecipeOrdinal,
    pub(crate) prepared_source: PreparedSourceId,
    pub(crate) pipeline: ShadowPipelineRequestId,
}

/// Final graph-attempt draw payload accepted at the pipeline epoch gate.
///
/// The gate is the linearization point for one scene pipeline transaction.
/// Past it, Direct finalization must consume only this Copy record and the
/// immutable geometry-physical store; it must never reopen a live asset epoch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AcceptedPreparedSource {
    pub(crate) frame_source: FrameSourceId,
    pub(crate) geometry: GeometryFrameId,
    pub(crate) material_bind_group: DrawBindGroupSlot,
    pub(crate) object_bind_group: (DrawBindGroupSlot, u32),
    pub(crate) commits_morph_history: bool,
    /// Preserve the established Shadow sort key without reopening Material.
    pub(crate) material_bind_group_id: u64,
}

impl AcceptedPreparedSource {
    #[must_use]
    pub(crate) const fn from_prepared(
        source: &PreparedDrawSource,
        material_bind_group_id: u64,
    ) -> Self {
        Self {
            frame_source: source.frame_source,
            geometry: source.geometry,
            material_bind_group: source.material_bind_group,
            object_bind_group: source.object_bind_group,
            commits_morph_history: source.commits_morph_history,
            material_bind_group_id,
        }
    }
}

/// Retained Gather/Plan scratch plus graph-attempt layout owners.
pub(crate) struct ScenePipelinePrepareStore {
    shader_identities: FxHashMap<MaterialShaderIdentityKey, u64>,
    main_requests: Vec<MainPipelineRequest>,
    main_lookup: FxHashMap<FastPipelineKey, MainPipelineRequestId>,
    main_order: Vec<MainPipelineRequestId>,
    prepass_requests: Vec<PrepassPipelineRequest>,
    prepass_lookup: FxHashMap<PrepassPipelineL1Key, PrepassPipelineRequestId>,
    prepass_order: Vec<PrepassPipelineRequestId>,
    shadow_requests: Vec<ShadowPipelineRequest>,
    shadow_lookup: FxHashMap<FastShadowPipelineKey, ShadowPipelineRequestId>,
    shadow_order: Vec<ShadowPipelineRequestId>,
    main_draws: Vec<MainDrawRecipe>,
    prepass_draws: Vec<PrepassDrawRecipe>,
    shadow_draws: Vec<ShadowDrawRecipe>,
    main_ranges: Vec<Range<u32>>,
    prepass_ranges: Vec<Range<u32>>,
    shadow_ranges: Vec<Range<u32>>,
    shadow_layouts: Vec<Tracked<wgpu::BindGroupLayout>>,
    shadow_layout_lookup: FxHashMap<u64, ShadowPipelineLayoutId>,
    /// Unique sources referenced by at least one gathered recipe. Stamped
    /// joins avoid an O(high-water) clear on every warm frame.
    participating_sources: Vec<PreparedSourceId>,
    participating_source_stamps: Vec<u32>,
    accepted_sources: Vec<AcceptedPreparedSource>,
    accepted_source_indices: Vec<u32>,
    accepted_source_stamps: Vec<u32>,
    source_join_generation: u32,
    source_domain_len: usize,
    source_gate_finalized: bool,
    planned: bool,
}

impl Default for ScenePipelinePrepareStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ScenePipelinePrepareStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            shader_identities: FxHashMap::default(),
            main_requests: Vec::with_capacity(PIPELINE_REQUEST_BASELINE),
            main_lookup: FxHashMap::default(),
            main_order: Vec::with_capacity(PIPELINE_REQUEST_BASELINE),
            prepass_requests: Vec::with_capacity(PIPELINE_REQUEST_BASELINE),
            prepass_lookup: FxHashMap::default(),
            prepass_order: Vec::with_capacity(PIPELINE_REQUEST_BASELINE),
            shadow_requests: Vec::with_capacity(PIPELINE_REQUEST_BASELINE),
            shadow_lookup: FxHashMap::default(),
            shadow_order: Vec::with_capacity(PIPELINE_REQUEST_BASELINE),
            main_draws: Vec::with_capacity(DRAW_RECIPE_BASELINE),
            prepass_draws: Vec::with_capacity(DRAW_RECIPE_BASELINE),
            shadow_draws: Vec::with_capacity(DRAW_RECIPE_BASELINE),
            main_ranges: Vec::with_capacity(DRAW_PLAN_RANGE_BASELINE),
            prepass_ranges: Vec::with_capacity(DRAW_PLAN_RANGE_BASELINE),
            shadow_ranges: Vec::with_capacity(DRAW_PLAN_RANGE_BASELINE),
            shadow_layouts: Vec::with_capacity(SHADOW_LAYOUT_BASELINE),
            shadow_layout_lookup: FxHashMap::default(),
            participating_sources: Vec::with_capacity(PIPELINE_SOURCE_BASELINE),
            participating_source_stamps: Vec::with_capacity(PIPELINE_SOURCE_BASELINE),
            accepted_sources: Vec::with_capacity(PIPELINE_SOURCE_BASELINE),
            accepted_source_indices: Vec::with_capacity(PIPELINE_SOURCE_BASELINE),
            accepted_source_stamps: Vec::with_capacity(PIPELINE_SOURCE_BASELINE),
            source_join_generation: 1,
            source_domain_len: 0,
            source_gate_finalized: false,
            planned: false,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.shader_identities.clear();
        self.main_requests.clear();
        self.main_lookup.clear();
        self.main_order.clear();
        self.prepass_requests.clear();
        self.prepass_lookup.clear();
        self.prepass_order.clear();
        self.shadow_requests.clear();
        self.shadow_lookup.clear();
        self.shadow_order.clear();
        self.main_draws.clear();
        self.prepass_draws.clear();
        self.shadow_draws.clear();
        self.main_ranges.clear();
        self.prepass_ranges.clear();
        self.shadow_ranges.clear();
        self.shadow_layout_lookup.clear();
        self.shadow_layouts.clear();
        self.participating_sources.clear();
        self.accepted_sources.clear();
        self.source_domain_len = 0;
        self.source_gate_finalized = false;
        self.advance_source_join_generation();
        self.planned = false;
    }

    #[inline]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.shader_identities.is_empty()
            && self.main_requests.is_empty()
            && self.main_lookup.is_empty()
            && self.main_order.is_empty()
            && self.prepass_requests.is_empty()
            && self.prepass_lookup.is_empty()
            && self.prepass_order.is_empty()
            && self.shadow_requests.is_empty()
            && self.shadow_lookup.is_empty()
            && self.shadow_order.is_empty()
            && self.main_draws.is_empty()
            && self.prepass_draws.is_empty()
            && self.shadow_draws.is_empty()
            && self.main_ranges.is_empty()
            && self.prepass_ranges.is_empty()
            && self.shadow_ranges.is_empty()
            && self.shadow_layouts.is_empty()
            && self.shadow_layout_lookup.is_empty()
            && self.participating_sources.is_empty()
            && self.accepted_sources.is_empty()
            && self.source_domain_len == 0
            && !self.source_gate_finalized
            && !self.planned
    }

    #[must_use]
    pub(crate) fn capacity_usage(&self) -> CapacityUsage {
        estimated_hash_map_footprint(&self.shader_identities, SHADER_IDENTITY_BASELINE)
            + vec_footprint(&self.main_requests, PIPELINE_REQUEST_BASELINE)
            + estimated_hash_map_footprint(&self.main_lookup, PIPELINE_REQUEST_BASELINE)
            + vec_footprint(&self.main_order, PIPELINE_REQUEST_BASELINE)
            + vec_footprint(&self.prepass_requests, PIPELINE_REQUEST_BASELINE)
            + estimated_hash_map_footprint(&self.prepass_lookup, PIPELINE_REQUEST_BASELINE)
            + vec_footprint(&self.prepass_order, PIPELINE_REQUEST_BASELINE)
            + vec_footprint(&self.shadow_requests, PIPELINE_REQUEST_BASELINE)
            + estimated_hash_map_footprint(&self.shadow_lookup, PIPELINE_REQUEST_BASELINE)
            + vec_footprint(&self.shadow_order, PIPELINE_REQUEST_BASELINE)
            + vec_footprint(&self.main_draws, DRAW_RECIPE_BASELINE)
            + vec_footprint(&self.prepass_draws, DRAW_RECIPE_BASELINE)
            + vec_footprint(&self.shadow_draws, DRAW_RECIPE_BASELINE)
            + vec_footprint(&self.main_ranges, DRAW_PLAN_RANGE_BASELINE)
            + vec_footprint(&self.prepass_ranges, DRAW_PLAN_RANGE_BASELINE)
            + vec_footprint(&self.shadow_ranges, DRAW_PLAN_RANGE_BASELINE)
            + vec_footprint(&self.shadow_layouts, SHADOW_LAYOUT_BASELINE)
            + estimated_hash_map_footprint(&self.shadow_layout_lookup, SHADOW_LAYOUT_BASELINE)
            + vec_footprint(&self.participating_sources, PIPELINE_SOURCE_BASELINE)
            + vec_footprint(&self.accepted_sources, PIPELINE_SOURCE_BASELINE)
            + source_join_footprint(
                self.source_domain_len,
                &self.participating_source_stamps,
                &self.accepted_source_indices,
                &self.accepted_source_stamps,
            )
    }

    pub(crate) fn trim_capacity(&mut self, target_bytes: usize, retained_bytes: usize) {
        assert!(
            self.is_empty(),
            "live scene-pipeline recipes and layout owners cannot be trimmed"
        );
        shrink_hash_map_by_family_ratio(
            &mut self.shader_identities,
            SHADER_IDENTITY_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.main_requests,
            PIPELINE_REQUEST_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.main_lookup,
            PIPELINE_REQUEST_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.main_order,
            PIPELINE_REQUEST_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.prepass_requests,
            PIPELINE_REQUEST_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.prepass_lookup,
            PIPELINE_REQUEST_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.prepass_order,
            PIPELINE_REQUEST_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.shadow_requests,
            PIPELINE_REQUEST_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.shadow_lookup,
            PIPELINE_REQUEST_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.shadow_order,
            PIPELINE_REQUEST_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.main_draws,
            DRAW_RECIPE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.prepass_draws,
            DRAW_RECIPE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.shadow_draws,
            DRAW_RECIPE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        for ranges in [
            &mut self.main_ranges,
            &mut self.prepass_ranges,
            &mut self.shadow_ranges,
        ] {
            shrink_vec_by_family_ratio(
                ranges,
                DRAW_PLAN_RANGE_BASELINE,
                target_bytes,
                retained_bytes,
            );
        }
        shrink_vec_by_family_ratio(
            &mut self.shadow_layouts,
            SHADOW_LAYOUT_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_hash_map_by_family_ratio(
            &mut self.shadow_layout_lookup,
            SHADOW_LAYOUT_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.participating_sources,
            PIPELINE_SOURCE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        shrink_vec_by_family_ratio(
            &mut self.accepted_sources,
            PIPELINE_SOURCE_BASELINE,
            target_bytes,
            retained_bytes,
        );
        trim_source_join(
            &mut self.participating_source_stamps,
            &mut self.accepted_source_indices,
            &mut self.accepted_source_stamps,
            target_bytes,
            retained_bytes,
        );
    }

    pub(crate) fn intern_shadow_layout(
        &mut self,
        layout: &Tracked<wgpu::BindGroupLayout>,
    ) -> ShadowPipelineLayoutId {
        assert!(
            !self.planned,
            "shadow layouts must be registered before planning"
        );
        if let Some(&id) = self.shadow_layout_lookup.get(&layout.id()) {
            return id;
        }
        let id = ShadowPipelineLayoutId::from_index(self.shadow_layouts.len());
        self.shadow_layouts.push(layout.clone());
        assert!(self.shadow_layout_lookup.insert(layout.id(), id).is_none());
        id
    }

    /// Return one stable shader-source identity per exact material generation.
    ///
    /// `derive` runs only on a cold entry. The exact key makes hash collisions
    /// fall through to structural equality and keeps hot shared-material draws
    /// allocation-free after the retained map reaches its working capacity.
    pub(crate) fn material_shader_source_identity(
        &mut self,
        handle: MaterialHandle,
        asset_version: u32,
        asset_identity: u64,
        material_version: u64,
        derive: impl FnOnce() -> u64,
    ) -> u64 {
        let key = MaterialShaderIdentityKey {
            handle,
            asset_version,
            asset_identity,
            material_version,
        };
        *self.shader_identities.entry(key).or_insert_with(derive)
    }

    #[must_use]
    pub(crate) fn shadow_layout(
        &self,
        id: ShadowPipelineLayoutId,
    ) -> &Tracked<wgpu::BindGroupLayout> {
        &self.shadow_layouts[id.index()]
    }

    #[must_use]
    pub(crate) fn registered_shadow_layout(&self) -> Option<ShadowPipelineLayoutId> {
        match self.shadow_layouts.len() {
            0 => None,
            1 => Some(ShadowPipelineLayoutId::from_index(0)),
            _ => panic!("one scene attempt registered conflicting Shadow Group 0 layouts"),
        }
    }

    fn advance_source_join_generation(&mut self) {
        self.source_join_generation = self.source_join_generation.wrapping_add(1);
        if self.source_join_generation == 0 {
            // The only O(high-water) clear happens once per u32 wrap. This
            // keeps warm-frame clear independent of the retained peak domain.
            self.participating_source_stamps.fill(0);
            self.accepted_source_stamps.fill(0);
            self.source_join_generation = 1;
        }
    }

    fn mark_participating_source(&mut self, source: PreparedSourceId) {
        assert!(!self.planned, "cannot register sources after planning");
        let required = source.index() + 1;
        if self.participating_source_stamps.len() < required {
            self.participating_source_stamps.resize(required, 0);
            self.accepted_source_indices.resize(required, 0);
            self.accepted_source_stamps.resize(required, 0);
        }
        if self.participating_source_stamps[source.index()] == self.source_join_generation {
            return;
        }
        self.source_domain_len = self.source_domain_len.max(required);
        self.participating_source_stamps[source.index()] = self.source_join_generation;
        self.participating_sources.push(source);
    }

    #[must_use]
    pub(crate) fn participating_source_count(&self) -> usize {
        assert!(self.planned, "source epoch gate requires planned recipes");
        self.participating_sources.len()
    }

    #[must_use]
    pub(crate) fn participating_source_at(&self, index: usize) -> PreparedSourceId {
        assert!(self.planned, "source epoch gate requires planned recipes");
        self.participating_sources[index]
    }

    /// Publish one source into the immutable accepted stream. The coordinator
    /// calls this only after the source's single final epoch validation.
    pub(crate) fn accept_source(
        &mut self,
        source: PreparedSourceId,
        accepted: AcceptedPreparedSource,
    ) {
        assert!(self.planned, "cannot accept a source before planning");
        assert!(
            !self.source_gate_finalized,
            "cannot accept a source after the epoch gate"
        );
        assert_eq!(
            self.participating_source_stamps
                .get(source.index())
                .copied(),
            Some(self.source_join_generation),
            "accepted source did not participate in any recipe"
        );
        assert_ne!(
            self.accepted_source_stamps[source.index()],
            self.source_join_generation,
            "one prepared source passed the epoch gate twice"
        );
        let accepted_index = u32::try_from(self.accepted_sources.len())
            .expect("accepted prepared-source index exceeds u32::MAX");
        self.accepted_sources.push(accepted);
        self.accepted_source_indices[source.index()] = accepted_index;
        self.accepted_source_stamps[source.index()] = self.source_join_generation;
    }

    /// Close the final epoch gate and invalidate every request whose
    /// deterministic owner did not enter the accepted stream.
    pub(crate) fn finish_source_gate(&mut self) {
        assert!(
            self.planned,
            "cannot finalize source epochs before planning"
        );
        assert!(
            !self.source_gate_finalized,
            "scene source epoch gate finalized twice"
        );
        let generation = self.source_join_generation;
        let accepted_stamps = &self.accepted_source_stamps;
        for request in &mut self.main_requests {
            if !source_is_stamped(accepted_stamps, generation, request.owner) {
                request.result = None;
                request.publish_l1 = false;
            }
        }
        for request in &mut self.prepass_requests {
            if !source_is_stamped(accepted_stamps, generation, request.owner) {
                request.result = None;
                request.publish_l1 = false;
            }
        }
        for request in &mut self.shadow_requests {
            if !source_is_stamped(accepted_stamps, generation, request.owner) {
                request.result = None;
                request.publish_l1 = false;
            }
        }
        self.source_gate_finalized = true;
    }

    #[inline]
    #[must_use]
    pub(crate) fn accepted_source(
        &self,
        source: PreparedSourceId,
    ) -> Option<&AcceptedPreparedSource> {
        assert!(
            self.source_gate_finalized,
            "accepted stream is unavailable before the final epoch gate"
        );
        if !source_is_stamped(
            &self.accepted_source_stamps,
            self.source_join_generation,
            source,
        ) {
            return None;
        }
        self.accepted_sources
            .get(self.accepted_source_indices[source.index()] as usize)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_main(
        &mut self,
        draw_plan: DrawPlanId,
        ordinal: SceneRecipeOrdinal,
        prepared_source: PreparedSourceId,
        key: FastPipelineKey,
        material_index: u32,
        distance_sq: f32,
        transparent: bool,
    ) {
        assert!(!self.planned, "cannot gather after scene pipeline planning");
        self.mark_participating_source(prepared_source);
        let request = match self.main_lookup.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let id = *entry.get();
                let record = &mut self.main_requests[id.index()];
                if ordinal < record.owner_ordinal {
                    record.owner_ordinal = ordinal;
                    record.owner = prepared_source;
                }
                id
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let id = MainPipelineRequestId::from_index(self.main_requests.len());
                self.main_requests.push(MainPipelineRequest {
                    key,
                    owner: prepared_source,
                    owner_ordinal: ordinal,
                    result: None,
                    publish_l1: false,
                });
                entry.insert(id);
                id
            }
        };
        self.main_draws.push(MainDrawRecipe {
            draw_plan,
            ordinal,
            prepared_source,
            pipeline: request,
            material_index,
            distance_sq,
            transparent,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_prepass_compact(
        &mut self,
        draw_plan: DrawPlanId,
        ordinal: SceneRecipeOrdinal,
        prepared_source: PreparedSourceId,
        key: PrepassPipelineL1Key,
        cached: Option<RenderPipelineId>,
        material_index: u32,
        distance_sq: f32,
    ) {
        assert!(!self.planned, "cannot gather after scene pipeline planning");
        self.mark_participating_source(prepared_source);
        let request = match self.prepass_lookup.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let id = *entry.get();
                let record = &mut self.prepass_requests[id.index()];
                if let Some(cached) = cached {
                    assert!(
                        record.result.is_none_or(|current| current == cached),
                        "one exact Prepass L1 key resolved to conflicting pipeline IDs"
                    );
                    record.result = Some(cached);
                }
                if ordinal < record.owner_ordinal {
                    record.owner_ordinal = ordinal;
                    record.owner = prepared_source;
                }
                id
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let id = PrepassPipelineRequestId::from_index(self.prepass_requests.len());
                self.prepass_requests.push(PrepassPipelineRequest {
                    key,
                    owner: prepared_source,
                    owner_ordinal: ordinal,
                    result: cached,
                    publish_l1: false,
                });
                entry.insert(id);
                id
            }
        };
        self.prepass_draws.push(PrepassDrawRecipe {
            draw_plan,
            ordinal,
            prepared_source,
            pipeline: request,
            material_index,
            distance_sq,
        });
    }

    /// Benchmark/test compatibility wrapper. Production Gather carries only
    /// the accepted dense [`PreparedSourceId`] beyond its single frame join.
    #[cfg(any(test, feature = "rdg_bench_support"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_prepass(
        &mut self,
        draw_plan: DrawPlanId,
        ordinal: SceneRecipeOrdinal,
        _frame_source: FrameSourceId,
        prepared_source: PreparedSourceId,
        key: PrepassPipelineL1Key,
        cached: Option<RenderPipelineId>,
        material_index: u32,
        distance_sq: f32,
    ) {
        self.push_prepass_compact(
            draw_plan,
            ordinal,
            prepared_source,
            key,
            cached,
            material_index,
            distance_sq,
        );
    }

    pub(crate) fn push_shadow(
        &mut self,
        draw_plan: DrawPlanId,
        ordinal: SceneRecipeOrdinal,
        prepared_source: PreparedSourceId,
        key: FastShadowPipelineKey,
        layout: ShadowPipelineLayoutId,
    ) {
        assert!(!self.planned, "cannot gather after scene pipeline planning");
        self.mark_participating_source(prepared_source);
        let request = match self.shadow_lookup.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let id = *entry.get();
                let record = &mut self.shadow_requests[id.index()];
                assert_eq!(
                    record.layout, layout,
                    "one exact Shadow L1 key used conflicting physical layouts"
                );
                if ordinal < record.owner_ordinal {
                    record.owner_ordinal = ordinal;
                    record.owner = prepared_source;
                }
                id
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let id = ShadowPipelineRequestId::from_index(self.shadow_requests.len());
                self.shadow_requests.push(ShadowPipelineRequest {
                    key,
                    owner: prepared_source,
                    owner_ordinal: ordinal,
                    layout,
                    result: None,
                    publish_l1: false,
                });
                entry.insert(id);
                id
            }
        };
        self.shadow_draws.push(ShadowDrawRecipe {
            draw_plan,
            ordinal,
            prepared_source,
            pipeline: request,
        });
    }

    pub(crate) fn plan(&mut self, plan_count: usize) {
        assert!(!self.planned, "scene pipeline recipes were planned twice");
        self.main_order
            .extend((0..self.main_requests.len()).map(MainPipelineRequestId::from_index));
        self.prepass_order
            .extend((0..self.prepass_requests.len()).map(PrepassPipelineRequestId::from_index));
        self.shadow_order
            .extend((0..self.shadow_requests.len()).map(ShadowPipelineRequestId::from_index));
        self.main_order
            .sort_unstable_by_key(|id| self.main_requests[id.index()].owner_ordinal);
        self.prepass_order
            .sort_unstable_by_key(|id| self.prepass_requests[id.index()].owner_ordinal);
        self.shadow_order
            .sort_unstable_by_key(|id| self.shadow_requests[id.index()].owner_ordinal);

        self.main_draws
            .sort_unstable_by_key(|recipe| recipe.ordinal);
        self.prepass_draws
            .sort_unstable_by_key(|recipe| recipe.ordinal);
        self.shadow_draws
            .sort_unstable_by_key(|recipe| recipe.ordinal);
        build_plan_ranges(&self.main_draws, plan_count, &mut self.main_ranges);
        build_plan_ranges(&self.prepass_draws, plan_count, &mut self.prepass_ranges);
        build_plan_ranges(&self.shadow_draws, plan_count, &mut self.shadow_ranges);
        self.planned = true;
    }

    #[must_use]
    pub(crate) fn main_request_count(&self) -> usize {
        self.main_order.len()
    }

    #[must_use]
    pub(crate) fn main_request_at(&self, index: usize) -> MainPipelineRequestId {
        self.main_order[index]
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn prepass_request_order(&self) -> &[PrepassPipelineRequestId] {
        assert!(self.planned, "scene pipeline recipes are not planned");
        &self.prepass_order
    }

    #[must_use]
    pub(crate) fn prepass_request_count(&self) -> usize {
        self.prepass_order.len()
    }

    #[must_use]
    pub(crate) fn prepass_request_at(&self, index: usize) -> PrepassPipelineRequestId {
        self.prepass_order[index]
    }

    #[must_use]
    pub(crate) fn shadow_request_count(&self) -> usize {
        self.shadow_order.len()
    }

    #[must_use]
    pub(crate) fn shadow_request_at(&self, index: usize) -> ShadowPipelineRequestId {
        self.shadow_order[index]
    }

    #[must_use]
    pub(crate) fn main_request(
        &self,
        id: MainPipelineRequestId,
    ) -> (FastPipelineKey, PreparedSourceId, Option<RenderPipelineId>) {
        let request = self.main_requests[id.index()];
        (request.key, request.owner, request.result)
    }

    pub(crate) fn resolve_main_request(
        &mut self,
        id: MainPipelineRequestId,
        result: RenderPipelineId,
        publish_l1: bool,
    ) {
        let request = &mut self.main_requests[id.index()];
        assert!(
            request.result.is_none(),
            "main pipeline request resolved twice"
        );
        request.result = Some(result);
        request.publish_l1 = publish_l1;
    }

    #[must_use]
    pub(crate) fn prepass_request(
        &self,
        id: PrepassPipelineRequestId,
    ) -> (
        PrepassPipelineL1Key,
        PreparedSourceId,
        Option<RenderPipelineId>,
    ) {
        let request = self.prepass_requests[id.index()];
        (request.key, request.owner, request.result)
    }

    pub(crate) fn resolve_prepass_request(
        &mut self,
        id: PrepassPipelineRequestId,
        result: RenderPipelineId,
        publish_l1: bool,
    ) {
        let request = &mut self.prepass_requests[id.index()];
        assert!(
            request.result.is_none(),
            "cold prepass pipeline request resolved twice"
        );
        request.result = Some(result);
        request.publish_l1 = publish_l1;
    }

    #[must_use]
    pub(crate) fn shadow_request(
        &self,
        id: ShadowPipelineRequestId,
    ) -> (
        FastShadowPipelineKey,
        PreparedSourceId,
        ShadowPipelineLayoutId,
        Option<RenderPipelineId>,
    ) {
        let request = self.shadow_requests[id.index()];
        (request.key, request.owner, request.layout, request.result)
    }

    pub(crate) fn resolve_shadow_request(
        &mut self,
        id: ShadowPipelineRequestId,
        result: RenderPipelineId,
        publish_l1: bool,
    ) {
        let request = &mut self.shadow_requests[id.index()];
        assert!(
            request.result.is_none(),
            "shadow pipeline request resolved twice"
        );
        request.result = Some(result);
        request.publish_l1 = publish_l1;
    }

    #[must_use]
    pub(crate) fn main_result(&self, id: MainPipelineRequestId) -> Option<RenderPipelineId> {
        self.main_requests[id.index()].result
    }

    #[must_use]
    pub(crate) fn prepass_result(&self, id: PrepassPipelineRequestId) -> Option<RenderPipelineId> {
        self.prepass_requests[id.index()].result
    }

    #[must_use]
    pub(crate) fn shadow_result(&self, id: ShadowPipelineRequestId) -> Option<RenderPipelineId> {
        self.shadow_requests[id.index()].result
    }

    pub(crate) fn publish_main_l1(&self, cache: &mut crate::pipeline::PipelineCache) {
        assert!(
            self.source_gate_finalized,
            "main L1 publication must follow the final source epoch gate"
        );
        for request in &self.main_requests {
            if request.publish_l1 {
                cache.insert_pipeline_fast(
                    request.key,
                    request
                        .result
                        .expect("publishable main request lacks a result"),
                );
            }
        }
    }

    pub(crate) fn publish_shadow_l1(&self, cache: &mut crate::pipeline::PipelineCache) {
        assert!(
            self.source_gate_finalized,
            "shadow L1 publication must follow the final source epoch gate"
        );
        for request in &self.shadow_requests {
            if request.publish_l1 {
                cache.insert_shadow_pipeline_fast(
                    request.key,
                    request
                        .result
                        .expect("publishable shadow request lacks a result"),
                );
            }
        }
    }

    pub(crate) fn prepass_publications(
        &self,
    ) -> impl Iterator<Item = (PrepassPipelineL1Key, RenderPipelineId)> + '_ {
        assert!(
            self.source_gate_finalized,
            "Prepass L1 publication must follow the final source epoch gate"
        );
        self.prepass_requests
            .iter()
            .filter(|request| request.publish_l1)
            .map(|request| {
                (
                    request.key,
                    request.result.expect("publishable prepass lacks result"),
                )
            })
    }

    #[must_use]
    pub(crate) fn main_draws(&self, plan_index: usize) -> &[MainDrawRecipe] {
        range_slice(&self.main_draws, &self.main_ranges, plan_index)
    }

    #[must_use]
    pub(crate) fn prepass_draws(&self, plan_index: usize) -> &[PrepassDrawRecipe] {
        range_slice(&self.prepass_draws, &self.prepass_ranges, plan_index)
    }

    #[must_use]
    pub(crate) fn shadow_draws(&self, plan_index: usize) -> &[ShadowDrawRecipe] {
        range_slice(&self.shadow_draws, &self.shadow_ranges, plan_index)
    }
}

#[inline]
fn source_is_stamped(stamps: &[u32], generation: u32, source: PreparedSourceId) -> bool {
    stamps.get(source.index()).copied() == Some(generation)
}

fn source_join_footprint(
    active_slots: usize,
    participating_stamps: &Vec<u32>,
    accepted_indices: &Vec<u32>,
    accepted_stamps: &Vec<u32>,
) -> CapacityUsage {
    let slot_bytes = std::mem::size_of::<u32>();
    CapacityUsage::new(
        active_slots.saturating_mul(slot_bytes).saturating_mul(3),
        participating_stamps
            .capacity()
            .saturating_add(accepted_indices.capacity())
            .saturating_add(accepted_stamps.capacity())
            .saturating_mul(slot_bytes),
        PIPELINE_SOURCE_BASELINE
            .saturating_mul(slot_bytes)
            .saturating_mul(3),
    )
}

fn trim_source_join(
    participating_stamps: &mut Vec<u32>,
    accepted_indices: &mut Vec<u32>,
    accepted_stamps: &mut Vec<u32>,
    target_bytes: usize,
    retained_bytes: usize,
) {
    let current_len = participating_stamps
        .len()
        .max(accepted_indices.len())
        .max(accepted_stamps.len());
    let current_capacity = participating_stamps
        .capacity()
        .max(accepted_indices.capacity())
        .max(accepted_stamps.capacity());
    let target_capacity = if retained_bytes == 0 || target_bytes >= retained_bytes {
        current_capacity
    } else {
        (((current_capacity as u128) * (target_bytes as u128))
            .div_ceil(retained_bytes as u128)
            .min(usize::MAX as u128) as usize)
            .max(PIPELINE_SOURCE_BASELINE)
            .min(current_capacity)
    };
    let target_len = current_len.min(target_capacity);
    for values in [participating_stamps, accepted_indices, accepted_stamps] {
        values.truncate(target_len);
        values.shrink_to(target_capacity);
    }
}

trait PlannedDrawRecipe {
    fn ordinal(&self) -> SceneRecipeOrdinal;
}

impl PlannedDrawRecipe for MainDrawRecipe {
    fn ordinal(&self) -> SceneRecipeOrdinal {
        self.ordinal
    }
}

impl PlannedDrawRecipe for PrepassDrawRecipe {
    fn ordinal(&self) -> SceneRecipeOrdinal {
        self.ordinal
    }
}

impl PlannedDrawRecipe for ShadowDrawRecipe {
    fn ordinal(&self) -> SceneRecipeOrdinal {
        self.ordinal
    }
}

fn build_plan_ranges<T: PlannedDrawRecipe>(
    recipes: &[T],
    plan_count: usize,
    ranges: &mut Vec<Range<u32>>,
) {
    ranges.clear();
    ranges.resize(plan_count, 0..0);
    let mut cursor = 0usize;
    while cursor < recipes.len() {
        let plan_index = recipes[cursor].ordinal().plan_index as usize;
        assert!(
            plan_index < plan_count,
            "recipe addresses a foreign draw plan"
        );
        let start = cursor;
        cursor += 1;
        while cursor < recipes.len() && recipes[cursor].ordinal().plan_index as usize == plan_index
        {
            cursor += 1;
        }
        ranges[plan_index] = u32::try_from(start).expect("recipe range start exceeds u32")
            ..u32::try_from(cursor).expect("recipe range end exceeds u32");
    }
}

fn range_slice<'a, T>(values: &'a [T], ranges: &[Range<u32>], index: usize) -> &'a [T] {
    let Some(range) = ranges.get(index) else {
        return &[];
    };
    &values[range.start as usize..range.end as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ItemShaderFeatures;
    use crate::graph::frame::{DrawPlanKey, DrawPlanStore};
    use slotmap::KeyData;

    fn prepass_key(discriminator: u64) -> PrepassPipelineL1Key {
        let asset_version = u32::try_from(discriminator).expect("test discriminator exceeds u32");
        PrepassPipelineL1Key {
            geometry_handle: GeometryHandle::from(KeyData::from_ffi(discriminator * 2 + 1)),
            geometry_asset_version: asset_version,
            geometry_asset_identity: discriminator * 11 + 1,
            geometry_layout_version: discriminator * 13 + 1,
            material_handle: MaterialHandle::from(KeyData::from_ffi(discriminator * 2 + 2)),
            material_asset_version: asset_version,
            material_asset_identity: discriminator * 17 + 1,
            material_version: discriminator * 19 + 1,
            item_variants: asset_version,
            item_shader_features: ItemShaderFeatures::from_bits_retain(discriminator as u8 & 3),
            shader_source_identity: discriminator * 29 + 1,
            scene_defines_hash: discriminator * 31 + 1,
            pipeline_settings_version: discriminator * 37 + 1,
            global_binding_interface_id: BindingInterfaceId::from_raw(discriminator * 3 + 1),
            material_binding_interface_id: BindingInterfaceId::from_raw(discriminator * 3 + 2),
            object_binding_interface_id: BindingInterfaceId::from_raw(discriminator * 3 + 3),
            vertex_layout_id: discriminator * 41 + 1,
            bind_group_layout_ids: [
                discriminator * 43 + 1,
                discriminator * 43 + 2,
                discriminator * 43 + 3,
            ],
            depth_format: wgpu::TextureFormat::Depth32Float,
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            front_face: wgpu::FrontFace::Ccw,
            needs_normal: discriminator & 1 != 0,
            needs_feature_id: discriminator & 2 != 0,
            needs_velocity: discriminator & 4 != 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_prepass_test_recipe(
        store: &mut ScenePipelinePrepareStore,
        plan: DrawPlanId,
        plan_index: usize,
        intent_index: usize,
        source_index: usize,
        key: PrepassPipelineL1Key,
    ) {
        store.push_prepass(
            plan,
            SceneRecipeOrdinal::new(plan_index, intent_index),
            FrameSourceId::from_index(source_index),
            PreparedSourceId::from_index(source_index),
            key,
            None,
            u32::try_from(source_index).expect("test source index exceeds u32"),
            source_index as f32,
        );
    }

    fn mark_test_source_accepted(store: &mut ScenePipelinePrepareStore, source: PreparedSourceId) {
        assert_eq!(
            store
                .participating_source_stamps
                .get(source.index())
                .copied(),
            Some(store.source_join_generation)
        );
        store.accepted_source_stamps[source.index()] = store.source_join_generation;
    }

    #[test]
    fn planning_restores_semantic_order_and_deduplicates_to_the_earliest_owner() {
        let mut plans = DrawPlanStore::new();
        let plan_ids = [
            plans.reserve_plan(DrawPlanKey::Opaque),
            plans.reserve_plan(DrawPlanKey::Transparent),
            plans.reserve_plan(DrawPlanKey::Prepass {
                needs_normal: true,
                needs_feature_id: true,
                needs_velocity: true,
            }),
        ];
        let keys = [prepass_key(1), prepass_key(2), prepass_key(3)];
        let mut store = ScenePipelinePrepareStore::new();

        // Gather in deliberately non-semantic order. The same exact request
        // also appears in several plans, as worker-local batches will in M8.
        for &(plan_index, intent_index, source_index, key_index) in &[
            (2, 2, 22, 0),
            (0, 2, 2, 0),
            (1, 0, 10, 0),
            (0, 0, 0, 1),
            (2, 1, 21, 1),
            (0, 1, 1, 2),
        ] {
            push_prepass_test_recipe(
                &mut store,
                plan_ids[plan_index],
                plan_index,
                intent_index,
                source_index,
                keys[key_index],
            );
        }

        store.plan(plan_ids.len());

        assert_eq!(
            store
                .prepass_draws(0)
                .iter()
                .map(|draw| draw.ordinal)
                .collect::<Vec<_>>(),
            [
                SceneRecipeOrdinal::new(0, 0),
                SceneRecipeOrdinal::new(0, 1),
                SceneRecipeOrdinal::new(0, 2),
            ]
        );
        assert_eq!(
            store
                .prepass_draws(1)
                .iter()
                .map(|draw| draw.ordinal)
                .collect::<Vec<_>>(),
            [SceneRecipeOrdinal::new(1, 0)]
        );
        assert_eq!(
            store
                .prepass_draws(2)
                .iter()
                .map(|draw| draw.ordinal)
                .collect::<Vec<_>>(),
            [SceneRecipeOrdinal::new(2, 1), SceneRecipeOrdinal::new(2, 2),]
        );
        assert!(store.prepass_draws(3).is_empty());
        for (plan_index, &plan) in plan_ids.iter().enumerate() {
            assert!(
                store
                    .prepass_draws(plan_index)
                    .iter()
                    .all(|draw| draw.draw_plan == plan)
            );
        }

        // Exact keys share one coordinated request across plans, and its
        // owner is the first semantic consumer rather than the first worker
        // that happened to gather it.
        assert_eq!(store.prepass_request_count(), 3);
        let shared_request = store.prepass_draws(0)[2].pipeline;
        assert_eq!(store.prepass_draws(1)[0].pipeline, shared_request);
        assert_eq!(store.prepass_draws(2)[1].pipeline, shared_request);
        let (_, owner, result) = store.prepass_request(shared_request);
        assert_eq!(owner, PreparedSourceId::from_index(2));
        assert_eq!(result, None);

        // Request coordination order follows each key's first semantic
        // consumer, independent of worker gather order.
        let ordered_keys = store
            .prepass_request_order()
            .iter()
            .map(|&id| store.prepass_request(id).0)
            .collect::<Vec<_>>();
        assert_eq!(ordered_keys, [keys[1], keys[2], keys[0]]);
    }

    #[test]
    fn shader_identity_is_derived_once_per_exact_material_generation() {
        let handle = MaterialHandle::from(KeyData::from_ffi(91));
        let mut store = ScenePipelinePrepareStore::new();
        let mut derives = 0_u32;

        let first = store.material_shader_source_identity(handle, 3, 5, 7, || {
            derives += 1;
            0xabc
        });
        let repeated = store.material_shader_source_identity(handle, 3, 5, 7, || {
            derives += 1;
            0xdef
        });
        let next_generation = store.material_shader_source_identity(handle, 3, 5, 8, || {
            derives += 1;
            0x123
        });

        assert_eq!((first, repeated, next_generation), (0xabc, 0xabc, 0x123));
        assert_eq!(derives, 2);
        store.clear();
        assert!(store.shader_identities.is_empty());
    }

    #[test]
    fn participating_source_generations_and_capacity_follow_retained_store_contract() {
        let mut plans = DrawPlanStore::new();
        let plan = plans.reserve_plan(DrawPlanKey::Opaque);
        let mut store = ScenePipelinePrepareStore::new();
        let source = PreparedSourceId::from_index(7);
        for intent_index in 0..3 {
            push_prepass_test_recipe(
                &mut store,
                plan,
                0,
                intent_index,
                source.index(),
                prepass_key(intent_index as u64 + 1),
            );
        }
        store.plan(1);
        assert_eq!(store.participating_source_count(), 1);
        assert_eq!(store.participating_source_at(0), source);

        store.clear();
        store.source_join_generation = u32::MAX;
        push_prepass_test_recipe(&mut store, plan, 0, 0, source.index(), prepass_key(9));
        assert_eq!(store.participating_source_stamps[source.index()], u32::MAX);
        store.clear();
        assert_eq!(store.source_join_generation, 1);
        assert_eq!(store.participating_source_stamps[source.index()], 0);

        push_prepass_test_recipe(&mut store, plan, 0, 0, source.index(), prepass_key(10));
        store.plan(1);
        assert_eq!(store.participating_source_count(), 1);

        // The same store retains a high-water allocation across clear/reuse,
        // then releases only excess capacity when explicitly trimmed.
        store.clear();
        let key = prepass_key(11);
        for intent_index in 0..(DRAW_RECIPE_BASELINE * 16) {
            push_prepass_test_recipe(&mut store, plan, 0, intent_index, intent_index, key);
        }
        store.plan(1);
        let populated = store.capacity_usage();
        assert!(populated.active_bytes > 0);
        assert!(populated.retained_bytes > populated.baseline_bytes);

        store.clear();
        let retained = store.capacity_usage();
        assert_eq!(retained.active_bytes, 0);
        assert_eq!(retained.retained_bytes, populated.retained_bytes);

        for intent_index in 0..(DRAW_RECIPE_BASELINE * 16) {
            push_prepass_test_recipe(&mut store, plan, 0, intent_index, intent_index, key);
        }
        store.plan(1);
        assert_eq!(
            store.capacity_usage().retained_bytes,
            retained.retained_bytes
        );

        store.clear();
        let retained = store.capacity_usage();
        store.trim_capacity(0, retained.retained_bytes);
        let trimmed = store.capacity_usage();
        assert!(trimmed.retained_bytes < retained.retained_bytes);
        assert!(
            trimmed.retained_bytes
                >= ScenePipelinePrepareStore::new()
                    .capacity_usage()
                    .retained_bytes
        );
        assert!(store.is_empty());
    }

    #[test]
    fn source_gate_rejects_drifting_owner_but_keeps_accepted_owner_request() {
        let mut plans = DrawPlanStore::new();
        let plan = plans.reserve_plan(DrawPlanKey::Opaque);
        let key = prepass_key(21);
        let pipeline = RenderPipelineId(77);

        // The earliest semantic consumer is the deterministic owner. Even
        // though another consumer remains current, an owner drift rejects the
        // shared request exactly as the pre-gate coordinator did.
        let mut rejected_owner = ScenePipelinePrepareStore::new();
        rejected_owner.push_prepass(
            plan,
            SceneRecipeOrdinal::new(0, 5),
            FrameSourceId::from_index(5),
            PreparedSourceId::from_index(5),
            key,
            Some(pipeline),
            0,
            0.0,
        );
        rejected_owner.push_prepass(
            plan,
            SceneRecipeOrdinal::new(0, 2),
            FrameSourceId::from_index(2),
            PreparedSourceId::from_index(2),
            key,
            Some(pipeline),
            0,
            0.0,
        );
        rejected_owner.plan(1);
        mark_test_source_accepted(&mut rejected_owner, PreparedSourceId::from_index(5));
        let request = rejected_owner.prepass_draws(0)[0].pipeline;
        assert_eq!(
            rejected_owner.prepass_request(request).1,
            PreparedSourceId::from_index(2)
        );
        rejected_owner.finish_source_gate();
        assert_eq!(rejected_owner.prepass_result(request), None);
        assert_eq!(rejected_owner.prepass_publications().count(), 0);

        // If the owner remains current, one drifting non-owner does not poison
        // the shared request. Direct finalization will skip only that recipe
        // through the accepted-source join.
        let mut accepted_owner = ScenePipelinePrepareStore::new();
        push_prepass_test_recipe(&mut accepted_owner, plan, 0, 5, 5, key);
        push_prepass_test_recipe(&mut accepted_owner, plan, 0, 2, 2, key);
        accepted_owner.plan(1);
        let request = accepted_owner.prepass_draws(0)[0].pipeline;
        accepted_owner.resolve_prepass_request(request, pipeline, true);
        mark_test_source_accepted(&mut accepted_owner, PreparedSourceId::from_index(2));
        accepted_owner.finish_source_gate();
        assert_eq!(accepted_owner.prepass_result(request), Some(pipeline));
        assert_eq!(accepted_owner.prepass_publications().count(), 1);
        assert!(!source_is_stamped(
            &accepted_owner.accepted_source_stamps,
            accepted_owner.source_join_generation,
            PreparedSourceId::from_index(5),
        ));
    }
}
