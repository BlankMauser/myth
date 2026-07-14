//! DPE-projected scene-global (Group 0) preparation.
//!
//! Graph construction captures only coherent CPU buffer stamps. Passes declare
//! a lightweight [`GlobalStateKey`] demand, and topology compilation projects
//! demands from the live execution queue onto the single plan owned by this
//! graph attempt. Physical buffers and the finalized bind group are retained
//! only after that demand survives DPE.

use myth_resources::buffer::{CpuBuffer, CpuBufferStamp, GpuData};
use myth_resources::uniforms::{EnvironmentUniforms, GpuLightStorage, RenderStateUniforms};

use crate::core::gpu::{
    BufferUploadContext, GlobalStateKey, HintedBufferSnapshotInput, MaterializedBufferToken,
};

/// Coherent logical identity of the three persistent buffers in Group 0.
///
/// A Composer renders one scene/global domain, so one fixed plan is sufficient;
/// no per-frame hash table or generational plan ID is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlobalStatePlan {
    pub(crate) key: GlobalStateKey,
    camera: CpuBufferStamp,
    environment: CpuBufferStamp,
    directional_lights: CpuBufferStamp,
}

impl GlobalStatePlan {
    #[inline]
    #[must_use]
    pub(crate) fn capture(key: GlobalStateKey, sources: GlobalStateSources<'_>) -> Self {
        Self {
            key,
            camera: sources.camera.stamp(),
            environment: sources.environment.stamp(),
            directional_lights: sources.directional_lights.stamp(),
        }
    }
}

/// Explicit CPU sources supplied at graph construction and again at the
/// post-DPE materialization barrier.
#[derive(Clone, Copy)]
pub(crate) struct GlobalStateSources<'a> {
    pub(crate) camera: &'a CpuBuffer<RenderStateUniforms>,
    pub(crate) environment: &'a CpuBuffer<EnvironmentUniforms>,
    pub(crate) directional_lights: &'a CpuBuffer<Vec<GpuLightStorage>>,
}

/// Terminal outcome for the fixed three-buffer materialization batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlobalStateMaterializationFailure {
    /// A CPU source changed after graph construction captured the plan.
    SnapshotDrift,
    /// Persistent ResourceManager state conflicts with the exact plan.
    ConflictingPhysicalState,
    /// This attempt already failed, or a live standalone demand has no plan.
    Unavailable,
}

/// Exact physical state owned by one graph attempt.
///
/// `attempted` makes both success and failure sticky within an attempt. The
/// three Copy buffer tokens avoid a heap allocation and preserve binding
/// order; only the finalized BindGroup is a destructor-aware owner.
#[derive(Default)]
pub(crate) struct PreparedGlobalState {
    attempted: bool,
    camera: Option<MaterializedBufferToken>,
    environment: Option<MaterializedBufferToken>,
    directional_lights: Option<MaterializedBufferToken>,
    finalized: Option<(GlobalStateKey, wgpu::BindGroup)>,
}

impl PreparedGlobalState {
    pub(crate) fn clear(&mut self) {
        self.attempted = false;
        self.camera = None;
        self.environment = None;
        self.directional_lights = None;
        self.finalized = None;
    }

    #[inline]
    #[must_use]
    fn succeeded(&self) -> bool {
        self.attempted
            && self.camera.is_some()
            && self.environment.is_some()
            && self.directional_lights.is_some()
    }

    #[inline]
    fn record_failure(&mut self) {
        self.clear();
        self.attempted = true;
    }

    fn record_buffers(
        &mut self,
        camera: MaterializedBufferToken,
        environment: MaterializedBufferToken,
        directional_lights: MaterializedBufferToken,
    ) {
        assert!(!self.attempted, "global state batch was attempted twice");
        self.attempted = true;
        self.camera = Some(camera);
        self.environment = Some(environment);
        self.directional_lights = Some(directional_lights);
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn resource_ids(&self) -> Option<[u64; 3]> {
        self.succeeded().then(|| {
            [
                self.camera
                    .as_ref()
                    .expect("camera buffer missing")
                    .physical_id,
                self.environment
                    .as_ref()
                    .expect("environment buffer missing")
                    .physical_id,
                self.directional_lights
                    .as_ref()
                    .expect("directional buffer missing")
                    .physical_id,
            ]
        })
    }

    #[inline]
    #[must_use]
    pub(crate) fn finalized_key(&self) -> Option<GlobalStateKey> {
        self.finalized.as_ref().map(|(key, _)| *key)
    }

    pub(crate) fn stage_finalized(
        &mut self,
        allowed_key: Option<GlobalStateKey>,
        key: GlobalStateKey,
        bind_group: &wgpu::BindGroup,
    ) {
        assert_eq!(
            allowed_key,
            Some(key),
            "finalized global state does not match the DPE-live materialized plan"
        );
        assert!(
            self.finalized.is_none(),
            "global bind group was finalized twice for one graph attempt"
        );
        self.finalized = Some((key, bind_group.clone()));
    }

    #[inline]
    #[must_use]
    pub(crate) fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.finalized.as_ref().map(|(_, bind_group)| bind_group)
    }

    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        !self.attempted
            && self.camera.is_none()
            && self.environment.is_none()
            && self.directional_lights.is_none()
            && self.finalized.is_none()
    }
}

/// Capture and DPE-projection state held directly by [`GraphStorage`].
#[derive(Default)]
pub(crate) struct GlobalStateAttempt {
    plan: Option<GlobalStatePlan>,
    live_key: Option<GlobalStateKey>,
    pub(crate) prepared: PreparedGlobalState,
}

impl GlobalStateAttempt {
    pub(crate) fn clear(&mut self) {
        self.plan = None;
        self.live_key = None;
        self.prepared.clear();
    }

    pub(crate) fn set_plan(&mut self, plan: GlobalStatePlan) {
        match self.plan {
            None => self.plan = Some(plan),
            Some(current) if current == plan => {}
            Some(_) => panic!("one RenderGraph attempt cannot capture two global state plans"),
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn plan_key(&self) -> Option<GlobalStateKey> {
        self.plan.map(|plan| plan.key)
    }

    pub(crate) fn project_live_key(
        &mut self,
        live_key: Option<GlobalStateKey>,
        require_plan: bool,
    ) {
        self.live_key = live_key;
        let Some(live_key) = live_key else {
            return;
        };
        if require_plan {
            assert_eq!(
                self.plan_key(),
                Some(live_key),
                "DPE-live renderer pass demanded a missing or mismatched global state plan"
            );
        } else if let Some(plan_key) = self.plan_key() {
            assert_eq!(
                plan_key, live_key,
                "DPE-live standalone pass demanded a mismatched global state plan"
            );
        }
    }

    /// Exact fixed-size batch materialization. Every source lock and every
    /// physical preflight succeeds before the first upload is issued.
    pub(crate) fn materialize_live(
        &mut self,
        sources: GlobalStateSources<'_>,
        uploads: &mut BufferUploadContext<'_>,
    ) -> Result<(), GlobalStateMaterializationFailure> {
        let Some(live_key) = self.live_key else {
            return Ok(());
        };
        if self.prepared.attempted {
            return if self.prepared.succeeded() {
                Ok(())
            } else {
                Err(GlobalStateMaterializationFailure::Unavailable)
            };
        }
        let Some(plan) = self.plan.filter(|plan| plan.key == live_key) else {
            self.prepared.record_failure();
            return Err(GlobalStateMaterializationFailure::Unavailable);
        };

        // Hold all three guards through validation and upload. A CPU writer
        // cannot produce a split Group 0 snapshot while this batch is active.
        let (camera_stamp, camera) = sources.camera.read_stamped();
        let (environment_stamp, environment) = sources.environment.read_stamped();
        let (directional_stamp, directional_lights) = sources.directional_lights.read_stamped();
        if camera_stamp != plan.camera
            || environment_stamp != plan.environment
            || directional_stamp != plan.directional_lights
        {
            self.prepared.record_failure();
            return Err(GlobalStateMaterializationFailure::SnapshotDrift);
        }

        let Some([camera_buffer, environment_buffer, directional_buffer]) = uploads
            .materialize_hinted_buffer_snapshot_batch([
                HintedBufferSnapshotInput {
                    stamp: camera_stamp,
                    bytes: camera.as_bytes(),
                    cached_handle: sources.camera.gpu_handle(),
                },
                HintedBufferSnapshotInput {
                    stamp: environment_stamp,
                    bytes: environment.as_bytes(),
                    cached_handle: sources.environment.gpu_handle(),
                },
                HintedBufferSnapshotInput {
                    stamp: directional_stamp,
                    bytes: directional_lights.as_bytes(),
                    cached_handle: sources.directional_lights.gpu_handle(),
                },
            ])
        else {
            self.prepared.record_failure();
            return Err(GlobalStateMaterializationFailure::ConflictingPhysicalState);
        };

        // Publish cached handles only after the complete atomic batch has
        // materialized exactly. Subsequent warm attempts can then bypass the
        // reverse CPU-ID index without exposing a partial batch on failure.
        sources.camera.set_gpu_handle(camera_buffer.handle);
        sources
            .environment
            .set_gpu_handle(environment_buffer.handle);
        sources
            .directional_lights
            .set_gpu_handle(directional_buffer.handle);
        self.prepared
            .record_buffers(camera_buffer, environment_buffer, directional_buffer);
        Ok(())
    }

    #[inline]
    #[must_use]
    pub(crate) fn allowed_key(&self) -> Option<GlobalStateKey> {
        self.prepared.succeeded().then_some(self.live_key).flatten()
    }

    /// Return exact Copy tokens only after the complete batch succeeded. This
    /// prevents finalization from reading a newer CPU stamp or repeating the
    /// CPU-ID hash lookup after the upload barrier.
    #[must_use]
    pub(crate) fn materialized_buffer_tokens(&self) -> Option<[MaterializedBufferToken; 3]> {
        self.prepared.succeeded().then(|| {
            let camera = self.prepared.camera.expect("camera token missing");
            let environment = self
                .prepared
                .environment
                .expect("environment token missing");
            let directional = self
                .prepared
                .directional_lights
                .expect("directional token missing");
            [camera, environment, directional]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ResourceManager, WgpuContext};
    use crate::graph::core::diagnostics::RdgDiagnosticsRecorder;
    use crate::graph::core::{
        ExecuteContext, FrameArena, GraphStorage, PassNode, PrepareContext, RenderGraph,
    };
    use crate::settings::{RendererInitConfig, RendererSettings};
    use myth_resources::buffer::BufferRef;

    struct EmptyNode;

    impl PassNode<'_> for EmptyNode {
        fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {}

        fn execute(&self, _ctx: &ExecuteContext<'_>, _encoder: &mut wgpu::CommandEncoder) {}
    }

    struct TestSources {
        camera: CpuBuffer<RenderStateUniforms>,
        environment: CpuBuffer<EnvironmentUniforms>,
        directional: CpuBuffer<Vec<GpuLightStorage>>,
    }

    impl TestSources {
        fn new() -> Self {
            Self {
                camera: CpuBuffer::new(
                    RenderStateUniforms::default(),
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    Some("global state camera test"),
                ),
                environment: CpuBuffer::new(
                    EnvironmentUniforms::default(),
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    Some("global state environment test"),
                ),
                directional: CpuBuffer::new(
                    vec![GpuLightStorage::default()],
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    Some("global state directional test"),
                ),
            }
        }

        fn get(&self) -> GlobalStateSources<'_> {
            GlobalStateSources {
                camera: &self.camera,
                environment: &self.environment,
                directional_lights: &self.directional,
            }
        }
    }

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

    fn materialize(
        graph: &mut RenderGraph<'_>,
        sources: &TestSources,
        wgpu: &WgpuContext,
        resources: &mut ResourceManager,
    ) -> Result<(), GlobalStateMaterializationFailure> {
        let mut uploads = BufferUploadContext::new(
            &wgpu.device,
            &wgpu.queue,
            resources.frame_index,
            &mut resources.resources,
            &resources.internal_textures,
        );
        graph.materialize_live_global_state(sources.get(), &mut uploads)
    }

    #[test]
    fn all_dead_demand_does_not_materialize_or_allow_a_key() {
        let wgpu = gpu();
        let sources = TestSources::new();
        let key = GlobalStateKey::new(1, 2);
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            graph.capture_global_state_plan(key, sources.get());
            graph.add_pass("dead global consumer", |builder| {
                builder.demand_global_state(key);
                (EmptyNode, ())
            });
            graph.compile_topology();
            assert_eq!(
                materialize(&mut graph, &sources, &wgpu, &mut resources),
                Ok(())
            );
            assert_eq!(graph.global_state_allowed_key(), None);
            assert_eq!(graph.materialized_global_buffer_ids(), None);
            assert!(graph.materialized_global_buffer_tokens().is_none());
            assert_eq!(graph.finalized_global_state_key(), None);
        }

        for id in [
            sources.camera.id(),
            sources.environment.id(),
            sources.directional.id(),
        ] {
            assert!(resources.get_gpu_buffer_by_cpu_id(id).is_none());
        }
        assert_eq!(resources.global_state_cached_handle_hit_count(), 0);
    }

    #[test]
    fn live_repeated_demand_materializes_once_and_stages_finalized_owner() {
        let wgpu = gpu();
        let sources = TestSources::new();
        let key = GlobalStateKey::new(3, 4);
        let expected_stamps = [
            sources.camera.stamp(),
            sources.environment.stamp(),
            sources.directional.stamp(),
        ];
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            graph.capture_global_state_plan(key, sources.get());
            graph.add_pass("live global consumer", |builder| {
                builder.demand_global_state(key);
                builder.demand_global_state(key);
                builder.mark_side_effect();
                (EmptyNode, ())
            });
            graph.compile_topology();
            assert!(graph.materialized_global_buffer_tokens().is_none());
            assert_eq!(graph.finalized_global_state_key(), None);
            assert_eq!(
                materialize(&mut graph, &sources, &wgpu, &mut resources),
                Ok(())
            );
            let ids = graph
                .materialized_global_buffer_ids()
                .expect("live global buffers missing");
            assert_eq!(graph.global_state_allowed_key(), Some(key));
            let expected_tokens = graph
                .materialized_global_buffer_tokens()
                .expect("live global tokens missing");
            assert_eq!(expected_tokens.map(|token| token.stamp), expected_stamps);
            assert_eq!(expected_tokens.map(|token| token.physical_id), ids);
            assert_eq!(
                [
                    sources.camera.gpu_handle(),
                    sources.environment.gpu_handle(),
                    sources.directional.gpu_handle(),
                ],
                expected_tokens.map(|token| Some(token.handle)),
                "the complete batch must publish exact cached-handle hints"
            );

            // A later CPU write must not change the owned exact refs exposed
            // for finalizing this already-materialized graph snapshot.
            drop(sources.camera.write());
            assert_eq!(
                graph.materialized_global_buffer_tokens(),
                Some(expected_tokens)
            );
            assert_eq!(graph.finalized_global_state_key(), None);

            // Success is sticky and does not touch or recreate the physical set.
            assert_eq!(
                materialize(&mut graph, &sources, &wgpu, &mut resources),
                Ok(())
            );
            assert_eq!(graph.materialized_global_buffer_ids(), Some(ids));

            let layout = wgpu
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("global state owner test"),
                    entries: &[],
                });
            let bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("global state owner test"),
                layout: &layout,
                entries: &[],
            });
            graph.stage_finalized_global_bind_group(key, &bind_group);
            assert!(graph.finalized_global_bind_group().is_some());
            assert_eq!(graph.finalized_global_state_key(), Some(key));
        }

        assert!(storage.global_state_attempt.prepared.is_empty());
        for source in expected_stamps {
            let physical = resources
                .get_gpu_buffer_by_cpu_id(source.id)
                .expect("materialized global buffer missing from ResourceManager");
            assert_eq!(physical.last_uploaded_version, source.version);
        }
    }

    #[test]
    fn drift_fails_before_any_member_upload_and_next_attempt_retries() {
        let wgpu = gpu();
        let sources = TestSources::new();
        let key = GlobalStateKey::new(5, 6);
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            graph.capture_global_state_plan(key, sources.get());
            graph.add_pass("drifting global consumer", |builder| {
                builder.demand_global_state(key);
                builder.mark_side_effect();
                (EmptyNode, ())
            });
            graph.compile_topology();
            drop(sources.environment.write());
            assert_eq!(
                materialize(&mut graph, &sources, &wgpu, &mut resources),
                Err(GlobalStateMaterializationFailure::SnapshotDrift)
            );
            assert_eq!(
                materialize(&mut graph, &sources, &wgpu, &mut resources),
                Err(GlobalStateMaterializationFailure::Unavailable)
            );
        }
        for id in [
            sources.camera.id(),
            sources.environment.id(),
            sources.directional.id(),
        ] {
            assert!(resources.get_gpu_buffer_by_cpu_id(id).is_none());
        }

        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            graph.capture_global_state_plan(key, sources.get());
            graph.add_pass("global retry", |builder| {
                builder.demand_global_state(key);
                builder.mark_side_effect();
                (EmptyNode, ())
            });
            graph.compile_topology();
            assert_eq!(
                materialize(&mut graph, &sources, &wgpu, &mut resources),
                Ok(())
            );
            assert_eq!(graph.global_state_allowed_key(), Some(key));
        }
    }

    #[test]
    fn warm_batch_uses_exact_cached_handles_and_wrong_owner_hint_falls_back() {
        let wgpu = gpu();
        let sources = TestSources::new();
        let key = GlobalStateKey::new(13, 17);
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();

        let run_live = |storage: &mut GraphStorage,
                        resources: &mut ResourceManager|
         -> [MaterializedBufferToken; 3] {
            let mut graph = RenderGraph::new(storage, &arena);
            graph.capture_global_state_plan(key, sources.get());
            graph.add_pass("cached global consumer", |builder| {
                builder.demand_global_state(key);
                builder.mark_side_effect();
                (EmptyNode, ())
            });
            graph.compile_topology();
            assert_eq!(materialize(&mut graph, &sources, &wgpu, resources), Ok(()));
            graph
                .materialized_global_buffer_tokens()
                .expect("materialized global tokens missing")
        };

        let cold = run_live(&mut storage, &mut resources);
        assert_eq!(resources.global_state_cached_handle_hit_count(), 0);

        let warm = run_live(&mut storage, &mut resources);
        assert_eq!(warm, cold);
        assert_eq!(
            resources.global_state_cached_handle_hit_count(),
            3,
            "all three warm global buffers should bypass the reverse index"
        );

        sources.camera.set_gpu_handle(warm[1].handle);
        let recovered = run_live(&mut storage, &mut resources);
        assert_eq!(recovered, cold);
        assert_eq!(
            resources.global_state_cached_handle_hit_count(),
            5,
            "the wrong-owner camera hint must fall back while the other two stay fast"
        );
        assert_eq!(sources.camera.gpu_handle(), Some(cold[0].handle));
    }

    #[test]
    fn newer_directional_physical_version_rejects_whole_batch() {
        let wgpu = gpu();
        let sources = TestSources::new();
        let key = GlobalStateKey::new(7, 8);
        let directional = sources.directional.stamp();
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let newer = BufferRef::with_fixed_id(
            directional.id,
            directional.size,
            directional.usage,
            directional.version + 1,
            None,
        );
        resources.ensure_buffer_ref(&newer, sources.directional.read().as_bytes());

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            graph.capture_global_state_plan(key, sources.get());
            graph.add_pass("conflicting global consumer", |builder| {
                builder.demand_global_state(key);
                builder.mark_side_effect();
                (EmptyNode, ())
            });
            graph.compile_topology();
            assert_eq!(
                materialize(&mut graph, &sources, &wgpu, &mut resources),
                Err(GlobalStateMaterializationFailure::ConflictingPhysicalState)
            );
        }

        assert!(
            resources
                .get_gpu_buffer_by_cpu_id(sources.camera.id())
                .is_none()
        );
        assert!(
            resources
                .get_gpu_buffer_by_cpu_id(sources.environment.id())
                .is_none()
        );
        assert_eq!(
            resources
                .get_gpu_buffer_by_cpu_id(directional.id)
                .expect("seeded directional buffer missing")
                .last_uploaded_version,
            directional.version + 1
        );
    }

    #[test]
    fn standalone_live_demand_may_omit_plan_but_cannot_materialize() {
        let wgpu = gpu();
        let sources = TestSources::new();
        let key = GlobalStateKey::new(9, 10);
        let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        graph.add_pass("standalone global consumer", |builder| {
            builder.demand_global_state(key);
            builder.mark_side_effect();
            (EmptyNode, ())
        });
        graph.compile_topology();
        assert_eq!(
            materialize(&mut graph, &sources, &wgpu, &mut resources),
            Err(GlobalStateMaterializationFailure::Unavailable)
        );
    }

    #[test]
    fn renderer_live_demand_requires_matching_plan() {
        let key = GlobalStateKey::new(11, 12);
        let diagnostics = RdgDiagnosticsRecorder::default();
        diagnostics.start_renderer_attempt(1);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut graph =
                RenderGraph::from_started_renderer_attempt(&mut storage, &arena, &diagnostics);
            graph.add_pass("renderer global consumer", |builder| {
                builder.demand_global_state(key);
                builder.mark_side_effect();
                (EmptyNode, ())
            });
            graph.compile_topology();
        }));
        assert!(result.is_err());
    }

    #[test]
    fn conflicting_demands_fail_at_pass_or_live_graph_boundary() {
        let first = GlobalStateKey::new(1, 1);
        let second = GlobalStateKey::new(2, 2);
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let same_pass = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut graph = RenderGraph::new(&mut storage, &arena);
            graph.add_pass("conflicting pass", |builder| {
                builder.demand_global_state(first);
                builder.demand_global_state(second);
                (EmptyNode, ())
            });
        }));
        assert!(same_pass.is_err());

        let mut graph = RenderGraph::new(&mut storage, &arena);
        for (name, key) in [("first global", first), ("second global", second)] {
            graph.add_pass(name, |builder| {
                builder.demand_global_state(key);
                builder.mark_side_effect();
                (EmptyNode, ())
            });
        }
        let live_graph = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            graph.compile_topology();
        }));
        assert!(live_graph.is_err());
    }
}
