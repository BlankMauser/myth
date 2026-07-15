use super::*;
use crate::core::{ResourceManager, WgpuContext};
use crate::graph::core::diagnostics::RdgDiagnosticsRecorder;
use crate::graph::core::{
    ExecuteContext, FrameArena, GraphStorage, PassNode, PrepareContext, RenderGraph,
};
use crate::settings::{RendererInitConfig, RendererSettings};
use crate::test_support::SharedTestContext;
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
    pollster::block_on(SharedTestContext::new_headless(
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
