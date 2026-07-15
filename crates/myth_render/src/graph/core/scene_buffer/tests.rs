use super::*;
use crate::core::ResourceManager;
use crate::core::gpu::Tracked;
use crate::graph::core::graph::RenderGraph;
use crate::graph::core::{ExecuteContext, FrameArena, GraphStorage, PassNode, PrepareContext};
use crate::settings::{RendererInitConfig, RendererSettings};
use crate::test_support::SharedTestContext as WgpuContext;

struct ConsumerNode;

impl PassNode<'_> for ConsumerNode {
    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {}

    fn execute(&self, _ctx: &ExecuteContext<'_>, _encoder: &mut wgpu::CommandEncoder) {}
}

fn metadata_buffer() -> CpuBuffer<LightBufferMetadata> {
    CpuBuffer::new(
        LightBufferMetadata::default(),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        Some("scene buffer metadata test"),
    )
}

fn storage_buffer() -> CpuBuffer<Vec<GpuLightStorage>> {
    CpuBuffer::new(
        vec![GpuLightStorage::default()],
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        Some("scene buffer storage test"),
    )
}

#[test]
fn dead_imports_do_not_touch_the_real_resource_manager() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let metadata = metadata_buffer();
    let lights = storage_buffer();
    let metadata_id = metadata.id();
    let lights_id = lights.id();
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);
    let mut storage = GraphStorage::new();
    let arena = FrameArena::new();

    {
        let mut graph = RenderGraph::new(&mut storage, &arena);
        graph.import_scene_light_metadata_buffer("dead metadata", &metadata);
        graph.import_scene_light_storage_buffer("dead lights", &lights);
        graph.compile_topology();

        let mut uploads = BufferUploadContext::new(
            &wgpu.device,
            &wgpu.queue,
            resources.frame_index,
            &mut resources.resources,
            &resources.internal_textures,
        );
        assert_eq!(
            graph.materialize_live_scene_buffers(
                SceneBufferSources {
                    light_metadata: &metadata,
                    light_storage: &lights,
                },
                &mut uploads,
            ),
            Ok(())
        );
    }

    assert!(resources.get_gpu_buffer_by_cpu_id(metadata_id).is_none());
    assert!(resources.get_gpu_buffer_by_cpu_id(lights_id).is_none());
}

#[test]
fn live_import_deduplicates_materializes_once_and_resolves() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let metadata = metadata_buffer();
    let mut storage = GraphStorage::new();
    let arena = FrameArena::new();
    let mut calls = 0;

    {
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let first = graph.import_scene_light_metadata_buffer("metadata first", &metadata);
        let duplicate = graph.import_scene_light_metadata_buffer("metadata duplicate", &metadata);
        assert_eq!(first, duplicate);
        assert_eq!(graph.storage.scene_buffer_imports.entries.len(), 1);

        graph.add_pass("scene buffer consumer", |builder| {
            builder.read_buffer(first);
            builder.mark_side_effect();
            (ConsumerNode, ())
        });
        graph.compile_topology();

        let id = graph.storage.resources[first.index() as usize]
            .scene_buffer_import_id()
            .expect("lazy buffer backing missing");
        let first_result = graph
            .storage
            .scene_buffer_imports
            .materialize_live_for_test(&graph.storage.resources, |_, _| {
                calls += 1;
                let buffer = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("scene buffer materialization test"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                });
                Some(Tracked::new(buffer))
            });
        assert_eq!(first_result, Ok(()));
        let second_result = graph
            .storage
            .scene_buffer_imports
            .materialize_live_for_test(&graph.storage.resources, |_, _| {
                calls += 1;
                None
            });
        assert_eq!(second_result, Ok(()));

        let resolver = graph.storage.scene_buffer_imports.resolver();
        assert!(resolver.buffer(id).is_some());
        assert!(resolver.was_materialized(id));
    }

    assert_eq!(calls, 1);
}

#[test]
fn drift_fails_the_whole_live_batch_before_any_upload() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let metadata = metadata_buffer();
    let lights = storage_buffer();
    let metadata_id = metadata.id();
    let lights_id = lights.id();
    let mut storage = GraphStorage::new();
    let arena = FrameArena::new();
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);

    {
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let metadata_node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
        let storage_node = graph.import_scene_light_storage_buffer("lights", &lights);
        graph.add_pass("scene buffer consumer", |builder| {
            builder.read_buffer(metadata_node);
            builder.read_buffer(storage_node);
            builder.mark_side_effect();
            (ConsumerNode, ())
        });
        graph.compile_topology();

        // Invalidate only one member after graph construction. Atomic
        // validation must prevent the still-current member from uploading.
        drop(metadata.write());
        let mut uploads = BufferUploadContext::new(
            &wgpu.device,
            &wgpu.queue,
            resources.frame_index,
            &mut resources.resources,
            &resources.internal_textures,
        );
        let result = graph.materialize_live_scene_buffers(
            SceneBufferSources {
                light_metadata: &metadata,
                light_storage: &lights,
            },
            &mut uploads,
        );
        assert_eq!(
            result,
            Err(SceneBufferMaterializationFailure::SnapshotDrift)
        );

        let resolver = graph.storage.scene_buffer_imports.resolver();
        for node in [metadata_node, storage_node] {
            let id = graph.storage.resources[node.index() as usize]
                .scene_buffer_import_id()
                .expect("lazy buffer backing missing");
            assert!(resolver.was_materialized(id));
            assert!(resolver.buffer(id).is_none());
        }
    }

    assert!(resources.get_gpu_buffer_by_cpu_id(metadata_id).is_none());
    assert!(resources.get_gpu_buffer_by_cpu_id(lights_id).is_none());

    // A failed attempt owns only its unavailable slots. Rebuilding the
    // graph from the current stamps must retry successfully without stale
    // OnceLock state leaking across attempts.
    {
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let metadata_node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
        let storage_node = graph.import_scene_light_storage_buffer("lights", &lights);
        graph.add_pass("scene buffer retry consumer", |builder| {
            builder.read_buffer(metadata_node);
            builder.read_buffer(storage_node);
            builder.mark_side_effect();
            (ConsumerNode, ())
        });
        graph.compile_topology();

        let mut uploads = BufferUploadContext::new(
            &wgpu.device,
            &wgpu.queue,
            resources.frame_index,
            &mut resources.resources,
            &resources.internal_textures,
        );
        assert_eq!(
            graph.materialize_live_scene_buffers(
                SceneBufferSources {
                    light_metadata: &metadata,
                    light_storage: &lights,
                },
                &mut uploads,
            ),
            Ok(())
        );

        let resolver = graph.storage.scene_buffer_imports.resolver();
        for node in [metadata_node, storage_node] {
            let id = graph.storage.resources[node.index() as usize]
                .scene_buffer_import_id()
                .expect("retry lazy buffer backing missing");
            assert!(resolver.buffer(id).is_some());
        }
    }

    let metadata_gpu = resources
        .get_gpu_buffer_by_cpu_id(metadata_id)
        .expect("retry metadata upload missing");
    assert_eq!(metadata_gpu.last_uploaded_version, metadata.stamp().version);
    let lights_gpu = resources
        .get_gpu_buffer_by_cpu_id(lights_id)
        .expect("retry storage upload missing");
    assert_eq!(lights_gpu.last_uploaded_version, lights.stamp().version);
}

#[test]
fn equal_version_undersized_physical_state_fails_before_batch_upload() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let metadata = metadata_buffer();
    let lights = storage_buffer();
    let metadata_id = metadata.id();
    let light_stamp = lights.stamp();
    assert!(light_stamp.size > 16, "test source must exceed seed buffer");
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);

    // Seed a same-ID, same-version allocation that is too small for the
    // later graph snapshot. Equal versions cause write_buffer to skip, so
    // preflight must reject this state before uploading the other member.
    let seed = BufferRef::with_fixed_id(
        light_stamp.id,
        16,
        light_stamp.usage,
        light_stamp.version,
        None,
    );
    resources.ensure_buffer_ref(&seed, &[0; 16]);

    let mut storage = GraphStorage::new();
    let arena = FrameArena::new();
    {
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let metadata_node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
        let storage_node = graph.import_scene_light_storage_buffer("lights", &lights);
        graph.add_pass("scene buffer consumer", |builder| {
            builder.read_buffer(metadata_node);
            builder.read_buffer(storage_node);
            builder.mark_side_effect();
            (ConsumerNode, ())
        });
        graph.compile_topology();

        let mut uploads = BufferUploadContext::new(
            &wgpu.device,
            &wgpu.queue,
            resources.frame_index,
            &mut resources.resources,
            &resources.internal_textures,
        );
        let result = graph.materialize_live_scene_buffers(
            SceneBufferSources {
                light_metadata: &metadata,
                light_storage: &lights,
            },
            &mut uploads,
        );
        assert_eq!(
            result,
            Err(SceneBufferMaterializationFailure::ConflictingPhysicalState)
        );
    }

    assert!(resources.get_gpu_buffer_by_cpu_id(metadata_id).is_none());
    let seeded = resources
        .get_gpu_buffer_by_cpu_id(light_stamp.id)
        .expect("seeded physical buffer missing");
    assert_eq!(seeded.size, 16);
    assert_eq!(seeded.last_uploaded_version, light_stamp.version);
}

#[test]
fn newer_physical_version_fails_before_companion_upload() {
    let wgpu = pollster::block_on(WgpuContext::new_headless(
        &RendererInitConfig::default(),
        &RendererSettings::default(),
        8,
        8,
        None,
    ))
    .expect("headless GPU context required");
    let metadata = metadata_buffer();
    let lights = storage_buffer();
    let metadata_id = metadata.id();
    let light_stamp = lights.stamp();
    let mut resources = ResourceManager::new(wgpu.device.clone(), wgpu.queue.clone(), 1);

    let newer = BufferRef::with_fixed_id(
        light_stamp.id,
        light_stamp.size,
        light_stamp.usage,
        light_stamp.version + 1,
        None,
    );
    resources.ensure_buffer_ref(&newer, lights.read().as_bytes());

    let mut storage = GraphStorage::new();
    let arena = FrameArena::new();
    {
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let metadata_node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
        let storage_node = graph.import_scene_light_storage_buffer("lights", &lights);
        graph.add_pass("scene buffer consumer", |builder| {
            builder.read_buffer(metadata_node);
            builder.read_buffer(storage_node);
            builder.mark_side_effect();
            (ConsumerNode, ())
        });
        graph.compile_topology();

        let mut uploads = BufferUploadContext::new(
            &wgpu.device,
            &wgpu.queue,
            resources.frame_index,
            &mut resources.resources,
            &resources.internal_textures,
        );
        assert_eq!(
            graph.materialize_live_scene_buffers(
                SceneBufferSources {
                    light_metadata: &metadata,
                    light_storage: &lights,
                },
                &mut uploads,
            ),
            Err(SceneBufferMaterializationFailure::ConflictingPhysicalState)
        );
    }

    assert!(resources.get_gpu_buffer_by_cpu_id(metadata_id).is_none());
    assert_eq!(
        resources
            .get_gpu_buffer_by_cpu_id(light_stamp.id)
            .expect("newer physical state missing")
            .last_uploaded_version,
        light_stamp.version + 1
    );
}

#[test]
fn clear_and_graph_drop_release_attempt_slots_and_stale_ids() {
    let metadata = metadata_buffer();
    let mut storage = GraphStorage::new();
    let arena = FrameArena::new();

    let stale_id = {
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
        let id = graph.storage.resources[node.index() as usize]
            .scene_buffer_import_id()
            .expect("lazy buffer backing missing");
        assert!(!graph.storage.scene_buffer_imports.is_empty());
        id
    };

    assert!(storage.scene_buffer_imports.is_empty());
    assert!(
        storage
            .scene_buffer_imports
            .resolver()
            .buffer(stale_id)
            .is_none()
    );

    {
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let node = graph.import_scene_light_metadata_buffer("metadata", &metadata);
        let old_attempt = graph.storage.resources[node.index() as usize]
            .scene_buffer_import_id()
            .expect("lazy buffer backing missing");
        graph.storage.clear();
        assert!(graph.storage.scene_buffer_imports.is_empty());
        assert!(
            graph
                .storage
                .scene_buffer_imports
                .resolver()
                .buffer(old_attempt)
                .is_none()
        );
    }
    assert!(
        storage
            .scene_buffer_imports
            .resolver()
            .buffer(stale_id)
            .is_none()
    );
}

#[test]
fn foreign_ids_and_mismatched_stamps_fail_closed() {
    let metadata = metadata_buffer();
    let lights = storage_buffer();
    let mut first_storage = GraphStorage::new();
    let mut second_storage = GraphStorage::new();
    let first_arena = FrameArena::new();
    let second_arena = FrameArena::new();

    let mut first_graph = RenderGraph::new(&mut first_storage, &first_arena);
    let first_node = first_graph.import_scene_light_metadata_buffer("metadata", &metadata);
    let first_id = first_graph.storage.resources[first_node.index() as usize]
        .scene_buffer_import_id()
        .expect("lazy buffer backing missing");
    let expected = first_graph
        .storage
        .scene_buffer_imports
        .snapshot(first_id)
        .expect("snapshot missing");

    let mut second_graph = RenderGraph::new(&mut second_storage, &second_arena);
    let second_node = second_graph.import_scene_light_storage_buffer("lights", &lights);
    let foreign_id = second_graph.storage.resources[second_node.index() as usize]
        .scene_buffer_import_id()
        .expect("lazy buffer backing missing");

    let resolver = first_graph.storage.scene_buffer_imports.resolver();
    assert!(!resolver.record(foreign_id, expected, None));
    assert!(!resolver.was_materialized(foreign_id));

    let stale = SceneBufferSnapshot {
        stamp: CpuBufferStamp {
            version: expected.stamp.version.wrapping_add(1),
            ..expected.stamp
        },
        ..expected
    };
    assert!(!resolver.record(first_id, stale, None));
    assert!(resolver.was_materialized(first_id));
    assert!(resolver.buffer(first_id).is_none());
}
