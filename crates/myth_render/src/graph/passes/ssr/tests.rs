use super::*;
use crate::graph::core::graph::FrameConfig;
use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};
use myth_resources::ssr::SsrSettings;
use myth_scene::camera::Camera;

const CUSTOM_RUNTIME_BIT: u32 = 1 << 7;

fn camera(camera_id: u128) -> RenderCamera {
    let mut camera = Camera::new_perspective(45.0, 1.0, 0.1);
    let mut render_camera = camera.extract_render_camera();
    render_camera.camera_id = camera_id;
    render_camera.camera_cut = 0;
    render_camera
}

fn extract(
    feature: &mut SsrFeature,
    histories: &mut HistoryRegistry,
    settings: &SsrSettings,
    camera: &RenderCamera,
    scene_id: u32,
    temporal_serial: u64,
    size: (u32, u32),
) -> SsrSnapshot {
    feature.extract(
        histories,
        scene_id,
        camera,
        temporal_serial,
        &settings.uniforms,
        size,
        temporal_serial,
        temporal_serial as u32,
        false,
        GlobalStateKey::new(1, scene_id),
    )
}

fn commit(histories: &mut HistoryRegistry, snapshot: SsrSnapshot) -> bool {
    histories.commit(
        snapshot.history.commit,
        &snapshot.history.next.map(|texture| texture.id),
    )
}

struct TestNode;

impl<'a> PassNode<'a> for TestNode {
    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
}

fn test_frame_config() -> FrameConfig {
    FrameConfig {
        width: 1280,
        height: 720,
        depth_format: wgpu::TextureFormat::Depth32Float,
        msaa_samples: 1,
        surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
        hdr_format: HDR_TEXTURE_FORMAT,
    }
}

fn register_test_inputs(
    graph: &mut RenderGraph<'_>,
) -> (TextureNodeId, TextureNodeId, TextureNodeId) {
    let color = graph.register_texture(
        "SSR_Test_Color",
        TextureDesc::new_2d(
            1280,
            720,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        ),
        true,
    );
    let data = graph.register_texture(
        "SSR_Test_Data",
        TextureDesc::new_2d(
            1280,
            720,
            SSR_TEXTURE_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        ),
        true,
    );
    let depth = graph.register_texture(
        "SSR_Test_Depth",
        TextureDesc::new_2d(
            1280,
            720,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
        ),
        true,
    );
    (color, depth, data)
}

fn add_root_consumer(graph: &mut RenderGraph<'_>, source: TextureNodeId) {
    let output = graph.register_texture(
        "SSR_Test_Output",
        TextureDesc::new_2d(
            1280,
            720,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
        true,
    );
    graph.add_pass("SSR_Test_Root", |builder| {
        builder.read_texture(source);
        builder.write_texture(output);
        (TestNode, ())
    });
}

#[test]
fn computes_trace_resolution_from_quality_budget() {
    assert_eq!(ssr_trace_resolution(1, 1, true), (1, 1));
    assert_eq!(ssr_trace_resolution(1920, 1080, true), (960, 540));
    assert_eq!(ssr_trace_resolution(2560, 1440, false), (2560, 1440));
}

#[test]
fn history_validity_comes_only_from_grouped_commit() {
    let mut feature = SsrFeature::new();
    let mut histories = HistoryRegistry::new();
    let mut settings = SsrSettings::default();
    settings.set_history_flags(CUSTOM_RUNTIME_BIT);
    let camera = camera(7);

    let first = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        1,
        1,
        (1920, 1080),
    );
    assert!(!first.history.valid);
    assert_eq!(first.uniforms.frame_params.w, CUSTOM_RUNTIME_BIT);
    assert_eq!(settings.uniforms.read().frame_params.w, CUSTOM_RUNTIME_BIT);

    assert!(commit(&mut histories, first));
    let second = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        1,
        2,
        (1920, 1080),
    );
    assert!(second.history.valid);
    assert_eq!(
        second.uniforms.frame_params.w,
        CUSTOM_RUNTIME_BIT | SSR_HISTORY_FLAG_VALID
    );
}

#[test]
fn camera_cut_and_resolution_change_invalidate_history() {
    let mut feature = SsrFeature::new();
    let mut histories = HistoryRegistry::new();
    let mut settings = SsrSettings::default();
    let mut camera = camera(8);

    let first = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        1,
        1,
        (1920, 1080),
    );
    assert!(commit(&mut histories, first));

    settings.set_half_resolution_enabled(false);
    let resized = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        1,
        2,
        (1920, 1080),
    );
    assert!(!resized.history.valid);
    assert_eq!(resized.trace_resolution, (1920, 1080));
    assert!(commit(&mut histories, resized));

    camera.camera_cut = 1;
    let cut = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        1,
        3,
        (1920, 1080),
    );
    assert!(!cut.history.valid);
    assert_ne!(cut.uniforms.frame_params.w & SSR_HISTORY_FLAG_CAMERA_CUT, 0);
}

#[test]
fn full_resolution_change_invalidates_when_half_extent_is_unchanged() {
    let mut feature = SsrFeature::new();
    let mut histories = HistoryRegistry::new();
    let settings = SsrSettings::default();
    let camera = camera(9);

    let first = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        1,
        1,
        (1920, 1080),
    );
    assert_eq!(first.trace_resolution, (960, 540));
    assert!(commit(&mut histories, first));

    let odd_resize = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        1,
        2,
        (1921, 1080),
    );
    assert_eq!(odd_resize.trace_resolution, (960, 540));
    assert!(!odd_resize.history.valid);
}

#[test]
fn alternating_domains_keep_independent_history() {
    let mut feature = SsrFeature::new();
    let mut histories = HistoryRegistry::new();
    let settings = SsrSettings::default();
    let camera_a = camera(10);
    let camera_b = camera(11);

    let mut previous = [None, None];
    for serial in 1..=4 {
        for (index, camera) in [&camera_a, &camera_b].into_iter().enumerate() {
            let global_frame = 100 + serial * 2 + index as u64;
            let snapshot = feature.extract(
                &mut histories,
                1,
                camera,
                serial,
                &settings.uniforms,
                (1280, 720),
                global_frame,
                global_frame as u32,
                false,
                GlobalStateKey::new(1, 1),
            );
            assert_eq!(snapshot.uniforms.frame_params.x, serial as u32);
            assert_eq!(snapshot.history.valid, serial > 1);
            if let Some(previous) = previous[index] {
                assert_eq!(snapshot.history.previous[0].id, previous);
            }
            previous[index] = Some(snapshot.history.next[0].id);
            assert!(commit(&mut histories, snapshot));
        }
        assert_ne!(previous[0], previous[1]);
    }
}

#[derive(Clone, Copy)]
enum RootDemand {
    #[cfg(feature = "debug_view")]
    TraceDiagnostic,
    #[cfg(feature = "debug_view")]
    Resolved,
    Merged,
}

struct RootDemandCase {
    name: &'static str,
    root: RootDemand,
    expected_live: &'static [&'static str],
    expected_history_valid: bool,
}

#[test]
fn root_demand_matrix_preserves_live_chain_and_history_commit() {
    let cases = [
        #[cfg(feature = "debug_view")]
        RootDemandCase {
            name: "trace diagnostic",
            root: RootDemand::TraceDiagnostic,
            expected_live: &["SSR_Trace", "SSR_Test_Root"],
            expected_history_valid: false,
        },
        #[cfg(feature = "debug_view")]
        RootDemandCase {
            name: "resolved diagnostic",
            root: RootDemand::Resolved,
            expected_live: &["SSR_Trace", "SSR_Temporal", "SSR_Resolve", "SSR_Test_Root"],
            expected_history_valid: true,
        },
        RootDemandCase {
            name: "merged output",
            root: RootDemand::Merged,
            expected_live: &[
                "SSR_Trace",
                "SSR_Temporal",
                "SSR_Resolve",
                "SSR_Merge",
                "SSR_Test_Root",
            ],
            expected_history_valid: true,
        },
    ];

    for (case_index, case) in cases.into_iter().enumerate() {
        let mut feature = SsrFeature::new();
        let mut histories = HistoryRegistry::new();
        let settings = SsrSettings::default();
        let camera = camera(13 + case_index as u128);
        let snapshot = extract(
            &mut feature,
            &mut histories,
            &settings,
            &camera,
            1,
            1,
            (1280, 720),
        );
        #[cfg(feature = "debug_view")]
        let snapshot = {
            let mut snapshot = snapshot;
            if matches!(case.root, RootDemand::TraceDiagnostic) {
                snapshot.trace_diagnostics = true;
            }
            snapshot
        };

        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let frame_config = test_frame_config();
        let (color, depth, data) = register_test_inputs(&mut graph);
        let mut ctx = GraphBuilderContext::new(&mut graph, &frame_config, 0);
        let outputs = feature.add_to_graph(
            &mut ctx, snapshot, color, depth, data, data, data, data, data,
        );
        let root = match case.root {
            #[cfg(feature = "debug_view")]
            RootDemand::TraceDiagnostic => outputs
                .raw_diagnostic
                .expect("trace diagnostic output missing"),
            #[cfg(feature = "debug_view")]
            RootDemand::Resolved => outputs.clean_reflection,
            RootDemand::Merged => outputs.merged_color,
        };
        add_root_consumer(&mut graph, root);

        graph.compile_topology();
        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(live, case.expected_live, "{} live chain", case.name);

        graph.commit_submitted_histories(&mut histories);
        let followup = histories.plan_set(
            snapshot.history.commit.set_id,
            snapshot.history.next.map(|texture| texture.desc),
            2,
            false,
        );
        assert_eq!(
            followup.valid, case.expected_history_valid,
            "{} history validity",
            case.name
        );

        graph.finish();
        assert!(feature.raw.pipelines.is_empty());
        assert!(feature.raw.layout.is_none());
        assert!(feature.temporal.pipeline_id.is_none());
        assert!(feature.temporal.layout.is_none());
        assert!(feature.spatial.pipeline_id.is_none());
        assert!(feature.spatial.layout.is_none());
        assert!(feature.merge.pipeline_id.is_none());
        assert!(feature.merge.layout.is_none());
    }
}

#[test]
fn abandoned_domain_expires_after_ttl_and_reenters_with_invalid_history() {
    let mut histories = HistoryRegistry::new();
    let mut feature = SsrFeature::new();
    let settings = SsrSettings::default();
    let camera = camera(41);

    let first = feature.extract(
        &mut histories,
        1,
        &camera,
        1,
        &settings.uniforms,
        (1280, 720),
        10,
        1,
        false,
        GlobalStateKey::new(1, 1),
    );
    assert!(commit(&mut histories, first));

    assert_eq!(feature.garbage_collect(&mut histories, 12, 2), 0);
    assert_eq!(feature.histories.len(), 1);
    assert_eq!(feature.garbage_collect(&mut histories, 13, 2), 1);
    assert!(feature.histories.is_empty());

    let replacement = feature.extract(
        &mut histories,
        1,
        &camera,
        2,
        &settings.uniforms,
        (1280, 720),
        13,
        2,
        false,
        GlobalStateKey::new(1, 1),
    );
    assert!(!replacement.history.valid);
    assert_ne!(
        replacement.history.commit.set_id,
        first.history.commit.set_id
    );
    assert_ne!(replacement.history.next[0].id, first.history.next[0].id);
}

#[test]
fn reentry_expires_stale_history_even_without_a_periodic_sweep() {
    let mut histories = HistoryRegistry::new();
    let mut feature = SsrFeature::new();
    let settings = SsrSettings::default();
    let camera = camera(42);

    let first = feature.extract(
        &mut histories,
        1,
        &camera,
        1,
        &settings.uniforms,
        (1280, 720),
        1,
        1,
        false,
        GlobalStateKey::new(1, 1),
    );
    assert!(commit(&mut histories, first));

    let replacement = feature.extract(
        &mut histories,
        1,
        &camera,
        2,
        &settings.uniforms,
        (1280, 720),
        TEMPORAL_DOMAIN_TTL_FRAMES + 2,
        2,
        false,
        GlobalStateKey::new(1, 1),
    );
    assert!(!replacement.history.valid);
    assert_ne!(
        replacement.history.commit.set_id,
        first.history.commit.set_id
    );
}
