use super::*;
use crate::graph::core::graph::FrameConfig;
use crate::graph::core::{FrameArena, GraphStorage, RenderGraph};
use myth_resources::ssgi::SsgiSettings;
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
    feature: &mut SsgiFeature,
    histories: &mut HistoryRegistry,
    settings: &SsgiSettings,
    camera: &RenderCamera,
    temporal_serial: u64,
    size: (u32, u32),
) -> SsgiSnapshot {
    feature.extract(
        histories,
        1,
        camera,
        temporal_serial,
        temporal_serial,
        &settings.uniforms,
        size,
        temporal_serial as u32,
        None,
        GlobalStateKey::new(1, 1),
    )
}

fn commit_indirect(histories: &mut HistoryRegistry, snapshot: SsgiSnapshot) -> bool {
    histories.commit(
        snapshot.indirect_history.commit,
        &snapshot.indirect_history.next.map(|texture| texture.id),
    )
}

fn commit_source(histories: &mut HistoryRegistry, snapshot: SsgiSnapshot) -> bool {
    histories.commit(
        snapshot.source_history.commit,
        &snapshot.source_history.next.map(|texture| texture.id),
    )
}

struct TestNode;

impl<'a> PassNode<'a> for TestNode {
    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
}

fn frame_config() -> FrameConfig {
    FrameConfig {
        width: 1280,
        height: 720,
        depth_format: wgpu::TextureFormat::Depth32Float,
        msaa_samples: 1,
        surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
        hdr_format: HDR_TEXTURE_FORMAT,
    }
}

fn register_inputs(
    graph: &mut RenderGraph<'_>,
) -> (
    TextureNodeId,
    TextureNodeId,
    TextureNodeId,
    TextureNodeId,
    TextureNodeId,
    TextureNodeId,
) {
    let sampled = wgpu::TextureUsages::TEXTURE_BINDING;
    let color = graph.register_texture(
        "SSGI_Test_Color",
        TextureDesc::new_2d(1280, 720, HDR_TEXTURE_FORMAT, sampled),
        true,
    );
    let depth = graph.register_texture(
        "SSGI_Test_Depth",
        TextureDesc::new_2d(1280, 720, wgpu::TextureFormat::Depth32Float, sampled),
        true,
    );
    let hiz = graph.register_texture(
        "SSGI_Test_HiZ",
        TextureDesc::new_2d(1280, 720, wgpu::TextureFormat::R32Float, sampled),
        true,
    );
    let data = graph.register_texture(
        "SSGI_Test_Data",
        TextureDesc::new_2d(1280, 720, SSGI_TEXTURE_FORMAT, sampled),
        true,
    );
    (color, depth, hiz, data, data, data)
}

fn add_graph<'a>(
    feature: &'a mut SsgiFeature,
    graph: &mut RenderGraph<'a>,
    snapshot: SsgiSnapshot,
) -> SsgiOutputs {
    let config = frame_config();
    let (color, depth, hiz, normals, velocity, material) = register_inputs(graph);
    let mut ctx = GraphBuilderContext::new(graph, &config, 0);
    feature.add_to_graph(
        &mut ctx, snapshot, color, depth, hiz, normals, velocity, material, None,
    )
}

fn add_root(graph: &mut RenderGraph<'_>, source: TextureNodeId) {
    let output = graph.register_texture(
        "SSGI_Test_Output",
        TextureDesc::new_2d(
            1280,
            720,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
        true,
    );
    graph.add_pass("SSGI_Test_Root", |builder| {
        builder.read_texture(source);
        builder.write_texture(output);
        (TestNode, ())
    });
}

fn followup_validity(histories: &mut HistoryRegistry, snapshot: SsgiSnapshot) -> (bool, bool) {
    let indirect = histories.plan_set(
        snapshot.indirect_history.commit.set_id,
        snapshot.indirect_history.next.map(|texture| texture.desc),
        2,
        false,
    );
    let source = histories.plan_set(
        snapshot.source_history.commit.set_id,
        snapshot.source_history.next.map(|texture| texture.desc),
        2,
        false,
    );
    (indirect.valid, source.valid)
}

#[test]
fn extract_does_not_mutate_scene_uniforms_and_validity_requires_commit() {
    let mut feature = SsgiFeature::new();
    let mut histories = HistoryRegistry::new();
    let mut settings = SsgiSettings::default();
    settings.set_history_flags(CUSTOM_RUNTIME_BIT);
    let camera = camera(7);
    let before = *settings.uniforms.read();

    let first = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        1,
        (1920, 1080),
    );
    assert!(!first.indirect_history.valid);
    assert!(!first.source_history.valid);
    assert_eq!(first.uniforms.frame_params.w, CUSTOM_RUNTIME_BIT);
    assert_eq!(
        bytemuck::bytes_of(&*settings.uniforms.read()),
        bytemuck::bytes_of(&before)
    );

    assert!(commit_indirect(&mut histories, first));
    assert!(commit_source(&mut histories, first));
    let second = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        2,
        (1920, 1080),
    );
    assert!(second.indirect_history.valid);
    assert!(second.source_history.valid);
    assert_eq!(
        second.uniforms.frame_params.w,
        CUSTOM_RUNTIME_BIT | HISTORY_FLAG_INDIRECT_VALID | HISTORY_FLAG_SOURCE_VALID
    );
}

#[test]
fn taa_source_is_gated_by_committed_ssgi_source_history() {
    let mut feature = SsgiFeature::new();
    let mut histories = HistoryRegistry::new();
    let settings = SsgiSettings::default();
    let camera = camera(70);
    let taa_set = histories.create_set();
    let taa_desc = TextureDesc::new_2d(
        1280,
        720,
        HDR_TEXTURE_FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let taa_plan = histories.plan_set(taa_set, [taa_desc], 1, false);
    let taa_previous = taa_plan.previous[0];

    let first = feature.extract(
        &mut histories,
        1,
        &camera,
        1,
        1,
        &settings.uniforms,
        (1280, 720),
        1,
        Some(taa_previous),
        GlobalStateKey::new(1, 1),
    );
    assert!(first.taa_history.is_none());
    assert!(commit_indirect(&mut histories, first));
    assert!(commit_source(&mut histories, first));

    let second = feature.extract(
        &mut histories,
        1,
        &camera,
        2,
        2,
        &settings.uniforms,
        (1280, 720),
        2,
        Some(taa_previous),
        GlobalStateKey::new(1, 1),
    );
    assert_eq!(second.taa_history, Some(taa_previous));
}

#[test]
fn alternating_scene_camera_domains_keep_independent_continuity() {
    let mut feature = SsgiFeature::new();
    let mut histories = HistoryRegistry::new();
    let mut settings = SsgiSettings::default();
    settings.set_checkerboard_enabled(true);
    let camera_a = camera(71);
    let camera_b = camera(72);

    let mut previous = [None, None];
    let mut sampled_phases = [0_u32; 2];
    for serial in 1..=4 {
        for (index, camera) in [&camera_a, &camera_b].into_iter().enumerate() {
            let scene_id = index as u32 + 1;
            let global_frame = 100 + serial * 2 + index as u64;
            let snapshot = feature.extract(
                &mut histories,
                scene_id,
                camera,
                serial,
                global_frame,
                &settings.uniforms,
                (1280, 720),
                global_frame as u32,
                None,
                GlobalStateKey::new(1, scene_id),
            );
            assert_eq!(snapshot.uniforms.frame_params.x, serial as u32);
            sampled_phases[index] |= 1 << (snapshot.uniforms.frame_params.x & 3);
            assert_eq!(snapshot.indirect_history.valid, serial > 1);
            assert_eq!(snapshot.source_history.valid, serial > 1);
            if let Some(previous) = previous[index] {
                assert_eq!(snapshot.indirect_history.previous[0].id, previous);
            }
            previous[index] = Some(snapshot.indirect_history.next[0].id);
            assert!(commit_indirect(&mut histories, snapshot));
            assert!(commit_source(&mut histories, snapshot));
        }
        assert_ne!(previous[0], previous[1]);
    }
    assert_eq!(sampled_phases, [0b1111, 0b1111]);
}

#[test]
fn odd_full_resize_and_camera_cut_reset_both_sets() {
    let mut feature = SsgiFeature::new();
    let mut histories = HistoryRegistry::new();
    let settings = SsgiSettings::default();
    let mut camera = camera(8);
    let first = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        1,
        (1280, 720),
    );
    assert!(commit_indirect(&mut histories, first));
    assert!(commit_source(&mut histories, first));

    let resized = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        2,
        (1280, 721),
    );
    assert_eq!(first.half_resolution, resized.half_resolution);
    assert!(!resized.indirect_history.valid);
    assert!(!resized.source_history.valid);
    assert!(commit_indirect(&mut histories, resized));
    assert!(commit_source(&mut histories, resized));

    camera.camera_cut = 1;
    let cut = extract(
        &mut feature,
        &mut histories,
        &settings,
        &camera,
        3,
        (1280, 721),
    );
    assert!(!cut.indirect_history.valid);
    assert!(!cut.source_history.valid);
    assert_ne!(cut.uniforms.frame_params.w & HISTORY_FLAG_CAMERA_CUT, 0);
}

#[test]
fn abandoned_domain_gc_recreates_both_sets_as_invalid() {
    let mut feature = SsgiFeature::new();
    let mut histories = HistoryRegistry::new();
    let settings = SsgiSettings::default();
    let camera = camera(9);
    let first = feature.extract(
        &mut histories,
        1,
        &camera,
        1,
        10,
        &settings.uniforms,
        (1280, 720),
        1,
        None,
        GlobalStateKey::new(1, 1),
    );
    assert!(commit_indirect(&mut histories, first));
    assert!(commit_source(&mut histories, first));
    assert_eq!(feature.garbage_collect(&mut histories, 12, 2), 0);
    assert_eq!(feature.garbage_collect(&mut histories, 13, 2), 1);

    let replacement = feature.extract(
        &mut histories,
        1,
        &camera,
        2,
        13,
        &settings.uniforms,
        (1280, 720),
        2,
        None,
        GlobalStateKey::new(1, 1),
    );
    assert!(!replacement.indirect_history.valid);
    assert!(!replacement.source_history.valid);
    assert_ne!(
        replacement.indirect_history.commit.set_id,
        first.indirect_history.commit.set_id
    );
    assert_ne!(
        replacement.source_history.commit.set_id,
        first.source_history.commit.set_id
    );
}

#[test]
fn reentry_expires_stale_domain_even_without_periodic_gc() {
    let mut feature = SsgiFeature::new();
    let mut histories = HistoryRegistry::new();
    let settings = SsgiSettings::default();
    let camera = camera(73);
    let first = feature.extract(
        &mut histories,
        1,
        &camera,
        1,
        1,
        &settings.uniforms,
        (1280, 720),
        1,
        None,
        GlobalStateKey::new(1, 1),
    );
    assert!(commit_indirect(&mut histories, first));
    assert!(commit_source(&mut histories, first));

    let replacement = feature.extract(
        &mut histories,
        1,
        &camera,
        2,
        TEMPORAL_DOMAIN_TTL_FRAMES + 2,
        &settings.uniforms,
        (1280, 720),
        2,
        None,
        GlobalStateKey::new(1, 1),
    );
    assert!(!replacement.indirect_history.valid);
    assert!(!replacement.source_history.valid);
    assert_ne!(
        replacement.indirect_history.commit.set_id,
        first.indirect_history.commit.set_id
    );
    assert_ne!(
        replacement.source_history.commit.set_id,
        first.source_history.commit.set_id
    );
}

#[test]
fn root_demand_commits_only_live_history_sets() {
    #[derive(Clone, Copy)]
    enum RootDemand {
        Raw,
        Clean,
        Merged,
    }

    struct Case {
        name: &'static str,
        demand: RootDemand,
        expected_live: &'static [&'static str],
        expected_validity: (bool, bool),
    }

    let cases = [
        Case {
            name: "raw",
            demand: RootDemand::Raw,
            expected_live: &["SSGI_Raw", "SSGI_Test_Root"],
            expected_validity: (false, false),
        },
        Case {
            name: "clean",
            demand: RootDemand::Clean,
            expected_live: &[
                "SSGI_Raw",
                "SSGI_Temporal",
                "SSGI_ATrous_Step_1",
                "SSGI_ATrous_Step_2",
                "SSGI_ATrous_Step_4",
                "SSGI_Test_Root",
            ],
            expected_validity: (true, false),
        },
        Case {
            name: "merged",
            demand: RootDemand::Merged,
            expected_live: &[
                "SSGI_Raw",
                "SSGI_Temporal",
                "SSGI_ATrous_Step_1",
                "SSGI_ATrous_Step_2",
                "SSGI_ATrous_Step_4",
                "SSGI_Merge",
                "SSGI_Test_Root",
            ],
            expected_validity: (true, true),
        },
    ];

    for (index, case) in cases.iter().enumerate() {
        let mut feature = SsgiFeature::new();
        let mut histories = HistoryRegistry::new();
        let settings = SsgiSettings::default();
        let camera = camera(11 + index as u128);
        let snapshot = extract(
            &mut feature,
            &mut histories,
            &settings,
            &camera,
            1,
            (1280, 720),
        );
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let outputs = add_graph(&mut feature, &mut graph, snapshot);
        let root = match case.demand {
            RootDemand::Raw => outputs.raw_indirect,
            RootDemand::Clean => outputs.clean_indirect,
            RootDemand::Merged => outputs.merged_color,
        };
        add_root(&mut graph, root);

        graph.compile_topology();
        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(live, case.expected_live, "{} root", case.name);

        graph.commit_submitted_histories(&mut histories);
        assert_eq!(
            followup_validity(&mut histories, snapshot),
            case.expected_validity,
            "{} root",
            case.name
        );

        graph.finish();
        assert!(feature.raw.pipelines.is_empty());
        assert!(feature.raw.layout.is_none());
        assert!(feature.temporal.pipeline_id.is_none());
        assert!(feature.temporal.layout.is_none());
        assert!(
            feature
                .atrous
                .iter()
                .all(|stage| stage.pipeline_id.is_none() && stage.layout.is_none())
        );
        assert!(feature.merge.pipeline_id.is_none());
        assert!(feature.merge.layout.is_none());
    }
}
