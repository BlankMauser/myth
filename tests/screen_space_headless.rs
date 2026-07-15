use myth::prelude::*;
use myth::renderer::graph::RdgAttemptOutcome;
use myth::resources::{ImageDimension, PixelFormat};

fn assert_not_black(pixels: &[u8], label: &str) {
    let any_color = pixels
        .chunks_exact(4)
        .any(|px| px[0] > 0 || px[1] > 0 || px[2] > 0);
    assert!(any_color, "{label}: rendered image is entirely black");
}

fn render_submitted_frame(engine: &mut Engine, label: &str) -> usize {
    engine.update(1.0 / 60.0);
    assert!(
        engine.render_active_scene(),
        "{label}: frame was not rendered"
    );

    let diagnostics = engine
        .renderer
        .latest_rdg_diagnostics()
        .expect("a rendered frame must publish RDG diagnostics");
    assert_eq!(
        diagnostics.outcome(),
        RdgAttemptOutcome::Submitted,
        "{label}"
    );
    let live = diagnostics
        .live_passes()
        .expect("a submitted graph must report its DPE result");
    assert!(live > 0, "{label}: submitted graph had no live passes");
    assert!(diagnostics.declared_passes() >= live, "{label}");
    assert_eq!(diagnostics.prepared_passes(), live, "{label}");
    assert_eq!(diagnostics.executed_passes(), live, "{label}");
    assert!(diagnostics.queue_submitted(), "{label}");
    live
}

fn assert_readback(engine: &mut Engine, width: u32, height: u32, label: &str) {
    let pixels = engine.readback_pixels().expect("readback failed");
    assert_eq!(pixels.len(), (width * height * 4) as usize, "{label}");
    assert_not_black(&pixels, label);
}

fn solid_image(color: [u8; 4]) -> Image {
    Image::new(
        8,
        8,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(color.repeat(8 * 8)),
    )
}

fn render_and_read(engine: &mut Engine, frames: usize, label: &str) -> Vec<u8> {
    for _ in 0..frames {
        render_submitted_frame(engine, label);
    }
    engine.readback_pixels().expect("readback failed")
}

fn mean_rgb_delta(left: &[u8], right: &[u8]) -> f32 {
    assert_eq!(left.len(), right.len());
    let (sum, samples) = left.chunks_exact(4).zip(right.chunks_exact(4)).fold(
        (0_u64, 0_u64),
        |(sum, samples), (a, b)| {
            let delta = (i16::from(a[0]) - i16::from(b[0])).unsigned_abs() as u64
                + (i16::from(a[1]) - i16::from(b[1])).unsigned_abs() as u64
                + (i16::from(a[2]) - i16::from(b[2])).unsigned_abs() as u64;
            (sum + delta, samples + 3)
        },
    );
    sum as f32 / samples as f32
}

#[test]
fn temporal_screen_space_pipeline_survives_runtime_transitions() {
    let mut engine = Engine::default();
    let mut width: u32 = 128;
    let mut height: u32 = 128;

    pollster::block_on(engine.init_headless(width, height, None)).expect("headless init failed");

    let scene = engine.scene_manager.create_active();
    scene.ssao.set_enabled(true);
    scene.ssgi.set_enabled(true);
    scene.ssgi.set_quality(SsgiQuality::Low);
    scene.ssr.set_enabled(true);
    scene.ssr.set_max_steps(12);
    scene.ssr.set_spatial_radius(1);

    let floor = PhysicalMaterial::new(Vec4::new(0.75, 0.75, 0.78, 1.0))
        .with_roughness(0.28)
        .with_metalness(0.0);
    let box_mat = PhysicalMaterial::new(Vec4::new(0.85, 0.3, 0.22, 1.0))
        .with_roughness(0.14)
        .with_metalness(0.92);

    let floor_node = scene.spawn_box(6.0, 0.2, 6.0, floor, &engine.assets);
    scene.node(&floor_node).set_position(0.0, -1.2, 0.0);

    let box_node = scene.spawn_box(1.2, 1.2, 1.2, box_mat, &engine.assets);
    scene.node(&box_node).set_position(0.0, 0.0, 0.0);

    scene.add_light(Light::new_directional(Vec3::ONE, 4.0));

    let camera_a = scene.add_camera(Camera::new_perspective(
        45.0,
        width as f32 / height as f32,
        0.1,
    ));
    scene
        .node(&camera_a)
        .set_position(0.0, 1.4, 4.5)
        .look_at(Vec3::new(0.0, -0.2, 0.0));
    let camera_b = scene.add_camera(Camera::new_perspective(
        45.0,
        width as f32 / height as f32,
        0.1,
    ));
    scene
        .node(&camera_b)
        .set_position(2.5, 1.1, 3.8)
        .look_at(Vec3::new(0.0, -0.2, 0.0));
    scene.active_camera = Some(camera_a);

    let mut warm_live = 0;
    for frame in 0..3 {
        warm_live = render_submitted_frame(&mut engine, &format!("warm frame {frame}"));
    }
    assert_readback(&mut engine, width, height, "warm temporal pipeline");

    {
        let scene = engine
            .scene_manager
            .active_scene_mut()
            .expect("active scene missing");
        scene
            .get_camera_mut(camera_a)
            .expect("primary camera missing")
            .mark_camera_cut();
    }
    render_submitted_frame(&mut engine, "camera cut");

    {
        let scene = engine
            .scene_manager
            .active_scene_mut()
            .expect("active scene missing");
        scene.active_camera = Some(camera_b);
    }
    render_submitted_frame(&mut engine, "alternate camera");
    {
        let scene = engine
            .scene_manager
            .active_scene_mut()
            .expect("active scene missing");
        scene.active_camera = Some(camera_a);
    }
    render_submitted_frame(&mut engine, "return to primary camera");

    width = 129;
    height = 127;
    engine.resize(width, height);
    render_submitted_frame(&mut engine, "odd resize cold frame");
    render_submitted_frame(&mut engine, "odd resize warm frame");
    assert_readback(&mut engine, width, height, "odd resized temporal pipeline");

    {
        let scene = engine
            .scene_manager
            .active_scene_mut()
            .expect("active scene missing");
        scene.ssgi.set_enabled(false);
        scene.ssr.set_enabled(false);
    }
    let disabled_live = render_submitted_frame(&mut engine, "temporal effects disabled");
    assert!(
        disabled_live < warm_live,
        "disabling SSGI and SSR must remove their live graph work"
    );

    {
        let scene = engine
            .scene_manager
            .active_scene_mut()
            .expect("active scene missing");
        scene.ssgi.set_enabled(true);
        scene.ssr.set_enabled(true);
    }
    let reenabled_live = render_submitted_frame(&mut engine, "temporal effects re-enabled");
    assert!(
        reenabled_live > disabled_live,
        "re-enabling SSGI and SSR must restore their live graph work"
    );
    render_submitted_frame(&mut engine, "re-enabled warm frame");
    assert_readback(&mut engine, width, height, "re-enabled temporal pipeline");
}

#[test]
fn pending_texture_becomes_visible_and_refreshes_under_the_same_handles() {
    let mut engine = Engine::default();
    pollster::block_on(engine.init_headless(96, 96, None)).expect("headless init failed");

    let image = engine.assets.images.reserve();
    let texture = engine
        .assets
        .textures
        .add(Texture::new_2d(Some("streamed color"), image));

    let scene = engine.scene_manager.create_active();
    scene.spawn_box(
        2.5,
        2.5,
        2.5,
        UnlitMaterial::new(Vec4::ONE).with_map(texture),
        &engine.assets,
    );
    let camera = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&camera)
        .set_position(0.0, 0.0, 5.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(camera);

    let pending = render_and_read(&mut engine, 2, "pending image");

    engine
        .assets
        .images
        .insert_ready(image, solid_image([240, 24, 24, 255]));
    let ready = render_and_read(&mut engine, 3, "ready image");
    assert!(
        mean_rgb_delta(&pending, &ready) > 4.0,
        "making the pending image ready did not change the rendered result"
    );

    assert_eq!(
        engine
            .assets
            .images
            .update(image, solid_image([24, 48, 240, 255])),
        Some(2),
        "same-handle image refresh did not advance its content version"
    );
    let refreshed = render_and_read(&mut engine, 3, "refreshed image");
    assert!(
        mean_rgb_delta(&ready, &refreshed) > 4.0,
        "same-handle image refresh did not reach the live material"
    );
}
