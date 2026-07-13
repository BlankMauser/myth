//! RDG Shadow Render Pass
//!
//! Renders shadow depth maps for all shadow-casting lights. Two transient
//! textures are managed per frame:
//!
//! - **`Texture2DArray`** — directional light cascades and spot light shadows.
//! - **`TextureCubeArray`** — point light omnidirectional shadows (6 faces per
//!   light, hardware-seamless PCF across edges).
//!
//! VP matrices are uploaded to a per-layer dynamic uniform buffer shared by
//! both texture types.
//!
//! # RDG Integration
//!
//! Both shadow map textures are **transient RDG resources** — their lifetimes
//! are fully managed by the render graph compiler. Consumer passes (Opaque,
//! Transparent, etc.) declare explicit `read_texture` dependencies to
//! establish DAG ordering and enable automatic memory aliasing.
//!
//! # Data Flow
//!
//! ```text
//! RenderLists::active_views (shadow views)     ← built by scene extraction
//! RenderLists::shadow_queues (CPU intents)     ← built by collect_draw_intents
//!         │
//!         ▼
//!  Shadow_VP_Upload (transient uniform producer)
//!         │
//!         ├──────────────┐
//!         ▼              ▼
//!  Shadow_2D_Pass  Shadow_Cube_Pass
//!         │
//!         ▼  (TextureNodeId via explicit wiring)
//!  Consumer passes (Opaque / Transparent)
//!         read_texture(shadow_2d_id) → DAG edge
//!         read_texture(shadow_cube_id) → DAG edge
//! ```

use std::sync::OnceLock;

use glam::Mat4;
use myth_scene::light::LightKind;
use smallvec::SmallVec;

use crate::core::binding::PreparedBindGroupId;
use crate::core::gpu::{Tracked, binding_layout};
use crate::core::view::ViewTarget;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::scene_submission::{SceneSubmissionKey, SceneSubmissionPlanId};
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, FrameArena, PassNode, PrepareContext, TextureDesc,
    TextureNodeId,
};
use crate::graph::extracted::ExtractedScene;
use crate::graph::frame::{RenderLists, ShadowLightInstance};
use crate::graph::passes::draw::submit_scene_submission;

/// Output returned by [`ShadowFeature::add_to_graph`].
///
/// Consumer passes wire `read_texture` dependencies against these IDs.
pub struct ShadowOutput {
    /// 2D array shadow map (directional/spot). `None` when unused.
    pub shadow_2d: Option<TextureNodeId>,
    /// Cube array shadow map (point lights). `None` when unused.
    pub shadow_cube: Option<TextureNodeId>,
}

/// Lightweight frame snapshot captured while Composer builds the graph.
///
/// It contains only CPU scene data. Physical buffers, layouts, and bind groups are
/// deliberately absent and are materialized by the live upload node after DPE.
#[derive(Clone, Copy)]
pub(crate) struct ShadowSnapshot<'a> {
    d2_lights: &'a [ShadowLightInstance],
    cube_lights: &'a [ShadowLightInstance],
    shadow_2d_resolution: u32,
    shadow_cube_resolution: u32,
}

/// Shadow rendering feature.
///
/// Owns only the cross-frame static shadow uniform layout and its alignment
/// requirement. Per-frame view data is returned from [`Self::extract`], while
/// the VP buffer and bind group are created by live RDG nodes.
///
/// The shadow map textures are **transient RDG resources** — their physical
/// memory is allocated by the graph compiler and can be aliased with other
/// non-overlapping intermediates. If no shadow-casting lights are active,
/// no GPU memory is allocated at all.
pub struct ShadowFeature {
    /// Aligned stride between consecutive VP matrices.
    uniform_stride: u32,
    /// Cross-frame static Group 0 layout. Created only by a live upload node.
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl ShadowFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let min_alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let stride = align_to(std::mem::size_of::<Mat4>() as u32, min_alignment);

        Self {
            uniform_stride: stride,
            bind_group_layout: None,
        }
    }

    fn ensure_bind_group_layout(
        &mut self,
        device: &wgpu::Device,
    ) -> &Tracked<wgpu::BindGroupLayout> {
        self.bind_group_layout.get_or_insert_with(|| {
            Tracked::new(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Shadow Light BGL"),
                    entries: &[binding_layout::buffer(
                        0,
                        wgpu::ShaderStages::VERTEX,
                        wgpu::BufferBindingType::Uniform,
                        true,
                        wgpu::BufferSize::new(std::mem::size_of::<Mat4>() as u64),
                    )],
                }),
            )
        })
    }
}

impl ShadowFeature {
    /// Classify shadow views and copy the minimum scene data needed to build
    /// graph descriptors. This method is CPU-only and performs no WGPU work.
    #[must_use]
    #[allow(clippy::unused_self)] // Keep a uniform Feature::extract API.
    pub(crate) fn extract<'a>(
        &self,
        render_lists: &RenderLists,
        extracted_scene: &ExtractedScene,
        arena: &'a FrameArena,
    ) -> ShadowSnapshot<'a> {
        let mut d2_lights = SmallVec::<[ShadowLightInstance; 16]>::new();
        let mut cube_lights = SmallVec::<[ShadowLightInstance; 16]>::new();
        let mut shadow_2d_resolution = 1;
        let mut shadow_cube_resolution = 1;

        for view in &render_lists.active_views {
            let ViewTarget::ShadowLight {
                light_id,
                layer_index: view_layer_index,
            } = view.target
            else {
                continue;
            };

            let is_point = extracted_scene
                .lights
                .get(view.light_buffer_index)
                .is_some_and(|light| {
                    light.id == light_id && matches!(light.kind, LightKind::Point(_))
                });
            let (texture_layer_index, target) = if is_point {
                shadow_cube_resolution = shadow_cube_resolution.max(view.viewport_size.0);
                (cube_lights.len() as u32, &mut cube_lights)
            } else {
                shadow_2d_resolution = shadow_2d_resolution.max(view.viewport_size.0);
                (d2_lights.len() as u32, &mut d2_lights)
            };

            target.push(ShadowLightInstance {
                light_id,
                view_layer_index,
                texture_layer_index,
                light_buffer_index: view.light_buffer_index,
                light_view_projection: view.view_projection,
                is_point,
            });
        }

        ShadowSnapshot {
            d2_lights: arena.alloc_slice_copy(&d2_lights),
            cube_lights: arena.alloc_slice_copy(&cube_lights),
            shadow_2d_resolution,
            shadow_cube_resolution,
        }
    }

    /// Create ephemeral shadow pass nodes and add them to the render graph.
    ///
    /// Registers up to two transient `Depth32Float` textures:
    /// - A **2D array** for directional/spot shadow layers.
    /// - A **cube array** for point light shadow faces.
    ///
    /// Returns [`ShadowOutput`] containing the [`TextureNodeId`]s so that
    /// consumer passes can wire explicit read dependencies.
    pub(crate) fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: ShadowSnapshot<'a>,
    ) -> ShadowOutput {
        let mut output = ShadowOutput {
            shadow_2d: None,
            shadow_cube: None,
        };

        if snapshot.d2_lights.is_empty() && snapshot.cube_lights.is_empty() {
            return output;
        }

        let d2_lights = snapshot.d2_lights;
        let cube_lights = snapshot.cube_lights;
        let mut d2_submissions = SmallVec::<[SceneSubmissionPlanId; 16]>::new();
        for light in d2_lights {
            d2_submissions.push(
                ctx.graph
                    .reserve_scene_submission(SceneSubmissionKey::Shadow {
                        light_id: light.light_id,
                        layer_index: light.view_layer_index,
                    }),
            );
        }
        let d2_submissions = ctx.graph.alloc_slice(&d2_submissions);
        let mut cube_submissions = SmallVec::<[SceneSubmissionPlanId; 16]>::new();
        for light in cube_lights {
            cube_submissions.push(
                ctx.graph
                    .reserve_scene_submission(SceneSubmissionKey::Shadow {
                        light_id: light.light_id,
                        layer_index: light.view_layer_index,
                    }),
            );
        }
        let cube_submissions = ctx.graph.alloc_slice(&cube_submissions);
        let prepared = ctx.graph.alloc_shared_state(ShadowPreparedState {
            bind_group: OnceLock::new(),
        });
        let uniform_stride = self.uniform_stride;
        let uniform_count = d2_lights.len() + cube_lights.len();
        let vp_buffer_size = u64::from(uniform_stride) * uniform_count as u64;
        let vp_buffer = ctx.graph.add_pass("Shadow_VP_Upload", |builder| {
            let buffer = builder.create_buffer(
                "Shadow_VP_Buffer",
                BufferDesc::new(
                    vp_buffer_size,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
            );
            (
                ShadowVpUploadNode {
                    feature: self,
                    d2_lights,
                    cube_lights,
                    uniform_stride,
                    buffer,
                    prepared,
                },
                buffer,
            )
        });

        if !d2_lights.is_empty() {
            let desc = TextureDesc::new(
                snapshot.shadow_2d_resolution,
                snapshot.shadow_2d_resolution,
                d2_lights.len() as u32,
                1,
                1,
                wgpu::TextureDimension::D2,
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let shadow_2d_id = ctx.graph.add_pass("Shadow_2D_Pass", |builder| {
                for &submission in d2_submissions {
                    builder.demand_scene_submission(submission);
                }
                builder.read_buffer(vp_buffer);
                let tex_id = builder.create_texture("Shadow_2D_Array", desc);

                let node = Shadow2DPassNode {
                    lights: d2_lights,
                    submissions: d2_submissions,
                    uniform_stride,
                    prepared,
                    texture_id: tex_id,
                };
                (node, tex_id)
            });

            output.shadow_2d = Some(shadow_2d_id);
        }

        if !cube_lights.is_empty() {
            assert!(
                cube_lights.len().is_multiple_of(6),
                "point-light shadow views must form complete six-face cubes"
            );
            let desc = TextureDesc::new(
                snapshot.shadow_cube_resolution,
                snapshot.shadow_cube_resolution,
                cube_lights.len() as u32,
                1,
                1,
                wgpu::TextureDimension::D2,
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let shadow_cube_id = ctx.graph.add_pass("Shadow_Cube_Pass", |builder| {
                for &submission in cube_submissions {
                    builder.demand_scene_submission(submission);
                }
                builder.read_buffer(vp_buffer);
                let tex_id = builder.create_texture("Shadow_Cube_Array", desc);

                let node = ShadowCubePassNode {
                    lights: cube_lights,
                    submissions: cube_submissions,
                    uniform_base: d2_lights.len() as u32,
                    uniform_stride,
                    prepared,
                    texture_id: tex_id,
                };
                (node, tex_id)
            });

            output.shadow_cube = Some(shadow_cube_id);
        }

        output
    }
}

/// Frame-scoped handoff from the live VP producer's prepare step to both
/// render consumers. The cell stores only a prepared-store ID and owns no GPU
/// object.
struct ShadowPreparedState {
    bind_group: OnceLock<PreparedBindGroupId>,
}

/// Real RDG data producer for the shared per-layer VP uniform buffer.
///
/// Because both shadow render nodes declare a read dependency on `buffer`, DPE
/// keeps this node alive exactly when at least one shadow texture is demanded.
struct ShadowVpUploadNode<'a> {
    feature: &'a mut ShadowFeature,
    d2_lights: &'a [ShadowLightInstance],
    cube_lights: &'a [ShadowLightInstance],
    uniform_stride: u32,
    buffer: BufferNodeId,
    prepared: &'a ShadowPreparedState,
}

impl<'a> PassNode<'a> for ShadowVpUploadNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let uniform_data =
            pack_shadow_matrices(self.d2_lights, self.cube_lights, self.uniform_stride);
        ctx.queue
            .write_buffer(ctx.views.get_buffer(self.buffer), 0, &uniform_data);

        let layout = self.feature.ensure_bind_group_layout(ctx.device);
        let matrix_size = wgpu::BufferSize::new(std::mem::size_of::<Mat4>() as u64)
            .expect("Mat4 has a non-zero binding size");
        let bind_group = ctx
            .build_bind_group(layout, Some("Shadow Light BG"))
            .bind_buffer_range(0, self.buffer, 0, matrix_size)
            .build();
        assert!(
            self.prepared.bind_group.set(bind_group).is_ok(),
            "Shadow VP producer prepared more than once"
        );
    }

    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
}

// ─── 2D Array Shadow Pass Node ────────────────────────────────────────────────

/// Renders depth layers for directional and spot light shadows into a 2D
/// array texture.
struct Shadow2DPassNode<'a> {
    lights: &'a [ShadowLightInstance],
    submissions: &'a [SceneSubmissionPlanId],
    uniform_stride: u32,
    prepared: &'a ShadowPreparedState,
    texture_id: TextureNodeId,
}

impl<'a> PassNode<'a> for Shadow2DPassNode<'a> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.lights.is_empty() {
            return;
        }
        debug_assert_eq!(self.lights.len(), self.submissions.len());

        let texture = ctx.get_texture(self.texture_id);
        let bind_group_id = self
            .prepared
            .bind_group
            .get()
            .copied()
            .expect("Shadow VP bind group missing after prepare");
        let bind_group = ctx.resolve_prepared_bind_group(bind_group_id);

        for (uniform_idx, (shadow_light, &submission)) in
            self.lights.iter().zip(self.submissions).enumerate()
        {
            let layer_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Shadow_2D_Layer"),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: shadow_light.texture_layer_index,
                array_layer_count: Some(1),
                ..Default::default()
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Depth (2D)"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let dynamic_offset = uniform_idx as u32 * self.uniform_stride;
            pass.set_bind_group(0, bind_group, &[dynamic_offset]);

            submit_scene_submission(
                &mut pass,
                ctx.scene_submissions,
                submission,
                ctx.pipeline_cache,
            );
        }
    }
}

// ─── Cube Array Shadow Pass Node ──────────────────────────────────────────────

/// Renders depth layers for point light shadows into a cube array texture.
///
/// Each point light occupies 6 consecutive layers (one per cube face).
/// The texture is created as a 2D array with `depth_or_array_layers = 6 * N`,
/// and consumers view it as `TextureViewDimension::CubeArray` for seamless
/// hardware-filtered PCF sampling.
struct ShadowCubePassNode<'a> {
    lights: &'a [ShadowLightInstance],
    submissions: &'a [SceneSubmissionPlanId],
    uniform_base: u32,
    uniform_stride: u32,
    prepared: &'a ShadowPreparedState,
    texture_id: TextureNodeId,
}

impl<'a> PassNode<'a> for ShadowCubePassNode<'a> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.lights.is_empty() {
            return;
        }
        debug_assert_eq!(self.lights.len(), self.submissions.len());

        let texture = ctx.get_texture(self.texture_id);
        let bind_group_id = self
            .prepared
            .bind_group
            .get()
            .copied()
            .expect("Shadow VP bind group missing after prepare");
        let bind_group = ctx.resolve_prepared_bind_group(bind_group_id);

        for (uniform_index, (shadow_light, &submission)) in
            self.lights.iter().zip(self.submissions).enumerate()
        {
            // Each face is a single 2D layer within the cube array texture.
            let layer_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Shadow_Cube_Face"),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: shadow_light.texture_layer_index,
                array_layer_count: Some(1),
                ..Default::default()
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Depth (Cube)"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let uniform_index = self.uniform_base + uniform_index as u32;
            let dynamic_offset = uniform_index * self.uniform_stride;
            pass.set_bind_group(0, bind_group, &[dynamic_offset]);

            submit_scene_submission(
                &mut pass,
                ctx.scene_submissions,
                submission,
                ctx.pipeline_cache,
            );
        }
    }
}

fn pack_shadow_matrices(
    d2_lights: &[ShadowLightInstance],
    cube_lights: &[ShadowLightInstance],
    uniform_stride: u32,
) -> Vec<u8> {
    let light_count = d2_lights.len() + cube_lights.len();
    let mut data = vec![0; uniform_stride as usize * light_count];
    for (index, light) in d2_lights.iter().chain(cube_lights).enumerate() {
        let offset = index * uniform_stride as usize;
        let matrix = bytemuck::bytes_of(&light.light_view_projection);
        data[offset..offset + matrix.len()].copy_from_slice(matrix);
    }
    data
}

#[inline]
fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::view::RenderView;
    use crate::graph::core::graph::FrameConfig;
    use crate::graph::core::{GraphStorage, RenderGraph};
    use crate::graph::extracted::ExtractedLight;
    use myth_scene::camera::Frustum;
    use myth_scene::light::PointLight;

    const TEST_STRIDE: u32 = 256;

    struct TestNode;

    impl PassNode<'_> for TestNode {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    #[derive(Clone, Copy)]
    enum RootDemand {
        None,
        D2,
        Cube,
        Both,
    }

    struct ResourceFact {
        name: &'static str,
        producer: Option<&'static str>,
        consumers: Vec<&'static str>,
        texture_desc: Option<TextureDesc>,
        buffer_desc: Option<BufferDesc>,
    }

    struct GraphFacts {
        live_passes: Vec<&'static str>,
        registered_passes: Vec<&'static str>,
        resources: Vec<ResourceFact>,
        has_d2_output: bool,
        has_cube_output: bool,
        layout_initialized: bool,
    }

    impl GraphFacts {
        fn resource(&self, name: &str) -> &ResourceFact {
            self.resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap_or_else(|| panic!("resource '{name}' missing"))
        }
    }

    fn frame_config() -> FrameConfig {
        FrameConfig {
            width: 320,
            height: 180,
            depth_format: wgpu::TextureFormat::Depth32Float,
            msaa_samples: 1,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            hdr_format: crate::HDR_TEXTURE_FORMAT,
        }
    }

    fn test_feature() -> ShadowFeature {
        ShadowFeature {
            uniform_stride: TEST_STRIDE,
            bind_group_layout: None,
        }
    }

    fn shadow_light(
        light_id: u64,
        view_layer_index: u32,
        texture_layer_index: u32,
        is_point: bool,
        matrix_seed: f32,
    ) -> ShadowLightInstance {
        ShadowLightInstance {
            light_id,
            view_layer_index,
            texture_layer_index,
            light_buffer_index: light_id as usize,
            light_view_projection: Mat4::from_cols_array(&[
                matrix_seed,
                2.0,
                3.0,
                4.0,
                5.0,
                matrix_seed + 1.0,
                7.0,
                8.0,
                9.0,
                10.0,
                matrix_seed + 2.0,
                12.0,
                13.0,
                14.0,
                15.0,
                matrix_seed + 3.0,
            ]),
            is_point,
        }
    }

    fn build_graph(d2_count: usize, cube_count: usize, demand: RootDemand) -> GraphFacts {
        let mut feature = test_feature();
        let (live_passes, registered_passes, resources, has_d2_output, has_cube_output) = {
            let mut storage = GraphStorage::new();
            let arena = FrameArena::new();
            let d2_lights: Vec<_> = (0..d2_count)
                .map(|index| shadow_light(10, index as u32, index as u32, false, index as f32))
                .collect();
            let cube_lights: Vec<_> = (0..cube_count)
                .map(|index| {
                    shadow_light(
                        20,
                        40 + index as u32,
                        index as u32,
                        true,
                        100.0 + index as f32,
                    )
                })
                .collect();
            let snapshot = ShadowSnapshot {
                d2_lights: arena.alloc_slice_copy(&d2_lights),
                cube_lights: arena.alloc_slice_copy(&cube_lights),
                shadow_2d_resolution: 1024,
                shadow_cube_resolution: 512,
            };
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let config = frame_config();
            let outputs = {
                let mut ctx = GraphBuilderContext::new(&mut graph, &config, 0);
                feature.add_to_graph(&mut ctx, snapshot)
            };

            if !matches!(demand, RootDemand::None) {
                graph.add_pass("Shadow_Test_Root", |builder| {
                    if matches!(demand, RootDemand::D2 | RootDemand::Both) {
                        builder.read_texture(outputs.shadow_2d.expect("2D output missing"));
                    }
                    if matches!(demand, RootDemand::Cube | RootDemand::Both) {
                        builder.read_texture(outputs.shadow_cube.expect("cube output missing"));
                    }
                    builder.mark_side_effect();
                    (TestNode, ())
                });
            }

            graph.compile_topology();

            let live_passes = graph
                .storage
                .execution_queue
                .iter()
                .map(|&index| graph.storage.passes[index].name)
                .collect();
            let registered_passes = graph.storage.passes.iter().map(|pass| pass.name).collect();
            let resources = graph
                .storage
                .resources
                .iter()
                .enumerate()
                .map(|(resource_index, resource)| ResourceFact {
                    name: resource.name,
                    producer: resource
                        .producer
                        .map(|index| graph.storage.passes[index].name),
                    consumers: graph
                        .storage
                        .resource_consumers(resource_index)
                        .map(|index| graph.storage.passes[index].name)
                        .collect(),
                    texture_desc: resource.kind.texture_desc(),
                    buffer_desc: resource.kind.buffer_desc(),
                })
                .collect();
            (
                live_passes,
                registered_passes,
                resources,
                outputs.shadow_2d.is_some(),
                outputs.shadow_cube.is_some(),
            )
        };

        GraphFacts {
            live_passes,
            registered_passes,
            resources,
            has_d2_output,
            has_cube_output,
            layout_initialized: feature.bind_group_layout.is_some(),
        }
    }

    fn assert_shadow_texture_desc(facts: &GraphFacts, name: &str, resolution: u32, layers: u32) {
        let desc = facts
            .resource(name)
            .texture_desc
            .expect("shadow output must be a texture");
        assert_eq!(desc.size.width, resolution);
        assert_eq!(desc.size.height, resolution);
        assert_eq!(desc.size.depth_or_array_layers, layers);
        assert_eq!(desc.mip_level_count, 1);
        assert_eq!(desc.sample_count, 1);
        assert_eq!(desc.dimension, wgpu::TextureDimension::D2);
        assert_eq!(desc.format, wgpu::TextureFormat::Depth32Float);
        assert_eq!(
            desc.usage,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
        );
    }

    #[test]
    fn dead_shadow_graph_does_not_initialize_or_upload() {
        let facts = build_graph(2, 6, RootDemand::None);

        assert!(facts.live_passes.is_empty());
        assert!(!facts.layout_initialized);
        assert!(facts.has_d2_output);
        assert!(facts.has_cube_output);
        assert_eq!(
            facts
                .registered_passes
                .iter()
                .filter(|&&name| name == "Shadow_VP_Upload")
                .count(),
            1
        );
    }

    #[test]
    fn d2_only_graph_has_one_upload_and_correct_descriptor() {
        let facts = build_graph(2, 0, RootDemand::D2);

        assert_eq!(
            facts.live_passes,
            ["Shadow_VP_Upload", "Shadow_2D_Pass", "Shadow_Test_Root"]
        );
        assert!(facts.has_d2_output);
        assert!(!facts.has_cube_output);
        assert!(!facts.registered_passes.contains(&"Shadow_Cube_Pass"));
        assert_shadow_texture_desc(&facts, "Shadow_2D_Array", 1024, 2);

        let buffer = facts.resource("Shadow_VP_Buffer");
        assert_eq!(buffer.producer, Some("Shadow_VP_Upload"));
        assert_eq!(buffer.consumers, ["Shadow_2D_Pass"]);
        assert_eq!(
            buffer.buffer_desc,
            Some(BufferDesc::new(
                2 * u64::from(TEST_STRIDE),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ))
        );
    }

    #[test]
    fn cube_only_graph_has_one_upload_and_correct_descriptor() {
        let facts = build_graph(0, 6, RootDemand::Cube);

        assert_eq!(
            facts.live_passes,
            ["Shadow_VP_Upload", "Shadow_Cube_Pass", "Shadow_Test_Root"]
        );
        assert!(!facts.has_d2_output);
        assert!(facts.has_cube_output);
        assert!(!facts.registered_passes.contains(&"Shadow_2D_Pass"));
        assert_shadow_texture_desc(&facts, "Shadow_Cube_Array", 512, 6);

        let buffer = facts.resource("Shadow_VP_Buffer");
        assert_eq!(buffer.producer, Some("Shadow_VP_Upload"));
        assert_eq!(buffer.consumers, ["Shadow_Cube_Pass"]);
        assert_eq!(
            buffer
                .buffer_desc
                .expect("VP buffer must have a buffer descriptor")
                .logical_size,
            6 * u64::from(TEST_STRIDE)
        );
    }

    #[test]
    fn d2_and_cube_share_exactly_one_vp_upload() {
        let facts = build_graph(2, 6, RootDemand::Both);

        assert!(facts.live_passes.contains(&"Shadow_2D_Pass"));
        assert!(facts.live_passes.contains(&"Shadow_Cube_Pass"));
        assert_eq!(
            facts
                .live_passes
                .iter()
                .filter(|&&name| name == "Shadow_VP_Upload")
                .count(),
            1
        );
        assert_shadow_texture_desc(&facts, "Shadow_2D_Array", 1024, 2);
        assert_shadow_texture_desc(&facts, "Shadow_Cube_Array", 512, 6);

        let buffer = facts.resource("Shadow_VP_Buffer");
        assert_eq!(buffer.producer, Some("Shadow_VP_Upload"));
        assert_eq!(buffer.consumers, ["Shadow_2D_Pass", "Shadow_Cube_Pass"]);
        assert_eq!(
            buffer
                .buffer_desc
                .expect("VP buffer must have a buffer descriptor")
                .logical_size,
            8 * u64::from(TEST_STRIDE)
        );
    }

    #[test]
    fn dpe_keeps_only_the_demanded_shadow_branch() {
        let d2 = build_graph(2, 6, RootDemand::D2);
        assert!(d2.live_passes.contains(&"Shadow_VP_Upload"));
        assert!(d2.live_passes.contains(&"Shadow_2D_Pass"));
        assert!(!d2.live_passes.contains(&"Shadow_Cube_Pass"));

        let cube = build_graph(2, 6, RootDemand::Cube);
        assert!(cube.live_passes.contains(&"Shadow_VP_Upload"));
        assert!(!cube.live_passes.contains(&"Shadow_2D_Pass"));
        assert!(cube.live_passes.contains(&"Shadow_Cube_Pass"));
    }

    #[test]
    fn extract_preserves_view_indices_and_assigns_per_texture_indices() {
        let feature = test_feature();
        let mut scene = ExtractedScene::new();
        scene.lights.push(ExtractedLight {
            id: 20,
            color: glam::Vec3::ONE,
            intensity: 1.0,
            flags: 0,
            cast_shadows: true,
            kind: LightKind::Point(PointLight { range: 10.0 }),
            position: glam::Vec3::ZERO,
            direction: glam::Vec3::NEG_Z,
            shadow: None,
        });
        let mut render_lists = RenderLists::new();
        let mut push_view = |light_id, view_layer, resolution| {
            let matrix = Mat4::IDENTITY;
            let light_buffer_index = if light_id == 20 { 0 } else { usize::MAX };
            render_lists.active_views.push(RenderView::new_shadow(
                light_id,
                view_layer,
                light_buffer_index,
                format!("Shadow_{light_id}_{view_layer}"),
                matrix,
                Frustum::from_matrix_standard_z(matrix),
                (resolution, resolution),
                None,
            ));
        };
        push_view(10, 9, 128);
        push_view(20, 40, 64);
        push_view(30, 2, 256);
        for face in 1..6 {
            push_view(20, 40 + face, 64);
        }
        let arena = FrameArena::new();

        let snapshot = feature.extract(&render_lists, &scene, &arena);

        assert_eq!(snapshot.shadow_2d_resolution, 256);
        assert_eq!(snapshot.shadow_cube_resolution, 64);
        assert_eq!(snapshot.d2_lights.len(), 2);
        assert_eq!(snapshot.cube_lights.len(), 6);
        assert_eq!(snapshot.d2_lights[0].view_layer_index, 9);
        assert_eq!(snapshot.d2_lights[1].view_layer_index, 2);
        assert_eq!(snapshot.d2_lights[0].texture_layer_index, 0);
        assert_eq!(snapshot.d2_lights[1].texture_layer_index, 1);
        for (index, light) in snapshot.cube_lights.iter().enumerate() {
            assert_eq!(light.view_layer_index, 40 + index as u32);
            assert_eq!(light.texture_layer_index, index as u32);
            assert!(light.is_point);
        }
    }

    #[test]
    fn matrix_packing_uses_aligned_slots_and_zeroes_padding() {
        let d2 = shadow_light(10, 0, 0, false, 11.0);
        let cube = shadow_light(20, 0, 0, true, 101.0);

        let packed = pack_shadow_matrices(&[d2], &[cube], TEST_STRIDE);
        let matrix_size = std::mem::size_of::<Mat4>();

        assert_eq!(packed.len(), 2 * TEST_STRIDE as usize);
        assert_eq!(
            &packed[..matrix_size],
            bytemuck::bytes_of(&d2.light_view_projection)
        );
        assert!(
            packed[matrix_size..TEST_STRIDE as usize]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert_eq!(
            &packed[TEST_STRIDE as usize..TEST_STRIDE as usize + matrix_size],
            bytemuck::bytes_of(&cube.light_view_projection)
        );
        assert!(
            packed[TEST_STRIDE as usize + matrix_size..]
                .iter()
                .all(|&byte| byte == 0)
        );
    }

    #[test]
    fn uniform_alignment_rounds_up_to_the_next_slot() {
        assert_eq!(align_to(64, 1), 64);
        assert_eq!(align_to(64, 256), 256);
        assert_eq!(align_to(256, 256), 256);
        assert_eq!(align_to(257, 256), 512);
    }
}
