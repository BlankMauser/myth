//! TAA Feature + logical cross-frame history.
//!
//! Graph construction captures a small [`TaaSnapshot`] and declares two-bank
//! color/depth history through [`HistoryRegistry`]. Physical history textures,
//! the pipeline, the parameter upload, and the bind group are all materialized
//! only after the resolve node survives DPE. The next bank is published only
//! after successful queue submission.

use rustc_hash::FxHashMap;

use crate::HDR_TEXTURE_FORMAT;
use crate::core::gpu::binding_layout::{depth_texture_2d, sampler, texture_2d, uniform_buffer};
use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::persistent::TEMPORAL_DOMAIN_TTL_FRAMES;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, HistoryPlan, HistoryRegistry, HistorySetId,
    HistoryTextureRef, PassNode, PrepareContext, RenderTargetOps, TemporalDomainKey, TextureDesc,
    TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_scene::camera::RenderCamera;

#[derive(Clone, Copy)]
pub struct TaaSnapshot {
    params: [f32; 4],
    history: HistoryPlan<2>,
    extent: wgpu::Extent3d,
}

impl TaaSnapshot {
    /// Previous-frame color history for temporal consumers that execute before
    /// TAA in the current frame (currently SSGI).
    #[must_use]
    pub fn previous_color(self) -> Option<HistoryTextureRef> {
        self.history.valid.then_some(self.history.previous[0])
    }
}

#[derive(Default)]
struct TaaPipelineResources {
    pipeline_id: Option<RenderPipelineId>,
    layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

#[derive(Debug, Clone, Copy)]
struct TaaHistoryDomain {
    set_id: HistorySetId,
    last_planned_frame: u64,
}

/// Long-lived TAA feature. Static pipeline state remains Feature-owned while
/// dynamic cross-frame textures live in the renderer's HistoryRegistry.
pub struct TaaFeature {
    pipeline: TaaPipelineResources,
    histories: FxHashMap<TemporalDomainKey, TaaHistoryDomain>,
}

impl Default for TaaFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl TaaFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pipeline: TaaPipelineResources::default(),
            histories: FxHashMap::default(),
        }
    }

    /// Capture POD parameters and plan logical history banks. No WGPU object is
    /// created and no GPU upload occurs here.
    #[allow(clippy::too_many_arguments)]
    pub fn extract(
        &mut self,
        histories: &mut HistoryRegistry,
        scene_id: u32,
        camera: &RenderCamera,
        feedback_weight: f32,
        temporal_serial: u64,
        renderer_frame: u64,
        size: (u32, u32),
        depth_format: wgpu::TextureFormat,
    ) -> TaaSnapshot {
        let key = TemporalDomainKey::new(scene_id, camera.camera_id);
        if self.histories.get(&key).is_some_and(|domain| {
            renderer_frame.saturating_sub(domain.last_planned_frame) > TEMPORAL_DOMAIN_TTL_FRAMES
        }) {
            let expired = self
                .histories
                .remove(&key)
                .expect("expired TAA history domain must exist");
            let removed = histories.remove_set(expired.set_id);
            debug_assert!(removed, "TAA history domain lost its registry set");
        }
        let set_id = if let Some(domain) = self.histories.get_mut(&key) {
            domain.last_planned_frame = renderer_frame;
            domain.set_id
        } else {
            let id = histories.create_set();
            self.histories.insert(
                key,
                TaaHistoryDomain {
                    set_id: id,
                    last_planned_frame: renderer_frame,
                },
            );
            id
        };

        let extent = wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        };
        let color_desc = TextureDesc::new_2d(
            extent.width,
            extent.height,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let depth_desc = TextureDesc::new_2d(
            extent.width,
            extent.height,
            depth_format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let camera_cut = camera.camera_cut != 0;
        let history = histories.plan_set(
            set_id,
            [color_desc, depth_desc],
            temporal_serial,
            camera_cut,
        );
        let reject_history = camera_cut || !history.valid;

        TaaSnapshot {
            params: [feedback_weight, camera.near, f32::from(reject_history), 0.0],
            history,
            extent,
        }
    }

    /// Retire history for scene/camera domains that have not planned a TAA
    /// frame within `ttl` renderer frames.
    pub(crate) fn garbage_collect(
        &mut self,
        histories: &mut HistoryRegistry,
        current_frame: u64,
        ttl: u64,
    ) -> usize {
        let before = self.histories.len();
        self.histories.retain(|_, domain| {
            let live = current_frame.saturating_sub(domain.last_planned_frame) <= ttl;
            if !live {
                let removed = histories.remove_set(domain.set_id);
                debug_assert!(removed, "TAA history domain lost its registry set");
            }
            live
        });
        before - self.histories.len()
    }

    /// Resolve into the next persistent color bank, then copy its clean result
    /// to an isolated transient output before any downstream work. The same
    /// node archives scene depth, so later in-place transparent/composite
    /// passes cannot pollute history and color/depth publish atomically.
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        snapshot: TaaSnapshot,
        active_color: TextureNodeId,
        velocity: TextureNodeId,
        scene_depth: TextureNodeId,
    ) -> TextureNodeId {
        let history_color = ctx
            .graph
            .import_persistent_texture("TAA_History_Color_Previous", snapshot.history.previous[0]);
        let history_depth = ctx
            .graph
            .import_persistent_texture("TAA_History_Depth_Previous", snapshot.history.previous[1]);
        let next_color = ctx
            .graph
            .import_persistent_texture("TAA_History_Color_Next", snapshot.history.next[0]);
        let next_depth = ctx
            .graph
            .import_persistent_texture("TAA_History_Depth_Next", snapshot.history.next[1]);

        let pipeline = &mut self.pipeline;
        let resolved_color = ctx.graph.add_pass("TAA_Resolve", |builder| {
            builder.read_texture(active_color);
            builder.read_texture(history_color);
            builder.read_texture(velocity);
            builder.read_texture(scene_depth);
            builder.read_texture(history_depth);
            builder.write_texture(next_color);
            builder.write_texture(next_depth);
            let resolved_color = builder.create_texture(
                "TAA_Resolved",
                TextureDesc::new_2d(
                    snapshot.extent.width,
                    snapshot.extent.height,
                    HDR_TEXTURE_FORMAT,
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                ),
            );
            let params_buffer = builder.create_buffer(
                "TAA_Params",
                BufferDesc::new(
                    16,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
            );

            let node = TaaPassNode {
                resources: pipeline,
                params: snapshot.params,
                current_color: active_color,
                history_color,
                velocity,
                scene_depth,
                history_depth,
                resolved_color,
                output_color_history: next_color,
                output_depth: next_depth,
                params_buffer,
                extent: snapshot.extent,
                pipeline_id: None,
                bind_group: None,
            };
            (node, resolved_color)
        });
        ctx.graph
            .register_history_commit(snapshot.history.commit, &[next_color, next_depth]);
        resolved_color
    }
}

impl TaaPipelineResources {
    fn ensure_resources(&mut self, ctx: &mut PrepareContext<'_>) {
        if self.pipeline_id.is_some() && self.layout.is_some() {
            return;
        }

        let (shader, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/post_process/taa_resolve"),
            &ShaderCompilationOptions::default(),
        );
        let target = ColorTargetKey::from(wgpu::ColorTargetState {
            format: HDR_TEXTURE_FORMAT,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let key = FullscreenPipelineKey::fullscreen(shader_hash, smallvec::smallvec![target], None);

        if let Some(id) = ctx.pipeline_cache.get_fullscreen_pipeline_id(&key) {
            self.pipeline_id = Some(id);
            self.layout = Some(ctx.pipeline_cache.get_tracked_layout(id, 0).clone());
            return;
        }

        let layout = create_layout(ctx.device);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("TAA Pipeline Layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let id = ctx.pipeline_cache.get_or_create_fullscreen(
            ctx.device,
            shader,
            &pipeline_layout,
            &key,
            "TAA Resolve Pipeline",
        );
        ctx.pipeline_cache
            .register_render_layouts(id, smallvec::smallvec![layout.clone()]);
        self.pipeline_id = Some(id);
        self.layout = Some(layout);
    }

    fn pipeline_id(&self) -> RenderPipelineId {
        self.pipeline_id
            .expect("TAA pipeline must be prepared by the live resolve node")
    }

    fn layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        self.layout
            .as_ref()
            .expect("TAA layout must be prepared by the live resolve node")
    }
}

fn create_layout(device: &wgpu::Device) -> Tracked<wgpu::BindGroupLayout> {
    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA BindGroup Layout"),
            entries: &[
                texture_2d(
                    0,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                texture_2d(
                    2,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::TextureSampleType::Float { filterable: false },
                ),
                depth_texture_2d(3, wgpu::ShaderStages::FRAGMENT),
                depth_texture_2d(4, wgpu::ShaderStages::FRAGMENT),
                sampler(
                    5,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                ),
                sampler(
                    6,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::NonFiltering,
                ),
                uniform_buffer(7, wgpu::ShaderStages::FRAGMENT),
            ],
        }),
    )
}

struct TaaPassNode<'a> {
    resources: &'a mut TaaPipelineResources,
    params: [f32; 4],
    current_color: TextureNodeId,
    history_color: TextureNodeId,
    velocity: TextureNodeId,
    scene_depth: TextureNodeId,
    history_depth: TextureNodeId,
    resolved_color: TextureNodeId,
    output_color_history: TextureNodeId,
    output_depth: TextureNodeId,
    params_buffer: BufferNodeId,
    extent: wgpu::Extent3d,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for TaaPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.resources.ensure_resources(ctx);
        self.pipeline_id = Some(self.resources.pipeline_id());
        ctx.queue.write_buffer(
            ctx.views.get_buffer(self.params_buffer),
            0,
            bytemuck::cast_slice(&self.params),
        );
        self.bind_group = Some(
            crate::myth_bind_group!(ctx, self.resources.layout(), Some("TAA BindGroup"), [
                0 => self.current_color,
                1 => self.history_color,
                2 => self.velocity,
                3 => self.scene_depth,
                4 => self.history_depth,
                5 => CommonSampler::LinearClamp,
                6 => CommonSampler::NearestClamp,
                7 => self.params_buffer,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let target =
            ctx.get_color_attachment(self.output_color_history, RenderTargetOps::DontCare, None);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA Resolve Pass"),
                color_attachments: &[target],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(
                ctx.pipeline_cache
                    .get_render_pipeline(self.pipeline_id.expect("TAA pipeline not prepared")),
            );
            pass.set_bind_group(
                0,
                self.bind_group.expect("TAA bind group not prepared"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: ctx.get_texture(self.output_color_history),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: ctx.get_texture(self.resolved_color),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            self.extent,
        );
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: ctx.get_texture(self.scene_depth),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: ctx.get_texture(self.output_depth),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            self.extent,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use myth_scene::camera::Camera;

    fn render_camera(camera_id: u128) -> RenderCamera {
        let mut camera = Camera::new_perspective(45.0, 1.0, 0.1);
        let mut render_camera = camera.extract_render_camera();
        render_camera.camera_id = camera_id;
        render_camera.camera_cut = 0;
        render_camera
    }

    #[test]
    fn submitted_history_becomes_available_on_the_next_frame() {
        let mut registry = HistoryRegistry::new();
        let mut feature = TaaFeature::new();
        let camera = render_camera(7);

        let first = feature.extract(
            &mut registry,
            1,
            &camera,
            0.9,
            1,
            1,
            (128, 64),
            wgpu::TextureFormat::Depth32Float,
        );
        assert!(first.previous_color().is_none());
        assert_eq!(first.params[2], 1.0);

        assert!(registry.commit(
            first.history.commit,
            &first.history.next.map(|texture| texture.id),
        ));
        let second = feature.extract(
            &mut registry,
            1,
            &camera,
            0.9,
            2,
            2,
            (128, 64),
            wgpu::TextureFormat::Depth32Float,
        );
        assert_eq!(second.previous_color(), Some(first.history.next[0]));
        assert_eq!(second.params[2], 0.0);
    }

    #[test]
    fn histories_are_partitioned_by_scene_and_camera() {
        let mut registry = HistoryRegistry::new();
        let mut feature = TaaFeature::new();
        let camera_a = render_camera(11);
        let camera_b = render_camera(12);

        let scene_a_camera_a = feature.extract(
            &mut registry,
            1,
            &camera_a,
            0.9,
            1,
            1,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
        );
        let scene_a_camera_b = feature.extract(
            &mut registry,
            1,
            &camera_b,
            0.9,
            1,
            1,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
        );
        let scene_b_camera_a = feature.extract(
            &mut registry,
            2,
            &camera_a,
            0.9,
            1,
            1,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
        );

        assert_ne!(
            scene_a_camera_a.history.next[0].id,
            scene_a_camera_b.history.next[0].id
        );
        assert_ne!(
            scene_a_camera_a.history.next[0].id,
            scene_b_camera_a.history.next[0].id
        );
    }

    #[test]
    fn alternating_camera_domains_keep_independent_continuity() {
        let mut registry = HistoryRegistry::new();
        let mut feature = TaaFeature::new();
        let camera_a = render_camera(21);
        let camera_b = render_camera(22);

        let a1 = feature.extract(
            &mut registry,
            1,
            &camera_a,
            0.9,
            1,
            1,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
        );
        assert!(registry.commit(
            a1.history.commit,
            &a1.history.next.map(|texture| texture.id),
        ));
        let b1 = feature.extract(
            &mut registry,
            1,
            &camera_b,
            0.9,
            1,
            1,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
        );
        assert!(registry.commit(
            b1.history.commit,
            &b1.history.next.map(|texture| texture.id),
        ));

        let a2 = feature.extract(
            &mut registry,
            1,
            &camera_a,
            0.9,
            2,
            2,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
        );
        let b2 = feature.extract(
            &mut registry,
            1,
            &camera_b,
            0.9,
            2,
            2,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
        );

        assert!(a2.history.valid);
        assert!(b2.history.valid);
        assert_eq!(a2.previous_color(), Some(a1.history.next[0]));
        assert_eq!(b2.previous_color(), Some(b1.history.next[0]));
    }

    #[test]
    fn abandoned_domain_expires_after_ttl_and_reenters_with_invalid_history() {
        let mut registry = HistoryRegistry::new();
        let mut feature = TaaFeature::new();
        let camera = render_camera(31);

        let first = feature.extract(
            &mut registry,
            1,
            &camera,
            0.9,
            1,
            10,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
        );
        assert!(registry.commit(
            first.history.commit,
            &first.history.next.map(|texture| texture.id),
        ));

        assert_eq!(feature.garbage_collect(&mut registry, 12, 2), 0);
        assert_eq!(feature.histories.len(), 1);
        assert_eq!(feature.garbage_collect(&mut registry, 13, 2), 1);
        assert!(feature.histories.is_empty());

        let replacement = feature.extract(
            &mut registry,
            1,
            &camera,
            0.9,
            2,
            13,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
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
        let mut registry = HistoryRegistry::new();
        let mut feature = TaaFeature::new();
        let camera = render_camera(32);

        let first = feature.extract(
            &mut registry,
            1,
            &camera,
            0.9,
            1,
            1,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
        );
        assert!(registry.commit(
            first.history.commit,
            &first.history.next.map(|texture| texture.id),
        ));

        let replacement = feature.extract(
            &mut registry,
            1,
            &camera,
            0.9,
            2,
            TEMPORAL_DOMAIN_TTL_FRAMES + 2,
            (64, 64),
            wgpu::TextureFormat::Depth32Float,
        );
        assert!(!replacement.history.valid);
        assert_ne!(
            replacement.history.commit.set_id,
            first.history.commit.set_id
        );
    }
}
