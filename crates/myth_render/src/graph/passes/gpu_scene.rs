//! Batched GPU Scene persistent-table patching.
//!
//! One node owns every instance-table write for a graph attempt. It records
//! staging copies into the graph's real command encoder and acknowledges the
//! persistent commit group only after all copies were encoded. Consequently a
//! dead branch, failed target acquisition, abandoned encoder, or missing queue
//! submit can never publish dirty CPU state.

mod cull;
pub(crate) use cull::{
    GPU_SCENE_CULL_WORKGROUP_SIZE, GPU_SCENE_INDEXED_ARGUMENT_SIZE,
    GPU_SCENE_VISIBLE_INSTANCE_SIZE, GpuSceneCullFeature, GpuSceneCullViewDescriptor,
    GpuSceneMultiBinCullDescriptor, GpuSceneMultiBinCullOutput, GpuSceneMultiBinDrawOutput,
};

use crate::graph::core::graph::{PersistentBufferCommitGroupId, PersistentBufferWriteCoverage};
use crate::graph::core::{BufferNodeId, ExecuteContext, PassNode, PrepareContext, RenderGraph};
#[cfg(test)]
use crate::graph::gpu_scene::GpuSceneInstanceRecord;
use crate::graph::gpu_scene::{
    GpuSceneActiveBatchSet, GpuSceneDirtyRange, GpuSceneMultiBinCandidate, GpuSceneTableSetPlan,
    GpuSceneTableSetPublicationProbe,
};
#[cfg(test)]
use crate::graph::gpu_scene::{GpuScenePublicationProbe, GpuSceneTablePlan};

/// Logical instance table consumed by the later Cull node.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuScenePatchOutput {
    pub(crate) instances: BufferNodeId,
    pub(crate) slot_span: u32,
    /// Copy-only exact target retained outside the graph for post-submit
    /// acknowledgement after the graph releases its borrowed CPU slices.
    pub(crate) publication_probe: GpuScenePublicationProbe,
}

/// Atomic logical image of the complete M7.4 GPU Scene authority.
///
/// Geometry and Material are deliberately imported and patched by the same
/// owner even though the portable Cull shader currently reads only Instance
/// and Batch. This keeps publication coherent while exposing only the nodes
/// needed by today's topology.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuSceneTableSetPatchOutput<'a> {
    pub(crate) instances: BufferNodeId,
    pub(crate) batches: BufferNodeId,
    pub(crate) instance_slot_span: u32,
    pub(crate) batch_slot_span: u32,
    pub(crate) candidates: &'a [GpuSceneMultiBinCandidate],
    pub(crate) active_batches: GpuSceneActiveBatchSet<'a>,
}

// Keep the graph-topology handle compact. Exact publication identity is a
// separate post-submit probe and must never bloat every main/prepass/shadow
// topology copy again.
const _: () = assert!(std::mem::size_of::<GpuSceneTableSetPatchOutput<'static>>() <= 64);

/// Add the persistent table import and, only when the exact revision is not
/// already published, its sole writer node.
#[cfg(test)]
pub(crate) fn add_gpu_scene_patch<'a>(
    graph: &mut RenderGraph<'a>,
    plan: GpuSceneTablePlan<'a>,
) -> GpuScenePatchOutput {
    let publication_probe = plan.publication_probe;
    let slot_span = plan.slot_span;
    let [instances] = add_gpu_scene_patch_family(
        graph,
        [GpuScenePatchMemberPlan {
            name: "GpuScene_Instances",
            persistent: plan.persistent,
            image: GpuScenePatchImage::from_records(
                plan.persistent.buffer.spec.desc.logical_size,
                plan.records,
                plan.dirty_ranges,
            ),
        }],
    );

    GpuScenePatchOutput {
        instances,
        slot_span,
        publication_probe,
    }
}

/// Import and, when needed, patch all four persistent GPU Scene tables in one
/// commit group. No subset can be published independently.
pub(crate) fn add_gpu_scene_table_set_patch<'a>(
    graph: &mut RenderGraph<'a>,
    plan: &GpuSceneTableSetPlan<'a>,
) -> (
    GpuSceneTableSetPatchOutput<'a>,
    GpuSceneTableSetPublicationProbe,
) {
    let publication_probe = plan.publication_probe;
    let instance_slot_span = plan.instances.slot_span;
    let batch_slot_span = plan.batches.slot_span;
    let candidates = plan.candidates;
    let active_batches = plan.active_batches;
    let [instances, _geometries, _materials, batches] = add_gpu_scene_patch_family(
        graph,
        [
            GpuScenePatchMemberPlan {
                name: "GpuScene_Instances",
                persistent: plan.instances.persistent,
                image: GpuScenePatchImage::from_records(
                    plan.instances.persistent.buffer.spec.desc.logical_size,
                    plan.instances.records,
                    plan.instances.dirty_ranges,
                ),
            },
            GpuScenePatchMemberPlan {
                name: "GpuScene_Geometries",
                persistent: plan.geometries.persistent,
                image: GpuScenePatchImage::from_records(
                    plan.geometries.persistent.buffer.spec.desc.logical_size,
                    plan.geometries.records,
                    plan.geometries.dirty_ranges,
                ),
            },
            GpuScenePatchMemberPlan {
                name: "GpuScene_Materials",
                persistent: plan.materials.persistent,
                image: GpuScenePatchImage::from_records(
                    plan.materials.persistent.buffer.spec.desc.logical_size,
                    plan.materials.records,
                    plan.materials.dirty_ranges,
                ),
            },
            GpuScenePatchMemberPlan {
                name: "GpuScene_Batches",
                persistent: plan.batches.persistent,
                image: GpuScenePatchImage::from_records(
                    plan.batches.persistent.buffer.spec.desc.logical_size,
                    plan.batches.records,
                    plan.batches.dirty_ranges,
                ),
            },
        ],
    );

    (
        GpuSceneTableSetPatchOutput {
            instances,
            batches,
            instance_slot_span,
            batch_slot_span,
            candidates,
            active_batches,
        },
        publication_probe,
    )
}

#[derive(Clone, Copy)]
struct GpuScenePatchMemberPlan<'a> {
    name: &'static str,
    persistent: crate::graph::core::PersistentBufferPlan,
    image: GpuScenePatchImage<'a>,
}

#[derive(Clone, Copy)]
struct GpuScenePatchMember<'a> {
    node: BufferNodeId,
    persistent_id: crate::graph::core::PersistentBufferId,
    image: GpuScenePatchImage<'a>,
}

struct GpuScenePatchNode<'a, const N: usize> {
    members: [GpuScenePatchMember<'a>; N],
    commit_group: PersistentBufferCommitGroupId,
    prepared: Option<GpuScenePreparedPatch>,
}

fn add_gpu_scene_patch_family<'a, const N: usize>(
    graph: &mut RenderGraph<'a>,
    plans: [GpuScenePatchMemberPlan<'a>; N],
) -> [BufferNodeId; N] {
    assert_ne!(N, 0, "GPU Scene Patch family cannot be empty");
    let nodes = std::array::from_fn(|index| {
        graph.import_persistent_buffer(plans[index].name, plans[index].persistent)
    });
    let valid = plans[0].persistent.valid;
    assert!(
        plans.iter().all(|plan| plan.persistent.valid == valid),
        "GPU Scene Patch family must be coherently all-valid or all-invalid"
    );
    if valid {
        debug_assert!(plans.iter().all(|plan| plan.persistent.commit.is_none()));
        return nodes;
    }

    let commit_members: [_; N] = std::array::from_fn(|index| {
        (
            plans[index]
                .persistent
                .commit
                .expect("invalid GPU Scene table plan requires a publication token"),
            nodes[index],
        )
    });
    let commit_group = graph.register_persistent_buffer_commit_group(&commit_members);
    let members: [GpuScenePatchMember<'a>; N] = std::array::from_fn(|index| GpuScenePatchMember {
        node: nodes[index],
        persistent_id: plans[index].persistent.buffer.id,
        image: plans[index].image,
    });
    graph.add_pass("GpuScene_Patch", |builder| {
        for &node in &nodes {
            builder.write_buffer(node);
        }
        (
            GpuScenePatchNode {
                members,
                commit_group,
                prepared: None,
            },
            (),
        )
    });
    nodes
}

#[derive(Debug, Clone, Copy)]
struct GpuScenePreparedPatch {
    coverage: PersistentBufferWriteCoverage,
    upload_size: u64,
}

/// Above this density, a full image adds at most one third more transfer bytes
/// than the delta while reducing the command stream to one copy per table.
const GPU_SCENE_DENSE_DELTA_NUMERATOR: u128 = 3;
const GPU_SCENE_DENSE_DELTA_DENOMINATOR: u128 = 4;
/// Hundreds of disjoint `copy_buffer_to_buffer` commands are already more
/// expensive to encode and validate than a moderately larger sequential DMA.
const GPU_SCENE_DELTA_COPY_RANGE_BUDGET: usize = 256;
/// Crossing the normal range budget upgrades only when at least one quarter
/// of the coherent image is dirty, avoiding a full multi-megabyte upload for a
/// few hundred isolated records in an otherwise huge table.
const GPU_SCENE_FRAGMENTED_DELTA_NUMERATOR: u128 = 1;
const GPU_SCENE_FRAGMENTED_DELTA_DENOMINATOR: u128 = 4;
/// Regardless of byte density, thousands of copy commands are an unacceptable
/// driver/encoder spike; a four-copy coherent replay is the bounded fallback.
const GPU_SCENE_DELTA_COPY_RANGE_HARD_LIMIT: usize = 4096;

#[inline]
fn ratio_at_least(part: u64, whole: u64, numerator: u128, denominator: u128) -> bool {
    debug_assert_ne!(whole, 0);
    u128::from(part) * denominator >= u128::from(whole) * numerator
}

#[inline]
fn gpu_scene_patch_prefers_full_image(
    logical_size: u64,
    dirty_size: u64,
    dirty_range_count: usize,
) -> bool {
    ratio_at_least(
        dirty_size,
        logical_size,
        GPU_SCENE_DENSE_DELTA_NUMERATOR,
        GPU_SCENE_DENSE_DELTA_DENOMINATOR,
    ) || (dirty_range_count > GPU_SCENE_DELTA_COPY_RANGE_BUDGET
        && ratio_at_least(
            dirty_size,
            logical_size,
            GPU_SCENE_FRAGMENTED_DELTA_NUMERATOR,
            GPU_SCENE_FRAGMENTED_DELTA_DENOMINATOR,
        ))
        || dirty_range_count > GPU_SCENE_DELTA_COPY_RANGE_HARD_LIMIT
}

/// Type-erased immutable image of one member in the fixed GPU Scene table
/// family. Record bytes remain borrowed from the CPU authority; the Patch
/// node owns only compact metadata and never clones a full table.
#[derive(Debug, Clone, Copy)]
struct GpuScenePatchImage<'a> {
    logical_size: u64,
    record_stride: u64,
    records: &'a [u8],
    dirty_ranges: &'a [GpuSceneDirtyRange],
}

impl<'a> GpuScenePatchImage<'a> {
    fn from_records<T: bytemuck::Pod>(
        logical_size: u64,
        records: &'a [T],
        dirty_ranges: &'a [GpuSceneDirtyRange],
    ) -> Self {
        let record_stride = std::mem::size_of::<T>() as u64;
        assert_ne!(record_stride, 0, "GPU Scene table record cannot be ZST");
        let bytes = bytemuck::cast_slice(records);
        assert_eq!(
            u64::try_from(bytes.len()).expect("GPU Scene table image exceeds u64"),
            logical_size,
            "GPU Scene full image does not cover its complete logical table"
        );
        assert_eq!(
            logical_size % wgpu::COPY_BUFFER_ALIGNMENT,
            0,
            "GPU Scene table size violates COPY_BUFFER_ALIGNMENT"
        );
        Self {
            logical_size,
            record_stride,
            records: bytes,
            dirty_ranges,
        }
    }

    #[inline]
    fn dirty_byte_offset(self, range: GpuSceneDirtyRange) -> u64 {
        u64::from(range.first_slot)
            .checked_mul(self.record_stride)
            .expect("GPU Scene dirty byte offset overflowed")
    }

    #[inline]
    fn dirty_byte_size(self, range: GpuSceneDirtyRange) -> u64 {
        u64::from(range.slot_count)
            .checked_mul(self.record_stride)
            .expect("GPU Scene dirty byte size overflowed")
    }
}

#[cfg(test)]
fn create_gpu_scene_patch_staging(
    device: &wgpu::Device,
    logical_size: u64,
    records: &[GpuSceneInstanceRecord],
    dirty_ranges: &[GpuSceneDirtyRange],
    full_initialization: bool,
) -> (wgpu::Buffer, GpuScenePreparedPatch) {
    create_gpu_scene_patch_staging_for_images(
        device,
        &[GpuScenePatchImage::from_records(
            logical_size,
            records,
            dirty_ranges,
        )],
        full_initialization,
    )
}

fn create_gpu_scene_patch_staging_for_images(
    device: &wgpu::Device,
    images: &[GpuScenePatchImage<'_>],
    full_initialization: bool,
) -> (wgpu::Buffer, GpuScenePreparedPatch) {
    assert!(!images.is_empty(), "GPU Scene Patch requires table images");
    let logical_size = images.iter().fold(0u64, |total, image| {
        total
            .checked_add(image.logical_size)
            .expect("GPU Scene coherent image size overflowed")
    });
    assert_ne!(
        logical_size, 0,
        "GPU Scene Patch requires a non-empty coherent image"
    );
    let mut dirty_range_count = 0usize;
    let dirty_size = images.iter().fold(0u64, |table_total, image| {
        dirty_range_count = dirty_range_count
            .checked_add(image.dirty_ranges.len())
            .expect("GPU Scene dirty range count overflowed");
        image
            .dirty_ranges
            .iter()
            .fold(table_total, |total, &range| {
                total
                    .checked_add(image.dirty_byte_size(range))
                    .expect("GPU Scene dirty upload size overflowed")
            })
    });
    let use_full_image = full_initialization
        || gpu_scene_patch_prefers_full_image(logical_size, dirty_size, dirty_range_count);
    let coverage = if use_full_image {
        PersistentBufferWriteCoverage::Full
    } else {
        PersistentBufferWriteCoverage::Delta
    };
    let upload_size = if use_full_image {
        logical_size
    } else {
        dirty_size
    };
    assert_ne!(
        upload_size, 0,
        "invalid GPU Scene revision has neither a full initialization nor dirty records"
    );
    assert_eq!(
        upload_size % wgpu::COPY_BUFFER_ALIGNMENT,
        0,
        "GPU Scene upload size violates COPY_BUFFER_ALIGNMENT"
    );

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GpuScene_Patch_Staging"),
        size: upload_size,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    {
        let mut mapped = staging.slice(..).get_mapped_range_mut();
        if use_full_image {
            let mut staging_offset = 0usize;
            for image in images {
                let staging_end = staging_offset
                    .checked_add(image.records.len())
                    .expect("GPU Scene full staging range overflowed");
                mapped
                    .slice(staging_offset..staging_end)
                    .copy_from_slice(image.records);
                staging_offset = staging_end;
            }
            debug_assert_eq!(staging_offset, mapped.len());
        } else {
            let mut staging_offset = 0usize;
            for image in images {
                for &range in image.dirty_ranges {
                    let source_start = usize::try_from(image.dirty_byte_offset(range))
                        .expect("GPU Scene dirty source offset exceeds usize");
                    let source_size = usize::try_from(image.dirty_byte_size(range))
                        .expect("GPU Scene dirty source size exceeds usize");
                    let source_end = source_start
                        .checked_add(source_size)
                        .expect("GPU Scene dirty source range overflowed");
                    let source = image.records.get(source_start..source_end).unwrap_or_else(|| {
                        panic!(
                            "GPU Scene dirty byte range {source_start}..{source_end} exceeds {} bytes",
                            image.records.len()
                        )
                    });
                    let staging_end = staging_offset
                        .checked_add(source.len())
                        .expect("GPU Scene staging range overflowed");
                    mapped
                        .slice(staging_offset..staging_end)
                        .copy_from_slice(source);
                    staging_offset = staging_end;
                }
            }
            debug_assert_eq!(staging_offset, mapped.len());
        }
    }
    staging.unmap();
    (
        staging,
        GpuScenePreparedPatch {
            coverage,
            upload_size,
        },
    )
}

impl<'a, const N: usize> PassNode<'a> for GpuScenePatchNode<'a, N> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        assert!(
            self.prepared.is_none(),
            "GPU Scene Patch was prepared more than once"
        );
        let full_initialization = self.members.iter().any(|member| {
            ctx.views
                .persistent_buffers
                .materialization(member.persistent_id)
                .unwrap_or_else(|| {
                    panic!(
                        "live GPU Scene table {:?} was not materialized for this graph attempt",
                        member.persistent_id
                    )
                })
                .requires_full_initialization
        });
        let images: [GpuScenePatchImage<'a>; N] =
            std::array::from_fn(|index| self.members[index].image);
        let (staging, prepared) =
            create_gpu_scene_patch_staging_for_images(ctx.device, &images, full_initialization);
        ctx.stage_persistent_buffer_upload(self.commit_group, staging, prepared.coverage);
        self.prepared = Some(prepared);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let prepared = self
            .prepared
            .expect("live GPU Scene Patch reached execute without preparation");
        let staging = ctx.resolve_persistent_buffer_upload(self.commit_group);
        if prepared.coverage == PersistentBufferWriteCoverage::Full {
            let mut staging_offset = 0u64;
            for member in &self.members {
                encoder.copy_buffer_to_buffer(
                    staging,
                    staging_offset,
                    ctx.get_buffer(member.node),
                    0,
                    member.image.logical_size,
                );
                staging_offset = staging_offset
                    .checked_add(member.image.logical_size)
                    .expect("GPU Scene full staging copy offset overflowed");
            }
            debug_assert_eq!(staging_offset, prepared.upload_size);
        } else {
            let mut staging_offset = 0u64;
            for member in &self.members {
                for &range in member.image.dirty_ranges {
                    let size = member.image.dirty_byte_size(range);
                    encoder.copy_buffer_to_buffer(
                        staging,
                        staging_offset,
                        ctx.get_buffer(member.node),
                        member.image.dirty_byte_offset(range),
                        size,
                    );
                    staging_offset = staging_offset
                        .checked_add(size)
                        .expect("GPU Scene staging copy offset overflowed");
                }
            }
            debug_assert_eq!(staging_offset, prepared.upload_size);
        }

        // The command buffer now owns the staging resource and every table
        // copy. Only this final statement opens the real-submit gate.
        ctx.mark_persistent_buffer_commit_group_encoded_with_coverage(
            self.commit_group,
            prepared.coverage,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WgpuContext;
    use crate::graph::core::{
        BufferDesc, FrameArena, GraphStorage, PersistentBufferCommitRequest, PersistentBufferId,
        PersistentBufferPlan, PersistentBufferPlanRequest, PersistentBufferRegistry,
        PersistentBufferSpec,
    };
    use crate::graph::gpu_scene::{
        GpuScene, GpuSceneBatchRecord, GpuSceneGeometryRecord, GpuSceneInstanceInput,
        GpuSceneMaterialRecord,
    };
    use crate::settings::{RendererInitConfig, RendererSettings};
    use bytemuck::Zeroable;
    use glam::{Mat4, Vec3};
    use myth_resources::BoundingBox;
    use myth_scene::NodeHandle;
    use slotmap::KeyData;

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

    fn read_staging(gpu: &WgpuContext, staging: &wgpu::Buffer) -> Vec<u8> {
        let size = staging.size();
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Scene Patch staging readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GPU Scene Patch staging readback encoder"),
            });
        encoder.copy_buffer_to_buffer(staging, 0, &readback, 0, size);
        gpu.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU Scene Patch readback poll failed");
        rx.recv()
            .expect("GPU Scene Patch map callback was dropped")
            .expect("GPU Scene Patch readback map failed");
        let mapped = slice.get_mapped_range();
        let bytes = mapped.to_vec();
        drop(mapped);
        readback.unmap();
        bytes
    }

    struct TestConsumer;

    impl PassNode<'_> for TestConsumer {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    fn input() -> GpuSceneInstanceInput {
        GpuSceneInstanceInput::new(
            NodeHandle::from(KeyData::from_ffi(1)),
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            BoundingBox {
                min: Vec3::splat(-1.0),
                max: Vec3::splat(1.0),
            },
        )
    }

    fn four_table_ids(registry: &mut PersistentBufferRegistry) -> [PersistentBufferId; 4] {
        std::array::from_fn(|_| registry.create_buffer())
    }

    fn four_table_specs() -> [PersistentBufferSpec; 4] {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        [
            PersistentBufferSpec::new(
                BufferDesc::new(std::mem::size_of::<GpuSceneInstanceRecord>() as u64, usage),
                101,
                std::mem::size_of::<GpuSceneInstanceRecord>() as u64,
            ),
            PersistentBufferSpec::new(
                BufferDesc::new(std::mem::size_of::<GpuSceneGeometryRecord>() as u64, usage),
                102,
                std::mem::size_of::<GpuSceneGeometryRecord>() as u64,
            ),
            PersistentBufferSpec::new(
                BufferDesc::new(std::mem::size_of::<GpuSceneMaterialRecord>() as u64, usage),
                103,
                std::mem::size_of::<GpuSceneMaterialRecord>() as u64,
            ),
            PersistentBufferSpec::new(
                BufferDesc::new(std::mem::size_of::<GpuSceneBatchRecord>() as u64, usage),
                104,
                std::mem::size_of::<GpuSceneBatchRecord>() as u64,
            ),
        ]
    }

    fn plan_four_tables(
        registry: &mut PersistentBufferRegistry,
        ids: [PersistentBufferId; 4],
        specs: [PersistentBufferSpec; 4],
        attempt: u64,
        revision: u64,
    ) -> [PersistentBufferPlan; 4] {
        let requests: [_; 4] =
            std::array::from_fn(|index| PersistentBufferPlanRequest::new(ids[index], specs[index]));
        let plans = registry
            .plan_coherent_batch(attempt, revision, &requests)
            .expect("four-table coherent planning failed");
        [plans[0], plans[1], plans[2], plans[3]]
    }

    fn four_patch_members<'a>(
        plans: [PersistentBufferPlan; 4],
        instances: &'a [GpuSceneInstanceRecord],
        geometries: &'a [GpuSceneGeometryRecord],
        materials: &'a [GpuSceneMaterialRecord],
        batches: &'a [GpuSceneBatchRecord],
        dirty_ranges: &'a [GpuSceneDirtyRange],
    ) -> [GpuScenePatchMemberPlan<'a>; 4] {
        [
            GpuScenePatchMemberPlan {
                name: "GpuScene_Instances",
                persistent: plans[0],
                image: GpuScenePatchImage::from_records(
                    plans[0].buffer.spec.desc.logical_size,
                    instances,
                    dirty_ranges,
                ),
            },
            GpuScenePatchMemberPlan {
                name: "GpuScene_Geometries",
                persistent: plans[1],
                image: GpuScenePatchImage::from_records(
                    plans[1].buffer.spec.desc.logical_size,
                    geometries,
                    dirty_ranges,
                ),
            },
            GpuScenePatchMemberPlan {
                name: "GpuScene_Materials",
                persistent: plans[2],
                image: GpuScenePatchImage::from_records(
                    plans[2].buffer.spec.desc.logical_size,
                    materials,
                    dirty_ranges,
                ),
            },
            GpuScenePatchMemberPlan {
                name: "GpuScene_Batches",
                persistent: plans[3],
                image: GpuScenePatchImage::from_records(
                    plans[3].buffer.spec.desc.logical_size,
                    batches,
                    dirty_ranges,
                ),
            },
        ]
    }

    #[test]
    fn patch_is_one_batched_non_rooting_writer() {
        let mut registry = PersistentBufferRegistry::new();
        let mut gpu_scene = GpuScene::new();
        gpu_scene.synchronize_scene(1, [input()]);
        let plan = gpu_scene.plan_instance_table(&mut registry, 1).unwrap();
        let publication_probe = plan.publication_probe;
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let patched = add_gpu_scene_patch(&mut graph, plan);
        assert_eq!(patched.publication_probe, publication_probe);
        assert_eq!(patched.slot_span, 1);
        graph.add_pass("GpuScene_Test_Consumer", |builder| {
            builder.read_buffer(patched.instances);
            builder.mark_side_effect();
            (TestConsumer, ())
        });

        graph.compile_topology();
        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(live, ["GpuScene_Patch", "GpuScene_Test_Consumer"]);
        assert!(graph.has_persistent_buffer_commit_groups());
    }

    #[test]
    fn four_table_family_uses_one_patch_node_and_one_commit_group() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let ids = four_table_ids(&mut registry);
        let plans = plan_four_tables(&mut registry, ids, four_table_specs(), 1, 1);
        let instances = [GpuSceneInstanceRecord::zeroed()];
        let geometries = [GpuSceneGeometryRecord::zeroed()];
        let materials = [GpuSceneMaterialRecord::zeroed()];
        let batches = [GpuSceneBatchRecord::zeroed()];
        let dirty = [GpuSceneDirtyRange {
            first_slot: 0,
            slot_count: 1,
        }];
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let nodes = add_gpu_scene_patch_family(
            &mut graph,
            four_patch_members(plans, &instances, &geometries, &materials, &batches, &dirty),
        );
        graph.add_pass("GpuScene_FourTable_Consumer", |builder| {
            for &node in &nodes {
                builder.read_buffer(node);
            }
            builder.mark_side_effect();
            (TestConsumer, ())
        });

        graph.compile_topology();
        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(live, ["GpuScene_Patch", "GpuScene_FourTable_Consumer"]);
        assert_eq!(
            graph
                .storage
                .passes
                .iter()
                .find(|pass| pass.name == "GpuScene_Patch")
                .unwrap()
                .writes
                .len(),
            4
        );
        assert_eq!(graph.storage.persistent_buffer_commit_groups.len(), 1);

        graph
            .materialize_live_persistent_buffers(&mut registry, &gpu.device)
            .unwrap();
        assert_eq!(graph.storage.persistent_buffer_materializations.len(), 4);
        let encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("unprepared four-table GPU Scene submit"),
            });
        assert!(
            graph
                .submit_command_buffer(&gpu.queue, encoder.finish())
                .is_err(),
            "live atomic family submitted without its Patch encoding"
        );
        assert!(plans.iter().all(|plan| !registry.is_published(plan.buffer)));
    }

    #[test]
    fn dead_four_table_family_materializes_and_publishes_nothing() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let ids = four_table_ids(&mut registry);
        let plans = plan_four_tables(&mut registry, ids, four_table_specs(), 1, 1);
        let instances = [GpuSceneInstanceRecord::zeroed()];
        let geometries = [GpuSceneGeometryRecord::zeroed()];
        let materials = [GpuSceneMaterialRecord::zeroed()];
        let batches = [GpuSceneBatchRecord::zeroed()];
        let dirty = [GpuSceneDirtyRange {
            first_slot: 0,
            slot_count: 1,
        }];
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        add_gpu_scene_patch_family(
            &mut graph,
            four_patch_members(plans, &instances, &geometries, &materials, &batches, &dirty),
        );

        graph.compile_topology();
        assert!(graph.storage.execution_queue.is_empty());
        graph
            .materialize_live_persistent_buffers(&mut registry, &gpu.device)
            .unwrap();
        assert!(graph.storage.persistent_buffer_materializations.is_empty());
        assert!(plans.iter().all(|plan| !registry.is_published(plan.buffer)));
    }

    #[test]
    fn warm_four_table_family_is_read_only_without_patch_or_group() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let ids = four_table_ids(&mut registry);
        let specs = four_table_specs();
        let first = plan_four_tables(&mut registry, ids, specs, 1, 9);
        let materializations = first.map(|plan| {
            registry
                .materialize(&gpu.device, plan.buffer, Some("warm four-table setup"))
                .unwrap()
        });
        let commits: [_; 4] = std::array::from_fn(|index| {
            PersistentBufferCommitRequest::new(
                first[index].commit.unwrap(),
                materializations[index].token,
            )
        });
        registry.commit_batch(&commits).unwrap();
        let warm = plan_four_tables(&mut registry, ids, specs, 2, 9);
        assert!(warm.iter().all(|plan| plan.valid && plan.commit.is_none()));

        let instances = [GpuSceneInstanceRecord::zeroed()];
        let geometries = [GpuSceneGeometryRecord::zeroed()];
        let materials = [GpuSceneMaterialRecord::zeroed()];
        let batches = [GpuSceneBatchRecord::zeroed()];
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let nodes = add_gpu_scene_patch_family(
            &mut graph,
            four_patch_members(warm, &instances, &geometries, &materials, &batches, &[]),
        );
        graph.add_pass("GpuScene_Warm_FourTable_Consumer", |builder| {
            for &node in &nodes {
                builder.read_buffer(node);
            }
            builder.mark_side_effect();
            (TestConsumer, ())
        });

        graph.compile_topology();
        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(live, ["GpuScene_Warm_FourTable_Consumer"]);
        assert!(!graph.has_persistent_buffer_commit_groups());
    }

    #[test]
    fn dead_gpu_scene_branch_has_no_live_patch_or_consumer() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let mut gpu_scene = GpuScene::new();
        gpu_scene.synchronize_scene(1, [input()]);
        let plan = gpu_scene.plan_instance_table(&mut registry, 1).unwrap();
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let patched = add_gpu_scene_patch(&mut graph, plan);
        graph.add_pass("GpuScene_Dead_Consumer", |builder| {
            builder.read_buffer(patched.instances);
            (TestConsumer, ())
        });

        graph.compile_topology();
        graph
            .materialize_live_persistent_buffers(&mut registry, &gpu.device)
            .unwrap();
        assert!(graph.storage.execution_queue.is_empty());
        assert!(
            !graph
                .storage
                .persistent_buffer_materializations
                .contains_key(
                    &graph.storage.resources[patched.instances.index() as usize]
                        .persistent_buffer_id()
                        .unwrap()
                )
        );
    }

    #[test]
    fn published_table_is_a_read_only_import_without_patch_or_commit_group() {
        let gpu = gpu();
        let mut registry = PersistentBufferRegistry::new();
        let mut gpu_scene = GpuScene::new();
        gpu_scene.synchronize_scene(1, [input()]);
        let (persistent, probe) = {
            let first = gpu_scene.plan_instance_table(&mut registry, 1).unwrap();
            (first.persistent, first.publication_probe)
        };
        let materialization = registry
            .materialize(
                &gpu.device,
                persistent.buffer,
                Some("published GPU Scene test"),
            )
            .unwrap();
        registry
            .commit_batch(&[PersistentBufferCommitRequest::new(
                persistent.commit.unwrap(),
                materialization.token,
            )])
            .unwrap();
        assert!(gpu_scene.acknowledge_if_published(&registry, probe));

        let plan = gpu_scene.plan_instance_table(&mut registry, 2).unwrap();
        assert!(plan.persistent.valid);
        assert!(plan.persistent.commit.is_none());
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let patched = add_gpu_scene_patch(&mut graph, plan);
        graph.add_pass("GpuScene_Published_Consumer", |builder| {
            builder.read_buffer(patched.instances);
            builder.mark_side_effect();
            (TestConsumer, ())
        });

        graph.compile_topology();
        graph
            .materialize_live_persistent_buffers(&mut registry, &gpu.device)
            .unwrap();
        let live: Vec<_> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&index| graph.storage.passes[index].name)
            .collect();
        assert_eq!(live, ["GpuScene_Published_Consumer"]);
        assert!(!graph.has_persistent_buffer_commit_groups());
        assert!(graph.has_persistent_buffer_imports());
    }

    #[test]
    fn staging_covers_empty_full_image_and_packs_sparse_delta_ranges() {
        let gpu = gpu();
        let empty = [GpuSceneInstanceRecord::zeroed()];
        let (full_staging, full) = create_gpu_scene_patch_staging(
            &gpu.device,
            std::mem::size_of_val(&empty) as u64,
            &empty,
            &[],
            true,
        );
        assert_eq!(full.coverage, PersistentBufferWriteCoverage::Full);
        assert_eq!(full.upload_size, 256);
        assert_eq!(
            read_staging(&gpu, &full_staging),
            bytemuck::cast_slice::<_, u8>(&empty)
        );

        let mut records = [GpuSceneInstanceRecord::zeroed(); 8];
        for (index, record) in records.iter_mut().enumerate() {
            record.metadata = [u32::try_from(index).unwrap() + 1, 1, 0, 0];
        }
        let ranges = [
            GpuSceneDirtyRange {
                first_slot: 0,
                slot_count: 1,
            },
            GpuSceneDirtyRange {
                first_slot: 2,
                slot_count: 2,
            },
        ];
        let (delta_staging, delta) = create_gpu_scene_patch_staging(
            &gpu.device,
            std::mem::size_of_val(&records) as u64,
            &records,
            &ranges,
            false,
        );
        assert_eq!(delta.coverage, PersistentBufferWriteCoverage::Delta);
        assert_eq!(delta.upload_size, 3 * 256);
        let mut expected = Vec::with_capacity(delta.upload_size as usize);
        expected.extend_from_slice(bytemuck::bytes_of(&records[0]));
        expected.extend_from_slice(bytemuck::cast_slice(&records[2..4]));
        assert_eq!(read_staging(&gpu, &delta_staging), expected);

        let dense_ranges = [GpuSceneDirtyRange {
            first_slot: 0,
            slot_count: 6,
        }];
        let (dense_staging, dense) = create_gpu_scene_patch_staging(
            &gpu.device,
            std::mem::size_of_val(&records) as u64,
            &records,
            &dense_ranges,
            false,
        );
        assert_eq!(dense.coverage, PersistentBufferWriteCoverage::Full);
        assert_eq!(dense.upload_size, std::mem::size_of_val(&records) as u64);
        assert_eq!(
            read_staging(&gpu, &dense_staging),
            bytemuck::cast_slice::<_, u8>(&records)
        );
    }

    #[test]
    fn fragmented_delta_upgrades_to_one_complete_image() {
        let gpu = gpu();
        let record_count = (GPU_SCENE_DELTA_COPY_RANGE_BUDGET + 1) * 4;
        let mut records = vec![GpuSceneInstanceRecord::zeroed(); record_count];
        let mut ranges = Vec::with_capacity(GPU_SCENE_DELTA_COPY_RANGE_BUDGET + 1);
        for range_index in 0..=GPU_SCENE_DELTA_COPY_RANGE_BUDGET {
            let slot = range_index * 4;
            records[slot].metadata = [u32::try_from(slot).unwrap() + 1, 1, 0, 0];
            ranges.push(GpuSceneDirtyRange {
                first_slot: u32::try_from(slot).unwrap(),
                slot_count: 1,
            });
        }

        let logical_size = std::mem::size_of_val(records.as_slice()) as u64;
        let (staging, prepared) =
            create_gpu_scene_patch_staging(&gpu.device, logical_size, &records, &ranges, false);

        assert_eq!(prepared.coverage, PersistentBufferWriteCoverage::Full);
        assert_eq!(prepared.upload_size, logical_size);
        assert_eq!(
            read_staging(&gpu, &staging),
            bytemuck::cast_slice::<_, u8>(records.as_slice())
        );
    }

    #[test]
    fn staging_packs_four_table_images_in_canonical_member_order() {
        let gpu = gpu();
        let mut instances = [GpuSceneInstanceRecord::zeroed(); 2];
        instances[0].metadata = [1, 1, 7, 0];
        instances[1].metadata = [2, 1, 7, 0];
        let mut geometries = [GpuSceneGeometryRecord::zeroed(); 2];
        geometries[0].metadata = [3, 1, 1, 12];
        geometries[1].metadata = [4, 1, 1, 24];
        let mut materials = [GpuSceneMaterialRecord::zeroed(); 2];
        materials[0].metadata = [5, 1, 1, 0];
        materials[1].metadata = [6, 1, 1, 1];
        let mut batches = [GpuSceneBatchRecord::zeroed(); 2];
        batches[0].metadata = [7, 1, 7, 1];
        batches[1].metadata = [8, 1, 7, 1];
        let dirty_first = [GpuSceneDirtyRange {
            first_slot: 0,
            slot_count: 1,
        }];
        let dirty_second = [GpuSceneDirtyRange {
            first_slot: 1,
            slot_count: 1,
        }];
        let images = [
            GpuScenePatchImage::from_records(
                std::mem::size_of_val(&instances) as u64,
                &instances,
                &dirty_second,
            ),
            GpuScenePatchImage::from_records(
                std::mem::size_of_val(&geometries) as u64,
                &geometries,
                &dirty_first,
            ),
            GpuScenePatchImage::from_records(
                std::mem::size_of_val(&materials) as u64,
                &materials,
                &dirty_second,
            ),
            GpuScenePatchImage::from_records(
                std::mem::size_of_val(&batches) as u64,
                &batches,
                &dirty_first,
            ),
        ];

        let (full_staging, full) =
            create_gpu_scene_patch_staging_for_images(&gpu.device, &images, true);
        assert_eq!(full.coverage, PersistentBufferWriteCoverage::Full);
        let mut expected_full = Vec::<u8>::new();
        expected_full.extend_from_slice(bytemuck::cast_slice(&instances));
        expected_full.extend_from_slice(bytemuck::cast_slice(&geometries));
        expected_full.extend_from_slice(bytemuck::cast_slice(&materials));
        expected_full.extend_from_slice(bytemuck::cast_slice(&batches));
        assert_eq!(full.upload_size, expected_full.len() as u64);
        assert_eq!(read_staging(&gpu, &full_staging), expected_full);

        let (delta_staging, delta) =
            create_gpu_scene_patch_staging_for_images(&gpu.device, &images, false);
        assert_eq!(delta.coverage, PersistentBufferWriteCoverage::Delta);
        let mut expected_delta = Vec::<u8>::new();
        expected_delta.extend_from_slice(bytemuck::bytes_of(&instances[1]));
        expected_delta.extend_from_slice(bytemuck::bytes_of(&geometries[0]));
        expected_delta.extend_from_slice(bytemuck::bytes_of(&materials[1]));
        expected_delta.extend_from_slice(bytemuck::bytes_of(&batches[0]));
        assert_eq!(delta.upload_size, expected_delta.len() as u64);
        assert_eq!(read_staging(&gpu, &delta_staging), expected_delta);
    }
}
