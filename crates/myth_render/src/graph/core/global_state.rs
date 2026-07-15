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
mod tests;
