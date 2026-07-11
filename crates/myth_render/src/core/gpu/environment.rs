use std::hash::{Hash, Hasher};

use glam::Vec3;
use wgpu::TextureViewDimension;

use myth_assets::AssetServer;
use myth_resources::image::Image;
use myth_resources::texture::{Texture, TextureSource};
use myth_scene::background::{BackgroundMode, ProceduralSkyParams};
use myth_scene::environment::Environment;

use super::ResourceManager;

/// How the scene environment source must be converted before PMREM filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeSourceType {
    /// Procedural atmosphere bake populates the base cube directly.
    Procedural,
    /// 2D equirectangular HDR panorama.
    Equirectangular,
    /// Cubemap source texture.
    Cubemap,
}

/// CPU-only description of a scene's environment source.
///
/// Physical base-cube/PMREM textures and their publication state belong to
/// RDG's persistent-resource registry. Keeping only source metadata here
/// prevents graph construction from allocating GPU resources before DPE.
#[derive(Debug, Clone, Copy)]
pub struct GpuEnvironment {
    pub base_cube_size: u32,
    pub pmrem_size: u32,
    /// Logical content fingerprint of the base environment source.
    ///
    /// Texture sizes are intentionally not folded into this value: the base
    /// and PMREM persistent-cache keys own their respective size fields.
    pub source_version: u64,
    pub source_type: CubeSourceType,
    pub source_key: Option<TextureSource>,
    pub source_ready: bool,
    pub procedural_physics_hash: u64,
    pub procedural_static_bake_hash: u64,
    pub last_used_frame: u64,
}

impl GpuEnvironment {
    #[inline]
    #[must_use]
    pub fn env_map_max_mip_level(&self) -> f32 {
        self.pmrem_size.max(1).ilog2() as f32
    }
}

/// CPU-only identity and readiness snapshot for one texture source.
///
/// This type deliberately contains no physical WGPU resource. Live pass
/// preparation can use the source after DPE while graph construction relies on
/// the logical version and view dimension for cache and pipeline decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LogicalTextureSourceSnapshot {
    pub(crate) version: u64,
    pub(crate) ready: bool,
    pub(crate) view_dimension: TextureViewDimension,
}

/// One coherent CPU-only snapshot of every source that contributes to a
/// procedural environment bake.
///
/// Composer shares this value between Atmosphere and Skybox. Comparing its
/// identity against the earlier frame metadata also closes the async-asset
/// window before a persistent cache attempt is planned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProceduralEnvironmentSnapshot {
    pub(crate) source_ready: bool,
    pub(crate) source_version: u64,
    pub(crate) physics_hash: u64,
    pub(crate) static_bake_hash: u64,
    pub(crate) moon_texture: Option<LogicalTextureSourceSnapshot>,
    pub(crate) starbox_texture: Option<LogicalTextureSourceSnapshot>,
}

impl ProceduralEnvironmentSnapshot {
    /// Whether begin-frame metadata and the Composer-time source snapshot still
    /// describe the exact same procedural bake input.
    #[inline]
    #[must_use]
    pub(crate) fn matches_gpu_environment(self, environment: &GpuEnvironment) -> bool {
        environment.source_type == CubeSourceType::Procedural
            && environment.source_ready == self.source_ready
            && environment.source_version == self.source_version
            && environment.procedural_physics_hash == self.physics_hash
            && environment.procedural_static_bake_hash == self.static_bake_hash
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedEnvironmentSource {
    source_type: CubeSourceType,
    key: TextureSource,
    version: u64,
    source_ready: bool,
}

fn hash_f32<H: Hasher>(hasher: &mut H, value: f32) {
    value.to_bits().hash(hasher);
}

fn hash_vec3<H: Hasher>(hasher: &mut H, value: Vec3) {
    hash_f32(hasher, value.x);
    hash_f32(hasher, value.y);
    hash_f32(hasher, value.z);
}

pub(super) fn snapshot_ready_asset_texture_source(
    source: TextureSource,
    texture: &Texture,
    texture_version: u32,
    texture_identity: u64,
    image: &Image,
    image_version: u32,
    image_identity: u64,
) -> LogicalTextureSourceSnapshot {
    let mut hasher = rustc_hash::FxHasher::default();
    source.hash(&mut hasher);
    1u8.hash(&mut hasher);
    texture.uuid.hash(&mut hasher);
    texture_version.hash(&mut hasher);
    texture_identity.hash(&mut hasher);
    texture.image.hash(&mut hasher);
    texture.view_dimension.hash(&mut hasher);
    texture.color_space.hash(&mut hasher);
    1u8.hash(&mut hasher);
    image.uuid.hash(&mut hasher);
    image_version.hash(&mut hasher);
    image_identity.hash(&mut hasher);
    image.width.hash(&mut hasher);
    image.height.hash(&mut hasher);
    image.depth.hash(&mut hasher);
    image.mip_level_count.hash(&mut hasher);
    image.dimension.hash(&mut hasher);
    image.format.hash(&mut hasher);

    let image_dimension = image.dimension.to_wgpu();
    let ready = match texture.view_dimension {
        TextureViewDimension::D1 => image_dimension == wgpu::TextureDimension::D1,
        TextureViewDimension::D2 => {
            image_dimension == wgpu::TextureDimension::D2 && image.depth == 1
        }
        TextureViewDimension::D2Array => image_dimension == wgpu::TextureDimension::D2,
        TextureViewDimension::Cube => {
            image_dimension == wgpu::TextureDimension::D2
                && image.depth == 6
                && image.width == image.height
        }
        TextureViewDimension::CubeArray => {
            image_dimension == wgpu::TextureDimension::D2
                && image.depth >= 6
                && image.depth.is_multiple_of(6)
                && image.width == image.height
        }
        TextureViewDimension::D3 => image_dimension == wgpu::TextureDimension::D3,
    };

    LogicalTextureSourceSnapshot {
        version: hasher.finish(),
        ready,
        view_dimension: texture.view_dimension,
    }
}

pub(super) fn snapshot_attachment_texture_source(
    source: TextureSource,
    revision: Option<u64>,
) -> LogicalTextureSourceSnapshot {
    let TextureSource::Attachment(_, view_dimension) = source else {
        unreachable!("attachment snapshot requires an attachment source")
    };
    let mut hasher = rustc_hash::FxHasher::default();
    source.hash(&mut hasher);
    revision.hash(&mut hasher);
    LogicalTextureSourceSnapshot {
        version: hasher.finish(),
        ready: revision.is_some(),
        view_dimension,
    }
}

fn hash_optional_texture_source(
    source: Option<TextureSource>,
    logical: Option<LogicalTextureSourceSnapshot>,
) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    source.hash(&mut hasher);
    if let Some(logical) = logical {
        logical.version.hash(&mut hasher);
    }
    hasher.finish()
}

pub(crate) fn procedural_physics_hash(params: &ProceduralSkyParams) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    hash_vec3(&mut hasher, params.rayleigh_scattering);
    hash_f32(&mut hasher, params.rayleigh_scale_height);
    hash_f32(&mut hasher, params.mie_scattering);
    hash_f32(&mut hasher, params.mie_absorption);
    hash_f32(&mut hasher, params.mie_scale_height);
    hash_f32(&mut hasher, params.mie_anisotropy);
    hash_vec3(&mut hasher, params.ozone_absorption);
    hash_f32(&mut hasher, params.planet_radius);
    hash_f32(&mut hasher, params.atmosphere_radius);
    hasher.finish()
}

fn procedural_static_bake_hash(
    params: &ProceduralSkyParams,
    moon_texture_hash: u64,
    starbox_hash: u64,
) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    hash_f32(&mut hasher, params.sun_intensity);
    hash_f32(&mut hasher, params.sun_disk_size);
    hash_f32(&mut hasher, params.moon_intensity);
    hash_f32(&mut hasher, params.moon_disk_size);
    moon_texture_hash.hash(&mut hasher);
    hash_f32(&mut hasher, params.star_intensity);
    hash_vec3(&mut hasher, params.star_axis);
    starbox_hash.hash(&mut hasher);
    // Dynamic celestial directions and `star_rotation` are deliberately
    // excluded. The persistent-cache coordinator throttles sun/moon motion
    // against post-submit publication baselines instead of treating every
    // tiny DayNight tick as a static configuration change.
    hasher.finish()
}

fn procedural_source_version(
    physics_hash: u64,
    static_bake_hash: u64,
    sun_direction: Vec3,
    moon_direction: Vec3,
) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    physics_hash.hash(&mut hasher);
    static_bake_hash.hash(&mut hasher);
    hash_vec3(&mut hasher, sun_direction);
    hash_vec3(&mut hasher, moon_direction);
    hasher.finish()
}

impl ResourceManager {
    /// Resolves the logical identity and readiness of a texture without
    /// uploading it or otherwise interacting with the GPU.
    pub(crate) fn snapshot_texture_source(
        &self,
        assets: &AssetServer,
        source: TextureSource,
    ) -> LogicalTextureSourceSnapshot {
        let mut hasher = rustc_hash::FxHasher::default();
        source.hash(&mut hasher);

        match source {
            TextureSource::Asset(handle) => {
                let Some((texture, texture_version, texture_identity)) =
                    assets.textures.get_entry_with_identity(handle)
                else {
                    0u8.hash(&mut hasher);
                    return LogicalTextureSourceSnapshot {
                        version: hasher.finish(),
                        ready: false,
                        view_dimension: TextureViewDimension::D2,
                    };
                };

                let Some((image, image_version, image_identity)) =
                    assets.images.get_entry_with_identity(texture.image)
                else {
                    1u8.hash(&mut hasher);
                    texture.uuid.hash(&mut hasher);
                    texture_version.hash(&mut hasher);
                    texture_identity.hash(&mut hasher);
                    texture.image.hash(&mut hasher);
                    texture.view_dimension.hash(&mut hasher);
                    texture.color_space.hash(&mut hasher);
                    0u8.hash(&mut hasher);
                    return LogicalTextureSourceSnapshot {
                        version: hasher.finish(),
                        ready: false,
                        view_dimension: texture.view_dimension,
                    };
                };

                snapshot_ready_asset_texture_source(
                    source,
                    &texture,
                    texture_version,
                    texture_identity,
                    &image,
                    image_version,
                    image_identity,
                )
            }
            TextureSource::Attachment(id, _) => {
                // The revision changes when a stable attachment ID is rebound
                // to new content, so cached environment data cannot silently
                // survive an in-place producer update.
                let revision = self.get_internal_texture_revision(id);
                snapshot_attachment_texture_source(source, revision)
            }
        }
    }

    /// Capture procedural physics and optional texture identities once for all
    /// graph consumers, without touching WGPU.
    pub(crate) fn snapshot_procedural_environment(
        &self,
        assets: &AssetServer,
        params: &ProceduralSkyParams,
    ) -> ProceduralEnvironmentSnapshot {
        let moon_texture = params
            .moon_albedo_texture
            .map(|source| self.snapshot_texture_source(assets, source));
        let starbox_texture = params
            .starbox_texture
            .map(|source| self.snapshot_texture_source(assets, source));
        let source_ready = moon_texture
            .is_none_or(|source| source.ready && source.view_dimension == TextureViewDimension::D2)
            && starbox_texture.is_none_or(|source| {
                source.ready
                    && matches!(
                        source.view_dimension,
                        TextureViewDimension::D2 | TextureViewDimension::Cube
                    )
            });
        let physics_hash = procedural_physics_hash(params);
        let moon_texture_hash =
            hash_optional_texture_source(params.moon_albedo_texture, moon_texture);
        let starbox_hash = hash_optional_texture_source(params.starbox_texture, starbox_texture);
        let static_bake_hash = procedural_static_bake_hash(params, moon_texture_hash, starbox_hash);
        let source_version = procedural_source_version(
            physics_hash,
            static_bake_hash,
            params.sun_direction,
            params.moon_direction,
        );

        ProceduralEnvironmentSnapshot {
            source_ready,
            source_version,
            physics_hash,
            static_bake_hash,
            moon_texture,
            starbox_texture,
        }
    }

    /// Resolve or create the CPU environment metadata for a scene.
    ///
    /// This method performs asset-state inspection and hashing only. It never
    /// uploads a texture, creates a WGPU object, or mutates RDG publication
    /// state. Sun-motion throttling belongs to the feature-owned persistent
    /// cache domain, where it can compare against post-submit publication
    /// truth instead of mutating this current-frame snapshot.
    pub fn resolve_gpu_environment(
        &mut self,
        scene_id: u32,
        assets: &AssetServer,
        environment: &Environment,
        background: &BackgroundMode,
    ) -> f32 {
        // A background alone does not imply IBL except for procedural sky.
        // Leave dormant metadata untouched so TTL reflects actual IBL use.
        let has_active_environment =
            matches!(background, BackgroundMode::Procedural(_)) || environment.has_env_map();
        if !has_active_environment {
            return 0.0;
        }

        let config = environment.map_config();

        let (source_type, source_key, source_ready, source_version, physics_hash, static_bake_hash) =
            if let BackgroundMode::Procedural(params) = background {
                let snapshot = self.snapshot_procedural_environment(assets, params);
                (
                    CubeSourceType::Procedural,
                    None,
                    snapshot.source_ready,
                    snapshot.source_version,
                    snapshot.physics_hash,
                    snapshot.static_bake_hash,
                )
            } else {
                let source = *environment
                    .source_env_map()
                    .expect("active non-procedural environment must have a source map");
                let logical_source = self.snapshot_texture_source(assets, source);
                let (source_type, supported_dimension) = match logical_source.view_dimension {
                    TextureViewDimension::D2 => (CubeSourceType::Equirectangular, true),
                    TextureViewDimension::Cube => (CubeSourceType::Cubemap, true),
                    _ => (CubeSourceType::Equirectangular, false),
                };
                let resolved = ResolvedEnvironmentSource {
                    source_type,
                    key: source,
                    version: logical_source.version,
                    source_ready: logical_source.ready && supported_dimension,
                };

                (
                    resolved.source_type,
                    Some(resolved.key),
                    resolved.source_ready,
                    resolved.version,
                    0,
                    0,
                )
            };

        if let Some(gpu_env) = self.environments.scene_environment_mut(scene_id) {
            gpu_env.base_cube_size = config.base_cube_size.max(1);
            gpu_env.pmrem_size = config.pmrem_size.max(1);
            gpu_env.source_version = source_version;
            gpu_env.source_type = source_type;
            gpu_env.source_key = source_key;
            gpu_env.source_ready = source_ready;
            gpu_env.procedural_physics_hash = physics_hash;
            gpu_env.procedural_static_bake_hash = static_bake_hash;
            gpu_env.last_used_frame = self.frame_index;
        } else {
            self.environments.insert_scene_environment(
                scene_id,
                GpuEnvironment {
                    base_cube_size: config.base_cube_size.max(1),
                    pmrem_size: config.pmrem_size.max(1),
                    source_version,
                    source_type,
                    source_key,
                    source_ready,
                    procedural_physics_hash: physics_hash,
                    procedural_static_bake_hash: static_bake_hash,
                    last_used_frame: self.frame_index,
                },
            );
        }

        self.environments
            .scene_environment(scene_id)
            .map_or(0.0, GpuEnvironment::env_map_max_mip_level)
    }

    #[inline]
    #[must_use]
    pub fn gpu_environment(&self, scene_id: u32) -> Option<&GpuEnvironment> {
        self.environments.scene_environment(scene_id)
    }

    #[inline]
    #[must_use]
    pub fn gpu_environment_mut(&mut self, scene_id: u32) -> Option<&mut GpuEnvironment> {
        self.environments.scene_environment_mut(scene_id)
    }

    /// Returns whether the current logical source can be consumed.
    ///
    /// This does not report RDG persistent-cache publication state; callers
    /// needing a usable PMREM must query that registry separately.
    #[inline]
    #[must_use]
    #[deprecated(
        since = "0.3.0",
        note = "reports source readiness only; use scene_environment_source_ready or Renderer::published_environment_max_mip_level"
    )]
    pub fn scene_environment_ready(&self, scene_id: u32) -> bool {
        self.scene_environment_source_ready(scene_id)
    }

    /// Returns whether the current logical source is available to a live
    /// conversion node. This is intentionally distinct from post-submit PMREM
    /// publication readiness.
    #[inline]
    #[must_use]
    pub fn scene_environment_source_ready(&self, scene_id: u32) -> bool {
        self.environments
            .scene_environment(scene_id)
            .is_some_and(|gpu_env| gpu_env.source_ready)
    }

    /// Desired maximum mip level from the current scene configuration.
    ///
    /// A pending replacement may still sample an older published PMREM with a
    /// different mip count. External consumers should query publication truth
    /// through `Renderer::published_environment_max_mip_level`.
    #[inline]
    #[must_use]
    pub fn desired_environment_max_mip_level(&self, scene_id: u32) -> f32 {
        self.environments
            .scene_environment(scene_id)
            .map_or(0.0, GpuEnvironment::env_map_max_mip_level)
    }

    #[inline]
    #[must_use]
    #[deprecated(
        since = "0.3.0",
        note = "returns desired configuration, not published PMREM state; use desired_environment_max_mip_level or Renderer::published_environment_max_mip_level"
    )]
    pub fn get_env_map_max_mip_level(&self, scene_id: u32) -> f32 {
        self.desired_environment_max_mip_level(scene_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_static_bake_hash_excludes_dynamic_celestial_motion() {
        let params = ProceduralSkyParams::golden_hour();
        let initial = procedural_static_bake_hash(&params, 11, 22);

        let mut rotated_stars = params.clone();
        rotated_stars.star_rotation += 1.0;
        assert_eq!(
            procedural_static_bake_hash(&rotated_stars, 11, 22),
            initial,
            "continuous star rotation must not invalidate PMREM"
        );

        let mut moved_moon = params;
        moved_moon.moon_direction = Vec3::X;
        assert_eq!(
            procedural_static_bake_hash(&moved_moon, 11, 22),
            initial,
            "continuous moon motion must be handled by the angular throttle"
        );
        assert_ne!(
            procedural_static_bake_hash(&moved_moon, 12, 22),
            initial,
            "changing the moon texture remains a static bake invalidation"
        );
    }

    #[test]
    fn procedural_source_version_tracks_both_celestial_directions() {
        let params = ProceduralSkyParams::golden_hour();
        let initial = procedural_source_version(1, 2, params.sun_direction, params.moon_direction);
        let moved_sun = procedural_source_version(1, 2, Vec3::X, params.moon_direction);
        let moved_moon = procedural_source_version(1, 2, params.sun_direction, Vec3::X);

        assert_ne!(initial, moved_sun);
        assert_ne!(initial, moved_moon);
    }

    #[test]
    fn composer_snapshot_rejects_stale_begin_frame_metadata() {
        let snapshot = ProceduralEnvironmentSnapshot {
            source_ready: true,
            source_version: 17,
            physics_hash: 23,
            static_bake_hash: 29,
            moon_texture: None,
            starbox_texture: None,
        };
        let environment = GpuEnvironment {
            base_cube_size: 64,
            pmrem_size: 32,
            source_version: snapshot.source_version,
            source_type: CubeSourceType::Procedural,
            source_key: None,
            source_ready: snapshot.source_ready,
            procedural_physics_hash: snapshot.physics_hash,
            procedural_static_bake_hash: snapshot.static_bake_hash,
            last_used_frame: 1,
        };

        assert!(snapshot.matches_gpu_environment(&environment));

        let mut stale_version = environment;
        stale_version.source_version += 1;
        assert!(!snapshot.matches_gpu_environment(&stale_version));

        let mut stale_readiness = environment;
        stale_readiness.source_ready = false;
        assert!(!snapshot.matches_gpu_environment(&stale_readiness));
    }
}
