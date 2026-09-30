//! Concise constructors for WGPU bind-group layout descriptors.
//!
//! The helpers keep binding visibility explicit while centralizing the safe
//! defaults used by most renderer features: no dynamic buffer offset, no
//! minimum binding size, and no binding array. Advanced callers can use
//! [`buffer`] and [`entry_with_count`] without losing access to WGPU semantics.

use std::num::NonZeroU32;

#[must_use]
pub const fn entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
) -> wgpu::BindGroupLayoutEntry {
    entry_with_count(binding, visibility, ty, None)
}

#[must_use]
pub const fn entry_with_count(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
    count: Option<NonZeroU32>,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count,
    }
}

#[must_use]
pub const fn buffer_type(
    ty: wgpu::BufferBindingType,
    has_dynamic_offset: bool,
    min_binding_size: Option<wgpu::BufferSize>,
) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset,
        min_binding_size,
    }
}

#[must_use]
pub const fn uniform_buffer_type() -> wgpu::BindingType {
    buffer_type(wgpu::BufferBindingType::Uniform, false, None)
}

#[must_use]
pub const fn storage_buffer_type(read_only: bool) -> wgpu::BindingType {
    buffer_type(wgpu::BufferBindingType::Storage { read_only }, false, None)
}

#[must_use]
pub const fn texture_type(
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
    multisampled: bool,
) -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type,
        view_dimension,
        multisampled,
    }
}

#[must_use]
pub const fn sampler_type(ty: wgpu::SamplerBindingType) -> wgpu::BindingType {
    wgpu::BindingType::Sampler(ty)
}

#[must_use]
pub const fn storage_texture_type(
    access: wgpu::StorageTextureAccess,
    format: wgpu::TextureFormat,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access,
        format,
        view_dimension,
    }
}

#[must_use]
pub const fn buffer(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
    has_dynamic_offset: bool,
    min_binding_size: Option<wgpu::BufferSize>,
) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        visibility,
        buffer_type(ty, has_dynamic_offset, min_binding_size),
    )
}

#[must_use]
pub const fn uniform_buffer(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    buffer(
        binding,
        visibility,
        wgpu::BufferBindingType::Uniform,
        false,
        None,
    )
}

#[must_use]
pub const fn storage_buffer(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    buffer(
        binding,
        visibility,
        wgpu::BufferBindingType::Storage { read_only },
        false,
        None,
    )
}

#[must_use]
pub const fn texture(
    binding: u32,
    visibility: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
    multisampled: bool,
) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        visibility,
        texture_type(sample_type, view_dimension, multisampled),
    )
}

#[must_use]
pub const fn texture_2d(
    binding: u32,
    visibility: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
) -> wgpu::BindGroupLayoutEntry {
    texture(
        binding,
        visibility,
        sample_type,
        wgpu::TextureViewDimension::D2,
        false,
    )
}

#[must_use]
pub const fn depth_texture_2d(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    texture_2d(binding, visibility, wgpu::TextureSampleType::Depth)
}

#[must_use]
pub const fn sampler(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    entry(binding, visibility, sampler_type(ty))
}

#[must_use]
pub const fn storage_texture(
    binding: u32,
    visibility: wgpu::ShaderStages,
    access: wgpu::StorageTextureAccess,
    format: wgpu::TextureFormat,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        visibility,
        storage_texture_type(access, format, view_dimension),
    )
}
