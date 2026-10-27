//! Shader Template Manager
//!
//! Manages WGSL shaders using the minijinja template engine and provides
//! a centralized `ShaderModule` cache shared by all pipeline-creation paths.
//!
//! All shaders — whether loaded from built-in template files or provided as
//! inline WGSL strings — are compiled through the unified
//! [`ShaderManager::get_or_compile`] method, which accepts a [`ShaderSource`]
//! discriminant to select the source kind.

use minijinja::value::{Object, Value};
use minijinja::{Environment, Error, ErrorKind, syntax::SyntaxConfig};
use myth_resources::material::ShaderTemplateMode;
use rust_embed::RustEmbed;
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use xxhash_rust::xxh3::xxh3_128;

use super::shader_gen::{ShaderCompilationOptions, ShaderGenerator};

pub static SHADER_ENV: OnceLock<Environment<'static>> = OnceLock::new();

#[derive(RustEmbed)]
#[folder = "src/pipeline/shaders"]
struct ShaderAssets;

pub fn get_env() -> &'static Environment<'static> {
    SHADER_ENV.get_or_init(|| {
        let mut env = Environment::new();

        let syntax = SyntaxConfig::builder()
            .block_delimiters("{$", "$}")
            .variable_delimiters("{{", "}}")
            .line_statement_prefix("$$")
            .build()
            .expect("Failed to configure Jinja2 syntax");

        env.set_syntax(syntax);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_undefined_behavior(minijinja::UndefinedBehavior::SemiStrict);

        env.set_loader(shader_loader);

        env.add_function("next_loc", next_location);

        env
    })
}

fn shader_loader(name: &str) -> Result<Option<String>, Error> {
    let filename = if std::path::Path::new(name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wgsl"))
    {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("{name}.wgsl"))
    };

    #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
    {
        let path = std::path::Path::new("src/pipeline/shaders").join(filename.as_ref());
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(source) => return Ok(Some(source)),
                Err(e) => {
                    return Err(Error::new(
                        ErrorKind::TemplateNotFound,
                        format!("Failed to read file: {e}"),
                    ));
                }
            }
        }
    }

    if let Some(file) = ShaderAssets::get(&filename)
        && let Ok(source) = std::str::from_utf8(file.data.as_ref())
    {
        return Ok(Some(source.to_string()));
    }

    Ok(None)
}

fn next_location(allocator: &LocationAllocator) -> u32 {
    allocator.next()
}

#[derive(Debug, Serialize)]
pub struct LocationAllocator {
    #[serde(skip)]
    counter: AtomicU32,
}

impl Default for LocationAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl LocationAllocator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            counter: AtomicU32::new(0),
        }
    }

    pub fn next(&self) -> u32 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }
}

impl Object for LocationAllocator {
    fn call_method(
        self: &Arc<Self>,
        _state: &minijinja::State,
        name: &str,
        _args: &[Value],
    ) -> Result<Value, Error> {
        if name == "next" {
            Ok(Value::from(self.next()))
        } else {
            Err(Error::new(
                ErrorKind::UnknownMethod,
                format!("method {name} not found"),
            ))
        }
    }
}

// ─── ShaderSource ─────────────────────────────────────────────────────────────

/// Identifies the origin of a shader's WGSL source.
///
/// * [`File`](Self::File) — a named template lookup. Registered template
///   overrides take priority; otherwise the embedded shader asset loader is
///   used (e.g. `"entry/main/physical"` or `"entry/utility/skybox"`).
/// * [`Inline`](Self::Inline) — a WGSL string supplied directly at call-time,
///   together with the mode that controls whether it is interpreted as a full
///   template or only a material body.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub enum ShaderSource<'a> {
    /// Resolve a named template from registered overrides or built-in assets.
    File(&'a str),
    /// Use an inline WGSL source string identified by `name` for labels.
    Inline {
        name: &'a str,
        source: &'a str,
        mode: ShaderTemplateMode,
    },
}

// ─── ShaderManager ────────────────────────────────────────────────────────────

/// Centralized shader module cache.
///
/// Deduplicates compiled `wgpu::ShaderModule`s by hashing the **final** WGSL
/// source with xxh3-128. Shaders are compiled through the single
/// [`get_or_compile`](Self::get_or_compile) entry point, which accepts a
/// [`ShaderSource`] to distinguish file-based templates from inline sources.
///
/// Custom shader templates registered via [`register_template`](Self::register_template)
/// are stored alongside the built-in environment. When a template name is
/// resolved, custom templates take priority over built-in ones.
///
/// Owned by `RendererState`; references are handed out via `PrepareContext`.
pub struct ShaderManager {
    /// xxh3-128 of final WGSL → compiled module.
    module_cache: FxHashMap<u128, wgpu::ShaderModule>,
    /// User-registered custom shader templates (name → WGSL source).
    custom_templates: FxHashMap<String, RegisteredShaderTemplate>,
}

#[derive(Debug, Clone)]
struct RegisteredShaderTemplate {
    source: String,
    mode: ShaderTemplateMode,
    identity_hash: u64,
}

fn hash_file_template_identity(name: &str) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    0u8.hash(&mut hasher);
    name.hash(&mut hasher);
    hasher.finish()
}

fn hash_inline_template_identity(source: &str, mode: ShaderTemplateMode) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    1u8.hash(&mut hasher);
    mode.hash(&mut hasher);
    source.hash(&mut hasher);
    hasher.finish()
}

fn hash_pipeline_shader_key(source_identity_hash: u64, options: &ShaderCompilationOptions) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    source_identity_hash.hash(&mut hasher);
    options.compute_hash().hash(&mut hasher);
    hasher.finish()
}

impl Default for ShaderManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            module_cache: FxHashMap::default(),
            custom_templates: FxHashMap::default(),
        }
    }

    /// Registers a custom full shader template source under the given name.
    ///
    /// Once registered, any named lookup that resolves this `name` via
    /// [`ShaderSource::File`] will use the provided WGSL source instead of
    /// looking up a built-in template.
    ///
    /// The source string is treated as an exact WGSL template and goes through
    /// the minijinja engine as-is, so `{$ include "chunks/..." $}` directives
    /// are fully supported.
    pub fn register_template(&mut self, name: impl Into<String>, source: impl Into<String>) {
        self.register_template_with_mode(name, source, ShaderTemplateMode::Template);
    }

    /// Registers a custom shader source with an explicit interpretation mode.
    fn register_template_with_mode(
        &mut self,
        name: impl Into<String>,
        source: impl Into<String>,
        mode: ShaderTemplateMode,
    ) {
        let name = name.into();
        let source = source.into();
        log::info!("Registered custom shader template: {name}");
        self.custom_templates.insert(
            name,
            RegisteredShaderTemplate {
                identity_hash: hash_inline_template_identity(&source, mode),
                source,
                mode,
            },
        );
    }

    /// Returns the stable source-identity hash for a named or inline shader.
    #[must_use]
    pub fn source_identity_hash(&self, source: ShaderSource<'_>) -> u64 {
        match source {
            ShaderSource::File(path) => self.custom_templates.get(path).map_or_else(
                || hash_file_template_identity(path),
                |template| template.identity_hash,
            ),
            ShaderSource::Inline { source, mode, .. } => {
                hash_inline_template_identity(source, mode)
            }
        }
    }

    /// Returns the graphics-pipeline shader hash used by the L2 pipeline cache.
    #[must_use]
    pub fn pipeline_shader_hash(
        &self,
        source: ShaderSource<'_>,
        options: &ShaderCompilationOptions,
    ) -> u64 {
        hash_pipeline_shader_key(self.source_identity_hash(source), options)
    }

    /// Compile a shader (or return a cached module).
    ///
    /// For [`ShaderSource::File`], if a custom template was registered under
    /// the same name, its source is rendered instead of the built-in template.
    /// For [`ShaderSource::Inline`], the provided WGSL string is rendered
    /// directly using the supplied [`ShaderTemplateMode`], without consulting
    /// the registered-template map.
    ///
    /// Returns `(module_ref, source_hash)`.
    pub fn get_or_compile(
        &mut self,
        device: &wgpu::Device,
        source: ShaderSource,
        options: &ShaderCompilationOptions,
    ) -> (&wgpu::ShaderModule, u128) {
        let final_wgsl = match source {
            ShaderSource::File(path) => {
                if let Some(custom_src) = self.custom_templates.get(path) {
                    match custom_src.mode {
                        ShaderTemplateMode::Template => ShaderGenerator::generate_custom_shader(
                            path,
                            &custom_src.source,
                            options,
                        ),
                        ShaderTemplateMode::MaterialBody => {
                            ShaderGenerator::generate_material_shader(
                                path,
                                &custom_src.source,
                                options,
                            )
                        }
                    }
                } else {
                    ShaderGenerator::generate_shader(path, options)
                }
            }
            ShaderSource::Inline {
                name,
                source: inline_src,
                mode,
            } => match mode {
                ShaderTemplateMode::Template => {
                    ShaderGenerator::generate_custom_shader(name, inline_src, options)
                }
                ShaderTemplateMode::MaterialBody => {
                    ShaderGenerator::generate_material_shader(name, inline_src, options)
                }
            },
        };

        let hash = xxh3_128(final_wgsl.as_bytes());

        let label = match source {
            ShaderSource::File(path) => path,
            ShaderSource::Inline { name, .. } => name,
        };

        // Study harness: dump every final WGSL Myth compiles (exact source the
        // port must match) when MYTH_STUDY_DUMP_WGSL names a directory.
        #[cfg(feature = "study_post_chain")]
        if let Ok(dir) = std::env::var("MYTH_STUDY_DUMP_WGSL") {
            let file = format!("{}_{:08x}.wgsl", label.replace('/', "__"), hash as u32);
            let _ = std::fs::write(std::path::Path::new(&dir).join(file), &final_wgsl);
        }

        let module = self.module_cache.entry(hash).or_insert_with(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("Shader Module {label}")),
                source: wgpu::ShaderSource::Wgsl(final_wgsl.into()),
            })
        });

        (module, hash)
    }

    /// Returns the number of cached shader modules.
    #[must_use]
    pub fn module_count(&self) -> usize {
        self.module_cache.len()
    }
}
