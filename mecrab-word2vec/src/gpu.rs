//! GPU-accelerated Word2Vec training via wgpu.
//!
//! This module is only compiled when the `gpu` feature is enabled.
//! It dispatches skip-gram negative-sampling training to the GPU using a WGSL compute
//! shader and falls back silently to the CPU Hogwild! path when no adapter is present.
//!
//! # Design
//!
//! One WGSL invocation per training pair (center, context, label).
//! The shader mirrors `train_word_pair_hogwild` in `trainer.rs` exactly — dot product,
//! clamped sigmoid, gradient `g = (label - sigmoid) * alpha` — and performs Hogwild!
//! style concurrent writes (race-tolerant, no atomics needed).
//!
//! # WGSL Shader
//!
//! ```wgsl
//! // See WORD2VEC_SHADER below for the full shader source.
//! ```

#[cfg(feature = "gpu")]
use bytemuck;
#[cfg(feature = "gpu")]
use pollster::FutureExt as _;
#[cfg(feature = "gpu")]
use wgpu;

// ─── WGSL source ─────────────────────────────────────────────────────────────

/// WGSL compute shader for skip-gram negative-sampling training.
///
/// One workgroup invocation per training pair:
/// - Computes dot(syn0[center], syn1neg[context])
/// - Applies sigmoid with clamping to ±6
/// - Computes gradient g = (label - sigmoid) * alpha
/// - Writes weight updates directly (Hogwild! — no atomics, race-tolerant)
#[cfg(feature = "gpu")]
const WORD2VEC_SHADER: &str = r#"
// ─── Bindings ─────────────────────────────────────────────────────────────────

struct Params {
    vector_size : u32,
    vocab_size  : u32,  // reserved — not needed in shader; kept for ABI alignment
    alpha       : f32,
    n_pairs     : u32,
}

struct Pair {
    center  : u32,
    context : u32,
    label   : u32,   // 1 = positive sample, 0 = negative sample
    _pad    : u32,
}

@group(0) @binding(0) var<storage, read_write> syn0    : array<f32>;
@group(0) @binding(1) var<storage, read_write> syn1neg : array<f32>;
@group(0) @binding(2) var<storage, read>       pairs   : array<Pair>;
@group(0) @binding(3) var<uniform>             params  : Params;

// ─── Compute entry point ──────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx : u32 = gid.x;
    if idx >= params.n_pairs { return; }

    let pair        : Pair = pairs[idx];
    let vs          : u32  = params.vector_size;
    let center_base : u32  = pair.center  * vs;
    let ctx_base    : u32  = pair.context * vs;
    let label_f     : f32  = f32(pair.label);

    // ── dot product ──────────────────────────────────────────────────────────
    var f : f32 = 0.0;
    for (var i : u32 = 0u; i < vs; i++) {
        f += syn0[center_base + i] * syn1neg[ctx_base + i];
    }

    // ── clamped sigmoid ──────────────────────────────────────────────────────
    f = clamp(f, -6.0, 6.0);
    let sigmoid_f : f32 = 1.0 / (1.0 + exp(-f));
    let g         : f32 = (label_f - sigmoid_f) * params.alpha;

    // ── weight updates (Hogwild! — concurrent writes, no atomics) ────────────
    for (var i : u32 = 0u; i < vs; i++) {
        let c_grad : f32 = g * syn1neg[ctx_base    + i];
        let t_grad : f32 = g * syn0[center_base    + i];
        syn0[center_base + i]    += c_grad;
        syn1neg[ctx_base + i]    += t_grad;
    }
}
"#;

// ─── Public types (only compiled with the gpu feature) ───────────────────────

/// A training pair for GPU dispatch.
///
/// `repr(C)` with explicit padding ensures `bytemuck::Pod` safety
/// (4 × u32 = 16 bytes, no hidden padding).
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TrainingPair {
    /// Remapped (dense) center-word index into `syn0`.
    pub center: u32,
    /// Remapped (dense) context-word index into `syn1neg`.
    pub context: u32,
    /// 1 = positive sample, 0 = negative sample.
    pub label: u32,
    /// Explicit padding to reach 16-byte alignment.
    pub _pad: u32,
}

/// GPU context wrapping a wgpu [`Device`][wgpu::Device] and [`Queue`][wgpu::Queue].
///
/// Constructed once per training run via [`GpuContext::try_new`].
#[cfg(feature = "gpu")]
pub struct GpuContext {
    /// Logical GPU device.
    pub device: wgpu::Device,
    /// Command submission queue.
    pub queue: wgpu::Queue,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Try to acquire a GPU adapter and open a logical device.
    ///
    /// Returns `None` when no suitable adapter is available — the caller should
    /// fall back to CPU training in that case.
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mecrab-word2vec"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .block_on()
            .ok()?;

        Some(Self { device, queue })
    }
}

// ─── Uniform buffer layout (must match WGSL `Params`) ────────────────────────

/// Host-side mirror of the `Params` uniform buffer in the WGSL shader.
///
/// All fields map 1-to-1 to the WGSL struct; `repr(C)` + Pod guarantees byte-safe
/// upload via `bytemuck::bytes_of`.
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    /// Number of dimensions per embedding vector.
    vector_size: u32,
    /// Total vocabulary size (kept for ABI alignment; unused in shader body).
    vocab_size: u32,
    /// Current learning rate α for this batch.
    alpha: f32,
    /// Number of training pairs in this dispatch.
    n_pairs: u32,
}

// ─── GpuTrainer ──────────────────────────────────────────────────────────────

/// GPU-backed skip-gram trainer.
///
/// Holds GPU buffers for `syn0` and `syn1neg` and the compiled compute pipeline.
/// Use [`GpuTrainer::train_batch`] to submit work and [`GpuTrainer::read_back`]
/// to retrieve the trained weights after all epochs are complete.
#[cfg(feature = "gpu")]
pub struct GpuTrainer<'a> {
    /// Borrowed GPU context (device + queue).
    ctx: &'a GpuContext,
    /// GPU storage buffer for input embeddings (syn0). STORAGE | COPY_SRC | COPY_DST.
    syn0_buf: wgpu::Buffer,
    /// GPU storage buffer for output embeddings (syn1neg). STORAGE | COPY_SRC | COPY_DST.
    syn1neg_buf: wgpu::Buffer,
    /// Compiled WGSL compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout matching the shader's `@group(0)` bindings.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Embedding dimension (number of f32 values per word).
    vector_size: u32,
    /// Total vocabulary size — needed for buffer-size bookkeeping.
    vocab_size: u32,
}

#[cfg(feature = "gpu")]
impl<'a> GpuTrainer<'a> {
    /// Allocate GPU buffers and compile the WGSL compute shader.
    ///
    /// `syn0` and `syn1neg` are uploaded immediately; call [`read_back`] at the
    /// end of training to retrieve the updated weights.
    ///
    /// [`read_back`]: GpuTrainer::read_back
    pub fn new(
        ctx: &'a GpuContext,
        syn0: &[f32],
        syn1neg: &[f32],
        vector_size: usize,
    ) -> Self {
        let device = &ctx.device;
        let queue = &ctx.queue;

        let vocab_size = (syn0.len() / vector_size) as u32;

        // ── Compile shader ────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("word2vec_shader"),
            source: wgpu::ShaderSource::Wgsl(WORD2VEC_SHADER.into()),
        });

        // ── Bind group layout ─────────────────────────────────────────────────
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("word2vec_bgl"),
                entries: &[
                    // binding 0 — syn0 (read_write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1 — syn1neg (read_write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2 — pairs (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 3 — params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // ── Pipeline layout + compute pipeline ───────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("word2vec_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("word2vec_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── Allocate + upload syn0 ────────────────────────────────────────────
        let syn0_bytes = bytemuck::cast_slice::<f32, u8>(syn0);
        let syn0_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("syn0"),
            size: syn0_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = syn0_buf.get_mapped_range_mut(..);
            view.copy_from_slice(syn0_bytes);
        }
        syn0_buf.unmap();

        // ── Allocate + upload syn1neg ─────────────────────────────────────────
        let syn1neg_bytes = bytemuck::cast_slice::<f32, u8>(syn1neg);
        let syn1neg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("syn1neg"),
            size: syn1neg_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = syn1neg_buf.get_mapped_range_mut(..);
            view.copy_from_slice(syn1neg_bytes);
        }
        syn1neg_buf.unmap();

        // Upload the initial weight data to the GPU.
        // (The buffers are already mapped-at-creation; unmap() above finalises upload.)
        queue.submit([]);

        Self {
            ctx,
            syn0_buf,
            syn1neg_buf,
            pipeline,
            bind_group_layout,
            vector_size: vector_size as u32,
            vocab_size,
        }
    }

    /// Train one batch of training pairs with learning rate `alpha`.
    ///
    /// Dispatches the WGSL compute shader with one invocation per pair.
    /// Pairs are uploaded to a transient GPU buffer each call; results are
    /// written back into `syn0_buf` / `syn1neg_buf` in place.
    ///
    /// Use chunk sizes ≤ 16 384 to avoid GPU timeout on embedded / mobile adapters.
    pub fn train_batch(&self, pairs: &[TrainingPair], alpha: f32) {
        if pairs.is_empty() {
            return;
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        // ── Upload pairs ──────────────────────────────────────────────────────
        let pairs_bytes = bytemuck::cast_slice::<TrainingPair, u8>(pairs);
        let pairs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pairs"),
            size: pairs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = pairs_buf.get_mapped_range_mut(..);
            view.copy_from_slice(pairs_bytes);
        }
        pairs_buf.unmap();

        // ── Upload params uniform ─────────────────────────────────────────────
        let shader_params = ShaderParams {
            vector_size: self.vector_size,
            vocab_size: self.vocab_size,
            alpha,
            n_pairs: pairs.len() as u32,
        };
        let params_bytes = bytemuck::bytes_of(&shader_params);
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: params_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = params_buf.get_mapped_range_mut(..);
            view.copy_from_slice(params_bytes);
        }
        params_buf.unmap();

        // ── Bind group ────────────────────────────────────────────────────────
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("word2vec_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.syn0_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.syn1neg_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pairs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // ── Encode + dispatch ─────────────────────────────────────────────────
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("word2vec_enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("word2vec_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // workgroup_size = 64; round up
            let workgroups = pairs.len().div_ceil(64) as u32;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit([encoder.finish()]);

        // Block until the GPU finishes — necessary to keep Hogwild! ordering
        // and avoid overwriting the pair buffer before the previous dispatch is done.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Read `syn0` and `syn1neg` back from the GPU into the provided host slices.
    ///
    /// This is a synchronous, blocking operation and should be called once,
    /// after all training batches have been dispatched.
    pub fn read_back(&self, syn0: &mut [f32], syn1neg: &mut [f32]) {
        self.read_buffer_back(&self.syn0_buf, bytemuck::cast_slice_mut(syn0));
        self.read_buffer_back(&self.syn1neg_buf, bytemuck::cast_slice_mut(syn1neg));
    }

    /// Copy a GPU storage buffer into a host byte slice (blocking).
    fn read_buffer_back(&self, src: &wgpu::Buffer, dst: &mut [u8]) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_read"),
            size: dst.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_enc"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, dst.len() as u64);
        queue.submit([encoder.finish()]);

        // Map → copy → unmap
        let (tx, rx) = std::sync::mpsc::channel::<bool>();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res.is_ok());
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let success = rx.recv().unwrap_or(false);
        if success {
            let view = staging.get_mapped_range(..);
            dst.copy_from_slice(&view);
        }
        staging.unmap();
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    /// Verify that requesting a GPU adapter never panics — regardless of whether
    /// a physical GPU is present.  Returns `Some` on GPU machines, `None` on CI.
    #[test]
    fn gpu_context_try_new_does_not_panic() {
        #[cfg(feature = "gpu")]
        {
            let _ = crate::gpu::GpuContext::try_new();
        }
    }
}
