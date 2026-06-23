//! CPU-vs-GPU training **parity** test for the skip-gram negative-sampling update.
//!
//! Unlike `test_gpu_flag_with_no_adapter_falls_back_to_cpu` (which only checks
//! graceful fallback), this test asserts that the wgpu compute path produces the
//! **same numerical result** as a sequential CPU reference applying the identical
//! update formula.
//!
//! ## Why this is non-trivial
//!
//! The WGSL shader performs Hogwild!-style parallel writes with no atomics. If two
//! training pairs in one batch shared a `syn0` row (same center) or a `syn1neg`
//! row (same context), GPU threads would race and the outcome would be
//! order-dependent — making any parity assertion against a sequential CPU loop
//! ill-defined. We therefore build a **race-free batch**: every pair has a
//! distinct center *and* a distinct context. With no shared rows, the parallel GPU
//! result is independent of execution order and MUST equal the sequential CPU
//! result (modulo f32 rounding in `exp`/FMA).
//!
//! ## Graceful skip
//!
//! When no wgpu adapter is available (headless / CI), `GpuContext::try_new()`
//! returns `None`; the test prints a notice and returns a PASS rather than failing.
//!
//! The whole file is gated behind the `gpu` cargo feature, so without it the file
//! compiles to nothing.

#![cfg(feature = "gpu")]

use mecrab_word2vec::gpu::{GpuContext, GpuTrainer, TrainingPair};
use std::collections::HashSet;

/// Embedding dimension (small, deterministic).
const VECTOR_SIZE: usize = 8;
/// Vocabulary size (small, deterministic).
const VOCAB_SIZE: usize = 16;
/// Fixed learning rate for the single batch under test.
const ALPHA: f32 = 0.025;
/// Absolute tolerance for element-wise parity.
///
/// The per-element update magnitude here is ~`g * weight` ≈ `0.0125 * 0.3` ≈
/// `4e-3`. Pure float noise between the GPU `exp`/FMA and the CPU `exp` is on the
/// order of `1e-7`. A tolerance of `1e-4` sits two-plus orders of magnitude below
/// the update magnitude (so any real formula discrepancy — a wrong learning-rate
/// application, a sign flip, a deferred-vs-immediate center update — produces a
/// diff ≫ `1e-4` and is caught) while staying two-plus orders of magnitude above
/// float noise (so it never flakes). The observed max diff is reported either way.
const TOL: f32 = 1e-4;

/// Build a deterministic weight matrix WITHOUT any RNG (COOLJAPAN policy).
///
/// Row `i`, column `j` is filled by `(i * row_mul + j * col_mul) / denom`. Distinct
/// `(row_mul, col_mul, denom)` triples give `syn0` and `syn1neg` different content
/// so accidental symmetry can't mask a bug.
fn deterministic_fill(row_mul: usize, col_mul: usize, denom: f32) -> Vec<f32> {
    let mut weights = vec![0.0f32; VOCAB_SIZE * VECTOR_SIZE];
    for (i, row) in weights.chunks_mut(VECTOR_SIZE).enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = (i * row_mul + j * col_mul) as f32 / denom;
        }
    }
    weights
}

/// Sequential CPU reference for ONE training pair.
///
/// This mirrors the WGSL shader body in `src/gpu.rs` line-for-line:
///
/// ```wgsl
/// var f : f32 = 0.0;
/// for (var i : u32 = 0u; i < vs; i++) {
///     f += syn0[center_base + i] * syn1neg[ctx_base + i];
/// }
/// f = clamp(f, -6.0, 6.0);
/// let sigmoid_f : f32 = 1.0 / (1.0 + exp(-f));
/// let g         : f32 = (label_f - sigmoid_f) * params.alpha;
/// for (var i : u32 = 0u; i < vs; i++) {
///     let c_grad : f32 = g * syn1neg[ctx_base + i];
///     let t_grad : f32 = g * syn0[center_base + i];
///     syn0[center_base + i]    += c_grad;
///     syn1neg[ctx_base + i]    += t_grad;
/// }
/// ```
///
/// Key fidelity points:
/// * dot product accumulates in the same left-to-right order (`f` starts at `0.0`);
/// * the sigmoid uses `clamp(f, -6, 6)` then `1/(1+exp(-f))` (NOT the branch form
///   in `train_word_pair_hogwild`; the two agree whenever `f ∈ [-6, 6]`, which
///   holds for this data);
/// * both gradients are read from the PRE-update values before either write;
/// * the center row is updated immediately per pair (the shader does NOT defer the
///   center update via a `neu1e` accumulator).
fn cpu_reference_update(syn0: &mut [f32], syn1neg: &mut [f32], pair: &TrainingPair, alpha: f32) {
    let center_base = pair.center as usize * VECTOR_SIZE;
    let ctx_base = pair.context as usize * VECTOR_SIZE;
    let label_f = pair.label as f32;

    // ── dot product (same left-fold order as the shader) ──────────────────────
    let mut f = 0.0f32;
    for (a, b) in syn0[center_base..center_base + VECTOR_SIZE]
        .iter()
        .zip(syn1neg[ctx_base..ctx_base + VECTOR_SIZE].iter())
    {
        f += a * b;
    }

    // ── clamped sigmoid + gradient (matches the shader exactly) ───────────────
    let f = f.clamp(-6.0, 6.0);
    let sigmoid_f = 1.0 / (1.0 + (-f).exp());
    let g = (label_f - sigmoid_f) * alpha;

    // ── Hogwild! weight update: grads from PRE-update values, then write both ──
    let center_row = &mut syn0[center_base..center_base + VECTOR_SIZE];
    let ctx_row = &mut syn1neg[ctx_base..ctx_base + VECTOR_SIZE];
    for (center_val, ctx_val) in center_row.iter_mut().zip(ctx_row.iter_mut()) {
        let c_grad = g * *ctx_val;
        let t_grad = g * *center_val;
        *center_val += c_grad;
        *ctx_val += t_grad;
    }
}

/// Build a race-free batch: distinct center rows AND distinct context rows.
///
/// `center = k`, `context = VOCAB_SIZE/2 + k` for `k ∈ 0..n`, labels alternating
/// `0,1,0,1,…` to exercise both the positive and negative label paths. Centers
/// `{0..n}` are pairwise distinct and contexts `{8..8+n}` are pairwise distinct,
/// and the two sets index *separate* GPU buffers (`syn0` vs `syn1neg`), so no two
/// invocations ever touch the same memory location.
fn build_race_free_batch(n: usize) -> Vec<TrainingPair> {
    (0..n)
        .map(|k| TrainingPair {
            center: k as u32,
            context: (VOCAB_SIZE / 2 + k) as u32,
            label: (k % 2) as u32,
            _pad: 0,
        })
        .collect()
}

#[test]
fn gpu_cpu_skipgram_update_parity() {
    // ── Acquire a GPU adapter, or skip gracefully ─────────────────────────────
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => {
            eprintln!(
                "no GPU adapter; skipping CPU-vs-GPU parity test (this is a PASS in headless/CI)"
            );
            return;
        }
    };
    eprintln!("GPU adapter acquired — running CPU-vs-GPU parity test");

    // ── Deterministic initial weights (no RNG) ────────────────────────────────
    let syn0_init = deterministic_fill(31, 1, 1000.0);
    let syn1neg_init = deterministic_fill(17, 3, 997.0);

    // ── Race-free batch (distinct centers AND distinct contexts) ──────────────
    let pairs = build_race_free_batch(VOCAB_SIZE / 2);

    // Self-check the race-free invariant so future edits can't silently break it.
    let distinct_centers: HashSet<u32> = pairs.iter().map(|p| p.center).collect();
    let distinct_contexts: HashSet<u32> = pairs.iter().map(|p| p.context).collect();
    assert_eq!(
        distinct_centers.len(),
        pairs.len(),
        "batch must have pairwise-distinct centers for a well-defined parity"
    );
    assert_eq!(
        distinct_contexts.len(),
        pairs.len(),
        "batch must have pairwise-distinct contexts for a well-defined parity"
    );

    // ── GPU: one batch, then read weights back ────────────────────────────────
    let mut syn0_gpu = syn0_init.clone();
    let mut syn1neg_gpu = syn1neg_init.clone();
    {
        let trainer = GpuTrainer::new(&ctx, &syn0_gpu, &syn1neg_gpu, VECTOR_SIZE);
        trainer.train_batch(&pairs, ALPHA);
        trainer.read_back(&mut syn0_gpu, &mut syn1neg_gpu);
    }

    // ── CPU: same batch, sequential, identical formula ────────────────────────
    let mut syn0_cpu = syn0_init.clone();
    let mut syn1neg_cpu = syn1neg_init.clone();
    for pair in &pairs {
        cpu_reference_update(&mut syn0_cpu, &mut syn1neg_cpu, pair, ALPHA);
    }

    // Sanity: training must actually have changed something on the GPU side
    // (otherwise a no-op backend would "pass" trivially against an unchanged CPU).
    assert!(
        syn0_gpu
            .iter()
            .zip(syn0_init.iter())
            .any(|(a, b)| (a - b).abs() > 0.0),
        "GPU produced no change in syn0 — backend did not execute the shader"
    );

    // ── Element-wise parity ───────────────────────────────────────────────────
    let max_abs_diff = syn0_gpu
        .iter()
        .zip(syn0_cpu.iter())
        .chain(syn1neg_gpu.iter().zip(syn1neg_cpu.iter()))
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_abs_diff <= TOL,
        "CPU-vs-GPU parity FAILED: max abs diff {max_abs_diff:e} exceeds tolerance {TOL:e}. \
         A diff this large indicates the GPU shader and the CPU reference disagree on the \
         update formula (learning-rate handling, sigmoid/label convention, or center-row \
         update), not mere float rounding."
    );

    eprintln!("CPU-vs-GPU parity OK — max abs diff = {max_abs_diff:e} (tolerance {TOL:e})");
}
