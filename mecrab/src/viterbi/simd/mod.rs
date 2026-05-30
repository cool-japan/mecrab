//! SIMD-accelerated cost calculations for Viterbi algorithm
//!
//! On aarch64 (Apple Silicon / ARMv8), uses ARM NEON intrinsics for
//! vectorized i32 minimum finding and predecessor selection.
//! On x86_64, uses AVX2 (256-bit, 8 × i32) with SSE4.1 fallback (128-bit, 4 × i32).
//! On wasm32 with simd128, uses WebAssembly SIMD.
//! On other platforms, provides auto-vectorization-friendly scalar fallbacks.
//!
//! ## Performance
//!
//! - **x86_64 AVX2**: 8-way SIMD parallelism for i32 cost calculations (Haswell+)
//! - **x86_64 SSE4.1**: 4-way SIMD parallelism (Sandy Bridge+)
//! - **aarch64**: 4-way NEON SIMD parallelism for i32 cost calculations
//! - **wasm32+simd128**: 4-way SIMD parallelism via WebAssembly SIMD
//! - **Other**: Optimized scalar implementations with auto-vectorization hints
//!
//! ## Module structure
//!
//! - [`neon`]: ARM NEON implementations (aarch64 only)
//! - [`x86`]: x86_64 AVX2/SSE4.1 implementations
//! - [`wasm`]: WebAssembly SIMD128 implementations
//! - [`scalar`]: Scalar fallback implementations (all platforms)

pub mod neon;
pub mod scalar;
pub mod wasm;
pub mod x86;

// Re-export platform-specific submodules under the names expected by the
// public dispatch functions and tests.
#[cfg(target_arch = "aarch64")]
use neon::neon_gather;
#[cfg(target_arch = "aarch64")]
pub use neon::neon_impl;
#[cfg(target_arch = "aarch64")]
use neon::neon_i64;

#[cfg(target_arch = "x86_64")]
use x86::x86_gather;
#[cfg(target_arch = "x86_64")]
pub use x86::x86_impl;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use wasm::wasm_gather;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use wasm::wasm32_simd;

pub use scalar as scalar_impl;

/// SIMD batch size for cost calculations (4 i32 values for NEON/SSE 128-bit registers;
/// AVX2 processes 8 per iteration internally)
pub const BATCH_SIZE: usize = 4;

// ── Public API ───────────────────────────────────────────────────────────────

/// Batch-fetch up to 16 connection costs for a given row and a slice of right_ids,
/// storing widened i16→i32 results into `out`.
///
/// This is the hot-path gather function for the Viterbi forward pass:
/// given the pre-fetched row for a target node's `left_id` (from
/// `ConnectionMatrix::row_for_left_id`), and a slice of predecessor `right_id`s,
/// fill `out[i] = row_data[right_ids[i]]` (widened to i32), or `i16::MAX as i32`
/// for any out-of-bounds `right_id`.
///
/// ## Platform dispatch
///
/// - **aarch64**: Manual gather + NEON `vmovl_s16` to widen 4 lanes at a time.
/// - **x86_64 AVX2**: Manual gather + `_mm256_cvtepi16_epi32` (8-lane widen).
/// - **x86_64 SSE4.1**: Manual gather + `_mm_cvtepi16_epi32` (4-lane widen).
/// - **wasm32+simd128**: Manual gather + `i32x4_extend_low/high_i16x8`.
/// - **Scalar**: Simple indexed loop.
///
/// ## Arguments
///
/// - `row_data`: contiguous row slice from `ConnectionMatrix::row_for_left_id`
///   (length = `rsize`).
/// - `right_ids`: predecessor right_ids to look up (up to 16 elements).
/// - `out`: output buffer, must be at least `right_ids.len()` elements long.
///
/// ## Returns
///
/// The number of costs written into `out` (≤ `right_ids.len().min(out.len()).min(16)`).
#[inline]
pub fn batch_connection_costs(row_data: &[i16], right_ids: &[u16], out: &mut [i32]) -> usize {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on all ARMv8/Apple Silicon.
    return unsafe { neon_gather::gather_widen(row_data, right_ids, out) };

    #[cfg(target_arch = "x86_64")]
    return x86_gather::gather_widen(row_data, right_ids, out);

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 compile-time cfg.
    return unsafe { wasm_gather::gather_widen(row_data, right_ids, out) };

    // Scalar fallback (other platforms or no SIMD feature)
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128"),
    )))]
    {
        let count = right_ids.len().min(out.len()).min(16);
        let row_len = row_data.len();
        for i in 0..count {
            let rid = right_ids[i] as usize;
            out[i] = if rid < row_len {
                // SAFETY: bounds checked above.
                unsafe { *row_data.get_unchecked(rid) as i32 }
            } else {
                i16::MAX as i32
            };
        }
        count
    }
}

/// SIMD-accelerated batch addition of costs.
///
/// Adds `addition` to each element using saturating arithmetic.
/// LLVM auto-vectorizes this loop on all platforms.
#[inline]
pub fn batch_add_costs(costs: &mut [i32], addition: i32) {
    for cost in costs.iter_mut() {
        *cost = cost.saturating_add(addition);
    }
}

/// Find minimum cost and its index.
///
/// Dispatch priority:
/// 1. WASM SIMD (`wasm32` + `simd128` target feature)
/// 2. ARM NEON (`aarch64`)
/// 3. x86_64 AVX2 / SSE4.1 (compile-time selected)
/// 4. Scalar fallback (all other targets)
#[inline]
pub fn find_min(costs: &[i32]) -> Option<(usize, i32)> {
    if costs.is_empty() {
        return None;
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm32_simd::find_min(costs);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on all ARMv8 cores including Apple Silicon.
        Some(unsafe { neon_impl::find_min_neon(costs) })
    }

    #[cfg(target_arch = "x86_64")]
    {
        x86_impl::find_min(costs)
    }

    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64",
        target_arch = "x86_64"
    )))]
    {
        scalar_impl::find_min(costs)
    }
}

/// Find best predecessor: argmin(prev_costs[i] + connection_costs[i]).
///
/// Dispatch priority:
/// 1. WASM SIMD (`wasm32` + `simd128` target feature)
/// 2. ARM NEON (`aarch64`)
/// 3. x86_64 AVX2 / SSE4.1 (compile-time selected)
/// 4. Scalar fallback (all other targets)
#[inline]
pub fn find_best_predecessor(prev_costs: &[i32], connection_costs: &[i16]) -> Option<(usize, i32)> {
    if prev_costs.is_empty() || connection_costs.is_empty() {
        return None;
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm32_simd::find_best_predecessor(prev_costs, connection_costs);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on all ARMv8 cores including Apple Silicon.
        unsafe { neon_impl::find_best_predecessor_neon(prev_costs, connection_costs) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        x86_impl::find_best_predecessor(prev_costs, connection_costs)
    }

    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64",
        target_arch = "x86_64"
    )))]
    {
        scalar_impl::find_best_predecessor(prev_costs, connection_costs)
    }
}

/// SIMD-accelerated minimum argmin: `argmin_i(prev[i] + conn[i]) + wcost`.
///
/// Returns `Some((index, min_total))` where `min_total` includes `wcost`, but
/// **only** when `min_total < best_so_far`.  Returns `None` if the slice is
/// empty or no candidate beats `best_so_far`.
///
/// ## Platform dispatch
///
/// - **aarch64** (NEON mandatory): 2 × i64 lanes via `int64x2_t`.
/// - **x86_64**: AVX2 (4 × i64) → SSE4.1 (2 × i64) → scalar, with runtime
///   `is_x86_feature_detected!` for baseline builds.
/// - **wasm32 + simd128**: scalar (no i64 wasm SIMD path yet).
/// - **other**: scalar reference implementation.
#[inline]
pub fn batch_min_argmin_i64(
    prev: &[i64],
    conn: &[i32],
    wcost: i64,
    best_so_far: i64,
) -> Option<(usize, i64)> {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on all ARMv8 / Apple Silicon; no runtime check needed.
        return unsafe {
            neon_i64::batch_min_argmin_i64_neon(prev, conn, wcost, best_so_far)
        };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            x86_impl::batch_min_argmin_i64(prev, conn, wcost, best_so_far)
        }

        // Scalar fallback: wasm32+simd128, other architectures.
        #[cfg(not(target_arch = "x86_64"))]
        scalar_impl::batch_min_argmin_i64(prev, conn, wcost, best_so_far)
    }
}

/// Batch process predecessor costs: result[i] = prev_costs[i] + connection_costs[i].
pub fn predecessor_batch_simd(prev_costs: &[i32], connection_costs: &[i16]) -> Vec<i32> {
    let len = prev_costs.len().min(connection_costs.len());
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        result.push(prev_costs[i].saturating_add(connection_costs[i] as i32));
    }

    result
}

/// Batch process predecessor costs using a 2-D connection matrix.
pub fn predecessor_batch(
    prev_costs: &[i32],
    connection_matrix: &[i16],
    right_id: u16,
    matrix_width: usize,
) -> Vec<i32> {
    prev_costs
        .iter()
        .enumerate()
        .map(|(i, &cost)| {
            let conn_cost = connection_matrix
                .get(i * matrix_width + right_id as usize)
                .copied()
                .unwrap_or(0) as i32;
            cost.saturating_add(conn_cost)
        })
        .collect()
}

/// SIMD statistics for diagnostics / debugging.
#[derive(Debug, Default)]
pub struct SimdStats {
    /// Whether ARM NEON SIMD is available
    pub neon: bool,
    /// Whether WebAssembly SIMD128 is available
    pub simd128: bool,
    /// Whether x86_64 AVX2 (256-bit) is available
    pub avx2: bool,
    /// Whether x86_64 SSE4.1 (128-bit) is available
    pub sse41: bool,
    /// SIMD lane width (number of elements processed in parallel)
    pub batch_size: usize,
    /// Kept for backward compatibility; reflects whether any SIMD path is active
    pub portable_simd: bool,
}

impl SimdStats {
    /// Detect SIMD capabilities at compile time (and runtime for x86_64).
    pub fn detect() -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let simd128 = true;
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        let simd128 = false;

        #[cfg(target_arch = "x86_64")]
        let avx2 = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;

        #[cfg(target_arch = "x86_64")]
        let sse41 = is_x86_feature_detected!("sse4.1");
        #[cfg(not(target_arch = "x86_64"))]
        let sse41 = false;

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                neon: true,
                simd128,
                avx2,
                sse41,
                batch_size: BATCH_SIZE,
                portable_simd: true,
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                neon: false,
                simd128,
                avx2,
                sse41,
                batch_size: BATCH_SIZE,
                portable_simd: avx2 || sse41,
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self {
                neon: false,
                simd128,
                avx2,
                sse41,
                batch_size: BATCH_SIZE,
                portable_simd: simd128,
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
