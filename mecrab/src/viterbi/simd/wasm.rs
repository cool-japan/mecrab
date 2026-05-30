//! WebAssembly SIMD128 implementations for cost calculations.
//!
//! Provides 4-way i32 parallelism via WebAssembly SIMD128 instructions.
//! Compile with `RUSTFLAGS="-C target-feature=+simd128"` for wasm32.

// ── WASM SIMD find_min / find_best_predecessor ───────────────────────────────

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm32_simd {
    use core::arch::wasm32::*;

    /// Find minimum value and its index in a slice using WASM SIMD.
    ///
    /// Uses i32x4 SIMD to process 4 elements at a time.
    /// Returns (index, min_value).
    ///
    /// # Safety
    ///
    /// Requires `simd128` target feature.
    /// Compile with `RUSTFLAGS="-C target-feature=+simd128"`.
    #[target_feature(enable = "simd128")]
    pub unsafe fn find_min_wasm(costs: &[i32]) -> (usize, i32) {
        let len = costs.len();
        debug_assert!(len > 0, "find_min_wasm called with empty slice");

        let chunks = len / 4;

        let mut best_idx = 0usize;
        let mut best_val = costs[0];

        // SIMD loop: 4 elements at a time
        for chunk_idx in 0..chunks {
            let base = chunk_idx * 4;
            // SAFETY: base + 3 < len (chunks = len / 4, base = chunk_idx * 4 < chunks * 4 ≤ len)
            let v = v128_load(costs.as_ptr().add(base) as *const v128);

            // Extract all four lanes and compare
            let vals = [
                i32x4_extract_lane::<0>(v),
                i32x4_extract_lane::<1>(v),
                i32x4_extract_lane::<2>(v),
                i32x4_extract_lane::<3>(v),
            ];
            for (i, &val) in vals.iter().enumerate() {
                if val < best_val {
                    best_val = val;
                    best_idx = base + i;
                }
            }
        }

        // Handle remainder (len % 4 != 0)
        for i in (chunks * 4)..len {
            if costs[i] < best_val {
                best_val = costs[i];
                best_idx = i;
            }
        }

        (best_idx, best_val)
    }

    /// Compute predecessor costs: prev_costs[i] + conn_costs[i] (widened from i16).
    ///
    /// Processes 4 elements at a time using WASM SIMD i32x4.
    ///
    /// # Safety
    ///
    /// Requires `simd128` target feature.
    #[target_feature(enable = "simd128")]
    pub unsafe fn compute_total_costs_wasm(
        prev_costs: &[i32],
        conn_costs: &[i16],
        node_wcost: i32,
    ) -> Vec<i32> {
        let len = prev_costs.len().min(conn_costs.len());
        let mut result = vec![i32::MAX; len];
        let wcost_v = i32x4_splat(node_wcost);

        let chunks = len / 4;
        for chunk_idx in 0..chunks {
            let base = chunk_idx * 4;

            // SAFETY: base + 3 < len, pointers valid
            let prev_v = v128_load(prev_costs.as_ptr().add(base) as *const v128);

            // Widen 4 × i16 → 4 × i32
            let conn_i16_ptr = conn_costs.as_ptr().add(base);
            let c0 = *conn_i16_ptr as i32;
            let c1 = *conn_i16_ptr.add(1) as i32;
            let c2 = *conn_i16_ptr.add(2) as i32;
            let c3 = *conn_i16_ptr.add(3) as i32;
            let conn_v = i32x4(c0, c1, c2, c3);

            // total = prev + conn + wcost
            let total_v = i32x4_add(i32x4_add(prev_v, conn_v), wcost_v);

            // Store 4 results
            v128_store(result.as_mut_ptr().add(base) as *mut v128, total_v);
        }

        // Handle remainder with overflow-safe arithmetic
        for i in (chunks * 4)..len {
            let prev = prev_costs[i];
            let conn = conn_costs[i] as i32;
            if prev != i32::MAX && conn != i32::MAX {
                result[i] = prev.saturating_add(conn).saturating_add(node_wcost);
            }
        }

        result
    }

    /// Public entry point: find minimum cost and its index.
    pub fn find_min(costs: &[i32]) -> Option<(usize, i32)> {
        if costs.is_empty() {
            return None;
        }
        // SAFETY: simd128 feature is required at module level via cfg guard.
        Some(unsafe { find_min_wasm(costs) })
    }

    /// Public entry point: find best predecessor (argmin of pairwise sum).
    pub fn find_best_predecessor(prev_costs: &[i32], conn_costs: &[i16]) -> Option<(usize, i32)> {
        let len = prev_costs.len().min(conn_costs.len());
        if len == 0 {
            return None;
        }
        // SAFETY: simd128 feature is required at module level via cfg guard.
        let totals = unsafe { compute_total_costs_wasm(prev_costs, conn_costs, 0) };
        totals
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != i32::MAX)
            .min_by_key(|&(_, &v)| v)
            .map(|(i, &v)| (i, v))
    }
}

// ── WASM i64x2 SIMD argmin ───────────────────────────────────────────────────

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm_i64 {
    use core::arch::wasm32::*;

    /// Argmin of prev[i] + conn[i] + wcost over the slice.
    ///
    /// Uses i64x2 SIMD (2 lanes per iteration). Returns `None` if no candidate
    /// strictly improves on `best_so_far`.
    ///
    /// # Safety
    ///
    /// Requires `simd128` target feature, enforced by the enclosing `cfg` guard.
    #[target_feature(enable = "simd128")]
    pub unsafe fn batch_min_argmin_i64_wasm(
        prev: &[i64],
        conn: &[i32],
        wcost: i64,
        best_so_far: i64,
    ) -> Option<(usize, i64)> {
        let len = prev.len().min(conn.len());
        if len == 0 {
            return None;
        }

        // Broadcast the word cost into both SIMD lanes.
        let wcost_v = i64x2_splat(wcost);

        let mut best_val = best_so_far;
        // `usize::MAX` is our "no-improvement" sentinel.
        let mut best_idx = usize::MAX;

        let chunks = len / 2;

        for chunk in 0..chunks {
            let base = chunk * 2;
            // SAFETY: chunk < chunks = len/2, so base = chunk*2 and base+1 are
            // both strictly less than len, which is within bounds for both slices.
            let p0 = *prev.get_unchecked(base);
            let p1 = *prev.get_unchecked(base + 1);
            let prev_v = i64x2(p0, p1);

            // Widen i32 → i64 before constructing the SIMD vector.
            let c0 = *conn.get_unchecked(base) as i64;
            let c1 = *conn.get_unchecked(base + 1) as i64;
            let conn_v = i64x2(c0, c1);

            // total = prev + conn + wcost  (2 lanes in parallel).
            let total_v = i64x2_add(i64x2_add(prev_v, conn_v), wcost_v);

            // WASM has no horizontal i64 min instruction, so we extract each lane
            // and compare scalarly — the vectorised add above still saves work.
            let t0 = i64x2_extract_lane::<0>(total_v);
            let t1 = i64x2_extract_lane::<1>(total_v);

            if t0 < best_val {
                best_val = t0;
                best_idx = base;
            }
            if t1 < best_val {
                best_val = t1;
                best_idx = base + 1;
            }
        }

        // Scalar tail for the remainder element (if len is odd).
        for i in (chunks * 2)..len {
            // SAFETY: i < len, both slices have length ≥ len.
            let total = prev.get_unchecked(i) + *conn.get_unchecked(i) as i64 + wcost;
            if total < best_val {
                best_val = total;
                best_idx = i;
            }
        }

        if best_idx == usize::MAX {
            None
        } else {
            Some((best_idx, best_val))
        }
    }
}

// ── WASM SIMD gather+widen ────────────────────────────────────────────────────

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm_gather {
    use core::arch::wasm32::{
        i32x4_extend_high_i16x8, i32x4_extend_low_i16x8, v128_load, v128_store,
    };

    /// Gather up to 16 connection costs from `row_data` at `right_ids[i]`,
    /// widen i16→i32, and store into `out[i]`.
    ///
    /// Uses WASM `i32x4_extend_low_i16x8` / `i32x4_extend_high_i16x8` to process
    /// 4 lanes at a time after gathering.
    ///
    /// # Safety
    ///
    /// Requires `simd128` target feature. `out` must hold ≥ `count` elements.
    #[target_feature(enable = "simd128")]
    pub unsafe fn gather_widen(row_data: &[i16], right_ids: &[u16], out: &mut [i32]) -> usize {
        let count = right_ids.len().min(out.len()).min(16);
        if count == 0 {
            return 0;
        }
        let row_len = row_data.len();

        let mut gathered: [i16; 16] = [i16::MAX; 16];
        for (i, &rid) in right_ids[..count].iter().enumerate() {
            if (rid as usize) < row_len {
                gathered[i] = unsafe { *row_data.get_unchecked(rid as usize) };
            }
        }

        // WASM: load 8 × i16 → extend low 4 and high 4 to i32x4
        // We have up to 16 values; process two 8-element chunks.
        let chunks8 = count / 8;
        let ptr_g = gathered.as_ptr();

        for chunk in 0..chunks8 {
            let base = chunk * 8;
            // SAFETY: base + 7 < 16.
            let v8 = unsafe { v128_load(ptr_g.add(base) as *const core::arch::wasm32::v128) };
            let lo = i32x4_extend_low_i16x8(v8);
            let hi = i32x4_extend_high_i16x8(v8);
            // SAFETY: out[base..base+4] and out[base+4..base+8] within bounds.
            unsafe {
                v128_store(
                    out.as_mut_ptr().add(base) as *mut core::arch::wasm32::v128,
                    lo,
                );
                if base + 8 <= count {
                    v128_store(
                        out.as_mut_ptr().add(base + 4) as *mut core::arch::wasm32::v128,
                        hi,
                    );
                } else {
                    // Partial upper half: extract lane by lane.
                    use core::arch::wasm32::i32x4_extract_lane;
                    let hi_vals = [
                        i32x4_extract_lane::<0>(hi),
                        i32x4_extract_lane::<1>(hi),
                        i32x4_extract_lane::<2>(hi),
                        i32x4_extract_lane::<3>(hi),
                    ];
                    let start = base + 4;
                    for j in 0..(count.saturating_sub(start)) {
                        out[start + j] = hi_vals[j];
                    }
                }
            }
        }

        // Scalar remainder
        let covered = chunks8 * 8;
        for i in covered..count {
            out[i] = gathered[i] as i32;
        }

        count
    }
}
