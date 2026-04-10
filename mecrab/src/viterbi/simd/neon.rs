//! ARM NEON SIMD implementations for cost calculations.
//!
//! NEON is mandatory on all ARMv8 / Apple Silicon cores; no runtime check needed.
//! Provides 4-way i32 parallelism via 128-bit NEON registers.

// ── ARM NEON find_min / find_best_predecessor ────────────────────────────────

/// ARM NEON intrinsic implementations of find_min and find_best_predecessor.
#[cfg(target_arch = "aarch64")]
pub mod neon_impl {
    use core::arch::aarch64::{
        int16x4_t, int32x4_t, vaddq_s32, vld1_s16, vld1q_s32, vminq_s32, vminvq_s32, vmovl_s16,
    };

    /// Find the minimum value and its index using ARM NEON intrinsics.
    ///
    /// Processes 4 × i32 per iteration using 128-bit NEON registers.
    /// Remainder elements (len % 4) are handled with a scalar loop.
    ///
    /// # Safety
    ///
    /// NEON is mandatory on all ARMv8 / Apple Silicon cores; no runtime check
    /// needed.  `vld1q_s32` requires 4-byte-aligned data, which is guaranteed
    /// for `Vec<i32>` heap allocations (alignment ≥ 4 bytes on all Rust
    /// allocators).
    #[target_feature(enable = "neon")]
    pub unsafe fn find_min_neon(costs: &[i32]) -> (usize, i32) {
        let len = costs.len();
        debug_assert!(len > 0, "find_min_neon called with empty slice");

        let chunks = len / 4;

        if chunks == 0 {
            return scalar_min_forward(costs);
        }

        let ptr = costs.as_ptr();

        // Initialise with the first chunk so we never compare against i32::MAX.
        // SAFETY: ptr is valid for at least 4 × i32 (chunks >= 1).
        let mut min_vec: int32x4_t = unsafe { vld1q_s32(ptr) };

        for chunk in 1..chunks {
            // SAFETY: ptr + chunk*4 is within bounds (chunk < chunks ≤ len/4).
            let v = unsafe { vld1q_s32(ptr.add(chunk * 4)) };
            min_vec = vminq_s32(min_vec, v);
        }

        // Horizontal reduce: scalar minimum across the NEON-covered prefix.
        let neon_min = vminvq_s32(min_vec);

        // Also scan the remainder (len % 4 != 0) with scalar.
        let covered = chunks * 4;
        let (rem_idx, rem_val) = if covered < len {
            scalar_min_from(&costs[covered..])
                .map(|(i, v)| (i + covered, v))
                .unwrap_or((0, i32::MAX))
        } else {
            (0, i32::MAX)
        };

        // Overall minimum across NEON prefix and scalar remainder.
        let global_min = neon_min.min(rem_val);

        if rem_val < neon_min {
            (rem_idx, rem_val)
        } else {
            // Find the first index in the NEON-covered prefix equal to global_min.
            find_index_of(&costs[..covered], global_min)
        }
    }

    /// Find the best predecessor: argmin(prev_costs[i] + conn_costs[i]).
    ///
    /// `conn_costs` is i16; widens to i32 via `vmovl_s16` before addition.
    ///
    /// # Safety
    ///
    /// Same NEON availability guarantee as `find_min_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn find_best_predecessor_neon(
        prev_costs: &[i32],
        conn_costs: &[i16],
    ) -> Option<(usize, i32)> {
        let len = prev_costs.len().min(conn_costs.len());
        if len == 0 {
            return None;
        }

        let chunks = len / 4;
        let ptr_p = prev_costs.as_ptr();
        let ptr_c = conn_costs.as_ptr();

        // NEON: reduce the SIMD-covered prefix to its minimum sum value.
        // This gives us an upper bound — any prefix element above `prefix_min`
        // cannot be the global minimum, so the final scalar scan can use it
        // as an early-exit guard for the prefix portion.
        let prefix_min: i32 = if chunks > 0 {
            // SAFETY: pointers valid for ≥ 4 elements (chunks ≥ 1).
            let prev_v: int32x4_t = unsafe { vld1q_s32(ptr_p) };
            let conn_narrow: int16x4_t = unsafe { vld1_s16(ptr_c) };
            let conn_v: int32x4_t = vmovl_s16(conn_narrow);
            let mut sum_min: int32x4_t = vaddq_s32(prev_v, conn_v);

            for chunk in 1..chunks {
                // SAFETY: chunk < chunks ≤ len/4, offsets in-bounds.
                let pv: int32x4_t = unsafe { vld1q_s32(ptr_p.add(chunk * 4)) };
                let cn: int16x4_t = unsafe { vld1_s16(ptr_c.add(chunk * 4)) };
                let cv: int32x4_t = vmovl_s16(cn);
                let sv: int32x4_t = vaddq_s32(pv, cv);
                sum_min = vminq_s32(sum_min, sv);
            }

            vminvq_s32(sum_min)
        } else {
            i32::MAX
        };

        // Scalar scan: find first occurrence of prefix_min in the prefix,
        // or run a full scan if the remainder contains a smaller value.
        //
        // First check the remainder to see if it beats the prefix minimum.
        let covered = chunks * 4;
        let mut best_idx = 0usize;
        let mut best_cost = prefix_min;

        // Scan remainder (may beat prefix_min)
        for i in covered..len {
            let sum = prev_costs[i].saturating_add(conn_costs[i] as i32);
            if sum < best_cost {
                best_cost = sum;
                best_idx = i;
            }
        }

        // If remainder did NOT beat prefix_min, locate the first index in the
        // prefix whose sum equals prefix_min.
        if best_cost == prefix_min {
            // We still need the index; do a forward scan over the prefix.
            for i in 0..covered {
                let sum = prev_costs[i].saturating_add(conn_costs[i] as i32);
                if sum == prefix_min {
                    best_idx = i;
                    break;
                }
            }
        }
        // Otherwise the remainder winner is already in best_idx/best_cost.

        Some((best_idx, best_cost))
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Scalar forward min scan, returning (index, min_value).
    #[inline]
    fn scalar_min_forward(costs: &[i32]) -> (usize, i32) {
        let mut best_idx = 0usize;
        let mut best_val = costs[0];
        for (i, &c) in costs.iter().enumerate().skip(1) {
            if c < best_val {
                best_val = c;
                best_idx = i;
            }
        }
        (best_idx, best_val)
    }

    /// Return Some((index, min_value)) for a non-empty slice, None if empty.
    #[inline]
    fn scalar_min_from(costs: &[i32]) -> Option<(usize, i32)> {
        if costs.is_empty() {
            None
        } else {
            Some(scalar_min_forward(costs))
        }
    }

    /// Find the first index in `costs` whose value equals `target`.
    /// Falls back to a full min scan if not found (should not happen).
    #[inline]
    fn find_index_of(costs: &[i32], target: i32) -> (usize, i32) {
        for (i, &c) in costs.iter().enumerate() {
            if c == target {
                return (i, target);
            }
        }
        scalar_min_forward(costs)
    }
}

// ── ARM NEON gather+widen ─────────────────────────────────────────────────────

/// ARM NEON gather+widen for batch connection cost lookups.
#[cfg(target_arch = "aarch64")]
pub mod neon_gather {
    use core::arch::aarch64::{int16x4_t, vld1_s16, vmovl_s16, vst1q_s32};

    /// Gather up to 8 connection costs from `row_data` at `right_ids[i]`,
    /// widen i16→i32, and store into `out[i]`.
    ///
    /// Processes 4 lanes per iteration using `vmovl_s16`.
    /// Returns the number of elements written.
    ///
    /// # Safety
    ///
    /// NEON is mandatory on all ARMv8 / Apple Silicon cores.
    /// `out` must have capacity ≥ `count` (≤ 16).
    #[target_feature(enable = "neon")]
    pub unsafe fn gather_widen(row_data: &[i16], right_ids: &[u16], out: &mut [i32]) -> usize {
        let count = right_ids.len().min(out.len()).min(16);
        if count == 0 {
            return 0;
        }

        let row_len = row_data.len();

        // Gather individual i16 values with bounds-checked indexing.
        // Build a small stack-allocated scratch array of gathered i16 values,
        // then process in 4-lane NEON chunks.
        let mut gathered: [i16; 16] = [i16::MAX; 16];
        for (i, &rid) in right_ids[..count].iter().enumerate() {
            gathered[i] = if (rid as usize) < row_len {
                // SAFETY: bounds checked above.
                unsafe { *row_data.get_unchecked(rid as usize) }
            } else {
                i16::MAX
            };
        }

        let chunks = count / 4;
        let ptr_g = gathered.as_ptr();

        for chunk in 0..chunks {
            let base = chunk * 4;
            // SAFETY: base + 3 < count ≤ 16, gathered has length 16.
            let lane4: int16x4_t = unsafe { vld1_s16(ptr_g.add(base)) };
            let wide = vmovl_s16(lane4);
            // SAFETY: out[base..base+4] is within bounds (chunk < chunks ≤ count/4).
            unsafe { vst1q_s32(out.as_mut_ptr().add(base), wide) };
        }

        // Scalar remainder
        let covered = chunks * 4;
        for i in covered..count {
            out[i] = gathered[i] as i32;
        }

        count
    }
}
