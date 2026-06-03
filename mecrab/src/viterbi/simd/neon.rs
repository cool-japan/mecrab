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

// ── ARM NEON i64 argmin kernel ────────────────────────────────────────────────

/// ARM NEON batch minimum argmin over i64 costs (2 × i64 per register).
#[cfg(target_arch = "aarch64")]
pub mod neon_i64 {
    use core::arch::aarch64::{
        int32x2_t, int64x2_t, vaddq_s64, vbslq_s64, vcgtq_s64, vgetq_lane_s64, vld1_s32, vmovl_s32,
    };

    /// NEON-accelerated `batch_min_argmin_i64`.
    ///
    /// Computes `argmin_i(prev[i] + conn[i]) + wcost`, returning
    /// `Some((index, min_total))` when a strictly-better candidate than
    /// `best_so_far` is found, or `None` otherwise.
    ///
    /// Uses 2-lane `int64x2_t` registers.  NEON has no integer 64-bit min
    /// intrinsic, so we use `vcgtq_s64` + `vbslq_s64` to compute
    /// `min(a, b)` branchlessly.
    ///
    /// # Safety
    ///
    /// NEON is mandatory on all ARMv8 / Apple Silicon cores; no runtime check
    /// needed.  Pointer arithmetic stays within `len`.
    #[target_feature(enable = "neon")]
    pub unsafe fn batch_min_argmin_i64_neon(
        prev: &[i64],
        conn: &[i32],
        wcost: i64,
        best_so_far: i64,
    ) -> Option<(usize, i64)> {
        let len = prev.len().min(conn.len());
        if len == 0 {
            return None;
        }

        let chunks2 = len / 2;
        let ptr_p = prev.as_ptr();
        let ptr_c = conn.as_ptr();

        // Track the SIMD-prefix minimum as a 2-lane i64 vector.
        let simd_min: i64 = if chunks2 > 0 {
            // SAFETY: pointers valid for ≥ 2 elements (chunks2 ≥ 1).
            // vld1_s32 loads 2 × i32; vmovl_s32 sign-extends to 2 × i64.
            let p0: int64x2_t = unsafe { core::arch::aarch64::vld1q_s64(ptr_p) };
            let c_narrow: int32x2_t = unsafe { vld1_s32(ptr_c) };
            let c0: int64x2_t = vmovl_s32(c_narrow);
            let mut lane_min: int64x2_t = vaddq_s64(p0, c0);

            for chunk in 1..chunks2 {
                // SAFETY: chunk < chunks2 ≤ len/2, offsets in-bounds.
                let pv: int64x2_t = unsafe { core::arch::aarch64::vld1q_s64(ptr_p.add(chunk * 2)) };
                let cn: int32x2_t = unsafe { vld1_s32(ptr_c.add(chunk * 2)) };
                let cv: int64x2_t = vmovl_s32(cn);
                let sv: int64x2_t = vaddq_s64(pv, cv);
                // NEON has no vminq_s64; use compare + bitwise-select for min.
                // vcgtq_s64 returns a mask with all-ones in lanes where lane_min > sv.
                let gt_mask = vcgtq_s64(lane_min, sv);
                // vbslq_s64(mask, a, b) selects a where mask is all-ones, b elsewhere.
                // When lane_min > sv, we want sv (smaller), so: select sv where gt.
                lane_min = vbslq_s64(
                    core::mem::transmute::<int64x2_t, core::arch::aarch64::uint64x2_t>(gt_mask),
                    sv,
                    lane_min,
                );
            }

            // Horizontal reduce: min of lane 0 and lane 1.
            let l0 = vgetq_lane_s64(lane_min, 0);
            let l1 = vgetq_lane_s64(lane_min, 1);
            l0.min(l1)
        } else {
            i64::MAX
        };

        let covered = chunks2 * 2;
        let mut best_idx = 0usize;
        let mut best_total = best_so_far;
        let mut found = false;

        // Scan remainder (may beat SIMD prefix minimum).
        for i in covered..len {
            let total = prev[i] + conn[i] as i64 + wcost;
            if total < best_total {
                best_total = total;
                best_idx = i;
                found = true;
            }
        }

        // Check whether SIMD prefix contains something better.
        let simd_with_wcost = simd_min.saturating_add(wcost);
        if simd_with_wcost < best_total {
            for i in 0..covered {
                let total = prev[i] + conn[i] as i64 + wcost;
                if total == simd_with_wcost {
                    best_total = simd_with_wcost;
                    best_idx = i;
                    found = true;
                    break;
                }
            }
        }

        if found {
            Some((best_idx, best_total))
        } else {
            None
        }
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
