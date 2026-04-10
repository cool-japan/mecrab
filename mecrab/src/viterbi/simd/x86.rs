//! x86_64 AVX2/SSE4.1 SIMD implementations for cost calculations.
//!
//! Provides 8-way (AVX2) or 4-way (SSE4.1) i32 parallelism.
//! Path is selected at compile time based on available target features.

// ── x86_64 AVX2/SSE4.1 find_min / find_best_predecessor ─────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_impl {
    // ── Internal helpers shared by both AVX2 and SSE4.1 paths ────────────────

    /// Scalar forward min scan returning (index, min_value).
    /// Used when the slice is too small for SIMD or as a remainder handler.
    #[inline]
    fn scalar_min_forward(costs: &[i32]) -> (usize, i32) {
        debug_assert!(
            !costs.is_empty(),
            "scalar_min_forward called with empty slice"
        );
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

    /// Scan `costs` linearly for the first element equal to `target`.
    /// Falls back to a full min scan on not-found (indicates logic error).
    #[inline]
    fn find_index_of(costs: &[i32], target: i32) -> (usize, i32) {
        for (i, &c) in costs.iter().enumerate() {
            if c == target {
                return (i, target);
            }
        }
        // Should never be reached; scalar_min_forward is a safe fallback.
        scalar_min_forward(costs)
    }

    // ── AVX2 path (256-bit = 8 × i32 per register) ───────────────────────────

    /// Find minimum value and its index using AVX2 (8 × i32 per iteration).
    ///
    /// The algorithm:
    ///   1. Reduce 8-lane chunks to a single minimum vector via `_mm256_min_epi32`.
    ///   2. Horizontal-reduce the 256-bit register: split hi/lo 128, min pairwise,
    ///      then two shuffle+min passes to collapse 4 lanes → 2 → 1.
    ///   3. Handle the remainder (len % 8) with a scalar scan.
    ///   4. Return the index of the global minimum (first occurrence).
    ///
    /// # Safety
    ///
    /// Requires `avx2` target feature. Enabled via `#[target_feature(enable = "avx2")]`.
    /// Pointer arithmetic stays within bounds because `chunk < chunks = len / 8`.
    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn find_min_avx2(costs: &[i32]) -> (usize, i32) {
        use core::arch::x86_64::{
            _mm_extract_epi32, _mm_min_epi32, _mm_shuffle_epi32, _mm256_castsi256_si128,
            _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_min_epi32,
        };

        let len = costs.len();
        let chunks = len / 8;

        if chunks == 0 {
            return scalar_min_forward(costs);
        }

        let ptr = costs.as_ptr();

        // SAFETY: ptr is valid for at least 8 × i32 = 32 bytes (chunks ≥ 1).
        let mut min_vec = _mm256_loadu_si256(ptr as *const core::arch::x86_64::__m256i);

        for chunk in 1..chunks {
            // SAFETY: chunk < chunks ≤ len/8, so ptr.add(chunk*8) is in bounds.
            let v = _mm256_loadu_si256(ptr.add(chunk * 8) as *const core::arch::x86_64::__m256i);
            min_vec = _mm256_min_epi32(min_vec, v);
        }

        // Horizontal reduce: 8 lanes → 1
        // Step 1: split 256-bit into two 128-bit halves and min them → 4 lanes.
        let lo = _mm256_castsi256_si128(min_vec);
        let hi = _mm256_extracti128_si256(min_vec, 1);
        let v128 = _mm_min_epi32(lo, hi);

        // Step 2: swap pairs (DCBA → CDAB) and min → 2 distinct lanes.
        // Shuffle immediate 0b_10_11_00_01 = 0xB1 swaps adjacent 32-bit lanes.
        let shuf1 = _mm_shuffle_epi32(v128, 0b_10_11_00_01);
        let v64 = _mm_min_epi32(v128, shuf1);

        // Step 3: move lane 2 to lane 0 (0b_00_00_10_10 = 0xAA) and min → 1 lane.
        let shuf2 = _mm_shuffle_epi32(v64, 0b_00_00_10_10);
        let v32 = _mm_min_epi32(v64, shuf2);

        // SAFETY: _mm_extract_epi32 lane 0 is always valid.
        let avx_min = _mm_extract_epi32(v32, 0);

        // Scalar remainder
        let covered = chunks * 8;
        let (rem_idx, rem_val) = if covered < len {
            scalar_min_from(&costs[covered..])
                .map(|(i, v)| (i + covered, v))
                .unwrap_or((0, i32::MAX))
        } else {
            (0, i32::MAX)
        };

        let global_min = avx_min.min(rem_val);

        if rem_val < avx_min {
            (rem_idx, rem_val)
        } else {
            find_index_of(&costs[..covered], global_min)
        }
    }

    /// Find best predecessor using AVX2: argmin(prev_costs[i] + conn_costs[i]).
    ///
    /// Connection costs are stored as i16; this function widens them to i32 via
    /// `_mm256_cvtepi16_epi32` before the pairwise addition.
    ///
    /// # Safety
    ///
    /// Requires `avx2` target feature. Pointer arithmetic stays within `len`.
    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn find_best_predecessor_avx2(
        prev_costs: &[i32],
        conn_costs: &[i16],
    ) -> Option<(usize, i32)> {
        use core::arch::x86_64::{
            _mm_extract_epi32, _mm_loadu_si128, _mm_min_epi32, _mm_shuffle_epi32, _mm256_add_epi32,
            _mm256_castsi256_si128, _mm256_cvtepi16_epi32, _mm256_extracti128_si256,
            _mm256_loadu_si256, _mm256_min_epi32,
        };

        let len = prev_costs.len().min(conn_costs.len());
        if len == 0 {
            return None;
        }

        let chunks = len / 8;
        let ptr_p = prev_costs.as_ptr();
        let ptr_c = conn_costs.as_ptr();

        // Compute the minimum pairwise sum over all full 8-element chunks.
        let prefix_min: i32 = if chunks > 0 {
            // SAFETY: pointers valid for at least 8 elements (chunks ≥ 1).
            let prev_v = _mm256_loadu_si256(ptr_p as *const core::arch::x86_64::__m256i);
            // _mm_loadu_si128 loads 8 × i16 (128 bits); cvtepi16_epi32 widens to 8 × i32.
            let conn_narrow = _mm_loadu_si128(ptr_c as *const core::arch::x86_64::__m128i);
            let conn_v = _mm256_cvtepi16_epi32(conn_narrow);
            let mut sum_min = _mm256_add_epi32(prev_v, conn_v);

            for chunk in 1..chunks {
                // SAFETY: chunk < chunks ≤ len/8, offsets in-bounds.
                let pv =
                    _mm256_loadu_si256(ptr_p.add(chunk * 8) as *const core::arch::x86_64::__m256i);
                let cn =
                    _mm_loadu_si128(ptr_c.add(chunk * 8) as *const core::arch::x86_64::__m128i);
                let cv = _mm256_cvtepi16_epi32(cn);
                let sv = _mm256_add_epi32(pv, cv);
                sum_min = _mm256_min_epi32(sum_min, sv);
            }

            // Horizontal reduce: 8 lanes → 1 (same pattern as find_min_avx2).
            let lo = _mm256_castsi256_si128(sum_min);
            let hi = _mm256_extracti128_si256(sum_min, 1);
            let v128 = _mm_min_epi32(lo, hi);
            let shuf1 = _mm_shuffle_epi32(v128, 0b_10_11_00_01);
            let v64 = _mm_min_epi32(v128, shuf1);
            let shuf2 = _mm_shuffle_epi32(v64, 0b_00_00_10_10);
            let v32 = _mm_min_epi32(v64, shuf2);
            // SAFETY: lane 0 always valid.
            _mm_extract_epi32(v32, 0)
        } else {
            i32::MAX
        };

        let covered = chunks * 8;
        let mut best_idx = 0usize;
        let mut best_cost = prefix_min;

        // Scan remainder — may beat the SIMD prefix minimum.
        for i in covered..len {
            let sum = prev_costs[i].saturating_add(conn_costs[i] as i32);
            if sum < best_cost {
                best_cost = sum;
                best_idx = i;
            }
        }

        // If the remainder did not improve on prefix_min, find the index in
        // the SIMD prefix that achieves prefix_min.
        if best_cost == prefix_min {
            for i in 0..covered {
                let sum = prev_costs[i].saturating_add(conn_costs[i] as i32);
                if sum == prefix_min {
                    best_idx = i;
                    break;
                }
            }
        }

        Some((best_idx, best_cost))
    }

    // ── SSE4.1 path (128-bit = 4 × i32 per register) ─────────────────────────

    /// Find minimum value and its index using SSE4.1 (4 × i32 per iteration).
    ///
    /// Falls back to this path when AVX2 is not available but SSE4.1 is
    /// (Sandy Bridge / Bulldozer and later, universally present on x86_64 since ~2011).
    ///
    /// # Safety
    ///
    /// Requires `sse4.1` target feature. Pointer arithmetic stays within bounds.
    #[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.1"))]
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn find_min_sse41(costs: &[i32]) -> (usize, i32) {
        use core::arch::x86_64::{
            _mm_extract_epi32, _mm_loadu_si128, _mm_min_epi32, _mm_shuffle_epi32,
        };

        let len = costs.len();
        let chunks = len / 4;

        if chunks == 0 {
            return scalar_min_forward(costs);
        }

        let ptr = costs.as_ptr();

        // SAFETY: ptr valid for at least 4 × i32 = 16 bytes (chunks ≥ 1).
        let mut min_vec = _mm_loadu_si128(ptr as *const core::arch::x86_64::__m128i);

        for chunk in 1..chunks {
            // SAFETY: chunk < chunks ≤ len/4, ptr.add(chunk*4) is in bounds.
            let v = _mm_loadu_si128(ptr.add(chunk * 4) as *const core::arch::x86_64::__m128i);
            min_vec = _mm_min_epi32(min_vec, v);
        }

        // Horizontal reduce: 4 lanes → 1.
        // Step 1: swap adjacent pairs (DCBA → CDAB) and min → 2 distinct lanes.
        let shuf1 = _mm_shuffle_epi32(min_vec, 0b_10_11_00_01);
        let v2 = _mm_min_epi32(min_vec, shuf1);
        // Step 2: move lane 2 to lane 0 and min → 1 lane.
        let shuf2 = _mm_shuffle_epi32(v2, 0b_00_00_10_10);
        let v1 = _mm_min_epi32(v2, shuf2);
        // SAFETY: lane 0 always valid.
        let sse_min = _mm_extract_epi32(v1, 0);

        let covered = chunks * 4;
        let (rem_idx, rem_val) = if covered < len {
            scalar_min_from(&costs[covered..])
                .map(|(i, v)| (i + covered, v))
                .unwrap_or((0, i32::MAX))
        } else {
            (0, i32::MAX)
        };

        let global_min = sse_min.min(rem_val);

        if rem_val < sse_min {
            (rem_idx, rem_val)
        } else {
            find_index_of(&costs[..covered], global_min)
        }
    }

    /// Find best predecessor using SSE4.1: argmin(prev_costs[i] + conn_costs[i]).
    ///
    /// Connection costs (i16) are widened to i32 via `_mm_cvtepi16_epi32`.
    ///
    /// # Safety
    ///
    /// Requires `sse4.1` target feature. Pointer arithmetic stays within `len`.
    #[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.1"))]
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn find_best_predecessor_sse41(
        prev_costs: &[i32],
        conn_costs: &[i16],
    ) -> Option<(usize, i32)> {
        use core::arch::x86_64::{
            _mm_add_epi32, _mm_cvtepi16_epi32, _mm_extract_epi32, _mm_loadu_si64, _mm_loadu_si128,
            _mm_min_epi32, _mm_shuffle_epi32,
        };

        let len = prev_costs.len().min(conn_costs.len());
        if len == 0 {
            return None;
        }

        let chunks = len / 4;
        let ptr_p = prev_costs.as_ptr();
        let ptr_c = conn_costs.as_ptr();

        let prefix_min: i32 = if chunks > 0 {
            // SAFETY: pointers valid for at least 4 elements (chunks ≥ 1).
            let prev_v = _mm_loadu_si128(ptr_p as *const core::arch::x86_64::__m128i);
            // _mm_loadu_si64 loads 4 × i16 (64 bits); cvtepi16_epi32 widens to 4 × i32.
            let conn_narrow = _mm_loadu_si64(ptr_c as *const core::ffi::c_void);
            let conn_v = _mm_cvtepi16_epi32(conn_narrow);
            let mut sum_min = _mm_add_epi32(prev_v, conn_v);

            for chunk in 1..chunks {
                // SAFETY: chunk < chunks ≤ len/4, offsets in-bounds.
                let pv =
                    _mm_loadu_si128(ptr_p.add(chunk * 4) as *const core::arch::x86_64::__m128i);
                let cn = _mm_loadu_si64(ptr_c.add(chunk * 4) as *const core::ffi::c_void);
                let cv = _mm_cvtepi16_epi32(cn);
                let sv = _mm_add_epi32(pv, cv);
                sum_min = _mm_min_epi32(sum_min, sv);
            }

            // Horizontal reduce 4 → 1.
            let shuf1 = _mm_shuffle_epi32(sum_min, 0b_10_11_00_01);
            let v2 = _mm_min_epi32(sum_min, shuf1);
            let shuf2 = _mm_shuffle_epi32(v2, 0b_00_00_10_10);
            let v1 = _mm_min_epi32(v2, shuf2);
            // SAFETY: lane 0 always valid.
            _mm_extract_epi32(v1, 0)
        } else {
            i32::MAX
        };

        let covered = chunks * 4;
        let mut best_idx = 0usize;
        let mut best_cost = prefix_min;

        for i in covered..len {
            let sum = prev_costs[i].saturating_add(conn_costs[i] as i32);
            if sum < best_cost {
                best_cost = sum;
                best_idx = i;
            }
        }

        if best_cost == prefix_min {
            for i in 0..covered {
                let sum = prev_costs[i].saturating_add(conn_costs[i] as i32);
                if sum == prefix_min {
                    best_idx = i;
                    break;
                }
            }
        }

        Some((best_idx, best_cost))
    }

    // ── Public dispatch ───────────────────────────────────────────────────────

    /// Public entry point for `find_min` on x86_64.
    ///
    /// Selects the best available SIMD path at compile time:
    ///   - AVX2 (8 × i32) when `avx2` target feature is present
    ///   - SSE4.1 (4 × i32) when `sse4.1` is present but not `avx2`
    ///   - Scalar fallback otherwise
    pub fn find_min(costs: &[i32]) -> Option<(usize, i32)> {
        if costs.is_empty() {
            return None;
        }

        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: avx2 feature verified at compile time via cfg.
            return Some(unsafe { find_min_avx2(costs) });
        }

        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.1"))]
        {
            // SAFETY: sse4.1 feature verified at compile time via cfg.
            return Some(unsafe { find_min_sse41(costs) });
        }

        #[cfg(not(any(target_feature = "avx2", target_feature = "sse4.1")))]
        crate::viterbi::simd::scalar::find_min(costs)
    }

    /// Public entry point for `find_best_predecessor` on x86_64.
    ///
    /// Selects AVX2 → SSE4.1 → scalar at compile time.
    pub fn find_best_predecessor(prev_costs: &[i32], conn_costs: &[i16]) -> Option<(usize, i32)> {
        let len = prev_costs.len().min(conn_costs.len());
        if len == 0 {
            return None;
        }

        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: avx2 feature verified at compile time via cfg.
            return unsafe { find_best_predecessor_avx2(prev_costs, conn_costs) };
        }

        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.1"))]
        {
            // SAFETY: sse4.1 feature verified at compile time via cfg.
            return unsafe { find_best_predecessor_sse41(prev_costs, conn_costs) };
        }

        #[cfg(not(any(target_feature = "avx2", target_feature = "sse4.1")))]
        crate::viterbi::simd::scalar::find_best_predecessor(prev_costs, conn_costs)
    }
}

// ── x86_64 AVX2/SSE4.1 gather+widen ─────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_gather {
    /// Gather up to 16 connection costs from `row_data` at `right_ids[i]`,
    /// widen i16→i32, and store into `out[i]`.
    ///
    /// Uses AVX2 `_mm256_cvtepi16_epi32` (8-lane) when available, then
    /// SSE4.1 `_mm_cvtepi16_epi32` (4-lane), then scalar fallback.
    /// Returns the number of elements written (≤ `count.min(16)`).
    ///
    /// # Safety
    ///
    /// `out` must have capacity ≥ `count`.
    pub fn gather_widen(row_data: &[i16], right_ids: &[u16], out: &mut [i32]) -> usize {
        let count = right_ids.len().min(out.len()).min(16);
        if count == 0 {
            return 0;
        }

        let row_len = row_data.len();

        // Gather individual i16 values; OOB → i16::MAX.
        let mut gathered: [i16; 16] = [i16::MAX; 16];
        for (i, &rid) in right_ids[..count].iter().enumerate() {
            if (rid as usize) < row_len {
                // SAFETY: bounds checked.
                gathered[i] = unsafe { *row_data.get_unchecked(rid as usize) };
            }
        }

        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: avx2 compile-time cfg ensures the feature is present.
            unsafe { avx2_widen(&gathered, count, out) };
            return count;
        }

        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.1"))]
        {
            // SAFETY: sse4.1 compile-time cfg.
            unsafe { sse41_widen(&gathered, count, out) };
            return count;
        }

        // Scalar fallback
        #[cfg(not(any(target_feature = "avx2", target_feature = "sse4.1")))]
        {
            for i in 0..count {
                out[i] = gathered[i] as i32;
            }
            count
        }
    }

    /// Widen gathered[..count] i16→i32 using AVX2 (8-lane).
    ///
    /// # Safety
    ///
    /// Requires `avx2` target feature. `gathered` must be a 16-element array.
    /// `out` must hold at least `count` elements.
    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_widen(gathered: &[i16; 16], count: usize, out: &mut [i32]) {
        use core::arch::x86_64::{_mm_loadu_si128, _mm256_cvtepi16_epi32, _mm256_storeu_si256};

        let chunks = count / 8;
        let ptr_g = gathered.as_ptr();

        for chunk in 0..chunks {
            let base = chunk * 8;
            // SAFETY: base + 7 < 16 (chunks ≤ count/8 ≤ 16/8 = 2).
            let narrow =
                unsafe { _mm_loadu_si128(ptr_g.add(base) as *const core::arch::x86_64::__m128i) };
            let wide = _mm256_cvtepi16_epi32(narrow);
            // SAFETY: out[base..base+8] is within bounds.
            unsafe {
                _mm256_storeu_si256(
                    out.as_mut_ptr().add(base) as *mut core::arch::x86_64::__m256i,
                    wide,
                )
            };
        }

        // Scalar remainder
        let covered = chunks * 8;
        for i in covered..count {
            out[i] = gathered[i] as i32;
        }
    }

    /// Widen gathered[..count] i16→i32 using SSE4.1 (4-lane).
    ///
    /// # Safety
    ///
    /// Requires `sse4.1` target feature. `gathered` must be a 16-element array.
    /// `out` must hold at least `count` elements.
    #[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.1"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sse41_widen(gathered: &[i16; 16], count: usize, out: &mut [i32]) {
        use core::arch::x86_64::{_mm_cvtepi16_epi32, _mm_loadu_si64, _mm_storeu_si128};

        let chunks = count / 4;
        let ptr_g = gathered.as_ptr();

        for chunk in 0..chunks {
            let base = chunk * 4;
            // SAFETY: base + 3 < 16.
            let narrow = unsafe { _mm_loadu_si64(ptr_g.add(base) as *const core::ffi::c_void) };
            let wide = _mm_cvtepi16_epi32(narrow);
            // SAFETY: out[base..base+4] within bounds.
            unsafe {
                _mm_storeu_si128(
                    out.as_mut_ptr().add(base) as *mut core::arch::x86_64::__m128i,
                    wide,
                )
            };
        }

        // Scalar remainder
        let covered = chunks * 4;
        for i in covered..count {
            out[i] = gathered[i] as i32;
        }
    }
}
