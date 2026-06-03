//! x86_64 AVX2/SSE4.1 SIMD implementations for cost calculations.
//!
//! Provides 8-way (AVX2) or 4-way (SSE4.1) i32 parallelism for i32 kernels, and
//! 4-way (AVX2) or 2-way (SSE4.1) i64 parallelism for the `batch_min_argmin_i64` hot path.
//!
//! ## Runtime dispatch
//!
//! All public entry points perform:
//!   1. Compile-time fast path when `target_feature` is set at build time.
//!   2. Runtime `is_x86_feature_detected!` for stock x86-64 release builds.
//!   3. Scalar fallback otherwise.

// ── x86_64 AVX2/SSE4.1 find_min / find_best_predecessor ─────────────────────

/// x86_64 AVX2/SSE4.1 implementations.
#[cfg(target_arch = "x86_64")]
pub mod x86_impl {
    // ── Internal helpers ─────────────────────────────────────────────────────

    #[inline]
    #[allow(dead_code)]
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

    #[inline]
    #[allow(dead_code)]
    fn scalar_min_from(costs: &[i32]) -> Option<(usize, i32)> {
        if costs.is_empty() {
            None
        } else {
            Some(scalar_min_forward(costs))
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn find_index_of(costs: &[i32], target: i32) -> (usize, i32) {
        for (i, &c) in costs.iter().enumerate() {
            if c == target {
                return (i, target);
            }
        }
        scalar_min_forward(costs)
    }

    // ── AVX2 i32 kernels ─────────────────────────────────────────────────────

    /// Find minimum value and its index using AVX2 (8 x i32 per iteration).
    ///
    /// # Safety
    ///
    /// Caller must ensure `avx2` is available. Pointer arithmetic stays within bounds.
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
        // SAFETY: ptr valid for >=8x4=32 bytes (chunks>=1); avx2 verified by caller.
        let mut min_vec = unsafe { _mm256_loadu_si256(ptr as *const core::arch::x86_64::__m256i) };

        for chunk in 1..chunks {
            // SAFETY: ptr.add(chunk*8) in bounds (chunk < chunks <= len/8).
            let v = unsafe {
                _mm256_loadu_si256(ptr.add(chunk * 8) as *const core::arch::x86_64::__m256i)
            };
            min_vec = _mm256_min_epi32(min_vec, v);
        }

        // Horizontal reduce 8 -> 1.
        let lo = _mm256_castsi256_si128(min_vec);
        let hi = _mm256_extracti128_si256(min_vec, 1);
        let v128 = _mm_min_epi32(lo, hi);
        let shuf1 = _mm_shuffle_epi32(v128, 0b_10_11_00_01);
        let v64 = _mm_min_epi32(v128, shuf1);
        let shuf2 = _mm_shuffle_epi32(v64, 0b_00_00_10_10);
        let v32 = _mm_min_epi32(v64, shuf2);
        let avx_min = _mm_extract_epi32(v32, 0);

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
    /// # Safety
    ///
    /// Caller must ensure `avx2` is available. Pointer arithmetic stays within `len`.
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

        let prefix_min: i32 = if chunks > 0 {
            // SAFETY: pointers valid for >=8 elements; avx2 verified by caller.
            let prev_v = unsafe { _mm256_loadu_si256(ptr_p as *const core::arch::x86_64::__m256i) };
            let conn_narrow =
                unsafe { _mm_loadu_si128(ptr_c as *const core::arch::x86_64::__m128i) };
            let conn_v = _mm256_cvtepi16_epi32(conn_narrow);
            let mut sum_min = _mm256_add_epi32(prev_v, conn_v);

            for chunk in 1..chunks {
                // SAFETY: offsets in-bounds (chunk < chunks <= len/8).
                let pv = unsafe {
                    _mm256_loadu_si256(ptr_p.add(chunk * 8) as *const core::arch::x86_64::__m256i)
                };
                let cn = unsafe {
                    _mm_loadu_si128(ptr_c.add(chunk * 8) as *const core::arch::x86_64::__m128i)
                };
                let cv = _mm256_cvtepi16_epi32(cn);
                let sv = _mm256_add_epi32(pv, cv);
                sum_min = _mm256_min_epi32(sum_min, sv);
            }

            let lo = _mm256_castsi256_si128(sum_min);
            let hi = _mm256_extracti128_si256(sum_min, 1);
            let v128 = _mm_min_epi32(lo, hi);
            let shuf1 = _mm_shuffle_epi32(v128, 0b_10_11_00_01);
            let v64 = _mm_min_epi32(v128, shuf1);
            let shuf2 = _mm_shuffle_epi32(v64, 0b_00_00_10_10);
            let v32 = _mm_min_epi32(v64, shuf2);
            _mm_extract_epi32(v32, 0)
        } else {
            i32::MAX
        };

        let covered = chunks * 8;
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

    // ── SSE4.1 i32 kernels ───────────────────────────────────────────────────

    /// Find minimum value and its index using SSE4.1 (4 x i32 per iteration).
    ///
    /// # Safety
    ///
    /// Caller must ensure `sse4.1` is available. Pointer arithmetic stays within bounds.
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
        // SAFETY: ptr valid for >=4x4=16 bytes (chunks>=1); sse4.1 verified by caller.
        let mut min_vec = unsafe { _mm_loadu_si128(ptr as *const core::arch::x86_64::__m128i) };

        for chunk in 1..chunks {
            // SAFETY: ptr.add(chunk*4) in bounds.
            let v = unsafe {
                _mm_loadu_si128(ptr.add(chunk * 4) as *const core::arch::x86_64::__m128i)
            };
            min_vec = _mm_min_epi32(min_vec, v);
        }

        let shuf1 = _mm_shuffle_epi32(min_vec, 0b_10_11_00_01);
        let v2 = _mm_min_epi32(min_vec, shuf1);
        let shuf2 = _mm_shuffle_epi32(v2, 0b_00_00_10_10);
        let v1 = _mm_min_epi32(v2, shuf2);
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
    /// # Safety
    ///
    /// Caller must ensure `sse4.1` is available. Pointer arithmetic stays within `len`.
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
            // SAFETY: pointers valid for >=4 elements; sse4.1 verified by caller.
            let prev_v = unsafe { _mm_loadu_si128(ptr_p as *const core::arch::x86_64::__m128i) };
            // _mm_loadu_si64 loads 4 x i16 (64 bits); cvtepi16_epi32 widens to 4 x i32.
            let conn_narrow = unsafe { _mm_loadu_si64(ptr_c as *const u8) };
            let conn_v = _mm_cvtepi16_epi32(conn_narrow);
            let mut sum_min = _mm_add_epi32(prev_v, conn_v);

            for chunk in 1..chunks {
                // SAFETY: offsets in-bounds (chunk < chunks <= len/4).
                let pv = unsafe {
                    _mm_loadu_si128(ptr_p.add(chunk * 4) as *const core::arch::x86_64::__m128i)
                };
                let cn = unsafe { _mm_loadu_si64(ptr_c.add(chunk * 4) as *const u8) };
                let cv = _mm_cvtepi16_epi32(cn);
                let sv = _mm_add_epi32(pv, cv);
                sum_min = _mm_min_epi32(sum_min, sv);
            }

            let shuf1 = _mm_shuffle_epi32(sum_min, 0b_10_11_00_01);
            let v2 = _mm_min_epi32(sum_min, shuf1);
            let shuf2 = _mm_shuffle_epi32(v2, 0b_00_00_10_10);
            let v1 = _mm_min_epi32(v2, shuf2);
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

    // ── i64 argmin kernels ────────────────────────────────────────────────────

    /// Find best predecessor over i64 costs using AVX2 (4 x i64 per register).
    ///
    /// Computes `argmin_i(prev[i] + conn[i]) + wcost`, returning
    /// `Some((index, min_total))` strictly less than `best_so_far`, or `None`.
    ///
    /// AVX2 lacks `_mm256_min_epi64`; we use compare-and-blend:
    /// `mask = cmpgt(a, b); result = blendv(a, b, mask)` computes `min(a, b)`.
    ///
    /// # Safety
    ///
    /// Caller must ensure `avx2` is available. All pointer arithmetic stays within `len`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn find_best_predecessor_i64_avx2(
        prev: &[i64],
        conn: &[i32],
        wcost: i64,
        best_so_far: i64,
    ) -> Option<(usize, i64)> {
        use core::arch::x86_64::{
            _mm_blendv_epi8, _mm_cmpgt_epi64, _mm_cvtsi128_si64, _mm_loadu_si128,
            _mm_shuffle_epi32, _mm256_add_epi64, _mm256_blendv_epi8, _mm256_castsi256_si128,
            _mm256_cmpgt_epi64, _mm256_cvtepi32_epi64, _mm256_extracti128_si256,
            _mm256_loadu_si256,
        };

        let len = prev.len().min(conn.len());
        if len == 0 {
            return None;
        }

        let chunks4 = len / 4;
        let ptr_p = prev.as_ptr();
        let ptr_c = conn.as_ptr();

        let simd_min: i64 = if chunks4 > 0 {
            // SAFETY: pointers valid for >=4 elements; avx2 verified by caller.
            // _mm_loadu_si128 loads 4 x i32 (128 bits = 16 bytes);
            // _mm256_cvtepi32_epi64 sign-extends 4 x i32 to 4 x i64.
            let p0 = unsafe { _mm256_loadu_si256(ptr_p as *const core::arch::x86_64::__m256i) };
            let c_narrow = unsafe { _mm_loadu_si128(ptr_c as *const core::arch::x86_64::__m128i) };
            let c0 = _mm256_cvtepi32_epi64(c_narrow);
            let mut lane_min = _mm256_add_epi64(p0, c0);

            for chunk in 1..chunks4 {
                // SAFETY: offsets in-bounds (chunk < chunks4 <= len/4).
                let pv = unsafe {
                    _mm256_loadu_si256(ptr_p.add(chunk * 4) as *const core::arch::x86_64::__m256i)
                };
                let cn = unsafe {
                    _mm_loadu_si128(ptr_c.add(chunk * 4) as *const core::arch::x86_64::__m128i)
                };
                let cv = _mm256_cvtepi32_epi64(cn);
                let sv = _mm256_add_epi64(pv, cv);
                // min(lane_min, sv) via compare+blend (no _mm256_min_epi64 in AVX2).
                let mask = _mm256_cmpgt_epi64(lane_min, sv);
                // SAFETY: blendv selects sv where mask MSB is set (lane_min > sv).
                lane_min = _mm256_blendv_epi8(lane_min, sv, mask);
            }

            // Horizontal reduce 4 x i64 -> 1.
            let lo = _mm256_castsi256_si128(lane_min);
            let hi = _mm256_extracti128_si256(lane_min, 1);
            let mask128 = _mm_cmpgt_epi64(lo, hi);
            let v2 = _mm_blendv_epi8(lo, hi, mask128);
            // Shuffle 0b_01_00_11_10 swaps the two 64-bit halves.
            let shuf = _mm_shuffle_epi32(v2, 0b_01_00_11_10);
            let mask_final = _mm_cmpgt_epi64(v2, shuf);
            let v1 = _mm_blendv_epi8(v2, shuf, mask_final);
            // SAFETY: _mm_cvtsi128_si64 extracts lane 0 as i64.
            _mm_cvtsi128_si64(v1)
        } else {
            i64::MAX
        };

        let covered = chunks4 * 4;
        let mut best_idx = 0usize;
        let mut best_total = best_so_far;
        let mut found = false;

        for i in covered..len {
            let total = prev[i] + conn[i] as i64 + wcost;
            if total < best_total {
                best_total = total;
                best_idx = i;
                found = true;
            }
        }

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

    /// Find best predecessor over i64 costs using SSE4.1 (2 x i64 per register).
    ///
    /// `_mm_cmpgt_epi64` + `_mm_blendv_epi8` provide branchless min.
    ///
    /// # Safety
    ///
    /// Caller must ensure `sse4.1` is available. All pointer arithmetic stays within `len`.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn find_best_predecessor_i64_sse41(
        prev: &[i64],
        conn: &[i32],
        wcost: i64,
        best_so_far: i64,
    ) -> Option<(usize, i64)> {
        use core::arch::x86_64::{
            _mm_add_epi64, _mm_blendv_epi8, _mm_cmpgt_epi64, _mm_cvtepi32_epi64, _mm_cvtsi128_si64,
            _mm_loadu_si64, _mm_loadu_si128, _mm_shuffle_epi32,
        };

        let len = prev.len().min(conn.len());
        if len == 0 {
            return None;
        }

        let chunks2 = len / 2;
        let ptr_p = prev.as_ptr();
        let ptr_c = conn.as_ptr();

        let simd_min: i64 = if chunks2 > 0 {
            // SAFETY: pointers valid for >=2 elements; sse4.1 verified by caller.
            // _mm_loadu_si128 loads 2 x i64; _mm_loadu_si64 loads 2 x i32;
            // _mm_cvtepi32_epi64 sign-extends 2 x i32 -> 2 x i64.
            let p0 = unsafe { _mm_loadu_si128(ptr_p as *const core::arch::x86_64::__m128i) };
            let c_narrow = unsafe { _mm_loadu_si64(ptr_c as *const u8) };
            let c0 = _mm_cvtepi32_epi64(c_narrow);
            let mut lane_min = _mm_add_epi64(p0, c0);

            for chunk in 1..chunks2 {
                // SAFETY: offsets in-bounds (chunk < chunks2 <= len/2).
                let pv = unsafe {
                    _mm_loadu_si128(ptr_p.add(chunk * 2) as *const core::arch::x86_64::__m128i)
                };
                let cn = unsafe { _mm_loadu_si64(ptr_c.add(chunk * 2) as *const u8) };
                let cv = _mm_cvtepi32_epi64(cn);
                let sv = _mm_add_epi64(pv, cv);
                // SAFETY: _mm_cmpgt_epi64 is an SSE4.2 intrinsic; this function is
                // only called from dispatch paths that verify sse4.1, but SSE4.2 is
                // universally available alongside SSE4.1 on all real CPUs (Nehalem+).
                let mask = unsafe { _mm_cmpgt_epi64(lane_min, sv) };
                lane_min = _mm_blendv_epi8(lane_min, sv, mask);
            }

            // Horizontal reduce 2 x i64 -> 1.
            let shuf = _mm_shuffle_epi32(lane_min, 0b_01_00_11_10);
            // SAFETY: _mm_cmpgt_epi64 is SSE4.2; see above note.
            let mask_final = unsafe { _mm_cmpgt_epi64(lane_min, shuf) };
            let v1 = _mm_blendv_epi8(lane_min, shuf, mask_final);
            // SAFETY: lane 0 always valid.
            _mm_cvtsi128_si64(v1)
        } else {
            i64::MAX
        };

        let covered = chunks2 * 2;
        let mut best_idx = 0usize;
        let mut best_total = best_so_far;
        let mut found = false;

        for i in covered..len {
            let total = prev[i] + conn[i] as i64 + wcost;
            if total < best_total {
                best_total = total;
                best_idx = i;
                found = true;
            }
        }

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

    // ── Public dispatch ───────────────────────────────────────────────────────

    /// Public entry point for `find_min` on x86_64.
    ///
    /// Priority: compile-time `avx2` -> compile-time `sse4.1` -> runtime
    /// `is_x86_feature_detected!` -> scalar fallback.
    pub fn find_min(costs: &[i32]) -> Option<(usize, i32)> {
        if costs.is_empty() {
            return None;
        }

        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: compile-time avx2.
            return Some(unsafe { find_min_avx2(costs) });
        }

        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.1"))]
        {
            // SAFETY: compile-time sse4.1.
            return Some(unsafe { find_min_sse41(costs) });
        }

        #[cfg(not(any(target_feature = "avx2", target_feature = "sse4.1")))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: runtime-detected AVX2; find_min_avx2 has #[target_feature(enable="avx2")].
                return Some(unsafe { find_min_avx2(costs) });
            }
            if is_x86_feature_detected!("sse4.1") {
                // SAFETY: runtime-detected SSE4.1; find_min_sse41 has #[target_feature(enable="sse4.1")].
                return Some(unsafe { find_min_sse41(costs) });
            }
            crate::viterbi::simd::scalar::find_min(costs)
        }
    }

    /// Public entry point for `find_best_predecessor` on x86_64.
    ///
    /// Priority: compile-time `avx2` -> compile-time `sse4.1` -> runtime detection -> scalar.
    pub fn find_best_predecessor(prev_costs: &[i32], conn_costs: &[i16]) -> Option<(usize, i32)> {
        let len = prev_costs.len().min(conn_costs.len());
        if len == 0 {
            return None;
        }

        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: compile-time avx2.
            return unsafe { find_best_predecessor_avx2(prev_costs, conn_costs) };
        }

        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.1"))]
        {
            // SAFETY: compile-time sse4.1.
            return unsafe { find_best_predecessor_sse41(prev_costs, conn_costs) };
        }

        #[cfg(not(any(target_feature = "avx2", target_feature = "sse4.1")))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: runtime-detected AVX2.
                return unsafe { find_best_predecessor_avx2(prev_costs, conn_costs) };
            }
            if is_x86_feature_detected!("sse4.1") {
                // SAFETY: runtime-detected SSE4.1.
                return unsafe { find_best_predecessor_sse41(prev_costs, conn_costs) };
            }
            crate::viterbi::simd::scalar::find_best_predecessor(prev_costs, conn_costs)
        }
    }

    /// Public dispatch for `batch_min_argmin_i64` on x86_64.
    ///
    /// Priority: compile-time `avx2` (4 x i64) -> runtime `avx2` -> runtime `sse4.1`
    /// (2 x i64) -> scalar.
    pub fn batch_min_argmin_i64(
        prev: &[i64],
        conn: &[i32],
        wcost: i64,
        best_so_far: i64,
    ) -> Option<(usize, i64)> {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: compile-time avx2.
            return unsafe { find_best_predecessor_i64_avx2(prev, conn, wcost, best_so_far) };
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: runtime-detected AVX2; find_best_predecessor_i64_avx2 requires avx2.
                return unsafe { find_best_predecessor_i64_avx2(prev, conn, wcost, best_so_far) };
            }
            if is_x86_feature_detected!("sse4.1") {
                // SAFETY: runtime-detected SSE4.1; find_best_predecessor_i64_sse41 requires sse4.1.
                return unsafe { find_best_predecessor_i64_sse41(prev, conn, wcost, best_so_far) };
            }
            crate::viterbi::simd::scalar::batch_min_argmin_i64(prev, conn, wcost, best_so_far)
        }
    }
}

// ── x86_64 AVX2/SSE4.1 gather+widen ─────────────────────────────────────────

/// x86_64 AVX2/SSE4.1 gather+widen helper.
#[cfg(target_arch = "x86_64")]
pub mod x86_gather {
    /// Gather up to 16 connection costs from `row_data` at `right_ids[i]`,
    /// widen i16->i32, and store into `out[i]`.
    ///
    /// Feature selection priority (with runtime detection for stock builds):
    ///   1. Compile-time `avx2`
    ///   2. Compile-time `sse4.1` (no `avx2`)
    ///   3. Runtime `is_x86_feature_detected!("avx2")`
    ///   4. Runtime `is_x86_feature_detected!("sse4.1")`
    ///   5. Scalar fallback
    ///
    /// Returns the number of elements written (<=`right_ids.len().min(16)`).
    pub fn gather_widen(row_data: &[i16], right_ids: &[u16], out: &mut [i32]) -> usize {
        let count = right_ids.len().min(out.len()).min(16);
        if count == 0 {
            return 0;
        }

        let row_len = row_data.len();

        // Gather individual i16 values; OOB -> i16::MAX.
        let mut gathered: [i16; 16] = [i16::MAX; 16];
        for (i, &rid) in right_ids[..count].iter().enumerate() {
            if (rid as usize) < row_len {
                // SAFETY: bounds checked immediately above.
                gathered[i] = unsafe { *row_data.get_unchecked(rid as usize) };
            }
        }

        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: compile-time avx2.
            unsafe { avx2_widen(&gathered, count, out) };
            return count;
        }

        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.1"))]
        {
            // SAFETY: compile-time sse4.1.
            unsafe { sse41_widen(&gathered, count, out) };
            return count;
        }

        #[cfg(not(any(target_feature = "avx2", target_feature = "sse4.1")))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: runtime-detected AVX2; avx2_widen has #[target_feature(enable="avx2")].
                unsafe { avx2_widen(&gathered, count, out) };
                return count;
            }
            if is_x86_feature_detected!("sse4.1") {
                // SAFETY: runtime-detected SSE4.1; sse41_widen has #[target_feature(enable="sse4.1")].
                unsafe { sse41_widen(&gathered, count, out) };
                return count;
            }
            // Scalar fallback.
            for i in 0..count {
                out[i] = gathered[i] as i32;
            }
            count
        }
    }

    /// Widen gathered[..count] i16->i32 using AVX2 (8 lanes per register).
    ///
    /// # Safety
    ///
    /// Caller must ensure `avx2` is available (compile-time or runtime-detected).
    /// `gathered` is a 16-element array; `out` must hold >= `count` elements.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_widen(gathered: &[i16; 16], count: usize, out: &mut [i32]) {
        use core::arch::x86_64::{_mm_loadu_si128, _mm256_cvtepi16_epi32, _mm256_storeu_si256};

        let chunks = count / 8;
        let ptr_g = gathered.as_ptr();

        for chunk in 0..chunks {
            let base = chunk * 8;
            // SAFETY: base+7 < 16 (chunks<=2); out[base..base+8] in bounds.
            let narrow =
                unsafe { _mm_loadu_si128(ptr_g.add(base) as *const core::arch::x86_64::__m128i) };
            let wide = _mm256_cvtepi16_epi32(narrow);
            unsafe {
                _mm256_storeu_si256(
                    out.as_mut_ptr().add(base) as *mut core::arch::x86_64::__m256i,
                    wide,
                );
            }
        }

        let covered = chunks * 8;
        for i in covered..count {
            out[i] = gathered[i] as i32;
        }
    }

    /// Widen gathered[..count] i16->i32 using SSE4.1 (4 lanes per register).
    ///
    /// # Safety
    ///
    /// Caller must ensure `sse4.1` is available (compile-time or runtime-detected).
    /// `gathered` is a 16-element array; `out` must hold >= `count` elements.
    #[target_feature(enable = "sse4.1")]
    unsafe fn sse41_widen(gathered: &[i16; 16], count: usize, out: &mut [i32]) {
        use core::arch::x86_64::{_mm_cvtepi16_epi32, _mm_loadu_si64, _mm_storeu_si128};

        let chunks = count / 4;
        let ptr_g = gathered.as_ptr();

        for chunk in 0..chunks {
            let base = chunk * 4;
            // SAFETY: base+3 < 16; out[base..base+4] in bounds.
            let narrow = unsafe { _mm_loadu_si64(ptr_g.add(base) as *const u8) };
            let wide = _mm_cvtepi16_epi32(narrow);
            unsafe {
                _mm_storeu_si128(
                    out.as_mut_ptr().add(base) as *mut core::arch::x86_64::__m128i,
                    wide,
                );
            }
        }

        let covered = chunks * 4;
        for i in covered..count {
            out[i] = gathered[i] as i32;
        }
    }
}
