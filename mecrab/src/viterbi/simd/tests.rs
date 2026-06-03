//! Tests for SIMD cost calculations.

use super::*;

// ── batch_add_costs ──────────────────────────────────────────────────────

#[test]
fn test_batch_add_costs() {
    let mut costs = vec![10, 20, 30, 40, 50, 60, 70, 80];
    batch_add_costs(&mut costs, 5);
    assert_eq!(costs, vec![15, 25, 35, 45, 55, 65, 75, 85]);
}

#[test]
fn test_batch_add_costs_with_remainder() {
    let mut costs = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    batch_add_costs(&mut costs, 5);
    assert_eq!(costs, vec![15, 25, 35, 45, 55, 65, 75, 85, 95, 105]);
}

// ── find_min ─────────────────────────────────────────────────────────────

#[test]
fn test_find_min_empty() {
    let costs: Vec<i32> = vec![];
    assert_eq!(find_min(&costs), None);
}

#[test]
fn test_find_min_single() {
    let costs = vec![42];
    assert_eq!(find_min(&costs), Some((0, 42)));
}

#[test]
fn test_find_min_first() {
    let costs = vec![1, 2, 3, 4, 5];
    assert_eq!(find_min(&costs), Some((0, 1)));
}

#[test]
fn test_find_min_last() {
    let costs = vec![5, 4, 3, 2, 1];
    assert_eq!(find_min(&costs), Some((4, 1)));
}

#[test]
fn test_find_min_large() {
    let costs: Vec<i32> = (0..100).rev().collect();
    assert_eq!(find_min(&costs), Some((99, 0)));
}

#[test]
fn test_find_min_simd_path() {
    let costs = vec![50, 40, 30, 20, 10, 5, 15, 25, 35];
    assert_eq!(find_min(&costs), Some((5, 5)));
}

// ── x86_64-specific tests ────────────────────────────────────────────────

/// Verify x86_impl::find_min agrees with scalar_impl for various slice sizes.
#[cfg(target_arch = "x86_64")]
#[test]
fn test_find_min_x86_vs_scalar() {
    let owned: Vec<i32> = (0..100).map(|i| 100 - i).collect();
    let test_cases: &[&[i32]] = &[
        &[5],
        &[3, 1, 4, 1, 5, 9, 2, 6],
        &[10, 20, 30, -5, 40, 50, 60, 70, -10, 80],
        &owned,
    ];
    for &costs in test_cases {
        let scalar = scalar_impl::find_min(costs);
        let x86 = x86_impl::find_min(costs);
        assert_eq!(
            scalar.map(|(_, v)| v),
            x86.map(|(_, v)| v),
            "value mismatch for costs = {:?}",
            costs
        );
        // Ensure the returned index actually points to the minimum value.
        if let Some((idx, val)) = x86 {
            assert_eq!(
                costs[idx], val,
                "x86 index {idx} does not point to min value {val}"
            );
        }
    }
}

/// Verify x86_impl::find_best_predecessor agrees with scalar for known data.
#[cfg(target_arch = "x86_64")]
#[test]
fn test_find_best_predecessor_x86_vs_scalar() {
    // prev[2]=5, conn[2]=1 → sum=6; all others ≥ 12
    let prev: Vec<i32> = vec![10, 20, 5, 15, 8, 9, 3, 100, 7, 50];
    let conn: Vec<i16> = vec![2, 1, 1, 3, 10, 7, 5, 1, 20, 3];
    // Sums: 12, 21, 6, 18, 18, 16, 8, 101, 27, 53 → min = 6 at index 2

    let scalar = scalar_impl::find_best_predecessor(&prev, &conn);
    let x86 = x86_impl::find_best_predecessor(&prev, &conn);

    assert!(scalar.is_some(), "scalar must find a predecessor");
    assert!(x86.is_some(), "x86 must find a predecessor");

    let (_, sv) = scalar.expect("scalar result");
    let (xi, xv) = x86.expect("x86 result");

    assert_eq!(xv, sv, "x86 and scalar must agree on minimum sum");
    assert_eq!(
        prev[xi].saturating_add(conn[xi] as i32),
        xv,
        "x86 index {xi} does not yield expected sum {xv}"
    );
}

// ── NEON-specific tests (aarch64 only) ───────────────────────────────────

/// Verify NEON find_min agrees with scalar for various slice sizes.
#[cfg(target_arch = "aarch64")]
#[test]
fn test_find_min_neon_vs_scalar() {
    let test_cases: Vec<Vec<i32>> = vec![
        vec![42],                     // len = 1
        vec![3, 1, 2],                // len = 3
        vec![4, 3, 2, 1],             // len = 4 (exact NEON batch)
        vec![7, 6, 5, 4, 3, 2, 1],    // len = 7
        vec![8, 7, 6, 5, 4, 3, 2, 1], // len = 8 (two NEON batches)
        (0_i32..100).rev().collect(), // len = 100
    ];

    for costs in &test_cases {
        let scalar = scalar_impl::find_min_scalar(costs);
        // SAFETY: NEON is always available on aarch64 / Apple Silicon.
        let neon = unsafe { neon_impl::find_min_neon(costs) };

        assert_eq!(
            scalar.1,
            neon.1,
            "min value mismatch len={}: scalar={scalar:?} neon={neon:?}",
            costs.len()
        );
        assert_eq!(
            costs[neon.0],
            neon.1,
            "neon index {} does not point to min value {} (len={})",
            neon.0,
            neon.1,
            costs.len()
        );
    }
}

/// Verify find_best_predecessor_neon on arrays with a known minimum.
#[cfg(target_arch = "aarch64")]
#[test]
fn test_find_best_predecessor_neon() {
    // prev[2]=5, conn[2]=1 → sum=6; all others > 6
    let prev: Vec<i32> = vec![10, 20, 5, 15, 8, 9];
    let conn: Vec<i16> = vec![2, 1, 1, 3, 10, 7];
    // Sums: 12, 21, 6, 18, 18, 16 → min = 6 at index 2

    // SAFETY: NEON mandatory on aarch64.
    let neon = unsafe { neon_impl::find_best_predecessor_neon(&prev, &conn) };
    let scalar = scalar_impl::find_best_predecessor_scalar(&prev, &conn);

    assert!(neon.is_some());
    assert!(scalar.is_some());
    let (ni, nv) = neon.expect("neon result");
    let (_si, sv) = scalar.expect("scalar result");

    assert_eq!(nv, sv, "neon and scalar must agree on minimum sum");
    assert_eq!(
        prev[ni].saturating_add(conn[ni] as i32),
        nv,
        "neon index {ni} does not yield expected sum {nv}"
    );
}

// ── find_best_predecessor (public API) ───────────────────────────────────

#[test]
fn test_find_best_predecessor() {
    let prev = vec![10, 20, 5, 15];
    let conn = vec![2i16, 1, 3, 1];
    let result = find_best_predecessor(&prev, &conn);
    // Sums: 12, 21, 8, 16 → min at index 2, value 8
    assert_eq!(result, Some((2, 8)));
}

// ── predecessor_batch_simd ───────────────────────────────────────────────

#[test]
fn test_predecessor_batch_simd() {
    let prev = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let conn: Vec<i16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let result = predecessor_batch_simd(&prev, &conn);
    assert_eq!(result, vec![11, 22, 33, 44, 55, 66, 77, 88]);
}

// ── predecessor_batch ────────────────────────────────────────────────────

#[test]
fn test_predecessor_batch() {
    let prev = vec![10, 20, 30];
    let matrix: Vec<i16> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9]; // 3×3 matrix
    let result = predecessor_batch(&prev, &matrix, 1, 3);
    // [10+2, 20+5, 30+8] = [12, 25, 38]
    assert_eq!(result, vec![12, 25, 38]);
}

// ── batch_connection_costs ───────────────────────────────────────────────

/// Basic correctness: all right_ids in-bounds.
#[test]
fn test_batch_connection_costs_basic() {
    // row_data simulates a connection matrix row of length 8.
    let row: Vec<i16> = vec![10, 20, -5, 100, 42, -100, 7, 3];
    let right_ids: Vec<u16> = vec![0, 1, 2, 3];
    let mut out = [0i32; 16];

    let written = batch_connection_costs(&row, &right_ids, &mut out);
    assert_eq!(written, 4, "should write exactly 4 costs");
    assert_eq!(out[0], 10i32, "right_id 0 → row[0] = 10");
    assert_eq!(out[1], 20i32, "right_id 1 → row[1] = 20");
    assert_eq!(out[2], -5i32, "right_id 2 → row[2] = -5");
    assert_eq!(out[3], 100i32, "right_id 3 → row[3] = 100");
}

/// Out-of-bounds right_ids must produce i16::MAX (widened to i32).
#[test]
fn test_batch_connection_costs_oob() {
    let row: Vec<i16> = vec![1, 2, 3];
    let right_ids: Vec<u16> = vec![0, 99, 2, 255]; // 99 and 255 are OOB
    let mut out = [0i32; 16];

    let written = batch_connection_costs(&row, &right_ids, &mut out);
    assert_eq!(written, 4);
    assert_eq!(out[0], 1i32);
    assert_eq!(out[1], i16::MAX as i32, "OOB must yield i16::MAX");
    assert_eq!(out[2], 3i32);
    assert_eq!(out[3], i16::MAX as i32, "OOB must yield i16::MAX");
}

/// Cap at 16 elements even if right_ids is longer.
#[test]
fn test_batch_connection_costs_cap_at_16() {
    let row: Vec<i16> = (0i16..32).collect();
    let right_ids: Vec<u16> = (0u16..20).collect(); // 20 > 16
    let mut out = [0i32; 20];

    let written = batch_connection_costs(&row, &right_ids, &mut out);
    assert_eq!(written, 16, "must cap at 16");
    for (i, item) in out.iter().enumerate().take(16) {
        assert_eq!(*item, i as i32, "out[{i}] must equal row[{i}]");
    }
}

/// Negative i16 values widen correctly to i32.
#[test]
fn test_batch_connection_costs_negative_widening() {
    let row: Vec<i16> = vec![-1, -32768, -100, 32767, -200, 0, 1, -1000];
    let right_ids: Vec<u16> = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let mut out = [0i32; 16];

    let written = batch_connection_costs(&row, &right_ids, &mut out);
    assert_eq!(written, 8);
    assert_eq!(out[0], -1i32);
    assert_eq!(out[1], -32768i32);
    assert_eq!(out[2], -100i32);
    assert_eq!(out[3], 32767i32);
    assert_eq!(out[4], -200i32);
    assert_eq!(out[5], 0i32);
    assert_eq!(out[6], 1i32);
    assert_eq!(out[7], -1000i32);
}

/// Empty right_ids slice writes nothing.
#[test]
fn test_batch_connection_costs_empty() {
    let row: Vec<i16> = vec![1, 2, 3];
    let right_ids: &[u16] = &[];
    let mut out = [0i32; 4];

    let written = batch_connection_costs(&row, right_ids, &mut out);
    assert_eq!(written, 0);
}

/// Exactly 16 elements (maximum batch).
#[test]
fn test_batch_connection_costs_full_16() {
    let row: Vec<i16> = (0i16..16).map(|x| x * 7 - 50).collect(); // varied values
    let right_ids: Vec<u16> = (0u16..16).collect();
    let mut out = [0i32; 16];

    let written = batch_connection_costs(&row, &right_ids, &mut out);
    assert_eq!(written, 16);
    for (i, item) in out.iter().enumerate().take(16) {
        assert_eq!(*item, row[i] as i32, "mismatch at index {i}");
    }
}

/// Non-sequential right_ids (sparse gather).
#[test]
fn test_batch_connection_costs_non_sequential() {
    let row: Vec<i16> = vec![100, 200, 300, 400, 500, 600, 700, 800];
    let right_ids: Vec<u16> = vec![7, 0, 5, 3, 1]; // reverse + sparse
    let mut out = [0i32; 16];

    let written = batch_connection_costs(&row, &right_ids, &mut out);
    assert_eq!(written, 5);
    assert_eq!(out[0], 800i32); // row[7]
    assert_eq!(out[1], 100i32); // row[0]
    assert_eq!(out[2], 600i32); // row[5]
    assert_eq!(out[3], 400i32); // row[3]
    assert_eq!(out[4], 200i32); // row[1]
}

// ── batch_min_argmin_i64 ─────────────────────────────────────────────────

/// Helper: compute expected result via the scalar reference, then compare
/// against the platform-dispatch version.
fn check_batch_min_argmin_i64(prev: &[i64], conn: &[i32], wcost: i64, best_so_far: i64) {
    let expected = scalar_impl::batch_min_argmin_i64(prev, conn, wcost, best_so_far);
    let actual = batch_min_argmin_i64(prev, conn, wcost, best_so_far);

    match (expected, actual) {
        (None, None) => {}
        (Some((ei, ev)), Some((ai, av))) => {
            assert_eq!(
                ev, av,
                "value mismatch: scalar={ev} dispatch={av} (prev={prev:?}, conn={conn:?}, \
                 wcost={wcost}, best_so_far={best_so_far})"
            );
            // Both indices must point to the same total cost (there may be ties).
            let expected_sum = prev[ei] + conn[ei] as i64 + wcost;
            let actual_sum = prev[ai] + conn[ai] as i64 + wcost;
            assert_eq!(
                expected_sum, actual_sum,
                "index mismatch produces different sums: scalar_idx={ei} dispatch_idx={ai}"
            );
        }
        (expected, actual) => panic!(
            "None/Some mismatch: scalar={expected:?} dispatch={actual:?} \
             (prev={prev:?}, conn={conn:?}, wcost={wcost}, best_so_far={best_so_far})"
        ),
    }
}

/// Basic correctness: minimum at a known index with wcost = 0.
#[test]
fn test_batch_min_argmin_i64_basic() {
    let prev: Vec<i64> = vec![100, 200, 50, 300, 150];
    let conn: Vec<i32> = vec![10, 5, 3, 8, 1];
    // Sums: 110, 205, 53, 308, 151 → min = 53 at index 2 (+ wcost 0)
    check_batch_min_argmin_i64(&prev, &conn, 0, i64::MAX);
}

/// Positive wcost shifts all totals but does not change the argmin.
#[test]
fn test_batch_min_argmin_i64_positive_wcost() {
    let prev: Vec<i64> = vec![100, 200, 50, 300];
    let conn: Vec<i32> = vec![10, 5, 3, 8];
    for wcost in [1_i64, 100, 1000, 32_000] {
        check_batch_min_argmin_i64(&prev, &conn, wcost, i64::MAX);
    }
}

/// Negative wcost (common in Japanese morphology dictionaries).
#[test]
fn test_batch_min_argmin_i64_negative_wcost() {
    let prev: Vec<i64> = vec![500, 300, 800, 100];
    let conn: Vec<i32> = vec![50, 10, -200, 5];
    for wcost in [-100_i64, -32_000, -1] {
        check_batch_min_argmin_i64(&prev, &conn, wcost, i64::MAX);
    }
}

/// best_so_far guard: no improvement → None.
#[test]
fn test_batch_min_argmin_i64_no_improvement() {
    let prev: Vec<i64> = vec![100, 200, 50];
    let conn: Vec<i32> = vec![10, 5, 3];
    // min total = 53; best_so_far = 53 means we require *strictly* better.
    let result = batch_min_argmin_i64(&prev, &conn, 0, 53);
    assert_eq!(result, None, "should return None when min == best_so_far");

    // best_so_far = 52 → also no improvement.
    let result2 = batch_min_argmin_i64(&prev, &conn, 0, 52);
    assert_eq!(result2, None, "should return None when min > best_so_far");
}

/// Empty slice returns None.
#[test]
fn test_batch_min_argmin_i64_empty() {
    let prev: Vec<i64> = vec![];
    let conn: Vec<i32> = vec![];
    assert_eq!(batch_min_argmin_i64(&prev, &conn, 0, i64::MAX), None);
}

/// Single-element slice.
#[test]
fn test_batch_min_argmin_i64_single() {
    let prev = vec![42_i64];
    let conn = vec![8_i32];
    // total = 42 + 8 + 5 = 55
    let result = batch_min_argmin_i64(&prev, &conn, 5, i64::MAX);
    assert_eq!(result, Some((0, 55)));
}

/// Larger slice exercising SIMD lanes (> 4 elements for AVX2, > 2 for SSE/NEON).
#[test]
fn test_batch_min_argmin_i64_larger_slice() {
    let prev: Vec<i64> = vec![1000, 2000, 3000, 4000, 500, 6000, 7000, 8000, 9000, 10_000];
    let conn: Vec<i32> = vec![50, 100, -100, 200, 10, 300, 400, 500, -50, 600];
    // index 4: 500 + 10 = 510; index 2: 3000 - 100 = 2900; check index 4 wins
    check_batch_min_argmin_i64(&prev, &conn, 0, i64::MAX);
    check_batch_min_argmin_i64(&prev, &conn, -200, i64::MAX);
}

/// Negative previous costs (accumulated costs can be negative with very cheap nodes).
#[test]
fn test_batch_min_argmin_i64_negative_prev() {
    let prev: Vec<i64> = vec![-5000, -3000, -1000, -8000, -100];
    let conn: Vec<i32> = vec![100, 200, 300, 50, 10];
    check_batch_min_argmin_i64(&prev, &conn, 0, i64::MAX);
    check_batch_min_argmin_i64(&prev, &conn, 5000, i64::MAX);
    check_batch_min_argmin_i64(&prev, &conn, -5000, i64::MAX);
}

/// Exactly 4 elements (AVX2 boundary) and 2 elements (SSE4.1/NEON boundary).
#[test]
fn test_batch_min_argmin_i64_boundary_sizes() {
    for &n in &[1_usize, 2, 3, 4, 5, 7, 8, 9, 15, 16] {
        let prev: Vec<i64> = (0..n as i64).map(|i| (n as i64 - i) * 100).collect();
        let conn: Vec<i32> = (0..n as i32).map(|i| i * 3 - 10).collect();
        check_batch_min_argmin_i64(&prev, &conn, 0, i64::MAX);
        check_batch_min_argmin_i64(&prev, &conn, -99, i64::MAX);
    }
}

/// Scalar reference agrees with itself (sanity check).
#[test]
fn test_scalar_batch_min_argmin_i64_reference() {
    let prev: Vec<i64> = vec![10, 20, 5, 15, 8];
    let conn: Vec<i32> = vec![2, 1, 1, 3, 10];
    // Sums: 12, 21, 6, 18, 18 → min = 6 at index 2
    let result = scalar_impl::batch_min_argmin_i64(&prev, &conn, 0, i64::MAX);
    assert_eq!(result, Some((2, 6)));

    // With wcost = 100: all totals + 100; min = 106 at index 2
    let result2 = scalar_impl::batch_min_argmin_i64(&prev, &conn, 100, i64::MAX);
    assert_eq!(result2, Some((2, 106)));
}

// ── wasm32 i64x2 SIMD tests (platform-independent correctness) ───────────

#[test]
fn test_batch_min_argmin_i64_matches_scalar() {
    // Dispatch result must match scalar reference on several representative inputs.
    let test_cases: &[(&[i64], &[i32], i64, i64)] = &[
        (&[10, 5, 8, 3, 12], &[2, 4, 1, 6, 3], 1, i64::MAX),
        (&[100, 200], &[50, 25], 10, i64::MAX),
        (
            &[1, 1, 1, 1, 1, 1, 1, 1],
            &[8, 7, 6, 5, 4, 3, 2, 1],
            0,
            i64::MAX,
        ),
        (&[0], &[0], 42, i64::MAX),
    ];
    for &(prev, conn, wcost, best_so_far) in test_cases {
        let scalar =
            crate::viterbi::simd::scalar_impl::batch_min_argmin_i64(prev, conn, wcost, best_so_far);
        let dispatch = crate::viterbi::simd::batch_min_argmin_i64(prev, conn, wcost, best_so_far);
        // Both must return the same minimum *value* (ties on index are acceptable).
        assert_eq!(
            scalar.map(|(_, v)| v),
            dispatch.map(|(_, v)| v),
            "SIMD/scalar value mismatch: prev={prev:?}, conn={conn:?}, \
             wcost={wcost}, best={best_so_far}"
        );
        // If an index is returned, it must point to the same total cost.
        if let (Some((si, _)), Some((di, _))) = (scalar, dispatch) {
            let s_total = prev[si] + conn[si] as i64 + wcost;
            let d_total = prev[di] + conn[di] as i64 + wcost;
            assert_eq!(
                s_total, d_total,
                "SIMD/scalar index mismatch produces different sums: \
                 scalar_idx={si} dispatch_idx={di}"
            );
        }
    }
}

#[test]
fn test_batch_min_argmin_i64_no_improvement_returns_none() {
    // When all totals >= best_so_far the function must return None.
    let prev = [10i64, 20, 30];
    let conn = [5i32, 5, 5];
    let wcost = 0i64;
    // 10+5+0=15, 20+5+0=25, 30+5+0=35 — all > 10.
    let best_so_far = 10i64;
    let result = crate::viterbi::simd::batch_min_argmin_i64(&prev, &conn, wcost, best_so_far);
    assert!(result.is_none(), "expected None, got {result:?}");
}

// ── SimdStats ────────────────────────────────────────────────────────────

#[test]
fn test_simd_stats() {
    let stats = SimdStats::detect();
    assert_eq!(stats.batch_size, BATCH_SIZE);

    #[cfg(target_arch = "aarch64")]
    {
        assert!(stats.neon, "neon must be true on aarch64");
        assert!(stats.portable_simd);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        assert!(!stats.neon);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        assert!(
            stats.simd128,
            "simd128 must be true when target_feature=simd128"
        );
        assert!(stats.portable_simd);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        assert!(!stats.simd128);
    }
    #[cfg(target_arch = "x86_64")]
    {
        // On any x86_64 machine from 2013+ (Haswell), AVX2 should be present.
        // We only assert that the field is correctly populated (not a fixed value,
        // since CI runners may vary).
        let _ = stats.avx2;
        let _ = stats.sse41;
    }
}
