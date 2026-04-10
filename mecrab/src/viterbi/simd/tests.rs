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
