//! Scalar fallback implementations for SIMD cost calculations.
//!
//! These are the reference implementations used on non-SIMD platforms and as
//! correctness references in unit tests.

/// Scalar find_min: iterates over all elements, returns (index, min_value).
///
/// Active code path on non-aarch64/x86_64 targets (or when no SIMD feature
/// is available); reference implementation used in unit tests.
#[inline]
pub fn find_min(costs: &[i32]) -> Option<(usize, i32)> {
    if costs.is_empty() {
        return None;
    }
    let mut min_idx = 0usize;
    let mut min_val = costs[0];
    for (i, &cost) in costs.iter().enumerate().skip(1) {
        if cost < min_val {
            min_val = cost;
            min_idx = i;
        }
    }
    Some((min_idx, min_val))
}

/// Scalar find_best_predecessor.
///
/// Active code path on non-aarch64/x86_64 targets; reference implementation
/// used in unit tests.
#[allow(dead_code)]
#[inline]
pub fn find_best_predecessor(prev_costs: &[i32], conn_costs: &[i16]) -> Option<(usize, i32)> {
    let len = prev_costs.len().min(conn_costs.len());
    if len == 0 {
        return None;
    }

    let mut best_idx = 0usize;
    let mut best_cost = prev_costs[0].saturating_add(conn_costs[0] as i32);

    for i in 1..len {
        let cost = prev_costs[i].saturating_add(conn_costs[i] as i32);
        if cost < best_cost {
            best_cost = cost;
            best_idx = i;
        }
    }

    Some((best_idx, best_cost))
}

// Keep old names as aliases for compatibility with aarch64 test references.
// These are used in cfg-gated test blocks; allow dead_code unconditionally.

/// Alias for [`find_min`] returning a tuple; used in platform-specific tests.
#[allow(dead_code)]
#[inline]
pub fn find_min_scalar(costs: &[i32]) -> (usize, i32) {
    find_min(costs).unwrap_or((0, i32::MAX))
}

/// Alias for [`find_best_predecessor`]; used in platform-specific tests.
#[allow(dead_code)]
#[inline]
pub fn find_best_predecessor_scalar(
    prev_costs: &[i32],
    conn_costs: &[i16],
) -> Option<(usize, i32)> {
    find_best_predecessor(prev_costs, conn_costs)
}

/// Scalar reference implementation of `batch_min_argmin_i64`.
///
/// Computes `argmin_i(prev[i] + conn[i]) + wcost` over a chunk of up to 16
/// predecessors and returns `Some((index, min_total))` when the minimum total
/// is strictly less than `best_so_far`, or `None` otherwise.
///
/// This is the platform-independent reference used by the dispatch layer on
/// non-x86/non-aarch64 targets, and also called directly by tests on all
/// platforms to provide a known-correct baseline.
#[inline]
pub fn batch_min_argmin_i64(
    prev: &[i64],
    conn: &[i32],
    wcost: i64,
    best_so_far: i64,
) -> Option<(usize, i64)> {
    let len = prev.len().min(conn.len());
    if len == 0 {
        return None;
    }

    let mut best_idx: usize = 0;
    // Initialise to best_so_far so we only report an improvement.
    let mut best_total: i64 = best_so_far;
    let mut found = false;

    for i in 0..len {
        let total = prev[i] + conn[i] as i64 + wcost;
        if total < best_total {
            best_total = total;
            best_idx = i;
            found = true;
        }
    }

    if found {
        Some((best_idx, best_total))
    } else {
        None
    }
}
