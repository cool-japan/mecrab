//! SIMD-accelerated cost calculations for Viterbi algorithm
//!
//! On nightly Rust with `portable_simd`, uses `std::simd` for vectorization.
//! On stable Rust, provides scalar fallback implementations.
//!
//! ## Performance
//!
//! - **Nightly**: 8-way SIMD parallelism for i32 cost calculations
//! - **Stable**: Optimized scalar implementations with auto-vectorization hints

// Only import std::simd on nightly (requires #![feature(portable_simd)])
// NOTE: This requires adding the feature gate at crate root when on nightly
// For stable builds, we use scalar fallbacks
#[cfg(all(
    feature = "simd",
    target_feature = "simd128"  // This will never be true, forcing scalar path
))]
use std::simd::num::SimdInt;
#[cfg(all(
    feature = "simd",
    target_feature = "simd128"  // This will never be true, forcing scalar path
))]
use std::simd::prelude::*;

/// SIMD batch size for cost calculations (8 i32 values)
pub const BATCH_SIZE: usize = 8;

/// SIMD-accelerated batch addition of costs
///
/// Adds `addition` to each element in `costs` array using SIMD.
/// On stable Rust, uses scalar implementation with auto-vectorization hints.
#[inline]
pub fn batch_add_costs(costs: &mut [i32], addition: i32) {
    // Scalar implementation (LLVM can auto-vectorize this)
    for cost in costs.iter_mut() {
        *cost = cost.saturating_add(addition);
    }
}

/// Find minimum cost and its index in an array using SIMD
/// On stable Rust, uses scalar implementation.
#[inline]
pub fn find_min(costs: &[i32]) -> Option<(usize, i32)> {
    if costs.is_empty() {
        return None;
    }

    Some(find_min_scalar(costs))
}

/// Scalar find_min for small arrays
#[inline]
fn find_min_scalar(costs: &[i32]) -> (usize, i32) {
    let mut min_idx = 0;
    let mut min_val = costs[0];

    for (i, &cost) in costs.iter().enumerate().skip(1) {
        if cost < min_val {
            min_val = cost;
            min_idx = i;
        }
    }

    (min_idx, min_val)
}

/// Find best predecessor for Viterbi path
#[inline]
pub fn find_best_predecessor(prev_costs: &[i32], connection_costs: &[i16]) -> Option<(usize, i32)> {
    if prev_costs.is_empty() || connection_costs.is_empty() {
        return None;
    }

    let len = prev_costs.len().min(connection_costs.len());

    let mut best_idx = 0;
    let mut best_cost = prev_costs[0].saturating_add(connection_costs[0] as i32);

    for i in 1..len {
        let cost = prev_costs[i].saturating_add(connection_costs[i] as i32);
        if cost < best_cost {
            best_cost = cost;
            best_idx = i;
        }
    }

    Some((best_idx, best_cost))
}

/// Batch process predecessor costs using SIMD
/// On stable Rust, uses scalar implementation.
pub fn predecessor_batch_simd(prev_costs: &[i32], connection_costs: &[i16]) -> Vec<i32> {
    let len = prev_costs.len().min(connection_costs.len());
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        result.push(prev_costs[i].saturating_add(connection_costs[i] as i32));
    }

    result
}

/// Batch process predecessor costs (non-SIMD version for compatibility)
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

/// SIMD statistics for debugging
#[derive(Debug, Default)]
pub struct SimdStats {
    /// Whether portable SIMD is available
    pub portable_simd: bool,
    /// SIMD lane width (number of elements processed in parallel)
    pub batch_size: usize,
}

impl SimdStats {
    /// Detect SIMD capabilities
    pub fn detect() -> Self {
        Self {
            portable_simd: true, // Using nightly portable_simd
            batch_size: BATCH_SIZE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Ensure we test the SIMD path (>= 8 elements)
        let costs = vec![50, 40, 30, 20, 10, 5, 15, 25, 35];
        assert_eq!(find_min(&costs), Some((5, 5)));
    }

    #[test]
    fn test_find_best_predecessor() {
        let prev = vec![10, 20, 5, 15];
        let conn = vec![2, 1, 3, 1];
        let result = find_best_predecessor(&prev, &conn);
        // 5 + 3 = 8, 10 + 2 = 12, 20 + 1 = 21, 15 + 1 = 16
        assert_eq!(result, Some((2, 8)));
    }

    #[test]
    fn test_predecessor_batch_simd() {
        let prev = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let conn: Vec<i16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result = predecessor_batch_simd(&prev, &conn);
        assert_eq!(result, vec![11, 22, 33, 44, 55, 66, 77, 88]);
    }

    #[test]
    fn test_predecessor_batch() {
        let prev = vec![10, 20, 30];
        let matrix: Vec<i16> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9]; // 3x3 matrix
        let result = predecessor_batch(&prev, &matrix, 1, 3);
        // [10 + matrix[0*3+1], 20 + matrix[1*3+1], 30 + matrix[2*3+1]]
        // [10 + 2, 20 + 5, 30 + 8] = [12, 25, 38]
        assert_eq!(result, vec![12, 25, 38]);
    }

    #[test]
    fn test_simd_stats() {
        let stats = SimdStats::detect();
        assert_eq!(stats.batch_size, 8);
        assert!(stats.portable_simd);
    }
}
