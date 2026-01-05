# SIMD Optimization for MeCrab

**Status:** ✅ Implemented with Portable SIMD (Nightly)

Technical document for SIMD-accelerated Viterbi algorithm.

---

## Current Implementation

The SIMD module (`mecrab/src/viterbi/simd.rs`) uses **Nightly Rust's Portable SIMD** (`std::simd`) for cross-platform vectorization.

### Requirements

```rust
#![feature(portable_simd)]  // In crate root (lib.rs)
```

Build with: `cargo +nightly build`

### Available Functions

```rust
use mecrab::viterbi::simd::*;

// Batch add costs to an array (8-wide SIMD)
pub fn batch_add_costs(costs: &mut [i32], addition: i32);

// Find minimum value and index
pub fn find_min(costs: &[i32]) -> Option<(usize, i32)>;

// Find best predecessor for path finding
pub fn find_best_predecessor(
    prev_costs: &[i32],
    connection_costs: &[i16],
) -> Option<(usize, i32)>;

// SIMD predecessor batch processing
pub fn predecessor_batch_simd(
    prev_costs: &[i32],
    connection_costs: &[i16],
) -> Vec<i32>;

// Non-SIMD predecessor batch (for matrix lookups)
pub fn predecessor_batch(
    prev_costs: &[i32],
    connection_matrix: &[i16],
    right_id: u16,
    matrix_width: usize,
) -> Vec<i32>;
```

---

## Platform Support

Portable SIMD automatically maps to the best available instruction set:

| Platform | SIMD Width | Backend | Status |
|----------|------------|---------|--------|
| x86_64 (AVX2) | 256-bit | 8x i32 | ✅ Auto |
| x86_64 (SSE2) | 128-bit | 4x i32 | ✅ Auto |
| ARM64 (NEON) | 128-bit | 4x i32 | ✅ Auto |
| WASM (SIMD128) | 128-bit | 4x i32 | ✅ Auto |
| Other | Scalar | 1x i32 | ✅ Fallback |

---

## Portable SIMD Implementation

### Batch Cost Addition

```rust
use std::simd::prelude::*;
use std::simd::num::SimdInt;

pub fn batch_add_costs(costs: &mut [i32], addition: i32) {
    let add_vec = i32x8::splat(addition);
    let chunks = costs.len() / 8;

    for i in 0..chunks {
        let start = i * 8;
        let values = i32x8::from_slice(&costs[start..start + 8]);
        let result = values.saturating_add(add_vec);
        costs[start..start + 8].copy_from_slice(result.as_array());
    }

    // Handle remainder
    for cost in costs[(chunks * 8)..].iter_mut() {
        *cost = cost.saturating_add(addition);
    }
}
```

### Minimum Finding

```rust
pub fn find_min(costs: &[i32]) -> Option<(usize, i32)> {
    // SIMD horizontal minimum reduction
    let first_chunk = i32x8::from_slice(&costs[0..8]);
    let mut min_vals = first_chunk;

    for chunk in costs[8..].chunks_exact(8) {
        let values = i32x8::from_slice(chunk);
        min_vals = min_vals.simd_min(values);
    }

    // Reduce 8-lane vector to scalar
    let arr = min_vals.to_array();
    let simd_min = *arr.iter().min()?;

    // Find actual index in original array
    costs.iter().position(|&c| c == simd_min)
        .map(|idx| (idx, simd_min))
}
```

---

## Expected Speedup

| Operation | Scalar | SIMD (8-wide) | Speedup |
|-----------|--------|---------------|---------|
| Cost addition | 1 per iter | 8 per iter | 8x |
| Min finding | O(n) | O(n/8) | 8x |
| **Overall Viterbi** | - | - | **4-6x** |

*Overhead from lane reduction and remainder handling reduces theoretical 8x to practical 4-6x.*

---

## SIMD Statistics

```rust
use mecrab::viterbi::simd::SimdStats;

let stats = SimdStats::detect();
println!("Portable SIMD: {}", stats.portable_simd);
println!("Batch size: {}", stats.batch_size);  // 8
```

---

## Benchmarking

```bash
# Run SIMD benchmarks with nightly
cargo +nightly bench

# Profile with perf
perf record cargo +nightly bench viterbi
perf report
```

---

## Future Work

### Struct of Arrays (SoA) Layout

Convert from AoS to SoA for better cache utilization:

```rust
// Current: Array of Structs
struct ViterbiEntry { cost: i64, prev: usize, ... }
Vec<ViterbiEntry>

// Proposed: Struct of Arrays
struct ViterbiColumn {
    costs: Vec<i64>,      // Contiguous for SIMD
    prevs: Vec<u32>,      // Contiguous
}
```

### Wider SIMD (AVX-512)

Portable SIMD supports wider types:

```rust
use std::simd::i32x16;  // 16-wide for AVX-512

fn batch_add_costs_wide(costs: &mut [i32], addition: i32) {
    let add_vec = i32x16::splat(addition);
    // Process 16 elements at a time on AVX-512 hardware
}
```

### Batched Connection Matrix Lookup

```rust
fn cost_batch_8(&self, right_ids: [u16; 8], left_id: u16) -> [i16; 8];
```

---

*Copyright 2026 COOLJAPAN OU (Team KitaSan)*
