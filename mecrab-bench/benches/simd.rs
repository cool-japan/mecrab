//! SIMD vs scalar connection-cost benchmark for MeCrab
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Directly measures Phase 1.1 target: 5× improvement when using
//! `batch_connection_costs` (SIMD gather + widen) over a plain scalar loop
//! over a synthetic `i16` connection-matrix row.
//!
//! No real dictionary is needed — all data is generated in-process.
//!
//! Run with:
//!   cargo bench --bench simd -p mecrab-bench

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mecrab::batch_connection_costs;
use std::hint::black_box;

// ── Synthetic data generation ──────────────────────────────────────────────

/// Build a synthetic connection-matrix row of length `width`.
///
/// Values are a mix of positive and negative `i16` to exercise sign-extension
/// in the widening path.
fn make_row(width: usize) -> Vec<i16> {
    (0..width)
        .map(|i| {
            let v = (i as i32 * 37 + 12345) % 65536;
            // Shift into signed i16 range, keeping some negatives
            (v as i16).wrapping_sub(16384)
        })
        .collect()
}

/// Build a right-id lookup table of `count` indices, all in-bounds for `row`.
fn make_right_ids(count: usize, row_len: usize) -> Vec<u16> {
    (0..count)
        .map(|i| ((i * 17 + 3) % row_len) as u16)
        .collect()
}

// ── Scalar baseline ────────────────────────────────────────────────────────

/// Plain scalar gather: index each right_id into the row and widen to i32.
///
/// This is exactly the loop that `batch_connection_costs` replaces.
/// Capped at 16 elements to match the SIMD path contract.
#[inline(never)]
fn scalar_gather(row: &[i16], right_ids: &[u16], out: &mut [i32]) -> usize {
    let count = right_ids.len().min(out.len()).min(16);
    let row_len = row.len();
    for i in 0..count {
        let rid = right_ids[i] as usize;
        out[i] = if rid < row_len {
            row[rid] as i32
        } else {
            i16::MAX as i32
        };
    }
    count
}

// ── Benchmark groups ───────────────────────────────────────────────────────

/// Compare scalar vs SIMD for different batch sizes (4, 8, 16 right-ids).
fn bench_batch_sizes(c: &mut Criterion) {
    const ROW_WIDTH: usize = 1024;
    let row = make_row(ROW_WIDTH);

    let mut group = c.benchmark_group("simd_vs_scalar/batch_size");
    group.throughput(Throughput::Elements(16)); // maximum per call

    for batch in [4usize, 8, 16] {
        let right_ids = make_right_ids(batch, ROW_WIDTH);
        let mut out = vec![0i32; 16];

        group.throughput(Throughput::Elements(batch as u64));

        group.bench_with_input(BenchmarkId::new("scalar", batch), &batch, |b, _| {
            b.iter(|| scalar_gather(black_box(&row), black_box(&right_ids), black_box(&mut out)))
        });

        group.bench_with_input(BenchmarkId::new("simd", batch), &batch, |b, _| {
            b.iter(|| {
                batch_connection_costs(black_box(&row), black_box(&right_ids), black_box(&mut out))
            })
        });
    }

    group.finish();
}

/// Throughput over a larger synthetic Viterbi inner-loop:
/// simulate 1 000 Viterbi nodes each performing one 16-way cost lookup.
fn bench_viterbi_inner_loop(c: &mut Criterion) {
    const ROW_WIDTH: usize = 2048;
    const NODES: usize = 1_000;

    let row = make_row(ROW_WIDTH);
    let right_ids = make_right_ids(16, ROW_WIDTH);
    let mut out = vec![0i32; 16];

    let mut group = c.benchmark_group("simd_vs_scalar/viterbi_inner_loop");
    group.throughput(Throughput::Elements(NODES as u64));

    group.bench_function("scalar", |b| {
        b.iter(|| {
            for _ in 0..NODES {
                scalar_gather(black_box(&row), black_box(&right_ids), black_box(&mut out));
            }
        })
    });

    group.bench_function("simd", |b| {
        b.iter(|| {
            for _ in 0..NODES {
                batch_connection_costs(black_box(&row), black_box(&right_ids), black_box(&mut out));
            }
        })
    });

    group.finish();
}

/// Row-width sensitivity: test narrow (128), medium (1024), wide (4096) rows.
fn bench_row_widths(c: &mut Criterion) {
    const BATCH: usize = 16;

    let mut group = c.benchmark_group("simd_vs_scalar/row_width");

    for row_width in [128usize, 1024, 4096] {
        let row = make_row(row_width);
        let right_ids = make_right_ids(BATCH, row_width);
        let mut out = vec![0i32; BATCH];

        group.throughput(Throughput::Elements(BATCH as u64));

        group.bench_with_input(BenchmarkId::new("scalar", row_width), &row_width, |b, _| {
            b.iter(|| scalar_gather(black_box(&row), black_box(&right_ids), black_box(&mut out)))
        });

        group.bench_with_input(BenchmarkId::new("simd", row_width), &row_width, |b, _| {
            b.iter(|| {
                batch_connection_costs(black_box(&row), black_box(&right_ids), black_box(&mut out))
            })
        });
    }

    group.finish();
}

// ── Criterion main ─────────────────────────────────────────────────────────

criterion_group!(
    name = batch_size_benches;
    config = Criterion::default();
    targets = bench_batch_sizes
);

criterion_group!(
    name = inner_loop_benches;
    config = Criterion::default();
    targets = bench_viterbi_inner_loop
);

criterion_group!(
    name = row_width_benches;
    config = Criterion::default();
    targets = bench_row_widths
);

criterion_main!(batch_size_benches, inner_loop_benches, row_width_benches);
