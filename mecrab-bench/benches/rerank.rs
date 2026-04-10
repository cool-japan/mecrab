//! Reranker overhead benchmarks
//!
//! Measures the cost of NullReranker vs CostReranker for different N-best sizes.
//! These benchmarks don't require a real MeCab dictionary — all data is
//! generated synthetically in-process.
//!
//! Run with:
//!   cargo bench --bench rerank -p mecrab-bench
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mecrab::rerank::{CostReranker, NullReranker, RerankCandidate, Reranker};
use std::hint::black_box;

// ── Synthetic data generation ─────────────────────────────────────────────────

/// Build a synthetic N-best candidate list.
///
/// Candidates alternate between two surface splits of the same three-morpheme
/// span so that `CostReranker` must scan the entire list to find the minimum.
fn make_candidates(n: usize) -> Vec<RerankCandidate> {
    (0..n)
        .map(|i| RerankCandidate {
            surfaces: vec!["東京".to_string(), "は".to_string(), "日本".to_string()],
            pos_tags: vec!["名詞".to_string(), "助詞".to_string(), "名詞".to_string()],
            // Cost deliberately not sorted: minimum is at the last element to
            // force `CostReranker` to scan the whole slice.
            cost: (n as i64 - i as i64) * 100 + 500,
        })
        .collect()
}

// ── NullReranker benchmarks ───────────────────────────────────────────────────

fn bench_null_reranker(c: &mut Criterion) {
    let reranker = NullReranker;
    let mut group = c.benchmark_group("rerank_null");

    for n in [5usize, 10, 20] {
        let candidates = make_candidates(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &candidates, |b, cands| {
            b.iter(|| reranker.rerank(black_box(cands)))
        });
    }

    group.finish();
}

// ── CostReranker benchmarks ───────────────────────────────────────────────────

fn bench_cost_reranker(c: &mut Criterion) {
    let reranker = CostReranker;
    let mut group = c.benchmark_group("rerank_cost");

    for n in [5usize, 10, 20] {
        let candidates = make_candidates(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &candidates, |b, cands| {
            b.iter(|| reranker.rerank(black_box(cands)))
        });
    }

    group.finish();
}

// ── Direct comparison at N=10 ─────────────────────────────────────────────────

fn bench_reranker_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("rerank_comparison");
    let candidates = make_candidates(10);

    group.bench_function("null_n10", |b| {
        b.iter(|| NullReranker.rerank(black_box(&candidates)))
    });
    group.bench_function("cost_n10", |b| {
        b.iter(|| CostReranker.rerank(black_box(&candidates)))
    });

    group.finish();
}

// ── Trait-object dispatch overhead ───────────────────────────────────────────

fn bench_trait_object_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("rerank_trait_object");
    let candidates = make_candidates(10);

    let null_boxed: &dyn Reranker = &NullReranker;
    let cost_boxed: &dyn Reranker = &CostReranker;

    group.bench_function("null_dyn", |b| {
        b.iter(|| null_boxed.rerank(black_box(&candidates)))
    });
    group.bench_function("cost_dyn", |b| {
        b.iter(|| cost_boxed.rerank(black_box(&candidates)))
    });

    group.finish();
}

// ── Criterion main ────────────────────────────────────────────────────────────

criterion_group!(
    name = null_benches;
    config = Criterion::default();
    targets = bench_null_reranker
);
criterion_group!(
    name = cost_benches;
    config = Criterion::default();
    targets = bench_cost_reranker
);
criterion_group!(
    name = comparison_benches;
    config = Criterion::default();
    targets = bench_reranker_comparison
);
criterion_group!(
    name = dispatch_benches;
    config = Criterion::default();
    targets = bench_trait_object_dispatch
);

criterion_main!(
    null_benches,
    cost_benches,
    comparison_benches,
    dispatch_benches
);
