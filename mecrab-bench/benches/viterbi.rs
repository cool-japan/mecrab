//! Viterbi algorithm benchmarks
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Run with: cargo bench --bench viterbi

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mecrab::MeCrab;
use std::hint::black_box;

/// Test sentences of varying complexity
const SHORT_TEXT: &str = "テスト";
const MEDIUM_TEXT: &str = "すもももももももものうち";
const LONG_TEXT: &str = "東京は日本の首都であり、世界有数の大都市である。人口は約1400万人で、政治・経済・文化の中心地として機能している。";
const VERY_LONG_TEXT: &str = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。";

fn parse_benchmark(c: &mut Criterion) {
    let mecrab = MeCrab::new().expect("Failed to load dictionary");

    let mut group = c.benchmark_group("parse");

    // Short text (3 chars, 9 bytes)
    group.throughput(Throughput::Bytes(SHORT_TEXT.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("short", SHORT_TEXT.len()),
        SHORT_TEXT,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    // Medium text (12 chars, 36 bytes)
    group.throughput(Throughput::Bytes(MEDIUM_TEXT.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("medium", MEDIUM_TEXT.len()),
        MEDIUM_TEXT,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    // Long text (~100 chars)
    group.throughput(Throughput::Bytes(LONG_TEXT.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("long", LONG_TEXT.len()),
        LONG_TEXT,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    // Very long text (~200 chars)
    group.throughput(Throughput::Bytes(VERY_LONG_TEXT.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("very_long", VERY_LONG_TEXT.len()),
        VERY_LONG_TEXT,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    group.finish();
}

fn wakati_benchmark(c: &mut Criterion) {
    let mecrab = MeCrab::new().expect("Failed to load dictionary");

    let mut group = c.benchmark_group("wakati");

    group.bench_function("medium", |b| {
        b.iter(|| mecrab.wakati(black_box(MEDIUM_TEXT)))
    });

    group.bench_function("long", |b| b.iter(|| mecrab.wakati(black_box(LONG_TEXT))));

    group.finish();
}

fn batch_benchmark(c: &mut Criterion) {
    let mecrab = MeCrab::new().expect("Failed to load dictionary");

    // Create batch of 100 sentences
    let batch: Vec<&str> = vec![MEDIUM_TEXT; 100];

    let mut group = c.benchmark_group("batch");
    group.throughput(Throughput::Elements(batch.len() as u64));

    group.bench_function("sequential_100", |b| {
        b.iter(|| {
            for text in &batch {
                let _ = mecrab.parse(black_box(text));
            }
        })
    });

    // Test parse_batch (parallel when feature enabled)
    group.bench_function("parse_batch_100", |b| {
        b.iter(|| mecrab.parse_batch(black_box(&batch)))
    });

    group.finish();
}

fn large_batch_benchmark(c: &mut Criterion) {
    let mecrab = MeCrab::new().expect("Failed to load dictionary");

    // Create batch of 1000 sentences
    let batch: Vec<&str> = vec![MEDIUM_TEXT; 1000];

    let mut group = c.benchmark_group("large_batch");
    group.throughput(Throughput::Elements(batch.len() as u64));
    group.sample_size(20); // Reduce samples for long benchmarks

    group.bench_function("sequential_1000", |b| {
        b.iter(|| {
            for text in &batch {
                let _ = mecrab.parse(black_box(text));
            }
        })
    });

    group.bench_function("parse_batch_1000", |b| {
        b.iter(|| mecrab.parse_batch(black_box(&batch)))
    });

    group.finish();
}

criterion_group!(
    benches,
    parse_benchmark,
    wakati_benchmark,
    batch_benchmark,
    large_batch_benchmark,
);

criterion_main!(benches);
