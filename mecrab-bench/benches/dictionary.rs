//! Dictionary lookup benchmarks
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Run with: cargo bench --bench dictionary

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mecrab::dict::{Dictionary, OverlayDictionary};
use std::hint::black_box;
use std::path::Path;

/// Common Japanese words for benchmarking
const LOOKUP_WORDS: &[&str] = &[
    "の",
    "は",
    "が",
    "を",
    "に",
    "東京",
    "日本",
    "テスト",
    "すもも",
    "形態素解析",
];

fn dictionary_load_benchmark(c: &mut Criterion) {
    let path = Path::new("/var/lib/mecab/dic/ipadic-utf8");
    if !path.exists() {
        eprintln!("IPADIC not found, skipping dictionary benchmarks");
        return;
    }

    c.bench_function("dictionary_load", |b| {
        b.iter(|| Dictionary::load(black_box(path)))
    });
}

fn lookup_benchmark(c: &mut Criterion) {
    let path = Path::new("/var/lib/mecab/dic/ipadic-utf8");
    if !path.exists() {
        eprintln!("IPADIC not found, skipping dictionary benchmarks");
        return;
    }

    let dict = Dictionary::load(path).expect("Failed to load dictionary");

    let mut group = c.benchmark_group("lookup");

    for word in LOOKUP_WORDS {
        group.bench_with_input(BenchmarkId::from_parameter(word), word, |b, word| {
            b.iter(|| dict.lookup(black_box(word)))
        });
    }

    group.finish();
}

fn connection_cost_benchmark(c: &mut Criterion) {
    let path = Path::new("/var/lib/mecab/dic/ipadic-utf8");
    if !path.exists() {
        eprintln!("IPADIC not found, skipping dictionary benchmarks");
        return;
    }

    let dict = Dictionary::load(path).expect("Failed to load dictionary");

    let mut group = c.benchmark_group("connection_cost");
    group.throughput(Throughput::Elements(1000));

    // Benchmark 1000 random connection cost lookups
    group.bench_function("1000_lookups", |b| {
        b.iter(|| {
            for i in 0..1000u16 {
                let _ = dict.connection_cost(black_box(i % 1316), black_box((i * 7) % 1316));
            }
        })
    });

    group.finish();
}

fn char_info_benchmark(c: &mut Criterion) {
    let path = Path::new("/var/lib/mecab/dic/ipadic-utf8");
    if !path.exists() {
        eprintln!("IPADIC not found, skipping dictionary benchmarks");
        return;
    }

    let dict = Dictionary::load(path).expect("Failed to load dictionary");

    let test_chars: Vec<char> = "あいうえおアイウエオ漢字英語123ABCαβγ".chars().collect();

    let mut group = c.benchmark_group("char_info");
    group.throughput(Throughput::Elements(test_chars.len() as u64));

    group.bench_function("mixed_chars", |b| {
        b.iter(|| {
            for c in &test_chars {
                let _ = dict.char_info(black_box(*c));
            }
        })
    });

    group.finish();
}

fn overlay_benchmark(c: &mut Criterion) {
    let overlay = OverlayDictionary::new();

    // Add 1000 words to overlay
    for i in 0..1000 {
        let surface = format!("テスト単語{}", i);
        overlay.add_simple(&surface, "テストタンゴ", "テストタンゴ", 5000);
    }

    // Force trie rebuild
    let _ = overlay.lookup("テスト");

    let mut group = c.benchmark_group("overlay");
    group.throughput(Throughput::Elements(1));

    // Benchmark lookup with trie (should be O(m) where m is key length)
    group.bench_function("lookup_with_trie", |b| {
        b.iter(|| overlay.lookup(black_box("テスト単語500を含む文")))
    });

    // Benchmark add_word (marks trie dirty)
    group.bench_function("add_word", |b| {
        let mut counter = 0;
        b.iter(|| {
            let surface = format!("新規単語{}", counter);
            overlay.add_simple(black_box(&surface), "シンキタンゴ", "シンキタンゴ", 5000);
            counter += 1;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    dictionary_load_benchmark,
    lookup_benchmark,
    connection_cost_benchmark,
    char_info_benchmark,
    overlay_benchmark,
);

criterion_main!(benches);
