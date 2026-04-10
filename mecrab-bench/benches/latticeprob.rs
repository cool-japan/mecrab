//! Forward-backward algorithm benchmarks
//!
//! Benchmarks the overhead of the forward-backward algorithm (for LatticeProb
//! and BpeCompatible outputs) compared to plain Viterbi parsing.
//!
//! These benchmarks require a MeCab IPADIC dictionary to be installed.
//! Run with: cargo bench -p mecrab-bench --bench latticeprob
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mecrab::MeCrab;
use std::hint::black_box;

fn load_mecrab() -> Option<MeCrab> {
    MeCrab::builder().build().ok()
}

fn bench_viterbi_vs_forward_backward(c: &mut Criterion) {
    let Some(mecrab) = load_mecrab() else {
        eprintln!("Skipping latticeprob benchmarks: no MeCab dictionary installed");
        return;
    };

    let texts = [
        ("short", "東京は日本の首都です"),
        (
            "medium",
            "日本語の形態素解析は自然言語処理の基礎的な技術であり、テキストを意味のある最小単位に分割する処理です",
        ),
        (
            "long",
            "日本は東アジアに位置する島国で、首都は東京です。人口は約1億2600万人で、世界第3位の経済大国です。日本語が公用語であり、独自の文字体系（平仮名、片仮名、漢字）を持ちます",
        ),
    ];

    let mut group = c.benchmark_group("viterbi_vs_forward_backward");

    for (name, text) in &texts {
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::new("viterbi", name), text, |b, t| {
            b.iter(|| mecrab.parse(black_box(t)))
        });

        group.bench_with_input(BenchmarkId::new("forward_backward", name), text, |b, t| {
            b.iter(|| mecrab.parse_with_probs(black_box(t)))
        });
    }

    group.finish();
}

fn bench_output_formats(c: &mut Criterion) {
    let Some(mecrab) = load_mecrab() else {
        eprintln!("Skipping output_formats benchmarks: no MeCab dictionary installed");
        return;
    };

    let text = "日本語の形態素解析エンジンMeCrabは高速で正確な解析を提供します";

    let mut group = c.benchmark_group("output_formats");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("default_format", |b| {
        b.iter(|| {
            let result = mecrab.parse(black_box(text)).unwrap();
            format!("{result}")
        })
    });

    group.bench_function("wakati_format", |b| {
        b.iter(|| mecrab.wakati(black_box(text)))
    });

    group.bench_function("lattice_prob", |b| {
        b.iter(|| mecrab.parse_with_probs(black_box(text)))
    });

    group.finish();
}

fn bench_forward_backward_scaling(c: &mut Criterion) {
    let Some(mecrab) = load_mecrab() else {
        eprintln!("Skipping forward_backward_scaling benchmarks: no MeCab dictionary installed");
        return;
    };

    // Texts of increasing length to characterise O(n) scaling of forward-backward.
    let texts: &[(&str, &str)] = &[
        ("len_10", "東京は首都です"),
        (
            "len_50",
            "自然言語処理（しぜんげんごしょり、英: natural language processing）は、人間が日常的に使っている言語をコンピュータで処理する技術です",
        ),
        (
            "len_100",
            "日本語の形態素解析は、テキストを形態素（意味を持つ最小の言語単位）に分割し、それぞれの品詞や読み方などの情報を付与する処理です。MeCabやJUMANなどのツールが広く使われており、自然言語処理の前処理ステップとして欠かせない技術です。",
        ),
    ];

    let mut group = c.benchmark_group("forward_backward_scaling");

    for (name, text) in texts {
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::new("viterbi_only", name), text, |b, t| {
            b.iter(|| mecrab.parse(black_box(t)))
        });

        group.bench_with_input(BenchmarkId::new("viterbi_plus_fb", name), text, |b, t| {
            b.iter(|| mecrab.parse_with_probs(black_box(t)))
        });
    }

    group.finish();
}

criterion_group!(
    name = viterbi_vs_fb;
    config = Criterion::default();
    targets = bench_viterbi_vs_forward_backward
);
criterion_group!(
    name = output_formats;
    config = Criterion::default();
    targets = bench_output_formats
);
criterion_group!(
    name = fb_scaling;
    config = Criterion::default();
    targets = bench_forward_backward_scaling
);
criterion_main!(viterbi_vs_fb, output_formats, fb_scaling);
