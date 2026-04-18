//! Output format rendering benchmarks.
//!
//! Compares the time to render a pre-built `AnalysisResult` into different
//! output formats (CoNLL-U, JSON, Wakati, BPE-compatible, default).
//! No MeCab dictionary is required — synthetic morphemes are constructed directly.
//!
//! Benchmark groups:
//! - `formats/render_*` — single result with N morphemes
//! - `formats/batch_*`  — 100-result batch rendering

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mecrab::{AnalysisResult, Morpheme, OutputFormat};
use std::hint::black_box;

/// Build a synthetic Japanese-like morpheme.
fn make_morpheme(surface: &str, feature: &str, start: usize, end: usize) -> Morpheme {
    Morpheme {
        surface: surface.to_string(),
        feature: feature.to_string(),
        word_id: 0,
        pos_id: 0,
        wcost: 100,
        start_byte: start,
        end_byte: end,
        entities: Vec::new(),
        embedding: None,
        pronunciation: None,
    }
}

/// Build a realistic synthetic AnalysisResult of the given number of morphemes.
///
/// Uses real IPADIC-style feature strings for representative rendering overhead.
fn make_result(morpheme_count: usize, format: OutputFormat) -> AnalysisResult {
    // Representative Japanese morphemes with realistic feature strings
    let templates: &[(&str, &str)] = &[
        ("東京", "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ"),
        ("は", "助詞,係助詞,*,*,*,*,は,ハ,ワ"),
        ("日本", "名詞,固有名詞,地域,一般,*,*,日本,ニホン,ニホン"),
        ("の", "助詞,連体化,*,*,*,*,の,ノ,ノ"),
        ("首都", "名詞,一般,*,*,*,*,首都,シュト,シュト"),
        ("です", "助動詞,*,*,*,特殊・デス,基本形,です,デス,デス"),
        ("。", "記号,句点,*,*,*,*,。,。,。"),
        ("私", "名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ"),
        ("たち", "名詞,接尾,一般,*,*,*,たち,タチ,タチ"),
        ("が", "助詞,格助詞,一般,*,*,*,が,ガ,ガ"),
        ("住ん", "動詞,自立,*,*,五段・マ行,連用タ接続,住む,スン,スン"),
        ("で", "助詞,接続助詞,*,*,*,*,で,デ,デ"),
        ("いる", "動詞,非自立,*,*,一段,基本形,いる,イル,イル"),
        ("街", "名詞,一般,*,*,*,*,街,マチ,マチ"),
        ("美しい", "形容詞,自立,*,*,形容詞・アウオ段,基本形,美しい,ウツクシイ,ウツクシイ"),
    ];

    let mut morphemes = Vec::with_capacity(morpheme_count + 1);
    let mut byte_offset = 0;

    for i in 0..morpheme_count {
        let (surface, feature) = templates[i % templates.len()];
        let start = byte_offset;
        let end = start + surface.len();
        morphemes.push(make_morpheme(surface, feature, start, end));
        byte_offset = end;
    }

    // Always end with EOS
    morphemes.push(make_morpheme("EOS", "BOS/EOS,*,*,*,*,*,*,*,*", byte_offset, byte_offset));

    AnalysisResult::new(morphemes, format)
}

/// Benchmark rendering a single AnalysisResult.
fn bench_format_render(c: &mut Criterion) {
    let morpheme_counts = [10usize, 50, 100];
    let formats: &[(&str, OutputFormat)] = &[
        ("default", OutputFormat::Default),
        ("wakati", OutputFormat::Wakati),
        ("json", OutputFormat::Json),
        ("bpe_compatible", OutputFormat::BpeCompatible),
        ("conllu", OutputFormat::ConllU),
    ];

    let mut group = c.benchmark_group("formats/render");

    for &n in &morpheme_counts {
        group.throughput(Throughput::Elements(n as u64));

        for (format_name, format) in formats {
            let result = make_result(n, *format);
            group.bench_with_input(
                BenchmarkId::new(*format_name, n),
                &n,
                |b, _| {
                    b.iter(|| {
                        let s = format!("{}", black_box(&result));
                        black_box(s);
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark rendering 100 AnalysisResults (batch simulation).
fn bench_format_batch(c: &mut Criterion) {
    const BATCH: usize = 100;
    const MORPHEMES: usize = 20; // morphemes per result

    let formats: &[(&str, OutputFormat)] = &[
        ("default", OutputFormat::Default),
        ("wakati", OutputFormat::Wakati),
        ("json", OutputFormat::Json),
        ("conllu", OutputFormat::ConllU),
    ];

    let mut group = c.benchmark_group("formats/batch");
    group.throughput(Throughput::Elements((BATCH * MORPHEMES) as u64));

    for (format_name, format) in formats {
        let batch: Vec<AnalysisResult> = (0..BATCH)
            .map(|_| make_result(MORPHEMES, *format))
            .collect();

        group.bench_function(*format_name, |b| {
            b.iter(|| {
                let mut total_len = 0usize;
                for result in black_box(&batch) {
                    let s = format!("{result}");
                    total_len += s.len();
                }
                black_box(total_len);
            })
        });
    }

    group.finish();
}

criterion_group!(format_benches, bench_format_render, bench_format_batch);
criterion_main!(format_benches);
