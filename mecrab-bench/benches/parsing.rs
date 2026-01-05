//! Comprehensive parsing benchmarks for MeCrab
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Run with: cargo bench --bench parsing
//!
//! This benchmark suite measures:
//! 1. Short text parsing (10-20 chars)
//! 2. Medium text parsing (100+ chars)
//! 3. Long text parsing (1000+ chars)
//! 4. Batch parsing throughput
//! 5. Dictionary loading time
//! 6. N-best path search (n=1, n=5, n=10)
//! 7. Lattice building vs Viterbi solving (separate measurements)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mecrab::MeCrab;
use mecrab::dict::Dictionary;
use mecrab::lattice::Lattice;
use mecrab::viterbi::ViterbiSolver;
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Test Data: Realistic Japanese Text Samples
// ============================================================================

/// Short texts (10-20 chars / ~30-60 bytes UTF-8)
/// Mix of hiragana-heavy and kanji-heavy samples
mod short_texts {
    /// Kanji-heavy: "Tokyo" (3 chars, 9 bytes)
    pub const TOKYO: &str = "東京都";
    /// Hiragana-heavy: "It is a plum" (6 chars, 18 bytes)
    pub const SUMOMO: &str = "すもももも";
    /// Mixed: "Test data" (5 chars, 15 bytes)
    pub const TEST_DATA: &str = "テストデータ";
    /// Kanji-heavy: "Morphological analysis" (5 chars, 15 bytes)
    pub const KEITAISO: &str = "形態素解析";
    /// Mixed with particles: "I am a cat" (5 chars, 15 bytes)
    pub const WAGAHAI: &str = "吾輩は猫だ";
    /// Hiragana only: "It is" (5 chars, 15 bytes)
    pub const DEARU: &str = "であります";
}

/// Medium texts (100+ chars / ~300+ bytes UTF-8)
/// Typical sentences one might encounter in real applications
mod medium_texts {
    /// Famous proverb about plums (12 chars, 36 bytes)
    pub const PLUM_PROVERB: &str = "すもももももももものうち";

    /// Kanji-heavy news-style sentence (about 50 chars)
    pub const NEWS_STYLE: &str =
        "東京都は本日、新型コロナウイルスの新規感染者数が過去最多を更新したと発表した。";

    /// Technical document style (about 45 chars)
    pub const TECHNICAL: &str =
        "形態素解析とは、文を最小の意味単位である形態素に分割し、品詞を付与する処理である。";

    /// Conversational style with hiragana (about 40 chars)
    pub const CONVERSATIONAL: &str = "今日はとてもいい天気ですね。どこかに出かけませんか。";

    /// Mixed formal style (about 55 chars)
    pub const FORMAL: &str = "本システムは、高速かつ正確な形態素解析を実現するために、最新のアルゴリズムを採用しております。";
}

/// Long texts (1000+ chars / ~3000+ bytes UTF-8)
/// Paragraphs for stress testing
mod long_texts {
    /// Opening of "I Am a Cat" by Natsume Soseki (classic literature)
    pub const WAGAHAI_NEKO: &str = concat!(
        "吾輩は猫である。名前はまだ無い。",
        "どこで生れたかとんと見当がつかぬ。",
        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",
        "吾輩はここで始めて人間というものを見た。",
        "しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。",
        "この書生というのは時々我々を捕えて煮て食うという話である。",
        "しかしその当時は何という考もなかったから別段恐しいとも思わなかった。",
        "ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。",
        "掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。",
        "この時妙なものだと思った感じが今でも残っている。",
        "第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。",
        "その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。",
        "のみならず顔の真中があまりに突起している。",
        "そうしてその穴の中から時々ぷうぷうと煙を吹く。",
        "どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。"
    );

    /// Wikipedia-style informational text about Tokyo
    pub const TOKYO_INFO: &str = concat!(
        "東京は日本の首都であり、世界有数の大都市である。",
        "人口は約1400万人で、政治・経済・文化の中心地として機能している。",
        "東京都は23の特別区、26の市、5つの町、8つの村から構成されている。",
        "江戸時代には江戸と呼ばれ、徳川幕府の所在地として栄えた。",
        "明治維新後、首都が京都から東京に移され、急速に近代化が進んだ。",
        "現在では、新宿、渋谷、池袋などの副都心が発展し、",
        "世界的な経済・金融の中心地となっている。",
        "東京スカイツリーや東京タワーなどのランドマークがあり、",
        "観光地としても人気が高い。",
        "交通網も発達しており、JR、私鉄、地下鉄などが縦横に走っている。",
        "羽田空港と成田空港から世界各地への便が就航している。",
        "2020年には東京オリンピック・パラリンピックが開催され、",
        "世界中から多くの注目を集めた。"
    );

    /// Technical documentation style
    pub const TECHNICAL_DOC: &str = concat!(
        "形態素解析は自然言語処理の基礎技術であり、",
        "テキストを最小の意味単位である形態素に分割する処理である。",
        "日本語の形態素解析では、辞書と文法規則に基づいて解析を行う。",
        "代表的な形態素解析器としてはMeCab、JUMAN、ChaSenなどがある。",
        "MeCrabはMeCab互換の形態素解析器であり、Rustで実装されている。",
        "SIMDによる高速化や、メモリマップドファイルによる効率的な辞書読み込みを特徴とする。",
        "ビタビアルゴリズムを用いて最適なパスを探索し、",
        "N-best探索にも対応している。",
        "辞書はIPADIC形式を採用しており、",
        "システム辞書に加えてオーバーレイ辞書による動的な単語追加が可能である。",
        "WebAssemblyビルドにも対応予定であり、",
        "ブラウザやエッジデバイスでの利用を想定している。",
        "Pythonバインディングも提供予定であり、",
        "データサイエンスや機械学習のワークフローへの統合を目指している。"
    );
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get the default dictionary path
fn get_dictionary_path() -> Option<&'static Path> {
    let paths = [
        "/var/lib/mecab/dic/ipadic-utf8",
        "/usr/lib/mecab/dic/ipadic-utf8",
        "/usr/local/lib/mecab/dic/ipadic-utf8",
        "/usr/share/mecab/dic/ipadic-utf8",
    ];

    for path in paths {
        let p = Path::new(path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Skip benchmark if dictionary not available
macro_rules! require_dictionary {
    ($dict_path:expr) => {
        match $dict_path {
            Some(p) => p,
            None => {
                eprintln!("IPADIC not found, skipping benchmark");
                return;
            }
        }
    };
}

// ============================================================================
// 1. Short Text Parsing Benchmarks (10-20 chars)
// ============================================================================

fn bench_short_text_parsing(c: &mut Criterion) {
    let dict_path = require_dictionary!(get_dictionary_path());
    let mecrab = MeCrab::builder()
        .dicdir(Some(dict_path.to_path_buf()))
        .build()
        .expect("Failed to load dictionary");

    let mut group = c.benchmark_group("short_text");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    // Kanji-heavy samples
    group.throughput(Throughput::Bytes(short_texts::TOKYO.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("kanji/tokyo", short_texts::TOKYO.len()),
        &short_texts::TOKYO,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    group.throughput(Throughput::Bytes(short_texts::KEITAISO.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("kanji/keitaiso", short_texts::KEITAISO.len()),
        &short_texts::KEITAISO,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    // Hiragana-heavy samples
    group.throughput(Throughput::Bytes(short_texts::SUMOMO.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("hiragana/sumomo", short_texts::SUMOMO.len()),
        &short_texts::SUMOMO,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    group.throughput(Throughput::Bytes(short_texts::DEARU.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("hiragana/dearu", short_texts::DEARU.len()),
        &short_texts::DEARU,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    // Mixed samples
    group.throughput(Throughput::Bytes(short_texts::TEST_DATA.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("mixed/test_data", short_texts::TEST_DATA.len()),
        &short_texts::TEST_DATA,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    group.throughput(Throughput::Bytes(short_texts::WAGAHAI.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("mixed/wagahai", short_texts::WAGAHAI.len()),
        &short_texts::WAGAHAI,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    group.finish();
}

// ============================================================================
// 2. Medium Text Parsing Benchmarks (100+ chars)
// ============================================================================

fn bench_medium_text_parsing(c: &mut Criterion) {
    let dict_path = require_dictionary!(get_dictionary_path());
    let mecrab = MeCrab::builder()
        .dicdir(Some(dict_path.to_path_buf()))
        .build()
        .expect("Failed to load dictionary");

    let mut group = c.benchmark_group("medium_text");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let texts = [
        ("plum_proverb", medium_texts::PLUM_PROVERB),
        ("news_style", medium_texts::NEWS_STYLE),
        ("technical", medium_texts::TECHNICAL),
        ("conversational", medium_texts::CONVERSATIONAL),
        ("formal", medium_texts::FORMAL),
    ];

    for (name, text) in texts {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new(name, text.len()), &text, |b, text| {
            b.iter(|| mecrab.parse(black_box(text)))
        });
    }

    group.finish();
}

// ============================================================================
// 3. Long Text Parsing Benchmarks (1000+ chars)
// ============================================================================

fn bench_long_text_parsing(c: &mut Criterion) {
    let dict_path = require_dictionary!(get_dictionary_path());
    let mecrab = MeCrab::builder()
        .dicdir(Some(dict_path.to_path_buf()))
        .build()
        .expect("Failed to load dictionary");

    let mut group = c.benchmark_group("long_text");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    let texts = [
        ("literature/wagahai_neko", long_texts::WAGAHAI_NEKO),
        ("wiki/tokyo_info", long_texts::TOKYO_INFO),
        ("technical/documentation", long_texts::TECHNICAL_DOC),
    ];

    for (name, text) in texts {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new(name, text.len()), &text, |b, text| {
            b.iter(|| mecrab.parse(black_box(text)))
        });
    }

    // Also test very long text by repeating
    let very_long = long_texts::WAGAHAI_NEKO.repeat(5);
    group.throughput(Throughput::Bytes(very_long.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("very_long/repeated_5x", very_long.len()),
        &very_long,
        |b, text| b.iter(|| mecrab.parse(black_box(text))),
    );

    group.finish();
}

// ============================================================================
// 4. Batch Parsing Throughput Benchmarks
// ============================================================================

fn bench_batch_parsing(c: &mut Criterion) {
    let dict_path = require_dictionary!(get_dictionary_path());
    let mecrab = MeCrab::builder()
        .dicdir(Some(dict_path.to_path_buf()))
        .build()
        .expect("Failed to load dictionary");

    let mut group = c.benchmark_group("batch");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));

    // Small batch (100 texts)
    let batch_100: Vec<&str> = vec![medium_texts::NEWS_STYLE; 100];
    let total_bytes_100: u64 = batch_100.iter().map(|s| s.len() as u64).sum();

    group.throughput(Throughput::Elements(100));
    group.bench_function("sequential/100_texts", |b| {
        b.iter(|| {
            for text in &batch_100 {
                let _ = black_box(mecrab.parse(black_box(text)));
            }
        })
    });

    group.throughput(Throughput::Bytes(total_bytes_100));
    group.bench_function("parse_batch/100_texts", |b| {
        b.iter(|| mecrab.parse_batch(black_box(&batch_100)))
    });

    // Medium batch (500 texts)
    let batch_500: Vec<&str> = vec![medium_texts::NEWS_STYLE; 500];
    let total_bytes_500: u64 = batch_500.iter().map(|s| s.len() as u64).sum();

    group.throughput(Throughput::Elements(500));
    group.sample_size(30);
    group.bench_function("sequential/500_texts", |b| {
        b.iter(|| {
            for text in &batch_500 {
                let _ = black_box(mecrab.parse(black_box(text)));
            }
        })
    });

    group.throughput(Throughput::Bytes(total_bytes_500));
    group.bench_function("parse_batch/500_texts", |b| {
        b.iter(|| mecrab.parse_batch(black_box(&batch_500)))
    });

    // Large batch (1000 texts) - target metric for "100ms for 1000 texts"
    let batch_1000: Vec<&str> = vec![medium_texts::NEWS_STYLE; 1000];
    let total_bytes_1000: u64 = batch_1000.iter().map(|s| s.len() as u64).sum();

    group.throughput(Throughput::Elements(1000));
    group.sample_size(20);
    group.bench_function("sequential/1000_texts", |b| {
        b.iter(|| {
            for text in &batch_1000 {
                let _ = black_box(mecrab.parse(black_box(text)));
            }
        })
    });

    group.throughput(Throughput::Bytes(total_bytes_1000));
    group.bench_function("parse_batch/1000_texts", |b| {
        b.iter(|| mecrab.parse_batch(black_box(&batch_1000)))
    });

    // Mixed batch (varied text lengths)
    let mixed_batch: Vec<&str> = [
        short_texts::TOKYO,
        short_texts::KEITAISO,
        medium_texts::NEWS_STYLE,
        medium_texts::TECHNICAL,
        long_texts::TOKYO_INFO,
    ]
    .iter()
    .cycle()
    .take(200)
    .copied()
    .collect();

    group.throughput(Throughput::Elements(mixed_batch.len() as u64));
    group.sample_size(50);
    group.bench_function("mixed/200_varied_texts", |b| {
        b.iter(|| mecrab.parse_batch(black_box(&mixed_batch)))
    });

    group.finish();
}

// ============================================================================
// 5. Dictionary Loading Time Benchmarks
// ============================================================================

fn bench_dictionary_loading(c: &mut Criterion) {
    let dict_path = require_dictionary!(get_dictionary_path());

    let mut group = c.benchmark_group("dictionary_load");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    // Cold load (measures full mmap + parsing)
    group.bench_function("cold_load", |b| {
        b.iter(|| Dictionary::load(black_box(dict_path)))
    });

    // MeCrab builder (includes dictionary loading)
    group.bench_function("mecrab_builder", |b| {
        b.iter(|| {
            MeCrab::builder()
                .dicdir(Some(dict_path.to_path_buf()))
                .build()
        })
    });

    group.finish();
}

// ============================================================================
// 6. N-Best Path Search Benchmarks (n=1, n=5, n=10)
// ============================================================================

fn bench_nbest_search(c: &mut Criterion) {
    let dict_path = require_dictionary!(get_dictionary_path());
    let mecrab = MeCrab::builder()
        .dicdir(Some(dict_path.to_path_buf()))
        .build()
        .expect("Failed to load dictionary");

    let mut group = c.benchmark_group("nbest");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    // Test with ambiguous text (plum proverb has multiple valid segmentations)
    let ambiguous_text = medium_texts::PLUM_PROVERB;

    // N=1 (equivalent to regular parse, baseline)
    group.bench_with_input(BenchmarkId::new("ambiguous", 1), &1usize, |b, &n| {
        b.iter(|| mecrab.parse_nbest(black_box(ambiguous_text), black_box(n)))
    });

    // N=5
    group.bench_with_input(BenchmarkId::new("ambiguous", 5), &5usize, |b, &n| {
        b.iter(|| mecrab.parse_nbest(black_box(ambiguous_text), black_box(n)))
    });

    // N=10
    group.bench_with_input(BenchmarkId::new("ambiguous", 10), &10usize, |b, &n| {
        b.iter(|| mecrab.parse_nbest(black_box(ambiguous_text), black_box(n)))
    });

    // Test with longer text
    let longer_text = medium_texts::NEWS_STYLE;

    group.bench_with_input(BenchmarkId::new("news_style", 1), &1usize, |b, &n| {
        b.iter(|| mecrab.parse_nbest(black_box(longer_text), black_box(n)))
    });

    group.bench_with_input(BenchmarkId::new("news_style", 5), &5usize, |b, &n| {
        b.iter(|| mecrab.parse_nbest(black_box(longer_text), black_box(n)))
    });

    group.bench_with_input(BenchmarkId::new("news_style", 10), &10usize, |b, &n| {
        b.iter(|| mecrab.parse_nbest(black_box(longer_text), black_box(n)))
    });

    // Compare N-best vs regular parse overhead
    group.bench_function("compare/parse_regular", |b| {
        b.iter(|| mecrab.parse(black_box(ambiguous_text)))
    });

    group.bench_function("compare/parse_nbest_1", |b| {
        b.iter(|| mecrab.parse_nbest(black_box(ambiguous_text), black_box(1)))
    });

    group.finish();
}

// ============================================================================
// 7. Lattice Building vs Viterbi Solving (Separate Measurements)
// ============================================================================

fn bench_lattice_vs_viterbi(c: &mut Criterion) {
    let dict_path = require_dictionary!(get_dictionary_path());
    let dictionary = Arc::new(Dictionary::load(dict_path).expect("Failed to load dictionary"));

    let mut group = c.benchmark_group("components");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let test_texts = [
        ("short", short_texts::TOKYO),
        ("medium", medium_texts::NEWS_STYLE),
        ("long", long_texts::TOKYO_INFO),
    ];

    for (label, text) in test_texts {
        // Measure lattice building only
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("lattice_build", label),
            &text,
            |b, text| b.iter(|| Lattice::build(black_box(text), &dictionary)),
        );

        // Measure Viterbi solving only (pre-build lattice)
        let lattice = Lattice::build(text, &dictionary).expect("Failed to build lattice");
        let solver = ViterbiSolver::new(&dictionary);

        group.bench_with_input(
            BenchmarkId::new("viterbi_solve", label),
            &lattice,
            |b, lattice| b.iter(|| solver.solve(black_box(lattice))),
        );

        // Measure combined (for reference)
        group.bench_with_input(BenchmarkId::new("combined", label), &text, |b, text| {
            b.iter(|| {
                let lat = Lattice::build(black_box(text), &dictionary).unwrap();
                solver.solve(&lat)
            })
        });
    }

    group.finish();
}

// ============================================================================
// 8. Scaling Analysis (Text Length Impact)
// ============================================================================

fn bench_scaling_analysis(c: &mut Criterion) {
    let dict_path = require_dictionary!(get_dictionary_path());
    let mecrab = MeCrab::builder()
        .dicdir(Some(dict_path.to_path_buf()))
        .build()
        .expect("Failed to load dictionary");

    let mut group = c.benchmark_group("scaling");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    // Generate texts of varying lengths by repeating base text
    let base = "東京都は日本の首都である。";
    let lengths = [1, 2, 4, 8, 16, 32];

    for multiplier in lengths {
        let text = base.repeat(multiplier);
        let byte_len = text.len();

        group.throughput(Throughput::Bytes(byte_len as u64));
        group.bench_with_input(
            BenchmarkId::new("linear_scaling", byte_len),
            &text,
            |b, text| b.iter(|| mecrab.parse(black_box(text))),
        );
    }

    group.finish();
}

// ============================================================================
// 9. Memory-Bound vs CPU-Bound Analysis
// ============================================================================

fn bench_cache_effects(c: &mut Criterion) {
    let dict_path = require_dictionary!(get_dictionary_path());
    let mecrab = MeCrab::builder()
        .dicdir(Some(dict_path.to_path_buf()))
        .build()
        .expect("Failed to load dictionary");

    let mut group = c.benchmark_group("cache_effects");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    // Repeated parsing of same text (should benefit from cache)
    let text = medium_texts::NEWS_STYLE;

    group.bench_function("repeated_same_text", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let _ = black_box(mecrab.parse(black_box(text)));
            }
        })
    });

    // Varied texts (more cache pressure)
    let varied_texts = [
        medium_texts::NEWS_STYLE,
        medium_texts::TECHNICAL,
        medium_texts::CONVERSATIONAL,
        medium_texts::FORMAL,
        medium_texts::PLUM_PROVERB,
    ];

    group.bench_function("varied_texts", |b| {
        b.iter(|| {
            for text in &varied_texts {
                let _ = black_box(mecrab.parse(black_box(text)));
                let _ = black_box(mecrab.parse(black_box(text)));
            }
        })
    });

    group.finish();
}

// ============================================================================
// Criterion Groups and Main
// ============================================================================

criterion_group!(
    name = short_text_benches;
    config = Criterion::default();
    targets = bench_short_text_parsing
);

criterion_group!(
    name = medium_text_benches;
    config = Criterion::default();
    targets = bench_medium_text_parsing
);

criterion_group!(
    name = long_text_benches;
    config = Criterion::default();
    targets = bench_long_text_parsing
);

criterion_group!(
    name = batch_benches;
    config = Criterion::default();
    targets = bench_batch_parsing
);

criterion_group!(
    name = dictionary_benches;
    config = Criterion::default();
    targets = bench_dictionary_loading
);

criterion_group!(
    name = nbest_benches;
    config = Criterion::default();
    targets = bench_nbest_search
);

criterion_group!(
    name = component_benches;
    config = Criterion::default();
    targets = bench_lattice_vs_viterbi
);

criterion_group!(
    name = analysis_benches;
    config = Criterion::default();
    targets = bench_scaling_analysis, bench_cache_effects
);

criterion_main!(
    short_text_benches,
    medium_text_benches,
    long_text_benches,
    batch_benches,
    dictionary_benches,
    nbest_benches,
    component_benches,
    analysis_benches
);
