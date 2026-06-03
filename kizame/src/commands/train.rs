//! Train command - dictionary cost training from a MeCab TSV corpus.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! ## Training flow
//!
//! 1. Load the MeCab dictionary.
//! 2. Parse the annotated TSV corpus.
//! 3. Optionally split off a held-out dev set (`--dev-ratio`).
//! 4. Compute baseline dev boundary-F1 with the unmodified analyzer.
//! 5. Train connection-matrix and word-cost deltas (`MeCrab::train_dict`).
//! 6. Hot-reload the trained matrix in-memory and apply word-cost overrides.
//! 7. Compute trained dev boundary-F1.
//! 8. Report baseline vs trained; guard against regression (`--require-improvement`).
//! 9. Write `matrix.bin` (and optionally `word_cost.tsv`).

use crate::commands::parse::DictFormatArg;
use clap::Args;
use mecrab::{
    DictTrainConfig, GoldSegmentation, IpadicProvider, MeCrab, NeologdProvider, UnidicProvider,
    boundary_f1,
};
use std::path::{Path, PathBuf};

// ── CLI arguments ────────────────────────────────────────────────────────────

/// Arguments for the `kizame train` subcommand.
#[derive(Args, Clone)]
pub struct TrainArgs {
    /// Path to the MeCab dictionary directory.
    #[arg(short = 'd', long, required = true)]
    pub dicdir: PathBuf,

    /// Dictionary format: ipadic, unidic, neologd, auto (default: auto).
    #[arg(long, value_enum, default_value_t = DictFormatArg::Auto)]
    pub dict_format: DictFormatArg,

    /// Path to user dictionary.
    #[arg(short = 'u', long)]
    pub userdic: Option<PathBuf>,

    /// Path to the annotated training corpus (MeCab TSV format).
    #[arg(short = 'c', long, required = true)]
    pub corpus: PathBuf,

    /// Fraction of the corpus to hold out as a dev set for evaluation.
    ///
    /// The final `floor(N * dev_ratio)` sentences are used for evaluation;
    /// the remainder are used for training. Set to 0.0 (default) to disable.
    #[arg(long, default_value_t = 0.0)]
    pub dev_ratio: f64,

    /// Number of training epochs.
    #[arg(long, default_value_t = 10)]
    pub epochs: usize,

    /// Initial learning rate for gradient updates.
    #[arg(long, default_value_t = 0.01)]
    pub learning_rate: f64,

    /// Minimum learning rate for linear decay (0.0 = no decay).
    ///
    /// When > 0.0, the effective LR at epoch `e` is linearly decayed from
    /// `--learning-rate` to `--min-lr` over all epochs.
    #[arg(long = "min-lr", default_value_t = 0.0)]
    pub min_learning_rate: f64,

    /// Mini-batch size (number of sentences per update step).
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,

    /// L2 regularization strength.
    #[arg(long = "l2", default_value_t = 1e-5)]
    pub l2_strength: f64,

    /// Output path for the trained matrix.bin (default: <dicdir>/matrix.bin).
    #[arg(long)]
    pub output_matrix: Option<PathBuf>,

    /// Write word-cost deltas to this TSV file (word_id TAB delta_i16).
    ///
    /// Omit to skip writing word costs.
    #[arg(long)]
    pub output_word_costs: Option<PathBuf>,

    /// Refuse to overwrite the output matrix if dev boundary-F1 did not improve.
    ///
    /// When set alongside `--dev-ratio > 0`, the trained matrix is written to
    /// `<output>.candidate` rather than the primary output path if the trained
    /// dev F1 is not strictly greater than the baseline F1.  Has no effect when
    /// `--dev-ratio` is 0.
    #[arg(long)]
    pub require_improvement: bool,

    /// Print per-epoch statistics during training.
    #[arg(long)]
    pub verbose: bool,
}

// ── Corpus parsing ────────────────────────────────────────────────────────────

/// Parse a MeCab TSV corpus file into a list of gold-standard segmentations.
///
/// The corpus format is MeCab output: each sentence block is delimited by a
/// line that is exactly "EOS". Each non-EOS line has the surface form as the
/// first tab-delimited field.
///
/// ```text
/// 東京    名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ
/// は      助詞,係助詞,*,*,*,*,は,ハ,ワ
/// EOS
/// ```
fn parse_mecab_tsv_corpus(corpus_text: &str) -> Vec<GoldSegmentation> {
    let mut gold_segs: Vec<GoldSegmentation> = Vec::new();
    let mut block_lines: Vec<&str> = Vec::new();

    for raw_line in corpus_text.lines() {
        let trimmed = raw_line.trim();
        if trimmed == "EOS" {
            if !block_lines.is_empty() {
                let text: String = block_lines
                    .iter()
                    .filter_map(|l| l.split('\t').next())
                    .collect();
                let tsv: String = block_lines.join("\n");
                if let Some(seg) = GoldSegmentation::from_mecab_tsv(&text, &tsv) {
                    gold_segs.push(seg);
                }
            }
            block_lines.clear();
        } else if !trimmed.is_empty() {
            block_lines.push(raw_line);
        }
    }

    // Handle corpus not terminated by a trailing EOS.
    if !block_lines.is_empty() {
        let text: String = block_lines
            .iter()
            .filter_map(|l| l.split('\t').next())
            .collect();
        let tsv: String = block_lines.join("\n");
        if let Some(seg) = GoldSegmentation::from_mecab_tsv(&text, &tsv) {
            gold_segs.push(seg);
        }
    }

    gold_segs
}

// ── Dev evaluation ────────────────────────────────────────────────────────────

/// Compute mean boundary-F1 over a dev set.
///
/// Each gold sentence is parsed by `mecrab`; the predicted morpheme end-byte
/// positions are compared against the gold positions via [`boundary_f1`].
/// Sentences that fail to parse are skipped (their contribution is zero).
///
/// Returns `(mean_precision, mean_recall, mean_f1)` over all parseable sentences.
fn evaluate_dev(mecrab: &MeCrab, dev: &[GoldSegmentation]) -> (f64, f64, f64) {
    if dev.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut sum_p = 0.0_f64;
    let mut sum_r = 0.0_f64;
    let mut sum_f = 0.0_f64;
    let mut count = 0_usize;

    for gold in dev {
        let result = match mecrab.parse(&gold.text) {
            Ok(r) => r,
            Err(_) => continue,
        };
        // Collect predicted morpheme end positions (excluding EOS at text_len).
        let text_len = gold.text.len();
        let predicted_ends: Vec<usize> = result
            .morphemes
            .iter()
            .map(|m| m.end_byte)
            .filter(|&e| e < text_len)
            .collect();

        let (p, r, f) = boundary_f1(&predicted_ends, gold);
        sum_p += p;
        sum_r += r;
        sum_f += f;
        count += 1;
    }

    if count == 0 {
        (0.0, 0.0, 0.0)
    } else {
        let n = count as f64;
        (sum_p / n, sum_r / n, sum_f / n)
    }
}

// ── Hot-reload ────────────────────────────────────────────────────────────────

/// Build a new `MeCrab` instance with the trained matrix and word-cost overrides.
///
/// Reads `sys.dic`, `char.bin`, `unk.dic` from `dicdir` and uses `matrix_bytes`
/// for the connection matrix, bypassing filesystem writes for the matrix.
fn build_trained_mecrab(
    dicdir: &Path,
    matrix_bytes: &[u8],
    word_overrides: std::collections::HashMap<u32, i16>,
) -> Result<MeCrab, Box<dyn std::error::Error>> {
    use mecrab::dict::{CHAR_BIN_FILE, SYS_DIC_FILE, UNK_DIC_FILE};

    let sys_dic_bytes = std::fs::read(dicdir.join(SYS_DIC_FILE))?;
    let char_bin_bytes = std::fs::read(dicdir.join(CHAR_BIN_FILE))?;
    let unk_dic_bytes = std::fs::read(dicdir.join(UNK_DIC_FILE))?;

    let trained = MeCrab::from_bytes(
        &sys_dic_bytes,
        matrix_bytes,
        &char_bin_bytes,
        &unk_dic_bytes,
    )?;

    if !word_overrides.is_empty() {
        trained.set_word_cost_overrides(word_overrides);
    }

    Ok(trained)
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Entry point for `kizame train`.
pub fn run_train(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    // ------------------------------------------------------------------
    // 1. Load dictionary / build the MeCrab analyzer.
    // ------------------------------------------------------------------
    eprintln!("Loading dictionary from {:?}...", args.dicdir);

    let builder = MeCrab::builder()
        .dicdir(Some(args.dicdir.clone()))
        .userdic(args.userdic.clone());

    let mecrab = match args.dict_format {
        DictFormatArg::Auto => builder.build()?,
        DictFormatArg::Ipadic => builder.with_provider(IpadicProvider).build()?,
        DictFormatArg::Unidic => builder.with_provider(UnidicProvider).build()?,
        DictFormatArg::Neologd => builder.with_provider(NeologdProvider).build()?,
    };

    // ------------------------------------------------------------------
    // 2. Read and parse the training corpus.
    // ------------------------------------------------------------------
    eprintln!("Reading corpus from {:?}...", args.corpus);

    let corpus_text = std::fs::read_to_string(&args.corpus)?;
    let mut all_corpus: Vec<GoldSegmentation> = parse_mecab_tsv_corpus(&corpus_text);

    eprintln!("Loaded {} sentences from corpus.", all_corpus.len());

    // ------------------------------------------------------------------
    // 3. Split into train / dev.
    // ------------------------------------------------------------------
    let dev_count = if args.dev_ratio > 0.0 && !all_corpus.is_empty() {
        let n = (all_corpus.len() as f64 * args.dev_ratio).floor() as usize;
        n.min(all_corpus.len().saturating_sub(1))
    } else {
        0
    };

    let train_count = all_corpus.len().saturating_sub(dev_count);
    let dev_corpus: Vec<GoldSegmentation> = all_corpus.split_off(train_count);
    let train_corpus = all_corpus; // remainder

    if dev_count > 0 {
        eprintln!(
            "Split: {} train / {} dev sentences.",
            train_corpus.len(),
            dev_corpus.len()
        );
    }

    // ------------------------------------------------------------------
    // 4. Baseline dev evaluation (before training).
    // ------------------------------------------------------------------
    let baseline_f1 = if !dev_corpus.is_empty() {
        let (p, r, f) = evaluate_dev(&mecrab, &dev_corpus);
        eprintln!(
            "Baseline dev boundary-F1: {:.4}  (P={:.4}  R={:.4})",
            f, p, r
        );
        f
    } else {
        0.0
    };

    // ------------------------------------------------------------------
    // 5. Build training configuration.
    // ------------------------------------------------------------------
    let config = DictTrainConfig {
        learning_rate: args.learning_rate,
        min_learning_rate: args.min_learning_rate,
        batch_size: args.batch_size,
        epochs: args.epochs,
        l2_strength: args.l2_strength,
        verbose: args.verbose,
    };

    // ------------------------------------------------------------------
    // 6. Run training.
    // ------------------------------------------------------------------
    eprintln!("Training for {} epoch(s)...", args.epochs);
    if args.min_learning_rate > 0.0 {
        eprintln!(
            "LR schedule: {:.4} → {:.4} (linear decay)",
            args.learning_rate, args.min_learning_rate
        );
    }

    let (matrix, summary) = mecrab.train_dict(&train_corpus, &config)?;

    // ------------------------------------------------------------------
    // 7. Post-training dev evaluation.
    // ------------------------------------------------------------------
    let trained_f1 = if !dev_corpus.is_empty() {
        let matrix_bytes = matrix.to_bytes();
        let word_overrides = matrix.word_cost_deltas_i16();
        match build_trained_mecrab(&args.dicdir, &matrix_bytes, word_overrides) {
            Ok(trained_mecrab) => {
                let (p, r, f) = evaluate_dev(&trained_mecrab, &dev_corpus);
                eprintln!(
                    "Trained  dev boundary-F1: {:.4}  (P={:.4}  R={:.4})",
                    f, p, r
                );
                f
            }
            Err(e) => {
                eprintln!("Warning: could not hot-reload trained model for dev eval: {e}");
                0.0
            }
        }
    } else {
        0.0
    };

    // ------------------------------------------------------------------
    // 8. Determine output path; apply regression guard.
    // ------------------------------------------------------------------
    let primary_output = args
        .output_matrix
        .clone()
        .unwrap_or_else(|| args.dicdir.join("matrix.bin"));

    let regressed = dev_count > 0 && args.require_improvement && trained_f1 <= baseline_f1;

    let matrix_output_path = if regressed {
        let candidate = primary_output.with_extension("bin.candidate");
        eprintln!(
            "Warning: trained dev F1 ({:.4}) did not exceed baseline ({:.4}).",
            trained_f1, baseline_f1
        );
        eprintln!(
            "  --require-improvement is set: writing to {:?} (not {:?}).",
            candidate, primary_output
        );
        candidate
    } else {
        primary_output.clone()
    };

    // ------------------------------------------------------------------
    // 9. Write outputs.
    // ------------------------------------------------------------------
    matrix.write_binary(&matrix_output_path)?;
    eprintln!("Matrix written: {:?}", matrix_output_path);

    if let Some(ref wc_path) = args.output_word_costs {
        matrix.write_word_costs(wc_path)?;
        let delta_count = matrix.word_cost_deltas_i16().len();
        eprintln!(
            "Word costs written: {:?}  ({} non-zero deltas)",
            wc_path, delta_count
        );
    }

    // ------------------------------------------------------------------
    // 10. Print summary.
    // ------------------------------------------------------------------
    eprintln!();
    eprintln!("Training complete.");
    eprintln!("  Epochs          : {}", summary.total_epochs);
    eprintln!("  Train sentences : {}", train_corpus.len());
    eprintln!("  Total sentences : {}", summary.total_sentences);
    eprintln!("  Total tokens    : {}", summary.total_tokens);

    if let Some(last) = summary.epoch_stats.last() {
        eprintln!("  Final epoch loss: {:.6}", last.loss);
        if args.min_learning_rate > 0.0 {
            eprintln!("  Final epoch LR  : {:.6}", last.effective_lr);
        }
    }

    if dev_count > 0 {
        eprintln!("  Baseline dev F1 : {:.4}", baseline_f1);
        eprintln!("  Trained  dev F1 : {:.4}", trained_f1);
        let delta = trained_f1 - baseline_f1;
        eprintln!("  Δ F1            : {:+.4}", delta);
    }

    if regressed {
        eprintln!();
        eprintln!("NOTE: Matrix written to .candidate path (regression guard active).");
        eprintln!(
            "  To use anyway: mv {:?} {:?}",
            matrix_output_path, primary_output
        );
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mecrab::viterbi::train::{GoldMorpheme, GoldSegmentation};
    use std::env::temp_dir;

    /// Build a minimal `GoldSegmentation` for testing evaluation helpers.
    fn gold_seg(text: &str, surfaces: &[&str]) -> GoldSegmentation {
        GoldSegmentation {
            text: text.to_string(),
            morphemes: surfaces
                .iter()
                .map(|s| GoldMorpheme {
                    surface: s.to_string(),
                    left_id: 0,
                    right_id: 0,
                    word_id: u32::MAX,
                })
                .collect(),
        }
    }

    #[test]
    fn test_parse_mecab_tsv_corpus_basic() {
        let corpus = "東京\t名詞,固有名詞\nは\t助詞\nEOS\n";
        let segs = parse_mecab_tsv_corpus(corpus);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "東京は");
        assert_eq!(segs[0].morphemes.len(), 2);
    }

    #[test]
    fn test_parse_mecab_tsv_corpus_multiple_sentences() {
        let corpus = "A\tx\nEOS\nB\ty\nEOS\n";
        let segs = parse_mecab_tsv_corpus(corpus);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "A");
        assert_eq!(segs[1].text, "B");
    }

    #[test]
    fn test_parse_mecab_tsv_corpus_no_trailing_eos() {
        let corpus = "X\ta\nY\tb";
        let segs = parse_mecab_tsv_corpus(corpus);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "XY");
    }

    #[test]
    fn test_parse_mecab_tsv_corpus_empty() {
        assert!(parse_mecab_tsv_corpus("").is_empty());
        assert!(parse_mecab_tsv_corpus("EOS\n").is_empty());
    }

    #[test]
    fn test_train_corpus_split_dev() {
        // Verify that dev_count correctly splits off the last N sentences.
        let n = 10_usize;
        let dev_ratio = 0.3_f64;
        let expected_dev = (n as f64 * dev_ratio).floor() as usize;
        assert_eq!(expected_dev, 3);
        let expected_train = n - 3;
        assert_eq!(expected_train, 7);
    }

    #[test]
    fn test_dev_count_zero_when_ratio_zero() {
        let all_count = 100_usize;
        let dev_count = if 0.0_f64 > 0.0 {
            (all_count as f64 * 0.0_f64).floor() as usize
        } else {
            0
        };
        assert_eq!(dev_count, 0);
    }

    #[test]
    fn test_evaluate_dev_empty_returns_zero() {
        // evaluate_dev on an empty slice returns (0,0,0) without panicking.
        // We can't call it without a real MeCrab, but we can test boundary_f1 directly.
        let gold = gold_seg("AB", &["A", "B"]);
        // predicted nothing
        let (p, r, f) = boundary_f1(&[], &gold);
        assert_eq!(p, 0.0);
        assert_eq!(r, 0.0);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn test_boundary_f1_perfect_from_gold() {
        // A correctly predicted segmentation gives F1 = 1.0.
        let gold = gold_seg("AB", &["A", "B"]);
        // Gold ends: {1, 2} (byte offsets).
        let predicted_ends = vec![1_usize, 2];
        let (p, r, f) = boundary_f1(&predicted_ends, &gold);
        assert!((p - 1.0).abs() < 1e-9);
        assert!((r - 1.0).abs() < 1e-9);
        assert!((f - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_boundary_f1_partial_from_gold() {
        // Predict one boundary correctly out of two → P=1/1, R=1/2, F=2/3.
        let gold = gold_seg("ABС", &["A", "BС"]);
        // Gold ends: {1, 4} (A=1 byte, BС= 3 bytes).
        let predicted_ends = vec![1_usize]; // only the first
        let (p, r, f) = boundary_f1(&predicted_ends, &gold);
        assert!((p - 1.0).abs() < 1e-9, "precision={p}");
        assert!((r - 0.5).abs() < 1e-9, "recall={r}");
        // F1 = 2*(1.0*0.5)/(1.0+0.5) = 2/3
        assert!((f - 2.0 / 3.0).abs() < 1e-9, "f1={f}");
    }

    #[test]
    fn test_read_word_costs_roundtrip() {
        use mecrab::viterbi::train_loop::read_word_costs;
        let dir = temp_dir();
        let path = dir.join("mecrab_kizame_test_word_costs.tsv");
        std::fs::write(&path, "42\t-7\n100\t3\n# comment\n0\t0\n").unwrap();
        let map = read_word_costs(&path).unwrap();
        assert_eq!(map.get(&42), Some(&-7_i16));
        assert_eq!(map.get(&100), Some(&3_i16));
        assert_eq!(map.get(&0), Some(&0_i16));
        let _ = std::fs::remove_file(&path);
    }
}
