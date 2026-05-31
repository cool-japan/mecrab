//! Train command - dictionary cost training from a MeCab TSV corpus.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use crate::commands::parse::DictFormatArg;
use clap::Args;
use mecrab::{DictTrainConfig, GoldSegmentation, IpadicProvider, MeCrab, NeologdProvider, UnidicProvider};
use std::path::PathBuf;

/// Arguments for the `kizame train` subcommand.
#[derive(Args, Clone)]
pub struct TrainArgs {
    /// Path to the MeCab dictionary directory
    #[arg(short = 'd', long, required = true)]
    pub dicdir: PathBuf,

    /// Dictionary format: ipadic, unidic, neologd, auto (default: auto)
    #[arg(long, value_enum, default_value_t = DictFormatArg::Auto)]
    pub dict_format: DictFormatArg,

    /// Path to user dictionary
    #[arg(short = 'u', long)]
    pub userdic: Option<PathBuf>,

    /// Path to the annotated training corpus (MeCab TSV format)
    #[arg(short = 'c', long, required = true)]
    pub corpus: PathBuf,

    /// Number of training epochs
    #[arg(long, default_value_t = 10)]
    pub epochs: usize,

    /// Learning rate for gradient updates
    #[arg(long, default_value_t = 0.01)]
    pub learning_rate: f64,

    /// Mini-batch size (number of sentences per update step)
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,

    /// L2 regularization strength
    #[arg(long = "l2", default_value_t = 1e-5)]
    pub l2_strength: f64,

    /// Output path for the trained matrix.bin (default: <dicdir>/matrix.bin)
    #[arg(long)]
    pub output_matrix: Option<PathBuf>,

    /// Print per-epoch statistics during training
    #[arg(long)]
    pub verbose: bool,
}

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

    // Split the corpus into sentence blocks on "EOS" boundary lines.
    // We accumulate lines until we hit an EOS, then process the block.
    let mut block_lines: Vec<&str> = Vec::new();

    for raw_line in corpus_text.lines() {
        let trimmed = raw_line.trim();
        if trimmed == "EOS" {
            // Process the accumulated block (may be empty — skip).
            if !block_lines.is_empty() {
                // Reconstruct the original surface text by concatenating the
                // first tab-delimited field of each line.
                let text: String = block_lines
                    .iter()
                    .filter_map(|l| l.split('\t').next())
                    .collect();

                // Re-join the block lines into the TSV blob expected by
                // `GoldSegmentation::from_mecab_tsv`.
                let tsv: String = block_lines.join("\n");

                if let Some(seg) = GoldSegmentation::from_mecab_tsv(&text, &tsv) {
                    gold_segs.push(seg);
                }
            }
            block_lines.clear();
        } else if !trimmed.is_empty() {
            // Non-EOS, non-blank line — add to the current block.
            block_lines.push(raw_line);
        }
        // Blank lines between blocks are silently ignored.
    }

    // Handle a corpus that is not terminated by a trailing EOS.
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
    let corpus: Vec<GoldSegmentation> = parse_mecab_tsv_corpus(&corpus_text);

    eprintln!("Loaded {} sentences from corpus.", corpus.len());

    // ------------------------------------------------------------------
    // 3. Build the training configuration.
    // ------------------------------------------------------------------
    let config = DictTrainConfig {
        learning_rate: args.learning_rate,
        batch_size: args.batch_size,
        epochs: args.epochs,
        l2_strength: args.l2_strength,
        verbose: args.verbose,
    };

    // ------------------------------------------------------------------
    // 4. Run training.
    // ------------------------------------------------------------------
    eprintln!("Training for {} epochs...", args.epochs);

    let (matrix, summary) = mecrab.train_dict(&corpus, &config)?;

    // ------------------------------------------------------------------
    // 5. Write the trained matrix to disk.
    // ------------------------------------------------------------------
    let output_path = args
        .output_matrix
        .clone()
        .unwrap_or_else(|| args.dicdir.join("matrix.bin"));

    matrix.write_binary(&output_path)?;

    // ------------------------------------------------------------------
    // 6. Print summary.
    // ------------------------------------------------------------------
    eprintln!();
    eprintln!("Training complete.");
    eprintln!("  Epochs          : {}", summary.total_epochs);
    eprintln!("  Total sentences : {}", summary.total_sentences);
    eprintln!("  Total tokens    : {}", summary.total_tokens);

    if let Some(last) = summary.epoch_stats.last() {
        eprintln!("  Final epoch loss: {:.6}", last.loss);
    }

    eprintln!("  Matrix written  : {:?}", output_path);

    Ok(())
}
