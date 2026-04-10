//! Parse command - morphological analysis
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use crate::commands::{Format, colors, pos_color};
use clap::{Args, ValueEnum};
use mecrab::api::format::format_lattice_prob;
use mecrab::{IpadicProvider, MeCrab, NeologdProvider, OutputFormat, UnidicProvider};
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

/// CLI argument for selecting the dictionary format.
///
/// Maps 1-to-1 to `mecrab::DictionaryFormat` but derives `ValueEnum` so that
/// clap can parse it from the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DictFormatArg {
    /// IPADIC format (9 fields) — most common MeCab dictionary
    Ipadic,
    /// UniDic 2.x format (28 fields)
    Unidic,
    /// NEologd extension of IPADIC (9 fields, larger lexicon)
    Neologd,
    /// Automatically detect from the loaded dictionary (default)
    Auto,
}

#[derive(Args, Clone)]
pub struct ParseArgs {
    /// Path to the dictionary directory
    #[arg(short = 'd', long)]
    pub dicdir: Option<PathBuf>,

    /// Dictionary format: ipadic (default), unidic, neologd, auto
    ///
    /// When set to `auto` (the default), the format is detected automatically
    /// by sampling the feature field count from the loaded dictionary.
    /// Explicitly set this to override auto-detection.
    #[arg(long, value_enum, default_value_t = DictFormatArg::Auto)]
    pub dict_format: DictFormatArg,

    /// Path to user dictionary
    #[arg(short = 'u', long)]
    pub userdic: Option<PathBuf>,

    /// Path to semantic pool file (semantic.bin)
    #[arg(short = 's', long)]
    pub semantic_pool: Option<PathBuf>,

    /// Path to vector pool file (vectors.bin)
    #[arg(short = 'v', long)]
    pub vector_pool: Option<PathBuf>,

    /// Include semantic URIs in output (requires semantic pool)
    #[arg(long)]
    pub with_semantic: bool,

    /// Include IPA pronunciation in output
    #[arg(long)]
    pub with_ipa: bool,

    /// Include word embeddings in output (requires vector pool)
    #[arg(long)]
    pub with_vector: bool,

    /// Output format
    #[arg(short = 'O', long, value_enum, default_value_t = Format::Default)]
    pub output_format: Format,

    /// Output wakati (space-separated surface forms only)
    #[arg(short = 'w', long)]
    pub wakati: bool,

    /// Output wakati with word_id instead of surface (for Word2Vec training)
    #[arg(long)]
    pub wakati_word_id: bool,

    /// N-best output (number of alternative analyses to show)
    #[arg(short = 'n', long)]
    pub nbest: Option<usize>,

    /// Enable color output (auto-detected for terminals)
    #[arg(short = 'c', long)]
    pub color: bool,

    /// Disable color output
    #[arg(long)]
    pub no_color: bool,

    /// Input file (reads from stdin if not specified)
    #[arg(short = 'i', long)]
    pub input: Option<PathBuf>,

    /// Output file (writes to stdout if not specified)
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,
}

pub fn run_parse(args: ParseArgs) -> Result<(), Box<dyn std::error::Error>> {
    let format = if args.wakati {
        OutputFormat::Wakati
    } else {
        args.output_format.into()
    };

    let builder = MeCrab::builder()
        .dicdir(args.dicdir)
        .userdic(args.userdic)
        .semantic_pool(args.semantic_pool)
        .vector_pool(args.vector_pool)
        .with_semantic(args.with_semantic)
        .with_ipa(args.with_ipa)
        .with_vector(args.with_vector)
        .output_format(format);

    // Wire the explicit format hint into the provider selection.
    // `Auto` is handled transparently by `MeCrabBuilder::build()` via
    // `Dictionary::sample_feature_count()` — no explicit provider is set so
    // the auto-detection pipeline runs automatically.
    let mecrab = match args.dict_format {
        DictFormatArg::Auto => builder.build()?,
        DictFormatArg::Ipadic => builder.with_provider(IpadicProvider).build()?,
        DictFormatArg::Unidic => builder.with_provider(UnidicProvider).build()?,
        DictFormatArg::Neologd => builder.with_provider(NeologdProvider).build()?,
    };

    // Determine input source
    let input: Box<dyn BufRead> = match &args.input {
        Some(path) => {
            let file = std::fs::File::open(path)?;
            Box::new(io::BufReader::new(file))
        }
        None => Box::new(io::stdin().lock()),
    };

    // Determine output destination
    let mut output: Box<dyn Write> = match &args.output {
        Some(path) => {
            let file = std::fs::File::create(path)?;
            Box::new(io::BufWriter::new(file))
        }
        None => Box::new(io::stdout().lock()),
    };

    // Determine if we should use colors
    let use_color = if args.no_color {
        false
    } else if args.color {
        true
    } else {
        // Auto-detect: only if stdout is a terminal, no output file, and not json formats
        args.output.is_none()
            && io::stdout().is_terminal()
            && matches!(format, OutputFormat::Default | OutputFormat::Dump)
    };

    // Setup progress bar for large files
    let show_progress = args.input.is_some() && io::stderr().is_terminal();
    let progress = if show_progress {
        let file_size = args
            .input
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .unwrap_or(0);

        if file_size > 1024 * 1024 {
            // Only show for files > 1MB
            let pb = indicatif::ProgressBar::new(file_size);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        }
    } else {
        None
    };

    let mut bytes_processed = 0u64;

    for line in input.lines() {
        let line = line?;
        bytes_processed += line.len() as u64 + 1; // +1 for newline

        if let Some(ref pb) = progress {
            pb.set_position(bytes_processed);
        }

        if line.is_empty() {
            continue;
        }

        // Check if N-best mode is requested
        if let Some(n) = args.nbest {
            let results = mecrab.parse_nbest(&line, n)?;
            for (i, (result, cost)) in results.iter().enumerate() {
                if use_color {
                    writeln!(
                        output,
                        "{}# {} (cost={}){}",
                        colors::DIM,
                        i + 1,
                        cost,
                        colors::RESET
                    )?;
                    write_colored_result(&mut output, result)?;
                } else {
                    writeln!(output, "# {} (cost={})", i + 1, cost)?;
                    write!(output, "{result}")?;
                }
                if i < results.len() - 1 {
                    writeln!(output)?;
                }
            }
        } else if matches!(format, OutputFormat::LatticeProb) {
            // LatticeProb requires the forward-backward probability table.
            let (result, probs) = mecrab.parse_with_probs(&line)?;
            writeln!(output, "{}", format_lattice_prob(&result, &probs))?;
        } else {
            let result = mecrab.parse(&line)?;

            // Special handling for wakati-word-id mode (for Word2Vec training)
            if args.wakati_word_id {
                let word_ids: Vec<String> = result
                    .morphemes
                    .iter()
                    .map(|m| m.word_id.to_string())
                    .collect();
                writeln!(output, "{}", word_ids.join(" "))?;
            } else if use_color && matches!(format, OutputFormat::Default | OutputFormat::Dump) {
                write_colored_result(&mut output, &result)?;
            } else {
                writeln!(output, "{result}")?;
            }
        }
    }

    if let Some(pb) = progress {
        pb.finish_with_message("done");
    }

    output.flush()?;
    Ok(())
}

/// Write an analysis result with ANSI colors
pub fn write_colored_result<W: Write>(
    w: &mut W,
    result: &mecrab::AnalysisResult,
) -> io::Result<()> {
    for morpheme in &result.morphemes {
        let features: Vec<&str> = morpheme.feature.split(',').collect();
        let pos = features.first().copied().unwrap_or("*");
        let color = pos_color(pos);

        write!(w, "{}{}{}\t", color, morpheme.surface, colors::RESET)?;
        write!(w, "{}{}{}", colors::DIM, morpheme.feature, colors::RESET)?;
        writeln!(w)?;
    }
    writeln!(w, "{}EOS{}", colors::DIM, colors::RESET)
}
