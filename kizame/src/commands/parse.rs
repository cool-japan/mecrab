//! Parse command - morphological analysis
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use crate::commands::{Format, colors, pos_color};
use clap::{Args, ValueEnum};
use mecrab::api::format::format_lattice_prob;
use mecrab::{
    IpadicProvider, MeCrab, NeologdProvider, OutputFormat, ParseConstraints, UnidicProvider,
};
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

    /// MeCab-compatible node-format template (e.g. "%m,%f[0]\n").
    ///
    /// When set, this template is expanded for every content morpheme and the
    /// result is written to the output, overriding the `-O` format flag.
    /// Supports the same placeholders as MeCab's `--node-format` option.
    #[arg(short = 'F', long = "node-format", value_name = "TEMPLATE")]
    pub node_format: Option<String>,

    /// N-best output (number of alternative analyses to show)
    #[arg(short = 'n', long)]
    pub nbest: Option<usize>,

    /// Force specific byte spans to be single tokens. Format: "start:end" (byte offsets).
    /// May be repeated: --force-span 0:3 --force-span 6:9
    #[arg(long = "force-span", value_name = "START:END", action = clap::ArgAction::Append)]
    pub force_spans: Vec<String>,

    /// Output bunsetsu (文節) phrase chunks instead of morphemes.
    #[arg(long = "bunsetsu")]
    pub bunsetsu: bool,

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

/// Parse a `"start:end"` span string into `(usize, usize)`.
///
/// Returns an error message if the string is malformed or the offsets are
/// out of order.
fn parse_span_str(s: &str) -> Result<(usize, usize), String> {
    let (start_s, end_s) = s
        .split_once(':')
        .ok_or_else(|| format!("invalid span '{}': expected format START:END", s))?;
    let start = start_s.parse::<usize>().map_err(|_| {
        format!(
            "invalid span '{}': '{}' is not a valid byte offset",
            s, start_s
        )
    })?;
    let end = end_s.parse::<usize>().map_err(|_| {
        format!(
            "invalid span '{}': '{}' is not a valid byte offset",
            s, end_s
        )
    })?;
    if start >= end {
        return Err(format!(
            "invalid span '{}': start ({}) must be less than end ({})",
            s, start, end
        ));
    }
    Ok((start, end))
}

/// Write bunsetsu (文節) chunks for a parse result, one per line.
///
/// Format: `SURFACE\t[morph1 morph2 ...]`
/// Followed by `EOS`.
fn write_bunsetsu_result<W: Write>(w: &mut W, result: &mecrab::AnalysisResult) -> io::Result<()> {
    let chunks = result.bunsetsu();
    for chunk in &chunks {
        let morph_surfaces: Vec<&str> = result.morphemes[chunk.morpheme_range.clone()]
            .iter()
            .map(|m| m.surface.as_str())
            .collect();
        writeln!(w, "{}\t[{}]", chunk.surface, morph_surfaces.join(" "))?;
    }
    writeln!(w, "EOS")
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

    // Parse forced-span strings once (not inside the hot loop).
    let forced_spans: Vec<(usize, usize)> = args
        .force_spans
        .iter()
        .map(|s| parse_span_str(s).map_err(|e| -> Box<dyn std::error::Error> { e.into() }))
        .collect::<Result<Vec<_>, _>>()?;

    // Warn if --bunsetsu is combined with -O or -F (bunsetsu takes priority).
    if args.bunsetsu
        && (args.node_format.is_some() || args.output_format != Format::Default || args.wakati)
    {
        eprintln!("warning: --bunsetsu takes priority over -O/-F/--wakati output flags");
    }

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
            // Choose parse path: constrained (forced spans) or unconstrained.
            let result = if forced_spans.is_empty() {
                mecrab.parse(&line)?
            } else {
                let mut constraints = ParseConstraints::new();
                for &(start, end) in &forced_spans {
                    constraints.add_span(start, end, None);
                }
                mecrab.parse_with_constraints(&line, &constraints)?
            };

            // --bunsetsu takes priority over all other format flags.
            if args.bunsetsu {
                write_bunsetsu_result(&mut output, &result)?;
            } else if args.wakati_word_id {
                // Special handling for wakati-word-id mode (for Word2Vec training)
                let word_ids: Vec<String> = result
                    .morphemes
                    .iter()
                    .map(|m| m.word_id.to_string())
                    .collect();
                writeln!(output, "{}", word_ids.join(" "))?;
            } else if let Some(ref tmpl) = args.node_format {
                // `-F` / `--node-format`: MeCab-compatible per-morpheme template.
                // The template is applied to all content morphemes; the result is
                // emitted directly without a trailing EOS line.
                let rendered = result.format_with_template(tmpl);
                write!(output, "{}", rendered)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_span_str ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_force_span_string_parsing_valid() {
        let result = parse_span_str("3:6");
        assert_eq!(result, Ok((3usize, 6usize)));
    }

    #[test]
    fn test_parse_force_span_string_parsing_zero_start() {
        let result = parse_span_str("0:9");
        assert_eq!(result, Ok((0usize, 9usize)));
    }

    #[test]
    fn test_parse_force_span_string_parsing_invalid_alpha() {
        let result = parse_span_str("abc");
        assert!(result.is_err(), "non-numeric span string must fail");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("expected format START:END"),
            "error message should mention expected format, got: {msg}"
        );
    }

    #[test]
    fn test_parse_force_span_string_parsing_invalid_start() {
        let result = parse_span_str("abc:6");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("abc"),
            "error message should name the bad token, got: {msg}"
        );
    }

    #[test]
    fn test_parse_force_span_string_parsing_invalid_end() {
        let result = parse_span_str("3:xyz");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("xyz"),
            "error message should name the bad token, got: {msg}"
        );
    }

    #[test]
    fn test_parse_force_span_string_parsing_equal_offsets() {
        // start must be strictly less than end
        let result = parse_span_str("5:5");
        assert!(result.is_err(), "start == end must be rejected");
    }

    #[test]
    fn test_parse_force_span_string_parsing_inverted_offsets() {
        // start > end is invalid
        let result = parse_span_str("9:3");
        assert!(result.is_err(), "start > end must be rejected");
    }

    // ── write_bunsetsu_result ─────────────────────────────────────────────────

    /// Build a minimal `Morpheme` without a real dictionary.
    fn make_morpheme(
        surface: &str,
        feature: &str,
        start_byte: usize,
        end_byte: usize,
    ) -> mecrab::Morpheme {
        mecrab::Morpheme {
            surface: surface.to_owned(),
            word_id: 0,
            pos_id: 0,
            wcost: 0,
            feature: feature.to_owned(),
            entities: vec![],
            pronunciation: None,
            embedding: None,
            start_byte,
            end_byte,
        }
    }

    #[test]
    fn test_bunsetsu_format_nonempty() {
        // 私は本を読む — three bunsetsu expected:
        //   "私は"  [私 は]
        //   "本を"  [本 を]
        //   "読む"  [読む]
        let morphemes = vec![
            make_morpheme("私", "名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ", 0, 3),
            make_morpheme("は", "助詞,係助詞,*,*,*,*,は,ハ,ワ", 3, 6),
            make_morpheme("本", "名詞,一般,*,*,*,*,本,ホン,ホン", 6, 9),
            make_morpheme("を", "助詞,格助詞,一般,*,*,*,を,ヲ,ヲ", 9, 12),
            make_morpheme(
                "読む",
                "動詞,自立,*,*,五段・マ行,基本形,読む,ヨム,ヨム",
                12,
                24,
            ),
        ];
        let result = mecrab::AnalysisResult::new(morphemes, mecrab::OutputFormat::Default);

        let mut buf = Vec::new();
        write_bunsetsu_result(&mut buf, &result).expect("write should succeed");
        let output = String::from_utf8(buf).expect("valid UTF-8");

        // Three bunsetsu lines + EOS
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 4, "expected 3 bunsetsu + EOS, got: {output}");
        assert_eq!(lines[0], "私は\t[私 は]");
        assert_eq!(lines[1], "本を\t[本 を]");
        assert_eq!(lines[2], "読む\t[読む]");
        assert_eq!(lines[3], "EOS");
    }

    #[test]
    fn test_bunsetsu_format_empty_morphemes() {
        let result = mecrab::AnalysisResult::new(vec![], mecrab::OutputFormat::Default);
        let mut buf = Vec::new();
        write_bunsetsu_result(&mut buf, &result).expect("write should succeed");
        let output = String::from_utf8(buf).expect("valid UTF-8");
        // No bunsetsu chunks; just the trailing EOS line.
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "EOS");
    }
}
