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

    /// Partial-parsing constraint file (MeCab --partial format): one token per line,
    /// `surface` or `surface<TAB>feature`; surfaces concatenate to the sentence,
    /// blank line or `EOS` ends a sentence.
    #[arg(short = 'p', long = "partial", value_name = "FILE")]
    pub partial: Option<PathBuf>,

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

/// A single forced token span produced from a `--partial` constraint line:
/// `(start_byte, end_byte, forced_feature)`. `forced_feature` is `Some` only
/// when the line carried a `<TAB>feature` suffix.
type PartialSpan = (usize, usize, Option<String>);

/// A parsed `--partial` sentence: the concatenated sentence text plus the list
/// of forced spans that cover it.
type PartialSentence = (String, Vec<PartialSpan>);

/// Parse a MeCab `--partial` constraint file into sentences and forced spans.
///
/// In partial-parsing mode the input file describes one or more sentences as a
/// sequence of constraint lines. Each non-empty, non-`EOS` line is one forced
/// token: either a bare `surface` or `surface<TAB>feature`. The sentence text
/// is the concatenation of all surfaces in order (no separators are inserted),
/// and each surface forces a token boundary covering its byte span within that
/// concatenation. When a line carries a `<TAB>feature` suffix, that feature
/// string is forced onto the synthetic node (returned as `Some(feature)`);
/// otherwise only the boundary is forced and the feature is left free
/// (`None`), so the analyzer fills POS/feature from the dictionary.
///
/// A blank line or a line equal to `EOS` (after stripping a trailing `\r`)
/// terminates the current sentence; a single file may therefore contain
/// multiple sentences. CRLF line endings are handled by `str::lines`, which
/// strips the trailing `\r`. The surface itself is never trimmed.
///
/// Returns a `Vec` of `(sentence_text, spans)` where each span is
/// `(start_byte, end_byte, feature)`. Byte offsets are accumulated from the
/// UTF-8 byte length of each surface, so the concatenated text is always valid
/// UTF-8 and the spans always fall on char boundaries.
fn parse_partial_file(contents: &str) -> Vec<PartialSentence> {
    let mut sentences: Vec<PartialSentence> = Vec::new();
    let mut current_text = String::new();
    let mut current_spans: Vec<PartialSpan> = Vec::new();

    // `str::lines` splits on `\n` and strips a trailing `\r`, so CRLF and LF
    // inputs are handled uniformly without trimming surface content.
    for line in contents.lines() {
        // A blank line or an explicit `EOS` terminates the current sentence.
        if line.is_empty() || line == "EOS" {
            if !current_spans.is_empty() {
                sentences.push((
                    std::mem::take(&mut current_text),
                    std::mem::take(&mut current_spans),
                ));
            } else {
                // No tokens accumulated yet: discard any stray partial text and
                // treat consecutive terminators as a single boundary.
                current_text.clear();
            }
            continue;
        }

        // Split off an optional `<TAB>feature` suffix. The surface is everything
        // up to the first tab; the remainder (which may itself contain commas)
        // is the forced IPADIC feature string.
        let (surface, feature) = match line.split_once('\t') {
            Some((surface, feature)) => (surface, Some(feature.to_owned())),
            None => (line, None),
        };

        let start = current_text.len();
        current_text.push_str(surface);
        let end = current_text.len();

        // A zero-length surface forces no boundary; skip it so we never emit an
        // empty span.
        if end > start {
            current_spans.push((start, end, feature));
        }
    }

    // Flush a trailing sentence that was not terminated by a blank line / `EOS`.
    if !current_spans.is_empty() {
        sentences.push((current_text, current_spans));
    }

    sentences
}

/// Emit a single analysis result honoring the active output flags.
///
/// Shared by the normal line loop and the `--partial` path so both render a
/// result identically: `--bunsetsu` takes priority, then `--wakati-word-id`,
/// then `-F`/`--node-format`, then colored default/dump output, otherwise the
/// plain `Display` form.
///
/// The relevant flags are passed individually (rather than as a `&ParseArgs`)
/// because the builder partially moves `ParseArgs` before parsing begins.
fn emit_result<W: Write>(
    output: &mut W,
    result: &mecrab::AnalysisResult,
    bunsetsu: bool,
    wakati_word_id: bool,
    node_format: Option<&str>,
    use_color: bool,
    format: OutputFormat,
) -> io::Result<()> {
    if bunsetsu {
        // --bunsetsu takes priority over all other format flags.
        write_bunsetsu_result(output, result)
    } else if wakati_word_id {
        // Special handling for wakati-word-id mode (for Word2Vec training).
        let word_ids: Vec<String> = result
            .morphemes
            .iter()
            .map(|m| m.word_id.to_string())
            .collect();
        writeln!(output, "{}", word_ids.join(" "))
    } else if let Some(tmpl) = node_format {
        // `-F` / `--node-format`: MeCab-compatible per-morpheme template.
        let rendered = result.format_with_template(tmpl);
        write!(output, "{}", rendered)
    } else if use_color && matches!(format, OutputFormat::Default | OutputFormat::Dump) {
        write_colored_result(output, result)
    } else {
        writeln!(output, "{result}")
    }
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

    // Partial-parsing mode (MeCab `--partial` / `-p`): the constraint file fully
    // describes the sentence(s), so stdin and `-i` are ignored entirely. This
    // short-circuits before the normal line-reading loop.
    if let Some(partial_path) = &args.partial {
        // N-best is not supported here: `parse_nbest` cannot take constraints,
        // so we honor the boundary constraints and emit the single best path.
        if args.nbest.is_some() {
            eprintln!("warning: n-best (-n) is not supported with --partial; ignoring -n");
        }

        let contents = std::fs::read_to_string(partial_path)?;
        for (text, spans) in parse_partial_file(&contents) {
            let mut constraints = ParseConstraints::new();
            for (start, end, feature) in spans {
                constraints.add_span(start, end, feature);
            }
            let result = mecrab.parse_with_constraints(&text, &constraints)?;
            emit_result(
                &mut output,
                &result,
                args.bunsetsu,
                args.wakati_word_id,
                args.node_format.as_deref(),
                use_color,
                format,
            )?;
        }

        output.flush()?;
        return Ok(());
    }

    // Determine input source
    let input: Box<dyn BufRead> = match &args.input {
        Some(path) => {
            let file = std::fs::File::open(path)?;
            Box::new(io::BufReader::new(file))
        }
        None => Box::new(io::stdin().lock()),
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

            emit_result(
                &mut output,
                &result,
                args.bunsetsu,
                args.wakati_word_id,
                args.node_format.as_deref(),
                use_color,
                format,
            )?;
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

    // ── parse_partial_file ────────────────────────────────────────────────────

    #[test]
    fn test_partial_file_basic_example() {
        // The canonical MeCab `--partial` example:
        //   私<TAB>名詞,代名詞 / は / 本を / 読む<TAB>動詞 / EOS
        // The sentence is the concatenation of the surfaces, with boundaries
        // forced at each surface's byte span and features forced where given.
        let contents = "私\t名詞,代名詞\nは\n本を\n読む\t動詞\nEOS\n";
        let sentences = parse_partial_file(contents);
        assert_eq!(sentences.len(), 1, "expected a single sentence");

        let (text, spans) = &sentences[0];
        assert_eq!(text.as_str(), "私は本を読む");
        assert_eq!(spans.len(), 4, "expected 4 forced spans");

        // 私 → bytes 0..3, feature forced.
        assert_eq!(spans[0], (0usize, 3usize, Some("名詞,代名詞".to_owned())));
        // は → bytes 3..6, feature free.
        assert_eq!(spans[1], (3usize, 6usize, None));
        // 本を → bytes 6..12 (two 3-byte kanji), feature free.
        assert_eq!(spans[2], (6usize, 12usize, None));
        // 読む → bytes 12..18, feature forced.
        assert_eq!(spans[3], (12usize, 18usize, Some("動詞".to_owned())));
    }

    #[test]
    fn test_partial_file_two_sentences_eos() {
        // Two sentences separated by an explicit `EOS` terminator.
        let contents = "猫\nです\nEOS\n犬\nだ\nEOS\n";
        let sentences = parse_partial_file(contents);
        assert_eq!(sentences.len(), 2, "expected two sentences");

        assert_eq!(sentences[0].0.as_str(), "猫です");
        assert_eq!(sentences[0].1.len(), 2);
        assert_eq!(sentences[0].1[0], (0usize, 3usize, None)); // 猫
        assert_eq!(sentences[0].1[1], (3usize, 9usize, None)); // です (で+す = 6 bytes)

        assert_eq!(sentences[1].0.as_str(), "犬だ");
        assert_eq!(sentences[1].1.len(), 2);
        assert_eq!(sentences[1].1[0], (0usize, 3usize, None)); // 犬
        assert_eq!(sentences[1].1[1], (3usize, 6usize, None)); // だ
    }

    #[test]
    fn test_partial_file_blank_line_terminator() {
        // A blank line terminates a sentence just like `EOS`. The trailing
        // sentence here has no terminator and must still be flushed.
        let contents = "猫\nです\n\n犬\nだ\n";
        let sentences = parse_partial_file(contents);
        assert_eq!(sentences.len(), 2, "blank line should split sentences");
        assert_eq!(sentences[0].0.as_str(), "猫です");
        assert_eq!(sentences[1].0.as_str(), "犬だ");
    }

    #[test]
    fn test_partial_file_tab_feature() {
        // `本を<TAB>名詞` forces the feature `名詞` onto the span.
        let contents = "本を\t名詞\n";
        let sentences = parse_partial_file(contents);
        assert_eq!(sentences.len(), 1);
        let (text, spans) = &sentences[0];
        assert_eq!(text.as_str(), "本を");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0], (0usize, 6usize, Some("名詞".to_owned())));
    }

    #[test]
    fn test_partial_file_multibyte_byte_offsets() {
        // Each kanji is 3 UTF-8 bytes; offsets must accumulate by byte length
        // and every span must land on a char boundary.
        let contents = "東京\n都\n";
        let sentences = parse_partial_file(contents);
        assert_eq!(sentences.len(), 1);
        let (text, spans) = &sentences[0];
        assert_eq!(text.as_str(), "東京都");
        assert_eq!(text.len(), 9, "3 kanji × 3 bytes = 9 bytes");
        assert_eq!(spans[0], (0usize, 6usize, None)); // 東京
        assert_eq!(spans[1], (6usize, 9usize, None)); // 都
        for &(start, end, _) in spans {
            assert!(
                text.is_char_boundary(start),
                "start must be a char boundary"
            );
            assert!(text.is_char_boundary(end), "end must be a char boundary");
        }
    }

    #[test]
    fn test_partial_file_crlf() {
        // CRLF endings: the trailing `\r` is stripped, surface content is intact.
        let contents = "私\t名詞\r\nは\r\nEOS\r\n";
        let sentences = parse_partial_file(contents);
        assert_eq!(sentences.len(), 1);
        let (text, spans) = &sentences[0];
        assert_eq!(text.as_str(), "私は");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], (0usize, 3usize, Some("名詞".to_owned())));
        assert_eq!(spans[1], (3usize, 6usize, None));
    }

    #[test]
    fn test_partial_file_empty_and_terminator_only() {
        // Empty input and terminator-only input yield no sentences.
        assert!(parse_partial_file("").is_empty());
        assert!(parse_partial_file("EOS\n").is_empty());
        assert!(parse_partial_file("\n\n").is_empty());
        assert!(parse_partial_file("EOS\nEOS\n").is_empty());
    }
}
