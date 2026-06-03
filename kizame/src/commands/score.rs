//! Score command: compute Viterbi cost, perplexity, and entropy for input text.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use crate::commands::parse::DictFormatArg;
use clap::Args;
use mecrab::viterbi::analysis::LatticeProbTable;
use mecrab::{IpadicProvider, MeCrab, NeologdProvider, UnidicProvider};
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

/// Output format for the score command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoreOutput {
    /// Tab-separated values (default)
    Tsv,
    /// One JSON object per line (NDJSON)
    Json,
    /// Human-readable summary
    Summary,
}

/// Arguments for the `kizame score` subcommand.
#[derive(Args, Clone)]
pub struct ScoreArgs {
    /// Path to the dictionary directory
    #[arg(short = 'd', long)]
    pub dicdir: Option<PathBuf>,

    /// Dictionary format: ipadic (default), unidic, neologd, auto
    #[arg(long, value_enum, default_value_t = DictFormatArg::Auto)]
    pub dict_format: DictFormatArg,

    /// Path to user dictionary
    #[arg(short = 'u', long)]
    pub userdic: Option<PathBuf>,

    /// Score a single text string instead of reading from stdin
    #[arg(short = 't', long, value_name = "TEXT")]
    pub text: Option<String>,

    /// Input file (reads from stdin if not specified and --text is not given)
    #[arg(short = 'i', long)]
    pub input: Option<PathBuf>,

    /// Emit one JSON object per line instead of TSV
    #[arg(long)]
    pub json: bool,

    /// Emit a human-readable summary per line instead of TSV
    #[arg(long)]
    pub summary: bool,
}

/// Raw row of computed scores for one input line.
struct ScoreRow<'a> {
    text: &'a str,
    viterbi_cost: i64,
    perplexity: f64,
    entropy: f64,
    morphemes: usize,
    oov: usize,
}

impl<'a> ScoreRow<'a> {
    /// Render as TSV line.
    fn as_tsv(&self) -> String {
        format!(
            "{}\t{}\t{:.6}\t{:.6}\t{}\t{}",
            self.text, self.viterbi_cost, self.perplexity, self.entropy, self.morphemes, self.oov,
        )
    }

    /// Render as a single JSON object (no trailing newline).
    fn as_json(&self) -> String {
        // Escape the text field so embedded quotes / backslashes are safe.
        let escaped = escape_json_str(self.text);
        format!(
            "{{\"text\":{escaped},\"viterbi_cost\":{},\"perplexity\":{:.6},\"entropy\":{:.6},\"morphemes\":{},\"oov\":{}}}",
            self.viterbi_cost, self.perplexity, self.entropy, self.morphemes, self.oov,
        )
    }

    /// Render as a human-readable multi-line summary block.
    fn as_summary(&self) -> String {
        format!(
            "Text       : {}\n\
             Viterbi    : {}\n\
             Perplexity : {:.4}\n\
             Entropy    : {:.4} nats\n\
             Morphemes  : {}\n\
             OOV tokens : {}\n\
             ---",
            self.text, self.viterbi_cost, self.perplexity, self.entropy, self.morphemes, self.oov,
        )
    }
}

/// Minimal JSON string escaper — handles the characters that break JSON strings.
fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Count OOV tokens (word_id == u32::MAX) in an analysis result.
fn count_oov(result: &mecrab::AnalysisResult) -> usize {
    result
        .morphemes
        .iter()
        .filter(|m| m.word_id == u32::MAX)
        .count()
}

/// Compute scores for a single text line and return a [`ScoreRow`].
///
/// This calls both `parse` (for Viterbi cost) and `parse_with_probs`
/// (for forward-backward marginals) on the same input.
fn score_line<'a>(
    mecrab: &MeCrab,
    text: &'a str,
) -> Result<ScoreRow<'a>, Box<dyn std::error::Error>> {
    // Viterbi parse for cost + OOV count
    let result = mecrab.parse(text)?;
    let viterbi_cost: i64 = result.morphemes.iter().map(|m| m.wcost as i64).sum();
    let oov = count_oov(&result);
    let morpheme_count_viterbi = result.morphemes.len();

    // Forward-backward for perplexity + entropy
    let (_, probs): (_, LatticeProbTable) = mecrab.parse_with_probs(text)?;
    let perplexity = probs.perplexity();
    let entropy = probs.segmentation_entropy();

    // Use the forward-backward estimate if it's non-zero, else fall back to Viterbi count
    let fb_morphemes = probs.morpheme_count_estimate();
    let morphemes = if fb_morphemes > 0 {
        fb_morphemes
    } else {
        morpheme_count_viterbi
    };

    Ok(ScoreRow {
        text,
        viterbi_cost,
        perplexity,
        entropy,
        morphemes,
        oov,
    })
}

/// Entry point for `kizame score`.
pub fn run_score(args: ScoreArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve output mode.  `--json` takes precedence over `--summary`.
    let mode = if args.json {
        ScoreOutput::Json
    } else if args.summary {
        ScoreOutput::Summary
    } else {
        ScoreOutput::Tsv
    };

    let builder = MeCrab::builder().dicdir(args.dicdir).userdic(args.userdic);

    let mecrab = match args.dict_format {
        DictFormatArg::Auto => builder.build()?,
        DictFormatArg::Ipadic => builder.with_provider(IpadicProvider).build()?,
        DictFormatArg::Unidic => builder.with_provider(UnidicProvider).build()?,
        DictFormatArg::Neologd => builder.with_provider(NeologdProvider).build()?,
    };

    let mut stdout = io::stdout().lock();

    // Print TSV header when writing to a terminal in TSV mode.
    if mode == ScoreOutput::Tsv && io::stdout().is_terminal() {
        writeln!(
            stdout,
            "text\tviterbi_cost\tperplexity\tentropy\tmorphemes\toov"
        )?;
    }

    // Branch on input source.
    if let Some(ref text) = args.text {
        // Single --text argument
        if !text.is_empty() {
            emit_line(&mecrab, text, mode, &mut stdout)?;
        }
    } else {
        // Read from file or stdin, line by line.
        let input: Box<dyn BufRead> = match &args.input {
            Some(path) => {
                let file = std::fs::File::open(path)?;
                Box::new(io::BufReader::new(file))
            }
            None => Box::new(io::stdin().lock()),
        };

        for line_result in input.lines() {
            let line = line_result?;
            if line.is_empty() {
                continue;
            }
            emit_line(&mecrab, &line, mode, &mut stdout)?;
        }
    }

    stdout.flush()?;
    Ok(())
}

/// Score a single line and write output in the chosen format.
fn emit_line<W: Write>(
    mecrab: &MeCrab,
    text: &str,
    mode: ScoreOutput,
    out: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    let row = score_line(mecrab, text)?;
    let line = match mode {
        ScoreOutput::Tsv => row.as_tsv(),
        ScoreOutput::Json => row.as_json(),
        ScoreOutput::Summary => row.as_summary(),
    };
    writeln!(out, "{line}")?;
    Ok(())
}
