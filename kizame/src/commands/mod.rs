//! Shared types and utilities for all command handlers.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

pub mod dict;
#[cfg(feature = "full")]
pub mod dict_compile;
pub mod explore;
#[cfg(feature = "lsp")]
pub mod lsp;
pub mod parse;
pub mod score;
#[cfg(feature = "server")]
pub mod serve;
pub mod train;
pub mod vectors;

use clap::ValueEnum;
use mecrab::OutputFormat;

/// ANSI color codes for terminal output
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    #[allow(dead_code)]
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";

    // POS colors
    pub const NOUN: &str = "\x1b[38;5;39m"; // Blue
    pub const VERB: &str = "\x1b[38;5;208m"; // Orange
    pub const ADJ: &str = "\x1b[38;5;118m"; // Green
    pub const PARTICLE: &str = "\x1b[38;5;243m"; // Gray
    pub const AUX: &str = "\x1b[38;5;141m"; // Purple
    pub const SYMBOL: &str = "\x1b[38;5;245m"; // Light gray
    pub const OTHER: &str = "\x1b[38;5;250m"; // White
}

/// Get color for a POS category
pub fn pos_color(pos: &str) -> &'static str {
    if pos.starts_with("名詞") {
        colors::NOUN
    } else if pos.starts_with("動詞") {
        colors::VERB
    } else if pos.starts_with("形容詞") || pos.starts_with("形状詞") {
        colors::ADJ
    } else if pos.starts_with("助詞") {
        colors::PARTICLE
    } else if pos.starts_with("助動詞") {
        colors::AUX
    } else if pos.starts_with("記号") || pos.starts_with("補助記号") {
        colors::SYMBOL
    } else {
        colors::OTHER
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// Default MeCab format
    Default,
    /// Wakati (space-separated)
    Wakati,
    /// Dump all lattice information
    Dump,
    /// JSON output
    Json,
    /// JSON-LD output with semantic URIs
    Jsonld,
    /// Turtle (TTL) RDF format
    Turtle,
    /// N-Triples RDF format
    Ntriples,
    /// N-Quads RDF format
    Nquads,
    /// JSON with lattice marginal probabilities (forward-backward algorithm)
    LatticeProb,
    /// SentencePiece-compatible format with ▁ word-initial markers
    BpeCompatible,
    /// CoNLL-U Universal Dependencies format
    #[value(name = "conllu")]
    ConllU,
}

impl From<Format> for OutputFormat {
    fn from(f: Format) -> Self {
        match f {
            Format::Default => OutputFormat::Default,
            Format::Wakati => OutputFormat::Wakati,
            Format::Dump => OutputFormat::Dump,
            Format::Json => OutputFormat::Json,
            Format::Jsonld => OutputFormat::Jsonld,
            Format::Turtle => OutputFormat::Turtle,
            Format::Ntriples => OutputFormat::Ntriples,
            Format::Nquads => OutputFormat::Nquads,
            Format::LatticeProb => OutputFormat::LatticeProb,
            Format::BpeCompatible => OutputFormat::BpeCompatible,
            Format::ConllU => OutputFormat::ConllU,
        }
    }
}
