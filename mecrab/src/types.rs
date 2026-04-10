//! Core output types for MeCrab morphological analysis.
//!
//! This module contains [`OutputFormat`], [`Morpheme`], and [`AnalysisResult`]
//! together with their `impl` blocks, including linguistic helper methods such
//! as [`AnalysisResult::spans`], [`AnalysisResult::noun_phrases`],
//! [`AnalysisResult::named_entities`], and [`AnalysisResult::verb_chunks`].

use std::fmt;

use crate::semantic;

/// Output format for morphological analysis results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Default MeCab format: surface\tfeatures
    #[default]
    Default,
    /// Wakati format: space-separated surface forms
    Wakati,
    /// Dump all lattice information for debugging
    Dump,
    /// JSON output format
    Json,
    /// JSON-LD output format with semantic URIs
    Jsonld,
    /// Turtle (TTL) RDF format
    Turtle,
    /// N-Triples RDF format
    Ntriples,
    /// N-Quads RDF format
    Nquads,
    /// JSON output with marginal probabilities from the forward-backward algorithm.
    ///
    /// Suitable for subword regularization and LLM pre-training data pipelines.
    /// Use `MeCrab::parse_with_probs` to obtain both the result and the probability
    /// table, then call `api::format::format_lattice_prob` to render this format.
    LatticeProb,
    /// SentencePiece-compatible format with ▁ (U+2581) word-initial markers.
    ///
    /// Matches the mT5/SentencePiece tokenization convention where each
    /// word-initial morpheme is prefixed with ▁.  Suitable as input for BPE
    /// or SentencePiece tokenizer training.
    BpeCompatible,
}

/// A single morpheme (token) in the analysis result
#[derive(Debug, Clone)]
pub struct Morpheme {
    /// Surface form (the actual text)
    pub surface: String,
    /// Word ID (token index in dictionary, used for embeddings and training)
    pub word_id: u32,
    /// Part-of-speech ID
    pub pos_id: u16,
    /// Word cost
    pub wcost: i16,
    /// Feature string (comma-separated POS info, reading, etc.)
    pub feature: String,
    /// Semantic entity references (optional)
    pub entities: Vec<semantic::extension::EntityReference>,
    /// IPA pronunciation (optional, populated when ipa_enabled=true)
    pub pronunciation: Option<String>,
    /// Word embedding vector (optional, populated when vector_enabled=true)
    pub embedding: Option<Vec<f32>>,
    /// Start byte offset of this morpheme in the normalized input text
    pub start_byte: usize,
    /// End byte offset (exclusive) of this morpheme in the normalized input text
    pub end_byte: usize,
}

impl Morpheme {
    /// Return the char-indexed (not byte-indexed) start position.
    ///
    /// Requires the original text to convert from byte offset to char offset.
    #[must_use]
    pub fn start_char(&self, text: &str) -> usize {
        text[..self.start_byte].chars().count()
    }

    /// Return the char-indexed (not byte-indexed) end position.
    ///
    /// Requires the original text to convert from byte offset to char offset.
    #[must_use]
    pub fn end_char(&self, text: &str) -> usize {
        text[..self.end_byte].chars().count()
    }
}

impl fmt::Display for Morpheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Main line: MeCab-compatible format
        write!(f, "{}\t{}", self.surface, self.feature)?;

        // Optional IPA pronunciation line
        if let Some(ref ipa) = self.pronunciation {
            write!(f, "\n  IPA: /{}/", ipa)?;
        }

        // Optional embedding vector (show first 8 dimensions for readability)
        if let Some(ref emb) = self.embedding {
            write!(f, "\n  Vector: [")?;
            let show_dims = emb.len().min(8);
            for (i, val) in emb.iter().take(show_dims).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.3}", val)?;
            }
            if emb.len() > show_dims {
                write!(f, ", ...")?;
            }
            write!(f, "] (dim={})", emb.len())?;
        }

        Ok(())
    }
}

/// Analysis result containing a sequence of morphemes
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// The morphemes in the analysis result
    pub morphemes: Vec<Morpheme>,
    /// Output format
    pub(crate) format: OutputFormat,
}

impl fmt::Display for AnalysisResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.format {
            OutputFormat::Default => {
                for morpheme in &self.morphemes {
                    writeln!(f, "{morpheme}")?;
                }
                writeln!(f, "EOS")
            }
            OutputFormat::Wakati => {
                let surfaces: Vec<&str> =
                    self.morphemes.iter().map(|m| m.surface.as_str()).collect();
                writeln!(f, "{}", surfaces.join(" "))
            }
            OutputFormat::Dump => {
                for (i, morpheme) in self.morphemes.iter().enumerate() {
                    writeln!(
                        f,
                        "[{}] {} (pos_id={}, wcost={})\t{}",
                        i, morpheme.surface, morpheme.pos_id, morpheme.wcost, morpheme.feature
                    )?;
                }
                writeln!(f, "EOS")
            }
            OutputFormat::Json => self.format_json(f),
            OutputFormat::Jsonld => self.format_jsonld(f),
            OutputFormat::Turtle => self.format_turtle(f),
            OutputFormat::Ntriples => self.format_ntriples(f),
            OutputFormat::Nquads => self.format_nquads(f),
            OutputFormat::BpeCompatible => self.format_bpe_compatible(f),
            // LatticeProb requires probability data — display falls back to JSON.
            // Use `MeCrab::parse_with_probs` + `api::format::format_lattice_prob` for
            // the full probabilistic output.
            OutputFormat::LatticeProb => self.format_json(f),
        }
    }
}

// format_json / format_jsonld / format_turtle / format_ntriples / format_nquads
// are implemented in api/format.rs as `impl AnalysisResult { ... }` so that the
// formatting logic is kept out of this file.  The methods are pub(crate) there
// and called from the Display impl above.

impl AnalysisResult {
    /// Return byte-offset spans for all non-EOS morphemes.
    ///
    /// Returns `(start_byte, end_byte)` pairs in order.
    pub fn spans(&self) -> Vec<(usize, usize)> {
        self.morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .map(|m| (m.start_byte, m.end_byte))
            .collect()
    }

    /// Extract compound noun phrases.
    ///
    /// Consecutive morphemes whose POS (part-of-speech) starts with `"名詞"` are
    /// joined into a single surface string.  Single-morpheme sequences whose
    /// surface is a punctuation mark or symbol are excluded.
    ///
    /// Returns each compound noun as a `(surface, start_byte, end_byte)` tuple.
    pub fn noun_phrases(&self) -> Vec<(String, usize, usize)> {
        let mut result = Vec::new();
        let mut i = 0;
        let non_eos: Vec<&Morpheme> = self
            .morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .collect();

        while i < non_eos.len() {
            let m = non_eos[i];
            let feature_pos = m.feature.split(',').next().unwrap_or("");
            if feature_pos.starts_with("名詞") {
                let start_byte = m.start_byte;
                let mut end_byte = m.end_byte;
                let mut compound = m.surface.clone();
                let mut j = i + 1;
                while j < non_eos.len() {
                    let next = non_eos[j];
                    let next_pos = next.feature.split(',').next().unwrap_or("");
                    if next_pos.starts_with("名詞") {
                        compound.push_str(&next.surface);
                        end_byte = next.end_byte;
                        j += 1;
                    } else {
                        break;
                    }
                }
                // Only include phrases that are not purely punctuation/symbols
                if !compound.chars().all(|c| {
                    !c.is_alphanumeric()
                        && !matches!(c, 'ぁ'..='ん' | 'ァ'..='ン' | '\u{4E00}'..='\u{9FFF}' | 'A'..='z')
                }) {
                    result.push((compound, start_byte, end_byte));
                }
                i = j;
            } else {
                i += 1;
            }
        }
        result
    }

    /// Extract named entities (proper nouns with 固有名詞 in feature).
    ///
    /// Returns `(surface, entity_type, start_byte, end_byte)` tuples where
    /// `entity_type` is the IPADIC sub-category (地域, 人名, 組織, 一般, etc.)
    pub fn named_entities(&self) -> Vec<(String, String, usize, usize)> {
        let mut result = Vec::new();
        let mut i = 0;
        let non_eos: Vec<&Morpheme> = self
            .morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .collect();

        while i < non_eos.len() {
            let m = non_eos[i];
            let fields: Vec<&str> = m.feature.splitn(4, ',').collect();
            if fields.first().copied() == Some("名詞") && fields.get(1).copied() == Some("固有名詞")
            {
                let entity_type = fields.get(2).unwrap_or(&"一般").to_string();
                let start_byte = m.start_byte;
                let mut end_byte = m.end_byte;
                let mut compound = m.surface.clone();
                let mut j = i + 1;
                while j < non_eos.len() {
                    let next = non_eos[j];
                    let next_fields: Vec<&str> = next.feature.splitn(4, ',').collect();
                    if next_fields.first().copied() == Some("名詞")
                        && next_fields.get(1).copied() == Some("固有名詞")
                    {
                        compound.push_str(&next.surface);
                        end_byte = next.end_byte;
                        j += 1;
                    } else {
                        break;
                    }
                }
                result.push((compound, entity_type, start_byte, end_byte));
                i = j;
            } else {
                i += 1;
            }
        }
        result
    }

    /// Extract verb chunks (verb + following auxiliaries / verb endings).
    ///
    /// Collects a base verb (`動詞`) followed by any adjacent auxiliaries
    /// (`助動詞`), verb suffixes, or adjectival forms that modify it.
    ///
    /// Returns `(surface, start_byte, end_byte)` tuples.
    pub fn verb_chunks(&self) -> Vec<(String, usize, usize)> {
        let mut result = Vec::new();
        let mut i = 0;
        let non_eos: Vec<&Morpheme> = self
            .morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .collect();

        while i < non_eos.len() {
            let m = non_eos[i];
            let pos = m.feature.split(',').next().unwrap_or("");
            if pos == "動詞" {
                let start_byte = m.start_byte;
                let mut end_byte = m.end_byte;
                let mut chunk = m.surface.clone();
                let mut j = i + 1;
                while j < non_eos.len() {
                    let next = non_eos[j];
                    let next_pos = next.feature.split(',').next().unwrap_or("");
                    match next_pos {
                        "動詞" => break, // new independent verb — stop here
                        "助動詞" | "接尾" | "形容詞" => {
                            chunk.push_str(&next.surface);
                            end_byte = next.end_byte;
                            j += 1;
                        }
                        _ => break,
                    }
                }
                result.push((chunk, start_byte, end_byte));
                i = j;
            } else {
                i += 1;
            }
        }
        result
    }
}
