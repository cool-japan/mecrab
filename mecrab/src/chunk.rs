//! Bunsetsu (文節) phrase chunker for Japanese morphological analysis.
//!
//! A *bunsetsu* is the minimal grammatical unit in Japanese surface syntax:
//! one **content word** (自立語) followed by zero or more **functional words**
//! (付属語). This module provides a pure post-processing pass over an already
//! analysed `&[Morpheme]` — it does not access the dictionary or the lattice.
//!
//! # Japanese chunking rules implemented
//!
//! * **Content words** (自立語, starts a new bunsetsu): 名詞, 動詞, 形容詞,
//!   形容動詞, 副詞, 連体詞, 接続詞, 感動詞, 記号 (as its own singleton chunk).
//! * **Functional words** (付属語, attaches to preceding bunsetsu): 助詞, 助動詞,
//!   接尾, 接頭.
//! * Leading functional words before any content word form their own bunsetsu
//!   (graceful fallback).
//!
//! # Example
//!
//! ```rust
//! use mecrab::chunk::{BunsetsuChunker, BunsetsuType};
//! use mecrab::Morpheme;
//!
//! // Hand-built morphemes representing "私は" "本を" "読む"
//! let morphemes = vec![
//!     make_morpheme("私", "名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ", 0, 3),
//!     make_morpheme("は", "助詞,係助詞,*,*,*,*,は,ハ,ワ", 3, 6),
//!     make_morpheme("本", "名詞,一般,*,*,*,*,本,ホン,ホン", 6, 9),
//!     make_morpheme("を", "助詞,格助詞,一般,*,*,*,を,ヲ,ヲ", 9, 12),
//!     make_morpheme("読む", "動詞,自立,*,*,五段・マ行,基本形,読む,ヨム,ヨム", 12, 24),
//! ];
//!
//! let chunks = BunsetsuChunker::chunk(&morphemes);
//! assert_eq!(chunks.len(), 3);
//! # fn make_morpheme(s: &str, f: &str, sb: usize, eb: usize) -> mecrab::Morpheme {
//! #     mecrab::Morpheme { surface: s.to_string(), word_id: 0, pos_id: 0, wcost: 0,
//! #         feature: f.to_string(), entities: vec![], pronunciation: None, embedding: None,
//! #         start_byte: sb, end_byte: eb }
//! # }
//! ```

use crate::Morpheme;

// ── Types ────────────────────────────────────────────────────────────────────

/// The grammatical role of a bunsetsu phrase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BunsetsuType {
    /// Bunsetsu headed by a noun (名詞).
    NounPhrase,
    /// Bunsetsu headed by a verb (動詞).
    VerbPhrase,
    /// Bunsetsu headed by an adjective (形容詞) or adjectival noun (形容動詞).
    AdjectivalPhrase,
    /// Bunsetsu headed by an adverb (副詞).
    AdverbialPhrase,
    /// Bunsetsu headed by a conjunction (接続詞) or interjection (感動詞).
    ConjunctionalPhrase,
    /// Bunsetsu headed by a symbol (記号) or anything unclassified.
    Other,
}

/// A single bunsetsu (文節) phrase: a contiguous span of morphemes beginning
/// with one content word followed by zero or more functional words.
#[derive(Debug, Clone)]
pub struct Bunsetsu {
    /// The surface text of the entire bunsetsu.
    pub surface: String,
    /// Byte offset of the first character (inclusive).
    pub start_byte: usize,
    /// Byte offset one past the last character (exclusive).
    pub end_byte: usize,
    /// The half-open range `[head, tail)` into the source morpheme slice.
    pub morpheme_range: std::ops::Range<usize>,
    /// Index within `morpheme_range` of the head (content word).
    ///
    /// Always 0 when the bunsetsu starts with a content word.
    pub head_idx: usize,
    /// Grammatical type derived from the head morpheme's POS.
    pub chunk_type: BunsetsuType,
}

// ── Classifier helpers ───────────────────────────────────────────────────────

/// Returns `true` when the IPADIC top-level POS tag is a content word (自立語).
fn is_content_word(pos: &str) -> bool {
    matches!(
        pos,
        "名詞" | "動詞" | "形容詞" | "形容動詞" | "副詞" | "連体詞" | "接続詞" | "感動詞"
    )
}

/// Returns `true` for POS tags that unconditionally start their own single-morpheme bunsetsu
/// (punctuation symbols and fillers).
fn is_singleton_symbol(pos: &str) -> bool {
    matches!(pos, "記号" | "フィラー" | "感動詞")
}

/// Returns `true` when the POS is a functional word (付属語) that attaches to
/// the preceding content word.
fn is_functional_word(pos: &str) -> bool {
    matches!(pos, "助詞" | "助動詞" | "接尾" | "接頭" | "特殊")
}

/// Determine the `BunsetsuType` from the head morpheme's top-level POS.
fn bunsetsu_type_from_pos(pos: &str) -> BunsetsuType {
    match pos {
        "名詞" => BunsetsuType::NounPhrase,
        "動詞" => BunsetsuType::VerbPhrase,
        "形容詞" | "形容動詞" => BunsetsuType::AdjectivalPhrase,
        "副詞" | "連体詞" => BunsetsuType::AdverbialPhrase,
        "接続詞" | "感動詞" => BunsetsuType::ConjunctionalPhrase,
        _ => BunsetsuType::Other,
    }
}

/// Extract the top-level POS tag (field 0 of the comma-separated feature string).
fn top_pos(morpheme: &Morpheme) -> &str {
    morpheme.feature.split(',').next().unwrap_or("*")
}

// ── BunsetsuChunker ──────────────────────────────────────────────────────────

/// Stateless bunsetsu chunker.
///
/// All logic is in the [`chunk`](BunsetsuChunker::chunk) associated function;
/// the struct exists primarily to carry doc comments and serve as a future
/// extension point (e.g. for domain-specific chunking rules).
pub struct BunsetsuChunker;

impl BunsetsuChunker {
    /// Group `morphemes` into bunsetsu phrases.
    ///
    /// An empty slice produces an empty `Vec`.  Functional words that precede
    /// any content word are grouped into a leading bunsetsu of type `Other`
    /// rather than being dropped.
    pub fn chunk(morphemes: &[Morpheme]) -> Vec<Bunsetsu> {
        if morphemes.is_empty() {
            return Vec::new();
        }

        let mut result: Vec<Bunsetsu> = Vec::new();
        let mut start_idx: usize = 0;

        for (i, morpheme) in morphemes.iter().enumerate() {
            let pos = top_pos(morpheme);
            let is_content = is_content_word(pos);
            let is_singleton = is_singleton_symbol(pos);
            let is_functional = is_functional_word(pos);

            let should_start_new = if i == 0 {
                false // first morpheme always belongs to the first chunk
            } else if is_singleton {
                // symbols always begin a fresh bunsetsu (and end it immediately)
                true
            } else if is_content {
                // content words begin a fresh bunsetsu …
                true
            } else if !is_functional {
                // unknown / unclassified POS: start fresh
                true
            } else {
                // functional word → keep in current bunsetsu
                false
            };

            if should_start_new && start_idx < i {
                // Seal the bunsetsu that ended at i-1
                result.push(Self::build_chunk(morphemes, start_idx, i));
                start_idx = i;
            }

            // Singleton symbols always seal their own bunsetsu immediately
            if is_singleton && i == start_idx {
                result.push(Self::build_chunk(morphemes, i, i + 1));
                start_idx = i + 1;
            }
        }

        // Seal the final (possibly only) bunsetsu
        if start_idx < morphemes.len() {
            result.push(Self::build_chunk(morphemes, start_idx, morphemes.len()));
        }

        result
    }

    /// Build a `Bunsetsu` for `morphemes[start..end]`.
    fn build_chunk(morphemes: &[Morpheme], start: usize, end: usize) -> Bunsetsu {
        debug_assert!(start < end, "build_chunk: empty range");
        debug_assert!(end <= morphemes.len(), "build_chunk: end out of bounds");

        let head = &morphemes[start];
        let tail = &morphemes[end - 1];
        let chunk_type = bunsetsu_type_from_pos(top_pos(head));

        // Concatenate surfaces
        let surface: String = morphemes[start..end]
            .iter()
            .map(|m| m.surface.as_str())
            .collect();

        Bunsetsu {
            surface,
            start_byte: head.start_byte,
            end_byte: tail.end_byte,
            morpheme_range: start..end,
            head_idx: 0,
            chunk_type,
        }
    }
}

// ── Convenience method on AnalysisResult ────────────────────────────────────

use crate::types::AnalysisResult;

impl AnalysisResult {
    /// Segment this analysis result into bunsetsu (文節) phrases.
    ///
    /// This is a pure post-processing pass; it does not access the dictionary.
    pub fn bunsetsu(&self) -> Vec<Bunsetsu> {
        BunsetsuChunker::chunk(&self.morphemes)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_morpheme(surface: &str, feature: &str, start_byte: usize, end_byte: usize) -> Morpheme {
        Morpheme {
            surface: surface.to_string(),
            word_id: 0,
            pos_id: 0,
            wcost: 0,
            feature: feature.to_string(),
            entities: Vec::new(),
            pronunciation: None,
            embedding: None,
            start_byte,
            end_byte,
        }
    }

    /// 私は本を読む → [私/は][本/を][読む]
    #[test]
    fn test_basic_chunks_watashi() {
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
        let chunks = BunsetsuChunker::chunk(&morphemes);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].surface, "私は");
        assert_eq!(chunks[1].surface, "本を");
        assert_eq!(chunks[2].surface, "読む");
        assert_eq!(chunks[0].chunk_type, BunsetsuType::NounPhrase);
        assert_eq!(chunks[1].chunk_type, BunsetsuType::NounPhrase);
        assert_eq!(chunks[2].chunk_type, BunsetsuType::VerbPhrase);
    }

    #[test]
    fn test_empty_input_produces_empty_chunks() {
        let chunks = BunsetsuChunker::chunk(&[]);
        assert!(chunks.is_empty());
    }

    /// A leading particle (before any content word) becomes its own Other chunk.
    #[test]
    fn test_leading_functional_word() {
        let morphemes = vec![
            make_morpheme("は", "助詞,係助詞,*,*,*,*,は,ハ,ワ", 0, 3),
            make_morpheme("本", "名詞,一般,*,*,*,*,本,ホン,ホン", 3, 6),
        ];
        let chunks = BunsetsuChunker::chunk(&morphemes);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].surface, "は");
        assert_eq!(chunks[1].surface, "本");
    }

    /// Punctuation (記号) forms a singleton bunsetsu.
    #[test]
    fn test_punctuation_singleton() {
        let morphemes = vec![
            make_morpheme(
                "東京",
                "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ",
                0,
                6,
            ),
            make_morpheme("。", "記号,句点,*,*,*,*,。,。,。", 6, 9),
            make_morpheme(
                "大阪",
                "名詞,固有名詞,地域,一般,*,*,大阪,オオサカ,オオサカ",
                9,
                15,
            ),
        ];
        let chunks = BunsetsuChunker::chunk(&morphemes);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].surface, "東京");
        assert_eq!(chunks[1].surface, "。");
        assert_eq!(chunks[2].surface, "大阪");
    }

    /// Auxiliary verb (助動詞) attaches to preceding verb bunsetsu.
    #[test]
    fn test_auxiliary_verb_attaches() {
        let morphemes = vec![
            make_morpheme(
                "行き",
                "動詞,自立,*,*,五段・カ行促音便,連用形,行く,イキ,イキ",
                0,
                6,
            ),
            make_morpheme(
                "ます",
                "助動詞,*,*,*,特殊・マス,基本形,ます,マス,マス",
                6,
                12,
            ),
        ];
        let chunks = BunsetsuChunker::chunk(&morphemes);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].surface, "行きます");
        assert_eq!(chunks[0].chunk_type, BunsetsuType::VerbPhrase);
    }

    /// Byte spans are correctly propagated.
    #[test]
    fn test_byte_spans() {
        let morphemes = vec![
            make_morpheme(
                "東京",
                "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ",
                0,
                6,
            ),
            make_morpheme("に", "助詞,格助詞,一般,*,*,*,に,ニ,ニ", 6, 9),
        ];
        let chunks = BunsetsuChunker::chunk(&morphemes);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_byte, 0);
        assert_eq!(chunks[0].end_byte, 9);
    }

    /// Single-morpheme input (no functional words).
    #[test]
    fn test_single_morpheme() {
        let morphemes = vec![make_morpheme("犬", "名詞,一般,*,*,*,*,犬,イヌ,イヌ", 0, 3)];
        let chunks = BunsetsuChunker::chunk(&morphemes);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].surface, "犬");
        assert_eq!(chunks[0].morpheme_range, 0..1);
    }

    /// morpheme_range indices correctly index the original slice.
    #[test]
    fn test_morpheme_range_indices() {
        let morphemes = vec![
            make_morpheme("猫", "名詞,一般,*,*,*,*,猫,ネコ,ネコ", 0, 3),
            make_morpheme("が", "助詞,格助詞,一般,*,*,*,が,ガ,ガ", 3, 6),
            make_morpheme(
                "走る",
                "動詞,自立,*,*,五段・ラ行,基本形,走る,ハシル,ハシル",
                6,
                18,
            ),
        ];
        let chunks = BunsetsuChunker::chunk(&morphemes);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].morpheme_range, 0..2);
        assert_eq!(chunks[1].morpheme_range, 2..3);
        // Confirm the range can actually index the source
        let in_chunk: Vec<&str> = morphemes[chunks[0].morpheme_range.clone()]
            .iter()
            .map(|m| m.surface.as_str())
            .collect();
        assert_eq!(in_chunk, ["猫", "が"]);
    }
}
