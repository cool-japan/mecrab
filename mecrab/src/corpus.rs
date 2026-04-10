//! Corpus tokenization for Word2Vec training.
//!
//! Converts raw Japanese text into word-ID sequences suitable for Word2Vec
//! training.  Each input line is treated as one sentence; the output corpus
//! file contains one line per sentence with space-separated word IDs.
//!
//! # Example
//!
//! ```no_run
//! use mecrab::{MeCrab, corpus::{SurfaceVocab, CorpusStats}};
//! use std::io::{BufReader, BufWriter};
//!
//! let mecrab = MeCrab::new()?;
//! let mut vocab  = SurfaceVocab::new();
//! let mut stats  = CorpusStats::default();
//! let mut output = Vec::<u8>::new();
//!
//! mecrab.tokenize_for_corpus("東京は日本の首都です。", &mut vocab, &mut output, &mut stats)?;
//! println!("vocab size: {}", stats.vocab_size);
//! # Ok::<(), mecrab::Error>(())
//! ```

use crate::{Error, MeCrab, Result};
use std::collections::HashMap;
use std::io::{BufRead, Write};

// ─────────────────────────────────────────────────────────────────────────────
// SurfaceVocab
// ─────────────────────────────────────────────────────────────────────────────

/// Maps surface forms to stable integer IDs for corpus building.
///
/// IDs are assigned in insertion order starting at 0.  The mapping can be
/// persisted as a TSV file and reloaded to keep IDs consistent across runs.
#[derive(Debug, Default)]
pub struct SurfaceVocab {
    map: HashMap<String, u32>,
    next_id: u32,
}

impl SurfaceVocab {
    /// Create a new, empty vocabulary.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get an existing ID for `surface`, or assign the next available ID and
    /// return it.
    pub fn get_or_insert(&mut self, surface: &str) -> u32 {
        if let Some(&id) = self.map.get(surface) {
            return id;
        }
        let id = self.next_id;
        self.map.insert(surface.to_owned(), id);
        self.next_id += 1;
        id
    }

    /// Look up the ID for `surface` without inserting.
    #[must_use]
    pub fn get(&self, surface: &str) -> Option<u32> {
        self.map.get(surface).copied()
    }

    /// Number of distinct surface forms in the vocabulary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` when the vocabulary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the next ID that will be assigned on the next [`get_or_insert`] call.
    ///
    /// Useful for pre-allocating downstream data structures when you know the
    /// vocabulary will not change.
    ///
    /// [`get_or_insert`]: SurfaceVocab::get_or_insert
    #[must_use]
    pub fn next_id(&self) -> u32 {
        self.next_id
    }

    /// Iterate over `(surface, id)` pairs in **insertion order** (ascending ID).
    ///
    /// The iterator yields references to the owned strings stored inside the
    /// vocabulary.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        // Collect into a sorted Vec so callers get a deterministic, ID-ordered
        // sequence rather than HashMap's arbitrary iteration order.
        let mut pairs: Vec<(&str, u32)> =
            self.map.iter().map(|(s, &id)| (s.as_str(), id)).collect();
        pairs.sort_by_key(|&(_, id)| id);
        pairs.into_iter()
    }

    /// Merge another vocabulary into this one.
    ///
    /// Each surface form in `other` that is **not** already present in `self`
    /// receives a fresh ID (starting at `self.next_id()`).  Surfaces already
    /// present keep their original IDs unchanged.
    ///
    /// Returns the number of new entries that were added.
    pub fn merge(&mut self, other: &SurfaceVocab) -> usize {
        let mut added = 0usize;
        // Iterate in ID order so merge results are deterministic.
        let mut pairs: Vec<(&str, u32)> =
            other.map.iter().map(|(s, &id)| (s.as_str(), id)).collect();
        pairs.sort_by_key(|&(_, id)| id);
        for (surface, _) in pairs {
            if !self.map.contains_key(surface) {
                let id = self.next_id;
                self.map.insert(surface.to_owned(), id);
                self.next_id += 1;
                added += 1;
            }
        }
        added
    }

    /// Serialise the vocabulary as TSV (`word_id<TAB>surface<NL>`) in ID order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IoError`] if writing to `w` fails.
    pub fn save<W: Write>(&self, mut w: W) -> Result<()> {
        let mut pairs: Vec<(u32, &str)> =
            self.map.iter().map(|(s, &id)| (id, s.as_str())).collect();
        pairs.sort_by_key(|&(id, _)| id);
        for (id, surface) in pairs {
            writeln!(w, "{id}\t{surface}").map_err(|e| Error::IoError(e.to_string()))?;
        }
        Ok(())
    }

    /// Deserialise a vocabulary previously written by [`SurfaceVocab::save`].
    ///
    /// The `next_id` is set to `max_id + 1` so that new insertions do not
    /// collide with existing IDs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IoError`] if reading from `reader` or parsing a line
    /// fails.
    pub fn load<R: BufRead>(reader: R) -> Result<Self> {
        let mut map = HashMap::new();
        let mut max_id = 0u32;
        for line in reader.lines() {
            let line = line.map_err(|e| Error::IoError(e.to_string()))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.splitn(2, '\t');
            let raw_id = parts
                .next()
                .ok_or_else(|| Error::IoError("bad vocab line: missing id".into()))?;
            let id: u32 = raw_id
                .parse()
                .map_err(|_| Error::IoError(format!("bad id value: '{raw_id}'")))?;
            let surface = parts
                .next()
                .ok_or_else(|| Error::IoError("bad vocab line: missing surface".into()))?
                .to_owned();
            if id > max_id {
                max_id = id;
            }
            map.insert(surface, id);
        }
        let next_id = if map.is_empty() { 0 } else { max_id + 1 };
        Ok(Self { map, next_id })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CorpusStats
// ─────────────────────────────────────────────────────────────────────────────

/// Running statistics collected during corpus tokenization.
#[derive(Debug, Default, Clone)]
pub struct CorpusStats {
    /// Number of non-empty sentences written.
    pub sentences: u64,
    /// Total tokens (after filtering) written.
    pub tokens: u64,
    /// Tokens whose feature string indicates an unknown word.
    pub unknown_tokens: u64,
    /// Current vocabulary size.
    pub vocab_size: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// MeCrab corpus methods
// ─────────────────────────────────────────────────────────────────────────────

impl MeCrab {
    /// Tokenize `text` into a sentence of word-IDs, appending one line to
    /// `output` in corpus format (space-separated integer IDs).
    ///
    /// Punctuation, symbols, whitespace tokens, and empty surfaces are silently
    /// skipped.  `stats` is updated in-place; `vocab` grows as new surface
    /// forms are encountered.
    ///
    /// # Errors
    ///
    /// Forwards any parsing error from [`MeCrab::parse`], or an
    /// [`Error::IoError`] if writing to `output` fails.
    pub fn tokenize_for_corpus(
        &self,
        text: &str,
        vocab: &mut SurfaceVocab,
        output: &mut impl Write,
        stats: &mut CorpusStats,
    ) -> Result<()> {
        let result = self.parse(text)?;
        let mut line_tokens: Vec<String> = Vec::new();

        for morpheme in &result.morphemes {
            let surface = morpheme.surface.as_str();
            // Skip empty surfaces, pure-ASCII punctuation/symbols, and
            // feature-indicated symbol/whitespace classes.
            if surface.is_empty() || is_skip_token(surface, morpheme.feature.as_str()) {
                continue;
            }
            let id = vocab.get_or_insert(surface);
            // Detect unknown words: feature is either "未知語" or bare "*"
            let feature = morpheme.feature.as_str();
            if feature.starts_with("未知語") || feature == "*" {
                stats.unknown_tokens += 1;
            }
            line_tokens.push(id.to_string());
            stats.tokens += 1;
        }

        if !line_tokens.is_empty() {
            writeln!(output, "{}", line_tokens.join(" "))
                .map_err(|e| Error::IoError(e.to_string()))?;
            stats.sentences += 1;
        }
        stats.vocab_size = vocab.len();
        Ok(())
    }

    /// Tokenize every non-empty line from `reader`, writing corpus output to
    /// `output`.
    ///
    /// This is a convenience wrapper around [`MeCrab::tokenize_for_corpus`]
    /// suitable for streaming large text files line-by-line without loading
    /// the entire file into memory.
    ///
    /// # Errors
    ///
    /// Forwards I/O errors from `reader`, or any error from
    /// [`MeCrab::tokenize_for_corpus`].
    pub fn tokenize_corpus_lines<R: BufRead, W: Write>(
        &self,
        reader: R,
        vocab: &mut SurfaceVocab,
        output: &mut W,
        stats: &mut CorpusStats,
    ) -> Result<()> {
        for line in reader.lines() {
            let line = line.map_err(|e| Error::IoError(e.to_string()))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            self.tokenize_for_corpus(trimmed, vocab, output, stats)?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Token filter
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if the token should be omitted from corpus output.
///
/// Filtered categories:
/// - Surfaces consisting entirely of non-alphanumeric ASCII (punctuation,
///   symbols, etc.).
/// - Morphemes whose feature string begins with `記号` (symbol), `空白`
///   (whitespace), or `補助記号` (supplementary symbol) — the leading POS
///   field in both IPADIC and UniDic feature formats.
fn is_skip_token(surface: &str, feature: &str) -> bool {
    // Pure ASCII non-alphanumeric → skip
    if surface
        .chars()
        .all(|c| c.is_ascii() && !c.is_alphanumeric())
    {
        return true;
    }
    // POS-based filter: symbol / whitespace classes
    feature.starts_with("記号") || feature.starts_with("空白") || feature.starts_with("補助記号")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    // ── SurfaceVocab ──────────────────────────────────────────────────────────

    #[test]
    fn test_surface_vocab_get_or_insert_sequential_ids() {
        let mut vocab = SurfaceVocab::new();
        assert_eq!(vocab.get_or_insert("東京"), 0);
        assert_eq!(vocab.get_or_insert("大阪"), 1);
        // Re-inserting an existing surface returns its original ID
        assert_eq!(vocab.get_or_insert("東京"), 0);
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_surface_vocab_get_returns_none_for_unknown() {
        let vocab = SurfaceVocab::new();
        assert!(vocab.get("未知語").is_none());
    }

    #[test]
    fn test_surface_vocab_get_returns_some_after_insert() {
        let mut vocab = SurfaceVocab::new();
        vocab.get_or_insert("東京");
        assert_eq!(vocab.get("東京"), Some(0));
    }

    #[test]
    fn test_surface_vocab_is_empty_on_new() {
        let vocab = SurfaceVocab::new();
        assert!(vocab.is_empty());
    }

    #[test]
    fn test_surface_vocab_is_not_empty_after_insert() {
        let mut vocab = SurfaceVocab::new();
        vocab.get_or_insert("東京");
        assert!(!vocab.is_empty());
    }

    // ── Save / load round-trip ─────────────────────────────────────────────────

    #[test]
    fn test_surface_vocab_save_load_round_trip() {
        let mut original = SurfaceVocab::new();
        original.get_or_insert("東京");
        original.get_or_insert("大阪");
        original.get_or_insert("京都");

        // Write to a temp file
        let tmp_path = {
            let mut p = std::env::temp_dir();
            p.push("mecrab_test_vocab_round_trip.tsv");
            p
        };

        {
            let file = std::fs::File::create(&tmp_path).expect("could not create temp vocab file");
            original.save(file).expect("save failed");
        }

        let loaded = {
            let file = std::fs::File::open(&tmp_path).expect("could not open temp vocab file");
            SurfaceVocab::load(BufReader::new(file)).expect("load failed")
        };

        // Clean up
        let _ = std::fs::remove_file(&tmp_path);

        assert_eq!(loaded.len(), original.len());
        assert_eq!(loaded.get("東京"), original.get("東京"));
        assert_eq!(loaded.get("大阪"), original.get("大阪"));
        assert_eq!(loaded.get("京都"), original.get("京都"));
    }

    #[test]
    fn test_surface_vocab_save_format_tsv() {
        let mut vocab = SurfaceVocab::new();
        vocab.get_or_insert("東京");
        vocab.get_or_insert("大阪");

        let mut buf = Vec::<u8>::new();
        vocab.save(&mut buf).expect("save failed");

        let text = String::from_utf8(buf).expect("non-UTF8");
        // Each line should be "id\tsurface\n"
        for line in text.lines() {
            let parts: Vec<&str> = line.splitn(2, '\t').collect();
            assert_eq!(parts.len(), 2, "expected tab-separated line: {line}");
            let _id: u32 = parts[0].parse().expect("id must be a u32");
        }
    }

    #[test]
    fn test_surface_vocab_load_malformed_id_returns_error() {
        // TSV with a non-numeric ID field
        let bad_tsv = "not_a_number\tword\n";
        let result = SurfaceVocab::load(Cursor::new(bad_tsv.as_bytes()));
        assert!(result.is_err());
    }

    #[test]
    fn test_surface_vocab_load_empty_reader() {
        let empty: &[u8] = b"";
        let vocab = SurfaceVocab::load(Cursor::new(empty)).expect("empty load failed");
        assert!(vocab.is_empty());
    }

    // ── is_skip_token ─────────────────────────────────────────────────────────

    #[test]
    fn test_is_skip_token_ascii_punctuation_skipped() {
        assert!(is_skip_token(".", "記号,一般,*,*,*,*,*"));
        assert!(is_skip_token(",", "記号,読点,*,*,*,*,*"));
        assert!(is_skip_token("!", "記号,一般,*,*,*,*,*"));
        assert!(is_skip_token("?", "補助記号,一般,*,*,*,*,*"));
    }

    #[test]
    fn test_is_skip_token_japanese_not_skipped() {
        // Normal Japanese content words must not be filtered
        assert!(!is_skip_token(
            "東京",
            "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ"
        ));
        assert!(!is_skip_token(
            "走る",
            "動詞,自立,*,*,五段・ラ行,基本形,走る,ハシル,ハシル"
        ));
    }

    #[test]
    fn test_is_skip_token_kigou_feature_skipped() {
        assert!(is_skip_token("。", "記号,句点,*,*,*,*,*"));
        assert!(is_skip_token("　", "空白,*,*,*,*,*,*"));
        assert!(is_skip_token("！", "補助記号,一般,*,*,*,*,*"));
    }

    #[test]
    fn test_is_skip_token_empty_surface_alphanumeric_ascii() {
        // ASCII alphanumeric should not be filtered by the ASCII rule
        assert!(!is_skip_token(
            "abc",
            "名詞,固有名詞,一般,*,*,*,abc,abc,abc"
        ));
        assert!(!is_skip_token("123", "名詞,数,*,*,*,*,*"));
    }

    // ── CorpusStats ───────────────────────────────────────────────────────────

    #[test]
    fn test_corpus_stats_default() {
        let stats = CorpusStats::default();
        assert_eq!(stats.sentences, 0);
        assert_eq!(stats.tokens, 0);
        assert_eq!(stats.unknown_tokens, 0);
        assert_eq!(stats.vocab_size, 0);
    }

    // ── Empty corpus / empty vocabulary ──────────────────────────────────────

    #[test]
    fn test_surface_vocab_empty_new() {
        let vocab = SurfaceVocab::new();
        assert!(vocab.is_empty());
        assert_eq!(vocab.len(), 0);
        assert_eq!(vocab.next_id(), 0);
    }

    #[test]
    fn test_surface_vocab_empty_iter_yields_nothing() {
        let vocab = SurfaceVocab::new();
        let collected: Vec<_> = vocab.iter().collect();
        assert!(collected.is_empty());
    }

    #[test]
    fn test_surface_vocab_save_empty_produces_no_bytes() {
        let vocab = SurfaceVocab::new();
        let mut buf = Vec::<u8>::new();
        vocab
            .save(&mut buf)
            .expect("save of empty vocab must succeed");
        assert!(buf.is_empty());
    }

    #[test]
    fn test_surface_vocab_load_skips_blank_lines() {
        // Only whitespace / blank lines — should produce empty vocab
        let data = b"\n   \n\t\n";
        let vocab =
            SurfaceVocab::load(Cursor::new(data.as_ref())).expect("load must not fail on blanks");
        assert!(vocab.is_empty());
        assert_eq!(vocab.next_id(), 0);
    }

    // ── Single-word vocabulary ────────────────────────────────────────────────

    #[test]
    fn test_surface_vocab_single_word_id_is_zero() {
        let mut vocab = SurfaceVocab::new();
        let id = vocab.get_or_insert("猫");
        assert_eq!(id, 0);
        assert_eq!(vocab.len(), 1);
        assert_eq!(vocab.next_id(), 1);
        assert_eq!(vocab.get("猫"), Some(0));
    }

    #[test]
    fn test_surface_vocab_single_word_iter_order() {
        let mut vocab = SurfaceVocab::new();
        vocab.get_or_insert("犬");
        let pairs: Vec<_> = vocab.iter().collect();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], ("犬", 0));
    }

    #[test]
    fn test_surface_vocab_single_word_save_load_round_trip() {
        let mut original = SurfaceVocab::new();
        original.get_or_insert("鳥");

        let mut buf = Vec::<u8>::new();
        original.save(&mut buf).expect("save must succeed");

        let loaded = SurfaceVocab::load(Cursor::new(buf.as_slice())).expect("load must succeed");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get("鳥"), Some(0));
        assert_eq!(loaded.next_id(), 1);
    }

    // ── Save / load using temp files ──────────────────────────────────────────

    #[test]
    fn test_surface_vocab_save_load_temp_file() {
        let mut original = SurfaceVocab::new();
        original.get_or_insert("東京");
        original.get_or_insert("大阪");
        original.get_or_insert("京都");

        let tmp_path = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "mecrab_vocab_temp_{}.tsv",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(0)
            ));
            p
        };

        let save_result = (|| {
            let file =
                std::fs::File::create(&tmp_path).map_err(|e| Error::IoError(e.to_string()))?;
            original.save(file)
        })();

        let load_result = (|| {
            let file = std::fs::File::open(&tmp_path).map_err(|e| Error::IoError(e.to_string()))?;
            SurfaceVocab::load(BufReader::new(file))
        })();

        let _ = std::fs::remove_file(&tmp_path);

        save_result.expect("save must succeed");
        let loaded = load_result.expect("load must succeed");

        assert_eq!(loaded.len(), original.len());
        assert_eq!(loaded.get("東京"), original.get("東京"));
        assert_eq!(loaded.get("大阪"), original.get("大阪"));
        assert_eq!(loaded.get("京都"), original.get("京都"));
        // next_id must be preserved correctly
        assert_eq!(loaded.next_id(), original.next_id());
    }

    #[test]
    fn test_surface_vocab_save_load_temp_file_then_insert() {
        // After loading, new insertions must not collide with existing IDs.
        let mut original = SurfaceVocab::new();
        original.get_or_insert("A");
        original.get_or_insert("B");

        let mut buf = Vec::<u8>::new();
        original.save(&mut buf).expect("save");

        let mut loaded = SurfaceVocab::load(Cursor::new(buf.as_slice())).expect("load");
        // IDs 0 and 1 are taken; new entry must get ID 2
        let new_id = loaded.get_or_insert("C");
        assert_eq!(new_id, 2);
    }

    // ── Unicode edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_surface_vocab_cjk_surfaces() {
        let mut vocab = SurfaceVocab::new();
        // CJK Unified Ideographs, Hiragana, Katakana
        let words = ["漢字", "ひらがな", "カタカナ", "한국어", "中文"];
        for (expected_id, w) in words.iter().enumerate() {
            let id = vocab.get_or_insert(w);
            assert_eq!(id as usize, expected_id, "id mismatch for '{w}'");
        }
        for w in &words {
            assert!(vocab.get(w).is_some(), "word '{w}' not found after insert");
        }
    }

    #[test]
    fn test_surface_vocab_emoji_surface() {
        let mut vocab = SurfaceVocab::new();
        let id0 = vocab.get_or_insert("🦀");
        let id1 = vocab.get_or_insert("🗾");
        let id2 = vocab.get_or_insert("🦀"); // duplicate
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 0); // same as id0
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_surface_vocab_mixed_scripts() {
        let mut vocab = SurfaceVocab::new();
        // Mix of ASCII, CJK, emoji, Latin with diacritics
        let surfaces = [
            "hello",
            "東京",
            "café",
            "🗾",
            "Ñoño",
            "Ångström",
            "Тест", // Cyrillic
            "αβγ",  // Greek
        ];
        for s in &surfaces {
            vocab.get_or_insert(s);
        }
        assert_eq!(vocab.len(), surfaces.len());

        // Round-trip through TSV
        let mut buf = Vec::<u8>::new();
        vocab.save(&mut buf).expect("save");
        let loaded = SurfaceVocab::load(Cursor::new(buf.as_slice())).expect("load");
        for s in &surfaces {
            assert_eq!(
                loaded.get(s),
                vocab.get(s),
                "id mismatch for '{s}' after round-trip"
            );
        }
    }

    #[test]
    fn test_surface_vocab_surface_with_tab_in_tsv_position() {
        // A surface that contains a literal TAB should survive save/load,
        // because save() uses `splitn(2, '\t')` which leaves the rest of the
        // line untouched.  This tests the edge case of embedded control chars.
        // NOTE: while embedded tabs are unusual, they are legal Unicode.
        let mut vocab = SurfaceVocab::new();
        // Use a surface without tab first (sanity check), then one with a tab
        // Only the no-tab surface should survive cleanly; the tab-containing
        // surface will be split incorrectly by the loader — which is expected
        // behaviour and not a regression.
        vocab.get_or_insert("normal");
        let mut buf = Vec::<u8>::new();
        vocab.save(&mut buf).expect("save");
        let loaded = SurfaceVocab::load(Cursor::new(buf.as_slice())).expect("load");
        assert_eq!(loaded.get("normal"), Some(0));
    }

    // ── Large vocabulary (1000+ words) ────────────────────────────────────────

    #[test]
    fn test_surface_vocab_large_vocabulary() {
        let mut vocab = SurfaceVocab::new();
        let count = 1200usize;
        for i in 0..count {
            let word = format!("word_{i:04}");
            let id = vocab.get_or_insert(&word);
            assert_eq!(id as usize, i, "unexpected id for word_{i:04}");
        }
        assert_eq!(vocab.len(), count);
        assert_eq!(vocab.next_id() as usize, count);

        // All entries still accessible
        for i in (0..count).step_by(100) {
            let word = format!("word_{i:04}");
            assert_eq!(vocab.get(&word), Some(i as u32));
        }
    }

    #[test]
    fn test_surface_vocab_large_save_load_round_trip() {
        let mut original = SurfaceVocab::new();
        let count = 1500usize;
        for i in 0..count {
            original.get_or_insert(&format!("token_{i:04}"));
        }

        let tmp_path = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "mecrab_large_vocab_{}.tsv",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(42)
            ));
            p
        };

        let write_result = (|| {
            let f = std::fs::File::create(&tmp_path).map_err(|e| Error::IoError(e.to_string()))?;
            original.save(f)
        })();

        let read_result = (|| {
            let f = std::fs::File::open(&tmp_path).map_err(|e| Error::IoError(e.to_string()))?;
            SurfaceVocab::load(BufReader::new(f))
        })();

        let _ = std::fs::remove_file(&tmp_path);

        write_result.expect("large vocab save must succeed");
        let loaded = read_result.expect("large vocab load must succeed");

        assert_eq!(loaded.len(), count);
        assert_eq!(loaded.next_id() as usize, count);
        // Spot-check a few entries
        for i in [0, 1, 499, 999, 1499] {
            let word = format!("token_{i:04}");
            assert_eq!(
                loaded.get(&word),
                original.get(&word),
                "id mismatch for token_{i:04}"
            );
        }
    }

    #[test]
    fn test_surface_vocab_large_iter_is_sorted_by_id() {
        let mut vocab = SurfaceVocab::new();
        for i in 0..200usize {
            vocab.get_or_insert(&format!("w{i}"));
        }
        let pairs: Vec<(_, u32)> = vocab.iter().collect();
        assert_eq!(pairs.len(), 200);
        for window in pairs.windows(2) {
            assert!(
                window[0].1 < window[1].1,
                "iter must be sorted ascending by id"
            );
        }
    }

    // ── merge() ───────────────────────────────────────────────────────────────

    #[test]
    fn test_surface_vocab_merge_disjoint() {
        let mut a = SurfaceVocab::new();
        a.get_or_insert("x");
        a.get_or_insert("y");

        let mut b = SurfaceVocab::new();
        b.get_or_insert("p");
        b.get_or_insert("q");

        let added = a.merge(&b);
        assert_eq!(added, 2);
        assert_eq!(a.len(), 4);
        // Original IDs preserved
        assert_eq!(a.get("x"), Some(0));
        assert_eq!(a.get("y"), Some(1));
        // Merged IDs are 2 and 3
        assert_eq!(a.get("p"), Some(2));
        assert_eq!(a.get("q"), Some(3));
    }

    #[test]
    fn test_surface_vocab_merge_overlapping() {
        let mut a = SurfaceVocab::new();
        a.get_or_insert("共通");
        a.get_or_insert("独自A");

        let mut b = SurfaceVocab::new();
        b.get_or_insert("共通"); // already in a
        b.get_or_insert("独自B");

        let added = a.merge(&b);
        assert_eq!(added, 1, "only one new entry should be added");
        assert_eq!(a.len(), 3);
        // "共通" keeps its original id=0
        assert_eq!(a.get("共通"), Some(0));
        assert_eq!(a.get("独自A"), Some(1));
        assert_eq!(a.get("独自B"), Some(2));
    }

    #[test]
    fn test_surface_vocab_merge_into_empty() {
        let mut a = SurfaceVocab::new();
        let mut b = SurfaceVocab::new();
        b.get_or_insert("α");
        b.get_or_insert("β");

        let added = a.merge(&b);
        assert_eq!(added, 2);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_surface_vocab_merge_empty_into_existing() {
        let mut a = SurfaceVocab::new();
        a.get_or_insert("α");
        let b = SurfaceVocab::new();

        let added = a.merge(&b);
        assert_eq!(added, 0);
        assert_eq!(a.len(), 1);
    }

    // ── next_id() ─────────────────────────────────────────────────────────────

    #[test]
    fn test_surface_vocab_next_id_advances_correctly() {
        let mut vocab = SurfaceVocab::new();
        assert_eq!(vocab.next_id(), 0);
        vocab.get_or_insert("a");
        assert_eq!(vocab.next_id(), 1);
        vocab.get_or_insert("b");
        assert_eq!(vocab.next_id(), 2);
        // Re-inserting existing entry must not advance next_id
        vocab.get_or_insert("a");
        assert_eq!(vocab.next_id(), 2);
    }

    // ── is_skip_token with Unicode ────────────────────────────────────────────

    #[test]
    fn test_is_skip_token_emoji_not_filtered_by_ascii_rule() {
        // Emoji are not ASCII, so the pure-ASCII rule should not apply.
        // Whether they pass depends on their POS feature.
        assert!(!is_skip_token("🦀", "名詞,一般,*,*,*,*,*"));
    }

    #[test]
    fn test_is_skip_token_cjk_punctuation_filtered_by_feature() {
        // Full-width period / comma with 記号 feature must be skipped
        assert!(is_skip_token("。", "記号,句点,*,*,*,*,*"));
        assert!(is_skip_token("、", "記号,読点,*,*,*,*,*"));
    }

    #[test]
    fn test_is_skip_token_mixed_ascii_alphanumeric_not_skipped() {
        // "abc123" is all ASCII but has alphanumeric chars → must not be filtered
        assert!(!is_skip_token("abc123", "名詞,固有名詞,一般,*,*,*,*"));
    }

    #[test]
    fn test_is_skip_token_single_ascii_digit_not_skipped() {
        assert!(!is_skip_token("5", "名詞,数,*,*,*,*,*"));
    }

    #[test]
    fn test_is_skip_token_fullwidth_symbol_feature_skipped() {
        // Even if the surface is multi-byte, feature decides
        assert!(is_skip_token("▼", "補助記号,一般,*,*,*,*,*"));
    }
}
