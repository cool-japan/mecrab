//! Byte-Pair Encoding (BPE) vocabulary trainer for mecrab-word2vec.
//!
//! Implements an efficient priority-queue-based BPE merge trainer with lazy
//! deletion, avoiding the naive O(N×M) per-merge approach.
//!
//! # Algorithm overview
//!
//! 1. **Tokenize** corpus into character-level sequences with `</w>` end markers.
//! 2. **Build** initial pair frequency map over the word-type vocabulary.
//! 3. **Merge loop** with a max-heap and lazy deletion:
//!    - Pop the highest-frequency pair.
//!    - If the stored frequency is stale (pair_freq diverged), skip.
//!    - Apply merge to every word containing the pair; update neighbour counts.
//!    - Push updated pairs back into heap.
//!    - Repeat until `vocab_size + merges_done >= target_vocab_size`.
//! 4. **Build** final symbol→ID table.

use crate::{Result, Word2VecError};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::Path;

// ── Symbol table ─────────────────────────────────────────────────────────────

/// Compact integer ID for a symbol string (character or merged token).
type SymId = u32;

// ── Public types ─────────────────────────────────────────────────────────────

/// One BPE merge rule: the pair `(left, right)` is collapsed into `merged`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeMerge {
    /// Left symbol of the merge pair.
    pub left: String,
    /// Right symbol of the merge pair.
    pub right: String,
    /// The merged symbol (equals `left + right`).
    pub merged: String,
    /// Pair frequency at the time this merge was selected.
    pub freq: u64,
}

/// Trained BPE vocabulary: ordered merge rules plus a symbol-to-ID map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeVocab {
    /// Merge rules in application order (first rule applied first during encoding).
    pub merges: Vec<BpeMerge>,
    /// Symbol → integer ID mapping (special tokens, base chars, merged tokens).
    pub vocab: HashMap<String, u32>,
}

impl BpeVocab {
    /// Serialize to JSON and write to `path`.
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| Word2VecError::Training(format!("JSON serialization failed: {e}")))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Deserialize from a JSON file at `path`.
    pub fn load_json(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let vocab: Self = serde_json::from_str(&content)
            .map_err(|e| Word2VecError::Training(format!("JSON deserialization failed: {e}")))?;
        Ok(vocab)
    }

    /// Encode `text` (a space-separated sequence of morpheme surfaces) using
    /// the trained merge rules.
    ///
    /// The input is first converted to a character-level token sequence with
    /// `</w>` end-of-word markers, then each merge rule is applied left-to-right.
    pub fn encode(&self, text: &str) -> Vec<String> {
        // Build a fast merge-pair → merged-symbol lookup for O(1) lookup.
        let merge_map: HashMap<(&str, &str), &str> = self
            .merges
            .iter()
            .map(|m| ((m.left.as_str(), m.right.as_str()), m.merged.as_str()))
            .collect();

        // Tokenize input into character-level tokens per morpheme.
        let mut tokens: Vec<String> = Vec::new();
        for (word_idx, word) in text.split_whitespace().enumerate() {
            if word_idx > 0 {
                // morpheme boundary is implicit; no separator needed
            }
            // Explode morpheme into chars + end-of-word marker.
            let chars: Vec<char> = word.chars().collect();
            for (i, ch) in chars.iter().enumerate() {
                if i + 1 == chars.len() {
                    // Last character of the morpheme gets the </w> suffix.
                    tokens.push(format!("{ch}</w>"));
                } else {
                    tokens.push(ch.to_string());
                }
            }
        }

        if tokens.is_empty() {
            return tokens;
        }

        // Apply each merge rule in order (greedy left-to-right scan).
        for merge in &self.merges {
            let left = merge.left.as_str();
            let right = merge.right.as_str();

            if !merge_map.contains_key(&(left, right)) {
                continue; // defensive, should not happen
            }

            let mut i = 0;
            let mut new_tokens: Vec<String> = Vec::with_capacity(tokens.len());
            while i < tokens.len() {
                if i + 1 < tokens.len()
                    && tokens[i].as_str() == left
                    && tokens[i + 1].as_str() == right
                {
                    new_tokens.push(merge.merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        tokens
    }
}

// ── Trainer ───────────────────────────────────────────────────────────────────

/// BPE vocabulary trainer.
///
/// ```
/// use mecrab_word2vec::BpeTrainer;
///
/// let corpus = vec![
///     "東京 は 日本 の 首都 です".to_string(),
///     "大阪 は 日本 の 第二 の 都市 です".to_string(),
/// ];
/// let vocab = BpeTrainer::new(500).train(corpus.into_iter()).unwrap();
/// assert!(!vocab.vocab.is_empty());
/// ```
pub struct BpeTrainer {
    /// Desired vocabulary size (number of symbols after training).
    target_vocab_size: usize,
    /// Minimum frequency for a symbol to be included in the base vocabulary.
    min_frequency: u64,
    /// Special tokens prepended before base characters.
    special_tokens: Vec<String>,
}

impl BpeTrainer {
    /// Create a new trainer targeting `target_vocab_size` total symbols.
    pub fn new(target_vocab_size: usize) -> Self {
        Self {
            target_vocab_size,
            min_frequency: 2,
            special_tokens: vec![
                "[UNK]".to_string(),
                "[BOS]".to_string(),
                "[EOS]".to_string(),
            ],
        }
    }

    /// Override the minimum pair frequency required to perform a merge.
    pub fn with_min_frequency(mut self, freq: u64) -> Self {
        self.min_frequency = freq;
        self
    }

    /// Replace the default special-token list.
    pub fn with_special_tokens(mut self, tokens: Vec<String>) -> Self {
        self.special_tokens = tokens;
        self
    }

    /// Train BPE from an iterator of corpus lines.
    ///
    /// Each line should be a space-separated sequence of morpheme surfaces
    /// (e.g., as produced by `kizame vectors tokenize`).
    pub fn train(&self, lines: impl Iterator<Item = String>) -> Result<BpeVocab> {
        // ── Phase 1: build word-type frequency map ────────────────────────────
        // Each "word" is a morpheme surface converted to char-level tokens.
        // We collapse identical sequences to reduce memory.
        let word_freqs = build_word_freqs(lines);

        if word_freqs.is_empty() {
            return Ok(BpeVocab {
                merges: Vec::new(),
                vocab: self.build_vocab_map(&[], &HashSet::new()),
            });
        }

        // ── Phase 2 & 3: efficient merge loop ────────────────────────────────
        let mut state = BpeState::new(word_freqs, &self.special_tokens)?;
        let initial_vocab_size = state.num_base_symbols();
        let budget = self
            .target_vocab_size
            .saturating_sub(self.special_tokens.len() + initial_vocab_size);

        let mut merges: Vec<BpeMerge> = Vec::new();

        loop {
            if merges.len() >= budget {
                break;
            }

            match state.find_best_pair() {
                None => break,
                Some((freq, pair)) if freq < self.min_frequency => {
                    let _ = (freq, pair);
                    break;
                }
                Some((freq, pair)) => {
                    let sym_a = &state.sym_table[pair.0 as usize].clone();
                    let sym_b = &state.sym_table[pair.1 as usize].clone();
                    let merged = format!("{sym_a}{sym_b}");

                    let merged_id = state.apply_merge(pair, &merged)?;
                    let _ = merged_id;

                    merges.push(BpeMerge {
                        left: sym_a.clone(),
                        right: sym_b.clone(),
                        merged: merged.clone(),
                        freq,
                    });
                }
            }
        }

        // ── Phase 4: build final vocab map ───────────────────────────────────
        let base_syms: HashSet<String> = state
            .sym_table
            .iter()
            .take(state.num_base_symbols())
            .cloned()
            .collect();
        let vocab_map = self.build_vocab_map(&merges, &base_syms);

        Ok(BpeVocab {
            merges,
            vocab: vocab_map,
        })
    }

    /// Assign integer IDs: special tokens first, then base chars, then merged.
    fn build_vocab_map(
        &self,
        merges: &[BpeMerge],
        base_syms: &HashSet<String>,
    ) -> HashMap<String, u32> {
        let mut map: HashMap<String, u32> = HashMap::new();
        let mut next_id: u32 = 0;

        for tok in &self.special_tokens {
            map.entry(tok.clone()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
        }

        // Base symbols — sorted for determinism.
        let mut base_sorted: Vec<&String> = base_syms.iter().collect();
        base_sorted.sort_unstable();
        for sym in base_sorted {
            map.entry(sym.clone()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
        }

        for m in merges {
            map.entry(m.merged.clone()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
        }

        map
    }
}

// ── Internal BPE state ────────────────────────────────────────────────────────

/// A word entry in the BPE corpus: frequency and current token sequence.
struct WordEntry {
    count: u64,
    tokens: Vec<SymId>,
}

/// All mutable state for the BPE merge loop.
struct BpeState {
    /// sym_id → string representation
    pub sym_table: Vec<String>,
    /// string → sym_id (intern table)
    sym_intern: HashMap<String, SymId>,
    /// Number of base symbols (single characters + `</w>` variants) before any merge.
    base_sym_count: usize,
    /// word-type corpus
    word_entries: Vec<WordEntry>,
    /// pair → total frequency across corpus
    pair_freq: HashMap<(SymId, SymId), u64>,
    /// pair → set of word indices that contain at least one occurrence
    pair_words: HashMap<(SymId, SymId), HashSet<usize>>,
    /// max-heap (via Reverse min-heap trick): entries are `Reverse((freq, sym_a, sym_b))`
    heap: BinaryHeap<Reverse<HeapEntry>>,
}

/// Heap entry ordered by frequency (descending via Reverse<…>).
#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct HeapEntry {
    /// Negated frequency so that max-freq gives smallest Reverse value.
    neg_freq: i64,
    sym_a: SymId,
    sym_b: SymId,
}

impl BpeState {
    fn new(word_freqs: HashMap<Vec<String>, u64>, special_tokens: &[String]) -> Result<Self> {
        let mut sym_table: Vec<String> = Vec::new();
        let mut sym_intern: HashMap<String, SymId> = HashMap::new();

        // Pre-intern special tokens so they have well-known IDs but do NOT
        // appear in the character-level token sequences themselves.
        for tok in special_tokens {
            if !sym_intern.contains_key(tok) {
                let id = sym_table.len() as SymId;
                sym_intern.insert(tok.clone(), id);
                sym_table.push(tok.clone());
            }
        }

        // Convert word_freqs into WordEntry list, interning all char symbols.
        let mut word_entries: Vec<WordEntry> = Vec::with_capacity(word_freqs.len());
        for (char_seq, count) in word_freqs {
            let mut token_ids: Vec<SymId> = Vec::with_capacity(char_seq.len());
            for sym in &char_seq {
                let id = if let Some(&existing) = sym_intern.get(sym) {
                    existing
                } else {
                    let id = sym_table.len() as SymId;
                    sym_intern.insert(sym.clone(), id);
                    sym_table.push(sym.clone());
                    id
                };
                token_ids.push(id);
            }
            word_entries.push(WordEntry {
                count,
                tokens: token_ids,
            });
        }

        // The number of base symbols = everything interned so far (including
        // special tokens + all single-char symbols).
        let base_sym_count = sym_table.len();

        // ── Build pair_freq and pair_words ───────────────────────────────────
        let mut pair_freq: HashMap<(SymId, SymId), u64> = HashMap::new();
        let mut pair_words: HashMap<(SymId, SymId), HashSet<usize>> = HashMap::new();

        for (word_idx, entry) in word_entries.iter().enumerate() {
            for window in entry.tokens.windows(2) {
                let pair = (window[0], window[1]);
                *pair_freq.entry(pair).or_insert(0) += entry.count;
                pair_words.entry(pair).or_default().insert(word_idx);
            }
        }

        // ── Populate heap ────────────────────────────────────────────────────
        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(pair_freq.len());
        for (&(a, b), &freq) in &pair_freq {
            heap.push(Reverse(HeapEntry {
                neg_freq: -(freq as i64),
                sym_a: a,
                sym_b: b,
            }));
        }

        Ok(Self {
            sym_table,
            sym_intern,
            base_sym_count,
            word_entries,
            pair_freq,
            pair_words,
            heap,
        })
    }

    /// Number of base (pre-merge) symbols.
    pub fn num_base_symbols(&self) -> usize {
        self.base_sym_count
    }

    /// Pop the heap until we find a non-stale entry, or return `None`.
    pub fn find_best_pair(&mut self) -> Option<(u64, (SymId, SymId))> {
        loop {
            let Reverse(entry) = self.heap.pop()?;
            let pair = (entry.sym_a, entry.sym_b);
            let stored_freq = -entry.neg_freq as u64;
            // Lazy deletion: skip stale entries.
            let current_freq = self.pair_freq.get(&pair).copied().unwrap_or(0);
            if current_freq == 0 {
                continue;
            }
            if current_freq != stored_freq {
                // Stale — push a fresh entry and continue.
                self.heap.push(Reverse(HeapEntry {
                    neg_freq: -(current_freq as i64),
                    sym_a: pair.0,
                    sym_b: pair.1,
                }));
                continue;
            }
            return Some((stored_freq, pair));
        }
    }

    /// Apply merge `(sym_a, sym_b) → merged_str` to all word entries.
    ///
    /// Updates `pair_freq`, `pair_words`, and `sym_table` in place.
    /// Returns the ID of the new merged symbol.
    pub fn apply_merge(&mut self, pair: (SymId, SymId), merged_str: &str) -> Result<SymId> {
        let (sym_a, sym_b) = pair;

        // Intern the merged symbol.
        let merged_id = if let Some(&id) = self.sym_intern.get(merged_str) {
            id
        } else {
            let id = self.sym_table.len() as SymId;
            self.sym_intern.insert(merged_str.to_string(), id);
            self.sym_table.push(merged_str.to_string());
            id
        };

        // Collect affected word indices before mutably borrowing word_entries.
        let affected_words: Vec<usize> = self
            .pair_words
            .get(&pair)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();

        for word_idx in affected_words {
            let entry = &mut self.word_entries[word_idx];
            let count = entry.count;
            let tokens = &mut entry.tokens;

            // Scan left-to-right to find all occurrences of (sym_a, sym_b).
            // We collect merge positions first, then apply them in reverse order
            // to keep indices valid.
            let mut merge_positions: Vec<usize> = Vec::new();
            {
                let mut i = 0;
                while i + 1 < tokens.len() {
                    if tokens[i] == sym_a && tokens[i + 1] == sym_b {
                        merge_positions.push(i);
                        i += 2; // skip the second token to avoid overlapping merges
                    } else {
                        i += 1;
                    }
                }
            }

            if merge_positions.is_empty() {
                continue;
            }

            // For each merge position, update neighbour pair frequencies
            // before we mutate `tokens`.
            for &pos in &merge_positions {
                // Left neighbour: (tokens[pos-1], sym_a) → (tokens[pos-1], merged_id)
                if pos > 0 {
                    let left_sym = tokens[pos - 1];
                    // Skip if the left neighbour is itself about to be merged
                    // (i.e., it's sym_a in an immediately preceding merge).
                    // We handle this correctly because merge_positions are
                    // non-overlapping (skip i += 2 above).
                    let old_left_pair = (left_sym, sym_a);
                    let new_left_pair = (left_sym, merged_id);
                    Self::transfer_pair_count(
                        &mut self.pair_freq,
                        &mut self.pair_words,
                        &mut self.heap,
                        old_left_pair,
                        new_left_pair,
                        count,
                        word_idx,
                    );
                }
                // Right neighbour: (sym_b, tokens[pos+2]) → (merged_id, tokens[pos+2])
                if pos + 2 < tokens.len() {
                    let right_sym = tokens[pos + 2];
                    let old_right_pair = (sym_b, right_sym);
                    let new_right_pair = (merged_id, right_sym);
                    Self::transfer_pair_count(
                        &mut self.pair_freq,
                        &mut self.pair_words,
                        &mut self.heap,
                        old_right_pair,
                        new_right_pair,
                        count,
                        word_idx,
                    );
                }
            }

            // Now apply merges in reverse order (so indices stay valid).
            for &pos in merge_positions.iter().rev() {
                tokens[pos] = merged_id;
                tokens.remove(pos + 1);
            }

            // Update pair_words for the new merged pairs that now appear in this word.
            for &pos in &merge_positions {
                // After removal the absolute positions have shifted, but we
                // only care about updating pair_words membership here — the
                // actual counts were already updated above.
                // Insert this word into the pair_words sets for new pairs.
                // (left_sym, merged_id) and (merged_id, right_sym) were already
                // updated via transfer_pair_count above.
                let _ = pos; // positions already handled
            }
        }

        // Remove the merged pair from frequency tables.
        self.pair_freq.remove(&pair);
        self.pair_words.remove(&pair);

        Ok(merged_id)
    }

    /// Helper: subtract `count` from `old_pair` and add it to `new_pair`,
    /// updating `pair_words` membership and pushing fresh heap entries.
    #[allow(clippy::too_many_arguments)]
    fn transfer_pair_count(
        pair_freq: &mut HashMap<(SymId, SymId), u64>,
        pair_words: &mut HashMap<(SymId, SymId), HashSet<usize>>,
        heap: &mut BinaryHeap<Reverse<HeapEntry>>,
        old_pair: (SymId, SymId),
        new_pair: (SymId, SymId),
        count: u64,
        word_idx: usize,
    ) {
        // Subtract from old pair.
        if let Some(freq) = pair_freq.get_mut(&old_pair) {
            *freq = freq.saturating_sub(count);
            if *freq == 0 {
                pair_freq.remove(&old_pair);
                pair_words.remove(&old_pair);
            } else {
                if let Some(words) = pair_words.get_mut(&old_pair) {
                    words.remove(&word_idx);
                }
            }
        }

        // Add to new pair.
        if old_pair != new_pair {
            let new_freq = {
                let entry = pair_freq.entry(new_pair).or_insert(0);
                *entry += count;
                *entry
            };
            pair_words.entry(new_pair).or_default().insert(word_idx);
            // Push fresh heap entry (old stale entry will be skipped lazily).
            heap.push(Reverse(HeapEntry {
                neg_freq: -(new_freq as i64),
                sym_a: new_pair.0,
                sym_b: new_pair.1,
            }));
        }
    }
}

// ── Corpus preprocessing ──────────────────────────────────────────────────────

/// Convert an iterator of corpus lines into a word-type frequency map.
///
/// Each morpheme surface is exploded into character-level tokens:
/// `"東京"` → `["東", "京</w>"]`
///
/// Identical character sequences are collapsed to reduce memory.
fn build_word_freqs(lines: impl Iterator<Item = String>) -> HashMap<Vec<String>, u64> {
    let mut word_freqs: HashMap<Vec<String>, u64> = HashMap::new();
    for line in lines {
        for morpheme in line.split_whitespace() {
            if morpheme.is_empty() {
                continue;
            }
            let key = morpheme_to_char_tokens(morpheme);
            *word_freqs.entry(key).or_insert(0) += 1;
        }
    }
    word_freqs
}

/// Convert a single morpheme surface into a character-level token sequence
/// with a `</w>` end-of-word suffix on the last character token.
///
/// Single-character morphemes become `["char</w>"]`.
fn morpheme_to_char_tokens(surface: &str) -> Vec<String> {
    let chars: Vec<char> = surface.chars().collect();
    let n = chars.len();
    chars
        .into_iter()
        .enumerate()
        .map(|(i, ch)| {
            if i + 1 == n {
                format!("{ch}</w>")
            } else {
                ch.to_string()
            }
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_corpus() -> Vec<String> {
        vec![
            "東京 は 東京 東京 東京 東京".to_string(),
            "東京 東京 日本 日本 日本".to_string(),
            "東京 は 日本 の 首都 です".to_string(),
            "日本 日本 東京 東京 東京 東京 東京".to_string(),
            "東京 東京 東京 東京 東京 東京".to_string(),
        ]
    }

    #[test]
    fn test_bpe_trainer_small_corpus() {
        let corpus = small_corpus();
        let trainer = BpeTrainer::new(200).with_min_frequency(2);
        let vocab = trainer
            .train(corpus.into_iter())
            .expect("train must succeed");

        assert!(
            !vocab.merges.is_empty(),
            "at least one merge should be performed on a repetitive corpus"
        );
        assert!(
            !vocab.vocab.is_empty(),
            "vocab map must be non-empty after training"
        );
        // Special tokens must be present.
        assert!(vocab.vocab.contains_key("[UNK]"), "[UNK] must be in vocab");
    }

    #[test]
    fn test_bpe_encode_decode() {
        // Corpus has many repetitions of "東京" so the chars should merge.
        let corpus = vec![
            "東京 東京 東京 東京 東京 東京 東京 東京 東京 東京".to_string(),
            "東京 東京 東京 東京 東京 東京 東京 東京".to_string(),
            "東京 東京 東京 東京 東京 東京".to_string(),
        ];
        let trainer = BpeTrainer::new(50).with_min_frequency(2);
        let vocab = trainer
            .train(corpus.into_iter())
            .expect("train must succeed");

        // Encoding the string should not panic and should return non-empty result.
        let tokens = vocab.encode("東京");
        assert!(
            !tokens.is_empty(),
            "encoding non-empty text must produce at least one token"
        );

        // If a merge of 東+京</w> happened, we should see the merged token.
        if !vocab.merges.is_empty() {
            // At minimum the token sequence is shorter or equal to the raw chars.
            let raw_char_count = "東京".chars().count() + 1; // +1 for </w>
            assert!(
                tokens.len() <= raw_char_count,
                "merged encoding must not be longer than unmerged char sequence"
            );
        }
    }

    #[test]
    fn test_bpe_save_load_json() {
        let corpus = small_corpus();
        let trainer = BpeTrainer::new(100).with_min_frequency(2);
        let vocab = trainer
            .train(corpus.into_iter())
            .expect("train must succeed");
        let merge_count = vocab.merges.len();

        let path = std::env::temp_dir().join("bpe_test.json");
        vocab.save_json(&path).expect("save_json must succeed");

        let loaded = BpeVocab::load_json(&path).expect("load_json must succeed");
        assert_eq!(
            loaded.merges.len(),
            merge_count,
            "loaded merge count must match saved merge count"
        );
        assert_eq!(
            loaded.vocab.len(),
            vocab.vocab.len(),
            "loaded vocab size must match"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_bpe_min_frequency() {
        // Corpus is tiny — no pair should reach frequency 100.
        let corpus = vec![
            "東京 大阪".to_string(),
            "京都 神戸".to_string(),
            "東京 大阪".to_string(),
        ];
        let trainer = BpeTrainer::new(500).with_min_frequency(100);
        let vocab = trainer
            .train(corpus.into_iter())
            .expect("train must succeed");

        assert_eq!(
            vocab.merges.len(),
            0,
            "no merges should occur when min_frequency > all pair counts"
        );
    }
}
