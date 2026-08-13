//! Lattice module for building the word graph
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! The lattice represents all possible segmentations of the input text
//! as a directed acyclic graph (DAG). Each node represents a potential
//! word, and edges connect adjacent words.

mod visualize;

pub use visualize::{DotBuilder, DotConfig, NodeShape, RankDir};

use crate::Result;
use crate::dict::{CharCategory, Dictionary, DictionaryEntry};
use std::sync::Arc;

/// MeCab's `max-grouping-size` default — the longest same-category run that is
/// still collapsed into a single unknown-word node.
///
/// Mirrors `DEFAULT_MAX_GROUPING_SIZE` in mecab-0.996 `src/tokenizer.cpp`.
/// MeCab measures the run from the *second* character (its `seekToOtherType`
/// call starts one character in), so the longest run that still yields one
/// grouped node is `MAX_GROUPING_SIZE + 1` = 25 characters. Verified against
/// mecab 0.996 + IPADIC: a run of 25 `a` characters is one token, 26 is two
/// (`a` + 25 `a`s), 27 is three; 26 katakana characters come out as a 2
/// character prefix node plus a 24 character group.
///
/// IPADIC's `dicrc` does not override `max-grouping-size`, so this default is
/// what the shipped dictionaries actually use.
const MAX_GROUPING_SIZE: usize = 24;

// ── Constraint types ─────────────────────────────────────────────────────────

/// Describes a byte span `[start, end)` that must form exactly one token.
#[derive(Debug, Clone)]
pub struct ForcedSpan {
    /// Inclusive start byte offset in the input text.
    pub start: usize,
    /// Exclusive end byte offset in the input text.
    pub end: usize,
    /// Optional IPADIC feature string to assign to the synthetic node.
    ///
    /// When `None`, an unknown-word feature (`"未知語,強制"`) is used unless
    /// a dictionary match is found (in which case the dictionary feature is used).
    pub feature: Option<String>,
}

/// A set of constraints that force specific byte spans to be treated as single tokens.
///
/// Construct with [`ParseConstraints::new`] and populate via [`add_span`](ParseConstraints::add_span).
#[derive(Debug, Clone, Default)]
pub struct ParseConstraints {
    spans: Vec<ForcedSpan>,
}

impl ParseConstraints {
    /// Create an empty constraint set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a forced span: `text[start..end]` must be exactly one token.
    ///
    /// `feature` is an optional IPADIC feature string; `None` means auto-detect.
    pub fn add_span(&mut self, start: usize, end: usize, feature: Option<String>) -> &mut Self {
        self.spans.push(ForcedSpan {
            start,
            end,
            feature,
        });
        self
    }

    /// Returns `true` when no constraints have been added.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Returns a slice of all registered forced spans.
    pub fn spans(&self) -> &[ForcedSpan] {
        &self.spans
    }
}

/// A node in the lattice representing a potential word/token
#[derive(Debug, Clone)]
pub struct LatticeNode<'a> {
    /// Surface form (slice into original text)
    pub surface: &'a str,
    /// Start position in bytes
    pub start: usize,
    /// End position in bytes
    pub end: usize,
    /// Word ID (token index in dictionary, used for embeddings)
    pub word_id: u32,
    /// Left context ID for connection matrix
    pub left_id: u16,
    /// Right context ID for connection matrix
    pub right_id: u16,
    /// Part-of-speech ID
    pub pos_id: u16,
    /// Word cost from dictionary
    pub wcost: i16,
    /// Feature string (shared via Arc to avoid redundant allocations)
    pub feature: Arc<str>,
    /// Whether this is an unknown word
    pub is_unknown: bool,
}

impl<'a> LatticeNode<'a> {
    /// Create a new lattice node from a dictionary entry (borrowing feature)
    pub fn from_entry(
        text: &'a str,
        start: usize,
        entry: &DictionaryEntry,
        feature: Arc<str>,
    ) -> Self {
        Self {
            surface: &text[start..start + entry.length],
            start,
            end: start + entry.length,
            word_id: entry.word_id,
            left_id: entry.left_id,
            right_id: entry.right_id,
            pos_id: entry.pos_id,
            wcost: entry.wcost,
            feature,
            is_unknown: false,
        }
    }

    /// Create a new lattice node from an owned dictionary entry (moves feature)
    #[inline]
    pub fn from_entry_owned(text: &'a str, start: usize, entry: DictionaryEntry) -> Self {
        let end = start + entry.length;
        Self {
            surface: &text[start..end],
            start,
            end,
            word_id: entry.word_id,
            left_id: entry.left_id,
            right_id: entry.right_id,
            pos_id: entry.pos_id,
            wcost: entry.wcost,
            feature: entry.feature, // Move, don't clone
            is_unknown: false,
        }
    }

    /// Create a BOS (Beginning of Sentence) node
    pub fn bos() -> Self {
        Self {
            surface: "",
            start: 0,
            end: 0,
            word_id: u32::MAX, // BOS/EOS don't have word_id
            left_id: 0,
            right_id: 0,
            pos_id: 0,
            wcost: 0,
            feature: Arc::from("BOS/EOS"),
            is_unknown: false,
        }
    }

    /// Create an EOS (End of Sentence) node
    pub fn eos(position: usize) -> Self {
        Self {
            surface: "",
            start: position,
            end: position,
            word_id: u32::MAX, // BOS/EOS don't have word_id
            left_id: 0,
            right_id: 0,
            pos_id: 0,
            wcost: 0,
            feature: Arc::from("BOS/EOS"),
            is_unknown: false,
        }
    }

    /// Create an unknown word node
    pub fn unknown(
        text: &'a str,
        start: usize,
        length: usize,
        entry: &DictionaryEntry,
        feature: Arc<str>,
    ) -> Self {
        Self {
            surface: &text[start..start + length],
            start,
            end: start + length,
            word_id: entry.word_id,
            left_id: entry.left_id,
            right_id: entry.right_id,
            pos_id: entry.pos_id,
            wcost: entry.wcost,
            feature,
            is_unknown: true,
        }
    }
}

/// The lattice structure representing all possible segmentations
#[derive(Debug)]
pub struct Lattice<'a> {
    /// Original input text
    pub text: &'a str,
    /// Flattened CSR storage of all lattice nodes, contiguous in memory.
    ///
    /// `flat[offsets[pos]..offsets[pos + 1]]` is the slice of nodes ending at
    /// byte position `pos`. This replaces the former `Vec<Vec<LatticeNode>>`,
    /// cutting per-lattice heap allocations from O(text_len) tiny vectors down
    /// to two contiguous buffers and giving the Viterbi read phase cache-friendly
    /// sequential access. Access positions via [`nodes_ending_at`](Self::nodes_ending_at)
    /// or [`positions`](Self::positions).
    flat: Vec<LatticeNode<'a>>,
    /// CSR row offsets into `flat`.
    ///
    /// Invariant: `offsets.len() == num_positions + 1`, `offsets[0] == 0`,
    /// `offsets` is monotonically non-decreasing, and `offsets[last] == flat.len()`.
    offsets: Vec<usize>,
}

impl<'a> Lattice<'a> {
    /// Build a lattice from the input text using the dictionary
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to analyze
    /// * `dict` - The dictionary to use for word lookup
    ///
    /// # Errors
    ///
    /// Returns an error if lattice construction fails.
    pub fn build(text: &'a str, dict: &Dictionary) -> Result<Self> {
        Ok(Self::from_nodes_at(text, Self::build_nodes_at(text, dict)))
    }

    /// Build the position-bucketed node lists (pre-freeze intermediate form).
    ///
    /// `result[pos]` holds the nodes ending at byte position `pos`; index 0 is
    /// the BOS slot and index `text_len + 1` is the EOS slot. This is the
    /// mutable representation used during construction and constraint
    /// application before being frozen into the CSR layout via
    /// [`from_nodes_at`](Self::from_nodes_at).
    fn build_nodes_at(text: &'a str, dict: &Dictionary) -> Vec<Vec<LatticeNode<'a>>> {
        let text_len = text.len();

        // Initialize nodes_at with one extra slot for BOS at position 0
        // and one for EOS at position text_len + 1
        let mut nodes_at: Vec<Vec<LatticeNode<'a>>> = vec![Vec::new(); text_len + 2];

        // Add BOS node at position 0
        nodes_at[0].push(LatticeNode::bos());

        #[cfg(feature = "parallel")]
        Self::build_parallel(text, text_len, dict, &mut nodes_at);

        #[cfg(not(feature = "parallel"))]
        Self::build_sequential(text, text_len, dict, &mut nodes_at);

        // Add EOS node at the final position. (Trailing unknown characters are
        // handled inside the builders; reaching the end is the caller's concern.)
        nodes_at[text_len + 1].push(LatticeNode::eos(text_len));

        nodes_at
    }

    /// Freeze position-bucketed node lists into the contiguous CSR representation.
    ///
    /// Preserves node order within every position exactly, so the frozen lattice
    /// yields byte-identical [`nodes_ending_at`](Self::nodes_ending_at) slices to
    /// the input buckets — the Viterbi/forward-backward results are unchanged.
    pub fn from_nodes_at(text: &'a str, nodes_at: Vec<Vec<LatticeNode<'a>>>) -> Self {
        let total: usize = nodes_at.iter().map(Vec::len).sum();
        let mut flat = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(nodes_at.len() + 1);
        offsets.push(0usize);
        for bucket in nodes_at {
            flat.extend(bucket);
            offsets.push(flat.len());
        }
        Self {
            text,
            flat,
            offsets,
        }
    }

    /// Sequential lattice builder — always compiled so the parallel test can use it as a
    /// reference implementation; only called from `build()` when the `parallel` feature is off.
    #[cfg_attr(feature = "parallel", allow(dead_code))]
    pub fn build_sequential(
        text: &'a str,
        text_len: usize,
        dict: &Dictionary,
        nodes_at: &mut Vec<Vec<LatticeNode<'a>>>,
    ) {
        for (pos, c) in text.char_indices() {
            let remaining = &text[pos..];
            let entries = dict.lookup(remaining);

            if entries.is_empty() {
                Self::add_unknown_nodes(text, pos, c, dict, nodes_at);
            } else {
                for entry in entries {
                    let node = LatticeNode::from_entry_owned(text, pos, entry);
                    let end_pos = node.end;
                    if end_pos <= text_len {
                        nodes_at[end_pos + 1].push(node);
                    }
                }
            }
        }
    }

    /// Parallel lattice builder — dictionary lookups run via rayon, unknown-word
    /// handling remains sequential.
    ///
    /// Enabled only when the `parallel` feature is active.
    #[cfg(feature = "parallel")]
    pub fn build_parallel(
        text: &'a str,
        text_len: usize,
        dict: &Dictionary,
        nodes_at: &mut Vec<Vec<LatticeNode<'a>>>,
    ) {
        use rayon::prelude::*;

        // Collect all (byte_pos, char, remaining_slice) for every character position.
        // The slices borrow from `text`, which is `'a` — valid for the whole function.
        let positions: Vec<(usize, char, &'a str)> = text
            .char_indices()
            .map(|(pos, c)| (pos, c, &text[pos..]))
            .collect();

        // --- Parallel dictionary lookups -------------------------------------------
        // Each thread calls dict.lookup() independently.  Dictionary is Sync (it
        // contains only Arc<Mmap> / immutable slices), so a shared reference is safe.
        //
        // We cannot push directly into `nodes_at` (requires &mut, non-Send) from
        // rayon threads, so we collect results into a Vec and merge sequentially below.
        //
        // The result type is `Vec<(usize /*pos*/, char, Vec<DictionaryEntry>)>`.
        // Using rayon::scope is not necessary here because `'a` is already a valid
        // shared borrow lifetime that outlives the par_iter closure — rayon only
        // requires closure arguments to be `Send`, and `&'a str` is `Send`.
        let lookup_results: Vec<(usize, char, Vec<DictionaryEntry>)> = positions
            .par_iter()
            .map(|&(pos, c, remaining)| {
                let entries = dict.lookup(remaining);
                (pos, c, entries)
            })
            .collect();

        // --- Sequential merge + unknown-word handling ------------------------------
        // Positions that had zero dictionary hits need unknown-word nodes.
        for (pos, c, entries) in lookup_results {
            if entries.is_empty() {
                Self::add_unknown_nodes(text, pos, c, dict, nodes_at);
            } else {
                for entry in entries {
                    let node = LatticeNode::from_entry_owned(text, pos, entry);
                    let end_pos = node.end;
                    if end_pos <= text_len {
                        nodes_at[end_pos + 1].push(node);
                    }
                }
            }
        }
    }

    /// Add unknown word nodes for a character
    fn add_unknown_nodes(
        text: &'a str,
        pos: usize,
        c: char,
        dict: &Dictionary,
        nodes_at: &mut [Vec<LatticeNode<'a>>],
    ) {
        let category = dict.char_category(c);

        // Get unknown entries for this category
        let entries = dict.unknown.generate_entries(category, c.len_utf8());

        if entries.is_empty() {
            // Create a default unknown entry if none exist
            let char_len = c.len_utf8();
            let end_pos = pos + char_len;

            if end_pos <= text.len() {
                nodes_at[end_pos + 1].push(LatticeNode {
                    surface: &text[pos..end_pos],
                    start: pos,
                    end: end_pos,
                    word_id: u32::MAX, // Unknown words don't have dictionary word_id
                    left_id: 0,
                    right_id: 0,
                    pos_id: 0,
                    wcost: 10000, // High cost for unknown
                    feature: Arc::from(format!("未知語,{category:?}").as_str()),
                    is_unknown: true,
                });
            }
        } else {
            for entry in &entries {
                let feature = entry.feature.clone();

                let node = LatticeNode::unknown(text, pos, entry.length, entry, feature);
                let end_pos = node.end;

                if end_pos <= text.len() {
                    nodes_at[end_pos + 1].push(node);
                }
            }
        }

        // Multi-character unknown nodes for the same-category run starting here.
        Self::add_unknown_run_nodes(text, pos, c, category, dict, nodes_at);
    }

    /// Add the multi-character unknown-word nodes for the same-category run that
    /// starts at `start`.
    ///
    /// This implements MeCab's `char.def` semantics (mecab-0.996
    /// `src/tokenizer.cpp::Tokenizer::lookup`), driven by the two per-character
    /// flags that the packed `char.bin` carries:
    ///
    /// * `GROUP = 1` → **one** node spanning the whole maximal same-category
    ///   run, emitted only while that run is short enough to group
    ///   ([`MAX_GROUPING_SIZE`] + 1 characters).
    /// * `LENGTH = n` → prefix nodes of 1..=n characters and nothing longer.
    ///   The single-character node is always emitted by
    ///   [`add_unknown_nodes`](Self::add_unknown_nodes), so this function emits
    ///   2..=n.
    ///
    /// Both bounds are small constants, so a run of `n` characters contributes
    /// O(1) nodes per position — O(n) for the run — and the scan never walks
    /// further than the semantics can use. The previous implementation emitted a
    /// node for *every* prefix of length ≥ 2 at *every* position (Θ(n²) nodes),
    /// found through a linear duplicate scan of the destination bucket (Θ(n³)
    /// work), and never consulted `LENGTH` at all: a 3,200 character ASCII run
    /// took 38.4 s and a 60,000 character one could not finish.
    ///
    /// The `GROUP` decision is taken per category via
    /// [`CharDef::should_group`](crate::dict::CharDef::should_group) rather than
    /// from the first character's own `CharInfo`. MeCab's `char.def` assigns the
    /// flags per category, so the two agree for every category a dictionary
    /// defines — but a packed `char.bin` that leaves unassigned code points
    /// zeroed (rather than filling them with the mandatory `DEFAULT` category,
    /// which IPADIC declares as `GROUP = 1`) would otherwise stop grouping
    /// DEFAULT runs such as astral-plane emoji.
    fn add_unknown_run_nodes(
        text: &'a str,
        start: usize,
        first: char,
        category: CharCategory,
        dict: &Dictionary,
        nodes_at: &mut [Vec<LatticeNode<'a>>],
    ) {
        // `LENGTH` from char.def — a 4 bit field, so at most 15.
        let max_prefix_chars = dict.char_info(first).length() as usize;
        let group = dict.char_def.should_group(category);

        if !group && max_prefix_chars < 2 {
            // Nothing beyond the single-character node the caller already added.
            return;
        }

        // Walk the run, but never further than the semantics can use: `LENGTH`
        // characters for the prefix nodes, and one character past the longest
        // groupable run to establish that a run is too long to group.
        let scan_limit = max_prefix_chars.max(if group { MAX_GROUPING_SIZE + 2 } else { 0 });

        let mut run_bytes = 0usize;
        let mut run_chars = 0usize;
        // Cleared when the scan stops early — i.e. the run is longer than
        // anything that could still be grouped.
        let mut run_complete = true;

        for c in text[start..].chars() {
            if dict.char_category(c) != category {
                break;
            }
            if run_chars == scan_limit {
                run_complete = false;
                break;
            }
            run_bytes += c.len_utf8();
            run_chars += 1;

            if run_chars >= 2 && run_chars <= max_prefix_chars {
                Self::push_unknown_nodes(text, start, run_bytes, category, dict, nodes_at);
            }
        }

        // The grouped node. Runs of at most `max_prefix_chars` characters were
        // already emitted by the prefix loop above; MeCab skips that duplicate
        // the same way (`begin3 == group_begin3` → `continue`).
        if group
            && run_complete
            && run_chars >= 2
            && run_chars > max_prefix_chars
            && run_chars <= MAX_GROUPING_SIZE + 1
        {
            Self::push_unknown_nodes(text, start, run_bytes, category, dict, nodes_at);
        }
    }

    /// Push one unknown-word node per `unk.dic` template for the span
    /// `text[start..start + length]`.
    ///
    /// Every template of the category is emitted, as MeCab's `ADDUNKNWON` macro
    /// does: the templates differ in context id and cost (IPADIC gives KATAKANA
    /// six, from 名詞,一般 to 感動詞), and Viterbi picks between them.
    fn push_unknown_nodes(
        text: &'a str,
        start: usize,
        length: usize,
        category: CharCategory,
        dict: &Dictionary,
        nodes_at: &mut [Vec<LatticeNode<'a>>],
    ) {
        let end_pos = start + length;
        if end_pos > text.len() {
            return;
        }

        for entry in &dict.unknown.generate_entries(category, length) {
            let feature = entry.feature.clone();
            nodes_at[end_pos + 1].push(LatticeNode::unknown(text, start, length, entry, feature));
        }
    }

    /// Get the number of byte positions in the lattice
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Check if the lattice is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get nodes ending at a specific position
    pub fn nodes_ending_at(&self, pos: usize) -> &[LatticeNode<'a>] {
        if pos + 1 < self.offsets.len() {
            &self.flat[self.offsets[pos]..self.offsets[pos + 1]]
        } else {
            &[]
        }
    }

    /// Iterate over every byte position's node slice in order (CSR rows).
    ///
    /// Yields the same slices as calling [`nodes_ending_at`](Self::nodes_ending_at)
    /// for `pos` in `0..len()`, in order. Replaces the former
    /// `lattice.nodes_at.iter()` over the bucketed representation.
    pub fn positions(&self) -> impl Iterator<Item = &[LatticeNode<'a>]> {
        self.offsets.windows(2).map(|w| &self.flat[w[0]..w[1]])
    }

    /// Build a constrained lattice: `text[span.start..span.end]` becomes exactly one token
    /// for every `ForcedSpan` in `constraints`.
    ///
    /// When `constraints.is_empty()` this is byte-identical to `build(text, dict)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `build` call fails.
    pub fn build_with_constraints(
        text: &'a str,
        dict: &Dictionary,
        constraints: &ParseConstraints,
    ) -> Result<Self> {
        let mut nodes_at = Self::build_nodes_at(text, dict);

        if constraints.is_empty() {
            return Ok(Self::from_nodes_at(text, nodes_at));
        }

        let text_len = text.len();

        // Step 1: remove nodes that partially overlap any forced span.
        // Indices `1..=text_len` cover every real position (skip the BOS slot 0
        // and the EOS slot `text_len + 1`).
        for nodes in &mut nodes_at[1..=text_len] {
            nodes.retain(|node| {
                // Keep BOS/EOS sentinels (they have equal start and end).
                if node.start == node.end {
                    return true;
                }
                // Keep if the node does NOT partially overlap any forced span.
                !constraints.spans().iter().any(|span| {
                    let overlaps = node.start < span.end && node.end > span.start;
                    let exact = node.start == span.start && node.end == span.end;
                    overlaps && !exact
                })
            });
        }

        // Step 2: inject synthetic nodes for forced spans that have no exact match.
        for span in constraints.spans() {
            let s = span.start;
            let e = span.end;

            // Validate byte offsets.
            if e > text_len || s >= e || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
                continue;
            }

            let slot = e + 1; // nodes_at index for nodes ending at byte `e`
            if slot >= nodes_at.len() {
                continue;
            }

            // Check if an exact node already exists.
            let already_present = nodes_at[slot].iter().any(|n| n.start == s && n.end == e);

            if already_present {
                // Nothing to inject; constraint is already satisfied.
                continue;
            }

            // Try a dictionary lookup for the exact span surface.
            let surface_slice = &text[s..e];
            let entries = dict.lookup(surface_slice);
            let exact_entry = entries.into_iter().find(|entry| entry.length == e - s);

            let node = if let Some(entry) = exact_entry {
                let feature: Arc<str> = match &span.feature {
                    Some(s) => Arc::from(s.as_str()),
                    None => Arc::clone(&entry.feature),
                };
                LatticeNode {
                    surface: &text[s..e],
                    start: s,
                    end: e,
                    word_id: entry.word_id,
                    left_id: entry.left_id,
                    right_id: entry.right_id,
                    pos_id: entry.pos_id,
                    wcost: entry.wcost,
                    feature,
                    is_unknown: false,
                }
            } else {
                // Fall back to a high-cost synthetic unknown node.
                let feature: Arc<str> = match &span.feature {
                    Some(s) => Arc::from(s.as_str()),
                    None => Arc::from("未知語,強制"),
                };
                LatticeNode {
                    surface: &text[s..e],
                    start: s,
                    end: e,
                    word_id: u32::MAX,
                    left_id: 0,
                    right_id: 0,
                    pos_id: 0,
                    wcost: 5000,
                    feature,
                    is_unknown: true,
                }
            };

            nodes_at[slot].push(node);
        }

        Ok(Self::from_nodes_at(text, nodes_at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bos_eos_nodes() {
        let bos = LatticeNode::bos();
        assert_eq!(bos.surface, "");
        assert_eq!(bos.start, 0);
        assert_eq!(bos.end, 0);

        let eos = LatticeNode::eos(10);
        assert_eq!(eos.surface, "");
        assert_eq!(eos.start, 10);
        assert_eq!(eos.end, 10);
    }

    /// Verify BOS and EOS fingerprinting for non-empty text.
    ///
    /// The parallel vs. sequential equivalence test lives in
    /// `mecrab-builder/tests/lattice_parallel.rs` where the synthetic dictionary
    /// is available without creating a duplicate-`mecrab`-crate diamond problem.
    #[test]
    fn test_bos_fingerprint() {
        let bos = LatticeNode::bos();
        assert_eq!(bos.word_id, u32::MAX);
        assert!(bos.feature.contains("BOS"));
    }
}
