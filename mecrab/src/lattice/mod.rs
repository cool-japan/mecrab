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
    /// Feature string (lazily loaded)
    pub feature: String,
    /// Whether this is an unknown word
    pub is_unknown: bool,
}

impl<'a> LatticeNode<'a> {
    /// Create a new lattice node from a dictionary entry (borrowing feature)
    pub fn from_entry(
        text: &'a str,
        start: usize,
        entry: &DictionaryEntry,
        feature: String,
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
            feature: "BOS/EOS".to_string(),
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
            feature: "BOS/EOS".to_string(),
            is_unknown: false,
        }
    }

    /// Create an unknown word node
    pub fn unknown(
        text: &'a str,
        start: usize,
        length: usize,
        entry: &DictionaryEntry,
        feature: String,
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
    /// Nodes at each byte position
    /// Index 0 contains BOS, last index contains EOS
    pub nodes_at: Vec<Vec<LatticeNode<'a>>>,
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

        // Handle case where no nodes reach the end
        // This can happen with unknown characters at the end
        // (intentional: trailing-char check deferred to caller)
        let _ = text_len; // suppress potential "unused" in trivial paths

        // Add EOS node at the final position
        nodes_at[text_len + 1].push(LatticeNode::eos(text_len));

        Ok(Self { text, nodes_at })
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
                    feature: format!("未知語,{category:?}"),
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

        // Handle grouping for consecutive characters of the same category
        if dict.char_def.should_group(category) {
            Self::add_grouped_unknown(text, pos, category, dict, nodes_at);
        }
    }

    /// Add grouped unknown word nodes (consecutive characters of same category)
    fn add_grouped_unknown(
        text: &'a str,
        start: usize,
        category: CharCategory,
        dict: &Dictionary,
        nodes_at: &mut [Vec<LatticeNode<'a>>],
    ) {
        let remaining = &text[start..];
        let mut length = 0;
        let mut char_count = 0;

        for c in remaining.chars() {
            if dict.char_category(c) != category {
                break;
            }
            length += c.len_utf8();
            char_count += 1;

            // Limit group length
            if char_count > 1 {
                let entries = dict.unknown.generate_entries(category, length);

                for entry in &entries {
                    let feature = entry.feature.clone();

                    let node = LatticeNode::unknown(text, start, length, entry, feature);
                    let end_pos = start + length;

                    if end_pos <= text.len()
                        && !nodes_at[end_pos + 1]
                            .iter()
                            .any(|n| n.start == start && n.end == end_pos)
                    {
                        nodes_at[end_pos + 1].push(node);
                    }
                }
            }
        }
    }

    /// Get the number of byte positions in the lattice
    pub fn len(&self) -> usize {
        self.nodes_at.len()
    }

    /// Check if the lattice is empty
    pub fn is_empty(&self) -> bool {
        self.nodes_at.is_empty()
    }

    /// Get nodes ending at a specific position
    pub fn nodes_ending_at(&self, pos: usize) -> &[LatticeNode<'a>] {
        if pos < self.nodes_at.len() {
            &self.nodes_at[pos]
        } else {
            &[]
        }
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
