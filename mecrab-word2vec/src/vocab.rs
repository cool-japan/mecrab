//! Vocabulary management for Word2Vec training

use crate::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Word frequency and metadata
#[derive(Debug, Clone)]
pub struct WordInfo {
    pub word_id: u32,
    pub remapped_id: u32, // Dense 0-based index for training
    pub count: u64,
    /// Subsampling probability (probability to keep the word)
    pub sample_prob: f32,
}

/// Vocabulary with frequency counts and subsampling
#[derive(Debug)]
pub struct Vocabulary {
    /// word_id → WordInfo
    words: HashMap<u32, WordInfo>,
    /// remapped_id → word_id (for saving output)
    remapped_to_word_id: Vec<u32>,
    /// Fast lookup: word_id → remapped_id (O(1) access for training)
    word_id_to_remapped: Vec<Option<u32>>,
    /// Total word count in corpus
    total_words: u64,
    /// Maximum word_id in vocabulary (for MCV1 format compatibility)
    max_word_id: u32,
    /// Minimum frequency threshold
    min_count: u64,
    /// Subsampling threshold (e.g., 1e-4)
    sample: f64,
}

impl Vocabulary {
    /// Create new vocabulary builder
    pub fn new(min_count: u64, sample: f64) -> Self {
        Self {
            words: HashMap::new(),
            remapped_to_word_id: Vec::new(),
            word_id_to_remapped: Vec::new(),
            total_words: 0,
            max_word_id: 0,
            min_count,
            sample,
        }
    }

    /// Build vocabulary from corpus file (space-separated word_ids per line)
    pub fn build_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        // First pass: count frequencies
        let mut counts: HashMap<u32, u64> = HashMap::new();
        let mut total = 0u64;

        for line in reader.lines() {
            let line = line?;
            for token in line.split_whitespace() {
                if let Ok(word_id) = token.parse::<u32>() {
                    *counts.entry(word_id).or_insert(0) += 1;
                    total += 1;
                }
            }
        }

        self.total_words = total;

        let unique_words = counts.len();

        // Filter by min_count and compute subsampling probabilities
        // First collect into a Vec for sorting
        let mut word_list: Vec<(u32, u64)> = counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_count)
            .collect();

        // Sort by frequency (descending) for better cache locality
        word_list.sort_by(|a, b| b.1.cmp(&a.1));

        // Assign remapped_ids 0, 1, 2, ... (dense indexing)
        for (remapped_id, (word_id, count)) in word_list.into_iter().enumerate() {
            let freq = count as f64 / total as f64;
            let sample_prob = if self.sample > 0.0 {
                // Mikolov's subsampling formula
                ((self.sample / freq).sqrt() + (self.sample / freq)).min(1.0) as f32
            } else {
                1.0
            };

            self.words.insert(
                word_id,
                WordInfo {
                    word_id,
                    remapped_id: remapped_id as u32,
                    count,
                    sample_prob,
                },
            );

            // Build reverse mapping
            self.remapped_to_word_id.push(word_id);

            // Track maximum word_id (for MCV1 format)
            if word_id > self.max_word_id {
                self.max_word_id = word_id;
            }
        }

        // Build fast lookup table: word_id → remapped_id (O(1) access)
        self.word_id_to_remapped = vec![None; (self.max_word_id + 1) as usize];
        for info in self.words.values() {
            self.word_id_to_remapped[info.word_id as usize] = Some(info.remapped_id);
        }

        eprintln!("Vocabulary built:");
        eprintln!("  Total words: {}", self.total_words);
        eprintln!("  Unique words (before filtering): {}", unique_words);
        eprintln!(
            "  Vocab size (after min_count={}): {}",
            self.min_count,
            self.words.len()
        );
        eprintln!(
            "  Remapped IDs: 0-{} (dense indexing)",
            self.words.len() - 1
        );
        eprintln!(
            "  Fast lookup table: {} entries",
            self.word_id_to_remapped.len()
        );

        Ok(())
    }

    /// Check if word_id is in vocabulary
    pub fn contains(&self, word_id: u32) -> bool {
        self.words.contains_key(&word_id)
    }

    /// Get word info
    pub fn get(&self, word_id: u32) -> Option<&WordInfo> {
        self.words.get(&word_id)
    }

    /// Get vocabulary size
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Check if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Get total word count
    pub fn total_words(&self) -> u64 {
        self.total_words
    }

    /// Get maximum word_id in vocabulary
    pub fn max_word_id(&self) -> u32 {
        self.max_word_id
    }

    /// Iterate over all words
    pub fn iter(&self) -> impl Iterator<Item = &WordInfo> {
        self.words.values()
    }

    /// Get word_id from remapped_id
    pub fn get_word_id(&self, remapped_id: u32) -> Option<u32> {
        self.remapped_to_word_id.get(remapped_id as usize).copied()
    }

    /// Fast lookup: word_id → remapped_id (O(1), for training hot path)
    #[inline]
    pub fn get_remapped_id(&self, word_id: u32) -> Option<u32> {
        self.word_id_to_remapped
            .get(word_id as usize)
            .and_then(|&opt| opt)
    }
}
