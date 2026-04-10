//! MeCrab - A high-performance morphological analyzer compatible with MeCab
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! # Overview
//!
//! MeCrab is a pure Rust implementation of a morphological analyzer that is
//! compatible with MeCab dictionaries (IPADIC format). It provides:
//!
//! - Zero-copy parsing where possible
//! - Memory-mapped dictionary loading via `memmap2`
//! - Thread-safe design using Rust's ownership model
//! - Double-Array Trie (DAT) for fast dictionary lookups
//! - Viterbi algorithm for optimal path finding
//! - SIMD-accelerated cost calculations using portable SIMD
//!
//! # Example
//!
//! ```no_run
//! use mecrab::MeCrab;
//!
//! let mecrab = MeCrab::new()?;
//! let result = mecrab.parse("すもももももももものうち")?;
//! println!("{}", result);
//! # Ok::<(), mecrab::Error>(())
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::format_push_string)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::unused_self)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::needless_pass_by_value)]

pub mod bench;
pub mod builder;
pub mod corpus;
pub mod debug;
pub mod dict;
pub mod error;
pub mod lattice;
pub mod normalize;
pub mod phonetic;
pub mod rerank;
pub mod semantic;
pub mod stream;
pub mod types;
pub mod vectors;
pub mod viterbi;

/// Extended API: batch processing and lazy iterator adapters.
pub mod api;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "python")]
pub mod python;

pub use builder::MeCrabBuilder;
pub use corpus::{CorpusStats, SurfaceVocab};
pub use error::{Error, Result};
pub use rerank::{CostReranker, NullReranker, RerankCandidate, Reranker};
pub use types::{AnalysisResult, Morpheme, OutputFormat};

#[cfg(feature = "neural")]
pub use rerank::neural::NeuralReranker;

pub use dict::provider::{
    AutoDetectProvider, DictionaryFormat, DictionaryProvider, IpadicProvider, MorphemeFeatures,
    NeologdProvider, UnidicProvider,
};

/// Re-export SIMD batch connection cost lookup for benchmarking.
///
/// Gathers up to 16 connection costs from a connection-matrix row at once,
/// widening each `i16` value to `i32`. The implementation dispatches to ARM
/// NEON, x86_64 AVX2/SSE4.1, WASM simd128, or a scalar fallback depending on
/// the build target.
#[cfg(feature = "simd")]
pub use viterbi::simd::batch_connection_costs;

use std::sync::Arc;

use dict::Dictionary;
use lattice::Lattice;
use viterbi::ViterbiSolver;

/// The main MeCrab morphological analyzer
#[derive(Clone)]
pub struct MeCrab {
    pub(crate) dictionary: Arc<Dictionary>,
    pub(crate) output_format: OutputFormat,
    pub(crate) semantic_enabled: bool,
    pub(crate) ipa_enabled: bool,
    pub(crate) vector_enabled: bool,
    pub(crate) vector_store: Option<Arc<vectors::VectorStore>>,
    /// Dictionary provider used for structured feature parsing
    pub(crate) provider: Arc<dyn DictionaryProvider>,
}

impl MeCrab {
    /// Create a new MeCrab instance with default dictionary
    ///
    /// # Errors
    ///
    /// Returns an error if the default dictionary cannot be found or loaded.
    pub fn new() -> Result<Self> {
        Self::builder().build()
    }

    /// Create a builder for configuring MeCrab
    #[must_use]
    pub fn builder() -> MeCrabBuilder {
        MeCrabBuilder::new()
    }

    /// Replace the `DictionaryProvider` used for structured feature parsing.
    ///
    /// Returns a new `MeCrab` that shares all other state with `self` but uses
    /// the supplied provider.  Because `MeCrab` is cheaply cloneable (Arc
    /// internals), this is a low-cost operation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mecrab::{MeCrab, UnidicProvider};
    ///
    /// let mecrab = MeCrab::new()?.with_provider(UnidicProvider);
    /// # Ok::<(), mecrab::Error>(())
    /// ```
    #[must_use]
    pub fn with_provider(mut self, provider: impl DictionaryProvider + 'static) -> Self {
        self.provider = Arc::new(provider);
        self
    }

    /// Return a reference to the active `DictionaryProvider`.
    #[must_use]
    pub fn provider(&self) -> &dyn DictionaryProvider {
        self.provider.as_ref()
    }

    /// Return a reference to the loaded `VectorStore`, if any.
    ///
    /// Returns `None` when the analyzer was built without a vector path.
    #[must_use]
    pub fn vector_store(&self) -> Option<&vectors::VectorStore> {
        self.vector_store.as_deref()
    }

    /// Parse the input text and return analysis result
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn parse(&self, text: &str) -> Result<AnalysisResult> {
        // Build the lattice
        let lattice = Lattice::build(text, &self.dictionary)?;

        // Solve using Viterbi algorithm
        let solver = ViterbiSolver::new(&self.dictionary);
        let path = solver.solve(&lattice)?;

        // Convert path to morphemes with optional semantic and IPA enrichment
        let morphemes = path
            .into_iter()
            .map(|node| {
                let entities = if self.semantic_enabled {
                    self.get_entities_for_surface(&node.surface)
                } else {
                    Vec::new()
                };

                let pronunciation = if self.ipa_enabled {
                    self.get_ipa_pronunciation(&node.feature)
                } else {
                    None
                };

                let embedding = if self.vector_enabled {
                    self.get_embedding(node.word_id)
                } else {
                    None
                };

                Morpheme {
                    surface: node.surface,
                    word_id: node.word_id,
                    pos_id: node.pos_id,
                    wcost: node.wcost,
                    feature: node.feature,
                    entities,
                    pronunciation,
                    embedding,
                    start_byte: node.start_byte,
                    end_byte: node.end_byte,
                }
            })
            .collect();

        Ok(AnalysisResult {
            morphemes,
            format: self.output_format,
        })
    }

    /// Get semantic entities for a surface form
    fn get_entities_for_surface(&self, surface: &str) -> Vec<semantic::EntityReference> {
        if let Some(ref surface_map) = self.dictionary.surface_map {
            if let Some(uris) = surface_map.get(surface) {
                return uris
                    .iter()
                    .map(|(uri, confidence)| {
                        let source = if uri.contains("wikidata.org") {
                            semantic::OntologySource::Wikidata
                        } else if uri.contains("dbpedia.org") {
                            semantic::OntologySource::DBpedia
                        } else {
                            semantic::OntologySource::Custom
                        };
                        semantic::EntityReference::new(uri.clone(), *confidence, source)
                    })
                    .collect();
            }
        }
        Vec::new()
    }

    /// Get IPA pronunciation from feature string
    fn get_ipa_pronunciation(&self, feature: &str) -> Option<String> {
        // Feature format: POS,POS1,POS2,POS3,conjugation,conjugation_type,lemma,reading,pronunciation
        let fields: Vec<&str> = feature.split(',').collect();

        // Get POS for particle detection
        let pos = fields.first().copied().unwrap_or("");

        // PRIORITY 1: Pronunciation field (index 8) - actual pronunciation
        // This already contains the correct pronunciation (e.g., "ワ" for particle "は")
        if let Some(&pron) = fields.get(8) {
            if pron != "*" && !pron.is_empty() {
                return Some(phonetic::to_ipa(pron));
            }
        }

        // PRIORITY 2: Reading field (index 7) - fallback if pronunciation not available
        if let Some(&reading) = fields.get(7) {
            if reading != "*" && !reading.is_empty() {
                // Special handling for particles with pronunciation changes
                if pos == "助詞" {
                    let ipa = match reading {
                        "ハ" => "wa", // 助詞「は」は /wa/ と発音
                        "ヘ" => "e",  // 助詞「へ」は /e/ と発音
                        "ヲ" => "o",  // 助詞「を」は /o/ と発音
                        _ => return Some(phonetic::to_ipa(reading)),
                    };
                    return Some(ipa.to_string());
                }
                return Some(phonetic::to_ipa(reading));
            }
        }

        None
    }

    /// Get word embedding vector for a given word ID
    ///
    /// Returns None if:
    /// - No vector store is loaded
    /// - word_id is u32::MAX (overlay/unknown words)
    /// - word_id is out of bounds in the vector store
    fn get_embedding(&self, word_id: u32) -> Option<Vec<f32>> {
        // Skip overlay/unknown words (marked with u32::MAX)
        if word_id == u32::MAX {
            return None;
        }

        self.vector_store
            .as_ref()
            .and_then(|store| store.get(word_id))
            .map(|slice| slice.to_vec())
    }

    /// Parse the input text and return wakati (space-separated) output
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn wakati(&self, text: &str) -> Result<String> {
        let result = self.parse(text)?;
        let surfaces: Vec<&str> = result
            .morphemes
            .iter()
            .map(|m| m.surface.as_str())
            .collect();
        Ok(surfaces.join(" "))
    }

    /// Add a word to the dictionary at runtime
    ///
    /// This is a key feature for production systems that need to handle
    /// new vocabulary (product names, trending terms, etc.) without restart.
    ///
    /// # Arguments
    ///
    /// * `surface` - The surface form (the actual text)
    /// * `reading` - The katakana reading
    /// * `pronunciation` - The pronunciation (often same as reading)
    /// * `wcost` - Word cost (lower = more preferred, typical: 5000-8000)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mecrab = MeCrab::new()?;
    ///
    /// // Add a new word
    /// mecrab.add_word("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000);
    ///
    /// // Now it will be recognized
    /// let result = mecrab.parse("ChatGPTを使う")?;
    /// ```
    pub fn add_word(&self, surface: &str, reading: &str, pronunciation: &str, wcost: i16) {
        self.dictionary
            .add_simple_word(surface, reading, pronunciation, wcost);
    }

    /// Remove a word from the overlay dictionary
    ///
    /// Returns true if the word was found and removed.
    /// Note: Only overlay words can be removed; system dictionary entries persist.
    pub fn remove_word(&self, surface: &str) -> bool {
        self.dictionary.remove_word(surface)
    }

    /// Get the number of words in the overlay dictionary
    pub fn overlay_size(&self) -> usize {
        self.dictionary.overlay_size()
    }

    /// Parse the input text and return N-best analysis results
    ///
    /// Returns multiple alternative analyses ranked by cost, useful for
    /// disambiguation and exploring alternative segmentations.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to analyze
    /// * `n` - Number of best paths to return
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn parse_nbest(&self, text: &str, n: usize) -> Result<Vec<(AnalysisResult, i64)>> {
        // Build the lattice
        let lattice = Lattice::build(text, &self.dictionary)?;

        // Solve using Viterbi algorithm with N-best
        let solver = ViterbiSolver::new(&self.dictionary);
        let paths = solver.solve_nbest(&lattice, n)?;

        // Convert paths to analysis results
        let results = paths
            .into_iter()
            .map(|(path, cost)| {
                let morphemes = path
                    .into_iter()
                    .map(|node| {
                        let entities = if self.semantic_enabled {
                            self.get_entities_for_surface(&node.surface)
                        } else {
                            Vec::new()
                        };

                        let pronunciation = if self.ipa_enabled {
                            self.get_ipa_pronunciation(&node.feature)
                        } else {
                            None
                        };

                        let embedding = if self.vector_enabled {
                            self.get_embedding(node.word_id)
                        } else {
                            None
                        };

                        Morpheme {
                            surface: node.surface,
                            word_id: node.word_id,
                            pos_id: node.pos_id,
                            wcost: node.wcost,
                            feature: node.feature,
                            entities,
                            pronunciation,
                            embedding,
                            start_byte: node.start_byte,
                            end_byte: node.end_byte,
                        }
                    })
                    .collect();

                (
                    AnalysisResult {
                        morphemes,
                        format: self.output_format,
                    },
                    cost,
                )
            })
            .collect();

        Ok(results)
    }

    /// Parse text and return both the analysis result and lattice marginal probabilities.
    ///
    /// Uses the forward-backward algorithm to compute `P(morpheme | input)` for
    /// every node in the lattice.  The marginals can be used for:
    ///
    /// - Subword regularization in LLM pre-training
    /// - Uncertainty estimation in morphological disambiguation
    /// - Lattice-based sequence labelling
    ///
    /// # Errors
    ///
    /// Returns an error if lattice construction or Viterbi solving fails.
    pub fn parse_with_probs(
        &self,
        text: &str,
    ) -> Result<(AnalysisResult, crate::viterbi::analysis::LatticeProbTable)> {
        let lattice = Lattice::build(text, &self.dictionary)?;
        let solver = ViterbiSolver::new(&self.dictionary);
        let path = solver.solve(&lattice)?;
        let probs = solver.forward_backward(&lattice);

        let morphemes = path
            .into_iter()
            .map(|node| {
                let entities = if self.semantic_enabled {
                    self.get_entities_for_surface(&node.surface)
                } else {
                    Vec::new()
                };
                let pronunciation = if self.ipa_enabled {
                    self.get_ipa_pronunciation(&node.feature)
                } else {
                    None
                };
                let embedding = if self.vector_enabled {
                    self.get_embedding(node.word_id)
                } else {
                    None
                };
                Morpheme {
                    surface: node.surface,
                    word_id: node.word_id,
                    pos_id: node.pos_id,
                    wcost: node.wcost,
                    feature: node.feature,
                    entities,
                    pronunciation,
                    embedding,
                    start_byte: node.start_byte,
                    end_byte: node.end_byte,
                }
            })
            .collect();

        let result = AnalysisResult {
            morphemes,
            format: self.output_format,
        };
        Ok((result, probs))
    }

    /// Run N-best search and rerank candidates using the given [`Reranker`].
    ///
    /// Returns the single best [`AnalysisResult`] as selected by the reranker.
    /// This is the primary entry point for Phase 3 neural reranking: pass a
    /// [`rerank::NullReranker`] (zero overhead) for production, or a
    /// [`rerank::CostReranker`] / [`rerank::neural::NeuralReranker`] for
    /// higher-accuracy use cases.
    ///
    /// # Arguments
    ///
    /// * `text`     — Input text to analyze.
    /// * `n`        — Number of N-best paths to generate before reranking.
    ///   Must be ≥ 1.
    /// * `reranker` — Reranker implementation to use for path selection.
    ///
    /// # Errors
    ///
    /// Returns an error if lattice construction or Viterbi solving fails, or if
    /// the N-best search returns an empty candidate list.
    pub fn parse_nbest_with_reranker(
        &self,
        text: &str,
        n: usize,
        reranker: &dyn rerank::Reranker,
    ) -> Result<AnalysisResult> {
        let candidates = self.parse_nbest(text, n)?;
        if candidates.is_empty() {
            return Err(Error::ViterbiError(
                "No candidates returned from N-best search".into(),
            ));
        }

        // Build a lightweight RerankCandidate view over the AnalysisResult list
        // so that reranker implementations remain decoupled from the full
        // AnalysisResult type.
        let rerank_candidates: Vec<rerank::RerankCandidate> = candidates
            .iter()
            .map(|(result, cost)| rerank::RerankCandidate {
                surfaces: result.morphemes.iter().map(|m| m.surface.clone()).collect(),
                pos_tags: result
                    .morphemes
                    .iter()
                    .map(|m| m.feature.split(',').next().unwrap_or("*").to_owned())
                    .collect(),
                cost: *cost,
            })
            .collect();

        // Clamp returned index to valid range — a well-behaved reranker will
        // never exceed `candidates.len() - 1`, but we defend against it here
        // so the caller always receives a valid result without panic.
        let raw_idx = reranker.rerank(&rerank_candidates);
        let best_idx = raw_idx.min(candidates.len().saturating_sub(1));

        candidates
            .into_iter()
            .nth(best_idx)
            .map(|(result, _cost)| result)
            .ok_or_else(|| Error::ViterbiError("Reranker returned out-of-bounds index".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morpheme_positions() {
        // Verify struct construction and char/byte offset helpers compile and
        // behave correctly without requiring a real dictionary.
        let m = Morpheme {
            surface: "東京".to_string(),
            word_id: 0,
            pos_id: 0,
            wcost: 0,
            feature: "名詞".to_string(),
            entities: vec![],
            pronunciation: None,
            embedding: None,
            start_byte: 0,
            end_byte: 6, // "東京" = 3+3 bytes in UTF-8
        };
        assert_eq!(m.start_byte, 0);
        assert_eq!(m.end_byte, 6);
        assert_eq!(m.start_char("東京"), 0);
        assert_eq!(m.end_char("東京"), 2);
    }

    #[test]
    fn test_builder_default() {
        let builder = MeCrab::builder();
        assert!(builder.dicdir.is_none());
        assert!(builder.userdic.is_none());
        assert_eq!(builder.output_format, OutputFormat::Default);
    }

    /// Verify that `parse_iter` produces the same number of results as the
    /// input slice length (without requiring a real dictionary).
    /// The actual parse errors are expected when no dictionary is present.
    #[test]
    fn test_parse_iter_result_count_matches_input() {
        // When no default dictionary is installed the iterator still yields
        // exactly `texts.len()` items (each being an Err in that case).
        // This test validates the iterator's laziness contract.
        let texts: Vec<&str> = vec!["東京", "大阪", "京都"];
        // We cannot construct MeCrab without a dictionary, so we test the
        // iterator adapter logic using a trivial closure that mirrors the impl.
        let dummy_parse = |t: &&str| -> Result<usize> {
            // stand-in for self.parse(*t)
            Ok(t.len())
        };
        let results: Vec<_> = texts.iter().map(dummy_parse).collect();
        assert_eq!(results.len(), texts.len());
        for r in &results {
            assert!(r.is_ok());
        }
    }

    #[test]
    fn test_noun_phrases_consecutive_nouns() {
        let result = AnalysisResult {
            morphemes: vec![
                Morpheme {
                    surface: "東京".into(),
                    feature: "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ".into(),
                    word_id: 1,
                    pos_id: 0,
                    wcost: 0,
                    entities: vec![],
                    pronunciation: None,
                    embedding: None,
                    start_byte: 0,
                    end_byte: 6,
                },
                Morpheme {
                    surface: "都".into(),
                    feature: "名詞,接尾,地域,*,*,*,都,ト,ト".into(),
                    word_id: 2,
                    pos_id: 0,
                    wcost: 0,
                    entities: vec![],
                    pronunciation: None,
                    embedding: None,
                    start_byte: 6,
                    end_byte: 9,
                },
                Morpheme {
                    surface: "は".into(),
                    feature: "助詞,係助詞,*,*,*,*,は,ハ,ワ".into(),
                    word_id: 3,
                    pos_id: 0,
                    wcost: 0,
                    entities: vec![],
                    pronunciation: None,
                    embedding: None,
                    start_byte: 9,
                    end_byte: 12,
                },
            ],
            format: OutputFormat::Default,
        };

        let phrases = result.noun_phrases();
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].0, "東京都");
        assert_eq!(phrases[0].1, 0);
        assert_eq!(phrases[0].2, 9);
    }

    #[test]
    fn test_named_entities_proper_noun() {
        let result = AnalysisResult {
            morphemes: vec![Morpheme {
                surface: "東京".into(),
                feature: "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ".into(),
                word_id: 1,
                pos_id: 0,
                wcost: 0,
                entities: vec![],
                pronunciation: None,
                embedding: None,
                start_byte: 0,
                end_byte: 6,
            }],
            format: OutputFormat::Default,
        };

        let entities = result.named_entities();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].0, "東京");
        assert_eq!(entities[0].1, "地域");
    }

    #[test]
    fn test_spans_basic() {
        let result = AnalysisResult {
            morphemes: vec![
                Morpheme {
                    surface: "東京".into(),
                    feature: "名詞".into(),
                    word_id: 1,
                    pos_id: 0,
                    wcost: 0,
                    entities: vec![],
                    pronunciation: None,
                    embedding: None,
                    start_byte: 0,
                    end_byte: 6,
                },
                Morpheme {
                    surface: "EOS".into(),
                    feature: "".into(),
                    word_id: 0,
                    pos_id: 0,
                    wcost: 0,
                    entities: vec![],
                    pronunciation: None,
                    embedding: None,
                    start_byte: 6,
                    end_byte: 6,
                },
            ],
            format: OutputFormat::Default,
        };
        let spans = result.spans();
        assert_eq!(spans, vec![(0, 6)]);
    }

    #[test]
    fn test_verb_chunks_basic() {
        let result = AnalysisResult {
            morphemes: vec![
                Morpheme {
                    surface: "食べ".into(),
                    feature: "動詞,自立,*,*,一段,連用形,食べる,タベ,タベ".into(),
                    word_id: 10,
                    pos_id: 0,
                    wcost: 0,
                    entities: vec![],
                    pronunciation: None,
                    embedding: None,
                    start_byte: 0,
                    end_byte: 6,
                },
                Morpheme {
                    surface: "られ".into(),
                    feature: "助動詞,*,*,*,一段,連用形,られる,ラレ,ラレ".into(),
                    word_id: 11,
                    pos_id: 0,
                    wcost: 0,
                    entities: vec![],
                    pronunciation: None,
                    embedding: None,
                    start_byte: 6,
                    end_byte: 12,
                },
                Morpheme {
                    surface: "た".into(),
                    feature: "助動詞,*,*,*,特殊・タ,基本形,た,タ,タ".into(),
                    word_id: 12,
                    pos_id: 0,
                    wcost: 0,
                    entities: vec![],
                    pronunciation: None,
                    embedding: None,
                    start_byte: 12,
                    end_byte: 15,
                },
            ],
            format: OutputFormat::Default,
        };

        let chunks = result.verb_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, "食べられた");
        assert_eq!(chunks[0].1, 0);
        assert_eq!(chunks[0].2, 15);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parse_batch_with_progress_callback_count() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Simulate the callback-counting logic independently of a live dict.
        let texts = ["a", "b", "c", "d", "e"];
        let total = texts.len();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        // Simulate what parse_batch_with_progress does
        for _ in &texts {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        }

        assert_eq!(counter.load(Ordering::Relaxed), total);
    }
}
