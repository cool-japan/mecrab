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
pub mod debug;
pub mod dict;
pub mod error;
pub mod lattice;
pub mod normalize;
pub mod phonetic;
pub mod semantic;
pub mod stream;
pub mod vectors;
pub mod viterbi;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "python")]
pub mod python;

pub use error::{Error, Result};

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use dict::Dictionary;
use lattice::Lattice;
use viterbi::ViterbiSolver;

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
    format: OutputFormat,
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
        }
    }
}

impl AnalysisResult {
    /// Format as JSON
    fn format_json(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, m) in self.morphemes.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(
                f,
                "{{\"surface\":\"{}\",\"feature\":\"{}\"}}",
                semantic::jsonld::escape_json(&m.surface),
                semantic::jsonld::escape_json(&m.feature)
            )?;
        }
        write!(f, "]")
    }

    /// Format as JSON-LD with semantic URIs
    fn format_jsonld(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{{")?;
        writeln!(f, "  \"@context\": {{")?;
        writeln!(f, "    \"wd\": \"http://www.wikidata.org/entity/\",")?;
        writeln!(f, "    \"dbr\": \"http://dbpedia.org/resource/\",")?;
        writeln!(f, "    \"schema\": \"http://schema.org/\",")?;
        writeln!(f, "    \"mecrab\": \"http://mecrab.io/ns#\"")?;
        writeln!(f, "  }},")?;
        writeln!(f, "  \"@type\": \"mecrab:Analysis\",")?;
        writeln!(f, "  \"tokens\": [")?;

        for (i, m) in self.morphemes.iter().enumerate() {
            // Parse feature string to extract reading if available
            let features: Vec<&str> = m.feature.split(',').collect();
            let reading = features.get(7).copied(); // IPADIC format: reading is at index 7

            writeln!(f, "    {{")?;
            writeln!(
                f,
                "      \"surface\": \"{}\",",
                semantic::jsonld::escape_json(&m.surface)
            )?;
            writeln!(
                f,
                "      \"pos\": \"{}\",",
                features.first().copied().unwrap_or("*")
            )?;
            if let Some(r) = reading {
                if r != "*" {
                    writeln!(f, "      \"reading\": \"{}\",", r)?;
                }
            }

            // Add IPA pronunciation if available
            if let Some(ref ipa) = m.pronunciation {
                writeln!(f, "      \"pronunciation\": \"/{}/ \",", ipa)?;
            }

            // Add embedding vector if available
            if let Some(ref embedding) = m.embedding {
                write!(f, "      \"embedding\": [")?;
                for (j, val) in embedding.iter().enumerate() {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.3}", val)?;
                }
                writeln!(f, "],")?;
            }

            // Determine if we need trailing comma after wcost
            let has_entities = !m.entities.is_empty();

            if has_entities {
                writeln!(f, "      \"wcost\": {},", m.wcost)?;
                writeln!(f, "      \"entities\": [")?;
                for (j, entity) in m.entities.iter().enumerate() {
                    let compact = semantic::compact_uri(&entity.uri);
                    write!(
                        f,
                        "        {{\"@id\": \"{}\", \"confidence\": {:.2}}}",
                        compact, entity.confidence
                    )?;
                    if j < m.entities.len() - 1 {
                        writeln!(f, ",")?;
                    } else {
                        writeln!(f)?;
                    }
                }
                write!(f, "      ]")?;
            } else {
                write!(f, "      \"wcost\": {}", m.wcost)?;
            }

            if i < self.morphemes.len() - 1 {
                writeln!(f, "\n    }},")?;
            } else {
                writeln!(f, "\n    }}")?;
            }
        }

        writeln!(f, "  ]")?;
        write!(f, "}}")
    }

    /// Format as Turtle (TTL) RDF
    fn format_turtle(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Prepare token data for export
        let tokens: Vec<(String, String, Option<String>, Vec<semantic::SemanticEntry>)> = self
            .morphemes
            .iter()
            .map(|m| {
                let features: Vec<&str> = m.feature.split(',').collect();
                let pos = features.first().copied().unwrap_or("*").to_string();
                let reading = features
                    .get(7)
                    .filter(|&&r| r != "*")
                    .map(|&r| r.to_string());

                // Convert EntityReference to SemanticEntry
                let entities: Vec<semantic::SemanticEntry> = m
                    .entities
                    .iter()
                    .map(|e| {
                        semantic::SemanticEntry::new(
                            &e.uri,
                            e.confidence,
                            semantic::OntologySource::Wikidata,
                        )
                    })
                    .collect();

                (m.surface.clone(), pos, reading, entities)
            })
            .collect();

        let turtle = semantic::rdf::export_turtle(&tokens, "http://example.org/analysis");
        write!(f, "{}", turtle)
    }

    /// Format as N-Triples RDF
    fn format_ntriples(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Prepare token data for export
        let tokens: Vec<(String, String, Option<String>, Vec<semantic::SemanticEntry>)> = self
            .morphemes
            .iter()
            .map(|m| {
                let features: Vec<&str> = m.feature.split(',').collect();
                let pos = features.first().copied().unwrap_or("*").to_string();
                let reading = features
                    .get(7)
                    .filter(|&&r| r != "*")
                    .map(|&r| r.to_string());

                let entities: Vec<semantic::SemanticEntry> = m
                    .entities
                    .iter()
                    .map(|e| {
                        semantic::SemanticEntry::new(
                            &e.uri,
                            e.confidence,
                            semantic::OntologySource::Wikidata,
                        )
                    })
                    .collect();

                (m.surface.clone(), pos, reading, entities)
            })
            .collect();

        let ntriples = semantic::rdf::export_ntriples(&tokens, "http://example.org/analysis");
        write!(f, "{}", ntriples)
    }

    /// Format as N-Quads RDF
    fn format_nquads(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Prepare token data for export
        let tokens: Vec<(String, String, Option<String>, Vec<semantic::SemanticEntry>)> = self
            .morphemes
            .iter()
            .map(|m| {
                let features: Vec<&str> = m.feature.split(',').collect();
                let pos = features.first().copied().unwrap_or("*").to_string();
                let reading = features
                    .get(7)
                    .filter(|&&r| r != "*")
                    .map(|&r| r.to_string());

                let entities: Vec<semantic::SemanticEntry> = m
                    .entities
                    .iter()
                    .map(|e| {
                        semantic::SemanticEntry::new(
                            &e.uri,
                            e.confidence,
                            semantic::OntologySource::Wikidata,
                        )
                    })
                    .collect();

                (m.surface.clone(), pos, reading, entities)
            })
            .collect();

        let nquads = semantic::rdf::export_nquads(
            &tokens,
            "http://example.org/analysis",
            "http://example.org/graph",
        );
        write!(f, "{}", nquads)
    }
}

/// Builder for configuring MeCrab instance
#[derive(Debug, Default)]
pub struct MeCrabBuilder {
    dicdir: Option<PathBuf>,
    userdic: Option<PathBuf>,
    semantic_pool: Option<PathBuf>,
    vector_pool: Option<PathBuf>,
    with_semantic: bool,
    with_ipa: bool,
    with_vector: bool,
    output_format: OutputFormat,
}

impl MeCrabBuilder {
    /// Create a new builder with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the dictionary directory
    #[must_use]
    pub fn dicdir(mut self, path: Option<PathBuf>) -> Self {
        self.dicdir = path;
        self
    }

    /// Set the user dictionary path
    #[must_use]
    pub fn userdic(mut self, path: Option<PathBuf>) -> Self {
        self.userdic = path;
        self
    }

    /// Set the semantic pool path
    #[must_use]
    pub fn semantic_pool(mut self, path: Option<PathBuf>) -> Self {
        self.semantic_pool = path;
        self
    }

    /// Enable semantic URI output (requires semantic pool to be loaded)
    #[must_use]
    pub fn with_semantic(mut self, enabled: bool) -> Self {
        self.with_semantic = enabled;
        self
    }

    /// Enable IPA pronunciation output
    #[must_use]
    pub fn with_ipa(mut self, enabled: bool) -> Self {
        self.with_ipa = enabled;
        self
    }

    /// Set the vector pool file path (vectors.bin)
    #[must_use]
    pub fn vector_pool(mut self, path: Option<PathBuf>) -> Self {
        self.vector_pool = path;
        self
    }

    /// Enable vector embedding output
    #[must_use]
    pub fn with_vector(mut self, enabled: bool) -> Self {
        self.with_vector = enabled;
        self
    }

    /// Set the output format
    #[must_use]
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Build the MeCrab instance
    ///
    /// # Errors
    ///
    /// Returns an error if the dictionary cannot be loaded.
    pub fn build(self) -> Result<MeCrab> {
        let dictionary = match (self.dicdir, self.semantic_pool) {
            (Some(dicdir), Some(semantic_path)) => {
                Dictionary::load_with_semantics(&dicdir, &semantic_path)?
            }
            (Some(dicdir), None) => {
                // Try to auto-load semantic.bin from dicdir if it exists
                let semantic_path = dicdir.join("semantic.bin");
                if semantic_path.exists() {
                    Dictionary::load_with_semantics(&dicdir, &semantic_path)?
                } else {
                    Dictionary::load(&dicdir)?
                }
            }
            (None, Some(semantic_path)) => {
                let dict = Dictionary::default_dictionary()?;
                let pool_file = std::fs::File::open(&semantic_path)?;
                let pool_data = unsafe { memmap2::Mmap::map(&pool_file)? };
                let pool = crate::semantic::pool::SemanticPool::from_bytes(&pool_data)?;
                let mut dict_mut = dict;
                dict_mut.semantic_pool = Some(Arc::new(pool));
                dict_mut
            }
            (None, None) => {
                // Try to auto-load from default directory
                Dictionary::default_dictionary()?
            }
        };

        // Load vector store if path provided
        let vector_store = if let Some(vector_path) = self.vector_pool {
            Some(Arc::new(vectors::VectorStore::from_file(&vector_path)?))
        } else {
            None
        };

        Ok(MeCrab {
            dictionary: Arc::new(dictionary),
            output_format: self.output_format,
            semantic_enabled: self.with_semantic,
            ipa_enabled: self.with_ipa,
            vector_enabled: self.with_vector,
            vector_store,
        })
    }
}

/// The main MeCrab morphological analyzer
#[derive(Clone)]
pub struct MeCrab {
    dictionary: Arc<Dictionary>,
    output_format: OutputFormat,
    semantic_enabled: bool,
    ipa_enabled: bool,
    vector_enabled: bool,
    vector_store: Option<Arc<vectors::VectorStore>>,
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

    /// Parse multiple texts in parallel using Rayon
    ///
    /// This method leverages all available CPU cores for batch processing,
    /// providing significant speedup for large workloads.
    ///
    /// # Errors
    ///
    /// Returns a vector of results, where each result may be an error.
    #[cfg(feature = "parallel")]
    pub fn parse_batch(&self, texts: &[&str]) -> Vec<Result<AnalysisResult>> {
        use rayon::prelude::*;
        texts.par_iter().map(|text| self.parse(text)).collect()
    }

    /// Parse multiple texts sequentially (fallback when parallel feature is disabled)
    #[cfg(not(feature = "parallel"))]
    pub fn parse_batch(&self, texts: &[&str]) -> Vec<Result<AnalysisResult>> {
        texts.iter().map(|text| self.parse(text)).collect()
    }

    /// Parse multiple texts and return wakati outputs in parallel
    ///
    /// # Errors
    ///
    /// Returns a vector of results.
    #[cfg(feature = "parallel")]
    pub fn wakati_batch(&self, texts: &[&str]) -> Vec<Result<String>> {
        use rayon::prelude::*;
        texts.par_iter().map(|text| self.wakati(text)).collect()
    }

    /// Parse multiple texts and return wakati outputs sequentially
    #[cfg(not(feature = "parallel"))]
    pub fn wakati_batch(&self, texts: &[&str]) -> Vec<Result<String>> {
        texts.iter().map(|text| self.wakati(text)).collect()
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        let builder = MeCrab::builder();
        assert!(builder.dicdir.is_none());
        assert!(builder.userdic.is_none());
        assert_eq!(builder.output_format, OutputFormat::Default);
    }
}
