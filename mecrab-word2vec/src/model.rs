//! Word2Vec model structure and builder

use crate::subword::CharNgramExtractor;
use crate::trainer::Trainer;
use crate::vocab::Vocabulary;
use crate::{Result, Word2VecError};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Compute the cosine similarity between two equal-length float slices.
///
/// Returns `0.0` if either vector has zero norm (avoids division by zero).
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Training objective for word2vec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingObjective {
    /// Skip-gram: predict context from center word (default, better for rare words).
    #[default]
    SkipGram,
    /// CBOW: predict center word from context (faster training, better for frequent words).
    Cbow,
}

/// Configuration for FastText-style subword embeddings.
#[derive(Debug, Clone)]
pub struct SubwordConfig {
    /// Minimum character n-gram length (default: 3)
    pub min_n: usize,
    /// Maximum character n-gram length (default: 6)
    pub max_n: usize,
    /// Number of hash buckets for n-gram table (default: 2_000_000)
    pub bucket_count: usize,
}

impl Default for SubwordConfig {
    fn default() -> Self {
        Self {
            min_n: 3,
            max_n: 6,
            bucket_count: 2_000_000,
        }
    }
}

/// Word2Vec model
pub struct Word2Vec {
    /// Model configuration
    config: TrainingConfig,
    /// Vocabulary
    vocab: Arc<Vocabulary>,
    /// Input embeddings (word vectors)
    /// Shape: [vocab_size, vector_size]
    pub syn0: Vec<f32>,
    /// Output embeddings (context vectors for negative sampling)
    /// Shape: [vocab_size, vector_size]
    pub syn1neg: Vec<f32>,
    /// Subword n-gram embedding table (FastText-style)
    /// Shape: [bucket_count, vector_size] — only populated when subword is enabled
    pub syn_ng: Vec<f32>,
    /// Optional surface map: word_id → surface string for subword training
    surface_map: Option<HashMap<u32, String>>,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Embedding vector size
    pub vector_size: usize,
    /// Context window size
    pub window_size: usize,
    /// Number of negative samples
    pub negative_samples: usize,
    /// Discard words with frequency below this threshold during vocabulary building.
    /// Words below min_count are treated as OOV during training.
    pub min_count: u32,
    /// Subsampling threshold
    pub sample: f64,
    /// Subsampling threshold for frequent words.
    /// Words are discarded during training with probability: 1 - sqrt(t / freq)
    /// where freq = word_count / total_words and t = subsample_threshold.
    /// Set to 0.0 to disable subsampling.
    /// Typical value: 1e-4 (= 0.0001).
    pub subsample_threshold: f32,
    /// Initial learning rate
    pub alpha: f32,
    /// Minimum learning rate
    pub min_alpha: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Number of threads
    pub threads: usize,
    /// Optional FastText-style subword configuration
    pub subword: Option<SubwordConfig>,
    /// Attempt GPU-accelerated training when the `gpu` feature is enabled.
    /// Falls back silently to CPU Hogwild! when no wgpu adapter is available.
    pub use_gpu: bool,
    /// Training objective: skip-gram (default) or CBOW.
    pub objective: TrainingObjective,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            vector_size: 100,
            window_size: 5,
            negative_samples: 5,
            min_count: 10,
            sample: 1e-4,
            subsample_threshold: 0.0,
            alpha: 0.025,
            min_alpha: 0.0001,
            epochs: 3,
            threads: 8,
            subword: None,
            use_gpu: false,
            objective: TrainingObjective::SkipGram,
        }
    }
}

impl Word2Vec {
    /// Create a new Word2Vec model with given configuration
    pub fn new(config: TrainingConfig, vocab: Vocabulary) -> Self {
        let vocab_size = vocab.len();
        let max_word_id = vocab.max_word_id();
        let vector_size = config.vector_size;

        // Use dense indexing: remapped_ids are 0-based and contiguous
        // This is MUCH more cache-friendly than sparse word_id indexing
        let array_size = vocab_size * vector_size;

        // Initialize embeddings with small random values
        let mut syn0 = vec![0.0f32; array_size];
        let syn1neg = vec![0.0f32; array_size];

        use rand::RngExt;
        let mut rng = rand::rng();

        // Initialize all vectors (remapped_ids are dense 0..vocab_size-1)
        for remapped_id in 0..vocab_size {
            let offset = remapped_id * vector_size;
            for i in 0..vector_size {
                syn0[offset + i] = (rng.random::<f32>() - 0.5) / vector_size as f32;
            }
        }

        // syn1neg initialized to zeros (common practice)

        // Initialize subword n-gram table if configured
        let syn_ng = if let Some(ref sw) = config.subword {
            let ng_size = sw.bucket_count * vector_size;
            let mut table = vec![0.0f32; ng_size];
            for val in table.iter_mut() {
                *val = (rng.random::<f32>() - 0.5) / vector_size as f32;
            }
            let mb = ng_size * 4 / 1024 / 1024;
            eprintln!("Subword table: {} buckets ({} MB)", sw.bucket_count, mb);
            table
        } else {
            Vec::new()
        };

        eprintln!("Model initialized:");
        eprintln!("  Vocab size (trained): {}", vocab_size);
        eprintln!("  Max word_id (MeCab): {}", max_word_id);
        eprintln!(
            "  Array size: {} elements ({} MB)",
            array_size,
            array_size * 4 / 1024 / 1024
        );
        if vocab_size > 0 {
            eprintln!("  Indexing: DENSE (remapped IDs 0-{})", vocab_size - 1);
        } else {
            eprintln!("  Indexing: DENSE (empty vocabulary)");
        }

        Self {
            config,
            vocab: Arc::new(vocab),
            syn0,
            syn1neg,
            syn_ng,
            surface_map: None,
        }
    }

    /// Attach a surface map for subword training: word_id → surface string.
    ///
    /// The map is optional; if omitted during subword training the word_id
    /// is stringified to produce digit character n-grams.
    pub fn with_surface_map(mut self, surface_map: HashMap<u32, String>) -> Self {
        self.surface_map = Some(surface_map);
        self
    }

    /// Set or replace the surface map after construction.
    pub fn set_surface_map(&mut self, surface_map: HashMap<u32, String>) {
        self.surface_map = Some(surface_map);
    }

    /// Train model from corpus file
    pub fn train_from_file<P: AsRef<Path>>(&mut self, corpus_path: P) -> Result<()> {
        let mut trainer = Trainer::new(corpus_path.as_ref(), self.vocab.clone(), &self.config);

        // Attach surface_map and extractor when subword is configured
        if let Some(ref sw) = self.config.subword.clone() {
            let extractor = CharNgramExtractor::new(sw.min_n, sw.max_n, sw.bucket_count);
            trainer = trainer.with_subword_extractor(extractor);
            if let Some(ref sm) = self.surface_map {
                trainer = trainer.with_surface_map(sm.clone());
            }
        }

        trainer.train(&mut self.syn0, &mut self.syn1neg, &mut self.syn_ng)?;
        Ok(())
    }

    /// Save embeddings in word2vec text format
    pub fn save_text<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        crate::io::save_word2vec_text(
            path,
            &self.syn0,
            self.vocab.as_ref(),
            self.config.vector_size,
        )
    }

    /// Save embeddings in MCV1 binary format
    pub fn save_mcv1<P: AsRef<Path>>(&self, path: P, max_word_id: u32) -> Result<()> {
        crate::io::save_mcv1_format(
            path,
            &self.syn0,
            self.vocab.as_ref(),
            self.config.vector_size,
            max_word_id,
        )
    }

    /// Save the FastText subword n-gram table to a companion text file.
    ///
    /// The companion file path is `<base_path>.subword` — e.g., if you called
    /// `save_text("vectors.txt")`, call `save_subword_text("vectors.txt.subword")`.
    ///
    /// Only non-zero n-gram buckets are written (sparse format).
    /// Returns `Ok(())` silently if subword is not enabled (`syn_ng` is empty).
    pub fn save_subword_text<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let config = match &self.config.subword {
            Some(c) => c,
            None => return Ok(()), // not a subword model — no-op
        };
        if self.syn_ng.is_empty() {
            return Ok(());
        }
        crate::io::save_subword_sparse_text(path, &self.syn_ng, config, self.config.vector_size)
    }

    /// Load the FastText subword n-gram table from a companion text file.
    ///
    /// After loading, `embed_surface()` and `embed_word_with_subword()` will work.
    /// The model does not need to have been trained with subword — any model can load
    /// a subword table (useful for inference-only deployments).
    ///
    /// Returns an error if the file is missing or malformed.
    pub fn load_subword_text<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let (config, syn_ng) = crate::io::load_subword_sparse_text(path)?;
        self.config.subword = Some(config);
        self.syn_ng = syn_ng;
        Ok(())
    }

    /// Get vocabulary
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Get configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Embed a surface form using subword n-grams only (for OOV words).
    ///
    /// Returns `None` if subword is not enabled or bucket table is empty.
    /// For OOV words, the embedding is the average of the n-gram bucket vectors.
    pub fn embed_surface(&self, surface: &str) -> Option<Vec<f32>> {
        let sw = self.config.subword.as_ref()?;
        if self.syn_ng.is_empty() {
            return None;
        }

        let extractor = CharNgramExtractor::new(sw.min_n, sw.max_n, sw.bucket_count);
        let bucket_ids = extractor.extract_bucket_ids(surface);

        if bucket_ids.is_empty() {
            return None;
        }

        let vector_size = self.config.vector_size;
        let num_buckets = bucket_ids.len() as f32;
        let mut result = vec![0.0f32; vector_size];

        for &bucket_id in &bucket_ids {
            let offset = bucket_id as usize * vector_size;
            if let Some(slice) = self.syn_ng.get(offset..offset + vector_size) {
                for (r, &v) in result.iter_mut().zip(slice.iter()) {
                    *r += v;
                }
            }
        }

        // Average over all n-gram buckets
        for r in result.iter_mut() {
            *r /= num_buckets;
        }

        Some(result)
    }

    /// Embed a word by combining its syn0 vector with subword n-gram vectors.
    ///
    /// If `word_remapped_id` is `None` (OOV), uses only n-gram vectors (average).
    /// If `word_remapped_id` is `Some`, returns syn0 + sum of n-gram bucket vectors.
    /// Returns `None` if subword is not enabled.
    pub fn embed_word_with_subword(
        &self,
        word_remapped_id: Option<u32>,
        surface: &str,
    ) -> Option<Vec<f32>> {
        let sw = self.config.subword.as_ref()?;
        if self.syn_ng.is_empty() {
            return None;
        }

        let extractor = CharNgramExtractor::new(sw.min_n, sw.max_n, sw.bucket_count);
        let bucket_ids = extractor.extract_bucket_ids(surface);
        let vector_size = self.config.vector_size;

        match word_remapped_id {
            None => {
                // OOV: average over n-gram buckets only
                self.embed_surface(surface)
            }
            Some(remapped_id) => {
                let syn0_offset = remapped_id as usize * vector_size;
                let syn0_slice = self.syn0.get(syn0_offset..syn0_offset + vector_size)?;

                let mut result: Vec<f32> = syn0_slice.to_vec();

                // Add all n-gram bucket vectors
                for &bucket_id in &bucket_ids {
                    let ng_offset = bucket_id as usize * vector_size;
                    if let Some(slice) = self.syn_ng.get(ng_offset..ng_offset + vector_size) {
                        for (r, &v) in result.iter_mut().zip(slice.iter()) {
                            *r += v;
                        }
                    }
                }

                Some(result)
            }
        }
    }

    // ── Query API ────────────────────────────────────────────────────────────

    /// Return the syn0 embedding slice for `word_id`, or `None` if not in vocabulary.
    ///
    /// `word_id` is the ORIGINAL word id (as in the corpus vocabulary),
    /// not the internal remapped (dense training) index.
    pub fn get_vector(&self, word_id: u32) -> Option<&[f32]> {
        let remapped = self.vocab.get_remapped_id(word_id)?;
        let vs = self.config.vector_size;
        let start = remapped as usize * vs;
        self.syn0.get(start..start + vs)
    }

    /// Cosine similarity between two words identified by their original `word_id`.
    ///
    /// Returns `None` if either word is out-of-vocabulary.
    pub fn similarity(&self, a: u32, b: u32) -> Option<f32> {
        Some(cosine_similarity(self.get_vector(a)?, self.get_vector(b)?))
    }

    /// Return the top-`k` most similar words to `word_id` (excluding itself).
    ///
    /// Results are `(word_id, cosine_similarity)` pairs sorted in descending order.
    /// Returns an empty `Vec` if `word_id` is out-of-vocabulary or `k == 0`.
    pub fn most_similar(&self, word_id: u32, k: usize) -> Vec<(u32, f32)> {
        if k == 0 {
            return Vec::new();
        }
        let query = match self.get_vector(word_id) {
            Some(v) => v,
            None => return Vec::new(),
        };
        self.most_similar_by_vec(query, k, Some(word_id))
    }

    /// Return the top-`k` most similar words to an arbitrary query vector.
    ///
    /// `exclude_id`: optional original `word_id` to exclude from results
    /// (useful to exclude the query word itself when the vector comes from the model).
    ///
    /// Returns an empty `Vec` if `query.len() != vector_size` or `k == 0`.
    pub fn most_similar_by_vec(
        &self,
        query: &[f32],
        k: usize,
        exclude_id: Option<u32>,
    ) -> Vec<(u32, f32)> {
        let vs = self.config.vector_size;
        if query.len() != vs || k == 0 {
            return Vec::new();
        }

        let exclude_remapped: Option<u32> =
            exclude_id.and_then(|id| self.vocab.get_remapped_id(id));

        let mut scores: Vec<(u32, f32)> = (0..self.vocab.len() as u32)
            .filter(|&ri| Some(ri) != exclude_remapped)
            .filter_map(|ri| {
                let start = ri as usize * vs;
                let vec = self.syn0.get(start..start + vs)?;
                let word_id = self.vocab.get_word_id(ri)?;
                Some((word_id, cosine_similarity(query, vec)))
            })
            .collect();

        scores.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores
    }

    /// Word analogy: find words similar to `(vec[a] - vec[b] + vec[c])`.
    ///
    /// Words `a`, `b`, and `c` are excluded from the result list.
    /// Returns top-`k` `(word_id, cosine_similarity)` pairs sorted descending.
    /// Returns an empty `Vec` if any of the three words is out-of-vocabulary.
    pub fn analogy(&self, a: u32, b: u32, c: u32, k: usize) -> Vec<(u32, f32)> {
        if k == 0 {
            return Vec::new();
        }
        let va = match self.get_vector(a) {
            Some(v) => v.to_vec(),
            None => return Vec::new(),
        };
        let vb = match self.get_vector(b) {
            Some(v) => v.to_vec(),
            None => return Vec::new(),
        };
        let vc = match self.get_vector(c) {
            Some(v) => v.to_vec(),
            None => return Vec::new(),
        };

        // Compute a - b + c
        let query: Vec<f32> = va
            .iter()
            .zip(vb.iter().zip(vc.iter()))
            .map(|(&ai, (&bi, &ci))| ai - bi + ci)
            .collect();

        // Collect remapped IDs to exclude (a, b, c)
        let mut exclude_remapped: std::collections::HashSet<u32> =
            std::collections::HashSet::with_capacity(3);
        for &id in &[a, b, c] {
            if let Some(ri) = self.vocab.get_remapped_id(id) {
                exclude_remapped.insert(ri);
            }
        }

        let vs = self.config.vector_size;
        let mut scores: Vec<(u32, f32)> = (0..self.vocab.len() as u32)
            .filter(|ri| !exclude_remapped.contains(ri))
            .filter_map(|ri| {
                let start = ri as usize * vs;
                let vec = self.syn0.get(start..start + vs)?;
                let word_id = self.vocab.get_word_id(ri)?;
                Some((word_id, cosine_similarity(&query, vec)))
            })
            .collect();

        scores.sort_unstable_by(|x, y| {
            y.1.partial_cmp(&x.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores
    }

    // ── Surface-string convenience wrappers ──────────────────────────────────

    /// Look up the original `word_id` for a surface string using the surface map.
    ///
    /// Returns `None` if no surface map is attached or the string is not found.
    fn surface_to_word_id(&self, surface: &str) -> Option<u32> {
        self.surface_map
            .as_ref()?
            .iter()
            .find_map(|(&id, s)| if s == surface { Some(id) } else { None })
    }

    /// Cosine similarity between two words identified by their surface string.
    ///
    /// Returns `None` if either surface is not in the surface map or vocabulary.
    pub fn similarity_by_surface(&self, a: &str, b: &str) -> Option<f32> {
        let a_id = self.surface_to_word_id(a)?;
        let b_id = self.surface_to_word_id(b)?;
        self.similarity(a_id, b_id)
    }

    /// Top-`k` most similar words to `surface`, excluding itself.
    ///
    /// Returns an empty `Vec` if the surface is not in the surface map.
    pub fn most_similar_by_surface(&self, surface: &str, k: usize) -> Vec<(u32, f32)> {
        match self.surface_to_word_id(surface) {
            Some(id) => self.most_similar(id, k),
            None => Vec::new(),
        }
    }

    // ── Model reload ─────────────────────────────────────────────────────────

    /// Load a previously saved text-format word2vec model.
    ///
    /// Reconstructs `syn0`, vocabulary mappings, and `surface_map` from the file.
    /// The returned model supports all query methods (`get_vector`, `most_similar`,
    /// `analogy`, etc.) but cannot resume training (`syn1neg` is not saved in text
    /// format and is left empty).
    ///
    /// # File format (written by `save_text`)
    /// ```text
    /// <vocab_size> <vector_size>
    /// <word_id> <v1> <v2> … <vN>
    /// …
    /// ```
    pub fn load_text<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::io::{BufRead, BufReader};

        let file = std::fs::File::open(path.as_ref())
            .map_err(Word2VecError::Io)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // ── Parse header: "<vocab_size> <vector_size>" ──────────────────────
        let header = lines
            .next()
            .ok_or_else(|| Word2VecError::InvalidParameter("empty file".into()))??;
        let mut header_parts = header.split_whitespace();
        let vocab_size: usize = header_parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| {
                Word2VecError::InvalidParameter("bad header: missing vocab_size".into())
            })?;
        let vector_size: usize = header_parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| {
                Word2VecError::InvalidParameter("bad header: missing vector_size".into())
            })?;

        let mut syn0: Vec<f32> = Vec::with_capacity(vocab_size * vector_size);
        // surface_map: word_id → surface string (here word_id IS the surface token)
        let mut surface_map: HashMap<u32, String> = HashMap::with_capacity(vocab_size);
        // vocab entries: (word_id, count) in load order; position == remapped_id
        let mut vocab_entries: Vec<(u32, u64)> = Vec::with_capacity(vocab_size);
        let mut word_id_counter: u32 = 0;

        // ── Parse data lines: "<word_id_str> <v1> <v2> … <vN>" ─────────────
        for raw_line in lines {
            let line = raw_line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // The first token is the word label (original word_id as a string).
            // Use splitn(2, ' ') so we can parse the rest lazily.
            let mut parts = line.splitn(2, ' ');
            let word_label = parts.next().unwrap_or("").to_string();
            let vec_str = parts.next().unwrap_or("");

            // Parse the vector
            let vec: Vec<f32> = vec_str
                .split_whitespace()
                .map(|s| {
                    s.parse::<f32>().map_err(|e| {
                        Word2VecError::InvalidParameter(format!(
                            "cannot parse float in row '{word_label}': {e}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            if vec.len() != vector_size {
                return Err(Word2VecError::InvalidParameter(format!(
                    "vector length mismatch for '{word_label}': expected {vector_size}, got {}",
                    vec.len()
                )));
            }

            syn0.extend_from_slice(&vec);

            // Try to interpret the label as a numeric word_id; fall back to counter.
            let word_id: u32 = word_label
                .parse::<u32>()
                .unwrap_or(word_id_counter);

            // position in vocab_entries == remapped_id
            let remapped_id = vocab_entries.len() as u32;
            _ = remapped_id; // used implicitly via push ordering
            vocab_entries.push((word_id, 1));
            surface_map.insert(word_id, word_label);
            word_id_counter += 1;
        }

        // ── Build vocabulary with identity-like mapping ─────────────────────
        let vocab = Vocabulary::from_word_id_list(&vocab_entries);

        // ── Assemble config (only vector_size is meaningful for inference) ──
        let config = TrainingConfig {
            vector_size,
            ..Default::default()
        };

        Ok(Self {
            config,
            vocab: Arc::new(vocab),
            syn0,
            syn1neg: Vec::new(), // not saved in text format
            syn_ng: Vec::new(),
            surface_map: Some(surface_map),
        })
    }
}

/// Builder for Word2Vec model
#[derive(Default)]
pub struct Word2VecBuilder {
    config: TrainingConfig,
}

impl Word2VecBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set vector size (default: 100)
    pub fn vector_size(mut self, size: usize) -> Self {
        self.config.vector_size = size;
        self
    }

    /// Set window size (default: 5)
    pub fn window_size(mut self, size: usize) -> Self {
        self.config.window_size = size;
        self
    }

    /// Set number of negative samples (default: 5)
    pub fn negative_samples(mut self, n: usize) -> Self {
        self.config.negative_samples = n;
        self
    }

    /// Set minimum word count threshold (default: 10).
    ///
    /// Words whose corpus frequency falls below this value are excluded from
    /// the vocabulary and treated as OOV during training.
    pub fn min_count(mut self, min_count: u32) -> Self {
        self.config.min_count = min_count;
        self
    }

    /// Set subsampling threshold (default: 1e-4)
    pub fn sample(mut self, threshold: f64) -> Self {
        self.config.sample = threshold;
        self
    }

    /// Set the subsampling threshold for frequent words (default: 0.0 = disabled).
    ///
    /// Each word token is discarded during training with probability
    /// `1 - sqrt(t / freq)` where `freq = word_count / total_words` and
    /// `t = subsample_threshold`.  A typical value is `1e-4`.
    pub fn subsample_threshold(mut self, threshold: f32) -> Self {
        self.config.subsample_threshold = threshold;
        self
    }

    /// Set initial learning rate (default: 0.025)
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set minimum learning rate (default: 0.0001)
    pub fn min_alpha(mut self, alpha: f32) -> Self {
        self.config.min_alpha = alpha;
        self
    }

    /// Set number of epochs (default: 3)
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.config.epochs = epochs;
        self
    }

    /// Set number of threads (default: 8)
    pub fn threads(mut self, threads: usize) -> Self {
        self.config.threads = threads;
        self
    }

    /// Enable FastText-style subword embeddings.
    ///
    /// - `min_n`: minimum n-gram length (3 recommended for Japanese)
    /// - `max_n`: maximum n-gram length (6 recommended for Japanese)
    /// - `bucket_count`: n-gram hash table size (2_000_000 typical)
    pub fn with_subword(mut self, min_n: usize, max_n: usize, bucket_count: usize) -> Self {
        self.config.subword = Some(SubwordConfig {
            min_n,
            max_n,
            bucket_count,
        });
        self
    }

    /// Enable GPU-accelerated training (requires `--features gpu` at compile time).
    ///
    /// When enabled, [`Word2Vec::train_from_file`] attempts to acquire a wgpu GPU
    /// adapter and dispatches training to [`crate::gpu::GpuTrainer`].  If no adapter
    /// is available the training falls back silently to CPU Hogwild!.
    pub fn use_gpu(mut self, flag: bool) -> Self {
        self.config.use_gpu = flag;
        self
    }

    /// Set the training objective (default: skip-gram).
    ///
    /// - [`TrainingObjective::SkipGram`]: predicts context words from a center word.
    ///   Better for rare words, richer representations. Default.
    /// - [`TrainingObjective::Cbow`]: predicts the center word from averaged context.
    ///   Faster training, better for frequent words.
    pub fn objective(mut self, obj: TrainingObjective) -> Self {
        self.config.objective = obj;
        self
    }

    /// Convenience wrapper: equivalent to `.objective(TrainingObjective::Cbow)`.
    pub fn cbow(self) -> Self {
        self.objective(TrainingObjective::Cbow)
    }

    /// Build vocabulary from corpus and create model
    pub fn build_from_corpus<P: AsRef<Path>>(self, corpus_path: P) -> Result<Word2Vec> {
        let mut vocab = Vocabulary::new(self.config.min_count as u64, self.config.sample);
        vocab.build_from_file(&corpus_path)?;

        if vocab.is_empty() {
            return Err(Word2VecError::Vocabulary(
                "Vocabulary is empty after filtering".to_string(),
            ));
        }

        Ok(Word2Vec::new(self.config, vocab))
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_path(stem: &str, ext: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        // Include thread ID to prevent collisions between parallel test threads.
        let tid = std::thread::current().id();
        std::env::temp_dir().join(format!("mecrab_model_{stem}_{nanos}_{tid:?}.{ext}"))
    }

    fn make_corpus(sentences: &[&str]) -> std::path::PathBuf {
        let path = temp_path("corpus", "txt");
        let mut f = std::fs::File::create(&path).expect("create corpus");
        for s in sentences {
            writeln!(f, "{s}").expect("write");
        }
        path
    }

    fn tiny_model(corpus: &std::path::Path) -> Word2Vec {
        Word2VecBuilder::new()
            .vector_size(4)
            .window_size(2)
            .negative_samples(2)
            .min_count(1)
            .sample(0.0)
            .epochs(2)
            .threads(1)
            .build_from_corpus(corpus)
            .expect("build_from_corpus")
    }

    fn tiny_sentences() -> Vec<&'static str> {
        vec![
            "0 1 2 3 4 5",
            "1 2 3 4 5 0",
            "2 3 4 5 0 1",
            "3 4 5 0 1 2",
            "4 5 0 1 2 3",
            "5 0 1 2 3 4",
            "0 1 2 3 4 5",
            "1 2 3 4 5 0",
            "0 2 4 1 3 5",
            "5 3 1 4 2 0",
        ]
    }

    // ── cosine_similarity ─────────────────────────────────────────────────────

    #[test]
    fn test_cosine_same_vector_is_one() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "got {sim}");
    }

    #[test]
    fn test_cosine_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_zero_norm_returns_zero() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_antiparallel_is_neg_one() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    // ── similarity ────────────────────────────────────────────────────────────

    #[test]
    fn test_similarity_symmetry() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = tiny_model(&corpus);
        model.train_from_file(&corpus).expect("train");
        for a in 0u32..=3 {
            for b in (a + 1)..=5 {
                if let (Some(ab), Some(ba)) = (model.similarity(a, b), model.similarity(b, a)) {
                    assert!((ab - ba).abs() < 1e-5, "not symmetric ({a},{b}): {ab} vs {ba}");
                }
            }
        }
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_similarity_self_is_one() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = tiny_model(&corpus);
        model.train_from_file(&corpus).expect("train");
        for id in 0u32..=5 {
            if let Some(sim) = model.similarity(id, id) {
                assert!((sim - 1.0).abs() < 1e-5, "word {id}: got {sim}");
            }
        }
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_similarity_oov_returns_none() {
        let corpus = make_corpus(&tiny_sentences());
        let model = tiny_model(&corpus);
        assert!(model.similarity(0, 9999).is_none());
        assert!(model.similarity(9999, 0).is_none());
        let _ = std::fs::remove_file(&corpus);
    }

    // ── get_vector ────────────────────────────────────────────────────────────

    #[test]
    fn test_get_vector_in_vocab() {
        let corpus = make_corpus(&tiny_sentences());
        let model = tiny_model(&corpus);
        let v = model.get_vector(0).expect("word 0 must be in vocab");
        assert_eq!(v.len(), 4);
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_get_vector_oov_returns_none() {
        let corpus = make_corpus(&tiny_sentences());
        let model = tiny_model(&corpus);
        assert!(model.get_vector(9999).is_none());
        let _ = std::fs::remove_file(&corpus);
    }

    // ── most_similar ──────────────────────────────────────────────────────────

    #[test]
    fn test_most_similar_excludes_self() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = tiny_model(&corpus);
        model.train_from_file(&corpus).expect("train");
        let results = model.most_similar(0, 3);
        assert!(!results.iter().any(|&(id, _)| id == 0));
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_most_similar_bounded_by_k() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = tiny_model(&corpus);
        model.train_from_file(&corpus).expect("train");
        for k in [1usize, 3, 5] {
            assert!(model.most_similar(0, k).len() <= k);
        }
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_most_similar_sorted_descending() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = tiny_model(&corpus);
        model.train_from_file(&corpus).expect("train");
        let results = model.most_similar(0, 5);
        for pair in results.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_most_similar_oov_returns_empty() {
        let corpus = make_corpus(&tiny_sentences());
        let model = tiny_model(&corpus);
        assert!(model.most_similar(9999, 5).is_empty());
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_most_similar_k_zero_returns_empty() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = tiny_model(&corpus);
        model.train_from_file(&corpus).expect("train");
        assert!(model.most_similar(0, 0).is_empty());
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_most_similar_by_vec_wrong_dim_returns_empty() {
        let corpus = make_corpus(&tiny_sentences());
        let model = tiny_model(&corpus);
        let bad = vec![1.0f32; 999];
        assert!(model.most_similar_by_vec(&bad, 3, None).is_empty());
        let _ = std::fs::remove_file(&corpus);
    }

    // ── analogy ───────────────────────────────────────────────────────────────

    #[test]
    fn test_analogy_returns_finite() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = tiny_model(&corpus);
        model.train_from_file(&corpus).expect("train");
        for &(_, sim) in &model.analogy(0, 1, 2, 3) {
            assert!(sim.is_finite(), "got {sim}");
        }
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_analogy_excludes_inputs() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = tiny_model(&corpus);
        model.train_from_file(&corpus).expect("train");
        let (a, b, c) = (0u32, 1u32, 2u32);
        for &(id, _) in &model.analogy(a, b, c, 5) {
            assert!(id != a && id != b && id != c, "must exclude inputs, got {id}");
        }
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_analogy_oov_returns_empty() {
        let corpus = make_corpus(&tiny_sentences());
        let model = tiny_model(&corpus);
        assert!(model.analogy(9999, 0, 1, 3).is_empty());
        assert!(model.analogy(0, 9999, 1, 3).is_empty());
        assert!(model.analogy(0, 1, 9999, 3).is_empty());
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_analogy_sorted_descending() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = tiny_model(&corpus);
        model.train_from_file(&corpus).expect("train");
        let results = model.analogy(0, 1, 2, 4);
        for pair in results.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }
        let _ = std::fs::remove_file(&corpus);
    }

    // ── surface wrappers ──────────────────────────────────────────────────────

    #[test]
    fn test_surface_similarity_no_map_returns_none() {
        let corpus = make_corpus(&tiny_sentences());
        let model = tiny_model(&corpus);
        assert!(model.similarity_by_surface("w0", "w1").is_none());
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_surface_similarity_with_map() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = tiny_model(&corpus);
        model.train_from_file(&corpus).expect("train");
        let sm: HashMap<u32, String> = (0u32..=5).map(|i| (i, format!("w{i}"))).collect();
        model.set_surface_map(sm);
        let sim = model.similarity_by_surface("w0", "w1");
        assert!(sim.is_some());
        assert!(sim.unwrap().is_finite());
        let _ = std::fs::remove_file(&corpus);
    }

    #[test]
    fn test_most_similar_by_surface_unknown_returns_empty() {
        let corpus = make_corpus(&tiny_sentences());
        let model = tiny_model(&corpus);
        assert!(model.most_similar_by_surface("unknown", 3).is_empty());
        let _ = std::fs::remove_file(&corpus);
    }

    // ── load_text round-trip ──────────────────────────────────────────────────

    #[test]
    fn test_load_text_round_trip() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = Word2VecBuilder::new()
            .vector_size(6)
            .window_size(2)
            .negative_samples(2)
            .min_count(1)
            .sample(0.0)
            .epochs(3)
            .threads(1)
            .build_from_corpus(&corpus)
            .expect("build");
        model.train_from_file(&corpus).expect("train");

        let out = temp_path("rt", "txt");
        model.save_text(&out).expect("save");

        let loaded = Word2Vec::load_text(&out).expect("load");

        assert_eq!(loaded.config().vector_size, model.config().vector_size);
        assert_eq!(loaded.vocab().len(), model.vocab().len());
        assert_eq!(loaded.syn0.len(), model.syn0.len());

        for id in 0u32..=5 {
            let orig = match model.get_vector(id) {
                Some(v) => v.to_vec(),
                None => continue,
            };
            let reloaded = loaded.get_vector(id).expect("must be loadable");
            for (i, (&o, &r)) in orig.iter().zip(reloaded.iter()).enumerate() {
                assert!((o - r).abs() < 1e-5, "syn0[{id}][{i}]: {o} vs {r}");
            }
        }

        let neighbors = loaded.most_similar(0, 3);
        assert!(!neighbors.is_empty());
        for &(_, s) in &neighbors {
            assert!(s.is_finite());
        }

        let _ = std::fs::remove_file(&corpus);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn test_load_text_empty_file_errors() {
        let path = temp_path("empty", "txt");
        std::fs::write(&path, "").expect("write");
        assert!(Word2Vec::load_text(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_text_bad_header_errors() {
        let path = temp_path("badheader", "txt");
        std::fs::write(&path, "NaN NaN\n").expect("write");
        assert!(Word2Vec::load_text(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_text_vector_mismatch_errors() {
        let path = temp_path("vecmismatch", "txt");
        // Header says vector_size=4 but row has 2 floats
        std::fs::write(&path, "1 4\n0 0.1 0.2\n").expect("write");
        assert!(Word2Vec::load_text(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_loaded_model_supports_analogy() {
        let corpus = make_corpus(&tiny_sentences());
        let mut model = Word2VecBuilder::new()
            .vector_size(4)
            .min_count(1)
            .sample(0.0)
            .epochs(2)
            .threads(1)
            .build_from_corpus(&corpus)
            .expect("build");
        model.train_from_file(&corpus).expect("train");

        let path = temp_path("analogy", "txt");
        model.save_text(&path).expect("save");
        let loaded = Word2Vec::load_text(&path).expect("load");

        assert!(!loaded.most_similar(0, 2).is_empty());
        for &(_, s) in &loaded.analogy(0, 1, 2, 2) {
            assert!(s.is_finite());
        }

        let _ = std::fs::remove_file(&corpus);
        let _ = std::fs::remove_file(&path);
    }
}
