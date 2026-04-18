//! Word2Vec model structure and builder

use crate::subword::CharNgramExtractor;
use crate::trainer::Trainer;
use crate::vocab::Vocabulary;
use crate::{Result, Word2VecError};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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
    /// Minimum word frequency
    pub min_count: u64,
    /// Subsampling threshold
    pub sample: f64,
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
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            vector_size: 100,
            window_size: 5,
            negative_samples: 5,
            min_count: 10,
            sample: 1e-4,
            alpha: 0.025,
            min_alpha: 0.0001,
            epochs: 3,
            threads: 8,
            subword: None,
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
            eprintln!(
                "Subword table: {} buckets ({} MB)",
                sw.bucket_count, mb
            );
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
        let mut trainer =
            Trainer::new(corpus_path.as_ref(), self.vocab.clone(), &self.config);

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

    /// Set minimum word count (default: 10)
    pub fn min_count(mut self, count: u64) -> Self {
        self.config.min_count = count;
        self
    }

    /// Set subsampling threshold (default: 1e-4)
    pub fn sample(mut self, threshold: f64) -> Self {
        self.config.sample = threshold;
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

    /// Build vocabulary from corpus and create model
    pub fn build_from_corpus<P: AsRef<Path>>(self, corpus_path: P) -> Result<Word2Vec> {
        let mut vocab = Vocabulary::new(self.config.min_count, self.config.sample);
        vocab.build_from_file(&corpus_path)?;

        if vocab.is_empty() {
            return Err(Word2VecError::Vocabulary(
                "Vocabulary is empty after filtering".to_string(),
            ));
        }

        Ok(Word2Vec::new(self.config, vocab))
    }
}
