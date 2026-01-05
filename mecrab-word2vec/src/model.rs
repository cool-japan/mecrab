//! Word2Vec model structure and builder

use crate::trainer::Trainer;
use crate::vocab::Vocabulary;
use crate::{Result, Word2VecError};
use std::path::Path;
use std::sync::Arc;

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

        use rand::Rng;
        let mut rng = rand::rng();

        // Initialize all vectors (remapped_ids are dense 0..vocab_size-1)
        for remapped_id in 0..vocab_size {
            let offset = remapped_id * vector_size;
            for i in 0..vector_size {
                syn0[offset + i] = (rng.random::<f32>() - 0.5) / vector_size as f32;
            }
        }

        // syn1neg initialized to zeros (common practice)

        eprintln!("Model initialized:");
        eprintln!("  Vocab size (trained): {}", vocab_size);
        eprintln!("  Max word_id (MeCab): {}", max_word_id);
        eprintln!(
            "  Array size: {} elements ({} MB)",
            array_size,
            array_size * 4 / 1024 / 1024
        );
        eprintln!("  Indexing: DENSE (remapped IDs 0-{})", vocab_size - 1);

        Self {
            config,
            vocab: Arc::new(vocab),
            syn0,
            syn1neg,
        }
    }

    /// Train model from corpus file
    pub fn train_from_file<P: AsRef<Path>>(&mut self, corpus_path: P) -> Result<()> {
        let mut trainer = Trainer::new(corpus_path.as_ref(), self.vocab.clone(), &self.config);

        trainer.train(&mut self.syn0, &mut self.syn1neg)?;
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

    /// Get vocabulary
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Get configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
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
