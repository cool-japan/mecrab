//! Multi-threaded word2vec trainer with Hogwild! algorithm

use crate::Result;
use crate::model::TrainingConfig;
use crate::skipgram::SkipGram;
use crate::vocab::Vocabulary;
use rand::Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Trainer for Word2Vec model
pub struct Trainer {
    corpus_path: PathBuf,
    vocab: Arc<Vocabulary>,
    config: TrainingConfig,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(corpus_path: &Path, vocab: Arc<Vocabulary>, config: &TrainingConfig) -> Self {
        Self {
            corpus_path: corpus_path.to_path_buf(),
            vocab,
            config: config.clone(),
        }
    }

    /// Train the model using Hogwild! algorithm
    ///
    /// Hogwild! is a lock-free parallel SGD algorithm where multiple threads
    /// update shared parameters without locks. Small race conditions are acceptable
    /// and don't affect convergence in practice.
    ///
    /// Reference: "Hogwild!: A Lock-Free Approach to Parallelizing SGD" (NIPS 2011)
    pub fn train(&mut self, syn0: &mut [f32], syn1neg: &mut [f32]) -> Result<()> {
        let vocab_size = self.vocab.len();
        let array_size = syn0.len(); // Actual array size: (max_word_id + 1) * vector_size

        eprintln!("\nStarting training using file {:?}", self.corpus_path);
        eprintln!("Vocab size: {}", vocab_size);
        eprintln!("Words in train file: {}", self.vocab.total_words());
        eprintln!("Parallelization: Hogwild! (lock-free)");

        // Build negative sampling table (using remapped_ids for cache efficiency)
        let mut skipgram = SkipGram::new();
        let word_counts: Vec<(u32, u64)> = self
            .vocab
            .iter()
            .map(|info| (info.remapped_id, info.count))
            .collect();
        skipgram.build_neg_table(&word_counts);
        let skipgram = Arc::new(skipgram);

        // Progress tracking - total across ALL epochs
        let words_processed = Arc::new(AtomicU64::new(0));
        let words_per_epoch = self.vocab.total_words();
        let total_words_all_epochs = words_per_epoch * self.config.epochs as u64;

        // Get raw pointers for Hogwild! updates
        // SAFETY: We ensure memory is valid for the entire training duration
        // Store as usize to make it Send (raw pointers are not Send)
        let syn0_addr = syn0.as_mut_ptr() as usize;
        let syn1neg_addr = syn1neg.as_mut_ptr() as usize;
        let vector_size = self.config.vector_size;

        // Load corpus once into memory (reuse across all epochs)
        eprintln!("Loading corpus into memory...");
        let sentences = self.load_corpus()?;
        let total_sentences = sentences.len();
        eprintln!("Loaded {} sentences", total_sentences);

        // Train for multiple epochs
        for epoch in 0..self.config.epochs {
            eprintln!("\nEpoch {}/{}", epoch + 1, self.config.epochs);

            // Calculate current alpha at start of this epoch
            let epoch_start_words = epoch as u64 * words_per_epoch;
            let current_alpha = self.config.alpha
                - (self.config.alpha - self.config.min_alpha)
                    * (epoch_start_words as f32 / total_words_all_epochs as f32);

            eprintln!("  Starting alpha: {:.6}", current_alpha);
            eprintln!("  Processing {} sentences...", total_sentences);

            // Process sentences in parallel (Hogwild!)
            let chunk_size = (total_sentences / self.config.threads).max(1);
            let sentence_chunks: Vec<_> = sentences.chunks(chunk_size).collect();

            // SAFETY: Hogwild! algorithm
            // Multiple threads write to syn0/syn1neg concurrently without locks.
            // Race conditions create minor noise but don't affect convergence.
            // This is the standard Word2Vec parallelization approach.
            sentence_chunks.into_par_iter().for_each(|chunk| {
                let mut rng = rand::rng();

                // SAFETY: Reconstruct pointers from addresses in each thread
                // The original memory is guaranteed to be valid for training duration
                let syn0_ptr = syn0_addr as *mut f32;
                let syn1neg_ptr = syn1neg_addr as *mut f32;

                // Thread-local counter to reduce atomic operation frequency
                let mut local_word_count = 0u64;

                for sentence in chunk {
                    // Skip empty sentences
                    if sentence.is_empty() {
                        continue;
                    }

                    local_word_count += sentence.len() as u64;

                    // Process each word in sentence
                    for (pos, &center_id) in sentence.iter().enumerate() {
                        // Skip if not in vocab
                        if !self.vocab.contains(center_id) {
                            continue;
                        }

                        // Subsampling
                        if let Some(info) = self.vocab.get(center_id) {
                            if rng.random::<f32>() > info.sample_prob {
                                continue;
                            }
                        }

                        // Dynamic window size
                        let window = rng.random_range(1..=self.config.window_size);

                        // Train with context words - SAFETY: using Hogwild! algorithm
                        for offset in 1..=window {
                            // Left context
                            if pos >= offset {
                                let context_id = sentence[pos - offset];
                                if self.vocab.contains(context_id) {
                                    unsafe {
                                        self.train_word_pair_hogwild(
                                            center_id,
                                            context_id,
                                            current_alpha,
                                            syn0_ptr,
                                            syn1neg_ptr,
                                            vector_size,
                                            array_size,
                                            &skipgram,
                                            &mut rng,
                                        );
                                    }
                                }
                            }

                            // Right context
                            if pos + offset < sentence.len() {
                                let context_id = sentence[pos + offset];
                                if self.vocab.contains(context_id) {
                                    unsafe {
                                        self.train_word_pair_hogwild(
                                            center_id,
                                            context_id,
                                            current_alpha,
                                            syn0_ptr,
                                            syn1neg_ptr,
                                            vector_size,
                                            array_size,
                                            &skipgram,
                                            &mut rng,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Batch update progress (once per thread chunk instead of per sentence)
                if local_word_count > 0 {
                    let processed = words_processed.fetch_add(local_word_count, Ordering::Relaxed);
                    if processed % 100000 < local_word_count {
                        // Progress across all epochs
                        let progress = (processed as f32 / total_words_all_epochs as f32) * 100.0;
                        // Alpha decreases linearly across all epochs
                        let alpha = self.config.alpha
                            - (self.config.alpha - self.config.min_alpha)
                                * (processed as f32 / total_words_all_epochs as f32);
                        eprint!("\rAlpha: {:.6}  Progress: {:.2}%  ", alpha, progress);
                    }
                }
            });

            eprintln!("\n  Epoch {} complete", epoch + 1);
        }

        eprintln!("\nTraining complete!");
        Ok(())
    }

    /// Train a single word pair using Hogwild! (lock-free)
    ///
    /// Inlined skip-gram with direct pointer arithmetic - NO slice creation overhead.
    /// This enables true lock-free parallelization.
    ///
    /// SAFETY: This function is unsafe because it writes to shared memory
    /// without synchronization. Caller must ensure:
    /// 1. Pointers are valid
    /// 2. Memory is large enough for all word_ids
    /// 3. Concurrent access is acceptable (Hogwild! assumption)
    #[inline]
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    unsafe fn train_word_pair_hogwild(
        &self,
        center_id: u32,
        context_id: u32,
        alpha: f32,
        syn0_ptr: *mut f32,
        syn1neg_ptr: *mut f32,
        vector_size: usize,
        array_size: usize,
        skipgram: &Arc<SkipGram>,
        rng: &mut impl Rng,
    ) -> f32 {
        // SAFETY: All pointer operations are wrapped in unsafe blocks
        // Caller guarantees pointers are valid and memory is large enough
        unsafe {
            let mut loss = 0.0f32;

            // Fast O(1) lookup: word_id → remapped_id
            let center_remapped = match self.vocab.get_remapped_id(center_id) {
                Some(id) => id,
                None => return loss,
            };
            let context_remapped = match self.vocab.get_remapped_id(context_id) {
                Some(id) => id,
                None => return loss,
            };

            // Get center word vector pointer (dense indexing for cache efficiency!)
            let l1 = center_remapped as usize * vector_size;
            if l1 + vector_size > array_size {
                return loss;
            }
            let center_vec = syn0_ptr.add(l1);

            // Gradient accumulator
            let mut neu1e = vec![0.0f32; vector_size];

            // Positive sample (actual context word)
            let label = 1.0f32;
            let l2 = context_remapped as usize * vector_size;

            if l2 + vector_size <= array_size {
                let context_vec = syn1neg_ptr.add(l2);

                // Dot product (direct pointer access)
                let mut f = 0.0f32;
                for i in 0..vector_size {
                    f += *center_vec.add(i) * *context_vec.add(i);
                }

                // Sigmoid function (inlined)
                let sigmoid_f = if f > 6.0 {
                    1.0
                } else if f < -6.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-f).exp())
                };

                let g = (label - sigmoid_f) * alpha;
                loss += if label > 0.5 {
                    -f.ln_1p()
                } else {
                    -(1.0 - f).ln_1p()
                };

                // Update gradients (direct memory writes)
                for i in 0..vector_size {
                    neu1e[i] += g * *context_vec.add(i);
                    *context_vec.add(i) += g * *center_vec.add(i);
                }
            }

            // Negative samples
            for _ in 0..self.config.negative_samples {
                let neg_remapped = skipgram.sample_negative(rng);

                // Skip if negative sample is same as context (compare remapped_ids)
                if neg_remapped == context_remapped {
                    continue;
                }

                let label = 0.0f32;
                let l2 = neg_remapped as usize * vector_size;

                if l2 + vector_size > array_size {
                    continue;
                }

                let neg_vec = syn1neg_ptr.add(l2);

                // Dot product
                let mut f = 0.0f32;
                for i in 0..vector_size {
                    f += *center_vec.add(i) * *neg_vec.add(i);
                }

                // Sigmoid
                let sigmoid_f = if f > 6.0 {
                    1.0
                } else if f < -6.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-f).exp())
                };

                let g = (label - sigmoid_f) * alpha;
                loss += if label > 0.5 {
                    -f.ln_1p()
                } else {
                    -(1.0 - f).ln_1p()
                };

                // Update gradients
                for i in 0..vector_size {
                    neu1e[i] += g * *neg_vec.add(i);
                    *neg_vec.add(i) += g * *center_vec.add(i);
                }
            }

            // Update center word vector
            for i in 0..vector_size {
                *center_vec.add(i) += neu1e[i];
            }

            loss
        }
    }

    /// Load corpus into memory
    fn load_corpus(&self) -> Result<Vec<Vec<u32>>> {
        let file = File::open(&self.corpus_path)?;
        let reader = BufReader::new(file);

        let mut sentences = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let sentence: Vec<u32> = line
                .split_whitespace()
                .filter_map(|token| token.parse::<u32>().ok())
                .collect();

            if !sentence.is_empty() {
                sentences.push(sentence);
            }
        }

        Ok(sentences)
    }
}
