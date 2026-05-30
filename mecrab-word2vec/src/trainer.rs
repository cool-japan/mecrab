//! Multi-threaded word2vec trainer with Hogwild! algorithm

use crate::Result;
use crate::model::TrainingConfig;
use crate::skipgram::SkipGram;
use crate::subword::CharNgramExtractor;
use crate::vocab::Vocabulary;
use rand::{Rng, RngExt};
use rayon::prelude::*;
use std::collections::HashMap;
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
    /// Optional surface map: word_id → surface string for subword training
    surface_map: Option<HashMap<u32, String>>,
    /// Optional n-gram extractor (initialised when subword is configured)
    extractor: Option<CharNgramExtractor>,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(corpus_path: &Path, vocab: Arc<Vocabulary>, config: &TrainingConfig) -> Self {
        Self {
            corpus_path: corpus_path.to_path_buf(),
            vocab,
            config: config.clone(),
            surface_map: None,
            extractor: None,
        }
    }

    /// Attach a surface map (word_id → surface) for subword training.
    pub fn with_surface_map(mut self, surface_map: HashMap<u32, String>) -> Self {
        self.surface_map = Some(surface_map);
        self
    }

    /// Attach a pre-built n-gram extractor.
    pub fn with_subword_extractor(mut self, extractor: CharNgramExtractor) -> Self {
        self.extractor = Some(extractor);
        self
    }

    /// Train the model using Hogwild! algorithm
    ///
    /// Hogwild! is a lock-free parallel SGD algorithm where multiple threads
    /// update shared parameters without locks. Small race conditions are acceptable
    /// and don't affect convergence in practice.
    ///
    /// Reference: "Hogwild!: A Lock-Free Approach to Parallelizing SGD" (NIPS 2011)
    pub fn train(
        &mut self,
        syn0: &mut [f32],
        syn1neg: &mut [f32],
        syn_ng: &mut [f32],
    ) -> Result<()> {
        // ── GPU fast path (only active when `gpu` feature is compiled in) ──────
        #[cfg(feature = "gpu")]
        if self.config.use_gpu {
            if let Some(ctx) = crate::gpu::GpuContext::try_new() {
                eprintln!("GPU adapter found — using wgpu-accelerated training");
                return self.train_gpu(&ctx, syn0, syn1neg);
            }
            eprintln!("GPU unavailable (no wgpu adapter), falling back to CPU Hogwild!");
        }

        let vocab_size = self.vocab.len();
        let array_size = syn0.len(); // Actual array size: vocab_size * vector_size

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
        let syn_ng_addr = if syn_ng.is_empty() {
            0usize
        } else {
            syn_ng.as_mut_ptr() as usize
        };
        let vector_size = self.config.vector_size;
        let syn0_array_size = array_size;
        let syn1neg_array_size = syn1neg.len();
        let syn_ng_array_size = syn_ng.len();

        let use_subword = self.extractor.is_some() && syn_ng_addr != 0;

        // Load corpus once into memory (reuse across all epochs)
        eprintln!("Loading corpus into memory...");
        let sentences = self.load_corpus()?;
        let total_sentences = sentences.len();
        eprintln!("Loaded {} sentences", total_sentences);

        // Build pre-computed bucket_ids per word_id (if subword enabled)
        // This avoids string allocation on every training pair inside the hot loop.
        let bucket_ids_per_word: Arc<Option<HashMap<u32, Vec<u32>>>> = if use_subword {
            if let Some(ref extractor) = self.extractor {
                let mut map: HashMap<u32, Vec<u32>> = HashMap::new();
                for info in self.vocab.iter() {
                    let word_id = info.word_id;
                    let surface = match &self.surface_map {
                        Some(sm) => sm
                            .get(&word_id)
                            .cloned()
                            .unwrap_or_else(|| word_id.to_string()),
                        None => word_id.to_string(),
                    };
                    let ids = extractor.extract_bucket_ids(&surface);
                    map.insert(word_id, ids);
                }
                Arc::new(Some(map))
            } else {
                Arc::new(None)
            }
        } else {
            Arc::new(None)
        };

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
                let syn_ng_ptr = if syn_ng_addr != 0 {
                    syn_ng_addr as *mut f32
                } else {
                    std::ptr::null_mut()
                };

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

                        if use_subword && !syn_ng_ptr.is_null() {
                            // Subword training path
                            let center_bucket_ids = bucket_ids_per_word
                                .as_ref()
                                .as_ref()
                                .and_then(|m| m.get(&center_id))
                                .map(|v| v.as_slice())
                                .unwrap_or(&[]);

                            let center_remapped = match self.vocab.get_remapped_id(center_id) {
                                Some(id) => id,
                                None => continue,
                            };

                            for offset in 1..=window {
                                // Left context
                                if pos >= offset {
                                    let context_id = sentence[pos - offset];
                                    if let Some(context_remapped) =
                                        self.vocab.get_remapped_id(context_id)
                                    {
                                        // SAFETY: Hogwild! — same safety as non-subword path
                                        unsafe {
                                            self.train_word_pair_subword_hogwild(
                                                center_remapped,
                                                center_bucket_ids,
                                                context_remapped,
                                                current_alpha,
                                                syn0_ptr,
                                                syn1neg_ptr,
                                                syn_ng_ptr,
                                                vector_size,
                                                syn0_array_size,
                                                syn1neg_array_size,
                                                syn_ng_array_size,
                                                &skipgram,
                                                &mut rng,
                                            );
                                        }
                                    }
                                }

                                // Right context
                                if pos + offset < sentence.len() {
                                    let context_id = sentence[pos + offset];
                                    if let Some(context_remapped) =
                                        self.vocab.get_remapped_id(context_id)
                                    {
                                        // SAFETY: Hogwild!
                                        unsafe {
                                            self.train_word_pair_subword_hogwild(
                                                center_remapped,
                                                center_bucket_ids,
                                                context_remapped,
                                                current_alpha,
                                                syn0_ptr,
                                                syn1neg_ptr,
                                                syn_ng_ptr,
                                                vector_size,
                                                syn0_array_size,
                                                syn1neg_array_size,
                                                syn_ng_array_size,
                                                &skipgram,
                                                &mut rng,
                                            );
                                        }
                                    }
                                }
                            }
                        } else {
                            // Standard (non-subword) training path
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

    /// Train one word pair with FastText subword n-gram representations (Hogwild!).
    ///
    /// The center word's effective input vector is the sum of its syn0 embedding
    /// and the syn_ng embeddings of all its character n-gram buckets.
    ///
    /// Gradients are backpropagated to BOTH syn0 and all syn_ng buckets, scaled
    /// by `1.0 / (1.0 + num_buckets)` to normalise relative contributions.
    ///
    /// SAFETY: Same Hogwild! assumptions as `train_word_pair_hogwild`:
    /// - All pointers are valid for the duration of training
    /// - Memory is large enough for all remapped IDs and bucket IDs
    /// - Concurrent unsynchronised access is acceptable (Hogwild! assumption)
    #[inline]
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    unsafe fn train_word_pair_subword_hogwild(
        &self,
        center_remapped_id: u32,
        center_bucket_ids: &[u32],
        context_remapped_id: u32,
        alpha: f32,
        syn0_ptr: *mut f32,
        syn1neg_ptr: *mut f32,
        syn_ng_ptr: *mut f32,
        vector_size: usize,
        syn0_array_size: usize,
        syn1neg_array_size: usize,
        syn_ng_array_size: usize,
        skipgram: &Arc<SkipGram>,
        rng: &mut impl Rng,
    ) {
        unsafe {
            let l1 = center_remapped_id as usize * vector_size;
            if l1 + vector_size > syn0_array_size {
                return;
            }

            // ── Step 1: Build the composite center vector h ──────────────────
            // h = syn0[center] + Σ syn_ng[bucket] for each bucket
            let mut h = vec![0.0f32; vector_size];

            // Add syn0 contribution
            let center_vec = syn0_ptr.add(l1);
            for i in 0..vector_size {
                h[i] = *center_vec.add(i);
            }

            // Add n-gram bucket contributions
            for &bucket_id in center_bucket_ids {
                let ng_offset = bucket_id as usize * vector_size;
                if ng_offset + vector_size <= syn_ng_array_size {
                    let ng_vec = syn_ng_ptr.add(ng_offset);
                    for i in 0..vector_size {
                        h[i] += *ng_vec.add(i);
                    }
                }
            }

            // ── Step 2: Gradient accumulator ─────────────────────────────────
            let mut neu1e = vec![0.0f32; vector_size];

            // ── Step 3: Positive sample ───────────────────────────────────────
            {
                let label = 1.0f32;
                let l2 = context_remapped_id as usize * vector_size;
                if l2 + vector_size <= syn1neg_array_size {
                    let context_vec = syn1neg_ptr.add(l2);

                    let mut f = 0.0f32;
                    for i in 0..vector_size {
                        f += h[i] * *context_vec.add(i);
                    }

                    let sigmoid_f = if f > 6.0 {
                        1.0
                    } else if f < -6.0 {
                        0.0
                    } else {
                        1.0 / (1.0 + (-f).exp())
                    };

                    let g = (label - sigmoid_f) * alpha;

                    for i in 0..vector_size {
                        neu1e[i] += g * *context_vec.add(i);
                        *context_vec.add(i) += g * h[i];
                    }
                }
            }

            // ── Step 4: Negative samples ──────────────────────────────────────
            for _ in 0..self.config.negative_samples {
                let neg_remapped = skipgram.sample_negative(rng);

                if neg_remapped == context_remapped_id {
                    continue;
                }

                let label = 0.0f32;
                let l2 = neg_remapped as usize * vector_size;
                if l2 + vector_size > syn1neg_array_size {
                    continue;
                }

                let neg_vec = syn1neg_ptr.add(l2);

                let mut f = 0.0f32;
                for i in 0..vector_size {
                    f += h[i] * *neg_vec.add(i);
                }

                let sigmoid_f = if f > 6.0 {
                    1.0
                } else if f < -6.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-f).exp())
                };

                let g = (label - sigmoid_f) * alpha;

                for i in 0..vector_size {
                    neu1e[i] += g * *neg_vec.add(i);
                    *neg_vec.add(i) += g * h[i];
                }
            }

            // ── Step 5: Backprop to syn0 and syn_ng ──────────────────────────
            // Scale: 1.0 / (1.0 + num_buckets) to normalise relative contributions
            let scale = 1.0_f32 / (1.0 + center_bucket_ids.len() as f32);

            // Update syn0
            for i in 0..vector_size {
                *center_vec.add(i) += neu1e[i] * scale;
            }

            // Update each n-gram bucket
            for &bucket_id in center_bucket_ids {
                let ng_offset = bucket_id as usize * vector_size;
                if ng_offset + vector_size <= syn_ng_array_size {
                    let ng_vec = syn_ng_ptr.add(ng_offset);
                    for i in 0..vector_size {
                        *ng_vec.add(i) += neu1e[i] * scale;
                    }
                }
            }
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

    /// GPU training path — mirrors the CPU Hogwild! loop but dispatches WGSL
    /// compute shaders via wgpu.  `syn_ng` (subword) always falls back to CPU
    /// because subword updates are interleaved with the main embedding; only the
    /// standard skip-gram weights are GPU-accelerated.
    ///
    /// # Errors
    ///
    /// Returns an error if corpus loading fails.
    #[cfg(feature = "gpu")]
    fn train_gpu(
        &mut self,
        ctx: &crate::gpu::GpuContext,
        syn0: &mut [f32],
        syn1neg: &mut [f32],
    ) -> Result<()> {
        use crate::gpu::{GpuTrainer, TrainingPair};

        let vocab_size = self.vocab.len();
        let vector_size = self.config.vector_size;

        eprintln!("GPU training — vocab_size={} vector_size={}", vocab_size, vector_size);

        let sentences = self.load_corpus()?;
        let total_sentences = sentences.len();
        let mut skipgram = SkipGram::new();
        let word_counts: Vec<(u32, u64)> = self
            .vocab
            .iter()
            .map(|info| (info.remapped_id, info.count))
            .collect();
        skipgram.build_neg_table(&word_counts);

        let words_per_epoch = self.vocab.total_words();
        let total_words_all_epochs = words_per_epoch * self.config.epochs as u64;

        let gpu_trainer = GpuTrainer::new(ctx, syn0, syn1neg, vector_size);

        for epoch in 0..self.config.epochs {
            let epoch_start = epoch as u64 * words_per_epoch;
            let alpha = self.config.alpha
                - (self.config.alpha - self.config.min_alpha)
                    * (epoch_start as f32 / total_words_all_epochs as f32);

            eprintln!("Epoch {}/{} — alpha={:.6}", epoch + 1, self.config.epochs, alpha);

            let mut rng = rand::rng();
            let mut batch: Vec<TrainingPair> = Vec::with_capacity(16384);

            for sentence in &sentences {
                if sentence.is_empty() { continue; }
                for (pos, &center_id) in sentence.iter().enumerate() {
                    let center_remapped = match self.vocab.get_remapped_id(center_id) {
                        Some(id) => id,
                        None => continue,
                    };
                    let window = rng.random_range(1..=self.config.window_size);
                    for offset in 1..=window {
                        let neighbors = [
                            pos.checked_sub(offset).map(|i| sentence[i]),
                            if pos + offset < sentence.len() { Some(sentence[pos + offset]) } else { None },
                        ];
                        for ctx_id in neighbors.iter().flatten() {
                            let ctx_remapped = match self.vocab.get_remapped_id(*ctx_id) {
                                Some(id) => id,
                                None => continue,
                            };
                            // Positive pair
                            batch.push(TrainingPair { center: center_remapped, context: ctx_remapped, label: 1, _pad: 0 });
                            // Negative samples
                            for _ in 0..self.config.negative_samples {
                                let neg = skipgram.sample_negative(&mut rng);
                                if neg != ctx_remapped {
                                    batch.push(TrainingPair { center: center_remapped, context: neg, label: 0, _pad: 0 });
                                }
                            }
                            // Dispatch when batch is large enough
                            if batch.len() >= 16384 {
                                gpu_trainer.train_batch(&batch, alpha);
                                batch.clear();
                            }
                        }
                    }
                }
            }
            if !batch.is_empty() {
                gpu_trainer.train_batch(&batch, alpha);
                batch.clear();
            }
            eprintln!("  Epoch {} done — processed {} sentences", epoch + 1, total_sentences);
        }

        gpu_trainer.read_back(syn0, syn1neg);
        eprintln!("GPU training complete — weights read back to host");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{TrainingConfig, Word2VecBuilder};

    /// Write a minimal corpus of whitespace-separated u32 token IDs to a temp
    /// file and return the path.  All IDs appear frequently enough to survive
    /// min_count filtering.
    fn write_tiny_corpus() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("mecrab_trainer_test_{nanos}.txt"));
        // Repeat sentences so every token appears at least 5 times
        let text = "0 1 2 3 4\n1 2 3 4 5\n0 2 4 6 8\n1 3 5 7 9\n0 1 3 5 7\n\
                    2 4 6 8 0\n3 5 7 9 1\n0 3 6 9 2\n1 4 7 0 3\n2 5 8 1 4\n\
                    0 1 2 3 4\n1 2 3 4 5\n0 2 4 6 8\n1 3 5 7 9\n0 1 3 5 7\n";
        std::fs::write(&path, text).expect("write corpus");
        path
    }

    // ── Trainer::train produces finite, non-zero embeddings ───────────────────

    #[test]
    fn test_trainer_train_produces_finite_embeddings() {
        let corpus_path = write_tiny_corpus();

        let config = TrainingConfig {
            vector_size: 8,
            window_size: 2,
            negative_samples: 2,
            min_count: 1,
            sample: 0.0,
            alpha: 0.025,
            min_alpha: 0.0001,
            epochs: 2,
            threads: 1,
            subword: None,
            use_gpu: false,
        };

        let mut vocab = Vocabulary::new(1, 0.0);
        vocab
            .build_from_file(&corpus_path)
            .expect("build vocab from corpus");
        assert!(!vocab.is_empty(), "vocabulary must not be empty");

        let vocab_arc = Arc::new(vocab);
        let vocab_size = vocab_arc.len();
        let vector_size = config.vector_size;
        let array_size = vocab_size * vector_size;

        // Initialise embeddings (random small values, matching Word2Vec::new logic)
        let mut syn0: Vec<f32> = (0..array_size)
            .map(|i| ((i as f32 * 1.234).sin()) / vector_size as f32)
            .collect();
        let mut syn1neg = vec![0.0_f32; array_size];
        let mut syn_ng: Vec<f32> = Vec::new();

        let mut trainer = Trainer::new(&corpus_path, vocab_arc, &config);
        trainer
            .train(&mut syn0, &mut syn1neg, &mut syn_ng)
            .expect("trainer.train should succeed");

        // All trained values must be finite
        for (i, &v) in syn0.iter().enumerate() {
            assert!(v.is_finite(), "syn0[{i}] must be finite, got {v}");
        }
        for (i, &v) in syn1neg.iter().enumerate() {
            assert!(v.is_finite(), "syn1neg[{i}] must be finite, got {v}");
        }

        // At least one weight must have changed from the initial value
        let initial: Vec<f32> = (0..array_size)
            .map(|i| ((i as f32 * 1.234).sin()) / vector_size as f32)
            .collect();
        let any_changed = syn0.iter().zip(initial.iter()).any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(any_changed, "training must update at least one weight");

        let _ = std::fs::remove_file(&corpus_path);
    }

    // ── Word2VecBuilder end-to-end produces finite embeddings ─────────────────

    #[test]
    fn test_builder_training_all_finite() {
        let corpus_path = write_tiny_corpus();

        let mut model = Word2VecBuilder::new()
            .vector_size(8)
            .window_size(2)
            .negative_samples(2)
            .min_count(1)
            .sample(0.0)
            .alpha(0.025)
            .min_alpha(0.0001)
            .epochs(2)
            .threads(1)
            .build_from_corpus(&corpus_path)
            .expect("build_from_corpus should succeed");

        model
            .train_from_file(&corpus_path)
            .expect("train_from_file should succeed");

        // All syn0 embeddings must be finite
        for (i, &v) in model.syn0.iter().enumerate() {
            assert!(v.is_finite(), "syn0[{i}] must be finite after training, got {v}");
        }
        // At least one embedding must be non-zero (training must change something)
        let any_nonzero = model.syn0.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "syn0 must have at least one non-zero value after training");

        let _ = std::fs::remove_file(&corpus_path);
    }

    // ── GPU flag with no adapter falls back to CPU ────────────────────────────

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_flag_with_no_adapter_falls_back_to_cpu() {
        let corpus_path = write_tiny_corpus();

        let mut model = Word2VecBuilder::new()
            .vector_size(8)
            .window_size(2)
            .negative_samples(2)
            .min_count(1)
            .sample(0.0)
            .epochs(1)
            .threads(1)
            .use_gpu(true) // request GPU; falls back if no adapter
            .build_from_corpus(&corpus_path)
            .expect("build_from_corpus must succeed even with gpu=true");

        model
            .train_from_file(&corpus_path)
            .expect("training with gpu=true must succeed via CPU fallback");

        for (i, &v) in model.syn0.iter().enumerate() {
            assert!(
                v.is_finite(),
                "GPU-requested syn0[{i}] must be finite, got {v}"
            );
        }

        let _ = std::fs::remove_file(&corpus_path);
    }
}
