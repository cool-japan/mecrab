//! Skip-gram training with negative sampling

use rand::{Rng, RngExt};

/// Vose's alias method table for O(N)-memory O(1)-sample negative sampling.
///
/// Replaces the 100 M-entry CDF table (400 MB) with a compact structure that
/// uses only `3 * N * 8` bytes for a vocabulary of size N.
struct AliasTable {
    /// prob[i] = probability (in [0, 1]) that bucket i returns its own word_id.
    prob: Vec<f64>,
    /// alias[i] = the "other" word_id used when the alias path is taken.
    alias: Vec<u32>,
    /// word_ids[i] = primary word_id for bucket i.
    word_ids: Vec<u32>,
    /// Number of buckets (= vocabulary size passed to the builder).
    n: usize,
}

impl AliasTable {
    /// Build a Vose alias table from a list of `(word_id, count)` pairs.
    ///
    /// The distribution used is `count^0.75` (same as the original word2vec
    /// negative-sampling distribution).
    fn build(word_counts: &[(u32, u64)]) -> Self {
        let n = word_counts.len();

        if n == 0 {
            return Self {
                prob: Vec::new(),
                alias: Vec::new(),
                word_ids: Vec::new(),
                n: 0,
            };
        }

        // ── Step 1: compute smoothed probabilities (raised to 0.75) ──────────
        const POWER: f64 = 0.75;

        let raw: Vec<(u32, f64)> = word_counts
            .iter()
            .map(|&(word_id, count)| (word_id, (count as f64).powf(POWER)))
            .collect();

        let total: f64 = raw.iter().map(|(_, p)| p).sum();

        // ── Step 2: populate word_ids and compute scaled probabilities in [0, n] ──
        let word_ids: Vec<u32> = raw.iter().map(|(id, _)| *id).collect();
        // scaled[i] = (raw_prob[i] / total) * n
        let mut scaled: Vec<f64> = raw
            .iter()
            .map(|(_, p)| (p / total) * (n as f64))
            .collect();

        let mut prob = vec![0.0f64; n];
        let mut alias = vec![0u32; n];

        // ── Step 3: Vose's construction using two worklists ──────────────────
        // small: indices where scaled[i] < 1
        // large: indices where scaled[i] >= 1
        let mut small: Vec<usize> = Vec::with_capacity(n);
        let mut large: Vec<usize> = Vec::with_capacity(n);

        for (i, &s) in scaled.iter().enumerate() {
            if s < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        while !small.is_empty() && !large.is_empty() {
            let s = small.pop().expect("small non-empty");
            let l = large.pop().expect("large non-empty");

            prob[s] = scaled[s];
            alias[s] = word_ids[l];

            // l donates the remainder to fill s's bucket
            scaled[l] -= 1.0 - scaled[s];

            if scaled[l] < 1.0 {
                small.push(l);
            } else {
                large.push(l);
            }
        }

        // Any survivors in large or small had (scaled ≈ 1.0) due to float rounding
        for i in large {
            prob[i] = 1.0;
        }
        for i in small {
            prob[i] = 1.0;
        }

        // word_ids is moved into the struct; alias holds secondary ids already
        Self {
            prob,
            alias,
            word_ids,
            n,
        }
    }

    /// Sample one word_id in O(1) using the alias table.
    #[inline]
    fn sample(&self, rng: &mut impl Rng) -> u32 {
        let i = rng.random_range(0..self.n);
        let p: f64 = rng.random();
        if p < self.prob[i] {
            self.word_ids[i]
        } else {
            self.alias[i]
        }
    }
}

/// Skip-gram trainer with negative sampling
pub struct SkipGram {
    /// Alias table for O(N)-memory negative sampling.
    alias_table: AliasTable,
}

impl SkipGram {
    /// Create new skip-gram trainer
    pub fn new() -> Self {
        Self {
            alias_table: AliasTable {
                prob: Vec::new(),
                alias: Vec::new(),
                word_ids: Vec::new(),
                n: 0,
            },
        }
    }

    /// Build the alias table using word frequencies.
    ///
    /// Uses the unigram distribution raised to the power of 0.75 (standard
    /// word2vec negative-sampling distribution).  The previous implementation
    /// allocated a 100 M-entry CDF table (≈ 400 MB); the alias method uses
    /// only `3 × vocab_size × 8` bytes.
    ///
    /// The function signature is intentionally identical to the old
    /// `build_neg_table` so that `trainer.rs` does not need changes.
    pub fn build_neg_table(&mut self, word_counts: &[(u32, u64)]) {
        self.alias_table = AliasTable::build(word_counts);
        eprintln!(
            "Alias-method negative sampling table built: {} buckets",
            self.alias_table.n
        );
    }

    /// Sample a negative word_id in O(1) using Vose's alias method.
    #[inline]
    pub fn sample_negative(&self, rng: &mut impl Rng) -> u32 {
        self.alias_table.sample(rng)
    }

    /// Train one word pair (center word and context word)
    /// Returns the loss for this pair
    ///
    /// NOTE: This function is no longer used in production code (algorithm is inlined in trainer.rs)
    /// Kept for reference and potential future use
    #[inline]
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn train_pair(
        &self,
        center_id: u32,
        context_id: u32,
        negative_samples: usize,
        alpha: f32,
        syn0: &mut [f32],
        syn1neg: &mut [f32],
        vector_size: usize,
        _vocab_size: usize,
        rng: &mut impl Rng,
    ) -> f32 {
        let mut loss = 0.0f32;

        // Get center word vector
        let l1 = center_id as usize * vector_size;
        if l1 + vector_size > syn0.len() {
            return loss;
        }

        let mut neu1e = vec![0.0f32; vector_size];

        // Positive sample (actual context word)
        let label = 1.0f32;
        let l2 = context_id as usize * vector_size;

        if l2 + vector_size <= syn1neg.len() {
            let f = dot_product(&syn0[l1..l1 + vector_size], &syn1neg[l2..l2 + vector_size]);
            let sigmoid_f = sigmoid(f);
            let g = (label - sigmoid_f) * alpha;
            const LOSS_EPS: f32 = 1e-7;
            loss += if label > 0.5 {
                -(sigmoid_f.max(LOSS_EPS)).ln()
            } else {
                -((1.0 - sigmoid_f).max(LOSS_EPS)).ln()
            };

            // Update gradients
            for i in 0..vector_size {
                neu1e[i] += g * syn1neg[l2 + i];
                syn1neg[l2 + i] += g * syn0[l1 + i];
            }
        }

        // Negative samples
        for _ in 0..negative_samples {
            let neg_id = self.sample_negative(rng);

            // Skip if negative sample is same as context (unlikely but possible)
            if neg_id == context_id {
                continue;
            }

            let label = 0.0f32;
            let l2 = neg_id as usize * vector_size;

            if l2 + vector_size > syn1neg.len() {
                continue;
            }

            let f = dot_product(&syn0[l1..l1 + vector_size], &syn1neg[l2..l2 + vector_size]);
            let sigmoid_f_neg = sigmoid(f);
            let g = (label - sigmoid_f_neg) * alpha;
            const LOSS_EPS_NEG: f32 = 1e-7;
            loss += if label > 0.5 {
                -(sigmoid_f_neg.max(LOSS_EPS_NEG)).ln()
            } else {
                -((1.0 - sigmoid_f_neg).max(LOSS_EPS_NEG)).ln()
            };

            // Update gradients
            for i in 0..vector_size {
                neu1e[i] += g * syn1neg[l2 + i];
                syn1neg[l2 + i] += g * syn0[l1 + i];
            }
        }

        // Update center word vector
        for i in 0..vector_size {
            syn0[l1 + i] += neu1e[i];
        }

        loss
    }
}

impl Default for SkipGram {
    fn default() -> Self {
        Self::new()
    }
}

/// Sigmoid function
#[inline]
#[allow(dead_code)]
fn sigmoid(x: f32) -> f32 {
    if x > 6.0 {
        1.0
    } else if x < -6.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Dot product of two vectors
#[inline]
#[allow(dead_code)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_alias_table_samples_in_range() {
        // Build a small alias table from 3 words with weights [3, 1, 2]
        let mut sg = SkipGram::new();
        let counts = vec![(0u32, 3u64), (1u32, 1u64), (2u32, 2u64)];
        sg.build_neg_table(&counts);

        let mut rng = rand::rng();
        for _ in 0..1000 {
            let sample = sg.sample_negative(&mut rng);
            assert!(sample <= 2, "sample {} out of range", sample);
        }
    }

    #[test]
    fn test_alias_table_distribution_roughly_correct() {
        // Word 0 has weight 3x word 1, so word 0 should appear ~3x as often
        // (after the 0.75 power: 3000^0.75 vs 1000^0.75 ≈ 2.83:1 ratio)
        let mut sg = SkipGram::new();
        let counts = vec![(0u32, 3000u64), (1u32, 1000u64)];
        sg.build_neg_table(&counts);

        let mut rng = rand::rng();
        let mut count_0 = 0usize;
        let mut count_1 = 0usize;
        for _ in 0..10_000 {
            let s = sg.sample_negative(&mut rng);
            if s == 0 {
                count_0 += 1;
            } else {
                count_1 += 1;
            }
        }
        // After 0.75 power the ratio is ~2.83:1; allow ±20% slack → (2.0, 4.5)
        let ratio = count_0 as f64 / count_1 as f64;
        assert!(ratio > 2.0 && ratio < 4.5, "ratio = {}", ratio);
    }

    #[test]
    fn test_alias_table_empty_input() {
        let mut sg = SkipGram::new();
        sg.build_neg_table(&[]);
        // No panic; table is empty (n = 0)
        assert_eq!(sg.alias_table.n, 0);
    }
}
