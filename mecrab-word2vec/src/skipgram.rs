//! Skip-gram training with negative sampling

use rand::Rng;

/// Skip-gram trainer with negative sampling
pub struct SkipGram {
    /// Negative sampling table for fast sampling
    neg_table: Vec<u32>,
    /// Table size (100M entries)
    table_size: usize,
}

impl SkipGram {
    /// Create new skip-gram trainer
    pub fn new() -> Self {
        Self {
            neg_table: Vec::new(),
            table_size: 100_000_000,
        }
    }

    /// Build negative sampling table using word frequencies
    /// Uses the unigram distribution raised to the power of 0.75
    pub fn build_neg_table(&mut self, word_counts: &[(u32, u64)]) {
        if word_counts.is_empty() {
            return;
        }

        // Compute power
        let power = 0.75f64;
        let mut train_words_pow = 0.0f64;

        let power_counts: Vec<(u32, f64)> = word_counts
            .iter()
            .map(|&(word_id, count)| {
                let pow_count = (count as f64).powf(power);
                train_words_pow += pow_count;
                (word_id, pow_count)
            })
            .collect();

        // Build table
        self.neg_table = Vec::with_capacity(self.table_size);

        let mut i = 0;
        let mut d1 = power_counts[i].1 / train_words_pow;

        for a in 0..self.table_size {
            self.neg_table.push(power_counts[i].0);

            if (a as f64) / (self.table_size as f64) > d1 {
                i += 1;
                if i >= power_counts.len() {
                    i = power_counts.len() - 1;
                }
                // Recalculate d1
                let sum: f64 = power_counts
                    .iter()
                    .take(i + 1)
                    .map(|(_, count)| count)
                    .sum();
                d1 = sum / train_words_pow;
            }
        }

        eprintln!(
            "Negative sampling table built: {} entries",
            self.neg_table.len()
        );
    }

    /// Sample a negative word_id
    #[inline]
    pub fn sample_negative(&self, rng: &mut impl Rng) -> u32 {
        let idx = rng.random_range(0..self.table_size);
        self.neg_table[idx]
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
            let g = (label - sigmoid(f)) * alpha;
            loss += if label > 0.5 {
                -f.ln_1p()
            } else {
                -(1.0 - f).ln_1p()
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
            let g = (label - sigmoid(f)) * alpha;
            loss += if label > 0.5 {
                -f.ln_1p()
            } else {
                -(1.0 - f).ln_1p()
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
}
