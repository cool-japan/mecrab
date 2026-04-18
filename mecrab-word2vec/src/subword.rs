//! FastText-style character n-gram extractor for subword embeddings.
//!
//! Implements the algorithm from Bojanowski et al. 2017 "Enriching Word Vectors
//! with Subword Information".

use std::collections::HashSet;

/// Extracts character n-grams from word surfaces and hashes them to bucket IDs.
pub struct CharNgramExtractor {
    min_n: usize,
    max_n: usize,
    bucket_count: usize,
}

impl CharNgramExtractor {
    /// Create a new extractor with given n-gram length range and bucket count.
    pub fn new(min_n: usize, max_n: usize, bucket_count: usize) -> Self {
        Self {
            min_n,
            max_n,
            bucket_count,
        }
    }

    /// Extract all n-gram bucket IDs for a surface form.
    ///
    /// The surface is wrapped with `<` and `>` boundary markers before extraction.
    /// Returns deduplicated, sorted bucket IDs.
    pub fn extract_bucket_ids(&self, surface: &str) -> Vec<u32> {
        if surface.is_empty() || self.min_n == 0 || self.bucket_count == 0 {
            return Vec::new();
        }

        // Wrap surface with boundary markers and collect into char vec
        let bounded: Vec<char> = std::iter::once('<')
            .chain(surface.chars())
            .chain(std::iter::once('>'))
            .collect();

        let n_chars = bounded.len();
        let mut seen: HashSet<u32> = HashSet::new();

        // Iterate over all start positions
        for start in 0..n_chars {
            // For each valid ngram length
            for length in self.min_n..=self.max_n {
                let end = start + length;
                if end > n_chars {
                    break;
                }

                // Build ngram string from char slice
                let ngram: String = bounded[start..end].iter().collect();
                let bucket_id = self.hash_ngram(&ngram);
                seen.insert(bucket_id);
            }
        }

        let mut ids: Vec<u32> = seen.into_iter().collect();
        ids.sort_unstable();
        ids
    }

    /// FNV-1a hash of an n-gram string, reduced to `[0, bucket_count)`.
    ///
    /// Uses FNV-1a: offset_basis = 14695981039346656037u64,
    /// prime = 1099511628211u64.
    fn hash_ngram(&self, ngram: &str) -> u32 {
        const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
        const PRIME: u64 = 1_099_511_628_211;

        let mut hash: u64 = OFFSET_BASIS;
        for byte in ngram.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(PRIME);
        }

        (hash % self.bucket_count as u64) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_ngram_deterministic() {
        let ext = CharNgramExtractor::new(2, 3, 1_000_000);
        let h1 = ext.hash_ngram("hello");
        let h2 = ext.hash_ngram("hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_extract_bucket_ids_basic() {
        let ext = CharNgramExtractor::new(2, 3, 100);
        let ids = ext.extract_bucket_ids("東京");
        // Should have multiple n-gram bucket IDs, all < 100
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|&id| id < 100));
        // Should be sorted and deduplicated
        assert!(ids.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_extract_empty_surface() {
        let ext = CharNgramExtractor::new(2, 3, 100);
        let ids = ext.extract_bucket_ids("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_min_n_greater_than_word_length() {
        // "ab" with markers "<ab>" = 4 chars; min_n=5 → no valid ngram
        let ext = CharNgramExtractor::new(5, 6, 100);
        let ids = ext.extract_bucket_ids("ab");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_bucket_count_one() {
        // All n-grams must hash to bucket 0 when count=1
        let ext = CharNgramExtractor::new(2, 3, 1);
        let ids = ext.extract_bucket_ids("test");
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|&id| id == 0));
    }
}
