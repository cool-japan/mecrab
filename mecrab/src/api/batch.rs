//! Batch processing API for [`crate::MeCrab`].
//!
//! Provides parallel and sequential helpers for processing multiple texts in
//! one call.  When the `parallel` Cargo feature is enabled, all `*_batch`
//! methods use Rayon to fan out across all available CPU cores.

use crate::{AnalysisResult, MeCrab, Result};

impl MeCrab {
    /// Parse multiple texts in parallel using Rayon.
    ///
    /// When the `parallel` feature is enabled this fans out across all
    /// available CPU cores, providing significant speedup for large workloads.
    ///
    /// # Errors
    ///
    /// Returns a vector of results, where each element may be an `Err`.
    #[cfg(feature = "parallel")]
    pub fn parse_batch(&self, texts: &[&str]) -> Vec<Result<AnalysisResult>> {
        use rayon::prelude::*;
        texts.par_iter().map(|text| self.parse(text)).collect()
    }

    /// Parse multiple texts sequentially (fallback when `parallel` feature is disabled).
    ///
    /// # Errors
    ///
    /// Returns a vector of results, where each element may be an `Err`.
    #[cfg(not(feature = "parallel"))]
    pub fn parse_batch(&self, texts: &[&str]) -> Vec<Result<AnalysisResult>> {
        texts.iter().map(|text| self.parse(text)).collect()
    }

    /// Parse multiple texts and return wakati outputs in parallel.
    ///
    /// # Errors
    ///
    /// Returns a vector of results.
    #[cfg(feature = "parallel")]
    pub fn wakati_batch(&self, texts: &[&str]) -> Vec<Result<String>> {
        use rayon::prelude::*;
        texts.par_iter().map(|text| self.wakati(text)).collect()
    }

    /// Parse multiple texts and return wakati outputs sequentially.
    ///
    /// # Errors
    ///
    /// Returns a vector of results.
    #[cfg(not(feature = "parallel"))]
    pub fn wakati_batch(&self, texts: &[&str]) -> Vec<Result<String>> {
        texts.iter().map(|text| self.wakati(text)).collect()
    }

    /// Parse multiple texts in parallel with a progress callback.
    ///
    /// `callback(processed, total)` is called after each text is parsed.
    /// This is useful for monitoring long-running batch jobs.
    ///
    /// # Errors
    ///
    /// Returns errors inline: each element in the result vector may be `Err`.
    #[cfg(feature = "parallel")]
    pub fn parse_batch_with_progress<F>(
        &self,
        texts: &[&str],
        callback: F,
    ) -> Vec<Result<AnalysisResult>>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let total = texts.len();
        let counter = AtomicUsize::new(0);

        texts
            .par_iter()
            .map(|text| {
                let result = self.parse(text);
                let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                callback(done, total);
                result
            })
            .collect()
    }

    /// Wakati tokenization of multiple texts in parallel with a progress callback.
    ///
    /// `callback(processed, total)` is called after each text is processed.
    ///
    /// # Errors
    ///
    /// Returns errors inline in the result vector.
    #[cfg(feature = "parallel")]
    pub fn wakati_batch_with_progress<F>(&self, texts: &[&str], callback: F) -> Vec<Result<String>>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let total = texts.len();
        let counter = AtomicUsize::new(0);

        texts
            .par_iter()
            .map(|text| {
                let result = self.wakati(text);
                let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                callback(done, total);
                result
            })
            .collect()
    }

    /// Parse multiple texts and return N-best results for each (parallel variant).
    ///
    /// Each element in the result vector holds up to `n` alternative analyses
    /// for the corresponding input text, ordered by ascending total path cost.
    ///
    /// # Errors
    ///
    /// Returns errors inline in the result vector.
    #[cfg(feature = "parallel")]
    pub fn parse_nbest_batch(
        &self,
        texts: &[&str],
        n: usize,
    ) -> Vec<Result<Vec<(AnalysisResult, i64)>>> {
        use rayon::prelude::*;
        texts
            .par_iter()
            .map(|text| self.parse_nbest(text, n))
            .collect()
    }

    /// Parse multiple texts and return N-best results for each (sequential variant).
    ///
    /// # Errors
    ///
    /// Returns errors inline in the result vector.
    #[cfg(not(feature = "parallel"))]
    pub fn parse_nbest_batch(
        &self,
        texts: &[&str],
        n: usize,
    ) -> Vec<Result<Vec<(AnalysisResult, i64)>>> {
        texts.iter().map(|text| self.parse_nbest(text, n)).collect()
    }
}
