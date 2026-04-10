//! Lazy iterator API for [`crate::MeCrab`].
//!
//! Provides memory-efficient alternatives to the batch methods for cases where
//! the entire result set does not need to be held in memory simultaneously.

use crate::{AnalysisResult, MeCrab, Result};

impl MeCrab {
    /// Parse texts lazily using an iterator.
    ///
    /// Unlike [`MeCrab::parse_batch`], this does **not** pre-collect all
    /// results in memory.  Useful for streaming large text sets where only
    /// one result at a time needs to be held in memory.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for result in mecrab.parse_iter(&texts) {
    ///     match result {
    ///         Ok(analysis) => process(analysis),
    ///         Err(e) => eprintln!("Error: {e}"),
    ///     }
    /// }
    /// ```
    pub fn parse_iter<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> impl Iterator<Item = Result<AnalysisResult>> + 'a {
        texts.iter().map(move |text| self.parse(text))
    }

    /// Wakati tokenization of texts lazily using an iterator.
    ///
    /// Memory-efficient alternative to [`MeCrab::wakati_batch`] for large
    /// text sets.
    pub fn wakati_iter<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> impl Iterator<Item = Result<String>> + 'a {
        texts.iter().map(move |text| self.wakati(text))
    }
}
