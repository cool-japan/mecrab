//! Python bindings for MeCrab using PyO3
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module provides Python bindings for MeCrab morphological analyzer.
//!
//! # Usage (Python)
//!
//! ```python
//! import mecrab
//!
//! # Create analyzer with default dictionary
//! m = mecrab.MeCrab()
//!
//! # Parse text
//! result = m.parse("すもももももももものうち")
//! print(result)
//!
//! # Parse to dictionary (Pythonic API)
//! morphemes = m.parse_to_dict("東京に行く")
//! for m in morphemes:
//!     print(m['surface'], m['pos'], m.get('ipa'))
//!
//! # Wakati (space-separated)
//! words = m.wakati("すもももももももものうち")
//! print(words)
//!
//! # Add custom words
//! m.add_word("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000)
//!
//! # Batch processing
//! results = m.parse_batch(["テスト1", "テスト2", "テスト3"])
//!
//! # With IPA pronunciation
//! m_ipa = mecrab.MeCrab(with_ipa=True)
//! result = m_ipa.parse_to_dict("こんにちは")
//! # => [{'surface': 'こんにちは', 'pos': '感動詞', 'ipa': '/koɲɲit͡ɕiɰa/', ...}]
//!
//! # With word embeddings
//! m_vec = mecrab.MeCrab(vector_path="vectors.bin")
//! result = m_vec.parse_to_dict("東京")
//! # => [{'surface': '東京', 'embedding': [0.1, -0.2, ...], ...}]
//! ```

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

/// Python wrapper for MeCrab morphological analyzer
#[pyclass(name = "MeCrab")]
pub struct PyMeCrab {
    inner: crate::MeCrab,
    with_ipa: bool,
    with_vector: bool,
}

#[pymethods]
impl PyMeCrab {
    /// Create a new MeCrab instance
    ///
    /// Args:
    ///     dicdir: Optional path to dictionary directory
    ///     with_ipa: Enable IPA pronunciation output (default: False)
    ///
    /// Returns:
    ///     MeCrab instance
    ///
    /// Raises:
    ///     RuntimeError: If dictionary cannot be loaded
    ///
    /// Example:
    ///     >>> # Basic usage
    ///     >>> m = MeCrab()
    ///
    ///     >>> # With IPA pronunciation
    ///     >>> m = MeCrab(with_ipa=True)
    ///     >>> morphemes = m.parse_to_dict("東京に行く")
    ///     >>> print(morphemes[0]['ipa'])  # => '/toːkʲoː/'
    #[new]
    #[pyo3(signature = (dicdir=None, with_ipa=false, vector_path=None))]
    fn new(dicdir: Option<String>, with_ipa: bool, vector_path: Option<String>) -> PyResult<Self> {
        let mut builder = crate::MeCrab::builder();

        if let Some(path) = dicdir {
            builder = builder.dicdir(Some(PathBuf::from(path)));
        }

        if with_ipa {
            builder = builder.with_ipa(true);
        }

        // Configure vector support if path provided
        let with_vector = vector_path.is_some();
        if let Some(ref path) = vector_path {
            builder = builder.vector_pool(Some(PathBuf::from(path)));
            builder = builder.with_vector(true);
        }

        match builder.build() {
            Ok(inner) => Ok(Self {
                inner,
                with_ipa,
                with_vector,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to load MeCrab: {e}"
            ))),
        }
    }

    /// Parse text and return analysis result
    ///
    /// Args:
    ///     text: Input text to analyze
    ///
    /// Returns:
    ///     Analysis result as formatted string
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    fn parse(&self, text: &str) -> PyResult<String> {
        match self.inner.parse(text) {
            Ok(result) => Ok(result.to_string()),
            Err(e) => Err(PyRuntimeError::new_err(format!("Parse error: {e}"))),
        }
    }

    /// Parse text and return wakati (space-separated) output
    ///
    /// Args:
    ///     text: Input text to analyze
    ///
    /// Returns:
    ///     Space-separated surface forms
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    fn wakati(&self, text: &str) -> PyResult<String> {
        match self.inner.wakati(text) {
            Ok(result) => Ok(result),
            Err(e) => Err(PyRuntimeError::new_err(format!("Parse error: {e}"))),
        }
    }

    /// Parse text and return list of morphemes
    ///
    /// Args:
    ///     text: Input text to analyze
    ///
    /// Returns:
    ///     List of (surface, feature) tuples
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    fn parse_to_list(&self, text: &str) -> PyResult<Vec<(String, String)>> {
        match self.inner.parse(text) {
            Ok(result) => Ok(result
                .morphemes
                .iter()
                .map(|m| (m.surface.clone(), m.feature.clone()))
                .collect()),
            Err(e) => Err(PyRuntimeError::new_err(format!("Parse error: {e}"))),
        }
    }

    /// Parse text and return list of dictionaries (Pythonic API)
    ///
    /// Args:
    ///     text: Input text to analyze
    ///
    /// Returns:
    ///     List of dictionaries with morpheme information
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    ///
    /// Example:
    ///     >>> m = MeCrab()
    ///     >>> result = m.parse_to_dict("東京に行く")
    ///     >>> for morph in result:
    ///     ...     print(morph[`"surface"`], morph[`"pos"`])
    #[allow(clippy::doc_link_with_quotes)]
    fn parse_to_dict<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
        match self.inner.parse(text) {
            Ok(result) => {
                let dicts: Vec<Bound<'_, PyDict>> = result
                    .morphemes
                    .iter()
                    .map(|m| {
                        let dict = PyDict::new(py);

                        // Basic fields
                        let _ = dict.set_item("surface", &m.surface);
                        let _ = dict.set_item("feature", &m.feature);

                        // Parse feature string
                        let parts: Vec<&str> = m.feature.split(',').collect();
                        if !parts.is_empty() {
                            let _ = dict.set_item("pos", parts[0]);

                            if parts.len() > 1 {
                                let _ = dict.set_item("pos1", parts[1]);
                            }
                            if parts.len() > 2 {
                                let _ = dict.set_item("pos2", parts[2]);
                            }
                            if parts.len() > 3 {
                                let _ = dict.set_item("pos3", parts[3]);
                            }
                            if parts.len() > 4 && parts[4] != "*" {
                                let _ = dict.set_item("inflection", parts[4]);
                            }
                            if parts.len() > 5 && parts[5] != "*" {
                                let _ = dict.set_item("conjugation", parts[5]);
                            }
                            if parts.len() > 6 && parts[6] != "*" {
                                let _ = dict.set_item("base", parts[6]);
                            }
                            if parts.len() > 7 && parts[7] != "*" {
                                let _ = dict.set_item("reading", parts[7]);
                            }
                            if parts.len() > 8 && parts[8] != "*" {
                                let _ = dict.set_item("pronunciation", parts[8]);
                            }
                        }

                        // Add IPA pronunciation if available
                        if let Some(ref ipa) = m.pronunciation {
                            let _ = dict.set_item("ipa", ipa.as_str());
                        }

                        // Add embedding vector if available
                        if let Some(ref embedding) = m.embedding {
                            let _ = dict.set_item("embedding", embedding.clone());
                        }

                        dict
                    })
                    .collect();

                Ok(dicts)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Parse error: {e}"))),
        }
    }

    /// Parse multiple texts in batch
    ///
    /// When compiled with 'parallel' feature, this uses Rayon for
    /// parallel processing across all available CPU cores.
    ///
    /// Args:
    ///     texts: List of texts to analyze
    ///
    /// Returns:
    ///     List of analysis results as formatted strings
    ///
    /// Raises:
    ///     RuntimeError: If any parsing fails
    fn parse_batch(&self, texts: Vec<String>) -> PyResult<Vec<String>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let results: Result<Vec<String>, _> = self
            .inner
            .parse_batch(&refs)
            .into_iter()
            .map(|r| r.map(|result| result.to_string()))
            .collect();

        results.map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    /// Parse multiple texts and return wakati outputs in batch
    ///
    /// Args:
    ///     texts: List of texts to analyze
    ///
    /// Returns:
    ///     List of space-separated surface forms
    ///
    /// Raises:
    ///     RuntimeError: If any parsing fails
    fn wakati_batch(&self, texts: Vec<String>) -> PyResult<Vec<String>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let results: Result<Vec<String>, _> = self.inner.wakati_batch(&refs).into_iter().collect();

        results.map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    /// Add a word to the overlay dictionary
    ///
    /// This allows adding custom words (new product names, slang, etc.)
    /// that will be recognized during parsing.
    ///
    /// Args:
    ///     surface: The surface form (the actual text)
    ///     reading: The katakana reading
    ///     pronunciation: The pronunciation
    ///     wcost: Word cost (lower = more preferred, typical: 5000-8000)
    fn add_word(&self, surface: &str, reading: &str, pronunciation: &str, wcost: i16) {
        self.inner.add_word(surface, reading, pronunciation, wcost);
    }

    /// Remove a word from the overlay dictionary
    ///
    /// Args:
    ///     surface: The surface form to remove
    ///
    /// Returns:
    ///     True if the word was found and removed
    fn remove_word(&self, surface: &str) -> bool {
        self.inner.remove_word(surface)
    }

    /// Get the number of words in the overlay dictionary
    ///
    /// Returns:
    ///     Number of overlay words
    fn overlay_size(&self) -> usize {
        self.inner.overlay_size()
    }

    /// Convert text to IPA pronunciation (one-shot conversion)
    ///
    /// This is a convenience method that parses the text and returns
    /// just the IPA pronunciations as a list of strings.
    ///
    /// Args:
    ///     text: Input text to convert
    ///
    /// Returns:
    ///     List of IPA pronunciation strings
    ///
    /// Raises:
    ///     RuntimeError: If IPA is not enabled or parsing fails
    ///
    /// Example:
    ///     >>> m = MeCrab(with_ipa=True)
    ///     >>> ipas = m.to_ipa("東京に行く")
    ///     >>> print(ipas)
    ///     [`"toːkʲoː"`, `"ɲi"`, `"ikɯ"`]
    ///
    ///     >>> # Join with spaces
    ///     >>> print(" ".join(ipas))
    ///     `"toːkʲoː ɲi ikɯ"`
    #[allow(clippy::doc_link_with_quotes)]
    fn to_ipa(&self, text: &str) -> PyResult<Vec<String>> {
        if !self.with_ipa {
            return Err(PyRuntimeError::new_err(
                "IPA support not enabled. Create MeCrab with with_ipa=True",
            ));
        }

        match self.inner.parse(text) {
            Ok(result) => {
                let ipas: Vec<String> = result
                    .morphemes
                    .iter()
                    .filter_map(|m| m.pronunciation.clone())
                    .collect();
                Ok(ipas)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Parse error: {e}"))),
        }
    }

    /// Convert text to IPA pronunciation as a single string
    ///
    /// Args:
    ///     text: Input text to convert
    ///     separator: Separator between morphemes (default: " ")
    ///
    /// Returns:
    ///     IPA pronunciation string
    ///
    /// Raises:
    ///     RuntimeError: If IPA is not enabled or parsing fails
    ///
    /// Example:
    ///     >>> m = MeCrab(with_ipa=True)
    ///     >>> ipa_text = m.to_ipa_text("東京に行く")
    ///     >>> print(ipa_text)
    ///     'toːkʲoː ɲi ikɯ'
    ///
    ///     >>> # Custom separator
    ///     >>> print(m.to_ipa_text("東京に行く", separator="-"))
    ///     'toːkʲoː-ɲi-ikɯ'
    #[pyo3(signature = (text, separator=" "))]
    fn to_ipa_text(&self, text: &str, separator: &str) -> PyResult<String> {
        let ipas = self.to_ipa(text)?;
        Ok(ipas.join(separator))
    }

    /// Compute cosine similarity between two words
    ///
    /// Parses both words and computes the cosine similarity between their
    /// embedding vectors. If a word tokenizes into multiple morphemes,
    /// uses the first morpheme's embedding.
    ///
    /// Args:
    ///     word1: First word
    ///     word2: Second word
    ///
    /// Returns:
    ///     Cosine similarity in range [-1.0, 1.0]
    ///
    /// Raises:
    ///     RuntimeError: If vectors not enabled or words not found in vocabulary
    ///
    /// Example:
    ///     >>> m = MeCrab(vector_path="vectors.bin")
    ///     >>> sim = m.similarity("東京", "京都")
    ///     >>> print(f"Similarity: {sim:.3f}")
    ///     Similarity: 0.856
    fn similarity(&self, word1: &str, word2: &str) -> PyResult<f32> {
        if !self.with_vector {
            return Err(PyRuntimeError::new_err(
                "Vector support not enabled. Create MeCrab with vector_path parameter",
            ));
        }

        // Parse both words to get embeddings
        let result1 = self
            .inner
            .parse(word1)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error for word1: {e}")))?;
        let result2 = self
            .inner
            .parse(word2)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error for word2: {e}")))?;

        // Get first morpheme's embedding from each
        let emb1 = result1
            .morphemes
            .first()
            .and_then(|m| m.embedding.as_ref())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "No embedding found for word1: '{}' (may be out-of-vocabulary)",
                    word1
                ))
            })?;

        let emb2 = result2
            .morphemes
            .first()
            .and_then(|m| m.embedding.as_ref())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "No embedding found for word2: '{}' (may be out-of-vocabulary)",
                    word2
                ))
            })?;

        // Compute cosine similarity
        crate::vectors::VectorStore::cosine_similarity(emb1, emb2).ok_or_else(|| {
            PyRuntimeError::new_err("Failed to compute cosine similarity (zero vectors?)")
        })
    }
}

/// A single morpheme from analysis
#[pyclass]
pub struct PyMorpheme {
    /// Surface form
    #[pyo3(get)]
    pub surface: String,
    /// Feature string
    #[pyo3(get)]
    pub feature: String,
    /// Part-of-speech ID
    #[pyo3(get)]
    pub pos_id: u16,
    /// Word cost
    #[pyo3(get)]
    pub wcost: i16,
}

#[pymethods]
impl PyMorpheme {
    fn __repr__(&self) -> String {
        format!("Morpheme('{}', '{}')", self.surface, self.feature)
    }

    fn __str__(&self) -> String {
        format!("{}\t{}", self.surface, self.feature)
    }
}

/// Get MeCrab version
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module definition
#[pymodule]
fn mecrab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMeCrab>()?;
    m.add_class::<PyMorpheme>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
}
