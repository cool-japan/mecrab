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
//! # Parse text → AnalysisResult (iterable)
//! result = m.parse("すもももももももものうち")
//! for morph in result:
//!     print(morph.surface, morph.pos)
//!
//! # len / index access
//! print(len(result))
//! print(result[0].surface)
//!
//! # Convenience: surfaces / readings / pos tags
//! print(result.surfaces())   # => ["すもも", "も", ...]
//! print(result.to_json())    # => JSON string
//!
//! # Parse to dictionary (Pythonic API)
//! morphemes = m.parse_to_dict("東京に行く")
//! for m in morphemes:
//!     print(m['surface'], m['pos'], m.get('ipa'))
//!
//! # Parse to Morpheme objects
//! morphemes = m.parse_to_morphemes("東京に行く")
//! for morph in morphemes:
//!     print(morph.surface, morph.pos, morph.reading)
//!
//! # N-best analysis
//! results = m.parse_nbest("すもももももももものうち", n=5)
//! for result, cost in results:
//!     print(f"Cost: {cost}, Analysis: {result.surfaces()}")
//!
//! # Wakati (space-separated)
//! words = m.wakati("すもももももももものうち")
//! print(words)
//!
//! # Wakati as list
//! word_list = m.wakati_list("すもももももももものうち")
//! print(word_list)  # => ["すもも", "も", ...]
//!
//! # Add custom words
//! m.add_word("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000)
//!
//! # Batch processing
//! results = m.parse_batch(["テスト1", "テスト2", "テスト3"])
//!
//! # Batch parsing with GIL release
//! results = m.parse_batch_py(["テスト1", "テスト2", "テスト3"])
//!
//! # Direct JSON output
//! json_str = m.parse_to_json("東京に行く")
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
//!
//! # Word2Vec operations
//! similar = m_vec.most_similar("東京", topn=10)
//! analogy = m_vec.analogy("王様", "男", "女")  # king - man + woman
//! embedding = m_vec.sentence_embedding("東京に行く")
//!
//! # JSON/JSON-LD output
//! json_result = m.parse_json("東京に行く")
//! jsonld_result = m.parse_jsonld("東京に行く")
//! ```

pub mod analysis;
pub mod corpus;
pub(crate) mod helpers;
pub mod parser;
pub mod rerank;
pub mod vectors;

pub use analysis::{PyAnalysisIterator, PyAnalysisResult, PyAnalysisResultIterator, PyMorpheme};
pub use corpus::PySurfaceVocab;
pub use parser::PyMeCrab;
pub use rerank::{PyCostReranker, PyNullReranker};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Module-level functions
// ─────────────────────────────────────────────────────────────────────────────

/// Get MeCrab version
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get default dictionary path
///
/// Returns:
///     Dictionary path if found, None otherwise
#[pyfunction]
fn default_dicdir() -> Option<String> {
    let locations = [
        "/var/lib/mecab/dic/ipadic-utf8",
        "/usr/lib/mecab/dic/ipadic-utf8",
        "/usr/local/lib/mecab/dic/ipadic-utf8",
        "/opt/homebrew/lib/mecab/dic/ipadic-utf8",
    ];

    for loc in locations {
        let path = std::path::Path::new(loc);
        if path.exists() && path.join("sys.dic").exists() {
            return Some(loc.to_string());
        }
    }
    None
}

/// Compute cosine similarity between two vectors
///
/// Args:
///     a: First vector
///     b: Second vector
///
/// Returns:
///     Cosine similarity in range [-1.0, 1.0]
///
/// Raises:
///     ValueError: If vectors have different dimensions or are zero
#[pyfunction]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "Vector dimensions must match: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    crate::vectors::VectorStore::cosine_similarity(&a, &b)
        .ok_or_else(|| PyValueError::new_err("Cannot compute similarity (zero vectors?)"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Python module definition
// ─────────────────────────────────────────────────────────────────────────────

/// Python module definition
#[pymodule]
pub fn mecrab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMeCrab>()?;
    m.add_class::<PyMorpheme>()?;
    m.add_class::<PyAnalysisResult>()?;
    m.add_class::<PyAnalysisResultIterator>()?;
    m.add_class::<PyAnalysisIterator>()?;
    m.add_class::<rerank::PyNullReranker>()?;
    m.add_class::<rerank::PyCostReranker>()?;
    m.add_class::<corpus::PySurfaceVocab>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(default_dicdir, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;

    // Add constants
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "COOLJAPAN OU (Team KitaSan)")?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
}
