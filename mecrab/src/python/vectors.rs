//! Vector / word-embedding operations for the Python MeCrab wrapper
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! All methods here live inside the `#[pymethods]` block of [`super::parser::PyMeCrab`]
//! but are physically separated into this module to keep `parser.rs` lean.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::parser::PyMeCrab;

// ─────────────────────────────────────────────────────────────────────────────
// Vector guard helper
// ─────────────────────────────────────────────────────────────────────────────

/// Return an error if the instance was not built with vector support.
#[inline]
fn require_vectors(with_vector: bool) -> PyResult<()> {
    if with_vector {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err(
            "Vector support not enabled. Create MeCrab with vector_path parameter",
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Word embedding helper
// ─────────────────────────────────────────────────────────────────────────────

/// Parse `word` and return `(word_id, embedding)` for its first morpheme.
pub(crate) fn parse_word_embedding(inner: &crate::MeCrab, word: &str) -> PyResult<(u32, Vec<f32>)> {
    let result = inner
        .parse(word)
        .map_err(|e| PyRuntimeError::new_err(format!("Parse error for '{word}': {e}")))?;
    let m = result.morphemes.first().ok_or_else(|| {
        PyRuntimeError::new_err(format!("No morpheme produced for word: '{word}'"))
    })?;
    let emb = m.embedding.clone().ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "No embedding found for '{word}' (may be out-of-vocabulary)"
        ))
    })?;
    Ok((m.word_id, emb))
}

// ─────────────────────────────────────────────────────────────────────────────
// pymethods impl block (vector operations)
// ─────────────────────────────────────────────────────────────────────────────

#[pymethods]
impl PyMeCrab {
    /// Compute cosine similarity between two words
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
    pub fn similarity(&self, word1: &str, word2: &str) -> PyResult<f32> {
        require_vectors(self.with_vector)?;

        let result1 = self
            .inner
            .parse(word1)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error for word1: {e}")))?;
        let result2 = self
            .inner
            .parse(word2)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error for word2: {e}")))?;

        let emb1 = result1
            .morphemes
            .first()
            .and_then(|m| m.embedding.as_ref())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "No embedding found for word1: '{word1}' (may be out-of-vocabulary)"
                ))
            })?;

        let emb2 = result2
            .morphemes
            .first()
            .and_then(|m| m.embedding.as_ref())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "No embedding found for word2: '{word2}' (may be out-of-vocabulary)"
                ))
            })?;

        crate::vectors::VectorStore::cosine_similarity(emb1, emb2).ok_or_else(|| {
            PyRuntimeError::new_err("Failed to compute cosine similarity (zero vectors?)")
        })
    }

    /// Find words most similar to the given word
    ///
    /// Args:
    ///     word: The query word
    ///     topn: Number of similar words to return (default: 10)
    ///
    /// Returns:
    ///     List of (word, similarity_score) tuples
    ///
    /// Raises:
    ///     RuntimeError: If vectors not enabled or word not found
    #[pyo3(signature = (word, topn=10))]
    pub fn most_similar(&self, word: &str, topn: usize) -> PyResult<Vec<(String, f32)>> {
        require_vectors(self.with_vector)?;

        let store = self
            .inner
            .vector_store()
            .ok_or_else(|| PyRuntimeError::new_err("Vector store not loaded"))?;

        // Parse the query word to get its embedding and word_id.
        let result = self
            .inner
            .parse(word)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))?;

        let first_morpheme = result.morphemes.first().ok_or_else(|| {
            PyRuntimeError::new_err(format!("No morpheme produced for word: '{word}'"))
        })?;

        let query_vec = first_morpheme
            .embedding
            .as_ref()
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "No embedding found for word: '{word}' (may be out-of-vocabulary)"
                ))
            })?
            .clone();

        let query_word_id = first_morpheme.word_id;

        // Exclude the query word itself from results. Request topn+1 to guard
        // against the rare case where the query word slips past the exclusion
        // set (e.g. word_id == u32::MAX for unknown words).
        let exclude = if query_word_id == u32::MAX {
            vec![]
        } else {
            vec![query_word_id]
        };

        let raw = store.most_similar_by_vec(&query_vec, topn + 1, &exclude);

        // Map word_ids to human-readable labels: surface form when the vocab
        // index is attached, otherwise the numeric word_id as a string.
        let results: Vec<(String, f32)> = raw
            .into_iter()
            .filter(|(id, _)| *id != query_word_id)
            .take(topn)
            .map(|(id, score)| {
                let label = store
                    .surface_of(id)
                    .map_or_else(|| id.to_string(), |s| s.to_string());
                (label, score)
            })
            .collect();

        Ok(results)
    }

    /// Perform word analogy: positive1 - negative + positive2 = ?
    ///
    /// Args:
    ///     positive1: First positive word (e.g., "king")
    ///     negative: Negative word to subtract (e.g., "man")
    ///     positive2: Second positive word to add (e.g., "woman")
    ///     topn: Number of results to return (default: 5)
    ///
    /// Returns:
    ///     List of (word, similarity_score) tuples
    ///
    /// Raises:
    ///     RuntimeError: If vectors not enabled or words not found
    #[pyo3(signature = (positive1, negative, positive2, topn=5))]
    pub fn analogy(
        &self,
        positive1: &str,
        negative: &str,
        positive2: &str,
        topn: usize,
    ) -> PyResult<Vec<(String, f32)>> {
        require_vectors(self.with_vector)?;

        let store = self
            .inner
            .vector_store()
            .ok_or_else(|| PyRuntimeError::new_err("Vector store not loaded"))?;

        let (id_p1, vec_p1) = parse_word_embedding(&self.inner, positive1)?;
        let (id_neg, vec_neg) = parse_word_embedding(&self.inner, negative)?;
        let (id_p2, vec_p2) = parse_word_embedding(&self.inner, positive2)?;

        // Analogy vector: positive1 - negative + positive2
        let dim = vec_p1.len();
        let mut query: Vec<f32> = vec![0.0_f32; dim];
        for i in 0..dim {
            query[i] = vec_p1[i] - vec_neg[i] + vec_p2[i];
        }

        // Exclude all three input words from the result set.
        let mut exclude: Vec<u32> = Vec::with_capacity(3);
        for &id in &[id_p1, id_neg, id_p2] {
            if id != u32::MAX && !exclude.contains(&id) {
                exclude.push(id);
            }
        }

        let raw = store.most_similar_by_vec(&query, topn + exclude.len(), &exclude);

        let results: Vec<(String, f32)> = raw
            .into_iter()
            .take(topn)
            .map(|(id, score)| {
                let label = store
                    .surface_of(id)
                    .map_or_else(|| id.to_string(), |s| s.to_string());
                (label, score)
            })
            .collect();

        Ok(results)
    }

    /// Get sentence embedding (mean pooling of word vectors)
    ///
    /// Args:
    ///     text: Input text
    ///
    /// Returns:
    ///     Embedding vector as list of floats
    ///
    /// Raises:
    ///     RuntimeError: If vectors not enabled or no words have embeddings
    pub fn sentence_embedding(&self, text: &str) -> PyResult<Vec<f32>> {
        require_vectors(self.with_vector)?;

        let result = self
            .inner
            .parse(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))?;

        let embeddings: Vec<&[f32]> = result
            .morphemes
            .iter()
            .filter_map(|m| m.embedding.as_deref())
            .collect();

        if embeddings.is_empty() {
            return Err(PyRuntimeError::new_err(
                "No embeddings found for any words in text",
            ));
        }

        let dim = embeddings[0].len();
        let mut sum = vec![0.0_f32; dim];

        for emb in &embeddings {
            for (i, &val) in emb.iter().enumerate() {
                sum[i] += val;
            }
        }

        let count = embeddings.len() as f32;
        for val in &mut sum {
            *val /= count;
        }

        Ok(sum)
    }
}
