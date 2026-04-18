//! Python wrapper for MeCrab morphological analyzer
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

use super::analysis::{PyAnalysisResult, PyMorpheme};
use super::corpus::PySurfaceVocab;

// ─────────────────────────────────────────────────────────────────────────────
// PyMeCrab
// ─────────────────────────────────────────────────────────────────────────────

/// Python wrapper for MeCrab morphological analyzer
#[pyclass(name = "MeCrab")]
pub struct PyMeCrab {
    pub(crate) inner: crate::MeCrab,
    pub(crate) with_ipa: bool,
    pub(crate) with_vector: bool,
}

#[pymethods]
impl PyMeCrab {
    /// Create a new MeCrab instance
    ///
    /// Args:
    ///     dicdir: Optional path to dictionary directory
    ///     with_ipa: Enable IPA pronunciation output (default: False)
    ///     vector_path: Optional path to word embeddings file
    ///
    /// Returns:
    ///     MeCrab instance
    ///
    /// Raises:
    ///     RuntimeError: If dictionary cannot be loaded
    ///
    /// Example (Python):
    /// ```python
    /// # Basic usage
    /// m = MeCrab()
    ///
    /// # With IPA pronunciation
    /// m = MeCrab(with_ipa=True)
    /// morphemes = m.parse_to_dict("東京に行く")
    /// print(morphemes[0]["ipa"])  # => "/toːkʲoː/"
    /// ```
    #[new]
    #[pyo3(signature = (dicdir=None, with_ipa=false, vector_path=None))]
    pub fn new(
        dicdir: Option<String>,
        with_ipa: bool,
        vector_path: Option<String>,
    ) -> PyResult<Self> {
        let mut builder = crate::MeCrab::builder();

        if let Some(path) = dicdir {
            builder = builder.dicdir(Some(PathBuf::from(path)));
        }

        if with_ipa {
            builder = builder.with_ipa(true);
        }

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

    // ── Core parse ──────────────────────────────────────────────────────────

    /// Parse text and return an `AnalysisResult` (iterable, indexable).
    ///
    /// Args:
    ///     text: Input text to analyze
    ///
    /// Returns:
    ///     AnalysisResult containing Morpheme objects
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    pub fn parse(&self, text: &str) -> PyResult<PyAnalysisResult> {
        self.inner
            .parse(text)
            .map(|r| PyAnalysisResult::from_result(text, &r))
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    /// Parse text and return JSON string directly.
    ///
    /// Args:
    ///     text: Input text to analyze
    ///
    /// Returns:
    ///     JSON string with text and morphemes
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    fn parse_to_json(&self, text: &str) -> PyResult<String> {
        let result = self.parse(text)?;
        Ok(result.to_json_string())
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
        self.inner
            .wakati(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    /// Parse text and return surface forms as a Python list.
    ///
    /// Args:
    ///     text: Input text to analyze
    ///
    /// Returns:
    ///     List[str] of surface forms
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    fn wakati_list<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyList>> {
        let result = self.parse(text)?;
        let list = PyList::empty(py);
        for m in &result.morphemes {
            list.append(m.surface.as_str())?;
        }
        Ok(list)
    }

    // ── Legacy / convenience parse variants ─────────────────────────────────

    /// Parse text and return list of morphemes as tuples.
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
        self.inner
            .parse(text)
            .map(|r| {
                r.morphemes
                    .iter()
                    .map(|m| (m.surface.clone(), m.feature.clone()))
                    .collect()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
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
    ///     ...     print(morph['surface'], morph['pos'])
    #[allow(clippy::doc_link_with_quotes)]
    fn parse_to_dict<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .parse(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))?
            .morphemes
            .iter()
            .map(|m| Ok(self.morpheme_to_dict(py, m)))
            .collect()
    }

    /// Parse text and return Morpheme objects
    ///
    /// Args:
    ///     text: Input text to analyze
    ///
    /// Returns:
    ///     List of Morpheme objects
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    ///
    /// Example:
    ///     >>> m = MeCrab()
    ///     >>> morphemes = m.parse_to_morphemes("東京に行く")
    ///     >>> for morph in morphemes:
    ///     ...     if morph.is_noun():
    ///     ...         print(f"Noun: {morph.surface} ({morph.reading})")
    fn parse_to_morphemes(&self, text: &str) -> PyResult<Vec<PyMorpheme>> {
        self.inner
            .parse(text)
            .map(|r| r.morphemes.iter().map(PyMorpheme::from_morpheme).collect())
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    // ── N-best ───────────────────────────────────────────────────────────────

    /// Parse text and return N-best analysis results
    ///
    /// Returns multiple alternative analyses ranked by cost.
    ///
    /// Args:
    ///     text: Input text to analyze
    ///     n: Number of best paths to return (default: 5)
    ///
    /// Returns:
    ///     List of (AnalysisResult, cost) tuples sorted by cost
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    #[pyo3(signature = (text, n=5))]
    fn parse_nbest(&self, text: &str, n: usize) -> PyResult<Vec<(PyAnalysisResult, i64)>> {
        self.inner
            .parse_nbest(text, n)
            .map(|results| {
                results
                    .into_iter()
                    .map(|(r, cost)| (PyAnalysisResult::from_result(text, &r), cost))
                    .collect()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    // ── JSON / JSON-LD output ────────────────────────────────────────────────

    /// Parse text and return JSON output (array of morpheme objects)
    ///
    /// Args:
    ///     text: Input text to analyze
    ///
    /// Returns:
    ///     JSON string
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    fn parse_json(&self, text: &str) -> PyResult<String> {
        self.inner
            .parse(text)
            .map(|result| {
                let mut json = String::from("[");
                for (i, m) in result.morphemes.iter().enumerate() {
                    if i > 0 {
                        json.push(',');
                    }
                    json.push_str(&format!(
                        "{{\"surface\":{},\"feature\":{},\"pos_id\":{},\"wcost\":{}}}",
                        super::helpers::serde_json_str(&m.surface),
                        super::helpers::serde_json_str(&m.feature),
                        m.pos_id,
                        m.wcost
                    ));
                }
                json.push(']');
                json
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    /// Parse text and return JSON-LD output with semantic annotations
    ///
    /// Args:
    ///     text: Input text to analyze
    ///
    /// Returns:
    ///     JSON-LD string
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    fn parse_jsonld(&self, text: &str) -> PyResult<String> {
        self.inner
            .parse(text)
            .map(|result| {
                let format_result = crate::AnalysisResult {
                    morphemes: result.morphemes,
                    format: crate::OutputFormat::Jsonld,
                };
                format!("{format_result}")
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    // ── BPE / probabilistic output ──────────────────────────────────────────

    /// Parse text and return in BPE-compatible (SentencePiece) format.
    ///
    /// Returns tokens marked with ▁ (U+2581) at word-initial positions,
    /// space-separated. Compatible with SentencePiece and mT5 tokenizers.
    ///
    /// Args:
    ///     text: Input Japanese text
    ///
    /// Returns:
    ///     Space-separated tokens with ▁ word-initial markers
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    pub fn parse_bpe(&self, text: &str) -> PyResult<String> {
        self.inner
            .parse(text)
            .map(|result| {
                let formatted = crate::AnalysisResult {
                    morphemes: result.morphemes,
                    format: crate::OutputFormat::BpeCompatible,
                };
                format!("{formatted}")
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    /// Parse text and return analysis result with marginal probabilities.
    ///
    /// Uses the forward-backward algorithm to compute P(morpheme | input)
    /// for every node in the lattice, enabling probabilistic analysis.
    ///
    /// Args:
    ///     text: Input Japanese text
    ///
    /// Returns:
    ///     Tuple of (AnalysisResult, list of dicts with probability info).
    ///     Each dict has keys: "surface", "log_prob", "prob".
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    pub fn parse_with_probs<'py>(
        &self,
        py: Python<'py>,
        text: &str,
    ) -> PyResult<(PyAnalysisResult, Bound<'py, PyList>)> {
        let (result, prob_table) = self
            .inner
            .parse_with_probs(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))?;

        let py_result = PyAnalysisResult::from_result(text, &result);

        let prob_list = super::helpers::build_prob_list(py, &prob_table)?;

        Ok((py_result, prob_list))
    }

    /// Parse multiple texts in parallel and return BPE-compatible format.
    ///
    /// Uses the GIL-release pattern (same as `parse_batch_py`) so other Python
    /// threads can run while the Rust side is busy.
    ///
    /// Args:
    ///     texts: List of input texts
    ///
    /// Returns:
    ///     List of BPE-formatted strings (▁-marked tokens, space-separated)
    ///
    /// Raises:
    ///     RuntimeError: If any parsing fails
    #[pyo3(text_signature = "(self, texts, /)")]
    pub fn parse_bpe_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<String>> {
        py.detach(|| {
            texts
                .iter()
                .map(|t| {
                    self.inner
                        .parse(t)
                        .map(|result| {
                            let formatted = crate::AnalysisResult {
                                morphemes: result.morphemes,
                                format: crate::OutputFormat::BpeCompatible,
                            };
                            format!("{formatted}")
                        })
                        .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
                })
                .collect::<PyResult<Vec<_>>>()
        })
    }

    /// Parse text and return analysis in CoNLL-U (Universal Dependencies) format.
    ///
    /// Produces tab-separated 10-field lines with Universal POS tags, morphological
    /// features, base forms (LEMMA), and heuristic dependency HEAD/DEPREL for Japanese.
    /// Each sentence ends with a blank line. Compatible with UD treebank tools.
    ///
    /// Args:
    ///     text: Input Japanese text
    ///
    /// Returns:
    ///     CoNLL-U formatted string
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails
    pub fn parse_conllu(&self, text: &str) -> PyResult<String> {
        self.inner
            .parse(text)
            .map(|result| {
                let formatted = crate::AnalysisResult {
                    morphemes: result.morphemes,
                    format: crate::OutputFormat::ConllU,
                };
                format!("{formatted}")
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    /// Parse multiple texts in parallel and return CoNLL-U format.
    ///
    /// Uses the GIL-release pattern so other Python threads can run while Rust parses.
    ///
    /// Args:
    ///     texts: List of input texts
    ///
    /// Returns:
    ///     List of CoNLL-U formatted strings (one per input text)
    ///
    /// Raises:
    ///     RuntimeError: If any parsing fails
    #[pyo3(text_signature = "(self, texts, /)")]
    pub fn parse_conllu_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<String>> {
        py.detach(|| {
            texts
                .iter()
                .map(|t| {
                    self.inner
                        .parse(t)
                        .map(|result| {
                            let formatted = crate::AnalysisResult {
                                morphemes: result.morphemes,
                                format: crate::OutputFormat::ConllU,
                            };
                            format!("{formatted}")
                        })
                        .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
                })
                .collect::<PyResult<Vec<_>>>()
        })
    }

    /// Parse multiple texts and return analysis results with marginal probabilities.
    ///
    /// For each text, uses the forward-backward algorithm to compute
    /// `P(morpheme | input)` for every lattice position.  Releases the GIL
    /// during the Rust computation phase.
    ///
    /// Args:
    ///     texts: List of input texts
    ///
    /// Returns:
    ///     List of (AnalysisResult, prob_list) tuples.
    ///     Each prob_list entry is a dict with keys: "surface", "log_prob", "prob".
    ///
    /// Raises:
    ///     RuntimeError: If any parsing fails
    pub fn parse_with_probs_batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Vec<(PyAnalysisResult, Bound<'py, PyList>)>> {
        // Release the GIL while we run all the forward-backward passes.
        // We collect raw Rust data first, then re-acquire the GIL to build
        // Python objects — the PyO3 `detach` closure returns plain Rust types.
        let raw_results: Vec<(
            String,                                     // text (for PyAnalysisResult)
            crate::AnalysisResult,                      // best-path result
            crate::viterbi::analysis::LatticeProbTable, // lattice probs
        )> = py.detach(|| {
            texts
                .iter()
                .map(|t| {
                    self.inner
                        .parse_with_probs(t)
                        .map(|(result, probs)| (t.clone(), result, probs))
                        .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
                })
                .collect::<PyResult<Vec<_>>>()
        })?;

        // Re-acquired GIL: build Python objects from raw data.
        raw_results
            .into_iter()
            .map(|(text, result, prob_table)| {
                let py_result = PyAnalysisResult::from_result(text.as_str(), &result);
                let prob_list = super::helpers::build_prob_list(py, &prob_table)?;
                Ok((py_result, prob_list))
            })
            .collect::<PyResult<Vec<_>>>()
    }

    /// Parse text and return N-best segmentations each paired with the lattice
    /// marginal probabilities.
    ///
    /// The marginal probabilities are computed once from the full lattice via the
    /// forward-backward algorithm — they are therefore shared across all N-best
    /// paths and reflect the probability of each morpheme position given *all*
    /// possible paths, not just the current N-best candidate.  This is the
    /// correct interpretation: the prob table is a property of the lattice, not
    /// of a single path.
    ///
    /// Args:
    ///     text: Input Japanese text
    ///     n: Number of N-best candidates to return (default: 5)
    ///
    /// Returns:
    ///     List of (AnalysisResult, cost, prob_list) tuples in ascending cost order.
    ///     prob_list: dicts with "surface", "log_prob", "prob" — same for every tuple.
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails or the lattice produces no candidates
    #[pyo3(signature = (text, n=5))]
    pub fn parse_nbest_with_probs<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        n: usize,
    ) -> PyResult<Vec<(PyAnalysisResult, i64, Bound<'py, PyList>)>> {
        let text_owned = text.to_owned();

        // Release GIL: run N-best + forward-backward in parallel without
        // blocking other Python threads.
        let (nbest_results, prob_table) = py.detach(|| -> PyResult<_> {
            let nbest = self
                .inner
                .parse_nbest(&text_owned, n)
                .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))?;

            let (_, prob_table) = self
                .inner
                .parse_with_probs(&text_owned)
                .map_err(|e| PyRuntimeError::new_err(format!("Prob computation error: {e}")))?;

            Ok((nbest, prob_table))
        })?;

        nbest_results
            .into_iter()
            .map(|(result, cost)| {
                let py_result = PyAnalysisResult::from_result(text, &result);
                // Rebuild the prob_list for each N-best entry (pyo3 `Bound` is
                // not `Clone`; the underlying CPython dicts are ref-counted so
                // this is negligible compared to the Viterbi work above).
                let prob_list = super::helpers::build_prob_list(py, &prob_table)?;
                Ok((py_result, cost, prob_list))
            })
            .collect::<PyResult<Vec<_>>>()
    }

    // ── Batch processing ─────────────────────────────────────────────────────

    /// Parse multiple texts in batch, returning raw MeCab-format strings.
    ///
    /// When compiled with `parallel` feature, uses Rayon for parallel processing.
    ///
    /// Args:
    ///     texts: List of texts to analyze
    ///
    /// Returns:
    ///     List of MeCab-format strings
    ///
    /// Raises:
    ///     RuntimeError: If any parsing fails
    fn parse_batch(&self, texts: Vec<String>) -> PyResult<Vec<String>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner
            .parse_batch(&refs)
            .into_iter()
            .map(|r| {
                r.map(|result| result.to_string())
                    .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
            })
            .collect()
    }

    /// Parse multiple texts in parallel, returning `AnalysisResult` objects.
    ///
    /// Releases the GIL during batch processing so other Python threads can run.
    ///
    /// Args:
    ///     texts: List of texts to analyze
    ///
    /// Returns:
    ///     List of AnalysisResult objects
    ///
    /// Raises:
    ///     RuntimeError: If any parsing fails
    fn parse_batch_py(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
    ) -> PyResult<Vec<PyAnalysisResult>> {
        py.detach(|| {
            texts
                .iter()
                .map(|t| {
                    self.inner
                        .parse(t)
                        .map(|r| PyAnalysisResult::from_result(t.as_str(), &r))
                        .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
                })
                .collect::<PyResult<Vec<_>>>()
        })
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
        self.inner
            .wakati_batch(&refs)
            .into_iter()
            .collect::<Result<Vec<String>, _>>()
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
    }

    /// Parse multiple texts and return Morpheme objects in batch
    ///
    /// Args:
    ///     texts: List of texts to analyze
    ///
    /// Returns:
    ///     List of lists of Morpheme objects
    ///
    /// Raises:
    ///     RuntimeError: If any parsing fails
    fn parse_batch_to_morphemes(&self, texts: Vec<String>) -> PyResult<Vec<Vec<PyMorpheme>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner
            .parse_batch(&refs)
            .into_iter()
            .map(|r| {
                r.map(|result| {
                    result
                        .morphemes
                        .iter()
                        .map(PyMorpheme::from_morpheme)
                        .collect()
                })
                .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
            })
            .collect()
    }

    /// Parse multiple texts and return dictionaries in batch
    ///
    /// Args:
    ///     texts: List of texts to analyze
    ///
    /// Returns:
    ///     List of lists of dictionaries
    ///
    /// Raises:
    ///     RuntimeError: If any parsing fails
    fn parse_batch_to_dict<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Vec<Vec<Bound<'py, PyDict>>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner
            .parse_batch(&refs)
            .into_iter()
            .map(|r| {
                r.map(|result| {
                    result
                        .morphemes
                        .iter()
                        .map(|m| self.morpheme_to_dict(py, m))
                        .collect()
                })
                .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
            })
            .collect()
    }

    // ── Dictionary management ────────────────────────────────────────────────

    /// Add a word to the overlay dictionary.
    ///
    /// Changes take effect immediately.
    ///
    /// Args:
    ///     surface: The surface form (the actual text)
    ///     reading: The katakana reading
    ///     pronunciation: The pronunciation (often same as reading)
    ///     wcost: Word cost (lower = more preferred, typical: 5000-8000)
    fn add_word(&self, surface: &str, reading: &str, pronunciation: &str, wcost: i16) {
        self.inner.add_word(surface, reading, pronunciation, wcost);
    }

    /// Remove a word from the overlay dictionary.
    ///
    /// Args:
    ///     surface: The surface form to remove
    ///
    /// Returns:
    ///     True if the word was found and removed, False otherwise
    fn remove_word(&self, surface: &str) -> bool {
        self.inner.remove_word(surface)
    }

    /// Get the number of words in the overlay dictionary.
    fn overlay_size(&self) -> usize {
        self.inner.overlay_size()
    }

    /// Get dictionary information.
    ///
    /// Returns:
    ///     Dictionary with keys: overlay_size, has_vectors, with_ipa
    fn dict_info<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        let _ = dict.set_item("overlay_size", self.inner.overlay_size());
        let _ = dict.set_item("has_vectors", self.with_vector);
        let _ = dict.set_item("with_ipa", self.with_ipa);
        dict
    }

    /// Check whether the analyser is fully loaded and ready.
    fn is_loaded(&self) -> bool {
        true
    }

    // ── Properties ───────────────────────────────────────────────────────────

    /// Check if this instance has vector support enabled
    #[getter]
    fn has_vectors(&self) -> bool {
        self.with_vector
    }

    /// Check if this instance has IPA support enabled
    #[getter]
    fn has_ipa(&self) -> bool {
        self.with_ipa
    }

    // ── IPA ──────────────────────────────────────────────────────────────────

    /// Convert text to IPA pronunciation (one-shot conversion)
    ///
    /// Requires: MeCrab initialized with with_ipa=True
    ///
    /// Args:
    ///     text: Input text to convert
    ///
    /// Returns:
    ///     List of IPA pronunciation strings
    ///
    /// Raises:
    ///     RuntimeError: If IPA is not enabled or parsing fails
    fn to_ipa(&self, text: &str) -> PyResult<Vec<String>> {
        if !self.with_ipa {
            return Err(PyRuntimeError::new_err(
                "IPA support not enabled. Create MeCrab with with_ipa=True",
            ));
        }

        self.inner
            .parse(text)
            .map(|result| {
                result
                    .morphemes
                    .iter()
                    .filter_map(|m| m.pronunciation.clone())
                    .collect()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
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
    #[pyo3(signature = (text, separator=" "))]
    fn to_ipa_text(&self, text: &str, separator: &str) -> PyResult<String> {
        let ipas = self.to_ipa(text)?;
        Ok(ipas.join(separator))
    }
    // ── Reranked N-best ──────────────────────────────────────────────────────

    /// Parse text, generate N-best candidates, and return the best one chosen
    /// by the specified reranker.
    ///
    /// Args:
    ///     text: Input text to analyze.
    ///     n: Number of N-best paths to generate (default: 5).
    ///     reranker: Reranker to use — ``"null"`` (default) or ``"cost"``.
    ///
    /// Returns:
    ///     AnalysisResult for the reranker-selected path.
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails or no candidates are produced.
    #[pyo3(signature = (text, n=5, reranker="null"))]
    pub fn parse_nbest_reranked(
        &self,
        py: Python<'_>,
        text: &str,
        n: usize,
        reranker: &str,
    ) -> PyResult<PyAnalysisResult> {
        let text_owned = text.to_owned();
        let reranker_owned = reranker.to_owned();
        let result = py.detach(|| {
            self.inner
                .parse_nbest_with_reranker(
                    &text_owned,
                    n,
                    match reranker_owned.as_str() {
                        "cost" => &crate::rerank::CostReranker as &dyn crate::rerank::Reranker,
                        _ => &crate::rerank::NullReranker,
                    },
                )
                .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))
        })?;
        Ok(PyAnalysisResult::from_result(text, &result))
    }

    // ── Corpus tokenization ──────────────────────────────────────────────────

    /// Parse `text` and return word IDs using the provided `SurfaceVocab`.
    ///
    /// New surface forms are automatically assigned the next available ID.
    /// Punctuation, symbols, and whitespace tokens are included (no filtering);
    /// for corpus-style filtered output use the Rust-level
    /// `tokenize_for_corpus` instead.
    ///
    /// Args:
    ///     text: Input text to tokenize.
    ///     vocab: A `SurfaceVocab` instance that maps surfaces to IDs.
    ///
    /// Returns:
    ///     List of integer word IDs, one per morpheme in the analysis result.
    ///
    /// Raises:
    ///     RuntimeError: If parsing fails or the vocab lock is poisoned.
    pub fn tokenize_text(&self, text: &str, vocab: &PySurfaceVocab) -> PyResult<Vec<u32>> {
        let result = self
            .inner
            .parse(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse error: {e}")))?;
        result
            .morphemes
            .iter()
            .map(|m| vocab.get_or_insert(m.surface.as_str()))
            .collect()
    }

    // ── Context manager ──────────────────────────────────────────────────────

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, pyo3::types::PyType>>,
        _exc_val: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_tb: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> bool {
        false
    }
}

impl PyMeCrab {
    /// Helper: convert a `crate::Morpheme` to a `PyDict`
    pub fn morpheme_to_dict<'py>(
        &self,
        py: Python<'py>,
        m: &crate::Morpheme,
    ) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);

        let _ = dict.set_item("surface", &m.surface);
        let _ = dict.set_item("feature", &m.feature);

        let parts: Vec<&str> = m.feature.split(',').collect();
        if !parts.is_empty() {
            let _ = dict.set_item("pos", parts[0]);
            if parts.len() > 1 && parts[1] != "*" {
                let _ = dict.set_item("pos1", parts[1]);
            }
            if parts.len() > 2 && parts[2] != "*" {
                let _ = dict.set_item("pos2", parts[2]);
            }
            if parts.len() > 3 && parts[3] != "*" {
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

        if let Some(ref ipa) = m.pronunciation {
            let _ = dict.set_item("ipa", ipa.as_str());
        }
        if let Some(ref embedding) = m.embedding {
            let _ = dict.set_item("embedding", embedding.clone());
        }
        let _ = dict.set_item("pos_id", m.pos_id);
        let _ = dict.set_item("wcost", m.wcost);
        let _ = dict.set_item("word_id", m.word_id);

        dict
    }
}
