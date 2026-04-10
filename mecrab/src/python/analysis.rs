//! Python types for morphological analysis results
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::helpers::serde_json_str;

// ─────────────────────────────────────────────────────────────────────────────
// PyMorpheme
// ─────────────────────────────────────────────────────────────────────────────

/// A single morpheme from analysis
#[pyclass(name = "Morpheme", from_py_object)]
#[derive(Clone)]
pub struct PyMorpheme {
    /// Surface form
    #[pyo3(get)]
    pub surface: String,
    /// Feature string (comma-separated)
    #[pyo3(get)]
    pub feature: String,
    /// Part-of-speech (main category)
    #[pyo3(get)]
    pub pos: String,
    /// Part-of-speech subcategory 1
    #[pyo3(get)]
    pub pos1: Option<String>,
    /// Part-of-speech subcategory 2
    #[pyo3(get)]
    pub pos2: Option<String>,
    /// Part-of-speech subcategory 3
    #[pyo3(get)]
    pub pos3: Option<String>,
    /// Conjugation type
    #[pyo3(get)]
    pub inflection: Option<String>,
    /// Conjugation form
    #[pyo3(get)]
    pub conjugation: Option<String>,
    /// Base form (lemma)
    #[pyo3(get)]
    pub base: Option<String>,
    /// Reading (katakana)
    #[pyo3(get)]
    pub reading: Option<String>,
    /// Pronunciation (katakana)
    #[pyo3(get)]
    pub pronunciation: Option<String>,
    /// IPA pronunciation (optional)
    #[pyo3(get)]
    pub ipa: Option<String>,
    /// Word embedding vector (optional)
    #[pyo3(get)]
    pub embedding: Option<Vec<f32>>,
    /// Part-of-speech ID
    #[pyo3(get)]
    pub pos_id: u16,
    /// Word cost
    #[pyo3(get)]
    pub wcost: i16,
    /// Word ID (for lookup in vector store)
    #[pyo3(get)]
    pub word_id: u32,
    /// Start byte offset in the input text
    pub start_byte: usize,
    /// End byte offset (exclusive) in the input text
    pub end_byte: usize,
}

impl PyMorpheme {
    /// Create a `PyMorpheme` from a crate `Morpheme`
    pub fn from_morpheme(m: &crate::Morpheme) -> Self {
        let parts: Vec<&str> = m.feature.split(',').collect();

        let pos = parts.first().map(|s| s.to_string()).unwrap_or_default();
        let pos1 = parts.get(1).filter(|&&s| s != "*").map(|s| s.to_string());
        let pos2 = parts.get(2).filter(|&&s| s != "*").map(|s| s.to_string());
        let pos3 = parts.get(3).filter(|&&s| s != "*").map(|s| s.to_string());
        let inflection = parts.get(4).filter(|&&s| s != "*").map(|s| s.to_string());
        let conjugation = parts.get(5).filter(|&&s| s != "*").map(|s| s.to_string());
        let base = parts.get(6).filter(|&&s| s != "*").map(|s| s.to_string());
        let reading = parts.get(7).filter(|&&s| s != "*").map(|s| s.to_string());
        let pronunciation = parts.get(8).filter(|&&s| s != "*").map(|s| s.to_string());

        Self {
            surface: m.surface.clone(),
            feature: m.feature.clone(),
            pos,
            pos1,
            pos2,
            pos3,
            inflection,
            conjugation,
            base,
            reading,
            pronunciation,
            ipa: m.pronunciation.clone(),
            embedding: m.embedding.clone(),
            pos_id: m.pos_id,
            wcost: m.wcost,
            word_id: m.word_id,
            start_byte: m.start_byte,
            end_byte: m.end_byte,
        }
    }

    /// Serialize morpheme to a JSON string (public Rust API)
    pub fn to_json_string(&self) -> String {
        let mut obj = format!(
            "{{\"surface\":{},\"pos\":{},\"feature\":{},\"start_byte\":{},\"end_byte\":{}",
            serde_json_str(&self.surface),
            serde_json_str(&self.pos),
            serde_json_str(&self.feature),
            self.start_byte,
            self.end_byte,
        );
        if let Some(ref v) = self.pos1 {
            obj.push_str(&format!(",\"pos1\":{}", serde_json_str(v)));
        }
        if let Some(ref v) = self.pos2 {
            obj.push_str(&format!(",\"pos2\":{}", serde_json_str(v)));
        }
        if let Some(ref v) = self.pos3 {
            obj.push_str(&format!(",\"pos3\":{}", serde_json_str(v)));
        }
        if let Some(ref v) = self.inflection {
            obj.push_str(&format!(",\"inflection\":{}", serde_json_str(v)));
        }
        if let Some(ref v) = self.conjugation {
            obj.push_str(&format!(",\"conjugation\":{}", serde_json_str(v)));
        }
        if let Some(ref v) = self.base {
            obj.push_str(&format!(",\"base\":{}", serde_json_str(v)));
        }
        if let Some(ref v) = self.reading {
            obj.push_str(&format!(",\"reading\":{}", serde_json_str(v)));
        }
        if let Some(ref v) = self.pronunciation {
            obj.push_str(&format!(",\"pronunciation\":{}", serde_json_str(v)));
        }
        if let Some(ref v) = self.ipa {
            obj.push_str(&format!(",\"ipa\":{}", serde_json_str(v)));
        }
        obj.push_str(&format!(
            ",\"pos_id\":{},\"wcost\":{},\"word_id\":{}",
            self.pos_id, self.wcost, self.word_id
        ));
        obj.push('}');
        obj
    }
}

#[pymethods]
impl PyMorpheme {
    fn __repr__(&self) -> String {
        format!(
            "Morpheme(surface={:?}, pos={:?}, reading={:?})",
            self.surface, self.pos, self.reading
        )
    }

    fn __str__(&self) -> String {
        format!("{}\t{}", self.surface, self.feature)
    }

    /// Convert morpheme to a Python dictionary
    #[allow(clippy::unnecessary_wraps)]
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("surface", &self.surface)?;
        dict.set_item("feature", &self.feature)?;
        dict.set_item("pos", &self.pos)?;

        if let Some(ref v) = self.pos1 {
            dict.set_item("pos1", v)?;
        }
        if let Some(ref v) = self.pos2 {
            dict.set_item("pos2", v)?;
        }
        if let Some(ref v) = self.pos3 {
            dict.set_item("pos3", v)?;
        }
        if let Some(ref v) = self.inflection {
            dict.set_item("inflection", v)?;
        }
        if let Some(ref v) = self.conjugation {
            dict.set_item("conjugation", v)?;
        }
        if let Some(ref v) = self.base {
            dict.set_item("base", v)?;
        }
        if let Some(ref v) = self.reading {
            dict.set_item("reading", v)?;
        }
        if let Some(ref v) = self.pronunciation {
            dict.set_item("pronunciation", v)?;
        }
        if let Some(ref v) = self.ipa {
            dict.set_item("ipa", v)?;
        }
        if let Some(ref v) = self.embedding {
            dict.set_item("embedding", v.clone())?;
        }
        dict.set_item("pos_id", self.pos_id)?;
        dict.set_item("wcost", self.wcost)?;
        dict.set_item("word_id", self.word_id)?;
        dict.set_item("start_byte", self.start_byte)?;
        dict.set_item("end_byte", self.end_byte)?;

        Ok(dict)
    }

    /// Serialize morpheme to a JSON string
    fn to_json(&self) -> String {
        self.to_json_string()
    }

    /// Check if morpheme has embedding
    #[getter]
    fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Check if morpheme has IPA pronunciation
    #[getter]
    fn has_ipa(&self) -> bool {
        self.ipa.is_some()
    }

    /// Get embedding dimension (or 0 if no embedding)
    #[getter]
    fn embedding_dim(&self) -> usize {
        self.embedding.as_ref().map(|e| e.len()).unwrap_or(0)
    }

    /// Check if morpheme is a noun
    fn is_noun(&self) -> bool {
        self.pos == "名詞"
    }

    /// Check if morpheme is a verb
    fn is_verb(&self) -> bool {
        self.pos == "動詞"
    }

    /// Check if morpheme is an adjective
    fn is_adjective(&self) -> bool {
        self.pos == "形容詞"
    }

    /// Check if morpheme is a particle
    fn is_particle(&self) -> bool {
        self.pos == "助詞"
    }

    /// Check if morpheme is an auxiliary verb
    fn is_auxiliary(&self) -> bool {
        self.pos == "助動詞"
    }

    /// Check if morpheme is a symbol/punctuation
    fn is_symbol(&self) -> bool {
        self.pos == "記号"
    }

    /// Byte offset where this morpheme starts in the input text
    #[getter]
    fn start_byte(&self) -> usize {
        self.start_byte
    }

    /// Byte offset (exclusive) where this morpheme ends in the input text
    #[getter]
    fn end_byte(&self) -> usize {
        self.end_byte
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyAnalysisResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of morphological analysis — iterable, indexable container of Morphemes.
#[pyclass(name = "AnalysisResult", from_py_object)]
#[derive(Clone)]
pub struct PyAnalysisResult {
    /// Original text that was analysed
    #[pyo3(get)]
    pub text: String,
    /// All morphemes in order
    #[pyo3(get)]
    pub morphemes: Vec<PyMorpheme>,
}

impl PyAnalysisResult {
    /// Construct from crate types
    pub fn from_result(text: impl Into<String>, result: &crate::AnalysisResult) -> Self {
        Self {
            text: text.into(),
            morphemes: result
                .morphemes
                .iter()
                .map(PyMorpheme::from_morpheme)
                .collect(),
        }
    }

    /// Serialize the analysis to a JSON string (public Rust API)
    pub fn to_json_string(&self) -> String {
        let morphemes_json: Vec<String> =
            self.morphemes.iter().map(|m| m.to_json_string()).collect();
        format!(
            "{{\"text\":{},\"morphemes\":[{}]}}",
            serde_json_str(&self.text),
            morphemes_json.join(",")
        )
    }
}

#[pymethods]
impl PyAnalysisResult {
    fn __repr__(&self) -> String {
        format!(
            "AnalysisResult(text={:?}, morphemes={})",
            self.text,
            self.morphemes.len()
        )
    }

    fn __str__(&self) -> String {
        let mut out = String::new();
        for m in &self.morphemes {
            out.push_str(&format!("{}\t{}\n", m.surface, m.feature));
        }
        out.push_str("EOS\n");
        out
    }

    fn __len__(&self) -> usize {
        self.morphemes.len()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAnalysisResultIterator>> {
        let iter = PyAnalysisResultIterator {
            morphemes: slf.morphemes.clone(),
            index: AtomicUsize::new(0),
        };
        Py::new(slf.py(), iter)
    }

    fn __getitem__(&self, idx: isize) -> PyResult<PyMorpheme> {
        let len = self.morphemes.len() as isize;
        let real = if idx < 0 { len + idx } else { idx };
        if real < 0 || real >= len {
            return Err(PyIndexError::new_err("index out of range"));
        }
        Ok(self.morphemes[real as usize].clone())
    }

    /// Return list of dicts (one per morpheme)
    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for m in &self.morphemes {
            list.append(m.to_dict(py)?)?;
        }
        Ok(list)
    }

    /// Serialize the entire analysis to a JSON string
    fn to_json(&self) -> String {
        self.to_json_string()
    }

    /// Return list of surface forms
    fn surfaces(&self) -> Vec<String> {
        self.morphemes.iter().map(|m| m.surface.clone()).collect()
    }

    /// Return list of readings (katakana); empty string when not available
    fn readings(&self) -> Vec<String> {
        self.morphemes
            .iter()
            .map(|m| m.reading.clone().unwrap_or_default())
            .collect()
    }

    /// Return list of part-of-speech tags
    fn pos_tags(&self) -> Vec<String> {
        self.morphemes.iter().map(|m| m.pos.clone()).collect()
    }

    /// Return byte-offset spans for all non-EOS morphemes.
    ///
    /// Returns a list of `(start_byte, end_byte)` pairs in order.
    fn spans(&self) -> Vec<(usize, usize)> {
        self.morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .map(|m| (m.start_byte, m.end_byte))
            .collect()
    }

    /// Extract compound noun phrases.
    ///
    /// Consecutive morphemes whose POS starts with `"名詞"` are joined.
    /// Returns list of `(surface, start_byte, end_byte)` tuples.
    fn noun_phrases(&self) -> Vec<(String, usize, usize)> {
        let mut result = Vec::new();
        let non_eos: Vec<&PyMorpheme> = self
            .morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .collect();

        let mut i = 0;
        while i < non_eos.len() {
            let m = non_eos[i];
            if m.pos.starts_with("名詞") {
                let start_byte = m.start_byte;
                let mut end_byte = m.end_byte;
                let mut compound = m.surface.clone();
                let mut j = i + 1;
                while j < non_eos.len() {
                    let next = non_eos[j];
                    if next.pos.starts_with("名詞") {
                        compound.push_str(&next.surface);
                        end_byte = next.end_byte;
                        j += 1;
                    } else {
                        break;
                    }
                }
                if !compound.chars().all(|c| {
                    !c.is_alphanumeric()
                        && !matches!(c, 'ぁ'..='ん' | 'ァ'..='ン' | '\u{4E00}'..='\u{9FFF}' | 'A'..='z')
                }) {
                    result.push((compound, start_byte, end_byte));
                }
                i = j;
            } else {
                i += 1;
            }
        }
        result
    }

    /// Extract named entities (proper nouns with 固有名詞 in feature).
    ///
    /// Returns list of `(surface, entity_type, start_byte, end_byte)` tuples
    /// where `entity_type` is the IPADIC sub-category (地域, 人名, 組織, 一般, etc.)
    fn named_entities(&self) -> Vec<(String, String, usize, usize)> {
        let mut result = Vec::new();
        let non_eos: Vec<&PyMorpheme> = self
            .morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .collect();

        let mut i = 0;
        while i < non_eos.len() {
            let m = non_eos[i];
            // pos == "名詞", pos1 == "固有名詞"
            if m.pos == "名詞" && m.pos1.as_deref() == Some("固有名詞") {
                let entity_type = m.pos2.clone().unwrap_or_else(|| "一般".to_string());
                let start_byte = m.start_byte;
                let mut end_byte = m.end_byte;
                let mut compound = m.surface.clone();
                let mut j = i + 1;
                while j < non_eos.len() {
                    let next = non_eos[j];
                    if next.pos == "名詞" && next.pos1.as_deref() == Some("固有名詞") {
                        compound.push_str(&next.surface);
                        end_byte = next.end_byte;
                        j += 1;
                    } else {
                        break;
                    }
                }
                result.push((compound, entity_type, start_byte, end_byte));
                i = j;
            } else {
                i += 1;
            }
        }
        result
    }

    /// Extract verb chunks (verb + following auxiliaries / verb endings).
    ///
    /// Collects a base verb (`動詞`) followed by adjacent `助動詞`, `接尾`, or `形容詞`.
    /// Returns list of `(surface, start_byte, end_byte)` tuples.
    fn verb_chunks(&self) -> Vec<(String, usize, usize)> {
        let mut result = Vec::new();
        let non_eos: Vec<&PyMorpheme> = self
            .morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .collect();

        let mut i = 0;
        while i < non_eos.len() {
            let m = non_eos[i];
            if m.pos == "動詞" {
                let start_byte = m.start_byte;
                let mut end_byte = m.end_byte;
                let mut chunk = m.surface.clone();
                let mut j = i + 1;
                while j < non_eos.len() {
                    let next = non_eos[j];
                    match next.pos.as_str() {
                        "動詞" => break,
                        "助動詞" | "接尾" | "形容詞" => {
                            chunk.push_str(&next.surface);
                            end_byte = next.end_byte;
                            j += 1;
                        }
                        _ => break,
                    }
                }
                result.push((chunk, start_byte, end_byte));
                i = j;
            } else {
                i += 1;
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyAnalysisResultIterator
// ─────────────────────────────────────────────────────────────────────────────

/// Iterator over morphemes in an `AnalysisResult`.
#[pyclass(name = "AnalysisResultIterator")]
pub struct PyAnalysisResultIterator {
    pub(crate) morphemes: Vec<PyMorpheme>,
    pub(crate) index: AtomicUsize,
}

#[pymethods]
impl PyAnalysisResultIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self) -> Option<PyMorpheme> {
        let idx = self.index.fetch_add(1, Ordering::SeqCst);
        self.morphemes.get(idx).cloned()
    }

    fn __len__(&self) -> usize {
        let current = self.index.load(Ordering::SeqCst);
        self.morphemes.len().saturating_sub(current)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy streaming iterator (kept for backwards compatibility)
// ─────────────────────────────────────────────────────────────────────────────

/// Analysis result iterator for streaming processing (legacy)
#[pyclass(name = "AnalysisIterator")]
pub struct PyAnalysisIterator {
    pub(crate) morphemes: Vec<PyMorpheme>,
    pub(crate) index: AtomicUsize,
}

#[pymethods]
impl PyAnalysisIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self) -> Option<PyMorpheme> {
        let idx = self.index.fetch_add(1, Ordering::SeqCst);
        self.morphemes.get(idx).cloned()
    }

    fn __len__(&self) -> usize {
        self.morphemes.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_result_getitem_negative() {
        let result = PyAnalysisResult {
            text: "test".to_string(),
            morphemes: vec![PyMorpheme {
                surface: "東京".to_string(),
                feature: "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ".to_string(),
                pos: "名詞".to_string(),
                pos1: Some("固有名詞".to_string()),
                pos2: Some("地域".to_string()),
                pos3: Some("一般".to_string()),
                inflection: None,
                conjugation: None,
                base: Some("東京".to_string()),
                reading: Some("トウキョウ".to_string()),
                pronunciation: Some("トウキョウ".to_string()),
                ipa: None,
                embedding: None,
                pos_id: 0,
                wcost: 3863,
                word_id: 0,
                start_byte: 0,
                end_byte: 6,
            }],
        };
        assert_eq!(result.__getitem__(0).unwrap().surface, "東京");
        assert_eq!(result.__getitem__(-1).unwrap().surface, "東京");
        assert!(result.__getitem__(5).is_err());
    }

    #[test]
    fn test_morpheme_to_json() {
        let m = PyMorpheme {
            surface: "東京".to_string(),
            feature: "名詞,固有名詞".to_string(),
            pos: "名詞".to_string(),
            pos1: Some("固有名詞".to_string()),
            pos2: None,
            pos3: None,
            inflection: None,
            conjugation: None,
            base: None,
            reading: Some("トウキョウ".to_string()),
            pronunciation: None,
            ipa: None,
            embedding: None,
            pos_id: 0,
            wcost: 3863,
            word_id: 0,
            start_byte: 0,
            end_byte: 6,
        };
        let json = m.to_json();
        assert!(json.contains("\"surface\":\"東京\""));
        assert!(json.contains("\"pos\":\"名詞\""));
    }
}
