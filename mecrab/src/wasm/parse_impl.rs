//! Parse methods on `MeCrabWasm`.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! All wasm-bindgen-exported `parse*` methods live here as a second `impl`
//! block on [`super::core::MeCrabWasm`].  Each method follows the same
//! pattern: guard against an uninitialised dictionary, build a lattice, run
//! the Viterbi solver, then delegate formatting to [`super::format`].

use wasm_bindgen::prelude::*;

use super::core::MeCrabWasm;

#[wasm_bindgen]
impl MeCrabWasm {
    /// Parse Japanese text and return morphological analysis as JSON.
    ///
    /// The returned value is a JSON object.  On success it has a `tokens` array
    /// where each element contains `surface`, `feature`, `pos_id`, and `wcost`.
    /// On failure the object has a single `error` field.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 Japanese text to analyse
    #[wasm_bindgen(js_name = "parse")]
    pub fn parse(&self, text: &str) -> String {
        let dict = match self.inner.as_ref() {
            Some(d) => d,
            None => {
                return serde_json::json!({
                    "error": "Dictionary not loaded. Call loadDictionary() first."
                })
                .to_string();
            }
        };

        let lattice = match crate::lattice::Lattice::build(text, dict) {
            Ok(l) => l,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Lattice build error: {}", e)
                })
                .to_string();
            }
        };

        let solver = crate::viterbi::ViterbiSolver::new(dict);
        let path = match solver.solve(&lattice) {
            Ok(p) => p,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Viterbi error: {}", e)
                })
                .to_string();
            }
        };

        super::format::format_json_output(&path)
    }

    /// Parse Japanese text and return space-separated surface forms (wakati style).
    ///
    /// On success returns the space-separated surface forms.  On failure returns
    /// a string beginning with `"Error: "`.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 Japanese text to analyse
    #[wasm_bindgen(js_name = "parseWakati")]
    pub fn parse_wakati(&self, text: &str) -> String {
        let dict = match self.inner.as_ref() {
            Some(d) => d,
            None => {
                return "Error: Dictionary not loaded. Call loadDictionary() first.".to_string();
            }
        };

        let lattice = match crate::lattice::Lattice::build(text, dict) {
            Ok(l) => l,
            Err(e) => return format!("Error: Lattice build error: {}", e),
        };

        let solver = crate::viterbi::ViterbiSolver::new(dict);
        match solver.solve(&lattice) {
            Ok(path) => path
                .iter()
                .map(|node| node.surface.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            Err(e) => format!("Error: Viterbi error: {}", e),
        }
    }

    /// Parse text and return in SentencePiece-compatible BPE format.
    ///
    /// Returns space-separated tokens with ▁ (U+2581) prefix on word-initial
    /// tokens, matching the SentencePiece/mT5 tokenization convention.
    ///
    /// Returns an error string prefixed with `"Error: "` if parsing fails.
    #[wasm_bindgen(js_name = "parseBpeCompatible")]
    pub fn parse_bpe_compatible(&self, text: &str) -> String {
        let dict = match self.inner.as_ref() {
            Some(d) => d,
            None => {
                return "Error: Dictionary not loaded. Call loadDictionary() first.".to_string();
            }
        };

        let lattice = match crate::lattice::Lattice::build(text, dict) {
            Ok(l) => l,
            Err(e) => return format!("Error: Lattice build error: {}", e),
        };

        let solver = crate::viterbi::ViterbiSolver::new(dict);
        let path = match solver.solve(&lattice) {
            Ok(p) => p,
            Err(e) => return format!("Error: Viterbi error: {}", e),
        };

        let morphemes: Vec<crate::Morpheme> = path
            .into_iter()
            .map(super::format::node_to_morpheme)
            .collect();

        let result = crate::AnalysisResult {
            morphemes,
            format: crate::OutputFormat::BpeCompatible,
        };

        format!("{}", result)
    }

    /// Parse Japanese text and return analysis in CoNLL-U format (Universal Dependencies).
    ///
    /// Produces tab-separated 10-field lines with:
    /// - FORM, LEMMA, UPOS (Universal POS), XPOS (IPADIC POS), FEATS (morphological)
    /// - HEAD and DEPREL use rule-based Japanese dependency heuristics
    /// - MISC: SpaceAfter=No (Japanese has no spaces between words)
    ///
    /// Returns an error string prefixed with `"Error: "` if parsing fails.
    #[wasm_bindgen(js_name = "parseConllu")]
    pub fn parse_conllu(&self, text: &str) -> String {
        let dict = match self.inner.as_ref() {
            Some(d) => d,
            None => {
                return "Error: Dictionary not loaded. Call loadDictionary() first.".to_string();
            }
        };

        let lattice = match crate::lattice::Lattice::build(text, dict) {
            Ok(l) => l,
            Err(e) => return format!("Error: Lattice build error: {}", e),
        };

        let solver = crate::viterbi::ViterbiSolver::new(dict);
        let path = match solver.solve(&lattice) {
            Ok(p) => p,
            Err(e) => return format!("Error: Viterbi error: {}", e),
        };

        let morphemes: Vec<crate::Morpheme> = path
            .into_iter()
            .map(super::format::node_to_morpheme)
            .collect();

        let result = crate::AnalysisResult {
            morphemes,
            format: crate::OutputFormat::ConllU,
        };

        format!("{}", result)
    }

    /// Parse text and return JSON with marginal probabilities (forward-backward).
    ///
    /// JSON array of objects with:
    /// - `"surface"`: morpheme surface form
    /// - `"feature"`: full feature string
    /// - `"log_prob"`: natural log marginal probability (≤ 0)
    /// - `"prob"`: marginal probability in [0, 1]
    ///
    /// Returns an error string prefixed with `"Error: "` if parsing fails.
    #[wasm_bindgen(js_name = "parseLatticeProb")]
    pub fn parse_lattice_prob(&self, text: &str) -> String {
        let dict = match self.inner.as_ref() {
            Some(d) => d,
            None => {
                return "Error: Dictionary not loaded. Call loadDictionary() first.".to_string();
            }
        };

        let lattice = match crate::lattice::Lattice::build(text, dict) {
            Ok(l) => l,
            Err(e) => return format!("Error: Lattice build error: {}", e),
        };

        let solver = crate::viterbi::ViterbiSolver::new(dict);
        let path = match solver.solve(&lattice) {
            Ok(p) => p,
            Err(e) => return format!("Error: Viterbi error: {}", e),
        };

        let probs = solver.forward_backward(&lattice);

        let morphemes: Vec<crate::Morpheme> = path
            .into_iter()
            .map(super::format::node_to_morpheme)
            .collect();

        let result = crate::AnalysisResult {
            morphemes,
            format: crate::OutputFormat::LatticeProb,
        };

        crate::api::format::format_lattice_prob(&result, &probs)
    }

    /// Parse Japanese text and return morphological analysis as a JSON array.
    ///
    /// Each element in the array contains `surface`, `feature`, `start` (start byte
    /// offset in the original UTF-8 input), and `end` (end byte offset).
    ///
    /// EOS tokens are excluded from the output.
    ///
    /// On failure returns a string prefixed with `"Error: "`.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 Japanese text to analyse
    #[wasm_bindgen(js_name = "parseToJson")]
    pub fn parse_to_json(&self, text: &str) -> String {
        let dict = match self.inner.as_ref() {
            Some(d) => d,
            None => {
                return "Error: Dictionary not loaded. Call loadDictionary() first.".to_string();
            }
        };

        let lattice = match crate::lattice::Lattice::build(text, dict) {
            Ok(l) => l,
            Err(e) => return format!("Error: Lattice build error: {}", e),
        };

        let solver = crate::viterbi::ViterbiSolver::new(dict);
        let path = match solver.solve(&lattice) {
            Ok(p) => p,
            Err(e) => return format!("Error: Viterbi error: {}", e),
        };

        let morphemes: Vec<crate::Morpheme> = path
            .into_iter()
            .filter(|node| node.surface != "EOS")
            .map(super::format::node_to_morpheme)
            .collect();

        let result = crate::AnalysisResult {
            morphemes,
            format: crate::OutputFormat::Json,
        };

        format!("{}", result)
    }

    /// Parse text and return results in CSV format (tab-separated).
    ///
    /// Each line: `surface\tfeature\tstart_byte\tend_byte`
    ///
    /// Tab-separated for compatibility with Japanese text containing commas.
    /// EOS tokens are excluded from the output.
    ///
    /// On failure returns a string prefixed with `"Error: "`.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 Japanese text to analyse
    #[wasm_bindgen(js_name = "parseToCsv")]
    pub fn parse_to_csv(&self, text: &str) -> String {
        let dict = match self.inner.as_ref() {
            Some(d) => d,
            None => {
                return "Error: Dictionary not loaded. Call loadDictionary() first.".to_string();
            }
        };

        let lattice = match crate::lattice::Lattice::build(text, dict) {
            Ok(l) => l,
            Err(e) => return format!("Error: Lattice build error: {}", e),
        };

        let solver = crate::viterbi::ViterbiSolver::new(dict);
        let path = match solver.solve(&lattice) {
            Ok(p) => p,
            Err(e) => return format!("Error: Viterbi error: {}", e),
        };

        super::format::format_csv_output(path)
    }
}
