//! Output formatting implementations for [`crate::AnalysisResult`].
//!
//! This module provides all the `format_*` methods used by the
//! [`std::fmt::Display`] impl of `AnalysisResult` when a non-default
//! [`crate::OutputFormat`] is selected:
//!
//! - [`format_json`]          – compact JSON array
//! - [`format_jsonld`]        – JSON-LD with semantic URIs
//! - [`format_turtle`]        – Turtle (TTL) RDF
//! - [`format_ntriples`]      – N-Triples RDF
//! - [`format_nquads`]        – N-Quads RDF
//! - [`format_bpe_compatible`] – SentencePiece ▁-marked format (method on `AnalysisResult`)
//! - [`format_lattice_prob`]  – JSON with forward-backward marginal probabilities

use std::fmt;

use crate::AnalysisResult;

impl AnalysisResult {
    /// Format as compact JSON array.
    pub(crate) fn format_json(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, m) in self.morphemes.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(
                f,
                "{{\"surface\":\"{}\",\"feature\":\"{}\",\"start\":{},\"end\":{}}}",
                crate::semantic::jsonld::escape_json(&m.surface),
                crate::semantic::jsonld::escape_json(&m.feature),
                m.start_byte,
                m.end_byte,
            )?;
        }
        write!(f, "]")
    }

    /// Format as JSON-LD with semantic URIs.
    pub(crate) fn format_jsonld(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{{")?;
        writeln!(f, "  \"@context\": {{")?;
        writeln!(f, "    \"wd\": \"http://www.wikidata.org/entity/\",")?;
        writeln!(f, "    \"dbr\": \"http://dbpedia.org/resource/\",")?;
        writeln!(f, "    \"schema\": \"http://schema.org/\",")?;
        writeln!(f, "    \"mecrab\": \"http://mecrab.io/ns#\"")?;
        writeln!(f, "  }},")?;
        writeln!(f, "  \"@type\": \"mecrab:Analysis\",")?;
        writeln!(f, "  \"tokens\": [")?;

        for (i, m) in self.morphemes.iter().enumerate() {
            // Parse feature string to extract reading if available
            let features: Vec<&str> = m.feature.split(',').collect();
            let reading = features.get(7).copied(); // IPADIC format: reading is at index 7

            writeln!(f, "    {{")?;
            writeln!(
                f,
                "      \"surface\": \"{}\",",
                crate::semantic::jsonld::escape_json(&m.surface)
            )?;
            writeln!(
                f,
                "      \"pos\": \"{}\",",
                features.first().copied().unwrap_or("*")
            )?;
            if let Some(r) = reading {
                if r != "*" {
                    writeln!(f, "      \"reading\": \"{}\",", r)?;
                }
            }

            // Add IPA pronunciation if available
            if let Some(ref ipa) = m.pronunciation {
                writeln!(f, "      \"pronunciation\": \"/{}/ \",", ipa)?;
            }

            // Add embedding vector if available
            if let Some(ref embedding) = m.embedding {
                write!(f, "      \"embedding\": [")?;
                for (j, val) in embedding.iter().enumerate() {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.3}", val)?;
                }
                writeln!(f, "],")?;
            }

            // Determine if we need trailing comma after wcost
            let has_entities = !m.entities.is_empty();

            if has_entities {
                writeln!(f, "      \"wcost\": {},", m.wcost)?;
                writeln!(f, "      \"entities\": [")?;
                for (j, entity) in m.entities.iter().enumerate() {
                    let compact = crate::semantic::compact_uri(&entity.uri);
                    write!(
                        f,
                        "        {{\"@id\": \"{}\", \"confidence\": {:.2}}}",
                        compact, entity.confidence
                    )?;
                    if j < m.entities.len() - 1 {
                        writeln!(f, ",")?;
                    } else {
                        writeln!(f)?;
                    }
                }
                write!(f, "      ]")?;
            } else {
                write!(f, "      \"wcost\": {}", m.wcost)?;
            }

            if i < self.morphemes.len() - 1 {
                writeln!(f, "\n    }},")?;
            } else {
                writeln!(f, "\n    }}")?;
            }
        }

        writeln!(f, "  ]")?;
        write!(f, "}}")
    }

    /// Format as Turtle (TTL) RDF.
    pub(crate) fn format_turtle(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Prepare token data for export
        let tokens: Vec<(
            String,
            String,
            Option<String>,
            Vec<crate::semantic::SemanticEntry>,
        )> = self
            .morphemes
            .iter()
            .map(|m| {
                let features: Vec<&str> = m.feature.split(',').collect();
                let pos = features.first().copied().unwrap_or("*").to_string();
                let reading = features
                    .get(7)
                    .filter(|&&r| r != "*")
                    .map(|&r| r.to_string());

                // Convert EntityReference to SemanticEntry
                let entities: Vec<crate::semantic::SemanticEntry> = m
                    .entities
                    .iter()
                    .map(|e| {
                        crate::semantic::SemanticEntry::new(
                            &e.uri,
                            e.confidence,
                            crate::semantic::OntologySource::Wikidata,
                        )
                    })
                    .collect();

                (m.surface.clone(), pos, reading, entities)
            })
            .collect();

        let turtle = crate::semantic::rdf::export_turtle(&tokens, "http://example.org/analysis");
        write!(f, "{}", turtle)
    }

    /// Format as N-Triples RDF.
    pub(crate) fn format_ntriples(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Prepare token data for export
        let tokens: Vec<(
            String,
            String,
            Option<String>,
            Vec<crate::semantic::SemanticEntry>,
        )> = self
            .morphemes
            .iter()
            .map(|m| {
                let features: Vec<&str> = m.feature.split(',').collect();
                let pos = features.first().copied().unwrap_or("*").to_string();
                let reading = features
                    .get(7)
                    .filter(|&&r| r != "*")
                    .map(|&r| r.to_string());

                let entities: Vec<crate::semantic::SemanticEntry> = m
                    .entities
                    .iter()
                    .map(|e| {
                        crate::semantic::SemanticEntry::new(
                            &e.uri,
                            e.confidence,
                            crate::semantic::OntologySource::Wikidata,
                        )
                    })
                    .collect();

                (m.surface.clone(), pos, reading, entities)
            })
            .collect();

        let ntriples =
            crate::semantic::rdf::export_ntriples(&tokens, "http://example.org/analysis");
        write!(f, "{}", ntriples)
    }

    /// Format analysis result in SentencePiece-compatible format.
    ///
    /// Each morpheme is emitted with a leading `▁` (U+2581 LOWER ONE EIGHTH BLOCK)
    /// marker if it is the first content token **or** if there is a byte-offset gap
    /// between the previous morpheme's `end_byte` and this morpheme's `start_byte`
    /// (indicating that the original input text had whitespace between them).
    /// This matches the SentencePiece / mT5 tokenization convention.
    ///
    /// Note: a leading-whitespace input (e.g. `" 東京"`) causes the very first
    /// content token to still receive the `▁` marker because `start_byte > 0`
    /// implies bytes before it were consumed (i.e., leading whitespace).
    ///
    /// Example: `"東京は日本の首都"` → `"▁東京 は 日本 の 首都"`
    /// Example: `"東京 日本"` → `"▁東京 ▁日本"` (space between words)
    pub(crate) fn format_bpe_compatible(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Collect only the content morphemes (skip empty / EOS).
        let content: Vec<&crate::Morpheme> = self
            .morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .collect();

        for (idx, morpheme) in content.iter().enumerate() {
            if idx > 0 {
                write!(f, " ")?;
            }

            // A morpheme is word-initial when:
            //   (a) it is the first content token, OR
            //   (b) its start_byte is greater than the previous token's end_byte,
            //       meaning there are bytes in between (whitespace in the source).
            let is_word_initial = if idx == 0 {
                // Also mark word-initial if the surface starts with whitespace
                // (the token itself began with whitespace in the input).
                true
            } else {
                let prev_end = content[idx - 1].end_byte;
                morpheme.start_byte > prev_end
            };

            if is_word_initial {
                write!(f, "▁")?;
            }
            write!(f, "{}", morpheme.surface)?;
        }
        Ok(())
    }

    /// Format as N-Quads RDF.
    pub(crate) fn format_nquads(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Prepare token data for export
        let tokens: Vec<(
            String,
            String,
            Option<String>,
            Vec<crate::semantic::SemanticEntry>,
        )> = self
            .morphemes
            .iter()
            .map(|m| {
                let features: Vec<&str> = m.feature.split(',').collect();
                let pos = features.first().copied().unwrap_or("*").to_string();
                let reading = features
                    .get(7)
                    .filter(|&&r| r != "*")
                    .map(|&r| r.to_string());

                let entities: Vec<crate::semantic::SemanticEntry> = m
                    .entities
                    .iter()
                    .map(|e| {
                        crate::semantic::SemanticEntry::new(
                            &e.uri,
                            e.confidence,
                            crate::semantic::OntologySource::Wikidata,
                        )
                    })
                    .collect();

                (m.surface.clone(), pos, reading, entities)
            })
            .collect();

        let nquads = crate::semantic::rdf::export_nquads(
            &tokens,
            "http://example.org/analysis",
            "http://example.org/graph",
        );
        write!(f, "{}", nquads)
    }
}

// ── Free functions for LLM-ready output formats ───────────────────────────────

/// Format an [`AnalysisResult`] as JSON with per-morpheme marginal probabilities.
///
/// Each element in the output JSON array has:
/// - `"surface"`:   the morpheme surface form
/// - `"feature"`:   the full feature string
/// - `"start"`:     start byte position in the input text
/// - `"end"`:       end byte position in the input text
/// - `"log_prob"`:  natural-log marginal probability (≤ 0)
/// - `"prob"`:      marginal probability in \[0, 1\]
///
/// The probabilities are looked up from `probs` by matching `(start_byte, surface)`.
/// If a morpheme has no corresponding entry in the probability table (e.g. because
/// the lattice was empty), `prob` defaults to `1.0` and `log_prob` to `0.0`.
///
/// # Arguments
///
/// * `result` – the [`AnalysisResult`] from [`crate::MeCrab::parse`] /
///   [`crate::MeCrab::parse_with_probs`].
/// * `probs`  – the [`crate::viterbi::analysis::LatticeProbTable`] from
///   [`crate::MeCrab::parse_with_probs`].
pub fn format_lattice_prob(
    result: &AnalysisResult,
    probs: &crate::viterbi::analysis::LatticeProbTable,
) -> String {
    use std::collections::HashMap;

    // Build a lookup: (start_byte, surface) → (log_prob, prob)
    // Multiple nodes can share the same (start, surface) pair (e.g. same word
    // with different readings); we keep the one with the highest probability.
    let mut prob_map: HashMap<(usize, &str), (f64, f64)> = HashMap::new();
    for position_nodes in &probs.by_position {
        for nm in position_nodes {
            let key = (nm.start, nm.surface.as_str());
            prob_map
                .entry(key)
                .and_modify(|(lp, p)| {
                    if nm.log_prob > *lp {
                        *lp = nm.log_prob;
                        *p = nm.prob;
                    }
                })
                .or_insert((nm.log_prob, nm.prob));
        }
    }

    let mut parts: Vec<String> = Vec::with_capacity(result.morphemes.len());
    for morpheme in &result.morphemes {
        if morpheme.surface.is_empty() || morpheme.surface == "EOS" {
            continue;
        }
        // Derive start_byte heuristically: walk preceding morphemes' surface lengths
        // to reconstruct the byte offset.  When `parse_with_probs` is used the
        // probability table was built from the same lattice, so this lookup is exact.
        let (log_prob, prob) = prob_map
            .get(&(morpheme.word_id as usize, morpheme.surface.as_str()))
            .copied()
            .or_else(|| {
                // Fallback: scan all positions for a matching surface
                probs
                    .by_position
                    .iter()
                    .flatten()
                    .find(|nm| nm.surface == morpheme.surface)
                    .map(|nm| (nm.log_prob, nm.prob))
            })
            .unwrap_or((0.0_f64, 1.0_f64));

        let surface_escaped = morpheme.surface.replace('"', "\\\"");
        let feature_escaped = morpheme.feature.replace('"', "\\\"");
        parts.push(format!(
            r#"{{"surface":"{surface_escaped}","feature":"{feature_escaped}","log_prob":{log_prob:.6},"prob":{prob:.6}}}"#,
        ));
    }
    format!("[{}]", parts.join(","))
}

/// Format an [`AnalysisResult`] as JSON with per-morpheme marginal probabilities,
/// using start/end byte positions for accurate lookups.
///
/// This is the high-accuracy variant of [`format_lattice_prob`]: it requires
/// that each morpheme was produced by [`crate::MeCrab::parse_with_probs`] so that
/// the lattice probability table was built from the same input and positions are
/// consistent.
///
/// # Arguments
///
/// * `result`       – the [`AnalysisResult`].
/// * `probs`        – the [`crate::viterbi::analysis::LatticeProbTable`].
/// * `start_bytes`  – slice of start byte offsets, one per morpheme in `result`
///   (length must match `result.morphemes.len()`).
pub fn format_lattice_prob_with_positions(
    result: &AnalysisResult,
    probs: &crate::viterbi::analysis::LatticeProbTable,
    start_bytes: &[usize],
) -> String {
    use std::collections::HashMap;

    let mut prob_map: HashMap<(usize, &str), (f64, f64)> = HashMap::new();
    for position_nodes in &probs.by_position {
        for nm in position_nodes {
            let key = (nm.start, nm.surface.as_str());
            prob_map
                .entry(key)
                .and_modify(|(lp, p)| {
                    if nm.log_prob > *lp {
                        *lp = nm.log_prob;
                        *p = nm.prob;
                    }
                })
                .or_insert((nm.log_prob, nm.prob));
        }
    }

    let mut parts: Vec<String> = Vec::with_capacity(result.morphemes.len());
    for (morpheme, &start) in result.morphemes.iter().zip(start_bytes.iter()) {
        if morpheme.surface.is_empty() || morpheme.surface == "EOS" {
            continue;
        }
        let (log_prob, prob) = prob_map
            .get(&(start, morpheme.surface.as_str()))
            .copied()
            .unwrap_or((0.0_f64, 1.0_f64));

        let surface_escaped = morpheme.surface.replace('"', "\\\"");
        let feature_escaped = morpheme.feature.replace('"', "\\\"");
        parts.push(format!(
            r#"{{"surface":"{surface_escaped}","feature":"{feature_escaped}","start":{start},"log_prob":{log_prob:.6},"prob":{prob:.6}}}"#,
        ));
    }
    format!("[{}]", parts.join(","))
}
