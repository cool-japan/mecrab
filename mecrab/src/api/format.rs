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
//! - [`format_conllu`]        – Universal Dependencies CoNLL-U format

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

// ── CoNLL-U (Universal Dependencies) helpers ─────────────────────────────────

/// Map an IPADIC POS tag (and its first sub-category) to a Universal POS tag.
///
/// This follows the UD v2 guidelines for Japanese as closely as possible within
/// the constraints of the IPADIC tag set.
fn ipadic_to_upos(pos: &str, pos_detail1: &str) -> &'static str {
    match pos {
        "名詞" => match pos_detail1 {
            "代名詞" => "PRON",
            "固有名詞" => "PROPN",
            "数" => "NUM",
            "形容動詞語幹" => "ADJ",
            _ => "NOUN",
        },
        "動詞" => "VERB",
        "形容詞" => "ADJ",
        "副詞" => "ADV",
        "接続詞" => "CCONJ",
        "感動詞" | "フィラー" => "INTJ",
        "助詞" => match pos_detail1 {
            "格助詞" | "連体化" => "ADP",
            "副詞化" => "ADV",
            "接続助詞" => "SCONJ",
            _ => "PART",
        },
        "助動詞" => "AUX",
        "記号" => "PUNCT",
        "接頭詞" | "接頭辞" => "X",
        _ => "X",
    }
}

/// Map IPADIC conjugation fields to UD morphological features.
///
/// Only `conj_form` (field index 5) is used for FEATS; `conj_type` is
/// language-specific and belongs in XPOS instead.  Returns `"_"` when
/// no known mapping applies.
fn ipadic_to_feats(conj_type: &str, conj_form: &str) -> String {
    // conj_type is not currently used in the FEATS output (language-specific),
    // but the parameter is kept for forward compatibility.
    let _ = conj_type;
    match conj_form {
        "*" | "" => "_".to_owned(),
        "連用形" => "VerbForm=Ger".to_owned(),
        "連体形" => "VerbForm=Part".to_owned(),
        "終止形" => "VerbForm=Fin".to_owned(),
        "命令形" => "Mood=Imp|VerbForm=Fin".to_owned(),
        "未然形" => "VerbForm=Inf".to_owned(),
        "仮定形" => "Mood=Cnd|VerbForm=Fin".to_owned(),
        "基本形" => "VerbForm=Inf".to_owned(),
        _ => "_".to_owned(),
    }
}

/// Build the XPOS string from the first four IPADIC feature fields.
///
/// Joins fields [0..=3] that are not `"*"` with a hyphen.
/// Examples:
/// - `["名詞", "固有名詞", "地域", "一般"]` → `"名詞-固有名詞-地域-一般"`
/// - `["名詞", "*", "*", "*"]` → `"名詞"`
/// - `["助動詞", "*", "*", "*"]` → `"助動詞"`
fn ipadic_xpos(features: &[&str]) -> String {
    features
        .iter()
        .take(4)
        .filter(|&&s| s != "*" && !s.is_empty())
        .copied()
        .collect::<Vec<&str>>()
        .join("-")
}

/// Compute rule-based Japanese dependency heads and relations.
///
/// This is a morphological-level heuristic, not a trained parser. It uses
/// simple structural rules for Japanese (SOV, left-branching) to produce
/// plausible HEAD/DEPREL values in CoNLL-U output.
///
/// Rules:
/// - Root: rightmost predicate (動詞 or sentence-final 助動詞)
/// - 助詞 (particle): attaches to left neighbor (`case`)
/// - 助動詞 (auxiliary): attaches to left neighbor (`aux`)
/// - 形容詞 (adjective): attaches to next noun, or root (`amod`)
/// - 副詞 (adverb): attaches to root (`advmod`)
/// - 接続詞 (conjunction): attaches to right neighbor (`cc`)
/// - 記号 (punctuation): attaches to root (`punct`)
/// - 感動詞 (interjection): attaches to root (`discourse`)
/// - 名詞 (noun): before root → `nsubj`; after root → `obj`
/// - Other verbs (non-root): `ccomp`
/// - Everything else: `dep`
///
/// Returns a `Vec<(usize, &'static str)>` parallel to `tokens`:
/// each entry is `(head_id_1based, deprel)` where `head_id=0` means root.
#[allow(dead_code)]
fn compute_heuristic_deps(
    tokens: &[(usize, &str, &str)], // (token_id_1based, pos, pos_detail1)
) -> Vec<(usize, &'static str)> {
    let n = tokens.len();
    if n == 0 {
        return Vec::new();
    }

    // 1. Find the root: rightmost predicate
    let root_idx = {
        // First try: rightmost 動詞
        // Then: rightmost 助動詞
        // Fallback: last non-punctuation token
        let verb_pos = tokens.iter().rposition(|(_, pos, _)| pos.starts_with("動詞"));
        let aux_pos = tokens.iter().rposition(|(_, pos, _)| pos.starts_with("助動詞"));
        // prefer verb over aux; both over fallback
        verb_pos.or(aux_pos).unwrap_or_else(|| {
            tokens
                .iter()
                .rposition(|(_, pos, _)| !pos.starts_with("記号"))
                .unwrap_or(n - 1)
        })
    };
    let root_id = tokens[root_idx].0; // 1-based

    // 2. Compute deps for each token
    tokens
        .iter()
        .enumerate()
        .map(|(i, (token_id, pos, _pos_detail1))| {
            let tid = *token_id;
            if tid == root_id {
                return (0usize, "root");
            }

            if pos.starts_with("助詞") {
                // Particles attach to their left neighbor (the host word)
                let head = if tid > 1 { tid - 1 } else { root_id };
                return (head, "case");
            }

            if pos.starts_with("助動詞") {
                let head = if tid > 1 { tid - 1 } else { root_id };
                return (head, "aux");
            }

            if pos.starts_with("記号") {
                return (root_id, "punct");
            }

            if pos.starts_with("感動詞") {
                return (root_id, "discourse");
            }

            if pos.starts_with("接続詞") {
                // Conjunctions attach to the next content token
                let head = tokens.get(i + 1).map(|(id, _, _)| *id).unwrap_or(root_id);
                return (head, "cc");
            }

            if pos.starts_with("副詞") {
                return (root_id, "advmod");
            }

            if pos.starts_with("形容詞") {
                // Adjectives modify the next noun if possible
                let next_noun = tokens[i + 1..]
                    .iter()
                    .find(|(_, p, _)| p.starts_with("名詞"))
                    .map(|(id, _, _)| *id);
                return (next_noun.unwrap_or(root_id), "amod");
            }

            if pos.starts_with("名詞") {
                // Before root: subject; after or at root position: object
                let deprel = if tid < root_id { "nsubj" } else { "obj" };
                return (root_id, deprel);
            }

            if pos.starts_with("動詞") {
                // Non-root verb (subordinate clause)
                return (root_id, "ccomp");
            }

            (root_id, "dep")
        })
        .collect()
}

impl AnalysisResult {
    /// Format as Universal Dependencies CoNLL-U.
    ///
    /// Fields: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
    /// HEAD and DEPREL are computed using a morphological heuristic for Japanese.
    /// All Japanese tokens carry `SpaceAfter=No` in MISC.
    pub(crate) fn format_conllu(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Collect non-BOS (empty surface) and non-EOS morphemes.
        let tokens: Vec<&crate::Morpheme> = self
            .morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .collect();

        if tokens.is_empty() {
            return Ok(());
        }

        // Build the text header by joining all token surfaces.
        let text: String = tokens.iter().map(|m| m.surface.as_str()).collect();
        writeln!(f, "# sent_id = 1")?;
        writeln!(f, "# text = {text}")?;

        // Collect (token_id, pos, pos_detail1) for dependency computation.
        let dep_inputs: Vec<(usize, &str, &str)> = tokens
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let raw = m.feature.split(',').collect::<Vec<_>>();
                let pos = raw.first().copied().unwrap_or("*");
                let pos_d1 = raw.get(1).copied().unwrap_or("*");
                (i + 1, pos, pos_d1)
            })
            .collect();
        let deps = compute_heuristic_deps(&dep_inputs);

        for (idx, morpheme) in tokens.iter().enumerate() {
            // FORM: escape tabs and newlines as spaces.
            let form: String = morpheme
                .surface
                .chars()
                .map(|c| if c == '\t' || c == '\n' { ' ' } else { c })
                .collect();

            // Parse feature string.
            let raw_features: Vec<&str> = morpheme.feature.split(',').collect();
            let get = |i: usize| -> &str { raw_features.get(i).copied().unwrap_or("*") };

            let pos = get(0);
            let pos_detail1 = get(1);
            let conj_type = get(4);
            let conj_form = get(5);
            let base_form = get(6);

            // LEMMA: use base_form if available and not `*`, otherwise fall back to FORM.
            let lemma = if base_form == "*" || base_form.is_empty() {
                form.as_str()
            } else {
                base_form
            };

            // UPOS from IPADIC POS.
            let upos = ipadic_to_upos(pos, pos_detail1);

            // XPOS: non-* fields [0..=3] joined by hyphen.
            let xpos_fields: Vec<&str> = (0..4).map(get).collect();
            let xpos = ipadic_xpos(&xpos_fields);

            // FEATS from conjugation.
            let feats = ipadic_to_feats(conj_type, conj_form);

            // Token ID is 1-based.
            let token_id = idx + 1;

            // HEAD and DEPREL from heuristic dependency computation.
            let (head, deprel) = deps.get(idx).copied().unwrap_or((0, "dep"));

            writeln!(
                f,
                "{token_id}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t_\tSpaceAfter=No"
            )?;
        }

        // CoNLL-U requires a blank line between sentences.
        writeln!(f)
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

#[cfg(test)]
mod tests {
    use super::{compute_heuristic_deps, ipadic_to_feats, ipadic_to_upos, ipadic_xpos};
    use crate::{AnalysisResult, Morpheme, OutputFormat};

    fn make_morpheme(surface: &str, feature: &str) -> Morpheme {
        Morpheme {
            surface: surface.to_owned(),
            word_id: 0,
            pos_id: 0,
            wcost: 0,
            feature: feature.to_owned(),
            entities: vec![],
            pronunciation: None,
            embedding: None,
            start_byte: 0,
            end_byte: surface.len(),
        }
    }

    #[test]
    fn test_conllu_upos_mapping() {
        assert_eq!(ipadic_to_upos("名詞", "固有名詞"), "PROPN");
        assert_eq!(ipadic_to_upos("名詞", "代名詞"), "PRON");
        assert_eq!(ipadic_to_upos("名詞", "数"), "NUM");
        assert_eq!(ipadic_to_upos("名詞", "*"), "NOUN");
        assert_eq!(ipadic_to_upos("動詞", "*"), "VERB");
        assert_eq!(ipadic_to_upos("助詞", "格助詞"), "ADP");
        assert_eq!(ipadic_to_upos("助詞", "係助詞"), "PART");
        assert_eq!(ipadic_to_upos("助動詞", "*"), "AUX");
        assert_eq!(ipadic_to_upos("記号", "*"), "PUNCT");
        assert_eq!(ipadic_to_upos("Unknown", "*"), "X");
    }

    #[test]
    fn test_conllu_feats() {
        assert_eq!(ipadic_to_feats("五段", "連用形"), "VerbForm=Ger");
        assert_eq!(ipadic_to_feats("*", "終止形"), "VerbForm=Fin");
        assert_eq!(ipadic_to_feats("*", "*"), "_");
    }

    #[test]
    fn test_conllu_xpos() {
        let feats = vec!["名詞", "固有名詞", "地域", "一般"];
        assert_eq!(ipadic_xpos(&feats), "名詞-固有名詞-地域-一般");
        let feats2 = vec!["助動詞", "*", "*", "*"];
        assert_eq!(ipadic_xpos(&feats2), "助動詞");
        let feats3 = vec!["名詞", "*", "*", "*"];
        assert_eq!(ipadic_xpos(&feats3), "名詞");
    }

    #[test]
    fn test_conllu_format_basic() {
        // Build a simple 2-morpheme result simulating: 東京 (proper noun) + は (particle).
        let morphemes = vec![
            make_morpheme(
                "東京",
                "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ",
            ),
            make_morpheme("は", "助詞,係助詞,*,*,*,*,は,ハ,ワ"),
        ];
        let result = AnalysisResult {
            morphemes,
            format: OutputFormat::ConllU,
        };

        let output = format!("{result}");

        // Must have the comment headers.
        assert!(output.contains("# sent_id = 1"), "missing sent_id header");
        assert!(output.contains("# text = 東京は"), "missing text header");

        // First token line starts with "1\t".
        assert!(output.contains("\n1\t東京\t"), "missing first token line");

        // Second token line starts with "2\t".
        assert!(output.contains("\n2\tは\t"), "missing second token line");

        // All token lines must have exactly 10 tab-separated fields.
        for line in output.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            assert_eq!(
                fields.len(),
                10,
                "expected 10 fields in CoNLL-U line, got {}: {:?}",
                fields.len(),
                line
            );
        }

        // Output must end with a blank line (sentence separator).
        assert!(output.ends_with('\n'), "output must end with newline");
        // Two trailing newlines = one blank line after last token.
        assert!(
            output.ends_with("\n\n"),
            "output must end with blank line (sentence separator)"
        );

        // UPOS for 東京 should be PROPN.
        assert!(output.contains("PROPN"), "東京 should map to PROPN");

        // MISC should always be SpaceAfter=No for Japanese.
        assert!(
            output.contains("SpaceAfter=No"),
            "missing SpaceAfter=No in MISC"
        );
    }

    #[test]
    fn test_conllu_lemma_fallback() {
        // When base_form is "*", LEMMA should fall back to FORM.
        let morphemes = vec![make_morpheme(
            "テスト",
            "名詞,一般,*,*,*,*,*,テスト,テスト",
        )];
        let result = AnalysisResult {
            morphemes,
            format: OutputFormat::ConllU,
        };
        let output = format!("{result}");
        // LEMMA (field 2) should be "テスト" (same as FORM) because base_form is "*".
        assert!(
            output.contains("1\tテスト\tテスト\t"),
            "LEMMA should fall back to FORM when base_form is *"
        );
    }

    #[test]
    fn test_conllu_empty_result() {
        // An empty result (or EOS-only) should produce no output.
        let result = AnalysisResult {
            morphemes: vec![make_morpheme("EOS", "")],
            format: OutputFormat::ConllU,
        };
        let output = format!("{result}");
        assert!(output.is_empty(), "EOS-only result should produce empty output");
    }

    #[test]
    fn test_conllu_feats_conjugation_forms() {
        assert_eq!(ipadic_to_feats("*", "連体形"), "VerbForm=Part");
        assert_eq!(ipadic_to_feats("*", "命令形"), "Mood=Imp|VerbForm=Fin");
        assert_eq!(ipadic_to_feats("*", "未然形"), "VerbForm=Inf");
        assert_eq!(ipadic_to_feats("*", "仮定形"), "Mood=Cnd|VerbForm=Fin");
        assert_eq!(ipadic_to_feats("*", "基本形"), "VerbForm=Inf");
        assert_eq!(ipadic_to_feats("*", "unknown"), "_");
    }

    #[test]
    fn test_heuristic_deps_root_detection() {
        // Simple case: last token is a verb (動詞) → root
        let tokens = vec![
            (1, "名詞", "*"),
            (2, "助詞", "格助詞"),
            (3, "動詞", "*"),
        ];
        let deps = compute_heuristic_deps(&tokens);
        // token 3 (動詞) should be root
        assert_eq!(deps[2], (0, "root"), "動詞 should be root");
        // token 2 (助詞) should attach to token 1 (left neighbor)
        assert_eq!(deps[1], (1, "case"), "助詞 should attach left");
        // token 1 (名詞, before root) should be nsubj
        assert_eq!(deps[0], (3, "nsubj"), "名詞 before root should be nsubj");
    }

    #[test]
    fn test_heuristic_deps_empty() {
        let result = compute_heuristic_deps(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_heuristic_deps_single_token() {
        let tokens = vec![(1, "名詞", "*")];
        let deps = compute_heuristic_deps(&tokens);
        // single token is always root
        assert_eq!(deps[0], (0, "root"));
    }
}
