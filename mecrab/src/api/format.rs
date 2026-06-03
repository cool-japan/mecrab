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

    /// Format all non-empty morphemes using a MeCab-compatible node-format template.
    ///
    /// The template is applied once per morpheme (BOS/EOS morphemes with an empty
    /// surface or the literal surface `"EOS"` are skipped for MeCab compatibility).
    ///
    /// # Supported placeholders
    ///
    /// | Placeholder | Meaning |
    /// |-------------|---------|
    /// | `%m`        | Surface form |
    /// | `%H`        | Full IPADIC feature string |
    /// | `%f[n]`     | n-th comma-separated feature field (0-based); `*` when absent |
    /// | `%ps`       | Start byte position in the input text |
    /// | `%pS`       | Same as `%ps` (MeCab ASCII-mode preceding-space alias) |
    /// | `%pe`       | End byte position in the input text |
    /// | `%phl`      | Left-context id (`pos_id` cast to u32, matching MeCab convention) |
    /// | `%phr`      | Right-context id (same value as `%phl` for IPADIC) |
    /// | `%c`        | Word cost (`wcost`) |
    /// | `\n`        | Newline |
    /// | `\t`        | Tab |
    /// | `%%`        | Literal `%` |
    ///
    /// Unknown placeholders (e.g. `%z`) are passed through literally as MeCab does.
    ///
    /// # Example
    ///
    /// ```text
    /// result.format_with_template("%m,%f[0]\n")
    /// // → "東京,名詞\nは,助詞\n"
    /// ```
    pub fn format_with_template(&self, template: &str) -> String {
        // Skip BOS/EOS entries: empty surface or the literal string "EOS".
        let content_morphemes: Vec<&crate::Morpheme> = self
            .morphemes
            .iter()
            .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
            .collect();

        if content_morphemes.is_empty() {
            return String::new();
        }

        let mut output = String::with_capacity(template.len() * content_morphemes.len());

        for morpheme in content_morphemes {
            expand_template(template, morpheme, &mut output);
        }

        output
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

// ── Node-format template engine ───────────────────────────────────────────────

/// Expand `template` for one `morpheme`, appending the result to `out`.
///
/// This is the core state machine used by [`AnalysisResult::format_with_template`].
/// It processes the template string character by character and handles all
/// recognised `%`- and `\`-escape sequences.
#[allow(clippy::too_many_lines, clippy::single_match_else)]
fn expand_template(template: &str, morpheme: &crate::Morpheme, out: &mut String) {
    let bytes = template.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        match bytes[i] {
            // ── `\`-escape sequences ─────────────────────────────────────
            b'\\' => {
                if i + 1 < len {
                    match bytes[i + 1] {
                        b'n' => {
                            out.push('\n');
                            i += 2;
                        }
                        b't' => {
                            out.push('\t');
                            i += 2;
                        }
                        other => {
                            // Unknown escape: pass through literally.
                            out.push('\\');
                            out.push(other as char);
                            i += 2;
                        }
                    }
                } else {
                    // Trailing backslash at end of template.
                    out.push('\\');
                    i += 1;
                }
            }

            // ── `%`-format placeholders ──────────────────────────────────
            b'%' => {
                if i + 1 >= len {
                    // Trailing `%` at end of template.
                    out.push('%');
                    i += 1;
                    continue;
                }

                match bytes[i + 1] {
                    // `%%` → literal `%`
                    b'%' => {
                        out.push('%');
                        i += 2;
                    }

                    // `%m` → surface form
                    b'm' => {
                        out.push_str(&morpheme.surface);
                        i += 2;
                    }

                    // `%H` → full feature string
                    b'H' => {
                        out.push_str(&morpheme.feature);
                        i += 2;
                    }

                    // `%f[n]` → n-th feature field
                    b'f' => {
                        // Expect `[`, then digits, then `]`.
                        if i + 2 < len && bytes[i + 2] == b'[' {
                            // Find the closing `]`.
                            let bracket_start = i + 3;
                            match bytes[bracket_start..].iter().position(|&b| b == b']') {
                                Some(bracket_len) => {
                                    let digit_bytes =
                                        &bytes[bracket_start..bracket_start + bracket_len];
                                    // Parse the field index (ASCII digits only).
                                    let n_opt = std::str::from_utf8(digit_bytes)
                                        .ok()
                                        .and_then(|s| s.parse::<usize>().ok());
                                    let field_val = n_opt.and_then(|n| {
                                        let v = morpheme.feature.split(',').nth(n)?;
                                        if v == "*" { None } else { Some(v.to_owned()) }
                                    });
                                    out.push_str(field_val.as_deref().unwrap_or("*"));
                                    // Advance past `%f[n]`
                                    i = bracket_start + bracket_len + 1;
                                }
                                None => {
                                    // No closing `]` — pass through literally.
                                    out.push('%');
                                    out.push('f');
                                    i += 2;
                                }
                            }
                        } else {
                            // `%f` not followed by `[` — pass through literally.
                            out.push('%');
                            out.push('f');
                            i += 2;
                        }
                    }

                    // `%p…` — positional / lattice attributes
                    b'p' => {
                        if i + 2 >= len {
                            out.push('%');
                            out.push('p');
                            i += 2;
                            continue;
                        }
                        match bytes[i + 2] {
                            // `%ps` or `%pS` — start byte position
                            b's' | b'S' => {
                                out.push_str(&morpheme.start_byte.to_string());
                                i += 3;
                            }
                            // `%pe` — end byte position
                            b'e' => {
                                out.push_str(&morpheme.end_byte.to_string());
                                i += 3;
                            }
                            // `%ph…` — left/right context ids
                            b'h' => {
                                if i + 3 < len {
                                    match bytes[i + 3] {
                                        b'l' => {
                                            out.push_str(&(morpheme.pos_id as u32).to_string());
                                            i += 4;
                                        }
                                        b'r' => {
                                            // Right-context id: same value for IPADIC.
                                            out.push_str(&(morpheme.pos_id as u32).to_string());
                                            i += 4;
                                        }
                                        _ => {
                                            out.push('%');
                                            out.push('p');
                                            out.push('h');
                                            i += 3;
                                        }
                                    }
                                } else {
                                    out.push('%');
                                    out.push('p');
                                    out.push('h');
                                    i += 3;
                                }
                            }
                            _ => {
                                // Unknown `%p?` — pass through.
                                out.push('%');
                                out.push('p');
                                i += 2;
                            }
                        }
                    }

                    // `%c` — word cost
                    b'c' => {
                        out.push_str(&morpheme.wcost.to_string());
                        i += 2;
                    }

                    // Unknown placeholder — pass through literally (MeCab behaviour).
                    other => {
                        out.push('%');
                        out.push(other as char);
                        i += 2;
                    }
                }
            }

            // ── Ordinary byte ────────────────────────────────────────────
            b => {
                // Safety: we have verified `b` is a valid ASCII byte or the start
                // of a multi-byte UTF-8 sequence.  Since we only match on single-byte
                // patterns above, any non-ASCII leading byte falls here.  We push it
                // back as a char obtained from the original `&str` slice.
                //
                // To avoid breaking multi-byte sequences we consume the full UTF-8
                // character at position `i`.
                //
                // `template` is a valid `&str`; unwrap is safe here.
                let ch = template[i..].chars().next().unwrap();
                out.push(ch);
                i += ch.len_utf8();
                let _ = b; // suppress unused-variable lint
            }
        }
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
/// - 名詞 (noun): determined by the immediately following case particle surface:
///   - が → `nsubj` (nominative subject)
///   - を → `obj`   (accusative object)
///   - に/へ/で/から/まで → `obl` (oblique)
///   - の → `nmod`  (genitive modifier; attaches to next noun rather than root)
///   - no particle → positional fallback (before root → `nsubj`, else `obj`)
/// - Other verbs (non-root): `ccomp`
/// - Everything else: `dep`
///
/// # Token tuple layout
///
/// Each element of `tokens` is `(token_id_1based, pos, pos_detail1, surface)`.
/// The surface field is used only for case-particle lookahead on nouns.
///
/// Returns a `Vec<(usize, &'static str)>` parallel to `tokens`:
/// each entry is `(head_id_1based, deprel)` where `head_id=0` means root.
#[allow(dead_code)]
fn compute_heuristic_deps(
    tokens: &[(usize, &str, &str, &str)], // (token_id_1based, pos, pos_detail1, surface)
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
        let verb_pos = tokens
            .iter()
            .rposition(|(_, pos, _, _)| pos.starts_with("動詞"));
        let aux_pos = tokens
            .iter()
            .rposition(|(_, pos, _, _)| pos.starts_with("助動詞"));
        // prefer verb over aux; both over fallback
        verb_pos.or(aux_pos).unwrap_or_else(|| {
            tokens
                .iter()
                .rposition(|(_, pos, _, _)| !pos.starts_with("記号"))
                .unwrap_or(n - 1)
        })
    };
    let root_id = tokens[root_idx].0; // 1-based

    // 2. Compute deps for each token
    tokens
        .iter()
        .enumerate()
        .map(|(i, (token_id, pos, _pos_detail1, _surface))| {
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
                let head = tokens
                    .get(i + 1)
                    .map(|(id, _, _, _)| *id)
                    .unwrap_or(root_id);
                return (head, "cc");
            }

            if pos.starts_with("副詞") {
                return (root_id, "advmod");
            }

            if pos.starts_with("形容詞") {
                // Adjectives modify the next noun if possible
                let next_noun = tokens[i + 1..]
                    .iter()
                    .find(|(_, p, _, _)| p.starts_with("名詞"))
                    .map(|(id, _, _, _)| *id);
                return (next_noun.unwrap_or(root_id), "amod");
            }

            if pos.starts_with("名詞") {
                // Find the immediately following case particle, skipping
                // punctuation tokens.  We look at the next token's POS; if it
                // is a particle we inspect its surface to determine the
                // grammatical relation of the noun.
                let following_particle: Option<&str> = tokens[i + 1..]
                    .iter()
                    .find(|(_, p, _, _)| p.starts_with("助詞") || p.starts_with("記号"))
                    .and_then(|(_, p, _, surf)| {
                        if p.starts_with("助詞") {
                            Some(*surf)
                        } else {
                            None // punctuation — no case particle found
                        }
                    });

                let deprel: &'static str = match following_particle {
                    Some("が") => "nsubj",
                    Some("を") => "obj",
                    Some("に" | "へ" | "で" | "から" | "まで") => "obl",
                    Some("の") => "nmod",
                    // Positional fallback when no case particle is present
                    _ if tid < root_id => "nsubj",
                    _ => "obj",
                };

                // For genitive "の": attach to the next noun rather than root
                let head = if following_particle == Some("の") {
                    tokens[i + 1..]
                        .iter()
                        .find(|(_, p, _, _)| p.starts_with("名詞"))
                        .map(|(id, _, _, _)| *id)
                        .unwrap_or(root_id)
                } else {
                    root_id
                };

                return (head, deprel);
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

        // Collect (token_id, pos, pos_detail1, surface) for dependency computation.
        // The surface field enables case-particle lookahead for accurate noun labeling.
        let dep_inputs: Vec<(usize, &str, &str, &str)> = tokens
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let raw = m.feature.split(',').collect::<Vec<_>>();
                let pos = raw.first().copied().unwrap_or("*");
                let pos_d1 = raw.get(1).copied().unwrap_or("*");
                (i + 1, pos, pos_d1, m.surface.as_str())
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
    use super::{
        compute_heuristic_deps, expand_template, ipadic_to_feats, ipadic_to_upos, ipadic_xpos,
    };
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
        let morphemes = vec![make_morpheme("テスト", "名詞,一般,*,*,*,*,*,テスト,テスト")];
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
        assert!(
            output.is_empty(),
            "EOS-only result should produce empty output"
        );
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
        // Simple case: last token is a verb (動詞) → root.
        // 名詞 "誰か" followed by が → nsubj via particle lookahead.
        let tokens = vec![
            (1, "名詞", "*", "誰か"),
            (2, "助詞", "格助詞", "が"),
            (3, "動詞", "*", "走る"),
        ];
        let deps = compute_heuristic_deps(&tokens);
        // token 3 (動詞) should be root
        assert_eq!(deps[2], (0, "root"), "動詞 should be root");
        // token 2 (助詞) should attach to token 1 (left neighbor)
        assert_eq!(deps[1], (1, "case"), "助詞 should attach left");
        // token 1 (名詞 followed by が) should be nsubj via particle lookahead
        assert_eq!(
            deps[0],
            (3, "nsubj"),
            "名詞 followed by が must be nsubj (particle lookahead)"
        );
    }

    #[test]
    fn test_heuristic_deps_empty() {
        let result = compute_heuristic_deps(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_heuristic_deps_single_token() {
        let tokens = vec![(1, "名詞", "*", "テスト")];
        let deps = compute_heuristic_deps(&tokens);
        // single token is always root
        assert_eq!(deps[0], (0, "root"));
    }

    /// Fix 2: case-particle lookahead assigns correct deprel based on surface.
    ///
    /// Sentence structure simulated:
    ///   太郎が  (nsubj)  花子を  (obj)  公園で  (obl)  会った (root=verb)
    #[test]
    fn test_conllu_case_particle_labeling() {
        // Build a synthetic AnalysisResult:
        //   太郎 (名詞固有名詞) が (助詞格助詞) 花子 (名詞固有名詞) を (助詞格助詞)
        //   公園 (名詞一般)      で (助詞格助詞) 会っ (動詞) た (助動詞)
        let morphemes = vec![
            make_morpheme("太郎", "名詞,固有名詞,人名,一般,*,*,太郎,タロウ,タロウ"),
            make_morpheme("が", "助詞,格助詞,一般,*,*,*,が,ガ,ガ"),
            make_morpheme("花子", "名詞,固有名詞,人名,一般,*,*,花子,ハナコ,ハナコ"),
            make_morpheme("を", "助詞,格助詞,一般,*,*,*,を,ヲ,ヲ"),
            make_morpheme("公園", "名詞,一般,*,*,*,*,公園,コウエン,コウエン"),
            make_morpheme("で", "助詞,格助詞,一般,*,*,*,で,デ,デ"),
            make_morpheme(
                "会っ",
                "動詞,自立,*,*,五段・ワ行促音便,連用タ接続,会う,アッ,アッ",
            ),
            make_morpheme("た", "助動詞,*,*,*,特殊・タ,基本形,た,タ,タ"),
        ];
        let result = AnalysisResult {
            morphemes,
            format: OutputFormat::ConllU,
        };

        let output = format!("{result}");

        // Exercise compute_heuristic_deps directly with surface-aware tuples.
        let tokens: Vec<(usize, &str, &str, &str)> = vec![
            (1, "名詞", "固有名詞", "太郎"),
            (2, "助詞", "格助詞", "が"),
            (3, "名詞", "固有名詞", "花子"),
            (4, "助詞", "格助詞", "を"),
            (5, "名詞", "一般", "公園"),
            (6, "助詞", "格助詞", "で"),
            (7, "動詞", "自立", "会っ"),
            (8, "助動詞", "*", "た"),
        ];
        let deps = compute_heuristic_deps(&tokens);

        // 動詞 (token 7) is root
        assert_eq!(deps[6], (0, "root"), "動詞 must be root");

        // 太郎 followed by が → nsubj
        assert_eq!(
            deps[0],
            (7, "nsubj"),
            "太郎+が must be nsubj (nominative particle)"
        );

        // 花子 followed by を → obj
        assert_eq!(
            deps[2],
            (7, "obj"),
            "花子+を must be obj (accusative particle)"
        );

        // 公園 followed by で → obl
        assert_eq!(
            deps[4],
            (7, "obl"),
            "公園+で must be obl (oblique particle)"
        );

        // CoNLL-U output must contain the correct deprel labels
        assert!(
            output.contains("nsubj"),
            "CoNLL-U output must contain nsubj"
        );
        assert!(output.contains("obj"), "CoNLL-U output must contain obj");

        // All 8 token lines must be present
        let token_line_count = output
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .count();
        assert_eq!(
            token_line_count, 8,
            "expected 8 token lines, got {}",
            token_line_count
        );
    }

    // ── format_with_template tests ────────────────────────────────────────────

    /// Build an `AnalysisResult` with custom morphemes for template tests.
    fn make_result(morphemes: Vec<Morpheme>) -> AnalysisResult {
        AnalysisResult {
            morphemes,
            format: OutputFormat::Default,
        }
    }

    /// Make a morpheme with explicit byte positions and cost.
    fn make_morpheme_at(
        surface: &str,
        feature: &str,
        start: usize,
        pos_id: u16,
        wcost: i16,
    ) -> Morpheme {
        Morpheme {
            surface: surface.to_owned(),
            word_id: 0,
            pos_id,
            wcost,
            feature: feature.to_owned(),
            entities: vec![],
            pronunciation: None,
            embedding: None,
            start_byte: start,
            end_byte: start + surface.len(),
        }
    }

    #[test]
    fn test_template_surface_and_feature_field() {
        // `%m,%f[0]\n` must produce one "surface,pos\n" line per morpheme.
        let morphemes = vec![
            make_morpheme(
                "東京",
                "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ",
            ),
            make_morpheme("は", "助詞,係助詞,*,*,*,*,は,ハ,ワ"),
        ];
        let result = make_result(morphemes);
        let out = result.format_with_template("%m,%f[0]\n");
        assert_eq!(out, "東京,名詞\nは,助詞\n");
    }

    #[test]
    fn test_template_full_feature_string() {
        // `%H` must emit the raw feature string unchanged.
        let feature = "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ";
        let morphemes = vec![make_morpheme("東京", feature)];
        let result = make_result(morphemes);
        let out = result.format_with_template("%H");
        assert_eq!(out, feature);
    }

    #[test]
    fn test_template_escaped_percent() {
        // `%%` must produce a single literal `%`.
        let morphemes = vec![make_morpheme("X", "名詞,*,*,*,*,*,*,*,*")];
        let result = make_result(morphemes);
        let out = result.format_with_template("%%");
        assert_eq!(out, "%");
    }

    #[test]
    fn test_template_tab_escape() {
        // `\t` must produce a tab character.
        let morphemes = vec![make_morpheme("X", "名詞,*,*,*,*,*,*,*,*")];
        let result = make_result(morphemes);
        let out = result.format_with_template("%m\t%f[0]");
        assert_eq!(out, "X\t名詞");
    }

    #[test]
    fn test_template_newline_escape() {
        // `\n` must produce a newline character.
        let morphemes = vec![make_morpheme("X", "名詞,*,*,*,*,*,*,*,*")];
        let result = make_result(morphemes);
        let out = result.format_with_template("%m\n");
        assert_eq!(out, "X\n");
    }

    #[test]
    fn test_template_unknown_placeholder_passthrough() {
        // Unknown `%z` must be emitted literally (MeCab behavior).
        let morphemes = vec![make_morpheme("X", "名詞,*,*,*,*,*,*,*,*")];
        let result = make_result(morphemes);
        let out = result.format_with_template("%z");
        assert_eq!(out, "%z");
    }

    #[test]
    fn test_template_feature_field_out_of_range_produces_star() {
        // `%f[99]` on a 9-field feature string must produce `*`.
        let morphemes = vec![make_morpheme(
            "東京",
            "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ",
        )];
        let result = make_result(morphemes);
        let out = result.format_with_template("%f[99]");
        assert_eq!(out, "*");
    }

    #[test]
    fn test_template_feature_field_star_value_produces_star() {
        // A field equal to `*` in the feature string must render as `*`.
        // Field 4 (活用型) is `*` for a proper noun.
        let morphemes = vec![make_morpheme(
            "東京",
            "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ",
        )];
        let result = make_result(morphemes);
        let out = result.format_with_template("%f[4]");
        assert_eq!(out, "*");
    }

    #[test]
    fn test_template_empty_morpheme_list_returns_empty_string() {
        // An `AnalysisResult` with no content morphemes must return `""`.
        let result = make_result(vec![]);
        let out = result.format_with_template("%m\n");
        assert_eq!(out, "");
    }

    #[test]
    fn test_template_eos_only_result_returns_empty_string() {
        // A result containing only the EOS sentinel must return `""`.
        let result = make_result(vec![make_morpheme("EOS", "")]);
        let out = result.format_with_template("%m\n");
        assert_eq!(out, "");
    }

    #[test]
    fn test_template_byte_positions() {
        // `%ps` / `%pe` must emit the start/end byte offsets.
        // "東京" is 6 bytes in UTF-8, start=3 → end=9.
        let morphemes = vec![make_morpheme_at("東京", "名詞,*,*,*,*,*,*,*,*", 3, 0, 0)];
        let result = make_result(morphemes);
        let out = result.format_with_template("%ps-%pe");
        assert_eq!(out, "3-9");
    }

    #[test]
    fn test_template_ps_alias_equals_ps() {
        // `%pS` (uppercase S) must produce the same output as `%ps`.
        let morphemes = vec![make_morpheme_at("A", "名詞,*,*,*,*,*,*,*,*", 10, 0, 0)];
        let result = make_result(morphemes);
        let with_lower = result.format_with_template("%ps");
        let with_upper = result.format_with_template("%pS");
        assert_eq!(with_lower, with_upper);
    }

    #[test]
    fn test_template_word_cost() {
        // `%c` must emit the wcost field.
        let morphemes = vec![make_morpheme_at("X", "名詞,*,*,*,*,*,*,*,*", 0, 0, -42)];
        let result = make_result(morphemes);
        let out = result.format_with_template("%c");
        assert_eq!(out, "-42");
    }

    #[test]
    fn test_template_phl_and_phr() {
        // `%phl` / `%phr` must emit the pos_id as u32.
        let morphemes = vec![make_morpheme_at("X", "名詞,*,*,*,*,*,*,*,*", 0, 7, 0)];
        let result = make_result(morphemes);
        let out_l = result.format_with_template("%phl");
        let out_r = result.format_with_template("%phr");
        assert_eq!(out_l, "7");
        assert_eq!(out_r, "7");
    }

    #[test]
    fn test_template_two_morphemes_produces_two_lines() {
        // A two-morpheme result with `%m\n` must yield two newline-terminated lines.
        let morphemes = vec![
            make_morpheme(
                "東京",
                "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ",
            ),
            make_morpheme("は", "助詞,係助詞,*,*,*,*,は,ハ,ワ"),
        ];
        let result = make_result(morphemes);
        let out = result.format_with_template("%m\n");
        assert_eq!(out, "東京\nは\n");
    }

    #[test]
    fn test_template_expand_template_directly() {
        // Test the internal `expand_template` function with a rich combined template.
        let morpheme = make_morpheme_at(
            "走る",
            "動詞,自立,*,*,五段・ラ行,基本形,走る,ハシル,ハシル",
            0,
            5,
            100,
        );
        let mut out = String::new();
        expand_template("%m\t%f[0]\t%f[6]\t%c\n", &morpheme, &mut out);
        assert_eq!(out, "走る\t動詞\t走る\t100\n");
    }
}
