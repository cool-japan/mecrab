//! DBpedia NTriples processor for Japanese entity data.
//!
//! Parses DBpedia NTriples dumps (`.nt` files, optionally gzipped)
//! and adds surface→URI mappings to the [`WikidataIndex`].
//!
//! # Supported triples
//! - `rdfs:label` — provides surface form (language-tagged `@ja`)
//! - `dbo:abstract` — provides confidence signal (abstract presence)
//! - `foaf:name` — alternative surface form

use crate::wikidata::WikidataIndex;
use crate::{BuildError, Result};
use flate2::read::GzDecoder;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::Path;

// ─────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────

/// Statistics from processing a DBpedia NTriples dump.
#[derive(Debug, Default)]
pub struct DBpediaStats {
    /// Total NTriple lines processed (excluding comments and blanks)
    pub triples_processed: u64,
    /// Number of Japanese-language label/name triples found
    pub labels_found: u64,
    /// Number of Japanese-language abstract triples found
    pub abstracts_found: u64,
    /// Number of surface→URI pairs added to the index
    pub surfaces_added: u64,
}

// ─────────────────────────────────────────────────────────────
// Internal representation
// ─────────────────────────────────────────────────────────────

/// Parsed NTriple components (subject IRI, predicate IRI, object, optional lang-tag).
#[derive(Debug)]
struct Triple {
    subject: String,
    predicate: String,
    object: String,
    lang: Option<String>,
}

// ─────────────────────────────────────────────────────────────
// DBpediaProcessor
// ─────────────────────────────────────────────────────────────

/// DBpedia NTriples dump processor.
///
/// Reads a `.nt` (or `.nt.gz`) file and populates a [`WikidataIndex`] with
/// surface→URI mappings derived from `rdfs:label`, `foaf:name`, and
/// `dbo:abstract` predicates for Japanese-language literals.
///
/// # Confidence model
///
/// Every label entry starts at `base_confidence` (default `0.6`).
/// If the same URI also has a Japanese-language abstract, the confidence
/// is boosted by `abstract_boost` (default `0.2`), yielding `0.8` at most.
/// Values are clamped to `[0.05, 1.0]` before insertion.
pub struct DBpediaProcessor {
    /// Minimum confidence assigned to label entries without an abstract.
    pub base_confidence: f32,
    /// Additional confidence for entries that have a `dbo:abstract`.
    pub abstract_boost: f32,
}

impl Default for DBpediaProcessor {
    fn default() -> Self {
        Self {
            base_confidence: 0.6,
            abstract_boost: 0.2,
        }
    }
}

impl DBpediaProcessor {
    /// Create a new processor with default confidence values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a DBpedia NTriples file and insert results into `index`.
    ///
    /// Handles both plain `.nt` and gzip-compressed `.nt.gz` (or `.gz`) files.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Io`] on file-system errors or [`BuildError::Io`]
    /// (wrapped `std::io::Error`) on line-read failures.
    pub fn process_dump(
        &self,
        path: &Path,
        index: &mut WikidataIndex,
        verbose: bool,
    ) -> Result<DBpediaStats> {
        let mut stats = DBpediaStats::default();

        let file = std::fs::File::open(path).map_err(BuildError::Io)?;

        let reader: Box<dyn BufRead> = if path.extension().is_some_and(|e| e == "gz") {
            Box::new(BufReader::with_capacity(1 << 20, GzDecoder::new(file)))
        } else {
            Box::new(BufReader::with_capacity(1 << 20, file))
        };

        // uri → list of Japanese surface forms
        let mut labels: HashMap<String, Vec<String>> = HashMap::new();
        // set of URIs that have at least one Japanese abstract
        let mut has_abstract: HashSet<String> = HashSet::new();

        for line_result in reader.lines() {
            let line = line_result.map_err(BuildError::Io)?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            stats.triples_processed += 1;

            if let Some(triple) = parse_ntriple(trimmed) {
                let is_japanese = triple.lang.as_deref() == Some("ja");

                if is_label_predicate(&triple.predicate) && is_japanese {
                    stats.labels_found += 1;
                    labels
                        .entry(triple.subject.clone())
                        .or_default()
                        .push(triple.object.clone());
                } else if is_abstract_predicate(&triple.predicate) && is_japanese {
                    stats.abstracts_found += 1;
                    has_abstract.insert(triple.subject.clone());
                }
            }
        }

        // Insert collected labels into the index
        for (uri, surface_forms) in &labels {
            let confidence = if has_abstract.contains(uri) {
                self.base_confidence + self.abstract_boost
            } else {
                self.base_confidence
            }
            .clamp(0.05, 1.0);

            for surface in surface_forms {
                index.add(surface, uri, confidence);
                stats.surfaces_added += 1;
            }
        }

        if verbose {
            eprintln!(
                "DBpedia: {} triples, {} labels, {} with abstracts, {} surfaces added",
                stats.triples_processed,
                stats.labels_found,
                stats.abstracts_found,
                stats.surfaces_added,
            );
        }

        Ok(stats)
    }
}

// ─────────────────────────────────────────────────────────────
// NTriple parser
// ─────────────────────────────────────────────────────────────

/// Parse a single NTriple line into a [`Triple`].
///
/// Returns `None` for lines that don't form a valid subject–predicate–object
/// triple (both subject and predicate must be IRIs; object may be IRI or
/// string literal).
fn parse_ntriple(line: &str) -> Option<Triple> {
    // Strip trailing " ." or "."
    let line = line.trim_end_matches(" .").trim_end_matches('.');

    // Extract subject (IRI only)
    let (subject, rest) = extract_iri(line.trim())?;
    // Extract predicate (IRI only)
    let (predicate, rest) = extract_iri(rest.trim())?;
    let rest = rest.trim();

    if rest.starts_with('<') {
        let (obj_iri, _) = extract_iri(rest)?;
        Some(Triple {
            subject,
            predicate,
            object: obj_iri,
            lang: None,
        })
    } else if rest.starts_with('"') {
        let (literal, lang) = extract_literal(rest)?;
        Some(Triple {
            subject,
            predicate,
            object: literal,
            lang,
        })
    } else {
        None
    }
}

/// Extract an IRI `<...>` from the start of `s`.
///
/// Returns `(iri_content, remaining_str)`, or `None` if `s` doesn't start
/// with `<`.
fn extract_iri(s: &str) -> Option<(String, &str)> {
    if !s.starts_with('<') {
        return None;
    }
    let end = s.find('>')?;
    Some((s[1..end].to_string(), &s[end + 1..]))
}

/// Extract a string literal `"..."@lang` or `"..."^^<type>` from `s`.
///
/// Handles `\"` escapes inside the literal.
/// Returns `(literal_content, lang_tag)`, or `None` if `s` doesn't start
/// with `"`.
fn extract_literal(s: &str) -> Option<(String, Option<String>)> {
    if !s.starts_with('"') {
        return None;
    }
    // Find the closing unescaped quote
    let mut end = 1usize;
    let bytes = s.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'"' && bytes[end - 1] != b'\\' {
            break;
        }
        end += 1;
    }
    if end >= bytes.len() {
        return None;
    }

    let content = s[1..end]
        .replace("\\\"", "\"")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\\", "\\");

    let rest = &s[end + 1..];

    // Optional language tag: @xx or @xx-YY
    let lang = if let Some(after_at) = rest.strip_prefix('@') {
        let lang_end = after_at
            .find(|c: char| !c.is_alphanumeric() && c != '-')
            .unwrap_or(after_at.len());
        Some(after_at[..lang_end].to_string())
    } else {
        None
    };

    Some((content, lang))
}

// ─────────────────────────────────────────────────────────────
// Predicate helpers
// ─────────────────────────────────────────────────────────────

/// Returns `true` if `pred` is a label-style predicate
/// (`rdfs:label`, `foaf:name`, `skos:prefLabel`, etc.).
fn is_label_predicate(pred: &str) -> bool {
    pred.ends_with("label")
        || pred.ends_with("name")
        || pred.contains("foaf/0.1/name")
        || pred.contains("rdfs#label")
        || pred.contains("skos/core#prefLabel")
        || pred.contains("skos/core#altLabel")
}

/// Returns `true` if `pred` is an abstract-style predicate
/// (`dbo:abstract`, etc.).
fn is_abstract_predicate(pred: &str) -> bool {
    pred.ends_with("abstract") || pred.contains("dbpedia.org/ontology/abstract")
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_iri_basic() {
        let s = "<http://example.org/foo> rest";
        let (iri, rest) = extract_iri(s).expect("should parse");
        assert_eq!(iri, "http://example.org/foo");
        assert_eq!(rest.trim(), "rest");
    }

    #[test]
    fn test_extract_iri_no_angle_bracket() {
        assert!(extract_iri("plain text").is_none());
    }

    #[test]
    fn test_extract_literal_with_lang() {
        let s = r#""東京都"@ja ."#;
        let (lit, lang) = extract_literal(s).expect("should parse");
        assert_eq!(lit, "東京都");
        assert_eq!(lang.as_deref(), Some("ja"));
    }

    #[test]
    fn test_extract_literal_escaped_quote() {
        let s = r#""foo\"bar"@en"#;
        let (lit, lang) = extract_literal(s).expect("should parse");
        assert_eq!(lit, r#"foo"bar"#);
        assert_eq!(lang.as_deref(), Some("en"));
    }

    #[test]
    fn test_extract_literal_no_lang() {
        let s = r#""hello"^^<http://www.w3.org/2001/XMLSchema#string>"#;
        let (lit, lang) = extract_literal(s).expect("should parse");
        assert_eq!(lit, "hello");
        assert!(lang.is_none());
    }

    #[test]
    fn test_parse_ntriple_label() {
        let line = r#"<http://ja.dbpedia.org/resource/東京都> <http://www.w3.org/2000/01/rdf-schema#label> "東京都"@ja ."#;
        let triple = parse_ntriple(line).expect("should parse");
        assert_eq!(triple.subject, "http://ja.dbpedia.org/resource/東京都");
        assert_eq!(
            triple.predicate,
            "http://www.w3.org/2000/01/rdf-schema#label"
        );
        assert_eq!(triple.object, "東京都");
        assert_eq!(triple.lang.as_deref(), Some("ja"));
    }

    #[test]
    fn test_parse_ntriple_iri_object() {
        let line = r#"<http://ja.dbpedia.org/resource/東京都> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Place> ."#;
        let triple = parse_ntriple(line).expect("should parse");
        assert_eq!(triple.object, "http://dbpedia.org/ontology/Place");
        assert!(triple.lang.is_none());
    }

    #[test]
    fn test_parse_ntriple_comment_returns_none() {
        // comment lines should be filtered before calling parse_ntriple,
        // but a non-IRI subject returns None
        assert!(parse_ntriple("# comment line").is_none());
    }

    #[test]
    fn test_is_label_predicate() {
        assert!(is_label_predicate(
            "http://www.w3.org/2000/01/rdf-schema#label"
        ));
        assert!(is_label_predicate("http://xmlns.com/foaf/0.1/name"));
        assert!(!is_label_predicate("http://dbpedia.org/ontology/abstract"));
    }

    #[test]
    fn test_is_abstract_predicate() {
        assert!(is_abstract_predicate(
            "http://dbpedia.org/ontology/abstract"
        ));
        assert!(!is_abstract_predicate(
            "http://www.w3.org/2000/01/rdf-schema#label"
        ));
    }

    #[test]
    fn test_process_dump_end_to_end() {
        let nt = concat!(
            "<http://ja.dbpedia.org/resource/東京都>",
            " <http://www.w3.org/2000/01/rdf-schema#label>",
            " \"東京都\"@ja .\n",
            "<http://ja.dbpedia.org/resource/東京都>",
            " <http://dbpedia.org/ontology/abstract>",
            " \"東京都は日本の都道府県\"@ja .\n",
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_mecrab_dbpedia.nt");
        std::fs::write(&path, nt).expect("write temp file");

        let processor = DBpediaProcessor::new();
        let mut index = WikidataIndex::default();
        let stats = processor
            .process_dump(&path, &mut index, false)
            .expect("process_dump should succeed");

        assert_eq!(stats.labels_found, 1);
        assert_eq!(stats.abstracts_found, 1);
        assert_eq!(stats.surfaces_added, 1);

        let results = index.lookup("東京都");
        assert!(results.is_some(), "surface '東京都' should be in index");
        let candidates = results.expect("just checked");
        assert!(!candidates.is_empty());
        // Confidence should be base + boost = 0.8
        assert!(
            (candidates[0].1 - 0.8_f32).abs() < 1e-4,
            "confidence should be ~0.8"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_process_dump_label_only_no_abstract() {
        let nt = concat!(
            "<http://ja.dbpedia.org/resource/渋谷区>",
            " <http://www.w3.org/2000/01/rdf-schema#label>",
            " \"渋谷区\"@ja .\n",
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_mecrab_dbpedia_label_only.nt");
        std::fs::write(&path, nt).expect("write temp file");

        let processor = DBpediaProcessor::new();
        let mut index = WikidataIndex::default();
        let stats = processor
            .process_dump(&path, &mut index, false)
            .expect("process_dump should succeed");

        assert_eq!(stats.labels_found, 1);
        assert_eq!(stats.abstracts_found, 0);

        let results = index.lookup("渋谷区").expect("should be in index");
        // base_confidence only (no abstract)
        assert!(
            (results[0].1 - 0.6_f32).abs() < 1e-4,
            "confidence should be ~0.6"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_process_dump_non_japanese_ignored() {
        let nt = concat!(
            "<http://ja.dbpedia.org/resource/Tokyo>",
            " <http://www.w3.org/2000/01/rdf-schema#label>",
            " \"Tokyo\"@en .\n",
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_mecrab_dbpedia_english.nt");
        std::fs::write(&path, nt).expect("write temp file");

        let processor = DBpediaProcessor::new();
        let mut index = WikidataIndex::default();
        let stats = processor
            .process_dump(&path, &mut index, false)
            .expect("process_dump should succeed");

        // English labels should be ignored
        assert_eq!(stats.labels_found, 0);
        assert!(index.is_empty());

        std::fs::remove_file(&path).ok();
    }
}
