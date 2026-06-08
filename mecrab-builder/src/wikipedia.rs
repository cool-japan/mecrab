//! Wikipedia abstract dump processor.
//!
//! Parses the Japanese Wikipedia abstract XML dump and extracts
//! surface forms (article titles) with their Wikipedia URLs.
//!
//! The dump format is a simple XML with predictable structure,
//! so we use a line-based state machine instead of a full XML parser.
//!
//! # Usage
//! ```ignore
//! let mut processor = WikipediaProcessor::new();
//! processor.process_dump(path, &mut index, false)?;
//! ```

use crate::wikidata::WikidataIndex;
use crate::{BuildError, Result};
use oxiarc_deflate::GzipStreamDecoder;
use rayon::prelude::*;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Statistics from processing a Wikipedia abstract dump
#[derive(Debug, Default)]
pub struct WikipediaStats {
    pub articles_processed: u64,
    pub surfaces_added: u64,
    pub skipped_disambiguation: u64,
    pub skipped_no_abstract: u64,
}

/// Wikipedia abstract dump processor.
///
/// Uses a line-based state machine to parse the abstract XML.
/// This avoids a full XML parser dependency while handling the
/// well-structured Wikipedia format.
pub struct WikipediaProcessor {
    /// Minimum abstract length (in chars) to consider an article notable
    pub min_abstract_chars: usize,
    /// Whether to include disambiguation pages
    pub include_disambiguation: bool,
    /// Confidence multiplier for Wikipedia vs Wikidata (Wikipedia = more curated)
    pub confidence_boost: f32,
}

impl Default for WikipediaProcessor {
    fn default() -> Self {
        Self {
            min_abstract_chars: 20,
            include_disambiguation: false,
            confidence_boost: 1.1,
        }
    }
}

impl WikipediaProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a Wikipedia abstract XML dump (plain or .gz) into the index.
    ///
    /// Processes entries in parallel chunks for performance.
    /// The `index.add()` merges with max-confidence semantics, so
    /// Wikipedia entries safely overlap with Wikidata entries.
    pub fn process_dump(
        &self,
        path: &Path,
        index: &mut WikidataIndex,
        verbose: bool,
    ) -> Result<WikipediaStats> {
        let mut stats = WikipediaStats::default();

        // Open file (handle .gz)
        let file = std::fs::File::open(path)?;
        let reader: Box<dyn BufRead> = if path.extension().is_some_and(|e| e == "gz") {
            Box::new(BufReader::with_capacity(
                1 << 20,
                GzipStreamDecoder::new(file),
            ))
        } else {
            Box::new(BufReader::with_capacity(1 << 20, file))
        };

        // Parse articles into (title, url, abstract_text) tuples
        let articles = Self::parse_articles(reader)?;
        let total = articles.len() as u64;
        stats.articles_processed = total;

        if verbose {
            eprintln!("Wikipedia: parsed {} articles", total);
        }

        // Process in parallel
        let min_chars = self.min_abstract_chars;
        let include_disam = self.include_disambiguation;
        let boost = self.confidence_boost;

        let (entries, skipped_disambiguation, skipped_no_abstract): (Vec<_>, u64, u64) = {
            let results: Vec<_> = articles
                .par_iter()
                .map(|(title, url, abstract_text)| {
                    // Skip disambiguation pages
                    if !include_disam
                        && (title.contains("曖昧さ回避")
                            || abstract_text.contains("曖昧さ回避")
                            || abstract_text.contains("disambiguation"))
                    {
                        return (vec![], true, false);
                    }

                    // Skip stubs with very short abstracts
                    if abstract_text.chars().count() < min_chars {
                        return (vec![], false, true);
                    }

                    // Confidence based on abstract richness
                    let confidence = compute_confidence(abstract_text, boost);

                    // Surface form: the article title (strip "Wikipedia: " prefix)
                    let surface = strip_wikipedia_prefix(title);
                    if surface.is_empty() {
                        return (vec![], false, true);
                    }

                    // Also add reading variants (hiragana/katakana in parentheses)
                    // e.g. "東京都（とうきょうと）" → also add "とうきょうと"
                    let mut results = vec![(surface.to_string(), url.clone(), confidence)];
                    if let Some(reading) = extract_parenthetical_reading(abstract_text) {
                        if is_kana(&reading) {
                            results.push((reading, url.clone(), confidence * 0.9));
                        }
                    }

                    (results, false, false)
                })
                .collect();

            let mut all_entries = Vec::new();
            let mut skipped_dis = 0u64;
            let mut skipped_no_abs = 0u64;

            for (batch, is_dis, is_no_abs) in results {
                if is_dis {
                    skipped_dis += 1;
                } else if is_no_abs {
                    skipped_no_abs += 1;
                }
                all_entries.extend(batch);
            }

            (all_entries, skipped_dis, skipped_no_abs)
        };

        stats.surfaces_added = entries.len() as u64;
        stats.skipped_disambiguation = skipped_disambiguation;
        stats.skipped_no_abstract = skipped_no_abstract;

        // Sequential merge into index
        for (surface, uri, confidence) in entries {
            index.add(&surface, &uri, confidence);
        }

        Ok(stats)
    }

    /// Parse articles from a Wikipedia abstract dump reader.
    ///
    /// Returns a Vec of (title, url, abstract) tuples.
    fn parse_articles(reader: impl BufRead) -> Result<Vec<(String, String, String)>> {
        let mut articles = Vec::new();
        let mut current_title = String::new();
        let mut current_url = String::new();
        let mut current_abstract = String::new();
        let mut in_doc = false;

        for line_result in reader.lines() {
            let line = line_result.map_err(BuildError::Io)?;
            let trimmed = line.trim();

            if trimmed == "<doc>" {
                in_doc = true;
                current_title.clear();
                current_url.clear();
                current_abstract.clear();
            } else if trimmed == "</doc>" {
                if in_doc && !current_title.is_empty() && !current_url.is_empty() {
                    articles.push((
                        current_title.clone(),
                        current_url.clone(),
                        current_abstract.clone(),
                    ));
                }
                in_doc = false;
            } else if in_doc {
                if let Some(content) = extract_xml_text(trimmed, "title") {
                    current_title = content.to_string();
                } else if let Some(content) = extract_xml_text(trimmed, "url") {
                    current_url = content.to_string();
                } else if let Some(content) = extract_xml_text(trimmed, "abstract") {
                    current_abstract = content.to_string();
                }
            }
        }

        Ok(articles)
    }
}

/// Compute confidence for a Wikipedia article based on its abstract.
///
/// Longer, more substantive abstracts indicate more notable entities.
/// For Japanese text (which lacks word boundaries), character count is used.
///
/// Score: log10(char_count + 1) / 3, clamped to [0.1, 1.0]
///
/// Examples:
/// - 1 char  → log10(2)/3  ≈ 0.10 → 0.10
/// - 20 chars → log10(21)/3 ≈ 0.43
/// - 200 chars → log10(201)/3 ≈ 0.77
/// - 1000 chars → log10(1001)/3 ≈ 1.00
fn compute_confidence(abstract_text: &str, boost: f32) -> f32 {
    let char_count = abstract_text.chars().count() as f32;
    let base = (char_count + 1.0).log10() / 3.0;
    (base * boost).clamp(0.1, 1.0)
}

/// Strip the "Wikipedia: " prefix from a title.
fn strip_wikipedia_prefix(title: &str) -> &str {
    title
        .strip_prefix("Wikipedia: ")
        .or_else(|| title.strip_prefix("Wikipedia:"))
        .unwrap_or(title)
}

/// Extract text content from a simple XML element on a single line.
/// e.g. `<title>東京都</title>` → `Some("東京都")`
fn extract_xml_text<'a>(line: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = line.find(&open)? + open.len();
    let end = line[start..].find(&close)? + start;
    Some(&line[start..end])
}

/// Extract the first parenthetical reading from a Japanese text.
/// e.g. "東京都（とうきょうと）は..." → Some("とうきょうと")
fn extract_parenthetical_reading(text: &str) -> Option<String> {
    // Look for （...） containing only kana
    let open = text.find('（')?;
    let rest = &text[open + '（'.len_utf8()..];
    let close = rest.find('）')?;
    Some(rest[..close].to_string())
}

/// Check if a string consists entirely of kana (hiragana/katakana)
fn is_kana(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            matches!(c,
                '\u{3041}'..='\u{3096}' |  // hiragana
                '\u{30A0}'..='\u{30FF}'    // katakana (includes ー U+30FC and ・ U+30FB)
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wikidata::WikidataIndex;

    #[test]
    fn test_strip_wikipedia_prefix() {
        assert_eq!(strip_wikipedia_prefix("Wikipedia: 東京都"), "東京都");
        assert_eq!(strip_wikipedia_prefix("東京都"), "東京都");
    }

    #[test]
    fn test_extract_xml_text() {
        let line = "<title>Wikipedia: 東京都</title>";
        assert_eq!(extract_xml_text(line, "title"), Some("Wikipedia: 東京都"));
        assert_eq!(extract_xml_text(line, "url"), None);
    }

    #[test]
    fn test_extract_parenthetical_reading() {
        assert_eq!(
            extract_parenthetical_reading("東京都（とうきょうと）は"),
            Some("とうきょうと".to_string())
        );
        assert_eq!(
            extract_parenthetical_reading("test (nothing)"),
            None // uses ASCII parens
        );
    }

    #[test]
    fn test_is_kana() {
        assert!(is_kana("とうきょうと"));
        assert!(is_kana("トウキョウト"));
        assert!(!is_kana("東京都"));
        assert!(!is_kana(""));
    }

    #[test]
    fn test_compute_confidence() {
        let short = compute_confidence("短い", 1.0);
        let long = compute_confidence(
            "東京都は日本の首都であり、政治・経済・文化の中心地です。人口は約1400万人。",
            1.0,
        );
        assert!(long > short);
        assert!(short >= 0.1);
        assert!(long <= 1.0);
    }

    #[test]
    fn test_parse_articles() {
        let xml = r#"<feed>
<doc>
<title>Wikipedia: 東京都</title>
<url>http://ja.wikipedia.org/wiki/東京都</url>
<abstract>東京都（とうきょうと）は、日本の都道府県のひとつ。</abstract>
</doc>
<doc>
<title>Wikipedia: 大阪府</title>
<url>http://ja.wikipedia.org/wiki/大阪府</url>
<abstract>大阪府（おおさかふ）は、近畿地方の府県のひとつ。</abstract>
</doc>
</feed>"#;
        let reader = std::io::BufReader::new(xml.as_bytes());
        let articles = WikipediaProcessor::parse_articles(reader).unwrap();
        assert_eq!(articles.len(), 2);
        assert_eq!(articles[0].0, "Wikipedia: 東京都");
        assert_eq!(articles[0].1, "http://ja.wikipedia.org/wiki/東京都");
    }

    #[test]
    fn test_process_dump_end_to_end() {
        let xml = r#"<feed>
<doc>
<title>Wikipedia: テスト記事</title>
<url>http://ja.wikipedia.org/wiki/テスト記事</url>
<abstract>テスト記事（てすときじ）は、テストのための記事です。テストテストテストテストテスト。</abstract>
</doc>
</feed>"#;

        let dir = std::env::temp_dir();
        let path = dir.join("test_wiki_abstract.xml");
        std::fs::write(&path, xml).expect("failed to write temp file");

        let processor = WikipediaProcessor::new();
        let mut index = WikidataIndex::default();
        let stats = processor
            .process_dump(&path, &mut index, false)
            .expect("process_dump failed");

        assert_eq!(stats.articles_processed, 1);
        assert!(stats.surfaces_added > 0);

        // "テスト記事" should be in index
        let results = index.lookup("テスト記事");
        assert!(results.is_some());

        std::fs::remove_file(&path).ok();
    }

    /// Exercises the gzip (`.gz`) decompression branch end-to-end through the
    /// Pure-Rust `oxiarc_deflate::GzipStreamDecoder` (the flate2 replacement).
    ///
    /// The dump is compressed with `oxiarc_deflate::gzip_compress`, written to
    /// a `.gz` file, then decoded by `process_dump`, proving the streaming
    /// decoder round-trips real gzip data.
    #[test]
    fn test_process_dump_gzip_roundtrip() {
        let xml = r#"<feed>
<doc>
<title>Wikipedia: テスト記事</title>
<url>http://ja.wikipedia.org/wiki/テスト記事</url>
<abstract>テスト記事（てすときじ）は、テストのための記事です。テストテストテストテストテスト。</abstract>
</doc>
</feed>"#;

        // Compress with the same Pure-Rust crate used for decoding.
        let compressed =
            oxiarc_deflate::gzip_compress(xml.as_bytes(), 6).expect("gzip_compress failed");

        let dir = std::env::temp_dir();
        let path = dir.join("test_mecrab_wiki_abstract_roundtrip.xml.gz");
        std::fs::write(&path, &compressed).expect("failed to write temp .gz file");

        let processor = WikipediaProcessor::new();
        let mut index = WikidataIndex::default();
        let stats = processor
            .process_dump(&path, &mut index, false)
            .expect("process_dump on .gz failed");

        assert_eq!(stats.articles_processed, 1);
        assert!(stats.surfaces_added > 0);

        // The decompressed content must have been parsed correctly.
        let results = index.lookup("テスト記事");
        assert!(results.is_some(), "surface from .gz dump should be indexed");

        std::fs::remove_file(&path).ok();
    }
}
