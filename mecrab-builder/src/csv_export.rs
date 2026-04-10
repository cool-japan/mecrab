//! CSV exporter for semantic-enriched IPADIC dictionaries.
//!
//! Produces an IPADIC-compatible CSV with additional columns for
//! semantic entity URIs sourced from the Wikidata index.
//!
//! ## Enhanced column layout
//!
//! Standard IPADIC (13 columns):
//! ```text
//! surface,left_id,right_id,wcost,pos1,pos2,pos3,pos4,conj_type,conj_form,base_form,reading,pronunciation
//! ```
//!
//! Enhanced output appends 2 columns:
//! ```text
//! …,semantic_uri,semantic_confidence
//! ```
//!
//! Where `semantic_uri` is the top Wikidata URI for this surface (or `""`)
//! and `semantic_confidence` is the associated score (or `0`).

use crate::Result;
use crate::wikidata::WikidataIndex;
use csv::WriterBuilder;
use std::io::Write;
use std::path::Path;

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

/// Configuration for CSV export
#[derive(Debug, Clone)]
pub struct CsvExportConfig {
    /// Whether to include entries with no semantic URI
    pub include_plain_entries: bool,
    /// Minimum confidence threshold; entries below this get empty URI columns
    pub min_confidence: f32,
    /// Maximum number of URI candidates per surface (used in fan-out mode)
    pub max_candidates: usize,
    /// When `true`, emit one row per URI candidate (fan-out).
    /// When `false` (default), emit one row with the top-confidence URI only.
    pub fan_out: bool,
}

impl Default for CsvExportConfig {
    fn default() -> Self {
        Self {
            include_plain_entries: true,
            min_confidence: 0.1,
            max_candidates: 3,
            fan_out: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────

/// Statistics returned after a CSV export operation
#[derive(Debug, Default)]
pub struct CsvExportStats {
    /// Total rows read from source CSV
    pub total_rows: u64,
    /// Rows for which at least one URI was written
    pub rows_with_uri: u64,
    /// Rows written with empty URI columns
    pub rows_plain: u64,
    /// Distinct surface forms that had at least one URI match
    pub unique_surfaces_with_uri: u64,
}

// ─────────────────────────────────────────────────────────────
// Main export function
// ─────────────────────────────────────────────────────────────

/// Export a source IPADIC CSV merged with Wikidata URIs.
///
/// Reads the source CSV line by line (streaming), looks up each surface
/// in the Wikidata index, and writes an enhanced CSV to `output`.
///
/// # Arguments
/// * `source_csv` – Path to the source IPADIC CSV (plain only; `.gz` rejected cleanly)
/// * `index`      – Wikidata surface-to-URI index
/// * `output`     – Destination writer (file, buffer, …)
/// * `config`     – Export configuration
pub fn export_csv_with_uris<W: Write>(
    source_csv: &Path,
    index: &WikidataIndex,
    output: W,
    config: &CsvExportConfig,
) -> Result<CsvExportStats> {
    use crate::BuildError;

    // Reject .gz to avoid silent truncation
    if source_csv
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gz"))
    {
        return Err(BuildError::InvalidInput(
            "Gzipped CSV is not supported by export_csv_with_uris; \
             decompress first or use build_wikidata_index_streaming"
                .to_string(),
        ));
    }

    let mut stats = CsvExportStats::default();

    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(source_csv)?;

    let mut csv_writer = WriterBuilder::new().has_headers(false).from_writer(output);

    // Track which surfaces we have already counted toward unique_surfaces_with_uri
    let mut seen_with_uri: std::collections::HashSet<String> = std::collections::HashSet::new();

    for record_result in csv_reader.records() {
        let record = record_result?;
        stats.total_rows += 1;

        // Surface form is column 0
        let surface = record.get(0).unwrap_or("").to_string();

        // Look up URIs for this surface
        let uris_opt = index.lookup(&surface);

        match uris_opt {
            Some(candidates) if !candidates.is_empty() => {
                if config.fan_out {
                    // One row per URI candidate (fan-out mode)
                    let take = candidates.len().min(config.max_candidates);
                    let mut wrote_any = false;
                    for (uri, confidence) in candidates.iter().take(take) {
                        if *confidence >= config.min_confidence {
                            let mut out: Vec<String> =
                                record.iter().map(|s| s.to_string()).collect();
                            out.push(uri.clone());
                            out.push(confidence.to_string());
                            csv_writer.write_record(&out)?;
                            wrote_any = true;
                        }
                    }
                    if wrote_any {
                        stats.rows_with_uri += 1;
                        if seen_with_uri.insert(surface.clone()) {
                            stats.unique_surfaces_with_uri += 1;
                        }
                    } else if config.include_plain_entries {
                        write_plain_row(&mut csv_writer, &record)?;
                        stats.rows_plain += 1;
                    }
                } else {
                    // Top-URI mode
                    match candidates.first() {
                        Some((top_uri, top_conf)) if *top_conf >= config.min_confidence => {
                            let mut out: Vec<String> =
                                record.iter().map(|s| s.to_string()).collect();
                            out.push(top_uri.clone());
                            out.push(top_conf.to_string());
                            csv_writer.write_record(&out)?;
                            stats.rows_with_uri += 1;
                            if seen_with_uri.insert(surface.clone()) {
                                stats.unique_surfaces_with_uri += 1;
                            }
                        }
                        _ => {
                            // Confidence below threshold
                            if config.include_plain_entries {
                                write_plain_row(&mut csv_writer, &record)?;
                                stats.rows_plain += 1;
                            }
                        }
                    }
                }
            }
            _ => {
                // No URI found
                if config.include_plain_entries {
                    write_plain_row(&mut csv_writer, &record)?;
                    stats.rows_plain += 1;
                }
            }
        }
    }

    csv_writer.flush()?;
    Ok(stats)
}

/// Write a CSV record with empty URI / zero confidence appended.
fn write_plain_row<W: Write>(
    writer: &mut csv::Writer<W>,
    record: &csv::StringRecord,
) -> Result<()> {
    let mut out: Vec<String> = record.iter().map(|s| s.to_string()).collect();
    out.push(String::new());
    out.push("0".to_string());
    writer.write_record(&out)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────
// Standalone index → CSV
// ─────────────────────────────────────────────────────────────

/// Export the Wikidata index as a standalone CSV mapping file.
///
/// Output columns: `surface,uri,confidence`
///
/// # Arguments
/// * `index`          – the index to export
/// * `output`         – destination writer
/// * `min_confidence` – entries below this threshold are omitted
///
/// Returns the number of rows written.
pub fn export_index_as_csv<W: Write>(
    index: &WikidataIndex,
    output: W,
    min_confidence: f32,
) -> Result<u64> {
    let mut writer = WriterBuilder::new().has_headers(true).from_writer(output);

    writer.write_record(["surface", "uri", "confidence"])?;

    let mut count = 0u64;
    for (surface, candidates) in index.iter_entries() {
        for (uri, confidence) in &candidates {
            if *confidence >= min_confidence {
                writer.write_record([surface, uri.as_str(), &confidence.to_string()])?;
                count += 1;
            }
        }
    }

    writer.flush()?;
    Ok(count)
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wikidata::WikidataIndex;
    use std::io::Cursor;

    fn make_index() -> WikidataIndex {
        let mut idx = WikidataIndex::new();
        idx.add("東京", "http://www.wikidata.org/entity/Q1490", 0.9);
        idx.add("東京", "http://dbpedia.org/resource/Tokyo", 0.5);
        idx.add("京都", "http://www.wikidata.org/entity/Q34600", 0.8);
        idx
    }

    #[test]
    fn test_export_index_as_csv_basic() {
        let idx = make_index();
        let mut buf = Vec::<u8>::new();
        let count = export_index_as_csv(&idx, &mut buf, 0.0).expect("export ok");
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("surface,uri,confidence"));
        assert!(text.contains("東京"));
        assert!(text.contains("Q1490"));
        assert_eq!(count, 3); // 2 for 東京, 1 for 京都
    }

    #[test]
    fn test_export_index_as_csv_min_confidence_filter() {
        let idx = make_index();
        let mut buf = Vec::<u8>::new();
        let count = export_index_as_csv(&idx, &mut buf, 0.6).expect("export ok");
        // Only 東京→Q1490 (0.9) and 京都→Q34600 (0.8) pass the threshold
        assert_eq!(count, 2);
    }

    #[test]
    fn test_export_csv_with_uris_top_mode() {
        let idx = make_index();
        let config = CsvExportConfig::default();

        // Write a minimal IPADIC-like CSV to a temp file
        let dir = std::env::temp_dir();
        let src = dir.join("mecrab_builder_test_export_top.csv");
        // Two rows: one for 東京 (has URI), one for 大阪 (no URI)
        std::fs::write(
            &src,
            "東京,1285,1285,3993,名詞,固有名詞,地名,一般,*,*,東京,トウキョウ,トウキョウ\n\
             大阪,1285,1285,4000,名詞,固有名詞,地名,一般,*,*,大阪,オオサカ,オオサカ\n",
        )
        .unwrap();

        let mut out = Vec::<u8>::new();
        let stats = export_csv_with_uris(&src, &idx, &mut out, &config).expect("export ok");
        let text = String::from_utf8(out).unwrap();

        assert_eq!(stats.total_rows, 2);
        assert_eq!(stats.rows_with_uri, 1);
        assert_eq!(stats.rows_plain, 1);
        assert_eq!(stats.unique_surfaces_with_uri, 1);
        assert!(text.contains("Q1490"));
        assert!(text.contains("大阪"));

        let _ = std::fs::remove_file(src);
    }

    #[test]
    fn test_export_csv_with_uris_fan_out_mode() {
        let idx = make_index();
        let config = CsvExportConfig {
            fan_out: true,
            min_confidence: 0.0,
            max_candidates: 5,
            include_plain_entries: false,
        };

        let dir = std::env::temp_dir();
        let src = dir.join("mecrab_builder_test_export_fanout.csv");
        std::fs::write(
            &src,
            "東京,1285,1285,3993,名詞,固有名詞,地名,一般,*,*,東京,トウキョウ,トウキョウ\n",
        )
        .unwrap();

        let mut out = Vec::<u8>::new();
        let stats = export_csv_with_uris(&src, &idx, &mut out, &config).expect("export ok");
        let text = String::from_utf8(out).unwrap();

        // 東京 has 2 URIs → 2 rows emitted
        assert_eq!(stats.total_rows, 1);
        assert_eq!(stats.rows_with_uri, 1);
        // Both URIs should appear
        assert!(text.contains("Q1490"));
        assert!(text.contains("dbpedia.org"));

        let _ = std::fs::remove_file(src);
    }

    #[test]
    fn test_export_csv_rejects_gz() {
        let idx = make_index();
        let config = CsvExportConfig::default();
        let path = std::path::Path::new("/tmp/fake_dump.json.gz");
        let mut out = Cursor::new(Vec::<u8>::new());
        let result = export_csv_with_uris(path, &idx, &mut out, &config);
        assert!(result.is_err());
    }
}
