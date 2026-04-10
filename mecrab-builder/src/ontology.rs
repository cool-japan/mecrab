//! Custom ontology import for semantic dictionary enrichment.
//!
//! Supports importing user-defined entity→URI mappings in three formats:
//! - Simple CSV (`surface,uri,confidence[,type]`)
//! - JSON array (structured, supports aliases)
//! - Lightweight RDF/OWL XML (rdfs:label extraction)
//!
//! All formats merge into the `WikidataIndex` using max-confidence semantics.
//!
//! # Example
//!
//! ```no_run
//! use mecrab_builder::{WikidataIndex, import_ontology};
//! use std::path::Path;
//!
//! let mut index = WikidataIndex::new();
//! let stats = import_ontology(
//!     Path::new("entities.csv"),
//!     &mut index,
//!     0.8,
//! ).unwrap();
//! println!("Added {} surfaces", stats.surfaces_added);
//! ```

use crate::wikidata::WikidataIndex;
use crate::{BuildError, Result};
use serde::{Deserialize, Serialize};
use std::io::BufRead;
use std::path::Path;

// ── Types ──────────────────────────────────────────────────────

/// An ontology entry in memory.
///
/// Represents a single surface→URI mapping with optional confidence,
/// entity type, and aliases for richer coverage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyEntry {
    /// Primary surface form (the text as it appears in running text)
    pub surface: String,
    /// Entity URI (Wikidata, internal, or any IRI)
    pub uri: String,
    /// Confidence score in [0.0, 1.0]; defaults to 0.7 if omitted
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// Entity type label (e.g., "person", "city", "product")
    #[serde(default)]
    pub entity_type: String,
    /// Alternative surface forms (aliases); each gets confidence * 0.9
    #[serde(default)]
    pub aliases: Vec<String>,
}

fn default_confidence() -> f32 {
    0.7
}

/// Statistics from an ontology import operation.
#[derive(Debug, Default)]
pub struct OntologyStats {
    /// Number of logical entries (primary surfaces) successfully read
    pub entries_read: u64,
    /// Number of primary surface→URI pairs inserted or updated
    pub surfaces_added: u64,
    /// Number of alias surface→URI pairs inserted or updated
    pub aliases_added: u64,
    /// Number of records skipped due to missing/invalid fields
    pub skipped_invalid: u64,
}

/// Supported ontology file formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OntologyFormat {
    /// Simple CSV with header `surface,uri[,confidence[,type]]`
    Csv,
    /// JSON array of [`OntologyEntry`] objects
    Json,
    /// Lightweight RDF/XML: extracts `rdfs:label xml:lang="ja"` values
    RdfXml,
}

impl OntologyFormat {
    /// Detect format from the file's extension.
    ///
    /// Returns `None` for unrecognised extensions.
    pub fn detect(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "csv" => Some(Self::Csv),
            "json" => Some(Self::Json),
            "owl" | "rdf" | "xml" => Some(Self::RdfXml),
            _ => None,
        }
    }
}

// ── Main API ────────────────────────────────────────────────────

/// Import a custom ontology file and merge entries into `index`.
///
/// The format is detected automatically from the file extension:
/// `.csv` → CSV, `.json` → JSON, `.owl`/`.rdf`/`.xml` → RDF/XML.
///
/// On success, returns statistics describing how many surfaces and aliases
/// were inserted.  Unknown extensions return
/// [`BuildError::InvalidInput`].
///
/// # Arguments
///
/// * `path` – Path to the ontology file.
/// * `index` – Mutable target index; entries are *merged*
///   (max-confidence semantics on conflicts).
/// * `default_confidence` – Fallback confidence when none is supplied by
///   the record (used for CSV / RDF/XML formats).
pub fn import_ontology(
    path: &Path,
    index: &mut WikidataIndex,
    default_confidence: f32,
) -> Result<OntologyStats> {
    let format = OntologyFormat::detect(path).ok_or_else(|| {
        BuildError::InvalidInput(format!(
            "Cannot detect ontology format from extension: {}",
            path.display()
        ))
    })?;

    match format {
        OntologyFormat::Csv => import_csv(path, index, default_confidence),
        OntologyFormat::Json => import_json(path, index),
        OntologyFormat::RdfXml => import_rdf_xml(path, index, default_confidence),
    }
}

// ── CSV import ──────────────────────────────────────────────────

/// Import from a simple CSV file.
///
/// Expected header (first row): `surface,uri[,confidence[,type]]`.
/// The `type` column is currently parsed but not stored; only `surface`,
/// `uri`, and `confidence` are used.
///
/// Rows where `surface` or `uri` are empty are counted as
/// `skipped_invalid`.
pub fn import_csv(
    path: &Path,
    index: &mut WikidataIndex,
    default_conf: f32,
) -> Result<OntologyStats> {
    let mut stats = OntologyStats::default();

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(reader);

    for record_result in csv_reader.records() {
        let record = record_result?;

        let surface = match record.get(0) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => {
                stats.skipped_invalid += 1;
                continue;
            }
        };

        let uri = match record.get(1) {
            Some(u) if !u.is_empty() => u.to_string(),
            _ => {
                stats.skipped_invalid += 1;
                continue;
            }
        };

        let confidence = record
            .get(2)
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(default_conf)
            .clamp(0.0, 1.0);

        index.add(&surface, &uri, confidence);
        stats.entries_read += 1;
        stats.surfaces_added += 1;
    }

    Ok(stats)
}

// ── JSON import ─────────────────────────────────────────────────

/// Import from a JSON array of [`OntologyEntry`] objects.
///
/// Each entry must have non-empty `surface` and `uri` fields.  Optional
/// fields `confidence` (default 0.7), `entity_type` (ignored), and
/// `aliases` are supported.
///
/// Aliases inherit the same URI at `confidence * 0.9` (clamped to
/// `[0.05, 1.0]`).
pub fn import_json(path: &Path, index: &mut WikidataIndex) -> Result<OntologyStats> {
    let mut stats = OntologyStats::default();

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let entries: Vec<OntologyEntry> = serde_json::from_reader(reader)?;

    for entry in entries {
        if entry.surface.is_empty() || entry.uri.is_empty() {
            stats.skipped_invalid += 1;
            continue;
        }

        let confidence = entry.confidence.clamp(0.0, 1.0);

        index.add(&entry.surface, &entry.uri, confidence);
        stats.entries_read += 1;
        stats.surfaces_added += 1;

        // Aliases get a 10% confidence reduction to signal they are secondary forms
        for alias in &entry.aliases {
            if !alias.is_empty() {
                let alias_conf = (confidence * 0.9_f32).clamp(0.05, 1.0);
                index.add(alias, &entry.uri, alias_conf);
                stats.aliases_added += 1;
            }
        }
    }

    Ok(stats)
}

// ── RDF/XML import ──────────────────────────────────────────────

/// Import from a lightweight RDF/XML file.
///
/// Extracts all `rdfs:label` values whose `xml:lang` attribute equals
/// `"ja"` and maps them to the enclosing `rdf:Description`'s
/// `rdf:about` URI.
///
/// The parser is a line-oriented state machine rather than a full DOM
/// parser; it requires each tag to occupy a single line (standard output
/// from Protégé, RDFLib, or the Wikidata JSON→RDF exporter).  Complex
/// multiline tags are silently skipped without error.
///
/// End-of-block detection handles both `</rdf:Description>` and
/// `</owl:Class>`.
pub fn import_rdf_xml(
    path: &Path,
    index: &mut WikidataIndex,
    default_conf: f32,
) -> Result<OntologyStats> {
    let mut stats = OntologyStats::default();

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut current_uri: Option<String> = None;
    let mut current_labels: Vec<String> = Vec::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();

        // Detect resource URI: <rdf:Description rdf:about="...">
        if let Some(uri) = extract_rdf_about(trimmed) {
            // Flush previous resource if any
            flush_labels(
                &mut current_uri,
                &mut current_labels,
                index,
                default_conf,
                &mut stats,
            );
            current_uri = Some(uri);
        }
        // Detect Japanese label: <rdfs:label xml:lang="ja">...</rdfs:label>
        else if let Some(label) = extract_rdfs_label_ja(trimmed) {
            current_labels.push(label);
        }
        // End of description block
        else if trimmed == "</rdf:Description>" || trimmed == "</owl:Class>" {
            flush_labels(
                &mut current_uri,
                &mut current_labels,
                index,
                default_conf,
                &mut stats,
            );
        }
    }

    // Flush last resource (file may not end with </rdf:Description>)
    flush_labels(
        &mut current_uri,
        &mut current_labels,
        index,
        default_conf,
        &mut stats,
    );

    Ok(stats)
}

/// Commit all pending labels for the current resource URI into the index,
/// then reset state.
fn flush_labels(
    current_uri: &mut Option<String>,
    current_labels: &mut Vec<String>,
    index: &mut WikidataIndex,
    confidence: f32,
    stats: &mut OntologyStats,
) {
    if let Some(ref uri) = *current_uri {
        if !current_labels.is_empty() {
            for label in current_labels.iter() {
                index.add(label, uri, confidence);
                stats.surfaces_added += 1;
            }
            stats.entries_read += 1;
        }
    }
    current_uri.take();
    current_labels.clear();
}

/// Extract the `rdf:about` URI from a description element line.
///
/// Matches: `<rdf:Description rdf:about="http://example.org/foo">`
/// Returns: `"http://example.org/foo"`
fn extract_rdf_about(line: &str) -> Option<String> {
    let marker = "rdf:about=\"";
    let start = line.find(marker)? + marker.len();
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

/// Extract the text content of a Japanese-language `rdfs:label` element.
///
/// Matches lines containing `xml:lang="ja"` (or `lang="ja"`) and a label
/// element.  Returns `None` when content is absent or the language is not
/// Japanese.
fn extract_rdfs_label_ja(line: &str) -> Option<String> {
    // Require Japanese language annotation
    if !line.contains("lang=\"ja\"") {
        return None;
    }
    // Require a label element (rdfs:label or any NS :label)
    if !line.contains("rdfs:label") && !line.contains(":label") {
        return None;
    }
    // Extract content between > and </
    let open = line.find('>')? + 1;
    let close = line[open..].find('<')? + open;
    let content = line[open..close].trim();
    if content.is_empty() {
        None
    } else {
        Some(content.to_string())
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_temp_file(name: &str, content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, content).expect("write temp file");
        path
    }

    #[test]
    fn test_import_csv_basic() {
        let csv = "surface,uri,confidence,type\n\
            東京,http://example.com/Q1,0.9,city\n\
            大阪,http://example.com/Q2,0.85,city\n";
        let path = make_temp_file("test_ontology_basic.csv", csv);

        let mut index = WikidataIndex::default();
        let stats = import_csv(&path, &mut index, 0.7).expect("import_csv");

        assert_eq!(stats.entries_read, 2, "entries_read");
        assert_eq!(stats.surfaces_added, 2, "surfaces_added");
        assert_eq!(stats.skipped_invalid, 0, "skipped_invalid");
        assert!(index.lookup("東京").is_some(), "東京 in index");
        assert!(index.lookup("大阪").is_some(), "大阪 in index");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_import_csv_missing_fields() {
        // rows with empty surface or uri should be skipped
        let csv = "surface,uri\n\
            ,http://example.com/Q1\n\
            東京,\n\
            valid,http://example.com/Q3\n";
        let path = make_temp_file("test_ontology_missing.csv", csv);

        let mut index = WikidataIndex::default();
        let stats = import_csv(&path, &mut index, 0.7).expect("import_csv");

        assert_eq!(stats.entries_read, 1);
        assert_eq!(stats.skipped_invalid, 2);
        assert!(index.lookup("valid").is_some());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_import_csv_default_confidence() {
        let csv = "surface,uri\ntest_surface,http://example.com/X\n";
        let path = make_temp_file("test_ontology_defconf.csv", csv);

        let mut index = WikidataIndex::default();
        import_csv(&path, &mut index, 0.42).expect("import_csv");

        let results = index.lookup("test_surface").expect("lookup");
        let (_, conf) = results.first().expect("first result");
        assert!(
            (conf - 0.42).abs() < 1e-4,
            "confidence should match default"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_import_json_with_aliases() {
        let json = r#"[
            {
                "surface": "東京",
                "uri": "http://example.com/Q1",
                "confidence": 0.9,
                "aliases": ["トウキョウ", "tokyo"]
            },
            {
                "surface": "大阪",
                "uri": "http://example.com/Q2"
            }
        ]"#;
        let path = make_temp_file("test_ontology_aliases.json", json);

        let mut index = WikidataIndex::default();
        let stats = import_json(&path, &mut index).expect("import_json");

        assert_eq!(stats.entries_read, 2, "entries_read");
        assert_eq!(stats.aliases_added, 2, "aliases_added");
        assert!(index.lookup("東京").is_some(), "東京 in index");
        assert!(index.lookup("トウキョウ").is_some(), "alias in index");
        assert!(index.lookup("tokyo").is_some(), "ascii alias in index");
        assert!(index.lookup("大阪").is_some(), "大阪 in index");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_import_json_alias_confidence_reduction() {
        let json =
            r#"[{"surface":"foo","uri":"http://x.com/1","confidence":1.0,"aliases":["bar"]}]"#;
        let path = make_temp_file("test_ontology_aliasconf.json", json);

        let mut index = WikidataIndex::default();
        import_json(&path, &mut index).expect("import_json");

        let foo_results = index.lookup("foo").expect("foo lookup");
        let bar_results = index.lookup("bar").expect("bar lookup");

        let foo_conf = foo_results.first().expect("foo first").1;
        let bar_conf = bar_results.first().expect("bar first").1;
        assert!(
            bar_conf < foo_conf,
            "alias confidence must be lower than primary"
        );
        assert!(
            (bar_conf - 0.9_f32).abs() < 1e-4,
            "alias confidence = primary * 0.9"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_import_json_skips_invalid() {
        let json = r#"[
            {"surface": "", "uri": "http://x.com/1"},
            {"surface": "ok", "uri": ""},
            {"surface": "valid", "uri": "http://x.com/3", "confidence": 0.5}
        ]"#;
        let path = make_temp_file("test_ontology_invalid.json", json);

        let mut index = WikidataIndex::default();
        let stats = import_json(&path, &mut index).expect("import_json");

        assert_eq!(stats.skipped_invalid, 2);
        assert_eq!(stats.entries_read, 1);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_import_rdf_xml() {
        let xml = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <rdf:Description rdf:about="http://example.com/Tokyo">
    <rdfs:label xml:lang="ja">東京</rdfs:label>
    <rdfs:label xml:lang="ja">とうきょう</rdfs:label>
    <rdfs:label xml:lang="en">Tokyo</rdfs:label>
  </rdf:Description>
</rdf:RDF>"#;
        let path = make_temp_file("test_ontology.rdf", xml);

        let mut index = WikidataIndex::default();
        let stats = import_rdf_xml(&path, &mut index, 0.8).expect("import_rdf_xml");

        assert!(stats.surfaces_added >= 2, "at least 2 Japanese labels");
        assert!(index.lookup("東京").is_some(), "東京 in index");
        assert!(index.lookup("とうきょう").is_some(), "とうきょう in index");
        // English label should NOT be indexed
        assert!(index.lookup("Tokyo").is_none(), "English label not indexed");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_import_rdf_xml_owl_class() {
        let xml = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <rdf:Description rdf:about="http://example.com/Osaka">
    <rdfs:label xml:lang="ja">大阪</rdfs:label>
  </rdf:Description>
</rdf:RDF>"#;
        let path = make_temp_file("test_ontology_owl.rdf", xml);

        let mut index = WikidataIndex::default();
        let stats = import_rdf_xml(&path, &mut index, 0.75).expect("import_rdf_xml");

        assert_eq!(stats.surfaces_added, 1);
        let results = index.lookup("大阪").expect("大阪 in index");
        let (uri, conf) = results.first().expect("first result");
        assert_eq!(uri, "http://example.com/Osaka");
        assert!((conf - 0.75_f32).abs() < 1e-4, "confidence matches default");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            OntologyFormat::detect(Path::new("test.csv")),
            Some(OntologyFormat::Csv)
        );
        assert_eq!(
            OntologyFormat::detect(Path::new("test.json")),
            Some(OntologyFormat::Json)
        );
        assert_eq!(
            OntologyFormat::detect(Path::new("test.owl")),
            Some(OntologyFormat::RdfXml)
        );
        assert_eq!(
            OntologyFormat::detect(Path::new("test.rdf")),
            Some(OntologyFormat::RdfXml)
        );
        assert_eq!(
            OntologyFormat::detect(Path::new("test.xml")),
            Some(OntologyFormat::RdfXml)
        );
        assert_eq!(OntologyFormat::detect(Path::new("test.txt")), None);
        assert_eq!(OntologyFormat::detect(Path::new("noext")), None);
    }

    #[test]
    fn test_import_ontology_auto_detect_csv() {
        let csv = "surface,uri\nAuto,http://example.com/auto\n";
        let path = make_temp_file("test_auto_detect.csv", csv);

        let mut index = WikidataIndex::default();
        let stats = import_ontology(&path, &mut index, 0.7).expect("import_ontology");
        assert_eq!(stats.entries_read, 1);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_import_ontology_unknown_extension_error() {
        let path = Path::new("/tmp/unknown.xyz");
        let mut index = WikidataIndex::default();
        let result = import_ontology(path, &mut index, 0.7);
        assert!(result.is_err(), "should fail on unknown extension");
    }

    #[test]
    fn test_max_confidence_semantics_on_merge() {
        // Adding the same surface+uri twice should keep the higher confidence
        let csv1 = "surface,uri,confidence\nfoo,http://x.com/1,0.4\n";
        let csv2 = "surface,uri,confidence\nfoo,http://x.com/1,0.9\n";
        let path1 = make_temp_file("test_merge1.csv", csv1);
        let path2 = make_temp_file("test_merge2.csv", csv2);

        let mut index = WikidataIndex::default();
        import_csv(&path1, &mut index, 0.7).expect("first import");
        import_csv(&path2, &mut index, 0.7).expect("second import");

        let results = index.lookup("foo").expect("foo");
        let conf = results.first().expect("first").1;
        assert!(
            (conf - 0.9_f32).abs() < 1e-4,
            "max-confidence semantics: expected 0.9, got {}",
            conf
        );

        std::fs::remove_file(&path1).ok();
        std::fs::remove_file(&path2).ok();
    }
}
