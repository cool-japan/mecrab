//! Semantic enrichment for morphological analysis
//!
//! This module provides:
//! - `SemanticPool`: Compact URI storage (5-byte entries)
//! - `TokenExtension`: Multi-candidate URI linking
//! - JSON-LD / RDF export formats
//! - Disambiguation strategies

pub mod disambiguation;
pub mod extension;
pub mod jsonld;
pub mod pool;
pub mod rdf;

pub use disambiguation::DisambiguationStrategy;
pub use extension::{EntityReference, ExtendedMorpheme, ExtendedResultBuilder, TokenExtension};
pub use pool::{OntologySource, SemanticPool, SemanticPoolBuilder, SemanticPoolStats};

use std::fmt;

/// A semantic URI with metadata
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticEntry {
    /// Full URI (e.g., <http://www.wikidata.org/entity/Q1490>)
    pub uri: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Source ontology
    pub source: OntologySource,
}

impl SemanticEntry {
    /// Create a new semantic entry
    pub fn new(uri: impl Into<String>, confidence: f32, source: OntologySource) -> Self {
        Self {
            uri: uri.into(),
            confidence: confidence.clamp(0.0, 1.0),
            source,
        }
    }

    /// Get compact URI (CURIE) form
    pub fn compact_uri(&self) -> String {
        compact_uri(&self.uri)
    }

    /// Expand a CURIE to full URI
    pub fn expand_uri(curie: &str) -> Option<String> {
        expand_uri(curie)
    }
}

/// Compact a full URI to CURIE form
pub fn compact_uri(uri: &str) -> String {
    if let Some(id) = uri.strip_prefix("http://www.wikidata.org/entity/") {
        return format!("wd:{}", id);
    }
    if let Some(id) = uri.strip_prefix("http://dbpedia.org/resource/") {
        return format!("dbr:{}", id);
    }
    if let Some(id) = uri.strip_prefix("http://schema.org/") {
        return format!("schema:{}", id);
    }
    uri.to_string()
}

/// Expand a CURIE to full URI
pub fn expand_uri(curie: &str) -> Option<String> {
    if let Some(id) = curie.strip_prefix("wd:") {
        return Some(format!("http://www.wikidata.org/entity/{}", id));
    }
    if let Some(id) = curie.strip_prefix("dbr:") {
        return Some(format!("http://dbpedia.org/resource/{}", id));
    }
    if let Some(id) = curie.strip_prefix("schema:") {
        return Some(format!("http://schema.org/{}", id));
    }
    if curie.starts_with("http://") || curie.starts_with("https://") {
        return Some(curie.to_string());
    }
    None
}

/// Detect ontology source from URI
pub fn detect_source(uri: &str) -> OntologySource {
    if uri.contains("wikidata.org") {
        OntologySource::Wikidata
    } else if uri.contains("dbpedia.org") {
        OntologySource::DBpedia
    } else if uri.contains("schema.org") {
        OntologySource::SchemaOrg
    } else {
        OntologySource::Custom
    }
}

/// Entity type for semantic classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    /// Person (人物)
    Person,
    /// Place/Location (場所)
    Place,
    /// Organization (組織)
    Organization,
    /// Event (イベント)
    Event,
    /// Work/Creative work (作品)
    Work,
    /// Product (製品)
    Product,
    /// Other/Unknown
    Other,
}

impl EntityType {
    /// Get schema.org type URI
    pub fn schema_org_type(&self) -> &'static str {
        match self {
            EntityType::Person => "http://schema.org/Person",
            EntityType::Place => "http://schema.org/Place",
            EntityType::Organization => "http://schema.org/Organization",
            EntityType::Event => "http://schema.org/Event",
            EntityType::Work => "http://schema.org/CreativeWork",
            EntityType::Product => "http://schema.org/Product",
            EntityType::Other => "http://schema.org/Thing",
        }
    }
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityType::Person => write!(f, "Person"),
            EntityType::Place => write!(f, "Place"),
            EntityType::Organization => write!(f, "Organization"),
            EntityType::Event => write!(f, "Event"),
            EntityType::Work => write!(f, "Work"),
            EntityType::Product => write!(f, "Product"),
            EntityType::Other => write!(f, "Other"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_uri() {
        assert_eq!(
            compact_uri("http://www.wikidata.org/entity/Q1490"),
            "wd:Q1490"
        );
        assert_eq!(
            compact_uri("http://dbpedia.org/resource/Tokyo"),
            "dbr:Tokyo"
        );
    }

    #[test]
    fn test_expand_uri_wikidata() {
        assert_eq!(
            expand_uri("wd:Q1490"),
            Some("http://www.wikidata.org/entity/Q1490".to_string())
        );
    }

    #[test]
    fn test_expand_uri_dbpedia() {
        assert_eq!(
            expand_uri("dbr:Tokyo"),
            Some("http://dbpedia.org/resource/Tokyo".to_string())
        );
    }

    #[test]
    fn test_detect_source() {
        assert_eq!(
            detect_source("http://www.wikidata.org/entity/Q1490"),
            OntologySource::Wikidata
        );
        assert_eq!(
            detect_source("http://dbpedia.org/resource/Tokyo"),
            OntologySource::DBpedia
        );
    }

    #[test]
    fn test_entity_type_schema_org() {
        assert_eq!(
            EntityType::Person.schema_org_type(),
            "http://schema.org/Person"
        );
    }
}
