//! Token extension for semantic URI linking
//!
//! Provides multi-candidate semantic URI linking for morphemes.

use super::{OntologySource, SemanticEntry, SemanticPool};

/// Type alias for entity references (same as SemanticEntry)
pub type EntityReference = SemanticEntry;

/// Extension for tokens with semantic URIs
#[derive(Debug, Clone, Default)]
pub struct TokenExtension {
    /// Starting index in SemanticPool (0 = no semantics)
    pub semantic_start: u32,
    /// Number of semantic candidates (max 255)
    pub semantic_count: u8,
}

impl TokenExtension {
    /// Create a new empty extension
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an extension with semantic candidates
    pub fn with_semantics(start: u32, count: u8) -> Self {
        Self {
            semantic_start: start,
            semantic_count: count,
        }
    }

    /// Check if this token has semantic information
    pub fn has_semantics(&self) -> bool {
        self.semantic_count > 0
    }

    /// Get the number of semantic candidates
    pub fn candidate_count(&self) -> usize {
        self.semantic_count as usize
    }

    /// Get semantic entries from a pool
    pub fn get_entries(&self, pool: &SemanticPool) -> Vec<SemanticEntry> {
        if !self.has_semantics() {
            return Vec::new();
        }

        let mut entries = Vec::with_capacity(self.semantic_count as usize);
        for i in 0..self.semantic_count as u32 {
            let id = self.semantic_start + i;
            if let Some(uri) = pool.get(id) {
                let confidence = pool.get_confidence(id).unwrap_or(0.0);
                entries.push(SemanticEntry::new(
                    uri,
                    confidence,
                    OntologySource::Wikidata,
                ));
            }
        }
        entries
    }

    /// Get the best (highest confidence) semantic entry
    pub fn best_entry(&self, pool: &SemanticPool) -> Option<SemanticEntry> {
        self.get_entries(pool).into_iter().next()
    }
}

/// Extended morpheme with semantic information
#[derive(Debug, Clone)]
pub struct ExtendedMorpheme {
    /// Surface form
    pub surface: String,
    /// Part-of-speech tag
    pub pos: String,
    /// Reading (katakana)
    pub reading: Option<String>,
    /// Pronunciation
    pub pronunciation: Option<String>,
    /// Base form (lemma)
    pub base_form: Option<String>,
    /// Semantic extension
    pub extension: TokenExtension,
}

impl ExtendedMorpheme {
    /// Create a new extended morpheme
    pub fn new(surface: impl Into<String>, pos: impl Into<String>) -> Self {
        Self {
            surface: surface.into(),
            pos: pos.into(),
            reading: None,
            pronunciation: None,
            base_form: None,
            extension: TokenExtension::new(),
        }
    }

    /// Set reading
    pub fn with_reading(mut self, reading: impl Into<String>) -> Self {
        self.reading = Some(reading.into());
        self
    }

    /// Set pronunciation
    #[must_use]
    pub fn with_pronunciation(mut self, pronunciation: impl Into<String>) -> Self {
        self.pronunciation = Some(pronunciation.into());
        self
    }

    /// Set base form
    #[must_use]
    pub fn with_base_form(mut self, base_form: impl Into<String>) -> Self {
        self.base_form = Some(base_form.into());
        self
    }

    /// Set semantic extension
    #[must_use]
    pub fn with_semantics(mut self, start: u32, count: u8) -> Self {
        self.extension = TokenExtension::with_semantics(start, count);
        self
    }

    /// Check if this morpheme has semantic information
    pub fn has_semantics(&self) -> bool {
        self.extension.has_semantics()
    }

    /// Get semantic URIs from a pool
    pub fn semantic_uris(&self, pool: &SemanticPool) -> Vec<String> {
        self.extension
            .get_entries(pool)
            .into_iter()
            .map(|e| e.uri)
            .collect()
    }
}

/// Builder for creating extended analysis results
#[derive(Debug, Default)]
pub struct ExtendedResultBuilder {
    morphemes: Vec<ExtendedMorpheme>,
}

impl ExtendedResultBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a morpheme
    pub fn add(&mut self, morpheme: ExtendedMorpheme) -> &mut Self {
        self.morphemes.push(morpheme);
        self
    }

    /// Build the result
    pub fn build(self) -> Vec<ExtendedMorpheme> {
        self.morphemes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_extension_default() {
        let ext = TokenExtension::new();
        assert!(!ext.has_semantics());
        assert_eq!(ext.candidate_count(), 0);
    }

    #[test]
    fn test_token_extension_with_semantics() {
        let ext = TokenExtension::with_semantics(10, 3);
        assert!(ext.has_semantics());
        assert_eq!(ext.candidate_count(), 3);
        assert_eq!(ext.semantic_start, 10);
    }

    #[test]
    fn test_extended_morpheme_builder() {
        let morpheme = ExtendedMorpheme::new("東京", "名詞")
            .with_reading("トウキョウ")
            .with_semantics(1, 2);

        assert_eq!(morpheme.surface, "東京");
        assert_eq!(morpheme.pos, "名詞");
        assert_eq!(morpheme.reading, Some("トウキョウ".to_string()));
        assert!(morpheme.has_semantics());
    }

    #[test]
    fn test_result_builder() {
        let mut builder = ExtendedResultBuilder::new();
        builder.add(ExtendedMorpheme::new("東京", "名詞"));
        builder.add(ExtendedMorpheme::new("都", "名詞"));

        let result = builder.build();
        assert_eq!(result.len(), 2);
    }
}
