//! RDF/Turtle/N-Triples export for semantic analysis results

use super::SemanticEntry;

/// Standard RDF prefixes
pub const PREFIXES: &[(&str, &str)] = &[
    ("wd", "http://www.wikidata.org/entity/"),
    ("wdt", "http://www.wikidata.org/prop/direct/"),
    ("dbr", "http://dbpedia.org/resource/"),
    ("dbo", "http://dbpedia.org/ontology/"),
    ("schema", "http://schema.org/"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("mecrab", "http://mecrab.io/ns#"),
];

/// Escape a string for Turtle/N-Triples
pub fn escape_turtle(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c => result.push(c),
        }
    }
    result
}

/// Expand a CURIE to full URI
pub fn expand_uri(curie: &str) -> String {
    for (prefix, uri) in PREFIXES {
        if let Some(local) = curie.strip_prefix(&format!("{}:", prefix)) {
            return format!("{}{}", uri, local);
        }
    }
    curie.to_string()
}

/// Export tokens as Turtle format
pub fn export_turtle(
    tokens: &[(String, String, Option<String>, Vec<SemanticEntry>)],
    base_uri: &str,
) -> String {
    let mut turtle = String::new();

    // Prefixes
    turtle.push_str("@base <");
    turtle.push_str(base_uri);
    turtle.push_str("> .\n");

    for (prefix, uri) in PREFIXES {
        turtle.push_str(&format!("@prefix {}: <{}> .\n", prefix, uri));
    }
    turtle.push('\n');

    // Tokens
    for (i, (surface, pos, reading, entities)) in tokens.iter().enumerate() {
        let token_uri = format!("<#token{}>", i);

        turtle.push_str(&token_uri);
        turtle.push_str(" a mecrab:Token ;\n");
        turtle.push_str(&format!(
            "    mecrab:surface \"{}\" ;\n",
            escape_turtle(surface)
        ));
        turtle.push_str(&format!("    mecrab:pos \"{}\"", escape_turtle(pos)));

        if let Some(reading) = reading {
            turtle.push_str(&format!(
                " ;\n    mecrab:reading \"{}\"",
                escape_turtle(reading)
            ));
        }

        if !entities.is_empty() {
            turtle.push_str(" ;\n    mecrab:entity ");
            let entity_strs: Vec<String> =
                entities.iter().map(|e| format!("<{}>", e.uri)).collect();
            turtle.push_str(&entity_strs.join(", "));
        }

        turtle.push_str(" .\n\n");
    }

    turtle
}

/// Export as N-Triples (line-based RDF)
pub fn export_ntriples(
    tokens: &[(String, String, Option<String>, Vec<SemanticEntry>)],
    base_uri: &str,
) -> String {
    let mut ntriples = String::new();

    for (i, (surface, pos, reading, entities)) in tokens.iter().enumerate() {
        let token_uri = format!("{}#token{}", base_uri, i);

        // Type triple
        ntriples.push_str(&format!(
            "<{}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://mecrab.io/ns#Token> .\n",
            token_uri
        ));

        // Surface
        ntriples.push_str(&format!(
            "<{}> <http://mecrab.io/ns#surface> \"{}\" .\n",
            token_uri,
            escape_turtle(surface)
        ));

        // POS
        ntriples.push_str(&format!(
            "<{}> <http://mecrab.io/ns#pos> \"{}\" .\n",
            token_uri,
            escape_turtle(pos)
        ));

        // Reading
        if let Some(reading) = reading {
            ntriples.push_str(&format!(
                "<{}> <http://mecrab.io/ns#reading> \"{}\" .\n",
                token_uri,
                escape_turtle(reading)
            ));
        }

        // Entities
        for entity in entities {
            ntriples.push_str(&format!(
                "<{}> <http://mecrab.io/ns#entity> <{}> .\n",
                token_uri, entity.uri
            ));
        }
    }

    ntriples
}

/// Export as N-Quads (with graph URI)
pub fn export_nquads(
    tokens: &[(String, String, Option<String>, Vec<SemanticEntry>)],
    base_uri: &str,
    graph_uri: &str,
) -> String {
    let mut nquads = String::new();

    for (i, (surface, pos, reading, entities)) in tokens.iter().enumerate() {
        let token_uri = format!("{}#token{}", base_uri, i);

        // Type quad
        nquads.push_str(&format!(
            "<{}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://mecrab.io/ns#Token> <{}> .\n",
            token_uri, graph_uri
        ));

        // Surface
        nquads.push_str(&format!(
            "<{}> <http://mecrab.io/ns#surface> \"{}\" <{}> .\n",
            token_uri,
            escape_turtle(surface),
            graph_uri
        ));

        // POS
        nquads.push_str(&format!(
            "<{}> <http://mecrab.io/ns#pos> \"{}\" <{}> .\n",
            token_uri,
            escape_turtle(pos),
            graph_uri
        ));

        // Reading
        if let Some(reading) = reading {
            nquads.push_str(&format!(
                "<{}> <http://mecrab.io/ns#reading> \"{}\" <{}> .\n",
                token_uri,
                escape_turtle(reading),
                graph_uri
            ));
        }

        // Entities
        for entity in entities {
            nquads.push_str(&format!(
                "<{}> <http://mecrab.io/ns#entity> <{}> <{}> .\n",
                token_uri, entity.uri, graph_uri
            ));
        }
    }

    nquads
}

/// Build a SPARQL query for entity mentions
pub fn sparql_entity_mentions(entity_uri: &str) -> String {
    format!(
        "SELECT ?token ?surface ?pos WHERE {{\n  ?token mecrab:entity <{}> ;\n         mecrab:surface ?surface ;\n         mecrab:pos ?pos .\n}}",
        entity_uri
    )
}

/// SPARQL query builder
pub struct SparqlBuilder {
    prefixes: Vec<(String, String)>,
    select_vars: Vec<String>,
    where_patterns: Vec<String>,
    filters: Vec<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

impl SparqlBuilder {
    /// Create a new SPARQL query builder with default prefixes
    pub fn new() -> Self {
        Self {
            prefixes: PREFIXES
                .iter()
                .map(|(p, u)| ((*p).to_string(), (*u).to_string()))
                .collect(),
            select_vars: Vec::new(),
            where_patterns: Vec::new(),
            filters: Vec::new(),
            limit: None,
            offset: None,
        }
    }

    /// Add SELECT variables to the query
    pub fn select(mut self, vars: &[&str]) -> Self {
        self.select_vars = vars.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Add a WHERE pattern to the query
    pub fn where_pattern(mut self, pattern: &str) -> Self {
        self.where_patterns.push(pattern.to_string());
        self
    }

    /// Add a FILTER expression to the query
    pub fn filter(mut self, filter: &str) -> Self {
        self.filters.push(filter.to_string());
        self
    }

    /// Set the LIMIT clause
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set the OFFSET clause
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    /// Build the SPARQL query string
    pub fn build(self) -> String {
        let mut query = String::new();

        // Prefixes
        for (prefix, uri) in &self.prefixes {
            query.push_str(&format!("PREFIX {}: <{}>\n", prefix, uri));
        }
        query.push('\n');

        // SELECT
        query.push_str("SELECT ");
        if self.select_vars.is_empty() {
            query.push('*');
        } else {
            query.push_str(
                &self
                    .select_vars
                    .iter()
                    .map(|v| format!("?{}", v))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        query.push_str(" WHERE {\n");

        // WHERE patterns
        for pattern in &self.where_patterns {
            query.push_str("  ");
            query.push_str(pattern);
            query.push_str(" .\n");
        }

        // FILTERs
        for filter in &self.filters {
            query.push_str("  FILTER(");
            query.push_str(filter);
            query.push_str(")\n");
        }

        query.push('}');

        // LIMIT/OFFSET
        if let Some(limit) = self.limit {
            query.push_str(&format!("\nLIMIT {}", limit));
        }
        if let Some(offset) = self.offset {
            query.push_str(&format!("\nOFFSET {}", offset));
        }

        query
    }
}

impl Default for SparqlBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::OntologySource;

    #[test]
    fn test_escape_turtle() {
        assert_eq!(escape_turtle("hello"), "hello");
        assert_eq!(escape_turtle("hello\"world"), "hello\\\"world");
    }

    #[test]
    fn test_expand_uri() {
        assert_eq!(
            expand_uri("wd:Q1490"),
            "http://www.wikidata.org/entity/Q1490"
        );
        assert_eq!(expand_uri("dbr:Tokyo"), "http://dbpedia.org/resource/Tokyo");
    }

    #[test]
    fn test_turtle_export() {
        let tokens = vec![(
            "東京".to_string(),
            "名詞".to_string(),
            Some("トウキョウ".to_string()),
            vec![SemanticEntry::new(
                "http://www.wikidata.org/entity/Q1490",
                0.95,
                OntologySource::Wikidata,
            )],
        )];

        let turtle = export_turtle(&tokens, "http://example.org/doc1");
        assert!(turtle.contains("@prefix wd:"));
        assert!(turtle.contains("mecrab:surface \"東京\""));
    }

    #[test]
    fn test_turtle_prefixes() {
        let tokens = vec![];
        let turtle = export_turtle(&tokens, "http://example.org/");
        assert!(turtle.contains("@prefix mecrab:"));
        assert!(turtle.contains("@prefix wd:"));
    }

    #[test]
    fn test_ntriples_export() {
        let tokens = vec![("テスト".to_string(), "名詞".to_string(), None, vec![])];

        let ntriples = export_ntriples(&tokens, "http://example.org/doc1");
        assert!(ntriples.contains("<http://example.org/doc1#token0>"));
        assert!(ntriples.contains("<http://mecrab.io/ns#Token>"));
    }

    #[test]
    fn test_nquads_export() {
        let tokens = vec![("テスト".to_string(), "名詞".to_string(), None, vec![])];

        let nquads = export_nquads(
            &tokens,
            "http://example.org/doc1",
            "http://example.org/graph1",
        );
        assert!(nquads.contains("<http://example.org/graph1>"));
    }

    #[test]
    fn test_sparql_entity_mentions() {
        let query = sparql_entity_mentions("http://www.wikidata.org/entity/Q1490");
        assert!(query.contains("SELECT ?token ?surface ?pos"));
        assert!(query.contains("mecrab:entity"));
    }

    #[test]
    fn test_sparql_builder() {
        let query = SparqlBuilder::new()
            .select(&["surface", "entity"])
            .where_pattern("?token mecrab:surface ?surface")
            .where_pattern("?token mecrab:entity ?entity")
            .filter("?entity = wd:Q1490")
            .limit(10)
            .build();

        assert!(query.contains("SELECT ?surface ?entity"));
        assert!(query.contains("LIMIT 10"));
    }
}
