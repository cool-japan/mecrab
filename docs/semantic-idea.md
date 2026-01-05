# Semantic Enrichment Design

Design document for entity linking and knowledge graph integration.

---

## Core Concept

Transform morphological analysis from simple POS tagging to **knowledge graph connectivity** by attaching Wikidata/DBpedia URIs to tokens.

---

## Use Cases

### 1. Named Entity Linking (NEL)

```
Input:  "東京都庁で会議があった"
Output: 東京都庁 → wd:Q1046099 (Tokyo Metropolitan Government Building)
```

### 2. Disambiguation

```
Input:  "りんごを食べた"
Output: りんご → wd:Q89 (Apple - fruit), confidence: 0.95
        りんご → wd:Q312 (Apple Inc.), confidence: 0.05
```

### 3. Knowledge Graph Queries

```sparql
SELECT ?name ?population WHERE {
  wd:Q1490 rdfs:label ?name .
  wd:Q1490 wdt:P1082 ?population .
  FILTER(LANG(?name) = "ja")
}
```

---

## Implementation Phases

### Phase 1: Static Linking (Implemented)

- Wikidata dump → Surface index
- Dictionary CSV merge
- Binary SemanticPool output
- JSON-LD / RDF export

### Phase 2: Context Disambiguation (Implemented)

- Use surrounding words for disambiguation
- Popularity-based prior probabilities
- POS tag hints (固有名詞 → higher entity probability)

### Phase 3: Online Resolution (Future)

- Wikidata API fallback for unknown entities
- Real-time entity linking
- Confidence calibration

---

## Data Format

### Token Extension

```rust
pub struct TokenExtension {
    pub semantic_start: u32,  // Index into SemanticPool
    pub semantic_count: u8,   // Number of candidates (max 255)
}
```

### SemanticPool Entry (5 bytes)

```
[offset:3][confidence:1][source+flags:1]
```

---

## Export Formats

| Format | Use Case | File |
|--------|----------|------|
| JSON-LD | Web applications | `semantic/jsonld.rs` |
| RDF/Turtle | Knowledge graphs | `semantic/rdf.rs` |
| N-Triples | Bulk processing | `semantic/rdf.rs` |
| N-Quads | Named graphs | `semantic/rdf.rs` |
| SPARQL | Query building | `semantic/rdf.rs` |

---

## Disambiguation Strategies

```rust
pub enum DisambiguationStrategy {
    HighestConfidence,    // Pick top confidence
    PopularityPrior,      // Use link frequency
    ContextWindow { size: usize },  // Use surrounding tokens
    None,                 // Return all candidates
}
```

---

## Memory Efficiency

| Storage | 1M URIs |
|---------|---------|
| Full strings | ~100MB |
| SemanticPool | ~5MB |
| **Savings** | **95%** |

---

*Copyright 2026 COOLJAPAN OU (Team KitaSan)*
