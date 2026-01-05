# Semantic Integration - Complete Implementation Guide

## Overview

MeCrab now supports **semantic entity linking** - mapping Japanese morphemes to knowledge graph URIs (Wikidata, DBpedia, etc.). This enables downstream applications to perform entity resolution, knowledge graph construction, and semantic search.

**Status:** ✅ Production-ready (2026-01-02)

## Architecture

### Data Flow

```
┌─────────────────┐
│ mecrab-builder  │  Build-time: Generate semantic data
└────────┬────────┘
         │ outputs
         ├─ semantic.bin      (SemanticPool: URI storage)
         └─ surface_map.json  (Surface → URIs mapping)
              │
              ▼
┌─────────────────┐
│   Dictionary    │  Runtime: Load semantic data
│  .load_with_    │
│   semantics()   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     MeCrab      │  Parse: Populate entities
│   .parse()      │  (when --with-semantic flag set)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AnalysisResult │  Output: JSON-LD with URIs
│   (JSON-LD)     │
└─────────────────┘
```

### File Formats

#### 1. `semantic.bin` (SemanticPool)
Binary format for storing URIs efficiently:
- Magic: `MCSP` (MeCrab Semantic Pool)
- Version: 1
- Entry size: 5 bytes (4-byte string offset + 1-byte length)
- Zero-copy memory-mapped access

#### 2. `surface_map.json`
JSON mapping of surface forms to URIs with confidence scores:
```json
{
  "東京": [
    ["http://www.wikidata.org/entity/Q1490", 0.95],
    ["http://www.wikidata.org/entity/Q7323", 0.82]
  ],
  "日本": [
    ["http://www.wikidata.org/entity/Q17", 0.98]
  ]
}
```

**Note:** Currently JSON for rapid development. Future: binary format for production.

## Usage

### 1. Building a Semantic Dictionary

```bash
# Using mecrab-builder (feature-gated)
kizame build \
  --source /path/to/ipadic.csv \
  --wikidata /path/to/latest-all.json.gz \
  --output /path/to/output_dict \
  --max-candidates 5

# Outputs:
#   output_dict/sys.dic
#   output_dict/matrix.bin
#   output_dict/char.bin
#   output_dict/unk.dic
#   output_dict/semantic.bin       ← NEW
#   output_dict/surface_map.json   ← NEW
```

### 2. Loading with Semantics

```rust
use mecrab::dict::Dictionary;
use std::path::Path;

// Explicit loading
let dict = Dictionary::load_with_semantics(
    Path::new("/var/lib/mecab/dic/ipadic-utf8"),
    Path::new("/path/to/semantic.bin"),
)?;

// Auto-loading (checks for semantic.bin in dicdir)
let dict = Dictionary::load(
    Path::new("/var/lib/mecab/dic/ipadic-utf8"),
)?;  // Loads semantic.bin if present
```

### 3. Parsing with Semantic Output

#### CLI Usage

```bash
# Without semantic data (minimal output)
echo "東京に行く" | kizame parse -O jsonld

# With semantic data (explicit flag required)
echo "東京に行く" | kizame parse -O jsonld --with-semantic

# With explicit semantic pool path
echo "東京に行く" | kizame parse \
  -d /var/lib/mecab/dic/ipadic-utf8 \
  -s /path/to/semantic.bin \
  --with-semantic \
  -O jsonld
```

#### Programmatic Usage

```rust
use mecrab::MeCrab;

let mecrab = MeCrab::builder()
    .dicdir(Some("/var/lib/mecab/dic/ipadic-utf8".into()))
    .semantic_pool(Some("/path/to/semantic.bin".into()))
    .with_semantic(true)  // Enable semantic output
    .build()?;

let result = mecrab.parse("東京に行く")?;

for morpheme in &result.morphemes {
    println!("{}: {:?}", morpheme.surface, morpheme.entities);
}
```

### 4. Output Formats

#### Without `--with-semantic` (Clean)
```json
{
  "@context": { ... },
  "@type": "mecrab:Analysis",
  "tokens": [
    {
      "surface": "東京",
      "pos": "名詞",
      "reading": "トウキョウ",
      "wcost": 3003
    }
  ]
}
```

#### With `--with-semantic` (Enriched)
```json
{
  "@context": { ... },
  "@type": "mecrab:Analysis",
  "tokens": [
    {
      "surface": "東京",
      "pos": "名詞",
      "reading": "トウキョウ",
      "wcost": 3003,
      "entities": [
        {"@id": "wd:Q1490", "confidence": 0.95},
        {"@id": "wd:Q7323", "confidence": 0.82}
      ]
    }
  ]
}
```

**Key Design:** `entities` field is **completely omitted** when empty (no `"entities": []` clutter).

## Performance Characteristics

### Zero-Cost Abstraction

When `--with-semantic` is **not** set:
- ✅ No entity lookup performed
- ✅ No surface_map access
- ✅ Zero overhead vs. non-semantic parsing
- ✅ Minimal output size (no entities field)

When `--with-semantic` **is** set:
- Hash map lookup per morpheme: ~O(1)
- Memory overhead: ~20-50 bytes per entity reference
- Typical cost: <5% parsing overhead

### Memory Usage

```
semantic.bin:        ~5 bytes × num_unique_URIs
surface_map.json:    ~50-100 bytes × num_surface_forms (uncompressed)
Runtime overhead:    ~Vec<EntityReference> per morpheme (empty when disabled)
```

## API Reference

### Dictionary

```rust
impl Dictionary {
    /// Load dictionary with semantic pool
    pub fn load_with_semantics(
        path: &Path,
        semantic_path: &Path,
    ) -> Result<Self>;

    // Fields (public)
    pub semantic_pool: Option<Arc<SemanticPool>>;
    pub surface_map: Option<Arc<SurfaceMap>>;
}

/// Type alias for surface → URIs mapping
pub type SurfaceMap = HashMap<String, Vec<(String, f32)>>;
```

### MeCrab

```rust
impl MeCrabBuilder {
    /// Set semantic pool path
    pub fn semantic_pool(self, path: Option<PathBuf>) -> Self;

    /// Enable semantic URI output
    pub fn with_semantic(self, enabled: bool) -> Self;
}

pub struct MeCrab {
    semantic_enabled: bool,  // Controls entity lookup
    // ...
}
```

### Morpheme

```rust
pub struct Morpheme {
    pub surface: String,
    pub pos_id: u16,
    pub wcost: i16,
    pub feature: String,
    pub entities: Vec<EntityReference>,  // Empty when disabled
}
```

### EntityReference

```rust
pub struct EntityReference {
    pub uri: String,           // Full URI
    pub confidence: f32,       // 0.0-1.0
    pub source: OntologySource, // Wikidata, DBpedia, etc.
}

pub enum OntologySource {
    Wikidata,
    DBpedia,
    Custom,
}
```

## Testing

```bash
# Unit tests
cargo test --package mecrab

# Integration test with semantic data
echo "東京" | kizame parse -O jsonld --with-semantic

# Verify no overhead when disabled
echo "東京" | kizame parse -O jsonld  # No --with-semantic

# Test TUI with semantics
kizame explore "東京に行く" -s /path/to/semantic.bin
```

## Future Enhancements

### Planned (docs/008.md Vision)

1. **Binary surface_map format**
   - Replace JSON with binary format
   - Reduce disk I/O and loading time
   - Enable incremental updates

2. **`--with-ipa` flag**
   - IPA pronunciation: `/toːkjoː/`
   - Katakana → IPA transducer
   - Zero-copy when disabled

3. **`--with-vector` flag**
   - Word embeddings output
   - Base64 encoding option for compact JSON
   - Zero-copy mmap vector store

4. **Online entity resolution**
   - Wikidata API fallback for OOV entities
   - Caching layer
   - Confidence calibration

### Performance Optimizations

- [ ] SIMD-accelerated confidence scoring
- [ ] Bloom filter for fast negative lookups
- [ ] Compressed trie for surface_map
- [ ] Multi-level caching strategy

## Troubleshooting

### Q: `--with-semantic` has no effect
**A:** Ensure semantic.bin and surface_map.json are in the correct location and loaded properly.

### Q: Empty entities array in output
**A:** This is expected when:
- Surface form not in surface_map
- `--with-semantic` not set (entities completely omitted)
- No Wikidata match during build

### Q: Performance degradation
**A:** Check if semantic_enabled is incorrectly always true. Use `--with-semantic` only when needed.

## References

- Design document: `docs/008.md` (Ultimate JSON-LD Output)
- SemanticPool format: `mecrab/src/semantic/pool.rs`
- Builder pipeline: `mecrab-builder/src/wikidata.rs`
- TUI integration: Session summary (UTF-8 fixes + scroll)

---

**Last updated:** 2026-01-02
**Contributors:** Team KitaSan (COOLJAPAN OU)
