# mecrab-builder

Semantic dictionary builder for MeCrab - Wikidata/Wikipedia data pipeline.

## Purpose

Build semantic-enriched dictionaries from Wikidata/Wikipedia dumps.

## Architecture

```
Wikidata JSON Dump ──┐
                     ├──▶ Surface → URI Index ──▶ Merge with Dictionary CSV
Wikipedia Abstracts ─┘                                      │
                                                            ▼
                                                   Extended Dictionary
                                                            │
                                                            ▼
                                                   SemanticPool Binary
```

## Why a Separate Crate?

Isolates heavy dependencies (tokio, reqwest, flate2) from the lightweight core library.

| Crate | Dependencies | Size |
|-------|--------------|------|
| mecrab | ~10 | Lightweight |
| mecrab-builder | ~100+ | Heavy |

## Usage

Via CLI:
```bash
# Download Wikidata dump
mecrab-builder download --source wikidata --output ./data/

# Build semantic index
mecrab-builder index \
    --wikidata ./data/latest-all.json.gz \
    --output ./data/wikidata-index.bin

# Enrich dictionary
mecrab-builder enrich \
    --dictionary ./ipadic.csv \
    --index ./data/wikidata-index.bin \
    --output ./extended-ipadic.csv
```

Via kizame (with --features builder):
```bash
kizame build \
  --source ipadic.csv \
  --wikidata latest-all.json.gz \
  --output ./semantic-dic
```

As library:
```rust
use mecrab_builder::{BuildConfig, build_dictionary_sync};

let config = BuildConfig {
    source_csv: "ipadic.csv".into(),
    wikidata_path: Some("latest-all.json.gz".into()),
    output_dir: "./output".into(),
    ..Default::default()
};
let result = build_dictionary_sync(config)?;
```

## Index Format

WikidataIndex stores surface → URI mappings:

```
Surface → Vec<(URI, Probability)>

"東京" → [
    (Q1490, 0.95),  // Tokyo (city)
    (Q127892, 0.03) // Tokyo Station
]
```

Probability is derived from Wikipedia link frequency.

## License

MIT OR Apache-2.0
