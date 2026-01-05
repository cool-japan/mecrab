# MeCrab - High-Performance Morphological Analyzer

<p align="center">
  <img src="docs/mecrab.png" alt="MeCrab" width="800">
</p>

A pure Rust implementation of a morphological analyzer compatible with MeCab dictionaries (IPADIC format).

## Workspace Structure

```
mecrab/
├── mecrab/          # Core library - runtime morphological analyzer
├── mecrab-builder/  # Data pipeline - Wikidata/Wikipedia processing
├── mecrab-word2vec/ # Word2Vec training with Hogwild! parallelization
├── kizame/          # CLI tool - KizaMe (刻め!)
├── fuzz/            # Fuzz testing
└── docs/            # Engineering directives and design docs
```

| Crate | Description | Dependencies |
|-------|-------------|--------------|
| `mecrab` | Core runtime library | Minimal (lightweight) |
| `mecrab-builder` | Semantic dictionary builder | Heavy (tokio, reqwest) |
| `mecrab-word2vec` | High-performance Word2Vec training | rayon, rand |
| `kizame` | CLI with optional builder | Lightweight default, heavy with `--features builder` |

## Features

- **High Performance**: SIMD-accelerated Viterbi (AVX2), parallel batch processing
- **Zero-copy Parsing**: Memory-mapped dictionary loading
- **Thread-safe**: Safe concurrent access using Rust's ownership model
- **Live Dictionary Updates**: Add/remove words at runtime without restart
- **Semantic Enrichment**: Wikidata URI linking, JSON-LD/RDF export (Turtle, N-Triples, N-Quads)
- **N-best Paths**: A* algorithm for multiple path analysis
- **Streaming Processing**: Sentence boundary detection for large text
- **Text Normalization**: NFKC, width conversion, case folding
- **Phonetic Transduction**: Kana ↔ Romaji, X-SAMPA, IPA conversion
- **Word Embeddings**: Pure Rust Word2Vec with Hogwild! parallelization (83% efficiency)
- **Interactive TUI Debugger**: Lattice explorer with cost visualization ([screenshot](docs/tui.png))
- **Cross-platform**: Native binaries, WASM, Python bindings

> **Note:** When using `--with-ipa` with default text output, pipe through `cat` for proper terminal display of Unicode IPA characters. This is not required for JSON-LD output.

## Installation

### CLI (KizaMe)

```bash
# Default (lightweight)
cargo install kizame

# With Wikidata builder
cargo install kizame --features builder
```

### Rust Library

```toml
[dependencies]
mecrab = "0.1"

# Optional features
mecrab = { version = "0.1", features = ["json", "parallel"] }
```

## Quick Start

### CLI Usage

```bash
# Initialize dictionary (finds system IPADIC)
kizame dict init

# Interactive parsing
echo "すもももももももものうち" | kizame

# With dictionary path
kizame -d /var/lib/mecab/dic/ipadic-utf8 parse

# Wakati (space-separated)
echo "日本語の形態素解析" | kizame -w

# JSON output
echo "東京都" | kizame -O json

# With IPA pronunciation (default format - requires | cat for terminal display)
echo "こんにちは" | kizame parse --with-ipa | cat

# With word embeddings (default format)
echo "私は学生です" | kizame parse --with-vector -v /path/to/vectors.bin | cat

# With both IPA and vectors (default format)
echo "東京に行く" | kizame parse --with-ipa --with-vector -v /path/to/vectors.bin | cat

# JSON-LD with semantic URIs
echo "東京に行く" | kizame -O jsonld --with-semantic

# JSON-LD with IPA pronunciation
echo "こんにちは" | kizame -O jsonld --with-ipa

# JSON-LD with word embeddings
echo "私は学生です" | kizame -O jsonld --with-vector -v /path/to/vectors.bin

# RDF formats (Turtle, N-Triples, N-Quads)
echo "東京に行く" | kizame -O turtle
echo "東京に行く" | kizame -O ntriples
echo "東京に行く" | kizame -O nquads

# Dictionary info
kizame dict info
kizame dict dump -d /path/to/dic

# Interactive TUI debugger
kizame explore "東京に行く"
```

### Rust API

```rust
use mecrab::MeCrab;

let mecrab = MeCrab::new()?;
let result = mecrab.parse("すもももももももものうち")?;
println!("{}", result);

// Add custom words at runtime
mecrab.add_word("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000);

// N-best paths
use mecrab::viterbi::NbestSearch;
let nbest = NbestSearch::new(&mecrab);
for path in nbest.search("東京", 5)? {
    println!("Cost: {}, Path: {:?}", path.total_cost, path.nodes);
}

// Streaming processing
use mecrab::stream::SentenceReader;
let reader = SentenceReader::new(input);
for sentence in reader {
    let result = mecrab.parse(&sentence)?;
}

// Phonetic conversion
use mecrab::phonetic::PhoneticTransducer;
let transducer = PhoneticTransducer::new();
let romaji = transducer.to_romaji("こんにちは"); // "konnichiha"
```

## Training Word Embeddings

KizaMe includes a high-performance pure Rust Word2Vec implementation with **Hogwild! lock-free parallelization**:

### Performance

- **83% parallel efficiency** on 6 cores (499.7% CPU usage)
- **~500K words/sec/core** training throughput
- **2.27x speedup** vs mutex-based baseline
- Memory-mapped MCV1 format for instant vector loading

### Training Pipeline

```bash
# Extract vocabulary
kizame dict dump -d /var/lib/mecab/dic/ipadic-utf8 --vocab > vocab.txt
MAX_WORD_ID=$(tail -1 vocab.txt | cut -f1)

# Parse corpus to word_id sequences
cat corpus.txt | kizame parse --wakati-word-id > corpus_ids.txt

# Train Word2Vec (MCV1 binary format - recommended)
kizame vectors train \
  -i corpus_ids.txt \
  -o vectors.bin \
  -f mcv1 \
  --max-word-id $MAX_WORD_ID \
  --size 100 \
  --window 5 \
  --negative 5 \
  --epochs 3 \
  --threads 6  # Automatically uses Hogwild!

# Use trained vectors
echo "東京に行く" | kizame parse --with-vector -v vectors.bin --with-ipa | cat
```

**Technical Details:** See [`mecrab-word2vec/IMPLEMENTATION.md`](mecrab-word2vec/IMPLEMENTATION.md) for Hogwild! algorithm details, safety analysis, and performance benchmarks.

**Training Guide:** See `docs/WORD2VEC_TRAINING_GUIDE.md` for full training pipeline with Japanese Wikipedia corpus.

## Building Semantic Dictionaries

With `--features builder`, KizaMe can build semantic-enriched dictionaries from Wikidata:

```bash
# Download Wikidata dump
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz

# Build extended dictionary
kizame build \
  --source ipadic.csv \
  --wikidata latest-all.json.gz \
  --output ./semantic-dic \
  --max-candidates 5

# Use semantic enrichment
echo "東京に行く" | kizame -d ./semantic-dic -O jsonld --with-semantic
```

Output includes Wikidata URIs:
```json
{
  "@context": { "wd": "http://www.wikidata.org/entity/", ... },
  "tokens": [
    {
      "surface": "東京",
      "pos": "名詞",
      "wcost": 3003,
      "entities": [
        {"@id": "wd:Q1490", "confidence": 0.95}
      ]
    }
  ]
}
```

## Development

```bash
# Run tests
cargo nextest run

# Clippy (no warnings policy)
cargo clippy --workspace

# Benchmarks
cargo bench -p mecrab

# Fuzz testing
cd fuzz && cargo +nightly fuzz run viterbi
```

## Statistics

- ~11,000 lines of Rust
- 174 tests (all passing)
- 4 fuzz targets
- 0 clippy warnings
- 83% parallel efficiency (Word2Vec training)

## License

MIT OR Apache-2.0

## Copyright

Copyright 2026 COOLJAPAN OU (Team KitaSan)
