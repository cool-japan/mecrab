# mecrab

Core runtime library for MeCrab morphological analyzer.

## Features

- **MeCab Compatible**: Works with IPADIC/UniDic dictionaries
- **High Performance**: Memory-mapped dictionaries, SIMD-optimized Viterbi (AVX2)
- **Thread-safe**: Safe concurrent access
- **Live Updates**: Add/remove words at runtime
- **Semantic Linking**: Wikidata URI attachment with JSON-LD/RDF export
- **N-best Search**: A* algorithm for multiple path analysis
- **Streaming**: Sentence boundary detection for large text processing
- **Phonetic Transduction**: Kana/Romaji/X-SAMPA/IPA conversion

## Installation

```toml
[dependencies]
mecrab = "0.1"
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `json` | JSON output format |
| `parallel` | Parallel batch processing (rayon) |
| `simd` | SIMD optimizations (AVX2) |
| `wasm` | WebAssembly bindings |
| `python` | Python bindings (PyO3) |

## Usage

```rust
use mecrab::MeCrab;

let mecrab = MeCrab::new()?;
let result = mecrab.parse("すもももももももものうち")?;
println!("{}", result);

// Add custom words
mecrab.add_word("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000);

// N-best paths
use mecrab::viterbi::NbestSearch;
let nbest = NbestSearch::new(&mecrab);
for path in nbest.search("東京", 5)? {
    println!("Cost: {}", path.total_cost);
}

// Phonetic conversion
use mecrab::phonetic::PhoneticTransducer;
let transducer = PhoneticTransducer::new();
println!("{}", transducer.to_romaji("こんにちは")); // konnichiha
```

## Module Structure

```
mecrab/src/
├── lib.rs           # Public API
├── dict/            # Dictionary loading
│   ├── mod.rs       # Token, SysDic, OverlayDictionary
│   └── user_dict.rs # User dictionary persistence
├── lattice/         # Lattice construction
├── viterbi/         # Viterbi algorithm
│   ├── mod.rs       # Core Viterbi
│   ├── simd.rs      # AVX2 acceleration
│   ├── nbest.rs     # N-best A* search
│   └── analysis.rs  # Cost analysis
├── semantic/        # Semantic enrichment
│   ├── mod.rs       # SemanticEntry, EntityType
│   ├── pool.rs      # SemanticPool (5-byte entries)
│   ├── jsonld.rs    # JSON-LD export
│   ├── rdf.rs       # RDF/Turtle/N-Triples export
│   ├── disambiguation.rs  # Disambiguation strategies
│   └── extension.rs # TokenExtension
├── phonetic/        # Phonetic processing
│   ├── mod.rs       # Reading extraction
│   └── transducer.rs # Kana/Romaji/X-SAMPA/IPA
├── stream.rs        # Streaming text processing
├── normalize.rs     # Text normalization
├── bench.rs         # Benchmarking utilities
└── error.rs         # Error types
```

## License

MIT OR Apache-2.0
