# mecrab TODO

## Completed

### Core
- [x] Dictionary loading (sys.dic, matrix.def, char.def, unk.def)
- [x] Double-Array Trie implementation
- [x] Viterbi algorithm
- [x] Memory-mapped file support
- [x] Character category handling
- [x] Unknown word processing
- [x] Live dictionary overlay
- [x] Thread-safe design

### SIMD Optimization
- [x] Portable SIMD (nightly `std::simd`)
- [x] 8-way parallel i32 batch cost addition
- [x] SIMD minimum finding with lane reduction
- [x] SIMD predecessor batch processing
- [x] Automatic vectorization across platforms

### Semantic Enrichment
- [x] SemanticPool (5-byte compact entries)
- [x] TokenExtension for multi-candidate URIs
- [x] JSON-LD export
- [x] RDF/Turtle/N-Triples/N-Quads export
- [x] SPARQL query generation
- [x] Disambiguation strategies

### Advanced Features
- [x] N-best path search (A* algorithm)
- [x] Cost analysis and profiling
- [x] Streaming text processing
- [x] Sentence boundary detection
- [x] Text normalization (NFKC, width, case)
- [x] Phonetic transduction (Kana/Romaji/X-SAMPA/IPA)
- [x] User dictionary persistence
- [x] Benchmarking utilities

### Visualization
- [x] Lattice visualization (DOT/Graphviz)
- [x] DotBuilder for configurable output
- [x] Multiple layout directions (LR, TB, RL, BT)
- [x] Node shape customization
- [x] Connection cost display

## Planned

### Performance
- [x] AVX-512 Viterbi optimization (implemented as AVX2/SSE4.1 — the production-relevant x86_64 targets)
- [x] SoA (Struct of Arrays) layout — ViterbiTable with separate hot costs / cold backtrack vecs
- [x] Dictionary preloading/caching
- [x] Character info lookup caching (CharDefCached — 256-slot direct-mapped cache)

### Features
- [x] ARM NEON SIMD support — vld1q_s32/vminq_s32/vminvq_s32 in viterbi/simd.rs
- [x] WASM SIMD support

### Batch API
- [x] Batch parsing API improvements (parse_batch_with_progress, parse_iter, wakati_iter, parse_nbest_batch, wakati_batch_with_progress)

### Bindings
- [x] Python bindings packaging
- [x] WASM bindings packaging
