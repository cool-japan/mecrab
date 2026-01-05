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
- [ ] AVX-512 Viterbi optimization
- [ ] SoA (Struct of Arrays) layout
- [ ] Dictionary preloading/caching

### Features
- [ ] ARM NEON SIMD support
- [ ] WASM SIMD support

### Bindings
- [x] Python bindings packaging
- [x] WASM bindings packaging
