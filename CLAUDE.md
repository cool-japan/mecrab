# MeCrab Project Instructions

## Vision

**MeCrab: The Warp-Speed Morphological Analyzer for the AI Era**

Not just a MeCab port - the world's best morphological analyzer for 2026.

```
🚀 Blazing Fast: SIMD-accelerated, 5x faster than MeCab
🧠 Smart: Optional Neural Reranking for ambiguous phrases
🔄 Live: Hot-swappable dictionaries without restart
🦀 Pure Rust: Memory safe, thread safe, panic free
```

## Development Guidelines

* Follow the no warnings policy (clippy + rustc)
* Use latest crate versions (see global CLAUDE.md)
* Target IPADIC/UniDic/NEologd compatibility
* SIMD-first design for hot paths
* Cache-line aware data structures

## Architecture

See `mecrab.md` for the full architecture blueprint.
See `ROADMAP.md` for the implementation roadmap.

## Immediate Implementation Priorities

### Phase 1: Performance Foundation
1. **SIMD Viterbi** - Vectorize cost computation in forward pass
2. **Connection Matrix Optimization** - Cache-friendly layout for matrix lookups
3. **Benchmark Suite** - criterion-based benchmarks for regression testing

### Phase 2: Dictionary Revolution
1. **Overlay Dictionary** - In-memory layer for live word additions
2. **Universal Dictionary Interface** - Abstract over IPADIC/UniDic/NEologd

### Phase 3: Platform Expansion
1. **WASM Build** - Browser and edge deployment
2. **Python Bindings** - PyO3-based pip package

## Performance Targets

| Metric | MeCab | MeCrab Target |
|--------|-------|---------------|
| Short text (10 chars) | 50 μs | 15 μs |
| Medium text (100 chars) | 200 μs | 40 μs |
| Batch (1000 texts) | 2 s | 100 ms |
