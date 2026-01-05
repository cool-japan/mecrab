# Flyweight Semantic Pool Architecture

**Status:** ✅ Implemented in `mecrab/src/semantic/`

A naive implementation that "adds CSV columns and loads everything into memory" will cause dictionary bloat, frequent cache misses, and severely degrade analysis speed (Viterbi algorithm). Semantic Web URIs (`http://...`) are long strings with high redundancy—storing them directly is extremely inefficient.

This document describes the **Rust-idiomatic dictionary extension architecture** implemented in MeCrab that balances memory efficiency and execution speed.

## Core Concept: Flyweight Semantic Pool (Separation & Sharing)

The key idea is to **physically separate** "data required for morphological analysis (Hot Data)" from "semantic information (Cold Data)", and compress semantic data through **String Interning** (string deduplication).

## 1. Data Structure Separation

Instead of mixing large strings into the existing `sys.dic` (IPADIC-compatible binary), offload them to a separate binary file (`semantic.bin`).

### sys.dic (Morphological Data / Hot)

- Stores only data used by the Viterbi algorithm for **cost computation** and **connection validation**
- Contains no semantic data "entities"—only 4-byte `SemanticID` (u32)
- **Benefit:** Node struct size remains small, fits in CPU cache, analysis speed unaffected

### semantic.bin (Semantic Data / Cold)

- Stores actual data like URIs and JSON-LD fragments
- **Critical:** Deduplicate duplicate strings (store only once)
- **Access:** Never read during analysis (Viterbi). Only referenced during output phase (Formatter)

## 2. String Interning (Deduplication) Mechanism

For example, if "Tokyo", "Osaka", and "Kyoto" all have the type definition `http://schema.org/Place`, storing it per record is wasteful.

### Compile-time Processing (mecrab-dict-index)

1. Read CSV extended columns
2. Extract unique semantic information strings
3. Create a large string buffer (Blob) concatenated with `\0` delimiters
4. Assign each token an offset (or ID) into that Blob

### Runtime Processing (mecrab Runtime)

1. `mmap` the `semantic.bin` into virtual memory (zero load time)
2. Jump to pointer and retrieve string only when needed

## 3. Rust Implementation Design

### A. Dictionary Binary Layout Extension

Quietly append `semantic_id` to the end of existing MeCab data structures (Token):

```rust
// Token definition in dictionary (conceptual)
#[repr(C, packed)]
struct TokenInfo {
    // --- Existing IPADIC-compatible fields ---
    left_id: u16,
    right_id: u16,
    cost: i16,
    pos_id: u16,
    // ... other existing info ...

    // --- Extended field (MeCrab Extension) ---
    // Not the string entity, but an index into semantic.bin
    semantic_id: u32,
}
```

### B. SemanticPool Implementation (Zero-Copy)

Structure `semantic.bin` as a simple "offset array" + "string data":

```rust
struct SemanticPool<'a> {
    // Reference to mmap'd binary data
    data: &'a [u8],
    // Index to look up offset position in data from ID
    offsets: &'a [u32],
}

impl<'a> SemanticPool<'a> {
    // O(1) retrieval of semantic string from ID
    // Called only during display, never affects analysis logic
    fn get(&self, id: u32) -> Option<&str> {
        if id == 0 { return None; } // 0 = "no semantic info"
        let start = self.offsets[id as usize] as usize;
        let end = self.offsets[id as usize + 1] as usize;
        std::str::from_utf8(&self.data[start..end]).ok()
    }
}
```

### C. CSV Extension Specification

Dictionary maintainers add columns to the end of CSV:

```
...POS,conjugation_type,conjugation_form,base_form,reading,pronunciation,{SEMANTIC_INFO}
東京,名詞,固有名詞,地域,一般,*,*,トウキョウ,トウキョウ,トウキョウ,http://www.wikidata.org/entity/Q1490
```

When running `mecrab-dict-index`, specify `--semantic-col-index=13`. The compiler automatically performs deduplication and packing to generate `sys.dic` and `semantic.bin`.

## 4. Architecture Strengths

### Zero Analysis Speed Degradation

The analyzer (Tokenizer/Viterbi) only copies and carries around `semantic_id` (u32). No string processing occurs, so MeCab's original blazing performance is maintained regardless of how rich the semantic information becomes.

### Memory Efficiency

Common prefixes like `http://www.wikidata.org/entity/` and frequently-used tags are physically stored in only one location, dramatically compressing memory usage.

### Extensibility

If you want to add "embedding vectors" to the dictionary in the future, simply create `embedding.bin` and link via ID—same pattern applies.

---

This architecture fully satisfies the seemingly contradictory requirements of **"rich semantic information"** and **"blazing execution speed"**.
