# mecrab-word2vec

High-performance Word2Vec training library for Japanese text, optimized for multi-core CPUs.

## Features

- ✅ **Skip-gram with Negative Sampling** - Industry-standard algorithm
- ✅ **Hogwild! Parallelization** - Lock-free multi-threading (83% efficiency on 6 cores)
- ✅ **MCV1 Binary Format** - Memory-mapped vector storage for instant loading
- ✅ **Pure Rust** - Memory-safe, no external C dependencies
- ✅ **Zero-copy** - Direct pointer arithmetic, minimal allocations

## Quick Start

### Training Vectors

```rust
use mecrab_word2vec::{Word2VecBuilder, TrainingConfig};

let model = Word2VecBuilder::new()
    .vector_size(100)
    .window_size(5)
    .negative_samples(5)
    .min_count(10)
    .epochs(3)
    .threads(6)
    .build_from_corpus("corpus_word_ids.txt")?;

model.save_mcv1("vectors.bin", 100000)?;
```

### Using Trained Vectors

```rust
use mecrab::vectors::VectorStorage;

let vectors = VectorStorage::load_mcv1("vectors.bin")?;
let vec1 = vectors.get(42)?;  // Get vector for word_id 42
let vec2 = vectors.get(123)?;

let similarity = vectors.cosine_similarity(42, 123)?;
```

## Performance

Tested on Japanese Wikipedia corpus (1B+ words):

| Metric | Value |
|--------|-------|
| **Training Speed** | ~500K words/sec/core |
| **CPU Efficiency** | 83% (6 cores) |
| **Memory Usage** | ~2GB for 160K vocab |
| **Vector Lookup** | O(1) with memory-mapping |

## Input Format

### Corpus File (Word IDs)

```
42 123 456 789
111 222 333
...
```

Each line is a sentence. Each number is a word_id (from MeCrab's vocabulary).

## Output Formats

### MCV1 Binary (Recommended)

```rust
model.save_mcv1("vectors.bin", max_word_id)?;
```

**Benefits:**
- Memory-mapped (instant loading, no RAM copy)
- Direct indexing by word_id
- Compatible with MeCrab's semantic features

### Word2Vec Text Format

```rust
model.save_text("vectors.txt")?;
```

**Format:**
```
163922 100
word1 0.123 -0.456 0.789 ...
word2 -0.234 0.567 -0.890 ...
```

## Architecture

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for technical details on:
- Hogwild! lock-free parallelization
- Direct pointer arithmetic optimization
- Safety guarantees and race condition analysis

## Configuration

```rust
TrainingConfig {
    vector_size: 100,           // Embedding dimension
    window_size: 5,             // Context window
    negative_samples: 5,        // Negative samples per positive
    min_count: 10,              // Minimum word frequency
    sample: 1e-4,               // Subsampling threshold
    alpha: 0.025,               // Initial learning rate
    min_alpha: 0.0001,          // Final learning rate
    epochs: 3,                  // Training epochs
    threads: 6,                 // Parallel threads
}
```

## License

See project LICENSE file.

## References

- Mikolov et al. (2013) - "Distributed Representations of Words and Phrases and their Compositionality"
- Recht et al. (2011) - "Hogwild!: A Lock-Free Approach to Parallelizing SGD"
