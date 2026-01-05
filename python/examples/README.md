# MeCrab Python Examples

Interactive tutorials and sample code for MeCrab Python bindings.

## Files

### Quick Start Script

**`quick_start.py`** - Standalone Python script demonstrating all major features:

```bash
python quick_start.py
```

Features demonstrated:
- Basic parsing
- Wakati (space-separated) output
- Structured output (lists and dicts)
- IPA pronunciation (one-shot API)
- Word embeddings (cosine similarity)
- Batch processing
- Custom word dictionary

### Jupyter Notebooks

**`01_basic_usage.ipynb`** - Core functionality and basic usage:

- Import and version check
- Basic parsing and wakati output
- Structured output (lists)
- Batch processing (parallel)
- Custom word dictionary (overlay)
- Word frequency counter
- POS extraction
- Performance benchmarking

**`03_advanced_features.ipynb`** - Advanced features (IPA, vectors):

- **IPA Pronunciation** (International Phonetic Alphabet)
  - One-shot conversion: `to_ipa()`, `to_ipa_text()`
  - Phonetic examples for common phrases
  - Phonetic search engine example

- **Pythonic Dictionary API**
  - `parse_to_dict()` with all morpheme information
  - Extract specific fields (nouns, readings, IPA)

- **Word Embeddings**
  - Cosine similarity: `similarity(word1, word2)`
  - Semantic clustering examples
  - Combined IPA + embeddings

- **Practical Applications**
  - Phonetic search
  - Semantic similarity
  - Word clustering

## Running Notebooks

### Setup

```bash
# Install Jupyter
pip install jupyter notebook

# Install MeCrab Python bindings
cd /path/to/mecrab
maturin develop --features python,parallel
```

### Launch Jupyter

```bash
cd python/examples
jupyter notebook
```

Then open:
- `01_basic_usage.ipynb` - Start here for basics
- `03_advanced_features.ipynb` - Advanced features (IPA, vectors)

## Prerequisites

### Required

1. **MeCrab Python package** (installed via maturin)
2. **IPADIC dictionary**:
   ```bash
   # Ubuntu/Debian
   sudo apt install mecab-ipadic-utf8

   # macOS
   brew install mecab-ipadic
   ```

### Optional (for advanced features)

3. **Word vectors** (for embedding/similarity features):

   **Option A: Download pre-trained vectors (Recommended)**

   ```bash
   # Install huggingface_hub
   pip install huggingface-hub

   # Download Japanese Wikipedia vectors (~65MB)
   huggingface-cli download KitaSan/mecrab-jawiki-word2vec vectors.bin --local-dir .
   ```

   **Dataset:** https://huggingface.co/datasets/KitaSan/mecrab-jawiki-word2vec

   **Details:**
   - Source: Japanese Wikipedia (全文)
   - Vocabulary: 163,922 words (IPADIC)
   - Vector Size: 100 dimensions
   - Training: Hogwild! algorithm (83% parallel efficiency on 6 cores)
   - Format: MCV1 (memory-mapped, instant loading)

   **Option B: Train custom vectors**

   For domain-specific corpora:
   ```bash
   # Extract vocabulary
   kizame dict dump -d /var/lib/mecab/dic/ipadic-utf8 --vocab > vocab.txt
   MAX_WORD_ID=$(tail -1 vocab.txt | cut -f1)

   # Parse corpus
   cat corpus.txt | kizame parse --wakati-word-id > corpus_ids.txt

   # Train Word2Vec (MCV1 binary format - memory-mapped, instant loading)
   kizame vectors train \
     -i corpus_ids.txt \
     -o vectors.bin \
     -f mcv1 \
     --max-word-id $MAX_WORD_ID \
     --size 100 \
     --window 5 \
     --negative 5 \
     --epochs 3 \
     --threads 6
   ```

   See `../../kizame/WORD2VEC_TRAINING_GUIDE.md` for complete training pipeline.

## Key Features Demonstrated

### 1. IPA Pronunciation (One-Shot API)

```python
import mecrab

m = mecrab.MeCrab(with_ipa=True)

# One-shot conversion
ipa = m.to_ipa_text("東京に行く")
print(ipa)  # => "toːkʲoː ɲi ikɯ"
```

This is the **key feature** - instant phonetic conversion without complex parsing!

### 2. Pythonic Dictionary API

```python
morphemes = m.parse_to_dict("東京に行く")

for morph in morphemes:
    print(morph['surface'], morph['pos'], morph.get('ipa'))
# =>
# 東京 名詞 toːkʲoː
# に 助詞 ɲi
# 行く 動詞 ikɯ
```

### 3. Word Embeddings (Cosine Similarity)

```python
m = mecrab.MeCrab(vector_path="vectors.bin")

sim = m.similarity("東京", "京都")
print(f"Similarity: {sim:.3f}")  # => 0.856 (high similarity)
```

### 4. Combined IPA + Embeddings

```python
m = mecrab.MeCrab(with_ipa=True, vector_path="vectors.bin")

morphemes = m.parse_to_dict("東京に行く")
for morph in morphemes:
    print(f"{morph['surface']}: IPA=/{morph.get('ipa')}/, Embedding={len(morph.get('embedding', []))}-dim")
```

## Use Cases

- **Language Learning**: Phonetic pronunciation guides
- **Speech Synthesis**: IPA → speech engines
- **Semantic Search**: Find similar documents/words
- **Text Classification**: Use embeddings as features
- **Chatbots/NLP**: Understand intent via similarity
- **Text Analysis**: Word frequency, POS distribution

## Documentation

- **Parent README**: [../README.md](../README.md) - Python API reference
- **Main README**: [../../README.md](../../README.md) - Project overview
- **CLI Guide**: [../../kizame/README.md](../../kizame/README.md) - KizaMe CLI tool
- **Word2Vec Training**: [../../kizame/WORD2VEC_TRAINING_GUIDE.md](../../kizame/WORD2VEC_TRAINING_GUIDE.md)

## Troubleshooting

### Import Error

```
ModuleNotFoundError: No module named 'mecrab'
```

**Solution**: Rebuild with maturin:
```bash
cd /path/to/mecrab
maturin develop --features python,parallel
```

### Dictionary Not Found

```
RuntimeError: Failed to load dictionary
```

**Solution**: Install IPADIC or specify path:
```bash
# Install
sudo apt install mecab-ipadic-utf8

# Or specify path explicitly
m = mecrab.MeCrab(dicdir="/var/lib/mecab/dic/ipadic-utf8")
```

### IPA Not Working

```
RuntimeError: IPA support not enabled
```

**Solution**: Initialize with `with_ipa=True`:
```python
m = mecrab.MeCrab(with_ipa=True)
```

### Vectors Not Working

```
RuntimeError: Vector support not enabled
```

**Solution**: Provide `vector_path`:
```python
m = mecrab.MeCrab(vector_path="vectors.bin")
```

## Contributing

Found an issue or want to add examples? Please submit a PR or issue on GitHub!

---

**Copyright 2026 COOLJAPAN OU (Team KitaSan)**
**License:** MIT OR Apache-2.0
