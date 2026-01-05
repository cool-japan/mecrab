# Word2Vec Training Guide for MeCrab

**Date:** 2026-01-02
**Purpose:** Train production-quality word embeddings aligned with IPADIC word_id indices
**Target:** 別マシンで学習する完全なワークフロー
**Status:** ✅ 完全実装済み（Pure Rust実装追加！）

---

## Overview

This guide shows how to train Word2Vec embeddings from scratch using Japanese Wikipedia, aligned with IPADIC dictionary word_id indices for use with MeCrab's vector features.

**🆕 New in this version:**
- **Pure Rust Word2Vec implementation** (`mecrab-word2vec`) - No Python/C dependencies!
- Native `kizame vectors train` command for direct training
- Native `kizame vectors convert` command for format conversion
- Optional MCV1 direct output (skip conversion step)

---

## Training Method Comparison

| Feature | mecrab-word2vec (Rust) | word2vec-c (C) | gensim (Python) |
|---------|------------------------|----------------|-----------------|
| Speed | ⚡ Fast | ⚡⚡ Fastest | 🐌 Slow |
| Memory | 🟢 Efficient | 🟢 Efficient | 🔴 High |
| Dependencies | ✅ None (Pure Rust) | C compiler | Python + NumPy |
| MCV1 output | ✅ Direct | ❌ Need conversion | ❌ Need conversion |
| Integration | ✅ Built-in KizaMe | ❌ External | ❌ External |
| **Recommended** | **✅ Yes** | For very large corpora | Legacy |

---

## Prerequisites

### Required Software

- **MeCrab/KizaMe** installed (for morphological analysis)
- **IPADIC Dictionary** (see installation below)

**Optional (for alternative methods):**
- **Python 3.8+** with gensim and numpy (Python method)
- **C compiler** (word2vec-c method)
- **GNU Parallel** (for faster corpus parsing)

### System Requirements

- **Storage:** ~15GB for intermediate files
- **RAM:** 4-8GB
- **CPU:** Multi-core recommended (for parallel processing)

### Install IPADIC Dictionary

The IPADIC dictionary is required for morphological analysis. Install it using your system's package manager:

```bash
# Ubuntu/Debian
sudo apt install mecab-ipadic-utf8

# Fedora/RHEL
sudo dnf install mecab-ipadic

# Arch Linux
sudo pacman -S mecab-ipadic

# macOS (Homebrew)
brew install mecab-ipadic
```

**Verify installation:**

```bash
# Check if dictionary exists
ls -la /var/lib/mecab/dic/ipadic-utf8/sys.dic

# Or use kizame to check
kizame dict init
# This will show you where IPADIC is installed
```

**Common installation paths:**
- Ubuntu/Debian: `/var/lib/mecab/dic/ipadic-utf8`
- Fedora/RHEL: `/usr/lib/mecab/dic/ipadic-utf8`
- macOS: `/usr/local/lib/mecab/dic/ipadic-utf8`

**Manual installation (if package not available):**

```bash
# Download IPADIC source
wget https://github.com/taku910/mecab/releases/download/mecab-0.996/mecab-ipadic-2.7.0-20070801.tar.gz
tar xzf mecab-ipadic-2.7.0-20070801.tar.gz
cd mecab-ipadic-2.7.0-20070801

# Compile with UTF-8 charset
./configure --with-charset=utf8
make
sudo make install
```

---

## Quick Start (Rust - Recommended!)

**Pure Rust implementation - No external dependencies!**

```bash
# 1. Extract vocabulary (find max word_id)
kizame dict dump -d /var/lib/mecab/dic/ipadic-utf8 --vocab > ipadic_vocab.txt
MAX_WORD_ID=$(tail -1 ipadic_vocab.txt | cut -f1)

# 2. Parse corpus to word_id sequences
cat jawiki_text.txt | kizame parse --wakati-word-id > corpus_word_ids.txt

# 3. Train Word2Vec (Rust) - Direct MCV1 output!
kizame vectors train \
  -i corpus_word_ids.txt \
  -o vectors.bin \
  -f mcv1 \
  --max-word-id $MAX_WORD_ID \
  --size 100 \
  --window 5 \
  --negative 5 \
  --epochs 3 \
  --threads 8

# Done! Ready to use
kizame parse --with-vector -v vectors.bin --with-ipa < input.txt | cat
```

---

## Quick Start (Python/C - Legacy)

**Note:** Replace `/var/lib/mecab/dic/ipadic-utf8` with your actual IPADIC path. Use `kizame dict init` to find it.

```bash
# 1. Extract vocabulary
kizame dict dump -d /var/lib/mecab/dic/ipadic-utf8 --vocab > ipadic_vocab.txt

# 2. Parse corpus to word_id sequences
cat jawiki_text.txt | kizame parse --wakati-word-id > corpus_word_ids.txt

# 3. Train with Python/C (see Step 4 below)
python3 train_word2vec.py  # or word2vec-c

# 4. Convert to MCV1
kizame vectors convert \
  -i word2vec.txt \
  -o vectors.bin \
  -f word2vec-text \
  -v ipadic_vocab.txt
```

---

## Step 1: Extract IPADIC Vocabulary (1 minute)

Extract the complete IPADIC vocabulary with word_id mapping:

```bash
# Generate vocabulary list (word_id → feature mapping)
kizame dict dump -d /var/lib/mecab/dic/ipadic-utf8 --vocab > ipadic_vocab.txt

# Check output
head ipadic_vocab.txt
# 0	名詞,一般,*,*,*,*,Tシャツ,ティーシャツ,ティーシャツ
# 1	記号,一般,*,*,*,*,£,ポンド,ポンド
# ...

# Total tokens
wc -l ipadic_vocab.txt
# 392127 ipadic_vocab.txt

# Get max word_id (needed for MCV1 format)
MAX_WORD_ID=$(tail -1 ipadic_vocab.txt | cut -f1)
echo "Max word_id: $MAX_WORD_ID"
# Max word_id: 392126
```

**Output:**
- `ipadic_vocab.txt` - Complete vocabulary (word_id → feature)
- `MAX_WORD_ID` - Maximum word_id for MCV1 format

---

## Step 2: Download Wikipedia Corpus (10-15 minutes)

Download and extract Japanese Wikipedia dump:

```bash
# Download latest dump (~3.5GB compressed)
wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2

# Install WikiExtractor
pip3 install wikiextractor

# Extract plain text (~4GB)
python3 -m wikiextractor.WikiExtractor \
  -o wiki_text \
  --json \
  --no-templates \
  jawiki-latest-pages-articles.xml.bz2

# Extract text field from JSON
find wiki_text -name "wiki_*" -exec cat {} \; | \
  jq -r '.text' | \
  grep -v '^$' > jawiki_text.txt

# Check size
wc -l jawiki_text.txt
# ~2-3M lines

ls -lh jawiki_text.txt
# ~4GB
```

**Output:**
- `jawiki_text.txt` - Plain text corpus

---

## Step 3: Convert Text to word_id Sequences (5 minutes - 1 hour)

### Option A: Single-threaded (Simple, ~1 hour)

```bash
# Parse entire corpus into word_id sequences
kizame parse --wakati-word-id < jawiki_text.txt > corpus_word_ids.txt

# Monitor progress (in another terminal)
watch -n 1 'ls -lh corpus_word_ids.txt'
```

### Option B: Parallel Processing (Recommended, ~5 minutes)

```bash
# Split corpus into chunks
split -l 100000 jawiki_text.txt wiki_chunk_

# Process in parallel (8 cores)
ls wiki_chunk_* | \
  parallel -j8 --progress \
    'kizame parse --wakati-word-id < {} > {}.ids'

# Concatenate results
cat wiki_chunk_*.ids > corpus_word_ids.txt

# Cleanup
rm wiki_chunk_*
```

**Output:**
- `corpus_word_ids.txt` - word_id sequences (space-separated)

**Example:**
```
305004 57066 250027 53044 337761
184614 47497 101 250027 53044
...
```

---

## Step 4: Train Word2Vec

Choose one of three methods:

### Method A: mecrab-word2vec (Rust) - Recommended! ⭐

**Pure Rust, no external dependencies, integrated into KizaMe**

```bash
# Train with direct MCV1 output (skip conversion!)
kizame vectors train \
  -i corpus_word_ids.txt \
  -o vectors.bin \
  -f mcv1 \
  --max-word-id $MAX_WORD_ID \
  --size 100 \
  --window 5 \
  --negative 5 \
  --min-count 10 \
  --sample 0.0001 \
  --alpha 0.025 \
  --min-alpha 0.0001 \
  --epochs 3 \
  --threads 8

# Or output word2vec text format
kizame vectors train \
  -i corpus_word_ids.txt \
  -o vectors.txt \
  -f text \
  --size 100 \
  --window 5 \
  --negative 5 \
  --epochs 3 \
  --threads 8
```

**Monitoring progress:**
```
Training Word2Vec model...
  Input: "corpus_word_ids.txt"
  Output: "vectors.bin"
  Vector size: 100
  Window: 5
  Negative samples: 5
  Min count: 10
  Sample: 0.0001
  Alpha: 0.025 → 0.0001
  Epochs: 3
  Threads: 8

Vocabulary built:
  Total words: 1046740582
  Unique words (before filtering): 320145
  Vocab size (after min_count=10): 163923
Negative sampling table built: 100000000 entries

Starting training using file "corpus_word_ids.txt"
Vocab size: 163923
Words in train file: 1046740582

Epoch 1/3
  Starting alpha: 0.025000
  Processing 2401245 sentences...
Alpha: 0.023958  Progress: 4.17%
...
```

**Advantages:**
- ✅ No external dependencies (Pure Rust)
- ✅ Integrated into KizaMe workflow
- ✅ Direct MCV1 output (skip conversion)
- ✅ Good performance with Rayon multi-threading
- ✅ Memory efficient

**Output:**
- `vectors.bin` - MCV1 binary format (ready to use!)
- OR `vectors.txt` - word2vec text format (if `-f text`)

---

### Method B: word2vec-c (C) - Fastest for very large corpora

**C implementation - Fastest but requires compilation**

Install:
```bash
git clone https://github.com/dav/word2vec.git
cd word2vec
make
```

Train:
```bash
./word2vec/bin/word2vec \
  -train corpus_word_ids.txt \
  -output vectors.txt \
  -size 100 \
  -window 5 \
  -sample 1e-4 \
  -negative 5 \
  -hs 0 \
  -binary 0 \
  -cbow 0 \
  -iter 3 \
  -min-count 10 \
  -threads 8
```

**Monitoring (in another terminal):**
```bash
# CPU usage
top -p $(pgrep word2vec)

# Output file size
watch -n 2 'ls -lh vectors.txt'
```

Convert to MCV1:
```bash
kizame vectors convert \
  -i vectors.txt \
  -o vectors.bin \
  -f word2vec-text \
  -v ipadic_vocab.txt
```

**Output:**
- `vectors.txt` - word2vec text format
- `vectors.bin` - MCV1 binary format (after conversion)

---

### Method C: gensim (Python) - Legacy

**Python implementation - Slowest, high memory usage**

Install gensim:
```bash
pip3 install gensim numpy
```

Create training script:

```bash
cat > train_word2vec.py << 'PYEOF'
#!/usr/bin/env python3
import gensim
from gensim.models import Word2Vec

print("Loading corpus...")
# Read word_id sequences
sentences = []
with open('corpus_word_ids.txt', 'r') as f:
    for line in f:
        if line.strip():
            # Split space-separated word_ids
            sentences.append(line.strip().split())

print(f"Loaded {len(sentences)} sentences")

print("Training Word2Vec...")
# Train Word2Vec model
model = Word2Vec(
    sentences,
    vector_size=300,    # Embedding dimension
    window=5,           # Context window
    min_count=5,        # Ignore words with freq < 5
    workers=8,          # Parallel workers
    sg=1,               # Skip-gram (1) or CBOW (0)
    epochs=5,           # Training iterations
    negative=5,         # Negative sampling
)

print("Saving model...")
model.save("word2vec_ipadic.model")

# Save in word2vec text format for kizame conversion
print("Saving word2vec text format...")
with open('word2vec.txt', 'w') as f:
    vocab_size = len(model.wv)
    dim = model.vector_size
    f.write(f"{vocab_size} {dim}\n")

    for word_id_str in model.wv.key_to_index.keys():
        vector = model.wv[word_id_str]
        vector_str = ' '.join(str(v) for v in vector)
        f.write(f"{word_id_str} {vector_str}\n")

print(f"Saved: word2vec.txt ({vocab_size} words × {dim} dims)")
print("")
print("Next step: Convert to MCV1 format")
print("  kizame vectors convert \\")
print("    -i word2vec.txt \\")
print("    -o vectors.bin \\")
print("    -f word2vec-text \\")
print("    -v ipadic_vocab.txt")
PYEOF

chmod +x train_word2vec.py
```

Run training:

```bash
python3 train_word2vec.py
# Training Word2Vec...
# (30min - 2hr depending on CPU)
# Saved: word2vec.txt
```

Convert to MCV1:
```bash
kizame vectors convert \
  -i word2vec.txt \
  -o vectors.bin \
  -f word2vec-text \
  -v ipadic_vocab.txt
```

**Output:**
- `word2vec.txt` - word2vec text format
- `word2vec_ipadic.model` - Gensim model (optional, for further training)
- `vectors.bin` - MCV1 binary format (after conversion)

---

## Step 5: Verify Vectors

**Important:** When using `--with-ipa` with default text output, you must pipe through `cat` for proper terminal display due to Unicode IPA character handling. This is not required for JSON-LD output.

Check header:

```bash
kizame vectors info -v vectors.bin
# Vector Pool Information: "vectors.bin"
# ================================================================================
#
# Vocab size:  392127
# Dimensions:  100
# File size:   156850848 bytes
```

Test with actual text:

```bash
echo "東京に行く" | kizame parse --with-vector -v vectors.bin --with-ipa | cat
# 東京	名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トーキョー
#   IPA: /toːkʲoː/
#   Vector: [0.123, -0.456, 0.789, ..., 0.234] (dim=100)
# に	助詞,格助詞,一般,*,*,*,に,ニ,ニ
#   IPA: /ɲi/
#   Vector: [-0.345, 0.678, -0.123, ..., 0.456] (dim=100)
# 行く	動詞,自立,*,*,五段・カ行促音便,基本形,行く,イク,イク
#   IPA: /ikɯ/
#   Vector: [0.567, -0.234, 0.901, ..., -0.123] (dim=100)
# EOS
```

---

## Complete Workflow Summary

### Recommended: Pure Rust Workflow

```bash
# === On Training Machine ===

# 0. Verify IPADIC installation
kizame dict init  # Shows where IPADIC is installed

# 1. Extract vocabulary (1 min)
kizame dict dump -d /var/lib/mecab/dic/ipadic-utf8 --vocab > ipadic_vocab.txt
MAX_WORD_ID=$(tail -1 ipadic_vocab.txt | cut -f1)

# 2. Download Wikipedia (10 min)
wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2
wikiextractor -o wiki_text --json jawiki-latest-pages-articles.xml.bz2
find wiki_text -name "wiki_*" -exec cat {} \; | jq -r '.text' > jawiki_text.txt

# 3. Parse to word_id sequences (5 min with parallel)
split -l 100000 jawiki_text.txt wiki_chunk_
ls wiki_chunk_* | parallel -j8 'kizame parse --wakati-word-id < {} > {}.ids'
cat wiki_chunk_*.ids > corpus_word_ids.txt

# 4. Train Word2Vec (Rust) with direct MCV1 output (30 min - 1 hour)
kizame vectors train \
  -i corpus_word_ids.txt \
  -o vectors.bin \
  -f mcv1 \
  --max-word-id $MAX_WORD_ID \
  --size 100 \
  --window 5 \
  --negative 5 \
  --epochs 3 \
  --threads 8

# 5. Verify
kizame vectors info -v vectors.bin

# === Deploy to Production ===

# Compress and transfer
gzip vectors.bin
scp vectors.bin.gz production:/opt/mecrab/data/

# Use
kizame parse --with-vector -v /opt/mecrab/data/vectors.bin < input.txt | cat
```

### Legacy: Python/C Workflow (with conversion step)

```bash
# ... Steps 0-3 same as above ...

# 4. Train Word2Vec (30 min - 2 hours)
# Option A: word2vec-c
./word2vec/bin/word2vec -train corpus_word_ids.txt -output vectors.txt ...

# Option B: Python
python3 train_word2vec.py

# 5. Convert to MCV1 (1 min)
kizame vectors convert \
  -i vectors.txt \
  -o vectors.bin \
  -f word2vec-text \
  -v ipadic_vocab.txt

# 6. Verify
kizame vectors info -v vectors.bin
```

---

## Resource Requirements

| Step | Time (Rust) | Time (C) | Time (Python) | Storage |
|------|-------------|----------|---------------|---------|
| 1. Vocab extraction | 1 min | 1 min | 1 min | 50 MB |
| 2. Wikipedia download | 10 min | 10 min | 10 min | 3.5 GB |
| 3. Text extraction | 20 min | 20 min | 20 min | 4 GB |
| 4. Morphological analysis | 5 min (parallel) | 5 min | 5 min | 2 GB |
| 5. Word2Vec training | 30-60 min | 15-30 min | 1-2 hours | 150 MB |
| 6. Conversion | N/A (direct) | 1 min | 1 min | - |
| **Total** | **~1 hour** | **~50 min** | **~2-3 hours** | **~10 GB** |

---

## Alternative: Smaller Test Corpus

For quick testing with a smaller corpus:

```bash
# Use only first 10K Wikipedia articles
head -n 10000 jawiki_text.txt | \
  kizame parse --wakati-word-id > small_corpus.txt

# Train with Rust (direct MCV1)
kizame vectors train \
  -i small_corpus.txt \
  -o vectors_small.bin \
  -f mcv1 \
  --max-word-id $MAX_WORD_ID \
  --size 50 \
  --window 5 \
  --min-count 2 \
  --epochs 5 \
  --threads 4

# Test
echo "東京に行く" | kizame parse --with-vector -v vectors_small.bin --with-ipa | cat
```

---

## Production Deployment

Copy to production server:

```bash
# Compress for transfer
gzip vectors.bin
# vectors.bin.gz (~50-100MB with 100-dim)

# Transfer
scp vectors.bin.gz production:/opt/mecrab/data/

# On production
gunzip /opt/mecrab/data/vectors.bin.gz

# Use in production
kizame parse --with-vector -v /opt/mecrab/data/vectors.bin < input.txt | cat > output.txt
```

---

## Troubleshooting

### IPADIC Dictionary Not Found

**Problem:** `kizame dict dump` fails with "dictionary not found" or similar error

**Solution:**

1. Check if IPADIC is installed:
```bash
kizame dict init
# This will show where IPADIC should be
```

2. Install IPADIC:
```bash
# Ubuntu/Debian
sudo apt install mecab-ipadic-utf8

# Or manually download and compile
wget https://github.com/taku910/mecab/releases/download/mecab-0.996/mecab-ipadic-2.7.0-20070801.tar.gz
tar xzf mecab-ipadic-2.7.0-20070801.tar.gz
cd mecab-ipadic-2.7.0-20070801
./configure --with-charset=utf8
make
sudo make install
```

3. Verify installation:
```bash
ls -la /var/lib/mecab/dic/ipadic-utf8/sys.dic
# Or check other common paths:
# /usr/lib/mecab/dic/ipadic-utf8
# /usr/local/lib/mecab/dic/ipadic-utf8
```

4. Use correct path in commands:
```bash
# Find your IPADIC path
find /usr -name "sys.dic" 2>/dev/null | grep ipadic

# Then use that path
kizame dict dump -d /path/to/ipadic-utf8 --vocab > vocab.txt
```

### Memory Issues

**Problem:** Training fails with OOM (Out of Memory)

**Solution for Rust:**
```bash
# Reduce vector size
kizame vectors train \
  -i corpus.txt \
  -o vectors.bin \
  --size 50 \
  --threads 4
```

**Solution for Python:**
```python
# Reduce vector size or use incremental training
model = Word2Vec(
    sentences,
    vector_size=200,      # Reduce from 300
    workers=4,            # Reduce workers
    max_vocab_size=300000 # Limit vocabulary
)
```

### Conversion Issues

**Problem:** "vocab file required for word2vec-text format"

**Solution:** Make sure to provide the vocabulary file:
```bash
kizame vectors convert \
  -i word2vec.txt \
  -o vectors.bin \
  -v ipadic_vocab.txt  # Required!
```

**Problem:** Low mapping rate (many unmapped words)

**Solution:** This is normal. Word2vec only learns vectors for words that appear in the corpus with `min_count` frequency. Unmapped words will have zero vectors.

### Coverage Issues

Check vocabulary coverage:

```bash
# Count unique word_ids in corpus
cat corpus_word_ids.txt | tr ' ' '\n' | sort -u | wc -l
# Should be close to vocab_size (392K for IPADIC)
```

Check mapping statistics:

```bash
# The conversion command shows:
# Mapping summary:
#   Mapped:   120543 words  (appeared in training corpus)
#   Unmapped: 4889 words    (didn't appear or below min_count)
```

---

## Command Reference

### Dictionary Operations

```bash
# Dump vocabulary with word_ids
kizame dict dump -d <dict_dir> --vocab > vocab.txt

# Show dictionary info
kizame dict info -d <dict_dir>
```

### Parsing Operations

```bash
# Parse to word_id sequences (for training)
echo "テキスト" | kizame parse --wakati-word-id

# Parse with vectors (for inference)
echo "テキスト" | kizame parse --with-vector -v vectors.bin

# Parse with IPA and vectors
echo "テキスト" | kizame parse --with-vector -v vectors.bin --with-ipa | cat
```

### Vector Operations

```bash
# Train Word2Vec (Rust)
kizame vectors train \
  -i corpus.txt \
  -o vectors.bin \
  -f mcv1 \
  --max-word-id 392126 \
  --size 100 \
  --window 5 \
  --negative 5 \
  --epochs 3 \
  --threads 8

# Show vector file info
kizame vectors info -v vectors.bin

# Convert word2vec text to MCV1
kizame vectors convert \
  -i word2vec.txt \
  -o vectors.bin \
  -f word2vec-text \
  -v vocab.txt
```

---

## Next Steps

- **Semantic similarity search** - Find similar words using embeddings
- **Document embeddings** - Average word vectors for document comparison
- **Fine-tuning** - Continue training on domain-specific corpus
- **Contextualized embeddings** - Integrate BERT/RoBERTa for dynamic embeddings

---

## File Format Specifications

### word2vec Text Format (Input/Output)

```
<vocab_size> <dim>
<word_id> <v1> <v2> ... <vN>
<word_id> <v1> <v2> ... <vN>
...
```

Example:
```
3 4
305004 0.1 0.2 0.3 0.4
184614 -0.1 -0.2 -0.3 -0.4
57066 0.5 0.4 0.3 0.2
```

### MCV1 Binary Format (Output)

```
Header (32 bytes):
  [0-3]   Magic: 0x3143564D ("MCV1")
  [4-7]   vocab_size: u32
  [8-11]  dim: u32
  [12-15] dtype: u32 (0=F32, 1=F16, 2=I8)
  [16-31] reserved: [0; 16]

Data (vocab_size * dim * sizeof(dtype)):
  Vector data in row-major order (C-style)
  vector[word_id][dimension]
```

---

**Generated by:** Claude Sonnet 4.5
**License:** MIT OR Apache-2.0
**Project:** MeCrab/KizaMe
**Last Updated:** 2026-01-02 (Added mecrab-word2vec Pure Rust implementation!)
