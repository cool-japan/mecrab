"""
MeCrab: High-performance Japanese morphological analyzer.

Copyright 2026 COOLJAPAN OU (Team KitaSan)

Pure Rust implementation with Python bindings via PyO3.

Usage
-----

Basic parsing::

    from mecrab import MeCrab

    m = MeCrab("/path/to/ipadic")
    result = m.parse("東京は日本の首都です")

    # Iterate over morphemes
    for morpheme in result:
        print(f"{morpheme.surface}\\t{morpheme.pos}")

    # Length and index access
    print(len(result))          # number of morphemes
    print(result[0].surface)    # first morpheme
    print(result[-1].surface)   # last morpheme

Convenience helpers::

    words  = result.surfaces()   # => ["東京", "は", ...]
    reads  = result.readings()   # => ["トウキョウ", "ハ", ...]
    tags   = result.pos_tags()   # => ["名詞", "助詞", ...]
    jstr   = result.to_json()    # => JSON string

Wakati (space-separated)::

    words = m.wakati("私は学生です")
    print(words)                  # => "私 は 学生 です"

    word_list = m.wakati_list("私は学生です")
    print(word_list)              # => ["私", "は", "学生", "です"]

Direct JSON output::

    json_str = m.parse_to_json("東京に行く")

Structured output (dictionary)::

    morphemes = m.parse_to_dict("東京に行く")
    for morph in morphemes:
        print(morph['surface'], morph['pos'], morph.get('reading'))

Batch processing::

    texts = ["今日は良い天気", "明日は雨"]
    results = m.parse_batch_py(texts)   # GIL released during processing

Custom words::

    m.add_word("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000)
    result = m.parse("ChatGPTは便利です")

IPA pronunciation (requires with_ipa=True)::

    m_ipa = MeCrab(with_ipa=True)
    ipa = m_ipa.to_ipa_text("東京に行く")
    print(ipa)  # => "toːkʲoː ɲi ikɯ"

Word embeddings (requires vector_path)::

    m_vec = MeCrab(vector_path="vectors.bin")
    sim = m_vec.similarity("東京", "京都")
    print(f"Similarity: {sim:.3f}")  # => 0.856

Features
--------

- Fast: Pure Rust implementation, ~10x faster than Python alternatives
- Thread-safe: Concurrent access with zero-copy where possible
- Batch processing: Parallel processing with GIL release
- Live dictionary updates: Add custom words at runtime
- IPA pronunciation: International Phonetic Alphabet
- Word embeddings: Word2Vec with cosine similarity
- MeCab compatible: Works with IPADIC/UniDic dictionaries

Installation
------------

From source (requires Rust toolchain)::

    pip install maturin
    maturin develop --features python,parallel

From PyPI::

    pip install mecrab

System requirements::

    # Ubuntu/Debian
    sudo apt install mecab-ipadic-utf8

    # macOS
    brew install mecab-ipadic

Documentation
-------------

- GitHub: https://github.com/cool-japan/mecrab
- Python API: See python/README.md
- Jupyter tutorials: See python/examples/
- Rust API: https://docs.rs/mecrab

License
-------

Apache-2.0
"""

# Re-export from native module.
# PyO3 exposes the Rust pyclasses under their Python names.
from mecrab.mecrab import (  # type: ignore[import]
    AnalysisIterator,
    AnalysisResult,
    AnalysisResultIterator,
    MeCrab,
    Morpheme,
    cosine_similarity,
    default_dicdir,
    version,
)

__all__ = [
    "MeCrab",
    "Morpheme",
    "AnalysisResult",
    "AnalysisResultIterator",
    "AnalysisIterator",
    "version",
    "default_dicdir",
    "cosine_similarity",
]
__version__ = "0.2.0"
__author__ = "COOLJAPAN OU (Team KitaSan)"
__license__ = "Apache-2.0"
__copyright__ = "Copyright 2026 COOLJAPAN OU (Team KitaSan)"
