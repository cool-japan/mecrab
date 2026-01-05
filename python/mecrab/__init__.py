"""
MeCrab - High-performance morphological analyzer compatible with MeCab

Copyright 2026 COOLJAPAN OU (Team KitaSan)

Pure Rust implementation with Python bindings via PyO3.

Quick Start
-----------

Basic parsing::

    import mecrab

    # Initialize
    m = mecrab.MeCrab()

    # Parse text
    result = m.parse("すもももももももものうち")
    print(result)

Wakati (space-separated)::

    words = m.wakati("私は学生です")
    print(words)  # => "私 は 学生 です"

Structured output (dictionary)::

    morphemes = m.parse_to_dict("東京に行く")
    for morph in morphemes:
        print(morph['surface'], morph['pos'], morph.get('reading'))
    # =>
    # 東京 名詞 トウキョウ
    # に 助詞 ニ
    # 行く 動詞 イク

Batch processing::

    texts = ["今日は良い天気", "明日は雨"]
    results = m.parse_batch(texts)

Custom words::

    m.add_word("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000)
    result = m.parse("ChatGPTは便利です")

IPA pronunciation (one-shot API)::

    m_ipa = mecrab.MeCrab(with_ipa=True)
    ipa = m_ipa.to_ipa_text("東京に行く")
    print(ipa)  # => "toːkʲoː ɲi ikɯ"

Word embeddings (cosine similarity)::

    m_vec = mecrab.MeCrab(vector_path="vectors.bin")
    sim = m_vec.similarity("東京", "京都")
    print(f"Similarity: {sim:.3f}")  # => 0.856

Combined IPA + Vectors::

    m = mecrab.MeCrab(with_ipa=True, vector_path="vectors.bin")
    morphemes = m.parse_to_dict("東京に行く")
    for morph in morphemes:
        print(morph['surface'], morph.get('ipa'), morph.get('embedding'))

Features
--------

- Fast: Pure Rust implementation, ~10x faster than Python alternatives
- Thread-safe: Concurrent access with zero-copy where possible
- Batch processing: Parallel processing for multiple texts
- Live dictionary updates: Add custom words at runtime
- IPA pronunciation: International Phonetic Alphabet with one-shot API
- Word embeddings: Word2Vec with cosine similarity
- MeCab compatible: Works with IPADIC/UniDic dictionaries

Installation
------------

From source (requires Rust toolchain)::

    pip install maturin
    maturin develop --features python,parallel

From PyPI (coming soon)::

    pip install mecrab

System requirements::

    # Ubuntu/Debian
    sudo apt install mecab-ipadic-utf8

    # macOS
    brew install mecab-ipadic

Documentation
-------------

- GitHub: https://github.com/kitasan/mecrab
- Python API: See python/README.md
- Jupyter tutorials: See python/examples/
- Rust API: https://docs.rs/mecrab

License
-------

Dual-licensed under MIT OR Apache-2.0

"""

# Re-export from native module
# PyO3 exposes PyMeCrab as "MeCrab" and PyMorpheme as "Morpheme" via #[pyclass(name = ...)]
from mecrab.mecrab import MeCrab, version

__all__ = ["MeCrab", "version"]
__version__ = version()
__author__ = "COOLJAPAN OU (Team KitaSan)"
__license__ = "MIT OR Apache-2.0"
__copyright__ = "Copyright 2026 COOLJAPAN OU (Team KitaSan)"
