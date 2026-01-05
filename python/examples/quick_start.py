#!/usr/bin/env python3
"""
MeCrab Python API - Quick Start Examples

Copyright 2026 COOLJAPAN OU (Team KitaSan)
License: MIT OR Apache-2.0
"""

import mecrab

def main():
    print("=" * 70)
    print("MeCrab Python API - Quick Start")
    print("=" * 70)
    print()

    # 1. Basic parsing
    print("1. Basic Parsing")
    print("-" * 70)
    m = mecrab.MeCrab()
    text = "すもももももももものうち"
    result = m.parse(text)
    print(f"Input: {text}")
    print(f"Output:\n{result}")
    print()

    # 2. Wakati (space-separated)
    print("2. Wakati Output")
    print("-" * 70)
    text = "私は学生です"
    words = m.wakati(text)
    print(f"Input: {text}")
    print(f"Wakati: {words}")
    print(f"Tokens: {words.split()}")
    print()

    # 3. Structured output (Pythonic API)
    print("3. Structured Output (Pythonic API)")
    print("-" * 70)
    text = "東京に行く"
    morphemes = m.parse_to_list(text)
    print(f"Input: {text}\n")
    for surface, feature in morphemes:
        parts = feature.split(',')
        pos = parts[0]
        reading = parts[7] if len(parts) > 7 else ""
        print(f"  {surface:8s} {pos:6s} {reading}")
    print()

    # 4. IPA Pronunciation
    print("4. IPA Pronunciation (One-Shot API)")
    print("-" * 70)
    m_ipa = mecrab.MeCrab(with_ipa=True)

    phrases = [
        "東京に行く",
        "こんにちは",
        "ありがとうございます"
    ]

    for phrase in phrases:
        ipa = m_ipa.to_ipa_text(phrase)
        print(f"{phrase:20s} → /{ipa}/")
    print()

    # 5. Pythonic Dictionary API
    print("5. Pythonic Dictionary API (with IPA)")
    print("-" * 70)
    text = "東京に行く"
    morphemes = m_ipa.parse_to_dict(text)
    print(f"Input: {text}\n")
    for morph in morphemes:
        print(f"Surface: {morph['surface']}")
        print(f"  POS: {morph['pos']}")
        print(f"  Reading: {morph.get('reading', 'N/A')}")
        print(f"  IPA: /{morph.get('ipa', 'N/A')}/")
        print()

    # 6. Word Embeddings (if available)
    print("6. Word Embeddings (Cosine Similarity)")
    print("-" * 70)
    try:
        m_vec = mecrab.MeCrab(vector_path="vectors.bin")

        word_pairs = [
            ("東京", "京都"),
            ("東京", "大阪"),
            ("猫", "犬"),
            ("東京", "食べる"),
        ]

        for word1, word2 in word_pairs:
            try:
                sim = m_vec.similarity(word1, word2)
                print(f"{word1:8s} ⟷ {word2:8s} : {sim:6.3f}")
            except RuntimeError as e:
                print(f"{word1:8s} ⟷ {word2:8s} : {e}")
        print()
    except RuntimeError:
        print("Vector file 'vectors.bin' not found.")
        print("See README.md for Word2Vec training instructions.")
        print()

    # 7. Batch Processing
    print("7. Batch Processing (Parallel)")
    print("-" * 70)
    texts = [
        "今日は良い天気です",
        "明日は雨が降るでしょう",
        "週末は買い物に行きます"
    ]

    wakati_results = m.wakati_batch(texts)
    for i, words in enumerate(wakati_results, 1):
        print(f"Text {i}: {words}")
    print()

    # 8. Custom Words (Overlay Dictionary)
    print("8. Custom Words (Overlay Dictionary)")
    print("-" * 70)
    text = "ChatGPTは便利です"
    print(f"Before adding custom word:")
    print(m.wakati(text))

    m.add_word("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000)

    print(f"\nAfter adding custom word:")
    print(m.wakati(text))
    print(f"Overlay dictionary size: {m.overlay_size()} words")
    print()

    print("=" * 70)
    print("For more examples, see:")
    print("  - examples/01_basic_usage.ipynb")
    print("  - examples/03_advanced_features.ipynb")
    print("  - python/README.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
