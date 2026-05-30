//! End-to-end pipeline tests using the synthetic dictionary.
//!
//! These tests exercise the full MeCrab pipeline without requiring any on-disk
//! dictionary installation.  All dictionary data is built in-memory from the
//! curated lexicon in `mecrab_builder::synthetic`.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use mecrab_builder::synthetic::build_synthetic_dictionary;

/// Build a `MeCrab` instance from the synthetic dictionary.
fn make_mecrab() -> mecrab::MeCrab {
    let d = build_synthetic_dictionary();
    mecrab::MeCrab::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
        .expect("synthetic dict load failed")
}

// ── Segmentation tests ────────────────────────────────────────────────────────

/// The classic Japanese morphological analysis test sentence.
/// Correct segmentation: すもも / も / もも / も / もも / の / うち
#[test]
fn segments_sumomo_sentence() {
    let m = make_mecrab();
    let r = m.parse("すもももももももものうち").expect("parse failed");
    let surfaces: Vec<&str> = r
        .morphemes
        .iter()
        .filter(|x| !x.surface.is_empty())
        .map(|x| x.surface.as_str())
        .collect();
    assert_eq!(
        surfaces,
        ["すもも", "も", "もも", "も", "もも", "の", "うち"],
        "segmentation was: {:?}",
        surfaces
    );
}

/// Wakati (space-separated) output for the same sentence.
#[test]
fn wakati_roundtrip() {
    let m = make_mecrab();
    let w = m.wakati("すもももももももものうち").expect("wakati failed");
    assert_eq!(
        w.trim(),
        "すもも も もも も もも の うち",
        "wakati output was: {:?}",
        w
    );
}

// ── Feature passthrough test ───────────────────────────────────────────────────

/// Known words should carry their features from sys.dic through the analysis.
#[test]
fn known_words_and_features() {
    let m = make_mecrab();
    let r = m.parse("東京は日本").expect("parse failed");
    let surfaces: Vec<&str> = r
        .morphemes
        .iter()
        .filter(|x| !x.surface.is_empty())
        .map(|x| x.surface.as_str())
        .collect();
    assert_eq!(
        surfaces,
        ["東京", "は", "日本"],
        "segmentation was: {:?}",
        surfaces
    );
    // Feature from sys.dic must carry through (starts with 名詞,固有名詞)
    let tokyo = r
        .morphemes
        .iter()
        .find(|m| m.surface == "東京")
        .expect("東京 must be present");
    assert!(
        tokyo.feature.starts_with("名詞,固有名詞"),
        "東京 feature should start with 名詞,固有名詞; got: {}",
        tokyo.feature
    );
}

// ── N-best test ───────────────────────────────────────────────────────────────

/// N-best paths must be non-decreasing in cost, and the best path must
/// match the expected segmentation.
#[test]
fn nbest_returns_paths_non_decreasing() {
    let m = make_mecrab();
    let paths = m
        .parse_nbest("すもももももももものうち", 3)
        .expect("parse_nbest failed");
    assert!(!paths.is_empty(), "n-best must return at least one path");

    // Costs must be non-decreasing
    for w in paths.windows(2) {
        assert!(
            w[0].1 <= w[1].1,
            "n-best costs not sorted: {} > {}",
            w[0].1,
            w[1].1
        );
    }

    // Best path must match intended segmentation
    let best: Vec<String> = paths[0]
        .0
        .morphemes
        .iter()
        .filter(|x| !x.surface.is_empty())
        .map(|x| x.surface.clone())
        .collect();
    assert_eq!(
        best,
        ["すもも", "も", "もも", "も", "もも", "の", "うち"],
        "best path surfaces: {:?}",
        best
    );
}

// ── Unknown word test ──────────────────────────────────────────────────────────

/// An out-of-vocabulary katakana string must produce at least one morpheme via
/// the unknown-word handler — it must never panic or return an empty result.
#[test]
fn unknown_word_no_panic() {
    let m = make_mecrab();
    // グーグル is not in the lexicon → unknown-word handler must fire
    let r = m.parse("グーグル").expect("parse must not fail for unknown word");
    let non_empty: Vec<&str> = r
        .morphemes
        .iter()
        .filter(|x| !x.surface.is_empty())
        .map(|x| x.surface.as_str())
        .collect();
    assert!(
        !non_empty.is_empty(),
        "unknown word must produce at least one morpheme, got: {:?}",
        r.morphemes
    );
}

// ── CoNLL-U format test ───────────────────────────────────────────────────────

/// The CoNLL-U output must contain the analysed surfaces.
#[test]
fn conllu_format() {
    let m = make_mecrab();
    let r = m.parse("東京は日本").expect("parse failed");
    let conllu = r.to_conllu();
    assert!(
        conllu.contains("東京"),
        "CoNLL-U output must contain 東京; got:\n{conllu}"
    );
    assert!(
        conllu.contains("日本"),
        "CoNLL-U output must contain 日本; got:\n{conllu}"
    );
    // Verify a noun-class UPOS tag appears for 東京 (proper nouns map to PROPN, common nouns to NOUN)
    assert!(
        conllu.contains("NOUN") || conllu.contains("PROPN"),
        "CoNLL-U output must contain NOUN or PROPN upos; got:\n{conllu}"
    );
}

// ── Empty input test ──────────────────────────────────────────────────────────

/// Parsing an empty string must not panic and must return a valid (empty) result.
#[test]
fn empty_input_no_panic() {
    let m = make_mecrab();
    let r = m.parse("");
    // Empty input: either Ok with empty morphemes or an Err — no panic
    match r {
        Ok(result) => {
            // If Ok, morphemes must be empty or only EOS
            let non_eos: Vec<_> = result
                .morphemes
                .iter()
                .filter(|m| !m.surface.is_empty() && m.surface != "EOS")
                .collect();
            assert!(
                non_eos.is_empty(),
                "empty input should produce no content morphemes; got: {:?}",
                non_eos
            );
        }
        Err(_) => {
            // Returning an error for empty input is also acceptable
        }
    }
}
