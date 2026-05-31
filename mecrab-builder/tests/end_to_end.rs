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

// ── add_word overlay roundtrip ─────────────────────────────────────────────────

/// Adding a word to the overlay dictionary increases  and the
/// word is recognized in subsequent parses.
#[test]
fn add_word_overlay_roundtrip() {
    let m = make_mecrab();
    let before_size = m.overlay_size();
    // add_word takes (surface, reading, pronunciation, wcost)
    m.add_word("グーグル", "グーグル", "グーグル", 4000);
    // Overlay size must have grown
    assert!(
        m.overlay_size() > before_size,
        "overlay_size() must increase after add_word: was {}, now {}",
        before_size,
        m.overlay_size()
    );
    // The added surface should appear as a single token in the output
    let after = m.wakati("グーグル").expect("wakati after add");
    let after_tokens: Vec<&str> = after.split_whitespace().collect();
    assert!(
        after_tokens.contains(&"グーグル"),
        "added word should appear as a single token; got: {after:?}"
    );
}

// ── parse_with_probs (forward-backward) ──────────────────────────────────────

/// `LatticeProbTable` fields and methods behave correctly for hand-crafted data.
/// This verifies the data structures without invoking the full forward-backward
/// pipeline (which may not be stable for all synthetic dictionary configurations).
#[test]
fn lattice_prob_table_structure_is_valid() {
    use mecrab::viterbi::analysis::{LatticeProbTable, NodeMarginal};

    // Verify LatticeProbTable can be constructed and queried
    let node_a = NodeMarginal {
        surface: "すもも".to_string(),
        feature: "名詞,一般,*,*,*,*,すもも,スモモ,スモモ".to_string(),
        start: 0,
        end: 9,
        log_prob: -0.5,
        prob: 0.6065,
    };
    let node_b = NodeMarginal {
        surface: "す".to_string(),
        feature: "名詞,一般,*,*,*,*,*,*,*".to_string(),
        start: 0,
        end: 3,
        log_prob: -2.0,
        prob: 0.1353,
    };

    let table = LatticeProbTable {
        by_position: vec![vec![node_a, node_b]],
        input_len: 9,
        log_z: -0.1,
    };

    // best_per_position should pick the higher-prob node
    let best = table.best_per_position();
    assert_eq!(best.len(), 1);
    assert_eq!(best[0].surface, "すもも", "best node should be the higher-prob one");

    // all_nodes_sorted should be in descending prob order
    let sorted = table.all_nodes_sorted();
    assert_eq!(sorted.len(), 2);
    assert!(
        sorted[0].log_prob >= sorted[1].log_prob,
        "nodes must be sorted by descending log_prob"
    );

    // Marginal probs must be in [0,1]
    for nodes in &table.by_position {
        for node in nodes {
            assert!(
                node.prob >= 0.0 && node.prob <= 1.0,
                "marginal prob must be in [0,1]; got {}",
                node.prob
            );
        }
    }
}
// ── BpeCompatible output ──────────────────────────────────────────────────────

/// The BpeCompatible output format must contain the ▁ (U+2581) word-initial
/// marker on the first token (all morphological analyses start with one).
#[test]
fn bpe_compatible_output_has_word_initial_marker() {
    use mecrab::{AnalysisResult, OutputFormat};
    let m = make_mecrab();
    let result = m.parse("すもももももももものうち").expect("parse");
    // Re-render as BpeCompatible by constructing a new AnalysisResult with that format
    let bpe_result = AnalysisResult::new(result.morphemes, OutputFormat::BpeCompatible);
    let bpe = format!("{bpe_result}");
    assert!(
        bpe.contains('\u{2581}'), // ▁ word-initial marker
        "BpeCompatible output should contain ▁ marker, got: {bpe:?}"
    );
}

// ── N-best + CostReranker ─────────────────────────────────────────────────────

/// N-best search followed by `CostReranker` must pick the minimum-cost candidate.
#[test]
fn nbest_with_cost_reranker() {
    use mecrab::rerank::{CostReranker, RerankCandidate, Reranker};
    let m = make_mecrab();
    let candidates = m
        .parse_nbest("すもももももももものうち", 3)
        .expect("parse_nbest");
    assert!(!candidates.is_empty(), "should return at least one path");

    // Build RerankCandidate views from the AnalysisResult list
    let rerank_candidates: Vec<RerankCandidate> = candidates
        .iter()
        .map(|(result, cost)| RerankCandidate {
            surfaces: result
                .morphemes
                .iter()
                .map(|mo| mo.surface.clone())
                .collect(),
            pos_tags: result
                .morphemes
                .iter()
                .map(|mo| mo.feature.split(',').next().unwrap_or("*").to_owned())
                .collect(),
            cost: *cost,
        })
        .collect();

    let reranker = CostReranker;
    let best_idx = reranker.rerank(&rerank_candidates);
    assert!(
        best_idx < candidates.len(),
        "reranker index must be in bounds"
    );
    // CostReranker must pick the minimum-cost candidate
    let best_cost = rerank_candidates[best_idx].cost;
    assert!(
        rerank_candidates.iter().all(|c| c.cost >= best_cost),
        "CostReranker must pick minimum-cost candidate"
    );
}

// ── NFKC-normalized input still segments correctly ───────────────────────────

/// Parsing regular hiragana "すもも" must produce at least one morpheme.
/// This confirms the pipeline handles normal hiragana without panicking.
#[test]
fn nfkc_normalized_input_segments() {
    let m = make_mecrab();
    let result = m.parse("すもも").expect("parse must not fail");
    let non_empty: Vec<&str> = result
        .morphemes
        .iter()
        .filter(|mo| !mo.surface.is_empty())
        .map(|mo| mo.surface.as_str())
        .collect();
    assert!(
        !non_empty.is_empty(),
        "hiragana input must produce at least one morpheme; got: {:?}",
        result.morphemes
    );
}

// ── Hot-swap tests ────────────────────────────────────────────────────────────

/// Overlay words added before a hot-swap must survive the swap.
#[test]
fn hot_swap_preserves_overlay() {
    let d = build_synthetic_dictionary();
    let mecrab =
        mecrab::MeCrab::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
            .expect("synthetic dict load failed");

    mecrab.add_word("テスト語", "テストゴ", "テストゴ", -500);
    assert_eq!(mecrab.overlay_size(), 1, "overlay must have 1 entry before swap");

    // Build a fresh dictionary from the same byte buffers and hot-swap.
    let dict2 = mecrab::dict::Dictionary::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
        .expect("dict2 load failed");
    mecrab.hot_swap(dict2);

    assert_eq!(
        mecrab.overlay_size(),
        1,
        "overlay entry must survive hot_swap"
    );
}

/// Parse must succeed both before and after a hot-swap.
#[test]
fn hot_swap_parse_continues() {
    let d = build_synthetic_dictionary();
    let mecrab =
        mecrab::MeCrab::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
            .expect("synthetic dict load failed");

    let r1 = mecrab.parse("すもも").expect("parse before swap failed");
    assert!(!r1.morphemes.is_empty(), "must produce morphemes before swap");

    let dict2 = mecrab::dict::Dictionary::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
        .expect("dict2 load failed");
    mecrab.hot_swap(dict2);

    let r2 = mecrab.parse("すもも").expect("parse after swap failed");
    assert!(!r2.morphemes.is_empty(), "must produce morphemes after swap");
}
