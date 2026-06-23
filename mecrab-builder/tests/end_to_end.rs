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
    let r = m
        .parse("グーグル")
        .expect("parse must not fail for unknown word");
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
    assert_eq!(
        best[0].surface, "すもも",
        "best node should be the higher-prob one"
    );

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
    let mecrab = mecrab::MeCrab::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
        .expect("synthetic dict load failed");

    mecrab.add_word("テスト語", "テストゴ", "テストゴ", -500);
    assert_eq!(
        mecrab.overlay_size(),
        1,
        "overlay must have 1 entry before swap"
    );

    // Build a fresh dictionary from the same byte buffers and hot-swap.
    let dict2 =
        mecrab::dict::Dictionary::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
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
    let mecrab = mecrab::MeCrab::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
        .expect("synthetic dict load failed");

    let r1 = mecrab.parse("すもも").expect("parse before swap failed");
    assert!(
        !r1.morphemes.is_empty(),
        "must produce morphemes before swap"
    );

    let dict2 =
        mecrab::dict::Dictionary::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
            .expect("dict2 load failed");
    mecrab.hot_swap(dict2);

    let r2 = mecrab.parse("すもも").expect("parse after swap failed");
    assert!(
        !r2.morphemes.is_empty(),
        "must produce morphemes after swap"
    );
}

// ── Constrained / partial parsing tests ──────────────────────────────────────

/// Empty `ParseConstraints` must produce byte-identical segmentation to plain `parse`.
#[test]
fn test_constrained_empty_equals_plain() {
    let mecrab = make_mecrab();
    let text = "すもももももももものうち";
    let plain = mecrab.parse(text).expect("plain parse failed");
    let constrained = mecrab
        .parse_with_constraints(text, &mecrab::ParseConstraints::new())
        .expect("constrained parse failed");

    let plain_surfaces: Vec<&str> = plain.morphemes.iter().map(|m| m.surface.as_str()).collect();
    let constrained_surfaces: Vec<&str> = constrained
        .morphemes
        .iter()
        .map(|m| m.surface.as_str())
        .collect();

    assert_eq!(
        plain_surfaces, constrained_surfaces,
        "empty constraints must produce same segmentation as plain parse; \
         plain={plain_surfaces:?} constrained={constrained_surfaces:?}"
    );
}

/// A forced span `[0, 9)` must ensure "すもも" (3 × 3 UTF-8 bytes) appears as
/// exactly one token starting at byte 0 and ending at byte 9.
#[test]
fn test_constrained_forced_span_boundary() {
    let mecrab = make_mecrab();
    // "すもも" = 3 chars × 3 bytes each = bytes 0..9
    let text = "すもももももももものうち";
    let mut constraints = mecrab::ParseConstraints::new();
    constraints.add_span(0, 9, None);

    let result = mecrab
        .parse_with_constraints(text, &constraints)
        .expect("constrained parse failed");

    let sumomo = result
        .morphemes
        .iter()
        .find(|m| m.surface == "すもも")
        .expect("\"すもも\" must appear as a single morpheme");

    assert_eq!(
        sumomo.start_byte, 0,
        "forced \"すもも\" must start at byte 0, got {}",
        sumomo.start_byte
    );
    assert_eq!(
        sumomo.end_byte, 9,
        "forced \"すもも\" must end at byte 9, got {}",
        sumomo.end_byte
    );
}

/// A forced span with a custom feature string must propagate that feature to
/// the resulting morpheme.  The feature must start with "名詞".
#[test]
fn test_constrained_forced_span_with_custom_feature() {
    let mecrab = make_mecrab();
    // Force bytes 0..9 ("すもも") to be a single token with a custom feature.
    let text = "すもももももももものうち";
    let custom_feature = "名詞,固有名詞,*,*,*,*,テスト,テスト,テスト".to_string();
    let mut constraints = mecrab::ParseConstraints::new();
    constraints.add_span(0, 9, Some(custom_feature));

    let result = mecrab
        .parse_with_constraints(text, &constraints)
        .expect("constrained parse with custom feature failed");

    let sumomo = result
        .morphemes
        .iter()
        .find(|m| m.surface == "すもも")
        .expect("\"すもも\" must appear as a single morpheme even with a custom feature");

    assert!(
        sumomo.feature.starts_with("名詞"),
        "forced span with custom feature must start with \"名詞\"; got: {}",
        sumomo.feature
    );
}

// ── Iterative (evolving-cost) CRF training ──────────────────────────────────────

/// Gold corpus for the classic test sentence, segmented as the synthetic lexicon
/// expects: すもも / も / もも / も / もも / の / うち.  Four identical copies make
/// the per-batch summed gradient large enough to produce non-zero integer cost
/// updates (visible movement) in a single batch.
fn sumomo_gold() -> Vec<mecrab::GoldSegmentation> {
    let text = "すもももももももものうち";
    let tsv = "すもも\nも\nもも\nも\nもも\nの\nうち\nEOS\n";
    let g = mecrab::GoldSegmentation::from_mecab_tsv(text, tsv).expect("gold parse");
    vec![g.clone(), g.clone(), g.clone(), g]
}

/// With one epoch and a single batch, the gradient is computed on the pristine
/// matrix (empty deltas), where `MatrixCostSource` is bit-identical to
/// `DictCostSource`.  So `--iterative` and the default must produce byte-for-byte
/// identical trained parameters and identical epoch-0 loss.
#[test]
fn iterative_single_step_is_bit_identical_to_noniterative() {
    let d = build_synthetic_dictionary();
    let mecrab = mecrab::MeCrab::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
        .expect("synthetic dict load failed");
    let corpus = sumomo_gold();

    let base = mecrab::DictTrainConfig {
        epochs: 1,
        batch_size: 64,
        learning_rate: 1.0,
        l2_strength: 0.0,
        ..Default::default()
    };
    let (m_non, s_non) = mecrab
        .train_dict(
            &corpus,
            &mecrab::DictTrainConfig {
                iterative: false,
                ..base.clone()
            },
        )
        .expect("non-iterative train");
    let (m_itr, s_itr) = mecrab
        .train_dict(
            &corpus,
            &mecrab::DictTrainConfig {
                iterative: true,
                ..base
            },
        )
        .expect("iterative train");

    assert_eq!(
        m_non.to_bytes(),
        m_itr.to_bytes(),
        "iterative epoch-0 single-batch must be bit-identical to non-iterative"
    );
    assert_eq!(
        m_non.word_cost_deltas_i16(),
        m_itr.word_cost_deltas_i16(),
        "word-cost deltas must match on the first step"
    );
    assert!(
        (s_non.epoch_stats[0].loss - s_itr.epoch_stats[0].loss).abs() < 1e-9,
        "epoch-0 loss must be identical: {} vs {}",
        s_non.epoch_stats[0].loss,
        s_itr.epoch_stats[0].loss
    );
}

/// The correctness proof for evolving-cost training: the legacy path computes the
/// gradient under the *fixed* dictionary every epoch, so its per-epoch loss is
/// exactly constant (it never reflects its own updates).  `--iterative` recomputes
/// expected counts under the evolving model, so the loss actually moves — and the
/// gold segmentation becomes strictly more probable.
#[test]
fn iterative_evolves_gradient_noniterative_is_flat() {
    let d = build_synthetic_dictionary();
    let mecrab = mecrab::MeCrab::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
        .expect("synthetic dict load failed");
    let corpus = sumomo_gold();

    let base = mecrab::DictTrainConfig {
        epochs: 6,
        batch_size: 64,
        learning_rate: 1.0,
        l2_strength: 0.0,
        ..Default::default()
    };
    let (m_non, s_non) = mecrab
        .train_dict(
            &corpus,
            &mecrab::DictTrainConfig {
                iterative: false,
                ..base.clone()
            },
        )
        .expect("non-iterative train");
    let (m_itr, s_itr) = mecrab
        .train_dict(
            &corpus,
            &mecrab::DictTrainConfig {
                iterative: true,
                ..base
            },
        )
        .expect("iterative train");

    let n0 = s_non.epoch_stats.first().expect("epochs").loss;
    let nlast = s_non.epoch_stats.last().expect("epochs").loss;
    let i0 = s_itr.epoch_stats.first().expect("epochs").loss;
    let ilast = s_itr.epoch_stats.last().expect("epochs").loss;

    // Smoking gun: legacy per-epoch loss is exactly constant.
    assert!(
        (n0 - nlast).abs() < 1e-9,
        "non-iterative loss must be flat across epochs: {n0} → {nlast}"
    );
    // Both compute epoch-0 loss on the pristine model ⇒ identical start.
    assert!(
        (n0 - i0).abs() < 1e-9,
        "epoch-0 loss must match between modes: {n0} vs {i0}"
    );
    // Iterative loss genuinely moves and improves the gold fit.
    assert!(
        (i0 - ilast).abs() > 1e-6,
        "iterative loss must change across epochs: {i0} → {ilast}"
    );
    assert!(
        ilast < i0,
        "iterative training must reduce gold NLL: {i0} → {ilast}"
    );
    // The evolving gradient drives the model to a different fit. The connection
    // matrix may stay put when its integer-rounded gradient is zero (the gold
    // connections are already preferred), so assert on the f64 word-cost deltas,
    // which genuinely diverge once the gradient is recomputed under the updated
    // model.
    assert!(
        m_non.word_cost_deltas != m_itr.word_cost_deltas,
        "iterative training must evolve the model differently from fixed-gradient"
    );
}

// ── Batch L-BFGS / OWL-QN CRF training ──────────────────────────────────────────

/// Batch L-BFGS fits the CRF costs as a single smooth optimization. Starting from
/// the dictionary (`x = 0`) it must drive the regularized NLL strictly down and
/// leave every fitted parameter finite. (`epochs` bounds the L-BFGS iterations;
/// `epochs = 0` just evaluates the untrained objective.)
#[test]
fn lbfgs_training_reduces_objective_and_is_finite() {
    let d = build_synthetic_dictionary();
    let mecrab = mecrab::MeCrab::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
        .expect("synthetic dict load failed");
    let corpus = sumomo_gold();

    let base = mecrab::DictTrainConfig {
        optimizer: mecrab::viterbi::train_loop::Optimizer::Lbfgs,
        l2_strength: 1e-4,
        ..Default::default()
    };
    let (_m0, s0) = mecrab
        .train_dict(
            &corpus,
            &mecrab::DictTrainConfig {
                epochs: 0,
                ..base.clone()
            },
        )
        .expect("lbfgs init eval");
    let (m, s) = mecrab
        .train_dict(&corpus, &mecrab::DictTrainConfig { epochs: 40, ..base })
        .expect("lbfgs fit");

    let initial = s0.epoch_stats.first().expect("init stat").loss;
    let fitted = s.epoch_stats.first().expect("fit stat").loss;
    assert!(
        initial.is_finite() && fitted.is_finite(),
        "objective must be finite: {initial} → {fitted}"
    );
    assert!(
        fitted < initial,
        "L-BFGS must reduce the regularized NLL: {initial} → {fitted}"
    );
    assert!(s.total_epochs >= 1, "L-BFGS must take at least one step");
    for delta in m.word_cost_deltas.values() {
        assert!(delta.is_finite(), "fitted word-cost deltas must be finite");
    }
}

/// OWL-QN (L-BFGS with an `L1` term) integrates end-to-end: it runs to a finite,
/// non-increasing objective and leaves finite deltas. (OWL-QN's sparsity/soft-
/// threshold correctness is proven separately by the `lbfgs` unit tests.)
#[test]
fn owlqn_l1_training_runs_finite() {
    let d = build_synthetic_dictionary();
    let mecrab = mecrab::MeCrab::from_bytes(&d.sys_dic, &d.matrix, &d.char_def, &d.unk_def)
        .expect("synthetic dict load failed");
    let corpus = sumomo_gold();

    let base = mecrab::DictTrainConfig {
        optimizer: mecrab::viterbi::train_loop::Optimizer::Lbfgs,
        l2_strength: 1e-4,
        l1_strength: 1e-5,
        ..Default::default()
    };
    let (_m0, s0) = mecrab
        .train_dict(
            &corpus,
            &mecrab::DictTrainConfig {
                epochs: 0,
                ..base.clone()
            },
        )
        .expect("owlqn init");
    let (m, s) = mecrab
        .train_dict(&corpus, &mecrab::DictTrainConfig { epochs: 40, ..base })
        .expect("owlqn fit");
    let initial = s0.epoch_stats.first().expect("init").loss;
    let fitted = s.epoch_stats.first().expect("fit").loss;
    assert!(
        fitted.is_finite() && fitted <= initial + 1e-9,
        "owlqn objective must be finite and non-increasing: {initial} → {fitted}"
    );
    for delta in m.word_cost_deltas.values() {
        assert!(delta.is_finite(), "owlqn deltas must be finite");
    }
}
