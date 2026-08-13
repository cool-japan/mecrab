//! Unknown-word grouping regression tests.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! These tests pin the unknown-word node generation implemented by
//! `lattice::Lattice::add_unknown_run_nodes` against MeCab's documented
//! semantics (`char.def`'s `GROUP` / `LENGTH` columns, mecab-0.996
//! `src/tokenizer.cpp::Tokenizer::lookup`):
//!
//! * `GROUP = 1` → **one** node spanning the whole maximal same-category run,
//!   emitted only while the run is short enough to group
//!   (`MAX_GROUPING_SIZE + 1 = 25` characters).
//! * `LENGTH = n` → prefix nodes of 1..=n characters, and nothing longer.
//!
//! Two properties are covered:
//!
//! 1. **Structural / complexity** — the number of lattice nodes a same-category
//!    run produces is linear in the run length, not quadratic. This is the
//!    timing-free regression guard for the O(n³) blow-up that made a 60,000
//!    character ASCII run untokenizable (uncatchable WASM trap).
//! 2. **Output identity** — the tokenisation of real Japanese text is
//!    unchanged. The corpus test is gated on a real IPADIC blob supplied via
//!    the `MECRAB_IPADIC_MCAB` environment variable and is a no-op without it;
//!    the synthetic-dictionary fixtures below always run.

use mecrab::dict::Dictionary;
use mecrab::lattice::Lattice;
use mecrab_builder::char_writer::{CharRange, build_char_bytes, pack_char_info};
use mecrab_builder::synthetic::build_synthetic_dictionary;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Total number of nodes in a lattice built over `text`.
fn lattice_node_count(text: &str, dict: &Dictionary) -> usize {
    let lattice = Lattice::build(text, dict).expect("lattice build must succeed");
    lattice.positions().map(<[_]>::len).sum()
}

/// Every `(start, end)` span an unknown-word node covers, sorted and deduplicated.
fn unknown_spans(text: &str, dict: &Dictionary) -> Vec<(usize, usize)> {
    let lattice = Lattice::build(text, dict).expect("lattice build must succeed");
    let mut spans: Vec<(usize, usize)> = lattice
        .positions()
        .flat_map(<[_]>::iter)
        .filter(|n| n.is_unknown)
        .map(|n| (n.start, n.end))
        .collect();
    spans.sort_unstable();
    spans.dedup();
    spans
}

/// Build a dictionary whose `char.bin` mirrors IPADIC's `char.def` flags for the
/// categories the tests exercise, so `GROUP` / `LENGTH` can be varied
/// independently of the synthetic dictionary's defaults.
///
/// `alpha` and `katakana` are `(length, group)` pairs applied to the ALPHA and
/// KATAKANA categories; every other category keeps the synthetic defaults.
fn dict_with_flags(alpha: (u8, bool), katakana: (u8, bool)) -> Dictionary {
    const CATEGORY_NAMES: &[&str] = &[
        "DEFAULT",
        "SPACE",
        "KANJI",
        "SYMBOL",
        "NUMERIC",
        "ALPHA",
        "HIRAGANA",
        "KATAKANA",
        "KANJINUMERIC",
        "GREEK",
        "CYRILLIC",
    ];

    // (category id, lo, hi, length, group)
    let ranges = [
        (1u8, 0x0020u32, 0x0020u32, 0u8, true),      // SPACE
        (3, 0x0021, 0x0021, 0, true),                // SYMBOL '!'
        (3, 0x3000, 0x303F, 0, true),                // SYMBOL (CJK punctuation)
        (4, 0x0030, 0x0039, 0, true),                // NUMERIC
        (5, 0x0041, 0x005A, alpha.0, alpha.1),       // ALPHA 'A'..'Z'
        (5, 0x0061, 0x007A, alpha.0, alpha.1),       // ALPHA 'a'..'z'
        (6, 0x3041, 0x3096, 2, true),                // HIRAGANA
        (7, 0x30A1, 0x30FC, katakana.0, katakana.1), // KATAKANA
        (8, 0x4E00, 0x4E00, 0, true),                // KANJINUMERIC
        (2, 0x4E01, 0x9FFF, 2, false),               // KANJI (group = 0, like IPADIC)
    ];

    let char_ranges: Vec<CharRange> = ranges
        .iter()
        .map(|&(id, lo, hi, length, group)| CharRange {
            lo,
            hi,
            info: pack_char_info(1u32 << id, id, length, group, true),
        })
        .collect();

    let char_bytes =
        build_char_bytes(CATEGORY_NAMES, &char_ranges).expect("char.bin must be buildable");

    let synth = build_synthetic_dictionary();
    Dictionary::from_bytes(&synth.sys_dic, &synth.matrix, &char_bytes, &synth.unk_def)
        .expect("dictionary must load")
}

// ── 1. Structural / complexity guards ────────────────────────────────────────

/// A same-category run must produce a node count that grows linearly with the
/// run length.
///
/// The pre-fix implementation emitted a node for *every* prefix of length ≥ 2 at
/// *every* position of the run — Θ(n²) nodes, found through a Θ(n) duplicate
/// scan, i.e. Θ(n³) work. This test fails loudly if that ever returns: it
/// compares the measured node count against a generous linear budget rather
/// than a wall-clock threshold, so it is deterministic on any machine.
#[test]
fn run_node_count_is_linear_not_quadratic() {
    let dict = dict_with_flags((0, true), (2, true));

    // Per position an unknown run may legitimately contribute:
    //   * one node per unk.dic template for the single character, plus
    //   * one node per template for each of the ≤ LENGTH prefixes, plus
    //   * one node per template for the whole run (when it is short enough).
    // The synthetic dictionary has ≤ 2 templates per category and LENGTH ≤ 2,
    // so 16 nodes per character is a comfortable ceiling — while a quadratic
    // implementation blows past it before n = 100.
    const BUDGET_PER_CHAR: usize = 16;

    for &n in &[100usize, 200, 400, 800, 1600] {
        let text = "a".repeat(n);
        let nodes = lattice_node_count(&text, &dict);
        assert!(
            nodes <= n * BUDGET_PER_CHAR,
            "ASCII run of {n} chars produced {nodes} nodes, over the linear budget of {}",
            n * BUDGET_PER_CHAR
        );
    }

    // Doubling the run length must not more than roughly double the node count.
    let n800 = lattice_node_count(&"a".repeat(800), &dict);
    let n1600 = lattice_node_count(&"a".repeat(1600), &dict);
    assert!(
        n1600 <= n800 * 3,
        "node count grew super-linearly: 800 chars → {n800} nodes, 1600 chars → {n1600} nodes"
    );

    // Same guard for a category that also emits prefix nodes (LENGTH = 2).
    let k800 = lattice_node_count(&"ア".repeat(800), &dict);
    let k1600 = lattice_node_count(&"ア".repeat(1600), &dict);
    assert!(
        k1600 <= k800 * 3,
        "katakana node count grew super-linearly: 800 → {k800}, 1600 → {k1600}"
    );
}

/// A 60,000 character ASCII run — the input shape that made the WASM demo trap —
/// must build a lattice and tokenise.
///
/// The pre-fix implementation needed 38.4 s for 3,200 characters, so this test
/// simply could not finish. It carries no timing assertion: completing at all is
/// the regression signal.
#[test]
fn very_long_run_completes() {
    let dict = dict_with_flags((0, true), (2, true));
    let text = "a".repeat(60_000);

    let nodes = lattice_node_count(&text, &dict);
    assert!(
        nodes <= 60_000 * 16,
        "60,000 char run produced {nodes} nodes — not linear"
    );

    let mecrab = mecrab::MeCrab::from_dictionary(dict);
    let result = mecrab.parse(&text).expect("60,000 char run must tokenise");
    assert!(!result.morphemes.is_empty());
    let surface: String = result
        .morphemes
        .iter()
        .map(|m| m.surface.as_str())
        .collect();
    assert_eq!(surface, text, "tokenisation must cover the whole input");
}

/// Pre-fix reference implementation, kept for the scaling measurement below.
///
/// This is the node emission `lattice::Lattice::add_grouped_unknown` used to
/// perform: a node for every prefix of length ≥ 2 at every position with no
/// dictionary hit, deduplicated by a linear scan of the destination bucket. It
/// exists only so that `scale_measurement` can time the old and the new
/// behaviour in one binary; nothing in the library calls it any more.
fn legacy_unknown_node_count(text: &str, dict: &Dictionary) -> usize {
    let mut buckets: Vec<Vec<(usize, usize)>> = vec![Vec::new(); text.len() + 2];

    for (pos, c) in text.char_indices() {
        if !dict.lookup(&text[pos..]).is_empty() {
            continue;
        }
        let category = dict.char_category(c);
        let templates = dict.unknown.generate_entries(category, c.len_utf8()).len();
        let end = pos + c.len_utf8();
        for _ in 0..templates.max(1) {
            buckets[end + 1].push((pos, end));
        }

        if !dict.char_def.should_group(category) {
            continue;
        }
        let mut length = 0;
        let mut char_count = 0;
        for c in text[pos..].chars() {
            if dict.char_category(c) != category {
                break;
            }
            length += c.len_utf8();
            char_count += 1;
            if char_count > 1 {
                let entries = dict.unknown.generate_entries(category, length);
                for _ in &entries {
                    let end = pos + length;
                    if end <= text.len()
                        && !buckets[end + 1].iter().any(|&(s, e)| s == pos && e == end)
                    {
                        buckets[end + 1].push((pos, end));
                    }
                }
            }
        }
    }

    buckets.iter().map(Vec::len).sum()
}

/// Manual measurement: how the pre-fix and post-fix implementations scale on
/// ASCII runs. Not a gate — run it explicitly:
///
/// ```text
/// cargo test -p mecrab --test unknown_grouping -- --ignored --nocapture scale
/// ```
#[test]
#[ignore = "manual measurement, not a correctness gate"]
fn scale_measurement() {
    use std::time::Instant;

    let dict = dict_with_flags((0, true), (2, true));

    println!("chars |     before (legacy)      |       after (current)");
    println!("------+--------------------------+--------------------------");
    for &n in &[400usize, 800, 1_600, 3_200, 8_000, 60_000] {
        let text = "a".repeat(n);

        // The legacy algorithm is Θ(n³); measuring it past a few thousand
        // characters is exactly the problem being fixed.
        let legacy = if n <= 3_200 {
            let t = Instant::now();
            let nodes = legacy_unknown_node_count(&text, &dict);
            Some((t.elapsed(), nodes))
        } else {
            None
        };

        let t = Instant::now();
        let nodes = lattice_node_count(&text, &dict);
        let current = (t.elapsed(), nodes);

        match legacy {
            Some((d, ln)) => println!(
                "{n:6}| {:>10.1?} {ln:>10} nodes | {:>10.1?} {:>10} nodes",
                d, current.0, current.1
            ),
            None => println!(
                "{n:6}| {:>10} {:>10}       | {:>10.1?} {:>10} nodes",
                "not run", "(Θ(n³))", current.0, current.1
            ),
        }
    }
}

// ── 2. MeCab semantics fixtures ──────────────────────────────────────────────

/// `GROUP = 1`, `LENGTH = 0` (IPADIC's ALPHA / NUMERIC / SYMBOL shape):
/// exactly one unknown node per position — the whole remaining run — and no
/// intermediate prefixes.
#[test]
fn group_without_length_emits_one_node_per_run() {
    let dict = dict_with_flags((0, true), (2, true));

    // 5 character run: at position i the run end is always 5.
    let spans = unknown_spans("abcde", &dict);
    assert_eq!(
        spans,
        vec![
            (0, 1),
            (0, 5),
            (1, 2),
            (1, 5),
            (2, 3),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
        "GROUP=1/LENGTH=0 must emit the single-character node plus one whole-run node"
    );
}

/// `LENGTH = n` caps prefix nodes at n characters.
///
/// KATAKANA in IPADIC is `INVOKE=1 GROUP=1 LENGTH=2`: prefixes of 1 and 2
/// characters, plus the whole-run node — never a 3 or 4 character prefix.
#[test]
fn length_caps_prefix_nodes() {
    let dict = dict_with_flags((0, true), (2, true));

    // 6 katakana characters (3 bytes each) with no dictionary entry.
    let text = "アアアアアア";
    let spans = unknown_spans(text, &dict);

    // From position 0: 1-char (0..3), 2-char (0..6) and the whole run (0..18).
    // Nothing of 3, 4 or 5 characters.
    assert!(spans.contains(&(0, 3)), "1 char prefix must exist");
    assert!(spans.contains(&(0, 6)), "2 char prefix must exist");
    assert!(spans.contains(&(0, 18)), "whole-run node must exist");
    for chars in 3..6 {
        let end = chars * 3;
        assert!(
            !spans.contains(&(0, end)),
            "LENGTH=2 must not emit a {chars} character prefix node (span 0..{end})"
        );
    }
}

/// `GROUP = 0` (IPADIC's KANJI shape) must not emit the whole-run node, but
/// `LENGTH` still applies.
#[test]
fn no_group_still_honours_length() {
    let dict = dict_with_flags((0, true), (2, true));

    // 侘 U+4F98, 寂 U+5BC2 … kanji that the synthetic lexicon does not contain.
    let text = "侘寂侘寂";
    let spans = unknown_spans(text, &dict);

    assert!(spans.contains(&(0, 3)), "1 char kanji node must exist");
    assert!(
        spans.contains(&(0, 6)),
        "LENGTH=2 must emit the 2 char kanji node even with GROUP=0"
    );
    assert!(
        !spans.contains(&(0, 12)),
        "GROUP=0 must not emit a whole-run node"
    );
}

/// The grouped node stops being emitted once the run exceeds MeCab's
/// `max-grouping-size`.
///
/// Verified against mecab-0.996 + IPADIC on the reference machine: a run of 25
/// `a` characters is one token, 26 is two (`a` + 25 `a`s), 27 is three. MeCab
/// counts the run from the *second* character, so the longest groupable run is
/// `MAX_GROUPING_SIZE + 1 = 25` characters.
#[test]
fn grouping_stops_past_max_grouping_size() {
    let dict = dict_with_flags((0, true), (2, true));

    let text25 = "a".repeat(25);
    let spans25 = unknown_spans(&text25, &dict);
    assert!(
        spans25.contains(&(0, 25)),
        "a 25 character run must still be grouped into one node"
    );

    let text26 = "a".repeat(26);
    let spans26 = unknown_spans(&text26, &dict);
    assert!(
        !spans26.contains(&(0, 26)),
        "a 26 character run must not produce a whole-run node"
    );
    assert!(
        spans26.contains(&(1, 26)),
        "the 25 character suffix of a 26 character run is still groupable"
    );
}

// ── 3. Real-dictionary output identity ───────────────────────────────────────

/// Parse the 36 byte MCAB packed-blob header and load the dictionary.
fn load_mcab_blob(path: &std::path::Path) -> Dictionary {
    let data = std::fs::read(path).expect("MECRAB_IPADIC_MCAB must be readable");
    assert!(data.len() > 36, "blob too small");
    let u32_at = |off: usize| -> usize {
        let bytes: [u8; 4] = data[off..off + 4].try_into().expect("4 bytes");
        u32::from_le_bytes(bytes) as usize
    };
    assert_eq!(u32_at(0), 0x4D43_4142, "blob magic must be MCAB");
    let section = |i: usize| -> &[u8] {
        let off = u32_at(4 + i * 8);
        let len = u32_at(8 + i * 8);
        &data[off..off + len]
    };
    Dictionary::from_bytes(section(0), section(1), section(2), section(3))
        .expect("packed IPADIC blob must load")
}

/// Lexicon size of the dictionary the expected fixture was captured against —
/// `ipadic-full-slim.mcab`, the blob the COOLJAPAN Playground /mecrab demo ships
/// (48,647,410 bytes, sha256 17a57626…).
///
/// Feature strings differ between IPADIC builds (the slim build drops the
/// base-form and pronunciation fields), so the fixture is only meaningful
/// against this dictionary; any other one skips instead of failing.
const FIXTURE_LEXICON_SIZE: usize = 325_871;

/// Tokenise `fixtures/corpus_ja.txt` with a real IPADIC dictionary and compare
/// against the checked-in expected analysis, field for field.
///
/// Skipped unless `MECRAB_IPADIC_MCAB` points at a packed MCAB dictionary blob,
/// so the repository stays dictionary-free. Set `MECRAB_CORPUS_DUMP=<path>` to
/// regenerate the fixture instead of comparing.
#[test]
fn ipadic_corpus_output_is_unchanged() {
    let Ok(blob_path) = std::env::var("MECRAB_IPADIC_MCAB") else {
        eprintln!("MECRAB_IPADIC_MCAB not set — skipping real-dictionary corpus test");
        return;
    };

    let dict = load_mcab_blob(std::path::Path::new(&blob_path));
    let dumping = std::env::var("MECRAB_CORPUS_DUMP").is_ok();
    if dict.size() != FIXTURE_LEXICON_SIZE && !dumping {
        eprintln!(
            "{blob_path} holds {} lexicon entries, but the expected fixture was captured \
             against the playground's ipadic-full-slim.mcab ({FIXTURE_LEXICON_SIZE} entries). \
             Feature strings differ between IPADIC builds — skipping rather than reporting \
             a false regression.",
            dict.size()
        );
        return;
    }
    let mecrab = mecrab::MeCrab::from_dictionary(dict);

    let corpus = include_str!("fixtures/corpus_ja.txt");
    let mut actual = String::new();
    for (idx, line) in corpus
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .enumerate()
    {
        let result = mecrab.parse(line).expect("corpus line must parse");
        for m in &result.morphemes {
            actual.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\n",
                idx + 1,
                m.start_byte,
                m.end_byte,
                m.surface,
                m.feature
            ));
        }
    }

    if let Ok(dump) = std::env::var("MECRAB_CORPUS_DUMP") {
        std::fs::write(&dump, &actual).expect("dump must be writable");
        eprintln!("wrote corpus analysis to {dump}");
        return;
    }

    let expected = include_str!("fixtures/corpus_ja.expected.tsv");
    if actual != expected {
        let diff: Vec<String> = expected
            .lines()
            .zip(actual.lines())
            .filter(|(e, a)| e != a)
            .take(20)
            .map(|(e, a)| format!("  expected: {e}\n  actual:   {a}"))
            .collect();
        panic!(
            "corpus analysis changed ({} expected lines, {} actual lines)\n{}",
            expected.lines().count(),
            actual.lines().count(),
            diff.join("\n")
        );
    }

    // Playground sample 1 (訪問記録風) is pinned separately: DESIGN-MECRAB §8.1
    // records exactly 42 morphemes for it.
    let sample1 = corpus
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .expect("corpus must have a first sentence");
    let morphemes = mecrab
        .parse(sample1)
        .expect("playground sample must parse")
        .morphemes
        .len();
    assert_eq!(
        morphemes, 42,
        "playground sample 1 must still analyse into 42 morphemes"
    );
}
