//! Integration tests verifying that the parallel lattice builder produces the
//! same node set as the sequential builder.
//!
//! These tests live in `mecrab-builder` rather than in `mecrab` itself so that
//! the synthetic dictionary is available without introducing a duplicate-crate
//! dependency diamond.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

/// Verify parallel and sequential lattice builders agree for the classic
/// Japanese sentence すもももももももものうち.
///
/// Node sets are compared as sorted (start, end, word_id) triples so that
/// bucket-internal ordering differences are ignored.
///
/// `mecrab` is compiled with `features = ["parallel"]` in mecrab-builder's
/// Cargo.toml, so `mecrab::lattice::Lattice::build_parallel` is always
/// available in this test crate.
#[test]
fn parallel_build_matches_sequential_sumomo() {
    use mecrab::lattice::{Lattice, LatticeNode};
    use mecrab_builder::synthetic::build_synthetic_dictionary;

    let synth = build_synthetic_dictionary();
    let dict = synth.load().expect("synthetic dictionary should load");

    let text = "すもももももももものうち";
    let text_len = text.len();

    // --- Sequential build ---
    let mut seq_nodes_at: Vec<Vec<LatticeNode>> = vec![Vec::new(); text_len + 2];
    seq_nodes_at[0].push(LatticeNode::bos());
    Lattice::build_sequential(text, text_len, &dict, &mut seq_nodes_at);
    seq_nodes_at[text_len + 1].push(LatticeNode::eos(text_len));

    // --- Parallel build ---
    let mut par_nodes_at: Vec<Vec<LatticeNode>> = vec![Vec::new(); text_len + 2];
    par_nodes_at[0].push(LatticeNode::bos());
    Lattice::build_parallel(text, text_len, &dict, &mut par_nodes_at);
    par_nodes_at[text_len + 1].push(LatticeNode::eos(text_len));

    // Compare node fingerprints bucket by bucket
    assert_eq!(
        seq_nodes_at.len(),
        par_nodes_at.len(),
        "nodes_at vectors must have the same length"
    );

    for (bucket_idx, (seq_bucket, par_bucket)) in
        seq_nodes_at.iter().zip(par_nodes_at.iter()).enumerate()
    {
        let mut seq_keys: Vec<(usize, usize, u32)> = seq_bucket
            .iter()
            .map(|n| (n.start, n.end, n.word_id))
            .collect();
        let mut par_keys: Vec<(usize, usize, u32)> = par_bucket
            .iter()
            .map(|n| (n.start, n.end, n.word_id))
            .collect();

        seq_keys.sort_unstable();
        par_keys.sort_unstable();

        assert_eq!(
            seq_keys, par_keys,
            "bucket {bucket_idx}: parallel and sequential node sets differ"
        );
    }
}
