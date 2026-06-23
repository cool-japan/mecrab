//! Forward-backward algorithm for marginal probability computation.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module contains the probabilistic forward-backward algorithm used to
//! compute per-node marginal probabilities over the lattice.  It is separated
//! from the deterministic Viterbi path-finding code in [`super`] to keep each
//! algorithm self-contained and within the 2000-line refactoring policy limit.

use std::collections::HashMap;

use crate::dict::Dictionary;
use crate::lattice::{Lattice, LatticeNode};
use crate::viterbi::ViterbiSolver;
use crate::viterbi::analysis::{LatticeProbTable, NodeMarginal};

/// Boltzmann temperature for converting integer MeCab costs to log-probabilities.
///
/// Dividing by 500 keeps `exp(-cost/T)` in a numerically workable range for
/// typical IPADIC costs (±32 000).  Shared by `forward_backward` and the edge
/// expected-count computation.
const TEMPERATURE: f64 = 500.0;

// ── Numerics ─────────────────────────────────────────────────────────────────

/// Numerically stable log-sum-exp for two values.
///
/// `log_sum_exp(a, b) = max(a, b) + ln(1 + exp(-|a - b|))`
///
/// Handles `NEG_INFINITY` as the log-domain zero (additive identity).
#[inline]
pub(super) fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (larger, smaller) = if a >= b { (a, b) } else { (b, a) };
    larger + (1.0_f64 + (smaller - larger).exp()).ln()
}

// ── Edge cost source ──────────────────────────────────────────────────────────

/// Source of the two cost families consumed by the forward-backward pass:
/// connection (arc) costs and per-node word costs.
///
/// Abstracting the lookup lets the *same* forward-backward / edge-count code run
/// either against the immutable system dictionary (inference: `parse_with_probs`,
/// `score`, `LatticeProb`), against an **evolving integer** parameter set during
/// iterative CRF training, or against a **continuous** parameter set during
/// batch L-BFGS / OWL-QN fitting.  Costs are returned as `f64` in MeCab cost
/// units: for the integer sources this is exact (an `i16`/`i32` value is
/// represented exactly by `f64`), while the L-BFGS source returns fractional
/// costs so the objective stays smooth for the line search.
///
/// The default [`DictCostSource`] reproduces the historical hard-coded
/// `dict.connection_cost(...)` / `node.wcost` lookups exactly: for any `i16`
/// value `v`, `f64::from(v) == v as f64`, so routing inference through this trait
/// is byte-identical to the previous direct field access.
pub(crate) trait EdgeCostSource {
    /// Connection cost for the arc `right_id → left_id`, in MeCab cost units.
    fn connection_cost(&self, right_id: u16, left_id: u16) -> f64;
    /// Word (node) cost for `node`, in MeCab cost units.
    fn word_cost(&self, node: &LatticeNode<'_>) -> f64;
}

/// [`EdgeCostSource`] backed by the immutable system dictionary — the inference
/// default.  Byte-identical to the original hard-coded cost lookups.
pub(crate) struct DictCostSource<'d> {
    /// Dictionary providing connection costs; node word costs come from the
    /// lattice node's baked-in `wcost`.
    pub dict: &'d Dictionary,
}

impl EdgeCostSource for DictCostSource<'_> {
    #[inline]
    fn connection_cost(&self, right_id: u16, left_id: u16) -> f64 {
        f64::from(self.dict.connection_cost(right_id, left_id))
    }

    #[inline]
    fn word_cost(&self, node: &LatticeNode<'_>) -> f64 {
        f64::from(node.wcost)
    }
}

// ── Forward-Backward impl on ViterbiSolver ───────────────────────────────────

#[allow(clippy::elidable_lifetime_names)]
impl<'a> ViterbiSolver<'a> {
    /// Run the forward-backward algorithm over the lattice to compute
    /// marginal probabilities for every node.
    ///
    /// ## Temperature scaling
    ///
    /// MeCab integer costs are on the order of ±32 000.  We divide by
    /// `TEMPERATURE = 500` so that the Boltzmann weights `exp(-cost/T)` stay
    /// in a numerically workable range and produce meaningful probability
    /// distributions.  Increasing the temperature flattens the distribution;
    /// decreasing it sharpens it toward the Viterbi path.
    ///
    /// ## Lattice layout
    ///
    /// The lattice stores nodes by **end position index**: `nodes_at[p]` holds
    /// all nodes whose `node.end == p - 1` (byte end position).  Position 0
    /// holds the BOS node; position `n - 1` holds the EOS node (where
    /// `n = lattice.len()`).
    ///
    /// ## Returns
    ///
    /// A [`crate::viterbi::analysis::LatticeProbTable`] with per-node marginals.
    /// The `by_position` field mirrors the lattice `nodes_at` indexing.
    pub fn forward_backward<'b>(&self, lattice: &'b Lattice<'b>) -> LatticeProbTable {
        self.forward_backward_with(
            lattice,
            &DictCostSource {
                dict: self.dictionary,
            },
        )
    }

    /// Forward-backward over the lattice using an explicit [`EdgeCostSource`].
    ///
    /// Identical to [`forward_backward`](Self::forward_backward) except the
    /// connection and node-word costs come from `costs` instead of being
    /// hard-coded to the system dictionary, so the same code drives both
    /// inference ([`DictCostSource`]) and iterative CRF training
    /// (`MatrixCostSource`).
    pub(crate) fn forward_backward_with<'b, S: EdgeCostSource>(
        &self,
        lattice: &'b Lattice<'b>,
        costs: &S,
    ) -> LatticeProbTable {
        let n = lattice.len();
        if n == 0 {
            return LatticeProbTable::default();
        }

        // Forward (alpha), backward (beta) and the log partition function are
        // computed by the shared helper so the subtle EOS handling lives in one
        // place (see [`forward_backward_tables`]).
        let (alpha, beta, log_z) = forward_backward_tables(lattice, costs);

        // ── Marginals ─────────────────────────────────────────────────────────
        // P(node at pos j) = exp(alpha[pos][j] + beta[pos][j] - log_z)
        let mut by_position: Vec<Vec<NodeMarginal>> = vec![vec![]; n];

        if log_z != f64::NEG_INFINITY {
            for pos in 0..n {
                let nodes = lattice.nodes_ending_at(pos);
                for (i, node) in nodes.iter().enumerate() {
                    let a = alpha[pos].get(i).copied().unwrap_or(f64::NEG_INFINITY);
                    let b = beta[pos].get(i).copied().unwrap_or(f64::NEG_INFINITY);
                    if a == f64::NEG_INFINITY || b == f64::NEG_INFINITY {
                        continue;
                    }
                    let log_prob = (a + b) - log_z;
                    // Clamp to [0, 1] after exp to guard against floating-point drift
                    let prob = log_prob.exp().clamp(0.0_f64, 1.0_f64);
                    by_position[pos].push(NodeMarginal {
                        surface: node.surface.to_string(),
                        feature: node.feature.to_string(),
                        start: node.start,
                        end: node.end,
                        log_prob,
                        prob,
                    });
                }
                // Sort by descending probability for easy reading
                by_position[pos].sort_by(|x, y| {
                    y.log_prob
                        .partial_cmp(&x.log_prob)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let input_len = lattice.text.len();
        LatticeProbTable {
            by_position,
            input_len,
            log_z,
        }
    }

    /// Compute expected edge counts from forward (alpha) and backward (beta) scores.
    ///
    /// Internally reruns the forward-backward algorithm to obtain the alpha and beta
    /// tables, then computes edge marginals for every arc in the lattice.
    ///
    /// # Returns
    /// A `HashMap<(right_id, left_id), expected_count>` where each entry is the
    /// expected number of times that connection type is used across all possible
    /// segmentations, weighted by their probability under the current model.
    ///
    /// The edge marginal for arc u→v is:
    ///   p(u→v) = exp(alpha[u] + log_conn(u,v) + (-v.wcost/T) + beta[v] - log_Z)
    pub fn compute_edge_expected_counts<'b>(
        &self,
        lattice: &'b Lattice<'b>,
    ) -> HashMap<(u16, u16), f64> {
        self.compute_edge_expected_counts_with(
            lattice,
            &DictCostSource {
                dict: self.dictionary,
            },
        )
    }

    /// Expected connection counts using an explicit [`EdgeCostSource`].
    ///
    /// Identical to [`compute_edge_expected_counts`](Self::compute_edge_expected_counts)
    /// except all connection/word costs come from `costs`, so the edge marginals
    /// reflect the *current* model during iterative CRF training.
    pub(crate) fn compute_edge_expected_counts_with<'b, S: EdgeCostSource>(
        &self,
        lattice: &'b Lattice<'b>,
        costs: &S,
    ) -> HashMap<(u16, u16), f64> {
        use crate::viterbi::train::edge_expected_counts_from_fb;

        let n = lattice.len();
        if n == 0 {
            return HashMap::new();
        }

        // Shared forward-backward; edge marginals are derived in
        // `edge_expected_counts_from_fb`.
        let (alpha, beta, log_z) = forward_backward_tables(lattice, costs);
        edge_expected_counts_from_fb(lattice, &alpha, &beta, log_z, costs)
    }
}

// ── Forward-backward tables ─────────────────────────────────────────────────────

/// Forward (alpha) and backward (beta) log-score tables plus the log partition
/// function for `lattice` under `costs`.
///
/// Shared by [`ViterbiSolver::forward_backward_with`] and
/// [`ViterbiSolver::compute_edge_expected_counts_with`] so the subtle EOS
/// handling lives in exactly one place.
///
/// ## Layout & EOS
///
/// `alpha[pos][i]` / `beta[pos][i]` index the `i`-th node of
/// `lattice.nodes_ending_at(pos)`. The lattice stores a node ending at byte `e`
/// at position `e + 1`, with BOS at position 0 and EOS (zero-length,
/// `start == end == text_len`) at position `n - 1` — the **same** position as the
/// real nodes ending at the last byte. The single arc `final_word → EOS` is
/// therefore intra-position: the forward pass reads same-position predecessors
/// from the in-progress row (`alpha_pos`), and the backward pass seeds position
/// `n - 1` by walking its bucket in reverse (EOS, the terminal, first).
///
/// The lattice invariant — a node ending at byte `b` is stored at exactly one
/// position, `b + 1` — means a node's predecessors all live in a single bucket,
/// so no multi-position "span" scan is needed (mirrors the Viterbi forward pass).
#[allow(clippy::too_many_lines)]
fn forward_backward_tables<S: EdgeCostSource>(
    lattice: &Lattice<'_>,
    costs: &S,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, f64) {
    let n = lattice.len();
    let mut alpha: Vec<Vec<f64>> = (0..n).map(|_| Vec::new()).collect();
    let mut beta: Vec<Vec<f64>> = (0..n).map(|_| Vec::new()).collect();
    if n == 0 {
        return (alpha, beta, f64::NEG_INFINITY);
    }

    // ── Forward: alpha[pos][j] = log P(BOS → node j ending at `pos`) ──────────
    alpha[0] = vec![0.0_f64; lattice.nodes_ending_at(0).len()];
    for pos in 1..n {
        let nodes = lattice.nodes_ending_at(pos);
        let mut alpha_pos = vec![f64::NEG_INFINITY; nodes.len()];
        for (j, node) in nodes.iter().enumerate() {
            let wcost_contrib = -costs.word_cost(node) / TEMPERATURE;
            let prev_pos = if node.start == 0 { 0 } else { node.start + 1 };
            if prev_pos >= n {
                continue;
            }
            let prev_nodes = lattice.nodes_ending_at(prev_pos);
            for (i, prev_node) in prev_nodes.iter().enumerate() {
                // Same-position arc (EOS only): valid predecessors are strictly
                // earlier in the bucket; the node itself and later entries aren't.
                if prev_pos == pos && i >= j {
                    break;
                }
                let a_prev = if prev_pos == pos {
                    alpha_pos[i]
                } else {
                    alpha[prev_pos][i]
                };
                if a_prev == f64::NEG_INFINITY {
                    continue;
                }
                let conn = costs.connection_cost(prev_node.right_id, node.left_id);
                let score = a_prev + (-conn / TEMPERATURE) + wcost_contrib;
                alpha_pos[j] = log_sum_exp(alpha_pos[j], score);
            }
        }
        alpha[pos] = alpha_pos;
    }

    // ── Backward: beta[pos][j] = log P(node j → EOS) ─────────────────────────
    // Seed position n-1 (final real nodes + EOS) by walking the bucket in
    // reverse, so EOS (the terminal, pushed last) is set before its predecessors.
    {
        let last = lattice.nodes_ending_at(n - 1);
        let mut beta_last = vec![f64::NEG_INFINITY; last.len()];
        for j in (0..last.len()).rev() {
            let node = &last[j];
            if node.start == node.end {
                // EOS / zero-length terminal: log P(→ EOS) = log 1 = 0.
                beta_last[j] = 0.0;
            } else {
                for k in (j + 1)..last.len() {
                    let succ = &last[k];
                    if succ.start == node.end {
                        let conn = costs.connection_cost(node.right_id, succ.left_id);
                        let succ_w = -costs.word_cost(succ) / TEMPERATURE;
                        let score = beta_last[k] + (-conn / TEMPERATURE) + succ_w;
                        beta_last[j] = log_sum_exp(beta_last[j], score);
                    }
                }
            }
        }
        beta[n - 1] = beta_last;
    }

    for pos in (0..n - 1).rev() {
        let nodes_at_pos = lattice.nodes_ending_at(pos);
        let mut beta_pos = vec![f64::NEG_INFINITY; nodes_at_pos.len()];
        // Iterate the successor beta rows by slice (not by index) to satisfy
        // clippy::needless_range_loop while keeping the positional `next_pos`.
        for (offset, beta_next) in beta[(pos + 1)..n].iter().enumerate() {
            let next_pos = pos + 1 + offset;
            let next_nodes = lattice.nodes_ending_at(next_pos);
            for (jn, next_node) in next_nodes.iter().enumerate() {
                let next_beta_j = beta_next.get(jn).copied().unwrap_or(f64::NEG_INFINITY);
                if next_beta_j == f64::NEG_INFINITY {
                    continue;
                }
                // node at `pos` precedes next_node iff next_node's predecessor
                // slot equals `pos` (all nodes at `pos` end at byte `pos - 1`).
                let expected_prev_pos = if next_node.start == 0 {
                    0
                } else {
                    next_node.start + 1
                };
                if expected_prev_pos != pos {
                    continue;
                }
                let next_w = -costs.word_cost(next_node) / TEMPERATURE;
                for (i, node) in nodes_at_pos.iter().enumerate() {
                    let conn = costs.connection_cost(node.right_id, next_node.left_id);
                    let score = next_beta_j + (-conn / TEMPERATURE) + next_w;
                    beta_pos[i] = log_sum_exp(beta_pos[i], score);
                }
            }
        }
        beta[pos] = beta_pos;
    }

    // ── Partition function: log Z = log Σ_i alpha[0][i] + beta[0][i] ─────────
    let bos = lattice.nodes_ending_at(0);
    let log_z = (0..bos.len()).fold(f64::NEG_INFINITY, |acc, i| {
        let a = alpha[0].get(i).copied().unwrap_or(f64::NEG_INFINITY);
        let b = beta[0].get(i).copied().unwrap_or(f64::NEG_INFINITY);
        if a == f64::NEG_INFINITY || b == f64::NEG_INFINITY {
            acc
        } else {
            log_sum_exp(acc, a + b)
        }
    });

    (alpha, beta, log_z)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{EdgeCostSource, forward_backward_tables, log_sum_exp};
    use crate::lattice::{Lattice, LatticeNode};

    /// Constant-cost source: every arc and word cost the same.
    struct ConstCost {
        conn: f64,
        word: f64,
    }
    impl EdgeCostSource for ConstCost {
        fn connection_cost(&self, _r: u16, _l: u16) -> f64 {
            self.conn
        }
        fn word_cost(&self, _node: &LatticeNode<'_>) -> f64 {
            self.word
        }
    }

    /// Zero connection cost; word cost read from each node's `wcost`.
    struct NodeWordCost;
    impl EdgeCostSource for NodeWordCost {
        fn connection_cost(&self, _r: u16, _l: u16) -> f64 {
            0.0
        }
        fn word_cost(&self, node: &LatticeNode<'_>) -> f64 {
            f64::from(node.wcost)
        }
    }

    /// Build a plain lattice node spanning bytes `s..e` of `text`.
    fn mk(text: &str, s: usize, e: usize) -> LatticeNode<'_> {
        LatticeNode {
            surface: &text[s..e],
            start: s,
            end: e,
            word_id: 0,
            left_id: 1,
            right_id: 1,
            pos_id: 0,
            wcost: 0,
            feature: std::sync::Arc::from(""),
            is_unknown: false,
        }
    }

    /// Hand-built lattice for "ab" with two equal-cost paths
    /// (BOS→a→b→EOS and BOS→ab→EOS). With zero costs every complete path is
    /// equally weighted, so forward Z (`alpha[EOS]`) and backward Z (`log_z`
    /// from BOS) must both equal ln 2, P(a)=P(b)=P(ab)=0.5, P(BOS)=P(EOS)=1.
    /// This exercises the intra-position EOS arc that previously panicked.
    #[test]
    fn test_forward_backward_tables_two_equal_paths() {
        let text = "ab";
        // positions 0..=3 (= text_len + 2); a node ending at byte e sits at e+1.
        let mut buckets: Vec<Vec<LatticeNode>> = vec![Vec::new(); 4];
        buckets[0].push(LatticeNode::bos()); // BOS at pos 0
        buckets[2].push(mk(text, 0, 1)); // "a" ends at 1 → pos 2
        buckets[3].push(mk(text, 1, 2)); // "b" ends at 2 → pos 3
        buckets[3].push(mk(text, 0, 2)); // "ab" ends at 2 → pos 3
        buckets[3].push(LatticeNode::eos(2)); // EOS at pos 3 (last in bucket)
        let lattice = Lattice::from_nodes_at(text, buckets);

        let (alpha, beta, log_z) = forward_backward_tables(
            &lattice,
            &ConstCost {
                conn: 0.0,
                word: 0.0,
            },
        );
        let ln2 = std::f64::consts::LN_2;

        assert!((log_z - ln2).abs() < 1e-9, "log_z={log_z} expected ln2");
        let eos_alpha = *alpha[3].last().unwrap();
        assert!(
            (eos_alpha - ln2).abs() < 1e-9,
            "forward Z (alpha[EOS])={eos_alpha} must equal backward Z {log_z}"
        );

        let marg = |pos: usize, i: usize| (alpha[pos][i] + beta[pos][i] - log_z).exp();
        assert!((marg(0, 0) - 1.0).abs() < 1e-9, "P(BOS) must be 1");
        assert!((marg(2, 0) - 0.5).abs() < 1e-9, "P(a) must be 0.5");
        assert!((marg(3, 0) - 0.5).abs() < 1e-9, "P(b) must be 0.5");
        assert!((marg(3, 1) - 0.5).abs() < 1e-9, "P(ab) must be 0.5");
        assert!((marg(3, 2) - 1.0).abs() < 1e-9, "P(EOS) must be 1");
    }

    /// Biased variant: the a→b path carries word cost so the single-token "ab"
    /// path dominates. Forward and backward Z must still agree, all marginals
    /// stay in [0, 1], and the two competing first-segment nodes' marginals
    /// (P(a) and P(ab)) sum to 1.
    #[test]
    fn test_forward_backward_tables_z_consistency_biased() {
        let text = "ab";
        let mut buckets: Vec<Vec<LatticeNode>> = vec![Vec::new(); 4];
        buckets[0].push(LatticeNode::bos());
        let mut a = mk(text, 0, 1);
        a.wcost = 100;
        buckets[2].push(a);
        let mut b = mk(text, 1, 2);
        b.wcost = 100;
        buckets[3].push(b);
        buckets[3].push(mk(text, 0, 2)); // "ab", wcost 0 (cheaper)
        buckets[3].push(LatticeNode::eos(2));
        let lattice = Lattice::from_nodes_at(text, buckets);

        let (alpha, beta, log_z) = forward_backward_tables(&lattice, &NodeWordCost);
        let eos_alpha = *alpha[3].last().unwrap();
        assert!(
            (eos_alpha - log_z).abs() < 1e-9,
            "forward Z {eos_alpha} must equal backward Z {log_z}"
        );

        let p_ab = (alpha[3][1] + beta[3][1] - log_z).exp();
        let p_a = (alpha[2][0] + beta[2][0] - log_z).exp();
        assert!(
            p_ab > p_a,
            "cheaper single-token path must dominate: P(ab)={p_ab} P(a)={p_a}"
        );
        assert!((0.0..=1.0).contains(&p_ab) && (0.0..=1.0).contains(&p_a));
        assert!(
            (p_ab + p_a - 1.0).abs() < 1e-9,
            "first-segment marginals must sum to 1: {p_ab} + {p_a}"
        );
    }

    // ── log_sum_exp unit tests ───────────────────────────────────────────────

    #[test]
    fn test_log_sum_exp_identity_with_neg_infinity() {
        // log_sum_exp(-inf, b) == b
        let b = -1.5_f64;
        let result = log_sum_exp(f64::NEG_INFINITY, b);
        assert!((result - b).abs() < 1e-12, "expected {b} got {result}");

        // log_sum_exp(a, -inf) == a
        let a = -2.0_f64;
        let result2 = log_sum_exp(a, f64::NEG_INFINITY);
        assert!((result2 - a).abs() < 1e-12, "expected {a} got {result2}");
    }

    #[test]
    fn test_log_sum_exp_equal_values() {
        // log_sum_exp(a, a) == a + ln(2)
        let a = -3.0_f64;
        let result = log_sum_exp(a, a);
        let expected = a + 2.0_f64.ln();
        assert!(
            (result - expected).abs() < 1e-10,
            "expected {expected} got {result}"
        );
    }

    #[test]
    fn test_log_sum_exp_greater_than_either_arg() {
        let (a, b) = (-2.0_f64, -5.0_f64);
        let result = log_sum_exp(a, b);
        assert!(result > a.max(b), "log_sum_exp must exceed both inputs");
        assert!(
            result < a.max(b) + 1.0,
            "log_sum_exp must not be unreasonably large"
        );
    }

    #[test]
    fn test_log_sum_exp_symmetry() {
        let a = -1.0_f64;
        let b = -4.0_f64;
        let r1 = log_sum_exp(a, b);
        let r2 = log_sum_exp(b, a);
        assert!((r1 - r2).abs() < 1e-12, "log_sum_exp must be symmetric");
    }
}
