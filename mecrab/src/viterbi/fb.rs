//! Forward-backward algorithm for marginal probability computation.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module contains the probabilistic forward-backward algorithm used to
//! compute per-node marginal probabilities over the lattice.  It is separated
//! from the deterministic Viterbi path-finding code in [`super`] to keep each
//! algorithm self-contained and within the 2000-line refactoring policy limit.

use std::collections::HashMap;

use crate::lattice::Lattice;
use crate::viterbi::analysis::{LatticeProbTable, NodeMarginal};
use crate::viterbi::ViterbiSolver;

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
    #[allow(clippy::too_many_lines)]
    pub fn forward_backward<'b>(
        &self,
        lattice: &'b Lattice<'b>,
    ) -> LatticeProbTable {
        let n = lattice.len();
        if n == 0 {
            return LatticeProbTable::default();
        }

        // ── Forward pass ──────────────────────────────────────────────────────
        // alpha[pos][node_idx] = log P(BOS → node ending at `pos`)
        //
        // For BOS (pos == 0): alpha = 0 (log 1), contributing -wcost/T = 0
        // because BOS has wcost = 0.
        //
        // For other nodes at pos `p` ending at byte `p - 1`:
        //   The predecessor position is `node.start + 1` (nodes ending at byte
        //   `node.start - 1`, i.e., right before this node begins).
        //   alpha[p][j] = log_sum_exp over predecessors i at prev_pos of
        //     alpha[prev_pos][i] + (-conn_cost(prev.right_id, node.left_id) / T)
        //                       + (-node.wcost / T)
        let mut alpha: Vec<Vec<f64>> = (0..n).map(|_| Vec::new()).collect();

        // Initialise BOS (position 0)
        {
            let bos_nodes = lattice.nodes_ending_at(0);
            alpha[0] = vec![0.0_f64; bos_nodes.len()];
        }

        for pos in 1..n {
            let nodes = lattice.nodes_ending_at(pos);
            let mut alpha_pos = vec![f64::NEG_INFINITY; nodes.len()];

            for (j, node) in nodes.iter().enumerate() {
                let wcost_contrib = -(node.wcost as f64) / TEMPERATURE;

                // Primary predecessor slot: nodes ending just before this node starts.
                // In the lattice layout, a node starting at byte `s` is preceded by
                // nodes stored at slot `s + 1` (nodes ending at byte `s`... wait,
                // nodes_at[p] stores nodes with end == p-1, so nodes ending at byte
                // `node.start` are at slot `node.start + 1`).
                // But the forward_pass uses `node.start + 1` when `node.start > 0`
                // or `0` when `node.start == 0`.
                let prev_pos = if node.start == 0 {
                    0_usize
                } else {
                    node.start + 1
                };

                if prev_pos < n {
                    let prev_nodes = lattice.nodes_ending_at(prev_pos);
                    for (i, prev_node) in prev_nodes.iter().enumerate() {
                        let conn = self
                            .dictionary
                            .connection_cost(prev_node.right_id, node.left_id)
                            as f64;
                        let arc_contrib = -conn / TEMPERATURE;
                        let score = alpha[prev_pos][i] + arc_contrib + wcost_contrib;
                        alpha_pos[j] = log_sum_exp(alpha_pos[j], score);
                    }
                }

                // Also scan earlier positions to handle nodes that span multiple
                // character slots (same logic as forward_pass).
                // We enumerate over the alpha slice from position 1..prev_pos to
                // satisfy clippy's needless_range_loop lint while keeping the index.
                let span_limit = prev_pos.min(n);
                let alpha_span = if span_limit > 1 {
                    &alpha[1..span_limit]
                } else {
                    &alpha[0..0]
                };
                for (offset, alpha_check) in alpha_span.iter().enumerate() {
                    let check_pos = 1 + offset;
                    let check_nodes = lattice.nodes_ending_at(check_pos);
                    for (i, prev_node) in check_nodes.iter().enumerate() {
                        // Only predecessors whose end byte matches our start byte
                        if prev_node.end == node.start {
                            let conn = self
                                .dictionary
                                .connection_cost(prev_node.right_id, node.left_id)
                                as f64;
                            let arc_contrib = -conn / TEMPERATURE;
                            let score = alpha_check.get(i).copied().unwrap_or(f64::NEG_INFINITY)
                                + arc_contrib
                                + wcost_contrib;
                            alpha_pos[j] = log_sum_exp(alpha_pos[j], score);
                        }
                    }
                }
            }

            alpha[pos] = alpha_pos;
        }

        // ── Backward pass ─────────────────────────────────────────────────────
        // beta[pos][node_idx] = log P(node → EOS)
        //
        // Initialise EOS (last position, n-1): beta = 0 (log 1).
        let mut beta: Vec<Vec<f64>> = (0..n).map(|_| Vec::new()).collect();

        {
            let eos_nodes = lattice.nodes_ending_at(n - 1);
            beta[n - 1] = vec![0.0_f64; eos_nodes.len()];
        }

        // For each position `pos` (in reverse order), compute beta[pos] by
        // scanning all successor positions `next_pos > pos`.
        // For each successor node at `next_pos`, if it can follow a node at `pos`
        // (same predecessor-slot logic as the forward pass), propagate backward.
        for pos in (0..n.saturating_sub(1)).rev() {
            let nodes_at_pos = lattice.nodes_ending_at(pos);
            let mut beta_pos = vec![f64::NEG_INFINITY; nodes_at_pos.len()];

            // Scan all possible successor positions.
            // We enumerate over the beta slice (skipping indices 0..=pos) so
            // that clippy's needless_range_loop lint is satisfied while still
            // having both the positional index and the beta row.
            let beta_slice_from_next = &beta[(pos + 1)..n];
            for (offset, beta_row) in beta_slice_from_next.iter().enumerate() {
                let next_pos = pos + 1 + offset;
                let next_nodes = lattice.nodes_ending_at(next_pos);
                for (j, next_node) in next_nodes.iter().enumerate() {
                    let next_wcost_contrib = -(next_node.wcost as f64) / TEMPERATURE;
                    let next_beta_j = beta_row.get(j).copied().unwrap_or(f64::NEG_INFINITY);
                    if next_beta_j == f64::NEG_INFINITY {
                        continue;
                    }

                    // The primary predecessor slot for next_node (mirrors forward_pass logic)
                    let expected_prev_pos = if next_node.start == 0 {
                        0_usize
                    } else {
                        next_node.start + 1
                    };

                    if expected_prev_pos == pos {
                        // Standard case: nodes at `pos` are direct predecessors
                        for (i, node) in nodes_at_pos.iter().enumerate() {
                            let conn = self
                                .dictionary
                                .connection_cost(node.right_id, next_node.left_id)
                                as f64;
                            let arc_contrib = -conn / TEMPERATURE;
                            let score = next_beta_j + arc_contrib + next_wcost_contrib;
                            beta_pos[i] = log_sum_exp(beta_pos[i], score);
                        }
                    } else if pos < expected_prev_pos {
                        // Span case: node at `pos` ends at next_node.start
                        for (i, node) in nodes_at_pos.iter().enumerate() {
                            if node.end == next_node.start {
                                let conn = self
                                    .dictionary
                                    .connection_cost(node.right_id, next_node.left_id)
                                    as f64;
                                let arc_contrib = -conn / TEMPERATURE;
                                let score = next_beta_j + arc_contrib + next_wcost_contrib;
                                beta_pos[i] = log_sum_exp(beta_pos[i], score);
                            }
                        }
                    }
                }
            }

            beta[pos] = beta_pos;
        }

        // ── Partition function Z ──────────────────────────────────────────────
        // Z = sum over all BOS entries of alpha[0][i] * beta[0][i]
        //   = sum over i of (alpha[0][i] + beta[0][i]) in log-domain
        let bos_nodes = lattice.nodes_ending_at(0);
        let log_z = bos_nodes
            .iter()
            .enumerate()
            .fold(f64::NEG_INFINITY, |acc, (i, _)| {
                let a = alpha[0].get(i).copied().unwrap_or(f64::NEG_INFINITY);
                let b = beta[0].get(i).copied().unwrap_or(f64::NEG_INFINITY);
                if a == f64::NEG_INFINITY || b == f64::NEG_INFINITY {
                    acc
                } else {
                    log_sum_exp(acc, a + b)
                }
            });

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
    #[allow(clippy::too_many_lines)]
    pub fn compute_edge_expected_counts<'b>(
        &self,
        lattice: &'b Lattice<'b>,
    ) -> HashMap<(u16, u16), f64> {
        use crate::viterbi::train::edge_expected_counts_from_fb;

        // Rerun forward-backward to build internal alpha/beta.
        // We rebuild them here rather than exposing them from forward_backward to
        // keep LatticeProbTable simple (it only needs marginals for the public API).
        let n = lattice.len();
        if n == 0 {
            return HashMap::new();
        }

        // ── Forward pass (mirrors forward_backward exactly) ───────────────────
        let mut alpha: Vec<Vec<f64>> = (0..n).map(|_| Vec::new()).collect();
        {
            let bos_nodes = lattice.nodes_ending_at(0);
            alpha[0] = vec![0.0_f64; bos_nodes.len()];
        }

        for pos in 1..n {
            let nodes = lattice.nodes_ending_at(pos);
            let mut alpha_pos = vec![f64::NEG_INFINITY; nodes.len()];

            for (j, node) in nodes.iter().enumerate() {
                let wcost_contrib = -(node.wcost as f64) / TEMPERATURE;
                let prev_pos = if node.start == 0 { 0_usize } else { node.start + 1 };

                if prev_pos < n {
                    let prev_nodes = lattice.nodes_ending_at(prev_pos);
                    for (i, prev_node) in prev_nodes.iter().enumerate() {
                        let conn = self
                            .dictionary
                            .connection_cost(prev_node.right_id, node.left_id)
                            as f64;
                        let score = alpha[prev_pos][i] + (-conn / TEMPERATURE) + wcost_contrib;
                        alpha_pos[j] = log_sum_exp(alpha_pos[j], score);
                    }
                }

                let span_limit = prev_pos.min(n);
                let alpha_span = if span_limit > 1 {
                    &alpha[1..span_limit]
                } else {
                    &alpha[0..0]
                };
                for (offset, alpha_check) in alpha_span.iter().enumerate() {
                    let check_pos = 1 + offset;
                    let check_nodes = lattice.nodes_ending_at(check_pos);
                    for (i, prev_node) in check_nodes.iter().enumerate() {
                        if prev_node.end == node.start {
                            let conn = self
                                .dictionary
                                .connection_cost(prev_node.right_id, node.left_id)
                                as f64;
                            let score = alpha_check
                                .get(i)
                                .copied()
                                .unwrap_or(f64::NEG_INFINITY)
                                + (-conn / TEMPERATURE)
                                + wcost_contrib;
                            alpha_pos[j] = log_sum_exp(alpha_pos[j], score);
                        }
                    }
                }
            }

            alpha[pos] = alpha_pos;
        }

        // ── Backward pass (mirrors forward_backward exactly) ──────────────────
        let mut beta: Vec<Vec<f64>> = (0..n).map(|_| Vec::new()).collect();
        {
            let eos_nodes = lattice.nodes_ending_at(n - 1);
            beta[n - 1] = vec![0.0_f64; eos_nodes.len()];
        }

        for pos in (0..n.saturating_sub(1)).rev() {
            let nodes_at_pos = lattice.nodes_ending_at(pos);
            let mut beta_pos = vec![f64::NEG_INFINITY; nodes_at_pos.len()];

            let beta_slice_from_next = &beta[(pos + 1)..n];
            for (offset, beta_row) in beta_slice_from_next.iter().enumerate() {
                let next_pos = pos + 1 + offset;
                let next_nodes = lattice.nodes_ending_at(next_pos);
                for (j, next_node) in next_nodes.iter().enumerate() {
                    let next_wcost_contrib = -(next_node.wcost as f64) / TEMPERATURE;
                    let next_beta_j =
                        beta_row.get(j).copied().unwrap_or(f64::NEG_INFINITY);
                    if next_beta_j == f64::NEG_INFINITY {
                        continue;
                    }

                    let expected_prev_pos = if next_node.start == 0 {
                        0_usize
                    } else {
                        next_node.start + 1
                    };

                    if expected_prev_pos == pos {
                        for (i, node) in nodes_at_pos.iter().enumerate() {
                            let conn = self
                                .dictionary
                                .connection_cost(node.right_id, next_node.left_id)
                                as f64;
                            let score = next_beta_j + (-conn / TEMPERATURE) + next_wcost_contrib;
                            beta_pos[i] = log_sum_exp(beta_pos[i], score);
                        }
                    } else if pos < expected_prev_pos {
                        for (i, node) in nodes_at_pos.iter().enumerate() {
                            if node.end == next_node.start {
                                let conn = self
                                    .dictionary
                                    .connection_cost(node.right_id, next_node.left_id)
                                    as f64;
                                let score =
                                    next_beta_j + (-conn / TEMPERATURE) + next_wcost_contrib;
                                beta_pos[i] = log_sum_exp(beta_pos[i], score);
                            }
                        }
                    }
                }
            }

            beta[pos] = beta_pos;
        }

        // ── Partition function ────────────────────────────────────────────────
        let bos_nodes = lattice.nodes_ending_at(0);
        let log_z = bos_nodes
            .iter()
            .enumerate()
            .fold(f64::NEG_INFINITY, |acc, (i, _)| {
                let a = alpha[0].get(i).copied().unwrap_or(f64::NEG_INFINITY);
                let b = beta[0].get(i).copied().unwrap_or(f64::NEG_INFINITY);
                if a == f64::NEG_INFINITY || b == f64::NEG_INFINITY {
                    acc
                } else {
                    log_sum_exp(acc, a + b)
                }
            });

        // ── Delegate to train::edge_expected_counts_from_fb ──────────────────
        edge_expected_counts_from_fb(lattice, &alpha, &beta, log_z, self.dictionary)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::log_sum_exp;

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
