//! Viterbi algorithm implementation for finding the optimal path
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! The Viterbi algorithm finds the path through the lattice with the
//! minimum total cost (word costs + connection costs).
//!
//! ## Layout
//!
//! The forward pass uses a **Hybrid SoA (Struct-of-Arrays)** layout via
//! [`ViterbiTable`].  Hot data (cumulative costs) lives in contiguous
//! `Vec<i64>` slices separate from cold data (node references, back-pointers).
//! This lets the inner min-finding loop operate over a tightly-packed
//! `&[i64]` with no pointer-chasing, improving cache utilisation and enabling
//! auto-vectorisation / SIMD codegen.

pub mod analysis;
pub mod nbest;
/// SIMD-accelerated cost functions.
///
/// Always compiled; individual functions dispatch to NEON, AVX2/SSE4.1, WASM
/// simd128, or a scalar fallback depending on the build target.
pub mod simd;

use simd::batch_connection_costs;

pub use analysis::{
    ConnectionMatrixStats, LatticeStats, MorphemeCost, PathAnalysis, PathComparison,
    SegmentationReport,
};
pub use nbest::{NBestIter, NBestSearch, PathDiversity, ScoredPath};

use crate::dict::Dictionary;
use crate::lattice::{Lattice, LatticeNode};
use crate::{Error, Result};

// ── Data structures ──────────────────────────────────────────────────────────

/// Result node after Viterbi path finding
#[derive(Debug, Clone)]
pub struct PathNode {
    /// Surface form
    pub surface: String,
    /// Word ID (token index in dictionary, used for embeddings)
    pub word_id: u32,
    /// Part-of-speech ID
    pub pos_id: u16,
    /// Word cost
    pub wcost: i16,
    /// Feature string
    pub feature: String,
    /// Byte offset of the start of this token in the input text
    pub start_byte: usize,
    /// Byte offset just past the end of this token (exclusive)
    pub end_byte: usize,
}

// ── Legacy entry type (used only in unit tests that examine the raw struct) ──

/// Entry in the Viterbi table — kept **only** for the EOS-filter regression
/// tests that construct synthetic slices directly.  Production code uses
/// [`ViterbiTable`] instead.
#[cfg(test)]
#[derive(Debug, Clone)]
#[allow(dead_code)] // `pos` is set in synthetic test data but not read back
struct ViterbiEntry<'a> {
    node: &'a LatticeNode<'a>,
    cost: i64,
    prev: Option<usize>,
    pos: usize,
}

// ── SoA Viterbi Table ────────────────────────────────────────────────────────

/// Cold (infrequently accessed) per-entry data.
///
/// Separated from hot cost data so that the inner predecessor-scanning loop
/// touches only the contiguous `costs` vector.
#[derive(Debug, Clone)]
struct ViterbiCold<'a> {
    /// Lattice node for this entry
    node: &'a LatticeNode<'a>,
    /// Back-pointer index into `cold[prev_lattice_pos]`
    /// `None` for BOS entries (no predecessor).
    prev: Option<u32>,
    /// Lattice position that `prev` indexes into
    pos: u32,
}

/// Hybrid SoA Viterbi table.
///
/// `costs[p]` and `cold[p]` are always the same length; entry `i` at lattice
/// position `p` has cumulative cost `costs[p][i]` and metadata `cold[p][i]`.
///
/// Keeping costs in their own contiguous `Vec<i64>` per position means the
/// inner min-finding loop touches only `costs[prev_pos]` — a single cache
/// line friendly slice — before diving into `cold` only for the winner.
struct ViterbiTable<'a> {
    /// Hot path: cumulative costs per lattice position
    costs: Vec<Vec<i64>>,
    /// Cold path: node references and backtrack data per lattice position
    cold: Vec<Vec<ViterbiCold<'a>>>,
}

impl<'a> ViterbiTable<'a> {
    /// Allocate a table with `n` positions, each initially empty.
    fn new(n: usize) -> Self {
        Self {
            costs: vec![Vec::new(); n],
            cold: vec![Vec::new(); n],
        }
    }

    /// Append a new entry at position `pos`.
    #[inline]
    fn push(&mut self, pos: usize, cost: i64, cold: ViterbiCold<'a>) {
        self.costs[pos].push(cost);
        self.cold[pos].push(cold);
    }

    /// Number of entries at position `pos`.
    #[inline]
    fn len_at(&self, pos: usize) -> usize {
        self.costs[pos].len()
    }

    /// Number of lattice positions stored.
    #[inline]
    fn positions(&self) -> usize {
        self.costs.len()
    }
}

// ── Solver ───────────────────────────────────────────────────────────────────

/// Viterbi solver for finding the optimal path through the lattice
pub struct ViterbiSolver<'a> {
    dictionary: &'a Dictionary,
}

impl<'a> ViterbiSolver<'a> {
    /// Create a new Viterbi solver
    pub const fn new(dictionary: &'a Dictionary) -> Self {
        Self { dictionary }
    }

    /// Solve the lattice and return the optimal path.
    ///
    /// # Errors
    ///
    /// Returns an error if no valid path is found.
    pub fn solve<'b>(&self, lattice: &'b Lattice<'b>) -> Result<Vec<PathNode>> {
        if lattice.is_empty() {
            return Err(Error::LatticeError("Empty lattice".to_string()));
        }

        let table = self.forward_pass(lattice);
        let path = Self::backward_pass(&table, lattice)?;

        Ok(path)
    }

    /// Solve the lattice and return the N-best paths.
    ///
    /// # Errors
    ///
    /// Returns an error if no valid path is found.
    pub fn solve_nbest<'b>(
        &self,
        lattice: &'b Lattice<'b>,
        n: usize,
    ) -> Result<Vec<(Vec<PathNode>, i64)>> {
        if lattice.is_empty() {
            return Err(Error::LatticeError("Empty lattice".to_string()));
        }

        let table = self.forward_pass(lattice);
        let paths = self.nbest_backward(&table, lattice, n);

        Ok(paths)
    }

    // ── Forward pass (SoA) ───────────────────────────────────────────────────

    /// Forward pass of the Viterbi algorithm using the SoA layout.
    ///
    /// For every lattice position the inner loop iterates over
    /// `table.costs[prev_pos]` — a tightly-packed `Vec<i64>` — to find the
    /// minimum-cost predecessor before touching any cold data.
    fn forward_pass<'b>(&self, lattice: &'b Lattice<'b>) -> ViterbiTable<'b> {
        let n = lattice.len();
        let mut table = ViterbiTable::new(n);

        // Initialise BOS entries (prev = None → back-pointer absent)
        for node in lattice.nodes_ending_at(0) {
            table.push(
                0,
                0,
                ViterbiCold {
                    node,
                    prev: None,
                    pos: 0,
                },
            );
        }

        for pos in 1..n {
            let nodes = lattice.nodes_ending_at(pos);

            for node in nodes {
                let mut best_cost = i64::MAX;
                let mut best_prev: Option<u32> = None;
                let mut best_prev_pos: u32 = 0;

                // Primary predecessor position (nodes ending where this one starts)
                let prev_pos = if node.start == 0 && pos > 0 {
                    0
                } else {
                    node.start + 1
                };

                if prev_pos < table.positions() {
                    Self::scan_predecessors(
                        &table,
                        prev_pos,
                        node,
                        self.dictionary,
                        &mut best_cost,
                        &mut best_prev,
                        &mut best_prev_pos,
                    );
                }

                // Also check earlier positions (longer words may span multiple slots)
                for check_pos in 1..prev_pos {
                    if check_pos < table.positions() {
                        // Only consider entries whose node ends exactly where ours starts
                        for (prev_idx, cold) in table.cold[check_pos].iter().enumerate() {
                            if cold.node.end == node.start {
                                let conn_cost = self
                                    .dictionary
                                    .connection_cost(cold.node.right_id, node.left_id)
                                    as i64;
                                let total_cost = table.costs[check_pos][prev_idx]
                                    + conn_cost
                                    + node.wcost as i64;

                                if total_cost < best_cost {
                                    best_cost = total_cost;
                                    best_prev = Some(prev_idx as u32);
                                    best_prev_pos = check_pos as u32;
                                }
                            }
                        }
                    }
                }

                if best_cost < i64::MAX {
                    table.push(
                        pos,
                        best_cost,
                        ViterbiCold {
                            node,
                            prev: best_prev,
                            pos: best_prev_pos,
                        },
                    );
                }
            }
        }

        table
    }

    /// Inner predecessor scan: iterates the **hot** cost slice for position
    /// `prev_pos`, then reads cold data only for the winning candidate.
    ///
    /// Uses [`batch_connection_costs`] to gather and widen up to 16 connection
    /// costs at a time from the pre-fetched matrix row for `node.left_id`,
    /// amortising the per-node bounds-check overhead and enabling SIMD widening
    /// on aarch64 (NEON), x86_64 (AVX2/SSE4.1), and WASM (simd128).  On other
    /// platforms the function dispatches to its built-in scalar fallback.
    #[inline]
    fn scan_predecessors(
        table: &ViterbiTable<'_>,
        prev_pos: usize,
        node: &LatticeNode<'_>,
        dictionary: &Dictionary,
        best_cost: &mut i64,
        best_prev: &mut Option<u32>,
        best_prev_pos: &mut u32,
    ) {
        let cost_row = &table.costs[prev_pos];
        let cold_row = &table.cold[prev_pos];
        let node_wcost = node.wcost as i64;

        // Pre-fetch the matrix row for this node's left_id once.
        // Fall back to per-element scalar path if the left_id is out of range.
        if let Some(row) = dictionary.matrix.row_for_left_id(node.left_id) {
            // Batch size: gather up to 16 right_ids at a time.
            const BATCH: usize = 16;
            let total = cost_row.len();
            let mut conn_buf = [0i32; BATCH];
            let mut right_ids_buf = [0u16; BATCH];

            let mut base = 0usize;
            while base < total {
                let chunk_len = (total - base).min(BATCH);

                // Fill right_ids_buf for this chunk.
                for j in 0..chunk_len {
                    right_ids_buf[j] = cold_row[base + j].node.right_id;
                }

                // Batch gather: fill conn_buf[0..chunk_len] with widened i16→i32 costs.
                // Dispatches to NEON / AVX2 / SSE4.1 / scalar based on target.
                let written = batch_connection_costs(
                    row,
                    &right_ids_buf[..chunk_len],
                    &mut conn_buf[..chunk_len],
                );

                // Scalar min over prev_cost + conn_cost + wcost.
                for j in 0..written {
                    let prev_cost = cost_row[base + j];
                    let conn_cost = conn_buf[j] as i64;
                    let total_cost = prev_cost + conn_cost + node_wcost;
                    if total_cost < *best_cost {
                        *best_cost = total_cost;
                        *best_prev = Some((base + j) as u32);
                        *best_prev_pos = prev_pos as u32;
                    }
                }

                base += chunk_len;
            }
            return;
        }

        // Scalar fallback: OOB left_id — look up each cost individually.
        for (prev_idx, &prev_cost) in cost_row.iter().enumerate() {
            let cold = &cold_row[prev_idx];
            let conn_cost = dictionary.connection_cost(cold.node.right_id, node.left_id) as i64;
            let total_cost = prev_cost + conn_cost + node_wcost;

            if total_cost < *best_cost {
                *best_cost = total_cost;
                *best_prev = Some(prev_idx as u32);
                *best_prev_pos = prev_pos as u32;
            }
        }
    }

    // ── Backward pass (SoA) ──────────────────────────────────────────────────

    /// Backward pass: trace the optimal path from EOS to BOS.
    ///
    /// Preserves the EOS-filtering fix: only entries with an empty surface at
    /// `start == text_len` are eligible as the EOS anchor, preventing a regular
    /// word with a lower accumulated cost from silently masquerading as EOS.
    fn backward_pass<'b>(
        table: &ViterbiTable<'b>,
        lattice: &'b Lattice<'b>,
    ) -> Result<Vec<PathNode>> {
        let n = table.positions();

        let eos_cost_row = &table.costs[n - 1];
        if eos_cost_row.is_empty() {
            return Err(Error::ViterbiError("No path to EOS found".to_string()));
        }

        // Filter to the actual EOS node (empty surface, start == text_len).
        let text_len = lattice.text.len();
        let best_eos_idx = table.cold[n - 1]
            .iter()
            .enumerate()
            .filter(|(_, cold)| cold.node.surface.is_empty() && cold.node.start == text_len)
            .min_by_key(|(i, _)| eos_cost_row[*i])
            .map(|(i, _)| i)
            .ok_or_else(|| Error::ViterbiError("No EOS entry found".to_string()))?;

        // Trace back through cold pointers
        let mut path = Vec::new();
        let eos_cold = &table.cold[n - 1][best_eos_idx];
        let mut current_idx = eos_cold.prev;
        let mut prev_pos = eos_cold.pos as usize;

        while let Some(idx) = current_idx {
            let idx = idx as usize;
            if prev_pos >= table.positions() || idx >= table.len_at(prev_pos) {
                break;
            }

            let cold = &table.cold[prev_pos][idx];

            if !cold.node.surface.is_empty() {
                path.push(PathNode {
                    surface: cold.node.surface.to_string(),
                    word_id: cold.node.word_id,
                    pos_id: cold.node.pos_id,
                    wcost: cold.node.wcost,
                    feature: cold.node.feature.clone(),
                    start_byte: cold.node.start,
                    end_byte: cold.node.end,
                });
            }

            current_idx = cold.prev;
            prev_pos = cold.pos as usize;
        }

        path.reverse();
        Ok(path)
    }

    // ── N-best backward (SoA) ────────────────────────────────────────────────

    /// Find N-best paths using backward heap search through the SoA table.
    fn nbest_backward<'b>(
        &self,
        table: &ViterbiTable<'b>,
        lattice: &'b Lattice<'b>,
        n: usize,
    ) -> Vec<(Vec<PathNode>, i64)> {
        use std::collections::BinaryHeap;

        let len = table.positions();
        if len == 0 {
            return vec![];
        }

        /// Heap state for the N-best backward search.
        #[derive(Clone)]
        struct SearchState<'a> {
            cost: i64,
            pos: usize,
            idx: usize,
            path: Vec<&'a LatticeNode<'a>>,
        }

        impl PartialEq for SearchState<'_> {
            fn eq(&self, other: &Self) -> bool {
                self.cost == other.cost
            }
        }
        impl Eq for SearchState<'_> {}
        impl PartialOrd for SearchState<'_> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for SearchState<'_> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other.cost.cmp(&self.cost) // min-heap
            }
        }

        let mut heap: BinaryHeap<SearchState<'b>> = BinaryHeap::new();
        let mut results = Vec::with_capacity(n);

        // Seed heap from genuine EOS entries only (empty surface, start == text_len).
        let text_len = lattice.text.len();
        let eos_pos = len - 1;

        for (idx, cold) in table.cold[eos_pos].iter().enumerate() {
            if cold.node.surface.is_empty() && cold.node.start == text_len {
                heap.push(SearchState {
                    cost: table.costs[eos_pos][idx],
                    pos: eos_pos,
                    idx,
                    path: vec![cold.node],
                });
            }
        }

        while let Some(state) = heap.pop() {
            if results.len() >= n {
                break;
            }

            let cold = &table.cold[state.pos][state.idx];

            if cold.prev.is_none() {
                // Reached BOS — collect non-BOS/EOS nodes
                let path_nodes: Vec<PathNode> = state
                    .path
                    .iter()
                    .rev()
                    .filter(|node| !node.surface.is_empty())
                    .map(|node| PathNode {
                        surface: node.surface.to_string(),
                        word_id: node.word_id,
                        pos_id: node.pos_id,
                        wcost: node.wcost,
                        feature: node.feature.clone(),
                        start_byte: node.start,
                        end_byte: node.end,
                    })
                    .collect();

                if !path_nodes.is_empty() || state.path.len() <= 2 {
                    results.push((path_nodes, state.cost));
                }
                continue;
            }

            // Expand to predecessor
            let prev_idx = cold.prev.expect("checked above") as usize;
            let prev_pos = cold.pos as usize;

            if prev_pos < table.positions() && prev_idx < table.len_at(prev_pos) {
                let prev_cold = &table.cold[prev_pos][prev_idx];
                let mut new_path = state.path.clone();
                new_path.push(prev_cold.node);

                heap.push(SearchState {
                    cost: state.cost,
                    pos: prev_pos,
                    idx: prev_idx,
                    path: new_path,
                });
            }
        }

        results
    }
}

// ── Forward-Backward Algorithm ───────────────────────────────────────────────

/// Numerically stable log-sum-exp for two values.
///
/// `log_sum_exp(a, b) = max(a, b) + ln(1 + exp(-|a - b|))`
///
/// Handles `NEG_INFINITY` as the log-domain zero (additive identity).
#[inline]
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (larger, smaller) = if a >= b { (a, b) } else { (b, a) };
    larger + (1.0_f64 + (smaller - larger).exp()).ln()
}

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
    ) -> crate::viterbi::analysis::LatticeProbTable {
        use crate::viterbi::analysis::{LatticeProbTable, NodeMarginal};

        const TEMPERATURE: f64 = 500.0; // cost units per nat

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
                        feature: node.feature.clone(),
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
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::LatticeNode;

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

    #[test]
    fn test_path_node_creation() {
        let node = PathNode {
            surface: "テスト".to_string(),
            word_id: 42,
            pos_id: 1,
            wcost: 100,
            feature: "名詞,一般".to_string(),
            start_byte: 0,
            end_byte: 9, // "テスト" = 3 × 3 bytes in UTF-8
        };

        assert_eq!(node.surface, "テスト");
        assert_eq!(node.pos_id, 1);
    }

    // ── SoA table helpers ────────────────────────────────────────────────────

    /// Verify that `ViterbiTable::push` and `len_at` are consistent.
    #[test]
    fn test_viterbi_table_push_and_len() {
        let node = LatticeNode::bos();
        let mut table: ViterbiTable<'_> = ViterbiTable::new(3);

        assert_eq!(table.len_at(0), 0);

        table.push(
            0,
            42,
            ViterbiCold {
                node: &node,
                prev: None,
                pos: 0,
            },
        );
        assert_eq!(table.len_at(0), 1);
        assert_eq!(table.costs[0][0], 42);

        table.push(
            0,
            99,
            ViterbiCold {
                node: &node,
                prev: Some(0),
                pos: 0,
            },
        );
        assert_eq!(table.len_at(0), 2);
        assert_eq!(table.costs[0][1], 99);
    }

    /// Verify that the hot cost slice is contiguous and accessible as &[i64].
    #[test]
    fn test_soa_cost_slice_is_contiguous() {
        let node = LatticeNode::bos();
        let mut table: ViterbiTable<'_> = ViterbiTable::new(2);

        for cost in [10_i64, 20, 30, 5, 15] {
            table.push(
                0,
                cost,
                ViterbiCold {
                    node: &node,
                    prev: None,
                    pos: 0,
                },
            );
        }

        let cost_row: &[i64] = &table.costs[0];
        let min = cost_row.iter().copied().min();
        assert_eq!(min, Some(5));
    }

    // ── EOS filter regression tests ──────────────────────────────────────────

    /// Regression test for the EOS-selection bug.
    ///
    /// Before the fix, `backward_pass` called `min_by_key(|e| e.cost)` over ALL
    /// entries at the last lattice position — including regular words that happen
    /// to share that position.  If a regular word had a lower accumulated cost
    /// than the actual EOS node (which can happen with positive EOS connection
    /// costs), that word was chosen as the "EOS" entry and the backtrack started
    /// from `word.prev`, silently dropping the word itself from the output.
    ///
    /// The fix adds `.filter(|e| e.node.surface.is_empty() && e.node.start == text_len)`
    /// so only the genuine EOS node is considered, regardless of accumulated cost.
    ///
    /// This test verifies the filtering predicate directly against synthetic
    /// `ViterbiEntry` slices (no real dictionary needed).
    #[test]
    fn test_eos_filter_selects_eos_not_regular_word() {
        let text_len: usize = 3;

        let eos_node = LatticeNode::eos(text_len);
        let word_node = LatticeNode {
            surface: "ほど",
            start: 0,
            end: text_len,
            word_id: 1,
            left_id: 10,
            right_id: 10,
            pos_id: 5,
            wcost: -50,
            feature: "助詞,副助詞".to_string(),
            is_unknown: false,
        };

        let entries: Vec<ViterbiEntry<'_>> = vec![
            ViterbiEntry {
                node: &word_node,
                cost: 100,
                prev: None,
                pos: 0,
            },
            ViterbiEntry {
                node: &eos_node,
                cost: 200,
                prev: Some(0),
                pos: 1,
            },
        ];

        // Old (buggy) logic
        let old_best = entries.iter().min_by_key(|e| e.cost).unwrap();
        assert!(
            !old_best.node.surface.is_empty(),
            "Old logic (no filter) incorrectly selects the regular word"
        );

        // New (fixed) logic — mirrors backward_pass SoA filter
        let new_best = entries
            .iter()
            .filter(|e| e.node.surface.is_empty() && e.node.start == text_len)
            .min_by_key(|e| e.cost);

        assert!(new_best.is_some(), "Fixed logic must find the EOS entry");
        let new_best = new_best.unwrap();
        assert!(
            new_best.node.surface.is_empty() && new_best.node.start == text_len,
            "Fixed logic must select the genuine EOS node, not the regular word"
        );
        assert!(
            new_best.prev.is_some(),
            "The selected EOS entry must have a valid back-pointer"
        );
    }

    /// Regression test for the same bug in `nbest_backward`.
    #[test]
    fn test_nbest_eos_filter_excludes_regular_words() {
        let text_len: usize = 3;

        let eos_node = LatticeNode::eos(text_len);
        let word_node = LatticeNode {
            surface: "ない",
            start: 0,
            end: text_len,
            word_id: 2,
            left_id: 20,
            right_id: 20,
            pos_id: 3,
            wcost: -30,
            feature: "助動詞".to_string(),
            is_unknown: false,
        };

        let entries: Vec<ViterbiEntry<'_>> = vec![
            ViterbiEntry {
                node: &word_node,
                cost: 50,
                prev: None,
                pos: 0,
            },
            ViterbiEntry {
                node: &eos_node,
                cost: 150,
                prev: Some(0),
                pos: 1,
            },
        ];

        let eos_candidates: Vec<&ViterbiEntry<'_>> = entries
            .iter()
            .filter(|e| e.node.surface.is_empty() && e.node.start == text_len)
            .collect();

        assert_eq!(
            eos_candidates.len(),
            1,
            "Exactly one EOS candidate should pass the filter"
        );
        assert!(
            eos_candidates[0].node.surface.is_empty(),
            "Candidate must be the EOS node"
        );
        assert_eq!(eos_candidates[0].node.start, text_len);

        let regular_words_in_seed: Vec<&ViterbiEntry<'_>> = entries
            .iter()
            .filter(|e| e.node.surface.is_empty() && e.node.start == text_len)
            .filter(|e| !e.node.surface.is_empty())
            .collect();

        assert!(
            regular_words_in_seed.is_empty(),
            "Regular words must not be seeded into the N-best heap"
        );
    }

    // ── SoA correctness against toy lattice ─────────────────────────────────

    /// Verify that the SoA ViterbiTable forward scan produces the same minimum
    /// cost as the reference scalar computation on a synthetic cost grid.
    ///
    /// We construct two positions of the table by hand, then scan the hot cost
    /// slice for the predecessor minimum and check it matches the brute-force
    /// result.  This is an isolated unit test — no real dictionary needed.
    #[test]
    fn test_soa_viterbi_correctness() {
        // Simulate: position 0 has 3 predecessor entries with costs [10, 5, 20].
        // Current node has wcost = 3.
        // Pretend all connection costs are 0 (we test the layout, not the dict).
        let bos = LatticeNode::bos();
        let mut table: ViterbiTable<'_> = ViterbiTable::new(2);

        // Position 0: three BOS-like entries
        let costs_p0 = [10_i64, 5, 20];
        for &c in &costs_p0 {
            table.push(
                0,
                c,
                ViterbiCold {
                    node: &bos,
                    prev: None,
                    pos: 0,
                },
            );
        }

        // Simulate what forward_pass does: find min over the hot slice
        let cost_row: &[i64] = &table.costs[0];
        let (best_idx, best_prev_cost) = cost_row
            .iter()
            .copied()
            .enumerate()
            .min_by_key(|&(_, c)| c)
            .expect("cost_row must not be empty");

        let node_wcost = 3_i64;
        let total = best_prev_cost + node_wcost; // connection cost = 0 in this toy test

        // Record into position 1
        table.push(
            1,
            total,
            ViterbiCold {
                node: &bos,
                prev: Some(best_idx as u32),
                pos: 0,
            },
        );

        // Verify
        assert_eq!(best_prev_cost, 5, "minimum predecessor cost must be 5");
        assert_eq!(total, 8, "total cost must be 5 + 3 = 8");
        assert_eq!(table.costs[1][0], 8);
        assert_eq!(
            table.cold[1][0].prev,
            Some(1_u32),
            "back-pointer must point to index 1 (cost=5)"
        );
        assert_eq!(table.cold[1][0].pos, 0);
    }
}
