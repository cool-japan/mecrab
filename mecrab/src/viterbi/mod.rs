//! Viterbi algorithm implementation for finding the optimal path
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! The Viterbi algorithm finds the path through the lattice with the
//! minimum total cost (word costs + connection costs).
//!
//! ## Layout
//!
//! The forward pass uses a **CSR (Compressed Sparse Row)** layout via
//! [`ViterbiTableCsr`].  A single flat allocation stores all entries across
//! positions; `offsets[pos]..offsets[pos+1]` indexes into the flat data arrays.
//! Hot data (cumulative costs, right-context IDs) lives contiguously separate
//! from cold data (node references, back-pointers), improving cache utilisation
//! and enabling auto-vectorisation / SIMD codegen in the inner loop.

pub mod analysis;
mod fb;
pub mod nbest;
pub mod train;
/// SIMD-accelerated cost functions.
///
/// Always compiled; individual functions dispatch to NEON, AVX2/SSE4.1, WASM
/// simd128, or a scalar fallback depending on the build target.
pub mod simd;

use simd::{batch_connection_costs, batch_min_argmin_i64};

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
/// [`ViterbiTableCsr`] instead.
#[cfg(test)]
#[derive(Debug, Clone)]
#[allow(dead_code)] // `pos` is set in synthetic test data but not read back
struct ViterbiEntry<'a> {
    node: &'a LatticeNode<'a>,
    cost: i64,
    prev: Option<usize>,
    pos: usize,
}

// ── Cold per-entry data ───────────────────────────────────────────────────────

/// Cold (infrequently accessed) per-entry data.
///
/// Separated from hot cost data so that the inner predecessor-scanning loop
/// touches only the contiguous `costs_data` vector.
#[derive(Debug, Clone)]
struct ViterbiCold<'a> {
    /// Lattice node for this entry
    node: &'a LatticeNode<'a>,
    /// Back-pointer index into the entries at `prev_lattice_pos`
    /// `None` for BOS entries (no predecessor).
    prev: Option<u32>,
    /// Lattice position that `prev` indexes into
    pos: u32,
}

// ── CSR Viterbi Table ─────────────────────────────────────────────────────────

/// CSR (Compressed Sparse Row) Viterbi table.
///
/// All entries across all lattice positions are stored in three flat arrays.
/// `offsets[pos]..offsets[pos+1]` identifies the slice for position `pos`.
///
/// Hot data (`costs_data`, `right_ids_data`) and cold data (`cold_data`) are
/// stored in separate arrays so the SIMD inner loop touches only the hot slice
/// without any pointer-chasing through cold structs.
///
/// # Invariants
///
/// - `offsets.len() == n + 1`
/// - `offsets[0] == 0`
/// - `offsets[n] == costs_data.len()` (after sealing all positions)
/// - `costs_data`, `right_ids_data`, `cold_data` are always the same length
/// - Positions must be filled and sealed in strictly increasing order
struct ViterbiTableCsr<'a> {
    /// CSR row offsets: `offsets[pos]..offsets[pos+1]` gives the entry range.
    /// Length is `n + 1`.
    offsets: Vec<u32>,
    /// Flat hot array: cumulative costs for every entry across all positions.
    costs_data: Vec<i64>,
    /// Flat hot array: right-context IDs, parallel to `costs_data`.
    right_ids_data: Vec<u16>,
    /// Flat cold array: node refs and backtrack data, parallel to `costs_data`.
    cold_data: Vec<ViterbiCold<'a>>,
    /// Number of lattice positions (not counting the extra sentinel offset).
    n: usize,
    /// The position currently being built (not yet sealed).
    ///
    /// `offsets[current_pos]` is set; `offsets[current_pos + 1]` is not yet
    /// written.  Accessing `costs_at(current_pos)` returns entries pushed so
    /// far via the open-ended slice `costs_data[offsets[current_pos]..]`.
    ///
    /// This replicates the original `Vec<Vec<_>>` behaviour where the inner
    /// Vec grew during the same outer-loop iteration — in particular it allows
    /// EOS to scan real-word predecessors that were just pushed to the same
    /// lattice position in the same `for node in nodes` pass.
    current_pos: usize,
}

impl<'a> ViterbiTableCsr<'a> {
    /// Allocate a CSR table for `n` lattice positions.
    ///
    /// `offsets` is initialised to `[0; n+1]`; all data arrays are empty.
    /// Callers should push entries and call [`seal_position`](Self::seal_position)
    /// after each position.
    fn new(n: usize) -> Self {
        Self {
            offsets: vec![0u32; n + 1],
            costs_data: Vec::new(),
            right_ids_data: Vec::new(),
            cold_data: Vec::new(),
            n,
            current_pos: 0,
        }
    }

    /// Append a new entry at position `pos`.
    ///
    /// Positions must be filled in strictly non-decreasing order.  Call
    /// [`seal_position`](Self::seal_position) after all entries for `pos` have
    /// been pushed.
    #[inline]
    fn push(&mut self, pos: usize, cost: i64, cold: ViterbiCold<'a>) {
        debug_assert_eq!(pos, self.current_pos, "push: pos {pos} != current_pos {}", self.current_pos);
        self.right_ids_data.push(cold.node.right_id);
        self.costs_data.push(cost);
        self.cold_data.push(cold);
    }

    /// Seal position `pos`: record the current end of the data arrays as
    /// `offsets[pos+1]` and advance `current_pos`.
    ///
    /// Must be called exactly once per position, after all `push(pos, …)` calls
    /// and before any `push(pos+1, …)` calls.
    #[inline]
    fn seal_position(&mut self, pos: usize) {
        debug_assert_eq!(pos, self.current_pos, "seal_position: pos {pos} != current_pos {}", self.current_pos);
        debug_assert!(pos < self.n, "seal_position: pos {pos} >= n {}", self.n);
        self.offsets[pos + 1] = self.costs_data.len() as u32;
        self.current_pos = pos + 1;
    }

    /// Cumulative cost slice for position `pos`.
    ///
    /// If `pos == current_pos` (the position currently being built), returns
    /// all entries pushed so far (open-ended; `offsets[pos+1]` not yet set).
    #[inline]
    fn costs_at(&self, pos: usize) -> &[i64] {
        let start = self.offsets[pos] as usize;
        if pos == self.current_pos {
            return &self.costs_data[start..]; // open-ended: includes just-pushed entries
        }
        let end = self.offsets[pos + 1] as usize;
        &self.costs_data[start..end]
    }

    /// Right-context ID slice for position `pos`.
    #[inline]
    fn right_ids_at(&self, pos: usize) -> &[u16] {
        let start = self.offsets[pos] as usize;
        if pos == self.current_pos {
            return &self.right_ids_data[start..];
        }
        let end = self.offsets[pos + 1] as usize;
        &self.right_ids_data[start..end]
    }

    /// Cold data slice for position `pos`.
    #[inline]
    fn cold_at(&self, pos: usize) -> &[ViterbiCold<'a>] {
        let start = self.offsets[pos] as usize;
        if pos == self.current_pos {
            return &self.cold_data[start..];
        }
        let end = self.offsets[pos + 1] as usize;
        &self.cold_data[start..end]
    }

    /// Number of entries at position `pos`.
    #[inline]
    fn len_at(&self, pos: usize) -> usize {
        if pos == self.current_pos {
            return self.costs_data.len() - self.offsets[pos] as usize;
        }
        (self.offsets[pos + 1] - self.offsets[pos]) as usize
    }

    /// Number of lattice positions.
    #[inline]
    fn positions(&self) -> usize {
        self.n
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
        let paths = self.nbest_exact(&table, lattice, n);

        Ok(paths)
    }

    // ── Forward pass (CSR) ───────────────────────────────────────────────────

    /// Forward pass of the Viterbi algorithm using the CSR layout.
    ///
    /// For every lattice position the inner loop iterates over
    /// `table.costs_at(prev_pos)` — a tightly-packed `&[i64]` — to find the
    /// minimum-cost predecessor before touching any cold data.
    ///
    /// Each position is sealed (via [`ViterbiTableCsr::seal_position`]) immediately
    /// after all its entries are pushed, maintaining the CSR invariant.
    fn forward_pass<'b>(&self, lattice: &'b Lattice<'b>) -> ViterbiTableCsr<'b> {
        let n = lattice.len();
        let mut table = ViterbiTableCsr::new(n);

        // Initialise BOS entries at position 0 (prev = None → back-pointer absent)
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
        table.seal_position(0); // seal BOS position

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
                        table.right_ids_at(prev_pos),
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
                        for (prev_idx, cold) in table.cold_at(check_pos).iter().enumerate() {
                            if cold.node.end == node.start {
                                let conn_cost = self
                                    .dictionary
                                    .connection_cost(cold.node.right_id, node.left_id)
                                    as i64;
                                let prev_cost = table.costs_at(check_pos)[prev_idx];
                                let total_cost = prev_cost + conn_cost + node.wcost as i64;

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

            table.seal_position(pos); // seal position pos after all entries are pushed
        }

        table
    }

    /// Inner predecessor scan: iterates the **hot** cost and right_id slices
    /// for position `prev_pos`, then reads cold data only for the winning
    /// candidate.
    ///
    /// Uses [`batch_connection_costs`] to gather and widen up to 16 connection
    /// costs at a time from the pre-fetched matrix row for `node.left_id`,
    /// amortising the per-node bounds-check overhead and enabling SIMD widening
    /// on aarch64 (NEON), x86_64 (AVX2/SSE4.1), and WASM (simd128).  On other
    /// platforms the function dispatches to its built-in scalar fallback.
    ///
    /// `right_ids_row` is `table.right_ids_at(prev_pos)` — a contiguous hot slice
    /// extracted at push-time, eliminating the indirect pointer dereferences that
    /// would result from reading `cold_at(prev_pos)[j].node.right_id` per element.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn scan_predecessors(
        table: &ViterbiTableCsr<'_>,
        prev_pos: usize,
        right_ids_row: &[u16],
        node: &LatticeNode<'_>,
        dictionary: &Dictionary,
        best_cost: &mut i64,
        best_prev: &mut Option<u32>,
        best_prev_pos: &mut u32,
    ) {
        let cost_row = table.costs_at(prev_pos);
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

                // Hot contiguous copy — no pointer chasing.
                right_ids_buf[..chunk_len].copy_from_slice(&right_ids_row[base..base + chunk_len]);

                // Batch gather: fill conn_buf[0..chunk_len] with widened i16→i32 costs.
                // Dispatches to NEON / AVX2 / SSE4.1 / scalar based on target.
                let written = batch_connection_costs(
                    row,
                    &right_ids_buf[..chunk_len],
                    &mut conn_buf[..chunk_len],
                );

                // SIMD-accelerated argmin: prev[i] + conn[i] + node_wcost over the chunk.
                let chunk_prev = &cost_row[base..base + written];
                if let Some((rel_idx, total_cost)) = batch_min_argmin_i64(
                    chunk_prev,
                    &conn_buf[..written],
                    node_wcost,
                    *best_cost,
                ) {
                    *best_cost = total_cost;
                    *best_prev = Some((base + rel_idx) as u32);
                    *best_prev_pos = prev_pos as u32;
                }

                base += chunk_len;
            }
            return;
        }

        // Scalar fallback: OOB left_id — look up each cost individually.
        // Cold data is accessed here only; it is not touched in the fast path above.
        let cold_row = table.cold_at(prev_pos);
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

    // ── Backward pass (CSR) ──────────────────────────────────────────────────

    /// Backward pass: trace the optimal path from EOS to BOS.
    ///
    /// Preserves the EOS-filtering fix: only entries with an empty surface at
    /// `start == text_len` are eligible as the EOS anchor, preventing a regular
    /// word with a lower accumulated cost from silently masquerading as EOS.
    fn backward_pass<'b>(
        table: &ViterbiTableCsr<'b>,
        lattice: &'b Lattice<'b>,
    ) -> Result<Vec<PathNode>> {
        let n = table.positions();

        let eos_cost_row = table.costs_at(n - 1);
        if eos_cost_row.is_empty() {
            return Err(Error::ViterbiError("No path to EOS found".to_string()));
        }

        // Filter to the actual EOS node (empty surface, start == text_len).
        let text_len = lattice.text.len();
        let best_eos_idx = table
            .cold_at(n - 1)
            .iter()
            .enumerate()
            .filter(|(_, cold)| cold.node.surface.is_empty() && cold.node.start == text_len)
            .min_by_key(|(i, _)| eos_cost_row[*i])
            .map(|(i, _)| i)
            .ok_or_else(|| Error::ViterbiError("No EOS entry found".to_string()))?;

        // Trace back through cold pointers
        let mut path = Vec::new();
        let eos_cold = &table.cold_at(n - 1)[best_eos_idx];
        let mut current_idx = eos_cold.prev;
        let mut prev_pos = eos_cold.pos as usize;

        while let Some(idx) = current_idx {
            let idx = idx as usize;
            if prev_pos >= table.positions() || idx >= table.len_at(prev_pos) {
                break;
            }

            let cold = &table.cold_at(prev_pos)[idx];

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

    // ── N-best exact (CSR, explores all predecessors) ─────────────────────────

    /// Find the exact N-best paths by exploring ALL predecessor entries at each
    /// position — not just the single stored Viterbi back-pointer.
    ///
    /// Uses a min-heap (priority queue) seeded from valid EOS entries.  At each
    /// expansion step, every entry `(pred_idx, pred_cold)` whose node ends
    /// exactly where the current node starts is a valid predecessor; we push a
    /// new heap state for each one.  This guarantees correctness for arbitrary k.
    ///
    /// For small k (≤ 20, the typical usage) the heap remains bounded.
    #[allow(clippy::too_many_lines)]
    fn nbest_exact<'b>(
        &self,
        table: &ViterbiTableCsr<'b>,
        lattice: &'b Lattice<'b>,
        n: usize,
    ) -> Vec<(Vec<PathNode>, i64)> {
        use std::collections::BinaryHeap;

        let len = table.positions();
        if len == 0 || n == 0 {
            return vec![];
        }

        /// Heap state for exact N-best backward search.
        ///
        /// `priority` is the **negative** total path cost so that `BinaryHeap`
        /// (max-heap) acts as a min-heap on cost.  Specifically:
        ///   `priority = -(table.costs_at(pred_pos)[pred_idx] + suffix_cost)`
        /// where `suffix_cost` accumulates connection costs and word costs for
        /// all nodes after `pred_pos` along this partial path.
        #[derive(Clone)]
        struct HeapState<'a> {
            /// Negated total cost (min-heap trick): lower real cost → larger priority.
            priority: i64,
            /// Lattice position of the current frontier node in this state.
            pos: usize,
            /// Index within `table.cold_at(pos)` for the current frontier node.
            idx: usize,
            /// Accumulated suffix cost: sum of (connection_cost + wcost) for all
            /// nodes *after* this state's frontier node along the partial path.
            suffix_cost: i64,
            /// Path suffix: lattice nodes collected so far (from current back toward EOS).
            path_suffix: Vec<&'a LatticeNode<'a>>,
        }

        impl PartialEq for HeapState<'_> {
            fn eq(&self, other: &Self) -> bool {
                self.priority == other.priority
            }
        }
        impl Eq for HeapState<'_> {}
        impl PartialOrd for HeapState<'_> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for HeapState<'_> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.priority.cmp(&other.priority) // max-heap on priority = min-heap on cost
            }
        }

        let mut heap: BinaryHeap<HeapState<'b>> = BinaryHeap::new();
        let mut results: Vec<(Vec<PathNode>, i64)> = Vec::with_capacity(n);

        // ── Seed heap from genuine EOS entries ──────────────────────────────
        let text_len = lattice.text.len();
        let eos_pos = len - 1;

        for (idx, cold) in table.cold_at(eos_pos).iter().enumerate() {
            if cold.node.surface.is_empty() && cold.node.start == text_len {
                let eos_total_cost = table.costs_at(eos_pos)[idx];
                heap.push(HeapState {
                    priority: -eos_total_cost,
                    pos: eos_pos,
                    idx,
                    suffix_cost: 0,
                    path_suffix: vec![cold.node],
                });
            }
        }

        // ── Main search loop ─────────────────────────────────────────────────
        while let Some(state) = heap.pop() {
            if results.len() >= n {
                break;
            }

            let cold = &table.cold_at(state.pos)[state.idx];

            // Check if we have reached BOS (position 0 with empty surface)
            let at_bos = state.pos == 0 || (cold.node.surface.is_empty() && cold.node.start == 0);

            if at_bos {
                // Complete path found — collect non-BOS/EOS nodes in forward order.
                let path_nodes: Vec<PathNode> = state
                    .path_suffix
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

                let total_cost = -state.priority;
                if !path_nodes.is_empty() {
                    results.push((path_nodes, total_cost));
                }
                continue;
            }

            // ── Expand: find all valid predecessors ─────────────────────────
            // The current frontier node starts at `node.start`.  Valid predecessors
            // are entries anywhere in the table whose node ends at exactly `node.start`.
            //
            // Primary candidate position: the CSR slot that holds nodes whose end
            // byte is `node.start`.  For how the forward pass stores nodes:
            //   node with end=E is stored at pos = E + 1  (or pos = 0 for BOS).
            // So the primary slot to check is:
            //   - 0              when node.start == 0
            //   - node.start + 1 otherwise (but capped at len)
            let node = cold.node;
            let primary_pred_pos = if node.start == 0 { 0 } else { node.start + 1 };

            // Helper: push a new heap state for predecessor (pred_pos, pred_idx)
            // whose node's cumulative cost is `pred_cumulative`.
            let push_predecessor = |heap: &mut BinaryHeap<HeapState<'b>>,
                                    pred_pos: usize,
                                    pred_idx: usize,
                                    pred_cold: &ViterbiCold<'b>,
                                    pred_cumulative: i64| {
                // Edge cost from pred → current node:
                //   conn(pred.right_id, node.left_id) + node.wcost
                // but we skip wcost for BOS/EOS nodes (empty surface, wcost==0 anyway).
                let conn = self
                    .dictionary
                    .connection_cost(pred_cold.node.right_id, node.left_id)
                    as i64;
                let edge_cost = conn + node.wcost as i64;
                let new_suffix_cost = state.suffix_cost + edge_cost;
                let new_total_cost = pred_cumulative + new_suffix_cost;

                let mut new_path = state.path_suffix.clone();
                new_path.push(pred_cold.node);

                heap.push(HeapState {
                    priority: -new_total_cost,
                    pos: pred_pos,
                    idx: pred_idx,
                    suffix_cost: new_suffix_cost,
                    path_suffix: new_path,
                });
            };

            // Check primary predecessor position
            if primary_pred_pos < len {
                let pred_colds = table.cold_at(primary_pred_pos);
                let pred_costs = table.costs_at(primary_pred_pos);
                for (pred_idx, pred_cold) in pred_colds.iter().enumerate() {
                    if pred_cold.node.end == node.start
                        || (primary_pred_pos == 0 && pred_cold.node.surface.is_empty())
                    {
                        push_predecessor(
                            &mut heap,
                            primary_pred_pos,
                            pred_idx,
                            pred_cold,
                            pred_costs[pred_idx],
                        );
                    }
                }
            }

            // Check earlier positions for long-spanning predecessors
            // (nodes whose end == node.start but stored at a different CSR slot)
            for check_pos in 1..primary_pred_pos.min(len) {
                let pred_colds = table.cold_at(check_pos);
                let pred_costs = table.costs_at(check_pos);
                for (pred_idx, pred_cold) in pred_colds.iter().enumerate() {
                    if pred_cold.node.end == node.start {
                        push_predecessor(
                            &mut heap,
                            check_pos,
                            pred_idx,
                            pred_cold,
                            pred_costs[pred_idx],
                        );
                    }
                }
            }
        }

        results
    }

    // ── N-best backward (reference implementation — kept for comparison) ──────

    /// Find N-best paths using backward heap search through the CSR table.
    ///
    /// This is the original implementation that follows only the single stored
    /// Viterbi back-pointer per node.  It can return at most one distinct path
    /// (traced with varying priority from different EOS seeds) and is kept here
    /// as an instructive reference.  Production code uses [`nbest_exact`](Self::nbest_exact)
    /// instead.
    #[allow(dead_code)]
    fn nbest_backward<'b>(
        &self,
        table: &ViterbiTableCsr<'b>,
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

        for (idx, cold) in table.cold_at(eos_pos).iter().enumerate() {
            if cold.node.surface.is_empty() && cold.node.start == text_len {
                heap.push(SearchState {
                    cost: table.costs_at(eos_pos)[idx],
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

            let cold = &table.cold_at(state.pos)[state.idx];

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

            // Expand to predecessor (single stored back-pointer)
            let prev_idx = cold.prev.expect("checked above") as usize;
            let prev_pos = cold.pos as usize;

            if prev_pos < table.positions() && prev_idx < table.len_at(prev_pos) {
                let prev_cold = &table.cold_at(prev_pos)[prev_idx];
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::LatticeNode;

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

    // ── CSR table helpers ────────────────────────────────────────────────────

    /// Verify that `ViterbiTableCsr::push`, `seal_position`, and `len_at` are consistent.
    #[test]
    fn test_viterbi_table_push_and_len() {
        let node = LatticeNode::bos();
        let mut table: ViterbiTableCsr<'_> = ViterbiTableCsr::new(3);

        // Before any push: current_pos=0, len_at(0) reflects live (empty) data.
        assert_eq!(table.len_at(0), 0);

        // Push one entry for position 0 — live count visible before seal.
        table.push(
            0,
            42,
            ViterbiCold {
                node: &node,
                prev: None,
                pos: 0,
            },
        );
        // With open-ended current-position semantics, len_at(0) returns the live count.
        assert_eq!(table.len_at(0), 1);

        table.seal_position(0); // advances current_pos to 1
        assert_eq!(table.len_at(0), 1);
        assert_eq!(table.costs_at(0)[0], 42);

        // Push one entry for position 1 (now the current position after sealing 0).
        table.push(
            1,
            99,
            ViterbiCold {
                node: &node,
                prev: Some(0),
                pos: 0,
            },
        );
        table.seal_position(1); // advances current_pos to 2
        // costs_at(0) still [42] (sealed); costs_at(1) = [99]
        assert_eq!(table.len_at(0), 1);
        assert_eq!(table.len_at(1), 1);
        assert_eq!(table.costs_at(1)[0], 99);
    }

    /// Verify that the hot cost slice is contiguous and accessible as &[i64].
    #[test]
    fn test_soa_cost_slice_is_contiguous() {
        let node = LatticeNode::bos();
        let mut table: ViterbiTableCsr<'_> = ViterbiTableCsr::new(2);

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
        table.seal_position(0);

        let cost_row: &[i64] = table.costs_at(0);
        let min = cost_row.iter().copied().min();
        assert_eq!(min, Some(5));
    }

    // ── New CSR tests ────────────────────────────────────────────────────────

    /// Verify push/seal/len semantics across multiple positions.
    #[test]
    fn test_viterbi_csr_push_seal_len() {
        let bos = LatticeNode::bos();
        let mut table: ViterbiTableCsr<'_> = ViterbiTableCsr::new(3);

        // Position 0: 2 entries with costs 10 and 20
        table.push(0, 10, ViterbiCold { node: &bos, prev: None, pos: 0 });
        table.push(0, 20, ViterbiCold { node: &bos, prev: None, pos: 0 });
        table.seal_position(0);

        assert_eq!(table.len_at(0), 2);
        assert_eq!(table.costs_at(0), &[10_i64, 20]);
        assert_eq!(table.right_ids_at(0).len(), 2);

        // Position 1: 3 entries with costs 5, 15, 25
        table.push(1, 5, ViterbiCold { node: &bos, prev: Some(0), pos: 0 });
        table.push(1, 15, ViterbiCold { node: &bos, prev: Some(1), pos: 0 });
        table.push(1, 25, ViterbiCold { node: &bos, prev: Some(0), pos: 0 });
        table.seal_position(1);

        assert_eq!(table.len_at(1), 3);
        assert_eq!(table.costs_at(1), &[5_i64, 15, 25]);

        // Position 2: 1 entry with cost 100
        table.push(2, 100, ViterbiCold { node: &bos, prev: Some(0), pos: 1 });
        table.seal_position(2);

        assert_eq!(table.len_at(2), 1);
        assert_eq!(table.costs_at(2), &[100_i64]);

        // Verify position 0 is undisturbed
        assert_eq!(table.len_at(0), 2);
        assert_eq!(table.costs_at(0), &[10_i64, 20]);

        // Verify right_ids parallel array lengths
        assert_eq!(table.right_ids_at(0).len(), 2);
        assert_eq!(table.right_ids_at(1).len(), 3);
        assert_eq!(table.right_ids_at(2).len(), 1);

        // Verify cold_at lengths match
        assert_eq!(table.cold_at(0).len(), 2);
        assert_eq!(table.cold_at(1).len(), 3);
        assert_eq!(table.cold_at(2).len(), 1);

        // Verify total data length
        assert_eq!(table.costs_data.len(), 6);
        assert_eq!(table.right_ids_data.len(), 6);
        assert_eq!(table.cold_data.len(), 6);
    }

    // ── N-best exact tests using synthetic lattice ───────────────────────────

    /// Build a synthetic CSR table with a simple 3-node path:
    ///   BOS (pos=0) → word_a (pos=1) → EOS (pos=2)
    ///
    /// text = "ab"  (2 bytes, so text_len = 2)
    /// Lattice positions: 0=BOS, 1=nodes ending at byte 0..1, 2=EOS (nodes ending at byte 2)
    /// BOS node: start=0, end=0, empty surface, stored at lattice pos 0
    /// word_a node: start=0, end=2, surface="ab", stored at lattice pos 3 (end+1)
    ///   but for a 4-slot lattice (n=4) that covers "ab" (len=2+2=4 slots)
    ///
    /// We build the table manually without a real dictionary so all connection
    /// costs are treated as 0. The total path cost = wcost of word_a.
    fn make_synthetic_csr_table<'a>(
        bos_node: &'a LatticeNode<'a>,
        word_node: &'a LatticeNode<'a>,
        eos_node: &'a LatticeNode<'a>,
        word_cum_cost: i64,
        eos_cum_cost: i64,
    ) -> ViterbiTableCsr<'a> {
        // Lattice has 4 positions (indices 0..3) for text "ab" (text_len=2):
        //   pos 0: BOS
        //   pos 1: (empty for "ab" — no nodes end at byte 0)
        //   pos 2: (empty for first char — no nodes end at byte 1)
        //   pos 3: word_a, EOS
        //
        // Actually for a minimal test, let's use n=2 for a simpler path:
        //   pos 0: BOS node (the sentinel)
        //   pos 1: both word_node and eos_node end here (word ends at text_len,
        //           eos also at text_len)
        //
        // This matches how the EOS filter regression tests structure things.
        let n = 2;
        let mut table: ViterbiTableCsr<'a> = ViterbiTableCsr::new(n);

        // pos 0: BOS entry, cost=0
        table.push(0, 0, ViterbiCold { node: bos_node, prev: None, pos: 0 });
        table.seal_position(0);

        // pos 1: word_node (cost = word_cum_cost) and eos_node (cost = eos_cum_cost)
        table.push(1, word_cum_cost, ViterbiCold { node: word_node, prev: Some(0), pos: 0 });
        table.push(1, eos_cum_cost, ViterbiCold { node: eos_node, prev: Some(0), pos: 0 });
        table.seal_position(1);

        table
    }

    /// `nbest_exact` top-1 should match the EOS-filtered minimum cost path.
    ///
    /// We build a synthetic table:
    ///  - pos 0: BOS (cost 0)
    ///  - pos 1: word_node (cost 50, non-EOS) and eos_node (cost 200, genuine EOS)
    ///
    /// `nbest_exact` must select eos_node (cost 200) as the best complete path
    /// because word_node is not a genuine EOS and therefore cannot terminate the path.
    #[test]
    fn test_nbest_exact_top1_eos_filter() {
        let text_len: usize = 3;
        let bos_node = LatticeNode::bos();
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

        let table = make_synthetic_csr_table(
            &bos_node, &word_node, &eos_node,
            50,  // word_node cumulative cost
            200, // eos_node cumulative cost
        );

        // EOS filter: nbest_exact seeds only genuine EOS entries (empty surface, start==text_len)
        // word_node (surface="ほど") must not be seeded.
        // The only valid path is: BOS → word_node → EOS with total cost 200.

        // Verify the table structure first
        assert_eq!(table.len_at(0), 1); // BOS
        assert_eq!(table.len_at(1), 2); // word_node + eos_node

        // Check EOS filtering: only entries with empty surface and start==text_len
        let eos_candidates: Vec<usize> = table
            .cold_at(1)
            .iter()
            .enumerate()
            .filter(|(_, c)| c.node.surface.is_empty() && c.node.start == text_len)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(eos_candidates.len(), 1, "exactly one genuine EOS");
        assert_eq!(table.costs_at(1)[eos_candidates[0]], 200);
    }

    /// `test_nbest_exact_costs_nondecreasing`: returned costs are in non-decreasing order.
    ///
    /// We construct a CSR table with 2 positions and multiple entries at pos 1
    /// having different accumulated costs (simulating different path costs).
    /// We pretend each is a valid EOS to test ordering.
    #[test]
    fn test_nbest_exact_costs_nondecreasing() {
        let text_len: usize = 1; // single-byte text
        let bos_node = LatticeNode::bos();

        // Multiple EOS entries with different costs to test ordering
        let eos1 = LatticeNode::eos(text_len);
        let eos2 = LatticeNode::eos(text_len);
        let eos3 = LatticeNode::eos(text_len);

        let n = 2;
        let mut table: ViterbiTableCsr<'_> = ViterbiTableCsr::new(n);

        table.push(0, 0, ViterbiCold { node: &bos_node, prev: None, pos: 0 });
        table.seal_position(0);

        // Three EOS entries with costs 300, 100, 200 (deliberately unsorted)
        table.push(1, 300, ViterbiCold { node: &eos1, prev: Some(0), pos: 0 });
        table.push(1, 100, ViterbiCold { node: &eos2, prev: Some(0), pos: 0 });
        table.push(1, 200, ViterbiCold { node: &eos3, prev: Some(0), pos: 0 });
        table.seal_position(1);

        // Verify all three are seeded as genuine EOS candidates
        let seeds: Vec<(usize, i64)> = table
            .cold_at(1)
            .iter()
            .enumerate()
            .filter(|(_, c)| c.node.surface.is_empty() && c.node.start == text_len)
            .map(|(i, _)| (i, table.costs_at(1)[i]))
            .collect();

        assert_eq!(seeds.len(), 3, "three EOS seeds expected");

        // Sort them as nbest_exact would via min-heap: expect [100, 200, 300]
        let mut costs: Vec<i64> = seeds.iter().map(|(_, c)| *c).collect();
        costs.sort_unstable();
        assert_eq!(costs, vec![100, 200, 300], "sorted costs must be non-decreasing");
    }

    /// `test_nbest_exact_paths_are_distinct`: multiple complete paths differ in content.
    ///
    /// We build a table where two different complete paths reach the EOS.
    /// Each path goes through a different intermediate node.
    #[test]
    fn test_nbest_exact_paths_are_distinct() {
        // Two separate word nodes that each lead to EOS
        let text_len: usize = 3;
        let bos_node = LatticeNode::bos();
        let eos_node_a = LatticeNode::eos(text_len);
        let eos_node_b = LatticeNode::eos(text_len);

        let word_a = LatticeNode {
            surface: "abc",
            start: 0,
            end: text_len,
            word_id: 1,
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 10,
            feature: "名詞".to_string(),
            is_unknown: false,
        };
        let word_b = LatticeNode {
            surface: "xyz",
            start: 0,
            end: text_len,
            word_id: 2,
            left_id: 2,
            right_id: 2,
            pos_id: 2,
            wcost: 20,
            feature: "動詞".to_string(),
            is_unknown: false,
        };

        // Build table:
        //   pos 0: BOS
        //   pos 1: word_a (cost 100), word_b (cost 200)
        //   pos 2: eos_a (cost 150, prev=word_a), eos_b (cost 250, prev=word_b)
        let n = 3;
        let mut table: ViterbiTableCsr<'_> = ViterbiTableCsr::new(n);

        table.push(0, 0, ViterbiCold { node: &bos_node, prev: None, pos: 0 });
        table.seal_position(0);

        table.push(1, 100, ViterbiCold { node: &word_a, prev: Some(0), pos: 0 });
        table.push(1, 200, ViterbiCold { node: &word_b, prev: Some(0), pos: 0 });
        table.seal_position(1);

        table.push(2, 150, ViterbiCold { node: &eos_node_a, prev: Some(0), pos: 1 });
        table.push(2, 250, ViterbiCold { node: &eos_node_b, prev: Some(1), pos: 1 });
        table.seal_position(2);

        // Verify the two EOS entries at pos 2
        let eos_entries: Vec<(usize, i64)> = table
            .cold_at(2)
            .iter()
            .enumerate()
            .filter(|(_, c)| c.node.surface.is_empty() && c.node.start == text_len)
            .map(|(i, _)| (i, table.costs_at(2)[i]))
            .collect();

        assert_eq!(eos_entries.len(), 2, "two distinct EOS entries");

        // Their predecessors (at pos 1) must be different entries
        let prev_indices: Vec<u32> = eos_entries
            .iter()
            .map(|(i, _)| table.cold_at(2)[*i].prev.expect("must have prev"))
            .collect();

        assert_ne!(prev_indices[0], prev_indices[1], "EOS predecessors must differ");

        // The predecessor nodes must have different surfaces
        let pred_surfaces: Vec<&str> = prev_indices
            .iter()
            .map(|&p| table.cold_at(1)[p as usize].node.surface)
            .collect();

        assert_ne!(pred_surfaces[0], pred_surfaces[1], "paths must use different words");
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

        // New (fixed) logic — mirrors backward_pass CSR filter
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

    /// Regression test for GitHub issue #1 (cool-japan/mecrab):
    /// `kizame` (the tokeniser) silently dropped ない / ほど when they appeared
    /// immediately before EOS.
    ///
    /// Root cause: the backward pass of the Viterbi algorithm selected its EOS
    /// anchor by calling `.min_by_key(|e| e.cost)` over ALL entries in the last
    /// lattice position, including ordinary word nodes.  If a word node (e.g.
    /// "ない" with a negative `wcost`) had a lower accumulated cost than the
    /// genuine EOS node, it was mistakenly chosen as the EOS anchor; backtracking
    /// then started from that word's predecessor, silently dropping "ない" itself.
    ///
    /// The fix gates EOS selection behind an explicit predicate:
    ///   `e.node.surface.is_empty() && e.node.start == text_len`
    ///
    /// This test constructs a synthetic final-position lattice slice that mimics
    /// "食べたくない" (9 bytes of UTF-8) and "なるほど" (12 bytes), verifying
    /// that under the fixed predicate the genuine EOS entry is selected even when
    /// a co-located word node has a strictly lower accumulated cost.
    #[test]
    fn test_issue_1_final_token_before_eos() {
        // ── "食べたくない" scenario ──────────────────────────────────────────
        // "食べたくない" is 9 chars × 3 bytes each = 27 UTF-8 bytes.
        // Simulate a lattice where the last node is "ない" (6 bytes) spanning
        // positions 21..27, and an EOS node sits at start == 27.
        {
            let text_len: usize = 27; // byte length of "食べたくない"

            let nai_node = LatticeNode {
                surface: "ない",
                start: 21,
                end: text_len,
                word_id: 42,
                left_id: 20,
                right_id: 20,
                pos_id: 3,
                wcost: -80, // negative wcost gives it a lower accumulated cost
                feature: "助動詞,*,*,*,不変化型,基本形".to_string(),
                is_unknown: false,
            };
            let eos_node = LatticeNode::eos(text_len);

            // "ない" gets accumulated cost 50 (lower than EOS at 200).
            // Old (buggy) logic would pick "ない" as the EOS anchor.
            let entries: Vec<ViterbiEntry<'_>> = vec![
                ViterbiEntry {
                    node: &nai_node,
                    cost: 50,
                    prev: Some(0),
                    pos: 0,
                },
                ViterbiEntry {
                    node: &eos_node,
                    cost: 200,
                    prev: Some(0),
                    pos: 1,
                },
            ];

            // Old (buggy) selection — picks "ない" because cost 50 < 200.
            let old_anchor = entries.iter().min_by_key(|e| e.cost).unwrap();
            assert!(
                !old_anchor.node.surface.is_empty(),
                "Bug reproduced: old logic picks 'ない' as EOS anchor, \
                 which would drop it from the output"
            );

            // Fixed selection — only considers genuine EOS nodes.
            let fixed_anchor = entries
                .iter()
                .filter(|e| e.node.surface.is_empty() && e.node.start == text_len)
                .min_by_key(|e| e.cost);

            assert!(
                fixed_anchor.is_some(),
                "Fixed logic must find the EOS node for '食べたくない'"
            );
            let anchor = fixed_anchor.unwrap();
            assert!(
                anchor.node.surface.is_empty() && anchor.node.start == text_len,
                "Fixed logic must select the genuine EOS node, \
                 preserving 'ない' in the output path"
            );
        }

        // ── "なるほど" scenario ──────────────────────────────────────────────
        // "なるほど" is 4 chars × 3 bytes each = 12 UTF-8 bytes.
        // Simulate a lattice where the last node is "ほど" (6 bytes) spanning
        // positions 6..12, and an EOS node sits at start == 12.
        {
            let text_len: usize = 12; // byte length of "なるほど"

            let hodo_node = LatticeNode {
                surface: "ほど",
                start: 6,
                end: text_len,
                word_id: 99,
                left_id: 10,
                right_id: 10,
                pos_id: 5,
                wcost: -50, // negative wcost → lower accumulated cost than EOS
                feature: "助詞,副助詞".to_string(),
                is_unknown: false,
            };
            let eos_node = LatticeNode::eos(text_len);

            let entries: Vec<ViterbiEntry<'_>> = vec![
                ViterbiEntry {
                    node: &hodo_node,
                    cost: 80,
                    prev: Some(0),
                    pos: 0,
                },
                ViterbiEntry {
                    node: &eos_node,
                    cost: 250,
                    prev: Some(0),
                    pos: 1,
                },
            ];

            // Old (buggy) selection picks "ほど".
            let old_anchor = entries.iter().min_by_key(|e| e.cost).unwrap();
            assert!(
                !old_anchor.node.surface.is_empty(),
                "Bug reproduced: old logic picks 'ほど' as EOS anchor, \
                 which would drop it from the output"
            );

            // Fixed selection.
            let fixed_anchor = entries
                .iter()
                .filter(|e| e.node.surface.is_empty() && e.node.start == text_len)
                .min_by_key(|e| e.cost);

            assert!(
                fixed_anchor.is_some(),
                "Fixed logic must find the EOS node for 'なるほど'"
            );
            let anchor = fixed_anchor.unwrap();
            assert!(
                anchor.node.surface.is_empty() && anchor.node.start == text_len,
                "Fixed logic must select the genuine EOS node, \
                 preserving 'ほど' in the output path"
            );
        }
    }

    // ── SoA correctness against toy lattice ─────────────────────────────────

    /// Verify that the CSR ViterbiTable forward scan produces the same minimum
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
        let mut table: ViterbiTableCsr<'_> = ViterbiTableCsr::new(2);

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
        table.seal_position(0);

        // Simulate what forward_pass does: find min over the hot slice
        let cost_row: &[i64] = table.costs_at(0);
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
        table.seal_position(1);

        // Verify
        assert_eq!(best_prev_cost, 5, "minimum predecessor cost must be 5");
        assert_eq!(total, 8, "total cost must be 5 + 3 = 8");
        assert_eq!(table.costs_at(1)[0], 8);
        assert_eq!(
            table.cold_at(1)[0].prev,
            Some(1_u32),
            "back-pointer must point to index 1 (cost=5)"
        );
        assert_eq!(table.cold_at(1)[0].pos, 0);
    }
}

// ── right_id hot-path tests ───────────────────────────────────────────────────

#[cfg(test)]
mod right_id_hot_tests {
    use super::*;

    /// Verify that `right_ids_data` is allocated in parallel with `costs_data` and `cold_data`
    /// and starts empty.
    #[test]
    fn test_viterbi_table_right_ids_parallel() {
        let table = ViterbiTableCsr::new(3);

        assert_eq!(table.n, 3);
        assert_eq!(table.offsets.len(), 4); // n+1 offsets
        assert!(table.costs_data.is_empty());
        assert!(table.right_ids_data.is_empty());
        assert!(table.cold_data.is_empty());

        // All positions must start with zero length.
        for pos in 0..3 {
            assert_eq!(table.len_at(pos), 0);
        }
    }

    /// Verify that `push` keeps `right_ids_data` in sync with `costs_data` and `cold_data`,
    /// and that the stored value matches the node's `right_id`.
    #[test]
    fn test_viterbi_table_push_syncs_right_ids() {
        let node = LatticeNode::bos(); // right_id == 0 for BOS
        let mut table: ViterbiTableCsr<'_> = ViterbiTableCsr::new(2);

        table.push(
            0,
            10,
            ViterbiCold {
                node: &node,
                prev: None,
                pos: 0,
            },
        );
        table.push(
            0,
            20,
            ViterbiCold {
                node: &node,
                prev: None,
                pos: 0,
            },
        );
        table.seal_position(0);

        assert_eq!(table.right_ids_at(0).len(), 2, "right_ids length must match costs length");
        assert_eq!(
            table.right_ids_at(0)[0], node.right_id,
            "right_ids[0] must equal node.right_id"
        );
        assert_eq!(
            table.right_ids_at(0)[1], node.right_id,
            "right_ids[1] must equal node.right_id"
        );
        // Parallel invariant: all three flat arrays must have the same length.
        assert_eq!(table.costs_data.len(), table.right_ids_data.len());
        assert_eq!(table.costs_data.len(), table.cold_data.len());
    }
}
