//! Viterbi algorithm implementation for finding the optimal path
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! The Viterbi algorithm finds the path through the lattice with the
//! minimum total cost (word costs + connection costs).

pub mod analysis;
pub mod nbest;
#[cfg(feature = "simd")]
pub mod simd;

pub use analysis::{
    ConnectionMatrixStats, LatticeStats, MorphemeCost, PathAnalysis, PathComparison,
    SegmentationReport,
};
pub use nbest::{NBestIter, NBestSearch, PathDiversity, ScoredPath};

use crate::dict::Dictionary;
use crate::lattice::{Lattice, LatticeNode};
use crate::{Error, Result};

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
}

/// Entry in the Viterbi table
#[derive(Debug, Clone)]
struct ViterbiEntry<'a> {
    /// The lattice node
    node: &'a LatticeNode<'a>,
    /// Cumulative cost to reach this node
    cost: i64,
    /// Back pointer to the previous entry
    prev: Option<usize>,
    /// Position in the nodes_at vector for backtracking
    pos: usize,
}

/// Viterbi solver for finding the optimal path through the lattice
pub struct ViterbiSolver<'a> {
    dictionary: &'a Dictionary,
}

impl<'a> ViterbiSolver<'a> {
    /// Create a new Viterbi solver
    pub const fn new(dictionary: &'a Dictionary) -> Self {
        Self { dictionary }
    }

    /// Solve the lattice and return the optimal path
    ///
    /// # Arguments
    ///
    /// * `lattice` - The lattice to solve
    ///
    /// # Errors
    ///
    /// Returns an error if no valid path is found.
    pub fn solve<'b>(&self, lattice: &'b Lattice<'b>) -> Result<Vec<PathNode>> {
        if lattice.is_empty() {
            return Err(Error::LatticeError("Empty lattice".to_string()));
        }

        // Forward pass: compute minimum costs to reach each node
        let entries = self.forward_pass(lattice);

        // Backward pass: trace back from EOS to BOS
        let path = Self::backward_pass(&entries, lattice)?;

        Ok(path)
    }

    /// Solve the lattice and return the N-best paths
    ///
    /// # Arguments
    ///
    /// * `lattice` - The lattice to solve
    /// * `n` - Number of best paths to return
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

        // Forward pass to compute cumulative costs
        let entries = self.forward_pass(lattice);

        // Build the N-best paths using backward search
        let paths = self.nbest_backward(&entries, lattice, n);

        Ok(paths)
    }

    /// Find N-best paths using backward search through the entries
    fn nbest_backward<'b>(
        &self,
        entries: &[Vec<ViterbiEntry<'b>>],
        lattice: &'b Lattice<'b>,
        n: usize,
    ) -> Vec<(Vec<PathNode>, i64)> {
        use std::collections::BinaryHeap;

        let len = lattice.len();
        if len == 0 || entries.is_empty() {
            return vec![];
        }

        // State for search: (cost, position, entry_index, partial_path)
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
                // Min-heap: reverse ordering
                other.cost.cmp(&self.cost)
            }
        }

        let mut heap: BinaryHeap<SearchState<'b>> = BinaryHeap::new();
        let mut results = Vec::with_capacity(n);

        // Start from EOS entries
        let eos_pos = len - 1;
        for (idx, entry) in entries[eos_pos].iter().enumerate() {
            heap.push(SearchState {
                cost: entry.cost,
                pos: eos_pos,
                idx,
                path: vec![entry.node],
            });
        }

        while let Some(state) = heap.pop() {
            if results.len() >= n {
                break;
            }

            let entry = &entries[state.pos][state.idx];

            // Check if we've reached BOS
            if entry.prev.is_none() {
                // Construct path from nodes (skip BOS/EOS)
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
                    })
                    .collect();

                if !path_nodes.is_empty() || state.path.len() <= 2 {
                    results.push((path_nodes, state.cost));
                }
                continue;
            }

            // Expand to predecessor
            let prev_idx = entry.prev.unwrap();
            let prev_pos = entry.pos;

            if prev_pos < entries.len() && prev_idx < entries[prev_pos].len() {
                let prev_entry = &entries[prev_pos][prev_idx];
                let mut new_path = state.path.clone();
                new_path.push(prev_entry.node);

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

    /// Forward pass of the Viterbi algorithm
    fn forward_pass<'b>(&self, lattice: &'b Lattice<'b>) -> Vec<Vec<ViterbiEntry<'b>>> {
        let n = lattice.len();
        let mut entries: Vec<Vec<ViterbiEntry<'b>>> = vec![Vec::new(); n];

        // Initialize BOS
        for node in lattice.nodes_ending_at(0) {
            entries[0].push(ViterbiEntry {
                node,
                cost: 0,
                prev: None,
                pos: 0,
            });
        }

        // Process each position
        for pos in 1..n {
            let nodes = lattice.nodes_ending_at(pos);

            for node in nodes {
                let mut best_cost = i64::MAX;
                let mut best_prev: Option<usize> = None;
                let mut best_prev_pos: usize = 0;

                // Find the best predecessor
                let _start_pos = if node.start == 0 { 0 } else { node.start };

                // Look at all nodes that end where this node starts
                // (shifted by 1 due to BOS)
                let prev_pos = if node.start == 0 && pos > 0 {
                    0 // Connect to BOS
                } else {
                    node.start + 1
                };

                if prev_pos < entries.len() {
                    for (prev_idx, prev_entry) in entries[prev_pos].iter().enumerate() {
                        // Calculate connection cost
                        let conn_cost = self
                            .dictionary
                            .connection_cost(prev_entry.node.right_id, node.left_id)
                            as i64;

                        let total_cost = prev_entry.cost + conn_cost + node.wcost as i64;

                        if total_cost < best_cost {
                            best_cost = total_cost;
                            best_prev = Some(prev_idx);
                            best_prev_pos = prev_pos;
                        }
                    }
                }

                // Also check connections from earlier positions (for longer words)
                for check_pos in 1..prev_pos {
                    if check_pos < entries.len() {
                        for (prev_idx, prev_entry) in entries[check_pos].iter().enumerate() {
                            if prev_entry.node.end == node.start {
                                let conn_cost = self
                                    .dictionary
                                    .connection_cost(prev_entry.node.right_id, node.left_id)
                                    as i64;

                                let total_cost = prev_entry.cost + conn_cost + node.wcost as i64;

                                if total_cost < best_cost {
                                    best_cost = total_cost;
                                    best_prev = Some(prev_idx);
                                    best_prev_pos = check_pos;
                                }
                            }
                        }
                    }
                }

                if best_cost < i64::MAX {
                    entries[pos].push(ViterbiEntry {
                        node,
                        cost: best_cost,
                        prev: best_prev,
                        pos: best_prev_pos,
                    });
                }
            }
        }

        entries
    }

    /// Backward pass: trace the optimal path
    fn backward_pass<'b>(
        entries: &[Vec<ViterbiEntry<'b>>],
        lattice: &'b Lattice<'b>,
    ) -> Result<Vec<PathNode>> {
        let n = lattice.len();

        // Find the best EOS entry
        let eos_entries = &entries[n - 1];
        if eos_entries.is_empty() {
            return Err(Error::ViterbiError("No path to EOS found".to_string()));
        }

        let best_eos = eos_entries
            .iter()
            .min_by_key(|e| e.cost)
            .ok_or_else(|| Error::ViterbiError("No EOS entry found".to_string()))?;

        // Trace back
        let mut path = Vec::new();
        let mut current_idx = best_eos.prev;
        let mut prev_pos = best_eos.pos;

        while let Some(idx) = current_idx {
            if prev_pos >= entries.len() || idx >= entries[prev_pos].len() {
                break;
            }

            let entry = &entries[prev_pos][idx];

            // Skip BOS and EOS
            if !entry.node.surface.is_empty() {
                path.push(PathNode {
                    surface: entry.node.surface.to_string(),
                    word_id: entry.node.word_id,
                    pos_id: entry.node.pos_id,
                    wcost: entry.node.wcost,
                    feature: entry.node.feature.clone(),
                });
            }

            current_idx = entry.prev;
            prev_pos = entry.pos;
        }

        path.reverse();
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_node_creation() {
        let node = PathNode {
            surface: "テスト".to_string(),
            word_id: 42,
            pos_id: 1,
            wcost: 100,
            feature: "名詞,一般".to_string(),
        };

        assert_eq!(node.surface, "テスト");
        assert_eq!(node.pos_id, 1);
    }
}
