//! Debug session for lattice visualization and analysis
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module provides a DebugSession struct that captures lattice state,
//! Viterbi costs, and best path information for interactive debugging.

use std::collections::HashMap;

use crate::Result;
use crate::dict::Dictionary;
use crate::lattice::{Lattice, LatticeNode};

/// A debug node representing a lattice node with computed costs
#[derive(Debug, Clone)]
pub struct DebugNode {
    /// Surface form
    pub surface: String,
    /// Start position in bytes
    pub start: usize,
    /// End position in bytes
    pub end: usize,
    /// Left context ID
    pub left_id: u16,
    /// Right context ID
    pub right_id: u16,
    /// Word cost from dictionary
    pub wcost: i16,
    /// Feature string
    pub feature: String,
    /// Whether this is an unknown word
    pub is_unknown: bool,
    /// Position in lattice (nodes_at index)
    pub lattice_pos: usize,
    /// Index within position
    pub node_idx: usize,
}

impl DebugNode {
    /// Create a DebugNode from a LatticeNode
    pub fn from_lattice_node(node: &LatticeNode<'_>, lattice_pos: usize, node_idx: usize) -> Self {
        Self {
            surface: node.surface.to_string(),
            start: node.start,
            end: node.end,
            left_id: node.left_id,
            right_id: node.right_id,
            wcost: node.wcost,
            feature: node.feature.clone(),
            is_unknown: node.is_unknown,
            lattice_pos,
            node_idx,
        }
    }

    /// Check if this is a BOS node
    pub fn is_bos(&self) -> bool {
        self.surface.is_empty() && self.start == 0 && self.end == 0
    }

    /// Check if this is an EOS node
    pub fn is_eos(&self) -> bool {
        self.surface.is_empty() && self.start == self.end && self.start > 0
    }
}

/// Viterbi entry containing cost information
#[derive(Debug, Clone)]
pub struct DebugViterbiEntry {
    /// Position in the lattice
    pub node_pos: usize,
    /// Index of the node at this position
    pub node_idx: usize,
    /// Cumulative cost to reach this node
    pub cumulative_cost: i64,
    /// Connection cost from predecessor
    pub connection_cost: i16,
    /// Predecessor position and index (if any)
    pub prev: Option<(usize, usize)>,
}

/// Edge between two nodes with cost information
#[derive(Debug, Clone)]
pub struct DebugEdge {
    /// Source node (position, index)
    pub from: (usize, usize),
    /// Target node (position, index)
    pub to: (usize, usize),
    /// Connection cost
    pub connection_cost: i16,
}

/// A complete debug session capturing lattice state and costs
#[derive(Debug)]
pub struct DebugSession {
    /// Original input text
    pub text: String,
    /// All nodes organized by position
    pub nodes_at: Vec<Vec<DebugNode>>,
    /// Viterbi entries indexed by (pos, idx)
    pub viterbi_entries: HashMap<(usize, usize), DebugViterbiEntry>,
    /// All edges in the lattice
    pub edges: Vec<DebugEdge>,
    /// The best path as a sequence of (pos, idx) pairs
    pub best_path: Vec<(usize, usize)>,
    /// Total cost of the best path
    pub best_path_cost: i64,
    /// Alternative paths (if computed)
    pub alternative_paths: Vec<(Vec<(usize, usize)>, i64)>,
}

impl DebugSession {
    /// Create a new debug session from input text and dictionary
    ///
    /// # Errors
    ///
    /// Returns an error if lattice construction or Viterbi algorithm fails.
    pub fn new(text: &str, dict: &Dictionary) -> Result<Self> {
        // Build the lattice
        let lattice = Lattice::build(text, dict)?;

        // Convert lattice nodes to debug nodes
        let nodes_at: Vec<Vec<DebugNode>> = lattice
            .nodes_at
            .iter()
            .enumerate()
            .map(|(pos, nodes)| {
                nodes
                    .iter()
                    .enumerate()
                    .map(|(idx, node)| DebugNode::from_lattice_node(node, pos, idx))
                    .collect()
            })
            .collect();

        // Run forward pass to compute costs
        let (viterbi_entries, edges) = Self::forward_pass(&lattice, dict);

        // Find best path
        let (best_path, best_path_cost) = Self::find_best_path(&viterbi_entries, lattice.len());

        Ok(Self {
            text: text.to_string(),
            nodes_at,
            viterbi_entries,
            edges,
            best_path,
            best_path_cost,
            alternative_paths: Vec::new(),
        })
    }

    /// Compute forward pass and return Viterbi entries and edges
    fn forward_pass(
        lattice: &Lattice<'_>,
        dict: &Dictionary,
    ) -> (HashMap<(usize, usize), DebugViterbiEntry>, Vec<DebugEdge>) {
        let n = lattice.len();
        let mut entries: HashMap<(usize, usize), DebugViterbiEntry> = HashMap::new();
        let mut edges: Vec<DebugEdge> = Vec::new();

        // Initialize BOS
        for (idx, _node) in lattice.nodes_ending_at(0).iter().enumerate() {
            entries.insert(
                (0, idx),
                DebugViterbiEntry {
                    node_pos: 0,
                    node_idx: idx,
                    cumulative_cost: 0,
                    connection_cost: 0,
                    prev: None,
                },
            );
        }

        // Process each position
        for pos in 1..n {
            let nodes = lattice.nodes_ending_at(pos);

            for (node_idx, node) in nodes.iter().enumerate() {
                let mut best_cost = i64::MAX;
                let mut best_prev: Option<(usize, usize)> = None;
                let mut best_conn_cost: i16 = 0;

                // Find the best predecessor
                let prev_pos = if node.start == 0 && pos > 0 {
                    0 // Connect to BOS
                } else {
                    node.start + 1
                };

                // Check predecessors at the expected position
                if prev_pos < n {
                    for (prev_idx, prev_node) in
                        lattice.nodes_ending_at(prev_pos).iter().enumerate()
                    {
                        if prev_node.end == node.start || (prev_pos == 0 && node.start == 0) {
                            if let Some(prev_entry) = entries.get(&(prev_pos, prev_idx)) {
                                let conn_cost =
                                    dict.connection_cost(prev_node.right_id, node.left_id);
                                let total_cost = prev_entry.cumulative_cost
                                    + conn_cost as i64
                                    + node.wcost as i64;

                                // Record edge
                                edges.push(DebugEdge {
                                    from: (prev_pos, prev_idx),
                                    to: (pos, node_idx),
                                    connection_cost: conn_cost,
                                });

                                if total_cost < best_cost {
                                    best_cost = total_cost;
                                    best_prev = Some((prev_pos, prev_idx));
                                    best_conn_cost = conn_cost;
                                }
                            }
                        }
                    }
                }

                // Also check earlier positions for longer words
                for check_pos in 1..prev_pos {
                    for (prev_idx, prev_node) in
                        lattice.nodes_ending_at(check_pos).iter().enumerate()
                    {
                        if prev_node.end == node.start {
                            if let Some(prev_entry) = entries.get(&(check_pos, prev_idx)) {
                                let conn_cost =
                                    dict.connection_cost(prev_node.right_id, node.left_id);
                                let total_cost = prev_entry.cumulative_cost
                                    + conn_cost as i64
                                    + node.wcost as i64;

                                // Record edge
                                edges.push(DebugEdge {
                                    from: (check_pos, prev_idx),
                                    to: (pos, node_idx),
                                    connection_cost: conn_cost,
                                });

                                if total_cost < best_cost {
                                    best_cost = total_cost;
                                    best_prev = Some((check_pos, prev_idx));
                                    best_conn_cost = conn_cost;
                                }
                            }
                        }
                    }
                }

                if best_cost < i64::MAX {
                    entries.insert(
                        (pos, node_idx),
                        DebugViterbiEntry {
                            node_pos: pos,
                            node_idx,
                            cumulative_cost: best_cost,
                            connection_cost: best_conn_cost,
                            prev: best_prev,
                        },
                    );
                }
            }
        }

        (entries, edges)
    }

    /// Find the best path by tracing back from EOS
    fn find_best_path(
        entries: &HashMap<(usize, usize), DebugViterbiEntry>,
        lattice_len: usize,
    ) -> (Vec<(usize, usize)>, i64) {
        let eos_pos = lattice_len - 1;

        // Find best EOS entry
        let mut best_eos: Option<&DebugViterbiEntry> = None;
        for idx in 0.. {
            if let Some(entry) = entries.get(&(eos_pos, idx)) {
                if best_eos.is_none() || entry.cumulative_cost < best_eos.unwrap().cumulative_cost {
                    best_eos = Some(entry);
                }
            } else {
                break;
            }
        }

        let Some(eos_entry) = best_eos else {
            return (Vec::new(), 0);
        };

        let best_cost = eos_entry.cumulative_cost;
        let mut path = vec![(eos_entry.node_pos, eos_entry.node_idx)];

        let mut current = eos_entry.prev;
        while let Some((pos, idx)) = current {
            path.push((pos, idx));
            if let Some(entry) = entries.get(&(pos, idx)) {
                current = entry.prev;
            } else {
                break;
            }
        }

        path.reverse();
        (path, best_cost)
    }

    /// Get a node by position and index
    pub fn get_node(&self, pos: usize, idx: usize) -> Option<&DebugNode> {
        self.nodes_at.get(pos).and_then(|nodes| nodes.get(idx))
    }

    /// Get the Viterbi entry for a node
    pub fn get_viterbi_entry(&self, pos: usize, idx: usize) -> Option<&DebugViterbiEntry> {
        self.viterbi_entries.get(&(pos, idx))
    }

    /// Check if a node is on the best path
    pub fn is_on_best_path(&self, pos: usize, idx: usize) -> bool {
        self.best_path.contains(&(pos, idx))
    }

    /// Get all edges from a node
    pub fn edges_from(&self, pos: usize, idx: usize) -> Vec<&DebugEdge> {
        self.edges.iter().filter(|e| e.from == (pos, idx)).collect()
    }

    /// Get all edges to a node
    pub fn edges_to(&self, pos: usize, idx: usize) -> Vec<&DebugEdge> {
        self.edges.iter().filter(|e| e.to == (pos, idx)).collect()
    }

    /// Get the number of positions in the lattice
    pub fn num_positions(&self) -> usize {
        self.nodes_at.len()
    }

    /// Get the maximum number of nodes at any position
    pub fn max_nodes_at_position(&self) -> usize {
        self.nodes_at
            .iter()
            .map(|nodes| nodes.len())
            .max()
            .unwrap_or(0)
    }

    /// Calculate cost delta from best alternative
    ///
    /// Returns how much more costly the best alternative is compared to the best path
    pub fn cost_delta_from_alternative(&self, pos: usize, idx: usize) -> Option<i64> {
        if self.is_on_best_path(pos, idx) {
            // Find the best alternative at this position
            let nodes = &self.nodes_at[pos];
            let mut best_alt_cost: Option<i64> = None;

            for (alt_idx, _) in nodes.iter().enumerate() {
                if alt_idx != idx {
                    if let Some(entry) = self.viterbi_entries.get(&(pos, alt_idx)) {
                        if best_alt_cost.is_none() || entry.cumulative_cost < best_alt_cost.unwrap()
                        {
                            best_alt_cost = Some(entry.cumulative_cost);
                        }
                    }
                }
            }

            best_alt_cost.map(|alt| {
                let this_cost = self
                    .viterbi_entries
                    .get(&(pos, idx))
                    .map(|e| e.cumulative_cost)
                    .unwrap_or(0);
                alt - this_cost
            })
        } else {
            // This is not on the best path, compute delta from best path node at this position
            let best_at_pos = self.best_path.iter().find(|(p, _)| *p == pos);
            if let Some(&(_, best_idx)) = best_at_pos {
                let best_cost = self
                    .viterbi_entries
                    .get(&(pos, best_idx))
                    .map(|e| e.cumulative_cost)
                    .unwrap_or(0);
                let this_cost = self
                    .viterbi_entries
                    .get(&(pos, idx))
                    .map(|e| e.cumulative_cost)
                    .unwrap_or(0);
                Some(this_cost - best_cost)
            } else {
                None
            }
        }
    }

    /// Recalculate costs with a modified connection cost (what-if scenario)
    ///
    /// Returns the new total path cost if we modified the connection cost between
    /// two context IDs.
    pub fn what_if_connection_cost(
        &self,
        left_id: u16,
        right_id: u16,
        new_cost: i16,
        dict: &Dictionary,
    ) -> i64 {
        // Recalculate the best path cost with the modified connection
        let mut total_cost: i64 = 0;

        for i in 1..self.best_path.len() {
            let (prev_pos, prev_idx) = self.best_path[i - 1];
            let (curr_pos, curr_idx) = self.best_path[i];

            if let (Some(prev_node), Some(curr_node)) = (
                self.get_node(prev_pos, prev_idx),
                self.get_node(curr_pos, curr_idx),
            ) {
                let conn_cost = if prev_node.right_id == left_id && curr_node.left_id == right_id {
                    new_cost
                } else {
                    dict.connection_cost(prev_node.right_id, curr_node.left_id)
                };

                total_cost += conn_cost as i64 + curr_node.wcost as i64;
            }
        }

        total_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_node_is_bos() {
        let node = DebugNode {
            surface: String::new(),
            start: 0,
            end: 0,
            left_id: 0,
            right_id: 0,
            wcost: 0,
            feature: "BOS/EOS".to_string(),
            is_unknown: false,
            lattice_pos: 0,
            node_idx: 0,
        };
        assert!(node.is_bos());
        assert!(!node.is_eos());
    }

    #[test]
    fn test_debug_node_is_eos() {
        let node = DebugNode {
            surface: String::new(),
            start: 10,
            end: 10,
            left_id: 0,
            right_id: 0,
            wcost: 0,
            feature: "BOS/EOS".to_string(),
            is_unknown: false,
            lattice_pos: 11,
            node_idx: 0,
        };
        assert!(!node.is_bos());
        assert!(node.is_eos());
    }
}
