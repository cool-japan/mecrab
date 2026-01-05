//! Lattice visualization in DOT/Graphviz format
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module provides visualization of the lattice structure as a DOT graph,
//! which can be rendered using Graphviz tools (dot, neato, etc.) or online viewers.
//!
//! # Example
//!
//! ```ignore
//! use mecrab::{MeCrab, Lattice};
//!
//! let mecrab = MeCrab::new()?;
//! let lattice = mecrab.build_lattice("東京都")?;
//! let dot = lattice.to_dot()?;
//! println!("{}", dot);
//! // Run: echo "$dot" | dot -Tpng -o lattice.png
//! ```

use std::fmt::Write;

use super::{Lattice, LatticeNode};
use crate::Result;
use crate::dict::Dictionary;

/// Configuration for DOT visualization
#[derive(Debug, Clone)]
pub struct DotConfig {
    /// Graph title (displayed at the top)
    pub title: Option<String>,
    /// Show word costs on nodes
    pub show_wcost: bool,
    /// Show connection costs on edges (requires dictionary)
    pub show_connection_cost: bool,
    /// Show POS ID on nodes
    pub show_pos_id: bool,
    /// Show byte positions on nodes
    pub show_positions: bool,
    /// Highlight unknown words with different color
    pub highlight_unknown: bool,
    /// Show the best path with thick edges (requires path indices)
    pub best_path: Option<Vec<usize>>,
    /// Use left-to-right layout (default: top-to-bottom)
    pub rankdir: RankDir,
    /// Font name for labels
    pub fontname: String,
    /// Node shape
    pub node_shape: NodeShape,
}

impl Default for DotConfig {
    fn default() -> Self {
        Self {
            title: None,
            show_wcost: true,
            show_connection_cost: false,
            show_pos_id: false,
            show_positions: false,
            highlight_unknown: true,
            best_path: None,
            rankdir: RankDir::LeftToRight,
            fontname: "Noto Sans CJK JP".to_string(),
            node_shape: NodeShape::Box,
        }
    }
}

impl DotConfig {
    /// Create a minimal configuration
    pub fn minimal() -> Self {
        Self {
            show_wcost: false,
            show_connection_cost: false,
            show_pos_id: false,
            show_positions: false,
            highlight_unknown: false,
            ..Default::default()
        }
    }

    /// Create a detailed configuration showing all information
    pub fn detailed() -> Self {
        Self {
            show_wcost: true,
            show_connection_cost: true,
            show_pos_id: true,
            show_positions: true,
            highlight_unknown: true,
            ..Default::default()
        }
    }

    /// Set the graph title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the best path to highlight
    #[must_use]
    pub fn with_best_path(mut self, path: Vec<usize>) -> Self {
        self.best_path = Some(path);
        self
    }

    /// Set the layout direction
    #[must_use]
    pub fn with_rankdir(mut self, rankdir: RankDir) -> Self {
        self.rankdir = rankdir;
        self
    }
}

/// Layout direction for the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RankDir {
    /// Left to right (default, good for Japanese text)
    #[default]
    LeftToRight,
    /// Top to bottom
    TopToBottom,
    /// Right to left
    RightToLeft,
    /// Bottom to top
    BottomToTop,
}

impl RankDir {
    fn as_str(self) -> &'static str {
        match self {
            Self::LeftToRight => "LR",
            Self::TopToBottom => "TB",
            Self::RightToLeft => "RL",
            Self::BottomToTop => "BT",
        }
    }
}

/// Node shape for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeShape {
    /// Box shape (default)
    #[default]
    Box,
    /// Rounded box
    RoundedBox,
    /// Ellipse
    Ellipse,
    /// Diamond (for BOS/EOS)
    Diamond,
    /// Circle
    Circle,
}

impl NodeShape {
    fn as_str(self) -> &'static str {
        match self {
            Self::Box => "box",
            Self::RoundedBox => "box,style=rounded",
            Self::Ellipse => "ellipse",
            Self::Diamond => "diamond",
            Self::Circle => "circle",
        }
    }
}

/// Builder for creating DOT output from a lattice
pub struct DotBuilder<'a> {
    lattice: &'a Lattice<'a>,
    dict: Option<&'a Dictionary>,
    config: DotConfig,
}

impl<'a> DotBuilder<'a> {
    /// Create a new DOT builder
    pub fn new(lattice: &'a Lattice<'a>) -> Self {
        Self {
            lattice,
            dict: None,
            config: DotConfig::default(),
        }
    }

    /// Set the dictionary (needed for connection costs)
    #[must_use]
    pub fn with_dict(mut self, dict: &'a Dictionary) -> Self {
        self.dict = Some(dict);
        self
    }

    /// Set the configuration
    #[must_use]
    pub fn with_config(mut self, config: DotConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the DOT string
    ///
    /// # Errors
    ///
    /// Returns an error if formatting fails.
    pub fn build(&self) -> Result<String> {
        let mut output = String::with_capacity(4096);

        // Graph header
        writeln!(output, "digraph lattice {{")?;
        writeln!(output, "  rankdir={};", self.config.rankdir.as_str())?;
        writeln!(output, "  node [fontname=\"{}\"];", self.config.fontname)?;
        writeln!(output, "  edge [fontname=\"{}\"];", self.config.fontname)?;

        if let Some(title) = &self.config.title {
            writeln!(output, "  labelloc=\"t\";")?;
            writeln!(output, "  label=\"{}\";", escape_dot_string(title))?;
        }

        // Generate unique node IDs
        // Structure: node_<pos>_<idx>
        // BOS is always at position 0, EOS at last position

        // Node definitions
        writeln!(output, "\n  // Nodes")?;

        for (pos, nodes) in self.lattice.nodes_at.iter().enumerate() {
            for (idx, node) in nodes.iter().enumerate() {
                let node_id = format!("n{}_{}", pos, idx);
                let label = self.format_node_label(node);
                let style = self.format_node_style(node, pos, self.lattice.nodes_at.len());

                writeln!(
                    output,
                    "  {} [label=\"{}\"{}];",
                    node_id,
                    escape_dot_string(&label),
                    style
                )?;
            }
        }

        // Edge definitions
        writeln!(output, "\n  // Edges")?;

        // For each node (except BOS), find possible predecessors
        for (end_pos, nodes) in self.lattice.nodes_at.iter().enumerate() {
            for (end_idx, node) in nodes.iter().enumerate() {
                let to_id = format!("n{}_{}", end_pos, end_idx);

                // Find predecessors: nodes that end where this node starts
                // BOS is at position 0, so nodes starting at byte 0 connect to BOS
                let start_pos = node.start;

                // Position in nodes_at for predecessors
                // nodes_at[i+1] contains nodes ending at byte position i
                let pred_pos = if end_pos == 0 {
                    continue; // BOS has no predecessors
                } else if node.surface.is_empty() && end_pos == self.lattice.nodes_at.len() - 1 {
                    // EOS - connects from all nodes that end at text.len()
                    end_pos // same position, EOS shares with last real nodes
                } else {
                    start_pos + 1 // nodes_at[start+1] contains nodes ending at start
                };

                // Handle EOS specially
                if node.feature == "BOS/EOS" && end_pos == self.lattice.nodes_at.len() - 1 {
                    // EOS: find all nodes ending at text.len()
                    let text_len = self.lattice.text.len();
                    if text_len + 1 < self.lattice.nodes_at.len() {
                        for (pred_idx, _pred_node) in
                            self.lattice.nodes_at[text_len + 1].iter().enumerate()
                        {
                            let from_id = format!("n{}_{}", text_len + 1, pred_idx);
                            Self::write_edge(&mut output, &from_id, &to_id, None)?;
                        }
                    }
                    // Also connect from position 0 if text is empty
                    if text_len == 0 {
                        Self::write_edge(&mut output, "n0_0", &to_id, None)?;
                    }
                    continue;
                }

                // Regular nodes
                if pred_pos < self.lattice.nodes_at.len() {
                    for (pred_idx, pred_node) in self.lattice.nodes_at[pred_pos].iter().enumerate()
                    {
                        // Check if predecessor actually ends at our start
                        if pred_node.end == start_pos {
                            let from_id = format!("n{}_{}", pred_pos, pred_idx);

                            let conn_cost = if self.config.show_connection_cost {
                                self.dict
                                    .map(|d| d.connection_cost(pred_node.right_id, node.left_id))
                            } else {
                                None
                            };

                            Self::write_edge(&mut output, &from_id, &to_id, conn_cost)?;
                        }
                    }
                }

                // Connect to BOS if node starts at position 0
                if start_pos == 0 && end_pos > 0 {
                    let conn_cost = if self.config.show_connection_cost {
                        // BOS has right_id = 0
                        self.dict.map(|d| d.connection_cost(0, node.left_id))
                    } else {
                        None
                    };
                    Self::write_edge(&mut output, "n0_0", &to_id, conn_cost)?;
                }
            }
        }

        writeln!(output, "}}")?;

        Ok(output)
    }

    fn format_node_label(&self, node: &LatticeNode<'_>) -> String {
        let mut label = String::new();

        // Surface form (or BOS/EOS)
        if node.surface.is_empty() {
            if node.feature == "BOS/EOS" {
                if node.start == 0 {
                    label.push_str("BOS");
                } else {
                    label.push_str("EOS");
                }
            }
        } else {
            label.push_str(node.surface);
        }

        // Additional info
        let mut extras = Vec::new();

        if self.config.show_wcost && !node.surface.is_empty() {
            extras.push(format!("w:{}", node.wcost));
        }

        if self.config.show_pos_id && !node.surface.is_empty() {
            extras.push(format!("pos:{}", node.pos_id));
        }

        if self.config.show_positions && !node.surface.is_empty() {
            extras.push(format!("[{}:{}]", node.start, node.end));
        }

        if !extras.is_empty() {
            write!(label, "\\n{}", extras.join(" ")).unwrap();
        }

        label
    }

    fn format_node_style(&self, node: &LatticeNode<'_>, _pos: usize, _total_pos: usize) -> String {
        let mut styles = Vec::new();

        // BOS/EOS special style
        if node.feature == "BOS/EOS" {
            styles.push("shape=diamond".to_string());
            styles.push("style=filled".to_string());
            styles.push("fillcolor=lightgray".to_string());
        } else {
            // Regular node shape
            styles.push(format!("shape={}", self.config.node_shape.as_str()));

            // Unknown word highlighting
            if self.config.highlight_unknown && node.is_unknown {
                styles.push("style=filled".to_string());
                styles.push("fillcolor=lightyellow".to_string());
            }
        }

        if styles.is_empty() {
            String::new()
        } else {
            format!(", {}", styles.join(", "))
        }
    }

    fn write_edge(
        output: &mut String,
        from: &str,
        to: &str,
        conn_cost: Option<i16>,
    ) -> std::fmt::Result {
        if let Some(cost) = conn_cost {
            writeln!(output, "  {} -> {} [label=\"{}\"];", from, to, cost)
        } else {
            writeln!(output, "  {} -> {};", from, to)
        }
    }
}

/// Escape special characters for DOT string
fn escape_dot_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

impl<'a> Lattice<'a> {
    /// Convert lattice to DOT format with default configuration
    ///
    /// # Errors
    ///
    /// Returns an error if formatting fails.
    pub fn to_dot(&self) -> Result<String> {
        DotBuilder::new(self).build()
    }

    /// Convert lattice to DOT format with custom configuration
    ///
    /// # Errors
    ///
    /// Returns an error if formatting fails.
    pub fn to_dot_with_config(&self, config: DotConfig) -> Result<String> {
        DotBuilder::new(self).with_config(config).build()
    }

    /// Convert lattice to DOT format with dictionary (for connection costs)
    ///
    /// # Errors
    ///
    /// Returns an error if formatting fails.
    pub fn to_dot_with_dict(&self, dict: &'a Dictionary, config: DotConfig) -> Result<String> {
        DotBuilder::new(self)
            .with_dict(dict)
            .with_config(config)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_dot_string() {
        assert_eq!(escape_dot_string("hello"), "hello");
        assert_eq!(escape_dot_string("hello\"world"), "hello\\\"world");
        assert_eq!(escape_dot_string("hello\nworld"), "hello\\nworld");
        assert_eq!(escape_dot_string("back\\slash"), "back\\\\slash");
    }

    #[test]
    fn test_dot_config_default() {
        let config = DotConfig::default();
        assert!(config.show_wcost);
        assert!(!config.show_connection_cost);
        assert_eq!(config.rankdir, RankDir::LeftToRight);
    }

    #[test]
    fn test_dot_config_minimal() {
        let config = DotConfig::minimal();
        assert!(!config.show_wcost);
        assert!(!config.show_pos_id);
    }

    #[test]
    fn test_dot_config_detailed() {
        let config = DotConfig::detailed();
        assert!(config.show_wcost);
        assert!(config.show_connection_cost);
        assert!(config.show_pos_id);
        assert!(config.show_positions);
    }

    #[test]
    fn test_rankdir_as_str() {
        assert_eq!(RankDir::LeftToRight.as_str(), "LR");
        assert_eq!(RankDir::TopToBottom.as_str(), "TB");
        assert_eq!(RankDir::RightToLeft.as_str(), "RL");
        assert_eq!(RankDir::BottomToTop.as_str(), "BT");
    }

    #[test]
    fn test_node_shape_as_str() {
        assert_eq!(NodeShape::Box.as_str(), "box");
        assert_eq!(NodeShape::Ellipse.as_str(), "ellipse");
        assert_eq!(NodeShape::Diamond.as_str(), "diamond");
    }

    #[test]
    fn test_dot_builder_new() {
        // Create a simple lattice manually
        let text = "";
        let nodes_at = vec![vec![LatticeNode::bos()], vec![LatticeNode::eos(0)]];
        let lattice = Lattice { text, nodes_at };

        let builder = DotBuilder::new(&lattice);
        let result = builder.build();
        assert!(result.is_ok());

        let dot = result.unwrap();
        assert!(dot.contains("digraph lattice"));
        assert!(dot.contains("rankdir=LR"));
    }

    #[test]
    fn test_lattice_to_dot() {
        let text = "";
        let nodes_at = vec![vec![LatticeNode::bos()], vec![LatticeNode::eos(0)]];
        let lattice = Lattice { text, nodes_at };

        let result = lattice.to_dot();
        assert!(result.is_ok());
    }
}
