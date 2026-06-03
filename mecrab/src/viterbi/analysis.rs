//! Cost analysis and profiling for Viterbi algorithm
//!
//! Provides detailed insights into path costs and segmentation decisions.

use std::fmt;

/// Detailed cost breakdown for a single morpheme
#[derive(Debug, Clone)]
pub struct MorphemeCost {
    /// Surface form
    pub surface: String,
    /// Word cost (from dictionary)
    pub word_cost: i16,
    /// Connection cost (from previous morpheme)
    pub connection_cost: i16,
    /// Total cost contribution
    pub total_cost: i32,
    /// Left context ID
    pub left_id: u16,
    /// Right context ID
    pub right_id: u16,
}

impl MorphemeCost {
    /// Create a new morpheme cost entry
    pub fn new(
        surface: String,
        word_cost: i16,
        connection_cost: i16,
        left_id: u16,
        right_id: u16,
    ) -> Self {
        Self {
            surface,
            word_cost,
            connection_cost,
            total_cost: word_cost as i32 + connection_cost as i32,
            left_id,
            right_id,
        }
    }
}

impl fmt::Display for MorphemeCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: wcost={}, ccost={}, total={}",
            self.surface, self.word_cost, self.connection_cost, self.total_cost
        )
    }
}

/// Full path cost analysis
#[derive(Debug, Clone, Default)]
pub struct PathAnalysis {
    /// Cost breakdown per morpheme
    pub morphemes: Vec<MorphemeCost>,
    /// Total path cost
    pub total_cost: i64,
    /// Number of morphemes
    pub morpheme_count: usize,
}

impl PathAnalysis {
    /// Create a new path analysis
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a morpheme to the analysis
    pub fn add_morpheme(&mut self, morpheme: MorphemeCost) {
        self.total_cost += morpheme.total_cost as i64;
        self.morphemes.push(morpheme);
        self.morpheme_count = self.morphemes.len();
    }

    /// Get average cost per morpheme
    pub fn average_cost(&self) -> f64 {
        if self.morpheme_count == 0 {
            0.0
        } else {
            self.total_cost as f64 / self.morpheme_count as f64
        }
    }

    /// Get total word cost
    pub fn total_word_cost(&self) -> i64 {
        self.morphemes.iter().map(|m| m.word_cost as i64).sum()
    }

    /// Get total connection cost
    pub fn total_connection_cost(&self) -> i64 {
        self.morphemes
            .iter()
            .map(|m| m.connection_cost as i64)
            .sum()
    }

    /// Get the morpheme with highest cost
    pub fn highest_cost_morpheme(&self) -> Option<&MorphemeCost> {
        self.morphemes.iter().max_by_key(|m| m.total_cost)
    }

    /// Get the morpheme with lowest cost
    pub fn lowest_cost_morpheme(&self) -> Option<&MorphemeCost> {
        self.morphemes.iter().min_by_key(|m| m.total_cost)
    }

    /// Get word cost to connection cost ratio
    pub fn word_connection_ratio(&self) -> f64 {
        let conn = self.total_connection_cost();
        if conn == 0 {
            f64::INFINITY
        } else {
            self.total_word_cost() as f64 / conn as f64
        }
    }
}

impl fmt::Display for PathAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Path Analysis ({} morphemes):", self.morpheme_count)?;
        writeln!(f, "  Total cost: {}", self.total_cost)?;
        writeln!(f, "  Word cost: {}", self.total_word_cost())?;
        writeln!(f, "  Connection cost: {}", self.total_connection_cost())?;
        writeln!(f, "  Average cost: {:.2}", self.average_cost())?;
        writeln!(f, "\n  Morphemes:")?;
        for m in &self.morphemes {
            writeln!(f, "    {}", m)?;
        }
        Ok(())
    }
}

/// Comparison between two path analyses
#[derive(Debug, Clone)]
pub struct PathComparison {
    /// First path analysis
    pub path1: PathAnalysis,
    /// Second path analysis
    pub path2: PathAnalysis,
    /// Cost difference (path1 - path2)
    pub cost_difference: i64,
    /// Morpheme count difference
    pub morpheme_diff: i32,
}

impl PathComparison {
    /// Compare two paths
    pub fn compare(path1: PathAnalysis, path2: PathAnalysis) -> Self {
        let cost_difference = path1.total_cost - path2.total_cost;
        let morpheme_diff = path1.morpheme_count as i32 - path2.morpheme_count as i32;

        Self {
            path1,
            path2,
            cost_difference,
            morpheme_diff,
        }
    }

    /// Check if path1 is preferred (lower cost)
    pub fn prefer_path1(&self) -> bool {
        self.cost_difference < 0
    }
}

/// Lattice statistics for analysis
#[derive(Debug, Clone, Default)]
pub struct LatticeStats {
    /// Total number of nodes
    pub node_count: usize,
    /// Number of character positions
    pub position_count: usize,
    /// Maximum nodes at any position
    pub max_nodes_at_position: usize,
    /// Average nodes per position
    pub avg_nodes_per_position: f64,
    /// Total candidate paths (estimate)
    pub estimated_paths: u64,
}

impl LatticeStats {
    /// Create new lattice stats
    pub fn new(nodes_per_position: &[usize]) -> Self {
        let node_count: usize = nodes_per_position.iter().sum();
        let position_count = nodes_per_position.len();
        let max_nodes = nodes_per_position.iter().copied().max().unwrap_or(0);

        let avg = if position_count > 0 {
            node_count as f64 / position_count as f64
        } else {
            0.0
        };

        // Estimate number of paths (product of nodes at each position, capped)
        let estimated_paths = nodes_per_position
            .iter()
            .filter(|&&n| n > 0)
            .fold(1u64, |acc, &n| acc.saturating_mul(n as u64));

        Self {
            node_count,
            position_count,
            max_nodes_at_position: max_nodes,
            avg_nodes_per_position: avg,
            estimated_paths,
        }
    }

    /// Calculate lattice density (node_count / position_count)
    pub fn density(&self) -> f64 {
        self.avg_nodes_per_position
    }
}

impl fmt::Display for LatticeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Lattice Statistics:")?;
        writeln!(f, "  Positions: {}", self.position_count)?;
        writeln!(f, "  Total nodes: {}", self.node_count)?;
        writeln!(f, "  Max nodes/position: {}", self.max_nodes_at_position)?;
        writeln!(
            f,
            "  Avg nodes/position: {:.2}",
            self.avg_nodes_per_position
        )?;
        writeln!(f, "  Estimated paths: {}", self.estimated_paths)?;
        Ok(())
    }
}

/// Connection matrix analysis
#[derive(Debug, Clone)]
pub struct ConnectionMatrixStats {
    /// Matrix dimensions (left_size x right_size)
    pub dimensions: (usize, usize),
    /// Total number of entries
    pub entry_count: usize,
    /// Minimum cost in matrix
    pub min_cost: i16,
    /// Maximum cost in matrix
    pub max_cost: i16,
    /// Average cost
    pub avg_cost: f64,
    /// Number of zero entries
    pub zero_count: usize,
    /// Sparsity ratio (zero entries / total)
    pub sparsity: f64,
}

impl ConnectionMatrixStats {
    /// Analyze a connection matrix
    pub fn analyze(matrix: &[i16], left_size: usize, right_size: usize) -> Self {
        let entry_count = matrix.len();
        let min_cost = matrix.iter().copied().min().unwrap_or(0);
        let max_cost = matrix.iter().copied().max().unwrap_or(0);
        let sum: i64 = matrix.iter().map(|&c| c as i64).sum();
        let avg_cost = if entry_count > 0 {
            sum as f64 / entry_count as f64
        } else {
            0.0
        };
        let zero_count = matrix.iter().filter(|&&c| c == 0).count();
        let sparsity = if entry_count > 0 {
            zero_count as f64 / entry_count as f64
        } else {
            0.0
        };

        Self {
            dimensions: (left_size, right_size),
            entry_count,
            min_cost,
            max_cost,
            avg_cost,
            zero_count,
            sparsity,
        }
    }
}

impl fmt::Display for ConnectionMatrixStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Connection Matrix Statistics:")?;
        writeln!(
            f,
            "  Dimensions: {}x{}",
            self.dimensions.0, self.dimensions.1
        )?;
        writeln!(f, "  Entries: {}", self.entry_count)?;
        writeln!(f, "  Cost range: {} to {}", self.min_cost, self.max_cost)?;
        writeln!(f, "  Average cost: {:.2}", self.avg_cost)?;
        writeln!(f, "  Sparsity: {:.2}%", self.sparsity * 100.0)?;
        Ok(())
    }
}

/// Aggregate analysis for multiple segmentations
#[derive(Debug, Clone, Default)]
pub struct SegmentationReport {
    /// Number of texts analyzed
    pub text_count: usize,
    /// Total morphemes produced
    pub total_morphemes: usize,
    /// Total characters processed
    pub total_chars: usize,
    /// Sum of all costs
    pub total_cost: i64,
    /// Distribution of morpheme counts
    pub morpheme_distribution: Vec<usize>,
}

impl SegmentationReport {
    /// Create a new report
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a segmentation result
    pub fn record(&mut self, char_count: usize, morpheme_count: usize, cost: i64) {
        self.text_count += 1;
        self.total_morphemes += morpheme_count;
        self.total_chars += char_count;
        self.total_cost += cost;

        // Update distribution
        while self.morpheme_distribution.len() <= morpheme_count {
            self.morpheme_distribution.push(0);
        }
        self.morpheme_distribution[morpheme_count] += 1;
    }

    /// Get average morphemes per text
    pub fn avg_morphemes_per_text(&self) -> f64 {
        if self.text_count == 0 {
            0.0
        } else {
            self.total_morphemes as f64 / self.text_count as f64
        }
    }

    /// Get average characters per morpheme
    pub fn avg_chars_per_morpheme(&self) -> f64 {
        if self.total_morphemes == 0 {
            0.0
        } else {
            self.total_chars as f64 / self.total_morphemes as f64
        }
    }

    /// Get average cost per morpheme
    pub fn avg_cost_per_morpheme(&self) -> f64 {
        if self.total_morphemes == 0 {
            0.0
        } else {
            self.total_cost as f64 / self.total_morphemes as f64
        }
    }
}

impl fmt::Display for SegmentationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Segmentation Report:")?;
        writeln!(f, "  Texts analyzed: {}", self.text_count)?;
        writeln!(f, "  Total morphemes: {}", self.total_morphemes)?;
        writeln!(f, "  Total characters: {}", self.total_chars)?;
        writeln!(
            f,
            "  Avg morphemes/text: {:.2}",
            self.avg_morphemes_per_text()
        )?;
        writeln!(
            f,
            "  Avg chars/morpheme: {:.2}",
            self.avg_chars_per_morpheme()
        )?;
        writeln!(
            f,
            "  Avg cost/morpheme: {:.2}",
            self.avg_cost_per_morpheme()
        )?;
        Ok(())
    }
}

/// Per-node marginal probability after forward-backward computation.
///
/// Represents P(node | input) — the probability that this particular morpheme
/// appears in the correct segmentation of the input, marginalising over all
/// other possible segmentations.
#[derive(Debug, Clone)]
pub struct NodeMarginal {
    /// Surface form of this node
    pub surface: String,
    /// Feature string
    pub feature: String,
    /// Start byte position in input text
    pub start: usize,
    /// End byte position in input text
    pub end: usize,
    /// Log marginal probability (natural log, will be ≤ 0)
    pub log_prob: f64,
    /// Marginal probability (exp of log_prob, in \[0,1\])
    pub prob: f64,
}

/// Result of the forward-backward algorithm: per-position node marginals.
///
/// For each position in the input, stores all active nodes and their
/// marginal probabilities P(node | input), computed via the
/// forward-backward algorithm over the Viterbi lattice.
///
/// These marginals can be used for:
/// - Subword regularization in LLM pre-training
/// - Uncertainty estimation in morphological disambiguation
/// - Lattice-based sequence labeling
#[derive(Debug, Clone, Default)]
pub struct LatticeProbTable {
    /// Marginals indexed by byte position (end position index in lattice,
    /// matching how nodes are stored in the lattice's `nodes_at` array).
    /// Each entry is a list of nodes at that lattice position (i.e., ending there).
    pub by_position: Vec<Vec<NodeMarginal>>,
    /// Total number of input bytes (for bounds checking)
    pub input_len: usize,
    /// Log-partition function Z = log(sum of exp(-cost/T) over all paths)
    pub log_z: f64,
}

/// Summary scores for a parsed text: statistical certainty and segmentation cost.
#[derive(Debug, Clone)]
pub struct TextScore {
    /// Viterbi cost (lower = better fit to dictionary costs)
    pub viterbi_cost: i64,
    /// Segmentation perplexity from forward-backward (lower = more certain)
    pub perplexity: f64,
    /// Segmentation entropy in nats (lower = less ambiguous)
    pub entropy: f64,
    /// Morpheme count (from Viterbi path)
    pub morpheme_count: usize,
    /// Count of out-of-vocabulary (unknown word) tokens
    pub oov_count: usize,
}

impl LatticeProbTable {
    /// Partition function perplexity from forward-backward marginals.
    ///
    /// Perplexity of the model = exp(-mean_log_marginal_per_position).
    /// For each non-trivial position (has at least one node with prob > 0),
    /// takes the marginal of the highest-probability node p_max and accumulates
    /// -ln(p_max). The mean over all live positions is then exponentiated.
    ///
    /// Returns 1.0 for an empty table (degenerate case: perfect certainty).
    pub fn perplexity(&self) -> f64 {
        let live_positions: Vec<&Vec<NodeMarginal>> = self
            .by_position
            .iter()
            .filter(|nodes| nodes.iter().any(|n| n.prob > 0.0))
            .collect();

        if live_positions.is_empty() {
            return 1.0;
        }

        let total_neg_log: f64 = live_positions
            .iter()
            .map(|nodes| {
                // p_max at this position
                let p_max = nodes.iter().map(|n| n.prob).fold(0.0_f64, f64::max);
                if p_max <= 0.0 { 0.0 } else { -p_max.ln() }
            })
            .sum();

        let mean = total_neg_log / live_positions.len() as f64;
        mean.exp()
    }

    /// Segmentation entropy in nats: H = -sum_{position,node} marginal * ln(marginal).
    ///
    /// Measures uncertainty in the segmentation. Zero for a unique (unambiguous) sentence.
    /// Only counts positions that are "live" (have at least one prob > 0).
    /// By convention, 0 * ln(0) = 0 (skipped).
    pub fn segmentation_entropy(&self) -> f64 {
        self.by_position
            .iter()
            .filter(|nodes| nodes.iter().any(|n| n.prob > 0.0))
            .flat_map(|nodes| nodes.iter())
            .filter(|n| n.prob > 0.0)
            .map(|n| -n.prob * n.prob.ln())
            .sum()
    }

    /// Estimate morpheme count as number of positions where at least one node
    /// has `prob >= 0.5` (heuristic for on-path membership).
    pub fn morpheme_count_estimate(&self) -> usize {
        self.by_position
            .iter()
            .filter(|nodes| nodes.iter().any(|n| n.prob >= 0.5))
            .count()
    }

    /// Return all `NodeMarginal`s for the highest-probability nodes
    /// at each position (i.e., for each lattice position, the node with the
    /// highest marginal probability).
    ///
    /// These correspond roughly to the Viterbi path's nodes.
    pub fn best_per_position(&self) -> Vec<&NodeMarginal> {
        self.by_position
            .iter()
            .filter_map(|nodes| {
                nodes.iter().max_by(|a, b| {
                    a.log_prob
                        .partial_cmp(&b.log_prob)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .collect()
    }

    /// Return all nodes across all positions, sorted by descending marginal probability.
    pub fn all_nodes_sorted(&self) -> Vec<&NodeMarginal> {
        let mut all: Vec<&NodeMarginal> = self.by_position.iter().flatten().collect();
        all.sort_by(|a, b| {
            b.log_prob
                .partial_cmp(&a.log_prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_marginal_fields() {
        let nm = NodeMarginal {
            surface: "東京".to_string(),
            feature: "名詞,固有名詞".to_string(),
            start: 0,
            end: 6,
            log_prob: -0.5_f64,
            prob: (-0.5_f64).exp(),
        };
        assert_eq!(nm.surface, "東京");
        assert_eq!(nm.start, 0);
        assert_eq!(nm.end, 6);
        assert!(nm.prob <= 1.0 && nm.prob >= 0.0);
    }

    #[test]
    fn test_lattice_prob_table_default() {
        let table = LatticeProbTable::default();
        assert!(table.by_position.is_empty());
        assert_eq!(table.input_len, 0);
        assert!(table.log_z.abs() < f64::EPSILON);
    }

    #[test]
    fn test_best_per_position_empty() {
        let table = LatticeProbTable::default();
        let best = table.best_per_position();
        assert!(best.is_empty());
    }

    #[test]
    fn test_best_per_position_selects_max_prob() {
        let nm_high = NodeMarginal {
            surface: "東京".to_string(),
            feature: "名詞,固有名詞".to_string(),
            start: 0,
            end: 6,
            log_prob: -0.1_f64,
            prob: 0.9,
        };
        let nm_low = NodeMarginal {
            surface: "東".to_string(),
            feature: "名詞,一般".to_string(),
            start: 0,
            end: 3,
            log_prob: -2.0_f64,
            prob: 0.135,
        };
        let table = LatticeProbTable {
            by_position: vec![vec![nm_high, nm_low]],
            input_len: 6,
            log_z: 0.0,
        };
        let best = table.best_per_position();
        assert_eq!(best.len(), 1);
        assert_eq!(best[0].surface, "東京");
    }

    #[test]
    fn test_all_nodes_sorted() {
        let nm_a = NodeMarginal {
            surface: "a".to_string(),
            feature: String::new(),
            start: 0,
            end: 1,
            log_prob: -1.0_f64,
            prob: 0.368,
        };
        let nm_b = NodeMarginal {
            surface: "b".to_string(),
            feature: String::new(),
            start: 0,
            end: 1,
            log_prob: -0.1_f64,
            prob: 0.905,
        };
        let table = LatticeProbTable {
            by_position: vec![vec![nm_a, nm_b]],
            input_len: 1,
            log_z: 0.0,
        };
        let sorted = table.all_nodes_sorted();
        assert_eq!(sorted.len(), 2);
        // Sorted descending: first should be the higher prob node ("b")
        assert_eq!(sorted[0].surface, "b");
        assert_eq!(sorted[1].surface, "a");
    }

    #[test]
    fn test_morpheme_cost() {
        let cost = MorphemeCost::new("東京".to_string(), 100, 50, 1, 2);
        assert_eq!(cost.total_cost, 150);
        assert!(cost.to_string().contains("東京"));
    }

    #[test]
    fn test_path_analysis() {
        let mut analysis = PathAnalysis::new();
        analysis.add_morpheme(MorphemeCost::new("東".to_string(), 100, 50, 1, 2));
        analysis.add_morpheme(MorphemeCost::new("京".to_string(), 80, 30, 2, 3));

        assert_eq!(analysis.morpheme_count, 2);
        assert_eq!(analysis.total_cost, 260);
        assert_eq!(analysis.total_word_cost(), 180);
        assert_eq!(analysis.total_connection_cost(), 80);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_path_analysis_empty() {
        let analysis = PathAnalysis::new();
        assert_eq!(analysis.average_cost(), 0.0);
        assert!(analysis.highest_cost_morpheme().is_none());
    }

    #[test]
    fn test_path_comparison() {
        let mut path1 = PathAnalysis::new();
        path1.add_morpheme(MorphemeCost::new("東京".to_string(), 100, 50, 1, 2));

        let mut path2 = PathAnalysis::new();
        path2.add_morpheme(MorphemeCost::new("東".to_string(), 80, 30, 1, 2));
        path2.add_morpheme(MorphemeCost::new("京".to_string(), 80, 30, 2, 3));

        let comparison = PathComparison::compare(path1, path2);
        assert_eq!(comparison.morpheme_diff, -1); // path1 has 1 fewer morpheme
    }

    #[test]
    fn test_lattice_stats() {
        let nodes_per_pos = vec![2, 3, 1, 4, 2];
        let stats = LatticeStats::new(&nodes_per_pos);

        assert_eq!(stats.node_count, 12);
        assert_eq!(stats.position_count, 5);
        assert_eq!(stats.max_nodes_at_position, 4);
        assert!(stats.density() > 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_lattice_stats_empty() {
        let stats = LatticeStats::new(&[]);
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.density(), 0.0);
    }

    #[test]
    fn test_connection_matrix_stats() {
        let matrix = vec![0i16, 10, -5, 20, 0, 15];
        let stats = ConnectionMatrixStats::analyze(&matrix, 2, 3);

        assert_eq!(stats.dimensions, (2, 3));
        assert_eq!(stats.min_cost, -5);
        assert_eq!(stats.max_cost, 20);
        assert_eq!(stats.zero_count, 2);
    }

    #[test]
    fn test_segmentation_report() {
        let mut report = SegmentationReport::new();
        report.record(10, 5, 100);
        report.record(20, 8, 200);

        assert_eq!(report.text_count, 2);
        assert_eq!(report.total_morphemes, 13);
        assert_eq!(report.total_chars, 30);
        assert!((report.avg_morphemes_per_text() - 6.5).abs() < 0.01);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_segmentation_report_empty() {
        let report = SegmentationReport::new();
        assert_eq!(report.avg_morphemes_per_text(), 0.0);
        assert_eq!(report.avg_chars_per_morpheme(), 0.0);
    }

    #[test]
    fn test_morpheme_distribution() {
        let mut report = SegmentationReport::new();
        report.record(5, 2, 50);
        report.record(10, 2, 100);
        report.record(8, 3, 80);

        assert_eq!(report.morpheme_distribution[2], 2);
        assert_eq!(report.morpheme_distribution[3], 1);
    }

    #[test]
    fn test_perplexity_empty_table() {
        let table = LatticeProbTable::default();
        assert!((table.perplexity() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_entropy_empty_table() {
        let table = LatticeProbTable::default();
        assert!(table.segmentation_entropy().abs() < 1e-9);
    }

    #[test]
    fn test_perplexity_certain_segmentation() {
        // Single position with prob = 1.0 → -ln(1.0) = 0.0 → exp(0.0) = 1.0
        let mut table = LatticeProbTable::default();
        table.by_position.push(vec![NodeMarginal {
            surface: "テスト".to_string(),
            feature: String::new(),
            start: 0,
            end: 9,
            log_prob: 0.0,
            prob: 1.0,
        }]);
        assert!((table.perplexity() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy_certain_segmentation() {
        // prob = 1.0 → -1.0 * ln(1.0) = 0.0
        let mut table = LatticeProbTable::default();
        table.by_position.push(vec![NodeMarginal {
            surface: "確実".to_string(),
            feature: String::new(),
            start: 0,
            end: 6,
            log_prob: 0.0,
            prob: 1.0,
        }]);
        assert!(table.segmentation_entropy().abs() < 1e-9);
    }

    #[test]
    fn test_perplexity_uniform_two_choices() {
        // Two nodes each with prob = 0.5 → p_max = 0.5 → -ln(0.5) ≈ 0.693 → exp ≈ 2.0
        let mut table = LatticeProbTable::default();
        table.by_position.push(vec![
            NodeMarginal {
                surface: "東京".to_string(),
                feature: String::new(),
                start: 0,
                end: 6,
                log_prob: -0.693_f64,
                prob: 0.5,
            },
            NodeMarginal {
                surface: "東".to_string(),
                feature: String::new(),
                start: 0,
                end: 3,
                log_prob: -0.693_f64,
                prob: 0.5,
            },
        ]);
        let perp = table.perplexity();
        assert!((perp - 2.0).abs() < 0.01, "expected ~2.0, got {perp}");
    }

    #[test]
    fn test_morpheme_count_estimate() {
        let mut table = LatticeProbTable::default();
        // Position 0: one node with prob >= 0.5 → counts
        table.by_position.push(vec![NodeMarginal {
            surface: "東京".to_string(),
            feature: String::new(),
            start: 0,
            end: 6,
            log_prob: -0.1_f64,
            prob: 0.9,
        }]);
        // Position 1: all below 0.5 → does not count
        table.by_position.push(vec![NodeMarginal {
            surface: "は".to_string(),
            feature: String::new(),
            start: 6,
            end: 9,
            log_prob: -1.5_f64,
            prob: 0.22,
        }]);
        assert_eq!(table.morpheme_count_estimate(), 1);
    }

    #[test]
    fn test_text_score_fields() {
        let score = TextScore {
            viterbi_cost: 1234,
            perplexity: 1.5,
            entropy: 0.7,
            morpheme_count: 3,
            oov_count: 0,
        };
        assert_eq!(score.viterbi_cost, 1234);
        assert!((score.perplexity - 1.5).abs() < 1e-9);
        assert_eq!(score.oov_count, 0);
    }
}
