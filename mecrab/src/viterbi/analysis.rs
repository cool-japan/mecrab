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

#[cfg(test)]
mod tests {
    use super::*;

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
}
