//! N-best path search for Viterbi algorithm
//!
//! Provides functionality to find multiple best segmentation paths,
//! useful for disambiguation and providing alternative analyses.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A path through the lattice with its total cost
#[derive(Debug, Clone)]
pub struct ScoredPath {
    /// Sequence of node indices in the path
    pub path: Vec<usize>,
    /// Total path cost (lower is better)
    pub cost: i64,
    /// Partial path indicator (for incremental search)
    pub partial: bool,
}

impl ScoredPath {
    /// Create a new scored path
    pub fn new(path: Vec<usize>, cost: i64) -> Self {
        Self {
            path,
            cost,
            partial: false,
        }
    }

    /// Create a partial path (for beam search)
    pub fn partial(path: Vec<usize>, cost: i64) -> Self {
        Self {
            path,
            cost,
            partial: true,
        }
    }

    /// Get the last node in the path
    pub fn last_node(&self) -> Option<usize> {
        self.path.last().copied()
    }

    /// Extend this path with a new node
    pub fn extend(&self, node: usize, additional_cost: i64) -> Self {
        let mut new_path = self.path.clone();
        new_path.push(node);
        Self {
            path: new_path,
            cost: self.cost.saturating_add(additional_cost),
            partial: self.partial,
        }
    }
}

impl PartialEq for ScoredPath {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for ScoredPath {}

impl PartialOrd for ScoredPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredPath {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (lower cost = higher priority)
        other.cost.cmp(&self.cost)
    }
}

/// N-best path finder using A* search
#[derive(Debug)]
pub struct NBestSearch {
    /// Maximum number of paths to return
    pub n: usize,
    /// Beam width for pruning (0 = no pruning)
    pub beam_width: usize,
    /// Maximum cost difference from best (0 = no limit)
    pub cost_threshold: i64,
}

impl NBestSearch {
    /// Create a new N-best search with given parameters
    pub fn new(n: usize) -> Self {
        Self {
            n,
            beam_width: 0,
            cost_threshold: 0,
        }
    }

    /// Set beam width for search pruning
    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width;
        self
    }

    /// Set cost threshold for pruning
    pub fn with_cost_threshold(mut self, threshold: i64) -> Self {
        self.cost_threshold = threshold;
        self
    }

    /// Find N-best paths through a cost matrix
    ///
    /// # Arguments
    /// * `node_costs` - Cost of each node
    /// * `edge_costs` - Edge cost function (from, to) -> cost
    /// * `adjacency` - Adjacency list (node -> list of successor nodes)
    /// * `start` - Start node index
    /// * `end` - End node index
    pub fn search<F>(
        &self,
        node_costs: &[i64],
        edge_costs: F,
        adjacency: &[Vec<usize>],
        start: usize,
        end: usize,
    ) -> Vec<ScoredPath>
    where
        F: Fn(usize, usize) -> i64,
    {
        let mut results = Vec::with_capacity(self.n);
        let mut heap = BinaryHeap::new();

        // Initialize with start node
        let initial_cost = node_costs.get(start).copied().unwrap_or(0);
        heap.push(ScoredPath::partial(vec![start], initial_cost));

        let mut best_cost = None;

        while let Some(current) = heap.pop() {
            // Check if we've reached the end
            if current.last_node() == Some(end) {
                // Apply cost threshold pruning
                if let Some(best) = best_cost {
                    if self.cost_threshold > 0 && current.cost > best + self.cost_threshold {
                        continue;
                    }
                } else {
                    best_cost = Some(current.cost);
                }

                results.push(ScoredPath::new(current.path, current.cost));
                if results.len() >= self.n {
                    break;
                }
                continue;
            }

            // Expand to successors
            let last = match current.last_node() {
                Some(n) => n,
                None => continue,
            };

            if let Some(successors) = adjacency.get(last) {
                for &next in successors {
                    let edge_cost = edge_costs(last, next);
                    let node_cost = node_costs.get(next).copied().unwrap_or(0);
                    let new_path = current.extend(next, edge_cost + node_cost);

                    // Apply beam width pruning
                    if self.beam_width > 0 && heap.len() >= self.beam_width {
                        if let Some(worst) = heap.peek() {
                            if new_path.cost > worst.cost {
                                continue;
                            }
                        }
                    }

                    heap.push(new_path);
                }
            }
        }

        results
    }

    /// Find N-best paths using backward search (from end to start)
    pub fn search_backward<F>(
        &self,
        node_costs: &[i64],
        edge_costs: F,
        predecessors: &[Vec<usize>],
        start: usize,
        end: usize,
    ) -> Vec<ScoredPath>
    where
        F: Fn(usize, usize) -> i64,
    {
        let mut results = Vec::with_capacity(self.n);
        let mut heap = BinaryHeap::new();

        // Initialize with end node
        let initial_cost = node_costs.get(end).copied().unwrap_or(0);
        heap.push(ScoredPath::partial(vec![end], initial_cost));

        while let Some(current) = heap.pop() {
            // Check if we've reached the start
            if current.last_node() == Some(start) {
                // Reverse path to get start -> end order
                let mut path = current.path;
                path.reverse();
                results.push(ScoredPath::new(path, current.cost));
                if results.len() >= self.n {
                    break;
                }
                continue;
            }

            // Expand to predecessors
            let last = match current.last_node() {
                Some(n) => n,
                None => continue,
            };

            if let Some(preds) = predecessors.get(last) {
                for &prev in preds {
                    let edge_cost = edge_costs(prev, last);
                    let node_cost = node_costs.get(prev).copied().unwrap_or(0);
                    let new_path = current.extend(prev, edge_cost + node_cost);
                    heap.push(new_path);
                }
            }
        }

        results
    }
}

impl Default for NBestSearch {
    fn default() -> Self {
        Self::new(5)
    }
}

/// Lazy N-best iterator for memory-efficient search
pub struct NBestIter<F>
where
    F: Fn(usize, usize) -> i64,
{
    heap: BinaryHeap<ScoredPath>,
    node_costs: Vec<i64>,
    edge_costs: F,
    adjacency: Vec<Vec<usize>>,
    end: usize,
    count: usize,
    max_count: usize,
}

impl<F> NBestIter<F>
where
    F: Fn(usize, usize) -> i64,
{
    /// Create a new lazy N-best iterator
    pub fn new(
        node_costs: Vec<i64>,
        edge_costs: F,
        adjacency: Vec<Vec<usize>>,
        start: usize,
        end: usize,
        max_count: usize,
    ) -> Self {
        let mut heap = BinaryHeap::new();
        let initial_cost = node_costs.get(start).copied().unwrap_or(0);
        heap.push(ScoredPath::partial(vec![start], initial_cost));

        Self {
            heap,
            node_costs,
            edge_costs,
            adjacency,
            end,
            count: 0,
            max_count,
        }
    }
}

impl<F> Iterator for NBestIter<F>
where
    F: Fn(usize, usize) -> i64,
{
    type Item = ScoredPath;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.max_count {
            return None;
        }

        while let Some(current) = self.heap.pop() {
            if current.last_node() == Some(self.end) {
                self.count += 1;
                return Some(ScoredPath::new(current.path, current.cost));
            }

            let last = match current.last_node() {
                Some(n) => n,
                None => continue,
            };

            if let Some(successors) = self.adjacency.get(last) {
                for &next in successors {
                    let edge_cost = (self.edge_costs)(last, next);
                    let node_cost = self.node_costs.get(next).copied().unwrap_or(0);
                    let new_path = current.extend(next, edge_cost + node_cost);
                    self.heap.push(new_path);
                }
            }
        }

        None
    }
}

/// Path diversity metrics
#[derive(Debug, Clone)]
pub struct PathDiversity {
    /// Average Jaccard distance between paths
    pub avg_jaccard_distance: f64,
    /// Number of unique nodes across all paths
    pub unique_nodes: usize,
    /// Number of unique edges across all paths
    pub unique_edges: usize,
}

impl PathDiversity {
    /// Calculate diversity metrics for a set of paths
    pub fn calculate(paths: &[ScoredPath]) -> Self {
        if paths.is_empty() {
            return Self {
                avg_jaccard_distance: 0.0,
                unique_nodes: 0,
                unique_edges: 0,
            };
        }

        let mut unique_nodes = std::collections::HashSet::new();
        let mut unique_edges = std::collections::HashSet::new();

        for path in paths {
            for &node in &path.path {
                unique_nodes.insert(node);
            }
            for window in path.path.windows(2) {
                unique_edges.insert((window[0], window[1]));
            }
        }

        // Calculate average Jaccard distance
        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..paths.len() {
            for j in (i + 1)..paths.len() {
                let set_i: std::collections::HashSet<_> = paths[i].path.iter().collect();
                let set_j: std::collections::HashSet<_> = paths[j].path.iter().collect();
                let intersection = set_i.intersection(&set_j).count();
                let union = set_i.union(&set_j).count();
                if union > 0 {
                    total_distance += 1.0 - (intersection as f64 / union as f64);
                    count += 1;
                }
            }
        }

        Self {
            avg_jaccard_distance: if count > 0 {
                total_distance / count as f64
            } else {
                0.0
            },
            unique_nodes: unique_nodes.len(),
            unique_edges: unique_edges.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scored_path_ordering() {
        let p1 = ScoredPath::new(vec![0, 1, 2], 100);
        let p2 = ScoredPath::new(vec![0, 1, 2], 50);

        // Lower cost should have higher priority (for min-heap)
        assert!(p2 > p1);
    }

    #[test]
    fn test_scored_path_extend() {
        let p = ScoredPath::new(vec![0, 1], 10);
        let extended = p.extend(2, 5);
        assert_eq!(extended.path, vec![0, 1, 2]);
        assert_eq!(extended.cost, 15);
    }

    #[test]
    fn test_nbest_simple_graph() {
        // Simple graph: 0 -> 1 -> 3
        //               0 -> 2 -> 3
        let node_costs = vec![0, 1, 2, 0];
        let adjacency = vec![
            vec![1, 2], // 0 -> 1, 2
            vec![3],    // 1 -> 3
            vec![3],    // 2 -> 3
            vec![],     // 3 (end)
        ];

        let search = NBestSearch::new(2);
        let paths = search.search(&node_costs, |_, _| 1, &adjacency, 0, 3);

        assert_eq!(paths.len(), 2);
        // Path 0->1->3 should have cost 0+1+1+0+1 = 3
        // Path 0->2->3 should have cost 0+1+2+1+0 = 4
        assert!(paths[0].cost <= paths[1].cost);
    }

    #[test]
    fn test_nbest_with_beam() {
        let node_costs = vec![0; 10];
        let adjacency: Vec<Vec<usize>> = (0..9).map(|i| vec![i + 1]).collect();
        let mut adj = adjacency;
        adj.push(vec![]); // End node

        let search = NBestSearch::new(3).with_beam_width(5);
        let paths = search.search(&node_costs, |_, _| 1, &adj, 0, 9);

        assert!(!paths.is_empty());
    }

    #[test]
    fn test_nbest_backward() {
        let node_costs = vec![0, 1, 2, 0];
        let predecessors = vec![
            vec![],     // 0 has no predecessors
            vec![0],    // 1 <- 0
            vec![0],    // 2 <- 0
            vec![1, 2], // 3 <- 1, 2
        ];

        let search = NBestSearch::new(2);
        let paths = search.search_backward(&node_costs, |_, _| 1, &predecessors, 0, 3);

        assert_eq!(paths.len(), 2);
        // Paths should be in start->end order
        assert_eq!(paths[0].path[0], 0);
        assert_eq!(*paths[0].path.last().unwrap(), 3);
    }

    #[test]
    fn test_path_diversity() {
        let paths = vec![
            ScoredPath::new(vec![0, 1, 2, 3], 10),
            ScoredPath::new(vec![0, 1, 4, 3], 12),
            ScoredPath::new(vec![0, 5, 6, 3], 15),
        ];

        let diversity = PathDiversity::calculate(&paths);
        assert!(diversity.avg_jaccard_distance > 0.0);
        assert!(diversity.unique_nodes >= 4); // At least 0, 3 and some middle nodes
        assert!(diversity.unique_edges >= 3);
    }

    #[test]
    fn test_nbest_iter() {
        let node_costs = vec![0, 1, 2, 0];
        let adjacency = vec![vec![1, 2], vec![3], vec![3], vec![]];

        let iter = NBestIter::new(node_costs, |_, _| 1, adjacency, 0, 3, 2);
        let paths: Vec<_> = iter.collect();

        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_scored_path_partial() {
        let p = ScoredPath::partial(vec![0, 1], 10);
        assert!(p.partial);

        let completed = ScoredPath::new(p.path, p.cost);
        assert!(!completed.partial);
    }

    #[test]
    fn test_empty_diversity() {
        let diversity = PathDiversity::calculate(&[]);
        assert_eq!(diversity.unique_nodes, 0);
        assert_eq!(diversity.unique_edges, 0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_single_path_diversity() {
        let paths = vec![ScoredPath::new(vec![0, 1, 2], 10)];
        let diversity = PathDiversity::calculate(&paths);
        assert_eq!(diversity.avg_jaccard_distance, 0.0);
        assert_eq!(diversity.unique_nodes, 3);
        assert_eq!(diversity.unique_edges, 2);
    }
}
