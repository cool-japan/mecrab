//! Python bindings for the reranker API.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Exposes [`NullReranker`] and [`CostReranker`] to Python so callers can
//! pass a reranker name string to [`super::parser::PyMeCrab::parse_nbest_reranked`].

use pyo3::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// PyNullReranker
// ─────────────────────────────────────────────────────────────────────────────

/// Always selects the Viterbi-optimal path (index 0). Zero overhead.
///
/// This is the default reranker and adds no overhead to N-best search.
/// Because the Viterbi algorithm already returns paths sorted by cost,
/// index 0 is always the globally optimal path.
#[pyclass(name = "NullReranker")]
#[derive(Default)]
pub struct PyNullReranker;

#[pymethods]
impl PyNullReranker {
    /// Create a new NullReranker.
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        "NullReranker"
    }

    /// Always returns 0 — the Viterbi-optimal path.
    ///
    /// Args:
    ///     costs: List of i64 candidate costs (unused).
    ///
    /// Returns:
    ///     Always 0.
    pub fn rerank(&self, _costs: Vec<i64>) -> usize {
        0
    }

    fn __repr__(&self) -> &'static str {
        "NullReranker()"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyCostReranker
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the candidate with the lowest total Viterbi cost.
///
/// Scans all candidates and returns the index of the minimum-cost entry.
/// Produces the same result as `NullReranker` when candidates are already
/// sorted by the Viterbi solver, but can differ with a custom unsorted list.
#[pyclass(name = "CostReranker")]
#[derive(Default)]
pub struct PyCostReranker;

#[pymethods]
impl PyCostReranker {
    /// Create a new CostReranker.
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        "CostReranker"
    }

    /// Return the index of the candidate with the lowest cost.
    ///
    /// Args:
    ///     costs: List of i64 candidate costs.
    ///
    /// Returns:
    ///     Index of the minimum-cost candidate (0 when costs is empty).
    pub fn rerank(&self, costs: Vec<i64>) -> usize {
        costs
            .iter()
            .enumerate()
            .min_by_key(|&(_, c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn __repr__(&self) -> &'static str {
        "CostReranker()"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_reranker_always_zero() {
        let r = PyNullReranker::new();
        assert_eq!(r.rerank(vec![100, 50, 200]), 0);
        assert_eq!(r.rerank(vec![]), 0);
    }

    #[test]
    fn test_cost_reranker_picks_min() {
        let r = PyCostReranker::new();
        assert_eq!(r.rerank(vec![100, 50, 200]), 1);
        assert_eq!(r.rerank(vec![]), 0);
    }

    #[test]
    fn test_null_reranker_name() {
        assert_eq!(PyNullReranker::new().name(), "NullReranker");
    }

    #[test]
    fn test_cost_reranker_name() {
        assert_eq!(PyCostReranker::new().name(), "CostReranker");
    }
}
