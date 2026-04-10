//! Internal helper utilities for Python bindings
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ─────────────────────────────────────────────────────────────────────────────
// Probability table helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a Python list of probability dicts from a [`LatticeProbTable`].
///
/// Each entry is a `{"surface": str, "log_prob": f64, "prob": f64}` dict
/// corresponding to the best marginal node at each lattice position.
pub fn build_prob_list<'py>(
    py: Python<'py>,
    prob_table: &crate::viterbi::analysis::LatticeProbTable,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for nm in prob_table.best_per_position() {
        let d = PyDict::new(py);
        d.set_item("surface", nm.surface.as_str())?;
        d.set_item("log_prob", nm.log_prob)?;
        d.set_item("prob", nm.prob)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Produce a JSON-encoded string literal (with surrounding quotes and escaping).
pub fn serde_json_str(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    result.push('"');
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde_json_str() {
        assert_eq!(serde_json_str("hello"), "\"hello\"");
        assert_eq!(serde_json_str("he\"llo"), "\"he\\\"llo\"");
        assert_eq!(serde_json_str("he\\llo"), "\"he\\\\llo\"");
        assert_eq!(serde_json_str("he\nllo"), "\"he\\nllo\"");
    }
}
