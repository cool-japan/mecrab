//! Python bindings for the corpus tokenization API.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Exposes [`PySurfaceVocab`] so Python callers can build surface-to-ID
//! mappings and use them with
//! [`super::parser::PyMeCrab::tokenize_text`].

use crate::corpus::SurfaceVocab;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Mutex;

// ─────────────────────────────────────────────────────────────────────────────
// PySurfaceVocab
// ─────────────────────────────────────────────────────────────────────────────

/// Maps surface forms to stable integer IDs for corpus building.
///
/// IDs are assigned in insertion order starting at 0.  Thread-safe: uses an
/// internal `Mutex` so the same vocab object can be shared across Python
/// threads without additional locking on the Python side.
///
/// Example::
///
///     vocab = SurfaceVocab()
///     id0 = vocab.get_or_insert("東京")   # => 0
///     id1 = vocab.get_or_insert("大阪")   # => 1
///     id0_again = vocab.get_or_insert("東京")  # => 0  (already present)
///     print(len(vocab))  # => 2
#[pyclass(name = "SurfaceVocab")]
pub struct PySurfaceVocab {
    inner: Mutex<SurfaceVocab>,
}

impl Default for PySurfaceVocab {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PySurfaceVocab {
    /// Create a new, empty vocabulary.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(SurfaceVocab::new()),
        }
    }

    /// Get an existing ID for `surface`, or assign the next available ID.
    ///
    /// Args:
    ///     surface: The surface form to look up or insert.
    ///
    /// Returns:
    ///     The (possibly newly assigned) integer ID for this surface.
    ///
    /// Raises:
    ///     RuntimeError: If the internal lock is poisoned.
    pub fn get_or_insert(&self, surface: &str) -> PyResult<u32> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SurfaceVocab lock poisoned"))
            .map(|mut v| v.get_or_insert(surface))
    }

    /// Look up the ID for `surface` without inserting.
    ///
    /// Args:
    ///     surface: The surface form to look up.
    ///
    /// Returns:
    ///     The integer ID, or None if the surface is not in the vocabulary.
    ///
    /// Raises:
    ///     RuntimeError: If the internal lock is poisoned.
    pub fn get(&self, surface: &str) -> PyResult<Option<u32>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SurfaceVocab lock poisoned"))
            .map(|v| v.get(surface))
    }

    /// Number of distinct surface forms in the vocabulary.
    pub fn __len__(&self) -> PyResult<usize> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SurfaceVocab lock poisoned"))
            .map(|v| v.len())
    }

    fn __repr__(&self) -> PyResult<String> {
        let len = self.__len__()?;
        Ok(format!("SurfaceVocab(size={len})"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_vocab_get_or_insert() {
        let v = PySurfaceVocab::new();
        assert_eq!(v.get_or_insert("東京").unwrap(), 0);
        assert_eq!(v.get_or_insert("大阪").unwrap(), 1);
        assert_eq!(v.get_or_insert("東京").unwrap(), 0);
    }

    #[test]
    fn test_surface_vocab_get() {
        let v = PySurfaceVocab::new();
        assert_eq!(v.get("東京").unwrap(), None);
        v.get_or_insert("東京").unwrap();
        assert_eq!(v.get("東京").unwrap(), Some(0));
    }

    #[test]
    fn test_surface_vocab_len() {
        let v = PySurfaceVocab::new();
        assert_eq!(v.__len__().unwrap(), 0);
        v.get_or_insert("東京").unwrap();
        assert_eq!(v.__len__().unwrap(), 1);
    }

    #[test]
    fn test_surface_vocab_repr() {
        let v = PySurfaceVocab::new();
        assert_eq!(v.__repr__().unwrap(), "SurfaceVocab(size=0)");
        v.get_or_insert("東京").unwrap();
        assert_eq!(v.__repr__().unwrap(), "SurfaceVocab(size=1)");
    }
}
