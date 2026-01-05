//! Error types for MeCrab
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for MeCrab operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for MeCrab operations
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Dictionary not found at the specified path
    #[error("Dictionary not found at: {0}")]
    DictionaryNotFound(PathBuf),

    /// Invalid dictionary format
    #[error("Invalid dictionary format: {0}")]
    InvalidDictionaryFormat(String),

    /// Dictionary file corrupted or truncated
    #[error("Dictionary file corrupted: {0}")]
    CorruptedDictionary(String),

    /// Memory mapping error
    #[error("Memory mapping error: {0}")]
    MmapError(String),

    /// Double-Array Trie lookup failed
    #[error("Trie lookup error: {0}")]
    TrieLookupError(String),

    /// Connection matrix error
    #[error("Connection matrix error: {0}")]
    MatrixError(String),

    /// Character definition error
    #[error("Character definition error: {0}")]
    CharDefError(String),

    /// Unknown word handling error
    #[error("Unknown word error: {0}")]
    UnknownWordError(String),

    /// Lattice construction error
    #[error("Lattice construction error: {0}")]
    LatticeError(String),

    /// Viterbi algorithm error
    #[error("Viterbi algorithm error: {0}")]
    ViterbiError(String),

    /// Encoding error (for EUC-JP dictionaries)
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Default dictionary not found
    #[error("Default dictionary not found. Run 'mecrab init' to download and install IPADIC.")]
    DefaultDictionaryNotFound,

    /// Feature string parsing error
    #[error("Feature parsing error: {0}")]
    FeatureParseError(String),

    /// Format error (for DOT output, etc.)
    #[error("Format error: {0}")]
    FormatError(String),

    /// Vector store error
    #[error("Vector store error: {0}")]
    VectorError(String),
}

impl From<std::fmt::Error> for Error {
    fn from(e: std::fmt::Error) -> Self {
        Error::FormatError(e.to_string())
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::FormatError(e.to_string())
    }
}
