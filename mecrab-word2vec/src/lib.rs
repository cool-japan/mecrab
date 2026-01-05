//! mecrab-word2vec: Pure Rust Word2Vec implementation
//!
//! Fast, memory-efficient word2vec training optimized for Japanese morphological analysis.
//!
//! # Features
//!
//! - Skip-gram with negative sampling
//! - Multi-threaded training with Rayon
//! - Direct MCV1 format output
//! - Memory-efficient streaming
//!
//! # Example
//!
//! ```no_run
//! use mecrab_word2vec::Word2VecBuilder;
//!
//! let model = Word2VecBuilder::new()
//!     .vector_size(100)
//!     .window_size(5)
//!     .negative_samples(5)
//!     .min_count(10)
//!     .epochs(3)
//!     .threads(8)
//!     .build()?;
//!
//! model.train_from_file("corpus.txt")?;
//! model.save_text("vectors.txt")?;
//! # Ok::<(), anyhow::Error>(())
//! ```

mod io;
mod model;
mod skipgram;
mod trainer;
mod vocab;

pub use model::{Word2Vec, Word2VecBuilder};
pub use vocab::Vocabulary;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Word2VecError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Vocabulary error: {0}")]
    Vocabulary(String),
}

pub type Result<T> = std::result::Result<T, Word2VecError>;
