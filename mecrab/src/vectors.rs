//! Zero-copy word vector storage and operations
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Implements memory-mapped word embeddings with zero-copy access.
//! Based on architecture design in docs/007.md.
//!
//! # Binary Format
//!
//! ```text
//! [ Header (32 bytes) ]
//! [ Vector 0 (dim * 4 bytes) ]
//! [ Vector 1 (dim * 4 bytes) ]
//! ...
//! [ Vector N ]
//! ```

use crate::{Error, Result};
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::fs::File;
use std::sync::Arc;

/// Magic number "MCV1" for vectors.bin format
const MAGIC: u32 = 0x3143_564D; // "MCV1" in little-endian

/// Data type identifier for vector elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VectorDataType {
    /// 32-bit floating point (4 bytes per element)
    F32 = 0,
    /// 16-bit floating point (2 bytes per element)
    F16 = 1,
    /// 8-bit integer (1 byte per element, quantized)
    I8 = 2,
}

impl VectorDataType {
    /// Size in bytes per vector element
    pub const fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
        }
    }

    /// Convert from u32
    fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::I8),
            _ => None,
        }
    }
}

/// Vector store header (32 bytes, C-compatible layout)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct VectorHeader {
    /// Magic number "MCV1"
    magic: u32,
    /// Number of words in vocabulary
    vocab_size: u32,
    /// Vector dimensionality (e.g., 300)
    dim: u32,
    /// Data type (0=f32, 1=f16, 2=i8)
    data_type: u32,
    /// Reserved for future use
    reserved: [u32; 4],
}

// Safety: VectorHeader is repr(C) with only u32 fields
unsafe impl Pod for VectorHeader {}
unsafe impl Zeroable for VectorHeader {}

/// Zero-copy word vector store
///
/// Provides fast access to pre-trained word embeddings without parsing or copying.
/// Vectors are accessed directly from memory-mapped file.
pub struct VectorStore {
    /// Memory-mapped file
    _mmap: Arc<Mmap>,
    /// Pointer to vector data (after header)
    data_ptr: *const u8,
    /// Vector dimensionality
    dim: usize,
    /// Vocabulary size
    vocab_size: usize,
    /// Data type of vectors
    data_type: VectorDataType,
}

// Safety: The data_ptr points to immutable memory-mapped data
unsafe impl Send for VectorStore {}
unsafe impl Sync for VectorStore {}

impl std::fmt::Debug for VectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStore")
            .field("vocab_size", &self.vocab_size)
            .field("dim", &self.dim)
            .field("data_type", &self.data_type)
            .finish()
    }
}

impl VectorStore {
    /// Header size in bytes
    const HEADER_SIZE: usize = 32;

    /// Load vector store from memory-mapped file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be opened or mapped
    /// - File is too small for header
    /// - Header magic number is invalid
    /// - Data type is unsupported
    /// - File size doesn't match expected size
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self> {
        let data = &mmap[..];

        // Validate minimum size
        if data.len() < Self::HEADER_SIZE {
            return Err(Error::VectorError(
                "Vector file too small for header".to_string(),
            ));
        }

        // Read and validate header
        let header: VectorHeader = bytemuck::pod_read_unaligned(&data[0..Self::HEADER_SIZE]);

        if header.magic != MAGIC {
            return Err(Error::VectorError(format!(
                "Invalid magic number: expected 0x{:08X}, got 0x{:08X}",
                MAGIC, header.magic
            )));
        }

        let data_type = VectorDataType::from_u32(header.data_type).ok_or_else(|| {
            Error::VectorError(format!("Invalid data type: {}", header.data_type))
        })?;

        let vocab_size = header.vocab_size as usize;
        let dim = header.dim as usize;

        // Validate file size
        let element_size = data_type.element_size();
        let expected_size = Self::HEADER_SIZE + (vocab_size * dim * element_size);

        if data.len() != expected_size {
            return Err(Error::VectorError(format!(
                "Vector file size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            )));
        }

        // Get pointer to vector data
        let data_ptr = data[Self::HEADER_SIZE..].as_ptr();

        Ok(Self {
            _mmap: mmap,
            data_ptr,
            dim,
            vocab_size,
            data_type,
        })
    }

    /// Load vector store from file path
    ///
    /// # Errors
    ///
    /// Returns an error if file cannot be opened or is invalid.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::VectorError(e.to_string()))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| Error::VectorError(e.to_string()))? };
        Self::from_mmap(Arc::new(mmap))
    }

    /// Get vector by word ID (zero-copy)
    ///
    /// Returns a slice directly referencing the memory-mapped data.
    /// No copying occurs.
    ///
    /// # Arguments
    ///
    /// * `word_id` - Index of the word in vocabulary
    ///
    /// # Returns
    ///
    /// `None` if word_id is out of bounds, otherwise a slice of f32 values.
    #[inline]
    pub fn get(&self, word_id: u32) -> Option<&[f32]> {
        if self.data_type != VectorDataType::F32 {
            // For now, only support F32
            return None;
        }

        let word_id = word_id as usize;
        if word_id >= self.vocab_size {
            return None;
        }

        let element_size = self.data_type.element_size();
        let start = word_id * self.dim * element_size;

        // Safety: We validated bounds and data type
        unsafe {
            let slice =
                std::slice::from_raw_parts(self.data_ptr.add(start), self.dim * element_size);
            Some(bytemuck::cast_slice(slice))
        }
    }

    /// Get vector dimensionality
    #[inline]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Get vocabulary size
    #[inline]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get data type
    #[inline]
    pub const fn data_type(&self) -> VectorDataType {
        self.data_type
    }

    /// Compute mean pooling of multiple vectors (sentence embedding)
    ///
    /// Takes the average of all input vectors to create a single
    /// sentence-level representation.
    ///
    /// # Arguments
    ///
    /// * `word_ids` - IDs of words to average
    ///
    /// # Returns
    ///
    /// Mean-pooled vector, or None if no valid vectors found.
    pub fn mean_pooling(&self, word_ids: &[u32]) -> Option<Vec<f32>> {
        let mut sum = vec![0.0_f32; self.dim];
        let mut count = 0;

        for &word_id in word_ids {
            if let Some(vec) = self.get(word_id) {
                for (i, &val) in vec.iter().enumerate() {
                    sum[i] += val;
                }
                count += 1;
            }
        }

        if count == 0 {
            return None;
        }

        let count_f32 = count as f32;
        for val in sum.iter_mut() {
            *val /= count_f32;
        }

        Some(sum)
    }

    /// Compute cosine similarity between two vectors
    ///
    /// # Arguments
    ///
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    ///
    /// Cosine similarity in range [-1.0, 1.0], or None if vectors are zero.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let mut dot = 0.0_f32;
        let mut norm_a = 0.0_f32;
        let mut norm_b = 0.0_f32;

        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-10 {
            return None;
        }

        Some(dot / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_data_type() {
        assert_eq!(VectorDataType::F32.element_size(), 4);
        assert_eq!(VectorDataType::F16.element_size(), 2);
        assert_eq!(VectorDataType::I8.element_size(), 1);

        assert_eq!(VectorDataType::from_u32(0), Some(VectorDataType::F32));
        assert_eq!(VectorDataType::from_u32(1), Some(VectorDataType::F16));
        assert_eq!(VectorDataType::from_u32(2), Some(VectorDataType::I8));
        assert_eq!(VectorDataType::from_u32(99), None);
    }

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<VectorHeader>(), 32);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((VectorStore::cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 1e-6);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((VectorStore::cosine_similarity(&a, &b).unwrap() - 0.0).abs() < 1e-6);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((VectorStore::cosine_similarity(&a, &b).unwrap() + 1.0).abs() < 1e-6);
    }
}
