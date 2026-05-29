//! File I/O for word2vec formats

use crate::vocab::Vocabulary;
use crate::{Result, Word2VecError};
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Save embeddings in word2vec text format
///
/// Format:
/// ```text
/// <vocab_size> <vector_size>
/// <word_id> <v1> <v2> ... <vN>
/// ...
/// ```
pub fn save_word2vec_text<P: AsRef<Path>>(
    path: P,
    syn0: &[f32],
    vocab: &Vocabulary,
    vector_size: usize,
) -> Result<()> {
    let path_ref = path.as_ref();
    let file = File::create(path_ref)?;
    let mut writer = BufWriter::new(file);

    // Header: vocab_size vector_size
    writeln!(writer, "{} {}", vocab.len(), vector_size)?;

    // Write each word vector
    for info in vocab.iter() {
        let word_id = info.word_id; // Original MeCab word_id
        let remapped_id = info.remapped_id; // Dense index in syn0
        let offset = remapped_id as usize * vector_size;

        // Ensure we don't go out of bounds
        if offset + vector_size > syn0.len() {
            continue;
        }

        write!(writer, "{}", word_id)?; // Write original word_id

        for i in 0..vector_size {
            write!(writer, " {}", syn0[offset + i])?; // Read from dense array
        }

        writeln!(writer)?;
    }

    writer.flush()?;
    eprintln!("Saved word2vec text format: {:?}", path_ref);
    Ok(())
}

/// Save embeddings in MCV1 binary format
///
/// MCV1 Format:
/// ```
/// Header (32 bytes):
///   [0-3]   Magic: 0x3143564D ("MCV1")
///   [4-7]   vocab_size: u32
///   [8-11]  dim: u32
///   [12-15] dtype: u32 (0=F32)
///   [16-31] reserved: [0; 16]
///
/// Data:
///   Vector data in row-major order
///   vector[word_id][dimension]
/// ```
pub fn save_mcv1_format<P: AsRef<Path>>(
    path: P,
    syn0: &[f32],
    vocab: &Vocabulary,
    vector_size: usize,
    max_word_id: u32,
) -> Result<()> {
    let path_ref = path.as_ref();
    let file = File::create(path_ref)?;
    let mut writer = BufWriter::new(file);

    // Use max_word_id + 1 as vocab size for alignment
    let vocab_size = (max_word_id + 1) as usize;

    eprintln!("Saving MCV1 format:");
    eprintln!(
        "  Vocab size: {} (max_word_id: {})",
        vocab_size, max_word_id
    );
    eprintln!("  Vector size: {}", vector_size);
    eprintln!("  Trained words: {}", vocab.len());

    // Write header
    writer.write_u32::<LittleEndian>(0x3143564D)?; // Magic: "MCV1"
    writer.write_u32::<LittleEndian>(vocab_size as u32)?;
    writer.write_u32::<LittleEndian>(vector_size as u32)?;
    writer.write_u32::<LittleEndian>(0)?; // dtype: F32
    writer.write_all(&[0u8; 16])?; // Reserved

    // Initialize all vectors to zeros
    let zero_vec = vec![0.0f32; vector_size];

    // Write vectors aligned by word_id (MCV1 format uses word_id as index)
    for word_id in 0..vocab_size {
        if let Some(info) = vocab.get(word_id as u32) {
            // This word was trained, write its vector
            let remapped_id = info.remapped_id; // Dense index in syn0
            let offset = remapped_id as usize * vector_size;
            if offset + vector_size <= syn0.len() {
                for i in 0..vector_size {
                    writer.write_f32::<LittleEndian>(syn0[offset + i])?;
                }
            } else {
                // Out of bounds, write zeros
                for &val in &zero_vec {
                    writer.write_f32::<LittleEndian>(val)?;
                }
            }
        } else {
            // Word not in trained vocab, write zeros
            for &val in &zero_vec {
                writer.write_f32::<LittleEndian>(val)?;
            }
        }
    }

    writer.flush()?;

    let file_size = vocab_size * vector_size * 4 + 32;
    eprintln!(
        "  File size: {} bytes ({} MB)",
        file_size,
        file_size / 1024 / 1024
    );
    eprintln!("Saved MCV1 format: {:?}", path_ref);

    Ok(())
}

/// Save FastText subword n-gram embeddings in sparse text format.
///
/// Only non-zero buckets are written to keep the file size manageable.
/// The companion file is loaded back with `load_subword_sparse_text`.
///
/// Format:
/// ```text
/// # mecrab-subword-v1 min_n=<N> max_n=<M> bucket_count=<B> vector_size=<V>
/// # non-zero buckets: <count>
/// <bucket_id> <v1> <v2> ... <vV>
/// ...
/// ```
pub fn save_subword_sparse_text<P: AsRef<Path>>(
    path: P,
    syn_ng: &[f32],
    config: &crate::model::SubwordConfig,
    vector_size: usize,
) -> Result<()> {
    let path_ref = path.as_ref();
    let file = File::create(path_ref)?;
    let mut writer = BufWriter::new(file);

    let bucket_count = config.bucket_count;

    // Count non-zero buckets first so we can write the count in the header.
    let non_zero_count = (0..bucket_count)
        .filter(|&bid| {
            let offset = bid * vector_size;
            if offset + vector_size > syn_ng.len() {
                return false;
            }
            syn_ng[offset..offset + vector_size]
                .iter()
                .map(|v| v.abs())
                .sum::<f32>()
                > 1e-7
        })
        .count();

    // Write header
    writeln!(
        writer,
        "# mecrab-subword-v1 min_n={} max_n={} bucket_count={} vector_size={}",
        config.min_n, config.max_n, bucket_count, vector_size
    )?;
    writeln!(writer, "# non-zero buckets: {non_zero_count}")?;

    // Write non-zero buckets
    for bid in 0..bucket_count {
        let offset = bid * vector_size;
        if offset + vector_size > syn_ng.len() {
            continue;
        }
        let slice = &syn_ng[offset..offset + vector_size];
        let sum_abs: f32 = slice.iter().map(|v| v.abs()).sum();
        if sum_abs <= 1e-7 {
            continue;
        }

        write!(writer, "{bid}")?;
        for &val in slice {
            write!(writer, " {val}")?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

/// Load FastText subword n-gram embeddings from sparse text format.
///
/// Returns `(SubwordConfig, syn_ng_vec)` so the caller can restore both.
///
/// Returns an error if the file header is malformed.
pub fn load_subword_sparse_text<P: AsRef<Path>>(
    path: P,
) -> Result<(crate::model::SubwordConfig, Vec<f32>)> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // --- Parse first header line ---
    let header = lines
        .next()
        .ok_or_else(|| Word2VecError::InvalidParameter("subword file is empty".to_string()))??;

    let mut min_n: Option<usize> = None;
    let mut max_n: Option<usize> = None;
    let mut bucket_count: Option<usize> = None;
    let mut vector_size: Option<usize> = None;

    for token in header.split_whitespace() {
        if let Some(val) = token.strip_prefix("min_n=") {
            min_n = Some(val.parse::<usize>().map_err(|_| {
                Word2VecError::InvalidParameter(format!("cannot parse min_n from '{token}'"))
            })?);
        } else if let Some(val) = token.strip_prefix("max_n=") {
            max_n = Some(val.parse::<usize>().map_err(|_| {
                Word2VecError::InvalidParameter(format!("cannot parse max_n from '{token}'"))
            })?);
        } else if let Some(val) = token.strip_prefix("bucket_count=") {
            bucket_count = Some(val.parse::<usize>().map_err(|_| {
                Word2VecError::InvalidParameter(format!("cannot parse bucket_count from '{token}'"))
            })?);
        } else if let Some(val) = token.strip_prefix("vector_size=") {
            vector_size = Some(val.parse::<usize>().map_err(|_| {
                Word2VecError::InvalidParameter(format!("cannot parse vector_size from '{token}'"))
            })?);
        }
    }

    let min_n =
        min_n.ok_or_else(|| Word2VecError::InvalidParameter("header missing min_n".to_string()))?;
    let max_n =
        max_n.ok_or_else(|| Word2VecError::InvalidParameter("header missing max_n".to_string()))?;
    let bucket_count = bucket_count.ok_or_else(|| {
        Word2VecError::InvalidParameter("header missing bucket_count".to_string())
    })?;
    let vector_size = vector_size
        .ok_or_else(|| Word2VecError::InvalidParameter("header missing vector_size".to_string()))?;

    // --- Skip count comment line (informational only) ---
    lines.next().ok_or_else(|| {
        Word2VecError::InvalidParameter("subword file missing count comment line".to_string())
    })??;

    // --- Allocate output table ---
    let total = bucket_count.checked_mul(vector_size).ok_or_else(|| {
        Word2VecError::InvalidParameter(format!(
            "bucket_count ({bucket_count}) * vector_size ({vector_size}) overflows usize"
        ))
    })?;
    let mut syn_ng = vec![0.0f32; total];

    // --- Parse data lines ---
    for raw_line in lines {
        let line = raw_line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut fields = line.split_whitespace();

        let bucket_id = fields
            .next()
            .ok_or_else(|| {
                Word2VecError::InvalidParameter("data line has no bucket_id field".to_string())
            })?
            .parse::<usize>()
            .map_err(|e| Word2VecError::InvalidParameter(format!("cannot parse bucket_id: {e}")))?;

        if bucket_id >= bucket_count {
            return Err(Word2VecError::InvalidParameter(format!(
                "bucket_id {bucket_id} >= bucket_count {bucket_count}"
            )));
        }

        let offset = bucket_id * vector_size;
        let slot = syn_ng
            .get_mut(offset..offset + vector_size)
            .ok_or_else(|| {
                Word2VecError::InvalidParameter(format!(
                    "bucket_id {bucket_id} out of allocated range"
                ))
            })?;

        let mut count = 0usize;
        for (dst, field) in slot.iter_mut().zip(fields.by_ref()) {
            *dst = field
                .parse::<f32>()
                .map_err(|e| Word2VecError::InvalidParameter(format!("cannot parse float: {e}")))?;
            count += 1;
        }

        if count != vector_size {
            return Err(Word2VecError::InvalidParameter(format!(
                "bucket {bucket_id}: expected {vector_size} floats, got {count}"
            )));
        }
    }

    let config = crate::model::SubwordConfig {
        min_n,
        max_n,
        bucket_count,
    };

    Ok((config, syn_ng))
}
