//! File I/O for word2vec formats

use crate::Result;
use crate::vocab::Vocabulary;
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Write};
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
