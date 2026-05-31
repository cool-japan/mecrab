//! Dictionary cost training loop and matrix writer.
//!
//! Provides [`TrainingMatrix`] (a mutable copy of the connection matrix),
//! [`DictTrainConfig`], and [`train_dict`] which runs gradient-descent epochs
//! over an annotated corpus to improve MeCab connection costs.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::{Error, Result};
use crate::dict::{ConnectionMatrix, Dictionary};
use crate::lattice::Lattice;
use crate::viterbi::train::{
    CrfGradient, GoldSegmentation, TrainStepSummary,
    accumulate_batch_gradient, apply_conn_gradient_update,
};

// ── TrainingMatrix ────────────────────────────────────────────────────────────

/// Mutable in-memory copy of the connection matrix for gradient-based training.
///
/// Layout: `data[right_id + lsize * left_id]` — identical to [`ConnectionMatrix::cost`].
pub struct TrainingMatrix {
    data: Vec<i16>,
    lsize: usize,
    rsize: usize,
}

impl TrainingMatrix {
    /// Build a `TrainingMatrix` by copying an existing [`ConnectionMatrix`].
    pub fn from_connection_matrix(cm: &ConnectionMatrix) -> Self {
        Self {
            data: cm.to_vec(),
            lsize: cm.left_size(),
            rsize: cm.right_size(),
        }
    }

    /// Number of left context IDs (stride).
    pub fn lsize(&self) -> usize {
        self.lsize
    }

    /// Number of right context IDs.
    pub fn rsize(&self) -> usize {
        self.rsize
    }

    /// Connection cost using the same formula as [`ConnectionMatrix::cost`].
    pub fn cost(&self, right_id: u16, left_id: u16) -> i16 {
        let rc = right_id as usize;
        let lc = left_id as usize;
        if rc >= self.rsize || lc >= self.lsize {
            return i16::MAX;
        }
        self.data[rc + self.lsize * lc]
    }

    /// Mutable slice over the raw matrix data.
    pub fn as_slice_mut(&mut self) -> &mut [i16] {
        &mut self.data
    }

    /// Apply a [`CrfGradient`] update to this matrix.
    pub fn apply_gradient(&mut self, gradient: &CrfGradient, lr: f64) {
        apply_conn_gradient_update(&mut self.data, self.lsize, gradient, lr);
    }

    /// Serialize to a MeCab-compatible binary `matrix.bin` file.
    ///
    /// Format: `[lsize: u16 LE][rsize: u16 LE][costs: i16 LE × lsize × rsize]`
    pub fn write_binary(&self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| Error::IoError(format!("cannot create {}: {e}", path.display())))?;
        let mut w = BufWriter::new(file);
        w.write_u16::<LittleEndian>(self.lsize as u16)
            .map_err(|e| Error::IoError(format!("{e}")))?;
        w.write_u16::<LittleEndian>(self.rsize as u16)
            .map_err(|e| Error::IoError(format!("{e}")))?;
        for &cost in &self.data {
            w.write_i16::<LittleEndian>(cost)
                .map_err(|e| Error::IoError(format!("{e}")))?;
        }
        w.flush().map_err(|e| Error::IoError(format!("{e}")))?;
        Ok(())
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for dictionary cost training.
#[derive(Debug, Clone)]
pub struct DictTrainConfig {
    /// SGD learning rate (step size). Default: 0.01.
    pub learning_rate: f64,
    /// Number of sentences per gradient update. Default: 64.
    pub batch_size: usize,
    /// Total training epochs. Default: 10.
    pub epochs: usize,
    /// L2 regularization strength (0.0 = disabled). Default: 1e-5.
    pub l2_strength: f64,
    /// Print per-epoch statistics to stderr. Default: false.
    pub verbose: bool,
}

impl Default for DictTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 64,
            epochs: 10,
            l2_strength: 1e-5,
            verbose: false,
        }
    }
}

// ── Summary types ─────────────────────────────────────────────────────────────

/// Per-epoch statistics.
#[derive(Debug, Clone)]
pub struct EpochStats {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Total NLL loss summed across all sentences in this epoch.
    pub loss: f64,
    /// Number of sentences processed.
    pub sentences: usize,
    /// Number of gold morphemes processed.
    pub tokens: usize,
}

/// Summary of a complete training run returned by [`train_dict`].
#[derive(Debug)]
pub struct DictTrainSummary {
    /// Total number of training epochs completed.
    pub total_epochs: usize,
    /// Total number of sentences processed across all epochs.
    pub total_sentences: usize,
    /// Total number of tokens processed across all epochs.
    pub total_tokens: usize,
    /// Per-epoch statistics.
    pub epoch_stats: Vec<EpochStats>,
}

// ── Training loop ─────────────────────────────────────────────────────────────

/// Train connection-matrix costs over `corpus` for `config.epochs` epochs.
///
/// Updates `matrix` in-place. Call [`TrainingMatrix::write_binary`] afterwards
/// to persist the trained matrix to disk.
///
/// # Arguments
/// * `matrix`  — mutable copy of the connection matrix to update
/// * `corpus`  — annotated sentences (see [`GoldSegmentation`])
/// * `dict`    — loaded system dictionary shared by all lattices
/// * `config`  — hyperparameters
pub fn train_dict(
    matrix: &mut TrainingMatrix,
    corpus: &[GoldSegmentation],
    dict: &Dictionary,
    config: &DictTrainConfig,
) -> DictTrainSummary {
    let mut epoch_stats: Vec<EpochStats> = Vec::with_capacity(config.epochs);
    let mut total_sentences = 0_usize;
    let mut total_tokens = 0_usize;

    for epoch in 0..config.epochs {
        let stats = run_epoch(matrix, corpus, dict, config, epoch);
        if config.verbose {
            eprintln!(
                "epoch {}/{}: loss={:.4}  sentences={}  tokens={}",
                epoch + 1,
                config.epochs,
                stats.loss,
                stats.sentences,
                stats.tokens,
            );
        }
        total_sentences += stats.sentences;
        total_tokens += stats.tokens;
        epoch_stats.push(stats);
    }

    DictTrainSummary {
        total_epochs: config.epochs,
        total_sentences,
        total_tokens,
        epoch_stats,
    }
}

fn run_epoch(
    matrix: &mut TrainingMatrix,
    corpus: &[GoldSegmentation],
    dict: &Dictionary,
    config: &DictTrainConfig,
    epoch_idx: usize,
) -> EpochStats {
    let mut total_loss = 0.0_f64;
    let mut total_sentences = 0_usize;
    let mut total_tokens = 0_usize;

    for batch in corpus.chunks(config.batch_size) {
        // Build lattices; skip sentences that fail to parse
        let lattices: Vec<(Lattice<'_>, &GoldSegmentation)> = batch
            .iter()
            .filter_map(|gold| {
                Lattice::build(&gold.text, dict)
                    .ok()
                    .map(|lattice| (lattice, gold))
            })
            .collect();

        if lattices.is_empty() {
            continue;
        }

        // Build reference slice for accumulate_batch_gradient
        let refs: Vec<(&Lattice<'_>, &GoldSegmentation)> =
            lattices.iter().map(|(l, g)| (l, *g)).collect();

        let mut gradient = CrfGradient::new();
        let summary: TrainStepSummary = accumulate_batch_gradient(&refs, dict, &mut gradient);

        // Optional L2 regularization: shrink gradient toward zero
        if config.l2_strength > 0.0 {
            apply_l2(&mut gradient, matrix, config.l2_strength);
        }

        matrix.apply_gradient(&gradient, config.learning_rate);

        total_loss += summary.loss;
        total_sentences += summary.sentence_count;
        total_tokens += summary.token_count;
    }

    EpochStats {
        epoch: epoch_idx,
        loss: total_loss,
        sentences: total_sentences,
        tokens: total_tokens,
    }
}

/// Augment gradient with L2 regularization: push each updated cost toward zero.
fn apply_l2(gradient: &mut CrfGradient, matrix: &TrainingMatrix, l2_strength: f64) {
    for (&(right_id, left_id), grad) in gradient.conn_gradients.iter_mut() {
        let cost = matrix.cost(right_id, left_id) as f64;
        *grad += l2_strength * cost;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viterbi::train::CrfGradient;
    use std::env::temp_dir;

    /// Build a tiny 3×3 TrainingMatrix directly from data for testing.
    fn tiny_matrix(lsize: usize, rsize: usize) -> TrainingMatrix {
        let data = vec![10i16; lsize * rsize];
        TrainingMatrix { data, lsize, rsize }
    }

    #[test]
    fn test_training_matrix_cost_formula() {
        // 2×3 matrix (lsize=2, rsize=3)
        // Layout: data[rc + lsize*lc]
        // (0,0)→idx 0, (1,0)→idx 1, (0,1)→idx 2
        let mut m = tiny_matrix(2, 3);
        m.data[0] = 100; // right=0, left=0
        m.data[1] = 200; // right=1, left=0
        m.data[2] = 300; // right=0, left=1
        assert_eq!(m.cost(0, 0), 100);
        assert_eq!(m.cost(1, 0), 200);
        assert_eq!(m.cost(0, 1), 300);
        assert_eq!(m.cost(99, 0), i16::MAX); // out of range
    }

    #[test]
    fn test_training_matrix_apply_gradient() {
        let mut m = tiny_matrix(2, 2);
        // data[rc + 2*lc]: right=0, left=1 → idx = 0 + 2*1 = 2
        m.data[2] = 50;
        let mut grad = CrfGradient::new();
        grad.add_conn(0, 1, 5.0); // key=(right=0, left=1), gradient=5
        m.apply_gradient(&grad, 1.0);
        // delta = round(1.0 * 5.0) = 5; 50 - 5 = 45
        assert_eq!(m.data[2], 45);
    }

    #[test]
    fn test_training_matrix_write_binary() {
        let mut m = tiny_matrix(2, 3);
        m.data[0] = 42;
        m.data[1] = -7;

        let path = temp_dir().join("mecrab_test_matrix.bin");
        m.write_binary(&path).expect("write_binary failed");

        let bytes = std::fs::read(&path).expect("read failed");
        // lsize=2 (u16 LE) + rsize=3 (u16 LE) + 6 i16 values = 4 + 12 = 16 bytes
        assert_eq!(bytes.len(), 4 + 2 * 3 * 2);
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 2); // lsize
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 3); // rsize
        assert_eq!(i16::from_le_bytes([bytes[4], bytes[5]]), 42);
        assert_eq!(i16::from_le_bytes([bytes[6], bytes[7]]), -7);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_dict_train_config_default() {
        let c = DictTrainConfig::default();
        assert!((c.learning_rate - 0.01).abs() < 1e-9);
        assert_eq!(c.epochs, 10);
        assert_eq!(c.batch_size, 64);
        assert!(!c.verbose);
    }

    #[test]
    fn test_epoch_stats_fields() {
        let s = EpochStats { epoch: 2, loss: 1.5, sentences: 10, tokens: 50 };
        assert_eq!(s.epoch, 2);
        assert!((s.loss - 1.5).abs() < 1e-9);
        assert_eq!(s.sentences, 10);
        assert_eq!(s.tokens, 50);
    }
}
