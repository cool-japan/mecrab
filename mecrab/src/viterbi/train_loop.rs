//! Dictionary cost training loop and matrix writer.
//!
//! Provides [`TrainingMatrix`] (a mutable copy of the connection matrix),
//! [`DictTrainConfig`], and [`train_dict`] which runs gradient-descent epochs
//! over an annotated corpus to improve MeCab connection costs.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::{Error, Result};
use crate::dict::{ConnectionMatrix, Dictionary};
use crate::lattice::Lattice;
use crate::viterbi::train::{
    CrfGradient, GoldSegmentation, TrainStepSummary,
    accumulate_batch_gradient, apply_conn_gradient_update, apply_word_gradient_update,
};

// ── TrainingMatrix ────────────────────────────────────────────────────────────

/// Mutable in-memory copy of the connection matrix for gradient-based training.
///
/// Layout: `data[right_id + lsize * left_id]` — identical to [`ConnectionMatrix::cost`].
pub struct TrainingMatrix {
    data: Vec<i16>,
    lsize: usize,
    rsize: usize,
    /// Accumulated word-cost deltas in f64 precision (word_id → total delta).
    ///
    /// Negative delta = decrease word cost = favour this word.
    /// Accumulated across all gradient updates in a training run; rounded to
    /// i16 at persistence time via [`word_cost_deltas_i16`].
    pub word_cost_deltas: HashMap<u32, f64>,
}

impl TrainingMatrix {
    /// Build a `TrainingMatrix` by copying an existing [`ConnectionMatrix`].
    pub fn from_connection_matrix(cm: &ConnectionMatrix) -> Self {
        Self {
            data: cm.to_vec(),
            lsize: cm.left_size(),
            rsize: cm.right_size(),
            word_cost_deltas: HashMap::new(),
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

    /// Apply a [`CrfGradient`] connection-cost update to this matrix.
    pub fn apply_gradient(&mut self, gradient: &CrfGradient, lr: f64) {
        apply_conn_gradient_update(&mut self.data, self.lsize, gradient, lr);
    }

    /// Apply a [`CrfGradient`] word-cost update, accumulating into `word_cost_deltas`.
    pub fn apply_word_gradient(&mut self, gradient: &CrfGradient, lr: f64) {
        apply_word_gradient_update(&mut self.word_cost_deltas, gradient, lr);
    }

    /// Return word-cost deltas rounded to i16 (word_id → delta_i16).
    ///
    /// Only non-zero deltas are returned.  The caller should apply these to the
    /// dictionary via [`Dictionary::set_word_cost_overrides`].
    pub fn word_cost_deltas_i16(&self) -> HashMap<u32, i16> {
        self.word_cost_deltas
            .iter()
            .filter_map(|(&wid, &delta)| {
                let rounded = delta.round() as i64;
                let clamped = rounded.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
                if clamped != 0 { Some((wid, clamped)) } else { None }
            })
            .collect()
    }

    /// Write word-cost deltas to a TSV file: `word_id\tdelta_i16` per line.
    ///
    /// Only non-zero deltas are written.  The file can be re-read via
    /// [`Dictionary::load_word_cost_overrides`].
    pub fn write_word_costs(&self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| Error::IoError(format!("cannot create {}: {e}", path.display())))?;
        let mut w = BufWriter::new(file);
        let deltas = self.word_cost_deltas_i16();
        // Sort by word_id for deterministic output
        let mut pairs: Vec<(u32, i16)> = deltas.into_iter().collect();
        pairs.sort_by_key(|&(wid, _)| wid);
        for (wid, delta) in pairs {
            writeln!(w, "{wid}\t{delta}")
                .map_err(|e| Error::IoError(format!("{e}")))?;
        }
        w.flush().map_err(|e| Error::IoError(format!("{e}")))?;
        Ok(())
    }

    /// Serialize to in-memory bytes (same layout as `write_binary`, for hot-reload).
    ///
    /// Format: `[lsize: u16 LE][rsize: u16 LE][costs: i16 LE × lsize × rsize]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + self.data.len() * 2);
        buf.extend_from_slice(&(self.lsize as u16).to_le_bytes());
        buf.extend_from_slice(&(self.rsize as u16).to_le_bytes());
        for &cost in &self.data {
            buf.extend_from_slice(&cost.to_le_bytes());
        }
        buf
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
    /// Minimum learning rate for linear decay. 0.0 = no decay (current behavior).
    ///
    /// When > 0.0, the effective LR at epoch `e` is:
    ///   `lr - (e / (epochs-1)) * (lr - min_learning_rate)`
    /// giving a linear schedule from `learning_rate` down to `min_learning_rate`.
    pub min_learning_rate: f64,
}

impl Default for DictTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 64,
            epochs: 10,
            l2_strength: 1e-5,
            verbose: false,
            min_learning_rate: 0.0,
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
    /// Effective learning rate used in this epoch.
    pub effective_lr: f64,
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
/// **Bug fix**: Resolves gold morpheme IDs from the dictionary before training
/// so that empirical connection counts use real left/right IDs rather than the
/// default (0, 0) placeholders that cause degenerate training signal.
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
    // Resolve gold IDs from the dictionary before training.
    // This is the fix for Bug 1: without resolving IDs, all empirical
    // connection counts collapse to (0,0) giving degenerate training signal.
    let mut resolved_corpus: Vec<GoldSegmentation> = corpus.to_vec();
    for gold in &mut resolved_corpus {
        gold.resolve_ids(dict);
    }

    let mut epoch_stats: Vec<EpochStats> = Vec::with_capacity(config.epochs);
    let mut total_sentences = 0_usize;
    let mut total_tokens = 0_usize;

    for epoch in 0..config.epochs {
        // Compute linearly-decayed effective learning rate for this epoch.
        let effective_lr = compute_effective_lr(config, epoch);

        let stats = run_epoch(matrix, &resolved_corpus, dict, config, effective_lr, epoch);
        if config.verbose {
            eprintln!(
                "epoch {}/{}: loss={:.4}  sentences={}  tokens={}  lr={:.6}",
                epoch + 1,
                config.epochs,
                stats.loss,
                stats.sentences,
                stats.tokens,
                stats.effective_lr,
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

/// Compute the effective learning rate for `epoch` using linear decay schedule.
///
/// When `config.min_learning_rate == 0.0` or `config.epochs <= 1`, returns
/// `config.learning_rate` unchanged (no decay).
pub fn compute_effective_lr(config: &DictTrainConfig, epoch: usize) -> f64 {
    if config.min_learning_rate > 0.0 && config.epochs > 1 {
        let frac = epoch as f64 / (config.epochs - 1) as f64;
        config.learning_rate - frac * (config.learning_rate - config.min_learning_rate)
    } else {
        config.learning_rate
    }
}

fn run_epoch(
    matrix: &mut TrainingMatrix,
    corpus: &[GoldSegmentation],
    dict: &Dictionary,
    config: &DictTrainConfig,
    effective_lr: f64,
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

        // Optional L2 regularization: shrink gradient toward zero.
        // Also regularizes accumulated word-cost deltas toward zero.
        if config.l2_strength > 0.0 {
            apply_l2(&mut gradient, matrix, config.l2_strength);
        }

        // Apply connection-cost updates
        matrix.apply_gradient(&gradient, effective_lr);
        // Apply word-cost updates (Bug 2 fix: these were previously dropped)
        matrix.apply_word_gradient(&gradient, effective_lr);

        total_loss += summary.loss;
        total_sentences += summary.sentence_count;
        total_tokens += summary.token_count;
    }

    EpochStats {
        epoch: epoch_idx,
        loss: total_loss,
        sentences: total_sentences,
        tokens: total_tokens,
        effective_lr,
    }
}

/// Augment gradient with L2 regularization.
///
/// For connection costs: push each updated cost toward zero.
/// For word costs: push each accumulated delta toward zero using the
/// delta as a proxy for the current cost deviation.
fn apply_l2(gradient: &mut CrfGradient, matrix: &TrainingMatrix, l2_strength: f64) {
    for (&(right_id, left_id), grad) in gradient.conn_gradients.iter_mut() {
        let cost = matrix.cost(right_id, left_id) as f64;
        *grad += l2_strength * cost;
    }
    // Regularize word cost deltas: push accumulated delta toward zero.
    // Since we can't read the actual wcost from the immutable sys_dic here,
    // we use the accumulated delta as a proxy for the current deviation.
    for (&word_id, grad) in gradient.word_gradients.iter_mut() {
        if let Some(&accum) = matrix.word_cost_deltas.get(&word_id) {
            *grad += l2_strength * accum;
        }
    }
}

// ── Evaluation helper ─────────────────────────────────────────────────────────

/// Compute precision/recall/F1 over morpheme boundary positions.
///
/// `predicted_ends` is a sorted slice of exclusive-end byte offsets
/// from the best Viterbi path (not including EOS). `gold` provides
/// the reference end positions from its morphemes.
///
/// Returns `(precision, recall, f1)` in [0.0, 1.0].
///
/// # Boundary definition
///
/// For a sentence segmented as "東京" + "は", the end positions are
/// {6, 9} (byte offsets).  A predicted boundary at offset 6 is a true positive
/// if the gold also has an end at offset 6.
pub fn boundary_f1(
    predicted_ends: &[usize],
    gold: &GoldSegmentation,
) -> (f64, f64, f64) {
    use std::collections::HashSet;

    let gold_ends: HashSet<usize> = gold
        .morphemes
        .iter()
        .scan(0_usize, |pos, m| {
            *pos += m.surface.len();
            Some(*pos)
        })
        .collect();

    let pred_ends: HashSet<usize> = predicted_ends.iter().copied().collect();

    let tp = pred_ends.intersection(&gold_ends).count() as f64;
    let prec = if pred_ends.is_empty() { 0.0 } else { tp / pred_ends.len() as f64 };
    let rec = if gold_ends.is_empty() { 0.0 } else { tp / gold_ends.len() as f64 };
    let f1 = if prec + rec == 0.0 {
        0.0
    } else {
        2.0 * prec * rec / (prec + rec)
    };
    (prec, rec, f1)
}

/// Parse a word-cost TSV file (word_id TAB delta_i16 per line) into a HashMap.
///
/// Lines beginning with '#' are treated as comments and skipped.
/// Returns an error if any line is malformed.
pub fn read_word_costs<P: AsRef<Path>>(path: P) -> Result<HashMap<u32, i16>> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::IoError(format!("cannot open {}: {e}", path.as_ref().display())))?;
    let reader = BufReader::new(file);
    let mut map = HashMap::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| Error::IoError(format!("{e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.splitn(2, '\t');
        let wid_str = parts.next().ok_or_else(|| {
            Error::IoError(format!("line {}: missing word_id", line_no + 1))
        })?;
        let delta_str = parts.next().ok_or_else(|| {
            Error::IoError(format!("line {}: missing delta", line_no + 1))
        })?;
        let word_id: u32 = wid_str.parse().map_err(|e| {
            Error::IoError(format!("line {}: invalid word_id: {e}", line_no + 1))
        })?;
        let delta: i16 = delta_str.parse().map_err(|e| {
            Error::IoError(format!("line {}: invalid delta: {e}", line_no + 1))
        })?;
        map.insert(word_id, delta);
    }
    Ok(map)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viterbi::train::{CrfGradient, GoldMorpheme, GoldSegmentation};
    use std::env::temp_dir;

    /// Build a tiny TrainingMatrix directly from data for testing.
    fn tiny_matrix(lsize: usize, rsize: usize) -> TrainingMatrix {
        let data = vec![10i16; lsize * rsize];
        TrainingMatrix { data, lsize, rsize, word_cost_deltas: HashMap::new() }
    }

    // ── Existing matrix tests ─────────────────────────────────────────────────

    #[test]
    fn test_training_matrix_cost_formula() {
        // 2×3 matrix (lsize=2, rsize=3)
        let mut m = tiny_matrix(2, 3);
        m.data[0] = 100;
        m.data[1] = 200;
        m.data[2] = 300;
        assert_eq!(m.cost(0, 0), 100);
        assert_eq!(m.cost(1, 0), 200);
        assert_eq!(m.cost(0, 1), 300);
        assert_eq!(m.cost(99, 0), i16::MAX);
    }

    #[test]
    fn test_training_matrix_apply_gradient() {
        let mut m = tiny_matrix(2, 2);
        m.data[2] = 50;
        let mut grad = CrfGradient::new();
        grad.add_conn(0, 1, 5.0);
        m.apply_gradient(&grad, 1.0);
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
        assert_eq!(bytes.len(), 4 + 2 * 3 * 2);
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 2);
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 3);
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
        assert!((c.min_learning_rate - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_epoch_stats_fields() {
        let s = EpochStats { epoch: 2, loss: 1.5, sentences: 10, tokens: 50, effective_lr: 0.01 };
        assert_eq!(s.epoch, 2);
        assert!((s.loss - 1.5).abs() < 1e-9);
        assert_eq!(s.sentences, 10);
        assert_eq!(s.tokens, 50);
    }

    // ── New tests ─────────────────────────────────────────────────────────────

    /// boundary_f1: perfect prediction → F1 = 1.0
    #[test]
    fn test_boundary_f1_perfect() {
        let gold = make_gold_seg(&["東京", "は"]);
        // "東京" = 6 bytes, "は" = 3 bytes → ends at 6, 9
        let predicted = vec![6_usize, 9_usize];
        let (prec, rec, f1) = boundary_f1(&predicted, &gold);
        assert!((prec - 1.0).abs() < 1e-9, "prec={prec}");
        assert!((rec - 1.0).abs() < 1e-9, "rec={rec}");
        assert!((f1 - 1.0).abs() < 1e-9, "f1={f1}");
    }

    /// boundary_f1: empty predicted → precision = 0, recall = 0
    #[test]
    fn test_boundary_f1_empty_predicted() {
        let gold = make_gold_seg(&["東京", "は"]);
        let predicted: Vec<usize> = vec![];
        let (prec, rec, f1) = boundary_f1(&predicted, &gold);
        assert!((prec - 0.0).abs() < 1e-9);
        assert!((rec - 0.0).abs() < 1e-9);
        assert!((f1 - 0.0).abs() < 1e-9);
    }

    /// boundary_f1: partial overlap — verify precision/recall/F1 arithmetic
    #[test]
    fn test_boundary_f1_partial() {
        // Gold: "東京"(6) + "は"(3) → ends {6, 9}
        // Predicted: correct at 6, wrong at 5 → predicted {5, 6}
        let gold = make_gold_seg(&["東京", "は"]);
        let predicted = vec![5_usize, 6_usize];
        let (prec, rec, f1) = boundary_f1(&predicted, &gold);
        // tp=1, |pred|=2, |gold|=2 → prec=0.5, rec=0.5, f1=0.5
        assert!((prec - 0.5).abs() < 1e-9, "prec={prec}");
        assert!((rec - 0.5).abs() < 1e-9, "rec={rec}");
        assert!((f1 - 0.5).abs() < 1e-9, "f1={f1}");
    }

    /// LR decay: epoch 0 → config.learning_rate, epoch max → min_learning_rate
    #[test]
    fn test_lr_decay_monotonic() {
        let config = DictTrainConfig {
            learning_rate: 0.1,
            min_learning_rate: 0.001,
            epochs: 5,
            ..Default::default()
        };
        let lr0 = compute_effective_lr(&config, 0);
        let lr4 = compute_effective_lr(&config, 4);
        let lr2 = compute_effective_lr(&config, 2);
        assert!((lr0 - 0.1).abs() < 1e-9, "epoch 0 lr should be 0.1, got {lr0}");
        assert!((lr4 - 0.001).abs() < 1e-9, "epoch 4 lr should be 0.001, got {lr4}");
        // Monotonically decreasing
        assert!(lr0 > lr2 && lr2 > lr4, "LR should be monotonically decreasing");
        // No decay when min_learning_rate == 0.0
        let config_nodecay = DictTrainConfig {
            learning_rate: 0.1,
            min_learning_rate: 0.0,
            epochs: 5,
            ..Default::default()
        };
        let lr_nd = compute_effective_lr(&config_nodecay, 3);
        assert!((lr_nd - 0.1).abs() < 1e-9);
    }

    /// apply_word_gradient_update then word_cost_deltas_i16 roundtrip
    #[test]
    fn test_word_cost_deltas_i16() {
        let mut m = tiny_matrix(2, 2);
        let mut grad = CrfGradient::new();
        grad.add_word(42, 5.0);   // delta -= lr * 5.0 = -0.1 * 5 = -0.5
        grad.add_word(99, -3.0);  // delta -= lr * -3.0 = +0.3
        m.apply_word_gradient(&grad, 0.1);

        // Deltas: 42 → -0.5, 99 → +0.3
        // Rounded to i16: 42 → 0 (rounds to 0), 99 → 0 (rounds to 0) — let's use bigger lr
        // Reset and use lr=10 to get nonzero rounded deltas
        m.word_cost_deltas.clear();
        let mut grad2 = CrfGradient::new();
        grad2.add_word(42, 5.0);   // delta = -10 * 5 = -50
        grad2.add_word(99, -3.0);  // delta = -10 * -3 = +30
        m.apply_word_gradient(&grad2, 10.0);

        let deltas_i16 = m.word_cost_deltas_i16();
        assert_eq!(deltas_i16.get(&42).copied(), Some(-50_i16));
        assert_eq!(deltas_i16.get(&99).copied(), Some(30_i16));
    }

    /// write_binary + to_bytes produce identical output
    #[test]
    fn test_to_bytes_roundtrip() {
        let mut m = tiny_matrix(2, 3);
        m.data[0] = 42;
        m.data[1] = -7;
        m.data[3] = 1000;

        let path = temp_dir().join("mecrab_test_to_bytes.bin");
        m.write_binary(&path).expect("write_binary failed");
        let file_bytes = std::fs::read(&path).expect("read failed");
        let mem_bytes = m.to_bytes();

        assert_eq!(
            file_bytes, mem_bytes,
            "to_bytes() and write_binary() must produce identical output"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// write_word_costs then re-read via read_word_costs → same map
    #[test]
    fn test_write_word_costs() {
        let mut m = tiny_matrix(2, 2);
        // Apply large gradient to get nonzero i16 deltas
        let mut grad = CrfGradient::new();
        grad.add_word(7, 2.0);    // delta = -100 * 2 = -200
        grad.add_word(13, -4.0);  // delta = -100 * -4 = +400
        m.apply_word_gradient(&grad, 100.0);

        let path = temp_dir().join("mecrab_test_word_costs.tsv");
        m.write_word_costs(&path).expect("write_word_costs failed");

        let loaded = read_word_costs(&path).expect("read_word_costs failed");
        let expected = m.word_cost_deltas_i16();

        assert_eq!(loaded, expected, "loaded map should match written map");
        let _ = std::fs::remove_file(&path);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a minimal GoldSegmentation from surface strings (no real dict lookup).
    fn make_gold_seg(surfaces: &[&str]) -> GoldSegmentation {
        let text: String = surfaces.concat();
        let morphemes: Vec<GoldMorpheme> = surfaces
            .iter()
            .map(|s| GoldMorpheme {
                surface: s.to_string(),
                left_id: 0,
                right_id: 0,
                word_id: u32::MAX,
            })
            .collect();
        GoldSegmentation { text, morphemes }
    }
}
