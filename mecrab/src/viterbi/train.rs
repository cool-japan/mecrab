//! CRF-based gradient computation for dictionary cost training.
//!
//! # Theory
//!
//! MeCrab models segmentation as a linear-chain CRF where the score of a
//! complete segmentation is the sum of connection costs between adjacent
//! morphemes plus their word costs. Training maximizes the conditional
//! likelihood P(gold | text) over a corpus.
//!
//! The gradient of the negative log-likelihood w.r.t. connection cost C(a,b) is:
//!   dL/dC(a,b) = expected_count(a,b) - empirical_count(a,b)
//!
//! where expected_count comes from forward-backward and empirical_count from
//! the gold segmentation. The cost update rule is:
//!   C(a,b) -= learning_rate * gradient
//!
//! Note: MeCab costs are integers (i16), but gradients are computed in f64
//! for numerical stability. Rounding to i16 happens at update time.

use std::collections::HashMap;

use crate::dict::Dictionary;
use crate::lattice::Lattice;
use crate::viterbi::analysis::LatticeProbTable;
use crate::viterbi::ViterbiSolver;

// ── Gold segmentation types ───────────────────────────────────────────────────

/// A single morpheme in a gold-standard segmentation.
#[derive(Debug, Clone)]
pub struct GoldMorpheme {
    /// Surface form (must match a contiguous span of the input text)
    pub surface: String,
    /// Left context ID (from dictionary entry)
    pub left_id: u16,
    /// Right context ID (from dictionary entry)
    pub right_id: u16,
    /// Word ID in the system dictionary (`u32::MAX` for unknown words)
    pub word_id: u32,
}

/// A gold-standard (human-annotated) segmentation for one sentence.
#[derive(Debug, Clone)]
pub struct GoldSegmentation {
    /// The original input text
    pub text: String,
    /// Ordered sequence of gold morphemes covering the full text
    pub morphemes: Vec<GoldMorpheme>,
}

impl GoldSegmentation {
    /// Parse from MeCab-format TSV (one token per line: "surface\tfeature\n", "EOS" ends).
    ///
    /// Feature field encodes left_id/right_id/word_cost as the first three fields when
    /// the line format is "surface\tleft_id\tright_id\tword_cost\tfeature".
    ///
    /// For plain MeCab output format (surface\tfeature), left/right ids are unknown (set to 0).
    ///
    /// Returns `None` if the input is malformed.
    pub fn from_mecab_tsv(text: &str, tsv: &str) -> Option<Self> {
        let mut morphemes = Vec::new();
        for line in tsv.lines() {
            let line = line.trim();
            if line == "EOS" || line.is_empty() {
                break;
            }
            let mut parts = line.splitn(2, '\t');
            let surface = parts.next()?.to_string();
            // We don't parse IDs from plain MeCab output; set to 0 (unknown)
            morphemes.push(GoldMorpheme {
                surface,
                left_id: 0,
                right_id: 0,
                word_id: u32::MAX, // unknown — will be resolved via dict lookup
            });
        }
        if morphemes.is_empty() {
            return None;
        }
        Some(Self {
            text: text.to_string(),
            morphemes,
        })
    }

    /// Resolve left_id, right_id, and word_id for each morpheme from the dictionary.
    ///
    /// For each morpheme whose `word_id == u32::MAX`, attempt a dictionary lookup
    /// using its surface. If exactly one match is found, fill in the IDs from the
    /// dictionary entry. Morphemes with `word_id != u32::MAX` are left unchanged.
    ///
    /// This is a best-effort resolution; morphemes not found in the dictionary retain
    /// their default IDs (0, 0, u32::MAX).
    pub fn resolve_ids(&mut self, dict: &Dictionary) {
        for m in &mut self.morphemes {
            if m.word_id != u32::MAX {
                continue;
            }
            let entries = dict.lookup(&m.surface);
            // Take the first exact-length match
            if let Some(entry) = entries.into_iter().find(|e| {
                // entry.length is in bytes; compare against surface byte length
                e.length == m.surface.len()
            }) {
                m.left_id = entry.left_id;
                m.right_id = entry.right_id;
                m.word_id = entry.word_id;
            }
        }
    }
}

// ── Gradient accumulator ──────────────────────────────────────────────────────

/// Gradient accumulator for CRF-based dictionary cost training.
///
/// Gradients are stored as f64 for numerical stability; they are rounded
/// to i16 when applied to the integer-cost dictionary.
#[derive(Debug, Default, Clone)]
pub struct CrfGradient {
    /// Connection cost gradients: (right_id, left_id) → gradient
    /// Positive gradient = increase this cost; negative = decrease.
    pub conn_gradients: HashMap<(u16, u16), f64>,
    /// Word cost gradients: word_id → gradient
    /// Positive gradient = increase this word's cost; negative = decrease.
    pub word_gradients: HashMap<u32, f64>,
    /// Number of sentences accumulated in this gradient (for averaging)
    pub sentence_count: usize,
}

impl CrfGradient {
    /// Create a new empty gradient accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a gradient contribution for a connection (right_id → left_id).
    pub fn add_conn(&mut self, right_id: u16, left_id: u16, delta: f64) {
        *self.conn_gradients.entry((right_id, left_id)).or_insert(0.0) += delta;
    }

    /// Add a gradient contribution for a word cost.
    pub fn add_word(&mut self, word_id: u32, delta: f64) {
        if word_id != u32::MAX {
            *self.word_gradients.entry(word_id).or_insert(0.0) += delta;
        }
    }

    /// Average gradients over accumulated sentences (for mini-batch SGD).
    pub fn average(&mut self) {
        if self.sentence_count <= 1 {
            return;
        }
        let n = self.sentence_count as f64;
        for v in self.conn_gradients.values_mut() {
            *v /= n;
        }
        for v in self.word_gradients.values_mut() {
            *v /= n;
        }
    }

    /// Merge another gradient into this one (for parallel accumulation).
    pub fn merge(&mut self, other: &CrfGradient) {
        for (&k, &v) in &other.conn_gradients {
            *self.conn_gradients.entry(k).or_insert(0.0) += v;
        }
        for (&k, &v) in &other.word_gradients {
            *self.word_gradients.entry(k).or_insert(0.0) += v;
        }
        self.sentence_count += other.sentence_count;
    }

    /// Total number of non-zero gradient entries across both maps.
    pub fn entry_count(&self) -> usize {
        self.conn_gradients.len() + self.word_gradients.len()
    }
}

// ── Training step summary ─────────────────────────────────────────────────────

/// Summary statistics for one CRF training step.
#[derive(Debug, Clone)]
pub struct TrainStepSummary {
    /// Negative log-likelihood loss (lower = better)
    pub loss: f64,
    /// Number of sentences in this step
    pub sentence_count: usize,
    /// Number of tokens in gold segmentations
    pub token_count: usize,
    /// Number of gradient entries (non-zero)
    pub gradient_entries: usize,
}

// ── CRF gradient computation ──────────────────────────────────────────────────

/// Temperature for Boltzmann weighting, matching the value used in `fb.rs`.
const TEMPERATURE: f64 = 500.0;

/// Compute the CRF gradient for one sentence.
///
/// Given a lattice (all possible segmentations) and a gold segmentation,
/// computes:
///   gradient = expected_count (from forward-backward) - empirical_count (from gold)
///
/// # Arguments
/// * `prob_table` — forward-backward marginals from `ViterbiSolver::forward_backward(lattice)`
/// * `lattice`    — the word lattice for the sentence
/// * `gold`       — the gold-standard segmentation
/// * `dict`       — the dictionary (for connection costs during expected count computation)
/// * `gradient`   — output: gradient is accumulated into this struct
///
/// # Returns
/// The negative log-likelihood of the gold segmentation under the current model.
#[allow(clippy::too_many_lines)]
pub fn compute_sentence_gradient(
    prob_table: &LatticeProbTable,
    lattice: &Lattice<'_>,
    gold: &GoldSegmentation,
    dict: &Dictionary,
    gradient: &mut CrfGradient,
) -> f64 {
    // ── A. Empirical counts from the gold segmentation ────────────────────────
    //
    // For each morpheme: empirical_word_count[word_id] += 1
    // For adjacent pairs (prev→curr): empirical_conn_count[(prev.right_id, curr.left_id)] += 1
    // BOS is represented by right_id = 0 (BOS node).
    let mut empirical_conn: HashMap<(u16, u16), f64> = HashMap::new();
    let mut empirical_word: HashMap<u32, f64> = HashMap::new();

    // BOS → first morpheme connection: BOS has right_id = 0
    if let Some(first) = gold.morphemes.first() {
        *empirical_conn.entry((0_u16, first.left_id)).or_insert(0.0) += 1.0;
    }

    for (idx, morph) in gold.morphemes.iter().enumerate() {
        // Word cost empirical count
        if morph.word_id != u32::MAX {
            *empirical_word.entry(morph.word_id).or_insert(0.0) += 1.0;
        }
        // Connection from prev → curr
        if idx + 1 < gold.morphemes.len() {
            let next = &gold.morphemes[idx + 1];
            *empirical_conn
                .entry((morph.right_id, next.left_id))
                .or_insert(0.0) += 1.0;
        }
    }
    // Last morpheme → EOS connection: EOS has left_id = 0
    if let Some(last) = gold.morphemes.last() {
        *empirical_conn.entry((last.right_id, 0_u16)).or_insert(0.0) += 1.0;
    }

    // ── B. Expected counts from the forward-backward marginals ────────────────
    //
    // Word expected counts: sum marginals for all nodes with the same word_id.
    // Since NodeMarginal only stores surface+feature, not word_id, we approximate
    // by grouping on surface string (exact match = same dictionary entry, same word_id).
    // For connection expected counts we compute edge marginals on-the-fly from
    // the lattice structure plus the raw alpha/beta scores, which are recomputed
    // internally via `compute_edge_expected_counts`.

    // Word marginals from prob_table: group by (start, end, surface)
    // so each distinct lattice node accumulates its own marginal.
    let mut expected_word_by_surface: HashMap<(usize, usize, String), f64> = HashMap::new();
    for pos_nodes in &prob_table.by_position {
        for nm in pos_nodes {
            // Skip BOS/EOS (start == end)
            if nm.start == nm.end {
                continue;
            }
            *expected_word_by_surface
                .entry((nm.start, nm.end, nm.surface.clone()))
                .or_insert(0.0) += nm.prob;
        }
    }

    // Expected connection counts via edge marginals computed from the lattice.
    let solver = ViterbiSolver::new(dict);
    let expected_conn = solver.compute_edge_expected_counts(lattice);

    // ── C. Accumulate gradient = empirical − expected ─────────────────────────

    // Connection gradients
    let all_conn_keys: std::collections::HashSet<(u16, u16)> = empirical_conn
        .keys()
        .chain(expected_conn.keys())
        .copied()
        .collect();
    for key in all_conn_keys {
        let emp = empirical_conn.get(&key).copied().unwrap_or(0.0);
        let exp = expected_conn.get(&key).copied().unwrap_or(0.0);
        let delta = emp - exp;
        if delta.abs() > f64::EPSILON {
            gradient.add_conn(key.0, key.1, delta);
        }
    }

    // Word cost gradients: match gold morphemes against expected_word_by_surface
    // to compute empirical − expected for each word_id.
    // We need to map surface spans back to word_ids via the lattice nodes.
    let mut expected_word_by_id: HashMap<u32, f64> = HashMap::new();
    let n = lattice.len();
    for pos in 0..n {
        let nodes = lattice.nodes_ending_at(pos);
        for node in nodes {
            if node.word_id == u32::MAX || node.start == node.end {
                continue;
            }
            let key = (node.start, node.end, node.surface.to_string());
            if let Some(&marginal) = expected_word_by_surface.get(&key) {
                *expected_word_by_id
                    .entry(node.word_id)
                    .or_insert(0.0) += marginal;
            }
        }
    }

    let all_word_ids: std::collections::HashSet<u32> = empirical_word
        .keys()
        .chain(expected_word_by_id.keys())
        .copied()
        .collect();
    for word_id in all_word_ids {
        if word_id == u32::MAX {
            continue;
        }
        let emp = empirical_word.get(&word_id).copied().unwrap_or(0.0);
        let exp = expected_word_by_id.get(&word_id).copied().unwrap_or(0.0);
        let delta = emp - exp;
        if delta.abs() > f64::EPSILON {
            gradient.add_word(word_id, delta);
        }
    }

    gradient.sentence_count += 1;

    // ── D. Negative log-likelihood of the gold segmentation ──────────────────
    //
    // NLL = -log P(gold | text)
    //     = -sum_{v in gold nodes} log_marginal(v)  (approximate)
    //
    // We look up each gold morpheme's log marginal from prob_table by matching
    // on (start_byte, end_byte, surface).  If a node is not in the lattice
    // (OOV or alignment mismatch), we use a large penalty instead.

    const MISSING_LOG_PROB: f64 = -20.0; // ≈ log(2e-9), used for lattice-absent gold nodes

    // Build a lookup: (start, end, surface) → log_prob from prob_table
    let mut log_prob_lookup: HashMap<(usize, usize, &str), f64> = HashMap::new();
    for pos_nodes in &prob_table.by_position {
        for nm in pos_nodes {
            log_prob_lookup
                .entry((nm.start, nm.end, nm.surface.as_str()))
                .or_insert(nm.log_prob);
        }
    }

    // Walk the gold path and accumulate log probs
    let mut byte_cursor = 0_usize;
    let mut nll = 0.0_f64;
    for morph in &gold.morphemes {
        let start = byte_cursor;
        let end = start + morph.surface.len();
        let surface: &str = &morph.surface;
        let log_p = log_prob_lookup
            .get(&(start, end, surface))
            .copied()
            .unwrap_or(MISSING_LOG_PROB);
        nll -= log_p;
        byte_cursor = end;
    }

    nll
}

/// Accumulate CRF gradients over a mini-batch of sentences.
///
/// For each `(lattice, gold)` pair, runs forward-backward to build `prob_table`
/// and then calls [`compute_sentence_gradient`].  Returns a merged gradient and
/// the total loss (summed NLL over all sentences in the batch).
///
/// # Arguments
/// * `batch`    — slice of (lattice, gold) pairs
/// * `dict`     — the dictionary shared by all lattices
/// * `gradient` — accumulator to which results are added (not cleared first)
///
/// # Returns
/// A [`TrainStepSummary`] containing the total NLL loss and batch statistics.
pub fn accumulate_batch_gradient(
    batch: &[(&Lattice<'_>, &GoldSegmentation)],
    dict: &Dictionary,
    gradient: &mut CrfGradient,
) -> TrainStepSummary {
    let solver = ViterbiSolver::new(dict);
    let mut total_loss = 0.0_f64;
    let mut total_tokens = 0_usize;

    for (lattice, gold) in batch {
        let prob_table = solver.forward_backward(lattice);
        let nll = compute_sentence_gradient(&prob_table, lattice, gold, dict, gradient);
        total_loss += nll;
        total_tokens += gold.morphemes.len();
    }

    TrainStepSummary {
        loss: total_loss,
        sentence_count: batch.len(),
        token_count: total_tokens,
        gradient_entries: gradient.entry_count(),
    }
}

/// Apply a gradient update to a mutable connection cost table (i16 slice, row-major).
///
/// # Arguments
/// * `matrix`        — mutable slice of connection costs, row-major by `right_id`
/// * `left_size`     — number of left context IDs (columns)
/// * `gradient`      — the gradient to apply
/// * `learning_rate` — step size (positive float, e.g. 0.01)
///
/// The update rule is: `cost[right_id][left_id] -= round(lr * grad)`.
/// Costs are clamped to i16 range after update.
pub fn apply_conn_gradient_update(
    matrix: &mut [i16],
    left_size: usize,
    gradient: &CrfGradient,
    learning_rate: f64,
) {
    for (&(right_id, left_id), &grad) in &gradient.conn_gradients {
        let idx = right_id as usize * left_size + left_id as usize;
        if idx < matrix.len() {
            let delta = (learning_rate * grad).round() as i64;
            let updated = matrix[idx] as i64 - delta;
            matrix[idx] = updated.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        }
    }
}

// ── Edge marginals (package-private helper) ───────────────────────────────────
//
// Used internally by compute_sentence_gradient; also exposed so that fb.rs can
// call it.  Implementation lives here; the fb.rs function delegates to this.

/// Compute edge marginals from alpha/beta scores obtained by running forward-backward.
///
/// This is a standalone function (not a method) so that it can be called from
/// both `train.rs` (via `ViterbiSolver::compute_edge_expected_counts`) and
/// directly in tests.
///
/// # Layout
/// For every lattice edge (u at `prev_pos`) → (v at `pos`):
///   p(u→v) = exp(alpha[prev_pos][i] + log_conn(u,v) + (-v.wcost/T) + beta[pos][j] - log_Z)
///
/// We sum these per (u.right_id, v.left_id) to get expected connection counts.
pub(super) fn edge_expected_counts_from_fb(
    lattice: &Lattice<'_>,
    alpha: &[Vec<f64>],
    beta: &[Vec<f64>],
    log_z: f64,
    dict: &Dictionary,
) -> HashMap<(u16, u16), f64> {
    let mut counts: HashMap<(u16, u16), f64> = HashMap::new();

    if log_z == f64::NEG_INFINITY {
        return counts;
    }

    let n = lattice.len();

    for (pos, beta_pos) in beta.iter().enumerate().skip(1) {
        let nodes = lattice.nodes_ending_at(pos);
        for (j, node) in nodes.iter().enumerate() {
            let beta_j = beta_pos.get(j).copied().unwrap_or(f64::NEG_INFINITY);
            if beta_j == f64::NEG_INFINITY {
                continue;
            }
            let wcost_contrib = -(node.wcost as f64) / TEMPERATURE;

            // Enumerate predecessor positions using the same logic as forward_backward
            let primary_prev_pos = if node.start == 0 {
                0_usize
            } else {
                node.start + 1
            };

            // Primary predecessor slot
            if primary_prev_pos < n {
                let prev_nodes = lattice.nodes_ending_at(primary_prev_pos);
                for (i, prev_node) in prev_nodes.iter().enumerate() {
                    let alpha_i = alpha[primary_prev_pos]
                        .get(i)
                        .copied()
                        .unwrap_or(f64::NEG_INFINITY);
                    if alpha_i == f64::NEG_INFINITY {
                        continue;
                    }
                    let conn =
                        dict.connection_cost(prev_node.right_id, node.left_id) as f64;
                    let arc_contrib = -conn / TEMPERATURE;
                    let log_edge_prob = alpha_i + arc_contrib + wcost_contrib + beta_j - log_z;
                    let edge_prob = log_edge_prob.exp().clamp(0.0, 1.0);
                    *counts
                        .entry((prev_node.right_id, node.left_id))
                        .or_insert(0.0) += edge_prob;
                }
            }

            // Span case: earlier positions whose nodes end at node.start
            let span_limit = primary_prev_pos.min(n);
            for (check_pos, alpha_check) in alpha.iter().enumerate().take(span_limit).skip(1) {
                let prev_nodes = lattice.nodes_ending_at(check_pos);
                for (i, prev_node) in prev_nodes.iter().enumerate() {
                    if prev_node.end != node.start {
                        continue;
                    }
                    let alpha_i = alpha_check.get(i).copied().unwrap_or(f64::NEG_INFINITY);
                    if alpha_i == f64::NEG_INFINITY {
                        continue;
                    }
                    let conn =
                        dict.connection_cost(prev_node.right_id, node.left_id) as f64;
                    let arc_contrib = -conn / TEMPERATURE;
                    let log_edge_prob = alpha_i + arc_contrib + wcost_contrib + beta_j - log_z;
                    let edge_prob = log_edge_prob.exp().clamp(0.0, 1.0);
                    *counts
                        .entry((prev_node.right_id, node.left_id))
                        .or_insert(0.0) += edge_prob;
                }
            }
        }
    }

    counts
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gold_segmentation_from_mecab_tsv() {
        let text = "東京は";
        let tsv = "東京\t名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ\nは\t助詞,係助詞,*,*,*,*,は,ハ,ワ\nEOS\n";
        let gold = GoldSegmentation::from_mecab_tsv(text, tsv).unwrap();
        assert_eq!(gold.morphemes.len(), 2);
        assert_eq!(gold.morphemes[0].surface, "東京");
        assert_eq!(gold.morphemes[1].surface, "は");
    }

    #[test]
    fn test_crf_gradient_new() {
        let mut grad = CrfGradient::new();
        grad.add_conn(1, 2, 0.5);
        grad.add_conn(1, 2, 0.3);
        assert!((grad.conn_gradients[&(1, 2)] - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_crf_gradient_merge() {
        let mut g1 = CrfGradient::new();
        let mut g2 = CrfGradient::new();
        g1.add_conn(1, 2, 1.0);
        g2.add_conn(1, 2, 2.0);
        g1.merge(&g2);
        assert!((g1.conn_gradients[&(1, 2)] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_gold_morpheme_default() {
        let m = GoldMorpheme {
            surface: "テスト".to_string(),
            left_id: 1,
            right_id: 2,
            word_id: 42,
        };
        assert_eq!(m.surface, "テスト");
        assert_eq!(m.word_id, 42);
    }

    #[test]
    fn test_train_step_summary_fields() {
        let s = TrainStepSummary {
            loss: 1.23,
            sentence_count: 10,
            token_count: 50,
            gradient_entries: 200,
        };
        assert_eq!(s.sentence_count, 10);
    }

    #[test]
    fn test_crf_gradient_add_word_skips_unknown() {
        let mut grad = CrfGradient::new();
        grad.add_word(u32::MAX, 5.0); // should be ignored
        grad.add_word(42, 3.0);
        assert!(!grad.word_gradients.contains_key(&u32::MAX));
        assert!((grad.word_gradients[&42] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_crf_gradient_average() {
        let mut grad = CrfGradient::new();
        grad.sentence_count = 2;
        grad.add_conn(0, 1, 4.0);
        grad.add_word(7, 6.0);
        grad.average();
        assert!((grad.conn_gradients[&(0, 1)] - 2.0).abs() < 1e-9);
        assert!((grad.word_gradients[&7] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_crf_gradient_entry_count() {
        let mut grad = CrfGradient::new();
        grad.add_conn(1, 2, 0.5);
        grad.add_word(10, 1.0);
        assert_eq!(grad.entry_count(), 2);
    }

    #[test]
    fn test_gold_segmentation_empty_tsv() {
        let result = GoldSegmentation::from_mecab_tsv("text", "EOS\n");
        assert!(result.is_none());
    }

    #[test]
    fn test_gold_segmentation_eos_stops_parsing() {
        let tsv = "東京\t名詞\nEOS\nは\t助詞\n";
        let gold = GoldSegmentation::from_mecab_tsv("東京は", tsv).unwrap();
        // Only "東京" should be parsed; "は" comes after EOS
        assert_eq!(gold.morphemes.len(), 1);
        assert_eq!(gold.morphemes[0].surface, "東京");
    }

    #[test]
    fn test_apply_conn_gradient_update_clamps_to_i16() {
        // Create a tiny 2x2 matrix (left_size=2)
        let mut matrix = vec![0i16, 0i16, 0i16, 0i16];
        let mut grad = CrfGradient::new();
        // Gradient of 100000 with lr=1.0 should clamp to i16::MAX
        grad.add_conn(0, 0, 100_000.0);
        apply_conn_gradient_update(&mut matrix, 2, &grad, 1.0);
        assert_eq!(matrix[0], i16::MIN);
    }

    #[test]
    fn test_apply_conn_gradient_update_decreases_cost() {
        // right_id=0, left_id=1: idx = 0*2 + 1 = 1
        let mut matrix = vec![0i16, 100i16, 0i16, 0i16];
        let mut grad = CrfGradient::new();
        // empirical > expected: gradient positive → decrease cost
        grad.add_conn(0, 1, 10.0);
        apply_conn_gradient_update(&mut matrix, 2, &grad, 0.5);
        // delta = round(0.5 * 10) = 5; 100 - 5 = 95
        assert_eq!(matrix[1], 95i16);
    }
}
