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
use crate::viterbi::ViterbiSolver;
use crate::viterbi::analysis::LatticeProbTable;
use crate::viterbi::fb::{DictCostSource, EdgeCostSource};

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
        *self
            .conn_gradients
            .entry((right_id, left_id))
            .or_insert(0.0) += delta;
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

// ── Optimizer ─────────────────────────────────────────────────────────────────

/// Adaptive gradient-descent optimizer for dictionary-cost training.
///
/// Each variant transforms a raw gradient into a per-parameter *step* — the
/// f64 amount **subtracted** from the parameter (matching the historical
/// `cost -= lr * grad` sign convention).  [`Optimizer::Sgd`] is the default and
/// yields `step = lr * grad`, byte-for-byte identical to the original
/// plain-SGD update path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Optimizer {
    /// Plain stochastic gradient descent: `step = lr · grad`.
    #[default]
    Sgd,
    /// AdaGrad: per-parameter learning-rate annealing via the running sum of
    /// squared gradients — `G += grad²; step = lr · grad / (√G + ε)`.
    AdaGrad,
    /// RMSProp: exponentially-decayed average of squared gradients —
    /// `G = decay·G + (1-decay)·grad²; step = lr · grad / (√G + ε)`.
    RmsProp,
    /// Adam: bias-corrected first and second moment estimates —
    /// `m = β₁·m + (1-β₁)·grad; v = β₂·v + (1-β₂)·grad²;`
    /// `step = lr · m̂ / (√v̂ + ε)` with `m̂ = m/(1-β₁ᵗ)`, `v̂ = v/(1-β₂ᵗ)`.
    Adam,
    /// Batch L-BFGS (or OWL-QN when an `L1` strength is set). Unlike the others,
    /// this is **not** a per-parameter step rule applied in the mini-batch loop —
    /// it drives a full-batch second-order fit via
    /// [`train_lbfgs`](crate::viterbi::train_lbfgs), so `train_dict` dispatches to
    /// a separate code path and the per-step machinery below is never invoked.
    Lbfgs,
}

/// Hyperparameters governing an [`OptimizerState`] step.
///
/// Bundled so the per-parameter step math stays decoupled from the full
/// [`DictTrainConfig`](crate::viterbi::train_loop::DictTrainConfig); the latter
/// produces one of these via `DictTrainConfig::optimizer_config`.
#[derive(Debug, Clone, Copy)]
pub struct OptimizerConfig {
    /// Which optimizer to apply.
    pub optimizer: Optimizer,
    /// Numerical-stability constant added to the denominator. Default `1e-8`.
    pub epsilon: f64,
    /// RMSProp squared-gradient decay rate. Default `0.9`.
    pub rmsprop_decay: f64,
    /// Adam first-moment decay (β₁). Default `0.9`.
    pub adam_beta1: f64,
    /// Adam second-moment decay (β₂). Default `0.999`.
    pub adam_beta2: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer: Optimizer::Sgd,
            epsilon: 1e-8,
            rmsprop_decay: 0.9,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
        }
    }
}

/// Persistent adaptive-optimizer state, keyed per parameter.
///
/// One instance is created at the start of a
/// [`train_dict`](crate::viterbi::train_loop::train_dict) run and threaded
/// through every batch of every epoch so that AdaGrad accumulation, RMSProp
/// decay, and Adam moment/timestep state persist across the *entire* run.
///
/// Connection-cost parameters are keyed by `(right_id, left_id)`; word-cost
/// parameters are keyed by `word_id`.  For [`Optimizer::Sgd`] the maps stay
/// empty and the step reduces to `lr · grad`.
#[derive(Debug, Clone, Default)]
pub struct OptimizerState {
    /// Optimizer this state was constructed for (authoritative for branching).
    optimizer: Optimizer,
    /// Second-moment accumulator per connection key:
    /// AdaGrad `G`, RMSProp decayed average, or Adam `v`.
    conn_v: HashMap<(u16, u16), f64>,
    /// Second-moment accumulator per word key.
    word_v: HashMap<u32, f64>,
    /// Adam first moment `m` per connection key (unused by other optimizers).
    conn_m: HashMap<(u16, u16), f64>,
    /// Adam first moment `m` per word key.
    word_m: HashMap<u32, f64>,
    /// Global Adam timestep, incremented once per parameter-update step and
    /// monotonically increasing across all batches and epochs.
    t: u64,
}

impl OptimizerState {
    /// Create empty state for `optimizer`.
    pub fn new(optimizer: Optimizer) -> Self {
        Self {
            optimizer,
            conn_v: HashMap::new(),
            word_v: HashMap::new(),
            conn_m: HashMap::new(),
            word_m: HashMap::new(),
            t: 0,
        }
    }

    /// The optimizer this state drives.
    pub fn optimizer(&self) -> Optimizer {
        self.optimizer
    }

    /// Current global Adam timestep (`0` for non-Adam optimizers).
    pub fn timestep(&self) -> u64 {
        self.t
    }

    /// Effective step for the connection parameter `(right_id, left_id)`.
    ///
    /// Returns the f64 amount to **subtract** from the cost (same sign
    /// convention as [`apply_conn_gradient_update`]).  Mutates the per-parameter
    /// adaptive state in place.
    pub fn conn_step(
        &mut self,
        key: (u16, u16),
        grad: f64,
        lr: f64,
        config: &OptimizerConfig,
    ) -> f64 {
        adaptive_step(
            self.optimizer,
            &mut self.conn_v,
            &mut self.conn_m,
            &mut self.t,
            key,
            grad,
            lr,
            config,
        )
    }

    /// Effective step for the word parameter `word_id`.
    ///
    /// Returns the f64 amount to **subtract** from the word-cost delta.
    pub fn word_step(&mut self, key: u32, grad: f64, lr: f64, config: &OptimizerConfig) -> f64 {
        adaptive_step(
            self.optimizer,
            &mut self.word_v,
            &mut self.word_m,
            &mut self.t,
            key,
            grad,
            lr,
            config,
        )
    }
}

/// Core per-parameter optimizer math shared by connection and word updates.
///
/// `v_map` holds the second-moment accumulator (AdaGrad `G`, RMSProp decayed
/// average, or Adam `v`) and `m_map` the Adam first moment; `t` is the shared
/// global Adam timestep.  Returns the f64 step to subtract from the parameter.
#[allow(clippy::too_many_arguments)]
fn adaptive_step<K>(
    optimizer: Optimizer,
    v_map: &mut HashMap<K, f64>,
    m_map: &mut HashMap<K, f64>,
    t: &mut u64,
    key: K,
    grad: f64,
    lr: f64,
    config: &OptimizerConfig,
) -> f64
where
    K: std::hash::Hash + Eq + Copy,
{
    match optimizer {
        // L-BFGS uses its own batch driver, never this per-step path; treat as
        // plain SGD if ever reached so the match stays total.
        Optimizer::Sgd | Optimizer::Lbfgs => lr * grad,
        Optimizer::AdaGrad => {
            let g = v_map.entry(key).or_insert(0.0);
            *g += grad * grad;
            lr * grad / (g.sqrt() + config.epsilon)
        }
        Optimizer::RmsProp => {
            let decay = config.rmsprop_decay;
            let g = v_map.entry(key).or_insert(0.0);
            *g = decay * *g + (1.0 - decay) * grad * grad;
            lr * grad / (g.sqrt() + config.epsilon)
        }
        Optimizer::Adam => {
            *t += 1;
            let b1 = config.adam_beta1;
            let b2 = config.adam_beta2;
            let m_new = {
                let m = m_map.entry(key).or_insert(0.0);
                *m = b1 * *m + (1.0 - b1) * grad;
                *m
            };
            let v_new = {
                let v = v_map.entry(key).or_insert(0.0);
                *v = b2 * *v + (1.0 - b2) * grad * grad;
                *v
            };
            let t_f = *t as f64;
            let m_hat = m_new / (1.0 - b1.powf(t_f));
            let v_hat = v_new / (1.0 - b2.powf(t_f));
            lr * m_hat / (v_hat.sqrt() + config.epsilon)
        }
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

/// Compute the CRF gradient for one sentence against the immutable dictionary.
///
/// Thin wrapper over `compute_sentence_gradient_with` using `DictCostSource`
/// — byte-identical to the historical behaviour.  `prob_table` must have been
/// produced by `ViterbiSolver::forward_backward(lattice)` (i.e. with the same
/// dictionary costs).
pub fn compute_sentence_gradient(
    prob_table: &LatticeProbTable,
    lattice: &Lattice<'_>,
    gold: &GoldSegmentation,
    dict: &Dictionary,
    gradient: &mut CrfGradient,
) -> f64 {
    compute_sentence_gradient_with(
        prob_table,
        lattice,
        gold,
        dict,
        &DictCostSource { dict },
        gradient,
    )
}

/// Compute the CRF gradient for one sentence using an explicit [`EdgeCostSource`].
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
pub(crate) fn compute_sentence_gradient_with<S: EdgeCostSource>(
    prob_table: &LatticeProbTable,
    lattice: &Lattice<'_>,
    gold: &GoldSegmentation,
    dict: &Dictionary,
    costs: &S,
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

    // Expected connection counts via edge marginals computed from the lattice
    // under the supplied cost source (the evolving model during iterative training).
    let solver = ViterbiSolver::new(dict);
    let expected_conn = solver.compute_edge_expected_counts_with(lattice, costs);

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
                *expected_word_by_id.entry(node.word_id).or_insert(0.0) += marginal;
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

/// Accumulate CRF gradients over a mini-batch against the immutable dictionary.
///
/// Thin wrapper over `accumulate_batch_gradient_with` using `DictCostSource`
/// — byte-identical to the historical behaviour.
pub fn accumulate_batch_gradient(
    batch: &[(&Lattice<'_>, &GoldSegmentation)],
    dict: &Dictionary,
    gradient: &mut CrfGradient,
) -> TrainStepSummary {
    accumulate_batch_gradient_with(batch, dict, &DictCostSource { dict }, gradient)
}

/// Accumulate CRF gradients over a mini-batch of sentences using an explicit
/// [`EdgeCostSource`].
///
/// For each `(lattice, gold)` pair, runs forward-backward (under `costs`) to
/// build `prob_table` and then calls [`compute_sentence_gradient_with`].
/// Returns a merged gradient and the total loss (summed NLL over all sentences).
///
/// # Arguments
/// * `batch`    — slice of (lattice, gold) pairs
/// * `dict`     — the dictionary shared by all lattices (lattice structure)
/// * `costs`    — cost source for connection/word costs (dictionary or evolving model)
/// * `gradient` — accumulator to which results are added (not cleared first)
///
/// # Returns
/// A [`TrainStepSummary`] containing the total NLL loss and batch statistics.
pub(crate) fn accumulate_batch_gradient_with<S: EdgeCostSource>(
    batch: &[(&Lattice<'_>, &GoldSegmentation)],
    dict: &Dictionary,
    costs: &S,
    gradient: &mut CrfGradient,
) -> TrainStepSummary {
    let solver = ViterbiSolver::new(dict);
    let mut total_loss = 0.0_f64;
    let mut total_tokens = 0_usize;

    for (lattice, gold) in batch {
        let prob_table = solver.forward_backward_with(lattice, costs);
        let nll = compute_sentence_gradient_with(&prob_table, lattice, gold, dict, costs, gradient);
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
        let idx = right_id as usize + left_size * left_id as usize;
        if idx < matrix.len() {
            let delta = (learning_rate * grad).round() as i64;
            let updated = matrix[idx] as i64 - delta;
            matrix[idx] = updated.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        }
    }
}

/// Apply word-cost gradient updates, accumulating into a delta map.
///
/// `deltas[word_id]` tracks the total f64 adjustment to be applied to that
/// word's wcost (negative = decrease cost = favor this word).
/// The update rule: `delta[word_id] -= learning_rate * grad`.
///
/// Note: the delta is in f64 space and is later rounded to i16 when persisted.
pub fn apply_word_gradient_update<S: std::hash::BuildHasher>(
    deltas: &mut HashMap<u32, f64, S>,
    gradient: &CrfGradient,
    learning_rate: f64,
) {
    for (&word_id, &grad) in &gradient.word_gradients {
        *deltas.entry(word_id).or_insert(0.0) -= learning_rate * grad;
    }
}

/// Apply a connection-cost update using an adaptive [`OptimizerState`].
///
/// Identical to [`apply_conn_gradient_update`] when `config.optimizer` is
/// [`Optimizer::Sgd`]; otherwise the per-parameter step is adapted via `state`,
/// whose accumulators persist across calls (and therefore across batches and
/// epochs).  The update rule remains `cost -= round(step)` with i16 clamping.
pub fn apply_conn_gradient_update_opt(
    matrix: &mut [i16],
    left_size: usize,
    gradient: &CrfGradient,
    learning_rate: f64,
    state: &mut OptimizerState,
    config: &OptimizerConfig,
) {
    for (&(right_id, left_id), &grad) in &gradient.conn_gradients {
        let idx = right_id as usize + left_size * left_id as usize;
        if idx < matrix.len() {
            let step = state.conn_step((right_id, left_id), grad, learning_rate, config);
            let delta = step.round() as i64;
            let updated = matrix[idx] as i64 - delta;
            matrix[idx] = updated.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        }
    }
}

/// Apply word-cost gradient updates using an adaptive [`OptimizerState`].
///
/// Identical to [`apply_word_gradient_update`] when `config.optimizer` is
/// [`Optimizer::Sgd`]; otherwise the per-parameter step is adapted via `state`.
/// The update rule remains `delta[word_id] -= step`.
pub fn apply_word_gradient_update_opt<S: std::hash::BuildHasher>(
    deltas: &mut HashMap<u32, f64, S>,
    gradient: &CrfGradient,
    learning_rate: f64,
    state: &mut OptimizerState,
    config: &OptimizerConfig,
) {
    for (&word_id, &grad) in &gradient.word_gradients {
        let step = state.word_step(word_id, grad, learning_rate, config);
        *deltas.entry(word_id).or_insert(0.0) -= step;
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
pub(super) fn edge_expected_counts_from_fb<S: EdgeCostSource>(
    lattice: &Lattice<'_>,
    alpha: &[Vec<f64>],
    beta: &[Vec<f64>],
    log_z: f64,
    costs: &S,
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
            let wcost_contrib = -costs.word_cost(node) / TEMPERATURE;

            // Enumerate predecessor positions using the same logic as forward_backward
            let primary_prev_pos = if node.start == 0 {
                0_usize
            } else {
                node.start + 1
            };

            // Predecessors of `node` all live in a single bucket
            // (`primary_prev_pos`) by the lattice invariant — no span scan needed.
            if primary_prev_pos < n {
                let prev_nodes = lattice.nodes_ending_at(primary_prev_pos);
                for (i, prev_node) in prev_nodes.iter().enumerate() {
                    // Same-position arc (EOS only): valid predecessors are
                    // strictly earlier in the bucket — this also excludes the
                    // spurious EOS→EOS self-edge.
                    if primary_prev_pos == pos && i >= j {
                        break;
                    }
                    let alpha_i = alpha[primary_prev_pos]
                        .get(i)
                        .copied()
                        .unwrap_or(f64::NEG_INFINITY);
                    if alpha_i == f64::NEG_INFINITY {
                        continue;
                    }
                    let conn = costs.connection_cost(prev_node.right_id, node.left_id);
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
        // right_id=0, left_id=1: idx = 0 + 2*1 = 2
        let mut matrix = vec![0i16, 0i16, 100i16, 0i16];
        let mut grad = CrfGradient::new();
        // empirical > expected: gradient positive → decrease cost
        grad.add_conn(0, 1, 10.0);
        apply_conn_gradient_update(&mut matrix, 2, &grad, 0.5);
        // delta = round(0.5 * 10) = 5; 100 - 5 = 95
        assert_eq!(matrix[2], 95i16);
    }

    // ── Optimizer tests ───────────────────────────────────────────────────────

    #[test]
    fn test_optimizer_default_is_sgd() {
        assert_eq!(Optimizer::default(), Optimizer::Sgd);
        let cfg = OptimizerConfig::default();
        assert_eq!(cfg.optimizer, Optimizer::Sgd);
        assert!((cfg.epsilon - 1e-8).abs() < 1e-20);
        assert!((cfg.rmsprop_decay - 0.9).abs() < 1e-12);
        assert!((cfg.adam_beta1 - 0.9).abs() < 1e-12);
        assert!((cfg.adam_beta2 - 0.999).abs() < 1e-12);
    }

    /// SGD via `OptimizerState` must produce exactly `lr * grad` and never
    /// advance the Adam timestep.
    #[test]
    fn test_optimizer_state_sgd_step_identity() {
        let cfg = OptimizerConfig::default(); // Sgd
        let mut state = OptimizerState::new(Optimizer::Sgd);
        let lr = 0.05_f64;
        let grad = 7.5_f64;
        let step = state.conn_step((3, 4), grad, lr, &cfg);
        assert!(
            (step - lr * grad).abs() < 1e-15,
            "SGD conn step must equal lr*grad, got {step}"
        );
        let wstep = state.word_step(42, grad, lr, &cfg);
        assert!(
            (wstep - lr * grad).abs() < 1e-15,
            "SGD word step must equal lr*grad, got {wstep}"
        );
        assert_eq!(
            state.timestep(),
            0,
            "SGD must not advance the Adam timestep"
        );
    }

    /// AdaGrad: first step ≈ `lr*grad/(|grad|+ε)`; a second update with the same
    /// gradient yields a strictly smaller step (per-parameter LR decay).
    #[test]
    fn test_optimizer_state_adagrad_per_param_decay() {
        let cfg = OptimizerConfig {
            optimizer: Optimizer::AdaGrad,
            ..OptimizerConfig::default()
        };
        let mut state = OptimizerState::new(Optimizer::AdaGrad);
        let lr = 1.0_f64;
        let grad = 20.0_f64;
        let step1 = state.conn_step((0, 0), grad, lr, &cfg);
        let step2 = state.conn_step((0, 0), grad, lr, &cfg);
        // G = grad² after the first update → √G = |grad|.
        let expected1 = lr * grad / (grad.abs() + cfg.epsilon);
        assert!(
            (step1 - expected1).abs() < 1e-9,
            "step1={step1} expected≈{expected1}"
        );
        assert!(step1.is_finite() && step2.is_finite());
        assert!(
            step2 < step1,
            "second AdaGrad step must be smaller: step1={step1} step2={step2}"
        );
    }

    /// Adam: first step is finite, non-zero, and ≈ `lr` for a unit gradient; the
    /// global timestep increments once per parameter update, monotonically.
    #[test]
    fn test_optimizer_state_adam_first_step_and_timestep() {
        let cfg = OptimizerConfig {
            optimizer: Optimizer::Adam,
            ..OptimizerConfig::default()
        };
        let mut state = OptimizerState::new(Optimizer::Adam);
        let lr = 0.01_f64;
        let step = state.conn_step((1, 1), 1.0, lr, &cfg);
        assert!(step.is_finite(), "Adam step must be finite, got {step}");
        assert!(step.abs() > 0.0, "Adam step must be non-zero");
        assert!(
            (step - lr).abs() < 1e-6,
            "bias-corrected first Adam step ≈ lr for unit grad, got {step}"
        );
        assert_eq!(state.timestep(), 1, "timestep must advance to 1");
        let _ = state.conn_step((2, 2), 1.0, lr, &cfg);
        assert_eq!(state.timestep(), 2, "timestep must be monotonic");
        let _ = state.word_step(7, -1.0, lr, &cfg);
        assert_eq!(
            state.timestep(),
            3,
            "word updates share the global Adam timestep"
        );
    }

    /// The SGD optimizer path must be byte-identical to the plain update.
    #[test]
    fn test_apply_conn_gradient_update_opt_sgd_matches_plain() {
        let mut m_plain = vec![0i16, 0, 100, 0];
        let mut m_opt = m_plain.clone();
        let mut grad = CrfGradient::new();
        grad.add_conn(0, 1, 10.0); // idx = 0 + 2*1 = 2
        apply_conn_gradient_update(&mut m_plain, 2, &grad, 0.5);
        let cfg = OptimizerConfig::default(); // Sgd
        let mut state = OptimizerState::new(Optimizer::Sgd);
        apply_conn_gradient_update_opt(&mut m_opt, 2, &grad, 0.5, &mut state, &cfg);
        assert_eq!(m_plain, m_opt, "SGD optimizer path must match plain update");
    }

    /// AdaGrad connection update differs from the SGD result and stays finite —
    /// the optimizer-path analogue of "the trained matrix differs from SGD".
    #[test]
    fn test_apply_conn_gradient_update_opt_adagrad_differs_from_sgd() {
        let mut m_sgd = vec![0i16; 4];
        let mut m_ada = vec![0i16; 4];
        let mut grad = CrfGradient::new();
        grad.add_conn(0, 0, 20.0); // idx 0
        let lr = 1.0;

        let sgd_cfg = OptimizerConfig::default();
        let mut sgd_state = OptimizerState::new(Optimizer::Sgd);
        apply_conn_gradient_update_opt(&mut m_sgd, 2, &grad, lr, &mut sgd_state, &sgd_cfg);

        let ada_cfg = OptimizerConfig {
            optimizer: Optimizer::AdaGrad,
            ..OptimizerConfig::default()
        };
        let mut ada_state = OptimizerState::new(Optimizer::AdaGrad);
        apply_conn_gradient_update_opt(&mut m_ada, 2, &grad, lr, &mut ada_state, &ada_cfg);

        // SGD: 0 - round(1.0*20) = -20. AdaGrad: 0 - round(~1.0) = -1.
        assert_eq!(m_sgd[0], -20, "SGD subtracts the full gradient");
        assert_eq!(m_ada[0], -1, "AdaGrad subtracts a normalised unit step");
        assert_ne!(m_sgd, m_ada, "AdaGrad update must differ from SGD update");
    }

    /// AdaGrad word updates accumulate across calls (state persists) and the
    /// second per-parameter step shrinks.
    #[test]
    fn test_apply_word_gradient_update_opt_adagrad_accumulates() {
        let mut deltas: HashMap<u32, f64> = HashMap::new();
        let mut grad = CrfGradient::new();
        grad.add_word(5, 4.0);
        let cfg = OptimizerConfig {
            optimizer: Optimizer::AdaGrad,
            ..OptimizerConfig::default()
        };
        let mut state = OptimizerState::new(Optimizer::AdaGrad);

        apply_word_gradient_update_opt(&mut deltas, &grad, 1.0, &mut state, &cfg);
        let d1 = deltas[&5];
        apply_word_gradient_update_opt(&mut deltas, &grad, 1.0, &mut state, &cfg);
        let d2 = deltas[&5];

        let dec1 = -d1; // first decrement magnitude (grad>0 → step>0 → delta down)
        let dec2 = -(d2 - d1); // second decrement magnitude
        assert!(d1.is_finite() && d2.is_finite());
        assert!(dec1 > 0.0 && dec2 > 0.0, "dec1={dec1} dec2={dec2}");
        assert!(
            dec2 < dec1,
            "AdaGrad second word step must shrink: dec1={dec1} dec2={dec2}"
        );
    }
}
