//! Batch L-BFGS / OWL-QN driver for CRF dictionary-cost training.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Unlike the per-batch step optimizers in [`super::train_loop`], this fits the
//! connection + word costs as a single smooth optimization problem:
//!
//! ```text
//!   minimize  f(x) = Σ_sentences NLL(x) + ½·l2·‖x‖²            ( + l1·‖x‖₁ )
//! ```
//!
//! where `x` is the vector of **deltas** added to the dictionary's connection
//! and word costs (so `x = 0` is the untrained dictionary), and
//! `NLL = cost_gold(x)/T + log Z(x)` is the true conditional negative
//! log-likelihood of the gold path (consistent with the analytic CRF gradient
//! `(expected − empirical)/T`). The smooth objective requires **continuous**
//! costs, which is why [`EdgeCostSource`] returns `f64`. The actual minimization
//! is delegated to the generic, independently-validated [`super::lbfgs`].

use std::collections::HashMap;

use crate::dict::Dictionary;
use crate::lattice::{Lattice, LatticeNode};
use crate::viterbi::ViterbiSolver;
use crate::viterbi::fb::EdgeCostSource;
use crate::viterbi::lbfgs::{self, LbfgsConfig};
use crate::viterbi::train::{
    CrfGradient, GoldSegmentation, accumulate_batch_gradient, compute_sentence_gradient_with,
};
use crate::viterbi::train_loop::{DictTrainConfig, DictTrainSummary, EpochStats, TrainingMatrix};

/// Boltzmann temperature — must match the value in `fb.rs`/`train.rs` so the
/// objective value and its analytic gradient stay mutually consistent.
const TEMPERATURE: f64 = 500.0;

/// Continuous cost source for L-BFGS: dictionary base cost plus an `f64` delta
/// read from the optimizer's parameter vector `x` via the active-set index.
struct LbfgsCostSource<'a> {
    dict: &'a Dictionary,
    conn_index: &'a HashMap<(u16, u16), usize>,
    word_index: &'a HashMap<u32, usize>,
    x: &'a [f64],
}

impl EdgeCostSource for LbfgsCostSource<'_> {
    #[inline]
    fn connection_cost(&self, right_id: u16, left_id: u16) -> f64 {
        let base = f64::from(self.dict.connection_cost(right_id, left_id));
        base + self
            .conn_index
            .get(&(right_id, left_id))
            .map_or(0.0, |&i| self.x[i])
    }

    #[inline]
    fn word_cost(&self, node: &LatticeNode<'_>) -> f64 {
        let base = f64::from(node.wcost);
        base + self
            .word_index
            .get(&node.word_id)
            .map_or(0.0, |&i| self.x[i])
    }
}

/// Cost of the gold path under the current parameters: Σ connection costs along
/// the gold sequence (including BOS→first and last→EOS, both context id `0`) plus
/// Σ word costs of resolved morphemes. Terms that do not depend on `x` (e.g. OOV
/// morphemes with no trainable word cost) are omitted — they are constant and
/// cancel in the line-search comparison.
fn gold_path_cost(
    gold: &GoldSegmentation,
    word_base: &HashMap<u32, f64>,
    word_index: &HashMap<u32, usize>,
    x: &[f64],
    costs: &LbfgsCostSource<'_>,
) -> f64 {
    let ms = &gold.morphemes;
    let mut total = 0.0;
    if let Some(first) = ms.first() {
        total += costs.connection_cost(0, first.left_id);
    }
    for pair in ms.windows(2) {
        total += costs.connection_cost(pair[0].right_id, pair[1].left_id);
    }
    if let Some(last) = ms.last() {
        total += costs.connection_cost(last.right_id, 0);
    }
    for m in ms {
        if m.word_id != u32::MAX {
            if let Some(&base) = word_base.get(&m.word_id) {
                let delta = word_index.get(&m.word_id).map_or(0.0, |&i| x[i]);
                total += base + delta;
            }
        }
    }
    total
}

/// Run batch L-BFGS (or OWL-QN when `config.l1_strength > 0`) to fit the
/// dictionary connection + word costs over `corpus` (whose gold IDs must already
/// be resolved). Updates `matrix` in place and returns a one-entry summary
/// holding the final objective value and the iteration count.
#[allow(clippy::too_many_lines)]
pub(crate) fn train_dict_lbfgs(
    matrix: &mut TrainingMatrix,
    corpus: &[GoldSegmentation],
    dict: &Dictionary,
    config: &DictTrainConfig,
) -> DictTrainSummary {
    // ── Build lattices once (immutable structure reused by every evaluation) ──
    let lattices: Vec<(Lattice<'_>, &GoldSegmentation)> = corpus
        .iter()
        .filter_map(|gold| Lattice::build(&gold.text, dict).ok().map(|l| (l, gold)))
        .collect();
    let refs: Vec<(&Lattice<'_>, &GoldSegmentation)> =
        lattices.iter().map(|(l, g)| (l, *g)).collect();
    let total_tokens: usize = corpus.iter().map(|g| g.morphemes.len()).sum();

    // ── Discover the active parameter set from a probe gradient at x = 0 ──────
    let mut probe = CrfGradient::new();
    accumulate_batch_gradient(&refs, dict, &mut probe);
    let mut conn_keys: Vec<(u16, u16)> = probe.conn_gradients.keys().copied().collect();
    conn_keys.sort_unstable();
    let mut word_keys: Vec<u32> = probe.word_gradients.keys().copied().collect();
    word_keys.sort_unstable();
    let n_conn = conn_keys.len();
    let n_params = n_conn + word_keys.len();

    if n_params == 0 || refs.is_empty() {
        return DictTrainSummary {
            total_epochs: 0,
            total_sentences: refs.len(),
            total_tokens,
            epoch_stats: Vec::new(),
        };
    }

    let conn_index: HashMap<(u16, u16), usize> =
        conn_keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let word_index: HashMap<u32, usize> = word_keys
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, n_conn + i))
        .collect();

    // Base (dictionary) word cost per active word_id, read off the lattice nodes.
    let mut word_base: HashMap<u32, f64> = HashMap::new();
    for &(lattice, _) in &refs {
        for pos in 0..lattice.len() {
            for node in lattice.nodes_ending_at(pos) {
                if word_index.contains_key(&node.word_id) {
                    word_base
                        .entry(node.word_id)
                        .or_insert_with(|| f64::from(node.wcost));
                }
            }
        }
    }

    let l2 = config.l2_strength;
    let solver = ViterbiSolver::new(dict);

    // ── Objective f(x) = Σ NLL(x) + ½·l2·‖x‖²,  gradient g = ∇f ───────────────
    let eval = |x: &[f64]| -> (f64, Vec<f64>) {
        let costs = LbfgsCostSource {
            dict,
            conn_index: &conn_index,
            word_index: &word_index,
            x,
        };
        let mut grad = CrfGradient::new();
        let mut nll = 0.0_f64;
        for &(lattice, gold) in &refs {
            let prob = solver.forward_backward_with(lattice, &costs);
            let log_z = prob.log_z;
            compute_sentence_gradient_with(&prob, lattice, gold, dict, &costs, &mut grad);
            nll += gold_path_cost(gold, &word_base, &word_index, x, &costs) / TEMPERATURE + log_z;
        }
        // Map the (empirical − expected) gradient to the dense vector, scaled by
        // 1/T to match d(NLL)/dx, then fold in the L2 term.
        let mut g = vec![0.0_f64; n_params];
        for (&k, &v) in &grad.conn_gradients {
            if let Some(&i) = conn_index.get(&k) {
                g[i] = v / TEMPERATURE;
            }
        }
        for (&k, &v) in &grad.word_gradients {
            if let Some(&i) = word_index.get(&k) {
                g[i] = v / TEMPERATURE;
            }
        }
        let mut f = nll;
        for (gi, &xi) in g.iter_mut().zip(x) {
            *gi += l2 * xi;
            f += 0.5 * l2 * xi * xi;
        }
        (f, g)
    };

    // ── Minimize ──────────────────────────────────────────────────────────────
    let mut x = vec![0.0_f64; n_params];
    let lbfgs_cfg = LbfgsConfig {
        memory: config.lbfgs_memory.max(1),
        max_iters: config.epochs,
        l1_strength: config.l1_strength,
        ..LbfgsConfig::default()
    };
    let report = lbfgs::minimize(&mut x, &lbfgs_cfg, eval);

    // ── Write the fitted parameters back into the training matrix ─────────────
    let lsize = matrix.lsize();
    {
        let data = matrix.as_slice_mut();
        for (&(r, l), &i) in &conn_index {
            let idx = r as usize + lsize * l as usize;
            if idx < data.len() {
                let fitted = (f64::from(dict.connection_cost(r, l)) + x[i]).round();
                data[idx] = fitted.clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16;
            }
        }
    }
    for (&wid, &i) in &word_index {
        matrix.word_cost_deltas.insert(wid, x[i]);
    }

    if config.verbose {
        eprintln!(
            "L-BFGS: {} params, {} iters, final loss {:.4} (converged={})",
            n_params, report.iterations, report.final_value, report.converged
        );
    }

    DictTrainSummary {
        total_epochs: report.iterations,
        total_sentences: refs.len(),
        total_tokens,
        epoch_stats: vec![EpochStats {
            epoch: 0,
            loss: report.final_value,
            sentences: refs.len(),
            tokens: total_tokens,
            effective_lr: 0.0,
        }],
    }
}
