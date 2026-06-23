//! Limited-memory BFGS (L-BFGS) and OWL-QN minimizers.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! A self-contained, dependency-free quasi-Newton minimizer for smooth convex
//! (and well-behaved non-convex) objectives, plus the OWL-QN extension for
//! `L1`-regularized objectives. It is used to fit the CRF dictionary costs as a
//! *batch* second-order method (see [`super::train_loop`]), but the core
//! [`minimize`] routine is a fully generic `f: ℝⁿ → ℝ` minimizer and is
//! validated independently of the dictionary on problems with known optima
//! (convex quadratic, Rosenbrock, `L1` soft-threshold).
//!
//! # Algorithm
//!
//! L-BFGS approximates the inverse Hessian implicitly from the last `m`
//! curvature pairs `(s_k, y_k)` where `s_k = x_{k+1} − x_k` and
//! `y_k = ∇f_{k+1} − ∇f_k`, computing the search direction via Nocedal's
//! two-loop recursion. A backtracking Armijo line search picks the step length.
//! Curvature pairs with `sᵀy ≤ 0` are skipped to keep the implicit Hessian
//! positive definite.
//!
//! For `l1_strength > 0` the routine switches to **OWL-QN** (Andrew & Gao,
//! 2007): the smooth gradient supplied by the objective is augmented with the
//! `L1` *pseudo-gradient*, the quasi-Newton direction is projected to not cross
//! an orthant boundary, and the line-search iterates are projected back onto the
//! orthant of the chosen pseudo-gradient. The curvature history still uses the
//! smooth gradient only.

// L-BFGS / OWL-QN are conventionally written with single-letter mathematical
// notation (s, y, q, r, α, β, γ, ρ, …); keeping those names matches Nocedal &
// Wright and the OWL-QN paper, so the readability lint is disabled module-wide.
#![allow(clippy::many_single_char_names)]

use std::collections::VecDeque;

/// Hyperparameters for [`minimize`].
#[derive(Debug, Clone, Copy)]
pub struct LbfgsConfig {
    /// History size `m` (number of retained `(s, y)` curvature pairs). Default 8.
    pub memory: usize,
    /// Maximum number of outer iterations. Default 100.
    pub max_iters: usize,
    /// Convergence tolerance on the (pseudo-)gradient ℓ²-norm. Default `1e-5`.
    pub epsilon: f64,
    /// OWL-QN `L1` regularization strength. `0.0` ⇒ plain L-BFGS. Default `0.0`.
    pub l1_strength: f64,
    /// Maximum backtracking steps per line search. Default 24.
    pub max_line_search: usize,
    /// Backtracking shrink factor in `(0, 1)`. Default `0.5`.
    pub line_search_decrease: f64,
    /// Armijo sufficient-decrease coefficient in `(0, 0.5)`. Default `1e-4`.
    pub armijo_c: f64,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            memory: 8,
            max_iters: 100,
            epsilon: 1e-5,
            l1_strength: 0.0,
            max_line_search: 24,
            line_search_decrease: 0.5,
            armijo_c: 1e-4,
        }
    }
}

/// Outcome of a [`minimize`] run.
#[derive(Debug, Clone, Copy)]
pub struct LbfgsReport {
    /// Number of outer iterations actually performed.
    pub iterations: usize,
    /// Final full objective value (including the `L1` term for OWL-QN).
    pub final_value: f64,
    /// Final (pseudo-)gradient ℓ²-norm.
    pub final_grad_norm: f64,
    /// Whether the gradient-norm tolerance was met.
    pub converged: bool,
}

// ── Small vector helpers ────────────────────────────────────────────────────────

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `y ← y + a·x`.
#[inline]
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
fn l1_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

/// OWL-QN pseudo-gradient: the smooth gradient `g` augmented with the
/// minimum-norm subgradient of `l1·‖x‖₁` (Andrew & Gao 2007, eq. 3).
fn pseudo_gradient(x: &[f64], g: &[f64], l1: f64) -> Vec<f64> {
    if l1 == 0.0 {
        return g.to_vec();
    }
    x.iter()
        .zip(g)
        .map(|(&xi, &gi)| {
            if xi > 0.0 {
                gi + l1
            } else if xi < 0.0 {
                gi - l1
            } else if gi + l1 < 0.0 {
                gi + l1
            } else if gi - l1 > 0.0 {
                gi - l1
            } else {
                0.0
            }
        })
        .collect()
}

/// Minimize `eval` starting from `x`, updating `x` in place to the minimizer.
///
/// `eval(x) -> (value, gradient)` must return the **smooth** objective value and
/// its gradient (the `L1` term is added internally when `config.l1_strength >
/// 0`). `value` and `gradient` must be mutually consistent (`gradient = ∇value`)
/// or the line search may fail to make progress.
///
/// Returns an [`LbfgsReport`]. `x` always holds the best point evaluated.
#[allow(clippy::too_many_lines)]
pub fn minimize<F>(x: &mut [f64], config: &LbfgsConfig, mut eval: F) -> LbfgsReport
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let n = x.len();
    let l1 = config.l1_strength;

    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(config.memory);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(config.memory);
    let mut rho_hist: VecDeque<f64> = VecDeque::with_capacity(config.memory);

    let (mut smooth_f, mut g) = eval(x);
    let mut value = smooth_f + l1 * l1_norm(x);

    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..config.max_iters {
        let pg = pseudo_gradient(x, &g, l1);
        let pg_norm = l2_norm(&pg);
        if pg_norm <= config.epsilon {
            converged = true;
            break;
        }

        // ── Two-loop recursion: r ≈ H·pg (implicit inverse Hessian) ──────────
        let k = s_hist.len();
        let mut q = pg.clone();
        let mut alphas = vec![0.0_f64; k];
        for i in (0..k).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &q);
            alphas[i] = a;
            axpy(&mut q, -a, &y_hist[i]);
        }
        // Initial Hessian scaling γ = sᵀy / yᵀy from the most recent pair.
        let gamma = if k > 0 {
            let sy = dot(&s_hist[k - 1], &y_hist[k - 1]);
            let yy = dot(&y_hist[k - 1], &y_hist[k - 1]);
            if yy > 0.0 { sy / yy } else { 1.0 }
        } else {
            1.0
        };
        let mut r: Vec<f64> = q.iter().map(|&v| v * gamma).collect();
        for i in 0..k {
            let beta = rho_hist[i] * dot(&y_hist[i], &r);
            axpy(&mut r, alphas[i] - beta, &s_hist[i]);
        }

        // Search direction d = −r, OWL-QN-projected to align with −pg.
        let mut d: Vec<f64> = r.iter().map(|&v| -v).collect();
        if l1 > 0.0 {
            for (di, &pgi) in d.iter_mut().zip(&pg) {
                if *di * -pgi <= 0.0 {
                    *di = 0.0;
                }
            }
        }
        // Guard: fall back to steepest descent if d is not a descent direction.
        if dot(&pg, &d) >= 0.0 {
            d = pg.iter().map(|&v| -v).collect();
            if l1 > 0.0 {
                for (di, &pgi) in d.iter_mut().zip(&pg) {
                    if *di * -pgi <= 0.0 {
                        *di = 0.0;
                    }
                }
            }
        }

        // ── Backtracking Armijo line search (orthant-projected for OWL-QN) ───
        let x_old = x.to_vec();
        let g_old = g.clone();
        // Orthant of the step: sign(x_i) where defined, else sign(−pg_i).
        let xi: Vec<f64> = if l1 > 0.0 {
            x_old
                .iter()
                .zip(&pg)
                .map(|(&xv, &pgv)| {
                    if xv == 0.0 {
                        (-pgv).signum()
                    } else {
                        xv.signum()
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        // First step heuristic: 1/‖pg‖ on iteration 0, then unit step.
        let mut t = if k == 0 { 1.0 / pg_norm.max(1.0) } else { 1.0 };

        let mut ls_ok = false;
        let mut new_value = value;
        for _ in 0..config.max_line_search {
            for idx in 0..n {
                let mut cand = x_old[idx] + t * d[idx];
                if l1 > 0.0 && cand * xi[idx] < 0.0 {
                    cand = 0.0; // project back onto the orthant
                }
                x[idx] = cand;
            }
            let (f_new, g_new) = eval(x);
            new_value = f_new + l1 * l1_norm(x);
            // Sufficient decrease along the actual (possibly projected) step.
            let actual: f64 = (0..n).map(|i| pg[i] * (x[i] - x_old[i])).sum();
            if new_value <= value + config.armijo_c * actual {
                g = g_new;
                ls_ok = true;
                break;
            }
            t *= config.line_search_decrease;
        }
        if !ls_ok {
            x.copy_from_slice(&x_old);
            g = g_old;
            break;
        }

        // ── Update curvature history (skip non-positive curvature) ───────────
        let s: Vec<f64> = (0..n).map(|i| x[i] - x_old[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| g[i] - g_old[i]).collect();
        let sy = dot(&s, &y);
        if sy > 1e-10 {
            if s_hist.len() == config.memory {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            rho_hist.push_back(1.0 / sy);
            s_hist.push_back(s);
            y_hist.push_back(y);
        }

        value = new_value;
        smooth_f = new_value - l1 * l1_norm(x);
        let _ = smooth_f; // retained for clarity; not needed past this point
        iterations += 1;
    }

    let final_grad_norm = l2_norm(&pseudo_gradient(x, &g, l1));
    LbfgsReport {
        iterations,
        final_value: value,
        final_grad_norm,
        converged,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{LbfgsConfig, minimize};

    /// Separable convex quadratic `f(x) = Σ (x_i − t_i)²`. Optimum is `x = t`,
    /// `f = 0`; L-BFGS must reach it to high precision.
    #[test]
    fn test_minimize_convex_quadratic() {
        let target = [3.0_f64, -2.0, 7.5, 0.25, -10.0];
        let cfg = LbfgsConfig {
            epsilon: 1e-8,
            max_iters: 200,
            ..LbfgsConfig::default()
        };
        let mut x = vec![0.0_f64; target.len()];
        let report = minimize(&mut x, &cfg, |x| {
            let f = x
                .iter()
                .zip(target)
                .map(|(&xi, ti)| (xi - ti).powi(2))
                .sum();
            let g = x
                .iter()
                .zip(target)
                .map(|(&xi, ti)| 2.0 * (xi - ti))
                .collect();
            (f, g)
        });
        assert!(report.converged, "must converge: {report:?}");
        for (&xi, ti) in x.iter().zip(target) {
            assert!((xi - ti).abs() < 1e-4, "x={xi} target={ti}");
        }
        assert!(report.final_value < 1e-8, "f={}", report.final_value);
    }

    /// Rosenbrock `f(x,y) = (1−x)² + 100(y−x²)²` — a hard non-convex valley with
    /// optimum `(1, 1)`. L-BFGS with backtracking must still converge.
    #[test]
    fn test_minimize_rosenbrock() {
        let cfg = LbfgsConfig {
            epsilon: 1e-7,
            max_iters: 1000,
            memory: 10,
            ..LbfgsConfig::default()
        };
        let mut x = vec![-1.2_f64, 1.0];
        let report = minimize(&mut x, &cfg, |x| {
            let (a, b) = (x[0], x[1]);
            let f = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
            let ga = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
            let gb = 200.0 * (b - a * a);
            (f, vec![ga, gb])
        });
        assert!((x[0] - 1.0).abs() < 1e-3, "x={} report={report:?}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-3, "y={}", x[1]);
    }

    /// OWL-QN on `f(x) = ½(x − t)² + C‖x‖₁` (separable). The optimum is the
    /// soft-threshold `x*_i = sign(t_i)·max(|t_i| − C, 0)`, so components with
    /// `|t_i| ≤ C` must be driven *exactly* to zero.
    #[test]
    fn test_minimize_owlqn_soft_threshold() {
        let target = [3.0_f64, -0.5, 0.8, -4.0, 0.0];
        let c = 1.0_f64;
        let cfg = LbfgsConfig {
            l1_strength: c,
            epsilon: 1e-8,
            max_iters: 300,
            ..LbfgsConfig::default()
        };
        let mut x = vec![0.0_f64; target.len()];
        minimize(&mut x, &cfg, |x| {
            let f = x
                .iter()
                .zip(target)
                .map(|(&xi, ti)| 0.5 * (xi - ti).powi(2))
                .sum();
            let g = x.iter().zip(target).map(|(&xi, ti)| xi - ti).collect();
            (f, g)
        });
        for (&xi, ti) in x.iter().zip(target) {
            let expected = ti.signum() * (ti.abs() - c).max(0.0);
            assert!(
                (xi - expected).abs() < 1e-4,
                "soft-threshold mismatch: got {xi}, expected {expected} (t={ti})"
            );
        }
        // |t| ≤ C components must be *exactly* zero (sparsity). Comparing to the
        // `0.0` literal keeps clippy::float_cmp satisfied while asserting exactness.
        assert!(x[1] == 0.0, "|t|=0.5 ≤ C must be exactly 0, got {}", x[1]);
        assert!(x[2] == 0.0, "|t|=0.8 ≤ C must be exactly 0, got {}", x[2]);
        assert!(x[4] == 0.0, "t=0 must be exactly 0, got {}", x[4]);
    }

    /// An already-optimal start converges in zero iterations.
    #[test]
    fn test_minimize_already_optimal() {
        let cfg = LbfgsConfig::default();
        let mut x = vec![0.0_f64; 3];
        let report = minimize(&mut x, &cfg, |x| {
            (
                x.iter().map(|v| v * v).sum(),
                x.iter().map(|&v| 2.0 * v).collect(),
            )
        });
        assert!(report.converged);
        assert_eq!(report.iterations, 0);
    }
}
