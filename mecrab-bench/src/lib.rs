//! MeCrab Benchmark Suite
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This crate contains criterion benchmarks for MeCrab.
//! It's kept separate from the core mecrab crate to avoid
//! pulling in heavy dependencies for regular builds.
//!
//! # Benchmark Regression Tracking
//!
//! The [`BenchBaseline`] type allows you to record baseline measurements and
//! detect regressions between benchmark runs.
//!
//! ```no_run
//! use mecrab_bench::{BenchBaseline, BenchRecord};
//! use std::path::Path;
//!
//! // Load (or create) a baseline file
//! let mut baseline = BenchBaseline::load_or_create(Path::new("/tmp/mecrab_baseline.json"));
//!
//! // Record a new measurement
//! baseline.record("parse/medium", 42_000.0, 500.0);
//!
//! // Check for regression (>10 % slower than baseline)
//! if let Some((expected, actual)) = baseline.check_regression("parse/medium", 50_000.0, 10.0) {
//!     eprintln!("REGRESSION: {expected:.0} ns → {actual:.0} ns");
//! }
//!
//! // Persist the updated baseline
//! baseline.save().expect("Failed to save baseline");
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ── BenchRecord ──────────────────────────────────────────────────────────────

/// A single benchmark baseline record.
///
/// Stores the benchmark name, a mean latency in nanoseconds, the standard
/// deviation, and a Unix-epoch timestamp of when the measurement was recorded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRecord {
    /// Benchmark identifier, e.g. `"parse/medium"`.
    pub name: String,
    /// Mean duration in nanoseconds.
    pub mean_ns: f64,
    /// Standard deviation in nanoseconds.
    pub std_dev_ns: f64,
    /// Unix epoch seconds when this record was captured.
    pub recorded_at_secs: u64,
}

impl BenchRecord {
    /// Create a new record with the current wall-clock time.
    pub fn new(name: impl Into<String>, mean_ns: f64, std_dev_ns: f64) -> Self {
        let recorded_at_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            name: name.into(),
            mean_ns,
            std_dev_ns,
            recorded_at_secs,
        }
    }
}

// ── BenchBaseline ─────────────────────────────────────────────────────────────

/// Persistent store for benchmark baselines.
///
/// The baseline is serialised as a JSON object whose keys are benchmark names
/// and whose values are [`BenchRecord`] objects.  This makes it easy to diff
/// two baseline files in version control.
///
/// # Typical workflow
///
/// 1. Call [`BenchBaseline::load_or_create`] at the start of a bench run.
/// 2. After each benchmark, call [`BenchBaseline::record`] with the measured
///    mean and standard deviation.
/// 3. Optionally call [`BenchBaseline::check_regression`] to assert that the
///    new measurement does not exceed a threshold relative to the saved value.
/// 4. Call [`BenchBaseline::save`] to persist the updated baseline for the
///    next run.
pub struct BenchBaseline {
    records: HashMap<String, BenchRecord>,
    path: PathBuf,
}

impl BenchBaseline {
    /// Load an existing baseline from `path`, or create an empty one if the
    /// file does not exist or cannot be parsed.
    pub fn load_or_create(path: &Path) -> Self {
        let records = if path.exists() {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|data| serde_json::from_str(&data).ok())
                .unwrap_or_default()
        } else {
            HashMap::new()
        };
        Self {
            records,
            path: path.to_path_buf(),
        }
    }

    /// Persist the current baseline to disk (pretty-printed JSON).
    ///
    /// Creates parent directories if they do not exist.
    pub fn save(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.records)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&self.path, json)
    }

    /// Record (or overwrite) the baseline for `name`.
    pub fn record(&mut self, name: &str, mean_ns: f64, std_dev_ns: f64) {
        self.records.insert(
            name.to_string(),
            BenchRecord::new(name, mean_ns, std_dev_ns),
        );
    }

    /// Check whether `actual_ns` represents a regression compared to the
    /// stored baseline for `name`.
    ///
    /// Returns `Some((baseline_mean_ns, actual_ns))` when the actual
    /// measurement exceeds `baseline_mean * (1 + threshold_pct / 100)`.
    /// Returns `None` when there is no baseline entry for `name`, or when the
    /// actual measurement is within the allowed threshold.
    pub fn check_regression(
        &self,
        name: &str,
        actual_ns: f64,
        threshold_pct: f64,
    ) -> Option<(f64, f64)> {
        self.records.get(name).and_then(|baseline| {
            let regression_limit = baseline.mean_ns * (1.0 + threshold_pct / 100.0);
            if actual_ns > regression_limit {
                Some((baseline.mean_ns, actual_ns))
            } else {
                None
            }
        })
    }

    /// Return the stored [`BenchRecord`] for `name`, if any.
    pub fn get(&self, name: &str) -> Option<&BenchRecord> {
        self.records.get(name)
    }

    /// Return `true` if a baseline exists for `name`.
    pub fn has_baseline(&self, name: &str) -> bool {
        self.records.contains_key(name)
    }

    /// Remove a single benchmark entry from the baseline.
    ///
    /// Returns the removed record, or `None` if it was not present.
    pub fn remove(&mut self, name: &str) -> Option<BenchRecord> {
        self.records.remove(name)
    }

    /// Number of stored baseline entries.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// `true` when no baselines have been recorded.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Iterate over all stored records.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &BenchRecord)> {
        self.records.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Produce a human-readable comparison report between `self` (the reference
    /// baseline) and `current` measurements.
    ///
    /// Each entry in `current` is compared to the matching record in `self`.
    /// Missing baselines are flagged as "NEW", regressions as "REGRESSED", and
    /// improvements as "IMPROVED".
    pub fn comparison_report(
        &self,
        current: &HashMap<String, f64>,
        threshold_pct: f64,
    ) -> Vec<RegressionEntry> {
        let mut entries = Vec::new();
        for (name, &actual_ns) in current {
            let status = match self.records.get(name) {
                None => RegressionStatus::New,
                Some(baseline) => {
                    let limit = baseline.mean_ns * (1.0 + threshold_pct / 100.0);
                    let improve = baseline.mean_ns * (1.0 - threshold_pct / 100.0);
                    if actual_ns > limit {
                        RegressionStatus::Regressed {
                            baseline_ns: baseline.mean_ns,
                        }
                    } else if actual_ns < improve {
                        RegressionStatus::Improved {
                            baseline_ns: baseline.mean_ns,
                        }
                    } else {
                        RegressionStatus::Stable {
                            baseline_ns: baseline.mean_ns,
                        }
                    }
                }
            };
            entries.push(RegressionEntry {
                name: name.clone(),
                actual_ns,
                status,
            });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }
}

// ── RegressionEntry / RegressionStatus ───────────────────────────────────────

/// Status of a single benchmark compared to its baseline.
#[derive(Debug, Clone)]
pub enum RegressionStatus {
    /// No prior baseline exists for this benchmark.
    New,
    /// Measurement is within the allowed threshold of the baseline.
    Stable { baseline_ns: f64 },
    /// Measurement exceeds the allowed threshold — performance has degraded.
    Regressed { baseline_ns: f64 },
    /// Measurement is better than `(1 - threshold) * baseline` — improvement.
    Improved { baseline_ns: f64 },
}

/// One entry in a regression comparison report.
#[derive(Debug, Clone)]
pub struct RegressionEntry {
    /// Benchmark name.
    pub name: String,
    /// Measured mean latency in nanoseconds.
    pub actual_ns: f64,
    /// Comparison status relative to the stored baseline.
    pub status: RegressionStatus,
}

impl RegressionEntry {
    /// `true` if this entry represents a regression.
    pub fn is_regression(&self) -> bool {
        matches!(self.status, RegressionStatus::Regressed { .. })
    }

    /// Format as a one-line human-readable string.
    pub fn summary(&self) -> String {
        match &self.status {
            RegressionStatus::New => {
                format!(
                    "[NEW     ] {} — {:.0} ns (no prior baseline)",
                    self.name, self.actual_ns
                )
            }
            RegressionStatus::Stable { baseline_ns } => {
                let pct = (self.actual_ns - baseline_ns) / baseline_ns * 100.0;
                format!(
                    "[STABLE  ] {} — {:.0} ns (baseline {:.0} ns, {:+.1}%)",
                    self.name, self.actual_ns, baseline_ns, pct
                )
            }
            RegressionStatus::Regressed { baseline_ns } => {
                let pct = (self.actual_ns - baseline_ns) / baseline_ns * 100.0;
                format!(
                    "[REGRESS ] {} — {:.0} ns (baseline {:.0} ns, {:+.1}%)",
                    self.name, self.actual_ns, baseline_ns, pct
                )
            }
            RegressionStatus::Improved { baseline_ns } => {
                let pct = (self.actual_ns - baseline_ns) / baseline_ns * 100.0;
                format!(
                    "[IMPROVE ] {} — {:.0} ns (baseline {:.0} ns, {:+.1}%)",
                    self.name, self.actual_ns, baseline_ns, pct
                )
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_baseline_path() -> PathBuf {
        let mut p = env::temp_dir();
        p.push(format!(
            "mecrab_test_baseline_{}.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn test_record_and_retrieve() {
        let path = temp_baseline_path();
        let mut baseline = BenchBaseline::load_or_create(&path);
        assert!(baseline.is_empty());

        baseline.record("parse/short", 15_000.0, 200.0);
        assert_eq!(baseline.len(), 1);

        let rec = baseline.get("parse/short").expect("record should exist");
        assert!((rec.mean_ns - 15_000.0).abs() < f64::EPSILON);
        assert!((rec.std_dev_ns - 200.0).abs() < f64::EPSILON);

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_and_reload() {
        let path = temp_baseline_path();
        {
            let mut baseline = BenchBaseline::load_or_create(&path);
            baseline.record("parse/medium", 42_000.0, 500.0);
            baseline.save().expect("save should succeed");
        }
        {
            let baseline = BenchBaseline::load_or_create(&path);
            let rec = baseline
                .get("parse/medium")
                .expect("record should survive round-trip");
            assert!((rec.mean_ns - 42_000.0).abs() < f64::EPSILON);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_no_regression_within_threshold() {
        let path = temp_baseline_path();
        let mut baseline = BenchBaseline::load_or_create(&path);
        baseline.record("viterbi/short", 10_000.0, 100.0);

        // 5 % slower than baseline, threshold is 10 % — should NOT trigger
        let result = baseline.check_regression("viterbi/short", 10_500.0, 10.0);
        assert!(result.is_none(), "no regression expected within threshold");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_regression_detected() {
        let path = temp_baseline_path();
        let mut baseline = BenchBaseline::load_or_create(&path);
        baseline.record("viterbi/long", 50_000.0, 1_000.0);

        // 20 % slower than baseline, threshold is 10 % — should trigger
        let result = baseline.check_regression("viterbi/long", 60_000.0, 10.0);
        assert!(result.is_some(), "regression should be detected");
        let (expected, actual) = result.expect("result is Some");
        assert!((expected - 50_000.0).abs() < f64::EPSILON);
        assert!((actual - 60_000.0).abs() < f64::EPSILON);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_missing_baseline_returns_none() {
        let path = temp_baseline_path();
        let baseline = BenchBaseline::load_or_create(&path);

        // No record for this key → check_regression should return None
        let result = baseline.check_regression("nonexistent/bench", 99_999.0, 5.0);
        assert!(result.is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_comparison_report() {
        let path = temp_baseline_path();
        let mut baseline = BenchBaseline::load_or_create(&path);
        baseline.record("a", 1_000.0, 10.0);
        baseline.record("b", 2_000.0, 20.0);
        baseline.record("c", 3_000.0, 30.0);

        let mut current = HashMap::new();
        current.insert("a".to_string(), 1_050.0); // stable (+5 %)
        current.insert("b".to_string(), 2_500.0); // regressed (+25 %)
        current.insert("c".to_string(), 2_400.0); // improved (−20 %)
        current.insert("d".to_string(), 900.0); // new

        let report = baseline.comparison_report(&current, 10.0);
        assert_eq!(report.len(), 4);

        let regressions: Vec<_> = report.iter().filter(|e| e.is_regression()).collect();
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].name, "b");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_remove_record() {
        let path = temp_baseline_path();
        let mut baseline = BenchBaseline::load_or_create(&path);
        baseline.record("tmp", 5_000.0, 50.0);
        assert!(baseline.has_baseline("tmp"));

        let removed = baseline.remove("tmp");
        assert!(removed.is_some());
        assert!(!baseline.has_baseline("tmp"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_regression_entry_summary() {
        let entry = RegressionEntry {
            name: "parse/short".to_string(),
            actual_ns: 18_000.0,
            status: RegressionStatus::Regressed {
                baseline_ns: 15_000.0,
            },
        };
        let summary = entry.summary();
        assert!(summary.contains("REGRESS"));
        assert!(summary.contains("parse/short"));
    }
}
