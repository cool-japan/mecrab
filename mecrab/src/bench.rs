//! Benchmarking and profiling utilities for MeCrab
//!
//! Provides tools for measuring and analyzing performance
//! of morphological analysis operations.

#![allow(clippy::cast_precision_loss)]

use std::time::{Duration, Instant};

/// Timing result for a single operation
#[derive(Debug, Clone)]
pub struct TimingResult {
    /// Operation name
    pub name: String,
    /// Total duration
    pub duration: Duration,
    /// Number of iterations
    pub iterations: usize,
    /// Bytes processed (if applicable)
    pub bytes: Option<u64>,
    /// Characters processed (if applicable)
    pub chars: Option<u64>,
}

impl TimingResult {
    /// Create a new timing result
    pub fn new(name: impl Into<String>, duration: Duration, iterations: usize) -> Self {
        Self {
            name: name.into(),
            duration,
            iterations,
            bytes: None,
            chars: None,
        }
    }

    /// Add byte count
    pub fn with_bytes(mut self, bytes: u64) -> Self {
        self.bytes = Some(bytes);
        self
    }

    /// Add character count
    pub fn with_chars(mut self, chars: u64) -> Self {
        self.chars = Some(chars);
        self
    }

    /// Get average duration per iteration
    pub fn avg_duration(&self) -> Duration {
        if self.iterations == 0 {
            Duration::ZERO
        } else {
            self.duration / self.iterations as u32
        }
    }

    /// Get operations per second
    pub fn ops_per_second(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            0.0
        } else {
            self.iterations as f64 / secs
        }
    }

    /// Get bytes per second (throughput)
    pub fn bytes_per_second(&self) -> Option<f64> {
        self.bytes.map(|b| {
            let secs = self.duration.as_secs_f64();
            if secs == 0.0 { 0.0 } else { b as f64 / secs }
        })
    }

    /// Get characters per second
    pub fn chars_per_second(&self) -> Option<f64> {
        self.chars.map(|c| {
            let secs = self.duration.as_secs_f64();
            if secs == 0.0 { 0.0 } else { c as f64 / secs }
        })
    }

    /// Get megabytes per second
    pub fn mb_per_second(&self) -> Option<f64> {
        self.bytes_per_second().map(|b| b / (1024.0 * 1024.0))
    }
}

impl std::fmt::Display for TimingResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}:", self.name)?;
        writeln!(f, "  Total time: {:?}", self.duration)?;
        writeln!(f, "  Iterations: {}", self.iterations)?;
        writeln!(f, "  Avg/iter: {:?}", self.avg_duration())?;
        writeln!(f, "  Ops/sec: {:.2}", self.ops_per_second())?;
        if let Some(mb) = self.mb_per_second() {
            writeln!(f, "  Throughput: {:.2} MB/s", mb)?;
        }
        if let Some(cps) = self.chars_per_second() {
            writeln!(f, "  Chars/sec: {:.0}", cps)?;
        }
        Ok(())
    }
}

/// Timer for measuring operation duration
#[derive(Debug)]
pub struct Timer {
    start: Instant,
    laps: Vec<(String, Duration)>,
}

impl Timer {
    /// Start a new timer
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            laps: Vec::new(),
        }
    }

    /// Record a lap time
    pub fn lap(&mut self, name: impl Into<String>) {
        let elapsed = self.start.elapsed();
        self.laps.push((name.into(), elapsed));
    }

    /// Get total elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get all laps
    pub fn laps(&self) -> &[(String, Duration)] {
        &self.laps
    }

    /// Reset the timer
    pub fn reset(&mut self) {
        self.start = Instant::now();
        self.laps.clear();
    }

    /// Get lap durations (time between laps)
    pub fn lap_durations(&self) -> Vec<(String, Duration)> {
        let mut prev = Duration::ZERO;
        self.laps
            .iter()
            .map(|(name, total)| {
                let lap_time = total.saturating_sub(prev);
                prev = *total;
                (name.clone(), lap_time)
            })
            .collect()
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark runner for repeated measurements
#[derive(Debug)]
pub struct Benchmark {
    /// Warmup iterations
    pub warmup: usize,
    /// Measurement iterations
    pub iterations: usize,
    /// Minimum measurement time
    pub min_time: Duration,
}

impl Benchmark {
    /// Create a new benchmark with default settings
    pub fn new() -> Self {
        Self {
            warmup: 3,
            iterations: 10,
            min_time: Duration::from_millis(100),
        }
    }

    /// Set warmup iterations
    #[must_use]
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    /// Set measurement iterations
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set minimum measurement time
    #[must_use]
    pub fn with_min_time(mut self, min_time: Duration) -> Self {
        self.min_time = min_time;
        self
    }

    /// Run a benchmark
    pub fn run<F, R>(&self, name: impl Into<String>, mut f: F) -> TimingResult
    where
        F: FnMut() -> R,
    {
        let name = name.into();

        // Warmup
        for _ in 0..self.warmup {
            let _ = f();
        }

        // Determine number of iterations
        let start = Instant::now();
        for _ in 0..self.iterations {
            let _ = f();
        }
        let initial_time = start.elapsed();

        // Calculate iterations needed for minimum time
        let iterations = if initial_time < self.min_time && initial_time > Duration::ZERO {
            let ratio = self.min_time.as_nanos() / initial_time.as_nanos().max(1);
            (self.iterations as u128 * ratio).max(self.iterations as u128) as usize
        } else {
            self.iterations
        };

        // Actual measurement
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = f();
        }
        let duration = start.elapsed();

        TimingResult::new(name, duration, iterations)
    }

    /// Run benchmark with byte counting
    pub fn run_with_bytes<F, R>(&self, name: impl Into<String>, bytes: u64, f: F) -> TimingResult
    where
        F: FnMut() -> R,
    {
        self.run(name, f).with_bytes(bytes)
    }

    /// Run benchmark with character counting
    pub fn run_with_chars<F, R>(&self, name: impl Into<String>, chars: u64, f: F) -> TimingResult
    where
        F: FnMut() -> R,
    {
        self.run(name, f).with_chars(chars)
    }
}

impl Default for Benchmark {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of benchmark results
#[derive(Debug, Default)]
pub struct BenchmarkSuite {
    results: Vec<TimingResult>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a result
    pub fn add(&mut self, result: TimingResult) {
        self.results.push(result);
    }

    /// Get all results
    pub fn results(&self) -> &[TimingResult] {
        &self.results
    }

    /// Find fastest result
    pub fn fastest(&self) -> Option<&TimingResult> {
        self.results.iter().min_by_key(|r| r.avg_duration())
    }

    /// Find slowest result
    pub fn slowest(&self) -> Option<&TimingResult> {
        self.results.iter().max_by_key(|r| r.avg_duration())
    }

    /// Get summary statistics
    pub fn summary(&self) -> BenchmarkSummary {
        if self.results.is_empty() {
            return BenchmarkSummary::default();
        }

        let total_duration: Duration = self.results.iter().map(|r| r.duration).sum();
        let total_iterations: usize = self.results.iter().map(|r| r.iterations).sum();
        let avg_ops_per_sec = self
            .results
            .iter()
            .map(TimingResult::ops_per_second)
            .sum::<f64>()
            / self.results.len() as f64;

        BenchmarkSummary {
            benchmark_count: self.results.len(),
            total_duration,
            total_iterations,
            avg_ops_per_sec,
        }
    }
}

/// Summary of benchmark results
#[derive(Debug, Clone, Default)]
pub struct BenchmarkSummary {
    /// Number of benchmarks run
    pub benchmark_count: usize,
    /// Total time spent
    pub total_duration: Duration,
    /// Total iterations across all benchmarks
    pub total_iterations: usize,
    /// Average operations per second
    pub avg_ops_per_sec: f64,
}

impl std::fmt::Display for BenchmarkSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Benchmark Summary:")?;
        writeln!(f, "  Benchmarks run: {}", self.benchmark_count)?;
        writeln!(f, "  Total time: {:?}", self.total_duration)?;
        writeln!(f, "  Total iterations: {}", self.total_iterations)?;
        writeln!(f, "  Avg ops/sec: {:.2}", self.avg_ops_per_sec)?;
        Ok(())
    }
}

/// Memory usage tracker
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_bytes: u64,
    /// Current memory usage in bytes
    pub current_bytes: u64,
    /// Number of allocations
    pub allocations: u64,
    /// Number of deallocations
    pub deallocations: u64,
}

impl MemoryStats {
    /// Create new empty stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an allocation
    pub fn record_alloc(&mut self, bytes: u64) {
        self.current_bytes += bytes;
        self.allocations += 1;
        if self.current_bytes > self.peak_bytes {
            self.peak_bytes = self.current_bytes;
        }
    }

    /// Record a deallocation
    pub fn record_dealloc(&mut self, bytes: u64) {
        self.current_bytes = self.current_bytes.saturating_sub(bytes);
        self.deallocations += 1;
    }

    /// Get peak memory in megabytes
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get current memory in megabytes
    pub fn current_mb(&self) -> f64 {
        self.current_bytes as f64 / (1024.0 * 1024.0)
    }
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Statistics:")?;
        writeln!(f, "  Peak: {:.2} MB", self.peak_mb())?;
        writeln!(f, "  Current: {:.2} MB", self.current_mb())?;
        writeln!(f, "  Allocations: {}", self.allocations)?;
        writeln!(f, "  Deallocations: {}", self.deallocations)?;
        Ok(())
    }
}

/// Throughput measurement
#[derive(Debug, Clone)]
pub struct Throughput {
    /// Items processed per second
    pub items_per_second: f64,
    /// Bytes processed per second
    pub bytes_per_second: f64,
    /// Measurement duration
    pub duration: Duration,
}

impl Throughput {
    /// Calculate throughput from items, bytes, and duration
    pub fn calculate(items: u64, bytes: u64, duration: Duration) -> Self {
        let secs = duration.as_secs_f64();
        Self {
            items_per_second: if secs > 0.0 { items as f64 / secs } else { 0.0 },
            bytes_per_second: if secs > 0.0 { bytes as f64 / secs } else { 0.0 },
            duration,
        }
    }

    /// Get megabytes per second
    pub fn mb_per_second(&self) -> f64 {
        self.bytes_per_second / (1024.0 * 1024.0)
    }

    /// Get gigabytes per second
    pub fn gb_per_second(&self) -> f64 {
        self.bytes_per_second / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get thousands of items per second
    pub fn k_items_per_second(&self) -> f64 {
        self.items_per_second / 1000.0
    }
}

impl std::fmt::Display for Throughput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.2}K items/s, {:.2} MB/s",
            self.k_items_per_second(),
            self.mb_per_second()
        )
    }
}

/// Parse rate statistics
#[derive(Debug, Clone, Default)]
pub struct ParseStats {
    /// Number of texts parsed
    pub texts_parsed: u64,
    /// Total characters processed
    pub total_chars: u64,
    /// Total bytes processed
    pub total_bytes: u64,
    /// Total tokens generated
    pub total_tokens: u64,
    /// Total parsing time
    pub total_time: Duration,
    /// Minimum parse time
    pub min_time: Duration,
    /// Maximum parse time
    pub max_time: Duration,
}

impl ParseStats {
    /// Create new empty stats
    pub fn new() -> Self {
        Self {
            min_time: Duration::MAX,
            ..Default::default()
        }
    }

    /// Record a parse operation
    pub fn record(&mut self, chars: usize, bytes: usize, tokens: usize, time: Duration) {
        self.texts_parsed += 1;
        self.total_chars += chars as u64;
        self.total_bytes += bytes as u64;
        self.total_tokens += tokens as u64;
        self.total_time += time;

        if time < self.min_time {
            self.min_time = time;
        }
        if time > self.max_time {
            self.max_time = time;
        }
    }

    /// Get average parse time
    pub fn avg_time(&self) -> Duration {
        if self.texts_parsed == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.texts_parsed as u32
        }
    }

    /// Get characters per second
    pub fn chars_per_second(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs > 0.0 {
            self.total_chars as f64 / secs
        } else {
            0.0
        }
    }

    /// Get tokens per second
    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs > 0.0 {
            self.total_tokens as f64 / secs
        } else {
            0.0
        }
    }

    /// Get average tokens per text
    pub fn avg_tokens_per_text(&self) -> f64 {
        if self.texts_parsed == 0 {
            0.0
        } else {
            self.total_tokens as f64 / self.texts_parsed as f64
        }
    }

    /// Get throughput
    pub fn throughput(&self) -> Throughput {
        Throughput::calculate(self.total_tokens, self.total_bytes, self.total_time)
    }
}

impl std::fmt::Display for ParseStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Parse Statistics:")?;
        writeln!(f, "  Texts parsed: {}", self.texts_parsed)?;
        writeln!(f, "  Total chars: {}", self.total_chars)?;
        writeln!(f, "  Total tokens: {}", self.total_tokens)?;
        writeln!(f, "  Total time: {:?}", self.total_time)?;
        writeln!(f, "  Avg time: {:?}", self.avg_time())?;
        if self.min_time != Duration::MAX {
            writeln!(f, "  Min time: {:?}", self.min_time)?;
        }
        writeln!(f, "  Max time: {:?}", self.max_time)?;
        writeln!(f, "  Chars/sec: {:.0}", self.chars_per_second())?;
        writeln!(f, "  Tokens/sec: {:.0}", self.tokens_per_second())?;
        Ok(())
    }
}

/// Histogram for distribution analysis
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bucket counts
    buckets: Vec<u64>,
    /// Bucket boundaries (n+1 elements for n buckets)
    boundaries: Vec<f64>,
    /// Total count
    total: u64,
    /// Sum of all values
    sum: f64,
    /// Minimum value
    min: f64,
    /// Maximum value
    max: f64,
}

impl Histogram {
    /// Create a linear histogram
    pub fn linear(min: f64, max: f64, bucket_count: usize) -> Self {
        let bucket_count = bucket_count.max(1);
        let step = (max - min) / bucket_count as f64;
        let boundaries: Vec<f64> = (0..=bucket_count).map(|i| min + i as f64 * step).collect();

        Self {
            buckets: vec![0; bucket_count],
            boundaries,
            total: 0,
            sum: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Create a logarithmic histogram
    pub fn logarithmic(min: f64, max: f64, bucket_count: usize) -> Self {
        let bucket_count = bucket_count.max(1);
        let log_min = min.max(f64::MIN_POSITIVE).ln();
        let log_max = max.ln();
        let step = (log_max - log_min) / bucket_count as f64;
        let boundaries: Vec<f64> = (0..=bucket_count)
            .map(|i| (log_min + i as f64 * step).exp())
            .collect();

        Self {
            buckets: vec![0; bucket_count],
            boundaries,
            total: 0,
            sum: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Record a value
    pub fn record(&mut self, value: f64) {
        self.total += 1;
        self.sum += value;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        // Find bucket
        for (i, window) in self.boundaries.windows(2).enumerate() {
            if value >= window[0] && value < window[1] {
                self.buckets[i] += 1;
                return;
            }
        }

        // Value out of range, put in last bucket
        if !self.buckets.is_empty() {
            *self.buckets.last_mut().unwrap() += 1;
        }
    }

    /// Get bucket counts
    pub fn buckets(&self) -> &[u64] {
        &self.buckets
    }

    /// Get bucket boundaries
    pub fn boundaries(&self) -> &[f64] {
        &self.boundaries
    }

    /// Get total count
    pub fn count(&self) -> u64 {
        self.total
    }

    /// Get mean value
    pub fn mean(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.sum / self.total as f64
        }
    }

    /// Get percentile (0-100)
    pub fn percentile(&self, p: f64) -> f64 {
        if self.total == 0 {
            return 0.0;
        }

        let target = (self.total as f64 * p / 100.0) as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return f64::midpoint(self.boundaries[i], self.boundaries[i + 1]);
            }
        }

        self.max
    }

    /// Get p50 (median)
    pub fn p50(&self) -> f64 {
        self.percentile(50.0)
    }

    /// Get p95
    pub fn p95(&self) -> f64 {
        self.percentile(95.0)
    }

    /// Get p99
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }
}

impl std::fmt::Display for Histogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Histogram (n={}):", self.total)?;
        writeln!(f, "  Min: {:.2}", self.min)?;
        writeln!(f, "  Max: {:.2}", self.max)?;
        writeln!(f, "  Mean: {:.2}", self.mean())?;
        writeln!(f, "  P50: {:.2}", self.p50())?;
        writeln!(f, "  P95: {:.2}", self.p95())?;
        writeln!(f, "  P99: {:.2}", self.p99())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_timing_result() {
        let result = TimingResult::new("test", Duration::from_secs(1), 100)
            .with_bytes(1000)
            .with_chars(500);

        assert_eq!(result.iterations, 100);
        assert_eq!(result.avg_duration(), Duration::from_millis(10));
        assert_eq!(result.ops_per_second(), 100.0);
        assert_eq!(result.bytes_per_second(), Some(1000.0));
        assert_eq!(result.chars_per_second(), Some(500.0));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_timing_result_zero() {
        let result = TimingResult::new("test", Duration::ZERO, 0);
        assert_eq!(result.avg_duration(), Duration::ZERO);
        assert_eq!(result.ops_per_second(), 0.0);
    }

    #[test]
    fn test_timer() {
        let mut timer = Timer::new();
        std::thread::sleep(Duration::from_millis(10));
        timer.lap("first");
        std::thread::sleep(Duration::from_millis(10));
        timer.lap("second");

        assert_eq!(timer.laps().len(), 2);
        assert!(timer.elapsed() >= Duration::from_millis(20));
    }

    #[test]
    fn test_timer_lap_durations() {
        let mut timer = Timer::new();
        timer
            .laps
            .push(("a".to_string(), Duration::from_millis(10)));
        timer
            .laps
            .push(("b".to_string(), Duration::from_millis(25)));

        let durations = timer.lap_durations();
        assert_eq!(durations[0].1, Duration::from_millis(10));
        assert_eq!(durations[1].1, Duration::from_millis(15));
    }

    #[test]
    fn test_benchmark() {
        let bench = Benchmark::new();
        let result = bench.run("counter", || {
            let mut x = 0;
            for i in 0..1000 {
                x += i;
            }
            x
        });

        assert!(result.iterations > 0);
        assert!(result.duration > Duration::ZERO);
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new();
        suite.add(TimingResult::new("fast", Duration::from_millis(10), 100));
        suite.add(TimingResult::new("slow", Duration::from_millis(100), 100));

        assert_eq!(suite.results().len(), 2);
        assert_eq!(suite.fastest().unwrap().name, "fast");
        assert_eq!(suite.slowest().unwrap().name, "slow");
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new();
        stats.record_alloc(1000);
        stats.record_alloc(500);
        stats.record_dealloc(200);

        assert_eq!(stats.peak_bytes, 1500);
        assert_eq!(stats.current_bytes, 1300);
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.deallocations, 1);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_throughput() {
        let throughput = Throughput::calculate(1000, 1024 * 1024, Duration::from_secs(1));

        assert_eq!(throughput.items_per_second, 1000.0);
        assert_eq!(throughput.mb_per_second(), 1.0);
    }

    #[test]
    fn test_parse_stats() {
        let mut stats = ParseStats::new();
        stats.record(100, 200, 20, Duration::from_millis(10));
        stats.record(50, 100, 10, Duration::from_millis(5));

        assert_eq!(stats.texts_parsed, 2);
        assert_eq!(stats.total_chars, 150);
        assert_eq!(stats.total_tokens, 30);
        assert_eq!(stats.min_time, Duration::from_millis(5));
        assert_eq!(stats.max_time, Duration::from_millis(10));
    }

    #[test]
    fn test_histogram_linear() {
        let mut hist = Histogram::linear(0.0, 100.0, 10);
        hist.record(5.0);
        hist.record(15.0);
        hist.record(95.0);

        assert_eq!(hist.count(), 3);
        assert!(hist.mean() > 0.0);
    }

    #[test]
    fn test_histogram_percentiles() {
        let mut hist = Histogram::linear(0.0, 100.0, 10);
        for i in 0..100 {
            hist.record(i as f64);
        }

        assert!(hist.p50() > 40.0 && hist.p50() < 60.0);
        assert!(hist.p95() > 90.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_histogram_empty() {
        let hist = Histogram::linear(0.0, 100.0, 10);
        assert_eq!(hist.count(), 0);
        assert_eq!(hist.mean(), 0.0);
        assert_eq!(hist.p50(), 0.0);
    }

    #[test]
    fn test_histogram_logarithmic() {
        let mut hist = Histogram::logarithmic(1.0, 1000.0, 3);
        hist.record(1.0);
        hist.record(10.0);
        hist.record(100.0);

        assert_eq!(hist.count(), 3);
    }

    #[test]
    fn test_benchmark_summary() {
        let mut suite = BenchmarkSuite::new();
        suite.add(TimingResult::new("a", Duration::from_secs(1), 100));
        suite.add(TimingResult::new("b", Duration::from_secs(2), 200));

        let summary = suite.summary();
        assert_eq!(summary.benchmark_count, 2);
        assert_eq!(summary.total_iterations, 300);
        assert_eq!(summary.total_duration, Duration::from_secs(3));
    }
}
