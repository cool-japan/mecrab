# MeCrab Fuzzing Guide

Quick reference for running fuzz tests.

---

## Setup

```bash
# Install cargo-fuzz (requires nightly)
cargo install cargo-fuzz

# List available targets
cargo +nightly fuzz list
```

---

## Available Fuzz Targets

| Target | Description | Command |
|--------|-------------|---------|
| `viterbi` | Tests Viterbi algorithm | `cargo +nightly fuzz run viterbi` |
| `dictionary` | Tests dictionary loading | `cargo +nightly fuzz run dictionary` |
| `parser` | Tests full parsing pipeline | `cargo +nightly fuzz run parser` |
| `lattice` | Tests lattice construction | `cargo +nightly fuzz run lattice` |

---

## Running with Timeout

```bash
# Run for 60 seconds
timeout 60s cargo +nightly fuzz run viterbi

# Run for 10 minutes
timeout 10m cargo +nightly fuzz run parser

# Run indefinitely (Ctrl+C to stop)
cargo +nightly fuzz run viterbi
```

---

## Corpus Management

```bash
# Minimize corpus (remove redundant inputs)
cargo +nightly fuzz cmin viterbi

# Show coverage report
cargo +nightly fuzz coverage viterbi

# View corpus directory
ls fuzz/corpus/viterbi/
```

---

## Reproducing Crashes

```bash
# Run specific crash input
cargo +nightly fuzz run viterbi fuzz/artifacts/viterbi/crash-xxx

# Debug with backtrace
RUST_BACKTRACE=1 cargo +nightly fuzz run viterbi fuzz/artifacts/viterbi/crash-xxx
```

---

## CI Integration

```yaml
name: Fuzz Tests

on: [push, pull_request]

jobs:
  fuzz:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - name: Install cargo-fuzz
        run: cargo install cargo-fuzz
      - name: Run fuzz tests (5 minutes each)
        run: |
          timeout 300s cargo +nightly fuzz run parser || true
          timeout 300s cargo +nightly fuzz run viterbi || true
```

---

## Writing New Fuzz Targets

```bash
# Create new target
cargo +nightly fuzz add new_target
```

Example target:

```rust
#![no_main]

use libfuzzer_sys::fuzz_target;
use mecrab::MeCrab;

fuzz_target!(|data: &str| {
    if data.len() > 5000 { return; }

    if let Ok(mecrab) = MeCrab::new() {
        let _ = mecrab.parse(data);
    }
});
```

---

*Copyright 2026 COOLJAPAN OU (Team KitaSan)*
