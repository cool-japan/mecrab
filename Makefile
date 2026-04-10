.PHONY: test clippy coverage bench clean fmt \
        bench-save bench-compare coverage-check \
        build-simd build-lsp check-wasm test-simd \
        clippy-simd clippy-lsp clippy-all

# ── Standard development commands ──────────────────────────────

## Run all tests (requires cargo-nextest)
test:
	cargo nextest run --all-features

## Run Clippy and treat all warnings as errors
clippy:
	cargo clippy --all-features --all-targets -- -D warnings

## Format all workspace crates
fmt:
	cargo fmt --all

## Generate LCOV coverage report and print summary
coverage:
	cargo llvm-cov nextest --all-features --lcov --output-path lcov.info
	cargo llvm-cov report --all-features --summary-only

## Run benchmarks (mecrab-bench subcrate)
bench:
	cargo bench --package mecrab-bench

## Remove build artefacts
clean:
	cargo clean

# ── Benchmark baseline helpers ──────────────────────────────────

## Run benchmarks and save results as the "main" baseline
bench-save:
	cargo bench --package mecrab-bench -- --save-baseline main

## Run benchmarks and compare against the saved "main" baseline
bench-compare:
	cargo bench --package mecrab-bench -- --baseline main

# ── Feature-gated build targets ────────────────────────────────

## Build mecrab with SIMD feature enabled
build-simd:
	cargo build -p mecrab --features simd

## Build kizame with LSP feature enabled (build only — LSP requires runtime tokio)
build-lsp:
	cargo build -p kizame --features lsp

## Check mecrab for the wasm32-unknown-unknown target with wasm feature
check-wasm:
	cargo check -p mecrab --target wasm32-unknown-unknown --features wasm

## Run nextest for mecrab with SIMD feature enabled
test-simd:
	cargo nextest run -p mecrab --features simd

## Run Clippy on mecrab with SIMD feature
clippy-simd:
	cargo clippy -p mecrab --features simd -- -D warnings

## Run Clippy on kizame with LSP feature
clippy-lsp:
	cargo clippy -p kizame --features lsp -- -D warnings

## Run all Clippy checks (default + simd + lsp)
clippy-all: clippy clippy-simd clippy-lsp
	@echo "All clippy checks passed"

# ── Coverage threshold enforcement ─────────────────────────────

## Fail if total line coverage is below 60 %
coverage-check:
	@cargo llvm-cov nextest --all-features --summary-only 2>&1 | \
		awk '/TOTAL/ { cov=$$NF+0; if (cov < 60) { printf "Coverage %.1f%% is below the 60%% threshold!\n", cov; exit 1 } else { printf "Coverage OK: %.1f%%\n", cov } }'
