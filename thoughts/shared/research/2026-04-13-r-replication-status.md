---
date: 2026-04-13T10:30:00-07:00
researcher: Claude
git_commit: ebe68c7caa55efc7a70d6870213e92f241fadee5
branch: main
repository: idtaxa-optim
topic: "Does oxidtaxa still exactly replicate R DECIPHER's IDTAXA on gold standard tests?"
tags: [research, codebase, golden-tests, r-replication, idtaxa, decipher]
status: complete
last_updated: 2026-04-13
last_updated_by: Claude
---

# Research: Does oxidtaxa still exactly replicate R DECIPHER's IDTAXA on gold standard tests?

**Date**: 2026-04-13T10:30:00-07:00
**Researcher**: Claude
**Git Commit**: ebe68c7
**Branch**: main
**Repository**: idtaxa-optim

## Research Question

Does the oxidtaxa Rust/Python reimplementation still exactly replicate the IDTAXA R package from DECIPHER on the project's gold standard tests?

## Summary

**All 54 golden tests pass** (`cargo test --release` — 0 failures). The test suite validates oxidtaxa against R-generated golden data across 10 functional levels: FASTA I/O, k-mer enumeration, alphabet size, gap removal, reverse complement, pattern counting, integer matching, training, classification, and full pipeline integration.

However, **"exact replication" depends on what layer you're asking about**:

| Layer | Replication Status | Tolerance |
|---|---|---|
| PRNG (MT19937) | **Bit-identical** to R | 1e-15 (effectively exact) |
| K-mer enumeration | **Exact** | 0 (i32 equality) |
| Sequence utilities | **Exact** | 0 (string/int equality) |
| Integer matching | **Exact** | 0 (boolean equality) |
| Alphabet size | **Near-exact** | 1e-10 |
| Training (structure) | **Exact** | 0 (taxonomy, children, parents, taxa, levels, kmers) |
| Training (floats) | **Near-exact** | 1e-10 (IDF weights, fractions) |
| Classification (taxa) | **Exact** | 0 (string equality) |
| Classification (confidence) | **Approximate** | **5.0 percentage points** |
| E2E pipeline (Python) | **Approximate** | **10.0 percentage points** |

The low-level building blocks (PRNG, k-mers, matching) are bit-identical to R. Training structure is exact. **Classification confidence has a 5-percentage-point tolerance** — the widest gap in the Rust test suite. The Python test suite widens this further to 10 points.

Additionally, two algorithmic changes mean **production mode diverges from R by design**:

1. **Fraction learning is batch/order-independent** (R is sequential/order-dependent). Documented path agreement with R's sequential approach: 87–93% on benchmark datasets.
2. **Classification uses per-sequence PRNGs** in production mode (`deterministic=false`), not R's shared sequential PRNG. Results are statistically equivalent but not bit-identical.

## Detailed Findings

### Golden Test Infrastructure

The golden test pipeline has three stages:

1. **R generates `.rds` files** via `tests/generate_golden.R` using Bioconductor DECIPHER + `set.seed(42)`. Covers 10 functional sections (S01–S10) producing 90+ `.rds` files.
2. **`.rds` → `.json` conversion** via `tests/export_golden_json.R` using `jsonlite` with 17-digit float precision. Also generates PRNG verification data.
3. **Rust tests compare against JSON** via `tests/test_*.rs` using exact or approximate assertions.

### Test Results (2026-04-13, commit ebe68c7)

```
test_alphabet:     5/5  passed
test_classify:    13/13 passed
test_fasta:        1/1  passed
test_integration:  1/1  passed
test_kmer:         9/9  passed
test_matching:     7/7  passed
test_rng:          4/4  passed
test_sequence:     6/6  passed
test_ties:         3/3  passed  (Rust-only feature, no R golden)
test_training:     5/5  passed
TOTAL:            54/54 passed
```

### Classification Confidence Tolerance (5.0)

The classification tests (`tests/test_classify.rs:16`) use `CONF_TOLERANCE = 5.0` — a 5-percentage-point absolute tolerance on confidence values. This means a Rust confidence of 62.0 passes against an R golden value of 60.0. The taxonomy path (which species/genus was assigned) must match exactly.

This tolerance accounts for accumulated floating-point divergence through the bootstrap pipeline. The R-to-R self-test (`tests/run_golden.R`) uses a tighter tolerance of 0.01 (1 percentage point).

### Two Intentional Divergences from R

#### 1. Batch Fraction Learning (Training)

R's `LearnTaxa` updates fractions sequentially — each sequence immediately modifies shared state, making results order-dependent. Oxidtaxa snapshots fractions per iteration and processes all sequences in parallel with content-based PRNG seeding.

Documented impact (`thoughts/shared/plans/2026-04-06-batch-fraction-learning-results.md`):
- Taxonomy nodes: identical sets
- K, fractions, IDF weights: identical (0 diffs)
- Decision k-mers: 99.9% overlap
- Classification path agreement (sequential vs batch): 87–93%
- Mean confidence difference: 1.95–3.46
- Max confidence difference: 23.01–67.01

The golden tests still pass because they compare against a **deterministic sequential mode** (`deterministic=true`) that consumes the PRNG in R's order. Production mode uses the batch approach.

#### 2. Per-Sequence PRNG in Classification

R uses a single shared PRNG across all queries sequentially. Oxidtaxa's production mode (`deterministic=false`, the default) seeds each query's PRNG with `seed XOR index`, enabling parallel classification. The `deterministic=true` mode (used by golden tests) preserves R's shared PRNG behavior.

#### 3. Tied-Species / LCA Capping (New Feature)

Not a divergence per se — this is a new feature. R's `max.col(ties.method="random")` randomly picks one tied column. Oxidtaxa deterministically splits credit among tied groups and reports all alternatives. When ties exist, the lineage is capped at the LCA. The 3 `test_ties` tests validate this Rust-only feature.

### What the Golden Tests Do NOT Cover

- **Real biological data at scale**: The golden tests use 80 synthetic training sequences and 15 queries. The `benchmarks/baselines/baseline_1k.json` file contains R's classifications for 500 real vert12S queries against 1,000 references, but there is no automated test that compares Rust output against this baseline.
- **Bootstrap counts other than 100**: The golden generator produces data for bootstraps=1/10/50/100/200, but the Rust tests only check bootstraps=100 (`tests/test_classify.rs:336`).
- **Production mode (non-deterministic)**: All golden tests use `deterministic=true`. The per-sequence PRNG mode is not validated against R.
- **Batch training mode**: The golden training tests compare against R's sequential approach using the deterministic code path. The batch training divergence is documented in the results file but not enforced by golden tests.

## Code References

- `tests/test_classify.rs:16` — `CONF_TOLERANCE = 5.0`
- `tests/test_rng.rs:7-21` — PRNG verified to 1e-15
- `tests/test_training.rs:50-119` — Training comparison function with tolerance details
- `tests/generate_golden.R` — R golden data generation (set.seed(42), DECIPHER)
- `tests/export_golden_json.R` — JSON export with 17-digit precision
- `src/rng.rs` — R-compatible MT19937 implementation
- `src/training.rs:262-418` — Batch fraction learning (divergent from R)
- `src/classify.rs:502-542` — Sequential (R-compatible) classification path
- `src/classify.rs:546-588` — Parallel (production) classification path
- `src/classify.rs:443-465` — Tied-species LCA logic (not in R)
- `benchmarks/baselines/baseline_1k.json` — R baseline for real data (not auto-tested)

## Architecture Documentation

The validation architecture uses a three-layer golden-file pattern:
1. R/DECIPHER generates canonical outputs once (`.rds`)
2. A separate R script converts `.rds` → `.json` at 17-digit precision
3. Rust tests deserialize JSON and compare with layer-appropriate tolerances

A deterministic code path (`deterministic=true`) preserves R's exact PRNG consumption order for golden test fidelity. Production code paths diverge intentionally for parallelism.

## Historical Context (from thoughts/)

- `thoughts/shared/plans/2026-04-06-batch-fraction-learning-results.md` — Quantitative comparison of batch vs sequential training: 87–93% path agreement, identical model structure
- `thoughts/shared/plans/2026-04-08-tied-species-reporting.md` — Documents that existing 12 `s09*` golden tests were left unchanged by tied-species feature
- `thoughts/shared/plans/2026-04-03-idtaxa-python-rust-port.md` — Original port plan specifying tolerance hierarchy

## Related Research

- `thoughts/shared/research/2026-04-05-rust-idtaxa-parameter-space.md`
- `thoughts/shared/research/2026-04-08-tied-species-reporting-at-truncated-rank.md`

## Open Questions

1. **Should the 1K baseline be auto-tested?** `benchmarks/baselines/baseline_1k.json` contains R's classifications for 500 real queries but no test compares Rust against it.
2. **Is 5.0 confidence tolerance acceptable?** The R self-test uses 0.01; Rust uses 5.0. This is a 500x wider tolerance. The gap likely comes from accumulated float divergence in bootstrap sampling, but it hasn't been profiled to confirm.
3. **What is the actual max confidence divergence on the golden data?** The tests assert `< 5.0` but don't report the actual maximum divergence observed.
