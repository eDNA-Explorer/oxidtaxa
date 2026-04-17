# Correlation-Aware Training Speedup: Phase-by-Phase Timings

Benchmark: `learn_taxa_scaling_corr_aware/train/{1000,5000,10000}refs`
Config: `correlation_aware_features: true, record_kmers_fraction: 0.44, processors: 8`
Criterion sample_size: 10

Record the median time per iteration after each phase. 10k runs only at Phase 0 baseline (if patient), after Phase 1, and after Phase 4.

## Phase 0 (deterministic order only — same algorithm as pre-v2)

| Dataset | Median | Notes |
|---------|--------|-------|
| 80 seq  | 1.53 s | committed criterion baseline |
| 1k      | DNF — killed after 17+ CPU-min, warmup not complete | projected 60-120 s/iter |
| 5k      | not attempted | too slow |
| 10k     | skipped (per user constraint) | |

## Phase 1 (incremental max_corr caching)

| Dataset | Median | Δ vs Phase 0 |
|---------|--------|--------------|
| 80 seq  | 859 ms | -43.9% (1.78×) |
| 1k      | 2.53 s | ≥ 25-50× (baseline DNF) |
| 5k      | not run | defer |

## Phase 2 (HashMap max_h + inverted profile build) — SKIPPED

Attempted three variants, all regressed at 1k:
- HashMap max_h + HashMap-based profile build: 2.97 s (+17% vs Phase 1)
- HashMap max_h + merge-join profile build: 2.88 s (+14% vs Phase 1)
- Nested scan max_h + merge-join profile build: 3.93 s (+55% vs Phase 1)

Root cause: `HashMap<usize, _>` with `std::RandomState` does SipHash per lookup
(~50-100 ns). At typical candidate pool sizes (hundreds to thousands) the hash
overhead beats the original binary-search approach. Split into two passes also
hurt cache locality vs original one-pass interleaved construction.

Reverted to Phase 1 implementation. Moving directly to Phase 3.

## Phase 3 (parallel candidate scan/update, PAR_THRESHOLD=2048)

Clean measurements on cool machine, Optuna killed, processors=8.

| Dataset | Phase 1 only (threshold=∞) | Phase 3 | Speedup |
|---------|----------------------------|---------|---------|
| 80 seq  | 895 ms (no threshold trigger — same as Phase 3) | 895 ms | 1.00× |
| 1k      | 2.61 s (no threshold trigger) | 2.61 s | 1.00× |
| 5k      | 13.78 s | 10.22 s | **1.35×** |

Phase 3 engages only at nodes with ≥2048 candidates. 1k data stays below
this; 5k has enough large nodes to hit the threshold and benefit.
The ~1.35× is lower than theoretical 8-way parallelism because (a) only
large nodes trigger; (b) outer tree-parallelism competes for the pool.
Expected to scale better at 10k+ where more nodes cross the threshold.

## Phase 4 (inlined SIMD sum_ab) — REVERTED

Attempted inlining `pearson_with_stats` into both parallel and sequential
update loops. Results:

| Dataset | Phase 3 | Phase 4 | Δ |
|---------|---------|---------|---|
| 1k      | 2.61 s  | 2.68 s  | +2.7% (n.s., p=0.10) |
| 5k      | 10.22 s | 10.26 s | +0.4% (n.s.) |

Root cause: `pearson_with_stats` already has `#[inline]`, so LLVM was
already inlining it at call sites. Explicit hand-inlining didn't expose
anything new to the optimizer — just duplicated the Pearson math across
two code paths. Reverted to keep the code clean; no performance cost.

## Final summary

| Phase | 80-seq | 1k | 5k | Output |
|-------|--------|----|----|--------|
| pre-Phase-1 | 1.53 s | DNF (60+ s/iter projected) | — | — |
| Phase 1 | 895 ms (1.7×) | 2.61 s (≥25×) | 13.78 s | bit-exact |
| Phase 1+3 (shipped) | 895 ms | 2.61 s | 10.22 s (1.35× on top) | bit-exact |
