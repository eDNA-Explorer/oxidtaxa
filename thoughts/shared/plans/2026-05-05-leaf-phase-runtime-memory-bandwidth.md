# Leaf-Phase Runtime and Memory-Bandwidth Optimization Plan

## Overview

Improve high-parallel classification runtime by reducing leaf-phase memory traffic in broad candidate sets. The current matcher materializes per-reference, per-bootstrap dense hit rows for every kept reference; with large databases and `processors` set high, the run can become memory-bandwidth-bound from allocation and zero-fill rather than CPU-bound on useful matching.

## Current State Analysis

Leaf scoring reaches `score_leaf_candidate_set()` from both greedy and beam descent. The scorer currently calls either `parallel_match_inverted()` or `parallel_match()` and receives a full `hits_flat` matrix plus `sum_hits`.

### Key Discoveries

- Greedy and beam both converge on `leaf_phase_score()`, so one scorer refactor covers both descent modes: `src/classify.rs:342` and `src/classify.rs:775`.
- `leaf_phase_score()` expands child winners into concrete training sequence indices in `keep`, samples query k-mers once, builds `u_sampling` / `positions` / `ranges`, and calls `score_leaf_candidate_set()`: `src/classify.rs:1806`, `src/classify.rs:1817`, `src/classify.rs:1909`.
- `score_leaf_candidate_set()` requests full per-ref rows from the matcher before it knows which refs are top-M per group: `src/classify.rs:985`.
- `parallel_match_inverted()` allocates `dense_map = vec![u32::MAX; max_idx + 1]` and `hits_flat = vec![0.0f64; keep.len() * block_count]`: `src/matching.rs:181`, `src/matching.rs:187`.
- The non-inverted fallback also allocates full `hits_flat`, and for `n > 32` allocates one per-sequence `Vec<f64>` in the inner Rayon map: `src/matching.rs:89`, `src/matching.rs:96`, `src/matching.rs:100`.
- Full `sum_hits` is required for top-ref / top-M selection and leaf telemetry: `src/classify.rs:1033`, `src/classify.rs:1061`, `src/classify.rs:2168`.
- Full per-replicate rows are not required for all kept refs. After top-M refs are selected per group, per-replicate scoring is required only for those selected refs to preserve bootstrap tie/share-split semantics: `src/classify.rs:1094`, `src/classify.rs:1101`.
- Prefix suppression is per-replicate and cannot be collapsed to row sums without changing behavior: `src/classify.rs:1131`.
- Default tie handling uses exact floating-point equality, so implementation must preserve addition order within any score path that should remain bit-identical: `src/classify.rs:1118`, `src/classify.rs:1181`.
- Existing tests cover `leaf_top_m`, prefix suppression, and deepest-rank diagnostics, but direct inverted-vs-merge matcher equivalence is missing. `tests/test_matching.rs` currently covers `int_match` only.

## Desired End State

Classification throughput improves under high `processors` on broad leaf-phase candidate sets because per-query memory traffic drops from `O(keep.len() * b)` per-replicate rows to roughly `O(keep.len())` sums plus `O(n_groups * leaf_top_m * b)` selected rows.

The implementation is considered successful when:

- Classification output is unchanged for existing deterministic tests.
- Inverted and non-inverted matching remain equivalent on focused fixtures.
- High-parallel benchmark runs show reduced peak RSS and improved wall time or throughput on large `keep` cases.
- If compact scoring ever regresses runtime on small or narrow candidate sets, the implementation keeps or adds an adaptive threshold so those cases remain on the existing path.

## What We're NOT Doing

- Not changing public classify defaults or API surface.
- Not changing classifier semantics, tie handling, prefix suppression, or deepest-rank diagnostics.
- Not changing `TrainingSet` serialization format.
- Not changing bootstrap sampling, RNG behavior, `sample_exponent`, or `bootstrap_coverage_cap`.
- Not solving unrelated descent or calibration issues from the decision-analysis plans.
- Not assuming memory reduction alone is a win; benchmark data must show runtime benefit under relevant parallelism.

## Implementation Approach

Treat this as a runtime optimization whose mechanism is memory-traffic reduction. First add reproducible performance instrumentation for broad leaf phases, then add correctness tests around matcher equivalence, then implement compact scoring.

The core design is two-stage leaf scoring:

1. Compute `sum_hits` for every kept ref without allocating `keep.len() * b` replicate rows.
2. Select each group's top ref or top-M refs by `sum_hits`, preserving current order and tie semantics.
3. Compute per-replicate rows only for the selected refs.
4. Run the existing per-replicate group accumulator over compact selected rows.
5. Return enough data for existing diagnostics: `lookup`, full `sum_hits`, selected group rows, `tot_hits`, winners, and margins.

This trades an additional matcher pass for a large reduction in allocation and zero-fill. The benchmark phase decides whether this should always run, only run when `keep.len() * b` exceeds a threshold, or be specialized further.

## Phase 1: Benchmark and Allocation Baseline

### Overview

Build a focused benchmark that reproduces the suspected slowdown: large `keep`, high `b`, and many concurrent query tasks. The benchmark should isolate leaf-phase matcher/scorer behavior enough to compare current vs compact implementations.

### Changes Required

#### 1. Leaf-Phase Stress Benchmark

**File**: `benches/oxidtaxa_bench.rs`

**Changes**:
- Add a benchmark case that trains or constructs a model with many references under a broad terminal candidate set.
- Run classification with settings that force broad leaf-phase scoring, including low/failed descent confidence or broad terminal-sibling challenge if needed.
- Include cases for:
  - `processors = 1`
  - moderate parallelism such as `processors = 8`
  - high parallelism available on the machine
  - `bootstrap_coverage_cap = false` to force `b = bootstraps` where useful
- Record wall time and throughput. Peak RSS can be collected manually with `/usr/bin/time -l` on macOS if Criterion does not capture memory.

#### 2. Optional Diagnostic Counters

**File**: `src/classify.rs`

**Changes**:
- Add an env-gated diagnostic such as `OXIDTAXA_LEAF_ALLOC_DIAG`.
- Emit aggregate counters, not per-query spam:
  - max `keep.len()`
  - max `b`
  - max estimated dense bytes `keep.len() * b * size_of::<f64>()`
  - number of leaf score calls above thresholds such as 10 MB, 100 MB, 500 MB
- Keep diagnostics off by default and avoid changing output.

### Success Criteria

#### Automated Verification

- [x] `cargo test` passes.
- [ ] `cargo bench --bench oxidtaxa_bench` runs and includes the new benchmark case.

#### Manual Verification

- [ ] Current implementation shows high estimated dense allocation in broad leaf-phase cases.
- [ ] At high `processors`, wall time or CPU utilization indicates memory-bandwidth/allocator pressure rather than clean CPU scaling.

## Phase 2: Matcher Correctness Coverage

### Overview

Add tests before refactoring so compact matching can be validated against the existing dense implementation.

### Changes Required

#### 1. Direct Matcher Equivalence Tests

**File**: `tests/test_matching.rs`

**Changes**:
- Add tests comparing `parallel_match()` and `parallel_match_inverted()` on a synthetic training set.
- Cover:
  - non-contiguous `keep`
  - multiple refs in the same group-like slice
  - repeated bootstrap positions
  - weighted hits
  - query k-mers outside the inverted-index range
  - empty `keep`
- Assert both `hits_flat` and `sum_hits` are equal.

#### 2. End-to-End Inverted Fallback Equivalence

**File**: `tests/test_classify.rs` or `tests/test_matching.rs`

**Changes**:
- Train a small model.
- Clone the `TrainingSet` and set `inverted_index = None`.
- Classify the same deterministic query/config/seed with both models.
- Assert taxon, confidence, similarity, alternatives, and relevant diagnostics match.

### Success Criteria

#### Automated Verification

- [x] `cargo test test_matching` passes.
- [x] End-to-end inverted-vs-merge classification equivalence test passes.
- [x] Existing `test_leaf_top_m`, `test_margin_aware`, and `test_deepest_rank_margin_rescue` suites still pass.

#### Manual Verification

- [x] Tests fail if the inverted path changes row order or sum semantics.

## Phase 3: Compact Leaf Scoring

### Overview

Refactor leaf scoring so large candidate sets no longer allocate per-replicate rows for every kept reference.

### Changes Required

#### 1. Add Sum-Only Matching Helpers

**File**: `src/matching.rs`

**Changes**:
- Add a sum-only variant for the inverted matcher, for example:

```rust
pub fn match_sums_inverted(
    query_kmers: &[i32],
    inverted_index: &[Vec<u32>],
    keep: &[usize],
    weights: &[f64],
    positions: &[usize],
    ranges: &[usize],
) -> Vec<f64>
```

- For each k-mer hit, add `weights[i] * (ranges[i + 1] - ranges[i]) as f64` to the row sum rather than updating every replicate position.
- Preserve current posting-list and `keep` position behavior.
- Add a merge-join fallback sum-only helper for `ts.inverted_index = None`.

#### 2. Add Selected-Row Matching Helpers

**File**: `src/matching.rs`

**Changes**:
- Add helpers that compute dense rows only for a selected ref set:

```rust
pub fn match_selected_rows_inverted(
    query_kmers: &[i32],
    inverted_index: &[Vec<u32>],
    keep: &[usize],
    selected_keep_positions: &[usize],
    weights: &[f64],
    block_count: usize,
    positions: &[usize],
    ranges: &[usize],
) -> Vec<f64>
```

- Output layout should be compact: `selected_row_idx * b + rep`.
- Preserve selected row order exactly as provided.
- The merge fallback can either call existing `parallel_match()` with selected sequence indices or have a specialized selected-row helper.

#### 3. Refactor LeafCandidateScores Shape

**File**: `src/classify.rs`

**Changes**:
- Replace `hits_flat: Vec<f64>` with compact selected rows and index maps, for example:

```rust
selected_ref_positions: Vec<usize>,
selected_hits_flat: Vec<f64>,
top_hits_selected_idx: Vec<usize>,
top_m_selected_idxs_per_group: Vec<Vec<usize>>,
```

- Keep `sum_hits: Vec<f64>` for all original `keep` refs.
- Keep `lookup`, `unique_groups`, `tot_hits`, and winner fields unchanged where possible.
- Preserve `top_hits_idx` only if downstream telemetry still benefits from original keep positions; otherwise replace with explicit selected mapping.

#### 4. Rewrite score_leaf_candidate_set Around Compact Rows

**File**: `src/classify.rs`

**Changes**:
- Pass 1:
  - Compute `sum_hits` only.
  - Apply `length_normalize` to `sum_hits` using the same normalization as current code.
  - Build `lookup`, `order`, `unique_groups`, top ref, and top-M refs by original keep position.
- Build `selected_ref_positions` as the union of all top-M refs for all groups, preserving current group/order semantics.
- Pass 2:
  - Compute selected compact rows.
  - Apply `length_normalize` to each selected row.
  - Run the existing per-replicate accumulator with compact row indices.
- Similarity:
  - For `leaf_top_m == 1`, use `sum_hits[top_ref] / davg` or selected compact row sum. Prefer `sum_hits` only if tests confirm bit identity after normalization.
  - For `leaf_top_m > 1`, compute from compact selected rows using `aggregate_top_m_at_rep()`.

#### 5. Adaptive Threshold

**File**: `src/classify.rs`

**Changes**:
- If Phase 1/3 benchmarks show two-pass overhead regresses small cases, add an internal threshold:

```rust
let dense_bytes = keep_seq_indices.len() * context.b * std::mem::size_of::<f64>();
let use_compact = dense_bytes >= COMPACT_LEAF_MIN_BYTES;
```

- Keep this internal and undocumented unless it becomes a public tuning need.
- Start with a conservative threshold only if benchmark data justifies it. Otherwise use compact scoring for all leaf scoring to keep one code path.

### Success Criteria

#### Automated Verification

- [x] `cargo test` passes.
- [x] `cargo test test_leaf_top_m` passes.
- [x] `cargo test test_margin_aware` passes.
- [x] `cargo test test_deepest_rank_margin_rescue` passes.
- [x] `cargo test test_classify` passes.
- [x] Existing deterministic classification golden tests remain unchanged within their current tolerances.

#### Manual Verification

- [x] On a broad candidate benchmark, estimated dense allocation is removed or reduced to compact selected rows.
- [ ] High-parallel classification wall time improves or does not regress relative to current implementation.
- [ ] Classification behavior remains unchanged on representative real marker runs.

## Phase 4: Dense Map Allocation Cleanup

### Overview

After shrinking `hits_flat`, address the secondary allocation: `dense_map = vec![u32::MAX; max_idx + 1]`.

### Changes Required

#### 1. Stamped Scratch Lookup or Adaptive Lookup

**File**: `src/matching.rs`

**Changes**:
- Evaluate one of:
  - Thread-local reusable dense map with stamps.
  - Caller-owned scratch passed through leaf scoring.
  - Adaptive map: dense vector for low `max_idx`, `HashMap` or sorted lookup when `max_idx` is large relative to `keep.len()`.
- Avoid a per-call full zero/fill of `max_idx + 1` for broad indexed sets when it is measurable in benchmarks.
- Keep implementation simple unless Phase 1/3 data shows this remains hot after compact scoring.

### Success Criteria

#### Automated Verification

- [ ] Direct matcher equivalence tests pass.
- [ ] `cargo test` passes.

#### Manual Verification

- [ ] Benchmark shows additional runtime improvement or neutral behavior.
- [ ] No increase in peak RSS from persistent scratch under high thread counts beyond acceptable bounds.

## Phase 5: Performance Validation and Rollout Decision

### Overview

Decide whether compact scoring should be the only path or an adaptive path based on measured runtime.

### Changes Required

#### 1. Benchmark Matrix

**Commands**:

```bash
cargo bench --bench oxidtaxa_bench
```

Run targeted classification workloads at:
- `processors = 1`
- moderate parallelism
- high parallelism used in the failing workload

Record:
- wall time
- throughput queries/sec
- peak RSS
- max estimated dense bytes per leaf call
- CPU utilization if available

#### 2. Real-Workload Check

Run the colleague's or benchmarking pipeline's representative high-parallel workload with the patched local OxidTaxa build.

### Success Criteria

#### Automated Verification

- [ ] `cargo test` passes.
- [ ] `cargo bench --bench oxidtaxa_bench` completes.

#### Manual Verification

- [ ] High-parallel workload is faster than baseline.
- [ ] Peak RSS is lower than baseline.
- [ ] No correctness differences appear in sampled output comparisons.
- [ ] If compact scoring regresses narrow cases, adaptive threshold is set and documented in source comments.

## Testing Strategy

### Unit Tests

- Direct matcher equivalence: `parallel_match()` vs `parallel_match_inverted()`.
- Sum-only helper equivalence to dense row sums.
- Selected-row helper equivalence to dense rows for selected refs.
- Length-normalized sum/row equivalence.
- Empty `keep`, single selected ref, repeated positions, invalid k-mers.

### Integration Tests

- Inverted-index model vs cloned model with `inverted_index = None`.
- `leaf_top_m` M=1 identity and M>1 aggregation.
- `suppress_ancestor_only_groups` with `leaf_top_m > 1`.
- Deepest-rank diagnostics and challenge scoring unchanged.
- Beam and greedy paths still converge through the same compact leaf scorer.

### Manual Testing Steps

1. Run `cargo test`.
2. Run `cargo test test_leaf_top_m`.
3. Run `cargo test test_margin_aware`.
4. Run `cargo test test_deepest_rank_margin_rescue`.
5. Run `cargo bench --bench oxidtaxa_bench`.
6. Run one real high-parallel classification workload with the same model/query/config as the observed slowdown.
7. Compare wall time, peak RSS, and sampled output rows before and after.

## Performance Considerations

- Dense current memory per leaf call is `keep.len() * b * 8` bytes for `hits_flat`, plus `max_idx + 1` `u32`s for `dense_map`.
- With `keep.len() = 100_000` and `b = 100`, dense `hits_flat` is about 80 MB per query.
- With `keep.len() = 876_000` and `b = 100`, dense `hits_flat` is about 700 MB per query.
- At high `processors`, concurrent zero-fill and row writes can dominate runtime through memory bandwidth and allocator pressure.
- Compact scoring reduces write volume but may increase matcher traversal. Benchmark data must decide whether to use compact scoring universally or only above a threshold.
- Exact tie behavior is sensitive to floating-point order; preserve current per-ref per-replicate addition order in any path that computes selected rows.

## Migration Notes

- No public API change.
- No model serialization change.
- No expected output change.
- Benchmarks and tests should land with the implementation so future scorer changes do not reintroduce dense all-ref row allocation.

## References

- Dense inverted matcher allocation: `src/matching.rs:181`, `src/matching.rs:187`
- Dense merge matcher allocation: `src/matching.rs:89`
- Leaf scorer matcher dispatch: `src/classify.rs:985`
- Top-M selection by `sum_hits`: `src/classify.rs:1033`, `src/classify.rs:1061`
- Per-replicate leaf accumulator: `src/classify.rs:1094`
- Prefix suppression inside bootstrap loop: `src/classify.rs:1131`
- Similarity currently reads dense rows: `src/classify.rs:1930`
- Leaf telemetry uses full `sum_hits`: `src/classify.rs:2168`
- Existing top-M tests: `tests/test_leaf_top_m.rs`
- Existing prefix suppression tests: `tests/test_margin_aware.rs`
- Existing deepest-rank diagnostics tests: `tests/test_deepest_rank_margin_rescue.rs`
- Historical prefix share-split plan: `thoughts/shared/plans/2026-04-30-prefix-aware-bootstrap-share-split.md`
- Historical computational optimization plan: `thoughts/shared/plans/2026-04-13-computational-optimizations.md`
