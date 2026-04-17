# Correlation-Aware Training Speed Optimization Plan (v2)

## Overview

The v1 plan (`2026-04-15-correlation-aware-training-speedup.md`) landed four compounding optimizations (`ProfileStats`, `target-cpu=native`, flat-matrix layout, parallel sibling subtrees — commit `aba091c`). Despite those wins, `correlation_aware_features=true` is still "extremely slow" in the user's workflow. This plan targets the remaining dominant cost: the greedy forward-selection loop at `src/training.rs:930-974` recomputes every candidate-vs-selected Pearson correlation from scratch on every outer iteration, so total work grows as O(R² × C × D) instead of O(R × C × D).

Four compounding phases land on top of the v1 work:

1. **Validation infrastructure** — make the correlation-aware path deterministic (`HashSet` → sorted `Vec`), add a scaling benchmark for 1k / 5k / 10k, add an equivalence test that locks `decision_kmers` via bincode hash.
2. **Incremental `max_corr` caching** — maintain a per-candidate running-max correlation vector across outer iterations; update only against the newly-selected feature. Big win.
3. **Pool-construction cleanups** — one-pass HashMap for `max_h` lookups; inverted-pass merge-join for candidate profile matrix construction.
4. **Parallel candidate scan/update** — `par_iter` the argmax and the `max_corr` update with deterministic reductions.
5. **Batched SIMD `sum_ab`** — fold the per-iteration update into one tight auto-vectorizable loop over `sorted_profiles_flat`.

After each phase, validation runs on the **1k and 5k** reference datasets (10k only at milestones, per user constraint).

## Current State Analysis

The v1 plan's four phases are all implemented in `src/training.rs` as of commit `0a684f5`:

- `ProfileStats` + `pearson_with_stats` (lines 691-722)
- `.cargo/config.toml` with `target-cpu=native`
- Flat contiguous candidate profile matrix `sorted_profiles_flat` (lines 876-914)
- `rayon::join` / `rayon::scope` over sibling subtrees in `create_tree` (lines 763-803)

The research document `2026-04-17-correlation-aware-training-bottlenecks.md` identifies the remaining bottlenecks, with quantitative estimates of speedup per additional optimization.

### Key Discoveries

- Greedy loop at `src/training.rs:930-974` recomputes all `C × t` correlations at outer iteration `t`; work is O(R² × C × D). At `k=6, record_kmers_fraction=0.44`, R can reach ~1,800 making redundant work factor ~R/2 ≈ 900× (`src/training.rs:930-974`).
- `max_h` lookup at `src/training.rs:890-895` is a nested linear scan per candidate per child; O(C × D × max_nonzero) total.
- Per-candidate profile vector build at `src/training.rs:882-889` does D binary searches per candidate; O(C × D × log max|p|).
- `cand_set: HashSet<usize>` (line 869) has randomized iteration order across runs, so tie-breaking in feature selection is currently non-deterministic. The final `keep_vec.sort_unstable()` (line 1006) sorts selected k-mers but does not resolve the selection non-determinism itself.
- `TrainConfig::default()` sets `correlation_aware_features: false`, so `bench_learn_taxa_scaling` (`benches/oxidtaxa_bench.rs:378-403`) does not exercise the correlation-aware path. Scaling coverage exists only for 80-sequence via `bench_learn_taxa_correlation_aware` (line 243).
- The existing test at `tests/test_algorithmic_improvements.rs:284-306` only asserts that correlation-aware produces *different* output from default — no equivalence lock on the correlation-aware output itself.
- Criterion requires `sample_size >= 10` per benchmark; cannot go lower.

## Desired End State

After all phases:

1. The correlation-aware feature selection produces deterministic output across runs given the same inputs.
2. A Criterion benchmark `learn_taxa_scaling_corr_aware` covers 1k / 5k / 10k reference datasets.
3. An integration test locks `decision_kmers` output via bincode hash so any semantic drift is caught.
4. `max_corr` is maintained incrementally: each outer iteration does O(C × D) work, not O(t × C × D).
5. Candidate pool construction is single-pass per hot path (no nested binary searches, no linear scans for `max_h`).
6. Both argmax and `max_corr` update use rayon work-stealing within expensive nodes.
7. The inner candidate-update loop is a tight, contiguous, SIMD-friendly pass over `sorted_profiles_flat`.

### Verification

- All 112+ existing golden tests pass unchanged (`cargo test`).
- New `test_correlation_aware_deterministic_output` passes at each phase.
- `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware` at 1k and 5k shows monotonic phase-over-phase improvement.
- 10k bench is captured once at end of Phase 1 and once at end of Phase 4 (not every phase).

## What We're NOT Doing

- Lazy-greedy / CELF heap ordering (research idea #8) — deferred, add only if post-Phase-4 performance still insufficient.
- `f32` profile representation (#9) — would require a broader refactor of `DecisionNode.profiles`.
- Profile-guided optimization setup (#10) — unrelated to the algorithm; can be explored separately.
- Candidate pool cap via entropy threshold (#7) — changes output; out of scope for "keep same result" constraint.
- Any changes to `prepare_data()` or `learn_fractions()` phases — this plan is scoped to `build_tree` / `create_tree`.
- Any changes to the round-robin (non-correlation-aware) selection path (`src/training.rs:976-1002`).
- Changes to the Python API surface.

## Implementation Approach

Each phase is self-contained: a single commit, a passing test suite, a captured benchmark result. Phase 0 makes the rest measurable. Phase 1 is the largest single win — if it alone gets performance into "tolerable" territory the remaining phases are optional polish.

---

## Phase 0: Validation Infrastructure

### Overview

Three setup tasks that make every subsequent phase measurable and verifiable:
1. Make the correlation-aware path deterministic (eliminate `HashSet` iteration-order non-determinism).
2. Add scaling benchmark for 1k / 5k / 10k under correlation-aware.
3. Add equivalence test that locks bincode-hash of `decision_kmers`.
4. Capture baseline numbers.

### Changes Required

#### 1. Deterministic candidate iteration

**File**: `src/training.rs`
**Changes**: Collect `cand_set` into a sorted `Vec<usize>` before iterating to build the profile matrix. This replaces the current non-deterministic `HashSet` iteration.

Current (line 869-899):
```rust
let mut cand_set: HashSet<usize> = HashSet::new();
for child_h in &sorted_h {
    for &(kmer_idx, _) in child_h.iter().take(per_child_limit) {
        cand_set.insert(kmer_idx);
    }
}

let mut kmer_indices = Vec::with_capacity(cand_set.len());
let mut entropies = Vec::with_capacity(cand_set.len());
let mut profiles_flat = Vec::with_capacity(cand_set.len() * n_children);

for &kmer_idx in &cand_set {
    // ... build prof_vec, max_h ...
}
```

New:
```rust
let mut cand_set: HashSet<usize> = HashSet::new();
for child_h in &sorted_h {
    for &(kmer_idx, _) in child_h.iter().take(per_child_limit) {
        cand_set.insert(kmer_idx);
    }
}
let mut cand_sorted: Vec<usize> = cand_set.into_iter().collect();
cand_sorted.sort_unstable();

let mut kmer_indices = Vec::with_capacity(cand_sorted.len());
let mut entropies = Vec::with_capacity(cand_sorted.len());
let mut profiles_flat = Vec::with_capacity(cand_sorted.len() * n_children);

for &kmer_idx in &cand_sorted {
    // ... build prof_vec, max_h ...
}
```

The only semantic change: deterministic tie-breaking in feature selection (lowest k-mer index wins on exact entropy+gain tie, rather than whichever the HashSet happened to yield first). Floats differing by ULPs won't tie exactly, so this affects only rare cases.

#### 2. Correlation-aware scaling benchmark

**File**: `benches/oxidtaxa_bench.rs`
**Changes**: Add a new benchmark group mirroring `bench_learn_taxa_scaling` but with `correlation_aware_features: true, record_kmers_fraction: 0.44`.

Insert after `bench_learn_taxa_scaling` (line 403):
```rust
fn bench_learn_taxa_scaling_corr_aware(c: &mut Criterion) {
    let mut group = c.benchmark_group("learn_taxa_scaling_corr_aware");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    for n in &[1000, 5000, 10000] {
        let fasta = bench_data_path(&format!("bench_{}_ref.fasta", n));
        let tax_path = bench_data_path(&format!("bench_{}_ref_taxonomy.tsv", n));
        if !std::path::Path::new(&fasta).exists() { continue; }

        let (names, seqs) = read_fasta(&fasta).unwrap();
        let taxonomy = oxidtaxa::fasta::read_taxonomy(&tax_path, &names).unwrap();
        let (fseqs, ftax) = filter_for_bench(&seqs, &taxonomy);
        let config = TrainConfig {
            correlation_aware_features: true,
            record_kmers_fraction: 0.44,
            processors: 4,
            ..TrainConfig::default()
        };

        group.bench_with_input(
            BenchmarkId::new("train", format!("{}refs", n)),
            &(&fseqs, &ftax),
            |b, (seqs, tax)| {
                b.iter(|| {
                    black_box(learn_taxa(black_box(seqs), black_box(tax), &config, 42, false).unwrap());
                });
            },
        );
    }
    group.finish();
}
```

Register in `criterion_group!` at line 442-459 (add `bench_learn_taxa_scaling_corr_aware`).

#### 3. Equivalence test (bincode hash lock)

**File**: `tests/test_algorithmic_improvements.rs`
**Changes**: Add a new test that trains with correlation-aware, bincode-serializes `decision_kmers`, computes a SHA-256-style hash via `DefaultHasher`, and compares to a captured constant. Update the constant once per phase after manual verification.

Insert after `test_correlation_aware_changes_decision_kmers` (line 306):
```rust
#[test]
fn test_correlation_aware_deterministic_output() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let (seqs, tax) = load_standard_data();
    let config = TrainConfig {
        correlation_aware_features: true,
        record_kmers_fraction: 0.44,
        ..TrainConfig::default()
    };

    // Train twice and assert bit-exact equality across runs (determinism).
    let m1 = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();
    let m2 = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();

    let bytes1 = bincode::serialize(&m1.decision_kmers).unwrap();
    let bytes2 = bincode::serialize(&m2.decision_kmers).unwrap();
    assert_eq!(bytes1, bytes2, "decision_kmers must be bit-exact across runs");

    // Lock the expected hash. Update this constant ONLY after manual
    // verification that the new output is intentional (e.g., after Phase 0
    // when we change tie-breaking). Record in the commit message.
    let mut hasher = DefaultHasher::new();
    bytes1.hash(&mut hasher);
    let actual = hasher.finish();

    // TODO(Phase 0): paste hash from first green run here.
    const EXPECTED_HASH: u64 = 0x_DEADBEEF_DEADBEEF;
    assert_eq!(actual, EXPECTED_HASH,
        "decision_kmers hash changed unexpectedly. If intentional, update constant.");
}
```

After the first green run on Phase 0, paste the observed hash into `EXPECTED_HASH`. Intended behavior for subsequent phases: this constant does not change. If a phase intentionally alters output (e.g., a floating-point reduction order shift causes tie-breaking at the ULP level), document and update the constant in that phase's commit.

#### 4. Baseline capture

**Manual step**: On post-merge of Phase 0 change, run:
```
cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/1000refs
cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/5000refs
```
Record Criterion's median time per iteration in a file `thoughts/shared/plans/2026-04-17-baseline-timings.md`. Run 10k once if patient (may take hours); otherwise skip.

### Success Criteria

#### Automated Verification
- [ ] `cargo test` passes (all golden tests + new equivalence test).
- [ ] `cargo test test_correlation_aware_deterministic_output` passes with the locked hash.
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/1000refs` runs successfully.
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/5000refs` runs successfully.

#### Manual Verification
- [ ] Baseline timings for 1k and 5k recorded in plan-local timings file.
- [ ] 10k bench attempted; result recorded if completes, marked "DNF > X min" otherwise.
- [ ] Run the equivalence test 3 times across different shell sessions — hash should be stable.

---

## Phase 1: Incremental `max_corr` Caching

### Overview

Replace the recomputation pattern in the greedy loop with an incrementally-maintained `Vec<f64>` of per-candidate running-max correlations. Each outer iteration: pick argmax using cached `max_corr`, then update each not-selected candidate's `max_corr` against ONLY the single newly-selected feature. Reduces correlation-call count from O(R² × C) to O(R × C).

### Changes Required

#### 1. Rewrite the greedy loop

**File**: `src/training.rs`
**Changes**: Replace lines 921-975 (the greedy selection loop) with the incremental-caching version. The pre-loop setup (lines 921-928) is retained and extended with one new allocation: `max_corr: Vec<f64>`.

Current (lines 921-974):
```rust
let n_cand = sorted_kmer_indices.len();
let mut is_selected = vec![false; n_cand];
let mut result_set = HashSet::new();

let mut sel_profiles_flat: Vec<f64> = Vec::with_capacity(record_kmers * n_children);
let mut sel_stats: Vec<ProfileStats> = Vec::with_capacity(record_kmers);
let mut n_selected: usize = 0;

for _ in 0..record_kmers {
    let mut best_ci = None;
    let mut best_gain = f64::NEG_INFINITY;

    for ci in 0..n_cand {
        if is_selected[ci] { continue; }
        let base_h = sorted_entropies[ci];
        if base_h <= best_gain { break; }

        let cand_prof = &sorted_profiles_flat[ci * n_children..(ci + 1) * n_children];
        let cand_st = &cand_stats[ci];

        let mut max_corr: f64 = 0.0;
        for si in 0..n_selected {
            let sel_prof = &sel_profiles_flat[si * n_children..(si + 1) * n_children];
            let sel_st = &sel_stats[si];
            let corr = pearson_with_stats(cand_prof, cand_st, sel_prof, sel_st);
            if corr > max_corr { max_corr = corr; }
            if max_corr >= 1.0 { break; }
        }

        let gain = base_h * (1.0 - max_corr);
        if gain > best_gain {
            best_gain = gain;
            best_ci = Some(ci);
        }
    }

    match best_ci {
        Some(ci) => {
            is_selected[ci] = true;
            result_set.insert(sorted_kmer_indices[ci]);
            let src = ci * n_children;
            sel_profiles_flat.extend_from_slice(
                &sorted_profiles_flat[src..src + n_children]
            );
            sel_stats.push(cand_stats[ci].clone());
            n_selected += 1;
        }
        None => break,
    }
}
result_set
```

New:
```rust
let n_cand = sorted_kmer_indices.len();
let mut is_selected = vec![false; n_cand];
let mut result_set = HashSet::new();

// Per-candidate running-max correlation against all selected features so far.
// Updated incrementally: each outer iteration adds exactly one correlation
// computation per not-selected candidate (against the newly-selected one).
let mut max_corr: Vec<f64> = vec![0.0; n_cand];

// Selected buffers kept for API compatibility with downstream profile
// materialization, even though the greedy loop itself no longer reads them.
let mut sel_profiles_flat: Vec<f64> = Vec::with_capacity(record_kmers * n_children);
let mut sel_stats: Vec<ProfileStats> = Vec::with_capacity(record_kmers);

for _ in 0..record_kmers {
    // 1. Argmax over not-selected candidates using cached max_corr.
    //    Early exit remains valid: candidates are sorted by entropy descending,
    //    gain <= base_h, so if base_h <= best_gain we cannot beat it.
    let mut best_ci = None;
    let mut best_gain = f64::NEG_INFINITY;
    for ci in 0..n_cand {
        if is_selected[ci] { continue; }
        let base_h = sorted_entropies[ci];
        if base_h <= best_gain { break; }
        let gain = base_h * (1.0 - max_corr[ci]);
        if gain > best_gain {
            best_gain = gain;
            best_ci = Some(ci);
        }
    }

    let ci = match best_ci { Some(ci) => ci, None => break };

    // 2. Commit selection.
    is_selected[ci] = true;
    result_set.insert(sorted_kmer_indices[ci]);
    let src = ci * n_children;
    let new_prof_slice = &sorted_profiles_flat[src..src + n_children];
    let new_st = cand_stats[ci].clone();

    // 3. Incremental max_corr update against the newly-selected feature.
    //    Skip candidates that are already selected or already saturated.
    for cj in 0..n_cand {
        if is_selected[cj] { continue; }
        if max_corr[cj] >= 1.0 { continue; }
        let cj_prof = &sorted_profiles_flat[cj * n_children..(cj + 1) * n_children];
        let corr = pearson_with_stats(cj_prof, &cand_stats[cj], new_prof_slice, &new_st);
        if corr > max_corr[cj] { max_corr[cj] = corr; }
    }

    sel_profiles_flat.extend_from_slice(new_prof_slice);
    sel_stats.push(new_st);
}
result_set
```

**Correctness argument**: At outer iteration `t` with selected set `S_t`, both old and new code compute `max_corr[ci] = max(corr(ci, s) for s in S_t)` for each not-selected `ci`. Old code recomputes from scratch; new code maintains incrementally. The `max_corr[cj] >= 1.0` skip is safe because once saturated, max_corr cannot change (it's a running max). Argmax tie-breaking is unchanged (strict `gain > best_gain`, iterate `ci` in 0..n_cand order).

**Floating-point determinism**: the correlation formula per `(cand, sel)` pair is called on the same slices via the same `pearson_with_stats`, so the `sum_ab` reduction is identical to the old code. `max_corr[ci]` ends up as the max over the same set of identical values. Output is bit-exact (expected).

#### 2. Validate equivalence hash unchanged

**File**: `tests/test_algorithmic_improvements.rs`
**Changes**: After Phase 1 compiles, re-run the equivalence test. It must pass with the SAME `EXPECTED_HASH` captured in Phase 0. If it doesn't, the incremental caching has introduced unintended output drift and must be investigated.

### Success Criteria

#### Automated Verification
- [ ] `cargo test` passes (all golden tests + equivalence test with unchanged Phase 0 hash).
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/1000refs` completes.
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/5000refs` completes.
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa/80seqs_corr_aware` completes (no regression on small bench).

#### Manual Verification
- [ ] 1k bench median drops by >10× vs Phase 0 baseline.
- [ ] 5k bench median drops by >30× vs Phase 0 baseline.
- [ ] `cargo bench -- learn_taxa/80seqs` (non-correlation-aware default) shows no regression (±5%).
- [ ] Milestone 10k run captured once at this phase and recorded in timings file.
- [ ] `EXPECTED_HASH` in equivalence test is unchanged — same bit-exact output as Phase 0.

---

## Phase 2: Pool-Construction Cleanups

### Overview

Two orthogonal cleanups in candidate-pool construction (`src/training.rs:867-914`):
- Precompute `max_h` per k-mer in one pass via a `HashMap<usize, f64>`, eliminating the per-candidate nested linear scan.
- Replace per-candidate binary search into each child's profile with an inverted-pass merge-join: iterate each child's sparse profile once, writing into the flat matrix by candidate row.

Both changes are output-identical — they compute the same `entropies` and `profiles_flat` values, just more cheaply.

### Changes Required

#### 1. One-pass `max_h` precompute

**File**: `src/training.rs`
**Changes**: Insert a HashMap build above the candidate loop. Replace the nested scan at lines 890-895 with an O(1) lookup.

Before the `for &kmer_idx in &cand_sorted` loop (after the `cand_sorted.sort_unstable();` line added in Phase 0):
```rust
let mut max_h_by_kmer: HashMap<usize, f64> = HashMap::with_capacity(cand_sorted.len());
for child_h in &sorted_h {
    for &(ki, h) in child_h.iter() {
        max_h_by_kmer
            .entry(ki)
            .and_modify(|v| { if h > *v { *v = h; } })
            .or_insert(h);
    }
}
```

Replace lines 890-895 (the nested `for child_h in &sorted_h { for &(ki, h) in ... }`) with:
```rust
let max_h = *max_h_by_kmer.get(&kmer_idx).unwrap_or(&0.0);
```

#### 2. Inverted-pass profile matrix construction

**File**: `src/training.rs`
**Changes**: Replace the per-candidate binary-search loop (lines 881-899) with a single-pass scan over each child's profile. Emits rows into `profiles_flat` indexed by candidate position.

Replace lines 877-899 with:
```rust
let n_cand = cand_sorted.len();

// Index from kmer_idx -> candidate row (for O(1) lookup during inverted scan).
let kmer_to_row: HashMap<usize, usize> = cand_sorted.iter().enumerate()
    .map(|(row, &k)| (k, row))
    .collect();

let mut kmer_indices: Vec<usize> = cand_sorted.clone();
let mut entropies: Vec<f64> = cand_sorted.iter()
    .map(|k| *max_h_by_kmer.get(k).unwrap_or(&0.0))
    .collect();
let mut profiles_flat = vec![0.0f64; n_cand * n_children];

for (child_idx, p) in profiles.iter().enumerate() {
    for &(kmer_idx, val) in p.iter() {
        if let Some(&row) = kmer_to_row.get(&kmer_idx) {
            profiles_flat[row * n_children + child_idx] = val;
        }
    }
}
```

The subsequent permutation sort at lines 902-914 remains unchanged — it operates on `kmer_indices`, `entropies`, `profiles_flat` with the same semantics.

#### 3. Verify output unchanged

**File**: `tests/test_algorithmic_improvements.rs`
**Changes**: None. The equivalence test's `EXPECTED_HASH` must remain identical to Phase 0/1.

### Success Criteria

#### Automated Verification
- [ ] `cargo test` passes with unchanged `EXPECTED_HASH`.
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/1000refs` completes.
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/5000refs` completes.

#### Manual Verification
- [ ] 1k and 5k benches show additional speedup vs Phase 1 (target: 1.2-2×).
- [ ] Pool-construction share of total time (inferable from Criterion flame or before/after comparison) drops substantially.
- [ ] No regression on `learn_taxa/80seqs` default path.

---

## Phase 3: Parallel Candidate Scan / Update

### Overview

After Phase 1, the hot path per outer iteration is (a) a C-element argmax and (b) a C-element `max_corr` update. Both are embarrassingly parallel. Use `rayon::par_iter` with a deterministic reduction for the argmax and `par_iter_mut` for the update, preserving output.

**Interaction with outer tree parallelism**: `create_tree` already parallelizes sibling subtrees via `rayon::join`/`rayon::scope`. At root-level nodes (2-5 kingdom children) sibling parallelism is starved and inner parallelism pays; at deeply-branched nodes (order/family with 50-200 siblings) sibling parallelism saturates the pool and inner `par_iter` becomes sequential with small overhead. Rayon's work-stealing handles the transition cleanly.

### Changes Required

#### 1. Parallel argmax

**File**: `src/training.rs`
**Changes**: Replace the sequential argmax loop in Phase 1's step 1 with a deterministic parallel reduction.

Current (Phase 1 output):
```rust
let mut best_ci = None;
let mut best_gain = f64::NEG_INFINITY;
for ci in 0..n_cand {
    if is_selected[ci] { continue; }
    let base_h = sorted_entropies[ci];
    if base_h <= best_gain { break; }
    let gain = base_h * (1.0 - max_corr[ci]);
    if gain > best_gain {
        best_gain = gain;
        best_ci = Some(ci);
    }
}
```

New (deterministic parallel reduction, preserves lowest-index tie-breaking):
```rust
// Par-iter cannot easily honor the entropy-descending early exit, so we pay
// the full C scan per iteration. For typical C (hundreds to few thousand)
// this is still a win from parallelism; the early exit primarily helps late
// iterations where few candidates survive.
let (best_ci_opt, best_gain) = (0..n_cand)
    .into_par_iter()
    .filter(|&ci| !is_selected[ci])
    .map(|ci| {
        let base_h = sorted_entropies[ci];
        let gain = base_h * (1.0 - max_corr[ci]);
        (ci, gain)
    })
    .reduce(
        || (usize::MAX, f64::NEG_INFINITY),
        |a, b| {
            // Deterministic tie-break: prefer strictly higher gain; on tie,
            // prefer lower ci (matches sequential iteration order).
            if b.1 > a.1 { b }
            else if b.1 < a.1 { a }
            else if b.0 < a.0 { b }
            else { a }
        },
    );

let best_ci = if best_ci_opt == usize::MAX { None } else { Some(best_ci_opt) };
let ci = match best_ci { Some(ci) => ci, None => break };
// best_gain is used only by the sequential early-exit which we no longer have.
let _ = best_gain;
```

Note the tradeoff: we drop the `if base_h <= best_gain { break; }` early exit in favor of a full parallel scan. This increases worst-case work by some factor but gains from parallelism. For typical C in hundreds-to-thousands this is a net win; empirically verify via bench.

#### 2. Parallel `max_corr` update

**File**: `src/training.rs`
**Changes**: Replace the sequential update loop with `par_iter_mut`.

Current (Phase 1 output):
```rust
for cj in 0..n_cand {
    if is_selected[cj] { continue; }
    if max_corr[cj] >= 1.0 { continue; }
    let cj_prof = &sorted_profiles_flat[cj * n_children..(cj + 1) * n_children];
    let corr = pearson_with_stats(cj_prof, &cand_stats[cj], new_prof_slice, &new_st);
    if corr > max_corr[cj] { max_corr[cj] = corr; }
}
```

New:
```rust
let new_prof_arc: &[f64] = new_prof_slice; // lifetime ok across par_iter_mut
let new_st_ref: &ProfileStats = &new_st;

max_corr
    .par_iter_mut()
    .enumerate()
    .zip(cand_stats.par_iter())
    .for_each(|((cj, max_c), cj_st)| {
        if is_selected[cj] || *max_c >= 1.0 { return; }
        let cj_prof = &sorted_profiles_flat[cj * n_children..(cj + 1) * n_children];
        let corr = pearson_with_stats(cj_prof, cj_st, new_prof_arc, new_st_ref);
        if corr > *max_c { *max_c = corr; }
    });
```

Each `(cj, max_c)` tuple touches exactly one `max_corr` slot, so no aliasing.

#### 3. Gate by candidate count to avoid small-node overhead

**File**: `src/training.rs`
**Changes**: Add a threshold: below `PAR_THRESHOLD = 256` candidates, stay sequential. Above, go parallel.

```rust
const PAR_THRESHOLD: usize = 256;
let use_par = n_cand >= PAR_THRESHOLD && config.processors > 1;

// ... dispatch between sequential and parallel versions of both argmax and update
```

Keep the Phase 1 sequential version behind this gate so tests at `processors=1` have bit-exact equivalence with earlier phases.

### Success Criteria

#### Automated Verification
- [ ] `cargo test` passes. Equivalence test under `processors=1` has unchanged `EXPECTED_HASH`.
- [ ] Add a new test `test_correlation_aware_parallel_matches_sequential` that trains with `processors=1` and `processors=4` on the same data and asserts `decision_kmers` byte-equality.
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/1000refs` completes.
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/5000refs` completes.

#### Manual Verification
- [ ] At `processors=4`, 5k bench shows additional speedup vs Phase 2 (target: 1.5-3×).
- [ ] At `processors=1`, bench shows no regression (same as Phase 2 median).
- [ ] Parallel/sequential equivalence test passes reliably across multiple runs.

---

## Phase 4: Batched SIMD `sum_ab`

### Overview

After Phase 3, the `max_corr` update loop calls `pearson_with_stats` per candidate in parallel. Each call is a D-length dot product plus scalar ops. Inline the dot product into a tight loop over the contiguous `sorted_profiles_flat` so LLVM auto-vectorizes the full inner body, with `target-cpu=native` emitting AVX2/FMA instructions.

Expected output: bit-exact preserved — we're only removing a function-call boundary; the reduction order of `sum_ab` is unchanged.

**Not doing** in this phase: the dead-candidate skip (`is_dead` vector). Leaving Phase 1's `max_corr[cj] >= 1.0` runtime check in place keeps the implementation simpler and avoids introducing a second monotonic state that could in principle drift from `max_corr` if the update logic ever changes.

### Changes Required

#### 1. Inline `sum_ab` into the update

**File**: `src/training.rs`
**Changes**: Replace the `pearson_with_stats` call inside the parallel update with an inline dot product + Pearson calculation. This gives LLVM the entire inner loop to optimize without a function-call boundary.

```rust
let new_sum = new_st_ref.sum;
let new_denom = new_st_ref.denom;
let n_f = n_children as f64;
let denom_eps = 1e-15;

max_corr
    .par_iter_mut()
    .enumerate()
    .zip(cand_stats.par_iter())
    .for_each(|((cj, max_c), cj_st)| {
        if is_selected[cj] || *max_c >= 1.0 { return; }
        if cj_st.denom < denom_eps || new_denom < denom_eps { return; }

        let cj_prof = &sorted_profiles_flat[cj * n_children..(cj + 1) * n_children];

        // Inline dot product; LLVM will vectorize the loop body with
        // target-cpu=native, and no function-call boundary blocks scheduling.
        let mut sum_ab = 0.0f64;
        for i in 0..n_children {
            sum_ab += cj_prof[i] * new_prof_arc[i];
        }

        let num = n_f * sum_ab - cj_st.sum * new_sum;
        let corr = (num / (cj_st.denom * new_denom)).abs();

        if corr > *max_c { *max_c = corr; }
    });
```

Keep `pearson_with_stats` around for other callers (e.g., the non-parallel fallback path).

#### 3. Verify bit-exact output

**File**: `tests/test_algorithmic_improvements.rs`
**Changes**: None. The inlined `sum_ab` uses the same index ordering (`0..n_children`) as `.iter().zip(...).map(...).sum()` uses, so the reduction tree is LLVM's choice in both cases. If this phase produces a different hash, two paths: (a) accept the ULP-level drift and update `EXPECTED_HASH`, documenting; (b) revert and keep `pearson_with_stats`. Default: investigate first.

### Success Criteria

#### Automated Verification
- [ ] `cargo test` passes.
- [ ] Equivalence test passes. If hash drifts at ULP level, documented in commit message and updated intentionally.
- [ ] Parallel-vs-sequential equivalence test still passes.
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/1000refs` completes.
- [ ] `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/5000refs` completes.

#### Manual Verification
- [ ] Additional speedup visible at 5k (target: 1.3-2.5×).
- [ ] Final 10k bench run captured — compare to Phase 0 baseline.
- [ ] Cumulative speedup over Phase 0 baseline documented in timings file. Target: >100× at 5k.

---

## Testing Strategy

### Unit Tests

Already defined in the phases above:
- `test_correlation_aware_deterministic_output` (Phase 0) — bincode hash lock across runs.
- `test_correlation_aware_parallel_matches_sequential` (Phase 3) — processors=1 vs processors=4 equivalence.

Existing tests retained unchanged:
- `tests/test_algorithmic_improvements.rs:285 test_correlation_aware_changes_decision_kmers` — sanity check that correlation-aware differs from round-robin.
- `tests/test_algorithmic_improvements.rs:313 test_all_improvements_combined` — integration across all optional knobs.
- All 112 golden tests under `tests/golden/`.

### Benchmarks

All under `benches/oxidtaxa_bench.rs`:
- `learn_taxa_scaling_corr_aware/train/1000refs` and `/5000refs` — run at every phase.
- `learn_taxa_scaling_corr_aware/train/10000refs` — run at Phase 0 (baseline, if patient), Phase 1 end (milestone), Phase 4 end (final).
- `learn_taxa/80seqs_corr_aware` — verify no regression on small-data correlation-aware path.
- `learn_taxa/80seqs` — verify no regression on default (non-correlation-aware) path.
- `learn_taxa_scaling/train/*` — default-config scaling; no regression expected.

### Manual Testing Steps

After each phase:

1. Run `cargo test`. All tests pass.
2. Run `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/1000refs` (usually seconds-to-minutes).
3. Run `cargo bench --bench oxidtaxa_bench -- learn_taxa_scaling_corr_aware/train/5000refs` (minutes).
4. Compare Criterion's "change" output to prior phase. If regression, stop and investigate.
5. Append timing to `thoughts/shared/plans/2026-04-17-baseline-timings.md` with phase label and median time.
6. At milestones (Phase 1 complete, Phase 4 complete): run 10k bench.

## Performance Considerations

**Expected cumulative speedup** (rough, geometric):

| Phase | Mechanism | Expected Multiplier | Cumulative |
|-------|-----------|---------------------|------------|
| 0 | Determinism + benchmark | 1.0× (no perf change) | 1× |
| 1 | `max_corr` caching | R/2 ≈ 100-900× | 100-900× |
| 2 | Pool cleanups | 1.2-2× | 120-1800× |
| 3 | Parallelism (at `processors=4`) | 2-4× | 240-7200× |
| 4 | Inlined SIMD `sum_ab` | 1.3-2× | 310-14400× |

The enormous Phase 1 range reflects sensitivity to R: at small R (default `record_kmers_fraction=0.10`, typical R ≈ 50) the win is ~25×; at large R (0.44, k=6) it's ~900×. The user's slow path is the large-R regime, so expect the high end.

**Memory**: Phase 1 adds `Vec<f64>` of length C (≤ few KB per node). Phase 2 adds two `HashMap`s sized C (a few tens of KB). Phase 3 has no allocation beyond what Phase 1 adds. Phase 4 adds `Vec<bool>` of length C. All negligible compared to existing profile matrix.

**Thread-pool contention**: at nodes with small C (< `PAR_THRESHOLD = 256`), Phase 3 falls back to sequential to avoid rayon overhead. At root-level expensive nodes, parallel is always used.

## Migration Notes

No API changes. `TrainConfig.correlation_aware_features` and `BuildTreeConfig.correlation_aware_features` retain their current shape. Serialized training sets (`TrainingSet`) remain binary-compatible — the `decision_kmers` structure and values are produced by the same algorithm.

Determinism change from Phase 0 means that any caller persisting `TrainingSet` outputs across pre- and post-Phase-0 versions may see differences in selected k-mers when entropy ties occurred. This is inherent in moving from random to deterministic ordering; not reversible without reintroducing `HashSet` non-determinism.

## References

- Research: `thoughts/shared/research/2026-04-17-correlation-aware-training-bottlenecks.md`
- Prior plan (v1, already implemented): `thoughts/shared/plans/2026-04-15-correlation-aware-training-speedup.md`
- Hot path: `src/training.rs:747-1058` (`create_tree`), `src/training.rs:857-975` (correlation-aware branch), `src/training.rs:930-974` (greedy loop)
- Config: `src/types.rs:216-272` (`TrainConfig`), `src/types.rs:106-112` (`BuildTreeConfig`)
- Benchmark harness: `benches/oxidtaxa_bench.rs:378-403` (`bench_learn_taxa_scaling` pattern to clone)
- Build config: `Cargo.toml:26-28`, `.cargo/config.toml:1-2`
- Existing correlation-aware test: `tests/test_algorithmic_improvements.rs:285-306`
