---
date: 2026-04-17T14:33:34Z
researcher: Ryan Martin
git_commit: 0a684f5263f9ffd7d58426b0a7528a61c5cc6800
branch: main
repository: eDNA-Explorer/oxidtaxa
topic: "Improving correlation_aware_features training speed beyond current optimizations"
tags: [research, training, correlation-aware, performance, feature-selection, greedy, pearson]
status: complete
last_updated: 2026-04-17
last_updated_by: Ryan Martin
---

# Research: Improving `correlation_aware_features` Training Speed

**Date**: 2026-04-17T14:33:34Z
**Researcher**: Ryan Martin
**Git Commit**: 0a684f5263f9ffd7d58426b0a7528a61c5cc6800
**Branch**: main
**Repository**: eDNA-Explorer/oxidtaxa

## Research Question

When `correlation_aware_features=true`, training is extremely slow. Investigate the current implementation, what has already been optimized, and identify concrete additional optimizations that would reduce wall-clock time without changing algorithmic output.

## Summary

The `2026-04-15-correlation-aware-training-speedup.md` plan has already been fully implemented (commit `aba091c`). That plan's four phases landed: a correlation-aware Criterion benchmark (`bench_learn_taxa_correlation_aware`), precomputed Pearson statistics (`ProfileStats` / `pearson_with_stats`), `.cargo/config.toml` with `target-cpu=native`, a flat row-major candidate profile matrix, and `rayon::join`/`rayon::scope` parallelism over sibling subtrees in `create_tree`.

Despite those wins, one dominant inefficiency remains that explains the user's "extremely slow" experience: **the inner greedy loop at `src/training.rs:930-974` recomputes every candidate-vs-selected correlation from scratch on every outer iteration**, giving total work O(R² × C × D) where R is the number of features to record, C the candidate pool size, D = `n_children`. At `k=6`, `record_kmers_fraction=0.44`, R can reach ~1,800 and C ~3,600, making the redundant work factor ~R/2 ≈ 900× beyond what's necessary.

Two secondary hotspots also remain: (a) per-candidate `max_h` lookup on lines 890-895 is O(C × D × max_nonzero) via linear scan, and (b) candidate profile construction on lines 881-899 uses binary search per candidate per child, an O(C × D × log) pattern, when an inverted-pass merge would be O(D × (|p| + C)).

This document describes the current implementation in detail (Section: "Detailed Findings"), then — because the user explicitly asked for improvements — lays out a prioritized ranking of optimizations that would not require changing algorithm output (Section: "Improvement Opportunities").

## Detailed Findings

### Public API surface

Training lives in `src/training.rs` (1,058 lines) and exposes four top-level Rust entry points:

- `learn_taxa()` (`src/training.rs:19-43`) — monolithic path, used by the legacy `train()` PyO3 function at `src/lib.rs:72-119`.
- `prepare_data()` (`src/training.rs:48-61`) — phase 1 (k-mer enumeration, taxonomy tree, IDF weights).
- `build_tree()` (`src/training.rs:66-75`) — phase 2 (decision tree construction; invokes `create_tree()`).
- `learn_fractions()` (`src/training.rs:81-92`) — phase 3 (fraction-learning loop).

Each phase creates its own local rayon thread pool sized to `config.processors` and runs all work inside `pool.install(...)`.

The Python layer (`src/lib.rs:11-360`) mirrors the three-phase split via `prepare_data_py()`, `build_tree_py()`, and `learn_fractions_py()`, all releasing the GIL before entering Rust.

### `create_tree` architecture (`src/training.rs:747-1058`)

`create_tree` is a recursive function producing `(SparseProfile, usize, Vec<(usize, DecisionNode)>)`:
- The `SparseProfile` is the weighted-average profile for this subtree.
- The `usize` is a descendant count.
- The `Vec<(usize, DecisionNode)>` returns all decision nodes this subtree built; the caller scatters them into `decision_kmers[idx]` back in `_build_tree_inner` (`src/training.rs:399-410`).

#### Sibling-subtree parallelism
Lines 763-803 route based on `config.processors`:
- `processors > 1`, exactly 2 children → `rayon::join` (lines 767-772).
- `processors > 1`, more children → `rayon::scope` with `s.spawn` per child (lines 774-786).
- `processors == 1` → sequential loop (lines 794-802).

Each spawned task receives `&children`, `&sequences`, `&kmers`, `n_kmers`, `&config` as shared borrows; return values are collected into `child_results` and consumed serially.

#### Profile merge and per-child entropy
After child subtrees finish, a weighted-average `q` profile is computed via `merge_sparse_profiles` (`src/training.rs:661-689`) using descendant weights determined by `config.descendant_weighting`. Per-child cross-entropy `H_i = -p_i × ln(q_i)` is built at lines 822-839, then each child's H vector is sorted entropy-descending (lines 842-845).

#### `record_kmers` calculation
`record_kmers = ceil(max_nonzero × record_kmers_fraction)` where `max_nonzero = max(|p_i|) across children` (`src/training.rs:848-855`).

### Correlation-aware feature selection (`src/training.rs:857-975`)

This is the branch the user turns on. Implementation steps:

1. **Candidate pool construction** (lines 867-874):
   ```rust
   let per_child_limit = record_kmers * 2;
   let mut cand_set: HashSet<usize> = HashSet::new();
   for child_h in &sorted_h {
       for &(kmer_idx, _) in child_h.iter().take(per_child_limit) {
           cand_set.insert(kmer_idx);
       }
   }
   ```
   Pool size C is up to `n_children × per_child_limit` = `n_children × record_kmers × 2` before dedup. After the HashSet dedupes across children, typical C is smaller but can reach low-thousands.

2. **Per-candidate profile vector construction** (lines 881-899):
   ```rust
   for &kmer_idx in &cand_set {
       for p in profiles.iter() {
           let val = match p.binary_search_by_key(&kmer_idx, |&(k, _)| k) {
               Ok(pos) => p[pos].1,
               Err(_) => 0.0,
           };
           prof_vec.push(val);
       }
       let mut max_h = 0.0f64;
       for child_h in &sorted_h {
           for &(ki, h) in child_h.iter() {
               if ki == kmer_idx && h > max_h { max_h = h; break; }
           }
       }
       kmer_indices.push(kmer_idx);
       entropies.push(max_h);
       profiles_flat.extend_from_slice(&prof_vec);
   }
   ```
   Two nested loops per candidate:
   - Profile build: binary-search each child's sparse profile for the k-mer — O(D × log max_|p|) per candidate.
   - `max_h` lookup: linear scan of every child's `sorted_h` looking for the current `kmer_idx`. This is O(D × max_nonzero) per candidate, so O(C × D × max_nonzero) total. At the largest nodes this can be significant.

3. **Permutation sort by entropy descending** (lines 902-914):
   Creates `sorted_kmer_indices`, `sorted_entropies`, `sorted_profiles_flat` arrays. One sort on order indices, then a materialization loop copying profile slices into the sorted positions.

4. **Precompute `ProfileStats`** (lines 917-919) — `ProfileStats::new()` computes sum, sum_sq, and `denom = sqrt(n × sum_sq - sum²)` for each candidate. Used by `pearson_with_stats()` (line 714-722) to avoid recomputing vector statistics for repeat pairings.

5. **Greedy forward-selection loop** (lines 930-974):
   ```rust
   for _ in 0..record_kmers {                            // outer: R iterations
       let mut best_ci = None;
       let mut best_gain = f64::NEG_INFINITY;

       for ci in 0..n_cand {                              // candidate scan: up to C
           if is_selected[ci] { continue; }
           let base_h = sorted_entropies[ci];
           if base_h <= best_gain { break; }              // entropy-sorted early exit

           let cand_prof = &sorted_profiles_flat[ci * n_children..(ci + 1) * n_children];
           let cand_st = &cand_stats[ci];

           let mut max_corr: f64 = 0.0;
           for si in 0..n_selected {                      // inner: grows 0→R-1
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
   ```

   **Behavior per outer iteration `t` (0-indexed):**
   - For each not-yet-selected candidate `ci`, compute `max_corr[ci]` by pairing with all `t` currently-selected features, then compute `gain = base_h × (1 − max_corr)`.
   - Pick the candidate with highest gain (ties go to lowest `ci`).

   **Work done per outer iteration:** Up to C candidates scanned, each computing up to `t` correlations, each correlation doing a D-length dot product. Per-iter cost ≈ C × t × D.

   **Total work across R outer iterations:** Σ(t=0..R-1) C × t × D = C × R(R−1)/2 × D ≈ C × R² × D / 2.

   The entropy-descending sort + `if base_h <= best_gain` is a pruning rule that helps when the best candidate is found early. When the pool is densely correlated (the user's slow scenario), pruning typically kicks in late in the scan, giving limited relief.

### `ProfileStats` and `pearson_with_stats` (`src/training.rs:691-722`)

```rust
#[derive(Clone)]
struct ProfileStats {
    sum: f64,
    denom: f64,  // sqrt(n * sum_sq - sum * sum)
}

impl ProfileStats {
    fn new(v: &[f64]) -> Self {
        let n_f = v.len() as f64;
        let sum: f64 = v.iter().sum();
        let sum_sq: f64 = v.iter().map(|x| x * x).sum();
        let denom = (n_f * sum_sq - sum * sum).sqrt();
        Self { sum, denom }
    }
}

#[inline]
fn pearson_with_stats(a: &[f64], a_stats: &ProfileStats,
                      b: &[f64], b_stats: &ProfileStats) -> f64 {
    let n_f = a.len() as f64;
    if a.len() < 2 { return 0.0; }
    if a_stats.denom < 1e-15 || b_stats.denom < 1e-15 { return 0.0; }
    let sum_ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let num = n_f * sum_ab - a_stats.sum * b_stats.sum;
    (num / (a_stats.denom * b_stats.denom)).abs()
}
```

This is the per-pair hot path. The only non-precomputed work is the `sum_ab` dot product over `a` and `b` — a D-length iterator chain. With `target-cpu=native` and LLVM auto-vectorization, this should compile to SIMD, but the dispatch overhead of the containing function call and the per-pair loop setup still dominates for short D.

### Build configuration

- `Cargo.toml:26-28`: `lto = true`, `codegen-units = 1` on release — maximal per-binary optimization.
- `.cargo/config.toml:1-2`: `rustflags = ["-C", "target-cpu=native"]` — AVX2/FMA enabled on local builds.
- No PGO (profile-guided optimization) setup present.

### Benchmark harness

`benches/oxidtaxa_bench.rs:243-280` — `bench_learn_taxa_correlation_aware` runs the 80-sequence test fasta with `correlation_aware_features: true` and `record_kmers_fraction: 0.44`. Registered in the `criterion_group!` macro at line 454. Runnable via:
```
cargo bench --bench oxidtaxa_bench -- learn_taxa/80seqs_corr_aware
```

### What has NOT been tried (based on plan + repo state)

- Incremental/cached `max_corr` across outer iterations.
- Matrix-batched `sum_ab` via a single pass over `sorted_profiles_flat` per outer iteration.
- Rayon parallelism over the candidate scan within a node.
- HashMap-based `max_h` lookup precompute.
- Inverted-pass candidate profile construction.
- `f32` profiles.
- PGO (profile-guided optimization).
- Lazy-greedy (CELF/CELF++) heap ordering.

## Improvement Opportunities

The user's command explicitly asked for improvements, so below is a prioritized list. Expected speedup multipliers stack roughly multiplicatively. Each item is labeled with risk category:
- **OUTPUT-IDENTICAL**: algorithm produces bit-exact output as today (within floating point summation order — effectively identical for practical purposes).
- **NEAR-IDENTICAL**: output may differ only on exact tie-breaking (edge case).
- **APPROXIMATE**: intentionally trades a small amount of selection quality for speed.

### #1 — Cache `max_corr` incrementally across outer iterations [OUTPUT-IDENTICAL]

**The big win.** The greedy algorithm has a key monotonicity property: as more features are selected, `max_corr_with_selected[ci]` is non-decreasing per candidate. So maintaining it across outer iterations — and only updating each entry against the single newly-selected feature — yields the same output with far less work.

**Current complexity:** O(R² × C × D)
**New complexity:** O(R × C × D)
**Speedup factor:** ~R/2 (at R=1,800, about **900×**)

Sketch:

```rust
// Before entering the outer loop:
let mut max_corr: Vec<f64> = vec![0.0; n_cand];

for _ in 0..record_kmers {
    // 1. Pick best candidate using cached max_corr.
    let mut best_ci = None;
    let mut best_gain = f64::NEG_INFINITY;
    for ci in 0..n_cand {
        if is_selected[ci] { continue; }
        let base_h = sorted_entropies[ci];
        // Same entropy-descending early exit works because gain ≤ base_h.
        if base_h <= best_gain { break; }
        let gain = base_h * (1.0 - max_corr[ci]);
        if gain > best_gain {
            best_gain = gain;
            best_ci = Some(ci);
        }
    }

    // 2. Select and update max_corr against only the newest feature.
    let ci = match best_ci { Some(ci) => ci, None => break };
    is_selected[ci] = true;
    result_set.insert(sorted_kmer_indices[ci]);
    let src = ci * n_children;
    let new_prof = &sorted_profiles_flat[src..src + n_children];
    let new_st = cand_stats[ci].clone();

    // 3. Update max_corr for every not-yet-selected candidate.
    for cj in 0..n_cand {
        if is_selected[cj] { continue; }
        if max_corr[cj] >= 1.0 { continue; } // no further updates can help
        let cj_prof = &sorted_profiles_flat[cj * n_children..(cj + 1) * n_children];
        let corr = pearson_with_stats(cj_prof, &cand_stats[cj], new_prof, &new_st);
        if corr > max_corr[cj] { max_corr[cj] = corr; }
    }

    sel_profiles_flat.extend_from_slice(new_prof); // kept for downstream selected_profiles, if needed
    sel_stats.push(new_st);
}
```

**Correctness argument:** At outer iteration `t`, with selected set `S_t` and candidate `ci`,
`max_corr[ci] = max(corr(ci, s) for s in S_t)`. The current code computes this in an inner loop; the cached version maintains it incrementally. Same value, same argmax (same tie-breaking: both iterate `ci` in 0..n_cand order and use strict `>`), same output.

**Caveats / things to verify:**
- Floating-point summation order differs slightly for `sum_ab`, but `ProfileStats::denom` uses the exact same formula and the only new arithmetic is the running max. Empirical equivalence on golden tests should hold — but run `cargo test` to confirm.
- The `if max_corr >= 1.0 { break }` from the original was an inner-loop optimization; here it becomes a per-candidate "is this candidate dead?" skip. Near-identical behavior.

### #2 — Batch the `sum_ab` for all candidates into one SIMD-friendly pass [OUTPUT-IDENTICAL]

After #1, each outer iteration does one "correlation update" against the newly-selected feature. That computes, for each not-selected candidate, a single D-length dot product. Expressed over the contiguous `sorted_profiles_flat` buffer this is a matrix-vector product: `sum_ab[i] = ⟨profiles_flat[i × D .. (i+1) × D], new_profile⟩`.

Done as a single tight loop, LLVM auto-vectorizes it cleanly with `target-cpu=native`:

```rust
let new_prof = &sorted_profiles_flat[src..src + n_children];
let new_sum = cand_stats[ci].sum;
let new_denom = cand_stats[ci].denom;
let n_f = n_children as f64;

sorted_profiles_flat
    .chunks_exact(n_children)
    .zip(cand_stats.iter().zip(max_corr.iter_mut()))
    .enumerate()
    .for_each(|(cj, (cj_prof, (cj_st, max_c)))| {
        if is_selected[cj] || *max_c >= 1.0 { return; }
        if cj_st.denom < 1e-15 { return; }
        let sum_ab: f64 = cj_prof.iter().zip(new_prof.iter())
            .map(|(&x, &y)| x * y).sum();
        let num = n_f * sum_ab - cj_st.sum * new_sum;
        let corr = (num / (cj_st.denom * new_denom)).abs();
        if corr > *max_c { *max_c = corr; }
    });
```

For larger tree nodes, replacing this hand loop with `matrixmultiply` (already a transitive dep-possible; pure Rust) or `ndarray` + `blas-src` gives a proper GEMV. For a typical profile-flat matrix of ~3,600 × 50, a single GEMV is ~180K FMAs — a couple of microseconds on modern CPU. Expected stack-on speedup over #1 alone: **1.5–4×**.

### #3 — Parallelize the candidate update loop with rayon [OUTPUT-IDENTICAL]

Both the argmax over C candidates (step 1) and the max_corr update (step 3) are embarrassingly parallel. Once #1 is in place:

```rust
let (best_ci, best_gain) = max_corr.par_iter()
    .zip(sorted_entropies.par_iter())
    .enumerate()
    .filter(|(ci, _)| !is_selected[*ci])
    .map(|(ci, (&mc, &base_h))| (ci, base_h * (1.0 - mc)))
    .reduce(|| (usize::MAX, f64::NEG_INFINITY),
            |a, b| if b.1 > a.1 { b } else { a });
```

And the update as a `par_iter_mut` over `max_corr` with the same index mapping.

**Important interaction** with existing `create_tree` parallelism: at root-level nodes (2-5 children), sibling parallelism is limited and inner parallelism fills the pool. At deeply-branched levels (order/family with 50-200 siblings), sibling parallelism saturates and inner `par_iter` becomes near-no-op overhead. Rayon's work-stealing handles this reasonably. Consider using `par_iter` unconditionally and letting the scheduler decide, or gating on `n_cand > threshold` (e.g., 512) to avoid overhead on small nodes.

Expected speedup: near-linear in number of idle threads during large-node evaluation. **Up to Nx** on root-level expensive nodes.

### #4 — Precompute `max_h` via one-pass HashMap [OUTPUT-IDENTICAL]

Current code at `src/training.rs:890-895` computes `max_h` per candidate by scanning every child's `sorted_h`:

```rust
let mut max_h = 0.0f64;
for child_h in &sorted_h {
    for &(ki, h) in child_h.iter() {
        if ki == kmer_idx && h > max_h { max_h = h; break; }
    }
}
```

This runs inside the `for &kmer_idx in &cand_set` loop, so total work is O(C × D × avg_child_h_len). At worst case (`avg_child_h_len ≈ C`), this is O(C² × D) — comparable to the correlation loop's cost.

Replacement: precompute once, before the candidate loop.

```rust
let mut max_h_by_kmer: HashMap<usize, f64> = HashMap::with_capacity(cand_set.len());
for child_h in &sorted_h {
    for &(ki, h) in child_h.iter() {
        max_h_by_kmer
            .entry(ki)
            .and_modify(|v| { if h > *v { *v = h; } })
            .or_insert(h);
    }
}
// Then per-candidate: *max_h_by_kmer.get(&kmer_idx).unwrap_or(&0.0)
```

Construction: O(Σ|child_h|) = O(D × max_nonzero). Per-candidate lookup: O(1) expected.

Expected speedup on pool construction: **3–10×**. Smaller absolute impact than #1 but easy win.

### #5 — Inverted-pass candidate profile construction [OUTPUT-IDENTICAL]

The current per-candidate binary search over every child's sparse profile (lines 882-889) is O(C × D × log max_|p|). Since both `cand_set` (once materialized sorted) and each `p` (already sorted by k-mer index) are sorted, you can merge-join once per child to emit all candidate rows for that child in one pass:

```rust
let mut sorted_cands: Vec<usize> = cand_set.iter().copied().collect();
sorted_cands.sort_unstable();
let c_count = sorted_cands.len();

// row-major: cand_row × n_children, but we'll fill column-by-column.
// Allocate once, zeroed.
let mut profiles_flat = vec![0.0f64; c_count * n_children];
for (child_idx, p) in profiles.iter().enumerate() {
    let mut ci = 0; // cursor into sorted_cands
    for &(kmer_idx, val) in p.iter() {
        while ci < c_count && sorted_cands[ci] < kmer_idx { ci += 1; }
        if ci >= c_count { break; }
        if sorted_cands[ci] == kmer_idx {
            profiles_flat[ci * n_children + child_idx] = val;
        }
    }
}
```

Complexity per child: O(|p| + C). Across D children: O(D × (max|p| + C)). Compared to the current O(C × D × log) this is 2–5× faster when C is large.

### #6 — Skip dead candidates with `max_corr >= 1.0` [OUTPUT-IDENTICAL]

Once a candidate's `max_corr` reaches 1.0 it contributes gain 0 and cannot be selected. Mark it dead and exclude from both argmax and update passes:

```rust
let mut is_dead = vec![false; n_cand];
// after update:
if max_corr[cj] >= 1.0 { is_dead[cj] = true; }
// in scans:
if is_selected[cj] || is_dead[cj] { continue; }
```

Minor constant-factor win. Already partly handled by the inner-loop break in today's code, but with the cached version we can make it a permanent skip.

### #7 — Cap or shrink the candidate pool via absolute limit [NEAR-IDENTICAL]

`per_child_limit = record_kmers * 2` scales with R. When R is large (e.g., 1,800 at k=6, fraction=0.44), the pool can exceed the useful information. Options:

a. Apply an absolute cap: `per_child_limit = min(record_kmers * 2, 1024)`.
b. Drop candidates with `max_h < some_small_epsilon` — they can't be chosen even with zero correlation.

Option (b) is safer — it only trims candidates whose best-case gain is below other candidates already explored. Risk: rarely, a low-entropy but uncorrelated candidate is eventually picked. Empirical verification via golden test is required.

Expected speedup: 1.2–2× via reduced C.

### #8 — Lazy-greedy / CELF heap ordering [OUTPUT-IDENTICAL]

Submodular greedy maximization has a well-known "lazy greedy" speedup (Minoux 1978; CELF/CELF++ in Leskovec et al. 2007). Keep a max-heap keyed on `(upper_bound_gain, ci)`. Each outer iteration:
- Pop top candidate.
- If its `max_corr[ci]` has been updated since last insertion (tracked via a small "last_updated_iter" per candidate), recompute its current gain and push back.
- If it's fresh (up to date), commit it.

For submodular functions, the marginal gain is non-increasing over iterations. So most candidates never need full recomputation after a few outer iterations — their stale upper bound already loses to the top of heap.

Expected speedup on top of #1+#2: **2–8×** in practice.

Implementation complexity is moderate (needs a heap + per-candidate update counter). Only worth it if #1 alone doesn't get you to acceptable performance.

### #9 — f32 profiles [NEAR-IDENTICAL]

Profile values are probabilities in [0,1]. f32 has ~7 decimal digits of precision, which is enough for correlation numerators/denominators. Switching `sorted_profiles_flat` and `ProfileStats` to f32 halves memory footprint and doubles SIMD throughput.

Cost: `DecisionNode.profiles` is `Vec<Vec<f64>>` and is consumed by `vector_sum` in the fraction-learning phase. Would require either (a) converting back to f64 at DecisionNode construction time (cheap, one-time) or (b) broader refactor.

Expected speedup: 1.5–2× on correlation inner loops. Small magnitude compared to #1 but composable.

### #10 — Profile-Guided Optimization (PGO) [OUTPUT-IDENTICAL]

Standard Cargo support (`cargo-pgo` helper crate). Workflow:
1. Build with `-C profile-generate=/tmp/pgo`
2. Run the correlation-aware benchmark
3. Merge profiles
4. Build with `-C profile-use=/tmp/pgo/merged.profdata`

Expected speedup: 5–15% on hot loops when the optimizer specializes branches toward observed behavior. Cheap to add since it's build-config only.

---

### Summary: Recommended Implementation Order

| Order | Change | Risk | Expected stacked speedup |
|-------|--------|------|--------------------------|
| 1 | Incremental `max_corr` caching | OUTPUT-IDENTICAL | **~R/2, ≈900× at R=1800** |
| 2 | Batched SIMD `sum_ab` pass | OUTPUT-IDENTICAL | 1.5–4× on top |
| 3 | Rayon parallel candidate update | OUTPUT-IDENTICAL | up to N× on root nodes |
| 4 | HashMap `max_h` precompute | OUTPUT-IDENTICAL | speeds pool construction 3–10× |
| 5 | Inverted-pass profile build | OUTPUT-IDENTICAL | speeds pool construction 2–5× |
| 6 | Skip dead candidates | OUTPUT-IDENTICAL | small constant |
| 7 | Candidate pool cap via entropy threshold | NEAR-IDENTICAL | 1.2–2× |
| 8 | Lazy-greedy heap | OUTPUT-IDENTICAL | 2–8× on top of #1 |
| 9 | f32 profiles | NEAR-IDENTICAL | 1.5–2× |
| 10 | PGO | OUTPUT-IDENTICAL | 1.05–1.15× |

**Do #1 first.** Everything else is additional leverage but #1 alone likely brings the correlation-aware path into "tolerable" territory for most workloads.

## Code References

- `src/training.rs:691-722` — `ProfileStats` struct and `pearson_with_stats` (already optimized)
- `src/training.rs:727-741` — `pearson_abs` (unused, kept for parity testing)
- `src/training.rs:747-1058` — `create_tree` recursive function
- `src/training.rs:763-803` — sibling-subtree rayon parallelism (already optimized)
- `src/training.rs:848-855` — `record_kmers` calculation (R)
- `src/training.rs:857-975` — correlation-aware feature selection branch (hot path)
- `src/training.rs:867-874` — candidate pool construction (C)
- `src/training.rs:881-899` — per-candidate profile vector + `max_h` lookup (improvement targets #4, #5)
- `src/training.rs:902-914` — permutation sort by entropy
- `src/training.rs:917-919` — `ProfileStats` precomputation
- `src/training.rs:930-974` — greedy forward-selection loop (improvement target #1, #2, #3)
- `src/training.rs:976-1002` — round-robin selection (fallback when correlation_aware_features=false)
- `src/types.rs:216-272` — `TrainConfig` with `correlation_aware_features` field at line 247
- `src/types.rs:106-124` — `BuildTreeConfig` / `LearnFractionsConfig`
- `src/lib.rs:72-119` — PyO3 `train()` function
- `src/lib.rs:234-260` — PyO3 `build_tree_py()` staged entry
- `benches/oxidtaxa_bench.rs:243-280` — `bench_learn_taxa_correlation_aware`
- `Cargo.toml:26-28` — release profile with `lto = true`, `codegen-units = 1`
- `.cargo/config.toml:1-2` — `target-cpu=native` rustflag

## Architecture Documentation

The training pipeline is now a three-phase staged API:

```
  prepare_data() ──▶ PreparedData ──▶ build_tree() ──▶ BuiltTree ──▶ learn_fractions() ──▶ TrainingSet
      ↑                (cheap)         ↑                (slow)         ↑               (modest)
  (k, seed_pattern)               (record_kmers_fraction,       (training_threshold,
                                   descendant_weighting,         use_idf_in_training,
                                   correlation_aware_features)   leave_one_out)
```

The `correlation_aware_features` flag lives in `BuildTreeConfig` / `TrainConfig` and is consumed only inside `create_tree`. Everything in `PreparedData` and `learn_fractions` is unaffected by the flag.

Each phase builds its own local rayon thread pool and runs work inside `pool.install(...)`, so `config.processors` is honored even when called from Python threads with the GIL released.

Sibling-subtree parallelism is the outermost level of parallelism inside `create_tree`. Rayon's work-stealing dynamically balances work across threads. At root-level nodes with few children, the inner feature-selection loop dominates wall-clock time, which is why inner parallelism (#3 above) has value despite the outer parallelism.

## Historical Context (from thoughts/)

- `thoughts/shared/plans/2026-04-15-correlation-aware-training-speedup.md` — the four-phase plan that has already been implemented as commit `aba091c`. All four phases (benchmark, ProfileStats, target-cpu=native, batch matrix layout, parallel create_tree) are reflected in the current code. This document's recommendations extend past those.
- `thoughts/shared/plans/2026-04-15-staged-training-cache-v2.md` — the three-phase API refactor, implemented as commit `29151a3`. This is the wrapper around `build_tree`; speedups to `create_tree` compose with this caching.
- `thoughts/shared/plans/2026-04-15-staged-training-cache-review.md` — review of the v1 plan, which led to the v2 revision.
- `thoughts/shared/research/2026-04-13-algorithmic-improvements.md` — broader context on recent feature-selection algorithmic knobs (`record_kmers_fraction`, `descendant_weighting`, `correlation_aware_features`).
- `thoughts/shared/research/2026-04-15-new-parameter-audit.md` — audit of parameter surface exposed to Optuna; relevant because high R = high `record_kmers_fraction` is what makes correlation-aware expensive.

## Related Research

- `thoughts/shared/research/2026-04-13-algorithmic-improvements.md`
- `thoughts/shared/research/2026-04-15-new-parameter-audit.md`
- `thoughts/shared/plans/2026-04-15-correlation-aware-training-speedup.md`

## Open Questions

1. **Profile floating-point equivalence**: does incremental `max_corr` caching produce identical output to the current code on the golden tests? Expected yes (same arithmetic, different loop ordering only for the skipped redundant recomputation). Verify via `cargo test` after change.
2. **Optimal parallelism granularity**: at what `n_cand` threshold does inner-loop rayon parallelism help versus hurt due to rayon task overhead? A small Criterion sweep can answer this.
3. **Empirical validation of candidate pool cap (#7)**: does capping `per_child_limit` change test outputs on realistic tree shapes, and if so by how much at classification time?
4. **Does PGO help?** Quick to test — build with profile-generate, run the bench, rebuild with profile-use, re-bench.
