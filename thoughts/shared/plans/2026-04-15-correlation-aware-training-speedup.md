# Correlation-Aware Training Speed Optimization Plan

## Overview

The correlation-aware feature selection path (`correlation_aware_features=true`) in `create_tree` becomes extremely expensive with certain parameter combinations (low `k` like 6, high `record_kmers_fraction` like 0.44) on large datasets (177k sequences). This plan implements four optimizations that compound to deliver a large overall speedup without changing any algorithmic behavior or output.

## Current State Analysis

The bottleneck is the greedy forward-selection loop at `src/training.rs:747-780`. For each of `record_kmers` slots, it scans all candidates and computes Pearson correlation against every already-selected feature. The per-node cost is O(R^2 x C x D) where:

- R = `record_kmers` = `ceil(max_nonzero * record_kmers_fraction)` (line 700)
- C = candidate pool size, up to `n_children * record_kmers * 2` (line 712)
- D = `n_children`, the length of each profile vector (line 638)

With `k=6` and `fraction=0.44`, R ~ 1,803 and C ~ 3,606 * n_children. The default path (k=8, fraction=0.10, round-robin) gives R ~ 50, making the correlation-aware path ~1,300x more expensive per node in raw iteration count.

### Key Discoveries:
- `pearson_abs` (line 609-623) recomputes `sum_a`, `sum_a2`, `sum_b`, `sum_b2` from scratch on every call. For a fixed candidate, `sum_a`/`sum_a2` are constant across all selected features. For a fixed selected feature, `sum_b`/`sum_b2` are constant across all candidates. This is pure redundant work.
- Profile vectors are short (2-200 f64s = `n_children`). SIMD on individual calls is marginal; the win comes from batching many candidate-vs-selected correlations into a matrix operation.
- `create_tree` (line 628-864) is fully sequential. Sibling subtrees are provably independent (write to disjoint `decision_kmers` indices, all other params are immutable borrows). This is the biggest structural gap.
- No existing benchmark exercises the correlation-aware path (all use `TrainConfig::default()` which sets `correlation_aware_features: false`).
- No `.cargo/config.toml` exists. All builds are local `maturin develop`. No CI pipeline. `target-cpu=native` is safe to add.

## Desired End State

After all phases:
1. A Criterion benchmark exists for correlation-aware training, enabling before/after measurement.
2. The Pearson correlation inner loop does ~50% less arithmetic per pair (precomputed statistics).
3. The batch matrix formulation computes all candidate-vs-selected correlations via contiguous memory access patterns that LLVM can auto-vectorize.
4. `target-cpu=native` unlocks AVX2/FMA instructions for all floating-point loops.
5. `create_tree` processes sibling subtrees in parallel using the existing rayon thread pool, giving near-linear speedup with processor count for the tree-building phase.

### Verification:
- All existing golden tests pass unchanged (`cargo test`)
- Criterion benchmarks show measurable improvement on the correlation-aware path
- The 80-sequence `learn_taxa` benchmark does not regress

## What We're NOT Doing

- Changing the algorithm or its output (all optimizations are performance-only)
- Adding new configuration parameters (no `max_record_kmers` cap -- that's a separate decision)
- Optimizing the fraction-learning loop (already parallelized via `par_iter`)
- Adding explicit SIMD intrinsics (relying on LLVM auto-vectorization instead)
- Setting up CI or wheel-building infrastructure

## Implementation Approach

The four phases build on each other in a logical progression. Phases 1 and 3 are tightly coupled (precompute stats, then restructure for batch). Phase 2 is independent. Phase 4 is the largest structural change and is done last so the simpler wins land first.

---

## Phase 0: Add Correlation-Aware Benchmark

### Overview
Add a Criterion benchmark that exercises the correlation-aware feature selection path. Without this, we cannot measure the impact of any optimization.

### Changes Required:

#### 1. Benchmark function
**File**: `benches/oxidtaxa_bench.rs`
**Changes**: Add `bench_learn_taxa_correlation_aware` after the existing `bench_learn_taxa` function (line 241). Reuse the same test data (80 sequences) but with `correlation_aware_features: true`.

```rust
fn bench_learn_taxa_correlation_aware(c: &mut Criterion) {
    let (names, seqs) = read_fasta(&test_data_path("test_ref.fasta")).unwrap();
    let taxonomy = oxidtaxa::fasta::read_taxonomy(
        &test_data_path("test_ref_taxonomy.tsv"), &names
    ).unwrap();

    let mut filtered_seqs = Vec::new();
    let mut filtered_tax = Vec::new();
    for (i, seq) in seqs.iter().enumerate() {
        let tax = &taxonomy[i];
        let full_tax = format!("Root; {}", tax.replace(";", "; "));
        let rank_count = full_tax.split("; ").count();
        if rank_count < 4 { continue; }
        if seq.len() < 30 { continue; }
        let n_count = seq.bytes().filter(|&b| b == b'N' || b == b'n').count();
        if (n_count as f64 / seq.len() as f64) > 0.3 { continue; }
        filtered_seqs.push(seq.clone());
        filtered_tax.push(full_tax);
    }

    let config = TrainConfig {
        correlation_aware_features: true,
        record_kmers_fraction: 0.44,
        ..TrainConfig::default()
    };

    c.bench_function("learn_taxa/80seqs_corr_aware", |b| {
        b.iter(|| {
            black_box(learn_taxa(
                black_box(&filtered_seqs),
                black_box(&filtered_tax),
                black_box(&config),
                42,
                false,
            ).unwrap());
        });
    });
}
```

#### 2. Register in criterion_group
**File**: `benches/oxidtaxa_bench.rs`
**Changes**: Add `bench_learn_taxa_correlation_aware` to the `criterion_group!` macro at line 403.

### Success Criteria:

#### Automated Verification:
- [x] `cargo bench --bench oxidtaxa_bench -- learn_taxa/80seqs_corr_aware` runs successfully
- [x] `cargo test` passes (no regressions)

#### Manual Verification:
- [ ] Benchmark produces stable, reasonable timings
- [ ] Save baseline numbers for comparison after each subsequent phase

---

## Phase 1: Precompute Pearson Statistics

### Overview
Eliminate redundant computation in `pearson_abs` by precomputing `sum`, `sum_sq`, and `n_f` for each profile vector once, then using a specialized correlation function that only needs to compute `sum_ab` (the cross-term).

### Changes Required:

#### 1. Add precomputed statistics struct and helper
**File**: `src/training.rs`
**Changes**: Add a struct and a one-pass statistics precompute function above or near `pearson_abs` (line 609).

```rust
/// Precomputed statistics for a profile vector, used to avoid redundant
/// computation in the correlation-aware feature selection hot loop.
struct ProfileStats {
    sum: f64,
    sum_sq: f64,
    // Precomputed denominator component: sqrt(n * sum_sq - sum * sum)
    denom: f64,
}

impl ProfileStats {
    fn new(v: &[f64]) -> Self {
        let n_f = v.len() as f64;
        let sum: f64 = v.iter().sum();
        let sum_sq: f64 = v.iter().map(|x| x * x).sum();
        let denom = (n_f * sum_sq - sum * sum).sqrt();
        Self { sum, sum_sq, denom }
    }
}

/// Pearson correlation using precomputed statistics for both vectors.
/// Only computes the cross-product sum_ab, which is the only term that
/// changes between different (a, b) pairings.
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

Note: Keep the original `pearson_abs` function unchanged (it may be useful for testing parity).

#### 2. Precompute stats for all candidates at build time
**File**: `src/training.rs`
**Changes**: After `cand_data` is built and sorted (line 740), precompute `ProfileStats` for each candidate.

```rust
// After line 740 (cand_data.sort_by(...))
let cand_stats: Vec<ProfileStats> = cand_data.iter()
    .map(|(_, _, prof)| ProfileStats::new(prof))
    .collect();
```

#### 3. Cache stats for selected features as they're chosen
**File**: `src/training.rs`
**Changes**: Add a parallel `selected_stats` vec alongside `selected_indices` (line 744). Push to it when a feature is selected (line 775).

```rust
// At line 744, add:
let mut selected_stats: Vec<ProfileStats> = Vec::with_capacity(record_kmers);

// At line 775 (inside Some(ci) match arm), add:
selected_stats.push(ProfileStats::new(&cand_data[ci].2));
// (Or equivalently, clone from cand_stats[ci] -- but ProfileStats is Copy-sized)
```

#### 4. Replace `pearson_abs` calls with `pearson_with_stats`
**File**: `src/training.rs`
**Changes**: Replace lines 758-763 (the inner correlation loop).

Current code:
```rust
let mut max_corr: f64 = 0.0;
for &si in &selected_indices {
    let corr = pearson_abs(&cand_data[ci].2, &cand_data[si].2);
    if corr > max_corr { max_corr = corr; }
    if max_corr >= 1.0 { break; }
}
```

New code:
```rust
let mut max_corr: f64 = 0.0;
for (idx, &si) in selected_indices.iter().enumerate() {
    let corr = pearson_with_stats(
        &cand_data[ci].2, &cand_stats[ci],
        &cand_data[si].2, &selected_stats[idx],
    );
    if corr > max_corr { max_corr = corr; }
    if max_corr >= 1.0 { break; }
}
```

### Success Criteria:

#### Automated Verification:
- [x] `cargo test` passes -- all golden tests produce identical output
- [x] `cargo bench --bench oxidtaxa_bench -- learn_taxa/80seqs_corr_aware` shows improvement over Phase 0 baseline

#### Manual Verification:
- [ ] Compare Criterion before/after HTML reports
- [ ] Verify the `learn_taxa/80seqs` (non-correlation-aware) benchmark does not regress

---

## Phase 2: Enable `target-cpu=native`

### Overview
Tell the Rust compiler to generate instructions for the local CPU's full instruction set (AVX2, FMA, etc.) instead of the generic baseline. This improves auto-vectorization of all floating-point loops, including the `sum_ab` dot product in `pearson_with_stats`, and costs zero code changes.

### Changes Required:

#### 1. Create `.cargo/config.toml`
**File**: `.cargo/config.toml` (new file)
**Changes**: Create with `target-cpu=native` rustflag.

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

#### 2. Document in README
**File**: `README.md`
**Changes**: Add a note near the build instructions explaining that the build targets the local CPU. If distributable wheels are ever needed, this flag should be removed or overridden.

No code example needed -- a single sentence in the existing build section is sufficient.

### Success Criteria:

#### Automated Verification:
- [x] `cargo test` passes
- [x] `cargo bench --bench oxidtaxa_bench` runs (full suite, check for regressions)

#### Manual Verification:
- [ ] Criterion reports show improvement on floating-point-heavy benchmarks (`learn_taxa`, `vector_sum`, `parallel_match`)
- [ ] `maturin develop --release` still works correctly

---

## Phase 3: Batch Matrix Pearson Correlation

### Overview
Restructure the candidate profile vectors from an array-of-structs (`Vec<(usize, f64, Vec<f64>)>`) into a contiguous flat matrix. This enables computing `sum_ab` for a candidate against all selected features in a single pass over the candidate's profile vector, reducing memory indirection and enabling better LLVM auto-vectorization.

### Changes Required:

#### 1. Replace `cand_data` with a struct-of-arrays layout
**File**: `src/training.rs`
**Changes**: Replace the `cand_data: Vec<(usize, f64, Vec<f64>)>` construction at lines 723-740 with a flat matrix layout.

```rust
// Replace lines 723-740 with:

// Struct-of-arrays for cache-friendly access in the hot loop.
// profiles_flat is row-major: profiles_flat[ci * n_children .. (ci+1) * n_children]
// is the profile vector for candidate ci.
struct CandidatePool {
    kmer_indices: Vec<usize>,   // cand_data[ci].0
    entropies: Vec<f64>,        // cand_data[ci].1
    profiles_flat: Vec<f64>,    // row-major, n_cand x n_children
    stats: Vec<ProfileStats>,   // precomputed per candidate
    n_children: usize,
}

let mut kmer_indices = Vec::with_capacity(cand_set.len());
let mut entropies = Vec::with_capacity(cand_set.len());
let mut profiles_flat = Vec::with_capacity(cand_set.len() * n_children);

for &kmer_idx in &cand_set {
    let mut prof_vec = Vec::with_capacity(n_children);
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

// Sort all arrays by entropy descending (permutation sort)
let mut order: Vec<usize> = (0..kmer_indices.len()).collect();
order.sort_by(|&a, &b| entropies[b].partial_cmp(&entropies[a])
    .unwrap_or(std::cmp::Ordering::Equal));

let sorted_kmer_indices: Vec<usize> = order.iter().map(|&i| kmer_indices[i]).collect();
let sorted_entropies: Vec<f64> = order.iter().map(|&i| entropies[i]).collect();
let mut sorted_profiles_flat = vec![0.0f64; profiles_flat.len()];
for (new_idx, &old_idx) in order.iter().enumerate() {
    let src = old_idx * n_children;
    let dst = new_idx * n_children;
    sorted_profiles_flat[dst..dst + n_children]
        .copy_from_slice(&profiles_flat[src..src + n_children]);
}

let sorted_stats: Vec<ProfileStats> = (0..order.len()).map(|ci| {
    ProfileStats::new(&sorted_profiles_flat[ci * n_children..(ci + 1) * n_children])
}).collect();

let pool = CandidatePool {
    kmer_indices: sorted_kmer_indices,
    entropies: sorted_entropies,
    profiles_flat: sorted_profiles_flat,
    stats: sorted_stats,
    n_children,
};
let n_cand = pool.kmer_indices.len();
```

#### 2. Batch correlation: candidate vs all selected features
**File**: `src/training.rs`
**Changes**: Replace the inner loop (lines 758-763) with a batch computation. Store selected feature profiles in a contiguous buffer so the dot-product loop has good cache behavior.

```rust
// Selected features buffer: row-major, selected_count x n_children
let mut sel_profiles_flat: Vec<f64> = Vec::with_capacity(record_kmers * n_children);
let mut sel_stats: Vec<ProfileStats> = Vec::with_capacity(record_kmers);
let mut n_selected: usize = 0;

// In the greedy loop, replace the inner correlation computation:
for _ in 0..record_kmers {
    let mut best_ci = None;
    let mut best_gain = f64::NEG_INFINITY;

    for ci in 0..n_cand {
        if is_selected[ci] { continue; }
        let base_h = pool.entropies[ci];
        if base_h <= best_gain { break; }

        let cand_prof = &pool.profiles_flat[ci * n_children..(ci + 1) * n_children];
        let cand_st = &pool.stats[ci];

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
            result_set.insert(pool.kmer_indices[ci]);
            // Append to selected buffers
            let src = ci * n_children;
            sel_profiles_flat.extend_from_slice(
                &pool.profiles_flat[src..src + n_children]
            );
            sel_stats.push(pool.stats[ci].clone());
            n_selected += 1;
        }
        None => break,
    }
}
```

The key improvement: `sel_profiles_flat` is contiguous memory. When iterating `si in 0..n_selected`, each `sel_prof` slice is adjacent to the next, giving excellent cache locality. Combined with `target-cpu=native`, the `sum_ab` dot product in `pearson_with_stats` auto-vectorizes well over the contiguous slices.

#### 3. Clean up: remove `cand_data` tuple vec
**File**: `src/training.rs`
**Changes**: Remove the old `cand_data: Vec<(usize, f64, Vec<f64>)>` and `selected_indices: Vec<usize>` variables. The `CandidatePool` struct and `sel_profiles_flat`/`sel_stats` buffers replace them entirely.

### Success Criteria:

#### Automated Verification:
- [x] `cargo test` passes -- all golden tests produce identical output
- [x] `cargo bench --bench oxidtaxa_bench -- learn_taxa/80seqs_corr_aware` shows improvement over Phase 1

#### Manual Verification:
- [ ] Criterion comparison shows the batch layout outperforms the tuple-of-vecs layout
- [ ] No regression on `learn_taxa/80seqs` (non-correlation-aware path unaffected)

---

## Phase 4: Parallelize `create_tree`

### Overview
Process sibling subtrees in parallel using `rayon::scope`, giving near-linear speedup with `processors` for the tree-building phase. This is the highest-impact optimization because the correlation-aware feature selection loop runs inside `create_tree` on a single thread today.

### Design Decision: Return-and-Merge vs Unsafe Disjoint Write

Two viable approaches:

| Approach | Pros | Cons |
|----------|------|------|
| **Return-and-merge**: return `Vec<(usize, DecisionNode)>` from each subtree, merge at caller | No unsafe. Idiomatic Rust. Easy to reason about. | Temporary Vec allocations for intermediate results. Extra copy when writing final vec. |
| **Unsafe disjoint write**: pass raw pointer + `rayon::scope` | Zero allocation overhead. In-place writes. | Requires `unsafe` block. Must manually argue safety. |

**Choice: Return-and-merge.** The temporary allocations are negligible compared to the feature selection cost. Safety and maintainability are more important. The `DecisionNode` structs are small (a Vec<i32> + Vec<Vec<f64>>), and we're moving them, not copying.

### Changes Required:

#### 1. Change `create_tree` signature and return type
**File**: `src/training.rs`
**Changes**: Modify `create_tree` to return collected decision nodes instead of mutating a shared vec.

Current signature (line 628):
```rust
fn create_tree(
    node: usize,
    children: &[Vec<usize>],
    sequences: &[Option<Vec<usize>>],
    kmers: &[Vec<i32>],
    n_kmers: usize,
    config: &TrainConfig,
    decision_kmers: &mut Vec<Option<DecisionNode>>,
) -> (SparseProfile, usize)
```

New signature:
```rust
fn create_tree(
    node: usize,
    children: &[Vec<usize>],
    sequences: &[Option<Vec<usize>>],
    kmers: &[Vec<i32>],
    n_kmers: usize,
    config: &TrainConfig,
) -> (SparseProfile, usize, Vec<(usize, DecisionNode)>)
```

The third return element collects all `(node_index, DecisionNode)` pairs from this subtree.

#### 2. Parallelize the child loop with `rayon::scope`
**File**: `src/training.rs`
**Changes**: Replace the sequential for loop at lines 644-649 with parallel execution.

Current code:
```rust
for &child in child_nodes {
    let (profile, desc) =
        create_tree(child, children, sequences, kmers, n_kmers, config, decision_kmers);
    profiles.push(profile);
    descendants.push(desc);
}
```

New code using `rayon::join` for 2 children, and a parallel collect pattern for >2:
```rust
// Collect child results in parallel
let child_results: Vec<(SparseProfile, usize, Vec<(usize, DecisionNode)>)> =
    if n_children == 2 {
        // rayon::join is optimal for exactly 2 subtrees
        let (r0, r1) = rayon::join(
            || create_tree(child_nodes[0], children, sequences, kmers, n_kmers, config),
            || create_tree(child_nodes[1], children, sequences, kmers, n_children, config),
        );
        vec![r0, r1]
    } else {
        // For >2 children, use scope with spawned tasks
        let mut results: Vec<Option<(SparseProfile, usize, Vec<(usize, DecisionNode)>)>> =
            (0..n_children).map(|_| None).collect();
        rayon::scope(|s| {
            for (i, result_slot) in results.iter_mut().enumerate() {
                let child = child_nodes[i];
                s.spawn(move |_| {
                    *result_slot = Some(create_tree(
                        child, children, sequences, kmers, n_kmers, config,
                    ));
                });
            }
        });
        results.into_iter().map(|r| r.unwrap()).collect()
    };

let mut profiles: Vec<SparseProfile> = Vec::with_capacity(n_children);
let mut descendants: Vec<usize> = Vec::with_capacity(n_children);
let mut collected_nodes: Vec<(usize, DecisionNode)> = Vec::new();

for (profile, desc, nodes) in child_results {
    profiles.push(profile);
    descendants.push(desc);
    collected_nodes.extend(nodes);
}
```

#### 3. Collect the current node's DecisionNode into the return vec
**File**: `src/training.rs`
**Changes**: Replace the direct `decision_kmers[node] = Some(...)` write at line 837 with appending to `collected_nodes` and returning it.

Current code (line 837-840):
```rust
decision_kmers[node] = Some(DecisionNode {
    keep: keep_indices,
    profiles: selected_profiles,
});

(q, total_desc)
```

New code:
```rust
collected_nodes.push((node, DecisionNode {
    keep: keep_indices,
    profiles: selected_profiles,
}));

(q, total_desc, collected_nodes)
```

For leaf nodes (line 862), return an empty vec:
```rust
(profile, 1, Vec::new())
```

#### 4. Update the call site to scatter results into the final vec
**File**: `src/training.rs`
**Changes**: At the call site (lines 266-275), collect the returned nodes and write them into `decision_kmers`.

```rust
let mut decision_kmers: Vec<Option<DecisionNode>> = vec![None; taxonomy.len()];
let (_root_profile, _root_desc, nodes) = create_tree(
    0,
    &children,
    &sequences_per_node,
    &kmers,
    n_kmers,
    config,
);
for (idx, dk) in nodes {
    decision_kmers[idx] = Some(dk);
}
```

### Important Notes

- The `rayon::scope` approach works because all `&`-borrowed data (`children`, `sequences`, `kmers`, `config`) has lifetime that outlives the scope. The `Send` bound is satisfied because `SparseProfile`, `DecisionNode`, and all component types are `Send`.
- Work stealing ensures load balancing: if one subtree (e.g., a large phylum) takes much longer, idle threads steal work from its children's subtrees.
- At the root level with only 2-5 domain children, parallelism is limited. But the recursive spawning means that by level 3-4 (order/family), there are 50-200+ concurrent tasks available, fully utilizing the thread pool.
- This runs inside the existing `pool.install(|| ...)` block (line 39), so it respects the `processors` config.

### Success Criteria:

#### Automated Verification:
- [x] `cargo test` passes -- all golden tests produce identical output
- [x] `cargo bench --bench oxidtaxa_bench` shows no regression on any benchmark

#### Manual Verification:
- [ ] Test with `processors=1` to verify single-threaded behavior matches pre-change output exactly
- [ ] Test with `processors=4` (or higher) on the correlation-aware path and observe wall-clock speedup
- [ ] Benchmark `learn_taxa/80seqs_corr_aware` with `processors: 1` vs `processors: 4` to quantify the parallel speedup

---

## Testing Strategy

### Unit Tests:
- Add a unit test that trains with `correlation_aware_features: true` on the 80-sequence test data and verifies the output matches a golden reference. This catches any accidental change to feature selection order or values.
- Add a test that runs `create_tree` with `processors: 1` and `processors: 4` and asserts the resulting `decision_kmers` are identical (the parallel version must produce the same result regardless of thread count, since sibling subtrees are independent).

### Integration Tests:
- All 112 existing golden tests must pass unchanged after every phase.
- The `learn_taxa/80seqs` benchmark (default config) must not regress.

### Manual Testing Steps:
1. Run `cargo bench --bench oxidtaxa_bench` before Phase 0 to capture full baseline
2. After each phase, re-run and compare Criterion HTML reports
3. After Phase 4, run a real-world test with 177k sequences, `k=6`, `record_kmers_fraction=0.44`, `correlation_aware_features=true`, `processors=8` and measure wall-clock improvement vs. pre-optimization single-threaded time

## Performance Considerations

### Expected Impact by Phase:

| Phase | Mechanism | Expected Speedup | Cumulative |
|-------|-----------|-------------------|------------|
| 0 | Benchmark only | (baseline) | 1x |
| 1 | Precompute stats | ~1.5-2x on correlation loop | 1.5-2x |
| 2 | target-cpu=native | ~1.3-2x on FP loops | 2-4x |
| 3 | Batch matrix layout | ~1.5-2x (cache + auto-vectorization) | 3-8x |
| 4 | Parallel create_tree | ~Nx for N processors | 3N-8Nx |

With `processors=8`, the combined effect could be 24-64x faster on the correlation-aware path. The exact numbers depend on tree shape, cache effects, and how well LLVM vectorizes the batch layout.

### Memory Impact:
- Phases 1-3: Negligible. The `ProfileStats` struct is 24 bytes per candidate. The flat matrix replaces the existing Vec-of-Vecs with a single contiguous allocation of the same total size.
- Phase 4: Temporary `Vec<(usize, DecisionNode)>` allocations during tree construction. Total size is bounded by the number of internal nodes (typically 30K-80K for a 177K-sequence dataset). Each `DecisionNode` is already allocated; we're just collecting them differently. Net memory increase is negligible.

## References

- Original analysis: this conversation's research phase
- Hot path: `src/training.rs:609-623` (pearson_abs), `src/training.rs:747-780` (greedy loop), `src/training.rs:628-864` (create_tree)
- Config: `src/types.rs:111-167` (TrainConfig)
- Benchmarks: `benches/oxidtaxa_bench.rs`
- Build config: `Cargo.toml:26-28` (release profile), `pyproject.toml` (maturin)
