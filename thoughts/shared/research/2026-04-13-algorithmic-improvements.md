---
date: 2026-04-13T15:00:00-07:00
researcher: Claude
git_commit: 5576821
branch: main
repository: idtaxa-optim
topic: "Algorithmic and computational improvements in oxidtaxa training and classification"
tags: [research, optimization, training, classification, kmer, matching]
status: complete
last_updated: 2026-04-13
last_updated_by: Claude
---

# Research: Algorithmic Improvements in oxidtaxa Training and Classification

**Date**: 2026-04-13
**Researcher**: Claude
**Git Commit**: 5576821
**Branch**: main
**Repository**: idtaxa-optim

## Research Question
What algorithmic improvements — both logical and computational — can be identified in either the training or testing (classification) phases of oxidtaxa?

## Summary

The oxidtaxa codebase is a well-structured Rust port of R's IDTAXA classifier. The analysis below identifies 14 improvement opportunities spanning complexity reductions, data structure upgrades, and constant-factor optimizations. The highest-impact items are: (1) rolling hash for k-mer enumeration reducing per-position cost from O(k) to O(1), (2) bitset k-mer representation enabling O(1) membership tests, (3) replacing a linear scan with a HashMap in `cross_index` construction, and (4) counting sort for the bootstrap sampling grouping step.

---

## Detailed Findings

### A. Training Phase (`learn_taxa` — training.rs)

#### A1. Rolling hash for k-mer enumeration — O(k) → O(1) per position
**File**: `src/kmer.rs:335-354`
**Severity**: High (hot inner loop, called for every position in every sequence)

The `enumerate_single` function computes each k-mer hash from scratch:
```rust
for j in (word_size - 1)..len {
    let mut sum = bases[0] as i32 * pwv[0];
    for k in 1..word_size {
        sum += bases[k] as i32 * pwv[k];
    }
}
```
Since `pwv[i] = 4^i`, the hash is `base[0]*1 + base[1]*4 + ... + base[k-1]*4^(k-1)`. When sliding right by one position:
```
new_sum = (old_sum - base[0]) / 4 + new_base * 4^(k-1)
```
This reduces per-position work from O(k) multiplications/additions to O(1). Ambiguous bases would invalidate the rolling sum and require a full recomputation, but that's the uncommon case. For k=8 (default), this is an 8x reduction in arithmetic per position.

Note: This does NOT apply to `enumerate_single_spaced` (spaced seeds have gaps that break the rolling property), though partial incremental updates could be designed for specific patterns.

#### A2. HashMap for cross_index construction — O(n·m) → O(n)
**File**: `src/training.rs:145-154`

```rust
let cross_index: Vec<usize> = classes.iter()
    .map(|c| all_taxa.iter().position(|t| t == c).map(|p| p + 1).unwrap_or(0))
    .collect();
```
This does a linear scan of `all_taxa` for every sequence. With n sequences and m unique taxa, this is O(n·m). Building a `HashMap<&str, usize>` from `all_taxa` first reduces this to O(n) with O(m) setup.

#### A3. Children construction uses string prefix matching in nested loops
**File**: `src/training.rs:182-197`

```rust
let w: Vec<usize> = levs[j][starts[j]..]
    .iter()
    .filter(|&&idx| taxonomy[idx].starts_with(&taxonomy[i]))
    .copied()
    .collect();
children[i] = w;
```
For each taxonomy node, this scans candidate children and checks `starts_with` on taxonomy strings. In the worst case (flat taxonomy), this is O(T² · string_len). A trie or sorted-prefix approach could reduce this to O(T · string_len).

#### A4. k-way merge in `merge_sparse_profiles` — linear scan vs min-heap
**File**: `src/training.rs:521-549`

The current k-way merge uses a linear scan across all profile cursors to find the minimum key each iteration:
```rust
for (i, profile) in profiles.iter().enumerate() {
    if cursors[i] < profile.len() && profile[cursors[i]].0 < min_key {
        min_key = profile[cursors[i]].0;
    }
}
```
For k children and N total entries, this is O(N·k). A BinaryHeap-based k-way merge would be O(N·log k). At typical branching factors (2-20 children) the constant factor difference may wash out, but for wide nodes (phylum-level with many classes) this matters.

#### A5. HashMap overhead in `selected_profiles` extraction
**File**: `src/training.rs:658-664`

```rust
let selected_profiles: Vec<Vec<f64>> = profiles.iter()
    .map(|p| {
        let p_map: HashMap<usize, f64> = p.iter().map(|&(k, v)| (k, v)).collect();
        keep_vec.iter().map(|&k| *p_map.get(&k).unwrap_or(&0.0)).collect()
    })
    .collect();
```
Both `profiles[i]` and `keep_vec` are sorted by k-mer index. A merge-join (two-pointer) would be O(|profile| + |keep|) without HashMap allocation overhead.

#### A6. Fraction vector cloned every iteration
**File**: `src/training.rs:295`

```rust
let fraction_snapshot = fraction.clone();
```
This clones the entire `Vec<Option<f64>>` (one entry per taxonomy node) each iteration. For large taxonomies (100K+ nodes), this is non-trivial. Alternative: use a generation counter per node and compare against the iteration number, or use a `Vec<(f64, u32)>` where the u32 is the last-modified iteration. Reads that see a stale generation use the value directly (it hasn't changed).

#### A7. Sequences-per-node string allocation in prefix walk
**File**: `src/training.rs:229-240`

```rust
for n in 1..=parts.len() {
    let prefix: String = parts[..n].iter().map(|s| format!("{};", s)).collect();
    if let Some(&tax_idx) = et_to_idx.get(prefix.as_str()) { ... }
}
```
Each iteration of the inner loop allocates a new String for the prefix. Since taxonomy strings are structured, you could walk `et_to_idx` incrementally: start with `parts[0] + ";"`, then append `parts[1] + ";"` etc., reusing a single `String` buffer.

---

### B. Classification Phase (`id_taxa` / `classify_one_pass` — classify.rs)

#### B1. Counting sort for bootstrap sampling — O(sb·log(sb)) → O(sb + 4^k)
**File**: `src/classify.rs:273-290`

```rust
let mut sort_idx: Vec<u32> = (0..sb as u32).collect();
sort_idx.sort_unstable_by_key(|&i| sampling[i as usize]);
```
This sorts sb indices (typically 1000-10000) by k-mer value. Since k-mer values are integers in [1, 4^k] (e.g., [1, 65536] for k=8), a counting sort or radix sort would reduce this from O(sb·log(sb)) to O(sb + 4^k). For k=8 with sb=5000, this replaces ~60K comparisons with a single 64KB pass.

#### B2. Bitset k-mer representation for O(1) membership testing
**File**: `src/matching.rs:1-19` (int_match), `src/matching.rs:76-160` (parallel_match)

Currently, k-mers are stored as sorted `Vec<i32>` and membership testing uses merge-join (O(|a|+|b|)). For k=8, all k-mer values fit in [0, 65535]. A bitset (8KB per sequence) would enable:
- O(1) membership test (replacing `int_match`)
- O(4^k / 64) set intersection via word-level AND
- Slightly higher memory per sequence (~8KB vs ~600B for typical ~150 unique k-mers), but dramatically faster operations

The bitset particularly benefits `parallel_match_inverted` where `keep_map` (line 177) is currently a HashMap. A sorted `keep` array with binary search or a bitset would eliminate hash overhead.

#### B3. Inverted index `keep_map` — HashMap → sorted binary search or bitset
**File**: `src/matching.rs:177`

```rust
let keep_map: HashMap<u32, usize> = keep.iter().enumerate()
    .map(|(pos, &idx)| (idx as u32, pos)).collect();
```
This builds a HashMap every call. Since `keep` is typically sorted (sequences within a taxonomy subtree), binary search on the sorted array would avoid allocation entirely. Even better: if the caller guarantees `keep` is sorted, create a dense mapping array indexed by sequence ID (sparse but O(1) lookup).

#### B4. `vector_sum` — SIMD opportunity
**File**: `src/matching.rs:31-57`

The inner loop accumulates weighted matches:
```rust
for k in 0..block_size {
    let idx = sampling[i * block_size + k];
    max_weight += weights[idx];
    if matches[idx] { cur_weight += weights[idx]; }
}
```
This is a conditional accumulation pattern ideal for SIMD. Using `std::simd` (nightly) or manual AVX2 intrinsics, the `matches` boolean could be converted to a mask for branchless weighted accumulation. For block_size=20 and block_count=100, this loop executes ~2000 iterations per call.

#### B5. Pre-allocated RNG buffers
**File**: `src/classify.rs:268`

```rust
let sampling: Vec<i32> = rng.sample_replace(my_kmers, s * b);
```
`sample_replace` (rng.rs:176) allocates a `Vec<usize>` of indices, then maps into a `Vec<T>`. In the classification hot path, pre-allocating a reusable buffer and using `sample_int_replace_into` would avoid two allocations per sequence.

#### B6. Davg computation duplicates work
**File**: `src/classify.rs:350-357`

```rust
let davg = {
    let mut row_sums = vec![0.0f64; b];
    for (idx, &sk) in sampling.iter().enumerate() {
        let w = if sk > 0 && (sk as usize) <= counts.len() { counts[(sk - 1) as usize] } else { 0.0 };
        row_sums[idx % b] += w;
    }
    row_sums.iter().sum::<f64>() / b as f64
};
```
This iterates over all s*b sampling elements to compute IDF-weighted totals. But the `u_weights` vector (line 293) already contains IDF weights for unique k-mers, and `positions`/`ranges` map these to bootstrap replicates. The sum could be computed as `sum(u_weights[i] * count_of_positions[i])` in O(|u_sampling|) instead of O(s*b).

#### B7. Group-max computation allocates sort indices
**File**: `src/classify.rs:324-325`

```rust
let mut order: Vec<usize> = (0..lookup.len()).collect();
order.sort_unstable_by_key(|&i| lookup[i]);
```
Allocates and sorts an index vector. Since `lookup` values (cross_index entries) are bounded integers, a counting sort or bucket approach would be more efficient. Alternatively, if `keep` is already grouped by taxonomy node (which it often is, since `sequences_per_node` stores them contiguously), the sort could be skipped.

---

### C. Shared / Cross-Cutting

#### C1. K-mer 1-indexing overhead
**File**: `src/training.rs:88`, `src/classify.rs:114`

K-mers are converted from 0-indexed to 1-indexed (`x + 1`) to match R's convention, then every comparison subtracts 1 back (`km - 1`). This adds an arithmetic operation on every k-mer access throughout the codebase. Switching to 0-indexed throughout and adjusting only at the serialization boundary would eliminate thousands of +1/-1 operations per classification call.

#### C2. String-based taxonomy representation
**File**: Throughout `training.rs` and `classify.rs`

Taxonomy paths are stored as `Vec<String>` with prefix matching via `starts_with`. An integer-ID representation (each taxon gets a unique u32 ID, parent/children encoded as ID arrays) would eliminate all string allocation and comparison in the hot paths. The string representation is already partially shadowed by the integer `parents`/`children` arrays — the remaining string operations (e.g., `end_taxonomy`, `classes`) could be replaced with ID lookups.

---

## Impact Assessment

| ID  | Improvement | Phase | Complexity Change | Estimated Impact |
|-----|------------|-------|-------------------|-----------------|
| A1  | Rolling hash k-mer enumeration | Train+Classify | O(k)→O(1) per pos | High |
| A2  | HashMap for cross_index | Train | O(nm)→O(n) | Medium |
| A3  | Trie for children construction | Train | O(T²·L)→O(T·L) | Medium |
| A4  | Min-heap k-way merge | Train | O(Nk)→O(N·log k) | Low-Medium |
| A5  | Merge-join for profile extraction | Train | O(n) but less alloc | Low |
| A6  | Avoid fraction clone | Train | Constant factor | Low |
| A7  | Reuse string buffer in prefix walk | Train | Constant factor | Low |
| B1  | Counting sort for sampling | Classify | O(sb·log sb)→O(sb+4^k) | Medium |
| B2  | Bitset k-mer representation | Classify | O(n+m)→O(1) per test | High |
| B3  | Binary search for keep_map | Classify | Hash→O(log n) | Medium |
| B4  | SIMD for vector_sum | Classify | Constant factor | Medium |
| B5  | Pre-allocated RNG buffers | Classify | Constant factor | Low |
| B6  | Optimized davg computation | Classify | O(sb)→O(|unique|) | Low |
| B7  | Counting sort for group-max | Classify | O(n log n)→O(n) | Low |
| C1  | 0-indexed k-mers throughout | Both | Constant factor | Low-Medium |
| C2  | Integer taxonomy IDs | Both | Eliminates string ops | Medium |

## Recommendations (Priority Order)

1. **A1 + B2**: Rolling hash + bitset k-mers — together these transform the two hottest loops (enumeration and matching)
2. **B1 + B3**: Counting sort + binary search — low-risk, high-certainty wins in classification
3. **A2 + C2**: HashMap cross_index + integer taxonomy — eliminates string overhead in training
4. **B4**: SIMD vector_sum — requires `unsafe` or nightly features but large constant-factor win
5. **A3-A7, B5-B7, C1**: Smaller wins, worth batching together

## Code References
- `src/kmer.rs:335-354` — k-mer enumeration inner loop (A1)
- `src/training.rs:145-154` — cross_index linear scan (A2)
- `src/training.rs:182-197` — children prefix matching (A3)
- `src/training.rs:521-549` — k-way merge (A4)
- `src/training.rs:658-664` — profile extraction (A5)
- `src/training.rs:295` — fraction clone (A6)
- `src/training.rs:229-240` — prefix string allocation (A7)
- `src/classify.rs:273-290` — sampling sort (B1)
- `src/matching.rs:1-19` — int_match merge-join (B2)
- `src/matching.rs:177` — keep_map HashMap (B3)
- `src/matching.rs:31-57` — vector_sum inner loop (B4)
- `src/classify.rs:268` — sampling allocation (B5)
- `src/classify.rs:350-357` — davg computation (B6)
- `src/classify.rs:324-325` — group-max sort (B7)

## Open Questions
- What are the typical dataset sizes in production? The relative impact of these optimizations varies significantly between 1K-reference and 100K-reference models.
- Is R-compatibility for the RNG still required? Switching to a faster PRNG (xoshiro256**) would speed up all sampling operations but break golden-test reproducibility.
- For B2 (bitset), memory tradeoff: ~8KB per sequence for k=8 vs ~600B average for sorted vec. At 100K sequences, that's 800MB of bitsets. A compressed bitset (roaring bitmap) could get the best of both worlds.
