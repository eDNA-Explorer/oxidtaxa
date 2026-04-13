# Computational Optimizations Implementation Plan

## Overview

Implement 6 computational optimizations that produce identical output to current behavior but run faster. Unlike the algorithmic improvements (separate plan), these don't need config flags — they're transparent speedups verified by the existing golden tests and baseline test.

## Current State Analysis

The codebase already has:
- Golden tests for every stage (k-mer enumeration, matching, training, classification) in `tests/`
- A real-data baseline test (`tests/test_baseline_1k.rs`) requiring bit-identical output in deterministic mode
- Criterion benchmarks (`benches/oxidtaxa_bench.rs`) covering all hot paths

These serve as both correctness guards and performance baselines. Every optimization must pass all existing tests without modification.

### Hot Path Profile (from algorithm structure):

1. **K-mer enumeration** (`kmer.rs`) — O(n_sequences × seq_length × k). Called during both training and classification.
2. **Bootstrap sampling + sorting** (`classify.rs:268-290`) — O(s×b × log(s×b)) per query sequence. Called for every query.
3. **K-mer matching** (`matching.rs`) — O(n_keep × n_query_kmers) per query. The innermost hot loop.
4. **Inverted index lookup** (`matching.rs:165-209`) — O(n_query_kmers × avg_posting_list) per query. Alternative to merge-join matching.
5. **Tree descent voting** (`classify.rs:190-248`, `training.rs:310-376`) — O(n_subtrees × b) per node per sequence.
6. **IDF accumulation** (`training.rs:454-478`) — O(n_sequences × avg_kmers_per_seq). Once during training.

## Desired End State

All optimizations are in place. Every existing test passes unchanged. Criterion benchmarks show measurable improvement on affected hot paths.

### Verification:
- `cargo test` — all golden tests and baseline test pass (bit-identical output)
- `cargo bench` — before/after comparison for each optimization

## What We're NOT Doing

- Algorithmic changes that alter output (those are in the algorithmic improvements plan)
- SIMD intrinsics (requires nightly or `unsafe`, consider as future work)
- Changing the RNG or breaking deterministic mode
- Changing serialization format (bincode) for TrainingSet

## Implementation Approach

Each phase is independent and can be landed separately. The ordering is by impact-to-effort ratio. Each phase follows the pattern:

1. Implement the optimization
2. Run `cargo test` — must pass unchanged
3. Run `cargo bench` — measure improvement
4. Record before/after numbers in commit message

---

## Phase 1: Rolling Hash for K-mer Enumeration

### Overview
Replace the O(k) per-position k-mer computation with an O(1) rolling hash update. For default k=8, this is an 8x reduction in arithmetic per position in the hottest inner loop.

**File**: `src/kmer.rs:335-354`

### Current code:
```rust
for j in (word_size - 1)..len {
    bases[word_size - 1] = base_to_index(seq[j]);

    let mut sum = bases[0] as i32 * pwv[0];
    let mut ambiguous = bases[0] < 0;
    for k in 1..word_size {
        sum += bases[k] as i32 * pwv[k];
        if bases[k] < 0 { ambiguous = true; }
    }

    let pos = j + 1 - word_size;
    result[pos] = if ambiguous { NA_INTEGER } else { sum };

    // Shift left
    for k in 0..(word_size - 1) {
        bases[k] = bases[k + 1];
    }
}
```

### New code:
```rust
// Compute first k-mer fully
let mut sum: i32 = 0;
let mut ambig_count: usize = 0;
for k in 0..word_size {
    let b = base_to_index(seq[k]);
    bases[k] = b;
    if b < 0 { ambig_count += 1; } else { sum += b as i32 * pwv[k]; }
}
result[0] = if ambig_count > 0 { NA_INTEGER } else { sum };

// Rolling update for subsequent positions
// Since pwv[i] = 4^i (fast_moving_side=true), we have:
//   sum = bases[0]*1 + bases[1]*4 + ... + bases[k-1]*4^(k-1)
// After shifting left by one:
//   new_sum = bases[1]*1 + bases[2]*4 + ... + new_base*4^(k-1)
//           = (sum - bases[0]) / 4 + new_base * 4^(k-1)
let divisor = 4i32;
let top_weight = pwv[word_size - 1]; // = 4^(k-1)

for j in word_size..len {
    let outgoing = bases[0];
    let incoming = base_to_index(seq[j]);

    // Update ambiguity count
    if outgoing < 0 { ambig_count -= 1; }
    if incoming < 0 { ambig_count += 1; }

    // Rolling hash update (only valid when no ambiguous bases involved)
    if ambig_count == 0 && outgoing >= 0 && incoming >= 0 {
        sum = (sum - outgoing as i32) / divisor + incoming as i32 * top_weight;
    } else if ambig_count == 0 {
        // Just became unambiguous — recompute from scratch
        sum = 0;
        for k in 0..word_size - 1 {
            sum += bases[k + 1] as i32 * pwv[k]; // shifted positions
        }
        sum += incoming as i32 * top_weight;
    }

    // Shift window
    for k in 0..(word_size - 1) {
        bases[k] = bases[k + 1];
    }
    bases[word_size - 1] = incoming;

    let pos = j + 1 - word_size;
    result[pos] = if ambig_count > 0 { NA_INTEGER } else { sum };
}
```

**Important**: The division `sum / 4` is exact integer division because `sum` was constructed from terms that are all multiples of 4 (except `bases[0]*pwv[0]` where `pwv[0]=1`). After subtracting `bases[0]*1` (which is 0-3), the remainder is divisible by 4 since all other terms are `bases[k]*4^k`. So `(sum - outgoing) / 4` is always exact.

**Note**: This optimization does NOT apply to `enumerate_single_spaced` — spaced seeds have gaps that break the rolling property. The contiguous path is the common case.

### Verification:
- All `test_kmer_*` golden tests pass (bit-identical output)
- `cargo bench -- enumerate` shows improvement
- Expected: ~4-7x speedup on `enumerate_single` for k=8

---

## Phase 2: HashMap for cross_index Construction

### Overview
Replace the O(n×m) linear scan with an O(n) HashMap lookup when building `cross_index` during training.

**File**: `src/training.rs:145-154`

### Current code:
```rust
let cross_index: Vec<usize> = classes.iter()
    .map(|c| all_taxa.iter().position(|t| t == c).map(|p| p + 1).unwrap_or(0))
    .collect();
```

### New code:
```rust
let taxa_to_idx: HashMap<&str, usize> = all_taxa.iter()
    .enumerate()
    .map(|(i, t)| (t.as_str(), i + 1))
    .collect();
let cross_index: Vec<usize> = classes.iter()
    .map(|c| *taxa_to_idx.get(c.as_str()).unwrap_or(&0))
    .collect();
```

### Verification:
- All `test_training_*` golden tests pass
- `cargo bench -- learn_taxa` shows improvement on larger datasets
- Expected: Negligible on small datasets, significant on 10K+ sequences

---

## Phase 3: Binary Search for Inverted Index keep_map

### Overview
Replace the per-call HashMap construction in `parallel_match_inverted` with binary search on the (already sorted) `keep` array.

**File**: `src/matching.rs:165-209`

### Current code:
```rust
let keep_map: HashMap<u32, usize> = keep.iter().enumerate()
    .map(|(pos, &idx)| (idx as u32, pos)).collect();

// ... later:
if let Some(&keep_pos) = keep_map.get(&seq_idx) {
```

### New code:
```rust
// keep is sorted (sequences within a taxonomy subtree are stored in order)
// Use binary search for O(log n) lookup without HashMap allocation

// ... later:
if let Ok(pos) = keep.binary_search(&(seq_idx as usize)) {
    let base = pos * block_count;
    // ...
}
```

Wait — `keep` contains sequence indices but may have duplicates or may not be sorted. Check the call sites.

Looking at `classify.rs:252-256`:
```rust
let mut keep: Vec<usize> = Vec::new();
for &wi in &w_indices {
    if wi < subtrees.len() {
        if let Some(ref sq) = sequences[subtrees[wi]] { keep.extend(sq); }
    }
}
```

The `sequences` per node are built during training in order, so each chunk is sorted, but concatenating multiple children's sequences may not be globally sorted. However, we need the `keep_pos` (position in `keep`) not just membership.

**Revised approach**: Sort `keep` and use binary search, mapping back to original positions:

Actually, the simpler fix: `keep` is typically small enough that a sorted Vec with binary search is fine, but we need the position, not just membership. The HashMap maps `seq_idx → position_in_keep`.

**Better approach**: Build a dense lookup array when the max index is reasonable:

```rust
let max_idx = keep.iter().copied().max().unwrap_or(0);
if max_idx < 100_000 {
    // Dense lookup: O(1) per query, O(max_idx) setup
    let mut dense_map = vec![u32::MAX; max_idx + 1];
    for (pos, &idx) in keep.iter().enumerate() {
        dense_map[idx] = pos as u32;
    }
    // ... use dense_map[seq_idx as usize] instead of keep_map.get()
} else {
    // Fall back to HashMap for very large indices
    // (current code)
}
```

For typical IDTAXA usage with <100K training sequences, the dense array is ~400KB and gives O(1) lookup with no hashing overhead.

### Verification:
- All `test_classify_*` and `test_baseline_1k` tests pass
- `cargo bench -- id_taxa` shows improvement
- Expected: 10-30% speedup on `parallel_match_inverted` due to eliminating HashMap allocation per call

---

## Phase 4: Counting Sort for Bootstrap Sampling

### Overview
Replace the O(sb × log(sb)) comparison sort with an O(sb + 4^k) counting sort when grouping bootstrap samples by k-mer value.

**File**: `src/classify.rs:273-290`

### Current code:
```rust
let mut sort_idx: Vec<u32> = (0..sb as u32).collect();
sort_idx.sort_unstable_by_key(|&i| sampling[i as usize]);

// Build u_sampling, positions, ranges in a single pass over sorted indices
let mut u_sampling: Vec<i32> = Vec::new();
let mut positions: Vec<usize> = Vec::with_capacity(sb);
let mut ranges: Vec<usize> = vec![0];
{
    let mut i = 0;
    while i < sb {
        let kmer = sampling[sort_idx[i] as usize];
        u_sampling.push(kmer);
        while i < sb && sampling[sort_idx[i] as usize] == kmer {
            positions.push(sort_idx[i] as usize % b);
            i += 1;
        }
        ranges.push(positions.len());
    }
}
```

### New code:
```rust
// Counting sort: k-mer values are bounded integers in [1, 4^k]
// For k=8, this is a 65536-entry count array — fits in L1 cache
let n_possible = 4usize.pow(ts.k as u32);

// Pass 1: count occurrences of each k-mer value
let mut counts = vec![0u32; n_possible + 2]; // +2 for 0 and 1-indexed values
for &km in &sampling {
    if km > 0 { counts[km as usize] += 1; }
}

// Build u_sampling, positions, ranges directly from counts
let mut u_sampling: Vec<i32> = Vec::new();
let mut positions: Vec<usize> = Vec::with_capacity(sb);
let mut ranges: Vec<usize> = vec![0];

// Pass 2: for each k-mer value that appeared, collect its bootstrap positions
// We need to know which positions in `sampling` had each value
// Build a per-kmer position list
let mut kmer_positions: Vec<Vec<usize>> = vec![Vec::new(); n_possible + 2];
for (idx, &km) in sampling.iter().enumerate() {
    if km > 0 {
        kmer_positions[km as usize].push(idx % b);
    }
}

for km in 1..=(n_possible as i32) {
    if counts[km as usize] > 0 {
        u_sampling.push(km);
        positions.extend(&kmer_positions[km as usize]);
        ranges.push(positions.len());
    }
}
```

**Memory note**: `kmer_positions` allocates a Vec per possible k-mer value (65K Vecs for k=8). To avoid this, use a two-pass approach:

```rust
// Pass 1: count
let mut counts = vec![0u32; n_possible + 2];
for &km in &sampling {
    if km > 0 { counts[km as usize] += 1; }
}

// Prefix sum for offsets
let mut offsets = vec![0usize; n_possible + 2];
for i in 1..offsets.len() {
    offsets[i] = offsets[i - 1] + counts[i - 1] as usize;
}

// Pass 2: place positions into sorted order
let mut sorted_positions = vec![0usize; sb];
let mut cursors = offsets.clone();
for (idx, &km) in sampling.iter().enumerate() {
    if km > 0 {
        sorted_positions[cursors[km as usize]] = idx % b;
        cursors[km as usize] += 1;
    }
}

// Pass 3: build u_sampling and ranges from counts
let mut u_sampling: Vec<i32> = Vec::new();
let mut ranges: Vec<usize> = vec![0];
let mut pos_cursor = 0usize;
for km in 1..=(n_possible as i32) {
    let c = counts[km as usize] as usize;
    if c > 0 {
        u_sampling.push(km);
        pos_cursor += c;
        ranges.push(pos_cursor);
    }
}
let positions = sorted_positions; // already in the right order
```

This uses O(4^k) auxiliary space (256KB for k=8 — the counts + offsets + cursors arrays) and O(sb) for the output. No per-kmer Vec allocation.

### Verification:
- All `test_classify_*` and `test_baseline_1k` tests pass (bit-identical output — same u_sampling, positions, ranges)
- `cargo bench -- id_taxa` shows improvement
- Expected: 20-40% speedup on the sampling-sort step for typical sb=5000

---

## Phase 5: Eliminate 1-Indexing Overhead

### Overview
K-mers are converted to 1-indexed after enumeration (`x + 1`) to match R's convention, then every consumer subtracts 1 back (`km - 1`). Switching to 0-indexed throughout eliminates thousands of +1/-1 operations per classification call.

### Scope of changes:

**K-mer producers** (change `+ 1` to nothing):
- `src/training.rs:90` — `x + 1` in k-mer post-processing
- `src/classify.rs:114` — `x + 1` in test k-mer post-processing
- `src/classify.rs:144` — `x + 1` in reverse k-mer post-processing

**K-mer consumers** (remove `- 1` adjustments):
- `src/training.rs:103` — `inverted_index[(km - 1) as usize]`
- `src/training.rs:464` — `local[(km - 1) as usize]`
- `src/training.rs:654` — `(k + 1) as i32` in keep_indices
- `src/training.rs:679-680` — `(km - 1) as usize` in leaf profile
- `src/classify.rs:294` — `counts[(uk - 1) as usize]`
- `src/classify.rs:353` — `counts[(sk - 1) as usize]`
- `src/matching.rs:186-189` — `(kmer - 1) as usize` in inverted index lookup

**Decision node profiles** — the `keep` field in `DecisionNode` stores 1-indexed values. Changing to 0-indexed affects:
- `src/training.rs:654` — where `keep_indices` is built
- `src/classify.rs:208` — where `int_match(&dk.keep, my_kmers)` is called
- `src/training.rs:329` — where `int_match(&dk.keep, &kmers[i])` is called

**Serialization compatibility**: Changing from 1-indexed to 0-indexed in `DecisionNode.keep` and `TrainingSet.kmers` breaks bincode compatibility with existing saved models. Options:
1. Accept the break (models must be retrained) — simplest
2. Add a version field to TrainingSet and convert on load — more complex
3. Do the conversion at the serialization boundary only — minimal code change

**Recommendation**: Option 1 (accept the break). The project is pre-1.0 and models are cheap to retrain. Add a version field for future-proofing:

```rust
pub struct TrainingSet {
    /// Format version. 0 = original 1-indexed, 1 = 0-indexed.
    #[serde(default)]
    pub version: u32,
    // ... rest unchanged
}
```

On load, if `version == 0`, return an error suggesting retraining.

### Verification:
- All golden tests must be regenerated (k-mer values shift by 1)
- `test_baseline_1k` must still pass (output is taxonomy paths, not k-mer values)
- `cargo bench` shows small but measurable improvement across all benchmarks

**Risk**: This touches many files. Should be done as a single atomic commit with all golden test data regenerated.

### Alternative: Skip this phase
The +1/-1 overhead is real but small (single integer add per k-mer per access). If the risk of a pervasive change outweighs the benefit, skip this and focus on phases 1-4 which are localized and safe.

---

## Phase 6: Merge-Join for Profile Extraction in create_tree

### Overview
Replace per-child HashMap construction with a two-pointer merge-join when extracting selected profiles from sparse profiles. Both are sorted by k-mer index.

**File**: `src/training.rs:658-664`

### Current code:
```rust
let selected_profiles: Vec<Vec<f64>> = profiles.iter()
    .map(|p| {
        let p_map: HashMap<usize, f64> = p.iter().map(|&(k, v)| (k, v)).collect();
        keep_vec.iter().map(|&k| *p_map.get(&k).unwrap_or(&0.0)).collect()
    })
    .collect();
```

### New code:
```rust
let selected_profiles: Vec<Vec<f64>> = profiles.iter()
    .map(|p| {
        // Both p and keep_vec are sorted by k-mer index — merge join
        let mut result = Vec::with_capacity(keep_vec.len());
        let mut pi = 0usize;
        for &k in &keep_vec {
            // Advance profile cursor to k or past it
            while pi < p.len() && p[pi].0 < k {
                pi += 1;
            }
            if pi < p.len() && p[pi].0 == k {
                result.push(p[pi].1);
            } else {
                result.push(0.0);
            }
        }
        result
    })
    .collect();
```

This eliminates n_children HashMap allocations per node during tree construction.

### Verification:
- All `test_training_*` golden tests pass
- `cargo bench -- learn_taxa` shows improvement
- Expected: Small but measurable improvement during training

---

## Testing Strategy

### Correctness:
Every phase must pass `cargo test` unchanged. The golden tests are the primary correctness guard — they verify bit-identical output against R-generated reference data.

### Performance:
Run `cargo bench` before and after each phase. Record results:

```bash
# Before:
cargo bench -- --save-baseline before

# After each phase:
cargo bench -- --baseline before
```

Key benchmarks to watch:
- Phase 1: `enumerate_sequences/1K_k8`, `enumerate_single/200bp_k8`
- Phase 2: `learn_taxa/80seqs` (small dataset, may not show much)
- Phase 3: `id_taxa_sequential/15q_80ref`
- Phase 4: `id_taxa_sequential/15q_80ref`
- Phase 5: All benchmarks (pervasive change)
- Phase 6: `learn_taxa/80seqs`

### Scaling test:
Phases 2-4 benefit most from larger datasets. Use the benchmark examples:
```bash
cargo run --example bench_inverted --release
```

## Performance Considerations

| Phase | Expected Speedup | Risk | Touches |
|-------|-----------------|------|---------|
| 1 | 4-7x on enumerate_single | Low — localized change | kmer.rs only |
| 2 | Negligible on small data, significant on 10K+ | Very low — 5 lines | training.rs only |
| 3 | 10-30% on inverted index matching | Low — localized change | matching.rs only |
| 4 | 20-40% on sampling sort step | Low — localized change | classify.rs only |
| 5 | ~5% across the board | **High** — pervasive, breaks serialization | 7+ files, golden tests |
| 6 | Small (training only) | Very low — localized change | training.rs only |

**Recommendation**: Do phases 1-4 and 6 first (high value, low risk, localized). Phase 5 is optional — the benefit is small relative to the risk and effort.

## References

- Research: `thoughts/shared/research/2026-04-13-algorithmic-improvements.md`
- Algorithmic improvements plan: `thoughts/shared/plans/2026-04-13-algorithmic-improvements.md`
- Existing benchmarks: `benches/oxidtaxa_bench.rs`
- Inverted index benchmark: `examples/bench_inverted.rs`
