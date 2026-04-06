# Rust IDTAXA Speed Optimization Plan

## Overview

Optimize the Rust implementation of IDTAXA train and classify to close the single-threaded classification gap with R/C (currently 0.7x at 10K) and scale to production datasets (200K+ reference sequences, 100s-10Ks query sequences). Establish Criterion micro-benchmarks to measure each optimization's impact. Classification taxon paths must remain identical; confidence values may have epsilon tolerance (<1e-9).

## Current State Analysis

### Benchmark Baseline (Apple M4 Pro, 14 cores)

```
                    R/C        Rust 1T    Rust 8T    Speedup(1T)  Speedup(8T)
  1K ref / 500q
  Train           4.81s        0.58s        —          8.3x           —
  Classify        1.39s        0.62s      0.12s        2.2x         11.6x

  5K ref / 500q
  Train          17.17s        1.78s        —          9.6x           —
  Classify        4.65s        3.88s      0.70s        1.2x          6.6x

  10K ref / 500q
  Train          29.63s        3.39s        —          8.7x           —
  Classify        9.00s       12.25s      1.91s        0.7x          4.7x
```

**Key problem**: Single-threaded classification degrades from 2.2x faster (1K) to 0.7x slower (10K) relative to R/C. At 200K refs this will be dramatically worse.

### Root Cause Analysis

Profiled the code paths and identified these bottlenecks ranked by estimated impact at 200K scale:

| # | Location | Issue | Complexity | Impact at 200K |
|---|----------|-------|------------|-----------------|
| 1 | `training.rs:86-103` | `all_taxa.contains()`, `u_classes.contains()` — linear scan for set membership | O(n) per insert = O(n²) total | **Catastrophic** — 200K² = 40B ops |
| 2 | `training.rs:192-203` | `sequences_per_node`: nested loop `classes × end_taxonomy` with `starts_with` | O(n × t) where t = taxonomy nodes | **Catastrophic** |
| 3 | `classify.rs:470-491` | `dereplicate()`: `unique_seqs.iter().position()` — linear scan | O(n²) | **Critical** at 10K+ queries |
| 4 | `classify.rs:87-147` | `parallel_match`: per-sequence Vec<f64> allocation in rayon hot loop | O(keep × queries) allocs | **Critical** — millions of allocs |
| 5 | `kmer.rs:312` | `enumerate_single`: `bases.remove(0)` — Vec shift in inner loop | O(word_size) per position | **High** — runs on every base of every sequence |
| 6 | `classify.rs:449` | `classify_parallel`: `boths.iter().position()` linear scan per sequence | O(n) per query | **High** at 10K+ queries |
| 7 | `classify.rs:152-378` | `classify_one_pass`: multiple small Vec allocations per call | Constant but frequent | **Medium** — 10Ks×2 calls |
| 8 | `sequence.rs` | `remove_gaps`, `reverse_complement`: char-level ops on ASCII data | Constant factor | **Low-Medium** |
| 9 | `training.rs:358-382` | IDF weight computation: sequential loop over all sequences | O(n × avg_kmers) | **Medium** at 200K |
| 10 | `training.rs:452-591` | `create_tree`: HashMap per node for sparse profile lookups | Constant per node | **Low** |

### Key Discoveries

- **C's `parallelMatch` (`vector_sums.c:59-146`)** uses a compact `temp` array that only stores matched indices (`temp[c++] = i`), then iterates *only* over matches. Rust's version (`matching.rs:86-122`) allocates per-sequence and iterates all query k-mers. At 200K the match rate is low, so C's approach skips most work.
- **R's `LearnTaxa` (`LearnTaxa.R:253-263`)** uses `match()` + `tapply()` for `sequences_per_node`, which is internally a hash-based O(n) operation. Rust uses nested loops with `starts_with` — O(n × t).
- **R's `sample(n, size, replace=TRUE)`** is a tight C loop (`y[i] = (int)(n * unif_rand()) + 1`). Rust's `sample_int_replace` allocates a Vec. For the hot classification path this matters.
- The `enumerate_single` sliding window in C (`enumerate_sequence.c:346-358`) uses `bases[k-1] = bases[k]` on a stack array. Rust uses `bases.remove(0)` on a Vec, which shifts all elements *and* has bounds-check overhead.

## Desired End State

- Rust single-threaded classification at least **2x faster than R/C** at all scales (1K-200K)
- Rust multi-threaded classification at least **10x faster than R/C** at production scale
- Training remains 8-10x faster (no regression)
- Criterion micro-benchmarks for all hot functions, runnable via `cargo bench`
- All 51 golden tests still pass (identical taxon paths, confidence within 1e-9)

### How to Verify

```bash
# Golden tests (correctness)
cd rust && cargo test --release

# Micro-benchmarks (per-function)
cd rust && cargo bench

# End-to-end benchmarks (comparison with R/C)
bash benchmarks/run_benchmark.sh
```

## What We're NOT Doing

- Changing the algorithm (tree structure, bootstrap logic, voting)
- SIMD intrinsics (future work — complex and platform-specific)
- GPU acceleration
- Changing the R/C implementation
- Changing the Python CLI layer
- Changing the bincode serialization format
- Memory-mapped I/O for FASTA files (marginal gain for these sizes)

## Implementation Approach

Three phases, ordered by impact. Each phase is independently testable and benchmarkable. Phase 1 establishes measurement infrastructure. Phase 2 fixes the critical algorithmic issues. Phase 3 tightens the constant factors.

---

## Phase 1: Criterion Micro-Benchmarks

### Overview
Add Criterion benchmarks for every hot function. This gives us precise before/after measurements for each optimization and catches regressions.

### Changes Required

#### 1. Add Criterion dependency
**File**: `rust/Cargo.toml`
**Changes**: Add criterion dev-dependency and benchmark harness config.

```toml
[dev-dependencies]
approx = "0.5"
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "idtaxa_bench"
harness = false
```

#### 2. Create benchmark suite
**File**: `rust/benches/idtaxa_bench.rs` (new)
**Changes**: Benchmarks for each hot function using the 1K benchmark dataset.

Benchmark groups to include:
- `enumerate_sequences` — 1K sequences, k=8
- `enumerate_single` — single 200bp sequence, k=8
- `int_match` — two sorted vectors of realistic sizes
- `vector_sum` — realistic sampling/matches/weights
- `parallel_match` — 500 query k-mers vs 1K training sequences
- `dereplicate` — 500 sequences with ~10% duplicates
- `classify_one_pass` — single sequence against a trained model
- `classify_sequential` — 500 sequences, single-threaded
- `create_tree` — tree building from 1K sequences
- `learn_taxa` — full training on 1K sequences
- `sample_int_replace` — 10K samples from range 1000
- `remove_gaps` — 500 sequences
- `reverse_complement` — 500 sequences
- `read_fasta` — 1K sequence file

Each benchmark should:
- Load data once in setup (not timed)
- Use `criterion::black_box` to prevent dead-code elimination
- Run with `--release` profile
- Report throughput where applicable (elements/sec)

### Success Criteria

#### Automated Verification:
- [x] `cd rust && cargo bench` completes without errors
- [x] `cd rust && cargo test --release` — all existing tests pass
- [ ] Benchmark results visible in `rust/target/criterion/`

#### Manual Verification:
- [ ] Each benchmark group produces stable results (low variance)
- [ ] Baseline numbers recorded in `benchmarks/criterion_baseline.txt`

---

## Phase 2: Algorithmic Fixes (High-Impact)

### Overview
Fix the O(n²) and O(n×t) algorithms that make training and classification scale poorly. These are the changes that will have dramatic impact at 200K.

### Changes Required

#### 1. Fix `dereplicate` — O(n²) → O(n) with HashMap
**File**: `rust/src/classify.rs` — `dereplicate()` (lines 470-491)
**Current**: Linear scan `unique_seqs.iter().position(|s| s == seq)` per sequence.
**Change**: Use `HashMap<&str, usize>` for O(1) lookup.

```rust
fn dereplicate(
    sequences: &[String],
    strand_mode: StrandMode,
) -> (Vec<String>, Vec<usize>, Vec<i32>) {
    use std::collections::HashMap;
    let strand_val = match strand_mode {
        StrandMode::Both => 1,
        StrandMode::Top => 2,
        StrandMode::Bottom => 3,
    };
    let mut seen: HashMap<&str, usize> = HashMap::with_capacity(sequences.len());
    let mut unique_seqs: Vec<String> = Vec::new();
    let mut map: Vec<usize> = Vec::with_capacity(sequences.len());
    for seq in sequences.iter() {
        if let Some(&idx) = seen.get(seq.as_str()) {
            map.push(idx);
        } else {
            let idx = unique_seqs.len();
            seen.insert(seq.as_str(), idx);
            map.push(idx);
            unique_seqs.push(seq.clone());
        }
    }
    let strands = vec![strand_val; unique_seqs.len()];
    (unique_seqs, map, strands)
}
```

#### 2. Fix `boths` lookup — O(n) → O(1) with HashSet
**File**: `rust/src/classify.rs` — `classify_parallel()` (line 449)
**Current**: `pre.boths.iter().position(|&bi| bi == i)` — linear scan per sequence.
**Change**: Add a `boths_map: HashMap<usize, usize>` to `PrecomputedData` that maps original index → rev_kmers index. Build it once in `precompute()`.

```rust
// In PrecomputedData struct:
boths_map: HashMap<usize, usize>,  // original_idx -> rev_kmers_idx

// In precompute():
let boths_map: HashMap<usize, usize> = boths.iter().enumerate()
    .map(|(rev_idx, &orig_idx)| (orig_idx, rev_idx))
    .collect();

// In classify_parallel(), replace:
//   if let Some(both_pos) = pre.boths.iter().position(|&bi| bi == i)
// with:
//   if let Some(&both_pos) = pre.boths_map.get(&i)
```

#### 3. Fix training `u_classes` and `all_taxa` — O(n²) → O(n)
**File**: `rust/src/training.rs` — `learn_taxa()` (lines 85-105)
**Current**: `uc.contains(c)` and `all_taxa.contains(&prefix)` — linear scan.
**Change**: Use `IndexSet` from the `indexmap` crate (preserves insertion order like the current Vec, but O(1) lookup), or use a parallel `HashSet` for membership + Vec for ordering.

```rust
// For u_classes (line 85-93):
let mut uc_set: HashSet<String> = HashSet::new();
let mut u_classes: Vec<String> = Vec::new();
for c in &classes {
    if uc_set.insert(c.clone()) {
        u_classes.push(c.clone());
    }
}

// For all_taxa (line 96-105):
let mut taxa_set: HashSet<String> = HashSet::new();
let mut all_taxa: Vec<String> = Vec::new();
for uc in &u_classes {
    let parts: Vec<&str> = uc.split(';').filter(|s| !s.is_empty()).collect();
    for n in 1..=parts.len() {
        let prefix: String = parts[..n].iter().map(|s| format!("{};", s)).collect();
        if taxa_set.insert(prefix.clone()) {
            all_taxa.push(prefix);
        }
    }
}
```

**Cargo.toml**: No new dependency needed — `std::collections::HashSet` suffices since we just need a parallel set for dedup. The Vec preserves order for index stability.

#### 4. Fix `sequences_per_node` — O(n×t) → O(n)
**File**: `rust/src/training.rs` — lines 180-203
**Current**: For each taxonomy node, scans ALL sequences checking `class.starts_with(et)`. This is O(sequences × taxonomy_nodes).
**Change**: Group sequences by their class string using a HashMap, then walk the taxonomy tree assigning groups. This matches R's `match()` + `tapply()` approach.

```rust
// Build class -> sequence indices mapping
let mut class_to_seqs: HashMap<&str, Vec<usize>> = HashMap::new();
for (seq_idx, class) in classes.iter().enumerate() {
    class_to_seqs.entry(class.as_str()).or_default().push(seq_idx);
}

// For each taxonomy node, collect sequences whose class starts with this node's taxonomy
// Use the cross_index we already computed: sequences_per_node[tax_idx] = all seq_idx where
// the sequence's deepest taxonomy is a descendant of tax_idx.
//
// Better approach: walk each sequence's full taxonomy prefix chain and assign to all ancestors.
let mut sequences_per_node: Vec<Option<Vec<usize>>> = vec![None; taxonomy.len()];

// Build taxonomy string -> index mapping for O(1) lookup
let mut tax_to_idx: HashMap<&str, usize> = HashMap::with_capacity(taxonomy.len());
for (i, t) in taxonomy.iter().enumerate() {
    tax_to_idx.insert(t.as_str(), i);
}

// For each sequence, walk up its taxonomy prefix chain
for (seq_idx, class) in classes.iter().enumerate() {
    let full = format!("Root;{}", class);
    // Generate all prefixes of this taxonomy
    let parts: Vec<&str> = full.split(';').filter(|s| !s.is_empty()).collect();
    for n in 1..=parts.len() {
        let prefix: String = parts[..n].iter().map(|s| format!("{};", s)).collect();
        if let Some(&tax_idx) = tax_to_idx.get(prefix.as_str()) {
            sequences_per_node[tax_idx]
                .get_or_insert_with(Vec::new)
                .push(seq_idx);
        }
    }
}
```

#### 5. Fix `enumerate_single` sliding window — Vec::remove(0) → fixed array
**File**: `rust/src/kmer.rs` — `enumerate_single()` (lines 270-332)
**Current**: `bases.remove(0)` on line 312 — shifts all elements left, O(word_size) per position.
**Change**: Use a fixed-size array with manual shift (matching C) or a circular buffer index.

```rust
fn enumerate_single(
    seq: &[u8],
    word_size: usize,
    pwv: &[i32],
    mask_reps: bool,
    mask_lcrs: bool,
    mask_num: Option<i32>,
) -> Vec<i32> {
    let len = seq.len();
    if len < word_size || word_size == 0 {
        return Vec::new();
    }

    let n_kmers = len - word_size + 1;
    let mut result = vec![0i32; n_kmers];

    // Fixed-size array for sliding window (max K=15)
    let mut bases = [0i8; 16];
    for j in 0..(word_size - 1) {
        bases[j] = base_to_index(seq[j]);
    }

    for j in (word_size - 1)..len {
        bases[word_size - 1] = base_to_index(seq[j]);

        let mut sum = bases[0] as i32 * pwv[0];
        let mut ambiguous = bases[0] < 0;
        for k in 1..word_size {
            sum += bases[k] as i32 * pwv[k];
            if bases[k] < 0 {
                ambiguous = true;
            }
        }

        let pos = j + 1 - word_size;
        result[pos] = if ambiguous { NA_INTEGER } else { sum };

        // Shift left: matches C's bases[k-1] = bases[k]
        for k in 0..(word_size - 1) {
            bases[k] = bases[k + 1];
        }
    }

    // Apply masking (unchanged)
    if mask_reps { mask_repeats(&mut result, word_size); }
    if mask_lcrs {
        mask_simple(&mut result, word_size, 4, 20, 12.66667);
        mask_simple(&mut result, word_size, 4, 95, 38.90749);
    }
    if let Some(max_count) = mask_num {
        let tot = 4usize.pow(word_size as u32);
        mask_numerous(&mut result, max_count, tot, word_size);
    }

    result
}
```

### Success Criteria

#### Automated Verification:
- [x] `cd rust && cargo test --release` — all 51 golden tests pass
- [ ] `cd rust && cargo bench` — shows improvement vs Phase 1 baselines
- [ ] `cd rust && cargo clippy` — no new warnings

#### Manual Verification:
- [ ] `bash benchmarks/run_benchmark.sh` — single-threaded classify is faster than R/C at all scales
- [ ] Training time does not regress
- [ ] Record new benchmark numbers in results file

---

## Phase 3: Constant-Factor Optimizations

### Overview
Reduce allocation overhead and improve cache efficiency in the hot classification loop. These are tighter optimizations that compound on the algorithmic fixes from Phase 2.

### Changes Required

#### 1. Reduce allocations in `classify_one_pass`
**File**: `rust/src/classify.rs` — `classify_one_pass()` (lines 152-378)
**Current**: Allocates `vote_counts`, `hits`, `w_indices`, `keep`, `sampling`, `u_sampling`, `grouped`, `positions`, `ranges`, `u_weights` per call.
**Change**: Pre-allocate reusable buffers in the caller and pass them in. For the parallel path, each thread gets its own buffer set (thread-local or passed via closure).

Key buffers to pre-allocate:
- `vote_counts: Vec<usize>` — max size = max children at any node
- `hits: Vec<Vec<f64>>` — max size = max children × bootstraps
- `sampling: Vec<i32>` — max size = s * b (known from PrecomputedData)

Create a `ClassifyBuffers` struct:

```rust
struct ClassifyBuffers {
    vote_counts: Vec<usize>,
    hits_flat: Vec<f64>,     // flattened [subtrees × b]
    sampling: Vec<i32>,
    u_sampling: Vec<i32>,
    grouped: Vec<Vec<usize>>,
    positions: Vec<usize>,
    ranges: Vec<usize>,
}

impl ClassifyBuffers {
    fn new(max_bootstraps: usize, max_subtrees: usize, max_s: usize) -> Self { ... }
    fn clear(&mut self) { ... }  // reset lengths without dealloc
}
```

#### 2. Optimize `sample_int_replace` for hot path
**File**: `rust/src/rng.rs` — `sample_int_replace()` (lines 150-154)
**Current**: Allocates Vec, collects iterator.
**Change**: Add a `sample_int_replace_into` variant that writes into a pre-allocated slice.

```rust
pub fn sample_int_replace_into(&mut self, n: usize, buf: &mut [usize]) {
    for slot in buf.iter_mut() {
        *slot = self.r_unif_index(n);
    }
}
```

#### 3. Optimize `parallel_match` — match C's compact-index approach
**File**: `rust/src/matching.rs` — `parallel_match()` (lines 75-122)
**Current**: Iterates all `size_x` query k-mers for every training sequence.
**Change**: Match C's approach — collect matched indices into a compact buffer first, then only iterate matches for the weight accumulation. At 200K refs, match rate is low (~1-5%), so this skips 95%+ of the inner loop.

```rust
// Per training sequence (inside par_iter):
let mut matched_indices: Vec<usize> = Vec::new(); // reuse across iterations
let mut j = 0usize;
for i in 0..size_x {
    while j < train_k.len() {
        if query_kmers[i] <= train_k[j] {
            if query_kmers[i] == train_k[j] {
                matched_indices.push(i);
            }
            break;
        }
        j += 1;
    }
}

// Only iterate over matches (typically ~1-5% of size_x)
let mut hits = vec![0.0f64; block_count];
for &mi in &matched_indices {
    for &pos in &positions[ranges[mi]..ranges[mi + 1]] {
        hits[pos] += weights[mi];
    }
}
```

#### 4. Byte-level DNA operations
**File**: `rust/src/sequence.rs`
**Current**: `remove_gaps` uses `seq.chars().filter(...)`. `reverse_complement` maps bytes then converts to char then collects String.
**Change**: Work entirely at byte level.

```rust
pub fn reverse_complement(seq: &str) -> String {
    let bytes: Vec<u8> = seq.bytes().rev().map(|b| match b {
        b'A' | b'a' => b'T', b'T' | b't' => b'A',
        b'C' | b'c' => b'G', b'G' | b'g' => b'C',
        b'M' | b'm' => b'K', b'K' | b'k' => b'M',
        b'R' | b'r' => b'Y', b'Y' | b'y' => b'R',
        b'W' | b'w' => b'W', b'S' | b's' => b'S',
        b'V' | b'v' => b'B', b'B' | b'b' => b'V',
        b'H' | b'h' => b'D', b'D' | b'd' => b'H',
        b'N' | b'n' => b'N',
        other => other,
    }).collect();
    // SAFETY: input is ASCII DNA, output is ASCII DNA
    unsafe { String::from_utf8_unchecked(bytes) }
}

pub fn remove_gaps(sequences: &[String]) -> Vec<String> {
    sequences.par_iter().map(|seq| {
        let bytes: Vec<u8> = seq.bytes().filter(|&b| b != b'-' && b != b'.').collect();
        unsafe { String::from_utf8_unchecked(bytes) }
    }).collect()
}
```

#### 5. IDF weight computation — parallelize with rayon
**File**: `rust/src/training.rs` — lines 358-382
**Current**: Sequential loop over all sequences for IDF weight accumulation.
**Change**: Use rayon to partition sequences across threads, compute partial counts per thread, then merge. At 200K sequences this is a significant win.

```rust
use rayon::prelude::*;

// Parallel IDF computation
let chunk_counts: Vec<Vec<f64>> = kmers.par_chunks(256).map(|chunk_kmers| {
    // Need to figure out the weight index range for this chunk
    // Actually, partition by sequence index
    let mut local_counts = vec![0.0f64; n_kmers];
    for (local_i, class_kmers) in chunk_kmers.iter().enumerate() {
        let global_i = /* compute global index */;
        let w = weights[global_i];
        for &km in class_kmers {
            if km > 0 && (km as usize) <= n_kmers {
                local_counts[(km - 1) as usize] += w;
            }
        }
    }
    local_counts
}).collect();

// Merge
let mut idf_counts = vec![0.0f64; n_kmers];
for partial in &chunk_counts {
    for (i, &v) in partial.iter().enumerate() {
        idf_counts[i] += v;
    }
}
```

Note: The parallel merge approach requires careful index management. An alternative is to use atomic floats or a simpler partitioning scheme. Implementation details will be refined during development.

#### 6. Pre-compute `cross_index` lookup table for classification
**File**: `rust/src/classify.rs` — `classify_one_pass()` (line 282)
**Current**: `keep.iter().map(|&idx| cross_index[idx])` — fine per-query, but `unique_groups` sort+dedup is repeated.
**Change**: Minor — this is already O(keep) which is unavoidable. But we can avoid the sort by using a HashSet.

### Success Criteria

#### Automated Verification:
- [ ] `cd rust && cargo test --release` — all golden tests pass
- [ ] `cd rust && cargo bench` — improvement over Phase 2 baselines for:
  - `classify_one_pass` (target: 2x improvement from allocation reduction)
  - `parallel_match` (target: 3-5x at low match rate)
  - `enumerate_single` (target: captured in Phase 2, verify no regression)
  - `sample_int_replace` (target: 1.5x from avoiding allocation)
- [ ] `cd rust && cargo clippy` — no warnings

#### Manual Verification:
- [ ] `bash benchmarks/run_benchmark.sh` — record final numbers
- [ ] Single-threaded classify ≥ 2x faster than R/C at 10K
- [ ] 8-thread classify ≥ 10x faster than R/C at 10K
- [ ] Training has not regressed
- [ ] All confidence values within 1e-9 of R golden baselines

---

## Testing Strategy

### Unit Tests
- Each optimized function retains its existing golden tests
- Add specific regression tests for edge cases:
  - `dereplicate` with 100% duplicates, 0% duplicates, empty input
  - `enumerate_single` with sequences shorter than word_size
  - `sequences_per_node` with single-sequence taxonomy nodes

### Integration Tests
- Full train→classify pipeline on test dataset (80 ref, 15 query)
- Full train→classify pipeline on 1K benchmark dataset
- Verify classification paths are identical to R output
- Verify confidence values within epsilon of R output

### Micro-Benchmarks (Criterion)
- Before/after comparison for each phase
- Parameterized benchmarks: vary dataset size (1K, 5K, 10K) to show scaling
- Throughput metrics: sequences/sec, k-mers/sec

### Manual Testing Steps
1. Run `bash benchmarks/run_benchmark.sh` and compare to baseline
2. If possible, test with a 200K reference dataset to verify scaling
3. Verify memory usage doesn't blow up (track peak RSS)

## Performance Considerations

- **Memory at 200K**: Each training sequence stores sorted unique k-mers. At 200K × ~150 k-mers × 4 bytes = ~120MB. The `parallel_match` hit matrix at 200K × 50 bootstraps × 8 bytes = ~80MB per query. Need to verify memory stays reasonable.
- **Cache locality**: The `parallel_match` inner loop accesses `train_kmers[idx]` which jumps across memory. At 200K this causes many cache misses. Consider sorting `keep` indices to improve locality.
- **Thread scaling**: With 200K training sequences, the rayon work-stealing in `parallel_match` should scale well since each unit of work is small and uniform.

## References

- Existing port plan: `thoughts/shared/plans/2026-04-03-idtaxa-python-rust-port.md`
- Current benchmarks: `benchmarks/results.txt`
- R source: `R/LearnTaxa.R`, `R/IdTaxa.R`
- C source: `src/vector_sums.c`, `src/utils.c`, `src/enumerate_sequence.c`
- Rust source: `rust/src/` (all modules)
- Expert guidelines: `.claude/projects/-Users-ryanmartin-idtaxa-optim/memory/expert_translation_guidelines.md`
