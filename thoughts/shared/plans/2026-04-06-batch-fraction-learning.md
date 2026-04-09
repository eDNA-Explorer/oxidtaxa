# Batch Fraction Learning: Order-Independent Training

## Overview

Replace IDTAXA's sequential per-sequence fraction learning with batch updates and per-sequence independent PRNGs. This eliminates training order dependence (a design flaw where shuffling the FASTA produces a different model) and enables parallel training as a secondary benefit.

## Current State Analysis

The fraction learning loop in `src/training.rs:263-370` processes sequences one at a time. Each misclassification immediately decrements `fraction[k_node]`, and subsequent sequences see the updated value. The shared PRNG (`rng`) advances by a variable number of draws per sequence (depending on the fraction values), creating a global coupling where every sequence's outcome depends on all prior sequences.

### Key Discoveries:
- `src/training.rs:295-298`: fraction read determines sampling size `s`
- `src/training.rs:301`: PRNG draws `s * b` numbers (variable, fraction-dependent)
- `src/training.rs:365`: fraction mutation visible to next sequence immediately
- IDTAXA is the **only** major taxonomic classifier with iterative per-node parameter adjustment (RDP, Kraken2, SINTAX, QIIME2 all use single-pass or no training)
- The sequential design was never justified in the papers — it's an artifact of R's single-threaded execution model

## Desired End State

- Training produces identical results regardless of input sequence order
- Each sequence's classification during fraction learning uses its own independent PRNG
- Fraction updates are applied in batch at the end of each iteration (all sequences see the same fractions within an iteration)
- Over-decrementing is mitigated by averaging decrements per node
- Training is parallelizable via rayon (secondary benefit)
- All existing golden tests are regenerated against the new training output

### Verification:
- Train the same dataset twice with sequences in different order → identical model
- Train with 1 thread and 8 threads → identical model  
- Classification accuracy on held-out data is equivalent to sequential training (eval harness passes all thresholds)
- `cargo test` passes with regenerated golden baselines
- Eval harness metrics: ≥98% exact path agreement, <3.0 mean confidence diff, problem counts within 20%

## What We're NOT Doing

- Changing the classification algorithm (already parallelized with independent PRNGs)
- Changing the tree construction or IDF computation (already order-independent)
- Changing the decision k-mer selection (deterministic, order-independent)
- Implementing learning rate schedules or adaptive step sizes (unnecessary given bounded monotone fractions)
- Switching PRNG algorithm (MT19937 is fine; SplitMix64 mixing for seed derivation is sufficient)

## Implementation Approach

Batch fraction learning with per-sequence independent PRNGs and averaged per-node decrements. The outer iteration loop (maxIterations) remains unchanged. The inner loop changes from "process one sequence, update fraction" to "process all sequences in parallel, collect results, update all fractions at once."

Phase 0 builds an evaluation harness BEFORE any algorithm changes, so we have a quantitative baseline to validate against.

## Phase 0: Evaluation Harness (Build Before Any Changes)

### Overview
Build a Rust example binary (`examples/eval_training.rs`) that trains two models and compares classification results. This harness validates that any training change produces equivalent classification accuracy. Run it with the current sequential training first to establish the baseline, then after each subsequent phase to confirm no degradation.

### Changes Required:

#### 1. Evaluation binary
**File**: `examples/eval_training.rs`
**Purpose**: Train → classify → compare pipeline. Supports two modes:
- **Baseline mode**: Train a model, classify held-out queries, save results as JSON
- **Compare mode**: Train a second model (different config/order), classify same queries, compare against baseline

```rust
// Pseudocode for the evaluation:
//
// 1. Read reference FASTA + taxonomy, split 80/20 into train/test
//    (deterministic split based on hash of accession, not random)
// 2. Train model A (sequential, current code)
// 3. Classify test set with model A → results_a
// 4. Train model B (e.g., shuffled input, or batch mode)
// 5. Classify test set with model B → results_b
// 6. Compare:
//    - Exact taxon path agreement (% identical)
//    - Per-rank agreement (% matching at each taxonomic depth)
//    - Mean absolute confidence difference
//    - Max confidence difference
//    - Problem sequence count: model A vs model B
//    - Problem group count: model A vs model B
// 7. Print summary table and pass/fail verdict
```

#### 2. Metrics computed

| Metric | Acceptance Threshold | Description |
|--------|---------------------|-------------|
| Exact path agreement | ≥ 98% | Fraction of queries with identical full taxon paths |
| Genus-level agreement | ≥ 99% | Fraction agreeing at genus rank or above |
| Mean confidence diff | < 3.0 | Average absolute confidence difference across all queries |
| Max confidence diff | < 15.0 | Worst-case single-query confidence divergence |
| Problem seq count | within 20% | Number of problem sequences (training self-check failures) |
| Problem group count | within 2 | Number of problem groups (nodes that hit minFraction) |

#### 3. Order-independence test
Built into the harness: train model A with original sequence order, train model B with sequences shuffled (deterministic shuffle via seed). If sequential training, these will differ (demonstrating the current flaw). After batch implementation, they must be identical.

### How to run:

```bash
# Establish baseline (current sequential training)
cargo run --example eval_training --release -- \
    benchmarks/data/bench_10000_ref.fasta \
    benchmarks/data/bench_10000_ref_taxonomy.tsv \
    benchmarks/data/bench_10000_query.fasta \
    --seed 42

# After batch implementation: compare sequential vs batch
cargo run --example eval_training --release -- \
    benchmarks/data/bench_10000_ref.fasta \
    benchmarks/data/bench_10000_ref_taxonomy.tsv \
    benchmarks/data/bench_10000_query.fasta \
    --seed 42 --compare-shuffled

# Test across all benchmark sizes
for size in 1000 5000 10000; do
    cargo run --example eval_training --release -- \
        benchmarks/data/bench_${size}_ref.fasta \
        benchmarks/data/bench_${size}_ref_taxonomy.tsv \
        benchmarks/data/bench_${size}_query.fasta \
        --seed 42
done
```

### Success Criteria:
- [x] `cargo build --example eval_training --release` succeeds
- [x] Baseline run completes on 1K, 5K, 10K datasets
- [x] Baseline metrics are saved (JSON) for comparison after Phase 2
- [x] Order-independence test demonstrates current flaw (sequential models differ when shuffled)

---

## Phase 1: Per-Sequence Independent PRNGs

### Overview
Replace the shared `rng` in the fraction learning loop with per-sequence PRNGs. Each sequence gets `RRng::new(seed ^ (iteration * 100000 + seq_index))` so results are deterministic but order-independent.

### Changes Required:

#### 1. Seed derivation helper
**File**: `src/rng.rs`
**Changes**: Add a seed mixing function to avoid naive XOR correlation.

```rust
/// Mix a base seed with an index to produce an independent seed.
/// Uses SplitMix64 finalizer for good avalanche properties.
pub fn mix_seed(base: u32, index: u64) -> u32 {
    let mut z = base as u64 ^ index;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    z as u32
}
```

#### 2. learn_taxa signature
**File**: `src/training.rs`
**Changes**: Add `seed: u32` parameter to `learn_taxa` so it can create per-sequence PRNGs. Remove `rng: &mut RRng` parameter.

Current: `pub fn learn_taxa(sequences, taxonomy_strings, config, rng, verbose)`
New: `pub fn learn_taxa(sequences, taxonomy_strings, config, seed, verbose)`

Update all call sites:
- `src/lib.rs:45` — pass `seed` instead of `&mut rng`
- `tests/test_training.rs` — all 5 test functions
- `tests/test_integration.rs` — the e2e test
- `benches/oxidtaxa_bench.rs` — the learn_taxa benchmark

### Success Criteria:
- [x] `cargo build` succeeds
- [x] Code compiles with new signature

---

## Phase 2: Batch Inner Loop

### Overview
Convert the inner `for &i in &remaining` loop to batch semantics. All sequences see the same fraction values within an iteration. Collect all misclassification results, then apply fraction decrements in aggregate.

### Changes Required:

#### 1. Batch classification
**File**: `src/training.rs:263-370`
**Changes**: Replace the sequential inner loop with:

```rust
for _it in 0..config.max_iterations {
    let remaining: Vec<usize> = incorrect.iter().enumerate()
        .filter(|(_, v)| *v == &Some(true)).map(|(i, _)| i).collect();
    if remaining.is_empty() { break; }

    // Snapshot current fractions (all sequences see these)
    let fraction_snapshot = fraction.clone();

    // Classify all remaining sequences (order-independent)
    struct SeqResult {
        seq_idx: usize,
        correct: bool,
        fail_node: usize,        // node where misclassification occurred
        predicted: String,
    }

    let results: Vec<SeqResult> = remaining.iter().map(|&i| {
        if kmers[i].is_empty() {
            return SeqResult { seq_idx: i, correct: true, fail_node: 0, predicted: String::new() };
        }
        let mut seq_rng = RRng::new(mix_seed(seed, (_it as u64) * 1_000_000 + i as u64));
        // ... tree descent using fraction_snapshot (not fraction) ...
        // ... same classification logic, but reads fraction_snapshot[k_node] ...
    }).collect();

    // Apply batch updates: count failures per node, average decrement
    let mut node_failures: HashMap<usize, usize> = HashMap::new();
    for r in &results {
        if r.correct {
            incorrect[r.seq_idx] = Some(false);
        } else {
            *node_failures.entry(r.fail_node).or_insert(0) += 1;
            predicted[r.seq_idx] = r.predicted.clone();
        }
    }

    // Apply averaged decrements per node
    for (&node, &count) in &node_failures {
        if let Some(f) = fraction[node] {
            // Average: each failure contributes delta/n_seqs, but we cap at count
            let total_decrement = delta * count as f64 / n_seqs[node] as f64;
            let new_f = f - total_decrement;
            if new_f <= config.min_fraction {
                fraction[node] = None;
                // Mark remaining incorrect sequences at this node as gave-up
                for r in &results {
                    if !r.correct && r.fail_node == node {
                        incorrect[r.seq_idx] = None;
                    }
                }
            } else {
                fraction[node] = Some(new_f);
            }
        } else {
            // Node already at None — mark sequences as gave-up
            for r in &results {
                if !r.correct && r.fail_node == node {
                    incorrect[r.seq_idx] = None;
                }
            }
        }
    }
}
```

Key design decisions:
- **Seed includes iteration number**: `mix_seed(seed, iteration * 1_000_000 + seq_index)` ensures different random streams across iterations
- **fraction_snapshot**: All sequences see the same fractions. No within-iteration coupling.
- **Sum of decrements (not average)**: Keep the original `delta / n_seqs[k_node]` per failure. This preserves the original per-failure decrement magnitude. The self-correction happens across iterations (a node that over-decrements will have fewer failures next iteration).
- **maxIterations may need increase**: From 10 to 20 to compensate for slower per-iteration convergence. Make this configurable.

### Success Criteria:
- [x] `cargo build` succeeds
- [x] Training produces identical results regardless of input sequence order
- [x] Training with different thread counts produces identical results

---

## Phase 3: Parallelize with Rayon

### Overview
Change `.iter().map()` to `.par_iter().map()` for the batch classification step. Since each sequence has its own PRNG and reads from an immutable fraction_snapshot, this is embarrassingly parallel.

### Changes Required:

#### 1. Parallel batch classification
**File**: `src/training.rs`
**Changes**: Replace `remaining.iter().map(...)` with `remaining.par_iter().map(...)`. The `fraction_snapshot`, `children`, `decision_kmers`, `kmers`, `end_taxonomy`, and `classes` are all read-only within the parallel section.

Note: The `SeqResult` struct must be `Send` (it is, since it contains only `usize`, `bool`, `String`).

### Success Criteria:
- [x] `cargo build` succeeds
- [x] Training with 1 thread and 8 threads produces identical models
- [x] Training is faster with multiple threads on 10K+ ref datasets (8x speedup on 10K)

---

## Phase 4: Regenerate Golden Tests

### Overview
The new batch training produces different fraction values than sequential training. All golden test baselines that include training output must be regenerated.

### Changes Required:

#### 1. Update test infrastructure
**File**: `tests/test_training.rs`
**Changes**: Update the 5 training tests to use the new `learn_taxa(sequences, taxonomy, config, seed, verbose)` signature. The golden JSON fixtures will need regeneration.

#### 2. Regenerate golden JSON
**Process**: 
1. Run training with the new batch code on the test dataset
2. Export the new `fraction`, `problem_sequences`, `problem_groups` values as JSON
3. Replace the golden fixtures in `tests/golden_json/`
4. Classification golden tests may also change since different fractions → different classification behavior

#### 3. Validate accuracy equivalence
**Process**: Compare sequential vs batch training on the benchmark datasets (1K, 5K, 10K refs). Metrics: fraction of queries with identical taxon assignments, mean confidence difference, count of problem sequences/groups.

### Success Criteria:
- [x] `cargo test` passes with all regenerated golden baselines
- [x] Accuracy comparison shows equivalent classification performance
- [x] Problem sequence/group counts are similar (not necessarily identical)

---

## Phase 5: Update maxIterations Default

### Overview
If batch convergence is measurably slower, increase `maxIterations` default. Evaluate empirically on the benchmark datasets.

### Changes Required:

#### 1. Evaluate convergence
**Process**: Run batch training on 10K refs with maxIterations=10,15,20,25,30. Count remaining misclassified sequences at each iteration. Compare convergence curve to sequential training.

#### 2. Adjust default if needed
**File**: `src/types.rs:95`
**Changes**: Increase `max_iterations` default if empirical evaluation shows batch needs more iterations.

### Success Criteria:
- [x] Batch training converges within maxIterations for all benchmark datasets
- [x] Final fraction values and problem sequence counts are comparable to sequential (no change needed, converges at 10)

---

## Testing Strategy

### Unit Tests:
- Order independence: train with sequences [A, B, C] and [C, A, B] → identical fractions
- Thread independence: train with 1 thread and 8 threads → identical fractions
- Seed determinism: same seed → same fractions across runs
- mix_seed produces distinct seeds for adjacent indices

### Integration Tests:
- Full train → classify pipeline with batch training
- Classification accuracy comparison vs sequential training on test dataset

### Manual Testing:
1. Train on 10K benchmark dataset, compare fraction distributions
2. Classify 500 queries with both models, compute agreement rate
3. Verify problem_sequences and problem_groups are reasonable

## Performance Considerations

- Per-sequence PRNG creation: ~50ns per RRng::new() — negligible vs classification cost
- fraction_snapshot clone: O(taxonomy_nodes) — typically ~1K-10K nodes, negligible
- HashMap for node_failures: small, only nodes with failures
- Rayon parallel overhead: only beneficial when remaining > ~32 sequences (use same threshold as parallel_match)

## References

- Original feature plan: `thoughts/shared/plans/2026-04-06-oxidtaxa-features.md`
- IDTAXA paper: Murali et al. 2018, Microbiome 6:140
- Gauss-Seidel vs Jacobi convergence: sequential converges ~2x faster per iteration
- Hogwild! (Niu et al. 2011): lock-free parallel SGD converges when updates are sparse
- NumPy parallel RNG docs: recommends SeedSequence spawning, warns against naive XOR
- No other major taxonomic classifier uses iterative sequential training (RDP, Kraken2, SINTAX, QIIME2 all single-pass or no training)
