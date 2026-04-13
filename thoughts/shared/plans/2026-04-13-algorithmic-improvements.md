# Algorithmic Improvements Implementation Plan

## Overview

Implement 7 algorithmic improvements to oxidtaxa's training and classification, each behind a config flag that defaults to current behavior. This allows A/B testing any combination of improvements against the existing baseline without breaking backward compatibility.

## Current State Analysis

The codebase has two config structs (`TrainConfig` at `types.rs:100-113`, `ClassifyConfig` at `types.rs:132-164`) that already follow the pattern of adding optional features with backward-compatible defaults (e.g., `length_normalize`, `rank_thresholds`, `seed_pattern`). Python bindings in `lib.rs:16-92` expose these as keyword arguments with defaults.

The existing test infrastructure includes:
- Golden tests against R output (`tests/test_training.rs`, `tests/test_classify.rs`)
- Real-data baseline test (`tests/test_baseline_1k.rs`) — 87-93% path agreement with R due to batch fraction learning divergence
- Criterion benchmarks (`benches/oxidtaxa_bench.rs`)

### Key Discoveries:
- `TrainConfig` fields flow through `learn_taxa()` at `training.rs:16`
- `ClassifyConfig` fields flow through `id_taxa()` at `classify.rs:33` → `classify_one_pass()` at `classify.rs:165`
- Python `train()` builds `TrainConfig` at `lib.rs:41-46`
- Python `classify()` builds `ClassifyConfig` at `lib.rs:102-111`
- The baseline test (`test_baseline_1k.rs:78-86`) uses `TrainConfig::default()` and `ClassifyConfig::default()` — new defaults must preserve existing behavior

## Desired End State

After implementation, users can test any combination of improvements from Python:
```python
# Train with improvements
oxidtaxa.train("ref.fasta", "tax.tsv", "model.bin",
    leave_one_out=True,
    training_threshold=0.98,
    use_idf_in_training=True,
    descendant_weighting="equal")

# Classify with improvements  
oxidtaxa.classify("query.fasta", "model.bin",
    beam_width=3,
    length_normalize=True)
```

All existing tests pass unchanged with default configs. Each improvement has its own test demonstrating the flag works.

### Verification:
- `cargo test` passes (all golden tests, baseline test, new per-feature tests)
- `cargo bench` shows no regression for default config
- Each flag can be toggled independently via Python

## What We're NOT Doing

- Computational/speed optimizations (rolling hash, bitset k-mers, counting sort, etc.) — separate effort
- Changing any defaults that would break backward compatibility (except length_normalize, gated on benchmarking)
- Modifying the R-compatible PRNG or deterministic mode
- Adding new output formats or changing ClassificationResult structure

## Implementation Approach

Each improvement is a self-contained phase that adds a config field, implements the logic behind a flag, and adds a test. Phases are ordered by complexity (simplest first) and are independent — any can be skipped or reordered.

The pattern for each phase is:
1. Add field to `TrainConfig` or `ClassifyConfig` with backward-compatible default
2. Add the field to `Default` impl
3. Implement the logic, gated on the new field
4. Add Python binding parameter
5. Add test

---

## Phase 1: Training Threshold Match

### Overview
Replace the hardcoded 0.8 threshold during training's fraction-learning descent with a configurable value. Currently training considers a descent "correct" at 80% bootstrap agreement, but classification requires 98% (`min_descend`). This means fractions are learned under a more permissive criterion than what's actually used.

### Changes Required:

#### 1. Config field
**File**: `src/types.rs`
**Changes**: Add `training_threshold` to `TrainConfig`

```rust
// In TrainConfig struct (after record_kmers_fraction):
/// Bootstrap vote fraction required to descend during fraction learning.
/// Default 0.8 matches R's hardcoded behavior. Set to match min_descend
/// (e.g., 0.98) for consistent training/classification thresholds.
pub training_threshold: f64,
```

```rust
// In Default impl:
training_threshold: 0.8,
```

#### 2. Use the config value
**File**: `src/training.rs:356`
**Changes**: Replace hardcoded `0.8` with config value

Current:
```rust
if vote_counts[w] < ((b as f64) * 0.8) as usize {
```

New:
```rust
if vote_counts[w] < ((b as f64) * config.training_threshold) as usize {
```

#### 3. Python binding
**File**: `src/lib.rs`
**Changes**: Add `training_threshold` parameter to `train()` function signature and pass through to `TrainConfig`.

#### 4. Test
**File**: `tests/test_training.rs`
**Changes**: Add test that trains with `training_threshold: 0.98` and verifies the model produces different (likely more) problem sequences than with 0.8.

### Success Criteria:
- [x] `cargo test` passes — all existing golden tests still pass with default 0.8
- [x] New test demonstrates the flag has an observable effect
- [x] Python: `oxidtaxa.train(..., training_threshold=0.98)` works

---

## Phase 2: Descendant Weighting Alternatives

### Overview
The merged profile `q` in `create_tree` weights children by descendant count (`training.rs:579`). Add configurable weighting strategies.

### Changes Required:

#### 1. Config field
**File**: `src/types.rs`
**Changes**: Add enum and field

```rust
/// Strategy for weighting child profiles during feature selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DescendantWeighting {
    /// Weight by raw descendant count (original IDTAXA behavior).
    Count,
    /// Equal weight per immediate child (1/n_children each).
    Equal,
    /// Weight by log(1 + descendants).
    Log,
}

// In TrainConfig:
/// How to weight child profiles when computing the merged profile for
/// cross-entropy feature selection. Default: Count (original behavior).
pub descendant_weighting: DescendantWeighting,
```

```rust
// In Default impl:
descendant_weighting: DescendantWeighting::Count,
```

#### 2. Apply weighting
**File**: `src/training.rs:578-582`
**Changes**: Compute `desc_weights` based on the config

Current:
```rust
let desc_weights: Vec<f64> = descendants.iter().map(|&d| d as f64).collect();
```

New:
```rust
let desc_weights: Vec<f64> = match descendant_weighting {
    DescendantWeighting::Count => descendants.iter().map(|&d| d as f64).collect(),
    DescendantWeighting::Equal => vec![1.0; descendants.len()],
    DescendantWeighting::Log => descendants.iter().map(|&d| (1.0 + d as f64).ln()).collect(),
};
```

This requires threading `descendant_weighting` through `create_tree`. Add it as a parameter (replacing the separate `max_children` and `record_kmers_fraction` params with a reference to the full config, or just add the field).

#### 3. Python binding
**File**: `src/lib.rs`
**Changes**: Add `descendant_weighting: &str` parameter (default `"count"`), parse to enum.

#### 4. Test
**File**: `tests/test_training.rs`
**Changes**: Train with each weighting strategy, verify they produce different decision k-mer selections.

### Success Criteria:
- [x] `cargo test` passes — default `Count` matches existing golden tests
- [x] Training with `Equal` and `Log` produces different decision_kmers
- [x] Python: `oxidtaxa.train(..., descendant_weighting="equal")` works

---

## Phase 3: IDF Scoring During Training

### Overview
During fraction learning (`training.rs:310-378`), the tree descent uses profile-frequency weights for scoring via `vector_sum`. Classification uses IDF weights. Add an option to use IDF weights during training too, so fractions are calibrated under the same scoring regime.

### Changes Required:

#### 1. Config field
**File**: `src/types.rs`
**Changes**: Add to `TrainConfig`

```rust
/// Use IDF weights (instead of profile weights) during the fraction-learning
/// tree descent. Makes training scoring match classification scoring.
/// Default false (original behavior uses profile weights).
pub use_idf_in_training: bool,
```

#### 2. Implement IDF-weighted training descent
**File**: `src/training.rs`
**Changes**: The IDF weights are computed at `training.rs:439-482`, AFTER the fraction learning loop. To use them during training, we need to either:

**Option A** (simpler): Move IDF computation before the fraction loop. IDF doesn't depend on fractions, only on k-mer frequencies and class counts, so this is safe.

**Option B** (minimal diff): Compute IDF weights early when the flag is on, skip when off.

Go with Option A. Move lines 440-482 (IDF computation) to before line 276 (fraction loop start). Then in the training descent loop, when `use_idf_in_training` is true, replace the profile-weight `vector_sum` call with an IDF-weighted equivalent.

The key change in the inner loop (`training.rs:329-333`):

Current:
```rust
let matches = int_match(&dk.keep, &kmers[i]);
let mut hits = vec![vec![0.0f64; b]; subtrees.len()];
for (j, _subtree) in subtrees.iter().enumerate() {
    hits[j] = vector_sum(&matches, &dk.profiles[j], &sampling, b);
}
```

When `use_idf_in_training` is true, instead of using `dk.profiles[j]` as weights in `vector_sum`, use IDF weights for the kept k-mers:
```rust
if config.use_idf_in_training {
    // Build IDF weights for the decision k-mers at this node
    let idf_for_keep: Vec<f64> = dk.keep.iter()
        .map(|&km| if km > 0 && (km as usize) <= idf_weights.len() {
            idf_weights[(km - 1) as usize]
        } else { 0.0 })
        .collect();
    for (j, _subtree) in subtrees.iter().enumerate() {
        // Multiply profile match by IDF weight
        let combined: Vec<f64> = matches.iter().zip(idf_for_keep.iter())
            .zip(dk.profiles[j].iter())
            .map(|((&m, &idf), &prof)| if m { idf * prof } else { 0.0 })
            .collect();
        // ... accumulate into hits via sampling
    }
}
```

Actually, this needs more thought. The `vector_sum` function takes `matches` (bool), `weights` (f64 per k-mer), and `sampling` (which k-mers to sample per replicate). The `weights` are currently the profile values. With IDF, we'd want weights = IDF * profile (or just IDF, treating profile as the match probability).

Simplest approach: when the flag is on, multiply each profile weight by the IDF weight for that k-mer, producing a combined weight. This way the scoring considers both the k-mer's discriminative power (profile) and its global rarity (IDF).

```rust
let weights_for_j: Vec<f64> = if config.use_idf_in_training {
    dk.profiles[j].iter().zip(dk.keep.iter())
        .map(|(&prof, &km)| {
            let idf = if km > 0 && (km as usize) <= idf_weights.len() {
                idf_weights[(km - 1) as usize]
            } else { 0.0 };
            prof * idf
        })
        .collect()
} else {
    dk.profiles[j].clone()
};
hits[j] = vector_sum(&matches, &weights_for_j, &sampling, b);
```

#### 3. Python binding
**File**: `src/lib.rs`
**Changes**: Add `use_idf_in_training: bool` parameter (default `false`).

#### 4. Test
**File**: `tests/test_training.rs`
**Changes**: Train with flag on, verify model trains successfully and produces different fractions than with flag off.

### Success Criteria:
- [x] `cargo test` passes — default `false` matches existing behavior
- [x] Training with `use_idf_in_training=true` produces valid model (different fractions on ambiguous data)
- [x] Python: `oxidtaxa.train(..., use_idf_in_training=True)` works

---

## Phase 4: Length Normalization Default

### Overview
The Rust port added `length_normalize` but defaults to `false`. Benchmark to determine if `true` is better, and if so, change the default.

### Changes Required:

#### 1. Benchmark first
Run the baseline test with `length_normalize: true` vs `false` and compare path agreement and confidence calibration against R baseline. Also test on the real 1K dataset.

#### 2. If benchmarking supports it
**File**: `src/types.rs:160`
**Changes**: Flip default

```rust
length_normalize: true,
```

#### 3. Update baseline test thresholds
**File**: `tests/test_baseline_1k.rs`
**Changes**: The baseline was generated with R (which has no length normalization). If we change the default, the baseline comparison thresholds may need adjustment, or we run the baseline with `length_normalize: false` to keep it as a pure R-comparison test.

Better approach: keep the baseline test using `length_normalize: false` explicitly, and add a separate test that verifies `length_normalize: true` doesn't crash and produces reasonable results.

### Success Criteria:
- [x] Decision documented: leave default as `false` (no change)
- [x] All tests pass with current default

---

## Phase 5: Leave-One-Out Training

### Overview
During fraction learning, each sequence is classified against profiles that include itself. For small groups (1-3 sequences), this creates a positive bias. Add an option to exclude the test sequence's k-mer contribution from node profiles during the training descent.

### Changes Required:

#### 1. Config field
**File**: `src/types.rs`
**Changes**: Add to `TrainConfig`

```rust
/// Exclude each sequence from its own node's profile during fraction
/// learning (leave-one-out). Reduces self-classification bias for small
/// groups. Default false (original behavior).
pub leave_one_out: bool,
```

#### 2. Implement profile adjustment
**File**: `src/training.rs`
**Changes**: This is the most involved change. During the fraction-learning loop, when classifying sequence `i`:

The current code uses `int_match(&dk.keep, &kmers[i])` and then `vector_sum(&matches, &dk.profiles[j], &sampling, b)`. The `dk.profiles[j]` are the stored decision node profiles computed by `create_tree`.

For leave-one-out, we need to adjust the profiles at each node to exclude sequence `i`'s contribution. The profiles are normalized k-mer frequencies. To remove sequence `i`:

1. During `create_tree`, store the total count (not just the normalized frequency) alongside the profile OR store the sequence count per node so we can reverse the normalization.

Actually, a simpler approach: the `DecisionNode.profiles` are derived from the sparse profiles computed in `create_tree`. The sparse profiles are `count / total_count` for each k-mer. To subtract one sequence:

- For each k-mer in the kept set: if sequence `i` has this k-mer, the adjusted profile = `(original_count - 1) / (total_count - seq_i_contribution)`.

But we don't store `original_count` or `total_count` — just the normalized proportion.

**Better approach**: Store per-node sequence counts alongside profiles in `DecisionNode`:

```rust
pub struct DecisionNode {
    pub keep: Vec<i32>,
    pub profiles: Vec<Vec<f64>>,
    /// Total k-mer count per child subtree (for leave-one-out adjustment).
    /// Only populated when leave_one_out is enabled during training.
    pub child_kmer_totals: Option<Vec<f64>>,
}
```

During `create_tree`, when building leaf profiles, store the raw total alongside the normalized profile. When LOO is enabled in the fraction loop:

For sequence `i` at node `k`:
1. Determine which child subtree sequence `i` belongs to
2. For each kept k-mer: if `kmers[i]` contains it, the adjusted weight = `profile[j][kmer] - 1.0 / child_kmer_totals[j]`
3. Use the adjusted weights in `vector_sum`

This is approximate (it adjusts the frequencies linearly rather than exactly renormalizing), but for groups with >3 sequences the approximation is excellent, and for groups with 1-2 sequences the adjustment is most needed.

**Even simpler approach**: Only apply LOO when the group is small (≤ threshold, e.g., ≤5 sequences). For large groups the bias is negligible. This avoids the overhead of tracking totals for every node.

```rust
if config.leave_one_out {
    // Check if sequence i's group at this node is small enough to bother
    let child_idx = subtrees.iter().position(|&s| 
        classes[i].starts_with(&end_taxonomy[s])
    );
    if let Some(ci) = child_idx {
        if let Some(ref seqs) = sequences_per_node[subtrees[ci]] {
            if seqs.len() <= 5 {
                // Adjust profile weights to exclude sequence i
                // ...
            }
        }
    }
}
```

Implementation detail: For the simplest first version, when LOO is on and the group is small, skip the sequence entirely (treat it as correctly classified). This avoids profile surgery and still eliminates the worst-case bias. A more sophisticated version can do the actual profile adjustment later.

Actually, the simplest correct approach:

When `leave_one_out` is true and sequence `i` is in a group with n_seqs ≤ `loo_threshold` (e.g., 5):
- Mark the sequence's `matches` vector: for each decision k-mer, if sequence `i` has this k-mer AND the group only has n_seqs sequences containing it, set the profile weight to 0 for that k-mer. This effectively removes the sequence's contribution.

This is getting complex. Let me propose a two-pass approach:

**Pass 1 (this phase)**: Simple LOO — when enabled, sequences in groups of size 1 are automatically marked correct (they can't be meaningfully self-classified). Sequences in groups of size 2-3 use a reduced profile weight of `(n-1)/n * original_weight`. This is an approximation but captures the main effect.

**Pass 2 (future)**: Exact LOO with stored totals.

#### 3. Python binding
**File**: `src/lib.rs`
**Changes**: Add `leave_one_out: bool` parameter (default `false`).

#### 4. Test
Train with a dataset known to have singleton taxa. Verify that `leave_one_out=true` produces fewer problem sequences for those singletons (since it won't try to self-classify them).

### Success Criteria:
- [x] `cargo test` passes — default `false` matches existing behavior
- [x] Singletons and small groups handled correctly with flag on
- [x] Python: `oxidtaxa.train(..., leave_one_out=True)` works

---

## Phase 6: Beam Search for Tree Descent

### Overview
Replace greedy single-path tree descent with configurable beam search. At beam_width=1, behavior is identical to current. At beam_width>1, the classifier maintains multiple candidate paths and picks the best after leaf-phase scoring.

### Changes Required:

#### 1. Config field
**File**: `src/types.rs`
**Changes**: Add to `ClassifyConfig`

```rust
/// Number of candidate paths to maintain during tree descent.
/// 1 = greedy descent (original behavior). Higher values explore
/// alternative paths at ambiguous nodes. Default 1.
pub beam_width: usize,
```

```rust
// In Default impl:
beam_width: 1,
```

#### 2. Refactor classify_one_pass
**File**: `src/classify.rs`
**Changes**: This is the largest change. Split `classify_one_pass` into two stages:

**Stage 1: Tree descent** — returns candidate (node, w_indices) pairs

```rust
struct BeamCandidate {
    node: usize,
    w_indices: Vec<usize>,
    /// Product of winning vote fractions along the path.
    score: f64,
}
```

**New function**: `tree_descent_beam`

```rust
fn tree_descent_beam(
    my_kmers: &[i32],
    beam_width: usize,
    ts: &TrainingSet,
    config: &ClassifyConfig,
    rng: &mut RRng,
) -> Vec<BeamCandidate> {
    let mut active = vec![BeamCandidate { node: 0, w_indices: vec![], score: 1.0 }];

    loop {
        let mut next_candidates: Vec<BeamCandidate> = Vec::new();
        let mut any_expanded = false;

        for candidate in &active {
            let k_node = candidate.node;
            let subtrees = &ts.children[k_node];
            let dk = &ts.decision_kmers[k_node];

            if dk.is_none() || ts.fraction[k_node].is_none() {
                // Can't descend further — keep as terminal candidate
                next_candidates.push(BeamCandidate {
                    node: k_node,
                    w_indices: (0..subtrees.len()).collect(),
                    score: candidate.score,
                });
                continue;
            }

            let dk = dk.as_ref().unwrap();
            let n = dk.keep.len();
            if n == 0 || subtrees.len() <= 1 {
                // Single child or no k-mers — descend without vote
                if subtrees.len() == 1 && !ts.children[subtrees[0]].is_empty() {
                    next_candidates.push(BeamCandidate {
                        node: subtrees[0],
                        w_indices: vec![],
                        score: candidate.score,
                    });
                    any_expanded = true;
                } else {
                    next_candidates.push(BeamCandidate {
                        node: k_node,
                        w_indices: (0..subtrees.len()).collect(),
                        score: candidate.score,
                    });
                }
                continue;
            }

            // Vote at this node (same logic as current)
            let frac = ts.fraction[k_node].unwrap();
            let s_dk = ((n as f64) * frac).ceil() as usize;
            let b = config.bootstraps; // use classification B, not training B
            let sampling = rng.sample_int_replace(n, s_dk * b);
            let matches = int_match(&dk.keep, my_kmers);

            let n_sub = subtrees.len();
            let mut hits_flat = vec![0.0f64; n_sub * b];
            for j in 0..n_sub {
                let row = vector_sum(&matches, &dk.profiles[j], &sampling, b);
                hits_flat[j * b..(j + 1) * b].copy_from_slice(&row);
            }
            let mut vote_counts = vec![0usize; n_sub];
            for rep in 0..b {
                let max_val = (0..n_sub).map(|j| hits_flat[j * b + rep]).fold(0.0f64, f64::max);
                if max_val > 0.0 {
                    for j in 0..n_sub {
                        if hits_flat[j * b + rep] == max_val { vote_counts[j] += 1; }
                    }
                }
            }

            // For beam search: keep top beam_width children that got any votes
            let total_votes: usize = vote_counts.iter().sum();
            if total_votes == 0 {
                next_candidates.push(BeamCandidate {
                    node: k_node,
                    w_indices: (0..n_sub).collect(),
                    score: candidate.score,
                });
                continue;
            }

            // Collect children with votes, sorted by vote count descending
            let mut children_by_votes: Vec<(usize, usize)> = vote_counts.iter()
                .enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(j, &c)| (j, c))
                .collect();
            children_by_votes.sort_by(|a, b| b.1.cmp(&a.1));

            // The top child must pass min_descend to be a "confident" descent
            let top_vote_frac = children_by_votes[0].1 as f64 / b as f64;

            if top_vote_frac >= config.min_descend {
                // Confident winner — descend it (and keep runner-ups for beam)
                let winner_idx = children_by_votes[0].0;
                let winner_child = subtrees[winner_idx];

                if ts.children[winner_child].is_empty() {
                    // Winner is a leaf — terminal candidate
                    next_candidates.push(BeamCandidate {
                        node: k_node,
                        w_indices: vec![winner_idx],
                        score: candidate.score * top_vote_frac,
                    });
                } else {
                    // Descend into winner
                    next_candidates.push(BeamCandidate {
                        node: winner_child,
                        w_indices: vec![],
                        score: candidate.score * top_vote_frac,
                    });
                    any_expanded = true;
                }

                // Also keep runner-ups for beam (if beam_width > 1)
                for &(j, votes) in children_by_votes.iter().skip(1) {
                    let vote_frac = votes as f64 / b as f64;
                    let child = subtrees[j];
                    if ts.children[child].is_empty() {
                        next_candidates.push(BeamCandidate {
                            node: k_node,
                            w_indices: vec![j],
                            score: candidate.score * vote_frac,
                        });
                    } else {
                        next_candidates.push(BeamCandidate {
                            node: child,
                            w_indices: vec![],
                            score: candidate.score * vote_frac,
                        });
                        any_expanded = true;
                    }
                }
            } else {
                // No confident winner — terminal (same as current fallback)
                // Apply current fallback logic (50% check, etc.)
                next_candidates.push(BeamCandidate {
                    node: k_node,
                    w_indices: /* current fallback w_indices logic */,
                    score: candidate.score * top_vote_frac,
                });
            }
        }

        // Prune to beam_width
        next_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        next_candidates.truncate(beam_width);
        active = next_candidates;

        if !any_expanded { break; }
    }

    active
}
```

**Stage 2: Leaf-phase scoring for each candidate**

For each `BeamCandidate`, run the existing leaf-phase logic (gather training sequences, bootstrap matching, confidence computation). Pick the candidate with the highest similarity or confidence.

```rust
fn classify_one_pass(...) -> Option<(ClassificationResult, f64)> {
    if config.beam_width <= 1 {
        // Current logic unchanged
        return classify_one_pass_greedy(...);
    }

    let candidates = tree_descent_beam(my_kmers, config.beam_width, ts, config, rng);

    let mut best_result: Option<(ClassificationResult, f64)> = None;
    for candidate in &candidates {
        if let Some((result, similarity)) = leaf_phase_score(
            candidate.node, &candidate.w_indices,
            my_kmers, s, b, ts, config, full_length, ls, rng,
        ) {
            match &best_result {
                None => best_result = Some((result, similarity)),
                Some((_, best_sim)) if similarity > *best_sim => {
                    best_result = Some((result, similarity));
                }
                _ => {}
            }
        }
    }
    best_result
}
```

The `leaf_phase_score` function extracts the existing code from `classify_one_pass` starting at line 250 (gather training sequences) through line 498 (return result).

#### 3. Python binding
**File**: `src/lib.rs`
**Changes**: Add `beam_width: usize` parameter (default `1`).

#### 4. Test
- Test that `beam_width=1` produces identical results to current implementation (regression test)
- Test that `beam_width=3` on the tied-species dataset resolves ties differently
- Test on the baseline dataset to measure path agreement improvement

### Success Criteria:
- [x] `beam_width=1` produces bit-identical results to current implementation
- [x] `beam_width>1` works without panics on all test datasets
- [x] Python: `oxidtaxa.classify(..., beam_width=3)` works
- [ ] Performance: beam_width=3 is no more than 3x slower than beam_width=1

---

## Phase 7: Correlation-Aware Feature Selection

### Overview
Replace the independent round-robin k-mer selection with greedy forward selection that accounts for k-mer correlation. At each step, select the k-mer that maximizes conditional information gain given already-selected k-mers.

### Changes Required:

#### 1. Config field
**File**: `src/types.rs`
**Changes**: Add to `TrainConfig`

```rust
/// Use correlation-aware greedy feature selection instead of independent
/// round-robin. Selects k-mers that maximize conditional information gain.
/// Produces a more efficient feature set but slower to train. Default false.
pub correlation_aware_features: bool,
```

#### 2. Implement greedy selection
**File**: `src/training.rs`
**Changes**: Replace the round-robin loop (`training.rs:630-649`) with a greedy forward selection when the flag is on.

```rust
if config.correlation_aware_features {
    // Greedy forward selection
    let mut selected: Vec<usize> = Vec::with_capacity(record_kmers);
    let mut selected_set: HashSet<usize> = HashSet::new();

    // Pre-compute: for each k-mer, which child profiles have non-zero values
    // This lets us compute conditional information gain efficiently

    while selected.len() < record_kmers {
        let mut best_kmer = None;
        let mut best_gain = f64::NEG_INFINITY;

        // For each candidate k-mer not yet selected
        for child_h in &sorted_h {
            for &(kmer_idx, base_entropy) in child_h {
                if selected_set.contains(&kmer_idx) { continue; }

                // Compute conditional gain: how much new discrimination
                // does this k-mer add given the already-selected set?
                // 
                // Approximation: penalize k-mers whose profile vectors
                // are highly correlated with already-selected k-mers.
                // gain = base_entropy * (1 - max_correlation_with_selected)
                let mut max_corr: f64 = 0.0;
                for &sel in &selected {
                    // Correlation between kmer_idx and sel across child profiles
                    let corr = profile_correlation(&profiles, kmer_idx, sel);
                    if corr > max_corr { max_corr = corr; }
                }
                let gain = base_entropy * (1.0 - max_corr);
                if gain > best_gain {
                    best_gain = gain;
                    best_kmer = Some(kmer_idx);
                }
            }
        }

        match best_kmer {
            Some(km) => {
                selected.push(km);
                selected_set.insert(km);
            }
            None => break,
        }
    }
    // Use `selected` instead of `keep_set`
} else {
    // Current round-robin logic (unchanged)
}
```

The `profile_correlation` helper computes Pearson correlation between two k-mers across child profiles (treating each child's profile value as a data point).

**Performance note**: This is O(record_kmers * total_candidate_kmers * n_selected) per node. For typical values (record_kmers ~50, candidates ~500, children ~5), this is ~1.25M operations per node — fast enough. For wide nodes at the top of the tree it could be slower. The flag lets users choose.

#### 3. Python binding
**File**: `src/lib.rs`
**Changes**: Add `correlation_aware_features: bool` parameter (default `false`).

#### 4. Test
Train with flag on/off, verify that the selected k-mers differ and that the model produces valid classifications.

### Success Criteria:
- [x] `cargo test` passes — default `false` matches existing behavior
- [x] Flag on produces different decision k-mer selections
- [x] Classification still works with models trained with this flag
- [x] Python: `oxidtaxa.train(..., correlation_aware_features=True)` works

---

## Testing Strategy

### Regression Tests:
- All existing golden tests pass with default configs (verifies backward compatibility)
- Baseline 1K test passes with default configs

### Per-Feature Tests:
Each phase adds a test that:
1. Trains and/or classifies with the flag ON
2. Trains and/or classifies with the flag OFF (default)
3. Asserts the results are different (the flag has an observable effect)
4. Asserts the "on" results are reasonable (no panics, valid taxonomy paths, confidences in [0, 100])

### Integration Test:
Add a test that enables ALL flags simultaneously and verifies the full pipeline works:
```rust
#[test]
fn test_all_improvements_combined() {
    let train_config = TrainConfig {
        training_threshold: 0.98,
        descendant_weighting: DescendantWeighting::Equal,
        use_idf_in_training: true,
        leave_one_out: true,
        correlation_aware_features: true,
        ..Default::default()
    };
    let classify_config = ClassifyConfig {
        beam_width: 3,
        length_normalize: true,
        ..Default::default()
    };
    // Train, classify, verify results are valid
}
```

### Benchmarking:
Use the 1K baseline dataset to compare each flag against the R baseline:
- Path agreement % (currently 87-93%)
- Mean/max confidence difference
- OC error rate at threshold=60
- Number of problem sequences/groups

This can be done via the `examples/eval_training.rs` pattern — an example binary that takes flags and prints metrics.

## Performance Considerations

- Phases 1-4: Negligible performance impact
- Phase 5 (LOO): Adds per-sequence overhead in the fraction loop, but only for small groups. Should be < 5% overhead.
- Phase 6 (Beam search): Linear scaling with beam_width. beam_width=3 ≈ 3x slower tree descent, but tree descent is a small fraction of total classification time (leaf-phase matching dominates). Expected ~20-50% total slowdown at beam_width=3.
- Phase 7 (Correlation features): Slower training (quadratic in feature count per node), but training is one-time. No impact on classification speed.

## References

- Research: `thoughts/shared/research/2026-04-13-algorithmic-improvements.md`
- Original paper: Murali et al. 2018, Microbiome 6:140
- R reference: `reference/R_orig/LearnTaxa.R`, `reference/R_orig/IdTaxa.R`
- Existing Rust improvements: `thoughts/shared/plans/2026-04-06-oxidtaxa-features.md`
