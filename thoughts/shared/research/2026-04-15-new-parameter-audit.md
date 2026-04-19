---
date: 2026-04-15T12:00:00-07:00
researcher: Claude
git_commit: 3d6cb91a16ec0ce3f9a5a9c119a9f966689fba9d
branch: main
repository: oxidtaxa
topic: "Audit of all new parameters added beyond original R IDTAXA: logic, justification, and bugs"
tags: [research, codebase, parameters, bugs, idtaxa, training, classification]
status: complete
last_updated: 2026-04-15
last_updated_by: Claude
---

# Research: New Parameter Audit — Logic, Justification, and Bugs

**Date**: 2026-04-15
**Researcher**: Claude
**Git Commit**: 3d6cb91a16ec0ce3f9a5a9c119a9f966689fba9d
**Branch**: main
**Repository**: oxidtaxa

## Research Question

For each parameter oxidtaxa adds beyond the original R IDTAXA (DECIPHER), analyze the logic and justification. Identify bugs, missing pieces, or correctness issues with extreme detail.

## Summary

11 new parameters were identified and audited. **2 confirmed bugs** were found (one critical), **1 design flaw** where the parameter doesn't achieve its stated goal, and several behavioral differences from R that users should be aware of.

### Severity Classification

| Severity | Parameter | Issue |
|----------|-----------|-------|
| **CRITICAL BUG** | `leave_one_out` | The LOO scaling is a complete no-op due to `vector_sum` self-normalization |
| **BUG** | `rank_thresholds` | Can produce non-contiguous lineages (skipping intermediate ranks) |
| **DESIGN FLAW** | `use_idf_in_training` | Does not actually match classification scoring as documented |
| **OVERFLOW RISK** | `seed_pattern` | No upper bound on seed weight; i32 overflow at weight >= 16 |
| **R DIVERGENCE** | `training_threshold` | Capped decrement + parallel snapshots differ from R's sequential behavior |
| **LIMITATION** | `correlation_aware_features` | Degenerates for binary splits (most common tree structure) |
| **CLEAN** | `sample_exponent` | Matches R exactly (the 0.47 default IS from R's IDTAXA) |
| **CLEAN** | `length_normalize` | Correctly implemented, no bugs found |
| **CLEAN** | `beam_width` | Well-implemented with intentional behavioral differences from greedy |
| **CLEAN** | `descendant_weighting` | Correctly implemented, no division-by-zero risks |
| **CLEAN** | `record_kmers_fraction` | Direct parameterization of R's hardcoded 0.1, correct edge case handling |

---

## Detailed Findings

### 1. `leave_one_out` — CRITICAL BUG: Complete No-Op

**Files**: `src/training.rs:477-508`, `src/matching.rs:31-57`
**Default**: `false`

**Intent**: When enabled, scale down the profile weights for a training sequence's own subtree during fraction learning, approximating the effect of removing that sequence from the profile to reduce self-classification bias for small groups.

**Implementation**: At `training.rs:503-508`, when the current sequence belongs to child subtree `j` and `group_size` is between 2 and 5, all weights in `weights_j` are multiplied by `(group_size - 1) / group_size`.

**The Bug**: `weights_j` is passed to `vector_sum` (`matching.rs:31-57`), which computes:

```
result[rep] = cur_weight / max_weight
```

where `cur_weight = sum(weights[sampled] where matched)` and `max_weight = sum(weights[sampled])`. Both numerator and denominator include the scaling factor. For any constant multiplier `c`:

```
(c * cur) / (c * max) = cur / max
```

**The scale factor cancels out completely.** The `hits[j][rep]` values for the LOO subtree are identical whether or not scaling is applied. Since each subtree's hits are independently normalized ratios, the cross-subtree vote comparison at `training.rs:514-525` sees the same values. **The `leave_one_out` parameter has zero effect on any output.**

**To fix**: The LOO correction needs to modify what the current sequence matches against, not just scale the weights. Options:
1. Actually remove the current sequence's k-mers from the profile before computing matches
2. Apply the correction to the match vector (`matches`) rather than the weights
3. Apply it asymmetrically — scale only `cur_weight` but not `max_weight`

---

### 2. `rank_thresholds` — BUG: Non-Contiguous Lineage Output

**Files**: `src/classify.rs:720-749`
**Default**: `None` (uses single flat `threshold`)

**Intent**: Allow per-rank confidence thresholds instead of a single threshold. `rank_thresholds[i]` applies at depth `i` (0=Root).

**The Bug**: The confidence values in `confidences` are monotonically non-increasing from root to leaf (guaranteed by the construction at `classify.rs:673-686`). With a single flat threshold, `above` is always a contiguous prefix `[0, 1, ..., k]`. But with per-rank thresholds that decrease faster than confidences, intermediate ranks can fail while deeper ranks pass.

**Example**: `confidences = [100, 75, 72]` with `rank_thresholds = [90, 80, 70]`. Rank 1 fails (75 < 80). Rank 2 passes (72 >= 70). Result: `above = [0, 2]`.

**Impact on output** (`classify.rs:735-749`): The result-building code uses `above` as indices into `predicteds`:
```rust
taxon: w.iter().map(|&i| taxa[predicteds[i]].clone()).collect()
```

With `above = [0, 2]`, the output taxon would be `[Root, Mammalia]` skipping `Kingdom` — a taxonomically incoherent lineage.

**Why the original R doesn't have this**: R uses a single flat threshold, so with monotonically non-increasing confidences, `above` is always a contiguous prefix. The non-contiguity path is never exercised.

**To fix**: Either:
1. Enforce that `above` is a contiguous prefix by stopping at the first rank that fails
2. Validate that `rank_thresholds` are non-increasing (matching the confidence monotonicity)

---

### 3. `use_idf_in_training` — DESIGN FLAW: Does Not Match Classification Scoring

**Files**: `src/training.rs:488-512`, `src/classify.rs:190-248, 449-673`
**Default**: `false`

**Intent**: "Makes training scoring match classification scoring" per the docstring at `types.rs:237-238`.

**Reality**: Three distinct scoring mechanisms exist:

| Context | Weights | Formula |
|---------|---------|---------|
| Training default (tree descent) | `profile[j]` | `sum(profile, matched) / sum(profile, all)` — ratio in [0,1] |
| Training with `use_idf_in_training=true` | `profile[j] * idf` | `sum(profile*idf, matched) / sum(profile*idf, all)` — ratio in [0,1] |
| Classification tree descent | `profile[j]` (always) | `sum(profile, matched) / sum(profile, all)` — same as training default |
| Classification leaf phase | `idf` only | Raw IDF sums, normalized externally by `davg` |

The classification **tree descent** always uses raw profiles (no IDF), identical to training's default mode. The classification **leaf phase** uses IDF alone (no profiles), with different normalization.

`use_idf_in_training=true` creates a third hybrid (`profile * idf`) that matches **neither** classification path. If anything, it makes training diverge *further* from classification tree descent (which uses raw profiles).

**Bounds check correctness**: The check `km > 0 && (km as usize) <= prepared.idf_weights.len()` at `training.rs:493-494` is correct. `km` is 1-indexed, `idf_weights` has length `n_kmers = 4^k`, and the access `idf_weights[(km-1)]` correctly maps to `[0, n_kmers-1]`.

---

### 4. `seed_pattern` — OVERFLOW RISK at Weight >= 16

**Files**: `src/kmer.rs:19-42, 414-489`, `src/training.rs:112-159`, `src/classify.rs:97-101`
**Default**: `None` (contiguous k-mers)

**Implementation**: Correctly implemented. The seed pattern is parsed, stored in the model, and re-parsed during classification. K-mer indices range in `[0, 4^weight - 1]`, matching the allocated k-mer space. Training/classification k-mer enumeration is consistent.

**Overflow Risk**: The position weight vector (`pwv`) at `kmer.rs:476-489` computes `4^i` as `i32`. For `weight >= 16`, `4^15 = 1,073,741,824` fits in i32 but `4^16 = 4,294,967,296` overflows. The contiguous path caps at k=13 (auto-k) or k=15 (fixed array size), but the spaced seed path has **no cap on `seed.weight`**. A pattern with 16+ `1`s would cause silent i32 overflow.

**Other Issues**:
- User-supplied `k` is silently ignored when `seed_pattern` is provided (`training.rs:120`, wildcard match `_` on k_param)
- Invalid seed pattern in loaded model causes a panic via `expect()` at `classify.rs:99` instead of returning an error

---

### 5. `training_threshold` — Behavioral Differences from R

**Files**: `src/training.rs:415-630`
**Default**: `0.8`

**Intent**: Parameterize the hardcoded 0.8 vote fraction used in R's fraction-learning loop. The comparison `vote_counts[w] < ((b as f64) * config.training_threshold) as usize` at `training.rs:534` matches R's `hits[w] < B*0.8` exactly for the default value. No off-by-one.

**Differences from R**:

1. **Capped decrement** (`training.rs:575`): `capped_decrement = raw_decrement.min(headroom * 0.5)` limits per-iteration fraction reduction to half the remaining headroom. R decrements unconditionally by `delta/nSeqs[k]` per failure, allowing the fraction to drop below `min_fraction` in a single step.

2. **Parallel snapshot** (`training.rs:441`): Rust clones `fraction` into `fraction_snapshot` before processing sequences in parallel. All sequences in one iteration see the same fractions. R processes sequentially, so later sequences see fraction updates from earlier failures within the same iteration.

3. **Tie-breaking** (`training.rs:530`): Rust's `max_by_key` returns the **last** tied element. R's `which.max` returns the **first**. Different descent paths when multiple children have equal vote counts.

**Hardcoded b=100** (`training.rs:423`): Intentional and matches R. Training always uses 100 bootstrap replicates. Classification's dynamic `b` is a separate mechanism.

---

### 6. `correlation_aware_features` — Degenerates for Binary Splits

**Files**: `src/training.rs:857-975, 693-722`
**Default**: `false`

**Algorithm**: Greedy forward selection maximizing `gain = entropy * (1 - max_abs_pearson_corr)`. Well-implemented with cache-friendly struct-of-arrays layout, precomputed Pearson statistics, and early-exit optimization.

**Degeneration for n_children=2**: Profile vectors have length 2. Pearson correlation of two non-constant 2-element vectors is always exactly ±1. So after the first k-mer is selected, ALL subsequent non-constant-profile candidates get `gain = entropy * (1 - 1.0) = 0`. The correlation penalty becomes meaningless, and selection degenerates to: first k-mer by entropy, then arbitrary ordering among zero-gain candidates.

Binary splits are the most common tree structure in taxonomic trees, so this limitation affects the majority of nodes.

**For n_children=1**: Pearson returns 0.0 (guard `a.len() < 2`), so gain = entropy. Selection is purely by entropy, equivalent to the non-correlation-aware path. Correct behavior.

**NaN risk**: If `n * sum_sq - sum * sum` is negative due to floating-point cancellation, `sqrt()` produces `NaN`. The guard at `training.rs:718` (`denom < 1e-15`) does NOT catch NaN because `NaN < 1e-15` evaluates to `false` in IEEE 754. The function would proceed and produce NaN gain values.

---

### 7. `descendant_weighting` — Clean Implementation

**Files**: `src/training.rs:806-814, 660-689`, `src/types.rs:204-213`
**Default**: `Count` (original R behavior)

Three variants correctly implemented:
- **Count**: Weight = descendant count. Original R behavior.
- **Equal**: Weight = 1.0 per child. `total_weight = n_children`.
- **Log**: Weight = `ln(1 + descendants)`.

**No division-by-zero risk**: Leaves return `d=1` (`training.rs:1056`), so the minimum Log weight is `ln(2) = 0.693`. `total_weight` is always positive.

**merge_sparse_profiles** (`training.rs:660-689`): Correctly implements weighted k-way merge. The denominator is always `total_weight` (not just the weight of children possessing the k-mer), so absent k-mers correctly dilute the average.

**Scope**: Weighting affects only the merged profile `q` used for cross-entropy feature selection. Individual child profiles stored in `DecisionNode.profiles` are unaffected. However, the weighted `q` is returned upward as the current node's profile (line 1036), so weighting choices cascade up the tree.

---

### 8. `record_kmers_fraction` — Clean Parameterization

**Files**: `src/training.rs:848-855, 976-1001, 857-975`
**Default**: `0.10` (matches R's hardcoded `ceiling(0.1 * max(lengths))`)

Direct parameterization of R's hardcoded `0.1`. `record_kmers = ceil(max_nonzero * fraction)` where `max_nonzero` is the maximum sparse profile length across children.

**Edge cases handled correctly**:
- `max_nonzero = 0`: `record_kmers = 0`, loops exit immediately, empty `DecisionNode` is created, all downstream consumers (classify, train) handle empty `keep` with early breaks.
- `record_kmers` exceeds available k-mers: Both round-robin and correlation-aware paths have exhaustion guards.

---

### 9. `sample_exponent` — Matches R Exactly (Not Actually New)

**Files**: `src/classify.rs:86-163, 870-899`
**Default**: `0.47`

**Important finding**: This parameter is NOT new — it parameterizes R's `samples=L^0.47` default from `DECIPHER::IdTaxa`. The formula, the raw-vs-unique count distinction, the `min_s` computation, and the `B = 5*U/S` conservation law all match R exactly.

**One difference**: Rust adds `.max(1.0)` to `b_values` computation (`classify.rs:129`) to guarantee at least 1 bootstrap replicate. R's `as.integer()` can produce 0, which is actually a bug in R that Rust fixes.

**`pbinom` implementation** (`classify.rs:887-899`): Numerically safe for the input ranges used. Called with `n <= 100` for the first call and `k = 0` for the second call. No overflow risk for realistic inputs.

---

### 10. `length_normalize` — Clean Implementation

**Files**: `src/classify.rs:557-571`
**Default**: `false`

**Formula**: `norm_factor = sqrt(n_unique / avg_unique)` where `n_unique` is a training sequence's distinct k-mer count and `avg_unique` is the mean across all `keep` sequences.

**Correctness**:
- `ls` measures unique k-mer counts (post-dedup), a reasonable proxy for sequence information content.
- Guards prevent division by zero (both `n_unique > 0` and `avg_unique > 0` checked).
- Uniform-length case correctly produces identity (norm_factor = 1.0).
- `davg` (the denominator used later for confidence) is computed independently from raw sampling, NOT affected by normalization. This is correct — normalization adjusts training-sequence-side scores while `davg` represents query-side expected maximum.

**Interaction with `full_length`**: When `full_length` pre-filters to similar-length sequences, normalization has diminished effect since `norm_factor` clusters around 1.0. The two features are complementary, not conflicting.

---

### 11. `beam_width` — Well-Implemented with Intentional Differences

**Files**: `src/classify.rs:254-444`
**Default**: `1` (greedy, identical code path to original)

**Key design decisions**:
- `beam_width=1` dispatches to the greedy code path via `beam_width > 1` check (line 178). Guaranteed equivalence by construction, not algorithmic coincidence.
- Scores are multiplicative products of vote fractions, used only for pruning, NOT for final selection.
- Final candidate selected by `similarity` from `leaf_phase_score`, not beam score.
- All candidates share one RNG, which is acceptable since runner-ups inherit parent voting results without additional RNG calls.

**Behavioral difference from greedy at `min_descend`**: In greedy, if multiple children exceed `min_descend`, it's treated as ambiguous (terminal). In beam, only the top child is checked — beam always descends when the top child is confident, adding others as runner-up candidates. This is intentional: beam search is designed to explore alternatives that greedy would reject.

---

## Code References

- `src/types.rs:216-272` — `TrainConfig` with all training parameters
- `src/types.rs:275-312` — `ClassifyConfig` with all classification parameters
- `src/training.rs:415-630` — `_learn_fractions_inner` (fraction learning)
- `src/training.rs:747-1058` — `create_tree` (tree construction with feature selection)
- `src/classify.rs:168-251` — `classify_one_pass` (greedy descent)
- `src/classify.rs:254-444` — `classify_one_pass_beam` (beam search)
- `src/classify.rs:449-752` — `leaf_phase_score` (leaf-phase scoring)
- `src/matching.rs:31-57` — `vector_sum` (self-normalizing bootstrap scoring)
- `src/kmer.rs:414-489` — Spaced seed k-mer enumeration
- `src/lib.rs:62-360` — Python bindings with parameter defaults

## Open Questions

1. **leave_one_out fix strategy**: Should the fix modify the match vector, the normalization, or actually recompute profiles? The match vector approach (subtracting the sequence's contribution from `matches`) would be most faithful to LOO semantics but requires tracking per-sequence k-mer contributions through the profile.

2. **rank_thresholds contiguity**: Should the fix enforce contiguous prefixes (simpler, avoids taxonomic nonsense) or explicitly handle gaps (more flexible but harder to interpret biologically)?

3. **use_idf_in_training redesign**: If the goal is training/classification alignment, should training tree descent also use IDF-only weights (matching classification leaf phase)? Or should classification tree descent use profiles (matching training)?

4. **correlation_aware_features for binary splits**: Is there a meaningful diversity metric for 2-element vectors that doesn't degenerate? Perhaps mutual information or a different distance metric?
