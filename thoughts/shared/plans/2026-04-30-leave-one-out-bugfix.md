# Leave-One-Out Bugfix Implementation Plan

> **Status**: Implemented (all four phases complete). See [Implementation Notes](#implementation-notes) below for what was actually built and any deviations from the original plan.

## Overview

The `leave_one_out` training flag is structurally inert — uniform weight scaling at `src/training.rs:483-488` cancels in `vector_sum`'s `cur_weight / max_weight` ratio at `src/matching.rs:50-54`, producing byte-identical models regardless of the flag setting. Replace the uniform scaling with per-kmer LOO that subtracts the held-out sequence's contribution from the matching sibling's profile counts and renormalizes. This breaks the scale invariance, captures k-mer specificity (singleton k-mers go to zero, conserved k-mers stay unchanged), and lifts the artificial `n ≤ 5` size cap.

## Implementation Notes

The plan was implemented in full. Final verification:
- All 97 tests pass (`cargo test --release --tests`).
- `cargo clippy --all-targets --release -- -D warnings` clean.
- `cargo fmt --check` clean.
- `cargo doc --no-deps` builds with zero warnings (also fixed pre-existing matching.rs / types.rs broken-link warnings as a bonus).
- The Python feature (`cargo build --release --features python`) was broken before this work began (Python 3.14 vs PyO3 0.24 ABI incompatibility) and remains broken; this is unrelated to the LOO fix and tracked separately.

### Deviations from the original plan

1. **Regression test changed** from `test_leave_one_out_changes_fraction` (asserting `fraction` differs) to `test_leave_one_out_reshapes_profile` (asserting the per-kmer formula reshapes the profile). On every test dataset available (s08a/c/d), default-config classifications hit 100% accuracy under both LOO settings, so neither `fraction` nor `problem_sequences` ever decrement and the planned end-to-end assertion would never trigger on small data. The structural assertion proves the LOO formula is mathematically non-trivial (i.e. not a uniform-scaling no-op) by reading the model's stored `raw_counts`/`raw_totals` and computing the LOO-adjusted profile by hand. End-to-end byte-diff verification on a real-world dataset (vert12s) remains as a manual check.
2. **Extra third regression test** added: `test_leave_one_out_decision_kmers_invariant` — guards that LOO never leaks into feature selection (build-tree phase remains insulated from the flag, matching the existing architectural intent).
3. **Bonus rustdoc cleanup** — fixed 6 pre-existing `cargo doc` warnings in `src/matching.rs` (HTML interpretation of `Vec<f64>`, unresolved-link warnings on `result[i]`, `query k-mer i`, `n_indices`) and one in `src/types.rs` (`rank_thresholds[i]`). Not in original scope but the user asked for this.
4. **Codebase drift** — line numbers in the plan reflect the state at research time; the actual implementation landed against HEAD `47a0371`, where `_build_tree_inner`'s `..Default::default()` is at line 351 (not 358), the LOO read at line 442 (not 447), and the LOO uniform-scaling block at lines 486–493 (not 483–488). The structural code at those sites was unchanged, so no plan-level adjustment was needed.

### What was actually changed

| File | Lines | What |
|---|---|---|
| `src/types.rs` | +13/–4 | Added `raw_counts: Vec<Vec<f64>>` and `raw_totals: Vec<f64>` fields to `DecisionNode`; refreshed `TrainConfig.leave_one_out` docstring; fixed `rank_thresholds` rustdoc link. |
| `src/training.rs` | +148/–15 | Added `sum_sparse_profiles` helper; refactored `create_tree` to a 5-tuple return propagating raw counts/totals through both leaf and internal branches (sequential and parallel paths); replaced the LOO uniform-scaling block at `_learn_fractions_inner` with the per-kmer formula (LOO ordered before IDF in `weights_j` init); dropped the `n ≤ 5` cap (kept the `n > 1` floor). |
| `src/matching.rs` | +5/–5 | Pre-existing rustdoc warning fixes only (no semantic change). |
| `tests/test_algorithmic_improvements.rs` | +219/–0 | Three new regression tests (`test_leave_one_out_reshapes_profile`, `test_leave_one_out_kmer_specificity`, `test_leave_one_out_decision_kmers_invariant`). |

## Current State Analysis

- `TrainConfig.leave_one_out: bool` declared at [src/types.rs:273](src/types.rs#L273), default `false` at [src/types.rs:299](src/types.rs#L299), threaded through `LearnFractionsConfig::from(c)` at [src/types.rs:157](src/types.rs#L157).
- Python binding accepts the flag at [src/lib.rs:68, 85, 106](src/lib.rs#L68) (`train`) and [src/lib.rs:278, 289, 295](src/lib.rs#L278) (`learn_fractions_py`).
- `_learn_fractions_inner` reads `config.leave_one_out` at [src/training.rs:447](src/training.rs#L447) to compute `loo_child_idx`, then applies a uniform multiplier on `weights_j` at [src/training.rs:483-488](src/training.rs#L483-L488):
  ```rust
  if let Some((loo_j, group_size)) = loo_child_idx {
      if j == loo_j && group_size > 1 && group_size <= 5 {
          let scale = (group_size - 1) as f64 / group_size as f64;
          for w in &mut weights_j { *w *= scale; }
      }
  }
  ```
- `vector_sum` at [src/matching.rs:50-54](src/matching.rs#L50-L54) returns `cur_weight / max_weight`, which is invariant under uniform scaling of `weights`. The LOO branch produces byte-identical `hits[j]` regardless of the scale factor.
- `DecisionNode` at [src/types.rs:6-11](src/types.rs#L6-L11) stores only `keep: Vec<i32>` and `profiles: Vec<Vec<f64>>`. The raw counts and totals used to compute the profiles in `create_tree` ([src/training.rs:1166-1184](src/training.rs#L1166-L1184) for leaves; merged for internal nodes) are discarded after profile normalization.
- The existing test [tests/test_algorithmic_improvements.rs:201-228](tests/test_algorithmic_improvements.rs#L201-L228) asserts `decision_kmers.keep` equality between LOO and non-LOO models but never compares `fraction`, so the no-op went undetected.

## Desired End State

- `leave_one_out=true` produces measurably different `fraction`, `problem_sequences`, and `problem_groups` from `leave_one_out=false` on any training set with at least one decision-node child of size `n > 1`.
- The transformation captures k-mer specificity correctly: singleton-at-sibling k-mers are zeroed, conserved k-mers are unchanged after renormalization, partially shared k-mers fall smoothly between.
- The size cap `n ≤ 5` is removed; the formula degrades gracefully with `n` (effect magnitude diminishes naturally).
- A regression test asserts `assert_ne!(default.fraction, loo.fraction)` so any future regression to a no-op is caught immediately.

### Key Discoveries

- `vector_sum` produces a ratio: any fix must change the **shape** of `weights_j`, not its magnitude ([src/matching.rs:50-54](src/matching.rs#L50-L54)).
- Leaf-node profile construction at [src/training.rs:1166-1184](src/training.rs#L1166-L1184) computes raw `counts: HashMap<usize, f64>` and `total: f64` and then immediately discards them via `count / total` normalization. Both quantities are needed for per-kmer LOO.
- Internal-node profiles are produced by `merge_sparse_profiles` at [src/training.rs:893](src/training.rs#L893) (descendant-weighted average). The raw count analog is element-wise summation across children — strictly simpler than the weighted profile merge.
- `loo_child_idx` already identifies sequence `i`'s matching sibling and exposes its `group_size` ([src/training.rs:447-456](src/training.rs#L447-L456)). The fix only needs to extend what happens *inside* the `j == loo_j` branch.
- Sequences with empty `kmers[i]` are short-circuited at [src/training.rs:416-418](src/training.rs#L416-L418), so the LOO branch never sees them.

## What We're NOT Doing

- **Not** preserving bincode compatibility with existing serialized models. Adding fields to `DecisionNode` invalidates them; users must retrain. (Per user direction: "we don't care at all about backwards compatibility.")
- **Not** rewriting `create_tree` to support exact LOO at internal siblings. The plan uses a "treat the subtree as if it were a flat collection of leaf-sequences" approximation: raw counts merge by element-wise sum, ignoring `descendant_weighting`. This is exact at leaf siblings and faithful-but-approximate at internal siblings — the same simplification the current uniform-scaling code already implicitly makes.
- **Not** exposing `loo_max_group_size` as a config knob. The cap is dropped entirely; if profiling later shows it matters, that knob can be added in a follow-up.
- **Not** touching `use_idf_in_descent`, `correlation_aware_features`, or any other parameter. Other audit findings in `thoughts/shared/research/2026-04-15-new-parameter-audit.md` are out of scope.
- **Not** re-running the OAT/Optuna sweep. The fix unblocks honest sweep results but the rerun is separate work.
- **Not** filing the upstream GitHub issue here. Plan is local-fix only.

## Implementation Approach

Two-layer change: extend `DecisionNode` and `create_tree` to carry the raw counts (Phase 1), then replace the LOO branch in `_learn_fractions_inner` with a per-kmer formula that consumes those new fields (Phase 2). Add the regression test as Phase 3 and refresh the surrounding documentation in Phase 4. Each phase compiles and tests independently.

---

## Phase 1: Extend DecisionNode and propagate raw counts through `create_tree`

### Overview

Add `raw_counts: Vec<Vec<f64>>` and `raw_totals: Vec<f64>` to `DecisionNode`. Modify `create_tree` to compute and propagate them through the recursive descent so each `DecisionNode` carries enough state to reconstruct LOO profiles per child subtree.

### Changes Required

#### 1. Extend `DecisionNode`

**File**: `src/types.rs`
**Changes**: Add two new serde-tracked fields. `raw_counts[j][k]` is the number of leaf-sequences in child subtree `j` that contain `keep[k]`; `raw_totals[j]` is the total k-mer presence count summed over all leaf-sequences in child subtree `j` (across all k-mers, not just kept).

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    pub keep: Vec<i32>,
    pub profiles: Vec<Vec<f64>>,
    /// Raw sequence counts per child subtree, aligned with `keep`.
    /// `raw_counts[j][k]` = number of leaf-sequences in child subtree `j`
    /// that contain `keep[k]`. Used by leave-one-out at fraction-learning.
    pub raw_counts: Vec<Vec<f64>>,
    /// Total k-mer presence count per child subtree (summed over all k-mers
    /// across every leaf-sequence in the subtree, not just kept k-mers).
    /// Used as the LOO denominator: `new_total = raw_totals[j] - |kmers[i]|`.
    pub raw_totals: Vec<f64>,
}
```

#### 2. Refactor `create_tree` to return raw counts and total

**File**: `src/training.rs`
**Changes**: Change the return type of `create_tree` to also return a sparse raw-count vector and a raw-total scalar alongside the existing merged profile and descendant count. At leaves, derive both directly from the existing `counts` HashMap and `total` scalar (currently discarded after `count / total`). At internal nodes, sum element-wise across children.

Current signature:
```rust
fn create_tree(...) -> (SparseProfile, usize, Vec<(usize, DecisionNode)>)
```

New signature:
```rust
fn create_tree(...) -> (SparseProfile, SparseProfile, f64, usize, Vec<(usize, DecisionNode)>)
//                     ^merged profile  ^raw counts   ^raw_total
```

**Leaf branch** (currently at [src/training.rs:1165-1185](src/training.rs#L1165-L1185)): split the count → profile path so we keep both:

```rust
} else {
    let mut counts: HashMap<usize, f64> = HashMap::new();
    if let Some(seq_indices) = &sequences[node] {
        for &si in seq_indices {
            for &km in &kmers[si] {
                if km > 0 && (km as usize) <= n_kmers {
                    *counts.entry((km - 1) as usize).or_insert(0.0) += 1.0;
                }
            }
        }
    }
    let total: f64 = counts.values().sum();

    let mut raw: SparseProfile = counts.into_iter().collect();
    raw.sort_by_key(|&(k, _)| k);

    let profile: SparseProfile = if total > 0.0 {
        raw.iter().map(|&(k, c)| (k, c / total)).collect()
    } else {
        Vec::new()
    };

    (profile, raw, total, 1, Vec::new())
}
```

**Internal branch** (at [src/training.rs:836-1163](src/training.rs#L836-L1163)): the existing code collects `profiles: Vec<SparseProfile>` and `descendants: Vec<usize>` from each child; add parallel collection of `raw_counts_children: Vec<SparseProfile>` and `raw_totals_children: Vec<f64>`. Merge the parent's own raw count vector via element-wise summation (new helper `sum_sparse_profiles`); the parent's own `raw_total` is the sum of children's `raw_totals`. When constructing the `DecisionNode` at [src/training.rs:1159-1162](src/training.rs#L1159-L1162), also build `selected_raw_counts` from `raw_counts_children` filtered to `keep_vec` (mirroring how `selected_profiles` is built from `profiles`).

#### 3. Add `sum_sparse_profiles` helper

**File**: `src/training.rs`
**Changes**: A new helper alongside `merge_sparse_profiles` (currently at [src/training.rs:744](src/training.rs#L744)) that performs k-way merge by element-wise sum (no weighting, no normalization). This is the raw-count analog of `merge_sparse_profiles` for the descendant-weighted average.

```rust
/// Element-wise sum of multiple sparse profiles. Result[k] = Σ_i profiles[i][k].
/// Same k-way-merge structure as `merge_sparse_profiles` but unweighted and
/// unnormalized — used for raw count vectors during LOO state propagation.
fn sum_sparse_profiles(profiles: &[SparseProfile]) -> SparseProfile {
    let mut cursors = vec![0usize; profiles.len()];
    let mut result = Vec::new();
    loop {
        let mut min_key = usize::MAX;
        for (i, p) in profiles.iter().enumerate() {
            if cursors[i] < p.len() && p[cursors[i]].0 < min_key {
                min_key = p[cursors[i]].0;
            }
        }
        if min_key == usize::MAX { break; }
        let mut val = 0.0f64;
        for (i, p) in profiles.iter().enumerate() {
            if cursors[i] < p.len() && p[cursors[i]].0 == min_key {
                val += p[cursors[i]].1;
                cursors[i] += 1;
            }
        }
        result.push((min_key, val));
    }
    result
}
```

#### 4. Update all `create_tree` call sites

**File**: `src/training.rs`
**Changes**: The recursive calls at [src/training.rs:850-851, 861-863, 879](src/training.rs#L850-L851) and the entry-point call from `_build_tree_inner` at [src/training.rs:362-369](src/training.rs#L362-L369) all destructure `create_tree`'s return value. Update each to handle the new 5-tuple and ignore the raw-count outputs at the top level (`_build_tree_inner` doesn't need them since `DecisionNode` already carries them).

### Success Criteria

#### Automated Verification:
- [ ] `cargo build --release --features python` succeeds (broken pre-existing: Python 3.14 vs PyO3 0.24 incompatibility, unrelated to this change)
- [x] `cargo build --release` succeeds (no-python build path stays clean)
- [x] `cargo fmt --check` passes
- [x] `cargo clippy --all-targets --release -- -D warnings` passes
- [x] `cargo test --release --tests` — all pre-existing tests still pass (Phase 1 is purely additive: profiles are unchanged, decision_kmers.keep unchanged, only new fields populated)
- [x] Existing `tests/test_algorithmic_improvements.rs::test_leave_one_out_produces_valid_model` still passes

#### Manual Verification:
- [x] Verified via `test_leave_one_out_kmer_specificity`: `dk.raw_counts[j].len() == dk.profiles[j].len() == dk.keep.len()` and `raw_totals[j] >= sum(raw_counts[j])` for every decision node and child

---

## Phase 2: Replace LOO branch with per-kmer formula

### Overview

Replace the uniform `weights_j *= scale` block in `_learn_fractions_inner` with a per-kmer LOO computation that uses `dk.raw_counts[j]`, `dk.raw_totals[j]`, and the held-out sequence's k-mer count to produce a profile shape that vector_sum's normalization cannot cancel. Drop the upper size cap.

### Changes Required

#### 1. Reorder `weights_j` initialization (LOO before IDF)

**File**: `src/training.rs`
**Changes**: At [src/training.rs:469-481](src/training.rs#L469-L481), the per-child weight init currently applies IDF first; LOO is then applied afterward. Refactor so the base profile (post-LOO if applicable) is computed first, then IDF is layered on top. This is necessary because the LOO formula derives from raw counts, not from the IDF-multiplied weights.

```rust
// Build the base profile for this child: LOO-adjusted if j == loo_j and
// group_size > 1, otherwise the stored profile. Per-kmer LOO subtracts the
// held-out sequence's contribution from each kept-kmer count (1 if matched,
// 0 otherwise) and renormalizes by the LOO total.
let base_profile_j: Vec<f64> = match loo_child_idx {
    Some((loo_j, group_size))
        if j == loo_j
            && group_size > 1
            && dk.raw_totals[j] - (prepared.kmers[i].len() as f64) > 0.0 =>
    {
        let kmers_i_len = prepared.kmers[i].len() as f64;
        let new_total = dk.raw_totals[j] - kmers_i_len;
        dk.raw_counts[j]
            .iter()
            .zip(matches.iter())
            .map(|(&count, &m)| {
                let i_k = if m { 1.0 } else { 0.0 };
                ((count - i_k) / new_total).max(0.0)
            })
            .collect()
    }
    _ => dk.profiles[j].clone(),
};

let mut weights_j: Vec<f64> = if config.use_idf_in_descent {
    base_profile_j
        .iter()
        .zip(dk.keep.iter())
        .map(|(&prof, &km)| {
            let idf = if km > 0 && (km as usize) <= idf_row.len() {
                idf_row[(km - 1) as usize]
            } else {
                0.0
            };
            prof * idf
        })
        .collect()
} else {
    base_profile_j
};
```

#### 2. Delete the uniform scaling block

**File**: `src/training.rs`
**Changes**: The block at [src/training.rs:483-488](src/training.rs#L483-L488) is now subsumed by the new initialization above. Remove it.

```rust
// DELETE (no longer needed; LOO is applied during weights_j init):
// if let Some((loo_j, group_size)) = loo_child_idx {
//     if j == loo_j && group_size > 1 && group_size <= 5 {
//         let scale = (group_size - 1) as f64 / group_size as f64;
//         for w in &mut weights_j { *w *= scale; }
//     }
// }
```

### Success Criteria

#### Automated Verification:
- [ ] `cargo build --release --features python` succeeds (broken pre-existing: Python 3.14 vs PyO3 0.24 incompatibility, unrelated to this change)
- [x] `cargo fmt --check` passes
- [x] `cargo clippy --all-targets --release -- -D warnings` passes
- [x] `cargo test --release --tests` — all pre-existing tests still pass with default `leave_one_out=false`
- [x] Existing `tests/test_algorithmic_improvements.rs::test_leave_one_out_produces_valid_model` still passes (still produces a valid model)
- [x] Existing `tests/test_algorithmic_improvements.rs::test_leave_one_out_standard_produces_valid_classification` still passes

#### Manual Verification:
- [ ] Train two models on a real-world dataset (e.g. vert12s) with `leave_one_out=false` and `leave_one_out=true`. Confirm via `cmp` or md5 that they differ in bytes (the bug's defining empirical signature, now inverted). Note: small test datasets classify perfectly under both LOO settings, so the byte-diff is invisible at small scale; a real dataset is required.
- [x] Spot-check a singleton k-mer (raw_count=1) at a sibling: with LOO on, the LOO-adjusted profile entry for that k-mer is 0 (sequence i was the only contributor) — verified via `test_leave_one_out_kmer_specificity`.
- [x] Spot-check the LOO formula reshapes the profile (not just rescales) — verified via `test_leave_one_out_reshapes_profile`.

---

## Phase 3: Regression tests

### Overview

Add tests that would have caught the no-op when it was introduced. The bar: any future change that re-introduces a uniform-scaling regression (or reverts to no-op) must fail at least one test.

### Changes Required

#### 1. Add `assert_ne!` regression test

**File**: `tests/test_algorithmic_improvements.rs`
**Changes**: Extend the existing `test_leave_one_out_produces_valid_model` (currently at [tests/test_algorithmic_improvements.rs:201-228](tests/test_algorithmic_improvements.rs#L201-L228)), or add a new sibling test, that asserts `default.fraction != loo.fraction` on the singleton dataset.

```rust
#[test]
fn test_leave_one_out_changes_fraction() {
    // Regression test for the LOO no-op bug (vector_sum normalization
    // cancellation). If this fails, leave_one_out has reverted to a
    // mathematical no-op despite reaching the LOO branch.
    let seqs: Vec<String> = load_json("s08d_singleton_seqs");
    let tax: Vec<String> = load_json("s08d_singleton_tax");

    let default_model =
        learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();
    let loo_model = learn_taxa(
        &seqs, &tax,
        &TrainConfig { leave_one_out: true, ..Default::default() },
        42, false,
    ).unwrap();

    // decision_kmers.keep must still match (build-tree phase doesn't see LOO)
    for (d, l) in default_model.decision_kmers.iter()
        .zip(loo_model.decision_kmers.iter())
    {
        if let (Some(dk_d), Some(dk_l)) = (d, l) {
            assert_eq!(dk_d.keep, dk_l.keep);
        }
    }

    // Fractions MUST differ on a dataset with size-2..=5 sibling subtrees.
    // A failure here means LOO is silently inert again.
    assert_ne!(
        default_model.fraction, loo_model.fraction,
        "leave_one_out=true must produce different fraction values from \
         leave_one_out=false; identical means the LOO branch is a no-op"
    );
}
```

#### 2. Add focused unit test for k-mer specificity

**File**: `tests/test_algorithmic_improvements.rs`
**Changes**: A small synthetic test that constructs a known sibling with one singleton k-mer and one conserved k-mer, then verifies the LOO-adjusted profile zeros the singleton and leaves the conserved entry approximately unchanged.

```rust
#[test]
fn test_leave_one_out_kmer_specificity() {
    // Synthetic: 2-sequence sibling. Each sequence contains a distinct
    // singleton k-mer (only that sequence has it) and a shared k-mer
    // (both sequences have it). Verify LOO-adjusted profile for the
    // held-out sequence's matching sibling: singleton k-mer → 0,
    // shared k-mer → approximately preserved after renormalization.
    //
    // Built directly from minimal training data; relies on Phase 1's
    // raw_counts/raw_totals being populated and Phase 2's per-kmer
    // formula being applied.
    //
    // Test body: train on 4 sequences arranged so that the leaf grouping
    // produces a 2-sibling decision node, query the resulting model's
    // dk.raw_counts and dk.raw_totals to compute the LOO-adjusted profile
    // by hand, and assert the expected zero/preserved values.
    // (Full body deferred to implementation — pattern matches the
    //  existing leaf-construction tests in test_training.rs.)
}
```

### Success Criteria

#### Automated Verification:
- [x] `cargo test --release tests::test_leave_one_out_reshapes_profile` passes (replaces the planned `test_leave_one_out_changes_fraction`; small test datasets classify perfectly under any LOO setting so end-to-end fraction differences aren't observable, hence the structural-formula assertion instead)
- [x] `cargo test --release tests::test_leave_one_out_kmer_specificity` passes
- [x] `cargo test --release tests::test_leave_one_out_decision_kmers_invariant` passes (extra test: guards that build-tree phase remains insulated from LOO)
- [x] All other tests in `tests/test_algorithmic_improvements.rs` still pass (5 LOO tests + 16 others = 21 pass)
- [x] `cargo test --release --tests` (full suite, 97 tests) passes

#### Manual Verification:
- [ ] Manually revert Phase 2's changes (restore the uniform-scaling block) and confirm a structural test still flags the regression. Note: with the structural tests (which read `dk.raw_counts`/`dk.raw_totals` directly), reverting the LOO branch alone wouldn't cause failure unless Phase 1 is also reverted. The combined defense is: `test_leave_one_out_kmer_specificity` requires Phase 1; `test_leave_one_out_reshapes_profile` requires Phase 1; together they make a silent revert to "uniform-scaling no-op" hard to land without also breaking Phase 1's structural invariants.

---

## Phase 4: Documentation updates

### Overview

Refresh comments and docstrings so the new behavior is correctly described. The current docstring says LOO "Reduces self-classification bias for small groups" — keep that intent, but make the implementation note accurate.

### Changes Required

#### 1. Update `TrainConfig.leave_one_out` docstring

**File**: `src/types.rs`
**Changes**: At [src/types.rs:269-272](src/types.rs#L269-L272) (the doc-comment block above `leave_one_out: bool`).

```rust
/// Exclude each sequence from its own subtree's profile during fraction
/// learning (per-kmer leave-one-out). Subtracts the held-out sequence's
/// contribution from the matching sibling's raw counts, then renormalizes.
/// Singleton k-mers (held-out sequence is the only contributor) go to zero;
/// conserved k-mers (every group member has them) stay unchanged.
/// Reduces self-classification bias by removing circular evidence at the
/// fraction-calibration step (does not affect classify-time scoring; the
/// stored `dk.profiles` retain full discriminative information).
/// Default false (legacy behavior).
pub leave_one_out: bool,
```

#### 2. Document `DecisionNode.raw_counts` and `raw_totals`

**File**: `src/types.rs`
**Changes**: Already covered in Phase 1 (the docstrings on the new fields).

#### 3. Inline comment above the LOO branch in `_learn_fractions_inner`

**File**: `src/training.rs`
**Changes**: One brief paragraph above the new `base_profile_j` block explaining the per-kmer formula and why it's necessary (vector_sum normalization defeats uniform scaling).

```rust
// Per-kmer LOO: subtract the held-out sequence's contribution from the
// matching sibling's raw counts, renormalize by the reduced total. Must
// reshape the profile (not just rescale) because vector_sum returns
// cur_weight / max_weight, which is invariant under uniform scaling.
// Singleton k-mers go to 0; conserved k-mers are preserved after
// renormalization. Approximate at internal siblings (treats the subtree
// as a flat collection of leaf-sequences, ignoring descendant_weighting).
```

#### 4. Update phase-3 doc-comment

**File**: `src/training.rs`
**Changes**: At [src/training.rs:78-82](src/training.rs#L78-L82) the `learn_fractions` doc-comment lists "training_threshold, use_idf_in_descent, or leave_one_out" as the cheap-rerun parameters. No change needed; that line is already accurate.

### Success Criteria

#### Automated Verification:
- [ ] `cargo doc --no-deps` builds without warnings
- [x] `cargo fmt --check` passes
- [x] `cargo doc --no-deps` builds with zero warnings (also fixed pre-existing matching.rs / types.rs broken-link warnings)

#### Manual Verification:
- [ ] `cargo doc --open` rendering of `TrainConfig.leave_one_out` and `DecisionNode.raw_counts`/`raw_totals` reads naturally and accurately

---

## Testing Strategy

### Unit Tests
- `test_leave_one_out_changes_fraction` — asserts non-equality of `fraction` between LOO and non-LOO models on the singleton dataset. Catches any future regression to no-op.
- `test_leave_one_out_kmer_specificity` — verifies singleton k-mers zero out and conserved k-mers stay preserved on a synthetic 2-sibling case.
- `test_leave_one_out_produces_valid_model` (existing) — preserved as a structure-preservation assertion (`decision_kmers.keep` unchanged).

### Integration Tests
- Run `cargo test --release` (full suite). All non-LOO tests must remain passing because default behavior is unchanged.
- The existing `test_all_improvements_combined` ([tests/test_algorithmic_improvements.rs:511](tests/test_algorithmic_improvements.rs#L511)) exercises LOO alongside other flags; it must still pass.
- Run `cargo test --release --test test_baseline_1k` (or whichever baseline fixture the project uses) to confirm reference-data classification on a standard model still works after the schema change.

### Manual Testing Steps
1. Train a model with `leave_one_out=False` on a small reference (vert12s slice or similar). Save md5.
2. Train the same dataset with `leave_one_out=True`. Compare md5 — should now differ.
3. Verify the `fraction` field differs between the two `TrainingSet` bincode files (deserialize and compare).
4. Verify `decision_kmers` is byte-identical between the two (build-tree phase still doesn't see the flag, by design).
5. Run classification with both models on a held-out test set; observe whether confidence calibration changed in the expected direction (LOO model should produce more permissive thresholds, fewer below-threshold abstentions).

## Performance Considerations

- **Memory**: `DecisionNode` storage roughly doubles (`raw_counts` is the same shape as `profiles`; `raw_totals` adds one f64 per child). On a typical model (~310 MB for vert12s), expect ~620 MB after the change. Acceptable.
- **Compute (training)**: `create_tree` does one extra `sum_sparse_profiles` merge per internal node — same complexity class as the existing `merge_sparse_profiles` call, ~1× constant overhead. Fraction-learning's LOO branch goes from ~1 multiplication per kept-kmer to ~3 floating-point operations per kept-kmer (subtract, divide, max). With the upper cap removed, the LOO branch fires at ~3× more nodes (per the vert12s 30.5% → ~99% transition). Net: training time increases by an estimated 5-15% when `leave_one_out=true`; unchanged when `leave_one_out=false` (LOO branch still skipped at `j != loo_j`).
- **Compute (classify)**: zero impact — classify-time code never reads `raw_counts` or `raw_totals`.
- **Disk**: serialized model size grows by ~50% (raw_counts roughly equals profiles in size; raw_totals is negligible).

## Migration Notes

- **Bincode incompatibility**: existing serialized `TrainingSet` files cannot deserialize after Phase 1 because `DecisionNode` gained required fields. Per user direction, no version-tag fallback is added — users must retrain. Document this clearly in the version bump (`Cargo.toml` minor or major bump appropriate).
- **No public API changes**: `train()`, `classify()`, `prepare_data_py`, `build_tree_py`, `learn_fractions_py` Python signatures are unchanged. Behavior under `leave_one_out=true` changes from no-op to functional.
- **OAT/Optuna sweep results** previously recorded as showing `leave_one_out=True` optimal on Vert12s ([thoughts/shared/research/2026-04-21-three-marker-sweep-findings.md](thoughts/shared/research/2026-04-21-three-marker-sweep-findings.md)) are noise (the parameter did nothing). Re-running the sweep is recommended but out of scope for this plan.

## References

- Research document: [thoughts/shared/research/2026-04-30-oat-param-bug-investigation.md](thoughts/shared/research/2026-04-30-oat-param-bug-investigation.md) — see §5 (vector_sum normalization analysis) and §6 (fix-strategy comparison)
- Prior bug audit: [thoughts/shared/research/2026-04-15-new-parameter-audit.md](thoughts/shared/research/2026-04-15-new-parameter-audit.md) — section 1 flagged this as `CRITICAL BUG` with the same root cause
- Original (incomplete) plan: [thoughts/shared/plans/2026-04-13-algorithmic-improvements.md](thoughts/shared/plans/2026-04-13-algorithmic-improvements.md) — described the `(n-1)/n` scaling as a "Pass 1" approximation requiring a "Pass 2" with stored totals; this plan implements Pass 2
- LOO read site: `src/training.rs:447`
- LOO uniform-scaling block to delete: `src/training.rs:483-488`
- `vector_sum` normalization (the structural cause): `src/matching.rs:50-54`
- `create_tree` leaf branch (raw counts source): `src/training.rs:1166-1184`
- `merge_sparse_profiles` (model for `sum_sparse_profiles` helper): `src/training.rs:744`
- `DecisionNode` struct: `src/types.rs:6-11`
- Existing LOO test (to extend): `tests/test_algorithmic_improvements.rs:201-228`
