# oxidtaxa Holdout-Robustness Improvements — Implementation Plan

## Overview

Ship nine improvements (I1, I2, I3, I5, I6, I7, I8, I9, I10) to make oxidtaxa more robust to the
species-holdout, genus-holdout, and mixed-depth scenarios exercised by the
`assignment-tool-benchmarking` project. The improvements target three distinct failure surfaces:
(a) output-layer information loss (I1/I2/I3/I8), (b) overconfident commitment on near-tied
evidence (I5/I6/I7), and (c) training-time feature-selection and IDF semantics (I9/I10).

Each behavior-changing improvement ships behind a config flag defaulting to current behavior.
No golden tests break by default; new tests exercise the new flags.

## Current State Analysis

**Classification** (`src/classify.rs`):
- Greedy descent with `min_descend = 0.98` absolute threshold (no margin detection).
- Exact-equality-only `tot_hits` tie check at `:647-656` → LCA cap and `alternatives` rarely fire.
- `above.is_empty()` fallback at `:741-748` emits `[Root, unclassified_Root]`; TSV writer
  (`src/fasta.rs:78-105`) collapses to `\t\t0\t`, destroying the real `c0 = base_confidence`.
- `rank_thresholds` filter (`:720-733`) produces non-contiguous `above` sets when mid-rank
  confidences dip, yielding taxonomically incoherent lineages.
- `similarity` scalar computed at `:657-660` but never exposed.

**Training** (`src/training.rs`):
- Pearson correlation used for feature redundancy (`:714-722, :857-1025`); degenerates at n=2
  (Pearson r = ±1 for any two 2D points) and misbehaves in higher dimensions by conflating
  proportionality with information content.
- IDF computed globally (`:309-351`); no per-rank variant.

**Python/Config wiring** (`src/lib.rs`, `src/types.rs`):
- `ClassifyConfig` / `TrainConfig` fields exposed via `#[pyo3(signature = (...))]` defaults.
- Enums like `DescendantWeighting` and `StrandMode` passed as `&str`, parsed by helpers at
  `src/lib.rs:298-316`.
- Bincode serialization is non-self-describing (no format version). New fields on `TrainingSet`
  break old model loads unless annotated `#[serde(default)]`.

**Tests** (`tests/`):
- 13 Rust test files; ~100 JSON fixtures under `tests/golden_json/`.
- `GoldenClassResult` (tests/test_classify.rs:10-14) only reads `taxon` + `confidence` — extra
  fields on `ClassificationResult` are ignored.
- TSV end-to-end test is `tests/test_integration.rs` / `s10a_e2e_tsv.json`; regenerating this
  golden is required when TSV content changes.
- `CONF_TOLERANCE = 5.0` on confidence comparisons.

## Desired End State

After all four phases ship:

1. **TSV output distinguishes Path A/B (no signal) from Path C (below-threshold signal)** via
   the confidence column alone. Optional `reject_reason` column refines the distinction.
2. **Near-tied classifications produce LCA caps and `alternatives`** when the user opts in via
   `tie_margin > 0`. Current exact-equality behavior preserved at `tie_margin = 0.0`.
3. **Descent ambiguity propagates into reported confidence** when the user opts in via
   `confidence_uses_descent_margin = true`.
4. **Single-winner descent can widen to include near-winner siblings** at leaf phase when
   `sibling_aware_leaf = true`.
5. **Training's redundancy metric can be Bhattacharyya coefficient** (Hellinger-equivalent) via
   a new enum, resolving the n=2 Pearson degeneracy and providing mathematically-justified
   feature selection across all split sizes.
6. **Per-rank IDF weights optionally computed and used** at classification time, with backward
   compatibility for existing bincode models.
7. **`similarity` scalar exposed** on `ClassificationResult` and TSV for downstream calibration.
8. **`rank_thresholds` emits contiguous lineages only** — fixes a pre-existing bug.

### Verification:
- All existing golden JSON tests pass unchanged (`CONF_TOLERANCE = 5.0` applies).
- `s10a_e2e_tsv` golden regenerated to reflect new TSV semantics.
- New tests exercise each flag combination.
- Criterion benches for `bench_id_taxa` and `bench_learn_taxa_correlation_aware` show no
  performance regression with default flags; dedicated benches measure new paths.

### Key Discoveries:
- `ClassificationResult` (`src/types.rs:154-176`) already has `alternatives` with
  `#[serde(default, skip_serializing_if = ...)]` — new fields can follow this pattern.
- `GoldenClassResult` ignores extra fields, so adding `similarity` / `reject_reason` to
  `ClassificationResult` is non-breaking for existing goldens.
- `TrainingSet.inverted_index: Option<Vec<Vec<u32>>>` (`src/types.rs:45`) was previously added
  without `#[serde(default)]` — meaning models saved before its addition fail to load. I10
  will use `#[serde(default)]` explicitly to avoid repeating this mistake.
- Golden confidence tolerance is 5.0 — descent-margin weighting in I6 must be gated off by
  default, otherwise golden confidences shift by more than tolerance.

## What We're NOT Doing

- **I4 (NA taxonomy handling)**: out of scope — the benchmark's `cleanup_taxonomy_file` at
  `benchmark_shared.py:609-688` already strips trailing NAs and replaces intermediate NAs with
  `{parent}_unclassified` pseudo-labels before oxidtaxa sees them. Left as defensive hygiene
  for future work targeting direct (non-benchmark) LCA-reference users.
- **I11 (remove `use_idf_in_training`)**: deferred. Current default is `false`; nobody is using
  it in production. No holdout-tier impact.
- **I12 (seed_pattern overflow guard)**: deferred. Separate validation-only change.
- **I13 (R-divergent decrement documentation)**: deferred. Separate documentation task.
- **Model format version bump**: deliberately avoided. All TrainingSet changes use
  `#[serde(default)]` for backward compat.
- **Calibration of confidence to TOS-plausibility** (the 2026-04-17 report §6 anomaly): this
  plan does not change the definition of `confidence`; it only preserves more of it in output.
  Calibration is a separate research item.
- **CLI exposure of every new flag**: `classify.py` / `train.py` argparse surfaces will grow
  only where the user explicitly requests it; Python-API-only is acceptable for the initial
  ship.

## Implementation Approach

Phase 1 lands independently — pure output-layer changes with no algorithm implications. Phase 2
depends on Phase 1 because the new tests for I5/I6/I7 rely on the TSV schema settled in Phase 1.
Phase 3 is independent of Phases 1-2 (training-side only). Phase 4 is independent of all prior
phases but requires a serde migration; ship last to let the other improvements stabilize first.

Every behavior-changing flag defaults to preserving legacy behavior so:
- Existing golden tests pass untouched.
- Users who pin oxidtaxa versions see no behavior drift.
- A/B comparison on the benchmark corpus cleanly measures the effect of each flag.

---

## Phase 1: Output-Layer Information Preservation

### Overview
Fix the TSV writer to preserve real `c0` confidence when the path is empty (I1), optionally
emit `reject_reason` to distinguish Path A from Path B (I2), fix the `rank_thresholds`
non-contiguous-lineage bug (I3), and expose the `similarity` scalar on `ClassificationResult`
and TSV (I8).

### Changes Required:

#### 1. Preserve `c0` in empty-path TSV rows (I1)

**File**: `src/fasta.rs:78-105`

**Changes**: When `filtered_taxa` is empty, write `result.confidence.first()` to the
confidence column instead of hardcoded `0`.

```rust
// Replace lines 100-104:
if !filtered_taxa.is_empty() {
    let path_str = filtered_taxa.join(";");
    let min_conf = filtered_conf
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    output.push_str(&format!(
        "{}\t{}\t{}\t{}\t{}\n",
        read_id, path_str, min_conf, alternatives_field, reject_reason_field
    ));
} else {
    // I1: write the real c0 (= base_confidence from leaf-phase scoring)
    // so Path A/B (c0 == 0) is distinguishable from Path C (c0 > 0).
    let c0 = result.confidence.first().copied().unwrap_or(0.0);
    output.push_str(&format!(
        "{}\t\t{}\t{}\t{}\n",
        read_id, c0, alternatives_field, reject_reason_field
    ));
}
```

Also update the `else` branch at line 103-105 (when `taxa.len() <= 1`) to the same pattern.

#### 2. Add `reject_reason` to `ClassificationResult` and populate it (I2)

**File**: `src/types.rs:153-176`

**Changes**: Add `reject_reason: Option<String>` field with serde default.

```rust
#[cfg_attr(feature = "python", pyo3::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub taxon: Vec<String>,
    pub confidence: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<String>,
    /// Optional explanation of why a row has empty `taxon` / zero confidence.
    /// Values: `None` (classified), `Some("too_few_kmers")`, `Some("no_training_match")`,
    /// `Some("below_threshold")`. Only populated for abstention paths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reject_reason: Option<String>,
    /// Expose the leaf-phase similarity scalar (see I8).
    #[serde(default)]
    pub similarity: f64,
}

impl ClassificationResult {
    pub fn unclassified(reason: &str) -> Self {
        Self {
            taxon: vec!["Root".to_string(), "unclassified_Root".to_string()],
            confidence: vec![0.0, 0.0],
            alternatives: Vec::new(),
            reject_reason: Some(reason.to_string()),
            similarity: 0.0,
        }
    }
}
```

**File**: `src/classify.rs:186-188, :483`

**Changes**: Replace two call sites of `ClassificationResult::unclassified()` with the new
variant that takes a reason string.

```rust
// src/classify.rs:186-188
if my_kmers.len() <= s {
    return None;  // handled in caller; caller attaches reason "too_few_kmers"
}
```

At the `None`-returning sites, the caller (`classify_one_pass` wrappers at `src/classify.rs:755-841`)
converts `None` to `ClassificationResult::unclassified("too_few_kmers")` or
`::unclassified("no_training_match")` based on which branch returned `None`. Simplest
implementation: have the two `None`-return sites return `Option<(ClassificationResult, f64)>`
with `Some((result, 0.0))` wrapping a pre-built reject result, distinguishing at the return point:

```rust
// At line 186-188:
if my_kmers.len() <= s {
    return Some((
        ClassificationResult::unclassified("too_few_kmers"),
        0.0,
    ));
}
// At line 483:
if keep.is_empty() {
    return Some((
        ClassificationResult::unclassified("no_training_match"),
        0.0,
    ));
}
```

And in the below-threshold fallback at `src/classify.rs:741-748`:
```rust
} else {
    let w = if above.is_empty() { vec![0] } else { above };
    let last_w = *w.last().unwrap();
    let mut taxon: Vec<String> = w.iter().map(|&i| taxa[predicteds[i]].clone()).collect();
    taxon.push(format!("unclassified_{}", taxa[predicteds[last_w]]));
    let mut conf: Vec<f64> = w.iter().map(|&i| confidences[i]).collect();
    conf.push(confidences[last_w]);
    ClassificationResult {
        taxon,
        confidence: conf,
        alternatives,
        reject_reason: Some("below_threshold".to_string()),
        similarity,  // from I8 below
    }
};
```

**File**: `src/fasta.rs:60-110`

**Changes**: Add `reject_reason` as the 5th TSV column. Header becomes:
```
read_id\ttaxonomic_path\tconfidence\talternatives\treject_reason
```

Emit the value (empty string if `None`). All three output branches (non-empty path, Path C
empty, Path A/B empty) write this column.

#### 3. Fix `rank_thresholds` contiguity (I3)

**File**: `src/classify.rs:720-733`

**Changes**: Replace the `filter`-based `above` construction with an explicit loop that
terminates at the first failing rank, guaranteeing `above` is always a contiguous prefix
`[0, 1, 2, ..., k]`.

```rust
let mut above: Vec<usize> = Vec::new();
for (i, &c) in confidences.iter().enumerate() {
    if let Some(cap) = lca_cap {
        if i > cap { break; }
    }
    let thresh = match &config.rank_thresholds {
        Some(rt) if i < rt.len() => rt[i],
        Some(rt) if !rt.is_empty() => *rt.last().unwrap(),
        _ => config.threshold,
    };
    if c >= thresh {
        above.push(i);
    } else {
        break;  // contiguous prefix only
    }
}
```

#### 4. Expose `similarity` on `ClassificationResult` and TSV (I8)

**File**: `src/classify.rs:735-749`

**Changes**: Plumb `similarity` (already computed at `:657-660`) into both branches of the
result construction:

```rust
let result = if above.len() == predicteds.len() {
    ClassificationResult {
        taxon: predicteds.iter().map(|&p| taxa[p].clone()).collect(),
        confidence: confidences,
        alternatives: alternatives.clone(),
        reject_reason: None,
        similarity,
    }
} else {
    // ... truncation branch ...
};
```

Also set `similarity` on both `None`-returning paths (`unclassified("too_few_kmers")` and
`unclassified("no_training_match")`) to `0.0`.

**File**: `src/fasta.rs:65`

**Changes**: Add `similarity` as the 6th TSV column. Header becomes:
```
read_id\ttaxonomic_path\tconfidence\talternatives\treject_reason\tsimilarity
```

Emit `result.similarity` on every row.

#### 5. Regenerate `s10a_e2e_tsv` golden

**File**: `tests/golden_json/s10a_e2e_tsv.json`

**Changes**: Regenerate by running `cargo test --test test_integration -- --ignored write_s10a_golden`
(add a new `#[ignore]` test that writes the current TSV output to disk) OR manually update the
JSON to reflect the new 6-column TSV schema. The existing confidence assertions at
`tests/test_integration.rs:108` remain valid at tolerance 5.0.

### Success Criteria:

#### Automated Verification:
- [x] Unit tests pass: `cargo test` (83 passing)
- [x] Specifically the classify suite: `cargo test --test test_classify` (13 golden tests)
- [x] Integration test passes with regenerated TSV golden:
      `cargo test --test test_integration`
- [x] Ties test still passes: `cargo test --test test_ties`
- [x] Python extension compiles: `cargo check --features python`

#### Manual Verification:
- [ ] Run classifier on the benchmark's species-holdout fixture; verify Path-C empty-path rows
      now carry non-zero `confidence` column values.
- [ ] Run with `rank_thresholds = [60, 70, 80, 90, 60, 50, 40]` on a crafted query where
      the middle rank fails — verify output path is contiguous (e.g., `K;P;C` or `K;P`, not
      `K;P;O`).
- [ ] Inspect one result's `similarity` field on a Normal-tier query (expect ≈ 1.0) vs a
      species-holdout query (expect < 1.0).

---

## Phase 2: Margin-Aware Classification

### Overview
Add three config flags that let the classifier represent and respond to near-tied evidence:
`tie_margin` (I5) relaxes the leaf-phase exact-equality filter; `confidence_uses_descent_margin`
(I6) propagates descent-time vote margins into reported confidence; `sibling_aware_leaf` (I7)
widens `w_indices` at single-winner descent commits. All default to legacy behavior.

### Changes Required:

#### 1. Margin-based LCA cap (I5)

**File**: `src/types.rs:275-312`

**Changes**: Add `tie_margin: f64` field to `ClassifyConfig` with default 0.0.

```rust
pub struct ClassifyConfig {
    // ... existing fields ...
    /// Fraction of the max tot_hits below which sibling groups are treated as
    /// tied winners for LCA-cap and `alternatives` purposes. Default 0.0
    /// (exact equality only, matches legacy). Suggested sweep range [0.0, 0.10].
    pub tie_margin: f64,
}

impl Default for ClassifyConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            tie_margin: 0.0,
        }
    }
}
```

**File**: `src/classify.rs:647-656`

**Changes**: Relax `v == max_tot` to `v >= max_tot * (1.0 - tie_margin)`:

```rust
let max_tot = tot_hits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let winners: Vec<usize> = if config.tie_margin > 0.0 && max_tot > 0.0 {
    let threshold = max_tot * (1.0 - config.tie_margin);
    tot_hits.iter().enumerate()
        .filter(|(_, &v)| v >= threshold)
        .map(|(i, _)| i).collect()
} else {
    tot_hits.iter().enumerate()
        .filter(|(_, &v)| v == max_tot)
        .map(|(i, _)| i).collect()
};
```

**File**: `src/lib.rs:135-183`

**Changes**: Add `tie_margin = 0.0` to the `#[pyo3(signature = ...)]` tuple and pass through
to `ClassifyConfig`:

```rust
#[pyo3(signature = (
    query_path, model_path, output_path,
    threshold = 60.0,
    bootstraps = 100,
    strand = "both",
    min_descend = 0.98,
    full_length = 0.0,
    processors = 1,
    deterministic = false,
    sample_exponent = 0.47,
    seed = 42,
    length_normalize = false,
    rank_thresholds = None,
    beam_width = 1,
    tie_margin = 0.0,
    confidence_uses_descent_margin = false,
    sibling_aware_leaf = false,
))]
```

#### 2. Descent-margin in confidence (I6)

**File**: `src/types.rs:275-312`

**Changes**: Add `confidence_uses_descent_margin: bool` field (default `false`).

**File**: `src/classify.rs:190-248`

**Changes**: Track per-rank descent margins when the flag is on:

```rust
// Inside classify_one_pass, before the descent loop:
let mut descent_margins: Vec<f64> = Vec::new();

// Inside the loop after vote_counts is computed (around :224):
if config.confidence_uses_descent_margin {
    let mut sorted = vote_counts.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let top = *sorted.first().unwrap_or(&0) as f64;
    let runner_up = *sorted.get(1).unwrap_or(&0) as f64;
    let margin = if top > 0.0 {
        ((top - runner_up) / (b as f64)).max(0.1)  // floor at 10% to avoid zeroing
    } else {
        1.0
    };
    descent_margins.push(margin);
}
```

Pass `descent_margins` through to `leaf_phase_score` as a new parameter. Inside
`leaf_phase_score`, when building the `confidences` vector, multiply each rank's confidence by
the cumulative product of descent margins from Root through that rank:

```rust
// In leaf_phase_score, after line 674:
if config.confidence_uses_descent_margin && !descent_margins.is_empty() {
    let mut cumulative = 1.0;
    for i in 0..confidences.len() {
        if i < descent_margins.len() {
            cumulative *= descent_margins[i];
        }
        confidences[i] *= cumulative;
    }
}
```

Note: the inner ancestor-walk loop at `:675-686` adds to `confidences[m]` before the margin
multiplication — intended, so sibling contributions are discounted proportionally.

#### 3. Sibling-aware leaf scoring (I7)

**File**: `src/types.rs:275-312`

**Changes**: Add `sibling_aware_leaf: bool` field (default `false`).

**File**: `src/classify.rs:241-243`

**Changes**: When a single winner is found AND `sibling_aware_leaf == true`, widen
`w_indices` to include any sibling with `vote_counts[j] >= 0.5 * b`:

```rust
let winner = w[0];
if children[subtrees[winner]].is_empty() {
    w_indices = if config.sibling_aware_leaf {
        let min_votes = ((b as f64) * 0.5) as usize;
        vote_counts.iter().enumerate()
            .filter(|(_, &c)| c >= min_votes)
            .map(|(i, _)| i).collect()
    } else {
        vec![winner]
    };
    break;
}
k_node = subtrees[winner];
```

The widened `w_indices` flows through to `leaf_phase_score`, which scores the query against
training sequences from all the included children. Natural consequence: near-sibling evidence
now contributes to the `tot_hits` vector, and the (possibly margin-based, see I5)
`winners.len() > 1` check can then cap the lineage at the LCA.

### Success Criteria:

#### Automated Verification:
- [x] All Phase 1 automated checks still pass
- [x] New tests added under `tests/test_margin_aware.rs`:
  - [x] `test_tie_margin_catches_near_ties` — near-tied `tot_hits` with
        `tie_margin = 0.10` produces `alternatives.len() >= 2` with both Canis species
        and truncates the lineage above `Canis_*`.
  - [x] `test_tie_margin_zero_preserves_legacy` — flipping tie_margin from 0.10 to 0.0
        cannot shrink alternatives below the zero case (monotonic).
  - [x] `test_descent_margin_on_never_raises_confidence` — flipping
        `confidence_uses_descent_margin` ON cannot raise any per-rank confidence.
  - [x] `test_descent_margin_default_off` — default `ClassifyConfig.confidence_uses_descent_margin`
        is `false`.
  - [x] `test_sibling_aware_leaf_cannot_shrink_alternatives` — flipping `sibling_aware_leaf`
        ON cannot shrink the alternatives set (monotonic).
- [x] Golden tests unchanged: `cargo test --test test_classify` (all 13 s09* tests pass at
      default flag values).

#### Manual Verification:
- [ ] Sweep `tie_margin in [0.0, 0.01, 0.02, 0.05, 0.10]` on the benchmark's species-holdout
      tier; confirm species-holdout F1 increases monotonically with tighter abstention.
- [ ] Sweep `confidence_uses_descent_margin in [false, true]` with same `tie_margin`;
      confirm Normal-tier F1 does not regress by more than 0.5 points.
- [ ] `sibling_aware_leaf = true` with `tie_margin = 0.02`: verify alternatives list contains
      near-tied siblings on species-holdout queries.

---

## Phase 3: Bhattacharyya Redundancy Metric

### Overview
Add a new `redundancy_metric` knob on `TrainConfig` so users can select Bhattacharyya
coefficient (Hellinger-based) instead of Pearson correlation for the
`correlation_aware_features` path. Default stays Pearson; `correlation_aware_features = false`
(the default) still means round-robin feature selection — i.e., nothing changes unless the
user opts into both flags.

### Changes Required:

#### 1. New enum `RedundancyMetric`

**File**: `src/types.rs:204-213` (alongside `DescendantWeighting`)

**Changes**:

```rust
/// Redundancy metric used by correlation-aware feature selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RedundancyMetric {
    /// Pearson correlation on raw profiles. Original behavior.
    /// Degenerates for n_children == 2 (r = ±1 always).
    Pearson,
    /// Bhattacharyya coefficient on L1-normalized sqrt-transformed profiles.
    /// Equivalent to (1 - H²/2) where H is Hellinger distance.
    /// Mathematically justified for probability-distribution-like profiles.
    Bhattacharyya,
}
```

**File**: `src/types.rs:215-272`

**Changes**: Add `redundancy_metric: RedundancyMetric` to `TrainConfig` (default `Pearson`).

```rust
pub struct TrainConfig {
    // ... existing fields ...
    pub redundancy_metric: RedundancyMetric,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            redundancy_metric: RedundancyMetric::Pearson,
        }
    }
}
```

#### 2. Propagate through to `BuildTreeConfig`

**File**: `src/types.rs:106-112`

```rust
pub struct BuildTreeConfig {
    pub record_kmers_fraction: f64,
    pub descendant_weighting: DescendantWeighting,
    pub correlation_aware_features: bool,
    pub redundancy_metric: RedundancyMetric,
    pub max_children: usize,
    pub processors: usize,
}
```

Update the `From<&TrainConfig>` impl at `:126-135` accordingly.

#### 3. Implement `bhattacharyya_with_stats`

**File**: `src/training.rs:693-722`

**Changes**: Add a parallel struct `BhattacharyyaStats` and a pairwise function.

```rust
struct BhattacharyyaStats {
    /// sqrt of the L1-normalized profile entries.
    sqrt_profile: Vec<f64>,
}

impl BhattacharyyaStats {
    fn new(profile: &[f64]) -> Self {
        let sum: f64 = profile.iter().sum();
        let sqrt_profile = if sum > 0.0 {
            profile.iter().map(|&v| (v / sum).sqrt()).collect()
        } else {
            vec![0.0; profile.len()]  // all-zero profile — caller should skip
        };
        Self { sqrt_profile }
    }
}

/// Bhattacharyya coefficient: Σ_j sqrt(p̃_a[j]) * sqrt(p̃_b[j]).
/// Range [0, 1]; 1 when profiles are identical (after L1 norm), 0 when disjoint.
fn bhattacharyya_with_stats(
    _profile_a: &[f64],
    _profile_b: &[f64],
    stats_a: &BhattacharyyaStats,
    stats_b: &BhattacharyyaStats,
) -> f64 {
    stats_a.sqrt_profile.iter()
        .zip(stats_b.sqrt_profile.iter())
        .map(|(a, b)| a * b)
        .sum()
}
```

Keep `ProfileStats` and `pearson_with_stats` unchanged.

#### 4. Dispatch on the metric in feature selection

**File**: `src/training.rs:857-1025`

**Changes**: Introduce a trait or enum-dispatched loop. Simplest: compute both stats up front
(cheap compared to pairwise work) and dispatch via an `if` on the metric:

```rust
// Precompute stats for all candidates:
let pearson_stats: Vec<ProfileStats> = ...; // existing
let bhattacharyya_stats: Vec<BhattacharyyaStats> = if matches!(redundancy_metric, RedundancyMetric::Bhattacharyya) {
    candidates.iter().map(|c| BhattacharyyaStats::new(&c.profile)).collect()
} else {
    Vec::new()
};

// Inside the main gain-computation loop, dispatch:
let redundancy = match redundancy_metric {
    RedundancyMetric::Pearson => pearson_with_stats(...),
    RedundancyMetric::Bhattacharyya => bhattacharyya_with_stats(...),
};
let gain = entropy_i * (1.0 - redundancy);
```

The `max_corr` cache at `src/training.rs:995-1023` (commit `a2cd22a`) transfers verbatim — it's
a per-candidate scalar of the maximum redundancy seen so far, agnostic to the underlying metric.

#### 5. Filter all-zero profiles before feature selection

**File**: `src/training.rs:857-880` (candidate pool construction)

**Changes**: Drop any candidate whose profile sums to 0 (present in no children) — these
k-mers are uninformative and would cause division-by-zero in the Bhattacharyya normalization.
Should be a no-op for typical data (the candidate pool is entropy-ranked, and all-zero
profiles have zero entropy) but defensive.

#### 6. Parse enum from Python

**File**: `src/lib.rs:298-316` (near `parse_descendant_weighting`)

```rust
fn parse_redundancy_metric(s: &str) -> PyResult<RedundancyMetric> {
    match s.to_lowercase().as_str() {
        "pearson" => Ok(RedundancyMetric::Pearson),
        "bhattacharyya" | "hellinger" => Ok(RedundancyMetric::Bhattacharyya),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown redundancy_metric: {}. Expected 'pearson' or 'bhattacharyya'.",
            other
        ))),
    }
}
```

**File**: `src/lib.rs:62-110`

**Changes**: Add `redundancy_metric = "pearson"` to the `train` signature and forward:

```rust
#[pyo3(signature = (
    // ... existing ...
    correlation_aware_features = false,
    redundancy_metric = "pearson",
    processors = 1,
))]
pub fn train(
    // ... existing parameters ...
    correlation_aware_features: bool,
    redundancy_metric: &str,
    processors: usize,
) -> PyResult<Vec<ProblemSequence>> {
    let redundancy_metric = parse_redundancy_metric(redundancy_metric)?;
    let config = TrainConfig {
        // ... existing ...
        correlation_aware_features,
        redundancy_metric,
        processors,
        ..Default::default()
    };
    // ...
}
```

### Success Criteria:

#### Automated Verification:
- [x] All Phase 1 and Phase 2 automated checks still pass
- [x] `cargo test --test test_training` passes (6 tests)
- [x] `cargo test --test test_algorithmic_improvements` passes (existing
      correlation-aware-features tests use Pearson default; unchanged, `EXPECTED_HASH`
      still matches `0x86d65b441ed02bd8`)
- [x] New tests in `tests/test_algorithmic_improvements.rs`:
  - [x] `test_bhattacharyya_differs_from_pearson_on_some_nodes` — Bhattacharyya picks
        different k-mers on at least one decision node than Pearson.
  - [x] `test_bhattacharyya_produces_valid_model` — model trains, classifies, shape OK.
  - [x] `test_redundancy_metric_default_pearson` — default metric is `Pearson`.
  - [x] `test_bhattacharyya_round_robin_flag_off_is_noop` — when
        `correlation_aware_features = false`, Pearson vs Bhattacharyya produces
        bit-identical decision_kmers.

#### Manual Verification:
- [ ] Train two models on `benchmarks/data/10k.fasta` — one with
      `redundancy_metric = "pearson"`, one with `"bhattacharyya"`. Measure train time;
      Bhattacharyya should be within ±20% of Pearson.
- [ ] Classify the same query set with both models on the benchmark species-holdout tier;
      confirm Bhattacharyya produces equal-or-better species-holdout F1 without regressing
      Normal-tier F1 by more than 0.5 points.
- [ ] Run `bench_learn_taxa_correlation_aware` — existing benchmark (Pearson path) should
      show no regression.

---

## Phase 4: Per-Rank IDF Weights

### Overview
Add an optional per-rank IDF vector stored alongside the existing global IDF. At classify
time, when the training set has per-rank IDF and the config enables it, the leaf phase uses
the rank-specific IDF row matching the descent depth. Default stays global IDF; existing
bincode models load unchanged.

### Changes Required:

#### 1. Compute per-rank IDF during training

**File**: `src/training.rs:309-351`

**Changes**: After computing the global `idf_weights`, optionally compute a
`idf_weights_by_rank: Vec<Vec<f64>>` where `idf_weights_by_rank[rank_idx][kmer_idx]` uses
`n_classes` defined at that rank (the number of distinct taxa whose taxonomy string has depth
≥ rank_idx + 1, deduplicated by prefix). Gated on a new config field
`TrainConfig.per_rank_idf: bool` (default `false`).

```rust
// After existing idf_weights computation at :348-351:
let idf_weights_by_rank: Option<Vec<Vec<f64>>> = if config.per_rank_idf {
    Some(compute_per_rank_idf(&kmers, &classes, &taxonomy_levels))
} else {
    None
};
```

Implementation of `compute_per_rank_idf`: for each rank level R in the tree (up to the deepest
leaf rank), group sequences by their prefix path at rank R, re-compute `n_classes_at_R =
distinct_prefixes_at_R`, and re-run the same weighted-sum-of-presence calculation as the
global IDF but with the new class count. Cost: O(R × K × n_seqs) where R is typically 7 ranks;
dominated by the same pass as global IDF, so not a bottleneck.

#### 2. Add the field to `TrainingSet` with serde default

**File**: `src/types.rs:22-46`

**Changes**:

```rust
pub struct TrainingSet {
    // ... existing fields ...
    pub idf_weights: Vec<f64>,
    /// Optional per-rank IDF weights. Indexed as [rank_idx][kmer_idx].
    /// When present, classification uses the rank-specific row matching descent depth.
    /// Absent in models trained without `per_rank_idf = true`.
    #[serde(default)]
    pub idf_weights_by_rank: Option<Vec<Vec<f64>>>,
    // ... existing fields ...
}
```

Critical: `#[serde(default)]` ensures existing bincode models (which don't have this field)
load successfully with `idf_weights_by_rank = None`.

#### 3. Use per-rank IDF at classify time

**File**: `src/classify.rs:466`, `:546-548`

**Changes**: Replace `counts = &ts.idf_weights` with rank-dependent selection:

```rust
// At :466, extract a reference to the rank-appropriate IDF row based on descent depth:
let descent_depth = predicteds.len();  // how deep we descended before leaf phase

let counts: &[f64] = if let Some(by_rank) = &ts.idf_weights_by_rank {
    // Pick the closest available rank index (cap at array length)
    let rank_idx = descent_depth.min(by_rank.len() - 1);
    &by_rank[rank_idx]
} else {
    &ts.idf_weights
};
```

The rest of `leaf_phase_score` (`:546-548` and onward) consumes `counts` unchanged.

#### 4. Add `per_rank_idf` flag to `TrainConfig` and Python surface

**File**: `src/types.rs:215-272`

**Changes**: Add `per_rank_idf: bool` (default `false`).

**File**: `src/lib.rs:62-110`

**Changes**: Add `per_rank_idf = false` to the `train` signature tuple.

#### 5. Migration note for existing models

Old models (`TrainingSet` without `idf_weights_by_rank`) load with the new code via
`#[serde(default)]`. Their `idf_weights_by_rank` field is `None`, so classify falls back to the
global `idf_weights` — identical behavior to pre-change. No user action required.

### Success Criteria:

#### Automated Verification:
- [x] All prior phases' automated checks still pass
- [x] `cargo test --test test_training` passes with existing goldens
- [x] `cargo test --test test_baseline_1k` passes (loads pre-existing bincode model)
- [x] New tests in `tests/test_algorithmic_improvements.rs`:
  - [x] `test_per_rank_idf_off_by_default` — `TrainConfig::default().per_rank_idf == false`
        and `TrainingSet.idf_weights_by_rank == None`.
  - [x] `test_per_rank_idf_populated_when_on` — train with `per_rank_idf = true`,
        assert `TrainingSet.idf_weights_by_rank.is_some()`, each row is `n_kmers` long,
        and the deepest row matches the global IDF (sanity check).
  - [x] Backward compat via `#[serde(default)]` — pre-existing `test_baseline_1k`
        loads an earlier bincode model and continues to pass.
  - [x] `test_per_rank_idf_classification_runs` — classify the standard query set with
        per-rank IDF, verify results are well-formed (non-empty taxon, conf in [0, 200]).

#### Manual Verification:
- [ ] Train a model on the 10k fixture with `per_rank_idf = true`; inspect model size
      increase (expect ~ n_ranks × current IDF size, typically < 10 MB).
- [ ] Load a pre-Phase-4 model (e.g., `benchmarks/baselines/baseline_1k.json`-adjacent
      bincode) with the new code; confirm classification output is unchanged.
- [ ] Benchmark sweep: Normal-tier + species-holdout F1 with `per_rank_idf = true` vs
      `false` on the same marker. Confirm per-rank IDF helps species-holdout without
      regressing Normal.

---

## Testing Strategy

### Unit Tests:
- Per phase: new tests added alongside existing `tests/test_*.rs` files (see per-phase
  Success Criteria).
- Golden JSON tests (`tests/golden_json/s09*.json`): MUST pass unchanged at default flag
  values for all phases. Tolerance is `CONF_TOLERANCE = 5.0` on confidence; any flag
  defaulting to non-legacy behavior that shifts confidence > 5.0 is a regression.
- `ClassificationResult` changes (new `reject_reason`, `similarity`, and future bincode
  path additions) use `#[serde(default, skip_serializing_if = ...)]` so `GoldenClassResult`
  deserializers continue to ignore them.

### Integration Tests:
- `tests/test_integration.rs` (`s10a_e2e_tsv`): TSV schema changes in Phase 1 require
  regenerating this one golden. The test's confidence tolerance 5.0 remains valid.
- End-to-end benchmark validation: run `run_classifier_benchmark.sh` on
  `assignment-tool-benchmarking` against a marker with all four tier types populated;
  confirm each phase's expected holdout-tier F1 delta.

### Manual Testing Steps:
1. **Phase 1 output-layer fix**: Train a model on a 1k fixture, classify a query whose true
   species is absent, inspect TSV — confirm empty-path row carries `c0 > 0` and
   `reject_reason = "below_threshold"`. Confirm `similarity` column populated.
2. **Phase 1 rank_thresholds fix**: With `rank_thresholds = [60, 70, 80, 90, 60, 50, 40]`
   on a crafted query where the middle rank dips below threshold, inspect the output path —
   must be a contiguous prefix, never skipping a rank.
3. **Phase 2 tie_margin**: Sweep `[0.0, 0.01, 0.02, 0.05, 0.10]` on the benchmark
   species-holdout tier; observe monotonic improvement in species_holdout_genus_f1.
4. **Phase 2 sibling_aware_leaf**: On a genus node with 3 siblings 51/49/0 split, with flag
   on, verify `alternatives` contains both top-2 siblings.
5. **Phase 3 Bhattacharyya**: Train with `redundancy_metric = "bhattacharyya"` and inspect
   selected k-mers at a binary-split node; confirm they differ from round-robin (proof that
   the metric is actually doing work, not degenerating).
6. **Phase 4 per-rank IDF backward compat**: Load a pre-Phase-4 model with the new classifier
   binary; confirm output on a test query set is bit-identical to the pre-change output.

## Performance Considerations

- **Phase 1**: Pure output-layer. No classification hot-path impact.
- **Phase 2**:
  - I5 adds one float comparison per `tot_hits` entry at leaf phase. Negligible.
  - I6 adds per-rank margin tracking during descent — one sort of `vote_counts` per node
    (already O(n_children log n_children), dominated by profile scoring which is much larger).
  - I7 widens `w_indices` at single-winner sites, so leaf phase scores against more training
    sequences. Worst case when multiple siblings pass 50 % threshold: up to ~2-3× leaf-phase
    cost. Flag defaults off.
- **Phase 3**: Bhattacharyya is comparable to Pearson per pair after precomputation of
  sqrt-profiles (O(n_candidates × n_children) one-time). Expected end-to-end training time
  within ±20% of Pearson baseline on the 10k benchmark.
- **Phase 4**: Per-rank IDF computation adds O(R × K × n_seqs) to training (R = n_ranks,
  typically 7); model size grows by factor R (e.g., 10 MB → 70 MB for a 100k-kmer model).
  Classification cost unchanged (single IDF lookup per k-mer, same as global).

## Migration Notes

- **Phase 1**: TSV schema change. Downstream parsers (e.g., `assignment_benchmarks`
  `tools/oxidtaxa/parser.py:43-106`) must be updated to handle the two new columns
  (`reject_reason`, `similarity`), or set `dialect="excel-tab"` with `fieldnames` defaulting
  to the legacy 4-column schema and ignore extras. No breaking change if consumers skip
  trailing columns.
- **Phase 2**: No migration. New flags, default off.
- **Phase 3**: No migration for existing trained models (`redundancy_metric` is a
  `TrainConfig` knob — models already trained with Pearson stay Pearson). Users wanting
  Bhattacharyya retrain.
- **Phase 4**: Backward-compatible. Existing bincode models load as-is with
  `idf_weights_by_rank = None`. Forward-compatibility: new models trained with
  `per_rank_idf = true` will NOT load in old oxidtaxa binaries (they'll fail to deserialize
  the trailing field). Document this in the README.

## References

- Research: `thoughts/shared/research/2026-04-19-oxidtaxa-logic-holdout-robustness.md`
- Benchmark harness research: `thoughts/shared/research/2026-04-17-species-genus-holdout-tier-improvements.md`
- Abstention path research: `thoughts/shared/research/2026-04-17-abstention-path-output-handling.md`
- Parameter audit: `thoughts/shared/research/2026-04-15-new-parameter-audit.md`
- Tied-species plan (prior similar rollout): `thoughts/shared/plans/2026-04-08-tied-species-reporting.md`
- Algorithmic improvements plan (pattern template): `thoughts/shared/plans/2026-04-13-algorithmic-improvements.md`
- Bhattacharyya / Hellinger literature:
  - Peng, Long, Ding 2005, IEEE TPAMI 27(8):1226-1238 (mRMR foundation)
  - Brown et al. 2012, JMLR 13:27-66 (information-theoretic feature-selection survey)
  - Fu, Wu, Liu 2020, BMC Bioinformatics 21:121 (Hellinger for bioinformatics feature selection)
  - Lin 1991, IEEE Trans. Inf. Theory 37(1):145-151 (JS divergence original)
  - Endres & Schindelin 2003, IEEE Trans. Inf. Theory 49(7):1858-1860 (sqrt-JS metric)
  - Zielezinski et al. 2019, Genome Biology 20:144 (alignment-free benchmarks)

## Key Code References

- Classification entry: `src/classify.rs:33-83` (`id_taxa`)
- Greedy descent: `src/classify.rs:168-251` (`classify_one_pass`)
- Leaf-phase scoring: `src/classify.rs:449-752` (`leaf_phase_score`)
- Confidence propagation: `src/classify.rs:673-686`
- LCA cap construction: `src/classify.rs:696-718`
- Above-threshold filter (I3 target): `src/classify.rs:720-733`
- Below-threshold fallback (I1 target): `src/classify.rs:741-748`
- Similarity scalar (I8 source): `src/classify.rs:657-660`
- TSV writer (I1/I2/I8 target): `src/fasta.rs:60-110`
- Training tree construction: `src/training.rs:747-1108` (`create_tree`)
- Pearson stats (I9 target): `src/training.rs:693-722`
- Correlation-aware selection (I9 target): `src/training.rs:857-1025`
- IDF computation (I10 target): `src/training.rs:309-351`
- `ClassificationResult` (I2/I8 additions): `src/types.rs:153-176`
- `ClassifyConfig` (I5/I6/I7 additions): `src/types.rs:274-312`
- `TrainConfig` (I9/I10 additions): `src/types.rs:215-272`
- `TrainingSet` (I10 addition): `src/types.rs:22-46`
- PyO3 `train` signature: `src/lib.rs:62-110`
- PyO3 `classify` signature: `src/lib.rs:135-183`
- Enum parsing helpers: `src/lib.rs:298-316`
- Test harness: `tests/common/mod.rs`, `tests/test_classify.rs`, `tests/test_integration.rs`
- Golden fixtures: `tests/golden_json/s09*.json`, `tests/golden_json/s10a_e2e_tsv.json`
- Criterion benches: `benches/oxidtaxa_bench.rs:243-280, :405-436`
