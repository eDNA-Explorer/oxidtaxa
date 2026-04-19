---
date: 2026-04-14T12:00:00-07:00
researcher: Claude
git_commit: ee8c3cb9754248634285be33ef6f2b9e8e285506
branch: main
repository: oxidtaxa
topic: "How can we get confidence scores for unclassified ASVs?"
tags: [research, codebase, classification, confidence, unclassified, threshold]
status: complete
last_updated: 2026-04-14
last_updated_by: Claude
---

# Research: Confidence Scores for Unclassified ASVs

**Date**: 2026-04-14
**Researcher**: Claude
**Git Commit**: ee8c3cb
**Branch**: main
**Repository**: oxidtaxa

## Research Question

OxidTaxa returns `score=0.0` for all unclassified ASVs. How can we get confidence scores for unassigned sequences? What would this look like?

## Summary

There are **two distinct populations** of unclassified ASVs, and the answer differs fundamentally for each:

1. **Threshold-truncated** (~majority of unclassified): The algorithm ran fully, computed per-rank confidences, but the values fell below the threshold. The confidence data **already exists inside the algorithm** — it is simply discarded at `classify.rs:741-749` when building the result.

2. **No-signal** (~minority): `classify_one_pass` returned `None` because the sequence was too short or had no k-mer overlap with training data. There is genuinely no confidence to report.

The key insight: **for threshold-truncated sequences, the full confidence vector exists at `classify.rs:674` before the threshold is applied.** A sequence unclassified at Species with 55% confidence currently looks identical in the output to one with 0 k-mer overlap — both show `confidence=0`. That 55% is computed and then thrown away.

## Detailed Findings

### The Three Ways a Sequence Becomes "Unclassified"

#### Path A: Too few k-mers → `None` → hardcoded 0.0

At `classify.rs:186-188`, if `my_kmers.len() <= s`, `classify_one_pass` returns `None` immediately. The caller substitutes `ClassificationResult::unclassified()` (`types.rs:64-70`), which is `["Root", "unclassified_Root"]` with `confidence: [0.0, 0.0]`.

**No intermediate data exists.** The algorithm never ran.

#### Path B: No training sequences match → `None` → hardcoded 0.0

At `classify.rs:483`, if `keep.is_empty()` after full-length filtering, `leaf_phase_score` returns `None`. Same hardcoded 0.0 result.

**No intermediate data exists.** The leaf phase started but found nothing to score against.

#### Path C: Below threshold → truncated with discarded confidences

This is the most common and most interesting case. The full classification pipeline runs:

1. Tree descent completes, landing at some node (`classify.rs:192-248`)
2. Leaf-phase scoring computes `tot_hits` per group (`classify.rs:613-644`)
3. Winner is selected, lineage is built root-to-leaf (`classify.rs:662-671`)
4. **Full confidence vector is computed** at `classify.rs:673-686`:
   ```rust
   let base_confidence = tot_hits[selected] / b as f64 * 100.0;
   let mut confidences = vec![base_confidence; predicteds.len()];
   // ancestor accumulation...
   ```
5. **Threshold filter discards sub-threshold ranks** at `classify.rs:720-733`
6. Result is truncated at `classify.rs:741-749`, appending `"unclassified_<LastPassingTaxon>"`
7. TSV output at `fasta.rs:84-89` strips `unclassified_*` entries and reports min confidence of remaining ranks (or literal `0` if none remain)

**The full confidence vector, the full predicted lineage, and the similarity score all exist but are discarded.**

### What Data Is Available but Currently Discarded

For threshold-truncated sequences (Path C), these values exist at classification time:

| Value | Location | Description |
|-------|----------|-------------|
| `confidences` (full vector) | `classify.rs:674` | Per-rank confidence for the entire predicted lineage, including sub-threshold ranks |
| `predicteds` (full lineage) | `classify.rs:664-671` | Node indices for the full root-to-leaf prediction |
| `similarity` | `classify.rs:657-660` | Overall k-mer overlap metric (sum of winner's per-replicate scores / davg) |
| `tot_hits` (all groups) | `classify.rs:613` | Per-group normalized scores — shows how much support each taxonomic group received |
| `winners` list | `classify.rs:648-649` | All groups tied at maximum, not just the selected one |
| `base_confidence` | `classify.rs:673` | The raw leaf-level confidence before ancestor accumulation |

### What This Would Look Like: Implementation Options

#### Option 1: Surface the full pre-threshold confidence vector

Add fields to `ClassificationResult`:

```rust
pub struct ClassificationResult {
    pub taxon: Vec<String>,           // (existing) truncated at threshold
    pub confidence: Vec<f64>,         // (existing) truncated at threshold
    pub alternatives: Vec<String>,    // (existing)
    // NEW:
    pub raw_taxon: Vec<String>,       // full predicted lineage, all ranks
    pub raw_confidence: Vec<f64>,     // confidence at every rank, pre-threshold
}
```

For an ASV that predicted `Root > Bacteria > Proteobacteria > Gammaproteobacteria > Enterobacterales > Enterobacteriaceae > Escherichia` with confidences `[100, 95, 88, 72, 55, 40, 30]` and a 60% threshold:

- **Current output**: `taxon = [Root, Bacteria, Proteobacteria, Gammaproteobacteria, unclassified_Gammaproteobacteria]`, `confidence = [100, 95, 88, 72, 72]`
- **With raw fields**: same as above, PLUS `raw_taxon = [Root, Bacteria, ..., Escherichia]`, `raw_confidence = [100, 95, 88, 72, 55, 40, 30]`

**Pros**: Maximum information. Users can analyze the full confidence gradient.
**Cons**: Doubles the output size. Requires serde/Python binding changes.

#### Option 2: Report the leaf-level confidence as a scalar

Add a single `raw_score: f64` field:

```rust
pub struct ClassificationResult {
    pub taxon: Vec<String>,
    pub confidence: Vec<f64>,
    pub alternatives: Vec<String>,
    pub raw_score: f64,   // NEW: base_confidence at deepest predicted rank
}
```

For the same example, `raw_score = 30.0` (the Species-level confidence that was below threshold). For Path A/B sequences, `raw_score = 0.0`.

**Pros**: Minimal change. Single number that answers "how confident was the algorithm's best guess?"
**Cons**: Loses the per-rank gradient.

#### Option 3: Report similarity score

The `similarity` value (`classify.rs:657-660`) is already computed but only used for strand selection. It measures raw k-mer overlap independent of the threshold mechanism.

```rust
pub struct ClassificationResult {
    pub taxon: Vec<String>,
    pub confidence: Vec<f64>,
    pub alternatives: Vec<String>,
    pub similarity: f64,   // NEW: k-mer overlap with best-matching training group
}
```

**Pros**: Already computed. Provides a quality-of-match metric orthogonal to confidence.
**Cons**: Not on the same scale as confidence percentages (it's a sum-of-IDF-weighted-scores / davg, not a 0-100 percentage). Requires calibration to be interpretable.

#### Option 4: Classify with threshold=0, post-filter in Python

No code changes needed. Users can already call `classify(threshold=0)` to get confidence scores for everything, then post-filter in their analysis notebook. Every sequence would get a full lineage with real confidence values.

**This works today.** The downside is that `threshold=0` means the `taxon` field won't have `unclassified_*` sentinels — you'd need to apply your own threshold to determine what's "classified" vs not. But for analysis purposes (histograms, gradient curves), this gives you the raw confidence at every rank.

#### Option 5: Reject reason enum

Add context for WHY something is unclassified:

```rust
pub enum RejectReason {
    None,                        // classified successfully
    BelowThreshold { rank: usize, confidence: f64 },
    TooFewKmers { n_kmers: usize, required: usize },
    NoTrainingMatch,
}
```

This distinguishes the three paths and provides the most diagnostic information for analysis.

### Where the Code Would Change

For any of Options 1-3 or 5:

1. **`types.rs:48-61`** — Add new field(s) to `ClassificationResult`
2. **`types.rs:64-70`** — Update `unclassified()` factory to populate new fields (0.0 / None / RejectReason)
3. **`classify.rs:735-749`** — Store full `confidences` / `predicteds` / `similarity` before truncation
4. **`classify.rs:449`** — Change `leaf_phase_score` return type to include new data
5. **`fasta.rs:59-110`** — Add new columns to TSV output
6. **`lib.rs:89-158`** — Expose new fields through Python bindings
7. **Tests** — Update golden values in test_classify.rs, test_integration.rs, test_baseline_1k.rs

### For the Notebook Analysis (Cell 8)

The immediate path for the unclassified ASV investigation doesn't require code changes:

**Use `threshold=0`** to classify everything, then partition results by your actual desired threshold in Python. This gives you:

- Real confidence scores for every ASV at every rank
- The ability to build histograms of confidence for "would-be-unclassified" ASVs
- A cumulative read fraction curve across the full confidence spectrum
- A direct view of the confidence gradient near the threshold boundary

```python
# Classify with no threshold (everything gets a score)
results_raw = classifier.classify(sequences, threshold=0.0)

# Partition into classified/unclassified at your chosen threshold
threshold = 60.0
for r in results_raw:
    min_conf = min(r.confidence[1:])  # skip Root
    if min_conf >= threshold:
        classified.append(r)
    else:
        unclassified_with_scores.append(r)
```

## Code References

- `src/types.rs:48-61` — `ClassificationResult` struct definition
- `src/types.rs:64-70` — `unclassified()` factory (hardcoded 0.0)
- `src/classify.rs:186-188` — "too few k-mers" early return (`None`)
- `src/classify.rs:483` — "no training sequences" early return (`None`)
- `src/classify.rs:601-686` — Confidence computation (full vector exists here)
- `src/classify.rs:673-674` — `base_confidence` and full `confidences` vector creation
- `src/classify.rs:720-733` — Threshold filter (where sub-threshold values are discarded)
- `src/classify.rs:735-749` — Result construction (truncation happens here)
- `src/classify.rs:657-660` — Similarity score computation
- `src/classify.rs:763` — Sequential fallback to `unclassified()`
- `src/classify.rs:834-837` — Parallel fallback to `unclassified()`
- `src/fasta.rs:59-110` — TSV output formatting

## Architecture Documentation

The classification pipeline has a clean separation between scoring and thresholding:
- `leaf_phase_score` (lines 449-752) handles all computation, including the threshold decision
- The threshold logic (lines 720-749) is at the *end* of `leaf_phase_score`, making it feasible to extract pre-threshold data without restructuring the algorithm
- The `similarity` score is computed at line 657-660 but only used for strand selection at line 784-791, never exposed in output

## Open Questions

1. **Which option should we implement?** Option 4 (threshold=0) works immediately for analysis. Options 1-3 require code changes but provide a better long-term interface.
2. **Should `similarity` be exposed regardless?** It's already computed and wasted — it's a useful quality metric even for classified sequences.
3. **Backward compatibility**: Adding fields to `ClassificationResult` with `#[serde(default)]` would maintain backward compatibility with existing serialized models and outputs.
4. **For the notebook specifically**: Would a `threshold=0` run be sufficient for the Cell 8 analysis, or do you want the structural changes for long-term use?
