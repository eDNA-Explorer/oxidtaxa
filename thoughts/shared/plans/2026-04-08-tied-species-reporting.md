---
date: 2026-04-08
author: Ryan Martin
status: implemented
related_research:
  - thoughts/shared/research/2026-04-08-tied-species-reporting-at-truncated-rank.md
tags: [plan, classify, ties, lca, output, tsv]
---

> **Implementation note (2026-04-08):** While implementing Phase 2 and writing the Phase 4 synthetic test, the synthetic tied-species case revealed that the original `tot_hits` accumulator used a strict `v > max_val` per-replicate winner-take-all that never produced a tied `tot_hits` vector when training sequences were bit-identical (group 0 always swept every replicate, giving `tot_hits = [full, 0, ..., 0]` instead of the split needed for the `winners` filter to fire). R's `IdTaxa` uses `max.col(..., ties.method = "random")` which randomly picks a tied column per replicate; in expectation each tied column receives equal credit. The port now deterministically splits per-replicate credit equally among tied max columns at `src/classify.rs:360-389`, which matches that expected value, is bit-stable, and leaves all 12 existing `s09*_ids_*.json` goldens unchanged (verified — no per-replicate ties occur in those fixtures). This was not part of the original plan but was required for Phase 4's test to meaningfully exercise identical-sequence ties.


# Tied-Species Reporting + LCA-Cap Implementation Plan

## Overview

When the classifier finds multiple reference groups with **exactly tied** `tot_hits` scores, preserve the identities of the tied species through to the output, and cap the reportable lineage at the lowest common ancestor (LCA) of the tied set so the classifier never reports a species-level assignment it cannot actually defend.

Three coupled changes:

1. **LCA cap.** When `winners.len() > 1` at `src/classify.rs:373`, the output lineage is truncated at the LCA of those winners regardless of whether per-rank confidence clears the threshold. The random `selected` picker stays in place (it's effectively invisible now because `predicteds` is never reported below the LCA), and all existing confidence math is untouched.
2. **Alternatives field.** A new `alternatives: Vec<String>` field on `ClassificationResult` carries the sorted short-labels of all tied leaf nodes (e.g. `["Canis_latrans", "Canis_lupus"]`). This field propagates through to a new 4th column in the TSV output.
3. **PyO3 in-memory return.** `ClassificationResult` becomes a `#[pyclass]` with native Python getters, and the `classify()` Python function returns `List[ClassificationResult]` instead of `None`. `output_path` becomes optional — when provided, the TSV is still written for backward compatibility with existing Dagster and script callers; when omitted, results are returned in-memory only. Python clients can then read `result.alternatives` as a native `list[str]` instead of having to split a pipe-separated string.

Clients can then answer "did we detect species X?" via (a) exact match in `taxon` when the classifier could resolve, or (b) membership in `alternatives` when it couldn't. "Resolved at genus + can't distinguish species" becomes a first-class, reportable outcome — and accessible as structured data from Python, not as a text blob that has to be re-parsed.

## Current State Analysis

See `thoughts/shared/research/2026-04-08-tied-species-reporting-at-truncated-rank.md` for the full map. Key facts that drive this plan:

- **Tie detection exists, tie preservation does not.** `src/classify.rs:371-380` collects tied `winners`, picks one randomly via `rng.sample_int_replace`, and discards the rest from output. Zero grep hits for `alternatives`/`ambiguous`/`sibling`/`LCA` across the whole repo.
- **All 5 `ClassificationResult` construction sites are in one file.** `src/classify.rs:425, 436, 451, 522, 525`. No tests, examples, or benches build it directly. Blast radius for adding a field = one file.
- **Golden JSON tests use a separate `GoldenClassResult` struct** (`tests/test_classify.rs:10-14`) that only deserializes `taxon` and `confidence`. Serde silently ignores unknown fields on deserialization → adding `alternatives` to `ClassificationResult` does NOT break the 12+ `s09*_ids_*.json` classification goldens.
- **Exactly one TSV consumer parses columns**: `tests/test_integration.rs:83-94` hard-asserts the 3-column header and positional `parts[0..=2]`. `benchmarks/run_real_data_idtaxa.py:149-150` only counts lines (no column parsing); `benchmarks/run_benchmark.sh` treats the file as opaque; no Python code under `python/` reads the TSV at all. So the TSV schema change ripples to exactly one test file + one golden fixture.
- **The LCA of winners is guaranteed to live inside `predicteds`.** Because `predicteds` is the root→`selected` lineage and the LCA is by definition an ancestor of every winner (including `selected`), every LCA computation reduces to a `parents[]` walk starting at `parents[unique_groups[j]]` for each non-selected winner, stopping at the first node found in `predicteds` — exactly the same walk pattern already used by the confidence propagation loop at `src/classify.rs:400-411`.
- **Confidence math stays correct unmodified.** At the LCA node, `confidences[lca_idx] = base_confidence + Σ_{j≠selected} tot_hits[j]/b*100 = Σ_{j∈winners} tot_hits[j]/b*100`. Because all tied winners share the LCA as an ancestor, every winner's contribution naturally lands there, giving an aggregate confidence that correctly reflects "it's one of the tied set" without any new formula.
- **No `#[serde(default)]` patterns currently used** anywhere in `src/`. This plan introduces the first use, but it's isolated to one field and well-understood.
- **`ClassificationResult` is currently not `#[pyclass]`-exposed.** The PyO3 layer at `src/lib.rs:54-113` uses it internally and writes a TSV; Python sees only the file path and gets `None` back from `classify()`. Adding a `#[pyclass]` wrapper is mechanical PyO3 boilerplate — the struct has only `Vec<String>` and `Vec<f64>` fields, both of which have native PyO3 conversions. `#[pyclass(get_all)]` provides the Python-side field getters with zero custom code.
- **PyO3 0.24 is already wired** (`Cargo.toml:16`, `features = ["python"]`) with the `extension-module` feature. No new dependencies or toolchain changes required.
- **No existing Python pytest setup.** `python/oxidtaxa/__init__.py` and `python/idtaxa/__init__.py` are bare re-exports. There is no `python/tests/` directory, no `conftest.py`, no `[tool.pytest]` block in `pyproject.toml`. Python-level testing in this plan is limited to one manual smoke test documented as a verification step — no pytest harness is introduced.
- **`output_path` currently is a required positional arg** at `src/lib.rs:56,60`. Changing it to optional-with-default is a non-breaking signature change from the Python side (existing callers that pass it by keyword continue to work; existing callers that pass it positionally also continue to work because it stays in the same position).

## Desired End State

After this plan is complete:

1. **Data model.** `ClassificationResult` (in `src/types.rs`) has three fields: `taxon: Vec<String>`, `confidence: Vec<f64>`, `alternatives: Vec<String>`. The last is `#[serde(default, skip_serializing_if = "Vec::is_empty")]` — empty for the non-tied case, populated with sorted short-labels for the tied case.
2. **Classifier behavior for exact ties.**
   - When `winners.len() == 1`: zero behavioral change. Byte-identical output to pre-change for all non-tied classifications. Goldens `s09*_ids_*.json` and `s10a_e2e_tsv.json` continue to pass numerically.
   - When `winners.len() > 1`: the `above` filter is capped at the LCA position in `predicteds`. The output `taxon` ends at the LCA, followed by the `unclassified_{taxa[lca_node]}` placeholder. The `alternatives` field contains every tied leaf's short label, alphabetically sorted.
3. **TSV output.** `write_classification_tsv` emits a 4th column `alternatives` — pipe-separated (`|`) list of species short-labels, empty string when not populated. Header becomes `"read_id\ttaxonomic_path\tconfidence\talternatives\n"`. All 4 columns are always present (tab-count stable) so positional parsers don't break on empty-tie rows.
4. **PyO3 API.** `ClassificationResult` is a `#[pyclass(get_all)]`-exposed struct. Python callers see:
   ```python
   import oxidtaxa
   
   # In-memory only
   results = oxidtaxa.classify(query_path="q.fa", model_path="m.bin")
   for r in results:
       print(r.taxon)          # list[str], native
       print(r.confidence)     # list[float], native
       print(r.alternatives)   # list[str], native
   
   # In-memory + TSV (backward compatible)
   results = oxidtaxa.classify(
       query_path="q.fa",
       model_path="m.bin",
       output_path="out.tsv",  # still written
   )
   # results is always returned regardless of output_path
   ```
   The `classify()` signature gains `output_path: Option<String>` (was required) and returns `PyResult<Vec<ClassificationResult>>` (was `PyResult<()>`).
5. **Tests.** A new `tests/test_ties.rs` file exercises a synthetic tied-species training set end-to-end and asserts both the Rust struct and the TSV column are populated correctly for: (a) 2-way tie under a common genus, (b) 3-way tie, (c) cross-genus tie that forces LCA back to family, and (d) non-tied classification in the same run (regression guard). `tests/test_integration.rs` is updated to assert the 4-column header and parse `parts[0..=3]`.
6. **Docs.** `README.md` gets one paragraph describing the new column, the in-memory Python API, and when ties populate.
7. **Verification commands pass:**
   - `cargo fmt --check`
   - `cargo clippy --all-targets -- -D warnings`
   - `cargo test` (default features — no python)
   - `cargo check --features python` — the PyO3 layer compiles cleanly with the new `#[pyclass]` attribute and signature changes
   - `cargo clippy --features python --all-targets -- -D warnings`
   - Manual: `maturin develop --features python && python -c "import oxidtaxa; help(oxidtaxa.classify)"` — the Python docstring shows the new signature, `ClassificationResult` is importable, and a minimal classify call returns a list

### Key Discoveries (referenced throughout the plan):

- Tie site: `src/classify.rs:371-380` — `winners` vec and random `selected` pick.
- Confidence propagation walk pattern (reusable for LCA computation): `src/classify.rs:400-411`.
- Result construction fork: `src/classify.rs:424-437` — full-lineage vs truncated branch, both need `alternatives` populated.
- `unclassified()` fallback: `src/types.rs:55-62`.
- Parallel-mode strand disagreement picker: `src/classify.rs:517-523` — picks by similarity; `alternatives` attached to whichever `ClassificationResult` wins, so no special handling needed (the field rides along with the chosen struct).
- TSV writer: `src/fasta.rs:60-105`. Header at line 65, row format strings at lines 94/96/99.
- Integration test header assert: `tests/test_integration.rs:83`. Positional parse: `tests/test_integration.rs:87-94`.
- Golden: `tests/golden_json/s10a_e2e_tsv.json` (10 entries, no ties in current data).

## What We're NOT Doing

- **Near-tie / epsilon ties.** Exact float equality only, per the user's explicit answer. No `(max - v).abs() < epsilon` threshold.
- **Changing the primary-assignment selection for non-tied cases.** Byte-identical behavior when `winners.len() == 1`.
- **Replacing the random `selected` picker for tied cases.** It stays. The LCA cap makes it invisible for the tied path (nothing below LCA is ever reported), so its randomness has no observable effect.
- **Adding a Python pytest harness.** The plan does not introduce `python/tests/`, `conftest.py`, pytest dev-dependencies in `pyproject.toml`, or a CI Python test job. Python-side verification is one manual smoke test (`maturin develop && python -c "..."`) documented in the phase's manual verification checklist. Adding a full pytest harness is a separate follow-up if the team wants it.
- **Streaming or chunked classification.** `id_taxa` still materializes `Vec<ClassificationResult>` in memory before either writing the TSV or returning it to Python. Current behavior already holds all results in memory (via `write_classification_tsv`'s `String::push_str` loop at `src/fasta.rs:94`), so this plan is not a regression, but it also doesn't add streaming.
- **Removing `write_classification_tsv` or changing the TSV schema beyond adding the 4th column.** Existing Dagster pipelines that read the TSV by file path continue to work. The TSV format stays authoritative for on-disk persistence; in-memory return is an *additional* path, not a replacement.
- **Making `ClassificationResult` Python-mutable.** `#[pyclass(get_all)]` exposes field getters only, not setters. Python callers can read `r.taxon` but not assign to it. This matches the struct's semantics (it's an output artifact, not a mutable carrier).
- **New configuration flags.** No `report_ties: bool`, no `emit_alternatives: bool`. Behavior is always-on — if there's a tie, we cap and report alternatives. This matches the invariant that the classifier should never claim more than it can defend.
- **Changes to the training phase** (`src/training.rs`, `learn_taxa`). No `ranks` field population; that's a separate task even though the research noted the field is unused.
- **Regenerating R-reference golden tests for tied cases.** The research agent confirmed no existing golden asserts tied behavior. If any `s09*` golden happens to contain a tied classification, we'll discover it during test runs and update it as part of Phase 2 — but we're not preemptively hunting.
- **Changing confidence values** at any rank. The LCA cap operates on the `above` filter only; `confidences[]` is computed exactly as today.
- **Parallel-mode special handling.** Ties are resolved per-sequence inside `classify_one_pass`, which both `classify_sequential` and `classify_parallel` call. No changes needed in the outer dispatchers.
- **Migration tooling.** The TSV format change is additive (old parsers break only if they hard-assert a column count of 3; nothing in this repo does). No migration script, no versioned format.

## Implementation Approach

Incremental, testable in six phases.

- **Phase 1** is a compile-only milestone (struct field added, empty at all construction sites).
- **Phase 2** is the core behavior change (LCA cap + populate alternatives, both inside `classify_one_pass`).
- **Phase 3** propagates to the TSV output layer.
- **Phase 4** adds a targeted Rust test for the tied-species path end-to-end.
- **Phase 5** changes the PyO3 surface: `ClassificationResult` becomes a `#[pyclass]`, `classify()` returns `Vec<ClassificationResult>`, and `output_path` becomes optional.
- **Phase 6** is the README.

The two tightly-coupled changes (LCA cap and alternatives population) land in the same phase because they both depend on having `winners` in scope at `src/classify.rs:413` and share the LCA computation. Splitting them would require writing the LCA walker twice.

Phase 5 intentionally lands *after* the TSV + Rust test phases because:
1. We want the tie behavior itself verified in Rust first (where the test feedback loop is fast and doesn't depend on the Python build toolchain).
2. The PyO3 changes are purely mechanical surface work — they don't affect correctness of the core algorithm, so moving them earlier would mix concerns.
3. If the PyO3 phase hits unexpected friction (e.g., PyO3 0.24 pyclass ergonomics around returning `Vec<T>`), we can ship phases 1-4 independently and land 5-6 as a follow-up without blocking the tie feature.

## Phase 1: Data model — add `alternatives` field carrier

### Overview
Add `alternatives: Vec<String>` to `ClassificationResult`, default to empty vec at every construction site. No behavior change — the field is always empty after this phase. Compile-only milestone.

### Changes Required:

#### 1. Struct definition
**File**: `src/types.rs`

Add the field with serde attributes, and update `unclassified()`:

```rust
/// Classification result for a single query sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub taxon: Vec<String>,
    pub confidence: Vec<f64>,
    /// Short-labels of all reference groups tied at the maximum `tot_hits`
    /// score during classification. Empty for non-tied classifications.
    /// When non-empty, the classifier was unable to distinguish between these
    /// leaves and has truncated `taxon` at their lowest common ancestor.
    /// Entries are sorted alphabetically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<String>,
}

impl ClassificationResult {
    pub fn unclassified() -> Self {
        Self {
            taxon: vec!["Root".to_string(), "unclassified_Root".to_string()],
            confidence: vec![0.0, 0.0],
            alternatives: Vec::new(),
        }
    }
}
```

**Why `#[serde(default, skip_serializing_if = "Vec::is_empty")]`:**
- `default` lets existing JSON blobs that lack the field deserialize cleanly (defensive; no known callers today, but cheap insurance).
- `skip_serializing_if` keeps serialized output unchanged for the common non-tied case. If any future golden is regenerated from `ClassificationResult`, the JSON remains byte-identical to pre-change for untied rows.

#### 2. Construction sites in `src/classify.rs`

All 5 sites. The field defaults to `Vec::new()` in this phase (populated in Phase 2).

**Line 425** (full-lineage branch):
```rust
ClassificationResult {
    taxon: predicteds.iter().map(|&p| taxa[p].clone()).collect(),
    confidence: confidences,
    alternatives: Vec::new(),  // populated in Phase 2
}
```

**Line 436** (truncated branch):
```rust
ClassificationResult {
    taxon,
    confidence: conf,
    alternatives: Vec::new(),  // populated in Phase 2
}
```

**Lines 451, 522, 525** use `ClassificationResult::unclassified()`, which already handles the default via the updated `impl` block above. No changes needed at these sites.

### Success Criteria:

#### Automated Verification:
- [x] `cargo check` compiles cleanly
- [ ] `cargo fmt --check` passes *(pre-existing formatting divergences in the codebase exist in files this phase did not touch — out-of-scope for this plan)*
- [ ] `cargo clippy --all-targets -- -D warnings` passes *(pre-existing `needless_range_loop` in `src/matching.rs:204` and `too_many_arguments` in `src/training.rs:554` — out-of-scope for this plan)*
- [x] `cargo test` passes — all existing tests still green, since the new field is always empty and `GoldenClassResult` ignores unknown fields during deserialization

#### Manual Verification:
- [x] Visual diff of `src/classify.rs` construction sites shows the field added at lines 425 and 437 only
- [x] `src/types.rs` struct has exactly three public fields: `taxon`, `confidence`, `alternatives`

---

## Phase 2: LCA cap + populate alternatives

### Overview
The behavior change. In `classify_one_pass` at `src/classify.rs`, between the confidence-propagation loop and the `above` filter:

1. Compute `lca_cap: Option<usize>` — the shallowest position in `predicteds` reachable as an ancestor from any non-selected winner. `None` when `winners.len() == 1`.
2. Compute `alternatives_vec: Vec<String>` — sorted short-labels from `taxa[unique_groups[w]]` for every `w in winners`, when `winners.len() > 1`. Empty otherwise.
3. Modify the `above` filter to additionally enforce `i <= lca_cap` when `lca_cap` is `Some`.
4. Attach `alternatives_vec` to both `ClassificationResult` construction branches.

### Changes Required:

#### 1. Compute LCA cap and alternatives between line 411 and line 413
**File**: `src/classify.rs`

Insert between the confidence loop end (line 411) and the `above` filter start (line 413):

```rust
// When multiple groups are tied at `max_tot`, the classifier cannot honestly
// resolve below the LCA of the tied set. Compute that LCA's position in
// `predicteds` so the `above` filter can cap the reportable lineage there.
//
// Every non-selected winner's pairwise LCA with `selected` is the deepest
// ancestor of `unique_groups[j]` that lives in `predicteds`. The group-wise
// LCA is the shallowest (smallest index) of those pairwise LCAs, since it
// must be an ancestor of every winner.
let (lca_cap, alternatives): (Option<usize>, Vec<String>) = if winners.len() > 1 {
    let mut deepest_allowed = predicteds.len() - 1;
    for &j in &winners {
        if j == selected { continue; }
        let mut p = parents[unique_groups[j]];
        loop {
            if let Some(pos) = predicteds.iter().position(|&x| x == p) {
                if pos < deepest_allowed { deepest_allowed = pos; }
                break;
            }
            if p == 0 || parents[p] == p { break; }
            p = parents[p];
        }
    }
    let mut alts: Vec<String> = winners
        .iter()
        .map(|&w| taxa[unique_groups[w]].clone())
        .collect();
    alts.sort();
    (Some(deepest_allowed), alts)
} else {
    (None, Vec::new())
};
```

**Why sort alphabetically:**
- Deterministic across runs, regardless of PRNG seed or parallel thread scheduling.
- Makes assertion writing trivial in tests.
- Human-readable.
- `winners` iteration order is stable (it's built in ascending `j` order at line 373), so in principle we could skip the sort, but explicit sort is cheap (≤ a few entries in practice) and makes the contract unambiguous.

**Why store short-labels, not node indices or full paths:**
- Consistency with `taxon: Vec<String>` — callers already know how to interpret these strings.
- Full paths would duplicate information already in `taxon` (since all tied species share the LCA lineage that's in `taxon`).
- Full path joining could be added later behind a config flag if clients need it; the research showed no downstream consumer today.

#### 2. Modify the `above` filter at lines 413-422
**File**: `src/classify.rs`

```rust
let above: Vec<usize> = confidences.iter().enumerate()
    .filter(|(i, &c)| {
        // Cap: when there's a tie, never report below the LCA.
        if let Some(cap) = lca_cap {
            if *i > cap { return false; }
        }
        let thresh = match &config.rank_thresholds {
            Some(rt) if *i < rt.len() => rt[*i],
            Some(rt) if !rt.is_empty() => *rt.last().unwrap(),
            _ => config.threshold,
        };
        c >= thresh
    })
    .map(|(i, _)| i).collect();
```

**Interaction with `rank_thresholds`:** The LCA cap takes effect before the per-rank threshold check. If the LCA happens to be at a rank whose threshold the confidence doesn't clear, the full truncated-branch logic at lines 429-437 handles the empty-`above` fallback (`let w = if above.is_empty() { vec![0] } else { above };`) — output falls back to `[Root, "unclassified_Root"]`. Correct.

#### 3. Populate `alternatives` at the two construction sites
**File**: `src/classify.rs`

**Line 425** (full-lineage branch):
```rust
let result = if above.len() == predicteds.len() {
    ClassificationResult {
        taxon: predicteds.iter().map(|&p| taxa[p].clone()).collect(),
        confidence: confidences,
        alternatives: alternatives.clone(),
    }
} else {
    let w = if above.is_empty() { vec![0] } else { above };
    let last_w = *w.last().unwrap();
    let mut taxon: Vec<String> = w.iter().map(|&i| taxa[predicteds[i]].clone()).collect();
    taxon.push(format!("unclassified_{}", taxa[predicteds[last_w]]));
    let mut conf: Vec<f64> = w.iter().map(|&i| confidences[i]).collect();
    conf.push(confidences[last_w]);
    ClassificationResult { taxon, confidence: conf, alternatives }
};
```

Note: only one branch needs `.clone()` because `alternatives` is moved into the truncated branch (the common case for tied outputs). The full-lineage branch for a tied case is unreachable in practice (LCA cap guarantees at least the leaf is cut for any real tie), but we still populate it defensively in case someone reuses this code path in the future.

**Observation for reviewers:** With the LCA cap in effect, any classification where `alternatives.len() > 1` will fall into the truncated `else` branch, because `lca_cap < predicteds.len() - 1` (the LCA is strictly above at least the selected winner's leaf). So in the tied case, `above.len() < predicteds.len()`, and execution always takes the truncated branch. The full-lineage branch receives an empty `alternatives` in practice. This is the intended invariant: **if `alternatives` is non-empty, `taxon` ends in `unclassified_{lca_name}`**.

### Success Criteria:

#### Automated Verification:
- [x] `cargo check` compiles cleanly
- [ ] `cargo fmt --check` passes *(pre-existing formatting divergences — out-of-scope for this plan)*
- [ ] `cargo clippy --all-targets -- -D warnings` passes *(pre-existing errors in other files — out-of-scope for this plan)*
- [x] `cargo test` passes:
  - Non-tied classifications in the `s09*_ids_*.json` goldens produce byte-identical results (verified: all 13 classify tests pass)
  - `test_full_pipeline_e2e` in `tests/test_integration.rs` continues to pass (Phase 3 extends it to the 4-column assertion)
  - **Note:** the tot_hits accumulator was updated in Phase 2 to split credit equally among per-replicate tied max columns (see implementation note at top of document). No `s09*` golden exercises a per-replicate tied path, so all 12+ existing goldens remain byte-identical.

#### Manual Verification:
- [x] Phase 4 exercises this via `two_way_tie_caps_at_genus_and_populates_alternatives`, verified:
  - `result.taxon` ends in `["...", "Canis", "unclassified_Canis"]` (no species)
  - `result.alternatives == ["Canis_latrans", "Canis_lupus"]`
  - `result.confidence[genus_idx]` reflects the sum of both winners' contributions
- [ ] Verify that a 3-way tie produces a 3-entry `alternatives` vec
- [ ] Verify that a cross-genus tie (e.g., `Canis_lupus` vs. `Vulpes_vulpes`) truncates to the family rank with `alternatives == ["Canis_lupus", "Vulpes_vulpes"]`

---

## Phase 3: TSV output column + integration-test update

### Overview
Add `alternatives` as the 4th TSV column, update the one existing TSV consumer (`tests/test_integration.rs`), and leave the golden JSON untouched (the existing 10 entries have no ties; the consumer uses `#[serde(default)]` so the missing field deserializes to an empty string that matches the TSV's empty 4th column).

### Changes Required:

#### 1. Update `write_classification_tsv`
**File**: `src/fasta.rs`

```rust
/// Write classification results as TSV.
pub fn write_classification_tsv(
    path: &str,
    names: &[String],
    results: &[crate::types::ClassificationResult],
) -> Result<(), String> {
    let mut output = String::from("read_id\ttaxonomic_path\tconfidence\talternatives\n");

    for (i, result) in results.iter().enumerate() {
        let read_id = names[i]
            .split_whitespace()
            .next()
            .unwrap_or(&names[i]);

        let alternatives_field = result.alternatives.join("|");

        let mut taxa = result.taxon.clone();
        let mut conf = result.confidence.clone();

        // Skip Root, filter unclassified
        if taxa.len() > 1 {
            taxa.remove(0);
            conf.remove(0);
            let mut filtered_taxa = Vec::new();
            let mut filtered_conf = Vec::new();
            for (t, c) in taxa.iter().zip(conf.iter()) {
                if !t.starts_with("unclassified_") {
                    filtered_taxa.push(t.as_str());
                    filtered_conf.push(*c);
                }
            }
            if !filtered_taxa.is_empty() {
                let path_str = filtered_taxa.join(";");
                let min_conf = filtered_conf
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min);
                output.push_str(&format!(
                    "{}\t{}\t{}\t{}\n",
                    read_id, path_str, min_conf, alternatives_field
                ));
            } else {
                output.push_str(&format!("{}\t\t0\t{}\n", read_id, alternatives_field));
            }
        } else {
            output.push_str(&format!("{}\t\t0\t{}\n", read_id, alternatives_field));
        }
    }

    std::fs::write(path, output).map_err(|e| format!("Write error: {}", e))?;
    Ok(())
}
```

**Why `|` separator:** `;` is already used within `taxonomic_path`; comma can appear inside taxonomy names in some databases; tab would break the TSV structure; `|` is unused in the existing taxonomy-name corpus.

**Why the 4th column is always present (even when empty):** keeps tab-count invariant across all rows. Any positional parser (including `test_integration.rs`) can rely on 4 columns. Downstream tooling that splits by `\t` gets a stable column count.

#### 2. Update integration test header assertion and parsing
**File**: `tests/test_integration.rs`

Update the `GoldenTsvRow` struct to include `alternatives` with a serde default, so the existing `s10a_e2e_tsv.json` fixture continues to deserialize (missing field → empty string):

```rust
#[derive(serde::Deserialize)]
struct GoldenTsvRow {
    read_id: String,
    taxonomic_path: String,
    confidence: f64,
    #[serde(default)]
    alternatives: String,
}
```

Update the header assertion at line 83:

```rust
assert_eq!(
    lines[0],
    "read_id\ttaxonomic_path\tconfidence\talternatives"
);
```

Update the row parser at lines 86-102 to also check the 4th column:

```rust
for (i, line) in lines[1..].iter().enumerate() {
    let parts: Vec<&str> = line.split('\t').collect();
    assert_eq!(parts.len(), 4, "expected 4 columns at row {}, got {}", i, parts.len());
    assert_eq!(parts[0], golden[i].read_id, "read_id mismatch at row {}", i);
    assert_eq!(
        parts[1], golden[i].taxonomic_path,
        "taxonomic_path mismatch at row {}",
        i
    );
    let conf: f64 = parts[2].parse().unwrap();
    assert!(
        (conf - golden[i].confidence).abs() < 5.0,
        "confidence mismatch at row {}: {} vs {}",
        i,
        conf,
        golden[i].confidence
    );
    assert_eq!(
        parts[3], golden[i].alternatives,
        "alternatives mismatch at row {}",
        i
    );
}
```

#### 3. Golden JSON fixture: **no change required**
**File**: `tests/golden_json/s10a_e2e_tsv.json`

The existing 10 entries in this fixture do not contain ties. With `#[serde(default)]` on the `alternatives` field in `GoldenTsvRow`, deserialization of the unmodified JSON yields `alternatives: ""` for every row. The TSV output produced by the updated `write_classification_tsv` will have `parts[3] == ""` for every row (since `result.alternatives` is empty when no ties exist). The assertion `parts[3] == golden[i].alternatives` reduces to `"" == ""` → passes.

If the e2e data somehow *did* surface a tie (unlikely — the fixture was generated from a small curated set with no identical-sequence species), the assertion would fail, and we'd need to either (a) update the golden to include the expected `alternatives` string, or (b) confirm the new behavior is correct and regenerate. This is a "wait and see" edge case that the test itself will catch.

### Success Criteria:

#### Automated Verification:
- [x] `cargo check` compiles cleanly
- [ ] `cargo fmt --check` passes *(pre-existing formatting divergences — out-of-scope)*
- [ ] `cargo clippy --all-targets -- -D warnings` passes *(pre-existing errors in other files — out-of-scope)*
- [x] `cargo test` passes, including `test_full_pipeline_e2e`
- [x] `cargo test --test test_integration` passes specifically

#### Manual Verification:
- [ ] Run `cargo run --example eval_training -- tests/data/test_ref.fasta tests/data/test_ref_taxonomy.tsv tests/data/test_query.fasta` and inspect `write_classification_tsv`-produced output (if the example calls it) to verify the new column is present
- [ ] Manually produce a TSV from a synthetic tied dataset and visually confirm the 4th column format uses `|` as the separator

---

## Phase 4: Synthetic tied-species test

### Overview
Add `tests/test_ties.rs` that exercises the tie path end-to-end with a small synthetic training set: train, classify, and assert both the Rust-side `ClassificationResult` and the TSV output are correct.

### Changes Required:

#### 1. New test file
**File**: `tests/test_ties.rs`

```rust
mod common;

use oxidtaxa::classify::id_taxa;
use oxidtaxa::fasta::write_classification_tsv;
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{ClassifyConfig, OutputType, StrandMode, TrainConfig};

/// Build a minimal training set:
///   Root > Mammalia > Carnivora > Canidae > Canis   > {Canis_lupus, Canis_latrans}
///                                         > Vulpes  > {Vulpes_vulpes}
///                                         > Felidae > Felis > {Felis_catus}
///
/// `Canis_lupus` and `Canis_latrans` are given IDENTICAL sequences so they
/// will tie at `tot_hits` during classification. `Vulpes_vulpes` and
/// `Felis_catus` have distinct sequences so the classifier has enough
/// context to disambiguate and so the tree has realistic shape.
fn build_tied_training_set() -> oxidtaxa::types::TrainingSet {
    // Two ~200bp synthetic sequences. Identical for Canis_lupus / Canis_latrans.
    let tied_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                    GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
                    TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA";
    let vulpes_seq = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                      GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                      GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    let felis_seq = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\
                     CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                     AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT";

    let sequences = vec![
        tied_seq.to_string(),
        tied_seq.to_string(),
        vulpes_seq.to_string(),
        felis_seq.to_string(),
    ];
    let taxonomy = vec![
        "Root;Mammalia;Carnivora;Canidae;Canis;Canis_lupus".to_string(),
        "Root;Mammalia;Carnivora;Canidae;Canis;Canis_latrans".to_string(),
        "Root;Mammalia;Carnivora;Canidae;Vulpes;Vulpes_vulpes".to_string(),
        "Root;Mammalia;Carnivora;Felidae;Felis;Felis_catus".to_string(),
    ];

    let config = TrainConfig::default();
    learn_taxa(&sequences, &taxonomy, &config, 42, false).unwrap()
}

#[test]
fn two_way_tie_caps_at_genus_and_populates_alternatives() {
    let ts = build_tied_training_set();

    // Query = the exact tied sequence → should hit both Canis species identically.
    let query = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA"
            .to_string(),
    ];
    let query_names = vec!["tied_query".to_string()];

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query,
        &query_names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true, // deterministic
    );

    assert_eq!(results.len(), 1);
    let r = &results[0];

    // Lineage should end at Canis, then the unclassified placeholder.
    assert!(
        r.taxon.contains(&"Canis".to_string()),
        "expected Canis in taxon, got {:?}",
        r.taxon
    );
    assert_eq!(
        r.taxon.last().unwrap(),
        "unclassified_Canis",
        "expected lineage to terminate at unclassified_Canis, got {:?}",
        r.taxon
    );
    assert!(
        !r.taxon.contains(&"Canis_lupus".to_string()),
        "species rank must not leak into taxon: {:?}",
        r.taxon
    );
    assert!(
        !r.taxon.contains(&"Canis_latrans".to_string()),
        "species rank must not leak into taxon: {:?}",
        r.taxon
    );

    // Alternatives should contain both tied species, sorted alphabetically.
    assert_eq!(
        r.alternatives,
        vec!["Canis_latrans".to_string(), "Canis_lupus".to_string()],
        "expected alternatives = [Canis_latrans, Canis_lupus], got {:?}",
        r.alternatives
    );
}

#[test]
fn non_tied_classification_has_empty_alternatives() {
    let ts = build_tied_training_set();

    // Query = the Vulpes sequence → no tie expected.
    let query = vec![
        "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT"
            .to_string(),
    ];
    let query_names = vec!["vulpes_query".to_string()];

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query,
        &query_names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(results.len(), 1);
    assert!(
        results[0].alternatives.is_empty(),
        "non-tied classification should have empty alternatives, got {:?}",
        results[0].alternatives
    );
}

#[test]
fn tied_alternatives_appear_in_tsv_output() {
    let ts = build_tied_training_set();

    let query = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA"
            .to_string(),
    ];
    let query_names = vec!["tied_query".to_string()];

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query,
        &query_names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let tmp_dir = std::env::temp_dir();
    let output_path = tmp_dir.join("oxidtaxa_ties_test.tsv");
    write_classification_tsv(
        output_path.to_str().unwrap(),
        &query_names,
        &results,
    )
    .unwrap();

    let content = std::fs::read_to_string(&output_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    assert_eq!(
        lines[0],
        "read_id\ttaxonomic_path\tconfidence\talternatives"
    );
    assert_eq!(lines.len(), 2, "expected header + 1 row, got {} lines", lines.len());

    let parts: Vec<&str> = lines[1].split('\t').collect();
    assert_eq!(parts.len(), 4, "expected 4 columns, got {}", parts.len());
    assert_eq!(parts[0], "tied_query");
    assert_eq!(parts[3], "Canis_latrans|Canis_lupus");

    let _ = std::fs::remove_file(output_path);
}
```

**Design notes for this test file:**

- **Uses `learn_taxa` + `id_taxa` directly**, no golden JSON. This keeps the test self-contained and makes the expected behavior readable without looking up fixture files.
- **Synthetic sequences** are simple repeated motifs, long enough to exceed the minimum k-mer count filters but short enough to read quickly. The tied pair is literally the same string, ensuring the k-mer sets are byte-identical and the `tot_hits` tie is exact.
- **The 4-sequence corpus** (2 tied Canis species + Vulpes + Felis) gives the tree realistic shape so that `create_tree`, `children`, and `parents` all populate meaningfully. A 2-sequence corpus might trigger degenerate code paths in training.
- **Strand = Top only** avoids any interaction with the bottom-strand re-classification pass at `src/classify.rs:472-478`. Ties are resolved inside `classify_one_pass`, so strand mode is irrelevant to the correctness test.
- **Deterministic mode** (`true`) for reproducible output with seed 42. Parallel mode would also produce the same `alternatives` because sorting is deterministic.
- **Four asserts per test** that are independent — each would pinpoint a different regression.
- **Does not exercise 3-way ties or cross-genus ties.** Those can be added later if they reveal bugs; the 2-way same-genus case is the common case and exercises all the critical code paths (LCA computation, `alternatives` population, truncated-branch selection, TSV emission). Adding more cases now risks making this test fragile before we've confirmed the baseline works.

**Consideration for the first test run:** If the synthetic sequences produce a degenerate training set (e.g., too few k-mers for `compute_min_sample_size` to accept, or all k-mers identical between the tied pair AND the Vulpes pair), classification might fail to descend the tree and return `ClassificationResult::unclassified()`. If that happens during Phase 4 development, lengthen the sequences and/or increase sequence diversity for the non-tied references. A 200bp repeat-motif sequence should have enough unique 8-mers to classify cleanly, but this is the known empirical risk with synthetic data.

### Success Criteria:

#### Automated Verification:
- [x] `cargo test --test test_ties` compiles and all 3 tests pass (verified: `two_way_tie_caps_at_genus_and_populates_alternatives`, `non_tied_classification_has_empty_alternatives`, `tied_alternatives_appear_in_tsv_output`)
- [ ] `cargo fmt --check` passes *(pre-existing formatting divergences — out-of-scope)*
- [ ] `cargo clippy --all-targets -- -D warnings` passes *(pre-existing errors in other files — out-of-scope)*
- [x] Full `cargo test` remains green: **54 passed, 0 failed**

#### Manual Verification:
- [ ] Inspect the TSV written to `std::env::temp_dir()/oxidtaxa_ties_test.tsv` during test run (add a `println!` temporarily if helpful, then remove) — confirm visually that the 4th column shows `Canis_latrans|Canis_lupus`
- [ ] If the synthetic training set fails to train (e.g., too few unique k-mers), adjust the sequences and document the reason in a comment

---

## Phase 5: PyO3 in-memory return + optional `output_path`

### Overview
Make `ClassificationResult` a `#[pyclass]`, expose its fields as native Python attributes, change the `classify()` Python function to return `List[ClassificationResult]`, and make `output_path` optional. Backward compatible: callers that pass `output_path` continue to get a TSV written; callers that omit it skip the write and rely on the returned list.

### Changes Required:

#### 1. `ClassificationResult` becomes a `#[pyclass]`
**File**: `src/types.rs`

Feature-gate the pyclass attribute so non-python builds don't pull in pyo3. `get_all` exposes every field as a Python getter (no setter — field is read-only from Python's perspective, matching the struct's semantics as an output artifact).

```rust
/// Classification result for a single query sequence.
#[cfg_attr(feature = "python", pyo3::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub taxon: Vec<String>,
    pub confidence: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<String>,
}
```

**Why `get_all`:** `Vec<String>` → Python `list[str]` and `Vec<f64>` → Python `list[float]` both have native `IntoPyObject` conversions in PyO3 0.24. `get_all` creates per-field getter functions automatically, no custom code.

**Why `#[cfg_attr]` rather than unconditional `#[pyclass]`:** `#[pyclass]` expands to code that requires `pyo3` in scope. Non-python builds don't depend on pyo3 (it's an `optional = true` dep at `Cargo.toml:16`), so unconditional `#[pyclass]` would break `cargo build` without `--features python`.

Optionally add a `#[pymethods]` impl block below the struct (same file) for Python-side ergonomics — a `__repr__` that shows the first few taxon entries and alternatives count, and `__len__` for the lineage depth. This is nice-to-have, not required for correctness:

```rust
#[cfg(feature = "python")]
#[pyo3::pymethods]
impl ClassificationResult {
    fn __repr__(&self) -> String {
        let path = self.taxon.join(";");
        let alts_suffix = if self.alternatives.is_empty() {
            String::new()
        } else {
            format!(" alternatives={:?}", self.alternatives)
        };
        format!("ClassificationResult(taxon=\"{}\"{})", path, alts_suffix)
    }

    fn __len__(&self) -> usize {
        self.taxon.len()
    }
}
```

#### 2. `classify()` returns `Vec<ClassificationResult>`, `output_path` becomes optional
**File**: `src/lib.rs`

Current signature at `src/lib.rs:54-61`:

```rust
#[pyfunction]
#[pyo3(signature = (
    query_path,
    model_path,
    output_path,              // required
    threshold = 60.0,
    /* ... */
))]
fn classify(
    query_path: &str,
    model_path: &str,
    output_path: &str,        // required, &str
    /* ... */
) -> PyResult<()> {           // returns nothing
    /* ... */
    crate::fasta::write_classification_tsv(output_path, &names, &results)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(())
}
```

Updated signature:

```rust
#[pyfunction]
#[pyo3(signature = (
    query_path,
    model_path,
    output_path = None,       // now optional
    threshold = 60.0,
    bootstraps = 100,
    strand = "both",
    min_descend = 0.98,
    full_length = 0.0,
    processors = 1,
    sample_exponent = 0.47,
    seed = 42,
    deterministic = false,
    length_normalize = false,
    rank_thresholds = None,
))]
#[allow(clippy::too_many_arguments)]
fn classify(
    query_path: &str,
    model_path: &str,
    output_path: Option<String>,   // now Option<String>
    threshold: f64,
    bootstraps: usize,
    strand: &str,
    min_descend: f64,
    full_length: f64,
    processors: usize,
    sample_exponent: f64,
    seed: u32,
    deterministic: bool,
    length_normalize: bool,
    rank_thresholds: Option<Vec<f64>>,
) -> PyResult<Vec<crate::types::ClassificationResult>> {
    // ... existing model load, query read, strand parsing, config build ...

    let results = crate::classify::id_taxa(
        &clean_seqs,
        &names,
        &model,
        &classify_config,
        strand_mode,
        OutputType::Extended,
        seed,
        deterministic,
    );

    // Write TSV only if output_path was provided
    if let Some(ref path) = output_path {
        crate::fasta::write_classification_tsv(path, &names, &results)
            .map_err(|e| PyValueError::new_err(e))?;
    }

    Ok(results)
}
```

**Notes on the signature change:**
- `Option<String>` (not `Option<&str>`): PyO3 prefers owned types when the value is optional, because `&str` would need a lifetime tied to the GIL frame.
- `output_path = None` in the `#[pyo3(signature = ...)]` block sets the default so Python callers don't need to pass it.
- Existing Python callers that pass `output_path="..."` positionally still work — same position as before.
- Existing callers that pass it as a keyword still work — same keyword name.
- Return type change from `PyResult<()>` to `PyResult<Vec<ClassificationResult>>`: PyO3 auto-converts `Vec<T>` where `T: IntoPyObject` into a Python `list`. `ClassificationResult` is now `#[pyclass]`-exposed, so PyO3 wraps each element in a `Py<ClassificationResult>` automatically. Callers that ignored the return value (e.g., `oxidtaxa.classify(...)` without assignment) get a list they silently drop — no breakage.
- The `#[allow(clippy::too_many_arguments)]` allow stays; this function had it before.

#### 3. Module registration unchanged
**File**: `src/lib.rs`

The `#[pymodule] fn _core` at `src/lib.rs:154-159` already registers `classify` and `train`. No changes — `wrap_pyfunction!(classify, m)` picks up the new signature automatically.

Optionally add `ClassificationResult` to the module so Python can `import oxidtaxa; oxidtaxa.ClassificationResult` for type hints and isinstance checks:

```rust
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(train, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(classify, m)?)?;
    m.add_class::<crate::types::ClassificationResult>()?;
    Ok(())
}
```

The `add_class` call exposes the type in the module namespace. Optional but useful for type hints on the Python side and for `isinstance(r, oxidtaxa.ClassificationResult)` checks.

#### 4. Python re-export — expose `ClassificationResult` from the package namespace
**File**: `python/oxidtaxa/__init__.py`

Current:
```python
from oxidtaxa._core import classify, train
__all__ = ["train", "classify"]
```

Updated:
```python
from oxidtaxa._core import classify, train, ClassificationResult
__all__ = ["train", "classify", "ClassificationResult"]
```

**File**: `python/idtaxa/__init__.py`

Same change — the `idtaxa` package is a duplicate re-export namespace for compatibility (`src/lib.rs`/research notes earlier). Keep it symmetric.

```python
from oxidtaxa._core import classify, train, ClassificationResult
__all__ = ["train", "classify", "ClassificationResult"]
```

#### 5. Docstring updates
**File**: `src/lib.rs`

Update the docstring on the `classify()` pyfunction to document the new return type and the optional `output_path`. The current docstring (if any) describes the file-out contract; it needs to reflect that results are now returned in-memory and the TSV is optional. Example:

```rust
/// Classify query sequences against a trained IDTAXA model.
///
/// Returns a list of `ClassificationResult` objects — one per input sequence,
/// in the same order as the query FASTA. Each result exposes:
/// - `taxon`: list[str] — root-to-leaf lineage
/// - `confidence`: list[float] — per-rank confidence percentages
/// - `alternatives`: list[str] — tied species short-labels when the classifier
///   could not resolve between multiple equally-scored references (empty for
///   non-tied classifications)
///
/// If `output_path` is provided, a TSV with columns
/// `read_id, taxonomic_path, confidence, alternatives` is also written to that
/// path. If omitted, no file is written and results are only returned in-memory.
#[pyfunction]
#[pyo3(signature = (/* ... */))]
fn classify(/* ... */) -> PyResult<Vec<ClassificationResult>> {
    /* ... */
}
```

### Success Criteria:

#### Automated Verification:
- [x] `cargo check` (default features, no python) still passes — the `#[cfg_attr(feature = "python", ...)]` guard keeps non-python builds pyo3-free
- [x] `cargo check --features python` passes — the pyclass derivation compiles cleanly (requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` on Python 3.14 with PyO3 0.24, which caps at 3.13; this is an environmental concern, not a code issue)
- [x] `cargo clippy --features python` passes (lib only; no errors, only the same pre-existing warnings in `src/lib.rs` about redundant `|e| PyValueError::new_err(e)` closures that existed before this plan)
- [ ] ~~`cargo build --features python --release` builds the extension module~~ **Not achievable via raw cargo:** on macOS, building the cdylib with the `extension-module` feature requires maturin's linker flags (`-undefined dynamic_lookup`). Use `maturin build --features python --release` instead. This is a PyO3+macOS environmental constraint, not a code defect.
- [x] `cargo test` passes (default features): **54 passed, 0 failed**
- [ ] ~~`cargo test --features python` passes~~ **Not achievable via raw cargo:** the `extension-module` PyO3 feature disables libpython linking, which cargo needs to build test binaries. This is the canonical PyO3 limitation — test binaries and the extension module cannot share the same feature configuration. Default `cargo test` (without `--features python`) covers all Rust-side correctness.

#### Manual Verification:
- [ ] `maturin develop --features python` builds and installs the extension into the active virtualenv without errors
- [ ] Python smoke test (run after `maturin develop`):
  ```bash
  python -c "
  import oxidtaxa
  print(oxidtaxa.ClassificationResult)
  print(oxidtaxa.classify.__doc__)
  "
  ```
  Expected: prints the class repr and the updated docstring showing the new return type and optional `output_path`.
- [ ] Python end-to-end smoke test using an existing training model (the user can point at any `.bin` model they already have):
  ```python
  import oxidtaxa
  results = oxidtaxa.classify(
      query_path="tests/data/test_query.fasta",
      model_path="tests/data/test_model.bin",  # or whatever the user has
  )
  assert isinstance(results, list)
  assert len(results) > 0
  r = results[0]
  assert isinstance(r.taxon, list)
  assert isinstance(r.confidence, list)
  assert isinstance(r.alternatives, list)
  print(f"First result: {r}")
  ```
- [ ] Backward-compat check: the same call *with* `output_path` still returns results AND writes the TSV:
  ```python
  import tempfile, os
  with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
      tsv_path = tmp.name
  results = oxidtaxa.classify(
      query_path="tests/data/test_query.fasta",
      model_path="tests/data/test_model.bin",
      output_path=tsv_path,
  )
  assert isinstance(results, list)
  assert os.path.getsize(tsv_path) > 0
  with open(tsv_path) as f:
      header = f.readline().strip()
  assert header == "read_id\ttaxonomic_path\tconfidence\talternatives"
  os.unlink(tsv_path)
  ```
- [ ] Dagster-equivalent call pattern (existing callers): verify that `oxidtaxa.classify(query_path=q, model_path=m, output_path=o)` — with all three as kwargs — still works without modification, since that's how the existing pipeline invokes it. No behavior change from the caller's perspective.
- [ ] Tied-species round-trip: train a model with the synthetic tied-species fixture from Phase 4's `build_tied_training_set`, save to bincode, load from Python, classify the tied query, verify `results[0].alternatives == ["Canis_latrans", "Canis_lupus"]` as a native Python list (no pipe-split needed). This is the actual functional goal of moving to Option B.

---

## Phase 6: README documentation

### Overview
Add a short section to `README.md` describing the new `alternatives` column, the in-memory Python API, and the LCA cap behavior. Keep it focused — one section with the concept, one code block each for the TSV and the Python API.

### Changes Required:

#### 1. README
**File**: `README.md`

Find the section that documents output formats (or add one near the top of the usage section) and add:

````markdown
### Tied-species resolution

When two or more reference sequences produce identical top-scoring matches for a query (common for marker genes where congeneric species share 100% sequence identity), the classifier will:

1. **Cap the primary assignment at the lowest common ancestor of the tied set.** For example, if *Canis lupus* and *Canis latrans* tie exactly, the `taxonomic_path` will end at `Canis` (not at either species), even if per-rank confidence at the species level would otherwise clear the threshold. The classifier never reports a species-level assignment it cannot distinguish.
2. **Report the tied species in an `alternatives` field** of the result. Entries are short-labels (e.g., `Canis_latrans`, `Canis_lupus`), sorted alphabetically.

**From Python:**

```python
import oxidtaxa

results = oxidtaxa.classify(
    query_path="queries.fa",
    model_path="model.bin",
)

for r in results:
    print(r.taxon)          # ['Root', ..., 'Canis', 'unclassified_Canis']
    print(r.confidence)     # [100.0, ..., 95.3, 95.3]
    print(r.alternatives)   # ['Canis_latrans', 'Canis_lupus']
```

Pass `output_path="results.tsv"` to also write a TSV file (existing callers continue to work unchanged).

**From the TSV output** (when `output_path` is provided):

| read_id  | taxonomic_path                                     | confidence | alternatives                    |
|----------|----------------------------------------------------|------------|---------------------------------|
| read_042 | Eukaryota;Chordata;Mammalia;Carnivora;Canidae;Canis | 100.0      | Canis_latrans\|Canis_lupus       |

The `alternatives` column is pipe-separated (`|`) and empty for non-tied classifications.

**Answering species-level presence queries:**
- **Exact match in `taxon`** (Python) or `taxonomic_path` (TSV) → the classifier could resolve to that species
- **Membership in `alternatives`** → the species was present in a tied set and could not be distinguished from its siblings on this marker
````

### Success Criteria:

#### Automated Verification:
- [x] `cargo test` still passes (README change is text-only): **54 passed, 0 failed**

#### Manual Verification:
- [ ] Render `README.md` in a Markdown preview and confirm the example table and Python code block render correctly
- [ ] Confirm the example values match what the actual classifier would emit for a real tied case (use Phase 4's synthetic fixture as a reference)
- [ ] Confirm the tone matches the rest of the README

---

## Testing Strategy

### Unit Tests (included in Phases 1-4)
- `tests/test_ties.rs::two_way_tie_caps_at_genus_and_populates_alternatives` — primary behavior assertion
- `tests/test_ties.rs::non_tied_classification_has_empty_alternatives` — regression guard
- `tests/test_ties.rs::tied_alternatives_appear_in_tsv_output` — end-to-end through the writer

### Integration Tests (updated in Phase 3)
- `tests/test_integration.rs::test_full_pipeline_e2e` — continues to pass against existing `s10a_e2e_tsv.json` golden, now with 4-column header assertion and empty-`alternatives` check per row

### Existing Golden Tests (verified in Phase 2)
- All 12+ `tests/golden_json/s09*_ids_*.json` continue to pass byte-identically in the non-tied case (`GoldenClassResult` ignores the new `alternatives` field on deserialization; `#[serde(skip_serializing_if = "Vec::is_empty")]` keeps serialized JSON unchanged for untied rows if goldens are ever regenerated)
- If any `s09*` golden exercised a tied path that was previously random-selected, the Phase 2 test run will surface the divergence and we regenerate that specific golden with a note in the commit message

### PyO3 Compile + Feature Tests (Phase 5)
- `cargo check --features python` — the `#[pyclass]` attribute compiles cleanly
- `cargo clippy --features python --all-targets -- -D warnings` — no lints
- `cargo test --features python` — existing Rust tests still pass with the python feature enabled (no Python interpreter involved; this just verifies the feature-gated code compiles and links into the test binary)

### Manual Testing Steps
1. **Synthetic identical-sequence test** (Phase 2 development):
   - Build a 2-entry training set with identical sequences and different species labels under the same genus
   - Classify a query matching both
   - Verify: `taxon` ends at `unclassified_{genus}`, `alternatives.len() == 2` and contains both species sorted alphabetically
2. **3-way tie test** (Phase 2 manual verification):
   - Same as above but with 3 identical training sequences under the same genus
   - Verify: `alternatives.len() == 3`, all 3 species present, sorted
3. **Cross-genus tie test** (Phase 2 manual verification):
   - 2 identical training sequences under different genera in the same family (e.g., one under `Canis`, one under `Vulpes`)
   - Verify: `taxon` ends at `unclassified_{family}`, `alternatives == [Canis_..., Vulpes_...]`
4. **Real-data spot check** (after Phase 3):
   - Run classification on `benchmarks/data/bench_1000*` data
   - `grep -v '^$' output.tsv | awk -F'\t' '{ if ($4 != "") print }'` to find any tied rows
   - Manually inspect a few and confirm (a) the `taxonomic_path` is strictly above the tied species' rank and (b) the `alternatives` list is sensible
5. **Python in-memory smoke test** (Phase 5 manual verification — see Phase 5 Success Criteria for detailed commands):
   - `maturin develop --features python`
   - `python -c "import oxidtaxa; print(oxidtaxa.ClassificationResult); print(oxidtaxa.classify.__doc__)"` confirms the class is exposed and the docstring is updated
   - Classify with no `output_path`, assert `isinstance(results, list)` and `isinstance(results[0].alternatives, list)`
   - Classify with `output_path`, assert both the return list and the TSV file are populated
   - Tied-species round-trip using the Phase 4 synthetic fixture saved as a bincode model, then loaded from Python — assert `results[0].alternatives == ["Canis_latrans", "Canis_lupus"]` as a native Python list
6. **Pandas TSV smoke test** (optional, after Phase 5):
   - `pd.read_csv(path, sep='\t')` on the TSV — confirm 4 columns load and `alternatives` column is a string (pipe-separated)
   - `df["alternatives"].str.split("|")` — confirms the pipe-split still works for clients that prefer the TSV path

## Performance Considerations

- **LCA computation cost.** When `winners.len() > 1`, we walk `parents[]` upward from each non-selected winner until we hit a node in `predicteds`. Worst case: `O(winners.len() * tree_depth)`. For the typical case of 2-3 tied winners and a depth-of-~8 taxonomy, this is 16-24 pointer chases per tied classification. Non-tied classifications pay zero extra cost (the `if winners.len() > 1` branch short-circuits).
- **`predicteds.iter().position()` lookup.** The current confidence propagation loop at lines 400-411 already uses this exact pattern, so the LCA computation adds no novel cost profile.
- **Alternatives vector construction.** One `Vec<String>` allocation + `winners.len()` `String::clone()` calls + one `sort()`. All O(winners.len()) with small constants. Negligible.
- **TSV output.** One extra `format!` argument per row (`join("|")` on a usually-empty vec). No measurable impact on writer throughput.
- **Memory (struct field).** One `Vec<String>` per `ClassificationResult`. Empty vec is 24 bytes (ptr + len + cap on 64-bit). For 1M classification results, that's ~23 MB baseline cost — negligible compared to the existing `taxon`/`confidence` vecs. Populated vecs add minor overhead only for the tied subset (expected to be <1% of results in real eDNA data).
- **PyO3 `#[pyclass]` overhead.** PyO3 wraps each `ClassificationResult` in a `Py<ClassificationResult>` (reference-counted pointer + object header) when returning to Python. For a list of 1M results, this is an extra ~16-24 bytes of header per result on top of the struct fields. Not free, but in the same order of magnitude as the existing `Vec<ClassificationResult>` overhead. Clients who want to minimize Python-side memory can still call with `output_path="..."` and ignore the return value (Python will drop the list on function return — the TSV becomes the persistence path).
- **`get_all` field accessor overhead.** Each call to `result.taxon` (or `.confidence`, `.alternatives`) from Python converts the underlying `Vec<String>` / `Vec<f64>` into a Python `list` object via PyO3's `IntoPyObject`. This is a per-access copy, not a view — the underlying Rust data is not shared with Python. For the common access pattern of reading a result once, this is fine. For clients that repeatedly index into `result.taxon`, caching the return into a local Python variable avoids redundant copies.
- **`classify()` return-value construction.** PyO3 builds a Python `list` containing `Py<ClassificationResult>` for every result in the vector. For 1M results this allocates 1M `Py<T>` wrappers + 1 list. Single-pass allocation, no secondary transforms. Compared to the current TSV-write path (`String::push_str` loop + one `fs::write`), the memory profile is similar — both paths hold all results in memory before yielding.
- **Regression risk on existing benchmarks.** `benches/oxidtaxa_bench.rs` does not reference `ClassificationResult` directly, so no bench changes required. The `#[pyclass]` attribute does not affect non-python builds (it's `#[cfg_attr]`-gated), so default-feature bench runs see zero overhead. Expect bench numbers to be within noise of pre-change.

## Migration Notes

**No migration tooling is required.** All changes are backward compatible for existing callers.

- **Data on disk.** Nothing. `ClassificationResult` is never bincoded or persisted (only `TrainingSet` is), so there are no old files to upgrade. `TrainingSet` is unchanged — existing `.bin` models load without regeneration.
- **TSV consumers.** The format change is **purely additive**. Any downstream tool that hard-codes 3 columns will break; any tool that splits by `\t` and ignores extra columns will continue to work. Internal to this repo: only `tests/test_integration.rs:83` hard-asserts 3 columns, and this plan updates it. External: the user runs this pipeline, so any downstream parser on their side needs the same 1-line update (add column, or use `pd.read_csv` with `usecols=[0,1,2]` to ignore the new column).
- **Python API — existing callers.** Unchanged in practice. `oxidtaxa.classify(query_path=..., model_path=..., output_path=...)` continues to work identically: the TSV is still written to `output_path`, and the call succeeds. The only difference is that the return value is now a `list[ClassificationResult]` instead of `None`. Callers that ignored the return value (`oxidtaxa.classify(...)` without assignment) silently drop the list — no breakage. Callers that asserted `result is None` would need updating, but no such call exists in the current codebase.
- **Python API — new callers.** Can omit `output_path` entirely to skip disk I/O: `results = oxidtaxa.classify(query_path=..., model_path=...)`. The return value is a native Python list of `ClassificationResult` objects with `list[str]` / `list[float]` fields.
- **Dagster/pipeline integration.** Pass-through unchanged. Existing assets that invoke `oxidtaxa.classify(..., output_path=...)` and then pass `output_path` to the next asset keep working with zero modification.
- **R reference bit-compat.** Untied classifications remain bit-identical to R. Tied classifications now diverge from R (which still uses its own random pick). This is the intended behavior change and is documented in the README.
- **Rust-side callers of `id_taxa()` and `write_classification_tsv()`.** The Rust public API is source-compatible: `id_taxa()` still returns `Vec<ClassificationResult>`, and `write_classification_tsv()` still takes `(&str, &[String], &[ClassificationResult])`. The only change is that `ClassificationResult` now has a third public field (`alternatives`), so any Rust code that pattern-matches on the struct with `ClassificationResult { taxon, confidence }` (without `..`) would need to add `alternatives` or add `..`. No such pattern exists in the current codebase.

## References

- Research doc: `thoughts/shared/research/2026-04-08-tied-species-reporting-at-truncated-rank.md`
- Tie-break site: `src/classify.rs:371-380`
- Confidence propagation walk (pattern reused for LCA): `src/classify.rs:398-411`
- Result construction fork: `src/classify.rs:424-437`
- `ClassificationResult` definition: `src/types.rs:49-62`
- TSV writer: `src/fasta.rs:60-105`
- PyO3 `classify` function + module registration: `src/lib.rs:54-113,154-159`
- Python package re-exports: `python/oxidtaxa/__init__.py:7`, `python/idtaxa/__init__.py:7`
- PyO3 dependency declaration: `Cargo.toml:16` (pyo3 0.24, `extension-module`), `Cargo.toml:30-32` (python feature gate)
- Maturin configuration: `pyproject.toml` (`module-name = "oxidtaxa._core"`, `python-source = "python"`)
- Integration test to update: `tests/test_integration.rs:10-106`
- Golden to leave untouched: `tests/golden_json/s10a_e2e_tsv.json`
- Original port plan (for historical context on the file-out API choice): `thoughts/shared/plans/2026-04-03-idtaxa-python-rust-port.md:1387-1477`
- R reference tie-break (diverges intentionally for tied cases): `reference/R_orig/IdTaxa.R:380-389`
