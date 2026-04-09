---
date: 2026-04-08T13:17:33-04:00
researcher: Ryan Martin
git_commit: a1572fb8d2f584f1a8fdf8ffa6fdc2e188e0a60f
branch: main
repository: eDNA-Explorer/oxidtaxa
topic: "Reporting tied species when lineage is truncated to a higher rank"
tags: [research, codebase, classify, ties, lca, output, taxonomy, classification-result]
status: complete
last_updated: 2026-04-08
last_updated_by: Ryan Martin
---

# Research: Reporting tied species when lineage is truncated to a higher rank

**Date**: 2026-04-08T13:17:33-04:00
**Researcher**: Ryan Martin
**Git Commit**: a1572fb8d2f584f1a8fdf8ffa6fdc2e188e0a60f
**Branch**: main
**Repository**: eDNA-Explorer/oxidtaxa

## Research Question
When two or more reference species have the exact same k-mer profile for a marker gene, the classifier's confidence propagation often forces the output lineage to be truncated at a higher rank (e.g., genus). The client context is: "we want to report the genus as the assignment, but also tell the user which species were in the tied set that we couldn't resolve between."

Research focus: **document the current state** of tied-species handling, the confidence-propagation mechanism that causes truncation, what data is in scope at the moment of result construction, and what the output format currently carries. No evaluation or proposals — just the map.

## Summary

The classifier **does not perform LCA aggregation on ties**. When multiple top-level groups have identical `tot_hits` scores, it picks one winner uniformly at random via `rng.sample_int_replace` and discards the rest from the output. The lineage truncation the user is observing is a *side effect* of confidence propagation, not explicit LCA: non-selected tied groups push their `tot_hits` contribution up the selected lineage to every shared ancestor (starting from `parents[group]`, never the group itself), so the ancestor rank — e.g., genus — passes the threshold while the species rank does not.

At the moment the `ClassificationResult` is constructed (`src/classify.rs:424`), **all data needed to enumerate the tied species is still in scope**: `winners` (the `Vec<usize>` of indices where `tot_hits[j] == max_tot`), `unique_groups` (taxonomy node indices), `tot_hits`, `top_hits_idx`, `taxa`, `parents`. None of this is preserved into the output struct.

The `ClassificationResult` struct itself has exactly two fields — `taxon: Vec<String>` and `confidence: Vec<f64>` — and the only serialization path (`write_classification_tsv` in `src/fasta.rs:60-105`) writes 3 columns: `read_id`, `taxonomic_path`, `confidence`. The `TsvRow` struct exists in `src/types.rs:65-70` but is never constructed anywhere in `src/`, `tests/`, `examples/`, or `benches/`; serialization builds strings directly with `format!()`.

The PyO3 `classify()` function (`src/lib.rs:54-113`) also returns `PyResult<()>` — no in-memory result list crosses the Python boundary. All output is the TSV file at `output_path`.

A codebase-wide search found **zero hits** for terms like "alternatives", "ambiguous", "sibling", "LCA", "lowest common ancestor", "shared confidence", or "multi-hit" in any source, test, plan, or doc — there is no existing code path that preserves tied candidates.

## Detailed Findings

### 1. The tie-breaking site and what "ties" actually means

**Location**: `src/classify.rs:371-380`

```rust
// Choose best group
let max_tot = tot_hits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let winners: Vec<usize> = tot_hits.iter().enumerate()
    .filter(|(_, &v)| v == max_tot).map(|(i, _)| i).collect();
let selected = if winners.len() > 1 {
    let idx = rng.sample_int_replace(winners.len(), 1)[0];
    winners[idx]
} else {
    winners[0]
};
```

- **What `tot_hits` is**: `Vec<f64>` of length `n_top = top_hits_idx.len()` (`src/classify.rs:359-360`). Entry `j` is the total, across `b` bootstrap reps, of `hits_flat[top_hits_idx[j] * b + rep] / davg` contributed whenever group `j` was the top-scoring group in that rep (`src/classify.rs:361-369`).
- **What `winners` is**: exact-float-equality filter — the set of group indices tied at the maximum `tot_hits`. When there is a unique maximum, `winners.len() == 1`; otherwise it contains every tied group.
- **What `selected` is**: one index, picked uniformly at random from `winners` via the seeded `RRng`. This matches R IdTaxa's `sample(w, 1)` at `reference/R_orig/IdTaxa.R:380-389`.
- **What happens to losing tied groups**: They remain in `unique_groups`, `top_hits_idx`, `tot_hits`, and `winners`, but the non-selected entries are never referenced again by name. Their `tot_hits` *values* still participate in the confidence propagation loop below.

### 2. Confidence propagation — why a tied sibling inflates the genus rank

**Location**: `src/classify.rs:388-411`

```rust
// Build prediction
let predicted_group = unique_groups[selected];
let mut predicteds: Vec<usize> = Vec::new();
let mut p = predicted_group;
loop {
    predicteds.push(p);
    if p == 0 || parents[p] == p { break; }
    p = parents[p];
}
predicteds.reverse();

let base_confidence = tot_hits[selected] / b as f64 * 100.0;
let mut confidences = vec![base_confidence; predicteds.len()];
for (j, &th) in tot_hits.iter().enumerate() {
    if th > 0.0 && j != selected {
        let mut p = parents[unique_groups[j]];
        loop {
            if let Some(m) = predicteds.iter().position(|&x| x == p) {
                confidences[m] += th / b as f64 * 100.0;
            }
            if p == 0 || parents[p] == p { break; }
            p = parents[p];
        }
    }
}
```

Critical detail (flagged by the analyzer): the inner walk starts from `parents[unique_groups[j]]`, **not from `unique_groups[j]` itself**. So a sibling species' contribution lands on the shared genus and every ancestor above it — never on the leaf/species rank of the selected lineage.

**Concrete walk-through of the user's scenario** (from the analyzer):

- Selected: `Root;...;Family_A;Genus_A;species_x`
- Tied sibling: `Root;...;Family_A;Genus_A;species_y` with `tot_hits[j] = 50`, `b = 100` ⟹ contribution = `50 / 100 * 100 = 50`
- Inner walk starts at `parents[species_y] == Genus_A`:
  - `Genus_A` is in `predicteds` → `confidences[genus_idx] += 50`
  - `parents[Genus_A] == Family_A`, in `predicteds` → `confidences[family_idx] += 50`
  - ...continues up through every shared ancestor to Root
- The `species_x` (leaf) rank is **not** touched by this sibling contribution because the walk never visits `species_y` itself.

This is exactly the mechanism that pushes genus confidence above threshold while leaving species confidence at its `base_confidence` value — producing the "truncated at genus" behavior the user asked about.

### 3. Threshold filter and result construction — where the ClassificationResult is born

**Location**: `src/classify.rs:413-437`

```rust
let above: Vec<usize> = confidences.iter().enumerate()
    .filter(|(i, &c)| {
        let thresh = match &config.rank_thresholds {
            Some(rt) if *i < rt.len() => rt[*i],
            Some(rt) if !rt.is_empty() => *rt.last().unwrap(),
            _ => config.threshold,
        };
        c >= thresh
    })
    .map(|(i, _)| i).collect();

let result = if above.len() == predicteds.len() {
    ClassificationResult {
        taxon: predicteds.iter().map(|&p| taxa[p].clone()).collect(),
        confidence: confidences,
    }
} else {
    let w = if above.is_empty() { vec![0] } else { above };
    let last_w = *w.last().unwrap();
    let mut taxon: Vec<String> = w.iter().map(|&i| taxa[predicteds[i]].clone()).collect();
    taxon.push(format!("unclassified_{}", taxa[predicteds[last_w]]));
    let mut conf: Vec<f64> = w.iter().map(|&i| confidences[i]).collect();
    conf.push(confidences[last_w]);
    ClassificationResult { taxon, confidence: conf }
};

Some((result, similarity))
```

**Data still in scope at `src/classify.rs:424`** (the moment `ClassificationResult` is built), per the analyzer:

- `winners: Vec<usize>` — tied-group indices (what the user wants to enumerate)
- `unique_groups: Vec<usize>` — taxonomy node index for each group `j`
- `top_hits_idx: Vec<usize>` — `keep`-position of each group's best training sequence
- `tot_hits: Vec<f64>` — per-group score
- `taxa: &Vec<String>` — short-label resolver (e.g., `taxa[unique_groups[j]] == "species_y"`)
- `parents: &Vec<usize>` — ancestor walker
- `predicteds: Vec<usize>` — root-to-leaf lineage of the selected group
- `above: Vec<usize>` — rank indices that met threshold (this is how the output is truncated)

The `else` branch (lines 429-437) is the "truncated" path: it takes only ranks in `above`, appends a synthetic `format!("unclassified_{}", taxa[predicteds[last_w]])` entry at the end, and duplicates the last confidence. None of `winners`, `unique_groups`, or `tot_hits` is read here — tied-species identities are simply dropped on the floor at this point.

### 4. ClassificationResult struct and output serialization

**Struct definition**: `src/types.rs:49-62`

```rust
pub struct ClassificationResult {
    pub taxon: Vec<String>,
    pub confidence: Vec<f64>,
}
```

There are no other fields. No `alternatives`, no `tied_groups`, no metadata.

**`TsvRow` definition**: `src/types.rs:65-70`

```rust
pub struct TsvRow {
    pub read_id: String,
    pub taxonomic_path: String,
    pub confidence: f64,
}
```

`TsvRow` is defined but **never imported or constructed anywhere** in `src/`, `tests/`, `examples/`, or `benches/`. The TSV writer bypasses it.

**Only serialization path**: `src/fasta.rs:60-105` — `write_classification_tsv()`

- Header: `"read_id\ttaxonomic_path\tconfidence\n"` (`src/fasta.rs:65`)
- Per-row (`src/fasta.rs:67-101`):
  - `read_id`: first whitespace token of `names[i]`
  - Clones `result.taxon`; if `taxa.len() > 1`, removes index 0 (`Root`) and its paired confidence
  - Filters out every element where `taxon.starts_with("unclassified_")` (drops the synthetic appended entry and its confidence)
  - `taxonomic_path`: `filtered_taxa.join(";")`
  - `confidence`: `min` of surviving per-rank confidences via `fold(f64::INFINITY, f64::min)`
  - If nothing remains: emits `"{read_id}\t\t0\n"`
- Written via `std::fs::write(path, output)` — not streaming, no through-struct.

**What reaches the user**: exactly `read_id`, a single `;`-joined taxonomy string (with the `unclassified_*` placeholder stripped), and one scalar confidence (the minimum across ranks).

**What is discarded before serialization**:
- The `Root` element at `taxon[0]` and `confidence[0]`
- Every `unclassified_*` element and its confidence (including the synthetic one added at `src/classify.rs:433`)
- All per-rank individual confidences — collapsed to `min`
- Rank labels (there are no labels — see §6 below)
- Any tied-group info (never entered the struct)

### 5. PyO3 bindings — no in-memory result crosses the Python boundary

**Location**: `src/lib.rs:11-160`, `#[cfg(feature = "python")]`

- `#[pymodule] fn _core` registers `train` and `classify` (`src/lib.rs:154-159`), built as `oxidtaxa._core` (`pyproject.toml:13`).
- `classify(...)` signature: `(query_path, model_path, output_path, threshold=60.0, bootstraps=100, strand="both", min_descend=0.98, full_length=0.0, processors=1, sample_exponent=0.47, seed=42, deterministic=False, length_normalize=False, rank_thresholds=None)` (`src/lib.rs:55-61`).
- Internally calls `write_classification_tsv(output_path, &names, &results)` at `src/lib.rs:109`.
- **Returns `PyResult<()>`** — no Python value on success. The Python wrapper at `python/oxidtaxa/__init__.py:7` just re-exports `classify` and `train` with no added logic; a duplicate re-export lives at `python/idtaxa/__init__.py:7`.

So Python users currently cannot access `ClassificationResult` in memory at all — their only artifact is the 3-column TSV written to disk.

### 6. Taxonomy tree structure (how to walk to tied species and find their common ancestor)

**Struct**: `src/types.rs:23-46` (`TrainingSet`). Population: `src/training.rs`.

Per-node fields (all length `taxonomy.len()`, indexed by node `p`):

| Field | Purpose | File:Line |
|---|---|---|
| `taxonomy: Vec<String>` | Full `"Root;A;B;..."` path per node, `;`-terminated, sorted by depth | `src/types.rs:24`, populated `src/training.rs:135-138` |
| `taxa: Vec<String>` | Short label only (last `;`-split component), e.g. `"Canis_lupus"` | `src/types.rs:25`, populated `src/training.rs:157-163` |
| `ranks: Option<Vec<String>>` | Would carry `"genus"`/`"species"` labels per node — **always `None`** in the Rust port (`src/training.rs:488`); R-side feature not wired up | `src/types.rs:26` |
| `levels: Vec<i32>` | Numeric depth; `Root == 1`, child of Root `== 2`, etc. Derived by counting `;`-separated components | `src/types.rs:27`, populated `src/training.rs:165-168` |
| `children: Vec<Vec<usize>>` | Immediate child node indices per node | `src/types.rs:28`, populated `src/training.rs:181-198` |
| `parents: Vec<usize>` | Parent node index; `parents[0] == 0` is the Root sentinel (loop termination pattern at `src/classify.rs:393,407`) | `src/types.rs:29`, populated `src/training.rs:200-205` |
| `sequences: Vec<Option<Vec<usize>>>` | Transitive-closure list of training sequence indices under each node | `src/types.rs:31`, populated `src/training.rs:226-240` |
| `cross_index: Vec<usize>` | Training-sequence index → exact leaf taxonomy node index (length = num training sequences, not num nodes) | `src/types.rs:34`, populated `src/training.rs:145-154` |

**Ranks in the current Rust port** are encoded only by `levels[p]` (integer depth). There is no string "genus"/"species" label per node in any produced model — the `ranks` field exists in the struct but `learn_taxa` unconditionally sets it to `None` at `src/training.rs:488`.

**Concrete example** from `tests/golden_json/s08b_asym_training_set.json:94749-94772`:

```
taxonomy[9] = "Root;Eukaryota;Chordata;Mammalia;Carnivora;Canidae;Canis;Canis_lupus;"
taxa[9]     = "Canis_lupus"
taxa[8]     = "Canis"        (its parent → genus-level node)
taxa[7]     = "Canidae"      (family-level)
```

**Walking operations relevant to reporting ties**:
- **Siblings of a tied group** under a common parent: `let parent = parents[g]; let siblings = &children[parent];` — not directly stored, computed via two lookups.
- **Short name for a group index**: `ts.taxa[g]` (e.g., `"Canis_lupus"`).
- **Full path**: `ts.taxonomy[g]` (e.g., `"Root;Eukaryota;...;Canis_lupus;"`).
- **Ancestor walk** (already used at `src/classify.rs:391-395` and `src/classify.rs:402-409`):
  ```rust
  let mut p = g;
  loop {
      // visit p
      if p == 0 || parents[p] == p { break; }
      p = parents[p];
  }
  ```

### 7. Does any existing code already track or emit tied candidates?

**No.** Per the explore agent's sweep (Rust, Python, R reference, tests, benchmarks, thoughts/ docs, README):

- **Zero hits** for `alternatives`, `ambiguous`, `sibling`, `siblings`, `candidates`, `LCA`, `lowest common ancestor`, `shared confidence`, `multiple hits`, `multi-hit`, `ambiguity` across the codebase.
- **`winners`** (`src/classify.rs:373`) exists but is only used for random tie-breaking; its contents are not surfaced.
- **`tie-breaking`** is mentioned in `thoughts/shared/plans/2026-04-03-idtaxa-python-rust-port.md:45,1154` only as a note that the Rust port matches R's `sample(w, 1)` behavior — no plan to preserve ties.
- **R upstream** (`reference/R_orig/IdTaxa.R:380-389`) also discards alternatives — same random-pick pattern.
- **No golden test** asserts behavior on tied identical sequences; the explore agent counted 112+ golden JSON fixtures and found none tagged or documented as tie scenarios.
- **Training-side** mention of "cross-entropy tie-breaking" in `thoughts/shared/plans/2026-04-06-batch-fraction-learning-results.md:18` is about decision-k-mer selection during training, not classification output.

### 8. Upstream/downstream data points worth noting

- **Sequence dereplication happens at `src/classify.rs:51`** via `dereplicate()`. If two reference species have identical marker sequences and the user submits one of those exact sequences, the *query* side does not get deduplicated against training references — but within a single query set, identical reads collapse to one classification pass and the same result is fanned out to each original index (`src/classify.rs:73,78`).
- **`cross_index` maps training sequences to their leaf node.** Two training sequences belonging to different species but having identical k-mers would still have different `cross_index` entries (one per species node), so both leaf nodes would appear in `unique_groups` after the groupMax pass at `src/classify.rs:320-346` — this is the data structure-level origin of the tied-group set the user is asking about.

## Code References

- `src/classify.rs:371-380` — tie detection (`winners`) and random selection
- `src/classify.rs:359-369` — `tot_hits` construction (per-group bootstrap aggregation)
- `src/classify.rs:320-346` — groupMax loop building `unique_groups` and `top_hits_idx`
- `src/classify.rs:388-411` — ancestor walk + confidence propagation (the "sibling contributions to genus" mechanism)
- `src/classify.rs:413-422` — per-rank threshold filter (`above`)
- `src/classify.rs:424-437` — `ClassificationResult` construction (both full and truncated branches)
- `src/classify.rs:451,522,525` — `ClassificationResult::unclassified()` fallbacks
- `src/types.rs:49-62` — `ClassificationResult` definition
- `src/types.rs:65-70` — `TsvRow` definition (unused)
- `src/types.rs:23-46` — `TrainingSet` fields
- `src/training.rs:145-154` — `cross_index` construction (training-seq → leaf node)
- `src/training.rs:157-163` — `taxa[]` (short label) construction
- `src/training.rs:181-205` — `children`/`parents` construction
- `src/training.rs:226-240` — `sequences[node]` (transitive closure of training-seq indices)
- `src/training.rs:488` — `ranks: None` hardcoded
- `src/fasta.rs:60-105` — `write_classification_tsv` (only serialization path)
- `src/lib.rs:54-113` — PyO3 `classify` function (returns `PyResult<()>`, writes to TSV)
- `src/lib.rs:154-159` — PyO3 `#[pymodule] fn _core` registration
- `python/oxidtaxa/__init__.py:7` and `python/idtaxa/__init__.py:7` — thin Python re-exports
- `reference/R_orig/IdTaxa.R:380-389` — R reference tie-break logic (matches Rust)
- `tests/golden_json/s08b_asym_training_set.json:94749-94772` — concrete example of `taxa`/`taxonomy` contents
- `thoughts/shared/plans/2026-04-03-idtaxa-python-rust-port.md:45,1154` — port-plan mention of tie-breaking
- `thoughts/shared/plans/2026-04-06-batch-fraction-learning-results.md:18` — unrelated training-side "cross-entropy tie-breaking"

## Architecture Documentation

**Classification result data flow** (as it exists today):

```
classify_one_pass (src/classify.rs:165)
  ├─ groupMax → unique_groups, top_hits_idx                      [src/classify.rs:320-346]
  ├─ tot_hits per group (bootstrap aggregation)                  [src/classify.rs:359-369]
  ├─ winners = {j : tot_hits[j] == max_tot}                      [src/classify.rs:373]
  ├─ selected = random choice from winners                       [src/classify.rs:375-380]
  ├─ predicteds = ancestor chain of unique_groups[selected]      [src/classify.rs:388-396]
  ├─ confidences = propagate tot_hits of non-selected groups
  │                 up parents into matching ancestors            [src/classify.rs:398-411]
  ├─ above = ranks where confidences[i] >= threshold[i]          [src/classify.rs:413-422]
  └─ ClassificationResult { taxon, confidence }                  [src/classify.rs:424-437]
        │                    ^ full lineage OR truncated + "unclassified_*" placeholder
        │
        ▼ (id_taxa → classify_sequential / classify_parallel → Vec<ClassificationResult>)
        │
        ▼ (PyO3 classify: src/lib.rs:109)
write_classification_tsv (src/fasta.rs:60-105)
  ├─ strip "Root" element
  ├─ strip every "unclassified_*" element
  ├─ join remaining with ";"
  ├─ min(confidence[1..])
  └─ write "{read_id}\t{path}\t{min_conf}\n"
```

**Tie information lifetime**: tied-group identities exist as `winners` / `unique_groups` / `tot_hits` for the duration of `classify_one_pass`. None of this data is copied into `ClassificationResult`; after `classify_one_pass` returns, it is dropped.

## Historical Context (from thoughts/)

- `thoughts/shared/plans/2026-04-03-idtaxa-python-rust-port.md:45,1154` — Documents that the Rust port's tie-break logic intentionally mirrors R's `sample(w, 1)` at `reference/R_orig/IdTaxa.R:387`. Treated as rare and correct-by-mirroring.
- `thoughts/shared/plans/2026-04-06-batch-fraction-learning-results.md:18` — Mentions "cross-entropy tie-breaking" in a training-side context (decision-k-mer scoring), unrelated to classification-output ties.
- `thoughts/shared/research/2026-04-05-rust-idtaxa-parameter-space.md` — Prior parameter-space research doc in the same directory; no tie-handling content.

## Related Research

- `thoughts/shared/research/2026-04-05-rust-idtaxa-parameter-space.md` — Parameter-space research for the Rust IDTAXA port.

## Open Questions

- **Does the tied-sibling set always collapse to one genus?** The propagation walk (`src/classify.rs:402-409`) is general — it lands on *any* shared ancestor of a non-selected group. If two tied species belong to different genera but the same family, their contribution would land on the family. The specific case the user described (two species sharing a genus) is the common case but not the only one.
- **Is "identical k-mer profile" the only way ties arise?** The `tot_hits == max_tot` check is exact-float-equality, so in practice ties are driven by identical integer-like bootstrap counts. Near-ties (e.g., one contributes 100.0 and another 99.9999…) are *not* treated as ties by this code.
- **Is `ranks: None` intentional?** The struct field exists in `src/types.rs:26` and the R reference supports populating it, but `src/training.rs:488` unconditionally sets it to `None`. If rank labels were populated, identifying "genus" vs "species" rank would not require guessing by depth.
