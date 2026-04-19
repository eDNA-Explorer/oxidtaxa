---
date: 2026-04-17T09:43:49-04:00
researcher: Ryan Martin
git_commit: 0a684f5263f9ffd7d58426b0a7528a61c5cc6800
branch: main
repository: eDNA-Explorer/oxidtaxa
topic: "How the below-threshold abstention case is handled end-to-end (assembly → writer → consumers → R reference)"
tags: [research, codebase, classify, fasta, output, threshold, unclassified, abstention, novelty]
status: complete
last_updated: 2026-04-17
last_updated_by: Ryan Martin
---

# Research: Below-threshold abstention path — end-to-end handling

**Date**: 2026-04-17T09:43:49-04:00
**Researcher**: Ryan Martin
**Git Commit**: 0a684f5263f9ffd7d58426b0a7528a61c5cc6800
**Branch**: main
**Repository**: eDNA-Explorer/oxidtaxa

## Research Question

Document the complete current behavior of oxidtaxa when the classifier's per-rank confidence vector has no rank clearing `config.threshold` (or `config.rank_thresholds`). Specifically, map every piece of code that touches the result between the point where `above: Vec<usize>` is computed in `leaf_phase_score` and the point where a downstream consumer sees output — including the Rust call chain, the TSV writer, tests that encode the current behavior, and the R/DECIPHER reference implementation. This research is the context for a proposal framed by the requester (ednaexplorer.org benchmark team) as an "output bug: abstention information being silently discarded."

### Requester's input, reproduced for context

The request includes empirical data on a 4-tier holdout benchmark (`normal`, `haplotype`, `species_holdout`, `genus_holdout`) over 309 ASVs, config `t60_se0.65_md0.95_k9_rkf0.1`:

| tier            | total | empty paths | %empty |
|-----------------|-------|-------------|--------|
| normal          | 71    | 0           | 0%     |
| haplotype       | 68    | 2           | 2.9%   |
| species_holdout | 88    | 33          | 37.5%  |
| genus_holdout   | 82    | 52          | 63.4%  |

The requester's analysis points to `src/classify.rs:741-748` (the `above.is_empty()` collapse to `vec![0]`) and `src/fasta.rs:78-105` (the writer's `Root` strip + `unclassified_*` filter) as the components producing empty `taxonomic_path` rows. The requester states that "R IDTAXA's behavior is to emit the deepest confident-prefix-with-`unclassified_` marker even when nothing passes the user threshold." (This research verifies what R actually does — see §6.)

This document does not evaluate whether the current behavior is correct. It documents the code paths, tests, consumers, and reference implementation as they exist.

## Summary

The abstention output is produced by the interaction of five components:

1. **Result assembly** (`src/classify.rs:720-749`) — builds `above`, then either emits the full lineage, a partial lineage + `unclassified_` sentinel, or (when `above` is empty) collapses to `["Root", "unclassified_Root"]` with two copies of the Root-level confidence.
2. **Fallback constructor** (`src/types.rs:168-176`) — `ClassificationResult::unclassified()` returns `["Root", "unclassified_Root"]` with `confidence: [0.0, 0.0]` and empty `alternatives`. Used when `classify_one_pass` returns `None`.
3. **Two `None`-return sites** (`src/classify.rs:186-188` "too few k-mers"; `src/classify.rs:483` "no training sequences after full-length filter").
4. **TSV writer** (`src/fasta.rs:60-110`) — strips index 0 of `taxon` (the `Root` element), filters out every element starting with `"unclassified_"`, joins the remainder with `;`, and emits `min(confidence)` — or writes `"<read_id>\t\t0\t<alternatives>\n"` when nothing survives.
5. **Python binding** (`src/lib.rs:145-204`) — returns the in-memory `Vec<ClassificationResult>` to the caller *and* (when `output_path` is supplied) writes the TSV.

Three distinct populations of "unclassified" output exist in the code, distinguishable by `confidence` but not by `taxon`:

| Source | `taxon` | `confidence` | `alternatives` |
|--------|---------|--------------|----------------|
| `ClassificationResult::unclassified()` (Paths A/B) | `["Root", "unclassified_Root"]` | `[0.0, 0.0]` | `[]` |
| `above.is_empty()` collapse (Path C) | `["Root", "unclassified_Root"]` | `[confidences[0], confidences[0]]` where `confidences[0] = base_confidence + ancestor-accumulation` | possibly non-empty if `winners.len() > 1` |
| Partial-prefix truncation (Path C') | `[Root, …<passing ranks>…, unclassified_<deepest passing>]` | `[…passing confidences…, last_passing_confidence]` | possibly non-empty |

The TSV writer collapses the first two rows to the same `"<read_id>\t\t0\t..."` line whenever no non-`Root` / non-`unclassified_` element survives filtering (`src/fasta.rs:100-104`). The literal `0` in the confidence column comes from the `format!("{}\t\t0\t...", ...)` template, not from `confidences[0]`.

The R reference (`reference/r_source/IdTaxa.R:450-451, 494-495`) performs the same collapse: `if (length(w) == 0) w <- 1 # assign to Root`, yielding the same `c("Root", "unclassified_Root")` output. The collapse semantic is inherited from DECIPHER's `IdTaxa`, not introduced by the Rust port.

## Detailed Findings

### 1. Result assembly in `leaf_phase_score` (src/classify.rs)

Entry point: `leaf_phase_score` at `src/classify.rs:449-752`. The function is shared by both the greedy descent (`classify_one_pass`, `src/classify.rs:168-251`) and the beam search (`classify_one_pass_beam`, `src/classify.rs:254-444`).

**Step 1 — Build `predicteds` (lineage walk, root → leaf)** — `src/classify.rs:662-671`

```rust
let predicted_group = unique_groups[selected];
let mut predicteds: Vec<usize> = Vec::new();
let mut p = predicted_group;
loop {
    predicteds.push(p);
    if p == 0 || parents[p] == p { break; }
    p = parents[p];
}
predicteds.reverse();
```

`predicteds[0]` is always the `Root` node index; `predicteds[len-1]` is the selected leaf group.

**Step 2 — Compute `confidences` (per-rank)** — `src/classify.rs:673-686`

```rust
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

Mechanically, every non-selected group `j` contributes `tot_hits[j] / b * 100` to each shared ancestor of `unique_groups[j]` with `predicteds`. The inner walk starts at `parents[unique_groups[j]]`, not `unique_groups[j]` itself — so sibling contributions never land on the leaf. Confidences are monotonically non-increasing from root to leaf under a single flat threshold (see `thoughts/shared/research/2026-04-15-new-parameter-audit.md` §2 for how `rank_thresholds` can break this invariant).

**Step 3 — Compute `lca_cap` (cap reportable lineage at the tie LCA)** — `src/classify.rs:688-718`

When `winners.len() > 1` (multiple groups tied at `max_tot`), `lca_cap = Some(deepest_allowed)` is the shallowest `predicteds` index that is still an ancestor of every tied winner. When there are no ties, `lca_cap = None` and `alternatives = Vec::new()`. (Added by `cbc1a35 Add tied-species reporting with LCA cap and in-memory Python API`; see `thoughts/shared/research/2026-04-08-tied-species-reporting-at-truncated-rank.md`.)

**Step 4 — Compute `above` (ranks passing threshold, subject to LCA cap and per-rank thresholds)** — `src/classify.rs:720-733`

```rust
let above: Vec<usize> = confidences.iter().enumerate()
    .filter(|(i, &c)| {
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

`above` is `Vec<usize>` of indices into `predicteds` that pass threshold. Three terminal states are possible:

- `above.len() == predicteds.len()` — every rank passes (full classification branch).
- `0 < above.len() < predicteds.len()` — partial prefix passes (truncation branch).
- `above.is_empty()` — no rank passes (collapse branch).

**Step 5 — Construct `ClassificationResult`** — `src/classify.rs:735-749`

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

Key semantic details:

- `last_w` — the *last* index in `w`. In the collapse branch (`w == vec![0]`), `last_w == 0`, so `taxa[predicteds[0]]` resolves to `"Root"` and the appended sentinel is `"unclassified_Root"`. The appended confidence is a duplicate of `confidences[0]` — the accumulated Root-level confidence (usually `> 0` in this branch — see §3 for the distinction from the `unclassified()` fallback).
- In the partial-prefix branch, `w` is the `above` vector; the sentinel is `"unclassified_<deepest passing taxon>"`, and the appended confidence duplicates `confidences[last_w]` (the last passing rank's confidence).
- `predicteds` is *not* used beyond indexing — the descent path below `last_w` is not copied into the result in either the collapse or partial-prefix branches.

### 2. `classify_one_pass` early returns (None → unclassified fallback)

Two sites in `leaf_phase_score` and `classify_one_pass` return `None`, which the caller converts to `ClassificationResult::unclassified()`:

- **Too few k-mers** — `src/classify.rs:186-188` and `src/classify.rs:269-271`:
  ```rust
  if my_kmers.len() <= s {
      return None;
  }
  ```
  (Identical guard at top of greedy and beam descent.)

- **No training sequences after full-length filter** — `src/classify.rs:482-484`:
  ```rust
  if keep.is_empty() { return None; }
  ```

- **No hits from `parallel_match`** — `src/classify.rs:555`: `if hits_flat.is_empty() { return None; }`

The `None` → `unclassified()` conversion happens at:
- `src/classify.rs:763` — `let mut results = vec![ClassificationResult::unclassified(); n];` (sequential path initializer)
- `src/classify.rs:834` — `(None, None) => ClassificationResult::unclassified()` (parallel path, strand=both)
- `src/classify.rs:837` — `fwd.map(|(r, _)| r).unwrap_or_else(ClassificationResult::unclassified)` (parallel, strand!=both)

### 3. `ClassificationResult::unclassified()` — the zero-confidence fallback

`src/types.rs:168-176`:
```rust
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

This produces a `taxon` value identical to the `above.is_empty()` collapse branch, but with `confidence: [0.0, 0.0]`. In the collapse branch, `confidence` carries the accumulated `confidences[0]` value — typically non-zero because ancestor accumulation from every non-selected group raises Root-level confidence. The two sources are distinguishable by inspecting `confidence` directly, but converge in the TSV writer (§4).

### 4. TSV writer (src/fasta.rs)

`write_classification_tsv` at `src/fasta.rs:60-110`:

```rust
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

Step-by-step reduction for each of the three result shapes:

- **Full-lineage result** (`taxon = [Root, Eukaryota, Chordata, …, Species]`, `confidence = [c0, c1, …, c7]`):
  - `taxa.remove(0)` → `[Eukaryota, Chordata, …, Species]`
  - No element starts with `unclassified_`, so `filtered_taxa` matches `taxa`.
  - `path_str = "Eukaryota;Chordata;…;Species"`
  - `min_conf = min(c1, …, c7)`
  - Emits: `query_id\tEukaryota;Chordata;…;Species\t<min>\t<alts>\n`

- **Partial-prefix result** (`taxon = [Root, Phylum, Class, unclassified_Class]`, `confidence = [c0, c1, c2, c2]`):
  - `taxa.remove(0)` → `[Phylum, Class, unclassified_Class]`
  - Filter drops `unclassified_Class` → `filtered_taxa = [Phylum, Class]`.
  - `path_str = "Phylum;Class"`
  - `min_conf = min(c1, c2)`
  - Emits: `query_id\tPhylum;Class\t<min>\t<alts>\n`

- **Collapse result** (`taxon = [Root, unclassified_Root]`, `confidence = [c0, c0]`, with `c0 > 0` when Path C, `c0 == 0` when Path A/B):
  - `taxa.len() > 1` (it's 2), enters the first branch.
  - `taxa.remove(0)` → `[unclassified_Root]`
  - Filter drops `unclassified_Root` → `filtered_taxa = []`.
  - Enters `filtered_taxa.is_empty()` branch: emits `query_id\t\t0\t<alts>\n`.
  - The literal `0` in the confidence column is from the format string; `c0` (the computed Root-level confidence) is not surfaced. The `alternatives` field is still emitted if the `above.is_empty()` result had a non-empty `alternatives` vector.

- **`unclassified()` fallback** (`taxon = [Root, unclassified_Root]`, `confidence = [0.0, 0.0]`, `alternatives = []`):
  - Identical path to the collapse case — emits `query_id\t\t0\t\n`.

The TSV writer treats Path A/B and Path C identically in the output. The in-memory `ClassificationResult` retains the distinction via the `confidence` field (`[0.0, 0.0]` vs `[c0, c0]` where `c0 > 0`), but neither the TSV format nor any code path downstream of the writer preserves this distinction.

### 5. Python binding and in-memory API

`src/lib.rs:145-204` defines `classify(...)`:

```rust
fn classify(
    ...,
    output_path: Option<String>,
    ...,
) -> PyResult<Vec<crate::types::ClassificationResult>> {
    let model = crate::types::TrainingSet::load(model_path)...;
    let (names, seqs) = crate::fasta::read_fasta(query_path)...;
    let clean_seqs = crate::sequence::remove_gaps(&seqs);
    let strand_mode = parse_strand(strand)?;
    let config = crate::types::ClassifyConfig { ... };

    let results = py.allow_threads(|| {
        crate::classify::id_taxa(...)
    });

    if let Some(ref path) = output_path {
        crate::fasta::write_classification_tsv(path, &names, &results)
            .map_err(|e| PyValueError::new_err(e))?;
    }

    Ok(results)
}
```

As of the tied-species work (commit `cbc1a35`), `classify()` returns `Vec<ClassificationResult>` to Python *in addition to* optionally writing the TSV. Each `ClassificationResult` exposes `.taxon`, `.confidence`, and `.alternatives` via `#[pyo3::pyclass(get_all)]` on `src/types.rs:154`. Python callers who work with the return value (rather than re-reading the TSV) see the raw `["Root", "unclassified_Root"]` with its actual `confidence` pair. Python callers who read the TSV see only `<read_id>\t\t0\t<alts>\n` for any abstention.

The Python re-export surface is `python/oxidtaxa/__init__.py`, which re-exports `classify`, `train`, `ClassificationResult`, and the staged training API.

### 6. R / DECIPHER reference implementation

Examined files:
- `reference/r_source/IdTaxa.R` (primary)
- `reference/R_orig/IdTaxa.R` (identical legacy copy)
- `reference/c_source/*.c` (k-mer enumeration, integer matching, `vectorSum`, `groupMax`, `parallelMatch`, `removeGaps`, etc. — no threshold logic; confirmed via grep for `threshold|unclassified|classify`)

The R flow after bootstrap/descent mirrors the Rust flow one-for-one:

1. `reference/r_source/IdTaxa.R:402-408` — build `predicteds` (walk `parents[]` from selected leaf to Root).
2. `IdTaxa.R:409-421` — populate `confidences` for every ancestor (same structure as Rust `src/classify.rs:673-686`).
3. `IdTaxa.R:424` — `w <- which(confidences >= threshold)` (the R analog of `above`).
4. `IdTaxa.R:450-451` (`type="collapsed"`) and `IdTaxa.R:494-495` (`type="extended"`):
   ```r
   if (length(w) == 0)
       w <- 1 # assign to Root
   ```
5. `IdTaxa.R:457-466` / `496-502` — build output:
   ```r
   c(taxa[predicteds[w]], paste("unclassified", taxa[predicteds[w[length(w)]]], sep="_"))
   ```
   With `w == 1` (the overwrite), this evaluates to `c("Root", "unclassified_Root")`, identical to the Rust collapse.
6. `IdTaxa.R:452-453, 501-502, 508-509` — confidence vector becomes `c(confidences[w], confidences[w[length(w)]])`; when `w == 1`, this is `c(confidences[1], confidences[1])` — i.e., the Root-level confidence duplicated (same as the Rust collapse branch).
7. `IdTaxa.R:240, 251-253` — the pre-loop default for skipped sequences is `taxon = c("Root","unclassified_Root"); confidence = rep(0, 2)` — analogous to `ClassificationResult::unclassified()`.

The R `.Call` C helpers (`reference/c_source/R_init_idtaxa.c:7-17`: `enumerateSequence`, `alphabetSize`, `intMatch`, `groupMax`, `detectCores`, `vectorSum`, `parallelMatch`, `removeGaps`) do not perform any threshold comparison or `unclassified_` emission. All taxonomy decisions happen in R.

**Conclusion (R comparison)**: R/DECIPHER's `IdTaxa` produces `c("Root", "unclassified_Root")` with duplicated Root-level confidence in the same case — when every rank fails the user threshold, both R and oxidtaxa emit the same collapsed two-element vector. The requester's claim that "R IDTAXA's behavior is to emit the deepest confident-prefix-with-`unclassified_` marker even when nothing passes the user threshold" is not supported by the reference source code in this repository. Both implementations take the same branch (`w <- 1` in R, `vec![0]` in Rust) and produce the same `[Root, unclassified_Root]` output. The R output *does* retain the Root-level confidence values in the in-memory result, but the tab-delimited exports built on top of `IdTaxa` vary (see `reference/classify_idtaxa.R:104-105` which writes `taxonomic_path = ""`, `confidence = 0.0` when `taxa` is empty — same empty-string TSV form).

### 7. Tests that encode the current abstention behavior

Abstention outputs are exercised and asserted in multiple golden tests:

| Test | File:Line | Config | What it asserts |
|------|-----------|--------|-----------------|
| `test_classify_9a_standard` | `tests/test_classify.rs:146-160` | `threshold=60` | `query_006` and `query_008` golden = `["Root", "unclassified_Root"]` with confidence `[57.06, 57.06]` and `[57.93, 57.93]` respectively |
| `test_classify_9c_novel` | `tests/test_classify.rs:184-198` | `threshold=60` | All 3 random queries → `["Root", "unclassified_Root"]` with low confidences (5-9%) |
| `test_classify_9d_threshold_sweep` | `tests/test_classify.rs:203-222` | `threshold ∈ {0, 30, 50, 60, 80, 95, 100}` | For `threshold >= 80`, all 5 queries collapse to `["Root", "unclassified_Root"]` |
| `test_classify_9e_strand_bottom` | `tests/test_classify.rs:245-259` | `strand=Bottom` | All 5 queries on wrong strand → `["Root", "unclassified_Root"]` (low k-mer overlap → low confidence → below threshold) |
| `test_classify_9g_short` | `tests/test_classify.rs:300-330` | short query | Single query → `["Root", "unclassified_Root"]` with confidence `[0, 0]` (`unclassified()` fallback, not the collapse branch) |
| `test_full_pipeline_e2e` | `tests/test_integration.rs:19-123` | `threshold=60` | TSV output for `query_006` and `query_008` = `taxonomic_path: ""`, `confidence: 0` (the writer-level empty-path form) |
| `test_baseline_1k_matches_r` | `tests/test_baseline_1k.rs:31-178` | R-vs-Rust parity | Applies the same `unclassified_` filter used by the TSV writer (lines 114-124) when comparing per-ASV paths against R golden output |

Golden fixtures under `tests/golden_json/`:
- `s09a_ids_standard.json` — 2 of 15 queries (`query_006`, `query_008`) have `["Root", "unclassified_Root"]` with non-zero confidence (~57%).
- `s09c_ids_novel.json` — 3 of 3.
- `s09d_ids_thresh_80.json`, `s09d_ids_thresh_95.json`, `s09d_ids_thresh_100.json` — 5 of 5.
- `s09e_ids_strand_bottom.json` — 5 of 5, confidences 6-9%.
- `s09g_ids_short.json` — 1 of 1 with confidence `[0, 0]`.
- `s10a_e2e_tsv.json` — encodes the empty-string TSV form directly: `"taxonomic_path": ""`, `"confidence": 0` for `query_006` and `query_008`.
- `golden_classification.json` — legacy, 2 abstention rows; not referenced from active Rust tests.
- `s09h_ids_boots_1.json` — contains one abstention row; only `boots=100` is iterated by the active test, so this file is inert.

No test asserts "abstention preserves descent path information" — the `["Root", "unclassified_Root"]` collapse is the locked-in golden behavior for `threshold=60` on both Rust and the R reference generator (see `reference/generate_golden.R` and `tests/generate_golden.R:493-521`, which produce these goldens from R's `IdTaxa` output).

### 8. Downstream TSV consumers

Code that actively parses the TSV columns:

- `tests/test_integration.rs:77-116` — reads the TSV, splits on `\t`, checks header = `"read_id\ttaxonomic_path\tconfidence\talternatives"`, asserts `parts[1] == golden[i].taxonomic_path` (which is `""` for abstention rows) and `parts[3] == alternatives`. The test is the primary guard on the empty-string column format.
- `tests/test_ties.rs:140-184` (`tied_alternatives_appear_in_tsv_output`) — writes TSV, reads back, asserts header + 4 columns + `parts[3] == "Canis_latrans|Canis_lupus"` for the tied case.
- `reference/pytest/test_golden.py:60-115` — pre-`alternatives` 3-column layout; compares `parts[1]` / `parts[2]` against golden JSON.
- `reference/rust_original/tests/test_integration.rs` — legacy copy of `test_integration.rs`.
- `tests/run_golden.R:480-521` and `reference/run_golden.R:480-521` — R-side validator; reconstructs the same 3-column DataFrame from R's `IdTaxa` output (setting `our_e2e$taxonomic_path[i] <- ""` at line 513 for empty groups) and compares read-id / path identity against `s10a_e2e_tsv` golden.

Code that opens the TSV but does not column-parse:
- `benchmarks/run_real_data_idtaxa.py:149` — does `sum(1 for _ in f) - 1` to count rows. No column inspection.

Code that uses the Python return value (`Vec<ClassificationResult>`) instead of the TSV:
- `python/oxidtaxa/__init__.py` — re-exports only; no processing logic.
- `examples/eval_training.rs:158-169` (`extract_path`, `extract_confidence`) — joins `r.taxon` (after its own `Root`/`unclassified_` filtering equivalent) and slices `r.confidence[1..]`.

Shell/CLI drivers:
- `classify.py` — thin argparse wrapper that calls `classify(..., output_path=...)`; does not read the TSV back.
- `benchmarks/run_benchmark.sh:81, 101, 110` — times classify runs; does not parse TSV.

### 9. How this interacts with the newer features

Since the prior `2026-04-14-confidence-scores-for-unclassified-asvs.md` research was written (commit `ee8c3cb`), the assembly block at `classify.rs:735-749` has gained two neighbors:

- **`lca_cap` + `alternatives`** (commit `cbc1a35`, `src/classify.rs:688-718`) — when `winners.len() > 1`, `above` is capped at the LCA index. This can produce `above.is_empty()` when the tied set's LCA is Root (all siblings unrelated) — in that sub-case, the collapse is taken with a non-empty `alternatives` vector. The TSV writer preserves `alternatives` even when `taxonomic_path` is empty: output is `<read_id>\t\t0\t<alt1>|<alt2>|...\n`.
- **`rank_thresholds`** (`src/classify.rs:726-729`) — allows per-rank thresholds. Since `confidences` is monotonically non-increasing from root to leaf only when the single `threshold` is used, per-rank thresholds can produce non-contiguous `above` (documented in `thoughts/shared/research/2026-04-15-new-parameter-audit.md` §2). The `above.is_empty()` collapse is still reached when *no* rank passes its rank-specific threshold, but partial-prefix truncation semantics can diverge from the "deepest passing rank" model.

The `ClassificationResult` struct (`src/types.rs:156-166`) carries `alternatives: Vec<String>` with `#[serde(default, skip_serializing_if = "Vec::is_empty")]`, so the in-memory API and serialized model files are backward-compatible with the pre-`cbc1a35` layout.

## Code References

- `src/classify.rs:168-251` — `classify_one_pass` (greedy descent driver).
- `src/classify.rs:254-444` — `classify_one_pass_beam` (beam search variant; shares `leaf_phase_score`).
- `src/classify.rs:449-752` — `leaf_phase_score` (scoring + result assembly).
- `src/classify.rs:186-188, 269-271, 482-484, 555` — early-`None` returns routed to `ClassificationResult::unclassified()`.
- `src/classify.rs:662-671` — `predicteds` construction (Root → leaf ancestor walk).
- `src/classify.rs:673-686` — `confidences` construction with sibling accumulation.
- `src/classify.rs:688-718` — `lca_cap` and `alternatives` computation.
- `src/classify.rs:720-733` — `above` filter with `lca_cap` + `rank_thresholds`.
- `src/classify.rs:735-749` — result assembly: full, partial-prefix, and collapse branches.
- `src/classify.rs:763, 834, 837` — callers that substitute `ClassificationResult::unclassified()` for `None`.
- `src/types.rs:156-166` — `ClassificationResult` struct (includes `alternatives`).
- `src/types.rs:168-176` — `unclassified()` fallback constructor.
- `src/fasta.rs:60-110` — TSV writer (Root strip, `unclassified_` filter, empty-path fallback).
- `src/lib.rs:145-204` — Python binding; returns in-memory results + optionally writes TSV.
- `reference/r_source/IdTaxa.R:402-421, 424, 450-451, 494-495, 457-466, 497-512` — R reference for the same data flow.
- `reference/r_source/IdTaxa.R:240, 251-253` — R pre-loop default for skipped sequences.
- `reference/c_source/R_init_idtaxa.c:7-17` — registered `.Call` primitives (none perform thresholding).
- `tests/test_classify.rs:146-436` — golden tests including abstention cases.
- `tests/test_integration.rs:19-123` — full-pipeline TSV test with empty-path rows.
- `tests/test_baseline_1k.rs:114-124` — path-extraction logic that mirrors the TSV filter for R-parity checks.
- `tests/golden_json/s09a_ids_standard.json:22-25, 30-33` — example abstention entries with non-zero confidence.
- `tests/golden_json/s10a_e2e_tsv.json:27-31, 37-41` — TSV-level empty-path goldens.
- `benchmarks/run_real_data_idtaxa.py:149` — line-counting TSV consumer.

## Architecture Documentation

**End-to-end data path (current state):**

```
classify_one_pass / classify_one_pass_beam (src/classify.rs:168, 254)
  └─ leaf_phase_score (src/classify.rs:449)
       ├─ [Early return → None] too-few-kmers / empty-keep / empty-hits
       ├─ predicteds        ← ancestor walk             [662-671]
       ├─ confidences       ← base + sibling propagate  [673-686]
       ├─ lca_cap/alts      ← tied winners LCA          [688-718]
       ├─ above             ← rank filter + cap         [720-733]
       └─ ClassificationResult
             ├─ full branch      (above.len == predicteds.len) [735-740]
             ├─ partial-prefix   (0 < above.len < predicteds.len) [741-748, w=above]
             └─ collapse         (above.is_empty())             [741-748, w=vec![0]]

id_taxa orchestrator (src/classify.rs:33-83)
  ├─ sequential path → None → ClassificationResult::unclassified()  [763]
  └─ parallel path   → None → ClassificationResult::unclassified()  [834, 837]

→ Vec<ClassificationResult>

(Python) oxidtaxa._core.classify (src/lib.rs:145-204)
  ├─ return results to Python caller (in-memory; carries full structure)
  └─ write_classification_tsv(path, names, results) if output_path set

write_classification_tsv (src/fasta.rs:60-110)
  └─ per row:
       ├─ taxa.remove(0)            (strip Root)
       ├─ filter out unclassified_* (drop sentinel)
       └─ if non-empty → "{id}\t{path}\t{min_conf}\t{alts}\n"
          else        → "{id}\t\t0\t{alts}\n"   ← abstention row form
```

**The two sources of `["Root", "unclassified_Root"]`:**

```
above.is_empty() collapse            ClassificationResult::unclassified()
  taxon:   [Root, unclassified_Root]   taxon:   [Root, unclassified_Root]
  conf:    [c0, c0]   (c0 ≥ 0)         conf:    [0.0, 0.0]
  alts:    [] or Vec<String>           alts:    []

  ── both flow through fasta.rs ──
  TSV row: "<id>\t\t0\t<alts?>\n"      TSV row: "<id>\t\t0\t\n"
```

The TSV format does not distinguish the two origins. The Python in-memory return preserves the `confidence` distinction.

## Historical Context (from thoughts/)

- **`thoughts/shared/research/2026-04-14-confidence-scores-for-unclassified-asvs.md`** — Prior research on exactly this topic. Identifies the three unclassified paths (A: too-few k-mers, B: no training match, C: below-threshold) and the per-rank confidence vector that exists at `classify.rs:674` before truncation. Recommends `threshold=0` as a no-code-change workaround for analysis. Predates the `alternatives` / `lca_cap` additions.
- **`thoughts/shared/research/2026-04-08-tied-species-reporting-at-truncated-rank.md`** — Context for how the tied-species mechanism was added. Documents the pre-`cbc1a35` state where `ClassificationResult` had only `taxon` and `confidence`. The `alternatives` field and `lca_cap` were motivated by distinguishing "tool abstained" from "tool found multiple indistinguishable candidates."
- **`thoughts/shared/research/2026-04-15-new-parameter-audit.md`** — §2 documents the `rank_thresholds` non-contiguous-`above` case (confidences `[100, 75, 72]` with thresholds `[90, 80, 70]` → `above = [0, 2]` → taxonomic lineage that skips a rank). §6 covers `correlation_aware_features`, unrelated but adjacent.
- **`thoughts/shared/research/2026-04-13-r-replication-status.md`** — Tracks Rust/R parity. Not specifically about abstention but covers the broader R reference.
- **`thoughts/shared/plans/2026-04-03-idtaxa-python-rust-port.md:45, 1154`** — Original port plan noting that tie-break and `unclassified_` semantics are inherited from DECIPHER's `IdTaxa` by design.

## Related Research

- `thoughts/shared/research/2026-04-14-confidence-scores-for-unclassified-asvs.md` — most closely related; enumerates what data is available at the truncation point.
- `thoughts/shared/research/2026-04-08-tied-species-reporting-at-truncated-rank.md` — history of the `alternatives` / `lca_cap` mechanism that shares the same assembly block.
- `thoughts/shared/research/2026-04-15-new-parameter-audit.md` — documents `rank_thresholds` and other new parameters that interact with the `above` filter.

## Open Questions

- The requester's empirical `%empty` numbers (63.4% for `genus_holdout`) measure the TSV column-1-empty rate. The split between Path A/B (`unclassified()` fallback with `confidence=[0,0]`) and Path C (`above.is_empty()` collapse with `confidence=[c0,c0]`) is not separately reported by the requester, and the in-memory `confidence` field that distinguishes them is discarded by the writer. Whether this split varies across tiers is not visible from the current TSV format.
- No test currently asserts on the in-memory `confidence[0]` value in the collapse branch (golden fixtures for `s09a_ids_standard.json` record values like `57.06` for `query_006`, but no test reads *only* these abstention rows to check the Path C vs Path A distinction). Changes to the collapse branch would move these golden numbers.
- The requester's report references a config `t60_se0.65_md0.95_k9_rkf0.1` — `se=0.65` maps to `sample_exponent=0.65` (default is 0.47), `md=0.95` to `min_descend=0.95` (default 0.98), `k9` to `k=9` (not the auto-k-max default), `rkf0.1` to `record_kmers_fraction=0.10` (default). This configuration is outside what the current golden test suite exercises directly (goldens use `ClassifyConfig::default()` or the `threshold`/`min_descend` sweep grids in `test_classify_9d` / `test_classify_9i`).
