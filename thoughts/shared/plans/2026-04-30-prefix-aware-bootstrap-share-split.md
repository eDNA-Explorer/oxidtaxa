---
date: 2026-04-30
researcher: Ryan Martin
git_commit: f36cd2eb90a7f5d9fa072f03d422b83375773f17
branch: main
repository: eDNA-Explorer/oxidtaxa
topic: "Prefix-aware bootstrap share-split: stop ancestor-only training entries from halving descendant tot_hits"
tags: [plan, classify, share-split, suppress_ancestor_only_groups, bootstrap, tot_hits, leaf_phase_score]
status: ready
last_updated: 2026-04-30
last_updated_by: Ryan Martin
---

# Prefix-aware bootstrap share-split — implementation plan

## Prerequisite

**This plan assumes `2026-04-30-cross-rank-accumulator-universal-fix.md` has already landed.** That fix is what makes the conservation invariant referenced throughout this plan actually hold at every rank. Specifically:

- The cross-rank accumulator and LCA-cap walks in `src/classify.rs` start each non-selected group's walk at `unique_groups[j]` (not `parents[unique_groups[j]]`). Without this, ancestor `tot_hits` is misrouted one rank up and the conservation claims below are not testable.
- The existing test `tests/test_margin_aware.rs::test_suppress_preserves_higher_rank_confidence` has been modified by the prerequisite plan to remove its `if i == len - 1 { continue; }` rank-skip workaround — the test now asserts equality at *every* rank.

When this plan refers to that test passing "with no source changes," it means *no further* source changes beyond the prerequisite plan's edits. Do not re-introduce the rank-skip.

## Overview

The leaf-phase bootstrap accumulator at `src/classify.rs:764-794` splits per-replicate credit equally across all groups tied at the per-replicate `max_val`. When a database contains an ancestor-only training entry (e.g. `"Oncorhynchus sp."` after canonical NA-trim) AND a species-resolved descendant under the same ancestor, the two groups tie in every replicate where their training sequences match the query equivalently. Each tied group receives `share = 1/n_tied = 0.5` of the credit, so descendant `tot_hits` is depressed by ~50 % relative to a curated database that lacks the ancestor entry.

The recently-added `suppress_ancestor_only_groups` flag (commit `f36cd2e`) filters ancestor-prefix winners *after* `tot_hits` is computed (`src/classify.rs:825-859`). It blocks LCA-cap collapse but cannot recover credit already absorbed into the ancestor's `tot_hits` during the bootstrap. This plan moves the prefix-suppression upstream into the per-replicate share-split so the descendant gets full credit at the moment the credit is being apportioned.

## Critical Research

### The pipeline (where the damage happens)

```
                       ┌───────────────────────────────────────────┐
                       │      bootstrap loop  b = 100 reps         │
                       │                                           │
   per-replicate hits ─┼─► find max_val across top_hits_idx        │
   (hits_flat)         │                                           │
                       │   n_tied = #{j : hit[j,rep] == max_val}   │
                       │   share  = 1 / n_tied      ◄── DAMAGE     │
                       │                                           │
                       │   for each tied j:                        │
                       │     tot_hits[j] += hit/davg * share       │
                       │                                           │
                       └─────────────────── ↓ ─────────────────────┘
                                          tot_hits
                                            ↓
                       ┌───────────────────────────────────────────┐
                       │   POST-BOOTSTRAP                          │
                       │                                           │
                       │   winners = argmax(tot_hits) (with        │
                       │             optional tie_margin band)     │
                       │                                           │
                       │   if suppress_ancestor_only_groups:       │
                       │     drop winners whose path is a strict   │
                       │     prefix of another winner's path       │
                       │   ─────────── existing flag fires ───────│
                       │                                           │
                       │   selected ← random from filtered winners │
                       └───────────────────────────────────────────┘
                                            ↓
                       ┌───────────────────────────────────────────┐
                       │   CROSS-RANK ACCUMULATION                 │
                       │   confidences[i] built from tot_hits[],   │
                       │   not winners                             │
                       └───────────────────────────────────────────┘
```

The cross-rank accumulator iterates `tot_hits[]` directly (`src/classify.rs:884-895`), so any half-credit the ancestor absorbed is permanently routed into ancestor-rank confidences and the descendant's species-rank confidence is permanently halved. The post-bootstrap winner-stage filter cannot reach back into `tot_hits` to redistribute the credit.

### Why tied-share splitting exists

Source comment at `src/classify.rs:771-783`: R's IDTAXA uses `max.col(..., ties.method = "random")` which randomly picks one tied column per replicate. In expectation each tied column receives credit proportional to its share of ties; deterministically splitting `1/n_tied` per replicate matches that expected value and surfaces bit-identical training sequences as ties downstream (so the existing winner-stage filter can act on them). This semantic is correct for **genuine sibling ties** (two species the database genuinely cannot distinguish). It is wrong for **ancestor-prefix ties**, which are a database-curation artifact rather than competing evidence.

### Quantitative trace (bytes-identical ancestor + descendant)

Configuration: 100 replicates, every replicate produces `hits_flat[ancestor] = hits_flat[descendant] = 10`, `davg = 10` (so `hit/davg = 1.0` per replicate per group).

```
                          tot_hits         confidence       confidence
                          (after b=100)    species rank     genus rank
   ───────────────────────────────────────────────────────────────────
   CURRENT (flag = off)
     ancestor              50               n/a              50  + 50  = 100
     descendant            50               50               climbs to genus
   ───────────────────────────────────────────────────────────────────
   CURRENT (flag = on)     same tot_hits as off
     ancestor              50               n/a              50 + 50  = 100
     descendant            50               50               climbs to genus
   ───────────────────────────────────────────────────────────────────
   PROPOSED (flag = on, prefix-aware share-split)
     ancestor               0               n/a              0 + 100  = 100
     descendant           100              100               climbs to genus
   ───────────────────────────────────────────────────────────────────
```

Species-rank confidence doubles. Genus-rank confidence is **conserved** because the cross-rank accumulator routes the descendant's now-full `tot_hits` upward through `parents[]` and lands the same total on the genus rank. Ranks above genus are untouched.

This conservation property is critical: the existing `test_suppress_preserves_higher_rank_confidence` regression test at `tests/test_margin_aware.rs:808-880` asserts that confidences for ranks at-or-above the dropped node match between flag=off and flag=on (within `1e-3`). The proposed fix preserves that assertion because it only redistributes credit at-and-below the deepest descendant rank.

### Decision: filter the *tied set*, not the input vector

Two equivalent-looking knobs exist for stopping ancestor pollution at the share-split. The plan picks (B):

```
   (A)  pre-filter top_hits_idx to drop ancestor-prefix groups
        before the bootstrap loop runs at all.
        ─────────────────────────────────────────────
        downside: ancestor's outright wins (replicates where
        ancestor strictly beats descendant) are also dropped,
        losing genuine genus-rank evidence the ancestor
        contributes from its own training k-mer pattern.

   (B)  filter only the *tied set* per replicate. ◄── chosen
        ─────────────────────────────────────────────
        ancestor's outright-win replicates (n_tied=1, ancestor
        is sole max) are unaffected. Ancestor still contributes
        when it has independent evidence; only the parasitic
        tied-replicate freeloading is suppressed.
```

(B) is strictly more conservative than (A) and is the surgical match for the concern.

### Decision tree the per-replicate filter implements

```
   for rep in 0..b:
      compute max_val across top_hits_idx for this rep
      tied = { j : hits_flat[ti(j), rep] == max_val }

      if config.suppress_ancestor_only_groups AND |tied| > 1:
          # for each j in tied, drop j if some k in tied has
          # path[j] as a strict prefix of path[k]
          # (i.e. j is an ancestor of someone else also tied)
          tied' = { j ∈ tied : ∄ k ∈ tied with path[k] starts_with path[j] }
          if tied' is non-empty: tied ← tied'

      n_tied = |tied|
      share  = 1 / n_tied
      for each j in tied:
          tot_hits[j] += hits_flat[ti(j), rep] / davg * share
```

The defensive `if tied' is non-empty` guard is structural: the deepest descendant in any prefix chain always survives (no one in the tied set is its descendant), so `tied'` is provably non-empty. Same defensive shape as the existing winner-stage filter at `src/classify.rs:847-853`.

### Edge-case matrix

| tied set in this replicate | prefix relations | filter action | result |
|---|---|---|---|
| `{ancestor}` (sole max) | none (singleton) | filter not entered (`|tied|=1`) | ancestor gets `1.0` — unchanged |
| `{descendant}` (sole max) | none | not entered | descendant gets `1.0` — unchanged |
| `{ancestor, descendant}` byte-identical | ancestor ⊂ descendant | drop ancestor | descendant gets `1.0` (was `0.5`) |
| `{C_lupus, C_latrans}` true sister tie | none | no drops | each `0.5` — unchanged |
| `{ancestor, C_lupus, C_latrans}` | ancestor ⊂ both species | drop ancestor | each species `0.5` |
| `{Family, Genus, Species}` chain | Family ⊂ Genus ⊂ Species | drop Family + Genus | Species gets `1.0` |
| `{ancestor_A, descendant_B}` (no relation) | none | no drops | each `0.5` — unchanged |
| `{ancestor, sibling_unrelated}` | ancestor not prefix of sibling, sibling not prefix of ancestor | no drops | each `0.5` |

The genuine-sibling-tie semantics (the design intent of share-splitting per the source comment) are preserved across every row that doesn't involve a prefix relation.

### Single dispatch point

Both descent strategies converge into the same scoring function:

```
   classify_one_pass        (greedy descent, src/classify.rs:168)
        │
        └──► leaf_phase_score (src/classify.rs:576) ◄─┐
                                                       │
   classify_one_pass_beam   (beam search,             │
   src/classify.rs:313)      src/classify.rs:558) ────┘
```

Both code paths converge into `leaf_phase_score`, which contains the bootstrap accumulator. A single edit at `src/classify.rs:764-794` covers both descent strategies. No duplicate wiring needed.

### Existing flag wiring

The `suppress_ancestor_only_groups` flag is already plumbed end-to-end (commit `f36cd2e`):

- `ClassifyConfig` field declaration: `src/types.rs:333` (default `false` at `:351`)
- pyo3 binding signature: `src/lib.rs:148`
- pyo3 binding parameter: `src/lib.rs:170`
- struct construction: `src/lib.rs:194`
- read site (existing winner-stage filter): `src/classify.rs:825`

The new share-split-stage filter reads the same flag from the same struct. No new public-API surface.

### Test coverage already present

The four existing tests at `tests/test_margin_aware.rs:472-880` cover the post-bootstrap behavior:

1. `test_suppress_ancestor_only_groups_default_off` (`:615`) — flag default is `false`.
2. `test_suppress_ancestor_only_groups_drops_ancestor_when_tied` (`:622`) — flag-on drops ancestor from `winners` so emission reaches species rank. **Important note:** this test sets `threshold = 40.0` *because* of the share-split halving. The test's own comment at `:627-634` documents the bug: *"Per-replicate tied-share splitting at classify.rs:725-734 depresses each tied group's tot_hits to ~half of the un-tied value... we explicitly lower threshold here to test the emission-depth question independently."* After the proposed fix, the test should pass at `threshold = 60.0` (the default) and the workaround can be removed.
3. `test_suppress_no_op_when_no_descendant_in_keep` (`:698`) — when the ancestor has no descendants in `keep`, flag-on is a no-op. The new share-split filter must preserve this.
4. `test_suppress_multi_descendant_still_caps_at_genus` (`:754`) — when the ancestor ties with multiple sibling descendants, the LCA-cap still fires at genus. The new filter must preserve this (sibling species still tie with each other after ancestor is dropped from tied set).
5. `test_suppress_preserves_higher_rank_confidence` (`:808`) — confidence at every rank in `predicteds` (including the dropped ancestor's own rank, after the prerequisite cross-rank-accumulator fix removes the `i == len - 1` skip) is unchanged between flag=off and flag=on. This is the conservation invariant the proposed fix MUST honor.

## Current State Analysis

**Where:** `leaf_phase_score` in `src/classify.rs:576-988`, specifically the bootstrap loop at `src/classify.rs:764-794`.

**What's there now:**

- `top_hits_idx[j]` and `unique_groups[j]` are co-indexed: `unique_groups[j]` is the training-tree group node for the `j`-th top-hit training sequence (`src/classify.rs:730-748`).
- `ts.taxonomy[node]` is the full taxonomic path for `node`, with a trailing `;` invariant (asserted at `src/classify.rs:830-835`).
- The bootstrap loop computes `max_val` per replicate, counts `n_tied`, computes `share = 1/n_tied`, and credits each tied group.
- The post-bootstrap filter at `src/classify.rs:825-859` performs the prefix-suppression on `winners` (after `tot_hits` is computed).

**What's missing:** No equivalent filter at the share-split stage. The prefix-relation logic in the post-bootstrap filter inspects `winner_paths` per call and recomputes prefix relations each time the filter fires; for a per-replicate filter the same work would be done `b` times per query (wasteful). The fix needs a one-pass pre-computation of prefix relations indexed by `top_hits_idx` position, reused across replicates.

**Constraints:**

- Default behavior (`suppress_ancestor_only_groups = false`) must be byte-identical to the current behavior. Gate the new filter on the same flag.
- The per-replicate filter must be O(1)-ish in `b` (the bootstrap count). Per-replicate `O(|tied|^2)` is acceptable since `|tied|` is typically 1-3.
- The `ts.taxonomy[*]` trailing-`;` invariant must be preserved; the new filter relies on it for correct `starts_with` semantics.
- Both descent paths (`classify_one_pass` and `classify_one_pass_beam`) must benefit; they share `leaf_phase_score` so this is automatic.

## Desired End State

When `suppress_ancestor_only_groups = true`:

- For every replicate where the per-replicate tied set contains both an ancestor (any group whose taxonomy path is a strict prefix of another tied group's path) and that descendant, the ancestor is dropped from the tied set BEFORE `share` is computed.
- `tot_hits[descendant]` accumulates full `hit/davg` credit instead of half.
- `tot_hits[ancestor]` accumulates only its outright-win credit (replicates where it strictly beat all descendants).
- Cross-rank confidence at every rank in `predicteds` (including the ancestor's own rank, post the prerequisite cross-rank-accumulator fix) is conserved within float tolerance compared to flag-off behavior — verified by the regression test at `tests/test_margin_aware.rs:808` in its post-prerequisite form (rank-skip already removed).
- Species-rank confidence rises from ~50 % to ~100 % in the byte-identical ancestor+descendant scenario.

When `suppress_ancestor_only_groups = false` (default):

- Behavior is byte-identical to current behavior. The new filter does not fire.

### Key Discoveries

- Bootstrap accumulator: `src/classify.rs:764-794` (the share-split lives here).
- `unique_groups` and `top_hits_idx` co-indexing: `src/classify.rs:730-748`.
- `ts.taxonomy[node]` trailing-`;` invariant: asserted at `src/classify.rs:830-835`.
- Existing post-bootstrap filter (winner-stage): `src/classify.rs:825-859`.
- Cross-rank accumulator (uses `tot_hits[]`, not `winners`): `src/classify.rs:884-895`.
- Single dispatch into `leaf_phase_score`: `src/classify.rs:305` (greedy) and `src/classify.rs:558` (beam).
- Flag wiring: `src/types.rs:333`, `src/lib.rs:148/170/194`.

## What We're NOT Doing

- **Not** introducing a new flag. Reuse `suppress_ancestor_only_groups`.
- **Not** removing the existing post-bootstrap winner-stage filter. It remains belt-and-suspenders for cases where prefix-related groups happen to land at the same `tot_hits` despite never sharing a per-replicate max (e.g. wins from disjoint replicate sets that happen to sum equal).
- **Not** changing the share-split semantics for genuine sibling ties (no prefix relation → no filter action).
- **Not** modifying `min_descend`, `tie_margin`, `confidence_uses_descent_margin`, or any other algorithmic-improvement flag.
- **Not** modifying training-time logic. This is a classify-time-only change.
- **Not** touching the parallel/sequential dispatch (`classify_sequential`, `classify_parallel`); they call `classify_one_pass*` which calls `leaf_phase_score` — single fix point upstream covers everything.
- **Not** attempting to recover ancestor evidence for ranks below the ancestor's depth (those don't exist — ancestor's depth is its taxonomic level; evidence above it is conserved via descendant climb).

## Implementation Approach

**Single edit site:** `src/classify.rs`, inside `leaf_phase_score`, between the construction of `unique_groups` (`:730-748`) and the bootstrap loop (`:764-794`).

**Two artifacts added:**

1. **Pre-computed table** `ancestor_drops_when_tied: Vec<Vec<usize>>` (or equivalent shape), built once per `leaf_phase_score` call BEFORE the bootstrap loop. Entry `[j]` is the list of `top_hits_idx`-positions whose paths are STRICT DESCENDANTS of `j`'s path. Built only when the flag is on; allocated empty (or skipped) otherwise.

2. **Per-replicate filter pass** inside the bootstrap loop, between `max_val` computation and `share` computation. Identifies the tied set, drops any `j` for which any of its descendants (per the pre-computed table) is also in the tied set, recomputes `share` from the surviving tied set.

**Allocation discipline:** Reuse a `is_tied: Vec<bool>` scratch buffer across replicates (allocated once before the loop, length `n_top`, reset to `false` each replicate). Avoid per-replicate `Vec` allocations.

## Phase 1: Implementation

### Overview

Add the pre-computed prefix-relation table and the per-replicate filter to `leaf_phase_score`. Default behavior is unchanged when the flag is off.

### Changes Required

#### 1. Pre-compute ancestor → descendant table

**File:** `src/classify.rs`

**Location:** Insert immediately after the `unique_groups` / `top_hits_idx` construction at `:730-748`, before the bootstrap loop at `:764`.

**Logic:** For each `j` in `0..unique_groups.len()`, find every `k != j` such that `taxonomy[unique_groups[k]].starts_with(taxonomy[unique_groups[j]])`. The result is "if `j` AND any of its descendants are tied in the same replicate, drop `j`."

Allocation: `Vec<Vec<u32>>` (the inner indices are `< n_top`, which is small; `u32` is plenty). When the flag is off, allocate `Vec::new()` to skip the work.

Cost: `O(n_top^2)` string-prefix comparisons, once per query. `n_top` is the number of training-tree groups represented in the leaf-phase `keep` set, typically a handful.

```rust
// Sketch (supporting evidence, not literal patch):
let descendants_of: Vec<Vec<u32>> = if config.suppress_ancestor_only_groups {
    let group_paths: Vec<&str> = unique_groups
        .iter()
        .map(|&g| ts.taxonomy[g].as_str())
        .collect();
    debug_assert!(
        group_paths.iter().all(|p| p.ends_with(';')),
        "ts.taxonomy invariant violated"
    );
    (0..unique_groups.len())
        .map(|j| {
            let j_path = group_paths[j];
            (0..unique_groups.len())
                .filter(|&k| k != j && group_paths[k].starts_with(j_path))
                .map(|k| k as u32)
                .collect()
        })
        .collect()
} else {
    Vec::new()
};
```

#### 2. Add per-replicate filter inside the bootstrap loop

**File:** `src/classify.rs`

**Location:** Inside the existing `for rep in 0..b` loop at `:764-794`, between `max_val` computation and `share` computation.

**Logic:**

1. Build `is_tied[j] = (hits_flat[ti*b+rep] == max_val)` for each `j` in `top_hits_idx` (replaces the existing `n_tied` count loop — which reads the same condition).
2. If flag is on AND `n_tied > 1`: for each tied `j`, scan `descendants_of[j]`; if any descendant `k` has `is_tied[k] = true`, mark `j` for suppression. After the scan, flip suppressed entries' `is_tied[j] = false` and decrement `n_tied`.
3. Defensive guard: if all tied entries got suppressed (impossible in practice — deepest descendant is always retained — but defensive), skip the suppression for this replicate (fall through to the unfiltered tied set).
4. Compute `share = 1 / n_tied` from the surviving count and credit each surviving tied group.

Reuse a `Vec<bool>` scratch buffer allocated outside the bootstrap loop (length `n_top`, `fill(false)` per iteration).

```rust
// Sketch (supporting evidence, not literal patch):
let mut is_tied = vec![false; n_top];  // allocated once, before `for rep`
for rep in 0..b {
    // ... compute max_val ...
    if davg == 0.0 { continue; }
    is_tied.iter_mut().for_each(|b| *b = false);
    let mut n_tied = 0usize;
    for (j, &ti) in top_hits_idx.iter().enumerate() {
        if hits_flat[ti * b + rep] == max_val {
            is_tied[j] = true;
            n_tied += 1;
        }
    }
    if config.suppress_ancestor_only_groups && n_tied > 1 {
        let mut to_drop_count = 0usize;
        // Mark suppressions in a separate pass so we don't read
        // mutated state mid-scan.
        let mut drop_mask = vec![false; n_top];  // could be a bitset
        for j in 0..n_top {
            if !is_tied[j] { continue; }
            for &k in &descendants_of[j] {
                if is_tied[k as usize] {
                    drop_mask[j] = true;
                    to_drop_count += 1;
                    break;
                }
            }
        }
        // Defensive: only apply if at least one survives.
        if to_drop_count < n_tied {
            for j in 0..n_top {
                if drop_mask[j] {
                    is_tied[j] = false;
                    n_tied -= 1;
                }
            }
        }
    }
    if n_tied == 0 { continue; }  // structural impossibility, but safe
    let share = 1.0 / n_tied as f64;
    for (j, &ti) in top_hits_idx.iter().enumerate() {
        if is_tied[j] {
            tot_hits[j] += hits_flat[ti * b + rep] / davg * share;
        }
    }
}
```

The `drop_mask` allocation in the sketch can be hoisted to a single allocation outside the bootstrap loop (mirroring `is_tied`); micro-optimize after correctness is in place.

#### 3. Update the source comment at the bootstrap loop

**File:** `src/classify.rs`

**Location:** `:771-783` (the existing comment block explaining tied-share splitting).

**Change:** Add a paragraph documenting that when `suppress_ancestor_only_groups` is on, ancestor-prefix groups are dropped from the tied set per-replicate before `share` is computed. Reference the post-bootstrap winner-stage filter at `:825` as the complementary guard for cases where prefix-related groups land at equal `tot_hits` from disjoint-replicate wins.

### Success Criteria

#### Automated Verification

- [ ] `cargo build` succeeds.
- [ ] `cargo test` passes the entire test suite (no regressions in `test_algorithmic_improvements`, `test_training`, `test_margin_aware`, or any other suite).
- [ ] `cargo clippy --all-targets -- -D warnings` is clean.
- [ ] Existing tests `test_suppress_ancestor_only_groups_default_off`, `test_suppress_no_op_when_no_descendant_in_keep`, and `test_suppress_multi_descendant_still_caps_at_genus` continue to pass with no source changes.
- [ ] `test_suppress_preserves_higher_rank_confidence` continues to pass in its post-prerequisite form (no rank-skip; asserts equality at every rank including the ancestor's own). Do not modify or re-add the skip.
- [ ] `test_suppress_leaf_only_credited_by_base` (added by the prerequisite plan) continues to pass.

#### Manual Verification

- [ ] When the flag is off, classification of a representative query produces byte-identical output to the previous commit (sanity check via a quick ad-hoc pass on any preset community).
- [ ] When the flag is on with a byte-identical ancestor+descendant fixture, descendant species-rank confidence rises into the 95–100 range (was ~50 with the old behavior).

---

## Phase 2: Tests

### Overview

Add three new tests to `tests/test_margin_aware.rs`. Update one existing test to remove the `threshold = 40.0` workaround.

### Changes Required

#### 1. New test: share-split now credits descendant in full

**File:** `tests/test_margin_aware.rs`

**New test name:** `test_suppress_share_split_credits_descendant_fully`

**Setup:** Reuse `build_ancestor_descendant_fixture()` (byte-identical Canis-genus + Canis_lupus). Run two configs at default `threshold = 60`:

- Config A: `suppress_ancestor_only_groups = false`. Expect species-rank confidence ≈ 50 (or below threshold, so emission truncated at Canidae).
- Config B: `suppress_ancestor_only_groups = true`. Expect species-rank confidence ≈ 100 and emission reaches Canis_lupus.

**Assertions:**

- Config B's emission contains `"Canis_lupus"` at the species rank.
- Config B's species-rank confidence is `>= 60` (passes default threshold).
- Config B's species-rank confidence is **strictly greater** than Config A's species-rank confidence (at the matching rank position) by `>= 30` percentage points (i.e. confirms the doubling behavior, not just a small bump).

#### 2. New test: ancestor outright wins still credited

**File:** `tests/test_margin_aware.rs`

**New test name:** `test_suppress_ancestor_outright_wins_unaffected`

**Setup:** Construct a fixture where the ancestor's training sequence has UNIQUE k-mers that the descendant's training sequence does NOT (so ancestor strictly beats descendant in some replicates) plus shared k-mers (so they tie in others). The simplest construction: ancestor is byte-identical to descendant in the bulk of the sequence but has a unique tail run. For some queries the unique tail runs will dominate per-replicate sampling, putting ancestor strictly above descendant.

**Assertions:**

- With flag-on and a query that lands roughly halfway, the ancestor's `tot_hits` (approximated by checking that genus-rank confidence is non-zero AND the ancestor is reachable as an alternative when the leaf phase ties) is positive — i.e. ancestor's outright wins were not erased by the new filter.

If approximating `tot_hits` from public output is too indirect, add a debug-only `pub` method to expose `tot_hits` to tests, OR add a low-noise eprintln-based diagnostic gated on an env var (consistent with the existing `OXIDTAXA_DEBUG_IDF_DESCENT` pattern at `src/classify.rs:227`). Prefer the debug-method route — it is more disciplined.

#### 3. New test: chain of ancestors collapses to deepest descendant

**File:** `tests/test_margin_aware.rs`

**New test name:** `test_suppress_share_split_chain_collapses`

**Setup:** Build a fixture with three byte-identical training sequences in a strict ancestor chain:

- `"Root; Mammalia; Carnivora; Canidae; Canis; Canis_lupus"`
- `"Root; Mammalia; Carnivora; Canidae; Canis"` (genus-only)
- `"Root; Mammalia; Carnivora; Canidae"` (family-only)

Plus a couple of cross-genus controls (Vulpes, Felis) so the tree has structure to descend.

**Assertions:**

- With flag-on, species-rank confidence ≈ full credit (~100), confirming both ancestors were dropped from each tied replicate.
- With flag-off, species-rank confidence ≈ 33 (third of credit per tied rep), confirming the bug's three-way share-split of the same evidence.

#### 4. Update existing test to use default threshold

**File:** `tests/test_margin_aware.rs`

**Location:** `test_suppress_ancestor_only_groups_drops_ancestor_when_tied` at `:622-696`.

**Changes:**

- Remove `threshold: 40.0,` from both `cfg_off` and `cfg_on`.
- Replace with `..Default::default()` (which gives `threshold = 60.0`).
- Update the test's prologue comment at `:627-634` to reflect the fix: replace the description of the share-split bug with a note that the default-threshold (60) test is now the regression for the upstream fix.

The test's assertions about emission depth (alternatives empty, taxon contains `Canis_lupus`) remain valid; they just now hold at the default threshold without the workaround.

### Success Criteria

#### Automated Verification

- [ ] `cargo test test_suppress_share_split_credits_descendant_fully` passes.
- [ ] `cargo test test_suppress_ancestor_outright_wins_unaffected` passes.
- [ ] `cargo test test_suppress_share_split_chain_collapses` passes.
- [ ] `cargo test test_suppress_ancestor_only_groups_drops_ancestor_when_tied` passes (now at threshold = 60).
- [ ] All other `test_margin_aware` tests still pass unmodified.

#### Manual Verification

- [ ] On the `vert12s` benchmark community at production-realistic settings, observed species-rank confidence on Canis_lupus-style genus-shared queries rises from the empirically-measured ~50 to ~95+.

---

## Phase 3: Documentation

### Overview

Update the README parameter table and the in-source doc comment for the flag to describe the new dual-stage behavior.

### Changes Required

#### 1. README parameter table

**File:** `README.md`

**Location:** The classify-side parameter table that lists `suppress_ancestor_only_groups`.

**Change:** Update the description column to note that the flag now also drops ancestor-prefix groups from the per-replicate tied set during the bootstrap, in addition to the existing post-bootstrap winner-stage filter. One short sentence is enough; the source comments carry the full rationale.

#### 2. In-source field doc

**File:** `src/types.rs`

**Location:** The `suppress_ancestor_only_groups` field doc-comment at `:329-333`.

**Change:** Append a line describing the share-split-stage behavior. Keep it terse — link to `src/classify.rs:764` and `:825` for the two read sites.

#### 3. CHANGELOG / release note

**File:** Release notes for the next version bump.

**Change:** One bullet under Bug Fixes: "When `suppress_ancestor_only_groups = true`, ancestor-only training entries no longer halve descendant species `tot_hits` via per-replicate tied-share splitting. The flag now applies prefix-suppression both at the share-split (new) and at the post-bootstrap winner stage (pre-existing)."

### Success Criteria

#### Automated Verification

- [ ] No automated checks at this phase (documentation-only).

#### Manual Verification

- [ ] README accurately describes the dual-stage behavior.
- [ ] In-source doc references both read sites.

---

## Testing Strategy

### Unit Tests

Covered in Phase 2:

- Default-off invariant (existing `test_suppress_ancestor_only_groups_default_off`, no change).
- Single-descendant share recovery (new `test_suppress_share_split_credits_descendant_fully`).
- Ancestor outright wins unaffected (new `test_suppress_ancestor_outright_wins_unaffected`).
- Multi-rank ancestor chain (new `test_suppress_share_split_chain_collapses`).
- Multi-descendant LCA-cap still fires (existing `test_suppress_multi_descendant_still_caps_at_genus`, no change).
- No-op when no descendant in `keep` (existing `test_suppress_no_op_when_no_descendant_in_keep`, no change).
- Cross-rank confidence conservation (`test_suppress_preserves_higher_rank_confidence` in its post-prerequisite form — rank-skip already removed by the cross-rank-accumulator fix; no further change in this plan).
- Leaf invariant (`test_suppress_leaf_only_credited_by_base`, added by the prerequisite plan; no change in this plan).
- Default-threshold emission (updated `test_suppress_ancestor_only_groups_drops_ancestor_when_tied`).

### Integration Tests

Run the existing benchmark harness `classifier_benchmark.py` on `vert12s` and `MiFish` markers with `suppress_ancestor_only_groups = true` and compare the resulting `unified_asv_truth_depth_f1` and species-rank precision/recall against the baseline. The fix should improve species-rank metrics on uncurated databases without disturbing curated-database performance.

### Manual Testing Steps

1. Build: `cargo build --release`.
2. Run the existing test suite: `cargo test --release`.
3. Build the Python wheel: `maturin develop --release`.
4. From the eDNA classifier benchmarking harness, run a single replicate of `vert12s` with `suppress_ancestor_only_groups=True`, threshold=60, default everything else. Inspect species-rank confidence histogram on Canidae/Canis/Oncorhynchus-shaped families.
5. Repeat with `suppress_ancestor_only_groups=False`. Diff the species-rank confidence distributions; expect a rightward shift in the `True` distribution by ~30-50 percentage points on the affected genera.

## Performance Considerations

- **Pre-computation:** `O(n_top^2)` string-prefix comparisons per query, gated on the flag. `n_top` is typically 2-10 (the number of training-tree groups represented in the leaf-phase `keep` set). At `n_top = 10`: 100 prefix comparisons of strings averaging ~80 bytes each — well under 1 µs.
- **Per-replicate filter:** `O(n_tied^2)` index-set scans plus `O(n_top)` allocations of fixed-size scratch buffers (hoisted to once-per-query if optimized). `n_tied` is typically 1-3. At `b = 100`, the total filter cost is ≤ a few thousand operations per query.
- **Net impact:** Negligible relative to the existing `parallel_match` / `parallel_match_inverted` cost (which dominates leaf-phase runtime).
- **Allocation:** Two `Vec<bool>` scratch buffers of length `n_top`, allocated once per `leaf_phase_score` call. Could be reduced to a single `u64` bitset for `n_top <= 64`, but micro-optimization deferred until profiling justifies it.

## Migration Notes

- **Public API:** No new fields, no signature changes. The existing `suppress_ancestor_only_groups` flag picks up the new behavior automatically.
- **Wire format:** No changes to `TrainingSet` serialization. The trained-model file format is untouched.
- **Backwards compatibility:** Default-off invariant preserved. Users with `suppress_ancestor_only_groups = false` see no behavior change. Users with `suppress_ancestor_only_groups = true` see strictly improved species-rank confidence on databases containing ancestor-only training entries; no degradation on curated databases.
- **Threshold recommendations:** The production setup currently uses `threshold ≈ 20` to compensate for the ~50 % depression. After the fix, users running with the flag on can raise `threshold` toward the IDTAXA default of `60` and recover stricter calibration without losing species-rank recall on the affected queries. Document this in the next release note.

## References

- Concern as stated by user: share-splitting at `src/classify.rs:764-794` halves descendant `tot_hits` when ancestor is co-tied; post-bootstrap winner-stage filter cannot recover the credit because the cross-rank accumulator at `src/classify.rs:884-895` reads `tot_hits[]` directly.
- Existing winner-stage filter (the half-fix): `src/classify.rs:825-859` — added in commit `f36cd2e` "Add suppress_ancestor_only_groups flag to drop ancestor-prefix winners".
- Single dispatch into `leaf_phase_score`: `src/classify.rs:305` (greedy) and `src/classify.rs:558` (beam).
- Existing test fixtures and assertions: `tests/test_margin_aware.rs:472-880`.
- Related research: `thoughts/shared/research/2026-04-30-oat-param-bug-investigation.md` (parameter-wiring audit; confirms the existing flag is wired correctly).
- Related research: `thoughts/shared/research/2026-04-27-oxidtaxa-vs-idtaxa-ablation-surface.md` (parameter neutralizing values; confirms `suppress_ancestor_only_groups` defaults to `false`).
