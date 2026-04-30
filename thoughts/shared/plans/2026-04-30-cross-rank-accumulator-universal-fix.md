# Cross-Rank Accumulator Universal Fix Implementation Plan

## Overview

Fix the cross-rank accumulator and LCA-cap walk in `src/classify.rs` so dropped-ancestor evidence credits the ancestor's own rank instead of being misrouted one rank up. Resolves the `asv_7997` regression where `suppress_ancestor_only_groups = true` causes the genus-rank confidence to fall below threshold and emission caps at the family rank.

## Current State Analysis

`HEAD = e5320a4ad31875923231911852136ac7dade5414`. Working tree is clean for source files (only an unrelated docs HTML edit is pending).

Two walk sites in `leaf_phase_score` (`src/classify.rs:576-989`) start each non-selected group's ancestor-walk at `parents[unique_groups[j]]`:

1. **Cross-rank confidence accumulator** at `src/classify.rs:872-883` — credits each non-selected group's `tot_hits` at every matching rank in `predicteds`.
2. **LCA-cap walk** at `src/classify.rs:919-941` — finds each non-selected winner's deepest ancestor in `predicteds` and uses it to cap the reportable lineage.

When `unique_groups[j]` itself lies on `predicteds` (which happens whenever the group is a strict ancestor of `selected` in the taxonomy — i.e., the suppress filter dropped it but its `tot_hits` is still in the array), both walks skip the group's natural rank by starting one node above it.

The existing test `test_suppress_preserves_higher_rank_confidence` (`tests/test_margin_aware.rs:809-880`) was authored under the assumption that the accumulator credits the dropped ancestor's own rank. The assertion loop contains `if i == len - 1 { continue; }` at line 868-870 — which lands exactly on the buggy rank. The skip masks the bug; removing it surfaces a failure under current code.

```
Tree relationship of relevant nodes (asv_7997 / Canis fixture analog):

   Root
    └── Mammalia
         └── Carnivora
              └── Canidae
                   └── Canis            ← ancestor-only group (dropped under FLAG=ON)
                        └── Canis_lupus ← descendant (= selected under FLAG=ON)

   parents[Canis] = Canidae   ← walk start under buggy code
   unique_groups[j] = Canis    ← walk start under universal fix
```

## Desired End State

Both walks start at `unique_groups[j]` (the group's own node). The accumulator credits each non-selected group's `tot_hits` at every rank in `predicteds` at-or-above the group's own node. The LCA-cap accepts `unique_groups[j]` itself as the LCA when the group is an ancestor of `selected`.

Concrete acceptance criteria:

1. The existing `test_suppress_preserves_higher_rank_confidence` test passes **without the rank-skip workaround**.
2. A new leaf-invariant test passes: under FLAG=ON with the byte-identical-tied fixture, every rank shallower than the deepest reported rank has strictly greater confidence than the deepest reported rank.
3. Full `cargo test` suite passes (no regressions in the other existing fixtures, including `test_suppress_multi_descendant_still_caps_at_genus`).
4. Real-data `asv_7997` query emits at `Oncorhynchus` (genus) with confidence ≈ 78% under FLAG=ON, matching the FLAG=OFF baseline.

### Key Discoveries

- `parents[c] = i` for every child `c` of node `i` (`src/training.rs:267-272`). `parents[X]` is never `X` itself.
- Walks visit each ancestor exactly once and stop at Root. Each rank receives at most one increment per non-selected group. ([Research §7.2](../research/2026-04-30-cross-rank-accumulator-ancestor-skip.md))
- Walks only climb. The leaf rank (= `unique_groups[selected]` at the deepest position in `predicteds`) is unreachable from any non-selected group's walk; it is credited only via `base_confidence`. The universal fix preserves this invariant. ([Research §7.3](../research/2026-04-30-cross-rank-accumulator-ancestor-skip.md))
- The fix's effect is monotone — it can only restore previously-skipped credit at the ancestor's own rank; every other rank is bit-identical to current code. ([Research §7.5](../research/2026-04-30-cross-rank-accumulator-ancestor-skip.md))
- The misleading commit-message claim in `f36cd2e` ("ancestor evidence still flows up to ancestor-rank confidences via the cross-rank accumulator") becomes true after the fix.

## What We're NOT Doing

- Not rewriting `f36cd2e`'s commit message. The new commit's message documents the correction.
- Not backporting metric/regression analyses to past evaluations of `suppress_ancestor_only_groups`.
- Not adding a multi-ancestor test fixture (universal fix handles it correctly by construction; deferred until a concrete real-world case surfaces).
- Not changing `classify_one_pass` or `classify_one_pass_beam` descent paths — only `leaf_phase_score`.
- Not introducing a feature flag for the fix (it's a bug correction, not a behavior toggle).
- Not touching the suppress filter itself (`src/classify.rs:811-845`) — only the two downstream walks.

## Implementation Approach

Two phases.

**Phase 1** swaps the walk-start at both bug sites in `src/classify.rs` (two single-line changes plus comment cleanup). The existing test suite must continue to pass after Phase 1 because the change is monotone — it only restores credit, never lowers or duplicates it.

**Phase 2** hardens the regression coverage: removes the rank-skip workaround in the existing test and adds a new leaf-invariant test that locks the no-double-count property explicitly.

The phases are intentionally separable: Phase 1's success criteria use the **existing** test suite (which already passes); Phase 2's success criteria use the new/updated tests, which would fail against Phase 1-only code if Phase 1 had a regression. This staging gives a clean before/after demarcation.

---

## Phase 1: Universal fix at both walk sites in classify.rs

### Overview
Change two walk-start lines from `parents[unique_groups[j]]` to `unique_groups[j]`. Update the surrounding doc comments so they describe the corrected semantic. Add a one-line rationale comment at each changed site.

### Changes Required

#### 1. Cross-rank confidence accumulator
**File**: `src/classify.rs`
**Location**: line 874 (inside the loop at lines 872-883)
**Change**: walk-start node

```rust
// BEFORE (line 874):
            let mut p = parents[unique_groups[j]];

// AFTER:
            // Start at the group's own node (not its parent) so a group whose
            // own node lies on `predicteds` — the case for an ancestor-only
            // training entry whose descendant is `selected` — credits its
            // tot_hits at its OWN rank instead of skipping past it. For
            // sibling/descendant groups, `unique_groups[j]` is not in
            // `predicteds` and the first iteration is a harmless no-match.
            let mut p = unique_groups[j];
```

What stays the same: the loop body, the stop condition, the per-rank `confidences[m] += th / b as f64 * 100.0` increment.

#### 2. LCA-cap walk
**File**: `src/classify.rs`
**Location**: line 923 (inside the loop at lines 919-941)
**Change**: walk-start node

```rust
// BEFORE (line 923):
            let mut p = parents[unique_groups[j]];

// AFTER:
            // Start at the winner's own node so the LCA-cap accepts
            // `unique_groups[j]` itself as the cap when this winner is a
            // strict ancestor of `selected` (i.e., the LCA is the winner's
            // own node). The pre-fix start at parents[X] capped one rank
            // too shallow in that case.
            let mut p = unique_groups[j];
```

What stays the same: the `break` after the first match, the `deepest_allowed = min(...)` update, the alternatives construction.

#### 3. Update the suppress-filter rationale comment
**File**: `src/classify.rs`
**Location**: lines 796-810 (the comment preceding the suppress filter)
**Change**: the comment claims the accumulator at lines "775-786" credits ancestor-rank confidences via tot_hits iteration. With the fix this becomes accurate, but the line range is stale and the wording can be sharpened.

Replace the existing comment block with:

```rust
    // When enabled, drop any winner whose full taxonomic path is a strict
    // prefix of another winner's path. This prevents ancestor-only training
    // entries (e.g. "Oncorhynchus sp." after canonical NA-trim) from
    // triggering LCA-cap collapse to the ancestor's rank when a species-
    // resolved descendant ties with them in the bootstrap. The ancestor's
    // kmer evidence is unaffected: `tot_hits[ancestor]` is unchanged and
    // still flows up to ancestor-rank confidences via the cross-rank
    // accumulator at lines 872-883 (which iterates `tot_hits`, not
    // `winners`, and starts each walk at the group's own node so the
    // ancestor's own rank is credited).
    //
    // ts.taxonomy[node] is stored with a trailing ';' (built in
    // training.rs:202-213 as taxa = format!("Root;{}", t) where t was built
    // with format!("{};", s).collect()). The trailing ';' makes starts_with
    // safe against false-prefix matches like "Salmo;" being treated as a
    // prefix of "Salmoninae;" — without it, "Salmo" would falsely match the
    // start of "Salmoninae".
```

#### 4. Update the LCA-cap rationale comment
**File**: `src/classify.rs`
**Location**: lines 911-918 (the comment preceding the LCA-cap walk)
**Change**: tighten "deepest ancestor of unique_groups[j]" → "deepest ancestor-or-equal of unique_groups[j]".

```rust
    // When multiple groups are tied at `max_tot`, the classifier cannot honestly
    // resolve below the LCA of the tied set. Compute that LCA's position in
    // `predicteds` so the `above` filter can cap the reportable lineage there.
    //
    // Every non-selected winner's pairwise LCA with `selected` is the deepest
    // ancestor-or-equal of `unique_groups[j]` that lives in `predicteds`. The
    // group-wise LCA is the shallowest (smallest index) of those pairwise
    // LCAs, since it must be an ancestor of every winner. The walk below
    // starts at `unique_groups[j]` itself so an ancestor-only winner's own
    // node is accepted as the cap (rather than its parent's position).
```

### Success Criteria

#### Automated Verification:
- [x] Code compiles: `cargo build --release`
- [ ] Format passes: `cargo fmt --check` — **pre-existing failures on HEAD unrelated to this fix** (binom_coeff helper formatting at `src/classify.rs:1208`); my edits introduce no new formatting issues
- [ ] Lint passes: `cargo clippy --all-targets -- -D warnings` — **pre-existing failures on HEAD unrelated to this fix** (`needless_range_loop` warnings at multiple sites, `type_complexity` in training.rs); my edits introduce no new lints
- [x] Full test suite passes: `cargo test` (94/94)
- [x] Targeted suppress-tests pass: `cargo test --test test_margin_aware suppress_` (6/6 — was 5/5 before Phase 2 added the leaf invariant)

#### Manual Verification:
- [ ] Real-data `asv_7997` query under FLAG=ON emits at `Oncorhynchus` (genus) with confidence approximately matching the FLAG=OFF baseline (~78%). **Requires user to run production pipeline.**
- [x] Spot-check: the existing `test_suppress_preserves_higher_rank_confidence` still passes (Phase 1 verified the change is monotone — passes both with and without the rank-skip workaround).

---

## Phase 2: Test coverage hardening

### Overview
Remove the `if i == len - 1 { continue; }` skip in `test_suppress_preserves_higher_rank_confidence` so the test asserts equality at every rank — making it a true regression test for the bug. Add a new `test_suppress_leaf_only_credited_by_base` that locks the no-double-count leaf invariant.

### Changes Required

#### 1. Remove the rank-skip workaround in the existing test
**File**: `tests/test_margin_aware.rs`
**Location**: lines 864-879 (the comparison loop in `test_suppress_preserves_higher_rank_confidence`)
**Change**: drop the skip; assert equality across all ranks.

Replace the loop body with:

```rust
    let len = off[0].confidence.len().min(on[0].confidence.len());
    assert!(len >= 4, "expected at least Root..Canidae reported, got off={}, on={}",
            off[0].confidence.len(), on[0].confidence.len());
    for i in 0..len {
        // No skip: the cross-rank accumulator credits each non-selected
        // group's tot_hits at every rank at-or-above the group's own node.
        // For the dropped ancestor, that includes the ancestor's natural
        // rank (the deepest shared rank between off and on lineages).
        // Pre-fix this assertion failed at i == len-1 because the walk
        // started at parents[unique_groups[j]] and skipped that rank.
        assert!(
            (off[0].confidence[i] - on[0].confidence[i]).abs() < 1e-3,
            "rank {} confidence differs: off={}, on={} \
             (cross-rank accumulator should credit ancestor's own rank)",
            i,
            off[0].confidence[i],
            on[0].confidence[i]
        );
    }
```

#### 2. Add the leaf-invariant test
**File**: `tests/test_margin_aware.rs`
**Location**: insert immediately after `test_suppress_preserves_higher_rank_confidence` (around line 880, before the `use_idf_in_descent` section banner at line 882)
**Change**: add a new `#[test]` function.

```rust
#[test]
fn test_suppress_leaf_only_credited_by_base() {
    // Leaf invariant: under FLAG=ON with a tied ancestor-descendant fixture,
    // the deepest reported rank receives ONLY base_confidence (= the
    // selected group's own tot_hits contribution). No walk from a
    // non-selected group can reach the leaf because walks only climb the
    // tree, and every other group lives at-or-above the selected.
    //
    // Concretely: every rank shallower than the leaf accumulates the
    // dropped ancestor's tot_hits via the cross-rank accumulator, so the
    // shallower ranks have strictly greater confidence than the leaf.
    // If this invariant fails, the accumulator is double-crediting the
    // leaf or the ancestor's evidence is being misrouted.
    let ts = build_ancestor_descendant_fixture();
    let query = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
            .to_string(),
    ];
    let names = vec!["q".to_string()];

    // threshold=1.0 so all ranks pass the threshold filter and the full
    // lineage is reported (we want to assert on every confidence, not on
    // the truncated emission output).
    let cfg = ClassifyConfig {
        threshold: 1.0,
        suppress_ancestor_only_groups: true,
        ..Default::default()
    };
    let result = id_taxa(
        &query, &names, &ts, &cfg, StrandMode::Top, OutputType::Extended, 42, true,
    );

    assert_eq!(result.len(), 1);
    let conf = &result[0].confidence;
    assert!(
        conf.len() >= 2,
        "expected at least leaf + one ancestor rank, got {}",
        conf.len()
    );

    let leaf = *conf.last().unwrap();
    for (i, &c) in conf.iter().enumerate().take(conf.len() - 1) {
        assert!(
            c > leaf + 1e-3,
            "rank {} confidence ({}) must be strictly greater than leaf ({}); \
             violation suggests ancestor evidence is misrouted or the leaf \
             is double-credited",
            i,
            c,
            leaf
        );
    }
}
```

### Success Criteria

#### Automated Verification:
- [ ] Format passes: `cargo fmt --check` — **pre-existing failures, see Phase 1 note**
- [ ] Lint passes: `cargo clippy --all-targets -- -D warnings` — **pre-existing failures, see Phase 1 note**
- [x] Updated `test_suppress_preserves_higher_rank_confidence` passes
- [x] New `test_suppress_leaf_only_credited_by_base` passes
- [x] Full suppress-test set passes (6/6): `cargo test --test test_margin_aware suppress_`
- [x] Full test suite passes (94/94): `cargo test`

#### Manual Verification:
- [x] Assertion messages cite "accumulator" / "ancestor evidence" — both updated/new tests have failure messages that name the mechanism so a future reader can recognize the bug class.
- [ ] (Optional) Revert Phase 1 locally and confirm both updated tests fail under the unfixed code. **Not run automatically; left to user as a stretch verification.**

---

## Testing Strategy

### Unit Tests

| Test | Purpose | Should pass before fix? | Should pass after fix? |
|---|---|---|---|
| `test_suppress_ancestor_only_groups_default_off` | Default flag is off | Yes | Yes |
| `test_suppress_ancestor_only_groups_drops_ancestor_when_tied` | Mode B reproduction; species emits when ancestor dropped | Yes | Yes |
| `test_suppress_no_op_when_no_descendant_in_keep` | Honest abstention preserved | Yes | Yes |
| `test_suppress_multi_descendant_still_caps_at_genus` | LCA-cap fires when multiple species tie | Yes | Yes |
| `test_suppress_preserves_higher_rank_confidence` (post-edit) | Cross-rank accumulator credits ancestor's own rank | **No** (fails at i=len-1) | Yes |
| `test_suppress_leaf_only_credited_by_base` (new) | Leaf gets only base; no double-counting | **No** (genus rank equals leaf under bug) | Yes |

The two updated/new tests are designed to fail against the unfixed code at exactly the rank where the bug manifests — that's the regression coverage.

### Integration Tests

Real-data `asv_7997` end-to-end run is the integration check. Manual step in Phase 1 verification.

### Manual Testing Steps

1. After Phase 1: run `cargo test --test test_margin_aware suppress_` and confirm all five existing tests pass.
2. After Phase 1: run the production classification pipeline against the asv_7997 reference set with `suppress_ancestor_only_groups = true` and confirm the genus call survives at ~78% confidence.
3. After Phase 2: run `cargo test --test test_margin_aware` (full file) and confirm both updated/new tests pass.
4. (Optional but recommended) Revert just the two `let mut p = ...` lines in Phase 1, re-run `cargo test --test test_margin_aware test_suppress_preserves_higher_rank_confidence test_suppress_leaf_only_credited_by_base` — confirm both fail. Re-apply Phase 1.

## Performance Considerations

The universal fix adds at most one extra `predicteds.iter().position()` call per non-selected group per accumulator invocation. `predicteds.len()` is bounded by lineage depth (≈ 8 for typical taxonomies), so the cost is `O(lineage_depth)` extra integer comparisons per non-selected group — negligible relative to the existing `b`-replicate bootstrap and per-replicate winner-take-all loops at `src/classify.rs:750-779`.

No allocation changes. No new fields on any struct. No new `use` imports.

## Migration Notes

No on-disk model format changes. No `ClassifyConfig` field changes. No Python-binding signature changes. Models trained before this fix produce identical `tot_hits` arrays under the same query — the fix changes only how those `tot_hits` are accumulated into `confidences`. So:

- Pre-fix and post-fix builds remain bincode-compatible.
- A user re-running classification with the fix applied may see different `confidence` numbers and emission depths for queries that exercise the bug path (specifically, queries whose bootstrap ties an ancestor-only group with a species descendant). All other queries are unaffected.
- The `suppress_ancestor_only_groups` flag's default remains `false`, so users who haven't opted into the suppress flag see no change at all unless they were relying on the rare FLAG=OFF tied-ancestor LCA-cap path (corner case).

## References

- Research document: `thoughts/shared/research/2026-04-30-cross-rank-accumulator-ancestor-skip.md`
- Suppress flag introduction: commit f36cd2e ("Add suppress_ancestor_only_groups flag to drop ancestor-prefix winners")
- Bug sites:
  - `src/classify.rs:872-883` — cross-rank confidence accumulator
  - `src/classify.rs:919-941` — LCA-cap walk
- Test sites:
  - `tests/test_margin_aware.rs:809-880` — `test_suppress_preserves_higher_rank_confidence` (rank-skip workaround at 868-870)
  - `tests/test_margin_aware.rs:491-534` — `build_ancestor_descendant_fixture` (Canis genus-only + Canis_lupus, byte-identical sequences)
- Supporting code:
  - `src/training.rs:267-272` — `parents[]` construction
  - `src/classify.rs:811-845` — suppress filter (unchanged by this plan)
  - `src/classify.rs:847-867` — `selected` and `predicteds` construction
- Project tooling:
  - `Cargo.toml` (Rust 2021 edition, optional pyo3 feature)
  - No `.github/workflows/` present; canonical commands are `cargo fmt`, `cargo clippy`, `cargo test`
