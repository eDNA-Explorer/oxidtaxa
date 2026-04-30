---
date: 2026-04-30T00:00:00-07:00
researcher: Ryan Martin
git_commit: e5320a4ad31875923231911852136ac7dade5414
branch: main
repository: eDNA-Explorer/oxidtaxa
topic: "Cross-rank accumulator misroutes dropped-ancestor evidence past the ancestor's own rank"
tags: [research, codebase, oxidtaxa, classify, accumulator, suppress_ancestor_only_groups, asv_7997, parents, predicteds, lca_cap, double_counting]
status: complete
last_updated: 2026-04-30
last_updated_by: Ryan Martin
last_updated_note: "Initial write. Covers bug claim verification, test-masking discovery, LCA-cap parallel pattern, selective vs universal fix scope, and rank-by-rank double-count analysis."
---

# Research: Cross-rank accumulator skips ancestor's own rank when suppress drops it

**Date**: 2026-04-30
**Researcher**: Ryan Martin
**Git Commit**: e5320a4ad31875923231911852136ac7dade5414 (HEAD); flag introduced in f36cd2e
**Branch**: main
**Repository**: eDNA-Explorer/oxidtaxa

## Research Question

Real-data classification of `asv_7997` regressed when the `suppress_ancestor_only_groups` flag was set to `true`:

- **FLAG = OFF (baseline)**: emission terminates at `Oncorhynchus` (genus) with confidence ≈ 78–82%.
- **FLAG = ON**: emission caps at `Salmoninae` (subfamily); the genus rank's confidence falls below the 60% threshold and the call is truncated. Species (`Oncorhynchus mykiss`) does not emit either.

The bootstrap-time evidence under both flag settings supports "Oncorhynchus genus" with combined `tot_hits` ≈ 78% (`mykiss` group + ancestor-only `Oncorhynchus` group, each contributing ≈ half via per-replicate tied-share splitting at `src/classify.rs:770-779`). The flag is not _changing_ the bootstrap evidence — it is changing where that evidence gets credited in the per-rank `confidences` vector.

This document investigates whether the regression is a wiring bug, characterizes the exact mechanism, surfaces a parallel pattern in the LCA-cap walk, evaluates two fix scopes, and proves the universal fix does not double-count any rank (in particular, the leaf rank).

## Summary

**Confirmed bug.** The cross-rank accumulator at `src/classify.rs:872-883` starts each non-selected group's ancestor-walk at `parents[unique_groups[j]]` rather than `unique_groups[j]` itself. When the dropped ancestor's own taxonomy node happens to lie on the reported lineage (which it does whenever `selected = descendant`), the walk skips one rank — exactly the dropped ancestor's natural rank — and credits that group's `tot_hits` to the next rank up.

The same mis-start pattern exists in the LCA-cap walk at `src/classify.rs:919-941`. With `suppress_ancestor_only_groups = ON` the dropped ancestor is filtered out of `winners`, so the LCA-cap loop is unaffected; but with FLAG = OFF and an ancestor tied with a descendant where the RNG happens to pick the descendant as `selected`, the LCA-cap caps the lineage one rank too shallow.

**Fix.** Either start each walk at `unique_groups[j]` (universal) or only for groups that the suppress filter dropped (selective). The universal fix is a single-line change, has identical behavior to current code on every non-ancestor group, and additionally repairs the tied-ancestor LCA-cap pattern. **It cannot double-count any rank** — proved by case analysis below.

| Aspect | Finding |
|---|---|
| Bug exists? | Yes, in `classify.rs:872-883` (accumulator) and `classify.rs:919-941` (LCA-cap) |
| Triggering condition (suppress=ON path) | Whenever the suppress filter drops an ancestor whose own node lies on `predicteds` |
| Triggering condition (suppress=OFF path) | Whenever an ancestor and descendant tie in the bootstrap and RNG picks the descendant as `selected` |
| Test coverage | `test_suppress_preserves_higher_rank_confidence` exists but **skips the buggy rank** with `if i == len - 1 { continue; }` |
| Commit message claim | f36cd2e claims "ancestor evidence still flows up to ancestor-rank confidences via tot_hits iteration" — this claim is false in the merged code |
| Recommended fix scope | Universal (single-line change at both walks; strict improvement; no double-counting) |
| Side effect on other ranks | None. Universal fix changes credit only at the previously-skipped rank; every other rank's credit is bit-identical to current code |

## Detailed Findings

### 1. Tree structure and `parents[]` semantics

The taxonomy is held as a flat array of node IDs with a parallel `parents[]` lookup. The construction at `src/training.rs:267-272` writes:

```
parents[c] = i   for every child c of node i
parents[Root] = Root   (self-loop; loop terminator)
```

So `parents[X]` is **always** the parent of X — never X itself. There is no way to ask "what is X's own node?" through this table.

```
                       Root  (parents[Root] = Root)
                        │
                        ▼
                     Mammalia                        parents[Mammalia]  = Root
                        │
                        ▼
                     Carnivora                       parents[Carnivora] = Mammalia
                        │
                        ▼
                     Canidae                         parents[Canidae]   = Carnivora
                        │
                        ▼
                     Canis                           parents[Canis]     = Canidae
                        │                            ──────────────────────────────
                        ▼                            ↑
                     Canis_lupus                     For ancestor walks that *start*
                                                     at parents[X], the walk's first
                                                     visit is one rank above X.
                                                     X itself is never visited.
```

(asv_7997 case: substitute Salmoninae → Canidae, Oncorhynchus → Canis, Oncorhynchus mykiss → Canis_lupus. The fixture in `tests/test_margin_aware.rs:491-534` uses the renamed lineage so unit tests run without a salmonid taxonomy.)

### 2. The accumulator's walk start

Source: `src/classify.rs:870-883`.

The accumulator initializes a `confidences[predicteds.len()]` vector with `base_confidence = tot_hits[selected] / b * 100` applied uniformly, then walks every other group's ancestors and adds that group's `tot_hits / b * 100` at every rank in `predicteds` the walk passes through.

```
For each non-selected group j with positive tot_hits:

   let p = parents[unique_groups[j]]              ← current code starts here
   loop:
       if p is in predicteds at position m:
           confidences[m] += th_j
       if p is Root: break
       p = parents[p]
```

**Where the bug lives.** When `unique_groups[j]` happens to be in `predicteds`, the walk's first `p` is one rank above it. The match at `unique_groups[j]`'s own position is silently skipped.

This _only matters_ for groups whose own taxonomy node lies on `predicteds`. By construction, that requires the group to be a strict ancestor of `selected`'s group — i.e., an ancestor-only training entry like `"Oncorhynchus sp."` after canonical NA-trim. For sibling groups and descendants of selected, `unique_groups[j]` is not in `predicteds` and the walk start at `parents[X]` is harmless (only one extra `position()` no-match check is saved).

### 3. Side-by-side: FLAG = OFF versus FLAG = ON

Setup:
- Ancestor-only entry `"Canis"` (taxonomy ends at the genus node) and species entry `"Canis_lupus"` share **byte-identical** training sequences. Bootstrap produces an exact tie under per-replicate tied-share splitting (`src/classify.rs:770-779`).
- Both groups end with `tot_hits ≈ X`, where each contribution per rank is `X/b * 100 ≈ ~28 percentage points` (orders of magnitude depending on shared-share split; the exact value tracks with how many replicates tied).

```
═══════════════════════════════ FLAG = OFF ═══════════════════════════════
                                         (RNG happens to pick ancestor)
   selected = Canis (the genus-only group)

   predicteds  =  [ Root │ Mammalia │ Carnivora │ Canidae │ Canis ]
   position    =     0   │    1     │     2     │    3    │   4
                                                              ↑
                                                       deepest reported

   Initialize confidences:  [ base │ base │ base │ base │ base ]   (base = X)

   Walk for the OTHER tied group (Canis_lupus, j ≠ selected):

     start = parents[Canis_lupus] = Canis           ← position 4 in predicteds
     visit Canis      → confidences[4] += X         ✓
     visit Canidae    → confidences[3] += X         ✓
     visit Carnivora  → confidences[2] += X
     visit Mammalia   → confidences[1] += X
     visit Root       → confidences[0] += X
     STOP.

   Final:
                  Root      Mam       Carn      Canidae    Canis
                  0         1         2         3          4
                  base+X    base+X    base+X    base+X     base+X
                                                            ≈ 78%   ← ✅ EMITS at genus

═══════════════════════════════ FLAG = ON ════════════════════════════════
                                  (suppress filter drops Canis from winners)
   selected = Canis_lupus (the species)

   predicteds  =  [ Root │ Mammalia │ Carnivora │ Canidae │ Canis │ Canis_lupus ]
   position    =     0   │    1     │     2     │    3    │   4   │     5
                                                              ↑       ↑
                                                          GENUS    species (= selected)

   Initialize confidences:  [ base │ base │ base │ base │ base │ base ]

   Walk for the dropped Canis ancestor (still in tot_hits, j ≠ selected):

     start = parents[Canis] = Canidae               ← !!! starts at PARENT of Canis !!!
     visit Canidae    → confidences[3] += X
     visit Carnivora  → confidences[2] += X
     visit Mammalia   → confidences[1] += X
     visit Root       → confidences[0] += X
     STOP.

     ⚠  position 4 (Canis genus rank) was NEVER VISITED  ⚠
     The ancestor's evidence skipped right OVER its own rank.

   Final:
                  Root      Mam       Carn      Canidae    Canis        Canis_lupus
                  0         1         2         3          4              5
                  base+X    base+X    base+X    base+X     base only      base only
                                                            ≈ 50%          ≈ 50%
                                                            ↑
                                              Falls below the 60 threshold.
                                              Emission caps at Canidae.
                                              Genus call is lost despite
                                              the bootstrap supporting it.
```

The drop in genus confidence is exactly the contribution `Canis_th / b * 100` that was misrouted to Canidae instead. Magnitude is comparable to `base` because the two tied groups have similar `tot_hits` under tied-share splitting.

### 4. The existing test was masking the bug

`tests/test_margin_aware.rs:864-879` (`test_suppress_preserves_higher_rank_confidence`) compares `off` and `on` confidence vectors at every rank EXCEPT the deepest shared rank:

```
   off (FLAG=OFF, selected = ancestor):    Root  Mam  Carn  Canidae  Canis
                                            0     1    2     3       4         (5 ranks)

   on  (FLAG=ON, selected = descendant):  Root  Mam  Carn  Canidae  Canis  Canis_lupus
                                            0     1    2     3       4        5     (6 ranks)

   len = min(5, 6) = 5
   Loop i = 0..5  but skips when i == len - 1 == 4

      Compared:  ✓     ✓    ✓     ✓        SKIP
                 0     1    2     3        4
                                            ↑
                                THE buggy rank — exactly the one skipped.
```

The skip rationale in the test comment ("Skip the rank at which the LCA-cap or threshold may have affected emission identity") sounds plausible but is actually papering over the accumulator bug. In this fixture, `len - 1` is the dropped ancestor's natural rank — the one place the bug's misrouting shows up.

**Removing the `continue` and asserting equality would cause the test to fail under current code and pass under the fix.** This becomes the formal regression test.

### 5. The LCA-cap walk has the same pattern

Source: `src/classify.rs:919-941`. The LCA-cap loop iterates over `winners` (post-suppress filter) and walks each non-selected winner's ancestors looking for the LCA position in `predicteds`:

```
For each j in winners, j ≠ selected:
   let p = parents[unique_groups[j]]              ← same mis-start pattern
   loop:
       if p is in predicteds at position pos:
           deepest_allowed = min(deepest_allowed, pos)
           break
       walk up
```

| Suppress flag | Ancestor in `winners`? | LCA-cap walk affected by bug? |
|---|---|---|
| ON | No (filter dropped it) | No — the loop never sees the ancestor |
| OFF | Yes | **Yes** — when RNG picks descendant as `selected`, the ancestor's walk caps `deepest_allowed` at its parent's position rather than its own, truncating the reportable lineage one rank too shallow |

So the same root cause produces a separate (rarer) mis-cap when running with `suppress_ancestor_only_groups = false` and the bootstrap ties an ancestor-only group with a species descendant.

### 6. Fix scope: selective versus universal

Two valid fixes:

```
SELECTIVE FIX
─────────────
   Track which winners were dropped by the suppress filter.
   For dropped ancestors, start the walk at unique_groups[j].
   For all other groups, keep the legacy parents[unique_groups[j]] start.

   Implementation: builds a HashSet<usize> of dropped indices outside the
   accumulator, branches on membership at each iteration.

   Pros: minimal blast radius — legacy path untouched for non-dropped groups.
   Cons: Adds plumbing. Does NOT fix the LCA-cap pattern at OFF-path tied
         ancestors, nor the rare OFF-path accumulator misrouting.

UNIVERSAL FIX
─────────────
   Always start the walk at unique_groups[j] (in both the accumulator
   and the LCA-cap walk).

   Implementation: change `parents[unique_groups[j]]` to `unique_groups[j]`
   at two call sites. Delete any HashSet plumbing.

   Pros: Single-line change per site. Repairs both bug variants
         (suppress=ON dropped-ancestor and suppress=OFF tied-ancestor).
         Strictly correct semantics: "credit the group's tot_hits at every
         rank in predicteds at-or-above the group's own node."

   Cons: One extra (cache-friendly) position() check at the walk start for
         non-ancestor groups. Negligible — predicteds.len() ≤ ~8.
```

**Behavioral equivalence on non-ancestor groups.**

| Group's relationship to `selected` | `unique_groups[j]` in `predicteds`? | Selective fix | Universal fix |
|---|---|---|---|
| Sibling subtree | No | Walk identical to current code | One extra no-match check, then identical |
| Strict descendant of selected (only possible when selected ≠ leaf) | No (predicteds ends at selected) | Walk identical | One extra no-match check, then identical |
| Strict ancestor of selected (the bug case) | **Yes** | **Restores credit at j's own rank** | **Restores credit at j's own rank** |

For the only relationship where the two fixes differ (strict ancestor of selected with `suppress = OFF` and RNG picks descendant), the universal fix is correct and the selective fix leaves the bug intact.

### 7. Double-counting analysis (universal fix is monotone)

A natural concern: if the walk now starts at the group's own node, does any rank receive _two_ credits from the same group, or does the leaf rank inherit credit from somewhere it shouldn't?

**Answer: no, never. Proof by case analysis.**

#### 7.1 Walks only climb

```
                                  ┌─── Root
                                  │
                                  ▼
                                Mammalia
                                  │
                                  ▼
                                Carnivora
                                  │
                                  ▼
                                Canidae
                                  │
                                  ▼
                                Canis           ← ancestor
                                  │
                                  ▼
                                Canis_lupus     ← LEAF (= deepest in predicteds)


   The accumulator's walk loop visits parents only — it never descends.
   So for any rank R in predicteds, R can only receive credits from
   groups whose nodes are AT R or BELOW R in the tree.

   The deepest rank in predicteds = unique_groups[selected] (selected's own node).
```

#### 7.2 Per-rank credit accounting

```
For any rank R in predicteds, confidences[R] is built from:

   1. base_confidence = tot_hits[selected] / b * 100
      → applied to every rank uniformly, exactly once.

   2. one increment per non-selected group j whose walk passes through R
      → a walk passes through R iff R is an ancestor of, or equal to,
        unique_groups[j].

The loop visits each j exactly once. Inside that visit, the walk visits
each ancestor exactly once and stops at Root. So each rank R receives
at most one increment per j.

There is no path through which the same group's tot_hits can credit R twice.
```

#### 7.3 The leaf rank is special — it can ONLY get base

```
position 5 is the deepest rank in predicteds (= unique_groups[selected]).

For another group j ≠ selected to credit position 5, the walk would need
to visit unique_groups[selected]. The walk only visits ancestors of j's
node, so this requires unique_groups[selected] to be an ancestor of
unique_groups[j].

That means j's group is a STRICT DESCENDANT of selected.

But predicteds is the lineage of selected: it terminates AT selected.
A strict descendant of selected has its own node at depth > depth(selected),
which is strictly deeper than predicteds' deepest entry — and that
node simply isn't in predicteds at all.

⇒ No walk from a non-selected group can reach the leaf rank.
⇒ confidences[leaf] = base_confidence, identically, under either current
   code or the universal fix.

(Note: this proof is symmetric in current code and universal fix —
neither can double-count the leaf. The fix only changes credit at the
ONE rank where the walk's first visit was being silently skipped.)
```

#### 7.4 Full accounting trace, FLAG=ON, universal fix

```
Setup: selected = Canis_lupus, two non-zero tot_hits:
       - Canis_lupus (selected; via base only)
       - Canis (the dropped ancestor; via the loop)

predicteds:    Root  Mammalia  Carnivora  Canidae  Canis  Canis_lupus
position:       0       1         2         3        4         5

Step A — initialize:
   confidences = [base, base, base, base, base, base]

Step B — walk for j = Canis (start at unique_groups[j] = Canis_node, position 4):
   visit Canis_node      → confidences[4] += Canis_th  ← previously skipped, now restored
   visit Canidae_node    → confidences[3] += Canis_th
   visit Carnivora_node  → confidences[2] += Canis_th
   visit Mammalia_node   → confidences[1] += Canis_th
   visit Root            → confidences[0] += Canis_th
   STOP.

   The walk visits positions {0,1,2,3,4}. Position 5 is NEVER touched.

Final:
              Root      Mam       Carn      Canidae    Canis     Canis_lupus
   contrib:   base+     base+     base+     base+      base+     base
              Canis_th  Canis_th  Canis_th  Canis_th   Canis_th
   numeric:   ~78%      ~78%      ~78%      ~78%       ~78%      ~50%
                                                        ↑          ↑
                                              fix restores      leaf gets ONLY
                                              the missing       base — single
                                              credit            contributor
```

#### 7.5 Side-by-side delta (current vs universal fix)

```
                    Root    Mam    Carn   Canidae  Canis    Canis_lupus
                    pos 0   pos 1  pos 2  pos 3    pos 4    pos 5  (leaf)
─────────────────  ──────  ─────  ─────  ───────  ───────  ────────────
CURRENT (buggy):
  base_confidence    ✓       ✓      ✓      ✓        ✓         ✓
  Canis ancestor     ✓       ✓      ✓      ✓        ✗         ─ (walk doesn't reach)
                                                    ↑
                                              SKIPPED — bug

UNIVERSAL FIX:
  base_confidence    ✓       ✓      ✓      ✓        ✓         ✓
  Canis ancestor     ✓       ✓      ✓      ✓        ✓         ─
                                                    ↑
                                              RESTORED — fix

DELTA per rank:    same    same   same   same    +Canis_th   same
                                                                ↑
                                                          LEAF UNCHANGED
                                                          (proves no double-count)
```

**Conclusion.** The universal fix is monotone: it can only ever raise confidences, never lower or duplicate them. It restores credit at exactly one rank — the previously-skipped one. Every other rank, including the leaf, is bit-identical to current code.

### 8. Numerical asv_7997 estimate

User-reported numbers (real-data run):
- FLAG=OFF baseline genus emit ≈ 78–82%.
- FLAG=ON genus confidence falls ≈ 28 percentage points below the 60 threshold (so roughly ≈ 50% or lower).

These are consistent with byte-identical-fixture math. With per-replicate tied-share splitting and two tied groups, each ends with `tot_hits ≈ X`. The buggy code routes only `base` (≈ X) to the genus rank under FLAG=ON, instead of `base + ancestor_th` (≈ 2X) under FLAG=OFF. The drop magnitude tracks `ancestor_th / b * 100` which is the same scale as `base_confidence`. A 28-point drop on an 80-point baseline is in the right ballpark for a tied two-group case.

### 9. The f36cd2e commit message claim

The commit that introduced `suppress_ancestor_only_groups` (f36cd2e) states:

> "The accumulator at classify.rs:775-786 is unchanged: ancestor evidence still flows up to ancestor-rank confidences via tot_hits iteration, so 'use these seqs at the levels they do go to' is preserved."

This claim is **false in the merged code**. The accumulator is unchanged from before, but the claim that ancestor evidence reaches ancestor-rank confidences via `tot_hits` iteration assumes the walk visits the ancestor's own node — which it does not, because it starts at `parents[ancestor]`. The "use these seqs at the levels they do go to" property holds at all ranks _shallower_ than the ancestor's natural rank, but fails at the ancestor's natural rank itself.

The universal fix makes the commit message claim true.

## Recommendations

1. **Apply the universal fix** at both walk sites:
   - `src/classify.rs:874` (accumulator): change `let mut p = parents[unique_groups[j]];` to `let mut p = unique_groups[j];`.
   - `src/classify.rs:923` (LCA-cap walk): same change.
   - Add a one-line comment at each site noting that starting at the group's own node is what makes ancestor-group evidence reach ancestor-rank confidences when the ancestor lies on `predicteds`.

2. **Remove the test skip** in `tests/test_margin_aware.rs:868-870`. Replace with an explicit assertion that confidence at the dropped-ancestor's rank is equal under FLAG=OFF and FLAG=ON (or that on's confidence at that rank is ≥ off's, since the fix can only equal or exceed the bug). This becomes the regression test.

3. **Add a leaf invariant test**: explicitly assert that under FLAG=ON, `confidences[predicteds.len() - 1] == base_confidence` (within float tolerance). This locks in the no-double-count property at the leaf.

4. **Update the f36cd2e commit message claim** by adding a follow-up commit message that documents the fix and the corrected semantic.

5. **(Optional)** Add a second fixture for the FLAG=OFF tied-ancestor case to cover the LCA-cap parallel pattern. Without RNG control this is non-deterministic; the test may need to fix the RNG seed and assert the LCA-cap correctly identifies the ancestor's own rank as the cap, not the parent's rank.

## Code References

### Bug sites
- `src/classify.rs:872-883` — accumulator with mis-start at `parents[unique_groups[j]]`
- `src/classify.rs:919-941` — LCA-cap walk with the same mis-start pattern

### Supporting code
- `src/training.rs:267-272` — `parents[]` construction (each child indexes into its parent)
- `src/classify.rs:811-845` — `suppress_ancestor_only_groups` filter (drops ancestor-prefix winners from `winners` before `selected` is picked)
- `src/classify.rs:847-867` — `selected` selection and `predicteds` construction (lineage from `unique_groups[selected]` up to Root, then reversed)
- `src/classify.rs:870-871` — `base_confidence` initialization (uniform across all ranks)
- `src/classify.rs:770-779` — per-replicate tied-share splitting (the source of equal `tot_hits` between byte-identical groups)

### Test masking
- `tests/test_margin_aware.rs:491-534` — `build_ancestor_descendant_fixture` (Canis genus-only + Canis_lupus species, byte-identical sequences)
- `tests/test_margin_aware.rs:809-880` — `test_suppress_preserves_higher_rank_confidence`
- `tests/test_margin_aware.rs:868-870` — the `if i == len - 1 { continue; }` skip that masks the bug

### Commit-message claim
- f36cd2e — "Add suppress_ancestor_only_groups flag to drop ancestor-prefix winners" — claims the accumulator's iteration over `tot_hits` preserves ancestor-rank confidences

## Architecture Documentation

The leaf-phase scoring path in `src/classify.rs:573-989` (`leaf_phase_score`) ends with a four-stage pipeline:

```
   bootstrap →  tot_hits per group
       │
       ▼
   ties → winners (subject to tie_margin and the suppress filter)
       │
       ▼
   selected = RNG pick from winners
       │
       ▼
   predicteds = lineage of unique_groups[selected]
       │
       ▼
   confidences[predicteds.len()] initialized to base_confidence
       │
       ▼
   accumulator: walks each non-selected group's ancestors and credits
   matching ranks in predicteds
       │
       ▼
   LCA-cap: walks each non-selected winner's ancestors to find the
   shallowest LCA and uses it as `deepest_allowed`
       │
       ▼
   confidence-margin discount → above-threshold filter → final ClassificationResult
```

Two of these stages (the accumulator and the LCA-cap) walk a non-selected group's ancestors. Both walks share the same starting convention — `parents[unique_groups[j]]` — and so share the same edge-case bug when `unique_groups[j]` itself lies on the lineage. The universal fix is to start both walks at `unique_groups[j]` instead. This treats "the group is itself an ancestor on the reportable lineage" as a normal walk case rather than an off-by-one one.

The bug class is _structurally invisible to non-ancestor groups_, because for sibling and descendant relationships `unique_groups[j]` is never on `predicteds` and the walk's first iteration is a harmless no-match.

## Open Questions

- Are there real-data scenarios beyond `asv_7997` where the bug matters? Specifically: how often do databases with NA-trimmed ancestor-only entries co-occur with descendant species in the same sample? An answer would inform whether to backport the fix to past evaluations of `suppress_ancestor_only_groups`.
- Should the LCA-cap walk's fix include any additional logic for the case where MULTIPLE ancestor-only groups lie on the lineage simultaneously (e.g., a "Salmoninae sp." entry plus an "Oncorhynchus sp." entry plus mykiss)? The universal fix handles each independently and credits each at its own rank — needs a multi-ancestor fixture to confirm.
- The commit message for f36cd2e should be amended (or a new commit's message should explicitly correct it). What's the project's convention here — a follow-up commit with a "fix:" prefix, or a CHANGELOG note?
