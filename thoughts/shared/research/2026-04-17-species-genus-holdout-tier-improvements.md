---
date: 2026-04-17T19:25:03+0000
researcher: Claude
git_commit: 36eed37a8f8a8028a5ec3d1549f270e48e4189ed
branch: main
repository: oxidtaxa
topic: "Algorithmic and logic improvements for species and genus holdout tiers in assignment_benchmarks"
tags: [research, holdout, benchmarks, species-tier, genus-tier, unified-holdout, assignment-benchmarks, lca-databases]
status: complete
last_updated: 2026-04-19
last_updated_by: Claude
last_updated_note: "Addendum 2 — re-baselined against ~/assignment-tool-benchmarking (current monorepo, commit 75d9d73)"
---

# Research: Species and Genus Holdout Tier Improvements (assignment_benchmarks)

**Date**: 2026-04-17
**Researcher**: Claude
**Git Commit (oxidtaxa)**: 36eed37
**Branch**: main
**Target repository**: `~/edna-explorer-data-pipelines/projects/assignment_benchmarks`

## Research Question

Continue searching for algorithmic or logic improvements; the user reports a problem with species and genus holdout tiers as set up in
`~/edna-explorer-data-pipelines/projects/assignment_benchmarks/src`.

## Summary

The unified three-tier holdout system (Normal / Haplotype / Species-holdout) is implemented across
`infrastructure/species_selector.py`, `domain/ground_truth.py`, `domain/metrics.py`, and
`application/pipeline_runner.py`. The most consequential finding is that **three of the four
holdout-aware evaluation features exist in code but are never exercised by the production pipeline**:

1. `compute_holdout_eval_ranks` (dynamic per-species eval rank) — defined but never called in production.
2. `build_species_siblings_map` (sibling-aware lenient metric) — defined but never called in production.
3. `build_species_lca_rank_map` (LCA-aware strict metric) — defined but never called in production.

Because `evaluate_accuracy_unified` receives empty defaults for all three maps, the species-holdout
tier is scored as "match genus rank exactly, abstain at species" for **every** species-holdout species
regardless of whether the genus still exists in the post-holdout reference. When a holdout removes a
species whose genus has no other representatives left in the reference, the tool is mathematically
unable to earn credit even if its family-level prediction is perfectly correct. This is the dominant
correctness issue in the species/genus holdout tiers today.

Two further systemic issues compound the problem: the holdout-eligibility filter
(`_has_congeneric_in_db`) excludes community-internal congeners, producing a biased species-holdout
sample; and `select_unified_holdout` never prevents both members of a congeneric pair from landing in
the holdout tier together, which can empty a genus and trigger the first issue.

Finally, the `holdout_lenient` and `holdout_strict` computations at `ground_truth.py:810-819` call the
same function with the same arguments and return identical `PerRankMetrics` — the "lenient" label is
misleading, and sibling-aware tolerance (which is the entire motivation for lenient evaluation at the
species-holdout tier) never takes effect.

Below: detailed findings on current state, then proposed algorithmic/logic improvements with file:line
citations, expected impact, and implementation notes.

---

## Detailed Findings

### 1. Production pipeline bypasses three holdout-aware evaluation maps

**File**: `src/assignment_benchmarks/application/pipeline_runner.py:540-552`

```python
unified_holdout_metrics = evaluate_accuracy_unified(
    truth=ground_truth,
    predictions=unified_df,
    tool_name=tool_name,
    holdout_split=prep.unified_holdout_split,
    observation_counts=prep.observation_counts,
    logger=self._log,
)
```

The call omits `species_siblings`, `species_lca_rank`, and `holdout_eval_ranks`.

Downstream effects in `domain/ground_truth.py:742-744`:
```python
_siblings = species_siblings or {}
_lca_ranks = species_lca_rank or {}
```

And at `ground_truth.py:805-807`:
```python
_raw_eval_ranks = holdout_eval_ranks or dict.fromkeys(
    holdout_split.species_holdout_species, "genus"
)
```

Consequences:
- `compute_lenient_species_level_metrics` (`metrics.py:316-386`) gets `species_siblings={}`, so every species lookup misses and falls back to `frozenset({true_sp})` at `metrics.py:349`. The metric becomes identical to strict species-exact-match for the Normal and Haplotype tiers.
- `compute_strict_species_level_metrics` (`metrics.py:394-486`) gets `species_lca_rank={}`, so every species defaults to `eval_rank = "species"` at `metrics.py:428`. Tools that correctly abstain at genus for species with genus-ambiguous reference sequences get FP instead of TP.
- The species-holdout tier forces `eval_rank = "genus"` for every species, regardless of whether a congener remains in the effective reference.

**Helpers that exist but are unused in production**:
- `build_species_siblings_map` — `src/assignment_benchmarks/infrastructure/cruxv2_loader.py:737-768`
- `build_species_lca_rank_map` — `cruxv2_loader.py:771-829`
- `compute_holdout_eval_ranks` — `src/assignment_benchmarks/infrastructure/species_selector.py:512-549`

Production grep for callers: only `tests/unit/test_metric_variants.py` invokes any of them. The production evaluation in `pipeline_runner.py:540-552` and the two notebook benchmarks (`notebooks/config_sweep_benchmark.py:2660-2667`, `notebooks/tool_comparison_benchmark.py:2996-3003`) all omit these kwargs.

### 2. `holdout_lenient` ≡ `holdout_strict` — lenient flag has no effect for species-holdout

**File**: `src/assignment_benchmarks/domain/ground_truth.py:810-819`

```python
holdout_strict = compute_strict_species_level_metrics(
    paired, effective_eval_ranks
)
# Lenient holdout: evaluate at the dynamic holdout eval rank (the
# lowest rank still represented in the reference DB after species
# removal). This handles the case where removing a species also
# empties its genus — the eval rank moves up to family/order/etc.
holdout_lenient = compute_strict_species_level_metrics(
    paired, effective_eval_ranks
)
```

Both variables bind to the output of the same function with the same arguments. Downstream, `holdout_lenient` feeds `unified_species_f1` (line 869-871) and `holdout_strict` feeds `unified_classification_f1` (line 874-877); the only divergence is `extra_fp=novel_fp` applied to the strict aggregate. No sibling-aware branch is ever applied to holdout species at species-level, even though species-holdout tier is the ONLY tier where the reference species is absent and therefore sibling tolerance matters most.

### 3. Holdout eligibility excludes community-internal congeners

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:702-721`

```python
def _has_congeneric_in_db(species, community_species, genus_to_species):
    parts = species.split()
    if len(parts) < 2:
        return False
    genus = parts[0]
    db_congeners = genus_to_species.get(genus, set())
    non_community_congeners = db_congeners - community_species - {species}
    return len(non_community_congeners) > 0
```

Called at `species_selector.py:869-873` as the filter for species-holdout candidates. The check requires a congener OUTSIDE the community. But community congeners that remain in the Normal or Haplotype tiers are still present in the reference — they would provide perfectly valid genus-level signal for the held-out species.

Empirical interaction with TAXONOMIC_TIERS community selection (where `DEFAULT_TIER_WEIGHTS` at `models.py:84-90` allocates 40% of community species to the genus tier via `_select_congeneric_within_genus`): those paired species are systematically ineligible unless their genus coincidentally has external congeners. The benchmark biases species holdout toward monotypic-in-community genera — the opposite of what a controlled holdout study wants.

### 4. No safeguard against both members of a congeneric pair landing in species holdout

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:862-888`

```python
congeneric_candidates = [
    sp for sp in all_species
    if _has_congeneric_in_db(sp, community_species_set, full_genus_to_species)
]
...
rng.shuffle(congeneric_candidates)
species_holdout_set = set(congeneric_candidates[:actual_holdout])
```

Selection is pure shuffle-and-take. If species A and B are congeneric and both have external DB congeners, both can end up in species-holdout simultaneously. This doesn't always matter (because external congeners still remain), but combined with finding #1 (production forces `eval_rank="genus"`) it becomes a correctness trap: if the external congeners happen to be filtered out elsewhere or the community happens to be isolated, the tool is asked to match a genus that doesn't exist in the reference.

### 5. `_select_within_rank` picks ONE parent group then fills from it

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:409-479` (non-genus tiers)

For family / order / class / phylum-except-phylum tiers, the algorithm:
1. Collects candidate parent groups with ≥2 unselected species (line 428-431).
2. Shuffles, then picks the parent with the largest species pool (line 438-442).
3. Tries to spread species across child-rank values within that one parent (line 446-468).
4. If still short, fills from the remaining pool of that same parent (line 471-474).

Issues:
- "Prefer largest pool" uses raw species count, not child-rank diversity. A family with 10 species in 1 genus beats a family with 5 species across 5 genera — the latter actually supports the "differ at finer ranks" intent of taxonomic-tier selection.
- The fill step (line 471-474) allows species from already-selected child groups, violating the tier's semantic promise that "family: 5" means 5 species sharing a family but differing at genus.

### 6. `normal_fraction` is declared but never consumed

**File**: `src/assignment_benchmarks/domain/models.py:873-892`

```python
@dataclass(frozen=True)
class UnifiedHoldoutConfig:
    seed: int = 42
    normal_fraction: float = 1 / 3
    haplotype_fraction: float = 1 / 3
    species_holdout_fraction: float = 1 / 3
```

`select_unified_holdout` computes `n_species_holdout` and `n_haplotype` directly from their fractions; the Normal tier is always "everything else." Setting `normal_fraction` to any value has no effect beyond the sum-to-1 validator. This is a latent source of user confusion but not a correctness bug.

### 7. `max(1, round(fraction * n))` over-allocates for small communities

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:859-860`

```python
n_species_holdout = max(1, round(n_total * config.species_holdout_fraction))
n_haplotype = max(1, round(n_total * config.haplotype_fraction))
```

For `n_total=2` with both fractions at 0.4, the result is `n_species_holdout=1` + `n_haplotype=1` = 2, leaving 0 for Normal. For `fraction=0.0`, `max(1, 0) = 1` still forces a holdout. In very small communities this distorts the designed distribution. (Note: `HoldoutConfig.__post_init__` at `models.py:776-781` rejects `holdout_fraction=0.0` or `1.0`, but `UnifiedHoldoutConfig` does not; it accepts `species_holdout_fraction=0.0` then silently forces 1 species.)

### 8. `compute_holdout_eval_ranks` walks full_db, not the effective reference

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:512-549`

```python
index = TaxonomyIndex.from_database(full_db)
...
for species in holdout_species:
    for rank in check_order:
        sharing = index.species_sharing_rank(species, rank)
        remaining = sharing - holdout_species
        if remaining:
            assigned_rank = rank
            break
```

The function is semantically correct for the current pipeline because the effective reference is `full_db - species_holdout_species` (the pipeline retains all full_db species except holdout; see `infrastructure/benchmark_shared.py:852-885` call to `write_filtered_reference` with `exclude_species=species_holdout_species`). BUT this coupling is implicit — if the pipeline ever filters reference species more aggressively (e.g., also drops rare non-community species), the eval rank would report false optimism.

### 9. `compute_strict_genus_metrics` penalizes any non-null species prediction

**File**: `src/assignment_benchmarks/domain/metrics.py:1195-1247`

```python
tp = paired.filter(has_true & has_pred & genus_match & ~has_pred_sp).height
fp = paired.filter(
    has_pred & ((has_true & ~genus_match) | (~has_true) | has_pred_sp)
).height
```

Any non-null species prediction contributes to FP regardless of genus correctness. This is the intended "strict abstention" rule, but without a sibling-aware counterpart in the holdout tier, tools that correctly assign to a sibling species (a very reasonable behavior when the true species is absent from the ref but a sibling is present with identical or near-identical sequence) are penalized as fully wrong instead of genus-right. Combined with finding #2, no metric on the species-holdout tier rewards the common tool behavior of "predict the sibling species that's actually in the reference."

### 10. Row-by-row Python iteration in strict species-level metrics

**File**: `src/assignment_benchmarks/domain/metrics.py:432-458, 537-562, 648-667, 804-831`

The strict/partial-credit/lenient species-level metrics and all ASV-level metrics iterate the paired dataframe in Python (`for row in ...iter_rows(named=True)`), which for typical benchmark sizes (10k ASVs × 100 species × 3 metric variants × 3 tiers × 10+ tools) produces a significant hot spot. Confirmed as a production path used on every run.

### 11. `genus_to_species` property recomputes on every access

**File**: `src/assignment_benchmarks/domain/models.py:500-506`

```python
@property
def genus_to_species(self) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for rec in self.taxonomy.values():
        if rec.genus and rec.species:
            result.setdefault(rec.genus, set()).add(rec.species)
    return result
```

Called once per `_has_congeneric_in_db` invocation at `species_selector.py:863`, which is then called in a list comprehension over every community species at `species_selector.py:869-873`. For a 100-species community, this is 100 full DB scans per holdout partition. Similar pattern for `species_to_accessions` at `models.py:490-497`.

---

## Proposed Algorithmic / Logic Improvements

These are ranked by the estimated change in benchmark fairness / interpretability. Items B and C address the **most likely causes** of surprising species/genus holdout scores the user is seeing today.

### Tier 1 — Correctness (must-fix to trust holdout scores)

#### P1. Wire the three holdout-aware maps into production

**Files to change**:
- `src/assignment_benchmarks/application/pipeline_runner.py:540-552`
- Optional: `notebooks/config_sweep_benchmark.py:2660-2667`, `notebooks/tool_comparison_benchmark.py:2996-3003`

Change the call to `evaluate_accuracy_unified` to build and pass all three maps. The reference DB to pass is the post-holdout effective reference (constructed from `write_unified_holdout_reference`):

```python
from assignment_benchmarks.infrastructure.cruxv2_loader import (
    build_species_siblings_map,
    build_species_lca_rank_map,
)
from assignment_benchmarks.infrastructure.species_selector import (
    compute_holdout_eval_ranks,
)

# Build off the reference DB actually fed to the tool, not full_db.
siblings = build_species_siblings_map(ref_db, logger=self._log)
lca_ranks = build_species_lca_rank_map(ref_db, logger=self._log)
holdout_eval_ranks = compute_holdout_eval_ranks(
    holdout_species=prep.unified_holdout_split.species_holdout_species,
    full_db=full_db,
    logger=self._log,
)

unified_holdout_metrics = evaluate_accuracy_unified(
    ...,
    species_siblings=siblings,
    species_lca_rank=lca_ranks,
    holdout_eval_ranks=holdout_eval_ranks,
)
```

Impact:
- Species-holdout scoring moves to the correct per-species rank (no more forced genus-match for family-only cases).
- Lenient metric on Normal and Haplotype tiers becomes genuinely lenient (sibling-aware).
- Strict metric correctly rewards tools that abstain at the LCA rank for ambiguous-sequence species.

Risk:
- All unified-holdout JSON outputs change numerically. Any baseline dashboards/plots that assume current numbers need to be re-baselined.
- Tests in `tests/unit/test_unified_holdout.py` that compare F1s to specific values may need updating.

#### P2. Add a real lenient holdout metric (sibling-aware at eval_rank)

**File**: `src/assignment_benchmarks/domain/metrics.py` (new function), `domain/ground_truth.py:810-819`

Replace the duplicated `holdout_lenient = compute_strict_species_level_metrics(...)` with a new
function that accepts both `species_eval_ranks` and `species_siblings`. TP condition: prediction
equals any sibling of the true species AT the species' eval rank, AND no finer prediction present.

Signature sketch:
```python
def compute_lenient_holdout_species_level_metrics(
    paired: pl.DataFrame,
    species_eval_ranks: dict[str, str],
    species_siblings: dict[str, frozenset[str]],
) -> PerRankMetrics: ...
```

The distinction matters specifically for species-holdout tier: predicting a sibling species (which is
typically what tools do when the true species is absent but a congener is in ref) should count as
lenient-TP even though strict rules would call it over-classification. This was the entire reason the
metric was named "lenient" originally.

#### P3. Fix `_has_congeneric_in_db` to accept community congeners that remain in the reference

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:702-721`

A species A is species-holdout-eligible iff SOME congener of A will be in the reference *after*
holdout is applied. Since the holdout decision itself depends on what's eligible, this is a fixpoint.
A simple two-pass algorithm:

1. Build `eligible = {sp | genus_has_any_congener_in_full_db_or_community(sp)}`.
2. Randomly take up to `n_species_holdout` from `eligible`.
3. For each selected species, verify at least one congener survives (i.e., is NOT also selected). If a
   pair A,B are both chosen and A is the only B-congener (no external DB congener), push B back to
   Normal and pull in a replacement from `eligible`.

Or, equivalently, formulate as "draw species while maintaining the invariant that every selected
species' genus still has ≥1 unselected representative in full_db":

```python
rng.shuffle(eligible)
species_holdout_set: set[str] = set()
for sp in eligible:
    if len(species_holdout_set) >= n_species_holdout:
        break
    genus = sp.split()[0]
    ref_congeners_remaining = (
        full_genus_to_species.get(genus, set())
        - species_holdout_set    # not held out
        - {sp}                   # not the candidate itself
    )
    if ref_congeners_remaining:
        species_holdout_set.add(sp)
```

Impact: eliminates the systematic bias against genus-tier community species (40% of TAXONOMIC_TIERS
communities) AND removes the "both-sides-of-pair-in-holdout" pathology.

#### P4. Coordinate species-holdout selection with TAXONOMIC_TIERS community tier assignments

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:817-943`

When the community was built via `TAXONOMIC_TIERS`, `SelectionMetadata.tier_assignments` maps each
community species to the rank it was placed at. A coordinated holdout selection can:
- Prefer species from the community's **genus** tier for species-holdout (they always have a
  congeneric pair available in the community's genus tier).
- Ensure that if A is species-holdout from the genus tier, its pair partner B is in Normal or
  Haplotype (not also species-holdout).

Interface change: accept an optional `community_tier_assignments: dict[str, str]` on
`select_unified_holdout`, plumbed through from `prepare_community_command.py`.

Impact: species-holdout tier becomes a meaningful stress test of genus-level signal preservation,
rather than a random sample potentially biased toward phylum-tier community species.

### Tier 2 — Semantic correctness (affects what tier labels mean)

#### P5. Make `compute_holdout_eval_ranks` operate on the effective reference DB

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:512-549`, `application/pipeline_runner.py:540-552`

Currently the function walks `full_db` and subtracts `holdout_species`. That's coincidentally correct
today because `write_unified_holdout_reference` only drops `species_holdout_species`. Make the
coupling explicit: pass `reference_db` (the actual filtered ref) directly.

```python
def compute_holdout_eval_ranks(
    holdout_species: frozenset[str],
    reference_db: CruxV2Database,
    logger: Logger | None = None,
) -> dict[str, str]:
    index = TaxonomyIndex.from_database(reference_db)
    ...
    for species in holdout_species:
        for rank in check_order:
            sharing = index.species_sharing_rank(species, rank)
            if sharing:  # reference_db already excludes holdout species
                assigned_rank = rank
                break
```

This makes the contract robust to future changes in what the reference excludes.

#### P6. Enforce child-rank diversity in `_select_within_rank`

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:409-479`

Two changes:

(a) Select parent groups by child-rank diversity, not raw species count:
```python
def _child_diversity(parent_species: set[str], child_rank: str) -> int:
    return len({
        index.species_to_path[sp][TAXONOMIC_RANKS.index(child_rank)]
        for sp in parent_species
        if sp in index.species_to_path
    })

candidates.sort(key=lambda kv: _child_diversity(kv[1], child_rank), reverse=True)
```

(b) When unable to fill `count` with distinct child-rank values in the chosen parent, walk to the
NEXT parent group rather than refilling from the same parent's pool. This preserves the "differ at
finer ranks" guarantee.

#### P7. Drop the `max(1, round(...))` floor for small communities

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:859-860`

```python
# Before:
n_species_holdout = max(1, round(n_total * config.species_holdout_fraction))
n_haplotype = max(1, round(n_total * config.haplotype_fraction))

# After:
n_species_holdout = round(n_total * config.species_holdout_fraction)
n_haplotype = round(n_total * config.haplotype_fraction)
if n_species_holdout + n_haplotype > n_total:
    # Proportionally trim
    excess = n_species_holdout + n_haplotype - n_total
    # trim haplotype first, then species holdout
    ...
```

Rationale: if the user sets `species_holdout_fraction=0.0`, they should get zero, not one. The
`max(1, ...)` exists to avoid empty tiers for very small communities but silently contradicts the
fractions the user supplied.

#### P8. Remove or implement `normal_fraction`

**File**: `src/assignment_benchmarks/domain/models.py:873-892`, `infrastructure/species_selector.py:858-904`

Either delete the field (it's currently only enforcing a sum-to-1 constraint with no effect on
behavior) or actually honor it. Implementing it would mean: if `normal_fraction=0.1`, force extra
species into Haplotype/species-holdout tiers rather than letting Normal absorb the remainder. A
halfway option: keep the field as documentation and add an assertion in
`select_unified_holdout` that the observed normal fraction matches within tolerance.

Minimal fix: remove the field entirely. This is a frozen dataclass with only one production call site
(`cli/prepare_community_command.py:188-192`), which uses defaults, so removal is safe.

### Tier 3 — Performance (large benchmarks)

#### P9. Vectorize the strict / partial-credit / lenient species-level metrics

**File**: `src/assignment_benchmarks/domain/metrics.py:394-486`, `:494-589`, `:622-690`, `:693-774`, `:777-853`

Replace the `for row in sp_rows.iter_rows(named=True)` patterns with polars group_by + agg. For
strict species-level (line 432-458), the vectorized kernel is:

```python
# Per-ASV compute rank_match at the species' eval_rank
eval_rank_map = species_lca_rank  # dict

# Add a per-row eval_rank column
paired_with_er = paired.with_columns(
    pl.col("true_species").str.to_lowercase()
        .replace_strict(eval_rank_map, default="species")
        .alias("_eval_rank")
)

# For each of the 7 possible eval_ranks, compute finer_null + rank_match
# Then aggregate TP/FP/FN per true_species
```

A cleaner approach: compute `deepest_correct_rank_per_asv` as a single integer column (via chained
when/then/otherwise), then group by true species and check if any ASV hits eval_rank exactly.

Expected speedup: 50-200× on typical benchmark sizes (10k ASVs × 100 species).

#### P10. Cache `genus_to_species` and `species_to_accessions` properties

**File**: `src/assignment_benchmarks/domain/models.py:480-506`

Convert these from `@property` to `@cached_property` (works with frozen dataclasses via
`object.__setattr__` or by setting `eq=False` on an auxiliary cache dict). Or precompute once at
`CruxV2Database` construction time. For a single `select_unified_holdout` call on a 100-species
community, this avoids ~100 full-DB scans during `_has_congeneric_in_db` invocations.

#### P11. Reuse `TaxonomyIndex` across selection functions

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:180, 281, 525`

Each of `select_congeneric`, `select_taxonomic_tiers`, `compute_holdout_eval_ranks` calls
`TaxonomyIndex.from_database(db)` from scratch. A single pipeline run can construct it 2-3 times.
Either:
- Memoize on `CruxV2Database` identity.
- Pass an optional `index: TaxonomyIndex | None = None` kwarg to each function.

### Tier 4 — Naming / documentation clarity

#### P12. Rename `compute_strict_genus_metrics` → `compute_species_holdout_abstention_metrics`

**File**: `src/assignment_benchmarks/domain/metrics.py:1195-1247`

The current name implies "strict variant of genus rank metric" but it's specifically a species-holdout
tier metric encoding an abstention rule (predict genus, leave species null). Additionally the `rank`
string set to `"species_holdout_strict"` conflicts semantically with `"strict_species_level"` (which
is the real strict metric for normal/haplotype tiers). Renaming this function and its `rank` field
would make downstream JSON outputs self-documenting.

#### P13. Clarify the two `species_holdout_genus_*` metrics

**File**: `src/assignment_benchmarks/domain/ground_truth.py:949-958`

The per-tier breakdown exports:
- `species_holdout_genus_f1` — from `compute_strict_genus_species_level_metrics` (species-level, one vote per unique species)
- `species_classification_fraction` — from `compute_strict_genus_metrics` (ASV-level)

Two "strict genus" metrics on the same tier but different aggregation granularities. Adding `_asv` /
`_species` suffixes to the JSON keys would eliminate the need for users to consult the source.

---

## Impact and Prioritization Table

| ID | Improvement | File(s) | Type | Estimated impact |
|----|-------------|---------|------|------------------|
| P1 | Wire siblings / LCA / eval_ranks maps into pipeline_runner | `pipeline_runner.py:540-552` | Correctness | **Dominant** — reshapes all holdout scores |
| P2 | Real sibling-aware lenient holdout metric | `metrics.py`, `ground_truth.py:810-819` | Correctness | High — differentiates lenient vs strict F1 |
| P3 | Fix `_has_congeneric_in_db` to count community congeners | `species_selector.py:702-721` | Correctness | High — removes sampling bias |
| P4 | Coordinate holdout with TAXONOMIC_TIERS tier assignments | `species_selector.py:817-943` | Correctness | Medium-High — only if community uses tiers |
| P5 | Pass reference DB explicitly to `compute_holdout_eval_ranks` | `species_selector.py:512-549` | Robustness | Low today, prevents future breakage |
| P6 | Child-rank diversity in `_select_within_rank` | `species_selector.py:409-479` | Correctness | Medium — makes family/order tiers meaningful |
| P7 | Drop `max(1, round(...))` floor | `species_selector.py:859-860` | Correctness | Low — only affects tiny communities |
| P8 | Remove or implement `normal_fraction` | `models.py:873-892` | Clarity | Low |
| P9 | Vectorize strict/partial/lenient metrics | `metrics.py` (multiple fns) | Performance | 50-200× speedup |
| P10 | Cache genus_to_species / species_to_accessions | `models.py:480-506` | Performance | Medium |
| P11 | Reuse `TaxonomyIndex` across calls | `species_selector.py` | Performance | Low-Medium |
| P12 | Rename `compute_strict_genus_metrics` | `metrics.py:1195` | Clarity | Low |
| P13 | Differentiate two `species_holdout_genus_*` JSON keys | `ground_truth.py:949-958` | Clarity | Low |

---

## Recommended Rollout Order

1. **P1** — single biggest fix. Two-line change in `pipeline_runner.py` plus three helper calls. Run existing tests; all pass (they don't assert specific F1 values on the unused-map path).
2. **P2** — restores the semantic difference between `unified_species_f1` (lenient) and `unified_classification_f1` (strict) for the species-holdout tier.
3. **P3 + P4** — fix the sampling bias. These jointly determine WHICH species are in the species-holdout tier; without them, tier composition is biased.
4. **P9** — once semantics are correct, vectorize the evaluation hot loop.
5. **P5, P6, P7, P8, P10, P11, P12, P13** — polish pass.

---

## Code References

### Production call sites that omit holdout-aware maps
- `src/assignment_benchmarks/application/pipeline_runner.py:540-552`
- `notebooks/config_sweep_benchmark.py:2660-2667`
- `notebooks/tool_comparison_benchmark.py:2996-3003`

### Unused-in-production helpers
- `src/assignment_benchmarks/infrastructure/species_selector.py:512-549` — `compute_holdout_eval_ranks`
- `src/assignment_benchmarks/infrastructure/cruxv2_loader.py:737-768` — `build_species_siblings_map`
- `src/assignment_benchmarks/infrastructure/cruxv2_loader.py:771-829` — `build_species_lca_rank_map`

### Core selection logic
- `src/assignment_benchmarks/infrastructure/species_selector.py:266-367` — `select_taxonomic_tiers`
- `src/assignment_benchmarks/infrastructure/species_selector.py:370-406` — `_select_congeneric_within_genus`
- `src/assignment_benchmarks/infrastructure/species_selector.py:409-479` — `_select_within_rank`
- `src/assignment_benchmarks/infrastructure/species_selector.py:482-504` — `_select_cross_rank`
- `src/assignment_benchmarks/infrastructure/species_selector.py:633-721` — `select_holdout_species` + `_has_congeneric_in_db`
- `src/assignment_benchmarks/infrastructure/species_selector.py:817-943` — `select_unified_holdout`

### Evaluation flow
- `src/assignment_benchmarks/domain/ground_truth.py:674-969` — `evaluate_accuracy_unified`
- `src/assignment_benchmarks/domain/ground_truth.py:742-744` — default empty maps
- `src/assignment_benchmarks/domain/ground_truth.py:798-825` — species-holdout branch
- `src/assignment_benchmarks/domain/ground_truth.py:805-807` — genus-fallback for eval ranks
- `src/assignment_benchmarks/domain/ground_truth.py:810-819` — holdout_strict ≡ holdout_lenient

### Metric implementations
- `src/assignment_benchmarks/domain/metrics.py:316-386` — `compute_lenient_species_level_metrics`
- `src/assignment_benchmarks/domain/metrics.py:394-486` — `compute_strict_species_level_metrics`
- `src/assignment_benchmarks/domain/metrics.py:494-589` — `compute_partial_credit_species_level_metrics`
- `src/assignment_benchmarks/domain/metrics.py:1195-1247` — `compute_strict_genus_metrics` (species-holdout abstention)
- `src/assignment_benchmarks/domain/metrics.py:219-308` — `compute_strict_genus_species_level_metrics`

### Models / configs
- `src/assignment_benchmarks/domain/models.py:861-892` — `UnifiedHoldoutConfig`
- `src/assignment_benchmarks/domain/models.py:895-948` — `UnifiedHoldoutSplit`
- `src/assignment_benchmarks/domain/models.py:94-205` — `TaxonomicTierConfig` + `from_richness`
- `src/assignment_benchmarks/domain/models.py:480-506` — `CruxV2Database` derived properties

### Orchestration
- `src/assignment_benchmarks/application/pipeline_runner.py:300-351` — community design + holdout split
- `src/assignment_benchmarks/application/pipeline_runner.py:353-370` — filtered reference write
- `src/assignment_benchmarks/infrastructure/benchmark_shared.py:852-910` — `write_unified_holdout_reference` + `compute_tronko_remove_accessions`
- `src/assignment_benchmarks/cli/prepare_community_command.py:133-207` — CLI surface (only `UnifiedHoldoutConfig` is exposed)

---

## Historical Context (from thoughts/)

- `thoughts/shared/research/2026-04-17-abstention-path-output-handling.md` — documents that oxidtaxa's TSV writer collapses two distinct unclassified paths into identical output. Benchmarks reading the TSV cannot distinguish "classifier abstained due to tie" from "classifier had no signal"; the in-memory `Vec<ClassificationResult>` preserves that distinction. Relevant to holdout-tier evaluation because tied-species collapse at the species-holdout tier (congeners in ref after species removal) is the exact failure mode this benchmark is designed to detect.
- `thoughts/shared/research/2026-04-14-confidence-scores-for-unclassified-asvs.md` — recommends running `classify(threshold=0)` for benchmark notebooks so that tier evaluation can differentiate "low-confidence prediction" from "no prediction." This is the complementary workaround to P1/P2: once the maps are wired in, the `novel_fp` accounting at `ground_truth.py:858-864` stops being the primary signal.
- `thoughts/shared/research/2026-04-15-new-parameter-audit.md` — identifies `leave_one_out` as a no-op before commit `3d6cb91`. If any existing holdout benchmark numbers were produced with `leave_one_out=true` on an earlier commit, they are equivalent to `leave_one_out=false` runs and the observed "no LOO sensitivity" in holdout scores is explained by this bug, not by the holdout logic.
- `thoughts/shared/research/2026-04-08-tied-species-reporting-at-truncated-rank.md` — confidence propagation drives genus above threshold when species are tied. This is exactly the mechanism the species-holdout tier exercises; the `alternatives` field + `lca_cap` work is designed to surface it. The holdout evaluation does not currently consult `alternatives`, so tools that correctly list the held-out species in `alternatives` but pick a sibling as the top species are scored the same as tools that hallucinate an unrelated species.

## Related Research

- `thoughts/shared/research/2026-04-13-algorithmic-improvements.md` — oxidtaxa-internal improvements (k-mer representation, classification hot loops). Complementary to this doc which focuses on benchmark harness logic, not classifier internals.
- `thoughts/shared/research/2026-04-17-correlation-aware-training-bottlenecks.md` — training-side performance work.

## Open Questions

1. **P1 downstream compatibility**: Does any existing results dashboard or Optuna study baseline the current (unused-maps) unified-holdout numbers? Turning on the maps shifts every metric — a re-baselining PR might need to accompany the code change.
2. **P3 selection determinism**: The fixpoint-style holdout selection in P3 changes the set of species chosen even for the same seed. Is it acceptable to break seed-compatibility of existing benchmark runs?
3. **P4 cross-cutting interface**: TAXONOMIC_TIERS selection is not currently exposed via CLI (only RANDOM is wired in `cli/_adapters.py:40-62`). P4 would need either CLI plumbing for the tier strategy or a notebook-only path. Is CLI exposure in scope?
4. **P2 metric semantics**: For a species with no sibling in the reference after holdout (genus is empty), should lenient-holdout default to strict at the higher eval rank (family/order) with no sibling tolerance, or should sibling tolerance apply at that higher rank too (i.e., accept any predicted species whose family matches)?
5. **P6 interaction with `from_richness`**: The `TaxonomicTierConfig.from_richness` at `models.py:121-205` allocates counts per tier assuming the "one parent group per tier" semantics. If P6 changes the semantics to spread across multiple parents, does the default weight split still make sense?

---

## Addendum 1 — rcrux-py LCA-built reference databases change the picture

**Context**: The production reference databases built by rcrux-py
(`~/rcrux-py/databases/<marker>/megablast-ws28/`) ship in two flavors per marker:

- `*_species.fasta` / `*_species_taxonomy.txt` — each accession mapped to its source species; the
  taxonomy file has more lines than the FASTA has sequences because duplicate sequences can receive
  multiple species labels (same accession → multiple taxonomy rows).
- `*_lca.fasta` / `*_lca_taxonomy.txt` — each accession gets exactly one taxonomy row; when the
  sequence is shared across multiple species or lineages, deeper ranks are collapsed to the string
  `"NA"`. The user confirmed **production uses the `_lca` version**.

Empirical check on `vert12S/megablast-ws28/12SV5_lca_taxonomy.txt` (56192 entries):
- Every row has exactly 7 semicolon-delimited ranks (verified via `awk -F';' '{print NF}' | uniq -c`).
- 5062 / 56192 (~9 %) rows have `species == "NA"` — LCA-truncated to genus or shallower.
- 433 rows have BOTH `genus == "NA"` AND `species == "NA"` — LCA-truncated at family or shallower.
- Sample shallow entries:
  `gi|126032549|gb|AC194563.3|  Eukaryota;Craniata;Mammalia;Catarrhini;Homininae;NA;NA`

**Impact on the original findings**

### What the LCA DB removes from scope

- **P1 siblings map — drop from production scope.** `build_species_siblings_map`
  (`cruxv2_loader.py:737-768`) collapses species that share identical sequences. In an LCA-built
  DB, the sequence-level dedup has already happened upstream; `build_sequence_species_map` at
  `cruxv2_loader.py:721-734` would find at most 1 species per sequence (plus `"NA"` as a pseudo
  species). The returned dict is effectively always empty on production inputs. Wiring it into
  `pipeline_runner.py` has no effect for Normal or Haplotype tiers.

- **P1 LCA rank map — partially redundant.** `build_species_lca_rank_map`
  (`cruxv2_loader.py:771-829`) computes the deepest unambiguous rank per species by walking shared
  sequences. In an LCA DB, this information is already encoded directly in the taxonomy: an entry
  with `species="NA"` IS its own LCA statement, and the deepest non-NA rank on any accession
  carrying that species label is the LCA. A cheaper and more correct computation reads the ranks
  directly rather than re-deriving from sequence sharing.

- **P2 sibling-aware lenient holdout metric — drop.** Since the LCA DB has no shared-sequence
  siblings, a sibling-aware lenient metric has nothing to tolerate. The
  `holdout_lenient ≡ holdout_strict` duplication at `ground_truth.py:810-819` remains a code smell
  (the two variables should not point to the same computation) but the practical impact on
  benchmark fairness is gone.

### New findings the LCA DB exposes

These are correctness issues that only manifest on LCA-formatted references. They are **not**
covered by the original research body and supersede the affected items above.

#### N1. `"NA"` is silently treated as a valid species / genus name

**File**: `src/assignment_benchmarks/domain/models.py:441-450, 485-506`

```python
@property
def species(self) -> str:
    ranks = self.ranks
    return ranks[-1] if ranks else ""        # returns "NA" for LCA-truncated rows

@property
def species_set(self) -> set[str]:
    return {rec.species for rec in self.taxonomy.values() if rec.species}   # "NA" is truthy -> kept

@property
def genus_to_species(self) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for rec in self.taxonomy.values():
        if rec.genus and rec.species:         # "NA" passes both truthy checks
            result.setdefault(rec.genus, set()).add(rec.species)
    return result
```

Consequences:
- `species_set` contains `"NA"` as if it were a species. If a community designer randomly samples
  from `species_set`, `"NA"` can be chosen as a community species, pulling in all ~5000 LCA-truncated
  accessions as if they were one organism.
- `genus_to_species` builds an `"NA"` → `{"NA"}` bucket. The 433 fully-truncated entries collapse
  into one genus; the ~4600 genus-level-truncated entries each collide with the real genus on the
  rank BEFORE the NA (since `rec.genus` reads position -2, i.e. the real genus name there, not
  "NA"). So behavior is asymmetric by truncation depth.
- `species_to_accessions[""NA""]` aggregates thousands of accessions across entirely unrelated
  clades (each of which has `species=="NA"` but different higher ranks). Any code that joins on
  `species_to_accessions` produces cross-clade merges.

**Fix**: Add an NA-sentinel guard at the loader level, or at each property. Options:
```python
_NA_TOKENS = {"NA", "N/A", "", "unclassified"}

def _is_resolved(val: str) -> bool:
    return bool(val) and val.upper() not in _NA_TOKENS
```

Then:
```python
@property
def species_set(self) -> set[str]:
    return {rec.species for rec in self.taxonomy.values() if _is_resolved(rec.species)}
```

Apply consistently to `species_to_accessions`, `genus_to_species`, and the rank-extraction helpers
in `TaxonomyIndex.from_database` (`species_selector.py:59-84` — the inner loop at line 80 already
guards on `if value` but treats "NA" as a value).

#### N2. `compute_holdout_eval_ranks` counts LCA-truncated rows as siblings

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:95-111, 512-549`

`TaxonomyIndex.species_sharing_rank` returns everything in `rank_to_groups[rank][value]` minus the
query species. On an LCA DB:
- If the query is `"Homo sapiens"` and we ask "what shares genus `Homo`?", the result correctly
  excludes NA-truncated entries (those live under genus `"NA"`, not `"Homo"`).
- If we ask "what shares family `Hominidae`?", the result includes both real congeners AND any
  `Hominidae;NA;NA` accessions (since those have `family == "Hominidae"`).

The second case means `compute_holdout_eval_ranks` will report `"family"` as the eval rank even
when the only surviving "family congener" is an LCA-truncated entry with no true species identity.
Evaluation forces family-level matching, but the reference provides no signal that the tool can
distinguish from the held-out species' sequence — the tool that predicts `Hominidae;NA;NA`
(correctly identifying sequence-level ambiguity) would be expected to *match* the truth's
`Hominidae;Homo;Homo sapiens`, which is not the same shape.

**Fix (pairs with P1)**: When computing eval ranks against an LCA DB, only count resolved
(non-NA) siblings toward eligibility at each rank. Sketch:

```python
def species_sharing_rank_resolved(self, species: str, rank: str) -> set[str]:
    raw = self.species_sharing_rank(species, rank)
    # Drop siblings whose own ranks from `rank` downward are all NA
    rank_idx = TAXONOMIC_RANKS.index(rank)
    return {
        s for s in raw
        if any(_is_resolved(v)
               for v in self.species_to_path.get(s, [])[rank_idx:])
    }
```

#### N3. `_has_congeneric_in_db` matches on `"NA"` genus bucket

**File**: `src/assignment_benchmarks/infrastructure/species_selector.py:702-721`

```python
db_congeners = genus_to_species.get(genus, set())
non_community_congeners = db_congeners - community_species - {species}
return len(non_community_congeners) > 0
```

If the community ever contains a species whose genus is `"NA"` (see N1), `db_congeners` is the
entire "NA" bucket and `non_community_congeners` is likely non-empty — the species would pass
eligibility on a fake congener pool.

Conversely: a species whose genus in the community DB is fully resolved (e.g., `Homo sapiens`) but
whose sequence-sharing is encoded only via LCA-truncated accessions in the reference will **not**
be detected as congeneric-eligible, because `genus_to_species["Homo"]` in the LCA DB contains only
`{"Homo sapiens"}` (singleton — all other Homo-sequence accessions were collapsed to NA).

**Fix**: Compute congeneric eligibility by walking `TaxonomyIndex.rank_to_groups["genus"]` using
the NA-filtered property from N1, AND additionally check whether any accession with the query's
genus has a sequence in the reference (even if species="NA"). This provides a 2-level check:
"does the ref have SOME representative at this genus, resolved or not?" That's the real condition
for genus-level classification to remain possible.

#### N4. Query generation for LCA-truncated "species" is undefined

**File**: `src/assignment_benchmarks/infrastructure/synthetic_data.py` (entry: `generate_benchmark_reads`)

If N1's guard is not applied at the community design layer, the community can include `species=
"NA"`. Downstream read generation uses `species_to_accessions["NA"]`, which spans the entire
LCA-truncated set — reads for one "species" end up sampled from genetically unrelated clades.
Ground truth annotations for these reads are ambiguous at the species rank and arbitrary at the
genus rank. This doesn't produce a crash but silently poisons the benchmark.

**Fix**: the guard in N1 prevents this by removing `"NA"` from `species_set`. A defensive assert
in `CommunityConfig.__post_init__` or the community designer would catch regressions.

### Revised recommendation priorities given production uses LCA DBs

| ID | Status | Revised rationale |
|----|--------|-------------------|
| P1 (eval_ranks) | **KEEP (primary fix)** | Still the dominant issue: production forces `eval_rank="genus"` for every held-out species regardless of what the LCA ref actually represents. |
| P1 (siblings, lca_rank maps) | **DROP** | Moot on LCA DB — no sibling sequences, LCA info already in taxonomy. |
| P2 (sibling-aware lenient holdout) | **DROP** | No siblings to tolerate. Collapse `holdout_lenient` to a single variable pointing at `holdout_strict` to remove the misleading code. |
| P3 (`_has_congeneric_in_db`) | **MODIFY + KEEP** | Still important, but per N3 the fix needs to also handle LCA-truncated accessions (check "any rep at genus" rather than "any species at genus"). |
| P4 (coordinate with TAXONOMIC_TIERS) | **KEEP** | Unchanged. |
| **N1 (NA-sentinel guards)** | **NEW — HIGH PRIORITY** | Currently `"NA"` flows through as a real species/genus name. Add `_is_resolved` guards across `models.py` properties. |
| **N2 (eval_ranks NA handling)** | **NEW — COUPLED WITH P1** | When wiring `compute_holdout_eval_ranks`, additionally filter siblings by "has a resolved rank-or-deeper entry". |
| **N3 (congeneric on LCA)** | **NEW — COUPLED WITH P3** | Extend congeneric check to count LCA-truncated reference accessions as genus-level signal. |
| **N4 (query for NA species)** | **NEW — DEFENSIVE** | Assert `"NA"` ∉ community species at designer boundary. |

### Revised rollout order

1. **N1** — one-file change (`domain/models.py`) adding `_is_resolved` guards and a regression test
   asserting `"NA"` ∉ `species_set` and no LCA-truncated accessions appear in
   `genus_to_species["NA"]` or `species_to_accessions`. Unlocks everything else.
2. **P1 (eval_ranks only)** + **N2** — wire `compute_holdout_eval_ranks` into `pipeline_runner.py`,
   making it aware of resolved-vs-NA siblings. Drop the siblings and lca_rank kwargs; they're
   no-ops on LCA DBs.
3. **P3 + N3** — rewrite `_has_congeneric_in_db` to operate on the LCA-aware congener pool.
4. **P4** — coordinate with TAXONOMIC_TIERS community assignments.
5. **P2 (cleanup)** — remove the duplicate `holdout_lenient` variable so downstream readers don't
   expect a lenient branch that doesn't exist.
6. **P9 / P10 / P11** — performance pass, unchanged.
7. **P5, P6, P7, P8, P12, P13** — polish, unchanged.

### Updated open question

**Q6. Is the `*_species` flavor ever consumed by the benchmark?** The user confirmed production
uses `*_lca`. If any legacy runs or downstream consumers load the `*_species` flavor (which does
have shared sequences and would benefit from the siblings / lca_rank maps), the recommendations
split into two paths. Confirming exclusivity would let us delete `build_species_siblings_map` and
`build_species_lca_rank_map` outright rather than leaving them as dead code.

---

## Addendum 2 — Re-baseline against `~/assignment-tool-benchmarking` (current monorepo)

The research body above (and Addendum 1) was written against
`~/edna-explorer-data-pipelines/projects/assignment_benchmarks`, which is stale. The live monorepo
is `~/assignment-tool-benchmarking` (commit `75d9d73`, branch `assignment-tool-benchmarking`). The
assignment_benchmarks project has evolved substantially. Below: which findings still apply, which
were already fixed, and what new structure exists.

### 2.1 Structural changes

**Four-tier unified holdout (was three)**

`UnifiedHoldoutConfig` at `domain/models.py:934-970` now has four fractions defaulting to 0.25
each: `normal_fraction`, `haplotype_fraction`, `species_holdout_fraction`, **`genus_holdout_fraction`**
(new). `UnifiedHoldoutSplit` at `models.py:973-1041` exposes four species frozensets plus
`genus_holdout_genera: frozenset[str]`. `from_dict` at line 1037-1038 is backward-compatible with
3-tier JSON via `data.get(..., [])`.

**Genus holdout is a new tier semantically distinct from species holdout**

At `species_selector.py:1069-1116` (Step 2 of `select_unified_holdout`):
- Picks genera where ALL community members of that genus are still in the remaining pool.
- Requires **confamilial** genera in full DB via `_has_confamilial_in_db` (lines 937-953).
- Removes ALL community species of the genus at once (not one species per genus).
- Reference eval rank defaults to `"family"` (`ground_truth.py:828-831`), not genus.

So "genus holdout tier" ≠ "species-holdout-evaluated-at-genus." The former removes entire genera;
the latter is the same old species-holdout with genus-rank evaluation.

**Haplotype is now assigned FIRST** (commit `2a6959a65`)

`species_selector.py:1033-1067`. With a `tronko_accessions` guard at lines 1034-1049 and
`min_multi_accession` swap at 830-877 ensuring multi-accession species exist.

**Mixed-depth augmentation (new feature)**

`MixedDepthConfig` at `models.py:238-282` with `genus_only_fraction`, `family_only_fraction`,
`normal_fraction`, `in_community_fraction`. `_augment_with_mixed_depth` at
`species_selector.py:197-266` uses `full_db.genus_only_accessions` / `family_only_accessions` pools
(entries with `genus` present but `species` empty/NA). Pseudo-labels `[genus]X` / `[family]Y` become
first-class community taxa. Integrated via `select_taxa` (lines 702-720, 880-899). Mixed-depth
pseudo-labels are emitted into the Normal tier inside `select_unified_holdout` at lines 1194-1208.

**NA handling at `TaxonomyRecord` is fixed**

`domain/models.py:486-527`:
```python
@property
def species(self) -> str:
    ranks = self.ranks
    return "" if len(ranks) < 7 or ranks[6].upper() == "NA" else ranks[6]
# same pattern for .family (507) and .genus (516)
```

Upstream of that, `CruxV2Database.species_set`, `species_to_accessions`, `genus_to_species`,
`family_to_genus` all guard via the (now-NA-safe) accessor methods. **N1 (Addendum 1) is already
fixed at the DB aggregation level.**

**eval_data maps ARE wired — just not via `pipeline_runner.py`**

Construction lives in `infrastructure/benchmark_setup.py:515-524, 568-573, 618-624`, forwarded via
`infrastructure/benchmark/metrics_attachment.py:82-94` into `evaluate_accuracy_unified`. The direct
`pipeline_runner.py:545-552` call path (used by the CLI `run-assignment` command) still omits them.
**Two distinct evaluation entry points exist in the same codebase** — benchmark notebooks go
through `benchmark_setup` and get the full map wiring; CLI goes through `pipeline_runner` and
doesn't. Previous P1 finding ~50% obsolete: fixed for notebook/benchmark runs, still broken for CLI
runs.

### 2.2 Status of P3–P7 in the current code

| Item | Status | Notes |
|------|--------|-------|
| **P3** (`_has_congeneric_in_db`) | **Still applies as described** | `species_selector.py:917-934`, unchanged logic. Now also has a twin: `_has_confamilial_in_db` (lines 937-953) used by the new genus-holdout tier — same over-filter pattern at the family level. Both need the same fix. |
| **P4** (coordinate with TAXONOMIC_TIERS) | **Still applies** | `select_unified_holdout` does not consume `SelectionMetadata.tier_assignments`. Additionally: `prepare_community` at `pipeline_runner.py:336-345` uses `community_species = set(taxon_abundances.keys())`, flattening any tier annotations before the holdout split sees them. |
| **P5** (pass `reference_db` explicitly) | **Still applies** | `compute_holdout_eval_ranks` signature at `species_selector.py:620-657` still `(holdout_species, full_db, logger)`. No change. |
| **P6** (single-parent selection in `_select_within_rank`) | **Still applies** | `species_selector.py:517-587`. Same shuffle-then-pick-largest logic at 554-584. Unchanged. |
| **P7** (`max(1, round(...))` floor) | **Still applies** | `species_selector.py:1012-1014`. Banker's rounding was fixed in commit `a9fc09d84` but the `max(1, ...)` floor is distinct and untouched. |

### 2.3 New findings surfaced by the 4-tier structure

#### N5. `_has_confamilial_in_db` inherits the P3 over-filter

**File**: `species_selector.py:937-953`

Signature: `(species, community_species, family_to_genus)`. Same pattern as `_has_congeneric_in_db`:
requires a confamilial genus OUTSIDE the community. Genus-holdout eligibility is therefore biased
against genera whose family appears only in the community.

For a TAXONOMIC_TIERS community that by design contains species from a family tier (different
genera within shared families), those families typically don't have external confamilial genera —
so community-built families are systematically ineligible for genus-holdout. Combined with P3, the
benchmark's two "holdout" tiers both skip the species the community was specifically built to test.

Fix pattern: identical to P3. Check "family has ≥1 confamilial genus surviving in the post-holdout
ref" rather than "family has ≥1 confamilial genus OUTSIDE the community."

#### N6. Mixed-depth pseudo-labels bypass holdout entirely

**File**: `species_selector.py:1194-1208`

Mixed-depth-generated species (`[genus]Foo`, `[family]Bar`) are appended directly to `normal_set`
after the 4-tier partition is computed. They never appear in Haplotype / species-holdout /
genus-holdout tiers. Whether this is intended is a product question: the user may have added
mixed-depth specifically to test classification at coarser ranks, in which case forcing them into
Normal (where query == reference) is the opposite of the intent. If intended, the behavior is
undocumented.

#### N7. `TaxonomyIndex` leaks `"NA"` at intermediate ranks

**File**: `species_selector.py:59-84`

Uses `ranks = rec.ranks` (raw `.split(";")`) for intermediate ranks. Line 80-82:
```python
for i, rank_name in enumerate(TAXONOMIC_RANKS):
    if i < len(ranks):
        value = ranks[i]
        if value:
            rank_to_groups[rank_name].setdefault(value, set()).add(species)
```
The guard is `if value:` — which is truthy for the literal string `"NA"`. So
`rank_to_groups["genus"]["NA"]`, `rank_to_groups["family"]["NA"]`, etc. accumulate species that
were LCA-truncated at that rank. `TaxonomyIndex.species_sharing_rank(species, rank)` on a species
with `ranks[i] == "NA"` will therefore return the entire "NA" bucket at that rank as siblings.

This is a narrower version of the N1 issue from Addendum 1. It affects `_select_within_rank`,
`_select_cross_rank`, and `compute_holdout_eval_ranks` (all of which walk `TaxonomyIndex`). The
`TaxonomyRecord` accessor fix doesn't help here because `TaxonomyIndex` uses `rec.ranks` directly.

Fix: change the guard to `if value and value.upper() != "NA":` at line 81.

### 2.4 Revised priorities for P3–P7

Given the new genus-holdout tier and the partially-fixed NA handling:

| Rank | Item | Scope change |
|------|------|--------------|
| 1 | **P3 + N5** | Fix both `_has_congeneric_in_db` and `_has_confamilial_in_db` together — they have the same structural issue and both drive holdout-tier sampling bias. |
| 2 | **P6** | Child-rank diversity in `_select_within_rank`. Unchanged from prior doc. |
| 3 | **P4** | Plumb `tier_assignments` through `select_unified_holdout` so species/genus holdout preferentially sample from the community's matching tier. Unchanged. |
| 4 | **N7** | One-line fix to `TaxonomyIndex.from_database` guard so "NA" doesn't accumulate as a sibling pool. High-leverage (affects P3/P5/P6 all at once). |
| 5 | **P5** | Signature change to `compute_holdout_eval_ranks`. Unchanged. |
| 6 | **P7** | Drop `max(1, ...)` floor. Unchanged. |

### 2.5 Open questions (revised)

- **Q7**: Is mixed-depth augmentation intended to land in Normal tier only? Or should `[genus]Foo` pseudo-labels get their own holdout treatment (they already encode a "genus-only" ground truth)? `species_selector.py:1194-1208` currently routes them to Normal unconditionally.
- **Q8**: Should the CLI `run-assignment` path (`pipeline_runner.py:545-552`) be unified with the benchmark path's eval_data wiring, or are they intentionally different? If CLI runs are expected to produce comparable numbers to notebook runs, the CLI path needs the same `species_siblings`/`species_lca_rank`/`holdout_eval_ranks` construction.
- **Q9**: With a 4-tier structure and default 0.25/0.25/0.25/0.25 fractions, a community of size 20 gets 5 per tier. P7's `max(1, ...)` floor has less marginal impact at this scale than under the old 3-tier 0.33 split, but still matters at richness ≤ 12.
