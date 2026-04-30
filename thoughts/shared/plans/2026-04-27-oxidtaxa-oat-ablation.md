---
date: 2026-04-27
author: Ryan Martin
git_commit: e19df516aa064a8eb37d956a58362838f29da63a
branch: main
repository: oxidtaxa
topic: "One-at-a-time (OAT) ablation sweep for oxidtaxa-added parameters across all markers"
tags: [plan, ablation, oxidtaxa, classifier_benchmark, seed_pool]
status: draft
last_updated: 2026-04-27
last_updated_by: Ryan Martin
based_on_research: thoughts/shared/research/2026-04-27-oxidtaxa-vs-idtaxa-ablation-surface.md
---

# Oxidtaxa One-at-a-Time (OAT) Parameter Ablation Plan

## Overview

A standalone harness — `notebooks/oxidtaxa_oat_ablation.py` — that sweeps each oxidtaxa-added parameter independently against an IDTAXA-default backdrop, using SEED_POOL communities across every marker that has seed amplicons, and reports truth-depth and pred-depth ASV-based F1 metrics. Each axis run differs from its baseline by exactly one parameter, so the F1 delta is attributable to that parameter alone.

## Current State Analysis

The existing harness at `notebooks/classifier_benchmark.py` runs a 480-config cartesian sweep that confounds parameter effects:

- `_OXIDTAXA_FIXED` (`classifier_benchmark.py:821-828`) pins `bootstraps=50` and `min_descend=0.95` — **neither is the IDTAXA-canonical value** (100 and 0.98 respectively).
- The cartesian generator (`_define_oxidtaxa_cartesian_sweep`, `911-992`) jointly varies threshold × sample_exponent × k × correlation_aware × tie_margin × confidence_uses_descent_margin × sibling_aware_leaf, so single-parameter contributions are not identifiable.
- All the underlying machinery we need is already factored: `_build_communities` (`148-160`), `select_unified_holdout` (`3195-3202`), `build_holdout_databases` (`3447-3476`), `build_oxidtaxa_variant_dbs` (`995-1079`), executor invocation, `_attach_holdout_metrics` (`3653-3687`), and `SweepResult` (`202-356`).
- Only `16s_bacteria` lacks `seed_amplicons_dir` (`benchmark_shared.py:354-419`); all six other presets (`vert12s`, `16smamm`, `12s_mifish`, `18s_euk`, `its1_fungi`, `its2_plants`) have it.
- `build_oxidtaxa_variant_dbs` already covers every training axis we need (k, record_kmers_fraction, seed_pattern, training_threshold, descendant_weighting, use_idf_in_training, leave_one_out, correlation_aware_features) at `1015-1078`.

## Desired End State

A new script `notebooks/oxidtaxa_oat_ablation.py` that, when run, produces:

- For each marker (6 markers with seed_amplicons): N seed_pool communities × {clean, realistic, degraded} baselines × (1 baseline + Σ test_values per axis + 1 R-IDTAXA reference) runs.
- The **R IDTAXA reference cell** runs the actual `idtaxa` tool (R DECIPHER via the existing `tools/idtaxa` executor) at canonical defaults — this is the yardstick. The point of the ablation is "does each oxidtaxa-added param push us above R IDTAXA on accuracy AND runtime?"
- **Runtime** is captured for every cell (both R IDTAXA and every oxidtaxa variant) and reported alongside accuracy.
- Per-cell predictions written to `unified.parquet` under `DATA_BASE / "oxidtaxa_oat" / "experiments" / <axis> / <value> / <community> / <baseline>/`.
- A consolidated `oat_results.json` (one `SweepResult` row per cell) under `RESULTS_BASE / "oxidtaxa_oat" / <marker>/`.
- A markdown report `oat_summary.md` per marker, with one table per axis: rows = test values, columns = (mean ± SE) of `unified_asv_truth_depth_f1`, `unified_asv_pred_depth_phylum_f1`, `asv_truth_depth_tier_hmean`, `asv_pred_depth_phylum_tier_hmean`, plus per-tier truth-depth F1 (normal / haplotype / species_holdout / genus_holdout). Baseline row first, deltas relative to baseline.
- A cross-marker rollup `oat_rollup.md` aggregating axis effects across markers (mean Δ-F1 with sign and across-marker dispersion).

### Verification:
- Within a single (marker, replicate, baseline) cell, the baseline row of every axis is byte-identical (same kwargs → same `unified.parquet` reused via deduplication).
- For each axis, only the named parameter differs between baseline and test rows in the persisted kwargs JSON sidecar.
- ASV truth-depth F1 columns are populated and non-null for all rows.

### Key Discoveries from research doc:
- 11 oxidtaxa-added algorithmic parameters with neutralizing values catalogued in research doc, summary table (lines 30-46).
- 5 are training-time (require retraining): `seed_pattern`, `descendant_weighting`, `use_idf_in_training`, `leave_one_out`, `correlation_aware_features`.
- 6 are classify-only (single trained model reused): `length_normalize`, `rank_thresholds`, `beam_width`, `tie_margin`, `confidence_uses_descent_margin`, `sibling_aware_leaf`.
- `sibling_aware_leaf` was empirically inert across 5,520 cartesian runs (research doc line 72) — included anyway for completeness; expect Δ ≈ 0.
- `_make_oxidtaxa_db_variant_key` (`classifier_benchmark.py:861-886`) and `build_oxidtaxa_variant_dbs` (`995-1079`) are reusable — we do not need to reimplement training dedup.

## What We're NOT Doing

- No cross-tool comparison **except R IDTAXA** (which is the explicit reference yardstick — the ablation is testing whether oxidtaxa-added params beat IDTAXA on accuracy and runtime).
- No multi-axis interactions (strict OAT; one parameter perturbed per run).
- No Optuna / hyperparameter search.
- No new metric definitions — reuse `SweepResult` columns from `classifier_benchmark.py:202-356`.
- No new community design — call `_build_communities_shared(use_seed_pool=True)` directly.
- No edits to `classifier_benchmark.py` (we *import* from it / from `benchmark_shared`).
- No sweep over computational/PRNG params (`processors`, `seed`, `deterministic`) — held fixed.
- No sweep over IDTAXA-original params (`threshold`, `bootstraps`, `strand`, `sample_exponent`, `min_descend`, `full_length`, `k`, `n`, `min_fraction`, `max_fraction`, `max_iterations`, `multiplier`, `max_children`, `record_kmers_fraction`, `training_threshold`) — these are the backdrop, frozen at IDTAXA-canonical defaults.
- No `16s_bacteria` (no seed_amplicons available).

## Implementation Approach

Standalone Python script that:
1. Defines the IDTAXA-default `BACKDROP_KWARGS` dict.
2. Defines `ABLATION_AXES: list[Axis]` where each `Axis` declares `(name, neutral_value, test_values, requires_retrain)`.
3. For each marker, builds N seed_pool communities once (via the existing shared builder), persists/loads `ground_truth.json`, builds per-tool unified-holdout reference DBs.
4. Pre-trains all unique oxidtaxa training-variant DBs by reusing `build_oxidtaxa_variant_dbs` on the OAT variant set.
5. Runs every (community, baseline, variant) cell via the existing in-process PyO3 executor; attaches metrics via `_attach_holdout_metrics`.
6. Saves incrementally; emits per-marker and cross-marker reports.

We keep `deterministic=False` and `processors=NUM_THREADS` for every run because (a) we are not chasing R-byte-equivalence in this study, (b) the user wants wall-clock parallelism, and (c) holding both fixed across all axes preserves OAT validity (any PRNG-mode effect is a constant offset baked into every row).

## Phase 1: Backdrop, axes, and marker discovery

### Overview
Define the immutable backdrop and the axes catalog at the top of the new script. This is the single source of truth for what "IDTAXA-default" means in this study.

### Changes Required:

#### 1. New file: `notebooks/oxidtaxa_oat_ablation.py`
**Backdrop**:
```python
# All values match Rust defaults from oxidtaxa/src/types.rs ClassifyConfig/TrainConfig
# except for processors/deterministic which we set explicitly for parallelism.
BACKDROP_KWARGS: dict[str, Any] = {
    # IDTAXA-original (frozen at canonical defaults)
    "threshold": 60.0,
    "bootstraps": 100,
    "strand": "both",
    "sample_exponent": 0.47,
    "min_descend": 0.98,            # canonical DECIPHER default (confirmed against R/IdTaxa.R)
    "full_length": 0.0,
    "k": None,                       # auto
    "record_kmers_fraction": 0.10,
    "training_threshold": 0.8,
    "seed": 42,
    # oxidtaxa-added, all set to neutralizing values
    "length_normalize": False,
    "rank_thresholds": None,
    "beam_width": 1,
    "tie_margin": 0.0,
    "confidence_uses_descent_margin": False,
    "sibling_aware_leaf": False,
    "seed_pattern": None,
    "descendant_weighting": "count",
    "use_idf_in_training": False,
    "leave_one_out": False,
    "correlation_aware_features": False,
    # computational (held fixed across all OAT cells)
    "deterministic": False,
    "threads": NUM_THREADS,          # forwarded as `processors` by executor
}
```

**Helper for rank-threshold gradients** (mirrors `oxidtaxa_hyperparameter_tuning.py:574-580`):
```python
def _rank_gradient(threshold: float, offset: float) -> list[float]:
    """Linear interpolation from Root (i=0, threshold+offset) to Species (i=6, threshold)."""
    return [threshold + offset * ((6 - i) / 6.0) for i in range(7)]
```

**Axes catalog**:
```python
@dataclass(frozen=True)
class Axis:
    name: str                         # parameter name, e.g. "beam_width"
    neutral_value: Any                # the IDTAXA-default value
    test_values: tuple[Any, ...]      # values to perturb to
    requires_retrain: bool            # True iff training-time

ABLATION_AXES: tuple[Axis, ...] = (
    # Classify-only (model reused)
    Axis("length_normalize",                False, (True,),                 False),
    Axis("rank_thresholds",                 None,  (
        # Linear Root→Species gradient, parameterized exactly like
        # oxidtaxa_hyperparameter_tuning.py:574-580:
        #   rank_thresholds[i] = threshold + offset * ((6 - i) / 6.0)  for i in 0..6
        # Species (i=6) keeps `threshold`; Root (i=0) gets `threshold + offset`.
        # Backdrop threshold=60 → three offsets bracketing the Optuna search range [1, 25].
        _rank_gradient(threshold=60.0, offset=6.0),   # gentle:   [66, 65, 64, 63, 62, 61, 60]
        _rank_gradient(threshold=60.0, offset=12.0),  # moderate: [72, 70, 68, 66, 64, 62, 60]
        _rank_gradient(threshold=60.0, offset=20.0),  # strong:   [80, 76.67, 73.33, 70, 66.67, 63.33, 60]
    ),                                                                       False),
    Axis("beam_width",                      1,     (2, 3, 5),                False),
    # tie_margin range deliberately wider than the Optuna prior's [0.0, 0.10]
    # cap (oxidtaxa_hyperparameter_tuning.py:593). 0.20 should start visibly
    # broadening LCAs; 0.40 should bracket the LCA-collapse breakdown point.
    Axis("tie_margin",                      0.0,   (0.05, 0.10, 0.20, 0.40), False),
    Axis("confidence_uses_descent_margin",  False, (True,),                  False),
    # `sibling_aware_leaf` is NOT pure OAT — see "sibling_aware_leaf × tie_margin
    # mini-grid" below. The single-axis Axis entry is replaced by a 2×3 grid
    # (sibling_aware_leaf ∈ {F,T} × tie_margin ∈ {0.0, 0.05, 0.10}) generated
    # by `_make_sibling_tie_minigrid_variants(...)`. Listed here for catalog
    # completeness only; consumed specially in Phase 2.
    Axis("sibling_aware_leaf",              False, "MINIGRID_WITH_TIE_MARGIN",   False),
    # Training-time (model rebuilt per value)
    # seed_pattern perturbs k as well (k = popcount(pattern)); to keep this
    # OAT-clean we generate patterns whose weight equals the per-marker auto-k.
    # Reuse `spaced_seeds_for_weight` from oxidtaxa_hyperparameter_tuning.py:65-113
    # which emits 4 patterns at increasing span ratios. We take the two
    # extremes (densest periodic-3 and most-aperiodic) per marker.
    # Test values are filled in per marker at run time (see Phase 2).
    Axis("seed_pattern",                    None,  "PER_MARKER_FROM_AUTO_K",      True),
    Axis("descendant_weighting",            "count", ("equal", "log"),       True),
    Axis("use_idf_in_training",             False, (True,),                  True),
    Axis("leave_one_out",                   False, (True,),                  True),
    Axis("correlation_aware_features",      False, (True,),                  True),
)
```

**Marker list** — hardcoded and validated. We list every marker we expect explicitly so the run is self-documenting; any missing `seed_amplicons.fasta` is a hard error (not a silent skip):
```python
# Hardcoded — every marker we run must appear here. Adding/removing a marker
# is a deliberate code change, not an environment-dependent inference.
OAT_MARKERS: tuple[str, ...] = (
    "vert12s",
    "16smamm",
    "12s_mifish",
    "18s_euk",
    "its1_fungi",
    "its2_plants",
)

def _validate_markers() -> None:
    """Fail loudly at startup if any expected marker is misconfigured.
    Better to abort before a multi-hour run than to silently drop a marker."""
    missing: list[str] = []
    for name in OAT_MARKERS:
        if name not in MARKER_PRESETS:
            missing.append(f"{name}: not in MARKER_PRESETS")
            continue
        preset = MARKER_PRESETS[name]
        if preset.seed_amplicons_dir is None:
            missing.append(f"{name}: seed_amplicons_dir is None")
            continue
        seed_fasta = preset.seed_amplicons_dir / "seed_amplicons.fasta"
        if not seed_fasta.exists():
            missing.append(f"{name}: missing {seed_fasta}")
    if missing:
        raise RuntimeError(
            "OAT marker validation failed:\n  " + "\n  ".join(missing)
            + "\nFix the marker preset or remove the entry from OAT_MARKERS."
        )
```
`_validate_markers()` is called at the top of `main()`, before any community building, so a misconfigured environment fails in seconds rather than after the first long-running phase.

**`16s_bacteria` is intentionally excluded** from `OAT_MARKERS` (no `seed_amplicons_dir`); to add it later, both populate its seed pool *and* add the string to `OAT_MARKERS`.

### Success Criteria:

#### Automated Verification:
- [x] `python -c "from notebooks.oxidtaxa_oat_ablation import BACKDROP_KWARGS, ABLATION_AXES, OAT_MARKERS, _validate_markers; _validate_markers(); print(OAT_MARKERS)"` exits 0 and prints the 6-marker tuple.
- [ ] Removing/renaming any `seed_amplicons.fasta` in a temporary checkout makes `_validate_markers()` raise with a precise list of what's missing.
- [x] Every neutral_value in `ABLATION_AXES` matches the corresponding entry in `BACKDROP_KWARGS` (assert in `__main__`).
- [ ] No axis name collides with a non-neutralized backdrop key.

#### Manual Verification:
- [ ] Backdrop value list matches research doc summary table line-for-line.

---

## Phase 2: Variant generation and DB pre-training

### Overview
Translate `ABLATION_AXES` into a `list[ConfigVariant]` per marker and reuse the existing variant-DB pre-training to avoid redundant model builds.

### Changes Required:

#### 1. `notebooks/oxidtaxa_oat_ablation.py` — variant generation

```python
def _make_variants_for_marker(marker: str) -> list[ConfigVariant]:
    variants: list[ConfigVariant] = []
    # one shared baseline per marker
    baseline_kwargs = dict(BACKDROP_KWARGS)
    variants.append(_build_variant(
        param_name="baseline",
        param_value="idtaxa_default",
        kwargs=baseline_kwargs,
        is_default=True,
    ))
    for axis in ABLATION_AXES:
        for v in axis.test_values:
            kwargs = dict(BACKDROP_KWARGS)
            kwargs[axis.name] = v
            variants.append(_build_variant(
                param_name=axis.name,
                param_value=_value_token(v),     # safe filename token
                kwargs=kwargs,
                is_default=False,
            ))
    return variants
```

`_build_variant` constructs a `ConfigVariant` (see `classifier_benchmark.py:187-200`) with `tool="oxidtaxa"`, `is_post_processing=False`, and `db_variant_key` derived from training-only kwargs via `_make_oxidtaxa_db_variant_key` (`classifier_benchmark.py:861-886`).

#### 1a. `sibling_aware_leaf × tie_margin` mini-grid (interaction-aware)

**Why this is not pure OAT.** Both knobs broaden the leaf-phase winner pool but at different points (`src/classify.rs:264-272` for `sibling_aware_leaf`, `src/classify.rs:738-749` for `tie_margin`). `sibling_aware_leaf` widens `w_indices` (which child branches enter `leaf_phase_score`) **before** scoring; `tie_margin` widens the *winner set among groups* **inside** `leaf_phase_score`. They are partially redundant — testing each in isolation against `tie_margin=0.0, sibling_aware_leaf=False` cannot distinguish "sibling_aware_leaf does nothing" from "sibling_aware_leaf does nothing **once tie_margin already widened the set**, but would help if tie_margin=0".

**Replacement design**: `_make_sibling_tie_minigrid_variants(backdrop)` emits a 2×5 grid spanning the full plausible `tie_margin` range:

| Variant id | sibling_aware_leaf | tie_margin | Source |
|---|---|---|---|
| `(F, 0.0)` | False | 0.0 | shared baseline (already in variants list) |
| `(F, 0.05)` | False | 0.05 | covered by `Axis("tie_margin")` |
| `(F, 0.10)` | False | 0.10 | covered by `Axis("tie_margin")` |
| `(F, 0.20)` | False | 0.20 | covered by `Axis("tie_margin")` |
| `(F, 0.40)` | False | 0.40 | covered by `Axis("tie_margin")` |
| `(T, 0.0)`  | True  | 0.0  | new — isolates sibling_aware_leaf alone |
| `(T, 0.05)` | True  | 0.05 | new — interaction cell |
| `(T, 0.10)` | True  | 0.10 | new — interaction cell |
| `(T, 0.20)` | True  | 0.20 | new — interaction cell |
| `(T, 0.40)` | True  | 0.40 | new — interaction cell, expected near LCA-collapse |

Net new cells beyond the strict OAT: **5** (the five `T` cells). The `Axis("sibling_aware_leaf")` entry contributes only the `(T, 0.0)` cell to the regular variant generator; the other four `T` cells come from this mini-grid helper. Combined with the wider `Axis("tie_margin")` range (4 test values vs the original 2), the total per-(marker, replicate, baseline) cell count grows by 7 cells over the strict-OAT plan (2 extra `F` cells from wider tie_margin + 5 new `T` cells).

**Reporting** for this cell block:

- 2×5 F1 table per marker (mean ± SE across replicates × baselines), columns `unified_asv_truth_depth_f1` and `asv_truth_depth_tier_hmean`.
- Δ-F1 from `(T, tm) − (F, tm)` at each `tm` — quantifies the marginal effect of `sibling_aware_leaf` *given* the current tie_margin (5 deltas).
- Δ-F1 from `(s, tm) − (s, 0.0)` at each `s` for `tm ∈ {0.05, 0.10, 0.20, 0.40}` — `tie_margin` response curve at each `sibling_aware_leaf` setting.
- Identify the `tie_margin` value where Δ-F1 turns negative ("LCA-collapse breakdown point") at each `s` — expected somewhere in `[0.10, 0.40]` based on the broadening semantics.

#### 1b. `sibling_aware_leaf` fire-rate instrumentation

**Goal**: count how often the branch *enters* (`branch_fired`) and how often it *changes the prediction* (`effective_fire`). Two measurement modes:

**Mode A — harness-side effective-fire diff (no oxidtaxa change required)**:
- For each `(community, baseline, tie_margin)` cell, compare predictions between the matching `(F, tm)` and `(T, tm)` `unified.parquet` files row-by-row on `asv_id`.
- `effective_fire_rate := |{asv : pred(T,tm) ≠ pred(F,tm)}| / |asvs|` per cell.
- Report per-marker mean and per-tier breakdown (normal / haplotype / species_holdout / genus_holdout).
- A zero `effective_fire_rate` across all 6 markers × 3 baselines × N replicates would be conclusive evidence that `sibling_aware_leaf` is inert under the current backdrop.

**Mode B — oxidtaxa-side branch-entry counter (optional, requires Rust change)**:
- Add an `AtomicUsize` to `ClassifyResults` (or a new `ClassifyStats` field) incremented at `src/classify.rs:264` whenever the `if config.sibling_aware_leaf` arm is taken **and** the widened `w_indices` actually contains more than `{winner}`.
- Expose via PyO3 as `result.stats.sibling_aware_leaf_fires`.
- Harness divides by total query count to get `branch_fire_rate`.
- This distinguishes "branch entered and broadened the pool" from "branch entered but only the winner survived the ≥0.5*b filter" (a no-op).

**Recommendation**: ship Mode A unconditionally (pure harness work, no Rust change, captures the actually-load-bearing definition of "fires"). Mode B is a follow-up if Mode A reports zero effective fires and you want to confirm the branch is reachable at all.

#### 1c. Per-marker `seed_pattern` resolution
The `seed_pattern` axis sets `k = popcount(seed_pattern)`. To keep the test OAT-clean (only `seed_pattern` differs from baseline), we need patterns whose weight equals the marker's auto-k. Auto-k is `floor(ln(n*L99)/ln(alphabet))` clamped to `[1,13]` (`oxidtaxa/src/types.rs:238-240`, `src/training.rs:48-54`); typical for these markers is 7-9.

```python
from notebooks.oxidtaxa_hyperparameter_tuning import spaced_seeds_for_weight

def _resolve_seed_pattern_test_values(reference_fasta: Path, taxonomy_file: Path) -> tuple[str, ...]:
    """Compute auto-k for this marker, then take the densest + most-aperiodic spaced
    seed at that weight. Both have popcount = auto-k, so swapping them in does not
    co-vary k with seed_pattern."""
    from oxidtaxa import compute_auto_k  # or compute via prepare_data probe
    auto_k = compute_auto_k(reference_fasta, taxonomy_file, n=500.0)
    patterns = spaced_seeds_for_weight(auto_k)  # returns 4 patterns
    return (patterns[0], patterns[3])           # densest periodic-3 + most-aperiodic
```

If `compute_auto_k` is not directly exposed, replicate the formula from `src/training.rs:48-54` in Python (need 99th-percentile sequence length from the filtered reference, plus alphabet size 4 for DNA). The variants list is constructed *after* this resolution, so `Axis("seed_pattern", ...)` gets its `test_values` substituted before `_make_variants_for_marker` runs.

#### 2. Pre-training reuse
Call `build_oxidtaxa_variant_dbs(variants, base_db_dir, reference_fasta, taxonomy_file, logger)` directly (`classifier_benchmark.py:995-1079`). All non-training axes share the same `db_variant_key` as the baseline, so only the 5 training axes × Σtest_values produce extra trainings. With the chosen test_values:
- `seed_pattern`: 2 extra
- `descendant_weighting`: 2 extra
- `use_idf_in_training`: 1 extra
- `leave_one_out`: 1 extra
- `correlation_aware_features`: 1 extra
→ 7 extra trained models per (marker, community), plus 1 baseline = 8 model builds per (marker, community).

### Success Criteria:

#### Automated Verification:
- [x] `len(_make_variants_for_marker("vert12s"))` equals `1 + sum(len(ax.test_values) for ax in ABLATION_AXES)` (25 variants total).
- [x] Number of unique `db_variant_key` values across the variant list equals `1 + 7 = 8`.
- [x] All non-baseline classify-only variants share `db_variant_key` with the baseline.

#### Manual Verification:
- [x] Inspect one variant's `config_kwargs` dict: differs from baseline by exactly one key (verified for all OAT non-baseline variants; the sibling × tie mini-grid intentionally varies two keys, which is documented).

---

## Phase 2.5: R IDTAXA reference cell

### Overview
Add R IDTAXA as a one-per-(community, baseline) reference run. Same holdout DBs, same metric attachment pipeline. Captured runtime is the wall-clock denominator the oxidtaxa speedup claim is measured against.

### Changes Required:

#### 1. IDTAXA config (canonical defaults)
```python
IDTAXA_REFERENCE_KWARGS: dict[str, Any] = {
    # Mirrors R DECIPHER IdTaxa() defaults exactly (see research doc).
    "threshold": 60.0,
    "bootstraps": 100,
    "strand": "both",
    "min_descend": 0.98,
    "full_length": 0.0,
    # processors: forwarded to R's `processors` arg. Set to NUM_THREADS to
    # give IDTAXA the same hardware as oxidtaxa for a fair runtime compare.
    "threads": NUM_THREADS,
}
```

#### 2. Variant generation
`_make_variants_for_marker` also emits one `ConfigVariant(tool="idtaxa", param_name="reference", param_value="canonical_defaults", ...)` per marker. The existing IDTAXA executor (`tools/idtaxa/`) and DB builder are already invoked in `classifier_benchmark.py` — reuse them via the same patterns used at `classifier_benchmark.py:367-572` (`SWEEP_DEFINITIONS["idtaxa"]`).

#### 3. IDTAXA DB pre-training
Train one R IDTAXA model per (marker, community) using the same holdout reference FASTA + taxonomy that oxidtaxa consumes. The existing IDTAXA `DatabaseBuilder` handles this. Cache under `DATA_BASE / "oxidtaxa_oat" / "idtaxa_models" / <marker> / <community>/`.

### Success Criteria:

#### Automated Verification:
- [ ] Variants list contains exactly one `tool="idtaxa"` row per marker.
- [ ] IDTAXA DB built once per (marker, community); reused across baselines.
- [ ] IDTAXA run produces a `unified.parquet` with the same schema as oxidtaxa cells.

#### Manual Verification:
- [ ] R IDTAXA actually trains and classifies (no missing R packages, no crashes).
- [ ] IDTAXA runtime is captured and looks plausible (typically minutes per community).

---

## Phase 2.6: Runtime capture

### Overview
We already get `duration` from `run_tool()` (used in `oxidtaxa_hyperparameter_tuning.py:709-716`). Persist it as a first-class column in every row.

### Changes Required:

#### 1. SweepResult extension or sidecar
Either extend the in-memory row dict with `classify_seconds` and `train_seconds` (preferred, since `SweepResult` already has fields like these — verify against `classifier_benchmark.py:202-356`), or write a per-cell `timing.json` next to `unified.parquet`:
```json
{"tool": "oxidtaxa", "variant": "beam_width=3", "train_seconds": 47.2, "classify_seconds": 18.6}
```

For the IDTAXA reference and every oxidtaxa cell:
- `train_seconds`: wall-clock to build/load the model (zero on cache hit).
- `classify_seconds`: wall-clock of the `classify`/`IdTaxa` call only (excludes parquet I/O and metric attachment).

#### 2. Aggregation
Per-marker rollup adds runtime columns alongside F1 columns:
- `classify_seconds_mean`, `classify_seconds_se`
- `speedup_vs_idtaxa = idtaxa_classify_seconds / oxidtaxa_classify_seconds` per (community, baseline), then averaged.

### Success Criteria:

#### Automated Verification:
- [ ] Every row in `oat_results.json` has non-null `classify_seconds`.
- [ ] IDTAXA reference rows have non-null `classify_seconds`.

#### Manual Verification:
- [ ] Reported speedups are in the expected order of magnitude (oxidtaxa typically ≥5× R IDTAXA on the same hardware).

---

## Phase 3: Sweep execution

### Overview
For each marker, build communities + holdout DBs once, pre-train oxidtaxa variant DBs, then iterate (community × baseline × variant) cells through the existing in-process executor.

### Changes Required:

#### 1. `notebooks/oxidtaxa_oat_ablation.py` — main loop

```python
def run_oat_for_marker(marker: str, *, num_replicates: int, baselines: list[str],
                      data_base: Path, results_base: Path, logger: logging.Logger) -> None:
    preset = MARKER_PRESETS[marker]
    reference_fasta, taxonomy_file = _resolve_reference(preset)

    # 1. Communities (always seed_pool)
    communities = _build_communities(
        marker=preset.marker,
        use_seed_pool=True,
        num_replicates=num_replicates,
        seed_pool_fasta=preset.seed_amplicons_dir / "seed_amplicons.fasta",
        # other args mirroring classifier_benchmark _build_communities call site
    )

    # 2. Holdout splits + per-tool DBs (reuse existing helpers)
    unified_splits = {c.name: select_unified_holdout(c, ...) for c in communities}
    holdout_db_dirs = build_holdout_databases(
        communities=communities, unified_splits=unified_splits,
        tools=("oxidtaxa",), output_root=data_base / "oxidtaxa_oat",
        logger=logger,
    )

    # 3. Variants + variant DB pre-training
    variants = _make_variants_for_marker(marker)
    variant_db_root = data_base / "oxidtaxa_oat" / "variant_models" / marker
    variant_db_dirs = build_oxidtaxa_variant_dbs(
        variants, base_db_dir=variant_db_root,
        reference_fasta=reference_fasta, taxonomy_file=taxonomy_file,
        logger=logger,
    )

    # 4. Cells
    sweep_rows: list[SweepResult] = []
    for community in communities:
        for baseline in baselines:
            reads_path, ground_truth = _generate_or_load_reads(community, baseline)
            for variant in variants:
                exp_dir = (data_base / "oxidtaxa_oat" / "experiments"
                           / variant.param_name / variant.param_value
                           / community.name / baseline)
                exp_dir.mkdir(parents=True, exist_ok=True)
                _run_oxidtaxa_cell(
                    variant=variant, reads=reads_path, exp_dir=exp_dir,
                    db_dir=variant_db_dirs[variant.db_variant_key],
                )
                row = _attach_holdout_metrics(
                    variant=variant, community=community, baseline=baseline,
                    exp_dir=exp_dir, holdout_split=unified_splits[community.name],
                    # ...same args as classifier_benchmark _attach_holdout_metrics
                )
                sweep_rows.append(row)
                _save_partial(sweep_rows, results_base / "oxidtaxa_oat" / marker / "oat_results.json")
```

#### 2. CLI entry point

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markers", nargs="*", default=None,
                    help="Subset of markers; default = all with seed_pool.")
    ap.add_argument("--replicates", type=int, default=NUM_REPLICATES)
    ap.add_argument("--baselines", nargs="*", default=list(BASELINES_CLASSIFIER))
    ap.add_argument("--smoke", action="store_true",
                    help="Single marker, 1 replicate, clean baseline only, 2 axes.")
    args = ap.parse_args()

    _validate_markers()                                   # hard-fail on any missing seed pool
    markers = args.markers or list(OAT_MARKERS)
    # If --markers was passed, every name in it must also be in OAT_MARKERS
    # (the hardcoded source of truth) so we can never accidentally process
    # an out-of-list marker.
    bad = [m for m in markers if m not in OAT_MARKERS]
    if bad:
        raise SystemExit(f"--markers contains entries not in OAT_MARKERS: {bad}")
    if args.smoke:
        markers = markers[:1]; args.replicates = 1; args.baselines = ["clean"]
        # plus monkey-patch ABLATION_AXES to a 2-element subset

    for m in markers:
        run_oat_for_marker(
            marker=m, num_replicates=args.replicates, baselines=args.baselines,
            data_base=DATA_BASE, results_base=RESULTS_BASE, logger=_make_logger(),
        )
```

### Success Criteria:

#### Automated Verification:
- [ ] Smoke run completes without error: `python notebooks/oxidtaxa_oat_ablation.py --smoke`.
- [ ] `oat_results.json` written incrementally — file exists after first cell finishes.
- [ ] Per-experiment `unified.parquet` exists for every (axis, value, community, baseline).
- [ ] Row count in final `oat_results.json` equals `len(communities) × len(baselines) × len(variants)`.

#### Manual Verification:
- [ ] Wall clock comparable to or faster than equivalent serial run (validate `processors` is being honored — look at CPU usage).
- [ ] Variant model directories under `variant_models/<marker>/oxidtaxa_db_<key>/` show exactly 8 unique keys per marker.
- [ ] No NaN values in `unified_asv_truth_depth_f1` column.
- [ ] Spot-check: baseline row in two different axes has matching F1 within numerical noise.

---

## Phase 4: Reporting

### Overview
Per-marker and cross-marker summaries that focus on the user's chosen metric family (truth-depth and pred-depth, ASV-level).

### Changes Required:

#### 1. `notebooks/oxidtaxa_oat_ablation.py` — report generation

**Headline columns** (subset of `SweepResult` fields, see `classifier_benchmark.py:271-292`):
- Primary: `unified_asv_truth_depth_f1`, `unified_asv_pred_depth_phylum_f1`
- Tier-harmonic-mean: `asv_truth_depth_tier_hmean`, `asv_pred_depth_phylum_tier_hmean`
- Per-tier truth-depth: `normal_asv_truth_depth_f1`, `haplotype_asv_truth_depth_f1`, `species_holdout_asv_truth_depth_f1`, `genus_holdout_asv_truth_depth_f1` (verify exact field names against `SweepResult` `285-292`)
- Composite Optuna scalar: `taxon_native_objective` (`SweepResult:283`) — for cross-checking sign of effect

```python
def _per_marker_report(rows: list[SweepResult], out_path: Path) -> None:
    df = _rows_to_df(rows)
    metric_cols = [
        "unified_asv_truth_depth_f1", "unified_asv_pred_depth_phylum_f1",
        "asv_truth_depth_tier_hmean", "asv_pred_depth_phylum_tier_hmean",
        # per-tier
        "normal_asv_truth_depth_f1", "haplotype_asv_truth_depth_f1",
        "species_holdout_asv_truth_depth_f1", "genus_holdout_asv_truth_depth_f1",
    ]
    # group by (param_name, param_value); aggregate over (community, baseline) with mean ± SE
    agg = df.groupby(["param_name", "param_value"])[metric_cols].agg(["mean", "sem"])
    # delta vs baseline row
    baseline = agg.xs(("baseline", "idtaxa_default"))
    deltas = agg.subtract(baseline, axis=1)
    _write_markdown(out_path, agg, deltas)
```

**Cross-marker rollup**: for each axis × test_value, compute mean Δ-F1 across markers + sign agreement count (e.g., "5/6 markers improved").

**R IDTAXA comparison block** (per marker, top of report):
| Variant | F1 (truth-depth ASV) | Δ vs IDTAXA | Classify s | Speedup vs IDTAXA |
|---|---|---|---|---|
| `idtaxa` (canonical) | … | 0.000 | … | 1.0× |
| `oxidtaxa baseline` (IDTAXA-default backdrop) | … | … | … | … |
| `oxidtaxa <axis>=<value>` (per axis × test value) | … | … | … | … |

The headline claim is read off this table: each oxidtaxa row should beat IDTAXA on F1 *and* speedup. Rows that beat IDTAXA on F1 but lose on speedup (or vice versa) are flagged.

#### 2. Output paths
- Per marker: `RESULTS_BASE / "oxidtaxa_oat" / <marker> / oat_summary.md`
- Cross-marker: `RESULTS_BASE / "oxidtaxa_oat" / oat_rollup.md`

### Success Criteria:

#### Automated Verification:
- [ ] `oat_summary.md` exists for each marker after a full run.
- [ ] `oat_rollup.md` exists at top level.
- [ ] Per-axis summary table contains exactly `1 + len(axis.test_values)` rows for that axis (e.g., 5 rows for `tie_margin`, 4 for `rank_thresholds`).
- [ ] Sibling × tie mini-grid table contains exactly 10 rows (2×5).
- [ ] Per-marker headline table includes the IDTAXA reference row + the 25 oxidtaxa variants.

#### Manual Verification:
- [ ] Baseline row deltas are zero (sanity).
- [ ] `sibling_aware_leaf` row shows Δ ≈ 0 across markers (matches prior empirical finding).
- [ ] Sign of `confidence_uses_descent_margin` effect is consistent with the I6 fix in commit `e287eb7`.

---

## Testing Strategy

### Unit checks (in `__main__` of the script):
- `BACKDROP_KWARGS` neutralizes every oxidtaxa-added param (assert against research doc table).
- Each `Axis.neutral_value == BACKDROP_KWARGS[Axis.name]`.
- Variant generation: exactly one differing key per non-baseline variant.
- DB-key dedup: only training axes change `db_variant_key`.

### Integration smoke:
`python notebooks/oxidtaxa_oat_ablation.py --smoke` → 1 marker (vert12s), 1 replicate, 1 baseline (clean), 2 axes (`length_normalize`, `beam_width`). Expected cells: 1 community × 1 baseline × (1 baseline + 1 + 3) variants = 5. Wall-clock target < 5 minutes on the dev machine.

### Manual verification:
1. Smoke run; inspect `oat_summary.md` for vert12s — table should be small but well-formed.
2. Full run for a single marker; verify model count = 8.
3. Two-marker run; confirm rollup table aggregates correctly.
4. Full multi-marker run.

## Performance Considerations

- **Variants per marker**: 1 baseline + 20 strict-OAT axis test values (1+3+3+4+1+1+2+2+1+1+1) + 4 mini-grid `T`-cells beyond `(T, 0.0)` = **25 oxidtaxa variants** + **1 R IDTAXA reference** = 26 cells per (marker, replicate, baseline).
- **Total oxidtaxa cells**: 6 markers × N_replicates × 3 baselines × 25. With `NUM_REPLICATES=3`, that is **1,350 oxidtaxa runs**.
- **Total IDTAXA cells**: 6 × 3 × 3 × 1 = **54 R IDTAXA runs**.
- **Total oxidtaxa trainings**: 6 markers × N_replicates × 8 unique training keys = **144 trainings** (one per community per key; holdout DBs are community-specific).
- **Total IDTAXA trainings**: 6 × N_replicates × 1 = **18 trainings** (R IDTAXA model per (marker, community)).
- **Parallelism**: `deterministic=False` + `processors=NUM_THREADS` means each oxidtaxa call uses the full thread pool. We do not parallelize cells across processes — they are run sequentially per marker so the rayon pool gets full hardware.
- **Disk**: each `unified.parquet` is small (≤ a few MB); aggregate footprint dominated by the 8 trained model directories per (marker, community).
- **Caching**: ground_truth.json and variant model dirs are skip-if-exists; reruns are cheap.

## Migration Notes

This is additive — no existing files are modified. Output lives under a new `oxidtaxa_oat/` subtree of `DATA_BASE` and `RESULTS_BASE`, parallel to the existing `oxidtaxa/` and `oxidtaxa_tuning/` trees.

## Prior Empirical Context (informs interpretation, not axis selection)

The Optuna script `oxidtaxa_hyperparameter_tuning.py:583-596` documents two axes that are *pinned False* in tuning because prior sweeps showed they degrade performance:

- `confidence_uses_descent_margin`: pinned False — "even after oxidtaxa fixes 79ecdb9, 17b492b, e287eb7, the 250-trial vert12s run showed True underperforms across the board (49 trials, mean 0.274, max 0.498 vs False n=201, mean 0.447, max 0.511)."
- `length_normalize`: pinned False — "KILL-flagged across all three markers in the prior sweep's top decile."

We **still** include these axes in the OAT (the whole point is single-axis attribution against an IDTAXA-default backdrop), but the prior is: expect Δ ≤ 0 for both. If the OAT confirms this on all six markers, the plan-level next step is to mark them deprecated; if any marker shows Δ > 0, that is a new finding worth investigating.

`sibling_aware_leaf` is similarly expected near-zero per the 5,520-run cartesian (research doc line 72).

## Open Decisions Resolved (no remaining open questions)

- **`deterministic`**: set to `False` for parallel execution. Justification: this study compares oxidtaxa configurations to each other, not to R IDTAXA, so PRNG-mode has no semantic role. Holding it constant across all rows preserves OAT validity.
- **`min_descend`**: set to `0.98` — confirmed canonical DECIPHER default (R signature: `IdTaxa(..., minDescend=0.98, ...)` in `R/IdTaxa.R`). This is also the oxidtaxa Rust default.
- **`strand`**: set to `"both"` — confirmed canonical DECIPHER default.
- **`record_kmers_fraction`**: set to `0.10` — `recordKmers` is not a user-exposed DECIPHER parameter; the value `0.10` is hardcoded inside `LearnTaxa.R`'s `.createTree` helper as `recordKmers <- ceiling(max(colSums(profile > 0)))*0.10`. Oxidtaxa exposes it as a knob, but the IDTAXA-faithful value is `0.10`.
- **`16s_bacteria`**: excluded (no seed_amplicons). All other 6 markers in scope.
- **Test value count per axis**: 1-3 values per axis, biased toward small N to keep total cell count manageable. Boolean axes get a single `True` test value; categorical (`descendant_weighting`) gets all non-default variants; continuous axes get 2 values bracketing the plausible range.

## References

- Research doc: `thoughts/shared/research/2026-04-27-oxidtaxa-vs-idtaxa-ablation-surface.md`
- Sweep harness: `notebooks/classifier_benchmark.py:148-160, 800-1079, 202-356, 3447-3476, 3653-3687`
- Marker presets: `assignment_benchmarks/infrastructure/benchmark_shared.py:354-419`
- Oxidtaxa config: `assignment_benchmarks/infrastructure/tools/oxidtaxa/config.py:48-127`
- Rust defaults: `oxidtaxa/src/types.rs:225-353`
- Prior sibling_aware_leaf inertness: `thoughts/shared/research/2026-04-21-three-marker-sweep-findings.md`
