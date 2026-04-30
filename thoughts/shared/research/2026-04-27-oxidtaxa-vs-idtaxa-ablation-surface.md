---
date: 2026-04-27
researcher: Ryan Martin
git_commit: e19df516aa064a8eb37d956a58362838f29da63a
branch: main
repository: oxidtaxa
topic: "Oxidtaxa parameters added beyond IDTAXA, IDTAXA-default neutralizing values, and the classifier_benchmark.py community + metrics design used to assess them"
tags: [research, codebase, oxidtaxa, classifier_benchmark, parameters, idtaxa, ablation]
status: complete
last_updated: 2026-04-27
last_updated_by: Ryan Martin
---

# Research: Oxidtaxa-vs-IDTAXA parameter surface and the benchmarking design used in `classifier_benchmark.py`

**Date**: 2026-04-27
**Researcher**: Ryan Martin
**Git Commit**: e19df516aa064a8eb37d956a58362838f29da63a
**Branch**: main
**Repository**: oxidtaxa

## Research Question

Several parameters have been added to oxidtaxa beyond what IDTAXA exposes. The user wants to test each one independently — every other parameter held at its IDTAXA-default value — across several markers, using the same community design and metrics that `classifier_benchmark.py` already uses. This document inventories (a) which oxidtaxa parameters are IDTAXA-original vs newly added, (b) the "neutralizing" value for each new parameter that reproduces stock IDTAXA behavior, and (c) the community construction, sweep harness, and metrics computed by `classifier_benchmark.py`.

## Summary

**Newly-added oxidtaxa parameters** (i.e., not present in stock IDTAXA / DECIPHER):

| Parameter | Phase | Neutralizing value (= IDTAXA behavior) |
|---|---|---|
| `length_normalize` | classify | `false` |
| `rank_thresholds` | classify | `None` |
| `beam_width` | classify | `1` (greedy descent) |
| `tie_margin` | classify | `0.0` |
| `confidence_uses_descent_margin` | classify | `false` |
| `sibling_aware_leaf` | classify | `false` |
| `deterministic` | classify | `true` (R-compatible sequential PRNG); set `false` for per-query independent PRNG |
| `seed_pattern` | training | `None` (contiguous k-mers) |
| `descendant_weighting` | training | `Count` (`"count"`) |
| `use_idf_in_training` | training | `false` |
| `leave_one_out` | training | `false` |
| `correlation_aware_features` | training | `false` |

`processors` (classify and train) is also oxidtaxa-only; `1` = serial like R.

**IDTAXA-original parameters** that oxidtaxa exposes — for an apples-to-apples ablation these need to be pinned to their IDTAXA defaults: `threshold=60.0`, `bootstraps=100`, `strand="both"`, `min_descend=0.98`, `full_length=0.0`, `sample_exponent=0.47` (`L^0.47`), `k=None` (auto), `n=500.0`, `min_fraction=0.01`, `max_fraction=0.06`, `max_iterations=10`, `multiplier=100.0`, `max_children=200`, `record_kmers_fraction=0.10` (hardcoded internal constant in DECIPHER), `training_threshold=0.8`. All confirmed against the canonical DECIPHER source — `R/IdTaxa.R` signature `IdTaxa(test, trainingSet, type="extended", strand="both", threshold=60, bootstraps=100, samples=L^0.47, minDescend=0.98, fullLength=0, processors=1, verbose=TRUE)` and `R/LearnTaxa.R` signature `LearnTaxa(train, taxonomy, rank=NULL, K=NULL, N=500, minFraction=0.01, maxFraction=0.06, maxIterations=10, multiplier=100, maxChildren=200, ...)`. `recordKmers` is not a user-exposed DECIPHER parameter — the `0.10` constant is hardcoded inside `.createTree` in `LearnTaxa.R`.

**Community design**: Each marker community contains 100 species. Selection strategy is auto-detected at `classifier_benchmark.py:2896-2916`: if the marker preset has a `seed_amplicons.fasta` available, **`SEED_POOL`** is used; otherwise the harness falls back to **`TAXONOMIC_TIERS`** (controlled congeneric/confamilial/conordinal distance distribution). `--strategy` overrides the auto-detection. The 100 species are partitioned into ~33 each across three tiers: **normal** (accession in DB), **haplotype holdout** (query accession excluded from DB), **species holdout** (entire species removed from DB). A fourth **genus_holdout** stratum exists in the metric output. A single unified F1 sums TP/FP/FN across all tiers. `NUM_REPLICATES` communities per marker; baselines `clean | realistic | degraded` are run for each.

**Metrics** are columns on `SweepResult` (lines 202-356); the headline scalar used for picking the best config is `_selection_score = 0.77 * unified_asv_truth_depth_f1 + 0.23 * unified_asv_pred_depth_phylum_f1` (lines 1910-1933).

## Detailed Findings

### Component 1 — Oxidtaxa configurable parameter surface

Authoritative source: `/Users/ryanmartin/oxidtaxa/src/types.rs`, `src/classify.rs`, `src/training.rs`, `src/lib.rs`, plus the README parameter tables.

#### Classify parameters (`ClassifyConfig`, src/types.rs:298-353)

- **`threshold`** (f64, default 60.0) — Per-rank confidence cutoff on a 0-100 scale; truncates the predicted lineage at the first rank below threshold. (`src/types.rs:299,339`; `src/classify.rs:861-867`; `src/lib.rs:138,153,179`) **Origin**: IDTAXA.
- **`bootstraps`** (usize, default 100) — Cap on bootstrap replicates per query; sized as `min(5*L/S, bootstraps)`. (`src/types.rs:302,340`; `src/classify.rs:107,124-131,244,773`) **Origin**: IDTAXA.
- **`strand`** (string, default `"both"`) — `top|bottom|both`. (`src/lib.rs:138,177,319-326`) **Origin**: IDTAXA.
- **`min_descend`** (f64, default 0.98) — Bootstrap-vote fraction a child must achieve to be descended into (and to be kept as beam runner-up). (`src/types.rs:303,341`; `src/classify.rs:243-258,430,460`) **Origin**: IDTAXA — canonical R default is also `0.98` (confirmed against `R/IdTaxa.R`).
- **`full_length`** (f64, default 0.0) — Length-ratio filter on training sequences in the leaf phase; `0.0` disables. (`src/types.rs:304,342`; `src/classify.rs:156-160,564-573`) **Origin**: IDTAXA.
- **`sample_exponent`** (f64, default 0.47) — Per-query sample size `S = ceil(L^sample_exponent)`. (`src/types.rs:308,344`; `src/classify.rs:108-111,576`) **Origin**: IDTAXA (matches Murali et al. 2018 `L^0.47`).
- **`length_normalize`** (bool, default false) — Divides per-training-seq hits and `sum_hits` by `sqrt(n_unique/avg_unique)`. (`src/types.rs:311,345`; `src/classify.rs:649-662`) **Origin**: oxidtaxa-added. Neutralizing = `false`.
- **`rank_thresholds`** (Option<Vec<f64>>, default None) — Per-depth threshold override; replaces the scalar `threshold` per rank. (`src/types.rs:314,346`; `src/classify.rs:858-862`) **Origin**: oxidtaxa-added. Neutralizing = `None`.
- **`beam_width`** (usize, default 1) — Switch and width for tree descent; values >1 dispatch to `classify_one_pass_beam` which keeps the top-k branches by accumulated vote-fraction product. (`src/types.rs:318,347`; `src/classify.rs:178-180,317,502-525`) **Origin**: oxidtaxa-added. Neutralizing = `1` (= greedy = original IDTAXA).
- **`tie_margin`** (f64, default 0.0) — Relative-margin tie definition for leaf-phase winners; with `>0` and `max_tot>0`, includes every group with `tot_hits >= max_tot * (1-tie_margin)`. (`src/types.rs:323,348`; `src/classify.rs:738-749`) **Origin**: oxidtaxa-added. Neutralizing = `0.0`.
- **`confidence_uses_descent_margin`** (bool, default false) — When true, multiplies per-rank confidences by `0.8 + 0.2*descent_margin` (non-cumulative). (`src/types.rs:328,349`; `src/classify.rs:230-241,392-407,804-812`) **Origin**: oxidtaxa-added. Neutralizing = `false`.
- **`sibling_aware_leaf`** (bool, default false) — When greedy descent terminates at a leaf parent with a single winner, widens `w_indices` to include siblings with ≥0.5*b votes. (`src/types.rs:333,350`; `src/classify.rs:264-271`) **Origin**: oxidtaxa-added. Neutralizing = `false`. **Note**: per the prior `2026-04-21-three-marker-sweep-findings` and 5,520 paired runs in this branch, this gate has been observed to be empirically inert under the cartesian sweep configuration.
- **`processors`** (usize, default 1) — Local rayon thread pool size. (`src/types.rs:305,343`; `src/classify.rs:46-49,950-952`) **Origin**: oxidtaxa-added (R is serial). Neutralizing = `1`.
- **`deterministic`** (bool, default true) — `true` runs sequentially with R-compatible PRNG semantics; `false` uses per-query independent PRNGs (parallel-safe but non-R-equivalent). (`src/lib.rs:141,161,70-82`; README.md:339) **Origin**: oxidtaxa-added knob over PRNG mode. Neutralizing = `true`.
- **`seed`** (u32, default 42) — PRNG seed. (`src/lib.rs:140,160,953`) **Origin**: IDTAXA-equivalent (R uses `set.seed`).

#### Train parameters (`TrainConfig`, src/types.rs:237-295)

- **`k`** (Option<usize>, default None=auto) — `floor(ln(n*L99)/ln(alphabet))` clamped to `[1,13]`. (`src/types.rs:238,278`; `src/training.rs:48-54,119-135`) **Origin**: IDTAXA.
- **`n`** (f64, default 500.0) — Auto-`k` numerator constant. (`src/types.rs:239,279`) **Origin**: IDTAXA.
- **`min_fraction` / `max_fraction` / `max_iterations` / `multiplier`** (defaults 0.01 / 0.06 / 10 / 100.0) — Bounds and step-size for the per-node `fraction` learning loop. (`src/types.rs:240-243,280-283,127-130`) **Origin**: IDTAXA.
- **`max_children`** (usize, default 200) — Max direct children retained in `build_tree` feature selection. (`src/types.rs:244,284`; `src/lib.rs:262`) **Origin**: IDTAXA.
- **`record_kmers_fraction`** (f64, default 0.10) — Fraction of top cross-entropy k-mers retained at each `DecisionNode.keep`. (`src/types.rs:247,285`; `src/lib.rs:65,79,245,252,259`) **Origin**: IDTAXA (DECIPHER `recordKmers=0.1`).
- **`seed_pattern`** (Option<String>, default None) — Spaced-seed mask `"11011011011"` etc.; when set, `k = popcount(seed_pattern)`. (`src/types.rs:249,286`; `src/training.rs:113-116,119-120,159`; `src/classify.rs:98-101`) **Origin**: oxidtaxa-added. Neutralizing = `None`.
- **`training_threshold`** (f64, default 0.8) — Vote-fraction threshold inside `learn_fractions` descent. (`src/types.rs:253,287`; `src/types.rs:124`; `src/lib.rs:67,82,274,284`) **Origin**: IDTAXA (R hardcodes 0.8).
- **`descendant_weighting`** (Count|Equal|Log, default Count) — Per-child weight in cross-entropy feature selection. (`src/types.rs:227-234,256,288`; `src/lib.rs:308-317`) **Origin**: oxidtaxa-added (variants); `Count` = original IDTAXA.
- **`use_idf_in_training`** (bool, default false) — Score with per-rank IDF instead of profile weights during fraction learning. (`src/types.rs:260,289`; `src/types.rs:125`) **Origin**: oxidtaxa-added.
- **`leave_one_out`** (bool, default false) — Exclude self-sequence from node profile during fraction learning. (`src/types.rs:264,290`; `src/types.rs:126`) **Origin**: oxidtaxa-added.
- **`correlation_aware_features`** (bool, default false) — Greedy correlation-aware feature selector with Bhattacharyya redundancy. (`src/types.rs:269,291`; `src/types.rs:117`) **Origin**: oxidtaxa-added.
- **`processors`** (usize, default 1) — Same as classify. (`src/types.rs:272,292`; `src/training.rs:37-42`) **Origin**: oxidtaxa-added.

Origin classifications and IDTAXA-default values are confirmed by the README parameter tables at `/Users/ryanmartin/oxidtaxa/README.md:113-146`, the in-source comments on each Config field (e.g., `types.rs:228` "original IDTAXA behavior"; `types.rs:316` "1 = greedy descent (original behavior)"; `types.rs:331` "default false: only single winner contributes (legacy behavior)"), and the README "Correctness and Divergence from R" section at lines 345-351.

### Component 2 — Python harness defaults

`/Users/ryanmartin/assignment-tool-benchmarking/projects/assignment_benchmarks/src/assignment_benchmarks/infrastructure/tools/oxidtaxa/config.py`

The `OxidtaxaConfig` dataclass (lines 48-75) mirrors every Rust default exactly except `processors`, which is derived from the inherited `threads` field on `AssignmentToolConfig` (`config.py:56`) and forwarded as the `processors` kwarg in `executor.py:127`. `__post_init__` (lines 77-127) enforces ranges:

- `threshold` 0-100, `bootstraps >= 1`, `strand` ∈ `{top,bottom,both}`, `sample_exponent` ∈ `(0,1]`, `min_descend` ∈ `(0,1]`, `full_length >= 0`, `k >= 1` or `None`, `record_kmers_fraction` ∈ `(0,1]`, `rank_thresholds` entries 0-100, `seed_pattern` chars in `"01"`, `training_threshold` ∈ `(0,1]`, `descendant_weighting` ∈ `{count,equal,log}`, `beam_width >= 1`, `tie_margin` ∈ `[0,1]`.

The executor inspects `classify`'s signature at runtime (`executor.py:115-117,146`) and silently drops kwargs the installed oxidtaxa version does not accept.

### Component 3 — Community design (classifier_benchmark.py)

#### Marker selection
`--marker` (`classifier_benchmark.py:2483-2489`) selects from `MARKER_PRESETS` (default `vert12s`); valid presets include `16smamm`, `18s_euk`, `MiFish`, `16s_bacteria` (the last is special-cased to drop the `harsh` baseline at `2938-2940`). Active marker resolved at `2782-2786`. Reference FASTA + taxonomy come from `m.cruxv2_dir / f"{m.name}.fasta"` and `..._taxonomy.txt` (`2753-2754, 3578-3579`). DB loaded by `prepare_db()` at `1164-1176` via `load_cruxv2_database()`.

#### Community construction
`_build_communities()` (`148-160`) delegates to `benchmark_setup.build_communities` with `NUM_REPLICATES` and a `use_seed_pool` flag. Each community has 100 species.

Strategy is resolved at `2896-2916`. The `--strategy` CLI flag (`2500-2506`) accepts `seed_pool | taxonomic_tiers | targeted`. **Auto-detection** (when `--strategy` is omitted): if `preset.seed_amplicons_dir / "seed_amplicons.fasta"` exists for the active marker, `use_seed_pool=True` and **`SEED_POOL`** is used (`2906-2912`); otherwise it falls back to **`TAXONOMIC_TIERS`** with a warning (`2913-2916`). The module docstring (`13-17`) describes the `TAXONOMIC_TIERS` semantics (controlled congeneric/confamilial/conordinal/etc. distance distribution) but the runtime path is governed by the seed-pool check above. Difficulty levels `easy|medium|hard` enumerated at `2473`.

Read generation at `3248-3261` via `generate_benchmark_reads()` using `READ_DEPTH`, `READ_LENGTH`, marker primers, and amplicon length window. Abundance: `ABUNDANCE_MODEL = "uniform"`, `ABUNDANCE_PARAMS = {}` (`139-140`); rationale at `135-138` (uniform abundance avoids ASV-count bias from abundance skew for classifier tuning).

#### Unified three-tier holdout
Module docstring `13-18`: each 100-species community is partitioned into ~33 species per tier:
- **Normal**: all accessions in reference DB (standard ID).
- **Haplotype holdout**: query accession excluded from DB (generalisation).
- **Species holdout**: entire species removed from DB; scored at strict genus level (calibration).

A fourth **genus_holdout** tier appears in `SweepResult` fields (`241,246,256,257`) — `genus_holdout_family_f1`, `n_genus_holdout_species`, `n_genus_holdout_genera` — scored at family rank.

Computed by `select_unified_holdout()` (`3195-3202`) with `UnifiedHoldoutConfig`. Returns `UnifiedHoldoutSplit` with `normal_species`, `haplotype_holdout_species`, `species_holdout_species`, `query_accessions`, `reference_accessions`. The `_tronko_accessions` filter (`3035-3067`) restricts holdout choices to accessions present in tronko's master DB. Unified split is persisted to `ground_truth.json` (`3285`) and restored on cache hit (`3105-3144`).

Filtered reference DBs are written by `write_unified_holdout_reference()` per community (`3401-3409`); per-tool DBs built in parallel by `build_holdout_databases()` at `3447-3476` under `DATA_BASE / "unified_holdout" / community.name / "tool_dbs"`.

A single unified F1 sums TP/FP/FN across all tiers (docstring `18`; `_attach_holdout_metrics` `3653-3687`).

#### Baselines and replicates
`BASELINES = BASELINES_CLASSIFIER` (`141`), imported from `benchmark_shared`. From the docstring (`19-26`): default baselines are `clean | realistic | degraded`. `NUM_REPLICATES` from `benchmark_setup`.

### Component 4 — Cartesian sweep grid for oxidtaxa

Defined at `classifier_benchmark.py:800-992`.

#### Fixed kwargs (`_OXIDTAXA_FIXED`, `821-828`)
```python
"bootstraps": 50,                    # NOT IDTAXA default (100)
"strand": "both",                    # IDTAXA default
"full_length": 0.0,                  # IDTAXA default
"threads": NUM_THREADS,
"min_descend": 0.95,                 # NOT IDTAXA default (0.98 in oxidtaxa; canonical DECIPHER undetermined)
"record_kmers_fraction": 0.10,       # IDTAXA default
```

Notable absences from the fixed dict (so they take their **Rust defaults**, which are IDTAXA-neutralizing): `length_normalize` (false), `rank_thresholds` (None), `beam_width` (1), `seed_pattern` (None), `descendant_weighting` ("count"), `use_idf_in_training` (false), `leave_one_out` (false), `deterministic` (true), `training_threshold` (0.8).

#### Swept lists (full-grid mode, 480 configs total — `919`)
- `_OXIDTAXA_THRESHOLD = [40, 50, 70]` (`834`) — Tier 1 classify-only.
- `_OXIDTAXA_SAMPLE_EXPONENT = [0.47, 0.65]` (`835`).
- `_OXIDTAXA_K = [6, 7, 8, 9, 10]` (`837`) — Tier 2, requires retraining.
- `_OXIDTAXA_CORRELATION_AWARE = [False, True]` (`838`) — Tier 2.
- `_OXIDTAXA_TIE_MARGIN = [0.0, 0.05]` (`842`) — Tier 3 v0.3.0.
- `_OXIDTAXA_CONFIDENCE_USES_DESCENT_MARGIN = [False, True]` (`843`).
- `_OXIDTAXA_SIBLING_AWARE_LEAF = [False, True]` (`844`).

`_OXIDTAXA_K` and `_OXIDTAXA_CORRELATION_AWARE` produce 10 unique trained models (5 × 2). All other axes are classify-time and reuse a model.

#### Smoke values (`852-858`) — 8 configs total (`920`)
`_OXIDTAXA_SMOKE_THRESHOLD=[40,60]`, `_OXIDTAXA_SMOKE_SAMPLE_EXPONENT=[0.40,0.55]`, `_OXIDTAXA_SMOKE_K=[7,9]`, the rest single-valued.

#### Grid generation
`_define_oxidtaxa_cartesian_sweep()` (`911-992`) iterates `itertools.product` of the seven lists. Each combo becomes a `ConfigVariant` (frozen dataclass at `187-200`) with `param_name="grid"`, `param_value=run_id` from `_make_oxidtaxa_run_id` (`889-908`), and `db_variant_key` from `_make_oxidtaxa_db_variant_key` (`861-886`) derived from training params (`k`, `record_kmers_fraction`, `seed_pattern`, `correlation_aware_features`, etc.).

`build_oxidtaxa_variant_dbs()` (`995-1079`) pre-builds one model per unique `db_variant_key` to avoid redundant training.

#### Spaced-seed exclusion
Comments at `814-816, 845-849` document that `seed_pattern` is intentionally NOT in this sweep — it is oxidtaxa-only with no analog in BLAST/SINTAX/IDTAXA/QIIME2, and including it would give oxidtaxa an asymmetric advantage in head-to-head comparison. Spaced seeds are evaluated separately in `oxidtaxa_hyperparameter_tuning.py` (Optuna).

### Component 5 — Metrics

All metrics are columns on `SweepResult` (`202-356`), populated by `_attach_holdout_metrics` (`3653-3687`) which calls `attach_unified_metrics` from `benchmark.metrics_attachment`.

#### Headline scalar
`_selection_score()` (`1910-1933`):
```
score = 0.77 * unified_asv_truth_depth_f1 + 0.23 * unified_asv_pred_depth_phylum_f1
```
Used by `_find_optimal_config_by_difficulty` (`1936`) and best-config selection (`1964, 1990`).

#### Unified species-level (`215-227`)
- `unified_species_f1`, `unified_precision`, `unified_recall` — sibling-aware species F1.
- `unified_classification_f1/precision/recall` — strict + false discovery.
- `unified_bestrank_f1/precision/recall` — partial credit, floor=family.

#### ASV-level unified (`228-237`)
- `unified_asv_species_f1/precision/recall`
- `unified_asv_classification_f1/precision/recall`
- `unified_asv_bestrank_f1/precision/recall`

#### Per-tier (`238-246`)
- `normal_species_f1`, `haplotype_species_f1`, `species_holdout_genus_f1`, `genus_holdout_family_f1`
- `normal_partial_f1`, `haplotype_partial_f1`, `species_holdout_partial_f1`, `genus_holdout_partial_f1` (floor=family)

#### Community ecology (`247-251`)
- `unified_bray_curtis`, `unified_jaccard`
- `unified_holdout_aware_bray_curtis`, `unified_holdout_aware_jaccard` (genus matching for holdout species)

#### Counts (`252-257`)
- `species_classification_fraction`, `n_normal_species`, `n_haplotype_species`, `n_holdout_species_unified`, `n_genus_holdout_species`, `n_genus_holdout_genera`

#### Chimera stratum (`258-264`) — scored separately
- `chimera_asv_count`, `chimera_rejection_rate`, `chimera_over_classification_rate`, `chimera_lca_accuracy`, `chimera_conservative_rate`, `chimera_abundance_impact`

#### IPS, QC effectiveness, off-target (`265-270`)
- `ips_genus`, `qc_biological_retention`, `off_target_fpr`, `off_target_classification_rate`, `off_target_asv_count`

#### Taxon-native ASV-level (`271-292`)
- `unified_asv_truth_depth_f1/precision/recall` — truth's native rank (species; `[genus]X`/`[family]Y` placeholders for holdout tiers force up to genus/family).
- `unified_asv_pred_depth_phylum_f1/precision/recall` — pred-depth, floor=phylum.
- `asv_truth_depth_tier_hmean`, `asv_pred_depth_phylum_tier_hmean`.
- `taxon_native_objective` — single-number Optuna primary (`283`).
- Per-tier breakdown at `285-292`.

#### Taxon-level taxon-native (`293-312`)
- `unified_taxon_truth_depth_f1/precision/recall`, `unified_taxon_pred_depth_phylum_f1/precision/recall`.
- `taxon_truth_depth_tier_hmean`, `taxon_pred_depth_phylum_tier_hmean`.
- `taxon_level_objective` — qc_pipeline_benchmark primary (`303`).
- Per-tier breakdown at `305-312`.

#### Abundance-bin stratification (`313-317`)
- `bin_1_10_species_f1`, `bin_10_50_species_f1`, `bin_50_200_species_f1`, `bin_200_plus_species_f1`.

#### Provenance read-weighted (`318-324`)
- `ips`, `biological_retention_rate`, `chimera_removal_rate`, `off_target_rejection_rate`, `off_target_misclassification_rate`.

### Component 6 — Evaluation harness

#### Per-experiment artifact path
- Per-experiment dir: `DATA_BASE / tool / "experiments" / param_name / param_value / community / baseline` (`1613-1621`).
- Per-cell prediction parquet: `exp_dir / "unified.parquet"` (`3821, 3968, 3974`).
- After scoring, parquet copied via `_copy_to_results(exp_dir / "unified.parquet", DATA_BASE, RESULTS_BASE)` (`3826-3830, 3973-3975`).

#### Aggregated outputs per tool (under `RESULTS_BASE / <tool>/`)
- `classifier_benchmark.json` — full `SweepResult` array, saved incrementally by `_save_partial` (`1587-1593`).
- `best_config_kwargs.json` — `write_best_config_kwargs` (`2158-2185`), consumed by `qc_pipeline_benchmark.py`.
- `optimal_config.json` — written by `generate_tool_report` (`2264-2267`).
- `classifier_benchmark.md` — markdown report (`2389-2391`).
- Plot PNGs from `_plot_param_sensitivity` (`1690`) and `_plot_metrics_heatmap` (`1833`).

There is no global aggregated output file — results are per-tool at `RESULTS_BASE/<tool>/classifier_benchmark.json`.

#### Scoring against ground truth
Ground truth comes from QC output: `qc_out.merged_ground_truth` and `qc_out.merged_observation_counts` (`3767-3768, 3897-3899`). `_attach_holdout_metrics` (`3653-3687`) passes:
- `holdout_split=unified_splits[community]`
- `eval_data` containing `species_siblings`, `species_lca_rank`, `holdout_eval_ranks` (built at `3221-3227`)
- `predictions_parquet=exp_dir / "unified.parquet"`
- `asv_chimera_status`, `asv_chimera_lca_rank` from QC

Actual metric computation lives in `assignment_benchmarks.infrastructure.benchmark.metrics_attachment.attach_unified_metrics`.

### Component 7 — Other tools in the comparison

`VALID_TOOLS` (`163-175`): `blast, tronko, megan, sintax, mmseqs2, idtaxa, raxtax, qiime2-nb, oxidtaxa`. Commented out: `neural` (`165`), `protax` (`173`); `metabuli` removed (`176`).

Per-tool sweep configs in `SWEEP_DEFINITIONS` (`367-572`):
- **idtaxa** (`449-465`): `threshold=[20,40,80,95]`, `bootstraps=[10,50,200]`, `strand`, `min_descend=[0.90,0.95,0.99]`, `full_length`.
- **blast** (`370-411`): `task`, `evalue=[1e-20,1e-50]`, `perc_identity=[90,95,97]`, `max_target_seqs=[10,100,200]`, `min_query_coverage`, `score_type`, `assignment_method` (`top_hit | majority_vote(75%) | majority_vote(90%)`).
- Others: sintax (`412-418`), raxtax (`419-429`), mmseqs2 (`430-448`), megan (`472-516`), qiime2-nb (`534-543`), tronko (`688-798`).

Interaction pairs (`578-598`) include `idtaxa: (threshold,bootstraps), (threshold,min_descend)`.

Per-tool DB paths from `_db_paths_for()` (`2583-2600`); `_build_single_database` dispatcher at `2657-2761`.

## Code References

- `oxidtaxa/src/types.rs:298-353` — `ClassifyConfig` struct + Default impl
- `oxidtaxa/src/types.rs:237-295` — `TrainConfig` struct + Default impl
- `oxidtaxa/src/types.rs:225-234` — `DescendantWeighting` enum, comments mark IDTAXA-original vs added
- `oxidtaxa/src/classify.rs:178-180` — beam dispatch
- `oxidtaxa/src/classify.rs:243-272` — greedy descent + sibling_aware_leaf gate
- `oxidtaxa/src/classify.rs:738-749` — tie_margin in leaf phase
- `oxidtaxa/src/classify.rs:649-662` — length_normalize
- `oxidtaxa/src/classify.rs:858-867` — threshold and rank_thresholds
- `oxidtaxa/src/lib.rs:138-190` — PyO3 classify signature and defaults
- `oxidtaxa/src/lib.rs:64-87` — PyO3 train signature and defaults
- `oxidtaxa/README.md:113-146` — Parameter tables (Classify and Train)
- `oxidtaxa/README.md:345-351` — "Correctness and Divergence from R"
- `assignment_benchmarks/.../oxidtaxa/config.py:48-127` — Python `OxidtaxaConfig` defaults + validators
- `assignment_benchmarks/.../oxidtaxa/executor.py:38-150` — PyO3 in-process invocation, kwargs filtering
- `classifier_benchmark.py:13-26` — community design overview (docstring)
- `classifier_benchmark.py:148-160` — `_build_communities`
- `classifier_benchmark.py:202-356` — `SweepResult` dataclass (every metric column)
- `classifier_benchmark.py:367-572` — `SWEEP_DEFINITIONS` for each tool
- `classifier_benchmark.py:800-992` — oxidtaxa cartesian sweep grid
- `classifier_benchmark.py:861-908` — `_make_oxidtaxa_db_variant_key`, `_make_oxidtaxa_run_id`
- `classifier_benchmark.py:911-992` — `_define_oxidtaxa_cartesian_sweep`
- `classifier_benchmark.py:995-1079` — `build_oxidtaxa_variant_dbs`
- `classifier_benchmark.py:1587-1593` — incremental `_save_partial`
- `classifier_benchmark.py:1910-1933` — `_selection_score` weighting
- `classifier_benchmark.py:2158-2185` — `write_best_config_kwargs`
- `classifier_benchmark.py:3195-3202` — `select_unified_holdout`
- `classifier_benchmark.py:3447-3476` — `build_holdout_databases`
- `classifier_benchmark.py:3653-3687` — `_attach_holdout_metrics`

## Architecture Documentation

The benchmark and oxidtaxa interface in two patterns:

1. **In-process PyO3 invocation**: `executor.py:83` does `from oxidtaxa import classify`; `classify(**kwargs)` is called per-experiment (`executor.py:150`). The executor signature-inspects `classify` (`executor.py:115-117,146`) and silently drops unknown kwargs, which means an older oxidtaxa wheel will not error on new params — it simply ignores them.

2. **Per-variant pre-trained DBs**: oxidtaxa's training params are isolated into `db_variant_key` (`classifier_benchmark.py:861-886`); `build_oxidtaxa_variant_dbs` (`995-1079`) trains each unique combination once, and all classify-only variants reuse those models. `oxidtaxa_db_<variant_key>` directories live under `output_base / oxidtaxa_tuning / unified_holdout / <community> / optuna_models/`.

The `ConfigVariant` dataclass (`classifier_benchmark.py:187-200`) is the unit of work in the sweep: `(tool, param_name, param_value, is_default, is_post_processing, config_kwargs, db_variant_key)`. For oxidtaxa the cartesian generator emits all 480 variants with `param_name="grid"` and `param_value=run_id`.

## Historical Context (from thoughts/)

- `thoughts/shared/research/2026-04-15-new-parameter-audit.md` — earlier audit of the oxidtaxa-added classify parameters; relevant background for this work.
- `thoughts/shared/research/2026-04-21-three-marker-sweep-findings.md` — the three-marker Optuna sweep results referenced in the user's investigation of `sibling_aware_leaf` inertness.
- `thoughts/shared/research/2026-04-19-oxidtaxa-logic-holdout-robustness.md` — holdout-tier robustness analysis.
- `thoughts/shared/research/2026-04-13-r-replication-status.md` — R-vs-oxidtaxa equivalence status, useful for understanding which params are IDTAXA-faithful.
- `thoughts/shared/research/2026-04-13-algorithmic-improvements.md` — design notes for several added classify params.
- `thoughts/shared/research/2026-04-05-rust-idtaxa-parameter-space.md` — original parameter-space scoping doc.
- `thoughts/shared/plans/2026-04-22-full-visual-walkthrough-report.md` — adjacent plan (uncommitted on this branch).

## Related Research

The thoughts/ documents above are the most directly related prior research. The `2026-04-15-new-parameter-audit.md` and `2026-04-21-three-marker-sweep-findings.md` are the closest precedents.

## Open Questions

1. ~~Canonical DECIPHER value for `min_descend`~~ **Resolved 2026-04-27**: canonical R default is `0.98`, confirmed against the `IdTaxa()` signature in `R/IdTaxa.R` (`IdTaxa(..., minDescend=0.98, ...)`). Oxidtaxa's Rust default matches; `classifier_benchmark.py`'s `_OXIDTAXA_FIXED` value of `0.95` is a sweep choice, not the canonical default.
2. ~~Canonical DECIPHER value for `strand`~~ **Resolved 2026-04-27**: canonical R default is `"both"`, confirmed against `R/IdTaxa.R`. Matches oxidtaxa default.
3. The `_OXIDTAXA_FIXED` baseline in `classifier_benchmark.py` pins `bootstraps=50` and `min_descend=0.95`, neither of which matches the IDTAXA-canonical / Rust-default values (100 and 0.98 respectively). Any "test one param against IDTAXA-default backdrop" study using this harness as-is would inherit those non-canonical fixed values unless overridden.

### Canonical DECIPHER defaults (confirmed against R source)

For reference, the full canonical signatures from `R/IdTaxa.R` and `R/LearnTaxa.R`:
```r
IdTaxa(test, trainingSet, type="extended", strand="both",
       threshold=60, bootstraps=100, samples=L^0.47,
       minDescend=0.98, fullLength=0, processors=1, verbose=TRUE)

LearnTaxa(train, taxonomy, rank=NULL, K=NULL, N=500,
          minFraction=0.01, maxFraction=0.06, maxIterations=10,
          multiplier=100, maxChildren=200,
          alphabet=AA_REDUCED[[139]], verbose=TRUE)
```
`recordKmers` is **not** a user-exposed DECIPHER parameter — it appears only as a local in `.createTree`: `recordKmers <- ceiling(max(colSums(profile > 0)))*0.10`. Oxidtaxa exposing `record_kmers_fraction` as a knob is itself a divergence (default `0.10` matches the hardcoded R constant).
