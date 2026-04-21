# Oxidtaxa Optuna sweeps across three markers: findings and recommendations

**Date:** 2026-04-21
**Sweeps analyzed:** Vert12s (≈190 trials), 12S MiFish (≈150 trials), 18S Euk (≈27 trials)
**Harness:** `assignment-tool-benchmarking/projects/assignment_benchmarks/notebooks/oxidtaxa_hyperparameter_tuning.py`
**Analysis notebook:** `…/oxidtaxa_hyperparameter_analysis.ipynb`
**Figures directory:** `./figures/2026-04-21-sweep/` (populate by re-exporting notebook cells — see [Appendix B](#appendix-b-figure-inventory))

---

## 1. Executive summary

Three Optuna TPE sweeps were run against the oxidtaxa classifier on three amplicon markers with different properties (taxonomic scope, amplicon length, reference-DB size). Objective function: harmonic mean of per-tier F1 across 4 unified-holdout tiers (normal, haplotype, species-holdout, genus-holdout), with 85% weight on truth-depth F1 and 15% on pred-depth phylum F1.

**Headline findings:**

1. **One parameter (`rank_threshold_preset`) dominated 83–92% of fANOVA variance on all three markers, but this turned out to be a design artifact of the sweep — not a real algorithmic signal.** The "best" preset (`uniform`) was a null value that disabled per-rank thresholding and deferred to the tunable scalar; the three "gradient" presets were hard-coded vectors with values outside the explored scalar range. Fixed in commit `662495225` (replaced preset categorical with a gated continuous gradient).

2. **Seven parameters are consistent winners across all three markers** — `k=6`, `beam_width=3`, `use_spaced_seed=True`, `use_idf_in_training=False`, `confidence_uses_descent_margin=False`, `rank_threshold_preset=uniform` (now equivalent to `use_rank_gradient=False`), `threshold ∈ [50, 57]`. These are candidates for reduction from the sweep space.

3. **Five parameters flip sign between markers**, correlated with reference-DB scope (sequences × taxonomic breadth): `descendant_weighting`, `leave_one_out`, `correlation_aware_features`, `sibling_aware_leaf`, `tie_margin`, `record_kmers_fraction`. These are genuine marker-dependent effects worth understanding, not noise.

4. **Holdout-robustness drives the objective** on 12S markers. In-distribution F1 is near-saturated; nearly all optimizer pressure is on species/genus-holdout tiers. The consistent recipe on holdouts is: low `threshold`, high `min_descend`, high `tie_margin`, uniform per-rank thresholds, no descent-margin-based confidence — essentially "be willing to commit, then fall back deep when leaf is absent." 18S breaks this pattern (see §5).

5. **Two red flags in the objective function itself**: (a) on MiFish, off-target classification rate correlates with objective at **r=+0.89**, meaning the optimizer is partly rewarding off-target false positives; (b) objective ceilings (0.34 on 18S, 0.51 on Vert12s, 0.62 on MiFish) look clipped and may reflect an upper-bound bug rather than a Pareto frontier.

6. **Actions already taken:** the rank_threshold_preset design was fixed upstream (see §7). Follow-up recommendations in §8 cover sweep-space reduction, `suggest_int` for ordered params, a `beam_width` probe, and an objective audit.

---

## 2. Method

### 2.1 Objective

The sweep optimizes `taxon_native_objective` (see commit `8f63b8828` in assignment-tool-benchmarking):

```
objective = 0.85 * HM(per-tier truth_depth F1) + 0.15 * HM(per-tier pred_depth_phylum F1)
```

over the 4 unified-holdout tiers:

- **normal** — queries drawn from species present in the reference DB
- **haplotype** — same-species queries with DB-absent haplotype variants
- **species_holdout** — queries from species absent from the reference (target genus still present)
- **genus_holdout** — queries from genera absent from the reference (family/above still present)

The harmonic mean zeroes when any tier collapses, so a configuration cannot win by sacrificing a single tier (e.g., over-fitting the normal tier at the cost of species_holdout).

### 2.2 Markers and reference databases

| Marker | DB size | Median ref length | Taxonomic scope |
|---|---|---|---|
| **Vert12s** (12SV5) | 174,428 | 156 bp | All vertebrates |
| **12S MiFish U** | 59,459 | 172 bp | Fish (narrow sub-clade) |
| **18S Euk** | 320,554 | 107 bp | All eukaryotes (very broad) |

Reference paths: `~/rcrux-py/databases/{vert12S,12S_MiFish_U,18S_Euk}/filtered/*_species.fasta`.

### 2.3 Search space (as swept)

Training params: `k ∈ {6, 7, 8, 9}`, `record_kmers_fraction ∈ [0.03, 0.50]`, `training_threshold ∈ [0.50, 0.99]`, `descendant_weighting ∈ {count, equal, log}`, `use_idf_in_training`, `leave_one_out`, `correlation_aware_features`, `use_spaced_seed` (+ conditional `seed_pattern_k{k}`).

Classification params: `threshold ∈ [50, 80]`, `beam_width ∈ {1, 2, 3}`, `min_descend ∈ [0.75, 0.99]`, `sample_exponent ∈ [0.30, 0.70]`, `tie_margin ∈ [0.00, 0.10]`, `sibling_aware_leaf`, `confidence_uses_descent_margin`, `rank_threshold_preset ∈ {uniform, strict_top, moderate_gradient, lenient_gradient}`.

Trial counts: Vert12s ≈190, MiFish ≈150, 18S ≈27. **The 18S sweep is notably under-sampled** — conclusions are tentative and any "flipped sign" findings on 18S should be treated as hypotheses pending a longer run.

---

## 3. Marker 1: Vert12s

### 3.1 Reference profile

174,428 sequences, median 156 bp. Broad vertebrate coverage — dense per-genus depth in well-studied clades, sparse in others.

### 3.2 Best configuration

Objective ceiling **~0.51**. Best trial (red circle in pairwise-params figure):

| Param | Value | Category |
|---|---|---|
| `k` | 6 | training |
| `beam_width` | 3 | classify |
| `threshold` | 55 | classify |
| `min_descend` | 0.95 | classify |
| `training_threshold` | 0.66 | training |
| `record_kmers_fraction` | 0.38 | training |
| `sample_exponent` | 0.51 | classify |
| `tie_margin` | 0.09 | classify |
| `descendant_weighting` | count | training |
| `correlation_aware_features` | True | training |
| `use_spaced_seed` | True | training |
| `leave_one_out` | True | training |
| `use_idf_in_training` | False | training |
| `rank_threshold_preset` | uniform | classify |
| `confidence_uses_descent_margin` | False | classify |
| `sibling_aware_leaf` | False | classify |

![Continuous params vs objective, Vert12s](figures/2026-04-21-sweep/01-vert12s-continuous.png)
![Categorical params vs objective, Vert12s](figures/2026-04-21-sweep/02-vert12s-categorical.png)
![Optimizer sampling over time, Vert12s](figures/2026-04-21-sweep/03-vert12s-sampling.png)
![Parameter importance (fANOVA), Vert12s](figures/2026-04-21-sweep/04-vert12s-importance.png)
![Pairwise continuous params, Vert12s](figures/2026-04-21-sweep/05-vert12s-pairwise.png)

### 3.3 Parameter importance

`rank_threshold_preset` at ~0.92, `threshold` at ~0.05, all else at noise floor. See §6.1 for why this is a design artifact and not a real signal.

### 3.4 Continuous params

| Param | Pearson r | Interpretation |
|---|---|---|
| `min_descend` | +0.46 | push toward ≥0.93 |
| `threshold` | −0.44 | lower is better (~55) |
| `tie_margin` | +0.41 | higher is better (0.09–0.10) |
| `training_threshold` | −0.29 | moderate (0.55–0.70) |
| `record_kmers_fraction` | +0.25 | moderate (0.30–0.40) |
| `sample_exponent` | −0.06 | weak |

### 3.5 Categorical params

Clear winners (median objective in parens): `k=6` (~0.50), `beam_width=3` (~0.50), `rank_threshold_preset=uniform` (~0.49), `descendant_weighting=count`, `correlation_aware_features=True`, `leave_one_out=True`, `use_spaced_seed=True`, `use_idf_in_training=False`, `confidence_uses_descent_margin=False`, `sibling_aware_leaf=False`.

`rank_threshold_preset=strict_top` catastrophic (median ~0.15).

### 3.6 Pipeline metrics

![Pipeline metrics vs objective, Vert12s](figures/2026-04-21-sweep/06-vert12s-pipeline-metrics.png)

- **IPS(genus)** r=+0.44, IPS(species) r=+0.19, classification_rate r=+0.30 — expected positive correlations.
- **Off-target FPR pinned at 0** for every trial — Vert12s's broad scope means few queries are off-target.
- Off-target classification rate r=+0.64 — rates are tiny (0–1.8%) so not operationally concerning, but the sign hints at the MiFish issue (see §4.6).
- Bray-Curtis r=−0.19 (lower dissimilarity = better, as expected).

### 3.7 Per-tier decomposition

![Continuous params × tier F1 heatmap, Vert12s](figures/2026-04-21-sweep/07-vert12s-tier-continuous.png)
![Categorical param×level × tier lift heatmap, Vert12s](figures/2026-04-21-sweep/08-vert12s-tier-categorical.png)

The holdout tiers drive the sweep:

- `threshold` r: −0.10 normal, −0.40 species_hol, **−0.49 genus_hol**
- `min_descend` r: +0.05 normal, +0.43 species_hol, **+0.51 genus_hol**
- `tie_margin` r: +0.19 normal, +0.41 species_hol, **+0.47 genus_hol**

Categorical lift on genus_holdout (ΔF1 from tier mean): `rank_threshold_preset=strict_top` **−0.170**, `rank_threshold_preset=uniform` +0.090, `k=6` +0.060, `beam_width=3` +0.061, `confidence_uses_descent_margin=True` **−0.091**.

### 3.8 Vert12s takeaways

- **Holdout strategy**: low threshold + high min_descend + high tie_margin + uniform preset. The recipe is "be willing to commit confidently but fall back deep to a well-resolved parent when the true leaf is missing."
- **In-distribution saturated**: normal-tier F1 varies little across configs. Only `sample_exponent` has meaningful normal-tier correlation (+0.29).
- **`strict_top` preset** is the biggest single killer — its Root threshold of 90 rejects classifications even when species-level confidence is high.

---

## 4. Marker 2: 12S MiFish U

### 4.1 Reference profile

59,459 sequences, median 172 bp. Fish-only — narrow taxonomic scope, moderate per-species depth.

### 4.2 Best configuration

Objective ceiling **~0.62** — highest of the three markers (fish coverage within this narrow DB is dense).

| Param | Value | vs Vert12s |
|---|---|---|
| `k` | 6 | same |
| `beam_width` | 3 | same |
| `threshold` | 54 | same region |
| `min_descend` | 0.93 | same region |
| `training_threshold` | 0.57 | lower (Vert12s 0.66) |
| `record_kmers_fraction` | **0.25** | **lower** (Vert12s 0.38) |
| `sample_exponent` | 0.49 | same region |
| `tie_margin` | **0.03** | **lower** (Vert12s 0.09) — *flipped* |
| `descendant_weighting` | count | same |
| `correlation_aware_features` | **False** | ***flipped*** |
| `leave_one_out` | **False** | ***flipped*** |
| `use_spaced_seed` | True | same |
| `use_idf_in_training` | False | same |
| `rank_threshold_preset` | uniform | same |
| `confidence_uses_descent_margin` | False | same |
| `sibling_aware_leaf` | **True** | ***flipped*** |

![Continuous params vs objective, MiFish](figures/2026-04-21-sweep/09-mifish-continuous.png)
![Categorical params vs objective, MiFish](figures/2026-04-21-sweep/10-mifish-categorical.png)
![Optimizer sampling over time, MiFish](figures/2026-04-21-sweep/11-mifish-sampling.png)
![Parameter importance (fANOVA), MiFish](figures/2026-04-21-sweep/12-mifish-importance.png)
![Pairwise continuous params, MiFish](figures/2026-04-21-sweep/13-mifish-pairwise.png)

### 4.3 Parameter importance

`rank_threshold_preset` ~0.83, `threshold` ~0.13, all else trivial. Same design-artifact dynamic as Vert12s.

### 4.4 Continuous params (stronger signals than Vert12s)

| Param | MiFish r | Vert12s r |
|---|---|---|
| `threshold` | **−0.57** | −0.44 |
| `training_threshold` | **−0.48** | −0.29 |
| `min_descend` | +0.32 | +0.46 |
| `tie_margin` | **−0.18** | +0.41 (*flipped*) |
| `record_kmers_fraction` | **−0.14** | +0.25 (*flipped*) |
| `sample_exponent` | −0.07 | −0.06 |

### 4.5 Categorical flips

The flips from Vert12s → MiFish correlate with reference-DB scope:

- **`leave_one_out=False`** — with only 59k fish sequences, leaving one out creates large coverage gaps per species. Vert12s has enough density to absorb LOO.
- **`correlation_aware_features=False`** — feature correlations may amplify noise at smaller DB sizes.
- **`sibling_aware_leaf=True`** — fish leaf taxa are tightly clustered (congeneric species share many kmers), so sibling-aware disambiguation pays off; in Vert12s's broader tree it fires spuriously.
- **Lower `tie_margin` / `record_kmers_fraction`** — smaller scope, fewer genuine ties, less room for aggressive kmer recording.

### 4.6 Pipeline metrics — ⚠ objective-validity concern

![Pipeline metrics vs objective, MiFish](figures/2026-04-21-sweep/14-mifish-pipeline-metrics.png)

| Metric | r with objective | Range at top configs |
|---|---|---|
| **Off-target Classification Rate** | **+0.89** | 0.18 – 0.22 |
| **Off-target FPR** | **+0.72** | 0.09 – 0.11 |
| IPS(genus) | +0.74 | 0.70 – 0.75 |
| Classification Rate | +0.73 | 0.55 – 0.70 |
| IPS(species) | +0.27 | 0.45 – 0.48 |
| Bray-Curtis | −0.26 | — |

**At the top-objective configs, the classifier calls ~22% of off-target queries with ~11% FPR.** Because MiFish has a narrow-scope DB (fish only) but the evaluation queries include broader vertebrate off-target sequences, every parameter choice that pushes the classifier to "commit" (low threshold, uniform preset, wider beam) simultaneously boosts IPS *and* boosts off-target FPR.

The species-IPS correlation with the objective is only +0.27 while off-target rate correlation is +0.89 — the optimizer is mostly solving "how do I classify more off-target things?", not "how do I do fish taxonomy better." Any conclusions about "best" MiFish params should be validated with an off-target-penalized objective. See §6.3 for recommendation.

### 4.7 Per-tier decomposition

![Continuous params × tier F1 heatmap, MiFish](figures/2026-04-21-sweep/15-mifish-tier-continuous.png)
![Categorical param×level × tier lift heatmap, MiFish](figures/2026-04-21-sweep/16-mifish-tier-categorical.png)

Continuous params on holdouts (Pearson r, species_hol / genus_hol):

- `threshold` −0.53 / **−0.59** (stronger than Vert12s)
- `training_threshold` −0.45 / **−0.51** (much stronger than Vert12s's −0.26 / −0.32)
- `min_descend` +0.31 / +0.34
- `tie_margin` −0.14 / −0.20 *(flipped from Vert12s)*

Categorical lift on genus_hol: `rank_threshold_preset=uniform` **+0.144** (strongest cross-marker lift observed), `strict_top` **−0.173**, `confidence_uses_descent_margin=True` **−0.115**, `k=6` +0.117, `beam_width=3` +0.091, `leave_one_out=True` −0.099 *(flipped)*, `correlation_aware_features=True` −0.079 *(flipped)*.

### 4.8 MiFish takeaways

- Same holdout-robustness strategy as Vert12s, but `training_threshold` matters much more here.
- **DB scope explains the flipped categoricals**: small-narrow DBs want `LOO=False, correlation_aware=False, sibling_aware=True, tie_margin low`; large-broad DBs want the opposite.
- **Objective is confounded with off-target calling** (r=+0.89). Needs auditing before drawing firm MiFish-specific conclusions.

---

## 5. Marker 3: 18S Euk

### 5.1 Reference profile

320,554 sequences (largest DB), median 107 bp (shortest). Broadest marker — all eukaryotes, not just a sub-clade. Short amplicon × enormous taxonomic scope × sparse per-taxon depth → hardest classification problem of the three.

### 5.2 Sweep status

**Only ~27 trials.** Optuna-TPE has not converged. Late trials still distribute roughly uniformly across each param's range — no green cluster forms. Conclusions below are tentative; need 150+ trials to trust.

No pipeline-metrics plot was generated for this marker, so off-target behavior can't be audited from the current sweep outputs.

### 5.3 Best configuration (tentative)

Objective ceiling **~0.34** — half of MiFish, two-thirds of Vert12s. 18S is genuinely harder.

| Param | Best | vs Vert12s/MiFish |
|---|---|---|
| `k` | 6 (or 8 — tied) | consistent |
| `beam_width` | 3 | consistent |
| `rank_threshold_preset` | uniform | consistent |
| `threshold` | ~51–55 | consistent |
| `min_descend` | **~0.75** | ***flipped*** — low |
| `record_kmers_fraction` | **~0.05–0.17** | ***lowest yet*** |
| `sample_exponent` | ~0.57 | similar |
| `tie_margin` | 0.03–0.04 | similar to MiFish |
| `descendant_weighting` | **equal** | ***flipped*** — equal > count > log |
| `leave_one_out` | True | same as Vert12s |
| `use_spaced_seed` | True | same |
| `confidence_uses_descent_margin` | False | same |
| `training_threshold` | flat signal | ***flat*** (r=0.00) |

![Continuous params vs objective, 18S](figures/2026-04-21-sweep/17-18s-continuous.png)
![Categorical params vs objective, 18S](figures/2026-04-21-sweep/18-18s-categorical.png)
![Optimizer sampling over time, 18S](figures/2026-04-21-sweep/19-18s-sampling.png)
![Parameter importance (fANOVA), 18S](figures/2026-04-21-sweep/20-18s-importance.png)

### 5.4 Continuous params

| Param | 18S r | MiFish r | Vert12s r |
|---|---|---|---|
| `record_kmers_fraction` | **−0.30** | −0.14 | +0.25 (*flipped vs Vert12s*) |
| `threshold` | −0.26 | −0.57 | −0.44 |
| `tie_margin` | +0.46 | −0.18 | +0.41 |
| `training_threshold` | **−0.00** | −0.48 | −0.29 |
| `min_descend` | **−0.14** | +0.32 | +0.46 (*flipped*) |
| `sample_exponent` | −0.17 | −0.07 | −0.06 |

### 5.5 Per-tier decomposition

![Continuous params × tier F1 heatmap, 18S](figures/2026-04-21-sweep/21-18s-tier-continuous.png)
![Categorical param×level × tier lift heatmap, 18S](figures/2026-04-21-sweep/22-18s-tier-categorical.png)

Most striking finding: **`record_kmers_fraction` has +0.46 on normal but −0.44 on species_holdout** — opposite signs, near-equal magnitudes. The other markers had this param weakly positive across all tiers. 18S has a real in-distribution / holdout tradeoff that wasn't present on 12S.

Similarly, `min_descend` is slightly positive on normal (+0.06) but **negative on holdouts** (−0.19, −0.14) — opposite of 12S markers (which want high min_descend on holdouts).

### 5.6 18S takeaways (tentative)

- **Different holdout strategy than 12S.** Deep descent hurts 18S holdouts — probably because the tree has long internal branches and sparse genus-level sampling, so forcing descent pushes calls into wrong neighborhoods. Letting the classifier stop at shallower ranks works better.
- **`descendant_weighting=equal` wins** — plausibly because 18S has massive descendant-count variance across the eukaryotic tree (metazoans vs. rare protist lineages); `count` weighting biases toward dense clades.
- **`training_threshold` signal is flat** — the one continuous param with no measurable effect. On 12S markers this was a −0.29 to −0.48 correlation. Worth understanding why the threshold that controls training-time ambiguity suppression stops mattering on 18S.
- **More trials needed** before any of these are trustworthy.

---

## 6. Design issues surfaced by the sweeps

### 6.1 `rank_threshold_preset` was a null-value knob, not a shape knob

`_RANK_THRESHOLD_PRESETS` as swept:

```python
{
    "uniform":           None,                                 # ← disables per-rank, defers to scalar threshold
    "strict_top":        [90, 80, 70, 60, 50, 40, 40],
    "moderate_gradient": [80, 70, 60, 50, 40, 30, 30],
    "lenient_gradient":  [70, 60, 50, 40, 30, 20, 20],
}
```

Why this dominates fANOVA at 83–92% despite being a 4-level categorical:

1. **`uniform` is literally `None`** — a null value, not a shape. When Optuna samples `uniform`, `rank_thresholds=None` and the classifier defers to the tunable scalar `threshold ∈ [50, 80]` (which converges near 54). So the 4-level knob is effectively a 2-level knob: "use the tunable threshold" vs "use one of three hard-coded vectors."

2. **The three non-uniform presets are strawmen.** `threshold` explores `[50, 80]` but the gradient presets impose Root thresholds of 70/80/90 — at or above the top of the explored scalar range. IDTAXA confidences are monotonically non-decreasing toward Root (per `src/types.rs:312`: "depth i (0=Root)"), and `src/classify.rs:854-857` breaks the `above` prefix at the first failing rank. So a too-strict Root threshold terminates the call entirely, even when deep-rank confidence would pass.

3. **The sweep never tested whether per-rank thresholding helps.** It only tested whether three arbitrary hard-coded vectors help, at values Optuna couldn't tune.

**Fixed in `assignment-tool-benchmarking` commit `662495225`**: replaced the preset categorical with a gated continuous gradient:

```python
use_rank_gradient = trial.suggest_categorical("use_rank_gradient", [False, True])
rank_thresholds: list[float] | None = None
if use_rank_gradient:
    threshold_root_offset = trial.suggest_float("threshold_root_offset", 1.0, 25.0, step=1.0)
    rank_thresholds = [
        threshold + threshold_root_offset * ((6 - i) / 6.0)
        for i in range(7)
    ]
```

Gate-off reproduces the old `uniform` path exactly; gate-on samples a strictly-positive Root offset and linearly interpolates across the 7 ranks. The `threshold` range was also narrowed from `[50, 80]` to `[50, 60]` per cross-marker evidence.

### 6.2 `confidence_uses_descent_margin=True` consistently hurts

Lift on genus_holdout when `confidence_uses_descent_margin=True`:
- Vert12s: −0.091
- MiFish: −0.115
- 18S: −0.007

The feature was re-enabled in the sweep (after a pause for bug fixes — oxidtaxa commits `79ecdb9`, `17b492b`, `e287eb7`), and still loses on every marker with measurable signal. Either the feature has another latent bug or its motivation is wrong for these objectives. See §8.4.

### 6.3 Objective reward for off-target classification on narrow DBs

On MiFish (fish-only DB with broader evaluation queries), off-target classification rate correlates with objective at r=+0.89. Every parameter that pushes the classifier to "commit" inflates both IPS *and* off-target FPR, and the current objective doesn't distinguish. On Vert12s (broad-scope DB), off-target FPR was pinned at 0 so the confound didn't manifest.

Worth checking if 18S has the same issue once pipeline metrics are regenerated.

### 6.4 Objective ceilings look clipped

Per-marker objective upper bounds: **0.34 on 18S, 0.51 on Vert12s, 0.62 on MiFish**. Late trials stack exactly on these values (see sampling-over-time figures: large green clusters at constant objective). For F1-based metrics, a ceiling near 0.5 is unusually round. Possible causes:

- Harmonic mean of 4 tier F1s: if one tier caps at ~0.2 (e.g., genus_holdout is genuinely hard), HM drags the whole objective down. This is expected behavior and not a bug.
- An averaging weight that effectively halves an F1 component.
- A metric denominator normalization issue.

Worth confirming before the next round of sweeps that 0.51 is a real Pareto frontier and not a bug-induced cap.

---

## 7. Cross-marker synthesis

### 7.1 Robust winners (consistent across all three markers)

These are pin-candidates — no marker ever prefers an alternative:

| Param | Consensus winner | Evidence |
|---|---|---|
| `rank_threshold_preset` | **uniform** *(post-fix: `use_rank_gradient=False`)* | dominates on every marker, every tier |
| `k` | **6** | holdout lift +0.060 / +0.117 / +0.005 |
| `beam_width` | **3** | holdout lift +0.061 / +0.091 / +0.018 |
| `use_spaced_seed` | **True** | consistent slight edge |
| `use_idf_in_training` | **False** | lift True vs False: −0.042 / −0.066 / −0.003 |
| `confidence_uses_descent_margin` | **False** | lift True vs False: −0.091 / −0.115 / −0.007 |
| `rank_threshold_preset=strict_top` | **always loses** | lift −0.170 / −0.173 / −0.029 |
| `threshold` | **~50–57** | optimum at 55 / 54 / 51–55 |

### 7.2 Marker-dependent parameters (flipped signs)

| Param | Vert12s | MiFish | 18S | Pattern |
|---|---|---|---|---|
| `descendant_weighting` | count | count | **equal** | depends on tree-balance |
| `leave_one_out` | True | **False** | True | depends on per-taxon depth |
| `correlation_aware_features` | True | **False** | neutral | depends on DB size |
| `sibling_aware_leaf` | False | **True** | neutral | depends on sibling density |
| `tie_margin` | 0.09 | **0.03** | 0.04 | scales with scope-breadth |
| `record_kmers_fraction` | 0.38 | 0.25 | **0.05** | inversely with scope |
| `min_descend` | 0.95 | 0.93 | **0.75** | 12S wants deep; 18S shallow |
| `training_threshold` | −0.29 r | −0.48 r | **0.00 r** | matters on 12S, flat on 18S |

### 7.3 The DB-scope hypothesis

The flips roughly align with reference-DB scope (breadth × depth):

- **Narrow / shallow** (MiFish: 59k seq, fish-only): `LOO=False` (avoid creating gaps), `correlation_aware=False` (low sample support), `sibling_aware=True` (tight clusters), low `tie_margin`/`record_kmers_fraction` (fewer ambiguous cases).
- **Moderate / deep** (Vert12s: 174k seq, all vertebrates): middle settings.
- **Broad / very sparse** (18S: 320k seq, all eukaryotes with short amplicons): even lower `record_kmers_fraction` (many distant lineages; specific kmers dilute), `descendant_weighting=equal` (correct for clade-density imbalance), low `min_descend` (shallow tree navigation).

The objective ceiling also scales with scope-difficulty: MiFish 0.62 > Vert12s 0.51 > 18S 0.34 — denser per-taxon coverage → higher ceiling.

### 7.4 Holdout-robustness dominates the 12S objective

In-distribution (normal tier) F1 is near-saturated on 12S markers — most parameters have |r|<0.2 with normal-tier F1. Holdout F1 is what the sweep is actually optimizing:

- Threshold correlations with genus_hol F1: −0.49 (Vert12s), −0.59 (MiFish), −0.32 (18S)
- Min_descend correlations: +0.51, +0.34, −0.14
- Tie_margin correlations: +0.47, −0.20, +0.47

On 12S, the winning holdout strategy is the "commit confidently, fall back deep" recipe. On 18S, this strategy breaks — shallow fallback wins. This is the single most important cross-marker difference and should inform future per-marker tuning.

---

## 8. Recommendations

### 8.1 Completed

✅ Replace `rank_threshold_preset` with gated gradient (`662495225`).
✅ Narrow `threshold` to `[50, 60]` (`662495225`).

### 8.2 Immediate follow-ups (low-risk sweep-space reductions)

1. **Pin consensus winners** to free trial budget for genuinely-uncertain knobs. The following are sweep-space dead weight given the evidence:
   - `k`: pin at `6` or narrow to `suggest_int("k", 5, 8)` (drop 9, add 5 — see §8.3)
   - `beam_width`: pin at `3` or widen to `suggest_int("beam_width", 1, 6)` (see §8.3)
   - `use_spaced_seed`: pin at `True`
   - `use_idf_in_training`: pin at `False`
   - `confidence_uses_descent_margin`: pin at `False` *pending investigation — see §8.4*

2. **Switch ordered categoricals to `suggest_int`**: `k` and `beam_width` both have natural numeric ordering that the categorical sampler ignores. TPE's density estimation across an integer range converges faster than over unordered categories. See §8.3.

3. **Don't pin marker-dependent flips**: `descendant_weighting`, `leave_one_out`, `correlation_aware_features`, `sibling_aware_leaf`, `tie_margin`, `record_kmers_fraction`, `min_descend` should stay in the sweep. They're real effects; a future feature could auto-configure them from ref-DB stats.

### 8.3 Probe recommendations

**`beam_width`**: the 1 → 2 → 3 trend is monotone-improving on all three markers with no sign of saturation. Worth testing 4 and 5. Recommended approach: run a small 10-trial probe on Vert12s and MiFish with everything else pinned at consensus winners, varying only `beam_width ∈ {3, 4, 5}`. Then decide whether to widen the range or pin at 3.

**`k`**: the same monotone trend (k=6 > 7 > 8 > 9 on holdouts) suggests the optimum may sit *below* 6. Worth extending to `suggest_int("k", 5, 8)` — drop k=9 (48 trials agree it loses) and add k=5. Be aware k=5 is unusually low for IDTAXA-style methods (vocabulary 4⁵=1024, low specificity) so spot-check top-k=5 trials for degenerate predictions.

### 8.4 `confidence_uses_descent_margin` investigation

This feature loses on every marker with measurable signal, despite three recent bug-fix commits in oxidtaxa (`79ecdb9`, `17b492b`, `e287eb7`) that addressed known issues. Options:

- **Option A**: run a dedicated 2x2x2 grid (margin on/off × beam_width × marker) with other params pinned, 20 trials per cell. If margin=True never wins even in the controlled grid, remove the feature from the codebase.
- **Option B**: investigate whether there's a further latent bug (e.g., margin being applied at the wrong scale in the beam path). Check if the margin value range is appropriate for the observed confidence distributions.

### 8.5 Objective audit (highest-priority)

Two concerns worth resolving before the next full sweep:

1. **Off-target confound on narrow-scope markers.** Add an explicit off-target penalty term to `taxon_native_objective`, or switch to a precision-weighted variant. Current form is effectively rewarding over-calling on MiFish (r=+0.89 with off-target rate).

2. **Investigate the objective ceilings** (0.34 / 0.51 / 0.62). Plateau behavior suggests either a genuine Pareto frontier or an upper-bound bug. Worth running one trial with `f1` terms logged separately to confirm each component is variable rather than some being clipped.

### 8.6 18S needs a longer sweep

The 27-trial 18S run has not converged. Re-run with ≥150 trials before trusting the flipped signs (`descendant_weighting=equal`, low `min_descend`, flat `training_threshold`, bimodal `record_kmers_fraction`). Also regenerate pipeline-metrics plots for the off-target audit.

### 8.7 Longer-term: marker-topology-aware defaults

The DB-scope hypothesis (§7.3) suggests a path to automated per-marker configuration:
- Compute tree-balance, per-taxon-depth, scope-breadth from the reference file at training time.
- Map those statistics to defaults for the flipped parameters.

This removes ~5 knobs from the sweep space entirely. Out of scope for this quarter; worth a design doc once the above items are resolved.

---

## Appendix A: Consolidated parameter reference

| Param | Vert12s | MiFish | 18S | Action |
|---|---|---|---|---|
| `k` | 6 | 6 | 6 or 8 | `suggest_int(5, 8)` |
| `beam_width` | 3 | 3 | 3 | `suggest_int(1, 6)` or pin 3 |
| `rank_threshold_preset` | uniform | uniform | uniform | **removed** (gated gradient) |
| `threshold` | 55 | 54 | 51–55 | narrowed to [50, 60] ✅ |
| `min_descend` | 0.95 | 0.93 | 0.75 | keep (marker-dependent) |
| `training_threshold` | 0.66 | 0.57 | flat | narrow to [0.50, 0.70] |
| `record_kmers_fraction` | 0.38 | 0.25 | 0.05 | keep (marker-dependent) |
| `sample_exponent` | 0.51 | 0.49 | 0.57 | keep (weak but only in-dist signal) |
| `tie_margin` | 0.09 | 0.03 | 0.04 | keep (marker-dependent) |
| `descendant_weighting` | count | count | **equal** | keep (marker-dependent) |
| `correlation_aware_features` | True | **False** | neutral | keep (marker-dependent) |
| `use_spaced_seed` | True | True | True | pin `True` |
| `use_idf_in_training` | False | False | False | pin `False` |
| `leave_one_out` | True | **False** | True | keep (marker-dependent) |
| `confidence_uses_descent_margin` | False | False | False | pin `False` pending §8.4 |
| `sibling_aware_leaf` | False | **True** | neutral | keep (marker-dependent) |

## Appendix B: Figure inventory

All figures live under `figures/2026-04-21-sweep/`. The sweep produced these via cells in `oxidtaxa_hyperparameter_analysis.ipynb` but they weren't persisted as PNGs. To populate this directory, re-run the notebook against the saved Optuna studies (at `projects/assignment_benchmarks/data/benchmark_runs/{marker}/oxidtaxa_tuning/optuna_study.db`) and export each matplotlib figure with `plt.savefig(f"figures/2026-04-21-sweep/{name}.png", dpi=120, bbox_inches="tight")`.

| # | Filename | Content |
|---|---|---|
| 01 | `01-vert12s-continuous.png` | Vert12s: 6 scatter panels of continuous params vs objective, with Pearson r |
| 02 | `02-vert12s-categorical.png` | Vert12s: 10 box-plot panels of categorical params vs objective |
| 03 | `03-vert12s-sampling.png` | Vert12s: 6 panels showing optimizer sampling density over trial number |
| 04 | `04-vert12s-importance.png` | Vert12s: fANOVA parameter importance bar chart |
| 05 | `05-vert12s-pairwise.png` | Vert12s: 5×3 grid of pairwise continuous-param scatter plots (best trial circled) |
| 06 | `06-vert12s-pipeline-metrics.png` | Vert12s: 6 panels of pipeline metrics (IPS, off-target FPR, etc.) vs objective |
| 07 | `07-vert12s-tier-continuous.png` | Vert12s: heatmap of continuous params × per-tier F1 (Pearson r) |
| 08 | `08-vert12s-tier-categorical.png` | Vert12s: heatmap of categorical param×level × per-tier F1 lift |
| 09 | `09-mifish-continuous.png` | MiFish: continuous params vs objective |
| 10 | `10-mifish-categorical.png` | MiFish: categorical params vs objective |
| 11 | `11-mifish-sampling.png` | MiFish: sampling density over time |
| 12 | `12-mifish-importance.png` | MiFish: fANOVA importance |
| 13 | `13-mifish-pairwise.png` | MiFish: pairwise continuous scatter grid |
| 14 | `14-mifish-pipeline-metrics.png` | MiFish: pipeline metrics vs objective — includes the r=+0.89 off-target panel |
| 15 | `15-mifish-tier-continuous.png` | MiFish: continuous × tier heatmap |
| 16 | `16-mifish-tier-categorical.png` | MiFish: categorical × tier lift heatmap |
| 17 | `17-18s-continuous.png` | 18S: continuous params vs objective |
| 18 | `18-18s-categorical.png` | 18S: categorical params vs objective |
| 19 | `19-18s-sampling.png` | 18S: sampling density (note: unconverged) |
| 20 | `20-18s-importance.png` | 18S: fANOVA importance |
| 21 | `21-18s-tier-continuous.png` | 18S: continuous × tier heatmap (record_kmers_fraction sign flip visible here) |
| 22 | `22-18s-tier-categorical.png` | 18S: categorical × tier lift heatmap |

**Optional: auto-save from the notebook.** Add a cell like this at the end of each per-marker section to persist outputs automatically:

```python
import matplotlib.pyplot as plt
from pathlib import Path

fig_dir = Path("~/oxidtaxa/thoughts/shared/research/figures/2026-04-21-sweep").expanduser()
fig_dir.mkdir(parents=True, exist_ok=True)

# After each figure is generated:
plt.savefig(fig_dir / "01-vert12s-continuous.png", dpi=120, bbox_inches="tight")
```

Future sweeps will then populate the figures directory without manual export.

## Appendix C: References

- **Sweep harness** (post-fix): `assignment-tool-benchmarking/projects/assignment_benchmarks/notebooks/oxidtaxa_hyperparameter_tuning.py`
- **Analysis notebook**: `…/oxidtaxa_hyperparameter_analysis.ipynb`
- **Optuna studies**: `…/data/benchmark_runs/{marker}/oxidtaxa_tuning/optuna_study.db`
- **Oxidtaxa source**:
  - `src/classify.rs:844-858` — per-rank threshold gate
  - `src/types.rs:312-314` — `rank_thresholds` field definition
  - `src/lib.rs:135-167` — PyO3 `classify()` signature
- **Prior research**:
  - `thoughts/shared/research/2026-04-15-new-parameter-audit.md` — original `rank_thresholds` bug audit
  - `thoughts/shared/research/2026-04-19-oxidtaxa-logic-holdout-robustness.md` — holdout-robustness deep-dive
  - `thoughts/shared/research/2026-04-20-oxidtaxa-sweep-state.md` — prior sweep state (referenced by commit `8f63b8828`)
- **Commit**: `662495225` (`assignment-tool-benchmarking/assignment-tool-benchmarking`) — replaced `rank_threshold_preset` with gated gradient
