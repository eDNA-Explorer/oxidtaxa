---
date: 2026-04-05
researcher: Claude
topic: "Rust IDTAXA parameter space for config sweep"
tags: [research, idtaxa, parameters, config-sweep, classification]
status: complete
last_updated: 2026-04-05
---

# Research: Rust IDTAXA Parameter Space for Config Sweep

## Research Question

What parameters are available for a parameter sweep on the Rust IDTAXA implementation? Are there additional parameters we could add that might improve classification?

## Summary

The Rust IDTAXA implementation has **11 Python-exposed parameters**, **6 config-struct parameters not exposed to Python**, and **~30 hardcoded algorithm constants**. The previous R/C config sweep on vert12S tested only 5 of these (threshold, bootstraps, strand, min_descend, full_length) and found threshold=40, bootstraps=50 optimal (F1=0.707 vs 0.597 at defaults). Many impactful parameters remain untested, and the literature reveals no published multi-parameter grid search for IDTAXA on any marker.

## Currently Exposed Parameters

### Classification (Python-exposed via `classify()`)

| Parameter | Type | Default | Swept Before? | Previous Best |
|-----------|------|---------|---------------|---------------|
| `threshold` | f64 | 60.0 | Yes (20,40,60,80,95) | 40 |
| `bootstraps` | usize | 100 | Yes (10,50,100,200) | 50 |
| `strand` | str | "both" | Yes ("top","both") | "both" |
| `min_descend` | f64 | 0.98 | Yes (0.90,0.95,0.98,0.99) | 0.98 |
| `full_length` | f64 | 0.0 | Yes (0.0,0.5,0.8,0.9) | 0.0 |
| `processors` | usize | 8 | No (runtime only) | 8 |
| `seed` | u32 | 42 | No | 42 |
| `deterministic` | bool | false | No | false |

### Training (Python-exposed via `train()`)

| Parameter | Type | Default | Swept Before? |
|-----------|------|---------|---------------|
| `k` | Option<usize> | None (auto) | No |
| `seed` | u32 | 42 | No |
| `verbose` | bool | true | No (no-op) |

## Not Exposed but in Config Structs

### TrainConfig (types.rs:67-95)

| Parameter | Default | Swept? | What it controls |
|-----------|---------|--------|-----------------|
| `n` | 500.0 | No | Expected unique k-mers for auto-K computation |
| `min_fraction` | 0.01 | No | Floor for decision node sampling fraction |
| `max_fraction` | 0.06 | No | Initial sampling fraction at each node |
| `max_iterations` | 10 | No | Training re-classification iterations |
| `multiplier` | 100.0 | No | Step size for fraction reduction |
| `max_children` | 200 | No | Max children before treating node as leaf |

### Hardcoded Constants Worth Exposing

| Constant | Value | Location | Potential impact |
|----------|-------|----------|-----------------|
| Training bootstrap B | 100 | training.rs:49 | Affects training precision |
| Training vote threshold | 0.80 (80%) | training.rs:309 | Confidence for tree descent during training |
| Record k-mers fraction | 0.10 (10%) | training.rs:552 | How many decision k-mers per node |
| Sample size exponent | 0.47 | classify.rs:102 | k-mers sampled per bootstrap = L^0.47 |
| Bootstrap multiplier | 5.0 | classify.rs:121 | Actual B = min(5*nKmers/s, bootstraps) |
| Fallback descend threshold | 0.50 | classify.rs:225 | Secondary descent check when minDescend fails |
| K-mer masking | all disabled | training.rs:71, classify.rs:94 | Repeat/LCR/numerous masking (3 modes) |
| Min sequence length | 30bp | lib.rs:121 | Training quality filter |
| Max N fraction | 0.30 | lib.rs:125 | Training quality filter |
| Min taxonomy ranks | 4 | lib.rs:118 | Training quality filter |

## Previous Sweep Results (R/C on vert12S)

Best config: threshold=40, bootstraps=50 (F1=0.707)
Default config: threshold=60, bootstraps=100 (F1=0.597)

Top 5 configs on clean data:
1. threshold=40, bootstraps=200: F1=0.707
2. threshold=40, bootstraps=100: F1=0.691
3. threshold=40, min_descend=0.99: F1=0.686
4. threshold=40, bootstraps=50: F1=0.679
5. threshold=40, bootstraps=10: F1=0.671

Key findings:
- threshold=40 dominates all top configs
- full_length=0.5 caused complete failure (F1=0.000)
- threshold=95 produced very low F1 (0.197-0.261)
- min_descend and bootstraps had smaller effects than threshold

## Literature Findings

### Published Parameter Sensitivity (Murali et al. 2018)
- Only threshold and sampling exponent were formally analyzed
- Exponent 0.47 was a single calibration point, not grid-searched
- No marker-specific recommendations exist for any parameter
- No published multi-parameter grid search exists

### Potential Novel Parameters (not in original IDTAXA)

**K-mer weighting alternatives:**
- BM25 (term frequency saturation + length normalization) - never applied to taxonomic classification
- Sublinear TF-IDF (1 + log(tf)) - current IDTAXA uses binary presence/absence per taxon
- Mutual information between k-mer presence and taxon identity

**Masking modes (exist in code, disabled by default):**
- Repeat masking (`mask_repeats`) - masks tandem repeats
- Low-complexity masking (`mask_simple`) - chi-squared test on nucleotide distribution
- Numerous k-mer masking (`mask_numerous`) - masks overly common k-mers

**Sampling strategy:**
- The exponent 0.47 has never been systematically varied
- Protein IDTAXA (Wright 2021) abandoned L^0.47 for an adaptive minimum

## Recommended Sweep Parameters for Rust Implementation

### Tier 1: Re-sweep (known impact, recalibrate for Rust PRNG)
- `threshold`: [20, 30, 40, 50, 60, 70, 80]
- `bootstraps`: [25, 50, 100, 200]
- `min_descend`: [0.90, 0.95, 0.98, 0.99]

### Tier 2: First-time sweep (never tested, potentially impactful)
- `k`: [6, 7, 8, 9, 10] (override auto-computation)
- Sample size exponent: [0.35, 0.40, 0.47, 0.55, 0.65]
- Record k-mers fraction: [0.05, 0.10, 0.15, 0.20]

### Tier 3: Masking (exists in code, never enabled for DNA)
- `mask_repeats`: [true, false]
- `mask_lcrs`: [true, false]

### Tier 4: Training parameters (affect model quality)
- `max_fraction`: [0.03, 0.06, 0.10, 0.15]
- `min_fraction`: [0.005, 0.01, 0.02]
- `max_iterations`: [5, 10, 20]

## References

- IDTAXA paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6085705/
- Wright 2021 protein IDTAXA: https://pmc.ncbi.nlm.nih.gov/articles/PMC8445202/
- raxtax comparison: https://pmc.ncbi.nlm.nih.gov/articles/PMC12677947/
- Previous sweep: assignment-tool-benchmarking/.../config_sweep/idtaxa/config_sweep.md
- Rust types: rust/src/types.rs:67-116
- Rust classify constants: rust/src/classify.rs:102,121,225
- Rust training constants: rust/src/training.rs:49,309,552
