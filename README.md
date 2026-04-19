# Oxidtaxa

High-performance taxonomic classifier for eDNA metabarcoding. A Rust rewrite of IDTAXA (DECIPHER) with Python bindings via PyO3.

10-13x faster than R/C at production scale. Identical classification algorithm with per-query independent PRNG for reproducible parallel execution.

## Quick Start

```bash
# Build (requires Rust toolchain + Python 3.10+)
pip install maturin
maturin develop --release
```

```python
from oxidtaxa import train, classify

train("reference.fasta", "taxonomy.tsv", "model.bin")
results = classify("query.fasta", "model.bin")
```

## Python API

```python
from oxidtaxa import train, classify

# Train
train(
    fasta_path="reference.fasta",
    taxonomy_path="taxonomy.tsv",      # tab-separated: accession<TAB>semicolon_path
    output_path="model.bin",
    seed=42,
    k=None,                            # k-mer size (None = auto from sequence lengths)
    record_kmers_fraction=0.10,        # fraction of top k-mers per decision node
    verbose=True,                      # print progress during training
    seed_pattern=None,                 # spaced seed (e.g., "11011011011"). None = contiguous
    training_threshold=0.8,            # vote fraction to descend during fraction learning
    descendant_weighting="count",      # "count", "equal", or "log"
    use_idf_in_training=False,         # IDF-weighted scoring during training descent
    leave_one_out=False,               # reduce self-classification bias for small groups
    correlation_aware_features=False,  # greedy feature selection with redundancy penalty
    processors=1,                      # threads for tree construction + fraction learning
)

# Classify — returns List[ClassificationResult] in-memory. Pass output_path
# to ALSO write a TSV file (backward compatible with existing pipelines).
results = classify(
    query_path="query.fasta",
    model_path="model.bin",
    output_path="results.tsv",         # optional; when omitted, results are only returned
    threshold=60.0,                    # confidence cutoff (0-100)
    bootstraps=100,                    # bootstrap replicates (50 for sweeps, 100 production)
    strand="both",                     # "top", "bottom", or "both"
    min_descend=0.98,                  # fraction of votes to descend tree
    full_length=0.0,                   # length filter (0 = disabled)
    processors=8,                      # threads
    sample_exponent=0.47,              # k-mers per bootstrap: S = L^exponent
    seed=42,
    deterministic=False,               # True = R-compatible sequential PRNG
    length_normalize=False,            # normalize scores by training sequence length
    rank_thresholds=None,              # per-rank thresholds (e.g., [90, 80, 70, 60, 50, 40])
    beam_width=1,                      # candidate paths during tree descent (1 = greedy)
)

for r in results:
    print(r.taxon)         # list[str] — root-to-leaf lineage
    print(r.confidence)    # list[float] — per-rank confidence percentages
    print(r.alternatives)  # list[str] — tied species (empty when unique)
```

### Staged Training API

For parameter sweeps (e.g., with Optuna), training can be decomposed into three independently serializable stages. This avoids recomputing expensive steps when only downstream parameters change.

```python
from oxidtaxa import prepare_data, build_tree, learn_fractions

# Stage 1: K-mer enumeration + IDF weights (reuse across all tree/fraction configs)
data = prepare_data("reference.fasta", "taxonomy.tsv", k=8, processors=8)

# Stage 2: Feature selection + tree construction (reuse across fraction configs)
tree = build_tree(data, record_kmers_fraction=0.10, processors=8)

# Stage 3: Fraction learning (fast — only this needs to re-run per config)
learn_fractions(data, tree, "model.bin", seed=42, training_threshold=0.8)
learn_fractions(data, tree, "model_strict.bin", seed=42, training_threshold=0.98)
learn_fractions(data, tree, "model_loo.bin", seed=42, leave_one_out=True)
```

`PreparedData` and `BuiltTree` objects can be saved/loaded with `.save(path)` / `.load(path)`.

### Tied-species resolution

When two or more reference sequences produce identical top-scoring matches for a query (common for marker genes where congeneric species share 100% sequence identity), the classifier will:

1. **Cap the primary assignment at the lowest common ancestor of the tied set.** For example, if *Canis lupus* and *Canis latrans* tie exactly, `taxon` ends at `Canis` — never at either species, even if per-rank confidence at the species level would otherwise clear the threshold. The classifier never reports an assignment it cannot defend.
2. **Report every tied species in `alternatives`**, as short-labels (e.g. `Canis_latrans`, `Canis_lupus`), sorted alphabetically.

```python
# Species-level presence query
for r in results:
    if "Canis_lupus" in r.taxon or "Canis_lupus" in r.alternatives:
        print(f"Detected Canis_lupus (resolved={'Canis_lupus' in r.taxon})")
```

When `output_path` is provided, the TSV gains a 4th column `alternatives` (pipe-separated, empty for non-tied rows):

| read_id  | taxonomic_path                                      | confidence | alternatives              |
|----------|-----------------------------------------------------|------------|---------------------------|
| read_042 | Eukaryota;Chordata;Mammalia;Carnivora;Canidae;Canis | 100.0      | Canis_latrans\|Canis_lupus |
| read_043 | Eukaryota;Chordata;Mammalia;Carnivora;Felidae;Felis | 95.3       |                           |

## Parameters

### Classification Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **threshold** | 60.0 | 0-100 | Confidence cutoff. Ranks below this are reported as unclassified. Lower = deeper classification, more false positives. **Most impactful parameter.** |
| **bootstraps** | 100 | 1-1000 | Bootstrap replicates. Higher = more precise confidence, slower. Use 50 for parameter sweeps, 100 for production. |
| **strand** | "both" | top/bottom/both | Which strand(s) to classify. "both" classifies forward and reverse complement, keeps the better hit. |
| **min_descend** | 0.98 | 0.5-1.0 | Fraction of bootstrap votes required to descend into a child node. Lower = more aggressive descent into uncertain branches. |
| **full_length** | 0.0 | 0.0+ | Length ratio filter for training sequences. 0 disables. When set (e.g. 2.0), excludes training sequences whose k-mer count differs by more than this fold from the query. |
| **sample_exponent** | 0.47 | 0.2-0.8 | Controls k-mers sampled per bootstrap: S = L^exponent where L = unique k-mers in query. Lower = fewer samples (faster, noisier). Higher = more samples (slower, more stable). |
| **length_normalize** | false | true/false | Divide each training sequence's score by sqrt(n_unique_kmers / avg_unique_kmers). Corrects bias from longer references accumulating more k-mer hits. Most useful for variable-length markers. |
| **rank_thresholds** | None | list of floats | Per-rank confidence thresholds (index 0 = Root, 1 = next rank, etc.). When set, overrides the single `threshold` parameter. Allows strict filtering at high ranks (e.g., 90 for phylum) and lenient filtering at low ranks (e.g., 40 for species). If shorter than the predicted path, the last value is reused. |
| **beam_width** | 1 | 1-10 | Number of candidate paths maintained during tree descent. At 1, classification uses greedy descent (original IDTAXA). At higher values, the classifier explores multiple paths at ambiguous nodes and picks the candidate with the highest leaf-phase similarity. Useful when the greedy path makes an early wrong turn. |
| **deterministic** | false | true/false | When true, uses a single shared PRNG for R-compatible sequential output. When false (default), each query gets an independent PRNG for parallel execution. |
| **processors** | 1 | 1+ | Number of threads for parallel classification. |
| **seed** | 42 | any u32 | PRNG seed. In non-deterministic mode, each query gets seed XOR query_index. |

### Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **k** | auto | 1-15 | K-mer word size. When None, auto-computed as floor(ln(500 * L_99th) / ln(alphabet_size)). Typical values: 7-9 for DNA. When using a spaced seed, this is overridden by the seed's weight. |
| **record_kmers_fraction** | 0.10 | 0.01-0.5 | Fraction of most-discriminating k-mers retained at each decision node. Higher = more features per node, larger model, potentially better discrimination. |
| **seed_pattern** | None | binary string | Spaced seed pattern for mutation-robust k-mer enumeration. A string of 1s and 0s (e.g., `"11011011011"`). Positions with `1` are used; `0` positions are skipped. See [Spaced K-mers](#spaced-k-mers) below. |
| **training_threshold** | 0.8 | 0.0-1.0 | Bootstrap vote fraction required to descend during fraction learning. R's IDTAXA hardcodes 0.8. Set closer to `min_descend` (e.g., 0.98) for consistent training/classification thresholds — sequences that wouldn't pass classification descent won't pass training descent either. |
| **descendant_weighting** | "count" | count/equal/log | How to weight child profiles when computing the merged profile for cross-entropy feature selection. "count" = weight by raw descendant count (original IDTAXA). "equal" = 1/n_children each, preventing large clades from dominating feature selection. "log" = log(1+descendants), a middle ground. |
| **use_idf_in_training** | false | true/false | Multiply profile weights by IDF weights during the fraction-learning tree descent. Without this, training uses raw profile frequencies while classification uses IDF-weighted scores — a train/test mismatch. Enabling this calibrates fractions under the same scoring regime used at classification time. |
| **leave_one_out** | false | true/false | Reduce self-classification bias for small taxonomy groups during fraction learning. For groups with 2-5 sequences, scales profile weights by (n-1)/n to approximate excluding the test sequence from its own group's profile. Singletons are skipped (no meaningful LOO signal with one member). Most impactful for databases with many low-count species. |
| **correlation_aware_features** | false | true/false | Replace the default round-robin k-mer selection with greedy forward selection that penalizes redundant features. At each step, selects the k-mer maximizing `entropy * (1 - max_correlation_with_selected)`. Produces a more diverse, efficient feature set. Slower training but no impact on classification speed. |
| **processors** | 1 | 1+ | Number of threads. Parallelizes tree construction (sibling subtrees) and fraction learning (sequences within each iteration). |
| **seed** | 42 | any u32 | PRNG seed for the training bootstrap loop. |
| **verbose** | true | true/false | Print progress during training (iteration counts, problem sequences/groups). |

### Spaced K-mers

A standard (contiguous) 8-mer is destroyed by a single SNP or sequencing error. Spaced seeds spread the k-mer across a wider window, so a point mutation only affects k-mers where the error lands on a `1` position.

**How it works:** The pattern `"11011011011"` has **weight** 8 (eight 1s) and **span** 11. The window slides across the sequence 1 base at a time, but only the 8 positions marked `1` contribute to the k-mer index. The index space is 4^weight = 4^8 = 65,536 — same as contiguous k=8.

```
Sequence:  A C G T A C G T A C G T A C G
Pattern:   1 1 0 1 1 0 1 1 0 1 1
Used:      A C . T A . G T . C G  →  k-mer from {A,C,T,A,G,T,C,G}
              1 1 0 1 1 0 1 1 0 1 1
              C G . A C . T A . G T  →  next k-mer
```

**Two independent axes to sweep:**

1. **Weight** (= effective k): determines specificity. Weight 6 = 4,096 possible k-mers (less specific), weight 10 = 1,048,576 (very specific).
2. **Span-to-weight ratio**: determines mutation robustness. Ratio 1.0 = contiguous, 1.4 = moderate spacing, 1.8+ = wide spacing.

**Important:** The seed pattern is a training parameter — it's stored in the model. You must train a separate model for each pattern. Classification reads the pattern from the model automatically.

**Choosing patterns:** Periodic patterns (like `"11011011011"`) are simple but suboptimal — the literature on spaced seeds (Li et al. 2004, PatternHunter) shows that aperiodic, asymmetric patterns perform better because they spread miss events more evenly. Avoid long runs of consecutive 0s.

### Parameter Sweep Guide

For optimizing classification on a new marker, sweep in tiers. Each tier's optimal values feed into the next.

**Tier 1 — Highest impact, sweep first:**

These are classification-only parameters (no retraining needed), so sweeps are fast.

```python
thresholds = [20, 30, 40, 50, 60, 70, 80]
sample_exponents = [0.35, 0.40, 0.47, 0.55, 0.65]
```

**Tier 2 — Training parameters (require retraining per value):**

```python
k_values = [6, 7, 8, 9, 10]                       # train separate model per k
record_kmers_fractions = [0.05, 0.10, 0.15, 0.20]  # train separate model per fraction
```

**Tier 3 — Secondary classification parameters:**

```python
min_descend_values = [0.90, 0.95, 0.98, 0.99]
length_normalize = [True, False]
beam_width_values = [1, 2, 3]         # 1 = greedy (original), >1 = beam search
rank_thresholds_options = [
    None,                              # single threshold (from Tier 1)
    [90, 80, 70, 60, 50, 40, 40],     # strict top, lenient bottom
    [80, 70, 60, 50, 40, 30, 30],     # moderate gradient
]
```

**Tier 4 — Algorithmic training variants (require retraining per combination):**

These are experimental improvements to the IDTAXA training algorithm. Each changes how the model is built. Sweep independently against default, then combine the winners.

```python
# Feature selection strategy
descendant_weighting_values = ["count", "equal", "log"]
correlation_aware_features = [False, True]

# Training/classification consistency
training_threshold_values = [0.8, 0.9, 0.98]  # match min_descend for consistency
use_idf_in_training = [False, True]

# Bias correction (most useful for sparse databases)
leave_one_out = [False, True]
```

**Tier 5 — Spaced seeds (require retraining per pattern):**

Train one model per pattern, then sweep Tier 1 classification params for each. Fix weight to the best k from Tier 2.

```python
# Weight-8 patterns (same index space as contiguous k=8)
seed_patterns = [
    None,                    # contiguous baseline
    "11011011011",           # periodic, span=11, ratio=1.38
    "1101001100101",         # aperiodic, span=13, ratio=1.62
    "110100110010101",       # aperiodic, span=15, ratio=1.88
]

# Weight-7 patterns (if Tier 2 found k=7 optimal)
seed_patterns_w7 = [
    None,                    # contiguous k=7
    "1101101011",            # span=10, ratio=1.43
    "11010010110",           # aperiodic, span=11, ratio=1.57
]
```

**Sweep strategy:**

```bash
# Tier 2: train models with different k values
for k in 6 7 8 9 10; do
    python train.py ref.fasta tax.tsv model_k${k}.bin --k $k
done

# Tier 4: train models with algorithmic variants
python train.py ref.fasta tax.tsv model_default.bin
python train.py ref.fasta tax.tsv model_equal.bin --descendant-weighting equal
python train.py ref.fasta tax.tsv model_idf.bin --use-idf-in-training
python train.py ref.fasta tax.tsv model_corr.bin --correlation-aware-features
python train.py ref.fasta tax.tsv model_loo.bin --leave-one-out
python train.py ref.fasta tax.tsv model_strict.bin --training-threshold 0.98

# Tier 5: train models with different spaced seeds (all weight=8)
python train.py ref.fasta tax.tsv model_contiguous.bin
python train.py ref.fasta tax.tsv model_s11.bin --seed-pattern "11011011011"
python train.py ref.fasta tax.tsv model_s13.bin --seed-pattern "1101001100101"
python train.py ref.fasta tax.tsv model_s15.bin --seed-pattern "110100110010101"

# Sweep classification params for each model
for model in model_*.bin; do
    for thresh in 20 30 40 50 60 70 80; do
        for bw in 1 3; do
            python classify.py query.fasta $model out_${model}_t${thresh}_bw${bw}.tsv \
                $thresh both 0.98 0.0 8 --bootstraps 50 --beam-width $bw
        done
    done
done
```

**Tip:** Use `bootstraps=50` during sweeps for 2x speedup. The optimal threshold found at 50 bootstraps transfers directly to production at 100 bootstraps.

## Performance

Benchmarked on Apple M4 Pro (14 cores), vert12S marker:

### vs R/C IDTAXA (DECIPHER)

```
                    R/C        Oxidtaxa 1T   Oxidtaxa 8T   Speedup(1T)  Speedup(8T)
1K ref / 500q
  Train           4.81s        0.43s           —          11.2x           —
  Classify        1.39s        0.46s         0.12s         3.0x         11.6x

5K ref / 500q
  Train          17.17s        1.30s           —          13.2x           —
  Classify        4.65s        0.91s         0.42s         5.1x         11.1x

10K ref / 500q
  Train          29.63s        2.29s           —          12.9x           —
  Classify        9.00s        1.52s         0.80s         5.9x         11.3x
```

### Production Scale (178K refs, 157K queries, 8 threads)

```
  Train:     47s
  Classify:  87 min (pre-inverted-index)
```

### Inverted K-mer Index

Models now include an inverted index (k-mer id -> list of training sequences containing it), built automatically during training. This replaces the O(|keep| x |query_kmers|) merge-join in `parallel_match` with O(|query_kmers| x avg_posting_list) lookups via `parallel_match_inverted`.

**Theoretical speedup** depends on how many training sequences remain after tree descent (`keep`):

| Scenario | keep size | Merge-join ops | Inverted ops | Speedup |
|----------|-----------|----------------|--------------|---------|
| Worst case (poor tree descent) | 178K | 178K x 250 = 44.5M | 100 x 407 = 40.7K | ~1,000x |
| Typical (good tree descent) | 500 | 500 x 250 = 125K | 100 x 407 = 40.7K | ~3x |
| Best case (narrow leaf) | 10 | 10 x 250 = 2.5K | 100 x 407 = 40.7K | <1x (merge-join wins) |

The inverted path dominates at large `keep` sizes — exactly the cases that were slowest before. At small `keep`, both paths are fast. The dispatch falls back to merge-join when no inverted index is present.

**Memory cost:** ~4 bytes per posting entry. For 178K refs at k=8: ~27M postings = ~108 MB. Model grows from ~195 MB to ~305 MB.

## Architecture

```
src/
  lib.rs          PyO3 bindings (train/classify entry points)
  types.rs        TrainConfig, ClassifyConfig, TrainingSet, ClassificationResult
  training.rs     LearnTaxa algorithm (tree building + fraction learning)
  classify.rs     IdTaxa algorithm (tree descent + bootstrap voting)
  kmer.rs         K-mer enumeration + masking (repeat, LCR, numerous)
  matching.rs     int_match, vector_sum, parallel_match (flat matrix)
  rng.rs          R-compatible MT19937 PRNG
  fasta.rs        FASTA/taxonomy I/O
  sequence.rs     reverse_complement, remove_gaps
  alphabet.rs     Alphabet entropy for auto-K computation
```

### Key Design Decisions

- **Per-query independent PRNG**: Each query gets `seed XOR index` instead of R's sequential shared PRNG. Classification of query N is independent of query N-1. Statistically equivalent, fully parallelizable.
- **Flat contiguous hits matrix**: Single allocation for all training sequence comparisons (matching C's `allocMatrix`), not Vec-of-Vec.
- **O(n) algorithms**: HashMap-based dereplicate, HashSet-based taxonomy dedup, prefix-walk for sequence-to-node mapping. No O(n^2) operations.
- **Inverted k-mer index**: Built at training time, enables O(query_kmers) classification instead of O(keep x query_kmers) merge-join. Up to 1000x faster at large database sizes.
- **Bincode serialization**: Trained models saved as compact binary. ~305 MB for 178K reference sequences (includes inverted index).

### Correctness and Divergence from R

Oxidtaxa was initially validated against R/C IDTAXA with 51 golden tests verifying bit-level agreement across 13 scenarios (threshold sweeps, strand modes, bootstrap counts, edge cases). Having proven algorithmic equivalence, Oxidtaxa now diverges from R in specific areas where the original design was constrained by R's single-threaded execution model rather than algorithmic necessity:

**Classification:** Each query gets an independent PRNG (`seed XOR index`) instead of R's shared sequential PRNG. This makes classification order-independent and fully parallelizable. Results are statistically equivalent but not bit-identical to R.

**Training fraction learning:** R's LearnTaxa uses sequential per-sequence fraction updates — a Gauss-Seidel-style approach where each misclassification immediately decrements the sampling fraction, and subsequent sequences see the updated value within the same iteration. Oxidtaxa instead processes all sequences in parallel within each iteration and applies batch fraction updates at the end. This enables multi-threaded training but produces slightly different fractions — baseline tests show 87-93% path agreement with R on identical inputs.

## File Format

### Input

- **Reference FASTA**: Standard FASTA, sequences uppercased automatically
- **Taxonomy TSV**: Tab-separated, `accession<TAB>semicolon_delimited_path`
  ```
  AF014587.1    Eukaryota;Chordata;Mammalia;Carnivora;Canidae;Canis;Canis_lupus
  ```
- **Query FASTA**: Standard FASTA, gaps removed automatically

### Output

Classification results are returned in-memory as `List[ClassificationResult]` from `classify()`. When `output_path` is provided, a TSV is also written with header:

```
read_id    taxonomic_path                                         confidence  alternatives
asv_1      Eukaryota;Craniata;Mammalia;Catarrhini;Homininae;Homo  93.85
asv_2      Eukaryota;Craniata;Mammalia;Myomorpha;Arvicolinae      54.30
asv_3                                                              0
asv_4      Eukaryota;Chordata;Mammalia;Carnivora;Canidae;Canis    100.00      Canis_latrans|Canis_lupus
```

- `read_id`: First whitespace-delimited word from FASTA header
- `taxonomic_path`: Semicolon-delimited, Root stripped. Truncated at the last confidently-classified rank; no synthetic placeholder ranks appended.
- `confidence`: Minimum confidence across all reported ranks (0-100)
- `alternatives`: Pipe-separated short-labels of tied species when the classifier capped at an LCA; empty otherwise (see [Tied-species resolution](#tied-species-resolution))

### Model Format

Binary (bincode). Not compatible with R's `.rds` format. Train separately for Oxidtaxa.

## Development

```bash
# Run tests
cargo test --release

# Run benchmarks (Criterion)
cargo bench

# Build Python wheel
maturin develop --release
```

## References

- Murali, Bhargava & Wright (2018). "IDTAXA: a novel approach for accurate taxonomic classification of microbiome sequences." *Microbiome* 6:140. https://doi.org/10.1186/s40168-018-0521-5
- Wright (2021). "Classifying and mapping microbial sequences with IDTAXA." *NAR Genomics and Bioinformatics* 3(3). https://doi.org/10.1093/nargab/lqab080
