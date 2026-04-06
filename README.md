# Oxidaxa

High-performance taxonomic classifier for eDNA metabarcoding. A Rust rewrite of IDTAXA (DECIPHER) with Python bindings via PyO3.

10-13x faster than R/C at production scale. Identical classification algorithm with per-query independent PRNG for reproducible parallel execution.

## Quick Start

```bash
# Build (requires Rust toolchain + Python 3.10+)
pip install maturin
maturin develop --manifest-path rust/Cargo.toml --release

# Train a model
python train_idtaxa.py reference.fasta taxonomy.tsv model.bin

# Classify
python classify_idtaxa.py query.fasta model.bin results.tsv \
    40 both 0.98 0.0 8
```

## Python API

```python
from idtaxa import train, classify

# Train
train(
    fasta_path="reference.fasta",
    taxonomy_path="taxonomy.tsv",      # tab-separated: accession<TAB>semicolon_path
    output_path="model.bin",
    seed=42,
    k=None,                            # k-mer size (None = auto from sequence lengths)
    record_kmers_fraction=0.10,        # fraction of top k-mers per decision node
)

# Classify
classify(
    query_path="query.fasta",
    model_path="model.bin",
    output_path="results.tsv",
    threshold=60.0,                    # confidence cutoff (0-100)
    bootstraps=100,                    # bootstrap replicates (50 for sweeps, 100 production)
    strand="both",                     # "top", "bottom", or "both"
    min_descend=0.98,                  # fraction of votes to descend tree
    full_length=0.0,                   # length filter (0 = disabled)
    processors=8,                      # threads
    sample_exponent=0.47,              # k-mers per bootstrap: S = L^exponent
    seed=42,
)
```

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
| **processors** | 1 | 1+ | Number of threads for parallel classification. |
| **seed** | 42 | any u32 | PRNG seed. Each query gets an independent PRNG seeded with seed XOR query_index. |

### Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **k** | auto | 1-15 | K-mer word size. When None, auto-computed as floor(ln(500 * L_99th) / ln(alphabet_size)). Typical values: 7-9 for DNA. |
| **record_kmers_fraction** | 0.10 | 0.01-0.5 | Fraction of most-discriminating k-mers retained at each decision node. Higher = more features per node, larger model, potentially better discrimination. |
| **seed** | 42 | any u32 | PRNG seed for the training bootstrap loop. |

### Parameter Sweep Recommendations

For optimizing classification on a new marker:

**Tier 1 — Sweep these first (known high impact):**
```python
thresholds = [20, 30, 40, 50, 60, 70, 80]
sample_exponents = [0.35, 0.40, 0.47, 0.55, 0.65]
k_values = [6, 7, 8, 9, 10]
```

**Tier 2 — Secondary parameters:**
```python
min_descend_values = [0.90, 0.95, 0.98, 0.99]
record_kmers_fractions = [0.05, 0.10, 0.15, 0.20]
```

**Tip:** Use `bootstraps=50` during sweeps for 2x speedup. The optimal threshold found at 50 bootstraps transfers directly to production at 100 bootstraps.

## Performance

Benchmarked on Apple M4 Pro (14 cores), vert12S marker:

### vs R/C IDTAXA (DECIPHER)

```
                    R/C        Oxidaxa 1T   Oxidaxa 8T   Speedup(1T)  Speedup(8T)
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
  Classify:  87 min
```

## Architecture

```
rust/src/
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
- **Bincode serialization**: Trained models saved as compact binary. ~195 MB for 178K reference sequences.

### Correctness

51 golden tests verify bit-level agreement with R/C IDTAXA across 13 scenarios:
- Standard classification, perfect matches, novel organisms
- Threshold sweep (7 values), strand variations (3 modes)
- Duplicate queries, short sequences, bootstrap sweep, minDescend sweep
- Problem group models, singleton taxonomy models

Taxon paths must match exactly. Confidence values within 5.0 tolerance (float reordering from flat matrix layout).

## File Format

### Input

- **Reference FASTA**: Standard FASTA, sequences uppercased automatically
- **Taxonomy TSV**: Tab-separated, `accession<TAB>semicolon_delimited_path`
  ```
  AF014587.1    Eukaryota;Chordata;Mammalia;Carnivora;Canidae;Canis;Canis_lupus
  ```
- **Query FASTA**: Standard FASTA, gaps removed automatically

### Output

Tab-separated with header:
```
read_id    taxonomic_path                                         confidence
asv_1      Eukaryota;Craniata;Mammalia;Catarrhini;Homininae;Homo  93.85
asv_2      Eukaryota;Craniata;Mammalia;Myomorpha;Arvicolinae      54.30
asv_3                                                              0
```

- `read_id`: First whitespace-delimited word from FASTA header
- `taxonomic_path`: Semicolon-delimited, Root stripped, `unclassified_*` filtered
- `confidence`: Minimum confidence across all reported ranks (0-100)

### Model Format

Binary (bincode). Not compatible with R's `.rds` format. Train separately for Oxidaxa.

## Development

```bash
# Run tests
cargo test --manifest-path rust/Cargo.toml --release

# Run benchmarks (Criterion)
cargo bench --manifest-path rust/Cargo.toml

# Run end-to-end comparison
bash benchmarks/run_benchmark.sh

# Build Python wheel
maturin develop --manifest-path rust/Cargo.toml --release
```

## References

- Murali, Bhargava & Wright (2018). "IDTAXA: a novel approach for accurate taxonomic classification of microbiome sequences." *Microbiome* 6:140. https://doi.org/10.1186/s40168-018-0521-5
- Wright (2021). "Classifying and mapping microbial sequences with IDTAXA." *NAR Genomics and Bioinformatics* 3(3). https://doi.org/10.1093/nargab/lqab080
