# Batch Fraction Learning: Implementation Results

## Summary

Replaced IDTAXA's sequential per-sequence fraction learning with batch updates, per-sequence independent PRNGs, and rayon parallelization. Training is now order-independent and ~8x faster.

## Comparison vs Original R Package (golden test datasets, 80 seqs)

| Field | vs R |
|-------|------|
| Taxonomy nodes | Identical sets, different ordering (sorted for order-independence) |
| K | Identical |
| Fractions | Identical (0 diffs across all 6 test sets) |
| IDF weights | Identical (0 diffs) |
| Problem sequences | Identical (0 in both) |
| Problem groups | Identical (0 in both) |
| Children (by name) | Identical |
| Decision k-mers | 99.9% overlap — 1 k-mer swapped per affected node due to cross-entropy tie-breaking |

## Order-Independence Verification

| Dataset | Path Agreement (shuffled) | Confidence Diff | Problem Seqs (A vs B) |
|---------|--------------------------|-----------------|----------------------|
| 1K | 100.0% | 0.00 | 0 vs 0 |
| 5K | 100.0% | 0.00 | 0 vs 0 |
| 10K | 100.0% | 0.00 | 0 vs 0 |

## Performance

| Dataset | Sequential | Batch+Parallel | Speedup |
|---------|-----------|----------------|---------|
| 5K | 4.35s | 0.60s | 7.3x |
| 10K | 8.15s | 1.09s | 7.5x |

## Sequential vs Batch Classification Comparison (on held-out queries)

| Dataset | Path Agreement | Genus Agreement | Mean Conf Diff | Max Conf Diff | Problem Seqs |
|---------|---------------|-----------------|----------------|---------------|--------------|
| 1K | 92.8% | 92.8% | 1.95 | 23.01 | 0 → 0 |
| 5K | 86.8% | 87.2% | 3.29 | 40.86 | 0 → 0 |
| 10K | 87.4% | 88.6% | 3.46 | 67.01 | 0 → 0 |

The ~13% path disagreement between sequential and batch comes from:
1. Sorted taxonomy ordering changing internal node indices (~7%, visible in 1K where no fraction learning occurs)
2. Per-sequence PRNGs vs shared PRNG producing different random streams (~6%)

All divergent queries are near the confidence threshold — the direction of change is symmetric (roughly equal queries classify deeper in batch vs sequential).

## Key Design Decisions

- **Capped batch decrements**: `min(raw_decrement, headroom * 0.5)` prevents a single batch from cratering a node's fraction. Without this, Root would become a problem group on 5K+ datasets (857 problem seqs). With capping, fractions decrease gradually over multiple iterations.
- **Hash-based PRNG seeding**: Per-sequence PRNG seeded from `mix_seed(seed, iteration * 1M + hash(sequence + taxonomy))` — uses sequence identity rather than array index, ensuring true order-independence.
- **Sorted u_classes**: Taxonomy tree construction now sorts unique classes before building prefixes, making the tree structure deterministic regardless of input order.
