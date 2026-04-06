# Oxidaxa Feature Additions: Inverted Index, Length Normalization, Spaced K-mers, Per-Rank Thresholds

## Overview

Four features to improve Oxidaxa's classification speed and accuracy. Each is independently testable and sweepable. Ordered by implementation dependency.

## Features

### 1. Length Normalization (sweepable, high impact for variable-length markers)

**Problem**: vert12S refs range 50-467bp (CV=0.47). Longer refs have more k-mers → more matches → inflated scores.

**Change**: Add `length_normalize: bool` to ClassifyConfig (default false). When enabled, divide each training sequence's accumulated score by `sqrt(n_unique_kmers_in_training_seq / avg_unique_kmers)`.

**Where**: `matching.rs:parallel_match` — after accumulating hits for a training sequence, apply normalization factor. Also need to store per-sequence unique k-mer counts in the model (already available as `ts.kmers[idx].len()`).

**Exposed as**: `length_normalize=True/False` on `classify()`.

### 2. Spaced K-mers (sweepable, mutation robustness)

**Problem**: A single SNP/error destroys one contiguous 8-mer. For eDNA with 1-3% error rates, this loses signal.

**Change**: Add `seed_pattern: Option<String>` to both TrainConfig and ClassifyConfig. A pattern like `"11011011011"` means: match at positions 0,1,3,4,6,7,9,10 (weight=8, span=11). Default `None` = contiguous (current behavior).

**Where**: 
- `kmer.rs:enumerate_single` — read bases at pattern-specified offsets instead of consecutive positions. K-mer index computation uses only match positions. Window slides by 1 base but reads from span-width window.
- `types.rs:TrainingSet` — add `seed_pattern: Option<String>` so classification uses the same pattern as training.
- `lib.rs` — expose `seed_pattern` on both `train()` and `classify()`.

**Constraints**: Pattern must be same for training and classification. Stored in model. Weight (number of 1s) determines k-mer index space (4^weight). Must validate: all chars are '0' or '1', at least 1 '1'.

### 3. Per-Rank Confidence Thresholds (accuracy improvement)

**Problem**: Single threshold=40 forces tradeoff: miss obvious phylum assignments vs accept uncertain species.

**Change**: Add `rank_thresholds: Option<Vec<f64>>` to ClassifyConfig. When Some, use `rank_thresholds[i]` for rank i (0=Root, etc.). When None, use single `threshold` for all ranks (current behavior).

**Where**: `classify.rs:classify_one_pass` — change the confidence filtering at lines ~380-395. Instead of `confidences[i] >= config.threshold`, check `confidences[i] >= rank_threshold_for_depth(i)`.

**Exposed as**: `rank_thresholds=[90, 80, 70, 60, 50, 40, 40]` on `classify()`.

### 4. Inverted K-mer Index (massive speed improvement)

**Problem**: `parallel_match` does merge-join of query k-mers against EACH training sequence in `keep`. At 178K refs with poor tree descent, this is O(keep × query_kmers). Takes 87 minutes for 157K queries.

**Change**: Build an inverted index `kmer_id → Vec<u32>` (posting lists) at training time. Store in model. At classification time, for each sampled query k-mer, look up which training sequences have it and accumulate scores directly.

**Architecture**:
```
Training:
  enumerate_sequences → sort/dedup → kmers: Vec<Vec<i32>>
  NEW: build inverted_index: Vec<Vec<u32>> of size n_possible_kmers
       for each seq_idx, for each kmer in kmers[seq_idx]:
           inverted_index[kmer - 1].push(seq_idx as u32)

Classification (parallel_match_inverted):
  1. Build keep_map: HashMap<u32, usize> mapping global_seq_idx → position in keep array
  2. For each unique query k-mer in u_sampling:
     a. Look up posting_list = inverted_index[kmer - 1]
     b. For each seq_idx in posting_list:
        if let Some(&keep_pos) = keep_map.get(&seq_idx):
           hits_flat[keep_pos * b + bootstrap_positions] += idf_weight
```

**Memory**: ~110 MB for 178K refs at k=8 (27.6M postings × 4 bytes). Model grows from ~195MB to ~305MB.

**Speed**: At keep=178K (worst case), 891x fewer operations. At keep=500 (typical after tree descent), 3x fewer. The improvement scales with database size — at 500K refs it would be even more dramatic.

**Where**:
- `types.rs:TrainingSet` — add `inverted_index: Option<Vec<Vec<u32>>>`
- `training.rs:learn_taxa` — build inverted index after k-mer enumeration
- `matching.rs` — new `parallel_match_inverted` function
- `classify.rs:classify_one_pass` — dispatch to inverted version when available

**Backward compat**: `Option` field. Old models without inverted index fall back to current `parallel_match`.

## Implementation Order

### Phase 1: Length Normalization
- Add `length_normalize: bool` to ClassifyConfig (types.rs)
- Apply normalization in `parallel_match` (matching.rs)
- Expose in PyO3 + CLI
- Test: golden tests pass with `length_normalize=false`, new test with `true`

### Phase 2: Spaced K-mers
- Add `seed_pattern` parsing + validation (new helper in kmer.rs)
- Modify `enumerate_single` to support patterns (kmer.rs)
- Add `seed_pattern` to TrainConfig, TrainingSet, ClassifyConfig (types.rs)
- Expose in PyO3 + CLI
- Test: golden tests pass with `seed_pattern=None`, new test with a pattern

### Phase 3: Per-Rank Thresholds
- Add `rank_thresholds: Option<Vec<f64>>` to ClassifyConfig (types.rs)
- Modify confidence filtering in `classify_one_pass` (classify.rs)
- Expose in PyO3 + CLI
- Test: golden tests pass with `rank_thresholds=None`, new test with per-rank values

### Phase 4: Inverted Index
- Add `inverted_index: Option<Vec<Vec<u32>>>` to TrainingSet (types.rs)
- Build index in `learn_taxa` (training.rs)
- Add `parallel_match_inverted` (matching.rs)
- Add dispatch in `classify_one_pass` (classify.rs)
- Backward-compat model loading (try new format, fall back to legacy)
- Test: golden tests pass (same results via inverted path), benchmark speedup

### Phase 5: Verification
- All 51 golden tests pass with defaults (no new features enabled)
- End-to-end benchmark on 10K dataset: verify identical results
- Production benchmark on 178K refs: measure speed improvement
- Update README with new parameters

## What We're NOT Doing
- BM25 (data showed tf-saturation useless: 99%+ k-mers appear once)
- NA handling changes (complicates benchmarking comparison)
- Multi-resolution k-mers (complex, uncertain benefit)
- Confidence calibration (post-processing, separate project)

## Files Modified
- `rust/src/types.rs` — new fields on TrainConfig, ClassifyConfig, TrainingSet
- `rust/src/kmer.rs` — spaced k-mer support in enumerate_single
- `rust/src/matching.rs` — parallel_match_inverted, length normalization
- `rust/src/classify.rs` — dispatch logic, per-rank thresholds
- `rust/src/training.rs` — inverted index construction
- `rust/src/lib.rs` — PyO3 bindings for new params
- `train_idtaxa.py`, `classify_idtaxa.py` — CLI args
- `README.md` — document new parameters

## New Sweepable Parameters (complete list after implementation)

### Training
| Parameter | Default | Sweep values |
|-----------|---------|-------------|
| k | auto (~8) | [6, 7, 8, 9, 10] |
| record_kmers_fraction | 0.10 | [0.05, 0.10, 0.15, 0.20] |
| seed_pattern | None (contiguous) | [None, "11011011011", "110110110110"] |

### Classification
| Parameter | Default | Sweep values |
|-----------|---------|-------------|
| threshold | 60.0 | [20, 30, 40, 50, 60, 70, 80] |
| bootstraps | 100 | 50 for sweep, 100 production |
| min_descend | 0.98 | [0.90, 0.95, 0.98, 0.99] |
| sample_exponent | 0.47 | [0.35, 0.40, 0.47, 0.55, 0.65] |
| length_normalize | false | [true, false] |
| rank_thresholds | None | [None, [90,80,70,60,50,40,40]] |
