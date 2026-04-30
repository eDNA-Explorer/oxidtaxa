---
date: 2026-04-30T00:00:00-07:00
researcher: Ryan Martin
git_commit: f36cd2eb90a7f5d9fa072f03d422b83375773f17
branch: main
repository: eDNA-Explorer/oxidtaxa
topic: "OAT-flagged inert params: beam_width, leave_one_out, use_idf_in_training"
tags: [research, codebase, oxidtaxa, ablation, beam_width, leave_one_out, use_idf_in_training, classify, training, vector_sum, normalization]
status: complete
last_updated: 2026-04-30
last_updated_by: Ryan Martin
last_updated_note: "Follow-up §5: vector_sum normalization makes LOO uniform scaling structurally inert. Follow-up §6: fix-strategy analysis (option A asymmetric scaling vs option B per-kmer LOO with stored counts), size-cap discussion, singleton k-mer rationale."
---

# Research: OAT-flagged inert params — `beam_width`, `leave_one_out`, `use_idf_in_training`

**Date**: 2026-04-30
**Researcher**: Ryan Martin
**Git Commit**: f36cd2eb90a7f5d9fa072f03d422b83375773f17
**Branch**: main
**Repository**: eDNA-Explorer/oxidtaxa

## Research Question

An OAT (one-at-a-time) ablation study reported three oxidtaxa parameters as suspected
no-ops or wiring bugs:

1. `beam_width` (classify-time): predictions bit-identical between `2` and `5`. Hypothesized as gated by `tie_margin > 0`.
2. `leave_one_out` (training-time): trained-model md5 byte-identical between `True` and `False`. Hypothesized as not wired into the training loop.
3. `use_idf_in_training` (training-time): different model md5 between `True` and `False`, but classify-time predictions byte-identical. Hypothesized as classify-side bug — the flag's training-time output not actually consumed at classify time.

For each: (a) trace the read sites, (b) determine whether the OAT observation is consistent with the code (no-op by design / bug / data-dependent), (c) name any specific function:line where wiring is missing.

## Summary

**No genuine wiring bugs.** All three parameters are wired correctly end-to-end through the Python binding into the relevant Rust config struct, and each is read at well-defined locations during training or classification. The OAT observations are all consistent with the code, but for different reasons in each case.

| Param | Wired? | OAT observation consistent because | Classification |
|---|---|---|---|
| `beam_width` | yes | Beam frontier expansion is gated by `min_descend` (default **0.98**) at `src/classify.rs:460`, NOT by `tie_margin`. With `min_descend = 0.98` and runner-ups requiring `vote_fraction >= 0.98`, the beam frontier almost never exceeds size 1 — so `next.truncate(beam_width)` at `src/classify.rs:504` is a no-op for `beam_width >= 1`. | **No-op by design (gating threshold)** |
| `leave_one_out` | yes | The flag fires only at internal nodes whose subtree size is between 2 and 5 sequences inclusive (`src/training.rs:484`: `group_size > 1 && group_size <= 5`). It also only ever mutates `fraction` / `problem_sequences` / `problem_groups`; `decision_kmers` is built in a separate phase that hardcodes `leave_one_out = false` via `..Default::default()` (`src/training.rs:358`). On databases with no nodes in the 2–5 size range, the flag is a complete no-op. | **No-op by design (narrow firing condition)** |
| `use_idf_in_training` | yes | Toggling the flag changes only `fraction` / `problem_sequences` / `problem_groups` (via altered training-time bootstrap winners). `idf_weights_by_rank` and `decision_kmers` are byte-identical across the two settings (computed in flag-independent phases at `src/training.rs:313` and `_build_tree_inner`). At classify time, the descent path at `src/classify.rs:217` and `src/classify.rs:376` uses raw `dk.profiles[j]` (no IDF), and the leaf-phase IDF reads at `src/classify.rs:553-554` consume the always-identical `ts.idf_weights_by_rank`. So predictions equality reduces to whether the differing `fraction` values flip descent decisions on the test set. | **Data-dependent** |

The OAT hypothesis was directionally correct for `beam_width` (it IS gated, just by `min_descend` not `tie_margin`) and for `leave_one_out` (its scope is narrow enough to be invisible on most databases). For `use_idf_in_training`, the OAT hypothesis "classify doesn't consume IDF" is too strong: classify DOES consume `idf_weights_by_rank` in the leaf phase, but `idf_weights_by_rank` is identical across the two compared models, so the ablation toggle never reaches classify-time IDF math.

## Detailed Findings

### 1. `beam_width`

#### 1.1. Field declaration and default

- Field: `pub beam_width: usize` ([src/types.rs:318](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L318))
- Doc: "Number of candidate paths to maintain during tree descent. 1 = greedy descent (original behavior). Higher values explore alternative paths at ambiguous nodes. Default 1." ([src/types.rs:315-317](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L315-L317))
- Default: `beam_width: 1` ([src/types.rs:357](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L357))

#### 1.2. Python binding

- pyo3 signature default: `beam_width = 1` ([src/lib.rs:142](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L142))
- Rust function parameter: `beam_width: usize,` ([src/lib.rs:165](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L165))
- `ClassifyConfig` struct init (shorthand): `beam_width,` ([src/lib.rs:189](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L189))

End-to-end binding verified.

#### 1.3. Read sites in `src/classify.rs`

Three read sites only:

1. **Dispatch gate** at the top of `classify_one_pass` ([src/classify.rs:178](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L178)):
   ```rust
   if config.beam_width > 1 {
       return classify_one_pass_beam(my_kmers, s, b, ts, config, full_length, ls, rng);
   }
   ```
   When `beam_width <= 1`, dispatch falls through to greedy descent (`src/classify.rs:182-285`); the value is never read again on that code path.

2. **Local capture** at top of `classify_one_pass_beam` ([src/classify.rs:317](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L317)): `let beam_width = config.beam_width;`. This site is reached only via the `> 1` branch above.

3. **Beam frontier prune** inside the descent loop ([src/classify.rs:504](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L504)): `next.truncate(beam_width);` — caps the size of the per-iteration expanded `next: Vec<BeamCandidate>` after sorting by score.

No other read sites exist in `src/classify.rs`. Not referenced in `leaf_phase_score`, `classify_sequential`, `classify_parallel`, the LCA-cap construction, or the descent fallback.

#### 1.4. What grows the beam frontier (the actual gating)

The `next` vector is built by the per-candidate expansion loop at `src/classify.rs:329-500`. Inside that loop at the multi-child decision branch:

- The winner child (`children_by_votes[0]`) is pushed to `next` at [src/classify.rs:434-449](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L434-L449), gated by `top_vote_frac >= config.min_descend` at [src/classify.rs:430](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L430).
- Runner-ups are pushed to `next` at [src/classify.rs:458-480](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L458-L480), each gated by `vf >= config.min_descend` at [src/classify.rs:460](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L460):
   ```rust
   for &(j, votes) in children_by_votes.iter().skip(1) {
       let vf = votes as f64 / b as f64;
       if vf < config.min_descend {
           break;  // children_by_votes is sorted, so all remaining fail
       }
       ...
       next.push(BeamCandidate { ... });
   }
   ```

The `min_descend` default is **0.98** ([src/types.rs:351](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L351)). For a runner-up to enter the beam frontier, it must capture vote-fraction `>= 0.98` of bootstrap replicates. Because tied replicates increment all tied children's vote counts (`vote_counts[j] += 1` for each `j` whose hit-row equals the per-replicate max at [src/classify.rs:382-386](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L382-L386)), it is possible for two children to each hit `vote_count >= 0.98 * b` only if nearly every replicate produces a tie between exactly those two children — a narrow scenario.

`tie_margin` is **never consulted** in the beam expansion code path (the read sites of `tie_margin` are in `leaf_phase_score` only, not in `classify_one_pass_beam`).

#### 1.5. Consistency with OAT observation

OAT reported `beam_width=2` and `beam_width=5` produced bit-identical predictions to baseline (presumably `beam_width=1`). This is consistent with the code:

- The `beam_width=1` path goes greedy via the early-return at `src/classify.rs:179`.
- The `beam_width >= 2` path goes into `classify_one_pass_beam`. Inside, the frontier is built one node at a time and pruned at `src/classify.rs:504`. With `min_descend = 0.98`, runner-ups rarely pass the gate at `src/classify.rs:460`, so `next` typically has size 1, and `next.truncate(2)` and `next.truncate(5)` are both no-ops.
- When `next` has size 1, the beam-search single-step output exactly matches greedy descent's single-step output (same winner, same score, same `next_margins`). So beam-with-trivial-frontier is functionally indistinguishable from greedy.

The OAT hypothesis pointed at `tie_margin` as the gate; the actual gate is `min_descend`. The conclusion (toggling `beam_width` is inert for the tested config) holds.

### 2. `leave_one_out`

#### 2.1. Field declarations

- On `TrainConfig`: `pub leave_one_out: bool` with doc "Exclude each sequence from its own node's profile during fraction learning (leave-one-out). Reduces self-classification bias for small groups. Default false (original behavior)." ([src/types.rs:261-264](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L261-L264)). Default `false` at [src/types.rs:290](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L290).
- On `LearnFractionsConfig`: `pub leave_one_out: bool` ([src/types.rs:126](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L126)). Copied from `TrainConfig` via the `From` impl at [src/types.rs:151](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L151).
- **NOT** on `BuildTreeConfig` ([src/types.rs:114-120](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L114-L120)) — this is significant; see §2.4.

#### 2.2. Python binding

For `train`:
- pyo3 signature default: `leave_one_out = false` ([src/lib.rs:68](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L68))
- Rust function parameter: `leave_one_out: bool,` ([src/lib.rs:85](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L85))
- `TrainConfig` struct init: `leave_one_out,` ([src/lib.rs:106](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L106))

For `learn_fractions_py` (the phase-3 entry point):
- pyo3 signature default: `leave_one_out = false` ([src/lib.rs:278](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L278))
- Rust function parameter: `leave_one_out: bool,` ([src/lib.rs:289](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L289))
- `LearnFractionsConfig` struct init: `leave_one_out,` ([src/lib.rs:295](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L295))

End-to-end binding verified for both entry points.

#### 2.3. Read site in `src/training.rs`

A single read site, in `_learn_fractions_inner` ([src/training.rs:377-626](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L377-L626)):

```rust
// src/training.rs:447-456
let loo_child_idx = if config.leave_one_out {
    subtrees.iter().enumerate().find(|(_, &st)| {
        prepared.classes[i].starts_with(&prepared.end_taxonomy[st])
    }).map(|(j, &st)| {
        let group_size = prepared.n_seqs[st];
        (j, group_size)
    })
} else {
    None
};
```

`loo_child_idx: Option<(usize, usize)>` is populated only when (a) the flag is true AND (b) one of the current node's child subtrees has a taxonomic prefix that the training sequence's class extends.

Consumed inside the per-child profile loop at [src/training.rs:469-492](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L469-L492):

```rust
// src/training.rs:483-488
if let Some((loo_j, group_size)) = loo_child_idx {
    if j == loo_j && group_size > 1 && group_size <= 5 {
        let scale = (group_size - 1) as f64 / group_size as f64;
        for w in &mut weights_j { *w *= scale; }
    }
}
```

The scaling is `(group_size - 1) / group_size`, applied uniformly to every element of the local `weights_j: Vec<f64>` (a copy or IDF-multiplied derivation of `dk.profiles[j]`, declared at [src/training.rs:470](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L470)). It does NOT mutate the stored `dk.profiles[j]`.

#### 2.4. What state actually depends on `leave_one_out`

Walking the construction of the returned `TrainingSet` at [src/training.rs:607-625](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L607-L625):

| Field | Source | Depends on `leave_one_out`? |
|---|---|---|
| `taxonomy`, `taxa`, `levels`, `children`, `parents`, `sequences`, `kmers`, `cross_index`, `k`, `inverted_index`, `idf_weights_by_rank`, `seed_pattern` | `prepared.*.clone()` | **No** — `_prepare_data_inner` does not read the flag |
| `decision_kmers` | `built_tree.decision_kmers.clone()` ([src/training.rs:619](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L619)) | **No** — see below |
| `fraction` | mutated by descent loop at [src/training.rs:550-570](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L550-L570) | Yes (when LOO scaling flips a vote-count winner) |
| `problem_sequences` | derived from `incorrect[i] != Some(false)` ([src/training.rs:589-598](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L589-L598)) | Yes (downstream of differing votes) |
| `problem_groups` | derived from `fraction[node].is_none()` ([src/training.rs:600-605](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L600-L605)) | Yes (downstream of `fraction`) |

The reason `decision_kmers` is independent is structural: `_build_tree_inner` ([src/training.rs:348-375](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L348-L375)) constructs its own `TrainConfig` from `BuildTreeConfig` fields and `..Default::default()`:

```rust
// src/training.rs:352-359
let train_config = TrainConfig {
    descendant_weighting: build_config.descendant_weighting,
    max_children: build_config.max_children,
    record_kmers_fraction: build_config.record_kmers_fraction,
    correlation_aware_features: build_config.correlation_aware_features,
    processors: build_config.processors,
    ..Default::default()
};
```

`..Default::default()` fills `leave_one_out: false` regardless of the user's setting (because `BuildTreeConfig` does not carry this flag). So the build-tree phase always runs with `leave_one_out = false`.

#### 2.5. Firing conditions

For the LOO scaling at `src/training.rs:485-487` to actually fire on any sequence/iteration:

1. `config.leave_one_out == true` ([src/training.rs:447](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L447))
2. The current decision node has more than one child (`subtrees.len() > 1` at [src/training.rs:437](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L437))
3. There exists a child subtree `st` whose `end_taxonomy[st]` is a prefix of the sequence's class ([src/training.rs:448-449](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L448-L449))
4. The current iteration of the per-child loop has `j == loo_j` ([src/training.rs:484](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L484))
5. `group_size > 1 && group_size <= 5` ([src/training.rs:484](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L484)), where `group_size = prepared.n_seqs[st]` (count of training sequences anywhere under that subtree).

The size cap of 5 is the tightest gate. On databases where every relevant subtree at decision points has either 1 sequence or more than 5, condition (5) fails for every iteration and the LOO branch is structurally inert.

Even when (5) passes, the scaling factor `(group_size - 1) / group_size` is in `{0.5, 0.667, 0.75, 0.8}` for `group_size` in `{2, 3, 4, 5}`. This must additionally tip a per-replicate `vote_counts[w]` winner ([src/training.rs:494-512](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L494-L512)) to actually change the descent outcome and therefore `fraction[node]`. If it never tips a winner, the descent decisions are byte-identical across the two flag values and so are `fraction`, `problem_sequences`, `problem_groups`.

#### 2.6. Consistency with OAT observation

Bit-identical model md5 between `leave_one_out=True` and `leave_one_out=False` is consistent with the code: most of the model bytes are guaranteed identical (decision_kmers, idf_weights_by_rank, all of `prepared`'s outputs), and the remaining flag-dependent fields (`fraction`, `problem_sequences`, `problem_groups`) only differ when the size-5 gate AND the vote-tipping condition both fire. On a database with no nodes in the 2–5 size range — or where the scaling never tips a winner — the model bytes are identical.

There is no missing wiring. The flag works as documented and as defined; it just has a narrow operating window.

### 3. `use_idf_in_training`

#### 3.1. Field declarations

- On `TrainConfig`: `pub use_idf_in_training: bool` with doc "Use IDF weights (instead of profile weights) during the fraction-learning tree descent. Makes training scoring match classification scoring. Default false (original behavior uses profile weights)." ([src/types.rs:257-260](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L257-L260)). Default `false` at [src/types.rs:289](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L289).
- On `LearnFractionsConfig`: `pub use_idf_in_training: bool` ([src/types.rs:125](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L125)), copied from `TrainConfig` via the `From` impl at [src/types.rs:150](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L150).
- **NOT** on `BuildTreeConfig`. Same containment as `leave_one_out`.

#### 3.2. Python binding

For `train`:
- pyo3 signature default: `use_idf_in_training = false` ([src/lib.rs:67](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L67))
- Rust function parameter ([src/lib.rs:84](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L84))
- `TrainConfig` struct init ([src/lib.rs:105](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L105))

For `learn_fractions_py`:
- pyo3 signature default ([src/lib.rs:278](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L278))
- Rust function parameter ([src/lib.rs:288](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L288))
- `LearnFractionsConfig` struct init ([src/lib.rs:294](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L294))

End-to-end binding verified for both entry points.

#### 3.3. Read site in `src/training.rs`

A single read site in `_learn_fractions_inner`, inside the per-child-subtree loop ([src/training.rs:469-492](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L469-L492)):

```rust
// src/training.rs:470-481
let mut weights_j: Vec<f64> = if config.use_idf_in_training {
    // weights = profiles[j] elementwise * idf_row at this depth
    ...
        prof * idf
    ...
} else {
    dk.profiles[j].clone()  // raw profile weights
};
```

The `idf_row` is selected at [src/training.rs:462-466](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L462-L466) and indexes into `prepared.idf_weights_by_rank` by descent depth.

`weights_j` is then consumed by `vector_sum(&matches, &weights_j, &sampling, b)` at [src/training.rs:490-491](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L490-L491), feeding the bootstrap vote at [src/training.rs:494-505](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L494-L505).

#### 3.4. The IDF matrix is computed unconditionally

`prepared.idf_weights_by_rank` is computed in `_prepare_data_inner` at [src/training.rs:313](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L313):

```rust
let idf_weights_by_rank = compute_idf_by_rank(&classes, &kmers, n_kmers);
```

This call is NOT gated by `use_idf_in_training`. The matrix is always produced and always written into the trained model at [src/training.rs:624](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L624). So `ts.idf_weights_by_rank` is byte-identical across the two flag settings.

#### 3.5. What state actually depends on `use_idf_in_training`

Walking `TrainingSet` construction at `src/training.rs:607-625` (same fields as §2.4):

| Field | Depends on `use_idf_in_training`? |
|---|---|
| All `prepared.*` fields including `idf_weights_by_rank` | **No** — `_prepare_data_inner` does not read the flag |
| `decision_kmers` | **No** — `_build_tree_inner` does not read the flag |
| `fraction` | Yes (via `weights_j` → `hits[j]` → `vote_counts` → winner → fraction adjustment) |
| `problem_sequences`, `problem_groups` | Yes (downstream of `fraction`) |

Same containment as `leave_one_out`: only `fraction`, `problem_sequences`, `problem_groups` differ when the flag is toggled.

#### 3.6. Classify-side IDF reads

Five hits for `idf` (case-insensitive) in `src/classify.rs`; three are real reads, two are comments.

**Real reads, all in `leaf_phase_score`** ([src/classify.rs:530-944](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L530-L944)):

1. **Row-length read** at [src/classify.rs:553](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L553): `let rank_idx = descent_depth.min(ts.idf_weights_by_rank.len().saturating_sub(1));`
2. **Row borrow** at [src/classify.rs:554](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L554): `let counts: &[f64] = &ts.idf_weights_by_rank[rank_idx];`
3. **Query-kmer weights** at [src/classify.rs:635-637](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L635-L637):
   ```rust
   let u_weights: Vec<f64> = u_sampling.iter()
       .map(|&uk| if uk > 0 && (uk as usize) <= counts.len() { counts[(uk - 1) as usize] } else { 0.0 })
       .collect();
   ```

`u_weights` is then passed to the matchers at [src/classify.rs:639-643](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L639-L643) (`parallel_match_inverted` or `parallel_match`), which accumulate into `hits_flat` weighted by `u_weights[i]`. The IDF row also drives the `davg` normalizer at [src/classify.rs:694-701](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L694-L701), and both feed the `tot_hits` accumulator at [src/classify.rs:732](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L732).

**The leaf-phase IDF read is unconditional** — there is no `ClassifyConfig` flag named `use_idf_in_classify` or similar; the IDF row is always selected and consumed.

#### 3.7. Classify-side descent does NOT consume IDF

The greedy descent in `classify_one_pass` calls `vector_sum` with raw `dk.profiles[j]` at [src/classify.rs:217](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L217):

```rust
let row = vector_sum(&matches, &dk.profiles[j], &sampling, b);
```

The beam descent in `classify_one_pass_beam` does the same at [src/classify.rs:376](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L376):

```rust
let row = vector_sum(&matches, &dk.profiles[j], &sampling, b);
```

Neither descent path multiplies by `idf_row` or reads `ts.idf_weights_by_rank`. The descent uses raw profile weights at classify time, regardless of how the model was trained.

#### 3.8. ClassifyConfig has no IDF gate

The `ClassifyConfig` struct ([src/types.rs:298-344](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L298-L344)) has 13 fields: `threshold`, `bootstraps`, `min_descend`, `full_length`, `processors`, `sample_exponent`, `length_normalize`, `rank_thresholds`, `beam_width`, `tie_margin`, `confidence_uses_descent_margin`, `sibling_aware_leaf`, `suppress_ancestor_only_groups`. None gates IDF use.

#### 3.9. Consistency with OAT observation

The OAT observation: different model md5 between `True`/`False`, but byte-identical predictions.

- **Why md5 differs**: `fraction`, `problem_sequences`, `problem_groups` differ across the two flag settings (when the training-time descent winner changes for at least one sequence/iteration). This is sufficient to change bincode bytes.
- **Why predictions can be byte-identical**: At classify time, the descent path uses raw `dk.profiles[j]` ([src/classify.rs:217, 376](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L217)), and `decision_kmers` is byte-identical across the two models. The leaf phase reads `ts.idf_weights_by_rank` ([src/classify.rs:553-554](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L553-L554)), but that matrix is also byte-identical. The only differing inputs at classify time are the `fraction` values, which are read in the descent paths at e.g. [src/classify.rs:200](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L200), [src/classify.rs:210](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L210), [src/classify.rs:300](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L300), [src/classify.rs:334](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L334), [src/classify.rs:368](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L368). Whether they flip a descent decision on the test set depends on the data.

So predictions equality is **data-dependent**, not structurally guaranteed. The OAT hypothesis "classify scoring doesn't consume the IDF weights" is too strong — classify DOES consume IDF in the leaf phase. The accurate statement is: **the IDF weights stored in the model are not influenced by `use_idf_in_training`**. The flag changes only descent-phase training math and so only `fraction`/`problem_sequences`/`problem_groups`.

### 4. Cross-cutting: `BuildTreeConfig` containment shapes which fields can be flag-dependent

Three of the four phase-config structs in `src/types.rs` are relevant:

- `BuildTreeConfig` ([src/types.rs:114-120](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L114-L120))
- `LearnFractionsConfig` ([src/types.rs:123-127](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L123-L127))
- `TrainConfig` ([src/types.rs:237-273](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L237-L273))

`BuildTreeConfig` does not carry `leave_one_out` or `use_idf_in_training`. When `_build_tree_inner` builds its `TrainConfig` from `BuildTreeConfig` ([src/training.rs:352-359](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L352-L359)) it spreads `..Default::default()`, which fixes both flags to `false` for the duration of that phase. Consequently:

- `decision_kmers` is invariant under either flag toggle.
- `idf_weights_by_rank` is invariant under either flag toggle (computed in `_prepare_data_inner`, also flag-independent).
- Only `fraction`/`problem_sequences`/`problem_groups` can possibly differ, and only when the descent loop's per-child weight differences tip a vote winner.

This containment is the structural reason both flags produce small, narrow effects on model bytes and zero downstream effect on classify-time scoring math.

## Code References

### Field declarations
- `src/types.rs:318` — `beam_width` field on `ClassifyConfig`
- `src/types.rs:357` — `beam_width` default `1`
- `src/types.rs:351` — `min_descend` default `0.98`
- `src/types.rs:264` — `leave_one_out` field on `TrainConfig`
- `src/types.rs:290` — `leave_one_out` default `false`
- `src/types.rs:260` — `use_idf_in_training` field on `TrainConfig`
- `src/types.rs:289` — `use_idf_in_training` default `false`

### Python binding (lib.rs)
- `src/lib.rs:142, 165, 189` — `beam_width` for `classify`
- `src/lib.rs:68, 85, 106` — `leave_one_out` for `train`
- `src/lib.rs:278, 289, 295` — `leave_one_out` for `learn_fractions_py`
- `src/lib.rs:67, 84, 105` — `use_idf_in_training` for `train`
- `src/lib.rs:278, 288, 294` — `use_idf_in_training` for `learn_fractions_py`

### Read sites
- `src/classify.rs:178` — `beam_width` dispatch gate (greedy vs. beam)
- `src/classify.rs:317` — `beam_width` capture in `classify_one_pass_beam`
- `src/classify.rs:504` — `beam_width` truncation of `next` frontier
- `src/classify.rs:460` — `min_descend` runner-up gate (controls beam frontier size)
- `src/training.rs:447` — `leave_one_out` read; gates `loo_child_idx` construction
- `src/training.rs:483-488` — LOO weight-scaling branch with size cap (group_size in 2..=5)
- `src/training.rs:470` — `use_idf_in_training` read; gates `weights_j` computation

### Classify-time IDF reads
- `src/classify.rs:553-554` — IDF row selection in `leaf_phase_score`
- `src/classify.rs:635-637` — `u_weights` built from IDF row
- `src/classify.rs:639-643` — `parallel_match[_inverted]` consumes `u_weights`
- `src/classify.rs:694-701` — `davg` from IDF row
- `src/classify.rs:732` — `tot_hits` accumulator (numerator and denominator both IDF-derived)

### Classify-time descent (no IDF)
- `src/classify.rs:217` — greedy descent uses `dk.profiles[j]` (raw)
- `src/classify.rs:376` — beam descent uses `dk.profiles[j]` (raw)

### Build-tree phase (forces both flags to false)
- `src/training.rs:352-359` — `_build_tree_inner` rebuilds `TrainConfig` with `..Default::default()`
- `src/types.rs:114-120` — `BuildTreeConfig` does not carry the flags

### TrainingSet construction (which fields depend on which flags)
- `src/training.rs:607-625` — `TrainingSet` construction
- `src/training.rs:614` — `fraction` (LOO-dependent, IDF-dependent)
- `src/training.rs:619` — `decision_kmers` (always from `built_tree`, flag-independent)
- `src/training.rs:620-621` — `problem_sequences`, `problem_groups`
- `src/training.rs:624` — `idf_weights_by_rank` (always from `prepared`, flag-independent)
- `src/training.rs:313` — `compute_idf_by_rank` called unconditionally

## Architecture Documentation

The codebase factors training into three phases: prepare → build_tree → learn_fractions. Each phase has its own config struct (`PrepareDataConfig`, `BuildTreeConfig`, `LearnFractionsConfig`) which is a strict subset of `TrainConfig`. The top-level `train` function ([src/training.rs](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs)) sequences the three phases and synthesizes the result. Two flags — `leave_one_out` and `use_idf_in_training` — only belong to the `LearnFractionsConfig` and `TrainConfig` shape; the `BuildTreeConfig` shape does not include them. The `_build_tree_inner` function rebuilds a fresh `TrainConfig` from `BuildTreeConfig` using `..Default::default()`, which structurally guarantees that the build-tree phase always runs with both flags off, regardless of the user's setting. This is the architectural reason `decision_kmers` (the largest, structurally most important model artifact) is identical across either flag toggle.

The classify side has a parallel split: the descent path (`classify_one_pass`, `classify_one_pass_beam`) uses raw stored profile weights from `dk.profiles[j]`; the leaf-phase scoring (`leaf_phase_score`) uses IDF-weighted query-kmer counts pulled from `ts.idf_weights_by_rank`. There is no `ClassifyConfig` flag controlling whether IDF is applied — IDF is unconditionally applied in the leaf phase and unconditionally absent from the descent.

The `beam_width` parameter dispatches on `> 1` at the top of `classify_one_pass` ([src/classify.rs:178](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L178)) into a separate beam-search variant. The frontier-pruning consumer at [src/classify.rs:504](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L504) is paired with a runner-up admission gate at [src/classify.rs:460](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L460) that requires `vote_fraction >= min_descend`. With `min_descend` defaulting to 0.98, the admission gate is the binding constraint on frontier size in practice.

## Historical Context (from thoughts/)

- `thoughts/shared/research/2026-04-27-oxidtaxa-vs-idtaxa-ablation-surface.md` — prior research on the ablation surface for oxidtaxa vs IDTAXA; relevant context for what the OAT study is comparing.
- `thoughts/shared/plans/2026-04-27-oxidtaxa-oat-ablation.md` — plan for the OAT ablation that produced the bug report.
- Recent commits show iterative beam-path tuning: `622ca33 Restrict beam runner-ups to pass min_descend` and `17b492b Fix I6 beam path + switch to per-rank (non-cumulative) margin application` — the runner-up gate at [src/classify.rs:460](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L460) was introduced in commit `622ca33`.

## Related Research

- `thoughts/shared/research/2026-04-27-oxidtaxa-vs-idtaxa-ablation-surface.md`

## Open Questions

- For `leave_one_out`: what is the size distribution of subtree groups (`prepared.n_seqs[st]`) in the OAT-tested database? If no decision-node child satisfies `2 <= group_size <= 5`, the flag is structurally inert on that DB. (This can be answered by inspecting `n_seqs` after a prepare phase.) **— resolved in §5 follow-up: vert12s has 19,490 LOO-eligible nodes (30.5%), and bytes are still identical, so size distribution is not the cause.**
- For `use_idf_in_training`: on the test set used by the OAT, do any `fraction` differences (between flag=true and flag=false) sit on the `min_descend` threshold (default 0.98) for any sequence's descent? If not, the difference in `fraction` values is structurally invisible to classify-time output for that test set.
- For `beam_width`: would the frontier ever reach size 2+ if `min_descend` were lowered? The runner-up gate at [src/classify.rs:460](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/classify.rs#L460) is the binding constraint; lowering `min_descend` would expose `beam_width` differences.

---

## Follow-up Research 2026-04-30 (afternoon)

### Trigger

A more thorough OAT run on the vert12s reference set produced sharper evidence than the original observation:

- 19,490 LOO-eligible internal nodes in size 2..=5 (= **30.5%** of all internal nodes) — the size-cap "narrow firing window" framing in §2.5 cannot explain absence-of-effect.
- `cmp` returns 0 (byte-identical model files) across all three communities tested:

  | Community | Model size | baseline md5 | leave_one_out=True md5 | diff |
  |-----------|-----------:|--------------|------------------------|------|
  | unified_0 | 310,467,653 bytes | 0c238b… | 0c238b… | identical |
  | unified_1 | 309,552,025 bytes | fd2b91… | fd2b91… | identical |
  | unified_2 | 310,950,757 bytes | cc72ae… | cc72ae… | identical |

- Two ~310 MB training computations actually ran (`leave_one_out=True` was honored as a separate `train()` invocation with its own output directory).

The follow-up question: trace the parameter from `train()` entry through to `decision_kmers` (and to whatever else it touches), and document why bytes are identical despite a 30.5% eligible-node rate.

### 5.1. Verification of the structural claims (current commit `f36cd2e`)

| Claim | Verdict | Citation |
|---|---|---|
| `BuildTreeConfig` does not carry `leave_one_out` | **Confirmed** | [src/types.rs:114-120](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L114-L120) — five fields: `record_kmers_fraction`, `descendant_weighting`, `correlation_aware_features`, `max_children`, `processors`. No `leave_one_out`. |
| `..Default::default()` at `training.rs:358` forces `leave_one_out=false` in the temp `TrainConfig` used by `_build_tree_inner` | **Confirmed** | [src/training.rs:352-359](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L352-L359) — fills the unspecified flags from `TrainConfig::default()` ([src/types.rs:275-295](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L275-L295)), where `leave_one_out: false` is the default. |
| `decision_kmers` are computed without ever seeing `leave_one_out` | **Confirmed** | `create_tree` ([src/training.rs:828-1185](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L828-L1185)) is the function that produces every `DecisionNode`; a full-file grep finds zero references to `leave_one_out` inside it (the only `config.leave_one_out` read in the file is at line 447, inside `_learn_fractions_inner`). |
| `train()` accepts `leave_one_out` as a parameter | **Confirmed** | pyo3 signature default at [src/lib.rs:68](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L68); Rust parameter at [src/lib.rs:85](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L85); `TrainConfig` field-init at [src/lib.rs:106](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L106). |

All four structural claims hold against the current `main` (commit `f36cd2e`).

### 5.2. Where the parameter actually flows from `train()` entry

The full path through the single-shot `train()` Python entry point:

1. `train(... leave_one_out=true ...)` ([src/lib.rs:62-119](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L62-L119)) constructs a `TrainConfig` with `leave_one_out: true` ([src/lib.rs:99-110](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L99-L110)). The flag enters Rust correctly.
2. `learn_taxa(&filtered_seqs, &filtered_tax, &config, seed, verbose)` ([src/lib.rs:114](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L114)) → `_learn_taxa_inner(...)` ([src/training.rs:42](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L42)).
3. `_learn_taxa_inner` ([src/training.rs:628-640](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L628-L640)) makes three calls in order:
   - `_prepare_data_inner(...)` — does not read `leave_one_out` (`_prepare_data_inner` doesn't accept the field at all, per [src/training.rs:94-100](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L94-L100)).
   - `_build_tree_inner(&prepared, &BuildTreeConfig::from(config))` ([src/training.rs:638](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L638)). The `From` impl at [src/types.rs:135-143](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L135-L143) **drops** `leave_one_out` — it is not in `BuildTreeConfig`'s shape. `_build_tree_inner` then re-synthesizes a fresh `TrainConfig` with `..Default::default()` ([src/training.rs:352-359](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L352-L359)), which fills `leave_one_out: false` regardless of the user's setting. `create_tree` is invoked with this `false` config ([src/training.rs:362-369](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L362-L369)) but does not read the field anyway.
   - `_learn_fractions_inner(&prepared, &built_tree, &LearnFractionsConfig::from(config), seed)` ([src/training.rs:639](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L639)). The `From` impl at [src/types.rs:146-159](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L146-L159) **does** copy `leave_one_out: c.leave_one_out` ([src/types.rs:151](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L151)).
4. Inside `_learn_fractions_inner`, `config.leave_one_out` is read at [src/training.rs:447](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L447) to populate `loo_child_idx`, and consumed at [src/training.rs:483-488](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L483-L488) to multiply every element of `weights_j` by `(group_size - 1) / group_size`.

**Net flow**:
- `decision_kmers`: flag is dropped before reaching `create_tree` (because `BuildTreeConfig` doesn't carry it AND `..Default::default()` overwrites it) **and** `create_tree` doesn't read it anyway. Two independent reasons; either alone would suffice.
- `fraction` / `problem_sequences` / `problem_groups`: flag is preserved through `LearnFractionsConfig::from(c)` and IS read at [src/training.rs:447](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L447). The "silently dropped at the build-tree phase" framing applies only to `decision_kmers`; the fraction-learning path is wired correctly per the Rust source.

### 5.3. Why bytes are identical despite the flag firing — `vector_sum` normalization

The LOO scaling at [src/training.rs:483-488](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L483-L488):

```rust
if let Some((loo_j, group_size)) = loo_child_idx {
    if j == loo_j && group_size > 1 && group_size <= 5 {
        let scale = (group_size - 1) as f64 / group_size as f64;
        for w in &mut weights_j { *w *= scale; }
    }
}
```

multiplies **every element** of `weights_j` (a `Vec<f64>`) by a single scalar `scale ∈ {0.5, 0.667, 0.75, 0.8}`. This is uniform scaling.

`weights_j` is then passed (and only passed) to `vector_sum` at [src/training.rs:491](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L491):

```rust
hits[j] = vector_sum(&matches, &weights_j, &sampling, b);
```

The relevant body of `vector_sum` is at [src/matching.rs:40-55](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/matching.rs#L40-L55):

```rust
for i in 0..block_count {
    let mut cur_weight = 0.0;
    let mut max_weight = 0.0;
    for k in 0..block_size {
        let idx = sampling[i * block_size + k];
        max_weight += weights[idx];
        if matches[idx] {
            cur_weight += weights[idx];
        }
    }
    result[i] = if max_weight > 0.0 {
        cur_weight / max_weight
    } else {
        0.0
    };
}
```

For a single replicate `i`:
- `max_weight = Σ_k weights[sampling[i·B+k]]` — sum over all `block_size` sampled positions
- `cur_weight = Σ_{k : matches[sampling[i·B+k]]} weights[sampling[i·B+k]]` — the matching subset of the same sum
- `result[i] = cur_weight / max_weight` — a ratio in `[0, 1]`

If `weights` is multiplied elementwise by a scalar `c ≠ 0`:
- `cur_weight' = c · cur_weight`
- `max_weight' = c · max_weight`
- `result'[i] = (c · cur_weight) / (c · max_weight) = cur_weight / max_weight = result[i]`

The output is **invariant** under uniform scaling of the weights vector.

Therefore, every per-replicate value in `hits[j]` ([src/training.rs:491](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L491)) is byte-identical between `leave_one_out=true` and `leave_one_out=false`. Walking forward through `_learn_fractions_inner`:

- `vote_counts` ([src/training.rs:494-505](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L494-L505)) — derived from `hits[j][rep]` per replicate — byte-identical.
- Descent winner `w` ([src/training.rs:507-512](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L507-L512)) — argmax over `vote_counts` — byte-identical.
- The `correct` and `pred` outputs of the per-sequence inner loop ([src/training.rs:518-523](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L518-L523)) — byte-identical.
- `node_failures`, fraction adjustments at [src/training.rs:550-570](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L550-L570) — byte-identical.
- The `fraction`, `problem_sequences`, `problem_groups` fields of the returned `TrainingSet` ([src/training.rs:614, 620-621](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L614)) — byte-identical.

Combined with `decision_kmers` and `idf_weights_by_rank` already shown to be flag-independent (§2.4), every field of the saved `TrainingSet` is byte-identical, and the bincode-serialized model bytes are byte-identical. This matches the `cmp` evidence.

### 5.4. Updated explanation (supersedes §2.5/2.6 framing)

The §2.5 framing said the LOO scaling _might_ fail to tip a vote-count winner for narrow data reasons. The follow-up evidence (19,490 eligible nodes — 30.5% of all internal nodes) plus the `vector_sum` normalization (§5.3) converts that into a structural statement:

> **Uniform scaling of `weights_j` produces byte-identical `hits[j]` vectors regardless of the scale factor or how many nodes are eligible. The current LOO scaling at `src/training.rs:483-488` is mathematically a no-op for `vector_sum`-based scoring, because `vector_sum` outputs `cur_weight / max_weight` per replicate.**

The §2.5 size-cap framing is not the binding constraint. The §2.6 conclusion that `leave_one_out` "works as documented and as defined" is true only of its declaration and parameter routing — the runtime effect on `vector_sum`'s output is structurally zero, so the downstream `fraction`/`problem_sequences`/`problem_groups` are byte-identical for any database, not just databases with no 2..=5 sized groups.

The two-fold "silently dropped" path the user identified for `decision_kmers` (drop-at-`From`, then overwrite-by-`..Default::default()`) is real, but secondary: even if the flag reached `create_tree`, the function never reads it, so `decision_kmers` would be unchanged. The architecturally relevant fact is the interaction between the uniform-scaling LOO formula and the ratio-output `vector_sum`.

### 5.5. Citations to include in a GitHub issue

If the issue is filed against the `leave_one_out` behavior:

- The flag declaration: [src/types.rs:264](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L264) (`TrainConfig.leave_one_out`), [src/types.rs:126](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L126) (`LearnFractionsConfig.leave_one_out`).
- Python entry: [src/lib.rs:68](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L68), [src/lib.rs:85](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L85), [src/lib.rs:106](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/lib.rs#L106).
- Build-tree drop sites (real, but secondary as explained above):
  - `BuildTreeConfig` shape: [src/types.rs:114-120](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L114-L120).
  - `From<&TrainConfig> for BuildTreeConfig` (drops `leave_one_out`): [src/types.rs:135-143](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L135-L143).
  - `..Default::default()` reset: [src/training.rs:352-359](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L352-L359), defaults at [src/types.rs:275-295](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L275-L295).
- Fraction-learning use sites (the actually-firing path):
  - `LearnFractionsConfig::from` preserves the flag: [src/types.rs:146-159](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L146-L159) (line 151).
  - Read at [src/training.rs:447](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L447); uniform-scale weights at [src/training.rs:483-488](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L483-L488).
- The normalization that neutralizes the scaling:
  - `hits[j] = vector_sum(&matches, &weights_j, &sampling, b)` at [src/training.rs:491](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L491).
  - `vector_sum` returns `cur_weight / max_weight` per replicate at [src/matching.rs:50-54](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/matching.rs#L50-L54).
- Tests asserting LOO produces "valid" models but not asserting it changes them ([tests/test_algorithmic_improvements.rs:201-228](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/tests/test_algorithmic_improvements.rs#L201-L228)): the `assert_eq!(loo_model.taxonomy.len(), default_model.taxonomy.len())` and `decision_kmers.keep` equality assertions at lines 217 and 223 are consistent with byte-identical outputs and would pass even if `fraction` were also identical (they don't compare it).

### 5.6. What changes in the document overall

§2.5's "narrow firing window" framing is preserved as the literal scope of the size-cap (`group_size in 2..=5`) but is **not** the binding cause of byte-identical models. §5.3's `vector_sum` normalization is. The document's earlier conclusion ("no genuine wiring bug") still holds in the narrow sense that every binding-site reads the flag it's supposed to read — but the runtime effect of the LOO formula is structurally zero, which is a stronger and more useful framing for an issue report.

---

## §6. Fix-strategy analysis

This section captures the design discussion around the actual fix — what the candidate approaches are, what they trade off, and why singleton-k-mer behavior is a feature rather than a bug.

### 6.1. Is this a bug to fix?

Yes. The flag is declared, documented, parameter-routed end-to-end, has tests, and is offered as a tuning knob — but the runtime effect on `vector_sum`'s output is structurally zero, so toggling it never changes a model byte. That diverges from the documented contract at [src/types.rs:261-263](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L261-L263) ("Reduce self-classification bias for small groups"). Independently flagged as `CRITICAL BUG` in the April 15 audit at [thoughts/shared/research/2026-04-15-new-parameter-audit.md](thoughts/shared/research/2026-04-15-new-parameter-audit.md#L34) and as an unfinished "Pass 2" in the April 13 plan at [thoughts/shared/plans/2026-04-13-algorithmic-improvements.md](thoughts/shared/plans/2026-04-13-algorithmic-improvements.md#L389-L398).

The downstream consequence to flag for any reviewer: the OAT/Optuna sweep findings that "found" `leave_one_out=True` optimal on Vert12s (per [thoughts/shared/research/2026-04-21-three-marker-sweep-findings.md](thoughts/shared/research/2026-04-21-three-marker-sweep-findings.md#L122)) are noise — the parameter did nothing. Any sweep result citing LOO needs re-running once the fix lands.

### 6.2. Candidate fixes

The fix has to break the scale invariance of `vector_sum`. Uniform multiplication of `weights_j` cancels in `cur_weight / max_weight`, so any correct fix needs to change the **shape** of the matching sibling's weight vector, not just rescale it.

#### Option A — asymmetric uniform scaling on matched positions

Modify [src/training.rs:483-488](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L483-L488) to scale only weights at matched positions:

```rust
if let Some((loo_j, group_size)) = loo_child_idx {
    if j == loo_j && group_size > 1 && group_size <= 5 {
        let scale = (group_size - 1) as f64 / group_size as f64;
        for (k, w) in weights_j.iter_mut().enumerate() {
            if matches[k] {
                *w *= scale;
            }
        }
    }
}
```

Effect on the bootstrap ratio: with `cur_weight` and `max_weight` defined per [src/matching.rs:40-55](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/matching.rs#L40-L55):
- `cur_weight' = scale · cur_weight` (matched positions are scaled)
- `max_weight' = scale · cur_weight + (max_weight - cur_weight) = max_weight - (1 - scale) · cur_weight`
- Ratio drops by ≈ `(1 - scale) · cur_weight · (max_weight - cur_weight) / (max_weight · max_weight')`

Numerical example: `n = 5` (`scale = 0.8`), `cur = 0.5 · max` → new ratio ≈ `0.444` vs. original `0.5` (~11% drop). `n = 2` (`scale = 0.5`), `cur = 0.5 · max` → new ratio ≈ `0.333` (33% drop).

Pros: ~5 lines; no `DecisionNode` shape change; existing bincode models still load; existing tests still compile.

Cons: applies a uniform scale across matched positions, ignoring per-kmer specificity. Singleton k-mers (only sequence i has them) are scaled to `0.8 ·` original, not zeroed. Conserved k-mers (every group member has them) are also scaled, even though true LOO would leave them unchanged. So A breaks the no-op but doesn't capture the structure of LOO.

#### Option B — per-kmer LOO with stored sibling totals

Extend `DecisionNode` ([src/types.rs:5-11](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/types.rs#L5-L11)) to retain enough state to reconstruct the LOO profile per-kmer:

```rust
pub struct DecisionNode {
    pub keep: Vec<i32>,
    pub profiles: Vec<Vec<f64>>,
    /// Raw count vectors per child subtree, aligned with `keep`. Each entry is
    /// the number of training sequences in that subtree containing the
    /// corresponding kept k-mer. Required for exact LOO at fraction-learning.
    pub raw_counts: Vec<Vec<f64>>,
    /// Total k-mer presence count per child subtree (sum over all k-mers,
    /// not just kept). Used as the LOO profile denominator.
    pub raw_totals: Vec<f64>,
}
```

`create_tree` ([src/training.rs:828-1185](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L828-L1185)) populates these alongside `profiles`; the leaf branch already computes both `counts` (renamed `raw_counts`) and `total` ([src/training.rs:1166-1178](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L1166-L1178)) and currently discards the unnormalized form. The internal-node branch can carry these through the merge analogously.

At fraction-learning, replace [src/training.rs:483-488](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L483-L488) with a per-kmer recomputation:

```rust
if let Some((loo_j, _group_size)) = loo_child_idx {
    if j == loo_j {
        let kmers_i_count = prepared.kmers[i].len() as f64;
        let new_total = dk.raw_totals[j] - kmers_i_count;
        if new_total > 0.0 {
            for (k, w) in weights_j.iter_mut().enumerate() {
                let count = dk.raw_counts[j][k];
                let i_k = if matches[k] { 1.0 } else { 0.0 };
                *w = (count - i_k) / new_total;
            }
        }
    }
}
```

(With `use_idf_in_training`, the same per-kmer formula then multiplies by `idf_row[k]`.)

Pros:
- **Mathematically exact at leaf siblings** — `count[k] / total` is the actual leaf profile, and `(count[k] - I_k) / (total - K_i)` is the actual LOO leaf profile.
- **Captures k-mer specificity correctly**: a singleton-at-the-sibling k-mer (`count[k] = 1` and `k ∈ kmers[i]`) goes to `0`. A conserved k-mer (`count[k] = n`) becomes `(n-1) / (total - K_i)` ≈ `count[k] / total` (unchanged after renormalization). A partially shared k-mer falls smoothly between.
- **No size cap needed** — formula degrades gracefully with `n`; large groups naturally see smaller effects because most k-mers are conserved.

Cons:
- `DecisionNode` shape change → bincode incompatibility → existing serialized models become invalid. Requires either a versioned format or a rebuild.
- `raw_counts` ~doubles `DecisionNode` storage (one extra `Vec<f64>` per child the same length as `profiles[j]`).
- Approximate at internal siblings: the `raw_counts`/`raw_total` representation flattens an internal sibling as if it were a leaf with fractional sequence counts. The formula remains well-defined and captures the qualitative LOO structure, but it doesn't exactly invert the descendant-weighted merge.

#### Recommendation

- With backward-compat / effort constraints **in**: option A. Breaks the no-op, ships in a small PR, no migration.
- Without those constraints (max correctness): option B. The k-mer specificity behavior is the actual semantic of LOO.

The user's framing in conversation — "if we don't care at all about backwards compatibility or effort to implement" — selects B.

### 6.3. Size-cap analysis (`2 <= n <= 5`)

The current code at [src/training.rs:484](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/src/training.rs#L484) gates LOO on `group_size > 1 && group_size <= 5`.

#### Lower bound (`n > 1`) is required

At `n = 1`, sequence `i` is the **only** member of sibling `loo_j`'s subtree. After LOO removal:
- `new_total = total - |kmers[i]| = 0` (i was the only contributor) — divide-by-zero.
- Even ignoring the numerical issue, the sibling has no profile to classify against. The subtree's LOO profile is undefined.

The existing `n > 1` floor is correct under both A and B and must stay.

#### Upper bound (`n <= 5`) is unnecessary under option B

The `n <= 5` cap was a workaround for the uniform-scaling pathology in the original implementation:
- With uniform scaling by `(n - 1) / n`, the factor saturates near `1` quickly (`0.99` at `n = 100`). The cap was a "don't bother" heuristic.
- The April 13 plan describes it as "for large groups the bias is negligible" ([thoughts/shared/plans/2026-04-13-algorithmic-improvements.md](thoughts/shared/plans/2026-04-13-algorithmic-improvements.md#L368)).

Under option B, the cap is no longer mathematically motivated:
- Singleton k-mers go to `0` regardless of `n` — the same correction at `n = 2` as at `n = 2000`.
- Conserved k-mers stay unchanged regardless of `n`.
- Effect *magnitude* still diminishes with `n` (because most k-mers are shared in large groups), but it diminishes **gracefully** rather than being capped artificially.

Compute cost of dropping the cap: per-iteration LOO is `O(|dk.keep|)` per matching sibling. On vert12s, `30.5%` of internal nodes satisfy `n in 2..=5`; without the cap, ~99% of internal nodes (those with `n > 1`) get LOO applied. That's a ~3× constant factor on the LOO branch but doesn't change overall complexity — fraction-learning is dominated by `vector_sum` and `int_match` regardless.

Recommendation under option B: drop the upper cap. Optionally expose `loo_max_group_size: Option<usize>` (default `None` = no cap) for safety re-capping if profiling later shows it matters.

### 6.4. Singleton k-mer rationale (why zeroing them is correct)

A natural objection to option B: "singleton k-mers (only sequence `i` has them in `loo_j`'s subtree) are highly discriminative — won't zeroing them throw away useful signal?"

The objection is true in the abstract (those k-mers ARE discriminative), but it conflates two distinct phases:

1. **Fraction-learning (training-time threshold calibration)**: LOO branch applies. Sequence `i` is being self-tested against its own training group. A singleton k-mer at `loo_j` is in `loo_j`'s profile *only because `i` contributed it*, and it matches the query *only because the query IS `i`*. That's circular evidence — the classifier grading its own homework. Zeroing it removes the circular contribution; the resulting harder self-test calibrates `fraction[node]` honestly.

2. **Classification (real query against trained model)**: LOO does **not** apply. The full `dk.profiles[j]` is used (without subtraction). Sequence `i`'s singleton k-mers remain in the profile permanently. A new query `q ≠ i` that happens to share one of those k-mers benefits from the discriminative signal, and that benefit is non-circular because `q` ≠ `i`.

The chain:
- Training, LOO branch: zero out `i`'s circular contributions → harder self-test → fractions relax to a lower threshold.
- Classification, no LOO: full profile, all k-mers including the original singletons, evaluated against the relaxed fractions → real queries get a fair chance.

Concretely: the LOO branch is the in-loop analog of cross-validation. Without it, fractions calibrate against an over-fit self-test (passes trivially because every sequence's discriminative k-mers are in its own group's profile), so the trained thresholds are too tight, and out-of-sample queries fail more often than they should. With proper LOO, the self-test is honest, the calibrated thresholds reflect actual classification difficulty, and out-of-sample performance improves — *without losing any of the discriminative information stored in `dk.profiles`*.

This is the architectural reason singleton-zeroing is a feature rather than a bug under option B: the model retains the full discriminative power for inference; only the threshold-calibration phase strips circular evidence to measure self-classification difficulty honestly.

### 6.5. Test-suite gap

The existing test [tests/test_algorithmic_improvements.rs:201-228](https://github.com/eDNA-Explorer/oxidtaxa/blob/f36cd2eb90a7f5d9fa072f03d422b83375773f17/tests/test_algorithmic_improvements.rs#L201-L228) asserts:

```rust
assert_eq!(loo_model.taxonomy.len(), default_model.taxonomy.len());
// LOO shouldn't affect tree structure, only fractions
assert_eq!(loo_model.decision_kmers.len(), default_model.decision_kmers.len());
for (d, l) in default_model.decision_kmers.iter().zip(loo_model.decision_kmers.iter()) {
    match (d, l) {
        (Some(dk_d), Some(dk_l)) => assert_eq!(dk_d.keep, dk_l.keep),
        ...
    }
}
```

These all pass under the current no-op implementation: `decision_kmers.keep` is identical because the build-tree phase doesn't read the flag, and `taxonomy.len()` is data-derived. The test never compares `fraction`, `problem_sequences`, or `problem_groups` — the three fields LOO is *supposed* to alter — so byte-identical model bytes between `leave_one_out=true` and `leave_one_out=false` would silently pass.

A regression test for either option A or B should assert divergence on `fraction`:

```rust
let frac_default: Vec<Option<f64>> = default_model.fraction.clone();
let frac_loo: Vec<Option<f64>> = loo_model.fraction.clone();
assert_ne!(frac_default, frac_loo,
    "leave_one_out=true should produce different fraction values; \
     a regression here means LOO is silently inert");
```

This single `assert_ne!` would have caught the no-op months ago. It is an obligatory addition to either fix.
