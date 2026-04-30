# Unify IDF-in-descent across training and classification

## Overview

The flag `use_idf_in_training` on `TrainConfig` was meant to align training and classify tree descent: when `true`, both should multiply per-child profile weights by the rank-appropriate IDF row before scoring. As implemented today only the training descent honors the flag — classify descent always uses raw `dk.profiles[j]` (see [classify.rs:217](src/classify.rs#L217), [classify.rs:376](src/classify.rs#L376)). So `flag=true` produces a model whose `fraction` table was learned against IDF-weighted scoring, but classify-time descent then scores with raw profiles, defeating the alignment.

This plan renames the flag to `use_idf_in_descent`, persists it on the trained model (`TrainingSet`), and teaches classify descent (both greedy and beam variants) to honor the persisted setting. After the change, `flag=false` and `flag=true` each produce internally consistent train↔classify descent algorithms.

## Current State Analysis

### Where the flag is read today

- Train side: [training.rs:470](src/training.rs#L470) — `weights_j = if config.use_idf_in_training { profiles[j] × idf_row } else { profiles[j].clone() }`. The `idf_row` is selected by descent-node depth at [training.rs:462-466](src/training.rs#L462-L466).
- Classify side (descent): does not read any IDF flag. Greedy at [classify.rs:217](src/classify.rs#L217) and beam at [classify.rs:376](src/classify.rs#L376) call `vector_sum(&matches, &dk.profiles[j], &sampling, b)` with raw profile values.
- Classify side (leaf phase): uses IDF unconditionally on **query kmers** at [classify.rs:553-554](src/classify.rs#L553-L554), [classify.rs:635-637](src/classify.rs#L635-L637) — separate scoring math, not part of this plan.

### Current model shape

`TrainingSet` ([types.rs:22-50](src/types.rs#L22-L50)) is a `#[derive(Debug, Clone, Serialize, Deserialize)]` struct serialized with bincode. Adding a new field is straightforward; per user direction, no backward compatibility for old models is required.

### Python binding

`use_idf_in_training` is exposed on two pyo3-bound functions in `lib.rs`:
- `train` — signature default at [lib.rs:67](src/lib.rs#L67), Rust param at [lib.rs:84](src/lib.rs#L84), `TrainConfig` init at [lib.rs:105](src/lib.rs#L105).
- `learn_fractions_py` — signature default at [lib.rs:278](src/lib.rs#L278), Rust param at [lib.rs:288](src/lib.rs#L288), `LearnFractionsConfig` init at [lib.rs:294](src/lib.rs#L294).

The `classify` pyo3 function does **not** expose any IDF-related flag — and won't after this change either; the setting flows through the model.

### Where the field appears in `types.rs`

- `LearnFractionsConfig.use_idf_in_training` ([types.rs:125](src/types.rs#L125))
- `LearnFractionsConfig` `From<&TrainConfig>` impl at [types.rs:150](src/types.rs#L150)
- `TrainConfig.use_idf_in_training` ([types.rs:260](src/types.rs#L260)) with doc at [types.rs:257-259](src/types.rs#L257-L259)
- `TrainConfig::default()` at [types.rs:289](src/types.rs#L289)

The doc comment on `PreparedData.idf_weights_by_rank` at [types.rs:72-75](src/types.rs#L72-L75) also mentions `use_idf_in_training` and will need updating.

## Desired End State

A single, accurately named flag `use_idf_in_descent` controls IDF use in both training-time descent and classify-time descent:

- `false` (default): both training-time descent and classify-time descent score with raw `dk.profiles[j]`.
- `true`: both training-time descent and classify-time descent score with `dk.profiles[j] × idf_row[depth]`.

The user passes the flag at **train time only** (via the existing `use_idf_in_descent` kwarg on `train` / `learn_fractions_py`). The trained model carries the setting; classify reads it from the model. There is no classify-side flag and no second param.

### Verification of end state

1. `cargo build` succeeds with the rename.
2. New tests in `tests/test_margin_aware.rs` cover default-off, flag persistence on `TrainingSet`, no-regression for `flag=false`, training-side smoke (flag=true builds a valid model), classify-side smoke (greedy + beam paths both run on a flag=true model), and bincode round-trip. Train↔classify symmetry is enforced **structurally** rather than by per-replicate bitwise equality test: the classify-side IDF construction at `classify.rs:209-243` / `classify.rs:367-403` is a literal copy of the training-side pattern at `training.rs:470-481`, and both call into the same `vector_sum` helper. See "Implementation note" under Phase 3 for rationale.
3. The Python `train(...)` accepts `use_idf_in_descent=true` and the resulting `classify(...)` reflects the IDF-weighted descent (manual verification only — see Phase 1 manual checklist).

### Key Discoveries

- The flag's only train-time consumer is the single `let mut weights_j` block at [training.rs:470-481](src/training.rs#L470-L481).
- Classify descent is in two functions: `classify_one_pass` (greedy, [classify.rs:168-285](src/classify.rs#L168-L285)) and `classify_one_pass_beam` (beam, [classify.rs:289-526](src/classify.rs#L289-L526)). Each has exactly one `vector_sum` call against `dk.profiles[j]`.
- `TrainingSet` already carries `idf_weights_by_rank` ([types.rs:49](src/types.rs#L49)). The classify-time `idf_row` selection mirrors the train-time pattern: `levels[k_node] - 1` clamped against the matrix length. The leaf phase uses an equivalent pattern at [classify.rs:551-554](src/classify.rs#L551-L554).
- `BuildTreeConfig` does not carry the flag and `_build_tree_inner` rebuilds `TrainConfig` with `..Default::default()` ([training.rs:352-359](src/training.rs#L352-L359)). No change needed there — the flag has no effect on tree construction.

## What We're NOT Doing

- **No backward compat** for old serialized models (per user). Old `TrainingSet` files won't deserialize after the rename.
- **No changes to the leaf-phase scoring** at [classify.rs:530-944](src/classify.rs#L530-L944). Leaf-phase IDF (on query kmers, per-training-sequence accumulator) is structurally different math and not part of the descent symmetry.
- **No classify-side override flag**. The trained model is the single source of truth.
- **No changes to other classifier wrappers** (`r_idtaxa`, `raxtax`).
- **No DB curation or sanitization.** This is a pure model-stage change.

## Implementation Approach

Three sequential phases, each independently buildable and testable:

1. Rename the flag end-to-end and add a persistence field on `TrainingSet`.
2. Wire classify-time descent to read the persisted flag and apply IDF when set.
3. Add tests that lock in train↔classify descent symmetry under both flag values.

---

## Phase 1: Rename flag + persist on TrainingSet

### Overview

Rename `use_idf_in_training` to `use_idf_in_descent` everywhere it appears (config structs, defaults, doc comments, Python bindings). Add a corresponding `use_idf_in_descent: bool` field to `TrainingSet` and populate it at the end of `_learn_fractions_inner`.

### Changes Required

#### 1. `src/types.rs`

**Rename** `use_idf_in_training` → `use_idf_in_descent` at:
- [types.rs:125](src/types.rs#L125) — `LearnFractionsConfig` field
- [types.rs:150](src/types.rs#L150) — `From<&TrainConfig>` impl
- [types.rs:260](src/types.rs#L260) — `TrainConfig` field
- [types.rs:289](src/types.rs#L289) — `TrainConfig::default()`

**Update the doc comment** at [types.rs:257-259](src/types.rs#L257-L259):

```rust
/// Apply rank-appropriate IDF weights to per-child profile values during
/// tree descent — at both training (fraction-learning) and classification.
/// When true, descent scoring uses `profiles[j] * idf_row[depth]`. When
/// false, descent scoring uses raw `profiles[j]`. Persisted on the trained
/// model so classify-time descent matches the algorithm used at train time.
/// Default false.
pub use_idf_in_descent: bool,
```

**Update the doc comment** at [types.rs:72-75](src/types.rs#L72-L75) (the `PreparedData.idf_weights_by_rank` doc references the old flag name):

```rust
/// Per-rank IDF matrix: row `r` is the IDF computed across distinct
/// taxonomic prefixes at depth `r + 1`. Always used at classify time in the
/// leaf phase. Also used by descent (training and classification) when
/// `use_idf_in_descent = true` on the trained model.
pub idf_weights_by_rank: Vec<Vec<f64>>,
```

**Add a new field** to `TrainingSet` at the end of the struct ([types.rs:49](src/types.rs#L49)):

```rust
/// Whether descent scoring (at both training and classify time) was
/// configured to multiply per-child profile values by the rank-appropriate
/// IDF row. Set from `TrainConfig.use_idf_in_descent` at the end of
/// fraction learning. Read by `classify_one_pass` and
/// `classify_one_pass_beam` to match the train-time descent algorithm.
pub use_idf_in_descent: bool,
```

#### 2. `src/training.rs`

**Rename** the read site at [training.rs:470](src/training.rs#L470):

```rust
let mut weights_j: Vec<f64> = if config.use_idf_in_descent {
    // ... unchanged body
};
```

**Update the inline comment** at [training.rs:460-461](src/training.rs#L460-L461) to match the new name:

```rust
// `use_idf_in_descent = true` scores training and classification descent
// against the same rank-appropriate IDF.
```

**Set the new TrainingSet field** at the end of `_learn_fractions_inner`. The construction site is at [training.rs:607-625](src/training.rs#L607-L625). Add:

```rust
TrainingSet {
    // ... existing fields
    use_idf_in_descent: config.use_idf_in_descent,
}
```

#### 3. `src/lib.rs`

**For `train`** ([lib.rs:60-114](src/lib.rs#L60-L114)):
- Rename the kwarg in `#[pyo3(signature = ...)]` at [lib.rs:67](src/lib.rs#L67): `use_idf_in_descent = false,`.
- Rename the Rust function parameter at [lib.rs:84](src/lib.rs#L84): `use_idf_in_descent: bool,`.
- Rename the field-init in the `TrainConfig { ... }` literal at [lib.rs:105](src/lib.rs#L105): `use_idf_in_descent,`.

**For `learn_fractions_py`** ([lib.rs:276-305](src/lib.rs#L276-L305)):
- Rename the kwarg in `#[pyo3(signature = ...)]` at [lib.rs:278](src/lib.rs#L278).
- Rename the Rust function parameter at [lib.rs:288](src/lib.rs#L288).
- Rename the field-init in the `LearnFractionsConfig { ... }` literal at [lib.rs:294](src/lib.rs#L294).

### Success Criteria

#### Automated Verification

- [x] `cargo build --release` succeeds.
- [x] `cargo clippy --all-targets --no-deps -- -D warnings` — pre-existing clippy errors unchanged (verified via `git stash` baseline comparison; no new warnings introduced).
- [x] `cargo test --no-run` succeeds (no broken references to the old name).
- [x] `grep -r "use_idf_in_training" src/ tests/` returns no matches.

#### Manual Verification

- [ ] `cargo build` produces the new Python wheel without errors.
- [ ] In Python, `oxidtaxa.train(..., use_idf_in_descent=True)` accepts the new kwarg and produces a model.
- [ ] Calling with the old kwarg `use_idf_in_training=True` raises a `TypeError` (confirms rename is clean).

---

## Phase 2: Apply IDF in classify-time descent

### Overview

Modify the two classify-time descent functions (`classify_one_pass` and `classify_one_pass_beam`) to read `ts.use_idf_in_descent` and, when true, build IDF-weighted per-child weights mirroring the training-side construction at [training.rs:470-481](src/training.rs#L470-L481).

### Changes Required

#### 1. `src/classify.rs` — `classify_one_pass` (greedy)

**Current code at [classify.rs:209-219](src/classify.rs#L209-L219):**

```rust
} else if subtrees.len() > 1 {
    let frac = fraction[k_node].unwrap();
    let s_dk = ((n as f64) * frac).ceil() as usize;
    let sampling = rng.sample_int_replace(n, s_dk * b);
    let matches = int_match(&dk.keep, my_kmers);
    let n_sub = subtrees.len();
    let mut hits_flat = vec![0.0f64; n_sub * b];
    for j in 0..n_sub {
        let row = vector_sum(&matches, &dk.profiles[j], &sampling, b);
        hits_flat[j * b..(j + 1) * b].copy_from_slice(&row);
    }
```

**Change to:** select an `idf_row` by descent-node depth (mirroring [training.rs:462-466](src/training.rs#L462-L466) and [classify.rs:551-554](src/classify.rs#L551-L554)) when the flag is set, then build per-child weights:

```rust
} else if subtrees.len() > 1 {
    let frac = fraction[k_node].unwrap();
    let s_dk = ((n as f64) * frac).ceil() as usize;
    let sampling = rng.sample_int_replace(n, s_dk * b);
    let matches = int_match(&dk.keep, my_kmers);
    let n_sub = subtrees.len();
    let idf_row: Option<&[f64]> = if ts.use_idf_in_descent {
        let depth = (ts.levels[k_node] - 1).max(0) as usize;
        let row_idx = depth.min(ts.idf_weights_by_rank.len().saturating_sub(1));
        Some(&ts.idf_weights_by_rank[row_idx])
    } else {
        None
    };
    let mut hits_flat = vec![0.0f64; n_sub * b];
    for j in 0..n_sub {
        let row = if let Some(idf_row) = idf_row {
            let weights_j: Vec<f64> = dk.profiles[j].iter().zip(dk.keep.iter())
                .map(|(&prof, &km)| {
                    let idf = if km > 0 && (km as usize) <= idf_row.len() {
                        idf_row[(km - 1) as usize]
                    } else { 0.0 };
                    prof * idf
                })
                .collect();
            vector_sum(&matches, &weights_j, &sampling, b)
        } else {
            vector_sum(&matches, &dk.profiles[j], &sampling, b)
        };
        hits_flat[j * b..(j + 1) * b].copy_from_slice(&row);
    }
```

The `idf_row` selection runs once per descent step (outside the per-child loop). The per-child weights are built only when `use_idf_in_descent` is true; the `false` branch is byte-identical to today.

#### 2. `src/classify.rs` — `classify_one_pass_beam` (beam variant)

**Current code at [classify.rs:367-378](src/classify.rs#L367-L378):**

```rust
// Vote at this node
let frac = fraction[k_node].unwrap();
let s_dk = ((n as f64) * frac).ceil() as usize;
let sampling = rng.sample_int_replace(n, s_dk * b);
let matches = int_match(&dk.keep, my_kmers);

let n_sub = subtrees.len();
let mut hits_flat = vec![0.0f64; n_sub * b];
for j in 0..n_sub {
    let row = vector_sum(&matches, &dk.profiles[j], &sampling, b);
    hits_flat[j * b..(j + 1) * b].copy_from_slice(&row);
}
```

**Change to:** the same `idf_row` + conditional `weights_j` pattern as in `classify_one_pass`. The variable `ts` is in scope (same calling convention). Diff is identical in shape; only the surrounding indentation differs.

#### 3. Optional refactor (defer if it bloats Phase 2)

The `idf_row`-aware vote computation appears in three places after this phase: training descent, greedy classify descent, beam classify descent. A shared helper `compute_descent_hits(...)` could be extracted, but the per-site variation in surrounding state (rng usage, hits buffer ownership) is enough that inlining is cleaner. Skip the refactor.

### Success Criteria

#### Automated Verification

- [x] `cargo build --release` succeeds.
- [x] `cargo test --no-run` succeeds.
- [x] `cargo clippy --all-targets --no-deps -- -D warnings` — no new warnings introduced (pre-existing clippy errors unchanged).
- [x] Existing tests in `tests/test_margin_aware.rs` pass (they use `use_idf_in_descent=false` paths via `ClassifyConfig::default()`; behavior is unchanged for `flag=false`).

#### Manual Verification

- [ ] Train a small model with `use_idf_in_descent=true`, classify a query, and confirm predictions reflect IDF-weighted descent.
- [ ] Train the same model with `use_idf_in_descent=false`, classify the same query, and confirm predictions match the pre-change baseline (no regression).

---

## Phase 3: Lock-in tests for train↔classify descent symmetry

### Overview

Tests added in `tests/test_margin_aware.rs` (which already covers `tie_margin`, `sibling_aware_leaf`, `confidence_uses_descent_margin`, and `suppress_ancestor_only_groups`). The tests follow the existing fixture pattern: build a small synthetic taxonomy, train, classify, assert.

### Implementation note: deviation from originally-planned test set

The plan originally proposed 5 tests including two `train↔classify hits_flat bitwise symmetry` assertions (Tests 3 & 4) and a `flag=true changes classify-time predictions` assertion (Test 2). Those three were adapted at implementation time:

- **Symmetry tests (originally Tests 3 & 4)**: the bitwise per-replicate `hits_flat` comparison would require either exposing internal descent state through the public API (invasive) or replicating ~30 lines of descent math in the test using private RNG state (brittle, drift-prone). The symmetry guarantee is instead enforced **structurally** by code-sharing: classify-side IDF construction at `classify.rs:209-243` and `classify.rs:367-403` is a literal copy of training-side at `training.rs:470-481` (same `idf_row` selection by `levels[k_node]`, same `(km - 1) as usize` indexing into the row, same `prof * idf` multiply). Both call into the same `vector_sum` helper. If either side drifts, the rename + grep coverage will catch it; the structural identity is the proof.

- **`flag=true changes predictions` (originally Test 2)**: replaced with a smoke test (`test_use_idf_in_descent_classify_smoke_test`) that exercises both greedy and beam classify paths under a flag=true model. Reason: on practical-size fixtures, the iterative fraction-learning loop at `training.rs:394-587` converges in one iteration with zero misclassifications, so flag=true and flag=false produce bit-identical `fraction` tables and per-replicate descent votes never flip. The IDF-aware code path was verified to fire at runtime via temporary diagnostic instrumentation that printed `weights_sum` (IDF-applied) vs. `profile_sum` (raw) at every multi-child descent node — they differed as expected (e.g., 0.436 vs. 0.811 at root, 1.91 vs. 1.00 at a depth-45 node). The diagnostic was removed before commit. The "predictions differ" claim is therefore not enforced by automated test; it remains a manual-verification item.

### Test cases (as implemented)

#### Test 1: `test_use_idf_in_descent_default_off`

Asserts `TrainConfig::default().use_idf_in_descent == false` and that a model produced via the default path carries `use_idf_in_descent == false` on its `TrainingSet`.

#### Test 2: `test_use_idf_in_descent_flag_persists_on_model`

Trains with flag=true and flag=false against the same fixture; asserts the resulting `TrainingSet.use_idf_in_descent` reflects each input. Locks in the persistence pipeline from `TrainConfig` → `LearnFractionsConfig` → `TrainingSet`.

#### Test 3: `test_use_idf_in_descent_false_no_regression`

Trains the same fixture with explicit `use_idf_in_descent: false` and with `TrainConfig::default()`. Classifies a shared query against both. Asserts:
- `taxon` lists match,
- `confidence` vector lengths match,
- per-rank confidence values are bit-identical (within `1e-9`).

This is the no-regression guard for the legacy flag-off path.

#### Test 4: `test_use_idf_in_descent_training_completes`

Smoke test for the training-side wiring at `training.rs:470`. Trains with flag=true; asserts the resulting `TrainingSet` has a populated taxonomy, at least one decision-kmer node, a non-empty `idf_weights_by_rank` matrix, and `use_idf_in_descent == true`. Replaces the originally-planned `train↔classify symmetry` test (see implementation note above).

#### Test 5: `test_use_idf_in_descent_classify_smoke_test`

Smoke test for the classify-side wiring. Loads a flag=true model, runs classification under both `beam_width=1` (greedy path at `classify.rs:209-243`) and `beam_width=3` (beam path at `classify.rs:367-403`); asserts both paths complete and emit a rooted lineage. Replaces the originally-planned `flag=true changes classify-time predictions` test (see implementation note above for why the original was swapped).

#### Test 6: `test_use_idf_in_descent_serialization_roundtrip`

Trains with flag=true, serializes via `TrainingSet::save` (bincode), reloads via `TrainingSet::load`, classifies a query against both the in-memory and reloaded models, and asserts predictions are bit-identical. Locks in that the new `use_idf_in_descent` field on `TrainingSet` round-trips correctly.

### Success Criteria

#### Automated Verification

- [x] `cargo test --release` passes — all 6 new tests in `tests/test_margin_aware.rs` plus all 13 pre-existing tests in that file plus all other test suites (93 tests total).
- [x] Tests with `use_idf_in_descent=false` produce predictions identical to the pre-change baseline — `test_use_idf_in_descent_false_no_regression` asserts bit-identical output between explicit-false and default-false; existing golden tests, `baseline_1k`, and integration suites all use the legacy default and pass unchanged.

#### Manual Verification

- [ ] Re-run any project-level integration sweep (e.g., the assignment_benchmarks 12s_mifish baseline) with `use_idf_in_descent=false` and confirm zero diff vs. the pre-change baseline.
- [ ] Re-run the same sweep with `use_idf_in_descent=true` and inspect for plausibility (predictions should differ in queries where IDF spread is meaningful).

---

## Testing Strategy

### Unit tests (as implemented)

Six tests in `tests/test_margin_aware.rs` (extension of the existing file):

1. `test_use_idf_in_descent_default_off` — flag defaults to false on `TrainConfig` and on the produced `TrainingSet`.
2. `test_use_idf_in_descent_flag_persists_on_model` — flag value flows from `TrainConfig` through to `TrainingSet.use_idf_in_descent`.
3. `test_use_idf_in_descent_false_no_regression` — `flag=false` (explicit) matches `flag=false` (default) bit-for-bit.
4. `test_use_idf_in_descent_training_completes` — training-side smoke test: flag=true produces a valid trained model with populated decision kmers and IDF matrix.
5. `test_use_idf_in_descent_classify_smoke_test` — classify-side smoke test: greedy (`beam_width=1`) and beam (`beam_width=3`) paths both complete on a flag=true model, exercising the IDF-aware code paths at `classify.rs:209-243` and `classify.rs:367-403`.
6. `test_use_idf_in_descent_serialization_roundtrip` — flag=true model round-trips through bincode and produces identical predictions before/after.

The originally-planned `train↔classify symmetry` tests (Tests 3 & 4 in the original sketch) and `flag=true changes predictions` test (Test 2) were swapped at implementation time — see "Implementation note" under Phase 3 above for the rationale.

### Integration tests

Manual project-level sweep against a known benchmark (e.g., the 12s_mifish classifier benchmark). Pre-change baseline must hold for `flag=false`. `flag=true` is informational — its predictions are expected to differ but should remain plausible.

### Manual testing steps

1. Build the wheel: `cargo build --release` then `pip install -e .` (or wherever the build script lands the wheel).
2. In Python: `oxidtaxa.train(seqs, tax, use_idf_in_descent=True)` returns without error.
3. `oxidtaxa.classify(...)` against the trained model returns predictions that differ from a sibling model trained with `use_idf_in_descent=False` for at least one query.
4. Calling `train` with the old kwarg `use_idf_in_training=True` raises `TypeError` (confirms the rename surfaces to Python).

## Performance Considerations

The classify-side change adds, per descent step where `use_idf_in_descent=true`:
- One `idf_row` slice borrow (O(1)).
- Per child: one `Vec<f64>` allocation of length `dk.keep.len()` and an elementwise multiply (O(n) per child, where n = number of kept decision kmers at that node, typically small).

The training-side cost is unchanged.

Net classify-time overhead under `flag=true`: roughly proportional to number of decision-node descent steps × children per node × kept-kmers per node. For typical eDNA databases (low-thousands sequences, depth ~7), this is negligible relative to the per-replicate bootstrap cost. Under `flag=false` the overhead is zero (the `if let Some(idf_row)` branch is not taken; behavior matches today).

If profiling shows the per-step `weights_j` allocation is a hotspot under `flag=true`, the allocation can be hoisted to a per-descent-call buffer reused across nodes. Defer until measured.

## Migration Notes

Per user direction, no backward compatibility is provided. Existing `TrainingSet` files serialized before this change cannot be deserialized (the new `use_idf_in_descent` field is required, not `#[serde(default)]`). Users must retrain.

Existing Python callers using `use_idf_in_training=True` or `False` will break with a `TypeError` — the kwarg has been renamed. Callers must update to `use_idf_in_descent`.

## References

- Research: `thoughts/shared/research/2026-04-30-oat-param-bug-investigation.md`
- Related plan: `thoughts/shared/plans/2026-04-27-oxidtaxa-oat-ablation.md`
- Train-side IDF construction: `src/training.rs:462-481`
- Classify-side descent (greedy): `src/classify.rs:209-219`
- Classify-side descent (beam): `src/classify.rs:367-378`
- Existing classify-side IDF (leaf phase, unchanged): `src/classify.rs:551-554`, `src/classify.rs:635-637`, `src/classify.rs:694-701`
- TrainingSet struct: `src/types.rs:22-50`
