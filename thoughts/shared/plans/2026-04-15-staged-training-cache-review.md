# Review: Staged Training Cache Plan

**Reviewed**: 2026-04-15
**Plan**: `2026-04-15-staged-training-cache.md`
**Verdict**: Architecturally sound. Six issues found — one high-severity contradiction that would block implementation, two medium-severity issues that would cause production pain, and three low-severity gaps.

---

## High Severity

### 1. Line range contradiction — `prepare_data` includes `create_tree`

The plan states two split points: "line 265 (before `create_tree()`) and line 325 (before fraction loop)." It then defines `prepare_data()` as extracting "current lines 50-337." These are incompatible.

The actual code flow in `_learn_taxa_inner`:

```
Lines  50-264  k-mer enumeration, taxonomy tree, sequences_per_node
Lines 265-279  create_tree() call                                    <-- split 1
Lines 280-323  IDF computation
Lines 325-563  fraction learning loop                                <-- split 2
```

Lines 50-337 spans across `create_tree()` (265-279), which is supposed to be the separate `build_tree` phase. The `PreparedData` struct includes `idf_weights`, placing IDF in the prepare phase — but the extraction range contradicts the stated split point.

**Resolution**: `prepare_data` should be lines 50-264 + 280-323 (IDF), with `create_tree` (265-279) skipped. IDF depends only on k-mers and taxonomy, not on the tree, so reordering it before `create_tree` is safe. The implementation should:

1. Move IDF computation (lines 280-323) above the `create_tree` call
2. Extract lines 50-264 + (moved) IDF block as `prepare_data`
3. Extract lines 265-279 as `build_tree`
4. Extract lines 325-563 as `learn_fractions`

This preserves the plan's intent while fixing the range overlap.

---

## Medium Severity

### 2. `BuiltTree` clones `PreparedData` — ~35 MB per tree instance

The plan's `_build_tree_inner` does:

```rust
Ok(BuiltTree {
    prepared: prepared.clone(),  // deep copy of ~35 MB
    decision_kmers,
})
```

With `n_jobs=10` and the in-memory cache holding multiple `BuiltTree` entries for the same `PreparedData`, each carries its own 35 MB clone. Ten tree configs against one PreparedData = 350 MB of duplicate data. The plan acknowledges this in the Performance section as a future optimization, but implementing `Arc<PreparedData>` from the start is trivial and avoids the entire class of issues.

**Resolution**: Define `BuiltTree` as:

```rust
pub struct BuiltTree {
    pub prepared: Arc<PreparedData>,
    pub decision_kmers: Vec<Option<DecisionNode>>,
}
```

This requires `PreparedData` to not derive `Clone` on `BuiltTree` (or use `Arc::clone`). The Serde impls for `BuiltTree` would serialize the inner `PreparedData` normally — `Arc` is transparent to bincode. The PyO3 wrapper already uses `Arc`, so this aligns.

### 3. TOCTOU race in cache — duplicate multi-hour tree builds

Both `PreparedDataCache` and `BuiltTreeCache` release the lock between the memory miss and the disk check:

```python
# Under lock
with self._lock:
    if key in self._memory: return  # miss

# NO lock held
if disk_path.exists():  # miss
    ...

# NO lock held — start expensive compute
built = build_tree(...)  # hours
```

Two threads hitting the same cache miss simultaneously will both compute the same tree. The "double-check after lock" (`setdefault`) prevents duplicate memory storage but not duplicate computation. For `BuiltTree`, this wastes hours of CPU.

**Resolution**: Use a per-key coordination mechanism. Simplest approach:

```python
class BuiltTreeCache:
    def __init__(self, cache_dir):
        self._lock = threading.Lock()
        self._memory = {}
        self._building: dict[tuple, threading.Event] = {}

    def get_or_build(self, ...):
        key = (community_name, tree_params)

        with self._lock:
            if key in self._memory:
                return self._memory[key]
            if key in self._building:
                event = self._building[key]
            else:
                event = None

        if event is not None:
            event.wait()
            with self._lock:
                return self._memory[key]

        # Check disk (safe — only one thread gets past the event gate)
        if disk_path.exists():
            ...

        # Register intent to build
        event = threading.Event()
        with self._lock:
            # Double-check: another thread may have finished between our check and now
            if key in self._memory:
                return self._memory[key]
            self._building[key] = event

        try:
            built = build_tree(...)
            # atomic write to disk
            with self._lock:
                self._memory[key] = built
        finally:
            with self._lock:
                self._building.pop(key, None)
            event.set()

        return built
```

This ensures only one thread computes per key. Others block on the event and read the result from memory once it's ready.

---

## Low Severity

### 4. `n=500.0` hardcoded in `prepare_data_py`

The plan's `prepare_data_py` passes `500.0` as the `n` parameter. This matches `TrainConfig::default().n`, but creates a hidden coupling. If `n` ever becomes user-configurable, this binding won't track.

**Resolution**: Either expose `n` as an optional parameter in the Python signature (with default 500.0), or add a comment documenting the coupling. Low risk since `n` has been 500.0 since the R original.

### 5. `LearnFractionsConfig` hardcodes tunable params in `learn_fractions_py`

```rust
min_fraction: 0.01, max_fraction: 0.06,
max_iterations: 10, multiplier: 100.0,
```

These are hardcoded to `TrainConfig` defaults. The existing `train()` Python function also doesn't expose these, so this is consistent. But if Optuna ever sweeps `max_iterations` or `multiplier`, the three-phase API would need updating.

**Resolution**: Accept as-is for parity with the current API. Note the gap for future reference.

### 6. Cache key for `PreparedData` omits `n`

The cache keys on `(community, k, seed_pattern)`. The `prepare_data()` function takes `n` as a parameter, which affects auto-k computation when `k=None`. Two calls with different `n` but the same explicit `k` are fine. Two calls with `k=None` and different `n` would produce different results under the same cache key.

In practice, `n=500.0` is always hardcoded (see issue #4), so this can't trigger today.

**Resolution**: If `n` is ever exposed as a parameter, add it to the cache key. For now, document the assumption.

---

## Non-issues (verified correct)

These looked suspicious on first read but are correct:

- **`DescendantWeighting` lacks Serialize/Deserialize**: Not needed — `BuiltTree` only contains `decision_kmers` and `PreparedData`, neither of which includes `DescendantWeighting`. The enum is consumed during tree construction and its effects are baked into the `DecisionNode` profiles.

- **`seq_hashes` dependency on raw sequences**: The plan puts `seq_hashes` in `PreparedData`, computed during `prepare_data()` which has access to the raw `sequences` and `taxonomy_strings`. The fraction loop only reads `seq_hashes[i]`, not the raw sequences. Correct.

- **`_build_tree_inner` constructs a `TrainConfig` with `..Default::default()`**: Verified that `create_tree` only reads `max_children`, `record_kmers_fraction`, `descendant_weighting`, `correlation_aware_features`, and `processors` from the config. All are set explicitly in `BuildTreeConfig::from()`. The defaulted fields (`k`, `n`, `training_threshold`, etc.) are never touched by `create_tree`. Correct.

- **Equivalence test references functions that don't exist yet**: The test is Phase 5, which depends on Phases 1-2. The plan sequences phases correctly.

- **Python-side `_filter_fasta_for_taxonomy` duplicates Rust-side filtering**: These are different filters. The Python helper removes FASTA records with no taxonomy entry (pre-read). The Rust `filter_for_training` applies quality thresholds (min ranks, min length, max N%). Both are needed. The plan could be clearer about this distinction but the logic is correct.

---

## Architectural Notes

The plan's core thesis is valid: TPE concentrates trials on promising tree configs, and decoupling the ~2-minute fraction-learning loop from the multi-hour tree construction yields order-of-magnitude speedups for clustered trials. The three-phase split maps cleanly to the natural boundaries in `_learn_taxa_inner`.

The disk-backed cache design with atomic writes and crash recovery is well-suited to multi-day sweep runs. The estimated disk usage (10-40 GB) is reasonable for workstation use.

One design question worth revisiting: the `BuiltTree` struct embeds (or should `Arc`-reference) the full `PreparedData`. This means serialized `BuiltTree` files on disk each contain a full copy of `PreparedData` (~35 MB). The plan mentions a delta-file optimization in the Disk Space section. Given that hundreds of tree files are expected, this optimization would save 5-20 GB. It may be worth implementing in Phase 1 rather than deferring.
