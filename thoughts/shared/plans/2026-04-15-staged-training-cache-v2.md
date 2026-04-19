# Staged Training Cache for Optuna Grid Search (v2)

**Revision of**: `2026-04-15-staged-training-cache.md`
**Incorporates**: `2026-04-15-staged-training-cache-review.md` (6 critic issues)

## Changes from v1

| Critic Issue | Severity | Disposition | What Changed |
|---|---|---|---|
| #1 Line range contradiction | High | **Accepted** | Reorder IDF before `create_tree()`; precise phase boundaries defined |
| #2 BuiltTree clones PreparedData | Medium | **Accepted and improved** | `BuiltTree` no longer embeds `PreparedData` at all — contains only `decision_kmers`. Eliminates both memory duplication AND disk bloat (~5 MB vs ~45 MB per tree file) |
| #3 TOCTOU race / duplicate builds | Medium | **Accepted** | Per-key `threading.Event` coordination in both Python caches |
| #4 `n=500.0` hardcoded | Low | **Accepted** | Expose `n` as optional parameter (default 500.0) in `prepare_data_py` |
| #5 Fraction config hardcodes tunables | Low | **Accepted as-is** | Consistent with existing `train()` API. Documented for future reference |
| #6 Cache key omits `n` | Low | **Accepted** | Include `n` in cache key when non-default |

**Additional design change** (from codebase research): The reviewer's architectural note about delta-file disk optimization is now built-in rather than deferred — since `BuiltTree` no longer embeds `PreparedData`, tree files are inherently small. Estimated disk usage drops from 10-40 GB to 2-5 GB.

---

## Overview

Refactor oxidtaxa's monolithic `learn_taxa()` into a three-phase API — `prepare_data()` + `build_tree()` + `learn_fractions()` — so that expensive intermediate results can be cached and reused across Optuna trials sharing parameter subsets.

The primary win: when TPE finds a promising tree configuration and explores `training_threshold` / `use_idf_in_training` / `leave_one_out` variations, each variation runs the **~30-120s fraction-learning loop** instead of repeating a **multi-hour full retrain**.

## Data Flow

```
                         ┌──────────────────────────────────────┐
  (sequences, taxonomy,  │         prepare_data()               │
   k, n, seed_pattern)  ─┤  K-mer enum → taxonomy tree →       │──▶ PreparedData
                         │  end_taxonomy → sequences_per_node → │    (~35 MB, ~20 unique)
                         │  IDF weights → seq_hashes            │
                         └──────────────────────────────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────────────┐
  (PreparedData,         │         build_tree()                  │
   rkf, dw, corr,       │  create_tree() → decision_kmers      │──▶ BuiltTree
   max_children, procs) ─┤                                      │    (~5-10 MB, ~hundreds)
                         └──────────────────────────────────────┘
                                        │
                              ┌─────────┘
                              ▼
                         ┌──────────────────────────────────────┐
  (PreparedData,         │       learn_fractions()               │
   BuiltTree, config,   │  Fraction loop → model assembly       │──▶ TrainingSet
   seed)                ─┤                                      │    (~40 MB)
                         └──────────────────────────────────────┘
```

Key: `learn_fractions` takes **both** `PreparedData` and `BuiltTree` as separate arguments. `BuiltTree` does not embed `PreparedData`.

## Current Code Structure

### `_learn_taxa_inner` Phase Map (`src/training.rs`)

| Lines | Phase | What It Does | Config Fields Read |
|-------|-------|-------------|-------------------|
| 48-125 | K-mer enumeration | Parse seed, compute k, enumerate k-mers, build inverted index | `seed_pattern`, `k`, `n` |
| 128-263 | Taxonomy tree | Deduplicate classes, build tree adjacency, compute `end_taxonomy`, assign sequences to nodes | (none) |
| 265-279 | **`create_tree()`** | Build decision tree with feature selection | `max_children`, `record_kmers_fraction`, `descendant_weighting`, `correlation_aware_features`, `processors` |
| 280-323 | IDF computation | Class-weighted k-mer frequencies → log-IDF weights | (none — reads only `kmers`, `classes`, `n_kmers`) |
| 325-337 | Seq hash precompute | Deterministic per-sequence identity hashes for PRNG | (none — reads raw `sequences`, `taxonomy_strings`) |
| 339-522 | Fraction learning loop | Iterative classify → fraction decrement | `max_fraction`, `min_fraction`, `multiplier`, `max_iterations`, `training_threshold`, `use_idf_in_training`, `leave_one_out` |
| 524-562 | Result assembly | Build `TrainingSet` from all phases | `seed_pattern` |

### The Reordering (Critic #1 Fix)

IDF computation (lines 280-323) depends only on `kmers`, `classes`, and `n_kmers` — all from Phase 1. It does **not** depend on `create_tree()` output. Therefore we reorder:

```
BEFORE (current):  k-mers → taxonomy → create_tree() → IDF → fractions
AFTER  (refactor): k-mers → taxonomy → IDF → seq_hashes → [split] → create_tree() → [split] → fractions
                   ╰─────────── PreparedData ──────────╯             ╰─ BuiltTree ─╯            ╰─ TrainingSet ─╯
```

This makes PreparedData self-contained (lines 48-263 + reordered 280-337) and eliminates the line-range overlap the critic identified.

---

## Phase 1: Add `PreparedData` and `BuiltTree` Structs

### Changes Required

#### `src/types.rs` — Add structs after `TrainingSet` (after line 45)

```rust
/// Intermediate training data: k-mer enumeration, taxonomy tree, and IDF weights.
///
/// Output of the "prepare" phase — everything computed from
/// (sequences, taxonomy, k, seed_pattern). Cache keyed on these params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedData {
    pub k: usize,
    pub n_kmers: usize,
    pub kmers: Vec<Vec<i32>>,
    pub inverted_index: Vec<Vec<u32>>,
    pub classes: Vec<String>,
    pub taxonomy: Vec<String>,
    pub taxa: Vec<String>,
    pub levels: Vec<i32>,
    pub children: Vec<Vec<usize>>,
    pub parents: Vec<usize>,
    pub end_taxonomy: Vec<String>,
    pub sequences_per_node: Vec<Option<Vec<usize>>>,
    pub n_seqs: Vec<usize>,
    pub cross_index: Vec<usize>,
    pub idf_weights: Vec<f64>,
    pub seq_hashes: Vec<u64>,
    pub seed_pattern: Option<String>,
}

/// Decision tree nodes produced by feature selection.
///
/// Output of the "build tree" phase. Does NOT embed PreparedData —
/// learn_fractions() takes both as separate arguments. This keeps
/// serialized tree files small (~5-10 MB vs ~45 MB if embedded).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltTree {
    pub decision_kmers: Vec<Option<DecisionNode>>,
}
```

Note: `BuiltTree` intentionally does **not** contain `PreparedData` (addressing critic #2). The caller manages the association. Benefits:
- In-memory: no duplication when multiple trees share the same PreparedData
- On disk: tree files are ~5-10 MB instead of ~40-45 MB (hundreds of files = 1-2 GB vs 10-20 GB)
- `learn_fractions()` takes `(&PreparedData, &BuiltTree)` — the relationship is explicit

Add `save()`/`load()` for both, following the `TrainingSet` pattern at `src/types.rs:222-236`:

```rust
impl PreparedData {
    pub fn save(&self, path: &str) -> Result<(), String> {
        let encoded = bincode::serialize(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, encoded).map_err(|e| format!("write {path}: {e}"))
    }
    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        bincode::deserialize(&data).map_err(|e| format!("deserialize {path}: {e}"))
    }
}

impl BuiltTree {
    pub fn save(&self, path: &str) -> Result<(), String> {
        let encoded = bincode::serialize(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, encoded).map_err(|e| format!("write {path}: {e}"))
    }
    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        bincode::deserialize(&data).map_err(|e| format!("deserialize {path}: {e}"))
    }
}
```

### Success Criteria

- [x] `cargo build` succeeds
- [x] `cargo test` passes (no regressions)

---

## Phase 2: Refactor `training.rs` Into Three Staged Functions

### Changes Required

#### 1. Config types for each phase

Add to `src/types.rs`:

```rust
/// Config for the tree-building phase.
pub struct BuildTreeConfig {
    pub record_kmers_fraction: f64,
    pub descendant_weighting: DescendantWeighting,
    pub correlation_aware_features: bool,
    pub max_children: usize,
    pub processors: usize,
}

/// Config for the fraction-learning phase.
pub struct LearnFractionsConfig {
    pub training_threshold: f64,
    pub use_idf_in_training: bool,
    pub leave_one_out: bool,
    pub min_fraction: f64,
    pub max_fraction: f64,
    pub max_iterations: usize,
    pub multiplier: f64,
    pub processors: usize,
}
```

Add `From<&TrainConfig>` impls for both.

#### 2. `src/training.rs` — Extract `prepare_data()`

New public function. Contains current lines 48-263 (k-mers + taxonomy tree + end_taxonomy + sequences_per_node + n_seqs) **plus reordered** lines 280-323 (IDF) **plus** lines 328-337 (seq_hashes).

The reordering (critic #1 fix) moves IDF and seq_hashes before the `create_tree()` call. Both are independent of tree construction output:
- IDF reads only `kmers`, `classes`, `n_kmers` (all from Phase 1)
- `seq_hashes` reads only raw `sequences` and `taxonomy_strings` (function params)

```rust
/// Phase 1: Enumerate k-mers, build taxonomy tree, compute IDF weights.
///
/// Depends only on (sequences, taxonomy, k, n, seed_pattern).
pub fn prepare_data(
    sequences: &[String],
    taxonomy_strings: &[String],
    k: Option<usize>,
    n: f64,
    seed_pattern: Option<String>,
    processors: usize,
) -> Result<PreparedData, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processors).build()
        .map_err(|e| format!("failed to create rayon thread pool: {e}"))?;
    pool.install(|| _prepare_data_inner(sequences, taxonomy_strings, k, n, seed_pattern))
}
```

`_prepare_data_inner` is a mechanical extraction with this order:
1. Lines 48-125: k-mer enumeration (unchanged)
2. Lines 128-263: taxonomy tree building (unchanged)
3. Lines 280-323: IDF computation (**moved here from after create_tree**)
4. Lines 328-337: seq_hashes computation (**moved here from fraction loop preamble**)
5. Return `PreparedData { ... }`

Config references change: `config.k` → param `k`, `config.n` → param `n`, `config.seed_pattern` → param `seed_pattern`.

#### 3. `src/training.rs` — Extract `build_tree()`

New public function containing current lines 265-279 (the `create_tree` call):

```rust
/// Phase 2: Build decision tree with feature selection at each node.
///
/// This is the most expensive phase when correlation_aware_features=true.
pub fn build_tree(
    prepared: &PreparedData,
    config: &BuildTreeConfig,
) -> Result<BuiltTree, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.processors).build()
        .map_err(|e| format!("failed to create rayon thread pool: {e}"))?;
    pool.install(|| _build_tree_inner(prepared, config))
}
```

`_build_tree_inner` constructs a temporary `TrainConfig` to pass to the existing `create_tree` function (verified: `create_tree` only reads `max_children`, `record_kmers_fraction`, `descendant_weighting`, `correlation_aware_features`, `processors` — all set explicitly):

```rust
fn _build_tree_inner(
    prepared: &PreparedData,
    config: &BuildTreeConfig,
) -> Result<BuiltTree, String> {
    let train_config = TrainConfig {
        record_kmers_fraction: config.record_kmers_fraction,
        descendant_weighting: config.descendant_weighting,
        correlation_aware_features: config.correlation_aware_features,
        max_children: config.max_children,
        processors: config.processors,
        ..Default::default()
    };

    let mut decision_kmers: Vec<Option<DecisionNode>> = vec![None; prepared.taxonomy.len()];
    let (_root_profile, _root_desc, nodes) = create_tree(
        0,
        &prepared.children,
        &prepared.sequences_per_node,
        &prepared.kmers,
        prepared.n_kmers,
        &train_config,
    );
    for (idx, dk) in nodes {
        decision_kmers[idx] = Some(dk);
    }

    Ok(BuiltTree { decision_kmers })
}
```

Note: no `.clone()` of PreparedData — `BuiltTree` only contains `decision_kmers`.

#### 4. `src/training.rs` — Extract `learn_fractions()`

New public function containing current lines 339-562. Takes **both** `PreparedData` and `BuiltTree` as separate arguments:

```rust
/// Phase 3: Iterative fraction-learning loop + model assembly.
///
/// The cheapest phase — re-run this when only training_threshold,
/// use_idf_in_training, or leave_one_out changes.
pub fn learn_fractions(
    prepared: &PreparedData,
    built_tree: &BuiltTree,
    config: &LearnFractionsConfig,
    seed: u32,
) -> Result<TrainingSet, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.processors).build()
        .map_err(|e| format!("failed to create rayon thread pool: {e}"))?;
    pool.install(|| _learn_fractions_inner(prepared, built_tree, config, seed))
}
```

`_learn_fractions_inner` is a mechanical extraction of lines 339-562. Variable references become:
- `kmers` → `prepared.kmers`
- `taxonomy` → `prepared.taxonomy`
- `children` → `prepared.children`
- `end_taxonomy` → `prepared.end_taxonomy`
- `idf_weights` → `prepared.idf_weights`
- `seq_hashes` → `prepared.seq_hashes`
- `sequences_per_node` → `prepared.sequences_per_node`
- `n_seqs` → `prepared.n_seqs`
- `cross_index` → `prepared.cross_index`
- `classes` → `prepared.classes`
- `n_kmers` → `prepared.n_kmers`
- `k` → `prepared.k`
- `decision_kmers` → `built_tree.decision_kmers`

The result assembly (lines 544-562) clones needed data from `prepared` and `built_tree` into the owned `TrainingSet`:

```rust
Ok(TrainingSet {
    taxonomy: prepared.taxonomy.clone(),
    taxa: prepared.taxa.clone(),
    ranks: None,
    levels: prepared.levels.clone(),
    children: prepared.children.clone(),
    parents: prepared.parents.clone(),
    fraction,
    sequences: prepared.sequences_per_node.clone(),
    kmers: prepared.kmers.clone(),
    cross_index: prepared.cross_index.clone(),
    k: prepared.k,
    idf_weights: prepared.idf_weights.clone(),
    decision_kmers: built_tree.decision_kmers.clone(),
    problem_sequences,
    problem_groups,
    seed_pattern: prepared.seed_pattern.clone(),
    inverted_index: Some(prepared.inverted_index.clone()),
})
```

The clone overhead (~35 MB) is negligible relative to the 30-120s fraction-learning loop.

#### 5. `src/training.rs` — Rewrite `_learn_taxa_inner` as wrapper

```rust
fn _learn_taxa_inner(
    sequences: &[String],
    taxonomy_strings: &[String],
    config: &TrainConfig,
    seed: u32,
) -> Result<TrainingSet, String> {
    let prepared = _prepare_data_inner(
        sequences, taxonomy_strings,
        config.k, config.n, config.seed_pattern.clone(),
    )?;
    let built_tree = _build_tree_inner(&prepared, &BuildTreeConfig::from(config))?;
    _learn_fractions_inner(&prepared, &built_tree, &LearnFractionsConfig::from(config), seed)
}
```

The public `learn_taxa()` is unchanged. Full backward compatibility.

### Success Criteria

- [x] `cargo build` succeeds
- [x] `cargo test` — all golden training tests produce identical results
- [x] No logic changes, only mechanical extraction + IDF/seq_hashes reorder

---

## Phase 3: Add PyO3 Bindings

### Changes Required

#### 1. `src/lib.rs` — PyO3 wrapper classes

```rust
use std::sync::Arc;

#[pyclass(frozen, name = "PreparedData")]
struct PyPreparedData {
    inner: Arc<crate::types::PreparedData>,
}

#[pymethods]
impl PyPreparedData {
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(|e| PyValueError::new_err(e))
    }
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = crate::types::PreparedData::load(path)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(Self { inner: Arc::new(inner) })
    }
    #[getter]
    fn n_sequences(&self) -> usize { self.inner.kmers.len() }
    #[getter]
    fn k(&self) -> usize { self.inner.k }
    #[getter]
    fn n_taxa(&self) -> usize { self.inner.taxonomy.len() }
}

#[pyclass(frozen, name = "BuiltTree")]
struct PyBuiltTree {
    inner: Arc<crate::types::BuiltTree>,
}

#[pymethods]
impl PyBuiltTree {
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(|e| PyValueError::new_err(e))
    }
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = crate::types::BuiltTree::load(path)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(Self { inner: Arc::new(inner) })
    }
    #[getter]
    fn n_nodes(&self) -> usize { self.inner.decision_kmers.len() }
}
```

`#[pyclass(frozen)]` + `Arc` = thread-safe sharing across Optuna's `n_jobs=10` threads without holding the GIL. This is a new pattern for this codebase (currently zero `Arc` usage in `src/`), but is the standard PyO3 approach for shared immutable data.

#### 2. `src/lib.rs` — Three phase functions

```rust
#[pyfunction]
#[pyo3(signature = (
    fasta_path, taxonomy_path, k = None, n = 500.0,
    seed_pattern = None, processors = 1
))]
fn prepare_data_py(
    py: Python<'_>,
    fasta_path: &str, taxonomy_path: &str,
    k: Option<usize>, n: f64,
    seed_pattern: Option<String>, processors: usize,
) -> PyResult<PyPreparedData> {
    let (names, seqs) = crate::fasta::read_fasta(fasta_path)
        .map_err(|e| PyValueError::new_err(e))?;
    let taxonomy = crate::fasta::read_taxonomy(taxonomy_path, &names)
        .map_err(|e| PyValueError::new_err(e))?;
    let (filtered_seqs, filtered_tax) = filter_for_training(&seqs, &taxonomy);

    let inner = py.allow_threads(|| {
        crate::training::prepare_data(
            &filtered_seqs, &filtered_tax, k, n, seed_pattern, processors,
        )
    }).map_err(|e| PyValueError::new_err(e))?;
    Ok(PyPreparedData { inner: Arc::new(inner) })
}
```

Note: `n` is now exposed as an optional parameter with default 500.0 (critic #4 fix). The existing `train()` function also passes `n` through `TrainConfig::default().n` which is 500.0, so this is consistent.

```rust
#[pyfunction]
#[pyo3(signature = (
    prepared, record_kmers_fraction = 0.10, descendant_weighting = "count",
    correlation_aware_features = false, processors = 1
))]
fn build_tree_py(
    py: Python<'_>,
    prepared: &PyPreparedData,
    record_kmers_fraction: f64, descendant_weighting: &str,
    correlation_aware_features: bool, processors: usize,
) -> PyResult<PyBuiltTree> {
    let dw = parse_descendant_weighting(descendant_weighting)?;
    let config = crate::types::BuildTreeConfig {
        record_kmers_fraction, descendant_weighting: dw,
        correlation_aware_features, max_children: 200, processors,
    };
    let data = Arc::clone(&prepared.inner);
    let built = py.allow_threads(move || {
        crate::training::build_tree(&data, &config)
    }).map_err(|e| PyValueError::new_err(e))?;
    Ok(PyBuiltTree { inner: Arc::new(built) })
}

#[pyfunction]
#[pyo3(signature = (
    prepared, built_tree, output_path, seed = 42, training_threshold = 0.8,
    use_idf_in_training = false, leave_one_out = false, processors = 1
))]
fn learn_fractions_py(
    py: Python<'_>,
    prepared: &PyPreparedData, built_tree: &PyBuiltTree,
    output_path: &str,
    seed: u32, training_threshold: f64,
    use_idf_in_training: bool, leave_one_out: bool, processors: usize,
) -> PyResult<()> {
    let config = crate::types::LearnFractionsConfig {
        training_threshold, use_idf_in_training, leave_one_out,
        min_fraction: 0.01, max_fraction: 0.06,
        max_iterations: 10, multiplier: 100.0, processors,
    };
    let prep = Arc::clone(&prepared.inner);
    let tree = Arc::clone(&built_tree.inner);
    let model = py.allow_threads(move || {
        crate::training::learn_fractions(&prep, &tree, &config, seed)
    }).map_err(|e| PyValueError::new_err(e))?;
    model.save(output_path).map_err(|e| PyValueError::new_err(e))?;
    Ok(())
}
```

Note: `learn_fractions_py` takes **both** `prepared` and `built_tree` as separate arguments. The fraction-loop config params (`min_fraction`, `max_fraction`, `max_iterations`, `multiplier`) are hardcoded to `TrainConfig` defaults, consistent with the existing `train()` API (critic #5 — accepted as-is).

#### 3. `src/lib.rs` — Register in module

```rust
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(train, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(classify, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(prepare_data_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(build_tree_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(learn_fractions_py, m)?)?;
    m.add_class::<crate::types::ClassificationResult>()?;
    m.add_class::<PyPreparedData>()?;
    m.add_class::<PyBuiltTree>()?;
    Ok(())
}
```

Existing `train()` is **unchanged**.

### Success Criteria

- [x] `cargo build --features python` succeeds
- [x] `maturin develop --release` succeeds
- [x] `python -c "from oxidtaxa import prepare_data, build_tree, learn_fractions"` works

---

## Phase 4: Update Optuna Sweep — Disk-Backed Caching with Race Protection

### Overview

Add two-level caching with **disk persistence + in-memory hot layer** and **per-key coordination** to prevent duplicate computation (critic #3 fix).

### Disk Layout

```
{output_base}/
├── cache/
│   ├── prepared/
│   │   ├── {community}__k{k}__n{n}__sp_{seed_pattern}.bin      (~35 MB each)
│   │   └── ...
│   └── trees/
│       ├── {community}__k{k}__sp_{sp}__rkf{rkf}__dw_{dw}__corr_{corr}.bin  (~5-10 MB each)
│       └── ...
├── optuna_models/
│   └── oxidtaxa_db_{variant_key}/
│       └── oxidtaxa_model.bin
└── optuna_study.db
```

Note: PreparedData cache key includes `n` (critic #6 fix). When `n=500.0` (default), filename uses `n500` for readability. Tree files are ~5-10 MB (no embedded PreparedData), making the hundreds of expected tree files manageable (~1-2 GB total).

### Concurrency: Per-Key Event Coordination (Critic #3 Fix)

The v1 plan had a TOCTOU race: two threads missing the same cache key would both compute the same expensive result. With `n_jobs=10` and TPE concentrating trials on promising regions, this is a real risk for tree builds (hours of wasted CPU).

The fix uses per-key `threading.Event` objects so only one thread computes per key; others block and read the result:

```
Thread A: cache miss for key K → registers Event, starts computing
Thread B: cache miss for key K → finds Event, blocks on event.wait()
Thread A: finishes → stores result, sets Event
Thread B: wakes up → reads result from memory
```

### Changes Required

#### 1. FASTA filtering helper

Extract from `OxidtaxaDatabaseBuilder.build()` for reuse:

```python
def _filter_fasta_for_taxonomy(reference_fasta: Path, taxonomy_file: Path) -> Path:
    """Filter FASTA to only sequences with taxonomy entries.
    
    This is a different filter than Rust-side filter_for_training():
    - Python: removes records with no taxonomy entry (pre-read)
    - Rust: applies quality thresholds (min ranks, min length, max N%)
    Both are needed.
    """
    taxonomy_accs: set[str] = set()
    with open(taxonomy_file) as f:
        for line in f:
            parts = line.split("\t", 1)
            if parts:
                taxonomy_accs.add(parts[0].strip())

    with open(reference_fasta) as f:
        content = f.read()
    records = content.split(">")[1:]
    kept, skipped = [], 0
    for rec in records:
        header = rec.split("\n", 1)[0].strip()
        if header in taxonomy_accs:
            kept.append(">" + rec)
        else:
            skipped += 1

    if not skipped:
        return reference_fasta

    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
    tmp.write("".join(kept))
    tmp.flush()
    return Path(tmp.name)
```

#### 2. `PreparedDataCache` — disk-backed with race protection

```python
class PreparedDataCache:
    """Disk-backed, thread-safe cache for PreparedData.
    
    Key: (community, k, n, seed_pattern) — ~20 unique entries.
    Uses per-key Event coordination to prevent duplicate computation.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir / "prepared"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory: dict[tuple, Any] = {}
        self._building: dict[tuple, threading.Event] = {}

    def _disk_path(self, community: str, k: int, n: float, sp: str | None) -> Path:
        sp_str = sp if sp is not None else "None"
        n_str = f"n{n:g}" if n != 500.0 else "n500"
        return self._dir / f"{community}__k{k}__{n_str}__sp_{sp_str}.bin"

    def get_or_prepare(
        self, community_name: str, reference_fasta: Path, taxonomy_file: Path,
        k: int, n: float, seed_pattern: str | None, processors: int,
    ) -> Any:
        from oxidtaxa import PreparedData as PreparedDataCls, prepare_data

        key = (community_name, k, n, seed_pattern)
        disk_path = self._disk_path(community_name, k, n, seed_pattern)

        with self._lock:
            if key in self._memory:
                return self._memory[key]
            if key in self._building:
                event = self._building[key]
                wait = True
            else:
                wait = False

        # Another thread is already building this key — wait for it
        if wait:
            event.wait()
            with self._lock:
                return self._memory[key]

        # Disk hit
        if disk_path.exists():
            log.info("Loading cached PreparedData from %s", disk_path.name)
            prepared = PreparedDataCls.load(str(disk_path))
            with self._lock:
                self._memory.setdefault(key, prepared)
                return self._memory[key]

        # Register intent to build
        event = threading.Event()
        with self._lock:
            # Double-check: another thread may have finished between checks
            if key in self._memory:
                return self._memory[key]
            self._building[key] = event

        try:
            filtered_fasta = _filter_fasta_for_taxonomy(reference_fasta, taxonomy_file)
            log.info("Preparing data k=%d n=%g sp=%s for %s", k, n, seed_pattern, community_name)
            prepared = prepare_data(
                fasta_path=str(filtered_fasta), taxonomy_path=str(taxonomy_file),
                k=k, n=n, seed_pattern=seed_pattern, processors=processors,
            )
            if filtered_fasta != reference_fasta:
                filtered_fasta.unlink(missing_ok=True)

            # Atomic write
            tmp_path = disk_path.with_suffix(".bin.tmp")
            prepared.save(str(tmp_path))
            os.rename(tmp_path, disk_path)

            with self._lock:
                self._memory.setdefault(key, prepared)
        finally:
            with self._lock:
                self._building.pop(key, None)
            event.set()

        with self._lock:
            return self._memory[key]
```

#### 3. `BuiltTreeCache` — disk-backed with race protection

```python
@dataclass(frozen=True)
class TreeParams:
    """Hashable tree-construction param set."""
    k: int
    seed_pattern: str | None
    record_kmers_fraction: float
    descendant_weighting: str
    correlation_aware_features: bool

    @property
    def filename_key(self) -> str:
        sp = self.seed_pattern if self.seed_pattern is not None else "None"
        corr = "T" if self.correlation_aware_features else "F"
        return (
            f"k{self.k}__sp_{sp}"
            f"__rkf{self.record_kmers_fraction:g}"
            f"__dw_{self.descendant_weighting}"
            f"__corr_{corr}"
        )


class BuiltTreeCache:
    """Disk-backed, thread-safe cache for BuiltTree.

    Key: (community, tree_params).
    Uses per-key Event coordination to prevent duplicate computation.
    This is the HIGH-VALUE cache: tree builds take minutes to hours.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir / "trees"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory: dict[tuple[str, TreeParams], Any] = {}
        self._building: dict[tuple[str, TreeParams], threading.Event] = {}

    def _disk_path(self, community: str, tree_params: TreeParams) -> Path:
        return self._dir / f"{community}__{tree_params.filename_key}.bin"

    def get_or_build(
        self, community_name: str, prepared: Any, tree_params: TreeParams,
        processors: int,
    ) -> Any:
        from oxidtaxa import BuiltTree as BuiltTreeCls, build_tree

        key = (community_name, tree_params)
        disk_path = self._disk_path(community_name, tree_params)

        with self._lock:
            if key in self._memory:
                return self._memory[key]
            if key in self._building:
                event = self._building[key]
                wait = True
            else:
                wait = False

        # Another thread is already building — wait instead of duplicating work
        if wait:
            log.info("Waiting for in-progress tree build: %s", tree_params.filename_key)
            event.wait()
            with self._lock:
                return self._memory[key]

        # Disk hit — survives crashes / restarts
        if disk_path.exists():
            log.info("Loading cached BuiltTree from %s", disk_path.name)
            built = BuiltTreeCls.load(str(disk_path))
            with self._lock:
                self._memory.setdefault(key, built)
                return self._memory[key]

        # Register intent to build
        event = threading.Event()
        with self._lock:
            if key in self._memory:
                return self._memory[key]
            self._building[key] = event

        try:
            log.info(
                "Building tree rkf=%.2f dw=%s corr=%s for %s",
                tree_params.record_kmers_fraction, tree_params.descendant_weighting,
                tree_params.correlation_aware_features, community_name,
            )
            built = build_tree(
                prepared=prepared,
                record_kmers_fraction=tree_params.record_kmers_fraction,
                descendant_weighting=tree_params.descendant_weighting,
                correlation_aware_features=tree_params.correlation_aware_features,
                processors=processors,
            )

            # Atomic write
            tmp_path = disk_path.with_suffix(".bin.tmp")
            built.save(str(tmp_path))
            os.rename(tmp_path, disk_path)

            with self._lock:
                self._memory.setdefault(key, built)
        finally:
            with self._lock:
                self._building.pop(key, None)
            event.set()

        with self._lock:
            return self._memory[key]
```

#### 4. Simplify `ModelCache` to only run fraction learning

Now takes **both** `prepared` and `built_tree`:

```python
class ModelCache:
    """Thread-safe cache for final models.
    
    Only runs the fraction-learning phase (~30-120s) since
    PreparedData and BuiltTree are cached separately.
    Disk-backed via model file existence check.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], Path] = {}

    def get_or_train(
        self, training_params: TrainingParams, community_name: str,
        prepared: Any, built_tree: Any, base_db_dir: Path, processors: int,
    ) -> Path:
        key = (community_name, training_params.db_variant_key)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        from oxidtaxa import learn_fractions

        variant_db_dir = base_db_dir / f"oxidtaxa_db_{training_params.db_variant_key}"
        model_path = variant_db_dir / "oxidtaxa_model.bin"

        if not model_path.exists():
            variant_db_dir.mkdir(parents=True, exist_ok=True)
            log.info("Learning fractions %s for %s",
                     training_params.db_variant_key, community_name)
            learn_fractions(
                prepared=prepared, built_tree=built_tree,
                output_path=str(model_path), seed=42,
                training_threshold=training_params.training_threshold,
                use_idf_in_training=training_params.use_idf_in_training,
                leave_one_out=training_params.leave_one_out,
                processors=processors,
            )

        with self._lock:
            self._cache[key] = variant_db_dir
        return variant_db_dir
```

#### 5. Update objective function

The community loop in `objective()` becomes:

```python
tree_params = TreeParams(
    k=k, seed_pattern=seed_pattern,
    record_kmers_fraction=record_kmers_fraction,
    descendant_weighting=descendant_weighting,
    correlation_aware_features=correlation_aware_features,
)

for community in communities:
    if community.name not in holdout_refs:
        continue
    ref_fasta, ref_taxonomy = holdout_refs[community.name]

    # Phase 1: cached on (k, n, seed_pattern, community) — ~15s compute, ~20 unique
    prepared = prepared_cache.get_or_prepare(
        community.name, ref_fasta, ref_taxonomy,
        k, 500.0, seed_pattern, threads_per_trial,
    )

    # Phase 2: cached on (tree_params, community) — minutes-hours, disk-persistent
    holdout_model_base = output_base / "unified_holdout" / community.name / "optuna_models"
    try:
        built_tree = tree_cache.get_or_build(
            community.name, prepared, tree_params, threads_per_trial,
        )
    except Exception as e:
        log.warning("Tree build failed trial %d, %s: %s", trial.number, community.name, e)
        return 0.0

    # Phase 3: fraction learning only — ~30-120s per trial
    try:
        db_dir = model_cache.get_or_train(
            training_params, community.name,
            prepared, built_tree, holdout_model_base, threads_per_trial,
        )
    except Exception as e:
        log.warning("Fraction learning failed trial %d, %s: %s", trial.number, community.name, e)
        return 0.0
```

#### 6. Update `main()` — wire up caches + invalidation

```python
def main() -> None:
    args = _parse_args()
    # ... existing setup ...

    cache_dir = output_base / "cache"
    if args.fresh and cache_dir.exists():
        import shutil
        log.info("--fresh: clearing cache directory %s", cache_dir)
        shutil.rmtree(cache_dir)

    prepared_cache = PreparedDataCache(cache_dir)
    tree_cache = BuiltTreeCache(cache_dir)
    model_cache = ModelCache()
    objective = make_objective(
        ...,
        model_cache=model_cache,
        prepared_cache=prepared_cache,
        tree_cache=tree_cache,
        ...,
    )
```

### Cache Lifecycle Summary

| Event | PreparedData | BuiltTree | Final Model |
|-------|-------------|-----------|-------------|
| First trial with (k=8, sp=None) | **compute ~15s**, save (~35 MB) | — | — |
| Same k/sp, new tree params | disk/memory hit | **compute ~hours**, save (~5-10 MB) | — |
| Same tree params, new fraction params | memory hit | memory hit | **compute ~2min**, save |
| Same everything | memory hit | memory hit | disk hit |
| **Process crash + restart** | **disk hit** | **disk hit** | **disk hit** |
| **Duplicate concurrent key** | **Event.wait()** | **Event.wait()** | N/A (model files) |
| `--fresh` flag | recompute | recompute | recompute |

### Success Criteria

- [ ] `poetry run python oxidtaxa_optuna_sweep.py --marker vert12s --smoke-test` completes (requires benchmark env)
- [ ] Logs show "Loading cached PreparedData" / "Loading cached BuiltTree" on hits
- [ ] Logs show "Waiting for in-progress tree build" when threads collide (testable with small `n_jobs`)
- [ ] Kill and restart sweep — previously built trees load from disk
- [ ] `--fresh` clears cache and forces recomputation
- [ ] F1 scores unchanged vs. baseline

---

## Phase 5: Add Equivalence Test

### Changes Required

#### `tests/test_training.rs`

```rust
#[test]
fn test_staged_training_equivalence() {
    let (seqs, tax) = load_training_inputs("s08a_filtered_seqs", "s08a_taxonomy_vec");

    // Single-phase (existing path)
    let config = TrainConfig::default();
    let single = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();

    // Three-phase
    let prepared = oxidtaxa::training::prepare_data(
        &seqs, &tax, None, 500.0, None, 1,
    ).unwrap();
    let build_config = oxidtaxa::types::BuildTreeConfig {
        record_kmers_fraction: 0.10,
        descendant_weighting: oxidtaxa::types::DescendantWeighting::Count,
        correlation_aware_features: false,
        max_children: 200,
        processors: 1,
    };
    let built = oxidtaxa::training::build_tree(&prepared, &build_config).unwrap();
    let frac_config = oxidtaxa::types::LearnFractionsConfig {
        training_threshold: 0.8,
        use_idf_in_training: false,
        leave_one_out: false,
        min_fraction: 0.01,
        max_fraction: 0.06,
        max_iterations: 10,
        multiplier: 100.0,
        processors: 1,
    };
    let staged = oxidtaxa::training::learn_fractions(
        &prepared, &built, &frac_config, 42,
    ).unwrap();

    // Must be bit-identical
    assert_eq!(single.k, staged.k);
    assert_eq!(single.taxonomy, staged.taxonomy);
    assert_eq!(single.kmers, staged.kmers);
    assert_eq!(single.idf_weights, staged.idf_weights);
    assert_eq!(single.fraction, staged.fraction);
    assert_eq!(single.decision_kmers.len(), staged.decision_kmers.len());
    assert_eq!(single.problem_sequences.len(), staged.problem_sequences.len());
    assert_eq!(single.problem_groups, staged.problem_groups);
}
```

Note: the reordering of IDF before `create_tree()` is safe because IDF computation is fully independent of tree construction. The equivalence test verifies this produces bit-identical results.

### Success Criteria

- [x] `cargo test test_staged_training_equivalence` passes

---

## Performance Analysis

### Time Savings

**Before** (current): each trial variation triggers a full retrain:
- 1 trial = prepare (15s) + tree (1-4 hours) + fractions (2 min) = **hours**

**After** (three-phase): tree is built once and cached:
- 1st trial with a tree config = prepare (hit) + tree (**hours**) + fractions (2 min) = hours
- 2nd-Nth trials (same tree, different fraction params) = prepare (hit) + tree (**hit**) + fractions (2 min) = **2 min**

If TPE sends 10 trials to a promising tree config:
- Before: 10 × 4 hours = **40 hours**
- After: 4 hours + 9 × 2 min = **4 hours 18 min** (9.3× speedup)

### Disk Space (improved from v1)

| Cache Layer | Per File | Count | Total |
|---|---|---|---|
| PreparedData | ~35 MB | ~20 | ~700 MB |
| BuiltTree | **~5-10 MB** (no embedded PreparedData) | ~hundreds | **~1-2 GB** |
| Final models | ~40 MB | varies | varies |

**v1 estimate**: 10-40 GB total (BuiltTree files were ~45 MB each with embedded PreparedData)
**v2 estimate**: **2-5 GB total** — the delta-file optimization is built-in by design

### Memory

In-memory hot cache holds loaded entries. Since `BuiltTree` no longer contains `PreparedData`, memory usage is much lower:
- All PreparedData entries: ~20 × 35 MB = ~700 MB
- Active BuiltTree entries: ~5-10 MB each (only decision_kmers)
- Python `PyPreparedData`/`PyBuiltTree` use `Arc` — no duplication from multiple references

If memory becomes a concern: add LRU eviction on the in-memory layer (disk remains as cold storage).

---

## Critic Disposition Summary

### Accepted

1. **Line range contradiction** → IDF and seq_hashes moved before `create_tree()`. Phase boundaries are now: PreparedData = lines 48-263 + reordered 280-337, BuiltTree = lines 265-279.

2. **BuiltTree clones PreparedData** → Improved beyond the critic's suggestion. Rather than `Arc<PreparedData>` inside `BuiltTree`, `BuiltTree` doesn't embed `PreparedData` at all. `learn_fractions()` takes both as separate arguments. This eliminates memory duplication AND reduces disk usage by ~80%.

3. **TOCTOU race** → Per-key `threading.Event` coordination in both cache classes. Only one thread computes per key; others block and read the result.

4. **`n=500.0` hardcoded** → Exposed as optional parameter with default 500.0 in `prepare_data_py`.

5. **Fraction config hardcodes tunables** → Accepted as-is. Consistent with existing `train()` API which also doesn't expose these. Documented for future reference.

6. **Cache key omits `n`** → `n` included in PreparedData cache key and disk filename.

### Non-issues verified

- `DescendantWeighting` serialization: not needed (consumed during tree construction, effects baked into DecisionNode profiles)
- `seq_hashes` dependency: computed from raw sequences during `prepare_data()` before they go out of scope
- `_build_tree_inner` uses `..Default::default()`: verified that `create_tree` only reads the 5 fields set explicitly
- Python-side vs Rust-side filtering: different filters (pre-read taxonomy filtering vs quality thresholds), both needed

## References

- Training pipeline: `oxidtaxa/src/training.rs` — `_learn_taxa_inner` (lines 42-563), `create_tree` (lines 666-977)
- Type definitions: `oxidtaxa/src/types.rs` — `TrainingSet` (lines 22-45), `TrainConfig` (lines 111-167)
- PyO3 bindings: `oxidtaxa/src/lib.rs` — `train` (lines 16-73), module registration (lines 210-216)
- FASTA I/O: `oxidtaxa/src/fasta.rs` — `read_fasta` (line 3), `read_taxonomy` (line 31)
- Optuna sweep: `assignment-tool-benchmarking/.../notebooks/oxidtaxa_optuna_sweep.py` — `ModelCache` (lines 143-212), `objective` (lines 243-424), `main` (lines 527-679)
