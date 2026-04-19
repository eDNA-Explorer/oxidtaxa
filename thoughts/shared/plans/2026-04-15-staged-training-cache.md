# Staged Training Cache for Optuna Grid Search

## Overview

Refactor oxidtaxa's monolithic `learn_taxa()` into a three-phase API — `prepare_data()` + `build_tree()` + `learn_fractions()` — so that expensive intermediate results can be cached and reused across Optuna trials that share parameter subsets.

The primary win: when TPE finds a promising tree configuration and explores `training_threshold` / `use_idf_in_training` / `leave_one_out` variations around it, each variation runs the **~30-120s fraction-learning loop** instead of repeating a **multi-hour full retrain**.

## Current State Analysis

### The Problem

The Optuna sweep (`oxidtaxa_optuna_sweep.py`) runs ~2500 trials searching 8 training parameters. The existing `ModelCache` keys on **all** params, so changing even `training_threshold` (which only affects the fraction-learning loop) triggers a full retrain including the expensive tree construction.

### Where the Time Actually Goes

| Phase | Pipeline Stages | Wall Time | Params That Affect It |
|-------|----------------|-----------|----------------------|
| Prepare (k-mers, taxonomy, IDF) | 1-6, 8 | ~5-15s | `k`, `seed_pattern` |
| **Tree construction** | **7** | **minutes → hours** | `record_kmers_fraction`, `descendant_weighting`, `correlation_aware_features` |
| Fraction learning | 9 | ~30-120s | `training_threshold`, `use_idf_in_training`, `leave_one_out` |

The correlation-aware feature selection (`training.rs:776-894`) is the dominant cost. At each internal taxonomy node it runs an O(record_kmers² × n_candidates × n_children) greedy loop across potentially thousands of nodes.

Some models take 6+ hours total. Most of that time is tree construction — fraction learning is comparatively cheap.

### Cache Layer Analysis

| Cache Layer | Key | Unique Combos | Avg Reuse (2500 trials) |
|-------------|-----|---------------|------------------------|
| PreparedData | `(k, seed_pattern)` | ~20 | ~125× |
| BuiltTree | `(k, sp, rkf, dw, corr)` | ~5,760 | <1× uniform, but TPE concentrates on promising regions |
| Fractions | all 8 params | ~1.15M | no reuse (leaf level) |

The BuiltTree cache has sparse average reuse, but the critical insight is: **when TPE concentrates on a promising tree config, it explores many fraction-param variations against it.** With 200 fraction-param combos (50 × 2 × 2), even a modestly promising tree config might get 5-20 fraction-param trials. Each of those currently triggers a full rebuild. With three-phase caching, they reuse the cached tree and only rerun fraction learning.

### Key Discoveries

- `_learn_taxa_inner` is one 520-line monolithic function (`training.rs:42-563`)
- Two natural split points: line 265 (before `create_tree()`) and line 325 (before fraction loop)
- Stage 8 (IDF, lines 280-323) depends only on k-mers + taxonomy — belongs in PreparedData
- `TrainingSet` already has bincode `save()`/`load()` (`types.rs:222-236`)
- The sweep uses `n_jobs=10` threading — in-memory `Arc`-based caches work
- The PyO3 `train()` does file I/O + quality filtering before `learn_taxa()` (`lib.rs:26-73`)

## Desired End State

1. **Rust**: `PreparedData` struct (Stages 1-6 + 8) and `BuiltTree` struct (+ Stage 7 decision nodes), both with bincode `save()`/`load()`
2. **Rust**: Three public functions: `prepare_data()`, `build_tree()`, `learn_fractions()`
3. **PyO3**: Three new functions + two `#[pyclass(frozen)]` wrappers with `Arc` for thread-safe sharing, exposing `save()`/`load()` to Python
4. **Existing `train()`**: Unchanged — internally calls all three phases
5. **Optuna sweep**: Disk-backed caches with in-memory hot layer (`PreparedDataCache` + `BuiltTreeCache`), surviving crashes/restarts

### Verification

- All existing golden tests pass (`cargo test`)
- New equivalence test: three-phase produces identical `TrainingSet` as single-phase
- Optuna `--smoke-test` works with the new API

## What We're NOT Doing

- **Decoupling `record_kmers_fraction` via truncation**: The greedy selection's candidate pool (`per_child_limit = record_kmers * 2`) is budget-dependent, so truncation would require changing the pool heuristic — a behavioral change that needs separate validation
- **Changes to classification or the benchmarking framework's `CachedDatabaseBuilder`**

---

## Phase 1: Add `PreparedData` and `BuiltTree` Structs

### Overview

Define the two intermediate data structures that serve as cache boundaries.

### Changes Required

#### `src/types.rs` — Add structs after `TrainingSet` (after line 46)

```rust
/// Intermediate training data: k-mer enumeration, taxonomy tree, and IDF weights.
///
/// Output of the "prepare" phase — everything computed from
/// (sequences, taxonomy, k, seed_pattern). Cache keyed on these params
/// to avoid redundant k-mer enumeration across grid-search trials.
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

/// Prepared data plus decision tree: ready for fraction learning.
///
/// Output of the "build tree" phase. Cache keyed on PreparedData params +
/// (record_kmers_fraction, descendant_weighting, correlation_aware_features).
/// Avoids re-running expensive tree construction when only fraction-learning
/// params change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltTree {
    pub prepared: PreparedData,
    pub decision_kmers: Vec<Option<DecisionNode>>,
}
```

Add `save()`/`load()` for both (same pattern as `TrainingSet::save/load` at lines 222-236).

### Success Criteria

- [ ] `cargo build` succeeds
- [ ] `cargo test` passes (no regressions)

---

## Phase 2: Refactor `training.rs` Into Three Staged Functions

### Overview

Split `_learn_taxa_inner()` at the two natural boundaries (line 265 and line 325). The existing `learn_taxa()` becomes a thin wrapper.

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

New public function containing current lines 50-337 of `_learn_taxa_inner`:

```rust
/// Phase 1: Enumerate k-mers, build taxonomy tree, compute IDF weights.
///
/// Depends only on (sequences, taxonomy, k, seed_pattern).
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

`_prepare_data_inner` is a mechanical extraction of lines 50-337. Config references change:
- `config.k` → param `k`
- `config.n` → param `n`  
- `config.seed_pattern` → param `seed_pattern`

Returns `Ok(PreparedData { ... })`.

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

`_build_tree_inner` constructs a temporary `TrainConfig` from `BuildTreeConfig` to pass to the existing `create_tree` function (avoiding a refactor of `create_tree`'s signature):

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

    Ok(BuiltTree {
        prepared: prepared.clone(),
        decision_kmers,
    })
}
```

#### 4. `src/training.rs` — Extract `learn_fractions()`

New public function containing current lines 325-563:

```rust
/// Phase 3: Iterative fraction-learning loop + model assembly.
///
/// The cheapest phase — re-run this when only training_threshold,
/// use_idf_in_training, or leave_one_out changes.
pub fn learn_fractions(
    built_tree: &BuiltTree,
    config: &LearnFractionsConfig,
    seed: u32,
) -> Result<TrainingSet, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.processors).build()
        .map_err(|e| format!("failed to create rayon thread pool: {e}"))?;
    pool.install(|| _learn_fractions_inner(built_tree, config, seed))
}
```

`_learn_fractions_inner` is a mechanical extraction of lines 325-563. All local variable references become `p.kmers`, `p.taxonomy`, etc. (where `p = &built_tree.prepared`). `decision_kmers` comes from `&built_tree.decision_kmers`.

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
    _learn_fractions_inner(&built_tree, &LearnFractionsConfig::from(config), seed)
}
```

The public `learn_taxa()` is unchanged. Full backward compatibility.

### Success Criteria

- [ ] `cargo build` succeeds
- [ ] `cargo test` — all 5 golden training tests produce identical results
- [ ] No logic changes, only mechanical extraction

---

## Phase 3: Add PyO3 Bindings

### Overview

Expose all three phases to Python. Use `Arc` for thread-safe sharing across Optuna's `n_jobs=10` threads.

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
    fn n_sequences(&self) -> usize { self.inner.prepared.kmers.len() }
    #[getter]
    fn k(&self) -> usize { self.inner.prepared.k }
}
```

#### 2. `src/lib.rs` — Three phase functions

```rust
#[pyfunction]
#[pyo3(signature = (fasta_path, taxonomy_path, k = None, seed_pattern = None, processors = 1))]
fn prepare_data_py(
    py: Python<'_>,
    fasta_path: &str, taxonomy_path: &str,
    k: Option<usize>, seed_pattern: Option<String>, processors: usize,
) -> PyResult<PyPreparedData> {
    let (names, seqs) = crate::fasta::read_fasta(fasta_path)
        .map_err(|e| PyValueError::new_err(e))?;
    let taxonomy = crate::fasta::read_taxonomy(taxonomy_path, &names)
        .map_err(|e| PyValueError::new_err(e))?;
    let (filtered_seqs, filtered_tax) = filter_for_training(&seqs, &taxonomy);

    let inner = py.allow_threads(|| {
        crate::training::prepare_data(
            &filtered_seqs, &filtered_tax, k, 500.0, seed_pattern, processors,
        )
    }).map_err(|e| PyValueError::new_err(e))?;
    Ok(PyPreparedData { inner: Arc::new(inner) })
}

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
    let config = crate::training::BuildTreeConfig {
        record_kmers_fraction, descendant_weighting: dw,
        correlation_aware_features, max_children: 200, processors,
    };
    let data = Arc::clone(&prepared.inner);
    let built = py.allow_threads(|| {
        crate::training::build_tree(&data, &config)
    }).map_err(|e| PyValueError::new_err(e))?;
    Ok(PyBuiltTree { inner: Arc::new(built) })
}

#[pyfunction]
#[pyo3(signature = (
    built_tree, output_path, seed = 42, training_threshold = 0.8,
    use_idf_in_training = false, leave_one_out = false, processors = 1
))]
fn learn_fractions_py(
    py: Python<'_>,
    built_tree: &PyBuiltTree, output_path: &str,
    seed: u32, training_threshold: f64,
    use_idf_in_training: bool, leave_one_out: bool, processors: usize,
) -> PyResult<()> {
    let config = crate::training::LearnFractionsConfig {
        training_threshold, use_idf_in_training, leave_one_out,
        min_fraction: 0.01, max_fraction: 0.06,
        max_iterations: 10, multiplier: 100.0, processors,
    };
    let tree = Arc::clone(&built_tree.inner);
    let model = py.allow_threads(|| {
        crate::training::learn_fractions(&tree, &config, seed)
    }).map_err(|e| PyValueError::new_err(e))?;
    model.save(output_path).map_err(|e| PyValueError::new_err(e))?;
    Ok(())
}
```

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

- [ ] `cargo build --features python` succeeds
- [ ] `maturin develop --release` succeeds
- [ ] `python -c "from oxidtaxa import prepare_data, build_tree, learn_fractions"` works

---

## Phase 4: Update Optuna Sweep — Disk-Backed Caching

### Overview

Add two-level caching with **disk persistence + in-memory hot layer**. If the sweep crashes or restarts, cached PreparedData and BuiltTree files on disk prevent recomputation. The in-memory layer avoids repeated disk reads within a run.

### Cache Lookup Order

```
get_or_build(key):
  1. Check memory dict           → hit? return immediately
  2. Check disk (deterministic path) → hit? load, store in memory, return
  3. Miss: compute, write to disk atomically, store in memory, return
```

### Disk Layout

```
{output_base}/
├── cache/
│   ├── prepared/
│   │   ├── {community}__k{k}__sp_{seed_pattern}.bin
│   │   └── ...
│   └── trees/
│       ├── {community}__k{k}__sp_{sp}__rkf{rkf}__dw_{dw}__corr_{corr}.bin
│       └── ...
├── optuna_models/                  # final models (existing location)
│   └── oxidtaxa_db_{variant_key}/
│       └── oxidtaxa_model.bin
└── optuna_study.db
```

Filenames are deterministic from params — no hashing, params readable in the name.

### Concurrency Safety

With `n_jobs=10`, two threads may try to build the same tree simultaneously. We handle this with:

1. **Atomic writes**: build → write to `{path}.tmp` → `os.rename()` to final path. `rename()` is atomic on POSIX, so readers never see a partial file.
2. **Double-check after lock**: after building, re-check under lock before storing — another thread may have finished first.

### Cache Invalidation

- `--fresh` flag: deletes the `cache/` directory, forcing recomputation
- Otherwise, cached files are assumed valid (reference data is generated once during setup and doesn't change mid-run; code changes happen between runs)

### Changes Required

#### 1. FASTA filtering helper

Extract from `OxidtaxaDatabaseBuilder.build()` for reuse:

```python
def _filter_fasta_for_taxonomy(reference_fasta: Path, taxonomy_file: Path) -> Path:
    """Filter FASTA to only sequences with taxonomy entries."""
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

#### 2. `PreparedDataCache` — disk-backed

```python
class PreparedDataCache:
    """Disk-backed, thread-safe cache for PreparedData.
    
    Key: (community, k, seed_pattern) — ~20 unique entries.
    Disk path: cache/prepared/{community}__k{k}__sp_{seed_pattern}.bin
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir / "prepared"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory: dict[tuple[str, int, str | None], Any] = {}

    def _disk_path(self, community: str, k: int, sp: str | None) -> Path:
        sp_str = sp if sp is not None else "None"
        return self._dir / f"{community}__k{k}__sp_{sp_str}.bin"

    def get_or_prepare(
        self, community_name: str, reference_fasta: Path, taxonomy_file: Path,
        k: int, seed_pattern: str | None, processors: int,
    ) -> Any:
        from oxidtaxa import PreparedData as PreparedDataCls, prepare_data

        key = (community_name, k, seed_pattern)
        disk_path = self._disk_path(community_name, k, seed_pattern)

        # 1. Memory hit
        with self._lock:
            if key in self._memory:
                return self._memory[key]

        # 2. Disk hit
        if disk_path.exists():
            log.info("Loading cached PreparedData from %s", disk_path.name)
            prepared = PreparedDataCls.load(str(disk_path))
            with self._lock:
                self._memory.setdefault(key, prepared)
                return self._memory[key]

        # 3. Miss — compute, write to disk, store in memory
        filtered_fasta = _filter_fasta_for_taxonomy(reference_fasta, taxonomy_file)
        log.info("Preparing data k=%d sp=%s for %s", k, seed_pattern, community_name)
        prepared = prepare_data(
            fasta_path=str(filtered_fasta), taxonomy_path=str(taxonomy_file),
            k=k, seed_pattern=seed_pattern, processors=processors,
        )
        if filtered_fasta != reference_fasta:
            filtered_fasta.unlink(missing_ok=True)

        # Atomic write: .tmp → rename
        tmp_path = disk_path.with_suffix(".bin.tmp")
        prepared.save(str(tmp_path))
        os.rename(tmp_path, disk_path)

        with self._lock:
            self._memory.setdefault(key, prepared)
            return self._memory[key]
```

#### 3. `BuiltTreeCache` — disk-backed

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
    Disk path: cache/trees/{community}__{tree_params.filename_key}.bin

    This is the high-value cache: when TPE finds a promising tree config
    and explores fraction-learning variations, the multi-hour tree
    construction is not repeated — even across process restarts.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir / "trees"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory: dict[tuple[str, TreeParams], Any] = {}

    def _disk_path(self, community: str, tree_params: TreeParams) -> Path:
        return self._dir / f"{community}__{tree_params.filename_key}.bin"

    def get_or_build(
        self, community_name: str, prepared: Any, tree_params: TreeParams,
        processors: int,
    ) -> Any:
        from oxidtaxa import BuiltTree as BuiltTreeCls, build_tree

        key = (community_name, tree_params)
        disk_path = self._disk_path(community_name, tree_params)

        # 1. Memory hit
        with self._lock:
            if key in self._memory:
                return self._memory[key]

        # 2. Disk hit — survives crashes / restarts
        if disk_path.exists():
            log.info("Loading cached BuiltTree from %s", disk_path.name)
            built = BuiltTreeCls.load(str(disk_path))
            with self._lock:
                self._memory.setdefault(key, built)
                return self._memory[key]

        # 3. Miss — compute (expensive!), write to disk, store in memory
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
            return self._memory[key]
```

#### 4. Simplify `ModelCache` to only run fraction learning

The final model cache is already disk-backed (checks `model_path.exists()`). Now it only runs the cheap fraction-learning phase:

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
        built_tree: Any, base_db_dir: Path, processors: int,
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
                built_tree=built_tree, output_path=str(model_path), seed=42,
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
    # ...

    # Phase 1: disk + memory cached on (k, seed_pattern, community)
    prepared = prepared_cache.get_or_prepare(
        community.name, ref_fasta, ref_taxonomy,
        k, seed_pattern, threads_per_trial,
    )

    # Phase 2: disk + memory cached on (tree_params, community)
    # Survives crashes — a 4-hour tree build is never lost
    holdout_model_base = output_base / "unified_holdout" / community.name / "optuna_models"
    try:
        built_tree = tree_cache.get_or_build(
            community.name, prepared, tree_params, threads_per_trial,
        )
    except Exception as e:
        log.warning("Tree build failed trial %d, %s: %s", trial.number, community.name, e)
        return 0.0

    # Phase 3: fraction learning only — ~30-120s, disk-cached via model file
    try:
        db_dir = model_cache.get_or_train(
            training_params, community.name,
            built_tree, holdout_model_base, threads_per_trial,
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
| First trial with (k=8, sp=None) | **compute ~15s**, save to disk | — | — |
| Same k/sp, new tree params | disk hit, load to memory | **compute ~hours**, save to disk | — |
| Same tree params, new fraction params | memory hit | memory hit | **compute ~2min**, save to disk |
| Same everything | memory hit | memory hit | disk hit (model file exists) |
| **Process crash + restart** | **disk hit** | **disk hit** | **disk hit** |
| `--fresh` flag | recompute | recompute | recompute |

### Success Criteria

- [ ] `poetry run python oxidtaxa_optuna_sweep.py --marker vert12s --smoke-test` completes
- [ ] Logs show "Loading cached PreparedData" / "Loading cached BuiltTree" on disk hits
- [ ] Kill and restart sweep — previously built trees load from disk, not recomputed
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

    // Single-phase
    let config = TrainConfig::default();
    let single = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();

    // Three-phase
    let prepared = oxidtaxa::training::prepare_data(&seqs, &tax, None, 500.0, None, 1).unwrap();
    let build_config = oxidtaxa::training::BuildTreeConfig {
        record_kmers_fraction: 0.10,
        descendant_weighting: oxidtaxa::types::DescendantWeighting::Count,
        correlation_aware_features: false, max_children: 200, processors: 1,
    };
    let built = oxidtaxa::training::build_tree(&prepared, &build_config).unwrap();
    let frac_config = oxidtaxa::training::LearnFractionsConfig {
        training_threshold: 0.8, use_idf_in_training: false, leave_one_out: false,
        min_fraction: 0.01, max_fraction: 0.06, max_iterations: 10,
        multiplier: 100.0, processors: 1,
    };
    let staged = oxidtaxa::training::learn_fractions(&built, &frac_config, 42).unwrap();

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

### Success Criteria

- [ ] `cargo test test_staged_training_equivalence` passes

---

## Performance Considerations

### Where the savings come from

The key scenario: TPE finds a promising tree config and explores fraction-learning variations.

**Before** (current code): each variation of `training_threshold` / `use_idf` / `leave_one_out` triggers a full retrain:
- 1 variation = prepare (15s) + tree construction (1-4 hours) + fraction learning (2 min) = **hours**

**After** (three-phase): the tree is built once and cached:
- 1st trial with this tree config = prepare (cache hit, 0s) + tree construction (1-4 hours) + fraction learning (2 min) = **hours** (same as before)
- 2nd-Nth trials with same tree config but different fraction params = prepare (0s) + tree (cache hit, 0s) + fraction learning (2 min) = **2 minutes**

If TPE sends 10 trials to a promising tree config exploring different fraction params, that's:
- Before: 10 × 4 hours = **40 hours**
- After: 4 hours + 9 × 2 min = **4 hours 18 min**

### Disk Space

Per cached file (~20K sequence reference):
- `PreparedData`: ~35 MB (k-mers, inverted index, taxonomy, IDF)
- `BuiltTree`: ~40-45 MB (PreparedData clone + decision nodes)
- `Final model`: ~40-45 MB

Estimated disk usage:
- ~20 PreparedData files × 35 MB = ~700 MB
- ~hundreds of BuiltTree files × 45 MB = ~5-20 GB
- Final models: similar to current

Total: ~10-40 GB. Manageable on a workstation running multi-day sweeps.

**Optimization if disk is tight**: Split BuiltTree files into PreparedData (shared, ~35 MB each) + a delta file containing only `decision_kmers` (~5-10 MB each). Requires loading both and combining in memory. Saves ~35 MB per tree file.

### Memory

In-memory hot cache holds whatever has been loaded from disk during this run. Worst case (all cached entries loaded): same as disk usage.

If memory becomes a concern:
- LRU eviction on the in-memory layer (disk remains as cold storage)
- Refactor `BuiltTree` to hold `Arc<PreparedData>` instead of an owned clone (saves ~35 MB per in-memory entry)

## References

- Training pipeline: `oxidtaxa/src/training.rs:42-563`
- Type definitions: `oxidtaxa/src/types.rs:22-46` (TrainingSet), `111-145` (TrainConfig)
- PyO3 bindings: `oxidtaxa/src/lib.rs:26-73` (train)
- Correlation-aware selection: `oxidtaxa/src/training.rs:776-894`
- Optuna sweep: `assignment-tool-benchmarking/.../notebooks/oxidtaxa_optuna_sweep.py`
- OxidtaxaDatabaseBuilder: `assignment-tool-benchmarking/.../tools/oxidtaxa/database.py:36-183`
