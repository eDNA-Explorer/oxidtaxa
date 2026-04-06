# Agent Rules for eDNA Explorer Data Pipelines

**Version**: 1.0.0
**Last Updated**: 2025-10-10
**Purpose**: Enforceable coding rules for AI agents working on the eDNA Explorer data pipelines monorepo.

> **Package-specific guidance**: For LLM, QC tools, SQLAlchemy, and Rust patterns, see `.claude/guidelines/packages/` and `.claude/guidelines/rust/`.

---

## 1. Types & Safety

### Pyright Configuration

**MUST** configure in `pyproject.toml` (lines 126-130):
```toml
[tool.pyright]
executionEnvironments = [{ root = "." }]
stubPath = "stubs"
```

**MUST** run: `poetry run pyright`

### Type Annotation Requirements

**MUST** annotate ALL function signatures:
```python
# ✅ GOOD
def process_data(items: list[str], user_id: int) -> dict[str, int]:
    return {}

# ❌ BAD
def process_data(items, user_id):  # No annotations
    return {}
```

**MUST** use modern Python 3.10+ syntax:
- Use `X | None` not `Optional[X]`
- Use `list[T]` not `List[T]`
- Use `dict[K, V]` not `Dict[K, V]`

**MUST NOT** use `Any` without justification.

**MUST** use `Literal` for constrained strings:
```python
TemplateType = Literal["METABARCODING", "QPCR", "DDPCR", "INVALID"]
```

**MUST** use `Protocol` for duck-typed interfaces:
```python
class StructuredLogger(Protocol):
    def info(self, message: str) -> None: ...
    def warning(self, message: str) -> None: ...
```

---

## 2. DataFrames

### Polars Preference

**SHOULD** prefer Polars for new code:
```python
# ✅ Polars with lazy evaluation
df = pl.scan_csv("data.csv").filter(pl.col("value") > 0).collect()
```

### Pandera Validation

**MUST** validate DataFrames at pipeline boundaries using Polars:
```python
import polars as pl
import pandera.polars as pa

schema = pa.DataFrameSchema({
    "transaction_id": pa.Column(int, pa.Check.gt(0), unique=True),
    "amount": pa.Column(float, [pa.Check.ge(0), pa.Check.le(1_000_000)]),
})

def clean_sales_data(df: pl.DataFrame) -> pl.DataFrame:
    validated = schema.validate(df, lazy=True)
    return validated.drop_nulls()
```

**MUST** use `lazy=True` for comprehensive error reports.
**MUST NOT** validate row-by-row with Pydantic (extreme performance penalty).

---

## 3. Runtime Validation

### Pydantic at Boundaries

**MUST** use Pydantic for external data (API, config files, env vars, JSON).

**MUST** convert to lightweight types after validation:
```python
@dataclass
class ProcessingConfigInternal:
    project_id: str
    threshold: float

def process_api_request(raw_data: dict[str, Any]) -> ProcessingConfigInternal:
    validated = ProcessingConfig.model_validate(raw_data)
    return ProcessingConfigInternal(
        project_id=validated.project_id,
        threshold=validated.threshold,
    )
```

**MUST NOT** pass Pydantic deep into business logic (avoid "SerDes debt").

**MUST** use `@field_validator` for custom validation.

---

## 4. Errors

### Fail Fast

**MUST** validate early with descriptive context:
```python
if config.threshold <= 0 or config.threshold >= 1:
    raise ValueError(f"threshold must be between 0 and 1, got {config.threshold}")

if not blob.exists():
    raise FileNotFoundError(f"File not found: gs://{bucket}/{path}")
```

### Standard Exceptions

**MUST** use built-in exceptions:
- `ValueError`: Invalid inputs/configuration
- `FileNotFoundError`: Missing files
- `RuntimeError`: Operational failures
- `Exception`: External service failures (with context)

**MUST NOT** create custom exception hierarchies unless necessary.

### Cleanup Guarantee

**MUST** use `finally` for cleanup:
```python
temp_dir = None
try:
    temp_dir = create_temp_dir()
    return result
finally:
    if temp_dir:
        try:
            cleanup_temp_dir(temp_dir)
        except Exception as e:
            logger.warning(f"Failed cleanup {temp_dir}: {e}")
```

---

## 5. Logging

### Logger Pattern

**MUST** use a custom `Logger` Protocol for type-safe dependency injection in core functions:
```python
from typing import Protocol

class Logger(Protocol):
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str, exc_info: bool = False) -> None: ...

def process_data(
    config: ProcessingConfig,
    storage_client: storage.Client,
    logger: Logger | None = None
) -> ProcessingResult:
    if logger:
        logger.info(f"Starting processing for {config.project_id}")
    return result
```

This Protocol is compatible with `logging.Logger`, Dagster's `context.log`, and test mocks — no adapters needed.

**MUST** create module-level loggers with `__name__`:
```python
logger = logging.getLogger(__name__)
```

### Structured Context

**MUST** include context in log messages:
```python
logger.info(f"Processing tronko run {tronko_run_id} for project {project_id}")
logger.warning(f"Found {len(missing)} missing of {len(expected)} expected")
context.log.error(f"Failed gs://{bucket}/{path}: {error}", exc_info=True)
```

### Log Levels

**MUST** use appropriate levels:
- `logger.debug()`: Detailed debugging
- `logger.info()`: Progress, status
- `logger.warning()`: Non-blocking warnings
- `logger.error()`: Operation failures (include `exc_info=True`)

**MUST NOT** log PII.

---

## 6. Dagster

### Asset Tagging Requirements

**CRITICAL: ALL assets MUST include required tags.**

#### Required Tags (MUST have all four)

```python
@asset(
    tags={
        "domain": "taxonomy",           # REQUIRED
        "data_tier": "bronze",          # REQUIRED
        "tool": "python",               # REQUIRED
        "pipeline_stage": "ingestion",  # REQUIRED
        # ... additional recommended tags
    },
)
def my_asset(context, ...):
    pass
```

#### Tag Schema

**`domain` (REQUIRED)**
- Values: `genomics`, `taxonomy`, `biodiversity`, `quality_control`, `reporting`, `monitoring`

**`data_tier` (REQUIRED)**
- Values: `bronze`, `silver`, `gold`

**`tool` (REQUIRED)**
- Values: `python`, `dbt`, `r`, `spark`, `custom`

**`pipeline_stage` (REQUIRED)**
- Values: `ingestion`, `validation`, `transformation`, `enrichment`, `aggregation`, `export`

**`source` (RECOMMENDED)**
- Values: `genbank`, `gbif`, `ncbi`, `inaturalist`, `user_upload`, `internal`

**`sensitivity` (RECOMMENDED)**
- Values: `public`, `internal`, `restricted`

**`update_frequency` (RECOMMENDED)**
- Values: `realtime`, `hourly`, `daily`, `weekly`, `monthly`, `adhoc`

**`compute_intensity` (OPTIONAL)**
- Values: `light`, `medium`, `heavy`

**`priority` (OPTIONAL)**
- Values: `critical`, `high`, `medium`, `low`

### Separation: `tags` vs `op_tags`

**MUST** separate categorization from operational tags:

```python
@asset(
    tags={
        # Categorization
        "domain": "taxonomy",
        "data_tier": "bronze",
        "tool": "python",
        "pipeline_stage": "ingestion",
    },
    op_tags={
        # Operational configuration
        "dagster/concurrency_key": "ncbi_api_limit",
        "dagster-k8s/config": {
            "pod_template_spec_metadata": {"labels": {"duration": "extended"}},
        },
        "dagster-k8s/resource_requirements": {
            "requests": {"cpu": "4", "memory": "16Gi"},
            "limits": {"cpu": "4", "memory": "32Gi"},
        },
    },
)
def reference_taxonomies(context, ...):
    pass
```

### Tag-Based Job Selection

**SHOULD** use tag-based selection (enables automatic inclusion of new assets):

```python
from dagster import AssetSelection, define_asset_job

# ✅ GOOD - Tag-based
project_metadata_job = define_asset_job(
    name="project_metadata_job",
    selection=AssetSelection.tag("domain", "biodiversity").tag("pipeline_stage", "ingestion"),
)

# ✅ GOOD - Complex selection
critical_ingestion = AssetSelection.tag("priority", "critical") \
    .tag("data_tier", "bronze") \
    .tag("pipeline_stage", "ingestion")

# ❌ LEGACY - Direct list (avoid)
legacy_job = define_asset_job(
    name="legacy_job",
    selection=["asset1", "asset2"],  # Requires manual updates
)
```

### Shared Tag Dictionaries

**MUST** use shared tag dictionaries for consistency:

```python
# Module level
TAXONOMY_BASE_TAGS = {
    "domain": "taxonomy",
    "data_tier": "bronze",
    "tool": "python",
    "sensitivity": "public",
}

@asset(
    tags={
        **TAXONOMY_BASE_TAGS,
        "source": "ncbi",
        "pipeline_stage": "ingestion",
    },
)
def reference_taxonomies(context, ...):
    pass
```

### Tag Validation

**MUST** validate tags conform to schema. Required tags checked:
- `domain`
- `data_tier`
- `tool`
- `pipeline_stage`

**Validation**: Tags are validated at review time via the pre-commit checklist.

---

## 7. Machine Learning

### scikit-learn Pipelines

**MUST** use `Pipeline` for preprocessing + model:
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42)),
])
```

### Seeded RNG

**MUST** set `random_state` for reproducibility:
```python
model = RandomForestClassifier(random_state=42)  # ✅ GOOD
```

### Model Persistence

**MUST** persist models with metadata. Use `joblib` for scikit-learn models.

---

## 8. Testing

### Test Organization

**MUST** mirror source structure:
```
src/edna_dagster_pipelines/
├── core/qcassign/combine_tronko_results.py
└── ops/qcassign/combine_tronko_results.py

tests/
├── core/qcassign/test_combine_tronko_results.py  # Unit
└── ops/qcassign/test_combine_tronko_results.py   # Integration
```

### Test Types

**MUST** write unit tests for core functions:
```python
def test_get_failed_jobs_empty():
    mock_instance = Mock()
    mock_instance.get_runs.return_value = []
    result = get_failed_jobs(mock_instance, config)
    assert result.total_failures == 0
```

**MUST** write integration tests for ops/assets:
```python
@pytest.mark.integration
def test_combine_tronko_results_op():
    context = build_op_context(resources={"gcs": mock_gcs})
    result = combine_tronko_results(context=context, gcs=mock_gcs)
    assert result == "expected_run_id"
```

### Mocking

**MUST** mock external dependencies, not business logic:
```python
# ✅ GOOD - Mock external service
@patch("google.cloud.storage.Client")
def test_upload_file(mock_storage_client):
    # ...
    mock_blob.upload_from_filename.assert_called_once()

# ❌ BAD - Mocking function under test
@patch("mymodule.process_data")
def test_process_data(mock_process):
    # Tests nothing!
```

### Test Naming

**MUST** use descriptive names:
```python
# ✅ GOOD
def test_combine_tronko_results_with_multiple_chunks():
    pass

# ❌ BAD
def test_combine(): pass
```

### Deterministic Tests

**MUST** ensure determinism:
- Set `random_state` for ML
- Mock time-dependent functions
- Use fixtures for database state
- Avoid network calls (mock APIs)

---

## 9. Security

### Secrets Management

**MUST** load from environment/secret managers:
```python
DATABASE_URL = os.environ["DATABASE_URL"]  # ✅ GOOD
```

**MUST NOT** commit secrets. Add `.env` to `.gitignore`, use `gcp_secrets_manager`, never log secrets.

### Input Validation

**MUST** validate and sanitize external inputs:
```python
def safe_file_path(base_dir: str, user_path: str) -> str:
    safe_path = os.path.normpath(os.path.join(base_dir, user_path))
    if not safe_path.startswith(os.path.abspath(base_dir)):
        raise ValueError(f"Path outside base: {user_path}")
    return safe_path
```

---

## 10. Performance

### Algorithmic Complexity

**MUST** avoid O(n²) on large datasets:
```python
# ✅ GOOD - O(n)
list2_set = set(list2)
for item in list1:
    if item in list2_set:  # O(1) lookup
        pass
```

**MUST** document complexity for non-trivial algorithms.

### Lazy Evaluation (Polars)

**SHOULD** use lazy evaluation:
```python
df = pl.scan_csv("large.csv") \
    .filter(pl.col("value") > 0) \
    .select(["id", "value"]) \
    .collect()
```

### Memory Management

**SHOULD** process large files in chunks. **MUST** close resources (use context managers).

---

## Enforcement Checklist

Before committing:

- [ ] **Types**: All functions typed, `poetry run pyright` passes
- [ ] **DataFrames**: Pandera schemas for pipeline boundaries
- [ ] **Validation**: Pydantic models for external data
- [ ] **Errors**: Descriptive messages, cleanup in `finally`
- [ ] **Logging**: Duck-typed logger in core, structured context
- [ ] **Dagster**: ALL required tags (`domain`, `data_tier`, `tool`, `pipeline_stage`)
- [ ] **Dagster**: `tags` vs `op_tags` separation maintained
- [ ] **Testing**: Unit tests for core, integration tests for ops
- [ ] **Security**: No hardcoded secrets, paths sanitized
- [ ] **Performance**: No O(n²) on large data, complexity documented

---

## Pre-Commit Commands

**MUST** run all and fix errors:

```bash
poetry run ruff format    # 1. Format
poetry run ruff check     # 2. Lint
poetry run pyright        # 3. Type check
```

**Tests**: Run tests for the project(s) you modified, matching CI patterns:
```bash
poetry run pytest projects/edna_dagster_pipelines/tests -v   # Dagster pipelines
poetry run pytest libraries/core-analysis-lib/tests -v       # Core analysis lib
poetry run pytest projects/edna_explorer_reports/tests -v    # Explorer reports
poetry run pytest projects/report_api/tests -v               # Report API
```

---

## Validation Decision Guide

**When to use Pydantic vs Pandera:**

| Scenario | Tool | Why |
|----------|------|-----|
| API request/response, JSON | Pydantic | Optimized for JSON, generates schemas |
| Config files, env vars | Pydantic BaseSettings | Purpose-built for config |
| CSV/Parquet ingestion | Pandera | Efficient whole-DataFrame validation |
| ML feature validation | Pandera | Statistical constraints |
| Mixed: JSON with embedded table | Both | Pydantic for structure, Pandera for table |

**Production tip**: Use `PANDERA_VALIDATION_DEPTH=SCHEMA_ONLY` for lighter validation in production.

**Type narrowing**: Use `isinstance()` checks for type narrowing with union types:
```python
def handle_value(val: int | str) -> str:
    if isinstance(val, int):
        return str(val)  # Pyright knows val is int here
    return val  # Pyright knows val is str here
```

---

## References

- **Pyright Config**: `pyproject.toml` (see `[tool.pyright]` section)

---

**End of Agent Rules**
