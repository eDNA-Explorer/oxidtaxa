# Pre-Commit Review Checklist

This checklist MUST be completed before committing any code. All items must pass.

## Critical Subset (Minimum Viable Review)

If you only check a few things, check these:

1. `poetry run ruff format && poetry run ruff check` — zero violations
2. `poetry run pyright` — zero errors
3. `poetry run pytest <project>/tests -v` — all passing for modified project(s)
4. No Dagster imports in `core/` files
5. All function signatures have type annotations

## Automated Verification

Run these commands - ALL must pass with zero errors:

- [ ] **Type checking**: `poetry run pyright` - zero errors
- [ ] **Formatting**: `poetry run ruff format` - auto-applied
- [ ] **Linting**: `poetry run ruff check` - zero violations
- [ ] **Tests**: Run tests for modified project(s), matching CI patterns:
  - `poetry run pytest projects/edna_dagster_pipelines/tests -v`
  - `poetry run pytest libraries/core-analysis-lib/tests -v`
  - `poetry run pytest projects/edna_explorer_reports/tests -v`
  - `poetry run pytest projects/report_api/tests -v`

## Code Quality Checks

### Type Safety (Pyright)
- [ ] Pyright strict mode passes with zero errors
- [ ] All public function parameters have type annotations
- [ ] All public function return types have type annotations
- [ ] No unexplained `Any` types (must have `# pyright: ignore[reportUnknown*]` with rationale)
- [ ] Using modern syntax: `list[str]`, `dict[str, int]`, `str | None` (not `List`, `Dict`, `Optional`)

### Code Style (Ruff)
- [ ] `poetry run ruff format` applied successfully
- [ ] `poetry run ruff check` passes with zero violations
- [ ] No commented-out code blocks
- [ ] No console.log, print debugging statements, or breakpoints left in code

### Runtime Validation
- [ ] Pandera schemas added/updated for new DataFrame input/output
- [ ] Pydantic models created for new configs/DTOs at boundaries (API, CLI, external data)
- [ ] No Pydantic models propagated deep into business logic (avoiding "SerDes debt")
- [ ] DataFrame validation using Pandera (not row-by-row Pydantic)

### Testing (pytest)
- [ ] All tests pass with `poetry run pytest`
- [ ] Tests are deterministic (seeded RNG, no network/clock/filesystem flakiness)
- [ ] External dependencies mocked with `spec=` parameter
- [ ] Test names clearly describe scenarios
- [ ] Float comparisons use appropriate tolerance

## Architecture Checks

### Function-Op-Asset Trinity Pattern (MANDATORY for new Dagster jobs/assets only)
- [ ] Core function created in `edna_dagster_pipelines/core/{module}/`
- [ ] Core function has NO Dagster imports (`from dagster` forbidden)
- [ ] Core function depends on Protocol interfaces (not concrete implementations)
- [ ] All dependencies injected as function parameters (no globals, singletons)
- [ ] Config and Result dataclasses created for core function
- [ ] Dagster op/asset is thin wrapper calling core function
- [ ] Op/asset extracts concrete objects from Dagster resources
- [ ] Op/asset creates adapters implementing protocol interfaces
- [ ] CLI interface created if operation needs standalone execution (optional)

### Hexagonal Architecture (Resource Protocols)
- [ ] Protocol interfaces defined in `edna_dagster_pipelines/resources/protocols.py`
- [ ] Protocols use `typing.Protocol` for structural subtyping
- [ ] Protocols are focused (Interface Segregation Principle)
- [ ] Core functions depend on protocols (Dependency Inversion Principle)
- [ ] Adapters created for concrete implementations (GCS, BigQuery, Mocks)

### File Organization
- [ ] Pure business logic in `core/{module}/`
- [ ] Dagster wrappers in `ops/{module}/` or `assets/{module}/`
- [ ] Protocol interfaces in `resources/protocols.py`
- [ ] Adapters in `resources/gcs.py`, `resources/bigquery.py`, `resources/mocks.py`
- [ ] CLI interfaces in `cli/` (optional)
- [ ] Unit tests in `tests/core/{module}/`
- [ ] Integration tests in `tests/ops/{module}/` or `tests/assets/{module}/`

## Dagster-Specific

### Asset/Op Configuration
- [ ] Resources, IO, and partitions documented in docstrings
- [ ] Retries configured with `RetryPolicy` where appropriate
- [ ] Timeouts configured for external API calls
- [ ] Ops/assets are idempotent where feasible
- [ ] Asset checks added for data quality validation (blocking checks prevent bad data)

### CRITICAL: Asset Tagging (MANDATORY for all new assets)

**Required Tags** (must be present in `tags` parameter):
- [ ] `domain` - Business/scientific domain (e.g., `genomics`, `taxonomy`, `biodiversity`, `quality_control`, `reporting`, `monitoring`)
- [ ] `data_tier` - Medallion architecture tier (e.g., `bronze`, `silver`, `gold`)
- [ ] `tool` - Primary processing tool (e.g., `python`, `dbt`, `r`, `spark`, `custom`)
- [ ] `pipeline_stage` - Processing stage (e.g., `ingestion`, `validation`, `transformation`, `enrichment`, `aggregation`, `export`)

**Recommended Tags** (should be present when applicable):
- [ ] `source` - External data source (e.g., `genbank`, `gbif`, `ncbi`, `inaturalist`, `user_upload`, `internal`)
- [ ] `sensitivity` - Data sensitivity level (e.g., `public`, `internal`, `restricted`)
- [ ] `update_frequency` - Expected refresh rate (e.g., `realtime`, `hourly`, `daily`, `weekly`, `monthly`, `adhoc`)

**Optional Tags**:
- [ ] `compute_intensity` - Resource requirements (e.g., `light`, `medium`, `heavy`)
- [ ] `priority` - Business priority (e.g., `critical`, `high`, `medium`, `low`)
- [ ] `team` - Owning team
- [ ] `sla` - Service-level agreement (e.g., `15min`, `1h`, `24h`)
- [ ] `api` - API identifier for concurrency grouping

**Tag Separation** (CRITICAL):
- [ ] Categorization tags in `tags` parameter (for organization and automation)
- [ ] Operational tags in `op_tags` parameter (for K8s config and concurrency)
- [ ] NO mixing of concerns (categorization vs configuration)

**Tag Values** (MUST follow schema):
- [ ] All tag values follow allowed schema from `.claude/guidelines/dagster_tagging.md`
- [ ] No freeform or inconsistent tag values (e.g., "Taxonomy" vs "taxonomy")
- [ ] Tag values are lowercase and consistent

**Tag-Based Job Selection** (preferred approach):
- [ ] New jobs use `AssetSelection.tag()` for asset selection (where applicable)
- [ ] Avoid explicit asset lists in `define_asset_job()` when tag-based selection is clearer
- [ ] Tag-based selection enables self-service asset addition

**Example**:
```python
@asset(
    group_name="taxonomy",
    tags={  # MANDATORY: Categorization tags
        "domain": "taxonomy",
        "data_tier": "bronze",
        "tool": "python",
        "pipeline_stage": "ingestion",
        "source": "ncbi",
        "sensitivity": "public",
    },
    op_tags={  # Operational configuration
        "dagster/concurrency_key": "ncbi_api_limit",
        "dagster-k8s/resource_requirements": {...},
    },
)
def reference_taxonomies(...): ...
```

**Reference**: See `.claude/guidelines/dagster_tagging.md` for complete tag taxonomy and validation schema.

## ML-Specific (if applicable)

- [ ] ML pipelines use scikit-learn `Pipeline` objects
- [ ] Random number generators seeded for reproducibility (`random_state=42`)
- [ ] Train/val/test splits stratified where applicable
- [ ] Model artifacts versioned with metadata (model type, version, trained_at, random_state)
- [ ] Cross-validation seeded (`random_state` parameter)
- [ ] Model persistence includes metadata dictionary

## Logging & Error Handling

### Logging
- [ ] Core functions accept optional logger parameter (duck-typed, default `None`)
- [ ] Logs include context (IDs, counts, file paths, project names)
- [ ] No PII in log messages
- [ ] Appropriate log levels used (debug, info, warning, error)
- [ ] Error logs include `exc_info=True` for stack traces

### Error Handling
- [ ] Fail fast with descriptive, actionable error messages
- [ ] Error messages include context (identifiers, file paths, counts, values)
- [ ] Exceptions chained with `from e` for preserving stack traces
- [ ] Using standard exceptions (`ValueError`, `FileNotFoundError`, `RuntimeError`)
- [ ] No silent exception swallowing
- [ ] File paths validated (no directory traversal vulnerabilities)

## Performance

- [ ] Performance considerations noted in docstrings (if applicable)
- [ ] No obvious O(n²) operations over large datasets
- [ ] Lazy Polars evaluation (`scan_csv`, `lazy()`) used for large files
- [ ] Streaming/chunking used for large file processing
- [ ] Memory management considerations documented

## Security

- [ ] Secrets only via environment variables or Secret Manager
- [ ] No credentials committed to repository
- [ ] No secrets or API keys in log messages
- [ ] All external inputs validated
- [ ] File paths sanitized (no directory traversal)

## Documentation

### Code Documentation
- [ ] Docstrings added for all public functions/classes
- [ ] Docstrings explain "why", not just "what"
- [ ] Complex logic has inline comments explaining rationale
- [ ] Type hints provide self-documentation

### Guidelines (if modifying architecture/patterns)
- [ ] `.claude/guidelines/rules.md` updated if new rules added
- [ ] `.claude/guidelines/coding_principles.md` updated if Trinity pattern changed
- [ ] `.claude/guidelines/examples/patterns.md` updated if new patterns introduced
- [ ] `.claude/guidelines/arch_map.md` updated if structure changes

## Testing

### Test Coverage
- [ ] Unit tests added for core functions (`tests/core/{module}/`)
- [ ] Integration tests added for ops/assets (`tests/ops/{module}/` or `tests/assets/{module}/`)
- [ ] Test coverage for edge cases and error conditions
- [ ] Tests for protocol implementations (adapters)

### Test Quality
- [ ] Test names describe scenarios (not implementation details)
- [ ] One assertion per test when possible
- [ ] External dependencies mocked (network, filesystem, databases)
- [ ] Tests use fixtures for setup/teardown
- [ ] Tests are independent (no shared state between tests)

## Code Review

### RECOMMENDED: Code Reviewer Agent
- [ ] **`code-reviewer` agent run for detailed compliance review**
- [ ] Agent generated compliance score ≥ 70
- [ ] All MUST FIX items addressed
- [ ] All SHOULD FIX items addressed or documented

### Self-Review
- [ ] Code reviewed by author before requesting peer review
- [ ] No unnecessary complexity introduced
- [ ] Code follows DRY principle (Don't Repeat Yourself)
- [ ] No TODOs without issue numbers
- [ ] Commit messages explain "why", not just "what"

---

## Pre-Commit Command Summary

Run these commands in order before committing:

```bash
# 1. Format code
poetry run ruff format

# 2. Lint code
poetry run ruff check

# 3. Type check
poetry run pyright

# 4. Run tests
poetry run pytest

```

**If ANY command fails, DO NOT COMMIT. Fix the issues first.**

---

## References

- **Full Rules**: `.claude/guidelines/rules.md` - Non-negotiable requirements
- **Trinity Pattern**: `.claude/guidelines/tldr/coding_principles.tldr.md` - Architecture details
- **Best Practices**: `.claude/guidelines/best_practices.md` - Practical patterns
- **Style Guide**: `.claude/guidelines/style_guide.md` - Naming and formatting
- **Examples**: `.claude/guidelines/examples/patterns.md` - Runnable code snippets

---

**Last Updated**: 2026-02-10
