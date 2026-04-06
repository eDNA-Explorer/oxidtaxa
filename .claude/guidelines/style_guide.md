# eDNA Explorer - Python Style Guide

Quick reference for code formatting, naming conventions, and quality standards.

## Table of Contents
1. [Package Layout](#package-layout)
2. [Naming Conventions](#naming-conventions)
3. [Docstrings](#docstrings)
4. [Import Ordering](#import-ordering)
5. [Code Formatting](#code-formatting)
6. [Ruff Configuration](#ruff-configuration)
7. [Type Checking](#type-checking)

---

## Package Layout

### Function-Op-CLI Trinity Pattern (Dagster Development Only)

All new Dagster jobs and assets MUST follow this architecture. This pattern does **not** apply to standalone scripts, library code, or analysis notebooks:

```
edna_dagster_pipelines/
├── core/{module}/              # Pure business logic (NO Dagster deps)
│   ├── __init__.py
│   └── {feature_name}.py       # Core function + config/result dataclasses
│
├── ops/{module}/               # Dagster wrappers (thin layer)
│   ├── __init__.py
│   └── {feature_name}.py       # Dagster op calling core function
│
├── cli/                        # CLI debug interfaces
│   └── {feature_name}.py       # Click command calling core function
│
└── jobs/                       # Job definitions
    └── {job_name}.py
```

### Module Organization

```
projects/
├── edna_dagster_pipelines/     # Main data pipeline project
│   ├── src/edna_dagster_pipelines/
│   └── tests/
├── edna_explorer_reports/      # Report generation service
└── report_api/                 # Report API service

libraries/
├── edna-db-lib/                # Database models and sessions
├── logging-lib/                # Shared logging utilities
└── core-analysis-lib/          # Shared analysis functions
```

---

## Naming Conventions

### Files and Modules
- **Modules**: `snake_case.py`
- **Test files**: `test_{module_name}.py`
- **Core functions**: `{action}_{subject}.py` (e.g., `combine_tronko_results.py`)
- **Ops**: Match core function name (e.g., `combine_tronko_results.py`)

**Examples:**
```python
# Good
core/qcassign/combine_tronko_results.py
ops/qcassign/combine_tronko_results.py
tests/core/qcassign/test_combine_tronko_results.py

# Bad
core/qcassign/CombineTronkoResults.py
ops/qcassign/combine-tronko-results.py
```

### Functions and Methods
- **Functions**: `snake_case`
- **Core functions**: Descriptive verb + noun (e.g., `combine_tronko_results`). No `_core` suffix needed — being in `core/` directory makes the layer obvious.
- **Ops**: Match core function name (e.g., `combine_tronko_results`)
- **Private functions**: Leading underscore `_helper_function`
- **Test functions**: `test_{what_is_being_tested}` (e.g., `test_get_failed_jobs_no_failures`)

**Examples:**
```python
# Good - core function (lives in core/{module}/)
def combine_tronko_results(config: CombineTronkoConfig) -> CombineTronkoResult:
    """Core function to combine tronko results."""
    pass

# Good - Dagster op (lives in ops/{module}/)
@op
def combine_tronko_results(context: OpExecutionContext, ...):
    """Dagster op for combining tronko results."""
    config = CombineTronkoConfig(...)
    return combine_tronko_results(config, context.log)

def test_combine_tronko_results_empty_chunks():
    """Test combining with no chunk results."""
    pass

# Bad
def CombineTronkoResults(...):  # PascalCase for function
def combine_results(...):       # Too generic
def test_combine(...):          # Not descriptive enough
```

### Classes
- **Classes**: `PascalCase`
- **Config dataclasses**: `{Feature}Config`
- **Result dataclasses**: `{Feature}Result`
- **Private classes**: Leading underscore `_InternalHelper`

**Examples:**
```python
# Good
@dataclass
class CombineTronkoConfig:
    """Configuration for combining tronko results."""
    project_id: str
    primer: str
    tronko_run_id: str

@dataclass
class CombineTronkoResult:
    """Result of combining tronko results."""
    tronko_run_id: str
    duration_seconds: int

class MemoryMonitor:
    """Helper class for memory monitoring."""
    pass

# Bad
@dataclass
class combine_tronko_config:  # snake_case class name
    pass

@dataclass
class Config:  # Too generic
    pass
```

### Variables and Constants
- **Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE` (module-level only)
- **Type variables**: `PascalCase` with suffix `T` (e.g., `ConfigT`)

**Examples:**
```python
# Good
COMBINED_INPUT_FILES_PATH = "projects/{}/assign/{}/intermediary/assigned/{}"
MAX_RETRY_ATTEMPTS = 3

project_id = "test-project"
chunk_count = len(results)

# Bad
ProjectId = "test-project"  # PascalCase for variable
max_retry_attempts = 3      # lowercase for constant
CHUNK_COUNT = 5            # UPPER_CASE for local variable
```

### Dagster Ops and Assets
- **Op names**: Match function name, use `snake_case`
- **Asset names**: Use `snake_case`, descriptive of data product
- **Op output keys**: Use `snake_case`

**Examples:**
```python
# Good
@op(
    out={
        "tronko_run_id": Out(String),
        "files_combined": Out(Int),
    }
)
def combine_tronko_results(...):
    pass

@asset(
    name="project_tronko_results",
    key_prefix=["qcassign"]
)
def project_tronko_results(...):
    pass

# Bad
@op
def CombineTronkoResults(...):  # PascalCase
    pass

@asset(name="Project-Tronko-Results")  # kebab-case
def projectTronkoResults(...):  # camelCase
    pass
```

---

## Docstrings

### Style: Google Format

**Ruff enforces Google-style docstrings** (D213, D214, etc.).

### Module Docstrings
```python
"""
Core Tronko Results Combination Logic

This module contains the pure business logic for combining tronko results
without any Dagster-specific dependencies.
"""
```

### Function Docstrings
```python
def combine_tronko_results_core(
    config: CombineTronkoConfig,
    storage_client: storage.Client,
    logger=None
) -> CombineTronkoResult:
    """
    Core function to combine tronko results from multiple chunks.

    This function contains the pure business logic without any Dagster dependencies.

    Args:
        config: Configuration for the combine operation
        storage_client: Google Cloud Storage client
        logger: Optional logger for output (should have .info, .warning, .error methods)

    Returns:
        CombineTronkoResult with operation details

    Raises:
        ValueError: If config is invalid
        StorageError: If GCS operations fail
    """
    pass
```

### Class Docstrings
```python
@dataclass
class CombineTronkoConfig:
    """Configuration for combining tronko results."""
    project_id: str
    primer: str
```

### Test Docstrings
```python
def test_get_failed_jobs_no_failures(mock_instance):
    """Test get_failed_jobs with no failed jobs."""
    pass

@pytest.fixture
def mock_instance():
    """Mock Dagster instance for testing."""
    return Mock()
```

### Dagster Op Docstrings
```python
@op
def combine_tronko_results(
    context: OpExecutionContext,
    gcs: GCSResource,
    tronko_chunk_results: list[str],
):
    """
    Combines the results from multiple tronko_assign chunk operations.

    This op wraps the core business logic and handles Dagster-specific concerns
    like resource management and logging.

    Args:
        context: The Dagster execution context
        gcs: GCS resource
        tronko_chunk_results: List of result paths from tronko_assign operations

    Returns:
        The Tronko run ID (passed through)
    """
    pass
```

---

## Import Ordering

### Ruff isort Configuration

Imports are automatically sorted by Ruff with these rules:

```toml
[tool.ruff.lint.isort]
known-first-party = [
    "edna_dagster_pipelines",
    "edna_explorer_reports",
    "report_api",
    "edna_db",
    "core_analysis",
    "logging_lib"
]
```

### Standard Import Order

1. Standard library imports
2. Third-party imports
3. First-party imports (project-specific)
4. Local/relative imports

**Example:**
```python
# Standard library
import gc
import os
from dataclasses import dataclass
from datetime import datetime

# Third-party
import psutil
from dagster import op, OpExecutionContext, Out
from google.cloud import storage

# First-party (project-specific)
from edna_dagster_pipelines.constants import COMBINED_INPUT_FILES_PATH
from edna_dagster_pipelines.qcassign.helpers.combine_files import combine_files
from edna_db.schema import TronkoRun

# Local (relative imports - use sparingly)
from .helpers import process_chunks
```

### Import Style Guidelines

- **Prefer absolute imports** over relative imports
- **Group related imports** from same package
- **Use `from X import Y`** for frequently used items
- **Use `import X`** for packages/modules used infrequently

```python
# Good
from datetime import datetime
from edna_dagster_pipelines.core.qcassign.combine_tronko_results import (
    CombineTronkoConfig,
    combine_tronko_results_core,
)

# Bad
from datetime import *  # Wildcard imports
from edna_dagster_pipelines.core.qcassign.combine_tronko_results import CombineTronkoConfig, combine_tronko_results_core, CombineTronkoResult  # Long line
```

---

## Code Formatting

### Line Length
- **Maximum: 88 characters** (Black/Ruff standard)
- **Exception**: E501 ignored in `scripts/**`, `migration/**`, and legacy projects during migration

```python
# Good - within 88 characters
result = combine_tronko_results_core(
    config, storage_client, logger
)

# Good - multiline for readability
logger.info(
    f"Combining {len(config.tronko_chunk_results)} tronko chunk results "
    f"for primer {config.primer}"
)

# Bad - exceeds 88 characters
result = combine_tronko_results_core(config, storage_client, logger, extra_param, another_param, yet_another_param)
```

### Quotes
- **Double quotes** for strings: `"hello"`
- **Ruff enforces**: `quote-style = "double"`

```python
# Good
message = "Processing tronko results"
path = f"projects/{project_id}/assign"

# Bad
message = 'Processing tronko results'  # Single quotes
```

### Indentation
- **4 spaces** per indentation level
- **Never tabs**
- **Ruff enforces**: `indent-style = "space"`

### Trailing Commas
- **Preserve magic trailing commas** (helps with version control diffs)

```python
# Good
config = CombineTronkoConfig(
    project_id="test",
    primer="16S",
    tronko_run_id="123",  # Trailing comma
)

# Also good
config = CombineTronkoConfig(project_id="test", primer="16S")
```

### Line Endings
- **Auto-detected** (`line-ending = "auto"`)
- Generally LF (`\n`) on Unix/Mac, CRLF (`\r\n`) on Windows

---

## Ruff Configuration

### Target Version
- **Python 3.12** (`target-version = "py312"`)

### Enabled Rules

```toml
[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors (PEP 8 violations)
    "F",  # pyflakes (undefined names, unused imports)
    "I",  # isort (import sorting)
    "B",  # flake8-bugbear (likely bugs and design problems)
    "C4", # flake8-comprehensions (better list/dict/set comprehensions)
    "UP", # pyupgrade (upgrade syntax for newer Python versions)
]
ignore = []
```

### What Each Rule Category Covers

- **E (pycodestyle)**: Whitespace, line length, indentation, blank lines
- **F (pyflakes)**: Undefined variables, unused imports, syntax errors
- **I (isort)**: Import ordering and grouping
- **B (flake8-bugbear)**: Likely bugs (mutable defaults, bare except, etc.)
- **C4 (comprehensions)**: Simplify comprehensions and generator expressions
- **UP (pyupgrade)**: Modern Python syntax (f-strings, type hints, etc.)

### Running Ruff

```bash
# Check for issues
poetry run ruff check

# Format code
poetry run ruff format

# Fix auto-fixable issues
poetry run ruff check --fix
```

### Per-File Ignores

Certain files have relaxed rules during migration:

```toml
[tool.ruff.lint.per-file-ignores]
"dev/**/*.py" = ["E501"]  # Ignore line length
"scripts/**/*.py" = ["E501"]
"migration/**/*.py" = ["E501"]
"projects/edna_explorer_reports/**/*.py" = ["E501"]  # Temporary during migration
"projects/edna_dagster_pipelines/**/*.py" = ["E501"]
```

---

## Type Checking

### Pyright Configuration

```toml
[tool.pyright]
executionEnvironments = [
  { root = "." }
]
stubPath = "stubs"
```

### Type Hint Guidelines

- **All function signatures** should have type hints
- **Use modern syntax** (Python 3.10+): `list[str]` not `List[str]`
- **Use `T | None`** for nullable types (not `Optional[T]`)
- **Dataclasses** should have typed fields

**Examples:**
```python
# Good - Modern type hints (Python 3.10+)
def process_chunks(
    chunk_paths: list[str],
    output_path: str,
    logger: logging.Logger | None = None
) -> tuple[int, int]:
    """Process chunks and return (lines_processed, bytes_uploaded)."""
    pass

@dataclass
class CombineTronkoConfig:
    """Configuration for combining tronko results."""
    project_id: str
    primer: str
    tronko_run_id: str
    tronko_chunk_results: list[str]
    memory_usage_mb: float | None = None

# Bad - Old-style type hints
from typing import List, Tuple, Optional

def process_chunks(
    chunk_paths: List[str],  # Use list[str]
    output_path: str,
    logger: Optional[logging.Logger] = None  # Use Logger | None
) -> Tuple[int, int]:  # Use tuple[int, int]
    pass
```

### Running Pyright

```bash
# Type check entire project
poetry run pyright

# Type check specific file
poetry run pyright path/to/file.py
```

### Common Type Patterns

```python
from typing import Protocol

# Logger Protocol — see rules.md § 5. Logging for canonical definition
# Core function with Protocol-typed logger
def my_function(logger: Logger | None = None) -> None:
    if logger:
        logger.info("Processing")

# Storage client
from google.cloud import storage

def process_data(storage_client: storage.Client) -> None:
    pass

# Dagster context (ops only, never in core)
from dagster import OpExecutionContext

@op
def my_op(context: OpExecutionContext) -> str:
    context.log.info("Processing")
    return "done"

# Dataclass with optional fields
@dataclass
class Result:
    status: str
    error_message: str | None = None
    retry_count: int = 0
```

---

## Quality Checklist

Before committing code, run pre-commit commands from `rules.md` § Pre-Commit Commands, then verify:

- [ ] All functions have Google-style docstrings
- [ ] Type hints on all function signatures
- [ ] Imports sorted correctly (Ruff handles this)
- [ ] Line length ≤ 88 characters (except approved exceptions)
- [ ] No unused imports or variables

---

## Common Patterns

### Core Function Pattern

```python
"""Module docstring describing business logic."""

from dataclasses import dataclass

@dataclass
class MyFeatureConfig:
    """Configuration for my feature."""
    project_id: str
    some_param: int

@dataclass
class MyFeatureResult:
    """Result of my feature operation."""
    status: str
    items_processed: int

def my_feature_core(
    config: MyFeatureConfig,
    logger=None
) -> MyFeatureResult:
    """
    Core function implementing business logic.

    Args:
        config: Configuration for the operation
        logger: Optional logger

    Returns:
        MyFeatureResult with operation details
    """
    if logger:
        logger.info(f"Processing {config.project_id}")

    # Business logic here

    return MyFeatureResult(
        status="success",
        items_processed=42,
    )
```

### Dagster Op Pattern

```python
"""Dagster op wrapper for my feature."""

from dagster import op, OpExecutionContext, Out

from edna_dagster_pipelines.core.mymodule.my_feature import (
    MyFeatureConfig,
    my_feature_core,
)

@op(
    out={"status": Out(str)},
    tags={"dagster/concurrency_key": "my_feature"},
)
def my_feature(
    context: OpExecutionContext,
    project_id: str,
    some_param: int,
) -> str:
    """
    Dagster op for my feature.

    Args:
        context: Dagster execution context
        project_id: Project identifier
        some_param: Some parameter

    Returns:
        Operation status
    """
    config = MyFeatureConfig(
        project_id=project_id,
        some_param=some_param,
    )

    result = my_feature_core(config, context.log)

    context.log.info(f"Processed {result.items_processed} items")
    return result.status
```

### Test Pattern

```python
"""Tests for my feature core functionality."""

import pytest
from unittest.mock import Mock

from edna_dagster_pipelines.core.mymodule.my_feature import (
    MyFeatureConfig,
    MyFeatureResult,
    my_feature_core,
)

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock()

def test_my_feature_success(mock_logger):
    """Test successful execution of my_feature_core."""
    config = MyFeatureConfig(
        project_id="test-project",
        some_param=42,
    )

    result = my_feature_core(config, mock_logger)

    assert isinstance(result, MyFeatureResult)
    assert result.status == "success"
    assert result.items_processed > 0
    mock_logger.info.assert_called()

def test_my_feature_invalid_config():
    """Test my_feature_core with invalid configuration."""
    config = MyFeatureConfig(
        project_id="",  # Invalid empty project_id
        some_param=-1,
    )

    with pytest.raises(ValueError, match="project_id cannot be empty"):
        my_feature_core(config)
```

---

## Additional Resources

- **Ruff Documentation**: https://docs.astral.sh/ruff/
- **Google Style Guide**: https://google.github.io/styleguide/pyguide.html
