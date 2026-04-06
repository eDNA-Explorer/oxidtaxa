# Python Coding Best Practices

Practical patterns for implementing reliable, testable code in the eDNA Explorer Data Pipelines.

## 1. ABC vs Protocol Decision Matrix

### Use Protocol (PREFERRED)

Use Protocols for dependency injection and interface definitions.

```python
from typing import Protocol

class Logger(Protocol):
    def info(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...

def process_files(bucket: Bucket, logger: Logger) -> Result:
    """Works with ANY object implementing these methods."""
    logger.info("Starting...")
    return Result(processed=len(blobs))
```

**When to use Protocol:**
- Dependency injection interfaces
- Maximum decoupling needed
- No shared implementation required
- Duck typing with type safety

### Comparison

| Feature | Protocol | ABC |
|---------|----------|-----|
| Type check | Structural (behavior) | Nominal (inheritance) |
| Inheritance | Not required | Required |
| Runtime check | Requires `@runtime_checkable` | Built-in `isinstance()` |
| Best for | Decoupling, dependency injection | Frameworks, shared code |

**Use ABC**: Shared implementation in base class, runtime `isinstance()` checks needed, building framework.

---

## 2. Dataclass Patterns

### Configuration Objects

```python
@dataclass
class ProcessingConfig:
    """Configuration input - with sensible defaults."""
    max_errors: int = 2
    min_overlap: int = 10
    quality_threshold: float = 25.0
```

### Result Objects

```python
@dataclass
class ProcessingResult:
    """Result output - all fields explicit."""
    duration_seconds: int
    files_processed: int
    errors_found: int
    success: bool
    timestamp: datetime
    memory_usage_mb: float | None = None  # Optional metrics
```

**Why**: Type safety, IDE autocomplete, no boilerplate `__init__`, self-documenting, easy to test.

---

## 3. Dependency Injection Patterns

### Inject All Dependencies

```python
def process_files(
    file_paths: list[str],
    storage_client: storage.Client,  # Injected
    logger: Logger | None = None,    # Injected (Logger Protocol)
    config: ProcessingConfig         # Injected
) -> ProcessingResult:
    """All dependencies injected - easy to test with mocks."""
    logger.info(f"Processing {len(file_paths)} files")
    bucket = storage_client.bucket("my-bucket")
    return result
```

**Why**: Testability, flexibility, explicitness, no hidden dependencies.

### Testing with Injected Dependencies

```python
def test_process_files():
    mock_storage = Mock(spec=storage.Client)
    mock_logger = Mock(spec=Logger)
    config = ProcessingConfig(max_errors=1)

    result = process_files(["file1.txt"], mock_storage, mock_logger, config)

    mock_logger.info.assert_called()
    assert result.success
```

---

## 4. Error Handling Patterns

### Fail Fast with Validation

```python
def get_failed_jobs(instance: DagsterInstance, config: FailedJobsConfig) -> FailedJobsResult:
    # Validate inputs immediately
    if config.hours_lookback <= 0:
        raise ValueError("hours_lookback must be positive")
    if not instance:
        raise ValueError("instance cannot be None")
    # ... process with valid inputs
```

### Context-Rich Error Messages

```python
def generate_summary(failed_jobs_data: str) -> str | None:
    """Returns None on error for graceful degradation."""
    try:
        response = client.generate(data=failed_jobs_data)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        print(f"Data length: {len(failed_jobs_data) if failed_jobs_data else 'None'}")
        return None  # Graceful degradation
```

### Error Propagation

```python
try:
    result = process_file(file_path)
except Exception as e:
    logger.error(f"Failed to process {file_path}: {e}")
    raise  # Re-raise after logging context
```

---

## 5. Type Annotation Best Practices

### Modern Python Syntax (3.10+)

```python
# Modern syntax (GOOD)
results: list[str] = []
mapping: dict[str, int] = {}
optional_value: str | None = None

# Old syntax (AVOID)
from typing import List, Dict, Optional
results: List[str] = []  # Don't use
```

### Protocol with Attributes

```python
class FeatureImportanceExtractor(Protocol):
    """Protocol with both methods and attributes."""
    feature_importances_: np.ndarray  # Attribute

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
```

### Complete Function Signatures

```python
def process_data(
    input_path: str,
    config: ProcessingConfig,
    storage_client: storage.Client,
    logger: Logger | None = None
) -> ProcessingResult | None:
    """Fully typed signature with defaults."""
    pass
```

---

## 6. Testing Patterns with pytest

### Use Fixtures for Setup

```python
@pytest.fixture
def mock_instance():
    """Reusable mock for tests."""
    return Mock()

@pytest.fixture
def sample_config():
    """Reusable config for tests."""
    return FailedJobsConfig(hours_lookback=24)

def test_get_failed_jobs_no_failures(mock_instance, sample_config):
    """Test one thing: no failures scenario."""
    mock_instance.get_run_records.return_value = []

    result = get_failed_jobs(mock_instance, sample_config)

    assert isinstance(result, FailedJobsResult)
    assert result.total_failures == 0
```

### Use Patches for External Dependencies

```python
@patch("edna_dagster_pipelines.qcassign.ops.combine_tronko_results.combine_tronko_results_core")
@patch("edna_dagster_pipelines.qcassign.ops.combine_tronko_results.db_session")
def test_combine_tronko_results_success(mock_db_session, mock_combine_core, mock_context):
    """Patch at the module where they're USED, not where defined."""

    mock_combine_core.return_value = CombineTronkoResult(...)
    session = Mock()
    mock_db_session.return_value.__enter__.return_value = session

    result = combine_tronko_results(mock_context, ...)

    assert result == "tronko-run-1"
    mock_combine_core.assert_called_once()
```

### Test Naming Convention

```python
def test_{function_name}_{scenario}():
    """
    Pattern: test_<what>_<scenario>

    Examples:
    - test_get_failed_jobs_no_failures
    - test_get_failed_jobs_with_failures
    - test_config_validation_negative_hours
    """
    pass
```

### Test Organization

```python
class TestProcessingConfig:
    """Group related tests in classes."""

    def test_defaults(self):
        """Test default values."""
        config = ProcessingConfig()
        assert config.max_errors == 2

    def test_custom_values(self):
        """Test custom initialization."""
        config = ProcessingConfig(max_errors=5)
        assert config.max_errors == 5
```

---

## Summary

**Key Patterns**:
- Protocol for interfaces (PREFERRED over ABC)
- Dataclasses for config/results
- Inject ALL dependencies
- Fail fast with descriptive errors
- Complete type annotations with modern syntax
- Fixtures for test setup, patches for external deps

**Pre-commit commands**: See `rules.md` § Pre-Commit Commands.
