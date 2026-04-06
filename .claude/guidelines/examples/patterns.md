# Canonical Code Patterns

All snippets are runnable and follow project standards. Use these as templates when implementing similar functionality.

## 1. Polars Transform with Pandera Validation

Complete example showing DataFrame transformation with input/output schema validation.

```python
import polars as pl
import pandera.polars as pa

# Define schemas using Pandera's Polars-native API
input_schema = pa.DataFrameSchema({
    "transaction_id": pa.Column(int, pa.Check.gt(0), unique=True),
    "amount": pa.Column(float, pa.Check.ge(0)),
    "category": pa.Column(str, pa.Check.isin(["A", "B", "C"])),
})

output_schema = pa.DataFrameSchema({
    "transaction_id": pa.Column(int, pa.Check.gt(0), unique=True),
    "amount": pa.Column(float, pa.Check.ge(0)),
    "normalized_amount": pa.Column(float, [pa.Check.ge(0), pa.Check.le(1)]),
    "category": pa.Column(str, pa.Check.isin(["A", "B", "C"])),
})

def transform_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform transactions with schema validation.

    Args:
        df: Raw transaction data

    Returns:
        Transformed DataFrame with normalized amounts
    """
    # Validate input
    validated = input_schema.validate(df, lazy=True)

    # Get max value for normalization
    max_amount = validated.select(pl.col("amount").max()).item()

    # Apply transformation using with_columns
    result = validated.with_columns(
        (pl.col("amount") / max_amount).alias("normalized_amount")
    )

    # Validate output
    return output_schema.validate(result, lazy=True)

# Usage example
raw_data = pl.DataFrame({
    "transaction_id": [1, 2, 3],
    "amount": [100.0, 200.0, 150.0],
    "category": ["A", "B", "A"],
})

result = transform_transactions(raw_data)
# Pandera validates input and output
# Raises SchemaError if validation fails (lazy=True gives all errors at once)
```

## 2. Dagster Asset with Proper Tagging (CRITICAL)

Complete Function-Op-Asset Trinity pattern with comprehensive tagging strategy.

```python
from dagster import asset, AssetExecutionContext, AssetKey, AssetSelection, define_asset_job, ResourceParam
from edna_dagster_pipelines.resources.protocols import StoragePort
from dataclasses import dataclass
import polars as pl
import logging

# ============================================================================
# LAYER 1: Core Function (Pure Business Logic)
# ============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    project_id: str
    threshold: float = 0.05

@dataclass
class ProcessingResult:
    """Result from data processing."""
    files_processed: int
    data: pl.DataFrame

def process_data_core(
    storage: StoragePort,  # Protocol interface, not concrete implementation
    config: ProcessingConfig,
    logger: logging.Logger | None = None
) -> ProcessingResult:
    """
    Core business logic with NO Dagster dependencies.

    Args:
        storage: Storage abstraction (protocol)
        config: Processing configuration
        logger: Optional logger (duck-typed)

    Returns:
        Processing result with DataFrame
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Validate inputs
    if config.threshold <= 0 or config.threshold >= 1:
        raise ValueError(
            f"threshold must be between 0 and 1, got {config.threshold}"
        )

    logger.info(f"Processing project {config.project_id} with threshold {config.threshold}")

    # Business logic using protocol interface
    data_bytes = storage.read_bytes(f"projects/{config.project_id}/data.csv")

    # Process data
    data = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

    logger.info(f"Processed {len(data)} records")

    return ProcessingResult(files_processed=3, data=data)

# ============================================================================
# LAYER 2: Dagster Asset (Orchestration Wrapper)
# ============================================================================

# MANDATORY: All assets must include categorization tags
@asset(
    key=AssetKey(["reference_taxonomies"]),
    group_name="taxonomy",

    # REQUIRED: Categorization tags for organization and automation
    tags={
        # Required tags (MUST include all 4)
        "domain": "taxonomy",           # Business domain
        "data_tier": "bronze",          # Medallion architecture tier
        "tool": "python",               # Primary processing tool
        "pipeline_stage": "ingestion",  # Processing stage

        # Recommended tags (should include for better organization)
        "source": "ncbi",               # Data source system
        "sensitivity": "public",        # Data classification
        "update_frequency": "daily",    # Refresh rate

        # Optional tags (add as needed)
        "compute_intensity": "medium",  # Resource requirements
        "priority": "high",             # Business priority
    },

    # SEPARATE: Operational tags for execution control
    op_tags={
        # Concurrency control
        "dagster/concurrency_key": "ncbi_api_limit",

        # Kubernetes resource configuration
        "dagster-k8s/resource_requirements": {
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "8Gi"},
        },
    },
)
def reference_taxonomies(
    context: AssetExecutionContext,
    storage: ResourceParam[StoragePort],  # Type hint uses protocol
) -> pl.DataFrame:
    """
    Dagster asset wrapper with proper tagging.

    This is a thin wrapper that:
    1. Extracts configuration from context
    2. Calls core business logic
    3. Adds Dagster-specific metadata
    4. Returns result for materialization
    """
    # Build config from context/environment
    config = ProcessingConfig(
        project_id="edna-explorer",
        threshold=0.05
    )

    # Call core function (Dagster provides storage adapter via resources)
    result = process_data_core(
        storage=storage,  # Protocol implementation injected by Dagster
        config=config,
        logger=context.log  # Duck-typed logger
    )

    # Add Dagster-specific metadata
    context.add_output_metadata({
        "files_processed": result.files_processed,
        "rows_processed": len(result.data),
        "project_id": config.project_id,
    })

    return result.data

# ============================================================================
# Tag-Based Job Selection (PREFERRED over explicit asset lists)
# ============================================================================

# Select all assets with specific tags (automatically includes new assets)
taxonomy_ingestion_job = define_asset_job(
    name="taxonomy_ingestion",
    selection=AssetSelection.tag("domain", "taxonomy").tag("pipeline_stage", "ingestion"),
)

# Select all critical assets
critical_assets_job = define_asset_job(
    name="critical_updates",
    selection=AssetSelection.tag("priority", "critical"),
    tags={"dagster/priority": "10"},  # High execution priority
)

# Select all NCBI-sourced assets
ncbi_pipeline_job = define_asset_job(
    name="ncbi_pipeline",
    selection=AssetSelection.tag("source", "ncbi"),
)

# Select all bronze tier ingestion assets
bronze_ingestion_job = define_asset_job(
    name="bronze_ingestion",
    selection=AssetSelection.tag("data_tier", "bronze").tag("pipeline_stage", "ingestion"),
)
```

**Key Points:**
- **Separation of Concerns**: `tags` for categorization, `op_tags` for operational config
- **Required Tags**: MUST include `domain`, `data_tier`, `tool`, `pipeline_stage`
- **Tag-Based Selection**: Jobs select assets by tags, not explicit lists
- **Protocol-Based Dependencies**: Core function depends on abstractions, not concrete implementations

## 3. Pydantic Settings Object

Load configuration from environment with validation.

```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseSettings):
    """
    Load database configuration from environment variables.

    Environment variables:
        DB_HOST: Database hostname
        DB_PORT: Database port (default: 5432)
        DB_USER: Database username
        DB_PASSWORD: Database password
    """
    host: str = Field(..., env="DB_HOST")
    port: int = Field(5432, env="DB_PORT")
    username: str = Field(..., env="DB_USER")
    password: str = Field(..., env="DB_PASSWORD")

    class Config:
        env_file = ".env"  # Optional: load from .env file
        case_sensitive = False  # db_host, DB_HOST, Db_Host all work

# Usage
config = DatabaseConfig()  # Auto-loads from environment
print(f"Connecting to {config.host}:{config.port}")

# Validation happens automatically
# Raises ValidationError if required fields missing or invalid types
```

## 4. scikit-learn Pipeline with Seeded CV

Machine learning pipeline with reproducibility.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
from datetime import datetime

# MUST seed for reproducibility
RANDOM_STATE = 42

# Create pipeline with seeded components
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,  # ✅ Seeded
        n_jobs=-1
    ))
])

# MUST stratify splits for classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # ✅ Stratified
    random_state=RANDOM_STATE  # ✅ Seeded
)

# MUST seed cross-validation
scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5,
    scoring='accuracy',
    random_state=RANDOM_STATE  # ✅ Seeded
)

# Fit pipeline
pipeline.fit(X_train, y_train)

# MUST persist with metadata
metadata = {
    "model_type": "RandomForestClassifier",
    "version": "1.0.0",
    "trained_at": datetime.now().isoformat(),
    "random_state": RANDOM_STATE,
    "cv_scores": scores.tolist(),
    "mean_cv_score": scores.mean(),
    "std_cv_score": scores.std(),
    "test_score": pipeline.score(X_test, y_test),
}

# Save both pipeline and metadata
joblib.dump({
    "pipeline": pipeline,
    "metadata": metadata
}, "model_v1.joblib")

# Load later
loaded = joblib.load("model_v1.joblib")
model = loaded["pipeline"]
meta = loaded["metadata"]
print(f"Loaded model from {meta['trained_at']}")
```

## 5. Deterministic pytest Example

Testing with seeded data and mocked dependencies.

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch

@pytest.fixture
def seeded_data():
    """Fixture with seeded RNG for deterministic tests."""
    np.random.seed(42)
    return np.random.randn(100, 5)

@pytest.fixture
def mock_storage():
    """Fixture for mocked storage dependency."""
    mock = Mock(spec=['read_bytes', 'write_bytes'])
    mock.read_bytes.return_value = b"test data"
    return mock

def test_calculation_is_deterministic(seeded_data):
    """
    Test produces same result every run.

    Uses seeded fixture to ensure reproducibility.
    """
    result1 = calculate(seeded_data)
    result2 = calculate(seeded_data)

    # Should be exactly equal
    np.testing.assert_array_equal(result1, result2)

def test_calculation_output_shape(seeded_data):
    """Test output has expected shape."""
    result = calculate(seeded_data)

    assert result.shape == (100, 5)
    # Check specific values with tolerance
    expected_first_row = np.array([0.496714, -0.138264, 0.647689, 1.523030, -0.234153])
    np.testing.assert_array_almost_equal(result[0], expected_first_row, decimal=5)

def test_upload_with_mocked_storage(seeded_data, mock_storage):
    """
    Test business logic without real external dependencies.

    Uses mock storage to isolate logic under test.
    """
    result = upload_data(seeded_data, mock_storage)

    # Verify storage was called correctly
    mock_storage.write_bytes.assert_called_once()
    call_args = mock_storage.write_bytes.call_args
    assert call_args[0][0] == "output/data.csv"  # First positional arg (key)
    assert isinstance(call_args[0][1], bytes)    # Second positional arg (data)

    assert result.success is True

@patch('my_module.storage.Client')
def test_upload_with_patched_client(mock_client_class, seeded_data):
    """
    Test with patched external class.

    Useful when code creates instances internally.
    """
    # Setup mock instance
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance

    # Run code under test
    result = upload_data(seeded_data)

    # Verify client was instantiated and used
    mock_client_class.assert_called_once()
    mock_instance.upload.assert_called_once()
    assert result.success is True
```

## 6. Error Wrapping Pattern

Proper exception chaining with context.

```python
class DataProcessingError(Exception):
    """Domain-specific error for data processing failures."""
    pass

class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

def process_file(file_path: str, config: dict) -> ProcessingResult:
    """
    Process file with proper error handling and chaining.

    Args:
        file_path: Path to input file
        config: Processing configuration

    Returns:
        Processing result

    Raises:
        DataProcessingError: If processing fails
        ValidationError: If validation fails
    """
    # File I/O errors
    try:
        with open(file_path) as f:
            data = f.read()
    except FileNotFoundError as e:
        raise DataProcessingError(
            f"Input file not found: {file_path}"
        ) from e  # ✅ Chain with 'from e'
    except IOError as e:
        raise DataProcessingError(
            f"Failed to read {file_path}: {e}"
        ) from e

    # Validation errors
    try:
        validated_data = validate_data(data, config)
    except ValueError as e:
        raise ValidationError(
            f"Invalid data format in {file_path}: {e}"
        ) from e

    # Processing errors with context
    try:
        result = parse_data(validated_data)
    except Exception as e:
        raise DataProcessingError(
            f"Failed to process {file_path} with config {config}: {e}"
        ) from e

    return result

# Usage example
try:
    result = process_file("data.csv", {"threshold": 0.05})
except DataProcessingError as e:
    # Error includes full chain: DataProcessingError -> FileNotFoundError
    print(f"Processing failed: {e}")
    print(f"Original error: {e.__cause__}")  # Access chained exception
except ValidationError as e:
    print(f"Validation failed: {e}")
```

---

**Usage Notes:**

- All patterns pass `poetry run pyright` and `poetry run ruff check`
- Copy these patterns when implementing similar functionality
- Adapt to specific use case while maintaining the core structure
- See `.claude/guidelines/rules.md` for detailed requirements on each pattern
- See `.claude/guidelines/tldr/coding_principles.tldr.md` for Function-Op-Asset Trinity details

**Pattern Priority:**

1. **Dagster Asset with Proper Tagging** - MOST CRITICAL for new assets
2. **Polars Transform with Pandera Validation** - For all DataFrame operations
3. **Deterministic pytest Example** - For all new tests
4. **scikit-learn Pipeline** - For ML workflows
5. **Pydantic Settings** - For configuration management
6. **Error Wrapping** - For robust error handling
