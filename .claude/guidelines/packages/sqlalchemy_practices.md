
## SQLAlchemy Session Management

All database operations use the `edna_db.session.db_session` context manager, which automatically commits and closes sessions. SQLAlchemy ORM objects become **detached** when the session closes, making it unsafe to access their attributes outside the session context.

### Safe Patterns

#### Pattern 1: Extract Scalar Values Within Session (Recommended for Simple Cases)

Extract only the specific column values you need while the session is active:

```python
from edna_db.session import db_session
from edna_db.schema import ProjectFastqFile

# Extract values WITHIN session
with db_session(env_config["STAGE"], gcp_secrets_manager) as session:
    fastq_file = session.query(ProjectFastqFile).filter_by(id=file_id).first()

    if not fastq_file:
        raise ValueError(f"File {file_id} not found")

    # Extract scalar attributes to local variables
    project_id = fastq_file.projectId
    file_name = fastq_file.fileName
    checksum = fastq_file.md5CheckSum

# Use extracted values OUTSIDE session
gcs_key = f"projects/{project_id}/samples/{file_name}"
```

**When to use**: When you only need a few column values (strings, ints, bools, timestamps).

**Example**: `qcassign/ops/verify_sample_file.py:50-61`, `qcassign/ops/qc_process_sample_for_run.py:128-154`

#### Pattern 2: Convert to Dictionary Within Session (Recommended for Multiple Attributes)

Use the `fastq_file_to_dict()` helper to convert model instances to plain dictionaries:

```python
from edna_db.session import db_session
from edna_db.schema import ProjectFastqFile
from edna_dagster_pipelines.qcassign.helpers.fastq_to_dict import fastq_file_to_dict

with db_session(env_config["STAGE"], gcp_secrets_manager) as session:
    fastq_file = session.query(ProjectFastqFile).filter_by(id=file_id).first()

    # Convert to dict WITHIN session
    file_data = fastq_file_to_dict(fastq_file)

# Use plain dict OUTSIDE session
file_name = file_data["fileName"]
project_id = file_data["projectId"]
```

**When to use**: When you need many attributes from a model, or when passing data between ops.

**Example**: `qcassign/helpers/fetch_paired_sample.py:36-49`

#### Pattern 3: Access Attributes in Loop After .all() Query

Query with `.all()` to load objects into memory, then iterate and access column attributes:

```python
from edna_db.session import db_session
from edna_db.schema import ProjectMarker

with db_session(env_config["STAGE"], gcp_secrets_manager) as session:
    markers = (
        session.query(ProjectMarker)
        .filter(ProjectMarker.projectId == project_id)
        .all()
    )

# Objects are detached but column attributes are cached
for marker in markers:
    # Accessing simple column attributes works
    name = marker.name
    forward_seq = marker.forwardSequence
    reverse_seq = marker.reverseSequence
```

**When to use**: When processing multiple records in a loop, accessing only column attributes.

**Important**: Only simple column attributes work after `.all()`. Relationships (like `marker.Project_`) will fail.

**Example**: `qcassign/ops/qc_process_sample_for_run.py:315-361`

### Unsafe Patterns (DO NOT USE)

#### ❌ Accessing Attributes After Session Closes

```python
# WRONG - DetachedInstanceError will occur
with db_session(env_config["STAGE"], gcp_secrets_manager) as session:
    fastq_file = session.query(ProjectFastqFile).filter_by(id=file_id).first()
    # Session closes here

# ERROR: Accessing attribute outside session
file_name = fastq_file.fileName  # DetachedInstanceError!
```

#### ❌ Accessing Relationships After Session Closes

```python
# WRONG - Even worse, relationships always fail
with db_session(env_config["STAGE"], gcp_secrets_manager) as session:
    fastq_file = session.query(ProjectFastqFile).filter_by(id=file_id).first()
    # Session closes here

# ERROR: Accessing relationship triggers new query
project = fastq_file.Project_  # DetachedInstanceError!
```

### Why This Happens

The `db_session` context manager uses SQLAlchemy's default `expire_on_commit=True` setting:
1. `session.commit()` marks all objects as expired
2. `session.close()` detaches the session from the database
3. Accessing attributes on expired objects attempts to refresh from database
4. Refresh fails because session is closed → `DetachedInstanceError`

### Quick Reference

| Pattern | Use Case | Safety | Example File |
|---------|----------|--------|--------------|
| Extract scalars within session | Need 1-3 values | ✅ Safe | `verify_sample_file.py:50-61` |
| Convert to dict within session | Need many values | ✅ Safe | `fetch_paired_sample.py:36-49` |
| Loop after `.all()` query | Process multiple records | ✅ Safe (columns only) | `qc_process_sample_for_run.py:315-361` |
| Access after session closes | Any use | ❌ Unsafe | N/A |
| Access relationships outside session | Any use | ❌ Unsafe | N/A |

### Related Files

- Session context manager: `libraries/edna-db-lib/src/edna_db/session.py:179-214`
- Dict conversion helper: `projects/edna_dagster_pipelines/src/edna_dagster_pipelines/qcassign/helpers/fastq_to_dict.py`
- Research document: `thoughts/shared/research/2025-10-23-sqlalchemy-detached-instance-error.md`