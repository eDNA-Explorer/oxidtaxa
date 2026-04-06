# eDNA Explorer Architecture Map

Quick reference for navigating the eDNA Explorer monorepo data pipelines.

## Monorepo Structure

```
edna-explorer-data-pipelines/
├── 📦 projects/                     # Deployable applications
│   ├── edna_dagster_pipelines/      # Core data processing orchestration
│   ├── edna_explorer_reports/       # Report generation web service
│   ├── feature_importance/          # ML feature importance analysis
│   ├── report_api/                  # Report API service
│   └── common/                      # Common project utilities
│
├── 📚 libraries/                    # Shared reusable libraries
│   ├── edna-db-lib/                 # Database models and access
│   ├── core-analysis-lib/           # Core analysis algorithms
│   ├── logging-lib/                 # Structured logging utilities
│   └── adk-lib/                     # Agent Development Kit
│
├── 🛠️  dev/                          # Development utilities
├── 📖 docs/                         # Documentation
├── 🧪 specs/                        # Technical specifications
├── 🐳 docker/                       # Docker configurations
├── 🚀 deployment/                   # K8s/ArgoCD configs
├── 🗄️  storage/                     # Local Dagster storage
├── 💭 thoughts/                     # Planning & research docs
├── 🤖 agent/                        # AI agent knowledge base
│   ├── tldr/                        # Quick reference docs
│   └── examples/                    # Code examples
│
├── pyproject.toml                   # Root workspace config
├── workspace.yaml                   # Dagster workspace config
└── compose.yml                      # Docker Compose orchestration
```

## Dagster Code Organization

The `edna_dagster_pipelines` follows the **Function-Op-CLI Trinity** pattern for all new code:

```
projects/edna_dagster_pipelines/src/edna_dagster_pipelines/
├── 🧠 core/{module}/                # Pure business logic (Dagster-free)
│   ├── database/                    # Database operations
│   ├── monitoring/                  # Monitoring logic
│   └── qcassign/                    # QC assignment logic
│
├── ⚙️  ops/{module}/                 # Thin Dagster wrappers (LEGACY - in jobs/assets)
│
├── 📊 assets/                       # Dagster software-defined assets
│   ├── feature_importance/
│   ├── genbank/
│   ├── monitoring/
│   ├── pregen_reports/
│   ├── primer_db/
│   ├── project/                     # Project-specific assets
│   │   └── ops/                     # Asset-specific ops
│   └── taxonomy/
│
├── 🔧 jobs/                         # Dagster job definitions
│   ├── anacapa_qc/
│   ├── metadata/
│   ├── terradactyl/
│   ├── asv_generation_job.py
│   ├── calculate_tronko_metrics_job.py
│   ├── combine_tronko_results_job.py
│   └── create_sample_stubs_job.py
│
├── 📡 sensors/                      # Dagster sensors (trigger jobs on events)
│   ├── metadata_sensor.py
│   ├── qcinit_sensor.py
│   ├── qcassign_sensor.py
│   ├── terradactyl_sensor.py
│   ├── tronko_input_sensor.py
│   ├── calculate_tronko_metrics_sensor.py
│   └── combine_tronko_results_sensor.py
│
├── 🔌 resources/                    # Dagster resources (GCS, DB, etc.)
│   ├── __init__.py                  # Resource definitions
│   └── js2.py                       # Jetstream S3 client
│
├── 🗄️  db/                          # Database utilities
├── 🔄 dendra/                       # Dendra integration
├── 🎨 formatters/                   # Data formatters
├── 🔬 qcassign/                     # QC assignment module
│   ├── ops/
│   └── helpers/
│
├── constants.py                     # Path templates & constants
└── definitions.py                   # Main Dagster definitions (ENTRYPOINT)
```

## Dagster Code Locations

**Single Code Location:** `edna-pipelines`
- **Entrypoint:** `projects/edna_dagster_pipelines/src/edna_dagster_pipelines/definitions.py`
- **Workspace:** `workspace.yaml` (root)
- **Location:** All jobs, assets, sensors, schedules are defined in `definitions.py`

## Key Entrypoints

### Dagster Orchestration
- **Main Definitions:** `edna_dagster_pipelines/definitions.py`
  - All jobs (`all_jobs`)
  - All assets (`all_assets`)
  - All sensors (`all_sensors`)
  - All schedules (`all_schedules`)
  - Resources (GCS, secrets, JS2)

### Jobs (Orchestrated Workflows)
Located in `edna_dagster_pipelines/jobs/`:
- `qc_init_job` - Initialize QC process
- `qcassign_job` - Assign taxonomy to sequences
- `asv_generation_job` - Generate ASV tables
- `calculate_tronko_metrics_job` - Calculate Tronko metrics
- `combine_tronko_results_job` - Combine Tronko results
- `terradactyl_job` - Terradactyl processing
- `metadata_job` - Metadata processing
- `project_list_job` - Update project lists
- `genbank_wgs_job` - GenBank WGS processing

### Sensors (Event Triggers)
Located in `edna_dagster_pipelines/sensors/`:
- Monitor GCS bucket paths for `.run` files
- Trigger corresponding jobs when files appear
- Pattern: `queue/{module}/{identifier}.run`

### Schedules (Time-based Triggers)
- `weekly_genbank_schedule` - Weekly GenBank updates
- `project_list_schedule` - Periodic project list refresh
- `failed_jobs_monitoring_schedule` - Monitor failed jobs
- `taxonomy_schedules` - Taxonomy updates

### CLI Interfaces
**Note:** CLI directory not yet implemented. Legacy scripts exist as standalone Python files.

## Config & Secrets Flow

### Environment Variables
Set via `.env` (local) or Kubernetes secrets (deployed):
- `STAGE` - Deployment stage (staging/production/test)
- `GCS_PROJECT` - GCP project ID
- `DAGSTER_BUCKET` - Dagster-specific GCS bucket
- `PROJECT_BUCKET` - Project files GCS bucket
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account key

### Resource Configuration
Defined in `resources/__init__.py`:

```python
env_config = {
    "STAGE": "staging",
    "DAGSTER_BUCKET": "edna-dagster-{stage}",
    "PROJECT_BUCKET": "edna-project-files-{stage}",
    "GCS_PROJECT": "edna-explorer-web-services"
}
```

### GCP Secrets Manager
- Database credentials
- Jetstream S3 credentials (JS2)
- Secret pattern: `projects/350789020267/secrets/{stage}-edna-explorer-{service}/versions/latest`

### Test Environment
- Automatically mocks GCS, secrets manager, and JS2 when pytest is detected
- Uses `MockGCSResource`, `MockSecretsManager`, `MockJS2S3Resource`

## Storage Layout

### GCS Buckets (Pattern-based)

**Dagster Bucket:** `edna-dagster-{stage}`
- Prefix: `edna_dagster_pipelines-{stage}/`
- Stores: Dagster run artifacts, IO manager data

**Project Files Bucket:** `edna-project-files-{stage}`
- Projects: `projects/{project_id}/`
- Metadata: `projects/{project_id}/metadata/{marker}/`
  - `user-uploaded/latest.csv` - Raw user metadata
  - `processed/latest.csv` - Processed metadata
  - `geospatial.csv` - Geospatial enriched metadata
- ASV Data: `projects/{project_id}/assign/{marker}/`
  - `processed.asv` - Final ASV assignments
  - `intermediary/combined/{run_id}/` - Intermediate files
- Cache: `projects/{project_id}/cache/{marker}/`
  - `preprocessed/{variant}/` - Tronko preprocessed data
  - `range/range.csv.zst` - iNaturalist range cache
- Queue: `queue/{module}/` - Sensor trigger files
  - `queue/metabarcoding/qcinit/{project}.run`
  - `queue/metabarcoding/qcassign/{project}/{marker}.json`
  - `queue/geospatial/*.run`

### BigQuery Datasets
Pattern: `{stage}_edna_explorer`
- Accessed via `edna-db-lib` ORM models
- Contains project metadata, samples, ASVs, taxonomy

### Local Development Storage
- `./storage/` - Local Dagster storage (runs, schedules, sensors)
- `./cache/` - Local cache directory
  - `cache/analysis/` - Analysis cache
  - `cache/projects/` - Project-specific cache
  - `cache/spatial/` - Spatial data cache

## Typical Pipeline Flow

### 1. Ingestion: QC & Sequence Processing

```
User Upload → Sensor Detects → QC Init Job
                                    ↓
                          Validate Metadata
                                    ↓
                          QC Sample Processing
                                    ↓
                          QCAssign (Taxonomy)
                                    ↓
                          ASV Generation
                                    ↓
                          Store in BigQuery
```

**Key Files:**
- Sensors: `qcinit_sensor.py`, `qc_sample_sensor.py`, `qcassign_sensor.py`
- Jobs: `qc_init_job`, `qcassign_job`, `asv_generation_job`
- Storage: `queue/metabarcoding/` → `projects/{id}/assign/{marker}/`

### 2. Processing: Tronko Analysis

```
ASV Data Ready → Tronko Input Sensor
                        ↓
              Preprocess Data (Asset)
                        ↓
              Calculate Metrics Job
                        ↓
              Combine Results Job
                        ↓
              Tronko Output (Asset)
```

**Key Files:**
- Sensors: `tronko_input_sensor.py`, `calculate_tronko_metrics_sensor.py`
- Jobs: `calculate_tronko_metrics_job`, `combine_tronko_results_job`
- Assets: `tronko_preprocessed_data`, `tronko_inputs`, `tronko_output`
- Cache: `projects/{id}/cache/{marker}/preprocessed/`

### 3. Output: Reports & Analysis

```
Processed Data → Project Assets Update
                        ↓
              Feature Importance (ML)
                        ↓
              Pregen Reports (Asset)
                        ↓
              Report API Service
                        ↓
              User-facing Reports
```

**Key Components:**
- Assets: `project_samples`, `project_markers`, `feature_importance_asset`, `project_pregen_reports`
- Service: `projects/edna_explorer_reports/` (Flask app)
- API: `projects/report_api/` (FastAPI)

### 4. Enrichment: External Data

```
Schedule Trigger → GenBank Job (Weekly)
                        ↓
                Fetch WGS Data
                        ↓
                Update Taxonomy
                        ↓
                Refresh Assets

Metadata Upload → Terradactyl Sensor
                        ↓
                Geospatial Enrichment
                        ↓
                Update Coordinates
```

**Key Files:**
- Jobs: `genbank_wgs_job`, `terradactyl_job`, `metadata_job`
- Sensors: `terradactyl_sensor.py`, `metadata_sensor.py`
- Assets: `clustered_boundaries`, `coordinate_list`, `taxa_nation`, `taxa_state`

## Shared Libraries

### edna-db-lib
- SQLAlchemy ORM models
- Database session management
- Query utilities
- **Location:** `libraries/edna-db-lib/src/edna_db/`

### core-analysis-lib
- Feature importance analysis
- ML model utilities
- Statistical analysis
- **Location:** `libraries/core-analysis-lib/src/core_analysis/`

### logging-lib
- Structured logging
- Logger interfaces
- Logging utilities
- **Location:** `libraries/logging-lib/src/logging_lib/`

### adk-lib
- Agent Development Kit
- Tools for AI agents
- Knowledge base utilities
- **Location:** `libraries/adk-lib/src/adk/`

## Development Quick Reference

### Run Dagster Locally
```bash
# Via Docker Compose
docker compose up edna_dagster_pipelines

# Direct (requires local setup)
cd projects/edna_dagster_pipelines/src
dagster dev -h 0.0.0.0 -p 3003
```

### Run Tests
```bash
poetry run pytest                                      # All tests
poetry run pytest projects/edna_dagster_pipelines/tests/  # Dagster only
```

### Code Quality
```bash
poetry run ruff format .     # Format
poetry run ruff check .      # Lint
poetry run pyright           # Type check
```

### Generate New Function
```bash
python scripts/generate_dual_execution_template.py {function_name} {module_name}
```

## Architecture Patterns

### Dual-Execution (Function-Op-CLI Trinity)
**Required for all new code** - see `.claude/guidelines/coding_principles.md`

1. **Core Function** (`core/{module}/`) - Pure business logic
2. **Dagster Op** (`ops/{module}/` or in jobs/assets) - Thin wrapper
3. **CLI** (not yet implemented) - Debug interface

### Asset-based Design
- Software-defined assets represent data artifacts
- Dagster manages dependencies and materialization
- Located in `assets/{domain}/`

### Sensor-driven Execution
- Sensors monitor GCS paths for trigger files (`.run`, `.json`)
- Pattern: `queue/{module}/{identifier}.run`
- Decouple event detection from execution

## Key Constants & Patterns

From `constants.py`:
- Queue paths: `queue/{module}/*.run`
- Metadata: `projects/{id}/metadata/{marker}/{type}/latest.csv`
- ASV data: `projects/{id}/assign/{marker}/processed.asv`
- Cache: `projects/{id}/cache/{marker}/{type}/`

## Navigation Tips

- **Find a job:** `edna_dagster_pipelines/jobs/`
- **Find an asset:** `edna_dagster_pipelines/assets/{domain}/`
- **Find business logic:** `edna_dagster_pipelines/core/{module}/`
- **Find a sensor:** `edna_dagster_pipelines/sensors/`
- **Find DB models:** `libraries/edna-db-lib/src/edna_db/models/`
- **Find analysis code:** `libraries/core-analysis-lib/src/core_analysis/`
- **Find deployment config:** `deployment/{project}/{stage}/`
- **Find documentation:** `docs/` or `.claude/guidelines/tldr/`
