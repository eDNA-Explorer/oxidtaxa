---
date: 2025-10-10T19:20:26+0000
researcher: Jim Jeffers
git_commit: d0d763d94ac0c0b125905b8703ad21a7a32b713c
branch: feature/agent-knowledge-kit
repository: edna-explorer-data-pipelines
topic: "Dagster Tagging Strategy Guidelines for eDNA Explorer"
tags: [research, dagster, tagging, best-practices, asset-organization]
status: complete
last_updated: 2025-10-10
last_updated_by: Jim Jeffers
---

# Dagster Tagging Strategy Guidelines for eDNA Explorer Data Pipelines

**Date**: 2025-10-10T19:20:26+0000
**Researcher**: Jim Jeffers
**Git Commit**: d0d763d94ac0c0b125905b8703ad21a7a32b713c
**Branch**: feature/agent-knowledge-kit
**Repository**: edna-explorer-data-pipelines

## Purpose

This document establishes a comprehensive tagging strategy for the eDNA Explorer Dagster project. It is based on Dagster best practices and adapted to the specific organizational patterns and needs of the eDNA Explorer data platform.

## Current State Assessment

### Existing Tag Usage

The codebase currently has **minimal tag usage** for categorization and automation:

**✅ What exists:**
- `op_tags` parameter used on `@asset` decorators for operational concerns
- Kubernetes resource configuration via `dagster-k8s/config` and `dagster-k8s/resource_requirements`
- Concurrency control via `dagster/concurrency_key`
- Assets organized by `group_name` (7 distinct groups: `genbank_wgs`, `monitoring`, `taxonomy`, `project_tronko`, `project_gbif`, `project_geo`, `project_metadata`, `global_assets`)

**❌ What's missing:**
- No use of `tags` parameter on `@asset` decorators for categorization
- No tag-based asset selection in job definitions
- No domain, tier, or sensitivity classification via tags
- No kind tags for visual identification in the UI
- No policy-driven automation using tag-based selection

**Current job selection pattern:**
```python
# Direct asset list selection (current approach)
project_metadata_job = define_asset_job(
    name="project_metadata_job",
    selection=["project_markers", "project_samples"],
)
```

### Code References
- Current tag usage examples: `edna_dagster_pipelines/assets/genbank/wgs_download_links.py:39-49`
- Concurrency control: `edna_dagster_pipelines/assets/project/taxa_state.py:25`
- K8s configuration: `edna_dagster_pipelines/assets/taxonomy/taxonomy_ingestion.py:475-484`

## Recommended Tagging Strategy

### Three-Tiered Tag Framework

Following Dagster best practices, we recommend a three-tiered approach to tagging:

1. **Organization & Discovery** - Help humans find and understand assets
2. **Automation** - Enable policy-driven job and sensor definitions
3. **Execution Control** - Manage priority, concurrency, and backfill behavior

### Tier 1: Organization & Discovery Tags

These tags help categorize assets for human understanding and navigation.

#### Recommended Tag Taxonomy

| Tag Key | Purpose | Allowed Values | Example |
|---------|---------|----------------|---------|
| `domain` | Business/scientific domain | `genomics`, `taxonomy`, `biodiversity`, `quality_control`, `reporting`, `monitoring` | `{"domain": "taxonomy"}` |
| `data_tier` | Data processing stage (medallion architecture) | `bronze`, `silver`, `gold` | `{"data_tier": "bronze"}` |
| `source` | External data source | `genbank`, `gbif`, `ncbi`, `inaturalist`, `user_upload` | `{"source": "gbif"}` |
| `sensitivity` | Data sensitivity level | `public`, `internal`, `restricted` | `{"sensitivity": "public"}` |
| `pipeline_stage` | Processing stage | `ingestion`, `validation`, `transformation`, `enrichment`, `aggregation`, `export` | `{"pipeline_stage": "ingestion"}` |
| `tool` | Primary processing tool | `python`, `dbt`, `r`, `spark`, `custom` | `{"tool": "python"}` |
| `compute_intensity` | Resource requirements | `light`, `medium`, `heavy` | `{"compute_intensity": "heavy"}` |
| `update_frequency` | Expected refresh rate | `realtime`, `hourly`, `daily`, `weekly`, `monthly`, `adhoc` | `{"update_frequency": "daily"}` |

#### Implementation Example

```python
from dagster import asset, AssetKey

@asset(
    key=AssetKey(["reference_taxonomies"]),
    group_name="taxonomy",
    tags={
        "domain": "taxonomy",
        "data_tier": "bronze",
        "source": "ncbi",
        "sensitivity": "public",
        "pipeline_stage": "ingestion",
        "tool": "python",
        "compute_intensity": "medium",
        "update_frequency": "daily",
    },
    op_tags={  # Keep operational tags separate
        "dagster-k8s/config": {
            "pod_template_spec_metadata": {
                "labels": {"duration": "extended"},
            }
        }
    },
)
def reference_taxonomies(context, ...):
    """Discover and ingest reference taxonomy data from GCS to BigQuery"""
    # ... implementation
```

**Key principle:** Use `tags` for categorization, `op_tags` for operational configuration.

### Tier 2: Automation Tags

Tags that enable policy-driven orchestration through Dagster's Asset Selection Syntax.

#### Priority-Based Automation

Create tags that identify critical vs. non-critical workloads:

```python
# Critical production asset
@asset(
    name="project_samples",
    group_name="project_metadata",
    tags={
        "domain": "biodiversity",
        "priority": "critical",
        "sla": "15min",
        "team": "platform",
    },
)
def project_samples(context, ...):
    # ... implementation

# Lower priority backfill job
@asset(
    name="historical_gbif_data",
    group_name="project_gbif",
    tags={
        "domain": "biodiversity",
        "priority": "low",
        "team": "analytics",
    },
)
def historical_gbif_data(context, ...):
    # ... implementation
```

#### Tag-Based Job Definitions

Migrate from explicit asset lists to tag-based selection:

**Before (current approach):**
```python
project_metadata_job = define_asset_job(
    name="project_metadata_job",
    selection=["project_markers", "project_samples"],
)
```

**After (tag-based approach):**
```python
from dagster import AssetSelection, define_asset_job

# Select all assets with specific tag
project_metadata_job = define_asset_job(
    name="project_metadata_job",
    selection=AssetSelection.tag("domain", "biodiversity").tag("pipeline_stage", "ingestion"),
)

# Select all critical assets
critical_assets_job = define_asset_job(
    name="critical_assets_job",
    selection=AssetSelection.tag("priority", "critical"),
)

# Select all GBIF-related assets
gbif_pipeline_job = define_asset_job(
    name="gbif_pipeline_job",
    selection=AssetSelection.tag("source", "gbif"),
)

# Select all bronze tier ingestion assets
bronze_ingestion_job = define_asset_job(
    name="bronze_ingestion_job",
    selection=AssetSelection.tag("data_tier", "bronze").tag("pipeline_stage", "ingestion"),
)
```

**Benefits:**
- New assets automatically included in relevant jobs when tagged correctly
- Reduces coupling between asset definitions and job definitions
- Enables self-service: data producers can add assets without modifying job code
- Easier to understand job scope from its selection query

#### Tag-Based Sensor Targeting

```python
from dagster import AutomationConditionSensorDefinition, AssetSelection

# Sensor that only evaluates high-priority assets
high_priority_sensor = AutomationConditionSensorDefinition(
    name="high_priority_automation",
    target=AssetSelection.tag("priority", "critical").tag("priority", "high"),
    minimum_interval_seconds=60,
)

# Sensor for specific domain
taxonomy_sensor = AutomationConditionSensorDefinition(
    name="taxonomy_automation",
    target=AssetSelection.tag("domain", "taxonomy"),
    minimum_interval_seconds=300,
)
```

### Tier 3: Execution Control Tags

Special tags that control run execution behavior.

#### Priority Tags

Control execution order in the run queue:

```python
from dagster import define_asset_job, ScheduleDefinition

# High-priority production job
critical_job = define_asset_job(
    name="critical_updates",
    selection=AssetSelection.tag("priority", "critical"),
    tags={"dagster/priority": "10"},  # High priority
)

# Low-priority backfill job
backfill_job = define_asset_job(
    name="historical_backfill",
    selection=AssetSelection.tag("pipeline_stage", "backfill"),
    tags={"dagster/priority": "-5"},  # Low priority, won't block production
)

# Schedule with priority
daily_taxonomy_schedule = ScheduleDefinition(
    name="daily_taxonomy_ingestion",
    job=taxonomy_ingestion_job,
    cron_schedule="0 0 * * *",
    tags={"dagster/priority": "5"},  # Medium-high priority
)
```

#### Concurrency Tags

Migrate existing concurrency control to be more semantic:

**Before (current approach):**
```python
@asset(
    op_tags={"dagster/concurrency_key": "materialize_taxa_gbif"}
)
def taxa_state(context, ...):
    # ... implementation
```

**After (enhanced approach):**
```python
@asset(
    tags={
        "domain": "biodiversity",
        "source": "gbif",
        "api": "gbif_occurrences",  # Used for concurrency grouping
    },
    op_tags={
        "dagster/concurrency_key": "gbif_api_limit",  # More semantic name
    }
)
def taxa_state(context, ...):
    # ... implementation
```

Then configure in `dagster.yaml`:
```yaml
run_queue:
  tag_concurrency_limits:
    - key: "api"
      value: "gbif_occurrences"
      limit: 3
```

#### Backfill Tags

Tag backfill runs for tracking and priority management:

```bash
# CLI backfill with tags
dagster job backfill \
  --job historical_gbif_data \
  --tags '{"dagster/priority": "-5", "backfill_type": "historical", "initiated_by": "analytics_team"}'
```

### Kind Tags for Visual Identification

Use `compute_kind` parameter (distinct from `tags`) to display icons in the Dagster UI:

```python
@asset(
    compute_kind="python",  # Shows Python icon
    tags={"tool": "python"},  # Semantic tag for selection
)
def process_sequences(context, ...):
    # ... implementation

@asset(
    compute_kind="dbt",  # Shows dbt icon
    tags={"tool": "dbt"},
)
def dbt_transform(context, ...):
    # ... implementation

@asset(
    compute_kind="sql",  # Shows SQL icon
    tags={"tool": "sql", "source": "bigquery"},
)
def bigquery_aggregation(context, ...):
    # ... implementation
```

**Available kind tags:** `python`, `dbt`, `sql`, `snowflake`, `spark`, `jupyter`, `dagstermill`, etc.

## Proposed Tag Assignments by Asset Group

Based on the current 7 asset groups, here are recommended tag assignments:

### Group: `genbank_wgs`

```python
# Common tags for all GenBank assets
GENBANK_COMMON_TAGS = {
    "domain": "genomics",
    "data_tier": "bronze",
    "source": "genbank",
    "sensitivity": "public",
    "pipeline_stage": "ingestion",
    "tool": "python",
}

@asset(
    key=AssetKey(["wgs_selector_csv"]),
    group_name="genbank_wgs",
    tags={
        **GENBANK_COMMON_TAGS,
        "update_frequency": "weekly",
        "compute_intensity": "light",
    },
)
def wgs_selector_csv(context, ...): ...

@asset(
    key=AssetKey(["wgs_download_links"]),
    group_name="genbank_wgs",
    tags={
        **GENBANK_COMMON_TAGS,
        "update_frequency": "weekly",
        "compute_intensity": "heavy",
    },
    op_tags={
        "dagster/concurrency_key": "genbank_api_limit",
        "dagster-k8s/resource_requirements": {
            "requests": {"cpu": "4", "memory": "16Gi"},
            "limits": {"cpu": "4", "memory": "32Gi"},
        },
    },
)
def wgs_download_links(context, ...): ...
```

### Group: `taxonomy`

```python
@asset(
    key=AssetKey(["reference_taxonomies"]),
    group_name="taxonomy",
    tags={
        "domain": "taxonomy",
        "data_tier": "bronze",
        "source": "ncbi",
        "sensitivity": "public",
        "pipeline_stage": "ingestion",
        "tool": "python",
        "update_frequency": "daily",
        "compute_intensity": "medium",
        "priority": "high",
    },
    op_tags={
        "dagster-k8s/config": {
            "pod_template_spec_metadata": {
                "labels": {"duration": "extended"},
            }
        }
    },
)
def reference_taxonomies(context, ...): ...
```

### Group: `project_metadata`

```python
PROJECT_METADATA_COMMON = {
    "domain": "biodiversity",
    "data_tier": "bronze",
    "source": "user_upload",
    "sensitivity": "internal",
    "tool": "python",
    "priority": "critical",
}

@asset(
    key=AssetKey(["project_samples"]),
    group_name="project_metadata",
    partitions_def=project_partitions,
    tags={
        **PROJECT_METADATA_COMMON,
        "pipeline_stage": "ingestion",
        "update_frequency": "adhoc",
        "compute_intensity": "light",
        "sla": "15min",
    },
)
def project_samples(context, ...): ...

@asset(
    key=AssetKey(["project_markers"]),
    group_name="project_metadata",
    partitions_def=project_partitions,
    tags={
        **PROJECT_METADATA_COMMON,
        "pipeline_stage": "validation",
        "update_frequency": "adhoc",
        "compute_intensity": "light",
    },
)
def project_markers(context, ...): ...
```

### Group: `project_gbif`

```python
GBIF_COMMON = {
    "domain": "biodiversity",
    "data_tier": "silver",
    "source": "gbif",
    "sensitivity": "public",
    "pipeline_stage": "enrichment",
    "tool": "python",
    "api": "gbif_occurrences",
}

@asset(
    key=AssetKey(["taxa_state"]),
    group_name="project_gbif",
    partitions_def=project_partitions,
    tags={
        **GBIF_COMMON,
        "update_frequency": "daily",
        "compute_intensity": "medium",
    },
    op_tags={
        "dagster/concurrency_key": "gbif_api_limit",
    },
)
def taxa_state(context, ...): ...
```

### Group: `project_tronko`

```python
TRONKO_COMMON = {
    "domain": "taxonomy",
    "data_tier": "gold",
    "tool": "python",
    "sensitivity": "internal",
}

@asset(
    key=AssetKey(["tronko_output"]),
    group_name="project_tronko",
    partitions_def=project_partitions,
    tags={
        **TRONKO_COMMON,
        "pipeline_stage": "aggregation",
        "update_frequency": "adhoc",
        "compute_intensity": "heavy",
        "priority": "high",
    },
)
def tronko_output(context, ...): ...

@asset(
    key=AssetKey(["out_of_range_taxa"]),
    group_name="project_tronko",
    partitions_def=project_partitions,
    tags={
        **TRONKO_COMMON,
        "pipeline_stage": "validation",
        "source": "inaturalist",
        "update_frequency": "daily",
        "compute_intensity": "medium",
    },
    op_tags={
        "dagster-k8s/resource_requirements": {
            "requests": {"cpu": "4", "memory": "8Gi"},
            "limits": {"cpu": "4", "memory": "16Gi"},
        },
    },
)
def out_of_range_taxa(context, ...): ...
```

### Group: `monitoring`

```python
@asset(
    key=AssetKey(["failed_jobs_report"]),
    group_name="monitoring",
    tags={
        "domain": "monitoring",
        "pipeline_stage": "export",
        "tool": "python",
        "sensitivity": "internal",
        "update_frequency": "hourly",
        "compute_intensity": "light",
        "priority": "medium",
    },
)
def failed_jobs_report(context, ...): ...
```

### Group: `global_assets`

```python
@asset(
    key=AssetKey(["project_list"]),
    group_name="global_assets",
    tags={
        "domain": "biodiversity",
        "data_tier": "bronze",
        "source": "internal",
        "sensitivity": "internal",
        "pipeline_stage": "ingestion",
        "tool": "python",
        "update_frequency": "realtime",
        "compute_intensity": "light",
        "priority": "critical",
        "sla": "15min",
    },
)
def project_list(context, ...): ...

@asset(
    key=AssetKey(["primer_table"]),
    group_name="global_assets",
    tags={
        "domain": "genomics",
        "data_tier": "bronze",
        "source": "internal",
        "sensitivity": "internal",
        "pipeline_stage": "ingestion",
        "tool": "python",
        "update_frequency": "adhoc",
        "compute_intensity": "light",
        "priority": "high",
    },
)
def primer_table(context, ...): ...
```

## Migration Strategy

### Phase 1: Add Tags to Existing Assets (Non-Breaking)

1. **Start with high-value assets** (critical path, frequently modified)
2. **Add tags without removing existing configurations**
3. **Use shared tag dictionaries** to ensure consistency

```python
# Step 1: Define shared tag dictionaries at module level
# edna_dagster_pipelines/assets/taxonomy/common_tags.py
TAXONOMY_BASE_TAGS = {
    "domain": "taxonomy",
    "data_tier": "bronze",
    "tool": "python",
    "sensitivity": "public",
}

# Step 2: Apply to assets
from edna_dagster_pipelines.assets.taxonomy.common_tags import TAXONOMY_BASE_TAGS

@asset(
    tags={
        **TAXONOMY_BASE_TAGS,
        "source": "ncbi",
        "pipeline_stage": "ingestion",
        "update_frequency": "daily",
    },
    op_tags={...},  # Keep existing op_tags unchanged
)
def reference_taxonomies(context, ...): ...
```

### Phase 2: Migrate Jobs to Tag-Based Selection

1. **Create tag-based job definitions alongside existing ones**
2. **Test in development environment**
3. **Switch over once validated**

```python
# Before
project_metadata_job = define_asset_job(
    name="project_metadata_job",
    selection=["project_markers", "project_samples"],
)

# After - run both in parallel during migration
project_metadata_job_v2 = define_asset_job(
    name="project_metadata_job_v2",
    selection=AssetSelection.tag("domain", "biodiversity").tag("pipeline_stage", "ingestion"),
)

# Once validated, deprecate v1 and rename v2
```

### Phase 3: Implement Policy-Based Automation

1. **Define automation sensors using tag-based targeting**
2. **Configure concurrency limits using semantic tags**
3. **Establish priority-based scheduling**

### Phase 4: Documentation and Team Onboarding

1. **Document tag taxonomy** in project README
2. **Create PR checklist** requiring tags on new assets
3. **Add linting** to validate tag presence and values

## Tag Validation and Governance

### Validation Schema

Create a centralized tag schema for validation:

```python
# edna_dagster_pipelines/config/tag_schema.py
from typing import Literal

ALLOWED_DOMAINS = Literal[
    "genomics", "taxonomy", "biodiversity", "quality_control", "reporting", "monitoring"
]
ALLOWED_DATA_TIERS = Literal["bronze", "silver", "gold"]
ALLOWED_SOURCES = Literal[
    "genbank", "gbif", "ncbi", "inaturalist", "user_upload", "internal"
]
ALLOWED_SENSITIVITIES = Literal["public", "internal", "restricted"]
ALLOWED_PIPELINE_STAGES = Literal[
    "ingestion", "validation", "transformation", "enrichment", "aggregation", "export"
]
ALLOWED_TOOLS = Literal["python", "dbt", "r", "spark", "custom"]
ALLOWED_COMPUTE_INTENSITIES = Literal["light", "medium", "heavy"]
ALLOWED_UPDATE_FREQUENCIES = Literal[
    "realtime", "hourly", "daily", "weekly", "monthly", "adhoc"
]
ALLOWED_PRIORITIES = Literal["critical", "high", "medium", "low"]

def validate_asset_tags(tags: dict[str, str]) -> None:
    """Validate that asset tags conform to the schema."""
    required_tags = ["domain", "data_tier", "tool", "pipeline_stage"]
    for tag in required_tags:
        if tag not in tags:
            raise ValueError(f"Required tag '{tag}' missing from asset tags")

    # Add validation logic for allowed values
    # ...
```

### Pre-Commit Hook

Add tag validation to CI/CD:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: validate-dagster-tags
      name: Validate Dagster Asset Tags
      entry: python scripts/validate_dagster_tags.py
      language: python
      files: ^edna_dagster_pipelines/assets/.*\.py$
```

### PR Template Checklist

Update `.github/pull_request_template.md`:

```markdown
## Checklist for New Assets

- [ ] Asset includes all required tags: `domain`, `data_tier`, `tool`, `pipeline_stage`
- [ ] Tag values conform to allowed schema (see `config/tag_schema.py`)
- [ ] Asset has appropriate `compute_kind` for UI display
- [ ] Operational tags (`op_tags`) use semantic concurrency keys
```

## Advanced Tag Patterns

### Complex Asset Selection Queries

Combine multiple tag filters for sophisticated job definitions:

```python
from dagster import AssetSelection

# All critical bronze-tier ingestion assets
critical_ingestion = AssetSelection.tag("priority", "critical") \
    .tag("data_tier", "bronze") \
    .tag("pipeline_stage", "ingestion")

# All public genomics data from external sources
public_genomics = AssetSelection.tag("domain", "genomics") \
    .tag("sensitivity", "public") \
    .tag("source", "genbank").or_(AssetSelection.tag("source", "ncbi"))

# Heavy compute jobs for rate limiting
heavy_compute_job = define_asset_job(
    name="heavy_compute_batch",
    selection=AssetSelection.tag("compute_intensity", "heavy"),
    tags={"dagster/priority": "-3"},  # Lower priority
)
```

### Dynamic Tag-Based Routing

Use tags to route assets to different execution environments:

```python
@asset(
    tags={
        "domain": "genomics",
        "compute_intensity": "heavy",
        "execution_env": "gpu_cluster",
    },
)
def ml_model_training(context, ...): ...

# In job definition
gpu_cluster_job = define_asset_job(
    name="gpu_cluster_processing",
    selection=AssetSelection.tag("execution_env", "gpu_cluster"),
    tags={
        "dagster-k8s/config": {
            "pod_template_spec_metadata": {
                "labels": {"node_pool": "gpu_pool"},
            }
        }
    },
)
```

### Tag-Based Observability

Create monitoring jobs that check tag consistency:

```python
@asset(
    group_name="monitoring",
    tags={
        "domain": "monitoring",
        "pipeline_stage": "validation",
    },
)
def tag_consistency_report(context):
    """Generate report of assets with missing or invalid tags."""
    # Query all assets from context.instance
    # Check for required tags
    # Report violations
    return validation_report
```

## Reference Documentation

### Complete Tag Reference Table

| Tag Key | Type | Required | Values | Purpose |
|---------|------|----------|--------|---------|
| `domain` | String | Yes | `genomics`, `taxonomy`, `biodiversity`, `quality_control`, `reporting`, `monitoring` | Primary business domain |
| `data_tier` | String | Yes | `bronze`, `silver`, `gold` | Medallion architecture tier |
| `source` | String | Recommended | `genbank`, `gbif`, `ncbi`, `inaturalist`, `user_upload`, `internal` | Data source system |
| `sensitivity` | String | Recommended | `public`, `internal`, `restricted` | Data sensitivity classification |
| `pipeline_stage` | String | Yes | `ingestion`, `validation`, `transformation`, `enrichment`, `aggregation`, `export` | Processing stage |
| `tool` | String | Yes | `python`, `dbt`, `r`, `spark`, `custom` | Primary processing tool |
| `compute_intensity` | String | Optional | `light`, `medium`, `heavy` | Resource requirements |
| `update_frequency` | String | Recommended | `realtime`, `hourly`, `daily`, `weekly`, `monthly`, `adhoc` | Expected refresh rate |
| `priority` | String | Optional | `critical`, `high`, `medium`, `low` | Business priority |
| `team` | String | Optional | Team name | Owning team |
| `sla` | String | Optional | Duration (e.g., `15min`, `1h`, `24h`) | Service-level agreement |
| `api` | String | Optional | API identifier | For concurrency grouping |

### AssetSelection Syntax Reference

```python
# Single tag match
AssetSelection.tag("domain", "taxonomy")

# Multiple tags (AND)
AssetSelection.tag("domain", "taxonomy").tag("data_tier", "bronze")

# Tag OR
AssetSelection.tag("source", "gbif").or_(AssetSelection.tag("source", "ncbi"))

# Combine with group selection
AssetSelection.groups("project_tronko").tag("priority", "critical")

# Upstream and downstream navigation
AssetSelection.tag("domain", "taxonomy").upstream()
AssetSelection.tag("data_tier", "bronze").downstream(depth=2)
```

## Benefits of This Strategy

### Scalability
- **Self-service asset creation**: New assets automatically included in relevant jobs when tagged correctly
- **Reduced maintenance**: Jobs don't need updates when assets are added/removed
- **Clear ownership**: Tags identify responsible teams and priorities

### Observability
- **Better UI navigation**: Tags appear in Dagster UI for filtering and search
- **Kind tags**: Visual icons help identify asset types at a glance
- **Metadata-driven monitoring**: Query assets by tag for health checks

### Flexibility
- **Policy-based orchestration**: Define rules once, apply to all matching assets
- **Dynamic scheduling**: Different schedules for different priority levels
- **Resource optimization**: Route heavy compute to appropriate infrastructure

### Team Collaboration
- **Shared vocabulary**: Tags create common language across teams
- **Domain isolation**: Teams can filter to their assets using tags
- **Clear contracts**: SLAs and priorities visible via tags

## Common Patterns and Anti-Patterns

### ✅ Good Patterns

1. **Use shared tag dictionaries for consistency**
   ```python
   COMMON_TAGS = {"domain": "taxonomy", "tool": "python"}

   @asset(tags={**COMMON_TAGS, "source": "ncbi"})
   def asset1(context, ...): ...

   @asset(tags={**COMMON_TAGS, "source": "gbif"})
   def asset2(context, ...): ...
   ```

2. **Separate categorization tags from operational tags**
   ```python
   @asset(
       tags={...},      # For categorization and automation
       op_tags={...},   # For K8s config and concurrency
   )
   ```

3. **Use tag-based job selection for flexibility**
   ```python
   job = define_asset_job(
       selection=AssetSelection.tag("domain", "taxonomy"),
   )
   ```

### ❌ Anti-Patterns

1. **Don't mix categorization and configuration in tags**
   ```python
   # Bad: Mixing concerns
   tags={"domain": "taxonomy", "dagster/concurrency_key": "limit_5"}

   # Good: Separate concerns
   tags={"domain": "taxonomy"}
   op_tags={"dagster/concurrency_key": "taxonomy_limit"}
   ```

2. **Don't use freeform tag values**
   ```python
   # Bad: Inconsistent values
   tags={"domain": "Taxonomy"}  # Capitalized
   tags={"domain": "tax"}       # Abbreviated

   # Good: Consistent schema
   tags={"domain": "taxonomy"}
   ```

3. **Don't skip required tags**
   ```python
   # Bad: Missing required tags
   tags={"source": "gbif"}

   # Good: All required tags present
   tags={
       "domain": "biodiversity",
       "data_tier": "silver",
       "tool": "python",
       "pipeline_stage": "enrichment",
       "source": "gbif",
   }
   ```

## Next Steps

1. **Review and approve** this tagging strategy with the team
2. **Start Phase 1** migration by adding tags to critical assets
3. **Create shared tag dictionaries** at the module level
4. **Update documentation** with tag taxonomy and examples
5. **Implement validation** in CI/CD pipeline
6. **Train team** on tag-based asset selection patterns

## Related Research
- [Asset Selection Syntax](https://docs.dagster.io/guides/build/assets/asset-selection-syntax) - Official Dagster documentation

## Future Enhancements

- **Tag lineage visualization**: Build UI to show tag propagation through asset graph
- **Automated tag inference**: ML model to suggest tags based on asset code and dependencies
- **Tag-based alerting**: Create alerts when assets with specific tags fail
- **Cost tracking**: Use tags to track compute costs by domain/team
