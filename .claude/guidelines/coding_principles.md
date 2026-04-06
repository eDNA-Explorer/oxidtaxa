# Coding Principles: Architectural Standards for eDNA Explorer Data Pipelines

**Version**: 1.0.0 | **Last Updated**: 2025-10-10 | **Status**: Canonical Reference

## Purpose

Mandatory architectural principles for all code in the eDNA Explorer Data Pipelines monorepo.

**Foundation**: SOLID principles, hexagonal architecture, modern Python type system, and project-specific patterns.

---

## Section 1: MANDATORY - Function-Op-Asset Trinity Pattern

### Pattern Overview

**CRITICAL**: All new jobs and assets MUST follow the Function-Op-Asset Trinity pattern.

### Core Pattern: Four Layers (REQUIRED)

```
Layer 1: Core Function (Pure Business Logic)
    ↓
Layer 2: Dagster Op/Asset (Thin Orchestration)
    ↓
Layer 3: CLI Interface (OPTIONAL)
    ↓
Layer 4: Resource Protocols (REQUIRED)
```

#### Layer 1: Core Function

**Location**: `edna_dagster_pipelines/core/{module}/`

**Requirements**:
- ✅ Accept typed dataclasses for config/results
- ✅ Accept dependencies via protocols (NOT Dagster resources)
- ✅ Accept optional logger (duck-typed)
- ❌ NEVER import from `dagster`
- ❌ NEVER use `context.log` or `context.resources`

**Example**: Core function with protocols, dataclasses, optional logger.

```python
def calculate_diversity(
    storage: StoragePort,  # Protocol dependency
    config: DiversityConfig,
    logger: Logger | None = None  # Logger Protocol, not logging.Logger
) -> DiversityResult:
    if logger:
        logger.info("Processing")
    df = storage.read_dataframe(path)
    # Business logic...
    return DiversityResult(shannon=1.5, simpson=0.8, richness=10)
```

#### Layer 2: Dagster Op/Asset

**Location**: `edna_dagster_pipelines/ops/{module}/` or `assets/{module}/`

**Requirements**:
- ✅ Extract concrete objects from Dagster resources
- ✅ Call core function with dependencies
- ✅ Handle Dagster-specific concerns (metadata, triggering)
- ❌ NEVER implement business logic

**Use Assets (PREFERRED)**: Data products, materialization
**Use Ops**: Traditional DAGs, complex control flow

**Example**: Asset extracts resources, calls core.

```python
@asset(group_name="diversity")
def sample_diversity(context, gcs, env_config) -> pl.DataFrame:
    adapter = GCSStorageAdapter(gcs.get_client(), env_config["BUCKET"])
    config = DiversityConfig(sample_column="sample_id", abundance_column="count")
    result = calculate_diversity(adapter, config, context.log)
    context.add_output_metadata({"shannon": result.shannon_index})
    return pl.DataFrame({"metric": ["shannon"], "value": [result.shannon_index]})
```

#### Layer 3: CLI Interface (OPTIONAL)

**Location**: `edna_dagster_pipelines/cli/`

**When to Create**:
- ✅ Frequently debugged, complex operations
- ❌ Simple queries, trivial operations

**Example**: CLI for debugging.

```python
def main():
    args = parse_args()  # Get bucket, input_path
    adapter = GCSStorageAdapter(storage.Client(), args.bucket)
    result = calculate_diversity(adapter, DiversityConfig(), None)
    print(f"Shannon: {result.shannon_index:.2f}")
```

#### Layer 4: Resource Protocols (REQUIRED)

**Location**: `edna_dagster_pipelines/resources/protocols.py`

**Why Required**:
1. DIP: Core depends on abstractions
2. ISP: Focused interfaces
3. Testability: Easy mocking
4. Type safety with structural typing

**Example**: Protocols and adapters.

```python
# resources/protocols.py
class StoragePort(Protocol):
    def read_dataframe(self, path: str) -> pl.DataFrame: ...

# resources/gcs.py
class GCSStorageAdapter:  # Implements StoragePort
    def read_dataframe(self, path: str) -> pl.DataFrame:
        return pl.read_parquet(blob.download_as_bytes())

# resources/mocks.py
class MockStorageAdapter:  # Implements StoragePort
    def read_dataframe(self, path: str) -> pl.DataFrame:
        return self.storage.get(path, pl.DataFrame())
```

#### Asset Checks

**When to Use**: Critical invariants, blocking checks, pre-conditions

**Example**: Asset check validates output.

```python
@asset_check(asset=sample_diversity, blocking=True)
def sample_diversity_is_valid(context, sample_diversity: pl.DataFrame):
    passed = sample_diversity["shannon"][0] >= 0
    return AssetCheckResult(passed=passed)
```

### Implementation Rules

#### ✅ DO

- Separate business logic (core) from orchestration (op/asset)
- Define interfaces with `typing.Protocol`
- Depend on protocols, not concrete implementations
- Use typed dataclasses for config/results
- Create unit tests with mocks, integration tests for ops/assets

#### ❌ DON'T

- Mix business logic with Dagster code in core
- Import Dagster in core functions
- Use `context.resources` or `context.log` in core
- Return Dagster types from core
- Hard-code dependencies in core

### File Organization

```
edna_dagster_pipelines/
├── core/{domain}/              # Pure business logic
├── resources/
│   ├── protocols.py            # REQUIRED: Ports
│   ├── gcs.py                  # Adapters
│   └── mocks.py
├── ops/{domain}/               # Op wrappers
├── assets/{domain}/            # Asset definitions
├── cli/                        # OPTIONAL: CLI
└── tests/
    ├── core/{domain}/          # Unit tests
    └── assets/{domain}/        # Integration tests
```

### Decision Framework

1. **Does this need Dagster?** → Yes: op/asset pattern
2. **Business logic or orchestration?** → Logic: `core/`, Orchestration: `ops/`/`assets/`
3. **Op or asset?** → Asset (PREFERRED), Op for complex control flow
4. **Need CLI?** → High priority: frequently debugged
5. **What resources?** → Define protocols (ISP), depend on protocols (DIP)
6. **How to test?** → Core: unit tests with mocks, Op/Asset: integration tests

### Checklist

**Architecture**:
- [ ] Core function in `core/{domain}/`
- [ ] Dataclasses for config/results
- [ ] Protocols in `resources/protocols.py`
- [ ] Adapters implement protocols
- [ ] Op/Asset wrapper
- [ ] Asset check (if asset)
- [ ] CLI (if high value)

**Code Quality**:
- [ ] No Dagster imports in core
- [ ] Type hints, docstrings
- [ ] Comprehensive error handling

**Testing**:
- [ ] Unit tests for core
- [ ] Integration tests for ops/assets
- [ ] All tests pass: `poetry run pytest`

**Pre-Commit**: See `rules.md` § Pre-Commit Commands.

---

## Section 2: SOLID Principles Application

### 1. Single Responsibility Principle (SRP)

"One reason to change." Separate business logic from persistence. Core: business only. Ops/Assets: orchestration only.

### 2. Open/Closed Principle (OCP)

"Open for extension, closed for modification." Use abstraction (Protocol/ABC) to add features without modifying existing code.

### 3. Liskov Substitution Principle (LSP)

"Subtypes must be substitutable." Honor parent contracts. Never override with `NotImplementedError`.

### 4. Interface Segregation Principle (ISP)

"Clients shouldn't depend on unused methods." Create small, focused interfaces. **Prefer `typing.Protocol`** (structural) over ABC (nominal).

### 5. Dependency Inversion Principle (DIP)

"Depend on abstractions." Core accepts protocols, ops inject concrete implementations. Never create dependencies internally.

**Example**:
```python
def process_core(storage: StoragePort, config): ...  # Protocol

@op
def process_op(context, gcs):
    adapter = GCSStorageAdapter(gcs.get_client())  # Concrete
    return process_core(adapter, config)  # Inject
```

---

## Section 3: Hexagonal Architecture Alignment

### Core Concepts

1. **Core (Inside)**: Pure business logic
2. **Ports (Interfaces)**: Abstract contracts (Protocols)
3. **Adapters (Outside)**: Concrete implementations
4. **Infrastructure**: External systems

### Mapping

```
Core → Hexagon | Protocols → Ports | Adapters → Infrastructure | Ops/Assets → Application
```

### File Structure

```
core/                    # HEXAGON (INSIDE)
resources/
├── protocols.py         # PORTS
├── gcs.py               # ADAPTER
└── mocks.py             # ADAPTER (test)
assets/                  # APPLICATION
ops/                     # APPLICATION
```

### Benefits

- **Testable**: Core without infrastructure
- **Swappable**: Change storage without touching core
- **Maintainable**: Clear boundaries
- **Understandable**: Business logic isolated
- **Reversible**: Easy to change decisions

---

## Summary

All new code must:

1. **Follow Function-Op-Asset Trinity**:
   - Core (pure logic)
   - Ops/Assets (orchestration)
   - CLI (optional)
   - Protocols (REQUIRED)

2. **Apply SOLID**:
   - SRP, OCP, LSP, ISP, DIP

3. **Implement Hexagonal Architecture**:
   - Core isolated
   - Ports define contracts
   - Adapters provide infrastructure
   - Application orchestrates

**Remember**: These principles are interconnected and enable testable, maintainable, flexible systems.

---

**Version**: 1.0.0 | **Last Updated**: 2025-10-10 | **Status**: Canonical Reference
