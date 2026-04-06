## QC Tools Integration

All QC tool integrations use the `qc-tools-lib` library (`libraries/qc-tools-lib/`) with hexagonal architecture (ports and adapters pattern).

### Available Executors

- **FastP** (`fastp`): Quality filtering and adapter trimming
- **Cutadapt** (`cutadapt`): Primer/adapter removal and demultiplexing
- **DADA2** (`Rscript`): ASV inference and denoising
- **BBDuk** (`bbduk.sh`): Contamination removal
- **FastQC** (`fastqc`): Quality control reporting
- **MultiQC** (`multiqc`): Report aggregation
- **SeqKit** (`seqkit`): Stats, fq2fa conversion, and deduplication (3 separate executors)

All implement the `ToolExecutor` protocol from `qc_tools_lib/protocols.py`.

### Using QC Tool Executors

```python
from qc_tools_lib.executors.seqkit import SeqKitStatsExecutor
from qc_tools_lib.models import SeqKitStatsConfig
from qc_tools_lib.adapters.subprocess_adapter import SubprocessAdapter
from qc_tools_lib.adapters.storage_adapter import LocalStorageAdapter
from pathlib import Path

# Create configuration
config = SeqKitStatsConfig(
    input_files=["sample.fastq.gz"],
    output_dir="/tmp/seqkit",
    include_all=True,
)

# Execute with dependency injection
executor = SeqKitStatsExecutor()
result = executor.execute(
    config,
    SubprocessAdapter(),          # ProcessRunner adapter
    LocalStorageAdapter(Path(".")),  # StoragePort adapter
    logger=context.log,           # Optional, duck-typed (any .info()/.error())
)
```

Executors never raise exceptions for tool failures — they return `ToolResult(status="FAIL")`. Always check `result.success` before proceeding.

In Dagster, executors are used in **helper functions** (`qcassign/helpers/`), not directly in ops. Ops are thin wrappers that extract Dagster resources and call helpers.

### Parsing Tool Output

```python
from qc_tools_lib.parsers.seqkit_stats_parser import parse_seqkit_stats

# Parse seqkit stats tabular output
records = parse_seqkit_stats(result.output)

for record in records:
    print(f"{record.file}: {record.num_seqs} sequences")
    print(f"  GC content: {record.GC_percent:.1f}%")
    if record.Q20_percent:
        print(f"  Q20: {record.Q20_percent:.1f}%")
```

### Testing

Use `MockProcessRunner` and `MockStorageAdapter` from `qc_tools_lib.adapters.mock_adapter` to test without real binaries.
