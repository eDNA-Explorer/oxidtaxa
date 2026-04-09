"""Run Rust IDTAXA on real-world production data (River Partners plates).

Reuses pre-processed query FASTAs and CRUXv2 reference from the
assignment-tool-benchmarking project. Only runs IDTAXA (Rust implementation).
Saves results (model + classification TSVs) to this repo.

Usage:
    python benchmarks/run_real_data_idtaxa.py
    python benchmarks/run_real_data_idtaxa.py --plate A
    python benchmarks/run_real_data_idtaxa.py --all-plates
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — input data lives in the assignment-tool-benchmarking repo
# ---------------------------------------------------------------------------

BENCH_REPO = Path.home() / "assignment-tool-benchmarking" / "projects" / "assignment_benchmarks"
BENCH_DATA = BENCH_REPO / "data"

# CRUXv2 vert12S reference (178K seqs)
CRUXV2_REF_FASTA = Path.home() / "rcrux-py" / "databases" / "vert12s" / "unfiltered" / "vert12S_lca.fasta"
CRUXV2_TAX_FILE = Path.home() / "rcrux-py" / "databases" / "vert12s" / "unfiltered" / "vert12S_lca_taxonomy.txt"

# All 7 River Partners plates (vert12S)
RIVER_PARTNERS_PLATES: dict[str, str] = {
    "A": "cmdqxzmkm0007l104b4y3rooj",
    "B": "cme7t9tsp0001l204i70ou9qv",
    "C": "cme7vmz4h0001l8046po4nsse",
    "D": "cmgv2jif80001ky04uffqremg",
    "E": "cmgv53m3a0001gs041yofo4gn",
    "F": "cmgpr64ge0001ky04049q049l",
    "G": "cmgpou6i10001ld04dud4mo89",
}

# IDTAXA config (best from config sweep)
IDTAXA_CONFIG = {
    "threshold": 40,
    "bootstraps": 50,
    "strand": "both",
    "min_descend": 0.98,
    "full_length": 0.0,
    "processors": 8,
    "seed": 42,
}

# Output lives in this repo
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "real_data"


def find_query_fasta(project_id: str) -> Path | None:
    """Find the merged derep FASTA from the benchmarking repo's real_data output."""
    candidate = (
        BENCH_DATA / "real_data" / project_id / "results" / "pooled"
        / "denoised" / "vsearch_merge" / "project_merged_derep.fasta"
    )
    if candidate.exists():
        return candidate

    # Fallback: merged clean
    candidate2 = candidate.parent / "project_merged_clean.fasta"
    if candidate2.exists():
        return candidate2

    return None


def run_plate(plate_label: str, project_id: str) -> None:
    """Train + classify for one plate using Rust IDTAXA."""
    from oxidtaxa import train, classify

    print(f"\n{'=' * 60}")
    print(f"  Plate {plate_label} ({project_id})")
    print(f"{'=' * 60}")

    # Check inputs
    if not CRUXV2_REF_FASTA.exists():
        print(f"  [SKIP] Reference not found: {CRUXV2_REF_FASTA}")
        return
    if not CRUXV2_TAX_FILE.exists():
        print(f"  [SKIP] Taxonomy not found: {CRUXV2_TAX_FILE}")
        return

    query_fasta = find_query_fasta(project_id)
    if query_fasta is None:
        print(f"  [SKIP] No query FASTA found for {project_id}")
        print(f"         Expected at: {BENCH_DATA / 'real_data' / project_id / 'results' / 'pooled' / 'denoised' / 'vsearch_merge'}")
        return

    # Count sequences
    with open(CRUXV2_REF_FASTA) as f:
        n_ref = sum(1 for line in f if line.startswith(">"))
    with open(query_fasta) as f:
        n_query = sum(1 for line in f if line.startswith(">"))

    print(f"  Reference: {CRUXV2_REF_FASTA.name} ({n_ref:,} seqs)")
    print(f"  Query:     {query_fasta.name} ({n_query:,} seqs)")
    print(f"  Config:    {IDTAXA_CONFIG}")

    # Output paths
    plate_dir = OUTPUT_DIR / f"plate_{plate_label}"
    plate_dir.mkdir(parents=True, exist_ok=True)
    model_path = plate_dir / "model.bin"
    output_tsv = plate_dir / "classifications.tsv"

    # Train (shared model across plates since same reference, but save per-plate for clarity)
    # Use shared model if already trained
    shared_model = OUTPUT_DIR / "shared_model.bin"
    if shared_model.exists():
        print(f"\n  Using cached model: {shared_model}")
        model_path = shared_model
    else:
        print(f"\n  Training on {n_ref:,} reference sequences...")
        t0 = time.time()
        train(
            fasta_path=str(CRUXV2_REF_FASTA),
            taxonomy_path=str(CRUXV2_TAX_FILE),
            output_path=str(shared_model),
            seed=IDTAXA_CONFIG["seed"],
        )
        train_time = time.time() - t0
        print(f"  Train: {train_time:.1f}s")
        model_path = shared_model

    # Classify
    print(f"  Classifying {n_query:,} ASVs (processors={IDTAXA_CONFIG['processors']})...")
    t0 = time.time()
    classify(
        query_path=str(query_fasta),
        model_path=str(model_path),
        output_path=str(output_tsv),
        threshold=IDTAXA_CONFIG["threshold"],
        bootstraps=IDTAXA_CONFIG["bootstraps"],
        strand=IDTAXA_CONFIG["strand"],
        min_descend=IDTAXA_CONFIG["min_descend"],
        full_length=IDTAXA_CONFIG["full_length"],
        processors=IDTAXA_CONFIG["processors"],
        seed=IDTAXA_CONFIG["seed"],
    )
    classify_time = time.time() - t0
    print(f"  Classify: {classify_time:.1f}s")

    # Summary
    with open(output_tsv) as f:
        n_results = sum(1 for _ in f) - 1  # subtract header
    print(f"\n  Results: {output_tsv}")
    print(f"  {n_results:,} ASVs classified")
    print(f"  Total time: {classify_time:.1f}s (classify only, model was cached or {train_time:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Rust IDTAXA on real production data")
    parser.add_argument(
        "--plate",
        choices=list(RIVER_PARTNERS_PLATES.keys()),
        default="A",
        help="Which plate to run (default: A)",
    )
    parser.add_argument(
        "--all-plates",
        action="store_true",
        help="Run all 7 River Partners plates",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Rust IDTAXA — Real Data Benchmark")
    print("=" * 60)
    print(f"  Reference: {CRUXV2_REF_FASTA}")
    print(f"  Output:    {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_plates:
        for label, pid in RIVER_PARTNERS_PLATES.items():
            try:
                run_plate(label, pid)
            except Exception as e:
                print(f"  [FAIL] Plate {label}: {e}")
    else:
        label = args.plate
        run_plate(label, RIVER_PARTNERS_PLATES[label])


if __name__ == "__main__":
    main()
