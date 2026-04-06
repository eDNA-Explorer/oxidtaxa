#!/usr/bin/env bash
# Benchmark: R/C vs Python/Rust IDTAXA implementation
# Tests at multiple dataset sizes to show scaling behavior.
#
# Usage: bash benchmarks/run_benchmark.sh

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_FILE="benchmarks/results.txt"
DATA_DIR="benchmarks/data"
PYTHON=".venv/bin/python"

# Ensure release build
echo "Building Python/Rust in release mode..."
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --manifest-path rust/Cargo.toml --release > /dev/null 2>&1
echo "Build complete."
echo ""

THRESHOLD=40
BOOTSTRAPS=50
STRAND="both"
MIN_DESCEND=0.98
FULL_LENGTH=0
N_RUNS=3

{
echo "====================================================="
echo "  IDTAXA Benchmark: R/C vs Python/Rust"
echo "  $(date)"
echo "  Hardware: $(sysctl -n hw.ncpu) cores, $(sysctl -n machdep.cpu.brand_string)"
echo "====================================================="
echo ""
echo "Settings: threshold=$THRESHOLD bootstraps=$BOOTSTRAPS strand=$STRAND"
echo "Runs: $N_RUNS (best of)"
echo ""

# ── Helper ──────────────────────────────────────────────────────────────────
best_of() {
    printf '%s\n' "$@" | sort -n | head -1
}

time_cmd() {
    local start end
    start=$($PYTHON -c "import time; print(time.perf_counter())")
    "$@" > /dev/null 2>&1
    end=$($PYTHON -c "import time; print(time.perf_counter())")
    $PYTHON -c "print(f'{$end - $start:.4f}')"
}

for SIZE in 1000 5000 10000; do
    FASTA="$DATA_DIR/bench_${SIZE}_ref.fasta"
    TAX="$DATA_DIR/bench_${SIZE}_ref_taxonomy.tsv"
    QUERY="$DATA_DIR/bench_${SIZE}_query.fasta"

    if [ ! -f "$FASTA" ]; then
        echo "Skipping $SIZE: $FASTA not found"
        continue
    fi

    N_REF=$(grep -c '^>' "$FASTA")
    N_QUERY=$(grep -c '^>' "$QUERY")

    echo "====================================================="
    echo "  Dataset: ${SIZE} ref sequences, ${N_QUERY} query sequences"
    echo "====================================================="
    echo ""

    # ── R/C ─────────────────────────────────────────────────────────────
    echo "--- R/C ---"
    r_train_times=()
    for i in $(seq 1 $N_RUNS); do
        t=$(time_cmd Rscript train_idtaxa.R "$FASTA" "$TAX" /tmp/bench_r_${SIZE}.rds)
        r_train_times+=("$t")
        echo "  R train run $i: ${t}s"
    done
    r_train=$(best_of "${r_train_times[@]}")

    r_class_times=()
    for i in $(seq 1 $N_RUNS); do
        t=$(time_cmd Rscript classify_idtaxa.R "$QUERY" /tmp/bench_r_${SIZE}.rds /tmp/bench_r_${SIZE}.tsv \
            $THRESHOLD $BOOTSTRAPS $STRAND $MIN_DESCEND $FULL_LENGTH 1)
        r_class_times+=("$t")
        echo "  R classify run $i: ${t}s"
    done
    r_class=$(best_of "${r_class_times[@]}")
    echo ""

    # ── Python/Rust ─────────────────────────────────────────────────────
    echo "--- Python/Rust ---"
    rs_train_times=()
    for i in $(seq 1 $N_RUNS); do
        t=$(time_cmd $PYTHON train_idtaxa.py "$FASTA" "$TAX" /tmp/bench_rs_${SIZE}.bin --seed 42)
        rs_train_times+=("$t")
        echo "  Rust train run $i: ${t}s"
    done
    rs_train=$(best_of "${rs_train_times[@]}")

    rs_class_1t_times=()
    for i in $(seq 1 $N_RUNS); do
        t=$(time_cmd $PYTHON classify_idtaxa.py "$QUERY" /tmp/bench_rs_${SIZE}.bin /tmp/bench_rs_${SIZE}_1t.tsv \
            $THRESHOLD $BOOTSTRAPS $STRAND $MIN_DESCEND $FULL_LENGTH 1 --seed 42 --deterministic)
        rs_class_1t_times+=("$t")
        echo "  Rust classify (1T) run $i: ${t}s"
    done
    rs_class_1t=$(best_of "${rs_class_1t_times[@]}")

    rs_class_8t_times=()
    for i in $(seq 1 $N_RUNS); do
        t=$(time_cmd $PYTHON classify_idtaxa.py "$QUERY" /tmp/bench_rs_${SIZE}.bin /tmp/bench_rs_${SIZE}_8t.tsv \
            $THRESHOLD $BOOTSTRAPS $STRAND $MIN_DESCEND $FULL_LENGTH 8 --seed 42)
        rs_class_8t_times+=("$t")
        echo "  Rust classify (8T) run $i: ${t}s"
    done
    rs_class_8t=$(best_of "${rs_class_8t_times[@]}")
    echo ""

    # ── Summary for this size ───────────────────────────────────────────
    $PYTHON -c "
r_train = $r_train
r_class = $r_class
rs_train = $rs_train
rs_1t = $rs_class_1t
rs_8t = $rs_class_8t

print(f'  Summary ($SIZE ref / $N_QUERY query):')
print(f'  {\"\":28s} {\"R/C\":>8s} {\"Rust 1T\":>8s} {\"Rust 8T\":>8s} {\"Speedup (1T)\":>12s} {\"Speedup (8T)\":>12s}')
print(f'  {\"─\"*28} {\"─\"*8} {\"─\"*8} {\"─\"*8} {\"─\"*12} {\"─\"*12}')
print(f'  {\"Train\":28s} {r_train:7.2f}s {rs_train:7.2f}s {\"\":>8s} {r_train/rs_train:11.1f}x {\"\":>12s}')
print(f'  {\"Classify\":28s} {r_class:7.2f}s {rs_1t:7.2f}s {rs_8t:7.2f}s {r_class/rs_1t:11.1f}x {r_class/rs_8t:11.1f}x')
print(f'  {\"Total\":28s} {r_train+r_class:7.2f}s {rs_train+rs_1t:7.2f}s {rs_train+rs_8t:7.2f}s {(r_train+r_class)/(rs_train+rs_1t):11.1f}x {(r_train+r_class)/(rs_train+rs_8t):11.1f}x')
print()
"
done

echo "Results saved to $RESULTS_FILE"
} 2>&1 | tee "$RESULTS_FILE"
