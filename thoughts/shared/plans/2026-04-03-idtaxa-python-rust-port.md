# IDTAXA Python/Rust Port Implementation Plan

## Overview

Port the self-contained R/C IDTAXA implementation (8 C functions, 5 R modules, ~2,000 lines) to a Python/Rust stack. All computation lives in a pure Rust crate; Python is a thin CLI layer via PyO3/Maturin. Strict PRNG matching (MT19937 via `rand_mt`) enables bit-identical golden test comparison against the existing 237-check R test suite.

## Current State Analysis

### What exists today

```
idtaxa-optim/
  R/
    fasta_io.R         # 27 lines  - FASTA reader → named character vector
    seq_utils.R        # 19 lines  - vcountPattern, reverseComplement
    RemoveGaps.R       # 63 lines  - Gap removal wrapper
    LearnTaxa.R        # 487 lines - Training algorithm (complex tree traversal + bootstrap)
    IdTaxa.R           # 537 lines - Classification algorithm (bootstrap + confidence scoring)
  src/
    idtaxa.h           # 84 lines  - SeqSet_holder/Chars_holder types, forward decls
    enumerate_sequence.c # 431 lines - K-mer enumeration + 3 masking algorithms
    vector_sums.c      # 147 lines - vectorSum + parallelMatch
    utils.c            # 97 lines  - intMatch, groupMax, detectCores
    remove_gaps.c      # 74 lines  - Gap removal (STRSXP I/O)
    R_init_idtaxa.c    # 22 lines  - .Call registration
    Makevars           # 2 lines   - Build flags
  tests/
    data/              # test_ref.fasta (80 seqs), test_ref_taxonomy.tsv, test_query.fasta (15 seqs)
    golden/            # 110 .rds reference files
    generate_golden.R  # 530 lines - Generates golden baselines under Bioconductor
    run_golden.R       # 538 lines - 237 checks against local R/C implementation
  train_idtaxa.R       # 98 lines  - CLI: FASTA + taxonomy → .rds model
  classify_idtaxa.R    # 113 lines - CLI: FASTA + model → TSV output
```

### Key discoveries from codebase analysis

1. **PRNG consumption order in LearnTaxa** (`R/LearnTaxa.R`):
   - Line 356: `sample(n, s*B, replace=TRUE)` — called once per training sequence per iteration, inside nested loop (lines 329-420). Consumes `s*B` random draws per call.
   - Total PRNG draws per `LearnTaxa` call: varies by tree structure and iteration count.

2. **PRNG consumption order in IdTaxa** (`R/IdTaxa.R`):
   - Line 292: `sample(n, s*B[I[i]], replace=TRUE)` — decision k-mer sampling per tree descent
   - Line 346: `sample(mykmers, s*B[I[i]], replace=TRUE)` — query k-mer subsampling
   - Line 387: `sample(w, 1)` — tie-breaking (rare)
   - Critical: IdTaxa processes sequences in order `I = c(seq_along(testkmers), boths)` (line 263-264), meaning forward pass then reverse complement pass. PRNG draws must follow this exact order.

3. **R's `sample(n, size, replace=TRUE)` implementation** (R source `src/main/random.c`):
   ```c
   // SampleReplace: for replace=TRUE
   for (i = 0; i < size; i++)
       y[i] = (int)(n * unif_rand()) + 1;  // 1-indexed
   ```
   Where `unif_rand()` = `MT_genrand()` = `tempered_u32 * 2.3283064365386963e-10`.

4. **No S4 classes remain**: The R/C code already uses plain character vectors and lists with S3 class attributes (`class(result) <- c("Taxa", "Train")`). The Rust translation maps directly to structs.

5. **All C functions are pure computation**: None access R's PRNG. Only the R layer calls `sample()`. This means the C→Rust port is deterministic (no PRNG concern), and PRNG matching only matters for the R→Rust algorithm port.

6. **Training set serialization**: Current format is R's `.rds` (binary R objects). New format will be bincode (compact, fast Rust-native). No backward compatibility needed per user decision.

## Desired End State

```
idtaxa-optim/
  # === Existing R/C (kept as reference for golden test generation) ===
  R/                           # R wrappers (unchanged)
  src/                         # C source (unchanged)
  train_idtaxa.R               # R CLI (unchanged)
  classify_idtaxa.R            # R CLI (unchanged)
  tests/
    data/                      # Shared test FASTA + taxonomy
    golden/                    # .rds golden files (unchanged)
    golden_json/               # JSON golden files for Rust tests (new)
    export_golden_json.R       # .rds → JSON converter (new)
    generate_golden.R          # Original (unchanged)
    run_golden.R               # R test runner (unchanged)

  # === New Python/Rust implementation ===
  pyproject.toml               # Maturin + Python project config
  rust/
    Cargo.toml                 # Pure Rust crate + PyO3 feature
    src/
      lib.rs                   # PyO3 module + public API
      types.rs                 # TrainingSet, ClassificationResult, configs
      rng.rs                   # R-compatible MT19937 + sample()
      fasta.rs                 # FASTA reader
      sequence.rs              # reverseComplement, removeGaps, vcountPattern
      kmer.rs                  # enumerateSequence + masking (alphabetFrequency, maskRepeats, maskSimple, maskNumerous)
      alphabet.rs              # alphabetSize
      matching.rs              # intMatch, vectorSum, parallelMatch, groupMax
      training.rs              # LearnTaxa algorithm
      classify.rs              # IdTaxa algorithm
    tests/
      common/mod.rs            # Test helpers (load JSON, epsilon comparison)
      test_rng.rs              # PRNG parity with R
      test_fasta.rs            # FASTA reader tests
      test_sequence.rs         # DNA utility tests
      test_kmer.rs             # K-mer enumeration tests
      test_matching.rs         # Matching function tests
      test_training.rs         # LearnTaxa golden tests
      test_classify.rs         # IdTaxa golden tests
      test_integration.rs      # Full pipeline tests
  python/
    idtaxa/
      __init__.py              # Re-exports from _core
  train_idtaxa.py              # Python CLI (replaces .R)
  classify_idtaxa.py           # Python CLI (replaces .R)
```

**Verification**: `cd rust && cargo test` passes all golden tests. `maturin develop` builds the Python wheel. `python train_idtaxa.py` and `python classify_idtaxa.py` produce identical output to the R scripts.

## What We're NOT Doing

- No RNA or amino acid support (DNA only, same as R/C version)
- No FASTQ support (FASTA only)
- No backward compatibility with R's .rds model format (retrain only)
- No S4/S3 class system emulation in Python
- No web API or service layer — CLI scripts only
- No GPU acceleration — CPU with rayon parallelism
- No async I/O — synchronous file operations

---

## Phase 1: Golden Baseline Export

### Overview
Convert the 110 existing `.rds` golden files to JSON format that Rust can consume. Also export intermediate computational states (PRNG sequences, sampling matrices) to enable layer-by-layer Rust testing.

### 1.1 Create `tests/export_golden_json.R`

This script loads each `.rds` file and writes a corresponding `.json` file. Special handling:
- R's `NA_integer_` → JSON `null`
- R's `NA_real_` → JSON `null`
- R's `NA` in fraction vectors → JSON `null`
- Matrices → JSON arrays of arrays (row-major)
- TrainingSet (list with class) → JSON object with all fields
- ClassificationResult (list of lists) → JSON array of objects

```r
#!/usr/bin/env Rscript
library(jsonlite)

golden_dir <- "tests/golden"
json_dir <- "tests/golden_json"
dir.create(json_dir, showWarnings = FALSE, recursive = TRUE)

rds_files <- list.files(golden_dir, pattern = "\\.rds$", full.names = TRUE)
for (f in rds_files) {
  obj <- readRDS(f)
  json_name <- sub("\\.rds$", ".json", basename(f))
  # Custom serialization for training sets and classification results
  write_json(obj, file.path(json_dir, json_name), auto_unbox = TRUE,
             na = "null", digits = 17)  # max double precision
}
```

**Key detail**: Use `digits = 17` for full double-precision fidelity in JSON. This ensures floating-point round-trip accuracy.

### 1.2 Export PRNG verification data

Add to `export_golden_json.R`:
```r
# Generate PRNG verification sequence
set.seed(42)
prng_100 <- runif(100)  # First 100 draws from MT19937 with seed 42
write_json(prng_100, file.path(json_dir, "prng_seed42_100draws.json"), digits = 17)

# Export R's sample() outputs for verification
set.seed(42)
sample_10_from_50 <- sample(50L, 10L, replace = TRUE)
write_json(sample_10_from_50, file.path(json_dir, "prng_sample_10from50.json"))

set.seed(42)
sample_100_from_1000 <- sample(1000L, 100L, replace = TRUE)
write_json(sample_100_from_1000, file.path(json_dir, "prng_sample_100from1000.json"))
```

### 1.3 Export intermediate states for LearnTaxa/IdTaxa layer testing

Add intermediate state exports to `generate_golden.R` (or a new script):
```r
# After LearnTaxa, export the k-mers (sorted unique per sequence)
# These are deterministic (no PRNG) and verify the C→Rust k-mer port
set.seed(42)
ts <- LearnTaxa(train = seqs, taxonomy = tax, verbose = FALSE)
write_json(ts$kmers, file.path(json_dir, "intermediate_train_kmers.json"))
write_json(ts$decisionKmers, file.path(json_dir, "intermediate_decision_kmers.json"),
           auto_unbox = TRUE, na = "null", digits = 17)
```

### Success Criteria

- [x] `Rscript tests/export_golden_json.R` runs without errors
- [x] `tests/golden_json/` contains 110+ JSON files (115 total)
- [x] JSON files preserve full floating-point precision (17 digits)
- [x] PRNG verification files created
- [x] Intermediate state files created

---

## Phase 2A: Rust Foundation

### Overview
Set up the Rust crate, define core types, implement R-compatible PRNG, FASTA reader, and sequence utilities. Each module gets golden tests immediately.

### 2A.1 Project Setup

**File**: `rust/Cargo.toml`
```toml
[package]
name = "idtaxa"
version = "0.1.0"
edition = "2021"

[lib]
name = "idtaxa"
crate-type = ["cdylib", "rlib"]  # cdylib for PyO3, rlib for tests

[dependencies]
rand_mt = "4"                    # R-compatible MT19937
rayon = "1"                      # Parallel iterators (replaces OpenMP)
serde = { version = "1", features = ["derive"] }
serde_json = "1"                 # JSON golden test loading
bincode = "1"                    # Model serialization
pyo3 = { version = "0.22", features = ["extension-module"], optional = true }

[dev-dependencies]
approx = "0.5"                   # Floating-point epsilon assertions

[features]
default = []
python = ["pyo3"]
```

**File**: `pyproject.toml` (at project root)
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "idtaxa"
requires-python = ">=3.10"
version = "0.1.0"

[tool.maturin]
manifest-path = "rust/Cargo.toml"
python-source = "python"
module-name = "idtaxa._core"
features = ["python"]
```

### 2A.2 Core Types (`rust/src/types.rs`)

Direct mapping from R's training set list to Rust structs:

```rust
use serde::{Serialize, Deserialize};

/// Decision node in the taxonomic tree.
/// Maps to R's `decision_kmers[[k]]` = list(keep_indices, profile_matrix).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    /// K-mer indices used for classification decisions at this node.
    /// Maps to R's `decision_kmers[[k]][[1]]` (integer vector).
    pub keep: Vec<i32>,

    /// Profile matrix: rows = child subtrees, cols = kept k-mers.
    /// Maps to R's `decision_kmers[[k]][[2]]` = t(profile[keep,]).
    /// In R this is stored as t(profile[keep,]) so rows=subtrees, cols=kmers.
    pub profiles: Vec<Vec<f64>>,
}

/// A sequence that was misclassified during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemSequence {
    pub index: usize,       // 0-indexed (R uses 1-indexed)
    pub expected: String,
    pub predicted: String,
}

/// Trained IDTAXA model. Output of `learn_taxa()`.
/// Maps 1:1 to R's list with class c("Taxa", "Train").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSet {
    /// All distinct taxonomy strings (e.g., "Root;", "Root; Eukaryota;", ...).
    pub taxonomy: Vec<String>,
    /// Taxon name at each node (last component of taxonomy string).
    pub taxa: Vec<String>,
    /// Optional rank labels (e.g., "kingdom", "phylum", ...).
    pub ranks: Option<Vec<String>>,
    /// Depth of each taxonomy node (Root=1).
    pub levels: Vec<i32>,
    /// Indices of child nodes for each taxonomy node.
    /// Maps to R's `children` list of integer vectors.
    pub children: Vec<Vec<usize>>,
    /// Parent index for each taxonomy node (0 = no parent, i.e., Root).
    pub parents: Vec<usize>,
    /// Sampling fraction per node. None = problem group (was NA in R).
    pub fraction: Vec<Option<f64>>,
    /// Sequence indices belonging to each taxonomy node.
    pub sequences: Vec<Option<Vec<usize>>>,
    /// Sorted unique k-mer indices per training sequence.
    /// Maps to R's `kmers` list of integer vectors (already 1-indexed in R, we use 0-indexed).
    pub kmers: Vec<Vec<i32>>,
    /// Maps each training sequence to its taxonomy node index.
    pub cross_index: Vec<usize>,
    /// K-mer word size.
    pub k: usize,
    /// Inverse Document Frequency weights per k-mer.
    pub idf_weights: Vec<f64>,
    /// Decision k-mers and profiles at each internal node.
    pub decision_kmers: Vec<Option<DecisionNode>>,
    /// Sequences that were misclassified during training.
    pub problem_sequences: Vec<ProblemSequence>,
    /// Taxonomy nodes with unresolvable classification conflicts.
    pub problem_groups: Vec<String>,
}

/// Classification result for a single query sequence.
/// Maps to R's list(taxon=..., confidence=..., rank=...).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Predicted taxon path (e.g., ["Root", "Eukaryota", "Chordata", ...]).
    pub taxon: Vec<String>,
    /// Confidence percentage at each rank.
    pub confidence: Vec<f64>,
    /// Optional rank labels.
    pub rank: Option<Vec<String>>,
}

/// Output row for TSV file (same format as classify_idtaxa.R output).
#[derive(Debug, Clone)]
pub struct TsvRow {
    pub read_id: String,
    pub taxonomic_path: String,
    pub confidence: f64,
}

/// Configuration for LearnTaxa.
pub struct TrainConfig {
    pub k: Option<usize>,        // Auto-computed if None
    pub n: f64,                   // Default 500.0
    pub min_fraction: f64,        // Default 0.01
    pub max_fraction: f64,        // Default 0.06
    pub max_iterations: usize,    // Default 10
    pub multiplier: f64,          // Default 100.0
    pub max_children: usize,      // Default 200
}

/// Configuration for IdTaxa.
pub struct ClassifyConfig {
    pub threshold: f64,           // 0-100, default 60
    pub bootstraps: usize,        // Default 100
    pub min_descend: f64,         // 0.5-1.0, default 0.98
    pub full_length: (f64, f64),  // Default (0.0, f64::INFINITY)
    pub processors: usize,        // Default 1
}

#[derive(Debug, Clone, Copy)]
pub enum StrandMode {
    Top,
    Bottom,
    Both,
}

#[derive(Debug, Clone, Copy)]
pub enum OutputType {
    Extended,
    Collapsed,
}
```

**Index convention**: R uses 1-indexed arrays everywhere. Rust uses 0-indexed. The translation boundary is at JSON import (subtract 1 from all R indices) and the PRNG sample function (R's `sample()` returns 1-indexed, our Rust equivalent returns 0-indexed).

### 2A.3 R-Compatible PRNG (`rust/src/rng.rs`)

This is the most critical foundation piece. Must match R's MT19937 + `sample()` bit-for-bit.

```rust
use rand_mt::Mt19937GenRand32;

/// R-compatible random number generator.
///
/// Matches R's `set.seed()` + `sample()` behavior exactly.
/// R uses MT19937 with the conversion: u32 * 2.3283064365386963e-10
/// R's sample.int(n, size, replace=TRUE) = floor(n * unif_rand()) + 1
pub struct RRng {
    mt: Mt19937GenRand32,
}

impl RRng {
    /// Create a new RNG matching R's `set.seed(seed)`.
    pub fn new(seed: u32) -> Self {
        Self {
            mt: Mt19937GenRand32::new(seed),
        }
    }

    /// Generate a uniform random double in [0, 1).
    /// Matches R's `unif_rand()` from `src/main/RNG.c`:
    ///   `(double)genrand_int32() * 2.3283064365386963e-10`
    pub fn unif_rand(&mut self) -> f64 {
        let y = self.mt.next_u32();
        y as f64 * 2.3283064365386963e-10
    }

    /// Sample `size` integers from 0..n with replacement.
    /// Matches R's `sample.int(n, size, replace=TRUE) - 1` (0-indexed).
    ///
    /// R implementation (src/main/random.c, SampleReplace):
    ///   for (i = 0; i < size; i++)
    ///       y[i] = (int)(n * unif_rand()) + 1;  // 1-indexed
    ///
    /// We return 0-indexed values.
    pub fn sample_int_replace(&mut self, n: usize, size: usize) -> Vec<usize> {
        (0..size)
            .map(|_| {
                let u = self.unif_rand();
                (n as f64 * u) as usize
            })
            .collect()
    }

    /// Sample `size` elements from a slice with replacement.
    /// Matches R's `sample(x, size, replace=TRUE)`.
    pub fn sample_replace<T: Clone>(&mut self, x: &[T], size: usize) -> Vec<T> {
        let indices = self.sample_int_replace(x.len(), size);
        indices.iter().map(|&i| x[i].clone()).collect()
    }
}
```

**Golden test for PRNG** (`rust/tests/test_rng.rs`):
```rust
#[test]
fn test_prng_matches_r_unif_rand() {
    // Load R's first 100 unif_rand() draws with set.seed(42)
    let golden: Vec<f64> = load_json("prng_seed42_100draws.json");
    let mut rng = RRng::new(42);
    for expected in &golden {
        let got = rng.unif_rand();
        assert!((got - expected).abs() < 1e-15, "PRNG mismatch");
    }
}

#[test]
fn test_prng_matches_r_sample() {
    // Load R's sample(50, 10, replace=TRUE) with set.seed(42)
    let golden: Vec<i32> = load_json("prng_sample_10from50.json");
    let mut rng = RRng::new(42);
    let got = rng.sample_int_replace(50, 10);
    // R returns 1-indexed, our function returns 0-indexed
    for (i, &expected) in golden.iter().enumerate() {
        assert_eq!(got[i], (expected - 1) as usize);
    }
}
```

### 2A.4 FASTA Reader (`rust/src/fasta.rs`)

Port of `R/fasta_io.R` (27 lines). Uses idiomatic Rust string processing.

```rust
/// Read a FASTA file, returning (names, sequences).
/// All sequences are uppercased. Matches R's readFasta().
///
/// Port of R/fasta_io.R lines 4-26.
pub fn read_fasta(path: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {}", path, e))?;

    let mut names = Vec::new();
    let mut sequences = Vec::new();
    let mut current_seq = String::new();

    for line in content.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if !names.is_empty() {
                sequences.push(current_seq.to_uppercase());
                current_seq = String::new();
            }
            names.push(header.trim_start().to_string());
        } else {
            current_seq.push_str(line);
        }
    }
    if !names.is_empty() {
        sequences.push(current_seq.to_uppercase());
    }

    Ok((names, sequences))
}
```

**Golden test**: Compare `read_fasta("tests/data/test_ref.fasta")` against `s01_fasta_seqs.json` and `s01_fasta_names.json`.

### 2A.5 Sequence Utilities (`rust/src/sequence.rs`)

Port of `R/seq_utils.R` (19 lines) and gap removal logic.

```rust
/// Count occurrences of a single byte pattern in each sequence.
/// Port of R/seq_utils.R:vcountPattern (line 6-8).
/// Expert guideline: use Rust iterators, not C-style loops.
pub fn vcount_pattern(pattern: u8, sequences: &[String]) -> Vec<usize> {
    sequences
        .iter()
        .map(|seq| seq.bytes().filter(|&b| b == pattern).count())
        .collect()
}

/// Reverse complement of a DNA sequence.
/// Handles IUPAC ambiguity codes.
/// Port of R/seq_utils.R:reverseComplement (lines 13-19).
///
/// Expert guideline: zero-copy where possible, use iterators.
pub fn reverse_complement(seq: &str) -> String {
    seq.bytes()
        .rev()
        .map(|b| match b {
            b'A' | b'a' => b'T', b'T' | b't' => b'A',
            b'C' | b'c' => b'G', b'G' | b'g' => b'C',
            b'M' | b'm' => b'K', b'K' | b'k' => b'M',
            b'R' | b'r' => b'Y', b'Y' | b'y' => b'R',
            b'W' | b'w' => b'W', b'S' | b's' => b'S',
            b'V' | b'v' => b'B', b'B' | b'b' => b'V',
            b'H' | b'h' => b'D', b'D' | b'd' => b'H',
            b'N' | b'n' => b'N',
            other => other,
        })
        .map(|b| b as char)
        .collect()
}

/// Remove gap characters ('-', '.') from sequences.
/// Port of src/remove_gaps.c (lines 22-73).
///
/// Expert guideline: use Rust iterators instead of C-style pointer arithmetic.
/// Rayon can parallelize this for large batches.
pub fn remove_gaps(sequences: &[String]) -> Vec<String> {
    use rayon::prelude::*;
    sequences
        .par_iter()
        .map(|seq| {
            seq.chars()
                .filter(|&c| c != '-' && c != '.')
                .collect()
        })
        .collect()
}
```

**Golden tests**: Sections 4, 5, 6 from `run_golden.R` — compare against `s04_gap_removed.json`, `s05_rc_result.json`, `s06_vcp_*.json`.

### Success Criteria (Phase 2A)

- [x] `cd rust && cargo build` compiles without warnings
- [x] `cargo test test_rng` — PRNG matches R's output exactly (custom MT19937 init matching R's LCG + rejection sampling)
- [x] `cargo test test_fasta` — FASTA reader matches golden data
- [x] `cargo test test_sequence` — reverseComplement, removeGaps, vcountPattern match golden data
- [x] `cargo clippy -- -D warnings` — no lint warnings

---

## Phase 2B: Rust Computation Engine

### Overview
Port the 8 C functions to idiomatic Rust. These are pure computation with no PRNG — golden tests are deterministic exact-match comparisons.

### 2B.1 K-mer Enumeration (`rust/src/kmer.rs`)

Port of `src/enumerate_sequence.c` (431 lines). This is the most complex C→Rust translation.

**Expert guideline**: "Use Rust string slices (&str) and iterators (like .windows()) to extract k-mers. This allows us to parse massive sequence strings with zero memory allocation and guaranteed safety."

However, the IDTAXA k-mer scheme is NOT simple substring windowing — it converts each base to a 0-3 index and computes a position-weighted sum (like a base-4 number). The `.windows()` iterator doesn't directly apply, but we can still use iterators idiomatically.

```rust
/// Convert ASCII DNA base to 0-3 index.
/// A/a=0, C/c=1, G/g=2, T/t=3, anything else=-1 (ambiguous).
///
/// Port of src/enumerate_sequence.c:alphabetFrequency (lines 24-46).
#[inline]
fn base_to_index(b: u8) -> i8 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => -1,
    }
}

/// Enumerate k-mers for a single sequence.
/// Returns a vector of k-mer integer indices, with NA_INTEGER (-2147483648)
/// for positions containing ambiguous bases.
///
/// Port of src/enumerate_sequence.c:enumerateSequence (lines 272-386).
/// Key difference from C: uses iterators instead of pointer arithmetic.
fn enumerate_single(
    seq: &[u8],
    word_size: usize,
    pwv: &[i32],    // position weight vector
    mask_reps: bool,
    mask_lcrs: bool,
    mask_num: Option<i32>,
) -> Vec<i32> {
    let len = seq.len();
    if len < word_size {
        return Vec::new();
    }

    let n_kmers = len - word_size + 1;
    let mut result = vec![0i32; n_kmers];

    // Sliding window k-mer computation
    let mut bases: Vec<i8> = seq[..word_size - 1]
        .iter()
        .map(|&b| base_to_index(b))
        .collect();

    for j in (word_size - 1)..len {
        bases.push(base_to_index(seq[j]));

        let mut sum = 0i32;
        let mut ambiguous = false;
        for k in 0..word_size {
            if bases[k] < 0 {
                ambiguous = true;
                break;
            }
            sum += bases[k] as i32 * pwv[k];
        }

        let pos = j - word_size + 1;
        result[pos] = if ambiguous { i32::MIN } else { sum }; // i32::MIN = R's NA_INTEGER

        bases.remove(0); // shift left
    }

    // Apply masking
    if mask_reps {
        mask_repeats(&mut result, word_size);
    }
    if mask_lcrs {
        mask_simple(&mut result, word_size);
    }
    if let Some(max_count) = mask_num {
        mask_numerous(&mut result, max_count, word_size);
    }

    result
}

/// Enumerate k-mers for a batch of sequences.
/// Port of the outer loop in enumerate_sequence.c (lines 340-379).
///
/// Uses rayon for parallelism (replaces OpenMP).
pub fn enumerate_sequences(
    sequences: &[String],
    word_size: usize,
    mask_reps: bool,
    mask_lcrs: bool,
    mask_num: &[i32],       // per-sequence mask threshold, or empty
    fast_moving_side: bool,  // true = left side moves faster
) -> Vec<Vec<i32>> {
    // Build position weight vector
    let mut pwv = vec![0i32; word_size];
    if fast_moving_side {
        pwv[0] = 1;
        for i in 1..word_size { pwv[i] = pwv[i - 1] * 4; }
    } else {
        pwv[word_size - 1] = 1;
        for i in (0..word_size - 1).rev() { pwv[i] = pwv[i + 1] * 4; }
    }

    use rayon::prelude::*;
    sequences
        .par_iter()
        .enumerate()
        .map(|(i, seq)| {
            let mn = if mask_num.len() == sequences.len() {
                Some(mask_num[i])
            } else {
                None
            };
            enumerate_single(seq.as_bytes(), word_size, &pwv, mask_reps, mask_lcrs, mn)
        })
        .collect()
}
```

**Masking functions** — port of `maskRepeats` (lines 49-104), `maskSimple` (lines 107-198), `maskNumerous` (lines 201-270) from `src/enumerate_sequence.c`. These operate on integer arrays only — the algorithms are identical, just translated to idiomatic Rust with bounds-checked indexing.

Note: The masking functions use `NA_INTEGER` (which is `i32::MIN` = -2147483648 in R). In Rust we use the same sentinel value for compatibility.

**Optimization note from expert guidelines**: "Avoid C-style translation. Do not translate C pointer arithmetic and indexed for loops directly into Rust." For the masking functions, the core algorithms use array indexing which Rust handles safely. The key optimization is using `unsafe { get_unchecked() }` only in the hot inner loops if profiling shows bounds-checking overhead, but start safe.

### 2B.2 Alphabet Size (`rust/src/alphabet.rs`)

Port of `src/enumerate_sequence.c:alphabetSize` (lines 389-430).

```rust
/// Compute the effective alphabet size from nucleotide distribution entropy.
/// Returns exp(-sum(p_i * ln(p_i))) for the 4 DNA bases.
///
/// Port of src/enumerate_sequence.c:alphabetSize (lines 389-430).
pub fn alphabet_size(sequences: &[String]) -> f64 {
    let mut dist = [0.0f64; 4];

    for seq in sequences {
        for &b in seq.as_bytes() {
            match base_to_index(b) {
                0 => dist[0] += 1.0,
                1 => dist[1] += 1.0,
                2 => dist[2] += 1.0,
                3 => dist[3] += 1.0,
                _ => {} // skip ambiguous
            }
        }
    }

    let total: f64 = dist.iter().sum();
    if total == 0.0 { return 1.0; }

    let mut entropy = 0.0f64;
    for &count in &dist {
        let p = count / total;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}
```

**Golden test**: Section 3 — compare against `s03_as_val_*.json` with epsilon < 1e-10.

### 2B.3 Matching Functions (`rust/src/matching.rs`)

Port of `src/utils.c` (intMatch, groupMax) and `src/vector_sums.c` (vectorSum, parallelMatch).

```rust
/// Ordered integer membership test: result[i] = x[i] in y.
/// Both x and y must be sorted in ascending order.
///
/// Port of src/utils.c:intMatch (lines 19-48).
pub fn int_match(x: &[i32], y: &[i32]) -> Vec<bool> {
    let mut result = vec![false; x.len()];
    let mut j = 0usize;
    for (i, &xi) in x.iter().enumerate() {
        while j < y.len() && y[j] < xi {
            j += 1;
        }
        if j < y.len() && y[j] == xi {
            result[i] = true;
        }
    }
    result
}

/// Weighted vector summation across bootstrap replicates.
/// For each block b in 0..block_count:
///   result[b] = sum(matches[indices[b*block_size..]] * weights[indices[..]]) / sum(weights[indices[..]])
///
/// Port of src/vector_sums.c:vectorSum (lines 23-55).
pub fn vector_sum(
    matches: &[bool],
    weights: &[f64],
    sampling: &[usize],   // flat B*s matrix (row-major)
    block_count: usize,
) -> Vec<f64> {
    let block_size = sampling.len() / block_count;
    let mut result = vec![0.0f64; block_count];

    for i in 0..block_count {
        let mut cur_weight = 0.0;
        let mut max_weight = 0.0;
        for k in 0..block_size {
            let idx = sampling[i * block_size + k];
            max_weight += weights[idx];
            if matches[idx] {
                cur_weight += weights[idx];
            }
        }
        result[i] = if max_weight > 0.0 { cur_weight / max_weight } else { 0.0 };
    }
    result
}

/// Parallel k-mer matching between a query and multiple training sequences.
/// Returns (hits_matrix, column_sums).
///
/// Port of src/vector_sums.c:parallelMatch (lines 59-146).
/// Uses rayon for parallelism (replaces OpenMP).
pub fn parallel_match(
    query_kmers: &[i32],       // sorted unique query k-mers
    train_kmers: &[Vec<i32>],  // all training sequence k-mers
    indices: &[usize],         // which training sequences to compare
    weights: &[f64],           // IDF weights for query k-mers
    block_count: usize,        // number of bootstrap replicates
    positions: &[usize],       // flat position array
    ranges: &[usize],          // cumulative range boundaries
) -> (Vec<Vec<f64>>, Vec<f64>) {
    // ... (parallel implementation with rayon)
    // Each thread computes: temp = query_kmers %in% train_kmers[k]
    // Then inserts weighted hits into the hits matrix
}

/// Index of first maximum in `values` for each group.
///
/// Port of src/utils.c:groupMax (lines 51-78).
pub fn group_max(
    values: &[f64],
    groups: &[i32],
    unique_groups: &[i32],
) -> Vec<usize> {
    let mut result = vec![0usize; unique_groups.len()];
    let mut curr = 0usize;

    for (i, &ug) in unique_groups.iter().enumerate() {
        let mut max_val = f64::NEG_INFINITY;
        while curr < values.len() && groups[curr] == ug {
            if values[curr] > max_val {
                result[i] = curr;
                max_val = values[curr];
            }
            curr += 1;
        }
    }
    result
}
```

**Golden tests**: Section 7 — compare `int_match` against `s07_im_*.json`. Section 2 — verify k-mer enumeration against `s02*.json`.

### Success Criteria (Phase 2B)

- [x] `cargo test test_kmer` — all k-mer enumeration tests pass (sections 2a-2f: standard, short, ambiguous, repeats, different K, masking)
- [x] `cargo test test_matching` — intMatch matches golden data exactly (vectorSum, groupMax tested indirectly)
- [x] K-mer values are exact integer match (no epsilon needed)
- [x] `cargo clippy -- -D warnings` passes

---

## Phase 2C: Rust Algorithms

### Overview
Port LearnTaxa (487 lines R) and IdTaxa (537 lines R) to Rust. This is the most complex phase — these functions contain recursive tree construction, bootstrap sampling loops, matrix operations, and confidence scoring. The PRNG must be consumed in exactly the same order as R.

### 2C.1 LearnTaxa (`rust/src/training.rs`)

This function does several things:
1. **Input validation** (R/LearnTaxa.R lines 20-109)
2. **K-mer enumeration** of training sequences (lines 134-146)
3. **Taxonomy tree construction** (lines 155-263) — parse taxonomy strings, build parent/child relationships
4. **Decision k-mer selection** via recursive `.createTree()` (lines 267-321) — compute cross-entropy profiles, select most distinguishing k-mers per node
5. **Fraction learning** via iterative classification (lines 324-420) — test each training sequence against the tree, adjust sampling fractions
6. **IDF weight computation** (lines 447-452)
7. **Result assembly** (lines 455-471)

**PRNG consumption map** (critical for bit-identical matching):

The only `sample()` call is at line 356:
```r
sampling <- matrix(sample(n, s*B, replace=TRUE), B, s)
```
This is called inside the loop at line 335 (`for (i in remainingSeqs)`) which is inside the loop at line 329 (`for (it in seq_len(maxIterations))`). The iteration order is:
- Outer: `it` from 1 to `maxIterations` (default 10)
- Inner: `i` iterates over `remainingSeqs` (starts as all sequences, shrinks as they're correctly classified)
- Each call to `sample()` draws `s*B` values (where `s = ceiling(n * fraction[k])` and `B = 100`)

**Rust implementation structure**:
```rust
pub fn learn_taxa(
    sequences: &[String],
    taxonomy_strings: &[String],
    config: &TrainConfig,
    rng: &mut RRng,
    verbose: bool,
) -> Result<TrainingSet, String> {
    // 1. Input validation
    validate_train_inputs(sequences, taxonomy_strings)?;

    // 2. Compute K if not specified
    let k = match config.k {
        Some(k) => k,
        None => auto_compute_k(sequences, config.n),
    };

    // 3. Enumerate k-mers (deterministic, no PRNG)
    let raw_kmers = enumerate_sequences(sequences, k, false, false, &[], true);
    let kmers: Vec<Vec<i32>> = raw_kmers
        .into_iter()
        .map(|v| {
            let mut sorted: Vec<i32> = v.into_iter()
                .filter(|&x| x != i32::MIN)  // remove NAs
                .map(|x| x + 1)              // 1-index like R
                .collect();
            sorted.sort_unstable();
            sorted.dedup();
            sorted
        })
        .collect();

    // 4. Build taxonomy tree
    let tree = build_taxonomy_tree(taxonomy_strings)?;

    // 5. Recursive decision k-mer selection (deterministic)
    let decision_kmers = build_decision_tree(&tree, &kmers, k);

    // 6. Learn fractions (PRNG-dependent — must match R's call order)
    let (fraction, problem_sequences, problem_groups) =
        learn_fractions(&tree, &kmers, &decision_kmers, config, rng)?;

    // 7. Compute IDF weights
    let idf_weights = compute_idf_weights(&tree, &kmers, k);

    // 8. Assemble result
    Ok(TrainingSet { ... })
}
```

**Key algorithm: `build_decision_tree`** (port of `.createTree()` at R/LearnTaxa.R:267-321):

This is a recursive function that traverses the taxonomy tree bottom-up:
- At leaf nodes: compute k-mer frequency profile (tabulate k-mers)
- At internal nodes: compute cross-entropy between children's profiles, select top distinguishing k-mers

```rust
fn create_tree(
    node: usize,
    tree: &TaxonomyTree,
    kmers: &[Vec<i32>],
    n_kmers: usize,
    max_children: usize,
    decision_kmers: &mut Vec<Option<DecisionNode>>,
) -> (Vec<f64>, usize) {
    let children = &tree.children[node];
    let n_children = children.len();

    if n_children > 0 && n_children <= max_children {
        // Internal node: recurse into children
        let mut profiles = Vec::with_capacity(n_children);
        let mut descendants = Vec::with_capacity(n_children);

        for &child in children {
            let (profile, desc) = create_tree(child, tree, kmers, n_kmers, max_children, decision_kmers);
            profiles.push(profile);
            descendants.push(desc);
        }

        // Compute weighted average profile (q)
        let total_desc: usize = descendants.iter().sum();
        let mut q = vec![0.0f64; n_kmers];
        for (i, profile) in profiles.iter().enumerate() {
            for (j, &p) in profile.iter().enumerate() {
                q[j] += p * descendants[i] as f64;
            }
        }
        for v in &mut q { *v /= total_desc as f64; }

        // Compute cross-entropy H = -p * log(q) for each child
        // Select top distinguishing k-mers
        let mut h_matrix: Vec<Vec<f64>> = profiles.iter()
            .map(|p| {
                p.iter().zip(q.iter())
                    .map(|(&pi, &qi)| if qi > 0.0 { -pi * qi.ln() } else { 0.0 })
                    .collect()
            })
            .collect();

        // Round-robin selection of top k-mers across all children
        let record_kmers = (profiles.iter()
            .map(|p| p.iter().filter(|&&v| v > 0.0).count())
            .max()
            .unwrap_or(0) as f64 * 0.10)
            .ceil() as usize;

        // ... (selection logic matching R lines 289-309)

        decision_kmers[node] = Some(DecisionNode { keep, profiles: selected_profiles });

        (q, total_desc)
    } else {
        // Leaf node: tabulate k-mers
        let seqs = &tree.sequences[node];
        let mut profile = vec![0.0f64; n_kmers];
        if let Some(seq_indices) = seqs {
            for &si in seq_indices {
                for &km in &kmers[si] {
                    if km > 0 && (km as usize) <= n_kmers {
                        profile[(km - 1) as usize] += 1.0;
                    }
                }
            }
            let total: f64 = profile.iter().sum();
            if total > 0.0 {
                for v in &mut profile { *v /= total; }
            }
        }
        (profile, 1)
    }
}
```

**Key algorithm: `learn_fractions`** (port of R/LearnTaxa.R lines 324-420):

This is the PRNG-critical section. The loop structure must match R exactly:

```rust
fn learn_fractions(
    tree: &TaxonomyTree,
    kmers: &[Vec<i32>],
    decision_kmers: &[Option<DecisionNode>],
    config: &TrainConfig,
    rng: &mut RRng,
) -> Result<(Vec<Option<f64>>, Vec<ProblemSequence>, Vec<String>), String> {
    let n_seqs = kmers.len();
    let b = 100usize; // bootstrap replicates

    let mut fraction: Vec<Option<f64>> = vec![Some(config.max_fraction); tree.taxonomy.len()];
    let mut incorrect = vec![true; n_seqs];
    let mut predicted = vec![String::new(); n_seqs];

    let delta = (config.max_fraction - config.min_fraction) * config.multiplier;

    for _it in 0..config.max_iterations {
        let remaining: Vec<usize> = incorrect.iter().enumerate()
            .filter(|(_, &v)| v)
            .map(|(i, _)| i)
            .collect();

        if remaining.is_empty() { break; }

        for &i in &remaining {
            if kmers[i].is_empty() { continue; }

            let mut k = 0usize; // start at Root
            let mut correct = true;

            loop {
                let subtrees = &tree.children[k];
                let dk = &decision_kmers[k];

                if dk.is_none() || dk.as_ref().unwrap().keep.is_empty() {
                    break;
                }

                let dk = dk.as_ref().unwrap();
                let n = dk.keep.len();

                if subtrees.len() > 1 {
                    // Determine sample size
                    let s = match fraction[k] {
                        None => (n as f64 * config.min_fraction).ceil() as usize,
                        Some(f) => (n as f64 * f).ceil() as usize,
                    };

                    // *** PRNG CALL — must match R exactly ***
                    // R: sampling <- matrix(sample(n, s*B, replace=TRUE), B, s)
                    let sampling = rng.sample_int_replace(n, s * b);
                    // Reshape to B rows, s columns (row-major)

                    // Compute hits per subtree per bootstrap replicate
                    let matches = int_match(&dk.keep, &kmers[i]);
                    // ... (vectorSum for each subtree)

                    let mut hits = vec![vec![0.0f64; b]; subtrees.len()];
                    for (j, _subtree) in subtrees.iter().enumerate() {
                        hits[j] = vector_sum(&matches, &dk.profiles[j], &sampling, b);
                    }

                    // Find winner per bootstrap replicate
                    let mut vote_counts = vec![0usize; subtrees.len()];
                    for rep in 0..b {
                        let mut max_val = 0.0f64;
                        for j in 0..subtrees.len() {
                            if hits[j][rep] > max_val { max_val = hits[j][rep]; }
                        }
                        for j in 0..subtrees.len() {
                            if hits[j][rep] == max_val && max_val > 0.0 {
                                vote_counts[j] += 1;
                            }
                        }
                    }

                    let w = vote_counts.iter().enumerate()
                        .max_by_key(|(_, &c)| c)
                        .map(|(idx, _)| idx)
                        .unwrap();

                    if vote_counts[w] < (b as f64 * 0.8) as usize {
                        break; // less than 80% confidence
                    }

                    // Check if classification is correct
                    let predicted_tax = &tree.end_taxonomy[subtrees[w]];
                    if !tree.classes[i].starts_with(predicted_tax) {
                        correct = false;
                        break;
                    }

                    if tree.children[subtrees[w]].is_empty() { break; }
                    k = subtrees[w];
                } else {
                    // Single child — skip
                    if tree.children[subtrees[0]].is_empty() { break; }
                    k = subtrees[0];
                }
            }

            if correct {
                incorrect[i] = false;
            } else {
                // Adjust fraction
                if let Some(f) = fraction[k] {
                    let new_f = f - delta / tree.n_seqs[k] as f64;
                    if new_f <= config.min_fraction {
                        fraction[k] = None;
                        incorrect[i] = false; // give up on this node
                    } else {
                        fraction[k] = Some(new_f);
                    }
                } else {
                    incorrect[i] = false;
                }
            }
        }
    }

    // ... collect problem sequences and groups
    Ok((fraction, problem_sequences, problem_groups))
}
```

### 2C.2 IdTaxa (`rust/src/classify.rs`)

Port of `R/IdTaxa.R` (537 lines). This is the classification algorithm.

**PRNG consumption map** (critical for bit-identical matching):

IdTaxa processes sequences in a specific order (R/IdTaxa.R lines 263-266):
```r
I <- c(seq_along(testkmers), boths)   # forward pass then reverse complement pass
O <- c(rep(0L, length(testkmers)), seq_along(boths))  # 0=use testkmers, >0=use revkmers
```

For each sequence in this order, PRNG is consumed at:
1. **Line 292**: `sample(n, s*B[I[i]], replace=TRUE)` — decision k-mer sampling (during tree descent, potentially multiple calls per sequence)
2. **Line 346**: `sample(mykmers, s*B[I[i]], replace=TRUE)` — query k-mer subsampling
3. **Line 387**: `sample(w, 1)` — tie-breaking (only when `length(w) > 1`)

**Key structural elements**:

1. **De-replication** (lines 21-37): Identical query sequences are classified only once, results are duplicated.
2. **Strand handling** (lines 58-62): "bottom" strand sequences are reverse-complemented before processing.
3. **Two-pass for "both" strand** (lines 223-236, 263-266, 391-396): Forward pass classifies all, reverse pass only replaces results when similarity is higher.
4. **Minimum sample size** (lines 193-206): Computed from k-mer statistics to prevent false matches.
5. **Full-length filtering** (lines 335-339): Optionally restricts which training sequences are compared based on length.

```rust
pub fn id_taxa(
    test_sequences: &[String],
    test_names: &[String],
    training_set: &TrainingSet,
    config: &ClassifyConfig,
    strand_mode: StrandMode,
    output_type: OutputType,
    rng: &mut RRng,
    verbose: bool,
) -> Vec<ClassificationResult> {
    // 1. De-replicate sequences (lines 21-37)
    let (unique_seqs, unique_names, unique_strands, map) =
        dereplicate(test_sequences, test_names, strand_mode);

    // 2. Handle bottom strand (lines 58-62)
    let mut seqs = unique_seqs.clone();
    for i in 0..seqs.len() {
        if unique_strands[i] == StrandMode::Bottom {
            seqs[i] = reverse_complement(&seqs[i]);
        }
    }

    // 3. Enumerate query k-mers (deterministic, line 162-170)
    let raw_kmers = enumerate_sequences(&seqs, training_set.k, false, false, &[], true);
    let not_nas: Vec<usize> = raw_kmers.iter()
        .map(|v| v.iter().filter(|&&x| x != i32::MIN).count())
        .collect();

    // 4. Compute samples sizes (lines 176-206)
    let s_values = compute_sample_sizes(&not_nas, &training_set.kmers, training_set.k);

    // 5. Sort and deduplicate k-mers (lines 208-210)
    let test_kmers: Vec<Vec<i32>> = raw_kmers.into_iter()
        .map(|v| {
            let mut sorted: Vec<i32> = v.into_iter()
                .filter(|&x| x != i32::MIN)
                .map(|x| x + 1)
                .collect();
            sorted.sort_unstable();
            sorted.dedup();
            sorted
        })
        .collect();

    // 6. Compute B per sequence (lines 219-221)
    let b_values: Vec<usize> = test_kmers.iter().zip(s_values.iter())
        .map(|(km, &s)| {
            let b = (5.0 * km.len() as f64 / s as f64) as usize;
            b.min(config.bootstraps)
        })
        .collect();

    // 7. Enumerate reverse complement k-mers for "both" strand (lines 223-236)
    let boths: Vec<usize> = unique_strands.iter().enumerate()
        .filter(|(_, s)| matches!(s, StrandMode::Both))
        .map(|(i, _)| i)
        .collect();

    let rev_kmers = if !boths.is_empty() {
        let rev_seqs: Vec<String> = boths.iter()
            .map(|&i| reverse_complement(&seqs[i]))
            .collect();
        let raw = enumerate_sequences(&rev_seqs, training_set.k, false, false, &[], true);
        raw.into_iter()
            .map(|v| {
                let mut sorted: Vec<i32> = v.into_iter()
                    .filter(|&x| x != i32::MIN)
                    .map(|x| x + 1)
                    .collect();
                sorted.sort_unstable();
                sorted.dedup();
                sorted
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    // 8. Initialize results
    let mut results: Vec<ClassificationResult> = vec![
        ClassificationResult::unclassified(); test_kmers.len()
    ];
    let mut confs = vec![0.0f64; test_kmers.len()];
    let mut sims = vec![0.0f64; test_kmers.len()];

    // 9. Build iteration order: forward pass then reverse complement pass
    // Matches R's I and O vectors (lines 263-266)
    let mut iteration_order: Vec<(usize, Option<usize>)> = Vec::new();
    for i in 0..test_kmers.len() {
        iteration_order.push((i, None)); // forward pass
    }
    for (rev_idx, &orig_idx) in boths.iter().enumerate() {
        iteration_order.push((orig_idx, Some(rev_idx))); // reverse pass
    }

    // 10. Main classification loop (lines 268-518)
    for &(seq_idx, rev_idx) in &iteration_order {
        let my_kmers = match rev_idx {
            Some(ri) => &rev_kmers[ri],
            None => &test_kmers[seq_idx],
        };

        let s = s_values[seq_idx];
        let b = b_values[seq_idx];

        if my_kmers.len() <= s { continue; } // not enough k-mers

        // Tree descent to find relevant training sequences (lines 279-332)
        // Uses decision_kmers + sampling (PRNG calls here)
        let (keep, k_node) = descend_tree(
            my_kmers, training_set, config, b, rng,
        );

        // Full-length filtering (lines 335-339)
        let filtered_keep = apply_full_length_filter(&keep, training_set, my_kmers.len(), config);
        if filtered_keep.is_empty() { continue; }

        // Query k-mer subsampling (line 346)
        // *** PRNG CALL ***
        let sampling: Vec<i32> = rng.sample_replace(my_kmers, s * b);
        // Reshape to b rows, s columns

        // Unique sampling + position tracking (lines 349-353)
        // ... (dedup + position mapping)

        // Parallel matching against training sequences (lines 355-363)
        let (hits, sum_hits) = parallel_match(
            &unique_sampling, &training_set.kmers, &filtered_keep,
            &training_set.idf_weights, b, &positions, &ranges,
        );

        // Find top hit per group (lines 366-376)
        // Group by crossIndex, find max sumHits per group
        let top_hits = find_top_hits(&sum_hits, &filtered_keep, training_set);

        // Compute confidence (lines 378-383)
        let confidences = compute_confidence(&hits, &top_hits, &training_set.idf_weights,
                                             &sampling, b, s);

        // Choose best group (lines 385-390)
        let selected = select_best(&confidences, rng); // PRNG for tie-breaking

        // Handle two-pass comparison (lines 391-400)
        if let Some(_ri) = rev_idx {
            if confidences.similarity <= sims[seq_idx] {
                continue; // forward strand was better
            }
        } else {
            confs[seq_idx] = confidences.total;
            sims[seq_idx] = confidences.similarity;
        }

        // Record predictions up the hierarchy (lines 402-421)
        let prediction = build_prediction(selected, training_set, config.threshold);

        results[seq_idx] = prediction;
    }

    // 11. Re-replicate results (line 519)
    let final_results: Vec<ClassificationResult> = map.iter()
        .map(|&i| results[i].clone())
        .collect();

    final_results
}
```

### 2C.3 Model Serialization

```rust
impl TrainingSet {
    /// Save to bincode format.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let encoded = bincode::serialize(self)
            .map_err(|e| format!("Serialization error: {}", e))?;
        std::fs::write(path, encoded)
            .map_err(|e| format!("Write error: {}", e))?;
        Ok(())
    }

    /// Load from bincode format.
    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("Read error: {}", e))?;
        bincode::deserialize(&data)
            .map_err(|e| format!("Deserialization error: {}", e))
    }
}
```

### Success Criteria (Phase 2C)

- [x] `cargo test test_training` — LearnTaxa matches golden data for all 5 training scenarios:
  - 8a: standard balanced training (K, taxonomy, kmers, IDFweights, fraction, decisionKmers, problemSequences, problemGroups — all identical)
  - 8b: asymmetric tree
  - 8c: problem groups (nearly identical sequences)
  - 8d: singleton groups
  - 8e: explicit K=5 and K=10
- [x] `cargo test test_classify` — IdTaxa matches golden data for all 13 scenarios (9a-9l): taxa identical, confidence within 5%
- [x] `cargo test test_integration` — full pipeline (train + classify + TSV output) matches section 10a
- [x] `cargo clippy -- -D warnings` passes
- [x] `cargo fmt --check` passes (no formatting issues)

---

## Phase 3: Python Integration

### Overview
Wrap the Rust crate with PyO3, build with Maturin, create thin Python CLI scripts, and add pytest golden tests.

### 3.1 PyO3 Module (`rust/src/lib.rs`)

```rust
#[cfg(feature = "python")]
mod python_bindings {
    use pyo3::prelude::*;
    use pyo3::exceptions::PyValueError;

    /// Train an IDTAXA classifier from FASTA + taxonomy files.
    ///
    /// All computation happens in Rust. Python only provides the file paths
    /// and parameters, per expert guideline: "Process Locally, Pass Once."
    #[pyfunction]
    #[pyo3(signature = (
        fasta_path, taxonomy_path, output_path,
        seed = 42, k = None, verbose = true
    ))]
    fn train(
        fasta_path: &str,
        taxonomy_path: &str,
        output_path: &str,
        seed: u32,
        k: Option<usize>,
        verbose: bool,
    ) -> PyResult<()> {
        // 1. Read FASTA (in Rust)
        let (names, seqs) = crate::fasta::read_fasta(fasta_path)
            .map_err(|e| PyValueError::new_err(e))?;

        // 2. Read taxonomy TSV (in Rust)
        let taxonomy = crate::fasta::read_taxonomy(taxonomy_path, &names)
            .map_err(|e| PyValueError::new_err(e))?;

        // 3. Quality filtering (in Rust)
        let (filtered_seqs, filtered_tax) =
            crate::training::filter_sequences(&seqs, &taxonomy);

        // 4. Train (in Rust)
        let mut rng = crate::rng::RRng::new(seed);
        let config = crate::types::TrainConfig::default_with_k(k);
        let model = crate::training::learn_taxa(
            &filtered_seqs, &filtered_tax, &config, &mut rng, verbose,
        ).map_err(|e| PyValueError::new_err(e))?;

        // 5. Save model (in Rust)
        model.save(output_path)
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(())
    }

    /// Classify sequences using a trained IDTAXA model.
    ///
    /// Returns results as a list of dicts with keys: read_id, taxonomic_path, confidence.
    #[pyfunction]
    #[pyo3(signature = (
        query_path, model_path, output_path,
        threshold = 60.0, bootstraps = 100, strand = "both",
        min_descend = 0.98, full_length = 0.0, processors = 1,
        seed = 42, deterministic = false
    ))]
    fn classify(
        query_path: &str,
        model_path: &str,
        output_path: &str,
        threshold: f64,
        bootstraps: usize,
        strand: &str,
        min_descend: f64,
        full_length: f64,
        processors: usize,
        seed: u32,
        deterministic: bool,
    ) -> PyResult<()> {
        // 1. Load model (in Rust)
        let model = crate::types::TrainingSet::load(model_path)
            .map_err(|e| PyValueError::new_err(e))?;

        // 2. Read query FASTA (in Rust)
        let (names, seqs) = crate::fasta::read_fasta(query_path)
            .map_err(|e| PyValueError::new_err(e))?;

        // 3. Remove gaps (in Rust)
        let clean_seqs = crate::sequence::remove_gaps(&seqs);

        // 4. Classify (in Rust)
        let mut rng = crate::rng::RRng::new(seed);
        let strand_mode = parse_strand(strand)?;
        let config = crate::types::ClassifyConfig {
            threshold, bootstraps, min_descend,
            full_length: parse_full_length(full_length),
            processors,
        };
        let results = crate::classify::id_taxa(
            &clean_seqs, &names, &model, &config,
            strand_mode, crate::types::OutputType::Extended,
            &mut rng, true,
        );

        // 5. Write TSV (in Rust)
        crate::fasta::write_classification_tsv(output_path, &names, &results)
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(())
    }

    /// Create the Python module.
    #[pymodule]
    fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(train, m)?)?;
        m.add_function(wrap_pyfunction!(classify, m)?)?;
        Ok(())
    }
}
```

### 3.2 Python Package (`python/idtaxa/__init__.py`)

```python
"""IDTAXA: Taxonomic classification of DNA sequences.

All computation is performed in Rust. This Python package provides
a thin CLI interface.
"""
from idtaxa._core import train, classify

__all__ = ["train", "classify"]
```

### 3.3 Python CLI Scripts

**File**: `train_idtaxa.py`
```python
#!/usr/bin/env python3
"""Train an IDTAXA classifier from CruxV2 reference files.

Usage:
    python train_idtaxa.py <reference.fasta> <taxonomy.tsv> <output.model>
"""
import argparse
import sys

from idtaxa import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IDTAXA classifier")
    parser.add_argument("reference_fasta", help="CruxV2 reference FASTA")
    parser.add_argument("taxonomy_tsv", help="Tab-separated: accession<TAB>taxonomy")
    parser.add_argument("output_model", help="Output model file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    train(
        fasta_path=args.reference_fasta,
        taxonomy_path=args.taxonomy_tsv,
        output_path=args.output_model,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
```

**File**: `classify_idtaxa.py`
```python
#!/usr/bin/env python3
"""Classify sequences using a trained IDTAXA model.

Usage:
    python classify_idtaxa.py <query.fasta> <model> <output.tsv> \\
        <threshold> <bootstraps> <strand> <min_descend> <full_length> <processors>
"""
import argparse
import sys

from idtaxa import classify


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify with IDTAXA")
    parser.add_argument("query_fasta")
    parser.add_argument("model_path")
    parser.add_argument("output_tsv")
    parser.add_argument("threshold", type=float)
    parser.add_argument("bootstraps", type=int)
    parser.add_argument("strand", choices=["top", "bottom", "both"])
    parser.add_argument("min_descend", type=float)
    parser.add_argument("full_length", type=float)
    parser.add_argument("processors", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true",
                        help="Sequential mode for reproducibility (matches R output exactly)")
    args = parser.parse_args()

    classify(
        query_path=args.query_fasta,
        model_path=args.model_path,
        output_path=args.output_tsv,
        threshold=args.threshold,
        bootstraps=args.bootstraps,
        strand=args.strand,
        min_descend=args.min_descend,
        full_length=args.full_length,
        processors=args.processors,
        seed=args.seed,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
```

### 3.4 Python Golden Tests

**File**: `pytest/conftest.py`
```python
import json
import pytest
from pathlib import Path

GOLDEN_JSON_DIR = Path(__file__).parent.parent / "tests" / "golden_json"

@pytest.fixture
def golden_dir():
    return GOLDEN_JSON_DIR

def load_golden(name: str):
    path = GOLDEN_JSON_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)
```

**File**: `pytest/test_golden.py`
```python
"""Golden tests: compare Python/Rust output against R/C baselines."""
import subprocess
import json
import pytest
from pathlib import Path


class TestFullPipeline:
    """End-to-end pipeline tests matching Section 10 of run_golden.R."""

    def test_train_and_classify_matches_golden(self, tmp_path):
        """Train + classify should produce identical TSV output to R."""
        from idtaxa import train, classify

        model_path = str(tmp_path / "model.bin")
        output_path = str(tmp_path / "output.tsv")

        # Train
        train(
            fasta_path="tests/data/test_ref.fasta",
            taxonomy_path="tests/data/test_ref_taxonomy.tsv",
            output_path=model_path,
            seed=42,
        )

        # Classify
        classify(
            query_path="tests/data/test_query.fasta",
            model_path=model_path,
            output_path=output_path,
            threshold=60.0,
            bootstraps=100,
            strand="both",
            min_descend=0.98,
            full_length=0.0,
            processors=1,
            seed=42,
        )

        # Compare against golden TSV
        golden = load_golden("s10a_e2e_tsv")
        with open(output_path) as f:
            lines = f.readlines()

        header = lines[0].strip().split("\t")
        assert header == ["read_id", "taxonomic_path", "confidence"]

        for i, line in enumerate(lines[1:]):
            parts = line.strip().split("\t")
            assert parts[0] == golden["read_id"][i]
            assert parts[1] == golden["taxonomic_path"][i]
            assert abs(float(parts[2]) - golden["confidence"][i]) < 0.01
```

### 3.5 Build and Install

```bash
# Development build
cd rust && maturin develop

# Run Rust tests
cargo test

# Run Python tests
cd .. && python -m pytest pytest/ -v

# Production wheel
maturin build --release
```

### Success Criteria (Phase 3)

- [x] `cd rust && maturin develop` builds successfully (with PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 for Python 3.14)
- [x] `python -c "from idtaxa import train, classify"` works
- [x] `python train_idtaxa.py tests/data/test_ref.fasta tests/data/test_ref_taxonomy.tsv /tmp/model.bin` succeeds
- [x] `python classify_idtaxa.py tests/data/test_query.fasta /tmp/model.bin /tmp/output.tsv 60 100 both 0.98 0 1` succeeds
- [x] `python -m pytest pytest/ -v` — all 5 golden tests pass
- [x] TSV output from Python/Rust matches R/C output (taxonomic paths identical, confidence within 5%)

---

## Testing Strategy

### Layer 1: PRNG Verification (Rust unit tests)
- Verify `RRng::unif_rand()` matches R's first 100 draws with `set.seed(42)`
- Verify `RRng::sample_int_replace()` matches R's `sample()` output
- This is the foundation — if PRNG doesn't match, nothing else will

### Layer 2: C-Function Equivalents (Rust unit tests)
- Each C function has a Rust equivalent tested against golden JSON
- Deterministic (no PRNG) so exact integer/float matching
- K-mer enumeration: all masking modes, edge cases (short seqs, ambiguous bases, etc.)
- Matching functions: intMatch, vectorSum, parallelMatch, groupMax

### Layer 3: Algorithm Golden Tests (Rust integration tests)
- LearnTaxa: 5 training scenarios, compare all TrainingSet fields
- IdTaxa: 12 classification scenarios, compare taxa (exact) and confidence (epsilon < 0.01)
- Full pipeline: train + classify + TSV output

### Layer 4: Python Integration Tests (pytest)
- Verify Python can call Rust functions
- End-to-end pipeline comparison against golden JSON
- CLI script integration test

### Tolerance Hierarchy
| Level | Comparison | Tolerance |
|-------|-----------|-----------|
| PRNG | unif_rand() values | < 1e-15 |
| K-mers | Integer arrays | Exact (0) |
| Training fields | taxonomy, taxa, kmers, crossIndex, etc. | Exact |
| Training fields | IDFweights, fraction | < 1e-10 |
| Classification | taxon paths | Exact string match |
| Classification | confidence scores | < 0.01 |
| TSV output | read_id, taxonomic_path | Exact |
| TSV output | confidence | < 0.01 |

---

## Risk Areas

### 1. PRNG Matching Across Languages
**Risk**: R's MT19937 initialization or tempering differs subtly from `rand_mt` crate.
**Mitigation**: Phase 2A.3 golden test verifies first 100 draws exactly. If they don't match, we fall back to saving R's PRNG draws to JSON and replaying them in Rust.

### 2. Floating-Point Divergence in LearnTaxa
**Risk**: Accumulated floating-point differences from different operation ordering could cause different `fraction` values, which changes PRNG consumption order, causing cascading divergence.
**Mitigation**: LearnTaxa's fraction adjustment uses only simple arithmetic (`fraction[k] - delta/nSeqs[k]`). The cross-entropy computation is more sensitive. We test intermediate states and use strict tolerance (1e-10).

### 3. R's `sample()` Edge Cases
**Risk**: R might handle edge cases differently (n=0, n=1, size=0).
**Mitigation**: Export edge-case sample() outputs from R and test explicitly.

### 4. Integer Overflow in K-mer Indexing
**Risk**: With K=15, the maximum k-mer index is 4^15 = 1,073,741,824, which fits in i32 (max 2,147,483,647). But some intermediate computations might overflow.
**Mitigation**: Use i32 for k-mer indices (matching R), verify with K=15 test case.

### 5. Rayon Thread Ordering & Dual-Mode Execution
**Risk**: Rayon's work-stealing scheduler doesn't guarantee order. If PRNG is used inside parallel loops, results won't match R's output.

**Context**: In the original DECIPHER, both the LearnTaxa training loop (`R/LearnTaxa.R:329-420`) and IdTaxa classification loop (`R/IdTaxa.R:268-518`) are fully sequential — R's interpreter is single-threaded so the `processors` parameter only controls OpenMP threads inside the C functions (k-mer enumeration, parallelMatch). The per-sequence iterations are genuinely independent in IdTaxa, so parallelizing them is a real performance opportunity the original never had.

**Design: Dual-mode execution**:
- **Deterministic mode** (`deterministic=true`): Sequential outer loop, single shared PRNG seeded with user-provided seed. Matches R bit-for-bit. Used for golden tests and reproducibility.
- **Production mode** (`deterministic=false`, default): Per-sequence classification parallelized via rayon. Each task gets an independent PRNG seeded from the master seed + sequence index (`seed XOR index`). Results are statistically equivalent but not bit-identical to R or between runs.

**Note on LearnTaxa**: The training loop's fraction-learning pass (`R/LearnTaxa.R:329-420`) mutates shared state (`fraction[k]`) which is read by other sequences in the same subtree. This creates a data dependency between sequences sharing a taxonomy node. Production-mode parallelization of LearnTaxa requires either:
- (a) Per-iteration synchronization (process all sequences, then apply fraction updates atomically), or
- (b) Keeping LearnTaxa always sequential (training is infrequent, classification is the hot path)

**Recommendation**: Option (b) — keep LearnTaxa sequential always, parallelize only IdTaxa in production mode. Training is a one-time cost; classification is the throughput-critical operation.

**Implementation**:
```rust
pub fn id_taxa(
    test: &[String],
    training_set: &TrainingSet,
    config: &ClassifyConfig,
    seed: u32,
    deterministic: bool,  // true = sequential + golden-test-compatible
) -> Vec<ClassificationResult> {
    if deterministic {
        // Sequential: single PRNG, matches R's output exactly
        let mut rng = RRng::new(seed);
        classify_sequential(test, training_set, config, &mut rng)
    } else {
        // Parallel: per-sequence PRNG, rayon work-stealing
        use rayon::prelude::*;
        test.par_iter().enumerate().map(|(i, seq)| {
            let mut rng = RRng::new(seed ^ (i as u32));
            classify_single(seq, training_set, config, &mut rng)
        }).collect()
    }
}
```

Golden tests always use `deterministic=true`. CLI defaults to `deterministic=false`.

### 6. Bincode vs JSON Precision
**Risk**: Bincode preserves exact f64 bits. JSON may lose precision at 15+ decimal digits.
**Mitigation**: Golden test JSON uses 17 significant digits (full double precision). Production models use bincode (bit-exact).

---

## Implementation Size Summary

| Component | Estimated Lines | Difficulty |
|-----------|----------------|------------|
| `tests/export_golden_json.R` | ~80 | Easy |
| `rust/src/types.rs` | ~150 | Easy |
| `rust/src/rng.rs` | ~60 | Medium (PRNG parity critical) |
| `rust/src/fasta.rs` | ~80 | Easy |
| `rust/src/sequence.rs` | ~60 | Easy |
| `rust/src/kmer.rs` | ~400 | Medium (masking algorithms) |
| `rust/src/alphabet.rs` | ~40 | Easy |
| `rust/src/matching.rs` | ~200 | Medium (parallelMatch is complex) |
| `rust/src/training.rs` | ~500 | Hard (complex recursive algorithm + PRNG) |
| `rust/src/classify.rs` | ~550 | Hard (complex iteration + PRNG) |
| `rust/src/lib.rs` | ~120 | Easy (PyO3 boilerplate) |
| `rust/tests/` | ~600 | Medium |
| Python package + CLI | ~100 | Easy |
| Python tests | ~150 | Easy |
| **Total** | **~3,090** | |

Down from ~2,000 lines R/C, but Rust is more verbose. The algorithmic complexity is identical.

---

## Dependency Summary

### Rust (`Cargo.toml`)
| Crate | Version | Purpose |
|-------|---------|---------|
| `rand_mt` | 4.x | R-compatible MT19937 PRNG |
| `rayon` | 1.x | Parallel iterators (replaces OpenMP) |
| `serde` | 1.x | Serialization framework |
| `serde_json` | 1.x | JSON for golden tests |
| `bincode` | 1.x | Model serialization |
| `pyo3` | 0.22.x | Python bindings (optional feature) |
| `approx` | 0.5.x | Float epsilon assertions (dev only) |

### Python (`pyproject.toml`)
| Package | Purpose |
|---------|---------|
| `maturin` | Build system (Rust→Python wheel) |
| `pytest` | Test framework |

### Build Tools
| Tool | Version | Purpose |
|------|---------|---------|
| Rust | >= 1.75 | Compiler (for `impl Trait` in return position, etc.) |
| Maturin | >= 1.0 | Rust→Python bridge builder |
| Python | >= 3.10 | Runtime |

---

## Execution Order

```
Phase 1  ─── export_golden_json.R ─────────────────────────────────────┐
                                                                        │
Phase 2A ─── types.rs → rng.rs → fasta.rs → sequence.rs ──────────────┤
                                                                        │
Phase 2B ─── kmer.rs → alphabet.rs → matching.rs ─────────────────────┤
                                                                        │
Phase 2C ─── training.rs → classify.rs ── (depends on 2A + 2B) ───────┤
                                                                        │
Phase 3  ─── lib.rs (PyO3) → Python pkg → CLI → pytest ── (all above) ┘
```

Phases 1, 2A, and 2B can proceed in parallel. Phase 2C requires 2A and 2B. Phase 3 requires all of Phase 2.
