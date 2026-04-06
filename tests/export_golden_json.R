#!/usr/bin/env Rscript
# Convert golden .rds files to JSON for Rust tests.
# Requires: jsonlite
#
# Usage: Rscript tests/export_golden_json.R

library(jsonlite)

golden_dir <- "tests/golden"
json_dir <- "tests/golden_json"
dir.create(json_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Exporting golden .rds files to JSON ===\n\n")

# ---------------------------------------------------------------------------
# Helper: write JSON with full double-precision fidelity
# jsonlite converts NULL list elements to {} instead of null.
# We post-process to fix this: write with a sentinel, then replace.
# ---------------------------------------------------------------------------
write_golden_json <- function(obj, name, auto_unbox = TRUE) {
  path <- file.path(json_dir, paste0(name, ".json"))
  json_str <- toJSON(obj, auto_unbox = auto_unbox, na = "null", digits = 17,
                     pretty = TRUE, null = "null")
  writeLines(json_str, path)
}

# ---------------------------------------------------------------------------
# Helper: serialize a TrainingSet (R list with class c("Taxa", "Train"))
# decisionKmers needs special handling: list of list(keep, profile_matrix)
# ---------------------------------------------------------------------------
serialize_training_set <- function(ts) {
  result <- list()
  result$taxonomy   <- ts$taxonomy
  result$taxa       <- ts$taxa
  result$ranks      <- ts$ranks  # NULL → null in JSON
  result$levels     <- ts$levels
  result$K          <- ts$K

  # children: list of integer vectors → list of integer arrays
  # Use I() to prevent auto_unbox from turning [2] into 2
  result$children <- lapply(ts$children, function(x) {
    if (is.null(x) || length(x) == 0) I(integer(0)) else I(as.integer(x))
  })

  result$parents     <- as.integer(ts$parents)
  result$crossIndex  <- as.integer(ts$crossIndex)

  # fraction: may contain NAs (problem groups)
  result$fraction <- lapply(ts$fraction, function(x) {
    if (is.na(x)) NULL else x
  })

  # sequences: list of integer vectors (or NULL)
  # Use I() to prevent auto_unbox of single-element vectors
  result$sequences <- lapply(ts$sequences, function(x) {
    if (is.null(x) || length(x) == 0) NULL else I(as.integer(x))
  })

  # kmers: list of integer vectors — use I() to prevent auto_unbox
  result$kmers <- lapply(ts$kmers, function(x) I(as.integer(x)))

  result$IDFweights <- as.numeric(ts$IDFweights)

  # decisionKmers: list, each element is NULL or list(keep_int_vec, profile_matrix)
  result$decisionKmers <- lapply(ts$decisionKmers, function(dk) {
    if (is.null(dk)) return(NULL)
    keep <- I(as.integer(dk[[1]]))  # I() prevents auto_unbox
    prof <- dk[[2]]
    # Profile is a matrix: rows=subtrees, cols=kmers. Convert to list of rows.
    if (is.matrix(prof)) {
      profile_rows <- lapply(seq_len(nrow(prof)), function(r) as.numeric(prof[r, ]))
    } else {
      # Single-row case: prof might be a numeric vector
      profile_rows <- list(as.numeric(prof))
    }
    list(keep = keep, profiles = profile_rows)
  })

  # problemSequences: data.frame with Index, Expected, Predicted
  if (!is.null(ts$problemSequences) && nrow(ts$problemSequences) > 0) {
    result$problemSequences <- lapply(seq_len(nrow(ts$problemSequences)), function(i) {
      list(
        index     = as.integer(ts$problemSequences$Index[i]),
        expected  = ts$problemSequences$Expected[i],
        predicted = ts$problemSequences$Predicted[i]
      )
    })
  } else {
    result$problemSequences <- list()
  }

  result$problemGroups <- ts$problemGroups

  result
}

# ---------------------------------------------------------------------------
# Helper: serialize IdTaxa classification results
# (list of lists, each with $taxon and $confidence)
# ---------------------------------------------------------------------------
serialize_classification <- function(ids) {
  cls <- class(ids)
  if ("character" %in% cls) {
    # Collapsed format: just a character vector
    return(as.character(ids))
  }
  lapply(ids, function(x) {
    list(taxon = x$taxon, confidence = x$confidence)
  })
}

# ---------------------------------------------------------------------------
# Generic .rds → JSON export
# ---------------------------------------------------------------------------
rds_files <- list.files(golden_dir, pattern = "\\.rds$", full.names = TRUE)
cat("Found", length(rds_files), ".rds files\n")

for (f in rds_files) {
  name <- sub("\\.rds$", "", basename(f))
  obj <- readRDS(f)
  cls <- class(obj)

  if ("Taxa" %in% cls && "Train" %in% cls) {
    # TrainingSet
    write_golden_json(serialize_training_set(obj), name)
    cat("  [TrainingSet] ", name, "\n")

  } else if ("Taxa" %in% cls && "Test" %in% cls) {
    # Classification result (extended format)
    write_golden_json(serialize_classification(obj), name)
    cat("  [Classification] ", name, "\n")

  } else if (is.character(obj) && length(cls) == 1 && cls == "character" &&
             grepl("collapsed", name)) {
    # Collapsed classification (character vector)
    write_golden_json(as.character(obj), name)
    cat("  [Collapsed] ", name, "\n")

  } else if (is.data.frame(obj)) {
    # Data frame (e.g., e2e TSV)
    write_golden_json(obj, name)
    cat("  [DataFrame] ", name, "\n")

  } else if (is.list(obj) && !is.data.frame(obj)) {
    # List of lists (e.g., intMatch cases, k-mer results)
    if (!is.null(names(obj)) && all(sapply(obj, is.list))) {
      # Named list of lists (intMatch cases) - disable auto_unbox to preserve arrays
      write_golden_json(obj, name, auto_unbox = FALSE)
    } else if (all(sapply(obj, is.integer))) {
      # List of integer vectors (k-mer results) — replace NA with sentinel
      # R's NA_integer_ IS -2147483648, so we must convert to double first
      # Use auto_unbox=FALSE to preserve single-element vectors as arrays
      write_golden_json(lapply(obj, function(x) {
        x <- as.numeric(x)
        x[is.na(x)] <- -2147483648
        x
      }), name, auto_unbox = FALSE)
    } else {
      write_golden_json(obj, name)
    }
    cat("  [List] ", name, "\n")

  } else if (is.integer(obj)) {
    # Integer vector - convert NAs to sentinel for k-mer compatibility
    obj[is.na(obj)] <- -2147483648L
    write_golden_json(as.integer(obj), name, auto_unbox = FALSE)
    cat("  [IntVec] ", name, "\n")

  } else if (is.logical(obj)) {
    # Logical vector - disable auto_unbox to preserve single-element arrays
    write_golden_json(obj, name, auto_unbox = FALSE)
    cat("  [LogicalVec] ", name, "\n")

  } else {
    # Character vector, numeric vector, etc.
    write_golden_json(obj, name)
    cat("  [Other] ", name, "\n")
  }
}

# ---------------------------------------------------------------------------
# PRNG verification data
# ---------------------------------------------------------------------------
cat("\n== Generating PRNG verification data ==\n")

# First 100 unif_rand() draws with set.seed(42)
set.seed(42)
prng_100 <- runif(100)
write_golden_json(prng_100, "prng_seed42_100draws")
cat("  prng_seed42_100draws.json\n")

# sample() verification
set.seed(42)
sample_10_from_50 <- sample(50L, 10L, replace = TRUE)
write_golden_json(as.integer(sample_10_from_50), "prng_sample_10from50")
cat("  prng_sample_10from50.json\n")

set.seed(42)
sample_100_from_1000 <- sample(1000L, 100L, replace = TRUE)
write_golden_json(as.integer(sample_100_from_1000), "prng_sample_100from1000")
cat("  prng_sample_100from1000.json\n")

# Edge cases for sample
set.seed(42)
sample_1_from_1 <- sample(1L, 1L, replace = TRUE)
write_golden_json(as.integer(sample_1_from_1), "prng_sample_1from1")
cat("  prng_sample_1from1.json\n")

set.seed(42)
sample_0_from_5 <- sample(5L, 0L, replace = TRUE)
write_golden_json(as.integer(sample_0_from_5), "prng_sample_0from5")
cat("  prng_sample_0from5.json\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
json_files <- list.files(json_dir, pattern = "\\.json$")
cat("\n=== Export complete ===\n")
cat("Total JSON files:", length(json_files), "\n")
