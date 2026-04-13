#!/usr/bin/env Rscript
# Generate R/C baseline classifications for the 1K benchmark dataset.
# Outputs JSON in the same format as eval_training.rs --save-baseline.
#
# Usage: Rscript benchmarks/generate_baseline.R

ref_dir <- file.path("reference")
dyn.load(file.path(ref_dir, "c_source", "idtaxa.so"))
for (f in list.files(file.path(ref_dir, "r_source"), pattern = "\\.R$", full.names = TRUE))
  source(f, local = FALSE)

# Load data
cat("Loading 1K benchmark data...\n")
seqs <- readFasta("benchmarks/data/bench_1000_ref.fasta")
cat("Loaded", length(seqs), "reference sequences\n")

tax_map <- read.delim("benchmarks/data/bench_1000_ref_taxonomy.tsv",
                       header = FALSE, stringsAsFactors = FALSE,
                       col.names = c("accession", "taxonomy"))
seq_names <- sub(" .*", "", names(seqs))
idx <- match(seq_names, tax_map$accession)
taxonomy <- paste0("Root; ", gsub(";", "; ", tax_map$taxonomy[idx]))

# Quality filter (same as train_idtaxa.R / Rust filter_for_training)
rank_counts <- sapply(strsplit(taxonomy, "; "), length)
keep <- rank_counts >= 4 & nchar(seqs) >= 30
n_frac <- vcountPattern("N", seqs) / nchar(seqs)
keep <- keep & n_frac <= 0.3
seqs <- seqs[keep]
taxonomy <- taxonomy[keep]
cat("After filtering:", length(seqs), "sequences\n")

# Train
cat("Training...\n")
set.seed(42)
ts <- LearnTaxa(train = seqs, taxonomy = taxonomy, verbose = FALSE)
cat("K =", ts$K, ", nodes =", length(ts$taxonomy), "\n")

problem_seqs <- length(ts$problemSequences)
problem_groups <- length(ts$problemGroups)
cat("Problem sequences:", problem_seqs, "\n")
cat("Problem groups:", problem_groups, "\n")

# Classify with defaults: threshold=60, bootstraps=100, strand=both
cat("Loading queries...\n")
query <- readFasta("benchmarks/data/bench_1000_query.fasta")
query <- RemoveGaps(query)
cat("Loaded", length(query), "query sequences\n")

cat("Classifying (threshold=60, bootstraps=100, strand=both, processors=1)...\n")
set.seed(42)
ids <- IdTaxa(test = query, trainingSet = ts, type = "extended", strand = "both",
  threshold = 60, bootstraps = 100, minDescend = 0.98, fullLength = 0,
  processors = 1, verbose = FALSE)
cat("Classified", length(ids), "sequences\n")

# Convert to baseline JSON
results <- vector("list", length(ids))
for (i in seq_along(ids)) {
  x <- ids[[i]]
  taxa <- x$taxon
  conf <- x$confidence

  if (length(taxa) > 1) {
    taxa <- taxa[-1]  # skip Root
    conf <- conf[-1]
    classified <- !startsWith(taxa, "unclassified_")
    taxa <- taxa[classified]
    conf <- conf[classified]
  } else {
    taxa <- character(0)
    conf <- numeric(0)
  }

  if (length(taxa) > 0) {
    results[[i]] <- list(
      path = paste(taxa, collapse = ";"),
      confidence = min(conf)
    )
  } else {
    # For unclassified: report the raw confidence of the deepest attempted rank
    results[[i]] <- list(
      path = "",
      confidence = min(x$confidence)
    )
  }
}

baseline <- list(
  n_queries = length(ids),
  problem_sequences = problem_seqs,
  problem_groups = problem_groups,
  results = results
)

output_path <- "benchmarks/baselines/baseline_1k.json"
cat("Writing", output_path, "\n")
json <- jsonlite::toJSON(baseline, auto_unbox = TRUE, pretty = TRUE, digits = 17)
writeLines(json, output_path)
cat("Done.\n")
