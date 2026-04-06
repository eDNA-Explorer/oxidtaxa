#!/usr/bin/env Rscript
# Time ONLY the classify step using a pre-built model (no training overhead)
script_dir <- "."
dyn.load(file.path(script_dir, "src", "idtaxa.so"))
for (f in list.files(file.path(script_dir, "R"), pattern = "\\.R$", full.names = TRUE))
  source(f, local = FALSE)

args <- commandArgs(trailingOnly = TRUE)
model_path <- args[1]
query_path <- args[2]

cat("Loading model:", model_path, "\n")
ts <- readRDS(model_path)
cat("K =", ts$K, ", nodes =", length(ts$taxonomy), "\n")

query <- readFasta(query_path)
query <- RemoveGaps(query)
cat("Query:", length(query), "sequences\n")

t1 <- proc.time()
set.seed(42)
ids <- IdTaxa(test = query, trainingSet = ts, type = "extended", strand = "both",
  threshold = 40, bootstraps = 50, minDescend = 0.98, fullLength = 0, processors = 1, verbose = FALSE)
t2 <- proc.time()
cat("Classify time:", (t2 - t1)[3], "s\n")
