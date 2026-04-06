#!/usr/bin/env Rscript
script_dir <- "."
dyn.load(file.path(script_dir, "src", "idtaxa.so"))
for (f in list.files(file.path(script_dir, "R"), pattern = "\\.R$", full.names = TRUE))
  source(f, local = FALSE)

args <- commandArgs(trailingOnly = TRUE)
fasta <- args[1]
tax_file <- args[2]

seqs <- readFasta(fasta)
cat("Loaded", length(seqs), "sequences\n")

tax_map <- read.delim(tax_file, header = FALSE, col.names = c("acc", "tax"), stringsAsFactors = FALSE)
idx <- match(names(seqs), tax_map$acc)
taxonomy_vec <- paste0("Root; ", gsub(";", "; ", tax_map$tax[idx]))

rank_counts <- sapply(strsplit(taxonomy_vec, "; "), length)
keep <- rank_counts >= 4 & nchar(seqs) >= 30
n_frac <- vcountPattern("N", seqs) / nchar(seqs)
keep <- keep & n_frac <= 0.3
seqs <- seqs[keep]
taxonomy_vec <- taxonomy_vec[keep]
cat("After filter:", length(seqs), "sequences\n")

t1 <- proc.time()
set.seed(42)
ts <- LearnTaxa(train = seqs, taxonomy = taxonomy_vec, verbose = FALSE)
t2 <- proc.time()
cat("Train time:", (t2 - t1)[3], "s, K =", ts$K, ", nodes =", length(ts$taxonomy), "\n")

# Classify
query <- readFasta("benchmarks/data/bench_1000_query.fasta")
query <- RemoveGaps(query)
t1 <- proc.time()
set.seed(42)
ids <- IdTaxa(test = query, trainingSet = ts, type = "extended", strand = "both",
  threshold = 40, bootstraps = 50, minDescend = 0.98, fullLength = 0, processors = 1, verbose = FALSE)
t2 <- proc.time()
cat("Classify time:", (t2 - t1)[3], "s,", length(ids), "sequences\n")
