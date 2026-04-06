#!/usr/bin/env Rscript
# Train an IDTAXA classifier from CruxV2 reference files.
#
# Usage:
#   Rscript train_idtaxa.R <reference.fasta> <taxonomy.tsv> <output.rds>
#
# Arguments:
#   reference.fasta  CruxV2 {primer}.fasta reference sequences
#   taxonomy.tsv     Tab-separated: accession<TAB>semicolon_delimited_path
#   output.rds       Path for the trained IDTAXA model (.rds)
#
# The taxonomy file uses CruxV2 format (accession\tpath). This script
# converts to IDTAXA format by prepending "Root; " and replacing ";"
# with "; " (semicolon-space delimiter required by DECIPHER).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript train_idtaxa.R <reference.fasta> <taxonomy.tsv> <output.rds>")
}

fasta_path <- args[1]
tax_path <- args[2]
output_path <- args[3]

# Load local implementation (no external package dependencies)
script_dir <- dirname(normalizePath(sys.frame(1)$ofile %||% "."))
dyn.load(file.path(script_dir, "src", "idtaxa.so"))
for (f in list.files(file.path(script_dir, "R"), pattern = "\\.R$", full.names = TRUE))
  source(f, local = FALSE)

cat("Loading reference sequences:", fasta_path, "\n")
seqs <- readFasta(fasta_path)
cat("Loaded", length(seqs), "sequences\n")

# Filter out sequences that are too short or N-rich to produce valid k-mers.
# DECIPHER's LearnTaxa fails if any internal group contains only k-mer-less
# sequences after subsampling (error: "All training sequences must have at
# least one k-mer").
seq_widths <- nchar(seqs)
n_counts <- vcountPattern("N", seqs, fixed = TRUE)
n_fracs <- n_counts / seq_widths
min_length <- 30
max_n_frac <- 0.3

keep_quality <- (seq_widths >= min_length) & (n_fracs <= max_n_frac)
n_removed <- sum(!keep_quality)
if (n_removed > 0) {
  cat("Removed", n_removed, "sequences: too short (<", min_length,
      "bp) or N-rich (>", max_n_frac * 100, "% N)\n")
  seqs <- seqs[keep_quality]
}
cat("Retained", length(seqs), "sequences after quality filter\n")

# Load taxonomy mapping
cat("Loading taxonomy:", tax_path, "\n")
tax_map <- read.delim(tax_path, header = FALSE, stringsAsFactors = FALSE,
                       col.names = c("accession", "taxonomy"))

# Match taxonomy to sequences by accession
seq_names <- sub(" .*", "", names(seqs))  # extract accession from header
idx <- match(seq_names, tax_map$accession)

if (any(is.na(idx))) {
  n_missing <- sum(is.na(idx))
  cat("WARNING:", n_missing, "sequences have no taxonomy mapping\n")
  # Remove unmatched sequences
  keep <- !is.na(idx)
  seqs <- seqs[keep]
  seq_names <- seq_names[keep]
  idx <- idx[keep]
}

# Convert CruxV2 format to IDTAXA format:
# "Eukaryota;Chordata;Mammalia" -> "Root; Eukaryota; Chordata; Mammalia"
raw_taxonomy <- tax_map$taxonomy[idx]
taxonomy <- paste0("Root; ", gsub(";", "; ", raw_taxonomy))

# Filter out sequences with shallow taxonomy (< 4 ranks including Root).
# DECIPHER's LearnTaxa fails when subtrees contain only shallow entries.
rank_counts <- sapply(strsplit(taxonomy, "; "), length)
keep_deep <- rank_counts >= 4
n_shallow <- sum(!keep_deep)
if (n_shallow > 0) {
  cat("Removed", n_shallow, "sequences with fewer than 4 taxonomy ranks\n")
  seqs <- seqs[keep_deep]
  taxonomy <- taxonomy[keep_deep]
}

cat("Training IDTAXA classifier on", length(seqs), "sequences...\n")
trainingSet <- LearnTaxa(
  train = seqs,
  taxonomy = taxonomy,
  verbose = TRUE
)

cat("Saving trained model:", output_path, "\n")
saveRDS(trainingSet, output_path)
cat("Training complete.\n")
