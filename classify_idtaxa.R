#!/usr/bin/env Rscript
# Classify sequences using a trained IDTAXA model.
#
# Usage:
#   Rscript classify_idtaxa.R <query.fasta> <model.rds> <output.tsv> \
#       <threshold> <bootstraps> <strand> <min_descend> <full_length> <processors>
#
# Arguments:
#   query.fasta   Query FASTA sequences to classify
#   model.rds     Trained IDTAXA model from train_idtaxa.R
#   output.tsv    Output TSV with columns: read_id, taxonomic_path, confidence
#   threshold     Confidence threshold (0-100), ranks below this are dropped
#   bootstraps    Number of bootstrap replicates (default 100)
#   strand        Strand(s) to search: "top", "bottom", or "both"
#   min_descend   Fraction of bootstraps that must agree before descending (0-1)
#   full_length   Length filter for training sequences (0 = disabled)
#   processors    Number of parallel R processors for IdTaxa (>= 1)
#
# Output format (tab-separated with header):
#   read_id          taxonomic_path                                    confidence
#   asv_001          Eukaryota;Chordata;Mammalia;Carnivora;Canidae     85.2
#   asv_002          Eukaryota;Chordata;Actinopteri                    62.0

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 9) {
  stop("Usage: Rscript classify_idtaxa.R <query.fasta> <model.rds> <output.tsv> <threshold> <bootstraps> <strand> <min_descend> <full_length> <processors>")
}

query_fasta <- args[1]
model_path <- args[2]
output_tsv <- args[3]
threshold <- as.numeric(args[4])
bootstraps <- as.integer(args[5])
strand <- args[6]
min_descend <- as.numeric(args[7])
full_length <- as.numeric(args[8])
processors <- as.integer(args[9])

# Load local implementation (no external package dependencies)
script_dir <- dirname(normalizePath(sys.frame(1)$ofile %||% "."))
dyn.load(file.path(script_dir, "src", "idtaxa.so"))
for (f in list.files(file.path(script_dir, "R"), pattern = "\\.R$", full.names = TRUE))
  source(f, local = FALSE)

cat("Loading query sequences:", query_fasta, "\n")
seqs <- readFasta(query_fasta)
seqs <- RemoveGaps(seqs)
cat("Loaded", length(seqs), "query sequences\n")

cat("Loading trained model:", model_path, "\n")
trainingSet <- readRDS(model_path)

cat("Classifying with threshold =", threshold, ", bootstraps =", bootstraps,
    ", strand =", strand, ", minDescend =", min_descend,
    ", fullLength =", full_length, "...\n")
ids <- IdTaxa(
  test = seqs,
  trainingSet = trainingSet,
  type = "extended",
  strand = strand,
  threshold = threshold,
  bootstraps = bootstraps,
  minDescend = min_descend,
  fullLength = full_length,
  processors = processors
)

# Convert to data frame
cat("Converting results to TSV...\n")
results <- data.frame(
  read_id = character(length(ids)),
  taxonomic_path = character(length(ids)),
  confidence = numeric(length(ids)),
  stringsAsFactors = FALSE
)

for (i in seq_along(ids)) {
  x <- ids[[i]]
  # Extract read ID from sequence name (first whitespace-delimited word)
  results$read_id[i] <- sub(" .*", "", names(seqs)[i])

  # Get taxon names and confidences, excluding "Root" (always first)
  taxa <- x$taxon
  conf <- x$confidence

  if (length(taxa) > 1) {
    # Skip Root (index 1)
    taxa <- taxa[-1]
    conf <- conf[-1]

    # Filter out unclassified entries
    classified <- !startsWith(taxa, "unclassified_")
    taxa <- taxa[classified]
    conf <- conf[classified]
  } else {
    taxa <- character(0)
    conf <- numeric(0)
  }

  if (length(taxa) > 0) {
    results$taxonomic_path[i] <- paste(taxa, collapse = ";")
    results$confidence[i] <- min(conf)
  } else {
    results$taxonomic_path[i] <- ""
    results$confidence[i] <- 0.0
  }
}

cat("Writing output:", output_tsv, "\n")
write.table(results, file = output_tsv, sep = "\t", row.names = FALSE,
            quote = FALSE)
cat("Classification complete:", nrow(results), "sequences classified.\n")
