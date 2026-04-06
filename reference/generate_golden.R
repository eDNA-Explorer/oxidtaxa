#!/usr/bin/env Rscript
# Generate golden reference outputs using Bioconductor packages.
# Run ONCE while DECIPHER and Biostrings are still installed.
#
# Usage: Rscript tests/generate_golden.R

suppressPackageStartupMessages(library(DECIPHER))
suppressPackageStartupMessages(library(Biostrings))

cat("=== Generating golden reference outputs ===\n\n")

golden_dir <- "tests/golden"
dir.create(golden_dir, showWarnings = FALSE, recursive = TRUE)

save_golden <- function(obj, name) {
  saveRDS(obj, file.path(golden_dir, paste0(name, ".rds")))
}

# ============================================================================
# SECTION 1: FASTA Reading
# ============================================================================
cat("== Section 1: FASTA Reading ==\n")

seqs_raw <- readDNAStringSet("tests/data/test_ref.fasta")
save_golden(as.character(seqs_raw), "s01_fasta_seqs")
save_golden(names(seqs_raw), "s01_fasta_names")

# ============================================================================
# SECTION 2: K-mer Enumeration Edge Cases
# ============================================================================
cat("== Section 2: K-mer Enumeration ==\n")

# 2a. Standard sequences at different lengths
std_seqs <- DNAStringSet(c(
  seq_250bp = paste(sample(c("A","C","G","T"), 250, replace=TRUE), collapse=""),
  seq_1500bp = paste(sample(c("A","C","G","T"), 1500, replace=TRUE), collapse="")
))
set.seed(100)  # reproducible for any masking
kmers_std <- .Call("enumerateSequence", std_seqs, 8L, FALSE, FALSE, integer(), 1L, 1L, PACKAGE="DECIPHER")
save_golden(as.character(std_seqs), "s02a_std_seqs")
save_golden(kmers_std, "s02a_std_kmers")

# 2b. Sequences shorter than K (K=8)
short_seqs <- DNAStringSet(c(
  len_0  = "",
  len_1  = "A",
  len_3  = "ACG",
  len_7  = "ACGTACG",
  len_8  = "ACGTACGT",     # exactly K
  len_9  = "ACGTACGTA"     # K+1
))
kmers_short <- .Call("enumerateSequence", short_seqs, 8L, FALSE, FALSE, integer(), 1L, 1L, PACKAGE="DECIPHER")
save_golden(as.character(short_seqs), "s02b_short_seqs")
save_golden(kmers_short, "s02b_short_kmers")

# 2c. Ambiguous bases (N-heavy sequences)
ambig_seqs <- DNAStringSet(c(
  all_N      = paste(rep("N", 100), collapse=""),
  mostly_N   = paste(c(rep("N", 90), rep("A", 5), rep("C", 5)), collapse=""),
  scattered_N = paste(sample(c("A","C","G","T","N"), 200, replace=TRUE, prob=c(.2,.2,.2,.2,.2)), collapse=""),
  few_N      = paste(c("ACGTACGTACGT", "N", "ACGTACGTACGT", "N", "ACGTACGT"), collapse=""),
  no_N       = paste(sample(c("A","C","G","T"), 200, replace=TRUE), collapse="")
))
kmers_ambig <- .Call("enumerateSequence", ambig_seqs, 8L, FALSE, FALSE, integer(), 1L, 1L, PACKAGE="DECIPHER")
save_golden(as.character(ambig_seqs), "s02c_ambig_seqs")
save_golden(kmers_ambig, "s02c_ambig_kmers")

# 2d. Homopolymers and simple repeats
repeat_seqs <- DNAStringSet(c(
  poly_A   = paste(rep("A", 200), collapse=""),
  poly_C   = paste(rep("C", 200), collapse=""),
  poly_G   = paste(rep("G", 200), collapse=""),
  poly_T   = paste(rep("T", 200), collapse=""),
  dinuc_AT = paste(rep("AT", 100), collapse=""),
  dinuc_GC = paste(rep("GC", 100), collapse=""),
  trinuc   = paste(rep("ACG", 67), collapse=""),
  quad     = paste(rep("ACGT", 50), collapse="")
))
kmers_repeat <- .Call("enumerateSequence", repeat_seqs, 8L, FALSE, FALSE, integer(), 1L, 1L, PACKAGE="DECIPHER")
save_golden(as.character(repeat_seqs), "s02d_repeat_seqs")
save_golden(kmers_repeat, "s02d_repeat_kmers")

# 2e. Different K values
set.seed(200)
k_test_seq <- DNAStringSet(paste(sample(c("A","C","G","T"), 300, replace=TRUE), collapse=""))
names(k_test_seq) <- "test_seq"
k_test_char <- as.character(k_test_seq)
save_golden(k_test_char, "s02e_ktest_seq")
for (k_val in c(1L, 3L, 5L, 8L, 10L, 13L, 15L)) {
  km <- .Call("enumerateSequence", k_test_seq, k_val, FALSE, FALSE, integer(), 1L, 1L, PACKAGE="DECIPHER")
  save_golden(km, paste0("s02e_kmers_K", k_val))
}

# 2f. K-mer masking options (repeats, low-complexity)
set.seed(201)
mask_seq <- DNAStringSet(c(
  normal   = paste(sample(c("A","C","G","T"), 500, replace=TRUE), collapse=""),
  low_comp = paste(c(rep("A", 100), sample(c("A","C","G","T"), 400, replace=TRUE)), collapse=""),
  repetitive = paste(rep("ACGTACGT", 62), collapse="")
))
mask_seq_char <- as.character(mask_seq)
save_golden(mask_seq_char, "s02f_mask_seqs")
# No masking
km_nomask <- .Call("enumerateSequence", mask_seq, 8L, FALSE, FALSE, integer(), 1L, 1L, PACKAGE="DECIPHER")
save_golden(km_nomask, "s02f_kmers_nomask")
# With repeat masking
km_maskrep <- .Call("enumerateSequence", mask_seq, 8L, TRUE, FALSE, integer(), 1L, 1L, PACKAGE="DECIPHER")
save_golden(km_maskrep, "s02f_kmers_maskrep")
# With LCR masking
km_masklcr <- .Call("enumerateSequence", mask_seq, 8L, FALSE, TRUE, integer(), 1L, 1L, PACKAGE="DECIPHER")
save_golden(km_masklcr, "s02f_kmers_masklcr")
# With both
km_maskboth <- .Call("enumerateSequence", mask_seq, 8L, TRUE, TRUE, integer(), 1L, 1L, PACKAGE="DECIPHER")
save_golden(km_maskboth, "s02f_kmers_maskboth")

# ============================================================================
# SECTION 3: alphabetSize
# ============================================================================
cat("== Section 3: alphabetSize ==\n")

as_seqs <- list(
  uniform    = DNAStringSet(paste(rep(c("A","C","G","T"), each=100), collapse="")),
  biased_A   = DNAStringSet(paste(sample(c("A","C","G","T"), 400, replace=TRUE, prob=c(.7,.1,.1,.1)), collapse="")),
  biased_GC  = DNAStringSet(paste(sample(c("A","C","G","T"), 400, replace=TRUE, prob=c(.1,.4,.4,.1)), collapse="")),
  all_A      = DNAStringSet(paste(rep("A", 200), collapse="")),
  many_seqs  = DNAStringSet(as.character(seqs_raw[1:20]))
)
for (nm in names(as_seqs)) {
  val <- .Call("alphabetSize", as_seqs[[nm]], PACKAGE="DECIPHER")
  save_golden(as.character(as_seqs[[nm]]), paste0("s03_as_seqs_", nm))
  save_golden(val, paste0("s03_as_val_", nm))
}

# ============================================================================
# SECTION 4: RemoveGaps Edge Cases
# ============================================================================
cat("== Section 4: RemoveGaps ==\n")

gap_seqs <- DNAStringSet(c(
  no_gaps     = "ACGTACGTACGTACGT",
  all_dash    = "----------------",
  all_dot     = "................",
  mixed       = "AC-GT.AC-GT.ACGT",
  start_gap   = "-ACGTACGT",
  end_gap     = "ACGTACGT-",
  single_base = "A",
  gap_base    = "-A-",
  only_gaps   = "---...---"
))
gap_removed <- as.character(RemoveGaps(gap_seqs))
save_golden(as.character(gap_seqs), "s04_gap_seqs")
save_golden(gap_removed, "s04_gap_removed")

# ============================================================================
# SECTION 5: reverseComplement Edge Cases
# ============================================================================
cat("== Section 5: reverseComplement ==\n")

rc_seqs <- DNAStringSet(c(
  standard    = "ACGTACGTACGT",
  single_A    = "A",
  single_T    = "T",
  palindrome  = "ATAT",
  palindrome2 = "ACGT",
  all_A       = "AAAAAAAAAA",
  all_N       = "NNNNNNNN",
  iupac_M     = "MACGT",   # M = A or C -> K
  iupac_R     = "RACGT",   # R = A or G -> Y
  iupac_W     = "WACGT",   # W = A or T -> W
  iupac_S     = "SACGT",   # S = C or G -> S
  iupac_Y     = "YACGT",   # Y = C or T -> R
  iupac_K     = "KACGT",   # K = G or T -> M
  iupac_V     = "VACGT",   # V = A,C,G  -> B
  iupac_H     = "HACGT",   # H = A,C,T  -> D
  iupac_D     = "DACGT",   # D = A,G,T  -> H
  iupac_B     = "BACGT",   # B = C,G,T  -> V
  long_iupac  = "MRWSYKVHDBN"
))
rc_result <- as.character(reverseComplement(rc_seqs))
save_golden(as.character(rc_seqs), "s05_rc_seqs")
save_golden(rc_result, "s05_rc_result")

# ============================================================================
# SECTION 6: vcountPattern Edge Cases
# ============================================================================
cat("== Section 6: vcountPattern ==\n")

vcp_seqs <- DNAStringSet(c(
  no_match   = "ACGTACGTACGT",
  all_match  = "NNNNNNNN",
  some_match = "ACNGTNCNGT",
  empty      = "",
  single_N   = "N",
  single_A   = "A"
))
for (pat in c("N", "-", ".", "A")) {
  val <- vcountPattern(pat, vcp_seqs, fixed=TRUE)
  save_golden(as.character(vcp_seqs), "s06_vcp_seqs")
  save_golden(val, paste0("s06_vcp_", pat))
}

# ============================================================================
# SECTION 7: intMatch Edge Cases
# ============================================================================
cat("== Section 7: intMatch ==\n")

# Ordered integer matching edge cases
im_cases <- list(
  basic      = list(x = 1:10, y = c(2L, 5L, 8L)),
  no_overlap = list(x = 1:5,  y = 6:10),
  all_match  = list(x = 1:5,  y = 1:5),
  empty_x    = list(x = integer(0), y = 1:5),
  empty_y    = list(x = 1:5,  y = integer(0)),
  single     = list(x = 5L,   y = 5L),
  large      = list(x = seq(1L, 10000L, 2L), y = seq(1L, 10000L, 3L))
)
for (nm in names(im_cases)) {
  val <- .Call("intMatch", im_cases[[nm]]$x, im_cases[[nm]]$y, PACKAGE="DECIPHER")
  save_golden(val, paste0("s07_im_", nm))
}
save_golden(im_cases, "s07_im_cases")

# ============================================================================
# SECTION 8: LearnTaxa Edge Cases
# ============================================================================
cat("== Section 8: LearnTaxa Edge Cases ==\n")

# 8a. Standard training (balanced tree, same as before)
seqs_ng <- RemoveGaps(seqs_raw)
tax_map <- read.delim("tests/data/test_ref_taxonomy.tsv", header = FALSE,
                       stringsAsFactors = FALSE, col.names = c("accession", "taxonomy"))
seq_names <- sub(" .*", "", names(seqs_ng))
idx <- match(seq_names, tax_map$accession)
keep <- !is.na(idx)
seqs_filt <- seqs_ng[keep]
idx <- idx[keep]
raw_taxonomy <- tax_map$taxonomy[idx]
taxonomy_vec <- paste0("Root; ", gsub(";", "; ", raw_taxonomy))
rank_counts <- sapply(strsplit(taxonomy_vec, "; "), length)
keep_deep <- rank_counts >= 4
seqs_filt <- seqs_filt[keep_deep]
taxonomy_vec <- taxonomy_vec[keep_deep]
seq_widths <- width(seqs_filt)
n_counts <- vcountPattern("N", seqs_filt, fixed = TRUE)
n_fracs <- n_counts / seq_widths
keep_q <- (seq_widths >= 30) & (n_fracs <= 0.3)
seqs_filt <- seqs_filt[keep_q]
taxonomy_vec <- taxonomy_vec[keep_q]
filtered_char <- as.character(seqs_filt)
save_golden(filtered_char, "s08a_filtered_seqs")
save_golden(taxonomy_vec, "s08a_taxonomy_vec")

set.seed(42)
ts_standard <- LearnTaxa(train=seqs_filt, taxonomy=taxonomy_vec, verbose=FALSE)
save_golden(ts_standard, "s08a_training_set")
cat("   8a: standard training K=", ts_standard$K, " nodes=", length(ts_standard$taxonomy), "\n")

# 8b. Asymmetric taxonomy tree
# One branch deep (to species), one shallow (to order only)
set.seed(300)
n_deep <- 30
n_shallow <- 20
deep_seqs <- paste(sample(c("A","C","G","T"), 300*n_deep, replace=TRUE, prob=c(.3,.2,.3,.2)), collapse="")
deep_seqs <- substring(deep_seqs, seq(1, by=300, length.out=n_deep), seq(300, by=300, length.out=n_deep))
shallow_seqs <- paste(sample(c("A","C","G","T"), 300*n_shallow, replace=TRUE, prob=c(.2,.3,.2,.3)), collapse="")
shallow_seqs <- substring(shallow_seqs, seq(1, by=300, length.out=n_shallow), seq(300, by=300, length.out=n_shallow))
asym_seqs <- DNAStringSet(c(deep_seqs, shallow_seqs))
deep_tax <- rep("Root; Eukaryota; Chordata; Mammalia; Carnivora; Canidae; Canis; Canis_lupus;", n_deep)
shallow_tax <- rep("Root; Eukaryota; Arthropoda; Insecta;", n_shallow)
asym_tax <- c(deep_tax, shallow_tax)

set.seed(42)
ts_asym <- LearnTaxa(train=asym_seqs, taxonomy=asym_tax, verbose=FALSE)
save_golden(as.character(asym_seqs), "s08b_asym_seqs")
save_golden(asym_tax, "s08b_asym_tax")
save_golden(ts_asym, "s08b_asym_training_set")
cat("   8b: asymmetric tree K=", ts_asym$K, "\n")

# 8c. Problem groups: nearly identical sequences, different taxa
set.seed(301)
base_seq <- paste(sample(c("A","C","G","T"), 300, replace=TRUE), collapse="")
# Create 20 sequences with only 2% difference each, split into 2 taxa
problem_seqs_list <- character(20)
for (i in 1:20) {
  chars <- strsplit(base_seq, "")[[1]]
  # Mutate 2% of positions
  mut_pos <- sample(300, 6)
  chars[mut_pos] <- sample(c("A","C","G","T"), 6, replace=TRUE)
  problem_seqs_list[i] <- paste(chars, collapse="")
}
problem_seqs <- DNAStringSet(problem_seqs_list)
# First 10 = species A, second 10 = species B
problem_tax <- c(
  rep("Root; Eukaryota; Chordata; Mammalia; Carnivora; Canidae; Canis; Canis_lupus;", 10),
  rep("Root; Eukaryota; Chordata; Mammalia; Carnivora; Canidae; Canis; Canis_familiaris;", 10)
)

set.seed(42)
ts_problem <- LearnTaxa(train=problem_seqs, taxonomy=problem_tax, verbose=FALSE)
save_golden(as.character(problem_seqs), "s08c_problem_seqs")
save_golden(problem_tax, "s08c_problem_tax")
save_golden(ts_problem, "s08c_problem_training_set")
has_problems <- length(ts_problem$problemGroups) > 0
cat("   8c: problem groups - has problems:", has_problems,
    "problem seqs:", nrow(ts_problem$problemSequences), "\n")

# 8d. Singleton groups (1 sequence per taxon)
set.seed(302)
sing_seqs_list <- character(8)
sing_tax <- character(8)
taxa_names <- c("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta")
for (i in 1:8) {
  sing_seqs_list[i] <- paste(sample(c("A","C","G","T"), 300, replace=TRUE,
    prob=c(.25 + (i-4)*0.03, .25 - (i-4)*0.01, .25, .25 + (i-4)*0.01)), collapse="")
  sing_tax[i] <- paste0("Root; Kingdom; Phylum; Class; Order; Family; Genus; ", taxa_names[i], ";")
}
sing_seqs <- DNAStringSet(sing_seqs_list)
set.seed(42)
ts_singletons <- LearnTaxa(train=sing_seqs, taxonomy=sing_tax, verbose=FALSE)
save_golden(as.character(sing_seqs), "s08d_singleton_seqs")
save_golden(sing_tax, "s08d_singleton_tax")
save_golden(ts_singletons, "s08d_singleton_training_set")
cat("   8d: singletons K=", ts_singletons$K, "\n")

# 8e. Explicit K values
set.seed(42)
ts_k5 <- LearnTaxa(train=seqs_filt, taxonomy=taxonomy_vec, K=5, verbose=FALSE)
save_golden(ts_k5, "s08e_training_set_k5")
set.seed(42)
ts_k10 <- LearnTaxa(train=seqs_filt, taxonomy=taxonomy_vec, K=10, verbose=FALSE)
save_golden(ts_k10, "s08e_training_set_k10")
cat("   8e: explicit K=5, K=10\n")

# ============================================================================
# SECTION 9: IdTaxa Classification Edge Cases
# ============================================================================
cat("== Section 9: IdTaxa Edge Cases ==\n")

# Use the standard training set for classification tests
query_raw <- readDNAStringSet("tests/data/test_query.fasta")
query_ng <- RemoveGaps(query_raw)

# 9a. Standard classification (same as before - our baseline)
set.seed(42)
ids_standard <- IdTaxa(test=query_ng, trainingSet=ts_standard,
  type="extended", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(as.character(query_ng), "s09a_query_seqs")
save_golden(ids_standard, "s09a_ids_standard")
cat("   9a: standard classification done\n")

# 9b. Perfect match: query IS a training sequence
perfect_query <- seqs_filt[c(1, 20, 40, 60)]
set.seed(42)
ids_perfect <- IdTaxa(test=perfect_query, trainingSet=ts_standard,
  type="extended", strand="top", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(as.character(perfect_query), "s09b_perfect_query")
save_golden(ids_perfect, "s09b_ids_perfect")
cat("   9b: perfect match done\n")

# 9c. Novel organism: completely random sequence unrelated to training
set.seed(303)
novel_seqs <- DNAStringSet(c(
  random1 = paste(sample(c("A","C","G","T"), 300, replace=TRUE, prob=c(.1,.4,.4,.1)), collapse=""),
  random2 = paste(sample(c("A","C","G","T"), 300, replace=TRUE, prob=c(.4,.1,.1,.4)), collapse=""),
  random3 = paste(sample(c("A","C","G","T"), 200, replace=TRUE), collapse="")
))
set.seed(42)
ids_novel <- IdTaxa(test=novel_seqs, trainingSet=ts_standard,
  type="extended", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(as.character(novel_seqs), "s09c_novel_seqs")
save_golden(ids_novel, "s09c_ids_novel")
cat("   9c: novel organism done (depths: ",
    paste(sapply(ids_novel, function(x) length(x$taxon)), collapse=","), ")\n")

# 9d. Different threshold values
for (thresh in c(0, 30, 50, 60, 80, 95, 100)) {
  set.seed(42)
  ids_th <- IdTaxa(test=query_ng[1:5], trainingSet=ts_standard,
    type="extended", strand="both", threshold=thresh, bootstraps=100,
    minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
  save_golden(ids_th, paste0("s09d_ids_thresh_", thresh))
}
cat("   9d: threshold sweep done\n")

# 9e. Strand variations
set.seed(42)
ids_top <- IdTaxa(test=query_ng[1:5], trainingSet=ts_standard,
  type="extended", strand="top", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(ids_top, "s09e_ids_strand_top")

set.seed(42)
ids_bottom <- IdTaxa(test=query_ng[1:5], trainingSet=ts_standard,
  type="extended", strand="bottom", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(ids_bottom, "s09e_ids_strand_bottom")

set.seed(42)
ids_both <- IdTaxa(test=query_ng[1:5], trainingSet=ts_standard,
  type="extended", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(ids_both, "s09e_ids_strand_both")
cat("   9e: strand variations done\n")

# 9f. Duplicate query sequences (tests de-replication logic)
dup_query <- c(query_ng[1:3], query_ng[1:3], query_ng[2])  # duplicates
names(dup_query) <- paste0("dup_", seq_along(dup_query))
set.seed(42)
ids_dup <- IdTaxa(test=dup_query, trainingSet=ts_standard,
  type="extended", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(as.character(dup_query), "s09f_dup_query")
save_golden(ids_dup, "s09f_ids_dup")
cat("   9f: duplicate queries done\n")

# 9g. Very short query (shorter than K) - should get unclassified
short_query <- DNAStringSet(c(tiny = "ACG"))
set.seed(42)
ids_short <- IdTaxa(test=short_query, trainingSet=ts_standard,
  type="extended", strand="top", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(as.character(short_query), "s09g_short_query")
save_golden(ids_short, "s09g_ids_short")
cat("   9g: short query done (depth: ", length(ids_short[[1]]$taxon), ")\n")

# 9h. Different bootstrap counts
for (boots in c(1L, 10L, 50L, 100L, 200L)) {
  set.seed(42)
  ids_b <- IdTaxa(test=query_ng[1:3], trainingSet=ts_standard,
    type="extended", strand="both", threshold=60, bootstraps=boots,
    minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
  save_golden(ids_b, paste0("s09h_ids_boots_", boots))
}
cat("   9h: bootstrap sweep done\n")

# 9i. Different minDescend values
for (md in c(0.5, 0.7, 0.9, 0.98, 1.0)) {
  set.seed(42)
  ids_md <- IdTaxa(test=query_ng[1:5], trainingSet=ts_standard,
    type="extended", strand="both", threshold=60, bootstraps=100,
    minDescend=md, fullLength=0, processors=1, verbose=FALSE)
  save_golden(ids_md, paste0("s09i_ids_minDescend_", gsub("[.]", "_", as.character(md))))
}
cat("   9i: minDescend sweep done\n")

# 9j. type="collapsed" output format
set.seed(42)
ids_collapsed <- IdTaxa(test=query_ng[1:5], trainingSet=ts_standard,
  type="collapsed", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(ids_collapsed, "s09j_ids_collapsed")
cat("   9j: collapsed output done\n")

# 9k. Classification against problem group training set
set.seed(42)
# Query with one of the problem sequences
problem_query <- problem_seqs[c(1, 11)]  # one from each confusable group
ids_problem <- IdTaxa(test=problem_query, trainingSet=ts_problem,
  type="extended", strand="top", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(as.character(problem_query), "s09k_problem_query")
save_golden(ids_problem, "s09k_ids_problem")
cat("   9k: problem group classification done\n")

# 9l. Classification against singleton training set
set.seed(42)
sing_query <- sing_seqs[c(1, 4, 8)]
ids_sing <- IdTaxa(test=sing_query, trainingSet=ts_singletons,
  type="extended", strand="top", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
save_golden(as.character(sing_query), "s09l_singleton_query")
save_golden(ids_sing, "s09l_ids_singleton")
cat("   9l: singleton classification done\n")

# ============================================================================
# SECTION 10: Full Pipeline Integration
# ============================================================================
cat("== Section 10: Full Pipeline Integration ==\n")

# 10a. End-to-end train + classify + TSV output
set.seed(42)
ts_e2e <- LearnTaxa(train=seqs_filt, taxonomy=taxonomy_vec, verbose=FALSE)
set.seed(42)
ids_e2e <- IdTaxa(test=query_ng, trainingSet=ts_e2e,
  type="extended", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)

# Build the same TSV output as classify_idtaxa.R
results_e2e <- data.frame(
  read_id = character(length(ids_e2e)),
  taxonomic_path = character(length(ids_e2e)),
  confidence = numeric(length(ids_e2e)),
  stringsAsFactors = FALSE
)
for (i in seq_along(ids_e2e)) {
  x <- ids_e2e[[i]]
  results_e2e$read_id[i] <- sub(" .*", "", names(query_ng)[i])
  taxa <- x$taxon
  conf <- x$confidence
  if (length(taxa) > 1) {
    taxa <- taxa[-1]
    conf <- conf[-1]
    classified <- !startsWith(taxa, "unclassified_")
    taxa <- taxa[classified]
    conf <- conf[classified]
  } else {
    taxa <- character(0)
    conf <- numeric(0)
  }
  if (length(taxa) > 0) {
    results_e2e$taxonomic_path[i] <- paste(taxa, collapse = ";")
    results_e2e$confidence[i] <- min(conf)
  } else {
    results_e2e$taxonomic_path[i] <- ""
    results_e2e$confidence[i] <- 0.0
  }
}
save_golden(results_e2e, "s10a_e2e_tsv")
cat("   10a: end-to-end pipeline done\n")

# ============================================================================
# Summary
# ============================================================================
rds_files <- list.files(golden_dir, pattern = "\\.rds$")
cat("\n=== Golden reference generation complete ===\n")
cat("Total golden files:", length(rds_files), "\n")
