#!/usr/bin/env Rscript
# Compare local implementation against golden reference outputs.
# Prerequisites:
#   1. Run tests/generate_golden.R (with Bioconductor) to produce golden/*.rds
#   2. Build: cd src && R CMD SHLIB -o idtaxa.so *.c
#
# Usage: Rscript tests/run_golden.R

cat("=== Running golden tests ===\n\n")

# Load local implementation
dyn.load("src/idtaxa.so")
for (f in list.files("R", pattern = "[.]R$", full.names = TRUE))
  source(f, local = FALSE)

pass_count <- 0L
fail_count <- 0L
section_pass <- 0L
section_fail <- 0L

check <- function(name, condition) {
  if (condition) {
    cat("  PASS:", name, "\n")
    pass_count <<- pass_count + 1L
    section_pass <<- section_pass + 1L
  } else {
    cat("  FAIL:", name, "\n")
    fail_count <<- fail_count + 1L
    section_fail <<- section_fail + 1L
  }
}

load_golden <- function(name) {
  readRDS(file.path("tests/golden", paste0(name, ".rds")))
}

section_start <- function(title) {
  cat("\n", title, "\n", sep="")
  section_pass <<- 0L
  section_fail <<- 0L
}

section_end <- function() {
  cat("  --- ", section_pass, " passed, ", section_fail, " failed ---\n", sep="")
}

# Compare IdTaxa results field by field
compare_ids <- function(our, golden, label) {
  check(paste0(label, ": count"), length(our) == length(golden))
  all_taxa_ok <- TRUE
  max_conf_diff <- 0
  for (i in seq_along(our)) {
    if (!identical(our[[i]]$taxon, golden[[i]]$taxon)) {
      cat("    MISMATCH ", label, " seq ", i, ":\n")
      cat("      Expected:", paste(golden[[i]]$taxon, collapse="; "), "\n")
      cat("      Got:     ", paste(our[[i]]$taxon, collapse="; "), "\n")
      all_taxa_ok <- FALSE
    }
    if (length(our[[i]]$confidence) == length(golden[[i]]$confidence)) {
      d <- max(abs(our[[i]]$confidence - golden[[i]]$confidence))
      if (d > max_conf_diff) max_conf_diff <- d
    }
  }
  check(paste0(label, ": taxa identical"), all_taxa_ok)
  check(paste0(label, ": confidence (max diff: ", format(max_conf_diff, digits=3), ")"),
        max_conf_diff < 0.01)
}

# Compare training sets field by field
compare_ts <- function(our, golden, label) {
  check(paste0(label, ": K"), identical(our$K, golden$K))
  check(paste0(label, ": taxonomy"), identical(our$taxonomy, golden$taxonomy))
  check(paste0(label, ": taxa"), identical(our$taxa, golden$taxa))
  check(paste0(label, ": levels"), identical(our$levels, golden$levels))
  check(paste0(label, ": children"), identical(our$children, golden$children))
  check(paste0(label, ": parents"), identical(our$parents, golden$parents))
  check(paste0(label, ": crossIndex"), identical(our$crossIndex, golden$crossIndex))
  check(paste0(label, ": kmers"), identical(our$kmers, golden$kmers))
  check(paste0(label, ": sequences"), identical(our$sequences, golden$sequences))

  # IDFweights (float)
  if (!is.null(golden$IDFweights) && !is.null(our$IDFweights)) {
    d <- max(abs(our$IDFweights - golden$IDFweights))
    check(paste0(label, ": IDFweights (max diff: ", format(d, digits=3), ")"), d < 1e-10)
  }

  # fraction (float with NAs)
  if (!is.null(golden$fraction) && !is.null(our$fraction)) {
    na_mismatch <- any(xor(is.na(our$fraction), is.na(golden$fraction)))
    check(paste0(label, ": fraction NA pattern"), !na_mismatch)
    if (!na_mismatch) {
      non_na <- !is.na(our$fraction)
      if (any(non_na)) {
        d <- max(abs(our$fraction[non_na] - golden$fraction[non_na]))
        check(paste0(label, ": fraction values (max diff: ", format(d, digits=3), ")"), d < 1e-10)
      }
    }
  }

  # decisionKmers structure
  check(paste0(label, ": decisionKmers length"),
        identical(length(our$decisionKmers), length(golden$decisionKmers)))

  # problemSequences
  check(paste0(label, ": problemSequences"),
        identical(nrow(our$problemSequences), nrow(golden$problemSequences)))
  check(paste0(label, ": problemGroups"),
        identical(our$problemGroups, golden$problemGroups))
}

# ============================================================================
# SECTION 1: FASTA Reading
# ============================================================================
section_start("== Section 1: FASTA Reading ==")

golden_seqs <- load_golden("s01_fasta_seqs")
golden_names <- load_golden("s01_fasta_names")
seqs <- readFasta("tests/data/test_ref.fasta")

check("sequence count", length(seqs) == length(golden_seqs))
check("sequence content", identical(unname(seqs), unname(golden_seqs)))
check("sequence names", identical(names(seqs), golden_names))
section_end()

# ============================================================================
# SECTION 2: K-mer Enumeration Edge Cases
# ============================================================================
section_start("== Section 2: K-mer Enumeration ==")

# 2a. Standard sequences
cat("  2a. Standard sequences (250bp, 1500bp)...\n")
std_seqs <- load_golden("s02a_std_seqs")
golden_kmers <- load_golden("s02a_std_kmers")
our_kmers <- .Call("enumerateSequence", std_seqs, 8L, FALSE, FALSE, integer(), 1L, 1L)
check("2a: standard k-mers identical", identical(our_kmers, golden_kmers))

# 2b. Sequences shorter than K
cat("  2b. Short sequences (0, 1, 3, 7, 8, 9 bp)...\n")
short_seqs <- load_golden("s02b_short_seqs")
golden_short <- load_golden("s02b_short_kmers")
our_short <- .Call("enumerateSequence", short_seqs, 8L, FALSE, FALSE, integer(), 1L, 1L)
check("2b: short k-mers identical", identical(our_short, golden_short))
# Verify specific expectations
check("2b: len=0 yields empty", length(our_short[[1]]) == 0)
check("2b: len=1 yields empty", length(our_short[[2]]) == 0)
check("2b: len=7 yields empty", length(our_short[[4]]) == 0)
check("2b: len=8 yields 1 k-mer", length(our_short[[5]]) == 1)
check("2b: len=9 yields 2 k-mers", length(our_short[[6]]) == 2)

# 2c. Ambiguous bases
cat("  2c. Ambiguous bases (N-heavy)...\n")
ambig_seqs <- load_golden("s02c_ambig_seqs")
golden_ambig <- load_golden("s02c_ambig_kmers")
our_ambig <- .Call("enumerateSequence", ambig_seqs, 8L, FALSE, FALSE, integer(), 1L, 1L)
check("2c: ambiguous k-mers identical", identical(our_ambig, golden_ambig))
# All-N should produce all NAs
check("2c: all-N yields all NA k-mers", all(is.na(our_ambig[[1]])))

# 2d. Homopolymers and repeats
cat("  2d. Homopolymers and repeats...\n")
repeat_seqs <- load_golden("s02d_repeat_seqs")
golden_repeat <- load_golden("s02d_repeat_kmers")
our_repeat <- .Call("enumerateSequence", repeat_seqs, 8L, FALSE, FALSE, integer(), 1L, 1L)
check("2d: repeat k-mers identical", identical(our_repeat, golden_repeat))
# Homopolymer check: poly-A should have all same k-mer value
poly_a_kmers <- our_repeat[[1]]
poly_a_valid <- poly_a_kmers[!is.na(poly_a_kmers)]
check("2d: poly-A k-mers all same value", length(unique(poly_a_valid)) == 1)

# 2e. Different K values
cat("  2e. Different K values...\n")
k_seq <- load_golden("s02e_ktest_seq")
for (k_val in c(1L, 3L, 5L, 8L, 10L, 13L, 15L)) {
  golden_k <- load_golden(paste0("s02e_kmers_K", k_val))
  our_k <- .Call("enumerateSequence", k_seq, k_val, FALSE, FALSE, integer(), 1L, 1L)
  check(paste0("2e: K=", k_val, " k-mers identical"), identical(our_k, golden_k))
}

# 2f. Masking options
cat("  2f. K-mer masking options...\n")
mask_seqs <- load_golden("s02f_mask_seqs")

golden_nomask <- load_golden("s02f_kmers_nomask")
our_nomask <- .Call("enumerateSequence", mask_seqs, 8L, FALSE, FALSE, integer(), 1L, 1L)
check("2f: no masking identical", identical(our_nomask, golden_nomask))

golden_maskrep <- load_golden("s02f_kmers_maskrep")
our_maskrep <- .Call("enumerateSequence", mask_seqs, 8L, TRUE, FALSE, integer(), 1L, 1L)
check("2f: repeat masking identical", identical(our_maskrep, golden_maskrep))

golden_masklcr <- load_golden("s02f_kmers_masklcr")
our_masklcr <- .Call("enumerateSequence", mask_seqs, 8L, FALSE, TRUE, integer(), 1L, 1L)
check("2f: LCR masking identical", identical(our_masklcr, golden_masklcr))

golden_maskboth <- load_golden("s02f_kmers_maskboth")
our_maskboth <- .Call("enumerateSequence", mask_seqs, 8L, TRUE, TRUE, integer(), 1L, 1L)
check("2f: both masking identical", identical(our_maskboth, golden_maskboth))

section_end()

# ============================================================================
# SECTION 3: alphabetSize
# ============================================================================
section_start("== Section 3: alphabetSize ==")

for (nm in c("uniform", "biased_A", "biased_GC", "all_A", "many_seqs")) {
  as_seq <- load_golden(paste0("s03_as_seqs_", nm))
  golden_val <- load_golden(paste0("s03_as_val_", nm))
  our_val <- .Call("alphabetSize", as_seq)
  d <- abs(our_val - golden_val)
  check(paste0("3: ", nm, " (diff: ", format(d, digits=3), ")"), d < 1e-10)
}
section_end()

# ============================================================================
# SECTION 4: RemoveGaps Edge Cases
# ============================================================================
section_start("== Section 4: RemoveGaps ==")

gap_seqs <- load_golden("s04_gap_seqs")
golden_gap <- load_golden("s04_gap_removed")
our_gap <- RemoveGaps(gap_seqs)
check("4: all gap cases identical", identical(unname(our_gap), unname(golden_gap)))

# Verify specific cases
check("4: no-gaps unchanged", our_gap[1] == "ACGTACGTACGTACGT")
check("4: all-dash yields empty", our_gap[2] == "")
check("4: all-dot yields empty", our_gap[3] == "")
check("4: mixed gaps removed", our_gap[4] == "ACGTACGTACGT")
check("4: single base preserved", our_gap[7] == "A")
check("4: gap-base-gap yields base", our_gap[8] == "A")
section_end()

# ============================================================================
# SECTION 5: reverseComplement Edge Cases
# ============================================================================
section_start("== Section 5: reverseComplement ==")

rc_seqs <- load_golden("s05_rc_seqs")
golden_rc <- load_golden("s05_rc_result")
our_rc <- reverseComplement(rc_seqs)
check("5: all RC cases identical", identical(unname(our_rc), unname(golden_rc)))

# Verify specific cases
check("5: standard ACGTACGTACGT -> ACGTACGTACGT", our_rc[1] == golden_rc[1])
check("5: single A -> T", our_rc[2] == "T")
check("5: single T -> A", our_rc[3] == "A")
check("5: all-N -> all-N", nchar(gsub("N", "", our_rc[7])) == 0)
check("5: IUPAC M -> K at end", endsWith(our_rc[8], "K"))
check("5: IUPAC long", our_rc[18] == golden_rc[18])
section_end()

# ============================================================================
# SECTION 6: vcountPattern Edge Cases
# ============================================================================
section_start("== Section 6: vcountPattern ==")

vcp_seqs <- load_golden("s06_vcp_seqs")
for (pat in c("N", "-", ".", "A")) {
  golden_val <- load_golden(paste0("s06_vcp_", pat))
  our_val <- vcountPattern(pat, vcp_seqs, fixed = TRUE)
  check(paste0("6: pattern '", pat, "' identical"), identical(unname(our_val), unname(golden_val)))
}
section_end()

# ============================================================================
# SECTION 7: intMatch Edge Cases
# ============================================================================
section_start("== Section 7: intMatch ==")

im_cases <- load_golden("s07_im_cases")
for (nm in names(im_cases)) {
  golden_val <- load_golden(paste0("s07_im_", nm))
  our_val <- .Call("intMatch", im_cases[[nm]]$x, im_cases[[nm]]$y)
  check(paste0("7: ", nm), identical(our_val, golden_val))
}
section_end()

# ============================================================================
# SECTION 8: LearnTaxa Edge Cases
# ============================================================================
section_start("== Section 8: LearnTaxa ==")

# 8a. Standard training
cat("  8a. Standard balanced training...\n")
filt_seqs <- load_golden("s08a_filtered_seqs")
tax_vec <- load_golden("s08a_taxonomy_vec")
golden_ts <- load_golden("s08a_training_set")
set.seed(42)
our_ts <- LearnTaxa(train=filt_seqs, taxonomy=tax_vec, verbose=FALSE)
compare_ts(our_ts, golden_ts, "8a")

# 8b. Asymmetric tree
cat("  8b. Asymmetric taxonomy tree...\n")
asym_seqs <- load_golden("s08b_asym_seqs")
asym_tax <- load_golden("s08b_asym_tax")
golden_asym <- load_golden("s08b_asym_training_set")
set.seed(42)
our_asym <- LearnTaxa(train=asym_seqs, taxonomy=asym_tax, verbose=FALSE)
compare_ts(our_asym, golden_asym, "8b")

# 8c. Problem groups (nearly identical sequences)
cat("  8c. Problem groups (confusable taxa)...\n")
prob_seqs <- load_golden("s08c_problem_seqs")
prob_tax <- load_golden("s08c_problem_tax")
golden_prob <- load_golden("s08c_problem_training_set")
set.seed(42)
our_prob <- LearnTaxa(train=prob_seqs, taxonomy=prob_tax, verbose=FALSE)
compare_ts(our_prob, golden_prob, "8c")

# 8d. Singletons
cat("  8d. Singleton groups...\n")
sing_seqs <- load_golden("s08d_singleton_seqs")
sing_tax <- load_golden("s08d_singleton_tax")
golden_sing <- load_golden("s08d_singleton_training_set")
set.seed(42)
our_sing <- LearnTaxa(train=sing_seqs, taxonomy=sing_tax, verbose=FALSE)
compare_ts(our_sing, golden_sing, "8d")

# 8e. Explicit K values
cat("  8e. Explicit K values...\n")
golden_k5 <- load_golden("s08e_training_set_k5")
set.seed(42)
our_k5 <- LearnTaxa(train=filt_seqs, taxonomy=tax_vec, K=5, verbose=FALSE)
compare_ts(our_k5, golden_k5, "8e-K5")

golden_k10 <- load_golden("s08e_training_set_k10")
set.seed(42)
our_k10 <- LearnTaxa(train=filt_seqs, taxonomy=tax_vec, K=10, verbose=FALSE)
compare_ts(our_k10, golden_k10, "8e-K10")

section_end()

# ============================================================================
# SECTION 9: IdTaxa Classification Edge Cases
# ============================================================================
section_start("== Section 9: IdTaxa ==")

query_seqs <- load_golden("s09a_query_seqs")

# 9a. Standard classification
cat("  9a. Standard classification...\n")
golden_std <- load_golden("s09a_ids_standard")
set.seed(42)
our_std <- IdTaxa(test=query_seqs, trainingSet=our_ts,
  type="extended", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
compare_ids(our_std, golden_std, "9a")

# 9b. Perfect match
cat("  9b. Perfect match...\n")
perfect_q <- load_golden("s09b_perfect_query")
golden_perf <- load_golden("s09b_ids_perfect")
set.seed(42)
our_perf <- IdTaxa(test=perfect_q, trainingSet=our_ts,
  type="extended", strand="top", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
compare_ids(our_perf, golden_perf, "9b-perfect")
# Perfect matches should classify deeply
for (i in seq_along(our_perf)) {
  depth <- length(our_perf[[i]]$taxon)
  check(paste0("9b: perfect seq ", i, " classified deeply (depth=", depth, ")"), depth >= 4)
}

# 9c. Novel organism
cat("  9c. Novel organism (unrelated to training)...\n")
novel_q <- load_golden("s09c_novel_seqs")
golden_novel <- load_golden("s09c_ids_novel")
set.seed(42)
our_novel <- IdTaxa(test=novel_q, trainingSet=our_ts,
  type="extended", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
compare_ids(our_novel, golden_novel, "9c-novel")

# 9d. Threshold sweep
cat("  9d. Threshold variations...\n")
for (thresh in c(0, 30, 50, 60, 80, 95, 100)) {
  golden_th <- load_golden(paste0("s09d_ids_thresh_", thresh))
  set.seed(42)
  our_th <- IdTaxa(test=query_seqs[1:5], trainingSet=our_ts,
    type="extended", strand="both", threshold=thresh, bootstraps=100,
    minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
  compare_ids(our_th, golden_th, paste0("9d-thresh", thresh))
}

# 9e. Strand variations
cat("  9e. Strand variations...\n")
for (strand_val in c("top", "bottom", "both")) {
  golden_s <- load_golden(paste0("s09e_ids_strand_", strand_val))
  set.seed(42)
  our_s <- IdTaxa(test=query_seqs[1:5], trainingSet=our_ts,
    type="extended", strand=strand_val, threshold=60, bootstraps=100,
    minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
  compare_ids(our_s, golden_s, paste0("9e-", strand_val))
}

# 9f. Duplicate queries
cat("  9f. Duplicate queries...\n")
dup_q <- load_golden("s09f_dup_query")
golden_dup <- load_golden("s09f_ids_dup")
set.seed(42)
our_dup <- IdTaxa(test=dup_q, trainingSet=our_ts,
  type="extended", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
compare_ids(our_dup, golden_dup, "9f-dup")
# Verify duplicates get same result
check("9f: dup[1]==dup[4]", identical(our_dup[[1]]$taxon, our_dup[[4]]$taxon))
check("9f: dup[2]==dup[5]", identical(our_dup[[2]]$taxon, our_dup[[5]]$taxon))
check("9f: dup[2]==dup[7]", identical(our_dup[[2]]$taxon, our_dup[[7]]$taxon))

# 9g. Very short query
cat("  9g. Very short query...\n")
short_q <- load_golden("s09g_short_query")
golden_short_id <- load_golden("s09g_ids_short")
set.seed(42)
our_short_id <- IdTaxa(test=short_q, trainingSet=our_ts,
  type="extended", strand="top", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
compare_ids(our_short_id, golden_short_id, "9g-short")

# 9h. Bootstrap sweep
cat("  9h. Bootstrap variations...\n")
for (boots in c(1L, 10L, 50L, 100L, 200L)) {
  golden_b <- load_golden(paste0("s09h_ids_boots_", boots))
  set.seed(42)
  our_b <- IdTaxa(test=query_seqs[1:3], trainingSet=our_ts,
    type="extended", strand="both", threshold=60, bootstraps=boots,
    minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
  compare_ids(our_b, golden_b, paste0("9h-boots", boots))
}

# 9i. minDescend sweep
cat("  9i. minDescend variations...\n")
for (md in c(0.5, 0.7, 0.9, 0.98, 1.0)) {
  md_str <- gsub("[.]", "_", as.character(md))
  golden_md <- load_golden(paste0("s09i_ids_minDescend_", md_str))
  set.seed(42)
  our_md <- IdTaxa(test=query_seqs[1:5], trainingSet=our_ts,
    type="extended", strand="both", threshold=60, bootstraps=100,
    minDescend=md, fullLength=0, processors=1, verbose=FALSE)
  compare_ids(our_md, golden_md, paste0("9i-md", md))
}

# 9j. Collapsed output format
cat("  9j. Collapsed output format...\n")
golden_coll <- load_golden("s09j_ids_collapsed")
set.seed(42)
our_coll <- IdTaxa(test=query_seqs[1:5], trainingSet=our_ts,
  type="collapsed", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
check("9j: collapsed identical", identical(our_coll, golden_coll))

# 9k. Problem group classification
cat("  9k. Problem group classification...\n")
prob_q <- load_golden("s09k_problem_query")
golden_prob_id <- load_golden("s09k_ids_problem")
set.seed(42)
our_prob_id <- IdTaxa(test=prob_q, trainingSet=our_prob,
  type="extended", strand="top", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
compare_ids(our_prob_id, golden_prob_id, "9k-problem")

# 9l. Singleton classification
cat("  9l. Singleton group classification...\n")
sing_q <- load_golden("s09l_singleton_query")
golden_sing_id <- load_golden("s09l_ids_singleton")
set.seed(42)
our_sing_id <- IdTaxa(test=sing_q, trainingSet=our_sing,
  type="extended", strand="top", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)
compare_ids(our_sing_id, golden_sing_id, "9l-singleton")

section_end()

# ============================================================================
# SECTION 10: Full Pipeline Integration
# ============================================================================
section_start("== Section 10: Full Pipeline Integration ==")

golden_e2e <- load_golden("s10a_e2e_tsv")

set.seed(42)
ts_e2e <- LearnTaxa(train=filt_seqs, taxonomy=tax_vec, verbose=FALSE)
set.seed(42)
ids_e2e <- IdTaxa(test=query_seqs, trainingSet=ts_e2e,
  type="extended", strand="both", threshold=60, bootstraps=100,
  minDescend=0.98, fullLength=0, processors=1, verbose=FALSE)

# Build TSV output same as classify_idtaxa.R
our_e2e <- data.frame(
  read_id = character(length(ids_e2e)),
  taxonomic_path = character(length(ids_e2e)),
  confidence = numeric(length(ids_e2e)),
  stringsAsFactors = FALSE
)
qnames <- names(query_seqs)
if (is.null(qnames)) qnames <- rep("", length(ids_e2e))
for (i in seq_along(ids_e2e)) {
  x <- ids_e2e[[i]]
  our_e2e$read_id[i] <- sub(" .*", "", qnames[i])
  taxa <- x$taxon; conf <- x$confidence
  if (length(taxa) > 1) {
    taxa <- taxa[-1]; conf <- conf[-1]
    classified <- !startsWith(taxa, "unclassified_")
    taxa <- taxa[classified]; conf <- conf[classified]
  } else {
    taxa <- character(0); conf <- numeric(0)
  }
  if (length(taxa) > 0) {
    our_e2e$taxonomic_path[i] <- paste(taxa, collapse = ";")
    our_e2e$confidence[i] <- min(conf)
  } else {
    our_e2e$taxonomic_path[i] <- ""
    our_e2e$confidence[i] <- 0.0
  }
}

check("10a: read_id identical", identical(our_e2e$read_id, golden_e2e$read_id))
check("10a: taxonomic_path identical", identical(our_e2e$taxonomic_path, golden_e2e$taxonomic_path))
conf_diff <- max(abs(our_e2e$confidence - golden_e2e$confidence))
check(paste0("10a: confidence (max diff: ", format(conf_diff, digits=3), ")"), conf_diff < 0.01)

section_end()

# ============================================================================
# Summary
# ============================================================================
cat("\n========================================\n")
cat("TOTAL: ", pass_count, " passed, ", fail_count, " failed\n", sep="")
cat("========================================\n")

if (fail_count > 0) {
  cat("\nGOLDEN TEST FAILED\n")
  quit(status = 1)
} else {
  cat("\nALL GOLDEN TESTS PASSED\n")
}
