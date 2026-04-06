# Sequence utility functions replacing Biostrings operations.
# All operate on plain character vectors.

# Count occurrences of a pattern in each element of a character vector.
# Replaces Biostrings::vcountPattern() for the single-char, fixed=TRUE case.
vcountPattern <- function(pattern, subject, fixed = TRUE) {
  nchar(subject) - nchar(gsub(pattern, "", subject, fixed = fixed))
}

# Reverse complement of DNA sequences.
# Replaces Biostrings::reverseComplement().
# Handles IUPAC ambiguity codes.
reverseComplement <- function(x) {
  comp <- chartr("ACGTacgtMRWSYKVHDBNmrwsykvhdbn",
                 "TGCAtgcaKYWSRMBDHVNkywsrmbdhvn", x)
  vapply(comp, function(s) {
    paste(rev(strsplit(s, "")[[1]]), collapse = "")
  }, character(1), USE.NAMES = FALSE)
}
