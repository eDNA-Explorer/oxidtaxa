# Pure-R FASTA reader. Returns a named character vector (uppercase).
# Replaces Biostrings::readDNAStringSet().

readFasta <- function(filepath) {
  lines <- readLines(filepath)
  # Remove empty lines at end
  while (length(lines) > 0 && lines[length(lines)] == "") {
    lines <- lines[-length(lines)]
  }
  if (length(lines) == 0) return(character(0))

  header_idx <- which(startsWith(lines, ">"))
  if (length(header_idx) == 0) return(character(0))

  names_vec <- sub("^>\\s*", "", lines[header_idx])
  starts <- header_idx + 1L
  ends <- c(header_idx[-1] - 1L, length(lines))

  seqs <- vapply(seq_along(header_idx), function(i) {
    if (starts[i] > ends[i]) return("")
    paste(lines[starts[i]:ends[i]], collapse = "")
  }, character(1))

  names(seqs) <- names_vec
  toupper(seqs)
}
