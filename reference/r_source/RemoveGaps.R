# Remove gap characters from DNA sequences (character vector).
# Adapted from DECIPHER::RemoveGaps by Erik Wright.
# DNA-only, operates on plain character vectors.

RemoveGaps <- function(myXStringSet,
  removeGaps = "all",
  includeMask = FALSE,
  processors = 1) {

  GAPS <- c("none", "all", "common")
  removeGaps <- pmatch(removeGaps[1], GAPS)
  if (is.na(removeGaps))
    stop("Invalid removeGaps method.")
  if (removeGaps == -1)
    stop("Ambiguous removeGaps method.")
  if (!is.logical(includeMask))
    stop("includeMask must be a logical.")
  if (!is.null(processors) && !is.numeric(processors))
    stop("processors must be a numeric.")
  if (!is.null(processors) && floor(processors) != processors)
    stop("processors must be a whole number.")
  if (!is.null(processors) && processors < 1)
    stop("processors must be at least 1.")
  if (is.null(processors)) {
    processors <- .Call("detectCores")
  } else {
    processors <- as.integer(processors)
  }

  if (removeGaps == 1L) {
    return(myXStringSet)  # "none" - no change
  }

  ns <- names(myXStringSet)

  if (removeGaps == 2L) {
    # "all" - remove all gaps via C
    myXStringSet <- .Call("removeGaps",
      myXStringSet,
      1L,  # type (ignored in our implementation, kept for API compat)
      as.integer(includeMask),
      processors)
  } else if (removeGaps == 3L) {
    # "common" - remove columns that are all gaps (pure R)
    if (length(myXStringSet) == 0) return(myXStringSet)
    # Split all sequences into character matrices
    chars <- strsplit(myXStringSet, "")
    lens <- lengths(chars)
    if (length(unique(lens)) != 1)
      stop("All sequences must be the same length for removeGaps='common'.")
    mat <- do.call(rbind, chars)
    # Find columns where ALL characters are gaps
    all_gap <- apply(mat, 2, function(col) all(col %in% c("-", ".")))
    if (any(all_gap)) {
      mat <- mat[, !all_gap, drop = FALSE]
      myXStringSet <- apply(mat, 1, paste, collapse = "")
    }
  }

  names(myXStringSet) <- ns
  return(myXStringSet)
}
