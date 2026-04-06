# Train a taxonomic classifier from DNA sequences.
# Adapted from DECIPHER::LearnTaxa by Erik Wright.
# Modified: uses plain character vectors instead of DNAStringSet.
# DNA-only (no AA/RNA support).

LearnTaxa <- function(train,
	taxonomy,
	rank=NULL,
	K=NULL,
	N=500,
	minFraction=0.01,
	maxFraction=0.06,
	maxIterations=10,
	multiplier=100,
	maxChildren=200,
	alphabet=NULL,
	verbose=TRUE) {

	# error checking
	if (!is.character(train))
		stop("train must be a character vector of DNA sequences.")
	l <- length(train)
	if (l < 2)
		stop("At least two training sequences are required.")
	if (!is.character(taxonomy))
		stop("taxonomy must be a character vector.")
	if (length(taxonomy) != l)
		stop("taxonomy must be the same length as train.")
	if (!is.null(rank)) {
		if (!is.data.frame(rank))
			stop("rank must be a data.frame.")
		ns <- c('Index', 'Name', 'Parent', 'Level', 'Rank')
		w <- which(!(ns %in% names(rank)))
		if (length(w) > 0)
			stop("rank is missing columns named: ",
				paste(ns[w], collapse=", "))
		if (any(duplicated(rank$Index)))
			stop("All values of Index within rank must be unique.")
	}
	if (any(!grepl("Root;", taxonomy, fixed=TRUE)))
		stop("All elements of taxonomy must contain 'Root;'.")

	# DNA-only: fixed alphabet
	alphabet <- NULL
	size <- 4

	if (is.null(K)) {
		if (!is.numeric(N))
			stop("N must be a numeric.")
		if (N <= 1)
			stop("N must be greater than one.")
		quant <- quantile(nchar(train), 0.99, names=FALSE)
		K <- floor(log(N*quant,
			.Call("alphabetSize",
				train)))
		maxK <- 13 # theoretically 15, but may encounter memory errors
		if (K < 1) {
			K <- 1
		} else if (K > maxK) {
			K <- maxK
			warning("K may be smaller than ideal.")
		}
	} else {
		if (!is.numeric(K))
			stop("K must be a numeric.")
		if (length(K) != 1)
			stop("K must be a single numeric.")
		if (K != floor(K))
			stop("K must be a whole number.")
		if (K < 1)
			stop("K must be at least one.")
		maxK <- 15
		if (K > maxK)
			stop(paste("K can be at most ", maxK, ".", sep=""))
	}
	if (!is.numeric(minFraction))
		stop("minFraction must be a numeric.")
	if (minFraction <= 0)
		stop("minFraction must be greater than zero.")
	if (!is.numeric(maxFraction))
		stop("maxFraction must be a numeric.")
	if (maxFraction <= minFraction)
		stop("maxFraction must be greater than minFraction.")
	if (maxFraction >= 1)
		stop("maxFraction must be less than one.")
	if (!is.numeric(maxIterations))
		stop("maxIterations must be a numeric.")
	if (maxIterations < 1)
		stop("maxIterations must be at least one.")
	if (maxIterations != floor(maxIterations))
		stop("maxIterations must be a whole number.")
	if (!is.numeric(multiplier))
		stop("multiplier must be a numeric.")
	if (multiplier <= 0)
		stop("multiplier must be greater than zero.")
	if (!is.numeric(maxChildren))
		stop("maxChildren must be a numeric.")
	if (maxChildren < 1)
		stop("maxChildren must be at least one.")
	if (maxChildren != floor(maxChildren))
		stop("maxChildren must be a whole number.")
	if (!is.logical(verbose))
		stop("verbose must be a logical.")
	a <- vcountPattern("-", train)
	if (any(a > 0))
		stop("Gap characters ('-') are not permitted in train.")
	a <- vcountPattern(".", train)
	if (any(a > 0))
		stop("Unknown characters ('.') are not permitted in train.")

	if (verbose) {
		num <- 0L
		time.1 <- Sys.time()
		pBar <- txtProgressBar(style=ifelse(interactive(), 3, 1))
	}

	# set default values
	nKmers <- size^K
	B <- 100L

	# get the full classification beginning from "Root"
	classes <- gsub(" *; *", ";", taxonomy)
	classes <- sapply(strsplit(classes, "Root;"),
		`[`,
		2L)
	classes <- gsub(" +$", "", classes)
	classes <- paste(classes,
		ifelse(endsWith(classes, ";"),
			"",
			";"),
		sep="")

	# index and sort all unique (non-ambiguous) k-mers
	kmers <- .Call("enumerateSequence",
		train,
		K,
		FALSE, # mask repeats
		FALSE, # mask low complexity regions
		integer(), # mask numerous k-mers
		1L, # left is fast moving side
		1L) # processors
	kmers <- lapply(kmers,
		function(x)
			sort(unique(x + 1L), na.last=NA))
	if (any(lengths(kmers) == 0))
		stop("All training sequences must have at least one k-mer.")

	# create lists of:
	# taxonomy = all distinct taxa ranks
	# taxa = the taxon added in each taxonomy
	# crossIndex = index of taxonomy for each sequence
	# children = index (in taxonomy) of immediate children for each taxa
	# parents = index (in taxonomy) of the parent level
	# sequences = index (in classes/train/kmers) of sequences belonging to each taxa
	u_classes <- unique(classes)
	if (length(u_classes) <= 1)
		stop("More than one taxonomic group is required.")
	taxonomy <- strsplit(u_classes, ";", fixed=TRUE)
	taxonomy <- lapply(taxonomy,
		function(x) {
			sapply(seq_along(x),
				function(n)
					paste(x[seq_len(n)],
						";",
						sep="",
						collapse=""))
		})
	taxonomy <- unique(unlist(taxonomy))
	crossIndex <- match(classes, taxonomy) + 1L # offset by 1 to add "Root"
	taxonomy <- c("Root;", paste("Root;", taxonomy, sep=""))
	s <- strsplit(taxonomy, ";", fixed=TRUE)
	taxa <- unlist(lapply(s,
		function(x) x[length(x)]))
	levels <- lengths(s)

	maxL <- max(levels) - 1L # max level with children
	levs <- vector("list", maxL)
	for (i in seq_len(maxL))
		levs[[i]] <- which(levels == (i + 1L))
	starts <- rep(1L, maxL)
	children <- vector("list", length(taxonomy))
	for (i in seq_along(children)) {
		j <- levels[i]
		if (j <= maxL) {
			while (starts[j] <= length(levs[[j]]) &&
				levs[[j]][starts[j]] <= i)
				starts[j] <- starts[j] + 1L
			if (starts[j] <= length(levs[[j]])) {
				w <- levs[[j]][starts[j]:length(levs[[j]])]
				w <- w[startsWith(taxonomy[w], taxonomy[i])]
			} else {
				w <- integer()
			}
		} else {
			w <- integer()
		}
		children[[i]] <- w
	}

	parents <- numeric(length(taxonomy))
	for (i in seq_along(children))
		parents[children[[i]]] <- i
	# record the rank of each taxon
	if (is.null(rank)) {
		ranks <- NULL
	} else {
		ranks <- character(length(taxa))
		for (i in seq_along(ranks)) {
			w <- which(rank$Name == taxa[i])
			if (length(w) == 0) {
				stop("rank is missing the name: ",
					taxa[i])
			} else if (length(w) > 1) {
				w <- w[which(rank$Level[w] == (levels[i] - 1L))]
				if (length(w) == 0) {
					stop("rank is missing the name: ",
						taxa[i],
						" at level ",
						levels[i] - 1L)
				} else if (length(w) > 1) {
					stop("rank contains a duplicated name: ",
						taxa[i],
						" at level ",
						levels[i] - 1L)
				}
			} else {
				if (rank$Level[w] != (levels[i] - 1L))
					stop("rank matches a name at the wrong level: ",
						taxa[i])
			}
			ranks[i] <- rank$Rank[w]

			# confirm the correct parent
			w <- rank$Parent[w] # Index of Parent
			if (w > 0) { # not Root
				w <- which(rank$Index == w) # Index matching Parent
				if (length(w) == 0)
					stop("All values of Parent in rank must have a corresponding Index.")
				if (rank$Name[w] != taxa[parents[i]])
					stop("rank contains an unexpected Parent for ",
						taxa[i],
						": ",
						rank$Name[w],
						" (expected ",
						taxa[parents[i]],
						")")
			}
		}
	}

	# find the sequences corresponding to each taxonomy
	end_taxonomy <- substring(taxonomy, 6) # without "Root;"
	sequences <- vector("list", length(end_taxonomy))
	nchars <- nchar(end_taxonomy)
	u_nchars <- unique(nchars)
	nchars <- match(nchars, u_nchars)
	nchars <- tapply(seq_along(nchars), nchars, c, simplify=FALSE)
	for (i in seq_along(u_nchars)) {
		m <- match(substring(classes, 0, u_nchars[i]),
			end_taxonomy[nchars[[i]]])
		m <- tapply(seq_along(m), m, c, simplify=FALSE)
		sequences[nchars[[i]]] <- m
	}
	nSeqs <- lengths(sequences)

	# record the most distinctive k-mers at each node in the taxonomic tree
	.createTree <- function(k) {
		l <- length(children[[k]])
		if (l > 0 && l <= maxChildren) {
			# obtain relative k-mer frequency by group
			profile <- vector("list", l)
			descendants <- integer(l)
			for (i in seq_len(l)) {
				results <- .createTree(children[[k]][i])
				profile[[i]] <- results[[1]]
				descendants[i] <- results[[2]]
			}

			profile <- matrix(unlist(profile),
				ncol=l)

			# calculate cross-entropy between parent and children
			q <- colSums(t(profile)*descendants) # weighted k-mer frequencies
			descendants <- sum(descendants) # total descendant groups
			q <- q/descendants # relative k-mer frequency overall
			H <- -profile*log(q) # entropy_i = -p_i*log(q_i)

			# select the most distinguishing k-mers for each group
			o <- apply(H, 2, order, decreasing=TRUE)
			keep <- logical(length(q))
			count <- 0L # number of k-mers
			i <- 1L # k-mer
			j <- 0L # group
			recordKmers <- ceiling(max(colSums(profile > 0)))*0.10
			while (count < recordKmers) {
				j <- j + 1L
				if (j > l) {
					j <- 1L
					i <- i + 1L
				}

				if (!keep[o[i, j]]) {
					count <- count + 1L
					keep[o[i, j]] <- TRUE
				}
			}
			keep <- which(keep)

			decision_kmers[[k]] <<- list(keep, t(profile[keep,]))
			invisible(list(q, descendants))
		} else {
			w <- sequences[[k]]

			profile <- tabulate(unlist(kmers[w]),
				nKmers)
			profile <- profile/sum(profile) # normalize
			invisible(list(profile, 1L))
		}
	}
	decision_kmers <- vector("list", length(taxonomy))
	.createTree(1) # traverse the taxonomic tree

	# learn the appropriate fraction for sampling
	fraction <- rep(maxFraction, length(taxonomy))
	incorrect <- rep(TRUE, l)
	predicted <- character(l)
	# allow each sequence maxIterations*delta control over fraction
	delta <- (maxFraction - minFraction)*multiplier
	for (it in seq_len(maxIterations)) {
		remainingSeqs <- which(incorrect)
		if (length(remainingSeqs) == 0)
			break

		# test whether the sequences can be correctly classified
		for (i in remainingSeqs) {
			if (length(kmers[[i]]) == 0)
				next # no k-mers to test

			k <- 1L
			correct <- TRUE
			repeat {
				subtrees <- children[[k]]
				n <- length(decision_kmers[[k]][[1]])

				if (n == 0) { # no decision k-mers
					break
				} else if (length(subtrees) > 1) {
					# set number of k-mers to choose each bootstrap replicate
					if (is.na(fraction[k])) {
						s <- ceiling(n*minFraction)
					} else {
						s <- ceiling(n*fraction[k])
					}

					# sample the decision k-mers
					sampling <- matrix(sample(n, s*B, replace=TRUE), B, s)

					hits <- matrix(0,
						nrow=length(subtrees),
						ncol=B)
					matches <- .Call("intMatch",
						decision_kmers[[k]][[1]],
						kmers[[i]])
					myweights <- decision_kmers[[k]][[2]]
					for (j in seq_along(subtrees)) {
						hits[j,] <- .Call("vectorSum",
							matches,
							myweights[j,],
							sampling,
							B)
					}

					maxes <- apply(hits, 2, max) # max hits per bootstrap replicate
					hits <- colSums(t(hits) == maxes & maxes > 0)
					w <- which.max(hits)
					if (hits[w] < B*0.8) # require 80% confidence to further descend the taxonomic tree
						break
				} else { # only one child
					w <- 1
				}

				if (!startsWith(classes[i], end_taxonomy[subtrees[w]])) {
					correct <- FALSE
					break
				}

				if (length(children[[subtrees[w]]]) == 0)
					break

				k <- subtrees[w]
			}

			if (correct) { # correct group
				incorrect[i] <- FALSE
				if (verbose) {
					num <- num + 1L
					setTxtProgressBar(pBar, num/l)
				}
			} else { # incorrect group
				predicted[i] <- taxonomy[subtrees[w]]
				if (is.na(fraction[k])) {
					incorrect[i] <- NA
					if (verbose) {
						num <- num + 1L
						setTxtProgressBar(pBar, num/l)
					}
				} else {
					fraction[k] <- fraction[k] - delta/nSeqs[k]
					if (fraction[k] <= minFraction) {
						incorrect[i] <- NA
						fraction[k] <- NA
						if (verbose) {
							num <- num + 1L
							setTxtProgressBar(pBar, num/l)
						}
					}
				}
			}
		}
	}

	# record problematic taxonomic groups and sequences
	w <- which(incorrect | is.na(incorrect))
	if (length(w) > 0) {
		problemSequences <- data.frame(Index=w,
			Expected=paste("Root;", classes[w], sep=""),
			Predicted=predicted[w],
			stringsAsFactors=FALSE)
	} else {
		problemSequences <- data.frame(Index=w,
			Expected=character(),
			Predicted=predicted[w],
			stringsAsFactors=FALSE)
	}

	w <- which(is.na(fraction))
	if (length(w) > 0) {
		problemGroups <- taxonomy[w]
	} else {
		problemGroups <- character()
	}

	# count the number of representatives per class
	classTable <- table(classes)
	weight <- 1/classTable[classes] # weigh each class equally
	# compute Inverse Document Frequency (IDF) weights
	counts <- numeric(nKmers)
	for (i in seq_along(classes)) {
		counts[kmers[[i]]] <- counts[kmers[[i]]] + weight[i]
	}
	counts <- length(classTable)/(1 + counts) # Max/(1 + F)
	counts <- log(counts) # IDF weight is log(Max/(1 + F))

	# prepare the result to return
	result <- list(taxonomy=taxonomy,
		taxa=taxa,
		ranks=ranks,
		levels=levels,
		children=children,
		parents=parents,
		fraction=fraction,
		sequences=sequences,
		kmers=kmers,
		crossIndex=crossIndex,
		K=K,
		IDFweights=counts,
		decisionKmers=decision_kmers,
		problemSequences=problemSequences,
		problemGroups=problemGroups,
		alphabet=alphabet)
	class(result) <- c("Taxa", "Train")

	if (verbose) {
		setTxtProgressBar(pBar, 1)
		close(pBar)
		cat("\n")
		time.2 <- Sys.time()
		print(round(difftime(time.2,
				time.1,
				units='secs'),
			digits=2))
		cat("\n")
	}

	invisible(result)
}
