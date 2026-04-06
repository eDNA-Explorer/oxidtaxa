# Classify sequences using a trained IDTAXA model.
# Adapted from DECIPHER::IdTaxa by Erik Wright.
# Modified: uses plain character vectors instead of DNAStringSet.
# DNA-only (no AA/RNA support).

IdTaxa <- function(test,
	trainingSet,
	type="extended",
	strand="both",
	threshold=60,
	bootstraps=100,
	samples=L^0.47,
	minDescend=0.98,
	fullLength=0,
	processors=1,
	verbose=TRUE) {

	# error checking
	if (!is.character(test))
		stop("test must be a character vector of DNA sequences.")
	# de-replicate sequences
	ns <- names(test)
	d <- !duplicated(test)
	map <- match(test, test[d]) # mapping of unique sequences
	# ensure that the strand vector is consistently mapped
	if (length(strand) == length(test)) {
		# check for different strand with same sequence
		w <- strand[d][map] != strand
		if (any(w)) { # include both strands
			d <- d | w
			map <- match(test, test[d])
		}
		strand <- strand[d]
	} else if (length(strand) > 1) {
		stop("strand must be length 1 or the length of test.")
	}
	test <- test[d]
	l <- length(test)
	if (l < 1)
		stop("At least one test sequence is required.")
	if (!inherits(trainingSet, "Taxa") || !inherits(trainingSet, "Train"))
		stop("trainingSet must be an object of class 'Taxa' (subclass 'Train').")
	TYPES <- c("collapsed", "extended")
	type <- pmatch(type[1], TYPES)
	if (is.na(type))
		stop("Invalid type.")
	if (type == -1)
		stop("Ambiguous type.")
	if (length(strand) != 1 &&
		length(strand) != length(test))
		stop("strand must be length 1 or the length of test.")
	STRANDS <- c("both", "top", "bottom")
	strand <- pmatch(strand, STRANDS)
	if (any(is.na(strand)))
		stop("Invalid strand.")
	if (any(strand == -1))
		stop("Ambiguous strand.")
	if (length(strand) == 1)
		strand <- rep(strand, length(test))
	w <- which(strand == 3)
	if (length(w) > 0) # use opposite orientation
		test[w] <- reverseComplement(test[w])
	if (!is.numeric(threshold))
		stop("threshold must be a numeric.")
	if (threshold < 0 || threshold > 100)
		stop("threshold must be between 0 and 100 (inclusive).")
	if (!is.numeric(minDescend))
		stop("minDescend must be a numeric.")
	if (minDescend < 0.5 || minDescend > 1)
		stop("minDescend must be between 0.5 and 1 (inclusive).")
	sexpr <- substitute(samples)
	if (!is.numeric(sexpr)) {
		if (is.name(sexpr)) { # function name
			sexpr <- call(as.character(sexpr),
				as.name("L")) # pass in 'L'
		} else if (!(is.call(sexpr) && # call
			"L" %in% all.vars(sexpr))) { # containing 'L'
			stop("samples must be a call containing 'L'.")
		}
	}
	if (!is.numeric(fullLength))
		stop("fullLength should be a numeric.")
	if (length(fullLength) == 1L && fullLength == 0) {
		fullLength <- c(0, Inf)
	} else {
		if (any(fullLength < 0))
			stop("fullLength must be greater than zero.")
		if (length(fullLength) == 1L) {
			if (fullLength < 1L) { # upper/lower quantiles
				fullLength <- sort(c(fullLength,
					1 - fullLength))
			} else { # fold-differences
				fullLength <- c(1/fullLength,
					fullLength)
			}
		} else if (length(fullLength) == 2L) {
			if (all.equal(fullLength[1], fullLength[2]))
				stop("fullLength must be 2 different numerics.")
			fullLength <- sort(fullLength)
		} else {
			stop("fullLength must be 1 or 2 numerics.")
		}
	}
	if (!is.logical(verbose))
		stop("verbose must be a logical.")
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
	a <- vcountPattern("-", test)
	if (any(a > 0))
		stop("Gap characters ('-') are not permitted in test.")
	a <- vcountPattern(".", test)
	if (any(a > 0))
		stop("Unknown characters ('.') are not permitted in test.")

	# set default values
	taxonomy <- trainingSet$taxonomy
	taxa <- trainingSet$taxa
	children <- trainingSet$children
	parents <- trainingSet$parents
	fraction <- trainingSet$fraction
	sequences <- trainingSet$sequences
	kmers <- trainingSet$kmers
	crossIndex <- trainingSet$crossIndex
	K <- trainingSet$K
	counts <- trainingSet$IDFweights
	decision_kmers <- trainingSet$decisionKmers
	nKmers <- 4^K

	Ls <- lengths(kmers) # number of k-mers per sequence
	if (all(fullLength < 1)) { # infer length quantiles
		t <- tapply(Ls,
			crossIndex,
			function(x) {
				if (length(x) == 1L) {
					NA_real_
				} else {
					x/mean(x)
				}
			})
		fullLength <- quantile(unlist(t),
			fullLength,
			na.rm=TRUE)
		if (is.na(fullLength[1]))
			stop("fullLength cannot be inferred from trainingSet.")
	}

	if (verbose) {
		time.1 <- Sys.time()
		pBar <- txtProgressBar(style=ifelse(interactive(), 3, 1))
	}

	# index and sort all unique (non-ambiguous) k-mers
	testkmers <- .Call("enumerateSequence",
		test,
		K,
		FALSE, # mask repeats
		FALSE, # mask low complexity regions
		integer(), # mask numerous k-mers
		1L, # left is fast moving side
		processors)

	# determine the total number of k-mers per sequence
	notNAs <- unlist(lapply(testkmers,
		function(x)
			sum(!is.na(x))))
	# calculate the number of k-mers to subsample
	S <- eval(sexpr,
		envir=list(L=notNAs),
		enclos=parent.frame())
	if (!is.numeric(S))
		stop("samples did not evaluate to a numeric.")
	if (any(is.na(S)))
		stop("samples evalated to NA.")
	if (any(S < 0))
		stop("samples evaluated to a negative number of samples.")
	if (length(S) == 1) {
		S <- rep(S, length(notNAs))
	} else if (length(S) != length(notNAs)) {
		stop("samples did not evaluate to a vector of length equal to the number of sequences in test.")
	}
	S <- ceiling(S)

	# require a minimum sample size
	L <- quantile(Ls, 0.9)
	for (minS in seq(2L, 100L, 2L)) {
		# probability of observing half of k-mers by chance
		P <- 1 - pbinom(minS*0.5 - 1,
			minS,
			L/nKmers)
		# probability of occurrence in one or more sequences
		P <- 1 - pbinom(0, # observed once or more
			length(kmers), # number of sequences
			P)
		if (P < 0.01) # less than 1% of bootstrap replicates
			break
	}
	S <- ifelse(S < minS, minS, S)

	testkmers <- lapply(testkmers,
		function(x)
			sort(unique(x + 1L), na.last=NA))

	if (!is.numeric(bootstraps))
		stop("bootstraps must be a numeric.")
	if (bootstraps != floor(bootstraps))
		stop("bootstraps must be a whole number.")
	if (bootstraps < 1)
		stop("bootstraps must be at least one.")
	# set B such that > 99% of k-mers are sampled
	B <- 5*lengths(testkmers)/S # 1 - P(0) = 1 - exp(-5)
	B <- ifelse(B > bootstraps, bootstraps, B)
	B <- as.integer(B)

	boths <- which(strand == 1)
	if (length(boths) > 0) {
		revkmers <- .Call("enumerateSequence",
			reverseComplement(test[boths]),
			K,
			FALSE, # mask repeats
			FALSE, # mask low complexity regions
			integer(), # mask numerous k-mers
			1L, # left is fast moving side
			processors)
		revkmers <- lapply(revkmers,
			function(x)
				sort(unique(x + 1L), na.last=NA))
	}

	if (type == 1L) {
		if (is.null(trainingSet$ranks)) {
			results <- "Root [0%]; unclassified_Root [0%]"
		} else {
			results <- paste("Root [",
				trainingSet$ranks[1],
				", 0%]; unclassified_Root [",
				trainingSet$ranks[2],
				", 0%]",
				sep="")
		}
	} else {
		if (is.null(trainingSet$ranks)) {
				results <- list(list(taxon=c("Root",
						"unclassified_Root"),
					confidence=rep(0, 2)))
		} else {
			results <- list(list(taxon=c("Root",
					"unclassified_Root"),
				confidence=rep(0, 2),
				rank=c(trainingSet$ranks[1],
					trainingSet$ranks[2])))
		}
	}
	results <- rep(results, length(testkmers))
	I <- c(seq_along(testkmers),
		boths)
	O <- c(rep(0L, length(testkmers)),
		seq_along(boths))
	confs <- sims <- numeric(length(testkmers))
	for (i in seq_along(I)) {
		if (O[i]) { # use revkerms
			mykmers <- revkmers[[O[i]]]
		} else { # use testkmers
			mykmers <- testkmers[[I[i]]]
		}

		if (length(mykmers) <= S[I[i]]) # not enough k-mers to test
			next

		# choose which training sequences to use for benchmarking
		k <- 1L
		repeat {
			subtrees <- children[[k]]
			n <- length(decision_kmers[[k]][[1]])

			if (n == 0 || is.na(fraction[k])) { # use every subtree
				w <- seq_along(subtrees)
				break
			} else if (length(subtrees) > 1) {
				# set number of k-mers to choose each bootstrap replicate
				s <- ceiling(n*fraction[k])

				# sample the decision k-mers
				sampling <- matrix(sample(n, s*B[I[i]], replace=TRUE), B[I[i]], s)

				hits <- matrix(0,
					nrow=length(subtrees),
					ncol=B[I[i]])
				matches <- .Call("intMatch",
					decision_kmers[[k]][[1]],
					mykmers)
				myweights <- decision_kmers[[k]][[2]]
				for (j in seq_along(subtrees)) {
					hits[j,] <- .Call("vectorSum",
						matches,
						myweights[j,],
						sampling,
						B[I[i]])
				}

				maxes <- apply(hits, 2, max) # max hits per bootstrap replicate
				hits <- colSums(t(hits) == maxes & maxes > 0)
				w <- which(hits >= minDescend*B[I[i]])
				if (length(w) != 1) { # zero or multiple groups with 100% confidence
					w <- which(hits >= B[I[i]]*0.5) # require 50% confidence to use a subset of groups
					if (length(w) == 0) {
						w <- seq_along(hits) # use all groups
						break
					}

					w <- which(hits > 0) # use any group with hits
					if (length(w) == 0)
						w <- seq_along(hits) # use all groups
					break
				}
			} else { # only one child
				w <- 1
			}

			if (length(children[[subtrees[w]]]) == 0)
				break

			k <- subtrees[w]
		}

		keep <- unlist(sequences[children[[k]][w]])
		if (fullLength[1] > 0 || is.finite(fullLength[2])) {
			keep <- keep[Ls[keep] >= fullLength[1]*length(mykmers) &
				Ls[keep] <= fullLength[2]*length(mykmers)]
			if (length(keep) == 0L) # no sequences
				next
		}

		# determine the number of k-mers per bootstrap replicate
		s <- S[I[i]] # subsample size

		# sample the same query k-mers for all comparisons
		sampling <- matrix(sample(mykmers, s*B[I[i]], replace=TRUE), B[I[i]], s)

		# only match unique k-mers to improve speed
		uSampling <- sort(unique(as.vector(sampling)))
		# record indices of sampled k-mers in hits matrix
		m <- split(seq(0L, length(sampling) - 1L) %% B[I[i]], # row number
			match(sampling, uSampling))

		# record the matches to each group
		hits <- .Call("parallelMatch",
			uSampling,
			kmers,
			keep,
			counts[uSampling], # weights of k-mers
			B[I[i]],
			unlist(m), # positions of k-mers in hits matrix
			c(0L, cumsum(lengths(m))), # range of positions per k-mer
			processors)

		# find the index of the top hit per group
		sumHits <- hits[[2]] # score per sequence
		hits <- hits[[1]] # matrix of hits
		lookup <- crossIndex[keep] # indices in taxonomy
		index <- sort(unique(lookup)) # index of each tested group
		o <- order(lookup)
		topHits <- o[.Call("groupMax",
			sumHits[o],
			lookup[o],
			index)]
		hits <- hits[, topHits, drop=FALSE] # only look at the top sequence per group

		# compute confidence from the number of hits per group
		totHits <- numeric(length(topHits))
		davg <- mean(rowSums(matrix(counts[sampling], B[I[i]], s)))
		maxes <- max.col(hits)
		for (j in seq_len(B[I[i]])) # each bootstrap replicate
			totHits[maxes[j]] <- totHits[maxes[j]] + hits[j, maxes[j]]/davg

		# choose the group with highest confidence
		w <- which(totHits == max(totHits))
		if (length(w) > 1) {
			selected <- sample(w, 1)
		} else {
			selected <- w
		}
		if (O[i]) { # second pass
			if (sum(hits[, selected])/davg <= sims[O[i]]) {
				if (verbose)
					setTxtProgressBar(pBar, i/length(I))
				next # other strand was higher similarity
			}
		} else { # first pass (record result)
			confs[i] <- totHits[selected] # confidence
			sims[i] <- sum(hits[, selected])/davg # weighted k-mer similarity
		}

		# record confidences up the hierarchy
		predicteds <- index[selected]
		p <- parents[predicteds]
		while (p > 0) {
			predicteds <- c(p, predicteds)
			p <- parents[p]
		}
		confidences <- rep(totHits[selected]/B[I[i]]*100, length(predicteds))
		w <- totHits > 0
		w[selected] <- FALSE
		w <- which(w)
		for (j in seq_along(w)) {
			p <- parents[index[w[j]]]
			while (p > 0) {
				m <- match(p, predicteds)
				if (!is.na(m))
					confidences[m] <- confidences[m] + totHits[w[j]]/B[I[i]]*100
				p <- parents[p]
			}
		}

		# record the results for this sequence
		w <- which(confidences >= threshold)
		if (type == 1L) {
			if (length(w) == length(predicteds)) {
				confidences <- formatC(confidences,
					digits=1,
					format="f")
				if (is.null(trainingSet$ranks)) {
					results[I[i]] <- paste(taxa[predicteds],
						paste(" [",
							confidences,
							"%]",
							sep=""),
						sep="",
						collapse="; ")
				} else {
					results[I[i]] <- paste(taxa[predicteds],
						paste(" [",
							trainingSet$ranks[predicteds],
							", ",
							confidences,
							"%]",
							sep=""),
						sep="",
						collapse="; ")
				}
			} else {
				if (length(w) == 0)
					w <- 1 # assign to Root
				confidences <- formatC(c(confidences[w],
						confidences[w[length(w)]]),
					digits=1,
					format="f")
				if (is.null(trainingSet$ranks)) {
					results[I[i]] <- paste(c(taxa[predicteds[w]],
							paste("unclassified",
								taxa[predicteds[w[length(w)]]],
								sep="_")),
						paste(" [",
							confidences,
							"%]",
							sep=""),
						sep="",
						collapse="; ")
				} else {
					results[I[i]] <- paste(c(taxa[predicteds[w]],
							paste("unclassified",
								taxa[predicteds[w[length(w)]]],
								sep="_")),
						paste(" [",
							c(trainingSet$ranks[predicteds[w]],
								trainingSet$ranks[predicteds[w[length(w)] + 1]]),
							", ",
							confidences,
							"%]",
							sep=""),
						sep="",
						collapse="; ")
				}
			}
		} else { # type == 2L
			if (length(w) == length(predicteds)) {
				if (is.null(trainingSet$ranks)) {
					results[[I[i]]] <- list(taxon=taxa[predicteds],
						confidence=confidences)
				} else {
					results[[I[i]]] <- list(taxon=taxa[predicteds],
						confidence=confidences,
						rank=trainingSet$ranks[predicteds])
				}
			} else {
				if (length(w) == 0)
					w <- 1 # assign to Root
				if (is.null(trainingSet$ranks)) {
					results[[I[i]]] <- list(taxon=c(taxa[predicteds[w]],
							paste("unclassified",
								taxa[predicteds[w[length(w)]]],
								sep="_")),
						confidence=c(confidences[w],
							confidences[w[length(w)]]))
				} else {
					results[[I[i]]] <- list(taxon=c(taxa[predicteds[w]],
							paste("unclassified",
								taxa[predicteds[w[length(w)]]],
								sep="_")),
						confidence=c(confidences[w],
							confidences[w[length(w)]]),
						rank=c(trainingSet$ranks[predicteds[w]],
							trainingSet$ranks[predicteds[w[length(w)] + 1]]))
				}
			}
		}

		if (verbose)
			setTxtProgressBar(pBar, i/length(I))
	}
	results <- results[map] # re-replicate
	names(results) <- ns # inherit names from test
	if (type == 2L) # extended
		class(results) <- c("Taxa", "Test")

	if (verbose) {
		close(pBar)
		cat("\n")
		time.2 <- Sys.time()
		print(round(difftime(time.2,
				time.1,
				units='secs'),
			digits=2))
		cat("\n")
	}

	return(results)
}
