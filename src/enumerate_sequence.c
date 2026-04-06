/****************************************************************************
 *                       Converts Sequence To Numbers                       *
 *                           Author: Erik Wright                            *
 *                    Adapted for plain character vectors                    *
 ****************************************************************************/

// for OpenMP parallel processing
#ifdef _OPENMP
#include <omp.h>
#undef match
#endif

#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Utils.h>
#include <R_ext/RS.h>
#include <math.h>

#include "idtaxa.h"

/* Convert ASCII DNA base to 0-3 index.
 * Original used Biostrings 2-bit encoding (A=1,C=2,G=4,T=8).
 * This version uses plain ASCII characters. */
static void alphabetFrequency(const Chars_holder *P, int *bits, int position)
{
	const char *p;
	p = (P->ptr + position);

	switch (*p) {
		case 'A': case 'a':
			*(bits) = 0;
			break;
		case 'C': case 'c':
			*(bits) = 1;
			break;
		case 'G': case 'g':
			*(bits) = 2;
			break;
		case 'T': case 't':
			*(bits) = 3;
			break;
		default: // N, ambiguity codes, etc.
			*(bits) = -1;
			break;
	}
}

// changes repeat regions to NAs
static void maskRepeats(int *x, int n, int l1, int l2, int l3, int l4, int l, double prob)
{
	// n: word size
	// l1: min period
	// l2: max period
	// l3: min length of repeat
	// l4: max positions between matches
	// prob: expected probability of match

	int i, p, j, k, c;
	double m, s, t;

	i = 0; // current position
	while (i < l) {
		if (x[i] != NA_INTEGER) {
			for (p = l1; p <= l2 && i + p < l; p++) { // periodicity
				if (x[i] == x[i + p]) { // repeat
					m = 1; // number of matches
					s = m - prob*p; // score = matches above expectation
					j = i + 1; // last match
					k = j; // position
					c = 0; // mismatch count
					while (k < l - p) {
						if (x[k] == x[k + p]) {
							m++; // increment match
							t = m - prob*(k + p - i); // temp score
							if (t <= 0) { // fewer matches than expected
								break;
							} else if (t > s) {
								s = t; // new high score
								j = ++k; // update last position
							}
							c = 0; // reset mismatch count
						} else {
							if (c >= l4) // too many mismatches
								break;
							c++; // increment mismatch count
							k++;
						}
					}

					if (s > 0 && // higher score than expected
						(j - i + n) > p && // continuous repeat
						(j + p - i + n) > l3) {
						// mask all repeat units after the first unit
						for (k = i + p; k <= (j + p - 1); k++)
							x[k] = NA_INTEGER;
						i = k - 1;
						break;
					}
				}
			}
		}
		i++;
	}
}

// changes low complexity regions to NAs
static void maskSimple(int *x, int n, double *E, int l1, int l2, double l3, int l)
{
	// n: word size
	// E: expected counts
	// l1: number of bins
	// l2: window size
	// l3: threshold for statistical significance

	int j, k, sum;
	double freq[l1]; // level frequencies
	for (j = 0; j < l1; j++)
		freq[j] = 0;
	int s = 0; // sum of frequencies
	int c = 0; // count of positions
	int pos[n]; // store previous positions for masking
	int prev[l2]; // previous values before masking
	int curr = 0; // index in prev

	for (j = 0; j < l; j++) {
		if (s == l2) { // remove position outside of window
			sum = prev[curr];
			freq[sum]--;
			s--;
		}
		sum = *(x + j);
		if (sum >= 0 && sum != NA_INTEGER) {
			sum %= l1;
			prev[curr++] = sum;
			if (curr == l2)
				curr = 0;
			freq[sum]++;
			s++;
		} // else continue with previous frequencies
		double score = 0;
		double temp;
		double expected;
		for (k = 0; k < l1; k++) { // Pearson's chi-squared test
			expected = E[k]*s;
			temp = freq[k] - expected;
			temp *= temp;
			score += temp/expected;
		}
		if (score > l3) { // mask
			if (c < n) {
				pos[c] = j - s/2;
				c++;
			} else if (c == n) {
				for (c = 0; c < n; c++)
					*(x + pos[c]) = NA_INTEGER;
				*(x + j - s/2) = NA_INTEGER;
				c++;
			} else { // c > n
				*(x + j - s/2) = NA_INTEGER;
			}
		} else {
			c = 0;
		}
	}
	while (s > 1) {
		curr--;
		if (curr < 0)
			curr = l2 - 1;
		sum = prev[curr];
		freq[sum]--;

		double score = 0;
		double temp;
		double expected;
		for (k = 0; k < l1; k++) { // Pearson's chi-squared test
			expected = E[k]*s;
			temp = freq[k] - expected;
			temp *= temp;
			score += temp/expected;
		}
		if (score > l3) { // mask
			if (c < n) {
				pos[c] = j - s/2;
				c++;
			} else if (c == n) {
				for (c = 0; c < n; c++)
					*(x + pos[c]) = NA_INTEGER;
				*(x + j - s/2) = NA_INTEGER;
				c++;
			} else { // c > n
				*(x + j - s/2) = NA_INTEGER;
			}
		} else {
			c = 0;
		}
		s--;
	}
}

// changes k-mers that are too numerous to NAs
static void maskNumerous(int *x, int n, int tot, int l, int wS)
{
	// n: mask k-mers more numerous
	// tot: number of k-mers possible
	// l: number of k-mers in x
	// wS: word size (minimum mask length)

	int i;
	unsigned int j, k, mod;
	int maxCollisions = 100;
	if (l < tot) {
		mod = (unsigned int)l; // collisions possible
	} else {
		mod = (unsigned int)tot; // collisions impossible
	}
	int *counts = R_Calloc(mod, int); // initialized to zero
	int *keys = R_Calloc(mod, int); // initialized to zero

	// count k-mers (assumes most frequent occur before maxCollisions)
	for (i = 0; i < l; i++) {
		if (x[i] != NA_INTEGER) {
			k = 0;
			do {
				j = (unsigned int)x[i] + k*(k + 1)/2;
				j %= mod;
				if (counts[j] == 0) { // new key
					counts[j] = 1;
					keys[j] = x[i];
					break;
				} else if (x[i] == keys[j]) { // existing key
					counts[j]++;
					break;
				} // else collision
				k++;
			} while (k < maxCollisions);
		}
	}

	// convert k-mers that are too numerous to NA
	int count = 0;
	for (i = 0; i < l; i++) {
		if (x[i] == NA_INTEGER) { // already masked
			count++;
		} else {
			k = 0;
			do {
				j = (unsigned int)x[i] + k*(k + 1)/2;
				j %= mod;
				if (x[i] == keys[j]) { // matching key
					if (counts[j] > n) {
						count++;
						if (count == wS) {
							for (j = 0; j < wS; j++)
								x[i - j] = NA_INTEGER;
						} else if (count > wS) {
							x[i] = NA_INTEGER;
						}
					} else {
						count = 0;
					}
					break;
				}
				k++;
			} while (k < maxCollisions);
		}
	}

	R_Free(counts);
	R_Free(keys);
}

SEXP enumerateSequence(SEXP x, SEXP wordSize, SEXP mask, SEXP maskLCRs, SEXP maskNum, SEXP fastMovingSide, SEXP nThreads)
{
	SeqSet_holder x_set;
	Chars_holder x_i;
	int x_length, i, j, k, wS, maskReps, sum, ambiguous, *rans;
	int fast = asInteger(fastMovingSide);
	int nthreads = asInteger(nThreads);

	// initialize the sequence set
	x_set = hold_SeqSet(x);
	x_length = get_length_from_SeqSet_holder(&x_set);
	wS = asInteger(wordSize); // [1 to 15]
	maskReps = asInteger(mask);
	double prob = pow(0.7, wS); // expected probability of match
	double missed = -48.35429; // log(probability of no matches)
	int maxMismatches = (int)(missed/log(1 - prob)); // interval between matches

	// initialize variables for masking low complexity regions
	double threshold = 12.66667;
	double threshold2 = 38.90749;
	int maskSimp_ = (asInteger(maskLCRs) == 0) ? 0 : 20; // window size
	int maskSimp2 = (asInteger(maskLCRs) == 0) ? 0 : 95; // window size
	double E[4] = {0.25, 0.25, 0.25, 0.25}; // expected frequency

	// initialize variables for masking numerous k-mers
	int l = length(maskNum);
	int *mN;
	int tot; // total number of k-mers
	if (l > 0) {
		mN = INTEGER(maskNum);
		tot = 1;
		for (i = 0; i < wS; i++)
			tot *= 4;
	}

	SEXP ret_list;
	PROTECT(ret_list = allocVector(VECSXP, x_length));

	// fill the position weight vector
	int pwv[wS];
	if (fast) { // left side moves faster
		pwv[0] = 1;
		for (i = 1; i < wS; i++)
			pwv[i] = pwv[i - 1]*4;
	} else { // right side moves faster
		pwv[wS - 1] = 1;
		for (i = wS - 2; i >= 0; i--)
			pwv[i] = pwv[i + 1]*4;
	}

	// build a vector of thread-safe pointers
	int **ptrs = R_Calloc(x_length, int *); // vectors
	for (i = 0; i < x_length; i++) {
		x_i = get_elt_from_SeqSet_holder(&x_set, i);
		SEXP ans;
		if ((x_i.length - wS + 1) < 1) {
			PROTECT(ans = allocVector(INTSXP, 0));
		} else {
			PROTECT(ans = allocVector(INTSXP, x_i.length - wS + 1));
			ptrs[i] = INTEGER(ans);
		}

		SET_VECTOR_ELT(ret_list, i, ans);
		UNPROTECT(1);
	}

	#ifdef _OPENMP
	#pragma omp parallel for private(i,j,k,x_i,rans,sum,ambiguous) num_threads(nthreads)
	#endif
	for (i = 0; i < x_length; i++) {
		x_i = get_elt_from_SeqSet_holder(&x_set, i);
		if ((x_i.length - wS + 1) >= 1) {
			rans = ptrs[i];
			int bases[wS];
			for (j = 0; j < (wS - 1); j++) {
				alphabetFrequency(&x_i, &bases[j], j); // fill initial numbers
			}
			for (j = wS - 1; j < x_i.length; j++) {
				alphabetFrequency(&x_i, &bases[wS - 1], j);
				sum = bases[0]*pwv[0];
				ambiguous = 0;
				if (bases[0] < 0)
					ambiguous = 1;
				for (k = 1; k < wS; k++) {
					sum += bases[k]*pwv[k];
					if (bases[k] < 0)
						ambiguous = 1;
					bases[k - 1] = bases[k]; // shift numbers left
				}
				if (ambiguous) {
					*(rans + j - wS + 1) = NA_INTEGER;
				} else {
					*(rans + j - wS + 1) = sum;
				}
			}

			if (maskReps)
				maskRepeats(rans, wS, 1, 700, 25, maxMismatches, x_i.length - wS + 1, prob);

			if (maskSimp_)
				maskSimple(rans, wS, E, 4, maskSimp_, threshold, x_i.length - wS + 1);
			if (maskSimp2)
				maskSimple(rans, wS, E, 4, maskSimp2, threshold2, x_i.length - wS + 1);

			if (l == x_length)
				maskNumerous(rans, mN[i], tot, x_i.length - wS + 1, wS);
		}
	}

	R_Free(ptrs);

	UNPROTECT(1);

	return ret_list;
}

// returns the size of a balanced alphabet with equivalent entropy
SEXP alphabetSize(SEXP x)
{
	SeqSet_holder x_set;
	Chars_holder x_i;
	int x_length, i, j, letter;
	double sum = 0;

	// initialize the sequence set
	x_set = hold_SeqSet(x);
	x_length = get_length_from_SeqSet_holder(&x_set);

	SEXP ans;
	PROTECT(ans = allocVector(REALSXP, 1));
	double *rans = REAL(ans);
	rans[0] = 0;

	double dist[4] = {0}; // distribution of nucleotides
	for (i = 0; i < x_length; i++) {
		x_i = get_elt_from_SeqSet_holder(&x_set, i);

		for (j = 0; j < x_i.length; j++) {
			alphabetFrequency(&x_i, &letter, j);
			if (letter >= 0)
				dist[letter]++;
		}
	}

	for (i = 0; i < 4; i++)
		sum += dist[i];

	double p; // proportion of each letter
	for (i = 0; i < 4; i++) {
		p = dist[i]/sum;
		if (p > 0)
			rans[0] -= p*log(p); // negative entropy
	}
	rans[0] = exp(rans[0]);

	UNPROTECT(1);

	return ans;
}
