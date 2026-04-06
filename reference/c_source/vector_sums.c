/****************************************************************************
 *                        Weighted Vector Summation                         *
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
#include <stdlib.h>

#include "idtaxa.h"

// Summation of x[z]*y[z] for b blocks,
// divided by sum of y[z] in each block
SEXP vectorSum(SEXP x, SEXP y, SEXP z, SEXP b)
{
	int *v = LOGICAL(x); // vector of matches
	double *w = REAL(y); // vector of weights
	int *m = INTEGER(z); // vector of indices
	int size = asInteger(b); // block count
	int l = length(z)/size; // block size

	SEXP ans;
	PROTECT(ans = allocVector(REALSXP, size));
	double *rans = REAL(ans);

	double curWeight, maxWeight;
	int i, j = 0, k, index;
	for (i = 0; i < size; i++) {
		curWeight = 0;
		maxWeight = 0;
		for (k = 0; k < l; k++, j++) {
			index = m[j] - 1;
			maxWeight += w[index];
			if (v[index])
				curWeight += w[index];
		}
		if (maxWeight > 0) {
			rans[i] = curWeight/maxWeight;
		} else {
			rans[i] = 0;
		}
	}

	UNPROTECT(1);

	return ans;
}

// First, temp = x %in% y for ordered integer vectors
// Second, summation of temp[z]*a in b blocks by rows
SEXP parallelMatch(SEXP x, SEXP y, SEXP indices, SEXP a, SEXP b, SEXP pos, SEXP rng, SEXP nThreads)
{
	int *v = INTEGER(x);
	int size_x = length(x);
	double *weights = REAL(a);
	int size = asInteger(b); // block count
	int *u = INTEGER(indices);
	int n = length(indices);
	int *p = INTEGER(pos);
	int *r = INTEGER(rng);
	int nthreads = asInteger(nThreads);
	int i, j, k;

	// build a vector of thread-safe pointers
	int **ptrs = R_Calloc(n, int *); // sequences
	int *size_y = R_Calloc(n, int); // lengths
	for (i = 0; i < n; i++) {
		ptrs[i] = INTEGER(VECTOR_ELT(y, u[i] - 1));
		size_y[i] = length(VECTOR_ELT(y, u[i] - 1));
	}

	SEXP ans;
	PROTECT(ans = allocMatrix(REALSXP, size, n));
	double *rans;
	rans = REAL(ans);
	for (i = 0; i < n*size; i++)
		rans[i] = 0;

	SEXP cSums;
	PROTECT(cSums = allocVector(REALSXP, n));
	double *cS;
	cS = REAL(cSums);
	for (i = 0; i < n; i++)
		cS[i] = 0;

	#ifdef _OPENMP
	#pragma omp parallel num_threads(nthreads)
	{
	#endif
		int *temp = (int *) malloc(size_x*sizeof(int)); // thread-safe on Windows

		#ifdef _OPENMP
		#pragma omp for private(i, j, k) schedule(guided)
		#endif
		for (k = 0; k < n; k++) {
			int *w = ptrs[k];

			// temp = x %in% y
			int c = 0; // count of k-mer matches
			j = 0;
			for (i = 0; i < size_x; i++) {
				for (; j < size_y[k]; j++) {
					if (v[i] <= w[j]) {
						if (v[i] == w[j])
							temp[c++] = i;
						break;
					}
				}
			}

			// insert k-mer weights into hits matrix
			for (i = 0; i < c; i++) {
				for (j = r[temp[i]]; j < r[temp[i] + 1]; j++)
					rans[k*size + p[j]] += weights[temp[i]];
			}

			// sum temp[m]*weights by rows
			for (i = 0; i < size; i++)
				cS[k] += rans[k*size + i];
		}

		free(temp);
	#ifdef _OPENMP
	}
	#endif

	R_Free(ptrs);
	R_Free(size_y);

	SEXP ret_list;
	PROTECT(ret_list = allocVector(VECSXP, 2));
	SET_VECTOR_ELT(ret_list, 0, ans);
	SET_VECTOR_ELT(ret_list, 1, cSums);

	UNPROTECT(3);

	return ret_list;
}
