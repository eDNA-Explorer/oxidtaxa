/****************************************************************************
 *                          Utility Functions                                *
 *                           Author: Erik Wright                            *
 *                    Adapted for plain character vectors                    *
 ****************************************************************************/

#ifdef _OPENMP
#include <omp.h>
#undef match
#endif

#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Utils.h>

#include "idtaxa.h"

// same as x %in% y for ordered integer vectors
SEXP intMatch(SEXP x, SEXP y)
{
	int *v = INTEGER(x);
	int *w = INTEGER(y);
	int i, j;
	int size_x = length(x);
	int size_y = length(y);

	SEXP ans;
	PROTECT(ans = allocVector(LGLSXP, size_x));
	int *rans = INTEGER(ans);

	int s = 0;
	for (i = 0; i < size_x; i++) {
		rans[i] = 0;
		for (j = s; j < size_y; j++) {
			if (v[i] == w[j]) {
				rans[i] = 1;
				break;
			} else if (v[i] < w[j]) {
				break;
			}
		}
		s = j;
	}

	UNPROTECT(1);

	return ans;
}

// index of first maxes in x by z groups in y
SEXP groupMax(SEXP x, SEXP y, SEXP z)
{
	double *v = REAL(x); // values
	int *w = INTEGER(y); // groups
	int *u = INTEGER(z); // unique groups
	int l = length(x); // number of values
	int n = length(z); // number of groups

	SEXP ans;
	PROTECT(ans = allocVector(INTSXP, n));
	int *rans = INTEGER(ans);

	int curr = 0;
	for (int i = 0; i < n; i++) {
		double max = -1e53;
		while (curr < l && w[curr] == u[i]) {
			if (v[curr] > max) {
				rans[i] = curr + 1; // index starting at 1
				max = v[curr];
			}
			curr++;
		}
	}

	UNPROTECT(1);

	return ans;
}

// detect the number of available cores
SEXP detectCores()
{
	SEXP ans;
	PROTECT(ans = allocVector(INTSXP, 1));
	int *rans = INTEGER(ans);

	#ifdef _OPENMP
	rans[0] = omp_get_num_procs();
	#else
	rans[0] = 1;
	#endif

	UNPROTECT(1);

	return ans;
}
