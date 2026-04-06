/****************************************************************************
 *                     Remove Gaps from Sequences                           *
 *                           Author: Erik Wright                            *
 *              Rewritten for plain character vector I/O                     *
 ****************************************************************************/

#ifdef _OPENMP
#include <omp.h>
#undef match
#endif

#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Utils.h>

#include "idtaxa.h"

// Remove all gap characters ('-', '.') from each sequence.
// Takes STRSXP, returns STRSXP.
// The 'type' argument is accepted for API compatibility but ignored
// (we always use ASCII gap detection).
SEXP removeGaps(SEXP x, SEXP type, SEXP mask, SEXP nThreads)
{
	int x_length = LENGTH(x);
	int m = asInteger(mask);
	int nthreads = asInteger(nThreads);
	int i, j;

	// Pass 1: count non-gap characters per sequence
	int *widths = (int *) R_alloc(x_length, sizeof(int));

	#ifdef _OPENMP
	#pragma omp parallel for private(i, j) schedule(guided) num_threads(nthreads)
	#endif
	for (i = 0; i < x_length; i++) {
		const char *seq = CHAR(STRING_ELT(x, i));
		int len = LENGTH(STRING_ELT(x, i));
		int w = 0;
		for (j = 0; j < len; j++) {
			if (seq[j] != '-' && seq[j] != '.' && (!m || seq[j] != '+'))
				w++;
		}
		widths[i] = w;
	}

	// Pass 2: build output character vector
	// mkChar/SET_STRING_ELT are not thread-safe, so this must be serial
	SEXP ans;
	PROTECT(ans = allocVector(STRSXP, x_length));

	for (i = 0; i < x_length; i++) {
		const char *seq = CHAR(STRING_ELT(x, i));
		int len = LENGTH(STRING_ELT(x, i));

		if (widths[i] == len) {
			// no gaps - reuse the original CHARSXP
			SET_STRING_ELT(ans, i, STRING_ELT(x, i));
		} else {
			char *buf = R_alloc(widths[i] + 1, 1);
			int p = 0;
			for (j = 0; j < len; j++) {
				if (seq[j] != '-' && seq[j] != '.' && (!m || seq[j] != '+'))
					buf[p++] = seq[j];
			}
			buf[p] = '\0';
			SET_STRING_ELT(ans, i, mkCharLen(buf, widths[i]));
		}
	}

	UNPROTECT(1);

	return ans;
}
