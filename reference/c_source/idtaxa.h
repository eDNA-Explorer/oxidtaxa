#ifndef IDTAXA_H
#define IDTAXA_H

/*
 * Unified header for the self-contained IDTAXA library.
 * Replaces DECIPHER.h + Biostrings_interface.h with a minimal
 * character-vector-backed sequence holder.
 *
 * Adapted from DECIPHER by Erik Wright.
 */

#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Utils.h>
#include <string.h>

/* ---- Sequence holder types -------------------------------------------- */

/* Replaces Biostrings' Chars_holder: a view into one sequence */
typedef struct {
    const char *ptr;
    int length;
} Chars_holder;

/* Replaces Biostrings' XStringSet_holder: wraps an R character vector */
typedef struct {
    SEXP char_vec;  /* STRSXP */
    int length;
} SeqSet_holder;

static inline SeqSet_holder hold_SeqSet(SEXP x) {
    SeqSet_holder h;
    h.char_vec = x;
    h.length = LENGTH(x);
    return h;
}

static inline int get_length_from_SeqSet_holder(const SeqSet_holder *h) {
    return h->length;
}

static inline Chars_holder get_elt_from_SeqSet_holder(const SeqSet_holder *h, int i) {
    Chars_holder ch;
    SEXP s = STRING_ELT(h->char_vec, i);
    ch.ptr = CHAR(s);
    ch.length = LENGTH(s);
    return ch;
}

/* ---- Interrupt check (from DECIPHER.h) -------------------------------- */

static void chkIntFn(void *dummy) {
    R_CheckUserInterrupt();
}

static inline int checkInterrupt() {
    if (R_ToplevelExec(chkIntFn, NULL) == FALSE) {
        return(-1);
    } else {
        return(0);
    }
}

/* ---- Forward declarations --------------------------------------------- */

/* enumerate_sequence.c */
SEXP enumerateSequence(SEXP x, SEXP wordSize, SEXP mask, SEXP maskLCRs,
                       SEXP maskNum, SEXP fastMovingSide, SEXP nThreads);
SEXP alphabetSize(SEXP x);

/* vector_sums.c */
SEXP vectorSum(SEXP x, SEXP y, SEXP z, SEXP b);
SEXP parallelMatch(SEXP x, SEXP y, SEXP indices, SEXP a, SEXP b,
                   SEXP pos, SEXP rng, SEXP nThreads);

/* utils.c */
SEXP intMatch(SEXP x, SEXP y);
SEXP groupMax(SEXP x, SEXP y, SEXP z);
SEXP detectCores(void);

/* remove_gaps.c */
SEXP removeGaps(SEXP x, SEXP type, SEXP mask, SEXP nThreads);

#endif /* IDTAXA_H */
