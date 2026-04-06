/*
 * Registration of .Call entry points for the idtaxa shared library.
 */

#include "idtaxa.h"

static const R_CallMethodDef callMethods[] = {
    {"enumerateSequence", (DL_FUNC) &enumerateSequence, 7},
    {"alphabetSize",      (DL_FUNC) &alphabetSize,      1},
    {"intMatch",          (DL_FUNC) &intMatch,           2},
    {"groupMax",          (DL_FUNC) &groupMax,           3},
    {"detectCores",       (DL_FUNC) &detectCores,        0},
    {"vectorSum",         (DL_FUNC) &vectorSum,          4},
    {"parallelMatch",     (DL_FUNC) &parallelMatch,      8},
    {"removeGaps",        (DL_FUNC) &removeGaps,         4},
    {NULL, NULL, 0}
};

void R_init_idtaxa(DllInfo *info) {
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
