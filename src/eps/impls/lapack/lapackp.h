/*
   Private data structure used by the LAPACK interface.
*/
#if !defined(__LAPACKP_H)
#define __LAPACKP_H

#include "src/eps/epsimpl.h" 

typedef struct {
  Mat     BA,A;
} EPS_LAPACK;

#endif
