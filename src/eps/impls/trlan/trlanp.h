/*
   Private data structure used by the TRLAN interface
*/

#if !defined(__TRLANP_H)
#define __TRLANP_H

#include "src/eps/epsimpl.h"

typedef struct {
  int       maxlan;
  int       restart;
  PetscReal *work;
  int       lwork;
} EPS_TRLAN;

/*
   Definition of routines from the TRLAN package
*/

#include "slepcblaslapack.h"

/*
    These are real case. TRLAN currently only has DOUBLE PRECISION version
*/

#define TRLan_     SLEPC_FORTRAN(trlan77,TRLAN77)

EXTERN_C_BEGIN

extern void  TRLan_ (int(*op)(int*,int*,PetscReal*,int*,PetscReal*,int*),
                     int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,
		     int*);

EXTERN_C_END

#endif

