/*
   Private data structure used by the PLANSO interface
*/

#if !defined(__PLANSOP_H)
#define __PLANSOP_H

#include "src/eps/epsimpl.h"

typedef struct {
  PetscReal *work;
  int        lwork;
} EPS_PLANSO;

/*
   Definition of routines from the PLANSO package
*/

#if defined(SLEPC_PLANSO_HAVE_UNDERSCORE)
#define SLEPC_PLANSO(lcase,ucase) lcase##_
#elif defined(SLEPC_PLANSO_HAVE_CAPS)
#define SLEPC_PLANSO(lcase,ucase) ucase
#else
#define SLEPC_PLANSO(lcase,ucase) lcase
#endif

/*
    These are real case. PLANSO currently only has DOUBLE PRECISION version
*/

#define PLANdr2_   SLEPC_PLANSO(plandr2,PLANDR2)
#define PLANop_    SLEPC_PLANSO(op,OP)
#define PLANopm_   SLEPC_PLANSO(opm,OPM)

EXTERN_C_BEGIN

EXTERN void  PLANdr2_(int*,int*,int*,int*,PetscReal*,
                      PetscReal*,int*,int*,PetscScalar*,PetscScalar*,PetscReal*,int*,
                      int*,int*,MPI_Comm*);

EXTERN_C_END

#endif

