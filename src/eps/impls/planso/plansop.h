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

#include "slepcblaslapack.h"

/*
    These are real case. PLANSO currently only has DOUBLE PRECISION version
*/

#define PLANdr_    SLEPC_FORTRAN(plandr,PLANDR)
#define PLANdr2_   SLEPC_FORTRAN(plandr2,PLANDR2)
#define PLANop_    SLEPC_FORTRAN(op,OP)
#define PLANopm_   SLEPC_FORTRAN(opm,OPM)

EXTERN_C_BEGIN

extern void  PLANdr_ (int*,int*,int*,PetscReal*,PetscReal*,PetscReal*,PetscTruth*,
                      PetscReal*,int*,int*,PetscScalar*,PetscScalar*,PetscReal*,
		      int*,int*,int*,MPI_Comm*);
extern void  PLANdr2_(int*,int*,int*,int*,PetscReal*,
                      PetscReal*,int*,int*,PetscScalar*,PetscScalar*,PetscReal*,int*,
                      int*,int*,MPI_Comm*);

EXTERN_C_END

#endif

