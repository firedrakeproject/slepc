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
#include "petsc.h"

/*
   This include file on the Cray T3D/T3E defines the interface between 
  Fortran and C representations of charactor strings.
*/
#if defined(PETSC_USES_CPTOFCD)
#include <fortran.h>
#endif

#if !defined(PETSC_USE_COMPLEX)

/*
    These are real case. PLANSO currently only has DOUBLE PRECISION version
*/
#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define PLANdr_    plandr_
#define PLANdr2_   plandr2_
#define PLANop_    op_
#define PLANopm_   opm_
#define PLANstore_ store_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define PLANdr_    PLANDR
#define PLANdr2_   PLANDR2
#define PLANop_    OP
#define PLANopm_   OPM
#define PLANstore_ STORE
#else
#define PLANdr_    plandr
#define PLANdr2_   plandr2
#define PLANop_    op
#define PLANopm_   opm
#define PLANstore_ store
#endif

#endif

EXTERN_C_BEGIN

extern void  PLANdr_ (int*,int*,int*,PetscReal*,PetscReal*,PetscReal*,PetscTruth*,
                      PetscReal*,int*,int*,PetscScalar*,PetscScalar*,PetscReal*,
		      int*,int*,int*,MPI_Comm*);
extern void  PLANdr2_(int*,int*,int*,int*,PetscReal*,
                      PetscReal*,int*,int*,PetscScalar*,PetscScalar*,PetscReal*,int*,
                      int*,int*,MPI_Comm*);

EXTERN_C_END

#endif

