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
    These are real case. TRLAN currently only has DOUBLE PRECISION version
*/
#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define TRLan_     trlan77_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define TRLan_     TRLAN77
#else
#define TRLan_     trlan77
#endif

#endif

EXTERN_C_BEGIN

extern void  TRLan_ (int(*op)(int*,int*,PetscReal*,int*,PetscReal*,int*),
                     int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,
		     int*);

EXTERN_C_END

#endif

