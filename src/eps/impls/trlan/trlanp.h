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
   These are real case. TRLAN currently only has DOUBLE PRECISION version
*/

#if defined(SLEPC_TRLAN_HAVE_UNDERSCORE)
#define TRLan_ trlan77_
#elif defined(SLEPC_TRLAN_HAVE_CAPS)
#define TRLan_ TRLAN77
#else
#define TRLan_ trlan77
#endif

EXTERN_C_BEGIN

extern void  TRLan_ (int(*op)(int*,int*,PetscReal*,int*,PetscReal*,int*),
                     int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,
		     int*);

EXTERN_C_END

#endif

