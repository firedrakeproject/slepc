/*
   Private data structure used by the TRLAN interface

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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

