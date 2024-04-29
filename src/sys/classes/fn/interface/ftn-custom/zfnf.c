/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepcfn.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fndestroy_                 FNDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fndestroy_                 fndestroy
#endif

SLEPC_EXTERN void fndestroy_(FN *fn,PetscErrorCode *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(fn);
  *ierr = FNDestroy(fn); if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(fn);
}
