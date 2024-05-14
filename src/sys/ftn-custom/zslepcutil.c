/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepcsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define slepcgetversion_              SLEPCGETVERSION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define slepcgetversion_              slepcgetversion
#endif

SLEPC_EXTERN void slepcgetversion_(char *version,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len1)
{
  *ierr = SlepcGetVersion(version,len1);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,version,len1);
}
