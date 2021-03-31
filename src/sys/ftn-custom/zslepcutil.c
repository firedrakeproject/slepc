/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/slepcimpl.h>
#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define slepcgetversion_              SLEPCGETVERSION
#define slepcgetversionnumber_        SLEPCGETVERSIONNUMBER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define slepcgetversion_              slepcgetversion
#define slepcgetversionnumber_        slepcgetversionnumber
#endif

SLEPC_EXTERN void slepcgetversion_(char *version,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len1)
{
  *ierr = SlepcGetVersion(version,len1);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,version,len1);
}

SLEPC_EXTERN void slepcgetversionnumber_(PetscInt *major,PetscInt *minor,PetscInt *subminor,PetscInt *release,PetscInt *ierr)
{
  CHKFORTRANNULLINTEGER(major);
  CHKFORTRANNULLINTEGER(minor);
  CHKFORTRANNULLINTEGER(subminor);
  CHKFORTRANNULLINTEGER(release);
  *ierr = SlepcGetVersionNumber(major,minor,subminor,release);
}

