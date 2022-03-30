/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/slepcimpl.h>
#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define slepcgetversion_              SLEPCGETVERSION
#define slepcgetversionnumber_        SLEPCGETVERSIONNUMBER
#define slepchasexternalpackage_      SLEPCHASEXTERNALPACKAGE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define slepcgetversion_              slepcgetversion
#define slepcgetversionnumber_        slepcgetversionnumber
#define slepchasexternalpackage_      slepchasexternalpackage
#endif

SLEPC_EXTERN void slepcgetversion_(char *version,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len1)
{
  *ierr = SlepcGetVersion(version,len1);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,version,len1);
}

SLEPC_EXTERN void slepcgetversionnumber_(PetscInt *major,PetscInt *minor,PetscInt *subminor,PetscInt *release,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(major);
  CHKFORTRANNULLINTEGER(minor);
  CHKFORTRANNULLINTEGER(subminor);
  CHKFORTRANNULLINTEGER(release);
  *ierr = SlepcGetVersionNumber(major,minor,subminor,release);
}

SLEPC_EXTERN void slepchasexternalpackage_(char* pkg,PetscBool *has,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t1;

  FIXCHAR(pkg,len,t1);
  *ierr = SlepcHasExternalPackage(t1,has);if (*ierr) return;
  FREECHAR(pkg,t1);
}

