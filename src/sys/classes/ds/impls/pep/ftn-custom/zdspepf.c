/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepc/private/dsimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dspepsetcoefficients_    DSPEPSETCOEFFICIENTS
#define dspepgetcoefficients_    DSPEPGETCOEFFICIENTS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dspepsetcoefficients_    dspepsetcoefficients
#define dspepgetcoefficients_    dspepgetcoefficients
#endif

SLEPC_EXTERN void PETSC_STDCALL dspepsetcoefficients_(DS *ds,PetscReal *pbc,PetscErrorCode *ierr)
{
  CHKFORTRANNULLREAL(pbc);
  *ierr = DSPEPSetCoefficients(*ds,pbc);
}

SLEPC_EXTERN void PETSC_STDCALL dspepgetcoefficients_(DS *ds,PetscReal *pbc,PetscErrorCode *ierr)
{
  PetscReal *opbc;
  PetscInt  d;

  CHKFORTRANNULLREAL(pbc);
  *ierr = DSPEPGetCoefficients(*ds,&opbc); if (*ierr) return;
  *ierr = DSPEPGetDegree(*ds,&d); if (*ierr) return;
  *ierr = PetscArraycpy(pbc,opbc,3*(d+1)); if (*ierr) return;
  *ierr = PetscFree(opbc);
}

