/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepc/private/pepimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pepstoargetinertias_        PEPSTOARGETINERTIAS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pepstoargetinertias_        pepstoargetinertias
#endif

PETSC_EXTERN void PETSC_STDCALL pepstoargetinertias_(PEP *pep,PetscInt *nshift,PetscReal *shifts,PetscInt *inertias,PetscErrorCode *ierr)
{
  PetscReal *oshifts;
  PetscInt  *oinertias;
  PetscInt  n;

  CHKFORTRANNULLREAL(shifts);
  CHKFORTRANNULLINTEGER(inertias);
  *ierr = PEPSTOARGetInertias(*pep,&n,&oshifts,&oinertias); if (*ierr) return;
  if (shifts) { *ierr = PetscMemcpy(shifts,oshifts,n*sizeof(PetscReal)); if (*ierr) return; }
  if (inertias) { *ierr = PetscMemcpy(inertias,oinertias,n*sizeof(PetscInt)); if (*ierr) return; }
  *nshift = n;
  *ierr = PetscFree(oshifts);
  *ierr = PetscFree(oinertias);
}

