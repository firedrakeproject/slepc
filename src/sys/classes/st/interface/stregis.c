/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/stimpl.h>          /*I   "slepcst.h"   I*/

PETSC_EXTERN PetscErrorCode STCreate_Shell(ST);
PETSC_EXTERN PetscErrorCode STCreate_Shift(ST);
PETSC_EXTERN PetscErrorCode STCreate_Sinvert(ST);
PETSC_EXTERN PetscErrorCode STCreate_Cayley(ST);
PETSC_EXTERN PetscErrorCode STCreate_Precond(ST);
PETSC_EXTERN PetscErrorCode STCreate_Filter(ST);

/*@C
   STRegisterAll - Registers all of the spectral transformations in the ST package.

   Not Collective

   Level: advanced

.seealso: STRegister()
@*/
PetscErrorCode STRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (STRegisterAllCalled) PetscFunctionReturn(0);
  STRegisterAllCalled = PETSC_TRUE;
  ierr = STRegister(STSHELL,STCreate_Shell);CHKERRQ(ierr);
  ierr = STRegister(STSHIFT,STCreate_Shift);CHKERRQ(ierr);
  ierr = STRegister(STSINVERT,STCreate_Sinvert);CHKERRQ(ierr);
  ierr = STRegister(STCAYLEY,STCreate_Cayley);CHKERRQ(ierr);
  ierr = STRegister(STPRECOND,STCreate_Precond);CHKERRQ(ierr);
  ierr = STRegister(STFILTER,STCreate_Filter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

