/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/fnimpl.h>      /*I "slepcfn.h" I*/

PETSC_EXTERN PetscErrorCode FNCreate_Combine(FN);
PETSC_EXTERN PetscErrorCode FNCreate_Rational(FN);
PETSC_EXTERN PetscErrorCode FNCreate_Exp(FN);
PETSC_EXTERN PetscErrorCode FNCreate_Log(FN);
PETSC_EXTERN PetscErrorCode FNCreate_Phi(FN);
PETSC_EXTERN PetscErrorCode FNCreate_Sqrt(FN);
PETSC_EXTERN PetscErrorCode FNCreate_Invsqrt(FN);

/*@C
   FNRegisterAll - Registers all of the math functions in the FN package.

   Not Collective

   Level: advanced
@*/
PetscErrorCode FNRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (FNRegisterAllCalled) PetscFunctionReturn(0);
  FNRegisterAllCalled = PETSC_TRUE;
  ierr = FNRegister(FNCOMBINE,FNCreate_Combine);CHKERRQ(ierr);
  ierr = FNRegister(FNRATIONAL,FNCreate_Rational);CHKERRQ(ierr);
  ierr = FNRegister(FNEXP,FNCreate_Exp);CHKERRQ(ierr);
  ierr = FNRegister(FNLOG,FNCreate_Log);CHKERRQ(ierr);
  ierr = FNRegister(FNPHI,FNCreate_Phi);CHKERRQ(ierr);
  ierr = FNRegister(FNSQRT,FNCreate_Sqrt);CHKERRQ(ierr);
  ierr = FNRegister(FNINVSQRT,FNCreate_Invsqrt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

