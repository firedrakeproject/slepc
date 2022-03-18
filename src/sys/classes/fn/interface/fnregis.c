/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/fnimpl.h>      /*I "slepcfn.h" I*/

SLEPC_EXTERN PetscErrorCode FNCreate_Combine(FN);
SLEPC_EXTERN PetscErrorCode FNCreate_Rational(FN);
SLEPC_EXTERN PetscErrorCode FNCreate_Exp(FN);
SLEPC_EXTERN PetscErrorCode FNCreate_Log(FN);
SLEPC_EXTERN PetscErrorCode FNCreate_Phi(FN);
SLEPC_EXTERN PetscErrorCode FNCreate_Sqrt(FN);
SLEPC_EXTERN PetscErrorCode FNCreate_Invsqrt(FN);

/*@C
   FNRegisterAll - Registers all of the math functions in the FN package.

   Not Collective

   Level: advanced

.seealso: FNRegister()
@*/
PetscErrorCode FNRegisterAll(void)
{
  PetscFunctionBegin;
  if (FNRegisterAllCalled) PetscFunctionReturn(0);
  FNRegisterAllCalled = PETSC_TRUE;
  CHKERRQ(FNRegister(FNCOMBINE,FNCreate_Combine));
  CHKERRQ(FNRegister(FNRATIONAL,FNCreate_Rational));
  CHKERRQ(FNRegister(FNEXP,FNCreate_Exp));
  CHKERRQ(FNRegister(FNLOG,FNCreate_Log));
  CHKERRQ(FNRegister(FNPHI,FNCreate_Phi));
  CHKERRQ(FNRegister(FNSQRT,FNCreate_Sqrt));
  CHKERRQ(FNRegister(FNINVSQRT,FNCreate_Invsqrt));
  PetscFunctionReturn(0);
}
