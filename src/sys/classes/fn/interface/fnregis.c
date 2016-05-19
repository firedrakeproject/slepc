/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
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

#undef __FUNCT__
#define __FUNCT__ "FNRegisterAll"
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

