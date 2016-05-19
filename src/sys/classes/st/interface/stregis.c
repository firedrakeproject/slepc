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

#include <slepc/private/stimpl.h>          /*I   "slepcst.h"   I*/

PETSC_EXTERN PetscErrorCode STCreate_Shell(ST);
PETSC_EXTERN PetscErrorCode STCreate_Shift(ST);
PETSC_EXTERN PetscErrorCode STCreate_Sinvert(ST);
PETSC_EXTERN PetscErrorCode STCreate_Cayley(ST);
PETSC_EXTERN PetscErrorCode STCreate_Precond(ST);

#undef __FUNCT__
#define __FUNCT__ "STRegisterAll"
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
  PetscFunctionReturn(0);
}

