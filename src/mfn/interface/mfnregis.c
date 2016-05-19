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

#include <slepc/private/mfnimpl.h>  /*I "slepcmfn.h" I*/

PETSC_EXTERN PetscErrorCode MFNCreate_Krylov(MFN);
PETSC_EXTERN PetscErrorCode MFNCreate_Expokit(MFN);

#undef __FUNCT__
#define __FUNCT__ "MFNRegisterAll"
/*@C
  MFNRegisterAll - Registers all the matrix functions in the MFN package.

  Not Collective

  Level: advanced

.seealso:  MFNRegister()
@*/
PetscErrorCode MFNRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MFNRegisterAllCalled) PetscFunctionReturn(0);
  MFNRegisterAllCalled = PETSC_TRUE;
  ierr = MFNRegister(MFNKRYLOV,MFNCreate_Krylov);CHKERRQ(ierr);
  ierr = MFNRegister(MFNEXPOKIT,MFNCreate_Expokit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

