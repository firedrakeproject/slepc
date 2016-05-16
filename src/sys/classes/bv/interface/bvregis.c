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

#include <slepc/private/bvimpl.h>          /*I   "slepcbv.h"   I*/

PETSC_EXTERN PetscErrorCode BVCreate_Vecs(BV);
PETSC_EXTERN PetscErrorCode BVCreate_Contiguous(BV);
PETSC_EXTERN PetscErrorCode BVCreate_Svec(BV);
PETSC_EXTERN PetscErrorCode BVCreate_Mat(BV);

#undef __FUNCT__
#define __FUNCT__ "BVRegisterAll"
/*@C
   BVRegisterAll - Registers all of the storage variants in the BV package.

   Not Collective

   Level: advanced

.seealso: BVRegister()
@*/
PetscErrorCode BVRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (BVRegisterAllCalled) PetscFunctionReturn(0);
  BVRegisterAllCalled = PETSC_TRUE;
  ierr = BVRegister(BVVECS,BVCreate_Vecs);CHKERRQ(ierr);
  ierr = BVRegister(BVCONTIGUOUS,BVCreate_Contiguous);CHKERRQ(ierr);
  ierr = BVRegister(BVSVEC,BVCreate_Svec);CHKERRQ(ierr);
  ierr = BVRegister(BVMAT,BVCreate_Mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

