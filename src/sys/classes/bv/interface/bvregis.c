/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/bvimpl.h>          /*I   "slepcbv.h"   I*/

SLEPC_EXTERN PetscErrorCode BVCreate_Vecs(BV);
SLEPC_EXTERN PetscErrorCode BVCreate_Contiguous(BV);
SLEPC_EXTERN PetscErrorCode BVCreate_Svec(BV);
SLEPC_EXTERN PetscErrorCode BVCreate_Mat(BV);
SLEPC_EXTERN PetscErrorCode BVCreate_Tensor(BV);

/*@C
   BVRegisterAll - Registers all of the storage variants in the BV package.

   Not Collective

   Level: advanced

.seealso: BVRegister()
@*/
PetscErrorCode BVRegisterAll(void)
{
  PetscFunctionBegin;
  if (BVRegisterAllCalled) PetscFunctionReturn(0);
  BVRegisterAllCalled = PETSC_TRUE;
  CHKERRQ(BVRegister(BVVECS,BVCreate_Vecs));
  CHKERRQ(BVRegister(BVCONTIGUOUS,BVCreate_Contiguous));
  CHKERRQ(BVRegister(BVSVEC,BVCreate_Svec));
  CHKERRQ(BVRegister(BVMAT,BVCreate_Mat));
  CHKERRQ(BVRegister(BVTENSOR,BVCreate_Tensor));
  PetscFunctionReturn(0);
}
