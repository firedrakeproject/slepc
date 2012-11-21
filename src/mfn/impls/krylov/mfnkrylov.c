/*                       

   SLEPc matrix function solver: "krylov"

   Method: Krylov

   Algorithm:

       Single-vector Krylov-Schur method to build a Krylov subspace, then
       compute f(B) on the projected matrix B.

   References:

       [1] R. Sidje, "Expokit: a software package for computing matrix
           exponentials", ACM Trans. Math. Softw., 24(1):130–156, 1998.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Anal. Appl., 23(3):601-614, 2001. 

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/mfnimpl.h>                /*I "slepcmfn.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "MFNSetUp_Krylov"
PetscErrorCode MFNSetUp_Krylov(MFN mfn)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!mfn->ncv) mfn->ncv = PetscMin(30,mfn->n);
  if (!mfn->max_it) mfn->max_it = PetscMax(100,2*mfn->n/mfn->ncv);
  ierr = VecDuplicateVecs(mfn->t,mfn->ncv,&mfn->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(mfn,mfn->ncv,mfn->V);CHKERRQ(ierr);
  mfn->allocated_ncv = mfn->ncv;
  ierr = DSAllocate(mfn->ds,mfn->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNSolve_Krylov"
PetscErrorCode MFNSolve_Krylov(MFN mfn,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNReset_Krylov"
PetscErrorCode MFNReset_Krylov(MFN mfn)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (mfn->allocated_ncv > 0) {
    ierr = VecDestroyVecs(mfn->allocated_ncv,&mfn->V);CHKERRQ(ierr);
    mfn->allocated_ncv = 0;
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MFNCreate_Krylov"
PetscErrorCode MFNCreate_Krylov(MFN mfn)
{
  PetscFunctionBegin;
  mfn->ops->solve          = MFNSolve_Krylov;
  mfn->ops->setup          = MFNSetUp_Krylov;
  mfn->ops->reset          = MFNReset_Krylov;
  PetscFunctionReturn(0);
}
EXTERN_C_END

