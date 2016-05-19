/*
     This file contains some simple default routines for common operations.

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

#include <slepc/private/svdimpl.h>      /*I "slepcsvd.h" I*/

#undef __FUNCT__
#define __FUNCT__ "SVDConvergedRelative"
/*
  SVDConvergedRelative - Checks convergence relative to the eigenvalue.
*/
PetscErrorCode SVDConvergedRelative(SVD svd,PetscReal sigma,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res/sigma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDConvergedAbsolute"
/*
  SVDConvergedAbsolute - Checks convergence absolutely.
*/
PetscErrorCode SVDConvergedAbsolute(SVD svd,PetscReal sigma,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDStoppingBasic"
/*@C
   SVDStoppingBasic - Default routine to determine whether the outer singular value
   solver iteration must be stopped.

   Collective on SVD

   Input Parameters:
+  svd    - singular value solver context obtained from SVDCreate()
.  its    - current number of iterations
.  max_it - maximum number of iterations
.  nconv  - number of currently converged singular triplets
.  nsv    - number of requested singular triplets
-  ctx    - context (not used here)

   Output Parameter:
.  reason - result of the stopping test

   Notes:
   A positive value of reason indicates that the iteration has finished successfully
   (converged), and a negative value indicates an error condition (diverged). If
   the iteration needs to be continued, reason must be set to SVD_CONVERGED_ITERATING
   (zero).

   SVDStoppingBasic() will stop if all requested singular values are converged, or if
   the maximum number of iterations has been reached.

   Use SVDSetStoppingTest() to provide your own test instead of using this one.

   Level: advanced

.seealso: SVDSetStoppingTest(), SVDConvergedReason, SVDGetConvergedReason()
@*/
PetscErrorCode SVDStoppingBasic(SVD svd,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nsv,SVDConvergedReason *reason,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *reason = SVD_CONVERGED_ITERATING;
  if (nconv >= nsv) {
    ierr = PetscInfo2(svd,"Singular value solver finished successfully: %D singular triplets converged at iteration %D\n",nconv,its);CHKERRQ(ierr);
    *reason = SVD_CONVERGED_TOL;
  } else if (its >= max_it) {
    *reason = SVD_DIVERGED_ITS;
    ierr = PetscInfo1(svd,"Singular value solver iteration reached maximum number of iterations (%D)\n",its);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

