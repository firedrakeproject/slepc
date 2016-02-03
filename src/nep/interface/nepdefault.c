/*
     This file contains some simple default routines for common NEP operations.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/nepimpl.h>     /*I "slepcnep.h" I*/

#undef __FUNCT__
#define __FUNCT__ "NEPSetWorkVecs"
/*@
   NEPSetWorkVecs - Sets a number of work vectors into a NEP object

   Collective on NEP

   Input Parameters:
+  nep - nonlinear eigensolver context
-  nw  - number of work vectors to allocate

   Developers Note:
   This is PETSC_EXTERN because it may be required by user plugin NEP
   implementations.

   Level: developer
@*/
PetscErrorCode NEPSetWorkVecs(NEP nep,PetscInt nw)
{
  PetscErrorCode ierr;
  Vec            t;

  PetscFunctionBegin;
  if (nep->nwork < nw) {
    ierr = VecDestroyVecs(nep->nwork,&nep->work);CHKERRQ(ierr);
    nep->nwork = nw;
    ierr = BVGetColumn(nep->V,0,&t);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(t,nw,&nep->work);CHKERRQ(ierr);
    ierr = BVRestoreColumn(nep->V,0,&t);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(nep,nw,nep->work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetDefaultShift"
/*
  NEPGetDefaultShift - Return the value of sigma to start the nonlinear iteration.
 */
PetscErrorCode NEPGetDefaultShift(NEP nep,PetscScalar *sigma)
{
  PetscFunctionBegin;
  PetscValidPointer(sigma,2);
  switch (nep->which) {
    case NEP_LARGEST_MAGNITUDE:
    case NEP_LARGEST_IMAGINARY:
      *sigma = 1.0;   /* arbitrary value */
      break;
    case NEP_SMALLEST_MAGNITUDE:
    case NEP_SMALLEST_IMAGINARY:
      *sigma = 0.0;
      break;
    case NEP_LARGEST_REAL:
      *sigma = PETSC_MAX_REAL;
      break;
    case NEP_SMALLEST_REAL:
      *sigma = PETSC_MIN_REAL;
      break;
    case NEP_TARGET_MAGNITUDE:
    case NEP_TARGET_REAL:
    case NEP_TARGET_IMAGINARY:
      *sigma = nep->target;
      break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPConvergedEigRelative"
/*
  NEPConvergedEigRelative - Checks convergence relative to the eigenvalue.
*/
PetscErrorCode NEPConvergedEigRelative(NEP nep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscReal w;

  PetscFunctionBegin;
  w = SlepcAbsEigenvalue(eigr,eigi);
  *errest = res/w;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPConvergedAbsolute"
/*
  NEPConvergedAbsolute - Checks convergence absolutely.
*/
PetscErrorCode NEPConvergedAbsolute(NEP nep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPComputeVectors_Schur"
PetscErrorCode NEPComputeVectors_Schur(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       n,i;
  Mat            Z;
  Vec            v;

  PetscFunctionBegin;
  ierr = DSGetDimensions(nep->ds,&n,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DSVectors(nep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetMat(nep->ds,DS_MAT_X,&Z);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(nep->V,0,n);CHKERRQ(ierr);
  ierr = BVMultInPlace(nep->V,Z,0,n);CHKERRQ(ierr);
  ierr = MatDestroy(&Z);CHKERRQ(ierr);

  /* normalization */
  for (i=0;i<n;i++) {
    ierr = BVGetColumn(nep->V,i,&v);CHKERRQ(ierr);
    ierr = VecNormalize(v,NULL);CHKERRQ(ierr);
    ierr = BVRestoreColumn(nep->V,i,&v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

