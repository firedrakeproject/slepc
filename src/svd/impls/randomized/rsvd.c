/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc singular value solver: "randomized"

   Method: RSVD

   Algorithm:

       Randomized singular value decomposition.

   References:

       [1] N. Halko, P.-G. Martinsson, and J. A. Tropp, "Finding
           structure with randomness: Probabilistic algorithms for
           constructing approximate matrix decompositions", SIAM Rev.,
           53(2):217–288, 2011.
*/

#include <slepc/private/svdimpl.h>                /*I "slepcsvd.h" I*/

PetscErrorCode SVDSetUp_Randomized(SVD svd)
{
  PetscErrorCode ierr;
  SVD_RANDOMIZED *ctx = (SVD_RANDOMIZED*)svd->data;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,NULL,&N);CHKERRQ(ierr);
  ierr = SVDSetDimensions_Default(svd);CHKERRQ(ierr);
  if (svd->ncv>svd->nsv+svd->mpd) SETERRQ(PetscObjectComm((PetscObject)svd),1,"The value of ncv must not be larger than nev+mpd");
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = PetscMax(N/svd->ncv,100);
  if (!ctx->rank) ctx->rank = 10;
  svd->leftbasis = PETSC_FALSE;
  ierr = SVDAllocateSolution(svd,1);CHKERRQ(ierr);
  ierr = DSSetType(svd->ds,DSSVD);CHKERRQ(ierr);
  ierr = DSSetCompact(svd->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(svd->ds,svd->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_Randomized(SVD svd)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)svd),1,"Not implemented yet");
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_Randomized(SVD svd)
{
  PetscFunctionBegin;
  svd->ops->setup          = SVDSetUp_Randomized;
  svd->ops->solve          = SVDSolve_Randomized;
  PetscFunctionReturn(0);
}

