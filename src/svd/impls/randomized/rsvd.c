/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
           53(2):217-288, 2011.
*/

#include <slepc/private/svdimpl.h>                /*I "slepcsvd.h" I*/

PetscErrorCode SVDSetUp_Randomized(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  if (svd->which!=SVD_LARGEST) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"This solver supports only largest singular values");
  ierr = MatGetSize(svd->A,NULL,&N);CHKERRQ(ierr);
  ierr = SVDSetDimensions_Default(svd);CHKERRQ(ierr);
  if (svd->ncv<svd->nsv) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The value of ncv must not be smaller than nsv");
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = PetscMax(N/svd->ncv,100);
  svd->leftbasis = PETSC_TRUE;
  svd->mpd = svd->ncv;
  ierr = SVDAllocateSolution(svd,0);CHKERRQ(ierr);
  ierr = DSSetType(svd->ds,DSSVD);CHKERRQ(ierr);
  ierr = DSAllocate(svd->ds,svd->ncv);CHKERRQ(ierr);
  ierr = SVDSetWorkVecs(svd,1,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDRandomizedResidualNorm(SVD svd,PetscInt i,PetscScalar sigma,PetscReal *res)
{
  PetscErrorCode ierr;
  PetscReal      norm1,norm2;
  Vec            u,v,wu,wv;

  PetscFunctionBegin;
  wu = svd->swapped? svd->workr[0]: svd->workl[0];
  wv = svd->swapped? svd->workl[0]: svd->workr[0];
  if (svd->conv!=SVD_CONV_MAXIT) {
    ierr = BVGetColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->U,i,&u);CHKERRQ(ierr);
    /* norm1 = ||A*v-sigma*u||_2 */
    ierr = MatMult(svd->A,v,wu);CHKERRQ(ierr);
    ierr = VecAXPY(wu,-sigma,u);CHKERRQ(ierr);
    ierr = VecNorm(wu,NORM_2,&norm1);CHKERRQ(ierr);
    /* norm2 = ||A^T*u-sigma*v||_2 */
    ierr = MatMult(svd->AT,u,wv);CHKERRQ(ierr);
    ierr = VecAXPY(wv,-sigma,v);CHKERRQ(ierr);
    ierr = VecNorm(wv,NORM_2,&norm2);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,i,&u);CHKERRQ(ierr);
    *res = PetscSqrtReal(norm1*norm1+norm2*norm2);
  } else {
    *res = 1.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_Randomized(SVD svd)
{
  PetscErrorCode ierr;
  PetscScalar    *w;
  PetscReal      res=1.0;
  PetscInt       i,k=0;
  Mat            A,U,V;

  PetscFunctionBegin;
  /* Form random matrix, G. Complete the initial basis with random vectors */
  ierr = BVSetActiveColumns(svd->V,svd->nini,svd->ncv);CHKERRQ(ierr);
  ierr = BVSetRandomNormal(svd->V);CHKERRQ(ierr);
  ierr = PetscCalloc1(svd->ncv,&w);CHKERRQ(ierr);

  /* Subspace Iteration */
  do {
    svd->its++;
    ierr = BVSetActiveColumns(svd->V,svd->nconv,svd->ncv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(svd->U,svd->nconv,svd->ncv);CHKERRQ(ierr);
    /* Form AG */
    ierr = BVMatMult(svd->V,svd->A,svd->U);CHKERRQ(ierr);
    /* Orthogonalization Q=qr(AG)*/
    ierr = BVOrthogonalize(svd->U,NULL);CHKERRQ(ierr);
    /* Form B^*= AQ */
    ierr = BVMatMult(svd->U,svd->AT,svd->V);CHKERRQ(ierr);

    ierr = DSSetDimensions(svd->ds,svd->ncv,svd->nconv,svd->ncv);CHKERRQ(ierr);
    ierr = DSSVDSetDimensions(svd->ds,svd->ncv);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_A,&A);CHKERRQ(ierr);
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    ierr = BVOrthogonalize(svd->V,A);CHKERRQ(ierr);
    ierr = DSRestoreMat(svd->ds,DS_MAT_A,&A);CHKERRQ(ierr);
    ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSSynchronize(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_U,&U);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_V,&V);CHKERRQ(ierr);
    ierr = BVMultInPlace(svd->U,V,svd->nconv,svd->ncv);CHKERRQ(ierr);
    ierr = BVMultInPlace(svd->V,U,svd->nconv,svd->ncv);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);
    ierr = MatDestroy(&V);CHKERRQ(ierr);
    /* Check convergence */
    k = 0;
    for (i=svd->nconv;i<svd->ncv;i++) {
      ierr = SVDRandomizedResidualNorm(svd,i,w[i],&res);CHKERRQ(ierr);
      svd->sigma[i] = PetscRealPart(w[i]);
      ierr = (*svd->converged)(svd,svd->sigma[i],res,&svd->errest[i],svd->convergedctx);CHKERRQ(ierr);
      if (svd->errest[i] < svd->tol) k++;
      else break;
    }
    if (svd->conv == SVD_CONV_MAXIT && svd->its >= svd->max_it) {
      k = svd->nsv;
      for (i=0;i<svd->ncv;i++) svd->sigma[i] = PetscRealPart(w[i]);
    }
    ierr = (*svd->stopping)(svd,svd->its,svd->max_it,svd->nconv+k,svd->nsv,&svd->reason,svd->stoppingctx);CHKERRQ(ierr);
    svd->nconv += k;
    ierr = SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,svd->ncv);CHKERRQ(ierr);
  } while (svd->reason == SVD_CONVERGED_ITERATING);
  ierr = PetscFree(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_Randomized(SVD svd)
{
  PetscFunctionBegin;
  svd->ops->setup          = SVDSetUp_Randomized;
  svd->ops->solve          = SVDSolve_Randomized;
  PetscFunctionReturn(0);
}

