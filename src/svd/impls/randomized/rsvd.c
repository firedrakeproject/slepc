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
  PetscInt       N;

  PetscFunctionBegin;
  PetscCheck(svd->which==SVD_LARGEST,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"This solver supports only largest singular values");
  CHKERRQ(MatGetSize(svd->A,NULL,&N));
  CHKERRQ(SVDSetDimensions_Default(svd));
  PetscCheck(svd->ncv>=svd->nsv,PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The value of ncv must not be smaller than nsv");
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = PetscMax(N/svd->ncv,100);
  svd->leftbasis = PETSC_TRUE;
  svd->mpd = svd->ncv;
  CHKERRQ(SVDAllocateSolution(svd,0));
  CHKERRQ(DSSetType(svd->ds,DSSVD));
  CHKERRQ(DSAllocate(svd->ds,svd->ncv));
  CHKERRQ(SVDSetWorkVecs(svd,1,1));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDRandomizedResidualNorm(SVD svd,PetscInt i,PetscScalar sigma,PetscReal *res)
{
  PetscReal      norm1,norm2;
  Vec            u,v,wu,wv;

  PetscFunctionBegin;
  wu = svd->swapped? svd->workr[0]: svd->workl[0];
  wv = svd->swapped? svd->workl[0]: svd->workr[0];
  if (svd->conv!=SVD_CONV_MAXIT) {
    CHKERRQ(BVGetColumn(svd->V,i,&v));
    CHKERRQ(BVGetColumn(svd->U,i,&u));
    /* norm1 = ||A*v-sigma*u||_2 */
    CHKERRQ(MatMult(svd->A,v,wu));
    CHKERRQ(VecAXPY(wu,-sigma,u));
    CHKERRQ(VecNorm(wu,NORM_2,&norm1));
    /* norm2 = ||A^T*u-sigma*v||_2 */
    CHKERRQ(MatMult(svd->AT,u,wv));
    CHKERRQ(VecAXPY(wv,-sigma,v));
    CHKERRQ(VecNorm(wv,NORM_2,&norm2));
    CHKERRQ(BVRestoreColumn(svd->V,i,&v));
    CHKERRQ(BVRestoreColumn(svd->U,i,&u));
    *res = PetscSqrtReal(norm1*norm1+norm2*norm2);
  } else {
    *res = 1.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_Randomized(SVD svd)
{
  PetscScalar    *w;
  PetscReal      res=1.0;
  PetscInt       i,k=0;
  Mat            A,U,V;

  PetscFunctionBegin;
  /* Form random matrix, G. Complete the initial basis with random vectors */
  CHKERRQ(BVSetActiveColumns(svd->V,svd->nini,svd->ncv));
  CHKERRQ(BVSetRandomNormal(svd->V));
  CHKERRQ(PetscCalloc1(svd->ncv,&w));

  /* Subspace Iteration */
  do {
    svd->its++;
    CHKERRQ(BVSetActiveColumns(svd->V,svd->nconv,svd->ncv));
    CHKERRQ(BVSetActiveColumns(svd->U,svd->nconv,svd->ncv));
    /* Form AG */
    CHKERRQ(BVMatMult(svd->V,svd->A,svd->U));
    /* Orthogonalization Q=qr(AG)*/
    CHKERRQ(BVOrthogonalize(svd->U,NULL));
    /* Form B^*= AQ */
    CHKERRQ(BVMatMult(svd->U,svd->AT,svd->V));

    CHKERRQ(DSSetDimensions(svd->ds,svd->ncv,svd->nconv,svd->ncv));
    CHKERRQ(DSSVDSetDimensions(svd->ds,svd->ncv));
    CHKERRQ(DSGetMat(svd->ds,DS_MAT_A,&A));
    CHKERRQ(MatZeroEntries(A));
    CHKERRQ(BVOrthogonalize(svd->V,A));
    CHKERRQ(DSRestoreMat(svd->ds,DS_MAT_A,&A));
    CHKERRQ(DSSetState(svd->ds,DS_STATE_RAW));
    CHKERRQ(DSSolve(svd->ds,w,NULL));
    CHKERRQ(DSSort(svd->ds,w,NULL,NULL,NULL,NULL));
    CHKERRQ(DSSynchronize(svd->ds,w,NULL));
    CHKERRQ(DSGetMat(svd->ds,DS_MAT_U,&U));
    CHKERRQ(DSGetMat(svd->ds,DS_MAT_V,&V));
    CHKERRQ(BVMultInPlace(svd->U,V,svd->nconv,svd->ncv));
    CHKERRQ(BVMultInPlace(svd->V,U,svd->nconv,svd->ncv));
    CHKERRQ(MatDestroy(&U));
    CHKERRQ(MatDestroy(&V));
    /* Check convergence */
    k = 0;
    for (i=svd->nconv;i<svd->ncv;i++) {
      CHKERRQ(SVDRandomizedResidualNorm(svd,i,w[i],&res));
      svd->sigma[i] = PetscRealPart(w[i]);
      CHKERRQ((*svd->converged)(svd,svd->sigma[i],res,&svd->errest[i],svd->convergedctx));
      if (svd->errest[i] < svd->tol) k++;
      else break;
    }
    if (svd->conv == SVD_CONV_MAXIT && svd->its >= svd->max_it) {
      k = svd->nsv;
      for (i=0;i<svd->ncv;i++) svd->sigma[i] = PetscRealPart(w[i]);
    }
    CHKERRQ((*svd->stopping)(svd,svd->its,svd->max_it,svd->nconv+k,svd->nsv,&svd->reason,svd->stoppingctx));
    svd->nconv += k;
    CHKERRQ(SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,svd->ncv));
  } while (svd->reason == SVD_CONVERGED_ITERATING);
  CHKERRQ(PetscFree(w));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_Randomized(SVD svd)
{
  PetscFunctionBegin;
  svd->ops->setup          = SVDSetUp_Randomized;
  svd->ops->solve          = SVDSolve_Randomized;
  PetscFunctionReturn(0);
}
