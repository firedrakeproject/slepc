/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Simple default routines for common SVD operations
*/

#include <slepc/private/svdimpl.h>      /*I "slepcsvd.h" I*/

/*
  SVDConvergedAbsolute - Checks convergence absolutely.
*/
PetscErrorCode SVDConvergedAbsolute(SVD svd,PetscReal sigma,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res;
  PetscFunctionReturn(0);
}

/*
  SVDConvergedRelative - Checks convergence relative to the singular value.
*/
PetscErrorCode SVDConvergedRelative(SVD svd,PetscReal sigma,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res/sigma;
  PetscFunctionReturn(0);
}

/*
  SVDConvergedNorm - Checks convergence relative to the matrix norms.
*/
PetscErrorCode SVDConvergedNorm(SVD svd,PetscReal sigma,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res/SlepcAbs(svd->nrma,svd->nrmb);
  PetscFunctionReturn(0);
}

/*
  SVDConvergedMaxIt - Always returns Inf to force reaching the maximum number of iterations.
*/
PetscErrorCode SVDConvergedMaxIt(SVD svd,PetscReal sigma,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = PETSC_MAX_REAL;
  PetscFunctionReturn(0);
}

/*@C
   SVDStoppingBasic - Default routine to determine whether the outer singular value
   solver iteration must be stopped.

   Collective on svd

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
    if (svd->conv == SVD_CONV_MAXIT) *reason = SVD_CONVERGED_MAXIT;
    else {
      *reason = SVD_DIVERGED_ITS;
      ierr = PetscInfo1(svd,"Singular value solver iteration reached maximum number of iterations (%D)\n",its);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   SVDSetWorkVecs - Sets a number of work vectors into an SVD object.

   Collective on svd

   Input Parameters:
+  svd    - singular value solver context
.  nleft  - number of work vectors of dimension equal to left singular vector
-  nright - number of work vectors of dimension equal to right singular vector

   Developers Note:
   This is SLEPC_EXTERN because it may be required by user plugin SVD
   implementations.

   Level: developer
@*/
PetscErrorCode SVDSetWorkVecs(SVD svd,PetscInt nleft,PetscInt nright)
{
  PetscErrorCode ierr;
  Vec            t;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveInt(svd,nleft,2);
  PetscValidLogicalCollectiveInt(svd,nright,3);
  if (nleft <= 0) SETERRQ1(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"nleft must be > 0: nleft = %D",nleft);
  if (nright <= 0) SETERRQ1(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"nright must be > 0: nright = %D",nright);
  if (svd->nworkl < nleft) {
    ierr = VecDestroyVecs(svd->nworkl,&svd->workl);CHKERRQ(ierr);
    svd->nworkl = nleft;
    if (svd->isgeneralized) { ierr = SVDCreateLeftTemplate(svd,&t);CHKERRQ(ierr); }
    else { ierr = MatCreateVecsEmpty(svd->OP,NULL,&t);CHKERRQ(ierr); }
    ierr = VecDuplicateVecs(t,nleft,&svd->workl);CHKERRQ(ierr);
    ierr = VecDestroy(&t);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(svd,nleft,svd->workl);CHKERRQ(ierr);
  }
  if (svd->nworkr < nright) {
    ierr = VecDestroyVecs(svd->nworkr,&svd->workr);CHKERRQ(ierr);
    svd->nworkr = nright;
    ierr = MatCreateVecsEmpty(svd->OP,&t,NULL);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(t,nright,&svd->workr);CHKERRQ(ierr);
    ierr = VecDestroy(&t);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(svd,nright,svd->workr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

