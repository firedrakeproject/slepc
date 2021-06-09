/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SVD routines related to the solution process
*/

#include <slepc/private/svdimpl.h>   /*I "slepcsvd.h" I*/

/*
  SVDComputeVectors_Left - Compute left singular vectors as U=A*V.
  Only done if the leftbasis flag is false. Assumes V is available.
 */
PetscErrorCode SVDComputeVectors_Left(SVD svd)
{
  PetscErrorCode ierr;
  Vec            tl;
  PetscInt       oldsize;

  PetscFunctionBegin;
  if (!svd->leftbasis) {
    /* generate left singular vectors on U */
    if (!svd->U) { ierr = SVDGetBV(svd,NULL,&svd->U);CHKERRQ(ierr); }
    ierr = BVGetSizes(svd->U,NULL,NULL,&oldsize);CHKERRQ(ierr);
    if (!oldsize) {
      if (!((PetscObject)(svd->U))->type_name) {
        ierr = BVSetType(svd->U,BVSVEC);CHKERRQ(ierr);
      }
      ierr = MatCreateVecsEmpty(svd->A,NULL,&tl);CHKERRQ(ierr);
      ierr = BVSetSizesFromVec(svd->U,tl,svd->ncv);CHKERRQ(ierr);
      ierr = VecDestroy(&tl);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(svd->V,0,svd->nconv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(svd->U,0,svd->nconv);CHKERRQ(ierr);
    ierr = BVMatMult(svd->V,svd->A,svd->U);CHKERRQ(ierr);
    ierr = BVOrthogonalize(svd->U,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDComputeVectors(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SVDCheckSolved(svd,1);
  if (svd->state==SVD_STATE_SOLVED && svd->ops->computevectors) {
    ierr = (*svd->ops->computevectors)(svd);CHKERRQ(ierr);
  }
  svd->state = SVD_STATE_VECTORS;
  PetscFunctionReturn(0);
}

/*@
   SVDSolve - Solves the singular value problem.

   Collective on svd

   Input Parameter:
.  svd - singular value solver context obtained from SVDCreate()

   Options Database Keys:
+  -svd_view - print information about the solver used
.  -svd_view_mat0 binary - save the first matrix (A) to the default binary viewer
.  -svd_view_mat1 binary - save the second matrix (B) to the default binary viewer
.  -svd_view_vectors binary - save the computed singular vectors to the default binary viewer
.  -svd_view_values - print computed singular values
.  -svd_converged_reason - print reason for convergence, and number of iterations
.  -svd_error_absolute - print absolute errors of each singular triplet
-  -svd_error_relative - print relative errors of each singular triplet

   Level: beginner

.seealso: SVDCreate(), SVDSetUp(), SVDDestroy()
@*/
PetscErrorCode SVDSolve(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       i,*workperm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (svd->state>=SVD_STATE_SOLVED) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(SVD_Solve,svd,0,0,0);CHKERRQ(ierr);

  /* call setup */
  ierr = SVDSetUp(svd);CHKERRQ(ierr);
  svd->its = 0;
  svd->nconv = 0;
  for (i=0;i<svd->ncv;i++) {
    svd->sigma[i]  = 0.0;
    svd->errest[i] = 0.0;
    svd->perm[i]   = i;
  }
  ierr = SVDViewFromOptions(svd,NULL,"-svd_view_pre");CHKERRQ(ierr);

  switch (svd->problem_type) {
    case SVD_STANDARD:
      ierr = (*svd->ops->solve)(svd);CHKERRQ(ierr);
      break;
    case SVD_GENERALIZED:
      ierr = (*svd->ops->solveg)(svd);CHKERRQ(ierr);
      break;
  }
  svd->state = SVD_STATE_SOLVED;

  /* sort singular triplets */
  if (svd->which == SVD_SMALLEST) {
    ierr = PetscSortRealWithPermutation(svd->nconv,svd->sigma,svd->perm);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc1(svd->nconv,&workperm);CHKERRQ(ierr);
    for (i=0;i<svd->nconv;i++) workperm[i] = i;
    ierr = PetscSortRealWithPermutation(svd->nconv,svd->sigma,workperm);CHKERRQ(ierr);
    for (i=0;i<svd->nconv;i++) svd->perm[i] = workperm[svd->nconv-i-1];
    ierr = PetscFree(workperm);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(SVD_Solve,svd,0,0,0);CHKERRQ(ierr);

  /* various viewers */
  ierr = SVDViewFromOptions(svd,NULL,"-svd_view");CHKERRQ(ierr);
  ierr = SVDConvergedReasonViewFromOptions(svd);CHKERRQ(ierr);
  ierr = SVDErrorViewFromOptions(svd);CHKERRQ(ierr);
  ierr = SVDValuesViewFromOptions(svd);CHKERRQ(ierr);
  ierr = SVDVectorsViewFromOptions(svd);CHKERRQ(ierr);
  ierr = MatViewFromOptions(svd->OP,(PetscObject)svd,"-svd_view_mat0");CHKERRQ(ierr);
  if (svd->isgeneralized) {
    ierr = MatViewFromOptions(svd->OPb,(PetscObject)svd,"-svd_view_mat1");CHKERRQ(ierr);
  }

  /* Remove the initial subspaces */
  svd->nini = 0;
  svd->ninil = 0;
  PetscFunctionReturn(0);
}

/*@
   SVDGetIterationNumber - Gets the current iteration number. If the
   call to SVDSolve() is complete, then it returns the number of iterations
   carried out by the solution method.

   Not Collective

   Input Parameter:
.  svd - the singular value solver context

   Output Parameter:
.  its - number of iterations

   Note:
   During the i-th iteration this call returns i-1. If SVDSolve() is
   complete, then parameter "its" contains either the iteration number at
   which convergence was successfully reached, or failure was detected.
   Call SVDGetConvergedReason() to determine if the solver converged or
   failed and why.

   Level: intermediate

.seealso: SVDGetConvergedReason(), SVDSetTolerances()
@*/
PetscErrorCode SVDGetIterationNumber(SVD svd,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidIntPointer(its,2);
  *its = svd->its;
  PetscFunctionReturn(0);
}

/*@
   SVDGetConvergedReason - Gets the reason why the SVDSolve() iteration was
   stopped.

   Not Collective

   Input Parameter:
.  svd - the singular value solver context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged
   (see SVDConvergedReason)

   Notes:

   Possible values for reason are
+  SVD_CONVERGED_TOL - converged up to tolerance
.  SVD_CONVERGED_USER - converged due to a user-defined condition
.  SVD_CONVERGED_MAXIT - reached the maximum number of iterations with SVD_CONV_MAXIT criterion
.  SVD_DIVERGED_ITS - required more than max_it iterations to reach convergence
-  SVD_DIVERGED_BREAKDOWN - generic breakdown in method

   Can only be called after the call to SVDSolve() is complete.

   Level: intermediate

.seealso: SVDSetTolerances(), SVDSolve(), SVDConvergedReason
@*/
PetscErrorCode SVDGetConvergedReason(SVD svd,SVDConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidIntPointer(reason,2);
  SVDCheckSolved(svd,1);
  *reason = svd->reason;
  PetscFunctionReturn(0);
}

/*@
   SVDGetConverged - Gets the number of converged singular values.

   Not Collective

   Input Parameter:
.  svd - the singular value solver context

   Output Parameter:
.  nconv - number of converged singular values

   Note:
   This function should be called after SVDSolve() has finished.

   Level: beginner

@*/
PetscErrorCode SVDGetConverged(SVD svd,PetscInt *nconv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidIntPointer(nconv,2);
  SVDCheckSolved(svd,1);
  *nconv = svd->nconv;
  PetscFunctionReturn(0);
}

/*@C
   SVDGetSingularTriplet - Gets the i-th triplet of the singular value decomposition
   as computed by SVDSolve(). The solution consists in the singular value and its left
   and right singular vectors.

   Not Collective, but vectors are shared by all processors that share the SVD

   Input Parameters:
+  svd - singular value solver context
-  i   - index of the solution

   Output Parameters:
+  sigma - singular value
.  u     - left singular vector
-  v     - right singular vector

   Note:
   Both u or v can be NULL if singular vectors are not required.
   Otherwise, the caller must provide valid Vec objects, i.e.,
   they must be created by the calling program with e.g. MatCreateVecs().

   The index i should be a value between 0 and nconv-1 (see SVDGetConverged()).
   Singular triplets are indexed according to the ordering criterion established
   with SVDSetWhichSingularTriplets().

   In the case of GSVD, the solution consists in three vectors u,v,x that are
   returned as follows. Vector x is returned in the right singular vector
   (argument v) and has length equal to the number of columns of A and B.
   The other two vectors are returned stacked on top of each other [u;v] in
   the left singular vector argument, with length equal to m+n (number of rows
   of A plus number of rows of B).

   Level: beginner

.seealso: SVDSolve(), SVDGetConverged(), SVDSetWhichSingularTriplets()
@*/
PetscErrorCode SVDGetSingularTriplet(SVD svd,PetscInt i,PetscReal *sigma,Vec u,Vec v)
{
  PetscErrorCode ierr;
  PetscInt       M,N;
  Vec            w;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveInt(svd,i,2);
  SVDCheckSolved(svd,1);
  if (u) { PetscValidHeaderSpecific(u,VEC_CLASSID,4); PetscCheckSameComm(svd,1,u,4); }
  if (v) { PetscValidHeaderSpecific(v,VEC_CLASSID,5); PetscCheckSameComm(svd,1,v,5); }
  if (i<0) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"The index cannot be negative");
  if (i>=svd->nconv) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"The index can be nconv-1 at most, see SVDGetConverged()");
  if (sigma) *sigma = svd->sigma[svd->perm[i]];
  if (u || v) {
    if (!svd->isgeneralized) {
      ierr = MatGetSize(svd->OP,&M,&N);CHKERRQ(ierr);
      if (M<N) { w = u; u = v; v = w; }
    }
    ierr = SVDComputeVectors(svd);CHKERRQ(ierr);
    if (u) { ierr = BVCopyVec(svd->U,svd->perm[i],u);CHKERRQ(ierr); }
    if (v) { ierr = BVCopyVec(svd->V,svd->perm[i],v);CHKERRQ(ierr); }
  }
  PetscFunctionReturn(0);
}

/*
   SVDComputeResidualNorms_Standard - Computes the norms of the left and
   right residuals associated with the i-th computed singular triplet.

   Input Parameters:
     sigma - singular value
     u,v   - singular vectors
     x,y   - two work vectors with the same dimensions as u,v
@*/
static PetscErrorCode SVDComputeResidualNorms_Standard(SVD svd,PetscReal sigma,Vec u,Vec v,Vec x,Vec y,PetscReal *norm1,PetscReal *norm2)
{
  PetscErrorCode ierr;
  PetscInt       M,N;

  PetscFunctionBegin;
  /* norm1 = ||A*v-sigma*u||_2 */
  if (norm1) {
    ierr = MatMult(svd->OP,v,x);CHKERRQ(ierr);
    ierr = VecAXPY(x,-sigma,u);CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_2,norm1);CHKERRQ(ierr);
  }
  /* norm2 = ||A^T*u-sigma*v||_2 */
  if (norm2) {
    ierr = MatGetSize(svd->OP,&M,&N);CHKERRQ(ierr);
    if (M<N) {
      ierr = MatMult(svd->A,u,y);CHKERRQ(ierr);
    } else {
      ierr = MatMult(svd->AT,u,y);CHKERRQ(ierr);
    }
    ierr = VecAXPY(y,-sigma,v);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,norm2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   SVDComputeResidualNorms_Generalized - In GSVD, compute the residual norms
   norm1 = ||A*x-c*u||_2 and norm2 = ||B*x-s*v||_2.

   Input Parameters:
     sigma - singular value
     x     - right singular vector
*/
static PetscErrorCode SVDComputeResidualNorms_Generalized(SVD svd,PetscReal sigma,Vec uv,Vec x,PetscReal *norm1,PetscReal *norm2)
{
  PetscErrorCode ierr;
  Vec            u,v,ax,bx,nest,aux[2];
  PetscReal      c,s;

  PetscFunctionBegin;
  ierr = MatCreateVecs(svd->OP,NULL,&u);CHKERRQ(ierr);
  ierr = MatCreateVecs(svd->OPb,NULL,&v);CHKERRQ(ierr);
  aux[0] = u;
  aux[1] = v;
  ierr = VecCreateNest(PetscObjectComm((PetscObject)svd),2,NULL,aux,&nest);CHKERRQ(ierr);
  ierr = VecCopy(uv,nest);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&ax);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&bx);CHKERRQ(ierr);

  s = 1.0/PetscSqrtReal(1.0+sigma*sigma);
  c = sigma*s;

  /* norm1 = ||A*x-c*u||_2 */
  if (norm1) {
    ierr = MatMult(svd->OP,x,ax);CHKERRQ(ierr);
    ierr = VecAXPY(ax,-c,u);CHKERRQ(ierr);
    ierr = VecNorm(ax,NORM_2,norm1);CHKERRQ(ierr);
  }
  /* norm2 = ||B*x-s*v||_2 */
  if (norm2) {
    ierr = MatMult(svd->OPb,x,bx);CHKERRQ(ierr);
    ierr = VecAXPY(bx,-s,v);CHKERRQ(ierr);
    ierr = VecNorm(bx,NORM_2,norm2);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&nest);CHKERRQ(ierr);
  ierr = VecDestroy(&ax);CHKERRQ(ierr);
  ierr = VecDestroy(&bx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   SVDComputeError - Computes the error (based on the residual norm) associated
   with the i-th singular triplet.

   Collective on svd

   Input Parameters:
+  svd  - the singular value solver context
.  i    - the solution index
-  type - the type of error to compute

   Output Parameter:
.  error - the error

   Notes:
   The error can be computed in various ways, all of them based on the residual
   norm obtained as sqrt(n1^2+n2^2) with n1 = ||A*v-sigma*u||_2 and
   n2 = ||A^T*u-sigma*v||_2, where sigma is the singular value, u is the left
   singular vector and v is the right singular vector.

   Level: beginner

.seealso: SVDErrorType, SVDSolve()
@*/
PetscErrorCode SVDComputeError(SVD svd,PetscInt i,SVDErrorType type,PetscReal *error)
{
  PetscErrorCode ierr;
  PetscReal      sigma,norm1,norm2;
  Vec            u,v,x,y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveInt(svd,i,2);
  PetscValidLogicalCollectiveEnum(svd,type,3);
  PetscValidRealPointer(error,4);
  SVDCheckSolved(svd,1);

  /* allocate work vectors */
  ierr = SVDSetWorkVecs(svd,2,2);CHKERRQ(ierr);
  u = svd->workl[0];
  v = svd->workr[0];
  x = svd->workl[1];
  y = svd->workr[1];

  /* compute residual norm and error */
  ierr = SVDGetSingularTriplet(svd,i,&sigma,u,v);CHKERRQ(ierr);
  switch (svd->problem_type) {
    case SVD_STANDARD:
      ierr = SVDComputeResidualNorms_Standard(svd,sigma,u,v,x,y,&norm1,&norm2);CHKERRQ(ierr);
      break;
    case SVD_GENERALIZED:
      ierr = SVDComputeResidualNorms_Generalized(svd,sigma,u,v,&norm1,&norm2);CHKERRQ(ierr);
      break;
  }
  *error = PetscSqrtReal(norm1*norm1+norm2*norm2);
  switch (type) {
    case SVD_ERROR_ABSOLUTE:
      break;
    case SVD_ERROR_RELATIVE:
      *error /= sigma;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"Invalid error type");
  }
  PetscFunctionReturn(0);
}

