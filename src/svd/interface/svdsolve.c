/*
      SVD routines related to the solution process.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/svdimpl.h"   /*I "slepcsvd.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve"
/*@
   SVDSolve - Solves the singular value problem.

   Collective on SVD

   Input Parameter:
.  svd - singular value solver context obtained from SVDCreate()

   Options Database:
.   -svd_view - print information about the solver used

   Level: beginner

.seealso: SVDCreate(), SVDSetUp(), SVDDestroy() 
@*/
PetscErrorCode SVDSolve(SVD svd) 
{
  PetscErrorCode ierr;
  PetscTruth     flg;
  PetscInt       i,*workperm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);

  if (!svd->setupcalled) { ierr = SVDSetUp(svd);CHKERRQ(ierr); }
  svd->its = 0;
  svd->matvecs = 0;
  svd->nconv = 0;
  svd->reason = SVD_CONVERGED_ITERATING;
  ierr = IPResetOperationCounters(svd->ip);CHKERRQ(ierr);
  for (i=0;i<svd->ncv;i++) svd->sigma[i]=svd->errest[i]=0.0;
  SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,svd->ncv);

  ierr = PetscLogEventBegin(SVD_Solve,svd,0,0,0);CHKERRQ(ierr);
  ierr = (*svd->ops->solve)(svd);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SVD_Solve,svd,0,0,0);CHKERRQ(ierr);

  /* sort singular triplets */
  if (svd->which == SVD_SMALLEST) {
    for (i=0;i<svd->nconv;i++) svd->perm[i] = i;
    ierr = PetscSortRealWithPermutation(svd->nconv,svd->sigma,svd->perm);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc(sizeof(PetscInt)*svd->nconv,&workperm);CHKERRQ(ierr);
    for (i=0;i<svd->nconv;i++) workperm[i] = i;
    ierr = PetscSortRealWithPermutation(svd->nconv,svd->sigma,workperm);CHKERRQ(ierr);
    for (i=0;i<svd->nconv;i++) svd->perm[i] = workperm[svd->nconv-i-1];
    ierr = PetscFree(workperm);CHKERRQ(ierr);
  }

  ierr = PetscOptionsHasName(((PetscObject)svd)->prefix,"-svd_view",&flg);CHKERRQ(ierr); 
  if (flg && !PetscPreLoadingOn) { ierr = SVDView(svd,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetIterationNumber"
/*@
   SVDGetIterationNumber - Gets the current iteration number. If the 
   call to SVDSolve() is complete, then it returns the number of iterations 
   carried out by the solution method.
 
   Not Collective

   Input Parameter:
.  svd - the singular value solver context

   Output Parameter:
.  its - number of iterations

   Level: intermediate

   Notes:
      During the i-th iteration this call returns i-1. If SVDSolve() is 
      complete, then parameter "its" contains either the iteration number at
      which convergence was successfully reached, or failure was detected.  
      Call SVDGetConvergedReason() to determine if the solver converged or 
      failed and why.

@*/
PetscErrorCode SVDGetIterationNumber(SVD svd,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidIntPointer(its,2);
  *its = svd->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetConvergedReason"
/*@C
   SVDGetConvergedReason - Gets the reason why the SVDSolve() iteration was 
   stopped.

   Not Collective

   Input Parameter:
.  svd - the singular value solver context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged
   (see SVDConvergedReason)

   Possible values for reason:
+  SVD_CONVERGED_TOL - converged up to tolerance
.  SVD_DIVERGED_ITS - required more than its to reach convergence
-  SVD_DIVERGED_BREAKDOWN - generic breakdown in method

   Level: intermediate

   Notes: Can only be called after the call to SVDSolve() is complete.

.seealso: SVDSetTolerances(), SVDSolve(), SVDConvergedReason
@*/
PetscErrorCode SVDGetConvergedReason(SVD svd,SVDConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidIntPointer(reason,2);
  *reason = svd->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetConverged"
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
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidIntPointer(nconv,2);
  if (svd->reason == SVD_CONVERGED_ITERATING) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "SVDSolve must be called first"); 
  }
  *nconv = svd->nconv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetSingularTriplet" 
/*@
   SVDGetSingularTriplet - Gets the i-th triplet of the singular value decomposition
   as computed by SVDSolve(). The solution consists in the singular value and its left 
   and right singular vectors.

   Not Collective

   Input Parameters:
+  svd - singular value solver context 
-  i   - index of the solution

   Output Parameters:
+  sigma - singular value
.  u     - left singular vector
-  v     - right singular vector

   Note:
   The index i should be a value between 0 and nconv-1 (see SVDGetConverged()).
   Both U or V can be PETSC_NULL if singular vectors are not required. 

   Level: beginner

.seealso: SVDSolve(),  SVDGetConverged()
@*/
PetscErrorCode SVDGetSingularTriplet(SVD svd, PetscInt i, PetscReal *sigma, Vec u, Vec v)
{
  PetscErrorCode ierr;
  PetscReal      norm;
  PetscInt       j,nloc;
  PetscScalar    *pU;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidPointer(sigma,3);
  if (svd->reason == SVD_CONVERGED_ITERATING) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "SVDSolve must be called first"); 
  }
  if (i<0 || i>=svd->nconv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }
  *sigma = svd->sigma[svd->perm[i]];
  if (u) {
    PetscValidHeaderSpecific(u,VEC_COOKIE,4);
    if (!svd->U) {
      ierr = PetscMalloc(sizeof(Vec)*svd->ncv,&svd->U);CHKERRQ(ierr);
      ierr = SVDMatGetLocalSize(svd,&nloc,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscMalloc(svd->ncv*nloc*sizeof(PetscScalar),&pU);CHKERRQ(ierr);
      for (i=0;i<svd->ncv;i++) {
        ierr = VecCreateMPIWithArray(((PetscObject)svd)->comm,nloc,PETSC_DECIDE,pU+i*nloc,&svd->U[i]);CHKERRQ(ierr);
      }
      for (j=0;j<svd->nconv;j++) {
        ierr = SVDMatMult(svd,PETSC_FALSE,svd->V[j],svd->U[j]);CHKERRQ(ierr);
        ierr = IPOrthogonalize(svd->ip,j,PETSC_NULL,svd->U,svd->U[j],PETSC_NULL,&norm,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        ierr = VecScale(svd->U[j],1.0/norm);CHKERRQ(ierr);
      }
    }
    ierr = VecCopy(svd->U[svd->perm[i]],u);CHKERRQ(ierr);
  }
  if (v) {
    PetscValidHeaderSpecific(v,VEC_COOKIE,5);   
    ierr = VecCopy(svd->V[svd->perm[i]],v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDComputeResidualNorms"
/*@
   SVDComputeResidualNorms - Computes the norms of the residual vectors associated with 
   the i-th computed singular triplet.

   Collective on SVD

   Input Parameters:
+  svd  - the singular value solver context
-  i    - the solution index

   Output Parameters:
+  norm1 - the norm ||A*v-sigma*u||_2 where sigma is the 
           singular value, u and v are the left and right singular vectors. 
-  norm2 - the norm ||A^T*u-sigma*v||_2 with the same sigma, u and v

   Note:
   The index i should be a value between 0 and nconv-1 (see SVDGetConverged()).
   Both output parameters can be PETSC_NULL on input if not needed.

   Level: beginner

.seealso: SVDSolve(), SVDGetConverged(), SVDComputeRelativeError()
@*/
PetscErrorCode SVDComputeResidualNorms(SVD svd, PetscInt i, PetscReal *norm1, PetscReal *norm2)
{
  PetscErrorCode ierr;
  Vec            u,v,x = PETSC_NULL,y = PETSC_NULL;
  PetscReal      sigma;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (svd->reason == SVD_CONVERGED_ITERATING) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "SVDSolve must be called first"); 
  }
  if (i<0 || i>=svd->nconv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }
  
  ierr = SVDMatGetVecs(svd,&v,&u);CHKERRQ(ierr);
  ierr = SVDGetSingularTriplet(svd,i,&sigma,u,v);CHKERRQ(ierr);
  if (norm1) {
    ierr = VecDuplicate(u,&x);CHKERRQ(ierr);
    ierr = SVDMatMult(svd,PETSC_FALSE,v,x);CHKERRQ(ierr);
    ierr = VecAXPY(x,-sigma,u);CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_2,norm1);CHKERRQ(ierr);
  }
  if (norm2) {
    ierr = VecDuplicate(v,&y);CHKERRQ(ierr);
    ierr = SVDMatMult(svd,PETSC_TRUE,u,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,-sigma,v);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,norm2);CHKERRQ(ierr);
  }

  ierr = VecDestroy(v);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  if (x) { ierr = VecDestroy(x);CHKERRQ(ierr); }
  if (y) { ierr = VecDestroy(y);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDComputeRelativeError"
/*@
   SVDComputeRelativeError - Computes the relative error bound associated 
   with the i-th singular triplet.

   Collective on SVD

   Input Parameter:
+  svd - the singular value solver context
-  i   - the solution index

   Output Parameter:
.  error - the relative error bound, computed as sqrt(n1^2+n2^2)/sigma
   where n1 = ||A*v-sigma*u||_2 , n2 = ||A^T*u-sigma*v||_2 , sigma is the singular value, 
   u and v are the left and right singular vectors.
   If sigma is too small the relative error is computed as sqrt(n1^2+n2^2).

   Level: beginner

.seealso: SVDSolve(), SVDComputeResidualNorms()
@*/
PetscErrorCode SVDComputeRelativeError(SVD svd,PetscInt i,PetscReal *error)
{
  PetscErrorCode ierr;
  PetscReal      sigma,norm1,norm2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidPointer(error,2);
  ierr = SVDGetSingularTriplet(svd,i,&sigma,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = SVDComputeResidualNorms(svd,i,&norm1,&norm2);CHKERRQ(ierr);
  *error = sqrt(norm1*norm1+norm2*norm2);
  if (sigma>*error) *error /= sigma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetOperationCounters"
/*@
   SVDGetOperationCounters - Gets the total number of matrix vector and dot 
   products used by the SVD object during the last SVDSolve() call.

   Not Collective

   Input Parameter:
.  svd - SVD context

   Output Parameter:
+  matvecs - number of matrix vector product operations
-  dots    - number of dot product operations

   Notes:
   These counters are reset to zero at each successive call to SVDSolve().

   Level: intermediate

@*/
PetscErrorCode SVDGetOperationCounters(SVD svd,PetscInt* matvecs,PetscInt* dots)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (matvecs) *matvecs = svd->matvecs; 
  if (dots) {
    ierr = IPGetOperationCounters(svd->ip,dots);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
