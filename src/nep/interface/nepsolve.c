/*
      NEP routines related to the solution process.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/nepimpl.h>       /*I "slepcnep.h" I*/
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "NEPComputeVectors"
PETSC_STATIC_INLINE PetscErrorCode NEPComputeVectors(NEP nep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  NEPCheckSolved(nep,1);
  switch (nep->state) {
  case NEP_STATE_SOLVED:
    if (nep->ops->computevectors) {
      ierr = (*nep->ops->computevectors)(nep);CHKERRQ(ierr);
    }
    break;
  default:
    break;
  }
  nep->state = NEP_STATE_EIGENVECTORS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSolve"
/*@
   NEPSolve - Solves the nonlinear eigensystem.

   Collective on NEP

   Input Parameter:
.  nep - eigensolver context obtained from NEPCreate()

   Options Database Keys:
+  -nep_view - print information about the solver used
-  -nep_plot_eigs - plot computed eigenvalues

   Level: beginner

.seealso: NEPCreate(), NEPSetUp(), NEPDestroy(), NEPSetTolerances()
@*/
PetscErrorCode NEPSolve(NEP nep)
{
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         re,im;
  PetscBool         flg;
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscDraw         draw;
  PetscDrawSP       drawsp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscLogEventBegin(NEP_Solve,nep,0,0,0);CHKERRQ(ierr);

  /* call setup */
  ierr = NEPSetUp(nep);CHKERRQ(ierr);
  nep->nconv = 0;
  nep->its = 0;
  for (i=0;i<nep->ncv;i++) {
    nep->eigr[i]   = 0.0;
    nep->eigi[i]   = 0.0;
    nep->errest[i] = 0.0;
  }
  nep->ktol = 0.1;
  ierr = NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->errest,nep->ncv);CHKERRQ(ierr);

  ierr = (*nep->ops->solve)(nep);CHKERRQ(ierr);

  if (!nep->reason) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");

  nep->state = NEP_STATE_SOLVED;

  if (nep->refine==NEP_REFINE_SIMPLE && nep->rits>0) {
    ierr = NEPComputeVectors(nep);CHKERRQ(ierr);
    ierr = NEPNewtonRefinementSimple(nep,&nep->rits,&nep->reftol,nep->nconv);CHKERRQ(ierr);
    nep->state = NEP_STATE_EIGENVECTORS;
  }

  /* sort eigenvalues according to nep->which parameter */
  ierr = SlepcSortEigenvalues(nep->sc,nep->nconv,nep->eigr,nep->eigi,nep->perm);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(NEP_Solve,nep,0,0,0);CHKERRQ(ierr);

  /* various viewers */
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->prefix,"-nep_view",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = NEPView(nep,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)nep)->prefix,"-nep_plot_eigs",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Computed Eigenvalues",PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
    for (i=0;i<nep->nconv;i++) {
#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(nep->eigr[i]);
      im = PetscImaginaryPart(nep->eigr[i]);
#else
      re = nep->eigr[i];
      im = nep->eigi[i];
#endif
      ierr = PetscDrawSPAddPoint(drawsp,&re,&im);CHKERRQ(ierr);
    }
    ierr = PetscDrawSPDraw(drawsp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscDrawSPDestroy(&drawsp);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Remove the initial subspace */
  nep->nini = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPProjectOperator"
/*@
   NEPProjectOperator - Computes the projection of the nonlinear operator.

   Collective on NEP

   Input Parameters:
+  nep - the nonlinear eigensolver context
.  j0  - initial index
-  j1  - final index

   Notes:
   This is available for split operator only.

   The nonlinear operator T(lambda) is projected onto span(V), where V is
   an orthonormal basis built internally by the solver. The projected
   operator is equal to sum_i V'*A_i*V*f_i(lambda), so this function
   computes all matrices Ei = V'*A_i*V, and stores them in the extra
   matrices inside DS. Only rows/columns in the range [j0,j1-1] are computed,
   the previous ones are assumed to be available already.

   Level: developer

.seealso: NEPSetSplitOperator()
@*/
PetscErrorCode NEPProjectOperator(NEP nep,PetscInt j0,PetscInt j1)
{
  PetscErrorCode ierr;
  PetscInt       k;
  Mat            G;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,j0,2);
  PetscValidLogicalCollectiveInt(nep,j1,3);
  if (!nep->split) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_WRONGSTATE,"This solver requires a split operator");
  ierr = BVSetActiveColumns(nep->V,j0,j1);CHKERRQ(ierr);
  for (k=0;k<nep->nt;k++) {
    ierr = DSGetMat(nep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
    ierr = BVMatProject(nep->V,nep->A[k],nep->V,G);CHKERRQ(ierr);
    ierr = DSRestoreMat(nep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPApplyFunction"
/*@
   NEPApplyFunction - Applies the nonlinear function T(lambda) to a given vector.

   Collective on NEP

   Input Parameters:
+  nep    - the nonlinear eigensolver context
.  lambda - scalar argument
.  x      - vector to be multiplied against
-  v      - workspace vector

   Output Parameters:
+  y   - result vector
.  A   - Function matrix
-  B   - optional preconditioning matrix

   Note:
   If the nonlinear operator is represented in split form, the result 
   y = T(lambda)*x is computed without building T(lambda) explicitly. In
   that case, parameters A and B are not used. Otherwise, the matrix
   T(lambda) is built and the effect is the same as a call to
   NEPComputeFunction() followed by a MatMult().

   Level: developer

.seealso: NEPSetSplitOperator(), NEPComputeFunction()
@*/
PetscErrorCode NEPApplyFunction(NEP nep,PetscScalar lambda,Vec x,Vec v,Vec y,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveScalar(nep,lambda,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  PetscValidHeaderSpecific(y,VEC_CLASSID,5);
  if (nep->split) {
    ierr = VecZeroEntries(y);CHKERRQ(ierr);
    for (i=0;i<nep->nt;i++) {
      ierr = FNEvaluateFunction(nep->f[i],lambda,&alpha);CHKERRQ(ierr);
      ierr = MatMult(nep->A[i],x,v);CHKERRQ(ierr);
      ierr = VecAXPY(y,alpha,v);CHKERRQ(ierr);
    }
  } else {
    ierr = NEPComputeFunction(nep,lambda,A,B);CHKERRQ(ierr);
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPApplyJacobian"
/*@
   NEPApplyJacobian - Applies the nonlinear Jacobian T'(lambda) to a given vector.

   Collective on NEP

   Input Parameters:
+  nep    - the nonlinear eigensolver context
.  lambda - scalar argument
.  x      - vector to be multiplied against
-  v      - workspace vector

   Output Parameters:
+  y   - result vector
-  A   - Jacobian matrix

   Note:
   If the nonlinear operator is represented in split form, the result 
   y = T'(lambda)*x is computed without building T'(lambda) explicitly. In
   that case, parameter A is not used. Otherwise, the matrix
   T'(lambda) is built and the effect is the same as a call to
   NEPComputeJacobian() followed by a MatMult().

   Level: developer

.seealso: NEPSetSplitOperator(), NEPComputeJacobian()
@*/
PetscErrorCode NEPApplyJacobian(NEP nep,PetscScalar lambda,Vec x,Vec v,Vec y,Mat A)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveScalar(nep,lambda,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  PetscValidHeaderSpecific(y,VEC_CLASSID,5);
  if (nep->split) {
    ierr = VecZeroEntries(y);CHKERRQ(ierr);
    for (i=0;i<nep->nt;i++) {
      ierr = FNEvaluateDerivative(nep->f[i],lambda,&alpha);CHKERRQ(ierr);
      ierr = MatMult(nep->A[i],x,v);CHKERRQ(ierr);
      ierr = VecAXPY(y,alpha,v);CHKERRQ(ierr);
    }
  } else {
    ierr = NEPComputeJacobian(nep,lambda,A);CHKERRQ(ierr);
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetIterationNumber"
/*@
   NEPGetIterationNumber - Gets the current iteration number. If the
   call to NEPSolve() is complete, then it returns the number of iterations
   carried out by the solution method.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameter:
.  its - number of iterations

   Level: intermediate

   Note:
   During the i-th iteration this call returns i-1. If NEPSolve() is
   complete, then parameter "its" contains either the iteration number at
   which convergence was successfully reached, or failure was detected.
   Call NEPGetConvergedReason() to determine if the solver converged or
   failed and why.

.seealso: NEPGetConvergedReason(), NEPSetTolerances()
@*/
PetscErrorCode NEPGetIterationNumber(NEP nep,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidIntPointer(its,2);
  *its = nep->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetConverged"
/*@
   NEPGetConverged - Gets the number of converged eigenpairs.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameter:
.  nconv - number of converged eigenpairs

   Note:
   This function should be called after NEPSolve() has finished.

   Level: beginner

.seealso: NEPSetDimensions(), NEPSolve()
@*/
PetscErrorCode NEPGetConverged(NEP nep,PetscInt *nconv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidIntPointer(nconv,2);
  NEPCheckSolved(nep,1);
  *nconv = nep->nconv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetConvergedReason"
/*@C
   NEPGetConvergedReason - Gets the reason why the NEPSolve() iteration was
   stopped.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged

   Possible values for reason:
+  NEP_CONVERGED_FNORM_ABS - function norm satisfied absolute tolerance
.  NEP_CONVERGED_FNORM_RELATIVE - function norm satisfied relative tolerance
.  NEP_CONVERGED_SNORM_RELATIVE - step norm satisfied relative tolerance
.  NEP_DIVERGED_LINEAR_SOLVE - inner linear solve failed
.  NEP_DIVERGED_FUNCTION_COUNT - reached maximum allowed function evaluations
.  NEP_DIVERGED_MAX_IT - required more than its to reach convergence
.  NEP_DIVERGED_BREAKDOWN - generic breakdown in method
-  NEP_DIVERGED_FNORM_NAN - Inf or NaN detected in function evaluation

   Note:
   Can only be called after the call to NEPSolve() is complete.

   Level: intermediate

.seealso: NEPSetTolerances(), NEPSolve(), NEPConvergedReason
@*/
PetscErrorCode NEPGetConvergedReason(NEP nep,NEPConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(reason,2);
  NEPCheckSolved(nep,1);
  *reason = nep->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetEigenpair"
/*@
   NEPGetEigenpair - Gets the i-th solution of the eigenproblem as computed by
   NEPSolve(). The solution consists in both the eigenvalue and the eigenvector.

   Logically Collective on NEP

   Input Parameters:
+  nep - nonlinear eigensolver context
-  i   - index of the solution

   Output Parameters:
+  eigr - real part of eigenvalue
.  eigi - imaginary part of eigenvalue
.  Vr   - real part of eigenvector
-  Vi   - imaginary part of eigenvector

   Notes:
   If the eigenvalue is real, then eigi and Vi are set to zero. If PETSc is
   configured with complex scalars the eigenvalue is stored
   directly in eigr (eigi is set to zero) and the eigenvector in Vr (Vi is
   set to zero).

   The index i should be a value between 0 and nconv-1 (see NEPGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established
   with NEPSetWhichEigenpairs().

   Level: beginner

.seealso: NEPSolve(), NEPGetConverged(), NEPSetWhichEigenpairs()
@*/
PetscErrorCode NEPGetEigenpair(NEP nep,PetscInt i,PetscScalar *eigr,PetscScalar *eigi,Vec Vr,Vec Vi)
{
  PetscInt       k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,i,2);
  if (Vr) { PetscValidHeaderSpecific(Vr,VEC_CLASSID,5); PetscCheckSameComm(nep,1,Vr,5); }
  if (Vi) { PetscValidHeaderSpecific(Vi,VEC_CLASSID,6); PetscCheckSameComm(nep,1,Vi,6); }
  NEPCheckSolved(nep,1);
  if (i<0 || i>=nep->nconv) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range");

  ierr = NEPComputeVectors(nep);CHKERRQ(ierr);
  if (!nep->perm) k = i;
  else k = nep->perm[i];

  /* eigenvalue */
#if defined(PETSC_USE_COMPLEX)
  if (eigr) *eigr = nep->eigr[k];
  if (eigi) *eigi = 0;
#else
  if (eigr) *eigr = nep->eigr[k];
  if (eigi) *eigi = nep->eigi[k];
#endif

  /* eigenvector */
#if defined(PETSC_USE_COMPLEX)
  if (Vr) { ierr = BVCopyVec(nep->V,k,Vr);CHKERRQ(ierr); }
  if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
#else
  if (nep->eigi[k]>0) { /* first value of conjugate pair */
    if (Vr) { ierr = BVCopyVec(nep->V,k,Vr);CHKERRQ(ierr); }
    if (Vi) { ierr = BVCopyVec(nep->V,k+1,Vi);CHKERRQ(ierr); }
  } else if (nep->eigi[k]<0) { /* second value of conjugate pair */
    if (Vr) { ierr = BVCopyVec(nep->V,k-1,Vr);CHKERRQ(ierr); }
    if (Vi) {
      ierr = BVCopyVec(nep->V,k,Vi);CHKERRQ(ierr);
      ierr = VecScale(Vi,-1.0);CHKERRQ(ierr);
    }
  } else { /* real eigenvalue */
    if (Vr) { ierr = BVCopyVec(nep->V,k,Vr);CHKERRQ(ierr); }
    if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetErrorEstimate"
/*@
   NEPGetErrorEstimate - Returns the error estimate associated to the i-th
   computed eigenpair.

   Not Collective

   Input Parameter:
+  nep - nonlinear eigensolver context
-  i   - index of eigenpair

   Output Parameter:
.  errest - the error estimate

   Notes:
   This is the error estimate used internally by the eigensolver. The actual
   error bound can be computed with NEPComputeRelativeError().

   Level: advanced

.seealso: NEPComputeRelativeError()
@*/
PetscErrorCode NEPGetErrorEstimate(NEP nep,PetscInt i,PetscReal *errest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(errest,3);
  NEPCheckSolved(nep,1);
  if (i<0 || i>=nep->nconv) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range");
  if (nep->perm) i = nep->perm[i];
  if (errest) *errest = nep->errest[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPComputeResidualNorm_Private"
/*
   NEPComputeResidualNorm_Private - Computes the norm of the residual vector
   associated with an eigenpair.
*/
PetscErrorCode NEPComputeResidualNorm_Private(NEP nep,PetscScalar lambda,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  Vec            u;
  Mat            T=nep->function;

  PetscFunctionBegin;
  ierr = BVGetVec(nep->V,&u);CHKERRQ(ierr);
  ierr = NEPComputeFunction(nep,lambda,T,T);CHKERRQ(ierr);
  ierr = MatMult(T,x,u);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,norm);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPComputeResidualNorm"
/*@
   NEPComputeResidualNorm - Computes the norm of the residual vector associated with
   the i-th computed eigenpair.

   Collective on NEP

   Input Parameter:
+  nep - the nonlinear eigensolver context
-  i   - the solution index

   Output Parameter:
.  norm - the residual norm, computed as ||T(lambda)x||_2 where lambda is the
   eigenvalue and x is the eigenvector.

   Notes:
   The index i should be a value between 0 and nconv-1 (see NEPGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established
   with NEPSetWhichEigenpairs().

   Level: beginner

.seealso: NEPSolve(), NEPGetConverged(), NEPSetWhichEigenpairs()
@*/
PetscErrorCode NEPComputeResidualNorm(NEP nep,PetscInt i,PetscReal *norm)
{
  PetscErrorCode ierr;
  Vec            xr,xi;
  PetscScalar    kr,ki;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,i,2);
  PetscValidPointer(norm,3);
  NEPCheckSolved(nep,1);
  ierr = BVGetVec(nep->V,&xr);CHKERRQ(ierr);
  ierr = BVGetVec(nep->V,&xi);CHKERRQ(ierr);
  ierr = NEPGetEigenpair(nep,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (ki) SETERRQ(PETSC_COMM_SELF,1,"Not implemented for complex eigenvalues with real scalars");
#endif
  ierr = NEPComputeResidualNorm_Private(nep,kr,xr,norm);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPComputeRelativeError_Private"
/*
   NEPComputeRelativeError_Private - Computes the relative error bound
   associated with an eigenpair.
*/
PetscErrorCode NEPComputeRelativeError_Private(NEP nep,PetscScalar lambda,Vec x,PetscReal *error)
{
  PetscErrorCode ierr;
  PetscReal      norm,er;

  PetscFunctionBegin;
  ierr = NEPComputeResidualNorm_Private(nep,lambda,x,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&er);CHKERRQ(ierr);
  if (PetscAbsScalar(lambda) > norm) {
    *error = norm/(PetscAbsScalar(lambda)*er);
  } else {
    *error = norm/er;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPComputeRelativeError"
/*@
   NEPComputeRelativeError - Computes the relative error bound associated
   with the i-th computed eigenpair.

   Collective on NEP

   Input Parameter:
+  nep - the nonlinear eigensolver context
-  i   - the solution index

   Output Parameter:
.  error - the relative error bound, computed as ||T(lambda)x||_2/||lambda*x||_2
   where lambda is the eigenvalue and x is the eigenvector.
   If lambda=0 the relative error is computed as ||T(lambda)x||_2/||x||_2.

   Level: beginner

.seealso: NEPSolve(), NEPComputeResidualNorm(), NEPGetErrorEstimate()
@*/
PetscErrorCode NEPComputeRelativeError(NEP nep,PetscInt i,PetscReal *error)
{
  PetscErrorCode ierr;
  Vec            xr,xi;
  PetscScalar    kr,ki;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,i,2);
  PetscValidPointer(error,3);
  NEPCheckSolved(nep,1);
  ierr = BVGetVec(nep->V,&xr);CHKERRQ(ierr);
  ierr = BVGetVec(nep->V,&xi);CHKERRQ(ierr);
  ierr = NEPGetEigenpair(nep,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (ki) SETERRQ(PETSC_COMM_SELF,1,"Not implemented for complex eigenvalues with real scalars");
#endif
  ierr = NEPComputeRelativeError_Private(nep,kr,xr,error);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPComputeFunction"
/*@
   NEPComputeFunction - Computes the function matrix T(lambda) that has been
   set with NEPSetFunction().

   Collective on NEP and Mat

   Input Parameters:
+  nep    - the NEP context
-  lambda - the scalar argument

   Output Parameters:
+  A   - Function matrix
-  B   - optional preconditioning matrix

   Notes:
   NEPComputeFunction() is typically used within nonlinear eigensolvers
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso: NEPSetFunction(), NEPGetFunction()
@*/
PetscErrorCode NEPComputeFunction(NEP nep,PetscScalar lambda,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);

  if (nep->split) {

    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    for (i=0;i<nep->nt;i++) {
      ierr = FNEvaluateFunction(nep->f[i],lambda,&alpha);CHKERRQ(ierr);
      ierr = MatAXPY(A,alpha,nep->A[i],nep->mstr);CHKERRQ(ierr);
    }
    if (A != B) SETERRQ(PetscObjectComm((PetscObject)nep),1,"Not implemented");

  } else {

    if (!nep->computefunction) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_USER,"Must call NEPSetFunction() first");

    ierr = PetscLogEventBegin(NEP_FunctionEval,nep,A,B,0);CHKERRQ(ierr);

    PetscStackPush("NEP user Function function");
    ierr = (*nep->computefunction)(nep,lambda,A,B,nep->functionctx);CHKERRQ(ierr);
    PetscStackPop;

    ierr = PetscLogEventEnd(NEP_FunctionEval,nep,A,B,0);CHKERRQ(ierr);
    nep->nfuncs++;

  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPComputeJacobian"
/*@
   NEPComputeJacobian - Computes the Jacobian matrix T'(lambda) that has been
   set with NEPSetJacobian().

   Collective on NEP and Mat

   Input Parameters:
+  nep    - the NEP context
-  lambda - the scalar argument

   Output Parameters:
.  A   - Jacobian matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear eigensolvers.

   Level: developer

.seealso: NEPSetJacobian(), NEPGetJacobian()
@*/
PetscErrorCode NEPComputeJacobian(NEP nep,PetscScalar lambda,Mat A)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);

  if (nep->split) {

    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    for (i=0;i<nep->nt;i++) {
      ierr = FNEvaluateDerivative(nep->f[i],lambda,&alpha);CHKERRQ(ierr);
      ierr = MatAXPY(A,alpha,nep->A[i],nep->mstr);CHKERRQ(ierr);
    }

  } else {

    if (!nep->computejacobian) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_USER,"Must call NEPSetJacobian() first");

    ierr = PetscLogEventBegin(NEP_JacobianEval,nep,A,0,0);CHKERRQ(ierr);

    PetscStackPush("NEP user Jacobian function");
    ierr = (*nep->computejacobian)(nep,lambda,A,nep->jacobianctx);CHKERRQ(ierr);
    PetscStackPop;

    ierr = PetscLogEventEnd(NEP_JacobianEval,nep,A,0,0);CHKERRQ(ierr);

  }
  PetscFunctionReturn(0);
}

