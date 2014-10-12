/*
      PEP routines related to the solution process.

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

#include <slepc-private/pepimpl.h>       /*I "slepcpep.h" I*/
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "PEPComputeVectors"
PETSC_STATIC_INLINE PetscErrorCode PEPComputeVectors(PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PEPCheckSolved(pep,1);
  switch (pep->state) {
  case PEP_STATE_SOLVED:
    if (pep->ops->computevectors) {
      ierr = (*pep->ops->computevectors)(pep);CHKERRQ(ierr);
    }
    break;
  default:
    break;
  }
  pep->state = PEP_STATE_EIGENVECTORS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSolve"
/*@
   PEPSolve - Solves the polynomial eigensystem.

   Collective on PEP

   Input Parameter:
.  pep - eigensolver context obtained from PEPCreate()

   Options Database Keys:
+  -pep_view - print information about the solver used
-  -pep_plot_eigs - plot computed eigenvalues

   Level: beginner

.seealso: PEPCreate(), PEPSetUp(), PEPDestroy(), PEPSetTolerances()
@*/
PetscErrorCode PEPSolve(PEP pep)
{
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         re,im;
  PetscBool         flg,islinear;
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscDraw         draw;
  PetscDrawSP       drawsp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  ierr = PetscLogEventBegin(PEP_Solve,pep,0,0,0);CHKERRQ(ierr);

  /* call setup */
  ierr = PEPSetUp(pep);CHKERRQ(ierr);
  pep->nconv = 0;
  pep->its   = 0;
  for (i=0;i<pep->ncv;i++) {
    pep->eigr[i]   = 0.0;
    pep->eigi[i]   = 0.0;
    pep->errest[i] = 0.0;
  }
  ierr = PEPMonitor(pep,pep->its,pep->nconv,pep->eigr,pep->eigi,pep->errest,pep->ncv);CHKERRQ(ierr);

  ierr = (*pep->ops->solve)(pep);CHKERRQ(ierr);
  
  ierr = PetscObjectTypeCompare((PetscObject)pep,PEPLINEAR,&islinear);CHKERRQ(ierr);
  if (!islinear) {
    ierr = STPostSolve(pep->st);CHKERRQ(ierr);
  }

  if (!pep->reason) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");

  if (!islinear) {
    /* Map eigenvalues back to the original problem */
    ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = STBackTransform(pep->st,pep->nconv,pep->eigr,pep->eigi);CHKERRQ(ierr);
    }
  }

  pep->state = PEP_STATE_SOLVED;

  if (pep->refine==PEP_REFINE_SIMPLE && pep->rits>0) {
    ierr = PEPComputeVectors(pep);CHKERRQ(ierr);
    ierr = PEPNewtonRefinementSimple(pep,&pep->rits,&pep->rtol,pep->nconv);CHKERRQ(ierr);
    pep->state = PEP_STATE_EIGENVECTORS;
  }

#if !defined(PETSC_USE_COMPLEX)
  /* reorder conjugate eigenvalues (positive imaginary first) */
  for (i=0;i<pep->nconv-1;i++) {
    if (pep->eigi[i] != 0) {
      if (pep->eigi[i] < 0) {
        pep->eigi[i] = -pep->eigi[i];
        pep->eigi[i+1] = -pep->eigi[i+1];
        /* the next correction only works with eigenvectors */
        ierr = PEPComputeVectors(pep);CHKERRQ(ierr);
        ierr = BVScaleColumn(pep->V,i+1,-1.0);CHKERRQ(ierr);
      }
      i++;
    }
  }
#endif

  /* sort eigenvalues according to pep->which parameter */
  ierr = SlepcSortEigenvalues(pep->sc,pep->nconv,pep->eigr,pep->eigi,pep->perm);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(PEP_Solve,pep,0,0,0);CHKERRQ(ierr);

  /* various viewers */
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)pep),((PetscObject)pep)->prefix,"-pep_view",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = PEPView(pep,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)pep)->prefix,"-pep_plot_eigs",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Computed Eigenvalues",PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
    for (i=0;i<pep->nconv;i++) {
#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(pep->eigr[i]);
      im = PetscImaginaryPart(pep->eigr[i]);
#else
      re = pep->eigr[i];
      im = pep->eigi[i];
#endif
      ierr = PetscDrawSPAddPoint(drawsp,&re,&im);CHKERRQ(ierr);
    }
    ierr = PetscDrawSPDraw(drawsp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscDrawSPDestroy(&drawsp);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Remove the initial subspace */
  pep->nini = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetIterationNumber"
/*@
   PEPGetIterationNumber - Gets the current iteration number. If the
   call to PEPSolve() is complete, then it returns the number of iterations
   carried out by the solution method.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameter:
.  its - number of iterations

   Level: intermediate

   Note:
   During the i-th iteration this call returns i-1. If PEPSolve() is
   complete, then parameter "its" contains either the iteration number at
   which convergence was successfully reached, or failure was detected.
   Call PEPGetConvergedReason() to determine if the solver converged or
   failed and why.

.seealso: PEPGetConvergedReason(), PEPSetTolerances()
@*/
PetscErrorCode PEPGetIterationNumber(PEP pep,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidIntPointer(its,2);
  *its = pep->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetConverged"
/*@
   PEPGetConverged - Gets the number of converged eigenpairs.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameter:
.  nconv - number of converged eigenpairs

   Note:
   This function should be called after PEPSolve() has finished.

   Level: beginner

.seealso: PEPSetDimensions(), PEPSolve()
@*/
PetscErrorCode PEPGetConverged(PEP pep,PetscInt *nconv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidIntPointer(nconv,2);
  PEPCheckSolved(pep,1);
  *nconv = pep->nconv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetConvergedReason"
/*@C
   PEPGetConvergedReason - Gets the reason why the PEPSolve() iteration was
   stopped.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged

   Possible values for reason:
+  PEP_CONVERGED_TOL - converged up to tolerance
.  PEP_DIVERGED_ITS - required more than its to reach convergence
-  PEP_DIVERGED_BREAKDOWN - generic breakdown in method

   Note:
   Can only be called after the call to PEPSolve() is complete.

   Level: intermediate

.seealso: PEPSetTolerances(), PEPSolve(), PEPConvergedReason
@*/
PetscErrorCode PEPGetConvergedReason(PEP pep,PEPConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(reason,2);
  PEPCheckSolved(pep,1);
  *reason = pep->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetEigenpair"
/*@
   PEPGetEigenpair - Gets the i-th solution of the eigenproblem as computed by
   PEPSolve(). The solution consists in both the eigenvalue and the eigenvector.

   Logically Collective on EPS

   Input Parameters:
+  pep - polynomial eigensolver context
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

   The index i should be a value between 0 and nconv-1 (see PEPGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established
   with PEPSetWhichEigenpairs().

   Level: beginner

.seealso: PEPSolve(), PEPGetConverged(), PEPSetWhichEigenpairs()
@*/
PetscErrorCode PEPGetEigenpair(PEP pep,PetscInt i,PetscScalar *eigr,PetscScalar *eigi,Vec Vr,Vec Vi)
{
  PetscInt       k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(pep,i,2);
  if (Vr) { PetscValidHeaderSpecific(Vr,VEC_CLASSID,5); PetscCheckSameComm(pep,1,Vr,5); }
  if (Vi) { PetscValidHeaderSpecific(Vi,VEC_CLASSID,6); PetscCheckSameComm(pep,1,Vi,6); }
  PEPCheckSolved(pep,1);
  if (i<0 || i>=pep->nconv) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range");

  ierr = PEPComputeVectors(pep);CHKERRQ(ierr);
  if (!pep->perm) k = i;
  else k = pep->perm[i];

  /* eigenvalue */
#if defined(PETSC_USE_COMPLEX)
  if (eigr) *eigr = pep->eigr[k];
  if (eigi) *eigi = 0;
#else
  if (eigr) *eigr = pep->eigr[k];
  if (eigi) *eigi = pep->eigi[k];
#endif

  /* eigenvector */
#if defined(PETSC_USE_COMPLEX)
  if (Vr) { ierr = BVCopyVec(pep->V,k,Vr);CHKERRQ(ierr); }
  if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
#else
  if (pep->eigi[k]>0) { /* first value of conjugate pair */
    if (Vr) { ierr = BVCopyVec(pep->V,k,Vr);CHKERRQ(ierr); }
    if (Vi) { ierr = BVCopyVec(pep->V,k+1,Vi);CHKERRQ(ierr); }
  } else if (pep->eigi[k]<0) { /* second value of conjugate pair */
    if (Vr) { ierr = BVCopyVec(pep->V,k-1,Vr);CHKERRQ(ierr); }
    if (Vi) {
      ierr = BVCopyVec(pep->V,k,Vi);CHKERRQ(ierr);
      ierr = VecScale(Vi,-1.0);CHKERRQ(ierr);
    }
  } else { /* real eigenvalue */
    if (Vr) { ierr = BVCopyVec(pep->V,k,Vr);CHKERRQ(ierr); }
    if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetErrorEstimate"
/*@
   PEPGetErrorEstimate - Returns the error estimate associated to the i-th
   computed eigenpair.

   Not Collective

   Input Parameter:
+  pep - polynomial eigensolver context
-  i   - index of eigenpair

   Output Parameter:
.  errest - the error estimate

   Notes:
   This is the error estimate used internally by the eigensolver. The actual
   error bound can be computed with PEPComputeRelativeError(). See also the users
   manual for details.

   Level: advanced

.seealso: PEPComputeRelativeError()
@*/
PetscErrorCode PEPGetErrorEstimate(PEP pep,PetscInt i,PetscReal *errest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(errest,3);
  PEPCheckSolved(pep,1);
  if (i<0 || i>=pep->nconv) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range");
  if (pep->perm) i = pep->perm[i];
  if (errest) *errest = pep->errest[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPComputeResidualNorm_Private"
/*
   PEPComputeResidualNorm_Private - Computes the norm of the residual vector
   associated with an eigenpair.
*/
PetscErrorCode PEPComputeResidualNorm_Private(PEP pep,PetscScalar kr,PetscScalar ki,Vec xr,Vec xi,PetscReal *norm)
{
  PetscErrorCode ierr;
  Vec            u,w;
  Mat            *A=pep->A;
  PetscInt       i,nmat=pep->nmat;
  PetscScalar    t[20],*vals=t,*ivals=NULL;
#if !defined(PETSC_USE_COMPLEX)
  Vec            ui,wi;
  PetscReal      ni;
  PetscBool      imag;
  PetscScalar    it[20];
#endif

  PetscFunctionBegin;
  ierr = BVGetVec(pep->V,&u);CHKERRQ(ierr);
  ierr = BVGetVec(pep->V,&w);CHKERRQ(ierr);
  ierr = VecZeroEntries(u);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ivals = it; 
#endif
  if (nmat>20) {
    ierr = PetscMalloc(nmat*sizeof(PetscScalar),&vals);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(nmat*sizeof(PetscScalar),&ivals);CHKERRQ(ierr);
#endif
  }
  ierr = PEPEvaluateBasis(pep,kr,ki,vals,ivals);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (ki == 0 || PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON))
    imag = PETSC_FALSE;
  else {
    imag = PETSC_TRUE;
    ierr = VecDuplicate(u,&ui);CHKERRQ(ierr);
    ierr = VecDuplicate(u,&wi);CHKERRQ(ierr);
    ierr = VecZeroEntries(ui);CHKERRQ(ierr);
  }
#endif
  for (i=0;i<nmat;i++) {
    if (vals[i]!=0.0) {
      ierr = MatMult(A[i],xr,w);CHKERRQ(ierr);
      ierr = VecAXPY(u,vals[i],w);CHKERRQ(ierr);
    }
#if !defined(PETSC_USE_COMPLEX)
    if (imag) {
      if (ivals[i]!=0 || vals[i]!=0) {
        ierr = MatMult(A[i],xi,wi);CHKERRQ(ierr);
        if (vals[i]==0) {
          ierr = MatMult(A[i],xr,w);CHKERRQ(ierr);
        }
      }
      if (ivals[i]!=0){
        ierr = VecAXPY(u,-ivals[i],wi);CHKERRQ(ierr);
        ierr = VecAXPY(ui,ivals[i],w);CHKERRQ(ierr);
      }
      if (vals[i]!=0) {
        ierr = VecAXPY(ui,vals[i],wi);CHKERRQ(ierr);
      }
    }
#endif
  }
  ierr = VecNorm(u,NORM_2,norm);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (imag) {
    ierr = VecNorm(ui,NORM_2,&ni);CHKERRQ(ierr);
    *norm = SlepcAbsEigenvalue(*norm,ni);
  }
#endif
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  if (nmat>20) {
    ierr = PetscFree(vals);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscFree(ivals);CHKERRQ(ierr);
#endif
  }
#if !defined(PETSC_USE_COMPLEX)
  if (imag) {
    ierr = VecDestroy(&wi);CHKERRQ(ierr);
    ierr = VecDestroy(&ui);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPComputeResidualNorm"
/*@
   PEPComputeResidualNorm - Computes the norm of the residual vector associated
   with the i-th computed eigenpair.

   Collective on PEP

   Input Parameter:
+  pep - the polynomial eigensolver context
-  i   - the solution index

   Output Parameter:
.  norm - the residual norm, computed as ||P(l)x||_2 where l is the
   eigenvalue and x is the eigenvector.

   Notes:
   The index i should be a value between 0 and nconv-1 (see PEPGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established
   with PEPSetWhichEigenpairs().

   Level: beginner

.seealso: PEPSolve(), PEPGetConverged(), PEPSetWhichEigenpairs()
@*/
PetscErrorCode PEPComputeResidualNorm(PEP pep,PetscInt i,PetscReal *norm)
{
  PetscErrorCode ierr;
  Vec            xr,xi;
  PetscScalar    kr,ki;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(pep,i,2);
  PetscValidPointer(norm,3);
  PEPCheckSolved(pep,1);
  ierr = BVGetVec(pep->V,&xr);CHKERRQ(ierr);
  ierr = BVGetVec(pep->V,&xi);CHKERRQ(ierr);
  ierr = PEPGetEigenpair(pep,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
  ierr = PEPComputeResidualNorm_Private(pep,kr,ki,xr,xi,norm);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPComputeRelativeError_Private"
/*
   PEPComputeRelativeError_Private - Computes the relative error bound
   associated with an eigenpair.
*/
PetscErrorCode PEPComputeRelativeError_Private(PEP pep,PetscScalar kr,PetscScalar ki,Vec xr,Vec xi,PetscReal *error)
{
  PetscErrorCode ierr;
  PetscReal      norm,er;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      ei;
#endif

  PetscFunctionBegin;
  ierr = PEPComputeResidualNorm_Private(pep,kr,ki,xr,xi,&norm);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (ki == 0) {
#endif
    ierr = VecNorm(xr,NORM_2,&er);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  } else {
    ierr = VecNorm(xr,NORM_2,&er);CHKERRQ(ierr);
    ierr = VecNorm(xi,NORM_2,&ei);CHKERRQ(ierr);
    er = SlepcAbsEigenvalue(er,ei);
  }
#endif
  ierr = (*pep->converged)(pep,kr,ki,norm/er,error,pep->convergedctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPComputeRelativeError"
/*@
   PEPComputeRelativeError - Computes the relative error bound associated
   with the i-th computed eigenpair.

   Collective on PEP

   Input Parameter:
+  pep - the polynomial eigensolver context
-  i   - the solution index

   Output Parameter:
.  error - the relative error bound, computed as ||P(l)x||_2/||lx||_2 where
   l is the eigenvalue and x is the eigenvector.

   Level: beginner

.seealso: PEPSolve(), PEPComputeResidualNorm(), PEPGetErrorEstimate()
@*/
PetscErrorCode PEPComputeRelativeError(PEP pep,PetscInt i,PetscReal *error)
{
  PetscErrorCode ierr;
  Vec            xr,xi;
  PetscScalar    kr,ki;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(pep,i,2);
  PetscValidPointer(error,3);
  PEPCheckSolved(pep,1);
  ierr = BVGetVec(pep->V,&xr);CHKERRQ(ierr);
  ierr = BVGetVec(pep->V,&xi);CHKERRQ(ierr);
  ierr = PEPGetEigenpair(pep,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
  ierr = PEPComputeRelativeError_Private(pep,kr,ki,xr,xi,error);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

