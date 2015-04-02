/*
      EPS routines related to the solution process.

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

#include <slepc-private/epsimpl.h>   /*I "slepceps.h" I*/
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "EPSComputeVectors"
PETSC_STATIC_INLINE PetscErrorCode EPSComputeVectors(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  EPSCheckSolved(eps,1);
  switch (eps->state) {
  case EPS_STATE_SOLVED:
    if (eps->ops->computevectors) {
      ierr = (*eps->ops->computevectors)(eps);CHKERRQ(ierr);
    }
    break;
  default:
    break;
  }
  eps->state = EPS_STATE_EIGENVECTORS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve"
/*@
   EPSSolve - Solves the eigensystem.

   Collective on EPS

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Options Database Keys:
+  -eps_view - print information about the solver used
.  -eps_view_mat0 binary - save the first matrix (A) to the default binary viewer
.  -eps_view_mat1 binary - save the second matrix (B) to the default binary viewer
-  -eps_plot_eigs - plot computed eigenvalues

   Level: beginner

.seealso: EPSCreate(), EPSSetUp(), EPSDestroy(), EPSSetTolerances()
@*/
PetscErrorCode EPSSolve(EPS eps)
{
  PetscErrorCode    ierr;
  PetscInt          i,nmat;
  PetscReal         re,im;
  PetscScalar       dot;
  PetscBool         flg,iscayley;
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscDraw         draw;
  PetscDrawSP       drawsp;
  STMatMode         matmode;
  Mat               A,B;
  Vec               w,x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscLogEventBegin(EPS_Solve,eps,0,0,0);CHKERRQ(ierr);

  /* call setup */
  ierr = EPSSetUp(eps);CHKERRQ(ierr);
  eps->nconv = 0;
  eps->its   = 0;
  for (i=0;i<eps->ncv;i++) {
    eps->eigr[i]   = 0.0;
    eps->eigi[i]   = 0.0;
    eps->errest[i] = 0.0;
  }
  ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->ncv);CHKERRQ(ierr);

  /* call solver */
  ierr = (*eps->ops->solve)(eps);CHKERRQ(ierr);
  eps->state = EPS_STATE_SOLVED;

  ierr = STGetMatMode(eps->st,&matmode);CHKERRQ(ierr);
  if (matmode == ST_MATMODE_INPLACE && eps->ispositive) {
    /* Purify eigenvectors before reverting operator */
    ierr = EPSComputeVectors(eps);CHKERRQ(ierr);
  }
  ierr = STPostSolve(eps->st);CHKERRQ(ierr);

  if (!eps->reason) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");

  /* Map eigenvalues back to the original problem, necessary in some
  * spectral transformations */
  if (eps->ops->backtransform) {
    ierr = (*eps->ops->backtransform)(eps);CHKERRQ(ierr);
  }

#if !defined(PETSC_USE_COMPLEX)
  /* reorder conjugate eigenvalues (positive imaginary first) */
  for (i=0; i<eps->nconv-1; i++) {
    if (eps->eigi[i] != 0) {
      if (eps->eigi[i] < 0) {
        eps->eigi[i] = -eps->eigi[i];
        eps->eigi[i+1] = -eps->eigi[i+1];
        /* the next correction only works with eigenvectors */
        ierr = EPSComputeVectors(eps);CHKERRQ(ierr);
        ierr = BVScaleColumn(eps->V,i+1,-1.0);CHKERRQ(ierr);
      }
      i++;
    }
  }
#endif

  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }

  /* In the case of Cayley transform, eigenvectors need to be B-normalized */
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STCAYLEY,&iscayley);CHKERRQ(ierr);
  if (iscayley && eps->isgeneralized && eps->ishermitian) {
    ierr = MatGetVecs(B,NULL,&w);CHKERRQ(ierr);
    ierr = EPSComputeVectors(eps);CHKERRQ(ierr);
    for (i=0;i<eps->nconv;i++) {
      ierr = BVGetColumn(eps->V,i,&x);CHKERRQ(ierr);
      ierr = MatMult(B,x,w);CHKERRQ(ierr);
      ierr = VecDot(w,x,&dot);CHKERRQ(ierr);
      ierr = VecScale(x,1.0/PetscSqrtScalar(dot));CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,i,&x);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&w);CHKERRQ(ierr);
  }

  /* sort eigenvalues according to eps->which parameter */
  ierr = SlepcSortEigenvalues(eps->sc,eps->nconv,eps->eigr,eps->eigi,eps->perm);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(EPS_Solve,eps,0,0,0);CHKERRQ(ierr);

  /* various viewers */
  ierr = MatViewFromOptions(A,((PetscObject)eps)->prefix,"-eps_view_mat0");CHKERRQ(ierr);
  if (nmat>1) { ierr = MatViewFromOptions(B,((PetscObject)eps)->prefix,"-eps_view_mat1");CHKERRQ(ierr); }

  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)eps),((PetscObject)eps)->prefix,"-eps_view",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = EPSView(eps,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)eps)->prefix,"-eps_plot_eigs",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Computed Eigenvalues",PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
    for (i=0;i<eps->nconv;i++) {
#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(eps->eigr[i]);
      im = PetscImaginaryPart(eps->eigr[i]);
#else
      re = eps->eigr[i];
      im = eps->eigi[i];
#endif
      ierr = PetscDrawSPAddPoint(drawsp,&re,&im);CHKERRQ(ierr);
    }
    ierr = PetscDrawSPDraw(drawsp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscDrawSPDestroy(&drawsp);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Remove deflation and initial subspaces */
  if (eps->nds) {
    ierr = BVSetNumConstraints(eps->V,0);CHKERRQ(ierr);
    eps->nds = 0;
  }
  eps->nini = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetIterationNumber"
/*@
   EPSGetIterationNumber - Gets the current iteration number. If the
   call to EPSSolve() is complete, then it returns the number of iterations
   carried out by the solution method.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  its - number of iterations

   Level: intermediate

   Note:
   During the i-th iteration this call returns i-1. If EPSSolve() is
   complete, then parameter "its" contains either the iteration number at
   which convergence was successfully reached, or failure was detected.
   Call EPSGetConvergedReason() to determine if the solver converged or
   failed and why.

.seealso: EPSGetConvergedReason(), EPSSetTolerances()
@*/
PetscErrorCode EPSGetIterationNumber(EPS eps,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(its,2);
  *its = eps->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetConverged"
/*@
   EPSGetConverged - Gets the number of converged eigenpairs.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  nconv - number of converged eigenpairs

   Note:
   This function should be called after EPSSolve() has finished.

   Level: beginner

.seealso: EPSSetDimensions(), EPSSolve()
@*/
PetscErrorCode EPSGetConverged(EPS eps,PetscInt *nconv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(nconv,2);
  EPSCheckSolved(eps,1);
  *nconv = eps->nconv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetConvergedReason"
/*@C
   EPSGetConvergedReason - Gets the reason why the EPSSolve() iteration was
   stopped.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged

   Possible values for reason:
+  EPS_CONVERGED_TOL - converged up to tolerance
.  EPS_DIVERGED_ITS - required more than its to reach convergence
-  EPS_DIVERGED_BREAKDOWN - generic breakdown in method

   Note:
   Can only be called after the call to EPSSolve() is complete.

   Level: intermediate

.seealso: EPSSetTolerances(), EPSSolve(), EPSConvergedReason
@*/
PetscErrorCode EPSGetConvergedReason(EPS eps,EPSConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(reason,2);
  EPSCheckSolved(eps,1);
  *reason = eps->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetInvariantSubspace"
/*@
   EPSGetInvariantSubspace - Gets an orthonormal basis of the computed invariant
   subspace.

   Not Collective, but vectors are shared by all processors that share the EPS

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  v - an array of vectors

   Notes:
   This function should be called after EPSSolve() has finished.

   The user should provide in v an array of nconv vectors, where nconv is
   the value returned by EPSGetConverged().

   The first k vectors returned in v span an invariant subspace associated
   with the first k computed eigenvalues (note that this is not true if the
   k-th eigenvalue is complex and matrix A is real; in this case the first
   k+1 vectors should be used). An invariant subspace X of A satisfies Ax
   in X for all x in X (a similar definition applies for generalized
   eigenproblems).

   Level: intermediate

.seealso: EPSGetEigenpair(), EPSGetConverged(), EPSSolve()
@*/
PetscErrorCode EPSGetInvariantSubspace(EPS eps,Vec *v)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(v,2);
  PetscValidHeaderSpecific(*v,VEC_CLASSID,2);
  EPSCheckSolved(eps,1);
  if (!eps->ishermitian && eps->state==EPS_STATE_EIGENVECTORS) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"EPSGetInvariantSubspace must be called before EPSGetEigenpair,EPSGetEigenvector,EPSComputeRelativeError or EPSComputeResidualNorm");
  for (i=0;i<eps->nconv;i++) {
    ierr = BVCopyVec(eps->V,i,v[i]);CHKERRQ(ierr);
    if (eps->balance!=EPS_BALANCE_NONE && eps->D) {
      ierr = VecPointwiseDivide(v[i],v[i],eps->D);CHKERRQ(ierr);
      ierr = VecNormalize(v[i],NULL);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetEigenpair"
/*@
   EPSGetEigenpair - Gets the i-th solution of the eigenproblem as computed by
   EPSSolve(). The solution consists in both the eigenvalue and the eigenvector.

   Logically Collective on EPS

   Input Parameters:
+  eps - eigensolver context
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

   The index i should be a value between 0 and nconv-1 (see EPSGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established
   with EPSSetWhichEigenpairs().

   The 2-norm of the eigenvector is one unless the problem is generalized
   Hermitian. In this case the eigenvector is normalized with respect to the
   norm defined by the B matrix.

   Level: beginner

.seealso: EPSGetEigenvalue(), EPSGetEigenvector(), EPSSolve(),
          EPSGetConverged(), EPSSetWhichEigenpairs(), EPSGetInvariantSubspace()
@*/
PetscErrorCode EPSGetEigenpair(EPS eps,PetscInt i,PetscScalar *eigr,PetscScalar *eigi,Vec Vr,Vec Vi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,i,2);
  EPSCheckSolved(eps,1);
  if (i<0 || i>=eps->nconv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range");
  ierr = EPSGetEigenvalue(eps,i,eigr,eigi);CHKERRQ(ierr);
  if (Vr || Vi) { ierr = EPSGetEigenvector(eps,i,Vr,Vi);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetEigenvalue"
/*@
   EPSGetEigenvalue - Gets the i-th eigenvalue as computed by EPSSolve().

   Not Collective

   Input Parameters:
+  eps - eigensolver context
-  i   - index of the solution

   Output Parameters:
+  eigr - real part of eigenvalue
-  eigi - imaginary part of eigenvalue

   Notes:
   If the eigenvalue is real, then eigi is set to zero. If PETSc is
   configured with complex scalars the eigenvalue is stored
   directly in eigr (eigi is set to zero).

   The index i should be a value between 0 and nconv-1 (see EPSGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established
   with EPSSetWhichEigenpairs().

   Level: beginner

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs(), EPSGetEigenpair()
@*/
PetscErrorCode EPSGetEigenvalue(EPS eps,PetscInt i,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscInt       k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  EPSCheckSolved(eps,1);
  if (i<0 || i>=eps->nconv) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range");
  if (!eps->perm) k = i;
  else k = eps->perm[i];
#if defined(PETSC_USE_COMPLEX)
  if (eigr) *eigr = eps->eigr[k];
  if (eigi) *eigi = 0;
#else
  if (eigr) *eigr = eps->eigr[k];
  if (eigi) *eigi = eps->eigi[k];
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetEigenvector"
/*@
   EPSGetEigenvector - Gets the i-th right eigenvector as computed by EPSSolve().

   Logically Collective on EPS

   Input Parameters:
+  eps - eigensolver context
-  i   - index of the solution

   Output Parameters:
+  Vr   - real part of eigenvector
-  Vi   - imaginary part of eigenvector

   Notes:
   If the corresponding eigenvalue is real, then Vi is set to zero. If PETSc is
   configured with complex scalars the eigenvector is stored
   directly in Vr (Vi is set to zero).

   The index i should be a value between 0 and nconv-1 (see EPSGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established
   with EPSSetWhichEigenpairs().

   The 2-norm of the eigenvector is one unless the problem is generalized
   Hermitian. In this case the eigenvector is normalized with respect to the
   norm defined by the B matrix.

   Level: beginner

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs(), EPSGetEigenpair()
@*/
PetscErrorCode EPSGetEigenvector(EPS eps,PetscInt i,Vec Vr,Vec Vi)
{
  PetscErrorCode ierr;
  PetscInt       k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,i,2);
  PetscValidHeaderSpecific(Vr,VEC_CLASSID,3);
  PetscCheckSameComm(eps,1,Vr,3);
  if (Vi) { PetscValidHeaderSpecific(Vi,VEC_CLASSID,4); PetscCheckSameComm(eps,1,Vi,4); }
  EPSCheckSolved(eps,1);
  if (i<0 || i>=eps->nconv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range");
  ierr = EPSComputeVectors(eps);CHKERRQ(ierr);
  if (!eps->perm) k = i;
  else k = eps->perm[i];
#if defined(PETSC_USE_COMPLEX)
  ierr = BVCopyVec(eps->V,k,Vr);CHKERRQ(ierr);
  if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
#else
  if (eps->eigi[k] > 0) { /* first value of conjugate pair */
    ierr = BVCopyVec(eps->V,k,Vr);CHKERRQ(ierr);
    if (Vi) {
      ierr = BVCopyVec(eps->V,k+1,Vi);CHKERRQ(ierr);
    }
  } else if (eps->eigi[k] < 0) { /* second value of conjugate pair */
    ierr = BVCopyVec(eps->V,k-1,Vr);CHKERRQ(ierr);
    if (Vi) {
      ierr = BVCopyVec(eps->V,k,Vi);CHKERRQ(ierr);
      ierr = VecScale(Vi,-1.0);CHKERRQ(ierr);
    }
  } else { /* real eigenvalue */
    ierr = BVCopyVec(eps->V,k,Vr);CHKERRQ(ierr);
    if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetErrorEstimate"
/*@
   EPSGetErrorEstimate - Returns the error estimate associated to the i-th
   computed eigenpair.

   Not Collective

   Input Parameter:
+  eps - eigensolver context
-  i   - index of eigenpair

   Output Parameter:
.  errest - the error estimate

   Notes:
   This is the error estimate used internally by the eigensolver. The actual
   error bound can be computed with EPSComputeRelativeError(). See also the users
   manual for details.

   Level: advanced

.seealso: EPSComputeRelativeError()
@*/
PetscErrorCode EPSGetErrorEstimate(EPS eps,PetscInt i,PetscReal *errest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(errest,3);
  EPSCheckSolved(eps,1);
  if (i<0 || i>=eps->nconv) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range");
  if (eps->perm) i = eps->perm[i];
  if (errest) *errest = eps->errest[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSComputeResidualNorm_Private"
/*
   EPSComputeResidualNorm_Private - Computes the norm of the residual vector
   associated with an eigenpair.
*/
PetscErrorCode EPSComputeResidualNorm_Private(EPS eps,PetscScalar kr,PetscScalar ki,Vec xr,Vec xi,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscInt       nmat;
  Vec            u,w;
  Mat            A,B;
#if !defined(PETSC_USE_COMPLEX)
  Vec            v;
  PetscReal      ni,nr;
#endif

  PetscFunctionBegin;
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }
  ierr = BVGetVec(eps->V,&u);CHKERRQ(ierr);
  ierr = BVGetVec(eps->V,&w);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
  if (ki == 0 || PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    ierr = MatMult(A,xr,u);CHKERRQ(ierr);                             /* u=A*x */
    if (PetscAbsScalar(kr) > PETSC_MACHINE_EPSILON) {
      if (eps->isgeneralized) { ierr = MatMult(B,xr,w);CHKERRQ(ierr); }
      else { ierr = VecCopy(xr,w);CHKERRQ(ierr); }                    /* w=B*x */
      ierr = VecAXPY(u,-kr,w);CHKERRQ(ierr);                          /* u=A*x-k*B*x */
    }
    ierr = VecNorm(u,NORM_2,norm);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  } else {
    ierr = BVGetVec(eps->V,&v);CHKERRQ(ierr);
    ierr = MatMult(A,xr,u);CHKERRQ(ierr);                             /* u=A*xr */
    if (SlepcAbsEigenvalue(kr,ki) > PETSC_MACHINE_EPSILON) {
      if (eps->isgeneralized) { ierr = MatMult(B,xr,v);CHKERRQ(ierr); }
      else { ierr = VecCopy(xr,v);CHKERRQ(ierr); }                    /* v=B*xr */
      ierr = VecAXPY(u,-kr,v);CHKERRQ(ierr);                          /* u=A*xr-kr*B*xr */
      if (eps->isgeneralized) { ierr = MatMult(B,xi,w);CHKERRQ(ierr); }
      else { ierr = VecCopy(xi,w);CHKERRQ(ierr); }                    /* w=B*xi */
      ierr = VecAXPY(u,ki,w);CHKERRQ(ierr);                           /* u=A*xr-kr*B*xr+ki*B*xi */
    }
    ierr = VecNorm(u,NORM_2,&nr);CHKERRQ(ierr);
    ierr = MatMult(A,xi,u);CHKERRQ(ierr);                             /* u=A*xi */
    if (SlepcAbsEigenvalue(kr,ki) > PETSC_MACHINE_EPSILON) {
      ierr = VecAXPY(u,-kr,w);CHKERRQ(ierr);                          /* u=A*xi-kr*B*xi */
      ierr = VecAXPY(u,-ki,v);CHKERRQ(ierr);                          /* u=A*xi-kr*B*xi-ki*B*xr */
    }
    ierr = VecNorm(u,NORM_2,&ni);CHKERRQ(ierr);
    *norm = SlepcAbsEigenvalue(nr,ni);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  }
#endif

  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSComputeResidualNorm"
/*@
   EPSComputeResidualNorm - Computes the norm of the residual vector associated
   with the i-th computed eigenpair.

   Collective on EPS

   Input Parameter:
+  eps - the eigensolver context
-  i   - the solution index

   Output Parameter:
.  norm - the residual norm, computed as ||Ax-kBx||_2 where k is the
   eigenvalue and x is the eigenvector.
   If k=0 then the residual norm is computed as ||Ax||_2.

   Notes:
   The index i should be a value between 0 and nconv-1 (see EPSGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established
   with EPSSetWhichEigenpairs().

   Level: beginner

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs()
@*/
PetscErrorCode EPSComputeResidualNorm(EPS eps,PetscInt i,PetscReal *norm)
{
  PetscErrorCode ierr;
  Vec            xr,xi;
  PetscScalar    kr,ki;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,i,2);
  PetscValidPointer(norm,3);
  EPSCheckSolved(eps,1);
  ierr = BVGetVec(eps->V,&xr);CHKERRQ(ierr);
  ierr = BVGetVec(eps->V,&xi);CHKERRQ(ierr);
  ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
  ierr = EPSComputeResidualNorm_Private(eps,kr,ki,xr,xi,norm);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSComputeRelativeError_Private"
/*
   EPSComputeRelativeError_Private - Computes the relative error bound
   associated with an eigenpair.
*/
PetscErrorCode EPSComputeRelativeError_Private(EPS eps,PetscScalar kr,PetscScalar ki,Vec xr,Vec xi,PetscReal *error)
{
  PetscErrorCode ierr;
  PetscReal      norm,er;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      ei;
#endif

  PetscFunctionBegin;
  ierr = EPSComputeResidualNorm_Private(eps,kr,ki,xr,xi,&norm);CHKERRQ(ierr);

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
  ierr = (*eps->converged)(eps,kr,ki,norm/er,error,eps->convergedctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSComputeRelativeError"
/*@
   EPSComputeRelativeError - Computes the relative error bound associated
   with the i-th computed eigenpair.

   Collective on EPS

   Input Parameter:
+  eps - the eigensolver context
-  i   - the solution index

   Output Parameter:
.  error - the relative error bound, computed as ||Ax-kBx||_2/||kx||_2 where
   k is the eigenvalue and x is the eigenvector.
   If k=0 the relative error is computed as ||Ax||_2/||x||_2.

   Level: beginner

.seealso: EPSSolve(), EPSComputeResidualNorm(), EPSGetErrorEstimate()
@*/
PetscErrorCode EPSComputeRelativeError(EPS eps,PetscInt i,PetscReal *error)
{
  PetscErrorCode ierr;
  Vec            xr,xi;
  PetscScalar    kr,ki;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,i,2);
  PetscValidPointer(error,3);
  EPSCheckSolved(eps,1);
  ierr = BVGetVec(eps->V,&xr);CHKERRQ(ierr);
  ierr = BVGetVec(eps->V,&xi);CHKERRQ(ierr);
  ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
  ierr = EPSComputeRelativeError_Private(eps,kr,ki,xr,xi,error);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetStartVector"
/*
   EPSGetStartVector - Generate a suitable vector to be used as the starting vector
   for the recurrence that builds the right subspace.

   Collective on EPS and Vec

   Input Parameters:
+  eps - the eigensolver context
-  i   - iteration number

   Output Parameters:
.  breakdown - flag indicating that a breakdown has occurred

   Notes:
   The start vector is computed from another vector: for the first step (i=0),
   the first initial vector is used (see EPSSetInitialSpace()); otherwise a random
   vector is created. Then this vector is forced to be in the range of OP (only
   for generalized definite problems) and orthonormalized with respect to all
   V-vectors up to i-1. The resulting vector is placed in V[i].

   The flag breakdown is set to true if either i=0 and the vector belongs to the
   deflation space, or i>0 and the vector is linearly dependent with respect
   to the V-vectors.
*/
PetscErrorCode EPSGetStartVector(EPS eps,PetscInt i,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscReal      norm;
  PetscBool      lindep;
  Vec            w,z;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,i,2);

  /* For the first step, use the first initial vector, otherwise a random one */
  if (i>0 || eps->nini==0) {
    ierr = BVSetRandomColumn(eps->V,i,eps->rand);CHKERRQ(ierr);
  }
  ierr = BVGetVec(eps->V,&w);CHKERRQ(ierr);
  ierr = BVCopyVec(eps->V,i,w);CHKERRQ(ierr);

  /* Force the vector to be in the range of OP for definite generalized problems */
  ierr = BVGetColumn(eps->V,i,&z);CHKERRQ(ierr);
  if (eps->ispositive || (eps->isgeneralized && eps->ishermitian)) {
    ierr = STApply(eps->st,w,z);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(w,z);CHKERRQ(ierr);
  }
  ierr = BVRestoreColumn(eps->V,i,&z);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);

  /* Orthonormalize the vector with respect to previous vectors */
  ierr = BVOrthogonalizeColumn(eps->V,i,NULL,&norm,&lindep);CHKERRQ(ierr);
  if (breakdown) *breakdown = lindep;
  else if (lindep || norm == 0.0) {
    if (i==0) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Initial vector is zero or belongs to the deflation space");
    else SETERRQ(PetscObjectComm((PetscObject)eps),1,"Unable to generate more start vectors");
  }
  ierr = BVScaleColumn(eps->V,i,1.0/norm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

