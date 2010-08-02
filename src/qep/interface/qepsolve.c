/*
      QEP routines related to the solution process.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/qepimpl.h"   /*I "slepcqep.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "QEPSolve"
/*@
   QEPSolve - Solves the quadratic eigensystem.

   Collective on QEP

   Input Parameter:
.  qep - eigensolver context obtained from QEPCreate()

   Options Database:
+   -qep_view - print information about the solver used
.   -qep_view_binary - save the matrices to the default binary file
-   -qep_plot_eigs - plot computed eigenvalues

   Level: beginner

.seealso: QEPCreate(), QEPSetUp(), QEPDestroy(), QEPSetTolerances() 
@*/
PetscErrorCode QEPSolve(QEP qep) 
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      re,im;
  PetscTruth     flg;
  PetscViewer    viewer;
  PetscDraw      draw;
  PetscDrawSP    drawsp;
  char           filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);

  flg = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(((PetscObject)qep)->prefix,"-qep_view_binary",&flg,PETSC_NULL);CHKERRQ(ierr); 
  if (flg) {
    ierr = MatView(qep->M,PETSC_VIEWER_BINARY_(((PetscObject)qep)->comm));CHKERRQ(ierr);
    ierr = MatView(qep->C,PETSC_VIEWER_BINARY_(((PetscObject)qep)->comm));CHKERRQ(ierr);
    ierr = MatView(qep->K,PETSC_VIEWER_BINARY_(((PetscObject)qep)->comm));CHKERRQ(ierr);
  }

  /* reset the convergence flag from the previous solves */
  qep->reason = QEP_CONVERGED_ITERATING;

  if (!qep->setupcalled){ ierr = QEPSetUp(qep);CHKERRQ(ierr); }
  ierr = IPResetOperationCounters(qep->ip);CHKERRQ(ierr);
  qep->nconv = 0;
  qep->its = 0;
  qep->matvecs = 0;
  qep->linits = 0;
  for (i=0;i<qep->ncv;i++) qep->eigr[i]=qep->eigi[i]=qep->errest[i]=0.0;
  QEPMonitor(qep,qep->its,qep->nconv,qep->eigr,qep->eigi,qep->errest,qep->ncv);

  ierr = PetscLogEventBegin(QEP_Solve,qep,0,0,0);CHKERRQ(ierr);
  ierr = (*qep->ops->solve)(qep);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(QEP_Solve,qep,0,0,0);CHKERRQ(ierr);

  if (!qep->reason) {
    SETERRQ(1,"Internal error, solver returned without setting converged reason");
  }

#ifndef PETSC_USE_COMPLEX
  /* reorder conjugate eigenvalues (positive imaginary first) */
  for (i=0;i<qep->nconv-1;i++) {
    if (qep->eigi[i] != 0) {
      if (qep->eigi[i] < 0) {
        qep->eigi[i] = -qep->eigi[i];
        qep->eigi[i+1] = -qep->eigi[i+1];
        ierr = VecScale(qep->V[i+1],-1.0);CHKERRQ(ierr);
      }
      i++;
    }
  }
#endif

  /* sort eigenvalues according to qep->which parameter */
  ierr = PetscFree(qep->perm);CHKERRQ(ierr);
  if (qep->nconv > 0) {
    ierr = PetscMalloc(sizeof(PetscInt)*qep->nconv,&qep->perm);CHKERRQ(ierr);
    ierr = QEPSortEigenvalues(qep,qep->nconv,qep->eigr,qep->eigi,qep->perm);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetString(((PetscObject)qep)->prefix,"-qep_view",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = PetscViewerASCIIOpen(((PetscObject)qep)->comm,filename,&viewer);CHKERRQ(ierr);
    ierr = QEPView(qep,viewer);CHKERRQ(ierr); 
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }

  flg = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(((PetscObject)qep)->prefix,"-qep_plot_eigs",&flg,PETSC_NULL);CHKERRQ(ierr); 
  if (flg) { 
    ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Computed Eigenvalues",
                             PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
    for(i=0;i<qep->nconv;i++) {
#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(qep->eigr[i]);
      im = PetscImaginaryPart(qep->eigi[i]);
#else
      re = qep->eigr[i];
      im = qep->eigi[i];
#endif
      ierr = PetscDrawSPAddPoint(drawsp,&re,&im);CHKERRQ(ierr);
    }
    ierr = PetscDrawSPDraw(drawsp);CHKERRQ(ierr);
    ierr = PetscDrawSPDestroy(drawsp);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }

  /* Remove the initial subspace */
  qep->nini = 0;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetIterationNumber"
/*@
   QEPGetIterationNumber - Gets the current iteration number. If the 
   call to QEPSolve() is complete, then it returns the number of iterations 
   carried out by the solution method.
 
   Not Collective

   Input Parameter:
.  qep - the quadratic eigensolver context

   Output Parameter:
.  its - number of iterations

   Level: intermediate

   Note:
   During the i-th iteration this call returns i-1. If QEPSolve() is 
   complete, then parameter "its" contains either the iteration number at
   which convergence was successfully reached, or failure was detected.  
   Call QEPGetConvergedReason() to determine if the solver converged or 
   failed and why.

.seealso: QEPGetConvergedReason(), QEPSetTolerances()
@*/
PetscErrorCode QEPGetIterationNumber(QEP qep,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidIntPointer(its,2);
  *its = qep->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetConverged"
/*@
   QEPGetConverged - Gets the number of converged eigenpairs.

   Not Collective

   Input Parameter:
.  qep - the quadratic eigensolver context
  
   Output Parameter:
.  nconv - number of converged eigenpairs 

   Note:
   This function should be called after QEPSolve() has finished.

   Level: beginner

.seealso: QEPSetDimensions(), QEPSolve()
@*/
PetscErrorCode QEPGetConverged(QEP qep,PetscInt *nconv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidIntPointer(nconv,2);
  *nconv = qep->nconv;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "QEPGetConvergedReason"
/*@C
   QEPGetConvergedReason - Gets the reason why the QEPSolve() iteration was 
   stopped.

   Not Collective

   Input Parameter:
.  qep - the quadratic eigensolver context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged

   Possible values for reason:
+  QEP_CONVERGED_TOL - converged up to tolerance
.  QEP_DIVERGED_ITS - required more than its to reach convergence
-  QEP_DIVERGED_BREAKDOWN - generic breakdown in method

   Note:
   Can only be called after the call to QEPSolve() is complete.

   Level: intermediate

.seealso: QEPSetTolerances(), QEPSolve(), QEPConvergedReason
@*/
PetscErrorCode QEPGetConvergedReason(QEP qep,QEPConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidIntPointer(reason,2);
  *reason = qep->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetEigenpair" 
/*@
   QEPGetEigenpair - Gets the i-th solution of the eigenproblem as computed by 
   QEPSolve(). The solution consists in both the eigenvalue and the eigenvector.

   Not Collective, but vectors are shared by all processors that share the QEP

   Input Parameters:
+  qep - quadratic eigensolver context 
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

   The index i should be a value between 0 and nconv-1 (see QEPGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established 
   with QEPSetWhichEigenpairs().

   Level: beginner

.seealso: QEPSolve(), QEPGetConverged(), QEPSetWhichEigenpairs()
@*/
PetscErrorCode QEPGetEigenpair(QEP qep, PetscInt i, PetscScalar *eigr, PetscScalar *eigi, Vec Vr, Vec Vi)
{
  PetscInt       k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (!qep->eigr || !qep->eigi || !qep->V) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "QEPSolve must be called first"); 
  }
  if (i<0 || i>=qep->nconv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }

  if (!qep->perm) k = i;
  else k = qep->perm[i];

  /* eigenvalue */
#ifdef PETSC_USE_COMPLEX
  if (eigr) *eigr = qep->eigr[k];
  if (eigi) *eigi = 0;
#else
  if (eigr) *eigr = qep->eigr[k];
  if (eigi) *eigi = qep->eigi[k];
#endif
  
  /* eigenvector */
#ifdef PETSC_USE_COMPLEX
  if (Vr) { ierr = VecCopy(qep->V[k],Vr);CHKERRQ(ierr); }
  if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
#else
  if (qep->eigi[k]>0) { /* first value of conjugate pair */
    if (Vr) { ierr = VecCopy(qep->V[k],Vr);CHKERRQ(ierr); }
    if (Vi) { ierr = VecCopy(qep->V[k+1],Vi);CHKERRQ(ierr); }
  } else if (qep->eigi[k]<0) { /* second value of conjugate pair */
    if (Vr) { ierr = VecCopy(qep->V[k-1],Vr);CHKERRQ(ierr); }
    if (Vi) { 
      ierr = VecCopy(qep->V[k],Vi);CHKERRQ(ierr); 
      ierr = VecScale(Vi,-1.0);CHKERRQ(ierr); 
    }
  } else { /* real eigenvalue */
    if (Vr) { ierr = VecCopy(qep->V[k],Vr);CHKERRQ(ierr); }
    if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
  }
#endif
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetErrorEstimate" 
/*@
   QEPGetErrorEstimate - Returns the error estimate associated to the i-th 
   computed eigenpair.

   Not Collective

   Input Parameter:
+  qep - quadratic eigensolver context 
-  i   - index of eigenpair

   Output Parameter:
.  errest - the error estimate

   Notes:
   This is the error estimate used internally by the eigensolver. The actual
   error bound can be computed with QEPComputeRelativeError(). See also the users
   manual for details.

   Level: advanced

.seealso: QEPComputeRelativeError()
@*/
PetscErrorCode QEPGetErrorEstimate(QEP qep, PetscInt i, PetscReal *errest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (!qep->eigr || !qep->eigi) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "QEPSolve must be called first"); 
  }
  if (i<0 || i>=qep->nconv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }
  if (qep->perm) i = qep->perm[i];  
  if (errest) *errest = qep->errest[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPComputeResidualNorm_Private"
/*
   QEPComputeResidualNorm_Private - Computes the norm of the residual vector 
   associated with an eigenpair.
*/
PetscErrorCode QEPComputeResidualNorm_Private(QEP qep, PetscScalar kr, PetscScalar ki, Vec xr, Vec xi, PetscReal *norm)
{
  PetscErrorCode ierr;
  Vec            u,w;
  Mat            M=qep->M,C=qep->C,K=qep->K;
#ifndef PETSC_USE_COMPLEX
  Vec            v,y,z;
  PetscReal      ni,nr;
  PetscScalar    a1,a2;
#endif
  
  PetscFunctionBegin;
  ierr = VecDuplicate(qep->V[0],&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&w);CHKERRQ(ierr);
  
#ifndef PETSC_USE_COMPLEX
  if (ki == 0 || 
    PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    ierr = MatMult(K,xr,u);CHKERRQ(ierr);                 /* u=K*x */
    if (PetscAbsScalar(kr) > PETSC_MACHINE_EPSILON) {
      ierr = MatMult(C,xr,w);CHKERRQ(ierr);               /* w=C*x */
      ierr = VecAXPY(u,kr,w);CHKERRQ(ierr);               /* u=l*C*x+K*x */
      ierr = MatMult(M,xr,w);CHKERRQ(ierr);               /* w=M*x */
      ierr = VecAXPY(u,kr*kr,w);CHKERRQ(ierr);            /* u=l^2*M*x+l*C*x+K*x */
    }
    ierr = VecNorm(u,NORM_2,norm);CHKERRQ(ierr);  
#ifndef PETSC_USE_COMPLEX
  } else {
    ierr = VecDuplicate(u,&v);CHKERRQ(ierr);
    ierr = VecDuplicate(u,&y);CHKERRQ(ierr);
    ierr = VecDuplicate(u,&z);CHKERRQ(ierr);
    a1 = kr*kr-ki*ki;
    a2 = 2.0*kr*ki;
    ierr = MatMult(K,xr,u);CHKERRQ(ierr);           /* u=K*xr */
    if (SlepcAbsEigenvalue(kr,ki) > PETSC_MACHINE_EPSILON) {
      ierr = MatMult(C,xr,v);CHKERRQ(ierr);         /* v=C*xr */
      ierr = MatMult(C,xi,w);CHKERRQ(ierr);         /* w=C*xi */
      ierr = MatMult(M,xr,y);CHKERRQ(ierr);         /* y=M*xr */
      ierr = MatMult(M,xi,z);CHKERRQ(ierr);         /* z=M*xi */
      ierr = VecAXPY(u,kr,v);CHKERRQ(ierr);         /* u=kr*C*xr+K*xr */
      ierr = VecAXPY(u,-ki,w);CHKERRQ(ierr);        /* u=kr*C*xr-ki*C*xi+K*xr */
      ierr = VecAXPY(u,a1,y);CHKERRQ(ierr);         /* u=a1*M*xr+kr*C*xr-ki*C*xi+K*xr */
      ierr = VecAXPY(u,-a2,z);CHKERRQ(ierr);        /* u=a1*M*xr-a2*M*xi+kr*C*xr-ki*C*xi+K*xr */
    }
    ierr = VecNorm(u,NORM_2,&nr);CHKERRQ(ierr);
    ierr = MatMult(K,xi,u);CHKERRQ(ierr);         /* u=K*xi */
    if (SlepcAbsEigenvalue(kr,ki) > PETSC_MACHINE_EPSILON) {
      ierr = VecAXPY(u,kr,w);CHKERRQ(ierr);         /* u=kr*C*xi+K*xi */
      ierr = VecAXPY(u,ki,v);CHKERRQ(ierr);         /* u=kr*C*xi+ki*C*xi+K*xi */
      ierr = VecAXPY(u,a1,z);CHKERRQ(ierr);         /* u=a1*M*xi+kr*C*xi+ki*C*xi+K*xi */
      ierr = VecAXPY(u,a2,y);CHKERRQ(ierr);         /* u=a1*M*xi+a2*M*ki+kr*C*xi+ki*C*xi+K*xi */
    }
    ierr = VecNorm(u,NORM_2,&ni);CHKERRQ(ierr);
    *norm = SlepcAbsEigenvalue(nr,ni);
    ierr = VecDestroy(v);CHKERRQ(ierr);
    ierr = VecDestroy(y);CHKERRQ(ierr);
    ierr = VecDestroy(z);CHKERRQ(ierr);
  }
#endif

  ierr = VecDestroy(w);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPComputeResidualNorm"
/*@
   QEPComputeResidualNorm - Computes the norm of the residual vector associated with 
   the i-th computed eigenpair.

   Collective on QEP

   Input Parameter:
.  qep - the quadratic eigensolver context
.  i   - the solution index

   Output Parameter:
.  norm - the residual norm, computed as ||(l^2*M+l*C+K)x||_2 where l is the 
   eigenvalue and x is the eigenvector. 
   If l=0 then the residual norm is computed as ||Kx||_2.

   Notes:
   The index i should be a value between 0 and nconv-1 (see QEPGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established 
   with QEPSetWhichEigenpairs().

   Level: beginner

.seealso: QEPSolve(), QEPGetConverged(), QEPSetWhichEigenpairs()
@*/
PetscErrorCode QEPComputeResidualNorm(QEP qep, PetscInt i, PetscReal *norm)
{
  PetscErrorCode ierr;
  Vec            xr,xi;
  PetscScalar    kr,ki;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidPointer(norm,3);
  ierr = VecDuplicate(qep->V[0],&xr);CHKERRQ(ierr);
  ierr = VecDuplicate(qep->V[0],&xi);CHKERRQ(ierr);
  ierr = QEPGetEigenpair(qep,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
  ierr = QEPComputeResidualNorm_Private(qep,kr,ki,xr,xi,norm);CHKERRQ(ierr);
  ierr = VecDestroy(xr);CHKERRQ(ierr);
  ierr = VecDestroy(xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPComputeRelativeError_Private"
/*
   QEPComputeRelativeError_Private - Computes the relative error bound 
   associated with an eigenpair.
*/
PetscErrorCode QEPComputeRelativeError_Private(QEP qep, PetscScalar kr, PetscScalar ki, Vec xr, Vec xi, PetscReal *error)
{
  PetscErrorCode ierr;
  PetscReal      norm, er;
#ifndef PETSC_USE_COMPLEX
  PetscReal      ei;
#endif
  
  PetscFunctionBegin;
  ierr = QEPComputeResidualNorm_Private(qep,kr,ki,xr,xi,&norm);CHKERRQ(ierr);

#ifndef PETSC_USE_COMPLEX
  if (ki == 0 || 
    PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    ierr = VecNorm(xr,NORM_2,&er);CHKERRQ(ierr);
    if (PetscAbsScalar(kr) > norm) {
      *error = norm/(PetscAbsScalar(kr)*er);
    } else {
      *error = norm/er;
    }
#ifndef PETSC_USE_COMPLEX
  } else {
    ierr = VecNorm(xr,NORM_2,&er);CHKERRQ(ierr);  
    ierr = VecNorm(xi,NORM_2,&ei);CHKERRQ(ierr);  
    if (SlepcAbsEigenvalue(kr,ki) > norm) {
      *error = norm/(SlepcAbsEigenvalue(kr,ki)*SlepcAbsEigenvalue(er,ei));
    } else {
      *error = norm/SlepcAbsEigenvalue(er,ei);
    }
  }
#endif    
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPComputeRelativeError"
/*@
   QEPComputeRelativeError - Computes the relative error bound associated 
   with the i-th computed eigenpair.

   Collective on QEP

   Input Parameter:
.  qep - the quadratic eigensolver context
.  i   - the solution index

   Output Parameter:
.  error - the relative error bound, computed as ||(l^2*M+l*C+K)x||_2/||lx||_2 where 
   l is the eigenvalue and x is the eigenvector. 
   If l=0 the relative error is computed as ||Kx||_2/||x||_2.

   Level: beginner

.seealso: QEPSolve(), QEPComputeResidualNorm(), QEPGetErrorEstimate()
@*/
PetscErrorCode QEPComputeRelativeError(QEP qep, PetscInt i, PetscReal *error)
{
  PetscErrorCode ierr;
  Vec            xr,xi;  
  PetscScalar    kr,ki;  
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);  
  PetscValidPointer(error,3);
  ierr = VecDuplicate(qep->V[0],&xr);CHKERRQ(ierr);
  ierr = VecDuplicate(qep->V[0],&xi);CHKERRQ(ierr);
  ierr = QEPGetEigenpair(qep,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
  ierr = QEPComputeRelativeError_Private(qep,kr,ki,xr,xi,error);CHKERRQ(ierr);  
  ierr = VecDestroy(xr);CHKERRQ(ierr);
  ierr = VecDestroy(xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSortEigenvalues"
/*@
   QEPSortEigenvalues - Sorts a list of eigenvalues according to the criterion 
   specified via QEPSetWhichEigenpairs().

   Not Collective

   Input Parameters:
+  qep   - the quadratic eigensolver context
.  n     - number of eigenvalues in the list
.  eigr  - pointer to the array containing the eigenvalues
-  eigi  - imaginary part of the eigenvalues (only when using real numbers)

   Output Parameter:
.  perm  - resulting permutation

   Note:
   The result is a list of indices in the original eigenvalue array 
   corresponding to the first nev eigenvalues sorted in the specified
   criterion.

   Level: developer

.seealso: QEPSortEigenvaluesReal(), QEPSetWhichEigenpairs()
@*/
PetscErrorCode QEPSortEigenvalues(QEP qep,PetscInt n,PetscScalar *eigr,PetscScalar *eigi,PetscInt *perm)
{
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,result,tmp;

  PetscFunctionBegin;
  for (i=0; i<n; i++) { perm[i] = i; }
  /* insertion sort */
  for (i=n-1; i>=0; i--) {
    re = eigr[perm[i]];
    im = eigi[perm[i]];
    j = i + 1;
#ifndef PETSC_USE_COMPLEX
    if (im != 0) {
      /* complex eigenvalue */
      i--;
      im = eigi[perm[i]];
    }
#endif
    while (j<n) {
      ierr = QEPCompareEigenvalues(qep,re,im,eigr[perm[j]],eigi[perm[j]],&result);CHKERRQ(ierr);
      if (result >= 0) break;
#ifndef PETSC_USE_COMPLEX
      /* keep together every complex conjugated eigenpair */
      if (im == 0) { 
        if (eigi[perm[j]] == 0) {
#endif
          tmp = perm[j-1]; perm[j-1] = perm[j]; perm[j] = tmp;
          j++;
#ifndef PETSC_USE_COMPLEX
        } else {
          tmp = perm[j-1]; perm[j-1] = perm[j]; perm[j] = perm[j+1]; perm[j+1] = tmp;
          j+=2;
        }
      } else {
        if (eigi[perm[j]] == 0) {
          tmp = perm[j-2]; perm[j-2] = perm[j]; perm[j] = perm[j-1]; perm[j-1] = tmp;
          j++;
        } else {
          tmp = perm[j-2]; perm[j-2] = perm[j]; perm[j] = tmp;
          tmp = perm[j-1]; perm[j-1] = perm[j+1]; perm[j+1] = tmp;
          j+=2;
        }
      }
#endif
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSortEigenvaluesReal"
/*@
   QEPSortEigenvaluesReal - Sorts a list of eigenvalues according to a certain
   criterion (version for real eigenvalues only).

   Not Collective

   Input Parameters:
+  qep   - the quadratic eigensolver context
.  n     - number of eigenvalue in the list
-  eig   - pointer to the array containing the eigenvalues (real)

   Output Parameter:
.  perm  - resulting permutation

   Note:
   The result is a list of indices in the original eigenvalue array 
   corresponding to the first nev eigenvalues sorted in the specified
   criterion.

   Level: developer

.seealso: QEPSortEigenvalues(), QEPSetWhichEigenpairs(), QEPCompareEigenvalues()
@*/
PetscErrorCode QEPSortEigenvaluesReal(QEP qep,PetscInt n,PetscReal *eig,PetscInt *perm)
{
  PetscErrorCode ierr;
  PetscScalar    re;
  PetscInt       i,j,result,tmp;

  PetscFunctionBegin;
  for (i=0; i<n; i++) { perm[i] = i; }
  /* insertion sort */
  for (i=1; i<n; i++) {
    re = eig[perm[i]];
    j = i-1;
    ierr = QEPCompareEigenvalues(qep,re,0.0,eig[perm[j]],0.0,&result);CHKERRQ(ierr);
    while (result>0 && j>=0) {
      tmp = perm[j]; perm[j] = perm[j+1]; perm[j+1] = tmp; j--;
      if (j>=0) {
        ierr = QEPCompareEigenvalues(qep,re,0.0,eig[perm[j]],0.0,&result);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPCompareEigenvalues"
/*@
   QEPCompareEigenvalues - Compares two (possibly complex) eigenvalues according
   to a certain criterion.

   Not Collective

   Input Parameters:
+  qep    - the quadratic eigensolver context
.  ar     - real part of the 1st eigenvalue
.  ai     - imaginary part of the 1st eigenvalue
.  br     - real part of the 2nd eigenvalue
-  bi     - imaginary part of the 2nd eigenvalue

   Output Parameter:
.  res    - result of comparison

   Notes:
   Returns an integer less than, equal to, or greater than zero if the first
   eigenvalue is considered to be respectively less than, equal to, or greater
   than the second one.

   The criterion of comparison is related to the 'which' parameter set with
   QEPSetWhichEigenpairs().

   Level: developer

.seealso: QEPSortEigenvalues(), QEPSetWhichEigenpairs()
@*/
PetscErrorCode QEPCompareEigenvalues(QEP qep,PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *result)
{
  PetscReal      a,b;

  PetscFunctionBegin;
  switch(qep->which) {
    case QEP_LARGEST_MAGNITUDE:
    case QEP_SMALLEST_MAGNITUDE:
      a = SlepcAbsEigenvalue(ar,ai);
      b = SlepcAbsEigenvalue(br,bi);
      break;
    case QEP_LARGEST_REAL:
    case QEP_SMALLEST_REAL:
      a = PetscRealPart(ar);
      b = PetscRealPart(br);
      break;
    case QEP_LARGEST_IMAGINARY:
    case QEP_SMALLEST_IMAGINARY:
#if defined(PETSC_USE_COMPLEX)
      a = PetscImaginaryPart(ar);
      b = PetscImaginaryPart(br);
#else
      a = PetscAbsReal(ai);
      b = PetscAbsReal(bi);
#endif
      break;
    default: SETERRQ(1,"Wrong value of which");
  }
  switch(qep->which) {
    case QEP_LARGEST_MAGNITUDE:
    case QEP_LARGEST_REAL:
    case QEP_LARGEST_IMAGINARY:
      if (a<b) *result = -1;
      else if (a>b) *result = 1;
      else *result = 0;
      break;
    default:
      if (a>b) *result = -1;
      else if (a<b) *result = 1;
      else *result = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetOperationCounters"
/*@
   QEPGetOperationCounters - Gets the total number of matrix-vector products, dot 
   products, and linear solve iterations used by the QEP object during the last
   QEPSolve() call.

   Not Collective

   Input Parameter:
.  qep - quadratic eigensolver context

   Output Parameter:
+  matvecs - number of matrix-vector product operations
.  dots    - number of dot product operations
-  lits    - number of linear iterations

   Notes:
   These counters are reset to zero at each successive call to QEPSolve().

   Level: intermediate

@*/
PetscErrorCode QEPGetOperationCounters(QEP qep,PetscInt* matvecs,PetscInt* dots,PetscInt* lits)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (matvecs) *matvecs = qep->matvecs; 
  if (dots) {
    ierr = IPGetOperationCounters(qep->ip,dots);CHKERRQ(ierr);
  }
  if (lits) *lits = qep->linits; 
  PetscFunctionReturn(0);
}
