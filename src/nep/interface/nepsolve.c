/*
      NEP routines related to the solution process.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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
  ierr = NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->eigi,nep->errest,nep->ncv);CHKERRQ(ierr);

  ierr = DSSetEigenvalueComparison(nep->ds,nep->which_func,nep->which_ctx);CHKERRQ(ierr);

  ierr = (*nep->ops->solve)(nep);CHKERRQ(ierr);

  if (!nep->reason) SETERRQ(((PetscObject)nep)->comm,PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");

#if !defined(PETSC_USE_COMPLEX)
  /* reorder conjugate eigenvalues (positive imaginary first) */
  for (i=0;i<nep->nconv-1;i++) {
    if (nep->eigi[i] != 0) {
      if (nep->eigi[i] < 0) {
        nep->eigi[i] = -nep->eigi[i];
        nep->eigi[i+1] = -nep->eigi[i+1];
        ierr = VecScale(nep->V[i+1],-1.0);CHKERRQ(ierr);
      }
      i++;
    }
  }
#endif

  /* sort eigenvalues according to nep->which parameter */
  ierr = NEPSortEigenvalues(nep,nep->nconv,nep->eigr,nep->eigi,nep->perm);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(NEP_Solve,nep,0,0,0);CHKERRQ(ierr);

  /* various viewers */
  ierr = PetscOptionsGetViewer(((PetscObject)nep)->comm,((PetscObject)nep)->prefix,"-nep_view",&viewer,&format,&flg);CHKERRQ(ierr);
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
      im = PetscImaginaryPart(nep->eigi[i]);
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
  *reason = nep->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "NEPGetEigenpair" 
/*@
   NEPGetEigenpair - Gets the i-th solution of the eigenproblem as computed by 
   NEPSolve(). The solution consists in both the eigenvalue and the eigenvector.

   Logically Collective on EPS

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
  if (Vr) { PetscValidHeaderSpecific(Vr,VEC_CLASSID,6); PetscCheckSameComm(nep,1,Vr,6); }
  if (Vi) { PetscValidHeaderSpecific(Vi,VEC_CLASSID,7); PetscCheckSameComm(nep,1,Vi,7); }
  if (!nep->eigr || !nep->eigi || !nep->V) SETERRQ(((PetscObject)nep)->comm,PETSC_ERR_ARG_WRONGSTATE,"NEPSolve must be called first"); 
  if (i<0 || i>=nep->nconv) SETERRQ(((PetscObject)nep)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range"); 

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
  if (Vr) { ierr = VecCopy(nep->V[k],Vr);CHKERRQ(ierr); }
  if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
#else
  if (nep->eigi[k]>0) { /* first value of conjugate pair */
    if (Vr) { ierr = VecCopy(nep->V[k],Vr);CHKERRQ(ierr); }
    if (Vi) { ierr = VecCopy(nep->V[k+1],Vi);CHKERRQ(ierr); }
  } else if (nep->eigi[k]<0) { /* second value of conjugate pair */
    if (Vr) { ierr = VecCopy(nep->V[k-1],Vr);CHKERRQ(ierr); }
    if (Vi) { 
      ierr = VecCopy(nep->V[k],Vi);CHKERRQ(ierr); 
      ierr = VecScale(Vi,-1.0);CHKERRQ(ierr); 
    }
  } else { /* real eigenvalue */
    if (Vr) { ierr = VecCopy(nep->V[k],Vr);CHKERRQ(ierr); }
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
  if (!nep->eigr || !nep->eigi) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"NEPSolve must be called first"); 
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
PetscErrorCode NEPComputeResidualNorm_Private(NEP nep,PetscScalar kr,PetscScalar ki,Vec xr,Vec xi,PetscReal *norm)
{
  PetscErrorCode ierr;
  Vec            u;
  Mat            T=nep->function;
  MatStructure   mats;
  
  PetscFunctionBegin;
  ierr = VecDuplicate(nep->V[0],&u);CHKERRQ(ierr);
  ierr = NEPComputeFunction(nep,kr,ki,&T,&T,&mats);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (ki == 0 || PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    ierr = MatMult(T,xr,u);CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_2,norm);CHKERRQ(ierr);  
#if !defined(PETSC_USE_COMPLEX)
  } else {
    SETERRQ(PETSC_COMM_SELF,1,"Not implemented");
  }
#endif
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
.  nep - the nonlinear eigensolver context
.  i   - the solution index

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
  ierr = VecDuplicate(nep->V[0],&xr);CHKERRQ(ierr);
  ierr = VecDuplicate(nep->V[0],&xi);CHKERRQ(ierr);
  ierr = NEPGetEigenpair(nep,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
  ierr = NEPComputeResidualNorm_Private(nep,kr,ki,xr,xi,norm);CHKERRQ(ierr);
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
PetscErrorCode NEPComputeRelativeError_Private(NEP nep,PetscScalar kr,PetscScalar ki,Vec xr,Vec xi,PetscReal *error)
{
  PetscErrorCode ierr;
  PetscReal      norm,er;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      ei;
#endif
  
  PetscFunctionBegin;
  ierr = NEPComputeResidualNorm_Private(nep,kr,ki,xr,xi,&norm);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (ki == 0 || PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    ierr = VecNorm(xr,NORM_2,&er);CHKERRQ(ierr);
    if (PetscAbsScalar(kr) > norm) {
      *error = norm/(PetscAbsScalar(kr)*er);
    } else {
      *error = norm/er;
    }
#if !defined(PETSC_USE_COMPLEX)
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
#define __FUNCT__ "NEPComputeRelativeError"
/*@
   NEPComputeRelativeError - Computes the relative error bound associated 
   with the i-th computed eigenpair.

   Collective on NEP

   Input Parameter:
.  nep - the nonlinear eigensolver context
.  i   - the solution index

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
  ierr = VecDuplicate(nep->V[0],&xr);CHKERRQ(ierr);
  ierr = VecDuplicate(nep->V[0],&xi);CHKERRQ(ierr);
  ierr = NEPGetEigenpair(nep,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
  ierr = NEPComputeRelativeError_Private(nep,kr,ki,xr,xi,error);CHKERRQ(ierr);  
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "NEPSortEigenvalues"
/*@
   NEPSortEigenvalues - Sorts a list of eigenvalues according to the criterion 
   specified via NEPSetWhichEigenpairs().

   Not Collective

   Input Parameters:
+  nep   - the nonlinear eigensolver context
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

.seealso: NEPSetWhichEigenpairs()
@*/
PetscErrorCode NEPSortEigenvalues(NEP nep,PetscInt n,PetscScalar *eigr,PetscScalar *eigi,PetscInt *perm)
{
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,result,tmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);  
  PetscValidScalarPointer(eigr,3);
  PetscValidScalarPointer(eigi,4);
  PetscValidIntPointer(perm,5);
  for (i=0;i<n;i++) perm[i] = i;
  /* insertion sort */
  for (i=n-1;i>=0;i--) {
    re = eigr[perm[i]];
    im = eigi[perm[i]];
    j = i + 1;
#if !defined(PETSC_USE_COMPLEX)
    if (im != 0) {
      /* complex eigenvalue */
      i--;
      im = eigi[perm[i]];
    }
#endif
    while (j<n) {
      ierr = NEPCompareEigenvalues(nep,re,im,eigr[perm[j]],eigi[perm[j]],&result);CHKERRQ(ierr);
      if (result < 0) break;
#if !defined(PETSC_USE_COMPLEX)
      /* keep together every complex conjugated eigenpair */
      if (im == 0) { 
        if (eigi[perm[j]] == 0) {
#endif
          tmp = perm[j-1]; perm[j-1] = perm[j]; perm[j] = tmp;
          j++;
#if !defined(PETSC_USE_COMPLEX)
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
#define __FUNCT__ "NEPCompareEigenvalues"
/*@
   NEPCompareEigenvalues - Compares two (possibly complex) eigenvalues according
   to a certain criterion.

   Not Collective

   Input Parameters:
+  nep    - the nonlinear eigensolver context
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
   NEPSetWhichEigenpairs().

   Level: developer

.seealso: NEPSortEigenvalues(), NEPSetWhichEigenpairs()
@*/
PetscErrorCode NEPCompareEigenvalues(NEP nep,PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *result)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);  
  PetscValidIntPointer(result,6);
  if (!nep->which_func) SETERRQ(PETSC_COMM_SELF,1,"Undefined eigenvalue comparison function");
  ierr = (*nep->which_func)(ar,ai,br,bi,result,nep->which_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "NEPGetOperationCounters"
/*@
   NEPGetOperationCounters - Gets the total number of function evaluations, dot 
   products, and linear solve iterations used by the NEP object during the last
   NEPSolve() call.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigensolver context

   Output Parameter:
+  nfuncs - number of function evaluations
.  dots   - number of dot product operations
-  lits   - number of linear iterations

   Notes:
   These counters are reset to zero at each successive call to NEPSolve().

   Level: intermediate

@*/
PetscErrorCode NEPGetOperationCounters(NEP nep,PetscInt* nfuncs,PetscInt* dots,PetscInt* lits)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (nfuncs) *nfuncs = nep->nfuncs; 
  if (dots) {
    if (!nep->ip) { ierr = NEPGetIP(nep,&nep->ip);CHKERRQ(ierr); }
    ierr = IPGetOperationCounters(nep->ip,dots);CHKERRQ(ierr);
  }
  if (lits) *lits = nep->linits; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPComputeFunction"
/*@
   NEPComputeFunction - Computes the function matrix T(lambda) that has been
   set with NEPSetFunction().

   Collective on NEP and Mat

   Input Parameters:
+  nep - the NEP context
.  wr  - real part of the scalar argument
-  wi  - imaginary part of the scalar argument

   Output Parameters:
+  A   - Function matrix
.  B   - optional preconditioning matrix
-  flg - flag indicating matrix structure (see MatStructure enum)

   Notes:
   NEPComputeFunction() is typically used within nonlinear eigensolvers
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso: NEPSetFunction(), NEPGetFunction()
@*/
PetscErrorCode NEPComputeFunction(NEP nep,PetscScalar wr,PetscScalar wi,Mat *A,Mat *B,MatStructure *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(flg,6);

  if (!nep->fun_func) SETERRQ(((PetscObject)nep)->comm,PETSC_ERR_USER,"Must call NEPSetFunction() first");

  *flg = DIFFERENT_NONZERO_PATTERN;
  ierr = PetscLogEventBegin(NEP_FunctionEval,nep,*A,*B,0);CHKERRQ(ierr);

  PetscStackPush("NEP user Function function");
  ierr = (*nep->fun_func)(nep,wr,wi,A,B,flg,nep->fun_ctx);CHKERRQ(ierr);
  PetscStackPop;

  ierr = PetscLogEventEnd(NEP_FunctionEval,nep,*A,*B,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPComputeJacobian"
/*@
   NEPComputeJacobian - Computes the Jacobian matrix T'(lambda) that has been
   set with NEPSetJacobian().

   Collective on NEP and Mat

   Input Parameters:
+  nep - the NEP context
.  wr  - real part of the scalar argument
-  wi  - imaginary part of the scalar argument

   Output Parameters:
+  A   - Jacobian matrix
.  B   - optional preconditioning matrix
-  flg - flag indicating matrix structure (see MatStructure enum)

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear eigensolvers.

   Level: developer

.seealso: NEPSetJacobian(), NEPGetJacobian()
@*/
PetscErrorCode NEPComputeJacobian(NEP nep,PetscScalar wr,PetscScalar wi,Mat *A,Mat *B,MatStructure *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(flg,6);

  if (!nep->jac_func) SETERRQ(((PetscObject)nep)->comm,PETSC_ERR_USER,"Must call NEPSetJacobian() first");

  *flg = DIFFERENT_NONZERO_PATTERN;
  ierr = PetscLogEventBegin(NEP_JacobianEval,nep,*A,*B,0);CHKERRQ(ierr);

  PetscStackPush("NEP user Jacobian function");
  ierr = (*nep->jac_func)(nep,wr,wi,A,B,flg,nep->jac_ctx);CHKERRQ(ierr);
  PetscStackPop;

  ierr = PetscLogEventEnd(NEP_JacobianEval,nep,*A,*B,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

