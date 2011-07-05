/*
      EPS routines related to the solution process.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include <private/epsimpl.h>   /*I "slepceps.h" I*/

typedef struct {
  /* old values of eps */
  EPSWhich old_which;
  PetscErrorCode (*old_which_func)(EPS,PetscScalar,PetscScalar,PetscScalar,PetscScalar, PetscInt*,void*);
  void *old_which_ctx;
} EPSSortForSTData;

#undef __FUNCT__  
#define __FUNCT__ "EPSSortForSTFunc"
PetscErrorCode EPSSortForSTFunc(EPS eps,PetscScalar ar,PetscScalar ai,
                                PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx)
{
  EPSSortForSTData *data = (EPSSortForSTData*)ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* Back-transform the harmonic values */
  ierr = STBackTransform(eps->OP,1,&ar,&ai);CHKERRQ(ierr);
  ierr = STBackTransform(eps->OP,1,&br,&bi);CHKERRQ(ierr);

  /* Compare values using the user options for the eigenpairs selection */
  if (data->old_which==EPS_ALL) eps->which = EPS_TARGET_MAGNITUDE;
  else eps->which = data->old_which;
  eps->which_func = data->old_which_func;
  eps->which_ctx = data->old_which_ctx;
  ierr = EPSCompareEigenvalues(eps,ar,ai,br,bi,r);CHKERRQ(ierr);

  /* Restore the eps values */
  eps->which = EPS_WHICH_USER;
  eps->which_func = EPSSortForSTFunc;
  eps->which_ctx = data;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve"
/*@
   EPSSolve - Solves the eigensystem.

   Collective on EPS

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Options Database:
+   -eps_view - print information about the solver used
.   -eps_view_binary - save the matrices to the default binary file
-   -eps_plot_eigs - plot computed eigenvalues

   Level: beginner

.seealso: EPSCreate(), EPSSetUp(), EPSDestroy(), EPSSetTolerances() 
@*/
PetscErrorCode EPSSolve(EPS eps) 
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      re,im;
  PetscScalar    dot;
  PetscBool      flg,isfold,viewed=PETSC_FALSE;
  PetscViewer    viewer;
  PetscDraw      draw;
  PetscDrawSP    drawsp;
  STMatMode      matmode;
  char           filename[PETSC_MAX_PATH_LEN];
  char           view[10];
  EPSSortForSTData data;
  Mat            A,B;
  KSP            ksp;
  Vec            w,x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);

  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)eps)->prefix,"-eps_view_binary",&flg,PETSC_NULL);CHKERRQ(ierr); 
  if (flg) {
    ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
    ierr = MatView(A,PETSC_VIEWER_BINARY_(((PetscObject)eps)->comm));CHKERRQ(ierr);
    if (B) ierr = MatView(B,PETSC_VIEWER_BINARY_(((PetscObject)eps)->comm));CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetString(((PetscObject)eps)->prefix,"-eps_view",view,10,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscStrcmp(view,"before",&viewed);CHKERRQ(ierr);
    if (viewed){
      PetscViewer viewer;
      ierr = PetscViewerASCIIGetStdout(((PetscObject)eps)->comm,&viewer);CHKERRQ(ierr);
      ierr = EPSView(eps,viewer);CHKERRQ(ierr); 
    }
  }

  /* reset the convergence flag from the previous solves */
  eps->reason = EPS_CONVERGED_ITERATING;

  /* call setup */
  if (!eps->setupcalled) { ierr = EPSSetUp(eps);CHKERRQ(ierr); }
  eps->evecsavailable = PETSC_FALSE;
  eps->nconv = 0;
  eps->its = 0;
  for (i=0;i<eps->ncv;i++) eps->eigr[i]=eps->eigi[i]=eps->errest[i]=0.0;
  ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->ncv);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(EPS_Solve,eps,eps->V[0],eps->V[0],0);CHKERRQ(ierr);

  ierr = PetscTypeCompareAny((PetscObject)eps,&flg,EPSARPACK,EPSBLZPACK,EPSTRLAN,EPSBLOPEX,EPSPRIMME,"");CHKERRQ(ierr);
  if (!flg) {
    /* temporarily change which */
    data.old_which = eps->which;
    data.old_which_func = eps->which_func;
    data.old_which_ctx = eps->which_ctx;
    eps->which = EPS_WHICH_USER;
    eps->which_func = EPSSortForSTFunc;
    eps->which_ctx = &data;
  }

  /* call solver */
  ierr = (*eps->ops->solve)(eps);CHKERRQ(ierr);

  if (!flg) {
    /* restore which */
    eps->which = data.old_which;
    eps->which_func = data.old_which_func;
    eps->which_ctx = data.old_which_ctx;
  }

  ierr = STGetMatMode(eps->OP,&matmode);CHKERRQ(ierr);
  if (matmode == ST_MATMODE_INPLACE && eps->ispositive) {
    /* Purify eigenvectors before reverting operator */
    ierr = (*eps->ops->computevectors)(eps);CHKERRQ(ierr);    
  }
  ierr = STPostSolve(eps->OP);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Solve,eps,eps->V[0],eps->V[0],0);CHKERRQ(ierr);

  if (!eps->reason) {
    SETERRQ(((PetscObject)eps)->comm,1,"Internal error, solver returned without setting converged reason");
  }

  /* Map eigenvalues back to the original problem, necessary in some 
  * spectral transformations */
  if (eps->ops->backtransform) {
    ierr = (*eps->ops->backtransform)(eps);CHKERRQ(ierr);
  }

  /* Adjust left eigenvectors in generalized problems: y = B^T y */
  if (eps->isgeneralized && eps->leftvecs) {
    ierr = STGetOperators(eps->OP,PETSC_NULL,&B);CHKERRQ(ierr);
    ierr = KSPCreate(((PetscObject)eps)->comm,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,B,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = MatGetVecs(B,PETSC_NULL,&w);CHKERRQ(ierr);
    for (i=0;i<eps->nconv;i++) {
      ierr = VecCopy(eps->W[i],w);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(ksp,w,eps->W[i]);CHKERRQ(ierr);
    }
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = VecDestroy(&w);CHKERRQ(ierr);
  }

#if !defined(PETSC_USE_COMPLEX)
  /* reorder conjugate eigenvalues (positive imaginary first) */
  for (i=0; i<eps->nconv-1; i++) {
    if (eps->eigi[i] != 0) {
      if (eps->eigi[i] < 0) {
        eps->eigi[i] = -eps->eigi[i];
        eps->eigi[i+1] = -eps->eigi[i+1];
        if (!eps->evecsavailable) {
          /* the next correction only works with eigenvectors */
          ierr = (*eps->ops->computevectors)(eps);CHKERRQ(ierr);
        }
        ierr = VecScale(eps->V[i+1],-1.0);CHKERRQ(ierr);
      }
      i++;
    }
  }
#endif

  /* quick and dirty solution for FOLD: recompute eigenvalues as Rayleigh quotients */
  ierr = PetscTypeCompare((PetscObject)eps->OP,STFOLD,&isfold);CHKERRQ(ierr);
  if (isfold) {
    ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
    ierr = MatGetVecs(A,&w,PETSC_NULL);CHKERRQ(ierr);
    if (!eps->evecsavailable) { ierr = (*eps->ops->computevectors)(eps);CHKERRQ(ierr); }
    for (i=0;i<eps->nconv;i++) {
      x = eps->V[i];
      ierr = MatMult(A,x,w);CHKERRQ(ierr);
      ierr = VecDot(w,x,&eps->eigr[i]);CHKERRQ(ierr);
      if (eps->isgeneralized) {
        ierr = MatMult(B,x,w);CHKERRQ(ierr);
        ierr = VecDot(w,x,&dot);CHKERRQ(ierr);
        eps->eigr[i] /= dot;
      }
    }
    ierr = VecDestroy(&w);CHKERRQ(ierr);
  }

  /* sort eigenvalues according to eps->which parameter */
  flg = (eps->which == EPS_ALL);
  if (flg) eps->which = EPS_SMALLEST_REAL;
  ierr = EPSSortEigenvalues(eps,eps->nconv,eps->eigr,eps->eigi,eps->perm);CHKERRQ(ierr);
  if (flg) eps->which = EPS_ALL;

  if (!viewed) {
    ierr = PetscOptionsGetString(((PetscObject)eps)->prefix,"-eps_view",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg && !PetscPreLoadingOn) {
      ierr = PetscViewerASCIIOpen(((PetscObject)eps)->comm,filename,&viewer);CHKERRQ(ierr);
      ierr = EPSView(eps,viewer);CHKERRQ(ierr); 
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
  }

  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)eps)->prefix,"-eps_plot_eigs",&flg,PETSC_NULL);CHKERRQ(ierr); 
  if (flg) { 
    ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Computed Eigenvalues",
                             PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
    for (i=0;i<eps->nconv;i++) {
#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(eps->eigr[i]);
      im = PetscImaginaryPart(eps->eigi[i]);
#else
      re = eps->eigr[i];
      im = eps->eigi[i];
#endif
      ierr = PetscDrawSPAddPoint(drawsp,&re,&im);CHKERRQ(ierr);
    }
    ierr = PetscDrawSPDraw(drawsp);CHKERRQ(ierr);
    ierr = PetscDrawSPDestroy(&drawsp);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Remove the initial subspaces */
  eps->nini = 0;
  eps->ninil = 0;
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
#define __FUNCT__ "EPSGetOperationCounters"
/*@
   EPSGetOperationCounters - Gets the total number of operator applications,
   inner product operations and linear iterations used by the ST object 
   during the last EPSSolve() call.

   Not Collective

   Input Parameter:
.  eps - EPS context

   Output Parameter:
+  ops  - number of operator applications
.  dots - number of inner product operations
-  lits - number of linear iterations

   Notes:
   When the eigensolver algorithm invokes STApply() then a linear system 
   must be solved (except in the case of standard eigenproblems and shift
   transformation). The number of iterations required in this solve is
   accumulated into a counter whose value is returned by this function.

   These counters are reset to zero at each successive call to EPSSolve().

   Level: intermediate

@*/
PetscErrorCode EPSGetOperationCounters(EPS eps,PetscInt* ops,PetscInt* dots,PetscInt* lits)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (!eps->OP) { ierr = EPSGetST(eps,&eps->OP);CHKERRQ(ierr); }
  ierr = STGetOperationCounters(eps->OP,ops,lits);CHKERRQ(ierr);
  if (dots) {
    if (!eps->ip) { ierr = EPSGetIP(eps,&eps->ip);CHKERRQ(ierr); }
    ierr = IPGetOperationCounters(eps->ip,dots);CHKERRQ(ierr);
  }
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

.seealso: EPSGetEigenpair(), EPSGetConverged(), EPSSolve(), EPSGetInvariantSubspaceLeft()
@*/
PetscErrorCode EPSGetInvariantSubspace(EPS eps,Vec *v)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(v,2);
  PetscValidHeaderSpecific(*v,VEC_CLASSID,2);
  if (!eps->V) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSSolve must be called first"); 
  }
  if (!eps->ishermitian && eps->evecsavailable) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSGetInvariantSubspace must be called before EPSGetEigenpair,EPSGetEigenvector,EPSComputeRelativeError or EPSComputeResidualNorm"); 
  }
  if (eps->balance!=EPS_BALANCE_NONE && eps->D) {
    for (i=0;i<eps->nconv;i++) {
      ierr = VecPointwiseDivide(v[i],eps->V[i],eps->D);CHKERRQ(ierr);
      ierr = VecNormalize(v[i],PETSC_NULL);CHKERRQ(ierr);
    }
  }
  else {
    for (i=0;i<eps->nconv;i++) {
      ierr = VecCopy(eps->V[i],v[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetInvariantSubspaceLeft" 
/*@
   EPSGetInvariantSubspaceLeft - Gets an orthonormal basis of the computed left
   invariant subspace (only available in two-sided eigensolvers).

   Not Collective, but vectors are shared by all processors that share the EPS

   Input Parameter:
.  eps - the eigensolver context
  
   Output Parameter:
.  v - an array of vectors

   Notes:
   This function should be called after EPSSolve() has finished.

   The user should provide in v an array of nconv vectors, where nconv is
   the value returned by EPSGetConverged().

   The first k vectors returned in v span a left invariant subspace associated 
   with the first k computed eigenvalues (note that this is not true if the 
   k-th eigenvalue is complex and matrix A is real; in this case the first 
   k+1 vectors should be used). A left invariant subspace Y of A satisfies y'A 
   in Y for all y in Y (a similar definition applies for generalized 
   eigenproblems). 

   Level: intermediate

.seealso: EPSGetEigenpair(), EPSGetConverged(), EPSSolve(), EPSGetInvariantSubspace
@*/
PetscErrorCode EPSGetInvariantSubspaceLeft(EPS eps,Vec *v)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(v,2);
  PetscValidHeaderSpecific(*v,VEC_CLASSID,2);
  if (!eps->leftvecs) {
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must request left vectors with EPSSetLeftVectorsWanted"); 
  }
  if (!eps->W) {
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSSolve must be called first");
  }
  if (!eps->ishermitian && eps->evecsavailable) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSGetInvariantSubspaceLeft must be called before EPSGetEigenpairLeft,EPSComputeRelativeErrorLeft or EPSComputeResidualNormLeft"); 
  }
  for (i=0;i<eps->nconv;i++) {
    ierr = VecCopy(eps->W[i],v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetEigenpair" 
/*@
   EPSGetEigenpair - Gets the i-th solution of the eigenproblem as computed by 
   EPSSolve(). The solution consists in both the eigenvalue and the eigenvector.

   Not Collective, but vectors are shared by all processors that share the EPS

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

.seealso: EPSGetEigenvalue(), EPSGetEigenvector(), EPSGetEigenvectorLeft(), EPSSolve(), 
          EPSGetConverged(), EPSSetWhichEigenpairs(), EPSGetInvariantSubspace()
@*/
PetscErrorCode EPSGetEigenpair(EPS eps,PetscInt i,PetscScalar *eigr,PetscScalar *eigi,Vec Vr,Vec Vi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (!eps->eigr || !eps->eigi || !eps->V) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSSolve must be called first"); 
  }
  if (i<0 || i>=eps->nconv) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range"); 
  }
  ierr = EPSGetEigenvalue(eps,i,eigr,eigi);CHKERRQ(ierr);
  ierr = EPSGetEigenvector(eps,i,Vr,Vi);CHKERRQ(ierr);
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

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs(), 
          EPSGetEigenpair()
@*/
PetscErrorCode EPSGetEigenvalue(EPS eps,PetscInt i,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscInt       k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (!eps->eigr || !eps->eigi) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSSolve must be called first"); 
  }
  if (i<0 || i>=eps->nconv) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range"); 
  }
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

   Not Collective, but vectors are shared by all processors that share the EPS

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

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs(), 
          EPSGetEigenpair(), EPSGetEigenvectorLeft()
@*/
PetscErrorCode EPSGetEigenvector(EPS eps,PetscInt i,Vec Vr,Vec Vi)
{
  PetscErrorCode ierr;
  PetscInt       k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (Vr) { PetscValidHeaderSpecific(Vr,VEC_CLASSID,3); PetscCheckSameComm(eps,1,Vr,3); }
  if (Vi) { PetscValidHeaderSpecific(Vi,VEC_CLASSID,4); PetscCheckSameComm(eps,1,Vi,4); }
  if (!eps->V) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSSolve must be called first"); 
  }
  if (i<0 || i>=eps->nconv) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range"); 
  }
  if (!eps->evecsavailable && (Vr || Vi)) { 
    ierr = (*eps->ops->computevectors)(eps);CHKERRQ(ierr);
  }  
  if (!eps->perm) k = i;
  else k = eps->perm[i];
#if defined(PETSC_USE_COMPLEX)
  if (Vr) { ierr = VecCopy(eps->V[k],Vr);CHKERRQ(ierr); }
  if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
#else
  if (eps->eigi[k] > 0) { /* first value of conjugate pair */
    if (Vr) { ierr = VecCopy(eps->V[k],Vr);CHKERRQ(ierr); }
    if (Vi) { ierr = VecCopy(eps->V[k+1],Vi);CHKERRQ(ierr); }
  } else if (eps->eigi[k] < 0) { /* second value of conjugate pair */
    if (Vr) { ierr = VecCopy(eps->V[k-1],Vr);CHKERRQ(ierr); }
    if (Vi) { 
      ierr = VecCopy(eps->V[k],Vi);CHKERRQ(ierr); 
      ierr = VecScale(Vi,-1.0);CHKERRQ(ierr); 
    }
  } else { /* real eigenvalue */
    if (Vr) { ierr = VecCopy(eps->V[k],Vr);CHKERRQ(ierr); }
    if (Vi) { ierr = VecSet(Vi,0.0);CHKERRQ(ierr); }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetEigenvectorLeft" 
/*@
   EPSGetEigenvectorLeft - Gets the i-th left eigenvector as computed by EPSSolve() 
   (only available in two-sided eigensolvers). 

   Not Collective, but vectors are shared by all processors that share the EPS

   Input Parameters:
+  eps - eigensolver context 
-  i   - index of the solution

   Output Parameters:
+  Wr   - real part of eigenvector
-  Wi   - imaginary part of eigenvector

   Notes:
   If the corresponding eigenvalue is real, then Wi is set to zero. If PETSc is 
   configured with complex scalars the eigenvector is stored 
   directly in Wr (Wi is set to zero).

   The index i should be a value between 0 and nconv-1 (see EPSGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established 
   with EPSSetWhichEigenpairs().

   Level: beginner

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs(), 
          EPSGetEigenpair(), EPSGetEigenvector()
@*/
PetscErrorCode EPSGetEigenvectorLeft(EPS eps,PetscInt i,Vec Wr,Vec Wi)
{
  PetscErrorCode ierr;
  PetscInt       k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (Wr) { PetscValidHeaderSpecific(Wr,VEC_CLASSID,3); PetscCheckSameComm(eps,1,Wr,3); }
  if (Wi) { PetscValidHeaderSpecific(Wi,VEC_CLASSID,4); PetscCheckSameComm(eps,1,Wi,4); }
  if (!eps->leftvecs) {
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must request left vectors with EPSSetLeftVectorsWanted"); 
  }
  if (!eps->W) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSSolve must be called first");
  }
  if (i<0 || i>=eps->nconv) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range"); 
  }
  if (!eps->evecsavailable && (Wr || Wi)) { 
    ierr = (*eps->ops->computevectors)(eps);CHKERRQ(ierr);
  }  
  if (!eps->perm) k = i;
  else k = eps->perm[i];
#if defined(PETSC_USE_COMPLEX)
  if (Wr) { ierr = VecCopy(eps->W[k],Wr);CHKERRQ(ierr); }
  if (Wi) { ierr = VecSet(Wi,0.0);CHKERRQ(ierr); }
#else
  if (eps->eigi[k] > 0) { /* first value of conjugate pair */
    if (Wr) { ierr = VecCopy(eps->W[k],Wr);CHKERRQ(ierr); }
    if (Wi) { ierr = VecCopy(eps->W[k+1],Wi);CHKERRQ(ierr); }
  } else if (eps->eigi[k] < 0) { /* second value of conjugate pair */
    if (Wr) { ierr = VecCopy(eps->W[k-1],Wr);CHKERRQ(ierr); }
    if (Wi) { 
      ierr = VecCopy(eps->W[k],Wi);CHKERRQ(ierr); 
      ierr = VecScale(Wi,-1.0);CHKERRQ(ierr); 
    }
  } else { /* real eigenvalue */
    if (Wr) { ierr = VecCopy(eps->W[k],Wr);CHKERRQ(ierr); }
    if (Wi) { ierr = VecSet(Wi,0.0);CHKERRQ(ierr); }
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
  if (!eps->eigr || !eps->eigi) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSSolve must be called first"); 
  }
  if (i<0 || i>=eps->nconv) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range"); 
  }
  if (eps->perm) i = eps->perm[i];  
  if (errest) *errest = eps->errest[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetErrorEstimateLeft" 
/*@
   EPSGetErrorEstimateLeft - Returns the left error estimate associated to the i-th 
   computed eigenpair (only available in two-sided eigensolvers).

   Not Collective

   Input Parameter:
+  eps - eigensolver context 
-  i   - index of eigenpair

   Output Parameter:
.  errest - the left error estimate

   Notes:
   This is the error estimate used internally by the eigensolver. The actual
   error bound can be computed with EPSComputeRelativeErrorLeft(). See also the users
   manual for details.

   Level: advanced

.seealso: EPSComputeRelativeErrorLeft()
@*/
PetscErrorCode EPSGetErrorEstimateLeft(EPS eps,PetscInt i,PetscReal *errest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(errest,3);
  if (!eps->eigr || !eps->eigi) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSSolve must be called first"); 
  }
  if (!eps->leftvecs) {
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must request left vectors with EPSSetLeftVectorsWanted"); 
  }
  if (i<0 || i>=eps->nconv) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range"); 
  }
  if (eps->perm) i = eps->perm[i];  
  if (errest) *errest = eps->errest_left[i];
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
  Vec            u,w;
  Mat            A,B;
#if !defined(PETSC_USE_COMPLEX)
  Vec            v;
  PetscReal      ni,nr;
#endif
  
  PetscFunctionBegin;
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->V[0],&u);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->V[0],&w);CHKERRQ(ierr);
  
#if !defined(PETSC_USE_COMPLEX)
  if (ki == 0 || 
    PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
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
    ierr = VecDuplicate(eps->V[0],&v);CHKERRQ(ierr);
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
   EPSComputeResidualNorm - Computes the norm of the residual vector associated with 
   the i-th computed eigenpair.

   Collective on EPS

   Input Parameter:
.  eps - the eigensolver context
.  i   - the solution index

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
  ierr = VecDuplicate(eps->V[0],&xr);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->V[0],&xi);CHKERRQ(ierr);
  ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
  ierr = EPSComputeResidualNorm_Private(eps,kr,ki,xr,xi,norm);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeResidualNormLeft"
/*@
   EPSComputeResidualNormLeft - Computes the norm of the residual vector associated with 
   the i-th computed left eigenvector (only available in two-sided eigensolvers).

   Collective on EPS

   Input Parameter:
.  eps - the eigensolver context
.  i   - the solution index

   Output Parameter:
.  norm - the residual norm, computed as ||y'A-ky'B||_2 where k is the 
   eigenvalue and y is the left eigenvector. 
   If k=0 then the residual norm is computed as ||y'A||_2.

   Notes:
   The index i should be a value between 0 and nconv-1 (see EPSGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established 
   with EPSSetWhichEigenpairs().

   Level: beginner

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs()
@*/
PetscErrorCode EPSComputeResidualNormLeft(EPS eps,PetscInt i,PetscReal *norm)
{
  PetscErrorCode ierr;
  Vec            u,v,w,xr,xi;
  Mat            A,B;
  PetscScalar    kr,ki;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      ni,nr;
#endif
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,i,2);
  PetscValidPointer(norm,3);
  if (!eps->leftvecs) {
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must request left vectors with EPSSetLeftVectorsWanted"); 
  }
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->W[0],&u);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->W[0],&v);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->W[0],&w);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->W[0],&xr);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->W[0],&xi);CHKERRQ(ierr);
  ierr = EPSGetEigenvalue(eps,i,&kr,&ki);CHKERRQ(ierr);
  ierr = EPSGetEigenvectorLeft(eps,i,xr,xi);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
  if (ki == 0 || 
    PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    ierr = MatMultTranspose(A,xr,u);CHKERRQ(ierr); /* u=A'*x */
    if (PetscAbsScalar(kr) > PETSC_MACHINE_EPSILON) {
      if (eps->isgeneralized) { ierr = MatMultTranspose(B,xr,w);CHKERRQ(ierr); }
      else { ierr = VecCopy(xr,w);CHKERRQ(ierr); } /* w=B'*x */
      ierr = VecAXPY(u,-kr,w);CHKERRQ(ierr); /* u=A'*x-k*B'*x */
    }
    ierr = VecNorm(u,NORM_2,norm);CHKERRQ(ierr);  
#if !defined(PETSC_USE_COMPLEX)
  } else {
    ierr = MatMultTranspose(A,xr,u);CHKERRQ(ierr); /* u=A'*xr */
    if (eps->isgeneralized) { ierr = MatMultTranspose(B,xr,v);CHKERRQ(ierr); }
    else { ierr = VecCopy(xr,v);CHKERRQ(ierr); } /* v=B'*xr */
    ierr = VecAXPY(u,-kr,v);CHKERRQ(ierr); /* u=A'*xr-kr*B'*xr */
    if (eps->isgeneralized) { ierr = MatMultTranspose(B,xi,w);CHKERRQ(ierr); }
    else { ierr = VecCopy(xi,w);CHKERRQ(ierr); } /* w=B'*xi */
    ierr = VecAXPY(u,ki,w);CHKERRQ(ierr); /* u=A'*xr-kr*B'*xr+ki*B'*xi */
    ierr = VecNorm(u,NORM_2,&nr);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,xi,u);CHKERRQ(ierr); /* u=A'*xi */
    ierr = VecAXPY(u,-kr,w);CHKERRQ(ierr); /* u=A'*xi-kr*B'*xi */
    ierr = VecAXPY(u,-ki,v);CHKERRQ(ierr); /* u=A'*xi-kr*B'*xi-ki*B'*xr */
    ierr = VecNorm(u,NORM_2,&ni);CHKERRQ(ierr);
    *norm = SlepcAbsEigenvalue(nr,ni);
  }
#endif

  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
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
  ierr = (*eps->conv_func)(eps,kr,ki,norm/er,error,eps->conv_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeRelativeError"
/*@
   EPSComputeRelativeError - Computes the relative error bound associated 
   with the i-th computed eigenpair.

   Collective on EPS

   Input Parameter:
.  eps - the eigensolver context
.  i   - the solution index

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
  ierr = VecDuplicate(eps->V[0],&xr);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->V[0],&xi);CHKERRQ(ierr);
  ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
  ierr = EPSComputeRelativeError_Private(eps,kr,ki,xr,xi,error);CHKERRQ(ierr);  
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeRelativeErrorLeft"
/*@
   EPSComputeRelativeErrorLeft - Computes the relative error bound associated 
   with the i-th computed eigenvalue and left eigenvector (only available in
   two-sided eigensolvers).

   Collective on EPS

   Input Parameter:
.  eps - the eigensolver context
.  i   - the solution index

   Output Parameter:
.  error - the relative error bound, computed as ||y'A-ky'B||_2/||ky||_2 where 
   k is the eigenvalue and y is the left eigenvector. 
   If k=0 the relative error is computed as ||y'A||_2/||y||_2.

   Level: beginner

.seealso: EPSSolve(), EPSComputeResidualNormLeft(), EPSGetErrorEstimateLeft()
@*/
PetscErrorCode EPSComputeRelativeErrorLeft(EPS eps,PetscInt i,PetscReal *error)
{
  PetscErrorCode ierr;
  Vec            xr,xi;  
  PetscScalar    kr,ki;  
  PetscReal      norm,er;
#if !defined(PETSC_USE_COMPLEX)
  Vec            u;
  PetscReal      ei;
#endif
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);  
  ierr = EPSComputeResidualNormLeft(eps,i,&norm);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(eps,i,2);
  PetscValidPointer(error,3);
  ierr = VecDuplicate(eps->W[0],&xr);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->W[0],&xi);CHKERRQ(ierr);
  ierr = EPSGetEigenvalue(eps,i,&kr,&ki);CHKERRQ(ierr);
  ierr = EPSGetEigenvectorLeft(eps,i,xr,xi);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
  if (ki == 0 || 
    PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    ierr = VecNorm(xr,NORM_2,&er);CHKERRQ(ierr);
    if (PetscAbsScalar(kr) > PETSC_MACHINE_EPSILON) {
      *error =  norm / (PetscAbsScalar(kr) * er);
    } else {
      *error = norm / er;
    }
#if !defined(PETSC_USE_COMPLEX)
  } else {
    ierr = VecDuplicate(xi,&u);CHKERRQ(ierr);  
    ierr = VecCopy(xi,u);CHKERRQ(ierr);  
    ierr = VecAXPBY(u,kr,-ki,xr);CHKERRQ(ierr);   
    ierr = VecNorm(u,NORM_2,&er);CHKERRQ(ierr);  
    ierr = VecAXPBY(xi,kr,ki,xr);CHKERRQ(ierr);      
    ierr = VecNorm(xi,NORM_2,&ei);CHKERRQ(ierr);  
    ierr = VecDestroy(&u);CHKERRQ(ierr);  
    *error = norm / SlepcAbsEigenvalue(er,ei);
  }
#endif    
  
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define SWAP(a,b,t) {t=a;a=b;b=t;}

#undef __FUNCT__  
#define __FUNCT__ "EPSSortEigenvalues"
/*@
   EPSSortEigenvalues - Sorts a list of eigenvalues according to the criterion 
   specified via EPSSetWhichEigenpairs().

   Not Collective

   Input Parameters:
+  eps   - the eigensolver context
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

.seealso: EPSSortEigenvaluesReal(), EPSSetWhichEigenpairs()
@*/
PetscErrorCode EPSSortEigenvalues(EPS eps,PetscInt n,PetscScalar *eigr,PetscScalar *eigi,PetscInt *perm)
{
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,result,tmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);  
  PetscValidScalarPointer(eigr,3);
  PetscValidScalarPointer(eigi,4);
  PetscValidIntPointer(perm,5);
  for (i=0; i<n; i++) { perm[i] = i; }
  /* insertion sort */
  for (i=n-1; i>=0; i--) {
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
      ierr = EPSCompareEigenvalues(eps,re,im,eigr[perm[j]],eigi[perm[j]],&result);CHKERRQ(ierr);
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
#define __FUNCT__ "EPSSortEigenvaluesReal"
/*@
   EPSSortEigenvaluesReal - Sorts a list of eigenvalues according to a certain
   criterion (version for real eigenvalues only).

   Not Collective

   Input Parameters:
+  eps   - the eigensolver context
.  n     - number of eigenvalue in the list
-  eig   - pointer to the array containing the eigenvalues (real)

   Output Parameter:
.  perm  - resulting permutation

   Note:
   The result is a list of indices in the original eigenvalue array 
   corresponding to the first nev eigenvalues sorted in the specified
   criterion.

   Level: developer

.seealso: EPSSortEigenvalues(), EPSSetWhichEigenpairs(), EPSCompareEigenvalues()
@*/
PetscErrorCode EPSSortEigenvaluesReal(EPS eps,PetscInt n,PetscReal *eig,PetscInt *perm)
{
  PetscErrorCode ierr;
  PetscScalar    re;
  PetscInt       i,j,result,tmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);  
  PetscValidPointer(eig,3);
  PetscValidIntPointer(perm,4);
  for (i=0; i<n; i++) { perm[i] = i; }
  /* insertion sort */
  for (i=1; i<n; i++) {
    re = eig[perm[i]];
    j = i-1;
    ierr = EPSCompareEigenvalues(eps,re,0.0,eig[perm[j]],0.0,&result);CHKERRQ(ierr);
    while (result<=0 && j>=0) {
      tmp = perm[j]; perm[j] = perm[j+1]; perm[j+1] = tmp; j--;
      if (j>=0) {
        ierr = EPSCompareEigenvalues(eps,re,0.0,eig[perm[j]],0.0,&result);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSCompareEigenvalues"
/*@
   EPSCompareEigenvalues - Compares two (possibly complex) eigenvalues according
   to a certain criterion.

   Not Collective

   Input Parameters:
+  eps   - the eigensolver context
.  ar     - real part of the 1st eigenvalue
.  ai     - imaginary part of the 1st eigenvalue
.  br     - real part of the 2nd eigenvalue
-  bi     - imaginary part of the 2nd eigenvalue

   Output Parameter:
.  res    - result of comparison

   Notes:
   The returning parameter 'res' can be:
+  negative - if the 1st eigenvalue is preferred to the 2st one
.  zero     - if both eigenvalues are equally preferred
-  positive - if the 2st eigenvalue is preferred to the 1st one

   The criterion of comparison is related to the 'which' parameter set with
   EPSSetWhichEigenpairs().

   Level: developer

.seealso: EPSSortEigenvalues(), EPSSetWhichEigenpairs()
@*/
PetscErrorCode EPSCompareEigenvalues(EPS eps,PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *result)
{
  PetscErrorCode ierr;
  PetscReal      a,b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);  
  PetscValidIntPointer(result,6);
  switch(eps->which) {
    case EPS_WHICH_USER:
      if (!eps->which_func) SETERRQ(((PetscObject)eps)->comm,1,"Undefined eigenvalue comparison function");
      ierr = (*eps->which_func)(eps,ar,ai,br,bi,result,eps->which_ctx);CHKERRQ(ierr);
      a = 0.0; b = 0.0;
      break;
    case EPS_LARGEST_MAGNITUDE:
    case EPS_SMALLEST_MAGNITUDE:
      a = SlepcAbsEigenvalue(ar,ai);
      b = SlepcAbsEigenvalue(br,bi);
      break;
    case EPS_LARGEST_REAL:
    case EPS_SMALLEST_REAL:
      a = PetscRealPart(ar);
      b = PetscRealPart(br);
      break;
    case EPS_LARGEST_IMAGINARY:
    case EPS_SMALLEST_IMAGINARY:
#if defined(PETSC_USE_COMPLEX)
      a = PetscImaginaryPart(ar);
      b = PetscImaginaryPart(br);
#else
      a = PetscAbsReal(ai);
      b = PetscAbsReal(bi);
#endif
      break;
    case EPS_TARGET_MAGNITUDE:
      /* complex target only allowed if scalartype=complex */
      a = SlepcAbsEigenvalue(ar-eps->target,ai);
      b = SlepcAbsEigenvalue(br-eps->target,bi);
      break;
    case EPS_TARGET_REAL:
      a = PetscAbsReal(PetscRealPart(ar-eps->target));
      b = PetscAbsReal(PetscRealPart(br-eps->target));
      break;
    case EPS_TARGET_IMAGINARY:
#if !defined(PETSC_USE_COMPLEX)
      /* complex target only allowed if scalartype=complex */
      a = PetscAbsReal(ai);
      b = PetscAbsReal(bi);
#else
      a = PetscAbsReal(PetscImaginaryPart(ar-eps->target));
      b = PetscAbsReal(PetscImaginaryPart(br-eps->target));
#endif
      break;
    default: SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of which");
  }
  switch(eps->which) {
    case EPS_WHICH_USER:
      break;
    case EPS_LARGEST_MAGNITUDE:
    case EPS_LARGEST_REAL:
    case EPS_LARGEST_IMAGINARY:
      if (a<b) *result = 1;
      else if (a>b) *result = -1;
      else *result = 0;
      break;
    default:
      if (a>b) *result = 1;
      else if (a<b) *result = -1;
      else *result = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetStartVector"
/*@
   EPSGetStartVector - Gets a suitable vector to be used as the starting vector
   for the recurrence that builds the right subspace.

   Collective on EPS and Vec

   Input Parameters:
+  eps - the eigensolver context
-  i   - iteration number

   Output Parameters:
+  vec - the start vector
-  breakdown - flag indicating that a breakdown has occurred

   Notes:
   The start vector is computed from another vector: for the first step (i=0),
   the first initial vector is used (see EPSSetInitialSpace()); otherwise a random
   vector is created. Then this vector is forced to be in the range of OP (only
   for generalized definite problems) and orthonormalized with respect to all
   V-vectors up to i-1.

   The flag breakdown is set to true if either i=0 and the vector belongs to the
   deflation space, or i>0 and the vector is linearly dependent with respect
   to the V-vectors.

   The caller must pass a vector already allocated with dimensions conforming
   to the initial vector. This vector is overwritten.

   Level: developer

.seealso: EPSSetInitialSpace()
@*/
PetscErrorCode EPSGetStartVector(EPS eps,PetscInt i,Vec vec,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscReal      norm;
  PetscBool      lindep;
  Vec            w;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,i,2);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,3);
  PetscCheckSameComm(eps,1,vec,3);

  ierr = VecDuplicate(eps->V[0],&w);CHKERRQ(ierr);

  /* For the first step, use the first initial vector, otherwise a random one */
  if (i==0 && eps->nini>0) {
    ierr = VecCopy(eps->V[0],w);CHKERRQ(ierr);
  } else {
    ierr = SlepcVecSetRandom(w,eps->rand);CHKERRQ(ierr);
  }

  /* Force the vector to be in the range of OP for definite generalized problems */
  if (eps->ispositive) {
    ierr = STApply(eps->OP,w,vec);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(w,vec);CHKERRQ(ierr);
  }

  /* Orthonormalize the vector with respect to previous vectors */
  ierr = IPOrthogonalize(eps->ip,eps->nds,eps->DS,i,PETSC_NULL,eps->V,vec,PETSC_NULL,&norm,&lindep);CHKERRQ(ierr);
  if (breakdown) *breakdown = lindep;
  else if (lindep || norm == 0.0) {
    if (i==0) { SETERRQ(((PetscObject)eps)->comm,1,"Initial vector is zero or belongs to the deflation space"); } 
    else { SETERRQ(((PetscObject)eps)->comm,1,"Unable to generate more start vectors"); }
  }
  ierr = VecScale(vec,1.0/norm);CHKERRQ(ierr);

  ierr = VecDestroy(&w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetStartVectorLeft"
/*@
   EPSGetStartVectorLeft - Gets a suitable vector to be used as the starting vector
   in the recurrence that builds the left subspace (in methods that work with two
   subspaces).

   Collective on EPS and Vec

   Input Parameters:
+  eps - the eigensolver context
-  i   - iteration number

   Output Parameter:
+  vec - the start vector
-  breakdown - flag indicating that a breakdown has occurred

   Notes:
   The start vector is computed from another vector: for the first step (i=0),
   the first left initial vector is used (see EPSSetInitialSpaceLeft()); otherwise 
   a random vector is created. Then this vector is forced to be in the range 
   of OP' and orthonormalized with respect to all W-vectors up to i-1.

   The flag breakdown is set to true if i>0 and the vector is linearly dependent
   with respect to the W-vectors.

   The caller must pass a vector already allocated with dimensions conforming
   to the left initial vector. This vector is overwritten.

   Level: developer

.seealso: EPSSetInitialSpaceLeft()

@*/
PetscErrorCode EPSGetStartVectorLeft(EPS eps,PetscInt i,Vec vec,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscReal      norm;
  PetscBool      lindep;
  Vec            w;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,i,2);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,3);
  PetscCheckSameComm(eps,1,vec,3);

  ierr = VecDuplicate(eps->W[0],&w);CHKERRQ(ierr);

  /* For the first step, use the first initial left vector, otherwise a random one */
  if (i==0 && eps->ninil>0) {
    ierr = VecCopy(eps->W[0],w);CHKERRQ(ierr);
  } else {
    ierr = SlepcVecSetRandom(w,eps->rand);CHKERRQ(ierr);
  }

  /* Force the vector to be in the range of OP' */
  ierr = STApplyTranspose(eps->OP,w,vec);CHKERRQ(ierr);

  /* Orthonormalize the vector with respect to previous vectors */
  ierr = IPOrthogonalize(eps->ip,0,PETSC_NULL,i,PETSC_NULL,eps->W,vec,PETSC_NULL,&norm,&lindep);CHKERRQ(ierr);
  if (breakdown) *breakdown = lindep;
  else if (lindep || norm == 0.0) {
    if (i==0) { SETERRQ(((PetscObject)eps)->comm,1,"Left initial vector is zero"); }
    else { SETERRQ(((PetscObject)eps)->comm,1,"Unable to generate more left start vectors"); }
  }
  ierr = VecScale(vec,1/norm);CHKERRQ(ierr);

  ierr = VecDestroy(&w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
