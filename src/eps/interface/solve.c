#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/

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
  int            i;
  PetscReal      re,im;
  PetscTruth     flg;
  PetscViewer    viewer;
  PetscDraw      draw;
  PetscDrawSP    drawsp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  ierr = PetscOptionsHasName(eps->prefix,"-eps_view_binary",&flg);CHKERRQ(ierr); 
  if (flg) {
    Mat A,B;
    ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
    ierr = MatView(A,PETSC_VIEWER_BINARY_(eps->comm));CHKERRQ(ierr);
    if (B) ierr = MatView(B,PETSC_VIEWER_BINARY_(eps->comm));CHKERRQ(ierr);
  }

  /* reset the convergence flag from the previous solves */
  eps->reason = EPS_CONVERGED_ITERATING;

  if (!eps->setupcalled){ ierr = EPSSetUp(eps);CHKERRQ(ierr); }
  ierr = STResetNumberLinearIterations(eps->OP);
  eps->evecsavailable = PETSC_FALSE;
  ierr = PetscLogEventBegin(EPS_Solve,eps,eps->V[0],eps->V[0],0);CHKERRQ(ierr);
  ierr = STPreSolve(eps->OP,eps);CHKERRQ(ierr);
  ierr = (*eps->ops->solve)(eps);CHKERRQ(ierr);
  ierr = STPostSolve(eps->OP,eps);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Solve,eps,eps->V[0],eps->V[0],0);CHKERRQ(ierr);
  if (!eps->reason) {
    SETERRQ(1,"Internal error, solver returned without setting converged reason");
  }

  /* Map eigenvalues back to the original problem, necessary in some 
  * spectral transformations */
  ierr = (*eps->ops->backtransform)(eps);CHKERRQ(ierr);

#ifndef PETSC_USE_COMPLEX
  /* reorder conjugate eigenvalues (positive imaginary first) */
  for (i=0; i<eps->nconv-1; i++) {
    PetscScalar minus = -1.0;
    if (eps->eigi[i] != 0) {
      if (eps->eigi[i] < 0) {
        eps->eigi[i] = -eps->eigi[i];
        eps->eigi[i+1] = -eps->eigi[i+1];
        ierr = VecScale(&minus, eps->V[i+1]); CHKERRQ(ierr);
      }
      i++;
    }
  }
#endif

  /* sort eigenvalues according to eps->which parameter */
  if (eps->perm) {
    ierr = PetscFree(eps->perm); CHKERRQ(ierr);
    eps->perm = PETSC_NULL;
  }
  if (eps->nconv > 0) {
    ierr = PetscMalloc(sizeof(int)*eps->nconv, &eps->perm); CHKERRQ(ierr);
    ierr = EPSSortEigenvalues(eps->nconv, eps->eigr, eps->eigi, eps->which, eps->nconv, eps->perm); CHKERRQ(ierr);
  }

  ierr = PetscOptionsHasName(eps->prefix,"-eps_view",&flg);CHKERRQ(ierr); 
  if (flg && !PetscPreLoadingOn) { ierr = EPSView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); }

  ierr = PetscOptionsHasName(eps->prefix,"-eps_plot_eigs",&flg);CHKERRQ(ierr); 
  if (flg) { 
    ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Computed Eigenvalues",
                             PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
    for( i=0; i<eps->nconv; i++ ) {
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
    ierr = PetscDrawSPDestroy(drawsp);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }

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

   Notes:
      During the i-th iteration this call returns i-1. If EPSSolve() is 
      complete, then parameter "its" contains either the iteration number at
      which convergence was successfully reached, or failure was detected.  
      Call EPSGetConvergedReason() to determine if the solver converged or 
      failed and why.

@*/
PetscErrorCode EPSGetIterationNumber(EPS eps,int *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidIntPointer(its,2);
  *its = eps->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetNumberLinearIterations"
/*@
   EPSGetNumberLinearIterations - Gets the total number of iterations
   required by the linear solves associated to the ST object during the 
   last EPSSolve() call.

   Not Collective

   Input Parameter:
.  eps - EPS context

   Output Parameter:
.  lits - number of linear iterations

   Notes:
   When the eigensolver algorithm invokes STApply() then a linear system 
   must be solved (except in the case of standard eigenproblems and shift
   transformation). The number of iterations required in this solve is
   accumulated into a counter whose value is returned by this function.

   The iteration counter is reset to zero at each successive call to EPSSolve().

   Level: intermediate

@*/
PetscErrorCode EPSGetNumberLinearIterations(EPS eps,int* lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidIntPointer(lits,2);
  STGetNumberLinearIterations(eps->OP, lits);
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

.seealso: EPSSetDimensions()
@*/
PetscErrorCode EPSGetConverged(EPS eps,int *nconv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (nconv) *nconv = eps->nconv;
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
   (see EPSConvergedReason)

   Possible values for reason:
+  EPS_CONVERGED_TOL - converged up to tolerance
.  EPS_DIVERGED_ITS - required more than its to reach convergence
.  EPS_DIVERGED_BREAKDOWN - generic breakdown in method
-  EPS_DIVERGED_NONSYMMETRIC - The operator is nonsymmetric

   Level: intermediate

   Notes: Can only be called after the call to EPSSolve() is complete.

.seealso: EPSSetTolerances(), EPSSolve(), EPSConvergedReason
@*/
PetscErrorCode EPSGetConvergedReason(EPS eps,EPSConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *reason = eps->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetInvariantSubspace" 
/*@
   EPSGetInvariantSubspace - Gets a basis of the computed invariant subspace.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context
  
   Output Parameter:
.  v - an array of vectors

   Notes:
   This function should be called after EPSSolve() has finished.

   The user should provide in v an array of nconv vectors, where nconv is
   the value returned by EPSGetConverged().

   The vectors returned in v span an invariant subspace associated with the
   (nconv) computed eigenvalues. An invariant subspace X of A satisfies Ax 
   in X for all x in X (a similar definition applies for generalized 
   eigenproblems). 

   Level: intermediate

.seealso: EPSGetEigenpair(), EPSGetConverged(), EPSSolve()
@*/
PetscErrorCode EPSGetInvariantSubspace(EPS eps, Vec *v)
{
  PetscErrorCode ierr;
  int            i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(v,VEC_COOKIE,3);
  if (!eps->V) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "EPSSolve must be called first"); 
  }
  for (i=0;i<eps->nconv;i++) {
    ierr = VecCopy(eps->V[i],v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetEigenpair" 
/*@
   EPSGetEigenpair - Gets the i-th solution of the eigenproblem 
   as computed by EPSSolve(). The solution consists in both the eigenvalue
   and the eigenvector (if available).

   Not Collective

   Input Parameters:
+  eps - eigensolver context 
-  i   - index of the solution

   Output Parameters:
+  eigr - real part of eigenvalue
.  eigi - imaginary part of eigenvalue
.  Vr   - real part of eigenvector
-  Vi   - imaginary part of eigenvector

   Notes:
   If the eigenvalue is real, then eigi and Vi are set to zero. In the 
   complex case (e.g. with BOPT=O_complex) the eigenvalue is stored 
   directly in eigr (eigi is set to zero) and the eigenvector Vr (Vi is 
   set to zero).

   The index i should be a value between 0 and nconv (see EPSGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established 
   with EPSSetWhichEigenpairs().

   Level: beginner

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs(), 
          EPSGetInvariantSubspace()
@*/
PetscErrorCode EPSGetEigenpair(EPS eps, int i, PetscScalar *eigr, PetscScalar *eigi, Vec Vr, Vec Vi)
{
  PetscErrorCode ierr;
  int            k;
  PetscScalar    zero = 0.0;
#ifndef PETSC_USE_COMPLEX
  PetscScalar    minus = -1.0;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (!eps->eigr || !eps->eigi || !eps->V) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "EPSSolve must be called first"); 
  }
  if (i<0 || i>=eps->nconv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }
  if (!eps->evecsavailable && (Vr || Vi) ) { 
    ierr = (*eps->ops->computevectors)(eps);CHKERRQ(ierr);
  }  

  if (!eps->perm) k = i;
  else k = eps->perm[i];
#ifdef PETSC_USE_COMPLEX
  if (eigr) *eigr = eps->eigr[k];
  if (eigi) *eigi = 0;
  if (Vr) { ierr = VecCopy(eps->AV[k], Vr); CHKERRQ(ierr); }
  if (Vi) { ierr = VecSet(&zero, Vi); CHKERRQ(ierr); }
#else
  if (eigr) *eigr = eps->eigr[k];
  if (eigi) *eigi = eps->eigi[k];
  if (eps->eigi[k] > 0) { /* first value of conjugate pair */
    if (Vr) { ierr = VecCopy(eps->AV[k], Vr); CHKERRQ(ierr); }
    if (Vi) { ierr = VecCopy(eps->AV[k+1], Vi); CHKERRQ(ierr); }
  } else if (eps->eigi[k] < 0) { /* second value of conjugate pair */
    if (Vr) { ierr = VecCopy(eps->AV[k-1], Vr); CHKERRQ(ierr); }
    if (Vi) { 
      ierr = VecCopy(eps->AV[k], Vi); CHKERRQ(ierr); 
      ierr = VecScale(&minus, Vi); CHKERRQ(ierr); 
    }
  } else { /* real eigenvalue */
    if (Vr) { ierr = VecCopy(eps->AV[k], Vr); CHKERRQ(ierr); }
    if (Vi) { ierr = VecSet(&zero, Vi); CHKERRQ(ierr); }
  }
#endif
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetErrorEstimate" 
/*@
   EPSGetErrorEstimate - Returns the error bound associated to the i-th 
   approximate eigenpair.

   Not Collective

   Input Parameter:
+  eps - eigensolver context 
-  i   - index of eigenpair

   Output Parameter:
.  errest - the error estimate

   Level: advanced

.seealso: EPSComputeRelativeError()
@*/
PetscErrorCode EPSGetErrorEstimate(EPS eps, int i, PetscReal *errest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (!eps->eigr || !eps->eigi) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "EPSSolve must be called first"); 
  }
  if (i<0 || i>=eps->nconv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }
  if (eps->perm) i = eps->perm[i];  
  if (errest) *errest = eps->errest[i];
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "EPSComputeResidualNorm"
/*@
   EPSComputeResidualNorm - Computes the residual norm associated with 
   the i-th converged approximate eigenpair.

   Collective on EPS

   Input Parameter:
.  eps - the eigensolver context
.  i   - the solution index

   Output Parameter:
.  norm - the residual norm, computed as ||Ax-kBx|| where k is the 
   eigenvalue and x is the eigenvector. 
   If k=0 then the residual norm is computed as ||Ax||.

   Notes:
   The index i should be a value between 0 and nconv (see EPSGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established 
   with EPSSetWhichEigenpairs().

   Level: beginner

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs()
@*/
PetscErrorCode EPSComputeResidualNorm(EPS eps, int i, PetscReal *norm)
{
  PetscErrorCode ierr;
  Vec            u, v, w, xr, xi;
  Mat            A, B;
  PetscScalar    alpha, kr, ki;
#ifndef PETSC_USE_COMPLEX
  PetscReal      ni, nr;
#endif
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&v); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&w); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&xr); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&xi); CHKERRQ(ierr);
  ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi); CHKERRQ(ierr);

#ifndef PETSC_USE_COMPLEX
  if (ki == 0 || 
    PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    ierr = MatMult( A, xr, u ); CHKERRQ(ierr); /* u=A*x */
    if (PetscAbsScalar(kr) > PETSC_MACHINE_EPSILON) {
      if (eps->isgeneralized) { ierr = MatMult( B, xr, w ); CHKERRQ(ierr); }
      else { ierr = VecCopy( xr, w ); CHKERRQ(ierr); } /* w=B*x */
      alpha = -kr; 
      ierr = VecAXPY( &alpha, w, u ); CHKERRQ(ierr); /* u=A*x-k*B*x */
    }
    ierr = VecNorm( u, NORM_2, norm); CHKERRQ(ierr);  
#ifndef PETSC_USE_COMPLEX
  } else {
    ierr = MatMult( A, xr, u ); CHKERRQ(ierr); /* u=A*xr */
    if (eps->isgeneralized) { ierr = MatMult( B, xr, v ); CHKERRQ(ierr); }
    else { ierr = VecCopy( xr, v ); CHKERRQ(ierr); } /* v=B*xr */
    alpha = -kr;
    ierr = VecAXPY( &alpha, v, u ); CHKERRQ(ierr); /* u=A*xr-kr*B*xr */
    if (eps->isgeneralized) { ierr = MatMult( B, xi, w ); CHKERRQ(ierr); }
    else { ierr = VecCopy( xi, w ); CHKERRQ(ierr); } /* w=B*xi */
    alpha = ki;
    ierr = VecAXPY( &alpha, w, u ); CHKERRQ(ierr); /* u=A*xr-kr*B*xr+ki*B*xi */
    ierr = VecNorm( u, NORM_2, &nr ); CHKERRQ(ierr);
    ierr = MatMult( A, xi, u ); CHKERRQ(ierr); /* u=A*xi */
    alpha = -kr;
    ierr = VecAXPY( &alpha, w, u ); CHKERRQ(ierr); /* u=A*xi-kr*B*xi */
    alpha = -ki;
    ierr = VecAXPY( &alpha, v, u ); CHKERRQ(ierr); /* u=A*xi-kr*B*xi-ki*B*xr */
    ierr = VecNorm( u, NORM_2, &ni ); CHKERRQ(ierr);
    *norm = SlepcAbsEigenvalue( nr, ni );
  }
#endif

  ierr = VecDestroy(w); CHKERRQ(ierr);
  ierr = VecDestroy(v); CHKERRQ(ierr);
  ierr = VecDestroy(u); CHKERRQ(ierr);
  ierr = VecDestroy(xr); CHKERRQ(ierr);
  ierr = VecDestroy(xi); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeRelativeError"
/*@
   EPSComputeRelativeError - Computes the actual relative error associated 
   with the i-th converged approximate eigenpair.

   Collective on EPS

   Input Parameter:
.  eps - the eigensolver context
.  i   - the solution index

   Output Parameter:
.  error - the relative error, computed as ||Ax-kBx||/||kx|| where k is the 
   eigenvalue and x is the eigenvector. 
   If k=0 the relative error is computed as ||Ax||/||x||.

   Level: beginner

.seealso: EPSSolve(), EPSComputeResidualNorm()
@*/
PetscErrorCode EPSComputeRelativeError(EPS eps, int i, PetscReal *error)
{
  PetscErrorCode ierr;
  Vec            xr, xi;  
  PetscScalar    kr, ki;  
  PetscReal      norm, er;
#ifndef PETSC_USE_COMPLEX
  Vec            u;
  PetscScalar    alpha;
  PetscReal      ei;
#endif
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);  
  ierr = EPSComputeResidualNorm(eps,i,&norm); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&xr); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&xi); CHKERRQ(ierr);
  ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi); CHKERRQ(ierr);

#ifndef PETSC_USE_COMPLEX
  if (ki == 0 || 
    PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    if (PetscAbsScalar(kr) > PETSC_MACHINE_EPSILON) {
      ierr = VecScale(&kr, xr); CHKERRQ(ierr);
    }
    ierr = VecNorm(xr, NORM_2, &er); CHKERRQ(ierr);
    *error = norm / er; 
#ifndef PETSC_USE_COMPLEX
  } else {
    ierr = VecDuplicate(xi, &u); CHKERRQ(ierr);  
    ierr = VecCopy(xi, u); CHKERRQ(ierr);  
    alpha = -ki;
    ierr = VecAXPBY(&kr, &alpha, xr, u); CHKERRQ(ierr);   
    ierr = VecNorm(u, NORM_2, &er); CHKERRQ(ierr);  
    ierr = VecAXPBY(&kr, &ki, xr, xi);  CHKERRQ(ierr);      
    ierr = VecNorm(xi, NORM_2, &ei); CHKERRQ(ierr);  
    ierr = VecDestroy(u); CHKERRQ(ierr);  
    *error = norm / SlepcAbsEigenvalue(er, ei);
  }
#endif    
  
  ierr = VecDestroy(xr); CHKERRQ(ierr);
  ierr = VecDestroy(xi); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSReverseProjection"
/*@
   EPSReverseProjection - Compute the operation V=V*S, where the columns of
   V are m of the basis vectors of the EPS object and S is an mxm dense
   matrix.

   Collective on EPS

   Input Parameter:
+  eps - the eigenproblem solver context
.  V - basis vectors
.  S - pointer to the values of matrix S
.  k - starting column
.  m - dimension of matrix S
-  work - workarea of m vectors for intermediate results

   Level: developer

@*/
PetscErrorCode EPSReverseProjection(EPS eps,Vec* V,PetscScalar *S,int k,int m,Vec* work)
{
  PetscErrorCode ierr;
  int            i;
  PetscScalar    zero = 0.0;
  
  PetscFunctionBegin;
  for (i=k;i<m;i++) {
    ierr = VecSet(&zero,work[i]);CHKERRQ(ierr);
    ierr = VecMAXPY(m,S+m*i,work[i],V);CHKERRQ(ierr);
  }    
  for (i=k;i<m;i++) {
    ierr = VecCopy(work[i],V[i]);CHKERRQ(ierr);
  }    
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeExplicitOperator"
/*@
    EPSComputeExplicitOperator - Computes the explicit operator associated
    to the eigenvalue problem with the specified spectral transformation.  

    Collective on EPS

    Input Parameter:
.   eps - the eigenvalue solver context

    Output Parameter:
.   mat - the explicit operator

    Notes:
    This routine builds a matrix containing the explicit operator. For 
    example, in generalized problems with shift-and-invert spectral
    transformation the result would be matrix (A - s B)^-1 B.
    
    This computation is done by applying the operator to columns of the 
    identity matrix.

    Currently, this routine uses a dense matrix format when 1 processor
    is used and a sparse format otherwise.  This routine is costly in general,
    and is recommended for use only with relatively small systems.

    Level: advanced

.seealso: STApply()   
@*/
PetscErrorCode EPSComputeExplicitOperator(EPS eps,Mat *mat)
{
  PetscErrorCode ierr;
  Vec            in,out;
  int            i,M,m,size,*rows,start,end;
  MPI_Comm       comm;
  PetscScalar    *array,zero = 0.0,one = 1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(mat,2);
  comm = eps->comm;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = VecDuplicate(eps->vec_initial,&in);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&out);CHKERRQ(ierr);
  ierr = VecGetSize(in,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(in,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(in,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(int),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) {rows[i] = start + i;}

  if (size == 1) {
    ierr = MatCreateSeqDense(comm,M,M,PETSC_NULL,mat);CHKERRQ(ierr);
  } else {
    ierr = MatCreateMPIAIJ(comm,m,m,M,M,0,0,0,0,mat);CHKERRQ(ierr);
  }
  
  for (i=0; i<M; i++) {

    ierr = VecSet(&zero,in);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    ierr = STApply(eps->OP,in,out); CHKERRQ(ierr);
    
    ierr = VecGetArray(out,&array);CHKERRQ(ierr);
    ierr = MatSetValues(*mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = VecRestoreArray(out,&array);CHKERRQ(ierr);

  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(in);CHKERRQ(ierr);
  ierr = VecDestroy(out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
