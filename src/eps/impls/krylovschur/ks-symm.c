/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur

   Algorithm:

       Single-vector Krylov-Schur method for both symmetric and non-symmetric
       problems.

   References:

       [1] "Krylov-Schur Methods in SLEPc", SLEPc Technical Report STR-7, 
           available at http://www.grycap.upv.es/slepc.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Analysis and App., 23(3), pp. 601-614, 2001. 

   Last update: Oct 2006

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSProjectedKSSym"
/*
   EPSProjectedKSSym - Solves the projected eigenproblem in the Krylov-Schur
   method (symmetric case).

   On input:
     l is the number of vectors kept in previous restart (0 means first restart)
     S is the projected matrix (leading dimension is lds)

   On output:
     S is diagonal with diagonal elements (eigenvalues) sorted appropriately
     Q is the eigenvector matrix

   Workspace:
     work is workspace to store a working copy of Ritz values and the permutation 
          used for sorting values
*/
PetscErrorCode EPSProjectedKSSym(EPS eps,int l,PetscScalar *S,int lds,PetscScalar *Q,int n,PetscReal *work)
{
  PetscErrorCode ierr;
  int            i,j;
  PetscReal      *ritz = work;
  int            *perm = (int*)(work+n);

  PetscFunctionBegin;
  /* Reduce S to diagonal form, S <- Q S Q' */
  if (l==0) {
    ierr = EPSDenseTridiagonal(n,S+eps->nconv*(lds+1),lds,ritz,Q+eps->nconv*n);CHKERRQ(ierr);
  } else {
    ierr = EPSDenseHEP(n,S+eps->nconv*(lds+1),lds,ritz,Q+eps->nconv*n);CHKERRQ(ierr);
  }
  /* Sort the remaining columns of the Schur form */
  if (eps->which == EPS_SMALLEST_REAL) {
    for (i=0;i<n;i++)
      eps->eigr[i+eps->nconv] = ritz[i];
  } else {
#ifdef PETSC_USE_COMPLEX
    for (i=0;i<n;i++)
      eps->eigr[i+eps->nconv] = ritz[i];
    ierr = EPSSortEigenvalues(n,eps->eigr+eps->nconv,eps->eigi,eps->which,n,perm);CHKERRQ(ierr);
#else
    ierr = EPSSortEigenvalues(n,ritz,eps->eigi+eps->nconv,eps->which,n,perm);CHKERRQ(ierr);
#endif
    for (i=0;i<n;i++)
      eps->eigr[i+eps->nconv] = ritz[perm[i]];
    ierr = PetscMemcpy(S,Q+eps->nconv*n,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
    for (j=0;j<n;j++)
      for (i=0;i<n;i++)
        Q[(j+eps->nconv)*n+i] = S[perm[j]*n+i];
  }
  /* Rebuild S from eigr */
  for (i=eps->nconv;i<eps->nv;i++) {
    S[i*(eps->ncv+1)] = eps->eigr[i];
    for (j=i+1;j<eps->ncv;j++)
      S[i*eps->ncv+j] = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KRYLOVSCHUR_SYMM"
PetscErrorCode EPSSolve_KRYLOVSCHUR_SYMM(EPS eps)
{
  PetscErrorCode ierr;
  int            i,k,l,n,lwork;
  Vec            u=eps->work[1];
  PetscScalar    *S=eps->T,*Q;
  PetscReal      beta,*work;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  ierr = PetscMemzero(S,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  lwork = eps->ncv*sizeof(PetscReal) + eps->ncv*sizeof(int);
  ierr = PetscMalloc(lwork,&work);CHKERRQ(ierr);

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    eps->nv = eps->ncv;
    ierr = EPSBasicArnoldi(eps,PETSC_FALSE,S,eps->V,eps->nconv+l,&eps->nv,u,&beta,&breakdown);CHKERRQ(ierr);

    /* Solve projected problem and compute residual norm estimates */ 
    n = eps->nv-eps->nconv;
    ierr = EPSProjectedKSSym(eps,l,S,eps->ncv,Q,n,work);CHKERRQ(ierr);
    for (i=eps->nconv;i<eps->nv;i++)
      eps->errest[i] = beta*PetscAbsScalar(Q[(i+1)*n-1]) / PetscAbsScalar(eps->eigr[i]);

    /* Check convergence */
    k = eps->nconv;
    while (k<eps->nv && eps->errest[k]<eps->tol) k++;    
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else l = (eps->nv-k)/2;
           
    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Arnoldi factorization */
        PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%i norm=%g)\n",eps->its,beta);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          PetscInfo(eps,"Unable to generate more start vectors\n");
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        for (i=k;i<k+l;i++) {
          S[i*eps->ncv+k+l] = Q[(i+1)*n-1]*beta;
        }
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    for (i=eps->nconv;i<k+l;i++) {
      ierr = VecSet(eps->AV[i],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(eps->AV[i],n,Q+i*n,eps->V+eps->nconv);CHKERRQ(ierr);
    }
    for (i=eps->nconv;i<k+l;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }
    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecAXPBY(eps->V[k+l],1.0/beta,0.0,u);CHKERRQ(ierr);
    }
    eps->nconv = k;

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nv);
    
  } 

  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

