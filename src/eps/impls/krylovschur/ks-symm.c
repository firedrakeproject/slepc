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

#include "private/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSProjectedKSSym"
/*
   EPSProjectedKSSym - Solves the projected eigenproblem in the Krylov-Schur
   method (symmetric case).

   On input:
     l is the number of vectors kept in previous restart (0 means first restart)
     S is the projected matrix (order n, leading dimension is lds)

   On output:
     S is diagonal with diagonal elements (eigenvalues) sorted appropriately
     eig is the sorted list of eigenvalues
     Q is the eigenvector matrix (order n)

   Workspace:
     work is workspace to store 2n reals and 2n integers
*/
PetscErrorCode EPSProjectedKSSym(EPS eps,PetscInt n,PetscInt l,PetscScalar *S,PetscInt lds,PetscScalar *eig,PetscScalar *Q,PetscReal *work)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      *ritz = work;
  PetscReal      *worksort = work+n;
  PetscInt       *perm = ((PetscInt*)(work+n))+n;

  PetscFunctionBegin;

  /* Compute eigendecomposition of S, S <- Q S Q' */
  if (l==0) {
    ierr = EPSDenseTridiagonal(n,S,lds,ritz,Q);CHKERRQ(ierr);
  } else {
    ierr = EPSDenseHEP(n,S,lds,ritz,Q);CHKERRQ(ierr);
  }

  /* Sort eigendecomposition according to eps->which */
  ierr = EPSSortEigenvaluesReal(n,ritz,eps->which,n,perm,worksort);CHKERRQ(ierr);
  for (i=0;i<n;i++)
    eig[i] = ritz[perm[i]];
  for (j=0;j<n;j++)
    for (i=0;i<n;i++)
      S[i+j*lds] = Q[i+j*n];
  for (j=0;j<n;j++)
    for (i=0;i<n;i++)
      Q[i+j*n] = S[i+perm[j]*lds];

  /* Rebuild S from eig */
  for (i=0;i<n;i++) {
    S[i+i*lds] = eig[i];
    for (j=i+1;j<n;j++)
      S[j+i*lds] = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KRYLOVSCHUR_SYMM"
PetscErrorCode EPSSolve_KRYLOVSCHUR_SYMM(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,n,lwork;
  Vec            u=eps->work[1];
  PetscScalar    *S=eps->T,*Q;
  PetscReal      beta,*work;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  ierr = PetscMemzero(S,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  lwork = 2*eps->ncv*sizeof(PetscReal) + 2*eps->ncv*sizeof(PetscInt);
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
    ierr = EPSProjectedKSSym(eps,n,l,S+eps->nconv*(eps->ncv+1),eps->ncv,eps->eigr+eps->nconv,Q,work);CHKERRQ(ierr);
    for (i=eps->nconv;i<eps->nv;i++)
      eps->errest[i] = beta*PetscAbsScalar(Q[(i-eps->nconv+1)*n-1]) / PetscAbsScalar(eps->eigr[i]);

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
          S[i*eps->ncv+k+l] = Q[(i-eps->nconv+1)*n-1]*beta;
        }
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    for (i=eps->nconv;i<k+l;i++) {
      ierr = VecSet(eps->AV[i],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(eps->AV[i],n,Q+(i-eps->nconv)*n,eps->V+eps->nconv);CHKERRQ(ierr);
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

