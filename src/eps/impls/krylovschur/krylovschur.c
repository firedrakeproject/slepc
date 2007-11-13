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
#define __FUNCT__ "EPSSetUp_KRYLOVSCHUR"
PetscErrorCode EPSSetUp_KRYLOVSCHUR(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev+1) SETERRQ(1,"The value of ncv must be at least nev+1"); 
  }
  else eps->ncv = PetscMin(N,PetscMax(2*eps->nev,eps->nev+15));
  if (!eps->max_it) eps->max_it = PetscMax(100,2*N/eps->ncv);
  if (eps->ishermitian && (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY))
    SETERRQ(1,"Wrong value of eps->which");

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSProjectedKSSymm"
/*
   EPSProjectedKSSym - Solves the projected eigenproblem in the Krylov-Schur
   method (symmetric case).

   On input:
     l is the number of vectors kept in previous restart (0 means first restart)
     S is the projected matrix (leading dimension is lds)
     Q is an orthogonal transformation matrix if l=0 (leading dimension is n)

   Workspace:
     ritz temporarily stores computed Ritz values
     perm is used for representing the permutation used for sorting values

   On output:
     S has (real) Schur form with diagonal blocks sorted appropriately
     Q contains the accumulated orthogonal transformations used in the process
*/
PetscErrorCode EPSProjectedKSSym(EPS eps,int l,PetscScalar *S,int lds,PetscScalar *Q,int n,PetscReal *ritz,int *perm)
{
  PetscErrorCode ierr;
  int            i,j;

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
  /* rebuild S from eigr */
  for (i=eps->nconv;i<eps->nv;i++) {
    S[i*(eps->ncv+1)] = eps->eigr[i];
    for (j=i+1;j<eps->ncv;j++)
      S[i*eps->ncv+j] = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSProjectedKSNonsymm"
/*
   EPSProjectedKSNonsym - Solves the projected eigenproblem in the Krylov-Schur
   method (non-symmetric case).

   On input:
     l is the number of vectors kept in previous restart (0 means first restart)
     S is the projected matrix (leading dimension is lds)
     Q is an orthogonal transformation matrix if l=0 (leading dimension is n)

   On output:
     S has (real) Schur form with diagonal blocks sorted appropriately
     Q contains the accumulated orthogonal transformations used in the process
*/
PetscErrorCode EPSProjectedKSNonsym(EPS eps,int l,PetscScalar *S,int lds,PetscScalar *Q,int n)
{
  PetscErrorCode ierr;
  int            i;

  PetscFunctionBegin;
  if (l==0) {
    ierr = PetscMemzero(Q,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<n;i++) 
      Q[i*(n+1)] = 1.0;
  } else {
    /* Reduce S to Hessenberg form, S <- Q S Q' */
    ierr = EPSDenseHessenberg(n,eps->nconv,S,lds,Q);CHKERRQ(ierr);
  }
  /* Reduce S to (quasi-)triangular form, S <- Q S Q' */
  ierr = EPSDenseSchur(n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi);CHKERRQ(ierr);
  /* Sort the remaining columns of the Schur form */
  ierr = EPSSortDenseSchur(n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi,eps->which);CHKERRQ(ierr);    
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KRYLOVSCHUR"
PetscErrorCode EPSSolve_KRYLOVSCHUR(EPS eps)
{
  PetscErrorCode ierr;
  int            i,k,l,n,lwork,*perm;
  Vec            u=eps->work[1];
  PetscScalar    *S=eps->T,*Q,*work,*b;
  PetscReal      beta,*ritz;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  ierr = PetscMemzero(S,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&b);CHKERRQ(ierr);
  lwork = (eps->ncv+4)*eps->ncv;
  if (!eps->ishermitian) {
    ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(int),&perm);CHKERRQ(ierr);
  }

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    eps->nv = eps->ncv;
    ierr = EPSBasicArnoldi(eps,PETSC_FALSE,S,eps->V,eps->nconv+l,&eps->nv,u,&beta,&breakdown);CHKERRQ(ierr);
    ierr = VecScale(u,1.0/beta);CHKERRQ(ierr);

    /* Solve projected problem and compute residual norm estimates */ 
    if (eps->ishermitian) {
      n = eps->nv-eps->nconv;
      ierr = EPSProjectedKSSym(eps,l,S,eps->ncv,Q,n,ritz,perm);CHKERRQ(ierr);
      for (i=eps->nconv;i<eps->nv;i++)
        eps->errest[i] = beta*PetscAbsScalar(Q[(i+1)*n-1]) / PetscAbsScalar(eps->eigr[i]);
    } else { /* non-hermitian */
      n = eps->nv;
      ierr = EPSProjectedKSNonsym(eps,l,S,eps->ncv,Q,n);CHKERRQ(ierr);
      ierr = ArnoldiResiduals(S,eps->ncv,Q,beta,eps->nconv,n,eps->eigr,eps->eigi,eps->errest,work);CHKERRQ(ierr);
    }

    /* Check convergence */
    k = eps->nconv;
    while (k<eps->nv && eps->errest[k]<eps->tol) k++;    
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = (eps->nv-k)/2;
#if !defined(PETSC_USE_COMPLEX)
      if (S[(k+l-1)*(eps->ncv+1)+1] != 0.0) {
        if (k+l<eps->nv-1) l = l+1;
	else l = l-1;
      }
#endif
    }
           
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    for (i=eps->nconv;i<k+l;i++) {
      ierr = VecSet(eps->AV[i],0.0);CHKERRQ(ierr);
      if (!eps->ishermitian) {
        ierr = VecMAXPY(eps->AV[i],n,Q+i*n,eps->V);CHKERRQ(ierr);        
      } else {
        ierr = VecMAXPY(eps->AV[i],n,Q+i*n,eps->V+eps->nconv);CHKERRQ(ierr);
      }
    }
    for (i=eps->nconv;i<k+l;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }
    eps->nconv = k;

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nv);
    
    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
	/* start a new Arnoldi factorization */
	PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%i norm=%g)\n",eps->its,beta);
	ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
	if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
	  PetscInfo(eps,"Unable to generate more start vectors\n");
	}
      } else {
        /* update the Arnoldi-Schur decomposition */
	for (i=k;i<k+l;i++) {
          S[i*eps->ncv+k+l] = Q[(i+1)*n-1]*beta;
	}
	ierr = VecCopy(u,eps->V[k+l]);CHKERRQ(ierr);
      }
    }
  } 

  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  if (!eps->ishermitian) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(ritz);CHKERRQ(ierr);
    ierr = PetscFree(perm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_KRYLOVSCHUR"
PetscErrorCode EPSCreate_KRYLOVSCHUR(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = PETSC_NULL;
  eps->ops->solve                = EPSSolve_KRYLOVSCHUR;
  eps->ops->solvets              = PETSC_NULL;
  eps->ops->setup                = EPSSetUp_KRYLOVSCHUR;
  eps->ops->setfromoptions       = PETSC_NULL;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = PETSC_NULL;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
EXTERN_C_END

