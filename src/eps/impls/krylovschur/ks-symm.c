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
#define __FUNCT__ "ArrowTridFlip"
/*
   ArrowTridFlip - Solves the arrowhead-tridiagonal eigenproblem by flipping
   the matrix and tridiagonalizing the bottom part.

   On input:
     l is the size of diagonal part
     d contains diagonal elements (length n)
     e contains offdiagonal elements (length n-1)

   On output:
     d contains the eigenvalues in ascending order
     Q is the eigenvector matrix (order n)

   Workspace:
     S is workspace to store a copy of the full matrix (nxn reals)
*/
PetscErrorCode ArrowTridFlip(PetscInt n,PetscInt l,PetscReal *d,PetscReal *e,PetscReal *Q,PetscReal *S)
{
#if defined(SLEPC_MISSING_LAPACK_SYTRD) || defined(SLEPC_MISSING_LAPACK_ORGTR) || defined(SLEPC_MISSING_LAPACK_STEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"SYTRD/ORGTR/STEQR - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBLASInt   n1,n2,lwork,info;

  PetscFunctionBegin;

  n1 = l+1;    /* size of leading block, including residuals */
  n2 = n-l-1;  /* size of trailing block */
  ierr = PetscMemzero(S,n*n*sizeof(PetscReal));CHKERRQ(ierr);

  /* Flip matrix S, copying the values saved in Q */
  for (i=0;i<n;i++) 
    S[(n-1-i)+(n-1-i)*n] = d[i];
  for (i=0;i<l;i++)
    S[(n-1-i)+(n-1-l)*n] = e[i];
  for (i=l;i<n;i++)
    S[(n-1-i)+(n-1-i-1)*n] = e[i];

  /* Reduce (2,2)-block of flipped S to tridiagonal form */
  lwork = n*n-n;
  LAPACKsytrd_("L",&n1,S+n2*(n+1),&n,d,e,Q,Q+n,&lwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xSYTRD %d",info);

  /* Flip back diag and subdiag, put them in d and e */
  for (i=0;i<n-1;i++) {
    d[n-i-1] = S[i+i*n];
    e[n-i-2] = S[i+1+i*n];
  }
  d[0] = S[n-1+(n-1)*n];

  /* Compute the orthogonal matrix used for tridiagonalization */
  LAPACKorgtr_("L",&n1,S+n2*(n+1),&n,Q,Q+n,&lwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xORGTR %d",info);

  /* Create full-size Q, flipped back to original order */
  for (i=0;i<n;i++) 
    for (j=0;j<n;j++) 
      Q[i+j*n] = 0.0;
  for (i=n1;i<n;i++) 
    Q[i+i*n] = 1.0;
  for (i=0;i<n1;i++) 
    for (j=0;j<n1;j++) 
      Q[i+j*n] = S[n-i-1+(n-j-1)*n];

  /* Solve the tridiagonal eigenproblem */
  LAPACKsteqr_("V",&n,d,e,Q,&n,S,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xSTEQR %d",info);

  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSProjectedKSSym"
/*
   EPSProjectedKSSym - Solves the projected eigenproblem in the Krylov-Schur
   method (symmetric case).

   On input:
     l is the number of vectors kept in previous restart (0 means first restart)
     S is the projected matrix (order n, leading dimension is lds)

   On output:
     S is overwritten
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
  PetscReal      *e = work+n;
  PetscInt       *perm = ((PetscInt*)(work+n))+n;
  PetscReal      *Sreal = (PetscReal*)S, *Qreal = (PetscReal*)Q;

  PetscFunctionBegin;

  /* Compute eigendecomposition of S, S <- Q S Q' */
  for (i=0;i<n;i++)
    ritz[i] = S[i+i*lds];
  for (i=0;i<l;i++)
    e[i] = S[l+i*lds];
  for (i=l;i<n;i++) 
    e[i] = S[i+1+i*lds];
  ierr = ArrowTridFlip(n,l,ritz,e,Qreal,Sreal);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  for (j=n-1;j>=0;j--)
    for (i=n-1;i>=0;i--) 
      Q[i+j*n] = Qreal[i+j*n];
#endif

  /* Sort eigendecomposition according to eps->which */
  ierr = EPSSortEigenvaluesReal(n,ritz,eps->which,n,perm,e);CHKERRQ(ierr);
  for (i=0;i<n;i++)
    eig[i] = ritz[perm[i]];
  for (j=0;j<n;j++)
    for (i=0;i<n;i++)
      S[i+j*lds] = Q[i+j*n];
  for (j=0;j<n;j++)
    for (i=0;i<n;i++)
      Q[i+j*n] = S[i+perm[j]*lds];

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicLanczosKS"
/*
   EPSBasicLanczosKS - Computes an m-step Lanczos factorization. The first k
   columns are assumed to be locked and therefore they are not modified. On
   exit, the following relation is satisfied:

                    OP * V - V * H = f * e_m^T

   where the columns of V are the Lanczos vectors (which are B-orthonormal),
   H is an upper Hessenberg matrix, f is the residual vector and e_m is
   the m-th vector of the canonical basis. The vector f is B-orthogonal to
   the columns of V. On exit, beta contains the B-norm of f and the next 
   Lanczos vector can be computed as v_{m+1} = f / beta. 
*/
PetscErrorCode EPSBasicLanczosKS(EPS eps,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M;
  PetscReal      norm;
  PetscScalar    *swork,*Hwork;

  PetscFunctionBegin;
  if (m > 100) {
    ierr = PetscMalloc(m*sizeof(PetscScalar),&swork);CHKERRQ(ierr);
  } else swork = PETSC_NULL;
  ierr = PetscMalloc((eps->nds+eps->nconv+m)*sizeof(PetscScalar),&Hwork);CHKERRQ(ierr);
  
  for (j=eps->nconv+k;j<eps->nconv+m-1;j++) {
    ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,eps->nds+j+1,PETSC_NULL,eps->DSV,V[j+1],Hwork,&norm,breakdown,eps->work[0],swork);CHKERRQ(ierr);
    H[j-eps->nconv+(j-eps->nconv)*ldh] = Hwork[j+eps->nds]; /* beta */
    H[j-1-eps->nconv+(j-eps->nconv)*ldh] = Hwork[j-1+eps->nds]; /* alpha */
    H[j+1-eps->nconv+(j-eps->nconv)*ldh] = norm;
    if (*breakdown) {
      *M = j+1-eps->nconv;
      *beta = norm;
      if (swork) { ierr = PetscFree(swork);CHKERRQ(ierr); }
      PetscFunctionReturn(0);
    } else {
      ierr = VecScale(V[j+1],1/norm);CHKERRQ(ierr);
    }
  }
  ierr = STApply(eps->OP,V[eps->nconv+m-1],f);CHKERRQ(ierr);
  ierr = IPOrthogonalize(eps->ip,eps->nds+eps->nconv+m,PETSC_NULL,eps->DSV,f,Hwork,beta,PETSC_NULL,eps->work[0],swork);CHKERRQ(ierr);
  H[m-1+(m-1)*ldh] = Hwork[eps->nconv+m-1+eps->nds]; /* beta */
  H[m-2+(m-1)*ldh] = Hwork[eps->nconv+m-2+eps->nds]; /* alpha */
  if (m > 100) {
    ierr = PetscFree(swork);CHKERRQ(ierr);
  }
  ierr = PetscFree(Hwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KRYLOVSCHUR_SYMM"
PetscErrorCode EPSSolve_KRYLOVSCHUR_SYMM(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,lwork,lds,nv;
  Vec            u=eps->work[1];
  PetscScalar    *S=eps->T,*Q;
  PetscReal      beta,*work;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  lds = PetscMin(eps->nev+eps->mpd,eps->ncv);
  ierr = PetscMemzero(S,lds*lds*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(lds*lds*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  lwork = 2*lds*sizeof(PetscReal) + 2*lds*sizeof(PetscInt);
  ierr = PetscMalloc(lwork,&work);CHKERRQ(ierr);

  /* Get the starting Lanczos vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(eps->mpd,eps->ncv-eps->nconv);
    ierr = EPSBasicLanczosKS(eps,S,lds,eps->V,l,&nv,u,&beta,&breakdown);CHKERRQ(ierr);

    /* Solve projected problem and compute residual norm estimates */ 
    ierr = EPSProjectedKSSym(eps,nv,l,S,lds,eps->eigr+eps->nconv,Q,work);CHKERRQ(ierr);
    for (i=0;i<nv;i++)
      eps->errest[i+eps->nconv] = beta*PetscAbsScalar(Q[(i+1)*nv-1]) / PetscAbsScalar(eps->eigr[i+eps->nconv]);

    /* Check convergence */
    k = eps->nconv;
    while (k<eps->nconv+nv && eps->errest[k]<eps->tol) k++;    
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else l = (eps->nconv+nv-k)/2;

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Lanczos factorization */
        PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%i norm=%g)\n",eps->its,beta);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          PetscInfo(eps,"Unable to generate more start vectors\n");
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        for (i=0;i<l;i++) {
          S[i+i*lds] = eps->eigr[i+k];
          S[l+i*lds] = Q[nv-1+(i+k-eps->nconv)*nv]*beta;
        }
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V+eps->nconv,0,k+l-eps->nconv,Q,nv,PETSC_FALSE);CHKERRQ(ierr);
    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecAXPBY(eps->V[k+l],1.0/beta,0.0,u);CHKERRQ(ierr);
    }

    EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv+eps->nconv);
    eps->nconv = k;
    
  } 

  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

