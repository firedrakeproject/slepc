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

   Last update: Feb 2009

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

#include <private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

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
PetscErrorCode ArrowTridFlip(PetscInt n_,PetscInt l,PetscReal *d,PetscReal *e,PetscReal *Q,PetscReal *S)
{
#if defined(SLEPC_MISSING_LAPACK_SYTRD) || defined(SLEPC_MISSING_LAPACK_ORGTR) || defined(SLEPC_MISSING_LAPACK_STEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SYTRD/ORGTR/STEQR - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBLASInt   n,n1,n2,lwork,info;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  /* quick return */
  if (n == 1) {
    Q[0] = 1.0;
    PetscFunctionReturn(0);    
  }
  n1 = PetscBLASIntCast(l+1);    /* size of leading block, including residuals */
  n2 = PetscBLASIntCast(n-l-1);  /* size of trailing block */
  ierr = PetscMemzero(S,n*n*sizeof(PetscReal));CHKERRQ(ierr);

  /* Flip matrix S, copying the values saved in Q */
  for (i=0;i<n;i++) 
    S[(n-1-i)+(n-1-i)*n] = d[i];
  for (i=0;i<l;i++)
    S[(n-1-i)+(n-1-l)*n] = e[i];
  for (i=l;i<n-1;i++)
    S[(n-1-i)+(n-1-i-1)*n] = e[i];

  /* Reduce (2,2)-block of flipped S to tridiagonal form */
  lwork = PetscBLASIntCast(n_*n_-n_);
  LAPACKsytrd_("L",&n1,S+n2*(n+1),&n,d,e,Q,Q+n,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSYTRD %d",info);

  /* Flip back diag and subdiag, put them in d and e */
  for (i=0;i<n-1;i++) {
    d[n-i-1] = S[i+i*n];
    e[n-i-2] = S[i+1+i*n];
  }
  d[0] = S[n-1+(n-1)*n];

  /* Compute the orthogonal matrix used for tridiagonalization */
  LAPACKorgtr_("L",&n1,S+n2*(n+1),&n,Q,Q+n,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGTR %d",info);

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
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSTEQR %d",info);

  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSProjectedKSSym"
/*
   EPSProjectedKSSym - Solves the projected eigenproblem in the Krylov-Schur
   method (symmetric case).

   On input:
     n is the matrix dimension
     l is the number of vectors kept in previous restart
     a contains diagonal elements (length n)
     b contains offdiagonal elements (length n-1)

   On output:
     eig is the sorted list of eigenvalues
     Q is the eigenvector matrix (order n, leading dimension n)

   Workspace:
     work is workspace to store a real square matrix of order n
     perm is workspace to store 2n integers
*/
PetscErrorCode EPSProjectedKSSym(EPS eps,PetscInt n,PetscInt l,PetscReal *a,PetscReal *b,PetscScalar *eig,PetscScalar *Q,PetscReal *work,PetscInt *perm)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,p;
  PetscReal      rtmp,*Qreal = (PetscReal*)Q;

  PetscFunctionBegin;
  /* Compute eigendecomposition of projected matrix */
  ierr = ArrowTridFlip(n,l,a,b,Qreal,work);CHKERRQ(ierr);

  /* Sort eigendecomposition according to eps->which */
  ierr = EPSSortEigenvaluesReal(eps,n,a,perm);CHKERRQ(ierr);
  for (i=0;i<n;i++)
    eig[i] = a[perm[i]];
  for (i=0;i<n;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap eigenvectors i and j */
      for (k=0;k<n;k++) {
        rtmp = Qreal[k+p*n]; Qreal[k+p*n] = Qreal[k+i*n]; Qreal[k+i*n] = rtmp;
      }
    }
  }

#if defined(PETSC_USE_COMPLEX)
  for (j=n-1;j>=0;j--)
    for (i=n-1;i>=0;i--) 
      Q[i+j*n] = Qreal[i+j*n];
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KrylovSchur_Symm"
PetscErrorCode EPSSolve_KrylovSchur_Symm(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,lds,lt,nv,m,*iwork;
  Vec            u=eps->work[0];
  PetscScalar    *Q;
  PetscReal      *a,*b,*work,beta;
  PetscBool      breakdown;

  PetscFunctionBegin;
  lds = PetscMin(eps->mpd,eps->ncv);
  ierr = PetscMalloc(lds*lds*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(lds*lds*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(2*lds*sizeof(PetscInt),&iwork);CHKERRQ(ierr);
  lt = PetscMin(eps->nev+eps->mpd,eps->ncv);
  ierr = PetscMalloc(lt*sizeof(PetscReal),&a);CHKERRQ(ierr);  
  ierr = PetscMalloc(lt*sizeof(PetscReal),&b);CHKERRQ(ierr);  

  /* Get the starting Lanczos vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    m = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = EPSFullLanczos(eps,a+l,b+l,eps->V,eps->nconv+l,&m,u,&breakdown);CHKERRQ(ierr);
    nv = m - eps->nconv;
    beta = b[nv-1];

    /* Solve projected problem and compute residual norm estimates */ 
    ierr = EPSProjectedKSSym(eps,nv,l,a,b,eps->eigr+eps->nconv,Q,work,iwork);CHKERRQ(ierr);

    /* Check convergence */
    ierr = EPSKrylovConvergence(eps,PETSC_TRUE,eps->nconv,nv,PETSC_NULL,nv,Q,eps->V+eps->nconv,nv,beta,1.0,&k,PETSC_NULL);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else l = (eps->nconv+nv-k)/2;

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Lanczos factorization */
        PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%G)\n",eps->its,beta);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          PetscInfo(eps,"Unable to generate more start vectors\n");
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        for (i=0;i<l;i++) {
          a[i] = PetscRealPart(eps->eigr[i+k]);
          b[i] = PetscRealPart(Q[nv-1+(i+k-eps->nconv)*nv]*beta);
        }
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V+eps->nconv,0,k+l-eps->nconv,Q,nv,PETSC_FALSE);CHKERRQ(ierr);
    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecAXPBY(eps->V[k+l],1.0/beta,0.0,u);CHKERRQ(ierr);
    }

    ierr = EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv+eps->nconv);CHKERRQ(ierr);
    eps->nconv = k;
  } 
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

