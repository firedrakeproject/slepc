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
#define __FUNCT__ "EPSTranslateHarmonic"
/*
   EPSTranslateHarmonic - Computes a translation of the Krylov decomposition
   in order to perform a harmonic projection.

   On input:
     S is the Rayleigh quotient (leading dimension is m)
     tau is the translation amount
     b is assumed to be beta*e_m^T

   On output:
     g = (B-sigma*eye(m))'\b
     S is updated as S + g*b'

   Workspace:
     work is workspace to store a working copy of S and the pivots (int 
     of length m)
*/
PetscErrorCode EPSTranslateHarmonic(PetscScalar *S,PetscInt m_,PetscScalar tau,PetscScalar beta,PetscScalar *g,PetscScalar *work)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) || defined(PETSC_MISSING_LAPACK_GETRS) 
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"GETRF,GETRS - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscBLASInt   info,one = 1,m=m_,i;
  PetscScalar    *B = work; 
  PetscBLASInt   *ipiv = (PetscBLASInt*)(work+m*m);

  PetscFunctionBegin;
  /* Copy S to workspace B */
  ierr = PetscMemcpy(B,S,m*m*sizeof(PetscScalar));CHKERRQ(ierr);
  /* Vector g initialy stores b */
  ierr = PetscMemzero(g,m*sizeof(PetscScalar));CHKERRQ(ierr);
  g[m-1] = beta;
 
  /* g = (B-sigma*eye(m))'\b */
  for (i=0;i<m;i++) 
    B[i+i*m] -= tau;
  LAPACKgetrf_(&m,&m,B,&m,ipiv,&info);
  if (info<0) SETERRQ(PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");
  ierr = PetscLogFlops(2*m*m*m/3);CHKERRQ(ierr);
  LAPACKgetrs_("C",&m,&one,B,&m,ipiv,g,&m,&info);
  if (info) SETERRQ(PETSC_ERR_LIB,"GETRS - Bad solve");
  ierr = PetscLogFlops(2*m*m-m);CHKERRQ(ierr);

  /* S = S + g*b' */
  for (i=0;i<m;i++) 
    S[i+(m-1)*m] = S[i+(m-1)*m] + g[i]*beta;

  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSRecoverHarmonic"
/*
   EPSRecoverHarmonic - Computes a translation of the truncated Krylov 
   decomposition in order to recover the original non-translated state

   On input:
     S is the truncated Rayleigh quotient (size n, leading dimension m)
     k and l indicate the active columns of S
     [U, u] is the basis of the Krylov subspace
     g is the vector computed in the original translation
     Q is the similarity transformation used to reduce to sorted Schur form

   On output:
     S is updated as S + g*b'
     u is re-orthonormalized with respect to U
     b is re-scaled
     g is destroyed

   Workspace:
     ghat is workspace to store a vector of length n
*/
PetscErrorCode EPSRecoverHarmonic(PetscScalar *S,PetscInt n_,PetscInt k,PetscInt l,PetscInt m_,PetscScalar *g,PetscScalar *Q,Vec *U,Vec u,PetscScalar *ghat)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscBLASInt   one=1,ncol=k+l,n=n_,m=m_;
  PetscScalar    done=1.0,dmone=-1.0,dzero=0.0;
  PetscReal      gamma,gnorm;
  PetscBLASInt   i,j;

  PetscFunctionBegin;

  /* g^ = -Q(:,idx)'*g */
  BLASgemv_("C",&n,&ncol,&dmone,Q,&m,g,&one,&dzero,ghat,&one);

  /* S = S + g^*b' */
  for (i=0;i<k+l;i++) {
    for (j=k;j<k+l;j++) {
      S[i+j*m] += ghat[i]*S[k+l+j*m];
    }
  }

  /* g~ = (I-Q(:,idx)*Q(:,idx)')*g = g+Q(:,idx)*g^ */
  BLASgemv_("N",&n,&ncol,&done,Q,&m,ghat,&one,&done,g,&one);

  /* gamma u^ = u - U*g~ */
  for (i=0;i<n;i++) 
    g[i] = -g[i];
  ierr = VecMAXPY(u,m,g,U);CHKERRQ(ierr);        

  /* Renormalize u */
  gnorm = 0.0;
  for (i=0;i<n;i++)
    gnorm = gnorm + PetscRealPart(g[i]*PetscConj(g[i]));
  gamma = sqrt(1.0+gnorm);
  ierr = VecScale(u,1.0/gamma);CHKERRQ(ierr);

  /* b = gamma*b */
  for (i=k;i<k+l;i++) {
    S[i*m+k+l] *= gamma;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSProjectedKSHarmonic"
/*
   EPSProjectedKSHarmonic - Solves the projected eigenproblem in the 
   Krylov-Schur method (non-symmetric harmonic case).

   On input:
     l is the number of vectors kept in previous restart (0 means first restart)
     S is the projected matrix (leading dimension is lds)

   On output:
     S has (real) Schur form with diagonal blocks sorted appropriately
     Q contains the corresponding Schur vectors (order n, leading dimension n)
*/
PetscErrorCode EPSProjectedKSHarmonic(EPS eps,PetscInt l,PetscScalar *S,PetscInt lds,PetscScalar *Q,PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Reduce S to Hessenberg form, S <- Q S Q' */
  ierr = EPSDenseHessenberg(n,eps->nconv,S,lds,Q);CHKERRQ(ierr);
  /* Reduce S to (quasi-)triangular form, S <- Q S Q' */
  ierr = EPSDenseSchur(n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi);CHKERRQ(ierr);
  /* Sort the remaining columns of the Schur form */
  ierr = EPSSortDenseSchurTarget(n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi,eps->target,eps->which);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KRYLOVSCHUR_HARMONIC"
PetscErrorCode EPSSolve_KRYLOVSCHUR_HARMONIC(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,n,lwork;
  Vec            u=eps->work[1];
  PetscScalar    *S=eps->T,*Q,*g,*work;
  PetscReal      beta,gnorm;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  ierr = PetscMemzero(S,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  lwork = (eps->ncv+4)*eps->ncv;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&g);CHKERRQ(ierr);

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

    /* Compute translation of Krylov decomposition if harmonic projection used */ 
    ierr = EPSTranslateHarmonic(S,eps->ncv,eps->target,(PetscScalar)beta,g,work);CHKERRQ(ierr);

    /* Solve projected problem and compute residual norm estimates */ 
    n = eps->nv;
    ierr = EPSProjectedKSHarmonic(eps,l,S,eps->ncv,Q,n);CHKERRQ(ierr);
    ierr = ArnoldiResiduals(S,eps->ncv,Q,beta,eps->nconv,n,eps->eigr,eps->eigi,eps->errest,work);CHKERRQ(ierr);

    /* Fix residual norms */
    gnorm = 0.0;
    for (i=0;i<eps->nv;i++)
      gnorm = gnorm + PetscRealPart(g[i]*PetscConj(g[i]));
    for (i=eps->nconv;i<eps->nv;i++)
      eps->errest[i] *= sqrt(1.0+gnorm);

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
        ierr = EPSRecoverHarmonic(S,n,k,l,eps->ncv,g,Q,eps->V,u,work);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    for (i=eps->nconv;i<k+l;i++) {
      ierr = VecSet(eps->AV[i],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(eps->AV[i],n,Q+i*n,eps->V);CHKERRQ(ierr);        
    }
    for (i=eps->nconv;i<k+l;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecCopy(u,eps->V[k+l]);CHKERRQ(ierr);
    }
    eps->nconv = k;

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nv);
    
  } 

  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

