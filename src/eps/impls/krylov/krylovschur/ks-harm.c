/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur with harmonic extraction

   References:

       [1] "Practical Implementation of Harmonic Krylov-Schur", SLEPc Technical
            Report STR-9, available at http://www.grycap.upv.es/slepc.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

#undef __FUNCT__  
#define __FUNCT__ "EPSTranslateHarmonic"
/*
   EPSTranslateHarmonic - Computes a translation of the Krylov decomposition
   in order to perform a harmonic extraction.

   On input:
     S is the Rayleigh quotient (order m, leading dimension is lds)
     tau is the translation amount
     b is assumed to be beta*e_m^T

   On output:
     g = (B-sigma*eye(m))'\b
     S is updated as S + g*b'

   Workspace:
     work is workspace to store a working copy of S and the pivots (int 
     of length m)
*/
PetscErrorCode EPSTranslateHarmonic(PetscInt m_,PetscScalar *S,PetscInt lds,PetscScalar tau,PetscScalar beta,PetscScalar *g,PetscScalar *work)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) || defined(PETSC_MISSING_LAPACK_GETRS) 
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRF,GETRS - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBLASInt   info,m,one = 1;
  PetscScalar    *B = work; 
  PetscBLASInt   *ipiv = (PetscBLASInt*)(work+m_*m_);

  PetscFunctionBegin;
  m = PetscBLASIntCast(m_);
  /* Copy S to workspace B */
  for (i=0;i<m;i++) 
    for (j=0;j<m;j++) 
      B[i+j*m] = S[i+j*lds];
  /* Vector g initialy stores b */
  ierr = PetscMemzero(g,m*sizeof(PetscScalar));CHKERRQ(ierr);
  g[m-1] = beta;
 
  /* g = (B-sigma*eye(m))'\b */
  for (i=0;i<m;i++) 
    B[i+i*m] -= tau;
  LAPACKgetrf_(&m,&m,B,&m,ipiv,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");
  ierr = PetscLogFlops(2.0*m*m*m/3.0);CHKERRQ(ierr);
  LAPACKgetrs_("C",&m,&one,B,&m,ipiv,g,&m,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");
  ierr = PetscLogFlops(2.0*m*m-m);CHKERRQ(ierr);

  /* S = S + g*b' */
  for (i=0;i<m;i++) 
    S[i+(m-1)*lds] = S[i+(m-1)*lds] + g[i]*beta;
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
  PetscErrorCode ierr;
  PetscBLASInt   one=1,ncol=k+l,n,m;
  PetscScalar    done=1.0,dmone=-1.0,dzero=0.0;
  PetscReal      gamma,gnorm;
  PetscBLASInt   i,j;

  PetscFunctionBegin;
  n = PetscBLASIntCast(n_);
  m = PetscBLASIntCast(m_);

  /* g^ = -Q(:,idx)'*g */
  BLASgemv_("C",&n,&ncol,&dmone,Q,&n,g,&one,&dzero,ghat,&one);

  /* S = S + g^*b' */
  for (i=0;i<k+l;i++) {
    for (j=k;j<k+l;j++) {
      S[i+j*m] += ghat[i]*S[k+l+j*m];
    }
  }

  /* g~ = (I-Q(:,idx)*Q(:,idx)')*g = g+Q(:,idx)*g^ */
  BLASgemv_("N",&n,&ncol,&done,Q,&n,ghat,&one,&done,g,&one);

  /* gamma u^ = u - U*g~ */
  ierr = SlepcVecMAXPBY(u,1.0,-1.0,m,g,U);CHKERRQ(ierr);        

  /* Renormalize u */
  gnorm = 0.0;
  for (i=0;i<n;i++)
    gnorm = gnorm + PetscRealPart(g[i]*PetscConj(g[i]));
  gamma = PetscSqrtReal(1.0+gnorm);
  ierr = VecScale(u,1.0/gamma);CHKERRQ(ierr);

  /* b = gamma*b */
  for (i=k;i<k+l;i++) {
    S[i*m+k+l] *= gamma;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KrylovSchur_Harmonic"
PetscErrorCode EPSSolve_KrylovSchur_Harmonic(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,lwork,nv;
  Vec            u=eps->work[0];
  PetscScalar    *S=eps->T,*Q,*g,*work;
  PetscReal      beta,gnorm;
  PetscBool      breakdown;

  PetscFunctionBegin;
  ierr = PetscMemzero(S,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  lwork = PetscMax((eps->ncv+1)*eps->ncv,7*eps->ncv);
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&g);CHKERRQ(ierr);

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = EPSBasicArnoldi(eps,PETSC_FALSE,S,eps->ncv,eps->V,eps->nconv+l,&nv,u,&beta,&breakdown);CHKERRQ(ierr);
    ierr = VecScale(u,1.0/beta);CHKERRQ(ierr);

    /* Compute translation of Krylov decomposition */ 
    ierr = EPSTranslateHarmonic(nv,S,eps->ncv,eps->target,(PetscScalar)beta,g,work);CHKERRQ(ierr);
    gnorm = 0.0;
    for (i=0;i<nv;i++)
      gnorm = gnorm + PetscRealPart(g[i]*PetscConj(g[i]));

    /* Solve projected problem and compute residual norm estimates */ 
    ierr = EPSProjectedKSNonsym(eps,l,S,eps->ncv,Q,nv);CHKERRQ(ierr);

    /* Check convergence */ 
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,eps->trackall,eps->nconv,nv-eps->nconv,S,eps->ncv,Q,eps->V,nv,beta,PetscSqrtReal(1.0+gnorm),&k,work);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = (nv-k)/2;
#if !defined(PETSC_USE_COMPLEX)
      if (S[(k+l-1)*(eps->ncv+1)+1] != 0.0) {
        if (k+l<nv-1) l = l+1;
        else l = l-1;
      }
#endif
    }
           
    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Arnoldi factorization */
        ierr = PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%G)\n",eps->its,beta);CHKERRQ(ierr);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        for (i=k;i<k+l;i++) {
          S[i*eps->ncv+k+l] = Q[(i+1)*nv-1]*beta;
        }
        ierr = EPSRecoverHarmonic(S,nv,k,l,eps->ncv,g,Q,eps->V,u,work);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V,eps->nconv,k+l,Q,nv,PETSC_FALSE);CHKERRQ(ierr);
    
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecCopy(u,eps->V[k+l]);CHKERRQ(ierr);
    }
    eps->nconv = k;
    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
  } 

  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

