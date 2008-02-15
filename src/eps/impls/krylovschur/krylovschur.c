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

  if (!eps->projection) {
    ierr = EPSSetProjection(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->projection!=EPS_RITZ && eps->projection!=EPS_HARMONIC) {
    SETERRQ(PETSC_ERR_SUP,"Unsupported projection type\n");
  }

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
     B is workspace to store a working copy of S
     ipiv is workspace for pivots (int of length m)
*/
PetscErrorCode EPSTranslateHarmonic(PetscScalar *S,int m,PetscScalar tau,PetscScalar beta,PetscScalar *g,PetscScalar *B,int *ipiv)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) || defined(PETSC_MISSING_LAPACK_GETRS) 
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"GETRF,GETRS - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscBLASInt   info,one = 1;
  int            i;

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
   EPSRecoverHarmonic - Computes a translation of the truncated Krylov decomposition
   in order to recover the original non-translated state

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
PetscErrorCode EPSRecoverHarmonic(PetscScalar *S,int n,int k,int l,int m,PetscScalar *g,PetscScalar *Q,Vec *U,Vec u,PetscScalar *ghat)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscBLASInt   one=1,ncol=k+l;
  PetscScalar    done=1.0,dmone=-1.0,dzero=0.0;
  PetscReal      gamma,gnorm;
  int            i,j;

  PetscFunctionBegin;

  /* g^ = -Q(:,idx)'*g */
  BLASgemv_("C",&n,&ncol,&dmone,Q,&m,g,&one,&dzero,ghat,&one);

  /* S = S + g^*b' */
  for (i=0;i<l;i++) {
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
    gnorm = gnorm + g[i]*g[i];
  gamma = sqrt(1.0+gnorm);
  ierr = VecScale(u,1.0/gamma);CHKERRQ(ierr);

  /* b = gamma*b */
  for (i=k;i<k+l;i++) {
    S[i*m+k+l] *= gamma;
  }
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

   On output:
     S has (real) Schur form with diagonal blocks sorted appropriately
     Q contains the accumulated orthogonal transformations used in the process

   Workspace:
     ritz temporarily stores computed Ritz values
     perm is used for representing the permutation used for sorting values
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
  if (eps->projection==EPS_HARMONIC) {
    ierr = EPSSortDenseSchurTarget(n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi,eps->target);CHKERRQ(ierr);    
  } else {
    ierr = EPSSortDenseSchur(n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi,eps->which);CHKERRQ(ierr);    
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KRYLOVSCHUR"
PetscErrorCode EPSSolve_KRYLOVSCHUR(EPS eps)
{
  PetscErrorCode ierr;
  int            i,k,l,n,lwork,*perm;
  Vec            u=eps->work[1];
  PetscScalar    *S=eps->T,*Q,*g,*work;
  PetscReal      beta,*ritz,gnorm;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  ierr = PetscMemzero(S,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&g);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  lwork = (eps->ncv+4)*eps->ncv;
  if (!eps->ishermitian) {
    ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(eps->ncv*sizeof(int),&perm);CHKERRQ(ierr);

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
    if (eps->projection==EPS_HARMONIC) {
      ierr = EPSTranslateHarmonic(S,eps->ncv,eps->target,(PetscScalar)beta,g,work,perm);CHKERRQ(ierr);
    }

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

    /* Fix residual norms if harmonic */
    if (eps->projection==EPS_HARMONIC) {
      gnorm = 0.0;
      for (i=0;i<eps->ncv;i++)
        gnorm = gnorm + g[i]*g[i];
      for (i=eps->nconv;i<eps->nv;i++)
        eps->errest[i] *= sqrt(1.0+gnorm);
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
        if (eps->projection==EPS_HARMONIC) {
          ierr = EPSRecoverHarmonic(S,n,k,l,eps->ncv,g,Q,eps->V,u,work);CHKERRQ(ierr);
        }
      }
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
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecCopy(u,eps->V[k+l]);CHKERRQ(ierr);
    }
    eps->nconv = k;

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nv);
    
  } 

  ierr = PetscFree(g);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  if (!eps->ishermitian) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(ritz);CHKERRQ(ierr);
  }
  ierr = PetscFree(perm);CHKERRQ(ierr);
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

