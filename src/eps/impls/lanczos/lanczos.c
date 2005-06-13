/*                       

   SLEPc eigensolver: "lanczos"

   Method: Explicitly Restarted Symmetric/Hermitian Lanczos

   Description:

       This solver implements the Lanczos method for symmetric (Hermitian)
       problems, with explicit restart and deflation. When building the 
       Lanczos factorization, several reorthogonalization strategies can
       be selected.

   Algorithm:

       The implemented algorithm builds a Lanczos factorization of order
       ncv. Converged eigenpairs are locked and the iteration is restarted
       with the rest of the columns being the active columns for the next
       Lanczos factorization. Currently, no filtering is applied to the
       vector used for restarting.

       The following reorthogonalization schemes are currently implemented:

       - Full reorthogonalization: at each Lanczos step, the corresponding
       Lanczos vector is orthogonalized with respect to all the previous
       vectors.

   References:

       [1] B.N. Parlett, "The Symmetric Eigenvalue Problem", SIAM Classics in 
       Applied Mathematics (1998), ch. 13.

       [2] L. Komzsik, "The Lanczos Method. Evolution and Application", SIAM
       (2003).

       [3] B.N. Parlett and D.S. Scott, The Lanczos algorithm with selective
       orthogonalization, Math. Comp., 33 (1979), no. 145, 217-238.

       [4] H.D. Simon, The Lanczos algorithm with partial reorthogonalization,
       Math. Comp., 42 (1984), no. 165, 115-142.

   Last update: October 2004

*/
#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

typedef struct {
  EPSLanczosOrthogType reorthog;
} EPS_LANCZOS;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LANCZOS"
PetscErrorCode EPSSetUp_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  int            N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->nev > N) eps->nev = N;
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMin(N,PetscMax(2*eps->nev,eps->nev+15));
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
/*  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");*/
/*  if (!eps->ishermitian)
    SETERRQ(PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");*/
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  if (eps->T) { ierr = PetscFree(eps->T);CHKERRQ(ierr); }  
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  if (eps->solverclass==EPS_TWO_SIDE) {
    if (eps->Tl) { ierr = PetscFree(eps->Tl);CHKERRQ(ierr); }  
    ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->Tl);CHKERRQ(ierr);
    ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  }
  else { ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFullLanczos"
/*
   EPSFullLanczos - Full reorthogonalization.

   In this variant, at each Lanczos step, the corresponding Lanczos vector 
   is orthogonalized with respect to all the previous Lanczos vectors.
*/
static PetscErrorCode EPSFullLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            j,m = *M;
  PetscReal      norm;

  PetscFunctionBegin;
  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,j+1,V,f,T+m*j,&norm,breakdown);CHKERRQ(ierr);
    if (*breakdown) {
      *M = j+1;
      break;
    }
    if (j<m-1) {
      T[m*j+j+1] = norm;
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSimpleLanczos"
/*
   EPSSimpleLanczos - Local reorthogonalization.

   This is the simplest variant. At each Lanczos step, the corresponding Lanczos vector 
   is orthogonalized with respect to the two previous Lanczos vectors, according to
   the three term Lanczos recurrence. WARNING: This variant does not track the loss of 
   orthogonality that occurs in finite-precision arithmetic and, therefore, the 
   generated vectors are not guaranteed to be (semi-)orthogonal.
*/
static PetscErrorCode EPSSimpleLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            j,m = *M;
  PetscReal      norm;

  PetscFunctionBegin;
  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    if (j == k) {
      ierr = EPSOrthogonalize(eps,1,V+j,f,T+m*j+j,&norm,breakdown);CHKERRQ(ierr);
    } else {
      ierr = EPSOrthogonalize(eps,2,V+j-1,f,T+m*j+j-1,&norm,breakdown);CHKERRQ(ierr);
    }
    if (*breakdown) {
      *M = j+1;
      break;
    }
    if (j<m-1) {
      T[m*j+j+1] = norm;
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSelectiveLanczos"
static PetscErrorCode EPSSelectiveLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown,PetscReal anorm)
{
  PetscErrorCode ierr;
  int            i,j,m = *M,n,info;
  PetscReal      *ritz,*E,*Y,*work,norm;
  PetscTruth     conv;

  PetscFunctionBegin;
  ierr = PetscMalloc(m*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscReal),&E);CHKERRQ(ierr);
  ierr = PetscMalloc(m*m*sizeof(PetscReal),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(2*m*sizeof(PetscReal),&work);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,k,V,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    if (j == k) {
      ierr = EPSOrthogonalize(eps,1,V+j,f,T+m*j+j,&norm,breakdown);CHKERRQ(ierr);
    } else {
      ierr = EPSOrthogonalize(eps,2,V+j-1,f,T+m*j+j-1,&norm,breakdown);CHKERRQ(ierr);
    }
    
    if (*breakdown) {
      *M = j+1;
      break;
    }

    n = j-k+1;
    for (i=0;i<n-1;i++) {
      ritz[i] = PetscRealPart(T[(i+k)*(m+1)]);
      E[i] = PetscRealPart(T[(i+k)*(m+1)+1]);
    }
    ritz[n-1] = PetscRealPart(T[(n-1+k)*(m+1)]);

    /* Compute eigenvalues and eigenvectors Y of the tridiagonal block */
    dstev_("V",&n,ritz,E,Y,&n,work,&info,1);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xSTEQR %i",info);
    
    /* Estimate ||A|| */
    for (i=0;i<n;i++) 
      if (PetscAbsReal(ritz[i]) > anorm) anorm = PetscAbsReal(ritz[i]);
    
    /* Exit if residual norm is small [Parlett, page 300] */
    conv = PETSC_FALSE;
    for (i=0;i<n && !conv;i++)
      if (norm*PetscAbsScalar(Y[i*n+n-1]) < PETSC_SQRT_MACHINE_EPSILON*anorm)
        conv = PETSC_TRUE;
    
    if (conv) {
      *M = j+1;
      break;
    }

    if (j<m-1) {
      T[m*j+j+1] = norm;
      ierr = VecScale(f,1.0 / norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;
  ierr = PetscFree(ritz);CHKERRQ(ierr);
  ierr = PetscFree(E);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifndef PETSC_USE_COMPLEX
#undef __FUNCT__  
#define __FUNCT__ "EPSUpdateOmega"
static PetscErrorCode EPSUpdateOmega(int j,PetscReal* alpha,PetscReal* beta,PetscReal anorm,PetscReal *omega,PetscReal *oldomega)
{
  int k;
  PetscReal w;
  PetscFunctionBegin;
  
  if (j>0) {
    for (k=0;k<j-1;k++) {
      if (k==0) 
        w = beta[k]*omega[k+1]+(alpha[k]-alpha[j])*omega[k]-beta[j-1]*oldomega[k];
      else 
        w = beta[k]*omega[k+1]+(alpha[k]-alpha[j])*omega[k]+
	    beta[k-1]*omega[k-1]-beta[j-1]*oldomega[k];
      if (w>0) 
	oldomega[k] = (w + 2*PETSC_MACHINE_EPSILON*anorm) / beta[j];
      else 
	oldomega[k] = (w - 2*PETSC_MACHINE_EPSILON*anorm) / beta[j];
    }
    oldomega[j-1] = PETSC_MACHINE_EPSILON;
  }
  oldomega[j] = 1;
    
//   alpha--; beta-=2; omega--; oldomega--;
//   if (j>1) {
//     t = beta[2]*omega[2]+(alpha[1]-alpha[j])*omega[1]-beta[j]*oldomega[1];
//     if (t>0) oldomega[1] = (t + theta)/beta[j+1];
//     else oldomega[1] = (t - theta)/beta[j+1];
//     for (k=2;k<j;k++) {
//       t = beta[k+1]*omega[k+1]+(alpha[k]-alpha[j])*omega[k]+
//           beta[k]*omega[k-1]-beta[j]*oldomega[k];
//       if (t>0) oldomega[k] = (t + theta)/beta[j+1];
//       else oldomega[k] = (t - theta)/beta[j+1];
//     }
//   }
//   oldomega[j] = psi;
  
  /* SWAP(oldomega,omega) */
  for (k=0;k<=j;k++) {
    w = oldomega[k];
    oldomega[k] = omega[k];
    omega[k] = w;
  }
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "EPSPeriodicLanczos"
static PetscErrorCode EPSPeriodicLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta, PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            i,j,m = *M;
  PetscReal      norm;
  PetscScalar    *omega;
  PetscTruth     reorthog;

  PetscFunctionBegin;
  reorthog = PETSC_FALSE;
  ierr = PetscMalloc(m*sizeof(PetscScalar),&omega);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    if (reorthog) {
      ierr = EPSOrthogonalize(eps,j+1,V,f,T+m*j,&norm,breakdown);CHKERRQ(ierr);
    } else {
      for (i=0;i<j-1;i++) T[m*j+i] = 0.0;
      if (j == k) {
        ierr = EPSOrthogonalize(eps,1,V+j,f,T+m*j+j,&norm,breakdown);CHKERRQ(ierr);
      } else {
        ierr = EPSOrthogonalize(eps,2,V+j-1,f,T+m*j+j-1,&norm,breakdown);CHKERRQ(ierr);
      }
    }

    if (*breakdown) {
      *M = j+1;
      break;
    }

    if (reorthog) {
      reorthog = PETSC_FALSE;
    } else if (j>1) {
      ierr = VecMDot(j-1,f,eps->V,omega);CHKERRQ(ierr);
      for (i=0;i<j-1 && !reorthog;i++) 
	if (PetscAbsScalar(omega[i]) > PETSC_SQRT_MACHINE_EPSILON*norm) {
	  reorthog = PETSC_TRUE;
	}
      if (reorthog) {
        ierr = EPSOrthogonalize(eps,j-1,V,f,T+m*j,&norm,PETSC_NULL);CHKERRQ(ierr);
      }
    }

    if (j<m-1) {
      T[m*j+j+1] = norm;
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;
  ierr = PetscFree(omega);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSPartialLanczos"
static PetscErrorCode EPSPartialLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta, PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            i,j,l,m = *M;
  PetscReal      norm;
  PetscScalar    *omega,nu;
  PetscTruth     reorthog,*which;

  PetscFunctionBegin;
  nu = sqrt(PETSC_MACHINE_EPSILON*PETSC_SQRT_MACHINE_EPSILON);
  reorthog = PETSC_FALSE;
  ierr = PetscMalloc(m*sizeof(PetscScalar),&omega);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscTruth),&which);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    if (j == k) {
      ierr = EPSOrthogonalize(eps,1,V+j,f,T+m*j+j,&norm,breakdown);CHKERRQ(ierr);
    } else {
      ierr = EPSOrthogonalize(eps,2,V+j-1,f,T+m*j+j-1,&norm,breakdown);CHKERRQ(ierr);
    }

    if (*breakdown) {
      *M = j+1;
      break;
    }

    if (reorthog) {
      for (i=0;i<j-1;i++) 
	if (which[i]) {
          ierr = EPSOrthogonalize(eps,1,V+i,f,PETSC_NULL,&norm,PETSC_NULL);CHKERRQ(ierr);
	}
      reorthog = PETSC_FALSE;
    } else if (j>1) {
      ierr = VecMDot(j-1,f,eps->V,omega);CHKERRQ(ierr);
      for (i=0;i<j-1;i++) {
        omega[i] /= norm;
	which[i] = PETSC_FALSE;
      }
      for (i=0;i<j-1;i++) 
	if (PetscAbsScalar(omega[i]) > PETSC_SQRT_MACHINE_EPSILON) {
	  reorthog = PETSC_TRUE;
	  which[i] = PETSC_TRUE;
	  for (l=i-1;l>0 && omega[l] > nu;l--) which[l] = PETSC_TRUE;
	  for (l=i+1;l<j-1 && omega[l] > nu;l++) which[l] = PETSC_TRUE;
	}
      if (reorthog) 
        for (i=0;i<j-1;i++) 
	  if (which[i]) {
            ierr = EPSOrthogonalize(eps,1,V+i,f,PETSC_NULL,&norm,PETSC_NULL);CHKERRQ(ierr);
	  }
    }

    if (j<m-1) {
      T[m*j+j+1] = norm;
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;
  ierr = PetscFree(omega);CHKERRQ(ierr);
  ierr = PetscFree(which);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicLanczos"
/*
   EPSBasicLanczos - Computes an m-step Lanczos factorization. The first k
   columns are assumed to be locked and therefore they are not modified. On
   exit, the following relation is satisfied:

                    OP * V - V * T = f * e_m^T

   where the columns of V are the Lanczos vectors, T is a tridiagonal matrix, 
   f is the residual vector and e_m is the m-th vector of the canonical basis. 
   The Lanczos vectors (together with vector f) are B-orthogonal (to working
   accuracy) if full reorthogonalization is being used, otherwise they are
   (B-)semi-orthogonal. On exit, beta contains the B-norm of f and the next 
   Lanczos vector can be computed as v_{m+1} = f / beta. 

   This function simply calls another function which depends on the selected
   reorthogonalization strategy.
*/
static PetscErrorCode EPSBasicLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *m,Vec f,PetscReal *beta,PetscTruth *breakdown,PetscReal anorm)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (lanczos->reorthog) {
    case EPSLANCZOS_ORTHOG_NONE:
      ierr = EPSSimpleLanczos(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_ORTHOG_SELECTIVE:
      ierr = EPSSelectiveLanczos(eps,T,V,k,m,f,beta,breakdown,anorm);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_ORTHOG_PARTIAL:
      ierr = EPSPartialLanczos(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_ORTHOG_PERIODIC:
      ierr = EPSPeriodicLanczos(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_ORTHOG_FULL:
      ierr = EPSFullLanczos(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

/*
#undef __FUNCT__  
#define __FUNCT__ "RefineBounds"
static PetscErrorCode RefineBounds(int n,PetscReal *ritz,PetscReal *bnd,PetscReal eps,PetscReal tol)
{
  int       i,mid;
  PetscReal gapl,gapr,max,eps34 = sqrt(eps*sqrt(eps));
  
  PetscFunctionBegin;
  
  max = bnd[0];
  mid = 0;
  for (i=1;i<n;i++) {
    if (bnd[i] > max) {
      max = bnd[i];
      mid = i;
    }
  }
  
  for (i=n-1;i>mid;i--) 
    if (PetscAbsReal(ritz[i-1]-ritz[i]) < eps34*PetscAbsReal(ritz[i]))
      if (bnd[i-1] > tol && bnd[i] > tol) {
        bnd[i-1] = sqrt(bnd[i-1]*bnd[i-1]+bnd[i]*bnd[i]);
        bnd[i] = 0;
      }
      
  for (i=0;i<mid;i++)
    if (PetscAbsReal(ritz[i+1]-ritz[i]) < eps34*PetscAbsReal(ritz[i]))
      if (bnd[i+1] > tol && bnd[i] > tol) {
        bnd[i+1] = sqrt(bnd[i+1]*bnd[i+1]+bnd[i]*bnd[i]);
        bnd[i] = 0;
      }
  
  gapl = ritz[n-1] - ritz[0]; 
  for (i=0;i<n-1;i++) {
    gapr = ritz[i+1] - ritz[i];
    if (gapl < gapr) gapl = gapr;
    if (bnd[i] < gapl) bnd[i] = bnd[i]*(bnd[i]/gapl);
    gapl = gapr;
  }
  if (bnd[n-1] < gapl) bnd[n-1] = bnd[n-1]*bnd[n-1]/gapl;
  
  PetscFunctionReturn(0);
}
*/

extern double dlamch_(const char*);

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LANCZOS"
PetscErrorCode EPSSolve_LANCZOS(EPS eps)
{
#if defined(SLEPC_MISSING_LAPACK_STEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"STEQR - Lapack routine is unavailable.");
#else
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscErrorCode ierr;
  int            nconv,i,j,k,n,m,info,N,*perm,restart,*ifail,*iwork,mout,
                 ncv=eps->ncv;
  Vec            f=eps->work[0];
  PetscScalar    *T=eps->T,*Y,*H;
  PetscReal      *ritz,*bnd,*D,*E,*work,anorm,beta,gap,norm,abstol;
  PetscTruth     breakdown,*conv;

  PetscFunctionBegin;
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&bnd);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&H);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&D);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&E);CHKERRQ(ierr);
  ierr = PetscMalloc(5*ncv*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(5*ncv*sizeof(int),&iwork);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(int),&ifail);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(int),&perm);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscTruth),&conv);CHKERRQ(ierr);

  /* The first Lanczos vector is the normalized initial vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0]);CHKERRQ(ierr);
  
  anorm = 1.0;
  nconv = 0;
  eps->its = 0;
  for (i=0;i<eps->ncv;i++) eps->eigi[i]=0.0;
  EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,ncv);
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  
  /* Restart loop */
  eps->reason = EPS_CONVERGED_ITERATING;
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    /* Compute an ncv-step Lanczos factorization */
    m = ncv;
    ierr = EPSBasicLanczos(eps,T,eps->V,nconv,&m,f,&beta,&breakdown,anorm);CHKERRQ(ierr);
    if (breakdown) {
      printf("Breakdown in Lanczos method (norm=%g) n=%i\n",beta,m);
      PetscLogInfo((eps,"Breakdown in Lanczos method (norm=%g)\n",beta));
    }

    /* At this point, T has the following structure

              | *     |               |
              |     * |               |
              | ------|-------------- |
          T = |       | *   *         |
              |       | *   *   *     |
              |       |     *   *   * |
              |       |         *   * |

       that is, a real symmetric tridiagonal matrix of order ncv whose 
       principal submatrix of order nconv is diagonal.  */

    /* Extract the tridiagonal block from T. Store the diagonal elements in D 
       and the off-diagonal elements in E  */

    n = m - nconv;
//  DIAGONAL SUPERIOR 
//    ritz[0] = PetscRealPart(T[nconv*(ncv+1)]);
//    for (i=1;i<n;i++) {
//      ritz[i] = PetscRealPart(T[(i+nconv)*(ncv+1)]);
//      E[i-1] = PetscRealPart(T[(i+nconv)*(ncv+1)-1]);
//    }

//  DIAGONAL INFERIOR
    for (i=0;i<n-1;i++) {
      ritz[i] = PetscRealPart(T[(i+nconv)*(ncv+1)]);
      E[i] = PetscRealPart(T[(i+nconv)*(ncv+1)+1]);
    }
    ritz[n-1] = PetscRealPart(T[(n-1+nconv)*(ncv+1)]);

    /* Compute eigenvalues and eigenvectors Y of the tridiagonal block */
    dstev_("V",&n,ritz,E,Y,&n,work,&info,1);
//    abstol = 2.0*dlamch_("S");
//    dstevx_("V","A",&n,D,E,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,&abstol,&mout,ritz,Y,&n,work,iwork,perm,&info);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xSTEQR %i",info);

// USAR H
//     for (i=0;i<n;i++) {
//       for (j=0;j<n;j++) {
//         H[j*n+i] = T[(nconv+j)*ncv+nconv+i];
//       }
//     }
//     ierr = PetscMemzero(Y,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
//     for (i=0;i<n;i++) { Y[i*(n+1)] = 1.0; }
//     ierr = EPSDenseSchur(n,0,H,Y,ritz,E);CHKERRQ(ierr);
//     ierr = EPSSortDenseSchur(n,0,H,Y,ritz,E);CHKERRQ(ierr);
        
    /* Estimate ||A|| */
    for (i=0;i<n;i++) 
      if (PetscAbsReal(ritz[i]) > anorm) anorm = PetscAbsReal(ritz[i]);
    
    /* Compute residual norm estimates as beta*abs(Y(m,:)) */
    for (i=0;i<n;i++) 
      bnd[i] = beta*PetscAbsScalar(Y[i*n+n-1]) + PETSC_MACHINE_EPSILON*anorm;
//     for (i=0;i<n;i++) {
//       gap = PETSC_MAX;
//       for (j=0;j<nconv;j++) 
//         if (PetscAbsScalar(ritz[i]-eps->eigr[j])<gap)
// 	  gap = PetscAbsScalar(ritz[i]-eps->eigr[j]);
//       for (j=0;j<n;j++) 
//         if (i!=j && PetscAbsScalar(ritz[i]-ritz[j])<gap)
// 	  gap = PetscAbsScalar(ritz[i]-ritz[j]);
//       bnd[i] = beta*PetscAbsScalar(Y[i*n+n-1]) * beta*PetscAbsScalar(Y[i*n+n-1]) / gap;
//     }  

/*    ierr = RefineBounds(n,ritz,bnd,eps->tol,eps->tol*anorm*N);CHKERRQ(ierr);*/
#ifdef PETSC_USE_COMPLEX
    for (i=0;i<n;i++) {
      eps->eigr[i+nconv] = ritz[i];
    }
    ierr = EPSSortEigenvalues(n,eps->eigr+nconv,eps->eigi,eps->which,n,perm);CHKERRQ(ierr);
#else
    ierr = EPSSortEigenvalues(n,ritz,eps->eigi,eps->which,n,perm);CHKERRQ(ierr);
#endif

/* CULLUM 
    m = n-1;
    for (i=0;i<m;i++) {
      D[i] = PetscRealPart(T[(i+nconv+1)*(ncv+1)]);
      E[i] = PetscRealPart(T[(i+nconv+1)*(ncv+1)+1]);
    }
    LAPACKsteqr_("N",&m,D,E,PETSC_NULL,&m,work,&info,1);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xSTEQR %i",info);

    k = nconv;
    for (i=0;i<n;i++) {
      eps->eigr[nconv+i] = ritz[perm[i]];
      eps->errest[nconv+i] = bnd[perm[i]] / PetscAbsReal(ritz[perm[i]]);
      repeated = PETSC_FALSE;
      while (i<n-1 && 
             PetscAbsReal(ritz[perm[i]]-ritz[perm[i+1]])
	     / PetscAbsReal(ritz[perm[i+1]]) < eps->tol) {
	i++;
	eps->eigr[nconv+i] = ritz[perm[i]];
	eps->errest[nconv+i] = bnd[perm[i]] / PetscAbsReal(ritz[perm[i]]);
	repeated = PETSC_TRUE;
      }
      if (eps->errest[nconv+i]<eps->tol) {
	j = 0; found = PETSC_FALSE;
	while (j<m && !found) {
          found = PetscAbsReal(D[j]-eps->eigr[nconv+i]) / PetscAbsReal(D[j]) < 100*eps->tol;
	  j++;
	}
	if (found) printf("found %i %e\n", nconv+i, eps->eigr[nconv+i]);
	if (!found || (found && repeated)) {
          if (i>k) {
	    j = perm[i]; perm[i] = perm[k]; perm[k] = j;
	    ts = eps->eigr[nconv+i]; eps->eigr[nconv+i] = eps->eigr[nconv+k]; eps->eigr[nconv+k] = ts;
	    ts = eps->errest[nconv+i]; eps->errest[nconv+i] = eps->errest[nconv+k]; eps->errest[nconv+k] = ts;
	  }
          k++;
	}
      } 
    }
*/
   
    /* Look for converged eigenpairs */
    k = nconv;
    for (i=0;i<n;i++) {
      conv[i] = PETSC_FALSE;
      eps->eigr[k] = ritz[perm[i]];
      eps->errest[k] = bnd[perm[i]] / PetscAbsReal(ritz[perm[i]]);      
      if (eps->errest[k] < eps->tol) {
        PetscReal res;
	ierr = VecSet(eps->AV[k],0.0);CHKERRQ(ierr);
	ierr = VecMAXPY(eps->AV[k],n,Y+perm[i]*n,eps->V+nconv);CHKERRQ(ierr);
        ierr = VecNorm(eps->AV[k],NORM_2,&norm);CHKERRQ(ierr);
        ierr = VecScale(eps->AV[k],1.0/norm);CHKERRQ(ierr);
        ierr = STApply(eps->OP,eps->AV[k],f);CHKERRQ(ierr);
  	ierr = VecAXPY(f,-eps->eigr[k],eps->AV[k]);CHKERRQ(ierr);
 	ierr = VecNorm(f,NORM_2,&res);CHKERRQ(ierr);
  	eps->errest[k] = res / PetscAbsScalar(eps->eigr[k]);
        if (eps->errest[k] < eps->tol) {
	  conv[i] = PETSC_TRUE;
          printf("[%i] C %i %g errest=%g norm=%g\n",eps->its,k,eps->eigr[k],eps->errest[k],norm);
   	  k++;
  	}
      }
    }

    /* Look for unconverged eigenpairs */
    j = k;
    restart = -1;
    for (i=0;i<n;i++) {
      if (!conv[i]) {
        if (restart == -1) restart = i;
        eps->eigr[j] = ritz[perm[i]];
        eps->errest[j] = bnd[perm[i]] / ritz[perm[i]];
//	printf("[%i] N %i %g (%g)\n",eps->its,j,eps->eigr[j],eps->errest[j]);
        j++;
      } 
    }
    
    if (k<eps->nev && restart != -1 && !breakdown) {
      /* use first not converged vector for restarting */
      ierr = VecSet(eps->AV[k],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(eps->AV[k],n,Y+perm[restart]*n,eps->V+nconv);CHKERRQ(ierr);
      ierr = VecNorm(eps->AV[k],NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecScale(eps->AV[k],1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(eps->AV[k],eps->V[k]);CHKERRQ(ierr);
      printf("[%i] R %i %g errest=%g norm=%g\n",eps->its,k,eps->eigr[k],eps->errest[k],norm);
    }
    
    /* copy converged vectors to V */
    for (i=nconv;i<k;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }

    if (k<eps->nev && (restart == -1 || breakdown)) {
      /* use random vector for restarting */
      printf("Using random vector for restart\n");
      PetscLogInfo((eps,"Using random vector for restart\n"));
//    ierr = EPSGetStartVector(eps,k,eps->V[k]);CHKERRQ(ierr);
      ierr = SlepcVecSetRandom(eps->V[k]);CHKERRQ(ierr);
      ierr = EPSOrthogonalize(eps,eps->nds+k,eps->DSV,eps->V[k],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr);
      if (breakdown) {
        eps->reason = EPS_DIVERGED_BREAKDOWN;
	printf("Unable to generate more start vectors\n");
	PetscLogInfo((eps,"Unable to generate more start vectors\n"));
      } else {
        ierr = VecScale(eps->V[k],1.0/norm);CHKERRQ(ierr);
      }
    }

    nconv = k;
    EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,nconv+n);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
  }
  
  eps->nconv = nconv;

  ierr = PetscFree(ritz);CHKERRQ(ierr);
  ierr = PetscFree(bnd);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(H);CHKERRQ(ierr);
  ierr = PetscFree(E);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_LANCZOS"
PetscErrorCode EPSSetFromOptions_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS *)eps->data;
  PetscTruth     flg;
  const char     *list[6] = { "none", "full" , "selective", "periodic", "partial" };

  PetscFunctionBegin;
  ierr = PetscOptionsHead("LANCZOS options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-eps_lanczos_orthog","Reorthogonalization type","EPSLanczosSetOrthog",list,5,list[lanczos->reorthog],(int*)&lanczos->reorthog,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosSetOrthog_LANCZOS"
PetscErrorCode EPSLanczosSetOrthog_LANCZOS(EPS eps,EPSLanczosOrthogType reorthog)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;

  PetscFunctionBegin;
  switch (reorthog) {
    case EPSLANCZOS_ORTHOG_NONE:
    case EPSLANCZOS_ORTHOG_FULL:
    case EPSLANCZOS_ORTHOG_SELECTIVE:
    case EPSLANCZOS_ORTHOG_PERIODIC:
    case EPSLANCZOS_ORTHOG_PARTIAL:
      lanczos->reorthog = reorthog;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid reorthogonalization type");
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosSetOrthog"
/*@
   EPSLanczosSetOrthog - Sets the type of reorthogonalization used during the Lanczos
   iteration. 

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  reorthog - the type of reorthogonalization

   Options Database Key:
.  -eps_lanczos_orthog - Sets the reorthogonalization type (either 'none' or 
                           'full')
   
   Level: advanced

.seealso: EPSLanczosGetOrthog()
@*/
PetscErrorCode EPSLanczosSetOrthog(EPS eps,EPSLanczosOrthogType reorthog)
{
  PetscErrorCode ierr, (*f)(EPS,EPSLanczosOrthogType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSLanczosSetOrthog_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,reorthog);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosGetOrthog_LANCZOS"
PetscErrorCode EPSLanczosGetOrthog_LANCZOS(EPS eps,EPSLanczosOrthogType *reorthog)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscFunctionBegin;
  *reorthog = lanczos->reorthog;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosGetOrthog"
/*@C
   EPSLanczosGetOrthog - Gets the type of reorthogonalization used during the Lanczos
   iteration. 

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Input Parameter:
.  reorthog - the type of reorthogonalization

   Level: advanced

.seealso: EPSLanczosSetOrthog()
@*/
PetscErrorCode EPSLanczosGetOrthog(EPS eps,EPSLanczosOrthogType *reorthog)
{
  PetscErrorCode ierr, (*f)(EPS,EPSLanczosOrthogType*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSLanczosGetOrthog_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,reorthog);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_LANCZOS"
PetscErrorCode EPSView_LANCZOS(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS *)eps->data;
  PetscTruth     isascii;
  const char     *list[5] = { "none", "full" , "selective", "periodic", "partial" };

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSLANCZOS",((PetscObject)viewer)->type_name);
  }  
  ierr = PetscViewerASCIIPrintf(viewer,"reorthogonalization: %s\n",list[lanczos->reorthog]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
EXTERN PetscErrorCode EPSSolve_TS_LANCZOS(EPS);
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_LANCZOS"
PetscErrorCode EPSCreate_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_LANCZOS,&lanczos);CHKERRQ(ierr);
  PetscMemzero(lanczos,sizeof(EPS_LANCZOS));
  PetscLogObjectMemory(eps,sizeof(EPS_LANCZOS));
  eps->data                      = (void *) lanczos;
  eps->ops->solve                = EPSSolve_LANCZOS;
  eps->ops->solvets              = EPSSolve_TS_LANCZOS;
  eps->ops->setup                = EPSSetUp_LANCZOS;
  eps->ops->setfromoptions       = EPSSetFromOptions_LANCZOS;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = EPSView_LANCZOS;
  eps->ops->backtransform        = EPSBackTransform_Default;
  if (eps->solverclass==EPS_TWO_SIDE)
       eps->ops->computevectors       = EPSComputeVectors_Schur;
  else eps->ops->computevectors       = EPSComputeVectors_Default;
  lanczos->reorthog              = EPSLANCZOS_ORTHOG_FULL;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosSetOrthog_C","EPSLanczosSetOrthog_LANCZOS",EPSLanczosSetOrthog_LANCZOS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosGetOrthog_C","EPSLanczosGetOrthog_LANCZOS",EPSLanczosGetOrthog_LANCZOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

