/*                       

   SLEPc eigensolver: "lanczos"

   Method: Explicitly Restarted Symmetric/Hermitian Lanczos

   Algorithm:

       Lanczos method for symmetric (Hermitian) problems, with explicit 
       restart and deflation. Several reorthogonalization strategies can
       be selected.

   References:

       [1] "Lanczos Methods in SLEPc", SLEPc Technical Report STR-5, 
           available at http://www.grycap.upv.es/slepc.

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

typedef struct {
  EPSLanczosReorthogType reorthog;
} EPS_LANCZOS;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LANCZOS"
PetscErrorCode EPSSetUp_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMin(N,PetscMax(2*eps->nev,eps->nev+15));
  if (!eps->max_it) eps->max_it = PetscMax(100,2*N/eps->ncv);

  if (eps->solverclass==EPS_ONE_SIDE) {
    if (eps->which == EPS_LARGEST_IMAGINARY || eps->which == EPS_SMALLEST_IMAGINARY)
      SETERRQ(1,"Wrong value of eps->which");
    if (!eps->ishermitian)
      SETERRQ(PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");
  } else {
    if (eps->which != EPS_LARGEST_MAGNITUDE)
      SETERRQ(1,"Wrong value of eps->which");
  }
  if (!eps->projection) {
    ierr = EPSSetProjection(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->projection!=EPS_RITZ) {
    SETERRQ(PETSC_ERR_SUP,"Unsupported projection type\n");
  }

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  if (eps->solverclass==EPS_TWO_SIDE) {
    ierr = PetscFree(eps->Tl);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->Tl);CHKERRQ(ierr);
  }
  ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSLocalLanczos"
/*
   EPSLocalLanczos - Local reorthogonalization.

   This is the simplest variant. At each Lanczos step, the corresponding Lanczos vector 
   is orthogonalized with respect to the two previous Lanczos vectors, according to
   the three term Lanczos recurrence. WARNING: This variant does not track the loss of 
   orthogonality that occurs in finite-precision arithmetic and, therefore, the 
   generated vectors are not guaranteed to be (semi-)orthogonal.
*/
static PetscErrorCode EPSLocalLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            i,j,m = *M;
  PetscReal      norm;
  PetscTruth     *which,lwhich[100];
  
  PetscFunctionBegin;  
  if (m>100) {
    ierr = PetscMalloc(sizeof(PetscTruth)*m,&which);CHKERRQ(ierr);
  } else which = lwhich;
  for (i=0;i<k;i++)
    which[i] = PETSC_TRUE;

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,eps->nds,PETSC_NULL,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL,eps->work[0]);CHKERRQ(ierr);
    which[j] = PETSC_TRUE;
    if (j-2>=k) which[j-2] = PETSC_FALSE;
    ierr = IPOrthogonalize(eps->ip,j+1,which,V,f,T+m*j,&norm,breakdown,eps->work[0]);CHKERRQ(ierr);
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

  if (m>100) { ierr = PetscFree(which);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSelectiveLanczos"
/*
   EPSSelectiveLanczos - Selective reorthogonalization.
*/
static PetscErrorCode EPSSelectiveLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown,PetscReal anorm)
{
  PetscErrorCode ierr;
  int            i,j,m = *M,n,nritz=0,nritzo;
  PetscReal      *ritz,norm;
  PetscScalar    *Y;
  PetscTruth     *which,lwhich[100];

  PetscFunctionBegin;
  ierr = PetscMalloc(m*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
  ierr = PetscMalloc(m*m*sizeof(PetscScalar),&Y);CHKERRQ(ierr);

  if (m>100) {
    ierr = PetscMalloc(sizeof(PetscTruth)*m,&which);CHKERRQ(ierr);
  } else which = lwhich;
  for (i=0;i<k;i++)
    which[i] = PETSC_TRUE;

  for (j=k;j<m;j++) {
    /* Lanczos step */
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,eps->nds,PETSC_NULL,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL,eps->work[0]);CHKERRQ(ierr);
    which[j] = PETSC_TRUE;
    if (j-2>=k) which[j-2] = PETSC_FALSE;
    ierr = IPOrthogonalize(eps->ip,j+1,which,V,f,T+m*j,&norm,breakdown,eps->work[0]);CHKERRQ(ierr);
    if (*breakdown) {
      *M = j+1;
      break;
    }

    /* Compute eigenvalues and eigenvectors Y of the tridiagonal block */
    n = j-k+1;
    ierr = EPSDenseTridiagonal(n,T+k*(m+1),m,ritz,Y);CHKERRQ(ierr);
    
    /* Estimate ||A|| */
    for (i=0;i<n;i++) 
      if (PetscAbsReal(ritz[i]) > anorm) anorm = PetscAbsReal(ritz[i]);

    /* Compute nearly converged Ritz vectors */
    nritzo = 0;
    for (i=0;i<n;i++)
      if (norm*PetscAbsScalar(Y[i*n+n-1]) < PETSC_SQRT_MACHINE_EPSILON*anorm)
	nritzo++;

    if (nritzo>nritz) {
      nritz = 0;
      for (i=0;i<n;i++) {
	if (norm*PetscAbsScalar(Y[i*n+n-1]) < PETSC_SQRT_MACHINE_EPSILON*anorm) {
	  ierr = VecSet(eps->AV[nritz],0.0);CHKERRQ(ierr);
	  ierr = VecMAXPY(eps->AV[nritz],n,Y+i*n,V+k);CHKERRQ(ierr);
          nritz++;
	}
      }
    }

    if (nritz > 0) {
      ierr = IPOrthogonalize(eps->ip,nritz,PETSC_NULL,eps->AV,f,PETSC_NULL,&norm,breakdown,eps->work[0]);CHKERRQ(ierr);
      if (*breakdown) {
	*M = j+1;
	break;
      }
    }
    
    if (j<m-1) {
      T[m*j+j+1] = norm;
      ierr = VecScale(f,1.0 / norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;
  
  ierr = PetscFree(ritz);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  if (m>100) { ierr = PetscFree(which);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "update_omega"
static void update_omega(PetscReal *omega,PetscReal *omega_old,int j,PetscReal *alpha,PetscReal *beta,PetscReal eps1,PetscReal anorm)
{
  int            k;
  PetscReal      T,binv,temp;

  PetscFunctionBegin;
  /* Estimate of contribution to roundoff errors from A*v 
       fl(A*v) = A*v + f, 
     where ||f|| \approx eps1*||A||.
     For a full matrix A, a rule-of-thumb estimate is eps1 = sqrt(n)*eps. */
  T = eps1*anorm;
  binv = 1.0/beta[j+1];

  /* Update omega(1) using omega(0)==0. */
  omega_old[0]= beta[1]*omega[1] + (alpha[0]-alpha[j])*omega[0] - 
                beta[j]*omega_old[0];
  if (omega_old[0] > 0) 
    omega_old[0] = binv*(omega_old[0] + T);
  else
    omega_old[0] = binv*(omega_old[0] - T);  
  
  /* Update remaining components. */
  for (k=1;k<j-1;k++) {
    omega_old[k] = beta[k+1]*omega[k+1] + (alpha[k]-alpha[j])*omega[k] +
                   beta[k]*omega[k-1] - beta[j]*omega_old[k];
    if (omega_old[k] > 0) 
      omega_old[k] = binv*(omega_old[k] + T);       
    else
      omega_old[k] = binv*(omega_old[k] - T);       
  }
  omega_old[j-1] = binv*T;
  
  /* Swap omega and omega_old. */
  for (k=0;k<j;k++) {
    temp = omega[k];
    omega[k] = omega_old[k];
    omega_old[k] = omega[k];
  }
  omega[j] = eps1;  
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "compute_int"
static void compute_int(PetscTruth *which,PetscReal *mu,int j,PetscReal delta,PetscReal eta)
{
  int        i,k,maxpos;
  PetscReal  max;
  PetscTruth found;
  
  PetscFunctionBegin;  
  /* initialize which */
  found = PETSC_FALSE;
  maxpos = 0;
  max = 0.0;
  for (i=0;i<j;i++) {
    if (PetscAbsReal(mu[i]) >= delta) {
      which[i] = PETSC_TRUE;
      found = PETSC_TRUE;
    } else which[i] = PETSC_FALSE;
    if (PetscAbsReal(mu[i]) > max) {
      maxpos = i;
      max = PetscAbsReal(mu[i]);
    }
  }
  if (!found) which[maxpos] = PETSC_TRUE;    
  
  for (i=0;i<j;i++)
    if (which[i]) {
      /* find left interval */
      for (k=i;k>=0;k--) {
        if (PetscAbsReal(mu[k])<eta || which[k]) break;
	else which[k] = PETSC_TRUE;
      }
      /* find right interval */
      for (k=i+1;k<j;k++) {
        if (PetscAbsReal(mu[k])<eta || which[k]) break;
	else which[k] = PETSC_TRUE;
      }
    }
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "EPSPartialLanczos"
/*
   EPSPartialLanczos - Partial reorthogonalization.
*/
static PetscErrorCode EPSPartialLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta, PetscTruth *breakdown,PetscReal anorm)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscErrorCode ierr;
  Mat            A;
  int            i,j,m = *M;
  PetscInt       n;
  PetscReal      norm,*omega,lomega[100],*omega_old,lomega_old[100],eps1,delta,eta,*b,lb[101],*a,la[100];
  PetscTruth     *which,lwhich[100],*which2,lwhich2[100],
                 reorth = PETSC_FALSE,force_reorth = PETSC_FALSE,fro = PETSC_FALSE,estimate_anorm = PETSC_FALSE;

  PetscFunctionBegin;
  if (m>100) {
    ierr = PetscMalloc(m*sizeof(PetscReal),&a);CHKERRQ(ierr);
    ierr = PetscMalloc((m+1)*sizeof(PetscReal),&b);CHKERRQ(ierr);
    ierr = PetscMalloc(m*sizeof(PetscReal),&omega);CHKERRQ(ierr);
    ierr = PetscMalloc(m*sizeof(PetscReal),&omega_old);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscTruth)*m,&which);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscTruth)*m,&which2);CHKERRQ(ierr);
  } else {
    a = la;
    b = lb;
    omega = lomega;
    omega_old = lomega_old;
    which = lwhich;
    which2 = lwhich2;
  }
  
  ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,&n,PETSC_NULL);CHKERRQ(ierr);
  eps1 = sqrt((PetscReal)n)*PETSC_MACHINE_EPSILON/2;
  delta = PETSC_SQRT_MACHINE_EPSILON/sqrt((PetscReal)eps->ncv);
  eta = pow(PETSC_MACHINE_EPSILON,3.0/4.0)/sqrt((PetscReal)eps->ncv);
  if (anorm < 0.0) {
    anorm = 1.0;
    estimate_anorm = PETSC_TRUE;
  }
  for (i=0;i<m-k;i++) 
    omega[i] = omega_old[i] = 0.0;
  for (i=0;i<k;i++)
    which[i] = PETSC_TRUE;  
  
  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,eps->nds,PETSC_NULL,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL,eps->work[0]);CHKERRQ(ierr);
    if (fro) {
      /* Lanczos step with full reorthogonalization */
      ierr = IPOrthogonalize(eps->ip,j+1,PETSC_NULL,V,f,T+m*j,&norm,breakdown,eps->work[0]);CHKERRQ(ierr);      
    } else {
      /* Lanczos step */
      which[j] = PETSC_TRUE;
      if (j-2>=k) which[j-2] = PETSC_FALSE;
      ierr = IPOrthogonalize(eps->ip,j+1,which,V,f,T+m*j,&norm,breakdown,eps->work[0]);CHKERRQ(ierr);
      a[j-k] = PetscRealPart(T[m*j+j]);
      b[j-k+1] = norm;
      
      /* Estimate ||A|| if needed */ 
      if (estimate_anorm) {
        if (j>k) anorm = PetscMax(anorm,PetscAbsReal(a[j-k])+norm+b[j-k]);
	else anorm = PetscMax(anorm,PetscAbsReal(a[j-k])+norm);
      }

      /* Check if reorthogonalization is needed */
      reorth = PETSC_FALSE;
      if (j>k) {      
	update_omega(omega,omega_old,j-k,a,b,eps1,anorm);
	for (i=0;i<j-k;i++)
	  if (PetscAbsScalar(omega[i]) > delta) reorth = PETSC_TRUE;
      }

      if (reorth || force_reorth) {
	if (lanczos->reorthog == EPSLANCZOS_REORTHOG_PERIODIC) {
	  /* Periodic reorthogonalization */
	  if (force_reorth) force_reorth = PETSC_FALSE;
	  else force_reorth = PETSC_TRUE;
	  ierr = IPOrthogonalize(eps->ip,j-k,PETSC_NULL,V+k,f,PETSC_NULL,&norm,breakdown,eps->work[0]);CHKERRQ(ierr);
	  for (i=0;i<j-k;i++)
            omega[i] = eps1;
	} else {
	  /* Partial reorthogonalization */
	  if (force_reorth) force_reorth = PETSC_FALSE;
	  else {
	    force_reorth = PETSC_TRUE;
	    compute_int(which2,omega,j-k,delta,eta);
	    for (i=0;i<j-k;i++) 
	      if (which2[i]) omega[i] = eps1;
	  }
	  ierr = IPOrthogonalize(eps->ip,j-k,which2,V+k,f,PETSC_NULL,&norm,breakdown,eps->work[0]);CHKERRQ(ierr);	
	}	
      }
    }
    
    if (*breakdown || norm < n*anorm*PETSC_MACHINE_EPSILON) {
      *M = j+1;
      break;
    }
    if (!fro && norm*delta < anorm*eps1) {
      fro = PETSC_TRUE;
      PetscInfo1(eps,"Switching to full reorthogonalization at iteration %i\n",eps->its);	
    }
    if (j<m-1) {
      T[m*j+j+1] = b[j-k+1] = norm;
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;

  if (m>100) {
    ierr = PetscFree(a);CHKERRQ(ierr);
    ierr = PetscFree(b);CHKERRQ(ierr);
    ierr = PetscFree(omega);CHKERRQ(ierr);
    ierr = PetscFree(omega_old);CHKERRQ(ierr);
    ierr = PetscFree(which);CHKERRQ(ierr);
    ierr = PetscFree(which2);CHKERRQ(ierr);
  }
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
  IPOrthogonalizationRefinementType orthog_ref;

  PetscFunctionBegin;
  switch (lanczos->reorthog) {
    case EPSLANCZOS_REORTHOG_LOCAL:
      ierr = EPSLocalLanczos(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_REORTHOG_SELECTIVE:
      ierr = EPSSelectiveLanczos(eps,T,V,k,m,f,beta,breakdown,anorm);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_REORTHOG_PARTIAL:
    case EPSLANCZOS_REORTHOG_PERIODIC:
      ierr = EPSPartialLanczos(eps,T,V,k,m,f,beta,breakdown,anorm);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_REORTHOG_FULL:
      ierr = EPSBasicArnoldi(eps,PETSC_FALSE,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_REORTHOG_DELAYED:
      ierr = IPGetOrthogonalization(eps->ip,PETSC_NULL,&orthog_ref,PETSC_NULL);CHKERRQ(ierr);
      if (orthog_ref == IP_ORTH_REFINE_NEVER) {
        ierr = EPSDelayedArnoldi1(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);       
      } else {
        ierr = EPSDelayedArnoldi(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid reorthogonalization type"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LANCZOS"
PetscErrorCode EPSSolve_LANCZOS(EPS eps)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscErrorCode ierr;
  int            nconv,i,j,k,n,m,*perm,restart,ncv=eps->ncv;
  Vec            f=eps->work[1];
  PetscScalar    *T=eps->T,*Y;
  PetscReal      *ritz,*bnd,anorm,beta,norm,*work;
  PetscTruth     breakdown;
  char           *conv;

  PetscFunctionBegin;
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&bnd);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(int),&perm);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(char),&conv);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal)+ncv*sizeof(int),&work);CHKERRQ(ierr);

  /* The first Lanczos vector is the normalized initial vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  
  anorm = -1.0;
  nconv = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;
    /* Compute an ncv-step Lanczos factorization */
    m = ncv;
    ierr = EPSBasicLanczos(eps,T,eps->V,nconv,&m,f,&beta,&breakdown,anorm);CHKERRQ(ierr);

    /* Compute eigenvalues and eigenvectors Y of the tridiagonal block */
    n = m - nconv;
    ierr = EPSDenseTridiagonal(n,T+nconv*(ncv+1),ncv,ritz,Y);CHKERRQ(ierr);

    /* Estimate ||A|| */
    for (i=0;i<n;i++) 
      if (PetscAbsReal(ritz[i]) > anorm) anorm = PetscAbsReal(ritz[i]);
    
    /* Compute residual norm estimates as beta*abs(Y(m,:)) + eps*||A|| */
    for (i=0;i<n;i++)
      bnd[i] = beta*PetscAbsScalar(Y[i*n+n-1]) + PETSC_MACHINE_EPSILON*anorm;

    /* Sort eigenvalues according to eps->which */
    if (eps->which == EPS_SMALLEST_REAL) {
      /* LAPACK function has already ordered the eigenvalues and eigenvectors */
      for (i=0;i<n;i++)
        perm[i] = i;
    } else {
      ierr = EPSSortEigenvaluesReal(n,ritz,eps->which,n,perm,work);CHKERRQ(ierr);
    }

    /* Look for converged eigenpairs */
    k = nconv;
    for (i=0;i<n;i++) {
      eps->eigr[k] = ritz[perm[i]];
      eps->errest[k] = bnd[perm[i]] / PetscAbsScalar(eps->eigr[k]);    
      if (eps->errest[k] < eps->tol) {
	      
	if (lanczos->reorthog == EPSLANCZOS_REORTHOG_LOCAL) {
          if (i>0 && PetscAbsScalar((eps->eigr[k]-ritz[perm[i-1]])/eps->eigr[k]) < eps->tol) {
  	    /* Discard repeated eigenvalues */
            conv[i] = 'R';
	    continue;
 	  }
	}
	  
	ierr = VecSet(eps->AV[k],0.0);CHKERRQ(ierr);
	ierr = VecMAXPY(eps->AV[k],n,Y+perm[i]*n,eps->V+nconv);CHKERRQ(ierr);

	if (lanczos->reorthog == EPSLANCZOS_REORTHOG_LOCAL) {
	  /* normalize locked vector and compute residual norm */
	  ierr = VecNorm(eps->AV[k],NORM_2,&norm);CHKERRQ(ierr);
          ierr = VecScale(eps->AV[k],1.0/norm);CHKERRQ(ierr);
	  ierr = STApply(eps->OP,eps->AV[k],f);CHKERRQ(ierr);
	  ierr = VecAXPY(f,-eps->eigr[k],eps->AV[k]);CHKERRQ(ierr);
	  ierr = VecNorm(f,NORM_2,&norm);CHKERRQ(ierr);
	  eps->errest[k] = norm / PetscAbsScalar(eps->eigr[k]);
          if (eps->errest[k] >= eps->tol) {
	    conv[i] = 'S';
	    continue;
	  }
	}
	  
        conv[i] = 'C';
        k++;
      } else conv[i] = 'N';
    }

    /* Look for non-converged eigenpairs */
    j = k;
    restart = -1;
    for (i=0;i<n;i++) {
      if (conv[i] != 'C') {
        if (restart == -1 && conv[i] == 'N') restart = i;
        eps->eigr[j] = ritz[perm[i]];
        eps->errest[j] = bnd[perm[i]] / ritz[perm[i]];
        j++;
      } 
    }

    if (breakdown) {
      restart = -1;
      PetscInfo2(eps,"Breakdown in Lanczos method (it=%i norm=%g)\n",eps->its,beta);
    }
    
    if (k<eps->nev) {
      if (restart != -1) {
	/* Use first non-converged vector for restarting */
	ierr = VecSet(eps->AV[k],0.0);CHKERRQ(ierr);
	ierr = VecMAXPY(eps->AV[k],n,Y+perm[restart]*n,eps->V+nconv);CHKERRQ(ierr);
	ierr = VecCopy(eps->AV[k],eps->V[k]);CHKERRQ(ierr);
      }
    }
    
    /* Copy converged vectors to V */
    for (i=nconv;i<k;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }

    if (k<eps->nev) {
      if (restart == -1) {
	/* Use random vector for restarting */
	PetscInfo(eps,"Using random vector for restart\n");
	ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
      } else if (lanczos->reorthog == EPSLANCZOS_REORTHOG_LOCAL) {
        /* Reorthonormalize restart vector */
	ierr = IPOrthogonalize(eps->ip,eps->nds+k,PETSC_NULL,eps->DSV,eps->V[k],PETSC_NULL,&norm,&breakdown,eps->work[0]);CHKERRQ(ierr);
	ierr = VecScale(eps->V[k],1.0/norm);CHKERRQ(ierr);
      } else breakdown = PETSC_FALSE;
      if (breakdown) {
	eps->reason = EPS_DIVERGED_BREAKDOWN;
	PetscInfo(eps,"Unable to generate more start vectors\n");
      }
    }

    EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nconv+n);
    nconv = k;
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
  }
  
  eps->nconv = nconv;

  ierr = PetscFree(ritz);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(bnd);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  ierr = PetscFree(conv);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const char *lanczoslist[6] = { "local", "full", "selective", "periodic", "partial" , "delayed" };

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_LANCZOS"
PetscErrorCode EPSSetFromOptions_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS *)eps->data;
  PetscTruth     flg;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("LANCZOS options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-eps_lanczos_reorthog","Lanczos reorthogonalization","EPSLanczosSetReorthog",lanczoslist,6,lanczoslist[lanczos->reorthog],&i,&flg);CHKERRQ(ierr);
  if (flg) lanczos->reorthog = (EPSLanczosReorthogType)i;
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosSetReorthog_LANCZOS"
PetscErrorCode EPSLanczosSetReorthog_LANCZOS(EPS eps,EPSLanczosReorthogType reorthog)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;

  PetscFunctionBegin;
  switch (reorthog) {
    case EPSLANCZOS_REORTHOG_LOCAL:
    case EPSLANCZOS_REORTHOG_FULL:
    case EPSLANCZOS_REORTHOG_DELAYED:
    case EPSLANCZOS_REORTHOG_SELECTIVE:
    case EPSLANCZOS_REORTHOG_PERIODIC:
    case EPSLANCZOS_REORTHOG_PARTIAL:
      lanczos->reorthog = reorthog;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid reorthogonalization type");
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosSetReorthog"
/*@
   EPSLanczosSetReorthog - Sets the type of reorthogonalization used during the Lanczos
   iteration. 

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  reorthog - the type of reorthogonalization

   Options Database Key:
.  -eps_lanczos_reorthog - Sets the reorthogonalization type (either 'local', 'selective',
                         'periodic', 'partial', 'full' or 'delayed')
   
   Level: advanced

.seealso: EPSLanczosGetReorthog(), EPSLanczosReorthogType
@*/
PetscErrorCode EPSLanczosSetReorthog(EPS eps,EPSLanczosReorthogType reorthog)
{
  PetscErrorCode ierr, (*f)(EPS,EPSLanczosReorthogType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSLanczosSetReorthog_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,reorthog);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosGetReorthog_LANCZOS"
PetscErrorCode EPSLanczosGetReorthog_LANCZOS(EPS eps,EPSLanczosReorthogType *reorthog)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscFunctionBegin;
  *reorthog = lanczos->reorthog;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosGetReorthog"
/*@C
   EPSLanczosGetReorthog - Gets the type of reorthogonalization used during the Lanczos
   iteration. 

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Input Parameter:
.  reorthog - the type of reorthogonalization

   Level: advanced

.seealso: EPSLanczosSetReorthog(), EPSLanczosReorthogType
@*/
PetscErrorCode EPSLanczosGetReorthog(EPS eps,EPSLanczosReorthogType *reorthog)
{
  PetscErrorCode ierr, (*f)(EPS,EPSLanczosReorthogType*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSLanczosGetReorthog_C",(void (**)())&f);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSLANCZOS",((PetscObject)viewer)->type_name);
  }  
  ierr = PetscViewerASCIIPrintf(viewer,"reorthogonalization: %s\n",lanczoslist[lanczos->reorthog]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
EXTERN PetscErrorCode EPSSolve_TS_LANCZOS(EPS);
*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_LANCZOS"
PetscErrorCode EPSCreate_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_LANCZOS,&lanczos);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_LANCZOS));
  eps->data                      = (void *) lanczos;
  eps->ops->solve                = EPSSolve_LANCZOS;
/*  eps->ops->solvets              = EPSSolve_TS_LANCZOS;*/
  eps->ops->setup                = EPSSetUp_LANCZOS;
  eps->ops->setfromoptions       = EPSSetFromOptions_LANCZOS;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = EPSView_LANCZOS;
  eps->ops->backtransform        = EPSBackTransform_Default;
  /*if (eps->solverclass==EPS_TWO_SIDE)
       eps->ops->computevectors       = EPSComputeVectors_Schur;
  else*/ eps->ops->computevectors       = EPSComputeVectors_Hermitian;
  lanczos->reorthog              = EPSLANCZOS_REORTHOG_LOCAL;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosSetReorthog_C","EPSLanczosSetReorthog_LANCZOS",EPSLanczosSetReorthog_LANCZOS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosGetReorthog_C","EPSLanczosGetReorthog_LANCZOS",EPSLanczosGetReorthog_LANCZOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

