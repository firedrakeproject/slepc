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

   Last update: Feb 2009

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

PetscErrorCode EPSSolve_Lanczos(EPS);

typedef struct {
  EPSLanczosReorthogType reorthog;
  Vec *AV;
} EPS_LANCZOS;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_Lanczos"
PetscErrorCode EPSSetUp_Lanczos(EPS eps)
{
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS*)eps->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (eps->ncv) { /* ncv set */
    if (eps->ncv<eps->nev) SETERRQ(((PetscObject)eps)->comm,1,"The value of ncv must be at least nev"); 
  } else if (eps->mpd) { /* mpd set */
    eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (eps->nev<500) eps->ncv = PetscMin(eps->n,PetscMax(2*eps->nev,eps->nev+15));
    else {
      eps->mpd = 500;
      eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd);
    }
  }
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (eps->ncv>eps->nev+eps->mpd) SETERRQ(((PetscObject)eps)->comm,1,"The value of ncv must not be larger than nev+mpd"); 
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);

  if (!eps->which) { ierr = EPSDefaultSetWhich(eps);CHKERRQ(ierr); }
  switch (eps->which) {
    case EPS_LARGEST_IMAGINARY:
    case EPS_SMALLEST_IMAGINARY:
    case EPS_TARGET_IMAGINARY:
      SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");
    default: ; /* default case to remove warning */
  }
  if (!eps->ishermitian) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");
  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Unsupported extraction type");
  if (eps->arbit_func) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  if (lanczos->reorthog == EPS_LANCZOS_REORTHOG_SELECTIVE) {
    ierr = VecDuplicateVecs(eps->t,eps->ncv,&lanczos->AV);CHKERRQ(ierr);
  }
  ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
  ierr = DSSetCompact(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(eps->ds,eps->ncv);CHKERRQ(ierr);
  if (lanczos->reorthog == EPS_LANCZOS_REORTHOG_LOCAL) {
    ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  } else {
    ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);
  }

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Left vectors not supported in this solver");
  eps->ops->solve = EPSSolve_Lanczos;
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
static PetscErrorCode EPSLocalLanczos(EPS eps,PetscReal *alpha,PetscReal *beta,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m = *M;
  PetscReal      norm;
  PetscBool      *which,lwhich[100];
  PetscScalar    *hwork,lhwork[100];
  
  PetscFunctionBegin;  
  if (m > 100) {
    ierr = PetscMalloc(sizeof(PetscBool)*m,&which);CHKERRQ(ierr);
    ierr = PetscMalloc(m*sizeof(PetscScalar),&hwork);CHKERRQ(ierr);
  } else {
    which = lwhich;
    hwork = lhwork;
  }
  for (i=0;i<k;i++)
    which[i] = PETSC_TRUE;

  for (j=k;j<m-1;j++) {
    ierr = STApply(eps->st,V[j],V[j+1]);CHKERRQ(ierr);
    which[j] = PETSC_TRUE;
    if (j-2>=k) which[j-2] = PETSC_FALSE;
    ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,j+1,which,V,V[j+1],hwork,&norm,breakdown);CHKERRQ(ierr);
    alpha[j] = PetscRealPart(hwork[j]);
    beta[j] = norm;
    if (*breakdown) {
      *M = j+1;
      if (m > 100) {
        ierr = PetscFree(which);CHKERRQ(ierr);
        ierr = PetscFree(hwork);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    } else {
      ierr = VecScale(V[j+1],1.0/norm);CHKERRQ(ierr);
    }
  }
  ierr = STApply(eps->st,V[m-1],f);CHKERRQ(ierr);
  ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,m,PETSC_NULL,V,f,hwork,&norm,PETSC_NULL);CHKERRQ(ierr);
  alpha[m-1] = PetscRealPart(hwork[m-1]); 
  beta[m-1] = norm;

  if (m > 100) {
    ierr = PetscFree(which);CHKERRQ(ierr);
    ierr = PetscFree(hwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DenseTridiagonal"
/*
   DenseTridiagonal - Solves a real tridiagonal Hermitian Eigenvalue Problem.

   Input Parameters:
+  n   - dimension of the eigenproblem
.  D   - pointer to the array containing the diagonal elements
-  E   - pointer to the array containing the off-diagonal elements

   Output Parameters:
+  w  - pointer to the array to store the computed eigenvalues
-  V  - pointer to the array to store the eigenvectors

   Notes:
   If V is PETSC_NULL then the eigenvectors are not computed.

   This routine use LAPACK routines xSTEVR.

*/
static PetscErrorCode DenseTridiagonal(PetscInt n_,PetscReal *D,PetscReal *E,PetscReal *w,PetscScalar *V)
{
#if defined(SLEPC_MISSING_LAPACK_STEVR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"STEVR - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscReal      abstol = 0.0,vl,vu,*work;
  PetscBLASInt   il,iu,m,*isuppz,n,lwork,*iwork,liwork,info;
  const char     *jobz;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       i,j;
  PetscReal      *VV;
#endif
  
  PetscFunctionBegin;
  n = PetscBLASIntCast(n_);
  lwork = PetscBLASIntCast(20*n_);
  liwork = PetscBLASIntCast(10*n_);
  if (V) {
    jobz = "V";
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(n*n*sizeof(PetscReal),&VV);CHKERRQ(ierr);
#endif
  } else jobz = "N";
  ierr = PetscMalloc(2*n*sizeof(PetscBLASInt),&isuppz);CHKERRQ(ierr);
  ierr = PetscMalloc(lwork*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(liwork*sizeof(PetscBLASInt),&iwork);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  LAPACKstevr_(jobz,"A",&n,D,E,&vl,&vu,&il,&iu,&abstol,&m,w,VV,&n,isuppz,work,&lwork,iwork,&liwork,&info);
#else
  LAPACKstevr_(jobz,"A",&n,D,E,&vl,&vu,&il,&iu,&abstol,&m,w,V,&n,isuppz,work,&lwork,iwork,&liwork,&info);
#endif
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DSTEVR %d",info);
#if defined(PETSC_USE_COMPLEX)
  if (V) {
    for (i=0;i<n;i++) 
      for (j=0;j<n;j++)
        V[i*n+j] = VV[i*n+j];
    ierr = PetscFree(VV);CHKERRQ(ierr);
  }
#endif
  ierr = PetscFree(isuppz);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSelectiveLanczos"
/*
   EPSSelectiveLanczos - Selective reorthogonalization.
*/
static PetscErrorCode EPSSelectiveLanczos(EPS eps,PetscReal *alpha,PetscReal *beta,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscBool *breakdown,PetscReal anorm)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS*)eps->data;
  PetscInt       i,j,m = *M,n,nritz=0,nritzo;
  PetscReal      *d,*e,*ritz,norm;
  PetscScalar    *Y,*hwork,lhwork[100];
  PetscBool      *which,lwhich[100];

  PetscFunctionBegin;
  ierr = PetscMalloc(m*sizeof(PetscReal),&d);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscReal),&e);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
  ierr = PetscMalloc(m*m*sizeof(PetscScalar),&Y);CHKERRQ(ierr);
  if (m > 100) {
    ierr = PetscMalloc(sizeof(PetscBool)*m,&which);CHKERRQ(ierr);
    ierr = PetscMalloc(m*sizeof(PetscScalar),&hwork);CHKERRQ(ierr);
  } else {
    which = lwhich;
    hwork = lhwork;
  }
  for (i=0;i<k;i++)
    which[i] = PETSC_TRUE;

  for (j=k;j<m;j++) {
    /* Lanczos step */
    ierr = STApply(eps->st,V[j],f);CHKERRQ(ierr);
    which[j] = PETSC_TRUE;
    if (j-2>=k) which[j-2] = PETSC_FALSE;
    ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,j+1,which,V,f,hwork,&norm,breakdown);CHKERRQ(ierr);
    alpha[j] = PetscRealPart(hwork[j]);
    beta[j] = norm;
    if (*breakdown) {
      *M = j+1;
      break;
    }

    /* Compute eigenvalues and eigenvectors Y of the tridiagonal block */
    n = j-k+1;
    for (i=0;i<n;i++) {
      d[i] = alpha[i+k];
      e[i] = beta[i+k];
    }
    ierr = DenseTridiagonal(n,d,e,ritz,Y);CHKERRQ(ierr);
    
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
          ierr = SlepcVecMAXPBY(lanczos->AV[nritz],0.0,1.0,n,Y+i*n,V+k);CHKERRQ(ierr);
          nritz++;
        }
      }
    }

    if (nritz > 0) {
      ierr = IPOrthogonalize(eps->ip,0,PETSC_NULL,nritz,PETSC_NULL,lanczos->AV,f,hwork,&norm,breakdown);CHKERRQ(ierr);
      if (*breakdown) {
        *M = j+1;
        break;
      }
    }
    
    if (j<m-1) {
      ierr = VecScale(f,1.0 / norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  
  ierr = PetscFree(d);CHKERRQ(ierr);
  ierr = PetscFree(e);CHKERRQ(ierr);
  ierr = PetscFree(ritz);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  if (m > 100) {
    ierr = PetscFree(which);CHKERRQ(ierr);
    ierr = PetscFree(hwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "update_omega"
static void update_omega(PetscReal *omega,PetscReal *omega_old,PetscInt j,PetscReal *alpha,PetscReal *beta,PetscReal eps1,PetscReal anorm)
{
  PetscInt       k;
  PetscReal      T,binv;

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
  if (omega_old[0] > 0) omega_old[0] = binv*(omega_old[0] + T);
  else omega_old[0] = binv*(omega_old[0] - T);  
  
  /* Update remaining components. */
  for (k=1;k<j-1;k++) {
    omega_old[k] = beta[k+1]*omega[k+1] + (alpha[k]-alpha[j])*omega[k] + beta[k]*omega[k-1] - beta[j]*omega_old[k];
    if (omega_old[k] > 0) omega_old[k] = binv*(omega_old[k] + T);       
    else omega_old[k] = binv*(omega_old[k] - T);       
  }
  omega_old[j-1] = binv*T;
  
  /* Swap omega and omega_old. */
  for (k=0;k<j;k++) {
    omega[k] = omega_old[k];
    omega_old[k] = omega[k];
  }
  omega[j] = eps1;  
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "compute_int"
static void compute_int(PetscBool *which,PetscReal *mu,PetscInt j,PetscReal delta,PetscReal eta)
{
  PetscInt  i,k,maxpos;
  PetscReal max;
  PetscBool found;
  
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
static PetscErrorCode EPSPartialLanczos(EPS eps,PetscReal *alpha,PetscReal *beta,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscBool *breakdown,PetscReal anorm)
{
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS*)eps->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = *M;
  PetscReal      norm,*omega,lomega[100],*omega_old,lomega_old[100],eps1,delta,eta;
  PetscBool      *which,lwhich[100],*which2,lwhich2[100];
  PetscBool      reorth = PETSC_FALSE,force_reorth = PETSC_FALSE;
  PetscBool      fro = PETSC_FALSE,estimate_anorm = PETSC_FALSE;
  PetscScalar    *hwork,lhwork[100];

  PetscFunctionBegin;
  if (m>100) {
    ierr = PetscMalloc(m*sizeof(PetscReal),&omega);CHKERRQ(ierr);
    ierr = PetscMalloc(m*sizeof(PetscReal),&omega_old);CHKERRQ(ierr);
  } else {
    omega = lomega;
    omega_old = lomega_old;
  }
  if (m > 100) {
    ierr = PetscMalloc(sizeof(PetscBool)*m,&which);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool)*m,&which2);CHKERRQ(ierr);
    ierr = PetscMalloc(m*sizeof(PetscScalar),&hwork);CHKERRQ(ierr);
  } else {
    which = lwhich;
    which2 = lwhich2;
    hwork = lhwork;
  }

  eps1 = PetscSqrtReal((PetscReal)eps->n)*PETSC_MACHINE_EPSILON/2;
  delta = PETSC_SQRT_MACHINE_EPSILON/PetscSqrtReal((PetscReal)eps->ncv);
  eta = pow(PETSC_MACHINE_EPSILON,3.0/4.0)/PetscSqrtReal((PetscReal)eps->ncv);
  if (anorm < 0.0) {
    anorm = 1.0;
    estimate_anorm = PETSC_TRUE;
  }
  for (i=0;i<m-k;i++) 
    omega[i] = omega_old[i] = 0.0;
  for (i=0;i<k;i++)
    which[i] = PETSC_TRUE;  
  
  for (j=k;j<m;j++) {
    ierr = STApply(eps->st,V[j],f);CHKERRQ(ierr);
    if (fro) {
      /* Lanczos step with full reorthogonalization */
      ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,j+1,PETSC_NULL,V,f,hwork,&norm,breakdown);CHKERRQ(ierr);      
      alpha[j] = PetscRealPart(hwork[j]);
    } else {
      /* Lanczos step */
      which[j] = PETSC_TRUE;
      if (j-2>=k) which[j-2] = PETSC_FALSE;
      ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,j+1,which,V,f,hwork,&norm,breakdown);CHKERRQ(ierr);
      alpha[j] = PetscRealPart(hwork[j]);
      beta[j] = norm;
      
      /* Estimate ||A|| if needed */ 
      if (estimate_anorm) {
        if (j>k) anorm = PetscMax(anorm,PetscAbsReal(alpha[j])+norm+beta[j-1]);
        else anorm = PetscMax(anorm,PetscAbsReal(alpha[j])+norm);
      }

      /* Check if reorthogonalization is needed */
      reorth = PETSC_FALSE;
      if (j>k) {      
        update_omega(omega,omega_old,j,alpha,beta-1,eps1,anorm);
        for (i=0;i<j-k;i++)
          if (PetscAbsScalar(omega[i]) > delta) reorth = PETSC_TRUE;
      }

      if (reorth || force_reorth) {
        if (lanczos->reorthog == EPS_LANCZOS_REORTHOG_PERIODIC) {
          /* Periodic reorthogonalization */
          if (force_reorth) force_reorth = PETSC_FALSE;
          else force_reorth = PETSC_TRUE;
          ierr = IPOrthogonalize(eps->ip,0,PETSC_NULL,j-k,PETSC_NULL,V+k,f,hwork,&norm,breakdown);CHKERRQ(ierr);
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
          ierr = IPOrthogonalize(eps->ip,0,PETSC_NULL,j-k,which2,V+k,f,hwork,&norm,breakdown);CHKERRQ(ierr);
        }
      }
    }
    
    if (*breakdown || norm < eps->n*anorm*PETSC_MACHINE_EPSILON) {
      *M = j+1;
      break;
    }
    if (!fro && norm*delta < anorm*eps1) {
      fro = PETSC_TRUE;
      ierr = PetscInfo1(eps,"Switching to full reorthogonalization at iteration %D\n",eps->its);CHKERRQ(ierr);
    }
    
    beta[j] = norm;
    if (j<m-1) {
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }

  if (m>100) {
    ierr = PetscFree(omega);CHKERRQ(ierr);
    ierr = PetscFree(omega_old);CHKERRQ(ierr);
    ierr = PetscFree(which);CHKERRQ(ierr);
    ierr = PetscFree(which2);CHKERRQ(ierr);
    ierr = PetscFree(hwork);CHKERRQ(ierr);
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
static PetscErrorCode EPSBasicLanczos(EPS eps,PetscReal *alpha,PetscReal *beta,Vec *V,PetscInt k,PetscInt *m,Vec f,PetscBool *breakdown,PetscReal anorm)
{
  EPS_LANCZOS        *lanczos = (EPS_LANCZOS*)eps->data;
  PetscScalar        *T;
  PetscInt           i,n=*m;
  PetscReal          betam;
  PetscErrorCode     ierr;
  IPOrthogRefineType orthog_ref;

  PetscFunctionBegin;
  switch (lanczos->reorthog) {
    case EPS_LANCZOS_REORTHOG_LOCAL:
      ierr = EPSLocalLanczos(eps,alpha,beta,V,k,m,f,breakdown);CHKERRQ(ierr);
      break;
    case EPS_LANCZOS_REORTHOG_SELECTIVE:
      ierr = EPSSelectiveLanczos(eps,alpha,beta,V,k,m,f,breakdown,anorm);CHKERRQ(ierr);
      break;
    case EPS_LANCZOS_REORTHOG_FULL:
      ierr = EPSFullLanczos(eps,alpha,beta,V,k,m,f,breakdown);CHKERRQ(ierr);
      break;
    case EPS_LANCZOS_REORTHOG_PARTIAL:
    case EPS_LANCZOS_REORTHOG_PERIODIC:
      ierr = EPSPartialLanczos(eps,alpha,beta,V,k,m,f,breakdown,anorm);CHKERRQ(ierr);
      break;
    case EPS_LANCZOS_REORTHOG_DELAYED:
      ierr = PetscMalloc(n*n*sizeof(PetscScalar),&T);CHKERRQ(ierr);
      ierr = IPGetOrthogonalization(eps->ip,PETSC_NULL,&orthog_ref,PETSC_NULL);CHKERRQ(ierr);
      if (orthog_ref == IP_ORTHOG_REFINE_NEVER) {
        ierr = EPSDelayedArnoldi1(eps,T,n,V,k,m,f,&betam,breakdown);CHKERRQ(ierr);       
      } else {
        ierr = EPSDelayedArnoldi(eps,T,n,V,k,m,f,&betam,breakdown);CHKERRQ(ierr);
      }
      for (i=k;i<n-1;i++) {
        alpha[i] = PetscRealPart(T[n*i+i]);
        beta[i] = PetscRealPart(T[n*i+i+1]);
      }
      alpha[n-1] = PetscRealPart(T[n*(n-1)+n-1]);
      beta[n-1] = betam;
      ierr = PetscFree(T);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid reorthogonalization type"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_Lanczos"
PetscErrorCode EPSSolve_Lanczos(EPS eps)
{
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS*)eps->data;
  PetscErrorCode ierr;
  PetscInt       nconv,i,j,k,l,x,n,*perm,restart,ncv=eps->ncv,r,ld;
  Vec            w=eps->work[1],f=eps->work[0];
  PetscScalar    *Y,*ritz,stmp;
  PetscReal      *d,*e,*bnd,anorm,beta,norm,rtmp,resnorm;
  PetscBool      breakdown;
  char           *conv,ctmp;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscScalar),&ritz);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&bnd);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscInt),&perm);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(char),&conv);CHKERRQ(ierr);

  /* The first Lanczos vector is the normalized initial vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  
  anorm = -1.0;
  nconv = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an ncv-step Lanczos factorization */
    n = PetscMin(nconv+eps->mpd,ncv);
    ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&d);CHKERRQ(ierr);
    e = d + ld;
    ierr = EPSBasicLanczos(eps,d,e,eps->V,nconv,&n,f,&breakdown,anorm);CHKERRQ(ierr);
    beta = e[n-1];
    ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&d);CHKERRQ(ierr);
    ierr = DSSetDimensions(eps->ds,n,PETSC_IGNORE,nconv,0);CHKERRQ(ierr);
    ierr = DSSetState(eps->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);

    /* Solve projected problem */ 
    ierr = DSSolve(eps->ds,ritz,PETSC_NULL);CHKERRQ(ierr);
    ierr = DSSort(eps->ds,ritz,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    
    /* Estimate ||A|| */
    for (i=nconv;i<n;i++) 
      anorm = PetscMax(anorm,PetscAbsReal(PetscRealPart(ritz[i])));
    
    /* Compute residual norm estimates as beta*abs(Y(m,:)) + eps*||A|| */
    ierr = DSGetArray(eps->ds,DS_MAT_Q,&Y);CHKERRQ(ierr);
    for (i=nconv;i<n;i++) {
      resnorm = beta*PetscAbsScalar(Y[n-1+i*ld]) + PETSC_MACHINE_EPSILON*anorm;
      ierr = (*eps->conv_func)(eps,ritz[i],eps->eigi[i],resnorm,&bnd[i],eps->conv_ctx);CHKERRQ(ierr);
      if (bnd[i]<eps->tol) conv[i] = 'C';
      else conv[i] = 'N';
    }
    ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Y);CHKERRQ(ierr);

    /* purge repeated ritz values */
    if (lanczos->reorthog == EPS_LANCZOS_REORTHOG_LOCAL)
      for (i=nconv+1;i<n;i++)
        if (conv[i] == 'C')
          if (PetscAbsScalar((ritz[i]-ritz[i-1])/ritz[i]) < eps->tol)
            conv[i] = 'R';

    /* Compute restart vector */
    if (breakdown) {
      ierr = PetscInfo2(eps,"Breakdown in Lanczos method (it=%D norm=%G)\n",eps->its,beta);CHKERRQ(ierr);
    } else {
      restart = nconv;
      while (restart<n && conv[restart] != 'N') restart++;
      if (restart >= n) {
        breakdown = PETSC_TRUE;
      } else {
        for (i=restart+1;i<n;i++)
          if (conv[i] == 'N') {
            ierr = (*eps->which_func)(ritz[restart],0.0,ritz[i],0.0,&r,eps->which_ctx);CHKERRQ(ierr);
            if (r>0) restart = i;
          }
        ierr = DSGetArray(eps->ds,DS_MAT_Q,&Y);CHKERRQ(ierr);
        ierr = SlepcVecMAXPBY(f,0.0,1.0,n-nconv,Y+restart*ld+nconv,eps->V+nconv);CHKERRQ(ierr);
        ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Y);CHKERRQ(ierr);
      }
    }

    /* Count and put converged eigenvalues first */
    for (i=nconv;i<n;i++) perm[i] = i;
    for (k=nconv;k<n;k++)
      if (conv[perm[k]] != 'C') {
        j = k + 1;
        while (j<n && conv[perm[j]] != 'C') j++;
        if (j>=n) break;
        l = perm[k]; perm[k] = perm[j]; perm[j] = l;
      }

    /* Sort eigenvectors according to permutation */
    ierr = DSGetArray(eps->ds,DS_MAT_Q,&Y);CHKERRQ(ierr);
    for (i=nconv;i<k;i++) {
      x = perm[i];
      if (x != i) {
        j = i + 1;
        while (perm[j] != i) j++;
        /* swap eigenvalues i and j */
        stmp = ritz[x]; ritz[x] = ritz[i]; ritz[i] = stmp;
        rtmp = bnd[x]; bnd[x] = bnd[i]; bnd[i] = rtmp;
        ctmp = conv[x]; conv[x] = conv[i]; conv[i] = ctmp;
        perm[j] = x; perm[i] = i;
        /* swap eigenvectors i and j */
        for (l=0;l<n;l++) {
          stmp = Y[l+x*ld]; Y[l+x*ld] = Y[l+i*ld]; Y[l+i*ld] = stmp;
        }
      }
    }
    
    /* compute converged eigenvectors */
    ierr = SlepcUpdateVectors(n,eps->V,nconv,nconv+k,Y,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Y);CHKERRQ(ierr);
    
    /* purge spurious ritz values */
    if (lanczos->reorthog == EPS_LANCZOS_REORTHOG_LOCAL) {
      for (i=nconv;i<k;i++) {
        ierr = VecNorm(eps->V[i],NORM_2,&norm);CHKERRQ(ierr);
        ierr = VecScale(eps->V[i],1.0/norm);CHKERRQ(ierr);
        ierr = STApply(eps->st,eps->V[i],w);CHKERRQ(ierr);
        ierr = VecAXPY(w,-ritz[i],eps->V[i]);CHKERRQ(ierr);
        ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
        ierr = (*eps->conv_func)(eps,ritz[i],eps->eigi[i],norm,&bnd[i],eps->conv_ctx);CHKERRQ(ierr);
        if (bnd[i]>=eps->tol) conv[i] = 'S';
      }
      for (i=nconv;i<k;i++)
        if (conv[i] != 'C') {
          j = i + 1;
          while (j<k && conv[j] != 'C') j++;
          if (j>=k) break;
          /* swap eigenvalues i and j */
          stmp = ritz[j]; ritz[j] = ritz[i]; ritz[i] = stmp;
          rtmp = bnd[j]; bnd[j] = bnd[i]; bnd[i] = rtmp;
          ctmp = conv[j]; conv[j] = conv[i]; conv[i] = ctmp;
          /* swap eigenvectors i and j */
          ierr = VecSwap(eps->V[i],eps->V[j]);CHKERRQ(ierr);
        }
      k = i;
    }
    
    /* store ritz values and estimated errors */
    for (i=nconv;i<n;i++) {
      eps->eigr[i] = ritz[i];
      eps->errest[i] = bnd[i];
    }
    ierr = EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,n);CHKERRQ(ierr);
    nconv = k;
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    if (eps->reason == EPS_CONVERGED_ITERATING) { /* copy restart vector */
      if (lanczos->reorthog == EPS_LANCZOS_REORTHOG_LOCAL && !breakdown) {
        /* Reorthonormalize restart vector */
        ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,nconv,PETSC_NULL,eps->V,f,PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr);
        ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      }
      if (breakdown) {
        /* Use random vector for restarting */
        ierr = PetscInfo(eps,"Using random vector for restart\n");CHKERRQ(ierr);
        ierr = EPSGetStartVector(eps,nconv,f,&breakdown);CHKERRQ(ierr);
      }
      if (breakdown) { /* give up */
        eps->reason = EPS_DIVERGED_BREAKDOWN;
        ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
      } else {
        ierr = VecCopy(f,eps->V[nconv]);CHKERRQ(ierr);
      }
    }    
  }
  
  eps->nconv = nconv;

  ierr = PetscFree(ritz);CHKERRQ(ierr);
  ierr = PetscFree(bnd);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  ierr = PetscFree(conv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_Lanczos"
PetscErrorCode EPSSetFromOptions_Lanczos(EPS eps)
{
  PetscErrorCode         ierr;
  EPS_LANCZOS            *lanczos = (EPS_LANCZOS*)eps->data;
  PetscBool              flg;
  EPSLanczosReorthogType reorthog;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EPS Lanczos Options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-eps_lanczos_reorthog","Lanczos reorthogonalization","EPSLanczosSetReorthog",EPSLanczosReorthogTypes,(PetscEnum)lanczos->reorthog,(PetscEnum*)&reorthog,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = EPSLanczosSetReorthog(eps,reorthog);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosSetReorthog_Lanczos"
PetscErrorCode EPSLanczosSetReorthog_Lanczos(EPS eps,EPSLanczosReorthogType reorthog)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS*)eps->data;

  PetscFunctionBegin;
  switch (reorthog) {
    case EPS_LANCZOS_REORTHOG_LOCAL:
    case EPS_LANCZOS_REORTHOG_FULL:
    case EPS_LANCZOS_REORTHOG_DELAYED:
    case EPS_LANCZOS_REORTHOG_SELECTIVE:
    case EPS_LANCZOS_REORTHOG_PERIODIC:
    case EPS_LANCZOS_REORTHOG_PARTIAL:
      lanczos->reorthog = reorthog;
      break;
    default:
      SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid reorthogonalization type");
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosSetReorthog"
/*@
   EPSLanczosSetReorthog - Sets the type of reorthogonalization used during the Lanczos
   iteration. 

   Logically Collective on EPS

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(eps,reorthog,2);
  ierr = PetscTryMethod(eps,"EPSLanczosSetReorthog_C",(EPS,EPSLanczosReorthogType),(eps,reorthog));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosGetReorthog_Lanczos"
PetscErrorCode EPSLanczosGetReorthog_Lanczos(EPS eps,EPSLanczosReorthogType *reorthog)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS*)eps->data;

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

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  reorthog - the type of reorthogonalization

   Level: advanced

.seealso: EPSLanczosSetReorthog(), EPSLanczosReorthogType
@*/
PetscErrorCode EPSLanczosGetReorthog(EPS eps,EPSLanczosReorthogType *reorthog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(reorthog,2);
  ierr = PetscTryMethod(eps,"EPSLanczosGetReorthog_C",(EPS,EPSLanczosReorthogType*),(eps,reorthog));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSReset_Lanczos"
PetscErrorCode EPSReset_Lanczos(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS*)eps->data;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(eps->ncv,&lanczos->AV);CHKERRQ(ierr);
  ierr = EPSReset_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_Lanczos"
PetscErrorCode EPSDestroy_Lanczos(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosSetReorthog_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosGetReorthog_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_Lanczos"
PetscErrorCode EPSView_Lanczos(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ1(((PetscObject)eps)->comm,1,"Viewer type %s not supported for EPS Lanczos",((PetscObject)viewer)->type_name);
  ierr = PetscViewerASCIIPrintf(viewer,"  Lanczos: %s reorthogonalization\n",EPSLanczosReorthogTypes[lanczos->reorthog]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_Lanczos"
PetscErrorCode EPSCreate_Lanczos(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,EPS_LANCZOS,&eps->data);CHKERRQ(ierr);
  eps->ops->setup                = EPSSetUp_Lanczos;
  eps->ops->setfromoptions       = EPSSetFromOptions_Lanczos;
  eps->ops->destroy              = EPSDestroy_Lanczos;
  eps->ops->reset                = EPSReset_Lanczos;
  eps->ops->view                 = EPSView_Lanczos;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Hermitian;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosSetReorthog_C","EPSLanczosSetReorthog_Lanczos",EPSLanczosSetReorthog_Lanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosGetReorthog_C","EPSLanczosGetReorthog_Lanczos",EPSLanczosGetReorthog_Lanczos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

