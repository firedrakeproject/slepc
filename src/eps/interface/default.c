/*
     This file contains some simple default routines for common operations.  

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/epsimpl.h"   /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_Default"
PetscErrorCode EPSDestroy_Default(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);

  /* free work vectors */
  ierr = EPSDefaultFreeWork(eps);CHKERRQ(ierr);
  ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBackTransform_Default"
PetscErrorCode EPSBackTransform_Default(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = STBackTransform(eps->OP,eps->nconv,eps->eigr,eps->eigi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_Default"
/*
  EPSComputeVectors_Default - Compute eigenvectors from the vectors
  provided by the eigensolver. This version just copies the vectors
  and is intended for solvers such as power that provide the eigenvector.
 */
PetscErrorCode EPSComputeVectors_Default(EPS eps)
{
  PetscFunctionBegin;
  eps->evecsavailable = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_Hermitian"
/*
  EPSComputeVectors_Hermitian - Copies the Lanczos vectors as eigenvectors
  using purification for generalized eigenproblems.
 */
PetscErrorCode EPSComputeVectors_Hermitian(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      norm;
  Vec            w;

  PetscFunctionBegin;
  if (eps->isgeneralized) {
    /* Purify eigenvectors */
    ierr = VecDuplicate(eps->V[0],&w);CHKERRQ(ierr);
    for (i=0;i<eps->nconv;i++) {
      ierr = VecCopy(eps->V[i],w);CHKERRQ(ierr);
      ierr = STApply(eps->OP,w,eps->V[i]);CHKERRQ(ierr);
      ierr = IPNorm(eps->ip,eps->V[i],&norm);CHKERRQ(ierr);
      ierr = VecScale(eps->V[i],1.0/norm);CHKERRQ(ierr);
    }
    ierr = VecDestroy(w);CHKERRQ(ierr);
  }
  eps->evecsavailable = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_Schur"
/*
  EPSComputeVectors_Schur - Compute eigenvectors from the vectors
  provided by the eigensolver. This version is intended for solvers 
  that provide Schur vectors. Given the partial Schur decomposition
  OP*V=V*T, the following steps are performed:
      1) compute eigenvectors of T: T*Z=Z*D
      2) compute eigenvectors of OP: X=V*Z
  If left eigenvectors are required then also do Z'*Tl=D*Z', Y=W*Z
 */
PetscErrorCode EPSComputeVectors_Schur(EPS eps)
{
#if defined(SLEPC_MISSING_LAPACK_TREVC)
  SETERRQ(PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   ncv,nconv,mout,info,one = 1; 
  PetscScalar    *Z,*work,tmp;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#else 
  PetscReal      normi;
#endif
  PetscReal      norm;
  Vec            w;
  
  PetscFunctionBegin;
  ncv = PetscBLASIntCast(eps->ncv);
  nconv = PetscBLASIntCast(eps->nconv);
  if (eps->ishermitian) {
    ierr = EPSComputeVectors_Hermitian(eps);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscMalloc(nconv*nconv*sizeof(PetscScalar),&Z);CHKERRQ(ierr);
  ierr = PetscMalloc(3*nconv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(nconv*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif

  /* right eigenvectors */
#if !defined(PETSC_USE_COMPLEX)
  LAPACKtrevc_("R","A",PETSC_NULL,&nconv,eps->T,&ncv,PETSC_NULL,&nconv,Z,&nconv,&nconv,&mout,work,&info);
#else
  LAPACKtrevc_("R","A",PETSC_NULL,&nconv,eps->T,&ncv,PETSC_NULL,&nconv,Z,&nconv,&nconv,&mout,work,rwork,&info);
#endif
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);

  /* normalize eigenvectors (when not using purification nor balancing)*/
  if (!(eps->ispositive || (eps->balance!=EPS_BALANCE_NONE && eps->D))) {
    for (i=0;i<eps->nconv;i++) {
#if !defined(PETSC_USE_COMPLEX)
      if (eps->eigi[i] != 0.0) {
        norm = BLASnrm2_(&nconv,Z+i*nconv,&one);
        normi = BLASnrm2_(&nconv,Z+(i+1)*nconv,&one);
        tmp = 1.0 / SlepcAbsEigenvalue(norm,normi);
        BLASscal_(&nconv,&tmp,Z+i*nconv,&one);
        BLASscal_(&nconv,&tmp,Z+(i+1)*nconv,&one);
        i++;     
      } else
#endif
      {
        norm = BLASnrm2_(&nconv,Z+i*nconv,&one);
        tmp = 1.0 / norm;
        BLASscal_(&nconv,&tmp,Z+i*nconv,&one);
      }
    }
  }
  
  /* AV = V * Z */
  ierr = SlepcUpdateVectors(eps->nconv,eps->V,0,eps->nconv,Z,eps->nconv,PETSC_FALSE);CHKERRQ(ierr);

  /* Purify eigenvectors */
  if (eps->ispositive) {
    ierr = VecDuplicate(eps->V[0],&w);CHKERRQ(ierr);
    for (i=0;i<eps->nconv;i++) {
      ierr = VecCopy(eps->V[i],w);CHKERRQ(ierr); 
      ierr = STApply(eps->OP,w,eps->V[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(w);CHKERRQ(ierr);
  }

  /* Fix eigenvectors if balancing was used */
  if (eps->balance!=EPS_BALANCE_NONE && eps->D) {
    for (i=0;i<eps->nconv;i++) {
      ierr = VecPointwiseDivide(eps->V[i],eps->V[i],eps->D);CHKERRQ(ierr);
    }
  }

  /* normalize eigenvectors (when using purification or balancing) */
  if (eps->ispositive || (eps->balance!=EPS_BALANCE_NONE && eps->D)) {
    for (i=0;i<eps->nconv;i++) {
#if !defined(PETSC_USE_COMPLEX)
      if (eps->eigi[i] != 0.0) {
        ierr = VecNorm(eps->V[i],NORM_2,&norm);CHKERRQ(ierr);
        ierr = VecNorm(eps->V[i+1],NORM_2,&normi);CHKERRQ(ierr);
        tmp = 1.0 / SlepcAbsEigenvalue(norm,normi);
        ierr = VecScale(eps->V[i],tmp);CHKERRQ(ierr);
        ierr = VecScale(eps->V[i+1],tmp);CHKERRQ(ierr);
        i++;     
      } else
#endif
      {
        ierr = VecNormalize(eps->V[i],PETSC_NULL);CHKERRQ(ierr);
      }
    }
  }
   
  /* left eigenvectors */
  if (eps->leftvecs) {
#if !defined(PETSC_USE_COMPLEX)
    LAPACKtrevc_("R","A",PETSC_NULL,&nconv,eps->Tl,&ncv,PETSC_NULL,&nconv,Z,&nconv,&nconv,&mout,work,&info);
#else
    LAPACKtrevc_("R","A",PETSC_NULL,&nconv,eps->Tl,&ncv,PETSC_NULL,&nconv,Z,&nconv,&nconv,&mout,work,rwork,&info);
#endif
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);

    /* AW = W * Z */
    ierr = SlepcUpdateVectors(eps->nconv,eps->W,0,eps->nconv,Z,eps->nconv,PETSC_FALSE);CHKERRQ(ierr);
  }
   
  ierr = PetscFree(Z);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  eps->evecsavailable = PETSC_TRUE;
  PetscFunctionReturn(0);
#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultGetWork"
/*
  EPSDefaultGetWork - Gets a number of work vectors.
 */
PetscErrorCode EPSDefaultGetWork(EPS eps, PetscInt nw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (eps->nwork != nw) {
    if (eps->nwork > 0) {
      ierr = VecDestroyVecs(eps->work,eps->nwork); CHKERRQ(ierr);
    }
    eps->nwork = nw;
    ierr = VecDuplicateVecs(eps->V[0],nw,&eps->work); CHKERRQ(ierr);
    ierr = PetscLogObjectParents(eps,nw,eps->work);
  }
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultFreeWork"
/*
  EPSDefaultFreeWork - Free work vectors.
 */
PetscErrorCode EPSDefaultFreeWork(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->work)  {
    ierr = VecDestroyVecs(eps->work,eps->nwork); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultConverged"
/*
  EPSDefaultConverged - Checks convergence with the relative error estimate.
*/
PetscErrorCode EPSDefaultConverged(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal *errest,PetscTruth *conv,void *ctx)
{
  PetscReal w;
  PetscFunctionBegin;
  w = SlepcAbsEigenvalue(eigr,eigi);
  if (w > *errest) *errest = *errest / w;
  if (*errest < eps->tol) *conv = PETSC_TRUE;
  else *conv = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSAbsoluteConverged"
/*
  EPSAbsoluteConverged - Checks convergence with the absolute error estimate.
*/
PetscErrorCode EPSAbsoluteConverged(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal *errest,PetscTruth *conv,void *ctx)
{
  PetscFunctionBegin;
  if (*errest < eps->tol) *conv = PETSC_TRUE;
  else conv = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeTrueResidual"
/*
  EPSComputeTrueResidual - Computes the true residual norm of a given Ritz pair:
    ||r|| = ||A*x - lambda*B*x||
  where lambda is the Ritz value and x is the Ritz vector.

  Real lambda:
    lambda = eigr
    x = V*Z  (V is an array of nv vectors, Z has length nv)

  Complex lambda:
    lambda = eigr+i*eigi
    x = V*Z[0*nv]+i*V*Z[1*nv]  (Z has length 2*nv)
*/
PetscErrorCode EPSComputeTrueResidual(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscScalar *Z,Vec *V,PetscInt nv,PetscReal *resnorm)
{
  PetscErrorCode ierr;
  Vec            x,y,z;
  PetscReal      w,norm;
  
  PetscFunctionBegin;
  
  /* allocate workspace */
  ierr = VecDuplicate(V[0],&x);CHKERRQ(ierr);
  ierr = VecDuplicate(V[0],&y);CHKERRQ(ierr);
  if (!eps->ishermitian && eps->ispositive) { ierr = VecDuplicate(V[0],&z);CHKERRQ(ierr); }

  /* compute eigenvalue */
  w = SlepcAbsEigenvalue(eigr,eigi);

  /* compute eigenvector */
  ierr = SlepcVecMAXPBY(x,0.0,1.0,nv,Z,V);CHKERRQ(ierr);

  /* purify eigenvector in positive generalized problems */
  if (eps->ispositive) {
    ierr = STApply(eps->OP,x,y);CHKERRQ(ierr);
    if (eps->ishermitian) {
      ierr = IPNorm(eps->ip,y,&norm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);          
    } 
    ierr = VecScale(y,1.0/norm);CHKERRQ(ierr);
    ierr = VecCopy(y,x);CHKERRQ(ierr);
  }
  /* fix eigenvector if balancing is used */
  if (!eps->ishermitian && eps->balance!=EPS_BALANCE_NONE && eps->D) {
    ierr = VecPointwiseDivide(x,x,eps->D);CHKERRQ(ierr);
    ierr = VecNormalize(x,&norm);CHKERRQ(ierr);
  }
#ifndef PETSC_USE_COMPLEX      
  /* compute imaginary part of eigenvector */
  if (!eps->ishermitian && eigi != 0.0) {
    ierr = SlepcVecMAXPBY(y,0.0,1.0,nv,Z+nv,V);CHKERRQ(ierr);
    if (eps->ispositive) {
      ierr = STApply(eps->OP,y,z);CHKERRQ(ierr);
      ierr = VecNorm(z,NORM_2,&norm);CHKERRQ(ierr);          
      ierr = VecScale(z,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(z,y);CHKERRQ(ierr);
    }
    if (eps->balance!=EPS_BALANCE_NONE && eps->D) {
      ierr = VecPointwiseDivide(y,y,eps->D);CHKERRQ(ierr);
      ierr = VecNormalize(y,&norm);CHKERRQ(ierr);
    }
  }
#endif
  /* compute relative error and update convergence flag */
  ierr = EPSComputeResidualNorm_Private(eps,eigr,eigi,x,y,resnorm);

  /* free workspace */
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  if (!eps->ishermitian && eps->ispositive) { ierr = VecDestroy(z);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSResidualConverged"
/*
  EPSResidualConverged - Checks convergence with the true relative residual for 
  each eigenpair whose error estimate is lower than the tolerance.
*/
PetscErrorCode EPSResidualConverged(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal *errest,PetscTruth *conv,void *ctx)
{
//  PetscErrorCode ierr;
//  Vec            x,y,z;
//  PetscInt       i;
//  PetscScalar    re,im;
//  PetscReal      w,norm;
  
  PetscFunctionBegin;
  if (!eps->Z)
    SETERRQ(PETSC_ERR_SUP,"Residual convergence test not supported in this solver");
  
  /* allocate workspace */
//  ierr = VecDuplicate(eps->V[0],&x);CHKERRQ(ierr);
//  ierr = VecDuplicate(eps->V[0],&y);CHKERRQ(ierr);
//  if (!eps->ishermitian && eps->ispositive) { ierr = VecDuplicate(eps->V[0],&z);CHKERRQ(ierr); }

  /* compute residual norm for eigenvalues with relative error below tolerance */
//  for (i=k; i<n; i++) {
    /* compute eigenvalue */
//    re = eigr[i]; im = eigi[i];
//    ierr = STBackTransform(eps->OP,1,&re,&im);CHKERRQ(ierr);
//    w = SlepcAbsEigenvalue(re,im);
//    if (w > errest[i]) errest[i] = errest[i] / w;
//    conv[i] = PETSC_FALSE;
//    if (errest[i] < eps->tol) {
      /* compute eigenvector */
//      if (eps->ishermitian) {
//        ierr = SlepcVecMAXPBY(x,0.0,1.0,n-eps->nconv,eps->Z+(i-eps->nconv)*eps->ldz,eps->V+eps->nconv);CHKERRQ(ierr);
//      } else {
//        ierr = SlepcVecMAXPBY(x,0.0,1.0,n,eps->Z+i*eps->ldz,eps->V);CHKERRQ(ierr);
//      }
      /* purify eigenvector in positive generalized problems */
//      if (eps->ispositive) {
//        ierr = STApply(eps->OP,x,y);CHKERRQ(ierr);
//        if (eps->ishermitian) {
//          ierr = IPNorm(eps->ip,y,&norm);CHKERRQ(ierr);
//        } else {
//          ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);          
//        } 
//        ierr = VecScale(y,1.0/norm);CHKERRQ(ierr);
//        ierr = VecCopy(y,x);CHKERRQ(ierr);
//      }
      /* fix eigenvector if balancing is used */
//      if (!eps->ishermitian && eps->balance!=EPS_BALANCE_NONE && eps->D) {
//        ierr = VecPointwiseDivide(x,x,eps->D);CHKERRQ(ierr);
//        ierr = VecNormalize(x,&norm);CHKERRQ(ierr);
//      }
#ifndef PETSC_USE_COMPLEX      
      /* compute imaginary part of eigenvector */
//      if (!eps->ishermitian && im != 0.0) {
//        ierr = SlepcVecMAXPBY(y,0.0,1.0,n,eps->Z+(i+1)*n,eps->V);CHKERRQ(ierr);
//        if (eps->ispositive) {
//          ierr = STApply(eps->OP,y,z);CHKERRQ(ierr);
//          ierr = VecNorm(z,NORM_2,&norm);CHKERRQ(ierr);          
//          ierr = VecScale(z,1.0/norm);CHKERRQ(ierr);
//          ierr = VecCopy(z,y);CHKERRQ(ierr);
//        }
//        if (eps->balance!=EPS_BALANCE_NONE && eps->D) {
//          ierr = VecPointwiseDivide(y,y,eps->D);CHKERRQ(ierr);
//          ierr = VecNormalize(y,&norm);CHKERRQ(ierr);
//        }
//      }
#endif
      /* compute relative error and update convergence flag */
//      ierr = EPSComputeRelativeError_Private(eps,re,im,x,y,&errest[i]);
//      if (errest[i] < eps->tol) conv[i] = PETSC_TRUE;
#ifndef PETSC_USE_COMPLEX      
//      if (!eps->ishermitian && im != 0.0) {
//        errest[i+1] = errest[i];
//        conv[i+1] = conv[i];
//        i++;
//      }
#endif
//    }
//  }

  /* free workspace */
//  ierr = VecDestroy(x);CHKERRQ(ierr);
//  ierr = VecDestroy(y);CHKERRQ(ierr);
//  if (!eps->ishermitian && eps->ispositive) { ierr = VecDestroy(z);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBuildBalance_Krylov"
/*
  EPSBuildBalance_Krylov - uses a Krylov subspace method to compute the
  diagonal matrix to be applied for balancing in non-Hermitian problems.
*/
PetscErrorCode EPSBuildBalance_Krylov(EPS eps)
{
  Vec            z, p, r;
  PetscInt       i, j;
  PetscReal      norma;
  PetscScalar    *pz, *pr, *pp, *pD;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(eps->V[0],&r);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->V[0],&p);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->V[0],&z);CHKERRQ(ierr);
  ierr = VecSet(eps->D,1.0);CHKERRQ(ierr);

  for (j=0;j<eps->balance_its;j++) {

    /* Build a random vector of +-1's */
    ierr = SlepcVecSetRandom(z,eps->rand);CHKERRQ(ierr);
    ierr = VecGetArray(z,&pz);CHKERRQ(ierr);
    for (i=0;i<eps->nloc;i++) {
      if (PetscRealPart(pz[i])<0.5) pz[i]=-1.0;
      else pz[i]=1.0;
    }
    ierr = VecRestoreArray(z,&pz);CHKERRQ(ierr);

    /* Compute p=DA(D\z) */
    ierr = VecPointwiseDivide(r,z,eps->D);CHKERRQ(ierr);
    ierr = STApply(eps->OP,r,p);CHKERRQ(ierr);
    ierr = VecPointwiseMult(p,p,eps->D);CHKERRQ(ierr);
    if (j==0) {
      /* Estimate the matrix inf-norm */
      ierr = VecAbs(p);CHKERRQ(ierr);
      ierr = VecMax(p,PETSC_NULL,&norma);CHKERRQ(ierr);
    }
    if (eps->balance == EPS_BALANCE_TWOSIDE) {
      /* Compute r=D\(A'Dz) */
      ierr = VecPointwiseMult(z,z,eps->D);CHKERRQ(ierr);
      ierr = STApplyTranspose(eps->OP,z,r);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(r,r,eps->D);CHKERRQ(ierr);
    }
    
    /* Adjust values of D */
    ierr = VecGetArray(r,&pr);CHKERRQ(ierr);
    ierr = VecGetArray(p,&pp);CHKERRQ(ierr);
    ierr = VecGetArray(eps->D,&pD);CHKERRQ(ierr);
    for (i=0;i<eps->nloc;i++) {
      if (eps->balance == EPS_BALANCE_TWOSIDE) {
        if (PetscAbsScalar(pp[i])>eps->balance_cutoff*norma && pr[i]!=0.0)
          pD[i] *= sqrt(PetscAbsScalar(pr[i]/pp[i]));
      } else {
        if (pp[i]!=0.0) pD[i] *= 1.0/PetscAbsScalar(pp[i]);
      }
    }
    ierr = VecRestoreArray(r,&pr);CHKERRQ(ierr);
    ierr = VecRestoreArray(p,&pp);CHKERRQ(ierr);
    ierr = VecRestoreArray(eps->D,&pD);CHKERRQ(ierr);
  }

  ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = VecDestroy(p);CHKERRQ(ierr);
  ierr = VecDestroy(z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

