/*
     This file contains some simple default routines for common operations.  

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

#include <slepc-private/epsimpl.h>   /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

#undef __FUNCT__  
#define __FUNCT__ "EPSReset_Default"
PetscErrorCode EPSReset_Default(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
    ierr = VecDestroy(&w);CHKERRQ(ierr);
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
  If left eigenvectors are required then also do Z'*T=D*Z', Y=W*Z
 */
PetscErrorCode EPSComputeVectors_Schur(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       n,i,ld;
  PetscBLASInt   one = 1; 
  PetscScalar    *Z,tmp;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      normi;
#endif
  PetscReal      norm;
  Vec            w;
  
  PetscFunctionBegin;
  if (eps->ishermitian) {
    ierr = EPSComputeVectors_Hermitian(eps);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PSGetLeadingDimension(eps->ps,&ld);CHKERRQ(ierr);
  ierr = PSGetDimensions(eps->ps,&n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* right eigenvectors */
  ierr = PSVectors(eps->ps,PS_MAT_X,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* normalize eigenvectors (when not using purification nor balancing)*/
  if (!(eps->ispositive || (eps->balance!=EPS_BALANCE_NONE && eps->D))) {
    ierr = PSGetArray(eps->ps,PS_MAT_X,&Z);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
#if !defined(PETSC_USE_COMPLEX)
      if (eps->eigi[i] != 0.0) {
        norm = BLASnrm2_(&n,Z+i*ld,&one);
        normi = BLASnrm2_(&n,Z+(i+1)*ld,&one);
        tmp = 1.0 / SlepcAbsEigenvalue(norm,normi);
        BLASscal_(&n,&tmp,Z+i*ld,&one);
        BLASscal_(&n,&tmp,Z+(i+1)*ld,&one);
        i++;     
      } else
#endif
      {
        norm = BLASnrm2_(&n,Z+i*ld,&one);
        tmp = 1.0 / norm;
        BLASscal_(&n,&tmp,Z+i*ld,&one);
      }
    }
    ierr = PSRestoreArray(eps->ps,PS_MAT_X,&Z);CHKERRQ(ierr);
  }
  
  /* V = V * Z */
  ierr = PSGetArray(eps->ps,PS_MAT_X,&Z);CHKERRQ(ierr);
  ierr = SlepcUpdateVectors(n,eps->V,0,n,Z,ld,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PSRestoreArray(eps->ps,PS_MAT_X,&Z);CHKERRQ(ierr);

  /* Purify eigenvectors */
  if (eps->ispositive) {
    ierr = VecDuplicate(eps->V[0],&w);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = VecCopy(eps->V[i],w);CHKERRQ(ierr); 
      ierr = STApply(eps->OP,w,eps->V[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&w);CHKERRQ(ierr);
  }

  /* Fix eigenvectors if balancing was used */
  if (eps->balance!=EPS_BALANCE_NONE && eps->D) {
    for (i=0;i<n;i++) {
      ierr = VecPointwiseDivide(eps->V[i],eps->V[i],eps->D);CHKERRQ(ierr);
    }
  }

  /* normalize eigenvectors (when using purification or balancing) */
  if (eps->ispositive || (eps->balance!=EPS_BALANCE_NONE && eps->D)) {
    for (i=0;i<n;i++) {
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
    ierr = PSVectors(eps->ps,PS_MAT_Y,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    /* W = W * Z */
    ierr = PSGetArray(eps->ps,PS_MAT_Y,&Z);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(n,eps->W,0,n,Z,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_Y,&Z);CHKERRQ(ierr);
  }
  eps->evecsavailable = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultGetWork"
/*
  EPSDefaultGetWork - Gets a number of work vectors.
 */
PetscErrorCode EPSDefaultGetWork(EPS eps,PetscInt nw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (eps->nwork != nw) {
    ierr = VecDestroyVecs(eps->nwork,&eps->work);CHKERRQ(ierr);
    eps->nwork = nw;
    ierr = VecDuplicateVecs(eps->t,nw,&eps->work);CHKERRQ(ierr);
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
  ierr = VecDestroyVecs(eps->nwork,&eps->work);CHKERRQ(ierr);
  eps->nwork = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultSetWhich"
/*
  EPSDefaultSetWhich - Sets the default value for which, depending on the ST.
 */
PetscErrorCode EPSDefaultSetWhich(EPS eps)
{
  PetscBool      target;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)eps->OP,&target,STSINVERT,STCAYLEY,STFOLD,"");CHKERRQ(ierr);
  if (target) eps->which = EPS_TARGET_MAGNITUDE;
  else eps->which = EPS_LARGEST_MAGNITUDE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSConvergedEigRelative"
/*
  EPSConvergedEigRelative - Checks convergence relative to the eigenvalue.
*/
PetscErrorCode EPSConvergedEigRelative(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscReal w;

  PetscFunctionBegin;
  w = SlepcAbsEigenvalue(eigr,eigi);
  *errest = res/w;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSConvergedAbsolute"
/*
  EPSConvergedAbsolute - Checks convergence absolutely.
*/
PetscErrorCode EPSConvergedAbsolute(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSConvergedNormRelative"
/*
  EPSConvergedNormRelative - Checks convergence relative to the eigenvalue and 
  the matrix norms.
*/
PetscErrorCode EPSConvergedNormRelative(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscReal w;

  PetscFunctionBegin;
  w = SlepcAbsEigenvalue(eigr,eigi);
  *errest = res / (eps->nrma + w*eps->nrmb);
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
  Vec            x,y,z=0;
  PetscReal      norm;
  
  PetscFunctionBegin;
  /* allocate workspace */
  ierr = VecDuplicate(V[0],&x);CHKERRQ(ierr);
  ierr = VecDuplicate(V[0],&y);CHKERRQ(ierr);
  if (!eps->ishermitian && eps->ispositive) { ierr = VecDuplicate(V[0],&z);CHKERRQ(ierr); }

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
#if !defined(PETSC_USE_COMPLEX)
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
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
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
  Vec               z,p,r;
  PetscInt          i,j;
  PetscReal         norma;
  PetscScalar       *pz,*pD;
  const PetscScalar *pr,*pp;
  PetscErrorCode    ierr;

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
    ierr = VecGetArrayRead(r,&pr);CHKERRQ(ierr);
    ierr = VecGetArrayRead(p,&pp);CHKERRQ(ierr);
    ierr = VecGetArray(eps->D,&pD);CHKERRQ(ierr);
    for (i=0;i<eps->nloc;i++) {
      if (eps->balance == EPS_BALANCE_TWOSIDE) {
        if (PetscAbsScalar(pp[i])>eps->balance_cutoff*norma && pr[i]!=0.0)
          pD[i] *= PetscSqrtReal(PetscAbsScalar(pr[i]/pp[i]));
      } else {
        if (pp[i]!=0.0) pD[i] *= 1.0/PetscAbsScalar(pp[i]);
      }
    }
    ierr = VecRestoreArrayRead(r,&pr);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(p,&pp);CHKERRQ(ierr);
    ierr = VecRestoreArray(eps->D,&pD);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

