/*

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur for symmetric-indefinite eigenproblems

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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
#include "krylovschur.h"

#undef __FUNCT__
#define __FUNCT__ "EPSFullLanczosIndef"
static PetscErrorCode EPSFullLanczosIndef(EPS eps,PetscReal *alpha,PetscReal *beta,PetscReal *omega,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscReal *cos,Vec w)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M;
  Vec            vj,vj1;
  PetscScalar    *hwork,lhwork[100];
  PetscReal      norm,norm1,norm2,t;

  PetscFunctionBegin;
  if (cos) *cos = 1.0;
  if (m > 100) {
    ierr = PetscMalloc1(m,&hwork);CHKERRQ(ierr);
  } else hwork = lhwork;

  ierr = BVSetActiveColumns(eps->V,0,m);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    ierr = BVGetColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVGetColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = STApply(eps->st,vj,vj1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = BVOrthogonalizeColumn(eps->V,j+1,hwork,&norm,breakdown);CHKERRQ(ierr);
    alpha[j] = PetscRealPart(hwork[j]);
    beta[j] = PetscAbsReal(norm);
    omega[j+1] = (norm<0.0)? -1.0: 1.0;
    ierr = BVScaleColumn(eps->V,j+1,1.0/norm);CHKERRQ(ierr);
    /* */
    ierr = BVNormColumn(eps->V,j+1,NORM_2,&norm1);CHKERRQ(ierr);
    ierr = BVGetColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = BVApplyMatrix(eps->V,vj1,w);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = BVNormVec(eps->V,w,NORM_2,&norm2);CHKERRQ(ierr);
    t = 1.0/(norm1*norm2);
    if (cos && *cos>t) *cos = t;
  }
  if (m > 100) {
    ierr = PetscFree(hwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_KrylovSchur_Indefinite"
PetscErrorCode EPSSolve_KrylovSchur_Indefinite(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i,k,l,ld,nv,t;
  Mat             U;
  Vec             vomega,u,w=eps->work[0];
  PetscScalar     *Q,*aux;
  PetscReal       *a,*b,*r,beta,beta1,beta2,*omega;
  PetscBool       breakdown=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);

  /* Get the starting Lanczos vector */
  ierr = EPSGetStartVector(eps,0,NULL);CHKERRQ(ierr);

  /* Extract sigma[0] from BV, computed during normalization */
  ierr = BVSetActiveColumns(eps->V,0,1);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&vomega);CHKERRQ(ierr);
  ierr = BVGetSignature(eps->V,vomega);CHKERRQ(ierr);
  ierr = VecGetArray(vomega,&aux);CHKERRQ(ierr);
  ierr = DSGetArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  omega[0] = PetscRealPart(aux[0]);
  ierr = DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  ierr = VecRestoreArray(vomega,&aux);CHKERRQ(ierr);
  ierr = VecDestroy(&vomega);CHKERRQ(ierr);
  l = 0;

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    b = a + ld;
    ierr = DSGetArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    ierr = EPSFullLanczosIndef(eps,a,b,omega,eps->nconv+l,&nv,&breakdown,NULL,w);CHKERRQ(ierr);
    beta = b[nv-1];
    ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    ierr = DSSetDimensions(eps->ds,nv,0,eps->nconv,eps->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(eps->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(eps->V,eps->nconv,nv);CHKERRQ(ierr);

    /* Solve projected problem */
    ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
    ierr = DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL);CHKERRQ(ierr);

    /* Check convergence */
    ierr = DSGetDimensions(eps->ds,NULL,NULL,NULL,NULL,&t);CHKERRQ(ierr);
    ierr = BVGetColumn(eps->V,nv,&u);CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_2,&beta1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,nv,&u);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&beta2);CHKERRQ(ierr);  /* w contains B*V[nv] */
    beta1 = PetscMax(beta1,beta2);
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,t-eps->nconv,beta*beta1,1.0,&k);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      l = PetscMin(l,t);
      ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
      if (*(a+ld+k+l-1)!=0) {
        if (k+l<t-1) l = l+1;
        else l = l-1;
      }
      ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    }

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_CONV_FAILED,"Breakdown in Indefinite Krylov-Schur (beta=%g)",beta);
      } else {
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSGetArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSGetArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
        b = a + ld;
        r = a + 2*ld;
        for (i=k;i<k+l;i++) {
          r[i] = PetscRealPart(Q[nv-1+i*ld]*beta);
        }
        b[k+l-1] = r[k+l-1];
        omega[k+l] = omega[nv];
        ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = DSGetMat(eps->ds,DS_MAT_Q,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(eps->V,U,eps->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);

    /* Append u to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = BVCopyColumn(eps->V,nv,k+l);CHKERRQ(ierr);
    }

    ierr = EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
    eps->nconv = k;
  }
  ierr = DSSetDimensions(eps->ds,eps->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

