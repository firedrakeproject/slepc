/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur for symmetric-indefinite eigenproblems
*/
#include <slepc/private/epsimpl.h>
#include "krylovschur.h"

PetscErrorCode EPSSolve_KrylovSchur_Indefinite(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i,k,l,ld,nv,t,nconv=0;
  Mat             U;
  Vec             vomega,w=eps->work[0];
  PetscScalar     *aux;
  PetscReal       *a,*b,beta,beta1=1.0,*omega;
  PetscBool       breakdown=PETSC_FALSE,symmlost=PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));

  /* Get the starting Lanczos vector */
  CHKERRQ(EPSGetStartVector(eps,0,NULL));

  /* Extract sigma[0] from BV, computed during normalization */
  CHKERRQ(BVSetActiveColumns(eps->V,0,1));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,1,&vomega));
  CHKERRQ(BVGetSignature(eps->V,vomega));
  CHKERRQ(VecGetArray(vomega,&aux));
  CHKERRQ(DSGetArrayReal(eps->ds,DS_MAT_D,&omega));
  omega[0] = PetscRealPart(aux[0]);
  CHKERRQ(DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega));
  CHKERRQ(VecRestoreArray(vomega,&aux));
  CHKERRQ(VecDestroy(&vomega));
  l = 0;

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    CHKERRQ(DSGetArrayReal(eps->ds,DS_MAT_T,&a));
    b = a + ld;
    CHKERRQ(DSGetArrayReal(eps->ds,DS_MAT_D,&omega));
    CHKERRQ(EPSPseudoLanczos(eps,a,b,omega,eps->nconv+l,&nv,&breakdown,&symmlost,NULL,w));
    if (symmlost) {
      eps->reason = EPS_DIVERGED_SYMMETRY_LOST;
      if (nv==eps->nconv+l+1) { eps->nconv = nconv; break; }
    }
    beta = b[nv-1];
    CHKERRQ(DSRestoreArrayReal(eps->ds,DS_MAT_T,&a));
    CHKERRQ(DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega));
    CHKERRQ(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    if (l==0) CHKERRQ(DSSetState(eps->ds,DS_STATE_INTERMEDIATE));
    else CHKERRQ(DSSetState(eps->ds,DS_STATE_RAW));
    CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv,nv));

    /* Solve projected problem */
    CHKERRQ(DSSolve(eps->ds,eps->eigr,eps->eigi));
    CHKERRQ(DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL));
    CHKERRQ(DSUpdateExtraRow(eps->ds));
    CHKERRQ(DSSynchronize(eps->ds,eps->eigr,eps->eigi));

    /* Check convergence */
    CHKERRQ(DSGetDimensions(eps->ds,NULL,NULL,NULL,&t));
#if 0
    /* take into account also left residual */
    CHKERRQ(BVGetColumn(eps->V,nv,&u));
    CHKERRQ(VecNorm(u,NORM_2,&beta1));
    CHKERRQ(BVRestoreColumn(eps->V,nv,&u));
    CHKERRQ(VecNorm(w,NORM_2,&beta2));  /* w contains B*V[nv] */
    beta1 = PetscMax(beta1,beta2);
#endif
    CHKERRQ(EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,t-eps->nconv,beta*beta1,0.0,1.0,&k));
    CHKERRQ((*eps->stopping)(eps,eps->its,eps->max_it,k,eps->nev,&eps->reason,eps->stoppingctx));
    nconv = k;

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      l = PetscMin(l,t);
      CHKERRQ(DSGetTruncateSize(eps->ds,k,t,&l));
    }
    if (!ctx->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */
    if (l) CHKERRQ(PetscInfo(eps,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      PetscCheck(!breakdown,PetscObjectComm((PetscObject)eps),PETSC_ERR_CONV_FAILED,"Breakdown in Indefinite Krylov-Schur (beta=%g)",(double)beta);
      /* Prepare the Rayleigh quotient for restart */
      CHKERRQ(DSTruncate(eps->ds,k+l,PETSC_FALSE));
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_Q,&U));
    CHKERRQ(BVMultInPlace(eps->V,U,eps->nconv,k+l));
    CHKERRQ(MatDestroy(&U));

    /* Move restart vector and update signature */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      CHKERRQ(BVCopyColumn(eps->V,nv,k+l));
      CHKERRQ(DSGetArrayReal(eps->ds,DS_MAT_D,&omega));
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,k+l,&vomega));
      CHKERRQ(VecGetArray(vomega,&aux));
      for (i=0;i<k+l;i++) aux[i] = omega[i];
      CHKERRQ(VecRestoreArray(vomega,&aux));
      CHKERRQ(BVSetActiveColumns(eps->V,0,k+l));
      CHKERRQ(BVSetSignature(eps->V,vomega));
      CHKERRQ(VecDestroy(&vomega));
      CHKERRQ(DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega));
    }

    eps->nconv = k;
    CHKERRQ(EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,nv));
  }
  CHKERRQ(DSTruncate(eps->ds,eps->nconv,PETSC_TRUE));
  PetscFunctionReturn(0);
}
