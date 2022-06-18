/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscInt        k,l,ld,nv,t,nconv=0;
  Mat             U,D;
  Vec             vomega,w=eps->work[0];
  PetscReal       *a,*b,beta,beta1=1.0,*omega;
  PetscBool       breakdown=PETSC_FALSE,symmlost=PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DSGetLeadingDimension(eps->ds,&ld));

  /* Get the starting Lanczos vector */
  PetscCall(EPSGetStartVector(eps,0,NULL));

  /* Extract sigma[0] from BV, computed during normalization */
  PetscCall(DSSetDimensions(eps->ds,1,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(BVSetActiveColumns(eps->V,0,1));
  PetscCall(DSGetMatAndColumn(eps->ds,DS_MAT_D,0,&D,&vomega));
  PetscCall(BVGetSignature(eps->V,vomega));
  PetscCall(DSRestoreMatAndColumn(eps->ds,DS_MAT_D,0,&D,&vomega));
  l = 0;

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    PetscCall(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    PetscCall(DSGetArrayReal(eps->ds,DS_MAT_T,&a));
    b = a + ld;
    PetscCall(DSGetArrayReal(eps->ds,DS_MAT_D,&omega));
    PetscCall(EPSPseudoLanczos(eps,a,b,omega,eps->nconv+l,&nv,&breakdown,&symmlost,NULL,w));
    if (symmlost) {
      eps->reason = EPS_DIVERGED_SYMMETRY_LOST;
      if (nv==eps->nconv+l+1) { eps->nconv = nconv; break; }
    }
    beta = b[nv-1];
    PetscCall(DSRestoreArrayReal(eps->ds,DS_MAT_T,&a));
    PetscCall(DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega));
    PetscCall(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    PetscCall(DSSetState(eps->ds,l?DS_STATE_RAW:DS_STATE_INTERMEDIATE));
    PetscCall(BVSetActiveColumns(eps->V,eps->nconv,nv));

    /* Solve projected problem */
    PetscCall(DSSolve(eps->ds,eps->eigr,eps->eigi));
    PetscCall(DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL));
    PetscCall(DSUpdateExtraRow(eps->ds));
    PetscCall(DSSynchronize(eps->ds,eps->eigr,eps->eigi));

    /* Check convergence */
    PetscCall(DSGetDimensions(eps->ds,NULL,NULL,NULL,&t));
#if 0
    /* take into account also left residual */
    PetscCall(BVGetColumn(eps->V,nv,&u));
    PetscCall(VecNorm(u,NORM_2,&beta1));
    PetscCall(BVRestoreColumn(eps->V,nv,&u));
    PetscCall(VecNorm(w,NORM_2,&beta2));  /* w contains B*V[nv] */
    beta1 = PetscMax(beta1,beta2);
#endif
    PetscCall(EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,t-eps->nconv,beta*beta1,0.0,1.0,&k));
    PetscCall((*eps->stopping)(eps,eps->its,eps->max_it,k,eps->nev,&eps->reason,eps->stoppingctx));
    nconv = k;

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      l = PetscMin(l,t);
      PetscCall(DSGetTruncateSize(eps->ds,k,t,&l));
    }
    if (!ctx->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */
    if (l) PetscCall(PetscInfo(eps,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      PetscCheck(!breakdown,PetscObjectComm((PetscObject)eps),PETSC_ERR_CONV_FAILED,"Breakdown in Indefinite Krylov-Schur (beta=%g)",(double)beta);
      /* Prepare the Rayleigh quotient for restart */
      PetscCall(DSTruncate(eps->ds,k+l,PETSC_FALSE));
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    PetscCall(DSGetMat(eps->ds,DS_MAT_Q,&U));
    PetscCall(BVMultInPlace(eps->V,U,eps->nconv,k+l));
    PetscCall(DSRestoreMat(eps->ds,DS_MAT_Q,&U));

    /* Move restart vector and update signature */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      PetscCall(BVCopyColumn(eps->V,nv,k+l));
      PetscCall(BVSetActiveColumns(eps->V,0,k+l));
      PetscCall(DSGetMatAndColumn(eps->ds,DS_MAT_D,0,&D,&vomega));
      PetscCall(BVSetSignature(eps->V,vomega));
      PetscCall(DSRestoreMatAndColumn(eps->ds,DS_MAT_D,0,&D,&vomega));
    }

    eps->nconv = k;
    PetscCall(EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,nv));
  }
  PetscCall(DSTruncate(eps->ds,eps->nconv,PETSC_TRUE));
  PetscFunctionReturn(0);
}
