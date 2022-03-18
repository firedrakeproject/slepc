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

   Method: Two-sided Arnoldi with Krylov-Schur restart (for left eigenvectors)

   References:

       [1] I.N. Zwaan and M.E. Hochstenbach, "Krylov-Schur-type restarts
           for the two-sided Arnoldi method", SIAM J. Matrix Anal. Appl.
           38(2):297-321, 2017.

*/

#include <slepc/private/epsimpl.h>
#include "krylovschur.h"
#include <slepcblaslapack.h>

static PetscErrorCode EPSTwoSidedRQUpdate1(EPS eps,Mat M,PetscInt nv,PetscReal beta,PetscReal betat)
{
  PetscScalar       *T,*S,*A,*w;
  const PetscScalar *pM;
  Vec               u;
  PetscInt          ld,ncv=eps->ncv,i,l,nnv;
  PetscBLASInt      info,n_,ncv_,*p,one=1;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(PetscMalloc3(nv,&p,ncv*ncv,&A,ncv,&w));
  CHKERRQ(BVGetActiveColumns(eps->V,&l,&nnv));
  CHKERRQ(BVSetActiveColumns(eps->V,0,nv));
  CHKERRQ(BVSetActiveColumns(eps->W,0,nv));
  CHKERRQ(BVGetColumn(eps->V,nv,&u));
  CHKERRQ(BVDotVec(eps->W,u,w));
  CHKERRQ(BVRestoreColumn(eps->V,nv,&u));
  CHKERRQ(MatDenseGetArrayRead(M,&pM));
  CHKERRQ(PetscArraycpy(A,pM,ncv*ncv));
  CHKERRQ(MatDenseRestoreArrayRead(M,&pM));
  CHKERRQ(PetscBLASIntCast(nv,&n_));
  CHKERRQ(PetscBLASIntCast(ncv,&ncv_));
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n_,&n_,A,&ncv_,p,&info));
  SlepcCheckLapackInfo("getrf",info);
  CHKERRQ(PetscLogFlops(2.0*n_*n_*n_/3.0));
  PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("N",&n_,&one,A,&ncv_,p,w,&ncv_,&info));
  SlepcCheckLapackInfo("getrs",info);
  CHKERRQ(PetscLogFlops(2.0*n_*n_-n_));
  CHKERRQ(BVMultColumn(eps->V,-1.0,1.0,nv,w));
  CHKERRQ(DSGetArray(eps->ds,DS_MAT_A,&S));
  for (i=0;i<nv;i++) S[(nv-1)*ld+i] += beta*w[i];
  CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_A,&S));
  CHKERRQ(BVGetColumn(eps->W,nv,&u));
  CHKERRQ(BVDotVec(eps->V,u,w));
  CHKERRQ(BVRestoreColumn(eps->W,nv,&u));
  PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("C",&n_,&one,A,&ncv_,p,w,&ncv_,&info));
  CHKERRQ(PetscFPTrapPop());
  CHKERRQ(BVMultColumn(eps->W,-1.0,1.0,nv,w));
  CHKERRQ(DSGetArray(eps->ds,DS_MAT_B,&T));
  for (i=0;i<nv;i++) T[(nv-1)*ld+i] += betat*w[i];
  CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_B,&T));
  CHKERRQ(PetscFree3(p,A,w));
  CHKERRQ(BVSetActiveColumns(eps->V,l,nnv));
  CHKERRQ(BVSetActiveColumns(eps->W,l,nnv));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSTwoSidedRQUpdate2(EPS eps,Mat M,PetscInt k)
{
  PetscScalar    *Q,*pM,*w,zero=0.0,sone=1.0,*c,*A;
  PetscBLASInt   n_,ncv_,ld_;
  PetscReal      norm;
  PetscInt       l,nv,ncv=eps->ncv,ld,i,j;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(BVGetActiveColumns(eps->V,&l,&nv));
  CHKERRQ(BVSetActiveColumns(eps->V,0,nv));
  CHKERRQ(BVSetActiveColumns(eps->W,0,nv));
  CHKERRQ(PetscMalloc2(ncv*ncv,&w,ncv,&c));
  /* u = u - V*V'*u */
  CHKERRQ(BVOrthogonalizeColumn(eps->V,k,c,&norm,NULL));
  CHKERRQ(BVScaleColumn(eps->V,k,1.0/norm));
  CHKERRQ(DSGetArray(eps->ds,DS_MAT_A,&A));
  /* H = H + V'*u*b' */
  for (j=l;j<k;j++) {
    for (i=0;i<k;i++) A[i+j*ld] += c[i]*A[k+j*ld];
    A[k+j*ld] *= norm;
  }
  CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_A,&A));
  CHKERRQ(BVOrthogonalizeColumn(eps->W,k,c,&norm,NULL));
  CHKERRQ(BVScaleColumn(eps->W,k,1.0/norm));
  CHKERRQ(DSGetArray(eps->ds,DS_MAT_B,&A));
  /* H = H + V'*u*b' */
  for (j=l;j<k;j++) {
    for (i=0;i<k;i++) A[i+j*ld] += c[i]*A[k+j*ld];
    A[k+j*ld] *= norm;
  }
  CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_B,&A));

  /* M = Q'*M*Q */
  CHKERRQ(MatDenseGetArray(M,&pM));
  CHKERRQ(PetscBLASIntCast(ncv,&ncv_));
  CHKERRQ(PetscBLASIntCast(nv,&n_));
  CHKERRQ(PetscBLASIntCast(ld,&ld_));
  CHKERRQ(DSGetArray(eps->ds,DS_MAT_Q,&Q));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,pM,&ncv_,Q,&ld_,&zero,w,&ncv_));
  CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_Q,&Q));
  CHKERRQ(DSGetArray(eps->ds,DS_MAT_Z,&Q));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&n_,&n_,&n_,&sone,Q,&ld_,w,&ncv_,&zero,pM,&ncv_));
  CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_Z,&Q));
  CHKERRQ(MatDenseRestoreArray(M,&pM));
  CHKERRQ(PetscFree2(w,c));
  CHKERRQ(BVSetActiveColumns(eps->V,l,nv));
  CHKERRQ(BVSetActiveColumns(eps->W,l,nv));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_KrylovSchur_TwoSided(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  Mat             M,U,Op,OpHT,S,T;
  PetscReal       norm,norm2,beta,betat;
  PetscInt        ld,l,nv,nvt,k,nconv,dsn,dsk;
  PetscBool       breakdownt,breakdown,breakdownl;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(EPSGetStartVector(eps,0,NULL));
  CHKERRQ(EPSGetLeftStartVector(eps,0,NULL));
  l = 0;
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,eps->ncv,eps->ncv,NULL,&M));

  CHKERRQ(STGetOperator(eps->st,&Op));
  CHKERRQ(MatCreateHermitianTranspose(Op,&OpHT));

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization for Op */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    CHKERRQ(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_A,&S));
    CHKERRQ(BVMatArnoldi(eps->V,Op,S,eps->nconv+l,&nv,&beta,&breakdown));
    CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_A,&S));

    /* Compute an nv-step Arnoldi factorization for Op' */
    nvt = nv;
    CHKERRQ(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_B,&T));
    CHKERRQ(BVMatArnoldi(eps->W,OpHT,T,eps->nconv+l,&nvt,&betat,&breakdownt));
    CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_B,&T));

    /* Make sure both factorizations have the same length */
    nv = PetscMin(nv,nvt);
    CHKERRQ(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    if (l==0) {
      CHKERRQ(DSSetState(eps->ds,DS_STATE_INTERMEDIATE));
    } else {
      CHKERRQ(DSSetState(eps->ds,DS_STATE_RAW));
    }
    breakdown = (breakdown || breakdownt)? PETSC_TRUE: PETSC_FALSE;

    /* Update M, modify Rayleigh quotients S and T */
    CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv+l,nv));
    CHKERRQ(BVSetActiveColumns(eps->W,eps->nconv+l,nv));
    CHKERRQ(BVMatProject(eps->V,NULL,eps->W,M));

    CHKERRQ(EPSTwoSidedRQUpdate1(eps,M,nv,beta,betat));

    /* Solve projected problem */
    CHKERRQ(DSSolve(eps->ds,eps->eigr,eps->eigi));
    CHKERRQ(DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL));
    CHKERRQ(DSSynchronize(eps->ds,eps->eigr,eps->eigi));
    CHKERRQ(DSUpdateExtraRow(eps->ds));

    /* Check convergence */
    CHKERRQ(BVNormColumn(eps->V,nv,NORM_2,&norm));
    CHKERRQ(BVNormColumn(eps->W,nv,NORM_2,&norm2));
    CHKERRQ(EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,nv-eps->nconv,beta*norm,betat*norm2,1.0,&k));
    CHKERRQ((*eps->stopping)(eps,eps->its,eps->max_it,k,eps->nev,&eps->reason,eps->stoppingctx));
    nconv = k;

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      CHKERRQ(DSGetTruncateSize(eps->ds,k,nv,&l));
    }
    if (!ctx->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */
    if (l) CHKERRQ(PetscInfo(eps,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv,nv));
    CHKERRQ(BVSetActiveColumns(eps->W,eps->nconv,nv));
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_Q,&U));
    CHKERRQ(BVMultInPlace(eps->V,U,eps->nconv,k+l));
    CHKERRQ(MatDestroy(&U));
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_Z,&U));
    CHKERRQ(BVMultInPlace(eps->W,U,eps->nconv,k+l));
    CHKERRQ(MatDestroy(&U));
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      CHKERRQ(BVCopyColumn(eps->V,nv,k+l));
      CHKERRQ(BVCopyColumn(eps->W,nv,k+l));
    }

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown || k==nv) {
        /* Start a new Arnoldi factorization */
        CHKERRQ(PetscInfo(eps,"Breakdown in Krylov-Schur method (it=%" PetscInt_FMT " norm=%g)\n",eps->its,(double)beta));
        if (k<eps->nev) {
          CHKERRQ(EPSGetStartVector(eps,k,&breakdown));
          CHKERRQ(EPSGetLeftStartVector(eps,k,&breakdownl));
          if (breakdown || breakdownl) {
            eps->reason = EPS_DIVERGED_BREAKDOWN;
            CHKERRQ(PetscInfo(eps,"Unable to generate more start vectors\n"));
          }
        }
      } else {
        CHKERRQ(DSGetDimensions(eps->ds,&dsn,NULL,&dsk,NULL));
        CHKERRQ(DSSetDimensions(eps->ds,dsn,k,dsk));
        CHKERRQ(DSTruncate(eps->ds,k+l,PETSC_FALSE));
      }
      CHKERRQ(EPSTwoSidedRQUpdate2(eps,M,k+l));
    }
    eps->nconv = k;
    CHKERRQ(EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,nv));
  }

  CHKERRQ(STRestoreOperator(eps->st,&Op));
  CHKERRQ(MatDestroy(&OpHT));

  CHKERRQ(DSTruncate(eps->ds,eps->nconv,PETSC_TRUE));
  CHKERRQ(MatDestroy(&M));
  PetscFunctionReturn(0);
}
