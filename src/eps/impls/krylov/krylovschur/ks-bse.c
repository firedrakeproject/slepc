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

   Method: thick-restarted Lanczos for Bethe-Salpeter pseudo-Hermitan matrices

   References:

       [1] M. Shao et al, "A structure preserving Lanczos algorithm for computing
           the optical absorption spectrum", SIAM J. Matrix Anal. App. 39(2), 2018.

*/
#include <slepc/private/epsimpl.h>
#include "krylovschur.h"

static PetscErrorCode Orthog1(Vec x,BV U,BV V,PetscInt j,PetscScalar *h,PetscScalar *c,PetscBool *breakdown)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(BVSetActiveColumns(U,0,j));
  PetscCall(BVSetActiveColumns(V,0,j));
  /* c = real(V^* x) ; c2 = imag(U^* x)*1i */
#if defined(PETSC_USE_COMPLEX)
  PetscCall(BVDotVecBegin(V,x,c));
  PetscCall(BVDotVecBegin(U,x,c+j));
  PetscCall(BVDotVecEnd(V,x,c));
  PetscCall(BVDotVecEnd(U,x,c+j));
#else
  PetscCall(BVDotVec(V,x,c));
#endif
  for (i=0; i<j; i++) {
    c[i] = PetscRealPart(c[i]);
#if defined(PETSC_USE_COMPLEX)
    c[j+i] = PetscCMPLX(0.0,PetscImaginaryPart(c[j+i]));
#endif
  }
  /* x = x-U*c-V*c2 */
  PetscCall(BVMultVec(U,-1.0,1.0,x,c));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(BVMultVec(V,-1.0,1.0,x,c+j));
#endif
  /* accumulate orthog coeffs into h */
  for (i=0; i<2*j; i++)
    h[i] += c[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Orthogonalize vector x against first j vectors in U and V */
static PetscErrorCode OrthogonalizeVector(Vec x,BV U,BV V,PetscInt j,PetscScalar *h,PetscBool *breakdown)
{
  PetscFunctionBegin;
  PetscCall(PetscArrayzero(h,2*j));
  /* Orghogonalize twice */
  PetscCall(Orthog1(x,U,V,j,h,h+2*j,breakdown));
  PetscCall(Orthog1(x,U,V,j,h,h+2*j,breakdown));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode EPSBSELanczos(EPS eps,Mat Htp,Mat Htm,BV U,BV V,PetscReal *alpha,PetscReal *beta,PetscInt k,PetscInt *M,PetscBool *breakdown)
{
  PetscInt       j,m = *M;
  Vec            v,x,y;
  PetscScalar    *hwork,lhwork[100],gamma;

  PetscFunctionBegin;
  if (4*m > 100) PetscCall(PetscMalloc1(4*m,&hwork));
  else hwork = lhwork;

  for (j=k;j<m;j++) {
    /* j+1 columns (indexes 0 to j) have been computed */
    PetscCall(BVGetColumn(V,j,&v));
    PetscCall(BVGetColumn(U,j+1,&x));
    PetscCall(BVGetColumn(V,j+1,&y));
    PetscCall(MatMult(Htm,v,x));
    PetscCall(OrthogonalizeVector(x,U,V,j+1,hwork,breakdown));
    alpha[j] = PetscRealPart(hwork[j]);
    PetscCall(MatMult(Htp,x,y));
    PetscCall(VecDot(x,y,&gamma));
    beta[j] = PetscSqrtReal(PetscRealPart(gamma));
    PetscCall(VecScale(x,1.0/beta[j]));
    PetscCall(VecScale(y,1.0/beta[j]));
    PetscCall(BVRestoreColumn(V,j,&v));
    PetscCall(BVRestoreColumn(U,j+1,&x));
    PetscCall(BVRestoreColumn(V,j+1,&y));
  }
  if (4*m > 100) PetscCall(PetscFree(hwork));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EPSComputeVectors_BSE(EPS eps)
{
  PetscFunctionBegin;
  PetscCall(EPSComputeVectors_Schur(eps));   /* FIXME */
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EPSSetUp_KrylovSchur_BSE(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscBool       flg,sinvert;

  PetscFunctionBegin;
  PetscCheck((eps->problem_type==EPS_BSE),PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Problem type should be BSE");
  EPSCheckUnsupportedCondition(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_EXTRACTION | EPS_FEATURE_BALANCE,PETSC_TRUE," with BSE structure");
  PetscCall(EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd));
  PetscCheck(eps->ncv<=eps->nev+eps->mpd,PetscObjectComm((PetscObject)eps),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nev+mpd");
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);

  PetscCall(PetscObjectTypeCompareAny((PetscObject)eps->st,&flg,STSINVERT,STSHIFT,""));
  PetscCheck(flg,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Krylov-Schur BSE only supports shift and shift-and-invert ST");
  PetscCall(PetscObjectTypeCompare((PetscObject)eps->st,STSINVERT,&sinvert));
  PetscCheck(!sinvert,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Shift-and-invert not implemented yet");
  if (!eps->which) {
    if (sinvert) eps->which = EPS_TARGET_MAGNITUDE;
    else eps->which = EPS_SMALLEST_MAGNITUDE;
  }

  if (!ctx->keep) ctx->keep = 0.5;

  PetscCall(EPSAllocateSolution(eps,1));
  eps->ops->solve = EPSSolve_KrylovSchur_BSE;
  eps->ops->computevectors = EPSComputeVectors_BSE;
  PetscCall(DSSetType(eps->ds,DSHEP));
  PetscCall(DSSetCompact(eps->ds,PETSC_TRUE));
  PetscCall(DSSetExtraRow(eps->ds,PETSC_TRUE));
  PetscCall(DSAllocate(eps->ds,eps->ncv+1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Computes the product y=A*x+B*conj(x)
 */
static PetscErrorCode MatMult_Htp(Mat Ht,Vec x,Vec y)
{
  EPS_BSE_MAT matctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Ht,&matctx));
  PetscCall(VecCopy(x,y));
  PetscCall(VecConjugate(y));
  PetscCall(MatMult(matctx->B,y,matctx->w));
  PetscCall(MatMultAdd(matctx->A,x,matctx->w,y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Computes the product y=A*x-B*conj(x)
 */
static PetscErrorCode MatMult_Htm(Mat Ht,Vec x,Vec y)
{
  EPS_BSE_MAT matctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Ht,&matctx));
  PetscCall(VecCopy(x,y));
  PetscCall(VecConjugate(y));
  PetscCall(VecScale(y,-1.0));
  PetscCall(MatMult(matctx->B,y,matctx->w));
  PetscCall(MatMultAdd(matctx->A,x,matctx->w,y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EPSSolve_KrylovSchur_BSE(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  EPS_BSE_MAT     matctx;
  PetscInt        M,N,m,n,k,l,ld,nv,nconv=0;
  Mat             H,Htp,Htm,Q;
  BV              U,V;
  IS              is[2];
  Vec             x,u1,v1;//w=eps->work[0];
  VecType         vtype;
  PetscReal       *a,*b,beta,nrm,delta;
  PetscScalar     alpha;
  PetscBool       breakdown=PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DSGetLeadingDimension(eps->ds,&ld));

  /* Extract matrix blocks */
  PetscCall(STGetMatrix(eps->st,0,&H));
  PetscCall(MatNestGetISs(H,is,NULL));
  PetscCall(PetscNew(&matctx));
  PetscCall(MatNestGetSubMat(H,0,0,&matctx->A));
  PetscCall(MatNestGetSubMat(H,0,1,&matctx->B));
  PetscCall(MatCreateVecs(matctx->A,&matctx->w,NULL));
  PetscCall(VecGetType(matctx->w,&vtype));
  PetscCall(MatGetSize(matctx->A,&M,&N));
  PetscCall(MatGetLocalSize(matctx->A,&m,&n));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)eps),m,n,M,N,matctx,&Htp));
  PetscCall(MatShellSetOperation(Htp,MATOP_MULT,(void(*)(void))MatMult_Htp));
  PetscCall(MatShellSetVecType(Htp,vtype));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)eps),m,n,M,N,matctx,&Htm));
  PetscCall(MatShellSetOperation(Htm,MATOP_MULT,(void(*)(void))MatMult_Htm));
  PetscCall(MatShellSetVecType(Htm,vtype));

  /* Normalize initial vector */
  PetscCall(EPSGetStartVector(eps,0,NULL));
  PetscCall(BVGetColumn(eps->V,0,&x));
  PetscCall(VecGetSubVector(x,is[0],&u1));
  PetscCall(VecGetSubVector(x,is[1],&v1));
  PetscCall(MatMult(Htp,u1,v1));
  PetscCall(VecDot(v1,u1,&alpha));
  nrm = PetscSqrtReal(PetscRealPart(alpha));
  PetscCall(VecScale(u1,1.0/nrm));
  PetscCall(MatMult(Htp,u1,v1));
  PetscCall(VecRestoreSubVector(x,is[0],&u1));
  PetscCall(VecRestoreSubVector(x,is[1],&v1));
  PetscCall(BVRestoreColumn(eps->V,0,&x));

  /* Get the split bases */
  PetscCall(BVGetSplitRows(eps->V,is[0],is[1],&U,&V));

  l = 0;

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    PetscCall(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    PetscCall(DSGetArrayReal(eps->ds,DS_MAT_T,&a));
    b = a + ld;
    PetscCall(EPSBSELanczos(eps,Htp,Htm,U,V,a,b,eps->nconv+l,&nv,&breakdown));
    beta = b[nv-1];
    PetscCall(DSRestoreArrayReal(eps->ds,DS_MAT_T,&a));
    PetscCall(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    PetscCall(DSSetState(eps->ds,l?DS_STATE_RAW:DS_STATE_INTERMEDIATE));
    PetscCall(BVSetActiveColumns(eps->V,eps->nconv,nv));

    /* Solve projected problem */
    PetscCall(DSSolve(eps->ds,eps->eigr,eps->eigi));
    PetscCall(DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL));
    PetscCall(DSUpdateExtraRow(eps->ds));
    PetscCall(DSSynchronize(eps->ds,eps->eigr,eps->eigi));

    /* Check convergence */
    PetscCall(EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,nv-eps->nconv,beta,0.0,1.0,&k));
    PetscCall((*eps->stopping)(eps,eps->its,eps->max_it,k,eps->nev,&eps->reason,eps->stoppingctx));
    nconv = k;

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
    if (!ctx->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */
    if (l) PetscCall(PetscInfo(eps,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      PetscCheck(!breakdown,PetscObjectComm((PetscObject)eps),PETSC_ERR_CONV_FAILED,"Breakdown in BSE Krylov-Schur (beta=%g)",(double)beta);
      /* Prepare the Rayleigh quotient for restart */
      PetscCall(DSTruncate(eps->ds,k+l,PETSC_FALSE));
    }
    /* Update the corresponding vectors
       U(:,idx) = U*Q(:,idx),  V(:,idx) = V*Q(:,idx) */
    PetscCall(DSGetMat(eps->ds,DS_MAT_Q,&Q));
    PetscCall(BVMultInPlace(U,Q,eps->nconv,k+l));
    PetscCall(BVMultInPlace(V,Q,eps->nconv,k+l));
    PetscCall(DSRestoreMat(eps->ds,DS_MAT_Q,&Q));

    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) PetscCall(BVCopyColumn(eps->V,nv,k+l));
    eps->nconv = k;
    PetscCall(EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,nv));
  }

  /* Obtain eigenvalues and eigenvectors */
  for (k=0; k<eps->nconv; k++) {
    delta = 1.0/PetscSqrtReal(1.0+eps->eigr[k]);
    eps->eigr[k] = PetscSqrtReal(eps->eigr[k]);
    PetscCall(BVGetColumn(U,k,&u1));
    PetscCall(BVGetColumn(V,k,&v1));
    /* approx eigenvector is [     u1*eigr[k]*delta+v1*delta ]
                             [conj(u1*eigr[k]*delta-v1*delta)]  */
    PetscCall(VecScale(u1,eps->eigr[k]*delta));
    PetscCall(VecAXPY(u1,delta,v1));
    PetscCall(VecAYPX(v1,-2.0*delta,u1));
    PetscCall(VecConjugate(v1));
    PetscCall(BVRestoreColumn(U,k,&u1));
    PetscCall(BVRestoreColumn(V,k,&v1));
  }

  PetscCall(DSTruncate(eps->ds,eps->nconv,PETSC_TRUE));
  PetscCall(BVRestoreSplitRows(eps->V,is[0],is[1],&U,&V));
  PetscCall(VecDestroy(&matctx->w));
  PetscCall(MatDestroy(&Htp));
  PetscCall(MatDestroy(&Htm));
  PetscCall(PetscFree(matctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
