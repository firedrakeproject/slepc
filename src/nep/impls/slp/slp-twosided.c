/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Two-sided variant of the NEPSLP solver.
*/

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include "slp.h"

typedef struct _n_nep_def_ctx *NEP_NEDEF_CTX;

struct _n_nep_def_ctx {
  PetscInt    n;
  PetscBool   ref;
  PetscScalar *eig;
  BV          V,W;
};

typedef struct {   /* context for two-sided solver */
  Mat         Ft;
  Mat         Jt;
  Vec         w;
} NEP_SLPTS_MATSHELL;

typedef struct {   /* context for non-equivalence deflation */
  NEP_NEDEF_CTX defctx;
  Mat           F;
  Mat           P;
  Mat           J;
  KSP           ksp;
  PetscBool     isJ;
  PetscScalar   lambda;
  Vec           w[2];
} NEP_NEDEF_MATSHELL;

static PetscErrorCode MatMult_SLPTS_Right(Mat M,Vec x,Vec y)
{
  NEP_SLPTS_MATSHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(MatMult(ctx->Jt,x,ctx->w));
  CHKERRQ(MatSolve(ctx->Ft,ctx->w,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SLPTS_Left(Mat M,Vec x,Vec y)
{
  NEP_SLPTS_MATSHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(MatMultTranspose(ctx->Jt,x,ctx->w));
  CHKERRQ(MatSolveTranspose(ctx->Ft,ctx->w,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SLPTS(Mat M)
{
  NEP_SLPTS_MATSHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(VecDestroy(&ctx->w));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
static PetscErrorCode MatCreateVecs_SLPTS(Mat M,Vec *left,Vec *right)
{
  NEP_SLPTS_MATSHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  if (right) {
    CHKERRQ(VecDuplicate(ctx->w,right));
  }
  if (left) {
    CHKERRQ(VecDuplicate(ctx->w,left));
  }
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode NEPSLPSetUpEPSMat(NEP nep,Mat F,Mat J,PetscBool left,Mat *M)
{
  Mat                Mshell;
  PetscInt           nloc,mloc;
  NEP_SLPTS_MATSHELL *shellctx;

  PetscFunctionBegin;
  /* Create mat shell */
  CHKERRQ(PetscNew(&shellctx));
  shellctx->Ft = F;
  shellctx->Jt = J;
  CHKERRQ(MatGetLocalSize(nep->function,&mloc,&nloc));
  CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)nep),nloc,mloc,PETSC_DETERMINE,PETSC_DETERMINE,shellctx,&Mshell));
  if (left) {
    CHKERRQ(MatShellSetOperation(Mshell,MATOP_MULT,(void(*)(void))MatMult_SLPTS_Left));
  } else {
    CHKERRQ(MatShellSetOperation(Mshell,MATOP_MULT,(void(*)(void))MatMult_SLPTS_Right));
  }
  CHKERRQ(MatShellSetOperation(Mshell,MATOP_DESTROY,(void(*)(void))MatDestroy_SLPTS));
#if defined(PETSC_HAVE_CUDA)
  CHKERRQ(MatShellSetOperation(Mshell,MATOP_CREATE_VECS,(void(*)(void))MatCreateVecs_SLPTS));
#endif
  *M = Mshell;
  CHKERRQ(MatCreateVecs(nep->function,&shellctx->w,NULL));
  PetscFunctionReturn(0);
}

/* Functions for deflation */
static PetscErrorCode NEPDeflationNEDestroy(NEP_NEDEF_CTX defctx)
{
  PetscFunctionBegin;
  if (!defctx) PetscFunctionReturn(0);
  CHKERRQ(PetscFree(defctx->eig));
  CHKERRQ(PetscFree(defctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationNECreate(NEP nep,BV V,BV W,PetscInt sz,NEP_NEDEF_CTX *defctx)
{
  NEP_NEDEF_CTX  op;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&op));
  *defctx = op;
  op->n   = 0;
  op->ref = PETSC_FALSE;
  CHKERRQ(PetscCalloc1(sz,&op->eig));
  CHKERRQ(PetscObjectStateIncrease((PetscObject)V));
  CHKERRQ(PetscObjectStateIncrease((PetscObject)W));
  op->V = V;
  op->W = W;
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationNEComputeFunction(NEP nep,Mat M,PetscScalar lambda)
{
  NEP_NEDEF_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  if (lambda==matctx->lambda) PetscFunctionReturn(0);
  CHKERRQ(NEPComputeFunction(nep,lambda,matctx->F,matctx->P));
  if (matctx->isJ) CHKERRQ(NEPComputeJacobian(nep,lambda,matctx->J));
  if (matctx->ksp) CHKERRQ(NEP_KSPSetOperators(matctx->ksp,matctx->F,matctx->P));
  matctx->lambda = lambda;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_NEPDeflationNE(Mat M,Vec x,Vec r)
{
  Vec                t,tt;
  PetscScalar        *h,*alpha,lambda,*eig;
  PetscInt           i,k;
  NEP_NEDEF_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  if (matctx->defctx->n && !matctx->defctx->ref) {
    k = matctx->defctx->n;
    lambda = matctx->lambda;
    eig = matctx->defctx->eig;
    t = matctx->w[0];
    CHKERRQ(VecCopy(x,t));
    CHKERRQ(PetscMalloc2(k,&h,k,&alpha));
    for (i=0;i<k;i++) alpha[i] = (lambda-eig[i]-1.0)/(lambda-eig[i]);
    CHKERRQ(BVDotVec(matctx->defctx->V,t,h));
    for (i=0;i<k;i++) h[i] *= alpha[i];
    CHKERRQ(BVMultVec(matctx->defctx->W,-1.0,1.0,t,h));
    CHKERRQ(MatMult(matctx->isJ?matctx->J:matctx->F,t,r));
    if (matctx->isJ) {
      for (i=0;i<k;i++) h[i] *= (1.0/((lambda-eig[i])*(lambda-eig[i])))/alpha[i];
      tt = matctx->w[1];
      CHKERRQ(BVMultVec(matctx->defctx->W,-1.0,0.0,tt,h));
      CHKERRQ(MatMult(matctx->F,tt,t));
      CHKERRQ(VecAXPY(r,1.0,t));
    }
    CHKERRQ(PetscFree2(h,alpha));
  } else {
    CHKERRQ(MatMult(matctx->isJ?matctx->J:matctx->F,x,r));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_NEPDeflationNE(Mat M,Vec x,Vec r)
{
  Vec                t,tt;
  PetscScalar        *h,*alphaC,lambda,*eig;
  PetscInt           i,k;
  NEP_NEDEF_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  t    = matctx->w[0];
  CHKERRQ(VecCopy(x,t));
  if (matctx->defctx->n && !matctx->defctx->ref) {
    CHKERRQ(VecConjugate(t));
    k = matctx->defctx->n;
    lambda = matctx->lambda;
    eig = matctx->defctx->eig;
    CHKERRQ(PetscMalloc2(k,&h,k,&alphaC));
    for (i=0;i<k;i++) alphaC[i] = PetscConj((lambda-eig[i]-1.0)/(lambda-eig[i]));
    CHKERRQ(BVDotVec(matctx->defctx->W,t,h));
    for (i=0;i<k;i++) h[i] *= alphaC[i];
    CHKERRQ(BVMultVec(matctx->defctx->V,-1.0,1.0,t,h));
    CHKERRQ(VecConjugate(t));
    CHKERRQ(MatMultTranspose(matctx->isJ?matctx->J:matctx->F,t,r));
    if (matctx->isJ) {
      for (i=0;i<k;i++) h[i] *= PetscConj(1.0/((lambda-eig[i])*(lambda-eig[i])))/alphaC[i];
      tt = matctx->w[1];
      CHKERRQ(BVMultVec(matctx->defctx->V,-1.0,0.0,tt,h));
      CHKERRQ(VecConjugate(tt));
      CHKERRQ(MatMultTranspose(matctx->F,tt,t));
      CHKERRQ(VecAXPY(r,1.0,t));
    }
    CHKERRQ(PetscFree2(h,alphaC));
  } else {
    CHKERRQ(MatMultTranspose(matctx->isJ?matctx->J:matctx->F,t,r));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_NEPDeflationNE(Mat M,Vec b,Vec x)
{
  PetscScalar        *h,*alpha,lambda,*eig;
  PetscInt           i,k;
  NEP_NEDEF_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  if (!matctx->ksp) {
    CHKERRQ(VecCopy(b,x));
    PetscFunctionReturn(0);
  }
  CHKERRQ(KSPSolve(matctx->ksp,b,x));
  if (matctx->defctx->n && !matctx->defctx->ref) {
    k = matctx->defctx->n;
    lambda = matctx->lambda;
    eig = matctx->defctx->eig;
    CHKERRQ(PetscMalloc2(k,&h,k,&alpha));
    CHKERRQ(BVDotVec(matctx->defctx->V,x,h));
    for (i=0;i<k;i++) alpha[i] = (lambda-eig[i]-1.0)/(lambda-eig[i]);
    for (i=0;i<k;i++) h[i] *= alpha[i]/(1.0-alpha[i]);
    CHKERRQ(BVMultVec(matctx->defctx->W,1.0,1.0,x,h));
    CHKERRQ(PetscFree2(h,alpha));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_NEPDeflationNE(Mat M,Vec b,Vec x)
{
  PetscScalar        *h,*alphaC,lambda,*eig;
  PetscInt           i,k;
  NEP_NEDEF_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  if (!matctx->ksp) {
    CHKERRQ(VecCopy(b,x));
    PetscFunctionReturn(0);
  }
  CHKERRQ(KSPSolveTranspose(matctx->ksp,b,x));
  if (matctx->defctx->n && !matctx->defctx->ref) {
    CHKERRQ(VecConjugate(x));
    k = matctx->defctx->n;
    lambda = matctx->lambda;
    eig = matctx->defctx->eig;
    CHKERRQ(PetscMalloc2(k,&h,k,&alphaC));
    CHKERRQ(BVDotVec(matctx->defctx->W,x,h));
    for (i=0;i<k;i++) alphaC[i] = PetscConj((lambda-eig[i]-1.0)/(lambda-eig[i]));
    for (i=0;i<k;i++) h[i] *= alphaC[i]/(1.0-alphaC[i]);
    CHKERRQ(BVMultVec(matctx->defctx->V,1.0,1.0,x,h));
    CHKERRQ(PetscFree2(h,alphaC));
    CHKERRQ(VecConjugate(x));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_NEPDeflationNE(Mat M)
{
  NEP_NEDEF_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  CHKERRQ(VecDestroy(&matctx->w[0]));
  CHKERRQ(VecDestroy(&matctx->w[1]));
  CHKERRQ(PetscFree(matctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateVecs_NEPDeflationNE(Mat M,Vec *right,Vec *left)
{
  NEP_NEDEF_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  CHKERRQ(MatCreateVecs(matctx->F,right,left));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationNEFunctionCreate(NEP_NEDEF_CTX defctx,NEP nep,Mat F,Mat P,Mat J,KSP ksp,PetscBool isJ,Mat *Mshell)
{
  NEP_NEDEF_MATSHELL *matctx;
  PetscInt           nloc,mloc;

  PetscFunctionBegin;
  /* Create mat shell */
  CHKERRQ(PetscNew(&matctx));
  CHKERRQ(MatGetLocalSize(nep->function,&mloc,&nloc));
  CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)nep),nloc,mloc,PETSC_DETERMINE,PETSC_DETERMINE,matctx,Mshell));
  matctx->F   = F;
  matctx->P   = P;
  matctx->J   = J;
  matctx->isJ = isJ;
  matctx->ksp = ksp;
  matctx->defctx = defctx;
  matctx->lambda = PETSC_MAX_REAL;
  CHKERRQ(MatCreateVecs(F,&matctx->w[0],NULL));
  CHKERRQ(VecDuplicate(matctx->w[0],&matctx->w[1]));
  CHKERRQ(MatShellSetOperation(*Mshell,MATOP_MULT,(void(*)(void))MatMult_NEPDeflationNE));
  CHKERRQ(MatShellSetOperation(*Mshell,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_NEPDeflationNE));
  CHKERRQ(MatShellSetOperation(*Mshell,MATOP_SOLVE,(void(*)(void))MatSolve_NEPDeflationNE));
  CHKERRQ(MatShellSetOperation(*Mshell,MATOP_SOLVE_TRANSPOSE,(void(*)(void))MatSolveTranspose_NEPDeflationNE));
  CHKERRQ(MatShellSetOperation(*Mshell,MATOP_DESTROY,(void(*)(void))MatDestroy_NEPDeflationNE));
  CHKERRQ(MatShellSetOperation(*Mshell,MATOP_CREATE_VECS,(void(*)(void))MatCreateVecs_NEPDeflationNE));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationNERecoverEigenvectors(NEP_NEDEF_CTX defctx,Vec u,Vec w,PetscScalar lambda)
{
  PetscScalar    *h,*alpha,*eig;
  PetscInt       i,k;

  PetscFunctionBegin;
  if (w) CHKERRQ(VecConjugate(w));
  if (defctx->n && !defctx->ref) {
    eig = defctx->eig;
    k = defctx->n;
    CHKERRQ(PetscMalloc2(k,&h,k,&alpha));
    for (i=0;i<k;i++) alpha[i] = (lambda-eig[i]-1.0)/(lambda-eig[i]);
    CHKERRQ(BVDotVec(defctx->V,u,h));
    for (i=0;i<k;i++) h[i] *= alpha[i];
    CHKERRQ(BVMultVec(defctx->W,-1.0,1.0,u,h));
    CHKERRQ(VecNormalize(u,NULL));
    if (w) {
      CHKERRQ(BVDotVec(defctx->W,w,h));
      for (i=0;i<k;i++) alpha[i] = PetscConj((lambda-eig[i]-1.0)/(lambda-eig[i]));
      for (i=0;i<k;i++) h[i] *= alpha[i];
      CHKERRQ(BVMultVec(defctx->V,-1.0,1.0,w,h));
      CHKERRQ(VecNormalize(w,NULL));
    }
    CHKERRQ(PetscFree2(h,alpha));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationNELocking(NEP_NEDEF_CTX defctx,Vec u,Vec w,PetscScalar lambda)
{
  PetscInt       n;

  PetscFunctionBegin;
  n = defctx->n++;
  defctx->eig[n] = lambda;
  CHKERRQ(BVInsertVec(defctx->V,n,u));
  CHKERRQ(BVInsertVec(defctx->W,n,w));
  CHKERRQ(BVSetActiveColumns(defctx->V,0,defctx->n));
  CHKERRQ(BVSetActiveColumns(defctx->W,0,defctx->n));
  CHKERRQ(BVBiorthonormalizeColumn(defctx->V,defctx->W,n,NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationNESetRefine(NEP_NEDEF_CTX defctx,PetscBool ref)
{
  PetscFunctionBegin;
  defctx->ref = ref;
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSolve_SLP_Twosided(NEP nep)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;
  Mat            mF,mJ,M,Mt;
  Vec            u,r,t,w;
  BV             X,Y;
  PetscScalar    sigma,lambda,mu,im=0.0,mu2,im2;
  PetscReal      resnorm,resl;
  PetscInt       nconv,nconv2,i;
  PetscBool      skip=PETSC_FALSE,lock=PETSC_FALSE;
  NEP_NEDEF_CTX  defctx=NULL;    /* Extended operator for deflation */

  PetscFunctionBegin;
  /* get initial approximation of eigenvalue and eigenvector */
  CHKERRQ(NEPGetDefaultShift(nep,&sigma));
  if (!nep->nini) {
    CHKERRQ(BVSetRandomColumn(nep->V,0));
  }
  CHKERRQ(BVSetRandomColumn(nep->W,0));
  lambda = sigma;
  if (!ctx->ksp) CHKERRQ(NEPSLPGetKSP(nep,&ctx->ksp));
  CHKERRQ(BVDuplicate(nep->V,&X));
  CHKERRQ(BVDuplicate(nep->W,&Y));
  CHKERRQ(NEPDeflationNECreate(nep,X,Y,nep->nev,&defctx));
  CHKERRQ(BVGetColumn(nep->V,0,&t));
  CHKERRQ(VecDuplicate(t,&u));
  CHKERRQ(VecDuplicate(t,&w));
  CHKERRQ(BVRestoreColumn(nep->V,0,&t));
  CHKERRQ(BVCopyVec(nep->V,0,u));
  CHKERRQ(BVCopyVec(nep->W,0,w));
  CHKERRQ(VecDuplicate(u,&r));
  CHKERRQ(NEPDeflationNEFunctionCreate(defctx,nep,nep->function,nep->function_pre?nep->function_pre:nep->function,NULL,ctx->ksp,PETSC_FALSE,&mF));
  CHKERRQ(NEPDeflationNEFunctionCreate(defctx,nep,nep->function,nep->function,nep->jacobian,NULL,PETSC_TRUE,&mJ));
  CHKERRQ(NEPSLPSetUpEPSMat(nep,mF,mJ,PETSC_FALSE,&M));
  CHKERRQ(NEPSLPSetUpEPSMat(nep,mF,mJ,PETSC_TRUE,&Mt));
  CHKERRQ(EPSSetOperators(ctx->eps,M,NULL));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(EPSSetOperators(ctx->epsts,Mt,NULL));
  CHKERRQ(MatDestroy(&Mt));

  /* Restart loop */
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;

    /* form residual,  r = T(lambda)*u (used in convergence test only) */
    CHKERRQ(NEPDeflationNEComputeFunction(nep,mF,lambda));
    CHKERRQ(MatMultTranspose(mF,w,r));
    CHKERRQ(VecNorm(r,NORM_2,&resl));
    CHKERRQ(MatMult(mF,u,r));

    /* convergence test */
    CHKERRQ(VecNorm(r,NORM_2,&resnorm));
    resnorm = PetscMax(resnorm,resl);
    CHKERRQ((*nep->converged)(nep,lambda,0,resnorm,&nep->errest[nep->nconv],nep->convergedctx));
    nep->eigr[nep->nconv] = lambda;
    if (nep->errest[nep->nconv]<=nep->tol || nep->errest[nep->nconv]<=ctx->deftol) {
      if (nep->errest[nep->nconv]<=ctx->deftol && !defctx->ref && nep->nconv) {
        CHKERRQ(NEPDeflationNERecoverEigenvectors(defctx,u,w,lambda));
        CHKERRQ(VecConjugate(w));
        CHKERRQ(NEPDeflationNESetRefine(defctx,PETSC_TRUE));
        CHKERRQ(MatMultTranspose(mF,w,r));
        CHKERRQ(VecNorm(r,NORM_2,&resl));
        CHKERRQ(MatMult(mF,u,r));
        CHKERRQ(VecNorm(r,NORM_2,&resnorm));
        resnorm = PetscMax(resnorm,resl);
        CHKERRQ((*nep->converged)(nep,lambda,0,resnorm,&nep->errest[nep->nconv],nep->convergedctx));
        if (nep->errest[nep->nconv]<=nep->tol) lock = PETSC_TRUE;
      } else if (nep->errest[nep->nconv]<=nep->tol) lock = PETSC_TRUE;
    }
    if (lock) {
      lock = PETSC_FALSE;
      skip = PETSC_TRUE;
      CHKERRQ(NEPDeflationNERecoverEigenvectors(defctx,u,w,lambda));
      CHKERRQ(NEPDeflationNELocking(defctx,u,w,lambda));
      CHKERRQ(NEPDeflationNESetRefine(defctx,PETSC_FALSE));
      CHKERRQ(BVInsertVec(nep->V,nep->nconv,u));
      CHKERRQ(BVInsertVec(nep->W,nep->nconv,w));
      CHKERRQ(VecConjugate(w));
      nep->nconv = nep->nconv + 1;
    }
    CHKERRQ((*nep->stopping)(nep,nep->its,nep->max_it,nep->nconv,nep->nev,&nep->reason,nep->stoppingctx));
    if (!skip || nep->reason>0) {
      CHKERRQ(NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->eigi,nep->errest,(nep->reason>0)?nep->nconv:nep->nconv+1));
    }

    if (nep->reason == NEP_CONVERGED_ITERATING) {
      if (!skip) {
        /* evaluate T(lambda) and T'(lambda) */
        CHKERRQ(NEPDeflationNEComputeFunction(nep,mF,lambda));
        CHKERRQ(NEPDeflationNEComputeFunction(nep,mJ,lambda));
        CHKERRQ(EPSSetInitialSpace(ctx->eps,1,&u));
        CHKERRQ(EPSSetInitialSpace(ctx->epsts,1,&w));

        /* compute new eigenvalue correction mu and eigenvector approximation u */
        CHKERRQ(EPSSolve(ctx->eps));
        CHKERRQ(EPSSolve(ctx->epsts));
        CHKERRQ(EPSGetConverged(ctx->eps,&nconv));
        CHKERRQ(EPSGetConverged(ctx->epsts,&nconv2));
        if (!nconv||!nconv2) {
          CHKERRQ(PetscInfo(nep,"iter=%" PetscInt_FMT ", inner iteration failed, stopping solve\n",nep->its));
          nep->reason = NEP_DIVERGED_LINEAR_SOLVE;
          break;
        }
        CHKERRQ(EPSGetEigenpair(ctx->eps,0,&mu,&im,u,NULL));
        for (i=0;i<nconv2;i++) {
          CHKERRQ(EPSGetEigenpair(ctx->epsts,i,&mu2,&im2,w,NULL));
          if (SlepcAbsEigenvalue(mu-mu2,im-im2)/SlepcAbsEigenvalue(mu,im)<nep->tol*1000) break;
        }
        if (i==nconv2) {
          CHKERRQ(PetscInfo(nep,"iter=%" PetscInt_FMT ", inner iteration failed, stopping solve\n",nep->its));
          nep->reason = NEP_DIVERGED_LINEAR_SOLVE;
          break;
        }

        mu = 1.0/mu;
        PetscCheck(PetscAbsScalar(im)<PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Complex eigenvalue approximation - not implemented in real scalars");
      } else {
        nep->its--;  /* do not count this as a full iteration */
        /* use second eigenpair computed in previous iteration */
        CHKERRQ(EPSGetConverged(ctx->eps,&nconv));
        if (nconv>=2 && nconv2>=2) {
          CHKERRQ(EPSGetEigenpair(ctx->eps,1,&mu,&im,u,NULL));
          CHKERRQ(EPSGetEigenpair(ctx->epsts,1,&mu2,&im2,w,NULL));
          mu = 1.0/mu;
        } else {
          CHKERRQ(BVSetRandomColumn(nep->V,nep->nconv));
          CHKERRQ(BVSetRandomColumn(nep->W,nep->nconv));
          CHKERRQ(BVCopyVec(nep->V,nep->nconv,u));
          CHKERRQ(BVCopyVec(nep->W,nep->nconv,w));
          mu = lambda-sigma;
        }
        skip = PETSC_FALSE;
      }
      /* correct eigenvalue */
      lambda = lambda - mu;
    }
  }
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&w));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(MatDestroy(&mF));
  CHKERRQ(MatDestroy(&mJ));
  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Y));
  CHKERRQ(NEPDeflationNEDestroy(defctx));
  PetscFunctionReturn(0);
}
