/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   NEP routines related to resolvent T^{-1}(z) = sum_i (z-lambda_i)^{-1} x_i y_i'
*/

#include <slepc/private/nepimpl.h>       /*I "slepcnep.h" I*/

typedef struct {
  NEP              nep;
  RG               rg;
  PetscScalar      omega;
  PetscScalar      *nfactor;         /* normalization factors y_i'*T'(lambda_i)*x_i */
  PetscBool        *nfactor_avail;
  PetscScalar      *dots;            /* inner products y_i'*v */
  PetscBool        *dots_avail;
  PetscObjectId    vid;
  PetscObjectState vstate;
} NEP_RESOLVENT_MATSHELL;

static PetscErrorCode MatMult_Resolvent(Mat M,Vec v,Vec r)
{
  NEP_RESOLVENT_MATSHELL *ctx;
  NEP                    nep;
  PetscInt               i,inside=1;
  PetscScalar            alpha;
  Vec                    x,y,z,w;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  nep = ctx->nep;
  w = nep->work[0];
  z = nep->work[1];
  if (((PetscObject)v)->id != ctx->vid || ((PetscObject)v)->state != ctx->vstate) {
    CHKERRQ(PetscArrayzero(ctx->dots_avail,ctx->nep->nconv));
    CHKERRQ(PetscObjectGetId((PetscObject)v,&ctx->vid));
    CHKERRQ(PetscObjectStateGet((PetscObject)v,&ctx->vstate));
  }
  CHKERRQ(VecSet(r,0.0));
  for (i=0;i<nep->nconv;i++) {
    if (ctx->rg) {
      CHKERRQ(RGCheckInside(ctx->rg,1,&nep->eigr[i],&nep->eigi[i],&inside));
    }
    if (inside>=0) {
      CHKERRQ(BVGetColumn(nep->V,i,&x));
      CHKERRQ(BVGetColumn(nep->W,i,&y));
      CHKERRQ(NEPApplyJacobian(nep,nep->eigr[i],x,z,w,NULL));
      if (!ctx->dots_avail[i]) {
        CHKERRQ(VecDot(v,y,&ctx->dots[i]));
        ctx->dots_avail[i] = PETSC_TRUE;
      }
      if (!ctx->nfactor_avail[i]) {
        CHKERRQ(VecDot(w,y,&ctx->nfactor[i]));
        ctx->nfactor_avail[i] = PETSC_TRUE;
      }
      alpha = ctx->dots[i]/(ctx->nfactor[i]*(ctx->omega-nep->eigr[i]));
      CHKERRQ(VecAXPY(r,alpha,x));
      CHKERRQ(BVRestoreColumn(nep->V,i,&x));
      CHKERRQ(BVRestoreColumn(nep->W,i,&y));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Resolvent(Mat M)
{
  NEP_RESOLVENT_MATSHELL *ctx;

  PetscFunctionBegin;
  if (M) {
    CHKERRQ(MatShellGetContext(M,&ctx));
    CHKERRQ(PetscFree4(ctx->nfactor,ctx->nfactor_avail,ctx->dots,ctx->dots_avail));
    CHKERRQ(PetscFree(ctx));
  }
  PetscFunctionReturn(0);
}

/*@
   NEPApplyResolvent - Applies the resolvent T^{-1}(z) to a given vector.

   Collective on nep

   Input Parameters:
+  nep   - eigensolver context obtained from NEPCreate()
.  rg    - optional region
.  omega - value where the resolvent must be evaluated
-  v     - input vector

   Output Parameter:
.  r     - result vector

   Notes:
   The resolvent T^{-1}(z) = sum_i (z-lambda_i)^{-1}*x_i*y_i' is evaluated at
   z=omega and the matrix-vector multiplication r = T^{-1}(omega)*v is computed.
   Vectors x_i and y_i are right and left eigenvectors, respectively, normalized
   so that y_i'*T'(lambda_i)*x_i=1. The sum contains only eigenvectors that have
   been previously computed with NEPSolve(), and if a region rg is given then only
   those corresponding to eigenvalues inside the region are considered.

   Level: intermediate

.seealso: NEPGetLeftEigenvector(), NEPSolve()
@*/
PetscErrorCode NEPApplyResolvent(NEP nep,RG rg,PetscScalar omega,Vec v,Vec r)
{
  NEP_RESOLVENT_MATSHELL *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveScalar(nep,omega,3);
  PetscValidHeaderSpecific(v,VEC_CLASSID,4);
  PetscValidHeaderSpecific(r,VEC_CLASSID,5);
  NEPCheckSolved(nep,1);

  CHKERRQ(PetscLogEventBegin(NEP_Resolvent,nep,0,0,0));
  if (!nep->resolvent) {
    CHKERRQ(PetscNew(&ctx));
    ctx->nep = nep;
    CHKERRQ(PetscCalloc4(nep->nconv,&ctx->nfactor,nep->nconv,&ctx->nfactor_avail,nep->nconv,&ctx->dots,nep->nconv,&ctx->dots_avail));
    CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)nep),nep->nloc,nep->nloc,nep->n,nep->n,ctx,&nep->resolvent));
    CHKERRQ(MatShellSetOperation(nep->resolvent,MATOP_MULT,(void(*)(void))MatMult_Resolvent));
    CHKERRQ(MatShellSetOperation(nep->resolvent,MATOP_DESTROY,(void(*)(void))MatDestroy_Resolvent));
  } else {
    CHKERRQ(MatShellGetContext(nep->resolvent,&ctx));
  }
  CHKERRQ(NEPComputeVectors(nep));
  CHKERRQ(NEPSetWorkVecs(nep,2));
  ctx->rg    = rg;
  ctx->omega = omega;
  CHKERRQ(MatMult(nep->resolvent,v,r));
  CHKERRQ(PetscLogEventEnd(NEP_Resolvent,nep,0,0,0));
  PetscFunctionReturn(0);
}
