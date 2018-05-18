/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Phi functions
      phi_0(x) = exp(x)
      phi_1(x) = (exp(x)-1)/x
      phi_k(x) = (phi_{k-1}(x)-1/(k-1)!)/x
*/

#include <slepc/private/fnimpl.h>      /*I "slepcfn.h" I*/

typedef struct {
  PetscInt k;    /* index of the phi-function, defaults to k=1 */
} FN_PHI;

#define MAX_INDEX 10

const static PetscReal rfactorial[MAX_INDEX+2] = { 1, 1, 0.5, 1.0/6, 1.0/24, 1.0/120, 1.0/720, 1.0/5040, 1.0/40320, 1.0/362880, 1.0/3628800, 1.0/39916800 };

PetscErrorCode FNEvaluateFunction_Phi(FN fn,PetscScalar x,PetscScalar *y)
{
  FN_PHI      *ctx = (FN_PHI*)fn->data;
  PetscInt    i;
  PetscScalar phi[MAX_INDEX+1];

  PetscFunctionBegin;
  if (x==0.0) *y = rfactorial[ctx->k];
  else {
    phi[0] = PetscExpScalar(x);
    for (i=1;i<=ctx->k;i++) phi[i] = (phi[i-1]-rfactorial[i-1])/x;
    *y = phi[ctx->k];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateDerivative_Phi(FN fn,PetscScalar x,PetscScalar *y)
{
  FN_PHI      *ctx = (FN_PHI*)fn->data;
  PetscInt    i;
  PetscScalar phi[MAX_INDEX+2];

  PetscFunctionBegin;
  if (x==0.0) *y = rfactorial[ctx->k+1];
  else {
    phi[0] = PetscExpScalar(x);
    for (i=1;i<=ctx->k+1;i++) phi[i] = (phi[i-1]-rfactorial[i-1])/x;
    *y = phi[ctx->k] - ctx->k*phi[ctx->k+1];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FNPhiSetIndex_Phi(FN fn,PetscInt k)
{
  FN_PHI *ctx = (FN_PHI*)fn->data;

  PetscFunctionBegin;
  if (k<0) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"Index cannot be negative");
  if (k>MAX_INDEX) SETERRQ1(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"Phi functions only implemented for k<=%d",MAX_INDEX);
  ctx->k = k;
  PetscFunctionReturn(0);
}

/*@
   FNPhiSetIndex - Sets the index of the phi-function.

   Logically Collective on FN

   Input Parameters:
+  fn - the math function context
-  k  - the index

   Notes:
   The phi-functions are defined as follows. The default is k=1.
.vb
      phi_0(x) = exp(x)
      phi_1(x) = (exp(x)-1)/x
      phi_k(x) = (phi_{k-1}(x)-1/(k-1)!)/x
.ve

   Level: intermediate

.seealso: FNPhiGetIndex()
@*/
PetscErrorCode FNPhiSetIndex(FN fn,PetscInt k)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidLogicalCollectiveInt(fn,k,2);
  ierr = PetscTryMethod(fn,"FNPhiSetIndex_C",(FN,PetscInt),(fn,k));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FNPhiGetIndex_Phi(FN fn,PetscInt *k)
{
  FN_PHI *ctx = (FN_PHI*)fn->data;

  PetscFunctionBegin;
  *k = ctx->k;
  PetscFunctionReturn(0);
}

/*@
   FNPhiGetIndex - Gets the index of the phi-function.

   Not Collective

   Input Parameter:
.  fn - the math function context

   Output Parameter:
.  k  - the index

   Level: intermediate

.seealso: FNPhiSetIndex()
@*/
PetscErrorCode FNPhiGetIndex(FN fn,PetscInt *k)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidPointer(k,2);
  ierr = PetscUseMethod(fn,"FNPhiGetIndex_C",(FN,PetscInt*),(fn,k));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FNView_Phi(FN fn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  FN_PHI         *ctx = (FN_PHI*)fn->data;
  PetscBool      isascii;
  char           str[50],strx[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Phi_%D: ",ctx->k);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    if (fn->beta!=(PetscScalar)1.0) {
      ierr = SlepcSNPrintfScalar(str,50,fn->beta,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s*",str);CHKERRQ(ierr);
    }
    if (fn->alpha==(PetscScalar)1.0) {
      ierr = PetscSNPrintf(strx,50,"x");CHKERRQ(ierr);
    } else {
      ierr = SlepcSNPrintfScalar(str,50,fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscSNPrintf(strx,50,"(%s*x)",str);CHKERRQ(ierr);
    }
    if (!ctx->k) {
      ierr = PetscViewerASCIIPrintf(viewer,"exp(%s)\n",strx);CHKERRQ(ierr);
    } else if (ctx->k==1) {
      ierr = PetscViewerASCIIPrintf(viewer,"(exp(%s)-1)/%s\n",strx,strx);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"(phi_%D(%s)-1/%D!)/%s\n",ctx->k-1,strx,ctx->k-1,strx);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FNSetFromOptions_Phi(PetscOptionItems *PetscOptionsObject,FN fn)
{
  PetscErrorCode ierr;
  FN_PHI         *ctx = (FN_PHI*)fn->data;
  PetscInt       k;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"FN Phi Options");CHKERRQ(ierr);

    ierr = PetscOptionsInt("-fn_phi_index","Index of the phi-function","FNPhiSetIndex",ctx->k,&k,&flag);CHKERRQ(ierr);
    if (flag) { ierr = FNPhiSetIndex(fn,k);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FNDuplicate_Phi(FN fn,MPI_Comm comm,FN *newfn)
{
  FN_PHI *ctx = (FN_PHI*)fn->data,*ctx2 = (FN_PHI*)(*newfn)->data;

  PetscFunctionBegin;
  ctx2->k = ctx->k;
  PetscFunctionReturn(0);
}

PetscErrorCode FNDestroy_Phi(FN fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(fn->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)fn,"FNPhiSetIndex_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)fn,"FNPhiGetIndex_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode FNCreate_Phi(FN fn)
{
  PetscErrorCode ierr;
  FN_PHI         *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(fn,&ctx);CHKERRQ(ierr);
  fn->data = (void*)ctx;
  ctx->k   = 1;

  fn->ops->evaluatefunction    = FNEvaluateFunction_Phi;
  fn->ops->evaluatederivative  = FNEvaluateDerivative_Phi;
  fn->ops->setfromoptions      = FNSetFromOptions_Phi;
  fn->ops->view                = FNView_Phi;
  fn->ops->duplicate           = FNDuplicate_Phi;
  fn->ops->destroy             = FNDestroy_Phi;
  ierr = PetscObjectComposeFunction((PetscObject)fn,"FNPhiSetIndex_C",FNPhiSetIndex_Phi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)fn,"FNPhiGetIndex_C",FNPhiGetIndex_Phi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

