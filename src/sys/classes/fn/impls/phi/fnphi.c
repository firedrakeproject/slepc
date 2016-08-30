/*
   Phi functions
      phi_0(x) = exp(x)
      phi_1(x) = (exp(x)-1)/x
      phi_k(x) = (phi_{k-1}(x)-1/(k-1)!)/x

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/fnimpl.h>      /*I "slepcfn.h" I*/

typedef struct {
  PetscInt k;    /* index of the phi-function, defaults to k=1 */
} FN_PHI;

const static PetscReal rfactorial[] = { 1, 1, 0.5, 1.0/6, 1.0/24, 1.0/120, 1.0/720, 1.0/5040, 1.0/40320, 1.0/362880 };

static void PhiFunction(PetscScalar x,PetscScalar *y,PetscInt k)
{
  PetscScalar phi;

  if (!k) *y = PetscExpScalar(x);
  else if (k==1) *y = (PetscExpScalar(x)-1.0)/x;
  else {
    /* phi_k(x) = (phi_{k-1}(x)-1/(k-1)!)/x */
    PhiFunction(x,&phi,k-1);
    *y = (phi-rfactorial[k-1])/x;
  }
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunction_Phi"
PetscErrorCode FNEvaluateFunction_Phi(FN fn,PetscScalar x,PetscScalar *y)
{
  FN_PHI *ctx = (FN_PHI*)fn->data;

  PetscFunctionBegin;
  PhiFunction(x,y,ctx->k);
  PetscFunctionReturn(0);
}

static void PhiDerivative(PetscScalar x,PetscScalar *y,PetscInt k)
{
  PetscScalar der,phi;

  if (!k) *y = PetscExpScalar(x);
  else if (k==1) {
    der = PetscExpScalar(x);
    phi = (der-1.0)/x;
    *y = (der-phi)/x;
  } else {
    PhiDerivative(x,&der,k-1);
    PhiFunction(x,&phi,k);
    *y = (der-phi)/x;
  }
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateDerivative_Phi"
PetscErrorCode FNEvaluateDerivative_Phi(FN fn,PetscScalar x,PetscScalar *y)
{
  FN_PHI *ctx = (FN_PHI*)fn->data;

  PetscFunctionBegin;
  PhiDerivative(x,y,ctx->k);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNPhiSetIndex_Phi"
static PetscErrorCode FNPhiSetIndex_Phi(FN fn,PetscInt k)
{
  FN_PHI *ctx = (FN_PHI*)fn->data;

  PetscFunctionBegin;
  ctx->k = k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNPhiSetIndex"
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
  if (k<0) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"Index cannot be negative");
  if (k>10) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"Only implemented for k<=10");
  ierr = PetscTryMethod(fn,"FNPhiSetIndex_C",(FN,PetscInt),(fn,k));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNPhiGetIndex_Phi"
static PetscErrorCode FNPhiGetIndex_Phi(FN fn,PetscInt *k)
{
  FN_PHI *ctx = (FN_PHI*)fn->data;

  PetscFunctionBegin;
  *k = ctx->k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNPhiGetIndex"
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

#undef __FUNCT__
#define __FUNCT__ "FNView_Phi"
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

#undef __FUNCT__
#define __FUNCT__ "FNSetFromOptions_Phi"
PetscErrorCode FNSetFromOptions_Phi(PetscOptionItems *PetscOptionsObject,FN fn)
{
  PetscErrorCode ierr;
  FN_PHI         *ctx = (FN_PHI*)fn->data;
  PetscInt       k;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"FN Phi Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-fn_phi_index","Index of the phi-function","FNPhiSetIndex",ctx->k,&k,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = FNPhiSetIndex(fn,k);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNDuplicate_Phi"
PetscErrorCode FNDuplicate_Phi(FN fn,MPI_Comm comm,FN *newfn)
{
  FN_PHI *ctx = (FN_PHI*)fn->data,*ctx2 = (FN_PHI*)(*newfn)->data;

  PetscFunctionBegin;
  ctx2->k = ctx->k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNDestroy_Phi"
PetscErrorCode FNDestroy_Phi(FN fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(fn->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)fn,"FNPhiSetIndex_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)fn,"FNPhiGetIndex_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNCreate_Phi"
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

