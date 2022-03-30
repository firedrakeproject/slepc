/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   A function that is obtained by combining two other functions (either by
   addition, multiplication, division or composition)

      addition:          f(x) = f1(x)+f2(x)
      multiplication:    f(x) = f1(x)*f2(x)
      division:          f(x) = f1(x)/f2(x)      f(A) = f2(A)\f1(A)
      composition:       f(x) = f2(f1(x))
*/

#include <slepc/private/fnimpl.h>      /*I "slepcfn.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  FN            f1,f2;    /* functions */
  FNCombineType comb;     /* how the functions are combined */
} FN_COMBINE;

PetscErrorCode FNEvaluateFunction_Combine(FN fn,PetscScalar x,PetscScalar *y)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;
  PetscScalar    a,b;

  PetscFunctionBegin;
  PetscCall(FNEvaluateFunction(ctx->f1,x,&a));
  switch (ctx->comb) {
    case FN_COMBINE_ADD:
      PetscCall(FNEvaluateFunction(ctx->f2,x,&b));
      *y = a+b;
      break;
    case FN_COMBINE_MULTIPLY:
      PetscCall(FNEvaluateFunction(ctx->f2,x,&b));
      *y = a*b;
      break;
    case FN_COMBINE_DIVIDE:
      PetscCall(FNEvaluateFunction(ctx->f2,x,&b));
      PetscCheck(b!=0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Function not defined in the requested value");
      *y = a/b;
      break;
    case FN_COMBINE_COMPOSE:
      PetscCall(FNEvaluateFunction(ctx->f2,a,y));
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateDerivative_Combine(FN fn,PetscScalar x,PetscScalar *yp)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;
  PetscScalar    a,b,ap,bp;

  PetscFunctionBegin;
  switch (ctx->comb) {
    case FN_COMBINE_ADD:
      PetscCall(FNEvaluateDerivative(ctx->f1,x,&ap));
      PetscCall(FNEvaluateDerivative(ctx->f2,x,&bp));
      *yp = ap+bp;
      break;
    case FN_COMBINE_MULTIPLY:
      PetscCall(FNEvaluateDerivative(ctx->f1,x,&ap));
      PetscCall(FNEvaluateDerivative(ctx->f2,x,&bp));
      PetscCall(FNEvaluateFunction(ctx->f1,x,&a));
      PetscCall(FNEvaluateFunction(ctx->f2,x,&b));
      *yp = ap*b+a*bp;
      break;
    case FN_COMBINE_DIVIDE:
      PetscCall(FNEvaluateDerivative(ctx->f1,x,&ap));
      PetscCall(FNEvaluateDerivative(ctx->f2,x,&bp));
      PetscCall(FNEvaluateFunction(ctx->f1,x,&a));
      PetscCall(FNEvaluateFunction(ctx->f2,x,&b));
      PetscCheck(b!=0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Derivative not defined in the requested value");
      *yp = (ap*b-a*bp)/(b*b);
      break;
    case FN_COMBINE_COMPOSE:
      PetscCall(FNEvaluateFunction(ctx->f1,x,&a));
      PetscCall(FNEvaluateDerivative(ctx->f1,x,&ap));
      PetscCall(FNEvaluateDerivative(ctx->f2,a,yp));
      *yp *= ap;
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Combine(FN fn,Mat A,Mat B)
{
  FN_COMBINE        *ctx = (FN_COMBINE*)fn->data;
  PetscScalar       *Ba,*Wa,one=1.0,zero=0.0;
  const PetscScalar *Za;
  PetscBLASInt      n,ld,ld2,inc=1,*ipiv,info;
  PetscInt          m;
  Mat               W,Z;

  PetscFunctionBegin;
  PetscCall(FN_AllocateWorkMat(fn,A,&W));
  PetscCall(MatGetSize(A,&m,NULL));
  PetscCall(PetscBLASIntCast(m,&n));
  ld  = n;
  ld2 = ld*ld;

  switch (ctx->comb) {
    case FN_COMBINE_ADD:
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f1,A,W,PETSC_FALSE));
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f2,A,B,PETSC_FALSE));
      PetscCall(MatDenseGetArray(B,&Ba));
      PetscCall(MatDenseGetArray(W,&Wa));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&ld2,&one,Wa,&inc,Ba,&inc));
      PetscCall(PetscLogFlops(1.0*n*n));
      PetscCall(MatDenseRestoreArray(B,&Ba));
      PetscCall(MatDenseRestoreArray(W,&Wa));
      break;
    case FN_COMBINE_MULTIPLY:
      PetscCall(FN_AllocateWorkMat(fn,A,&Z));
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f1,A,W,PETSC_FALSE));
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f2,A,Z,PETSC_FALSE));
      PetscCall(MatDenseGetArray(B,&Ba));
      PetscCall(MatDenseGetArray(W,&Wa));
      PetscCall(MatDenseGetArrayRead(Z,&Za));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Wa,&ld,Za,&ld,&zero,Ba,&ld));
      PetscCall(PetscLogFlops(2.0*n*n*n));
      PetscCall(MatDenseRestoreArray(B,&Ba));
      PetscCall(MatDenseRestoreArray(W,&Wa));
      PetscCall(MatDenseRestoreArrayRead(Z,&Za));
      PetscCall(FN_FreeWorkMat(fn,&Z));
      break;
    case FN_COMBINE_DIVIDE:
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f2,A,W,PETSC_FALSE));
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f1,A,B,PETSC_FALSE));
      PetscCall(PetscMalloc1(ld,&ipiv));
      PetscCall(MatDenseGetArray(B,&Ba));
      PetscCall(MatDenseGetArray(W,&Wa));
      PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&n,Wa,&ld,ipiv,Ba,&ld,&info));
      SlepcCheckLapackInfo("gesv",info);
      PetscCall(PetscLogFlops(2.0*n*n*n/3.0+2.0*n*n*n));
      PetscCall(MatDenseRestoreArray(B,&Ba));
      PetscCall(MatDenseRestoreArray(W,&Wa));
      PetscCall(PetscFree(ipiv));
      break;
    case FN_COMBINE_COMPOSE:
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f1,A,W,PETSC_FALSE));
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f2,W,B,PETSC_FALSE));
      break;
  }

  PetscCall(FN_FreeWorkMat(fn,&W));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMatVec_Combine(FN fn,Mat A,Vec v)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;
  PetscScalar    *va,*Za;
  PetscBLASInt   n,ld,*ipiv,info,one=1;
  PetscInt       m;
  Mat            Z;
  Vec            w;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&m,NULL));
  PetscCall(PetscBLASIntCast(m,&n));
  ld = n;

  switch (ctx->comb) {
    case FN_COMBINE_ADD:
      PetscCall(VecDuplicate(v,&w));
      PetscCall(FNEvaluateFunctionMatVec(ctx->f1,A,w));
      PetscCall(FNEvaluateFunctionMatVec(ctx->f2,A,v));
      PetscCall(VecAXPY(v,1.0,w));
      PetscCall(VecDestroy(&w));
      break;
    case FN_COMBINE_MULTIPLY:
      PetscCall(VecDuplicate(v,&w));
      PetscCall(FN_AllocateWorkMat(fn,A,&Z));
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f1,A,Z,PETSC_FALSE));
      PetscCall(FNEvaluateFunctionMatVec_Private(ctx->f2,A,w,PETSC_FALSE));
      PetscCall(MatMult(Z,w,v));
      PetscCall(FN_FreeWorkMat(fn,&Z));
      PetscCall(VecDestroy(&w));
      break;
    case FN_COMBINE_DIVIDE:
      PetscCall(VecDuplicate(v,&w));
      PetscCall(FN_AllocateWorkMat(fn,A,&Z));
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f2,A,Z,PETSC_FALSE));
      PetscCall(FNEvaluateFunctionMatVec_Private(ctx->f1,A,v,PETSC_FALSE));
      PetscCall(PetscMalloc1(ld,&ipiv));
      PetscCall(MatDenseGetArray(Z,&Za));
      PetscCall(VecGetArray(v,&va));
      PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&one,Za,&ld,ipiv,va,&ld,&info));
      SlepcCheckLapackInfo("gesv",info);
      PetscCall(PetscLogFlops(2.0*n*n*n/3.0+2.0*n*n));
      PetscCall(VecRestoreArray(v,&va));
      PetscCall(MatDenseRestoreArray(Z,&Za));
      PetscCall(PetscFree(ipiv));
      PetscCall(FN_FreeWorkMat(fn,&Z));
      PetscCall(VecDestroy(&w));
      break;
    case FN_COMBINE_COMPOSE:
      PetscCall(FN_AllocateWorkMat(fn,A,&Z));
      PetscCall(FNEvaluateFunctionMat_Private(ctx->f1,A,Z,PETSC_FALSE));
      PetscCall(FNEvaluateFunctionMatVec_Private(ctx->f2,Z,v,PETSC_FALSE));
      PetscCall(FN_FreeWorkMat(fn,&Z));
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FNView_Combine(FN fn,PetscViewer viewer)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    switch (ctx->comb) {
      case FN_COMBINE_ADD:
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Two added functions f1+f2\n"));
        break;
      case FN_COMBINE_MULTIPLY:
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Two multiplied functions f1*f2\n"));
        break;
      case FN_COMBINE_DIVIDE:
        PetscCall(PetscViewerASCIIPrintf(viewer,"  A quotient of two functions f1/f2\n"));
        break;
      case FN_COMBINE_COMPOSE:
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Two composed functions f2(f1(.))\n"));
        break;
    }
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(FNView(ctx->f1,viewer));
    PetscCall(FNView(ctx->f2,viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FNCombineSetChildren_Combine(FN fn,FNCombineType comb,FN f1,FN f2)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;

  PetscFunctionBegin;
  ctx->comb = comb;
  PetscCall(PetscObjectReference((PetscObject)f1));
  PetscCall(FNDestroy(&ctx->f1));
  ctx->f1 = f1;
  PetscCall(PetscLogObjectParent((PetscObject)fn,(PetscObject)ctx->f1));
  PetscCall(PetscObjectReference((PetscObject)f2));
  PetscCall(FNDestroy(&ctx->f2));
  ctx->f2 = f2;
  PetscCall(PetscLogObjectParent((PetscObject)fn,(PetscObject)ctx->f2));
  PetscFunctionReturn(0);
}

/*@
   FNCombineSetChildren - Sets the two child functions that constitute this
   combined function, and the way they must be combined.

   Logically Collective on fn

   Input Parameters:
+  fn   - the math function context
.  comb - how to combine the functions (addition, multiplication, division or composition)
.  f1   - first function
-  f2   - second function

   Level: intermediate

.seealso: FNCombineGetChildren()
@*/
PetscErrorCode FNCombineSetChildren(FN fn,FNCombineType comb,FN f1,FN f2)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidLogicalCollectiveEnum(fn,comb,2);
  PetscValidHeaderSpecific(f1,FN_CLASSID,3);
  PetscValidHeaderSpecific(f2,FN_CLASSID,4);
  PetscTryMethod(fn,"FNCombineSetChildren_C",(FN,FNCombineType,FN,FN),(fn,comb,f1,f2));
  PetscFunctionReturn(0);
}

static PetscErrorCode FNCombineGetChildren_Combine(FN fn,FNCombineType *comb,FN *f1,FN *f2)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;

  PetscFunctionBegin;
  if (comb) *comb = ctx->comb;
  if (f1) {
    if (!ctx->f1) {
      PetscCall(FNCreate(PetscObjectComm((PetscObject)fn),&ctx->f1));
      PetscCall(PetscLogObjectParent((PetscObject)fn,(PetscObject)ctx->f1));
    }
    *f1 = ctx->f1;
  }
  if (f2) {
    if (!ctx->f2) {
      PetscCall(FNCreate(PetscObjectComm((PetscObject)fn),&ctx->f2));
      PetscCall(PetscLogObjectParent((PetscObject)fn,(PetscObject)ctx->f2));
    }
    *f2 = ctx->f2;
  }
  PetscFunctionReturn(0);
}

/*@
   FNCombineGetChildren - Gets the two child functions that constitute this
   combined function, and the way they are combined.

   Not Collective

   Input Parameter:
.  fn   - the math function context

   Output Parameters:
+  comb - how to combine the functions (addition, multiplication, division or composition)
.  f1   - first function
-  f2   - second function

   Level: intermediate

.seealso: FNCombineSetChildren()
@*/
PetscErrorCode FNCombineGetChildren(FN fn,FNCombineType *comb,FN *f1,FN *f2)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscUseMethod(fn,"FNCombineGetChildren_C",(FN,FNCombineType*,FN*,FN*),(fn,comb,f1,f2));
  PetscFunctionReturn(0);
}

PetscErrorCode FNDuplicate_Combine(FN fn,MPI_Comm comm,FN *newfn)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data,*ctx2 = (FN_COMBINE*)(*newfn)->data;

  PetscFunctionBegin;
  ctx2->comb = ctx->comb;
  PetscCall(FNDuplicate(ctx->f1,comm,&ctx2->f1));
  PetscCall(FNDuplicate(ctx->f2,comm,&ctx2->f2));
  PetscFunctionReturn(0);
}

PetscErrorCode FNDestroy_Combine(FN fn)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;

  PetscFunctionBegin;
  PetscCall(FNDestroy(&ctx->f1));
  PetscCall(FNDestroy(&ctx->f2));
  PetscCall(PetscFree(fn->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)fn,"FNCombineSetChildren_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)fn,"FNCombineGetChildren_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode FNCreate_Combine(FN fn)
{
  FN_COMBINE     *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(fn,&ctx));
  fn->data = (void*)ctx;

  fn->ops->evaluatefunction          = FNEvaluateFunction_Combine;
  fn->ops->evaluatederivative        = FNEvaluateDerivative_Combine;
  fn->ops->evaluatefunctionmat[0]    = FNEvaluateFunctionMat_Combine;
  fn->ops->evaluatefunctionmatvec[0] = FNEvaluateFunctionMatVec_Combine;
  fn->ops->view                      = FNView_Combine;
  fn->ops->duplicate                 = FNDuplicate_Combine;
  fn->ops->destroy                   = FNDestroy_Combine;
  PetscCall(PetscObjectComposeFunction((PetscObject)fn,"FNCombineSetChildren_C",FNCombineSetChildren_Combine));
  PetscCall(PetscObjectComposeFunction((PetscObject)fn,"FNCombineGetChildren_C",FNCombineGetChildren_Combine));
  PetscFunctionReturn(0);
}
