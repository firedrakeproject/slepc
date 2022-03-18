/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
  CHKERRQ(FNEvaluateFunction(ctx->f1,x,&a));
  switch (ctx->comb) {
    case FN_COMBINE_ADD:
      CHKERRQ(FNEvaluateFunction(ctx->f2,x,&b));
      *y = a+b;
      break;
    case FN_COMBINE_MULTIPLY:
      CHKERRQ(FNEvaluateFunction(ctx->f2,x,&b));
      *y = a*b;
      break;
    case FN_COMBINE_DIVIDE:
      CHKERRQ(FNEvaluateFunction(ctx->f2,x,&b));
      PetscCheck(b!=0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Function not defined in the requested value");
      *y = a/b;
      break;
    case FN_COMBINE_COMPOSE:
      CHKERRQ(FNEvaluateFunction(ctx->f2,a,y));
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
      CHKERRQ(FNEvaluateDerivative(ctx->f1,x,&ap));
      CHKERRQ(FNEvaluateDerivative(ctx->f2,x,&bp));
      *yp = ap+bp;
      break;
    case FN_COMBINE_MULTIPLY:
      CHKERRQ(FNEvaluateDerivative(ctx->f1,x,&ap));
      CHKERRQ(FNEvaluateDerivative(ctx->f2,x,&bp));
      CHKERRQ(FNEvaluateFunction(ctx->f1,x,&a));
      CHKERRQ(FNEvaluateFunction(ctx->f2,x,&b));
      *yp = ap*b+a*bp;
      break;
    case FN_COMBINE_DIVIDE:
      CHKERRQ(FNEvaluateDerivative(ctx->f1,x,&ap));
      CHKERRQ(FNEvaluateDerivative(ctx->f2,x,&bp));
      CHKERRQ(FNEvaluateFunction(ctx->f1,x,&a));
      CHKERRQ(FNEvaluateFunction(ctx->f2,x,&b));
      PetscCheck(b!=0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Derivative not defined in the requested value");
      *yp = (ap*b-a*bp)/(b*b);
      break;
    case FN_COMBINE_COMPOSE:
      CHKERRQ(FNEvaluateFunction(ctx->f1,x,&a));
      CHKERRQ(FNEvaluateDerivative(ctx->f1,x,&ap));
      CHKERRQ(FNEvaluateDerivative(ctx->f2,a,yp));
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
  CHKERRQ(FN_AllocateWorkMat(fn,A,&W));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(PetscBLASIntCast(m,&n));
  ld  = n;
  ld2 = ld*ld;

  switch (ctx->comb) {
    case FN_COMBINE_ADD:
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f1,A,W,PETSC_FALSE));
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f2,A,B,PETSC_FALSE));
      CHKERRQ(MatDenseGetArray(B,&Ba));
      CHKERRQ(MatDenseGetArray(W,&Wa));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&ld2,&one,Wa,&inc,Ba,&inc));
      CHKERRQ(PetscLogFlops(1.0*n*n));
      CHKERRQ(MatDenseRestoreArray(B,&Ba));
      CHKERRQ(MatDenseRestoreArray(W,&Wa));
      break;
    case FN_COMBINE_MULTIPLY:
      CHKERRQ(FN_AllocateWorkMat(fn,A,&Z));
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f1,A,W,PETSC_FALSE));
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f2,A,Z,PETSC_FALSE));
      CHKERRQ(MatDenseGetArray(B,&Ba));
      CHKERRQ(MatDenseGetArray(W,&Wa));
      CHKERRQ(MatDenseGetArrayRead(Z,&Za));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Wa,&ld,Za,&ld,&zero,Ba,&ld));
      CHKERRQ(PetscLogFlops(2.0*n*n*n));
      CHKERRQ(MatDenseRestoreArray(B,&Ba));
      CHKERRQ(MatDenseRestoreArray(W,&Wa));
      CHKERRQ(MatDenseRestoreArrayRead(Z,&Za));
      CHKERRQ(FN_FreeWorkMat(fn,&Z));
      break;
    case FN_COMBINE_DIVIDE:
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f2,A,W,PETSC_FALSE));
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f1,A,B,PETSC_FALSE));
      CHKERRQ(PetscMalloc1(ld,&ipiv));
      CHKERRQ(MatDenseGetArray(B,&Ba));
      CHKERRQ(MatDenseGetArray(W,&Wa));
      PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&n,Wa,&ld,ipiv,Ba,&ld,&info));
      SlepcCheckLapackInfo("gesv",info);
      CHKERRQ(PetscLogFlops(2.0*n*n*n/3.0+2.0*n*n*n));
      CHKERRQ(MatDenseRestoreArray(B,&Ba));
      CHKERRQ(MatDenseRestoreArray(W,&Wa));
      CHKERRQ(PetscFree(ipiv));
      break;
    case FN_COMBINE_COMPOSE:
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f1,A,W,PETSC_FALSE));
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f2,W,B,PETSC_FALSE));
      break;
  }

  CHKERRQ(FN_FreeWorkMat(fn,&W));
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
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(PetscBLASIntCast(m,&n));
  ld = n;

  switch (ctx->comb) {
    case FN_COMBINE_ADD:
      CHKERRQ(VecDuplicate(v,&w));
      CHKERRQ(FNEvaluateFunctionMatVec(ctx->f1,A,w));
      CHKERRQ(FNEvaluateFunctionMatVec(ctx->f2,A,v));
      CHKERRQ(VecAXPY(v,1.0,w));
      CHKERRQ(VecDestroy(&w));
      break;
    case FN_COMBINE_MULTIPLY:
      CHKERRQ(VecDuplicate(v,&w));
      CHKERRQ(FN_AllocateWorkMat(fn,A,&Z));
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f1,A,Z,PETSC_FALSE));
      CHKERRQ(FNEvaluateFunctionMatVec_Private(ctx->f2,A,w,PETSC_FALSE));
      CHKERRQ(MatMult(Z,w,v));
      CHKERRQ(FN_FreeWorkMat(fn,&Z));
      CHKERRQ(VecDestroy(&w));
      break;
    case FN_COMBINE_DIVIDE:
      CHKERRQ(VecDuplicate(v,&w));
      CHKERRQ(FN_AllocateWorkMat(fn,A,&Z));
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f2,A,Z,PETSC_FALSE));
      CHKERRQ(FNEvaluateFunctionMatVec_Private(ctx->f1,A,v,PETSC_FALSE));
      CHKERRQ(PetscMalloc1(ld,&ipiv));
      CHKERRQ(MatDenseGetArray(Z,&Za));
      CHKERRQ(VecGetArray(v,&va));
      PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&one,Za,&ld,ipiv,va,&ld,&info));
      SlepcCheckLapackInfo("gesv",info);
      CHKERRQ(PetscLogFlops(2.0*n*n*n/3.0+2.0*n*n));
      CHKERRQ(VecRestoreArray(v,&va));
      CHKERRQ(MatDenseRestoreArray(Z,&Za));
      CHKERRQ(PetscFree(ipiv));
      CHKERRQ(FN_FreeWorkMat(fn,&Z));
      CHKERRQ(VecDestroy(&w));
      break;
    case FN_COMBINE_COMPOSE:
      CHKERRQ(FN_AllocateWorkMat(fn,A,&Z));
      CHKERRQ(FNEvaluateFunctionMat_Private(ctx->f1,A,Z,PETSC_FALSE));
      CHKERRQ(FNEvaluateFunctionMatVec_Private(ctx->f2,Z,v,PETSC_FALSE));
      CHKERRQ(FN_FreeWorkMat(fn,&Z));
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FNView_Combine(FN fn,PetscViewer viewer)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    switch (ctx->comb) {
      case FN_COMBINE_ADD:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Two added functions f1+f2\n"));
        break;
      case FN_COMBINE_MULTIPLY:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Two multiplied functions f1*f2\n"));
        break;
      case FN_COMBINE_DIVIDE:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  A quotient of two functions f1/f2\n"));
        break;
      case FN_COMBINE_COMPOSE:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Two composed functions f2(f1(.))\n"));
        break;
    }
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(FNView(ctx->f1,viewer));
    CHKERRQ(FNView(ctx->f2,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FNCombineSetChildren_Combine(FN fn,FNCombineType comb,FN f1,FN f2)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;

  PetscFunctionBegin;
  ctx->comb = comb;
  CHKERRQ(PetscObjectReference((PetscObject)f1));
  CHKERRQ(FNDestroy(&ctx->f1));
  ctx->f1 = f1;
  CHKERRQ(PetscLogObjectParent((PetscObject)fn,(PetscObject)ctx->f1));
  CHKERRQ(PetscObjectReference((PetscObject)f2));
  CHKERRQ(FNDestroy(&ctx->f2));
  ctx->f2 = f2;
  CHKERRQ(PetscLogObjectParent((PetscObject)fn,(PetscObject)ctx->f2));
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
  CHKERRQ(PetscTryMethod(fn,"FNCombineSetChildren_C",(FN,FNCombineType,FN,FN),(fn,comb,f1,f2)));
  PetscFunctionReturn(0);
}

static PetscErrorCode FNCombineGetChildren_Combine(FN fn,FNCombineType *comb,FN *f1,FN *f2)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;

  PetscFunctionBegin;
  if (comb) *comb = ctx->comb;
  if (f1) {
    if (!ctx->f1) {
      CHKERRQ(FNCreate(PetscObjectComm((PetscObject)fn),&ctx->f1));
      CHKERRQ(PetscLogObjectParent((PetscObject)fn,(PetscObject)ctx->f1));
    }
    *f1 = ctx->f1;
  }
  if (f2) {
    if (!ctx->f2) {
      CHKERRQ(FNCreate(PetscObjectComm((PetscObject)fn),&ctx->f2));
      CHKERRQ(PetscLogObjectParent((PetscObject)fn,(PetscObject)ctx->f2));
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
  CHKERRQ(PetscUseMethod(fn,"FNCombineGetChildren_C",(FN,FNCombineType*,FN*,FN*),(fn,comb,f1,f2)));
  PetscFunctionReturn(0);
}

PetscErrorCode FNDuplicate_Combine(FN fn,MPI_Comm comm,FN *newfn)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data,*ctx2 = (FN_COMBINE*)(*newfn)->data;

  PetscFunctionBegin;
  ctx2->comb = ctx->comb;
  CHKERRQ(FNDuplicate(ctx->f1,comm,&ctx2->f1));
  CHKERRQ(FNDuplicate(ctx->f2,comm,&ctx2->f2));
  PetscFunctionReturn(0);
}

PetscErrorCode FNDestroy_Combine(FN fn)
{
  FN_COMBINE     *ctx = (FN_COMBINE*)fn->data;

  PetscFunctionBegin;
  CHKERRQ(FNDestroy(&ctx->f1));
  CHKERRQ(FNDestroy(&ctx->f2));
  CHKERRQ(PetscFree(fn->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNCombineSetChildren_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNCombineGetChildren_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode FNCreate_Combine(FN fn)
{
  FN_COMBINE     *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(fn,&ctx));
  fn->data = (void*)ctx;

  fn->ops->evaluatefunction          = FNEvaluateFunction_Combine;
  fn->ops->evaluatederivative        = FNEvaluateDerivative_Combine;
  fn->ops->evaluatefunctionmat[0]    = FNEvaluateFunctionMat_Combine;
  fn->ops->evaluatefunctionmatvec[0] = FNEvaluateFunctionMatVec_Combine;
  fn->ops->view                      = FNView_Combine;
  fn->ops->duplicate                 = FNDuplicate_Combine;
  fn->ops->destroy                   = FNDestroy_Combine;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNCombineSetChildren_C",FNCombineSetChildren_Combine));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNCombineGetChildren_C",FNCombineGetChildren_Combine));
  PetscFunctionReturn(0);
}
