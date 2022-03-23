/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Rational function  r(x) = p(x)/q(x), where p(x) and q(x) are polynomials
*/

#include <slepc/private/fnimpl.h>      /*I "slepcfn.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  PetscScalar *pcoeff;    /* numerator coefficients */
  PetscInt    np;         /* length of array pcoeff, p(x) has degree np-1 */
  PetscScalar *qcoeff;    /* denominator coefficients */
  PetscInt    nq;         /* length of array qcoeff, q(x) has degree nq-1 */
} FN_RATIONAL;

PetscErrorCode FNEvaluateFunction_Rational(FN fn,PetscScalar x,PetscScalar *y)
{
  FN_RATIONAL *ctx = (FN_RATIONAL*)fn->data;
  PetscInt    i;
  PetscScalar p,q;

  PetscFunctionBegin;
  if (!ctx->np) p = 1.0;
  else {
    p = ctx->pcoeff[0];
    for (i=1;i<ctx->np;i++)
      p = ctx->pcoeff[i]+x*p;
  }
  if (!ctx->nq) *y = p;
  else {
    q = ctx->qcoeff[0];
    for (i=1;i<ctx->nq;i++)
      q = ctx->qcoeff[i]+x*q;
    PetscCheck(q!=0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Function not defined in the requested value");
    *y = p/q;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FNEvaluateFunctionMat_Rational_Private(FN fn,const PetscScalar *Aa,PetscScalar *Ba,PetscInt m,PetscBool firstonly)
{
  FN_RATIONAL    *ctx = (FN_RATIONAL*)fn->data;
  PetscBLASInt   n,k,ld,*ipiv,info;
  PetscInt       i,j;
  PetscScalar    *W,*P,*Q,one=1.0,zero=0.0;

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(m,&n));
  ld = n;
  k  = firstonly? 1: n;
  if (Aa==Ba) CHKERRQ(PetscMalloc4(m*m,&P,m*m,&Q,m*m,&W,ld,&ipiv));
  else {
    P = Ba;
    CHKERRQ(PetscMalloc3(m*m,&Q,m*m,&W,ld,&ipiv));
  }
  CHKERRQ(PetscArrayzero(P,m*m));
  if (!ctx->np) {
    for (i=0;i<m;i++) P[i+i*ld] = 1.0;
  } else {
    for (i=0;i<m;i++) P[i+i*ld] = ctx->pcoeff[0];
    for (j=1;j<ctx->np;j++) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,P,&ld,Aa,&ld,&zero,W,&ld));
      CHKERRQ(PetscArraycpy(P,W,m*m));
      for (i=0;i<m;i++) P[i+i*ld] += ctx->pcoeff[j];
    }
    CHKERRQ(PetscLogFlops(2.0*n*n*n*(ctx->np-1)));
  }
  if (ctx->nq) {
    CHKERRQ(PetscArrayzero(Q,m*m));
    for (i=0;i<m;i++) Q[i+i*ld] = ctx->qcoeff[0];
    for (j=1;j<ctx->nq;j++) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Q,&ld,Aa,&ld,&zero,W,&ld));
      CHKERRQ(PetscArraycpy(Q,W,m*m));
      for (i=0;i<m;i++) Q[i+i*ld] += ctx->qcoeff[j];
    }
    PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&k,Q,&ld,ipiv,P,&ld,&info));
    SlepcCheckLapackInfo("gesv",info);
    CHKERRQ(PetscLogFlops(2.0*n*n*n*(ctx->nq-1)+2.0*n*n*n/3.0+2.0*n*n*k));
  }
  if (Aa==Ba) {
    CHKERRQ(PetscArraycpy(Ba,P,m*k));
    CHKERRQ(PetscFree4(P,Q,W,ipiv));
  } else CHKERRQ(PetscFree3(Q,W,ipiv));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Rational(FN fn,Mat A,Mat B)
{
  PetscInt          m;
  const PetscScalar *Aa;
  PetscScalar       *Ba;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArrayRead(A,&Aa));
  CHKERRQ(MatDenseGetArray(B,&Ba));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(FNEvaluateFunctionMat_Rational_Private(fn,Aa,Ba,m,PETSC_FALSE));
  CHKERRQ(MatDenseRestoreArrayRead(A,&Aa));
  CHKERRQ(MatDenseRestoreArray(B,&Ba));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMatVec_Rational(FN fn,Mat A,Vec v)
{
  PetscInt          m;
  const PetscScalar *Aa;
  PetscScalar       *Ba;
  Mat               B;

  PetscFunctionBegin;
  CHKERRQ(FN_AllocateWorkMat(fn,A,&B));
  CHKERRQ(MatDenseGetArrayRead(A,&Aa));
  CHKERRQ(MatDenseGetArray(B,&Ba));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(FNEvaluateFunctionMat_Rational_Private(fn,Aa,Ba,m,PETSC_TRUE));
  CHKERRQ(MatDenseRestoreArrayRead(A,&Aa));
  CHKERRQ(MatDenseRestoreArray(B,&Ba));
  CHKERRQ(MatGetColumnVector(B,v,0));
  CHKERRQ(FN_FreeWorkMat(fn,&B));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateDerivative_Rational(FN fn,PetscScalar x,PetscScalar *yp)
{
  FN_RATIONAL *ctx = (FN_RATIONAL*)fn->data;
  PetscInt    i;
  PetscScalar p,q,pp,qp;

  PetscFunctionBegin;
  if (!ctx->np) {
    p = 1.0;
    pp = 0.0;
  } else {
    p = ctx->pcoeff[0];
    pp = 0.0;
    for (i=1;i<ctx->np;i++) {
      pp = p+x*pp;
      p = ctx->pcoeff[i]+x*p;
    }
  }
  if (!ctx->nq) *yp = pp;
  else {
    q = ctx->qcoeff[0];
    qp = 0.0;
    for (i=1;i<ctx->nq;i++) {
      qp = q+x*qp;
      q = ctx->qcoeff[i]+x*q;
    }
    PetscCheck(q!=0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Derivative not defined in the requested value");
    *yp = (pp*q-p*qp)/(q*q);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FNView_Rational(FN fn,PetscViewer viewer)
{
  FN_RATIONAL    *ctx = (FN_RATIONAL*)fn->data;
  PetscBool      isascii;
  PetscInt       i;
  char           str[50];

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    if (fn->alpha!=(PetscScalar)1.0 || fn->beta!=(PetscScalar)1.0) {
      CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),fn->alpha,PETSC_FALSE));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Scale factors: alpha=%s,",str));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
      CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),fn->beta,PETSC_FALSE));
      CHKERRQ(PetscViewerASCIIPrintf(viewer," beta=%s\n",str));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    }
    if (!ctx->nq) {
      if (!ctx->np) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Constant: 1.0\n"));
      else if (ctx->np==1) {
        CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),ctx->pcoeff[0],PETSC_FALSE));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Constant: %s\n",str));
      } else {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Polynomial: "));
        CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
        for (i=0;i<ctx->np-1;i++) {
          CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),ctx->pcoeff[i],PETSC_TRUE));
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s*x^%1" PetscInt_FMT,str,ctx->np-i-1));
        }
        CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),ctx->pcoeff[ctx->np-1],PETSC_TRUE));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s\n",str));
        CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
      }
    } else if (!ctx->np) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Inverse polinomial: 1 / ("));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
      for (i=0;i<ctx->nq-1;i++) {
        CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),ctx->qcoeff[i],PETSC_TRUE));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s*x^%1" PetscInt_FMT,str,ctx->nq-i-1));
      }
      CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),ctx->qcoeff[ctx->nq-1],PETSC_TRUE));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s)\n",str));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Rational function: ("));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
      for (i=0;i<ctx->np-1;i++) {
        CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),ctx->pcoeff[i],PETSC_TRUE));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s*x^%1" PetscInt_FMT,str,ctx->np-i-1));
      }
      CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),ctx->pcoeff[ctx->np-1],PETSC_TRUE));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s) / (",str));
      for (i=0;i<ctx->nq-1;i++) {
        CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),ctx->qcoeff[i],PETSC_TRUE));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s*x^%1" PetscInt_FMT,str,ctx->nq-i-1));
      }
      CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),ctx->qcoeff[ctx->nq-1],PETSC_TRUE));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s)\n",str));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FNRationalSetNumerator_Rational(FN fn,PetscInt np,PetscScalar *pcoeff)
{
  FN_RATIONAL    *ctx = (FN_RATIONAL*)fn->data;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheck(np>=0,PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"Argument np cannot be negative");
  ctx->np = np;
  CHKERRQ(PetscFree(ctx->pcoeff));
  if (np) {
    CHKERRQ(PetscMalloc1(np,&ctx->pcoeff));
    CHKERRQ(PetscLogObjectMemory((PetscObject)fn,np*sizeof(PetscScalar)));
    for (i=0;i<np;i++) ctx->pcoeff[i] = pcoeff[i];
  }
  PetscFunctionReturn(0);
}

/*@C
   FNRationalSetNumerator - Sets the parameters defining the numerator of the
   rational function.

   Logically Collective on fn

   Input Parameters:
+  fn     - the math function context
.  np     - number of coefficients
-  pcoeff - coefficients (array of scalar values)

   Notes:
   Let the rational function r(x) = p(x)/q(x), where p(x) and q(x) are polynomials.
   This function provides the coefficients of the numerator p(x).
   Hence, p(x) is of degree np-1.
   If np is zero, then the numerator is assumed to be p(x)=1.

   In polynomials, high order coefficients are stored in the first positions
   of the array, e.g. to represent x^2-3 use {1,0,-3}.

   Level: intermediate

.seealso: FNRationalSetDenominator(), FNRationalGetNumerator()
@*/
PetscErrorCode FNRationalSetNumerator(FN fn,PetscInt np,PetscScalar *pcoeff)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidLogicalCollectiveInt(fn,np,2);
  if (np) PetscValidScalarPointer(pcoeff,3);
  CHKERRQ(PetscTryMethod(fn,"FNRationalSetNumerator_C",(FN,PetscInt,PetscScalar*),(fn,np,pcoeff)));
  PetscFunctionReturn(0);
}

static PetscErrorCode FNRationalGetNumerator_Rational(FN fn,PetscInt *np,PetscScalar *pcoeff[])
{
  FN_RATIONAL    *ctx = (FN_RATIONAL*)fn->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (np) *np = ctx->np;
  if (pcoeff) {
    if (!ctx->np) *pcoeff = NULL;
    else {
      CHKERRQ(PetscMalloc1(ctx->np,pcoeff));
      for (i=0;i<ctx->np;i++) (*pcoeff)[i] = ctx->pcoeff[i];
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   FNRationalGetNumerator - Gets the parameters that define the numerator of the
   rational function.

   Not Collective

   Input Parameter:
.  fn     - the math function context

   Output Parameters:
+  np     - number of coefficients
-  pcoeff - coefficients (array of scalar values, length nq)

   Notes:
   The values passed by user with FNRationalSetNumerator() are returned (or null
   pointers otherwise).
   The pcoeff array should be freed by the user when no longer needed.

   Level: intermediate

.seealso: FNRationalSetNumerator()
@*/
PetscErrorCode FNRationalGetNumerator(FN fn,PetscInt *np,PetscScalar *pcoeff[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  CHKERRQ(PetscUseMethod(fn,"FNRationalGetNumerator_C",(FN,PetscInt*,PetscScalar**),(fn,np,pcoeff)));
  PetscFunctionReturn(0);
}

static PetscErrorCode FNRationalSetDenominator_Rational(FN fn,PetscInt nq,PetscScalar *qcoeff)
{
  FN_RATIONAL    *ctx = (FN_RATIONAL*)fn->data;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheck(nq>=0,PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"Argument nq cannot be negative");
  ctx->nq = nq;
  CHKERRQ(PetscFree(ctx->qcoeff));
  if (nq) {
    CHKERRQ(PetscMalloc1(nq,&ctx->qcoeff));
    CHKERRQ(PetscLogObjectMemory((PetscObject)fn,nq*sizeof(PetscScalar)));
    for (i=0;i<nq;i++) ctx->qcoeff[i] = qcoeff[i];
  }
  PetscFunctionReturn(0);
}

/*@C
   FNRationalSetDenominator - Sets the parameters defining the denominator of the
   rational function.

   Logically Collective on fn

   Input Parameters:
+  fn     - the math function context
.  nq     - number of coefficients
-  qcoeff - coefficients (array of scalar values)

   Notes:
   Let the rational function r(x) = p(x)/q(x), where p(x) and q(x) are polynomials.
   This function provides the coefficients of the denominator q(x).
   Hence, q(x) is of degree nq-1.
   If nq is zero, then the function is assumed to be polynomial, r(x) = p(x).

   In polynomials, high order coefficients are stored in the first positions
   of the array, e.g. to represent x^2-3 use {1,0,-3}.

   Level: intermediate

.seealso: FNRationalSetNumerator(), FNRationalGetDenominator()
@*/
PetscErrorCode FNRationalSetDenominator(FN fn,PetscInt nq,PetscScalar *qcoeff)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidLogicalCollectiveInt(fn,nq,2);
  if (nq) PetscValidScalarPointer(qcoeff,3);
  CHKERRQ(PetscTryMethod(fn,"FNRationalSetDenominator_C",(FN,PetscInt,PetscScalar*),(fn,nq,qcoeff)));
  PetscFunctionReturn(0);
}

static PetscErrorCode FNRationalGetDenominator_Rational(FN fn,PetscInt *nq,PetscScalar *qcoeff[])
{
  FN_RATIONAL    *ctx = (FN_RATIONAL*)fn->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (nq) *nq = ctx->nq;
  if (qcoeff) {
    if (!ctx->nq) *qcoeff = NULL;
    else {
      CHKERRQ(PetscMalloc1(ctx->nq,qcoeff));
      for (i=0;i<ctx->nq;i++) (*qcoeff)[i] = ctx->qcoeff[i];
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   FNRationalGetDenominator - Gets the parameters that define the denominator of the
   rational function.

   Not Collective

   Input Parameter:
.  fn     - the math function context

   Output Parameters:
+  nq     - number of coefficients
-  qcoeff - coefficients (array of scalar values, length nq)

   Notes:
   The values passed by user with FNRationalSetDenominator() are returned (or a null
   pointer otherwise).
   The qcoeff array should be freed by the user when no longer needed.

   Level: intermediate

.seealso: FNRationalSetDenominator()
@*/
PetscErrorCode FNRationalGetDenominator(FN fn,PetscInt *nq,PetscScalar *qcoeff[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  CHKERRQ(PetscUseMethod(fn,"FNRationalGetDenominator_C",(FN,PetscInt*,PetscScalar**),(fn,nq,qcoeff)));
  PetscFunctionReturn(0);
}

PetscErrorCode FNSetFromOptions_Rational(PetscOptionItems *PetscOptionsObject,FN fn)
{
#define PARMAX 10
  PetscScalar    array[PARMAX];
  PetscInt       i,k;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"FN Rational Options"));

    k = PARMAX;
    for (i=0;i<k;i++) array[i] = 0;
    CHKERRQ(PetscOptionsScalarArray("-fn_rational_numerator","Numerator coefficients (one or more scalar values separated with a comma without spaces)","FNRationalSetNumerator",array,&k,&flg));
    if (flg) CHKERRQ(FNRationalSetNumerator(fn,k,array));

    k = PARMAX;
    for (i=0;i<k;i++) array[i] = 0;
    CHKERRQ(PetscOptionsScalarArray("-fn_rational_denominator","Denominator coefficients (one or more scalar values separated with a comma without spaces)","FNRationalSetDenominator",array,&k,&flg));
    if (flg) CHKERRQ(FNRationalSetDenominator(fn,k,array));

  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode FNDuplicate_Rational(FN fn,MPI_Comm comm,FN *newfn)
{
  FN_RATIONAL    *ctx = (FN_RATIONAL*)fn->data,*ctx2 = (FN_RATIONAL*)(*newfn)->data;
  PetscInt       i;

  PetscFunctionBegin;
  ctx2->np = ctx->np;
  if (ctx->np) {
    CHKERRQ(PetscMalloc1(ctx->np,&ctx2->pcoeff));
    CHKERRQ(PetscLogObjectMemory((PetscObject)(*newfn),ctx->np*sizeof(PetscScalar)));
    for (i=0;i<ctx->np;i++) ctx2->pcoeff[i] = ctx->pcoeff[i];
  }
  ctx2->nq = ctx->nq;
  if (ctx->nq) {
    CHKERRQ(PetscMalloc1(ctx->nq,&ctx2->qcoeff));
    CHKERRQ(PetscLogObjectMemory((PetscObject)(*newfn),ctx->nq*sizeof(PetscScalar)));
    for (i=0;i<ctx->nq;i++) ctx2->qcoeff[i] = ctx->qcoeff[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FNDestroy_Rational(FN fn)
{
  FN_RATIONAL    *ctx = (FN_RATIONAL*)fn->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(ctx->pcoeff));
  CHKERRQ(PetscFree(ctx->qcoeff));
  CHKERRQ(PetscFree(fn->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNRationalSetNumerator_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNRationalGetNumerator_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNRationalSetDenominator_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNRationalGetDenominator_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode FNCreate_Rational(FN fn)
{
  FN_RATIONAL    *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(fn,&ctx));
  fn->data = (void*)ctx;

  fn->ops->evaluatefunction          = FNEvaluateFunction_Rational;
  fn->ops->evaluatederivative        = FNEvaluateDerivative_Rational;
  fn->ops->evaluatefunctionmat[0]    = FNEvaluateFunctionMat_Rational;
  fn->ops->evaluatefunctionmatvec[0] = FNEvaluateFunctionMatVec_Rational;
  fn->ops->setfromoptions            = FNSetFromOptions_Rational;
  fn->ops->view                      = FNView_Rational;
  fn->ops->duplicate                 = FNDuplicate_Rational;
  fn->ops->destroy                   = FNDestroy_Rational;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNRationalSetNumerator_C",FNRationalSetNumerator_Rational));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNRationalGetNumerator_C",FNRationalGetNumerator_Rational));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNRationalSetDenominator_C",FNRationalSetDenominator_Rational));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)fn,"FNRationalGetDenominator_C",FNRationalGetDenominator_Rational));
  PetscFunctionReturn(0);
}
