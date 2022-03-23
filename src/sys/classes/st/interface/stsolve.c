/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   ST interface routines, callable by users
*/

#include <slepc/private/stimpl.h>            /*I "slepcst.h" I*/

PetscErrorCode STApply_Generic(ST st,Vec x,Vec y)
{
  PetscFunctionBegin;
  if (st->M && st->P) {
    CHKERRQ(MatMult(st->M,x,st->work[0]));
    CHKERRQ(STMatSolve(st,st->work[0],y));
  } else if (st->M) CHKERRQ(MatMult(st->M,x,y));
  else CHKERRQ(STMatSolve(st,x,y));
  PetscFunctionReturn(0);
}

/*@
   STApply - Applies the spectral transformation operator to a vector, for
   instance (A - sB)^-1 B in the case of the shift-and-invert transformation
   and generalized eigenproblem.

   Collective on st

   Input Parameters:
+  st - the spectral transformation context
-  x  - input vector

   Output Parameter:
.  y - output vector

   Level: developer

.seealso: STApplyTranspose(), STApplyHermitianTranspose()
@*/
PetscErrorCode STApply(ST st,Vec x,Vec y)
{
  Mat            Op;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidType(st,1);
  STCheckMatrices(st,1);
  PetscCheck(x!=y,PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  CHKERRQ(VecSetErrorIfLocked(y,3));
  PetscCheck(st->ops->apply,PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST does not have apply");
  CHKERRQ(STGetOperator_Private(st,&Op));
  CHKERRQ(MatMult(Op,x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode STApplyMat_Generic(ST st,Mat B,Mat C)
{
  Mat            work;

  PetscFunctionBegin;
  if (st->M && st->P) {
    CHKERRQ(MatMatMult(st->M,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&work));
    CHKERRQ(STMatMatSolve(st,work,C));
    CHKERRQ(MatDestroy(&work));
  } else if (st->M) CHKERRQ(MatMatMult(st->M,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  else CHKERRQ(STMatMatSolve(st,B,C));
  PetscFunctionReturn(0);
}

/*@
   STApplyMat - Applies the spectral transformation operator to a matrix, for
   instance (A - sB)^-1 B in the case of the shift-and-invert transformation
   and generalized eigenproblem.

   Collective on st

   Input Parameters:
+  st - the spectral transformation context
-  X  - input matrix

   Output Parameter:
.  Y - output matrix

   Level: developer

.seealso: STApply()
@*/
PetscErrorCode STApplyMat(ST st,Mat X,Mat Y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(X,MAT_CLASSID,2);
  PetscValidHeaderSpecific(Y,MAT_CLASSID,3);
  PetscValidType(st,1);
  STCheckMatrices(st,1);
  PetscCheck(X!=Y,PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"X and Y must be different matrices");
  PetscCheck(st->ops->applymat,PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST does not have applymat");
  CHKERRQ((*st->ops->applymat)(st,X,Y));
  PetscFunctionReturn(0);
}

PetscErrorCode STApplyTranspose_Generic(ST st,Vec x,Vec y)
{
  PetscFunctionBegin;
  if (st->M && st->P) {
    CHKERRQ(STMatSolveTranspose(st,x,st->work[0]));
    CHKERRQ(MatMultTranspose(st->M,st->work[0],y));
  } else if (st->M) CHKERRQ(MatMultTranspose(st->M,x,y));
  else CHKERRQ(STMatSolveTranspose(st,x,y));
  PetscFunctionReturn(0);
}

/*@
   STApplyTranspose - Applies the transpose of the operator to a vector, for
   instance B^T(A - sB)^-T in the case of the shift-and-invert transformation
   and generalized eigenproblem.

   Collective on st

   Input Parameters:
+  st - the spectral transformation context
-  x  - input vector

   Output Parameter:
.  y - output vector

   Level: developer

.seealso: STApply(), STApplyHermitianTranspose()
@*/
PetscErrorCode STApplyTranspose(ST st,Vec x,Vec y)
{
  Mat            Op;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidType(st,1);
  STCheckMatrices(st,1);
  PetscCheck(x!=y,PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  CHKERRQ(VecSetErrorIfLocked(y,3));
  PetscCheck(st->ops->applytrans,PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST does not have applytrans");
  CHKERRQ(STGetOperator_Private(st,&Op));
  CHKERRQ(MatMultTranspose(Op,x,y));
  PetscFunctionReturn(0);
}

/*@
   STApplyHermitianTranspose - Applies the hermitian-transpose of the operator
   to a vector, for instance B^H(A - sB)^-H in the case of the shift-and-invert
   transformation and generalized eigenproblem.

   Collective on st

   Input Parameters:
+  st - the spectral transformation context
-  x  - input vector

   Output Parameter:
.  y - output vector

   Note:
   Currently implemented via STApplyTranspose() with appropriate conjugation.

   Level: developer

.seealso: STApply(), STApplyTranspose()
@*/
PetscErrorCode STApplyHermitianTranspose(ST st,Vec x,Vec y)
{
  Mat            Op;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidType(st,1);
  STCheckMatrices(st,1);
  PetscCheck(x!=y,PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  CHKERRQ(VecSetErrorIfLocked(y,3));
  PetscCheck(st->ops->applytrans,PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST does not have applytrans");
  CHKERRQ(STGetOperator_Private(st,&Op));
  CHKERRQ(MatMultHermitianTranspose(Op,x,y));
  PetscFunctionReturn(0);
}

/*@
   STGetBilinearForm - Returns the matrix used in the bilinear form with a
   generalized problem with semi-definite B.

   Not collective, though a parallel Mat may be returned

   Input Parameters:
.  st - the spectral transformation context

   Output Parameter:
.  B - output matrix

   Notes:
   The output matrix B must be destroyed after use. It will be NULL in
   case of standard eigenproblems.

   Level: developer

.seealso: BVSetMatrix()
@*/
PetscErrorCode STGetBilinearForm(ST st,Mat *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidType(st,1);
  PetscValidPointer(B,2);
  STCheckMatrices(st,1);
  CHKERRQ((*st->ops->getbilinearform)(st,B));
  PetscFunctionReturn(0);
}

PetscErrorCode STGetBilinearForm_Default(ST st,Mat *B)
{
  PetscFunctionBegin;
  if (st->nmat==1) *B = NULL;
  else {
    *B = st->A[1];
    CHKERRQ(PetscObjectReference((PetscObject)*B));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_STOperator(Mat Op,Vec x,Vec y)
{
  ST             st;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Op,&st));
  CHKERRQ(STSetUp(st));
  CHKERRQ(PetscLogEventBegin(ST_Apply,st,x,y,0));
  if (st->D) { /* with balancing */
    CHKERRQ(VecPointwiseDivide(st->wb,x,st->D));
    CHKERRQ((*st->ops->apply)(st,st->wb,y));
    CHKERRQ(VecPointwiseMult(y,y,st->D));
  } else CHKERRQ((*st->ops->apply)(st,x,y));
  CHKERRQ(PetscLogEventEnd(ST_Apply,st,x,y,0));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_STOperator(Mat Op,Vec x,Vec y)
{
  ST             st;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Op,&st));
  CHKERRQ(STSetUp(st));
  CHKERRQ(PetscLogEventBegin(ST_ApplyTranspose,st,x,y,0));
  if (st->D) { /* with balancing */
    CHKERRQ(VecPointwiseMult(st->wb,x,st->D));
    CHKERRQ((*st->ops->applytrans)(st,st->wb,y));
    CHKERRQ(VecPointwiseDivide(y,y,st->D));
  } else CHKERRQ((*st->ops->applytrans)(st,x,y));
  CHKERRQ(PetscLogEventEnd(ST_ApplyTranspose,st,x,y,0));
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_COMPLEX)
static PetscErrorCode MatMultHermitianTranspose_STOperator(Mat Op,Vec x,Vec y)
{
  ST             st;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Op,&st));
  CHKERRQ(STSetUp(st));
  CHKERRQ(PetscLogEventBegin(ST_ApplyTranspose,st,x,y,0));
  if (!st->wht) {
    CHKERRQ(MatCreateVecs(st->A[0],&st->wht,NULL));
    CHKERRQ(PetscLogObjectParent((PetscObject)st,(PetscObject)st->wht));
  }
  CHKERRQ(VecCopy(x,st->wht));
  CHKERRQ(VecConjugate(st->wht));
  if (st->D) { /* with balancing */
    CHKERRQ(VecPointwiseMult(st->wb,st->wht,st->D));
    CHKERRQ((*st->ops->applytrans)(st,st->wb,y));
    CHKERRQ(VecPointwiseDivide(y,y,st->D));
  } else CHKERRQ((*st->ops->applytrans)(st,st->wht,y));
  CHKERRQ(VecConjugate(y));
  CHKERRQ(PetscLogEventEnd(ST_ApplyTranspose,st,x,y,0));
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatMatMult_STOperator(Mat Op,Mat B,Mat C,void *ctx)
{
  ST             st;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Op,&st));
  CHKERRQ(STSetUp(st));
  CHKERRQ(PetscLogEventBegin(ST_Apply,st,B,C,0));
  CHKERRQ(STApplyMat_Generic(st,B,C));
  CHKERRQ(PetscLogEventEnd(ST_Apply,st,B,C,0));
  PetscFunctionReturn(0);
}

PetscErrorCode STGetOperator_Private(ST st,Mat *Op)
{
  PetscInt       m,n,M,N;
  Vec            v;
  VecType        vtype;

  PetscFunctionBegin;
  if (!st->Op) {
    if (Op) *Op = NULL;
    /* create the shell matrix */
    CHKERRQ(MatGetLocalSize(st->A[0],&m,&n));
    CHKERRQ(MatGetSize(st->A[0],&M,&N));
    CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)st),m,n,M,N,st,&st->Op));
    CHKERRQ(MatShellSetOperation(st->Op,MATOP_MULT,(void(*)(void))MatMult_STOperator));
    CHKERRQ(MatShellSetOperation(st->Op,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_STOperator));
#if defined(PETSC_USE_COMPLEX)
    CHKERRQ(MatShellSetOperation(st->Op,MATOP_MULT_HERMITIAN_TRANSPOSE,(void(*)(void))MatMultHermitianTranspose_STOperator));
#else
    CHKERRQ(MatShellSetOperation(st->Op,MATOP_MULT_HERMITIAN_TRANSPOSE,(void(*)(void))MatMultTranspose_STOperator));
#endif
    if (!st->D && st->ops->apply==STApply_Generic) {
      CHKERRQ(MatShellSetMatProductOperation(st->Op,MATPRODUCT_AB,NULL,MatMatMult_STOperator,NULL,MATDENSE,MATDENSE));
      CHKERRQ(MatShellSetMatProductOperation(st->Op,MATPRODUCT_AB,NULL,MatMatMult_STOperator,NULL,MATDENSECUDA,MATDENSECUDA));
    }
    /* make sure the shell matrix generates a vector of the same type as the problem matrices */
    CHKERRQ(MatCreateVecs(st->A[0],&v,NULL));
    CHKERRQ(VecGetType(v,&vtype));
    CHKERRQ(MatShellSetVecType(st->Op,vtype));
    CHKERRQ(VecDestroy(&v));
    /* build the operator matrices */
    CHKERRQ(STComputeOperator(st));
  }
  if (Op) *Op = st->Op;
  PetscFunctionReturn(0);
}

/*@
   STGetOperator - Returns a shell matrix that represents the operator of the
   spectral transformation.

   Collective on st

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  Op - operator matrix

   Notes:
   The operator is defined in linear eigenproblems only, not in polynomial ones,
   so the call will fail if more than 2 matrices were passed in STSetMatrices().

   The returned shell matrix is essentially a wrapper to the STApply() and
   STApplyTranspose() operations. The operator can often be expressed as

$     Op = D*inv(K)*M*inv(D)

   where D is the balancing matrix, and M and K are two matrices corresponding
   to the numerator and denominator for spectral transformations that represent
   a rational matrix function. In the case of STSHELL, the inner part inv(K)*M
   is replaced by the user-provided operation from STShellSetApply().

   The preconditioner matrix K typically depends on the value of the shift, and
   its inverse is handled via an internal KSP object. Normal usage does not
   require explicitly calling STGetOperator(), but it can be used to force the
   creation of K and M, and then K is passed to the KSP. This is useful for
   setting options associated with the PCFactor (to set MUMPS options, for instance).

   The returned matrix must NOT be destroyed by the user. Instead, when no
   longer needed it must be returned with STRestoreOperator(). In particular,
   this is required before modifying the ST matrices or the shift.

   A NULL pointer can be passed in Op in case the matrix is not required but we
   want to force its creation. In this case, STRestoreOperator() should not be
   called.

   Level: advanced

.seealso: STApply(), STApplyTranspose(), STSetBalanceMatrix(), STShellSetApply(),
          STGetKSP(), STSetShift(), STRestoreOperator(), STSetMatrices()
@*/
PetscErrorCode STGetOperator(ST st,Mat *Op)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidType(st,1);
  STCheckMatrices(st,1);
  STCheckNotSeized(st,1);
  PetscCheck(st->nmat<=2,PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_WRONGSTATE,"The operator is not defined in polynomial eigenproblems");
  CHKERRQ(STGetOperator_Private(st,Op));
  if (Op) st->opseized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   STRestoreOperator - Restore the previously seized operator matrix.

   Collective on st

   Input Parameters:
+  st - the spectral transformation context
-  Op - operator matrix

   Notes:
   The arguments must match the corresponding call to STGetOperator().

   Level: advanced

.seealso: STGetOperator()
@*/
PetscErrorCode STRestoreOperator(ST st,Mat *Op)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(Op,2);
  PetscValidHeaderSpecific(*Op,MAT_CLASSID,2);
  PetscCheck(st->opseized,PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_WRONGSTATE,"Must be called after STGetOperator()");
  *Op = NULL;
  st->opseized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*
   STComputeOperator - Computes the matrices that constitute the operator

      Op = D*inv(K)*M*inv(D).

   K and M are computed here (D is user-provided) from the system matrices
   and the shift sigma (whenever these are changed, this function recomputes
   K and M). This is used only in linear eigenproblems (nmat<3).

   K is the "preconditioner matrix": it is the denominator in rational operators,
   e.g. (A-sigma*B) in shift-and-invert. In non-rational transformations such
   as STFILTER, K=NULL which means identity. After computing K, it is passed to
   the internal KSP object via KSPSetOperators.

   M is the numerator in rational operators. If unused it is set to NULL (e.g.
   in STPRECOND).

   STSHELL does not compute anything here, but sets the flag as if it was ready.
*/
PetscErrorCode STComputeOperator(ST st)
{
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidType(st,1);
  if (!st->opready && st->ops->computeoperator) {
    CHKERRQ(PetscInfo(st,"Building the operator matrices\n"));
    STCheckMatrices(st,1);
    if (!st->T) {
      CHKERRQ(PetscCalloc1(PetscMax(2,st->nmat),&st->T));
      CHKERRQ(PetscLogObjectMemory((PetscObject)st,PetscMax(2,st->nmat)*sizeof(Mat)));
    }
    CHKERRQ(PetscLogEventBegin(ST_ComputeOperator,st,0,0,0));
    CHKERRQ((*st->ops->computeoperator)(st));
    CHKERRQ(PetscLogEventEnd(ST_ComputeOperator,st,0,0,0));
    if (st->usesksp) {
      if (!st->ksp) CHKERRQ(STGetKSP(st,&st->ksp));
      if (st->P) {
        CHKERRQ(STSetDefaultKSP(st));
        CHKERRQ(ST_KSPSetOperators(st,st->P,st->Pmat?st->Pmat:st->P));
      } else {
        /* STPRECOND defaults to PCNONE if st->P is empty */
        CHKERRQ(KSPGetPC(st->ksp,&pc));
        CHKERRQ(PCSetType(pc,PCNONE));
      }
    }
  }
  st->opready = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   STSetUp - Prepares for the use of a spectral transformation.

   Collective on st

   Input Parameter:
.  st - the spectral transformation context

   Level: advanced

.seealso: STCreate(), STApply(), STDestroy()
@*/
PetscErrorCode STSetUp(ST st)
{
  PetscInt       i,n,k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidType(st,1);
  STCheckMatrices(st,1);
  switch (st->state) {
    case ST_STATE_INITIAL:
      CHKERRQ(PetscInfo(st,"Setting up new ST\n"));
      if (!((PetscObject)st)->type_name) CHKERRQ(STSetType(st,STSHIFT));
      break;
    case ST_STATE_SETUP:
      PetscFunctionReturn(0);
    case ST_STATE_UPDATED:
      CHKERRQ(PetscInfo(st,"Setting up updated ST\n"));
      break;
  }
  CHKERRQ(PetscLogEventBegin(ST_SetUp,st,0,0,0));
  if (st->state!=ST_STATE_UPDATED) {
    if (!(st->nmat<3 && st->opready)) {
      if (st->T) {
        for (i=0;i<PetscMax(2,st->nmat);i++) CHKERRQ(MatDestroy(&st->T[i]));
      }
      CHKERRQ(MatDestroy(&st->P));
    }
  }
  if (st->D) {
    CHKERRQ(MatGetLocalSize(st->A[0],NULL,&n));
    CHKERRQ(VecGetLocalSize(st->D,&k));
    PetscCheck(n==k,PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_SIZ,"Balance matrix has wrong dimension %" PetscInt_FMT " (should be %" PetscInt_FMT ")",k,n);
    if (!st->wb) {
      CHKERRQ(VecDuplicate(st->D,&st->wb));
      CHKERRQ(PetscLogObjectParent((PetscObject)st,(PetscObject)st->wb));
    }
  }
  if (st->nmat<3 && st->transform) CHKERRQ(STComputeOperator(st));
  else {
    if (!st->T) {
      CHKERRQ(PetscCalloc1(PetscMax(2,st->nmat),&st->T));
      CHKERRQ(PetscLogObjectMemory((PetscObject)st,PetscMax(2,st->nmat)*sizeof(Mat)));
    }
  }
  if (st->ops->setup) CHKERRQ((*st->ops->setup)(st));
  st->state = ST_STATE_SETUP;
  CHKERRQ(PetscLogEventEnd(ST_SetUp,st,0,0,0));
  PetscFunctionReturn(0);
}

/*
   Computes coefficients for the transformed polynomial,
   and stores the result in argument S.

   alpha - value of the parameter of the transformed polynomial
   beta - value of the previous shift (only used in inplace mode)
   k - index of first matrix included in the computation
   coeffs - coefficients of the expansion
   initial - true if this is the first time
   precond - whether the preconditioner matrix must be computed
*/
PetscErrorCode STMatMAXPY_Private(ST st,PetscScalar alpha,PetscScalar beta,PetscInt k,PetscScalar *coeffs,PetscBool initial,PetscBool precond,Mat *S)
{
  PetscInt       *matIdx=NULL,nmat,i,ini=-1;
  PetscScalar    t=1.0,ta,gamma;
  PetscBool      nz=PETSC_FALSE;
  Mat            *A=precond?st->Psplit:st->A;
  MatStructure   str=precond?st->strp:st->str;

  PetscFunctionBegin;
  nmat = st->nmat-k;
  switch (st->matmode) {
  case ST_MATMODE_INPLACE:
    PetscCheck(st->nmat<=2,PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST_MATMODE_INPLACE not supported for polynomial eigenproblems");
    PetscCheck(!precond,PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST_MATMODE_INPLACE not supported for split preconditioner");
    if (initial) {
      CHKERRQ(PetscObjectReference((PetscObject)A[0]));
      *S = A[0];
      gamma = alpha;
    } else gamma = alpha-beta;
    if (gamma != 0.0) {
      if (st->nmat>1) CHKERRQ(MatAXPY(*S,gamma,A[1],str));
      else CHKERRQ(MatShift(*S,gamma));
    }
    break;
  case ST_MATMODE_SHELL:
    PetscCheck(!precond,PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST_MATMODE_SHELL not supported for split preconditioner");
    if (initial) {
      if (st->nmat>2) {
        CHKERRQ(PetscMalloc1(nmat,&matIdx));
        for (i=0;i<nmat;i++) matIdx[i] = k+i;
      }
      CHKERRQ(STMatShellCreate(st,alpha,nmat,matIdx,coeffs,S));
      CHKERRQ(PetscLogObjectParent((PetscObject)st,(PetscObject)*S));
      if (st->nmat>2) CHKERRQ(PetscFree(matIdx));
    } else CHKERRQ(STMatShellShift(*S,alpha));
    break;
  case ST_MATMODE_COPY:
    if (coeffs) {
      for (i=0;i<nmat && ini==-1;i++) {
        if (coeffs[i]!=0.0) ini = i;
        else t *= alpha;
      }
      if (coeffs[ini] != 1.0) nz = PETSC_TRUE;
      for (i=ini+1;i<nmat&&!nz;i++) if (coeffs[i]!=0.0) nz = PETSC_TRUE;
    } else { nz = PETSC_TRUE; ini = 0; }
    if ((alpha == 0.0 || !nz) && t==1.0) {
      CHKERRQ(PetscObjectReference((PetscObject)A[k+ini]));
      CHKERRQ(MatDestroy(S));
      *S = A[k+ini];
    } else {
      if (*S && *S!=A[k+ini]) {
        CHKERRQ(MatSetOption(*S,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
        CHKERRQ(MatCopy(A[k+ini],*S,DIFFERENT_NONZERO_PATTERN));
      } else {
        CHKERRQ(MatDestroy(S));
        CHKERRQ(MatDuplicate(A[k+ini],MAT_COPY_VALUES,S));
        CHKERRQ(MatSetOption(*S,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
        CHKERRQ(PetscLogObjectParent((PetscObject)st,(PetscObject)*S));
      }
      if (coeffs && coeffs[ini]!=1.0) CHKERRQ(MatScale(*S,coeffs[ini]));
      for (i=ini+k+1;i<PetscMax(2,st->nmat);i++) {
        t *= alpha;
        ta = t;
        if (coeffs) ta *= coeffs[i-k];
        if (ta!=0.0) {
          if (st->nmat>1) CHKERRQ(MatAXPY(*S,ta,A[i],str));
          else CHKERRQ(MatShift(*S,ta));
        }
      }
    }
  }
  CHKERRQ(MatSetOption(*S,MAT_SYMMETRIC,st->asymm));
  CHKERRQ(MatSetOption(*S,MAT_HERMITIAN,(PetscImaginaryPart(st->sigma)==0.0)?st->aherm:PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*
   Computes the values of the coefficients required by STMatMAXPY_Private
   for the case of monomial basis.
*/
PetscErrorCode STCoeffs_Monomial(ST st, PetscScalar *coeffs)
{
  PetscInt  k,i,ini,inip;

  PetscFunctionBegin;
  /* Compute binomial coefficients */
  ini = (st->nmat*(st->nmat-1))/2;
  for (i=0;i<st->nmat;i++) coeffs[ini+i]=1.0;
  for (k=st->nmat-1;k>=1;k--) {
    inip = ini+1;
    ini = (k*(k-1))/2;
    coeffs[ini] = 1.0;
    for (i=1;i<k;i++) coeffs[ini+i] = coeffs[ini+i-1]+coeffs[inip+i-1];
  }
  PetscFunctionReturn(0);
}

/*@
   STPostSolve - Optional post-solve phase, intended for any actions that must
   be performed on the ST object after the eigensolver has finished.

   Collective on st

   Input Parameters:
.  st  - the spectral transformation context

   Level: developer

.seealso: EPSSolve()
@*/
PetscErrorCode STPostSolve(ST st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidType(st,1);
  if (st->ops->postsolve) CHKERRQ((*st->ops->postsolve)(st));
  PetscFunctionReturn(0);
}

/*@
   STBackTransform - Back-transformation phase, intended for
   spectral transformations which require to transform the computed
   eigenvalues back to the original eigenvalue problem.

   Not Collective

   Input Parameters:
+  st   - the spectral transformation context
.  n    - number of eigenvalues
.  eigr - real part of a computed eigenvalues
-  eigi - imaginary part of a computed eigenvalues

   Level: developer

.seealso: STIsInjective()
@*/
PetscErrorCode STBackTransform(ST st,PetscInt n,PetscScalar* eigr,PetscScalar* eigi)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidType(st,1);
  if (st->ops->backtransform) CHKERRQ((*st->ops->backtransform)(st,n,eigr,eigi));
  PetscFunctionReturn(0);
}

/*@
   STIsInjective - Ask if this spectral transformation is injective or not
   (that is, if it corresponds to a one-to-one mapping). If not, then it
   does not make sense to call STBackTransform().

   Not collective

   Input Parameter:
.  st   - the spectral transformation context

   Output Parameter:
.  is - the answer

   Level: developer

.seealso: STBackTransform()
@*/
PetscErrorCode STIsInjective(ST st,PetscBool* is)
{
  PetscBool      shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidType(st,1);
  PetscValidBoolPointer(is,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)st,STSHELL,&shell));
  if (shell) CHKERRQ(STIsInjective_Shell(st,is));
  else *is = st->ops->backtransform? PETSC_TRUE: PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   STMatSetUp - Build the preconditioner matrix used in STMatSolve().

   Collective on st

   Input Parameters:
+  st     - the spectral transformation context
.  sigma  - the shift
-  coeffs - the coefficients (may be NULL)

   Note:
   This function is not intended to be called by end users, but by SLEPc
   solvers that use ST. It builds matrix st->P as follows, then calls KSPSetUp().
.vb
    If (coeffs)  st->P = Sum_{i=0..nmat-1} coeffs[i]*sigma^i*A_i
    else         st->P = Sum_{i=0..nmat-1} sigma^i*A_i
.ve

   Level: developer

.seealso: STMatSolve()
@*/
PetscErrorCode STMatSetUp(ST st,PetscScalar sigma,PetscScalar *coeffs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveScalar(st,sigma,2);
  STCheckMatrices(st,1);

  CHKERRQ(PetscLogEventBegin(ST_MatSetUp,st,0,0,0));
  CHKERRQ(STMatMAXPY_Private(st,sigma,0.0,0,coeffs,PETSC_TRUE,PETSC_FALSE,&st->P));
  if (st->Psplit) CHKERRQ(STMatMAXPY_Private(st,sigma,0.0,0,coeffs,PETSC_TRUE,PETSC_TRUE,&st->Pmat));
  CHKERRQ(ST_KSPSetOperators(st,st->P,st->Pmat?st->Pmat:st->P));
  CHKERRQ(KSPSetUp(st->ksp));
  CHKERRQ(PetscLogEventEnd(ST_MatSetUp,st,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   STSetWorkVecs - Sets a number of work vectors into the ST object.

   Collective on st

   Input Parameters:
+  st - the spectral transformation context
-  nw - number of work vectors to allocate

   Developer Notes:
   This is SLEPC_EXTERN because it may be required by shell STs.

   Level: developer

.seealso: STMatCreateVecs()
@*/
PetscErrorCode STSetWorkVecs(ST st,PetscInt nw)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveInt(st,nw,2);
  PetscCheck(nw>0,PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_OUTOFRANGE,"nw must be > 0: nw = %" PetscInt_FMT,nw);
  if (st->nwork < nw) {
    CHKERRQ(VecDestroyVecs(st->nwork,&st->work));
    st->nwork = nw;
    CHKERRQ(PetscMalloc1(nw,&st->work));
    for (i=0;i<nw;i++) CHKERRQ(STMatCreateVecs(st,&st->work[i],NULL));
    CHKERRQ(PetscLogObjectParents(st,nw,st->work));
  }
  PetscFunctionReturn(0);
}
