/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Subroutines that implement various operations of the matrix associated with
   the shift-and-invert technique for eigenvalue problems
*/

#include <slepc/private/stimpl.h>

typedef struct {
  PetscScalar alpha;
  PetscScalar *coeffs;
  ST          st;
  Vec         z;
  PetscInt    nmat;
  PetscInt    *matIdx;
} ST_MATSHELL;

PetscErrorCode STMatShellShift(Mat A,PetscScalar alpha)
{
  ST_MATSHELL    *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  ctx->alpha = alpha;
  CHKERRQ(PetscObjectStateIncrease((PetscObject)A));
  PetscFunctionReturn(0);
}

/*
  For i=0:nmat-1 computes y = (sum_i (coeffs[i]*alpha^i*st->A[idx[i]]))x
  If null coeffs computes with coeffs[i]=1.0
*/
static PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec y)
{
  ST_MATSHELL    *ctx;
  ST             st;
  PetscInt       i;
  PetscScalar    t=1.0,c;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  st = ctx->st;
  CHKERRQ(MatMult(st->A[ctx->matIdx[0]],x,y));
  if (ctx->coeffs && ctx->coeffs[0]!=1.0) CHKERRQ(VecScale(y,ctx->coeffs[0]));
  if (ctx->alpha!=0.0) {
    for (i=1;i<ctx->nmat;i++) {
      CHKERRQ(MatMult(st->A[ctx->matIdx[i]],x,ctx->z));
      t *= ctx->alpha;
      c = (ctx->coeffs)?t*ctx->coeffs[i]:t;
      CHKERRQ(VecAXPY(y,c,ctx->z));
    }
    if (ctx->nmat==1) CHKERRQ(VecAXPY(y,ctx->alpha,x)); /* y = (A + alpha*I) x */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Shell(Mat A,Vec x,Vec y)
{
  ST_MATSHELL    *ctx;
  ST             st;
  PetscInt       i;
  PetscScalar    t=1.0,c;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  st = ctx->st;
  CHKERRQ(MatMultTranspose(st->A[ctx->matIdx[0]],x,y));
  if (ctx->coeffs && ctx->coeffs[0]!=1.0) CHKERRQ(VecScale(y,ctx->coeffs[0]));
  if (ctx->alpha!=0.0) {
    for (i=1;i<ctx->nmat;i++) {
      CHKERRQ(MatMultTranspose(st->A[ctx->matIdx[i]],x,ctx->z));
      t *= ctx->alpha;
      c = (ctx->coeffs)?t*ctx->coeffs[i]:t;
      CHKERRQ(VecAXPY(y,c,ctx->z));
    }
    if (ctx->nmat==1) CHKERRQ(VecAXPY(y,ctx->alpha,x)); /* y = (A + alpha*I) x */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_Shell(Mat A,Vec diag)
{
  ST_MATSHELL    *ctx;
  ST             st;
  Vec            diagb;
  PetscInt       i;
  PetscScalar    t=1.0,c;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  st = ctx->st;
  CHKERRQ(MatGetDiagonal(st->A[ctx->matIdx[0]],diag));
  if (ctx->coeffs && ctx->coeffs[0]!=1.0) CHKERRQ(VecScale(diag,ctx->coeffs[0]));
  if (ctx->alpha!=0.0) {
    if (ctx->nmat==1) CHKERRQ(VecShift(diag,ctx->alpha)); /* y = (A + alpha*I) x */
    else {
      CHKERRQ(VecDuplicate(diag,&diagb));
      for (i=1;i<ctx->nmat;i++) {
        CHKERRQ(MatGetDiagonal(st->A[ctx->matIdx[i]],diagb));
        t *= ctx->alpha;
        c = (ctx->coeffs)?t*ctx->coeffs[i]:t;
        CHKERRQ(VecAYPX(diag,c,diagb));
      }
      CHKERRQ(VecDestroy(&diagb));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Shell(Mat A)
{
  ST_MATSHELL    *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(VecDestroy(&ctx->z));
  CHKERRQ(PetscFree(ctx->matIdx));
  CHKERRQ(PetscFree(ctx->coeffs));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode STMatShellCreate(ST st,PetscScalar alpha,PetscInt nmat,PetscInt *matIdx,PetscScalar *coeffs,Mat *mat)
{
  PetscInt       n,m,N,M,i;
  PetscBool      has=PETSC_FALSE,hasA,hasB;
  ST_MATSHELL    *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(st->A[0],&M,&N));
  CHKERRQ(MatGetLocalSize(st->A[0],&m,&n));
  CHKERRQ(PetscNew(&ctx));
  ctx->st = st;
  ctx->alpha = alpha;
  ctx->nmat = matIdx?nmat:st->nmat;
  CHKERRQ(PetscMalloc1(ctx->nmat,&ctx->matIdx));
  if (matIdx) {
    for (i=0;i<ctx->nmat;i++) ctx->matIdx[i] = matIdx[i];
  } else {
    ctx->matIdx[0] = 0;
    if (ctx->nmat>1) ctx->matIdx[1] = 1;
  }
  if (coeffs) {
    CHKERRQ(PetscMalloc1(ctx->nmat,&ctx->coeffs));
    for (i=0;i<ctx->nmat;i++) ctx->coeffs[i] = coeffs[i];
  }
  CHKERRQ(MatCreateVecs(st->A[0],&ctx->z,NULL));
  CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)st),m,n,M,N,(void*)ctx,mat));
  CHKERRQ(MatShellSetOperation(*mat,MATOP_MULT,(void(*)(void))MatMult_Shell));
  CHKERRQ(MatShellSetOperation(*mat,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Shell));
  CHKERRQ(MatShellSetOperation(*mat,MATOP_DESTROY,(void(*)(void))MatDestroy_Shell));

  CHKERRQ(MatHasOperation(st->A[ctx->matIdx[0]],MATOP_GET_DIAGONAL,&hasA));
  if (st->nmat>1) {
    has = hasA;
    for (i=1;i<ctx->nmat;i++) {
      CHKERRQ(MatHasOperation(st->A[ctx->matIdx[i]],MATOP_GET_DIAGONAL,&hasB));
      has = (has && hasB)? PETSC_TRUE: PETSC_FALSE;
    }
  }
  if ((hasA && st->nmat==1) || has) CHKERRQ(MatShellSetOperation(*mat,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Shell));
  PetscFunctionReturn(0);
}
