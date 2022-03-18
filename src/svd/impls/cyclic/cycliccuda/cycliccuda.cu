/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc singular value solver: "cyclic" (CUDA implementation)
*/
#include <slepc/private/svdimpl.h>
#include "../src/svd/impls/cyclic/cyclic.h"

PetscErrorCode MatMult_Cyclic_CUDA(Mat B,Vec x,Vec y)
{
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *d_px;
  PetscScalar       *d_py;
  PetscInt          m;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&ctx));
  CHKERRQ(MatGetLocalSize(ctx->A,&m,NULL));
  CHKERRQ(VecCUDAGetArrayRead(x,&d_px));
  CHKERRQ(VecCUDAGetArrayWrite(y,&d_py));
  CHKERRQ(VecCUDAPlaceArray(ctx->x1,d_px));
  CHKERRQ(VecCUDAPlaceArray(ctx->x2,d_px+m));
  CHKERRQ(VecCUDAPlaceArray(ctx->y1,d_py));
  CHKERRQ(VecCUDAPlaceArray(ctx->y2,d_py+m));
  CHKERRQ(MatMult(ctx->A,ctx->x2,ctx->y1));
  CHKERRQ(MatMult(ctx->AT,ctx->x1,ctx->y2));
  CHKERRQ(VecCUDAResetArray(ctx->x1));
  CHKERRQ(VecCUDAResetArray(ctx->x2));
  CHKERRQ(VecCUDAResetArray(ctx->y1));
  CHKERRQ(VecCUDAResetArray(ctx->y2));
  CHKERRQ(VecCUDARestoreArrayRead(x,&d_px));
  CHKERRQ(VecCUDARestoreArrayWrite(y,&d_py));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_ECross_CUDA(Mat B,Vec x,Vec y)
{
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *d_px;
  PetscScalar       *d_py;
  PetscInt          mn,m,n;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&ctx));
  CHKERRQ(MatGetLocalSize(ctx->A,NULL,&n));
  CHKERRQ(VecGetLocalSize(y,&mn));
  m = mn-n;
  CHKERRQ(VecCUDAGetArrayRead(x,&d_px));
  CHKERRQ(VecCUDAGetArrayWrite(y,&d_py));
  CHKERRQ(VecCUDAPlaceArray(ctx->x1,d_px));
  CHKERRQ(VecCUDAPlaceArray(ctx->x2,d_px+m));
  CHKERRQ(VecCUDAPlaceArray(ctx->y1,d_py));
  CHKERRQ(VecCUDAPlaceArray(ctx->y2,d_py+m));
  CHKERRQ(VecCopy(ctx->x1,ctx->y1));
  CHKERRQ(MatMult(ctx->A,ctx->x2,ctx->w));
  CHKERRQ(MatMult(ctx->AT,ctx->w,ctx->y2));
  CHKERRQ(VecCUDAResetArray(ctx->x1));
  CHKERRQ(VecCUDAResetArray(ctx->x2));
  CHKERRQ(VecCUDAResetArray(ctx->y1));
  CHKERRQ(VecCUDAResetArray(ctx->y2));
  CHKERRQ(VecCUDARestoreArrayRead(x,&d_px));
  CHKERRQ(VecCUDARestoreArrayWrite(y,&d_py));
  PetscFunctionReturn(0);
}
