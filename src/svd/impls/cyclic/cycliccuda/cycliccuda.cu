/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscCall(MatShellGetContext(B,&ctx));
  PetscCall(MatGetLocalSize(ctx->A,&m,NULL));
  PetscCall(VecCUDAGetArrayRead(x,&d_px));
  PetscCall(VecCUDAGetArrayWrite(y,&d_py));
  PetscCall(VecCUDAPlaceArray(ctx->x1,d_px));
  PetscCall(VecCUDAPlaceArray(ctx->x2,d_px+m));
  PetscCall(VecCUDAPlaceArray(ctx->y1,d_py));
  PetscCall(VecCUDAPlaceArray(ctx->y2,d_py+m));
  PetscCall(MatMult(ctx->A,ctx->x2,ctx->y1));
  PetscCall(MatMult(ctx->AT,ctx->x1,ctx->y2));
  PetscCall(VecCUDAResetArray(ctx->x1));
  PetscCall(VecCUDAResetArray(ctx->x2));
  PetscCall(VecCUDAResetArray(ctx->y1));
  PetscCall(VecCUDAResetArray(ctx->y2));
  PetscCall(VecCUDARestoreArrayRead(x,&d_px));
  PetscCall(VecCUDARestoreArrayWrite(y,&d_py));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_ECross_CUDA(Mat B,Vec x,Vec y)
{
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *d_px;
  PetscScalar       *d_py;
  PetscInt          mn,m,n;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(B,&ctx));
  PetscCall(MatGetLocalSize(ctx->A,NULL,&n));
  PetscCall(VecGetLocalSize(y,&mn));
  m = mn-n;
  PetscCall(VecCUDAGetArrayRead(x,&d_px));
  PetscCall(VecCUDAGetArrayWrite(y,&d_py));
  PetscCall(VecCUDAPlaceArray(ctx->x1,d_px));
  PetscCall(VecCUDAPlaceArray(ctx->x2,d_px+m));
  PetscCall(VecCUDAPlaceArray(ctx->y1,d_py));
  PetscCall(VecCUDAPlaceArray(ctx->y2,d_py+m));
  PetscCall(VecCopy(ctx->x1,ctx->y1));
  PetscCall(MatMult(ctx->A,ctx->x2,ctx->w));
  PetscCall(MatMult(ctx->AT,ctx->w,ctx->y2));
  PetscCall(VecCUDAResetArray(ctx->x1));
  PetscCall(VecCUDAResetArray(ctx->x2));
  PetscCall(VecCUDAResetArray(ctx->y1));
  PetscCall(VecCUDAResetArray(ctx->y2));
  PetscCall(VecCUDARestoreArrayRead(x,&d_px));
  PetscCall(VecCUDARestoreArrayWrite(y,&d_py));
  PetscFunctionReturn(0);
}
