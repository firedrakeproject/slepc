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
  PetscErrorCode    ierr;
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *d_px;
  PetscScalar       *d_py;
  PetscInt          m;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->A,&m,NULL);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(x,&d_px);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayWrite(y,&d_py);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(ctx->x1,d_px);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(ctx->x2,d_px+m);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(ctx->y1,d_py);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(ctx->y2,d_py+m);CHKERRQ(ierr);
  ierr = MatMult(ctx->A,ctx->x2,ctx->y1);CHKERRQ(ierr);
  ierr = MatMult(ctx->AT,ctx->x1,ctx->y2);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(ctx->y2);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(x,&d_px);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(y,&d_py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_ECross_CUDA(Mat B,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *d_px;
  PetscScalar       *d_py;
  PetscInt          mn,m,n;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->A,NULL,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&mn);CHKERRQ(ierr);
  m = mn-n;
  ierr = VecCUDAGetArrayRead(x,&d_px);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayWrite(y,&d_py);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(ctx->x1,d_px);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(ctx->x2,d_px+m);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(ctx->y1,d_py);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(ctx->y2,d_py+m);CHKERRQ(ierr);
  ierr = VecCopy(ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = MatMult(ctx->A,ctx->x2,ctx->w);CHKERRQ(ierr);
  ierr = MatMult(ctx->AT,ctx->w,ctx->y2);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(ctx->y2);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(x,&d_px);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(y,&d_py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

