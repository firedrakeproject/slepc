/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc singular value solver: "cyclic"

   Method: Uses a Hermitian eigensolver for H(A) = [ 0  A ; A^T 0 ]
*/

#include <slepc/private/svdimpl.h>                /*I "slepcsvd.h" I*/
#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/
#include "cyclic.h"

static PetscErrorCode MatMult_Cyclic(Mat B,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->A,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  ierr = MatMult(ctx->A,ctx->x2,ctx->y1);CHKERRQ(ierr);
  ierr = MatMult(ctx->AT,ctx->x1,ctx->y2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_Cyclic(Mat B,Vec diag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(diag,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Cyclic(Mat B)
{
  PetscErrorCode   ierr;
  SVD_CYCLIC_SHELL *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x2);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y2);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Builds cyclic matrix   C = | 0   A |
                              | AT  0 |
*/
static PetscErrorCode SVDCyclicGetCyclicMat(SVD svd,Mat A,Mat AT,Mat *C)
{
  PetscErrorCode   ierr;
  SVD_CYCLIC       *cyclic = (SVD_CYCLIC*)svd->data;
  SVD_CYCLIC_SHELL *ctx;
  PetscInt         i,M,N,m,n,Istart,Iend;
  VecType          vtype;
  Mat              Zm,Zn;
#if defined(PETSC_HAVE_CUDA)
  PetscBool        cuda;
#endif

  PetscFunctionBegin;
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);

  if (cyclic->explicitmatrix) {
    if (!svd->expltrans) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Cannot use explicit cyclic matrix with implicit transpose");
    ierr = MatCreate(PetscObjectComm((PetscObject)svd),&Zm);CHKERRQ(ierr);
    ierr = MatSetSizes(Zm,m,m,M,M);CHKERRQ(ierr);
    ierr = MatSetFromOptions(Zm);CHKERRQ(ierr);
    ierr = MatSetUp(Zm);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Zm,&Istart,&Iend);CHKERRQ(ierr);
    for (i=Istart;i<Iend;i++) {
      ierr = MatSetValue(Zm,i,i,0.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Zm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Zm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)svd),&Zn);CHKERRQ(ierr);
    ierr = MatSetSizes(Zn,n,n,N,N);CHKERRQ(ierr);
    ierr = MatSetFromOptions(Zn);CHKERRQ(ierr);
    ierr = MatSetUp(Zn);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Zn,&Istart,&Iend);CHKERRQ(ierr);
    for (i=Istart;i<Iend;i++) {
      ierr = MatSetValue(Zn,i,i,0.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Zn,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Zn,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatCreateTile(1.0,Zm,1.0,A,1.0,AT,1.0,Zn,C);CHKERRQ(ierr);
    ierr = MatDestroy(&Zm);CHKERRQ(ierr);
    ierr = MatDestroy(&Zn);CHKERRQ(ierr);
  } else {
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ctx->A       = A;
    ctx->AT      = AT;
    ctx->swapped = svd->swapped;
    ierr = MatCreateVecsEmpty(A,&ctx->x2,&ctx->x1);CHKERRQ(ierr);
    ierr = MatCreateVecsEmpty(A,&ctx->y2,&ctx->y1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->x1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->x2);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->y1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->y2);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)svd),m+n,m+n,M+N,M+N,ctx,C);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*C,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Cyclic);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*C,MATOP_DESTROY,(void(*)(void))MatDestroy_Cyclic);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    ierr = PetscObjectTypeCompareAny((PetscObject)(svd->swapped?AT:A),&cuda,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,"");CHKERRQ(ierr);
    if (cuda) {
      ierr = MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_Cyclic_CUDA);CHKERRQ(ierr);
    } else
#endif
    {
      ierr = MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_Cyclic);CHKERRQ(ierr);
    }
    ierr = MatGetVecType(A,&vtype);CHKERRQ(ierr);
    ierr = MatSetVecType(*C,vtype);CHKERRQ(ierr);
  }
  ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_ECross(Mat B,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          mn,m,n;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->A,NULL,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&mn);CHKERRQ(ierr);
  m = mn-n;
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  ierr = VecCopy(ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = MatMult(ctx->A,ctx->x2,ctx->w);CHKERRQ(ierr);
  ierr = MatMult(ctx->AT,ctx->w,ctx->y2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_ECross(Mat B,Vec d)
{
  PetscErrorCode    ierr;
  SVD_CYCLIC_SHELL  *ctx;
  PetscScalar       *pd;
  PetscMPIInt       len;
  PetscInt          mn,m,n,N,i,j,start,end,ncols;
  PetscScalar       *work1,*work2,*diag;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->A,NULL,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(d,&mn);CHKERRQ(ierr);
  m = mn-n;
  ierr = VecGetArrayWrite(d,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,pd);CHKERRQ(ierr);
  ierr = VecSet(ctx->y1,1.0);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,pd+m);CHKERRQ(ierr);
  if (!ctx->diag) {
    /* compute diagonal from rows and store in ctx->diag */
    ierr = VecDuplicate(ctx->y2,&ctx->diag);CHKERRQ(ierr);
    ierr = MatGetSize(ctx->A,NULL,&N);CHKERRQ(ierr);
    ierr = PetscCalloc2(N,&work1,N,&work2);CHKERRQ(ierr);
    if (ctx->swapped) {
      ierr = MatGetOwnershipRange(ctx->AT,&start,&end);CHKERRQ(ierr);
      for (i=start;i<end;i++) {
        ierr = MatGetRow(ctx->AT,i,&ncols,NULL,&vals);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) work1[i] += vals[j]*vals[j];
        ierr = MatRestoreRow(ctx->AT,i,&ncols,NULL,&vals);CHKERRQ(ierr);
      }
    } else {
      ierr = MatGetOwnershipRange(ctx->A,&start,&end);CHKERRQ(ierr);
      for (i=start;i<end;i++) {
        ierr = MatGetRow(ctx->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) work1[cols[j]] += vals[j]*vals[j];
        ierr = MatRestoreRow(ctx->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      }
    }
    ierr = PetscMPIIntCast(N,&len);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(work1,work2,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)B));CHKERRMPI(ierr);
    ierr = VecGetOwnershipRange(ctx->diag,&start,&end);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(ctx->diag,&diag);CHKERRQ(ierr);
    for (i=start;i<end;i++) diag[i-start] = work2[i];
    ierr = VecRestoreArrayWrite(ctx->diag,&diag);CHKERRQ(ierr);
    ierr = PetscFree2(work1,work2);CHKERRQ(ierr);
  }
  ierr = VecCopy(ctx->diag,ctx->y2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(d,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_ECross(Mat B)
{
  PetscErrorCode   ierr;
  SVD_CYCLIC_SHELL *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x2);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y2);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->diag);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Builds extended cross product matrix   C = | I_m   0  |
                                              |  0  AT*A |
   t is an auxiliary Vec used to take the dimensions of the upper block
*/
static PetscErrorCode SVDCyclicGetECrossMat(SVD svd,Mat A,Mat AT,Mat *C,Vec t)
{
  PetscErrorCode   ierr;
  SVD_CYCLIC       *cyclic = (SVD_CYCLIC*)svd->data;
  SVD_CYCLIC_SHELL *ctx;
  PetscInt         i,M,N,m,n,Istart,Iend;
  VecType          vtype;
  Mat              Id,Zm,Zn,ATA;
#if defined(PETSC_HAVE_CUDA)
  PetscBool        cuda;
#endif

  PetscFunctionBegin;
  ierr = MatGetSize(A,NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,NULL,&n);CHKERRQ(ierr);
  ierr = VecGetSize(t,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(t,&m);CHKERRQ(ierr);

  if (cyclic->explicitmatrix) {
    if (!svd->expltrans) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Cannot use explicit cyclic matrix with implicit transpose");
    ierr = MatCreate(PetscObjectComm((PetscObject)svd),&Id);CHKERRQ(ierr);
    ierr = MatSetSizes(Id,m,m,M,M);CHKERRQ(ierr);
    ierr = MatSetFromOptions(Id);CHKERRQ(ierr);
    ierr = MatSetUp(Id);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Id,&Istart,&Iend);CHKERRQ(ierr);
    for (i=Istart;i<Iend;i++) {
      ierr = MatSetValue(Id,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)svd),&Zm);CHKERRQ(ierr);
    ierr = MatSetSizes(Zm,m,n,M,N);CHKERRQ(ierr);
    ierr = MatSetFromOptions(Zm);CHKERRQ(ierr);
    ierr = MatSetUp(Zm);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Zm,&Istart,&Iend);CHKERRQ(ierr);
    for (i=Istart;i<Iend;i++) {
      if (i<N) { ierr = MatSetValue(Zm,i,i,0.0,INSERT_VALUES);CHKERRQ(ierr); }
    }
    ierr = MatAssemblyBegin(Zm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Zm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)svd),&Zn);CHKERRQ(ierr);
    ierr = MatSetSizes(Zn,n,m,N,M);CHKERRQ(ierr);
    ierr = MatSetFromOptions(Zn);CHKERRQ(ierr);
    ierr = MatSetUp(Zn);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Zn,&Istart,&Iend);CHKERRQ(ierr);
    for (i=Istart;i<Iend;i++) {
      if (i<m) { ierr = MatSetValue(Zn,i,i,0.0,INSERT_VALUES);CHKERRQ(ierr); }
    }
    ierr = MatAssemblyBegin(Zn,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Zn,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatProductCreate(AT,A,NULL,&ATA);CHKERRQ(ierr);
    ierr = MatProductSetType(ATA,MATPRODUCT_AB);CHKERRQ(ierr);
    ierr = MatProductSetFromOptions(ATA);CHKERRQ(ierr);
    ierr = MatProductSymbolic(ATA);CHKERRQ(ierr);
    ierr = MatProductNumeric(ATA);CHKERRQ(ierr);
    ierr = MatCreateTile(1.0,Id,1.0,Zm,1.0,Zn,1.0,ATA,C);CHKERRQ(ierr);
    ierr = MatDestroy(&Id);CHKERRQ(ierr);
    ierr = MatDestroy(&Zm);CHKERRQ(ierr);
    ierr = MatDestroy(&Zn);CHKERRQ(ierr);
    ierr = MatDestroy(&ATA);CHKERRQ(ierr);
  } else {
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ctx->A       = A;
    ctx->AT      = AT;
    ctx->swapped = svd->swapped;
    ierr = VecDuplicateEmpty(t,&ctx->x1);CHKERRQ(ierr);
    ierr = VecDuplicateEmpty(t,&ctx->y1);CHKERRQ(ierr);
    ierr = MatCreateVecsEmpty(A,&ctx->x2,NULL);CHKERRQ(ierr);
    ierr = MatCreateVecsEmpty(A,&ctx->y2,NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,NULL,&ctx->w);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->x1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->x2);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->y1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->y2);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)svd),m+n,m+n,M+N,M+N,ctx,C);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*C,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_ECross);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*C,MATOP_DESTROY,(void(*)(void))MatDestroy_ECross);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    ierr = PetscObjectTypeCompareAny((PetscObject)(svd->swapped?AT:A),&cuda,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,"");CHKERRQ(ierr);
    if (cuda) {
      ierr = MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_ECross_CUDA);CHKERRQ(ierr);
    } else
#endif
    {
      ierr = MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_ECross);CHKERRQ(ierr);
    }
    ierr = MatGetVecType(A,&vtype);CHKERRQ(ierr);
    ierr = MatSetVecType(*C,vtype);CHKERRQ(ierr);
  }
  ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetUp_Cyclic(SVD svd)
{
  PetscErrorCode    ierr;
  SVD_CYCLIC        *cyclic = (SVD_CYCLIC*)svd->data;
  PetscInt          M,N,m,n,i,isl;
  const PetscScalar *isa;
  PetscScalar       *va;
  PetscBool         trackall,issinv;
  Vec               v,t;
  ST                st;

  PetscFunctionBegin;
  ierr = MatGetSize(svd->A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->A,&m,&n);CHKERRQ(ierr);
  if (!cyclic->eps) { ierr = SVDCyclicGetEPS(svd,&cyclic->eps);CHKERRQ(ierr); }
  ierr = MatDestroy(&cyclic->C);CHKERRQ(ierr);
  ierr = MatDestroy(&cyclic->D);CHKERRQ(ierr);
  if (svd->isgeneralized) {
    if (svd->which==SVD_SMALLEST) {  /* alternative pencil */
      ierr = MatCreateVecs(svd->B,NULL,&t);CHKERRQ(ierr);
      ierr = SVDCyclicGetCyclicMat(svd,svd->B,svd->BT,&cyclic->C);CHKERRQ(ierr);
      ierr = SVDCyclicGetECrossMat(svd,svd->A,svd->AT,&cyclic->D,t);CHKERRQ(ierr);
    } else {
      ierr = MatCreateVecs(svd->A,NULL,&t);CHKERRQ(ierr);
      ierr = SVDCyclicGetCyclicMat(svd,svd->A,svd->AT,&cyclic->C);CHKERRQ(ierr);
      ierr = SVDCyclicGetECrossMat(svd,svd->B,svd->BT,&cyclic->D,t);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&t);CHKERRQ(ierr);
    ierr = EPSSetOperators(cyclic->eps,cyclic->C,cyclic->D);CHKERRQ(ierr);
    ierr = EPSSetProblemType(cyclic->eps,EPS_GHEP);CHKERRQ(ierr);
  } else {
    ierr = SVDCyclicGetCyclicMat(svd,svd->A,svd->AT,&cyclic->C);CHKERRQ(ierr);
    ierr = EPSSetOperators(cyclic->eps,cyclic->C,NULL);CHKERRQ(ierr);
    ierr = EPSSetProblemType(cyclic->eps,EPS_HEP);CHKERRQ(ierr);
  }
  if (!cyclic->usereps) {
    if (svd->which == SVD_LARGEST) {
      ierr = EPSGetST(cyclic->eps,&st);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)st,STSINVERT,&issinv);CHKERRQ(ierr);
      if (issinv) {
        ierr = EPSSetWhichEigenpairs(cyclic->eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
      } else {
        ierr = EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
      }
    } else {
      if (svd->isgeneralized) {  /* computes sigma^{-1} via alternative pencil */
        ierr = EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
      } else {
        ierr = EPSSetEigenvalueComparison(cyclic->eps,SlepcCompareSmallestPosReal,NULL);CHKERRQ(ierr);
        ierr = EPSSetTarget(cyclic->eps,0.0);CHKERRQ(ierr);
      }
    }
    ierr = EPSSetDimensions(cyclic->eps,svd->nsv,svd->ncv,svd->mpd);CHKERRQ(ierr);
    ierr = EPSSetTolerances(cyclic->eps,svd->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL/10.0:svd->tol,svd->max_it);CHKERRQ(ierr);
    switch (svd->conv) {
    case SVD_CONV_ABS:
      ierr = EPSSetConvergenceTest(cyclic->eps,EPS_CONV_ABS);CHKERRQ(ierr);break;
    case SVD_CONV_REL:
      ierr = EPSSetConvergenceTest(cyclic->eps,EPS_CONV_REL);CHKERRQ(ierr);break;
    case SVD_CONV_NORM:
      ierr = EPSSetConvergenceTest(cyclic->eps,EPS_CONV_NORM);CHKERRQ(ierr);break;
    case SVD_CONV_MAXIT:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Maxit convergence test not supported in this solver");
    case SVD_CONV_USER:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"User-defined convergence test not supported in this solver");
    }
  }
  SVDCheckUnsupported(svd,SVD_FEATURE_STOPPING);
  /* Transfer the trackall option from svd to eps */
  ierr = SVDGetTrackAll(svd,&trackall);CHKERRQ(ierr);
  ierr = EPSSetTrackAll(cyclic->eps,trackall);CHKERRQ(ierr);
  /* Transfer the initial subspace from svd to eps */
  if (svd->nini<0 || svd->ninil<0) {
    for (i=0;i<-PetscMin(svd->nini,svd->ninil);i++) {
      ierr = MatCreateVecs(cyclic->C,&v,NULL);CHKERRQ(ierr);
      ierr = VecGetArrayWrite(v,&va);CHKERRQ(ierr);
      if (i<-svd->ninil) {
        ierr = VecGetSize(svd->ISL[i],&isl);CHKERRQ(ierr);
        if (isl!=m) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Size mismatch for left initial vector");
        ierr = VecGetArrayRead(svd->ISL[i],&isa);CHKERRQ(ierr);
        ierr = PetscArraycpy(va,isa,m);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(svd->IS[i],&isa);CHKERRQ(ierr);
      } else {
        ierr = PetscArrayzero(&va,m);CHKERRQ(ierr);
      }
      if (i<-svd->nini) {
        ierr = VecGetSize(svd->IS[i],&isl);CHKERRQ(ierr);
        if (isl!=n) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Size mismatch for right initial vector");
        ierr = VecGetArrayRead(svd->IS[i],&isa);CHKERRQ(ierr);
        ierr = PetscArraycpy(va+m,isa,n);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(svd->IS[i],&isa);CHKERRQ(ierr);
      } else {
        ierr = PetscArrayzero(va+m,n);CHKERRQ(ierr);
      }
      ierr = VecRestoreArrayWrite(v,&va);CHKERRQ(ierr);
      ierr = VecDestroy(&svd->IS[i]);CHKERRQ(ierr);
      svd->IS[i] = v;
    }
    svd->nini = PetscMin(svd->nini,svd->ninil);
    ierr = EPSSetInitialSpace(cyclic->eps,-svd->nini,svd->IS);CHKERRQ(ierr);
    ierr = SlepcBasisDestroy_Private(&svd->nini,&svd->IS);CHKERRQ(ierr);
    ierr = SlepcBasisDestroy_Private(&svd->ninil,&svd->ISL);CHKERRQ(ierr);
  }
  ierr = EPSSetUp(cyclic->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(cyclic->eps,NULL,&svd->ncv,&svd->mpd);CHKERRQ(ierr);
  svd->ncv = PetscMin(svd->ncv,PetscMin(M,N));
  ierr = EPSGetTolerances(cyclic->eps,NULL,&svd->max_it);CHKERRQ(ierr);
  if (svd->tol==PETSC_DEFAULT) svd->tol = SLEPC_DEFAULT_TOL;

  svd->leftbasis = PETSC_TRUE;
  ierr = SVDAllocateSolution(svd,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_Cyclic(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  PetscInt       i,j,nconv;
  PetscScalar    sigma;

  PetscFunctionBegin;
  ierr = EPSSolve(cyclic->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(cyclic->eps,&nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(cyclic->eps,&svd->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(cyclic->eps,(EPSConvergedReason*)&svd->reason);CHKERRQ(ierr);
  for (i=0,j=0;i<nconv;i++) {
    ierr = EPSGetEigenvalue(cyclic->eps,i,&sigma,NULL);CHKERRQ(ierr);
    if (PetscRealPart(sigma) > 0.0) {
      if (svd->isgeneralized && svd->which==SVD_SMALLEST) svd->sigma[j] = 1.0/PetscRealPart(sigma);
      else svd->sigma[j] = PetscRealPart(sigma);
      j++;
    }
  }
  svd->nconv = j;
  PetscFunctionReturn(0);
}

PetscErrorCode SVDComputeVectors_Cyclic(SVD svd)
{
  PetscErrorCode    ierr;
  SVD_CYCLIC        *cyclic = (SVD_CYCLIC*)svd->data;
  PetscInt          i,j,m,p,nconv;
  PetscScalar       *dst,sigma;
  const PetscScalar *src,*px;
  Vec               u,v,x,x1,x2,uv;

  PetscFunctionBegin;
  ierr = EPSGetConverged(cyclic->eps,&nconv);CHKERRQ(ierr);
  ierr = MatCreateVecs(cyclic->C,&x,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->A,&m,NULL);CHKERRQ(ierr);
  if (svd->isgeneralized && svd->which==SVD_SMALLEST) {
    ierr = MatCreateVecsEmpty(svd->B,&x1,&x2);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecsEmpty(svd->A,&x2,&x1);CHKERRQ(ierr);
  }
  if (svd->isgeneralized) {
    ierr = MatCreateVecs(svd->A,NULL,&u);CHKERRQ(ierr);
    ierr = MatCreateVecs(svd->B,NULL,&v);CHKERRQ(ierr);
    ierr = MatGetLocalSize(svd->B,&p,NULL);CHKERRQ(ierr);
  }
  for (i=0,j=0;i<nconv;i++) {
    ierr = EPSGetEigenpair(cyclic->eps,i,&sigma,NULL,x,NULL);CHKERRQ(ierr);
    if (PetscRealPart(sigma) > 0.0) {
      if (svd->isgeneralized) {
        if (svd->which==SVD_SMALLEST) {
          /* evec_i = 1/sqrt(2)*[ v_i; w_i ],  w_i = x_i/c_i */
          ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
          ierr = VecPlaceArray(x2,px);CHKERRQ(ierr);
          ierr = VecPlaceArray(x1,px+p);CHKERRQ(ierr);
          ierr = VecCopy(x2,v);CHKERRQ(ierr);
          ierr = VecScale(v,PETSC_SQRT2);CHKERRQ(ierr);  /* v_i = sqrt(2)*evec_i_1 */
          ierr = VecScale(x1,PETSC_SQRT2);CHKERRQ(ierr); /* w_i = sqrt(2)*evec_i_2 */
          ierr = MatMult(svd->A,x1,u);CHKERRQ(ierr);     /* A*w_i = u_i */
          ierr = VecScale(x1,1.0/PetscSqrtScalar(1.0+sigma*sigma));CHKERRQ(ierr);  /* x_i = w_i*c_i */
          ierr = BVInsertVec(svd->V,j,x1);CHKERRQ(ierr);
          ierr = VecResetArray(x2);CHKERRQ(ierr);
          ierr = VecResetArray(x1);CHKERRQ(ierr);
          ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
        } else {
          /* evec_i = 1/sqrt(2)*[ u_i; w_i ],  w_i = x_i/s_i */
          ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
          ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
          ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
          ierr = VecCopy(x1,u);CHKERRQ(ierr);
          ierr = VecScale(u,PETSC_SQRT2);CHKERRQ(ierr);  /* u_i = sqrt(2)*evec_i_1 */
          ierr = VecScale(x2,PETSC_SQRT2);CHKERRQ(ierr); /* w_i = sqrt(2)*evec_i_2 */
          ierr = MatMult(svd->B,x2,v);CHKERRQ(ierr);     /* B*w_i = v_i */
          ierr = VecScale(x2,1.0/PetscSqrtScalar(1.0+sigma*sigma));CHKERRQ(ierr);  /* x_i = w_i*s_i */
          ierr = BVInsertVec(svd->V,j,x2);CHKERRQ(ierr);
          ierr = VecResetArray(x1);CHKERRQ(ierr);
          ierr = VecResetArray(x2);CHKERRQ(ierr);
          ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
        }
        /* copy [u;v] to U[j] */
        ierr = BVGetColumn(svd->U,j,&uv);CHKERRQ(ierr);
        ierr = VecGetArrayWrite(uv,&dst);CHKERRQ(ierr);
        ierr = VecGetArrayRead(u,&src);CHKERRQ(ierr);
        ierr = PetscArraycpy(dst,src,m);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(u,&src);CHKERRQ(ierr);
        ierr = VecGetArrayRead(v,&src);CHKERRQ(ierr);
        ierr = PetscArraycpy(dst+m,src,p);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(v,&src);CHKERRQ(ierr);
        ierr = VecRestoreArrayWrite(uv,&dst);CHKERRQ(ierr);
        ierr = BVRestoreColumn(svd->U,j,&uv);CHKERRQ(ierr);
      } else {
        ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
        ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
        ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
        ierr = BVInsertVec(svd->U,j,x1);CHKERRQ(ierr);
        ierr = BVScaleColumn(svd->U,j,PETSC_SQRT2);CHKERRQ(ierr);
        ierr = BVInsertVec(svd->V,j,x2);CHKERRQ(ierr);
        ierr = BVScaleColumn(svd->V,j,PETSC_SQRT2);CHKERRQ(ierr);
        ierr = VecResetArray(x1);CHKERRQ(ierr);
        ierr = VecResetArray(x2);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
      }
      j++;
    }
  }
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x1);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);
  if (svd->isgeneralized) {
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSMonitor_Cyclic(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i,j;
  SVD            svd = (SVD)ctx;
  PetscScalar    er,ei;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nconv = 0;
  for (i=0,j=0;i<PetscMin(nest,svd->ncv);i++) {
    er = eigr[i]; ei = eigi[i];
    ierr = STBackTransform(eps->st,1,&er,&ei);CHKERRQ(ierr);
    if (PetscRealPart(er) > 0.0) {
      svd->sigma[j] = PetscRealPart(er);
      svd->errest[j] = errest[i];
      if (errest[i] && errest[i] < svd->tol) nconv++;
      j++;
    }
  }
  nest = j;
  ierr = SVDMonitor(svd,its,nconv,svd->sigma,svd->errest,nest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetFromOptions_Cyclic(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  ST             st;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SVD Cyclic Options");CHKERRQ(ierr);

    ierr = PetscOptionsBool("-svd_cyclic_explicitmatrix","Use cyclic explicit matrix","SVDCyclicSetExplicitMatrix",cyclic->explicitmatrix,&val,&set);CHKERRQ(ierr);
    if (set) { ierr = SVDCyclicSetExplicitMatrix(svd,val);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (!cyclic->eps) { ierr = SVDCyclicGetEPS(svd,&cyclic->eps);CHKERRQ(ierr); }
  if (!cyclic->explicitmatrix && !cyclic->usereps) {
    /* use as default an ST with shell matrix and Jacobi */
    ierr = EPSGetST(cyclic->eps,&st);CHKERRQ(ierr);
    ierr = STSetMatMode(st,ST_MATMODE_SHELL);CHKERRQ(ierr);
  }
  ierr = EPSSetFromOptions(cyclic->eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCyclicSetExplicitMatrix_Cyclic(SVD svd,PetscBool explicitmatrix)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  if (cyclic->explicitmatrix != explicitmatrix) {
    cyclic->explicitmatrix = explicitmatrix;
    svd->state = SVD_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   SVDCyclicSetExplicitMatrix - Indicate if the eigensolver operator
   H(A) = [ 0  A ; A^T 0 ] must be computed explicitly.

   Logically Collective on svd

   Input Parameters:
+  svd      - singular value solver
-  explicit - boolean flag indicating if H(A) is built explicitly

   Options Database Key:
.  -svd_cyclic_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: SVDCyclicGetExplicitMatrix()
@*/
PetscErrorCode SVDCyclicSetExplicitMatrix(SVD svd,PetscBool explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,explicitmatrix,2);
  ierr = PetscTryMethod(svd,"SVDCyclicSetExplicitMatrix_C",(SVD,PetscBool),(svd,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCyclicGetExplicitMatrix_Cyclic(SVD svd,PetscBool *explicitmatrix)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  *explicitmatrix = cyclic->explicitmatrix;
  PetscFunctionReturn(0);
}

/*@
   SVDCyclicGetExplicitMatrix - Returns the flag indicating if H(A) is built explicitly.

   Not Collective

   Input Parameter:
.  svd  - singular value solver

   Output Parameter:
.  explicit - the mode flag

   Level: advanced

.seealso: SVDCyclicSetExplicitMatrix()
@*/
PetscErrorCode SVDCyclicGetExplicitMatrix(SVD svd,PetscBool *explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(explicitmatrix,2);
  ierr = PetscUseMethod(svd,"SVDCyclicGetExplicitMatrix_C",(SVD,PetscBool*),(svd,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCyclicSetEPS_Cyclic(SVD svd,EPS eps)
{
  PetscErrorCode  ierr;
  SVD_CYCLIC      *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(&cyclic->eps);CHKERRQ(ierr);
  cyclic->eps = eps;
  cyclic->usereps = PETSC_TRUE;
  ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->eps);CHKERRQ(ierr);
  svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   SVDCyclicSetEPS - Associate an eigensolver object (EPS) to the
   singular value solver.

   Collective on svd

   Input Parameters:
+  svd - singular value solver
-  eps - the eigensolver object

   Level: advanced

.seealso: SVDCyclicGetEPS()
@*/
PetscErrorCode SVDCyclicSetEPS(SVD svd,EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(svd,1,eps,2);
  ierr = PetscTryMethod(svd,"SVDCyclicSetEPS_C",(SVD,EPS),(svd,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCyclicGetEPS_Cyclic(SVD svd,EPS *eps)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  if (!cyclic->eps) {
    ierr = EPSCreate(PetscObjectComm((PetscObject)svd),&cyclic->eps);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)cyclic->eps,(PetscObject)svd,1);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(cyclic->eps,((PetscObject)svd)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(cyclic->eps,"svd_cyclic_");CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->eps);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)cyclic->eps,((PetscObject)svd)->options);CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
    ierr = EPSMonitorSet(cyclic->eps,EPSMonitor_Cyclic,svd,NULL);CHKERRQ(ierr);
  }
  *eps = cyclic->eps;
  PetscFunctionReturn(0);
}

/*@
   SVDCyclicGetEPS - Retrieve the eigensolver object (EPS) associated
   to the singular value solver.

   Not Collective

   Input Parameter:
.  svd - singular value solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: SVDCyclicSetEPS()
@*/
PetscErrorCode SVDCyclicGetEPS(SVD svd,EPS *eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(eps,2);
  ierr = PetscUseMethod(svd,"SVDCyclicGetEPS_C",(SVD,EPS*),(svd,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDView_Cyclic(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!cyclic->eps) { ierr = SVDCyclicGetEPS(svd,&cyclic->eps);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPrintf(viewer,"  %s matrix\n",cyclic->explicitmatrix?"explicit":"implicit");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = EPSView(cyclic->eps,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDReset_Cyclic(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  ierr = EPSReset(cyclic->eps);CHKERRQ(ierr);
  ierr = MatDestroy(&cyclic->C);CHKERRQ(ierr);
  ierr = MatDestroy(&cyclic->D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_Cyclic(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(&cyclic->eps);CHKERRQ(ierr);
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_Cyclic(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,&cyclic);CHKERRQ(ierr);
  svd->data                = (void*)cyclic;
  svd->ops->solve          = SVDSolve_Cyclic;
  svd->ops->solveg         = SVDSolve_Cyclic;
  svd->ops->setup          = SVDSetUp_Cyclic;
  svd->ops->setfromoptions = SVDSetFromOptions_Cyclic;
  svd->ops->destroy        = SVDDestroy_Cyclic;
  svd->ops->reset          = SVDReset_Cyclic;
  svd->ops->view           = SVDView_Cyclic;
  svd->ops->computevectors = SVDComputeVectors_Cyclic;
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetEPS_C",SVDCyclicSetEPS_Cyclic);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetEPS_C",SVDCyclicGetEPS_Cyclic);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C",SVDCyclicSetExplicitMatrix_Cyclic);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C",SVDCyclicGetExplicitMatrix_Cyclic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

