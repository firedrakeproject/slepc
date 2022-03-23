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
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&ctx));
  CHKERRQ(MatGetLocalSize(ctx->A,&m,NULL));
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArrayWrite(y,&py));
  CHKERRQ(VecPlaceArray(ctx->x1,px));
  CHKERRQ(VecPlaceArray(ctx->x2,px+m));
  CHKERRQ(VecPlaceArray(ctx->y1,py));
  CHKERRQ(VecPlaceArray(ctx->y2,py+m));
  CHKERRQ(MatMult(ctx->A,ctx->x2,ctx->y1));
  CHKERRQ(MatMult(ctx->AT,ctx->x1,ctx->y2));
  CHKERRQ(VecResetArray(ctx->x1));
  CHKERRQ(VecResetArray(ctx->x2));
  CHKERRQ(VecResetArray(ctx->y1));
  CHKERRQ(VecResetArray(ctx->y2));
  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArrayWrite(y,&py));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_Cyclic(Mat B,Vec diag)
{
  PetscFunctionBegin;
  CHKERRQ(VecSet(diag,0.0));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Cyclic(Mat B)
{
  SVD_CYCLIC_SHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&ctx));
  CHKERRQ(VecDestroy(&ctx->x1));
  CHKERRQ(VecDestroy(&ctx->x2));
  CHKERRQ(VecDestroy(&ctx->y1));
  CHKERRQ(VecDestroy(&ctx->y2));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

/*
   Builds cyclic matrix   C = | 0   A |
                              | AT  0 |
*/
static PetscErrorCode SVDCyclicGetCyclicMat(SVD svd,Mat A,Mat AT,Mat *C)
{
  SVD_CYCLIC       *cyclic = (SVD_CYCLIC*)svd->data;
  SVD_CYCLIC_SHELL *ctx;
  PetscInt         i,M,N,m,n,Istart,Iend;
  VecType          vtype;
  Mat              Zm,Zn;
#if defined(PETSC_HAVE_CUDA)
  PetscBool        cuda;
#endif

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetLocalSize(A,&m,&n));

  if (cyclic->explicitmatrix) {
    PetscCheck(svd->expltrans,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Cannot use explicit cyclic matrix with implicit transpose");
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)svd),&Zm));
    CHKERRQ(MatSetSizes(Zm,m,m,M,M));
    CHKERRQ(MatSetFromOptions(Zm));
    CHKERRQ(MatSetUp(Zm));
    CHKERRQ(MatGetOwnershipRange(Zm,&Istart,&Iend));
    for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(Zm,i,i,0.0,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(Zm,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Zm,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)svd),&Zn));
    CHKERRQ(MatSetSizes(Zn,n,n,N,N));
    CHKERRQ(MatSetFromOptions(Zn));
    CHKERRQ(MatSetUp(Zn));
    CHKERRQ(MatGetOwnershipRange(Zn,&Istart,&Iend));
    for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(Zn,i,i,0.0,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(Zn,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Zn,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatCreateTile(1.0,Zm,1.0,A,1.0,AT,1.0,Zn,C));
    CHKERRQ(MatDestroy(&Zm));
    CHKERRQ(MatDestroy(&Zn));
  } else {
    CHKERRQ(PetscNew(&ctx));
    ctx->A       = A;
    ctx->AT      = AT;
    ctx->swapped = svd->swapped;
    CHKERRQ(MatCreateVecsEmpty(A,&ctx->x2,&ctx->x1));
    CHKERRQ(MatCreateVecsEmpty(A,&ctx->y2,&ctx->y1));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->x1));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->x2));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->y1));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->y2));
    CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)svd),m+n,m+n,M+N,M+N,ctx,C));
    CHKERRQ(MatShellSetOperation(*C,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Cyclic));
    CHKERRQ(MatShellSetOperation(*C,MATOP_DESTROY,(void(*)(void))MatDestroy_Cyclic));
#if defined(PETSC_HAVE_CUDA)
    CHKERRQ(PetscObjectTypeCompareAny((PetscObject)(svd->swapped?AT:A),&cuda,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,""));
    if (cuda) CHKERRQ(MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_Cyclic_CUDA));
    else
#endif
      CHKERRQ(MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_Cyclic));
    CHKERRQ(MatGetVecType(A,&vtype));
    CHKERRQ(MatSetVecType(*C,vtype));
  }
  CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)*C));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_ECross(Mat B,Vec x,Vec y)
{
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          mn,m,n;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&ctx));
  CHKERRQ(MatGetLocalSize(ctx->A,NULL,&n));
  CHKERRQ(VecGetLocalSize(y,&mn));
  m = mn-n;
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArrayWrite(y,&py));
  CHKERRQ(VecPlaceArray(ctx->x1,px));
  CHKERRQ(VecPlaceArray(ctx->x2,px+m));
  CHKERRQ(VecPlaceArray(ctx->y1,py));
  CHKERRQ(VecPlaceArray(ctx->y2,py+m));
  CHKERRQ(VecCopy(ctx->x1,ctx->y1));
  CHKERRQ(MatMult(ctx->A,ctx->x2,ctx->w));
  CHKERRQ(MatMult(ctx->AT,ctx->w,ctx->y2));
  CHKERRQ(VecResetArray(ctx->x1));
  CHKERRQ(VecResetArray(ctx->x2));
  CHKERRQ(VecResetArray(ctx->y1));
  CHKERRQ(VecResetArray(ctx->y2));
  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArrayWrite(y,&py));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_ECross(Mat B,Vec d)
{
  SVD_CYCLIC_SHELL  *ctx;
  PetscScalar       *pd;
  PetscMPIInt       len;
  PetscInt          mn,m,n,N,i,j,start,end,ncols;
  PetscScalar       *work1,*work2,*diag;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&ctx));
  CHKERRQ(MatGetLocalSize(ctx->A,NULL,&n));
  CHKERRQ(VecGetLocalSize(d,&mn));
  m = mn-n;
  CHKERRQ(VecGetArrayWrite(d,&pd));
  CHKERRQ(VecPlaceArray(ctx->y1,pd));
  CHKERRQ(VecSet(ctx->y1,1.0));
  CHKERRQ(VecResetArray(ctx->y1));
  CHKERRQ(VecPlaceArray(ctx->y2,pd+m));
  if (!ctx->diag) {
    /* compute diagonal from rows and store in ctx->diag */
    CHKERRQ(VecDuplicate(ctx->y2,&ctx->diag));
    CHKERRQ(MatGetSize(ctx->A,NULL,&N));
    CHKERRQ(PetscCalloc2(N,&work1,N,&work2));
    if (ctx->swapped) {
      CHKERRQ(MatGetOwnershipRange(ctx->AT,&start,&end));
      for (i=start;i<end;i++) {
        CHKERRQ(MatGetRow(ctx->AT,i,&ncols,NULL,&vals));
        for (j=0;j<ncols;j++) work1[i] += vals[j]*vals[j];
        CHKERRQ(MatRestoreRow(ctx->AT,i,&ncols,NULL,&vals));
      }
    } else {
      CHKERRQ(MatGetOwnershipRange(ctx->A,&start,&end));
      for (i=start;i<end;i++) {
        CHKERRQ(MatGetRow(ctx->A,i,&ncols,&cols,&vals));
        for (j=0;j<ncols;j++) work1[cols[j]] += vals[j]*vals[j];
        CHKERRQ(MatRestoreRow(ctx->A,i,&ncols,&cols,&vals));
      }
    }
    CHKERRQ(PetscMPIIntCast(N,&len));
    CHKERRMPI(MPIU_Allreduce(work1,work2,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)B)));
    CHKERRQ(VecGetOwnershipRange(ctx->diag,&start,&end));
    CHKERRQ(VecGetArrayWrite(ctx->diag,&diag));
    for (i=start;i<end;i++) diag[i-start] = work2[i];
    CHKERRQ(VecRestoreArrayWrite(ctx->diag,&diag));
    CHKERRQ(PetscFree2(work1,work2));
  }
  CHKERRQ(VecCopy(ctx->diag,ctx->y2));
  CHKERRQ(VecResetArray(ctx->y2));
  CHKERRQ(VecRestoreArrayWrite(d,&pd));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_ECross(Mat B)
{
  SVD_CYCLIC_SHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&ctx));
  CHKERRQ(VecDestroy(&ctx->x1));
  CHKERRQ(VecDestroy(&ctx->x2));
  CHKERRQ(VecDestroy(&ctx->y1));
  CHKERRQ(VecDestroy(&ctx->y2));
  CHKERRQ(VecDestroy(&ctx->diag));
  CHKERRQ(VecDestroy(&ctx->w));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

/*
   Builds extended cross product matrix   C = | I_m   0  |
                                              |  0  AT*A |
   t is an auxiliary Vec used to take the dimensions of the upper block
*/
static PetscErrorCode SVDCyclicGetECrossMat(SVD svd,Mat A,Mat AT,Mat *C,Vec t)
{
  SVD_CYCLIC       *cyclic = (SVD_CYCLIC*)svd->data;
  SVD_CYCLIC_SHELL *ctx;
  PetscInt         i,M,N,m,n,Istart,Iend;
  VecType          vtype;
  Mat              Id,Zm,Zn,ATA;
#if defined(PETSC_HAVE_CUDA)
  PetscBool        cuda;
#endif

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,NULL,&N));
  CHKERRQ(MatGetLocalSize(A,NULL,&n));
  CHKERRQ(VecGetSize(t,&M));
  CHKERRQ(VecGetLocalSize(t,&m));

  if (cyclic->explicitmatrix) {
    PetscCheck(svd->expltrans,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Cannot use explicit cyclic matrix with implicit transpose");
    CHKERRQ(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)svd),m,m,M,M,1.0,&Id));
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)svd),&Zm));
    CHKERRQ(MatSetSizes(Zm,m,n,M,N));
    CHKERRQ(MatSetFromOptions(Zm));
    CHKERRQ(MatSetUp(Zm));
    CHKERRQ(MatGetOwnershipRange(Zm,&Istart,&Iend));
    for (i=Istart;i<Iend;i++) {
      if (i<N) CHKERRQ(MatSetValue(Zm,i,i,0.0,INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(Zm,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Zm,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)svd),&Zn));
    CHKERRQ(MatSetSizes(Zn,n,m,N,M));
    CHKERRQ(MatSetFromOptions(Zn));
    CHKERRQ(MatSetUp(Zn));
    CHKERRQ(MatGetOwnershipRange(Zn,&Istart,&Iend));
    for (i=Istart;i<Iend;i++) {
      if (i<m) CHKERRQ(MatSetValue(Zn,i,i,0.0,INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(Zn,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Zn,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatProductCreate(AT,A,NULL,&ATA));
    CHKERRQ(MatProductSetType(ATA,MATPRODUCT_AB));
    CHKERRQ(MatProductSetFromOptions(ATA));
    CHKERRQ(MatProductSymbolic(ATA));
    CHKERRQ(MatProductNumeric(ATA));
    CHKERRQ(MatCreateTile(1.0,Id,1.0,Zm,1.0,Zn,1.0,ATA,C));
    CHKERRQ(MatDestroy(&Id));
    CHKERRQ(MatDestroy(&Zm));
    CHKERRQ(MatDestroy(&Zn));
    CHKERRQ(MatDestroy(&ATA));
  } else {
    CHKERRQ(PetscNew(&ctx));
    ctx->A       = A;
    ctx->AT      = AT;
    ctx->swapped = svd->swapped;
    CHKERRQ(VecDuplicateEmpty(t,&ctx->x1));
    CHKERRQ(VecDuplicateEmpty(t,&ctx->y1));
    CHKERRQ(MatCreateVecsEmpty(A,&ctx->x2,NULL));
    CHKERRQ(MatCreateVecsEmpty(A,&ctx->y2,NULL));
    CHKERRQ(MatCreateVecs(A,NULL,&ctx->w));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->x1));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->x2));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->y1));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->y2));
    CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)svd),m+n,m+n,M+N,M+N,ctx,C));
    CHKERRQ(MatShellSetOperation(*C,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_ECross));
    CHKERRQ(MatShellSetOperation(*C,MATOP_DESTROY,(void(*)(void))MatDestroy_ECross));
#if defined(PETSC_HAVE_CUDA)
    CHKERRQ(PetscObjectTypeCompareAny((PetscObject)(svd->swapped?AT:A),&cuda,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,""));
    if (cuda) CHKERRQ(MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_ECross_CUDA));
    else
#endif
      CHKERRQ(MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_ECross));
    CHKERRQ(MatGetVecType(A,&vtype));
    CHKERRQ(MatSetVecType(*C,vtype));
  }
  CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)*C));
  PetscFunctionReturn(0);
}

/* Convergence test relative to the norm of R (used in GSVD only) */
static PetscErrorCode EPSConv_Cyclic(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  SVD svd = (SVD)ctx;

  PetscFunctionBegin;
  *errest = res/PetscMax(svd->nrma,svd->nrmb);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetUp_Cyclic(SVD svd)
{
  SVD_CYCLIC        *cyclic = (SVD_CYCLIC*)svd->data;
  PetscInt          M,N,m,n,i,isl;
  const PetscScalar *isa;
  PetscScalar       *va;
  PetscBool         trackall,issinv;
  Vec               v,t;
  ST                st;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(svd->A,&M,&N));
  CHKERRQ(MatGetLocalSize(svd->A,&m,&n));
  if (!cyclic->eps) CHKERRQ(SVDCyclicGetEPS(svd,&cyclic->eps));
  CHKERRQ(MatDestroy(&cyclic->C));
  CHKERRQ(MatDestroy(&cyclic->D));
  if (svd->isgeneralized) {
    if (svd->which==SVD_SMALLEST) {  /* alternative pencil */
      CHKERRQ(MatCreateVecs(svd->B,NULL,&t));
      CHKERRQ(SVDCyclicGetCyclicMat(svd,svd->B,svd->BT,&cyclic->C));
      CHKERRQ(SVDCyclicGetECrossMat(svd,svd->A,svd->AT,&cyclic->D,t));
    } else {
      CHKERRQ(MatCreateVecs(svd->A,NULL,&t));
      CHKERRQ(SVDCyclicGetCyclicMat(svd,svd->A,svd->AT,&cyclic->C));
      CHKERRQ(SVDCyclicGetECrossMat(svd,svd->B,svd->BT,&cyclic->D,t));
    }
    CHKERRQ(VecDestroy(&t));
    CHKERRQ(EPSSetOperators(cyclic->eps,cyclic->C,cyclic->D));
    CHKERRQ(EPSSetProblemType(cyclic->eps,EPS_GHEP));
  } else {
    CHKERRQ(SVDCyclicGetCyclicMat(svd,svd->A,svd->AT,&cyclic->C));
    CHKERRQ(EPSSetOperators(cyclic->eps,cyclic->C,NULL));
    CHKERRQ(EPSSetProblemType(cyclic->eps,EPS_HEP));
  }
  if (!cyclic->usereps) {
    if (svd->which == SVD_LARGEST) {
      CHKERRQ(EPSGetST(cyclic->eps,&st));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)st,STSINVERT,&issinv));
      if (issinv) CHKERRQ(EPSSetWhichEigenpairs(cyclic->eps,EPS_TARGET_MAGNITUDE));
      else CHKERRQ(EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL));
    } else {
      if (svd->isgeneralized) {  /* computes sigma^{-1} via alternative pencil */
        CHKERRQ(EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL));
      } else {
        CHKERRQ(EPSSetEigenvalueComparison(cyclic->eps,SlepcCompareSmallestPosReal,NULL));
        CHKERRQ(EPSSetTarget(cyclic->eps,0.0));
      }
    }
    CHKERRQ(EPSSetDimensions(cyclic->eps,svd->nsv,svd->ncv,svd->mpd));
    CHKERRQ(EPSSetTolerances(cyclic->eps,svd->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL/10.0:svd->tol,svd->max_it));
    switch (svd->conv) {
    case SVD_CONV_ABS:
      CHKERRQ(EPSSetConvergenceTest(cyclic->eps,EPS_CONV_ABS));break;
    case SVD_CONV_REL:
      CHKERRQ(EPSSetConvergenceTest(cyclic->eps,EPS_CONV_REL));break;
    case SVD_CONV_NORM:
      if (svd->isgeneralized) {
        if (!svd->nrma) CHKERRQ(MatNorm(svd->OP,NORM_INFINITY,&svd->nrma));
        if (!svd->nrmb) CHKERRQ(MatNorm(svd->OPb,NORM_INFINITY,&svd->nrmb));
        CHKERRQ(EPSSetConvergenceTestFunction(cyclic->eps,EPSConv_Cyclic,svd,NULL));
      } else {
        CHKERRQ(EPSSetConvergenceTest(cyclic->eps,EPS_CONV_NORM));break;
      }
      break;
    case SVD_CONV_MAXIT:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Maxit convergence test not supported in this solver");
    case SVD_CONV_USER:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"User-defined convergence test not supported in this solver");
    }
  }
  SVDCheckUnsupported(svd,SVD_FEATURE_STOPPING);
  /* Transfer the trackall option from svd to eps */
  CHKERRQ(SVDGetTrackAll(svd,&trackall));
  CHKERRQ(EPSSetTrackAll(cyclic->eps,trackall));
  /* Transfer the initial subspace from svd to eps */
  if (svd->nini<0 || svd->ninil<0) {
    for (i=0;i<-PetscMin(svd->nini,svd->ninil);i++) {
      CHKERRQ(MatCreateVecs(cyclic->C,&v,NULL));
      CHKERRQ(VecGetArrayWrite(v,&va));
      if (i<-svd->ninil) {
        CHKERRQ(VecGetSize(svd->ISL[i],&isl));
        PetscCheck(isl==m,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Size mismatch for left initial vector");
        CHKERRQ(VecGetArrayRead(svd->ISL[i],&isa));
        CHKERRQ(PetscArraycpy(va,isa,m));
        CHKERRQ(VecRestoreArrayRead(svd->IS[i],&isa));
      } else CHKERRQ(PetscArrayzero(&va,m));
      if (i<-svd->nini) {
        CHKERRQ(VecGetSize(svd->IS[i],&isl));
        PetscCheck(isl==n,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Size mismatch for right initial vector");
        CHKERRQ(VecGetArrayRead(svd->IS[i],&isa));
        CHKERRQ(PetscArraycpy(va+m,isa,n));
        CHKERRQ(VecRestoreArrayRead(svd->IS[i],&isa));
      } else CHKERRQ(PetscArrayzero(va+m,n));
      CHKERRQ(VecRestoreArrayWrite(v,&va));
      CHKERRQ(VecDestroy(&svd->IS[i]));
      svd->IS[i] = v;
    }
    svd->nini = PetscMin(svd->nini,svd->ninil);
    CHKERRQ(EPSSetInitialSpace(cyclic->eps,-svd->nini,svd->IS));
    CHKERRQ(SlepcBasisDestroy_Private(&svd->nini,&svd->IS));
    CHKERRQ(SlepcBasisDestroy_Private(&svd->ninil,&svd->ISL));
  }
  CHKERRQ(EPSSetUp(cyclic->eps));
  CHKERRQ(EPSGetDimensions(cyclic->eps,NULL,&svd->ncv,&svd->mpd));
  svd->ncv = PetscMin(svd->ncv,PetscMin(M,N));
  CHKERRQ(EPSGetTolerances(cyclic->eps,NULL,&svd->max_it));
  if (svd->tol==PETSC_DEFAULT) svd->tol = SLEPC_DEFAULT_TOL;

  svd->leftbasis = PETSC_TRUE;
  CHKERRQ(SVDAllocateSolution(svd,0));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_Cyclic(SVD svd)
{
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  PetscInt       i,j,nconv;
  PetscScalar    sigma;

  PetscFunctionBegin;
  CHKERRQ(EPSSolve(cyclic->eps));
  CHKERRQ(EPSGetConverged(cyclic->eps,&nconv));
  CHKERRQ(EPSGetIterationNumber(cyclic->eps,&svd->its));
  CHKERRQ(EPSGetConvergedReason(cyclic->eps,(EPSConvergedReason*)&svd->reason));
  for (i=0,j=0;i<nconv;i++) {
    CHKERRQ(EPSGetEigenvalue(cyclic->eps,i,&sigma,NULL));
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
  SVD_CYCLIC        *cyclic = (SVD_CYCLIC*)svd->data;
  PetscInt          i,j,m,p,nconv;
  PetscScalar       *dst,sigma;
  const PetscScalar *src,*px;
  Vec               u,v,x,x1,x2,uv;

  PetscFunctionBegin;
  CHKERRQ(EPSGetConverged(cyclic->eps,&nconv));
  CHKERRQ(MatCreateVecs(cyclic->C,&x,NULL));
  CHKERRQ(MatGetLocalSize(svd->A,&m,NULL));
  if (svd->isgeneralized && svd->which==SVD_SMALLEST) CHKERRQ(MatCreateVecsEmpty(svd->B,&x1,&x2));
  else CHKERRQ(MatCreateVecsEmpty(svd->A,&x2,&x1));
  if (svd->isgeneralized) {
    CHKERRQ(MatCreateVecs(svd->A,NULL,&u));
    CHKERRQ(MatCreateVecs(svd->B,NULL,&v));
    CHKERRQ(MatGetLocalSize(svd->B,&p,NULL));
  }
  for (i=0,j=0;i<nconv;i++) {
    CHKERRQ(EPSGetEigenpair(cyclic->eps,i,&sigma,NULL,x,NULL));
    if (PetscRealPart(sigma) > 0.0) {
      if (svd->isgeneralized) {
        if (svd->which==SVD_SMALLEST) {
          /* evec_i = 1/sqrt(2)*[ v_i; w_i ],  w_i = x_i/c_i */
          CHKERRQ(VecGetArrayRead(x,&px));
          CHKERRQ(VecPlaceArray(x2,px));
          CHKERRQ(VecPlaceArray(x1,px+p));
          CHKERRQ(VecCopy(x2,v));
          CHKERRQ(VecScale(v,PETSC_SQRT2));  /* v_i = sqrt(2)*evec_i_1 */
          CHKERRQ(VecScale(x1,PETSC_SQRT2)); /* w_i = sqrt(2)*evec_i_2 */
          CHKERRQ(MatMult(svd->A,x1,u));     /* A*w_i = u_i */
          CHKERRQ(VecScale(x1,1.0/PetscSqrtScalar(1.0+sigma*sigma)));  /* x_i = w_i*c_i */
          CHKERRQ(BVInsertVec(svd->V,j,x1));
          CHKERRQ(VecResetArray(x2));
          CHKERRQ(VecResetArray(x1));
          CHKERRQ(VecRestoreArrayRead(x,&px));
        } else {
          /* evec_i = 1/sqrt(2)*[ u_i; w_i ],  w_i = x_i/s_i */
          CHKERRQ(VecGetArrayRead(x,&px));
          CHKERRQ(VecPlaceArray(x1,px));
          CHKERRQ(VecPlaceArray(x2,px+m));
          CHKERRQ(VecCopy(x1,u));
          CHKERRQ(VecScale(u,PETSC_SQRT2));  /* u_i = sqrt(2)*evec_i_1 */
          CHKERRQ(VecScale(x2,PETSC_SQRT2)); /* w_i = sqrt(2)*evec_i_2 */
          CHKERRQ(MatMult(svd->B,x2,v));     /* B*w_i = v_i */
          CHKERRQ(VecScale(x2,1.0/PetscSqrtScalar(1.0+sigma*sigma)));  /* x_i = w_i*s_i */
          CHKERRQ(BVInsertVec(svd->V,j,x2));
          CHKERRQ(VecResetArray(x1));
          CHKERRQ(VecResetArray(x2));
          CHKERRQ(VecRestoreArrayRead(x,&px));
        }
        /* copy [u;v] to U[j] */
        CHKERRQ(BVGetColumn(svd->U,j,&uv));
        CHKERRQ(VecGetArrayWrite(uv,&dst));
        CHKERRQ(VecGetArrayRead(u,&src));
        CHKERRQ(PetscArraycpy(dst,src,m));
        CHKERRQ(VecRestoreArrayRead(u,&src));
        CHKERRQ(VecGetArrayRead(v,&src));
        CHKERRQ(PetscArraycpy(dst+m,src,p));
        CHKERRQ(VecRestoreArrayRead(v,&src));
        CHKERRQ(VecRestoreArrayWrite(uv,&dst));
        CHKERRQ(BVRestoreColumn(svd->U,j,&uv));
      } else {
        CHKERRQ(VecGetArrayRead(x,&px));
        CHKERRQ(VecPlaceArray(x1,px));
        CHKERRQ(VecPlaceArray(x2,px+m));
        CHKERRQ(BVInsertVec(svd->U,j,x1));
        CHKERRQ(BVScaleColumn(svd->U,j,PETSC_SQRT2));
        CHKERRQ(BVInsertVec(svd->V,j,x2));
        CHKERRQ(BVScaleColumn(svd->V,j,PETSC_SQRT2));
        CHKERRQ(VecResetArray(x1));
        CHKERRQ(VecResetArray(x2));
        CHKERRQ(VecRestoreArrayRead(x,&px));
      }
      j++;
    }
  }
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&x1));
  CHKERRQ(VecDestroy(&x2));
  if (svd->isgeneralized) {
    CHKERRQ(VecDestroy(&u));
    CHKERRQ(VecDestroy(&v));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSMonitor_Cyclic(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i,j;
  SVD            svd = (SVD)ctx;
  PetscScalar    er,ei;

  PetscFunctionBegin;
  nconv = 0;
  for (i=0,j=0;i<PetscMin(nest,svd->ncv);i++) {
    er = eigr[i]; ei = eigi[i];
    CHKERRQ(STBackTransform(eps->st,1,&er,&ei));
    if (PetscRealPart(er) > 0.0) {
      svd->sigma[j] = PetscRealPart(er);
      svd->errest[j] = errest[i];
      if (errest[i] && errest[i] < svd->tol) nconv++;
      j++;
    }
  }
  nest = j;
  CHKERRQ(SVDMonitor(svd,its,nconv,svd->sigma,svd->errest,nest));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetFromOptions_Cyclic(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscBool      set,val;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  ST             st;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"SVD Cyclic Options"));

    CHKERRQ(PetscOptionsBool("-svd_cyclic_explicitmatrix","Use cyclic explicit matrix","SVDCyclicSetExplicitMatrix",cyclic->explicitmatrix,&val,&set));
    if (set) CHKERRQ(SVDCyclicSetExplicitMatrix(svd,val));

  CHKERRQ(PetscOptionsTail());

  if (!cyclic->eps) CHKERRQ(SVDCyclicGetEPS(svd,&cyclic->eps));
  if (!cyclic->explicitmatrix && !cyclic->usereps) {
    /* use as default an ST with shell matrix and Jacobi */
    CHKERRQ(EPSGetST(cyclic->eps,&st));
    CHKERRQ(STSetMatMode(st,ST_MATMODE_SHELL));
  }
  CHKERRQ(EPSSetFromOptions(cyclic->eps));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCyclicSetExplicitMatrix_Cyclic(SVD svd,PetscBool explicitmat)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  if (cyclic->explicitmatrix != explicitmat) {
    cyclic->explicitmatrix = explicitmat;
    svd->state = SVD_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   SVDCyclicSetExplicitMatrix - Indicate if the eigensolver operator
   H(A) = [ 0  A ; A^T 0 ] must be computed explicitly.

   Logically Collective on svd

   Input Parameters:
+  svd         - singular value solver
-  explicitmat - boolean flag indicating if H(A) is built explicitly

   Options Database Key:
.  -svd_cyclic_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: SVDCyclicGetExplicitMatrix()
@*/
PetscErrorCode SVDCyclicSetExplicitMatrix(SVD svd,PetscBool explicitmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,explicitmat,2);
  CHKERRQ(PetscTryMethod(svd,"SVDCyclicSetExplicitMatrix_C",(SVD,PetscBool),(svd,explicitmat)));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCyclicGetExplicitMatrix_Cyclic(SVD svd,PetscBool *explicitmat)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  *explicitmat = cyclic->explicitmatrix;
  PetscFunctionReturn(0);
}

/*@
   SVDCyclicGetExplicitMatrix - Returns the flag indicating if H(A) is built explicitly.

   Not Collective

   Input Parameter:
.  svd  - singular value solver

   Output Parameter:
.  explicitmat - the mode flag

   Level: advanced

.seealso: SVDCyclicSetExplicitMatrix()
@*/
PetscErrorCode SVDCyclicGetExplicitMatrix(SVD svd,PetscBool *explicitmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(explicitmat,2);
  CHKERRQ(PetscUseMethod(svd,"SVDCyclicGetExplicitMatrix_C",(SVD,PetscBool*),(svd,explicitmat)));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCyclicSetEPS_Cyclic(SVD svd,EPS eps)
{
  SVD_CYCLIC      *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)eps));
  CHKERRQ(EPSDestroy(&cyclic->eps));
  cyclic->eps = eps;
  cyclic->usereps = PETSC_TRUE;
  CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->eps));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(svd,1,eps,2);
  CHKERRQ(PetscTryMethod(svd,"SVDCyclicSetEPS_C",(SVD,EPS),(svd,eps)));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCyclicGetEPS_Cyclic(SVD svd,EPS *eps)
{
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  if (!cyclic->eps) {
    CHKERRQ(EPSCreate(PetscObjectComm((PetscObject)svd),&cyclic->eps));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)cyclic->eps,(PetscObject)svd,1));
    CHKERRQ(EPSSetOptionsPrefix(cyclic->eps,((PetscObject)svd)->prefix));
    CHKERRQ(EPSAppendOptionsPrefix(cyclic->eps,"svd_cyclic_"));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->eps));
    CHKERRQ(PetscObjectSetOptions((PetscObject)cyclic->eps,((PetscObject)svd)->options));
    CHKERRQ(EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL));
    CHKERRQ(EPSMonitorSet(cyclic->eps,EPSMonitor_Cyclic,svd,NULL));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(eps,2);
  CHKERRQ(PetscUseMethod(svd,"SVDCyclicGetEPS_C",(SVD,EPS*),(svd,eps)));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDView_Cyclic(SVD svd,PetscViewer viewer)
{
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    if (!cyclic->eps) CHKERRQ(SVDCyclicGetEPS(svd,&cyclic->eps));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %s matrix\n",cyclic->explicitmatrix?"explicit":"implicit"));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(EPSView(cyclic->eps,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDReset_Cyclic(SVD svd)
{
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  CHKERRQ(EPSReset(cyclic->eps));
  CHKERRQ(MatDestroy(&cyclic->C));
  CHKERRQ(MatDestroy(&cyclic->D));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_Cyclic(SVD svd)
{
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  CHKERRQ(EPSDestroy(&cyclic->eps));
  CHKERRQ(PetscFree(svd->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetEPS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetEPS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_Cyclic(SVD svd)
{
  SVD_CYCLIC     *cyclic;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(svd,&cyclic));
  svd->data                = (void*)cyclic;
  svd->ops->solve          = SVDSolve_Cyclic;
  svd->ops->solveg         = SVDSolve_Cyclic;
  svd->ops->setup          = SVDSetUp_Cyclic;
  svd->ops->setfromoptions = SVDSetFromOptions_Cyclic;
  svd->ops->destroy        = SVDDestroy_Cyclic;
  svd->ops->reset          = SVDReset_Cyclic;
  svd->ops->view           = SVDView_Cyclic;
  svd->ops->computevectors = SVDComputeVectors_Cyclic;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetEPS_C",SVDCyclicSetEPS_Cyclic));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetEPS_C",SVDCyclicGetEPS_Cyclic));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C",SVDCyclicSetExplicitMatrix_Cyclic));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C",SVDCyclicGetExplicitMatrix_Cyclic));
  PetscFunctionReturn(0);
}
