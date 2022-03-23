/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV implemented as a single Vec
*/

#include <slepc/private/bvimpl.h>
#include "svec.h"

PetscErrorCode BVMult_Svec(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  BV_SVEC           *y = (BV_SVEC*)Y->data,*x = (BV_SVEC*)X->data;
  const PetscScalar *px,*q;
  PetscScalar       *py;
  PetscInt          ldq;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(x->v,&px));
  CHKERRQ(VecGetArray(y->v,&py));
  if (Q) {
    CHKERRQ(MatGetSize(Q,&ldq,NULL));
    CHKERRQ(MatDenseGetArrayRead(Q,&q));
    CHKERRQ(BVMult_BLAS_Private(Y,Y->n,Y->k-Y->l,X->k-X->l,ldq,alpha,px+(X->nc+X->l)*X->n,q+Y->l*ldq+X->l,beta,py+(Y->nc+Y->l)*Y->n));
    CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
  } else CHKERRQ(BVAXPY_BLAS_Private(Y,Y->n,Y->k-Y->l,alpha,px+(X->nc+X->l)*X->n,beta,py+(Y->nc+Y->l)*Y->n));
  CHKERRQ(VecRestoreArrayRead(x->v,&px));
  CHKERRQ(VecRestoreArray(y->v,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultVec_Svec(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  BV_SVEC        *x = (BV_SVEC*)X->data;
  PetscScalar    *px,*py,*qq=q;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(x->v,&px));
  CHKERRQ(VecGetArray(y,&py));
  if (!q) CHKERRQ(VecGetArray(X->buffer,&qq));
  CHKERRQ(BVMultVec_BLAS_Private(X,X->n,X->k-X->l,alpha,px+(X->nc+X->l)*X->n,qq,beta,py));
  if (!q) CHKERRQ(VecRestoreArray(X->buffer,&qq));
  CHKERRQ(VecRestoreArray(x->v,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlace_Svec(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_SVEC           *ctx = (BV_SVEC*)V->data;
  PetscScalar       *pv;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(Q,&ldq,NULL));
  CHKERRQ(VecGetArray(ctx->v,&pv));
  CHKERRQ(MatDenseGetArrayRead(Q,&q));
  CHKERRQ(BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,pv+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_FALSE));
  CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
  CHKERRQ(VecRestoreArray(ctx->v,&pv));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlaceHermitianTranspose_Svec(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_SVEC           *ctx = (BV_SVEC*)V->data;
  PetscScalar       *pv;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(Q,&ldq,NULL));
  CHKERRQ(VecGetArray(ctx->v,&pv));
  CHKERRQ(MatDenseGetArrayRead(Q,&q));
  CHKERRQ(BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,pv+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_TRUE));
  CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
  CHKERRQ(VecRestoreArray(ctx->v,&pv));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDot_Svec(BV X,BV Y,Mat M)
{
  BV_SVEC           *x = (BV_SVEC*)X->data,*y = (BV_SVEC*)Y->data;
  const PetscScalar *px,*py;
  PetscScalar       *m;
  PetscInt          ldm;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(M,&ldm,NULL));
  CHKERRQ(VecGetArrayRead(x->v,&px));
  CHKERRQ(VecGetArrayRead(y->v,&py));
  CHKERRQ(MatDenseGetArray(M,&m));
  CHKERRQ(BVDot_BLAS_Private(X,Y->k-Y->l,X->k-X->l,X->n,ldm,py+(Y->nc+Y->l)*Y->n,px+(X->nc+X->l)*X->n,m+X->l*ldm+Y->l,x->mpi));
  CHKERRQ(MatDenseRestoreArray(M,&m));
  CHKERRQ(VecRestoreArrayRead(x->v,&px));
  CHKERRQ(VecRestoreArrayRead(y->v,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDotVec_Svec(BV X,Vec y,PetscScalar *q)
{
  BV_SVEC           *x = (BV_SVEC*)X->data;
  const PetscScalar *px,*py;
  PetscScalar       *qq=q;
  Vec               z = y;

  PetscFunctionBegin;
  if (PetscUnlikely(X->matrix)) {
    CHKERRQ(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  CHKERRQ(VecGetArrayRead(x->v,&px));
  CHKERRQ(VecGetArrayRead(z,&py));
  if (!q) CHKERRQ(VecGetArray(X->buffer,&qq));
  CHKERRQ(BVDotVec_BLAS_Private(X,X->n,X->k-X->l,px+(X->nc+X->l)*X->n,py,qq,x->mpi));
  if (!q) CHKERRQ(VecRestoreArray(X->buffer,&qq));
  CHKERRQ(VecRestoreArrayRead(z,&py));
  CHKERRQ(VecRestoreArrayRead(x->v,&px));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDotVec_Local_Svec(BV X,Vec y,PetscScalar *m)
{
  BV_SVEC        *x = (BV_SVEC*)X->data;
  PetscScalar    *px,*py;
  Vec            z = y;

  PetscFunctionBegin;
  if (PetscUnlikely(X->matrix)) {
    CHKERRQ(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  CHKERRQ(VecGetArray(x->v,&px));
  CHKERRQ(VecGetArray(z,&py));
  CHKERRQ(BVDotVec_BLAS_Private(X,X->n,X->k-X->l,px+(X->nc+X->l)*X->n,py,m,PETSC_FALSE));
  CHKERRQ(VecRestoreArray(z,&py));
  CHKERRQ(VecRestoreArray(x->v,&px));
  PetscFunctionReturn(0);
}

PetscErrorCode BVScale_Svec(BV bv,PetscInt j,PetscScalar alpha)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(ctx->v,&array));
  if (PetscUnlikely(j<0)) CHKERRQ(BVScale_BLAS_Private(bv,(bv->k-bv->l)*bv->n,array+(bv->nc+bv->l)*bv->n,alpha));
  else CHKERRQ(BVScale_BLAS_Private(bv,bv->n,array+(bv->nc+j)*bv->n,alpha));
  CHKERRQ(VecRestoreArray(ctx->v,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Svec(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(ctx->v,&array));
  if (PetscUnlikely(j<0)) CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,type,val,ctx->mpi));
  else CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,1,array+(bv->nc+j)*bv->n,type,val,ctx->mpi));
  CHKERRQ(VecRestoreArray(ctx->v,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Local_Svec(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(ctx->v,&array));
  if (PetscUnlikely(j<0)) CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,type,val,PETSC_FALSE));
  else CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,1,array+(bv->nc+j)*bv->n,type,val,PETSC_FALSE));
  CHKERRQ(VecRestoreArray(ctx->v,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNormalize_Svec(BV bv,PetscScalar *eigi)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *array,*wi=NULL;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(ctx->v,&array));
  if (eigi) wi = eigi+bv->l;
  CHKERRQ(BVNormalize_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,wi,ctx->mpi));
  CHKERRQ(VecRestoreArray(ctx->v,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMatMult_Svec(BV V,Mat A,BV W)
{
  PetscInt       j;
  Mat            Vmat,Wmat;
  Vec            vv,ww;

  PetscFunctionBegin;
  if (V->vmm) {
    CHKERRQ(BVGetMat(V,&Vmat));
    CHKERRQ(BVGetMat(W,&Wmat));
    CHKERRQ(MatProductCreateWithMat(A,Vmat,NULL,Wmat));
    CHKERRQ(MatProductSetType(Wmat,MATPRODUCT_AB));
    CHKERRQ(MatProductSetFromOptions(Wmat));
    CHKERRQ(MatProductSymbolic(Wmat));
    CHKERRQ(MatProductNumeric(Wmat));
    CHKERRQ(MatProductClear(Wmat));
    CHKERRQ(BVRestoreMat(V,&Vmat));
    CHKERRQ(BVRestoreMat(W,&Wmat));
  } else {
    for (j=0;j<V->k-V->l;j++) {
      CHKERRQ(BVGetColumn(V,V->l+j,&vv));
      CHKERRQ(BVGetColumn(W,W->l+j,&ww));
      CHKERRQ(MatMult(A,vv,ww));
      CHKERRQ(BVRestoreColumn(V,V->l+j,&vv));
      CHKERRQ(BVRestoreColumn(W,W->l+j,&ww));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopy_Svec(BV V,BV W)
{
  BV_SVEC        *v = (BV_SVEC*)V->data,*w = (BV_SVEC*)W->data;
  PetscScalar    *pv,*pw,*pvc,*pwc;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(v->v,&pv));
  CHKERRQ(VecGetArray(w->v,&pw));
  pvc = pv+(V->nc+V->l)*V->n;
  pwc = pw+(W->nc+W->l)*W->n;
  CHKERRQ(PetscArraycpy(pwc,pvc,(V->k-V->l)*V->n));
  CHKERRQ(VecRestoreArray(v->v,&pv));
  CHKERRQ(VecRestoreArray(w->v,&pw));
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopyColumn_Svec(BV V,PetscInt j,PetscInt i)
{
  BV_SVEC        *v = (BV_SVEC*)V->data;
  PetscScalar    *pv;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(v->v,&pv));
  CHKERRQ(PetscArraycpy(pv+(V->nc+i)*V->n,pv+(V->nc+j)*V->n,V->n));
  CHKERRQ(VecRestoreArray(v->v,&pv));
  PetscFunctionReturn(0);
}

PetscErrorCode BVResize_Svec(BV bv,PetscInt m,PetscBool copy)
{
  BV_SVEC           *ctx = (BV_SVEC*)bv->data;
  PetscScalar       *pnew;
  const PetscScalar *pv;
  PetscInt          bs;
  Vec               vnew;
  char              str[50];

  PetscFunctionBegin;
  CHKERRQ(VecGetBlockSize(bv->t,&bs));
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject)bv->t),&vnew));
  CHKERRQ(VecSetType(vnew,((PetscObject)bv->t)->type_name));
  CHKERRQ(VecSetSizes(vnew,m*bv->n,PETSC_DECIDE));
  CHKERRQ(VecSetBlockSize(vnew,bs));
  CHKERRQ(PetscLogObjectParent((PetscObject)bv,(PetscObject)vnew));
  if (((PetscObject)bv)->name) {
    CHKERRQ(PetscSNPrintf(str,sizeof(str),"%s_0",((PetscObject)bv)->name));
    CHKERRQ(PetscObjectSetName((PetscObject)vnew,str));
  }
  if (copy) {
    CHKERRQ(VecGetArrayRead(ctx->v,&pv));
    CHKERRQ(VecGetArray(vnew,&pnew));
    CHKERRQ(PetscArraycpy(pnew,pv,PetscMin(m,bv->m)*bv->n));
    CHKERRQ(VecRestoreArrayRead(ctx->v,&pv));
    CHKERRQ(VecRestoreArray(vnew,&pnew));
  }
  CHKERRQ(VecDestroy(&ctx->v));
  ctx->v = vnew;
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetColumn_Svec(BV bv,PetscInt j,Vec *v)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *pv;
  PetscInt       l;

  PetscFunctionBegin;
  l = BVAvailableVec;
  CHKERRQ(VecGetArray(ctx->v,&pv));
  CHKERRQ(VecPlaceArray(bv->cv[l],pv+(bv->nc+j)*bv->n));
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreColumn_Svec(BV bv,PetscInt j,Vec *v)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscInt       l;

  PetscFunctionBegin;
  l = (j==bv->ci[0])? 0: 1;
  CHKERRQ(VecResetArray(bv->cv[l]));
  CHKERRQ(VecRestoreArray(ctx->v,NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetArray_Svec(BV bv,PetscScalar **a)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(ctx->v,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreArray_Svec(BV bv,PetscScalar **a)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;

  PetscFunctionBegin;
  CHKERRQ(VecRestoreArray(ctx->v,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetArrayRead_Svec(BV bv,const PetscScalar **a)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(ctx->v,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreArrayRead_Svec(BV bv,const PetscScalar **a)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;

  PetscFunctionBegin;
  CHKERRQ(VecRestoreArrayRead(ctx->v,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVView_Svec(BV bv,PetscViewer viewer)
{
  BV_SVEC           *ctx = (BV_SVEC*)bv->data;
  PetscViewerFormat format;
  PetscBool         isascii;
  const char        *bvname,*name;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
    CHKERRQ(VecView(ctx->v,viewer));
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      CHKERRQ(PetscObjectGetName((PetscObject)bv,&bvname));
      CHKERRQ(PetscObjectGetName((PetscObject)ctx->v,&name));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s=reshape(%s,%" PetscInt_FMT ",%" PetscInt_FMT ");clear %s\n",bvname,name,bv->N,bv->nc+bv->m,name));
      if (bv->nc) CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s=%s(:,%" PetscInt_FMT ":end);\n",bvname,bvname,bv->nc+1));
    }
  } else CHKERRQ(VecView(ctx->v,viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDestroy_Svec(BV bv)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&ctx->v));
  CHKERRQ(VecDestroy(&bv->cv[0]));
  CHKERRQ(VecDestroy(&bv->cv[1]));
  CHKERRQ(PetscFree(bv->data));
  bv->cuda = PETSC_FALSE;
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode BVCreate_Svec(BV bv)
{
  BV_SVEC           *ctx;
  PetscInt          nloc,N,bs,tglobal=0,tlocal,lsplit;
  PetscBool         seq;
  PetscScalar       *vv;
  const PetscScalar *aa,*array,*ptr;
  char              str[50];
  BV                parent;
  Vec               vpar;
#if defined(PETSC_HAVE_CUDA)
  PetscScalar       *gpuarray,*gptr;
#endif

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(bv,&ctx));
  bv->data = (void*)ctx;

  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)bv->t,&bv->cuda,VECSEQCUDA,VECMPICUDA,""));
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)bv->t,&ctx->mpi,VECMPI,VECMPICUDA,""));

  CHKERRQ(PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq));
  PetscCheck(seq || ctx->mpi || bv->cuda,PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"BVSVEC does not support the type of the provided template vector");

  CHKERRQ(VecGetLocalSize(bv->t,&nloc));
  CHKERRQ(VecGetSize(bv->t,&N));
  CHKERRQ(VecGetBlockSize(bv->t,&bs));
  tlocal = bv->m*nloc;
  CHKERRQ(PetscIntMultError(bv->m,N,&tglobal));

  if (PetscUnlikely(bv->issplit)) {
    /* split BV: create Vec sharing the memory of the parent BV */
    parent = bv->splitparent;
    lsplit = parent->lsplit;
    vpar = ((BV_SVEC*)parent->data)->v;
    if (bv->cuda) {
#if defined(PETSC_HAVE_CUDA)
      CHKERRQ(VecCUDAGetArray(vpar,&gpuarray));
      gptr = (bv->issplit==1)? gpuarray: gpuarray+lsplit*nloc;
      CHKERRQ(VecCUDARestoreArray(vpar,&gpuarray));
      if (ctx->mpi) CHKERRQ(VecCreateMPICUDAWithArray(PetscObjectComm((PetscObject)bv->t),bs,tlocal,bv->m*N,NULL,&ctx->v));
      else CHKERRQ(VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)bv->t),bs,tlocal,NULL,&ctx->v));
      CHKERRQ(VecCUDAPlaceArray(ctx->v,gptr));
#endif
    } else {
      CHKERRQ(VecGetArrayRead(vpar,&array));
      ptr = (bv->issplit==1)? array: array+lsplit*nloc;
      CHKERRQ(VecRestoreArrayRead(vpar,&array));
      if (ctx->mpi) CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,tlocal,bv->m*N,NULL,&ctx->v));
      else CHKERRQ(VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,tlocal,NULL,&ctx->v));
      CHKERRQ(VecPlaceArray(ctx->v,ptr));
    }
  } else {
    /* regular BV: create Vec to store the BV entries */
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)bv->t),&ctx->v));
    CHKERRQ(VecSetType(ctx->v,((PetscObject)bv->t)->type_name));
    CHKERRQ(VecSetSizes(ctx->v,tlocal,tglobal));
    CHKERRQ(VecSetBlockSize(ctx->v,bs));
  }
  CHKERRQ(PetscLogObjectParent((PetscObject)bv,(PetscObject)ctx->v));
  if (((PetscObject)bv)->name) {
    CHKERRQ(PetscSNPrintf(str,sizeof(str),"%s_0",((PetscObject)bv)->name));
    CHKERRQ(PetscObjectSetName((PetscObject)ctx->v,str));
  }

  if (PetscUnlikely(bv->Acreate)) {
    CHKERRQ(MatDenseGetArrayRead(bv->Acreate,&aa));
    CHKERRQ(VecGetArray(ctx->v,&vv));
    CHKERRQ(PetscArraycpy(vv,aa,tlocal));
    CHKERRQ(VecRestoreArray(ctx->v,&vv));
    CHKERRQ(MatDenseRestoreArrayRead(bv->Acreate,&aa));
    CHKERRQ(MatDestroy(&bv->Acreate));
  }

  CHKERRQ(VecDuplicateEmpty(bv->t,&bv->cv[0]));
  CHKERRQ(VecDuplicateEmpty(bv->t,&bv->cv[1]));

  if (bv->cuda) {
#if defined(PETSC_HAVE_CUDA)
    bv->ops->mult             = BVMult_Svec_CUDA;
    bv->ops->multvec          = BVMultVec_Svec_CUDA;
    bv->ops->multinplace      = BVMultInPlace_Svec_CUDA;
    bv->ops->multinplacetrans = BVMultInPlaceHermitianTranspose_Svec_CUDA;
    bv->ops->dot              = BVDot_Svec_CUDA;
    bv->ops->dotvec           = BVDotVec_Svec_CUDA;
    bv->ops->dotvec_local     = BVDotVec_Local_Svec_CUDA;
    bv->ops->scale            = BVScale_Svec_CUDA;
    bv->ops->matmult          = BVMatMult_Svec_CUDA;
    bv->ops->copy             = BVCopy_Svec_CUDA;
    bv->ops->copycolumn       = BVCopyColumn_Svec_CUDA;
    bv->ops->resize           = BVResize_Svec_CUDA;
    bv->ops->getcolumn        = BVGetColumn_Svec_CUDA;
    bv->ops->restorecolumn    = BVRestoreColumn_Svec_CUDA;
    bv->ops->restoresplit     = BVRestoreSplit_Svec_CUDA;
    bv->ops->getmat           = BVGetMat_Svec_CUDA;
    bv->ops->restoremat       = BVRestoreMat_Svec_CUDA;
#endif
  } else {
    bv->ops->mult             = BVMult_Svec;
    bv->ops->multvec          = BVMultVec_Svec;
    bv->ops->multinplace      = BVMultInPlace_Svec;
    bv->ops->multinplacetrans = BVMultInPlaceHermitianTranspose_Svec;
    bv->ops->dot              = BVDot_Svec;
    bv->ops->dotvec           = BVDotVec_Svec;
    bv->ops->dotvec_local     = BVDotVec_Local_Svec;
    bv->ops->scale            = BVScale_Svec;
    bv->ops->matmult          = BVMatMult_Svec;
    bv->ops->copy             = BVCopy_Svec;
    bv->ops->copycolumn       = BVCopyColumn_Svec;
    bv->ops->resize           = BVResize_Svec;
    bv->ops->getcolumn        = BVGetColumn_Svec;
    bv->ops->restorecolumn    = BVRestoreColumn_Svec;
  }
  bv->ops->norm             = BVNorm_Svec;
  bv->ops->norm_local       = BVNorm_Local_Svec;
  bv->ops->normalize        = BVNormalize_Svec;
  bv->ops->getarray         = BVGetArray_Svec;
  bv->ops->restorearray     = BVRestoreArray_Svec;
  bv->ops->getarrayread     = BVGetArrayRead_Svec;
  bv->ops->restorearrayread = BVRestoreArrayRead_Svec;
  bv->ops->destroy          = BVDestroy_Svec;
  if (!ctx->mpi) bv->ops->view = BVView_Svec;
  PetscFunctionReturn(0);
}
