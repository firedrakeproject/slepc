/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV implemented as an array of Vecs sharing a contiguous array for elements
*/

#include <slepc/private/bvimpl.h>

typedef struct {
  Vec         *V;
  PetscScalar *array;
  PetscBool   mpi;
} BV_CONTIGUOUS;

PetscErrorCode BVMult_Contiguous(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  BV_CONTIGUOUS     *y = (BV_CONTIGUOUS*)Y->data,*x = (BV_CONTIGUOUS*)X->data;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  if (Q) {
    CHKERRQ(MatGetSize(Q,&ldq,NULL));
    CHKERRQ(MatDenseGetArrayRead(Q,&q));
    CHKERRQ(BVMult_BLAS_Private(Y,Y->n,Y->k-Y->l,X->k-X->l,ldq,alpha,x->array+(X->nc+X->l)*X->n,q+Y->l*ldq+X->l,beta,y->array+(Y->nc+Y->l)*Y->n));
    CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
  } else CHKERRQ(BVAXPY_BLAS_Private(Y,Y->n,Y->k-Y->l,alpha,x->array+(X->nc+X->l)*X->n,beta,y->array+(Y->nc+Y->l)*Y->n));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultVec_Contiguous(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data;
  PetscScalar    *py,*qq=q;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(y,&py));
  if (!q) CHKERRQ(VecGetArray(X->buffer,&qq));
  CHKERRQ(BVMultVec_BLAS_Private(X,X->n,X->k-X->l,alpha,x->array+(X->nc+X->l)*X->n,qq,beta,py));
  if (!q) CHKERRQ(VecRestoreArray(X->buffer,&qq));
  CHKERRQ(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlace_Contiguous(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_CONTIGUOUS     *ctx = (BV_CONTIGUOUS*)V->data;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(Q,&ldq,NULL));
  CHKERRQ(MatDenseGetArrayRead(Q,&q));
  CHKERRQ(BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,ctx->array+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_FALSE));
  CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlaceHermitianTranspose_Contiguous(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_CONTIGUOUS     *ctx = (BV_CONTIGUOUS*)V->data;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(Q,&ldq,NULL));
  CHKERRQ(MatDenseGetArrayRead(Q,&q));
  CHKERRQ(BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,ctx->array+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_TRUE));
  CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDot_Contiguous(BV X,BV Y,Mat M)
{
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data,*y = (BV_CONTIGUOUS*)Y->data;
  PetscScalar    *m;
  PetscInt       ldm;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(M,&ldm,NULL));
  CHKERRQ(MatDenseGetArray(M,&m));
  CHKERRQ(BVDot_BLAS_Private(X,Y->k-Y->l,X->k-X->l,X->n,ldm,y->array+(Y->nc+Y->l)*Y->n,x->array+(X->nc+X->l)*X->n,m+X->l*ldm+Y->l,x->mpi));
  CHKERRQ(MatDenseRestoreArray(M,&m));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDotVec_Contiguous(BV X,Vec y,PetscScalar *q)
{
  BV_CONTIGUOUS     *x = (BV_CONTIGUOUS*)X->data;
  const PetscScalar *py;
  PetscScalar       *qq=q;
  Vec               z = y;

  PetscFunctionBegin;
  if (PetscUnlikely(X->matrix)) {
    CHKERRQ(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  CHKERRQ(VecGetArrayRead(z,&py));
  if (!q) CHKERRQ(VecGetArray(X->buffer,&qq));
  CHKERRQ(BVDotVec_BLAS_Private(X,X->n,X->k-X->l,x->array+(X->nc+X->l)*X->n,py,qq,x->mpi));
  if (!q) CHKERRQ(VecRestoreArray(X->buffer,&qq));
  CHKERRQ(VecRestoreArrayRead(z,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDotVec_Local_Contiguous(BV X,Vec y,PetscScalar *m)
{
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data;
  PetscScalar    *py;
  Vec            z = y;

  PetscFunctionBegin;
  if (PetscUnlikely(X->matrix)) {
    CHKERRQ(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  CHKERRQ(VecGetArray(z,&py));
  CHKERRQ(BVDotVec_BLAS_Private(X,X->n,X->k-X->l,x->array+(X->nc+X->l)*X->n,py,m,PETSC_FALSE));
  CHKERRQ(VecRestoreArray(z,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVScale_Contiguous(BV bv,PetscInt j,PetscScalar alpha)
{
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  if (PetscUnlikely(j<0)) CHKERRQ(BVScale_BLAS_Private(bv,(bv->k-bv->l)*bv->n,ctx->array+(bv->nc+bv->l)*bv->n,alpha));
  else CHKERRQ(BVScale_BLAS_Private(bv,bv->n,ctx->array+(bv->nc+j)*bv->n,alpha));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Contiguous(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  if (PetscUnlikely(j<0)) CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,ctx->array+(bv->nc+bv->l)*bv->n,type,val,ctx->mpi));
  else CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,1,ctx->array+(bv->nc+j)*bv->n,type,val,ctx->mpi));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Local_Contiguous(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  if (PetscUnlikely(j<0)) CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,ctx->array+(bv->nc+bv->l)*bv->n,type,val,PETSC_FALSE));
  else CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,1,ctx->array+(bv->nc+j)*bv->n,type,val,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNormalize_Contiguous(BV bv,PetscScalar *eigi)
{
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;
  PetscScalar    *wi=NULL;

  PetscFunctionBegin;
  if (eigi) wi = eigi+bv->l;
  CHKERRQ(BVNormalize_LAPACK_Private(bv,bv->n,bv->k-bv->l,ctx->array+(bv->nc+bv->l)*bv->n,wi,ctx->mpi));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMatMult_Contiguous(BV V,Mat A,BV W)
{
  BV_CONTIGUOUS  *v = (BV_CONTIGUOUS*)V->data,*w = (BV_CONTIGUOUS*)W->data;
  PetscInt       j;
  Mat            Vmat,Wmat;

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
    for (j=0;j<V->k-V->l;j++) CHKERRQ(MatMult(A,v->V[V->nc+V->l+j],w->V[W->nc+W->l+j]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopy_Contiguous(BV V,BV W)
{
  BV_CONTIGUOUS  *v = (BV_CONTIGUOUS*)V->data,*w = (BV_CONTIGUOUS*)W->data;
  PetscScalar    *pvc,*pwc;

  PetscFunctionBegin;
  pvc = v->array+(V->nc+V->l)*V->n;
  pwc = w->array+(W->nc+W->l)*W->n;
  CHKERRQ(PetscArraycpy(pwc,pvc,(V->k-V->l)*V->n));
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopyColumn_Contiguous(BV V,PetscInt j,PetscInt i)
{
  BV_CONTIGUOUS  *v = (BV_CONTIGUOUS*)V->data;

  PetscFunctionBegin;
  CHKERRQ(PetscArraycpy(v->array+(V->nc+i)*V->n,v->array+(V->nc+j)*V->n,V->n));
  PetscFunctionReturn(0);
}

PetscErrorCode BVResize_Contiguous(BV bv,PetscInt m,PetscBool copy)
{
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;
  PetscInt       j,bs;
  PetscScalar    *newarray;
  Vec            *newV;
  char           str[50];

  PetscFunctionBegin;
  CHKERRQ(VecGetBlockSize(bv->t,&bs));
  CHKERRQ(PetscMalloc1(m*bv->n,&newarray));
  CHKERRQ(PetscArrayzero(newarray,m*bv->n));
  CHKERRQ(PetscMalloc1(m,&newV));
  for (j=0;j<m;j++) {
    if (ctx->mpi) CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,bv->n,PETSC_DECIDE,newarray+j*bv->n,newV+j));
    else CHKERRQ(VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,bv->n,newarray+j*bv->n,newV+j));
  }
  CHKERRQ(PetscLogObjectParents(bv,m,newV));
  if (((PetscObject)bv)->name) {
    for (j=0;j<m;j++) {
      CHKERRQ(PetscSNPrintf(str,sizeof(str),"%s_%" PetscInt_FMT,((PetscObject)bv)->name,j));
      CHKERRQ(PetscObjectSetName((PetscObject)newV[j],str));
    }
  }
  if (copy) CHKERRQ(PetscArraycpy(newarray,ctx->array,PetscMin(m,bv->m)*bv->n));
  CHKERRQ(VecDestroyVecs(bv->m,&ctx->V));
  ctx->V = newV;
  CHKERRQ(PetscFree(ctx->array));
  ctx->array = newarray;
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetColumn_Contiguous(BV bv,PetscInt j,Vec *v)
{
  BV_CONTIGUOUS *ctx = (BV_CONTIGUOUS*)bv->data;
  PetscInt      l;

  PetscFunctionBegin;
  l = BVAvailableVec;
  bv->cv[l] = ctx->V[bv->nc+j];
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetArray_Contiguous(BV bv,PetscScalar **a)
{
  BV_CONTIGUOUS *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  *a = ctx->array;
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetArrayRead_Contiguous(BV bv,const PetscScalar **a)
{
  BV_CONTIGUOUS *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  *a = ctx->array;
  PetscFunctionReturn(0);
}

PetscErrorCode BVDestroy_Contiguous(BV bv)
{
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  if (!bv->issplit) {
    CHKERRQ(VecDestroyVecs(bv->nc+bv->m,&ctx->V));
    CHKERRQ(PetscFree(ctx->array));
  }
  CHKERRQ(PetscFree(bv->data));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode BVCreate_Contiguous(BV bv)
{
  BV_CONTIGUOUS  *ctx;
  PetscInt       j,nloc,bs,lsplit;
  PetscBool      seq;
  PetscScalar    *aa;
  char           str[50];
  PetscScalar    *array;
  BV             parent;
  Vec            *Vpar;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(bv,&ctx));
  bv->data = (void*)ctx;

  CHKERRQ(PetscObjectTypeCompare((PetscObject)bv->t,VECMPI,&ctx->mpi));
  if (!ctx->mpi) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq));
    PetscCheck(seq,PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Cannot create a contiguous BV from a non-standard template vector");
  }

  CHKERRQ(VecGetLocalSize(bv->t,&nloc));
  CHKERRQ(VecGetBlockSize(bv->t,&bs));

  if (PetscUnlikely(bv->issplit)) {
    /* split BV: share memory and Vecs of the parent BV */
    parent = bv->splitparent;
    lsplit = parent->lsplit;
    Vpar   = ((BV_CONTIGUOUS*)parent->data)->V;
    ctx->V = (bv->issplit==1)? Vpar: Vpar+lsplit;
    array  = ((BV_CONTIGUOUS*)parent->data)->array;
    ctx->array = (bv->issplit==1)? array: array+lsplit*nloc;
  } else {
    /* regular BV: allocate memory and Vecs for the BV entries */
    CHKERRQ(PetscCalloc1(bv->m*nloc,&ctx->array));
    CHKERRQ(PetscMalloc1(bv->m,&ctx->V));
    for (j=0;j<bv->m;j++) {
      if (ctx->mpi) CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,ctx->array+j*nloc,ctx->V+j));
      else CHKERRQ(VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,ctx->array+j*nloc,ctx->V+j));
    }
    CHKERRQ(PetscLogObjectParents(bv,bv->m,ctx->V));
  }
  if (((PetscObject)bv)->name) {
    for (j=0;j<bv->m;j++) {
      CHKERRQ(PetscSNPrintf(str,sizeof(str),"%s_%" PetscInt_FMT,((PetscObject)bv)->name,j));
      CHKERRQ(PetscObjectSetName((PetscObject)ctx->V[j],str));
    }
  }

  if (PetscUnlikely(bv->Acreate)) {
    CHKERRQ(MatDenseGetArray(bv->Acreate,&aa));
    CHKERRQ(PetscArraycpy(ctx->array,aa,bv->m*nloc));
    CHKERRQ(MatDenseRestoreArray(bv->Acreate,&aa));
    CHKERRQ(MatDestroy(&bv->Acreate));
  }

  bv->ops->mult             = BVMult_Contiguous;
  bv->ops->multvec          = BVMultVec_Contiguous;
  bv->ops->multinplace      = BVMultInPlace_Contiguous;
  bv->ops->multinplacetrans = BVMultInPlaceHermitianTranspose_Contiguous;
  bv->ops->dot              = BVDot_Contiguous;
  bv->ops->dotvec           = BVDotVec_Contiguous;
  bv->ops->dotvec_local     = BVDotVec_Local_Contiguous;
  bv->ops->scale            = BVScale_Contiguous;
  bv->ops->norm             = BVNorm_Contiguous;
  bv->ops->norm_local       = BVNorm_Local_Contiguous;
  bv->ops->normalize        = BVNormalize_Contiguous;
  bv->ops->matmult          = BVMatMult_Contiguous;
  bv->ops->copy             = BVCopy_Contiguous;
  bv->ops->copycolumn       = BVCopyColumn_Contiguous;
  bv->ops->resize           = BVResize_Contiguous;
  bv->ops->getcolumn        = BVGetColumn_Contiguous;
  bv->ops->getarray         = BVGetArray_Contiguous;
  bv->ops->getarrayread     = BVGetArrayRead_Contiguous;
  bv->ops->destroy          = BVDestroy_Contiguous;
  PetscFunctionReturn(0);
}
