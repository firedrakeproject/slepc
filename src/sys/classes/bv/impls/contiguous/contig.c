/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
    PetscCall(MatDenseGetLDA(Q,&ldq));
    PetscCall(MatDenseGetArrayRead(Q,&q));
    PetscCall(BVMult_BLAS_Private(Y,Y->n,Y->k-Y->l,X->k-X->l,ldq,alpha,x->array+(X->nc+X->l)*X->n,q+Y->l*ldq+X->l,beta,y->array+(Y->nc+Y->l)*Y->n));
    PetscCall(MatDenseRestoreArrayRead(Q,&q));
  } else PetscCall(BVAXPY_BLAS_Private(Y,Y->n,Y->k-Y->l,alpha,x->array+(X->nc+X->l)*X->n,beta,y->array+(Y->nc+Y->l)*Y->n));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultVec_Contiguous(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data;
  PetscScalar    *py,*qq=q;

  PetscFunctionBegin;
  PetscCall(VecGetArray(y,&py));
  if (!q) PetscCall(VecGetArray(X->buffer,&qq));
  PetscCall(BVMultVec_BLAS_Private(X,X->n,X->k-X->l,alpha,x->array+(X->nc+X->l)*X->n,qq,beta,py));
  if (!q) PetscCall(VecRestoreArray(X->buffer,&qq));
  PetscCall(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlace_Contiguous(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_CONTIGUOUS     *ctx = (BV_CONTIGUOUS*)V->data;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  PetscCall(MatDenseGetLDA(Q,&ldq));
  PetscCall(MatDenseGetArrayRead(Q,&q));
  PetscCall(BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,ctx->array+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_FALSE));
  PetscCall(MatDenseRestoreArrayRead(Q,&q));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlaceHermitianTranspose_Contiguous(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_CONTIGUOUS     *ctx = (BV_CONTIGUOUS*)V->data;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  PetscCall(MatDenseGetLDA(Q,&ldq));
  PetscCall(MatDenseGetArrayRead(Q,&q));
  PetscCall(BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,ctx->array+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_TRUE));
  PetscCall(MatDenseRestoreArrayRead(Q,&q));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDot_Contiguous(BV X,BV Y,Mat M)
{
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data,*y = (BV_CONTIGUOUS*)Y->data;
  PetscScalar    *m;
  PetscInt       ldm;

  PetscFunctionBegin;
  PetscCall(MatDenseGetLDA(M,&ldm));
  PetscCall(MatDenseGetArray(M,&m));
  PetscCall(BVDot_BLAS_Private(X,Y->k-Y->l,X->k-X->l,X->n,ldm,y->array+(Y->nc+Y->l)*Y->n,x->array+(X->nc+X->l)*X->n,m+X->l*ldm+Y->l,x->mpi));
  PetscCall(MatDenseRestoreArray(M,&m));
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
    PetscCall(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  PetscCall(VecGetArrayRead(z,&py));
  if (!q) PetscCall(VecGetArray(X->buffer,&qq));
  PetscCall(BVDotVec_BLAS_Private(X,X->n,X->k-X->l,x->array+(X->nc+X->l)*X->n,py,qq,x->mpi));
  if (!q) PetscCall(VecRestoreArray(X->buffer,&qq));
  PetscCall(VecRestoreArrayRead(z,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDotVec_Local_Contiguous(BV X,Vec y,PetscScalar *m)
{
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data;
  PetscScalar    *py;
  Vec            z = y;

  PetscFunctionBegin;
  if (PetscUnlikely(X->matrix)) {
    PetscCall(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  PetscCall(VecGetArray(z,&py));
  PetscCall(BVDotVec_BLAS_Private(X,X->n,X->k-X->l,x->array+(X->nc+X->l)*X->n,py,m,PETSC_FALSE));
  PetscCall(VecRestoreArray(z,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVScale_Contiguous(BV bv,PetscInt j,PetscScalar alpha)
{
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  if (PetscUnlikely(j<0)) PetscCall(BVScale_BLAS_Private(bv,(bv->k-bv->l)*bv->n,ctx->array+(bv->nc+bv->l)*bv->n,alpha));
  else PetscCall(BVScale_BLAS_Private(bv,bv->n,ctx->array+(bv->nc+j)*bv->n,alpha));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Contiguous(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  if (PetscUnlikely(j<0)) PetscCall(BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,ctx->array+(bv->nc+bv->l)*bv->n,type,val,ctx->mpi));
  else PetscCall(BVNorm_LAPACK_Private(bv,bv->n,1,ctx->array+(bv->nc+j)*bv->n,type,val,ctx->mpi));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Local_Contiguous(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  if (PetscUnlikely(j<0)) PetscCall(BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,ctx->array+(bv->nc+bv->l)*bv->n,type,val,PETSC_FALSE));
  else PetscCall(BVNorm_LAPACK_Private(bv,bv->n,1,ctx->array+(bv->nc+j)*bv->n,type,val,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNormalize_Contiguous(BV bv,PetscScalar *eigi)
{
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;
  PetscScalar    *wi=NULL;

  PetscFunctionBegin;
  if (eigi) wi = eigi+bv->l;
  PetscCall(BVNormalize_LAPACK_Private(bv,bv->n,bv->k-bv->l,ctx->array+(bv->nc+bv->l)*bv->n,wi,ctx->mpi));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMatMult_Contiguous(BV V,Mat A,BV W)
{
  BV_CONTIGUOUS  *v = (BV_CONTIGUOUS*)V->data,*w = (BV_CONTIGUOUS*)W->data;
  PetscInt       j;
  Mat            Vmat,Wmat;

  PetscFunctionBegin;
  if (V->vmm) {
    PetscCall(BVGetMat(V,&Vmat));
    PetscCall(BVGetMat(W,&Wmat));
    PetscCall(MatProductCreateWithMat(A,Vmat,NULL,Wmat));
    PetscCall(MatProductSetType(Wmat,MATPRODUCT_AB));
    PetscCall(MatProductSetFromOptions(Wmat));
    PetscCall(MatProductSymbolic(Wmat));
    PetscCall(MatProductNumeric(Wmat));
    PetscCall(MatProductClear(Wmat));
    PetscCall(BVRestoreMat(V,&Vmat));
    PetscCall(BVRestoreMat(W,&Wmat));
  } else {
    for (j=0;j<V->k-V->l;j++) PetscCall(MatMult(A,v->V[V->nc+V->l+j],w->V[W->nc+W->l+j]));
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
  PetscCall(PetscArraycpy(pwc,pvc,(V->k-V->l)*V->n));
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopyColumn_Contiguous(BV V,PetscInt j,PetscInt i)
{
  BV_CONTIGUOUS  *v = (BV_CONTIGUOUS*)V->data;

  PetscFunctionBegin;
  PetscCall(PetscArraycpy(v->array+(V->nc+i)*V->n,v->array+(V->nc+j)*V->n,V->n));
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
  PetscCall(VecGetBlockSize(bv->t,&bs));
  PetscCall(PetscMalloc1(m*bv->n,&newarray));
  PetscCall(PetscArrayzero(newarray,m*bv->n));
  PetscCall(PetscMalloc1(m,&newV));
  for (j=0;j<m;j++) {
    if (ctx->mpi) PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,bv->n,PETSC_DECIDE,newarray+j*bv->n,newV+j));
    else PetscCall(VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,bv->n,newarray+j*bv->n,newV+j));
  }
  if (((PetscObject)bv)->name) {
    for (j=0;j<m;j++) {
      PetscCall(PetscSNPrintf(str,sizeof(str),"%s_%" PetscInt_FMT,((PetscObject)bv)->name,j));
      PetscCall(PetscObjectSetName((PetscObject)newV[j],str));
    }
  }
  if (copy) PetscCall(PetscArraycpy(newarray,ctx->array,PetscMin(m,bv->m)*bv->n));
  PetscCall(VecDestroyVecs(bv->m,&ctx->V));
  ctx->V = newV;
  PetscCall(PetscFree(ctx->array));
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

PetscErrorCode BVRestoreColumn_Contiguous(BV bv,PetscInt j,Vec *v)
{
  PetscInt l;

  PetscFunctionBegin;
  l = (j==bv->ci[0])? 0: 1;
  bv->cv[l] = NULL;
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
    PetscCall(VecDestroyVecs(bv->nc+bv->m,&ctx->V));
    PetscCall(PetscFree(ctx->array));
  }
  PetscCall(PetscFree(bv->data));
  PetscFunctionReturn(0);
}

PetscErrorCode BVView_Contiguous(BV bv,PetscViewer viewer)
{
  PetscInt          j;
  Vec               v;
  PetscViewerFormat format;
  PetscBool         isascii,ismatlab=PETSC_FALSE;
  const char        *bvname,*name;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
    if (format == PETSC_VIEWER_ASCII_MATLAB) ismatlab = PETSC_TRUE;
  }
  if (ismatlab) {
    PetscCall(PetscObjectGetName((PetscObject)bv,&bvname));
    PetscCall(PetscViewerASCIIPrintf(viewer,"%s=[];\n",bvname));
  }
  for (j=0;j<bv->m;j++) {
    PetscCall(BVGetColumn(bv,j,&v));
    PetscCall(VecView(v,viewer));
    if (ismatlab) {
      PetscCall(PetscObjectGetName((PetscObject)v,&name));
      PetscCall(PetscViewerASCIIPrintf(viewer,"%s=[%s,%s];clear %s\n",bvname,bvname,name,name));
    }
    PetscCall(BVRestoreColumn(bv,j,&v));
  }
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
  PetscCall(PetscNew(&ctx));
  bv->data = (void*)ctx;

  PetscCall(PetscObjectTypeCompare((PetscObject)bv->t,VECMPI,&ctx->mpi));
  if (!ctx->mpi) {
    PetscCall(PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq));
    PetscCheck(seq,PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Cannot create a contiguous BV from a non-standard template vector");
  }

  PetscCall(VecGetLocalSize(bv->t,&nloc));
  PetscCall(VecGetBlockSize(bv->t,&bs));

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
    PetscCall(PetscCalloc1(bv->m*nloc,&ctx->array));
    PetscCall(PetscMalloc1(bv->m,&ctx->V));
    for (j=0;j<bv->m;j++) {
      if (ctx->mpi) PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,ctx->array+j*nloc,ctx->V+j));
      else PetscCall(VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,ctx->array+j*nloc,ctx->V+j));
    }
  }
  if (((PetscObject)bv)->name) {
    for (j=0;j<bv->m;j++) {
      PetscCall(PetscSNPrintf(str,sizeof(str),"%s_%" PetscInt_FMT,((PetscObject)bv)->name,j));
      PetscCall(PetscObjectSetName((PetscObject)ctx->V[j],str));
    }
  }

  if (PetscUnlikely(bv->Acreate)) {
    PetscCall(MatDenseGetArray(bv->Acreate,&aa));
    PetscCall(PetscArraycpy(ctx->array,aa,bv->m*nloc));
    PetscCall(MatDenseRestoreArray(bv->Acreate,&aa));
    PetscCall(MatDestroy(&bv->Acreate));
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
  bv->ops->restorecolumn    = BVRestoreColumn_Contiguous;
  bv->ops->getarray         = BVGetArray_Contiguous;
  bv->ops->getarrayread     = BVGetArrayRead_Contiguous;
  bv->ops->getmat           = BVGetMat_Default;
  bv->ops->restoremat       = BVRestoreMat_Default;
  bv->ops->destroy          = BVDestroy_Contiguous;
  bv->ops->view             = BVView_Contiguous;
  PetscFunctionReturn(0);
}
