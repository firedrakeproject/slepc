/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV implemented with a dense Mat
*/

#include <slepc/private/bvimpl.h>

typedef struct {
  Mat       A;
  PetscBool mpi;
} BV_MAT;

PetscErrorCode BVMult_Mat(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  BV_MAT            *y = (BV_MAT*)Y->data,*x = (BV_MAT*)X->data;
  PetscScalar       *py;
  const PetscScalar *px,*q;
  PetscInt          ldq;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(x->A,&px));
  PetscCall(MatDenseGetArray(y->A,&py));
  if (Q) {
    PetscCall(MatDenseGetLDA(Q,&ldq));
    PetscCall(MatDenseGetArrayRead(Q,&q));
    PetscCall(BVMult_BLAS_Private(Y,Y->n,Y->k-Y->l,X->k-X->l,ldq,alpha,px+(X->nc+X->l)*X->n,q+Y->l*ldq+X->l,beta,py+(Y->nc+Y->l)*Y->n));
    PetscCall(MatDenseRestoreArrayRead(Q,&q));
  } else PetscCall(BVAXPY_BLAS_Private(Y,Y->n,Y->k-Y->l,alpha,px+(X->nc+X->l)*X->n,beta,py+(Y->nc+Y->l)*Y->n));
  PetscCall(MatDenseRestoreArrayRead(x->A,&px));
  PetscCall(MatDenseRestoreArray(y->A,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultVec_Mat(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  BV_MAT            *x = (BV_MAT*)X->data;
  PetscScalar       *py,*qq=q;
  const PetscScalar *px;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(x->A,&px));
  PetscCall(VecGetArray(y,&py));
  if (!q) PetscCall(VecGetArray(X->buffer,&qq));
  PetscCall(BVMultVec_BLAS_Private(X,X->n,X->k-X->l,alpha,px+(X->nc+X->l)*X->n,qq,beta,py));
  if (!q) PetscCall(VecRestoreArray(X->buffer,&qq));
  PetscCall(MatDenseRestoreArrayRead(x->A,&px));
  PetscCall(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlace_Mat(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_MAT            *ctx = (BV_MAT*)V->data;
  PetscScalar       *pv;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  PetscCall(MatDenseGetLDA(Q,&ldq));
  PetscCall(MatDenseGetArray(ctx->A,&pv));
  PetscCall(MatDenseGetArrayRead(Q,&q));
  PetscCall(BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,pv+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_FALSE));
  PetscCall(MatDenseRestoreArrayRead(Q,&q));
  PetscCall(MatDenseRestoreArray(ctx->A,&pv));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlaceHermitianTranspose_Mat(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_MAT            *ctx = (BV_MAT*)V->data;
  PetscScalar       *pv;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  PetscCall(MatDenseGetLDA(Q,&ldq));
  PetscCall(MatDenseGetArray(ctx->A,&pv));
  PetscCall(MatDenseGetArrayRead(Q,&q));
  PetscCall(BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,pv+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_TRUE));
  PetscCall(MatDenseRestoreArrayRead(Q,&q));
  PetscCall(MatDenseRestoreArray(ctx->A,&pv));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDot_Mat(BV X,BV Y,Mat M)
{
  BV_MAT            *x = (BV_MAT*)X->data,*y = (BV_MAT*)Y->data;
  PetscScalar       *m;
  const PetscScalar *px,*py;
  PetscInt          ldm;

  PetscFunctionBegin;
  PetscCall(MatDenseGetLDA(M,&ldm));
  PetscCall(MatDenseGetArrayRead(x->A,&px));
  PetscCall(MatDenseGetArrayRead(y->A,&py));
  PetscCall(MatDenseGetArray(M,&m));
  PetscCall(BVDot_BLAS_Private(X,Y->k-Y->l,X->k-X->l,X->n,ldm,py+(Y->nc+Y->l)*Y->n,px+(X->nc+X->l)*X->n,m+X->l*ldm+Y->l,x->mpi));
  PetscCall(MatDenseRestoreArray(M,&m));
  PetscCall(MatDenseRestoreArrayRead(x->A,&px));
  PetscCall(MatDenseRestoreArrayRead(y->A,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDotVec_Mat(BV X,Vec y,PetscScalar *q)
{
  BV_MAT            *x = (BV_MAT*)X->data;
  PetscScalar       *qq=q;
  const PetscScalar *px,*py;
  Vec               z = y;

  PetscFunctionBegin;
  if (PetscUnlikely(X->matrix)) {
    PetscCall(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  PetscCall(MatDenseGetArrayRead(x->A,&px));
  PetscCall(VecGetArrayRead(z,&py));
  if (!q) PetscCall(VecGetArray(X->buffer,&qq));
  PetscCall(BVDotVec_BLAS_Private(X,X->n,X->k-X->l,px+(X->nc+X->l)*X->n,py,qq,x->mpi));
  if (!q) PetscCall(VecRestoreArray(X->buffer,&qq));
  PetscCall(VecRestoreArrayRead(z,&py));
  PetscCall(MatDenseRestoreArrayRead(x->A,&px));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDotVec_Local_Mat(BV X,Vec y,PetscScalar *m)
{
  BV_MAT            *x = (BV_MAT*)X->data;
  const PetscScalar *px,*py;
  Vec               z = y;

  PetscFunctionBegin;
  if (PetscUnlikely(X->matrix)) {
    PetscCall(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  PetscCall(MatDenseGetArrayRead(x->A,&px));
  PetscCall(VecGetArrayRead(z,&py));
  PetscCall(BVDotVec_BLAS_Private(X,X->n,X->k-X->l,px+(X->nc+X->l)*X->n,py,m,PETSC_FALSE));
  PetscCall(VecRestoreArrayRead(z,&py));
  PetscCall(MatDenseRestoreArrayRead(x->A,&px));
  PetscFunctionReturn(0);
}

PetscErrorCode BVScale_Mat(BV bv,PetscInt j,PetscScalar alpha)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(ctx->A,&array));
  if (PetscUnlikely(j<0)) PetscCall(BVScale_BLAS_Private(bv,(bv->k-bv->l)*bv->n,array+(bv->nc+bv->l)*bv->n,alpha));
  else PetscCall(BVScale_BLAS_Private(bv,bv->n,array+(bv->nc+j)*bv->n,alpha));
  PetscCall(MatDenseRestoreArray(ctx->A,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Mat(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  BV_MAT            *ctx = (BV_MAT*)bv->data;
  const PetscScalar *array;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(ctx->A,&array));
  if (PetscUnlikely(j<0)) PetscCall(BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,type,val,ctx->mpi));
  else PetscCall(BVNorm_LAPACK_Private(bv,bv->n,1,array+(bv->nc+j)*bv->n,type,val,ctx->mpi));
  PetscCall(MatDenseRestoreArrayRead(ctx->A,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Local_Mat(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  BV_MAT            *ctx = (BV_MAT*)bv->data;
  const PetscScalar *array;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(ctx->A,&array));
  if (PetscUnlikely(j<0)) PetscCall(BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,type,val,PETSC_FALSE));
  else PetscCall(BVNorm_LAPACK_Private(bv,bv->n,1,array+(bv->nc+j)*bv->n,type,val,PETSC_FALSE));
  PetscCall(MatDenseRestoreArrayRead(ctx->A,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNormalize_Mat(BV bv,PetscScalar *eigi)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *array,*wi=NULL;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(ctx->A,&array));
  if (eigi) wi = eigi+bv->l;
  PetscCall(BVNormalize_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,wi,ctx->mpi));
  PetscCall(MatDenseRestoreArray(ctx->A,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMatMult_Mat(BV V,Mat A,BV W)
{
  PetscInt       j;
  Mat            Vmat,Wmat;
  Vec            vv,ww;

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
    for (j=0;j<V->k-V->l;j++) {
      PetscCall(BVGetColumn(V,V->l+j,&vv));
      PetscCall(BVGetColumn(W,W->l+j,&ww));
      PetscCall(MatMult(A,vv,ww));
      PetscCall(BVRestoreColumn(V,V->l+j,&vv));
      PetscCall(BVRestoreColumn(W,W->l+j,&ww));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopy_Mat(BV V,BV W)
{
  BV_MAT            *v = (BV_MAT*)V->data,*w = (BV_MAT*)W->data;
  PetscScalar       *pw,*pwc;
  const PetscScalar *pv,*pvc;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(v->A,&pv));
  PetscCall(MatDenseGetArray(w->A,&pw));
  pvc = pv+(V->nc+V->l)*V->n;
  pwc = pw+(W->nc+W->l)*W->n;
  PetscCall(PetscArraycpy(pwc,pvc,(V->k-V->l)*V->n));
  PetscCall(MatDenseRestoreArrayRead(v->A,&pv));
  PetscCall(MatDenseRestoreArray(w->A,&pw));
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopyColumn_Mat(BV V,PetscInt j,PetscInt i)
{
  BV_MAT         *v = (BV_MAT*)V->data;
  PetscScalar    *pv;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(v->A,&pv));
  PetscCall(PetscArraycpy(pv+(V->nc+i)*V->n,pv+(V->nc+j)*V->n,V->n));
  PetscCall(MatDenseRestoreArray(v->A,&pv));
  PetscFunctionReturn(0);
}

PetscErrorCode BVResize_Mat(BV bv,PetscInt m,PetscBool copy)
{
  BV_MAT            *ctx = (BV_MAT*)bv->data;
  PetscScalar       *pnew;
  const PetscScalar *pA;
  Mat               A;
  char              str[50];

  PetscFunctionBegin;
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)bv->t),bv->n,PETSC_DECIDE,PETSC_DECIDE,m,NULL,&A));
  if (((PetscObject)bv)->name) {
    PetscCall(PetscSNPrintf(str,sizeof(str),"%s_0",((PetscObject)bv)->name));
    PetscCall(PetscObjectSetName((PetscObject)A,str));
  }
  if (copy) {
    PetscCall(MatDenseGetArrayRead(ctx->A,&pA));
    PetscCall(MatDenseGetArrayWrite(A,&pnew));
    PetscCall(PetscArraycpy(pnew,pA,PetscMin(m,bv->m)*bv->n));
    PetscCall(MatDenseRestoreArrayRead(ctx->A,&pA));
    PetscCall(MatDenseRestoreArrayWrite(A,&pnew));
  }
  PetscCall(MatDestroy(&ctx->A));
  ctx->A = A;
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetColumn_Mat(BV bv,PetscInt j,Vec *v)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *pA;
  PetscInt       l;

  PetscFunctionBegin;
  l = BVAvailableVec;
  PetscCall(MatDenseGetArray(ctx->A,&pA));
  PetscCall(VecPlaceArray(bv->cv[l],pA+(bv->nc+j)*bv->n));
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreColumn_Mat(BV bv,PetscInt j,Vec *v)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *pA;
  PetscInt       l;

  PetscFunctionBegin;
  l = (j==bv->ci[0])? 0: 1;
  PetscCall(VecResetArray(bv->cv[l]));
  PetscCall(MatDenseRestoreArray(ctx->A,&pA));
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetArray_Mat(BV bv,PetscScalar **a)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(ctx->A,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreArray_Mat(BV bv,PetscScalar **a)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  if (a) PetscCall(MatDenseRestoreArray(ctx->A,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetArrayRead_Mat(BV bv,const PetscScalar **a)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(ctx->A,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreArrayRead_Mat(BV bv,const PetscScalar **a)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  if (a) PetscCall(MatDenseRestoreArrayRead(ctx->A,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVView_Mat(BV bv,PetscViewer viewer)
{
  BV_MAT            *ctx = (BV_MAT*)bv->data;
  PetscViewerFormat format;
  PetscBool         isascii;
  const char        *bvname,*name;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
    PetscCall(MatView(ctx->A,viewer));
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      PetscCall(PetscObjectGetName((PetscObject)bv,&bvname));
      PetscCall(PetscObjectGetName((PetscObject)ctx->A,&name));
      PetscCall(PetscViewerASCIIPrintf(viewer,"%s=%s;clear %s\n",bvname,name,name));
      if (bv->nc) PetscCall(PetscViewerASCIIPrintf(viewer,"%s=%s(:,%" PetscInt_FMT ":end);\n",bvname,bvname,bv->nc+1));
    }
  } else PetscCall(MatView(ctx->A,viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDestroy_Mat(BV bv)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&ctx->A));
  PetscCall(VecDestroy(&bv->cv[0]));
  PetscCall(VecDestroy(&bv->cv[1]));
  PetscCall(PetscFree(bv->data));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode BVCreate_Mat(BV bv)
{
  BV_MAT         *ctx;
  PetscInt       nloc,bs,lsplit;
  PetscBool      seq;
  char           str[50];
  PetscScalar    *array,*ptr;
  BV             parent;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  bv->data = (void*)ctx;

  PetscCall(PetscObjectTypeCompare((PetscObject)bv->t,VECMPI,&ctx->mpi));
  if (!ctx->mpi) {
    PetscCall(PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq));
    PetscCheck(seq,PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Cannot create a BVMAT from a non-standard template vector");
  }

  PetscCall(VecGetLocalSize(bv->t,&nloc));
  PetscCall(VecGetBlockSize(bv->t,&bs));

  if (PetscUnlikely(bv->issplit)) {
    /* split BV: share the memory of the parent BV */
    parent = bv->splitparent;
    lsplit = parent->lsplit;
    PetscCall(MatDenseGetArray(((BV_MAT*)parent->data)->A,&array));
    ptr = (bv->issplit==1)? array: array+lsplit*nloc;
    PetscCall(MatDenseRestoreArray(((BV_MAT*)parent->data)->A,&array));
  } else {
    /* regular BV: allocate memory for the BV entries */
    ptr = NULL;
  }
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)bv->t),nloc,PETSC_DECIDE,PETSC_DECIDE,bv->m,ptr,&ctx->A));
  if (((PetscObject)bv)->name) {
    PetscCall(PetscSNPrintf(str,sizeof(str),"%s_0",((PetscObject)bv)->name));
    PetscCall(PetscObjectSetName((PetscObject)ctx->A,str));
  }

  if (PetscUnlikely(bv->Acreate)) {
    PetscCall(MatCopy(bv->Acreate,ctx->A,SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&bv->Acreate));
  }

  PetscCall(VecDuplicateEmpty(bv->t,&bv->cv[0]));
  PetscCall(VecDuplicateEmpty(bv->t,&bv->cv[1]));

  bv->ops->mult             = BVMult_Mat;
  bv->ops->multvec          = BVMultVec_Mat;
  bv->ops->multinplace      = BVMultInPlace_Mat;
  bv->ops->multinplacetrans = BVMultInPlaceHermitianTranspose_Mat;
  bv->ops->dot              = BVDot_Mat;
  bv->ops->dotvec           = BVDotVec_Mat;
  bv->ops->dotvec_local     = BVDotVec_Local_Mat;
  bv->ops->scale            = BVScale_Mat;
  bv->ops->norm             = BVNorm_Mat;
  bv->ops->norm_local       = BVNorm_Local_Mat;
  bv->ops->normalize        = BVNormalize_Mat;
  bv->ops->matmult          = BVMatMult_Mat;
  bv->ops->copy             = BVCopy_Mat;
  bv->ops->copycolumn       = BVCopyColumn_Mat;
  bv->ops->resize           = BVResize_Mat;
  bv->ops->getcolumn        = BVGetColumn_Mat;
  bv->ops->restorecolumn    = BVRestoreColumn_Mat;
  bv->ops->getarray         = BVGetArray_Mat;
  bv->ops->restorearray     = BVRestoreArray_Mat;
  bv->ops->getarrayread     = BVGetArrayRead_Mat;
  bv->ops->restorearrayread = BVRestoreArrayRead_Mat;
  bv->ops->getmat           = BVGetMat_Default;
  bv->ops->restoremat       = BVRestoreMat_Default;
  bv->ops->destroy          = BVDestroy_Mat;
  if (!ctx->mpi) bv->ops->view = BVView_Mat;
  PetscFunctionReturn(0);
}
