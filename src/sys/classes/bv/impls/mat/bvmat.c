/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
  CHKERRQ(MatDenseGetArrayRead(x->A,&px));
  CHKERRQ(MatDenseGetArray(y->A,&py));
  if (Q) {
    CHKERRQ(MatGetSize(Q,&ldq,NULL));
    CHKERRQ(MatDenseGetArrayRead(Q,&q));
    CHKERRQ(BVMult_BLAS_Private(Y,Y->n,Y->k-Y->l,X->k-X->l,ldq,alpha,px+(X->nc+X->l)*X->n,q+Y->l*ldq+X->l,beta,py+(Y->nc+Y->l)*Y->n));
    CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
  } else CHKERRQ(BVAXPY_BLAS_Private(Y,Y->n,Y->k-Y->l,alpha,px+(X->nc+X->l)*X->n,beta,py+(Y->nc+Y->l)*Y->n));
  CHKERRQ(MatDenseRestoreArrayRead(x->A,&px));
  CHKERRQ(MatDenseRestoreArray(y->A,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultVec_Mat(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  BV_MAT            *x = (BV_MAT*)X->data;
  PetscScalar       *py,*qq=q;
  const PetscScalar *px;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArrayRead(x->A,&px));
  CHKERRQ(VecGetArray(y,&py));
  if (!q) CHKERRQ(VecGetArray(X->buffer,&qq));
  CHKERRQ(BVMultVec_BLAS_Private(X,X->n,X->k-X->l,alpha,px+(X->nc+X->l)*X->n,qq,beta,py));
  if (!q) CHKERRQ(VecRestoreArray(X->buffer,&qq));
  CHKERRQ(MatDenseRestoreArrayRead(x->A,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlace_Mat(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_MAT            *ctx = (BV_MAT*)V->data;
  PetscScalar       *pv;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(Q,&ldq,NULL));
  CHKERRQ(MatDenseGetArray(ctx->A,&pv));
  CHKERRQ(MatDenseGetArrayRead(Q,&q));
  CHKERRQ(BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,pv+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_FALSE));
  CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
  CHKERRQ(MatDenseRestoreArray(ctx->A,&pv));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlaceHermitianTranspose_Mat(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_MAT            *ctx = (BV_MAT*)V->data;
  PetscScalar       *pv;
  const PetscScalar *q;
  PetscInt          ldq;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(Q,&ldq,NULL));
  CHKERRQ(MatDenseGetArray(ctx->A,&pv));
  CHKERRQ(MatDenseGetArrayRead(Q,&q));
  CHKERRQ(BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,pv+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_TRUE));
  CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
  CHKERRQ(MatDenseRestoreArray(ctx->A,&pv));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDot_Mat(BV X,BV Y,Mat M)
{
  BV_MAT            *x = (BV_MAT*)X->data,*y = (BV_MAT*)Y->data;
  PetscScalar       *m;
  const PetscScalar *px,*py;
  PetscInt          ldm;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(M,&ldm,NULL));
  CHKERRQ(MatDenseGetArrayRead(x->A,&px));
  CHKERRQ(MatDenseGetArrayRead(y->A,&py));
  CHKERRQ(MatDenseGetArray(M,&m));
  CHKERRQ(BVDot_BLAS_Private(X,Y->k-Y->l,X->k-X->l,X->n,ldm,py+(Y->nc+Y->l)*Y->n,px+(X->nc+X->l)*X->n,m+X->l*ldm+Y->l,x->mpi));
  CHKERRQ(MatDenseRestoreArray(M,&m));
  CHKERRQ(MatDenseRestoreArrayRead(x->A,&px));
  CHKERRQ(MatDenseRestoreArrayRead(y->A,&py));
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
    CHKERRQ(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  CHKERRQ(MatDenseGetArrayRead(x->A,&px));
  CHKERRQ(VecGetArrayRead(z,&py));
  if (!q) CHKERRQ(VecGetArray(X->buffer,&qq));
  CHKERRQ(BVDotVec_BLAS_Private(X,X->n,X->k-X->l,px+(X->nc+X->l)*X->n,py,qq,x->mpi));
  if (!q) CHKERRQ(VecRestoreArray(X->buffer,&qq));
  CHKERRQ(VecRestoreArrayRead(z,&py));
  CHKERRQ(MatDenseRestoreArrayRead(x->A,&px));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDotVec_Local_Mat(BV X,Vec y,PetscScalar *m)
{
  BV_MAT            *x = (BV_MAT*)X->data;
  const PetscScalar *px,*py;
  Vec               z = y;

  PetscFunctionBegin;
  if (PetscUnlikely(X->matrix)) {
    CHKERRQ(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  CHKERRQ(MatDenseGetArrayRead(x->A,&px));
  CHKERRQ(VecGetArrayRead(z,&py));
  CHKERRQ(BVDotVec_BLAS_Private(X,X->n,X->k-X->l,px+(X->nc+X->l)*X->n,py,m,PETSC_FALSE));
  CHKERRQ(VecRestoreArrayRead(z,&py));
  CHKERRQ(MatDenseRestoreArrayRead(x->A,&px));
  PetscFunctionReturn(0);
}

PetscErrorCode BVScale_Mat(BV bv,PetscInt j,PetscScalar alpha)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArray(ctx->A,&array));
  if (PetscUnlikely(j<0)) CHKERRQ(BVScale_BLAS_Private(bv,(bv->k-bv->l)*bv->n,array+(bv->nc+bv->l)*bv->n,alpha));
  else CHKERRQ(BVScale_BLAS_Private(bv,bv->n,array+(bv->nc+j)*bv->n,alpha));
  CHKERRQ(MatDenseRestoreArray(ctx->A,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Mat(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  BV_MAT            *ctx = (BV_MAT*)bv->data;
  const PetscScalar *array;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArrayRead(ctx->A,&array));
  if (PetscUnlikely(j<0)) CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,type,val,ctx->mpi));
  else CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,1,array+(bv->nc+j)*bv->n,type,val,ctx->mpi));
  CHKERRQ(MatDenseRestoreArrayRead(ctx->A,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Local_Mat(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  BV_MAT            *ctx = (BV_MAT*)bv->data;
  const PetscScalar *array;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArrayRead(ctx->A,&array));
  if (PetscUnlikely(j<0)) CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,type,val,PETSC_FALSE));
  else CHKERRQ(BVNorm_LAPACK_Private(bv,bv->n,1,array+(bv->nc+j)*bv->n,type,val,PETSC_FALSE));
  CHKERRQ(MatDenseRestoreArrayRead(ctx->A,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVNormalize_Mat(BV bv,PetscScalar *eigi)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *array,*wi=NULL;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArray(ctx->A,&array));
  if (eigi) wi = eigi+bv->l;
  CHKERRQ(BVNormalize_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,wi,ctx->mpi));
  CHKERRQ(MatDenseRestoreArray(ctx->A,&array));
  PetscFunctionReturn(0);
}

PetscErrorCode BVMatMult_Mat(BV V,Mat A,BV W)
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

PetscErrorCode BVCopy_Mat(BV V,BV W)
{
  BV_MAT            *v = (BV_MAT*)V->data,*w = (BV_MAT*)W->data;
  PetscScalar       *pw,*pwc;
  const PetscScalar *pv,*pvc;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArrayRead(v->A,&pv));
  CHKERRQ(MatDenseGetArray(w->A,&pw));
  pvc = pv+(V->nc+V->l)*V->n;
  pwc = pw+(W->nc+W->l)*W->n;
  CHKERRQ(PetscArraycpy(pwc,pvc,(V->k-V->l)*V->n));
  CHKERRQ(MatDenseRestoreArrayRead(v->A,&pv));
  CHKERRQ(MatDenseRestoreArray(w->A,&pw));
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopyColumn_Mat(BV V,PetscInt j,PetscInt i)
{
  BV_MAT         *v = (BV_MAT*)V->data;
  PetscScalar    *pv;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArray(v->A,&pv));
  CHKERRQ(PetscArraycpy(pv+(V->nc+i)*V->n,pv+(V->nc+j)*V->n,V->n));
  CHKERRQ(MatDenseRestoreArray(v->A,&pv));
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
  CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)bv->t),bv->n,PETSC_DECIDE,PETSC_DECIDE,m,NULL,&A));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogObjectParent((PetscObject)bv,(PetscObject)A));
  if (((PetscObject)bv)->name) {
    CHKERRQ(PetscSNPrintf(str,sizeof(str),"%s_0",((PetscObject)bv)->name));
    CHKERRQ(PetscObjectSetName((PetscObject)A,str));
  }
  if (copy) {
    CHKERRQ(MatDenseGetArrayRead(ctx->A,&pA));
    CHKERRQ(MatDenseGetArrayWrite(A,&pnew));
    CHKERRQ(PetscArraycpy(pnew,pA,PetscMin(m,bv->m)*bv->n));
    CHKERRQ(MatDenseRestoreArrayRead(ctx->A,&pA));
    CHKERRQ(MatDenseRestoreArrayWrite(A,&pnew));
  }
  CHKERRQ(MatDestroy(&ctx->A));
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
  CHKERRQ(MatDenseGetArray(ctx->A,&pA));
  CHKERRQ(VecPlaceArray(bv->cv[l],pA+(bv->nc+j)*bv->n));
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreColumn_Mat(BV bv,PetscInt j,Vec *v)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *pA;
  PetscInt       l;

  PetscFunctionBegin;
  l = (j==bv->ci[0])? 0: 1;
  CHKERRQ(VecResetArray(bv->cv[l]));
  CHKERRQ(MatDenseRestoreArray(ctx->A,&pA));
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetArray_Mat(BV bv,PetscScalar **a)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArray(ctx->A,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreArray_Mat(BV bv,PetscScalar **a)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  if (a) CHKERRQ(MatDenseRestoreArray(ctx->A,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetArrayRead_Mat(BV bv,const PetscScalar **a)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArrayRead(ctx->A,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreArrayRead_Mat(BV bv,const PetscScalar **a)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  if (a) CHKERRQ(MatDenseRestoreArrayRead(ctx->A,a));
  PetscFunctionReturn(0);
}

PetscErrorCode BVView_Mat(BV bv,PetscViewer viewer)
{
  BV_MAT            *ctx = (BV_MAT*)bv->data;
  PetscViewerFormat format;
  PetscBool         isascii;
  const char        *bvname,*name;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
    CHKERRQ(MatView(ctx->A,viewer));
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      CHKERRQ(PetscObjectGetName((PetscObject)bv,&bvname));
      CHKERRQ(PetscObjectGetName((PetscObject)ctx->A,&name));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s=%s;clear %s\n",bvname,name,name));
      if (bv->nc) CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s=%s(:,%" PetscInt_FMT ":end);\n",bvname,bvname,bv->nc+1));
    }
  } else CHKERRQ(MatView(ctx->A,viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode BVDestroy_Mat(BV bv)
{
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&ctx->A));
  CHKERRQ(VecDestroy(&bv->cv[0]));
  CHKERRQ(VecDestroy(&bv->cv[1]));
  CHKERRQ(PetscFree(bv->data));
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
  CHKERRQ(PetscNewLog(bv,&ctx));
  bv->data = (void*)ctx;

  CHKERRQ(PetscObjectTypeCompare((PetscObject)bv->t,VECMPI,&ctx->mpi));
  if (!ctx->mpi) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq));
    PetscCheck(seq,PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Cannot create a BVMAT from a non-standard template vector");
  }

  CHKERRQ(VecGetLocalSize(bv->t,&nloc));
  CHKERRQ(VecGetBlockSize(bv->t,&bs));

  if (PetscUnlikely(bv->issplit)) {
    /* split BV: share the memory of the parent BV */
    parent = bv->splitparent;
    lsplit = parent->lsplit;
    CHKERRQ(MatDenseGetArray(((BV_MAT*)parent->data)->A,&array));
    ptr = (bv->issplit==1)? array: array+lsplit*nloc;
    CHKERRQ(MatDenseRestoreArray(((BV_MAT*)parent->data)->A,&array));
  } else {
    /* regular BV: allocate memory for the BV entries */
    ptr = NULL;
  }
  CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)bv->t),nloc,PETSC_DECIDE,PETSC_DECIDE,bv->m,ptr,&ctx->A));
  CHKERRQ(MatAssemblyBegin(ctx->A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(ctx->A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogObjectParent((PetscObject)bv,(PetscObject)ctx->A));
  if (((PetscObject)bv)->name) {
    CHKERRQ(PetscSNPrintf(str,sizeof(str),"%s_0",((PetscObject)bv)->name));
    CHKERRQ(PetscObjectSetName((PetscObject)ctx->A,str));
  }

  if (PetscUnlikely(bv->Acreate)) {
    CHKERRQ(MatCopy(bv->Acreate,ctx->A,SAME_NONZERO_PATTERN));
    CHKERRQ(MatDestroy(&bv->Acreate));
  }

  CHKERRQ(VecDuplicateEmpty(bv->t,&bv->cv[0]));
  CHKERRQ(VecDuplicateEmpty(bv->t,&bv->cv[1]));

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
  bv->ops->destroy          = BVDestroy_Mat;
  if (!ctx->mpi) bv->ops->view = BVView_Mat;
  PetscFunctionReturn(0);
}
