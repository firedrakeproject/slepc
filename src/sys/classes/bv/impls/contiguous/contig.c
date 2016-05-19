/*
   BV implemented as an array of Vecs sharing a contiguous array for elements

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/bvimpl.h>

typedef struct {
  Vec         *V;
  PetscScalar *array;
  PetscBool   mpi;
} BV_CONTIGUOUS;

#undef __FUNCT__
#define __FUNCT__ "BVMult_Contiguous"
PetscErrorCode BVMult_Contiguous(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *y = (BV_CONTIGUOUS*)Y->data,*x = (BV_CONTIGUOUS*)X->data;
  PetscScalar    *q;
  PetscInt       ldq;

  PetscFunctionBegin;
  if (Q) {
    ierr = MatGetSize(Q,&ldq,NULL);CHKERRQ(ierr);
    ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
    ierr = BVMult_BLAS_Private(Y,Y->n,Y->k-Y->l,X->k-X->l,ldq,alpha,x->array+(X->nc+X->l)*X->n,q+Y->l*ldq+X->l,beta,y->array+(Y->nc+Y->l)*Y->n);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  } else {
    ierr = BVAXPY_BLAS_Private(Y,Y->n,Y->k-Y->l,alpha,x->array+(X->nc+X->l)*X->n,beta,y->array+(Y->nc+Y->l)*Y->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultVec_Contiguous"
PetscErrorCode BVMultVec_Contiguous(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data;
  PetscScalar    *py;

  PetscFunctionBegin;
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = BVMultVec_BLAS_Private(X,X->n,X->k-X->l,alpha,x->array+(X->nc+X->l)*X->n,q,beta,py);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultInPlace_Contiguous"
PetscErrorCode BVMultInPlace_Contiguous(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)V->data;
  PetscScalar    *q;
  PetscInt       ldq;

  PetscFunctionBegin;
  ierr = MatGetSize(Q,&ldq,NULL);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  ierr = BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,ctx->array+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultInPlaceTranspose_Contiguous"
PetscErrorCode BVMultInPlaceTranspose_Contiguous(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)V->data;
  PetscScalar    *q;
  PetscInt       ldq;

  PetscFunctionBegin;
  ierr = MatGetSize(Q,&ldq,NULL);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  ierr = BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,ctx->array+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDot_Contiguous"
PetscErrorCode BVDot_Contiguous(BV X,BV Y,Mat M)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data,*y = (BV_CONTIGUOUS*)Y->data;
  PetscScalar    *m;
  PetscInt       ldm;

  PetscFunctionBegin;
  ierr = MatGetSize(M,&ldm,NULL);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M,&m);CHKERRQ(ierr);
  ierr = BVDot_BLAS_Private(X,Y->k-Y->l,X->k-X->l,X->n,ldm,y->array+(Y->nc+Y->l)*Y->n,x->array+(X->nc+X->l)*X->n,m+X->l*ldm+Y->l,x->mpi);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(M,&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec_Contiguous"
PetscErrorCode BVDotVec_Contiguous(BV X,Vec y,PetscScalar *m)
{
  PetscErrorCode    ierr;
  BV_CONTIGUOUS     *x = (BV_CONTIGUOUS*)X->data;
  const PetscScalar *py;
  Vec               z = y;

  PetscFunctionBegin;
  if (X->matrix) {
    ierr = BV_IPMatMult(X,y);CHKERRQ(ierr);
    z = X->Bx;
  }
  ierr = VecGetArrayRead(z,&py);CHKERRQ(ierr);
  ierr = BVDotVec_BLAS_Private(X,X->n,X->k-X->l,x->array+(X->nc+X->l)*X->n,py,m,x->mpi);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(z,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec_Local_Contiguous"
PetscErrorCode BVDotVec_Local_Contiguous(BV X,Vec y,PetscScalar *m)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data;
  PetscScalar    *py;
  Vec            z = y;

  PetscFunctionBegin;
  if (X->matrix) {
    ierr = BV_IPMatMult(X,y);CHKERRQ(ierr);
    z = X->Bx;
  }
  ierr = VecGetArray(z,&py);CHKERRQ(ierr);
  ierr = BVDotVec_BLAS_Private(X,X->n,X->k-X->l,x->array+(X->nc+X->l)*X->n,py,m,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecRestoreArray(z,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVScale_Contiguous"
PetscErrorCode BVScale_Contiguous(BV bv,PetscInt j,PetscScalar alpha)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  if (j<0) {
    ierr = BVScale_BLAS_Private(bv,(bv->k-bv->l)*bv->n,ctx->array+(bv->nc+bv->l)*bv->n,alpha);CHKERRQ(ierr);
  } else {
    ierr = BVScale_BLAS_Private(bv,bv->n,ctx->array+(bv->nc+j)*bv->n,alpha);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVNorm_Contiguous"
PetscErrorCode BVNorm_Contiguous(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  if (j<0) {
    ierr = BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,ctx->array+(bv->nc+bv->l)*bv->n,type,val,ctx->mpi);CHKERRQ(ierr);
  } else {
    ierr = BVNorm_LAPACK_Private(bv,bv->n,1,ctx->array+(bv->nc+j)*bv->n,type,val,ctx->mpi);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVNorm_Local_Contiguous"
PetscErrorCode BVNorm_Local_Contiguous(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  if (j<0) {
    ierr = BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,ctx->array+(bv->nc+bv->l)*bv->n,type,val,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = BVNorm_LAPACK_Private(bv,bv->n,1,ctx->array+(bv->nc+j)*bv->n,type,val,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalize_Contiguous"
PetscErrorCode BVOrthogonalize_Contiguous(BV V,Mat R)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)V->data;
  PetscScalar    *r=NULL;

  PetscFunctionBegin;
  if (R) { ierr = MatDenseGetArray(R,&r);CHKERRQ(ierr); }
  ierr = BVOrthogonalize_LAPACK_Private(V,V->n,V->k,ctx->array+V->nc*V->n,r,ctx->mpi);CHKERRQ(ierr);
  if (R) { ierr = MatDenseRestoreArray(R,&r);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMatMult_Contiguous"
PetscErrorCode BVMatMult_Contiguous(BV V,Mat A,BV W)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *v = (BV_CONTIGUOUS*)V->data,*w = (BV_CONTIGUOUS*)W->data;
  PetscScalar    *pb,*pc;
  PetscInt       j,m;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = MatHasOperation(A,MATOP_MAT_MULT,&flg);CHKERRQ(ierr);
  if (V->vmm && flg) {
    m = V->k-V->l;
    if (V->vmm==BV_MATMULT_MAT_SAVE) {
      ierr = BV_AllocateMatMult(V,A,m);CHKERRQ(ierr);
      ierr = MatDenseGetArray(V->B,&pb);CHKERRQ(ierr);
      ierr = PetscMemcpy(pb,v->array+(V->nc+V->l)*V->n,m*V->n*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(V->B,&pb);CHKERRQ(ierr);
    } else {  /* BV_MATMULT_MAT */
      ierr = MatCreateDense(PetscObjectComm((PetscObject)V),V->n,PETSC_DECIDE,V->N,m,v->array+(V->nc+V->l)*V->n,&V->B);CHKERRQ(ierr);
    }
    if (!V->C) {
      ierr = MatMatMultSymbolic(A,V->B,PETSC_DEFAULT,&V->C);CHKERRQ(ierr);
    }
    ierr = MatMatMultNumeric(A,V->B,V->C);CHKERRQ(ierr);
    ierr = MatDenseGetArray(V->C,&pc);CHKERRQ(ierr);
    ierr = PetscMemcpy(w->array+(W->nc+W->l)*W->n,pc,m*V->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(V->C,&pc);CHKERRQ(ierr);
    if (V->vmm==BV_MATMULT_MAT) {
      ierr = MatDestroy(&V->B);CHKERRQ(ierr);
      ierr = MatDestroy(&V->C);CHKERRQ(ierr);
    }
  } else {
    for (j=0;j<V->k-V->l;j++) {
      ierr = MatMult(A,v->V[V->nc+V->l+j],w->V[W->nc+W->l+j]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVCopy_Contiguous"
PetscErrorCode BVCopy_Contiguous(BV V,BV W)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *v = (BV_CONTIGUOUS*)V->data,*w = (BV_CONTIGUOUS*)W->data;
  PetscScalar    *pvc,*pwc;

  PetscFunctionBegin;
  pvc = v->array+(V->nc+V->l)*V->n;
  pwc = w->array+(W->nc+W->l)*W->n;
  ierr = PetscMemcpy(pwc,pvc,(V->k-V->l)*V->n*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVResize_Contiguous"
PetscErrorCode BVResize_Contiguous(BV bv,PetscInt m,PetscBool copy)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;
  PetscInt       j,bs;
  PetscScalar    *newarray;
  Vec            *newV;
  char           str[50];

  PetscFunctionBegin;
  ierr = VecGetBlockSize(bv->t,&bs);CHKERRQ(ierr);
  ierr = PetscMalloc1(m*bv->n,&newarray);CHKERRQ(ierr);
  ierr = PetscMemzero(newarray,m*bv->n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&newV);CHKERRQ(ierr);
  for (j=0;j<m;j++) {
    if (ctx->mpi) {
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,bv->n,PETSC_DECIDE,newarray+j*bv->n,newV+j);CHKERRQ(ierr);
    } else {
      ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,bv->n,newarray+j*bv->n,newV+j);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogObjectParents(bv,m,newV);CHKERRQ(ierr);
  if (((PetscObject)bv)->name) {
    for (j=0;j<m;j++) {
      ierr = PetscSNPrintf(str,50,"%s_%d",((PetscObject)bv)->name,(int)j);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)newV[j],str);CHKERRQ(ierr);
    }
  }
  if (copy) {
    ierr = PetscMemcpy(newarray,ctx->array,PetscMin(m,bv->m)*bv->n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = VecDestroyVecs(bv->m,&ctx->V);CHKERRQ(ierr);
  ctx->V = newV;
  ierr = PetscFree(ctx->array);CHKERRQ(ierr);
  ctx->array = newarray;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetColumn_Contiguous"
PetscErrorCode BVGetColumn_Contiguous(BV bv,PetscInt j,Vec *v)
{
  BV_CONTIGUOUS *ctx = (BV_CONTIGUOUS*)bv->data;
  PetscInt      l;

  PetscFunctionBegin;
  l = BVAvailableVec;
  bv->cv[l] = ctx->V[bv->nc+j];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetArray_Contiguous"
PetscErrorCode BVGetArray_Contiguous(BV bv,PetscScalar **a)
{
  BV_CONTIGUOUS *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  *a = ctx->array;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetArrayRead_Contiguous"
PetscErrorCode BVGetArrayRead_Contiguous(BV bv,const PetscScalar **a)
{
  BV_CONTIGUOUS *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  *a = ctx->array;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDestroy_Contiguous"
PetscErrorCode BVDestroy_Contiguous(BV bv)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(bv->nc+bv->m,&ctx->V);CHKERRQ(ierr);
  ierr = PetscFree(ctx->array);CHKERRQ(ierr);
  ierr = PetscFree(bv->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVCreate_Contiguous"
PETSC_EXTERN PetscErrorCode BVCreate_Contiguous(BV bv)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx;
  PetscInt       j,nloc,bs;
  PetscBool      seq;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscNewLog(bv,&ctx);CHKERRQ(ierr);
  bv->data = (void*)ctx;

  ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECMPI,&ctx->mpi);CHKERRQ(ierr);
  if (!ctx->mpi) {
    ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq);CHKERRQ(ierr);
    if (!seq) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot create a contiguous BV from a non-standard template vector");
  }

  ierr = VecGetLocalSize(bv->t,&nloc);CHKERRQ(ierr);
  ierr = VecGetBlockSize(bv->t,&bs);CHKERRQ(ierr);
  ierr = PetscMalloc1(bv->m*nloc,&ctx->array);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx->array,bv->m*nloc*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc1(bv->m,&ctx->V);CHKERRQ(ierr);
  for (j=0;j<bv->m;j++) {
    if (ctx->mpi) {
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,ctx->array+j*nloc,ctx->V+j);CHKERRQ(ierr);
    } else {
      ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,ctx->array+j*nloc,ctx->V+j);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogObjectParents(bv,bv->m,ctx->V);CHKERRQ(ierr);
  if (((PetscObject)bv)->name) {
    for (j=0;j<bv->m;j++) {
      ierr = PetscSNPrintf(str,50,"%s_%d",((PetscObject)bv)->name,(int)j);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)ctx->V[j],str);CHKERRQ(ierr);
    }
  }

  bv->ops->mult             = BVMult_Contiguous;
  bv->ops->multvec          = BVMultVec_Contiguous;
  bv->ops->multinplace      = BVMultInPlace_Contiguous;
  bv->ops->multinplacetrans = BVMultInPlaceTranspose_Contiguous;
  bv->ops->dot              = BVDot_Contiguous;
  bv->ops->dotvec           = BVDotVec_Contiguous;
  bv->ops->dotvec_local     = BVDotVec_Local_Contiguous;
  bv->ops->scale            = BVScale_Contiguous;
  bv->ops->norm             = BVNorm_Contiguous;
  bv->ops->norm_local       = BVNorm_Local_Contiguous;
  /*bv->ops->orthogonalize    = BVOrthogonalize_Contiguous;*/
  bv->ops->matmult          = BVMatMult_Contiguous;
  bv->ops->copy             = BVCopy_Contiguous;
  bv->ops->resize           = BVResize_Contiguous;
  bv->ops->getcolumn        = BVGetColumn_Contiguous;
  bv->ops->getarray         = BVGetArray_Contiguous;
  bv->ops->getarrayread     = BVGetArrayRead_Contiguous;
  bv->ops->destroy          = BVDestroy_Contiguous;
  PetscFunctionReturn(0);
}

