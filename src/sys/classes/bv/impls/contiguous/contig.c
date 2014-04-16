/*
   BV implemented as an array of Vecs sharing a contiguous array for elements

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/bvimpl.h>          /*I "slepcbv.h" I*/
#include <petscblaslapack.h>

typedef struct {
  Vec         *V;
  PetscScalar *array;
} BV_CONTIGUOUS;

#undef __FUNCT__
#define __FUNCT__ "BVMult_Contiguous"
PetscErrorCode BVMult_Contiguous(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *y = (BV_CONTIGUOUS*)Y->data,*x = (BV_CONTIGUOUS*)X->data;
  PetscScalar    *q,*s;
  PetscInt       i,j,ldq;

  PetscFunctionBegin;
  ldq = X->k;
  if (alpha!=1.0) { ierr = PetscMalloc1(X->k,&s);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  for (j=0;j<Y->k;j++) {
    ierr = VecScale(y->V[j],beta);CHKERRQ(ierr);
    if (alpha!=1.0) {
      for (i=0;i<X->k;i++) s[i] = alpha*q[i+j*ldq];
    } else s = q+j*ldq;
    ierr = VecMAXPY(y->V[j],X->k,s,x->V);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  if (alpha!=1.0) { ierr = PetscFree(s);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultVec_Contiguous"
PetscErrorCode BVMultVec_Contiguous(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data;
  PetscScalar    *py;
  PetscBLASInt   n,k,one=1;

  PetscFunctionBegin;
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->k,&k);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->n,&n);CHKERRQ(ierr);
  if (n>0) PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&k,&alpha,x->array,&n,q,&one,&beta,py,&one));
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDot_Contiguous"
PetscErrorCode BVDot_Contiguous(BV X,BV Y,Mat M)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data,*y = (BV_CONTIGUOUS*)Y->data;
  PetscScalar    *m;
  PetscInt       j,ldm;

  PetscFunctionBegin;
  ldm = Y->k;
  ierr = MatDenseGetArray(M,&m);CHKERRQ(ierr);
  for (j=0;j<X->k;j++) {
    ierr = VecMDot(x->V[j],Y->k,y->V,m+j*ldm);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(M,&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec_Contiguous"
PetscErrorCode BVDotVec_Contiguous(BV X,Vec y,PetscScalar *m)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *x = (BV_CONTIGUOUS*)X->data;

  PetscFunctionBegin;
  ierr = VecMDot(y,X->k,x->V,m);CHKERRQ(ierr);
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
  bv->cv[l] = ctx->V[j];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVView_Contiguous"
PetscErrorCode BVView_Contiguous(BV bv,PetscViewer viewer)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;
  PetscInt       j;

  PetscFunctionBegin;
  for (j=0;j<bv->k;j++) {
    ierr = VecView(ctx->V[j],viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDestroy_Contiguous"
PetscErrorCode BVDestroy_Contiguous(BV bv)
{
  PetscErrorCode ierr;
  BV_CONTIGUOUS  *ctx = (BV_CONTIGUOUS*)bv->data;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(bv->k,&ctx->V);CHKERRQ(ierr);
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
  PetscBool      seq,mpi;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscNewLog(bv,&ctx);CHKERRQ(ierr);
  bv->data = (void*)ctx;

  ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECMPI,&mpi);CHKERRQ(ierr);
  if (!mpi) {
    ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq);CHKERRQ(ierr);
    if (!seq) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot create a contiguous BV from a non-standard template vector");
  }

  ierr = VecGetLocalSize(bv->t,&nloc);CHKERRQ(ierr);
  ierr = VecGetBlockSize(bv->t,&bs);CHKERRQ(ierr);
  ierr = PetscMalloc1(bv->k*nloc,&ctx->array);CHKERRQ(ierr);
  ierr = PetscMalloc1(bv->k,&ctx->V);CHKERRQ(ierr);
  for (j=0;j<bv->k;j++) {
    if (mpi) {
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,ctx->array+j*nloc,ctx->V+j);CHKERRQ(ierr);
    } else {
      ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,ctx->array+j*nloc,ctx->V+j);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogObjectParents(bv,bv->k,ctx->V);CHKERRQ(ierr);
  if (((PetscObject)bv)->name) {
    for (j=0;j<bv->k;j++) {
      ierr = PetscSNPrintf(str,50,"%s_%d",((PetscObject)bv)->name,j);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)ctx->V[j],str);CHKERRQ(ierr);
    }
  }

  bv->ops->mult           = BVMult_Contiguous;
  bv->ops->multvec        = BVMultVec_Contiguous;
  bv->ops->dot            = BVDot_Contiguous;
  bv->ops->dotvec         = BVDotVec_Contiguous;
  bv->ops->getcolumn      = BVGetColumn_Contiguous;
  bv->ops->view           = BVView_Contiguous;
  bv->ops->destroy        = BVDestroy_Contiguous;
  PetscFunctionReturn(0);
}

