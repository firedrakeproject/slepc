/*
   BV implemented with a dense Mat

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

#define BLOCKSIZE 64

typedef struct {
  Mat A;
} BV_MAT;

#undef __FUNCT__
#define __FUNCT__ "BVMult_Mat"
PetscErrorCode BVMult_Mat(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  PetscErrorCode ierr;
  BV_MAT         *y = (BV_MAT*)Y->data,*x = (BV_MAT*)X->data;
  PetscScalar    *px,*py,*q;
  PetscBLASInt   m,n,k,l,bs=BLOCKSIZE;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(x->A,&px);CHKERRQ(ierr);
  ierr = MatDenseGetArray(y->A,&py);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->k,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(Y->k,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->n,&n);CHKERRQ(ierr);
  k = n % bs;
  if (k) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k,&l,&m,&alpha,px,&n,q,&m,&beta,py,&n));
  for (;k<n;k+=bs) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&bs,&l,&m,&alpha,px+k,&n,q,&m,&beta,py,&n));
  }
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(x->A,&px);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(y->A,&py);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*n*m*l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultVec_Mat"
PetscErrorCode BVMultVec_Mat(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  PetscErrorCode ierr;
  BV_MAT         *x = (BV_MAT*)X->data;
  PetscScalar    *px,*py;
  PetscBLASInt   n,k,one=1;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(x->A,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->k,&k);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->n,&n);CHKERRQ(ierr);
  if (n>0) PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&k,&alpha,px,&n,q,&one,&beta,py,&one));
  ierr = MatDenseRestoreArray(x->A,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDot_Mat"
PetscErrorCode BVDot_Mat(BV X,BV Y,Mat M)
{
  PetscErrorCode ierr;
  BV_MAT         *x = (BV_MAT*)X->data,*y = (BV_MAT*)Y->data;
  PetscScalar    *px,*py,*m,zero=0.0,one=1.0;
  PetscBLASInt   k,n,l;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(x->A,&px);CHKERRQ(ierr);
  ierr = MatDenseGetArray(y->A,&py);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->k,&k);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(Y->k,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->n,&n);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&l,&k,&n,&one,py,&n,px,&n,&zero,m,&l));
  ierr = MatDenseRestoreArray(M,&m);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(x->A,&px);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(y->A,&py);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*n*k*l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec_Mat"
PetscErrorCode BVDotVec_Mat(BV X,Vec y,PetscScalar *m)
{
  PetscErrorCode ierr;
  BV_MAT         *x = (BV_MAT*)X->data;
  PetscScalar    *px,*py,zero=0.0,done=1.0;
  PetscBLASInt   k,n,one=1;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(x->A,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->k,&k);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->n,&n);CHKERRQ(ierr);
  if (n>0) PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&k,&done,px,&n,py,&one,&zero,m,&one));
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(x->A,&px);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetColumn_Mat"
PetscErrorCode BVGetColumn_Mat(BV bv,PetscInt j,Vec *v)
{
  PetscErrorCode ierr;
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *pA;
  PetscInt       l;

  PetscFunctionBegin;
  l = BVAvailableVec;
  ierr = MatDenseGetArray(ctx->A,&pA);CHKERRQ(ierr);
  ierr = VecPlaceArray(bv->cv[l],pA+j*bv->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVRestoreColumn_Mat"
PetscErrorCode BVRestoreColumn_Mat(BV bv,PetscInt j,Vec *v)
{
  PetscErrorCode ierr;
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *pA;
  PetscInt       l;

  PetscFunctionBegin;
  l = (j==bv->ci[0])? 0: 1;
  ierr = VecResetArray(bv->cv[l]);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(ctx->A,&pA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVView_Mat"
PetscErrorCode BVView_Mat(BV bv,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  BV_MAT            *ctx = (BV_MAT*)bv->data;
  PetscViewerFormat format;
  PetscBool         isascii;

  PetscFunctionBegin;
  ierr = MatView(ctx->A,viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"%s=%s;clear %s\n",((PetscObject)bv)->name,((PetscObject)ctx->A)->name,((PetscObject)ctx->A)->name);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDestroy_Mat"
PetscErrorCode BVDestroy_Mat(BV bv)
{
  PetscErrorCode ierr;
  BV_MAT         *ctx = (BV_MAT*)bv->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr = VecDestroy(&bv->cv[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&bv->cv[1]);CHKERRQ(ierr);
  ierr = PetscFree(bv->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVCreate_Mat"
PETSC_EXTERN PetscErrorCode BVCreate_Mat(BV bv)
{
  PetscErrorCode ierr;
  BV_MAT         *ctx;
  PetscInt       nloc,bs;
  PetscBool      seq,mpi;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscNewLog(bv,&ctx);CHKERRQ(ierr);
  bv->data = (void*)ctx;

  ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECMPI,&mpi);CHKERRQ(ierr);
  if (!mpi) {
    ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq);CHKERRQ(ierr);
    if (!seq) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot create a BVMAT from a non-standard template vector");
  }

  ierr = VecGetLocalSize(bv->t,&nloc);CHKERRQ(ierr);
  ierr = VecGetBlockSize(bv->t,&bs);CHKERRQ(ierr);

  ierr = MatCreateDense(PetscObjectComm((PetscObject)bv->t),nloc,bv->k,PETSC_DECIDE,bv->k,NULL,&ctx->A);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)bv,(PetscObject)ctx->A);CHKERRQ(ierr);
  if (((PetscObject)bv)->name) {
    ierr = PetscSNPrintf(str,50,"%s_0",((PetscObject)bv)->name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)ctx->A,str);CHKERRQ(ierr);
  }

  if (mpi) {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,NULL,&bv->cv[0]);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,NULL,&bv->cv[1]);CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,NULL,&bv->cv[0]);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,NULL,&bv->cv[1]);CHKERRQ(ierr);
  }

  bv->ops->mult           = BVMult_Mat;
  bv->ops->multvec        = BVMultVec_Mat;
  bv->ops->dot            = BVDot_Mat;
  bv->ops->dotvec         = BVDotVec_Mat;
  bv->ops->getcolumn      = BVGetColumn_Mat;
  bv->ops->restorecolumn  = BVRestoreColumn_Mat;
  bv->ops->view           = BVView_Mat;
  bv->ops->destroy        = BVDestroy_Mat;
  PetscFunctionReturn(0);
}

