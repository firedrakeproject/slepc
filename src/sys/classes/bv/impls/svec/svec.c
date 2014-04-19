/*
   BV implemented as a single Vec

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
  Vec         v;
  PetscScalar *pv;
} BV_SVEC;

#undef __FUNCT__
#define __FUNCT__ "BVMult_Svec"
PetscErrorCode BVMult_Svec(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  PetscErrorCode ierr;
  BV_SVEC        *y = (BV_SVEC*)Y->data,*x = (BV_SVEC*)X->data;
  PetscScalar    *px,*py,*q;
  PetscBLASInt   m,n,k,l,bs=BLOCKSIZE;

  PetscFunctionBegin;
  ierr = VecGetArray(x->v,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y->v,&py);CHKERRQ(ierr);
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
  ierr = VecRestoreArray(x->v,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y->v,&py);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*n*m*l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultVec_Svec"
PetscErrorCode BVMultVec_Svec(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  PetscErrorCode ierr;
  BV_SVEC        *x = (BV_SVEC*)X->data;
  PetscScalar    *px,*py;
  PetscBLASInt   n,k,one=1;

  PetscFunctionBegin;
  ierr = VecGetArray(x->v,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->k,&k);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->n,&n);CHKERRQ(ierr);
  if (n>0) PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&k,&alpha,px,&n,q,&one,&beta,py,&one));
  ierr = VecRestoreArray(x->v,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDot_Svec"
PetscErrorCode BVDot_Svec(BV X,BV Y,Mat M)
{
  PetscErrorCode ierr;
  BV_SVEC        *x = (BV_SVEC*)X->data,*y = (BV_SVEC*)Y->data;
  PetscScalar    *px,*py,*m,zero=0.0,one=1.0;
  PetscBLASInt   k,n,l;

  PetscFunctionBegin;
  ierr = VecGetArray(x->v,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y->v,&py);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->k,&k);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(Y->k,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->n,&n);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&l,&k,&n,&one,py,&n,px,&n,&zero,m,&l));
  ierr = MatDenseRestoreArray(M,&m);CHKERRQ(ierr);
  ierr = VecRestoreArray(x->v,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y->v,&py);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*n*k*l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec_Svec"
PetscErrorCode BVDotVec_Svec(BV X,Vec y,PetscScalar *m)
{
  PetscErrorCode ierr;
  BV_SVEC        *x = (BV_SVEC*)X->data;
  PetscScalar    *px,*py,zero=0.0,done=1.0;
  PetscBLASInt   k,n,one=1;

  PetscFunctionBegin;
  ierr = VecGetArray(x->v,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->k,&k);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->n,&n);CHKERRQ(ierr);
  if (n>0) PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&k,&done,px,&n,py,&one,&zero,m,&one));
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  ierr = VecRestoreArray(x->v,&px);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetColumn_Svec"
PetscErrorCode BVGetColumn_Svec(BV bv,PetscInt j,Vec *v)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscInt       l;

  PetscFunctionBegin;
  l = BVAvailableVec;
  ierr = VecGetArray(ctx->v,&ctx->pv);CHKERRQ(ierr);
  ierr = VecPlaceArray(bv->cv[l],ctx->pv+j*bv->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVRestoreColumn_Svec"
PetscErrorCode BVRestoreColumn_Svec(BV bv,PetscInt j,Vec *v)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscInt       l;

  PetscFunctionBegin;
  l = (j==bv->ci[0])? 0: 1;
  ierr = VecResetArray(bv->cv[l]);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->v,&ctx->pv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVView_Svec"
PetscErrorCode BVView_Svec(BV bv,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  BV_SVEC           *ctx = (BV_SVEC*)bv->data;
  PetscViewerFormat format;
  PetscBool         isascii;

  PetscFunctionBegin;
  ierr = VecView(ctx->v,viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"%s=reshape(%s,%d,%d)\n",((PetscObject)bv)->name,((PetscObject)ctx->v)->name,bv->N,bv->k);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDestroy_Svec"
PetscErrorCode BVDestroy_Svec(BV bv)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;

  PetscFunctionBegin;
  ierr = VecDestroy(&ctx->v);CHKERRQ(ierr);
  ierr = VecDestroy(&bv->cv[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&bv->cv[1]);CHKERRQ(ierr);
  ierr = PetscFree(bv->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVCreate_Svec"
PETSC_EXTERN PetscErrorCode BVCreate_Svec(BV bv)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx;
  PetscInt       nloc,bs;
  PetscBool      seq,mpi;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscNewLog(bv,&ctx);CHKERRQ(ierr);
  bv->data = (void*)ctx;

  ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECMPI,&mpi);CHKERRQ(ierr);
  if (!mpi) {
    ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq);CHKERRQ(ierr);
    if (!seq) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot create a BVSVEC from a non-standard template vector");
  }

  ierr = VecGetLocalSize(bv->t,&nloc);CHKERRQ(ierr);
  ierr = VecGetBlockSize(bv->t,&bs);CHKERRQ(ierr);

  ierr = VecCreate(PetscObjectComm((PetscObject)bv->t),&ctx->v);CHKERRQ(ierr);
  ierr = VecSetType(ctx->v,((PetscObject)bv->t)->type_name);CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->v,bv->k*nloc,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(ctx->v,bs);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)bv,(PetscObject)ctx->v);CHKERRQ(ierr);
  if (((PetscObject)bv)->name) {
    ierr = PetscSNPrintf(str,50,"%s_0",((PetscObject)bv)->name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)ctx->v,str);CHKERRQ(ierr);
  }

  if (mpi) {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,NULL,&bv->cv[0]);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,NULL,&bv->cv[1]);CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,NULL,&bv->cv[0]);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,NULL,&bv->cv[1]);CHKERRQ(ierr);
  }

  bv->ops->mult           = BVMult_Svec;
  bv->ops->multvec        = BVMultVec_Svec;
  bv->ops->dot            = BVDot_Svec;
  bv->ops->dotvec         = BVDotVec_Svec;
  bv->ops->getcolumn      = BVGetColumn_Svec;
  bv->ops->restorecolumn  = BVRestoreColumn_Svec;
  bv->ops->view           = BVView_Svec;
  bv->ops->destroy        = BVDestroy_Svec;
  PetscFunctionReturn(0);
}

