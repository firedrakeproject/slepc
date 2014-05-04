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

typedef struct {
  Mat       A;
  PetscBool mpi;
} BV_MAT;

#undef __FUNCT__
#define __FUNCT__ "BVMult_Mat"
PetscErrorCode BVMult_Mat(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  PetscErrorCode ierr;
  BV_MAT         *y = (BV_MAT*)Y->data,*x = (BV_MAT*)X->data;
  PetscScalar    *px,*py,*q;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(x->A,&px);CHKERRQ(ierr);
  ierr = MatDenseGetArray(y->A,&py);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  ierr = BVMult_BLAS_Private(Y,X->k,Y->k,X->n,alpha,px,q,beta,py);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(x->A,&px);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(y->A,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultVec_Mat"
PetscErrorCode BVMultVec_Mat(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  PetscErrorCode ierr;
  BV_MAT         *x = (BV_MAT*)X->data;
  PetscScalar    *px,*py;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(x->A,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = BVMultVec_BLAS_Private(X,X->n,X->k,alpha,px,q,beta,py);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(x->A,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultInPlace_Mat"
PetscErrorCode BVMultInPlace_Mat(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  BV_MAT         *ctx = (BV_MAT*)V->data;
  PetscScalar    *pv,*q;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(ctx->A,&pv);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  ierr = BVMultInPlace_BLAS_Private(V,V->k,s,e,V->n,pv,q);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(ctx->A,&pv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDot_Mat"
PetscErrorCode BVDot_Mat(BV X,BV Y,Mat M)
{
  PetscErrorCode ierr;
  BV_MAT         *x = (BV_MAT*)X->data,*y = (BV_MAT*)Y->data;
  PetscScalar    *px,*py,*m;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(x->A,&px);CHKERRQ(ierr);
  ierr = MatDenseGetArray(y->A,&py);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M,&m);CHKERRQ(ierr);
  ierr = BVDot_BLAS_Private(X,Y->k,X->k,X->n,py,px,m,x->mpi);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(M,&m);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(x->A,&px);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(y->A,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec_Mat"
PetscErrorCode BVDotVec_Mat(BV X,Vec y,PetscScalar *m)
{
  PetscErrorCode ierr;
  BV_MAT         *x = (BV_MAT*)X->data;
  PetscScalar    *px,*py;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(x->A,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = BVDotVec_BLAS_Private(X,X->n,X->k,px,py,m,x->mpi);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(x->A,&px);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVScale_Mat"
PetscErrorCode BVScale_Mat(BV bv,PetscInt j,PetscScalar alpha)
{
  PetscErrorCode ierr;
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(ctx->A,&array);CHKERRQ(ierr);
  if (j<0) {
    ierr = BVScale_BLAS_Private(bv,bv->k*bv->n,array,alpha);CHKERRQ(ierr);
  } else {
    ierr = BVScale_BLAS_Private(bv,bv->n,array+j*bv->n,alpha);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(ctx->A,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVNorm_Mat"
PetscErrorCode BVNorm_Mat(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  PetscErrorCode ierr;
  BV_MAT         *ctx = (BV_MAT*)bv->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(ctx->A,&array);CHKERRQ(ierr);
  if (j<0) {
    ierr = BVNorm_LAPACK_Private(bv,bv->n,bv->k,array,type,val,ctx->mpi);CHKERRQ(ierr);
  } else {
    ierr = BVNorm_LAPACK_Private(bv,bv->n,1,array+j*bv->n,type,val,ctx->mpi);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(ctx->A,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeAll_Mat"
PetscErrorCode BVOrthogonalizeAll_Mat(BV V,Mat R)
{
  PetscErrorCode ierr;
  BV_MAT         *ctx = (BV_MAT*)V->data;
  PetscScalar    *pv,*r=NULL;

  PetscFunctionBegin;
  if (R) { ierr = MatDenseGetArray(R,&r);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(ctx->A,&pv);CHKERRQ(ierr);
  ierr = BVOrthogonalize_LAPACK_Private(V,V->n,V->k,pv,r,ctx->mpi);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(ctx->A,&pv);CHKERRQ(ierr);
  if (R) { ierr = MatDenseRestoreArray(R,&r);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVCopy_Mat"
PetscErrorCode BVCopy_Mat(BV V,BV W)
{
  PetscErrorCode ierr;
  BV_MAT         *v = (BV_MAT*)V->data,*w = (BV_MAT*)W->data;
  PetscScalar    *pv,*pw;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(v->A,&pv);CHKERRQ(ierr);
  ierr = MatDenseGetArray(w->A,&pw);CHKERRQ(ierr);
  ierr = PetscMemcpy(pw,pv,V->k*V->n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(v->A,&pv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(w->A,&pw);CHKERRQ(ierr);
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
  PetscBool      seq;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscNewLog(bv,&ctx);CHKERRQ(ierr);
  bv->data = (void*)ctx;

  ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECMPI,&ctx->mpi);CHKERRQ(ierr);
  if (!ctx->mpi) {
    ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq);CHKERRQ(ierr);
    if (!seq) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot create a BVMAT from a non-standard template vector");
  }

  ierr = VecGetLocalSize(bv->t,&nloc);CHKERRQ(ierr);
  ierr = VecGetBlockSize(bv->t,&bs);CHKERRQ(ierr);

  ierr = MatCreateDense(PetscObjectComm((PetscObject)bv->t),nloc,PETSC_DECIDE,PETSC_DECIDE,bv->m,NULL,&ctx->A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(ctx->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)bv,(PetscObject)ctx->A);CHKERRQ(ierr);
  if (((PetscObject)bv)->name) {
    ierr = PetscSNPrintf(str,50,"%s_0",((PetscObject)bv)->name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)ctx->A,str);CHKERRQ(ierr);
  }

  if (ctx->mpi) {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,NULL,&bv->cv[0]);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,NULL,&bv->cv[1]);CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,NULL,&bv->cv[0]);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,NULL,&bv->cv[1]);CHKERRQ(ierr);
  }

  bv->ops->mult           = BVMult_Mat;
  bv->ops->multvec        = BVMultVec_Mat;
  bv->ops->multinplace    = BVMultInPlace_Mat;
  bv->ops->dot            = BVDot_Mat;
  bv->ops->dotvec         = BVDotVec_Mat;
  bv->ops->scale          = BVScale_Mat;
  bv->ops->norm           = BVNorm_Mat;
  bv->ops->orthogonalize  = BVOrthogonalizeAll_Mat;
  bv->ops->copy           = BVCopy_Mat;
  bv->ops->getcolumn      = BVGetColumn_Mat;
  bv->ops->restorecolumn  = BVRestoreColumn_Mat;
  bv->ops->view           = BVView_Mat;
  bv->ops->destroy        = BVDestroy_Mat;
  PetscFunctionReturn(0);
}

