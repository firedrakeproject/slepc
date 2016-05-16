/*
   BV implemented as a single Vec

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
  Vec       v;
  PetscBool mpi;
} BV_SVEC;

#undef __FUNCT__
#define __FUNCT__ "BVMult_Svec"
PetscErrorCode BVMult_Svec(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  PetscErrorCode    ierr;
  BV_SVEC           *y = (BV_SVEC*)Y->data,*x = (BV_SVEC*)X->data;
  const PetscScalar *px;
  PetscScalar       *py,*q;
  PetscInt          ldq;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x->v,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y->v,&py);CHKERRQ(ierr);
  if (Q) {
    ierr = MatGetSize(Q,&ldq,NULL);CHKERRQ(ierr);
    ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
    ierr = BVMult_BLAS_Private(Y,Y->n,Y->k-Y->l,X->k-X->l,ldq,alpha,px+(X->nc+X->l)*X->n,q+Y->l*ldq+X->l,beta,py+(Y->nc+Y->l)*Y->n);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  } else {
    ierr = BVAXPY_BLAS_Private(Y,Y->n,Y->k-Y->l,alpha,px+(X->nc+X->l)*X->n,beta,py+(Y->nc+Y->l)*Y->n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(x->v,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y->v,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultVec_Svec"
PetscErrorCode BVMultVec_Svec(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  PetscErrorCode ierr;
  BV_SVEC        *x = (BV_SVEC*)X->data;
  PetscScalar    *px,*py;

  PetscFunctionBegin;
  ierr = VecGetArray(x->v,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = BVMultVec_BLAS_Private(X,X->n,X->k-X->l,alpha,px+(X->nc+X->l)*X->n,q,beta,py);CHKERRQ(ierr);
  ierr = VecRestoreArray(x->v,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultInPlace_Svec"
PetscErrorCode BVMultInPlace_Svec(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)V->data;
  PetscScalar    *pv,*q;
  PetscInt       ldq;

  PetscFunctionBegin;
  ierr = MatGetSize(Q,&ldq,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->v,&pv);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  ierr = BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,pv+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->v,&pv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultInPlaceTranspose_Svec"
PetscErrorCode BVMultInPlaceTranspose_Svec(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)V->data;
  PetscScalar    *pv,*q;
  PetscInt       ldq;

  PetscFunctionBegin;
  ierr = MatGetSize(Q,&ldq,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->v,&pv);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  ierr = BVMultInPlace_BLAS_Private(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,pv+(V->nc+V->l)*V->n,q+V->l*ldq+V->l,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->v,&pv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDot_Svec"
PetscErrorCode BVDot_Svec(BV X,BV Y,Mat M)
{
  PetscErrorCode    ierr;
  BV_SVEC           *x = (BV_SVEC*)X->data,*y = (BV_SVEC*)Y->data;
  const PetscScalar *px,*py;
  PetscScalar       *m;
  PetscInt          ldm;

  PetscFunctionBegin;
  ierr = MatGetSize(M,&ldm,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x->v,&px);CHKERRQ(ierr);
  ierr = VecGetArrayRead(y->v,&py);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M,&m);CHKERRQ(ierr);
  ierr = BVDot_BLAS_Private(X,Y->k-Y->l,X->k-X->l,X->n,ldm,py+(Y->nc+Y->l)*Y->n,px+(X->nc+X->l)*X->n,m+X->l*ldm+Y->l,x->mpi);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(M,&m);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x->v,&px);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(y->v,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec_Svec"
PetscErrorCode BVDotVec_Svec(BV X,Vec y,PetscScalar *m)
{
  PetscErrorCode    ierr;
  BV_SVEC           *x = (BV_SVEC*)X->data;
  const PetscScalar *px,*py;
  Vec               z = y;

  PetscFunctionBegin;
  if (X->matrix) {
    ierr = BV_IPMatMult(X,y);CHKERRQ(ierr);
    z = X->Bx;
  }
  ierr = VecGetArrayRead(x->v,&px);CHKERRQ(ierr);
  ierr = VecGetArrayRead(z,&py);CHKERRQ(ierr);
  ierr = BVDotVec_BLAS_Private(X,X->n,X->k-X->l,px+(X->nc+X->l)*X->n,py,m,x->mpi);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(z,&py);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x->v,&px);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec_Local_Svec"
PetscErrorCode BVDotVec_Local_Svec(BV X,Vec y,PetscScalar *m)
{
  PetscErrorCode ierr;
  BV_SVEC        *x = (BV_SVEC*)X->data;
  PetscScalar    *px,*py;
  Vec            z = y;

  PetscFunctionBegin;
  if (X->matrix) {
    ierr = BV_IPMatMult(X,y);CHKERRQ(ierr);
    z = X->Bx;
  }
  ierr = VecGetArray(x->v,&px);CHKERRQ(ierr);
  ierr = VecGetArray(z,&py);CHKERRQ(ierr);
  ierr = BVDotVec_BLAS_Private(X,X->n,X->k-X->l,px+(X->nc+X->l)*X->n,py,m,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecRestoreArray(z,&py);CHKERRQ(ierr);
  ierr = VecRestoreArray(x->v,&px);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVScale_Svec"
PetscErrorCode BVScale_Svec(BV bv,PetscInt j,PetscScalar alpha)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecGetArray(ctx->v,&array);CHKERRQ(ierr);
  if (j<0) {
    ierr = BVScale_BLAS_Private(bv,(bv->k-bv->l)*bv->n,array+(bv->nc+bv->l)*bv->n,alpha);CHKERRQ(ierr);
  } else {
    ierr = BVScale_BLAS_Private(bv,bv->n,array+(bv->nc+j)*bv->n,alpha);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(ctx->v,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVNorm_Svec"
PetscErrorCode BVNorm_Svec(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecGetArray(ctx->v,&array);CHKERRQ(ierr);
  if (j<0) {
    ierr = BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,type,val,ctx->mpi);CHKERRQ(ierr);
  } else {
    ierr = BVNorm_LAPACK_Private(bv,bv->n,1,array+(bv->nc+j)*bv->n,type,val,ctx->mpi);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(ctx->v,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVNorm_Local_Svec"
PetscErrorCode BVNorm_Local_Svec(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecGetArray(ctx->v,&array);CHKERRQ(ierr);
  if (j<0) {
    ierr = BVNorm_LAPACK_Private(bv,bv->n,bv->k-bv->l,array+(bv->nc+bv->l)*bv->n,type,val,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = BVNorm_LAPACK_Private(bv,bv->n,1,array+(bv->nc+j)*bv->n,type,val,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(ctx->v,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalize_Svec"
PetscErrorCode BVOrthogonalize_Svec(BV V,Mat R)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)V->data;
  PetscScalar    *pv,*r=NULL;

  PetscFunctionBegin;
  if (R) { ierr = MatDenseGetArray(R,&r);CHKERRQ(ierr); }
  ierr = VecGetArray(ctx->v,&pv);CHKERRQ(ierr);
  ierr = BVOrthogonalize_LAPACK_Private(V,V->n,V->k,pv+V->nc*V->n,r,ctx->mpi);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->v,&pv);CHKERRQ(ierr);
  if (R) { ierr = MatDenseRestoreArray(R,&r);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMatMult_Svec"
PetscErrorCode BVMatMult_Svec(BV V,Mat A,BV W)
{
  PetscErrorCode ierr;
  BV_SVEC        *v = (BV_SVEC*)V->data,*w = (BV_SVEC*)W->data;
  PetscScalar    *pv,*pw,*pb,*pc;
  PetscInt       j,m;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = VecGetArray(v->v,&pv);CHKERRQ(ierr);
  ierr = VecGetArray(w->v,&pw);CHKERRQ(ierr);
  ierr = MatHasOperation(A,MATOP_MAT_MULT,&flg);CHKERRQ(ierr);
  if (V->vmm && flg) {
    m = V->k-V->l;
    if (V->vmm==BV_MATMULT_MAT_SAVE) {
      ierr = BV_AllocateMatMult(V,A,m);CHKERRQ(ierr);
      ierr = MatDenseGetArray(V->B,&pb);CHKERRQ(ierr);
      ierr = PetscMemcpy(pb,pv+(V->nc+V->l)*V->n,m*V->n*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(V->B,&pb);CHKERRQ(ierr);
    } else {  /* BV_MATMULT_MAT */
      ierr = MatCreateDense(PetscObjectComm((PetscObject)V),V->n,PETSC_DECIDE,V->N,m,pv+(V->nc+V->l)*V->n,&V->B);CHKERRQ(ierr);
    }
    if (!V->C) {
      ierr = MatMatMultSymbolic(A,V->B,PETSC_DEFAULT,&V->C);CHKERRQ(ierr);
    }
    ierr = MatMatMultNumeric(A,V->B,V->C);CHKERRQ(ierr);
    ierr = MatDenseGetArray(V->C,&pc);CHKERRQ(ierr);
    ierr = PetscMemcpy(pw+(W->nc+W->l)*W->n,pc,m*V->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(V->C,&pc);CHKERRQ(ierr);
    if (V->vmm==BV_MATMULT_MAT) {
      ierr = MatDestroy(&V->B);CHKERRQ(ierr);
      ierr = MatDestroy(&V->C);CHKERRQ(ierr);
    }
  } else {
    for (j=0;j<V->k-V->l;j++) {
      ierr = VecPlaceArray(V->cv[1],pv+(V->nc+V->l+j)*V->n);CHKERRQ(ierr);
      ierr = VecPlaceArray(W->cv[1],pw+(W->nc+W->l+j)*W->n);CHKERRQ(ierr);
      ierr = MatMult(A,V->cv[1],W->cv[1]);CHKERRQ(ierr);
      ierr = VecResetArray(V->cv[1]);CHKERRQ(ierr);
      ierr = VecResetArray(W->cv[1]);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(v->v,&pv);CHKERRQ(ierr);
  ierr = VecRestoreArray(w->v,&pw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVCopy_Svec"
PetscErrorCode BVCopy_Svec(BV V,BV W)
{
  PetscErrorCode ierr;
  BV_SVEC        *v = (BV_SVEC*)V->data,*w = (BV_SVEC*)W->data;
  PetscScalar    *pv,*pw,*pvc,*pwc;

  PetscFunctionBegin;
  ierr = VecGetArray(v->v,&pv);CHKERRQ(ierr);
  ierr = VecGetArray(w->v,&pw);CHKERRQ(ierr);
  pvc = pv+(V->nc+V->l)*V->n;
  pwc = pw+(W->nc+W->l)*W->n;
  ierr = PetscMemcpy(pwc,pvc,(V->k-V->l)*V->n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(v->v,&pv);CHKERRQ(ierr);
  ierr = VecRestoreArray(w->v,&pw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVResize_Svec"
PetscErrorCode BVResize_Svec(BV bv,PetscInt m,PetscBool copy)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *pv,*pnew;
  PetscInt       bs;
  Vec            vnew;
  char           str[50];

  PetscFunctionBegin;
  ierr = VecGetBlockSize(bv->t,&bs);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)bv->t),&vnew);CHKERRQ(ierr);
  ierr = VecSetType(vnew,((PetscObject)bv->t)->type_name);CHKERRQ(ierr);
  ierr = VecSetSizes(vnew,m*bv->n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(vnew,bs);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)bv,(PetscObject)vnew);CHKERRQ(ierr);
  if (((PetscObject)bv)->name) {
    ierr = PetscSNPrintf(str,50,"%s_0",((PetscObject)bv)->name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vnew,str);CHKERRQ(ierr);
  }
  if (copy) {
    ierr = VecGetArray(ctx->v,&pv);CHKERRQ(ierr);
    ierr = VecGetArray(vnew,&pnew);CHKERRQ(ierr);
    ierr = PetscMemcpy(pnew,pv,PetscMin(m,bv->m)*bv->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArray(ctx->v,&pv);CHKERRQ(ierr);
    ierr = VecRestoreArray(vnew,&pnew);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&ctx->v);CHKERRQ(ierr);
  ctx->v = vnew;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetColumn_Svec"
PetscErrorCode BVGetColumn_Svec(BV bv,PetscInt j,Vec *v)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *pv;
  PetscInt       l;

  PetscFunctionBegin;
  l = BVAvailableVec;
  ierr = VecGetArray(ctx->v,&pv);CHKERRQ(ierr);
  ierr = VecPlaceArray(bv->cv[l],pv+(bv->nc+j)*bv->n);CHKERRQ(ierr);
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
  ierr = VecRestoreArray(ctx->v,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetArray_Svec"
PetscErrorCode BVGetArray_Svec(BV bv,PetscScalar **a)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;

  PetscFunctionBegin;
  ierr = VecGetArray(ctx->v,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVRestoreArray_Svec"
PetscErrorCode BVRestoreArray_Svec(BV bv,PetscScalar **a)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;

  PetscFunctionBegin;
  ierr = VecRestoreArray(ctx->v,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetArrayRead_Svec"
PetscErrorCode BVGetArrayRead_Svec(BV bv,const PetscScalar **a)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(ctx->v,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVRestoreArrayRead_Svec"
PetscErrorCode BVRestoreArrayRead_Svec(BV bv,const PetscScalar **a)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;

  PetscFunctionBegin;
  ierr = VecRestoreArrayRead(ctx->v,a);CHKERRQ(ierr);
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
  const char        *bvname,*name;

  PetscFunctionBegin;
  ierr = VecView(ctx->v,viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscObjectGetName((PetscObject)bv,&bvname);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject)ctx->v,&name);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s=reshape(%s,%D,%D);clear %s\n",bvname,name,bv->N,bv->nc+bv->m,name);CHKERRQ(ierr);
      if (bv->nc) {
        ierr = PetscViewerASCIIPrintf(viewer,"%s=%s(:,%D:end);\n",bvname,bvname,bv->nc+1);CHKERRQ(ierr);
      }
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
  PetscBool      seq;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscNewLog(bv,&ctx);CHKERRQ(ierr);
  bv->data = (void*)ctx;

  ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECMPI,&ctx->mpi);CHKERRQ(ierr);
  if (!ctx->mpi) {
    ierr = PetscObjectTypeCompare((PetscObject)bv->t,VECSEQ,&seq);CHKERRQ(ierr);
    if (!seq) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot create a BVSVEC from a non-standard template vector");
  }

  ierr = VecGetLocalSize(bv->t,&nloc);CHKERRQ(ierr);
  ierr = VecGetBlockSize(bv->t,&bs);CHKERRQ(ierr);

  ierr = VecCreate(PetscObjectComm((PetscObject)bv->t),&ctx->v);CHKERRQ(ierr);
  ierr = VecSetType(ctx->v,((PetscObject)bv->t)->type_name);CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->v,bv->m*nloc,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(ctx->v,bs);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)bv,(PetscObject)ctx->v);CHKERRQ(ierr);
  if (((PetscObject)bv)->name) {
    ierr = PetscSNPrintf(str,50,"%s_0",((PetscObject)bv)->name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)ctx->v,str);CHKERRQ(ierr);
  }

  if (ctx->mpi) {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,NULL,&bv->cv[0]);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,PETSC_DECIDE,NULL,&bv->cv[1]);CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,NULL,&bv->cv[0]);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)bv->t),bs,nloc,NULL,&bv->cv[1]);CHKERRQ(ierr);
  }

  bv->ops->mult             = BVMult_Svec;
  bv->ops->multvec          = BVMultVec_Svec;
  bv->ops->multinplace      = BVMultInPlace_Svec;
  bv->ops->multinplacetrans = BVMultInPlaceTranspose_Svec;
  bv->ops->dot              = BVDot_Svec;
  bv->ops->dotvec           = BVDotVec_Svec;
  bv->ops->dotvec_local     = BVDotVec_Local_Svec;
  bv->ops->scale            = BVScale_Svec;
  bv->ops->norm             = BVNorm_Svec;
  bv->ops->norm_local       = BVNorm_Local_Svec;
  /*bv->ops->orthogonalize    = BVOrthogonalize_Svec;*/
  bv->ops->matmult          = BVMatMult_Svec;
  bv->ops->copy             = BVCopy_Svec;
  bv->ops->resize           = BVResize_Svec;
  bv->ops->getcolumn        = BVGetColumn_Svec;
  bv->ops->restorecolumn    = BVRestoreColumn_Svec;
  bv->ops->getarray         = BVGetArray_Svec;
  bv->ops->restorearray     = BVRestoreArray_Svec;
  bv->ops->getarrayread     = BVGetArrayRead_Svec;
  bv->ops->restorearrayread = BVRestoreArrayRead_Svec;
  bv->ops->destroy          = BVDestroy_Svec;
  if (!ctx->mpi) bv->ops->view = BVView_Svec;
  PetscFunctionReturn(0);
}

