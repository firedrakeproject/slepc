/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV implemented as a single Vec (CUDA version)
*/

#include <slepc/private/bvimpl.h>
#include <slepccublas.h>
#include "../src/sys/classes/bv/impls/svec/svec.h"

PetscErrorCode BVMult_Svec_CUDA(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  BV_SVEC           *y = (BV_SVEC*)Y->data,*x = (BV_SVEC*)X->data;
  const PetscScalar *d_px,*d_A;
  PetscScalar       *d_py,*d_q,*d_B,*d_C;
  PetscInt          ldq;

  PetscFunctionBegin;
  if (!Y->n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecCUDAGetArrayRead(x->v,&d_px));
  if (beta==(PetscScalar)0.0) PetscCall(VecCUDAGetArrayWrite(y->v,&d_py));
  else PetscCall(VecCUDAGetArray(y->v,&d_py));
  d_A = d_px+(X->nc+X->l)*X->n;
  d_C = d_py+(Y->nc+Y->l)*Y->n;
  if (Q) {
    PetscCall(MatDenseGetLDA(Q,&ldq));
    PetscCall(BV_MatDenseCUDAGetArrayRead(Y,Q,&d_q));
    d_B = d_q+Y->l*ldq+X->l;
    PetscCall(BVMult_BLAS_CUDA(Y,Y->n,Y->k-Y->l,X->k-X->l,ldq,alpha,d_A,d_B,beta,d_C));
    PetscCall(BV_MatDenseCUDARestoreArrayRead(Y,Q,&d_q));
  } else PetscCall(BVAXPY_BLAS_CUDA(Y,Y->n,Y->k-Y->l,alpha,d_A,beta,d_C));
  PetscCall(VecCUDARestoreArrayRead(x->v,&d_px));
  if (beta==(PetscScalar)0.0) PetscCall(VecCUDARestoreArrayWrite(y->v,&d_py));
  else PetscCall(VecCUDARestoreArray(y->v,&d_py));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVMultVec_Svec_CUDA(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  BV_SVEC           *x = (BV_SVEC*)X->data;
  PetscScalar       *d_py,*d_q;
  const PetscScalar *d_px;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArrayRead(x->v,&d_px));
  if (beta==(PetscScalar)0.0) PetscCall(VecCUDAGetArrayWrite(y,&d_py));
  else PetscCall(VecCUDAGetArray(y,&d_py));
  if (!q) PetscCall(VecCUDAGetArray(X->buffer,&d_q));
  else {
    PetscInt k=X->k-X->l;
    PetscCallCUDA(cudaMalloc((void**)&d_q,k*sizeof(PetscScalar)));
    PetscCallCUDA(cudaMemcpy(d_q,q,k*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCall(PetscLogCpuToGpu(k*sizeof(PetscScalar)));
  }
  PetscCall(BVMultVec_BLAS_CUDA(X,X->n,X->k-X->l,alpha,d_px+(X->nc+X->l)*X->n,d_q,beta,d_py));
  PetscCall(VecCUDARestoreArrayRead(x->v,&d_px));
  if (beta==(PetscScalar)0.0) PetscCall(VecCUDARestoreArrayWrite(y,&d_py));
  else PetscCall(VecCUDARestoreArray(y,&d_py));
  if (!q) PetscCall(VecCUDARestoreArray(X->buffer,&d_q));
  else PetscCallCUDA(cudaFree(d_q));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVMultInPlace_Svec_CUDA(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_SVEC           *ctx = (BV_SVEC*)V->data;
  PetscScalar       *d_pv,*d_q;
  PetscInt          ldq;

  PetscFunctionBegin;
  if (!V->n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDenseGetLDA(Q,&ldq));
  PetscCall(VecCUDAGetArray(ctx->v,&d_pv));
  PetscCall(BV_MatDenseCUDAGetArrayRead(V,Q,&d_q));
  PetscCall(BVMultInPlace_BLAS_CUDA(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,d_pv+(V->nc+V->l)*V->n,d_q+V->l*ldq+V->l,PETSC_FALSE));
  PetscCall(BV_MatDenseCUDARestoreArrayRead(V,Q,&d_q));
  PetscCall(VecCUDARestoreArray(ctx->v,&d_pv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVMultInPlaceHermitianTranspose_Svec_CUDA(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_SVEC           *ctx = (BV_SVEC*)V->data;
  PetscScalar       *d_pv,*d_q;
  PetscInt          ldq;

  PetscFunctionBegin;
  if (!V->n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDenseGetLDA(Q,&ldq));
  PetscCall(VecCUDAGetArray(ctx->v,&d_pv));
  PetscCall(BV_MatDenseCUDAGetArrayRead(V,Q,&d_q));
  PetscCall(BVMultInPlace_BLAS_CUDA(V,V->n,V->k-V->l,ldq,s-V->l,e-V->l,d_pv+(V->nc+V->l)*V->n,d_q+V->l*ldq+V->l,PETSC_TRUE));
  PetscCall(BV_MatDenseCUDARestoreArrayRead(V,Q,&d_q));
  PetscCall(VecCUDARestoreArray(ctx->v,&d_pv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVDot_Svec_CUDA(BV X,BV Y,Mat M)
{
  BV_SVEC           *x = (BV_SVEC*)X->data,*y = (BV_SVEC*)Y->data;
  const PetscScalar *d_px,*d_py;
  PetscScalar       *pm;
  PetscInt          ldm;

  PetscFunctionBegin;
  PetscCall(MatDenseGetLDA(M,&ldm));
  PetscCall(VecCUDAGetArrayRead(x->v,&d_px));
  PetscCall(VecCUDAGetArrayRead(y->v,&d_py));
  PetscCall(MatDenseGetArrayWrite(M,&pm));
  PetscCall(BVDot_BLAS_CUDA(X,Y->k-Y->l,X->k-X->l,X->n,ldm,d_py+(Y->nc+Y->l)*Y->n,d_px+(X->nc+X->l)*X->n,pm+X->l*ldm+Y->l,x->mpi));
  PetscCall(MatDenseRestoreArrayWrite(M,&pm));
  PetscCall(VecCUDARestoreArrayRead(x->v,&d_px));
  PetscCall(VecCUDARestoreArrayRead(y->v,&d_py));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVDotVec_Svec_CUDA(BV X,Vec y,PetscScalar *q)
{
  BV_SVEC           *x = (BV_SVEC*)X->data;
  const PetscScalar *d_px,*d_py;
  Vec               z = y;

  PetscFunctionBegin;
  if (PetscUnlikely(X->matrix)) {
    PetscCall(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  PetscCall(VecCUDAGetArrayRead(x->v,&d_px));
  PetscCall(VecCUDAGetArrayRead(z,&d_py));
  PetscCall(BVDotVec_BLAS_CUDA(X,X->n,X->k-X->l,d_px+(X->nc+X->l)*X->n,d_py,q,x->mpi));
  PetscCall(VecCUDARestoreArrayRead(z,&d_py));
  PetscCall(VecCUDARestoreArrayRead(x->v,&d_px));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVDotVec_Local_Svec_CUDA(BV X,Vec y,PetscScalar *m)
{
  BV_SVEC           *x = (BV_SVEC*)X->data;
  const PetscScalar *d_px,*d_py;
  Vec               z = y;

  PetscFunctionBegin;
  if (PetscUnlikely(X->matrix)) {
    PetscCall(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  PetscCall(VecCUDAGetArrayRead(x->v,&d_px));
  PetscCall(VecCUDAGetArrayRead(z,&d_py));
  PetscCall(BVDotVec_BLAS_CUDA(X,X->n,X->k-X->l,d_px+(X->nc+X->l)*X->n,d_py,m,PETSC_FALSE));
  PetscCall(VecCUDARestoreArrayRead(z,&d_py));
  PetscCall(VecCUDARestoreArrayRead(x->v,&d_px));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVScale_Svec_CUDA(BV bv,PetscInt j,PetscScalar alpha)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *d_array,*d_A;
  PetscInt       n=0;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArray(ctx->v,&d_array));
  if (PetscUnlikely(j<0)) {
    d_A = d_array+(bv->nc+bv->l)*bv->n;
    n = (bv->k-bv->l)*bv->n;
  } else {
    d_A = d_array+(bv->nc+j)*bv->n;
    n = bv->n;
  }
  PetscCall(BVScale_BLAS_CUDA(bv,n,d_A,alpha));
  PetscCall(VecCUDARestoreArray(ctx->v,&d_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVMatMult_Svec_CUDA(BV V,Mat A,BV W)
{
  BV_SVEC           *v = (BV_SVEC*)V->data,*w = (BV_SVEC*)W->data;
  Mat               Vmat,Wmat;
  const PetscScalar *d_pv;
  PetscScalar       *d_pw;
  PetscInt          j;

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
    PetscCall(VecCUDAGetArrayRead(v->v,&d_pv));
    PetscCall(VecCUDAGetArrayWrite(w->v,&d_pw));
    for (j=0;j<V->k-V->l;j++) {
      PetscCall(VecCUDAPlaceArray(V->cv[1],(PetscScalar *)d_pv+(V->nc+V->l+j)*V->n));
      PetscCall(VecCUDAPlaceArray(W->cv[1],d_pw+(W->nc+W->l+j)*W->n));
      PetscCall(MatMult(A,V->cv[1],W->cv[1]));
      PetscCall(VecCUDAResetArray(V->cv[1]));
      PetscCall(VecCUDAResetArray(W->cv[1]));
    }
    PetscCall(VecCUDARestoreArrayRead(v->v,&d_pv));
    PetscCall(VecCUDARestoreArrayWrite(w->v,&d_pw));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVCopy_Svec_CUDA(BV V,BV W)
{
  BV_SVEC           *v = (BV_SVEC*)V->data,*w = (BV_SVEC*)W->data;
  const PetscScalar *d_pv,*d_pvc;
  PetscScalar       *d_pw,*d_pwc;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArrayRead(v->v,&d_pv));
  PetscCall(VecCUDAGetArrayWrite(w->v,&d_pw));
  d_pvc = d_pv+(V->nc+V->l)*V->n;
  d_pwc = d_pw+(W->nc+W->l)*W->n;
  PetscCallCUDA(cudaMemcpy(d_pwc,d_pvc,(V->k-V->l)*V->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
  PetscCall(VecCUDARestoreArrayRead(v->v,&d_pv));
  PetscCall(VecCUDARestoreArrayWrite(w->v,&d_pw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVCopyColumn_Svec_CUDA(BV V,PetscInt j,PetscInt i)
{
  BV_SVEC        *v = (BV_SVEC*)V->data;
  PetscScalar    *d_pv;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArray(v->v,&d_pv));
  PetscCallCUDA(cudaMemcpy(d_pv+(V->nc+i)*V->n,d_pv+(V->nc+j)*V->n,V->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
  PetscCall(VecCUDARestoreArray(v->v,&d_pv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVResize_Svec_CUDA(BV bv,PetscInt m,PetscBool copy)
{
  BV_SVEC           *ctx = (BV_SVEC*)bv->data;
  const PetscScalar *d_pv;
  PetscScalar       *d_pnew;
  PetscInt          bs;
  Vec               vnew;
  char              str[50];

  PetscFunctionBegin;
  PetscCall(VecGetBlockSize(bv->t,&bs));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)bv->t),&vnew));
  PetscCall(VecSetType(vnew,((PetscObject)bv->t)->type_name));
  PetscCall(VecSetSizes(vnew,m*bv->n,PETSC_DECIDE));
  PetscCall(VecSetBlockSize(vnew,bs));
  if (((PetscObject)bv)->name) {
    PetscCall(PetscSNPrintf(str,sizeof(str),"%s_0",((PetscObject)bv)->name));
    PetscCall(PetscObjectSetName((PetscObject)vnew,str));
  }
  if (copy) {
    PetscCall(VecCUDAGetArrayRead(ctx->v,&d_pv));
    PetscCall(VecCUDAGetArrayWrite(vnew,&d_pnew));
    PetscCallCUDA(cudaMemcpy(d_pnew,d_pv,PetscMin(m,bv->m)*bv->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
    PetscCall(VecCUDARestoreArrayRead(ctx->v,&d_pv));
    PetscCall(VecCUDARestoreArrayWrite(vnew,&d_pnew));
  }
  PetscCall(VecDestroy(&ctx->v));
  ctx->v = vnew;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVGetColumn_Svec_CUDA(BV bv,PetscInt j,Vec *v)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *d_pv;
  PetscInt       l;

  PetscFunctionBegin;
  l = BVAvailableVec;
  PetscCall(VecCUDAGetArray(ctx->v,&d_pv));
  PetscCall(VecCUDAPlaceArray(bv->cv[l],d_pv+(bv->nc+j)*bv->n));
  (void)v; // avoid unused parameter warning
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVRestoreColumn_Svec_CUDA(BV bv,PetscInt j,Vec *v)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscInt       l;

  PetscFunctionBegin;
  l = (j==bv->ci[0])? 0: 1;
  PetscCall(VecCUDAResetArray(bv->cv[l]));
  PetscCall(VecCUDARestoreArray(ctx->v,NULL));
  (void)v; // avoid unused parameter warning
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVRestoreSplit_Svec_CUDA(BV bv,BV *L,BV *R)
{
  Vec               v;
  const PetscScalar *d_pv;
  PetscObjectState  lstate,rstate;
  PetscBool         change=PETSC_FALSE;

  PetscFunctionBegin;
  /* force sync flag to PETSC_CUDA_BOTH */
  if (L) {
    PetscCall(PetscObjectStateGet((PetscObject)*L,&lstate));
    if (lstate != bv->lstate) {
      v = ((BV_SVEC*)bv->L->data)->v;
      PetscCall(VecCUDAGetArrayRead(v,&d_pv));
      PetscCall(VecCUDARestoreArrayRead(v,&d_pv));
      change = PETSC_TRUE;
    }
  }
  if (R) {
    PetscCall(PetscObjectStateGet((PetscObject)*R,&rstate));
    if (rstate != bv->rstate) {
      v = ((BV_SVEC*)bv->R->data)->v;
      PetscCall(VecCUDAGetArrayRead(v,&d_pv));
      PetscCall(VecCUDARestoreArrayRead(v,&d_pv));
      change = PETSC_TRUE;
    }
  }
  if (change) {
    v = ((BV_SVEC*)bv->data)->v;
    PetscCall(VecCUDAGetArray(v,(PetscScalar **)&d_pv));
    PetscCall(VecCUDARestoreArray(v,(PetscScalar **)&d_pv));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVGetMat_Svec_CUDA(BV bv,Mat *A)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *vv,*aa;
  PetscBool      create=PETSC_FALSE;
  PetscInt       m,cols;

  PetscFunctionBegin;
  m = bv->k-bv->l;
  if (!bv->Aget) create=PETSC_TRUE;
  else {
    PetscCall(MatDenseCUDAGetArray(bv->Aget,&aa));
    PetscCheck(!aa,PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_WRONGSTATE,"BVGetMat already called on this BV");
    PetscCall(MatGetSize(bv->Aget,NULL,&cols));
    if (cols!=m) {
      PetscCall(MatDestroy(&bv->Aget));
      create=PETSC_TRUE;
    }
  }
  PetscCall(VecCUDAGetArray(ctx->v,&vv));
  if (create) {
    PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)bv),bv->n,PETSC_DECIDE,bv->N,m,vv,&bv->Aget)); /* pass a pointer to avoid allocation of storage */
    PetscCall(MatDenseCUDAReplaceArray(bv->Aget,NULL));  /* replace with a null pointer, the value after BVRestoreMat */
  }
  PetscCall(MatDenseCUDAPlaceArray(bv->Aget,vv+(bv->nc+bv->l)*bv->n));  /* set the actual pointer */
  *A = bv->Aget;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BVRestoreMat_Svec_CUDA(BV bv,Mat *A)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *vv,*aa;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDAGetArray(bv->Aget,&aa));
  vv = aa-(bv->nc+bv->l)*bv->n;
  PetscCall(MatDenseCUDAResetArray(bv->Aget));
  PetscCall(VecCUDARestoreArray(ctx->v,&vv));
  *A = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
