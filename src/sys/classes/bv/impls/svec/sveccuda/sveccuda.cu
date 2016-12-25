/*
   BV implemented as a single Vec (CUDA version)

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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
#include "../svecimpl.h"
#include <petsccuda.h>
#include <cublas_v2.h>

/* complex single */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define cublasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cublasCgemm((a),(b),(c),(d),(e),(f),(cuComplex*)(g),(cuComplex*)(h),(i),(cuComplex*)(j),(k),(cuComplex*)(l),(cuComplex*)(m),(n))
#define cublasXgemv(a,b,c,d,e,f,g,h,i,j,k,l) cublasCgemv((a),(b),(c),(d),(cuComplex*)(e),(cuComplex*)(f),(g),(cuComplex*)(h),(i),(cuComplex*)(j),(cuComplex*)(k),(l))
#define cublasXscal(a,b,c,d,e) cublasCscal(a,b,(const cuComplex*)(c),(cuComplex*)(d),e)
#define cublasXnrm2(a,b,c,d,e) cublasScnrm2(a,b,(const cuComplex*)(c),d,e)
#define cublasXaxpy(a,b,c,d,e,f,g) cublasCaxpy((a),(b),(cuComplex*)(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g))
#else /* complex double */
#define cublasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cublasZgemm((a),(b),(c),(d),(e),(f),(cuDoubleComplex*)(g),(cuDoubleComplex*)(h),(i),(cuDoubleComplex*)(j),(k),(cuDoubleComplex*)(l),(cuDoubleComplex*)(m),(n))
#define cublasXgemv(a,b,c,d,e,f,g,h,i,j,k,l) cublasZgemv((a),(b),(c),(d),(cuDoubleComplex*)(e),(cuDoubleComplex*)(f),(g),(cuDoubleComplex*)(h),(i),(cuDoubleComplex*)(j),(cuDoubleComplex*)(k),(l))
#define cublasXscal(a,b,c,d,e) cublasZscal(a,b,(const cuDoubleComplex*)(c),(cuDoubleComplex*)(d),e)
#define cublasXnrm2(a,b,c,d,e) cublasDznrm2(a,b,(const cuDoubleComplex*)(c),d,e)
#define cublasXaxpy(a,b,c,d,e,f,g) cublasZaxpy((a),(b),(cuDoubleComplex*)(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g))
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define cublasXgemm cublasSgemm
#define cublasXgemv cublasSgemv
#define cublasXscal cublasSscal
#define cublasXnrm2 cublasSnrm2
#define cublasXaxpy cublasSaxpy
#else /* real double */
#define cublasXgemm cublasDgemm
#define cublasXgemv cublasDgemv
#define cublasXscal cublasDscal
#define cublasXnrm2 cublasDnrm2
#define cublasXaxpy cublasDaxpy
#endif
#endif

/*
    B := alpha*A + beta*B

    A,B are nxk (ld=n)
 */
static PetscErrorCode BVAXPY_BLAS_CUDA(BV bv,PetscInt n_,PetscInt k_,PetscScalar alpha,const PetscScalar *d_A,PetscScalar beta,PetscScalar *d_B)
{
  PetscErrorCode ierr;
  PetscBLASInt   m,one=1;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n_*k_,&m);CHKERRQ(ierr);
  if (beta!=(PetscScalar)1.0) {
    cberr = cublasXscal(cublasv2handle,m,&beta,d_B,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogFlops(m);CHKERRQ(ierr);
  }
  cberr = cublasXaxpy(cublasv2handle,m,&alpha,d_A,one,d_B,one);CHKERRCUBLAS(cberr);
  ierr = PetscLogFlops(2.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    C := alpha*A*B + beta*C
*/
PetscErrorCode BVMult_Svec_CUDA(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  PetscErrorCode    ierr;
  BV_SVEC           *y = (BV_SVEC*)Y->data,*x = (BV_SVEC*)X->data;
  const PetscScalar *d_px,*d_A;
  PetscScalar       *d_py,*q,*d_q,*d_B,*d_C;
  PetscInt          m,n,k,ldq,mq;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  m = Y->n;
  n = Y->k-Y->l;
  k = X->k-X->l;
  if (!Y->n) PetscFunctionReturn(0);
  ierr = VecCUDAGetArrayRead(x->v,&d_px);CHKERRQ(ierr);
  if (beta==(PetscScalar)0.0) {
    ierr = VecCUDAGetArrayWrite(y->v,&d_py);CHKERRQ(ierr);
  } else {
    ierr = VecCUDAGetArrayReadWrite(y->v,&d_py);CHKERRQ(ierr);
  }
  d_A = d_px+(X->nc+X->l)*X->n;
  d_C = d_py+(Y->nc+Y->l)*Y->n;
  if (Q) {
    ierr = MatGetSize(Q,&ldq,&mq);CHKERRQ(ierr);
    ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
    err = cudaMalloc((void**)&d_q,ldq*mq*sizeof(PetscScalar));CHKERRCUDA(err);
    err = cudaMemcpy(d_q,q,ldq*mq*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);
    d_B = d_q+Y->l*ldq+X->l;
    cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,d_A,m,d_B,ldq,&beta,d_C,m);CHKERRCUBLAS(cberr);
    ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
    err = cudaFree(d_q);CHKERRCUDA(err);
    ierr = PetscLogFlops(2.0*m*n*k);CHKERRQ(ierr);
  } else {
    ierr = BVAXPY_BLAS_CUDA(Y,m,n,alpha,d_A,beta,d_C);CHKERRQ(ierr);
  }
  ierr = VecCUDARestoreArrayRead(x->v,&d_px);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(y->v,&d_py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    y := alpha*A*x + beta*y
*/
PetscErrorCode BVMultVec_Svec_CUDA(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  PetscErrorCode    ierr;
  BV_SVEC           *x = (BV_SVEC*)X->data;
  const PetscScalar *d_px,*d_A;
  PetscScalar       *d_py,*d_q,*d_x,*d_y,*qq=q;
  PetscBLASInt      n,k,one=1;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  n = X->n;
  k = X->k-X->l;
  ierr = VecCUDAGetArrayRead(x->v,&d_px);CHKERRQ(ierr);
  if (beta==(PetscScalar)0.0) {
    ierr = VecCUDAGetArrayWrite(y,&d_py);CHKERRQ(ierr);
  } else {
    ierr = VecCUDAGetArrayReadWrite(y,&d_py);CHKERRQ(ierr);
  }
  if (!q) { ierr = VecGetArray(X->buffer,&qq);CHKERRQ(ierr); }
  ierr = cudaMalloc((void**)&d_q,k*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = cudaMemcpy(d_q,qq,k*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRQ(ierr);
  d_A = d_px+(X->nc+X->l)*X->n;
  d_x = d_q;
  d_y = d_py;
  cberr = cublasXgemv(cublasv2handle,CUBLAS_OP_N,n,k,&alpha,d_A,n,d_x,one,&beta,d_y,one);CHKERRCUBLAS(cberr);
  ierr = VecCUDARestoreArrayRead(x->v,&d_px);CHKERRQ(ierr);
  if (beta==(PetscScalar)0.0) {
    ierr = VecCUDARestoreArrayWrite(y,&d_py);CHKERRQ(ierr);
  } else {
    ierr = VecCUDARestoreArrayReadWrite(y,&d_py);CHKERRQ(ierr);
  }
  if (!q) { ierr = VecRestoreArray(X->buffer,&qq);CHKERRQ(ierr); }
  ierr = cudaFree(d_q);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    A(:,s:e-1) := A*B(:,s:e-1)
*/
PetscErrorCode BVMultInPlace_Svec_CUDA(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)V->data;
  PetscScalar    *d_pv,*q,*d_q,*d_A,*d_B,*d_work,sone=1.0,szero=0.0;
  PetscInt       m,n,j,k,ldq,nq;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  m = V->n;
  n = e-s;
  k = V->k-V->l;
  if (!m) PetscFunctionReturn(0);
  ierr = MatGetSize(Q,&ldq,&nq);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadWrite(ctx->v,&d_pv);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  ierr = cudaMalloc((void**)&d_q,ldq*nq*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = cudaMemcpy(d_q,q,ldq*nq*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRQ(ierr);
  ierr = cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  d_A = d_pv+(V->nc+V->l)*m;
  d_B = d_q+V->l*ldq+V->l+(s-V->l)*ldq;
  cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&sone,d_A,m,d_B,ldq,&szero,d_work,m);CHKERRCUBLAS(cberr);
  for (j=0;j<n;j++) {
    ierr = cudaMemcpy(d_A+(s-V->l+j)*m,d_work+(j*m),m*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(ierr);
  }
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  ierr = cudaFree(d_q);CHKERRQ(ierr);
  ierr = cudaFree(d_work);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadWrite(ctx->v,&d_pv);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*m*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    A(:,s:e-1) := A*B(:,s:e-1)
*/
PetscErrorCode BVMultInPlaceTranspose_Svec_CUDA(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)V->data;
  PetscScalar    *d_pv,*q,*d_q,*d_A,*d_B,*d_work,sone=1.0,szero=0.0;
  PetscInt       m,n,j,k,ldq,nq;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  m = V->n;
  n = e-s;
  k = V->k-V->l;
  if (!m) PetscFunctionReturn(0);
  ierr = MatGetSize(Q,&ldq,&nq);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadWrite(ctx->v,&d_pv);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  ierr = cudaMalloc((void**)&d_q,ldq*nq*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = cudaMemcpy(d_q,q,ldq*nq*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRQ(ierr);
  ierr = cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  d_A = d_pv+(V->nc+V->l)*m;
  d_B = d_q+V->l*ldq+s;
  cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_C,m,n,k,&sone,d_A,m,d_B,ldq,&szero,d_work,m);CHKERRCUBLAS(cberr);
  for (j=0;j<n;j++) {
    ierr = cudaMemcpy(d_A+(s-V->l+j)*m,d_work+(j*m),m*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  ierr = cudaFree(d_q);CHKERRQ(ierr);
  ierr = cudaFree(d_work);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadWrite(ctx->v,&d_pv);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*m*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    C := A'*B
*/
PetscErrorCode BVDot_Svec_CUDA(BV X,BV Y,Mat M)
{
  PetscErrorCode    ierr;
  BV_SVEC           *x = (BV_SVEC*)X->data,*y = (BV_SVEC*)Y->data;
  const PetscScalar *d_px,*d_py,*d_A,*d_B;
  PetscScalar       *pm,*d_work,sone=1.0,szero=0.0,*C,*CC;
  PetscInt          ldm,m,n,k,len,j;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  m = Y->k-Y->l;
  n = X->k-X->l;
  k = X->n;
  ierr = MatGetSize(M,&ldm,NULL);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(x->v,&d_px);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(y->v,&d_py);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M,&pm);CHKERRQ(ierr);
  ierr = cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  d_A = d_py+(Y->nc+Y->l)*Y->n;
  d_B = d_px+(X->nc+X->l)*X->n;
  C = pm+X->l*ldm+Y->l;
  if (x->mpi) {
    if (ldm==m) {
      ierr = BVAllocateWork_Private(X,m*n);CHKERRQ(ierr);
      if (k) {
        cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,ldm);CHKERRCUBLAS(cberr);
        ierr = cudaMemcpy(X->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRQ(ierr);
      } else {
        ierr = PetscMemzero(X->work,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      ierr = PetscMPIIntCast(m*n,&len);CHKERRQ(ierr);
      ierr = MPI_Allreduce(X->work,C,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)X));CHKERRQ(ierr);
    } else {
      ierr = BVAllocateWork_Private(X,2*m*n);CHKERRQ(ierr);
      CC = X->work+m*n;
      if (k) {
        cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,m);CHKERRCUBLAS(cberr);
        ierr = cudaMemcpy(X->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRQ(ierr);
      } else {
        ierr = PetscMemzero(X->work,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      ierr = PetscMPIIntCast(m*n,&len);CHKERRQ(ierr);
      ierr = MPI_Allreduce(X->work,CC,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)X));CHKERRQ(ierr);
      for (j=0;j<n;j++) {
        ierr = PetscMemcpy(C+j*ldm,CC+j*m,m*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    }
  } else {
    if (k) {
      ierr = BVAllocateWork_Private(X,m*n);CHKERRQ(ierr);
      cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,m);CHKERRCUBLAS(cberr);
      ierr = cudaMemcpy(X->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRQ(ierr);
      for (j=0;j<n;j++) {
        ierr = PetscMemcpy(C+j*ldm,X->work+j*m,m*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    }
  }
  ierr = cudaFree(d_work);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(M,&pm);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(x->v,&d_px);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(y->v,&d_py);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*m*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    y := A'*x
*/
PetscErrorCode BVDotVec_Svec_CUDA(BV X,Vec y,PetscScalar *q)
{
  PetscErrorCode    ierr;
  BV_SVEC           *x = (BV_SVEC*)X->data;
  const PetscScalar *d_A,*d_x,*d_px,*d_py;
  PetscScalar       *d_y,*d_work,szero=0.0,sone=1.0,*qq=q;
  PetscBLASInt      n,k,one=1,len;
  Vec               z = y;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  n = X->n;
  k = X->k-X->l;
  if (X->matrix) {
    ierr = BV_IPMatMult(X,y);CHKERRQ(ierr);
    z = X->Bx;
  }
  ierr = VecCUDAGetArrayRead(x->v,&d_px);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(z,&d_py);CHKERRQ(ierr);
  if (!q) { ierr = VecGetArray(X->buffer,&qq);CHKERRQ(ierr); }
  d_A = d_px+(X->nc+X->l)*X->n;
  d_x = d_py;
  if (x->mpi) {
    ierr = BVAllocateWork_Private(X,k);CHKERRQ(ierr);
    ierr = cudaMalloc((void**)&d_work,k*sizeof(PetscScalar));CHKERRQ(ierr);
    if (n) {
      cberr = cublasXgemv(cublasv2handle,CUBLAS_OP_C,n,k,&sone,d_A,n,d_x,one,&szero,d_work,one);CHKERRCUBLAS(cberr);
      ierr = cudaMemcpy(X->work,d_work,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRQ(ierr);
    } else {
      ierr = PetscMemzero(X->work,k*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = PetscMPIIntCast(k,&len);CHKERRQ(ierr);
    ierr = MPI_Allreduce(X->work,qq,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)X));CHKERRQ(ierr);
    ierr = cudaFree(d_work);CHKERRQ(ierr);
  } else {
    if (n) {
      ierr = cudaMalloc((void**)&d_y,k*sizeof(PetscScalar));CHKERRQ(ierr);
      cberr = cublasXgemv(cublasv2handle,CUBLAS_OP_C,n,k,&sone,d_A,n,d_x,one,&szero,d_y,one);CHKERRCUBLAS(cberr);
      ierr = cudaMemcpy(qq,d_y,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRQ(ierr);
      ierr = cudaFree(d_y);CHKERRQ(ierr);
    }
  }
  if (!q) { ierr = VecRestoreArray(X->buffer,&qq);CHKERRQ(ierr); }
  ierr = VecCUDARestoreArrayRead(z,&d_py);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(x->v,&d_px);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    y := A'*x
*/
PetscErrorCode BVDotVec_Local_Svec_CUDA(BV X,Vec y,PetscScalar *m)
{
  PetscErrorCode    ierr;
  BV_SVEC           *x = (BV_SVEC*)X->data;
  const PetscScalar *d_A,*d_x,*d_px,*d_py;
  PetscScalar       *d_y,szero=0.0,sone=1.0;
  PetscBLASInt      n,k,one=1;
  Vec               z = y;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  n = X->n;
  k = X->k-X->l;
  if (X->matrix) {
    ierr = BV_IPMatMult(X,y);CHKERRQ(ierr);
    z = X->Bx;
  }
  ierr = VecCUDAGetArrayRead(x->v,&d_px);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(z,&d_py);CHKERRQ(ierr);
  d_A = d_px+(X->nc+X->l)*X->n;
  d_x = d_py;
  if (n) {
    ierr = cudaMalloc((void**)&d_y,k*sizeof(PetscScalar));CHKERRQ(ierr);
    cberr = cublasXgemv(cublasv2handle,CUBLAS_OP_C,n,k,&sone,d_A,n,d_x,one,&szero,d_y,one);CHKERRCUBLAS(cberr);
    ierr = cudaMemcpy(m,d_y,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  }
  ierr = cudaFree(d_y);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(z,&d_py);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(x->v,&d_px);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Scale n scalars
*/
PetscErrorCode BVScale_Svec_CUDA(BV bv,PetscInt j,PetscScalar alpha)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *d_array, *d_A;
  PetscBLASInt   n,one=1;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayReadWrite(ctx->v,&d_array);CHKERRQ(ierr);
  if (j<0) {
    d_A = d_array+(bv->nc+bv->l)*bv->n;
    n = (bv->k-bv->l)*bv->n;
  } else {
    d_A = d_array+(bv->nc+j)*bv->n;
    n = bv->n;
  }
  if (alpha == (PetscScalar)0.0) {
    ierr = cudaMemset(d_A,0,n*sizeof(PetscScalar));CHKERRQ(ierr);
  } else if (alpha != (PetscScalar)1.0) {
    cberr = cublasXscal(cublasv2handle,n,&alpha,d_A,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
  }
  ierr = VecCUDARestoreArrayReadWrite(ctx->v,&d_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode BVMatMult_Svec_CUDA(BV V,Mat A,BV W)
{
  PetscErrorCode    ierr;
  BV_SVEC           *v = (BV_SVEC*)V->data,*w = (BV_SVEC*)W->data;
  const PetscScalar *d_pv;
  PetscScalar       *d_pw;
  PetscInt          j;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayRead(v->v,&d_pv);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayWrite(w->v,&d_pw);CHKERRQ(ierr);
  for (j=0;j<V->k-V->l;j++) {
    ierr = VecCUDAPlaceArray(V->cv[1],(PetscScalar *)d_pv+(V->nc+V->l+j)*V->n);CHKERRQ(ierr);
    ierr = VecCUDAPlaceArray(W->cv[1],d_pw+(W->nc+W->l+j)*W->n);CHKERRQ(ierr);
    ierr = MatMult(A,V->cv[1],W->cv[1]);CHKERRQ(ierr);
    ierr = VecCUDAResetArray(V->cv[1]);CHKERRQ(ierr);
    ierr = VecCUDAResetArray(W->cv[1]);CHKERRQ(ierr);
  }
  ierr = VecCUDARestoreArrayRead(v->v,&d_pv);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(w->v,&d_pw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopy_Svec_CUDA(BV V,BV W)
{
  PetscErrorCode    ierr;
  BV_SVEC           *v = (BV_SVEC*)V->data,*w = (BV_SVEC*)W->data;
  const PetscScalar *d_pv,*d_pvc;
  PetscScalar       *d_pw,*d_pwc;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayRead(v->v,&d_pv);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayWrite(w->v,&d_pw);CHKERRQ(ierr);
  d_pvc = d_pv+(V->nc+V->l)*V->n;
  d_pwc = d_pw+(W->nc+W->l)*W->n;
  err = cudaMemcpy(d_pwc,d_pvc,(V->k-V->l)*V->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
  ierr = VecCUDARestoreArrayRead(v->v,&d_pv);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(w->v,&d_pw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode BVResize_Svec_CUDA(BV bv,PetscInt m,PetscBool copy)
{
  PetscErrorCode    ierr;
  BV_SVEC           *ctx = (BV_SVEC*)bv->data;
  const PetscScalar *d_pv;
  PetscScalar       *d_pnew;
  PetscInt          bs;
  Vec               vnew;
  char              str[50];
  cudaError_t       err;

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
    ierr = VecCUDAGetArrayRead(ctx->v,&d_pv);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWrite(vnew,&d_pnew);CHKERRQ(ierr);
    err = cudaMemcpy(d_pnew,d_pv,PetscMin(m,bv->m)*bv->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
    ierr = VecCUDARestoreArrayRead(ctx->v,&d_pv);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayWrite(vnew,&d_pnew);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&ctx->v);CHKERRQ(ierr);
  ctx->v = vnew;
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetColumn_Svec_CUDA(BV bv,PetscInt j,Vec *v)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *d_pv;
  PetscInt       l;

  PetscFunctionBegin;
  l = BVAvailableVec;
  ierr = VecCUDAGetArrayReadWrite(ctx->v,&d_pv);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(bv->cv[l],d_pv+(bv->nc+j)*bv->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreColumn_Svec_CUDA(BV bv,PetscInt j,Vec *v)
{
  PetscErrorCode ierr;
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscInt       l;

  PetscFunctionBegin;
  l = (j==bv->ci[0])? 0: 1;
  ierr = VecCUDAResetArray(bv->cv[l]);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadWrite(ctx->v,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

