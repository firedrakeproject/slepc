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
#include "../src/sys/classes/bv/impls/svec/svec.h"
#include <slepccublas.h>

#if defined(PETSC_USE_COMPLEX)
#include <thrust/device_ptr.h>
#endif

#define BLOCKSIZE 64

/*
    B := alpha*A + beta*B

    A,B are nxk (ld=n)
 */
static PetscErrorCode BVAXPY_BLAS_CUDA(BV bv,PetscInt n_,PetscInt k_,PetscScalar alpha,const PetscScalar *d_A,PetscScalar beta,PetscScalar *d_B)
{
  PetscCuBLASInt m=0,one=1;
  cublasHandle_t cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscCuBLASIntCast(n_*k_,&m));
  PetscCall(PetscLogGpuTimeBegin());
  if (beta!=(PetscScalar)1.0) {
    PetscCallCUBLAS(cublasXscal(cublasv2handle,m,&beta,d_B,one));
    PetscCall(PetscLogGpuFlops(1.0*m));
  }
  PetscCallCUBLAS(cublasXaxpy(cublasv2handle,m,&alpha,d_A,one,d_B,one));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0*m));
  PetscFunctionReturn(0);
}

/*
    C := alpha*A*Q + beta*C
*/
PetscErrorCode BVMult_Svec_CUDA(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  BV_SVEC           *y = (BV_SVEC*)Y->data,*x = (BV_SVEC*)X->data;
  const PetscScalar *d_px,*d_A,*q;
  PetscScalar       *d_py,*d_q,*d_B,*d_C;
  PetscInt          ldq,mq;
  PetscCuBLASInt    m=0,n=0,k=0,ldq_=0;
  cublasHandle_t    cublasv2handle;
  PetscBool         matiscuda;

  PetscFunctionBegin;
  if (!Y->n) PetscFunctionReturn(0);
  PetscCall(VecCUDAGetArrayRead(x->v,&d_px));
  if (beta==(PetscScalar)0.0) PetscCall(VecCUDAGetArrayWrite(y->v,&d_py));
  else PetscCall(VecCUDAGetArray(y->v,&d_py));
  d_A = d_px+(X->nc+X->l)*X->n;
  d_C = d_py+(Y->nc+Y->l)*Y->n;
  if (Q) {
    PetscCall(PetscCuBLASIntCast(Y->n,&m));
    PetscCall(PetscCuBLASIntCast(Y->k-Y->l,&n));
    PetscCall(PetscCuBLASIntCast(X->k-X->l,&k));
    PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
    PetscCall(MatGetSize(Q,NULL,&mq));
    PetscCall(MatDenseGetLDA(Q,&ldq));
    PetscCall(PetscCuBLASIntCast(ldq,&ldq_));
    PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSECUDA,&matiscuda));
    if (matiscuda) PetscCall(MatDenseCUDAGetArrayRead(Q,(const PetscScalar**)&d_q));
    else {
      PetscCall(MatDenseGetArrayRead(Q,&q));
      PetscCallCUDA(cudaMalloc((void**)&d_q,ldq*mq*sizeof(PetscScalar)));
      PetscCallCUDA(cudaMemcpy(d_q,q,ldq*mq*sizeof(PetscScalar),cudaMemcpyHostToDevice));
      PetscCall(PetscLogCpuToGpu(ldq*mq*sizeof(PetscScalar)));
    }
    d_B = d_q+Y->l*ldq+X->l;
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,d_A,m,d_B,ldq_,&beta,d_C,m));
    PetscCall(PetscLogGpuTimeEnd());
    if (matiscuda) PetscCall(MatDenseCUDARestoreArrayRead(Q,(const PetscScalar**)&d_q));
    else {
      PetscCall(MatDenseRestoreArrayRead(Q,&q));
      PetscCallCUDA(cudaFree(d_q));
    }
    PetscCall(PetscLogGpuFlops(2.0*m*n*k));
  } else PetscCall(BVAXPY_BLAS_CUDA(Y,Y->n,Y->k-Y->l,alpha,d_A,beta,d_C));
  PetscCall(VecCUDARestoreArrayRead(x->v,&d_px));
  PetscCall(VecCUDARestoreArrayWrite(y->v,&d_py));
  PetscFunctionReturn(0);
}

/*
    y := alpha*A*x + beta*y
*/
PetscErrorCode BVMultVec_Svec_CUDA(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  BV_SVEC           *x = (BV_SVEC*)X->data;
  const PetscScalar *d_px,*d_A;
  PetscScalar       *d_py,*d_q,*d_x,*d_y;
  PetscCuBLASInt    n=0,k=0,one=1;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCuBLASIntCast(X->n,&n));
  PetscCall(PetscCuBLASIntCast(X->k-X->l,&k));
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(VecCUDAGetArrayRead(x->v,&d_px));
  if (beta==(PetscScalar)0.0) PetscCall(VecCUDAGetArrayWrite(y,&d_py));
  else PetscCall(VecCUDAGetArray(y,&d_py));
  if (!q) PetscCall(VecCUDAGetArray(X->buffer,&d_q));
  else {
    PetscCallCUDA(cudaMalloc((void**)&d_q,k*sizeof(PetscScalar)));
    PetscCallCUDA(cudaMemcpy(d_q,q,k*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCall(PetscLogCpuToGpu(k*sizeof(PetscScalar)));
  }
  d_A = d_px+(X->nc+X->l)*X->n;
  d_x = d_q;
  d_y = d_py;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUBLAS(cublasXgemv(cublasv2handle,CUBLAS_OP_N,n,k,&alpha,d_A,n,d_x,one,&beta,d_y,one));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArrayRead(x->v,&d_px));
  if (beta==(PetscScalar)0.0) PetscCall(VecCUDARestoreArrayWrite(y,&d_py));
  else PetscCall(VecCUDARestoreArray(y,&d_py));
  if (!q) PetscCall(VecCUDARestoreArray(X->buffer,&d_q));
  else PetscCallCUDA(cudaFree(d_q));
  PetscCall(PetscLogGpuFlops(2.0*n*k));
  PetscFunctionReturn(0);
}

/*
    A(:,s:e-1) := A*B(:,s:e-1)
*/
PetscErrorCode BVMultInPlace_Svec_CUDA(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_SVEC           *ctx = (BV_SVEC*)V->data;
  PetscScalar       *d_pv,*d_q,*d_A,*d_B,*d_work,sone=1.0,szero=0.0;
  const PetscScalar *q;
  PetscInt          ldq,nq;
  PetscCuBLASInt    m=0,n=0,k=0,l=0,ldq_=0,bs=BLOCKSIZE;
  size_t            freemem,totmem;
  cublasHandle_t    cublasv2handle;
  PetscBool         matiscuda;

  PetscFunctionBegin;
  if (!V->n) PetscFunctionReturn(0);
  PetscCall(PetscCuBLASIntCast(V->n,&m));
  PetscCall(PetscCuBLASIntCast(e-s,&n));
  PetscCall(PetscCuBLASIntCast(V->k-V->l,&k));
  PetscCall(MatGetSize(Q,NULL,&nq));
  PetscCall(MatDenseGetLDA(Q,&ldq));
  PetscCall(PetscCuBLASIntCast(ldq,&ldq_));
  PetscCall(VecCUDAGetArray(ctx->v,&d_pv));
  PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSECUDA,&matiscuda));
  if (matiscuda) PetscCall(MatDenseCUDAGetArrayRead(Q,(const PetscScalar**)&d_q));
  else {
    PetscCall(MatDenseGetArrayRead(Q,&q));
    PetscCallCUDA(cudaMalloc((void**)&d_q,ldq*nq*sizeof(PetscScalar)));
    PetscCallCUDA(cudaMemcpy(d_q,q,ldq*nq*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCall(PetscLogCpuToGpu(ldq*nq*sizeof(PetscScalar)));
  }
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscLogGpuTimeBegin());
  /* try to allocate the whole matrix */
  PetscCallCUDA(cudaMemGetInfo(&freemem,&totmem));
  if (freemem>=m*n*sizeof(PetscScalar)) {
    PetscCallCUDA(cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar)));
    d_A = d_pv+(V->nc+V->l)*m;
    d_B = d_q+V->l*ldq+V->l+(s-V->l)*ldq;
    PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&sone,d_A,m,d_B,ldq_,&szero,d_work,m));
    PetscCallCUDA(cudaMemcpy2D(d_A+(s-V->l)*m,m*sizeof(PetscScalar),d_work,m*sizeof(PetscScalar),m*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
  } else {
    PetscCall(PetscCuBLASIntCast(freemem/(m*sizeof(PetscScalar)),&bs));
    PetscCallCUDA(cudaMalloc((void**)&d_work,bs*n*sizeof(PetscScalar)));
    PetscCall(PetscCuBLASIntCast(m % bs,&l));
    if (l) {
      d_A = d_pv+(V->nc+V->l)*m;
      d_B = d_q+V->l*ldq+V->l+(s-V->l)*ldq;
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,l,n,k,&sone,d_A,m,d_B,ldq_,&szero,d_work,l));
      PetscCallCUDA(cudaMemcpy2D(d_A+(s-V->l)*m,m*sizeof(PetscScalar),d_work,l*sizeof(PetscScalar),l*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
    }
    for (;l<m;l+=bs) {
      d_A = d_pv+(V->nc+V->l)*m+l;
      d_B = d_q+V->l*ldq+V->l+(s-V->l)*ldq;
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,bs,n,k,&sone,d_A,m,d_B,ldq_,&szero,d_work,bs));
      PetscCallCUDA(cudaMemcpy2D(d_A+(s-V->l)*m,m*sizeof(PetscScalar),d_work,bs*sizeof(PetscScalar),bs*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  if (matiscuda) PetscCall(MatDenseCUDARestoreArrayRead(Q,(const PetscScalar**)&d_q));
  else {
    PetscCall(MatDenseRestoreArrayRead(Q,&q));
    PetscCallCUDA(cudaFree(d_q));
  }
  PetscCallCUDA(cudaFree(d_work));
  PetscCall(VecCUDARestoreArray(ctx->v,&d_pv));
  PetscCall(PetscLogGpuFlops(2.0*m*n*k));
  PetscFunctionReturn(0);
}

/*
    A(:,s:e-1) := A*B(:,s:e-1)
*/
PetscErrorCode BVMultInPlaceHermitianTranspose_Svec_CUDA(BV V,Mat Q,PetscInt s,PetscInt e)
{
  BV_SVEC           *ctx = (BV_SVEC*)V->data;
  PetscScalar       *d_pv,*d_q,*d_A,*d_B,*d_work,sone=1.0,szero=0.0;
  const PetscScalar *q;
  PetscInt          ldq,nq;
  PetscCuBLASInt    m=0,n=0,k=0,ldq_=0;
  cublasHandle_t    cublasv2handle;
  PetscBool         matiscuda;

  PetscFunctionBegin;
  if (!V->n) PetscFunctionReturn(0);
  PetscCall(PetscCuBLASIntCast(V->n,&m));
  PetscCall(PetscCuBLASIntCast(e-s,&n));
  PetscCall(PetscCuBLASIntCast(V->k-V->l,&k));
  PetscCall(MatGetSize(Q,NULL,&nq));
  PetscCall(MatDenseGetLDA(Q,&ldq));
  PetscCall(PetscCuBLASIntCast(ldq,&ldq_));
  PetscCall(VecCUDAGetArray(ctx->v,&d_pv));
  PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSECUDA,&matiscuda));
  if (matiscuda) PetscCall(MatDenseCUDAGetArrayRead(Q,(const PetscScalar**)&d_q));
  else {
    PetscCall(MatDenseGetArrayRead(Q,&q));
    PetscCallCUDA(cudaMalloc((void**)&d_q,ldq*nq*sizeof(PetscScalar)));
    PetscCallCUDA(cudaMemcpy(d_q,q,ldq*nq*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCall(PetscLogCpuToGpu(ldq*nq*sizeof(PetscScalar)));
  }
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUDA(cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar)));
  d_A = d_pv+(V->nc+V->l)*m;
  d_B = d_q+V->l*ldq+s;
  PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_C,m,n,k,&sone,d_A,m,d_B,ldq_,&szero,d_work,m));
  PetscCallCUDA(cudaMemcpy2D(d_A+(s-V->l)*m,m*sizeof(PetscScalar),d_work,m*sizeof(PetscScalar),m*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
  PetscCall(PetscLogGpuTimeEnd());
  if (matiscuda) PetscCall(MatDenseCUDARestoreArrayRead(Q,(const PetscScalar**)&d_q));
  else {
    PetscCall(MatDenseRestoreArrayRead(Q,&q));
    PetscCallCUDA(cudaFree(d_q));
  }
  PetscCallCUDA(cudaFree(d_work));
  PetscCall(VecCUDARestoreArray(ctx->v,&d_pv));
  PetscCall(PetscLogGpuFlops(2.0*m*n*k));
  PetscFunctionReturn(0);
}

/*
    C := A'*B
*/
PetscErrorCode BVDot_Svec_CUDA(BV X,BV Y,Mat M)
{
  BV_SVEC           *x = (BV_SVEC*)X->data,*y = (BV_SVEC*)Y->data;
  const PetscScalar *d_px,*d_py,*d_A,*d_B;
  PetscScalar       *pm,*d_work,sone=1.0,szero=0.0,*C,*CC;
  PetscInt          j,ldm;
  PetscCuBLASInt    m=0,n=0,k=0,ldm_=0;
  PetscMPIInt       len;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCuBLASIntCast(Y->k-Y->l,&m));
  PetscCall(PetscCuBLASIntCast(X->k-X->l,&n));
  PetscCall(PetscCuBLASIntCast(X->n,&k));
  PetscCall(MatDenseGetLDA(M,&ldm));
  PetscCall(PetscCuBLASIntCast(ldm,&ldm_));
  PetscCall(VecCUDAGetArrayRead(x->v,&d_px));
  PetscCall(VecCUDAGetArrayRead(y->v,&d_py));
  PetscCall(MatDenseGetArrayWrite(M,&pm));
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCallCUDA(cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar)));
  d_A = d_py+(Y->nc+Y->l)*Y->n;
  d_B = d_px+(X->nc+X->l)*X->n;
  C = pm+X->l*ldm+Y->l;
  if (x->mpi) {
    if (ldm==m) {
      PetscCall(BVAllocateWork_Private(X,m*n));
      if (k) {
        PetscCall(PetscLogGpuTimeBegin());
        PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,ldm_));
        PetscCall(PetscLogGpuTimeEnd());
        PetscCallCUDA(cudaMemcpy(X->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
        PetscCall(PetscLogGpuToCpu(m*n*sizeof(PetscScalar)));
      } else PetscCall(PetscArrayzero(X->work,m*n));
      PetscCall(PetscMPIIntCast(m*n,&len));
      PetscCall(MPIU_Allreduce(X->work,C,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)X)));
    } else {
      PetscCall(BVAllocateWork_Private(X,2*m*n));
      CC = X->work+m*n;
      if (k) {
        PetscCall(PetscLogGpuTimeBegin());
        PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,m));
        PetscCall(PetscLogGpuTimeEnd());
        PetscCallCUDA(cudaMemcpy(X->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
        PetscCall(PetscLogGpuToCpu(m*n*sizeof(PetscScalar)));
      } else PetscCall(PetscArrayzero(X->work,m*n));
      PetscCall(PetscMPIIntCast(m*n,&len));
      PetscCall(MPIU_Allreduce(X->work,CC,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)X)));
      for (j=0;j<n;j++) PetscCall(PetscArraycpy(C+j*ldm,CC+j*m,m));
    }
  } else {
    if (k) {
      PetscCall(BVAllocateWork_Private(X,m*n));
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,m));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCallCUDA(cudaMemcpy(X->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      PetscCall(PetscLogGpuToCpu(m*n*sizeof(PetscScalar)));
      for (j=0;j<n;j++) PetscCall(PetscArraycpy(C+j*ldm,X->work+j*m,m));
    }
  }
  PetscCallCUDA(cudaFree(d_work));
  PetscCall(MatDenseRestoreArrayWrite(M,&pm));
  PetscCall(VecCUDARestoreArrayRead(x->v,&d_px));
  PetscCall(VecCUDARestoreArrayRead(y->v,&d_py));
  PetscCall(PetscLogGpuFlops(2.0*m*n*k));
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_COMPLEX)
struct conjugate
{
  __host__ __device__
    PetscScalar operator()(PetscScalar x)
    {
      return PetscConj(x);
    }
};

PetscErrorCode ConjugateCudaArray(PetscScalar *a, PetscInt n)
{
  thrust::device_ptr<PetscScalar> ptr;

  PetscFunctionBegin;
  try {
    ptr = thrust::device_pointer_cast(a);
    thrust::transform(ptr,ptr+n,ptr,conjugate());
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  PetscFunctionReturn(0);
}
#endif

/*
    y := A'*x computed as y' := x'*A
*/
PetscErrorCode BVDotVec_Svec_CUDA(BV X,Vec y,PetscScalar *q)
{
  BV_SVEC           *x = (BV_SVEC*)X->data;
  const PetscScalar *d_A,*d_x,*d_px,*d_py;
  PetscScalar       *d_work,szero=0.0,sone=1.0,*qq=q;
  PetscCuBLASInt    n=0,k=0,one=1;
  PetscMPIInt       len;
  Vec               z = y;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCuBLASIntCast(X->n,&n));
  PetscCall(PetscCuBLASIntCast(X->k-X->l,&k));
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  if (X->matrix) {
    PetscCall(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  PetscCall(VecCUDAGetArrayRead(x->v,&d_px));
  PetscCall(VecCUDAGetArrayRead(z,&d_py));
  if (!q) PetscCall(VecCUDAGetArrayWrite(X->buffer,&d_work));
  else PetscCallCUDA(cudaMalloc((void**)&d_work,k*sizeof(PetscScalar)));
  d_A = d_px+(X->nc+X->l)*X->n;
  d_x = d_py;
  if (x->mpi) {
    PetscCall(BVAllocateWork_Private(X,k));
    if (n) {
      PetscCall(PetscLogGpuTimeBegin());
#if defined(PETSC_USE_COMPLEX)
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,one,k,n,&sone,d_x,n,d_A,n,&szero,d_work,one));
      PetscCall(ConjugateCudaArray(d_work,k));
#else
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,one,k,n,&sone,d_x,one,d_A,n,&szero,d_work,one));
#endif
      PetscCall(PetscLogGpuTimeEnd());
      PetscCallCUDA(cudaMemcpy(X->work,d_work,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      PetscCall(PetscLogGpuToCpu(k*sizeof(PetscScalar)));
    } else PetscCall(PetscArrayzero(X->work,k));
    if (!q) {
      PetscCall(VecCUDARestoreArrayWrite(X->buffer,&d_work));
      PetscCall(VecGetArray(X->buffer,&qq));
    } else PetscCallCUDA(cudaFree(d_work));
    PetscCall(PetscMPIIntCast(k,&len));
    PetscCall(MPIU_Allreduce(X->work,qq,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)X)));
    if (!q) PetscCall(VecRestoreArray(X->buffer,&qq));
  } else {
    if (n) {
      PetscCall(PetscLogGpuTimeBegin());
#if defined(PETSC_USE_COMPLEX)
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,one,k,n,&sone,d_x,n,d_A,n,&szero,d_work,one));
      PetscCall(ConjugateCudaArray(d_work,k));
#else
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,one,k,n,&sone,d_x,one,d_A,n,&szero,d_work,one));
#endif
      PetscCall(PetscLogGpuTimeEnd());
    }
    if (!q) PetscCall(VecCUDARestoreArrayWrite(X->buffer,&d_work));
    else {
      PetscCallCUDA(cudaMemcpy(q,d_work,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      PetscCall(PetscLogGpuToCpu(k*sizeof(PetscScalar)));
      PetscCallCUDA(cudaFree(d_work));
    }
  }
  PetscCall(VecCUDARestoreArrayRead(z,&d_py));
  PetscCall(VecCUDARestoreArrayRead(x->v,&d_px));
  PetscCall(PetscLogGpuFlops(2.0*n*k));
  PetscFunctionReturn(0);
}

/*
    y := A'*x computed as y' := x'*A
*/
PetscErrorCode BVDotVec_Local_Svec_CUDA(BV X,Vec y,PetscScalar *m)
{
  BV_SVEC           *x = (BV_SVEC*)X->data;
  const PetscScalar *d_A,*d_x,*d_px,*d_py;
  PetscScalar       *d_y,szero=0.0,sone=1.0;
  PetscCuBLASInt    n=0,k=0,one=1;
  Vec               z = y;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCuBLASIntCast(X->n,&n));
  PetscCall(PetscCuBLASIntCast(X->k-X->l,&k));
  if (X->matrix) {
    PetscCall(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(VecCUDAGetArrayRead(x->v,&d_px));
  PetscCall(VecCUDAGetArrayRead(z,&d_py));
  d_A = d_px+(X->nc+X->l)*X->n;
  d_x = d_py;
  if (n) {
    PetscCallCUDA(cudaMalloc((void**)&d_y,k*sizeof(PetscScalar)));
    PetscCall(PetscLogGpuTimeBegin());
#if defined(PETSC_USE_COMPLEX)
    PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,one,k,n,&sone,d_x,n,d_A,n,&szero,d_y,one));
    PetscCall(ConjugateCudaArray(d_y,k));
#else
    PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,one,k,n,&sone,d_x,one,d_A,n,&szero,d_y,one));
#endif
    PetscCall(PetscLogGpuTimeEnd());
    PetscCallCUDA(cudaMemcpy(m,d_y,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    PetscCall(PetscLogGpuToCpu(k*sizeof(PetscScalar)));
    PetscCallCUDA(cudaFree(d_y));
  }
  PetscCall(VecCUDARestoreArrayRead(z,&d_py));
  PetscCall(VecCUDARestoreArrayRead(x->v,&d_px));
  PetscCall(PetscLogGpuFlops(2.0*n*k));
  PetscFunctionReturn(0);
}

/*
    Scale n scalars
*/
PetscErrorCode BVScale_Svec_CUDA(BV bv,PetscInt j,PetscScalar alpha)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *d_array, *d_A;
  PetscCuBLASInt n=0,one=1;
  cublasHandle_t cublasv2handle;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArray(ctx->v,&d_array));
  if (j<0) {
    d_A = d_array+(bv->nc+bv->l)*bv->n;
    PetscCall(PetscCuBLASIntCast((bv->k-bv->l)*bv->n,&n));
  } else {
    d_A = d_array+(bv->nc+j)*bv->n;
    PetscCall(PetscCuBLASIntCast(bv->n,&n));
  }
  if (alpha == (PetscScalar)0.0) PetscCallCUDA(cudaMemset(d_A,0,n*sizeof(PetscScalar)));
  else if (alpha != (PetscScalar)1.0) {
    PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXscal(cublasv2handle,n,&alpha,d_A,one));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(1.0*n));
  }
  PetscCall(VecCUDARestoreArray(ctx->v,&d_array));
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopyColumn_Svec_CUDA(BV V,PetscInt j,PetscInt i)
{
  BV_SVEC        *v = (BV_SVEC*)V->data;
  PetscScalar    *d_pv;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArray(v->v,&d_pv));
  PetscCallCUDA(cudaMemcpy(d_pv+(V->nc+i)*V->n,d_pv+(V->nc+j)*V->n,V->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
  PetscCall(VecCUDARestoreArray(v->v,&d_pv));
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreColumn_Svec_CUDA(BV bv,PetscInt j,Vec *v)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscInt       l;

  PetscFunctionBegin;
  l = (j==bv->ci[0])? 0: 1;
  PetscCall(VecCUDAResetArray(bv->cv[l]));
  PetscCall(VecCUDARestoreArray(ctx->v,NULL));
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}
