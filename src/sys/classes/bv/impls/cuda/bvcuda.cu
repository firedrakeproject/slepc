/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   CUDA-related code common to several BV impls
*/

#include <slepc/private/bvimpl.h>
#include <slepccublas.h>

#define BLOCKSIZE 64

/*
    C := alpha*A*B + beta*C
*/
PetscErrorCode BVMult_BLAS_CUDA(BV bv,PetscInt m_,PetscInt n_,PetscInt k_,PetscInt ldb_,PetscScalar alpha,const PetscScalar *d_A,const PetscScalar *d_B,PetscScalar beta,PetscScalar *d_C)
{
  PetscCuBLASInt    m=0,n=0,k=0,ldb=0;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  (void)bv; // avoid unused parameter warning
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscCuBLASIntCast(m_,&m));
  PetscCall(PetscCuBLASIntCast(n_,&n));
  PetscCall(PetscCuBLASIntCast(k_,&k));
  PetscCall(PetscCuBLASIntCast(ldb_,&ldb));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,d_A,m,d_B,ldb,&beta,d_C,m));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0*m*n*k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    y := alpha*A*x + beta*y
*/
PetscErrorCode BVMultVec_BLAS_CUDA(BV bv,PetscInt n_,PetscInt k_,PetscScalar alpha,const PetscScalar *d_A,const PetscScalar *d_x,PetscScalar beta,PetscScalar *d_y)
{
  PetscCuBLASInt    n=0,k=0,one=1;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  (void)bv; // avoid unused parameter warning
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscCuBLASIntCast(n_,&n));
  PetscCall(PetscCuBLASIntCast(k_,&k));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUBLAS(cublasXgemv(cublasv2handle,CUBLAS_OP_N,n,k,&alpha,d_A,n,d_x,one,&beta,d_y,one));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0*n*k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    A(:,s:e-1) := A*B(:,s:e-1)
*/
PetscErrorCode BVMultInPlace_BLAS_CUDA(BV bv,PetscInt m_,PetscInt k_,PetscInt ldb_,PetscInt s,PetscInt e,PetscScalar *d_A,const PetscScalar *d_B,PetscBool btrans)
{
  const PetscScalar *d_B1;
  PetscScalar       *d_work,sone=1.0,szero=0.0;
  PetscCuBLASInt    m=0,n=0,k=0,l=0,ldb=0,bs=BLOCKSIZE;
  size_t            freemem,totmem;
  cublasHandle_t    cublasv2handle;
  cublasOperation_t bt;

  PetscFunctionBegin;
  (void)bv; // avoid unused parameter warning
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscCuBLASIntCast(m_,&m));
  PetscCall(PetscCuBLASIntCast(e-s,&n));
  PetscCall(PetscCuBLASIntCast(k_,&k));
  PetscCall(PetscCuBLASIntCast(ldb_,&ldb));
  PetscCall(PetscLogGpuTimeBegin());
  if (PetscUnlikely(btrans)) {
    d_B1 = d_B+s;
    bt   = CUBLAS_OP_C;
  } else {
    d_B1 = d_B+s*ldb;
    bt   = CUBLAS_OP_N;
  }
  /* try to allocate the whole matrix */
  PetscCallCUDA(cudaMemGetInfo(&freemem,&totmem));
  if (freemem>=m*n*sizeof(PetscScalar)) {
    PetscCallCUDA(cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar)));
    PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,bt,m,n,k,&sone,d_A,m,d_B1,ldb,&szero,d_work,m));
    PetscCallCUDA(cudaMemcpy2D(d_A+s*m,m*sizeof(PetscScalar),d_work,m*sizeof(PetscScalar),m*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
  } else {
    PetscCall(PetscCuBLASIntCast(freemem/(m*sizeof(PetscScalar)),&bs));
    PetscCallCUDA(cudaMalloc((void**)&d_work,bs*n*sizeof(PetscScalar)));
    PetscCall(PetscCuBLASIntCast(m % bs,&l));
    if (l) {
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,bt,l,n,k,&sone,d_A,m,d_B1,ldb,&szero,d_work,l));
      PetscCallCUDA(cudaMemcpy2D(d_A+s*m,m*sizeof(PetscScalar),d_work,l*sizeof(PetscScalar),l*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
    }
    for (;l<m;l+=bs) {
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,bt,bs,n,k,&sone,d_A+l,m,d_B1,ldb,&szero,d_work,bs));
      PetscCallCUDA(cudaMemcpy2D(d_A+l+s*m,m*sizeof(PetscScalar),d_work,bs*sizeof(PetscScalar),bs*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCallCUDA(cudaFree(d_work));
  PetscCall(PetscLogGpuFlops(2.0*m*n*k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    B := alpha*A + beta*B
*/
PetscErrorCode BVAXPY_BLAS_CUDA(BV bv,PetscInt n_,PetscInt k_,PetscScalar alpha,const PetscScalar *d_A,PetscScalar beta,PetscScalar *d_B)
{
  PetscCuBLASInt m=0,one=1;
  cublasHandle_t cublasv2handle;

  PetscFunctionBegin;
  (void)bv; // avoid unused parameter warning
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    C := A'*B

    C is a CPU array
*/
PetscErrorCode BVDot_BLAS_CUDA(BV bv,PetscInt m_,PetscInt n_,PetscInt k_,PetscInt ldc_,const PetscScalar *d_A,const PetscScalar *d_B,PetscScalar *C,PetscBool mpi)
{
  PetscScalar       *d_work,sone=1.0,szero=0.0,*CC;
  PetscInt          j;
  PetscCuBLASInt    m=0,n=0,k=0,ldc=0;
  PetscMPIInt       len;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscCuBLASIntCast(m_,&m));
  PetscCall(PetscCuBLASIntCast(n_,&n));
  PetscCall(PetscCuBLASIntCast(k_,&k));
  PetscCall(PetscCuBLASIntCast(ldc_,&ldc));
  PetscCallCUDA(cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar)));
  if (mpi) {
    if (ldc==m) {
      PetscCall(BVAllocateWork_Private(bv,m*n));
      if (k) {
        PetscCall(PetscLogGpuTimeBegin());
        PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,ldc));
        PetscCall(PetscLogGpuTimeEnd());
        PetscCallCUDA(cudaMemcpy(bv->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
        PetscCall(PetscLogGpuToCpu(m*n*sizeof(PetscScalar)));
      } else PetscCall(PetscArrayzero(bv->work,m*n));
      PetscCall(PetscMPIIntCast(m*n,&len));
      PetscCall(MPIU_Allreduce(bv->work,C,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)bv)));
    } else {
      PetscCall(BVAllocateWork_Private(bv,2*m*n));
      CC = bv->work+m*n;
      if (k) {
        PetscCall(PetscLogGpuTimeBegin());
        PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,m));
        PetscCall(PetscLogGpuTimeEnd());
        PetscCallCUDA(cudaMemcpy(bv->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
        PetscCall(PetscLogGpuToCpu(m*n*sizeof(PetscScalar)));
      } else PetscCall(PetscArrayzero(bv->work,m*n));
      PetscCall(PetscMPIIntCast(m*n,&len));
      PetscCall(MPIU_Allreduce(bv->work,CC,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)bv)));
      for (j=0;j<n;j++) PetscCall(PetscArraycpy(C+j*ldc,CC+j*m,m));
    }
  } else {
    if (k) {
      PetscCall(BVAllocateWork_Private(bv,m*n));
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,m));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCallCUDA(cudaMemcpy(bv->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      PetscCall(PetscLogGpuToCpu(m*n*sizeof(PetscScalar)));
      for (j=0;j<n;j++) PetscCall(PetscArraycpy(C+j*ldc,bv->work+j*m,m));
    }
  }
  PetscCallCUDA(cudaFree(d_work));
  PetscCall(PetscLogGpuFlops(2.0*m*n*k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    y := A'*x computed as y' := x'*A

    y is a CPU array, if NULL bv->buffer is used as a workspace
*/
PetscErrorCode BVDotVec_BLAS_CUDA(BV bv,PetscInt n_,PetscInt k_,const PetscScalar *d_A,const PetscScalar *d_x,PetscScalar *y,PetscBool mpi)
{
  PetscScalar       *d_work,szero=0.0,sone=1.0,*yy=y;
  PetscCuBLASInt    n=0,k=0,one=1;
  PetscMPIInt       len;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscCuBLASIntCast(n_,&n));
  PetscCall(PetscCuBLASIntCast(k_,&k));
  if (!y) PetscCall(VecCUDAGetArrayWrite(bv->buffer,&d_work));
  else PetscCallCUDA(cudaMalloc((void**)&d_work,k*sizeof(PetscScalar)));
  if (mpi) {
    PetscCall(BVAllocateWork_Private(bv,k));
    if (n) {
      PetscCall(PetscLogGpuTimeBegin());
#if defined(PETSC_USE_COMPLEX)
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,one,k,n,&sone,d_x,n,d_A,n,&szero,d_work,one));
      PetscCall(BV_ConjugateCudaArray(d_work,k));
#else
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,one,k,n,&sone,d_x,one,d_A,n,&szero,d_work,one));
#endif
      PetscCall(PetscLogGpuTimeEnd());
      PetscCallCUDA(cudaMemcpy(bv->work,d_work,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      PetscCall(PetscLogGpuToCpu(k*sizeof(PetscScalar)));
    } else PetscCall(PetscArrayzero(bv->work,k));
    if (!y) {
      PetscCall(VecCUDARestoreArrayWrite(bv->buffer,&d_work));
      PetscCall(VecGetArray(bv->buffer,&yy));
    } else PetscCallCUDA(cudaFree(d_work));
    PetscCall(PetscMPIIntCast(k,&len));
    PetscCall(MPIU_Allreduce(bv->work,yy,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)bv)));
  } else {
    if (n) {
      PetscCall(PetscLogGpuTimeBegin());
#if defined(PETSC_USE_COMPLEX)
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,one,k,n,&sone,d_x,n,d_A,n,&szero,d_work,one));
      PetscCall(BV_ConjugateCudaArray(d_work,k));
#else
      PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,one,k,n,&sone,d_x,one,d_A,n,&szero,d_work,one));
#endif
      PetscCall(PetscLogGpuTimeEnd());
    }
    if (!y) PetscCall(VecCUDARestoreArrayWrite(bv->buffer,&d_work));
    else {
      PetscCallCUDA(cudaMemcpy(y,d_work,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      PetscCall(PetscLogGpuToCpu(k*sizeof(PetscScalar)));
      PetscCallCUDA(cudaFree(d_work));
    }
  }
  PetscCall(PetscLogGpuFlops(2.0*n*k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Scale n scalars
*/
PetscErrorCode BVScale_BLAS_CUDA(BV bv,PetscInt n_,PetscScalar *d_A,PetscScalar alpha)
{
  PetscCuBLASInt n=0,one=1;
  cublasHandle_t cublasv2handle;

  PetscFunctionBegin;
  (void)bv; // avoid unused parameter warning
  PetscCall(PetscCuBLASIntCast(n_,&n));
  if (PetscUnlikely(alpha == (PetscScalar)0.0)) PetscCallCUDA(cudaMemset(d_A,0,n*sizeof(PetscScalar)));
  else if (alpha != (PetscScalar)1.0) {
    PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXscal(cublasv2handle,n,&alpha,d_A,one));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(1.0*n));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_COMPLEX)
#include <thrust/device_ptr.h>

struct conjugate
{
  __host__ __device__
    PetscScalar operator()(PetscScalar x)
    {
      return PetscConj(x);
    }
};

PetscErrorCode BV_ConjugateCudaArray(PetscScalar *a,PetscInt n)
{
  thrust::device_ptr<PetscScalar> ptr;

  PetscFunctionBegin;
  try {
    ptr = thrust::device_pointer_cast(a);
    thrust::transform(ptr,ptr+n,ptr,conjugate());
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
