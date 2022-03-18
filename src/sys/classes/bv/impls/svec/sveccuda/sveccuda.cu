/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscCuBLASIntCast(n_*k_,&m));
  CHKERRQ(PetscLogGpuTimeBegin());
  if (beta!=(PetscScalar)1.0) {
    CHKERRCUBLAS(cublasXscal(cublasv2handle,m,&beta,d_B,one));
    CHKERRQ(PetscLogGpuFlops(1.0*m));
  }
  CHKERRCUBLAS(cublasXaxpy(cublasv2handle,m,&alpha,d_A,one,d_B,one));
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(PetscLogGpuFlops(2.0*m));
  PetscFunctionReturn(0);
}

/*
    C := alpha*A*B + beta*C
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
  CHKERRQ(VecCUDAGetArrayRead(x->v,&d_px));
  if (beta==(PetscScalar)0.0) {
    CHKERRQ(VecCUDAGetArrayWrite(y->v,&d_py));
  } else {
    CHKERRQ(VecCUDAGetArray(y->v,&d_py));
  }
  d_A = d_px+(X->nc+X->l)*X->n;
  d_C = d_py+(Y->nc+Y->l)*Y->n;
  if (Q) {
    CHKERRQ(PetscCuBLASIntCast(Y->n,&m));
    CHKERRQ(PetscCuBLASIntCast(Y->k-Y->l,&n));
    CHKERRQ(PetscCuBLASIntCast(X->k-X->l,&k));
    CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
    CHKERRQ(MatGetSize(Q,&ldq,&mq));
    CHKERRQ(PetscCuBLASIntCast(ldq,&ldq_));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSECUDA,&matiscuda));
    if (matiscuda) {
      CHKERRQ(MatDenseCUDAGetArrayRead(Q,(const PetscScalar**)&d_q));
    } else {
      CHKERRQ(MatDenseGetArrayRead(Q,&q));
      CHKERRCUDA(cudaMalloc((void**)&d_q,ldq*mq*sizeof(PetscScalar)));
      CHKERRCUDA(cudaMemcpy(d_q,q,ldq*mq*sizeof(PetscScalar),cudaMemcpyHostToDevice));
      CHKERRQ(PetscLogCpuToGpu(ldq*mq*sizeof(PetscScalar)));
    }
    d_B = d_q+Y->l*ldq+X->l;
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,d_A,m,d_B,ldq_,&beta,d_C,m));
    CHKERRQ(PetscLogGpuTimeEnd());
    if (matiscuda) {
      CHKERRQ(MatDenseCUDARestoreArrayRead(Q,(const PetscScalar**)&d_q));
    } else {
      CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
      CHKERRCUDA(cudaFree(d_q));
    }
    CHKERRQ(PetscLogGpuFlops(2.0*m*n*k));
  } else {
    CHKERRQ(BVAXPY_BLAS_CUDA(Y,Y->n,Y->k-Y->l,alpha,d_A,beta,d_C));
  }
  CHKERRQ(VecCUDARestoreArrayRead(x->v,&d_px));
  CHKERRQ(VecCUDARestoreArrayWrite(y->v,&d_py));
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
  CHKERRQ(PetscCuBLASIntCast(X->n,&n));
  CHKERRQ(PetscCuBLASIntCast(X->k-X->l,&k));
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(VecCUDAGetArrayRead(x->v,&d_px));
  if (beta==(PetscScalar)0.0) {
    CHKERRQ(VecCUDAGetArrayWrite(y,&d_py));
  } else {
    CHKERRQ(VecCUDAGetArray(y,&d_py));
  }
  if (!q) {
    CHKERRQ(VecCUDAGetArray(X->buffer,&d_q));
  } else {
    CHKERRCUDA(cudaMalloc((void**)&d_q,k*sizeof(PetscScalar)));
    CHKERRCUDA(cudaMemcpy(d_q,q,k*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    CHKERRQ(PetscLogCpuToGpu(k*sizeof(PetscScalar)));
  }
  d_A = d_px+(X->nc+X->l)*X->n;
  d_x = d_q;
  d_y = d_py;
  CHKERRQ(PetscLogGpuTimeBegin());
  CHKERRCUBLAS(cublasXgemv(cublasv2handle,CUBLAS_OP_N,n,k,&alpha,d_A,n,d_x,one,&beta,d_y,one));
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArrayRead(x->v,&d_px));
  if (beta==(PetscScalar)0.0) {
    CHKERRQ(VecCUDARestoreArrayWrite(y,&d_py));
  } else {
    CHKERRQ(VecCUDARestoreArray(y,&d_py));
  }
  if (!q) {
    CHKERRQ(VecCUDARestoreArray(X->buffer,&d_q));
  } else {
    CHKERRCUDA(cudaFree(d_q));
  }
  CHKERRQ(PetscLogGpuFlops(2.0*n*k));
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
  CHKERRQ(PetscCuBLASIntCast(V->n,&m));
  CHKERRQ(PetscCuBLASIntCast(e-s,&n));
  CHKERRQ(PetscCuBLASIntCast(V->k-V->l,&k));
  CHKERRQ(MatGetSize(Q,&ldq,&nq));
  CHKERRQ(PetscCuBLASIntCast(ldq,&ldq_));
  CHKERRQ(VecCUDAGetArray(ctx->v,&d_pv));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSECUDA,&matiscuda));
  if (matiscuda) {
    CHKERRQ(MatDenseCUDAGetArrayRead(Q,(const PetscScalar**)&d_q));
  } else {
    CHKERRQ(MatDenseGetArrayRead(Q,&q));
    CHKERRCUDA(cudaMalloc((void**)&d_q,ldq*nq*sizeof(PetscScalar)));
    CHKERRCUDA(cudaMemcpy(d_q,q,ldq*nq*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    CHKERRQ(PetscLogCpuToGpu(ldq*nq*sizeof(PetscScalar)));
  }
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscLogGpuTimeBegin());
  /* try to allocate the whole matrix */
  CHKERRCUDA(cudaMemGetInfo(&freemem,&totmem));
  if (freemem>=m*n*sizeof(PetscScalar)) {
    CHKERRCUDA(cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar)));
    d_A = d_pv+(V->nc+V->l)*m;
    d_B = d_q+V->l*ldq+V->l+(s-V->l)*ldq;
    CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&sone,d_A,m,d_B,ldq_,&szero,d_work,m));
    CHKERRCUDA(cudaMemcpy2D(d_A+(s-V->l)*m,m*sizeof(PetscScalar),d_work,m*sizeof(PetscScalar),m*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
  } else {
    CHKERRQ(PetscCuBLASIntCast(freemem/(m*sizeof(PetscScalar)),&bs));
    CHKERRCUDA(cudaMalloc((void**)&d_work,bs*n*sizeof(PetscScalar)));
    CHKERRQ(PetscCuBLASIntCast(m % bs,&l));
    if (l) {
      d_A = d_pv+(V->nc+V->l)*m;
      d_B = d_q+V->l*ldq+V->l+(s-V->l)*ldq;
      CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,l,n,k,&sone,d_A,m,d_B,ldq_,&szero,d_work,l));
      CHKERRCUDA(cudaMemcpy2D(d_A+(s-V->l)*m,m*sizeof(PetscScalar),d_work,l*sizeof(PetscScalar),l*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
    }
    for (;l<m;l+=bs) {
      d_A = d_pv+(V->nc+V->l)*m+l;
      d_B = d_q+V->l*ldq+V->l+(s-V->l)*ldq;
      CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,bs,n,k,&sone,d_A,m,d_B,ldq_,&szero,d_work,bs));
      CHKERRCUDA(cudaMemcpy2D(d_A+(s-V->l)*m,m*sizeof(PetscScalar),d_work,bs*sizeof(PetscScalar),bs*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
    }
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  if (matiscuda) {
    CHKERRQ(MatDenseCUDARestoreArrayRead(Q,(const PetscScalar**)&d_q));
  } else {
    CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
    CHKERRCUDA(cudaFree(d_q));
  }
  CHKERRCUDA(cudaFree(d_work));
  CHKERRQ(VecCUDARestoreArray(ctx->v,&d_pv));
  CHKERRQ(PetscLogGpuFlops(2.0*m*n*k));
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
  CHKERRQ(PetscCuBLASIntCast(V->n,&m));
  CHKERRQ(PetscCuBLASIntCast(e-s,&n));
  CHKERRQ(PetscCuBLASIntCast(V->k-V->l,&k));
  CHKERRQ(MatGetSize(Q,&ldq,&nq));
  CHKERRQ(PetscCuBLASIntCast(ldq,&ldq_));
  CHKERRQ(VecCUDAGetArray(ctx->v,&d_pv));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSECUDA,&matiscuda));
  if (matiscuda) {
    CHKERRQ(MatDenseCUDAGetArrayRead(Q,(const PetscScalar**)&d_q));
  } else {
    CHKERRQ(MatDenseGetArrayRead(Q,&q));
    CHKERRCUDA(cudaMalloc((void**)&d_q,ldq*nq*sizeof(PetscScalar)));
    CHKERRCUDA(cudaMemcpy(d_q,q,ldq*nq*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    CHKERRQ(PetscLogCpuToGpu(ldq*nq*sizeof(PetscScalar)));
  }
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscLogGpuTimeBegin());
  CHKERRCUDA(cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar)));
  d_A = d_pv+(V->nc+V->l)*m;
  d_B = d_q+V->l*ldq+s;
  CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_C,m,n,k,&sone,d_A,m,d_B,ldq_,&szero,d_work,m));
  CHKERRCUDA(cudaMemcpy2D(d_A+(s-V->l)*m,m*sizeof(PetscScalar),d_work,m*sizeof(PetscScalar),m*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
  CHKERRQ(PetscLogGpuTimeEnd());
  if (matiscuda) {
    CHKERRQ(MatDenseCUDARestoreArrayRead(Q,(const PetscScalar**)&d_q));
  } else {
    CHKERRQ(MatDenseRestoreArrayRead(Q,&q));
    CHKERRCUDA(cudaFree(d_q));
  }
  CHKERRCUDA(cudaFree(d_work));
  CHKERRQ(VecCUDARestoreArray(ctx->v,&d_pv));
  CHKERRQ(PetscLogGpuFlops(2.0*m*n*k));
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
  CHKERRQ(PetscCuBLASIntCast(Y->k-Y->l,&m));
  CHKERRQ(PetscCuBLASIntCast(X->k-X->l,&n));
  CHKERRQ(PetscCuBLASIntCast(X->n,&k));
  CHKERRQ(MatGetSize(M,&ldm,NULL));
  CHKERRQ(PetscCuBLASIntCast(ldm,&ldm_));
  CHKERRQ(VecCUDAGetArrayRead(x->v,&d_px));
  CHKERRQ(VecCUDAGetArrayRead(y->v,&d_py));
  CHKERRQ(MatDenseGetArray(M,&pm));
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRCUDA(cudaMalloc((void**)&d_work,m*n*sizeof(PetscScalar)));
  d_A = d_py+(Y->nc+Y->l)*Y->n;
  d_B = d_px+(X->nc+X->l)*X->n;
  C = pm+X->l*ldm+Y->l;
  if (x->mpi) {
    if (ldm==m) {
      CHKERRQ(BVAllocateWork_Private(X,m*n));
      if (k) {
        CHKERRQ(PetscLogGpuTimeBegin());
        CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,ldm_));
        CHKERRQ(PetscLogGpuTimeEnd());
        CHKERRCUDA(cudaMemcpy(X->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
        CHKERRQ(PetscLogGpuToCpu(m*n*sizeof(PetscScalar)));
      } else {
        CHKERRQ(PetscArrayzero(X->work,m*n));
      }
      CHKERRQ(PetscMPIIntCast(m*n,&len));
      CHKERRMPI(MPIU_Allreduce(X->work,C,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)X)));
    } else {
      CHKERRQ(BVAllocateWork_Private(X,2*m*n));
      CC = X->work+m*n;
      if (k) {
        CHKERRQ(PetscLogGpuTimeBegin());
        CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,m));
        CHKERRQ(PetscLogGpuTimeEnd());
        CHKERRCUDA(cudaMemcpy(X->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
        CHKERRQ(PetscLogGpuToCpu(m*n*sizeof(PetscScalar)));
      } else {
        CHKERRQ(PetscArrayzero(X->work,m*n));
      }
      CHKERRQ(PetscMPIIntCast(m*n,&len));
      CHKERRMPI(MPIU_Allreduce(X->work,CC,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)X)));
      for (j=0;j<n;j++) {
        CHKERRQ(PetscArraycpy(C+j*ldm,CC+j*m,m));
      }
    }
  } else {
    if (k) {
      CHKERRQ(BVAllocateWork_Private(X,m*n));
      CHKERRQ(PetscLogGpuTimeBegin());
      CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,m,n,k,&sone,d_A,k,d_B,k,&szero,d_work,m));
      CHKERRQ(PetscLogGpuTimeEnd());
      CHKERRCUDA(cudaMemcpy(X->work,d_work,m*n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      CHKERRQ(PetscLogGpuToCpu(m*n*sizeof(PetscScalar)));
      for (j=0;j<n;j++) {
        CHKERRQ(PetscArraycpy(C+j*ldm,X->work+j*m,m));
      }
    }
  }
  CHKERRCUDA(cudaFree(d_work));
  CHKERRQ(MatDenseRestoreArray(M,&pm));
  CHKERRQ(VecCUDARestoreArrayRead(x->v,&d_px));
  CHKERRQ(VecCUDARestoreArrayRead(y->v,&d_py));
  CHKERRQ(PetscLogGpuFlops(2.0*m*n*k));
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
  CHKERRQ(PetscCuBLASIntCast(X->n,&n));
  CHKERRQ(PetscCuBLASIntCast(X->k-X->l,&k));
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  if (X->matrix) {
    CHKERRQ(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  CHKERRQ(VecCUDAGetArrayRead(x->v,&d_px));
  CHKERRQ(VecCUDAGetArrayRead(z,&d_py));
  if (!q) {
    CHKERRQ(VecCUDAGetArrayWrite(X->buffer,&d_work));
  } else {
    CHKERRCUDA(cudaMalloc((void**)&d_work,k*sizeof(PetscScalar)));
  }
  d_A = d_px+(X->nc+X->l)*X->n;
  d_x = d_py;
  if (x->mpi) {
    CHKERRQ(BVAllocateWork_Private(X,k));
    if (n) {
      CHKERRQ(PetscLogGpuTimeBegin());
#if defined(PETSC_USE_COMPLEX)
      CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,one,k,n,&sone,d_x,n,d_A,n,&szero,d_work,one));
      CHKERRQ(ConjugateCudaArray(d_work,k));
#else
      CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,one,k,n,&sone,d_x,one,d_A,n,&szero,d_work,one));
#endif
      CHKERRQ(PetscLogGpuTimeEnd());
      CHKERRCUDA(cudaMemcpy(X->work,d_work,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      CHKERRQ(PetscLogGpuToCpu(k*sizeof(PetscScalar)));
    } else {
      CHKERRQ(PetscArrayzero(X->work,k));
    }
    if (!q) {
      CHKERRQ(VecCUDARestoreArrayWrite(X->buffer,&d_work));
      CHKERRQ(VecGetArray(X->buffer,&qq));
    } else {
      CHKERRCUDA(cudaFree(d_work));
    }
    CHKERRQ(PetscMPIIntCast(k,&len));
    CHKERRMPI(MPIU_Allreduce(X->work,qq,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)X)));
    if (!q) CHKERRQ(VecRestoreArray(X->buffer,&qq));
  } else {
    if (n) {
      CHKERRQ(PetscLogGpuTimeBegin());
#if defined(PETSC_USE_COMPLEX)
      CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,one,k,n,&sone,d_x,n,d_A,n,&szero,d_work,one));
      CHKERRQ(ConjugateCudaArray(d_work,k));
#else
      CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,one,k,n,&sone,d_x,one,d_A,n,&szero,d_work,one));
#endif
      CHKERRQ(PetscLogGpuTimeEnd());
    }
    if (!q) {
      CHKERRQ(VecCUDARestoreArrayWrite(X->buffer,&d_work));
    } else {
      CHKERRCUDA(cudaMemcpy(q,d_work,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      CHKERRQ(PetscLogGpuToCpu(k*sizeof(PetscScalar)));
      CHKERRCUDA(cudaFree(d_work));
    }
  }
  CHKERRQ(VecCUDARestoreArrayRead(z,&d_py));
  CHKERRQ(VecCUDARestoreArrayRead(x->v,&d_px));
  CHKERRQ(PetscLogGpuFlops(2.0*n*k));
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
  CHKERRQ(PetscCuBLASIntCast(X->n,&n));
  CHKERRQ(PetscCuBLASIntCast(X->k-X->l,&k));
  if (X->matrix) {
    CHKERRQ(BV_IPMatMult(X,y));
    z = X->Bx;
  }
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(VecCUDAGetArrayRead(x->v,&d_px));
  CHKERRQ(VecCUDAGetArrayRead(z,&d_py));
  d_A = d_px+(X->nc+X->l)*X->n;
  d_x = d_py;
  if (n) {
    CHKERRCUDA(cudaMalloc((void**)&d_y,k*sizeof(PetscScalar)));
    CHKERRQ(PetscLogGpuTimeBegin());
#if defined(PETSC_USE_COMPLEX)
    CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_C,CUBLAS_OP_N,one,k,n,&sone,d_x,n,d_A,n,&szero,d_y,one));
    CHKERRQ(ConjugateCudaArray(d_y,k));
#else
    CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,one,k,n,&sone,d_x,one,d_A,n,&szero,d_y,one));
#endif
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRCUDA(cudaMemcpy(m,d_y,k*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    CHKERRQ(PetscLogGpuToCpu(k*sizeof(PetscScalar)));
    CHKERRCUDA(cudaFree(d_y));
  }
  CHKERRQ(VecCUDARestoreArrayRead(z,&d_py));
  CHKERRQ(VecCUDARestoreArrayRead(x->v,&d_px));
  CHKERRQ(PetscLogGpuFlops(2.0*n*k));
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
  CHKERRQ(VecCUDAGetArray(ctx->v,&d_array));
  if (j<0) {
    d_A = d_array+(bv->nc+bv->l)*bv->n;
    CHKERRQ(PetscCuBLASIntCast((bv->k-bv->l)*bv->n,&n));
  } else {
    d_A = d_array+(bv->nc+j)*bv->n;
    CHKERRQ(PetscCuBLASIntCast(bv->n,&n));
  }
  if (alpha == (PetscScalar)0.0) {
    CHKERRCUDA(cudaMemset(d_A,0,n*sizeof(PetscScalar)));
  } else if (alpha != (PetscScalar)1.0) {
    CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasXscal(cublasv2handle,n,&alpha,d_A,one));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(PetscLogGpuFlops(1.0*n));
  }
  CHKERRQ(VecCUDARestoreArray(ctx->v,&d_array));
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
    CHKERRQ(BVGetMat(V,&Vmat));
    CHKERRQ(BVGetMat(W,&Wmat));
    CHKERRQ(MatProductCreateWithMat(A,Vmat,NULL,Wmat));
    CHKERRQ(MatProductSetType(Wmat,MATPRODUCT_AB));
    CHKERRQ(MatProductSetFromOptions(Wmat));
    CHKERRQ(MatProductSymbolic(Wmat));
    CHKERRQ(MatProductNumeric(Wmat));
    CHKERRQ(MatProductClear(Wmat));
    CHKERRQ(BVRestoreMat(V,&Vmat));
    CHKERRQ(BVRestoreMat(W,&Wmat));
  } else {
    CHKERRQ(VecCUDAGetArrayRead(v->v,&d_pv));
    CHKERRQ(VecCUDAGetArrayWrite(w->v,&d_pw));
    for (j=0;j<V->k-V->l;j++) {
      CHKERRQ(VecCUDAPlaceArray(V->cv[1],(PetscScalar *)d_pv+(V->nc+V->l+j)*V->n));
      CHKERRQ(VecCUDAPlaceArray(W->cv[1],d_pw+(W->nc+W->l+j)*W->n));
      CHKERRQ(MatMult(A,V->cv[1],W->cv[1]));
      CHKERRQ(VecCUDAResetArray(V->cv[1]));
      CHKERRQ(VecCUDAResetArray(W->cv[1]));
    }
    CHKERRQ(VecCUDARestoreArrayRead(v->v,&d_pv));
    CHKERRQ(VecCUDARestoreArrayWrite(w->v,&d_pw));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopy_Svec_CUDA(BV V,BV W)
{
  BV_SVEC           *v = (BV_SVEC*)V->data,*w = (BV_SVEC*)W->data;
  const PetscScalar *d_pv,*d_pvc;
  PetscScalar       *d_pw,*d_pwc;

  PetscFunctionBegin;
  CHKERRQ(VecCUDAGetArrayRead(v->v,&d_pv));
  CHKERRQ(VecCUDAGetArrayWrite(w->v,&d_pw));
  d_pvc = d_pv+(V->nc+V->l)*V->n;
  d_pwc = d_pw+(W->nc+W->l)*W->n;
  CHKERRCUDA(cudaMemcpy(d_pwc,d_pvc,(V->k-V->l)*V->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
  CHKERRQ(VecCUDARestoreArrayRead(v->v,&d_pv));
  CHKERRQ(VecCUDARestoreArrayWrite(w->v,&d_pw));
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopyColumn_Svec_CUDA(BV V,PetscInt j,PetscInt i)
{
  BV_SVEC        *v = (BV_SVEC*)V->data;
  PetscScalar    *d_pv;

  PetscFunctionBegin;
  CHKERRQ(VecCUDAGetArray(v->v,&d_pv));
  CHKERRCUDA(cudaMemcpy(d_pv+(V->nc+i)*V->n,d_pv+(V->nc+j)*V->n,V->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
  CHKERRQ(VecCUDARestoreArray(v->v,&d_pv));
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
  CHKERRQ(VecGetBlockSize(bv->t,&bs));
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject)bv->t),&vnew));
  CHKERRQ(VecSetType(vnew,((PetscObject)bv->t)->type_name));
  CHKERRQ(VecSetSizes(vnew,m*bv->n,PETSC_DECIDE));
  CHKERRQ(VecSetBlockSize(vnew,bs));
  CHKERRQ(PetscLogObjectParent((PetscObject)bv,(PetscObject)vnew));
  if (((PetscObject)bv)->name) {
    CHKERRQ(PetscSNPrintf(str,sizeof(str),"%s_0",((PetscObject)bv)->name));
    CHKERRQ(PetscObjectSetName((PetscObject)vnew,str));
  }
  if (copy) {
    CHKERRQ(VecCUDAGetArrayRead(ctx->v,&d_pv));
    CHKERRQ(VecCUDAGetArrayWrite(vnew,&d_pnew));
    CHKERRCUDA(cudaMemcpy(d_pnew,d_pv,PetscMin(m,bv->m)*bv->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
    CHKERRQ(VecCUDARestoreArrayRead(ctx->v,&d_pv));
    CHKERRQ(VecCUDARestoreArrayWrite(vnew,&d_pnew));
  }
  CHKERRQ(VecDestroy(&ctx->v));
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
  CHKERRQ(VecCUDAGetArray(ctx->v,&d_pv));
  CHKERRQ(VecCUDAPlaceArray(bv->cv[l],d_pv+(bv->nc+j)*bv->n));
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreColumn_Svec_CUDA(BV bv,PetscInt j,Vec *v)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscInt       l;

  PetscFunctionBegin;
  l = (j==bv->ci[0])? 0: 1;
  CHKERRQ(VecCUDAResetArray(bv->cv[l]));
  CHKERRQ(VecCUDARestoreArray(ctx->v,NULL));
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
    CHKERRQ(PetscObjectStateGet((PetscObject)*L,&lstate));
    if (lstate != bv->lstate) {
      v = ((BV_SVEC*)bv->L->data)->v;
      CHKERRQ(VecCUDAGetArrayRead(v,&d_pv));
      CHKERRQ(VecCUDARestoreArrayRead(v,&d_pv));
      change = PETSC_TRUE;
    }
  }
  if (R) {
    CHKERRQ(PetscObjectStateGet((PetscObject)*R,&rstate));
    if (rstate != bv->rstate) {
      v = ((BV_SVEC*)bv->R->data)->v;
      CHKERRQ(VecCUDAGetArrayRead(v,&d_pv));
      CHKERRQ(VecCUDARestoreArrayRead(v,&d_pv));
      change = PETSC_TRUE;
    }
  }
  if (change) {
    v = ((BV_SVEC*)bv->data)->v;
    CHKERRQ(VecCUDAGetArray(v,(PetscScalar **)&d_pv));
    CHKERRQ(VecCUDARestoreArray(v,(PetscScalar **)&d_pv));
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
    CHKERRQ(MatDenseCUDAGetArray(bv->Aget,&aa));
    PetscCheck(!aa,PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_WRONGSTATE,"BVGetMat already called on this BV");
    CHKERRQ(MatGetSize(bv->Aget,NULL,&cols));
    if (cols!=m) {
      CHKERRQ(MatDestroy(&bv->Aget));
      create=PETSC_TRUE;
    }
  }
  CHKERRQ(VecCUDAGetArray(ctx->v,&vv));
  if (create) {
    CHKERRQ(MatCreateDenseCUDA(PetscObjectComm((PetscObject)bv),bv->n,PETSC_DECIDE,bv->N,m,vv,&bv->Aget)); /* pass a pointer to avoid allocation of storage */
    CHKERRQ(MatDenseCUDAReplaceArray(bv->Aget,NULL));  /* replace with a null pointer, the value after BVRestoreMat */
    CHKERRQ(PetscLogObjectParent((PetscObject)bv,(PetscObject)bv->Aget));
  }
  CHKERRQ(MatDenseCUDAPlaceArray(bv->Aget,vv+(bv->nc+bv->l)*bv->n));  /* set the actual pointer */
  *A = bv->Aget;
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreMat_Svec_CUDA(BV bv,Mat *A)
{
  BV_SVEC        *ctx = (BV_SVEC*)bv->data;
  PetscScalar    *vv,*aa;

  PetscFunctionBegin;
  CHKERRQ(MatDenseCUDAGetArray(bv->Aget,&aa));
  vv = aa-(bv->nc+bv->l)*bv->n;
  CHKERRQ(MatDenseCUDAResetArray(bv->Aget));
  CHKERRQ(VecCUDARestoreArray(ctx->v,&vv));
  *A = NULL;
  PetscFunctionReturn(0);
}
