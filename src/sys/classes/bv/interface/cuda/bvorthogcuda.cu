/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV orthogonalization routines (CUDA)
*/

#include <slepc/private/bvimpl.h>          /*I   "slepcbv.h"   I*/
#include <slepcblaslapack.h>
#include <slepccublas.h>

/*
   BV_CleanCoefficients_CUDA - Sets to zero all entries of column j of the bv buffer
*/
PetscErrorCode BV_CleanCoefficients_CUDA(BV bv,PetscInt j,PetscScalar *h)
{
  PetscScalar    *d_hh,*d_a;
  PetscInt       i;

  PetscFunctionBegin;
  if (!h) {
    PetscCall(VecCUDAGetArray(bv->buffer,&d_a));
    PetscCall(PetscLogGpuTimeBegin());
    d_hh = d_a + j*(bv->nc+bv->m);
    PetscCallCUDA(cudaMemset(d_hh,0,(bv->nc+j)*sizeof(PetscScalar)));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecCUDARestoreArray(bv->buffer,&d_a));
  } else { /* cpu memory */
    for (i=0;i<bv->nc+j;i++) h[i] = 0.0;
  }
  PetscFunctionReturn(0);
}

/*
   BV_AddCoefficients_CUDA - Add the contents of the scratch (0-th column) of the bv buffer
   into column j of the bv buffer
 */
PetscErrorCode BV_AddCoefficients_CUDA(BV bv,PetscInt j,PetscScalar *h,PetscScalar *c)
{
  PetscScalar    *d_h,*d_c,sone=1.0;
  PetscInt       i;
  PetscCuBLASInt idx=0,one=1;
  cublasHandle_t cublasv2handle;

  PetscFunctionBegin;
  if (!h) {
    PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
    PetscCall(VecCUDAGetArray(bv->buffer,&d_c));
    d_h = d_c + j*(bv->nc+bv->m);
    PetscCall(PetscCuBLASIntCast(bv->nc+j,&idx));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXaxpy(cublasv2handle,idx,&sone,d_c,one,d_h,one));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(1.0*(bv->nc+j)));
    PetscCall(VecCUDARestoreArray(bv->buffer,&d_c));
  } else { /* cpu memory */
    for (i=0;i<bv->nc+j;i++) h[i] += c[i];
    PetscCall(PetscLogFlops(1.0*(bv->nc+j)));
  }
  PetscFunctionReturn(0);
}

/*
   BV_SetValue_CUDA - Sets value in row j (counted after the constraints) of column k
   of the coefficients array
*/
PetscErrorCode BV_SetValue_CUDA(BV bv,PetscInt j,PetscInt k,PetscScalar *h,PetscScalar value)
{
  PetscScalar    *d_h,*a;

  PetscFunctionBegin;
  if (!h) {
    PetscCall(VecCUDAGetArray(bv->buffer,&a));
    PetscCall(PetscLogGpuTimeBegin());
    d_h = a + k*(bv->nc+bv->m) + bv->nc+j;
    PetscCallCUDA(cudaMemcpy(d_h,&value,sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCall(PetscLogCpuToGpu(sizeof(PetscScalar)));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecCUDARestoreArray(bv->buffer,&a));
  } else { /* cpu memory */
    h[bv->nc+j] = value;
  }
  PetscFunctionReturn(0);
}

/*
   BV_SquareSum_CUDA - Returns the value h'*h, where h represents the contents of the
   coefficients array (up to position j)
*/
PetscErrorCode BV_SquareSum_CUDA(BV bv,PetscInt j,PetscScalar *h,PetscReal *sum)
{
  const PetscScalar *d_h;
  PetscScalar       dot;
  PetscInt          i;
  PetscCuBLASInt    idx=0,one=1;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  if (!h) {
    PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
    PetscCall(VecCUDAGetArrayRead(bv->buffer,&d_h));
    PetscCall(PetscCuBLASIntCast(bv->nc+j,&idx));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXdotc(cublasv2handle,idx,d_h,one,d_h,one,&dot));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(2.0*(bv->nc+j)));
    *sum = PetscRealPart(dot);
    PetscCall(VecCUDARestoreArrayRead(bv->buffer,&d_h));
  } else { /* cpu memory */
    *sum = 0.0;
    for (i=0;i<bv->nc+j;i++) *sum += PetscRealPart(h[i]*PetscConj(h[i]));
    PetscCall(PetscLogFlops(2.0*(bv->nc+j)));
  }
  PetscFunctionReturn(0);
}

/* pointwise multiplication */
__global__ void PointwiseMult_kernel(PetscInt xcount,PetscScalar *a,const PetscScalar *b,PetscInt n)
{
  PetscInt x;

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
  if (x<n) a[x] *= PetscRealPart(b[x]);
}

/* pointwise division */
__global__ void PointwiseDiv_kernel(PetscInt xcount,PetscScalar *a,const PetscScalar *b,PetscInt n)
{
  PetscInt x;

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
  if (x<n) a[x] /= PetscRealPart(b[x]);
}

/*
   BV_ApplySignature_CUDA - Computes the pointwise product h*omega, where h represents
   the contents of the coefficients array (up to position j) and omega is the signature;
   if inverse=TRUE then the operation is h/omega
*/
PetscErrorCode BV_ApplySignature_CUDA(BV bv,PetscInt j,PetscScalar *h,PetscBool inverse)
{
  PetscScalar       *d_h;
  const PetscScalar *d_omega,*omega;
  PetscInt          i,xcount;
  dim3              blocks3d, threads3d;

  PetscFunctionBegin;
  if (!(bv->nc+j)) PetscFunctionReturn(0);
  if (!h) {
    PetscCall(VecCUDAGetArray(bv->buffer,&d_h));
    PetscCall(VecCUDAGetArrayRead(bv->omega,&d_omega));
    PetscCall(SlepcKernelSetGrid1D(bv->nc+j,&blocks3d,&threads3d,&xcount));
    PetscCall(PetscLogGpuTimeBegin());
    if (inverse) {
      for (i=0;i<xcount;i++) PointwiseDiv_kernel<<<blocks3d,threads3d>>>(i,d_h,d_omega,bv->nc+j);
    } else {
      for (i=0;i<xcount;i++) PointwiseMult_kernel<<<blocks3d,threads3d>>>(i,d_h,d_omega,bv->nc+j);
    }
    PetscCallCUDA(cudaGetLastError());
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(1.0*(bv->nc+j)));
    PetscCall(VecCUDARestoreArrayRead(bv->omega,&d_omega));
    PetscCall(VecCUDARestoreArray(bv->buffer,&d_h));
  } else {
    PetscCall(VecGetArrayRead(bv->omega,&omega));
    if (inverse) for (i=0;i<bv->nc+j;i++) h[i] /= PetscRealPart(omega[i]);
    else for (i=0;i<bv->nc+j;i++) h[i] *= PetscRealPart(omega[i]);
    PetscCall(VecRestoreArrayRead(bv->omega,&omega));
    PetscCall(PetscLogFlops(1.0*(bv->nc+j)));
  }
  PetscFunctionReturn(0);
}

/*
   BV_SquareRoot_CUDA - Returns the square root of position j (counted after the constraints)
   of the coefficients array
*/
PetscErrorCode BV_SquareRoot_CUDA(BV bv,PetscInt j,PetscScalar *h,PetscReal *beta)
{
  const PetscScalar *d_h;
  PetscScalar       hh;

  PetscFunctionBegin;
  if (!h) {
    PetscCall(VecCUDAGetArrayRead(bv->buffer,&d_h));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUDA(cudaMemcpy(&hh,d_h+bv->nc+j,sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    PetscCall(PetscLogGpuToCpu(sizeof(PetscScalar)));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(BV_SafeSqrt(bv,hh,beta));
    PetscCall(VecCUDARestoreArrayRead(bv->buffer,&d_h));
  } else PetscCall(BV_SafeSqrt(bv,h[bv->nc+j],beta));
  PetscFunctionReturn(0);
}

/*
   BV_StoreCoefficients_CUDA - Copy the contents of the coefficients array to an array dest
   provided by the caller (only values from l to j are copied)
*/
PetscErrorCode BV_StoreCoefficients_CUDA(BV bv,PetscInt j,PetscScalar *h,PetscScalar *dest)
{
  const PetscScalar *d_h,*d_a;
  PetscInt          i;

  PetscFunctionBegin;
  if (!h) {
    PetscCall(VecCUDAGetArrayRead(bv->buffer,&d_a));
    PetscCall(PetscLogGpuTimeBegin());
    d_h = d_a + j*(bv->nc+bv->m)+bv->nc;
    PetscCallCUDA(cudaMemcpy(dest-bv->l,d_h,(j-bv->l)*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    PetscCall(PetscLogGpuToCpu((j-bv->l)*sizeof(PetscScalar)));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecCUDARestoreArrayRead(bv->buffer,&d_a));
  } else {
    for (i=bv->l;i<j;i++) dest[i-bv->l] = h[bv->nc+i];
  }
  PetscFunctionReturn(0);
}
