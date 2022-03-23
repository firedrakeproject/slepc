/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
    CHKERRQ(VecCUDAGetArray(bv->buffer,&d_a));
    CHKERRQ(PetscLogGpuTimeBegin());
    d_hh = d_a + j*(bv->nc+bv->m);
    CHKERRCUDA(cudaMemset(d_hh,0,(bv->nc+j)*sizeof(PetscScalar)));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(VecCUDARestoreArray(bv->buffer,&d_a));
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
    CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
    CHKERRQ(VecCUDAGetArray(bv->buffer,&d_c));
    d_h = d_c + j*(bv->nc+bv->m);
    CHKERRQ(PetscCuBLASIntCast(bv->nc+j,&idx));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasXaxpy(cublasv2handle,idx,&sone,d_c,one,d_h,one));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(PetscLogGpuFlops(1.0*(bv->nc+j)));
    CHKERRQ(VecCUDARestoreArray(bv->buffer,&d_c));
  } else { /* cpu memory */
    for (i=0;i<bv->nc+j;i++) h[i] += c[i];
    CHKERRQ(PetscLogFlops(1.0*(bv->nc+j)));
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
    CHKERRQ(VecCUDAGetArray(bv->buffer,&a));
    CHKERRQ(PetscLogGpuTimeBegin());
    d_h = a + k*(bv->nc+bv->m) + bv->nc+j;
    CHKERRCUDA(cudaMemcpy(d_h,&value,sizeof(PetscScalar),cudaMemcpyHostToDevice));
    CHKERRQ(PetscLogCpuToGpu(sizeof(PetscScalar)));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(VecCUDARestoreArray(bv->buffer,&a));
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
    CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
    CHKERRQ(VecCUDAGetArrayRead(bv->buffer,&d_h));
    CHKERRQ(PetscCuBLASIntCast(bv->nc+j,&idx));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasXdotc(cublasv2handle,idx,d_h,one,d_h,one,&dot));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(PetscLogGpuFlops(2.0*(bv->nc+j)));
    *sum = PetscRealPart(dot);
    CHKERRQ(VecCUDARestoreArrayRead(bv->buffer,&d_h));
  } else { /* cpu memory */
    *sum = 0.0;
    for (i=0;i<bv->nc+j;i++) *sum += PetscRealPart(h[i]*PetscConj(h[i]));
    CHKERRQ(PetscLogFlops(2.0*(bv->nc+j)));
  }
  PetscFunctionReturn(0);
}

#define X_AXIS        0
#define BLOCK_SIZE_X 64
#define TILE_SIZE_X  16 /* work to be done by any thread on axis x */

/*
   Set the kernels grid dimensions
   xcount: number of kernel calls needed for the requested size
 */
PetscErrorCode SetGrid1D(PetscInt n, dim3 *dimGrid, dim3 *dimBlock,PetscInt *xcount)
{
  PetscInt              one=1;
  PetscBLASInt          card;
  struct cudaDeviceProp devprop;

  PetscFunctionBegin;
  *xcount = 1;
  if (n>BLOCK_SIZE_X) {
    dimBlock->x = BLOCK_SIZE_X;
    dimGrid->x = (n+BLOCK_SIZE_X*TILE_SIZE_X-one)/BLOCK_SIZE_X*TILE_SIZE_X;
  } else {
    dimBlock->x = (n+TILE_SIZE_X-one)/TILE_SIZE_X;
    dimGrid->x = one;
  }
  CHKERRCUDA(cudaGetDevice(&card));
  CHKERRCUDA(cudaGetDeviceProperties(&devprop,card));
  if (dimGrid->x>(unsigned)devprop.maxGridSize[X_AXIS]) {
    *xcount = (dimGrid->x+devprop.maxGridSize[X_AXIS]-one)/devprop.maxGridSize[X_AXIS];
    dimGrid->x = devprop.maxGridSize[X_AXIS];
  }
  PetscFunctionReturn(0);
}

/* pointwise multiplication */
__global__ void PointwiseMult_kernel(PetscInt xcount,PetscScalar *a,const PetscScalar *b,PetscInt n)
{
  PetscInt i,x;

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;
  for (i=x;i<x+TILE_SIZE_X&&i<n;i++) {
    a[i] *= PetscRealPart(b[i]);
  }
}

/* pointwise division */
__global__ void PointwiseDiv_kernel(PetscInt xcount,PetscScalar *a,const PetscScalar *b,PetscInt n)
{
  PetscInt i,x;

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;
  for (i=x;i<x+TILE_SIZE_X&&i<n;i++) {
    a[i] /= PetscRealPart(b[i]);
  }
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
    CHKERRQ(VecCUDAGetArray(bv->buffer,&d_h));
    CHKERRQ(VecCUDAGetArrayRead(bv->omega,&d_omega));
    CHKERRQ(SetGrid1D(bv->nc+j,&blocks3d,&threads3d,&xcount));
    CHKERRQ(PetscLogGpuTimeBegin());
    if (inverse) {
      for (i=0;i<xcount;i++) {
        PointwiseDiv_kernel<<<blocks3d,threads3d>>>(i,d_h,d_omega,bv->nc+j);
      }
    } else {
      for (i=0;i<xcount;i++) {
        PointwiseMult_kernel<<<blocks3d,threads3d>>>(i,d_h,d_omega,bv->nc+j);
      }
    }
    CHKERRCUDA(cudaGetLastError());
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(PetscLogGpuFlops(1.0*(bv->nc+j)));
    CHKERRQ(VecCUDARestoreArrayRead(bv->omega,&d_omega));
    CHKERRQ(VecCUDARestoreArray(bv->buffer,&d_h));
  } else {
    CHKERRQ(VecGetArrayRead(bv->omega,&omega));
    if (inverse) for (i=0;i<bv->nc+j;i++) h[i] /= PetscRealPart(omega[i]);
    else for (i=0;i<bv->nc+j;i++) h[i] *= PetscRealPart(omega[i]);
    CHKERRQ(VecRestoreArrayRead(bv->omega,&omega));
    CHKERRQ(PetscLogFlops(1.0*(bv->nc+j)));
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
    CHKERRQ(VecCUDAGetArrayRead(bv->buffer,&d_h));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUDA(cudaMemcpy(&hh,d_h+bv->nc+j,sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    CHKERRQ(PetscLogGpuToCpu(sizeof(PetscScalar)));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(BV_SafeSqrt(bv,hh,beta));
    CHKERRQ(VecCUDARestoreArrayRead(bv->buffer,&d_h));
  } else CHKERRQ(BV_SafeSqrt(bv,h[bv->nc+j],beta));
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
    CHKERRQ(VecCUDAGetArrayRead(bv->buffer,&d_a));
    CHKERRQ(PetscLogGpuTimeBegin());
    d_h = d_a + j*(bv->nc+bv->m)+bv->nc;
    CHKERRCUDA(cudaMemcpy(dest-bv->l,d_h,(j-bv->l)*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    CHKERRQ(PetscLogGpuToCpu((j-bv->l)*sizeof(PetscScalar)));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(VecCUDARestoreArrayRead(bv->buffer,&d_a));
  } else {
    for (i=bv->l;i<j;i++) dest[i-bv->l] = h[bv->nc+i];
  }
  PetscFunctionReturn(0);
}
