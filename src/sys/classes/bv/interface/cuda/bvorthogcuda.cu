/*
   BV orthogonalization routines.

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

#include <slepc/private/bvimpl.h>          /*I   "slepcbv.h"   I*/
#include <slepcblaslapack.h>

/*
   BV_CleanCoefficients_CUDA - Sets to zero all entries of column j of the bv buffer
*/
PetscErrorCode BV_CleanCoefficients_CUDA(BV bv,PetscInt j,PetscScalar *h)
{
  PetscErrorCode ierr;
  PetscScalar    *d_hh,*d_a;
  PetscInt       i;

  PetscFunctionBegin;
  if (!h) {
    ierr = VecCUDAGetArrayReadWrite(bv->buffer,&d_a);CHKERRQ(ierr);
    d_hh = d_a + j*(bv->nc+bv->m);
    ierr = cudaMemset(d_hh,0,(bv->nc+j)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayReadWrite(bv->buffer,&d_a);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscScalar    *d_h,*d_c,sone=1.0;
  PetscInt       i;
  PetscBLASInt   one=1;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  if (!h) {
    ierr = VecCUDAGetArrayReadWrite(bv->buffer,&d_c);CHKERRQ(ierr);
    d_h = d_c + j*(bv->nc+bv->m);
    cberr = cublasXaxpy(cublasv2handle,bv->nc+j,&sone,d_c,one,d_h,one);CHKERRCUBLAS(cberr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayReadWrite(bv->buffer,&d_c);CHKERRQ(ierr);
  } else { /* cpu memory */
    for (i=0;i<bv->nc+j;i++) h[i] += c[i];
  }
  PetscFunctionReturn(0);
}

/*
   BV_SetValue_CUDA - Sets value in row j (counted after the constraints) of column k
   of the coefficients array
*/
PetscErrorCode BV_SetValue_CUDA(BV bv,PetscInt j,PetscInt k,PetscScalar *h,PetscScalar value)
{
  PetscErrorCode ierr;
  PetscScalar    *d_h,*a;
  cudaError_t    cerr;

  PetscFunctionBegin;
  if (!h) {
    ierr = VecCUDAGetArrayReadWrite(bv->buffer,&a);CHKERRQ(ierr);
    d_h = a + k*(bv->nc+bv->m) + bv->nc+j;
    cerr = cudaMemcpy(d_h,&value,sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayReadWrite(bv->buffer,&a);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  const PetscScalar *d_h;
  PetscScalar       dot;
  PetscInt          i;
  PetscBLASInt      one=1;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  if (!h) {
    ierr = VecCUDAGetArrayRead(bv->buffer,&d_h);CHKERRQ(ierr);
    cberr = cublasXdotc(cublasv2handle,bv->nc+j,d_h,one,d_h,one,&dot);CHKERRCUBLAS(cberr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    *sum = PetscRealPart(dot);
    ierr = VecCUDARestoreArrayRead(bv->buffer,&d_h);CHKERRQ(ierr);
  } else { /* cpu memory */
    *sum = 0.0;
    for (i=0;i<bv->nc+j;i++) *sum += PetscRealPart(h[i]*PetscConj(h[i]));
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
  PetscInt              one=1,card;
  struct cudaDeviceProp devprop;
  cudaError_t           cerr;

  PetscFunctionBegin;
  *xcount = 1;
  if (n>BLOCK_SIZE_X) {
    dimBlock->x = BLOCK_SIZE_X;
    dimGrid->x = (n+BLOCK_SIZE_X*TILE_SIZE_X-one)/BLOCK_SIZE_X*TILE_SIZE_X;
  } else {
    dimBlock->x = (n+TILE_SIZE_X-one)/TILE_SIZE_X;
    dimGrid->x = one;
  }
  cerr = cudaGetDevice(&card);CHKERRCUDA(cerr);
  cerr = cudaGetDeviceProperties(&devprop,card);CHKERRCUDA(cerr);
  if (dimGrid->x>(unsigned)devprop.maxGridSize[X_AXIS]) {
    *xcount = (dimGrid->x+devprop.maxGridSize[X_AXIS]-one)/devprop.maxGridSize[X_AXIS];
    dimGrid->x = devprop.maxGridSize[X_AXIS];
  }
  PetscFunctionReturn(0);
}

/* pointwise multiplication */
__global__ void PointwiseMult_kernel(PetscInt xcount,PetscScalar *a,const PetscReal *b,PetscInt n)
{
  PetscInt i,x;

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;
  for (i=x;i<x+TILE_SIZE_X&&i<n;i++) {
    a[i] *= b[i];
  }
}

/* pointwise division */
__global__ void PointwiseDiv_kernel(PetscInt xcount,PetscScalar *a,const PetscReal *b,PetscInt n)
{
  PetscInt i,x;

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;
  for (i=x;i<x+TILE_SIZE_X&&i<n;i++) {
    a[i] /= b[i];
  }
}

/*
   BV_ApplySignature_CUDA - Computes the pointwise product h*omega, where h represents
   the contents of the coefficients array (up to position j) and omega is the signature;
   if inverse=TRUE then the operation is h/omega
*/
PetscErrorCode BV_ApplySignature_CUDA(BV bv,PetscInt j,PetscScalar *h,PetscBool inverse)
{
  PetscErrorCode ierr;
  PetscScalar    *d_h;
  PetscReal      *d_omega;
  PetscInt       i,xcount;
  dim3           blocks3d, threads3d;
  cudaError_t    cerr;

  PetscFunctionBegin;
  if (!(bv->nc+j)) PetscFunctionReturn(0);
  if (!h) {
    ierr = VecCUDAGetArrayReadWrite(bv->buffer,&d_h);CHKERRQ(ierr);
    cerr = cudaMalloc((void**)&d_omega,(bv->nc+j)*sizeof(PetscReal));CHKERRCUDA(cerr);
    cerr = cudaMemcpy(d_omega,bv->omega,(bv->nc+j)*sizeof(PetscReal),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    ierr = SetGrid1D(bv->nc+j,&blocks3d,&threads3d,&xcount);CHKERRQ(ierr);
    if (inverse) {
      for (i=0;i<xcount;i++) {
        PointwiseDiv_kernel<<<blocks3d,threads3d>>>(i,d_h,d_omega,bv->nc+j);
      }
    } else {
      for (i=0;i<xcount;i++) {
        PointwiseMult_kernel<<<blocks3d,threads3d>>>(i,d_h,d_omega,bv->nc+j);
      }
    }
    cerr = cudaGetLastError();CHKERRCUDA(cerr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    cerr = cudaFree(d_omega);CHKERRCUDA(cerr);
    ierr = VecCUDARestoreArrayReadWrite(bv->buffer,&d_h);CHKERRQ(ierr);
  } else {
    if (inverse) for (i=0;i<bv->nc+j;i++) h[i] /= bv->omega[i];
    else for (i=0;i<bv->nc+j;i++) h[i] *= bv->omega[i];
  }
  PetscFunctionReturn(0);
}

/*
   BV_SquareRoot_CUDA - Returns the square root of position j (counted after the constraints)
   of the coefficients array
*/
PetscErrorCode BV_SquareRoot_CUDA(BV bv,PetscInt j,PetscScalar *h,PetscReal *beta)
{
  PetscErrorCode    ierr;
  const PetscScalar *d_h;
  PetscScalar       hh;
  cudaError_t       cerr;

  PetscFunctionBegin;
  if (!h) {
    ierr = VecCUDAGetArrayRead(bv->buffer,&d_h);CHKERRQ(ierr);
    cerr = cudaMemcpy(&hh,d_h+bv->nc+j,sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = BV_SafeSqrt(bv,hh,beta);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(bv->buffer,&d_h);CHKERRQ(ierr);
  } else {
    ierr = BV_SafeSqrt(bv,h[bv->nc+j],beta);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   BV_StoreCoefficients_CUDA - Copy the contents of the coefficients array to an array dest
   provided by the caller (only values from l to j are copied)
*/
PetscErrorCode BV_StoreCoefficients_CUDA(BV bv,PetscInt j,PetscScalar *h,PetscScalar *dest)
{
  PetscErrorCode    ierr;
  const PetscScalar *d_h,*d_a;
  PetscInt          i;
  cudaError_t       cerr;

  PetscFunctionBegin;
  if (!h) {
    ierr = VecCUDAGetArrayRead(bv->buffer,&d_a);CHKERRQ(ierr);
    d_h = d_a + j*(bv->nc+bv->m)+bv->nc;
    cerr = cudaMemcpy(dest-bv->l,d_h,(j-bv->l)*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayRead(bv->buffer,&d_a);CHKERRQ(ierr);
  } else {
    for (i=bv->l;i<j;i++) dest[i-bv->l] = h[bv->nc+i];
  }
  PetscFunctionReturn(0);
}

