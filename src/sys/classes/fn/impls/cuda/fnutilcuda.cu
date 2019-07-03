/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Utility subroutines common to several impls
*/

#include <petscsys.h>

#include "fnutilcuda.h"

#if defined(PETSC_HAVE_CUDA)

__global__ void clean_offdiagonal_kernel(PetscScalar *d_pa,PetscInt n,PetscInt ld,PetscScalar v,PetscInt xcount)
{
  PetscInt x,j;
  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;

  if (x<n) {
    for (j=0;j<n;j++){
      if (j != x) {
        d_pa[x+ld*j] = 0.0;
      } else {
//        d_pa[x+ld*j] = PetscSqrtScalar(d_pa[x+ld*j]);
        d_pa[x+ld*j] = d_pa[x+ld*j]*v;
      }
    }
  }
}

__host__ PetscErrorCode clean_offdiagonal(PetscScalar *d_pa,PetscInt n, PetscInt ld,PetscScalar v)
{
  /* XXX use 2D TBD */
  PetscInt        i,dimGrid_xcount;
  dim3            blocks3d,threads3d;
  cudaError_t     cerr;

  PetscFunctionBegin;
  get_params_1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    clean_offdiagonal_kernel<<<blocks3d, threads3d>>>(d_pa,n,ld,v,i);
    cerr = cudaGetLastError(); CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}

__global__ void set_diagonal_kernel(PetscScalar *d_pa,PetscInt n,PetscInt ld,PetscScalar v,PetscInt xcount)
{
  PetscInt x;
  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;

  if (x<n) {
    d_pa[x+ld*x] = v;
  }
}

__host__ PetscErrorCode set_diagonal(PetscScalar *d_pa,PetscInt n, PetscInt ld,PetscScalar v)
{
  PetscInt        i,dimGrid_xcount;
  dim3            blocks3d,threads3d;
  cudaError_t     cerr;

  PetscFunctionBegin;
  get_params_1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    set_diagonal_kernel<<<blocks3d, threads3d>>>(d_pa,n,ld,v,i);
    cerr = cudaGetLastError(); CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}

__global__ void shift_diagonal_kernel(PetscScalar *d_pa,PetscInt n,PetscInt ld,PetscScalar v,PetscInt xcount)
{
  PetscInt x;
  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;

  if (x<n) {
    d_pa[x+ld*x] += v;
  }
}

__host__ PetscErrorCode shift_diagonal(PetscScalar *d_pa,PetscInt n, PetscInt ld,PetscScalar v)
{
  PetscInt        i,dimGrid_xcount;
  dim3            blocks3d,threads3d;
  cudaError_t     cerr;

  PetscFunctionBegin;
  get_params_1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    shift_diagonal_kernel<<<blocks3d, threads3d>>>(d_pa,n,ld,v,i);
    cerr = cudaGetLastError(); CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}

//template <class T, unsigned int bs>
//__global__ void mult_diagonal_kernel(T *d_pa,PetscInt n,PetscInt ld,T *d_v,PetscInt xcount)
//{
//  PetscInt            x;
//  extern __shared__ T *shrdres;
//
//  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;
//
//  if (x<n) {
//    shrdres[x] = d_pa[x+ld*x];
//    __syncthreads();
//
//    /* reduction */
//    if ((bs >= 512) && (threadIdx.x < 256)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x + 256]; } __syncthreads();
//    if ((bs >= 256) && (threadIdx.x < 128)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x + 128]; } __syncthreads();
//    if ((bs >= 128) && (threadIdx.x <  64)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +  64]; } __syncthreads();
//    if ((bs >=  64) && (threadIdx.x <  32)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +  32]; } __syncthreads();
//    if ((bs >=  32) && (threadIdx.x <  16)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +  16]; } __syncthreads();
//    if ((bs >=  16) && (threadIdx.x <   8)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +   8]; } __syncthreads();
//    if ((bs >=   8) && (threadIdx.x <   4)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +   4]; } __syncthreads();
//    if ((bs >=   4) && (threadIdx.x <   2)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +   2]; } __syncthreads();
//    if ((bs >=   2) && (threadIdx.x <   1)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +   1]; } __syncthreads();
//
//    if (threadIdx.x == 0) d_v[blockIdx.x] = shrdres[threadIdx.x];
//  }
//
//}
//
//__host__ PetscErrorCode mult_diagonal(PetscScalar *d_pa,PetscInt n, PetscInt ld,PetscScalar *v)
//{
//  PetscInt       i,j,dimGrid_xcount;
//  PetscScalar    *part,*d_part;
//  PetscErrorCode ierr;
//  dim3           blocks3d,threads3d;
//  cudaError_t    cerr;
//
//  PetscFunctionBegin;
//  get_params_1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
//  cerr = cudaMalloc((void **)&d_part,sizeof(PetscScalar)*blocks3d.x);CHKERRCUDA(cerr);
//  ierr = PetscMalloc1(blocks3d.x,&part);CHKERRQ(ierr);
//  for (i=0;i<dimGrid_xcount;i++) {
//    mult_diagonal_kernel<threads3d.x><<<blocks3d, threads3d>>>(d_pa,n,ld,d_part,i);
//    cerr = cudaGetLastError();CHKERRCUDA(cerr);
//
//    cerr = cudaMemcpy(part,d_part,blocks3d.x*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
//    if (i == 0) {
//      *v = part[0];
//      j=1;
//    } else {
//      j=0;
//    }
//    for (; j<blocks3d.x; j++) {
//      *v *= part[j];
//    }
//  }
//  cerr = cudaFree(d_part);CHKERRCUDA(cerr);
//  ierr = PetscFree(part);CHKERRQ(ierr);
//  PetscFunctionReturn(0);
//}

__host__ PetscErrorCode get_params_1D(PetscInt rows,dim3 *dimGrid,dim3 *dimBlock,PetscInt *dimGrid_xcount)
{
  PetscInt              card;
  struct cudaDeviceProp devprop;
  cudaError_t           cerr;

  PetscFunctionBegin;
  cerr = cudaGetDevice(&card);CHKERRCUDA(cerr);
  cerr = cudaGetDeviceProperties(&devprop,card);CHKERRCUDA(cerr);

  *dimGrid_xcount = 1;

  // X axis
  dimGrid->x = 1;
  dimBlock->x = BLOCK_SIZE_X;
  if (rows>BLOCK_SIZE_X) {
    dimGrid->x = (rows+((BLOCK_SIZE_X*TILE_SIZE_X)-1))/BLOCK_SIZE_X*TILE_SIZE_X;
  } else {
    dimBlock->x = rows;
  }

  if (dimGrid->x>(unsigned)devprop.maxGridSize[X_AXIS]) {
    *dimGrid_xcount = (dimGrid->x+(devprop.maxGridSize[X_AXIS]-1))/devprop.maxGridSize[X_AXIS];
    dimGrid->x = devprop.maxGridSize[X_AXIS];
  }

  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_CUDA */
