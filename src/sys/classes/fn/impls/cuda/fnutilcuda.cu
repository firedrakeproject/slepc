/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Utility subroutines common to several impls
*/

#include <petscsys.h>
#include "fnutilcuda.h"

__global__ void clean_offdiagonal_kernel(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar v,PetscInt xcount)
{
  PetscInt x,j;
  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;

  if (x<n) {
    for (j=0;j<n;j++) {
      if (j != x) d_pa[x+j*ld] = 0.0;
      else d_pa[x+j*ld] = d_pa[x+j*ld]*v;
    }
  }
}

__host__ PetscErrorCode clean_offdiagonal(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar v)
{
  /* XXX use 2D TBD */
  PetscInt    i,dimGrid_xcount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  get_params_1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    clean_offdiagonal_kernel<<<blocks3d, threads3d>>>(n,d_pa,ld,v,i);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

__global__ void set_diagonal_kernel(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar v,PetscInt xcount)
{
  PetscInt x;
  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x+threadIdx.x;

  if (x<n) {
    d_pa[x+x*ld] = v;
  }
}

__host__ PetscErrorCode set_diagonal(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar v)
{
  PetscInt    i,dimGrid_xcount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  get_params_1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    set_diagonal_kernel<<<blocks3d, threads3d>>>(n,d_pa,ld,v,i);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

__global__ void set_Cdiagonal_kernel(PetscInt n,PetscComplex *d_pa,PetscInt ld,PetscReal vr,PetscReal vi,PetscInt xcount)
{
  PetscInt x;
  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x+threadIdx.x;

  if (x<n) {
    d_pa[x+x*ld] = thrust::complex<PetscReal>(vr, vi);
  }
}

__host__ PetscErrorCode set_Cdiagonal(PetscInt n,PetscComplex *d_pa,PetscInt ld,PetscReal vr,PetscReal vi)
{
  PetscInt    i,dimGrid_xcount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  get_params_1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    set_Cdiagonal_kernel<<<blocks3d, threads3d>>>(n,d_pa,ld,vr,vi,i);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

__global__ void shift_diagonal_kernel(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar v,PetscInt xcount)
{
  PetscInt x;
  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x+threadIdx.x;

  if (x<n) {
    d_pa[x+x*ld] += v;
  }
}

__host__ PetscErrorCode shift_diagonal(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar v)
{
  PetscInt    i,dimGrid_xcount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  get_params_1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    shift_diagonal_kernel<<<blocks3d, threads3d>>>(n,d_pa,ld,v,i);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

__global__ void shift_Cdiagonal_kernel(PetscInt n,PetscComplex *d_pa,PetscInt ld,PetscReal vr,PetscReal vi,PetscInt xcount)
{
  PetscInt x;
  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x+threadIdx.x;

  if (x<n) {
    d_pa[x+x*ld] += thrust::complex<PetscReal>(vr, vi);
  }
}

__host__ PetscErrorCode shift_Cdiagonal(PetscInt n,PetscComplex *d_pa,PetscInt ld,PetscReal vr,PetscReal vi)
{
  PetscInt    i,dimGrid_xcount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  get_params_1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    shift_Cdiagonal_kernel<<<blocks3d, threads3d>>>(n,d_pa,ld,vr,vi,i);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

__global__ void copy_array2D_S2C_kernel(PetscInt m,PetscInt n,PetscComplex *d_pa,PetscInt lda,PetscScalar *d_pb,PetscInt ldb,PetscInt xcount,PetscInt ycount)
{
  PetscInt x,y,i,j;

  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;
  y = (ycount*gridDim.y*blockDim.y)+blockIdx.y*blockDim.y*TILE_SIZE_Y+threadIdx.y*TILE_SIZE_Y;
  for (i=x;i<x+TILE_SIZE_X&&i<m;i++) {
    for (j=y;j<y+TILE_SIZE_Y&&j<n;j++) {
      d_pa[i+j*lda] = d_pb[i+j*ldb];
    }
  }
}

__host__ PetscErrorCode copy_array2D_S2C(PetscInt m,PetscInt n,PetscComplex *d_pa,PetscInt lda,PetscScalar *d_pb,PetscInt ldb)
{
  PetscInt    i,j,dimGrid_xcount,dimGrid_ycount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  get_params_2D(m,n,&blocks3d,&threads3d,&dimGrid_xcount,&dimGrid_ycount);
  for (i=0;i<dimGrid_xcount;i++) {
    for (j=0;j<dimGrid_ycount;j++) {
      copy_array2D_S2C_kernel<<<blocks3d,threads3d>>>(m,n,d_pa,lda,d_pb,ldb,i,j);
      PetscCallCUDA(cudaGetLastError());
    }
  }
  PetscFunctionReturn(0);
}

__global__ void copy_array2D_C2S_kernel(PetscInt m,PetscInt n,PetscScalar *d_pa,PetscInt lda,PetscComplex *d_pb,PetscInt ldb,PetscInt xcount,PetscInt ycount)
{
  PetscInt x,y,i,j;

  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;
  y = (ycount*gridDim.y*blockDim.y)+blockIdx.y*blockDim.y*TILE_SIZE_Y+threadIdx.y*TILE_SIZE_Y;
  for (i=x;i<x+TILE_SIZE_X&&i<m;i++) {
    for (j=y;j<y+TILE_SIZE_Y&&j<n;j++) {
      d_pa[i+j*lda] = PetscRealPartComplex(d_pb[i+j*ldb]);
    }
  }
}

__host__ PetscErrorCode copy_array2D_C2S(PetscInt m,PetscInt n,PetscScalar *d_pa,PetscInt lda,PetscComplex *d_pb,PetscInt ldb)
{
  PetscInt    i,j,dimGrid_xcount,dimGrid_ycount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  get_params_2D(m,n,&blocks3d,&threads3d,&dimGrid_xcount,&dimGrid_ycount);
  for (i=0;i<dimGrid_xcount;i++) {
    for (j=0;j<dimGrid_ycount;j++) {
      copy_array2D_C2S_kernel<<<blocks3d,threads3d>>>(m,n,d_pa,lda,d_pb,ldb,i,j);
      PetscCallCUDA(cudaGetLastError());
    }
  }
  PetscFunctionReturn(0);
}

__global__ void add_array2D_Conj_kernel(PetscInt m,PetscInt n,PetscComplex *d_pa,PetscInt lda,PetscInt xcount,PetscInt ycount)
{
  PetscInt x,y,i,j;

  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;
  y = (ycount*gridDim.y*blockDim.y)+blockIdx.y*blockDim.y*TILE_SIZE_Y+threadIdx.y*TILE_SIZE_Y;
  for (i=x;i<x+TILE_SIZE_X&&i<m;i++) {
    for (j=y;j<y+TILE_SIZE_Y&&j<n;j++) {
      d_pa[i+j*lda] += PetscConj(d_pa[i+j*lda]);
    }
  }
}

__host__ PetscErrorCode add_array2D_Conj(PetscInt m,PetscInt n,PetscComplex *d_pa,PetscInt lda)
{
  PetscInt    i,j,dimGrid_xcount,dimGrid_ycount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  get_params_2D(m,n,&blocks3d,&threads3d,&dimGrid_xcount,&dimGrid_ycount);
  for (i=0;i<dimGrid_xcount;i++) {
    for (j=0;j<dimGrid_ycount;j++) {
      add_array2D_Conj_kernel<<<blocks3d,threads3d>>>(m,n,d_pa,lda,i,j);
      PetscCallCUDA(cudaGetLastError());
    }
  }
  PetscFunctionReturn(0);
}

__global__ void getisreal_array2D_kernel(PetscInt m,PetscInt n,PetscComplex *d_pa,PetscInt lda,PetscBool *d_result,PetscInt xcount,PetscInt ycount)
{
  PetscInt x,y,i,j;

  x = (xcount*gridDim.x*blockDim.x)+blockIdx.x*blockDim.x*TILE_SIZE_X+threadIdx.x*TILE_SIZE_X;
  y = (ycount*gridDim.y*blockDim.y)+blockIdx.y*blockDim.y*TILE_SIZE_Y+threadIdx.y*TILE_SIZE_Y;
  if (*d_result) {
    for (i=x;i<x+TILE_SIZE_X&&i<m;i++) {
      for (j=y;j<y+TILE_SIZE_Y&&j<n;j++) {
        if (PetscImaginaryPartComplex(d_pa[i+j*lda])) *d_result=PETSC_FALSE;
      }
    }
  }
}

__host__ PetscErrorCode getisreal_array2D(PetscInt m,PetscInt n,PetscComplex *d_pa,PetscInt lda,PetscBool *d_result)
{
  PetscInt    i,j,dimGrid_xcount,dimGrid_ycount;
  PetscBool   result=PETSC_TRUE;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  PetscCallCUDA(cudaMemcpy(d_result,&result,sizeof(PetscBool),cudaMemcpyHostToDevice));
  get_params_2D(m,n,&blocks3d,&threads3d,&dimGrid_xcount,&dimGrid_ycount);
  for (i=0;i<dimGrid_xcount;i++) {
    for (j=0;j<dimGrid_ycount;j++) {
      getisreal_array2D_kernel<<<blocks3d,threads3d>>>(m,n,d_pa,lda,d_result,i,j);
      PetscCallCUDA(cudaGetLastError());
    }
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
//  dim3           blocks3d,threads3d;
//
//  PetscFunctionBegin;
//  get_params_1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
//  PetscCallCUDA(cudaMalloc((void **)&d_part,sizeof(PetscScalar)*blocks3d.x));
//  PetscCall(PetscMalloc1(blocks3d.x,&part));
//  for (i=0;i<dimGrid_xcount;i++) {
//    mult_diagonal_kernel<threads3d.x><<<blocks3d, threads3d>>>(d_pa,n,ld,d_part,i);
//    PetscCallCUDA(cudaGetLastError());
//
//    PetscCallCUDA(cudaMemcpy(part,d_part,blocks3d.x*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
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
//  PetscCallCUDA(cudaFree(d_part));
//  PetscCall(PetscFree(part));
//  PetscFunctionReturn(0);
//}

__host__ PetscErrorCode get_params_1D(PetscInt rows,dim3 *dimGrid,dim3 *dimBlock,PetscInt *dimGrid_xcount)
{
  int                   card;
  struct cudaDeviceProp devprop;

  PetscFunctionBegin;
  PetscCallCUDA(cudaGetDevice(&card));
  PetscCallCUDA(cudaGetDeviceProperties(&devprop,card));

  *dimGrid_xcount = 1;

  // X axis
  dimGrid->x  = 1;
  dimBlock->x = BLOCK_SIZE_X;
  if (rows>(BLOCK_SIZE_X*TILE_SIZE_X)) {
    dimGrid->x = (rows+((BLOCK_SIZE_X*TILE_SIZE_X)-1))/(BLOCK_SIZE_X*TILE_SIZE_X);
  } else {
    dimBlock->x = (rows+(TILE_SIZE_X-1))/TILE_SIZE_X;
  }

  if (dimGrid->x>(unsigned)devprop.maxGridSize[X_AXIS]) {
    *dimGrid_xcount = (dimGrid->x+(devprop.maxGridSize[X_AXIS]-1))/devprop.maxGridSize[X_AXIS];
    dimGrid->x = devprop.maxGridSize[X_AXIS];
  }
  PetscFunctionReturn(0);
}

__host__ PetscErrorCode get_params_2D(PetscInt rows,PetscInt cols,dim3 *dimGrid,dim3 *dimBlock,PetscInt *dimGrid_xcount,PetscInt *dimGrid_ycount)
{
  int                   card;
  struct cudaDeviceProp devprop;

  PetscFunctionBegin;
  PetscCallCUDA(cudaGetDevice(&card));
  PetscCallCUDA(cudaGetDeviceProperties(&devprop,card));

  *dimGrid_xcount = *dimGrid_ycount = 1;

  // X axis
  dimGrid->x  = 1;
  dimBlock->x = BLOCK_SIZE_X;
  if (rows > (BLOCK_SIZE_X*TILE_SIZE_X)) {
    dimGrid->x = (rows+((BLOCK_SIZE_X*TILE_SIZE_X)-1))/(BLOCK_SIZE_X*TILE_SIZE_X);
  } else {
    dimBlock->x = (rows+(TILE_SIZE_X-1))/TILE_SIZE_X;
  }

  if (dimGrid->x>(unsigned)devprop.maxGridSize[X_AXIS]) {
    *dimGrid_xcount = (dimGrid->x+(devprop.maxGridSize[X_AXIS]-1))/devprop.maxGridSize[X_AXIS];
    dimGrid->x = devprop.maxGridSize[X_AXIS];
  }

  // Y axis
  dimGrid->y  = 1;
  dimBlock->y = BLOCK_SIZE_Y;
  if (cols>(BLOCK_SIZE_Y*TILE_SIZE_Y)) {
    dimGrid->y = (cols+((BLOCK_SIZE_Y*TILE_SIZE_Y)-1))/(BLOCK_SIZE_Y*TILE_SIZE_Y);
  } else {
    dimBlock->y = (cols+(TILE_SIZE_Y-1))/TILE_SIZE_Y;
  }

  if (dimGrid->y>(unsigned)devprop.maxGridSize[Y_AXIS]) {
    *dimGrid_ycount = (dimGrid->y+(devprop.maxGridSize[Y_AXIS]-1))/devprop.maxGridSize[Y_AXIS];
    dimGrid->y = devprop.maxGridSize[Y_AXIS];
  }
  PetscFunctionReturn(0);
}
