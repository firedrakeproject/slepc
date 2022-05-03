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

#include "fnutilcuda.h"

__global__ void set_diagonal_kernel(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar v,PetscInt xcount)
{
  PetscInt x;
  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;

  if (x<n) {
    d_pa[x+x*ld] = v;
  }
}

__host__ PetscErrorCode set_diagonal(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar v)
{
  PetscInt    i,dimGrid_xcount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  SlepcKernelSetGrid1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    set_diagonal_kernel<<<blocks3d, threads3d>>>(n,d_pa,ld,v,i);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

__global__ void set_Cdiagonal_kernel(PetscInt n,PetscComplex *d_pa,PetscInt ld,PetscReal vr,PetscReal vi,PetscInt xcount)
{
  PetscInt x;
  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;

  if (x<n) {
    d_pa[x+x*ld] = thrust::complex<PetscReal>(vr, vi);
  }
}

__host__ PetscErrorCode set_Cdiagonal(PetscInt n,PetscComplex *d_pa,PetscInt ld,PetscReal vr,PetscReal vi)
{
  PetscInt    i,dimGrid_xcount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  SlepcKernelSetGrid1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    set_Cdiagonal_kernel<<<blocks3d, threads3d>>>(n,d_pa,ld,vr,vi,i);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

__global__ void shift_diagonal_kernel(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar v,PetscInt xcount)
{
  PetscInt x;
  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;

  if (x<n) {
    d_pa[x+x*ld] += v;
  }
}

__host__ PetscErrorCode shift_diagonal(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar v)
{
  PetscInt    i,dimGrid_xcount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  SlepcKernelSetGrid1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    shift_diagonal_kernel<<<blocks3d, threads3d>>>(n,d_pa,ld,v,i);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

__global__ void shift_Cdiagonal_kernel(PetscInt n,PetscComplex *d_pa,PetscInt ld,PetscReal vr,PetscReal vi,PetscInt xcount)
{
  PetscInt x;
  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;

  if (x<n) {
    d_pa[x+x*ld] += thrust::complex<PetscReal>(vr, vi);
  }
}

__host__ PetscErrorCode shift_Cdiagonal(PetscInt n,PetscComplex *d_pa,PetscInt ld,PetscReal vr,PetscReal vi)
{
  PetscInt    i,dimGrid_xcount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  SlepcKernelSetGrid1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  for (i=0;i<dimGrid_xcount;i++) {
    shift_Cdiagonal_kernel<<<blocks3d, threads3d>>>(n,d_pa,ld,vr,vi,i);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

__global__ void copy_array2D_S2C_kernel(PetscInt m,PetscInt n,PetscComplex *d_pa,PetscInt lda,PetscScalar *d_pb,PetscInt ldb,PetscInt xcount,PetscInt ycount)
{
  PetscInt x,y,i,j;

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
  y = ycount*gridDim.y*blockDim.y+blockIdx.y*blockDim.y+threadIdx.y;
  for (j=y;j<n;j+=blockDim.y) {
    for (i=x;i<m;i+=blockDim.x) {
      d_pa[i+j*lda] = d_pb[i+j*ldb];
    }
  }
}

__host__ PetscErrorCode copy_array2D_S2C(PetscInt m,PetscInt n,PetscComplex *d_pa,PetscInt lda,PetscScalar *d_pb,PetscInt ldb)
{
  PetscInt    i,j,dimGrid_xcount,dimGrid_ycount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  SlepcKernelSetGrid2DTiles(m,n,&blocks3d,&threads3d,&dimGrid_xcount,&dimGrid_ycount);
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

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
  y = ycount*gridDim.y*blockDim.y+blockIdx.y*blockDim.y+threadIdx.y;
  for (j=y;j<n;j+=blockDim.y) {
    for (i=x;i<m;i+=blockDim.x) {
      d_pa[i+j*lda] = PetscRealPartComplex(d_pb[i+j*ldb]);
    }
  }
}

__host__ PetscErrorCode copy_array2D_C2S(PetscInt m,PetscInt n,PetscScalar *d_pa,PetscInt lda,PetscComplex *d_pb,PetscInt ldb)
{
  PetscInt    i,j,dimGrid_xcount,dimGrid_ycount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  SlepcKernelSetGrid2DTiles(m,n,&blocks3d,&threads3d,&dimGrid_xcount,&dimGrid_ycount);
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

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
  y = ycount*gridDim.y*blockDim.y+blockIdx.y*blockDim.y+threadIdx.y;
  for (j=y;j<n;j+=blockDim.y) {
    for (i=x;i<m;i+=blockDim.x) {
      d_pa[i+j*lda] += PetscConj(d_pa[i+j*lda]);
    }
  }
}

__host__ PetscErrorCode add_array2D_Conj(PetscInt m,PetscInt n,PetscComplex *d_pa,PetscInt lda)
{
  PetscInt    i,j,dimGrid_xcount,dimGrid_ycount;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  SlepcKernelSetGrid2DTiles(m,n,&blocks3d,&threads3d,&dimGrid_xcount,&dimGrid_ycount);
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

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
  y = ycount*gridDim.y*blockDim.y+blockIdx.y*blockDim.y+threadIdx.y;
  if (*d_result) {
    for (j=y;j<n;j+=blockDim.y) {
      for (i=x;i<m;i+=blockDim.x) {
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
  SlepcKernelSetGrid2DTiles(m,n,&blocks3d,&threads3d,&dimGrid_xcount,&dimGrid_ycount);
  for (i=0;i<dimGrid_xcount;i++) {
    for (j=0;j<dimGrid_ycount;j++) {
      getisreal_array2D_kernel<<<blocks3d,threads3d>>>(m,n,d_pa,lda,d_result,i,j);
      PetscCallCUDA(cudaGetLastError());
    }
  }
  PetscFunctionReturn(0);
}

__global__ void mult_diagonal_kernel(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar *d_v,PetscInt xcount)
{
  PetscInt               x;
  unsigned int           bs=blockDim.x;
  __shared__ PetscScalar shrdres[SLEPC_BLOCK_SIZE_X];

  x = xcount*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
  shrdres[threadIdx.x] = (x<n)? d_pa[x+ld*x]: 1.0;
  __syncthreads();

  /* reduction */
  if ((bs >= 512) && (threadIdx.x < 256)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x + 256]; } __syncthreads();
  if ((bs >= 256) && (threadIdx.x < 128)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x + 128]; } __syncthreads();
  if ((bs >= 128) && (threadIdx.x <  64)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +  64]; } __syncthreads();
  if ((bs >=  64) && (threadIdx.x <  32)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +  32]; } __syncthreads();
  if ((bs >=  32) && (threadIdx.x <  16)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +  16]; } __syncthreads();
  if ((bs >=  16) && (threadIdx.x <   8)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +   8]; } __syncthreads();
  if ((bs >=   8) && (threadIdx.x <   4)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +   4]; } __syncthreads();
  if ((bs >=   4) && (threadIdx.x <   2)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +   2]; } __syncthreads();
  if ((bs >=   2) && (threadIdx.x <   1)) { shrdres[threadIdx.x] *= shrdres[threadIdx.x +   1]; } __syncthreads();
  if (threadIdx.x == 0) d_v[blockIdx.x] = shrdres[threadIdx.x];
}

__host__ PetscErrorCode mult_diagonal(PetscInt n,PetscScalar *d_pa,PetscInt ld,PetscScalar *v)
{
  PetscInt    i,j,dimGrid_xcount;
  PetscScalar *part,*d_part;
  dim3        blocks3d,threads3d;

  PetscFunctionBegin;
  SlepcKernelSetGrid1D(n,&blocks3d,&threads3d,&dimGrid_xcount);
  PetscCallCUDA(cudaMalloc((void **)&d_part,sizeof(PetscScalar)*blocks3d.x));
  PetscCall(PetscMalloc1(blocks3d.x,&part));
  for (i=0;i<dimGrid_xcount;i++) {
    mult_diagonal_kernel<<<blocks3d,threads3d>>>(n,d_pa,ld,d_part,i);
    PetscCallCUDA(cudaGetLastError());
    PetscCallCUDA(cudaMemcpy(part,d_part,blocks3d.x*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    if (i == 0) {
      *v = part[0];
      j=1;
    } else j=0;
    for (; j<(int)blocks3d.x; j++) *v *= part[j];
  }
  PetscCallCUDA(cudaFree(d_part));
  PetscCall(PetscFree(part));
  PetscFunctionReturn(0);
}
