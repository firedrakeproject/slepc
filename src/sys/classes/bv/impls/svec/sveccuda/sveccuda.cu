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
#else /* complex double */
#define cublasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cublasZgemm((a),(b),(c),(d),(e),(f),(cuDoubleComplex*)(g),(cuDoubleComplex*)(h),(i),(cuDoubleComplex*)(j),(k),(cuDoubleComplex*)(l),(cuDoubleComplex*)(m),(n))
#define cublasXgemv(a,b,c,d,e,f,g,h,i,j,k,l) cublasZgemv((a),(b),(c),(d),(cuDoubleComplex*)(e),(cuDoubleComplex*)(f),(g),(cuDoubleComplex*)(h),(i),(cuDoubleComplex*)(j),(cuDoubleComplex*)(k),(l))
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define cublasXgemm cublasSgemm
#define cublasXgemv cublasSgemv
#else /* real double */
#define cublasXgemm cublasDgemm
#define cublasXgemv cublasDgemv
#endif
#endif

#undef __FUNCT__
#define __FUNCT__ "BVMult_Svec_CUDA"
PetscErrorCode BVMult_Svec_CUDA(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  PetscErrorCode ierr;
  BV_SVEC        *y = (BV_SVEC*)Y->data,*x = (BV_SVEC*)X->data;
  PetscScalar    *px,*py,*q,*d_q;
  PetscInt       ldq,mq;
  cublasStatus_t cberr;
  cudaError_t    err;

  PetscFunctionBegin;
  if (!Y->n) PetscFunctionReturn(0);
  ierr = MatGetSize(Q,&ldq,&mq);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(x->v,&px);CHKERRQ(ierr);
  if (beta==0.0) {
    ierr = VecCUDAGetArrayWrite(y->v,&py);CHKERRQ(ierr);
  } else {
    ierr = VecCUDAGetArrayReadWrite(y->v,&py);CHKERRQ(ierr);
  }
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  err = cudaMalloc((void**)&d_q,ldq*mq*sizeof(PetscScalar*));CHKERRCUDA(err);
  err = cudaMemcpy(d_q,q,ldq*mq*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);

  cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,Y->n,Y->k-Y->l,X->k-X->l,(const PetscScalar*)&alpha,px+(X->nc+X->l)*X->n,Y->n,d_q+Y->l*ldq+X->l,ldq,(const PetscScalar*)&beta,py+(Y->nc+Y->l)*Y->n,Y->n);CHKERRCUBLAS(cberr);

  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  err = cudaFree(d_q);CHKERRCUDA(err);
  ierr = VecCUDARestoreArrayRead(x->v,&px);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(y->v,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

