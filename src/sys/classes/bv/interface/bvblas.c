/*
   BV private kernels that use the BLAS.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/bvimpl.h>      /*I "slepcbv.h" I*/
#include <slepcblaslapack.h>

#define BLOCKSIZE 64

#undef __FUNCT__
#define __FUNCT__ "BVMult_BLAS_Private"
PetscErrorCode BVMult_BLAS_Private(PetscInt m_,PetscInt n_,PetscInt k_,PetscScalar alpha,PetscScalar *A,PetscScalar *B,PetscScalar beta,PetscScalar *C)
{
  PetscErrorCode ierr;
  PetscBLASInt   m,n,k,l,bs=BLOCKSIZE;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(m_,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k_,&k);CHKERRQ(ierr);
  l = k % bs;
  if (l) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&m,&alpha,A,&k,B,&m,&beta,C,&k));
  for (;l<k;l+=bs) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&bs,&n,&m,&alpha,A+l,&k,B,&m,&beta,C,&k));
  }
  ierr = PetscLogFlops(2*m*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultVec_BLAS_Private"
PetscErrorCode BVMultVec_BLAS_Private(PetscInt n_,PetscInt k_,PetscScalar alpha,PetscScalar *A,PetscScalar *B,PetscScalar beta,PetscScalar *C)
{
  PetscErrorCode ierr;
  PetscBLASInt   n,k,one=1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k_,&k);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&k,&alpha,A,&n,B,&one,&beta,C,&one));
  ierr = PetscLogFlops(2*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDot_BLAS_Private"
PetscErrorCode BVDot_BLAS_Private(PetscInt m_,PetscInt n_,PetscInt k_,PetscScalar *A,PetscScalar *B,PetscScalar *C)
{
  PetscErrorCode ierr;
  PetscScalar    zero=0.0,one=1.0;
  PetscBLASInt   m,n,k;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(m_,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k_,&k);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&m,&n,&k,&one,A,&k,B,&k,&zero,C,&m));
  ierr = PetscLogFlops(2*m*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec_BLAS_Private"
PetscErrorCode BVDotVec_BLAS_Private(PetscInt n_,PetscInt k_,PetscScalar *A,PetscScalar *B,PetscScalar *C)
{
  PetscErrorCode ierr;
  PetscScalar    zero=0.0,done=1.0;
  PetscBLASInt   n,k,one=1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k_,&k);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&k,&done,A,&n,B,&one,&zero,C,&one));
  ierr = PetscLogFlops(2*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

