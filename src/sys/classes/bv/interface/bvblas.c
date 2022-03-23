/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV private kernels that use the BLAS
*/

#include <slepc/private/bvimpl.h>
#include <slepcblaslapack.h>

#define BLOCKSIZE 64

/*
    C := alpha*A*B + beta*C

    A is mxk (ld=m), B is kxn (ld=ldb), C is mxn (ld=m)
*/
PetscErrorCode BVMult_BLAS_Private(BV bv,PetscInt m_,PetscInt n_,PetscInt k_,PetscInt ldb_,PetscScalar alpha,const PetscScalar *A,const PetscScalar *B,PetscScalar beta,PetscScalar *C)
{
  PetscBLASInt   m,n,k,ldb;
#if defined(PETSC_HAVE_FBLASLAPACK) || defined(PETSC_HAVE_F2CBLASLAPACK)
  PetscBLASInt   l,bs=BLOCKSIZE;
#endif

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(m_,&m));
  CHKERRQ(PetscBLASIntCast(n_,&n));
  CHKERRQ(PetscBLASIntCast(k_,&k));
  CHKERRQ(PetscBLASIntCast(ldb_,&ldb));
#if defined(PETSC_HAVE_FBLASLAPACK) || defined(PETSC_HAVE_F2CBLASLAPACK)
  l = m % bs;
  if (l) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&k,&alpha,(PetscScalar*)A,&m,(PetscScalar*)B,&ldb,&beta,C,&m));
  for (;l<m;l+=bs) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&bs,&n,&k,&alpha,(PetscScalar*)A+l,&m,(PetscScalar*)B,&ldb,&beta,C+l,&m));
  }
#else
  if (m) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&m,&n,&k,&alpha,(PetscScalar*)A,&m,(PetscScalar*)B,&ldb,&beta,C,&m));
#endif
  CHKERRQ(PetscLogFlops(2.0*m*n*k));
  PetscFunctionReturn(0);
}

/*
    y := alpha*A*x + beta*y

    A is nxk (ld=n)
*/
PetscErrorCode BVMultVec_BLAS_Private(BV bv,PetscInt n_,PetscInt k_,PetscScalar alpha,const PetscScalar *A,const PetscScalar *x,PetscScalar beta,PetscScalar *y)
{
  PetscBLASInt   n,k,one=1;

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(n_,&n));
  CHKERRQ(PetscBLASIntCast(k_,&k));
  if (n) PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&k,&alpha,A,&n,x,&one,&beta,y,&one));
  CHKERRQ(PetscLogFlops(2.0*n*k));
  PetscFunctionReturn(0);
}

/*
    A(:,s:e-1) := A*B(:,s:e-1)

    A is mxk (ld=m), B is kxn (ld=ldb)  n=e-s
*/
PetscErrorCode BVMultInPlace_BLAS_Private(BV bv,PetscInt m_,PetscInt k_,PetscInt ldb_,PetscInt s,PetscInt e,PetscScalar *A,const PetscScalar *B,PetscBool btrans)
{
  PetscScalar    *pb,zero=0.0,one=1.0;
  PetscBLASInt   m,n,k,l,ldb,bs=BLOCKSIZE;
  PetscInt       j,n_=e-s;
  const char     *bt;

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(m_,&m));
  CHKERRQ(PetscBLASIntCast(n_,&n));
  CHKERRQ(PetscBLASIntCast(k_,&k));
  CHKERRQ(PetscBLASIntCast(ldb_,&ldb));
  CHKERRQ(BVAllocateWork_Private(bv,BLOCKSIZE*n_));
  if (PetscUnlikely(btrans)) {
    pb = (PetscScalar*)B+s;
    bt = "C";
  } else {
    pb = (PetscScalar*)B+s*ldb;
    bt = "N";
  }
  l = m % bs;
  if (l) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N",bt,&l,&n,&k,&one,A,&m,pb,&ldb,&zero,bv->work,&l));
    for (j=0;j<n;j++) CHKERRQ(PetscArraycpy(A+(s+j)*m,bv->work+j*l,l));
  }
  for (;l<m;l+=bs) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N",bt,&bs,&n,&k,&one,A+l,&m,pb,&ldb,&zero,bv->work,&bs));
    for (j=0;j<n;j++) CHKERRQ(PetscArraycpy(A+(s+j)*m+l,bv->work+j*bs,bs));
  }
  CHKERRQ(PetscLogFlops(2.0*m*n*k));
  PetscFunctionReturn(0);
}

/*
    V := V*B

    V is mxn (ld=m), B is nxn (ld=k)
*/
PetscErrorCode BVMultInPlace_Vecs_Private(BV bv,PetscInt m_,PetscInt n_,PetscInt k_,Vec *V,const PetscScalar *B,PetscBool btrans)
{
  PetscScalar       zero=0.0,one=1.0,*out,*pout;
  const PetscScalar *pin;
  PetscBLASInt      m = 0,n,k,l,bs=BLOCKSIZE;
  PetscInt          j;
  const char        *bt;

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(m_,&m));
  CHKERRQ(PetscBLASIntCast(n_,&n));
  CHKERRQ(PetscBLASIntCast(k_,&k));
  CHKERRQ(BVAllocateWork_Private(bv,2*BLOCKSIZE*n_));
  out = bv->work+BLOCKSIZE*n_;
  if (btrans) bt = "C";
  else bt = "N";
  l = m % bs;
  if (l) {
    for (j=0;j<n;j++) {
      CHKERRQ(VecGetArrayRead(V[j],&pin));
      CHKERRQ(PetscArraycpy(bv->work+j*l,pin,l));
      CHKERRQ(VecRestoreArrayRead(V[j],&pin));
    }
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N",bt,&l,&n,&n,&one,bv->work,&l,(PetscScalar*)B,&k,&zero,out,&l));
    for (j=0;j<n;j++) {
      CHKERRQ(VecGetArray(V[j],&pout));
      CHKERRQ(PetscArraycpy(pout,out+j*l,l));
      CHKERRQ(VecRestoreArray(V[j],&pout));
    }
  }
  for (;l<m;l+=bs) {
    for (j=0;j<n;j++) {
      CHKERRQ(VecGetArrayRead(V[j],&pin));
      CHKERRQ(PetscArraycpy(bv->work+j*bs,pin+l,bs));
      CHKERRQ(VecRestoreArrayRead(V[j],&pin));
    }
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N",bt,&bs,&n,&n,&one,bv->work,&bs,(PetscScalar*)B,&k,&zero,out,&bs));
    for (j=0;j<n;j++) {
      CHKERRQ(VecGetArray(V[j],&pout));
      CHKERRQ(PetscArraycpy(pout+l,out+j*bs,bs));
      CHKERRQ(VecRestoreArray(V[j],&pout));
    }
  }
  CHKERRQ(PetscLogFlops(2.0*n*n*k));
  PetscFunctionReturn(0);
}

/*
    B := alpha*A + beta*B

    A,B are nxk (ld=n)
*/
PetscErrorCode BVAXPY_BLAS_Private(BV bv,PetscInt n_,PetscInt k_,PetscScalar alpha,const PetscScalar *A,PetscScalar beta,PetscScalar *B)
{
  PetscBLASInt   m,one=1;

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(n_*k_,&m));
  if (beta!=(PetscScalar)1.0) {
    PetscStackCallBLAS("BLASscal",BLASscal_(&m,&beta,B,&one));
    CHKERRQ(PetscLogFlops(m));
  }
  PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&m,&alpha,A,&one,B,&one));
  CHKERRQ(PetscLogFlops(2.0*m));
  PetscFunctionReturn(0);
}

/*
    C := A'*B

    A' is mxk (ld=k), B is kxn (ld=k), C is mxn (ld=ldc)
*/
PetscErrorCode BVDot_BLAS_Private(BV bv,PetscInt m_,PetscInt n_,PetscInt k_,PetscInt ldc_,const PetscScalar *A,const PetscScalar *B,PetscScalar *C,PetscBool mpi)
{
  PetscScalar    zero=0.0,one=1.0,*CC;
  PetscBLASInt   m,n,k,ldc,j;
  PetscMPIInt    len;

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(m_,&m));
  CHKERRQ(PetscBLASIntCast(n_,&n));
  CHKERRQ(PetscBLASIntCast(k_,&k));
  CHKERRQ(PetscBLASIntCast(ldc_,&ldc));
  if (mpi) {
    if (ldc==m) {
      CHKERRQ(BVAllocateWork_Private(bv,m*n));
      if (k) PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&m,&n,&k,&one,(PetscScalar*)A,&k,(PetscScalar*)B,&k,&zero,bv->work,&ldc));
      else CHKERRQ(PetscArrayzero(bv->work,m*n));
      CHKERRQ(PetscMPIIntCast(m*n,&len));
      CHKERRMPI(MPIU_Allreduce(bv->work,C,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)bv)));
    } else {
      CHKERRQ(BVAllocateWork_Private(bv,2*m*n));
      CC = bv->work+m*n;
      if (k) PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&m,&n,&k,&one,(PetscScalar*)A,&k,(PetscScalar*)B,&k,&zero,bv->work,&m));
      else CHKERRQ(PetscArrayzero(bv->work,m*n));
      CHKERRQ(PetscMPIIntCast(m*n,&len));
      CHKERRMPI(MPIU_Allreduce(bv->work,CC,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)bv)));
      for (j=0;j<n;j++) CHKERRQ(PetscArraycpy(C+j*ldc,CC+j*m,m));
    }
  } else {
    if (k) PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&m,&n,&k,&one,(PetscScalar*)A,&k,(PetscScalar*)B,&k,&zero,C,&ldc));
  }
  CHKERRQ(PetscLogFlops(2.0*m*n*k));
  PetscFunctionReturn(0);
}

/*
    y := A'*x

    A is nxk (ld=n)
*/
PetscErrorCode BVDotVec_BLAS_Private(BV bv,PetscInt n_,PetscInt k_,const PetscScalar *A,const PetscScalar *x,PetscScalar *y,PetscBool mpi)
{
  PetscScalar    zero=0.0,done=1.0;
  PetscBLASInt   n,k,one=1;
  PetscMPIInt    len;

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(n_,&n));
  CHKERRQ(PetscBLASIntCast(k_,&k));
  if (mpi) {
    CHKERRQ(BVAllocateWork_Private(bv,k));
    if (n) PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&k,&done,A,&n,x,&one,&zero,bv->work,&one));
    else CHKERRQ(PetscArrayzero(bv->work,k));
    CHKERRQ(PetscMPIIntCast(k,&len));
    CHKERRMPI(MPIU_Allreduce(bv->work,y,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)bv)));
  } else {
    if (n) PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&k,&done,A,&n,x,&one,&zero,y,&one));
  }
  CHKERRQ(PetscLogFlops(2.0*n*k));
  PetscFunctionReturn(0);
}

/*
    Scale n scalars
*/
PetscErrorCode BVScale_BLAS_Private(BV bv,PetscInt n_,PetscScalar *A,PetscScalar alpha)
{
  PetscBLASInt   n,one=1;

  PetscFunctionBegin;
  if (PetscUnlikely(alpha == (PetscScalar)0.0)) CHKERRQ(PetscArrayzero(A,n_));
  else if (alpha!=(PetscScalar)1.0) {
    CHKERRQ(PetscBLASIntCast(n_,&n));
    PetscStackCallBLAS("BLASscal",BLASscal_(&n,&alpha,A,&one));
    CHKERRQ(PetscLogFlops(n));
  }
  PetscFunctionReturn(0);
}
