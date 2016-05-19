/*
   BV private kernels that use the BLAS.

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

#include <slepc/private/bvimpl.h>
#include <slepcblaslapack.h>

#define BLOCKSIZE 64

#undef __FUNCT__
#define __FUNCT__ "BVMult_BLAS_Private"
/*
    C := alpha*A*B + beta*C

    A is mxk (ld=m), B is kxn (ld=ldb), C is mxn (ld=m)
*/
PetscErrorCode BVMult_BLAS_Private(BV bv,PetscInt m_,PetscInt n_,PetscInt k_,PetscInt ldb_,PetscScalar alpha,const PetscScalar *A,const PetscScalar *B,PetscScalar beta,PetscScalar *C)
{
  PetscErrorCode ierr;
  PetscBLASInt   m,n,k,ldb;
#if defined(PETSC_HAVE_FBLASLAPACK) || defined(PETSC_HAVE_F2CBLASLAPACK)
  PetscBLASInt   l,bs=BLOCKSIZE;
#endif

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(m_,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k_,&k);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldb_,&ldb);CHKERRQ(ierr);
#if defined(PETSC_HAVE_FBLASLAPACK) || defined(PETSC_HAVE_F2CBLASLAPACK)
  l = m % bs;
  if (l) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&k,&alpha,(PetscScalar*)A,&m,(PetscScalar*)B,&ldb,&beta,C,&m));
  for (;l<m;l+=bs) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&bs,&n,&k,&alpha,(PetscScalar*)A+l,&m,(PetscScalar*)B,&ldb,&beta,C+l,&m));
  }
#else
  if (m) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&m,&n,&k,&alpha,(PetscScalar*)A,&m,(PetscScalar*)B,&ldb,&beta,C,&m));
#endif
  ierr = PetscLogFlops(2.0*m*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultVec_BLAS_Private"
/*
    y := alpha*A*x + beta*y

    A is nxk (ld=n)
*/
PetscErrorCode BVMultVec_BLAS_Private(BV bv,PetscInt n_,PetscInt k_,PetscScalar alpha,const PetscScalar *A,const PetscScalar *x,PetscScalar beta,PetscScalar *y)
{
  PetscErrorCode ierr;
  PetscBLASInt   n,k,one=1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k_,&k);CHKERRQ(ierr);
  if (n) PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&k,&alpha,A,&n,x,&one,&beta,y,&one));
  ierr = PetscLogFlops(2.0*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultInPlace_BLAS_Private"
/*
    A(:,s:e-1) := A*B(:,s:e-1)

    A is mxk (ld=m), B is kxn (ld=ldb)  n=e-s
*/
PetscErrorCode BVMultInPlace_BLAS_Private(BV bv,PetscInt m_,PetscInt k_,PetscInt ldb_,PetscInt s,PetscInt e,PetscScalar *A,const PetscScalar *B,PetscBool btrans)
{
  PetscErrorCode ierr;
  PetscScalar    *pb,zero=0.0,one=1.0;
  PetscBLASInt   m,n,k,l,ldb,bs=BLOCKSIZE;
  PetscInt       j,n_=e-s;
  const char     *bt;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(m_,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k_,&k);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldb_,&ldb);CHKERRQ(ierr);
  ierr = BVAllocateWork_Private(bv,BLOCKSIZE*n_);CHKERRQ(ierr);
  if (btrans) {
    pb = (PetscScalar*)B+s;
    bt = "C";
  } else {
    pb = (PetscScalar*)B+s*ldb;
    bt = "N";
  }
  l = m % bs;
  if (l) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N",bt,&l,&n,&k,&one,A,&m,pb,&ldb,&zero,bv->work,&l));
    for (j=0;j<n;j++) {
      ierr = PetscMemcpy(A+(s+j)*m,bv->work+j*l,l*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  for (;l<m;l+=bs) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N",bt,&bs,&n,&k,&one,A+l,&m,pb,&ldb,&zero,bv->work,&bs));
    for (j=0;j<n;j++) {
      ierr = PetscMemcpy(A+(s+j)*m+l,bv->work+j*bs,bs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  ierr = PetscLogFlops(2.0*m*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultInPlace_Vecs_Private"
/*
    V := V*B

    V is mxn (ld=m), B is nxn (ld=k)
*/
PetscErrorCode BVMultInPlace_Vecs_Private(BV bv,PetscInt m_,PetscInt n_,PetscInt k_,Vec *V,const PetscScalar *B,PetscBool btrans)
{
  PetscErrorCode    ierr;
  PetscScalar       zero=0.0,one=1.0,*out,*pout;
  const PetscScalar *pin;
  PetscBLASInt      m,n,k,l,bs=BLOCKSIZE;
  PetscInt          j;
  const char        *bt;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(m_,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k_,&k);CHKERRQ(ierr);
  ierr = BVAllocateWork_Private(bv,2*BLOCKSIZE*n_);CHKERRQ(ierr);
  out = bv->work+BLOCKSIZE*n_;
  if (btrans) bt = "C";
  else bt = "N";
  l = m % bs;
  if (l) {
    for (j=0;j<n;j++) {
      ierr = VecGetArrayRead(V[j],&pin);CHKERRQ(ierr);
      ierr = PetscMemcpy(bv->work+j*l,pin,l*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(V[j],&pin);CHKERRQ(ierr);
    }
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N",bt,&l,&n,&n,&one,bv->work,&l,(PetscScalar*)B,&k,&zero,out,&l));
    for (j=0;j<n;j++) {
      ierr = VecGetArray(V[j],&pout);CHKERRQ(ierr);
      ierr = PetscMemcpy(pout,out+j*l,l*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecRestoreArray(V[j],&pout);CHKERRQ(ierr);
    }
  }
  for (;l<m;l+=bs) {
    for (j=0;j<n;j++) {
      ierr = VecGetArrayRead(V[j],&pin);CHKERRQ(ierr);
      ierr = PetscMemcpy(bv->work+j*bs,pin+l,bs*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(V[j],&pin);CHKERRQ(ierr);
    }
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N",bt,&bs,&n,&n,&one,bv->work,&bs,(PetscScalar*)B,&k,&zero,out,&bs));
    for (j=0;j<n;j++) {
      ierr = VecGetArray(V[j],&pout);CHKERRQ(ierr);
      ierr = PetscMemcpy(pout+l,out+j*bs,bs*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecRestoreArray(V[j],&pout);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogFlops(2.0*n*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVAXPY_BLAS_Private"
/*
    B := alpha*A + beta*B

    A,B are nxk (ld=n)
*/
PetscErrorCode BVAXPY_BLAS_Private(BV bv,PetscInt n_,PetscInt k_,PetscScalar alpha,const PetscScalar *A,PetscScalar beta,PetscScalar *B)
{
  PetscErrorCode ierr;
  PetscBLASInt   m,one=1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n_*k_,&m);CHKERRQ(ierr);
  if (beta!=(PetscScalar)1.0) {
    PetscStackCallBLAS("BLASscal",BLASscal_(&m,&beta,B,&one));
    ierr = PetscLogFlops(m);CHKERRQ(ierr);
  }
  PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&m,&alpha,A,&one,B,&one));
  ierr = PetscLogFlops(2.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDot_BLAS_Private"
/*
    C := A'*B

    A' is mxk (ld=k), B is kxn (ld=k), C is mxn (ld=ldc)
*/
PetscErrorCode BVDot_BLAS_Private(BV bv,PetscInt m_,PetscInt n_,PetscInt k_,PetscInt ldc_,const PetscScalar *A,const PetscScalar *B,PetscScalar *C,PetscBool mpi)
{
  PetscErrorCode ierr;
  PetscScalar    zero=0.0,one=1.0,*CC;
  PetscBLASInt   m,n,k,ldc,j,len;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(m_,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k_,&k);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldc_,&ldc);CHKERRQ(ierr);
  if (mpi) {
    if (ldc==m) {
      ierr = BVAllocateWork_Private(bv,m*n);CHKERRQ(ierr);
      if (k) PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&m,&n,&k,&one,(PetscScalar*)A,&k,(PetscScalar*)B,&k,&zero,bv->work,&ldc));
      else { ierr = PetscMemzero(bv->work,m*n*sizeof(PetscScalar));CHKERRQ(ierr); }
      ierr = PetscMPIIntCast(m*n,&len);CHKERRQ(ierr);
      ierr = MPI_Allreduce(bv->work,C,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
    } else {
      ierr = BVAllocateWork_Private(bv,2*m*n);CHKERRQ(ierr);
      CC = bv->work+m*n;
      if (k) PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&m,&n,&k,&one,(PetscScalar*)A,&k,(PetscScalar*)B,&k,&zero,bv->work,&m));
      else { ierr = PetscMemzero(bv->work,m*n*sizeof(PetscScalar));CHKERRQ(ierr); }
      ierr = PetscMPIIntCast(m*n,&len);CHKERRQ(ierr);
      ierr = MPI_Allreduce(bv->work,CC,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
      for (j=0;j<n;j++) {
        ierr = PetscMemcpy(C+j*ldc,CC+j*m,m*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    }
  } else {
    if (k) PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&m,&n,&k,&one,(PetscScalar*)A,&k,(PetscScalar*)B,&k,&zero,C,&ldc));
  }
  ierr = PetscLogFlops(2.0*m*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec_BLAS_Private"
/*
    y := A'*x

    A is nxk (ld=n)
*/
PetscErrorCode BVDotVec_BLAS_Private(BV bv,PetscInt n_,PetscInt k_,const PetscScalar *A,const PetscScalar *x,PetscScalar *y,PetscBool mpi)
{
  PetscErrorCode ierr;
  PetscScalar    zero=0.0,done=1.0;
  PetscBLASInt   n,k,one=1,len;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k_,&k);CHKERRQ(ierr);
  if (mpi) {
    ierr = BVAllocateWork_Private(bv,k);CHKERRQ(ierr);
    if (n) {
      PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&k,&done,A,&n,x,&one,&zero,bv->work,&one));
    } else {
      ierr = PetscMemzero(bv->work,k*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = PetscMPIIntCast(k,&len);CHKERRQ(ierr);
    ierr = MPI_Allreduce(bv->work,y,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
  } else {
    if (n) PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&k,&done,A,&n,x,&one,&zero,y,&one));
  }
  ierr = PetscLogFlops(2.0*n*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVScale_BLAS_Private"
/*
    Scale n scalars
*/
PetscErrorCode BVScale_BLAS_Private(BV bv,PetscInt n_,PetscScalar *A,PetscScalar alpha)
{
  PetscErrorCode ierr;
  PetscBLASInt   n,one=1;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = PetscMemzero(A,n_*sizeof(PetscScalar));CHKERRQ(ierr);
  } else if (alpha!=(PetscScalar)1.0) {
    ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASscal",BLASscal_(&n,&alpha,A,&one));
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVNorm_LAPACK_Private"
/*
    Compute ||A|| for an mxn matrix
*/
PetscErrorCode BVNorm_LAPACK_Private(BV bv,PetscInt m_,PetscInt n_,const PetscScalar *A,NormType type,PetscReal *nrm,PetscBool mpi)
{
  PetscErrorCode ierr;
  PetscBLASInt   m,n,i,j;
  PetscMPIInt    len;
  PetscReal      lnrm,*rwork=NULL,*rwork2=NULL;

  PetscFunctionBegin;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m_,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  if (type==NORM_FROBENIUS || type==NORM_2) {
    lnrm = LAPACKlange_("F",&m,&n,(PetscScalar*)A,&m,rwork);
    if (mpi) {
      lnrm = lnrm*lnrm;
      ierr = MPI_Allreduce(&lnrm,nrm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
      *nrm = PetscSqrtReal(*nrm);
    } else *nrm = lnrm;
    ierr = PetscLogFlops(2.0*m*n);CHKERRQ(ierr);
  } else if (type==NORM_1) {
    if (mpi) {
      ierr = BVAllocateWork_Private(bv,2*n_);CHKERRQ(ierr);
      rwork = (PetscReal*)bv->work;
      rwork2 = rwork+n_;
      ierr = PetscMemzero(rwork,n_*sizeof(PetscReal));CHKERRQ(ierr);
      ierr = PetscMemzero(rwork2,n_*sizeof(PetscReal));CHKERRQ(ierr);
      for (j=0;j<n_;j++) {
        for (i=0;i<m_;i++) {
          rwork[j] += PetscAbsScalar(A[i+j*m_]);
        }
      }
      ierr = PetscMPIIntCast(n_,&len);CHKERRQ(ierr);
      ierr = MPI_Allreduce(rwork,rwork2,len,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
      *nrm = 0.0;
      for (j=0;j<n_;j++) if (rwork2[j] > *nrm) *nrm = rwork2[j];
    } else {
      *nrm = LAPACKlange_("O",&m,&n,(PetscScalar*)A,&m,rwork);
    }
    ierr = PetscLogFlops(1.0*m*n);CHKERRQ(ierr);
  } else if (type==NORM_INFINITY) {
    ierr = BVAllocateWork_Private(bv,m_);CHKERRQ(ierr);
    rwork = (PetscReal*)bv->work;
    lnrm = LAPACKlange_("I",&m,&n,(PetscScalar*)A,&m,rwork);
    if (mpi) {
      ierr = MPI_Allreduce(&lnrm,nrm,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
    } else *nrm = lnrm;
    ierr = PetscLogFlops(1.0*m*n);CHKERRQ(ierr);
  }
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalize_LAPACK_Private"
/*
    QR factorization of an mxn matrix
*/
PetscErrorCode BVOrthogonalize_LAPACK_Private(BV bv,PetscInt m_,PetscInt n_,PetscScalar *Q,PetscScalar *R,PetscBool mpi)
{
#if defined(PETSC_MISSING_LAPACK_GEQRF) || defined(PETSC_MISSING_LAPACK_ORGQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEQRF/ORGQR - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscBLASInt   m,n,i,j,k,l,nb,lwork,info;
  PetscScalar    *tau,*work,*Rl=NULL,*A=NULL,*C=NULL,one=1.0,zero=0.0;
  PetscMPIInt    rank,size,len;

  PetscFunctionBegin;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m_,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  k = PetscMin(m,n);
  nb = 16;
  if (mpi) {
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)bv),&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)bv),&size);CHKERRQ(ierr);
    ierr = BVAllocateWork_Private(bv,k+n*nb+n*n+n*n*size+m*n);CHKERRQ(ierr);
  } else {
    ierr = BVAllocateWork_Private(bv,k+n*nb);CHKERRQ(ierr);
   }
  tau = bv->work;
  work = bv->work+k;
  ierr = PetscBLASIntCast(n*nb,&lwork);CHKERRQ(ierr);
  if (mpi) {
    Rl = bv->work+k+n*nb;
    A  = bv->work+k+n*nb+n*n;
    C  = bv->work+k+n*nb+n*n+n*n*size;
  }

  /* Compute QR */
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&m,&n,Q,&m,tau,work,&lwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);

  /* Extract R */
  if (R || mpi) {
    ierr = PetscMemzero(mpi? Rl: R,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
    for (j=0;j<n;j++) {
      for (i=0;i<=j;i++) {
        if (mpi) Rl[i+j*n] = Q[i+j*m];
        else R[i+j*n] = Q[i+j*m];
      }
    }
  }

  /* Compute orthogonal matrix in Q */
  PetscStackCallBLAS("LAPACKungqr",LAPACKungqr_(&m,&n,&k,Q,&m,tau,work,&lwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGQR %d",info);

  if (mpi) {

    /* Stack triangular matrices */
    ierr = PetscBLASIntCast(n*size,&l);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(n,&len);CHKERRQ(ierr);
    for (j=0;j<n;j++) {
      ierr = MPI_Allgather(Rl+j*n,len,MPIU_SCALAR,A+j*l,len,MPIU_SCALAR,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
    }

    /* Compute QR */
    PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&l,&n,A,&l,tau,work,&lwork,&info));
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);

    /* Extract R */
    if (R) {
      ierr = PetscMemzero(R,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
      for (j=0;j<n;j++)
        for (i=0;i<=j;i++)
          R[i+j*n] = A[i+j*l];
    }

    /* Accumulate orthogonal matrix */
    PetscStackCallBLAS("LAPACKungqr",LAPACKungqr_(&l,&n,&n,A,&l,tau,work,&lwork,&info));
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGQR %d",info);
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&m,&n,&n,&one,Q,&m,A+rank*n,&l,&zero,C,&m));
    ierr = PetscMemcpy(Q,C,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  ierr = PetscLogFlops(3.0*m*n*n);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

