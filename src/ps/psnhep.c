/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/psimpl.h>      /*I "slepcps.h" I*/
#include <slepcblaslapack.h>

#undef __FUNCT__  
#define __FUNCT__ "PSAllocate_NHEP"
PetscErrorCode PSAllocate_NHEP(PS ps,PetscInt ld)
{
  PetscInt       sz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Q);CHKERRQ(ierr); 
  /* workspace:
       work[0..ld] - workspace for GEHRD, ORGHR, HSEQR, TREXC
       work[ld+1..2*ld] - scalar factors of the elementary reflectors (GEHRD)
   */
  sz = 2*ld*sizeof(PetscScalar);
  ierr = PetscMalloc(sz,&ps->work);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ps,sz);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_NHEP"
PetscErrorCode PSSolve_NHEP(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GEHRD) || defined(SLEPC_MISSING_LAPACK_ORGHR) || defined(PETSC_MISSING_LAPACK_HSEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEHRD/ORGHR/HSEQR - Lapack routines are unavailable.");
#else
  PetscScalar    *work=ps->work,*tau=ps->work+ps->ld;
  PetscInt       i,j;
  PetscBLASInt   ilo,lwork,info,n,lda;
  PetscScalar    *A = ps->mat[PS_MAT_A];
  PetscScalar    *Q = ps->mat[PS_MAT_Q];

  PetscFunctionBegin;
  n = PetscBLASIntCast(ps->n);
  lda = PetscBLASIntCast(ps->ld);
  ilo = PetscBLASIntCast(ps->l+1);
  lwork = n;
  /* reduce to upper Hessenberg form */
  if (ps->state<PS_STATE_INTERMEDIATE) {
    LAPACKgehrd_(&n,&ilo,&n,A,&lda,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEHRD %d",info);
    for (j=0;j<n-1;j++) {
      for (i=j+2;i<n;i++) {
        Q[i+j*n] = A[i+j*lda];
        A[i+j*lda] = 0.0;
      }
    }
    LAPACKorghr_(&n,&ilo,&n,Q,&n,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGHR %d",info);
  }
  /* compute the (real) Schur form */
  if (ps->state<PS_STATE_CONDENSED) {
#if !defined(PETSC_USE_COMPLEX)
    LAPACKhseqr_("S","V",&n,&ilo,&n,A,&lda,wr,wi,Q,&n,work,&lwork,&info);
    for (j=0;j<ps->l;j++) {
      if (j==n-1 || A[j*lda+j+1] == 0.0) { 
        /* real eigenvalue */
        wr[j] = A[j*lda+j];
        wi[j] = 0.0;
      } else {
        /* complex eigenvalue */
        wr[j] = A[j*lda+j];
        wr[j+1] = A[j*lda+j];
        wi[j] = PetscSqrtReal(PetscAbsReal(A[j*lda+j+1])) *
                PetscSqrtReal(PetscAbsReal(A[(j+1)*lda+j]));
        wi[j+1] = -wi[j];
        j++;
      }
    }
#else
    LAPACKhseqr_("S","V",&n,&ilo,&n,A,&lda,wr,Q,&n,work,&lwork,&info);
#endif
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",info);
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSSort_NHEP"
PetscErrorCode PSSort_NHEP(PS ps,PetscScalar *wr,PetscScalar *wi,PetscErrorCode (*comp_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void *comp_ctx)
{
#if defined(SLEPC_MISSING_LAPACK_TREXC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TREXC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,pos,result;
  PetscBLASInt   ifst,ilst,info,n,ldt;
  PetscScalar    *T = ps->mat[PS_MAT_A];
  PetscScalar    *Q = ps->mat[PS_MAT_Q];
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *work=ps->work;
#endif

  PetscFunctionBegin;
  n = PetscBLASIntCast(ps->n);
  ldt = PetscBLASIntCast(ps->ld);
  /* selection sort */
  for (i=ps->l;i<n-1;i++) {
    re = wr[i];
    im = wi[i];
    pos = 0;
    j=i+1; /* j points to the next eigenvalue */
#if !defined(PETSC_USE_COMPLEX)
    if (im != 0) j=i+2;
#endif
    /* find minimum eigenvalue */
    for (;j<n;j++) { 
      ierr = (*comp_func)(re,im,wr[j],wi[j],&result,comp_ctx);CHKERRQ(ierr);
      if (result > 0) {
        re = wr[j];
        im = wi[j];
        pos = j;
      }
#if !defined(PETSC_USE_COMPLEX)
      if (wi[j] != 0) j++;
#endif
    }
    if (pos) {
      /* interchange blocks */
      ifst = PetscBLASIntCast(pos + 1);
      ilst = PetscBLASIntCast(i + 1);
#if !defined(PETSC_USE_COMPLEX)
      LAPACKtrexc_("V",&n,T,&ldt,Q,&n,&ifst,&ilst,work,&info);
#else
      LAPACKtrexc_("V",&n,T,&ldt,Q,&n,&ifst,&ilst,&info);
#endif
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);
      /* recover original eigenvalues from T matrix */
      for (j=i;j<n;j++) {
        wr[j] = T[j*ldt+j];
#if !defined(PETSC_USE_COMPLEX)
        if (j<n-1 && T[j*ldt+j+1] != 0.0) {
          /* complex conjugate eigenvalue */
          wi[j] = PetscSqrtReal(PetscAbsReal(T[j*ldt+j+1])) *
                  PetscSqrtReal(PetscAbsReal(T[(j+1)*ldt+j]));
          wr[j+1] = wr[j];
          wi[j+1] = -wi[j];
          j++;
        } else
#endif
        wi[j] = 0.0;
      }
    }
#if !defined(PETSC_USE_COMPLEX)
    if (wi[i] != 0) i++;
#endif
  }
  PetscFunctionReturn(0);
#endif 
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PSCreate_NHEP"
PetscErrorCode PSCreate_NHEP(PS ps)
{
  PetscFunctionBegin;
  ps->ops->allocate      = PSAllocate_NHEP;
  //ps->ops->computevector = PSComputeVector_NHEP;
  ps->ops->solve         = PSSolve_NHEP;
  ps->ops->sort          = PSSort_NHEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

