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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Q);CHKERRQ(ierr); 
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
  PetscErrorCode ierr;
  PetscScalar    *work,*tau;
  PetscInt       i,j;
  PetscBLASInt   ilo,lwork,info,n,ld;
  PetscScalar    *A = ps->mat[PS_MAT_A];
  PetscScalar    *Q = ps->mat[PS_MAT_Q];

  PetscFunctionBegin;
  n   = PetscBLASIntCast(ps->n);
  ld  = PetscBLASIntCast(ps->ld);
  ilo = PetscBLASIntCast(ps->l+1);
  ierr = PSAllocateWork_Private(ps,2*ld,0,0);CHKERRQ(ierr); 
  work = ps->work;
  tau  = ps->work+ps->ld;
  lwork = ld;
  /* reduce to upper Hessenberg form */
  if (ps->state<PS_STATE_INTERMEDIATE) {
    LAPACKgehrd_(&n,&ilo,&n,A,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEHRD %d",info);
    for (j=0;j<n-1;j++) {
      for (i=j+2;i<n;i++) {
        Q[i+j*ld] = A[i+j*ld];
        A[i+j*ld] = 0.0;
      }
    }
    LAPACKorghr_(&n,&ilo,&n,Q,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGHR %d",info);
  } else {
    /* initialize orthogonal matrix */
    ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<n;i++) 
      Q[i+i*ld] = 1.0;
  }
  /* compute the (real) Schur form */
  if (ps->state<PS_STATE_CONDENSED) {
#if !defined(PETSC_USE_COMPLEX)
    LAPACKhseqr_("S","V",&n,&ilo,&n,A,&ld,wr,wi,Q,&ld,work,&lwork,&info);
    for (j=0;j<ps->l;j++) {
      if (j==n-1 || A[j+1+j*ld] == 0.0) { 
        /* real eigenvalue */
        wr[j] = A[j+j*ld];
        wi[j] = 0.0;
      } else {
        /* complex eigenvalue */
        wr[j] = A[j+j*ld];
        wr[j+1] = A[j+j*ld];
        wi[j] = PetscSqrtReal(PetscAbsReal(A[j+1+j*ld])) *
                PetscSqrtReal(PetscAbsReal(A[j+(j+1)*ld]));
        wi[j+1] = -wi[j];
        j++;
      }
    }
#else
    LAPACKhseqr_("S","V",&n,&ilo,&n,A,&ld,wr,Q,&ld,work,&lwork,&info);
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
  PetscBLASInt   ifst,ilst,info,n,ld;
  PetscScalar    *T = ps->mat[PS_MAT_A];
  PetscScalar    *Q = ps->mat[PS_MAT_Q];
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *work;
#endif

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PSAllocateWork_Private(ps,ld,0,0);CHKERRQ(ierr); 
  work = ps->work;
#endif
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
      ifst = PetscBLASIntCast(pos+1);
      ilst = PetscBLASIntCast(i+1);
#if !defined(PETSC_USE_COMPLEX)
      LAPACKtrexc_("V",&n,T,&ld,Q,&ld,&ifst,&ilst,work,&info);
#else
      LAPACKtrexc_("V",&n,T,&ld,Q,&ld,&ifst,&ilst,&info);
#endif
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);
      /* recover original eigenvalues from T matrix */
      for (j=i;j<n;j++) {
        wr[j] = T[j+j*ld];
#if !defined(PETSC_USE_COMPLEX)
        if (j<n-1 && T[j+1+j*ld] != 0.0) {
          /* complex conjugate eigenvalue */
          wi[j] = PetscSqrtReal(PetscAbsReal(T[j+1+j*ld])) *
                  PetscSqrtReal(PetscAbsReal(T[j+(j+1)*ld]));
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

#undef __FUNCT__  
#define __FUNCT__ "PSCond_NHEP"
PetscErrorCode PSCond_NHEP(PS ps,PetscReal *cond)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) || defined(SLEPC_MISSING_LAPACK_GETRI) || defined(SLEPC_MISSING_LAPACK_LANGE) || defined(SLEPC_MISSING_LAPACK_LANHS)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRF/GETRI/LANGE/LANHS - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    *work;
  PetscReal      *rwork;
  PetscBLASInt   *ipiv;
  PetscBLASInt   lwork,info,n,ld;
  PetscReal      hn,hin;
  PetscScalar    *A;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  lwork = 8*ld;
  ierr = PSAllocateWork_Private(ps,lwork,ld,ld);CHKERRQ(ierr); 
  work  = ps->work;
  rwork = ps->rwork;
  ipiv  = ps->iwork;

  /* use workspace matrix W to avoid overwriting A */
  if (!ps->mat[PS_MAT_W]) {
    ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr); 
  }
  A = ps->mat[PS_MAT_W];
  ierr = PetscMemcpy(A,ps->mat[PS_MAT_A],sizeof(PetscScalar)*ps->ld*ps->ld);CHKERRQ(ierr);

  /* norm of A */
  if (ps->state<PS_STATE_INTERMEDIATE) hn = LAPACKlange_("I",&n,&n,A,&ld,rwork);
  else hn = LAPACKlanhs_("I",&n,A,&ld,rwork);

  /* norm of inv(A) */
  LAPACKgetrf_(&n,&n,A,&ld,ipiv,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGETRF %d",info);
  LAPACKgetri_(&n,A,&ld,ipiv,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGETRI %d",info);
  hin = LAPACKlange_("I",&n,&n,A,&ld,rwork);

  *cond = hn*hin;
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
  ps->ops->cond          = PSCond_NHEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

