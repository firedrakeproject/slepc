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
#define __FUNCT__ "PSAllocate_HEP"
PetscErrorCode PSAllocate_HEP(PS ps,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Q);CHKERRQ(ierr); 
  ierr = PSAllocateMatReal_Private(ps,PS_MAT_T);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSView_HEP"
PetscErrorCode PSView_HEP(PS ps,PetscViewer viewer)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PSViewMat_Private(ps,viewer,PS_MAT_A);CHKERRQ(ierr); 
  if (ps->state>PS_STATE_INTERMEDIATE) {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_Q);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_HEP"
PetscErrorCode PSSolve_HEP(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_SYTRD) || defined(SLEPC_MISSING_LAPACK_ORGTR) || defined(SLEPC_MISSING_LAPACK_STEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SYTRD/ORGTR/STEQR - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   lwork,info,n,ld;
  PetscScalar    *A,*Q,*work,*tau;
  PetscReal      *d,*e;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  A  = ps->mat[PS_MAT_A];
  Q  = ps->mat[PS_MAT_Q];
  d  = ps->rmat[PS_MAT_T];
  e  = ps->rmat[PS_MAT_T]+ld;
  if (n==1) { d[0] = A[0]; Q[0] = 1.0; PetscFunctionReturn(0); }

  if (ps->state<PS_STATE_INTERMEDIATE) {
    /* reduce to tridiagonal form */
    ierr = PetscMemcpy(Q,A,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PSAllocateWork_Private(ps,ld+ld*ld,0,0);CHKERRQ(ierr); 
    tau  = ps->work;
    work = ps->work+ld;
    lwork = ld*ld;
    LAPACKsytrd_("L",&n,Q,&ld,d,e,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSYTRD %d",info);
    LAPACKorgtr_("L",&n,Q,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGTR %d",info);
  } else {
    /* initialize orthogonal matrix; copy tridiagonal to d,e */
    ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<n;i++) Q[i+i*ld] = 1.0;
    for (i=0;i<n;i++) d[i] = A[i+i*ld];
    for (i=0;i<n-1;i++) e[i] = A[(i+1)+i*ld];
  }

  /* Solve the tridiagonal eigenproblem */
  ierr = PSAllocateWork_Private(ps,0,2*ld,0);CHKERRQ(ierr); 
  LAPACKsteqr_("V",&n,d,e,Q,&ld,ps->rwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSTEQR %d",info);
  for (i=0;i<n;i++) wr[i] = d[i];
  ierr = PetscMemzero(A,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<n;i++) A[i+i*ld] = d[i];
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSSort_HEP"
PetscErrorCode PSSort_HEP(PS ps,PetscScalar *wr,PetscScalar *wi,PetscErrorCode (*comp_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void *comp_ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,p,*perm;
  PetscBLASInt   n,ld;
  PetscScalar    *A,*Q;
  PetscReal      *d,rtmp;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  A  = ps->mat[PS_MAT_A];
  Q  = ps->mat[PS_MAT_Q];
  d  = ps->rmat[PS_MAT_T];
  ierr = PSAllocateWork_Private(ps,0,0,ld);CHKERRQ(ierr); 
  perm = ps->iwork;
  ierr = PSSortEigenvaluesReal_Private(ps,n,d,perm,comp_func,comp_ctx);CHKERRQ(ierr);
  for (i=0;i<n;i++)
    wr[i] = d[perm[i]];
  for (i=0;i<n;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap eigenvectors i and j */
      for (k=0;k<n;k++) {
        rtmp = Q[k+p*ld]; Q[k+p*ld] = Q[k+i*ld]; Q[k+i*ld] = rtmp;
      }
    }
  }
  for (i=0;i<n;i++) A[i+i*ld] = wr[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSCond_HEP"
PetscErrorCode PSCond_HEP(PS ps,PetscReal *cond)
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
  if (!ps->mat[PS_MAT_W]) { ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr); }
  A = ps->mat[PS_MAT_W];
  ierr = PetscMemcpy(A,ps->mat[PS_MAT_A],sizeof(PetscScalar)*ps->ld*ps->ld);CHKERRQ(ierr);

  /* norm of A */
  //if (ps->state<PS_STATE_INTERMEDIATE)
  hn = LAPACKlange_("I",&n,&n,A,&ld,rwork);
  //else hn = LAPACKlanhs_("I",&n,A,&ld,rwork);

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
#define __FUNCT__ "PSCreate_HEP"
PetscErrorCode PSCreate_HEP(PS ps)
{
  PetscFunctionBegin;
  ps->ops->allocate      = PSAllocate_HEP;
  ps->ops->view          = PSView_HEP;
  //ps->ops->computevector = PSComputeVector_HEP;
  ps->ops->solve         = PSSolve_HEP;
  ps->ops->sort          = PSSort_HEP;
  ps->ops->cond          = PSCond_HEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

