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
#define __FUNCT__ "PSAllocate_SVD"
PetscErrorCode PSAllocate_SVD(PS ps,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_U);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_VT);CHKERRQ(ierr); 
  ierr = PSAllocateMatReal_Private(ps,PS_MAT_T);CHKERRQ(ierr); 
  ierr = PetscFree(ps->perm);CHKERRQ(ierr);
  ierr = PetscMalloc(ld*sizeof(PetscInt),&ps->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ps,ld*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSView_SVD"
PetscErrorCode PSView_SVD(PS ps,PetscViewer viewer)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PSViewMat_Private(ps,viewer,PS_MAT_A);CHKERRQ(ierr); 
  if (ps->state>PS_STATE_INTERMEDIATE) {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_U);CHKERRQ(ierr); 
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_VT);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_SVD"
PetscErrorCode PSVectors_SVD(PS ps,PSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscFunctionBegin;
  if (ps->state<PS_STATE_CONDENSED) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSSolve() first");
  switch (mat) {
    case PS_MAT_U:
    case PS_MAT_VT:
      break;
    default:
      SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_SVD_Sort"
/*
  Sort the singular value decomposition at the end of any PSSolve_SVD_* method. 
*/
static PetscErrorCode PSSolve_SVD_Sort(PS ps,PetscScalar *wr)
{
  PetscErrorCode ierr;
  PetscInt       n,l,i,*perm;
  PetscReal      *d;

  PetscFunctionBegin;
  if (!ps->comp_fun) PetscFunctionReturn(0);
  l = ps->l;
  n = PetscMin(ps->n,ps->m);
  d = ps->rmat[PS_MAT_T];
  perm = ps->perm;
  ierr = PSSortEigenvaluesReal_Private(ps,l,n,d,perm);CHKERRQ(ierr);
  for (i=l;i<n;i++) wr[i] = d[perm[i]];
  ierr = PSPermuteColumns_Private(ps,l,n,PS_MAT_U,perm);CHKERRQ(ierr);
  for (i=l;i<n;i++) d[i] = PetscRealPart(wr[i]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_SVD_DC"
PetscErrorCode PSSolve_SVD_DC(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GESDD) || defined(SLEPC_MISSING_LAPACK_BDSDC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESDD/BDSDC - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n1,n2,n3,info,l,n,m,nm,ld,off,lwork;
  PetscScalar    *A,*U,*VT,qwork;
  PetscReal      *d,*e,*Ur,*VTr;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       j;
#endif

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  m  = PetscBLASIntCast(ps->m);
  l  = PetscBLASIntCast(ps->l);
  ld = PetscBLASIntCast(ps->ld);
  n1 = PetscBLASIntCast(ps->k-l+1);  /* size of leading block, excluding locked */
  n2 = PetscBLASIntCast(n-ps->k-1);  /* size of trailing block */
  n3 = n1+n2;
  off = l+l*ld;
  A  = ps->mat[PS_MAT_A];
  U  = ps->mat[PS_MAT_U];
  VT = ps->mat[PS_MAT_VT];
  d  = ps->rmat[PS_MAT_T];
  e  = ps->rmat[PS_MAT_T]+ld;

  if (ps->state>PS_STATE_RAW) {

    /* Solve bidiagonal SVD problem */
    for (i=0;i<l;i++) wr[i] = d[i];
    ierr = PSSetIdentity(ps,PS_MAT_U);CHKERRQ(ierr); 
    ierr = PSSetIdentity(ps,PS_MAT_VT);CHKERRQ(ierr); 
    ierr = PSAllocateWork_Private(ps,0,3*ld*ld+4*ld,8*ld);CHKERRQ(ierr); 
#if defined(PETSC_USE_COMPLEX)
    ierr = PSAllocateMatReal_Private(ps,PS_MAT_U);CHKERRQ(ierr); 
    ierr = PSAllocateMatReal_Private(ps,PS_MAT_VT);CHKERRQ(ierr); 
    Ur  = ps->rmat[PS_MAT_U];
    VTr = ps->rmat[PS_MAT_VT];
#else
    Ur  = U;
    VTr = VT;
#endif
    LAPACKbdsdc_("U","I",&n3,d,e,Ur+off,&ld,VTr+off,&ld,PETSC_NULL,PETSC_NULL,ps->rwork,ps->iwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xBDSDC %d",info);
#if defined(PETSC_USE_COMPLEX)
    for (i=l;i<n;i++) {
      for (j=0;j<n;j++) {
        U[i+j*ld] = Ur[i+j*ld];
        VT[i+j*ld] = VTr[i+j*ld];
      }
    }
#endif

  } else {

    /* Solve general rectangular SVD problem */
    nm = PetscMin(n,m);
    ierr = PSAllocateWork_Private(ps,0,0,8*nm);CHKERRQ(ierr); 
    lwork = -1;
#if defined(PETSC_USE_COMPLEX)
    ierr = PSAllocateWork_Private(ps,0,5*nm*nm+7*nm,0);CHKERRQ(ierr); 
    LAPACKgesdd_("A",&n,&m,A,&ld,d,U,&ld,VT,&ld,&qwork,&lwork,ps->rwork,ps->iwork,&info);
#else
    LAPACKgesdd_("A",&n,&m,A,&ld,d,U,&ld,VT,&ld,&qwork,&lwork,ps->iwork,&info);
#endif 
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
    lwork = (PetscBLASInt)PetscRealPart(qwork);
    ierr = PSAllocateWork_Private(ps,lwork,0,0);CHKERRQ(ierr); 

    /* computation */  
#if defined(PETSC_USE_COMPLEX)
    LAPACKgesdd_("A",&n,&m,A,&ld,d,U,&ld,VT,&ld,ps->work,&lwork,ps->rwork,ps->iwork,&info);
#else
    LAPACKgesdd_("A",&n,&m,A,&ld,d,U,&ld,VT,&ld,ps->work,&lwork,ps->iwork,&info);
#endif
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);

  }

  ierr = PSSolve_SVD_Sort(ps,wr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PSCreate_SVD"
PetscErrorCode PSCreate_SVD(PS ps)
{
  PetscFunctionBegin;
  ps->ops->allocate      = PSAllocate_SVD;
  ps->ops->view          = PSView_SVD;
  ps->ops->vectors       = PSVectors_SVD;
  ps->ops->solve[0]      = PSSolve_SVD_DC;
  PetscFunctionReturn(0);
}
EXTERN_C_END

