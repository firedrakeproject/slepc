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
#define __FUNCT__ "PSAllocate_GNHEP"
PetscErrorCode PSAllocate_GNHEP(PS ps,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_B);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Z);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Q);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSView_GNHEP"
PetscErrorCode PSView_GNHEP(PS ps,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSViewMat_Private(ps,viewer,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSViewMat_Private(ps,viewer,PS_MAT_B);CHKERRQ(ierr); 
  if (ps->state>PS_STATE_INTERMEDIATE) {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_Z);CHKERRQ(ierr); 
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_Q);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_GNHEP_Eigen_All"
PetscErrorCode PSVectors_GNHEP_Eigen_All(PS ps,PetscBool left)
{
#if defined(SLEPC_MISSING_LAPACK_TGEVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TGEVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscBLASInt   n,ld,mout,info;
  PetscScalar    *X,*Y,*A = ps->mat[PS_MAT_A],*B = ps->mat[PS_MAT_B];
  const char     *side,*back;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  if (left) {
    X = PETSC_NULL;
    Y = ps->mat[PS_MAT_Y];
    side = "L";
  } else {
    X = ps->mat[PS_MAT_X];
    Y = PETSC_NULL;
    side = "R";
  }
  if (ps->state>=PS_STATE_CONDENSED) {
    /* PSSolve() has been called, backtransform with matrix Q */
    back = "B";
    ierr = PetscMemcpy(left?Y:X,ps->mat[left?PS_MAT_Z:PS_MAT_Q],ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  } else back = "A";
#if defined(PETSC_USE_COMPLEX)
  ierr = PSAllocateWork_Private(ps,2*ld,2*ld,0);CHKERRQ(ierr); 
  LAPACKtgevc_(side,back,PETSC_NULL,&n,A,&ld,B,&ld,Y,&ld,X,&ld,&n,&mout,ps->work,ps->rwork,&info);
#else
  ierr = PSAllocateWork_Private(ps,6*ld,0,0);CHKERRQ(ierr); 
  LAPACKtgevc_(side,back,PETSC_NULL,&n,A,&ld,B,&ld,Y,&ld,X,&ld,&n,&mout,ps->work,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_GNHEP"
PetscErrorCode PSVectors_GNHEP(PS ps,PSMatType mat,PetscInt *k,PetscReal *rnorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (k) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
  switch (mat) {
    case PS_MAT_X:
      ierr = PSVectors_GNHEP_Eigen_All(ps,PETSC_FALSE);CHKERRQ(ierr);
      break;
    case PS_MAT_Y:
      ierr = PSVectors_GNHEP_Eigen_All(ps,PETSC_TRUE);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_GNHEP_Sort"
/*
  Sort the condensed form at the end of any PSSolve_GNHEP_* method. 
*/
static PetscErrorCode PSSolve_GNHEP_Sort(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_TGEXC) || !defined(PETSC_USE_COMPLEX) && (defined(SLEPC_MISSING_LAPACK_LAMCH) || defined(SLEPC_MISSING_LAPACK_LAG2))
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TGEXC/LAMCH/LAG2 - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,pos,result;
  PetscBLASInt   ifst,ilst,info,n,ld,one=1;
  PetscScalar    *S = ps->mat[PS_MAT_A],*T = ps->mat[PS_MAT_B],*Z = ps->mat[PS_MAT_Z],*Q = ps->mat[PS_MAT_Q];
#if !defined(PETSC_USE_COMPLEX)
  PetscBLASInt   lwork;
  PetscScalar    *work,a,safmin,scale1,scale2;
#endif

  PetscFunctionBegin;
  if (!ps->comp_fun) PetscFunctionReturn(0);
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
#if !defined(PETSC_USE_COMPLEX)
  lwork = -1;
  LAPACKtgexc_(&one,&one,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,&one,&one,&a,&lwork,&info);
  safmin = LAPACKlamch_("S");
  lwork = a;
  ierr = PSAllocateWork_Private(ps,lwork,0,0);CHKERRQ(ierr); 
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
      ierr = (*ps->comp_fun)(re,im,wr[j],wi[j],&result,ps->comp_ctx);CHKERRQ(ierr);
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
      LAPACKtgexc_(&one,&one,&n,S,&ld,T,&ld,Z,&n,Q,&n,&ifst,&ilst,work,&lwork,&info);
#else
      LAPACKtgexc_(&one,&one,&n,S,&ld,T,&ld,Z,&ld,Q,&ld,&ifst,&ilst,&info);
#endif
      if (info) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_LIB,"Error in Lapack xTGEXC %i",info);
      /* recover original eigenvalues from T and S matrices */
      for (j=i;j<n;j++) {
#if !defined(PETSC_USE_COMPLEX)
        if (j<n-1 && S[j*ld+j+1] != 0.0) {
          /* complex conjugate eigenvalue */
          LAPACKlag2_(S+j*ld+j,&ld,T+j*ld+j,&ld,&safmin,&scale1,&scale2,&re,&a,&im);
          wr[j] = re / scale1;
          wi[j] = im / scale1;
          wr[j+1] = a / scale2;
          wi[j+1] = -wi[j];
          j++;
        } else
#endif
        {
          if (T[j*ld+j] == 0.0) wr[j] = (PetscRealPart(S[j*ld+j])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
          else wr[j] = S[j*ld+j] / T[j*ld+j];
          wi[j] = 0.0;
        }
      }
    }
#if !defined(PETSC_USE_COMPLEX)
    if (wi[i] != 0.0) i++;
#endif
  }
  PetscFunctionReturn(0);
#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_GNHEP"
PetscErrorCode PSSolve_GNHEP(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GGES)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GGES - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    *work,*beta,a;
  PetscInt       i;
  PetscBLASInt   lwork,info,n,ld,iaux;
  PetscScalar    *A = ps->mat[PS_MAT_A],*B = ps->mat[PS_MAT_B],*Z = ps->mat[PS_MAT_Z],*Q = ps->mat[PS_MAT_Q];

  PetscFunctionBegin;
  PetscValidPointer(wi,3);
  n   = PetscBLASIntCast(ps->n);
  ld  = PetscBLASIntCast(ps->ld);
  if (ps->state==PS_STATE_INTERMEDIATE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for the intermediate state");
  lwork = -1;
#if !defined(PETSC_USE_COMPLEX)
  LAPACKgges_("V","V","N",PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,&ld,PETSC_NULL,&ld,&a,&lwork,PETSC_NULL,&info);
  lwork = (PetscBLASInt)a;
  ierr = PSAllocateWork_Private(ps,lwork+ld,0,0);CHKERRQ(ierr); 
  beta = ps->work;
  work = beta+ps->n;
  lwork = PetscBLASIntCast(ps->lwork-ps->n);
  LAPACKgges_("V","V","N",PETSC_NULL,&n,A,&ld,B,&ld,&iaux,wr,wi,beta,Z,&ld,Q,&ld,work,&lwork,PETSC_NULL,&info);
#else
  LAPACKgges_("V","V","N",PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,&ld,PETSC_NULL,&ld,&a,&lwork,PETSC_NULL,PETSC_NULL,&info);
  lwork = (PetscBLASInt)PetscRealPart(a);
  ierr = PSAllocateWork_Private(ps,lwork+ld,8*ld,0);CHKERRQ(ierr); 
  beta = ps->work;
  work = beta+ps->n;
  lwork = PetscBLASIntCast(ps->lwork-ps->n);
  LAPACKgges_("V","V","N",PETSC_NULL,&n,A,&ld,B,&ld,&iaux,wr,beta,Z,&ld,Q,&ld,work,&lwork,ps->rwork,PETSC_NULL,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_LIB,"Error in Lapack xGGES %i",info);
  for (i=0;i<n;i++) {
    if (beta[i]==0.0) wr[i] = (PetscRealPart(wr[i])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else wr[i] /= beta[i];
#if !defined(PETSC_USE_COMPLEX)
    if (beta[i]==0.0) wi[i] = (wi[i]>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else wi[i] /= beta[i];
#endif
  }
  ierr = PSSolve_GNHEP_Sort(ps,wr,wi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PSCreate_GNHEP"
PetscErrorCode PSCreate_GNHEP(PS ps)
{
  PetscFunctionBegin;
  ps->ops->allocate      = PSAllocate_GNHEP;
  ps->ops->view          = PSView_GNHEP;
  ps->ops->vectors       = PSVectors_GNHEP;
  ps->ops->solve[0]      = PSSolve_GNHEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

