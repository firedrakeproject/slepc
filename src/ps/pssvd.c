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

/*   0       l           k                 n-1
    -----------------------------------------
    |*       ·           ·                  |
    |  *     ·           ·                  |
    |    *   ·           ·                  |
    |      * ·           ·                  |
    |        o           o                  |
    |          o         o                  |
    |            o       o                  |
    |              o     o                  |
    |                o   o                  |
    |                  o o                  |
    |                    o x                |
    |                      x x              |
    |                        x x            |
    |                          x x          |
    |                            x x        |
    |                              x x      |
    |                                x x    |
    |                                  x x  |
    |                                    x x|
    |                                      x|
    -----------------------------------------
*/

#undef __FUNCT__  
#define __FUNCT__ "PSSwitchFormat_SVD"
static PetscErrorCode PSSwitchFormat_SVD(PS ps,PetscBool tocompact)
{
  PetscErrorCode ierr;
  PetscReal      *T = ps->rmat[PS_MAT_T];
  PetscScalar    *A = ps->mat[PS_MAT_A];
  PetscInt       i,m=ps->m,k=ps->k,ld=ps->ld;

  PetscFunctionBegin;
  if (m==0) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_WRONG,"m was not set");
  if (tocompact) { /* switch from dense (arrow) to compact storage */
    ierr = PetscMemzero(T,3*ld*sizeof(PetscReal));CHKERRQ(ierr);
    for (i=0;i<k;i++) {
      T[i] = PetscRealPart(A[i+i*ld]);
      T[i+ld] = PetscRealPart(A[i+k*ld]);
    }
    for (i=k;i<m-1;i++) {
      T[i] = PetscRealPart(A[i+i*ld]);
      T[i+ld] = PetscRealPart(A[i+(i+1)*ld]);
    }
    T[m-1] = PetscRealPart(A[m-1+(m-1)*ld]);
  } else { /* switch from compact (arrow) to dense storage */
    ierr = PetscMemzero(A,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<k;i++) {
      A[i+i*ld] = T[i];
      A[i+k*ld] = T[i+ld];
    }
    A[k+k*ld] = T[k];
    for (i=k+1;i<m;i++) {
      A[i+i*ld] = T[i];
      A[i-1+i*ld] = T[i-1+ld];
    } 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSView_SVD"
PetscErrorCode PSView_SVD(PS ps,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;
  PetscInt          i,j,r,c;
  PetscReal         value;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscFunctionReturn(0);
  }
  if (ps->compact) {
    if (ps->m==0) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_WRONG,"m was not set");
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",ps->n,ps->m);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",2*ps->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
      for (i=0;i<PetscMin(ps->n,ps->m);i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,*(ps->rmat[PS_MAT_T]+i));CHKERRQ(ierr);
      }
      for (i=0;i<PetscMin(ps->n,ps->m)-1;i++) {
        r = PetscMax(i+2,ps->k+1);
        c = i+1;
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,*(ps->rmat[PS_MAT_T]+ps->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",PSMatName[PS_MAT_T]);CHKERRQ(ierr);
    } else {
      for (i=0;i<ps->n;i++) {
        for (j=0;j<ps->m;j++) {
          if (i==j) value = *(ps->rmat[PS_MAT_T]+i);
          else if (i<ps->k && j==ps->k) value = *(ps->rmat[PS_MAT_T]+ps->ld+PetscMin(i,j));
          else if (i==j+1 && i>ps->k) value = *(ps->rmat[PS_MAT_T]+ps->ld+i-1);
          else value = 0.0;
          ierr = PetscViewerASCIIPrintf(viewer," %18.16e ",value);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_A);CHKERRQ(ierr); 
  }
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
  ierr = PSPermuteBoth_Private(ps,l,n,PS_MAT_U,PS_MAT_VT,perm);CHKERRQ(ierr);
  for (i=l;i<n;i++) d[i] = PetscRealPart(wr[i]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_SVD_Update"
/*
  Helper function that is called at the end of any PSSolve_SVD_* method. 
*/
static PetscErrorCode PSSolve_SVD_Update(PS ps)
{
  PetscErrorCode ierr;
  PetscInt       i,l=ps->l;
  PetscBLASInt   n=ps->n,ld=ps->ld;
  PetscScalar    *A;
  PetscReal      *d,*e;

  PetscFunctionBegin;
  A  = ps->mat[PS_MAT_A];
  d  = ps->rmat[PS_MAT_T];
  e  = ps->rmat[PS_MAT_T]+ld;
  if (ps->compact) {
    ierr = PetscMemzero(e,(n-1)*sizeof(PetscReal));CHKERRQ(ierr);
  } else {
    for (i=l;i<n;i++) {
      ierr = PetscMemzero(A+l+i*ld,(n-l)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    for (i=l;i<n;i++) A[i+i*ld] = d[i];
  }
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
  PetscBLASInt   n1,n2,n3,m2,m3,info,l,n,m,nm,ld,off,lwork;
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
  m2 = PetscBLASIntCast(m-ps->k-1);
  m3 = n1+m2;
  off = l+l*ld;
  A  = ps->mat[PS_MAT_A];
  U  = ps->mat[PS_MAT_U];
  VT = ps->mat[PS_MAT_VT];
  d  = ps->rmat[PS_MAT_T];
  e  = ps->rmat[PS_MAT_T]+ld;
  ierr = PetscMemzero(U,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<l;i++) U[i+i*ld] = 1.0;
  ierr = PetscMemzero(VT,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<l;i++) VT[i+i*ld] = 1.0;

  if (ps->state>PS_STATE_RAW) {
    /* Solve bidiagonal SVD problem */
    for (i=0;i<l;i++) wr[i] = d[i];
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
    LAPACKbdsdc_("U","I",&n3,d+l,e+l,Ur+off,&ld,VTr+off,&ld,PETSC_NULL,PETSC_NULL,ps->rwork,ps->iwork,&info);
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
    if (ps->compact) { ierr = PSSwitchFormat_SVD(ps,PETSC_FALSE);CHKERRQ(ierr); }
    for (i=0;i<l;i++) wr[i] = d[i];
    nm = PetscMin(n,m);
    ierr = PSAllocateWork_Private(ps,0,0,8*nm);CHKERRQ(ierr); 
    lwork = -1;
#if defined(PETSC_USE_COMPLEX)
    ierr = PSAllocateWork_Private(ps,0,5*nm*nm+7*nm,0);CHKERRQ(ierr); 
    LAPACKgesdd_("A",&n3,&m3,A+off,&ld,d+l,U+off,&ld,VT+off,&ld,&qwork,&lwork,ps->rwork,ps->iwork,&info);
#else
    LAPACKgesdd_("A",&n3,&m3,A+off,&ld,d+l,U+off,&ld,VT+off,&ld,&qwork,&lwork,ps->iwork,&info);
#endif 
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
    lwork = (PetscBLASInt)PetscRealPart(qwork);
    ierr = PSAllocateWork_Private(ps,lwork,0,0);CHKERRQ(ierr); 
#if defined(PETSC_USE_COMPLEX)
    LAPACKgesdd_("A",&n3,&m3,A+off,&ld,d+l,U+off,&ld,VT+off,&ld,ps->work,&lwork,ps->rwork,ps->iwork,&info);
#else
    LAPACKgesdd_("A",&n3,&m3,A+off,&ld,d+l,U+off,&ld,VT+off,&ld,ps->work,&lwork,ps->iwork,&info);
#endif
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
  }

  ierr = PSSolve_SVD_Sort(ps,wr);CHKERRQ(ierr);
  ierr = PSSolve_SVD_Update(ps);CHKERRQ(ierr);
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

