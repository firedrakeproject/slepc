/*
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

#include <slepc/private/dsimpl.h>
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "DSAllocate_SVD"
PetscErrorCode DSAllocate_SVD(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_U);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_VT);CHKERRQ(ierr);
  ierr = DSAllocateMatReal_Private(ds,DS_MAT_T);CHKERRQ(ierr);
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*   0       l           k                 n-1
    -----------------------------------------
    |*       .           .                  |
    |  *     .           .                  |
    |    *   .           .                  |
    |      * .           .                  |
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
#define __FUNCT__ "DSSwitchFormat_SVD"
static PetscErrorCode DSSwitchFormat_SVD(DS ds,PetscBool tocompact)
{
  PetscErrorCode ierr;
  PetscReal      *T = ds->rmat[DS_MAT_T];
  PetscScalar    *A = ds->mat[DS_MAT_A];
  PetscInt       i,m=ds->m,k=ds->k,ld=ds->ld;

  PetscFunctionBegin;
  if (!m) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"m was not set");
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
#define __FUNCT__ "DSView_SVD"
PetscErrorCode DSView_SVD(DS ds,PetscViewer viewer)
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
  if (ds->compact) {
    if (!ds->m) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"m was not set");
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",ds->n,ds->m);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",2*ds->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
      for (i=0;i<PetscMin(ds->n,ds->m);i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,*(ds->rmat[DS_MAT_T]+i));CHKERRQ(ierr);
      }
      for (i=0;i<PetscMin(ds->n,ds->m)-1;i++) {
        r = PetscMax(i+2,ds->k+1);
        c = i+1;
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,*(ds->rmat[DS_MAT_T]+ds->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",DSMatName[DS_MAT_T]);CHKERRQ(ierr);
    } else {
      for (i=0;i<ds->n;i++) {
        for (j=0;j<ds->m;j++) {
          if (i==j) value = *(ds->rmat[DS_MAT_T]+i);
          else if (i<ds->k && j==ds->k) value = *(ds->rmat[DS_MAT_T]+ds->ld+PetscMin(i,j));
          else if (i==j+1 && i>ds->k) value = *(ds->rmat[DS_MAT_T]+ds->ld+i-1);
          else value = 0.0;
          ierr = PetscViewerASCIIPrintf(viewer," %18.16e ",value);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    ierr = DSViewMat(ds,viewer,DS_MAT_A);CHKERRQ(ierr);
  }
  if (ds->state>DS_STATE_INTERMEDIATE) {
    ierr = DSViewMat(ds,viewer,DS_MAT_U);CHKERRQ(ierr);
    ierr = DSViewMat(ds,viewer,DS_MAT_VT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSVectors_SVD"
PetscErrorCode DSVectors_SVD(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_U:
    case DS_MAT_VT:
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSort_SVD"
PetscErrorCode DSSort_SVD(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscErrorCode ierr;
  PetscInt       n,l,i,*perm,ld=ds->ld;
  PetscScalar    *A;
  PetscReal      *d;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  l = ds->l;
  n = PetscMin(ds->n,ds->m);
  A = ds->mat[DS_MAT_A];
  d = ds->rmat[DS_MAT_T];
  perm = ds->perm;
  if (!rr) {
    ierr = DSSortEigenvaluesReal_Private(ds,d,perm);CHKERRQ(ierr);
  } else {
    ierr = DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_FALSE);CHKERRQ(ierr);
  }
  for (i=l;i<n;i++) wr[i] = d[perm[i]];
  ierr = DSPermuteBoth_Private(ds,l,n,DS_MAT_U,DS_MAT_VT,perm);CHKERRQ(ierr);
  for (i=l;i<n;i++) d[i] = PetscRealPart(wr[i]);
  if (!ds->compact) {
    for (i=l;i<n;i++) A[i+i*ld] = wr[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSolve_SVD_DC"
PetscErrorCode DSSolve_SVD_DC(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GESDD) || defined(SLEPC_MISSING_LAPACK_BDSDC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESDD/BDSDC - Lapack routines are unavailable");
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
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->m,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->l,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->k-l+1,&n1);CHKERRQ(ierr); /* size of leading block, excl. locked */
  ierr = PetscBLASIntCast(n-ds->k-1,&n2);CHKERRQ(ierr); /* size of trailing block */
  ierr = PetscBLASIntCast(m-ds->k-1,&m2);CHKERRQ(ierr);
  n3 = n1+n2;
  m3 = n1+m2;
  off = l+l*ld;
  A  = ds->mat[DS_MAT_A];
  U  = ds->mat[DS_MAT_U];
  VT = ds->mat[DS_MAT_VT];
  d  = ds->rmat[DS_MAT_T];
  e  = ds->rmat[DS_MAT_T]+ld;
  ierr = PetscMemzero(U,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<l;i++) U[i+i*ld] = 1.0;
  ierr = PetscMemzero(VT,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<l;i++) VT[i+i*ld] = 1.0;

  if (ds->state>DS_STATE_RAW) {
    /* Solve bidiagonal SVD problem */
    for (i=0;i<l;i++) wr[i] = d[i];
    ierr = DSAllocateWork_Private(ds,0,3*ld*ld+4*ld,8*ld);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = DSAllocateMatReal_Private(ds,DS_MAT_U);CHKERRQ(ierr);
    ierr = DSAllocateMatReal_Private(ds,DS_MAT_VT);CHKERRQ(ierr);
    Ur  = ds->rmat[DS_MAT_U];
    VTr = ds->rmat[DS_MAT_VT];
#else
    Ur  = U;
    VTr = VT;
#endif
    PetscStackCallBLAS("LAPACKbdsdc",LAPACKbdsdc_("U","I",&n3,d+l,e+l,Ur+off,&ld,VTr+off,&ld,NULL,NULL,ds->rwork,ds->iwork,&info));
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
    if (ds->compact) { ierr = DSSwitchFormat_SVD(ds,PETSC_FALSE);CHKERRQ(ierr); }
    for (i=0;i<l;i++) wr[i] = d[i];
    nm = PetscMin(n,m);
    ierr = DSAllocateWork_Private(ds,0,0,8*nm);CHKERRQ(ierr);
    lwork = -1;
#if defined(PETSC_USE_COMPLEX)
    ierr = DSAllocateWork_Private(ds,0,5*nm*nm+7*nm,0);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n3,&m3,A+off,&ld,d+l,U+off,&ld,VT+off,&ld,&qwork,&lwork,ds->rwork,ds->iwork,&info));
#else
    PetscStackCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n3,&m3,A+off,&ld,d+l,U+off,&ld,VT+off,&ld,&qwork,&lwork,ds->iwork,&info));
#endif
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
    ierr = PetscBLASIntCast((PetscInt)PetscRealPart(qwork),&lwork);CHKERRQ(ierr);
    ierr = DSAllocateWork_Private(ds,lwork,0,0);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n3,&m3,A+off,&ld,d+l,U+off,&ld,VT+off,&ld,ds->work,&lwork,ds->rwork,ds->iwork,&info));
#else
    PetscStackCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n3,&m3,A+off,&ld,d+l,U+off,&ld,VT+off,&ld,ds->work,&lwork,ds->iwork,&info));
#endif
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
  }
  for (i=l;i<PetscMin(ds->n,ds->m);i++) wr[i] = d[i];

  /* Create diagonal matrix as a result */
  if (ds->compact) {
    ierr = PetscMemzero(e,(n-1)*sizeof(PetscReal));CHKERRQ(ierr);
  } else {
    for (i=l;i<n;i++) {
      ierr = PetscMemzero(A+l+i*ld,(n-l)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    for (i=l;i<n;i++) A[i+i*ld] = d[i];
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DSCreate_SVD"
PETSC_EXTERN PetscErrorCode DSCreate_SVD(DS ds)
{
  PetscFunctionBegin;
  ds->ops->allocate      = DSAllocate_SVD;
  ds->ops->view          = DSView_SVD;
  ds->ops->vectors       = DSVectors_SVD;
  ds->ops->solve[0]      = DSSolve_SVD_DC;
  ds->ops->sort          = DSSort_SVD;
  PetscFunctionReturn(0);
}

