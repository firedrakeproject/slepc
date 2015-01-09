/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/dsimpl.h>
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "DSAllocate_PEP"
PetscErrorCode DSAllocate_PEP(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!ds->d) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"DSPEP requires specifying the polynomial degree via DSSetDegree()");
  ierr = DSAllocateMat_Private(ds,DS_MAT_X);CHKERRQ(ierr);
  for (i=0;i<=ds->d;i++) {
    ierr = DSAllocateMat_Private(ds,DSMatExtra[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld*ds->d,&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*ds->d*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSView_PEP"
PetscErrorCode DSView_PEP(DS ds,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;
  PetscInt          i;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIPrintf(viewer,"  polynomial degree: %D\n",ds->d);CHKERRQ(ierr);
  for (i=0;i<ds->d;i++) {
    ierr = DSViewMat(ds,viewer,DSMatExtra[i]);CHKERRQ(ierr);
  }
  if (ds->state>DS_STATE_INTERMEDIATE) {
    ierr = DSViewMat(ds,viewer,DS_MAT_X);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSVectors_PEP"
PetscErrorCode DSVectors_PEP(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscFunctionBegin;
  if (rnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
  switch (mat) {
    case DS_MAT_X:
      break;
    case DS_MAT_Y:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSNormalize_PEP"
PetscErrorCode DSNormalize_PEP(DS ds,DSMatType mat,PetscInt col)
{
  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_X:
      break;
    case DS_MAT_Y:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSort_PEP"
PetscErrorCode DSSort_PEP(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscErrorCode ierr;
  PetscInt       n,i,*perm;
  PetscScalar    *A;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  n = ds->n*ds->d;
  A  = ds->mat[DS_MAT_A];
  perm = ds->perm;
  if (rr) {
    ierr = DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = DSSortEigenvalues_Private(ds,wr,wi,perm,PETSC_FALSE);CHKERRQ(ierr);
  }
  for (i=0;i<n;i++) A[i]  = wr[perm[i]];
  for (i=0;i<n;i++) wr[i] = A[i];
  for (i=0;i<n;i++) A[i]  = wi[perm[i]];
  for (i=0;i<n;i++) wi[i] = A[i];
  ierr = DSPermuteColumns_Private(ds,0,n,DS_MAT_X,perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSolve_PEP_QZ"
PetscErrorCode DSSolve_PEP_QZ(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GGEV)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GGEV - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,off;
  PetscScalar    *A,*B,*W,*X,*E,*work,*beta,norm;
  PetscBLASInt   info,n,ldd,nd,lrwork=0,lwork,one=1;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#else
  PetscScalar    norm0;
#endif

  PetscFunctionBegin;
  if (!ds->mat[DS_MAT_A]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr);
  }
  if (!ds->mat[DS_MAT_B]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_B);CHKERRQ(ierr);
  }
  if (!ds->mat[DS_MAT_W]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
  }
  ierr = PetscBLASIntCast(ds->n*ds->d,&nd);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld*ds->d,&ldd);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscBLASIntCast(nd+2*nd,&lwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(8*nd,&lrwork);CHKERRQ(ierr);
#else
  ierr = PetscBLASIntCast(nd+8*nd,&lwork);CHKERRQ(ierr);
#endif
  ierr = DSAllocateWork_Private(ds,lwork,lrwork,0);CHKERRQ(ierr);
  beta = ds->work;
  work = ds->work + nd;
  lwork -= nd;
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  W = ds->mat[DS_MAT_W];
  X = ds->mat[DS_MAT_X];
  E = ds->mat[DSMatExtra[ds->d]];

  /* build matrices A and B of the linearization */
  ierr = PetscMemzero(A,ldd*ldd*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<nd-ds->n;i++) A[i+(i+ds->n)*ldd] = -1.0;
  for (i=0;i<ds->d;i++) {
    off = i*ds->n*ldd+(ds->d-1)*ds->n;
    for (j=0;j<ds->n;j++) {
      ierr = PetscMemcpy(A+off+j*ldd,ds->mat[DSMatExtra[i]]+j*ds->ld,ds->n*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  ierr = PetscMemzero(B,ldd*ldd*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<nd-ds->n;i++) B[i+i*ldd] = -1.0;
  off = (ds->d-1)*ds->n*(ldd+1);
  for (j=0;j<ds->n;j++) {
    for (i=0;i<ds->n;i++) B[off+i+j*ldd] = -E[i+j*ds->ld];
  }

  /* solve generalized eigenproblem */
#if defined(PETSC_USE_COMPLEX)
  rwork = ds->rwork;
  PetscStackCallBLAS("LAPACKggev",LAPACKggev_("N","V",&nd,A,&ldd,B,&ldd,wr,beta,NULL,&ldd,W,&ldd,work,&lwork,rwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack ZGGEV %d",info);
#else
  PetscStackCallBLAS("LAPACKggev",LAPACKggev_("N","V",&nd,A,&ldd,B,&ldd,wr,wi,beta,NULL,&ldd,W,&ldd,work,&lwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DGGEV %d",info);
#endif

  /* copy eigenvalues */
  for (i=0;i<nd;i++) {
    if (beta[i]==0.0) wr[i] = (PetscRealPart(wr[i])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else wr[i] /= beta[i];
#if !defined(PETSC_USE_COMPLEX)
    if (beta[i]==0.0) wi[i] = (wi[i]>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else wi[i] /= beta[i];
#else
    if (wi) wi[i] = 0.0;
#endif
  }

  /* copy and normalize eigenvectors */
  for (j=0;j<nd;j++) {
    ierr = PetscMemcpy(X+j*ds->ld,W+j*ldd,ds->n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (j=0;j<nd;j++) {
#if !defined(PETSC_USE_COMPLEX)
    if (wi[j] != 0.0) {
      norm = BLASnrm2_(&n,X+j*ds->ld,&one);
      norm0 = BLASnrm2_(&n,X+(j+1)*ds->ld,&one);
      norm = 1.0/SlepcAbsEigenvalue(norm,norm0);
      PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,X+j*ds->ld,&one));
      PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,X+(j+1)*ds->ld,&one));
      j++;
    } else
#endif
    {
      norm = 1.0/BLASnrm2_(&n,X+i*ds->ld,&one);
      PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,X+i*ds->ld,&one));
    }
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DSCreate_PEP"
PETSC_EXTERN PetscErrorCode DSCreate_PEP(DS ds)
{
  PetscFunctionBegin;
  ds->ops->allocate      = DSAllocate_PEP;
  ds->ops->view          = DSView_PEP;
  ds->ops->vectors       = DSVectors_PEP;
  ds->ops->solve[0]      = DSSolve_PEP_QZ;
  ds->ops->sort          = DSSort_PEP;
  ds->ops->normalize     = DSNormalize_PEP;
  PetscFunctionReturn(0);
}

