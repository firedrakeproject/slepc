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

#include <slepc/private/dsimpl.h>       /*I "slepcds.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  PetscInt d;              /* polynomial degree */
} DS_PEP;

#undef __FUNCT__
#define __FUNCT__ "DSAllocate_PEP"
PetscErrorCode DSAllocate_PEP(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;
  DS_PEP         *ctx = (DS_PEP*)ds->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (!ctx->d) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"DSPEP requires specifying the polynomial degree via DSPEPSetDegree()");
  ierr = DSAllocateMat_Private(ds,DS_MAT_X);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_Y);CHKERRQ(ierr);
  for (i=0;i<=ctx->d;i++) {
    ierr = DSAllocateMat_Private(ds,DSMatExtra[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld*ctx->d,&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*ctx->d*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSView_PEP"
PetscErrorCode DSView_PEP(DS ds,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  DS_PEP            *ctx = (DS_PEP*)ds->data;
  PetscViewerFormat format;
  PetscInt          i;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  polynomial degree: %D\n",ctx->d);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  for (i=0;i<=ctx->d;i++) {
    ierr = DSViewMat(ds,viewer,DSMatExtra[i]);CHKERRQ(ierr);
  }
  if (ds->state>DS_STATE_INTERMEDIATE) {
    ds->m = ctx->d*ds->n;  /* temporarily set number of columns */
    ierr = DSViewMat(ds,viewer,DS_MAT_X);CHKERRQ(ierr);
    ds->m = 0;
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
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSort_PEP"
PetscErrorCode DSSort_PEP(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *kout)
{
  PetscErrorCode ierr;
  DS_PEP         *ctx = (DS_PEP*)ds->data;
  PetscInt       n,i,j,k,p,*perm,told,ld;
  PetscScalar    *A,*X,*Y,rtmp,rtmp2;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  n = ds->n*ctx->d;
  A  = ds->mat[DS_MAT_A];
  perm = ds->perm;
  for (i=0;i<n;i++) perm[i] = i;
  told = ds->t;
  ds->t = n;  /* force the sorting routines to consider d*n eigenvalues */
  if (rr) {
    ierr = DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = DSSortEigenvalues_Private(ds,wr,wi,perm,PETSC_FALSE);CHKERRQ(ierr);
  }
  ds->t = told;  /* restore value of t */
  for (i=0;i<n;i++) A[i]  = wr[perm[i]];
  for (i=0;i<n;i++) wr[i] = A[i];
  for (i=0;i<n;i++) A[i]  = wi[perm[i]];
  for (i=0;i<n;i++) wi[i] = A[i];
  /* cannot use DSPermuteColumns_Private() since matrix is not square */
  ld = ds->ld;
  X  = ds->mat[DS_MAT_X];
  Y  = ds->mat[DS_MAT_Y];
  for (i=0;i<n;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap columns i and j */
      for (k=0;k<ds->n;k++) {
        rtmp  = X[k+p*ld]; X[k+p*ld] = X[k+i*ld]; X[k+i*ld] = rtmp;
        rtmp2 = Y[k+p*ld]; Y[k+p*ld] = Y[k+i*ld]; Y[k+i*ld] = rtmp2;
      }
    }
  }
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
  DS_PEP         *ctx = (DS_PEP*)ds->data;
  PetscInt       i,j,off;
  PetscScalar    *A,*B,*W,*X,*U,*Y,*E,*work,*beta,norm;
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
  if (!ds->mat[DS_MAT_U]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_U);CHKERRQ(ierr);
  }
  ierr = PetscBLASIntCast(ds->n*ctx->d,&nd);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld*ctx->d,&ldd);CHKERRQ(ierr);
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
  U = ds->mat[DS_MAT_U];
  X = ds->mat[DS_MAT_X];
  Y = ds->mat[DS_MAT_Y];
  E = ds->mat[DSMatExtra[ctx->d]];

  /* build matrices A and B of the linearization */
  ierr = PetscMemzero(A,ldd*ldd*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<nd-ds->n;i++) A[i+(i+ds->n)*ldd] = -1.0;
  for (i=0;i<ctx->d;i++) {
    off = i*ds->n*ldd+(ctx->d-1)*ds->n;
    for (j=0;j<ds->n;j++) {
      ierr = PetscMemcpy(A+off+j*ldd,ds->mat[DSMatExtra[i]]+j*ds->ld,ds->n*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  ierr = PetscMemzero(B,ldd*ldd*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<nd-ds->n;i++) B[i+i*ldd] = -1.0;
  off = (ctx->d-1)*ds->n*(ldd+1);
  for (j=0;j<ds->n;j++) {
    for (i=0;i<ds->n;i++) B[off+i+j*ldd] = -E[i+j*ds->ld];
  }

  /* solve generalized eigenproblem */
#if defined(PETSC_USE_COMPLEX)
  rwork = ds->rwork;
  PetscStackCallBLAS("LAPACKggev",LAPACKggev_("V","V",&nd,A,&ldd,B,&ldd,wr,beta,U,&ldd,W,&ldd,work,&lwork,rwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack ZGGEV %d",info);
#else
  PetscStackCallBLAS("LAPACKggev",LAPACKggev_("V","V",&nd,A,&ldd,B,&ldd,wr,wi,beta,U,&ldd,W,&ldd,work,&lwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DGGEV %d",info);
#endif

  /* copy eigenvalues */
  for (i=0;i<nd;i++) {
    if (beta[i]==0.0) wr[i] = (PetscRealPart(wr[i])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else wr[i] /= beta[i];
#if !defined(PETSC_USE_COMPLEX)
    if (beta[i]==0.0) wi[i] = 0.0;
    else wi[i] /= beta[i];
#else
    if (wi) wi[i] = 0.0;
#endif
  }

  /* copy and normalize eigenvectors */
  for (j=0;j<nd;j++) {
    ierr = PetscMemcpy(X+j*ds->ld,W+j*ldd,ds->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemcpy(Y+j*ds->ld,U+ds->n*(ctx->d-1)+j*ldd,ds->n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (j=0;j<nd;j++) {
#if !defined(PETSC_USE_COMPLEX)
    if (wi[j] != 0.0) {
      norm = BLASnrm2_(&n,X+j*ds->ld,&one);
      norm0 = BLASnrm2_(&n,X+(j+1)*ds->ld,&one);
      norm = 1.0/SlepcAbsEigenvalue(norm,norm0);
      PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,X+j*ds->ld,&one));
      PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,X+(j+1)*ds->ld,&one));
      norm = BLASnrm2_(&n,Y+j*ds->ld,&one);
      norm0 = BLASnrm2_(&n,Y+(j+1)*ds->ld,&one);
      norm = 1.0/SlepcAbsEigenvalue(norm,norm0);
      PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,Y+j*ds->ld,&one));
      PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,Y+(j+1)*ds->ld,&one));
      j++;
    } else
#endif
    {
      norm = 1.0/BLASnrm2_(&n,X+j*ds->ld,&one);
      PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,X+j*ds->ld,&one));
      norm = 1.0/BLASnrm2_(&n,Y+j*ds->ld,&one);
      PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,Y+j*ds->ld,&one));
    }
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DSPEPSetDegree_PEP"
static PetscErrorCode DSPEPSetDegree_PEP(DS ds,PetscInt d)
{
  DS_PEP *ctx = (DS_PEP*)ds->data;

  PetscFunctionBegin;
  if (d<0) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"The degree must be a non-negative integer");
  if (d>=DS_NUM_EXTRA) SETERRQ1(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Only implemented for polynomials of degree at most %D",DS_NUM_EXTRA-1);
  ctx->d = d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSPEPSetDegree"
/*@
   DSPEPSetDegree - Sets the polynomial degree for a DSPEP.

   Logically Collective on DS

   Input Parameters:
+  ds - the direct solver context
-  d  - the degree

   Level: intermediate

.seealso: DSPEPGetDegree()
@*/
PetscErrorCode DSPEPSetDegree(DS ds,PetscInt d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,d,2);
  ierr = PetscTryMethod(ds,"DSPEPSetDegree_C",(DS,PetscInt),(ds,d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSPEPGetDegree_PEP"
static PetscErrorCode DSPEPGetDegree_PEP(DS ds,PetscInt *d)
{
  DS_PEP *ctx = (DS_PEP*)ds->data;

  PetscFunctionBegin;
  *d = ctx->d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSPEPGetDegree"
/*@
   DSPEPGetDegree - Returns the polynomial degree for a DSPEP.

   Not collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameters:
.  d - the degree

   Level: intermediate

.seealso: DSPEPSetDegree()
@*/
PetscErrorCode DSPEPGetDegree(DS ds,PetscInt *d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(d,2);
  ierr = PetscUseMethod(ds,"DSPEPGetDegree_C",(DS,PetscInt*),(ds,d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSDestroy_PEP"
PetscErrorCode DSDestroy_PEP(DS ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ds->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSPEPSetDegree_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSPEPGetDegree_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSCreate_PEP"
PETSC_EXTERN PetscErrorCode DSCreate_PEP(DS ds)
{
  DS_PEP         *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ds,&ctx);CHKERRQ(ierr);
  ds->data = (void*)ctx;

  ds->ops->allocate      = DSAllocate_PEP;
  ds->ops->view          = DSView_PEP;
  ds->ops->vectors       = DSVectors_PEP;
  ds->ops->solve[0]      = DSSolve_PEP_QZ;
  ds->ops->sort          = DSSort_PEP;
  ds->ops->normalize     = DSNormalize_PEP;
  ds->ops->destroy       = DSDestroy_PEP;
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSPEPSetDegree_C",DSPEPSetDegree_PEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSPEPGetDegree_C",DSPEPGetDegree_PEP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

