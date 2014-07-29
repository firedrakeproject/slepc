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
#define __FUNCT__ "DSAllocate_NEP"
PetscErrorCode DSAllocate_NEP(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ds->nf) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"DSNEP requires passing some functions via DSSetFN()");
  ierr = DSAllocateMat_Private(ds,DS_MAT_X);CHKERRQ(ierr);
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSView_NEP"
PetscErrorCode DSView_NEP(DS ds,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;
  PetscInt          i;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  for (i=0;i<ds->nf;i++) {
    ierr = FNView(ds->f[i],viewer);CHKERRQ(ierr);
    ierr = DSViewMat(ds,viewer,DSMatExtra[i]);CHKERRQ(ierr);
  }
  if (ds->state>DS_STATE_INTERMEDIATE) {
    ierr = DSViewMat(ds,viewer,DS_MAT_X);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSVectors_NEP"
PetscErrorCode DSVectors_NEP(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
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
#define __FUNCT__ "DSNormalize_NEP"
PetscErrorCode DSNormalize_NEP(DS ds,DSMatType mat,PetscInt col)
{
  PetscErrorCode ierr;
  PetscInt       i,i0,i1;
  PetscBLASInt   ld,n,one = 1;
  PetscScalar    norm,*x;

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
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = DSGetArray(ds,mat,&x);CHKERRQ(ierr);
  if (col < 0) {
    i0 = 0; i1 = ds->n;
  } else {
    i0 = col; i1 = col+1;
  }
  for (i=i0;i<i1;i++) {
    norm = BLASnrm2_(&n,&x[ld*i],&one);
    norm = 1.0/norm;
    PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,&x[ld*i],&one));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSort_NEP"
PetscErrorCode DSSort_NEP(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscErrorCode ierr;
  PetscInt       n,l,i,*perm,ld=ds->ld;
  PetscScalar    *A;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  n = ds->n;
  l = ds->l;
  A  = ds->mat[DS_MAT_A];
  perm = ds->perm;
  for (i=l;i<n;i++) wr[i] = A[i+i*ld];
  if (rr) {
    ierr = DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = DSSortEigenvalues_Private(ds,wr,NULL,perm,PETSC_FALSE);CHKERRQ(ierr);
  }
  for (i=l;i<n;i++) A[i+i*ld] = wr[perm[i]];
  for (i=l;i<n;i++) wr[i] = A[i+i*ld];
  ierr = DSPermuteColumns_Private(ds,l,n,DS_MAT_Q,perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSolve_NEP_SLP"
PetscErrorCode DSSolve_NEP_SLP(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GGEV)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GGEV - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscScalar    *A,*B,*W,*X,*work,*alpha,*beta;
  PetscScalar    norm,sigma,lambda,mu,re,re2;
  PetscBLASInt   info,n,ld,lrwork=0,lwork,one=1;
  PetscInt       it,pos,j,maxit=100,result;
  PetscReal      tol;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#else
  PetscReal      *alphai,im,im2;
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
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscBLASIntCast(2*ds->n+2*ds->n,&lwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(8*ds->n,&lrwork);CHKERRQ(ierr);
#else
  ierr = PetscBLASIntCast(3*ds->n+8*ds->n,&lwork);CHKERRQ(ierr);
#endif
  ierr = DSAllocateWork_Private(ds,lwork,lrwork,0);CHKERRQ(ierr);
  alpha = ds->work;
  beta = ds->work + ds->n;
#if defined(PETSC_USE_COMPLEX)
  work = ds->work + 2*ds->n;
  lwork -= 2*ds->n;
#else
  alphai = ds->work + 2*ds->n;
  work = ds->work + 3*ds->n;
  lwork -= 3*ds->n;
#endif
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  W = ds->mat[DS_MAT_W];
  X = ds->mat[DS_MAT_X];

  sigma = 0.0;
  lambda = sigma;
  tol = 1000*n*PETSC_MACHINE_EPSILON;

  for (it=0;it<maxit;it++) {

    /* evaluate T and T' */
    ierr = DSComputeMatrix(ds,lambda,PETSC_FALSE,DS_MAT_A);CHKERRQ(ierr);
    ierr = DSComputeMatrix(ds,lambda,PETSC_TRUE,DS_MAT_B);CHKERRQ(ierr);

    /* % compute eigenvalue correction mu and eigenvector u */
#if defined(PETSC_USE_COMPLEX)
    rwork = ds->rwork;
    PetscStackCallBLAS("LAPACKggev",LAPACKggev_("N","V",&n,A,&ld,B,&ld,alpha,beta,NULL,&ld,W,&ld,work,&lwork,rwork,&info));
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack ZGGEV %d",info);
#else
    PetscStackCallBLAS("LAPACKggev",LAPACKggev_("N","V",&n,A,&ld,B,&ld,alpha,alphai,beta,NULL,&ld,W,&ld,work,&lwork,&info));
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DGGEV %d",info);
#endif

    /* find smallest eigenvalue */
    j = 0;
    if (beta[j]==0.0) re = (PetscRealPart(alpha[j])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else re = alpha[j]/beta[j];
#if !defined(PETSC_USE_COMPLEX)
    if (beta[j]==0.0) im = (alphai[j]>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else im = alphai[j]/beta[j];
#endif
    pos = 0;
    for (j=1;j<n;j++) {
      if (beta[j]==0.0) re2 = (PetscRealPart(alpha[j])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
      else re2 = alpha[j]/beta[j];
#if !defined(PETSC_USE_COMPLEX)
      if (beta[j]==0.0) im2 = (alphai[j]>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
      else im2 = alphai[j]/beta[j];
      ierr = SlepcCompareSmallestMagnitude(re,im,re2,im2,&result,NULL);CHKERRQ(ierr);
#else
      ierr = SlepcCompareSmallestMagnitude(re,0.0,re2,0.0,&result,NULL);CHKERRQ(ierr);
#endif
      if (result > 0) {
        re = re2;
#if !defined(PETSC_USE_COMPLEX)
        im = im2;
#endif
        pos = j;
      }
    }

#if !defined(PETSC_USE_COMPLEX)
    if (im!=0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"DSNEP found a complex eigenvalue; try rerunning with complex scalars");
#endif
    mu = alpha[pos];
    ierr = PetscMemcpy(X,W+pos*ld,n*sizeof(PetscScalar));CHKERRQ(ierr);
    norm = BLASnrm2_(&n,X,&one);
    norm = 1.0/norm;
    PetscStackCallBLAS("BLASscal",BLASscal_(&n,&norm,X,&one));

    /* correct eigenvalue approximation */
    lambda = lambda - mu;
    if (PetscAbsScalar(mu)<=tol) break;
  }

  wr[0] = lambda;
  if (wi) wi[0] = 0.0;

  if (it==maxit) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED,"DSNEP did not converge");
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DSCreate_NEP"
PETSC_EXTERN PetscErrorCode DSCreate_NEP(DS ds)
{
  PetscFunctionBegin;
  ds->ops->allocate      = DSAllocate_NEP;
  ds->ops->view          = DSView_NEP;
  ds->ops->vectors       = DSVectors_NEP;
  ds->ops->solve[0]      = DSSolve_NEP_SLP;
  ds->ops->sort          = DSSort_NEP;
  ds->ops->normalize     = DSNormalize_NEP;
  PetscFunctionReturn(0);
}

