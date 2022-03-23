/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/dsimpl.h>
#include <slepcblaslapack.h>

PetscErrorCode DSAllocate_GHEP(DS ds,PetscInt ld)
{
  PetscFunctionBegin;
  CHKERRQ(DSAllocateMat_Private(ds,DS_MAT_A));
  CHKERRQ(DSAllocateMat_Private(ds,DS_MAT_B));
  CHKERRQ(DSAllocateMat_Private(ds,DS_MAT_Q));
  CHKERRQ(PetscFree(ds->perm));
  CHKERRQ(PetscMalloc1(ld,&ds->perm));
  CHKERRQ(PetscLogObjectMemory((PetscObject)ds,ld*sizeof(PetscInt)));
  PetscFunctionReturn(0);
}

PetscErrorCode DSView_GHEP(DS ds,PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  CHKERRQ(DSViewMat(ds,viewer,DS_MAT_A));
  CHKERRQ(DSViewMat(ds,viewer,DS_MAT_B));
  if (ds->state>DS_STATE_INTERMEDIATE) CHKERRQ(DSViewMat(ds,viewer,DS_MAT_Q));
  if (ds->mat[DS_MAT_X]) CHKERRQ(DSViewMat(ds,viewer,DS_MAT_X));
  PetscFunctionReturn(0);
}

PetscErrorCode DSVectors_GHEP(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscScalar    *Q = ds->mat[DS_MAT_Q];
  PetscInt       ld = ds->ld;

  PetscFunctionBegin;
  PetscCheck(!rnorm,PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented yet");
  switch (mat) {
    case DS_MAT_X:
    case DS_MAT_Y:
      if (j) {
        if (ds->state>=DS_STATE_CONDENSED) CHKERRQ(PetscArraycpy(ds->mat[mat]+(*j)*ld,Q+(*j)*ld,ld));
        else {
          CHKERRQ(PetscArrayzero(ds->mat[mat]+(*j)*ld,ld));
          *(ds->mat[mat]+(*j)+(*j)*ld) = 1.0;
        }
      } else {
        if (ds->state>=DS_STATE_CONDENSED) CHKERRQ(PetscArraycpy(ds->mat[mat],Q,ld*ld));
        else CHKERRQ(DSSetIdentity(ds,mat));
      }
      break;
    case DS_MAT_U:
    case DS_MAT_V:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented yet");
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSort_GHEP(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscInt       n,l,i,*perm,ld=ds->ld;
  PetscScalar    *A;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  n = ds->n;
  l = ds->l;
  A  = ds->mat[DS_MAT_A];
  perm = ds->perm;
  for (i=l;i<n;i++) wr[i] = A[i+i*ld];
  if (rr) CHKERRQ(DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_FALSE));
  else CHKERRQ(DSSortEigenvalues_Private(ds,wr,NULL,perm,PETSC_FALSE));
  for (i=l;i<n;i++) A[i+i*ld] = wr[perm[i]];
  for (i=l;i<n;i++) wr[i] = A[i+i*ld];
  CHKERRQ(DSPermuteColumns_Private(ds,l,n,n,DS_MAT_Q,perm));
  PetscFunctionReturn(0);
}

PetscErrorCode DSSolve_GHEP(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  PetscScalar    *work,*A,*B,*Q;
  PetscBLASInt   itype = 1,*iwork,info,n1,liwork,ld,lrwork=0,lwork;
  PetscInt       off,i;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork,*rr;
#endif

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(ds->n-ds->l,&n1));
  CHKERRQ(PetscBLASIntCast(ds->ld,&ld));
  CHKERRQ(PetscBLASIntCast(5*ds->n+3,&liwork));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscBLASIntCast(ds->n*ds->n+2*ds->n,&lwork));
  CHKERRQ(PetscBLASIntCast(2*ds->n*ds->n+5*ds->n+1+n1,&lrwork));
#else
  CHKERRQ(PetscBLASIntCast(2*ds->n*ds->n+6*ds->n+1,&lwork));
#endif
  CHKERRQ(DSAllocateWork_Private(ds,lwork,lrwork,liwork));
  work = ds->work;
  iwork = ds->iwork;
  off = ds->l+ds->l*ld;
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  Q = ds->mat[DS_MAT_Q];
#if defined(PETSC_USE_COMPLEX)
  rr = ds->rwork;
  rwork = ds->rwork + n1;
  lrwork = ds->lrwork - n1;
  PetscStackCallBLAS("LAPACKsygvd",LAPACKsygvd_(&itype,"V","U",&n1,A+off,&ld,B+off,&ld,rr,work,&lwork,rwork,&lrwork,iwork,&liwork,&info));
  for (i=0;i<n1;i++) wr[ds->l+i] = rr[i];
#else
  PetscStackCallBLAS("LAPACKsygvd",LAPACKsygvd_(&itype,"V","U",&n1,A+off,&ld,B+off,&ld,wr+ds->l,work,&lwork,iwork,&liwork,&info));
#endif
  SlepcCheckLapackInfo("sygvd",info);
  CHKERRQ(PetscArrayzero(Q+ds->l*ld,n1*ld));
  for (i=ds->l;i<ds->n;i++) CHKERRQ(PetscArraycpy(Q+ds->l+i*ld,A+ds->l+i*ld,n1));
  CHKERRQ(PetscArrayzero(B+ds->l*ld,n1*ld));
  CHKERRQ(PetscArrayzero(A+ds->l*ld,n1*ld));
  for (i=ds->l;i<ds->n;i++) {
    if (wi) wi[i] = 0.0;
    B[i+i*ld] = 1.0;
    A[i+i*ld] = wr[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSynchronize_GHEP(DS ds,PetscScalar eigr[],PetscScalar eigi[])
{
  PetscInt       ld=ds->ld,l=ds->l,k;
  PetscMPIInt    n,rank,off=0,size,ldn;

  PetscFunctionBegin;
  k = 2*(ds->n-l)*ld;
  if (ds->state>DS_STATE_RAW) k += (ds->n-l)*ld;
  if (eigr) k += (ds->n-l);
  CHKERRQ(DSAllocateWork_Private(ds,k,0,0));
  CHKERRQ(PetscMPIIntCast(k*sizeof(PetscScalar),&size));
  CHKERRQ(PetscMPIIntCast(ds->n-l,&n));
  CHKERRQ(PetscMPIIntCast(ld*(ds->n-l),&ldn));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ds),&rank));
  if (!rank) {
    CHKERRMPI(MPI_Pack(ds->mat[DS_MAT_A]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
    CHKERRMPI(MPI_Pack(ds->mat[DS_MAT_B]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
    if (ds->state>DS_STATE_RAW) CHKERRMPI(MPI_Pack(ds->mat[DS_MAT_Q]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
    if (eigr) CHKERRMPI(MPI_Pack(eigr+l,n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
  }
  CHKERRMPI(MPI_Bcast(ds->work,size,MPI_BYTE,0,PetscObjectComm((PetscObject)ds)));
  if (rank) {
    CHKERRMPI(MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_A]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
    CHKERRMPI(MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_B]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
    if (ds->state>DS_STATE_RAW) CHKERRMPI(MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_Q]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
    if (eigr) CHKERRMPI(MPI_Unpack(ds->work,size,&off,eigr+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSHermitian_GHEP(DS ds,DSMatType m,PetscBool *flg)
{
  PetscFunctionBegin;
  if (m==DS_MAT_A || m==DS_MAT_B) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*MC
   DSGHEP - Dense Generalized Hermitian Eigenvalue Problem.

   Level: beginner

   Notes:
   The problem is expressed as A*X = B*X*Lambda, where both A and B are
   real symmetric (or complex Hermitian) and B is positive-definite. Lambda
   is a diagonal matrix whose diagonal elements are the arguments of DSSolve().
   After solve, A is overwritten with Lambda, and B is overwritten with I.

   No intermediate state is implemented, nor compact storage.

   Used DS matrices:
+  DS_MAT_A - first problem matrix
.  DS_MAT_B - second problem matrix
-  DS_MAT_Q - matrix of B-orthogonal eigenvectors, which is equal to X

   Implemented methods:
.  0 - Divide and Conquer (_sygvd)

.seealso: DSCreate(), DSSetType(), DSType
M*/
SLEPC_EXTERN PetscErrorCode DSCreate_GHEP(DS ds)
{
  PetscFunctionBegin;
  ds->ops->allocate      = DSAllocate_GHEP;
  ds->ops->view          = DSView_GHEP;
  ds->ops->vectors       = DSVectors_GHEP;
  ds->ops->solve[0]      = DSSolve_GHEP;
  ds->ops->sort          = DSSort_GHEP;
  ds->ops->synchronize   = DSSynchronize_GHEP;
  ds->ops->hermitian     = DSHermitian_GHEP;
  PetscFunctionReturn(0);
}
