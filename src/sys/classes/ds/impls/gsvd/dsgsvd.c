/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/dsimpl.h>
#include <slepcblaslapack.h>

PetscErrorCode DSAllocate_GSVD(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_B);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_Q);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_U);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_VT);CHKERRQ(ierr);
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSView_GSVD(DS ds,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  ierr = DSViewMat(ds,viewer,DS_MAT_A);CHKERRQ(ierr);
  ierr = DSViewMat(ds,viewer,DS_MAT_B);CHKERRQ(ierr);
  if (ds->state>DS_STATE_INTERMEDIATE) {
    ierr = DSViewMat(ds,viewer,DS_MAT_Q);CHKERRQ(ierr);
    ierr = DSViewMat(ds,viewer,DS_MAT_U);CHKERRQ(ierr);
    ierr = DSViewMat(ds,viewer,DS_MAT_VT);CHKERRQ(ierr);
  }
  if (ds->mat[DS_MAT_X]) { ierr = DSViewMat(ds,viewer,DS_MAT_X);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

PetscErrorCode DSVectors_GSVD(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscErrorCode ierr;
  PetscBLASInt   n = 0,m = 0,q,r,ld;
  PetscScalar    *A,*B,*X,sone=1.0,smone=-1.0;

  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_U:
    case DS_MAT_VT:
      if (rnorm) *rnorm = 0.0;
      break;
    case DS_MAT_X:
      /* X = Q*inv(R) */
      ierr = PetscBLASIntCast(ds->n,&m);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(ds->m,&n);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
      A = ds->mat[DS_MAT_A];
      B = ds->mat[DS_MAT_B];
      X = ds->mat[DS_MAT_X];
      q = PetscMin(m,n);
      ierr = PetscArraycpy(X,ds->mat[DS_MAT_Q],ld*ld);CHKERRQ(ierr);
      PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&n,&q,&sone,A,&ld,X,&ld));
      if (m<n) {
        r = n-m;
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&r,&m,&sone,X,&ld,A,&ld,&smone,X+m*ld,&ld));
        PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&n,&r,&sone,B+m*ld,&ld,X+m*ld,&ld));
      }
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSort_GSVD(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
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

PetscErrorCode DSSolve_GSVD(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n1,m1,info,l = 0,n = 0,m = 0,p,k,nm,ld,off,lwork;
  PetscScalar    *A,*B,*Q,*U,*VT,qwork;
  PetscReal      *alpha,*beta;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif
#if !defined(SLEPC_MISSING_LAPACK_GGSVD3)
  PetscScalar    a,dummy;
  PetscReal      rdummy;
  PetscBLASInt   idummy;
#endif

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->m,&n);CHKERRQ(ierr);
  p = m;
//  ierr = PetscBLASIntCast(ds->l,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
//  n1 = n-l;     /* n1 = size of leading block, excl. locked + size of trailing block */
//  m1 = m-l;
//  off = l+l*ld;
  A  = ds->mat[DS_MAT_A];
  B  = ds->mat[DS_MAT_B];
  Q  = ds->mat[DS_MAT_Q];
  U  = ds->mat[DS_MAT_U];
  VT = ds->mat[DS_MAT_VT];
  ierr = PetscArrayzero(U,ld*ld);CHKERRQ(ierr);
  for (i=0;i<l;i++) U[i+i*ld] = 1.0;
  ierr = PetscArrayzero(VT,ld*ld);CHKERRQ(ierr);
  for (i=0;i<l;i++) VT[i+i*ld] = 1.0;

#if !defined(SLEPC_MISSING_LAPACK_GGSVD3)

  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  /* workspace query and memory allocation */
  lwork = -1;
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKggsvd3",LAPACKggsvd3_("U","V","Q",&m,&n,&p,&k,&l,&dummy,&ld,&dummy,&ld,&rdummy,&rdummy,&dummy,&ld,&dummy,&ld,&dummy,&ld,&a,&lwork,&idummy,&info));
  ierr = PetscBLASIntCast((PetscInt)a,&lwork);CHKERRQ(ierr);
#else
  PetscStackCallBLAS("LAPACKggsvd3",LAPACKggsvd3_("U","V","Q",&m,&n,&p,&k,&l,&dummy,&ld,&dummy,&ld,&rdummy,&rdummy,&dummy,&ld,&dummy,&ld,&dummy,&ld,&a,&lwork,&rdummy,&idummy,&info));
  ierr = PetscBLASIntCast((PetscInt)PetscRealPart(a),&lwork);CHKERRQ(ierr);
#endif

#if !defined (PETSC_USE_COMPLEX)
  ierr = DSAllocateWork_Private(ds,lwork,2*ds->ld,ds->ld);CHKERRQ(ierr);
  alpha = ds->rwork;
  beta  = ds->rwork+ds->ld;
  PetscStackCallBLAS("LAPACKggsvd3",LAPACKggsvd3_("U","V","Q",&m,&n,&p,&k,&l,A,&ld,B,&ld,alpha,beta,U,&ld,VT,&ld,Q,&ld,ds->work,&lwork,ds->iwork,&info));
#else
  ierr = DSAllocateWork_Private(ds,lwork,4*ds->ld,ds->ld);CHKERRQ(ierr);
  alpha = ds->rwork+2*ds->ld;
  beta  = ds->rwork+3*ds->ld;
  PetscStackCallBLAS("LAPACKggsvd3",LAPACKggsvd3_("U","V","Q",&m,&n,&p,&k,&l,A,&ld,B,&ld,alpha,beta,U,&ld,VT,&ld,Q,&ld,ds->work,&lwork,ds->rwork,ds->iwork,&info));
#endif
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  SlepcCheckLapackInfo("ggsvd3",info);

#else  // defined(SLEPC_MISSING_LAPACK_GGSVD3)

  lwork = PetscMax(PetscMax(3*n,m),p)+n;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined (PETSC_USE_COMPLEX)
  ierr = DSAllocateWork_Private(ds,lwork,2*ds->ld,ds->ld);CHKERRQ(ierr);
  alpha = ds->rwork;
  beta  = ds->rwork+ds->ld;
  PetscStackCallBLAS("LAPACKggsvd",LAPACKggsvd_("U","V","Q",&m,&n,&p,&k,&l,A,&ld,B,&ld,alpha,beta,U,&ld,VT,&ld,Q,&ld,ds->work,ds->iwork,&info));
#else
  ierr = DSAllocateWork_Private(ds,lwork,4*ds->ld,ds->ld);CHKERRQ(ierr);
  alpha = ds->rwork+2*ds->ld;
  beta  = ds->rwork+3*ds->ld;
  PetscStackCallBLAS("LAPACKggsvd",LAPACKggsvd_("U","V","Q",&m,&n,&p,&k,&l,A,&ld,B,&ld,alpha,beta,U,&ld,VT,&ld,Q,&ld,ds->work,ds->rwork,ds->iwork,&info));
#endif
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  SlepcCheckLapackInfo("ggsvd",info);

#endif

  if (k+l<n) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"The case k+l<n not supported yet");
  for (i=0;i<PetscMin(ds->n,k+l);i++) wr[i] = alpha[i]/beta[i];
  PetscFunctionReturn(0);
}

PetscErrorCode DSSynchronize_GSVD(DS ds,PetscScalar eigr[],PetscScalar eigi[])
{
  PetscErrorCode ierr;
  PetscInt       ld=ds->ld,l=ds->l,k=0,kr=0;
  PetscMPIInt    n,rank,off=0,size,ldn,ld3;

  PetscFunctionBegin;
  if (ds->compact) kr = 3*ld;
  else k = (ds->n-l)*ld;
  if (ds->state>DS_STATE_RAW) k += 2*(ds->n-l)*ld;
  if (eigr) k += ds->n-l;
  ierr = DSAllocateWork_Private(ds,k+kr,0,0);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(k*sizeof(PetscScalar)+kr*sizeof(PetscReal),&size);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(ds->n-l,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(ld*(ds->n-l),&ldn);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(3*ld,&ld3);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ds),&rank);CHKERRMPI(ierr);
  if (!rank) {
    if (ds->compact) {
      ierr = MPI_Pack(ds->rmat[DS_MAT_T],ld3,MPIU_REAL,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    } else {
      ierr = MPI_Pack(ds->mat[DS_MAT_A]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (ds->state>DS_STATE_RAW) {
      ierr = MPI_Pack(ds->mat[DS_MAT_U]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
      ierr = MPI_Pack(ds->mat[DS_MAT_VT]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigr) {
      ierr = MPI_Pack(eigr+l,n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
  }
  ierr = MPI_Bcast(ds->work,size,MPI_BYTE,0,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
  if (rank) {
    if (ds->compact) {
      ierr = MPI_Unpack(ds->work,size,&off,ds->rmat[DS_MAT_T],ld3,MPIU_REAL,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    } else {
      ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_A]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (ds->state>DS_STATE_RAW) {
      ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_U]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
      ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_VT]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigr) {
      ierr = MPI_Unpack(ds->work,size,&off,eigr+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSMatGetSize_GSVD(DS ds,DSMatType t,PetscInt *rows,PetscInt *cols)
{
  PetscFunctionBegin;
  switch (t) {
    case DS_MAT_A:
    case DS_MAT_B:
      *rows = ds->n;
      *cols = ds->m;
      break;
    case DS_MAT_U:
      *rows = ds->n;
      *cols = ds->n;
      break;
    case DS_MAT_VT:
    case DS_MAT_Q:
      *rows = ds->m;
      *cols = ds->m;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid t parameter");
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode DSCreate_GSVD(DS ds)
{
  PetscFunctionBegin;
  ds->ops->allocate      = DSAllocate_GSVD;
  ds->ops->view          = DSView_GSVD;
  ds->ops->vectors       = DSVectors_GSVD;
  ds->ops->solve[0]      = DSSolve_GSVD;
  ds->ops->sort          = DSSort_GSVD;
  ds->ops->synchronize   = DSSynchronize_GSVD;
  ds->ops->matgetsize    = DSMatGetSize_GSVD;
  PetscFunctionReturn(0);
}

