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

PetscErrorCode DSAllocate_NHEP(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_Q);CHKERRQ(ierr);
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSView_NHEP(DS ds,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  ierr = DSViewMat(ds,viewer,DS_MAT_A);CHKERRQ(ierr);
  if (ds->state>DS_STATE_INTERMEDIATE) { ierr = DSViewMat(ds,viewer,DS_MAT_Q);CHKERRQ(ierr); }
  if (ds->mat[DS_MAT_X]) { ierr = DSViewMat(ds,viewer,DS_MAT_X);CHKERRQ(ierr); }
  if (ds->mat[DS_MAT_Y]) { ierr = DSViewMat(ds,viewer,DS_MAT_Y);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

static PetscErrorCode DSVectors_NHEP_Refined_Some(DS ds,PetscInt *k,PetscReal *rnorm,PetscBool left)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBLASInt   info,ld,n,n1,lwork,inc=1;
  PetscScalar    sdummy,done=1.0,zero=0.0;
  PetscReal      *sigma;
  PetscBool      iscomplex = PETSC_FALSE;
  PetscScalar    *A = ds->mat[DS_MAT_A];
  PetscScalar    *Q = ds->mat[DS_MAT_Q];
  PetscScalar    *X = ds->mat[left?DS_MAT_Y:DS_MAT_X];
  PetscScalar    *W;

  PetscFunctionBegin;
  if (left) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented for left vectors");
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  n1 = n+1;
  if ((*k)<n-1 && A[(*k)+1+(*k)*ld]!=0.0) iscomplex = PETSC_TRUE;
  if (iscomplex) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for complex eigenvalues yet");
  ierr = DSAllocateWork_Private(ds,5*ld,6*ld,0);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
  W = ds->mat[DS_MAT_W];
  lwork = 5*ld;
  sigma = ds->rwork+5*ld;

  /* build A-w*I in W */
  for (j=0;j<n;j++)
    for (i=0;i<=n;i++)
      W[i+j*ld] = A[i+j*ld];
  for (i=0;i<n;i++)
    W[i+i*ld] -= A[(*k)+(*k)*ld];

  /* compute SVD of W */
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","O",&n1,&n,W,&ld,sigma,&sdummy,&ld,&sdummy,&ld,ds->work,&lwork,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","O",&n1,&n,W,&ld,sigma,&sdummy,&ld,&sdummy,&ld,ds->work,&lwork,ds->rwork,&info));
#endif
  SlepcCheckLapackInfo("gesvd",info);

  /* the smallest singular value is the new error estimate */
  if (rnorm) *rnorm = sigma[n-1];

  /* update vector with right singular vector associated to smallest singular value,
     accumulating the transformation matrix Q */
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&done,Q,&ld,W+n-1,&ld,&zero,X+(*k)*ld,&inc));
  PetscFunctionReturn(0);
}

static PetscErrorCode DSVectors_NHEP_Refined_All(DS ds,PetscBool left)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0;i<ds->n;i++) {
    ierr = DSVectors_NHEP_Refined_Some(ds,&i,NULL,left);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DSVectors_NHEP_Eigen_Some(DS ds,PetscInt *k,PetscReal *rnorm,PetscBool left)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   mm=1,mout,info,ld,n,*select,inc=1,cols=1,zero=0;
  PetscScalar    sone=1.0,szero=0.0;
  PetscReal      norm,done=1.0;
  PetscBool      iscomplex = PETSC_FALSE;
  PetscScalar    *A = ds->mat[DS_MAT_A];
  PetscScalar    *Q = ds->mat[DS_MAT_Q];
  PetscScalar    *X = ds->mat[left?DS_MAT_Y:DS_MAT_X];
  PetscScalar    *Y;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = DSAllocateWork_Private(ds,0,0,ld);CHKERRQ(ierr);
  select = ds->iwork;
  for (i=0;i<n;i++) select[i] = (PetscBLASInt)PETSC_FALSE;

  /* compute k-th eigenvector Y of A */
  Y = X+(*k)*ld;
  select[*k] = (PetscBLASInt)PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  if ((*k)<n-1 && A[(*k)+1+(*k)*ld]!=0.0) iscomplex = PETSC_TRUE;
  mm = iscomplex? 2: 1;
  if (iscomplex) select[(*k)+1] = (PetscBLASInt)PETSC_TRUE;
  ierr = DSAllocateWork_Private(ds,3*ld,0,0);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_(left?"L":"R","S",select,&n,A,&ld,Y,&ld,Y,&ld,&mm,&mout,ds->work,&info));
#else
  ierr = DSAllocateWork_Private(ds,2*ld,ld,0);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_(left?"L":"R","S",select,&n,A,&ld,Y,&ld,Y,&ld,&mm,&mout,ds->work,ds->rwork,&info));
#endif
  SlepcCheckLapackInfo("trevc",info);
  if (mout != mm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Inconsistent arguments");

  /* accumulate and normalize eigenvectors */
  if (ds->state>=DS_STATE_CONDENSED) {
    ierr = PetscArraycpy(ds->work,Y,mout*ld);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&sone,Q,&ld,ds->work,&inc,&szero,Y,&inc));
#if !defined(PETSC_USE_COMPLEX)
    if (iscomplex) PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&sone,Q,&ld,ds->work+ld,&inc,&szero,Y+ld,&inc));
#endif
    cols = 1;
    norm = BLASnrm2_(&n,Y,&inc);
#if !defined(PETSC_USE_COMPLEX)
    if (iscomplex) {
      norm = SlepcAbsEigenvalue(norm,BLASnrm2_(&n,Y+ld,&inc));
      cols = 2;
    }
#endif
    PetscStackCallBLAS("LAPACKlascl",LAPACKlascl_("G",&zero,&zero,&norm,&done,&n,&cols,Y,&ld,&info));
    SlepcCheckLapackInfo("lascl",info);
  }

  /* set output arguments */
  if (iscomplex) (*k)++;
  if (rnorm) {
    if (iscomplex) *rnorm = SlepcAbsEigenvalue(Y[n-1],Y[n-1+ld]);
    else *rnorm = PetscAbsScalar(Y[n-1]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DSVectors_NHEP_Eigen_All(DS ds,PetscBool left)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n,ld,mout,info,inc=1,cols,zero=0;
  PetscBool      iscomplex;
  PetscScalar    *X,*Y,*Z,*A = ds->mat[DS_MAT_A];
  PetscReal      norm,done=1.0;
  const char     *side,*back;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  if (left) {
    X = NULL;
    Y = ds->mat[DS_MAT_Y];
    side = "L";
  } else {
    X = ds->mat[DS_MAT_X];
    Y = NULL;
    side = "R";
  }
  Z = left? Y: X;
  if (ds->state>=DS_STATE_CONDENSED) {
    /* DSSolve() has been called, backtransform with matrix Q */
    back = "B";
    ierr = PetscArraycpy(Z,ds->mat[DS_MAT_Q],ld*ld);CHKERRQ(ierr);
  } else back = "A";
#if !defined(PETSC_USE_COMPLEX)
  ierr = DSAllocateWork_Private(ds,3*ld,0,0);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_(side,back,NULL,&n,A,&ld,Y,&ld,X,&ld,&n,&mout,ds->work,&info));
#else
  ierr = DSAllocateWork_Private(ds,2*ld,ld,0);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_(side,back,NULL,&n,A,&ld,Y,&ld,X,&ld,&n,&mout,ds->work,ds->rwork,&info));
#endif
  SlepcCheckLapackInfo("trevc",info);

  /* normalize eigenvectors */
  for (i=0;i<n;i++) {
    iscomplex = (i<n-1 && A[i+1+i*ld]!=0.0)? PETSC_TRUE: PETSC_FALSE;
    cols = 1;
    norm = BLASnrm2_(&n,Z+i*ld,&inc);
#if !defined(PETSC_USE_COMPLEX)
    if (iscomplex) {
      norm = SlepcAbsEigenvalue(norm,BLASnrm2_(&n,Z+(i+1)*ld,&inc));
      cols = 2;
    }
#endif
    PetscStackCallBLAS("LAPACKlascl",LAPACKlascl_("G",&zero,&zero,&norm,&done,&n,&cols,Z+i*ld,&ld,&info));
    SlepcCheckLapackInfo("lascl",info);
    if (iscomplex) i++;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSVectors_NHEP(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_X:
      if (ds->refined) {
        if (!ds->extrarow) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Refined vectors require activating the extra row");
        if (j) {
          ierr = DSVectors_NHEP_Refined_Some(ds,j,rnorm,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = DSVectors_NHEP_Refined_All(ds,PETSC_FALSE);CHKERRQ(ierr);
        }
      } else {
        if (j) {
          ierr = DSVectors_NHEP_Eigen_Some(ds,j,rnorm,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = DSVectors_NHEP_Eigen_All(ds,PETSC_FALSE);CHKERRQ(ierr);
        }
      }
      break;
    case DS_MAT_Y:
      if (ds->refined) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented yet");
      if (j) {
        ierr = DSVectors_NHEP_Eigen_Some(ds,j,rnorm,PETSC_TRUE);CHKERRQ(ierr);
      } else {
        ierr = DSVectors_NHEP_Eigen_All(ds,PETSC_TRUE);CHKERRQ(ierr);
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

static PetscErrorCode DSSort_NHEP_Arbitrary(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   info,n,ld,mout,lwork,*selection;
  PetscScalar    *T = ds->mat[DS_MAT_A],*Q = ds->mat[DS_MAT_Q],*work;
  PetscReal      dummy;
#if !defined(PETSC_USE_COMPLEX)
  PetscBLASInt   *iwork,liwork;
#endif

  PetscFunctionBegin;
  if (!k) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"Must supply argument k");
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  lwork = n;
  liwork = 1;
  ierr = DSAllocateWork_Private(ds,lwork,0,liwork+n);CHKERRQ(ierr);
  work = ds->work;
  lwork = ds->lwork;
  selection = ds->iwork;
  iwork = ds->iwork + n;
  liwork = ds->liwork - n;
#else
  lwork = 1;
  ierr = DSAllocateWork_Private(ds,lwork,0,n);CHKERRQ(ierr);
  work = ds->work;
  selection = ds->iwork;
#endif
  /* Compute the selected eigenvalue to be in the leading position */
  ierr = DSSortEigenvalues_Private(ds,rr,ri,ds->perm,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscArrayzero(selection,n);CHKERRQ(ierr);
  for (i=0;i<*k;i++) selection[ds->perm[i]] = 1;
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKtrsen",LAPACKtrsen_("N","V",selection,&n,T,&ld,Q,&ld,wr,wi,&mout,&dummy,&dummy,work,&lwork,iwork,&liwork,&info));
#else
  PetscStackCallBLAS("LAPACKtrsen",LAPACKtrsen_("N","V",selection,&n,T,&ld,Q,&ld,wr,&mout,&dummy,&dummy,work,&lwork,&info));
#endif
  SlepcCheckLapackInfo("trsen",info);
  *k = mout;
  PetscFunctionReturn(0);
}

PetscErrorCode DSSort_NHEP(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!rr || wr == rr) {
    ierr = DSSort_NHEP_Total(ds,ds->mat[DS_MAT_A],ds->mat[DS_MAT_Q],wr,wi);CHKERRQ(ierr);
  } else {
    ierr = DSSort_NHEP_Arbitrary(ds,wr,wi,rr,ri,k);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DSSortWithPermutation_NHEP(DS ds,PetscInt *perm,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSSortWithPermutation_NHEP_Private(ds,perm,ds->mat[DS_MAT_A],ds->mat[DS_MAT_Q],wr,wi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSUpdateExtraRow_NHEP(DS ds)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n,ld,incx=1;
  PetscScalar    *A,*Q,*x,*y,one=1.0,zero=0.0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  A  = ds->mat[DS_MAT_A];
  Q  = ds->mat[DS_MAT_Q];
  ierr = DSAllocateWork_Private(ds,2*ld,0,0);CHKERRQ(ierr);
  x = ds->work;
  y = ds->work+ld;
  for (i=0;i<n;i++) x[i] = PetscConj(A[n+i*ld]);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&n,&one,Q,&ld,x,&incx,&zero,y,&incx));
  for (i=0;i<n;i++) A[n+i*ld] = PetscConj(y[i]);
  ds->k = n;
  PetscFunctionReturn(0);
}

PetscErrorCode DSSolve_NHEP(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  PetscValidScalarPointer(wi,3);
#endif
  ierr = DSSolve_NHEP_Private(ds,ds->mat[DS_MAT_A],ds->mat[DS_MAT_Q],wr,wi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSSynchronize_NHEP(DS ds,PetscScalar eigr[],PetscScalar eigi[])
{
  PetscErrorCode ierr;
  PetscInt       ld=ds->ld,l=ds->l,k;
  PetscMPIInt    n,rank,off=0,size,ldn;

  PetscFunctionBegin;
  k = (ds->n-l)*ld;
  if (ds->state>DS_STATE_RAW) k += (ds->n-l)*ld;
  if (eigr) k += ds->n-l;
  if (eigi) k += ds->n-l;
  ierr = DSAllocateWork_Private(ds,k,0,0);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(k*sizeof(PetscScalar),&size);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(ds->n-l,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(ld*(ds->n-l),&ldn);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ds),&rank);CHKERRMPI(ierr);
  if (!rank) {
    ierr = MPI_Pack(ds->mat[DS_MAT_A]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    if (ds->state>DS_STATE_RAW) {
      ierr = MPI_Pack(ds->mat[DS_MAT_Q]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigr) {
      ierr = MPI_Pack(eigr+l,n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
#if !defined(PETSC_USE_COMPLEX)
    if (eigi) {
      ierr = MPI_Pack(eigi+l,n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
#endif
  }
  ierr = MPI_Bcast(ds->work,size,MPI_BYTE,0,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
  if (rank) {
    ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_A]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    if (ds->state>DS_STATE_RAW) {
      ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_Q]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigr) {
      ierr = MPI_Unpack(ds->work,size,&off,eigr+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
#if !defined(PETSC_USE_COMPLEX)
    if (eigi) {
      ierr = MPI_Unpack(ds->work,size,&off,eigi+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
#endif
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSTruncate_NHEP(DS ds,PetscInt n,PetscBool trim)
{
  PetscInt    i,ld=ds->ld,l=ds->l;
  PetscScalar *A = ds->mat[DS_MAT_A];

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  /* make sure diagonal 2x2 block is not broken */
  if (ds->state>=DS_STATE_CONDENSED && n>0 && n<ds->n && A[n+(n-1)*ld]!=0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"The given size would break a 2x2 block, call DSGetTruncateSize() first");
#endif
  if (trim) {
    if (ds->extrarow) {   /* clean extra row */
      for (i=l;i<ds->n;i++) A[ds->n+i*ld] = 0.0;
    }
    ds->l = 0;
    ds->k = 0;
    ds->n = n;
    ds->t = ds->n;   /* truncated length equal to the new dimension */
  } else {
    if (ds->extrarow && ds->k==ds->n) {
      /* copy entries of extra row to the new position, then clean last row */
      for (i=l;i<n;i++) A[n+i*ld] = A[ds->n+i*ld];
      for (i=l;i<ds->n;i++) A[ds->n+i*ld] = 0.0;
    }
    ds->k = (ds->extrarow)? n: 0;
    ds->t = ds->n;   /* truncated length equal to previous dimension */
    ds->n = n;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSCond_NHEP(DS ds,PetscReal *cond)
{
  PetscErrorCode ierr;
  PetscScalar    *work;
  PetscReal      *rwork;
  PetscBLASInt   *ipiv;
  PetscBLASInt   lwork,info,n,ld;
  PetscReal      hn,hin;
  PetscScalar    *A;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  lwork = 8*ld;
  ierr = DSAllocateWork_Private(ds,lwork,ld,ld);CHKERRQ(ierr);
  work  = ds->work;
  rwork = ds->rwork;
  ipiv  = ds->iwork;

  /* use workspace matrix W to avoid overwriting A */
  ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
  A = ds->mat[DS_MAT_W];
  ierr = PetscArraycpy(A,ds->mat[DS_MAT_A],ds->ld*ds->ld);CHKERRQ(ierr);

  /* norm of A */
  if (ds->state<DS_STATE_INTERMEDIATE) hn = LAPACKlange_("I",&n,&n,A,&ld,rwork);
  else hn = LAPACKlanhs_("I",&n,A,&ld,rwork);

  /* norm of inv(A) */
  PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n,&n,A,&ld,ipiv,&info));
  SlepcCheckLapackInfo("getrf",info);
  PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,A,&ld,ipiv,work,&lwork,&info));
  SlepcCheckLapackInfo("getri",info);
  hin = LAPACKlange_("I",&n,&n,A,&ld,rwork);

  *cond = hn*hin;
  PetscFunctionReturn(0);
}

PetscErrorCode DSTranslateHarmonic_NHEP(DS ds,PetscScalar tau,PetscReal beta,PetscBool recover,PetscScalar *gin,PetscReal *gammaout)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBLASInt   *ipiv,info,n,ld,one=1,ncol;
  PetscScalar    *A,*B,*Q,*g=gin,*ghat;
  PetscScalar    done=1.0,dmone=-1.0,dzero=0.0;
  PetscReal      gamma=1.0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  A  = ds->mat[DS_MAT_A];

  if (!recover) {

    ierr = DSAllocateWork_Private(ds,0,0,ld);CHKERRQ(ierr);
    ipiv = ds->iwork;
    if (!g) {
      ierr = DSAllocateWork_Private(ds,ld,0,0);CHKERRQ(ierr);
      g = ds->work;
    }
    /* use workspace matrix W to factor A-tau*eye(n) */
    ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
    B = ds->mat[DS_MAT_W];
    ierr = PetscArraycpy(B,A,ld*ld);CHKERRQ(ierr);

    /* Vector g initially stores b = beta*e_n^T */
    ierr = PetscArrayzero(g,n);CHKERRQ(ierr);
    g[n-1] = beta;

    /* g = (A-tau*eye(n))'\b */
    for (i=0;i<n;i++) B[i+i*ld] -= tau;
    PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n,&n,B,&ld,ipiv,&info));
    SlepcCheckLapackInfo("getrf",info);
    ierr = PetscLogFlops(2.0*n*n*n/3.0);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("C",&n,&one,B,&ld,ipiv,g,&ld,&info));
    SlepcCheckLapackInfo("getrs",info);
    ierr = PetscLogFlops(2.0*n*n-n);CHKERRQ(ierr);

    /* A = A + g*b' */
    for (i=0;i<n;i++) A[i+(n-1)*ld] += g[i]*beta;

  } else { /* recover */

    ierr = DSAllocateWork_Private(ds,ld,0,0);CHKERRQ(ierr);
    ghat = ds->work;
    Q    = ds->mat[DS_MAT_Q];

    /* g^ = -Q(:,idx)'*g */
    ierr = PetscBLASIntCast(ds->l+ds->k,&ncol);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&ncol,&dmone,Q,&ld,g,&one,&dzero,ghat,&one));

    /* A = A + g^*b' */
    for (i=0;i<ds->l+ds->k;i++)
      for (j=ds->l;j<ds->l+ds->k;j++)
        A[i+j*ld] += ghat[i]*Q[n-1+j*ld]*beta;

    /* g~ = (I-Q(:,idx)*Q(:,idx)')*g = g+Q(:,idx)*g^ */
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&ncol,&done,Q,&ld,ghat,&one,&done,g,&one));
  }

  /* Compute gamma factor */
  if (gammaout || (recover && ds->extrarow)) gamma = SlepcAbs(1.0,BLASnrm2_(&n,g,&one));
  if (gammaout) *gammaout = gamma;
  if (recover && ds->extrarow) {
    for (j=ds->l;j<ds->l+ds->k;j++) A[ds->n+j*ld] *= gamma;
  }
  PetscFunctionReturn(0);
}

/*MC
   DSNHEP - Dense Non-Hermitian Eigenvalue Problem.

   Level: beginner

   Notes:
   The problem is expressed as A*X = X*Lambda, where A is the input matrix.
   Lambda is a diagonal matrix whose diagonal elements are the arguments of
   DSSolve(). After solve, A is overwritten with the upper quasi-triangular
   matrix T of the (real) Schur form, A*Q = Q*T.

   In the intermediate state A is reduced to upper Hessenberg form.

   Computation of left eigenvectors is supported, but two-sided Krylov solvers
   usually rely on the related DSNHEPTS.

   Used DS matrices:
+  DS_MAT_A - problem matrix
-  DS_MAT_Q - orthogonal/unitary transformation that reduces to Hessenberg form
   (intermediate step) or matrix of orthogonal Schur vectors

   Implemented methods:
.  0 - Implicit QR (_hseqr)

.seealso: DSCreate(), DSSetType(), DSType
M*/
SLEPC_EXTERN PetscErrorCode DSCreate_NHEP(DS ds)
{
  PetscFunctionBegin;
  ds->ops->allocate        = DSAllocate_NHEP;
  ds->ops->view            = DSView_NHEP;
  ds->ops->vectors         = DSVectors_NHEP;
  ds->ops->solve[0]        = DSSolve_NHEP;
  ds->ops->sort            = DSSort_NHEP;
  ds->ops->sortperm        = DSSortWithPermutation_NHEP;
  ds->ops->synchronize     = DSSynchronize_NHEP;
  ds->ops->gettruncatesize = DSGetTruncateSize_Default;
  ds->ops->truncate        = DSTruncate_NHEP;
  ds->ops->update          = DSUpdateExtraRow_NHEP;
  ds->ops->cond            = DSCond_NHEP;
  ds->ops->transharm       = DSTranslateHarmonic_NHEP;
  PetscFunctionReturn(0);
}

