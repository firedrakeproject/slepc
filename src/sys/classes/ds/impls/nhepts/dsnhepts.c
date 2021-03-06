/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   DSNHEPTS: a special variant of NHEP to be used in two-sided Krylov solvers

   DS_MAT_A - upper Hessenberg matrix obtained from Arnoldi
   DS_MAT_B - upper Hessenberg matrix obtained from Arnoldi on the transpose
   DS_MAT_Q - orthogonal matrix of (right) Schur vectors
   DS_MAT_Z - orthogonal matrix of left Schur vectors (computed as Schur vectors of B)
   DS_MAT_X - right eigenvectors
   DS_MAT_Y - left eigenvectors (computed as right eigenvectors of B)
*/

#include <slepc/private/dsimpl.h>
#include <slepcblaslapack.h>

typedef struct {
  PetscScalar *wr,*wi;     /* eigenvalues of B */
} DS_NHEPTS;

PetscErrorCode DSAllocate_NHEPTS(DS ds,PetscInt ld)
{
  DS_NHEPTS      *ctx = (DS_NHEPTS*)ds->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_B);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_Q);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_Z);CHKERRQ(ierr);
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&ctx->wr);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*sizeof(PetscScalar));CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(ld,&ctx->wi);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*sizeof(PetscScalar));CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode DSView_NHEPTS(DS ds,PetscViewer viewer)
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
    ierr = DSViewMat(ds,viewer,DS_MAT_Z);CHKERRQ(ierr);
  }
  if (ds->mat[DS_MAT_X]) { ierr = DSViewMat(ds,viewer,DS_MAT_X);CHKERRQ(ierr); }
  if (ds->mat[DS_MAT_Y]) { ierr = DSViewMat(ds,viewer,DS_MAT_Y);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

static PetscErrorCode DSVectors_NHEPTS_Eigen_Some(DS ds,PetscInt *k,PetscReal *rnorm,PetscBool left)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   mm=1,mout,info,ld,n,*select,inc=1,cols=1,zero=0;
  PetscScalar    sone=1.0,szero=0.0;
  PetscReal      norm,done=1.0;
  PetscBool      iscomplex = PETSC_FALSE;
  PetscScalar    *A,*Q,*X,*Y;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  if (left) {
    A = ds->mat[DS_MAT_B];
    Q = ds->mat[DS_MAT_Z];
    X = ds->mat[DS_MAT_Y];
  } else {
    A = ds->mat[DS_MAT_A];
    Q = ds->mat[DS_MAT_Q];
    X = ds->mat[DS_MAT_X];
  }
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
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_("R","S",select,&n,A,&ld,Y,&ld,Y,&ld,&mm,&mout,ds->work,&info));
#else
  ierr = DSAllocateWork_Private(ds,2*ld,ld,0);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_("R","S",select,&n,A,&ld,Y,&ld,Y,&ld,&mm,&mout,ds->work,ds->rwork,&info));
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

static PetscErrorCode DSVectors_NHEPTS_Eigen_All(DS ds,PetscBool left)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n,ld,mout,info,inc=1,cols,zero=0;
  PetscBool      iscomplex;
  PetscScalar    *A,*Q,*X;
  PetscReal      norm,done=1.0;
  const char     *back;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  if (left) {
    A = ds->mat[DS_MAT_B];
    Q = ds->mat[DS_MAT_Z];
    X = ds->mat[DS_MAT_Y];
  } else {
    A = ds->mat[DS_MAT_A];
    Q = ds->mat[DS_MAT_Q];
    X = ds->mat[DS_MAT_X];
  }
  if (ds->state>=DS_STATE_CONDENSED) {
    /* DSSolve() has been called, backtransform with matrix Q */
    back = "B";
    ierr = PetscArraycpy(X,Q,ld*ld);CHKERRQ(ierr);
  } else back = "A";
#if !defined(PETSC_USE_COMPLEX)
  ierr = DSAllocateWork_Private(ds,3*ld,0,0);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_("R",back,NULL,&n,A,&ld,X,&ld,X,&ld,&n,&mout,ds->work,&info));
#else
  ierr = DSAllocateWork_Private(ds,2*ld,ld,0);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_("R",back,NULL,&n,A,&ld,X,&ld,X,&ld,&n,&mout,ds->work,ds->rwork,&info));
#endif
  SlepcCheckLapackInfo("trevc",info);

  /* normalize eigenvectors */
  for (i=0;i<n;i++) {
    iscomplex = (i<n-1 && A[i+1+i*ld]!=0.0)? PETSC_TRUE: PETSC_FALSE;
    cols = 1;
    norm = BLASnrm2_(&n,X+i*ld,&inc);
#if !defined(PETSC_USE_COMPLEX)
    if (iscomplex) {
      norm = SlepcAbsEigenvalue(norm,BLASnrm2_(&n,X+(i+1)*ld,&inc));
      cols = 2;
    }
#endif
    PetscStackCallBLAS("LAPACKlascl",LAPACKlascl_("G",&zero,&zero,&norm,&done,&n,&cols,X+i*ld,&ld,&info));
    SlepcCheckLapackInfo("lascl",info);
    if (iscomplex) i++;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSVectors_NHEPTS(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_X:
      if (ds->refined) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented yet");
      else {
        if (j) {
          ierr = DSVectors_NHEPTS_Eigen_Some(ds,j,rnorm,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = DSVectors_NHEPTS_Eigen_All(ds,PETSC_FALSE);CHKERRQ(ierr);
        }
      }
      break;
    case DS_MAT_Y:
      if (ds->refined) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented yet");
      if (j) {
        ierr = DSVectors_NHEPTS_Eigen_Some(ds,j,rnorm,PETSC_TRUE);CHKERRQ(ierr);
      } else {
        ierr = DSVectors_NHEPTS_Eigen_All(ds,PETSC_TRUE);CHKERRQ(ierr);
      }
      break;
    case DS_MAT_U:
    case DS_MAT_VT:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented yet");
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSort_NHEPTS(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscErrorCode ierr;
  DS_NHEPTS      *ctx = (DS_NHEPTS*)ds->data;
  PetscInt       i,j,cont,id=0,*p,*idx,*idx2;
  PetscReal      s,t;
#if defined(PETSC_USE_COMPLEX)
  Mat            A,U;
#endif

  PetscFunctionBegin;
  if (rr && wr != rr) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented yet");
  ierr = PetscMalloc3(ds->ld,&idx,ds->ld,&idx2,ds->ld,&p);CHKERRQ(ierr);
  ierr = DSSort_NHEP_Total(ds,ds->mat[DS_MAT_A],ds->mat[DS_MAT_Q],wr,wi);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = DSGetMat(ds,DS_MAT_B,&A);CHKERRQ(ierr);
  ierr = MatConjugate(A);CHKERRQ(ierr);
  ierr = DSRestoreMat(ds,DS_MAT_B,&A);CHKERRQ(ierr);
  ierr = DSGetMat(ds,DS_MAT_Z,&U);CHKERRQ(ierr);
  ierr = MatConjugate(U);CHKERRQ(ierr);
  ierr = DSRestoreMat(ds,DS_MAT_Z,&U);CHKERRQ(ierr);
  for (i=0;i<ds->n;i++) ctx->wr[i] = PetscConj(ctx->wr[i]);
#endif
  ierr = DSSort_NHEP_Total(ds,ds->mat[DS_MAT_B],ds->mat[DS_MAT_Z],ctx->wr,ctx->wi);CHKERRQ(ierr);
  /* check correct eigenvalue correspondence */
  cont = 0;
  for (i=0;i<ds->n;i++) {
    if (SlepcAbsEigenvalue(ctx->wr[i]-wr[i],ctx->wi[i]-wi[i])>PETSC_SQRT_MACHINE_EPSILON) {idx2[cont] = i; idx[cont++] = i;}
    p[i] = -1;
  }
  if (cont) {
    for (i=0;i<cont;i++) {
      t = PETSC_MAX_REAL;
      for (j=0;j<cont;j++) if (idx2[j]!=-1 && (s=SlepcAbsEigenvalue(ctx->wr[idx[j]]-wr[idx[i]],ctx->wi[idx[j]]-wi[idx[i]]))<t) { id = j; t = s; }
      p[idx[i]] = idx[id];
      idx2[id] = -1;
    }
    for (i=0;i<ds->n;i++) if (p[i]==-1) p[i] = i;
    ierr = DSSortWithPermutation_NHEP_Private(ds,p,ds->mat[DS_MAT_B],ds->mat[DS_MAT_Z],ctx->wr,ctx->wi);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_COMPLEX)
  ierr = DSGetMat(ds,DS_MAT_B,&A);CHKERRQ(ierr);
  ierr = MatConjugate(A);CHKERRQ(ierr);
  ierr = DSRestoreMat(ds,DS_MAT_B,&A);CHKERRQ(ierr);
  ierr = DSGetMat(ds,DS_MAT_Z,&U);CHKERRQ(ierr);
  ierr = MatConjugate(U);CHKERRQ(ierr);
  ierr = DSRestoreMat(ds,DS_MAT_Z,&U);CHKERRQ(ierr);
#endif
  ierr = PetscFree3(idx,idx2,p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSUpdateExtraRow_NHEPTS(DS ds)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n,ld,incx=1;
  PetscScalar    *A,*Q,*x,*y,one=1.0,zero=0.0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = DSAllocateWork_Private(ds,2*ld,0,0);CHKERRQ(ierr);
  x = ds->work;
  y = ds->work+ld;
  A = ds->mat[DS_MAT_A];
  Q = ds->mat[DS_MAT_Q];
  for (i=0;i<n;i++) x[i] = PetscConj(A[n+i*ld]);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&n,&one,Q,&ld,x,&incx,&zero,y,&incx));
  for (i=0;i<n;i++) A[n+i*ld] = PetscConj(y[i]);
  A = ds->mat[DS_MAT_B];
  Q = ds->mat[DS_MAT_Z];
  for (i=0;i<n;i++) x[i] = PetscConj(A[n+i*ld]);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&n,&one,Q,&ld,x,&incx,&zero,y,&incx));
  for (i=0;i<n;i++) A[n+i*ld] = PetscConj(y[i]);
  ds->k = n;
  PetscFunctionReturn(0);
}

PetscErrorCode DSSolve_NHEPTS(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  DS_NHEPTS      *ctx = (DS_NHEPTS*)ds->data;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  PetscValidScalarPointer(wi,3);
#endif
  ierr = DSSolve_NHEP_Private(ds,ds->mat[DS_MAT_A],ds->mat[DS_MAT_Q],wr,wi);CHKERRQ(ierr);
  ierr = DSSolve_NHEP_Private(ds,ds->mat[DS_MAT_B],ds->mat[DS_MAT_Z],ctx->wr,ctx->wi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSSynchronize_NHEPTS(DS ds,PetscScalar eigr[],PetscScalar eigi[])
{
  PetscErrorCode ierr;
  PetscInt       ld=ds->ld,l=ds->l,k;
  PetscMPIInt    n,rank,off=0,size,ldn;
  DS_NHEPTS      *ctx = (DS_NHEPTS*)ds->data;

  PetscFunctionBegin;
  k = 2*(ds->n-l)*ld;
  if (ds->state>DS_STATE_RAW) k += 2*(ds->n-l)*ld;
  if (eigr) k += ds->n-l;
  if (eigi) k += ds->n-l;
  if (ctx->wr) k += ds->n-l;
  if (ctx->wi) k += ds->n-l;
  ierr = DSAllocateWork_Private(ds,k,0,0);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(k*sizeof(PetscScalar),&size);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(ds->n-l,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(ld*(ds->n-l),&ldn);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ds),&rank);CHKERRMPI(ierr);
  if (!rank) {
    ierr = MPI_Pack(ds->mat[DS_MAT_A]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    ierr = MPI_Pack(ds->mat[DS_MAT_B]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    if (ds->state>DS_STATE_RAW) {
      ierr = MPI_Pack(ds->mat[DS_MAT_Q]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
      ierr = MPI_Pack(ds->mat[DS_MAT_Z]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigr) {
      ierr = MPI_Pack(eigr+l,n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigi) {
      ierr = MPI_Pack(eigi+l,n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (ctx->wr) {
      ierr = MPI_Pack(ctx->wr+l,n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (ctx->wi) {
      ierr = MPI_Pack(ctx->wi+l,n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
  }
  ierr = MPI_Bcast(ds->work,size,MPI_BYTE,0,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
  if (rank) {
    ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_A]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_B]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    if (ds->state>DS_STATE_RAW) {
      ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_Q]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
      ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_Z]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigr) {
      ierr = MPI_Unpack(ds->work,size,&off,eigr+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigi) {
      ierr = MPI_Unpack(ds->work,size,&off,eigi+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (ctx->wr) {
      ierr = MPI_Unpack(ds->work,size,&off,ctx->wr+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (ctx->wi) {
      ierr = MPI_Unpack(ds->work,size,&off,ctx->wi+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSGetTruncateSize_NHEPTS(DS ds,PetscInt l,PetscInt n,PetscInt *k)
{
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar *A = ds->mat[DS_MAT_A],*B = ds->mat[DS_MAT_B];
#endif

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  if (A[l+(*k)+(l+(*k)-1)*ds->ld] != 0.0 || B[l+(*k)+(l+(*k)-1)*ds->ld] != 0.0) {
    if (l+(*k)<n-1) (*k)++;
    else (*k)--;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode DSTruncate_NHEPTS(DS ds,PetscInt n,PetscBool trim)
{
  PetscInt    i,ld=ds->ld,l=ds->l;
  PetscScalar *A = ds->mat[DS_MAT_A],*B = ds->mat[DS_MAT_B];

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  /* make sure diagonal 2x2 block is not broken */
  if (ds->state>=DS_STATE_CONDENSED && n>0 && n<ds->n && A[n+(n-1)*ld]!=0.0 && B[n+(n-1)*ld]!=0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"The given size would break a 2x2 block, call DSGetTruncateSize() first");
#endif
  if (trim) {
    if (ds->extrarow) {   /* clean extra row */
      for (i=l;i<ds->n;i++) { A[ds->n+i*ld] = 0.0; B[ds->n+i*ld] = 0.0; }
    }
    ds->l = 0;
    ds->k = 0;
    ds->n = n;
    ds->t = ds->n;   /* truncated length equal to the new dimension */
  } else {
    if (ds->extrarow && ds->k==ds->n) {
      /* copy entries of extra row to the new position, then clean last row */
      for (i=l;i<n;i++) { A[n+i*ld] = A[ds->n+i*ld]; B[n+i*ld] = B[ds->n+i*ld]; }
      for (i=l;i<ds->n;i++) { A[ds->n+i*ld] = 0.0; B[ds->n+i*ld] = 0.0; }
    }
    ds->k = (ds->extrarow)? n: 0;
    ds->t = ds->n;   /* truncated length equal to previous dimension */
    ds->n = n;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSDestroy_NHEPTS(DS ds)
{
  PetscErrorCode ierr;
  DS_NHEPTS      *ctx = (DS_NHEPTS*)ds->data;

  PetscFunctionBegin;
  if (ctx->wr) { ierr = PetscFree(ctx->wr);CHKERRQ(ierr); }
  if (ctx->wi) { ierr = PetscFree(ctx->wi);CHKERRQ(ierr); }
  ierr = PetscFree(ds->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode DSCreate_NHEPTS(DS ds)
{
  DS_NHEPTS      *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ds,&ctx);CHKERRQ(ierr);
  ds->data = (void*)ctx;

  ds->ops->allocate        = DSAllocate_NHEPTS;
  ds->ops->view            = DSView_NHEPTS;
  ds->ops->vectors         = DSVectors_NHEPTS;
  ds->ops->solve[0]        = DSSolve_NHEPTS;
  ds->ops->sort            = DSSort_NHEPTS;
  ds->ops->synchronize     = DSSynchronize_NHEPTS;
  ds->ops->gettruncatesize = DSGetTruncateSize_NHEPTS;
  ds->ops->truncate        = DSTruncate_NHEPTS;
  ds->ops->update          = DSUpdateExtraRow_NHEPTS;
  ds->ops->destroy         = DSDestroy_NHEPTS;
  PetscFunctionReturn(0);
}

