/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/dsimpl.h>       /*I "slepcds.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  PetscInt m;              /* number of columns */
  PetscInt t;              /* number of rows of V after truncating */
} DS_SVD;

PetscErrorCode DSAllocate_SVD(DS ds,PetscInt ld)
{
  PetscFunctionBegin;
  PetscCall(DSAllocateMat_Private(ds,DS_MAT_A));
  PetscCall(DSAllocateMat_Private(ds,DS_MAT_U));
  PetscCall(DSAllocateMat_Private(ds,DS_MAT_V));
  PetscCall(DSAllocateMat_Private(ds,DS_MAT_T));
  PetscCall(PetscFree(ds->perm));
  PetscCall(PetscMalloc1(ld,&ds->perm));
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

static PetscErrorCode DSSwitchFormat_SVD(DS ds)
{
  DS_SVD         *ctx = (DS_SVD*)ds->data;
  PetscReal      *T;
  PetscScalar    *A;
  PetscInt       i,m=ctx->m,k=ds->k,ld=ds->ld;

  PetscFunctionBegin;
  PetscCheck(m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  /* switch from compact (arrow) to dense storage */
  PetscCall(MatDenseGetArrayWrite(ds->omat[DS_MAT_A],&A));
  PetscCall(DSGetArrayReal(ds,DS_MAT_T,&T));
  PetscCall(PetscArrayzero(A,ld*ld));
  for (i=0;i<k;i++) {
    A[i+i*ld] = T[i];
    A[i+k*ld] = T[i+ld];
  }
  A[k+k*ld] = T[k];
  for (i=k+1;i<m;i++) {
    A[i+i*ld]   = T[i];
    A[i-1+i*ld] = T[i-1+ld];
  }
  PetscCall(MatDenseRestoreArrayWrite(ds->omat[DS_MAT_A],&A));
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
  PetscFunctionReturn(0);
}

PetscErrorCode DSView_SVD(DS ds,PetscViewer viewer)
{
  DS_SVD            *ctx = (DS_SVD*)ds->data;
  PetscViewerFormat format;
  PetscInt          i,j,r,c,m=ctx->m,rows,cols;
  PetscReal         *T,value;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_ASCII_INFO) PetscFunctionReturn(0);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"number of columns: %" PetscInt_FMT "\n",m));
    PetscFunctionReturn(0);
  }
  PetscCheck(m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  if (ds->compact) {
    PetscCall(DSGetArrayReal(ds,DS_MAT_T,&T));
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    rows = ds->n;
    cols = ds->extrarow? m+1: m;
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"%% Size = %" PetscInt_FMT " %" PetscInt_FMT "\n",rows,cols));
      PetscCall(PetscViewerASCIIPrintf(viewer,"zzz = zeros(%" PetscInt_FMT ",3);\n",2*ds->n));
      PetscCall(PetscViewerASCIIPrintf(viewer,"zzz = [\n"));
      for (i=0;i<PetscMin(ds->n,m);i++) PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e\n",i+1,i+1,(double)T[i]));
      for (i=0;i<cols-1;i++) {
        r = PetscMax(i+2,ds->k+1);
        c = i+1;
        PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e\n",c,r,(double)T[i+ds->ld]));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",DSMatName[DS_MAT_T]));
    } else {
      for (i=0;i<rows;i++) {
        for (j=0;j<cols;j++) {
          if (i==j) value = T[i];
          else if (i<ds->k && j==ds->k) value = T[PetscMin(i,j)+ds->ld];
          else if (i+1==j && i>=ds->k) value = T[i+ds->ld];
          else value = 0.0;
          PetscCall(PetscViewerASCIIPrintf(viewer," %18.16e ",(double)value));
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      }
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
  } else PetscCall(DSViewMat(ds,viewer,DS_MAT_A));
  if (ds->state>DS_STATE_INTERMEDIATE) {
    PetscCall(DSViewMat(ds,viewer,DS_MAT_U));
    PetscCall(DSViewMat(ds,viewer,DS_MAT_V));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSVectors_SVD(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_U:
    case DS_MAT_V:
      if (rnorm) *rnorm = 0.0;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSort_SVD(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  DS_SVD         *ctx = (DS_SVD*)ds->data;
  PetscInt       n,l,i,*perm,ld=ds->ld;
  PetscScalar    *A;
  PetscReal      *d;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  PetscCheck(ctx->m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  l = ds->l;
  n = PetscMin(ds->n,ctx->m);
  PetscCall(DSGetArrayReal(ds,DS_MAT_T,&d));
  perm = ds->perm;
  if (!rr) PetscCall(DSSortEigenvaluesReal_Private(ds,d,perm));
  else PetscCall(DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_FALSE));
  for (i=l;i<n;i++) wr[i] = d[perm[i]];
  PetscCall(DSPermuteBoth_Private(ds,l,n,ds->n,ctx->m,DS_MAT_U,DS_MAT_V,perm));
  for (i=l;i<n;i++) d[i] = PetscRealPart(wr[i]);
  if (!ds->compact) {
    PetscCall(MatDenseGetArray(ds->omat[DS_MAT_A],&A));
    for (i=l;i<n;i++) A[i+i*ld] = wr[i];
    PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_A],&A));
  }
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&d));
  PetscFunctionReturn(0);
}

PetscErrorCode DSUpdateExtraRow_SVD(DS ds)
{
  DS_SVD            *ctx = (DS_SVD*)ds->data;
  PetscInt          i;
  PetscBLASInt      n=0,m=0,ld,incx=1;
  PetscScalar       *A,*x,*y,one=1.0,zero=0.0;
  PetscReal         *T,*e,beta;
  const PetscScalar *U;

  PetscFunctionBegin;
  PetscCheck(ctx->m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  PetscCall(PetscBLASIntCast(ds->n,&n));
  PetscCall(PetscBLASIntCast(ctx->m,&m));
  PetscCall(PetscBLASIntCast(ds->ld,&ld));
  PetscCall(MatDenseGetArrayRead(ds->omat[DS_MAT_U],&U));
  if (ds->compact) {
    PetscCall(DSGetArrayReal(ds,DS_MAT_T,&T));
    e = T+ld;
    beta = e[m-1];   /* in compact, we assume all entries are zero except the last one */
    for (i=0;i<n;i++) e[i] = PetscRealPart(beta*U[n-1+i*ld]);
    ds->k = m;
    PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
  } else {
    PetscCall(MatDenseGetArray(ds->omat[DS_MAT_A],&A));
    PetscCall(DSAllocateWork_Private(ds,2*ld,0,0));
    x = ds->work;
    y = ds->work+ld;
    for (i=0;i<n;i++) x[i] = PetscConj(A[i+m*ld]);
    PetscCallBLAS("BLASgemv",BLASgemv_("C",&n,&n,&one,U,&ld,x,&incx,&zero,y,&incx));
    for (i=0;i<n;i++) A[i+m*ld] = PetscConj(y[i]);
    ds->k = m;
    PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_A],&A));
  }
  PetscCall(MatDenseRestoreArrayRead(ds->omat[DS_MAT_U],&U));
  PetscFunctionReturn(0);
}

PetscErrorCode DSTruncate_SVD(DS ds,PetscInt n,PetscBool trim)
{
  PetscInt    i,ld=ds->ld,l=ds->l;
  PetscScalar *A;
  DS_SVD      *ctx = (DS_SVD*)ds->data;

  PetscFunctionBegin;
  if (!ds->compact && ds->extrarow) PetscCall(MatDenseGetArray(ds->omat[DS_MAT_A],&A));
  if (trim) {
    if (!ds->compact && ds->extrarow) {   /* clean extra column */
      for (i=l;i<ds->n;i++) A[i+ctx->m*ld] = 0.0;
    }
    ds->l  = 0;
    ds->k  = 0;
    ds->n  = n;
    ctx->m = n;
    ds->t  = ds->n;   /* truncated length equal to the new dimension */
    ctx->t = ctx->m;  /* must also keep the previous dimension of V */
  } else {
    if (!ds->compact && ds->extrarow && ds->k==ds->n) {
      /* copy entries of extra column to the new position, then clean last row */
      for (i=l;i<n;i++) A[i+n*ld] = A[i+ctx->m*ld];
      for (i=l;i<ds->n;i++) A[i+ctx->m*ld] = 0.0;
    }
    ds->k  = (ds->extrarow)? n: 0;
    ds->t  = ds->n;   /* truncated length equal to previous dimension */
    ctx->t = ctx->m;  /* must also keep the previous dimension of V */
    ds->n  = n;
    ctx->m = n;
  }
  if (!ds->compact && ds->extrarow) PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_A],&A));
  PetscFunctionReturn(0);
}

PetscErrorCode DSSolve_SVD_DC(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  DS_SVD         *ctx = (DS_SVD*)ds->data;
  PetscInt       i,j;
  PetscBLASInt   n1,m1,info,l = 0,n = 0,m = 0,nm,ld,off,lwork;
  PetscScalar    *A,*U,*V,*W,qwork;
  PetscReal      *d,*e,*Ur,*Vr;

  PetscFunctionBegin;
  PetscCheck(ctx->m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  PetscCall(PetscBLASIntCast(ds->n,&n));
  PetscCall(PetscBLASIntCast(ctx->m,&m));
  PetscCall(PetscBLASIntCast(ds->l,&l));
  PetscCall(PetscBLASIntCast(ds->ld,&ld));
  n1 = n-l;     /* n1 = size of leading block, excl. locked + size of trailing block */
  m1 = m-l;
  off = l+l*ld;
  PetscCall(MatDenseGetArray(ds->omat[DS_MAT_A],&A));
  PetscCall(MatDenseGetArrayWrite(ds->omat[DS_MAT_U],&U));
  PetscCall(MatDenseGetArrayWrite(ds->omat[DS_MAT_V],&V));
  PetscCall(DSGetArrayReal(ds,DS_MAT_T,&d));
  e = d+ld;
  PetscCall(PetscArrayzero(U,ld*ld));
  for (i=0;i<l;i++) U[i+i*ld] = 1.0;
  PetscCall(PetscArrayzero(V,ld*ld));
  for (i=0;i<l;i++) V[i+i*ld] = 1.0;

  if (ds->state>DS_STATE_RAW) {
    /* solve bidiagonal SVD problem */
    for (i=0;i<l;i++) wr[i] = d[i];
#if defined(PETSC_USE_COMPLEX)
    PetscCall(DSAllocateWork_Private(ds,0,3*n1*n1+4*n1+2*ld*ld,8*n1));
    Ur = ds->rwork+3*n1*n1+4*n1;
    Vr = ds->rwork+3*n1*n1+4*n1+ld*ld;
#else
    PetscCall(DSAllocateWork_Private(ds,0,3*n1*n1+4*n1+ld*ld,8*n1));
    Ur = U;
    Vr = ds->rwork+3*n1*n1+4*n1;
#endif
    PetscCallBLAS("LAPACKbdsdc",LAPACKbdsdc_("U","I",&n1,d+l,e+l,Ur+off,&ld,Vr+off,&ld,NULL,NULL,ds->rwork,ds->iwork,&info));
    SlepcCheckLapackInfo("bdsdc",info);
    for (i=l;i<n;i++) {
      for (j=l;j<n;j++) {
#if defined(PETSC_USE_COMPLEX)
        U[i+j*ld] = Ur[i+j*ld];
#endif
        V[i+j*ld] = PetscConj(Vr[j+i*ld]);  /* transpose VT returned by Lapack */
      }
    }
  } else {
    /* solve general rectangular SVD problem */
    PetscCall(DSAllocateMat_Private(ds,DS_MAT_W));
    PetscCall(MatDenseGetArrayWrite(ds->omat[DS_MAT_W],&W));
    if (ds->compact) PetscCall(DSSwitchFormat_SVD(ds));
    for (i=0;i<l;i++) wr[i] = d[i];
    nm = PetscMin(n,m);
    PetscCall(DSAllocateWork_Private(ds,0,0,8*nm));
    lwork = -1;
#if defined(PETSC_USE_COMPLEX)
    PetscCall(DSAllocateWork_Private(ds,0,5*nm*nm+7*nm,0));
    PetscCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n1,&m1,A+off,&ld,d+l,U+off,&ld,W+off,&ld,&qwork,&lwork,ds->rwork,ds->iwork,&info));
#else
    PetscCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n1,&m1,A+off,&ld,d+l,U+off,&ld,W+off,&ld,&qwork,&lwork,ds->iwork,&info));
#endif
    SlepcCheckLapackInfo("gesdd",info);
    PetscCall(PetscBLASIntCast((PetscInt)PetscRealPart(qwork),&lwork));
    PetscCall(DSAllocateWork_Private(ds,lwork,0,0));
#if defined(PETSC_USE_COMPLEX)
    PetscCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n1,&m1,A+off,&ld,d+l,U+off,&ld,W+off,&ld,ds->work,&lwork,ds->rwork,ds->iwork,&info));
#else
    PetscCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n1,&m1,A+off,&ld,d+l,U+off,&ld,W+off,&ld,ds->work,&lwork,ds->iwork,&info));
#endif
    SlepcCheckLapackInfo("gesdd",info);
    for (i=l;i<m;i++) {
      for (j=l;j<m;j++) V[i+j*ld] = PetscConj(W[j+i*ld]);  /* transpose VT returned by Lapack */
    }
    PetscCall(MatDenseRestoreArrayWrite(ds->omat[DS_MAT_W],&W));
  }
  for (i=l;i<PetscMin(ds->n,ctx->m);i++) wr[i] = d[i];

  /* create diagonal matrix as a result */
  if (ds->compact) PetscCall(PetscArrayzero(e,n-1));
  else {
    for (i=l;i<m;i++) PetscCall(PetscArrayzero(A+l+i*ld,n-l));
    for (i=l;i<n;i++) A[i+i*ld] = d[i];
  }
  PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_A],&A));
  PetscCall(MatDenseRestoreArrayWrite(ds->omat[DS_MAT_U],&U));
  PetscCall(MatDenseRestoreArrayWrite(ds->omat[DS_MAT_V],&V));
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&d));
  PetscFunctionReturn(0);
}

#if !defined(PETSC_HAVE_MPIUNI)
PetscErrorCode DSSynchronize_SVD(DS ds,PetscScalar eigr[],PetscScalar eigi[])
{
  PetscInt       ld=ds->ld,l=ds->l,k=0,kr=0;
  PetscMPIInt    n,rank,off=0,size,ldn,ld3;
  PetscScalar    *A,*U,*V;
  PetscReal      *T;

  PetscFunctionBegin;
  if (ds->compact) kr = 3*ld;
  else k = (ds->n-l)*ld;
  if (ds->state>DS_STATE_RAW) k += 2*(ds->n-l)*ld;
  if (eigr) k += ds->n-l;
  PetscCall(DSAllocateWork_Private(ds,k+kr,0,0));
  PetscCall(PetscMPIIntCast(k*sizeof(PetscScalar)+kr*sizeof(PetscReal),&size));
  PetscCall(PetscMPIIntCast(ds->n-l,&n));
  PetscCall(PetscMPIIntCast(ld*(ds->n-l),&ldn));
  PetscCall(PetscMPIIntCast(3*ld,&ld3));
  if (ds->compact) PetscCall(DSGetArrayReal(ds,DS_MAT_T,&T));
  else PetscCall(MatDenseGetArray(ds->omat[DS_MAT_A],&A));
  if (ds->state>DS_STATE_RAW) {
    PetscCall(MatDenseGetArray(ds->omat[DS_MAT_U],&U));
    PetscCall(MatDenseGetArray(ds->omat[DS_MAT_V],&V));
  }
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ds),&rank));
  if (!rank) {
    if (ds->compact) PetscCallMPI(MPI_Pack(T,ld3,MPIU_REAL,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
    else PetscCallMPI(MPI_Pack(A+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
    if (ds->state>DS_STATE_RAW) {
      PetscCallMPI(MPI_Pack(U+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
      PetscCallMPI(MPI_Pack(V+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
    }
    if (eigr) PetscCallMPI(MPI_Pack(eigr+l,n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
  }
  PetscCallMPI(MPI_Bcast(ds->work,size,MPI_BYTE,0,PetscObjectComm((PetscObject)ds)));
  if (rank) {
    if (ds->compact) PetscCallMPI(MPI_Unpack(ds->work,size,&off,T,ld3,MPIU_REAL,PetscObjectComm((PetscObject)ds)));
    else PetscCallMPI(MPI_Unpack(ds->work,size,&off,A+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
    if (ds->state>DS_STATE_RAW) {
      PetscCallMPI(MPI_Unpack(ds->work,size,&off,U+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
      PetscCallMPI(MPI_Unpack(ds->work,size,&off,V+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
    }
    if (eigr) PetscCallMPI(MPI_Unpack(ds->work,size,&off,eigr+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
  }
  if (ds->compact) PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
  PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_A],&A));
  if (ds->state>DS_STATE_RAW) {
    PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_U],&U));
    PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_V],&V));
  }
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode DSMatGetSize_SVD(DS ds,DSMatType t,PetscInt *rows,PetscInt *cols)
{
  DS_SVD *ctx = (DS_SVD*)ds->data;

  PetscFunctionBegin;
  PetscCheck(ctx->m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  switch (t) {
    case DS_MAT_A:
      *rows = ds->n;
      *cols = ds->extrarow? ctx->m+1: ctx->m;
      break;
    case DS_MAT_T:
      *rows = ds->n;
      *cols = PetscDefined(USE_COMPLEX)? 2: 3;
      break;
    case DS_MAT_U:
      *rows = ds->state==DS_STATE_TRUNCATED? ds->t: ds->n;
      *cols = ds->n;
      break;
    case DS_MAT_V:
      *rows = ds->state==DS_STATE_TRUNCATED? ctx->t: ctx->m;
      *cols = ctx->m;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid t parameter");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DSSVDSetDimensions_SVD(DS ds,PetscInt m)
{
  DS_SVD *ctx = (DS_SVD*)ds->data;

  PetscFunctionBegin;
  DSCheckAlloc(ds,1);
  if (m==PETSC_DECIDE || m==PETSC_DEFAULT) {
    ctx->m = ds->ld;
  } else {
    PetscCheck(m>0 && m<=ds->ld,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of m. Must be between 1 and ld");
    ctx->m = m;
  }
  PetscFunctionReturn(0);
}

/*@
   DSSVDSetDimensions - Sets the number of columns for a DSSVD.

   Logically Collective on ds

   Input Parameters:
+  ds - the direct solver context
-  m  - the number of columns

   Notes:
   This call is complementary to DSSetDimensions(), to provide a dimension
   that is specific to this DS type.

   Level: intermediate

.seealso: DSSVDGetDimensions(), DSSetDimensions()
@*/
PetscErrorCode DSSVDSetDimensions(DS ds,PetscInt m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,m,2);
  PetscTryMethod(ds,"DSSVDSetDimensions_C",(DS,PetscInt),(ds,m));
  PetscFunctionReturn(0);
}

static PetscErrorCode DSSVDGetDimensions_SVD(DS ds,PetscInt *m)
{
  DS_SVD *ctx = (DS_SVD*)ds->data;

  PetscFunctionBegin;
  *m = ctx->m;
  PetscFunctionReturn(0);
}

/*@
   DSSVDGetDimensions - Returns the number of columns for a DSSVD.

   Not collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameters:
.  m - the number of columns

   Level: intermediate

.seealso: DSSVDSetDimensions()
@*/
PetscErrorCode DSSVDGetDimensions(DS ds,PetscInt *m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidIntPointer(m,2);
  PetscUseMethod(ds,"DSSVDGetDimensions_C",(DS,PetscInt*),(ds,m));
  PetscFunctionReturn(0);
}

PetscErrorCode DSDestroy_SVD(DS ds)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(ds->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ds,"DSSVDSetDimensions_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ds,"DSSVDGetDimensions_C",NULL));
  PetscFunctionReturn(0);
}

/*MC
   DSSVD - Dense Singular Value Decomposition.

   Level: beginner

   Notes:
   The problem is expressed as A = U*Sigma*V', where A is rectangular in
   general, with n rows and m columns. Sigma is a diagonal matrix whose diagonal
   elements are the arguments of DSSolve(). After solve, A is overwritten
   with Sigma.

   The orthogonal (or unitary) matrices of left and right singular vectors, U
   and V, have size n and m, respectively. The number of columns m must
   be specified via DSSVDSetDimensions().

   If the DS object is in the intermediate state, A is assumed to be in upper
   bidiagonal form (possibly with an arrow) and is stored in compact format
   on matrix T. Otherwise, no particular structure is assumed. The compact
   storage is implemented for the square case only, m=n. The extra row should
   be interpreted in this case as an extra column.

   Used DS matrices:
+  DS_MAT_A - problem matrix
-  DS_MAT_T - upper bidiagonal matrix

   Implemented methods:
.  0 - Divide and Conquer (_bdsdc or _gesdd)

.seealso: DSCreate(), DSSetType(), DSType, DSSVDSetDimensions()
M*/
SLEPC_EXTERN PetscErrorCode DSCreate_SVD(DS ds)
{
  DS_SVD         *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  ds->data = (void*)ctx;

  ds->ops->allocate      = DSAllocate_SVD;
  ds->ops->view          = DSView_SVD;
  ds->ops->vectors       = DSVectors_SVD;
  ds->ops->solve[0]      = DSSolve_SVD_DC;
  ds->ops->sort          = DSSort_SVD;
#if !defined(PETSC_HAVE_MPIUNI)
  ds->ops->synchronize   = DSSynchronize_SVD;
#endif
  ds->ops->truncate      = DSTruncate_SVD;
  ds->ops->update        = DSUpdateExtraRow_SVD;
  ds->ops->destroy       = DSDestroy_SVD;
  ds->ops->matgetsize    = DSMatGetSize_SVD;
  PetscCall(PetscObjectComposeFunction((PetscObject)ds,"DSSVDSetDimensions_C",DSSVDSetDimensions_SVD));
  PetscCall(PetscObjectComposeFunction((PetscObject)ds,"DSSVDGetDimensions_C",DSSVDGetDimensions_SVD));
  PetscFunctionReturn(0);
}
