/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_U);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_V);CHKERRQ(ierr);
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

static PetscErrorCode DSSwitchFormat_SVD(DS ds)
{
  PetscErrorCode ierr;
  DS_SVD         *ctx = (DS_SVD*)ds->data;
  PetscReal      *T = ds->rmat[DS_MAT_T];
  PetscScalar    *A = ds->mat[DS_MAT_A];
  PetscInt       i,m=ctx->m,k=ds->k,ld=ds->ld;

  PetscFunctionBegin;
  if (!m) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  /* switch from compact (arrow) to dense storage */
  ierr = PetscArrayzero(A,ld*ld);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    A[i+i*ld] = T[i];
    A[i+k*ld] = T[i+ld];
  }
  A[k+k*ld] = T[k];
  for (i=k+1;i<m;i++) {
    A[i+i*ld]   = T[i];
    A[i-1+i*ld] = T[i-1+ld];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSView_SVD(DS ds,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  DS_SVD            *ctx = (DS_SVD*)ds->data;
  PetscViewerFormat format;
  PetscInt          i,j,r,c,m=ctx->m,rows,cols;
  PetscReal         value;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO) PetscFunctionReturn(0);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer,"number of columns: %D\n",m);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (!m) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  if (ds->compact) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    rows = ds->n;
    cols = ds->extrarow? m+1: m;
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",rows,cols);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",2*ds->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
      for (i=0;i<PetscMin(ds->n,m);i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,(double)*(ds->rmat[DS_MAT_T]+i));CHKERRQ(ierr);
      }
      for (i=0;i<cols-1;i++) {
        r = PetscMax(i+2,ds->k+1);
        c = i+1;
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,(double)*(ds->rmat[DS_MAT_T]+ds->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",DSMatName[DS_MAT_T]);CHKERRQ(ierr);
    } else {
      for (i=0;i<rows;i++) {
        for (j=0;j<cols;j++) {
          if (i==j) value = *(ds->rmat[DS_MAT_T]+i);
          else if (i<ds->k && j==ds->k) value = *(ds->rmat[DS_MAT_T]+ds->ld+PetscMin(i,j));
          else if (i+1==j && i>=ds->k) value = *(ds->rmat[DS_MAT_T]+ds->ld+i);
          else value = 0.0;
          ierr = PetscViewerASCIIPrintf(viewer," %18.16e ",(double)value);CHKERRQ(ierr);
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
    ierr = DSViewMat(ds,viewer,DS_MAT_V);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  DS_SVD         *ctx = (DS_SVD*)ds->data;
  PetscInt       n,l,i,*perm,ld=ds->ld;
  PetscScalar    *A;
  PetscReal      *d;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  if (!ctx->m) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  l = ds->l;
  n = PetscMin(ds->n,ctx->m);
  A = ds->mat[DS_MAT_A];
  d = ds->rmat[DS_MAT_T];
  perm = ds->perm;
  if (!rr) {
    ierr = DSSortEigenvaluesReal_Private(ds,d,perm);CHKERRQ(ierr);
  } else {
    ierr = DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_FALSE);CHKERRQ(ierr);
  }
  for (i=l;i<n;i++) wr[i] = d[perm[i]];
  ierr = DSPermuteBoth_Private(ds,l,n,ds->n,ctx->m,DS_MAT_U,DS_MAT_V,perm);CHKERRQ(ierr);
  for (i=l;i<n;i++) d[i] = PetscRealPart(wr[i]);
  if (!ds->compact) {
    for (i=l;i<n;i++) A[i+i*ld] = wr[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSUpdateExtraRow_SVD(DS ds)
{
  PetscErrorCode ierr;
  DS_SVD         *ctx = (DS_SVD*)ds->data;
  PetscInt       i;
  PetscBLASInt   n=0,m=0,ld,incx=1;
  PetscScalar    *A,*U,*x,*y,one=1.0,zero=0.0;
  PetscReal      *e,beta;

  PetscFunctionBegin;
  if (!ctx->m) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ctx->m,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  A = ds->mat[DS_MAT_A];
  U = ds->mat[DS_MAT_U];
  e = ds->rmat[DS_MAT_T]+ld;

  if (ds->compact) {
    beta = e[m-1];   /* in compact, we assume all entries are zero except the last one */
    for (i=0;i<n;i++) e[i] = PetscRealPart(beta*U[n-1+i*ld]);
    ds->k = m;
  } else {
    ierr = DSAllocateWork_Private(ds,2*ld,0,0);CHKERRQ(ierr);
    x = ds->work;
    y = ds->work+ld;
    for (i=0;i<n;i++) x[i] = PetscConj(A[i+m*ld]);
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&n,&one,U,&ld,x,&incx,&zero,y,&incx));
    for (i=0;i<n;i++) A[i+m*ld] = PetscConj(y[i]);
    ds->k = m;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSTruncate_SVD(DS ds,PetscInt n,PetscBool trim)
{
  PetscInt    i,ld=ds->ld,l=ds->l;
  PetscScalar *A = ds->mat[DS_MAT_A];
  DS_SVD      *ctx = (DS_SVD*)ds->data;

  PetscFunctionBegin;
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
  PetscFunctionReturn(0);
}

PetscErrorCode DSSolve_SVD_DC(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  DS_SVD         *ctx = (DS_SVD*)ds->data;
  PetscInt       i,j;
  PetscBLASInt   n1,m1,info,l = 0,n = 0,m = 0,nm,ld,off,lwork;
  PetscScalar    *A,*U,*V,*W,qwork;
  PetscReal      *d,*e,*Ur,*Vr;

  PetscFunctionBegin;
  if (!ctx->m) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ctx->m,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->l,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  n1 = n-l;     /* n1 = size of leading block, excl. locked + size of trailing block */
  m1 = m-l;
  off = l+l*ld;
  A = ds->mat[DS_MAT_A];
  U = ds->mat[DS_MAT_U];
  V = ds->mat[DS_MAT_V];
  d = ds->rmat[DS_MAT_T];
  e = ds->rmat[DS_MAT_T]+ld;
  ierr = PetscArrayzero(U,ld*ld);CHKERRQ(ierr);
  for (i=0;i<l;i++) U[i+i*ld] = 1.0;
  ierr = PetscArrayzero(V,ld*ld);CHKERRQ(ierr);
  for (i=0;i<l;i++) V[i+i*ld] = 1.0;

  if (ds->state>DS_STATE_RAW) {
    /* solve bidiagonal SVD problem */
    for (i=0;i<l;i++) wr[i] = d[i];
#if defined(PETSC_USE_COMPLEX)
    ierr = DSAllocateWork_Private(ds,0,3*n1*n1+4*n1,8*n1);CHKERRQ(ierr);
    ierr = DSAllocateMatReal_Private(ds,DS_MAT_U);CHKERRQ(ierr);
    ierr = DSAllocateMatReal_Private(ds,DS_MAT_V);CHKERRQ(ierr);
    Ur = ds->rmat[DS_MAT_U];
    Vr = ds->rmat[DS_MAT_V];
#else
    ierr = DSAllocateWork_Private(ds,0,3*n1*n1+4*n1+ld*ld,8*n1);CHKERRQ(ierr);
    Ur = U;
    Vr = ds->rwork+3*n1*n1+4*n1;
#endif
    PetscStackCallBLAS("LAPACKbdsdc",LAPACKbdsdc_("U","I",&n1,d+l,e+l,Ur+off,&ld,Vr+off,&ld,NULL,NULL,ds->rwork,ds->iwork,&info));
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
    ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
    W = ds->mat[DS_MAT_W];
    if (ds->compact) { ierr = DSSwitchFormat_SVD(ds);CHKERRQ(ierr); }
    for (i=0;i<l;i++) wr[i] = d[i];
    nm = PetscMin(n,m);
    ierr = DSAllocateWork_Private(ds,0,0,8*nm);CHKERRQ(ierr);
    lwork = -1;
#if defined(PETSC_USE_COMPLEX)
    ierr = DSAllocateWork_Private(ds,0,5*nm*nm+7*nm,0);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n1,&m1,A+off,&ld,d+l,U+off,&ld,W+off,&ld,&qwork,&lwork,ds->rwork,ds->iwork,&info));
#else
    PetscStackCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n1,&m1,A+off,&ld,d+l,U+off,&ld,W+off,&ld,&qwork,&lwork,ds->iwork,&info));
#endif
    SlepcCheckLapackInfo("gesdd",info);
    ierr = PetscBLASIntCast((PetscInt)PetscRealPart(qwork),&lwork);CHKERRQ(ierr);
    ierr = DSAllocateWork_Private(ds,lwork,0,0);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n1,&m1,A+off,&ld,d+l,U+off,&ld,W+off,&ld,ds->work,&lwork,ds->rwork,ds->iwork,&info));
#else
    PetscStackCallBLAS("LAPACKgesdd",LAPACKgesdd_("A",&n1,&m1,A+off,&ld,d+l,U+off,&ld,W+off,&ld,ds->work,&lwork,ds->iwork,&info));
#endif
    SlepcCheckLapackInfo("gesdd",info);
    for (i=l;i<m;i++) {
      for (j=l;j<m;j++) V[i+j*ld] = PetscConj(W[j+i*ld]);  /* transpose VT returned by Lapack */
    }
  }
  for (i=l;i<PetscMin(ds->n,ctx->m);i++) wr[i] = d[i];

  /* create diagonal matrix as a result */
  if (ds->compact) {
    ierr = PetscArrayzero(e,n-1);CHKERRQ(ierr);
  } else {
    for (i=l;i<m;i++) {
      ierr = PetscArrayzero(A+l+i*ld,n-l);CHKERRQ(ierr);
    }
    for (i=l;i<n;i++) A[i+i*ld] = d[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSynchronize_SVD(DS ds,PetscScalar eigr[],PetscScalar eigi[])
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
      ierr = MPI_Pack(ds->mat[DS_MAT_V]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
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
      ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_V]+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigr) {
      ierr = MPI_Unpack(ds->work,size,&off,eigr+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSMatGetSize_SVD(DS ds,DSMatType t,PetscInt *rows,PetscInt *cols)
{
  DS_SVD *ctx = (DS_SVD*)ds->data;

  PetscFunctionBegin;
  if (!ctx->m) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSSVDSetDimensions()");
  switch (t) {
    case DS_MAT_A:
    case DS_MAT_T:
      *rows = ds->n;
      *cols = ds->extrarow? ctx->m+1: ctx->m;
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
    if (m<1 || m>ds->ld) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of m. Must be between 1 and ld");
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,m,2);
  ierr = PetscTryMethod(ds,"DSSVDSetDimensions_C",(DS,PetscInt),(ds,m));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidIntPointer(m,2);
  ierr = PetscUseMethod(ds,"DSSVDGetDimensions_C",(DS,PetscInt*),(ds,m));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSDestroy_SVD(DS ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ds->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSSVDSetDimensions_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSSVDGetDimensions_C",NULL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ds,&ctx);CHKERRQ(ierr);
  ds->data = (void*)ctx;

  ds->ops->allocate      = DSAllocate_SVD;
  ds->ops->view          = DSView_SVD;
  ds->ops->vectors       = DSVectors_SVD;
  ds->ops->solve[0]      = DSSolve_SVD_DC;
  ds->ops->sort          = DSSort_SVD;
  ds->ops->synchronize   = DSSynchronize_SVD;
  ds->ops->truncate      = DSTruncate_SVD;
  ds->ops->update        = DSUpdateExtraRow_SVD;
  ds->ops->destroy       = DSDestroy_SVD;
  ds->ops->matgetsize    = DSMatGetSize_SVD;
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSSVDSetDimensions_C",DSSVDSetDimensions_SVD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSSVDGetDimensions_C",DSSVDGetDimensions_SVD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

