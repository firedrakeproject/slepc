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
} DS_HSVD;

PetscErrorCode DSAllocate_HSVD(DS ds,PetscInt ld)
{
  PetscFunctionBegin;
  PetscCall(DSAllocateMat_Private(ds,DS_MAT_A));
  PetscCall(DSAllocateMat_Private(ds,DS_MAT_U));
  PetscCall(DSAllocateMat_Private(ds,DS_MAT_V));
  PetscCall(DSAllocateMat_Private(ds,DS_MAT_T));
  PetscCall(DSAllocateMat_Private(ds,DS_MAT_D));
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

#if 0
static PetscErrorCode DSSwitchFormat_HSVD(DS ds)
{
  DS_HSVD        *ctx = (DS_HSVD*)ds->data;
  PetscReal      *T,*S;
  PetscScalar    *A,*B;
  PetscInt       i,m=ctx->m,k=ds->k,ld=ds->ld;

  PetscFunctionBegin;
  PetscCheck(m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSHSVDSetDimensions()");
  /* switch from compact (arrow) to dense storage */
  PetscCall(MatDenseGetArray(ds->omat[DS_MAT_A],&A));
  PetscCall(MatDenseGetArray(ds->omat[DS_MAT_B],&B));
  PetscCall(DSGetArrayReal(ds,DS_MAT_T,&T));
  PetscCall(DSGetArrayReal(ds,DS_MAT_D,&S));
  PetscCall(PetscArrayzero(A,ld*ld));
  PetscCall(PetscArrayzero(B,ld*ld));
  for (i=0;i<k;i++) {
    A[i+i*ld] = T[i];
    A[i+k*ld] = T[i+ld];
    B[i+i*ld] = S[i];
  }
  A[k+k*ld] = T[k];
  B[k+k*ld] = S[k];
  for (i=k+1;i<m;i++) {
    A[i+i*ld]   = T[i];
    A[i-1+i*ld] = T[i-1+ld];
    B[i+i*ld]   = S[i];
  }
  PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_A],&A));
  PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_B],&B));
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_D,&S));
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode DSView_HSVD(DS ds,PetscViewer viewer)
{
  DS_HSVD           *ctx = (DS_HSVD*)ds->data;
  PetscViewerFormat format;
  PetscInt          i,j,r,c,m=ctx->m,rows,cols;
  PetscReal         *T,*S,value;
  const char        *methodname[] = {
                     "Cross"
  };
  const int         nmeth=PETSC_STATIC_ARRAY_LENGTH(methodname);

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer,&format));
  PetscCheck(m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSHSVDSetDimensions()");
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscCall(PetscViewerASCIIPrintf(viewer,"number of columns: %" PetscInt_FMT "\n",m));
    if (ds->method<nmeth) PetscCall(PetscViewerASCIIPrintf(viewer,"solving the problem with: %s\n",methodname[ds->method]));
    PetscFunctionReturn(0);
  }
  if (ds->compact) {
    PetscCall(DSGetArrayReal(ds,DS_MAT_T,&T));
    PetscCall(DSGetArrayReal(ds,DS_MAT_D,&S));
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
      PetscCall(PetscViewerASCIIPrintf(viewer,"%% Size = %" PetscInt_FMT " %" PetscInt_FMT "\n",ds->n,ds->n));
      PetscCall(PetscViewerASCIIPrintf(viewer,"omega = zeros(%" PetscInt_FMT ",3);\n",3*ds->n));
      PetscCall(PetscViewerASCIIPrintf(viewer,"omega = [\n"));
      for (i=0;i<ds->n;i++) PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e\n",i+1,i+1,(double)S[i]));
      PetscCall(PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(omega);\n",DSMatName[DS_MAT_B]));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"T\n"));
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
      PetscCall(PetscViewerASCIIPrintf(viewer,"omega\n"));
      for (i=0;i<ds->n;i++) {
        for (j=0;j<ds->n;j++) {
          if (i==j) value = S[i];
          else value = 0.0;
          PetscCall(PetscViewerASCIIPrintf(viewer," %18.16e ",(double)value));
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      }
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
    PetscCall(DSRestoreArrayReal(ds,DS_MAT_D,&S));
  } else {
    PetscCall(DSViewMat(ds,viewer,DS_MAT_A));
    PetscCall(DSViewMat(ds,viewer,DS_MAT_B));
  }
  if (ds->state>DS_STATE_INTERMEDIATE) {
    PetscCall(DSViewMat(ds,viewer,DS_MAT_U));
    PetscCall(DSViewMat(ds,viewer,DS_MAT_V));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSVectors_HSVD(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
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

PetscErrorCode DSSort_HSVD(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  DS_HSVD        *ctx = (DS_HSVD*)ds->data;
  PetscInt       n,l,i,*perm,ld=ds->ld;
  PetscScalar    *A,*B;
  PetscReal      *d,*s;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  PetscCheck(ctx->m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSHSVDSetDimensions()");
  l = ds->l;
  n = PetscMin(ds->n,ctx->m);
  PetscCall(DSGetArrayReal(ds,DS_MAT_T,&d));
  PetscCall(DSGetArrayReal(ds,DS_MAT_D,&s));
  PetscCall(DSAllocateWork_Private(ds,0,ds->ld,0));
  perm = ds->perm;
  if (!rr) PetscCall(DSSortEigenvaluesReal_Private(ds,d,perm));
  else PetscCall(DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_FALSE));
  PetscCall(PetscArraycpy(ds->rwork,s,n));
  for (i=l;i<n;i++) s[i]  = ds->rwork[perm[i]];
  for (i=l;i<n;i++) wr[i] = d[perm[i]];
  PetscCall(DSPermuteBoth_Private(ds,l,n,ds->n,ctx->m,DS_MAT_U,DS_MAT_V,perm));
  for (i=l;i<n;i++) d[i] = PetscRealPart(wr[i]);
  if (!ds->compact) {
    PetscCall(MatDenseGetArray(ds->omat[DS_MAT_A],&A));
    for (i=l;i<n;i++) A[i+i*ld] = wr[i];
    PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_A],&A));
    PetscCall(MatDenseGetArray(ds->omat[DS_MAT_B],&B));
    for (i=l;i<n;i++) B[i+i*ld] = s[i];
    PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_B],&B));
  }
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&d));
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_D,&s));
  PetscFunctionReturn(0);
}

PetscErrorCode DSUpdateExtraRow_HSVD(DS ds)
{
  DS_HSVD           *ctx = (DS_HSVD*)ds->data;
  PetscInt          i;
  PetscBLASInt      n=0,m=0,ld,l,n1,incx=1;
  PetscScalar       *A,*U,*x,*y,one=1.0,zero=0.0;
  PetscReal         *T,*e,*Omega,beta;

  PetscFunctionBegin;
  PetscCheck(ctx->m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSHSVDSetDimensions()");
  PetscCall(PetscBLASIntCast(ds->n,&n));
  PetscCall(PetscBLASIntCast(ctx->m,&m));
  PetscCall(PetscBLASIntCast(ds->ld,&ld));
  PetscCall(PetscBLASIntCast(ds->l,&l));
  n1 = n-l;
  PetscCall(MatDenseGetArray(ds->omat[DS_MAT_U],&U));
  PetscCall(DSGetArrayReal(ds,DS_MAT_D,&Omega));
  if (ds->compact) {
    /* Update U: U<-U*Omega   TODO: should this go here or somewhere else? */
    for (i=l;i<n;i++) PetscCallBLAS("BLASscal",BLASscal_(&n1,Omega+i,U+i*ld+l,&incx));
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
  PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_U],&U));
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_D,&Omega));
  PetscFunctionReturn(0);
}

PetscErrorCode DSTruncate_HSVD(DS ds,PetscInt n,PetscBool trim)
{
  PetscInt    i,ld=ds->ld,l=ds->l;
  PetscScalar *A;
  DS_HSVD     *ctx = (DS_HSVD*)ds->data;

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

static PetscErrorCode ArrowTrididiag(PetscBLASInt n,PetscReal *d,PetscReal *e,PetscScalar *Q,PetscBLASInt ld)
{
  PetscBLASInt i,j,j2,one=1;
  PetscReal    c,s,p,off,temp;

  PetscFunctionBegin;
  if (n<=2) PetscFunctionReturn(0);

  for (j=0;j<n-2;j++) {

    /* Eliminate entry e(j) by a rotation in the planes (j,j+1) */
    temp = e[j+1];
    PetscCallBLAS("LAPACKlartg",LAPACKREALlartg_(&temp,&e[j],&c,&s,&e[j+1]));
    s = -s;

    /* Apply rotation to diagonal elements */
    temp   = d[j+1];
    e[j]   = c*s*(temp-d[j]);
    d[j+1] = s*s*d[j] + c*c*temp;
    d[j]   = c*c*d[j] + s*s*temp;

    /* Apply rotation to Q */
    j2 = j+2;
    PetscCallBLAS("BLASrot",BLASMIXEDrot_(&j2,Q+j*ld,&one,Q+(j+1)*ld,&one,&c,&s));

    /* Chase newly introduced off-diagonal entry to the top left corner */
    for (i=j-1;i>=0;i--) {
      off  = -s*e[i];
      e[i] = c*e[i];
      temp = e[i+1];
      PetscCallBLAS("LAPACKlartg",LAPACKREALlartg_(&temp,&off,&c,&s,&e[i+1]));
      s = -s;
      temp = (d[i]-d[i+1])*s - 2.0*c*e[i];
      p = s*temp;
      d[i+1] += p;
      d[i] -= p;
      e[i] = -e[i] - c*temp;
      j2 = j+2;
      PetscCallBLAS("BLASrot",BLASMIXEDrot_(&j2,Q+i*ld,&one,Q+(i+1)*ld,&one,&c,&s));
    }
  }
  PetscFunctionReturn(0);
}


PetscErrorCode DSSolve_HSVD_CROSS(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  DS_HSVD        *ctx = (DS_HSVD*)ds->data;
  PetscInt       i,j,k=ds->k,rwu=0,iwu=0,swu=0,nv;
  PetscBLASInt   n1,n2,info,l=0,n=0,m=0,ld,off,size,one=1,*perm,*cmplx;
  PetscScalar    *U,*V,scal,*R;
  PetscReal      *d,*e,*dd,*ee,*Omega;

  PetscFunctionBegin;
  PetscCheck(ctx->m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSHSVDSetDimensions()");
  PetscCall(PetscBLASIntCast(ds->n,&n));
  PetscCall(PetscBLASIntCast(ctx->m,&m));
  PetscCheck(!(n-m),PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented for non-square matrices");
  PetscCall(PetscBLASIntCast(ds->l,&l));
  PetscCall(PetscBLASIntCast(ds->ld,&ld));
  PetscCall(PetscBLASIntCast(PetscMax(0,ds->k-ds->l+1),&n2));
  n1 = n-l;     /* n1 = size of leading block, excl. locked + size of trailing block */
  off = l+l*ld;
  PetscCall(MatDenseGetArrayWrite(ds->omat[DS_MAT_U],&U));
  PetscCall(MatDenseGetArrayWrite(ds->omat[DS_MAT_V],&V));
  PetscCall(DSGetArrayReal(ds,DS_MAT_T,&d));
  e = d+ld;
  PetscCall(DSGetArrayReal(ds,DS_MAT_D,&Omega));
  PetscCall(PetscArrayzero(U,ld*ld));
  for (i=0;i<l;i++) U[i+i*ld] = 1.0;
  PetscCall(PetscArrayzero(V,ld*ld));
  for (i=0;i<n;i++) V[i+i*ld] = 1.0;

  PetscCheck(ds->compact,PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented for non compact storage");

  /* Form the arrow tridiagonal cross product T=A'*Omega*A, where A is the arrow
     bidiagonal matrix formed by d, e. T is stored in dd, ee */
  PetscCall(DSAllocateWork_Private(ds,(n+6)*ld,4*ld,2*ld));
  R = ds->work+swu;
  swu += n*n;
  perm = ds->iwork + iwu;
  iwu += n;
  cmplx = ds->iwork+iwu;
  iwu += n;
  dd = ds->rwork+rwu;
  rwu += ld;
  ee = ds->rwork+rwu;
  rwu += ld;
  for (i=0;i<l;i++) {dd[i] = d[i]*d[i]*Omega[i]; ee[i] = 0.0;}
  for (i=l;i<=ds->k;i++) {
    dd[i] = Omega[i]*d[i]*d[i];
    ee[i] = Omega[i]*d[i]*e[i];
  }
  for (i=l;i<k;i++) dd[k] += Omega[i]*e[i]*e[i];
  for (i=k+1;i<n;i++) {
    dd[i] = Omega[i]*d[i]*d[i]+Omega[i-1]*e[i-1]*e[i-1];
    ee[i] = Omega[i]*d[i]*e[i];
  }

  /* Reduce T to tridiagonal form */
  PetscCall(ArrowTrididiag(n2,dd+l,ee+l,V+off,ld));

  /* Solve the tridiagonal eigenproblem corresponding to T */
  for (i=0;i<l;i++) wr[i] = d[i];
  PetscCallBLAS("LAPACKsteqr",LAPACKsteqr_("V",&n1,dd+l,ee+l,V+off,&ld,ds->rwork+rwu,&info));
  SlepcCheckLapackInfo("steqr",info);
  for (i=l;i<n;i++) wr[i] = PetscSqrtScalar(PetscAbs(dd[i]));
  if (wi) for (i=l;i<n;i++) wi[i] = 0.0;

  /* Build left singular vectors: U=A*V*Sigma^-1 */
  size = n1*ld;
  PetscCall(PetscArrayzero(U+l*ld,size));
  for (i=l;i<n-1;i++) {
    PetscCallBLAS("BLASaxpy",BLASaxpy_(&n1,d+i,V+l*ld+i,&ld,U+l*ld+i,&ld));
    j = (i<k)?k:i+1;
    PetscCallBLAS("BLASaxpy",BLASaxpy_(&n1,e+i,V+l*ld+j,&ld,U+l*ld+i,&ld));
  }
  PetscCallBLAS("BLASaxpy",BLASaxpy_(&n1,&d[n-1],V+off+(n1-1),&ld,U+off+(n1-1),&ld));
  /* Multiply by Sigma^-1 */
  for (i=l;i<n;i++) {scal = 1.0/wr[i]; PetscCallBLAS("BLASscal",BLASscal_(&n1,&scal,U+i*ld+l,&one));}

  /* Reinforce orthogonality */
  nv = n1;
  for (i=0;i<n;i++) cmplx[i] = 0;
  PetscCall(DSPseudoOrthog_HR(&nv,U+off,ld,Omega+l,R,ld,perm,cmplx,NULL,ds->work+swu));

  /* Update projected problem */
  for (i=l;i<n;i++) d[i] = wr[i];
  PetscCall(PetscArrayzero(e,n-1));

  /* Update vectors V with R */
  scal = -1.0;
  for (i=0;i<nv;i++) {
    if (R[i+i*ld] < 0.0) PetscCallBLAS("BLASscal",BLASscal_(&n1,&scal,V+(i+l)*ld+l,&one));
  }

  /* create diagonal matrix as a result */
  PetscCall(MatDenseRestoreArrayWrite(ds->omat[DS_MAT_U],&U));
  PetscCall(MatDenseRestoreArrayWrite(ds->omat[DS_MAT_V],&V));
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&d));
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_D,&Omega));
  PetscFunctionReturn(0);
}

#if !defined(PETSC_HAVE_MPIUNI)
PetscErrorCode DSSynchronize_HSVD(DS ds,PetscScalar eigr[],PetscScalar eigi[])
{
  PetscInt       ld=ds->ld,l=ds->l,k=0,kr=0;
  PetscMPIInt    n,rank,off=0,size,ldn,ld3,ld_;
  PetscScalar    *A,*B,*U,*V;
  PetscReal      *T,*D;

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
  PetscCall(PetscMPIIntCast(ld,&ld_));
  if (ds->compact) {
    PetscCall(DSGetArrayReal(ds,DS_MAT_T,&T));
    PetscCall(DSGetArrayReal(ds,DS_MAT_D,&D));
  } else {
    PetscCall(MatDenseGetArray(ds->omat[DS_MAT_A],&A));
    PetscCall(MatDenseGetArray(ds->omat[DS_MAT_B],&B));
  }
  if (ds->state>DS_STATE_RAW) {
    PetscCall(MatDenseGetArray(ds->omat[DS_MAT_U],&U));
    PetscCall(MatDenseGetArray(ds->omat[DS_MAT_V],&V));
  }
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ds),&rank));
  if (!rank) {
    if (ds->compact) {
      PetscCallMPI(MPI_Pack(T,ld3,MPIU_REAL,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
      PetscCallMPI(MPI_Pack(D,ld_,MPIU_REAL,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
    } else {
      PetscCallMPI(MPI_Pack(A+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
      PetscCallMPI(MPI_Pack(B+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
    }
    if (ds->state>DS_STATE_RAW) {
      PetscCallMPI(MPI_Pack(U+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
      PetscCallMPI(MPI_Pack(V+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
    }
    if (eigr) PetscCallMPI(MPI_Pack(eigr+l,n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds)));
  }
  PetscCallMPI(MPI_Bcast(ds->work,size,MPI_BYTE,0,PetscObjectComm((PetscObject)ds)));
  if (rank) {
    if (ds->compact) {
      PetscCallMPI(MPI_Unpack(ds->work,size,&off,T,ld3,MPIU_REAL,PetscObjectComm((PetscObject)ds)));
      PetscCallMPI(MPI_Unpack(ds->work,size,&off,D,ld_,MPIU_REAL,PetscObjectComm((PetscObject)ds)));
    } else {
      PetscCallMPI(MPI_Unpack(ds->work,size,&off,A+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
      PetscCallMPI(MPI_Unpack(ds->work,size,&off,B+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
    }
    if (ds->state>DS_STATE_RAW) {
      PetscCallMPI(MPI_Unpack(ds->work,size,&off,U+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
      PetscCallMPI(MPI_Unpack(ds->work,size,&off,V+l*ld,ldn,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
    }
    if (eigr) PetscCallMPI(MPI_Unpack(ds->work,size,&off,eigr+l,n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds)));
  }
  if (ds->compact) {
    PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
    PetscCall(DSRestoreArrayReal(ds,DS_MAT_D,&D));
  } else {
    PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_A],&A));
    PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_B],&B));
  }
  if (ds->state>DS_STATE_RAW) {
    PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_U],&U));
    PetscCall(MatDenseRestoreArray(ds->omat[DS_MAT_V],&V));
  }
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode DSMatGetSize_HSVD(DS ds,DSMatType t,PetscInt *rows,PetscInt *cols)
{
  DS_HSVD *ctx = (DS_HSVD*)ds->data;

  PetscFunctionBegin;
  PetscCheck(ctx->m,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"You should set the number of columns with DSHSVDSetDimensions()");
  switch (t) {
    case DS_MAT_A:
      *rows = ds->n;
      *cols = ds->extrarow? ctx->m+1: ctx->m;
      break;
    case DS_MAT_B:
      *rows = PetscMax(ctx->m,ds->n);
      *cols = PetscMax(ctx->m,ds->n);
      break;
    case DS_MAT_T:
      *rows = ds->n;
      *cols = PetscDefined(USE_COMPLEX)? 2: 3;
      break;
    case DS_MAT_D:
      *rows = PetscMax(ctx->m,ds->n);
      *cols = 1;
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

static PetscErrorCode DSHSVDSetDimensions_HSVD(DS ds,PetscInt m)
{
  DS_HSVD *ctx = (DS_HSVD*)ds->data;

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
   DSHSVDSetDimensions - Sets the number of columns for a DSHSVD.

   Logically Collective on ds

   Input Parameters:
+  ds - the direct solver context
-  m  - the number of columns

   Notes:
   This call is complementary to DSSetDimensions(), to provide a dimension
   that is specific to this DS type.

   Level: intermediate

.seealso: DSHSVDGetDimensions(), DSSetDimensions()
@*/
PetscErrorCode DSHSVDSetDimensions(DS ds,PetscInt m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,m,2);
  PetscTryMethod(ds,"DSHSVDSetDimensions_C",(DS,PetscInt),(ds,m));
  PetscFunctionReturn(0);
}

static PetscErrorCode DSHSVDGetDimensions_HSVD(DS ds,PetscInt *m)
{
  DS_HSVD *ctx = (DS_HSVD*)ds->data;

  PetscFunctionBegin;
  *m = ctx->m;
  PetscFunctionReturn(0);
}

/*@
   DSHSVDGetDimensions - Returns the number of columns for a DSHSVD.

   Not collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameters:
.  m - the number of columns

   Level: intermediate

.seealso: DSHSVDSetDimensions()
@*/
PetscErrorCode DSHSVDGetDimensions(DS ds,PetscInt *m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidIntPointer(m,2);
  PetscUseMethod(ds,"DSHSVDGetDimensions_C",(DS,PetscInt*),(ds,m));
  PetscFunctionReturn(0);
}

PetscErrorCode DSDestroy_HSVD(DS ds)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(ds->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ds,"DSHSVDSetDimensions_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ds,"DSHSVDGetDimensions_C",NULL));
  PetscFunctionReturn(0);
}

/*MC
   DSHSVD - Dense Singular Value Decomposition.

   Level: beginner

   Notes:
   The problem is expressed as A = U*Sigma*V', where A is rectangular in
   general, with n rows and m columns. U is orthogonal with respect to a
   signature matrix, stored in B. V is orthogonal. Sigma is a diagonal
   matrix whose diagonal elements are the arguments of DSSolve(). After
   solve, A is overwritten with Sigma, B is overwritten with the new signature.

   The matrices of left and right singular vectors, U and V, have size n and m,
   respectively. The number of columns m must be specified via DSHSVDSetDimensions().

   If the DS object is in the intermediate state, A is assumed to be in upper
   bidiagonal form (possibly with an arrow) and is stored in compact format
   on matrix T, and then the signature is stored in D. Otherwise, no particular
   structure is assumed. The compact storage is implemented for the square case
   only, m=n. The extra row should be interpreted in this case as an extra column.

   Used DS matrices:
+  DS_MAT_A - problem matrix
.  DS_MAT_B - second problem matrix, storing the signature
.  DS_MAT_T - upper bidiagonal matrix
-  DS_MAT_D - diagonal matrix (signature)

   Implemented methods:
.  0 - Cross

.seealso: DSCreate(), DSSetType(), DSType, DSHSVDSetDimensions()
M*/
SLEPC_EXTERN PetscErrorCode DSCreate_HSVD(DS ds)
{
  DS_HSVD         *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  ds->data = (void*)ctx;

  ds->ops->allocate      = DSAllocate_HSVD;
  ds->ops->view          = DSView_HSVD;
  ds->ops->vectors       = DSVectors_HSVD;
  ds->ops->solve[0]      = DSSolve_HSVD_CROSS;
  ds->ops->sort          = DSSort_HSVD;
#if !defined(PETSC_HAVE_MPIUNI)
  ds->ops->synchronize   = DSSynchronize_HSVD;
#endif
  ds->ops->truncate      = DSTruncate_HSVD;
  ds->ops->update        = DSUpdateExtraRow_HSVD;
  ds->ops->destroy       = DSDestroy_HSVD;
  ds->ops->matgetsize    = DSMatGetSize_HSVD;
  PetscCall(PetscObjectComposeFunction((PetscObject)ds,"DSHSVDSetDimensions_C",DSHSVDSetDimensions_HSVD));
  PetscCall(PetscObjectComposeFunction((PetscObject)ds,"DSHSVDGetDimensions_C",DSHSVDGetDimensions_HSVD));
  PetscFunctionReturn(0);
}
