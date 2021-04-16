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

typedef struct {
  PetscInt m;              /* number of columns */
  PetscInt p;              /* number of rows */
} DS_GSVD;

PetscErrorCode DSAllocate_GSVD(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_B);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_X);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_U);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_VT);CHKERRQ(ierr);
  ierr = DSAllocateMatReal_Private(ds,DS_MAT_T);CHKERRQ(ierr);
  ierr = DSAllocateMatReal_Private(ds,DS_MAT_D);CHKERRQ(ierr);
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSView_GSVD(DS ds,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  DS_GSVD           *ctx = (DS_GSVD*)ds->data;
  PetscViewerFormat format;
  PetscInt          i,j,c,r,n=ds->n,m=ctx->m,p=ctx->p;
  PetscReal         value;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO) PetscFunctionReturn(0);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer,"number of columns: %D\n",m);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"number of rows of B: %D\n",p);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (ds->compact) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",n,n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",2*ds->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
      for (i=0;i<PetscMin(ds->n,n);i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,(double)*(ds->rmat[DS_MAT_T]+i));CHKERRQ(ierr);
      }
      for (i=0;i<PetscMin(ds->n,n)-1;i++) {
        r = PetscMax(i+2,ds->k+1);
        c = i+1;
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,(double)*(ds->rmat[DS_MAT_T]+ds->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",DSMatName[DS_MAT_T]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
      for (i=0;i<PetscMin(ds->n,n);i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,(double)*(ds->rmat[DS_MAT_D]+i));CHKERRQ(ierr);
      }
      for (i=0;i<PetscMin(ds->n,n)-1;i++) {
        r = PetscMax(i+2,ds->k+1);
        c = i+1;
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,(double)*(ds->rmat[DS_MAT_T]+2*ds->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",DSMatName[DS_MAT_D]);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"Matrix %s =\n",DSMatName[DS_MAT_T]);CHKERRQ(ierr);
      for (i=0;i<ds->n;i++) {
        for (j=0;j<n;j++) {
          if (i==j) value = *(ds->rmat[DS_MAT_T]+i);
          else if (i<ds->k && j==ds->k) value = *(ds->rmat[DS_MAT_T]+ds->ld+PetscMin(i,j));
          else if (i==j+1 && i>ds->k) value = *(ds->rmat[DS_MAT_T]+ds->ld+i-1);
          else value = 0.0;
          ierr = PetscViewerASCIIPrintf(viewer," %18.16e ",(double)value);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"Matrix %s =\n",DSMatName[DS_MAT_D]);CHKERRQ(ierr);
      for (i=0;i<ds->n;i++) {
        for (j=0;j<n;j++) {
          if (i==j) value = *(ds->rmat[DS_MAT_D]+i);
          else if (i<ds->k && j==ds->k) value = *(ds->rmat[DS_MAT_T]+2*ds->ld+PetscMin(i,j));
          else if (i==j+1 && i>ds->k) value = *(ds->rmat[DS_MAT_T]+2*ds->ld+i-1);
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
    ierr = DSViewMat(ds,viewer,DS_MAT_B);CHKERRQ(ierr);
  }
  if (ds->state>DS_STATE_INTERMEDIATE) {
    ierr = DSViewMat(ds,viewer,DS_MAT_X);CHKERRQ(ierr);
    ierr = DSViewMat(ds,viewer,DS_MAT_U);CHKERRQ(ierr);
    ierr = DSViewMat(ds,viewer,DS_MAT_VT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSVectors_GSVD(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_U:
    case DS_MAT_VT:
      if (rnorm) *rnorm = 0.0;
      break;
    case DS_MAT_X:
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSort_GSVD(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscErrorCode ierr;
  DS_GSVD        *ctx = (DS_GSVD*)ds->data;
  PetscInt       t,l,ld=ds->ld,i,*perm,*perm2;
  PetscReal      *T,*D,*eig;
  PetscScalar    *A,*B;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  l = ds->l;
  t = ds->t;
  perm = ds->perm;
  ierr = PetscMalloc2(t,&eig,t,&perm2);CHKERRQ(ierr);
  if (ds->compact) {
    T = ds->rmat[DS_MAT_T];
    D = ds->rmat[DS_MAT_D];
    for (i=0;i<t;i++) eig[i] = (D[i]==0)?PETSC_INFINITY:T[i]/D[i];
  } else {
    A = ds->mat[DS_MAT_A];
    B = ds->mat[DS_MAT_B];
    for (i=0;i<t;i++) eig[i] = (B[i+i*ld]==0)?PETSC_INFINITY:PetscRealPart(A[i+i*ld])/PetscRealPart(B[i*(1+ld)]);
  }
  ierr = DSSortEigenvaluesReal_Private(ds,eig,perm);CHKERRQ(ierr);
  ierr = PetscArraycpy(perm2,perm,t);CHKERRQ(ierr);
  for (i=l;i<t;i++) wr[i] = eig[perm[i]];
  if (ds->compact) {
    ierr = PetscArraycpy(eig,T,t);CHKERRQ(ierr);
    for (i=l;i<t;i++) T[i] = eig[perm[i]];
    ierr = PetscArraycpy(eig,D,t);CHKERRQ(ierr);
    for (i=l;i<t;i++) D[i] = eig[perm[i]];
  } else {
    for (i=l;i<t;i++) eig[i] = PetscRealPart(A[i*(1+ld)]);
    for (i=l;i<t;i++) A[i*(1+ld)] = eig[perm[i]];
    for (i=l;i<t;i++) eig[i] =  PetscRealPart(B[i*(1+ld)]);
    for (i=l;i<t;i++) B[i*(1+ld)] = eig[perm[i]];
  }
  ierr = DSPermuteColumns_Private(ds,l,t,ds->n,DS_MAT_U,perm2);CHKERRQ(ierr);
  ierr = PetscArraycpy(perm2,perm,t);CHKERRQ(ierr);
  ierr = DSPermuteColumns_Private(ds,l,t,ctx->m,DS_MAT_X,perm2);CHKERRQ(ierr);
  ierr = DSPermuteColumns_Private(ds,l,t,ctx->p,DS_MAT_VT,perm);CHKERRQ(ierr);
  ierr = PetscFree2(eig,perm2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSSwitchFormat_GSVD(DS ds)
{
  PetscErrorCode ierr;
  PetscReal      *T   = ds->rmat[DS_MAT_T];
  PetscReal      *D   = ds->rmat[DS_MAT_D];
  PetscScalar    *A   = ds->mat[DS_MAT_A];
  PetscScalar    *B   = ds->mat[DS_MAT_B];
  PetscInt       i,n=ds->n,k=ds->k,ld=ds->ld;

  PetscFunctionBegin;
  /* switch from compact (arrow) to dense storage */
  /* bidiagonal associated to B is stored in D and T+2*ld */
  ierr = PetscArrayzero(A,ld*ld);CHKERRQ(ierr);
  ierr = PetscArrayzero(B,ld*ld);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    A[i+i*ld] = T[i];
    A[i+k*ld] = T[i+ld];
    B[i+i*ld] = D[i];
    B[i+k*ld] = T[i+2*ld];
  }
  A[k+k*ld] = T[k];
  B[k+k*ld] = D[k];
  for (i=k+1;i<n;i++) {
    A[i+i*ld]   = T[i];
    A[i-1+i*ld] = T[i-1+ld];
    B[i+i*ld]   = D[i];
    B[i-1+i*ld] = T[i-1+2*ld];
  }
  PetscFunctionReturn(0);
}

/*
  Compact format is used when [A;B] has orthonormal columns.
  In this case R=I and the GSVD of (A,B) is the CS decomposition
*/

PetscErrorCode DSSolve_GSVD(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  DS_GSVD        *ctx = (DS_GSVD*)ds->data;
  PetscInt       i,j;
  PetscBLASInt   n1,m1,info,lc = 0,n = 0,m = 0,p,p1,l,k,q,ld,off,lwork,r;
  PetscScalar    *A,*B,*X,*U,*VT,sone=1.0,smone=-1.0;
  PetscReal      *alpha,*beta,*T,*D;
#if !defined(SLEPC_MISSING_LAPACK_GGSVD3)
  PetscScalar    a,dummy;
  PetscReal      rdummy;
  PetscBLASInt   idummy;
#endif

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ctx->m,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ctx->p,&p);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->l,&lc);CHKERRQ(ierr);
  if (!ds->compact) {
    if (lc!=0) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"DSGSVD with non-compact format does not support locking");
  } else if (m!=n) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"The case m!=n not supported in compact format");
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  n1 = n-lc;     /* n1 = size of leading block, excl. locked + size of trailing block */
  m1 = m-lc;
  p1 = p-lc;
  off = lc+lc*ld;
  A  = ds->mat[DS_MAT_A];
  B  = ds->mat[DS_MAT_B];
  X  = ds->mat[DS_MAT_X];
  U  = ds->mat[DS_MAT_U];
  VT = ds->mat[DS_MAT_VT];
  ierr = PetscArrayzero(X,ld*ld);CHKERRQ(ierr);
  for (i=0;i<lc;i++) X[i+i*ld] = 1.0;
  ierr = PetscArrayzero(U,ld*ld);CHKERRQ(ierr);
  for (i=0;i<lc;i++) U[i+i*ld] = 1.0;
  ierr = PetscArrayzero(VT,ld*ld);CHKERRQ(ierr);
  for (i=0;i<lc;i++) VT[i+i*ld] = 1.0;
  if (ds->compact) ierr = DSSwitchFormat_GSVD(ds);CHKERRQ(ierr);
  T  = ds->rmat[DS_MAT_T];
  D  = ds->rmat[DS_MAT_D];

#if !defined(SLEPC_MISSING_LAPACK_GGSVD3)
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  /* workspace query and memory allocation */
  lwork = -1;
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKggsvd3",LAPACKggsvd3_("U","V","Q",&m1,&n1,&p1,&k,&l,&dummy,&ld,&dummy,&ld,&rdummy,&rdummy,&dummy,&ld,&dummy,&ld,&dummy,&ld,&a,&lwork,&idummy,&info));
  ierr = PetscBLASIntCast((PetscInt)a,&lwork);CHKERRQ(ierr);
#else
  PetscStackCallBLAS("LAPACKggsvd3",LAPACKggsvd3_("U","V","Q",&m1,&n1,&p1,&k,&l,&dummy,&ld,&dummy,&ld,&rdummy,&rdummy,&dummy,&ld,&dummy,&ld,&dummy,&ld,&a,&lwork,&rdummy,&idummy,&info));
  ierr = PetscBLASIntCast((PetscInt)PetscRealPart(a),&lwork);CHKERRQ(ierr);
#endif

#if !defined (PETSC_USE_COMPLEX)
  ierr = DSAllocateWork_Private(ds,lwork,2*ds->ld,ds->ld);CHKERRQ(ierr);
  alpha = ds->rwork;
  beta  = ds->rwork+ds->ld;
  PetscStackCallBLAS("LAPACKggsvd3",LAPACKggsvd3_("U","V","Q",&m1,&n1,&p1,&k,&l,A+off,&ld,B+off,&ld,alpha,beta,U+off,&ld,VT+off,&ld,X+off,&ld,ds->work,&lwork,ds->iwork,&info));
#else
  ierr = DSAllocateWork_Private(ds,lwork,4*ds->ld,ds->ld);CHKERRQ(ierr);
  alpha = ds->rwork+2*ds->ld;
  beta  = ds->rwork+3*ds->ld;
  PetscStackCallBLAS("LAPACKggsvd3",LAPACKggsvd3_("U","V","Q",&m1,&n1,&p1,&k,&l,A+off,&ld,B+off,&ld,alpha,beta,U+off,&ld,VT+off,&ld,X+off,&ld,ds->work,&lwork,ds->rwork,ds->iwork,&info));
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
  PetscStackCallBLAS("LAPACKggsvd",LAPACKggsvd_("U","V","Q",&m1,&n1,&p1,&k,&l,A+off,&ld,B+off,&ld,alpha,beta,U+off,&ld,VT+off,&ld,X+off,&ld,ds->work,ds->iwork,&info));
#else
  ierr = DSAllocateWork_Private(ds,lwork,4*ds->ld,ds->ld);CHKERRQ(ierr);
  alpha = ds->rwork+2*ds->ld;
  beta  = ds->rwork+3*ds->ld;
  PetscStackCallBLAS("LAPACKggsvd",LAPACKggsvd_("U","V","Q",&m1,&n1,&p1,&k,&l,A+off,&ld,B+off,&ld,alpha,beta,U+off,&ld,VT+off,&ld,X+off,&ld,ds->work,ds->rwork,ds->iwork,&info));
#endif
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  SlepcCheckLapackInfo("ggsvd",info);

#endif

  if (k+l<n1) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"The case rank deficient not supported yet");
  if (ds->compact) {
    /* R is the identity matrix (excep the sign) */
    for (i=lc;i<n;i++) {
      if (PetscRealPart(A[i+i*ld])<0.0) { /* scale column i */
        for (j=lc;j<n;j++) X[j+i*ld] = -X[j+i*ld];
      }
    }
    ierr = PetscArrayzero(T+lc,3*ld-lc);CHKERRQ(ierr);
    ierr = PetscArrayzero(D+lc,ld-lc);CHKERRQ(ierr);
    for (i=lc;i<n;i++) {
      T[i] = alpha[i-lc];
      D[i] = beta[i-lc];
      if (D[i]==0.0) wr[i] = PETSC_INFINITY;
      else wr[i] = T[i]/D[i];
    }
    ds->t = n;
  } else {
    /* X = X*inv(R) */
    q = PetscMin(m,n);
    PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&n,&q,&sone,A,&ld,X,&ld));
    if (m<n) {
      r = n-m;
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&r,&m,&sone,X,&ld,A,&ld,&smone,X+m*ld,&ld));
      PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&n,&r,&sone,B+m*ld,&ld,X+m*ld,&ld));
    }
    if (k>0) {
      for (i=k;i<PetscMin(m,k+l);i++) {
        ierr = PetscArraycpy(X+(i-k)*ld,X+i*ld,ld);CHKERRQ(ierr);
        ierr = PetscArraycpy(U+(i-k)*ld,U+i*ld,ld);CHKERRQ(ierr);
      }
    }
    /* singular values */
    ierr = PetscArrayzero(A,ld*ld);CHKERRQ(ierr);
    ierr = PetscArrayzero(B,ld*ld);CHKERRQ(ierr);
    for (j=k;j<PetscMin(m,k+l);j++) {
      A[(j-k)*(1+ld)] = alpha[j];
      B[(j-k)*(1+ld)] = beta[j];
      wr[j-k] = alpha[j]/beta[j];
    }
    ds->t = PetscMin(m,k+l)-k; /* set numver of computed values */
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSynchronize_GSVD(DS ds,PetscScalar eigr[],PetscScalar eigi[])
{
  PetscErrorCode ierr;
  DS_GSVD        *ctx = (DS_GSVD*)ds->data;
  PetscInt       ld=ds->ld,l=ds->l,k=0,kr=0;
  PetscMPIInt    m=ctx->m,rank,off=0,size,n,ldn,ld3;

  PetscFunctionBegin;
  if (ds->compact) kr = 3*ld;
  else k = 2*(m-l)*ld;
  if (ds->state>DS_STATE_RAW) k += 3*(m-l)*ld;
  if (eigr) k += m-l;
  ierr = DSAllocateWork_Private(ds,k+kr,0,0);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(k*sizeof(PetscScalar)+kr*sizeof(PetscReal),&size);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(m-l,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(ld*(m-l),&ldn);CHKERRQ(ierr);
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
      ierr = MPI_Pack(ds->mat[DS_MAT_X]+l*ld,ldn,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
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
  DS_GSVD        *ctx = (DS_GSVD*)ds->data;

  PetscFunctionBegin;
  switch (t) {
    case DS_MAT_A:
      *rows = ds->n;
      *cols = ctx->m;
      break;
    case DS_MAT_B:
      *rows = ctx->p;
      *cols = ctx->m;
      break;
    case DS_MAT_U:
      *rows = ds->n;
      *cols = ds->n;
      break;
    case DS_MAT_VT:
      *rows = ctx->p;
      *cols = ctx->p;
      break;
    case DS_MAT_X:
      *rows = ctx->m;
      *cols = ctx->m;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid t parameter");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DSGSVDSetDimensions_GSVD(DS ds,PetscInt m,PetscInt p)
{
  DS_GSVD *ctx = (DS_GSVD*)ds->data;

  PetscFunctionBegin;
  DSCheckAlloc(ds,1);
  if (m==PETSC_DECIDE || m==PETSC_DEFAULT) {
    ctx->m = ds->ld;
  } else {
    if (m<1 || m>ds->ld) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of m. Must be between 1 and ld");
    ctx->m = m;
  }
  if (p==PETSC_DECIDE || p==PETSC_DEFAULT) {
    ctx->p = ds->n;
  } else {
    if (p<1 || p>ds->ld) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of p. Must be between 1 and ld");
    ctx->p = p;
  }  
  PetscFunctionReturn(0);
}

/*@
   DSGSVDSetDimensions - Sets the number of columns and rows for a DSGSVD.

   Logically Collective on ds

   Input Parameters:
+  ds - the direct solver context
.  m  - the number of columns
-  p  - the number of rows for the second matrix (B)

   Notes:
   This call is complementary to DSSetDimensions(), to provide two dimensions
   that are specific to this DS type. The number of rows for the first matrix (A)
   is set by DSSetDimensions().

   Level: intermediate

.seealso: DSGSVDGetDimensions(), DSSetDimensions()
@*/
PetscErrorCode DSGSVDSetDimensions(DS ds,PetscInt m,PetscInt p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,m,2);
  ierr = PetscTryMethod(ds,"DSGSVDSetDimensions_C",(DS,PetscInt,PetscInt),(ds,m,p));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSGSVDGetDimensions_GSVD(DS ds,PetscInt *m,PetscInt *p)
{
  DS_GSVD *ctx = (DS_GSVD*)ds->data;

  PetscFunctionBegin;
  *m = ctx->m;
  *p = ctx->p;
  PetscFunctionReturn(0);
}

/*@
   DSGSVDGetDimensions - Returns the number of columns and rows for a DSGSVD.

   Not collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameters:
+  m - the number of columns
-  p - the number of rows for the second matrix (B)

   Level: intermediate

.seealso: DSGSVDSetDimensions()
@*/
PetscErrorCode DSGSVDGetDimensions(DS ds,PetscInt *m,PetscInt *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidIntPointer(m,2);
  PetscValidIntPointer(p,3);
  ierr = PetscUseMethod(ds,"DSGSVDGetDimensions_C",(DS,PetscInt*,PetscInt*),(ds,m,p));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSDestroy_GSVD(DS ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ds->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSGSVDSetDimensions_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSGSVDGetDimensions_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode DSCreate_GSVD(DS ds)
{
  DS_GSVD        *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ds,&ctx);CHKERRQ(ierr);
  ds->data = (void*)ctx;

  ds->ops->allocate      = DSAllocate_GSVD;
  ds->ops->view          = DSView_GSVD;
  ds->ops->vectors       = DSVectors_GSVD;
  ds->ops->sort          = DSSort_GSVD;
  ds->ops->solve[0]      = DSSolve_GSVD;
  ds->ops->synchronize   = DSSynchronize_GSVD;
  ds->ops->matgetsize    = DSMatGetSize_GSVD;
  ds->ops->destroy       = DSDestroy_GSVD;
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSGSVDSetDimensions_C",DSGSVDSetDimensions_GSVD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSGSVDGetDimensions_C",DSGSVDGetDimensions_GSVD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

