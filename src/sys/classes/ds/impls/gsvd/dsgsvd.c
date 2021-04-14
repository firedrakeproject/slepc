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
  PetscViewerFormat format;
  PetscInt          i,j,c,r;
  PetscReal         value;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  if (ds->compact) {
    if (!ds->m) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"m was not set");
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",ds->n,ds->m);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",2*ds->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
      for (i=0;i<PetscMin(ds->n,ds->m);i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,(double)*(ds->rmat[DS_MAT_T]+i));CHKERRQ(ierr);
      }
      for (i=0;i<PetscMin(ds->n,ds->m)-1;i++) {
        r = PetscMax(i+2,ds->k+1);
        c = i+1;
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,(double)*(ds->rmat[DS_MAT_T]+ds->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",DSMatName[DS_MAT_T]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
      for (i=0;i<PetscMin(ds->n,ds->m);i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,(double)*(ds->rmat[DS_MAT_D]+i));CHKERRQ(ierr);
      }
      for (i=0;i<PetscMin(ds->n,ds->m)-1;i++) {
        r = PetscMax(i+2,ds->k+1);
        c = i+1;
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,(double)*(ds->rmat[DS_MAT_T]+2*ds->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",DSMatName[DS_MAT_D]);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"Matrix %s =\n",DSMatName[DS_MAT_T]);CHKERRQ(ierr);
      for (i=0;i<ds->n;i++) {
        for (j=0;j<ds->m;j++) {
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
        for (j=0;j<ds->m;j++) {
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
      X = ds->mat[DS_MAT_X];
      ierr = PetscArraycpy(X,ds->mat[DS_MAT_Q],ds->ld*ds->ld);CHKERRQ(ierr);
      if (!ds->compact) {
        /* X = Q*inv(R) */
        ierr = PetscBLASIntCast(ds->n,&m);CHKERRQ(ierr);
        ierr = PetscBLASIntCast(ds->m,&n);CHKERRQ(ierr);
        ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
        A = ds->mat[DS_MAT_A];
        B = ds->mat[DS_MAT_B];
        q = PetscMin(m,n);
        PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&n,&q,&sone,A,&ld,X,&ld));
        if (m<n) {
          r = n-m;
          PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&r,&m,&sone,X,&ld,A,&ld,&smone,X+m*ld,&ld));
          PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&n,&r,&sone,B+m*ld,&ld,X+m*ld,&ld));
        }
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
  PetscInt       n,l,i,*perm,*perm2;
  PetscReal      *T,*D,*eig;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  l = ds->l;
  n = PetscMin(ds->n,ds->m);
  perm = ds->perm;
  if (ds->compact) {
    ierr = PetscMalloc2(n,&eig,n,&perm2);CHKERRQ(ierr);
    T = ds->rmat[DS_MAT_T];
    D = ds->rmat[DS_MAT_D];
    for (i=0;i<n;i++) eig[i] = (D[i]==0)?PETSC_INFINITY:T[i]/D[i];
    ierr = DSSortEigenvaluesReal_Private(ds,eig,perm);CHKERRQ(ierr);
    ierr = PetscArraycpy(perm2,perm,n);CHKERRQ(ierr);
    for (i=l;i<n;i++) wr[i] = eig[perm[i]];
    ierr = PetscArraycpy(eig,T,n);CHKERRQ(ierr);
    for (i=l;i<n;i++) T[i] = eig[perm[i]];
    ierr = PetscArraycpy(eig,D,n);CHKERRQ(ierr);
    for (i=l;i<n;i++) D[i] = eig[perm[i]];
    ierr = DSPermuteColumns_Private(ds,l,n,DS_MAT_U,perm2);CHKERRQ(ierr);
    ierr = PetscArraycpy(perm2,perm,n);CHKERRQ(ierr);
    ierr = DSPermuteColumns_Private(ds,l,n,DS_MAT_Q,perm2);CHKERRQ(ierr);
    ierr = DSPermuteColumns_Private(ds,l,n,DS_MAT_VT,perm);CHKERRQ(ierr);
    ierr = PetscFree2(eig,perm2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DSSwitchFormat_GSVD(DS ds)
{
  PetscErrorCode ierr;
  PetscReal      *T = ds->rmat[DS_MAT_T];
  PetscReal      *D = ds->rmat[DS_MAT_D];
  PetscScalar    *A = ds->mat[DS_MAT_A];
  PetscScalar    *B = ds->mat[DS_MAT_B];
  PetscInt       i,m=ds->m,k=ds->k,ld=ds->ld;

  PetscFunctionBegin;
  if (!m) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"m was not set");
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
  for (i=k+1;i<m;i++) {
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
  PetscInt       i,j;
  PetscBLASInt   n1,m1,info,lc = 0,n = 0,m = 0,p,p1,l,k,ld,off,lwork;
  PetscScalar    *A,*B,*Q,*U,*VT;
  PetscReal      *alpha,*beta,*T,*D;
#if !defined(SLEPC_MISSING_LAPACK_GGSVD3)
  PetscScalar    a,dummy;
  PetscReal      rdummy;
  PetscBLASInt   idummy;
#endif

  PetscFunctionBegin;
  if (!ds->compact) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"The non-compact format is not supported yet in DSGSVD");
  ierr = PetscBLASIntCast(ds->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->m,&n);CHKERRQ(ierr);
  p = m;
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
  Q  = ds->mat[DS_MAT_Q];
  U  = ds->mat[DS_MAT_U];
  VT = ds->mat[DS_MAT_VT];
  ierr = PetscArrayzero(Q,ld*ld);CHKERRQ(ierr);
  for (i=0;i<lc;i++) Q[i+i*ld] = 1.0;
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
  PetscStackCallBLAS("LAPACKggsvd3",LAPACKggsvd3_("U","V","Q",&m1,&n1,&p1,&k,&l,A+off,&ld,B+off,&ld,alpha,beta,U+off,&ld,VT+off,&ld,Q+off,&ld,ds->work,&lwork,ds->iwork,&info));
#else
  ierr = DSAllocateWork_Private(ds,lwork,4*ds->ld,ds->ld);CHKERRQ(ierr);
  alpha = ds->rwork+2*ds->ld;
  beta  = ds->rwork+3*ds->ld;
  PetscStackCallBLAS("LAPACKggsvd3",LAPACKggsvd3_("U","V","Q",&m1,&n1,&p1,&k,&l,A+off,&ld,B+off,&ld,alpha,beta,U+off,&ld,VT+off,&ld,Q+off,&ld,ds->work,&lwork,ds->rwork,ds->iwork,&info));
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
  PetscStackCallBLAS("LAPACKggsvd",LAPACKggsvd_("U","V","Q",&m1,&n1,&p1,&k,&l,A+off,&ld,B+off,&ld,alpha,beta,U+off,&ld,VT+off,&ld,Q+off,&ld,ds->work,ds->iwork,&info));
#else
  ierr = DSAllocateWork_Private(ds,lwork,4*ds->ld,ds->ld);CHKERRQ(ierr);
  alpha = ds->rwork+2*ds->ld;
  beta  = ds->rwork+3*ds->ld;
  PetscStackCallBLAS("LAPACKggsvd",LAPACKggsvd_("U","V","Q",&m1,&n1,&p1,&k,&l,A+off,&ld,B+off,&ld,alpha,beta,U+off,&ld,VT+off,&ld,Q+off,&ld,ds->work,ds->rwork,ds->iwork,&info));
#endif
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  SlepcCheckLapackInfo("ggsvd",info);

#endif

  if (k+l<n1) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"The case rank deficient supported yet");
  /* form matrices (D1,D2) in (A,B) */
  if (!ds->compact) {
    ierr = PetscArrayzero(A,ld*ld);CHKERRQ(ierr);
    ierr = PetscArrayzero(B,ld*ld);CHKERRQ(ierr);
  } else {
    /* R is the identity matrix (excep the sign) */
    for (i=lc;i<n;i++) {
      if (PetscRealPart(A[i+i*ld])<0.0) { /* scale column i */
        for (j=lc;j<n;j++) Q[j+i*ld] = -Q[j+i*ld];
      }
    }
    ierr = PetscArrayzero(T+lc,3*ld-lc);CHKERRQ(ierr);
    ierr = PetscArrayzero(D+lc,ld-lc);CHKERRQ(ierr);
    for (i=lc;i<n;i++) {
      T[i] = alpha[i-lc];
      D[i] = beta[i-lc];
    }
  }
  for (i=0;i<lc;i++) {
    if (D[i]==0.0) wr[i] = PETSC_INFINITY;
    else wr[i] = T[i]/D[i];
  }
  for (i=0;i<k;i++) wr[i+lc] = PETSC_INFINITY;
  for (i=0;i<k+l;i++) wr[k+i+lc] = alpha[i]/beta[i];
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
  ds->ops->sort          = DSSort_GSVD;
  ds->ops->solve[0]      = DSSolve_GSVD;
  ds->ops->synchronize   = DSSynchronize_GSVD;
  ds->ops->matgetsize    = DSMatGetSize_GSVD;
  PetscFunctionReturn(0);
}

