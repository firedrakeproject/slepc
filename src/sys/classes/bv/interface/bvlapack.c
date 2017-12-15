/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2017, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV private kernels that use the LAPACK
*/

#include <slepc/private/bvimpl.h>
#include <slepcblaslapack.h>

/*
    Compute ||A|| for an mxn matrix
*/
PetscErrorCode BVNorm_LAPACK_Private(BV bv,PetscInt m_,PetscInt n_,const PetscScalar *A,NormType type,PetscReal *nrm,PetscBool mpi)
{
  PetscErrorCode ierr;
  PetscBLASInt   m,n,i,j;
  PetscMPIInt    len;
  PetscReal      lnrm,*rwork=NULL,*rwork2=NULL;

  PetscFunctionBegin;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m_,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  if (type==NORM_FROBENIUS || type==NORM_2) {
    lnrm = LAPACKlange_("F",&m,&n,(PetscScalar*)A,&m,rwork);
    if (mpi) {
      lnrm = lnrm*lnrm;
      ierr = MPI_Allreduce(&lnrm,nrm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
      *nrm = PetscSqrtReal(*nrm);
    } else *nrm = lnrm;
    ierr = PetscLogFlops(2.0*m*n);CHKERRQ(ierr);
  } else if (type==NORM_1) {
    if (mpi) {
      ierr = BVAllocateWork_Private(bv,2*n_);CHKERRQ(ierr);
      rwork = (PetscReal*)bv->work;
      rwork2 = rwork+n_;
      ierr = PetscMemzero(rwork,n_*sizeof(PetscReal));CHKERRQ(ierr);
      ierr = PetscMemzero(rwork2,n_*sizeof(PetscReal));CHKERRQ(ierr);
      for (j=0;j<n_;j++) {
        for (i=0;i<m_;i++) {
          rwork[j] += PetscAbsScalar(A[i+j*m_]);
        }
      }
      ierr = PetscMPIIntCast(n_,&len);CHKERRQ(ierr);
      ierr = MPI_Allreduce(rwork,rwork2,len,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
      *nrm = 0.0;
      for (j=0;j<n_;j++) if (rwork2[j] > *nrm) *nrm = rwork2[j];
    } else {
      *nrm = LAPACKlange_("O",&m,&n,(PetscScalar*)A,&m,rwork);
    }
    ierr = PetscLogFlops(1.0*m*n);CHKERRQ(ierr);
  } else if (type==NORM_INFINITY) {
    ierr = BVAllocateWork_Private(bv,m_);CHKERRQ(ierr);
    rwork = (PetscReal*)bv->work;
    lnrm = LAPACKlange_("I",&m,&n,(PetscScalar*)A,&m,rwork);
    if (mpi) {
      ierr = MPI_Allreduce(&lnrm,nrm,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
    } else *nrm = lnrm;
    ierr = PetscLogFlops(1.0*m*n);CHKERRQ(ierr);
  }
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Compute the upper Cholesky factor in R and its inverse in S.
   If S == R then the inverse overwrites the Cholesky factor.
 */
PetscErrorCode BVMatCholInv_LAPACK_Private(BV bv,Mat R,Mat S)
{
#if defined(PETSC_MISSING_LAPACK_POTRF) || defined(SLEPC_MISSING_LAPACK_TRTRI)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRF/TRTRI - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,k,l,n,m,ld,lds;
  PetscScalar    *pR,*pS;
  PetscBLASInt   info,n_,l_,m_,ld_,lds_;

  PetscFunctionBegin;
  l = bv->l;
  k = bv->k;
  ierr = MatGetSize(R,&m,NULL);CHKERRQ(ierr);
  n = k-l;
  ierr = PetscBLASIntCast(m,&m_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(l,&l_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ld  = m;
  ld_ = m_;
  ierr = MatDenseGetArray(R,&pR);CHKERRQ(ierr);

  if (S==R) {
    ierr = BVAllocateWork_Private(bv,m*k);CHKERRQ(ierr);
    pS = bv->work;
    lds = ld;
    lds_ = ld_;
  } else {
    ierr = MatDenseGetArray(S,&pS);CHKERRQ(ierr);
    ierr = MatGetSize(S,&lds,NULL);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  }

  /* save a copy of matrix in S */
  for (i=l;i<k;i++) {
    ierr = PetscMemcpy(pS+i*lds+l,pR+i*ld+l,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  /* compute upper Cholesky factor in R */
  PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("U",&n_,pR+l*ld+l,&ld_,&info));
  ierr = PetscLogFlops((1.0*n*n*n)/3.0);CHKERRQ(ierr);

  if (info) {  /* LAPACKpotrf failed, retry on diagonally perturbed matrix */
    for (i=l;i<k;i++) {
      ierr = PetscMemcpy(pR+i*ld+l,pS+i*lds+l,n*sizeof(PetscScalar));CHKERRQ(ierr);
      pR[i+i*ld] += 50.0*PETSC_MACHINE_EPSILON;
    }
    PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("U",&n_,pR+l*ld+l,&ld_,&info));
    SlepcCheckLapackInfo("potrf",info);
    ierr = PetscLogFlops((1.0*n*n*n)/3.0);CHKERRQ(ierr);
  }

  /* compute S = inv(R) */
  if (S==R) {
    PetscStackCallBLAS("LAPACKtrtri",LAPACKtrtri_("U","N",&n_,pR+l*ld+l,&ld_,&info));
  } else {
    ierr = PetscMemzero(pS+l*lds,(k-l)*k*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=l;i<k;i++) {
      ierr = PetscMemcpy(pS+i*lds+l,pR+i*ld+l,n*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    PetscStackCallBLAS("LAPACKtrtri",LAPACKtrtri_("U","N",&n_,pS+l*lds+l,&lds_,&info));
  }
  SlepcCheckLapackInfo("trtri",info);
  ierr = PetscLogFlops(1.0*n*n*n);CHKERRQ(ierr);

  /* Zero out entries below the diagonal */
  for (i=l;i<k-1;i++) {
    ierr = PetscMemzero(pR+i*ld+i+1,(k-i-1)*sizeof(PetscScalar));CHKERRQ(ierr);
    if (S!=R) { ierr = PetscMemzero(pS+i*lds+i+1,(k-i-1)*sizeof(PetscScalar));CHKERRQ(ierr); }
  }
  ierr = MatDenseRestoreArray(R,&pR);CHKERRQ(ierr);
  if (S!=R) { ierr = MatDenseRestoreArray(S,&pS);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
#endif
}

/*
    QR factorization of an mxn matrix via parallel TSQR
*/
PetscErrorCode BVOrthogonalize_LAPACK_Private(BV bv,PetscInt m_,PetscInt n_,PetscScalar *Q,PetscScalar *R,PetscInt ldr)
{
#if defined(PETSC_MISSING_LAPACK_GEQRF) || defined(PETSC_MISSING_LAPACK_ORGQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEQRF/ORGQR - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       level,plevel,nlevels,powtwo,lda;
  PetscBLASInt   m,n,i,j,k,l,s,nb,lwork,info;
  PetscScalar    *tau,*work,*A=NULL,*QQ=NULL,*C=NULL,one=1.0,zero=0.0;
  PetscMPIInt    rank,size,count,stride;
  PetscBool      mpi;
  MPI_Datatype   tmat;

  PetscFunctionBegin;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m_,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n_,&n);CHKERRQ(ierr);
  k = PetscMin(m,n);
  nb = 16;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)bv),&size);CHKERRQ(ierr);
  if (m<n) SETERRQ(PetscObjectComm((PetscObject)bv),1,"Not implemented yet for nlocal<ncolumns");
  nlevels = (PetscInt)PetscCeilReal(PetscLog2Real((PetscReal)size));
  powtwo = PetscPowInt(2,(PetscInt)PetscFloorReal(PetscLog2Real((PetscReal)size)));
  mpi = size>1? PETSC_TRUE: PETSC_FALSE;
  if (mpi) {
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)bv),&rank);CHKERRQ(ierr);
    ierr = BVAllocateWork_Private(bv,k+n*nb+2*n*n+2*n*n*(nlevels+1)+PetscMax(m*n,2*n*n));CHKERRQ(ierr);
  } else {
    ierr = BVAllocateWork_Private(bv,k+n*nb);CHKERRQ(ierr);
   }
  tau = bv->work;
  work = bv->work+k;
  ierr = PetscBLASIntCast(n*nb,&lwork);CHKERRQ(ierr);
  if (mpi) {
    lda = 2*n;
    A   = bv->work+k+n*nb;
    QQ  = bv->work+k+n*nb+n*lda;
    C   = bv->work+k+n*nb+n*lda+n*lda*(nlevels+1);
  }

  /* Compute QR */
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&m,&n,Q,&m,tau,work,&lwork,&info));
  SlepcCheckLapackInfo("geqrf",info);

  /* Extract R */
  if (R || mpi) {
    for (j=0;j<n;j++) {
      for (i=0;i<=j;i++) {
        if (mpi) A[i+j*lda] = Q[i+j*m];
        else R[i+j*ldr] = Q[i+j*m];
      }
      for (i=j+1;i<n;i++) {
        if (mpi) A[i+j*lda] = 0.0;
        else R[i+j*ldr] = 0.0;
      }
    }
  }

  /* Compute orthogonal matrix in Q */
  PetscStackCallBLAS("LAPACKungqr",LAPACKungqr_(&m,&n,&k,Q,&m,tau,work,&lwork,&info));
  SlepcCheckLapackInfo("ungqr",info);

  if (mpi) {

    ierr = PetscMPIIntCast(n,&count);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(lda,&stride);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(lda,&l);CHKERRQ(ierr);
    ierr = MPI_Type_vector(count,count,stride,MPIU_SCALAR,&tmat);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&tmat);CHKERRQ(ierr);

    for (level=nlevels;level>=1;level--) {

      plevel = PetscPowInt(2,level);
      s = plevel*PetscFloorReal(rank/(PetscReal)plevel)+(rank+PetscPowInt(2,level-1))%plevel;

      /* Stack triangular matrices */
      if (rank<s && s<size) {  /* send top part, receive bottom part */
        ierr = MPI_Sendrecv(A,1,tmat,s,111,A+n,1,tmat,s,111,PetscObjectComm((PetscObject)bv),MPI_STATUS_IGNORE);CHKERRQ(ierr);
      } else if (s<size) {  /* copy top to bottom, receive top part */
        ierr = MPI_Sendrecv(A,1,tmat,rank,111,A+n,1,tmat,rank,111,PetscObjectComm((PetscObject)bv),MPI_STATUS_IGNORE);CHKERRQ(ierr);
        ierr = MPI_Sendrecv(A+n,1,tmat,s,111,A,1,tmat,s,111,PetscObjectComm((PetscObject)bv),MPI_STATUS_IGNORE);CHKERRQ(ierr);
      }
      if (level<nlevels) {  /* for cases when size is not a power of 2 */
        if (s+PetscPowInt(2,level)>size-powtwo && s+PetscPowInt(2,level)<size) {  /* send bottom part */
          ierr = MPI_Send(A+n,1,tmat,s+PetscPowInt(2,level),111,PetscObjectComm((PetscObject)bv));CHKERRQ(ierr);
        } else if (s>=size) {  /* receive bottom part */
          ierr = MPI_Recv(A+n,1,tmat,s-PetscPowInt(2,level),111,PetscObjectComm((PetscObject)bv),MPI_STATUS_IGNORE);CHKERRQ(ierr);
        }
      }
      /* Compute QR and build orthogonal matrix */
      if (level<nlevels || (level==nlevels && s<size)) {
        PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&l,&n,A,&l,tau,work,&lwork,&info));
        SlepcCheckLapackInfo("geqrf",info);
        ierr = PetscMemcpy(QQ+(level-1)*n*lda,A,n*lda*sizeof(PetscScalar));CHKERRQ(ierr);
        PetscStackCallBLAS("LAPACKungqr",LAPACKungqr_(&l,&n,&n,QQ+(level-1)*n*lda,&l,tau,work,&lwork,&info));
        SlepcCheckLapackInfo("ungqr",info);
        for (j=0;j<n;j++) {
          for (i=j+1;i<n;i++) A[i+j*lda] = 0.0;
        }
      } else if (level==nlevels) {  /* only one triangular matrix, set Q=I */
        ierr = PetscMemzero(QQ+(level-1)*n*lda,n*lda*sizeof(PetscScalar));CHKERRQ(ierr);
        for (j=0;j<n;j++) QQ[j+j*lda+(level-1)*n*lda] = 1.0;
      }
    }

    /* Extract R */
    if (R) {
      for (j=0;j<n;j++) {
        for (i=0;i<=j;i++) R[i+j*ldr] = A[i+j*lda];
        for (i=j+1;i<n;i++) R[i+j*ldr] = 0.0;
      }
    }

    /* Accumulate orthogonal matrices */
    for (level=1;level<=nlevels;level++) {

      plevel = PetscPowInt(2,level);
      s = plevel*PetscFloorReal(rank/(PetscReal)plevel)+(rank+PetscPowInt(2,level-1))%plevel;

      if (rank<s) {  /* multiply top part */
        if (level<nlevels) {
          PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&n,&one,QQ+level*n*lda,&l,QQ+(level-1)*n*lda,&l,&zero,C,&l));
          ierr = PetscMemcpy(QQ+level*n*lda,C,n*lda*sizeof(PetscScalar));CHKERRQ(ierr);
        } else {
          PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&m,&n,&n,&one,Q,&m,QQ+(level-1)*n*lda,&l,&zero,C,&m));
          ierr = PetscMemcpy(Q,C,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
        }
      } else {  /* multiply bottom part */
        if (level<nlevels) {
          PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&n,&one,QQ+level*n*lda,&l,QQ+(level-1)*n*lda+n,&l,&zero,C,&l));
          ierr = PetscMemcpy(QQ+level*n*lda,C,n*lda*sizeof(PetscScalar));CHKERRQ(ierr);
        } else {
          PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&m,&n,&n,&one,Q,&m,QQ+(level-1)*n*lda+n,&l,&zero,C,&m));
          ierr = PetscMemcpy(Q,C,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
        }
      }
    }

    ierr = MPI_Type_free(&tmat);CHKERRQ(ierr);
  }

  ierr = PetscLogFlops(3.0*m*n*n);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

