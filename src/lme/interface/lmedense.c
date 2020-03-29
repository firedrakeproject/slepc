/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Routines for solving dense matrix equations, in some cases calling SLICOT
*/

#include <slepc/private/lmeimpl.h>     /*I "slepclme.h" I*/
#include <slepcblaslapack.h>

/*
   LMEDenseRankSVD - given a square matrix A, compute its SVD U*S*V', and determine the
   numerical rank. On exit, U contains U*S and A is overwritten with V'
*/
PetscErrorCode LMEDenseRankSVD(LME lme,PetscInt n,PetscScalar *A,PetscInt lda,PetscScalar *U,PetscInt ldu,PetscInt *rank)
{
  PetscErrorCode ierr;
  PetscInt       i,j,rk=0;
  PetscScalar    *work;
  PetscReal      tol,*sg,*rwork;
  PetscBLASInt   n_,lda_,ldu_,info,lw_;

  PetscFunctionBegin;
  ierr = PetscCalloc3(n,&sg,10*n,&work,5*n,&rwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lda,&lda_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldu,&ldu_);CHKERRQ(ierr);
  lw_ = 10*n_;
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","O",&n_,&n_,A,&lda_,sg,U,&ldu_,NULL,&n_,work,&lw_,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","O",&n_,&n_,A,&lda_,sg,U,&ldu_,NULL,&n_,work,&lw_,rwork,&info));
#endif
  SlepcCheckLapackInfo("gesvd",info);
  tol = 10*PETSC_MACHINE_EPSILON*n*sg[0];
  for (j=0;j<n;j++) {
    if (sg[j]>tol) {
      for (i=0;i<n;i++) U[i+j*n] *= sg[j];
      rk++;
    } else break;
  }
  *rank = rk;
  ierr = PetscFree3(sg,work,rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_INFO)
/*
   LyapunovResidual - compute the residual norm ||A*U'*U+U'*U*A'+r*r'||
*/
static PetscErrorCode LyapunovResidual(PetscInt m,PetscScalar *A,PetscInt lda,PetscScalar *r,PetscScalar *U,PetscInt ldu,PetscReal *res)
{
  PetscErrorCode ierr;
  PetscBLASInt   n,la,lu;
  PetscInt       i,j;
  PetscScalar    *M,*R,zero=0.0,done=1.0;

  PetscFunctionBegin;
  *res = 0;
  ierr = PetscBLASIntCast(lda,&la);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldu,&lu);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc2(m*m,&M,m*m,&R);CHKERRQ(ierr);

  /* R = r*r' */
  for (i=0;i<m;i++) {
    for (j=0;j<m;j++) R[i+j*m] = r[i]*r[j];
  }
  /* M = A*U' */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n,&n,&n,&done,A,&la,U,&lu,&zero,M,&n));
  /* R = R+M*U */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&done,M,&n,U,&lu,&done,R,&n));
  /* R = R+U'*M' */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","C",&n,&n,&n,&done,U,&lu,M,&n,&done,R,&n));

  *res = LAPACKlange_("F",&n,&n,R,&n,NULL);
  ierr = PetscFree2(M,R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(SLEPC_HAVE_SLICOT)
/*
   LyapunovChol_SLICOT - implementation used when SLICOT is available
*/
static PetscErrorCode LyapunovChol_SLICOT(PetscInt m,PetscScalar *H,PetscInt ldh,PetscScalar *r,PetscScalar *U,PetscInt ldu,PetscReal *res)
{
  PetscErrorCode ierr;
  PetscBLASInt   ilo=1,lwork,info,n,lu,ione=1;
  PetscInt       i,j;
  PetscReal      scal;
  PetscScalar    *Q,*W,*wr,*wi,*work;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ldu,&lu);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(6*m,&lwork);CHKERRQ(ierr);

  /* transpose W = H' */
  ierr = PetscMalloc5(m*m,&W,m*m,&Q,m,&wr,m,&wi,lwork,&work);CHKERRQ(ierr);
  for (j=0;j<m;j++) {
    for (i=0;i<m;i++) W[i+j*m] = H[j+i*ldh];
  }

  /* compute the real Schur form of W */
  PetscStackCallBLAS("LAPACKhseqr",LAPACKhseqr_("S","I",&n,&ilo,&n,W,&n,wr,wi,Q,&n,work,&lwork,&info));
  SlepcCheckLapackInfo("hseqr",info);
#if defined(PETSC_USE_DEBUG)
  for (i=0;i<m;i++) if (PetscRealPart(wr[i])>=0.0) SETERRQ(PETSC_COMM_SELF,1,"Eigenvalue with non-negative real part, the coefficient matrix is not stable");
#endif

  /* copy r into first row of U */
  for (j=0;j<m;j++) U[j*ldu] = r[j];

  /* solve Lyapunov equation (Hammarling) */
  PetscStackCallBLAS("SLICOTsb03od",SLICOTsb03od_("C","F","N",&n,&ione,W,&n,Q,&n,U,&lu,&scal,wr,wi,work,&lwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SLICOT subroutine SB03OD: info=%d",(int)info);
  if (scal!=1.0) SETERRQ1(PETSC_COMM_SELF,1,"Current implementation cannot handle scale factor %g",scal);

  /* resnorm = norm(H(m+1,:)*U'*U), use Q(:,1) = U'*U(:,m) */
  for (j=0;j<m;j++) Q[j] = U[j+(m-1)*ldu];
  PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","C","N",&n,U,&lu,Q,&ione));
  *res = H[m+(m-1)*ldh]*BLASnrm2_(&n,Q,&ione);

  ierr = PetscFree5(W,Q,wr,wi,work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else

/*
   Compute the upper Cholesky factor of A
 */
static PetscErrorCode CholeskyFactor(PetscInt m,PetscScalar *A,PetscInt lda)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *S;
  PetscBLASInt   info,n,ld;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lda,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc1(m*m,&S);CHKERRQ(ierr);

  /* save a copy of matrix in S */
  for (i=0;i<m;i++) {
    ierr = PetscArraycpy(S+i*m,A+i*lda,m);CHKERRQ(ierr);
  }

  /* compute upper Cholesky factor in R */
  PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("U",&n,A,&ld,&info));
  ierr = PetscLogFlops((1.0*n*n*n)/3.0);CHKERRQ(ierr);

  if (info) {
    ierr = PetscInfo(NULL,"potrf failed, retry on diagonally perturbed matrix\n");CHKERRQ(ierr);
    for (i=0;i<m;i++) {
      ierr = PetscArraycpy(A+i*lda,S+i*m,m);CHKERRQ(ierr);
      A[i+i*lda] += 50.0*PETSC_MACHINE_EPSILON;
    }
    PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("U",&n,A,&ld,&info));
    SlepcCheckLapackInfo("potrf",info);
    ierr = PetscLogFlops((1.0*n*n*n)/3.0);CHKERRQ(ierr);
  }

  /* Zero out entries below the diagonal */
  for (i=0;i<m-1;i++) {
    ierr = PetscArrayzero(A+i*lda+i+1,m-i-1);CHKERRQ(ierr);
  }
  ierr = PetscFree(S);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   LyapunovFact_LAPACK - alternative implementation when SLICOT is not available
*/
static PetscErrorCode LyapunovChol_LAPACK(PetscInt m,PetscScalar *H,PetscInt ldh,PetscScalar *r,PetscScalar *U,PetscInt ldu,PetscReal *res)
{
  PetscErrorCode ierr;
  PetscBLASInt   ilo=1,lwork,info,n,lu,ione=1;
  PetscInt       i,j;
  PetscReal      scal;
  PetscScalar    *Q,*C,*W,*z,*wr,*work,zero=0.0,done=1.0;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *wi;
#endif

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ldu,&lu);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(6*m,&lwork);CHKERRQ(ierr);
  C = U;

#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc6(m*m,&Q,m*m,&W,m,&z,m,&wr,m,&wi,lwork,&work);CHKERRQ(ierr);
#else
  ierr = PetscMalloc5(m*m,&Q,m*m,&W,m,&z,m,&wr,lwork,&work);CHKERRQ(ierr);
#endif

  /* save a copy W = H */
  for (j=0;j<m;j++) {
    for (i=0;i<m;i++) W[i+j*m] = H[i+j*ldh];
  }

  /* compute the (real) Schur form of W */
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKhseqr",LAPACKhseqr_("S","I",&n,&ilo,&n,W,&n,wr,wi,Q,&n,work,&lwork,&info));
#else
  PetscStackCallBLAS("LAPACKhseqr",LAPACKhseqr_("S","I",&n,&ilo,&n,W,&n,wr,Q,&n,work,&lwork,&info));
#endif
  SlepcCheckLapackInfo("hseqr",info);
#if defined(PETSC_USE_DEBUG)
  for (i=0;i<m;i++) if (PetscRealPart(wr[i])>=0.0) SETERRQ1(PETSC_COMM_SELF,1,"Eigenvalue with non-negative real part %g, the coefficient matrix is not stable",PetscRealPart(wr[i]));
#endif

  /* C = z*z', z = Q'*r */
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&n,&done,Q,&n,r,&ione,&zero,z,&ione));
  for (i=0;i<m;i++) {
    for (j=0;j<m;j++) C[i+j*ldu] = -z[i]*PetscConj(z[j]);
  }

  /* solve triangular Sylvester equation */
  PetscStackCallBLAS("LAPACKtrsyl",LAPACKtrsyl_("N","C",&ione,&n,&n,W,&n,W,&n,C,&lu,&scal,&info));
  SlepcCheckLapackInfo("trsyl",info);
  if (scal!=1.0) SETERRQ1(PETSC_COMM_SELF,1,"Current implementation cannot handle scale factor %g",scal);

  /* back-transform C = Q*C*Q' */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&done,Q,&n,C,&n,&zero,W,&n));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n,&n,&n,&done,W,&n,Q,&n,&zero,C,&lu));

  /* resnorm = norm(H(m+1,:)*Y) */
  *res = H[m+(m-1)*ldh]*BLASnrm2_(&n,C+m-1,&n);

  /* U = chol(C) */
  ierr = CholeskyFactor(m,C,ldu);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscFree6(Q,W,z,wr,wi,work);CHKERRQ(ierr);
#else
  ierr = PetscFree5(Q,W,z,wr,work);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#endif /* SLEPC_HAVE_SLICOT */

/*@C
   LMEDenseLyapunovChol - Computes the Cholesky factor of the solution of a
   dense Lyapunov equation with rank-1 right-hand side.

   Logically Collective on lme

   Input Parameters:
+  lme - linear matrix equation solver context
.  m   - problem size
.  H   - coefficient matrix
.  ldh - leading dimension of H
.  r   - right-hand side vector
-  ldu - leading dimension of U

   Output Parameter:
+  U   - Cholesky factor of the solution
-  res - residual norm

   Note:
   The Lyapunov equation has the form H*X + X*H' = -r*r', where H represents
   the leading mxm submatrix of argument H, and the solution X = U'*U.

   H is assumed to be in upper Hessenberg form, with dimensions (m+1)xm.
   The last row is used to compute the residual norm, assuming H and r come
   from the projection onto an Arnoldi basis.

   Level: developer

.seealso: LMESolve()
@*/
PetscErrorCode LMEDenseLyapunovChol(LME lme,PetscInt m,PetscScalar *H,PetscInt ldh,PetscScalar *r,PetscScalar *U,PetscInt ldu,PetscReal *res)
{
  PetscErrorCode ierr;
#if defined(PETSC_USE_INFO)
  PetscReal      error;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscValidLogicalCollectiveInt(lme,m,2);
  PetscValidPointer(H,3);
  PetscValidLogicalCollectiveInt(lme,ldh,4);
  PetscValidPointer(r,5);
  PetscValidPointer(U,6);
  PetscValidLogicalCollectiveInt(lme,ldu,7);

#if defined(SLEPC_HAVE_SLICOT)
  ierr = LyapunovChol_SLICOT(m,H,ldh,r,U,ldu,res);CHKERRQ(ierr);
#else
  ierr = LyapunovChol_LAPACK(m,H,ldh,r,U,ldu,res);CHKERRQ(ierr);
#endif

#if defined(PETSC_USE_INFO)
  if (PetscLogPrintInfo) {
    ierr = LyapunovResidual(m,H,ldh,r,U,ldu,&error);CHKERRQ(ierr);
    ierr = PetscInfo1(lme,"Residual norm of dense Lyapunov equation = %g\n",error);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

