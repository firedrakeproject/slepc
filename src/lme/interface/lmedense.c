/*
   Routines for solving dense matrix equations, in some cases calling SLICOT.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/lmeimpl.h>     /*I "slepclme.h" I*/
#include <slepcblaslapack.h>

#define CHECK
#if defined(CHECK)
#undef __FUNCT__
#define __FUNCT__ "LyapunovResidual"
/*
   LyapunovResidual - compute the residual norm ||H*L*L'+L*L'*H'+r*r'||
*/
static PetscErrorCode LyapunovResidual(PetscScalar *H,PetscInt m,PetscInt ldh,PetscScalar *r,PetscScalar *L,PetscInt ldl,PetscReal *res)
{
  PetscErrorCode ierr;
  PetscBLASInt   n,ld;
  PetscInt       i,j;
  PetscScalar    *M,*R,zero=0.0,done=1.0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ldh,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc2(m*m,&M,m*m,&R);CHKERRQ(ierr);

  /* R = r*r' */
  for (i=0;i<m;i++) {
    for (j=0;j<m;j++) R[i+j*m] = r[i]*r[j];
  }
  /* M = H*L */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&done,H,&ld,L,&n,&zero,M,&n));
  /* R = R+M*L' */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n,&n,&n,&done,M,&n,L,&n,&done,R,&n));
  /* R = R+L*M' */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n,&n,&n,&done,L,&n,M,&n,&done,R,&n));
  
  *res = LAPACKlange_("F",&n,&n,R,&m,NULL);
  ierr = PetscFree2(M,R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if 0
#undef __FUNCT__
#define __FUNCT__ "AbsEig"
/*
   AbsEig - given a matrix A that may be slightly indefinite (hence Cholesky fails)
   modify it by taking the absolute value of the eigenvalues: [U,S] = eig(A); A = U*abs(S)*U';
*/
static PetscErrorCode AbsEig(PetscScalar *A,PetscInt m)
{
#if defined(PETSC_MISSING_LAPACK_SYEV) || defined(SLEPC_MISSING_LAPACK_LACPY)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SYEV/LACPY - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBLASInt   n,ld,lwork,info;
  PetscScalar    *Q,*W,*work,a,one=1.0,zero=0.0;
  PetscReal      *eig,dummy;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork,rdummy;
#endif

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ld = n;

  /* workspace query and memory allocation */
  lwork = -1;
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","L",&n,A,&ld,&dummy,&a,&lwork,&rdummy,&info));
  ierr = PetscBLASIntCast((PetscInt)PetscRealPart(a),&lwork);CHKERRQ(ierr);
  ierr = PetscMalloc5(m,&eig,m*m,&Q,m*n,&W,lwork,&work,PetscMax(1,3*m-2),&rwork);CHKERRQ(ierr);
#else
  PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","L",&n,A,&ld,&dummy,&a,&lwork,&info));
  ierr = PetscBLASIntCast((PetscInt)PetscRealPart(a),&lwork);CHKERRQ(ierr);
  ierr = PetscMalloc4(m,&eig,m*m,&Q,m*n,&W,lwork,&work);CHKERRQ(ierr);
#endif

  /* compute eigendecomposition */
  PetscStackCallBLAS("LAPACKlacpy",LAPACKlacpy_("L",&n,&n,A,&ld,Q,&ld));
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","L",&n,Q,&ld,eig,work,&lwork,rwork,&info));
#else
  PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","L",&n,Q,&ld,eig,work,&lwork,&info));
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSYEV %i",info);

  /* W = f(Lambda)*Q' */
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) W[i+j*ld] = Q[j+i*ld]*PetscAbsScalar(eig[i]);
  }
  /* A = Q*W */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Q,&ld,W,&ld,&zero,A,&ld));
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree5(eig,Q,W,work,rwork);CHKERRQ(ierr);
#else
  ierr = PetscFree4(eig,Q,W,work);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
#endif
}
#endif

#undef __FUNCT__
#define __FUNCT__ "CholeskyFactor"
/*
   Compute the lower Cholesky factor of A
 */
static PetscErrorCode CholeskyFactor(PetscScalar *A,PetscInt m)
{
#if defined(PETSC_MISSING_LAPACK_POTRF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRF - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,ld;
  PetscScalar    *S;
  PetscBLASInt   info,n;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ld  = m;
  ierr = PetscMalloc1(m*m,&S);CHKERRQ(ierr);

  /* save a copy of matrix in S */
  for (i=0;i<m;i++) {
    ierr = PetscMemcpy(S+i*ld,A+i*ld,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  /* compute upper Cholesky factor in R */
  PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&n,A,&n,&info));
  ierr = PetscLogFlops((1.0*n*n*n)/3.0);CHKERRQ(ierr);

  if (info) {  /* LAPACKpotrf failed, retry on diagonally perturbed matrix */
    for (i=0;i<m;i++) {
      ierr = PetscMemcpy(A+i*ld,S+i*ld,n*sizeof(PetscScalar));CHKERRQ(ierr);
      A[i+i*ld] += 50.0*PETSC_MACHINE_EPSILON;
    }
    PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&n,A,&n,&info));
    if (info) SETERRQ1(PETSC_COMM_SELF,1,"Error in Cholesky factorization, info=%D",(PetscInt)info);
    ierr = PetscLogFlops((1.0*n*n*n)/3.0);CHKERRQ(ierr);
  }

  /* Zero out entries above the diagonal */
  for (i=1;i<m;i++) {
    ierr = PetscMemzero(A+i*ld,i*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscFree(S);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "LyapunovChol_LAPACK"
/*
   LyapunovFact_Private - alternative implementation when SLICOT is not available
*/
static PetscErrorCode LyapunovChol_LAPACK(PetscScalar *H,PetscInt m,PetscInt ldh,PetscScalar *r,PetscScalar *L,PetscInt ldl,PetscReal *res)
{
#if defined(PETSC_MISSING_LAPACK_HSEQR) || defined(SLEPC_MISSING_LAPACK_TRSYL)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"HSEQR/TRSYL - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscBLASInt   ilo=1,lwork,info,n,ld,ione=1;
  PetscInt       i,j;
  PetscReal      scal;
  PetscScalar    *Q,*C,*W,*z,*wr,*work,zero=0.0,done=1.0,alpha,beta;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *wi;
#endif

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ldh,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  lwork = n;
  C = L;

  /* compute the (real) Schur form of H */
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc6(m*m,&Q,m*m,&W,m,&z,m,&wr,m,&wi,m,&work);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKhseqr",LAPACKhseqr_("S","I",&n,&ilo,&n,H,&ld,wr,wi,Q,&n,work,&lwork,&info));
#else
  ierr = PetscMalloc5(m*m,&Q,m*m,&W,m,&z,m,&wr,m,&work);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKhseqr",LAPACKhseqr_("S","I",&n,&ilo,&n,H,&ld,wr,Q,&n,work,&lwork,&info));
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",info);
#if defined(PETSC_USE_DEBUG)
  for (i=0;i<m;i++) if (PetscRealPart(wr[i])>0.0) SETERRQ(PETSC_COMM_SELF,1,"Positive eigenvalue found, the coefficient matrix is not stable");
#endif

  /* C = z*z', z = Q'*r */
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&n,&done,Q,&n,r,&ione,&zero,z,&ione));
  for (i=0;i<m;i++) {
    for (j=0;j<m;j++) C[i+j*m] = -z[i]*z[j];
  }

  /* solve triangular Sylvester equation */
  PetscStackCallBLAS("LAPACKtrsyl",LAPACKtrsyl_("N","C",&ione,&n,&n,H,&ld,H,&ld,C,&n,&scal,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTRSYL %d",info);
  if (scal!=1.0) SETERRQ1(PETSC_COMM_SELF,1,"Current implementation cannot handle scale factor %g",scal);
  
  /* back-transform C = Q*C*Q' */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&done,Q,&n,C,&n,&zero,W,&n));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n,&n,&n,&done,W,&n,Q,&n,&zero,C,&n));

  /* resnorm = norm(H(m+1,:)*Y) */
  *res = 0.0;
  beta = H[m+(m-1)*ldh];
  for (j=0;j<m;j++) {
    alpha = beta*C[m-1+j*m];
    *res += alpha*alpha;
  }
  *res = PetscSqrtReal(*res);

#if 0
  /* avoid problems due to (small) negative eigenvalues */
  ierr = AbsEig(C,m);CHKERRQ(ierr);
#endif

  /* L = chol(C,'lower') */
  ierr = CholeskyFactor(C,m);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscFree6(Q,W,z,wr,wi,work);CHKERRQ(ierr);
#else
  ierr = PetscFree5(Q,W,z,wr,work);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "LMEDenseLyapunovChol"
/*@C
   LMEDenseLyapunovFact - Computes the Cholesky factor of the solution of a
   dense Lyapunov equation with rank-1 right-hand side.

   Logically Collective on LME

   Input Parameters:
+  lme - linear matrix equation solver context
.  H   - coefficient matrix
.  m   - problem size
.  ldh - leading dimension of H
.  r   - right-hand side vector
-  ldl - leading dimension of L

   Output Parameter:
+  L   - Cholesky factor of the solution
-  res - residual norm

   Note:
   The Lyapunov equation has the form H*X + X*H' = -r*r', where H represents
   the leading mxm submatrix of argument H, and the solution X = L*L'.

   H is assumed to be in upper Hessenberg form, with dimensions (m+1)xm.
   The last row is used to compute the residual norm, assuming H and r come
   from the projection onto an Arnoldi basis.

   Level: developer

.seealso: LMESolve()
@*/
PetscErrorCode LMEDenseLyapunovChol(LME lme,PetscScalar *H,PetscInt m,PetscInt ldh,PetscScalar *r,PetscScalar *L,PetscInt ldl,PetscReal *res)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *Hcopy;
#if defined(CHECK)
  PetscReal      error;
#endif

  PetscFunctionBegin;
  ierr = PetscMalloc1(m*m,&Hcopy);CHKERRQ(ierr);
  for (i=0;i<m;i++) {
    ierr = PetscMemcpy(Hcopy+i*m,H+i*ldh,m*sizeof(PetscScalar));CHKERRQ(ierr);
  }
#if defined(SLEPC_HAVE_SLICOT)
  // TODO
#else

  ierr = LyapunovChol_LAPACK(H,m,ldh,r,L,ldl,res);CHKERRQ(ierr);

#endif
#if defined(CHECK)
  ierr = LyapunovResidual(Hcopy,m,m,r,L,ldl,&error);CHKERRQ(ierr);
  ierr = PetscInfo1(lme,"Residual norm of dense Lyapunov equation = %g\n",error);CHKERRQ(ierr);
#endif
  ierr = PetscFree(Hcopy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

