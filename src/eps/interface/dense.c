/*                       
     This file contains routines for handling small-size dense problems.
     All routines are simply wrappers to LAPACK routines. Matrices passed in
     as arguments are assumed to be square matrices stored in column-major 
     format with a leading dimension equal to the number of rows.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include "private/epsimpl.h" /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseNHEP"
/*@
   EPSDenseNHEP - Solves a dense standard non-Hermitian Eigenvalue Problem.

   Not Collective

   Input Parameters:
+  n  - dimension of the eigenproblem
-  A  - pointer to the array containing the matrix values

   Output Parameters:
+  w  - pointer to the array to store the computed eigenvalues
.  wi - imaginary part of the eigenvalues (only when using real numbers)
.  V  - pointer to the array to store right eigenvectors
-  W  - pointer to the array to store left eigenvectors

   Notes:
   If either V or W are PETSC_NULL then the corresponding eigenvectors are 
   not computed.

   Matrix A is overwritten.
   
   This routine uses LAPACK routines xGEEVX.

   Level: developer

.seealso: EPSDenseGNHEP(), EPSDenseHEP(), EPSDenseGHEP()
@*/
PetscErrorCode EPSDenseNHEP(PetscInt n_,PetscScalar *A,PetscScalar *w,PetscScalar *wi,PetscScalar *V,PetscScalar *W)
{
#if defined(SLEPC_MISSING_LAPACK_GEEVX)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEEVX - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscReal      abnrm,*scale,dummy;
  PetscScalar    *work;
  PetscBLASInt   ilo,ihi,n,lwork,info;
  const char     *jobvr,*jobvl;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#else
  PetscBLASInt   idummy;
#endif 

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  lwork = PetscBLASIntCast(4*n_);
  if (V) jobvr = "V";
  else jobvr = "N";
  if (W) jobvl = "V";
  else jobvl = "N";
  ierr  = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr  = PetscMalloc(n*sizeof(PetscReal),&scale);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr  = PetscMalloc(2*n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  LAPACKgeevx_("B",jobvl,jobvr,"N",&n,A,&n,w,W,&n,V,&n,&ilo,&ihi,scale,&abnrm,&dummy,&dummy,work,&lwork,rwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack ZGEEVX %d",info);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAPACKgeevx_("B",jobvl,jobvr,"N",&n,A,&n,w,wi,W,&n,V,&n,&ilo,&ihi,scale,&abnrm,&dummy,&dummy,work,&lwork,&idummy,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DGEEVX %d",info);
#endif 
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(scale);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseGNHEP"
/*@
   EPSDenseGNHEP - Solves a dense Generalized non-Hermitian Eigenvalue Problem.

   Not Collective

   Input Parameters:
+  n  - dimension of the eigenproblem
.  A  - pointer to the array containing the matrix values for A
-  B  - pointer to the array containing the matrix values for B

   Output Parameters:
+  w  - pointer to the array to store the computed eigenvalues
.  wi - imaginary part of the eigenvalues (only when using real numbers)
.  V  - pointer to the array to store right eigenvectors
-  W  - pointer to the array to store left eigenvectors

   Notes:
   If either V or W are PETSC_NULL then the corresponding eigenvectors are 
   not computed.

   Matrices A and B are overwritten.
   
   This routine uses LAPACK routines xGGEVX.

   Level: developer

.seealso: EPSDenseNHEP(), EPSDenseHEP(), EPSDenseGHEP()
@*/
PetscErrorCode EPSDenseGNHEP(PetscInt n_,PetscScalar *A,PetscScalar *B,PetscScalar *w,PetscScalar *wi,PetscScalar *V,PetscScalar *W)
{
#if defined(SLEPC_MISSING_LAPACK_GGEVX)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GGEVX - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscReal      *rscale,*lscale,abnrm,bbnrm,dummy;
  PetscScalar    *alpha,*beta,*work;
  PetscInt       i;
  PetscBLASInt   ilo,ihi,idummy,info,n;
  const char     *jobvr,*jobvl;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
  PetscBLASInt   lwork;
#else
  PetscReal      *alphai;
  PetscBLASInt   lwork;
#endif 

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
#if defined(PETSC_USE_COMPLEX)
  lwork = PetscBLASIntCast(2*n_);
#else
  lwork = PetscBLASIntCast(6*n_);
#endif
  if (V) jobvr = "V";
  else jobvr = "N";
  if (W) jobvl = "V";
  else jobvl = "N";
  ierr  = PetscMalloc(n*sizeof(PetscScalar),&alpha);CHKERRQ(ierr);
  ierr  = PetscMalloc(n*sizeof(PetscScalar),&beta);CHKERRQ(ierr);
  ierr  = PetscMalloc(n*sizeof(PetscReal),&rscale);CHKERRQ(ierr);
  ierr  = PetscMalloc(n*sizeof(PetscReal),&lscale);CHKERRQ(ierr);
  ierr  = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr  = PetscMalloc(6*n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  LAPACKggevx_("B",jobvl,jobvr,"N",&n,A,&n,B,&n,alpha,beta,W,&n,V,&n,&ilo,&ihi,lscale,rscale,&abnrm,&bbnrm,&dummy,&dummy,work,&lwork,rwork,&idummy,&idummy,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack ZGGEVX %d",info);
  for (i=0;i<n;i++) {
    w[i] = alpha[i]/beta[i];
  }
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  ierr  = PetscMalloc(n*sizeof(PetscReal),&alphai);CHKERRQ(ierr);
  LAPACKggevx_("B",jobvl,jobvr,"N",&n,A,&n,B,&n,alpha,alphai,beta,W,&n,V,&n,&ilo,&ihi,lscale,rscale,&abnrm,&bbnrm,&dummy,&dummy,work,&lwork,&idummy,&idummy,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DGGEVX %d",info);
  for (i=0;i<n;i++) {
    w[i] = alpha[i]/beta[i];
    wi[i] = alphai[i]/beta[i];
  }
  ierr = PetscFree(alphai);CHKERRQ(ierr);
#endif 
  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(rscale);CHKERRQ(ierr);
  ierr = PetscFree(lscale);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseHEP"
/*@
   EPSDenseHEP - Solves a dense standard Hermitian Eigenvalue Problem.

   Not Collective

   Input Parameters:
+  n   - dimension of the eigenproblem
.  A   - pointer to the array containing the matrix values
-  lda - leading dimension of A

   Output Parameters:
+  w  - pointer to the array to store the computed eigenvalues
-  V  - pointer to the array to store the eigenvectors

   Notes:
   If V is PETSC_NULL then the eigenvectors are not computed.

   Matrix A is overwritten.
   
   This routine uses LAPACK routines DSYEVR or ZHEEVR.

   Level: developer

.seealso: EPSDenseNHEP(), EPSDenseGNHEP(), EPSDenseGHEP()
@*/
PetscErrorCode EPSDenseHEP(PetscInt n_,PetscScalar *A,PetscInt lda_,PetscReal *w,PetscScalar *V)
{
#if defined(SLEPC_MISSING_LAPACK_SYEVR) || defined(SLEPC_MISSING_LAPACK_HEEVR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"DSYEVR/ZHEEVR - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscReal      abstol = 0.0,vl,vu;
  PetscScalar    *work;
  PetscBLASInt   il,iu,m,*isuppz,*iwork,n,lda,liwork,info;
  const char     *jobz;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
  PetscBLASInt   lwork,lrwork;
#else
  PetscBLASInt   lwork;
#endif 

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  lda = PetscBLASIntCast(lda_);
  liwork = PetscBLASIntCast(10*n_);
#if defined(PETSC_USE_COMPLEX)
  lwork = PetscBLASIntCast(18*n_);
  lrwork = PetscBLASIntCast(24*n_);
#else
  lwork = PetscBLASIntCast(26*n_);
#endif
  if (V) jobz = "V";
  else jobz = "N";
  ierr  = PetscMalloc(2*n*sizeof(PetscBLASInt),&isuppz);CHKERRQ(ierr);
  ierr  = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr  = PetscMalloc(liwork*sizeof(PetscBLASInt),&iwork);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr  = PetscMalloc(lrwork*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  LAPACKsyevr_(jobz,"A","L",&n,A,&lda,&vl,&vu,&il,&iu,&abstol,&m,w,V,&n,isuppz,work,&lwork,rwork,&lrwork,iwork,&liwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack ZHEEVR %d",info);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAPACKsyevr_(jobz,"A","L",&n,A,&lda,&vl,&vu,&il,&iu,&abstol,&m,w,V,&n,isuppz,work,&lwork,iwork,&liwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DSYEVR %d",info);
#endif 
  ierr = PetscFree(isuppz);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseGHEP"
/*@
   EPSDenseGHEP - Solves a dense Generalized Hermitian Eigenvalue Problem.

   Not Collective

   Input Parameters:
+  n  - dimension of the eigenproblem
.  A  - pointer to the array containing the matrix values for A
-  B  - pointer to the array containing the matrix values for B

   Output Parameters:
+  w  - pointer to the array to store the computed eigenvalues
-  V  - pointer to the array to store the eigenvectors

   Notes:
   If V is PETSC_NULL then the eigenvectors are not computed.

   Matrices A and B are overwritten.
   
   This routine uses LAPACK routines DSYGVD or ZHEGVD.

   Level: developer

.seealso: EPSDenseNHEP(), EPSDenseGNHEP(), EPSDenseHEP()
@*/
PetscErrorCode EPSDenseGHEP(PetscInt n_,PetscScalar *A,PetscScalar *B,PetscReal *w,PetscScalar *V)
{
#if defined(SLEPC_MISSING_LAPACK_SYGVD) || defined(SLEPC_MISSING_LAPACK_HEGVD)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"DSYGVD/ZHEGVD - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    *work;
  PetscBLASInt   itype = 1,*iwork,info,n,
                 liwork;
  const char     *jobz;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
  PetscBLASInt   lwork,lrwork;
#else
  PetscBLASInt   lwork;
#endif 

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  if (V) {
    jobz = "V";
    liwork = PetscBLASIntCast(5*n_+3);
#if defined(PETSC_USE_COMPLEX)
    lwork  = PetscBLASIntCast(n_*n_+2*n_);
    lrwork = PetscBLASIntCast(2*n_*n_+5*n_+1);
#else
    lwork  = PetscBLASIntCast(2*n_*n_+6*n_+1);
#endif
  } else {
    jobz = "N";   
    liwork = 1;
#if defined(PETSC_USE_COMPLEX)
    lwork  = PetscBLASIntCast(n_+1);
    lrwork = PetscBLASIntCast(n_);
#else
    lwork  = PetscBLASIntCast(2*n_+1);
#endif
  }
  ierr  = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr  = PetscMalloc(liwork*sizeof(PetscBLASInt),&iwork);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr  = PetscMalloc(lrwork*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  LAPACKsygvd_(&itype,jobz,"U",&n,A,&n,B,&n,w,work,&lwork,rwork,&lrwork,iwork,&liwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack ZHEGVD %d",info);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAPACKsygvd_(&itype,jobz,"U",&n,A,&n,B,&n,w,work,&lwork,iwork,&liwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DSYGVD %d",info);
#endif 
  if (V) {
    ierr = PetscMemcpy(V,A,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseHessenberg"
/*@
   EPSDenseHessenberg - Computes the Hessenberg form of a dense matrix.

   Not Collective

   Input Parameters:
+  n     - dimension of the matrix 
.  k     - first active column
-  lda   - leading dimension of A

   Input/Output Parameters:
+  A  - on entry, the full matrix; on exit, the upper Hessenberg matrix (H)
-  Q  - on exit, orthogonal matrix of vectors A = Q*H*Q'

   Notes:
   Only active columns (from k to n) are computed. 

   Both A and Q are overwritten.
   
   This routine uses LAPACK routines xGEHRD and xORGHR/xUNGHR.

   Level: developer

.seealso: EPSDenseSchur(), EPSSortDenseSchur(), EPSDenseTridiagonal()
@*/
PetscErrorCode EPSDenseHessenberg(PetscInt n_,PetscInt k,PetscScalar *A,PetscInt lda_,PetscScalar *Q)
{
#if defined(SLEPC_MISSING_LAPACK_GEHRD) || defined(SLEPC_MISSING_LAPACK_ORGHR) || defined(SLEPC_MISSING_LAPACK_UNGHR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEHRD,ORGHR/UNGHR - Lapack routines are unavailable.");
#else
  PetscScalar    *tau,*work;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBLASInt   ilo,lwork,info,n,lda;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  lda = PetscBLASIntCast(lda_);
  ierr = PetscMalloc(n*sizeof(PetscScalar),&tau);CHKERRQ(ierr);
  lwork = n;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ilo = PetscBLASIntCast(k+1);
  LAPACKgehrd_(&n,&ilo,&n,A,&lda,tau,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEHRD %d",info);
  for (j=0;j<n-1;j++) {
    for (i=j+2;i<n;i++) {
      Q[i+j*n] = A[i+j*lda];
      A[i+j*lda] = 0.0;
    }      
  }
  LAPACKorghr_(&n,&ilo,&n,Q,&n,tau,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGHR %d",info);
  ierr = PetscFree(tau);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseSchur"
/*@
   EPSDenseSchur - Computes the upper (quasi-)triangular form of a dense 
   upper Hessenberg matrix.

   Not Collective

   Input Parameters:
+  n   - dimension of the matrix 
.  k   - first active column
-  ldh - leading dimension of H

   Input/Output Parameters:
+  H  - on entry, the upper Hessenber matrix; on exit, the upper 
        (quasi-)triangular matrix (T)
-  Z  - on entry, initial transformation matrix; on exit, orthogonal
        matrix of Schur vectors

   Output Parameters:
+  wr - pointer to the array to store the computed eigenvalues
-  wi - imaginary part of the eigenvalues (only when using real numbers)

   Notes:
   This function computes the (real) Schur decomposition of an upper
   Hessenberg matrix H: H*Z = Z*T,  where T is an upper (quasi-)triangular 
   matrix (returned in H), and Z is the orthogonal matrix of Schur vectors.
   Eigenvalues are extracted from the diagonal blocks of T and returned in
   wr,wi. Transformations are accumulated in Z so that on entry it can 
   contain the transformation matrix associated to the Hessenberg reduction.

   Only active columns (from k to n) are computed. 

   Both H and Z are overwritten.
   
   This routine uses LAPACK routines xHSEQR.

   Level: developer

.seealso: EPSDenseHessenberg(), EPSSortDenseSchur(), EPSDenseTridiagonal()
@*/
PetscErrorCode EPSDenseSchur(PetscInt n_,PetscInt k,PetscScalar *H,PetscInt ldh_,PetscScalar *Z,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_HSEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"HSEQR - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscBLASInt   ilo,lwork,info,n,ldh;
  PetscScalar    *work;
#if !defined(PETSC_USE_COMPLEX)
  PetscInt       j;
#endif
  
  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  ldh = PetscBLASIntCast(ldh_);
  lwork = n;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ilo = PetscBLASIntCast(k+1);
#if !defined(PETSC_USE_COMPLEX)
  LAPACKhseqr_("S","V",&n,&ilo,&n,H,&ldh,wr,wi,Z,&n,work,&lwork,&info);
  for (j=0;j<k;j++) {
    if (j==n-1 || H[j*ldh+j+1] == 0.0) { 
      /* real eigenvalue */
      wr[j] = H[j*ldh+j];
      wi[j] = 0.0;
    } else {
      /* complex eigenvalue */
      wr[j] = H[j*ldh+j];
      wr[j+1] = H[j*ldh+j];
      wi[j] = sqrt(PetscAbsReal(H[j*ldh+j+1])) *
              sqrt(PetscAbsReal(H[(j+1)*ldh+j]));
      wi[j+1] = -wi[j];
      j++;
    }
  }
#else
  LAPACKhseqr_("S","V",&n,&ilo,&n,H,&ldh,wr,Z,&n,work,&lwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",info);

  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSortDenseSchur"
/*@
   EPSSortDenseSchur - Reorders the Schur decomposition computed by
   EPSDenseSchur().

   Not Collective

   Input Parameters:
+  eps - the eigensolver context
.  n     - dimension of the matrix 
.  k     - first active column
-  ldt   - leading dimension of T

   Input/Output Parameters:
+  T  - the upper (quasi-)triangular matrix
.  Q  - the orthogonal matrix of Schur vectors
.  wr - pointer to the array to store the computed eigenvalues
-  wi - imaginary part of the eigenvalues (only when using real numbers)

   Notes:
   This function reorders the eigenvalues in wr,wi located in positions k
   to n according to the sort order specified in EPSetWhicheigenpairs. 
   The Schur decomposition Z*T*Z^T, is also reordered by means of rotations 
   so that eigenvalues in the diagonal blocks of T follow the same order.

   Both T and Q are overwritten.
   
   This routine uses LAPACK routines xTREXC.

   Level: developer

.seealso: EPSDenseHessenberg(), EPSDenseSchur(), EPSDenseTridiagonal()
@*/
PetscErrorCode EPSSortDenseSchur(EPS eps,PetscInt n_,PetscInt k,PetscScalar *T,PetscInt ldt_,PetscScalar *Q,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_TREXC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TREXC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,pos,result;
  PetscBLASInt   ifst,ilst,info,n,ldt;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *work;
#endif
  
  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  ldt = PetscBLASIntCast(ldt_);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(n*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#endif
  
  /* selection sort */
  for (i=k;i<n-1;i++) {
    re = wr[i];
    im = wi[i];
    pos = 0;
    j=i+1; /* j points to the next eigenvalue */
#if !defined(PETSC_USE_COMPLEX)
    if (im != 0) j=i+2;
#endif
    /* find minimum eigenvalue */
    for (;j<n;j++) { 
      ierr = EPSCompareEigenvalues(eps,re,im,wr[j],wi[j],&result);CHKERRQ(ierr);
      if (result > 0) {
        re = wr[j];
        im = wi[j];
        pos = j;
      }
#if !defined(PETSC_USE_COMPLEX)
      if (wi[j] != 0) j++;
#endif
    }
    if (pos) {
      /* interchange blocks */
      ifst = PetscBLASIntCast(pos + 1);
      ilst = PetscBLASIntCast(i + 1);
#if !defined(PETSC_USE_COMPLEX)
      LAPACKtrexc_("V",&n,T,&ldt,Q,&n,&ifst,&ilst,work,&info);
#else
      LAPACKtrexc_("V",&n,T,&ldt,Q,&n,&ifst,&ilst,&info);
#endif
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);
      /* recover original eigenvalues from T matrix */
      for (j=i;j<n;j++) {
        wr[j] = T[j*ldt+j];
#if !defined(PETSC_USE_COMPLEX)
        if (j<n-1 && T[j*ldt+j+1] != 0.0) {
          /* complex conjugate eigenvalue */
          wi[j] = sqrt(PetscAbsReal(T[j*ldt+j+1])) *
                  sqrt(PetscAbsReal(T[(j+1)*ldt+j]));
          wr[j+1] = wr[j];
          wi[j+1] = -wi[j];
          j++;
        } else
#endif
        wi[j] = 0.0;
      }
    }
#if !defined(PETSC_USE_COMPLEX)
    if (wi[i] != 0) i++;
#endif
  }

#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(work);CHKERRQ(ierr);
#endif
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);

#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSortDenseSchurGeneralized"
/*@
   EPSSortDenseSchurGeneralized - Reorders a generalized Schur decomposition.

   Not Collective

   Input Parameters:
+  eps   - the eigensolver context
.  n     - dimension of the matrix 
.  k0    - first active column
.  k1    - last column to be ordered
-  ldt   - leading dimension of T

   Input/Output Parameters:
+  T,S  - the upper (quasi-)triangular matrices
.  Q,Z  - the orthogonal matrix of Schur vectors
.  wr - pointer to the array to store the computed eigenvalues
-  wi - imaginary part of the eigenvalues (only when using real numbers)

   Notes:
   This function reorders the eigenvalues in wr,wi located in positions k0
   to n according to the sort order specified in EPSetWhicheigenpairs. 
   The selection sort is the method used to sort the eigenvalues, and it
   stops when the column k1-1 is ordered. The Schur decomposition Z*T*Z^T,
   is also reordered by means of rotations so that eigenvalues in the
   diagonal blocks of T follow the same order.

   T,S,Q and Z are overwritten.
   
   This routine uses LAPACK routines xTGEXC.

   Level: developer

.seealso:  EPSSortDenseSchur(), EPSDenseHessenberg(), EPSDenseSchur(), EPSDenseTridiagonal()
@*/
PetscErrorCode EPSSortDenseSchurGeneralized(EPS eps,PetscInt n_,PetscInt k0,PetscInt k1,PetscScalar *T,PetscScalar *S,PetscInt ldt_,PetscScalar *Q,PetscScalar *Z,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_TGEXC) || !defined(PETSC_USE_COMPLEX) && (defined(SLEPC_MISSING_LAPACK_LAMCH) || defined(SLEPC_MISSING_LAPACK_LAG2))
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TGEXC/LAMCH/LAG2 - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,result,pos;
  PetscBLASInt   ione = 1,ifst,ilst,info,n,ldt;
#if !defined(PETSC_USE_COMPLEX)
  PetscBLASInt   lwork;
  PetscScalar    *work,safmin,scale1,scale2,tmp;
#endif
  
  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  ldt = PetscBLASIntCast(ldt_);
#if !defined(PETSC_USE_COMPLEX)
  lwork = -1;
  LAPACKtgexc_(&ione,&ione,&n,T,&ldt,S,&ldt,Q,&n,Z,&n,&ione,&ione,&tmp,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTGEXC %d",info);
  lwork = (PetscBLASInt)tmp;
  safmin = LAPACKlamch_("S");
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#endif
  
  /* selection sort */
  for (i=k0;i<PetscMin(n-1,k1);i++) {
    re = wr[i];
    im = wi[i];
    pos = 0;
    j=i+1; /* j points to the next eigenvalue */
#if !defined(PETSC_USE_COMPLEX)
    if (im != 0) j=i+2;
#endif
    /* find minimum eigenvalue */
    for (;j<n;j++) { 
      ierr = EPSCompareEigenvalues(eps,re,im,wr[j],wi[j],&result);CHKERRQ(ierr);
      if (result > 0) {
        re = wr[j];
        im = wi[j];
        pos = j;
      }
#if !defined(PETSC_USE_COMPLEX)
      if (wi[j] != 0) j++;
#endif
    }
    if (pos) {
      /* interchange blocks */
      ifst = PetscBLASIntCast(pos + 1);
      ilst = PetscBLASIntCast(i + 1);
#if !defined(PETSC_USE_COMPLEX)
      LAPACKtgexc_(&ione,&ione,&n,T,&ldt,S,&ldt,Q,&n,Z,&n,&ifst,&ilst,work,&lwork,&info);
#else
      LAPACKtgexc_(&ione,&ione,&n,T,&ldt,S,&ldt,Q,&n,Z,&n,&ifst,&ilst,&info);
#endif
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTGEXC %d",info);
      /* recover original eigenvalues from T and S matrices */
      for (j=k0;j<n;j++) {
#if !defined(PETSC_USE_COMPLEX)
        if (j<n-1 && T[j*ldt+j+1] != 0.0) {
          /* complex conjugate eigenvalue */
          LAPACKlag2_(T+j*ldt+j,&ldt,S+j*ldt+j,&ldt,&safmin,&scale1,&scale2,&re,&tmp,&im);
          wr[j] = re / scale1;
          wi[j] = im / scale1;
          wr[j+1] = tmp / scale2;
          wi[j+1] = -wi[j];
          j++;
        } else
#endif
        {
          if (S[j*ldt+j] == 0.0) {
            if (PetscRealPart(T[j*ldt+j]) < 0.0) wr[j] = PETSC_MIN_REAL;
            else wr[j] = PETSC_MAX_REAL;
          } else wr[j] = T[j*ldt+j] / S[j*ldt+j];
          wi[j] = 0.0;
        }
      }
    }
#if !defined(PETSC_USE_COMPLEX)
    if (wi[i] != 0) i++;
#endif
  }

#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(work);CHKERRQ(ierr);
#endif
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);

#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseTridiagonal"
/*@
   EPSDenseTridiagonal - Solves a real tridiagonal Hermitian Eigenvalue Problem.

   Not Collective

   Input Parameters:
+  n   - dimension of the eigenproblem
.  D   - pointer to the array containing the diagonal elements
-  E   - pointer to the array containing the off-diagonal elements

   Output Parameters:
+  w  - pointer to the array to store the computed eigenvalues
-  V  - pointer to the array to store the eigenvectors

   Notes:
   If V is PETSC_NULL then the eigenvectors are not computed.

   This routine use LAPACK routines xSTEVR.

   Level: developer

.seealso: EPSDenseNHEP(), EPSDenseHEP(), EPSDenseGNHEP(), EPSDenseGHEP()
@*/
PetscErrorCode EPSDenseTridiagonal(PetscInt n_,PetscReal *D,PetscReal *E,PetscReal *w,PetscScalar *V)
{
#if defined(SLEPC_MISSING_LAPACK_STEVR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"STEVR - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscReal      abstol = 0.0,vl,vu,*work;
  PetscBLASInt   il,iu,m,*isuppz,n,lwork,*iwork,liwork,info;
  const char     *jobz;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       i,j;
  PetscReal      *VV;
#endif
  
  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  lwork = PetscBLASIntCast(20*n_);
  liwork = PetscBLASIntCast(10*n_);
  if (V) {
    jobz = "V";
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(n*n*sizeof(PetscReal),&VV);CHKERRQ(ierr);
#endif
  } else jobz = "N";
  ierr = PetscMalloc(2*n*sizeof(PetscBLASInt),&isuppz);CHKERRQ(ierr);
  ierr = PetscMalloc(lwork*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(liwork*sizeof(PetscBLASInt),&iwork);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  LAPACKstevr_(jobz,"A",&n,D,E,&vl,&vu,&il,&iu,&abstol,&m,w,VV,&n,isuppz,work,&lwork,iwork,&liwork,&info);
#else
  LAPACKstevr_(jobz,"A",&n,D,E,&vl,&vu,&il,&iu,&abstol,&m,w,V,&n,isuppz,work,&lwork,iwork,&liwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DSTEVR %d",info);
#if defined(PETSC_USE_COMPLEX)
  if (V) {
    for (i=0;i<n;i++) 
      for (j=0;j<n;j++)
        V[i*n+j] = VV[i*n+j];
    ierr = PetscFree(VV);CHKERRQ(ierr);
  }
#endif
  ierr = PetscFree(isuppz);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "DenseSelectedEvec"
/*
   DenseSelectedEvec - Computes a selected eigenvector of matrix in Schur form.

   Input Parameters:
     S - (quasi-)triangular matrix (dimension nv, leading dimension lds)
     U - orthogonal transformation matrix (dimension nv, leading dimension nv)
     i - which eigenvector to process
     iscomplex - true if a complex conjugate pair (in real scalars)

   Output parameters:
     Y - computed eigenvector, 2 columns if iscomplex=true (leading dimension nv)

   Workspace:
     work is workspace to store 3*nv scalars, nv booleans and nv reals
*/
PetscErrorCode DenseSelectedEvec(PetscScalar *S,PetscInt lds_,PetscScalar *U,PetscScalar *Y,PetscInt i,PetscBool iscomplex,PetscInt nv_,PetscScalar *work)
{
#if defined(SLEPC_MISSING_LAPACK_TREVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       k;
  PetscBLASInt   mm,mout,info,lds,nv,inc = 1;
  PetscScalar    tmp,done=1.0,zero=0.0;
  PetscReal      norm;
  PetscBool      *select=(PetscBool*)(work+4*nv_);
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork=(PetscReal*)(work+3*nv_);
#endif

  PetscFunctionBegin;
  lds = PetscBLASIntCast(lds_);
  nv = PetscBLASIntCast(nv_);
  for (k=0;k<nv;k++) select[k] = PETSC_FALSE;

  /* Compute eigenvectors Y of S */
  mm = iscomplex? 2: 1;
  select[i] = PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  if (iscomplex) select[i+1] = PETSC_TRUE;
  LAPACKtrevc_("R","S",select,&nv,S,&lds,PETSC_NULL,&nv,Y,&nv,&mm,&mout,work,&info);
#else
  LAPACKtrevc_("R","S",select,&nv,S,&lds,PETSC_NULL,&nv,Y,&nv,&mm,&mout,work,rwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);
  if (mout != mm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Inconsistent arguments");
  ierr = PetscMemcpy(work,Y,mout*nv*sizeof(PetscScalar));CHKERRQ(ierr);

  /* accumulate and normalize eigenvectors */
  BLASgemv_("N",&nv,&nv,&done,U,&nv,work,&inc,&zero,Y,&inc);
#if !defined(PETSC_USE_COMPLEX)
  if (iscomplex) BLASgemv_("N",&nv,&nv,&done,U,&nv,work+nv,&inc,&zero,Y+nv,&inc);
#endif
  mm = mm*nv;
  norm = BLASnrm2_(&mm,Y,&inc);
  tmp = 1.0 / norm;
  BLASscal_(&mm,&tmp,Y,&inc);

  PetscFunctionReturn(0);
#endif
}

