/*                       
     This file contains routines for handling small-size dense problems.
     All routines are simply wrappers to LAPACK routines. Matrices passed in
     as arguments are assumed to be square matrices stored in column-major 
     format with a leading dimension equal to the number of rows.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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
  SETERRQ(PETSC_ERR_SUP,"GEEVX - Lapack routine is unavailable.");
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
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack ZGEEVX %d",info);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAPACKgeevx_("B",jobvl,jobvr,"N",&n,A,&n,w,wi,W,&n,V,&n,&ilo,&ihi,scale,&abnrm,&dummy,&dummy,work,&lwork,&idummy,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack DGEEVX %d",info);
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
  SETERRQ(PETSC_ERR_SUP,"GGEVX - Lapack routine is unavailable.");
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
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack ZGGEVX %d",info);
  for (i=0;i<n;i++) {
    w[i] = alpha[i]/beta[i];
  }
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  ierr  = PetscMalloc(n*sizeof(PetscReal),&alphai);CHKERRQ(ierr);
  LAPACKggevx_("B",jobvl,jobvr,"N",&n,A,&n,B,&n,alpha,alphai,beta,W,&n,V,&n,&ilo,&ihi,lscale,rscale,&abnrm,&bbnrm,&dummy,&dummy,work,&lwork,&idummy,&idummy,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack DGGEVX %d",info);
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
  SETERRQ(PETSC_ERR_SUP,"DSYEVR/ZHEEVR - Lapack routine is unavailable.");
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
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack ZHEEVR %d",info);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAPACKsyevr_(jobz,"A","L",&n,A,&lda,&vl,&vu,&il,&iu,&abstol,&m,w,V,&n,isuppz,work,&lwork,iwork,&liwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack DSYEVR %d",info);
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
  SETERRQ(PETSC_ERR_SUP,"DSYGVD/ZHEGVD - Lapack routine is unavailable.");
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
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack ZHEGVD %d",info);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAPACKsygvd_(&itype,jobz,"U",&n,A,&n,B,&n,w,work,&lwork,iwork,&liwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack DSYGVD %d",info);
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
  SETERRQ(PETSC_ERR_SUP,"GEHRD,ORGHR/UNGHR - Lapack routines are unavailable.");
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
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGEHRD %d",info);
  for (j=0;j<n-1;j++) {
    for (i=j+2;i<n;i++) {
      Q[i+j*n] = A[i+j*lda];
      A[i+j*lda] = 0.0;
    }      
  }
  LAPACKorghr_(&n,&ilo,&n,Q,&n,tau,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xORGHR %d",info);
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
  SETERRQ(PETSC_ERR_SUP,"HSEQR - Lapack routine is unavailable.");
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
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",info);

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
+  n     - dimension of the matrix 
.  k     - first active column
.  ldt   - leading dimension of T
-  which - eigenvalue sort order

   Input/Output Parameters:
+  T  - the upper (quasi-)triangular matrix
.  Z  - the orthogonal matrix of Schur vectors
.  wr - pointer to the array to store the computed eigenvalues
-  wi - imaginary part of the eigenvalues (only when using real numbers)

   Notes:
   This function reorders the eigenvalues in wr,wi located in positions k
   to n according to the sort order specified in which. The Schur 
   decomposition Z*T*Z^T, is also reordered by means of rotations so that 
   eigenvalues in the diagonal blocks of T follow the same order.

   Both T and Z are overwritten.
   
   This routine uses LAPACK routines xTREXC.

   Level: developer

.seealso: EPSDenseHessenberg(), EPSDenseSchur(), EPSDenseTridiagonal()
@*/
PetscErrorCode EPSSortDenseSchur(PetscInt n_,PetscInt k,PetscScalar *T,PetscInt ldt_,PetscScalar *Z,PetscScalar *wr,PetscScalar *wi,EPSWhich which)
{
#if defined(SLEPC_MISSING_LAPACK_TREXC)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"TREXC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscReal      value,v;
  PetscInt       i,j;
  PetscBLASInt   ifst,ilst,info,pos,n,ldt;
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
  
  for (i=k;i<n-1;i++) {
    switch(which) {
      case EPS_LARGEST_MAGNITUDE:
      case EPS_SMALLEST_MAGNITUDE:
	value = SlepcAbsEigenvalue(wr[i],wi[i]);
	break;
      case EPS_LARGEST_REAL:
      case EPS_SMALLEST_REAL:
	value = PetscRealPart(wr[i]);
	break;
      case EPS_LARGEST_IMAGINARY:
      case EPS_SMALLEST_IMAGINARY:
#if !defined(PETSC_USE_COMPLEX)
	value = PetscAbsReal(wi[i]);
#else
        value = PetscImaginaryPart(wr[i]);
#endif
	break;
      default: SETERRQ(1,"Wrong value of which");
    }
    pos = 0;
    for (j=i+1;j<n;j++) {
      switch(which) {
	case EPS_LARGEST_MAGNITUDE:
	case EPS_SMALLEST_MAGNITUDE:
	  v = SlepcAbsEigenvalue(wr[j],wi[j]);
	  break;
	case EPS_LARGEST_REAL:
	case EPS_SMALLEST_REAL:
	  v = PetscRealPart(wr[j]);
	  break;
	case EPS_LARGEST_IMAGINARY:
	case EPS_SMALLEST_IMAGINARY:
#if !defined(PETSC_USE_COMPLEX)
	  v = PetscAbsReal(wi[j]);
#else
          v = PetscImaginaryPart(wr[j]);
#endif
	  break;
	default: SETERRQ(1,"Wrong value of which");
      }
      switch(which) {
	case EPS_LARGEST_MAGNITUDE:
	case EPS_LARGEST_REAL:
	case EPS_LARGEST_IMAGINARY:
	  if (v > value) {
            value = v;
            pos = j;
	  }
	  break;
	case EPS_SMALLEST_MAGNITUDE:
	case EPS_SMALLEST_REAL:
	case EPS_SMALLEST_IMAGINARY:
	  if (v < value) {
            value = v;
            pos = j;
	  }
	  break;
	default: SETERRQ(1,"Wrong value of which");
      }
#if !defined(PETSC_USE_COMPLEX)
      if (wi[j] != 0) j++;
#endif
    }
    if (pos) {
      ifst = PetscBLASIntCast(pos + 1);
      ilst = PetscBLASIntCast(i + 1);
#if !defined(PETSC_USE_COMPLEX)
      LAPACKtrexc_("V",&n,T,&ldt,Z,&n,&ifst,&ilst,work,&info);
#else
      LAPACKtrexc_("V",&n,T,&ldt,Z,&n,&ifst,&ilst,&info);
#endif
      if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);
      
      for (j=k;j<n;j++) {
#if !defined(PETSC_USE_COMPLEX)
        if (j==n-1 || T[j*ldt+j+1] == 0.0) { 
          /* real eigenvalue */
          wr[j] = T[j*ldt+j];
          wi[j] = 0.0;
        } else {
          /* complex eigenvalue */
          wr[j] = T[j*ldt+j];
          wr[j+1] = T[j*ldt+j];
          wi[j] = sqrt(PetscAbsReal(T[j*ldt+j+1])) *
                  sqrt(PetscAbsReal(T[(j+1)*ldt+j]));
          wi[j+1] = -wi[j];
          j++;
        }
#else
        wr[j] = T[j*(ldt+1)];
#endif
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
#define __FUNCT__ "EPSSortDenseSchurTarget"
/*@
   EPSSortDenseSchurTarget - Reorders the Schur decomposition computed by
   EPSDenseSchur().

   Not Collective

   Input Parameters:
+  n     - dimension of the matrix 
.  k     - first active column
.  ldt   - leading dimension of T
.  target - the target value
-  which - eigenvalue sort order

   Input/Output Parameters:
+  T  - the upper (quasi-)triangular matrix
.  Z  - the orthogonal matrix of Schur vectors
.  wr - pointer to the array to store the computed eigenvalues
-  wi - imaginary part of the eigenvalues (only when using real numbers)

   Notes:
   This function reorders the eigenvalues in wr,wi located in positions k
   to n according to increasing distance to the target. The parameter which
   is used to determine if distance is relative to magnitude, real axis,
   or imaginary axis. The Schur decomposition Z*T*Z^T, is also reordered 
   by means of rotations so that eigenvalues in the diagonal blocks of T 
   follow the same order.

   Both T and Z are overwritten.
   
   This routine uses LAPACK routines xTREXC.

   Level: developer

.seealso: EPSDenseHessenberg(), EPSDenseSchur(), EPSDenseTridiagonal()
@*/
PetscErrorCode EPSSortDenseSchurTarget(PetscInt n_,PetscInt k,PetscScalar *T,PetscInt ldt_,PetscScalar *Z,PetscScalar *wr,PetscScalar *wi,PetscScalar target,EPSWhich which)
{
#if defined(SLEPC_MISSING_LAPACK_TREXC)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"TREXC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscReal      value,v;
  PetscInt       i,j;
  PetscBLASInt   ifst,ilst,info,pos,n,ldt;
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
  
  for (i=k;i<n-1;i++) {
    switch(which) {
      case EPS_LARGEST_MAGNITUDE:
        /* complex target only allowed if scalartype=complex */
        value = SlepcAbsEigenvalue(wr[i]-target,wi[i]);
        break;
      case EPS_LARGEST_REAL:
        value = PetscAbsReal(PetscRealPart(wr[i]-target));
        break;
      case EPS_LARGEST_IMAGINARY:
#if !defined(PETSC_USE_COMPLEX)
        /* complex target only allowed if scalartype=complex */
        value = PetscAbsReal(wi[i]);
#else
        value = PetscAbsReal(PetscImaginaryPart(wr[i]-target));
#endif
        break;
      default: SETERRQ(1,"Wrong value of which");
    }
    pos = 0;
    for (j=i+1;j<n;j++) {
      switch(which) {
        case EPS_LARGEST_MAGNITUDE:
          /* complex target only allowed if scalartype=complex */
          v = SlepcAbsEigenvalue(wr[j]-target,wi[j]);
          break;
        case EPS_LARGEST_REAL:
          v = PetscAbsReal(PetscRealPart(wr[j]-target));
          break;
        case EPS_LARGEST_IMAGINARY:
#if !defined(PETSC_USE_COMPLEX)
          /* complex target only allowed if scalartype=complex */
          v = PetscAbsReal(wi[j]);
#else
          v = PetscAbsReal(PetscImaginaryPart(wr[j]-target));
#endif
          break;
        default: SETERRQ(1,"Wrong value of which");
      }
      if (v < value) {
        value = v;
        pos = j;
      }
#if !defined(PETSC_USE_COMPLEX)
      if (wi[j] != 0) j++;
#endif
    }
    if (pos) {
      ifst = PetscBLASIntCast(pos + 1);
      ilst = PetscBLASIntCast(i + 1);
#if !defined(PETSC_USE_COMPLEX)
      LAPACKtrexc_("V",&n,T,&ldt,Z,&n,&ifst,&ilst,work,&info);
#else
      LAPACKtrexc_("V",&n,T,&ldt,Z,&n,&ifst,&ilst,&info);
#endif
      if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);
      
      for (j=k;j<n;j++) {
#if !defined(PETSC_USE_COMPLEX)
        if (j==n-1 || T[j*ldt+j+1] == 0.0) { 
          /* real eigenvalue */
          wr[j] = T[j*ldt+j];
          wi[j] = 0.0;
        } else {
          /* complex eigenvalue */
          wr[j] = T[j*ldt+j];
          wr[j+1] = T[j*ldt+j];
          wi[j] = sqrt(PetscAbsReal(T[j*ldt+j+1])) *
                  sqrt(PetscAbsReal(T[(j+1)*ldt+j]));
          wi[j+1] = -wi[j];
          j++;
        }
#else
        wr[j] = T[j*(ldt+1)];
#endif
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
.  A   - pointer to the array containing the matrix values
-  lda - leading dimension of A

   Output Parameters:
+  w  - pointer to the array to store the computed eigenvalues
-  V  - pointer to the array to store the eigenvectors

   Notes:
   If V is PETSC_NULL then the eigenvectors are not computed.

   This routine use LAPACK routines DSTEVR.

   Level: developer

.seealso: EPSDenseNHEP(), EPSDenseHEP(), EPSDenseGNHEP(), EPSDenseGHEP()
@*/
PetscErrorCode EPSDenseTridiagonal(PetscInt n_,PetscReal *D,PetscReal *E,PetscReal *w,PetscScalar *V)
{
#if defined(SLEPC_MISSING_LAPACK_STEVR)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"STEVR - Lapack routine is unavailable.");
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
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack DSTEVR %d",info);
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
