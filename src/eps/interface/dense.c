/*                       
     This file contains routines for handling small-size dense problems.
     All routines are simply wrappers to LAPACK routines. Matrices passed in
     as arguments are assumed to be square matrices stored in column-major 
     format with a leading dimension equal to the number of rows.
*/
#include "slepceps.h" /*I "slepceps.h" I*/
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
PetscErrorCode EPSDenseNHEP(int n,PetscScalar *A,PetscScalar *w,PetscScalar *wi,PetscScalar *V,PetscScalar *W)
{
  PetscErrorCode ierr;
  PetscReal      abnrm,*scale;
  PetscScalar    *work;
  int            ilo,ihi,lwork = 4*n,info;
  char           *jobvr,*jobvl;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif 

  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
  SETERRQ(PETSC_ERR_SUP,"GEEVX - Lapack routine is unavailable.");
#endif 

  if (V) jobvr = "V";
  else jobvr = "N";
  if (W) jobvl = "V";
  else jobvl = "N";
  ierr  = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr  = PetscMalloc(n*sizeof(PetscReal),&scale);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr  = PetscMalloc(2*n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  LAgeevx_("B",jobvl,jobvr,"N",&n,A,&n,w,W,&n,V,&n,&ilo,&ihi,scale,&abnrm,PETSC_NULL,PETSC_NULL,work,&lwork,rwork,&info,1,1,1,1);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack DGEEVX %d",info);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAgeevx_("B",jobvl,jobvr,"N",&n,A,&n,w,wi,W,&n,V,&n,&ilo,&ihi,scale,&abnrm,PETSC_NULL,PETSC_NULL,work,&lwork,PETSC_NULL,&info,1,1,1,1);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack ZGEEVX %d",info);
#endif 
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
PetscErrorCode EPSDenseGNHEP(int n,PetscScalar *A,PetscScalar *B,PetscScalar *w,PetscScalar *wi,PetscScalar *V,PetscScalar *W)
{
  PetscErrorCode ierr;
  PetscReal      *rscale,*lscale,abnrm,bbnrm;
  PetscScalar    *alpha,*beta,*work;
  int            i,ilo,ihi,info;
  char           *jobvr,*jobvl;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
  int            lwork = 2*n;
#else
  PetscReal      *alphai;
  int            lwork = 6*n;
#endif 

  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY) || defined(PETSC_BLASLAPACK_F2C)
  SETERRQ(PETSC_ERR_SUP,"GGEVX - Lapack routine is unavailable.");
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
  LAggevx_("B",jobvl,jobvr,"N",&n,A,&n,B,&n,alpha,beta,W,&n,V,&n,&ilo,&ihi, lscale,rscale,&abnrm,&bbnrm,PETSC_NULL,PETSC_NULL,work,&lwork,rwork,PETSC_NULL,PETSC_NULL,&info,1,1,1,1);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack DGGEVX %d",info);
  for (i=0;i<n;i++) {
    w[i] = alpha[i]/beta[i];
  }
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  ierr  = PetscMalloc(n*sizeof(PetscReal),&alphai);CHKERRQ(ierr);
  LAggevx_("B",jobvl,jobvr,"N",&n,A,&n,B,&n,alpha,alphai,beta,W,&n,V,&n,&ilo,&ihi, lscale,rscale,&abnrm,&bbnrm,PETSC_NULL,PETSC_NULL,work,&lwork,PETSC_NULL,PETSC_NULL,&info,1,1,1,1);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack ZGGEVX %d",info);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseHEP"
/*@
   EPSDenseHEP - Solves a dense standard Hermitian Eigenvalue Problem.

   Not Collective

   Input Parameters:
+  n  - dimension of the eigenproblem
-  A  - pointer to the array containing the matrix values

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
PetscErrorCode EPSDenseHEP(int n,PetscScalar *A,PetscReal *w,PetscScalar *V)
{
  PetscErrorCode ierr;
  PetscReal      abstol = 0.0,dummy;
  PetscScalar    *work;
  int            m,*isuppz,*iwork,liwork = 10*n,info;
  char           *jobz;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
  int            lwork = 18*n,lrwork = 24*n;
#else
  int            lwork = 26*n;
#endif 

  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY) || defined(PETSC_BLASLAPACK_F2C)
  SETERRQ(PETSC_ERR_SUP,"DSYEVR/ZHEEVR - Lapack routine is unavailable.");
#endif 

  if (V) jobz = "V";
  else jobz = "N";
  ierr  = PetscMalloc(2*n*sizeof(int),&isuppz);CHKERRQ(ierr);
  ierr  = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr  = PetscMalloc(liwork*sizeof(int),&iwork);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr  = PetscMalloc(lrwork*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  LAsyevr_(jobz,"A","U",&n,A,&n,&dummy,&dummy,PETSC_NULL,PETSC_NULL,&abstol,&m,w,V,&n,isuppz,work,&lwork,rwork,&lrwork,iwork,&liwork,&info,1,1,1);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack ZHEEVR %d",info);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAsyevr_(jobz,"A","U",&n,A,&n,&dummy,&dummy,PETSC_NULL,PETSC_NULL,&abstol,&m,w,V,&n,isuppz,work,&lwork,iwork,&liwork,&info,1,1,1);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack DSYEVR %d",info);
#endif 
  ierr = PetscFree(isuppz);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
PetscErrorCode EPSDenseGHEP(int n,PetscScalar *A,PetscScalar *B,PetscReal *w,PetscScalar *V)
{
  PetscErrorCode ierr;
  PetscScalar    *work;
  int            itype = 1,*iwork,info,
                 liwork = V ? 5*n+3 : 1;
  char           *jobz;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
  int            lwork  = V ? n*n+2*n     : n+1,
                 lrwork = V ? 2*n*n+5*n+1 : n;
#else
  int            lwork  = V ? 2*n*n+6*n+1 : 2*n+1;
#endif 

  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY) || defined(PETSC_BLASLAPACK_F2C)
  SETERRQ(PETSC_ERR_SUP,"DSYGVD/ZHEGVD - Lapack routine is unavailable.");
#endif 

  if (V) jobz = "V";
  else jobz = "N";   
  ierr  = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr  = PetscMalloc(liwork*sizeof(int),&iwork);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr  = PetscMalloc(lrwork*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  LAsygvd_(&itype,jobz,"U",&n,A,&n,B,&n,w,work,&lwork,rwork,&lrwork,iwork,&liwork,&info,1,1);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack ZHEGVD %d",info);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAsygvd_(&itype,jobz,"U",&n,A,&n,B,&n,w,work,&lwork,iwork,&liwork,&info,1,1);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack DSYGVD %d",info);
#endif 
  if (V) {
    ierr = PetscMemcpy(V,A,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseSchur"
/*@
   EPSDenseSchur - Computes the upper (quasi-)triangular form of a dense 
   upper Hessenberg matrix.

   Not Collective

   Input Parameters:
+  n  - dimension of the matrix 
-  k  - first active column

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

.seealso: EPSSortDenseSchur()
@*/
PetscErrorCode EPSDenseSchur(int n,int k,PetscScalar *H,PetscScalar *Z,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  int ilo,lwork,info;
  PetscScalar *work;
#if !defined(PETSC_USE_COMPLEX)
  int j;
#endif
  
  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
  SETERRQ(PETSC_ERR_SUP,"HSEQR - Lapack routine is unavailable.");
#endif 

  lwork = n;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ilo = k+1;
#if !defined(PETSC_USE_COMPLEX)
  LAhseqr_("S","V",&n,&ilo,&n,H,&n,wr,wi,Z,&n,work,&lwork,&info,1,1);
  for (j=0;j<k;j++) {
    if (j==n-1 || H[j*n+j+1] == 0.0) { 
      /* real eigenvalue */
      wr[j] = H[j*n+j];
      wi[j] = 0.0;
    } else {
      /* complex eigenvalue */
      wr[j] = H[j*n+j];
      wr[j+1] = H[j*n+j];
      wi[j] = sqrt(PetscAbsReal(H[j*n+j+1])) *
              sqrt(PetscAbsReal(H[(j+1)*n+j]));
      wi[j+1] = -wi[j];
      j++;
    }
  }
#else
  LAhseqr_("S","V",&n,&ilo,&n,H,&n,wr,Z,&n,work,&lwork,&info,1,1);
#endif
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",info);

  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSortDenseSchur"
/*@
   EPSSortDenseSchur - Reorders the Schur decomposition computed by
   EPSDenseSchur().

   Not Collective

   Input Parameters:
+  n  - dimension of the matrix 
-  k  - first active column

   Input/Output Parameters:
+  T  - the upper (quasi-)triangular matrix
-  Z  - the orthogonal matrix of Schur vectors

   Output Parameters:
+  wr - pointer to the array to store the computed eigenvalues
-  wi - imaginary part of the eigenvalues (only when using real numbers)

   Notes:
   This function reorders the eigenvalues in wr,wi located in positions k
   to n in ascending order of magnitude. The Schur decomposition Z*T*Z^T,
   is also reordered by means of rotations so that eigenvalues in the
   diagonal blocks of T follow the same order.

   Both T and Z are overwritten.
   
   This routine uses LAPACK routines xTREXC.

   Level: developer

.seealso: EPSDenseSchur()
@*/
PetscErrorCode EPSSortDenseSchur(int n,int k,PetscScalar *T,PetscScalar *Z,PetscScalar *wr,PetscScalar *wi)
{
  int i,j,ifst,ilst,info,maxpos;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar *work;
  PetscErrorCode ierr;
#endif
  PetscReal   max,m;
  
  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
  SETERRQ(PETSC_ERR_SUP,"TREXC - Lapack routine is unavailable.");
#endif 

#if !defined(PETSC_USE_COMPLEX)

  ierr = PetscMalloc(n*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  
  for (i=k;i<n-1;i++) {
    max = SlepcAbsEigenvalue(wr[i],wi[i]);
    maxpos = 0;
    for (j=i+1;j<n;j++) {
      m = SlepcAbsEigenvalue(wr[j],wi[j]);
      if (m > max) {
        max = m;
        maxpos = j;
      }
      if (wi[j] != 0) j++;
    }
    if (maxpos) {
      ifst = maxpos + 1;
      ilst = i + 1;
      LAtrexc_("V",&n,T,&n,Z,&n,&ifst,&ilst,work,&info,1);
      if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);
      
      for (j=k;j<n;j++) {
        if (j==n-1 || T[j*n+j+1] == 0.0) { 
          /* real eigenvalue */
          wr[j] = T[j*n+j];
          wi[j] = 0.0;
        } else {
          /* complex eigenvalue */
          wr[j] = T[j*n+j];
          wr[j+1] = T[j*n+j];
          wi[j] = sqrt(PetscAbsReal(T[j*n+j+1])) *
                  sqrt(PetscAbsReal(T[(j+1)*n+j]));
          wi[j+1] = -wi[j];
          j++;
        }
      }
    }
    if (wi[i] != 0) i++;
  }
  
  ierr = PetscFree(work);CHKERRQ(ierr);

#else /* PETSC_USE_COMPLEX */

  for (i=k;i<n-1;i++) {
    max = SlepcAbsEigenvalue(wr[i],wi[i]);
    maxpos = 0;
    for (j=i+1;j<n;j++) {
      m = SlepcAbsEigenvalue(wr[j],wi[j]);
      if (m > max) {
        max = m;
        maxpos = j;
      }
    }
    if (maxpos) {
      ifst = maxpos + 1;
      ilst = i + 1;
      LAtrexc_("V",&n,T,&n,Z,&n,&ifst,&ilst,&info,1);
      if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);

      for (j=k;j<n;j++) {
        wr[j] = T[j*(n+1)];
      }
    }
  }

#endif
  
  PetscFunctionReturn(0);
}
