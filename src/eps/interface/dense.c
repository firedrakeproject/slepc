/*                       
     This file contains routines for handling small size dense problems.
*/
#include "slepceps.h" /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#define SWAP(a,b,t) {t=a;a=b;b=t;}

#undef __FUNCT__  
#define __FUNCT__ "EPSSortEigenvalues"
/*@
   EPSSortEigenvalues - Sorts a list of eigenvalues according to a certain
   criterion.

   Not Collective

   Input Parameters:
+  n     - number of eigenvalue in the list
.  eig   - pointer to the array containing the eigenvalues
.  eigi  - imaginary part of the eigenvalues (only when using real numbers)
.  which - sorting criterion
-  nev   - number of wanted eigenvalues

   Output Parameter:
.  permout - resulting permutation

   Notes:
   The result is a list of indices in the original eigenvalue array 
   corresponding to the first nev eigenvalues sorted in the specified
   criterion

   Level: developer

.seealso: EPSDenseNHEPSorted(), EPSSetWhichEigenpairs()
@*/
PetscErrorCode EPSSortEigenvalues(int n,PetscScalar *eig,PetscScalar *eigi,EPSWhich which,int nev,int *permout)
{
  PetscErrorCode ierr;
  int            i,*perm;
  PetscReal      *values;

  PetscFunctionBegin;
  ierr = PetscMalloc(n*sizeof(int),&perm);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscReal),&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) { perm[i] = i;}

  switch(which) {
    case EPS_LARGEST_MAGNITUDE:
    case EPS_SMALLEST_MAGNITUDE:
      for (i=0; i<n; i++) { values[i] = SlepcAbsEigenvalue(eig[i],eigi[i]); }
      break;
    case EPS_LARGEST_REAL:
    case EPS_SMALLEST_REAL:
      for (i=0; i<n; i++) { values[i] = PetscRealPart(eig[i]); }
      break;
    case EPS_LARGEST_IMAGINARY:
    case EPS_SMALLEST_IMAGINARY:
#if defined(PETSC_USE_COMPLEX)
      for (i=0; i<n; i++) { values[i] = PetscImaginaryPart(eig[i]); }
#else
      for (i=0; i<n; i++) { values[i] = PetscAbsReal(eigi[i]); }
#endif
      break;
    default: SETERRQ(1,"Wrong value of which");
  }

  ierr = PetscSortRealWithPermutation(n,values,perm);CHKERRQ(ierr);

  switch(which) {
    case EPS_LARGEST_MAGNITUDE:
    case EPS_LARGEST_REAL:
    case EPS_LARGEST_IMAGINARY:
      for (i=0; i<nev; i++) { permout[i] = perm[n-1-i]; }
      break;
    case EPS_SMALLEST_MAGNITUDE:
    case EPS_SMALLEST_REAL:
    case EPS_SMALLEST_IMAGINARY: 
      for (i=0; i<nev; i++) { permout[i] = perm[i]; }
      break;
    default: SETERRQ(1,"Wrong value of which");
  }

#if !defined(PETSC_USE_COMPLEX)
  for (i=0; i<nev-1; i++) {
    if (eigi[permout[i]] != 0.0) {
      if (eig[permout[i]] == eig[permout[i+1]] &&
          eigi[permout[i]] == -eigi[permout[i+1]] &&
          eigi[permout[i]] < 0.0) {
        int tmp;
        SWAP(permout[i], permout[i+1], tmp);
      }
    i++;
    }
  }
#endif

  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseNHEP"
/*@
   EPSDenseNHEP - Solves a dense non-Hermitian Eigenvalue Problem.

   Not Collective

   Input Parameters:
+  n  - dimension of the eigenproblem
-  A  - pointer to the array containing the matrix values

   Output Parameters:
+  w  - pointer to the array to store the computed eigenvalues
.  wi - imaginary part of the eigenvalues (only when using real numbers)
-  V  - pointer to the array to store the eigenvectors

   Notes:
   If V is PETSC_NULL then the eigenvectors are not computed.

   Matrix A is overwritten.
   
   This routine uses LAPACK routines xGEEV.

   Level: developer

.seealso: EPSDenseNHEPSorted()
@*/
PetscErrorCode EPSDenseNHEP(int n,PetscScalar *A,PetscScalar *w,PetscScalar *wi,PetscScalar *V)
{
  PetscErrorCode ierr;
  
#if defined(PETSC_HAVE_ESSL)

  /* ESSL has a different calling sequence for dgeev() and zgeev() than 
     standard LAPACK */
  PetscScalar *cwork;
  PetscReal   *work;
  int         i,clen,idummy,lwork,iopt;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  clen = n;
#else
  clen = 2*n;
#endif
  ierr   = PetscMalloc(clen*sizeof(PetscScalar),&cwork);CHKERRQ(ierr);
  idummy = n;
  lwork  = 3*n;
  ierr   = PetscMalloc(lwork*sizeof(PetscReal),&work);CHKERRQ(ierr);
  if (V) iopt = 1;
  else iopt = 0;
  LAgeev_(&iopt,A,&n,cwork,V,&n,&idummy,&n,work,&lwork);
  ierr = PetscFree(work);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  for (i=0; i<n; i++) {
    w[i]  = cwork[2*i];
    wi[i] = cwork[2*i+1];
  }
#else
  for (i=0; i<n; i++) w[i] = cwork[i];
#endif
  ierr = PetscFree(cwork);CHKERRQ(ierr);

#elif !defined(PETSC_USE_COMPLEX)

  PetscScalar *work,sdummy;
  int         lwork,info;
  char        *jobvr;

  PetscFunctionBegin;
  lwork    = 5*n;
  ierr     = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  if (V) jobvr = "V";
  else jobvr = "N";
  LAgeev_("N",jobvr,&n,A,&n,w,wi,&sdummy,&n,V,&n,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGEEV %d",info);
  ierr = PetscFree(work);CHKERRQ(ierr);

#else

  PetscScalar *work,sdummy;
  PetscReal   *rwork;
  int         lwork,info;
  char        *jobvr;

  PetscFunctionBegin;
#if defined(PETSC_MISSING_LAPACK_GEEV)
  SETERRQ(PETSC_ERR_SUP,"GEEV - Lapack routine is unavailable.");
#endif 
  lwork    = 5*n;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(2*n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  if (V) jobvr = "V";
  else jobvr = "N";
  LAgeev_("N",jobvr,&n,A,&n,w,&sdummy,&n,V,&n,work,&lwork,rwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGEEV %d",info);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(rwork);CHKERRQ(ierr);

#endif 

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDenseNHEPSorted"
/*@
   EPSDenseNHEPSorted - Solves a dense non-Hermitian Eigenvalue Problem and 
   then sorts the computed eigenpairs.

   Not Collective

   Input Parameters:
+  n  - dimension of the eigenproblem
-  A  - pointer to the array containing the matrix values

   Output Parameters:
+  w  - pointer to the array to store the computed eigenvalues
.  wi - imaginary part of the eigenvalues (only when using real numbers)
-  V  - pointer to the array to store the eigenvectors

   Notes:
   If V is PETSC_NULL then the eigenvectors are not computed.

   Matrix A is overwritten.

   Level: developer

.seealso: EPSDenseNHEP(), EPSSortEigenvalues()
@*/
PetscErrorCode EPSDenseNHEPSorted(int n,PetscScalar *A,PetscScalar *w,PetscScalar *wi,PetscScalar *V,int m,EPSWhich which)
{
  PetscErrorCode ierr;
  int            i,*perm,iwork[100];
  PetscScalar    *realpart,*imagpart,*vectors,work[200];

  PetscFunctionBegin;
  if (m<=100) perm = iwork;
  else { ierr = PetscMalloc(m*sizeof(int),&perm);CHKERRQ(ierr); }
  if (n<=100) { realpart = work; imagpart = work+100; }
  else { 
    ierr = PetscMalloc(n*sizeof(PetscScalar),&realpart);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(PetscScalar),&imagpart);CHKERRQ(ierr);
  }
  if (V) {
    ierr   = PetscMalloc(n*n*sizeof(PetscScalar),&vectors);CHKERRQ(ierr);
  } else vectors = PETSC_NULL;

  ierr = EPSDenseNHEP(n,A,realpart,imagpart,vectors);CHKERRQ(ierr);

  ierr = EPSSortEigenvalues(n,realpart,imagpart,which,m,perm);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    w[i]  = realpart[perm[i]];
#if !defined(PETSC_USE_COMPLEX)
    wi[i] = imagpart[perm[i]];
#endif
    if (V) {
      ierr = PetscMemcpy(V+i*n,vectors+perm[i]*n,n*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }

  if (m>100) { ierr = PetscFree(perm);CHKERRQ(ierr); }
  if (n>100) {
    ierr = PetscFree(realpart);CHKERRQ(ierr);
    ierr = PetscFree(imagpart);CHKERRQ(ierr);
  }
  if (V) {
    ierr = PetscFree(vectors);CHKERRQ(ierr);
  }

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
PetscErrorCode EPSDenseSchur(PetscScalar *H,PetscScalar *Z,PetscScalar *wr,PetscScalar *wi,int k,int n)
{
  PetscErrorCode ierr;
  int ilo,lwork,info;
  PetscScalar *work;
  
  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
  SETERRQ(PETSC_ERR_SUP,"HSEQR - Lapack routine is unavailable.");
#endif 

  lwork = n;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ilo = k+1;
#if !defined(PETSC_USE_COMPLEX)
  LAhseqr_("S","V",&n,&ilo,&n,H,&n,wr,wi,Z,&n,work,&lwork,&info,1,1);
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
PetscErrorCode EPSSortDenseSchur(PetscScalar *T,PetscScalar *Z,PetscScalar *wr,PetscScalar *wi,int k,int n)
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
#endif

  for (i=k;i<n-1;i++) {
    max = SlepcAbsEigenvalue(wr[i],wi[i]);
    maxpos = 0;
    for (j=i+1;j<n;j++) {
      m = SlepcAbsEigenvalue(wr[j],wi[j]);
      if (m > max) {
        max = m;
        maxpos = j;
      }
#if !defined(PETSC_USE_COMPLEX)
      if (wi[j] != 0) j++;
#endif
    }
    if (maxpos) {
      ifst = maxpos + 1;
      ilst = i + 1;
#if !defined(PETSC_USE_COMPLEX)
      LAtrexc_("V",&n,T,&n,Z,&n,&ifst,&ilst,work,&info,1);
#else
      LAtrexc_("V",&n,T,&n,Z,&n,&ifst,&ilst,&info,1);
#endif
      if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);

#if !defined(PETSC_USE_COMPLEX)
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
          j++;
        }
      }
#else
      for (j=k;j<n;j++) {
        wr[j] = T[j*(n+1)];
      }
#endif
    }
#if !defined(PETSC_USE_COMPLEX)
    if (wi[i] != 0) i++;
#endif
  }
  
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(work);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
