
/*                       
       This file implements interfaces to direct solvers in LAPACK
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
int EPSSortEigenvalues(int n,PetscScalar *eig,PetscScalar *eigi,EPSWhich which,int nev,int *permout)
{
  int       ierr,i,*perm;
  PetscReal *values;

  PetscFunctionBegin;
  ierr = PetscMalloc(n*sizeof(int),&perm);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscReal),&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) { perm[i] = i;}

  switch(which) {
    case EPS_LARGEST_MAGNITUDE:
    case EPS_SMALLEST_MAGNITUDE:
#if defined(PETSC_USE_COMPLEX)
      for (i=0; i<n; i++) { values[i] = PetscAbsScalar(eig[i]); }
#else
      for (i=0; i<n; i++) { values[i] = LAlapy2_(&eig[i],&eigi[i]); }
#endif
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

   Input Parameter:
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
int EPSDenseNHEP(int n,PetscScalar *A,PetscScalar *w,PetscScalar *wi,PetscScalar *V)
{
  int ierr;
  
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
  int         lwork;
  char        *jobvr;

  PetscFunctionBegin;
  lwork    = 5*n;
  ierr     = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  if (V) jobvr = "V";
  else jobvr = "N";
  LAgeev_("N",jobvr,&n,A,&n,w,wi,&sdummy,&n,V,&n,work,&lwork,&ierr);
  if (ierr) SETERRQ1(PETSC_ERR_LIB,"Error in LAPACK routine %d",ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);

#else

  PetscScalar *work,sdummy;
  PetscReal   *rwork;
  int         lwork;
  char        *jobvr;

  PetscFunctionBegin;
  lwork    = 5*n;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(2*n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  if (V) jobvr = "V";
  else jobvr = "N";
  LAgeev_("N",jobvr,&n,A,&n,w,&sdummy,&n,V,&n,work,&lwork,rwork,&ierr);
  if (ierr) SETERRQ1(PETSC_ERR_LIB,"Error in LAPACK routine %d",ierr);
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

   Input Parameter:
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
int EPSDenseNHEPSorted(int n,PetscScalar *A,PetscScalar *w,PetscScalar *wi,PetscScalar *V,int m,EPSWhich which)
{
  int         i,ierr,*perm,iwork[100];
  PetscScalar *realpart,*imagpart,*vectors,work[200];

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
int EPSDenseSchur(PetscScalar *H,PetscScalar *Z,PetscScalar *wr,PetscScalar *wi,int k,int n)
{
  int         ierr,i,j,ilo,ifst,ilst,lwork,maxpos;
  PetscScalar *work;
  PetscReal   max,m;
  
  PetscFunctionBegin;

  lwork = n;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ilo = k+1;
#if !defined(PETSC_USE_COMPLEX)
  LAhseqr_("S","V",&n,&ilo,&n,H,&n,wr,wi,Z,&n,work,&lwork,&ierr,1,1);
#else
  LAhseqr_("S","V",&n,&ilo,&n,H,&n,wr,Z,&n,work,&lwork,&ierr,1,1);
#endif
  if (ierr) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xHSEQR %i",ierr);


  for (i=k;i<n-1;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (wi[i] != 0) max = LAlapy2_(&wr[i],&wi[j]); else 
#endif
    max = PetscAbsScalar(wr[i]);
    maxpos = 0;
    for (j=i+1;j<n;j++) {
#if !defined(PETSC_USE_COMPLEX)
      if (wi[j] != 0) m = LAlapy2_(&wr[j],&wi[j]); else
#endif
      m = PetscAbsScalar(wr[j]);
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
      LAtrexc_("V",&n,H,&n,Z,&n,&ifst,&ilst,work,&ierr,1);
#else
      LAtrexc_("V",&n,H,&n,Z,&n,&ifst,&ilst,&ierr,1);
#endif
      if (ierr) SETERRQ(PETSC_ERR_LIB,"Error in Lapack xTREXC");
      for (j=i;j<n;j++) {
        wr[j] = H[j*(n+1)];
#if !defined(PETSC_USE_COMPLEX)
#endif
      }
    }
#if !defined(PETSC_USE_COMPLEX)
   if (wi[i] != 0) i++;
#endif
  }
  
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
