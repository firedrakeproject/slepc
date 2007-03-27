/*                       
     This file contains routines for handling small-size dense problems.
     All routines are simply wrappers to LAPACK routines.
*/

#include "slepcsvd.h"        /*I "slepcsvd.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "SVDDense"
/*@
   SVDDense - Solves a dense singular value problem.

   Not Collective

   Input Parameters:
+  M  - dimension of the problem (rows)
.  N  - dimension of the problem (colums)
-  A  - pointer to the array containing the matrix values

   Output Parameters:
+  sigma  - pointer to the array to store the computed singular values
.  U  - pointer to the array to store left singular vectors
-  VT  - pointer to the array to store right singular vectors

   Matrix A is overwritten.
   
   This routine uses LAPACK routines xGESDD.

   Level: developer

@*/
PetscErrorCode SVDDense(int M,int N,PetscScalar* A,PetscReal* sigma,PetscScalar* U,PetscScalar* VT)
{
#if defined(SLEPC_MISSING_LAPACK_GESDD)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"GESDD - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    qwork,*work;
  int            n,info,lwork,*iwork;
#if defined(PETSC_USE_COMPLEX)
  PetscReal       *rwork;
#endif 
  
  PetscFunctionBegin;
  /* workspace query & allocation */
  n = PetscMin(M,N);
  ierr = PetscMalloc(sizeof(int)*8*n,&iwork);CHKERRQ(ierr);
  lwork = -1;
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(sizeof(PetscReal)*(5*n*n+7*n),&rwork);CHKERRQ(ierr);
  LAPACKgesdd_("O",&M,&N,A,&M,sigma,U,&M,VT,&N,&qwork,&lwork,rwork,iwork,&info,1);
#else
  LAPACKgesdd_("O",&M,&N,A,&M,sigma,U,&M,VT,&N,&qwork,&lwork,iwork,&info,1);
#endif 
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
  lwork = qwork;
  ierr = PetscMalloc(sizeof(PetscScalar)*lwork,&work);CHKERRQ(ierr);
  
  /* computation */  
#if defined(PETSC_USE_COMPLEX)
  LAPACKgesdd_("O",&M,&N,A,&M,sigma,U,&M,VT,&N,work,&lwork,rwork,iwork,&info,1);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAPACKgesdd_("O",&M,&N,A,&M,sigma,U,&M,VT,&N,work,&lwork,iwork,&info,1);
#endif
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif 
}
