/*                       
     This file contains routines for handling small-size dense problems.
     All routines are simply wrappers to LAPACK routines.
*/

#include "slepcsvd.h"
#include "slepcblaslapack.h"

PetscErrorCode SVDDense(int M,int N,PetscScalar* A,PetscReal* sigma,PetscScalar* U,PetscScalar* VT)
{
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
}
