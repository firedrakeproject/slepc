/*                       
     This file contains routines for handling small-size dense problems.
     All routines are simply wrappers to LAPACK routines.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/svd/svdimpl.h"        /*I "slepcsvd.h" I*/
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
PetscErrorCode SVDDense(PetscInt M_,PetscInt N_,PetscScalar* A,PetscReal* sigma,PetscScalar* U,PetscScalar* VT)
{
#if defined(SLEPC_MISSING_LAPACK_GESDD)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"GESDD - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    qwork,*work;
  PetscBLASInt   n,info,lwork,*iwork,M=M_,N=N_;
#if defined(PETSC_USE_COMPLEX)
  PetscReal       *rwork;
#endif 
  
  PetscFunctionBegin;
  /* workspace query & allocation */
  ierr = PetscLogEventBegin(SVD_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscMin(M,N);
  ierr = PetscMalloc(sizeof(PetscInt)*8*n,&iwork);CHKERRQ(ierr);
  lwork = -1;
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(sizeof(PetscReal)*(5*n*n+7*n),&rwork);CHKERRQ(ierr);
  LAPACKgesdd_("O",&M,&N,A,&M,sigma,U,&M,VT,&N,&qwork,&lwork,rwork,iwork,&info,1);
#else
  LAPACKgesdd_("O",&M,&N,A,&M,sigma,U,&M,VT,&N,&qwork,&lwork,iwork,&info,1);
#endif 
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
  lwork = (PetscInt)PetscRealPart(qwork);
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
  ierr = PetscLogEventEnd(SVD_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif 
}
