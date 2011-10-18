/*                       
     This file contains routines for handling small-size dense problems.
     All routines are simply wrappers to LAPACK routines.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <private/svdimpl.h>        /*I "slepcsvd.h" I*/
#include <slepcblaslapack.h>

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

   Notes:
   Matrix A is overwritten.
   
   This routine uses LAPACK routines xGESDD with JOBZ='O'. Thus, if M>=N
   then U is not referenced and the left singular vectors are returned
   in A, and if M<N then VT is not referenced and the right singular
   vectors are returned in A.

   Level: developer
@*/
PetscErrorCode SVDDense(PetscInt M_,PetscInt N_,PetscScalar* A,PetscReal* sigma,PetscScalar* U,PetscScalar* VT)
{
#if defined(SLEPC_MISSING_LAPACK_GESDD)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESDD - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    qwork,*work;
  PetscBLASInt   n,info,lwork,*iwork,M,N;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif 
  
  PetscFunctionBegin;
  /* workspace query & allocation */
  ierr = PetscLogEventBegin(SVD_Dense,0,0,0,0);CHKERRQ(ierr);
  M = PetscBLASIntCast(M_);
  N = PetscBLASIntCast(N_);
  n = PetscMin(M,N);
  ierr = PetscMalloc(sizeof(PetscInt)*8*n,&iwork);CHKERRQ(ierr);
  lwork = -1;
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(sizeof(PetscReal)*(5*n*n+7*n),&rwork);CHKERRQ(ierr);
  LAPACKgesdd_("O",&M,&N,A,&M,sigma,U,&M,VT,&N,&qwork,&lwork,rwork,iwork,&info);
#else
  LAPACKgesdd_("O",&M,&N,A,&M,sigma,U,&M,VT,&N,&qwork,&lwork,iwork,&info);
#endif 
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
  lwork = (PetscBLASInt)PetscRealPart(qwork);
  ierr = PetscMalloc(sizeof(PetscScalar)*lwork,&work);CHKERRQ(ierr);
  
  /* computation */  
#if defined(PETSC_USE_COMPLEX)
  LAPACKgesdd_("O",&M,&N,A,&M,sigma,U,&M,VT,&N,work,&lwork,rwork,iwork,&info);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAPACKgesdd_("O",&M,&N,A,&M,sigma,U,&M,VT,&N,work,&lwork,iwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SVD_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif 
}

