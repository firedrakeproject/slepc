/*                       
       This file implements a wrapper to the LAPACK SVD subroutines.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/svdimpl.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "SVDSetup_LAPACK"
PetscErrorCode SVDSetup_LAPACK(SVD svd)
{
  PetscErrorCode  ierr;
  PetscInt        N;
  PetscInt        i;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,PETSC_NULL,&N);CHKERRQ(ierr);
  svd->ncv = N;
  svd->max_it = 1;
  if (svd->ncv!=svd->n) {  
    if (svd->U) {
      for (i=0;i<svd->n;i++) { ierr = VecDestroy(svd->U[i]); CHKERRQ(ierr); }
      ierr = PetscFree(svd->U);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(sizeof(Vec)*svd->ncv,&svd->U);CHKERRQ(ierr);
    for (i=0;i<svd->ncv;i++) { ierr = SVDMatGetVecs(svd,PETSC_NULL,svd->U+i);CHKERRQ(ierr); }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_LAPACK"
PetscErrorCode SVDSolve_LAPACK(SVD svd)
{
  PetscErrorCode  ierr;
  PetscInt        M,N,n,i,j,k;
  Mat             mat;
  PetscScalar     *pU,*pVT,*pmat,*pu,*pv;
  PetscReal       *sigma;
  
  PetscFunctionBegin;
  ierr = MatConvert(svd->OP,MATSEQDENSE,MAT_INITIAL_MATRIX,&mat);CHKERRQ(ierr);
  ierr = MatGetArray(mat,&pmat);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  if (M>=N) {
     n = N;
     pU = PETSC_NULL;
     ierr = PetscMalloc(sizeof(PetscScalar)*N*N,&pVT);CHKERRQ(ierr);
  } else {
     n = M;
     ierr = PetscMalloc(sizeof(PetscScalar)*M*M,&pU);CHKERRQ(ierr);
     pVT = PETSC_NULL;
  }
  ierr = PetscMalloc(sizeof(PetscReal)*n,&sigma);CHKERRQ(ierr);
  
  ierr = SVDDense(M,N,pmat,sigma,pU,pVT);CHKERRQ(ierr);

  /* copy singular vectors */
  for (i=0;i<n;i++) {
    if (svd->which == SVD_SMALLEST) k = n - i - 1;
    else k = i;
    svd->sigma[k] = sigma[i];
    ierr = VecGetArray(svd->U[k],&pu);CHKERRQ(ierr);
    ierr = VecGetArray(svd->V[k],&pv);CHKERRQ(ierr);
    if (M>=N) {
      for (j=0;j<M;j++) pu[j] = pmat[i*M+j];
      for (j=0;j<N;j++) pv[j] = pVT[j*N+i];
    } else {
      for (j=0;j<N;j++) pu[j] = pmat[j*M+i];
      for (j=0;j<M;j++) pv[j] = pU[i*M+j];
    }
    ierr = VecRestoreArray(svd->U[k],&pu);CHKERRQ(ierr);
    ierr = VecRestoreArray(svd->V[k],&pv);CHKERRQ(ierr);
  }

  svd->nconv = n;
  svd->reason = SVD_CONVERGED_TOL;

  ierr = MatRestoreArray(mat,&pmat);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  ierr = PetscFree(sigma);CHKERRQ(ierr);
  if (M>=N) {
     ierr = PetscFree(pVT);CHKERRQ(ierr);
  } else {
     ierr = PetscFree(pU);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCreate_LAPACK"
PetscErrorCode SVDCreate_LAPACK(SVD svd)
{
  PetscFunctionBegin;
  svd->ops->setup   = SVDSetup_LAPACK;
  svd->ops->solve   = SVDSolve_LAPACK;
  svd->ops->destroy = SVDDestroy_Default;
  if (svd->transmode == PETSC_DECIDE)
    svd->transmode = SVD_TRANSPOSE_IMPLICIT; /* don't build the transpose */
  PetscFunctionReturn(0);
}
EXTERN_C_END
