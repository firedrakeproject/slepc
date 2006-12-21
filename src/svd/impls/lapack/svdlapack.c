/*                       
       This file implements a wrapper to the LAPACK SVD subroutines.
*/
#include "src/svd/svdimpl.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "SVDSetup_LAPACK"
PetscErrorCode SVDSetup_LAPACK(SVD svd)
{
  PetscErrorCode  ierr;
  PetscInt        M,N;

  PetscFunctionBegin;
  ierr = MatGetSize(svd->A,&M,&N);CHKERRQ(ierr);
  svd->ncv = PetscMin(M,N);
  svd->max_it = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_LAPACK"
PetscErrorCode SVDSolve_LAPACK(SVD svd)
{
  PetscErrorCode  ierr;
  PetscInt        M,N,n;
  Mat             mat;
  PetscScalar     *pU,*pVT,*pmat,*pu,*pv;
  PetscReal       *sigma;
  int             i,j,k;
#if defined(PETSC_USE_COMPLEX)
  PetscReal       *rwork;
#endif 
  
  PetscFunctionBegin;
  ierr = MatConvert(svd->A,MATSEQDENSE,MAT_INITIAL_MATRIX,&mat);CHKERRQ(ierr);
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
    for (j=0;j<M;j++)
      if (M>=N) pu[j] = pmat[i*M+j];
      else pu[j] = pU[i*M+j];
    for (j=0;j<N;j++)
      if (M>=N) pv[j] = pVT[j*N+i];
      else pv[j] = pmat[j*M+i];
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
  svd->ops->setup = SVDSetup_LAPACK;
  svd->ops->solve = SVDSolve_LAPACK;
  if (svd->transmode == PETSC_DECIDE)
    svd->transmode = SVD_TRANSPOSE_IMPLICIT; /* don't build the transpose */
  PetscFunctionReturn(0);
}
EXTERN_C_END
