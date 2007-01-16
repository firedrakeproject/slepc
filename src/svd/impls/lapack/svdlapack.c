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
  PetscInt        N;
  int             i;

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
  PetscInt        M,N;
  Mat             mat;
  PetscScalar     *pU,*pVT,*pmat,*pu,*pv;
  PetscReal       *sigma;
  int             i,j,k;
  
  PetscFunctionBegin;
  if (svd->A) {
    ierr = MatConvert(svd->A,MATSEQDENSE,MAT_INITIAL_MATRIX,&mat);CHKERRQ(ierr);
  } else {
    ierr = MatTranspose(svd->AT,&mat);CHKERRQ(ierr);
    ierr = MatConvert(mat,MATSEQDENSE,MAT_REUSE_MATRIX,&mat);CHKERRQ(ierr);    
  }
  
  ierr = MatGetArray(mat,&pmat);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*N*N,&pVT);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*N,&sigma);CHKERRQ(ierr);
  
  ierr = SVDDense(M,N,pmat,sigma,pU,pVT);CHKERRQ(ierr);

  /* copy singular vectors */
  for (i=0;i<N;i++) {
    if (svd->which == SVD_SMALLEST) k = N - i - 1;
    else k = i;
    svd->sigma[k] = sigma[i];
    ierr = VecGetArray(svd->U[k],&pu);CHKERRQ(ierr);
    ierr = VecGetArray(svd->V[k],&pv);CHKERRQ(ierr);
    for (j=0;j<M;j++) pu[j] = pmat[i*M+j];
    for (j=0;j<N;j++) pv[j] = pVT[j*N+i];
    ierr = VecRestoreArray(svd->U[k],&pu);CHKERRQ(ierr);
    ierr = VecRestoreArray(svd->V[k],&pv);CHKERRQ(ierr);
  }

  svd->nconv = N;
  svd->reason = SVD_CONVERGED_TOL;

  ierr = MatRestoreArray(mat,&pmat);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  ierr = PetscFree(sigma);CHKERRQ(ierr);
  ierr = PetscFree(pVT);CHKERRQ(ierr);
  ierr = PetscFree(pU);CHKERRQ(ierr);
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
