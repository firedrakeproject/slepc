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
  svd->n = PetscMin(M,N);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_LAPACK"
PetscErrorCode SVDSolve_LAPACK(SVD svd)
{
  PetscErrorCode  ierr;
  PetscInt        M,N,n;
  Mat             mat;
  PetscScalar     *pU,*pVT,*pmat,*pu,*pv,*work,qwork;
  int             i,j,lwork,*iwork,info;
#if defined(PETSC_USE_COMPLEX)
  PetscReal       *rwork;
#endif 
  
  PetscFunctionBegin;
  ierr = MatConvert(svd->A,MATSEQDENSE,MAT_INITIAL_MATRIX,&mat);CHKERRQ(ierr);
  ierr = MatGetArray(mat,&pmat);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  svd->nconv = n = PetscMin(M,N);
  if (M>=N) {
     pU = PETSC_NULL;
     ierr = PetscMalloc(sizeof(PetscScalar)*N*N,&pVT);CHKERRQ(ierr);
  } else {
     ierr = PetscMalloc(sizeof(PetscScalar)*M*M,&pU);CHKERRQ(ierr);
     pVT = PETSC_NULL;
  }

  /* workspace query & allocation */
  ierr = PetscMalloc(sizeof(int)*8*n,&iwork);CHKERRQ(ierr);
  lwork = -1;
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(sizeof(PetscReal)*(5*n*n+7*n),&rwork);CHKERRQ(ierr);
  LAPACKgesdd("O",&M,&N,pmat,&M,svd->sigma,pU,&M,pVT,&N,&qwork,&lwork,rwork,iwork,&info,1);
#else
  LAPACKgesdd("O",&M,&N,pmat,&M,svd->sigma,pU,&M,pVT,&N,&qwork,&lwork,iwork,&info,1);
#endif 
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
  lwork = qwork;
  ierr = PetscMalloc(sizeof(PetscScalar)*lwork,&work);CHKERRQ(ierr);
  
  /* computation */  
#if defined(PETSC_USE_COMPLEX)
  LAPACKgesdd("O",&M,&N,pmat,&M,svd->sigma,pU,&M,pVT,&N,work,&lwork,rwork,iwork,&info,1);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#else
  LAPACKgesdd("O",&M,&N,pmat,&M,svd->sigma,pU,&M,pVT,&N,work,&lwork,iwork,&info,1);
#endif
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGESDD %d",info);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  
  /* copy singular vectors */
  for (i=0;i<n;i++) {
    ierr = VecGetArray(svd->U[i],&pu);CHKERRQ(ierr);
    ierr = VecGetArray(svd->V[i],&pv);CHKERRQ(ierr);
    if (M>=N) {
      for (j=0;j<M;j++)
        pu[j] = pmat[i*M+j];
      for (j=0;j<N;j++)
        pv[j] = pVT[j*N+i];
    } else {
      for (j=0;j<M;j++)
        pu[j] = pU[i*M+j];
      for (j=0;j<N;j++)
        pv[j] = pmat[j*N+i];
    }
    ierr = VecRestoreArray(svd->U[i],&pu);CHKERRQ(ierr);
    ierr = VecRestoreArray(svd->V[i],&pv);CHKERRQ(ierr);
  }

  ierr = MatRestoreArray(mat,&pmat);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
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
  if (svd->transmode == -1)
    svd->transmode = SVD_TRANSPOSE_MATMULT; /* don't build the transpose */
  PetscFunctionReturn(0);
}
EXTERN_C_END
