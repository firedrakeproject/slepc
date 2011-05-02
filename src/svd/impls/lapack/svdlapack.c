/*                       
       This file implements a wrapper to the LAPACK SVD subroutines.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include <private/svdimpl.h>      /*I "slepcsvd.h" I*/
#include <slepcblaslapack.h>

#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_LAPACK"
PetscErrorCode SVDSetUp_LAPACK(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       N,i,nloc;
  PetscScalar    *pU;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,PETSC_NULL,&N);CHKERRQ(ierr);
  svd->ncv = N;
  if (svd->mpd) PetscInfo(svd,"Warning: parameter mpd ignored\n");
  svd->max_it = 1;
  if (svd->ncv!=svd->n) {  
    if (svd->U) {
      ierr = VecGetArray(svd->U[0],&pU);CHKERRQ(ierr);
      for (i=0;i<svd->n;i++) { ierr = VecDestroy(&svd->U[i]);CHKERRQ(ierr); }
      ierr = PetscFree(pU);CHKERRQ(ierr);
      ierr = PetscFree(svd->U);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(sizeof(Vec)*svd->ncv,&svd->U);CHKERRQ(ierr);
    ierr = SVDMatGetLocalSize(svd,&nloc,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscMalloc(svd->ncv*nloc*sizeof(PetscScalar),&pU);CHKERRQ(ierr);
    for (i=0;i<svd->ncv;i++) {
      ierr = VecCreateMPIWithArray(((PetscObject)svd)->comm,nloc,PETSC_DECIDE,pU+i*nloc,&svd->U[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_LAPACK"
PetscErrorCode SVDSolve_LAPACK(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       M,N,n,i,j,k;
  Mat            mat;
  PetscScalar    *pU,*pVT,*pmat,*pu,*pv;
  PetscReal      *sigma;
  
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
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
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
  svd->ops->setup   = SVDSetUp_LAPACK;
  svd->ops->solve   = SVDSolve_LAPACK;
  svd->ops->destroy = SVDDestroy_Default;
  if (svd->transmode == PETSC_DECIDE)
    svd->transmode = SVD_TRANSPOSE_IMPLICIT; /* don't build the transpose */
  PetscFunctionReturn(0);
}
EXTERN_C_END
