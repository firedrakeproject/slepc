/*
   This file implements a wrapper to the LAPACK SVD subroutines.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/svdimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SVDSetUp_LAPACK"
PetscErrorCode SVDSetUp_LAPACK(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       M,N;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,&M,&N);CHKERRQ(ierr);
  svd->ncv = N;
  if (svd->mpd) { ierr = PetscInfo(svd,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (svd->stop!=SVD_STOP_BASIC) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"User-defined stopping test not supported in this solver");
  svd->max_it = 1;
  svd->leftbasis = PETSC_TRUE;
  ierr = SVDAllocateSolution(svd,0);CHKERRQ(ierr);
  ierr = DSSetType(svd->ds,DSSVD);CHKERRQ(ierr);
  ierr = DSAllocate(svd->ds,PetscMax(M,N));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSolve_LAPACK"
PetscErrorCode SVDSolve_LAPACK(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       M,N,n,i,j,k,ld;
  Mat            mat;
  Vec            u,v;
  PetscScalar    *pU,*pVT,*pmat,*pu,*pv,*A,*w;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
  ierr = MatConvert(svd->OP,MATSEQDENSE,MAT_INITIAL_MATRIX,&mat);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  ierr = DSSetDimensions(svd->ds,M,N,0,0);CHKERRQ(ierr);
  ierr = MatDenseGetArray(mat,&pmat);CHKERRQ(ierr);
  ierr = DSGetArray(svd->ds,DS_MAT_A,&A);CHKERRQ(ierr);
  for (i=0;i<M;i++)
    for (j=0;j<N;j++)
      A[i+j*ld] = pmat[i+j*M];
  ierr = DSRestoreArray(svd->ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(mat,&pmat);CHKERRQ(ierr);
  ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);

  n = PetscMin(M,N);
  ierr = PetscMalloc1(n,&w);CHKERRQ(ierr);
  ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
  ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /* copy singular vectors */
  ierr = DSGetArray(svd->ds,DS_MAT_U,&pU);CHKERRQ(ierr);
  ierr = DSGetArray(svd->ds,DS_MAT_VT,&pVT);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    if (svd->which == SVD_SMALLEST) k = n - i - 1;
    else k = i;
    svd->sigma[k] = PetscRealPart(w[i]);
    ierr = BVGetColumn(svd->U,k,&u);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->V,k,&v);CHKERRQ(ierr);
    ierr = VecGetArray(u,&pu);CHKERRQ(ierr);
    ierr = VecGetArray(v,&pv);CHKERRQ(ierr);
    if (M>=N) {
      for (j=0;j<M;j++) pu[j] = pU[i*ld+j];
      for (j=0;j<N;j++) pv[j] = PetscConj(pVT[j*ld+i]);
    } else {
      for (j=0;j<N;j++) pu[j] = PetscConj(pVT[j*ld+i]);
      for (j=0;j<M;j++) pv[j] = pU[i*ld+j];
    }
    ierr = VecRestoreArray(u,&pu);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&pv);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,k,&u);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->V,k,&v);CHKERRQ(ierr);
  }
  ierr = DSRestoreArray(svd->ds,DS_MAT_U,&pU);CHKERRQ(ierr);
  ierr = DSRestoreArray(svd->ds,DS_MAT_VT,&pVT);CHKERRQ(ierr);

  svd->nconv = n;
  svd->reason = SVD_CONVERGED_TOL;

  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscFree(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCreate_LAPACK"
PETSC_EXTERN PetscErrorCode SVDCreate_LAPACK(SVD svd)
{
  PetscFunctionBegin;
  svd->ops->setup   = SVDSetUp_LAPACK;
  svd->ops->solve   = SVDSolve_LAPACK;
  PetscFunctionReturn(0);
}

