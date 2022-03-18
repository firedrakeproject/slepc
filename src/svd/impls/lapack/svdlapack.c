/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to the LAPACK SVD subroutines
*/

#include <slepc/private/svdimpl.h>
#include <slepcblaslapack.h>

PetscErrorCode SVDSetUp_LAPACK(SVD svd)
{
  PetscInt       M,N,P=0;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(svd->A,&M,&N));
  if (!svd->isgeneralized) svd->ncv = N;
  else {
    CHKERRQ(MatGetSize(svd->OPb,&P,NULL));
    svd->ncv = PetscMin(M,PetscMin(N,P));
  }
  if (svd->mpd!=PETSC_DEFAULT) CHKERRQ(PetscInfo(svd,"Warning: parameter mpd ignored\n"));
  SVDCheckUnsupported(svd,SVD_FEATURE_STOPPING);
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = 1;
  svd->leftbasis = PETSC_TRUE;
  CHKERRQ(SVDAllocateSolution(svd,0));
  CHKERRQ(DSSetType(svd->ds,svd->isgeneralized?DSGSVD:DSSVD));
  CHKERRQ(DSAllocate(svd->ds,PetscMax(N,PetscMax(M,P))));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_LAPACK(SVD svd)
{
  PetscInt          M,N,n,i,j,k,ld,lowu,lowv,highu,highv;
  Mat               Ar,mat;
  Vec               u,v;
  PetscScalar       *pU,*pV,*pu,*pv,*A,*w;
  const PetscScalar *pmat;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(svd->ds,&ld));
  CHKERRQ(MatCreateRedundantMatrix(svd->OP,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Ar));
  CHKERRQ(MatConvert(Ar,MATSEQDENSE,MAT_INITIAL_MATRIX,&mat));
  CHKERRQ(MatDestroy(&Ar));
  CHKERRQ(MatGetSize(mat,&M,&N));
  CHKERRQ(DSSetDimensions(svd->ds,M,0,0));
  CHKERRQ(DSSVDSetDimensions(svd->ds,N));
  CHKERRQ(MatDenseGetArrayRead(mat,&pmat));
  CHKERRQ(DSGetArray(svd->ds,DS_MAT_A,&A));
  for (i=0;i<M;i++)
    for (j=0;j<N;j++)
      A[i+j*ld] = pmat[i+j*M];
  CHKERRQ(DSRestoreArray(svd->ds,DS_MAT_A,&A));
  CHKERRQ(MatDenseRestoreArrayRead(mat,&pmat));
  CHKERRQ(DSSetState(svd->ds,DS_STATE_RAW));

  n = PetscMin(M,N);
  CHKERRQ(PetscMalloc1(n,&w));
  CHKERRQ(DSSolve(svd->ds,w,NULL));
  CHKERRQ(DSSort(svd->ds,w,NULL,NULL,NULL,NULL));
  CHKERRQ(DSSynchronize(svd->ds,w,NULL));

  /* copy singular vectors */
  CHKERRQ(DSGetArray(svd->ds,DS_MAT_U,&pU));
  CHKERRQ(DSGetArray(svd->ds,DS_MAT_V,&pV));
  for (i=0;i<n;i++) {
    if (svd->which == SVD_SMALLEST) k = n - i - 1;
    else k = i;
    svd->sigma[k] = PetscRealPart(w[i]);
    CHKERRQ(BVGetColumn(svd->U,k,&u));
    CHKERRQ(BVGetColumn(svd->V,k,&v));
    CHKERRQ(VecGetOwnershipRange(u,&lowu,&highu));
    CHKERRQ(VecGetOwnershipRange(v,&lowv,&highv));
    CHKERRQ(VecGetArray(u,&pu));
    CHKERRQ(VecGetArray(v,&pv));
    if (M>=N) {
      for (j=lowu;j<highu;j++) pu[j-lowu] = pU[i*ld+j];
      for (j=lowv;j<highv;j++) pv[j-lowv] = pV[i*ld+j];
    } else {
      for (j=lowu;j<highu;j++) pu[j-lowu] = pV[i*ld+j];
      for (j=lowv;j<highv;j++) pv[j-lowv] = pU[i*ld+j];
    }
    CHKERRQ(VecRestoreArray(u,&pu));
    CHKERRQ(VecRestoreArray(v,&pv));
    CHKERRQ(BVRestoreColumn(svd->U,k,&u));
    CHKERRQ(BVRestoreColumn(svd->V,k,&v));
  }
  CHKERRQ(DSRestoreArray(svd->ds,DS_MAT_U,&pU));
  CHKERRQ(DSRestoreArray(svd->ds,DS_MAT_V,&pV));

  svd->nconv  = n;
  svd->its    = 1;
  svd->reason = SVD_CONVERGED_TOL;

  CHKERRQ(MatDestroy(&mat));
  CHKERRQ(PetscFree(w));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_LAPACK_GSVD(SVD svd)
{
  PetscInt          nsv,m,n,p,i,j,mlocal,plocal,ld,lowx,lowu,lowv,highx;
  Mat               Ar,A,Br,B;
  Vec               uv,x;
  PetscScalar       *Ads,*Bds,*U,*V,*X,*px,*puv,*w;
  const PetscScalar *pA,*pB;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(svd->ds,&ld));
  CHKERRQ(MatCreateRedundantMatrix(svd->OP,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Ar));
  CHKERRQ(MatConvert(Ar,MATSEQDENSE,MAT_INITIAL_MATRIX,&A));
  CHKERRQ(MatDestroy(&Ar));
  CHKERRQ(MatCreateRedundantMatrix(svd->OPb,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Br));
  CHKERRQ(MatConvert(Br,MATSEQDENSE,MAT_INITIAL_MATRIX,&B));
  CHKERRQ(MatDestroy(&Br));
  CHKERRQ(MatGetSize(A,&m,&n));
  CHKERRQ(MatGetLocalSize(svd->OP,&mlocal,NULL));
  CHKERRQ(MatGetLocalSize(svd->OPb,&plocal,NULL));
  CHKERRQ(MatGetSize(B,&p,NULL));
  CHKERRQ(DSSetDimensions(svd->ds,m,0,0));
  CHKERRQ(DSGSVDSetDimensions(svd->ds,n,p));
  CHKERRQ(MatDenseGetArrayRead(A,&pA));
  CHKERRQ(MatDenseGetArrayRead(B,&pB));
  CHKERRQ(DSGetArray(svd->ds,DS_MAT_A,&Ads));
  CHKERRQ(DSGetArray(svd->ds,DS_MAT_B,&Bds));
  for (j=0;j<n;j++) {
    for (i=0;i<m;i++) Ads[i+j*ld] = pA[i+j*m];
    for (i=0;i<p;i++) Bds[i+j*ld] = pB[i+j*p];
  }
  CHKERRQ(DSRestoreArray(svd->ds,DS_MAT_B,&Bds));
  CHKERRQ(DSRestoreArray(svd->ds,DS_MAT_A,&Ads));
  CHKERRQ(MatDenseRestoreArrayRead(B,&pB));
  CHKERRQ(MatDenseRestoreArrayRead(A,&pA));
  CHKERRQ(DSSetState(svd->ds,DS_STATE_RAW));

  nsv  = PetscMin(n,PetscMin(p,m));
  CHKERRQ(PetscMalloc1(nsv,&w));
  CHKERRQ(DSSolve(svd->ds,w,NULL));
  CHKERRQ(DSSort(svd->ds,w,NULL,NULL,NULL,NULL));
  CHKERRQ(DSSynchronize(svd->ds,w,NULL));
  CHKERRQ(DSGetDimensions(svd->ds,NULL,NULL,NULL,&nsv));

  /* copy singular vectors */
  CHKERRQ(MatGetOwnershipRange(svd->OP,&lowu,NULL));
  CHKERRQ(MatGetOwnershipRange(svd->OPb,&lowv,NULL));
  CHKERRQ(DSGetArray(svd->ds,DS_MAT_X,&X));
  CHKERRQ(DSGetArray(svd->ds,DS_MAT_U,&U));
  CHKERRQ(DSGetArray(svd->ds,DS_MAT_V,&V));
  for (j=0;j<nsv;j++) {
    svd->sigma[j] = PetscRealPart(w[j]);
    CHKERRQ(BVGetColumn(svd->V,j,&x));
    CHKERRQ(VecGetOwnershipRange(x,&lowx,&highx));
    CHKERRQ(VecGetArrayWrite(x,&px));
    for (i=lowx;i<highx;i++) px[i-lowx] = X[i+j*ld];
    CHKERRQ(VecRestoreArrayWrite(x,&px));
    CHKERRQ(BVRestoreColumn(svd->V,j,&x));
    CHKERRQ(BVGetColumn(svd->U,j,&uv));
    CHKERRQ(VecGetArrayWrite(uv,&puv));
    for (i=0;i<mlocal;i++) puv[i] = U[i+lowu+j*ld];
    for (i=0;i<plocal;i++) puv[i+mlocal] = V[i+lowv+j*ld];
    CHKERRQ(VecRestoreArrayWrite(uv,&puv));
    CHKERRQ(BVRestoreColumn(svd->U,j,&uv));
  }
  CHKERRQ(DSRestoreArray(svd->ds,DS_MAT_X,&X));
  CHKERRQ(DSRestoreArray(svd->ds,DS_MAT_U,&U));
  CHKERRQ(DSRestoreArray(svd->ds,DS_MAT_V,&V));

  svd->nconv  = nsv;
  svd->its    = 1;
  svd->reason = SVD_CONVERGED_TOL;

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFree(w));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_LAPACK(SVD svd)
{
  PetscFunctionBegin;
  svd->ops->setup   = SVDSetUp_LAPACK;
  svd->ops->solve   = SVDSolve_LAPACK;
  svd->ops->solveg  = SVDSolve_LAPACK_GSVD;
  PetscFunctionReturn(0);
}
