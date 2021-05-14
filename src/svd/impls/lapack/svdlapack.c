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
  PetscErrorCode ierr;
  PetscInt       M,N,P=0;

  PetscFunctionBegin;
  ierr = MatGetSize(svd->A,&M,&N);CHKERRQ(ierr);
  if (!svd->isgeneralized) svd->ncv = N;
  else {
    ierr = MatGetSize(svd->OPb,&P,NULL);CHKERRQ(ierr);
    svd->ncv = PetscMin(M,PetscMin(N,P));
  }
  if (svd->mpd!=PETSC_DEFAULT) { ierr = PetscInfo(svd,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  SVDCheckUnsupported(svd,SVD_FEATURE_STOPPING);
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = 1;
  svd->leftbasis = PETSC_TRUE;
  ierr = SVDAllocateSolution(svd,0);CHKERRQ(ierr);
  ierr = DSSetType(svd->ds,svd->isgeneralized?DSGSVD:DSSVD);CHKERRQ(ierr);
  ierr = DSAllocate(svd->ds,PetscMax(N,PetscMax(M,P)));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_LAPACK(SVD svd)
{
  PetscErrorCode    ierr;
  PetscInt          M,N,n,i,j,k,ld,lowu,lowv,highu,highv;
  Mat               Ar,mat;
  Vec               u,v;
  PetscScalar       *pU,*pV,*pu,*pv,*A,*w;
  const PetscScalar *pmat;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
  ierr = MatCreateRedundantMatrix(svd->OP,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Ar);CHKERRQ(ierr);
  ierr = MatConvert(Ar,MATSEQDENSE,MAT_INITIAL_MATRIX,&mat);CHKERRQ(ierr);
  ierr = MatDestroy(&Ar);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  ierr = DSSetDimensions(svd->ds,M,0,0);CHKERRQ(ierr);
  ierr = DSSVDSetDimensions(svd->ds,N);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(mat,&pmat);CHKERRQ(ierr);
  ierr = DSGetArray(svd->ds,DS_MAT_A,&A);CHKERRQ(ierr);
  for (i=0;i<M;i++)
    for (j=0;j<N;j++)
      A[i+j*ld] = pmat[i+j*M];
  ierr = DSRestoreArray(svd->ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(mat,&pmat);CHKERRQ(ierr);
  ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);

  n = PetscMin(M,N);
  ierr = PetscMalloc1(n,&w);CHKERRQ(ierr);
  ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
  ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DSSynchronize(svd->ds,w,NULL);CHKERRQ(ierr);

  /* copy singular vectors */
  ierr = DSGetArray(svd->ds,DS_MAT_U,&pU);CHKERRQ(ierr);
  ierr = DSGetArray(svd->ds,DS_MAT_V,&pV);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    if (svd->which == SVD_SMALLEST) k = n - i - 1;
    else k = i;
    svd->sigma[k] = PetscRealPart(w[i]);
    ierr = BVGetColumn(svd->U,k,&u);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->V,k,&v);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(u,&lowu,&highu);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(v,&lowv,&highv);CHKERRQ(ierr);
    ierr = VecGetArray(u,&pu);CHKERRQ(ierr);
    ierr = VecGetArray(v,&pv);CHKERRQ(ierr);
    if (M>=N) {
      for (j=lowu;j<highu;j++) pu[j-lowu] = pU[i*ld+j];
      for (j=lowv;j<highv;j++) pv[j-lowv] = pV[i*ld+j];
    } else {
      for (j=lowu;j<highu;j++) pu[j-lowu] = pV[i*ld+j];
      for (j=lowv;j<highv;j++) pv[j-lowv] = pU[i*ld+j];
    }
    ierr = VecRestoreArray(u,&pu);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&pv);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,k,&u);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->V,k,&v);CHKERRQ(ierr);
  }
  ierr = DSRestoreArray(svd->ds,DS_MAT_U,&pU);CHKERRQ(ierr);
  ierr = DSRestoreArray(svd->ds,DS_MAT_V,&pV);CHKERRQ(ierr);

  svd->nconv  = n;
  svd->its    = 1;
  svd->reason = SVD_CONVERGED_TOL;

  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscFree(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_LAPACK_GSVD(SVD svd)
{
  PetscErrorCode    ierr;
  PetscInt          nsv,m,n,p,i,j,mlocal,plocal,ld,lowx,lowu,lowv,highx;
  Mat               Ar,A,Br,B;
  Vec               uv,x;
  PetscScalar       *Ads,*Bds,*U,*V,*X,*px,*puv,*w;
  const PetscScalar *pA,*pB;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
  ierr = MatCreateRedundantMatrix(svd->OP,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Ar);CHKERRQ(ierr);
  ierr = MatConvert(Ar,MATSEQDENSE,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Ar);CHKERRQ(ierr);
  ierr = MatCreateRedundantMatrix(svd->OPb,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Br);CHKERRQ(ierr);
  ierr = MatConvert(Br,MATSEQDENSE,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatDestroy(&Br);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->OP,&mlocal,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->OPb,&plocal,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(B,&p,NULL);CHKERRQ(ierr);
  ierr = DSSetDimensions(svd->ds,m,0,0);CHKERRQ(ierr);
  ierr = DSGSVDSetDimensions(svd->ds,n,p);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(A,&pA);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(B,&pB);CHKERRQ(ierr);
  ierr = DSGetArray(svd->ds,DS_MAT_A,&Ads);CHKERRQ(ierr);
  ierr = DSGetArray(svd->ds,DS_MAT_B,&Bds);CHKERRQ(ierr);
  for (j=0;j<n;j++) {
    for (i=0;i<m;i++) Ads[i+j*ld] = pA[i+j*m];
    for (i=0;i<p;i++) Bds[i+j*ld] = pB[i+j*p];
  }
  ierr = DSRestoreArray(svd->ds,DS_MAT_B,&Bds);CHKERRQ(ierr);
  ierr = DSRestoreArray(svd->ds,DS_MAT_A,&Ads);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,&pB);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(A,&pA);CHKERRQ(ierr);
  ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);

  nsv  = PetscMin(n,PetscMin(p,m));
  ierr = PetscMalloc1(nsv,&w);CHKERRQ(ierr);
  ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
  ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DSSynchronize(svd->ds,w,NULL);CHKERRQ(ierr);
  ierr = DSGetDimensions(svd->ds,NULL,NULL,NULL,&nsv);CHKERRQ(ierr);

  /* copy singular vectors */
  ierr = MatGetOwnershipRange(svd->OP,&lowu,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(svd->OPb,&lowv,NULL);CHKERRQ(ierr);
  ierr = DSGetArray(svd->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = DSGetArray(svd->ds,DS_MAT_U,&U);CHKERRQ(ierr);
  ierr = DSGetArray(svd->ds,DS_MAT_V,&V);CHKERRQ(ierr);
  for (j=0;j<nsv;j++) {
    svd->sigma[j] = PetscRealPart(w[j]);
    ierr = BVGetColumn(svd->V,j,&x);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(x,&lowx,&highx);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(x,&px);CHKERRQ(ierr);
    for (i=lowx;i<highx;i++) px[i-lowx] = X[i+j*ld];
    ierr = VecRestoreArrayWrite(x,&px);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->V,j,&x);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->U,j,&uv);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(uv,&puv);CHKERRQ(ierr);
    for (i=0;i<mlocal;i++) puv[i] = U[i+lowu+j*ld];
    for (i=0;i<plocal;i++) puv[i+mlocal] = V[i+lowv+j*ld];
    ierr = VecRestoreArrayWrite(uv,&puv);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,j,&uv);CHKERRQ(ierr);
  }
  ierr = DSRestoreArray(svd->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = DSRestoreArray(svd->ds,DS_MAT_U,&U);CHKERRQ(ierr);
  ierr = DSRestoreArray(svd->ds,DS_MAT_V,&V);CHKERRQ(ierr);

  svd->nconv  = nsv;
  svd->its    = 1;
  svd->reason = SVD_CONVERGED_TOL;

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFree(w);CHKERRQ(ierr);
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

