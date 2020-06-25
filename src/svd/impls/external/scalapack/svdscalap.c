/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to the ScaLAPACK SVD solver
*/

#include <slepc/private/svdimpl.h>    /*I "slepcsvd.h" I*/
#include <slepc/private/slepcscalapack.h>

typedef struct {
  Mat As;        /* converted matrix */
} SVD_ScaLAPACK;

PetscErrorCode SVDSetUp_ScaLAPACK(SVD svd)
{
  PetscErrorCode ierr;
  SVD_ScaLAPACK  *ctx = (SVD_ScaLAPACK*)svd->data;
  PetscInt       M,N;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,&M,&N);CHKERRQ(ierr);
  svd->ncv = N;
  if (svd->mpd!=PETSC_DEFAULT) { ierr = PetscInfo(svd,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = 1;
  svd->leftbasis = PETSC_TRUE;
  SVDCheckUnsupported(svd,SVD_FEATURE_STOPPING);
  ierr = SVDAllocateSolution(svd,0);CHKERRQ(ierr);

  /* convert matrix */
  ierr = MatDestroy(&ctx->As);CHKERRQ(ierr);
  ierr = MatConvert(svd->OP,MATSCALAPACK,MAT_INITIAL_MATRIX,&ctx->As);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_ScaLAPACK(SVD svd)
{
  PetscErrorCode ierr;
  SVD_ScaLAPACK  *ctx = (SVD_ScaLAPACK*)svd->data;
  Mat            A = ctx->As,Z,Q,QT,U,V;
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data,*q,*z;
  PetscScalar    *work,minlwork;
  PetscBLASInt   info,lwork=-1,one=1;
#if defined(PETSC_USE_COMPLEX)
  PetscBLASInt   lrwork;
  PetscReal      *rwork,dummy;
#endif

  PetscFunctionBegin;
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Z);CHKERRQ(ierr);
  z = (Mat_ScaLAPACK*)Z->data;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&QT);CHKERRQ(ierr);
  ierr = MatSetSizes(QT,A->cmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(QT,MATSCALAPACK);CHKERRQ(ierr);
  ierr = MatSetUp(QT);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(QT,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(QT,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  q = (Mat_ScaLAPACK*)QT->data;

#if !defined(PETSC_USE_COMPLEX)
  /* allocate workspace */
  PetscStackCallBLAS("SCALAPACKgesvd",SCALAPACKgesvd_("V","V",&a->M,&a->N,a->loc,&one,&one,a->desc,svd->sigma,z->loc,&one,&one,z->desc,q->loc,&one,&one,q->desc,&minlwork,&lwork,&info));
  PetscCheckScaLapackInfo("gesvd",info);
  ierr = PetscBLASIntCast(minlwork,&lwork);CHKERRQ(ierr);
  ierr = PetscMalloc1(lwork,&work);CHKERRQ(ierr);
  /* call computational routine */
  PetscStackCallBLAS("SCALAPACKgesvd",SCALAPACKgesvd_("V","V",&a->M,&a->N,a->loc,&one,&one,a->desc,svd->sigma,z->loc,&one,&one,z->desc,q->loc,&one,&one,q->desc,work,&lwork,&info));
  PetscCheckScaLapackInfo("gesvd",info);
  ierr = PetscFree(work);CHKERRQ(ierr);
#else
  /* allocate workspace */
  PetscStackCallBLAS("SCALAPACKgesvd",SCALAPACKgesvd_("V","V",&a->M,&a->N,a->loc,&one,&one,a->desc,svd->sigma,z->loc,&one,&one,z->desc,q->loc,&one,&one,q->desc,&minlwork,&lwork,&dummy,&info));
  PetscCheckScaLapackInfo("gesvd",info);
  ierr = PetscBLASIntCast(minlwork,&lwork);CHKERRQ(ierr);
  lrwork = 1+4*PetscMax(a->M,a->N);
  ierr = PetscMalloc2(lwork,&work,lrwork,&rwork);CHKERRQ(ierr);
  /* call computational routine */
  PetscStackCallBLAS("SCALAPACKgesvd",SCALAPACKgesvd_("V","V",&a->M,&a->N,a->loc,&one,&one,a->desc,svd->sigma,z->loc,&one,&one,z->desc,q->loc,&one,&one,q->desc,work,&lwork,rwork,&info));
  PetscCheckScaLapackInfo("gesvd",info);
  ierr = PetscFree2(work,rwork);CHKERRQ(ierr);
#endif

  ierr = BVGetMat(svd->U,&U);CHKERRQ(ierr);
  ierr = MatConvert(Z,MATDENSE,MAT_REUSE_MATRIX,&U);CHKERRQ(ierr);
  ierr = BVRestoreMat(svd->U,&U);CHKERRQ(ierr);
  ierr = MatDestroy(&Z);CHKERRQ(ierr);
  ierr = MatHermitianTranspose(QT,MAT_INITIAL_MATRIX,&Q);CHKERRQ(ierr);
  ierr = MatDestroy(&QT);CHKERRQ(ierr);
  ierr = BVGetMat(svd->V,&V);CHKERRQ(ierr);
  ierr = MatConvert(Q,MATDENSE,MAT_REUSE_MATRIX,&V);CHKERRQ(ierr);
  ierr = BVRestoreMat(svd->V,&V);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);

  svd->nconv  = svd->ncv;
  svd->its    = 1;
  svd->reason = SVD_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_ScaLAPACK(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDReset_ScaLAPACK(SVD svd)
{
  PetscErrorCode ierr;
  SVD_ScaLAPACK  *ctx = (SVD_ScaLAPACK*)svd->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&ctx->As);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_ScaLAPACK(SVD svd)
{
  PetscErrorCode ierr;
  SVD_ScaLAPACK  *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,&ctx);CHKERRQ(ierr);
  svd->data = (void*)ctx;

  svd->ops->solve          = SVDSolve_ScaLAPACK;
  svd->ops->setup          = SVDSetUp_ScaLAPACK;
  svd->ops->destroy        = SVDDestroy_ScaLAPACK;
  svd->ops->reset          = SVDReset_ScaLAPACK;
  PetscFunctionReturn(0);
}

