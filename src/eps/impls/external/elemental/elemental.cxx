/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to eigensolvers in Elemental.
*/

#include <slepc/private/epsimpl.h>    /*I "slepceps.h" I*/
#include <petsc/private/petscelemental.h>

typedef struct {
  Mat Ae,Be;        /* converted matrices */
} EPS_Elemental;

PetscErrorCode EPSSetUp_Elemental(EPS eps)
{
  PetscErrorCode ierr;
  EPS_Elemental  *ctx = (EPS_Elemental*)eps->data;
  Mat            A,B;
  PetscInt       nmat;
  PetscBool      isshift;
  PetscScalar    shift;

  PetscFunctionBegin;
  EPSCheckHermitianDefinite(eps);
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift);CHKERRQ(ierr);
  if (!isshift) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support spectral transformations");
  eps->ncv = eps->n;
  if (eps->mpd!=PETSC_DEFAULT) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = 1;
  if (!eps->which) { ierr = EPSSetWhichEigenpairs_Default(eps);CHKERRQ(ierr); }
  if (eps->which==EPS_ALL && eps->inta!=eps->intb) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support interval computation");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_STOPPING);
  EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION | EPS_FEATURE_CONVERGENCE);
  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);

  /* convert matrices */
  ierr = MatDestroy(&ctx->Ae);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->Be);CHKERRQ(ierr);
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetMatrix(eps->st,0,&A);CHKERRQ(ierr);
  ierr = MatConvert(A,MATELEMENTAL,MAT_INITIAL_MATRIX,&ctx->Ae);CHKERRQ(ierr);
  if (nmat>1) {
    ierr = STGetMatrix(eps->st,1,&B);CHKERRQ(ierr);
    ierr = MatConvert(B,MATELEMENTAL,MAT_INITIAL_MATRIX,&ctx->Be);CHKERRQ(ierr);
  }
  ierr = STGetShift(eps->st,&shift);CHKERRQ(ierr);
  if (shift != 0.0) {
    if (nmat>1) {
      ierr = MatAXPY(ctx->Ae,-shift,ctx->Be,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    } else {
      ierr = MatShift(ctx->Ae,-shift);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_Elemental(EPS eps)
{
  PetscErrorCode ierr;
  EPS_Elemental  *ctx = (EPS_Elemental*)eps->data;
  Mat            A = ctx->Ae,B = ctx->Be,Q,V;
  Mat_Elemental  *a = (Mat_Elemental*)A->data,*b,*q;
  PetscInt       i,rrank,ridx,erow;

  PetscFunctionBegin;
  El::DistMatrix<PetscReal,El::VR,El::STAR> w(*a->grid);
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Q);CHKERRQ(ierr);
  q = (Mat_Elemental*)Q->data;

  if (B) {
    b = (Mat_Elemental*)B->data;
    El::HermitianGenDefEig(El::AXBX,El::LOWER,*a->emat,*b->emat,w,*q->emat);
  } else El::HermitianEig(El::LOWER,*a->emat,w,*q->emat);

  for (i=0;i<eps->ncv;i++) {
    P2RO(A,0,i,&rrank,&ridx);
    RO2E(A,0,rrank,ridx,&erow);
    eps->eigr[i] = w.Get(erow,0);
  }
  ierr = BVGetMat(eps->V,&V);CHKERRQ(ierr);
  ierr = MatConvert(Q,MATDENSE,MAT_REUSE_MATRIX,&V);CHKERRQ(ierr);
  ierr = BVRestoreMat(eps->V,&V);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);

  eps->nconv  = eps->ncv;
  eps->its    = 1;
  eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_Elemental(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_Elemental(EPS eps)
{
  PetscErrorCode ierr;
  EPS_Elemental  *ctx = (EPS_Elemental*)eps->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&ctx->Ae);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->Be);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_Elemental(EPS eps)
{
  EPS_Elemental  *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  eps->categ = EPS_CATEGORY_OTHER;

  eps->ops->solve          = EPSSolve_Elemental;
  eps->ops->setup          = EPSSetUp_Elemental;
  eps->ops->setupsort      = EPSSetUpSort_Basic;
  eps->ops->destroy        = EPSDestroy_Elemental;
  eps->ops->reset          = EPSReset_Elemental;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->setdefaultst   = EPSSetDefaultST_NoFactor;
  PetscFunctionReturn(0);
}

