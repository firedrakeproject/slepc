/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
  EPS_Elemental  *ctx = (EPS_Elemental*)eps->data;
  Mat            A,B;
  PetscInt       nmat;
  PetscBool      isshift;
  PetscScalar    shift;

  PetscFunctionBegin;
  EPSCheckHermitianDefinite(eps);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift));
  PetscCheck(isshift,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support spectral transformations");
  eps->ncv = eps->n;
  if (eps->mpd!=PETSC_DEFAULT) CHKERRQ(PetscInfo(eps,"Warning: parameter mpd ignored\n"));
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = 1;
  if (!eps->which) CHKERRQ(EPSSetWhichEigenpairs_Default(eps));
  PetscCheck(eps->which!=EPS_ALL || eps->inta==eps->intb,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support interval computation");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_STOPPING);
  EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION | EPS_FEATURE_CONVERGENCE);
  CHKERRQ(EPSAllocateSolution(eps,0));

  /* convert matrices */
  CHKERRQ(MatDestroy(&ctx->Ae));
  CHKERRQ(MatDestroy(&ctx->Be));
  CHKERRQ(STGetNumMatrices(eps->st,&nmat));
  CHKERRQ(STGetMatrix(eps->st,0,&A));
  CHKERRQ(MatConvert(A,MATELEMENTAL,MAT_INITIAL_MATRIX,&ctx->Ae));
  if (nmat>1) {
    CHKERRQ(STGetMatrix(eps->st,1,&B));
    CHKERRQ(MatConvert(B,MATELEMENTAL,MAT_INITIAL_MATRIX,&ctx->Be));
  }
  CHKERRQ(STGetShift(eps->st,&shift));
  if (shift != 0.0) {
    if (nmat>1) CHKERRQ(MatAXPY(ctx->Ae,-shift,ctx->Be,SAME_NONZERO_PATTERN));
    else CHKERRQ(MatShift(ctx->Ae,-shift));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_Elemental(EPS eps)
{
  EPS_Elemental  *ctx = (EPS_Elemental*)eps->data;
  Mat            A = ctx->Ae,B = ctx->Be,Q,V;
  Mat_Elemental  *a = (Mat_Elemental*)A->data,*b,*q;
  PetscInt       i,rrank,ridx,erow;

  PetscFunctionBegin;
  El::DistMatrix<PetscReal,El::VR,El::STAR> w(*a->grid);
  CHKERRQ(MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Q));
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
  CHKERRQ(BVGetMat(eps->V,&V));
  CHKERRQ(MatConvert(Q,MATDENSE,MAT_REUSE_MATRIX,&V));
  CHKERRQ(BVRestoreMat(eps->V,&V));
  CHKERRQ(MatDestroy(&Q));

  eps->nconv  = eps->ncv;
  eps->its    = 1;
  eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_Elemental(EPS eps)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(eps->data));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_Elemental(EPS eps)
{
  EPS_Elemental  *ctx = (EPS_Elemental*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&ctx->Ae));
  CHKERRQ(MatDestroy(&ctx->Be));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_Elemental(EPS eps)
{
  EPS_Elemental  *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&ctx));
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
