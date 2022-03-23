/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "arnoldi"

   Method: Explicitly Restarted Arnoldi

   Algorithm:

       Arnoldi method with explicit restart and deflation.

   References:

       [1] "Arnoldi Methods in SLEPc", SLEPc Technical Report STR-4,
           available at https://slepc.upv.es.
*/

#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/

typedef struct {
  PetscBool delayed;
} EPS_ARNOLDI;

PetscErrorCode EPSSetUp_Arnoldi(EPS eps)
{
  PetscFunctionBegin;
  EPSCheckDefinite(eps);
  CHKERRQ(EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd));
  PetscCheck(eps->ncv<=eps->nev+eps->mpd,PetscObjectComm((PetscObject)eps),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nev+mpd");
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) CHKERRQ(EPSSetWhichEigenpairs_Default(eps));
  PetscCheck(eps->which!=EPS_ALL,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support computing all eigenvalues");
  EPSCheckUnsupported(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_TWOSIDED);

  CHKERRQ(EPSAllocateSolution(eps,1));
  CHKERRQ(EPS_SetInnerProduct(eps));
  CHKERRQ(DSSetType(eps->ds,DSNHEP));
  if (eps->extraction==EPS_REFINED || eps->extraction==EPS_REFINED_HARMONIC) CHKERRQ(DSSetRefined(eps->ds,PETSC_TRUE));
  CHKERRQ(DSSetExtraRow(eps->ds,PETSC_TRUE));
  CHKERRQ(DSAllocate(eps->ds,eps->ncv+1));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_Arnoldi(EPS eps)
{
  PetscInt           k,nv,ld;
  Mat                U,Op,H;
  PetscScalar        *Harray;
  PetscReal          beta,gamma=1.0;
  PetscBool          breakdown,harmonic,refined;
  BVOrthogRefineType orthog_ref;
  EPS_ARNOLDI        *arnoldi = (EPS_ARNOLDI*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(DSGetRefined(eps->ds,&refined));
  harmonic = (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC)?PETSC_TRUE:PETSC_FALSE;
  CHKERRQ(BVGetOrthogonalization(eps->V,NULL,&orthog_ref,NULL,NULL));

  /* Get the starting Arnoldi vector */
  CHKERRQ(EPSGetStartVector(eps,0,NULL));

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    CHKERRQ(DSSetDimensions(eps->ds,nv,eps->nconv,0));
    if (!arnoldi->delayed) {
      CHKERRQ(STGetOperator(eps->st,&Op));
      CHKERRQ(DSGetMat(eps->ds,DS_MAT_A,&H));
      CHKERRQ(BVMatArnoldi(eps->V,Op,H,eps->nconv,&nv,&beta,&breakdown));
      CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_A,&H));
      CHKERRQ(STRestoreOperator(eps->st,&Op));
    } else if (orthog_ref == BV_ORTHOG_REFINE_NEVER) {
      CHKERRQ(DSGetArray(eps->ds,DS_MAT_A,&Harray));
      CHKERRQ(EPSDelayedArnoldi1(eps,Harray,ld,eps->nconv,&nv,&beta,&breakdown));
      CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_A,&Harray));
    } else {
      CHKERRQ(DSGetArray(eps->ds,DS_MAT_A,&Harray));
      CHKERRQ(EPSDelayedArnoldi(eps,Harray,ld,eps->nconv,&nv,&beta,&breakdown));
      CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_A,&Harray));
    }
    CHKERRQ(DSSetState(eps->ds,DS_STATE_INTERMEDIATE));
    CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv,nv));

    /* Compute translation of Krylov decomposition if harmonic extraction used */
    if (harmonic) CHKERRQ(DSTranslateHarmonic(eps->ds,eps->target,beta,PETSC_FALSE,NULL,&gamma));

    /* Solve projected problem */
    CHKERRQ(DSSolve(eps->ds,eps->eigr,eps->eigi));
    CHKERRQ(DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL));
    CHKERRQ(DSUpdateExtraRow(eps->ds));
    CHKERRQ(DSSynchronize(eps->ds,eps->eigr,eps->eigi));

    /* Check convergence */
    CHKERRQ(EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,nv-eps->nconv,beta,0.0,gamma,&k));
    if (refined) {
      CHKERRQ(DSGetMat(eps->ds,DS_MAT_X,&U));
      CHKERRQ(BVMultInPlace(eps->V,U,eps->nconv,k+1));
      CHKERRQ(MatDestroy(&U));
      CHKERRQ(BVOrthonormalizeColumn(eps->V,k,PETSC_FALSE,NULL,NULL));
    } else {
      CHKERRQ(DSGetMat(eps->ds,DS_MAT_Q,&U));
      CHKERRQ(BVMultInPlace(eps->V,U,eps->nconv,PetscMin(k+1,nv)));
      CHKERRQ(MatDestroy(&U));
    }
    CHKERRQ((*eps->stopping)(eps,eps->its,eps->max_it,k,eps->nev,&eps->reason,eps->stoppingctx));
    if (eps->reason == EPS_CONVERGED_ITERATING && breakdown) {
      CHKERRQ(PetscInfo(eps,"Breakdown in Arnoldi method (it=%" PetscInt_FMT " norm=%g)\n",eps->its,(double)beta));
      CHKERRQ(EPSGetStartVector(eps,k,&breakdown));
      if (breakdown) {
        eps->reason = EPS_DIVERGED_BREAKDOWN;
        CHKERRQ(PetscInfo(eps,"Unable to generate more start vectors\n"));
      }
    }
    eps->nconv = k;
    CHKERRQ(EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv));
  }
  CHKERRQ(DSTruncate(eps->ds,eps->nconv,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_Arnoldi(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscBool      set,val;
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"EPS Arnoldi Options"));

    CHKERRQ(PetscOptionsBool("-eps_arnoldi_delayed","Use delayed reorthogonalization","EPSArnoldiSetDelayed",arnoldi->delayed,&val,&set));
    if (set) CHKERRQ(EPSArnoldiSetDelayed(eps,val));

  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSArnoldiSetDelayed_Arnoldi(EPS eps,PetscBool delayed)
{
  EPS_ARNOLDI *arnoldi = (EPS_ARNOLDI*)eps->data;

  PetscFunctionBegin;
  arnoldi->delayed = delayed;
  PetscFunctionReturn(0);
}

/*@
   EPSArnoldiSetDelayed - Activates or deactivates delayed reorthogonalization
   in the Arnoldi iteration.

   Logically Collective on eps

   Input Parameters:
+  eps - the eigenproblem solver context
-  delayed - boolean flag

   Options Database Key:
.  -eps_arnoldi_delayed - Activates delayed reorthogonalization in Arnoldi

   Note:
   Delayed reorthogonalization is an aggressive optimization for the Arnoldi
   eigensolver than may provide better scalability, but sometimes makes the
   solver converge less than the default algorithm.

   Level: advanced

.seealso: EPSArnoldiGetDelayed()
@*/
PetscErrorCode EPSArnoldiSetDelayed(EPS eps,PetscBool delayed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,delayed,2);
  CHKERRQ(PetscTryMethod(eps,"EPSArnoldiSetDelayed_C",(EPS,PetscBool),(eps,delayed)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSArnoldiGetDelayed_Arnoldi(EPS eps,PetscBool *delayed)
{
  EPS_ARNOLDI *arnoldi = (EPS_ARNOLDI*)eps->data;

  PetscFunctionBegin;
  *delayed = arnoldi->delayed;
  PetscFunctionReturn(0);
}

/*@
   EPSArnoldiGetDelayed - Gets the type of reorthogonalization used during the Arnoldi
   iteration.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  delayed - boolean flag indicating if delayed reorthogonalization has been enabled

   Level: advanced

.seealso: EPSArnoldiSetDelayed()
@*/
PetscErrorCode EPSArnoldiGetDelayed(EPS eps,PetscBool *delayed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidBoolPointer(delayed,2);
  CHKERRQ(PetscUseMethod(eps,"EPSArnoldiGetDelayed_C",(EPS,PetscBool*),(eps,delayed)));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_Arnoldi(EPS eps)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(eps->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSArnoldiSetDelayed_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSArnoldiGetDelayed_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_Arnoldi(EPS eps,PetscViewer viewer)
{
  PetscBool      isascii;
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii && arnoldi->delayed) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using delayed reorthogonalization\n"));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_Arnoldi(EPS eps)
{
  EPS_ARNOLDI    *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&ctx));
  eps->data = (void*)ctx;

  eps->useds = PETSC_TRUE;

  eps->ops->solve          = EPSSolve_Arnoldi;
  eps->ops->setup          = EPSSetUp_Arnoldi;
  eps->ops->setupsort      = EPSSetUpSort_Default;
  eps->ops->setfromoptions = EPSSetFromOptions_Arnoldi;
  eps->ops->destroy        = EPSDestroy_Arnoldi;
  eps->ops->view           = EPSView_Arnoldi;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->computevectors = EPSComputeVectors_Schur;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSArnoldiSetDelayed_C",EPSArnoldiSetDelayed_Arnoldi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSArnoldiGetDelayed_C",EPSArnoldiGetDelayed_Arnoldi));
  PetscFunctionReturn(0);
}
