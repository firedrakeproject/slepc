/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2017, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "lyapii"

   Method: Lyapunov inverse iteration

   Algorithm:

       Lyapunov inverse iteration using LME solvers

   References:

       [1] H.C. Elman and M. Wu, "Lyapunov inverse iteration for computing a
           few rightmost eigenvalues of large generalized eigenvalue problems",
           SIAM J. Matrix Anal. Appl. 34(4):1685-1707, 2013.
*/

#include <slepc/private/epsimpl.h>          /*I "slepceps.h" I*/
#include <slepc/private/lmeimpl.h>          /*I "slepclme.h" I*/

typedef struct {
  LME lme;
} EPS_LYAPII;

PetscErrorCode EPSSetUp_LyapII(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;
  PetscBool      issinv,istrivial;

  PetscFunctionBegin;
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(PetscObjectComm((PetscObject)eps),1,"The value of ncv must be at least nev");
  } else eps->ncv = eps->nev;
  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (!eps->max_it) eps->max_it = PetscMax(1000*eps->nev,100*eps->n);
  if (!eps->which) eps->which=EPS_LARGEST_REAL;
  if (eps->which!=EPS_LARGEST_REAL) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");
  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }
  if (eps->balance!=EPS_BALANCE_NONE) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Balancing not supported in this solver");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support region filtering");

  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSINVERT,&issinv);CHKERRQ(ierr);
  if (!issinv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Must use STSINVERT spectral transformation");

  if (!ctx->lme) { ierr = EPSLyapIIGetLME(eps,&ctx->lme);CHKERRQ(ierr); }
  ierr = LMESetProblemType(ctx->lme,LME_LYAPUNOV);CHKERRQ(ierr);
  ierr = LMESetErrorIfNotConverged(ctx->lme,PETSC_TRUE);CHKERRQ(ierr);

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,2);CHKERRQ(ierr);
  ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(eps->ds,eps->nev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_LyapII(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;
  PetscInt       k,ld;
  Vec            v,y,e;
  Mat            S;
  PetscReal      relerr,norm;
  PetscScalar    theta,*T;
  PetscBool      breakdown;

  PetscFunctionBegin;
  y = eps->work[1];
  e = eps->work[0];

  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = EPSGetStartVector(eps,0,NULL);CHKERRQ(ierr);
  ierr = STGetOperator(eps->st,&S);CHKERRQ(ierr);
  ierr = LMESetCoefficients(ctx->lme,S,NULL,NULL,NULL);CHKERRQ(ierr);

  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;
    k = eps->nconv;

    /* Y = lyap(S,2*S*Z*S') */

    //ierr = LMESetRHS(ctx->lme,C);CHKERRQ(ierr);
    ierr = LMESolve(ctx->lme);CHKERRQ(ierr);

    ierr = BVGetColumn(eps->V,k,&v);CHKERRQ(ierr);
    ierr = STApply(eps->st,v,y);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,k,&v);CHKERRQ(ierr);

    /* purge previously converged eigenvectors */
    ierr = DSGetArray(eps->ds,DS_MAT_A,&T);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(eps->V,0,k);CHKERRQ(ierr);
    ierr = BVOrthogonalizeVec(eps->V,y,T+k*ld,&norm,NULL);CHKERRQ(ierr);

    /* theta = (v,y)_B */
    ierr = BVSetActiveColumns(eps->V,k,k+1);CHKERRQ(ierr);
    ierr = BVDotVec(eps->V,y,&theta);CHKERRQ(ierr);
    T[k+k*ld] = theta;
    ierr = DSRestoreArray(eps->ds,DS_MAT_A,&T);CHKERRQ(ierr);

    relerr = 0.0;
    eps->errest[eps->nconv] = relerr;

    /* normalize */
    ierr = BVInsertVec(eps->V,k,y);CHKERRQ(ierr);

    /* if relerr<tol, accept eigenpair */
    if (relerr<eps->tol) {
      eps->nconv = eps->nconv + 1;
      if (eps->nconv<eps->nev) {
        ierr = EPSGetStartVector(eps,eps->nconv,&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
          break;
        }
      }
    }
    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nconv+1);CHKERRQ(ierr);
    ierr = (*eps->stopping)(eps,eps->its,eps->max_it,eps->nconv,eps->nev,&eps->reason,eps->stoppingctx);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&S);CHKERRQ(ierr);

  ierr = DSSetDimensions(eps->ds,eps->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_LyapII(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS Lyapunov Inverse Iteration Options");CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (!ctx->lme) { ierr = EPSLyapIIGetLME(eps,&ctx->lme);CHKERRQ(ierr); }
  ierr = LMESetFromOptions(ctx->lme);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLyapIISetLME_LyapII(EPS eps,LME lme)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)lme);CHKERRQ(ierr);
  ierr = LMEDestroy(&ctx->lme);CHKERRQ(ierr);
  ctx->lme = lme;
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->lme);CHKERRQ(ierr);
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSLyapIISetLME - Associate a linear matrix equation solver object (LME) to the
   eigenvalue solver.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  lme - the linear matrix equation solver object

   Level: advanced

.seealso: EPSLyapIIGetLME()
@*/
PetscErrorCode EPSLyapIISetLME(EPS eps,LME lme)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidHeaderSpecific(lme,LME_CLASSID,2);
  PetscCheckSameComm(eps,1,lme,2);
  ierr = PetscTryMethod(eps,"EPSLyapIISetLME_C",(EPS,LME),(eps,lme));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLyapIIGetLME_LyapII(EPS eps,LME *lme)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  if (!ctx->lme) {
    ierr = LMECreate(PetscObjectComm((PetscObject)eps),&ctx->lme);CHKERRQ(ierr);
    ierr = LMESetOptionsPrefix(ctx->lme,((PetscObject)eps)->prefix);CHKERRQ(ierr);
    ierr = LMEAppendOptionsPrefix(ctx->lme,"eps_lyapii_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->lme,(PetscObject)eps,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->lme);CHKERRQ(ierr);
  }
  *lme = ctx->lme;
  PetscFunctionReturn(0);
}

/*@
   EPSLyapIIGetLME - Retrieve the linear matrix equation solver object (LME)
   associated with the eigenvalue solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  lme - the linear matrix equation solver object

   Level: advanced

.seealso: EPSLyapIISetLME()
@*/
PetscErrorCode EPSLyapIIGetLME(EPS eps,LME *lme)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(lme,2);
  ierr = PetscUseMethod(eps,"EPSLyapIIGetLME_C",(EPS,LME*),(eps,lme));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_LyapII(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!ctx->lme) { ierr = EPSLyapIIGetLME(eps,&ctx->lme);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = LMEView(ctx->lme,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_LyapII(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  if (!ctx->lme) { ierr = LMEReset(ctx->lme);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_LyapII(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  ierr = LMEDestroy(&ctx->lme);CHKERRQ(ierr);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIISetLME_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIIGetLME_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetDefaultST_LyapII(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STSINVERT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode EPSCreate_LyapII(EPS eps)
{
  EPS_LYAPII     *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  eps->ops->solve          = EPSSolve_LyapII;
  eps->ops->setup          = EPSSetUp_LyapII;
  eps->ops->setfromoptions = EPSSetFromOptions_LyapII;
  eps->ops->reset          = EPSReset_LyapII;
  eps->ops->destroy        = EPSDestroy_LyapII;
  eps->ops->view           = EPSView_LyapII;
  eps->ops->computevectors = EPSComputeVectors_Schur;
  eps->ops->setdefaultst   = EPSSetDefaultST_LyapII;

  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIISetLME_C",EPSLyapIISetLME_LyapII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIIGetLME_C",EPSLyapIIGetLME_LyapII);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

