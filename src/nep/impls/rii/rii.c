/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc nonlinear eigensolver: "rii"

   Method: Residual inverse iteration

   Algorithm:

       Simple residual inverse iteration with varying shift.

   References:

       [1] A. Neumaier, "Residual inverse iteration for the nonlinear
           eigenvalue problem", SIAM J. Numer. Anal. 22(5):914-923, 1985.
*/

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <../src/nep/impls/nepdefl.h>

typedef struct {
  PetscInt  max_inner_it;     /* maximum number of Newton iterations */
  PetscInt  lag;              /* interval to rebuild preconditioner */
  PetscBool cctol;            /* constant correction tolerance */
  KSP       ksp;              /* linear solver object */
} NEP_RII;

PetscErrorCode NEPSetUp_RII(NEP nep)
{
  PetscErrorCode ierr;
  PetscBool      istrivial;

  PetscFunctionBegin;
  if (nep->ncv) { ierr = PetscInfo(nep,"Setting ncv = nev, ignoring user-provided value\n");CHKERRQ(ierr); }
  nep->ncv = nep->nev;
  if (nep->mpd) { ierr = PetscInfo(nep,"Setting mpd = nev, ignoring user-provided value\n");CHKERRQ(ierr); }
  nep->mpd = nep->nev;
  if (!nep->max_it) nep->max_it = PetscMax(5000,2*nep->n/nep->ncv);
  if (nep->which && nep->which!=NEP_TARGET_MAGNITUDE) SETERRQ(PetscObjectComm((PetscObject)nep),1,"Wrong value of which");

  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver does not support region filtering");

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSolve_RII(NEP nep)
{
  PetscErrorCode     ierr;
  NEP_RII            *ctx = (NEP_RII*)nep->data;
  Mat                T,Tp,H;
  Vec                uu,u,r,delta;
  PetscScalar        lambda,sigma,a1,a2,corr,*Hp,*Ap;
  PetscReal          resnorm=1.0,ktol=0.1;
  PetscBool          skip=PETSC_FALSE;
  PetscInt           inner_its,its=0,ldh,ldds,i,j;
  NEP_EXT_OP         extop=NULL;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  /* get initial approximation of eigenvalue and eigenvector */
  ierr = NEPGetDefaultShift(nep,&sigma);CHKERRQ(ierr);
  lambda = sigma;
  if (!nep->nini) {
    ierr = BVSetRandomColumn(nep->V,0);CHKERRQ(ierr);
  }
  if (!ctx->ksp) { ierr = NEPRIIGetKSP(nep,&ctx->ksp);CHKERRQ(ierr); }
  ierr = NEPDeflationInitialize(nep,nep->V,ctx->ksp,PETSC_FALSE,nep->nev,&extop);CHKERRQ(ierr);
  ierr = NEPDeflationCreateVec(extop,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&delta);CHKERRQ(ierr);
  ierr = BVGetColumn(nep->V,0,&uu);CHKERRQ(ierr);
  ierr = NEPDeflationCopyToExtendedVec(extop,uu,NULL,u,PETSC_FALSE);CHKERRQ(ierr);
  ierr = BVRestoreColumn(nep->V,0,&uu);CHKERRQ(ierr);

  /* prepare linear solver */
  ierr = NEPDeflationSolveSetUp(extop,sigma);CHKERRQ(ierr);

  /* Restart loop */
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    its++;

    /* Use Newton's method to compute nonlinear Rayleigh functional. Current eigenvalue
       estimate as starting value. */
    inner_its=0;
    do {
      ierr = NEPDeflationComputeFunction(extop,lambda,&T);CHKERRQ(ierr);
      ierr = MatMult(T,u,r);CHKERRQ(ierr);
      ierr = VecDot(r,u,&a1);CHKERRQ(ierr);
      ierr = NEPDeflationComputeJacobian(extop,lambda,&Tp);CHKERRQ(ierr);
      ierr = MatMult(Tp,u,r);CHKERRQ(ierr);
      ierr = VecDot(r,u,&a2);CHKERRQ(ierr);
      corr = a1/a2;
      lambda = lambda - corr;
      inner_its++;
    } while (PetscAbsScalar(corr)>PETSC_SQRT_MACHINE_EPSILON && inner_its<ctx->max_inner_it);

    /* update preconditioner and set adaptive tolerance */
    if (ctx->lag && !(its%ctx->lag) && its>2*ctx->lag && resnorm<1e-2) {
      ierr = NEPDeflationSolveSetUp(extop,lambda+corr);CHKERRQ(ierr);
    }
    if (!ctx->cctol) {
      ktol = PetscMax(ktol/2.0,PETSC_MACHINE_EPSILON*10.0);
      ierr = KSPSetTolerances(ctx->ksp,ktol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    }

    /* form residual,  r = T(lambda)*u */
    ierr = NEPDeflationComputeFunction(extop,lambda,&T);CHKERRQ(ierr);
    ierr = MatMult(T,u,r);CHKERRQ(ierr);

    /* convergence test */
    ierr = VecNorm(r,NORM_2,&resnorm);CHKERRQ(ierr);
    ierr = (*nep->converged)(nep,lambda,0,resnorm,&nep->errest[nep->nconv],nep->convergedctx);CHKERRQ(ierr);
    nep->eigr[nep->nconv] = lambda;
    if (nep->errest[nep->nconv]<=nep->tol) {
      nep->nconv = nep->nconv + 1;
      ierr = NEPDeflationLocking(extop,u,lambda);CHKERRQ(ierr);
      nep->its += its;
      its = 0;
      skip = PETSC_TRUE;
    }
    ierr = (*nep->stopping)(nep,nep->its+its,nep->max_it,nep->nconv,nep->nev,&nep->reason,nep->stoppingctx);CHKERRQ(ierr);
    ierr = NEPMonitor(nep,nep->its+its,nep->nconv,nep->eigr,nep->eigi,nep->errest,nep->nconv+1);CHKERRQ(ierr);

    if (nep->reason == NEP_CONVERGED_ITERATING) {
      if (!skip) {
      /* eigenvector correction: delta = T(sigma)\r */
        ierr = NEPDeflationFunctionSolve(extop,r,delta);CHKERRQ(ierr);
        ierr = KSPGetConvergedReason(ctx->ksp,&kspreason);CHKERRQ(ierr);
        if (kspreason<0) {
          ierr = PetscInfo1(nep,"iter=%D, linear solve failed, stopping solve\n",nep->its);CHKERRQ(ierr);
          nep->reason = NEP_DIVERGED_LINEAR_SOLVE;
          break;
        }

        /* update eigenvector: u = u - delta */
        ierr = VecAXPY(u,-1.0,delta);CHKERRQ(ierr);

        /* normalize eigenvector */
        ierr = VecNormalize(u,NULL);CHKERRQ(ierr);
      } else {
        ierr = NEPDeflationSetRandomVec(extop,u);CHKERRQ(ierr);
        ierr = NEPDeflationSolveSetUp(extop,sigma);CHKERRQ(ierr);
        lambda = sigma;
        skip = PETSC_FALSE;
      }
    }
  }
  nep->its += its;
  ierr = NEPDeflationGetInvariantPair(extop,NULL,&H);CHKERRQ(ierr);
  ierr = MatGetSize(H,NULL,&ldh);CHKERRQ(ierr);
  ierr = DSSetType(nep->ds,DSNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(nep->ds,nep->nconv);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(nep->ds,&ldds);CHKERRQ(ierr);
  ierr = MatDenseGetArray(H,&Hp);CHKERRQ(ierr);
  ierr = DSGetArray(nep->ds,DS_MAT_A,&Ap);CHKERRQ(ierr);
  for (j=0;j<nep->nconv;j++)
    for (i=0;i<nep->nconv;i++) Ap[j*ldds+i] = Hp[j*ldh+i];
  ierr = DSRestoreArray(nep->ds,DS_MAT_A,&Ap);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(H,&Hp);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);
  ierr = DSSetDimensions(nep->ds,nep->nconv,0,0,nep->nconv);CHKERRQ(ierr);
  ierr = DSSolve(nep->ds,nep->eigr,nep->eigi);CHKERRQ(ierr);
  ierr = NEPDeflationReset(extop);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&delta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSetFromOptions_RII(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx = (NEP_RII*)nep->data;
  PetscBool      flg;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"NEP RII Options");CHKERRQ(ierr);

    ierr = PetscOptionsInt("-nep_rii_max_it","Maximum number of Newton iterations for updating Rayleigh functional","NEPRIISetMaximumIterations",ctx->max_inner_it,&ctx->max_inner_it,NULL);CHKERRQ(ierr);

    ierr = PetscOptionsBool("-nep_rii_const_correction_tol","Constant correction tolerance for the linear solver","NEPRIISetConstCorrectionTol",ctx->cctol,&ctx->cctol,NULL);CHKERRQ(ierr);

    i = 0;
    ierr = PetscOptionsInt("-nep_rii_lag_preconditioner","Interval to rebuild preconditioner","NEPRIISetLagPreconditioner",ctx->lag,&i,&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPRIISetLagPreconditioner(nep,i);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (!ctx->ksp) { ierr = NEPRIIGetKSP(nep,&ctx->ksp);CHKERRQ(ierr); }
  ierr = KSPSetFromOptions(ctx->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPRIISetMaximumIterations_RII(NEP nep,PetscInt its)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  ctx->max_inner_it = its;
  PetscFunctionReturn(0);
}

/*@
   NEPRIISetMaximumIterations - Sets the maximum number of inner iterations to be
   used in the RII solver. These are the Newton iterations related to the computation
   of the nonlinear Rayleigh functional.

   Logically Collective on NEP

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-  its - maximum inner iterations

   Level: advanced

.seealso: NEPRIIGetMaximumIterations()
@*/
PetscErrorCode NEPRIISetMaximumIterations(NEP nep,PetscInt its)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,its,2);
  ierr = PetscTryMethod(nep,"NEPRIISetMaximumIterations_C",(NEP,PetscInt),(nep,its));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPRIIGetMaximumIterations_RII(NEP nep,PetscInt *its)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  *its = ctx->max_inner_it;
  PetscFunctionReturn(0);
}

/*@
   NEPRIIGetMaximumIterations - Gets the maximum number of inner iterations of RII.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  its - maximum inner iterations

   Level: advanced

.seealso: NEPRIISetMaximumIterations()
@*/
PetscErrorCode NEPRIIGetMaximumIterations(NEP nep,PetscInt *its)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(its,2);
  ierr = PetscUseMethod(nep,"NEPRIIGetMaximumIterations_C",(NEP,PetscInt*),(nep,its));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPRIISetLagPreconditioner_RII(NEP nep,PetscInt lag)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  if (lag<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag must be non-negative");
  ctx->lag = lag;
  PetscFunctionReturn(0);
}

/*@
   NEPRIISetLagPreconditioner - Determines when the preconditioner is rebuilt in the
   nonlinear solve.

   Logically Collective on NEP

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-   lag - 0 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is
          computed within the nonlinear iteration, 2 means every second time
          the Jacobian is built, etc.

   Options Database Keys:
.  -nep_rii_lag_preconditioner <lag>

   Notes:
   The default is 1.
   The preconditioner is ALWAYS built in the first iteration of a nonlinear solve.

   Level: intermediate

.seealso: NEPRIIGetLagPreconditioner()
@*/
PetscErrorCode NEPRIISetLagPreconditioner(NEP nep,PetscInt lag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,lag,2);
  ierr = PetscTryMethod(nep,"NEPRIISetLagPreconditioner_C",(NEP,PetscInt),(nep,lag));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPRIIGetLagPreconditioner_RII(NEP nep,PetscInt *lag)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  *lag = ctx->lag;
  PetscFunctionReturn(0);
}

/*@
   NEPRIIGetLagPreconditioner - Indicates how often the preconditioner is rebuilt.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  lag - the lag parameter

   Level: intermediate

.seealso: NEPRIISetLagPreconditioner()
@*/
PetscErrorCode NEPRIIGetLagPreconditioner(NEP nep,PetscInt *lag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(lag,2);
  ierr = PetscUseMethod(nep,"NEPRIIGetLagPreconditioner_C",(NEP,PetscInt*),(nep,lag));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPRIISetConstCorrectionTol_RII(NEP nep,PetscBool cct)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  ctx->cctol = cct;
  PetscFunctionReturn(0);
}

/*@
   NEPRIISetConstCorrectionTol - Sets a flag to keep the tolerance used
   in the linear solver constant.

   Logically Collective on NEP

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-  cct - a boolean value

   Options Database Keys:
.  -nep_rii_const_correction_tol <bool> - set the boolean flag

   Notes:
   By default, an exponentially decreasing tolerance is set in the KSP used
   within the nonlinear iteration, so that each Newton iteration requests
   better accuracy than the previous one. The constant correction tolerance
   flag stops this behaviour.

   Level: intermediate

.seealso: NEPRIIGetConstCorrectionTol()
@*/
PetscErrorCode NEPRIISetConstCorrectionTol(NEP nep,PetscBool cct)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,cct,2);
  ierr = PetscTryMethod(nep,"NEPRIISetConstCorrectionTol_C",(NEP,PetscBool),(nep,cct));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPRIIGetConstCorrectionTol_RII(NEP nep,PetscBool *cct)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  *cct = ctx->cctol;
  PetscFunctionReturn(0);
}

/*@
   NEPRIIGetConstCorrectionTol - Returns the constant tolerance flag.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  cct - the value of the constant tolerance flag

   Level: intermediate

.seealso: NEPRIISetConstCorrectionTol()
@*/
PetscErrorCode NEPRIIGetConstCorrectionTol(NEP nep,PetscBool *cct)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(cct,2);
  ierr = PetscUseMethod(nep,"NEPRIIGetConstCorrectionTol_C",(NEP,PetscBool*),(nep,cct));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPRIISetKSP_RII(NEP nep,KSP ksp)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->ksp);CHKERRQ(ierr);
  ctx->ksp = ksp;
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp);CHKERRQ(ierr);
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPRIISetKSP - Associate a linear solver object (KSP) to the nonlinear
   eigenvalue solver.

   Collective on NEP

   Input Parameters:
+  nep - eigenvalue solver
-  ksp - the linear solver object

   Level: advanced

.seealso: NEPRIIGetKSP()
@*/
PetscErrorCode NEPRIISetKSP(NEP nep,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(nep,1,ksp,2);
  ierr = PetscTryMethod(nep,"NEPRIISetKSP_C",(NEP,KSP),(nep,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPRIIGetKSP_RII(NEP nep,KSP *ksp)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  if (!ctx->ksp) {
    ierr = KSPCreate(PetscObjectComm((PetscObject)nep),&ctx->ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)nep,1);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ctx->ksp,((PetscObject)nep)->prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(ctx->ksp,"nep_rii_");CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(ctx->ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ctx->ksp,SLEPC_DEFAULT_TOL,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  }
  *ksp = ctx->ksp;
  PetscFunctionReturn(0);
}

/*@
   NEPRIIGetKSP - Retrieve the linear solver object (KSP) associated with
   the nonlinear eigenvalue solver.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  ksp - the linear solver object

   Level: advanced

.seealso: NEPRIISetKSP()
@*/
PetscErrorCode NEPRIIGetKSP(NEP nep,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ksp,2);
  ierr = PetscUseMethod(nep,"NEPRIIGetKSP_C",(NEP,KSP*),(nep,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPView_RII(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx = (NEP_RII*)nep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!ctx->ksp) { ierr = NEPRIIGetKSP(nep,&ctx->ksp);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of inner iterations: %D\n",ctx->max_inner_it);CHKERRQ(ierr);
    if (ctx->cctol) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using a constant tolerance for the linear solver\n");CHKERRQ(ierr);
    }
    if (ctx->lag) {
      ierr = PetscViewerASCIIPrintf(viewer,"  updating the preconditioner every %D iterations\n",ctx->lag);CHKERRQ(ierr);
    }
    ierr = KSPView(ctx->ksp,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPReset_RII(NEP nep)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  ierr = KSPReset(ctx->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDestroy_RII(NEP nep)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  ierr = KSPDestroy(&ctx->ksp);CHKERRQ(ierr);
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIISetMaximumIterations_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIIGetMaximumIterations_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIISetLagPreconditioner_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIIGetLagPreconditioner_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIISetConstCorrectionTol_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIIGetConstCorrectionTol_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIISetKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIIGetKSP_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode NEPCreate_RII(NEP nep)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(nep,&ctx);CHKERRQ(ierr);
  nep->data = (void*)ctx;
  ctx->max_inner_it = 10;
  ctx->lag          = 1;
  ctx->cctol        = PETSC_FALSE;

  nep->ops->solve          = NEPSolve_RII;
  nep->ops->setup          = NEPSetUp_RII;
  nep->ops->setfromoptions = NEPSetFromOptions_RII;
  nep->ops->reset          = NEPReset_RII;
  nep->ops->destroy        = NEPDestroy_RII;
  nep->ops->view           = NEPView_RII;
  nep->ops->computevectors = NEPComputeVectors_Schur;

  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIISetMaximumIterations_C",NEPRIISetMaximumIterations_RII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIIGetMaximumIterations_C",NEPRIIGetMaximumIterations_RII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIISetLagPreconditioner_C",NEPRIISetLagPreconditioner_RII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIIGetLagPreconditioner_C",NEPRIIGetLagPreconditioner_RII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIISetConstCorrectionTol_C",NEPRIISetConstCorrectionTol_RII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIIGetConstCorrectionTol_C",NEPRIIGetConstCorrectionTol_RII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIISetKSP_C",NEPRIISetKSP_RII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIIGetKSP_C",NEPRIIGetKSP_RII);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

