/*

   SLEPc nonlinear eigensolver: "rii"

   Method: Residual inverse iteration

   Algorithm:

       Simple residual inverse iteration with varying shift.

   References:

       [1] A. Neumaier, "Residual inverse iteration for the nonlinear
           eigenvalue problem", SIAM J. Numer. Anal. 22(5):914-923, 1985.

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

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/

typedef struct {
  PetscInt  max_inner_it;     /* maximum number of Newton iterations */
  PetscInt  lag;              /* interval to rebuild preconditioner */
  PetscBool cctol;            /* constant correction tolerance */
  KSP       ksp;              /* linear solver object */
} NEP_RII;

#undef __FUNCT__
#define __FUNCT__ "NEPRII_KSPSolve"
PETSC_STATIC_INLINE PetscErrorCode NEPRII_KSPSolve(NEP nep,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       lits;
  NEP_RII        *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  ierr = KSPSolve(ctx->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ctx->ksp,&lits);CHKERRQ(ierr);
  ierr = PetscInfo2(nep,"iter=%D, linear solve iterations=%D\n",nep->its,lits);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_RII"
PetscErrorCode NEPSetUp_RII(NEP nep)
{
  PetscErrorCode ierr;
  PetscBool      istrivial;

  PetscFunctionBegin;
  if (nep->nev>1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Requested several eigenpairs but this solver can compute only one");
  if (nep->ncv) { ierr = PetscInfo(nep,"Setting ncv = 1, ignoring user-provided value\n");CHKERRQ(ierr); }
  nep->ncv = 1;
  if (nep->mpd) { ierr = PetscInfo(nep,"Setting mpd = 1, ignoring user-provided value\n");CHKERRQ(ierr); }
  nep->mpd = 1;
  if (!nep->max_it) nep->max_it = PetscMax(5000,2*nep->n/nep->ncv);
  if (nep->which && nep->which!=NEP_TARGET_MAGNITUDE) SETERRQ(PetscObjectComm((PetscObject)nep),1,"Wrong value of which");

  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver does not support region filtering");

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSolve_RII"
PetscErrorCode NEPSolve_RII(NEP nep)
{
  PetscErrorCode     ierr;
  NEP_RII            *ctx = (NEP_RII*)nep->data;
  Mat                T=nep->function,Tp=nep->jacobian,Tsigma;
  Vec                u,r=nep->work[0],delta=nep->work[1];
  PetscScalar        lambda,a1,a2,corr;
  PetscReal          resnorm=1.0,ktol=0.1;
  PetscBool          hascopy;
  PetscInt           inner_its;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  /* get initial approximation of eigenvalue and eigenvector */
  ierr = NEPGetDefaultShift(nep,&lambda);CHKERRQ(ierr);
  if (!nep->nini) {
    ierr = BVSetRandomColumn(nep->V,0);CHKERRQ(ierr);
  }
  ierr = BVGetColumn(nep->V,0,&u);CHKERRQ(ierr);
  ierr = NEPComputeFunction(nep,lambda,T,T);CHKERRQ(ierr);

  /* prepare linear solver */
  if (!ctx->ksp) { ierr = NEPRIIGetKSP(nep,&ctx->ksp);CHKERRQ(ierr); }
  ierr = MatDuplicate(T,MAT_COPY_VALUES,&Tsigma);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->ksp,Tsigma,Tsigma);CHKERRQ(ierr);

  /* Restart loop */
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;

    /* Use Newton's method to compute nonlinear Rayleigh functional. Current eigenvalue 
       estimate as starting value. */
    inner_its=0;
    do {
      ierr = NEPApplyFunction(nep,lambda,u,delta,r,T,T);CHKERRQ(ierr);
      ierr = VecDot(r,u,&a1);CHKERRQ(ierr);
      ierr = NEPApplyJacobian(nep,lambda,u,delta,r,Tp);CHKERRQ(ierr);
      ierr = VecDot(r,u,&a2);CHKERRQ(ierr);
      corr = a1/a2;
      lambda = lambda - corr;
      inner_its++;
    } while (PetscAbsScalar(corr)>PETSC_SQRT_MACHINE_EPSILON && inner_its<ctx->max_inner_it);

    /* update preconditioner and set adaptive tolerance */
    if (ctx->lag && !(nep->its%ctx->lag) && nep->its>2*ctx->lag && resnorm<1e-2) {
      ierr = MatHasOperation(T,MATOP_COPY,&hascopy);CHKERRQ(ierr);
      if (hascopy) {
        ierr = MatCopy(T,Tsigma,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatDestroy(&Tsigma);CHKERRQ(ierr);
        ierr = MatDuplicate(T,MAT_COPY_VALUES,&Tsigma);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(ctx->ksp,Tsigma,Tsigma);CHKERRQ(ierr);
    }
    if (!ctx->cctol) {
      ktol = PetscMax(ktol/2.0,PETSC_MACHINE_EPSILON*10.0);
      ierr = KSPSetTolerances(ctx->ksp,ktol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    }

    /* form residual,  r = T(lambda)*u */
    ierr = NEPApplyFunction(nep,lambda,u,delta,r,T,T);CHKERRQ(ierr);

    /* convergence test */
    ierr = VecNorm(r,NORM_2,&resnorm);CHKERRQ(ierr);
    ierr = (*nep->converged)(nep,lambda,0,resnorm,&nep->errest[nep->nconv],nep->convergedctx);CHKERRQ(ierr);
    nep->eigr[nep->nconv] = lambda;
    if (nep->errest[nep->nconv]<=nep->tol) {
      nep->nconv = nep->nconv + 1;
    }
    ierr = (*nep->stopping)(nep,nep->its,nep->max_it,nep->nconv,nep->nev,&nep->reason,nep->stoppingctx);CHKERRQ(ierr);
    ierr = NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->eigi,nep->errest,1);CHKERRQ(ierr);

    if (nep->reason == NEP_CONVERGED_ITERATING) {
      /* eigenvector correction: delta = T(sigma)\r */
      ierr = NEPRII_KSPSolve(nep,r,delta);CHKERRQ(ierr);
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
    }
  }
  ierr = MatDestroy(&Tsigma);CHKERRQ(ierr);
  ierr = BVRestoreColumn(nep->V,0,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetFromOptions_RII"
PetscErrorCode NEPSetFromOptions_RII(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx = (NEP_RII*)nep->data;
  PetscBool      flg;
  PetscInt       i;

  PetscFunctionBegin;
  if (!ctx->ksp) { ierr = NEPRIIGetKSP(nep,&ctx->ksp);CHKERRQ(ierr); }
  ierr = KSPSetOperators(ctx->ksp,nep->function,nep->function_pre);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ctx->ksp);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"NEP RII Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_rii_max_it","Maximum number of Newton iterations for updating Rayleigh functional","NEPRIISetMaximumIterations",ctx->max_inner_it,&ctx->max_inner_it,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-nep_rii_const_correction_tol","Constant correction tolerance for the linear solver","NEPRIISetConstCorrectionTol",ctx->cctol,&ctx->cctol,NULL);CHKERRQ(ierr);
    i = 0;
    ierr = PetscOptionsInt("-nep_rii_lag_preconditioner","Interval to rebuild preconditioner","NEPRIISetLagPreconditioner",ctx->lag,&i,&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPRIISetLagPreconditioner(nep,i);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRIISetMaximumIterations_RII"
static PetscErrorCode NEPRIISetMaximumIterations_RII(NEP nep,PetscInt its)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  ctx->max_inner_it = its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRIISetMaximumIterations"
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

#undef __FUNCT__
#define __FUNCT__ "NEPRIIGetMaximumIterations_RII"
static PetscErrorCode NEPRIIGetMaximumIterations_RII(NEP nep,PetscInt *its)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  *its = ctx->max_inner_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRIIGetMaximumIterations"
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

#undef __FUNCT__
#define __FUNCT__ "NEPRIISetLagPreconditioner_RII"
static PetscErrorCode NEPRIISetLagPreconditioner_RII(NEP nep,PetscInt lag)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  if (lag<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag must be non-negative");
  ctx->lag = lag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRIISetLagPreconditioner"
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

#undef __FUNCT__
#define __FUNCT__ "NEPRIIGetLagPreconditioner_RII"
static PetscErrorCode NEPRIIGetLagPreconditioner_RII(NEP nep,PetscInt *lag)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  *lag = ctx->lag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRIIGetLagPreconditioner"
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

#undef __FUNCT__
#define __FUNCT__ "NEPRIISetConstCorrectionTol_RII"
static PetscErrorCode NEPRIISetConstCorrectionTol_RII(NEP nep,PetscBool cct)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  ctx->cctol = cct;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRIISetConstCorrectionTol"
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

#undef __FUNCT__
#define __FUNCT__ "NEPRIIGetConstCorrectionTol_RII"
static PetscErrorCode NEPRIIGetConstCorrectionTol_RII(NEP nep,PetscBool *cct)
{
  NEP_RII *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  *cct = ctx->cctol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRIIGetConstCorrectionTol"
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

#undef __FUNCT__
#define __FUNCT__ "NEPRIISetKSP_RII"
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

#undef __FUNCT__
#define __FUNCT__ "NEPRIISetKSP"
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

#undef __FUNCT__
#define __FUNCT__ "NEPRIIGetKSP_RII"
static PetscErrorCode NEPRIIGetKSP_RII(NEP nep,KSP *ksp)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  if (!ctx->ksp) {
    ierr = KSPCreate(PetscObjectComm((PetscObject)nep),&ctx->ksp);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ctx->ksp,((PetscObject)nep)->prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(ctx->ksp,"nep_rii_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)nep,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(ctx->ksp,PETSC_TRUE);CHKERRQ(ierr);
  }
  *ksp = ctx->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRIIGetKSP"
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

#undef __FUNCT__
#define __FUNCT__ "NEPView_RII"
PetscErrorCode NEPView_RII(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx = (NEP_RII*)nep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!ctx->ksp) { ierr = NEPRIIGetKSP(nep,&ctx->ksp);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPrintf(viewer,"  RII: maximum number of inner iterations: %D\n",ctx->max_inner_it);CHKERRQ(ierr);
    if (ctx->cctol) {
      ierr = PetscViewerASCIIPrintf(viewer,"  RII: using a constant tolerance for the linear solver\n");CHKERRQ(ierr);
    }
    if (ctx->lag) {
      ierr = PetscViewerASCIIPrintf(viewer,"  RII: updating the preconditioner every %D iterations\n",ctx->lag);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(ctx->ksp,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy_RII"
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

#undef __FUNCT__
#define __FUNCT__ "NEPCreate_RII"
PETSC_EXTERN PetscErrorCode NEPCreate_RII(NEP nep)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(nep,&ctx);CHKERRQ(ierr);
  ctx->max_inner_it = 10;
  ctx->lag          = 1;
  ctx->cctol        = PETSC_FALSE;
  nep->data = (void*)ctx;

  nep->ops->solve          = NEPSolve_RII;
  nep->ops->setup          = NEPSetUp_RII;
  nep->ops->setfromoptions = NEPSetFromOptions_RII;
  nep->ops->destroy        = NEPDestroy_RII;
  nep->ops->view           = NEPView_RII;
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

