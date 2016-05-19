/*

   SLEPc nonlinear eigensolver: "slp"

   Method: Succesive linear problems

   Algorithm:

       Newton-type iteration based on first order Taylor approximation.

   References:

       [1] A. Ruhe, "Algorithms for the nonlinear eigenvalue problem", SIAM J.
           Numer. Anal. 10(4):674-689, 1973.

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
  EPS       eps;             /* linear eigensolver for T*z = mu*Tp*z */
} NEP_SLP;

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_SLP"
PetscErrorCode NEPSetUp_SLP(NEP nep)
{
  PetscErrorCode ierr;
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;
  ST             st;
  PetscBool      istrivial;

  PetscFunctionBegin;
  if (nep->nev>1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Requested several eigenpairs but this solver can compute only one");
  if (nep->ncv) { ierr = PetscInfo(nep,"Setting ncv = 1, ignoring user-provided value\n");CHKERRQ(ierr); }
  nep->ncv = 1;
  if (nep->mpd) { ierr = PetscInfo(nep,"Setting mpd = 1, ignoring user-provided value\n");CHKERRQ(ierr); }
  nep->mpd = 1;
  if (nep->ncv>nep->nev+nep->mpd) SETERRQ(PetscObjectComm((PetscObject)nep),1,"The value of ncv must not be larger than nev+mpd");
  if (!nep->max_it) nep->max_it = PetscMax(5000,2*nep->n/nep->ncv);
  if (nep->which && nep->which!=NEP_TARGET_MAGNITUDE) SETERRQ(PetscObjectComm((PetscObject)nep),1,"Wrong value of which");

  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver does not support region filtering");

  if (!ctx->eps) { ierr = NEPSLPGetEPS(nep,&ctx->eps);CHKERRQ(ierr); }
  ierr = EPSSetWhichEigenpairs(ctx->eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
  ierr = EPSSetTarget(ctx->eps,0.0);CHKERRQ(ierr);
  ierr = EPSGetST(ctx->eps,&st);CHKERRQ(ierr);
  ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);
  ierr = EPSSetTolerances(ctx->eps,nep->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL/10.0:nep->tol/10.0,nep->max_it?nep->max_it:PETSC_DEFAULT);CHKERRQ(ierr);

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSolve_SLP"
PetscErrorCode NEPSolve_SLP(NEP nep)
{
  PetscErrorCode ierr;
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;
  Mat            T=nep->function,Tp=nep->jacobian;
  Vec            u,r=nep->work[0];
  PetscScalar    lambda,mu,im;
  PetscReal      resnorm;
  PetscInt       nconv;

  PetscFunctionBegin;
  /* get initial approximation of eigenvalue and eigenvector */
  ierr = NEPGetDefaultShift(nep,&lambda);CHKERRQ(ierr);
  if (!nep->nini) {
    ierr = BVSetRandomColumn(nep->V,0);CHKERRQ(ierr);
  }
  ierr = BVGetColumn(nep->V,0,&u);CHKERRQ(ierr);

  /* Restart loop */
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;

    /* evaluate T(lambda) and T'(lambda) */
    ierr = NEPComputeFunction(nep,lambda,T,T);CHKERRQ(ierr);
    ierr = NEPComputeJacobian(nep,lambda,Tp);CHKERRQ(ierr);

    /* form residual,  r = T(lambda)*u (used in convergence test only) */
    ierr = MatMult(T,u,r);CHKERRQ(ierr);

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
      /* compute eigenvalue correction mu and eigenvector approximation u */
      ierr = EPSSetOperators(ctx->eps,T,Tp);CHKERRQ(ierr);
      ierr = EPSSetInitialSpace(ctx->eps,1,&u);CHKERRQ(ierr);
      ierr = EPSSolve(ctx->eps);CHKERRQ(ierr);
      ierr = EPSGetConverged(ctx->eps,&nconv);CHKERRQ(ierr);
      if (!nconv) {
        ierr = PetscInfo1(nep,"iter=%D, inner iteration failed, stopping solve\n",nep->its);CHKERRQ(ierr);
        nep->reason = NEP_DIVERGED_LINEAR_SOLVE;
        break;
      }
      ierr = EPSGetEigenpair(ctx->eps,0,&mu,&im,u,NULL);CHKERRQ(ierr);
      if (PetscAbsScalar(im)>PETSC_MACHINE_EPSILON) SETERRQ(PetscObjectComm((PetscObject)nep),1,"Complex eigenvalue approximation - not implemented in real scalars");

      /* correct eigenvalue */
      lambda = lambda - mu;
    }
  }
  ierr = BVRestoreColumn(nep->V,0,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetFromOptions_SLP"
PetscErrorCode NEPSetFromOptions_SLP(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscErrorCode ierr;
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"NEP SLP Options");CHKERRQ(ierr);
  if (!ctx->eps) { ierr = NEPSLPGetEPS(nep,&ctx->eps);CHKERRQ(ierr); }
  ierr = EPSSetFromOptions(ctx->eps);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSLPSetEPS_SLP"
static PetscErrorCode NEPSLPSetEPS_SLP(NEP nep,EPS eps)
{
  PetscErrorCode ierr;
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(&ctx->eps);CHKERRQ(ierr);
  ctx->eps = eps;
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->eps);CHKERRQ(ierr);
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSLPSetEPS"
/*@
   NEPSLPSetEPS - Associate a linear eigensolver object (EPS) to the
   nonlinear eigenvalue solver.

   Collective on NEP

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-  eps - the eigensolver object

   Level: advanced

.seealso: NEPSLPGetEPS()
@*/
PetscErrorCode NEPSLPSetEPS(NEP nep,EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(nep,1,eps,2);
  ierr = PetscTryMethod(nep,"NEPSLPSetEPS_C",(NEP,EPS),(nep,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSLPGetEPS_SLP"
static PetscErrorCode NEPSLPGetEPS_SLP(NEP nep,EPS *eps)
{
  PetscErrorCode ierr;
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;
  ST             st;

  PetscFunctionBegin;
  if (!ctx->eps) {
    ierr = EPSCreate(PetscObjectComm((PetscObject)nep),&ctx->eps);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(ctx->eps,((PetscObject)nep)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(ctx->eps,"nep_slp_");CHKERRQ(ierr);
    ierr = EPSGetST(ctx->eps,&st);CHKERRQ(ierr);
    ierr = STSetOptionsPrefix(st,((PetscObject)ctx->eps)->prefix);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->eps,(PetscObject)nep,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->eps);CHKERRQ(ierr);
  }
  *eps = ctx->eps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSLPGetEPS"
/*@
   NEPSLPGetEPS - Retrieve the linear eigensolver object (EPS) associated
   to the nonlinear eigenvalue solver.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: NEPSLPSetEPS()
@*/
PetscErrorCode NEPSLPGetEPS(NEP nep,EPS *eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(eps,2);
  ierr = PetscUseMethod(nep,"NEPSLPGetEPS_C",(NEP,EPS*),(nep,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPView_SLP"
PetscErrorCode NEPView_SLP(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!ctx->eps) { ierr = NEPSLPGetEPS(nep,&ctx->eps);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = EPSView(ctx->eps,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPReset_SLP"
PetscErrorCode NEPReset_SLP(NEP nep)
{
  PetscErrorCode ierr;
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  if (!ctx->eps) { ierr = EPSReset(ctx->eps);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy_SLP"
PetscErrorCode NEPDestroy_SLP(NEP nep)
{
  PetscErrorCode ierr;
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(&ctx->eps);CHKERRQ(ierr);
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPSLPSetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPSLPGetEPS_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCreate_SLP"
PETSC_EXTERN PetscErrorCode NEPCreate_SLP(NEP nep)
{
  PetscErrorCode ierr;
  NEP_SLP        *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(nep,&ctx);CHKERRQ(ierr);
  nep->data = (void*)ctx;

  nep->ops->solve          = NEPSolve_SLP;
  nep->ops->setup          = NEPSetUp_SLP;
  nep->ops->setfromoptions = NEPSetFromOptions_SLP;
  nep->ops->reset          = NEPReset_SLP;
  nep->ops->destroy        = NEPDestroy_SLP;
  nep->ops->view           = NEPView_SLP;
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPSLPSetEPS_C",NEPSLPSetEPS_SLP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPSLPGetEPS_C",NEPSLPGetEPS_SLP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

