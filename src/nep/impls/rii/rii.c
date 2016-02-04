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
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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
  PetscInt max_inner_it;         /* maximum number of Newton iterations */
} NEP_RII;

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_RII"
PetscErrorCode NEPSetUp_RII(NEP nep)
{
  PetscErrorCode ierr;
  PetscBool      istrivial;

  PetscFunctionBegin;
  if (nep->ncv) { /* ncv set */
    if (nep->ncv<nep->nev) SETERRQ(PetscObjectComm((PetscObject)nep),1,"The value of ncv must be at least nev");
  } else if (nep->mpd) { /* mpd set */
    nep->ncv = PetscMin(nep->n,nep->nev+nep->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (nep->nev<500) nep->ncv = PetscMin(nep->n,PetscMax(2*nep->nev,nep->nev+15));
    else {
      nep->mpd = 500;
      nep->ncv = PetscMin(nep->n,nep->nev+nep->mpd);
    }
  }
  if (!nep->mpd) nep->mpd = nep->ncv;
  if (nep->ncv>nep->nev+nep->mpd) SETERRQ(PetscObjectComm((PetscObject)nep),1,"The value of ncv must not be larger than nev+mpd");
  if (nep->nev>1) { ierr = PetscInfo(nep,"Warning: requested more than one eigenpair but RII can only compute one\n");CHKERRQ(ierr); }
  if (!nep->max_it) nep->max_it = PetscMax(5000,2*nep->n/nep->ncv);

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
  PetscReal          resnorm;
  PetscBool          hascopy;
  PetscInt           inner_its;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  /* get initial approximation of eigenvalue and eigenvector */
  ierr = NEPGetDefaultShift(nep,&lambda);CHKERRQ(ierr);
  if (!nep->nini) {
    ierr = BVSetRandomColumn(nep->V,0,nep->rand);CHKERRQ(ierr);
  }
  ierr = BVGetColumn(nep->V,0,&u);CHKERRQ(ierr);
  ierr = NEPComputeFunction(nep,lambda,T,T);CHKERRQ(ierr);

  /* prepare linear solver */
  ierr = MatDuplicate(T,MAT_COPY_VALUES,&Tsigma);CHKERRQ(ierr);
  ierr = KSPSetOperators(nep->ksp,Tsigma,Tsigma);CHKERRQ(ierr);

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
    if (nep->lag && !(nep->its%nep->lag) && nep->its>2*nep->lag && resnorm<1e-2) {
      ierr = MatHasOperation(T,MATOP_COPY,&hascopy);CHKERRQ(ierr);
      if (hascopy) {
        ierr = MatCopy(T,Tsigma,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatDestroy(&Tsigma);CHKERRQ(ierr);
        ierr = MatDuplicate(T,MAT_COPY_VALUES,&Tsigma);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(nep->ksp,Tsigma,Tsigma);CHKERRQ(ierr);
    }
    if (!nep->cctol) {
      nep->ktol = PetscMax(nep->ktol/2.0,PETSC_MACHINE_EPSILON*10.0);
      ierr = KSPSetTolerances(nep->ksp,nep->ktol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
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
    ierr = NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->errest,1);CHKERRQ(ierr);

    if (nep->reason == NEP_CONVERGED_ITERATING) {
      /* eigenvector correction: delta = T(sigma)\r */
      ierr = NEP_KSPSolve(nep,r,delta);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(nep->ksp,&kspreason);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"NEP RII Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_rii_max_it","Maximum number of Newton iterations for updating Rayleigh functional","NEPRIISetMaximumIterations",ctx->max_inner_it,&ctx->max_inner_it,NULL);CHKERRQ(ierr);
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

   Collective on NEP

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
  ierr = PetscTryMethod(nep,"NEPRIIGetMaximumIterations_C",(NEP,PetscInt*),(nep,its));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPView_RII"
PetscErrorCode NEPView_RII(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_RII        *ctx = (NEP_RII*)nep->data;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"  RII: maximum number of inner iterations: %D\n",ctx->max_inner_it);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy_RII"
PetscErrorCode NEPDestroy_RII(NEP nep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIISetMaximumIterations_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIIGetMaximumIterations_C",NULL);CHKERRQ(ierr);
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
  nep->data = (void*)ctx;

  nep->ops->solve          = NEPSolve_RII;
  nep->ops->setup          = NEPSetUp_RII;
  nep->ops->setfromoptions = NEPSetFromOptions_RII;
  nep->ops->destroy        = NEPDestroy_RII;
  nep->ops->view           = NEPView_RII;
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIISetMaximumIterations_C",NEPRIISetMaximumIterations_RII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPRIIGetMaximumIterations_C",NEPRIIGetMaximumIterations_RII);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

