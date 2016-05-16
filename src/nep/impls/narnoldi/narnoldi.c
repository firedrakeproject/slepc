/*

   SLEPc nonlinear eigensolver: "narnoldi"

   Method: Nonlinear Arnoldi

   Algorithm:

       Arnoldi for nonlinear eigenproblems.

   References:

       [1] H. Voss, "An Arnoldi method for nonlinear eigenvalue problems",
           BIT 44:387-401, 2004.

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
  KSP      ksp;              /* linear solver object */
} NEP_NARNOLDI;

#undef __FUNCT__
#define __FUNCT__ "NEPNArnoldi_KSPSolve"
PETSC_STATIC_INLINE PetscErrorCode NEPNArnoldi_KSPSolve(NEP nep,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       lits;
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  ierr = KSPSolve(ctx->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ctx->ksp,&lits);CHKERRQ(ierr);
  ierr = PetscInfo2(nep,"iter=%D, linear solve iterations=%D\n",nep->its,lits);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_NArnoldi"
PetscErrorCode NEPSetUp_NArnoldi(NEP nep)
{
  PetscErrorCode ierr;
  PetscBool      istrivial;

  PetscFunctionBegin;
  ierr = NEPSetDimensions_Default(nep,nep->nev,&nep->ncv,&nep->mpd);CHKERRQ(ierr);
  if (nep->ncv>nep->nev+nep->mpd) SETERRQ(PetscObjectComm((PetscObject)nep),1,"The value of ncv must not be larger than nev+mpd");
  if (nep->nev>1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Requested several eigenpairs but this solver can compute only one");
  if (!nep->max_it) nep->max_it = nep->ncv;
  if (nep->max_it < nep->ncv) SETERRQ(PetscObjectComm((PetscObject)nep),1,"Current implementation is unrestarted, must set max_it >= ncv");
  if (nep->which && nep->which!=NEP_TARGET_MAGNITUDE) SETERRQ(PetscObjectComm((PetscObject)nep),1,"Wrong value of which");
  if (nep->fui!=NEP_USER_INTERFACE_SPLIT) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"NARNOLDI only available for split operator");

  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver does not support region filtering");

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,3);CHKERRQ(ierr);

  /* set-up DS and transfer split operator functions */
  ierr = DSSetType(nep->ds,DSNEP);CHKERRQ(ierr);
  ierr = DSNEPSetFN(nep->ds,nep->nt,nep->f);CHKERRQ(ierr);
  ierr = DSAllocate(nep->ds,nep->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSolve_NArnoldi"
PetscErrorCode NEPSolve_NArnoldi(NEP nep)
{
  PetscErrorCode     ierr;
  NEP_NARNOLDI       *ctx = (NEP_NARNOLDI*)nep->data;
  Mat                T=nep->function,Tsigma;
  Vec                f,r=nep->work[0],x=nep->work[1],w=nep->work[2];
  PetscScalar        *X,lambda;
  PetscReal          beta,resnorm=0.0,nrm;
  PetscInt           n;
  PetscBool          breakdown;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  /* get initial space and shift */
  ierr = NEPGetDefaultShift(nep,&lambda);CHKERRQ(ierr);
  if (!nep->nini) {
    ierr = BVSetRandomColumn(nep->V,0);CHKERRQ(ierr);
    ierr = BVNormColumn(nep->V,0,NORM_2,&nrm);CHKERRQ(ierr);
    ierr = BVScaleColumn(nep->V,0,1.0/nrm);CHKERRQ(ierr);
    n = 1;
  } else n = nep->nini;

  /* build projected matrices for initial space */
  ierr = DSSetDimensions(nep->ds,n,0,0,0);CHKERRQ(ierr);
  ierr = NEPProjectOperator(nep,0,n);CHKERRQ(ierr);

  /* prepare linear solver */
  if (!ctx->ksp) { ierr = NEPNArnoldiGetKSP(nep,&ctx->ksp);CHKERRQ(ierr); }
  ierr = NEPComputeFunction(nep,lambda,T,T);CHKERRQ(ierr);
  ierr = MatDuplicate(T,MAT_COPY_VALUES,&Tsigma);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->ksp,Tsigma,Tsigma);CHKERRQ(ierr);

  /* Restart loop */
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;

    /* solve projected problem */
    ierr = DSSetDimensions(nep->ds,n,0,0,0);CHKERRQ(ierr);
    ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSSolve(nep->ds,nep->eigr,NULL);CHKERRQ(ierr);
    lambda = nep->eigr[0];

    /* compute Ritz vector, x = V*s */
    ierr = DSGetArray(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(nep->V,0,n);CHKERRQ(ierr);
    ierr = BVMultVec(nep->V,1.0,0.0,x,X);CHKERRQ(ierr);
    ierr = DSRestoreArray(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);

    /* compute the residual, r = T(lambda)*x */
    ierr = NEPApplyFunction(nep,lambda,x,w,r,NULL,NULL);CHKERRQ(ierr);

    /* convergence test */
    ierr = VecNorm(r,NORM_2,&resnorm);CHKERRQ(ierr);
    ierr = (*nep->converged)(nep,lambda,0,resnorm,&nep->errest[nep->nconv],nep->convergedctx);CHKERRQ(ierr);
    if (nep->errest[nep->nconv]<=nep->tol) {
      ierr = BVInsertVec(nep->V,nep->nconv,x);CHKERRQ(ierr);
      nep->nconv = nep->nconv + 1;
    }
    ierr = (*nep->stopping)(nep,nep->its,nep->max_it,nep->nconv,nep->nev,&nep->reason,nep->stoppingctx);CHKERRQ(ierr);
    ierr = NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->eigi,nep->errest,1);CHKERRQ(ierr);

    if (nep->reason == NEP_CONVERGED_ITERATING) {

      /* continuation vector: f = T(sigma)\r */
      ierr = BVGetColumn(nep->V,n,&f);CHKERRQ(ierr);
      ierr = NEPNArnoldi_KSPSolve(nep,r,f);CHKERRQ(ierr);
      ierr = BVRestoreColumn(nep->V,n,&f);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(ctx->ksp,&kspreason);CHKERRQ(ierr);
      if (kspreason<0) {
        ierr = PetscInfo1(nep,"iter=%D, linear solve failed, stopping solve\n",nep->its);CHKERRQ(ierr);
        nep->reason = NEP_DIVERGED_LINEAR_SOLVE;
        break;
      }

      /* orthonormalize */
      ierr = BVOrthogonalizeColumn(nep->V,n,NULL,&beta,&breakdown);CHKERRQ(ierr);
      if (breakdown || beta==0.0) {
        ierr = PetscInfo1(nep,"iter=%D, orthogonalization failed, stopping solve\n",nep->its);CHKERRQ(ierr);
        nep->reason = NEP_DIVERGED_BREAKDOWN;
        break;
      }
      ierr = BVScaleColumn(nep->V,n,1.0/beta);CHKERRQ(ierr);

      /* update projected matrices */
      ierr = DSSetDimensions(nep->ds,n+1,0,0,0);CHKERRQ(ierr);
      ierr = NEPProjectOperator(nep,n,n+1);CHKERRQ(ierr);
      n++;
    }
  }
  ierr = MatDestroy(&Tsigma);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetFromOptions_NArnoldi"
PetscErrorCode NEPSetFromOptions_NArnoldi(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscErrorCode ierr;
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  if (!ctx->ksp) { ierr = NEPNArnoldiGetKSP(nep,&ctx->ksp);CHKERRQ(ierr); }
  ierr = KSPSetOperators(ctx->ksp,nep->function,nep->function_pre);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ctx->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNArnoldiSetKSP_NArnoldi"
static PetscErrorCode NEPNArnoldiSetKSP_NArnoldi(NEP nep,KSP ksp)
{
  PetscErrorCode ierr;
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->ksp);CHKERRQ(ierr);
  ctx->ksp = ksp;
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp);CHKERRQ(ierr);
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNArnoldiSetKSP"
/*@
   NEPNArnoldiSetKSP - Associate a linear solver object (KSP) to the nonlinear
   eigenvalue solver.

   Collective on NEP

   Input Parameters:
+  nep - eigenvalue solver
-  ksp - the linear solver object

   Level: advanced

.seealso: NEPNArnoldiGetKSP()
@*/
PetscErrorCode NEPNArnoldiSetKSP(NEP nep,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(nep,1,ksp,2);
  ierr = PetscTryMethod(nep,"NEPNArnoldiSetKSP_C",(NEP,KSP),(nep,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNArnoldiGetKSP_NArnoldi"
static PetscErrorCode NEPNArnoldiGetKSP_NArnoldi(NEP nep,KSP *ksp)
{
  PetscErrorCode ierr;
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  if (!ctx->ksp) {
    ierr = KSPCreate(PetscObjectComm((PetscObject)nep),&ctx->ksp);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ctx->ksp,((PetscObject)nep)->prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(ctx->ksp,"nep_narnoldi_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)nep,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(ctx->ksp,PETSC_TRUE);CHKERRQ(ierr);
  }
  *ksp = ctx->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNArnoldiGetKSP"
/*@
   NEPNArnoldiGetKSP - Retrieve the linear solver object (KSP) associated with
   the nonlinear eigenvalue solver.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  ksp - the linear solver object

   Level: advanced

.seealso: NEPNArnoldiSetKSP()
@*/
PetscErrorCode NEPNArnoldiGetKSP(NEP nep,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ksp,2);
  ierr = PetscUseMethod(nep,"NEPNArnoldiGetKSP_C",(NEP,KSP*),(nep,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPView_NArnoldi"
PetscErrorCode NEPView_NArnoldi(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!ctx->ksp) { ierr = NEPNArnoldiGetKSP(nep,&ctx->ksp);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(ctx->ksp,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy_NArnoldi"
PetscErrorCode NEPDestroy_NArnoldi(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  ierr = KSPDestroy(&ctx->ksp);CHKERRQ(ierr);
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiSetKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiGetKSP_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCreate_NArnoldi"
PETSC_EXTERN PetscErrorCode NEPCreate_NArnoldi(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NARNOLDI   *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(nep,&ctx);CHKERRQ(ierr);
  nep->data = (void*)ctx;

  nep->ops->solve          = NEPSolve_NArnoldi;
  nep->ops->setup          = NEPSetUp_NArnoldi;
  nep->ops->setfromoptions = NEPSetFromOptions_NArnoldi;
  nep->ops->destroy        = NEPDestroy_NArnoldi;
  nep->ops->view           = NEPView_NArnoldi;
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiSetKSP_C",NEPNArnoldiSetKSP_NArnoldi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiGetKSP_C",NEPNArnoldiGetKSP_NArnoldi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

