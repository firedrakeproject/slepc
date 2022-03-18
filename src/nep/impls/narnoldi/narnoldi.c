/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc nonlinear eigensolver: "narnoldi"

   Method: Nonlinear Arnoldi

   Algorithm:

       Arnoldi for nonlinear eigenproblems.

   References:

       [1] H. Voss, "An Arnoldi method for nonlinear eigenvalue problems",
           BIT 44:387-401, 2004.
*/

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <../src/nep/impls/nepdefl.h>

typedef struct {
  PetscInt lag;             /* interval to rebuild preconditioner */
  KSP      ksp;             /* linear solver object */
} NEP_NARNOLDI;

PetscErrorCode NEPSetUp_NArnoldi(NEP nep)
{
  PetscFunctionBegin;
  CHKERRQ(NEPSetDimensions_Default(nep,nep->nev,&nep->ncv,&nep->mpd));
  PetscCheck(nep->ncv<=nep->nev+nep->mpd,PetscObjectComm((PetscObject)nep),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nev+mpd");
  if (nep->max_it==PETSC_DEFAULT) nep->max_it = nep->nev*nep->ncv;
  if (!nep->which) nep->which = NEP_TARGET_MAGNITUDE;
  PetscCheck(nep->which==NEP_TARGET_MAGNITUDE,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver supports only target magnitude eigenvalues");
  NEPCheckUnsupported(nep,NEP_FEATURE_CALLBACK | NEP_FEATURE_REGION | NEP_FEATURE_TWOSIDED);
  CHKERRQ(NEPAllocateSolution(nep,0));
  CHKERRQ(NEPSetWorkVecs(nep,3));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSolve_NArnoldi(NEP nep)
{
  NEP_NARNOLDI       *ctx = (NEP_NARNOLDI*)nep->data;
  Mat                T,H;
  Vec                f,r,u,uu;
  PetscScalar        *X,lambda=0.0,lambda2=0.0,*eigr,*Ap,sigma;
  const PetscScalar  *Hp;
  PetscReal          beta,resnorm=0.0,nrm,perr=0.0;
  PetscInt           n,i,j,ldds,ldh;
  PetscBool          breakdown,skip=PETSC_FALSE;
  BV                 Vext;
  DS                 ds;
  NEP_EXT_OP         extop=NULL;
  SlepcSC            sc;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  /* get initial space and shift */
  CHKERRQ(NEPGetDefaultShift(nep,&sigma));
  if (!nep->nini) {
    CHKERRQ(BVSetRandomColumn(nep->V,0));
    CHKERRQ(BVNormColumn(nep->V,0,NORM_2,&nrm));
    CHKERRQ(BVScaleColumn(nep->V,0,1.0/nrm));
    n = 1;
  } else n = nep->nini;

  if (!ctx->ksp) CHKERRQ(NEPNArnoldiGetKSP(nep,&ctx->ksp));
  CHKERRQ(NEPDeflationInitialize(nep,nep->V,ctx->ksp,PETSC_FALSE,nep->nev,&extop));
  CHKERRQ(NEPDeflationCreateBV(extop,nep->ncv,&Vext));

  /* prepare linear solver */
  CHKERRQ(NEPDeflationSolveSetUp(extop,sigma));

  CHKERRQ(BVGetColumn(Vext,0,&f));
  CHKERRQ(VecDuplicate(f,&r));
  CHKERRQ(VecDuplicate(f,&u));
  CHKERRQ(BVGetColumn(nep->V,0,&uu));
  CHKERRQ(NEPDeflationCopyToExtendedVec(extop,uu,NULL,f,PETSC_FALSE));
  CHKERRQ(BVRestoreColumn(nep->V,0,&uu));
  CHKERRQ(VecCopy(f,r));
  CHKERRQ(NEPDeflationFunctionSolve(extop,r,f));
  CHKERRQ(VecNorm(f,NORM_2,&nrm));
  CHKERRQ(VecScale(f,1.0/nrm));
  CHKERRQ(BVRestoreColumn(Vext,0,&f));

  CHKERRQ(DSCreate(PetscObjectComm((PetscObject)nep),&ds));
  CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ds));
  CHKERRQ(DSSetType(ds,DSNEP));
  CHKERRQ(DSNEPSetFN(ds,nep->nt,nep->f));
  CHKERRQ(DSAllocate(ds,nep->ncv));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = nep->sc->comparison;
  sc->comparisonctx = nep->sc->comparisonctx;
  CHKERRQ(DSSetFromOptions(ds));

  /* build projected matrices for initial space */
  CHKERRQ(DSSetDimensions(ds,n,0,0));
  CHKERRQ(NEPDeflationProjectOperator(extop,Vext,ds,0,n));

  CHKERRQ(PetscMalloc1(nep->ncv,&eigr));

  /* Restart loop */
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;

    /* solve projected problem */
    CHKERRQ(DSSetDimensions(ds,n,0,0));
    CHKERRQ(DSSetState(ds,DS_STATE_RAW));
    CHKERRQ(DSSolve(ds,eigr,NULL));
    CHKERRQ(DSSynchronize(ds,eigr,NULL));
    if (nep->its>1) lambda2 = lambda;
    lambda = eigr[0];
    nep->eigr[nep->nconv] = lambda;

    /* compute Ritz vector, x = V*s */
    CHKERRQ(DSGetArray(ds,DS_MAT_X,&X));
    CHKERRQ(BVSetActiveColumns(Vext,0,n));
    CHKERRQ(BVMultVec(Vext,1.0,0.0,u,X));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_X,&X));

    /* compute the residual, r = T(lambda)*x */
    CHKERRQ(NEPDeflationComputeFunction(extop,lambda,&T));
    CHKERRQ(MatMult(T,u,r));

    /* convergence test */
    CHKERRQ(VecNorm(r,NORM_2,&resnorm));
    if (nep->its>1) perr=nep->errest[nep->nconv];
    CHKERRQ((*nep->converged)(nep,lambda,0,resnorm,&nep->errest[nep->nconv],nep->convergedctx));
    if (nep->errest[nep->nconv]<=nep->tol) {
      nep->nconv = nep->nconv + 1;
      CHKERRQ(NEPDeflationLocking(extop,u,lambda));
      skip = PETSC_TRUE;
    }
    CHKERRQ((*nep->stopping)(nep,nep->its,nep->max_it,nep->nconv,nep->nev,&nep->reason,nep->stoppingctx));
    if (!skip || nep->reason>0) {
      CHKERRQ(NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->eigi,nep->errest,(nep->reason>0)?nep->nconv:nep->nconv+1));
    }

    if (nep->reason == NEP_CONVERGED_ITERATING) {
      if (!skip) {
        if (n>=nep->ncv) {
          nep->reason = NEP_DIVERGED_SUBSPACE_EXHAUSTED;
          break;
        }
        if (ctx->lag && !(nep->its%ctx->lag) && nep->its>=2*ctx->lag && perr && nep->errest[nep->nconv]>.5*perr) {
          CHKERRQ(NEPDeflationSolveSetUp(extop,lambda2));
        }

        /* continuation vector: f = T(sigma)\r */
        CHKERRQ(BVGetColumn(Vext,n,&f));
        CHKERRQ(NEPDeflationFunctionSolve(extop,r,f));
        CHKERRQ(BVRestoreColumn(Vext,n,&f));
        CHKERRQ(KSPGetConvergedReason(ctx->ksp,&kspreason));
        if (kspreason<0) {
          CHKERRQ(PetscInfo(nep,"iter=%" PetscInt_FMT ", linear solve failed, stopping solve\n",nep->its));
          nep->reason = NEP_DIVERGED_LINEAR_SOLVE;
          break;
        }

        /* orthonormalize */
        CHKERRQ(BVOrthonormalizeColumn(Vext,n,PETSC_FALSE,&beta,&breakdown));
        if (breakdown || beta==0.0) {
          CHKERRQ(PetscInfo(nep,"iter=%" PetscInt_FMT ", orthogonalization failed, stopping solve\n",nep->its));
          nep->reason = NEP_DIVERGED_BREAKDOWN;
          break;
        }

        /* update projected matrices */
        CHKERRQ(DSSetDimensions(ds,n+1,0,0));
        CHKERRQ(NEPDeflationProjectOperator(extop,Vext,ds,n,n+1));
        n++;
      } else {
        nep->its--;  /* do not count this as a full iteration */
        CHKERRQ(BVGetColumn(Vext,0,&f));
        CHKERRQ(NEPDeflationSetRandomVec(extop,f));
        CHKERRQ(NEPDeflationSolveSetUp(extop,sigma));
        CHKERRQ(VecCopy(f,r));
        CHKERRQ(NEPDeflationFunctionSolve(extop,r,f));
        CHKERRQ(VecNorm(f,NORM_2,&nrm));
        CHKERRQ(VecScale(f,1.0/nrm));
        CHKERRQ(BVRestoreColumn(Vext,0,&f));
        n = 1;
        CHKERRQ(DSSetDimensions(ds,n,0,0));
        CHKERRQ(NEPDeflationProjectOperator(extop,Vext,ds,n-1,n));
        skip = PETSC_FALSE;
      }
    }
  }

  CHKERRQ(NEPDeflationGetInvariantPair(extop,NULL,&H));
  CHKERRQ(MatGetSize(H,NULL,&ldh));
  CHKERRQ(DSSetType(nep->ds,DSNHEP));
  CHKERRQ(DSAllocate(nep->ds,PetscMax(nep->nconv,1)));
  CHKERRQ(DSGetLeadingDimension(nep->ds,&ldds));
  CHKERRQ(MatDenseGetArrayRead(H,&Hp));
  CHKERRQ(DSGetArray(nep->ds,DS_MAT_A,&Ap));
  for (j=0;j<nep->nconv;j++)
    for (i=0;i<nep->nconv;i++) Ap[j*ldds+i] = Hp[j*ldh+i];
  CHKERRQ(DSRestoreArray(nep->ds,DS_MAT_A,&Ap));
  CHKERRQ(MatDenseRestoreArrayRead(H,&Hp));
  CHKERRQ(MatDestroy(&H));
  CHKERRQ(DSSetDimensions(nep->ds,nep->nconv,0,nep->nconv));
  CHKERRQ(DSSolve(nep->ds,nep->eigr,nep->eigi));
  CHKERRQ(NEPDeflationReset(extop));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(BVDestroy(&Vext));
  CHKERRQ(DSDestroy(&ds));
  CHKERRQ(PetscFree(eigr));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPNArnoldiSetLagPreconditioner_NArnoldi(NEP nep,PetscInt lag)
{
  NEP_NARNOLDI *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  PetscCheck(lag>=0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag must be non-negative");
  ctx->lag = lag;
  PetscFunctionReturn(0);
}

/*@
   NEPNArnoldiSetLagPreconditioner - Determines when the preconditioner is rebuilt in the
   nonlinear solve.

   Logically Collective on nep

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-  lag - 0 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is
          computed within the nonlinear iteration, 2 means every second time
          the Jacobian is built, etc.

   Options Database Keys:
.  -nep_narnoldi_lag_preconditioner <lag> - the lag value

   Notes:
   The default is 1.
   The preconditioner is ALWAYS built in the first iteration of a nonlinear solve.

   Level: intermediate

.seealso: NEPNArnoldiGetLagPreconditioner()
@*/
PetscErrorCode NEPNArnoldiSetLagPreconditioner(NEP nep,PetscInt lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,lag,2);
  CHKERRQ(PetscTryMethod(nep,"NEPNArnoldiSetLagPreconditioner_C",(NEP,PetscInt),(nep,lag)));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPNArnoldiGetLagPreconditioner_NArnoldi(NEP nep,PetscInt *lag)
{
  NEP_NARNOLDI *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  *lag = ctx->lag;
  PetscFunctionReturn(0);
}

/*@
   NEPNArnoldiGetLagPreconditioner - Indicates how often the preconditioner is rebuilt.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  lag - the lag parameter

   Level: intermediate

.seealso: NEPNArnoldiSetLagPreconditioner()
@*/
PetscErrorCode NEPNArnoldiGetLagPreconditioner(NEP nep,PetscInt *lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidIntPointer(lag,2);
  CHKERRQ(PetscUseMethod(nep,"NEPNArnoldiGetLagPreconditioner_C",(NEP,PetscInt*),(nep,lag)));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSetFromOptions_NArnoldi(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscInt       i;
  PetscBool      flg;
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"NEP N-Arnoldi Options"));
    i = 0;
    CHKERRQ(PetscOptionsInt("-nep_narnoldi_lag_preconditioner","Interval to rebuild preconditioner","NEPNArnoldiSetLagPreconditioner",ctx->lag,&i,&flg));
    if (flg) CHKERRQ(NEPNArnoldiSetLagPreconditioner(nep,i));

  CHKERRQ(PetscOptionsTail());

  if (!ctx->ksp) CHKERRQ(NEPNArnoldiGetKSP(nep,&ctx->ksp));
  CHKERRQ(KSPSetFromOptions(ctx->ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPNArnoldiSetKSP_NArnoldi(NEP nep,KSP ksp)
{
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)ksp));
  CHKERRQ(KSPDestroy(&ctx->ksp));
  ctx->ksp = ksp;
  CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp));
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPNArnoldiSetKSP - Associate a linear solver object (KSP) to the nonlinear
   eigenvalue solver.

   Collective on nep

   Input Parameters:
+  nep - eigenvalue solver
-  ksp - the linear solver object

   Level: advanced

.seealso: NEPNArnoldiGetKSP()
@*/
PetscErrorCode NEPNArnoldiSetKSP(NEP nep,KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(nep,1,ksp,2);
  CHKERRQ(PetscTryMethod(nep,"NEPNArnoldiSetKSP_C",(NEP,KSP),(nep,ksp)));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPNArnoldiGetKSP_NArnoldi(NEP nep,KSP *ksp)
{
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  if (!ctx->ksp) {
    CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)nep),&ctx->ksp));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)nep,1));
    CHKERRQ(KSPSetOptionsPrefix(ctx->ksp,((PetscObject)nep)->prefix));
    CHKERRQ(KSPAppendOptionsPrefix(ctx->ksp,"nep_narnoldi_"));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp));
    CHKERRQ(PetscObjectSetOptions((PetscObject)ctx->ksp,((PetscObject)nep)->options));
    CHKERRQ(KSPSetErrorIfNotConverged(ctx->ksp,PETSC_TRUE));
    CHKERRQ(KSPSetTolerances(ctx->ksp,SlepcDefaultTol(nep->tol),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  }
  *ksp = ctx->ksp;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ksp,2);
  CHKERRQ(PetscUseMethod(nep,"NEPNArnoldiGetKSP_C",(NEP,KSP*),(nep,ksp)));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPView_NArnoldi(NEP nep,PetscViewer viewer)
{
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    if (ctx->lag) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  updating the preconditioner every %" PetscInt_FMT " iterations\n",ctx->lag));
    }
    if (!ctx->ksp) CHKERRQ(NEPNArnoldiGetKSP(nep,&ctx->ksp));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(KSPView(ctx->ksp,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPReset_NArnoldi(NEP nep)
{
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  CHKERRQ(KSPReset(ctx->ksp));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDestroy_NArnoldi(NEP nep)
{
  NEP_NARNOLDI   *ctx = (NEP_NARNOLDI*)nep->data;

  PetscFunctionBegin;
  CHKERRQ(KSPDestroy(&ctx->ksp));
  CHKERRQ(PetscFree(nep->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiSetLagPreconditioner_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiGetLagPreconditioner_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiSetKSP_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiGetKSP_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode NEPCreate_NArnoldi(NEP nep)
{
  NEP_NARNOLDI   *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(nep,&ctx));
  nep->data = (void*)ctx;
  ctx->lag  = 1;

  nep->useds = PETSC_TRUE;

  nep->ops->solve          = NEPSolve_NArnoldi;
  nep->ops->setup          = NEPSetUp_NArnoldi;
  nep->ops->setfromoptions = NEPSetFromOptions_NArnoldi;
  nep->ops->reset          = NEPReset_NArnoldi;
  nep->ops->destroy        = NEPDestroy_NArnoldi;
  nep->ops->view           = NEPView_NArnoldi;
  nep->ops->computevectors = NEPComputeVectors_Schur;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiSetLagPreconditioner_C",NEPNArnoldiSetLagPreconditioner_NArnoldi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiGetLagPreconditioner_C",NEPNArnoldiGetLagPreconditioner_NArnoldi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiSetKSP_C",NEPNArnoldiSetKSP_NArnoldi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPNArnoldiGetKSP_C",NEPNArnoldiGetKSP_NArnoldi));
  PetscFunctionReturn(0);
}
