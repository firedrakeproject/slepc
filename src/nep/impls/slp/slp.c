/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc nonlinear eigensolver: "slp"

   Method: Successive linear problems

   Algorithm:

       Newton-type iteration based on first order Taylor approximation.

   References:

       [1] A. Ruhe, "Algorithms for the nonlinear eigenvalue problem", SIAM J.
           Numer. Anal. 10(4):674-689, 1973.
*/

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <../src/nep/impls/nepdefl.h>
#include "slp.h"

typedef struct {
  NEP_EXT_OP extop;
  Vec        w;
} NEP_SLP_MATSHELL;

PetscErrorCode NEPSetUp_SLP(NEP nep)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;
  PetscBool      flg;
  ST             st;

  PetscFunctionBegin;
  if (nep->ncv!=PETSC_DEFAULT) CHKERRQ(PetscInfo(nep,"Setting ncv = nev, ignoring user-provided value\n"));
  nep->ncv = nep->nev;
  if (nep->mpd!=PETSC_DEFAULT) CHKERRQ(PetscInfo(nep,"Setting mpd = nev, ignoring user-provided value\n"));
  nep->mpd = nep->nev;
  PetscCheck(nep->ncv<=nep->nev+nep->mpd,PetscObjectComm((PetscObject)nep),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nev+mpd");
  if (nep->max_it==PETSC_DEFAULT) nep->max_it = PetscMax(5000,2*nep->n/nep->ncv);
  if (!nep->which) nep->which = NEP_TARGET_MAGNITUDE;
  PetscCheck(nep->which==NEP_TARGET_MAGNITUDE,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver supports only target magnitude eigenvalues");
  NEPCheckUnsupported(nep,NEP_FEATURE_REGION);

  if (!ctx->eps) CHKERRQ(NEPSLPGetEPS(nep,&ctx->eps));
  CHKERRQ(EPSGetST(ctx->eps,&st));
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)st,&flg,STSINVERT,STCAYLEY,""));
  PetscCheck(!flg,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"SLP does not support spectral transformation");
  CHKERRQ(EPSSetDimensions(ctx->eps,1,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(EPSSetWhichEigenpairs(ctx->eps,EPS_LARGEST_MAGNITUDE));
  CHKERRQ(EPSSetTolerances(ctx->eps,SlepcDefaultTol(nep->tol)/10.0,nep->max_it));
  if (nep->tol==PETSC_DEFAULT) nep->tol = SLEPC_DEFAULT_TOL;
  if (ctx->deftol==PETSC_DEFAULT) ctx->deftol = nep->tol;

  if (nep->twosided) {
    nep->ops->solve = NEPSolve_SLP_Twosided;
    nep->ops->computevectors = NULL;
    if (!ctx->epsts) CHKERRQ(NEPSLPGetEPSLeft(nep,&ctx->epsts));
    CHKERRQ(EPSGetST(ctx->epsts,&st));
    CHKERRQ(PetscObjectTypeCompareAny((PetscObject)st,&flg,STSINVERT,STCAYLEY,""));
    PetscCheck(!flg,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"SLP does not support spectral transformation");
    CHKERRQ(EPSSetDimensions(ctx->epsts,1,PETSC_DECIDE,PETSC_DECIDE));
    CHKERRQ(EPSSetWhichEigenpairs(ctx->epsts,EPS_LARGEST_MAGNITUDE));
    CHKERRQ(EPSSetTolerances(ctx->epsts,SlepcDefaultTol(nep->tol)/10.0,nep->max_it));
  } else {
    nep->ops->solve = NEPSolve_SLP;
    nep->ops->computevectors = NEPComputeVectors_Schur;
  }
  CHKERRQ(NEPAllocateSolution(nep,0));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SLP(Mat M,Vec x,Vec y)
{
  NEP_SLP_MATSHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(MatMult(ctx->extop->MJ,x,ctx->w));
  CHKERRQ(NEPDeflationFunctionSolve(ctx->extop,ctx->w,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SLP(Mat M)
{
  NEP_SLP_MATSHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(VecDestroy(&ctx->w));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
static PetscErrorCode MatCreateVecs_SLP(Mat M,Vec *left,Vec *right)
{
  NEP_SLP_MATSHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  if (right) {
    CHKERRQ(VecDuplicate(ctx->w,right));
  }
  if (left) {
    CHKERRQ(VecDuplicate(ctx->w,left));
  }
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode NEPSLPSetUpLinearEP(NEP nep,NEP_EXT_OP extop,PetscScalar lambda,Vec u,PetscBool ini)
{
  NEP_SLP          *slpctx = (NEP_SLP*)nep->data;
  Mat              Mshell;
  PetscInt         nloc,mloc;
  NEP_SLP_MATSHELL *shellctx;

  PetscFunctionBegin;
  if (ini) {
    /* Create mat shell */
    CHKERRQ(PetscNew(&shellctx));
    shellctx->extop = extop;
    CHKERRQ(NEPDeflationCreateVec(extop,&shellctx->w));
    CHKERRQ(MatGetLocalSize(nep->function,&mloc,&nloc));
    nloc += extop->szd; mloc += extop->szd;
    CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)nep),nloc,mloc,PETSC_DETERMINE,PETSC_DETERMINE,shellctx,&Mshell));
    CHKERRQ(MatShellSetOperation(Mshell,MATOP_MULT,(void(*)(void))MatMult_SLP));
    CHKERRQ(MatShellSetOperation(Mshell,MATOP_DESTROY,(void(*)(void))MatDestroy_SLP));
#if defined(PETSC_HAVE_CUDA)
    CHKERRQ(MatShellSetOperation(Mshell,MATOP_CREATE_VECS,(void(*)(void))MatCreateVecs_SLP));
#endif
    CHKERRQ(EPSSetOperators(slpctx->eps,Mshell,NULL));
    CHKERRQ(MatDestroy(&Mshell));
  }
  CHKERRQ(NEPDeflationSolveSetUp(extop,lambda));
  CHKERRQ(NEPDeflationComputeJacobian(extop,lambda,NULL));
  CHKERRQ(EPSSetInitialSpace(slpctx->eps,1,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSolve_SLP(NEP nep)
{
  NEP_SLP           *ctx = (NEP_SLP*)nep->data;
  Mat               F,H;
  Vec               uu,u,r;
  PetscScalar       sigma,lambda,mu,im,*Ap;
  const PetscScalar *Hp;
  PetscReal         resnorm;
  PetscInt          nconv,ldh,ldds,i,j;
  PetscBool         skip=PETSC_FALSE,lock=PETSC_FALSE;
  NEP_EXT_OP        extop=NULL;    /* Extended operator for deflation */

  PetscFunctionBegin;
  /* get initial approximation of eigenvalue and eigenvector */
  CHKERRQ(NEPGetDefaultShift(nep,&sigma));
  if (!nep->nini) {
    CHKERRQ(BVSetRandomColumn(nep->V,0));
  }
  lambda = sigma;
  if (!ctx->ksp) CHKERRQ(NEPSLPGetKSP(nep,&ctx->ksp));
  CHKERRQ(NEPDeflationInitialize(nep,nep->V,ctx->ksp,PETSC_TRUE,nep->nev,&extop));
  CHKERRQ(NEPDeflationCreateVec(extop,&u));
  CHKERRQ(VecDuplicate(u,&r));
  CHKERRQ(BVGetColumn(nep->V,0,&uu));
  CHKERRQ(NEPDeflationCopyToExtendedVec(extop,uu,NULL,u,PETSC_FALSE));
  CHKERRQ(BVRestoreColumn(nep->V,0,&uu));

  /* Restart loop */
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;

    /* form residual,  r = T(lambda)*u (used in convergence test only) */
    CHKERRQ(NEPDeflationComputeFunction(extop,lambda,&F));
    CHKERRQ(MatMult(F,u,r));

    /* convergence test */
    CHKERRQ(VecNorm(r,NORM_2,&resnorm));
    CHKERRQ((*nep->converged)(nep,lambda,0,resnorm,&nep->errest[nep->nconv],nep->convergedctx));
    nep->eigr[nep->nconv] = lambda;
    if (nep->errest[nep->nconv]<=nep->tol || nep->errest[nep->nconv]<=ctx->deftol) {
      if (nep->errest[nep->nconv]<=ctx->deftol && !extop->ref && nep->nconv) {
        CHKERRQ(NEPDeflationExtractEigenpair(extop,nep->nconv,u,lambda,nep->ds));
        CHKERRQ(NEPDeflationSetRefine(extop,PETSC_TRUE));
        CHKERRQ(MatMult(F,u,r));
        CHKERRQ(VecNorm(r,NORM_2,&resnorm));
        CHKERRQ((*nep->converged)(nep,lambda,0,resnorm,&nep->errest[nep->nconv],nep->convergedctx));
        if (nep->errest[nep->nconv]<=nep->tol) lock = PETSC_TRUE;
      } else if (nep->errest[nep->nconv]<=nep->tol) lock = PETSC_TRUE;
    }

    if (lock) {
      CHKERRQ(NEPDeflationSetRefine(extop,PETSC_FALSE));
      nep->nconv = nep->nconv + 1;
      skip = PETSC_TRUE;
      lock = PETSC_FALSE;
      CHKERRQ(NEPDeflationLocking(extop,u,lambda));
    }
    CHKERRQ((*nep->stopping)(nep,nep->its,nep->max_it,nep->nconv,nep->nev,&nep->reason,nep->stoppingctx));
    if (!skip || nep->reason>0) {
      CHKERRQ(NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->eigi,nep->errest,(nep->reason>0)?nep->nconv:nep->nconv+1));
    }

    if (nep->reason == NEP_CONVERGED_ITERATING) {
      if (!skip) {
        /* evaluate T(lambda) and T'(lambda) */
        CHKERRQ(NEPSLPSetUpLinearEP(nep,extop,lambda,u,nep->its==1?PETSC_TRUE:PETSC_FALSE));
        /* compute new eigenvalue correction mu and eigenvector approximation u */
        CHKERRQ(EPSSolve(ctx->eps));
        CHKERRQ(EPSGetConverged(ctx->eps,&nconv));
        if (!nconv) {
          CHKERRQ(PetscInfo(nep,"iter=%" PetscInt_FMT ", inner iteration failed, stopping solve\n",nep->its));
          nep->reason = NEP_DIVERGED_LINEAR_SOLVE;
          break;
        }
        CHKERRQ(EPSGetEigenpair(ctx->eps,0,&mu,&im,u,NULL));
        mu = 1.0/mu;
        PetscCheck(PetscAbsScalar(im)<PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Complex eigenvalue approximation - not implemented in real scalars");
      } else {
        nep->its--;  /* do not count this as a full iteration */
        /* use second eigenpair computed in previous iteration */
        CHKERRQ(EPSGetConverged(ctx->eps,&nconv));
        if (nconv>=2) {
          CHKERRQ(EPSGetEigenpair(ctx->eps,1,&mu,&im,u,NULL));
          mu = 1.0/mu;
        } else {
          CHKERRQ(NEPDeflationSetRandomVec(extop,u));
          mu = lambda-sigma;
        }
        skip = PETSC_FALSE;
      }
      /* correct eigenvalue */
      lambda = lambda - mu;
    }
  }
  CHKERRQ(NEPDeflationGetInvariantPair(extop,NULL,&H));
  CHKERRQ(MatGetSize(H,NULL,&ldh));
  CHKERRQ(DSReset(nep->ds));
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
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSetFromOptions_SLP(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;
  PetscBool      flg;
  PetscReal      r;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"NEP SLP Options"));

    r = 0.0;
    CHKERRQ(PetscOptionsReal("-nep_slp_deflation_threshold","Tolerance used as a threshold for including deflated eigenpairs","NEPSLPSetDeflationThreshold",ctx->deftol,&r,&flg));
    if (flg) CHKERRQ(NEPSLPSetDeflationThreshold(nep,r));

  CHKERRQ(PetscOptionsTail());

  if (!ctx->eps) CHKERRQ(NEPSLPGetEPS(nep,&ctx->eps));
  CHKERRQ(EPSSetFromOptions(ctx->eps));
  if (nep->twosided) {
    if (!ctx->epsts) CHKERRQ(NEPSLPGetEPSLeft(nep,&ctx->epsts));
    CHKERRQ(EPSSetFromOptions(ctx->epsts));
  }
  if (!ctx->ksp) CHKERRQ(NEPSLPGetKSP(nep,&ctx->ksp));
  CHKERRQ(KSPSetFromOptions(ctx->ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSLPSetDeflationThreshold_SLP(NEP nep,PetscReal deftol)
{
  NEP_SLP *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  if (deftol == PETSC_DEFAULT) {
    ctx->deftol = PETSC_DEFAULT;
    nep->state  = NEP_STATE_INITIAL;
  } else {
    PetscCheck(deftol>0.0,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of deftol. Must be > 0");
    ctx->deftol = deftol;
  }
  PetscFunctionReturn(0);
}

/*@
   NEPSLPSetDeflationThreshold - Sets the threshold value used to switch between
   deflated and non-deflated iteration.

   Logically Collective on nep

   Input Parameters:
+  nep    - nonlinear eigenvalue solver
-  deftol - the threshold value

   Options Database Keys:
.  -nep_slp_deflation_threshold <deftol> - set the threshold

   Notes:
   Normally, the solver iterates on the extended problem in order to deflate
   previously converged eigenpairs. If this threshold is set to a nonzero value,
   then once the residual error is below this threshold the solver will
   continue the iteration without deflation. The intention is to be able to
   improve the current eigenpair further, despite having previous eigenpairs
   with somewhat bad precision.

   Level: advanced

.seealso: NEPSLPGetDeflationThreshold()
@*/
PetscErrorCode NEPSLPSetDeflationThreshold(NEP nep,PetscReal deftol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(nep,deftol,2);
  CHKERRQ(PetscTryMethod(nep,"NEPSLPSetDeflationThreshold_C",(NEP,PetscReal),(nep,deftol)));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSLPGetDeflationThreshold_SLP(NEP nep,PetscReal *deftol)
{
  NEP_SLP *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  *deftol = ctx->deftol;
  PetscFunctionReturn(0);
}

/*@
   NEPSLPGetDeflationThreshold - Returns the threshold value that controls deflation.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  deftol - the threshold

   Level: advanced

.seealso: NEPSLPSetDeflationThreshold()
@*/
PetscErrorCode NEPSLPGetDeflationThreshold(NEP nep,PetscReal *deftol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidRealPointer(deftol,2);
  CHKERRQ(PetscUseMethod(nep,"NEPSLPGetDeflationThreshold_C",(NEP,PetscReal*),(nep,deftol)));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSLPSetEPS_SLP(NEP nep,EPS eps)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)eps));
  CHKERRQ(EPSDestroy(&ctx->eps));
  ctx->eps = eps;
  CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->eps));
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPSLPSetEPS - Associate a linear eigensolver object (EPS) to the
   nonlinear eigenvalue solver.

   Collective on nep

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-  eps - the eigensolver object

   Level: advanced

.seealso: NEPSLPGetEPS()
@*/
PetscErrorCode NEPSLPSetEPS(NEP nep,EPS eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(nep,1,eps,2);
  CHKERRQ(PetscTryMethod(nep,"NEPSLPSetEPS_C",(NEP,EPS),(nep,eps)));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSLPGetEPS_SLP(NEP nep,EPS *eps)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  if (!ctx->eps) {
    CHKERRQ(EPSCreate(PetscObjectComm((PetscObject)nep),&ctx->eps));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ctx->eps,(PetscObject)nep,1));
    CHKERRQ(EPSSetOptionsPrefix(ctx->eps,((PetscObject)nep)->prefix));
    CHKERRQ(EPSAppendOptionsPrefix(ctx->eps,"nep_slp_"));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->eps));
    CHKERRQ(PetscObjectSetOptions((PetscObject)ctx->eps,((PetscObject)nep)->options));
  }
  *eps = ctx->eps;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(eps,2);
  CHKERRQ(PetscUseMethod(nep,"NEPSLPGetEPS_C",(NEP,EPS*),(nep,eps)));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSLPSetEPSLeft_SLP(NEP nep,EPS eps)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)eps));
  CHKERRQ(EPSDestroy(&ctx->epsts));
  ctx->epsts = eps;
  CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->epsts));
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPSLPSetEPSLeft - Associate a linear eigensolver object (EPS) to the
   nonlinear eigenvalue solver, used to compute left eigenvectors in the
   two-sided variant of SLP.

   Collective on nep

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-  eps - the eigensolver object

   Level: advanced

.seealso: NEPSLPGetEPSLeft(), NEPSetTwoSided()
@*/
PetscErrorCode NEPSLPSetEPSLeft(NEP nep,EPS eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(nep,1,eps,2);
  CHKERRQ(PetscTryMethod(nep,"NEPSLPSetEPSLeft_C",(NEP,EPS),(nep,eps)));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSLPGetEPSLeft_SLP(NEP nep,EPS *eps)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  if (!ctx->epsts) {
    CHKERRQ(EPSCreate(PetscObjectComm((PetscObject)nep),&ctx->epsts));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ctx->epsts,(PetscObject)nep,1));
    CHKERRQ(EPSSetOptionsPrefix(ctx->epsts,((PetscObject)nep)->prefix));
    CHKERRQ(EPSAppendOptionsPrefix(ctx->epsts,"nep_slp_left_"));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->epsts));
    CHKERRQ(PetscObjectSetOptions((PetscObject)ctx->epsts,((PetscObject)nep)->options));
  }
  *eps = ctx->epsts;
  PetscFunctionReturn(0);
}

/*@
   NEPSLPGetEPSLeft - Retrieve the linear eigensolver object (EPS) associated
   to the nonlinear eigenvalue solver, used to compute left eigenvectors in the
   two-sided variant of SLP.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: NEPSLPSetEPSLeft(), NEPSetTwoSided()
@*/
PetscErrorCode NEPSLPGetEPSLeft(NEP nep,EPS *eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(eps,2);
  CHKERRQ(PetscUseMethod(nep,"NEPSLPGetEPSLeft_C",(NEP,EPS*),(nep,eps)));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSLPSetKSP_SLP(NEP nep,KSP ksp)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)ksp));
  CHKERRQ(KSPDestroy(&ctx->ksp));
  ctx->ksp = ksp;
  CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp));
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPSLPSetKSP - Associate a linear solver object (KSP) to the nonlinear
   eigenvalue solver.

   Collective on nep

   Input Parameters:
+  nep - eigenvalue solver
-  ksp - the linear solver object

   Level: advanced

.seealso: NEPSLPGetKSP()
@*/
PetscErrorCode NEPSLPSetKSP(NEP nep,KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(nep,1,ksp,2);
  CHKERRQ(PetscTryMethod(nep,"NEPSLPSetKSP_C",(NEP,KSP),(nep,ksp)));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSLPGetKSP_SLP(NEP nep,KSP *ksp)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  if (!ctx->ksp) {
    CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)nep),&ctx->ksp));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)nep,1));
    CHKERRQ(KSPSetOptionsPrefix(ctx->ksp,((PetscObject)nep)->prefix));
    CHKERRQ(KSPAppendOptionsPrefix(ctx->ksp,"nep_slp_"));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp));
    CHKERRQ(PetscObjectSetOptions((PetscObject)ctx->ksp,((PetscObject)nep)->options));
    CHKERRQ(KSPSetErrorIfNotConverged(ctx->ksp,PETSC_TRUE));
    CHKERRQ(KSPSetTolerances(ctx->ksp,SlepcDefaultTol(nep->tol),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  }
  *ksp = ctx->ksp;
  PetscFunctionReturn(0);
}

/*@
   NEPSLPGetKSP - Retrieve the linear solver object (KSP) associated with
   the nonlinear eigenvalue solver.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  ksp - the linear solver object

   Level: advanced

.seealso: NEPSLPSetKSP()
@*/
PetscErrorCode NEPSLPGetKSP(NEP nep,KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ksp,2);
  CHKERRQ(PetscUseMethod(nep,"NEPSLPGetKSP_C",(NEP,KSP*),(nep,ksp)));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPView_SLP(NEP nep,PetscViewer viewer)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    if (ctx->deftol) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  deflation threshold: %g\n",(double)ctx->deftol));
    }
    if (!ctx->eps) CHKERRQ(NEPSLPGetEPS(nep,&ctx->eps));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(EPSView(ctx->eps,viewer));
    if (nep->twosided) {
      if (!ctx->epsts) CHKERRQ(NEPSLPGetEPSLeft(nep,&ctx->epsts));
      CHKERRQ(EPSView(ctx->epsts,viewer));
    }
    if (!ctx->ksp) CHKERRQ(NEPSLPGetKSP(nep,&ctx->ksp));
    CHKERRQ(KSPView(ctx->ksp,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPReset_SLP(NEP nep)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  CHKERRQ(EPSReset(ctx->eps));
  if (nep->twosided) CHKERRQ(EPSReset(ctx->epsts));
  CHKERRQ(KSPReset(ctx->ksp));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDestroy_SLP(NEP nep)
{
  NEP_SLP        *ctx = (NEP_SLP*)nep->data;

  PetscFunctionBegin;
  CHKERRQ(KSPDestroy(&ctx->ksp));
  CHKERRQ(EPSDestroy(&ctx->eps));
  CHKERRQ(EPSDestroy(&ctx->epsts));
  CHKERRQ(PetscFree(nep->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPSetDeflationThreshold_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPGetDeflationThreshold_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPSetEPS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPGetEPS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPSetEPSLeft_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPGetEPSLeft_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPSetKSP_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPGetKSP_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode NEPCreate_SLP(NEP nep)
{
  NEP_SLP        *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(nep,&ctx));
  nep->data = (void*)ctx;

  nep->useds  = PETSC_TRUE;
  ctx->deftol = PETSC_DEFAULT;

  nep->ops->solve          = NEPSolve_SLP;
  nep->ops->setup          = NEPSetUp_SLP;
  nep->ops->setfromoptions = NEPSetFromOptions_SLP;
  nep->ops->reset          = NEPReset_SLP;
  nep->ops->destroy        = NEPDestroy_SLP;
  nep->ops->view           = NEPView_SLP;
  nep->ops->computevectors = NEPComputeVectors_Schur;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPSetDeflationThreshold_C",NEPSLPSetDeflationThreshold_SLP));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPGetDeflationThreshold_C",NEPSLPGetDeflationThreshold_SLP));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPSetEPS_C",NEPSLPSetEPS_SLP));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPGetEPS_C",NEPSLPGetEPS_SLP));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPSetEPSLeft_C",NEPSLPSetEPSLeft_SLP));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPGetEPSLeft_C",NEPSLPGetEPSLeft_SLP));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPSetKSP_C",NEPSLPSetKSP_SLP));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)nep,"NEPSLPGetKSP_C",NEPSLPGetKSP_SLP));
  PetscFunctionReturn(0);
}
