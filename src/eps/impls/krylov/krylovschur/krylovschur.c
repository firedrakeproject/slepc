/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur

   Algorithm:

       Single-vector Krylov-Schur method for non-symmetric problems,
       including harmonic extraction.

   References:

       [1] "Krylov-Schur Methods in SLEPc", SLEPc Technical Report STR-7,
           available at https://slepc.upv.es.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Anal. App. 23(3):601-614, 2001.

       [3] "Practical Implementation of Harmonic Krylov-Schur", SLEPc Technical
            Report STR-9, available at https://slepc.upv.es.
*/

#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/
#include "krylovschur.h"

PetscErrorCode EPSGetArbitraryValues(EPS eps,PetscScalar *rr,PetscScalar *ri)
{
  PetscInt       i,newi,ld,n,l;
  Vec            xr=eps->work[0],xi=eps->work[1];
  PetscScalar    re,im,*Zr,*Zi,*X;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(DSGetDimensions(eps->ds,&n,&l,NULL,NULL));
  for (i=l;i<n;i++) {
    re = eps->eigr[i];
    im = eps->eigi[i];
    CHKERRQ(STBackTransform(eps->st,1,&re,&im));
    newi = i;
    CHKERRQ(DSVectors(eps->ds,DS_MAT_X,&newi,NULL));
    CHKERRQ(DSGetArray(eps->ds,DS_MAT_X,&X));
    Zr = X+i*ld;
    if (newi==i+1) Zi = X+newi*ld;
    else Zi = NULL;
    CHKERRQ(EPSComputeRitzVector(eps,Zr,Zi,eps->V,xr,xi));
    CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_X,&X));
    CHKERRQ((*eps->arbitrary)(re,im,xr,xi,rr+i,ri+i,eps->arbitraryctx));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSSetUp_KrylovSchur_Filter(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscBool       estimaterange=PETSC_TRUE;
  PetscReal       rleft,rright;
  Mat             A;

  PetscFunctionBegin;
  EPSCheckHermitianCondition(eps,PETSC_TRUE," with polynomial filter");
  EPSCheckStandardCondition(eps,PETSC_TRUE," with polynomial filter");
  PetscCheck(eps->intb<PETSC_MAX_REAL || eps->inta>PETSC_MIN_REAL,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"The defined computational interval should have at least one of their sides bounded");
  EPSCheckUnsupportedCondition(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_EXTRACTION,PETSC_TRUE," with polynomial filter");
  if (eps->tol==PETSC_DEFAULT) eps->tol = SLEPC_DEFAULT_TOL*1e-2;  /* use tighter tolerance */
  CHKERRQ(STFilterSetInterval(eps->st,eps->inta,eps->intb));
  if (!ctx->estimatedrange) {
    CHKERRQ(STFilterGetRange(eps->st,&rleft,&rright));
    estimaterange = (!rleft && !rright)? PETSC_TRUE: PETSC_FALSE;
  }
  if (estimaterange) { /* user did not set a range */
    CHKERRQ(STGetMatrix(eps->st,0,&A));
    CHKERRQ(MatEstimateSpectralRange_EPS(A,&rleft,&rright));
    CHKERRQ(PetscInfo(eps,"Setting eigenvalue range to [%g,%g]\n",(double)rleft,(double)rright));
    CHKERRQ(STFilterSetRange(eps->st,rleft,rright));
    ctx->estimatedrange = PETSC_TRUE;
  }
  if (eps->ncv==PETSC_DEFAULT && eps->nev==1) eps->nev = 40;  /* user did not provide nev estimation */
  CHKERRQ(EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd));
  PetscCheck(eps->ncv<=eps->nev+eps->mpd,PetscObjectComm((PetscObject)eps),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nev+mpd");
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUp_KrylovSchur(EPS eps)
{
  PetscReal         eta;
  PetscBool         isfilt=PETSC_FALSE;
  BVOrthogType      otype;
  BVOrthogBlockType obtype;
  EPS_KRYLOVSCHUR   *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  enum { EPS_KS_DEFAULT,EPS_KS_SYMM,EPS_KS_SLICE,EPS_KS_FILTER,EPS_KS_INDEF,EPS_KS_TWOSIDED } variant;

  PetscFunctionBegin;
  if (eps->which==EPS_ALL) {  /* default values in case of spectrum slicing or polynomial filter  */
    CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STFILTER,&isfilt));
    if (isfilt) CHKERRQ(EPSSetUp_KrylovSchur_Filter(eps));
    else CHKERRQ(EPSSetUp_KrylovSchur_Slice(eps));
  } else {
    CHKERRQ(EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd));
    PetscCheck(eps->ncv<=eps->nev+eps->mpd,PetscObjectComm((PetscObject)eps),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nev+mpd");
    if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
    if (!eps->which) CHKERRQ(EPSSetWhichEigenpairs_Default(eps));
  }
  PetscCheck(ctx->lock || eps->mpd>=eps->ncv,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Should not use mpd parameter in non-locking variant");

  EPSCheckDefiniteCondition(eps,eps->arbitrary," with arbitrary selection of eigenpairs");

  PetscCheck(eps->extraction==EPS_RITZ || eps->extraction==EPS_HARMONIC,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");

  if (!ctx->keep) ctx->keep = 0.5;

  CHKERRQ(EPSAllocateSolution(eps,1));
  CHKERRQ(EPS_SetInnerProduct(eps));
  if (eps->arbitrary) CHKERRQ(EPSSetWorkVecs(eps,2));
  else if (eps->ishermitian && !eps->ispositive) CHKERRQ(EPSSetWorkVecs(eps,1));

  /* dispatch solve method */
  if (eps->ishermitian) {
    if (eps->which==EPS_ALL) {
      PetscCheck(!eps->isgeneralized || eps->ispositive,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Spectrum slicing not implemented for indefinite problems");
      else variant = isfilt? EPS_KS_FILTER: EPS_KS_SLICE;
    } else if (eps->isgeneralized && !eps->ispositive) {
      variant = EPS_KS_INDEF;
    } else {
      switch (eps->extraction) {
        case EPS_RITZ:     variant = EPS_KS_SYMM; break;
        case EPS_HARMONIC: variant = EPS_KS_DEFAULT; break;
        default: SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");
      }
    }
  } else if (eps->twosided) {
    variant = EPS_KS_TWOSIDED;
  } else {
    switch (eps->extraction) {
      case EPS_RITZ:     variant = EPS_KS_DEFAULT; break;
      case EPS_HARMONIC: variant = EPS_KS_DEFAULT; break;
      default: SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");
    }
  }
  switch (variant) {
    case EPS_KS_DEFAULT:
      eps->ops->solve = EPSSolve_KrylovSchur_Default;
      eps->ops->computevectors = EPSComputeVectors_Schur;
      CHKERRQ(DSSetType(eps->ds,DSNHEP));
      CHKERRQ(DSSetExtraRow(eps->ds,PETSC_TRUE));
      CHKERRQ(DSAllocate(eps->ds,eps->ncv+1));
      break;
    case EPS_KS_SYMM:
    case EPS_KS_FILTER:
      eps->ops->solve = EPSSolve_KrylovSchur_Default;
      eps->ops->computevectors = EPSComputeVectors_Hermitian;
      CHKERRQ(DSSetType(eps->ds,DSHEP));
      CHKERRQ(DSSetCompact(eps->ds,PETSC_TRUE));
      CHKERRQ(DSSetExtraRow(eps->ds,PETSC_TRUE));
      CHKERRQ(DSAllocate(eps->ds,eps->ncv+1));
      break;
    case EPS_KS_SLICE:
      eps->ops->solve = EPSSolve_KrylovSchur_Slice;
      eps->ops->computevectors = EPSComputeVectors_Slice;
      break;
    case EPS_KS_INDEF:
      eps->ops->solve = EPSSolve_KrylovSchur_Indefinite;
      eps->ops->computevectors = EPSComputeVectors_Indefinite;
      CHKERRQ(DSSetType(eps->ds,DSGHIEP));
      CHKERRQ(DSSetCompact(eps->ds,PETSC_TRUE));
      CHKERRQ(DSSetExtraRow(eps->ds,PETSC_TRUE));
      CHKERRQ(DSAllocate(eps->ds,eps->ncv+1));
      /* force reorthogonalization for pseudo-Lanczos */
      CHKERRQ(BVGetOrthogonalization(eps->V,&otype,NULL,&eta,&obtype));
      CHKERRQ(BVSetOrthogonalization(eps->V,otype,BV_ORTHOG_REFINE_ALWAYS,eta,obtype));
      break;
    case EPS_KS_TWOSIDED:
      eps->ops->solve = EPSSolve_KrylovSchur_TwoSided;
      eps->ops->computevectors = EPSComputeVectors_Schur;
      CHKERRQ(DSSetType(eps->ds,DSNHEPTS));
      CHKERRQ(DSAllocate(eps->ds,eps->ncv+1));
      CHKERRQ(DSSetExtraRow(eps->ds,PETSC_TRUE));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_PLIB,"Unexpected error");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUpSort_KrylovSchur(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  SlepcSC         sc;
  PetscBool       isfilt;

  PetscFunctionBegin;
  CHKERRQ(EPSSetUpSort_Default(eps));
  if (eps->which==EPS_ALL) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STFILTER,&isfilt));
    if (isfilt) {
      CHKERRQ(DSGetSlepcSC(eps->ds,&sc));
      sc->rg            = NULL;
      sc->comparison    = SlepcCompareLargestReal;
      sc->comparisonctx = NULL;
      sc->map           = NULL;
      sc->mapobj        = NULL;
    } else {
      if (!ctx->global && ctx->sr->numEigs>0) {
        CHKERRQ(DSGetSlepcSC(eps->ds,&sc));
        sc->rg            = NULL;
        sc->comparison    = SlepcCompareLargestMagnitude;
        sc->comparisonctx = NULL;
        sc->map           = NULL;
        sc->mapobj        = NULL;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_KrylovSchur_Default(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        j,*pj,k,l,nv,ld,nconv;
  Mat             U,Op,H;
  PetscScalar     *g;
  PetscReal       beta,gamma=1.0,*a,*b;
  PetscBool       breakdown,harmonic,hermitian;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  harmonic = (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC)?PETSC_TRUE:PETSC_FALSE;
  hermitian = (eps->ishermitian && !harmonic)?PETSC_TRUE:PETSC_FALSE;
  if (harmonic) CHKERRQ(PetscMalloc1(ld,&g));
  if (eps->arbitrary) pj = &j;
  else pj = NULL;

  /* Get the starting Arnoldi vector */
  CHKERRQ(EPSGetStartVector(eps,0,NULL));
  l = 0;

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    CHKERRQ(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    CHKERRQ(STGetOperator(eps->st,&Op));
    if (hermitian) {
      CHKERRQ(DSGetArrayReal(eps->ds,DS_MAT_T,&a));
      b = a + ld;
      CHKERRQ(BVMatLanczos(eps->V,Op,a,b,eps->nconv+l,&nv,&breakdown));
      beta = b[nv-1];
      CHKERRQ(DSRestoreArrayReal(eps->ds,DS_MAT_T,&a));
    } else {
      CHKERRQ(DSGetMat(eps->ds,DS_MAT_A,&H));
      CHKERRQ(BVMatArnoldi(eps->V,Op,H,eps->nconv+l,&nv,&beta,&breakdown));
      CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_A,&H));
    }
    CHKERRQ(STRestoreOperator(eps->st,&Op));
    CHKERRQ(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    CHKERRQ(DSSetState(eps->ds,l?DS_STATE_RAW:DS_STATE_INTERMEDIATE));
    CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv,nv));

    /* Compute translation of Krylov decomposition if harmonic extraction used */
    if (PetscUnlikely(harmonic)) CHKERRQ(DSTranslateHarmonic(eps->ds,eps->target,beta,PETSC_FALSE,g,&gamma));

    /* Solve projected problem */
    CHKERRQ(DSSolve(eps->ds,eps->eigr,eps->eigi));
    if (PetscUnlikely(eps->arbitrary)) {
      CHKERRQ(EPSGetArbitraryValues(eps,eps->rr,eps->ri));
      j=1;
    }
    CHKERRQ(DSSort(eps->ds,eps->eigr,eps->eigi,eps->rr,eps->ri,pj));
    CHKERRQ(DSUpdateExtraRow(eps->ds));
    CHKERRQ(DSSynchronize(eps->ds,eps->eigr,eps->eigi));

    /* Check convergence */
    CHKERRQ(EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,nv-eps->nconv,beta,0.0,gamma,&k));
    CHKERRQ((*eps->stopping)(eps,eps->its,eps->max_it,k,eps->nev,&eps->reason,eps->stoppingctx));
    nconv = k;

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      if (!hermitian) CHKERRQ(DSGetTruncateSize(eps->ds,k,nv,&l));
    }
    if (!ctx->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */
    if (l) CHKERRQ(PetscInfo(eps,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (PetscUnlikely(breakdown || k==nv)) {
        /* Start a new Arnoldi factorization */
        CHKERRQ(PetscInfo(eps,"Breakdown in Krylov-Schur method (it=%" PetscInt_FMT " norm=%g)\n",eps->its,(double)beta));
        if (k<eps->nev) {
          CHKERRQ(EPSGetStartVector(eps,k,&breakdown));
          if (breakdown) {
            eps->reason = EPS_DIVERGED_BREAKDOWN;
            CHKERRQ(PetscInfo(eps,"Unable to generate more start vectors\n"));
          }
        }
      } else {
        /* Undo translation of Krylov decomposition */
        if (PetscUnlikely(harmonic)) {
          CHKERRQ(DSSetDimensions(eps->ds,nv,k,l));
          CHKERRQ(DSTranslateHarmonic(eps->ds,0.0,beta,PETSC_TRUE,g,&gamma));
          /* gamma u^ = u - U*g~ */
          CHKERRQ(BVSetActiveColumns(eps->V,0,nv));
          CHKERRQ(BVMultColumn(eps->V,-1.0,1.0,nv,g));
          CHKERRQ(BVScaleColumn(eps->V,nv,1.0/gamma));
          CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv,nv));
          CHKERRQ(DSSetDimensions(eps->ds,nv,k,nv));
        }
        /* Prepare the Rayleigh quotient for restart */
        CHKERRQ(DSTruncate(eps->ds,k+l,PETSC_FALSE));
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_Q,&U));
    CHKERRQ(BVMultInPlace(eps->V,U,eps->nconv,k+l));
    CHKERRQ(MatDestroy(&U));

    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) CHKERRQ(BVCopyColumn(eps->V,nv,k+l));
    eps->nconv = k;
    CHKERRQ(EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,nv));
  }

  if (harmonic) CHKERRQ(PetscFree(g));
  CHKERRQ(DSTruncate(eps->ds,eps->nconv,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurSetRestart_KrylovSchur(EPS eps,PetscReal keep)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) ctx->keep = 0.5;
  else {
    PetscCheck(keep>=0.1 && keep<=0.9,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument %g must be in the range [0.1,0.9]",(double)keep);
    ctx->keep = keep;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurSetRestart - Sets the restart parameter for the Krylov-Schur
   method, in particular the proportion of basis vectors that must be kept
   after restart.

   Logically Collective on eps

   Input Parameters:
+  eps - the eigenproblem solver context
-  keep - the number of vectors to be kept at restart

   Options Database Key:
.  -eps_krylovschur_restart - Sets the restart parameter

   Notes:
   Allowed values are in the range [0.1,0.9]. The default is 0.5.

   Level: advanced

.seealso: EPSKrylovSchurGetRestart()
@*/
PetscErrorCode EPSKrylovSchurSetRestart(EPS eps,PetscReal keep)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,keep,2);
  CHKERRQ(PetscTryMethod(eps,"EPSKrylovSchurSetRestart_C",(EPS,PetscReal),(eps,keep)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetRestart_KrylovSchur(EPS eps,PetscReal *keep)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  *keep = ctx->keep;
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurGetRestart - Gets the restart parameter used in the
   Krylov-Schur method.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  keep - the restart parameter

   Level: advanced

.seealso: EPSKrylovSchurSetRestart()
@*/
PetscErrorCode EPSKrylovSchurGetRestart(EPS eps,PetscReal *keep)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidRealPointer(keep,2);
  CHKERRQ(PetscUseMethod(eps,"EPSKrylovSchurGetRestart_C",(EPS,PetscReal*),(eps,keep)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurSetLocking_KrylovSchur(EPS eps,PetscBool lock)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  ctx->lock = lock;
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurSetLocking - Choose between locking and non-locking variants of
   the Krylov-Schur method.

   Logically Collective on eps

   Input Parameters:
+  eps  - the eigenproblem solver context
-  lock - true if the locking variant must be selected

   Options Database Key:
.  -eps_krylovschur_locking - Sets the locking flag

   Notes:
   The default is to lock converged eigenpairs when the method restarts.
   This behaviour can be changed so that all directions are kept in the
   working subspace even if already converged to working accuracy (the
   non-locking variant).

   Level: advanced

.seealso: EPSKrylovSchurGetLocking()
@*/
PetscErrorCode EPSKrylovSchurSetLocking(EPS eps,PetscBool lock)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,lock,2);
  CHKERRQ(PetscTryMethod(eps,"EPSKrylovSchurSetLocking_C",(EPS,PetscBool),(eps,lock)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetLocking_KrylovSchur(EPS eps,PetscBool *lock)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  *lock = ctx->lock;
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurGetLocking - Gets the locking flag used in the Krylov-Schur
   method.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  lock - the locking flag

   Level: advanced

.seealso: EPSKrylovSchurSetLocking()
@*/
PetscErrorCode EPSKrylovSchurGetLocking(EPS eps,PetscBool *lock)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidBoolPointer(lock,2);
  CHKERRQ(PetscUseMethod(eps,"EPSKrylovSchurGetLocking_C",(EPS,PetscBool*),(eps,lock)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurSetPartitions_KrylovSchur(EPS eps,PetscInt npart)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscMPIInt     size;

  PetscFunctionBegin;
  if (ctx->npart!=npart) {
    if (ctx->commset) CHKERRQ(PetscSubcommDestroy(&ctx->subc));
    CHKERRQ(EPSDestroy(&ctx->eps));
  }
  if (npart == PETSC_DEFAULT || npart == PETSC_DECIDE) {
    ctx->npart = 1;
  } else {
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)eps),&size));
    PetscCheck(npart>0 && npart<=size,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of npart");
    ctx->npart = npart;
  }
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurSetPartitions - Sets the number of partitions for the
   case of doing spectrum slicing for a computational interval with the
   communicator split in several sub-communicators.

   Logically Collective on eps

   Input Parameters:
+  eps   - the eigenproblem solver context
-  npart - number of partitions

   Options Database Key:
.  -eps_krylovschur_partitions <npart> - Sets the number of partitions

   Notes:
   By default, npart=1 so all processes in the communicator participate in
   the processing of the whole interval. If npart>1 then the interval is
   divided into npart subintervals, each of them being processed by a
   subset of processes.

   The interval is split proportionally unless the separation points are
   specified with EPSKrylovSchurSetSubintervals().

   Level: advanced

.seealso: EPSKrylovSchurSetSubintervals(), EPSSetInterval()
@*/
PetscErrorCode EPSKrylovSchurSetPartitions(EPS eps,PetscInt npart)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,npart,2);
  CHKERRQ(PetscTryMethod(eps,"EPSKrylovSchurSetPartitions_C",(EPS,PetscInt),(eps,npart)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetPartitions_KrylovSchur(EPS eps,PetscInt *npart)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  *npart  = ctx->npart;
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurGetPartitions - Gets the number of partitions of the
   communicator in case of spectrum slicing.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  npart - number of partitions

   Level: advanced

.seealso: EPSKrylovSchurSetPartitions()
@*/
PetscErrorCode EPSKrylovSchurGetPartitions(EPS eps,PetscInt *npart)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(npart,2);
  CHKERRQ(PetscUseMethod(eps,"EPSKrylovSchurGetPartitions_C",(EPS,PetscInt*),(eps,npart)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurSetDetectZeros_KrylovSchur(EPS eps,PetscBool detect)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  ctx->detect = detect;
  eps->state  = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurSetDetectZeros - Sets a flag to enforce detection of
   zeros during the factorizations throughout the spectrum slicing computation.

   Logically Collective on eps

   Input Parameters:
+  eps    - the eigenproblem solver context
-  detect - check for zeros

   Options Database Key:
.  -eps_krylovschur_detect_zeros - Check for zeros; this takes an optional
   bool value (0/1/no/yes/true/false)

   Notes:
   A zero in the factorization indicates that a shift coincides with an eigenvalue.

   This flag is turned off by default, and may be necessary in some cases,
   especially when several partitions are being used. This feature currently
   requires an external package for factorizations with support for zero
   detection, e.g. MUMPS.

   Level: advanced

.seealso: EPSKrylovSchurSetPartitions(), EPSSetInterval()
@*/
PetscErrorCode EPSKrylovSchurSetDetectZeros(EPS eps,PetscBool detect)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,detect,2);
  CHKERRQ(PetscTryMethod(eps,"EPSKrylovSchurSetDetectZeros_C",(EPS,PetscBool),(eps,detect)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetDetectZeros_KrylovSchur(EPS eps,PetscBool *detect)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  *detect = ctx->detect;
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurGetDetectZeros - Gets the flag that enforces zero detection
   in spectrum slicing.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  detect - whether zeros detection is enforced during factorizations

   Level: advanced

.seealso: EPSKrylovSchurSetDetectZeros()
@*/
PetscErrorCode EPSKrylovSchurGetDetectZeros(EPS eps,PetscBool *detect)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidBoolPointer(detect,2);
  CHKERRQ(PetscUseMethod(eps,"EPSKrylovSchurGetDetectZeros_C",(EPS,PetscBool*),(eps,detect)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurSetDimensions_KrylovSchur(EPS eps,PetscInt nev,PetscInt ncv,PetscInt mpd)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  PetscCheck(nev>0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of nev. Must be > 0");
  ctx->nev = nev;
  if (ncv == PETSC_DECIDE || ncv == PETSC_DEFAULT) {
    ctx->ncv = PETSC_DEFAULT;
  } else {
    PetscCheck(ncv>0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
    ctx->ncv = ncv;
  }
  if (mpd == PETSC_DECIDE || mpd == PETSC_DEFAULT) {
    ctx->mpd = PETSC_DEFAULT;
  } else {
    PetscCheck(mpd>0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of mpd. Must be > 0");
    ctx->mpd = mpd;
  }
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurSetDimensions - Sets the dimensions used for each subsolve
   step in case of doing spectrum slicing for a computational interval.
   The meaning of the parameters is the same as in EPSSetDimensions().

   Logically Collective on eps

   Input Parameters:
+  eps - the eigenproblem solver context
.  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the subsolve
-  mpd - the maximum dimension allowed for the projected problem

   Options Database Key:
+  -eps_krylovschur_nev <nev> - Sets the number of eigenvalues
.  -eps_krylovschur_ncv <ncv> - Sets the dimension of the subspace
-  -eps_krylovschur_mpd <mpd> - Sets the maximum projected dimension

   Level: advanced

.seealso: EPSKrylovSchurGetDimensions(), EPSSetDimensions(), EPSSetInterval()
@*/
PetscErrorCode EPSKrylovSchurSetDimensions(EPS eps,PetscInt nev,PetscInt ncv,PetscInt mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,nev,2);
  PetscValidLogicalCollectiveInt(eps,ncv,3);
  PetscValidLogicalCollectiveInt(eps,mpd,4);
  CHKERRQ(PetscTryMethod(eps,"EPSKrylovSchurSetDimensions_C",(EPS,PetscInt,PetscInt,PetscInt),(eps,nev,ncv,mpd)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetDimensions_KrylovSchur(EPS eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  if (nev) *nev = ctx->nev;
  if (ncv) *ncv = ctx->ncv;
  if (mpd) *mpd = ctx->mpd;
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurGetDimensions - Gets the dimensions used for each subsolve
   step in case of doing spectrum slicing for a computational interval.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the subsolve
-  mpd - the maximum dimension allowed for the projected problem

   Level: advanced

.seealso: EPSKrylovSchurSetDimensions()
@*/
PetscErrorCode EPSKrylovSchurGetDimensions(EPS eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  CHKERRQ(PetscUseMethod(eps,"EPSKrylovSchurGetDimensions_C",(EPS,PetscInt*,PetscInt*,PetscInt*),(eps,nev,ncv,mpd)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurSetSubintervals_KrylovSchur(EPS eps,PetscReal* subint)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i;

  PetscFunctionBegin;
  PetscCheck(subint[0]==eps->inta && subint[ctx->npart]==eps->intb,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"First and last values must match the endpoints of EPSSetInterval()");
  for (i=0;i<ctx->npart;i++) PetscCheck(subint[i]<=subint[i+1],PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"Array must contain values in ascending order");
  if (ctx->subintervals) CHKERRQ(PetscFree(ctx->subintervals));
  CHKERRQ(PetscMalloc1(ctx->npart+1,&ctx->subintervals));
  for (i=0;i<ctx->npart+1;i++) ctx->subintervals[i] = subint[i];
  ctx->subintset = PETSC_TRUE;
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@C
   EPSKrylovSchurSetSubintervals - Sets the points that delimit the
   subintervals to be used in spectrum slicing with several partitions.

   Logically Collective on eps

   Input Parameters:
+  eps    - the eigenproblem solver context
-  subint - array of real values specifying subintervals

   Notes:
   This function must be called after EPSKrylovSchurSetPartitions(). For npart
   partitions, the argument subint must contain npart+1 real values sorted in
   ascending order, subint_0, subint_1, ..., subint_npart, where the first
   and last values must coincide with the interval endpoints set with
   EPSSetInterval().

   The subintervals are then defined by two consecutive points [subint_0,subint_1],
   [subint_1,subint_2], and so on.

   Level: advanced

.seealso: EPSKrylovSchurSetPartitions(), EPSKrylovSchurGetSubintervals(), EPSSetInterval()
@*/
PetscErrorCode EPSKrylovSchurSetSubintervals(EPS eps,PetscReal *subint)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  CHKERRQ(PetscTryMethod(eps,"EPSKrylovSchurSetSubintervals_C",(EPS,PetscReal*),(eps,subint)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetSubintervals_KrylovSchur(EPS eps,PetscReal **subint)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i;

  PetscFunctionBegin;
  if (!ctx->subintset) {
    PetscCheck(eps->state,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Must call EPSSetUp() first");
    PetscCheck(ctx->sr,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Only available in interval computations, see EPSSetInterval()");
  }
  CHKERRQ(PetscMalloc1(ctx->npart+1,subint));
  for (i=0;i<=ctx->npart;i++) (*subint)[i] = ctx->subintervals[i];
  PetscFunctionReturn(0);
}

/*@C
   EPSKrylovSchurGetSubintervals - Returns the points that delimit the
   subintervals used in spectrum slicing with several partitions.

   Logically Collective on eps

   Input Parameter:
.  eps    - the eigenproblem solver context

   Output Parameter:
.  subint - array of real values specifying subintervals

   Notes:
   If the user passed values with EPSKrylovSchurSetSubintervals(), then the
   same values are returned. Otherwise, the values computed internally are
   obtained.

   This function is only available for spectrum slicing runs.

   The returned array has length npart+1 (see EPSKrylovSchurGetPartitions())
   and should be freed by the user.

   Fortran Notes:
   The calling sequence from Fortran is
.vb
   EPSKrylovSchurGetSubintervals(eps,subint,ierr)
   double precision subint(npart+1) output
.ve

   Level: advanced

.seealso: EPSKrylovSchurSetSubintervals(), EPSKrylovSchurGetPartitions(), EPSSetInterval()
@*/
PetscErrorCode EPSKrylovSchurGetSubintervals(EPS eps,PetscReal **subint)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(subint,2);
  CHKERRQ(PetscUseMethod(eps,"EPSKrylovSchurGetSubintervals_C",(EPS,PetscReal**),(eps,subint)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetInertias_KrylovSchur(EPS eps,PetscInt *n,PetscReal **shifts,PetscInt **inertias)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i,numsh;
  EPS_SR          sr = ctx->sr;

  PetscFunctionBegin;
  PetscCheck(eps->state,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Must call EPSSetUp() first");
  PetscCheck(ctx->sr,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Only available in interval computations, see EPSSetInterval()");
  switch (eps->state) {
  case EPS_STATE_INITIAL:
    break;
  case EPS_STATE_SETUP:
    numsh = ctx->npart+1;
    if (n) *n = numsh;
    if (shifts) {
      CHKERRQ(PetscMalloc1(numsh,shifts));
      (*shifts)[0] = eps->inta;
      if (ctx->npart==1) (*shifts)[1] = eps->intb;
      else for (i=1;i<numsh;i++) (*shifts)[i] = ctx->subintervals[i];
    }
    if (inertias) {
      CHKERRQ(PetscMalloc1(numsh,inertias));
      (*inertias)[0] = (sr->dir==1)?sr->inertia0:sr->inertia1;
      if (ctx->npart==1) (*inertias)[1] = (sr->dir==1)?sr->inertia1:sr->inertia0;
      else for (i=1;i<numsh;i++) (*inertias)[i] = (*inertias)[i-1]+ctx->nconv_loc[i-1];
    }
    break;
  case EPS_STATE_SOLVED:
  case EPS_STATE_EIGENVECTORS:
    numsh = ctx->nshifts;
    if (n) *n = numsh;
    if (shifts) {
      CHKERRQ(PetscMalloc1(numsh,shifts));
      for (i=0;i<numsh;i++) (*shifts)[i] = ctx->shifts[i];
    }
    if (inertias) {
      CHKERRQ(PetscMalloc1(numsh,inertias));
      for (i=0;i<numsh;i++) (*inertias)[i] = ctx->inertias[i];
    }
    break;
  }
  PetscFunctionReturn(0);
}

/*@C
   EPSKrylovSchurGetInertias - Gets the values of the shifts and their
   corresponding inertias in case of doing spectrum slicing for a
   computational interval.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  n        - number of shifts, including the endpoints of the interval
.  shifts   - the values of the shifts used internally in the solver
-  inertias - the values of the inertia in each shift

   Notes:
   If called after EPSSolve(), all shifts used internally by the solver are
   returned (including both endpoints and any intermediate ones). If called
   before EPSSolve() and after EPSSetUp() then only the information of the
   endpoints of subintervals is available.

   This function is only available for spectrum slicing runs.

   The returned arrays should be freed by the user. Can pass NULL in any of
   the two arrays if not required.

   Fortran Notes:
   The calling sequence from Fortran is
.vb
   EPSKrylovSchurGetInertias(eps,n,shifts,inertias,ierr)
   integer n
   double precision shifts(*)
   integer inertias(*)
.ve
   The arrays should be at least of length n. The value of n can be determined
   by an initial call
.vb
   EPSKrylovSchurGetInertias(eps,n,PETSC_NULL_REAL,PETSC_NULL_INTEGER,ierr)
.ve

   Level: advanced

.seealso: EPSSetInterval(), EPSKrylovSchurSetSubintervals()
@*/
PetscErrorCode EPSKrylovSchurGetInertias(EPS eps,PetscInt *n,PetscReal **shifts,PetscInt **inertias)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(n,2);
  CHKERRQ(PetscUseMethod(eps,"EPSKrylovSchurGetInertias_C",(EPS,PetscInt*,PetscReal**,PetscInt**),(eps,n,shifts,inertias)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetSubcommInfo_KrylovSchur(EPS eps,PetscInt *k,PetscInt *n,Vec *v)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  EPS_SR          sr = ((EPS_KRYLOVSCHUR*)ctx->eps->data)->sr;

  PetscFunctionBegin;
  PetscCheck(eps->state,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Must call EPSSetUp() first");
  PetscCheck(ctx->sr,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Only available in interval computations, see EPSSetInterval()");
  if (k) *k = (ctx->npart==1)? 0: ctx->subc->color;
  if (n) *n = sr->numEigs;
  if (v) CHKERRQ(BVCreateVec(sr->V,v));
  PetscFunctionReturn(0);
}

/*@C
   EPSKrylovSchurGetSubcommInfo - Gets information related to the case of
   doing spectrum slicing for a computational interval with multiple
   communicators.

   Collective on the subcommunicator (if v is given)

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  k - index of the subinterval for the calling process
.  n - number of eigenvalues found in the k-th subinterval
-  v - a vector owned by processes in the subcommunicator with dimensions
       compatible for locally computed eigenvectors (or NULL)

   Notes:
   This function is only available for spectrum slicing runs.

   The returned Vec should be destroyed by the user.

   Level: advanced

.seealso: EPSSetInterval(), EPSKrylovSchurSetPartitions(), EPSKrylovSchurGetSubcommPairs()
@*/
PetscErrorCode EPSKrylovSchurGetSubcommInfo(EPS eps,PetscInt *k,PetscInt *n,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  CHKERRQ(PetscUseMethod(eps,"EPSKrylovSchurGetSubcommInfo_C",(EPS,PetscInt*,PetscInt*,Vec*),(eps,k,n,v)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetSubcommPairs_KrylovSchur(EPS eps,PetscInt i,PetscScalar *eig,Vec v)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  EPS_SR          sr = ((EPS_KRYLOVSCHUR*)ctx->eps->data)->sr;

  PetscFunctionBegin;
  EPSCheckSolved(eps,1);
  PetscCheck(ctx->sr,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Only available in interval computations, see EPSSetInterval()");
  PetscCheck(i>=0 && i<sr->numEigs,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Argument 2 out of range");
  if (eig) *eig = sr->eigr[sr->perm[i]];
  if (v) CHKERRQ(BVCopyVec(sr->V,sr->perm[i],v));
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurGetSubcommPairs - Gets the i-th eigenpair stored
   internally in the subcommunicator to which the calling process belongs.

   Collective on the subcommunicator (if v is given)

   Input Parameters:
+  eps - the eigenproblem solver context
-  i   - index of the solution

   Output Parameters:
+  eig - the eigenvalue
-  v   - the eigenvector

   Notes:
   It is allowed to pass NULL for v if the eigenvector is not required.
   Otherwise, the caller must provide a valid Vec objects, i.e.,
   it must be created by the calling program with EPSKrylovSchurGetSubcommInfo().

   The index i should be a value between 0 and n-1, where n is the number of
   vectors in the local subinterval, see EPSKrylovSchurGetSubcommInfo().

   Level: advanced

.seealso: EPSSetInterval(), EPSKrylovSchurSetPartitions(), EPSKrylovSchurGetSubcommInfo(), EPSKrylovSchurGetSubcommMats()
@*/
PetscErrorCode EPSKrylovSchurGetSubcommPairs(EPS eps,PetscInt i,PetscScalar *eig,Vec v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (v) PetscValidLogicalCollectiveInt(v,i,2);
  CHKERRQ(PetscUseMethod(eps,"EPSKrylovSchurGetSubcommPairs_C",(EPS,PetscInt,PetscScalar*,Vec),(eps,i,eig,v)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetSubcommMats_KrylovSchur(EPS eps,Mat *A,Mat *B)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  PetscCheck(ctx->sr,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Only available in interval computations, see EPSSetInterval()");
  PetscCheck(eps->state,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Must call EPSSetUp() first");
  CHKERRQ(EPSGetOperators(ctx->eps,A,B));
  PetscFunctionReturn(0);
}

/*@C
   EPSKrylovSchurGetSubcommMats - Gets the eigenproblem matrices stored
   internally in the subcommunicator to which the calling process belongs.

   Collective on the subcommunicator

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  A  - the matrix associated with the eigensystem
-  B  - the second matrix in the case of generalized eigenproblems

   Notes:
   This is the analog of EPSGetOperators(), but returns the matrices distributed
   differently (in the subcommunicator rather than in the parent communicator).

   These matrices should not be modified by the user.

   Level: advanced

.seealso: EPSSetInterval(), EPSKrylovSchurSetPartitions(), EPSKrylovSchurGetSubcommInfo()
@*/
PetscErrorCode EPSKrylovSchurGetSubcommMats(EPS eps,Mat *A,Mat *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  CHKERRQ(PetscTryMethod(eps,"EPSKrylovSchurGetSubcommMats_C",(EPS,Mat*,Mat*),(eps,A,B)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurUpdateSubcommMats_KrylovSchur(EPS eps,PetscScalar a,PetscScalar ap,Mat Au,PetscScalar b,PetscScalar bp, Mat Bu,MatStructure str,PetscBool globalup)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data,*subctx;
  Mat             A,B=NULL,Ag,Bg=NULL;
  PetscBool       reuse=PETSC_TRUE;

  PetscFunctionBegin;
  PetscCheck(ctx->sr,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Only available in interval computations, see EPSSetInterval()");
  PetscCheck(eps->state,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Must call EPSSetUp() first");
  CHKERRQ(EPSGetOperators(eps,&Ag,&Bg));
  CHKERRQ(EPSGetOperators(ctx->eps,&A,&B));

  CHKERRQ(MatScale(A,a));
  if (Au) CHKERRQ(MatAXPY(A,ap,Au,str));
  if (B) CHKERRQ(MatScale(B,b));
  if (Bu) CHKERRQ(MatAXPY(B,bp,Bu,str));
  CHKERRQ(EPSSetOperators(ctx->eps,A,B));

  /* Update stored matrix state */
  subctx = (EPS_KRYLOVSCHUR*)ctx->eps->data;
  CHKERRQ(PetscObjectStateGet((PetscObject)A,&subctx->Astate));
  if (B) CHKERRQ(PetscObjectStateGet((PetscObject)B,&subctx->Bstate));

  /* Update matrices in the parent communicator if requested by user */
  if (globalup) {
    if (ctx->npart>1) {
      if (!ctx->isrow) {
        CHKERRQ(MatGetOwnershipIS(Ag,&ctx->isrow,&ctx->iscol));
        reuse = PETSC_FALSE;
      }
      if (str==DIFFERENT_NONZERO_PATTERN || str==UNKNOWN_NONZERO_PATTERN) reuse = PETSC_FALSE;
      if (ctx->submata && !reuse) CHKERRQ(MatDestroyMatrices(1,&ctx->submata));
      CHKERRQ(MatCreateSubMatrices(A,1,&ctx->isrow,&ctx->iscol,(reuse)?MAT_REUSE_MATRIX:MAT_INITIAL_MATRIX,&ctx->submata));
      CHKERRQ(MatCreateMPIMatConcatenateSeqMat(((PetscObject)Ag)->comm,ctx->submata[0],PETSC_DECIDE,MAT_REUSE_MATRIX,&Ag));
      if (B) {
        if (ctx->submatb && !reuse) CHKERRQ(MatDestroyMatrices(1,&ctx->submatb));
        CHKERRQ(MatCreateSubMatrices(B,1,&ctx->isrow,&ctx->iscol,(reuse)?MAT_REUSE_MATRIX:MAT_INITIAL_MATRIX,&ctx->submatb));
        CHKERRQ(MatCreateMPIMatConcatenateSeqMat(((PetscObject)Bg)->comm,ctx->submatb[0],PETSC_DECIDE,MAT_REUSE_MATRIX,&Bg));
      }
    }
    CHKERRQ(PetscObjectStateGet((PetscObject)Ag,&ctx->Astate));
    if (Bg) CHKERRQ(PetscObjectStateGet((PetscObject)Bg,&ctx->Bstate));
  }
  CHKERRQ(EPSSetOperators(eps,Ag,Bg));
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurUpdateSubcommMats - Update the eigenproblem matrices stored
   internally in the subcommunicator to which the calling process belongs.

   Collective on eps

   Input Parameters:
+  eps - the eigenproblem solver context
.  s   - scalar that multiplies the existing A matrix
.  a   - scalar used in the axpy operation on A
.  Au  - matrix used in the axpy operation on A
.  t   - scalar that multiplies the existing B matrix
.  b   - scalar used in the axpy operation on B
.  Bu  - matrix used in the axpy operation on B
.  str - structure flag
-  globalup - flag indicating if global matrices must be updated

   Notes:
   This function modifies the eigenproblem matrices at the subcommunicator level,
   and optionally updates the global matrices in the parent communicator. The updates
   are expressed as A <-- s*A + a*Au,  B <-- t*B + b*Bu.

   It is possible to update one of the matrices, or both.

   The matrices Au and Bu must be equal in all subcommunicators.

   The str flag is passed to the MatAXPY() operations to perform the updates.

   If globalup is true, communication is carried out to reconstruct the updated
   matrices in the parent communicator. The user must be warned that if global
   matrices are not in sync with subcommunicator matrices, the errors computed
   by EPSComputeError() will be wrong even if the computed solution is correct
   (the synchronization may be done only once at the end).

   Level: advanced

.seealso: EPSSetInterval(), EPSKrylovSchurSetPartitions(), EPSKrylovSchurGetSubcommMats()
@*/
PetscErrorCode EPSKrylovSchurUpdateSubcommMats(EPS eps,PetscScalar s,PetscScalar a,Mat Au,PetscScalar t,PetscScalar b,Mat Bu,MatStructure str,PetscBool globalup)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveScalar(eps,s,2);
  PetscValidLogicalCollectiveScalar(eps,a,3);
  if (Au) PetscValidHeaderSpecific(Au,MAT_CLASSID,4);
  PetscValidLogicalCollectiveScalar(eps,t,5);
  PetscValidLogicalCollectiveScalar(eps,b,6);
  if (Bu) PetscValidHeaderSpecific(Bu,MAT_CLASSID,7);
  PetscValidLogicalCollectiveEnum(eps,str,8);
  PetscValidLogicalCollectiveBool(eps,globalup,9);
  CHKERRQ(PetscTryMethod(eps,"EPSKrylovSchurUpdateSubcommMats_C",(EPS,PetscScalar,PetscScalar,Mat,PetscScalar,PetscScalar,Mat,MatStructure,PetscBool),(eps,s,a,Au,t,b,Bu,str,globalup)));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSKrylovSchurGetChildEPS(EPS eps,EPS *childeps)
{
  EPS_KRYLOVSCHUR  *ctx=(EPS_KRYLOVSCHUR*)eps->data,*ctx_local;
  Mat              A,B=NULL,Ar=NULL,Br=NULL;
  PetscMPIInt      rank;
  PetscObjectState Astate,Bstate=0;
  PetscObjectId    Aid,Bid=0;
  STType           sttype;
  PetscInt         nmat;
  const char       *prefix;
  MPI_Comm         child;

  PetscFunctionBegin;
  CHKERRQ(EPSGetOperators(eps,&A,&B));
  if (ctx->npart==1) {
    if (!ctx->eps) CHKERRQ(EPSCreate(((PetscObject)eps)->comm,&ctx->eps));
    CHKERRQ(EPSGetOptionsPrefix(eps,&prefix));
    CHKERRQ(EPSSetOptionsPrefix(ctx->eps,prefix));
    CHKERRQ(EPSSetOperators(ctx->eps,A,B));
  } else {
    CHKERRQ(PetscObjectStateGet((PetscObject)A,&Astate));
    CHKERRQ(PetscObjectGetId((PetscObject)A,&Aid));
    if (B) {
      CHKERRQ(PetscObjectStateGet((PetscObject)B,&Bstate));
      CHKERRQ(PetscObjectGetId((PetscObject)B,&Bid));
    }
    if (!ctx->subc) {
      /* Create context for subcommunicators */
      CHKERRQ(PetscSubcommCreate(PetscObjectComm((PetscObject)eps),&ctx->subc));
      CHKERRQ(PetscSubcommSetNumber(ctx->subc,ctx->npart));
      CHKERRQ(PetscSubcommSetType(ctx->subc,PETSC_SUBCOMM_CONTIGUOUS));
      CHKERRQ(PetscLogObjectMemory((PetscObject)eps,sizeof(PetscSubcomm)));
      CHKERRQ(PetscSubcommGetChild(ctx->subc,&child));

      /* Duplicate matrices */
      CHKERRQ(MatCreateRedundantMatrix(A,0,child,MAT_INITIAL_MATRIX,&Ar));
      CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)Ar));
      ctx->Astate = Astate;
      ctx->Aid = Aid;
      CHKERRQ(MatPropagateSymmetryOptions(A,Ar));
      if (B) {
        CHKERRQ(MatCreateRedundantMatrix(B,0,child,MAT_INITIAL_MATRIX,&Br));
        CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)Br));
        ctx->Bstate = Bstate;
        ctx->Bid = Bid;
        CHKERRQ(MatPropagateSymmetryOptions(B,Br));
      }
    } else {
      CHKERRQ(PetscSubcommGetChild(ctx->subc,&child));
      if (ctx->Astate != Astate || (B && ctx->Bstate != Bstate) || ctx->Aid != Aid || (B && ctx->Bid != Bid)) {
        CHKERRQ(STGetNumMatrices(ctx->eps->st,&nmat));
        if (nmat) CHKERRQ(EPSGetOperators(ctx->eps,&Ar,&Br));
        CHKERRQ(MatCreateRedundantMatrix(A,0,child,MAT_INITIAL_MATRIX,&Ar));
        ctx->Astate = Astate;
        ctx->Aid = Aid;
        CHKERRQ(MatPropagateSymmetryOptions(A,Ar));
        if (B) {
          CHKERRQ(MatCreateRedundantMatrix(B,0,child,MAT_INITIAL_MATRIX,&Br));
          ctx->Bstate = Bstate;
          ctx->Bid = Bid;
          CHKERRQ(MatPropagateSymmetryOptions(B,Br));
        }
        CHKERRQ(EPSSetOperators(ctx->eps,Ar,Br));
        CHKERRQ(MatDestroy(&Ar));
        CHKERRQ(MatDestroy(&Br));
      }
    }

    /* Create auxiliary EPS */
    if (!ctx->eps) {
      CHKERRQ(EPSCreate(child,&ctx->eps));
      CHKERRQ(EPSGetOptionsPrefix(eps,&prefix));
      CHKERRQ(EPSSetOptionsPrefix(ctx->eps,prefix));
      CHKERRQ(EPSSetOperators(ctx->eps,Ar,Br));
      CHKERRQ(MatDestroy(&Ar));
      CHKERRQ(MatDestroy(&Br));
    }
    /* Create subcommunicator grouping processes with same rank */
    if (ctx->commset) CHKERRMPI(MPI_Comm_free(&ctx->commrank));
    CHKERRMPI(MPI_Comm_rank(child,&rank));
    CHKERRMPI(MPI_Comm_split(((PetscObject)eps)->comm,rank,ctx->subc->color,&ctx->commrank));
    ctx->commset = PETSC_TRUE;
  }
  CHKERRQ(EPSSetType(ctx->eps,((PetscObject)eps)->type_name));
  CHKERRQ(STGetType(eps->st,&sttype));
  CHKERRQ(STSetType(ctx->eps->st,sttype));

  ctx_local = (EPS_KRYLOVSCHUR*)ctx->eps->data;
  ctx_local->npart = ctx->npart;
  ctx_local->global = PETSC_FALSE;
  ctx_local->eps = eps;
  ctx_local->subc = ctx->subc;
  ctx_local->commrank = ctx->commrank;
  *childeps = ctx->eps;
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSKrylovSchurGetKSP_KrylovSchur(EPS eps,KSP *ksp)
{
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  ST              st;
  PetscBool       isfilt;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STFILTER,&isfilt));
  PetscCheck(eps->which==EPS_ALL && !isfilt,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Only available in interval computations with spectrum slicing");
  CHKERRQ(EPSKrylovSchurGetChildEPS(eps,&ctx->eps));
  CHKERRQ(EPSGetST(ctx->eps,&st));
  CHKERRQ(STGetOperator(st,NULL));
  CHKERRQ(STGetKSP(st,ksp));
  PetscFunctionReturn(0);
}

/*@
   EPSKrylovSchurGetKSP - Retrieve the linear solver object associated with the
   internal EPS object in case of doing spectrum slicing for a computational interval.

   Collective on eps

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  ksp - the internal KSP object

   Notes:
   When invoked to compute all eigenvalues in an interval with spectrum
   slicing, EPSKRYLOVSCHUR creates another EPS object internally that is
   used to compute eigenvalues by chunks near selected shifts. This function
   allows access to the KSP object associated to this internal EPS object.

   This function is only available for spectrum slicing runs. In case of
   having more than one partition, the returned KSP will be different
   in MPI processes belonging to different partitions. Hence, if required,
   EPSKrylovSchurSetPartitions() must be called BEFORE this function.

   Level: advanced

.seealso: EPSSetInterval(), EPSKrylovSchurSetPartitions()
@*/
PetscErrorCode EPSKrylovSchurGetKSP(EPS eps,KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  CHKERRQ(PetscUseMethod(eps,"EPSKrylovSchurGetKSP_C",(EPS,KSP*),(eps,ksp)));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_KrylovSchur(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscBool       flg,lock,b,f1,f2,f3,isfilt;
  PetscReal       keep;
  PetscInt        i,j,k;
  KSP             ksp;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"EPS Krylov-Schur Options"));

    CHKERRQ(PetscOptionsReal("-eps_krylovschur_restart","Proportion of vectors kept after restart","EPSKrylovSchurSetRestart",0.5,&keep,&flg));
    if (flg) CHKERRQ(EPSKrylovSchurSetRestart(eps,keep));

    CHKERRQ(PetscOptionsBool("-eps_krylovschur_locking","Choose between locking and non-locking variants","EPSKrylovSchurSetLocking",PETSC_TRUE,&lock,&flg));
    if (flg) CHKERRQ(EPSKrylovSchurSetLocking(eps,lock));

    i = ctx->npart;
    CHKERRQ(PetscOptionsInt("-eps_krylovschur_partitions","Number of partitions of the communicator for spectrum slicing","EPSKrylovSchurSetPartitions",ctx->npart,&i,&flg));
    if (flg) CHKERRQ(EPSKrylovSchurSetPartitions(eps,i));

    b = ctx->detect;
    CHKERRQ(PetscOptionsBool("-eps_krylovschur_detect_zeros","Check zeros during factorizations at subinterval boundaries","EPSKrylovSchurSetDetectZeros",ctx->detect,&b,&flg));
    if (flg) CHKERRQ(EPSKrylovSchurSetDetectZeros(eps,b));

    i = 1;
    j = k = PETSC_DECIDE;
    CHKERRQ(PetscOptionsInt("-eps_krylovschur_nev","Number of eigenvalues to compute in each subsolve (only for spectrum slicing)","EPSKrylovSchurSetDimensions",40,&i,&f1));
    CHKERRQ(PetscOptionsInt("-eps_krylovschur_ncv","Number of basis vectors in each subsolve (only for spectrum slicing)","EPSKrylovSchurSetDimensions",80,&j,&f2));
    CHKERRQ(PetscOptionsInt("-eps_krylovschur_mpd","Maximum dimension of projected problem in each subsolve (only for spectrum slicing)","EPSKrylovSchurSetDimensions",80,&k,&f3));
    if (f1 || f2 || f3) CHKERRQ(EPSKrylovSchurSetDimensions(eps,i,j,k));

  CHKERRQ(PetscOptionsTail());

  /* set options of child KSP in spectrum slicing */
  if (eps->which==EPS_ALL) {
    if (!eps->st) CHKERRQ(EPSGetST(eps,&eps->st));
    CHKERRQ(EPSSetDefaultST(eps));
    CHKERRQ(STSetFromOptions(eps->st));  /* need to advance this to check ST type */
    CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STFILTER,&isfilt));
    if (!isfilt) {
      CHKERRQ(EPSKrylovSchurGetKSP_KrylovSchur(eps,&ksp));
      CHKERRQ(KSPSetFromOptions(ksp));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_KrylovSchur(EPS eps,PetscViewer viewer)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscBool       isascii,isfilt;
  KSP             ksp;
  PetscViewer     sviewer;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %d%% of basis vectors kept after restart\n",(int)(100*ctx->keep)));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using the %slocking variant\n",ctx->lock?"":"non-"));
    if (eps->which==EPS_ALL) {
      CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STFILTER,&isfilt));
      if (isfilt) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using filtering to extract all eigenvalues in an interval\n"));
      else {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  doing spectrum slicing with nev=%" PetscInt_FMT ", ncv=%" PetscInt_FMT ", mpd=%" PetscInt_FMT "\n",ctx->nev,ctx->ncv,ctx->mpd));
        if (ctx->npart>1) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"  multi-communicator spectrum slicing with %" PetscInt_FMT " partitions\n",ctx->npart));
          if (ctx->detect) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  detecting zeros when factorizing at subinterval boundaries\n"));
        }
        /* view child KSP */
        CHKERRQ(EPSKrylovSchurGetKSP_KrylovSchur(eps,&ksp));
        CHKERRQ(PetscViewerASCIIPushTab(viewer));
        if (ctx->npart>1 && ctx->subc) {
          CHKERRQ(PetscViewerGetSubViewer(viewer,ctx->subc->child,&sviewer));
          if (!ctx->subc->color) CHKERRQ(KSPView(ksp,sviewer));
          CHKERRQ(PetscViewerFlush(sviewer));
          CHKERRQ(PetscViewerRestoreSubViewer(viewer,ctx->subc->child,&sviewer));
          CHKERRQ(PetscViewerFlush(viewer));
          /* extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer() */
          CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
        } else CHKERRQ(KSPView(ksp,viewer));
        CHKERRQ(PetscViewerASCIIPopTab(viewer));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_KrylovSchur(EPS eps)
{
  PetscBool      isfilt;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STFILTER,&isfilt));
  if (eps->which==EPS_ALL && !isfilt) CHKERRQ(EPSDestroy_KrylovSchur_Slice(eps));
  CHKERRQ(PetscFree(eps->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetRestart_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetRestart_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetLocking_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetLocking_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetPartitions_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetPartitions_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetDetectZeros_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetDetectZeros_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetDimensions_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetDimensions_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetSubintervals_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetSubintervals_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetInertias_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetSubcommInfo_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetSubcommPairs_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetSubcommMats_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurUpdateSubcommMats_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetKSP_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_KrylovSchur(EPS eps)
{
  PetscBool      isfilt;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STFILTER,&isfilt));
  if (eps->which==EPS_ALL && !isfilt) CHKERRQ(EPSReset_KrylovSchur_Slice(eps));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetDefaultST_KrylovSchur(EPS eps)
{
  PetscFunctionBegin;
  if (eps->which==EPS_ALL) {
    if (!((PetscObject)eps->st)->type_name) CHKERRQ(STSetType(eps->st,STSINVERT));
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_KrylovSchur(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&ctx));
  eps->data   = (void*)ctx;
  ctx->lock   = PETSC_TRUE;
  ctx->nev    = 1;
  ctx->ncv    = PETSC_DEFAULT;
  ctx->mpd    = PETSC_DEFAULT;
  ctx->npart  = 1;
  ctx->detect = PETSC_FALSE;
  ctx->global = PETSC_TRUE;

  eps->useds = PETSC_TRUE;

  /* solve and computevectors determined at setup */
  eps->ops->setup          = EPSSetUp_KrylovSchur;
  eps->ops->setupsort      = EPSSetUpSort_KrylovSchur;
  eps->ops->setfromoptions = EPSSetFromOptions_KrylovSchur;
  eps->ops->destroy        = EPSDestroy_KrylovSchur;
  eps->ops->reset          = EPSReset_KrylovSchur;
  eps->ops->view           = EPSView_KrylovSchur;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->setdefaultst   = EPSSetDefaultST_KrylovSchur;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetRestart_C",EPSKrylovSchurSetRestart_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetRestart_C",EPSKrylovSchurGetRestart_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetLocking_C",EPSKrylovSchurSetLocking_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetLocking_C",EPSKrylovSchurGetLocking_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetPartitions_C",EPSKrylovSchurSetPartitions_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetPartitions_C",EPSKrylovSchurGetPartitions_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetDetectZeros_C",EPSKrylovSchurSetDetectZeros_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetDetectZeros_C",EPSKrylovSchurGetDetectZeros_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetDimensions_C",EPSKrylovSchurSetDimensions_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetDimensions_C",EPSKrylovSchurGetDimensions_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetSubintervals_C",EPSKrylovSchurSetSubintervals_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetSubintervals_C",EPSKrylovSchurGetSubintervals_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetInertias_C",EPSKrylovSchurGetInertias_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetSubcommInfo_C",EPSKrylovSchurGetSubcommInfo_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetSubcommPairs_C",EPSKrylovSchurGetSubcommPairs_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetSubcommMats_C",EPSKrylovSchurGetSubcommMats_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurUpdateSubcommMats_C",EPSKrylovSchurUpdateSubcommMats_KrylovSchur));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetKSP_C",EPSKrylovSchurGetKSP_KrylovSchur));
  PetscFunctionReturn(0);
}
