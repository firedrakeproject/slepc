/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "subspace"

   Method: Subspace Iteration

   Algorithm:

       Subspace iteration with Rayleigh-Ritz projection and locking,
       based on the SRRIT implementation.

   References:

       [1] "Subspace Iteration in SLEPc", SLEPc Technical Report STR-3,
           available at https://slepc.upv.es.
*/

#include <slepc/private/epsimpl.h>

typedef struct {
  PetscBool estimatedrange;     /* the filter range was not set by the user */
} EPS_SUBSPACE;

static PetscErrorCode EPSSetUp_Subspace_Filter(EPS eps)
{
  EPS_SUBSPACE   *ctx = (EPS_SUBSPACE*)eps->data;
  PetscBool      estimaterange=PETSC_TRUE;
  PetscReal      rleft,rright;
  Mat            A;

  PetscFunctionBegin;
  EPSCheckHermitianCondition(eps,PETSC_TRUE," with polynomial filter");
  EPSCheckStandardCondition(eps,PETSC_TRUE," with polynomial filter");
  PetscCheck(eps->intb<PETSC_MAX_REAL || eps->inta>PETSC_MIN_REAL,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"The defined computational interval should have at least one of their sides bounded");
  EPSCheckUnsupportedCondition(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_EXTRACTION,PETSC_TRUE," with polynomial filter");
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
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUp_Subspace(EPS eps)
{
  PetscBool isfilt;

  PetscFunctionBegin;
  EPSCheckDefinite(eps);
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) CHKERRQ(EPSSetWhichEigenpairs_Default(eps));
  if (eps->which==EPS_ALL) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STFILTER,&isfilt));
    PetscCheck(isfilt,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Spectrum slicing not supported in this solver");
    CHKERRQ(EPSSetUp_Subspace_Filter(eps));
  } else {
    PetscCheck(eps->which==EPS_LARGEST_MAGNITUDE || eps->which==EPS_TARGET_MAGNITUDE,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver supports only largest magnitude or target magnitude eigenvalues");
    CHKERRQ(EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd));
  }
  EPSCheckUnsupported(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_EXTRACTION | EPS_FEATURE_TWOSIDED);
  PetscCheck(eps->converged==EPSConvergedRelative,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver only supports relative convergence test");

  CHKERRQ(EPSAllocateSolution(eps,0));
  CHKERRQ(EPS_SetInnerProduct(eps));
  if (eps->ishermitian) CHKERRQ(DSSetType(eps->ds,DSHEP));
  else CHKERRQ(DSSetType(eps->ds,DSNHEP));
  CHKERRQ(DSAllocate(eps->ds,eps->ncv));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUpSort_Subspace(EPS eps)
{
  SlepcSC sc;

  PetscFunctionBegin;
  CHKERRQ(EPSSetUpSort_Default(eps));
  if (eps->which==EPS_ALL) {
    CHKERRQ(DSGetSlepcSC(eps->ds,&sc));
    sc->rg            = NULL;
    sc->comparison    = SlepcCompareLargestReal;
    sc->comparisonctx = NULL;
    sc->map           = NULL;
    sc->mapobj        = NULL;
  }
  PetscFunctionReturn(0);
}

/*
   EPSSubspaceFindGroup - Find a group of nearly equimodular eigenvalues, provided
   in arrays wr and wi, according to the tolerance grptol. Also the 2-norms
   of the residuals must be passed in (rsd). Arrays are processed from index
   l to index m only. The output information is:

   ngrp - number of entries of the group
   ctr  - (w(l)+w(l+ngrp-1))/2
   ae   - average of wr(l),...,wr(l+ngrp-1)
   arsd - average of rsd(l),...,rsd(l+ngrp-1)
*/
static PetscErrorCode EPSSubspaceFindGroup(PetscInt l,PetscInt m,PetscScalar *wr,PetscScalar *wi,PetscReal *rsd,PetscReal grptol,PetscInt *ngrp,PetscReal *ctr,PetscReal *ae,PetscReal *arsd)
{
  PetscInt  i;
  PetscReal rmod,rmod1;

  PetscFunctionBegin;
  *ngrp = 0;
  *ctr = 0;
  rmod = SlepcAbsEigenvalue(wr[l],wi[l]);

  for (i=l;i<m;) {
    rmod1 = SlepcAbsEigenvalue(wr[i],wi[i]);
    if (PetscAbsReal(rmod-rmod1) > grptol*(rmod+rmod1)) break;
    *ctr = (rmod+rmod1)/2.0;
    if (wi[i] == 0.0) {
      (*ngrp)++;
      i++;
    } else {
      (*ngrp)+=2;
      i+=2;
    }
  }

  *ae = 0;
  *arsd = 0;
  if (*ngrp) {
    for (i=l;i<l+*ngrp;i++) {
      (*ae) += PetscRealPart(wr[i]);
      (*arsd) += rsd[i]*rsd[i];
    }
    *ae = *ae / *ngrp;
    *arsd = PetscSqrtReal(*arsd / *ngrp);
  }
  PetscFunctionReturn(0);
}

/*
   EPSSubspaceResidualNorms - Computes the column norms of residual vectors
   OP*V(1:n,l:m) - V*T(1:m,l:m), where, on entry, OP*V has been computed and
   stored in R. On exit, rsd(l) to rsd(m) contain the computed norms.
*/
static PetscErrorCode EPSSubspaceResidualNorms(BV R,BV V,Mat T,PetscInt l,PetscInt m,PetscScalar *eigi,PetscReal *rsd)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(BVMult(R,-1.0,1.0,V,T));
  for (i=l;i<m;i++) CHKERRQ(BVNormColumnBegin(R,i,NORM_2,rsd+i));
  for (i=l;i<m;i++) CHKERRQ(BVNormColumnEnd(R,i,NORM_2,rsd+i));
#if !defined(PETSC_USE_COMPLEX)
  for (i=l;i<m-1;i++) {
    if (eigi[i]!=0.0) {
      rsd[i]   = SlepcAbs(rsd[i],rsd[i+1]);
      rsd[i+1] = rsd[i];
      i++;
    }
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_Subspace(EPS eps)
{
  Mat            H,Q,S,T,B;
  BV             AV,R;
  PetscBool      indef;
  PetscInt       i,k,ld,ngrp,nogrp,*itrsd,*itrsdold;
  PetscInt       nxtsrr,idsrr,idort,nxtort,nv,ncv = eps->ncv,its,ninside;
  PetscReal      arsd,oarsd,ctr,octr,ae,oae,*rsd,*orsd,tcond=1.0,gamma;
  PetscScalar    *oeigr,*oeigi;
  /* Parameters */
  PetscInt       init = 5;        /* Number of initial iterations */
  PetscReal      stpfac = 1.5;    /* Max num of iter before next SRR step */
  PetscReal      alpha = 1.0;     /* Used to predict convergence of next residual */
  PetscReal      beta = 1.1;      /* Used to predict convergence of next residual */
  PetscReal      grptol = SLEPC_DEFAULT_TOL;   /* Tolerance for EPSSubspaceFindGroup */
  PetscReal      cnvtol = 1e-6;   /* Convergence criterion for cnv */
  PetscInt       orttol = 2;      /* Number of decimal digits whose loss
                                     can be tolerated in orthogonalization */

  PetscFunctionBegin;
  its = 0;
  CHKERRQ(PetscMalloc6(ncv,&rsd,ncv,&orsd,ncv,&oeigr,ncv,&oeigi,ncv,&itrsd,ncv,&itrsdold));
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(BVDuplicate(eps->V,&AV));
  CHKERRQ(BVDuplicate(eps->V,&R));
  CHKERRQ(STGetOperator(eps->st,&S));

  for (i=0;i<ncv;i++) {
    rsd[i] = 0.0;
    itrsd[i] = -1;
  }

  /* Complete the initial basis with random vectors and orthonormalize them */
  for (k=eps->nini;k<ncv;k++) {
    CHKERRQ(BVSetRandomColumn(eps->V,k));
    CHKERRQ(BVOrthonormalizeColumn(eps->V,k,PETSC_TRUE,NULL,NULL));
  }

  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;
    nv = PetscMin(eps->nconv+eps->mpd,ncv);
    CHKERRQ(DSSetDimensions(eps->ds,nv,eps->nconv,0));

    for (i=eps->nconv;i<nv;i++) {
      oeigr[i] = eps->eigr[i];
      oeigi[i] = eps->eigi[i];
      orsd[i]  = rsd[i];
    }

    /* AV(:,idx) = OP * V(:,idx) */
    CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv,nv));
    CHKERRQ(BVSetActiveColumns(AV,eps->nconv,nv));
    CHKERRQ(BVMatMult(eps->V,S,AV));

    /* T(:,idx) = V' * AV(:,idx) */
    CHKERRQ(BVSetActiveColumns(eps->V,0,nv));
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_A,&H));
    CHKERRQ(BVDot(AV,eps->V,H));
    CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_A,&H));
    CHKERRQ(DSSetState(eps->ds,DS_STATE_RAW));

    /* Solve projected problem */
    CHKERRQ(DSSolve(eps->ds,eps->eigr,eps->eigi));
    CHKERRQ(DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL));
    CHKERRQ(DSSynchronize(eps->ds,eps->eigr,eps->eigi));

    /* Update vectors V(:,idx) = V * U(:,idx) */
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_Q,&Q));
    CHKERRQ(BVSetActiveColumns(AV,0,nv));
    CHKERRQ(BVSetActiveColumns(R,0,nv));
    CHKERRQ(BVMultInPlace(eps->V,Q,eps->nconv,nv));
    CHKERRQ(BVMultInPlace(AV,Q,eps->nconv,nv));
    CHKERRQ(BVCopy(AV,R));
    CHKERRQ(MatDestroy(&Q));

    /* Convergence check */
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_A,&T));
    CHKERRQ(EPSSubspaceResidualNorms(R,eps->V,T,eps->nconv,nv,eps->eigi,rsd));
    CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_A,&T));

    if (eps->which==EPS_ALL && eps->its>1) {   /* adjust eigenvalue count */
      ninside = 0;
      CHKERRQ(STFilterGetThreshold(eps->st,&gamma));
      for (i=eps->nconv;i<nv;i++) {
        if (PetscRealPart(eps->eigr[i]) < gamma) break;
        ninside++;
      }
      eps->nev = eps->nconv+ninside;
    }
    for (i=eps->nconv;i<nv;i++) {
      itrsdold[i] = itrsd[i];
      itrsd[i] = its;
      eps->errest[i] = rsd[i];
    }

    for (;;) {
      /* Find clusters of computed eigenvalues */
      CHKERRQ(EPSSubspaceFindGroup(eps->nconv,nv,eps->eigr,eps->eigi,eps->errest,grptol,&ngrp,&ctr,&ae,&arsd));
      CHKERRQ(EPSSubspaceFindGroup(eps->nconv,nv,oeigr,oeigi,orsd,grptol,&nogrp,&octr,&oae,&oarsd));
      if (ngrp!=nogrp) break;
      if (ngrp==0) break;
      if (PetscAbsReal(ae-oae)>ctr*cnvtol*(itrsd[eps->nconv]-itrsdold[eps->nconv])) break;
      if (arsd>ctr*eps->tol) break;
      eps->nconv = eps->nconv + ngrp;
      if (eps->nconv>=nv) break;
    }

    CHKERRQ(EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv));
    CHKERRQ((*eps->stopping)(eps,eps->its,eps->max_it,eps->nconv,eps->nev,&eps->reason,eps->stoppingctx));
    if (eps->reason != EPS_CONVERGED_ITERATING) break;

    /* Compute nxtsrr (iteration of next projection step) */
    nxtsrr = PetscMin(eps->max_it,PetscMax((PetscInt)PetscFloorReal(stpfac*its),init));

    if (ngrp!=nogrp || ngrp==0 || arsd>=oarsd) {
      idsrr = nxtsrr - its;
    } else {
      idsrr = (PetscInt)PetscFloorReal(alpha+beta*(itrsdold[eps->nconv]-itrsd[eps->nconv])*PetscLogReal(arsd/eps->tol)/PetscLogReal(arsd/oarsd));
      idsrr = PetscMax(1,idsrr);
    }
    nxtsrr = PetscMin(nxtsrr,its+idsrr);

    /* Compute nxtort (iteration of next orthogonalization step) */
    CHKERRQ(DSCond(eps->ds,&tcond));
    idort = PetscMax(1,(PetscInt)PetscFloorReal(orttol/PetscMax(1,PetscLog10Real(tcond))));
    nxtort = PetscMin(its+idort,nxtsrr);
    CHKERRQ(PetscInfo(eps,"Updated iteration counts: nxtort=%" PetscInt_FMT ", nxtsrr=%" PetscInt_FMT "\n",nxtort,nxtsrr));

    /* V(:,idx) = AV(:,idx) */
    CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv,nv));
    CHKERRQ(BVSetActiveColumns(AV,eps->nconv,nv));
    CHKERRQ(BVCopy(AV,eps->V));
    its++;

    /* Orthogonalization loop */
    do {
      CHKERRQ(BVGetMatrix(eps->V,&B,&indef));
      CHKERRQ(BVSetMatrix(eps->V,NULL,PETSC_FALSE));
      while (its<nxtort) {
        /* A(:,idx) = OP*V(:,idx) with normalization */
        CHKERRQ(BVMatMult(eps->V,S,AV));
        CHKERRQ(BVCopy(AV,eps->V));
        CHKERRQ(BVNormalize(eps->V,NULL));
        its++;
      }
      CHKERRQ(BVSetMatrix(eps->V,B,indef));
      /* Orthonormalize vectors */
      CHKERRQ(BVOrthogonalize(eps->V,NULL));
      nxtort = PetscMin(its+idort,nxtsrr);
    } while (its<nxtsrr);
  }

  CHKERRQ(PetscFree6(rsd,orsd,oeigr,oeigi,itrsd,itrsdold));
  CHKERRQ(BVDestroy(&AV));
  CHKERRQ(BVDestroy(&R));
  CHKERRQ(STRestoreOperator(eps->st,&S));
  CHKERRQ(DSTruncate(eps->ds,eps->nconv,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_Subspace(EPS eps)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(eps->data));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_Subspace(EPS eps)
{
  EPS_SUBSPACE *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&ctx));
  eps->data  = (void*)ctx;

  eps->useds = PETSC_TRUE;
  eps->categ = EPS_CATEGORY_OTHER;

  eps->ops->solve          = EPSSolve_Subspace;
  eps->ops->setup          = EPSSetUp_Subspace;
  eps->ops->setupsort      = EPSSetUpSort_Subspace;
  eps->ops->destroy        = EPSDestroy_Subspace;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->computevectors = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
