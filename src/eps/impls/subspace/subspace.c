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
  PetscErrorCode ierr;
  EPS_SUBSPACE   *ctx = (EPS_SUBSPACE*)eps->data;
  PetscBool      estimaterange=PETSC_TRUE;
  PetscReal      rleft,rright;
  Mat            A;

  PetscFunctionBegin;
  EPSCheckHermitianCondition(eps,PETSC_TRUE," with polynomial filter");
  EPSCheckStandardCondition(eps,PETSC_TRUE," with polynomial filter");
  PetscCheck(eps->intb<PETSC_MAX_REAL || eps->inta>PETSC_MIN_REAL,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"The defined computational interval should have at least one of their sides bounded");
  EPSCheckUnsupportedCondition(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_EXTRACTION,PETSC_TRUE," with polynomial filter");
  ierr = STFilterSetInterval(eps->st,eps->inta,eps->intb);CHKERRQ(ierr);
  if (!ctx->estimatedrange) {
    ierr = STFilterGetRange(eps->st,&rleft,&rright);CHKERRQ(ierr);
    estimaterange = (!rleft && !rright)? PETSC_TRUE: PETSC_FALSE;
  }
  if (estimaterange) { /* user did not set a range */
    ierr = STGetMatrix(eps->st,0,&A);CHKERRQ(ierr);
    ierr = MatEstimateSpectralRange_EPS(A,&rleft,&rright);CHKERRQ(ierr);
    ierr = PetscInfo(eps,"Setting eigenvalue range to [%g,%g]\n",(double)rleft,(double)rright);CHKERRQ(ierr);
    ierr = STFilterSetRange(eps->st,rleft,rright);CHKERRQ(ierr);
    ctx->estimatedrange = PETSC_TRUE;
  }
  if (eps->ncv==PETSC_DEFAULT && eps->nev==1) eps->nev = 40;  /* user did not provide nev estimation */
  ierr = EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd);CHKERRQ(ierr);
  PetscCheck(eps->ncv<=eps->nev+eps->mpd,PetscObjectComm((PetscObject)eps),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nev+mpd");
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUp_Subspace(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      isfilt;

  PetscFunctionBegin;
  EPSCheckDefinite(eps);
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) { ierr = EPSSetWhichEigenpairs_Default(eps);CHKERRQ(ierr); }
  if (eps->which==EPS_ALL) {
    ierr = PetscObjectTypeCompare((PetscObject)eps->st,STFILTER,&isfilt);CHKERRQ(ierr);
    PetscCheck(isfilt,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Spectrum slicing not supported in this solver");
    ierr = EPSSetUp_Subspace_Filter(eps);CHKERRQ(ierr);
  } else {
    PetscCheck(eps->which==EPS_LARGEST_MAGNITUDE || eps->which==EPS_TARGET_MAGNITUDE,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver supports only largest magnitude or target magnitude eigenvalues");
    ierr = EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd);CHKERRQ(ierr);
  }
  EPSCheckUnsupported(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_EXTRACTION | EPS_FEATURE_TWOSIDED);
  PetscCheck(eps->converged==EPSConvergedRelative,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver only supports relative convergence test");

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = EPS_SetInnerProduct(eps);CHKERRQ(ierr);
  if (eps->ishermitian) {
    ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
  } else {
    ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
  }
  ierr = DSAllocate(eps->ds,eps->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUpSort_Subspace(EPS eps)
{
  PetscErrorCode ierr;
  SlepcSC        sc;

  PetscFunctionBegin;
  ierr = EPSSetUpSort_Default(eps);CHKERRQ(ierr);
  if (eps->which==EPS_ALL) {
    ierr = DSGetSlepcSC(eps->ds,&sc);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = BVMult(R,-1.0,1.0,V,T);CHKERRQ(ierr);
  for (i=l;i<m;i++) { ierr = BVNormColumnBegin(R,i,NORM_2,rsd+i);CHKERRQ(ierr); }
  for (i=l;i<m;i++) { ierr = BVNormColumnEnd(R,i,NORM_2,rsd+i);CHKERRQ(ierr); }
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
  PetscErrorCode ierr;
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
  ierr = PetscMalloc6(ncv,&rsd,ncv,&orsd,ncv,&oeigr,ncv,&oeigi,ncv,&itrsd,ncv,&itrsdold);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = BVDuplicate(eps->V,&AV);CHKERRQ(ierr);
  ierr = BVDuplicate(eps->V,&R);CHKERRQ(ierr);
  ierr = STGetOperator(eps->st,&S);CHKERRQ(ierr);

  for (i=0;i<ncv;i++) {
    rsd[i] = 0.0;
    itrsd[i] = -1;
  }

  /* Complete the initial basis with random vectors and orthonormalize them */
  for (k=eps->nini;k<ncv;k++) {
    ierr = BVSetRandomColumn(eps->V,k);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(eps->V,k,PETSC_TRUE,NULL,NULL);CHKERRQ(ierr);
  }

  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;
    nv = PetscMin(eps->nconv+eps->mpd,ncv);
    ierr = DSSetDimensions(eps->ds,nv,eps->nconv,0);CHKERRQ(ierr);

    for (i=eps->nconv;i<nv;i++) {
      oeigr[i] = eps->eigr[i];
      oeigi[i] = eps->eigi[i];
      orsd[i]  = rsd[i];
    }

    /* AV(:,idx) = OP * V(:,idx) */
    ierr = BVSetActiveColumns(eps->V,eps->nconv,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(AV,eps->nconv,nv);CHKERRQ(ierr);
    ierr = BVMatMult(eps->V,S,AV);CHKERRQ(ierr);

    /* T(:,idx) = V' * AV(:,idx) */
    ierr = BVSetActiveColumns(eps->V,0,nv);CHKERRQ(ierr);
    ierr = DSGetMat(eps->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = BVDot(AV,eps->V,H);CHKERRQ(ierr);
    ierr = DSRestoreMat(eps->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);

    /* Solve projected problem */
    ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
    ierr = DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSSynchronize(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);

    /* Update vectors V(:,idx) = V * U(:,idx) */
    ierr = DSGetMat(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(AV,0,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(R,0,nv);CHKERRQ(ierr);
    ierr = BVMultInPlace(eps->V,Q,eps->nconv,nv);CHKERRQ(ierr);
    ierr = BVMultInPlace(AV,Q,eps->nconv,nv);CHKERRQ(ierr);
    ierr = BVCopy(AV,R);CHKERRQ(ierr);
    ierr = MatDestroy(&Q);CHKERRQ(ierr);

    /* Convergence check */
    ierr = DSGetMat(eps->ds,DS_MAT_A,&T);CHKERRQ(ierr);
    ierr = EPSSubspaceResidualNorms(R,eps->V,T,eps->nconv,nv,eps->eigi,rsd);CHKERRQ(ierr);
    ierr = DSRestoreMat(eps->ds,DS_MAT_A,&T);CHKERRQ(ierr);

    if (eps->which==EPS_ALL && eps->its>1) {   /* adjust eigenvalue count */
      ninside = 0;
      ierr = STFilterGetThreshold(eps->st,&gamma);CHKERRQ(ierr);
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
      ierr = EPSSubspaceFindGroup(eps->nconv,nv,eps->eigr,eps->eigi,eps->errest,grptol,&ngrp,&ctr,&ae,&arsd);CHKERRQ(ierr);
      ierr = EPSSubspaceFindGroup(eps->nconv,nv,oeigr,oeigi,orsd,grptol,&nogrp,&octr,&oae,&oarsd);CHKERRQ(ierr);

      if (ngrp!=nogrp) break;
      if (ngrp==0) break;
      if (PetscAbsReal(ae-oae)>ctr*cnvtol*(itrsd[eps->nconv]-itrsdold[eps->nconv])) break;
      if (arsd>ctr*eps->tol) break;
      eps->nconv = eps->nconv + ngrp;
      if (eps->nconv>=nv) break;
    }

    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
    ierr = (*eps->stopping)(eps,eps->its,eps->max_it,eps->nconv,eps->nev,&eps->reason,eps->stoppingctx);CHKERRQ(ierr);
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
    ierr = DSCond(eps->ds,&tcond);CHKERRQ(ierr);
    idort = PetscMax(1,(PetscInt)PetscFloorReal(orttol/PetscMax(1,PetscLog10Real(tcond))));
    nxtort = PetscMin(its+idort,nxtsrr);
    ierr = PetscInfo(eps,"Updated iteration counts: nxtort=%" PetscInt_FMT ", nxtsrr=%" PetscInt_FMT "\n",nxtort,nxtsrr);CHKERRQ(ierr);

    /* V(:,idx) = AV(:,idx) */
    ierr = BVSetActiveColumns(eps->V,eps->nconv,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(AV,eps->nconv,nv);CHKERRQ(ierr);
    ierr = BVCopy(AV,eps->V);CHKERRQ(ierr);
    its++;

    /* Orthogonalization loop */
    do {
      ierr = BVGetMatrix(eps->V,&B,&indef);CHKERRQ(ierr);
      ierr = BVSetMatrix(eps->V,NULL,PETSC_FALSE);CHKERRQ(ierr);
      while (its<nxtort) {
        /* A(:,idx) = OP*V(:,idx) with normalization */
        ierr = BVMatMult(eps->V,S,AV);CHKERRQ(ierr);
        ierr = BVCopy(AV,eps->V);CHKERRQ(ierr);
        ierr = BVNormalize(eps->V,NULL);CHKERRQ(ierr);
        its++;
      }
      ierr = BVSetMatrix(eps->V,B,indef);CHKERRQ(ierr);
      /* Orthonormalize vectors */
      ierr = BVOrthogonalize(eps->V,NULL);CHKERRQ(ierr);
      nxtort = PetscMin(its+idort,nxtsrr);
    } while (its<nxtsrr);
  }

  ierr = PetscFree6(rsd,orsd,oeigr,oeigi,itrsd,itrsdold);CHKERRQ(ierr);
  ierr = BVDestroy(&AV);CHKERRQ(ierr);
  ierr = BVDestroy(&R);CHKERRQ(ierr);
  ierr = STRestoreOperator(eps->st,&S);CHKERRQ(ierr);
  ierr = DSTruncate(eps->ds,eps->nconv,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_Subspace(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_Subspace(EPS eps)
{
  EPS_SUBSPACE   *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
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

