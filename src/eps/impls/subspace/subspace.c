/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

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

PetscErrorCode EPSSetUp_Subspace(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  EPSCheckDefinite(eps);
  ierr = EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd);CHKERRQ(ierr);
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) { ierr = EPSSetWhichEigenpairs_Default(eps);CHKERRQ(ierr); }
  if (eps->which!=EPS_LARGEST_MAGNITUDE && eps->which!=EPS_TARGET_MAGNITUDE) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver supports only largest magnitude or target magnitude eigenvalues");
  EPSCheckUnsupported(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_EXTRACTION | EPS_FEATURE_TWOSIDED);
  if (eps->converged != EPSConvergedRelative) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver only supports relative convergence test");

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
    if (wi[i] != 0.0) {
      (*ngrp)+=2;
      i+=2;
    } else {
      (*ngrp)++;
      i++;
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
    *arsd = PetscSqrtScalar(*arsd / *ngrp);
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
  PetscInt       nxtsrr,idsrr,idort,nxtort,nv,ncv = eps->ncv,its;
  PetscReal      arsd,oarsd,ctr,octr,ae,oae,*rsd,tcond=1.0;
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
  ierr = PetscMalloc3(ncv,&rsd,ncv,&itrsd,ncv,&itrsdold);CHKERRQ(ierr);
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
    ierr = DSSetDimensions(eps->ds,nv,0,eps->nconv,0);CHKERRQ(ierr);

    /* Find group in previously computed eigenvalues */
    ierr = EPSSubspaceFindGroup(eps->nconv,nv,eps->eigr,eps->eigi,rsd,grptol,&nogrp,&octr,&oae,&oarsd);CHKERRQ(ierr);

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

    for (i=eps->nconv;i<nv;i++) {
      itrsdold[i] = itrsd[i];
      itrsd[i] = its;
      eps->errest[i] = rsd[i];
    }

    for (;;) {
      /* Find group in currently computed eigenvalues */
      ierr = EPSSubspaceFindGroup(eps->nconv,nv,eps->eigr,eps->eigi,eps->errest,grptol,&ngrp,&ctr,&ae,&arsd);CHKERRQ(ierr);
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

  ierr = PetscFree3(rsd,itrsd,itrsdold);CHKERRQ(ierr);
  ierr = BVDestroy(&AV);CHKERRQ(ierr);
  ierr = BVDestroy(&R);CHKERRQ(ierr);
  ierr = STRestoreOperator(eps->st,&S);CHKERRQ(ierr);
  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(eps->ds,eps->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
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
  PetscFunctionBegin;
  eps->useds = PETSC_TRUE;
  eps->categ = EPS_CATEGORY_OTHER;

  eps->ops->solve          = EPSSolve_Subspace;
  eps->ops->setup          = EPSSetUp_Subspace;
  eps->ops->setupsort      = EPSSetUpSort_Default;
  eps->ops->destroy        = EPSDestroy_Subspace;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->computevectors = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}

