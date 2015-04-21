/*

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur

   Algorithm:

       Single-vector Krylov-Schur method for non-symmetric problems,
       including harmonic extraction.

   References:

       [1] "Krylov-Schur Methods in SLEPc", SLEPc Technical Report STR-7,
           available at http://slepc.upv.es.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Anal. App. 23(3):601-614, 2001.

       [3] "Practical Implementation of Harmonic Krylov-Schur", SLEPc Technical
            Report STR-9, available at http://slepc.upv.es.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>                /*I "slepceps.h" I*/
#include "krylovschur.h"

#undef __FUNCT__
#define __FUNCT__ "EPSGetArbitraryValues"
PetscErrorCode EPSGetArbitraryValues(EPS eps,PetscScalar *rr,PetscScalar *ri)
{
  PetscErrorCode ierr;
  PetscInt       i,newi,ld,n,l;
  Vec            xr=eps->work[0],xi=eps->work[1];
  PetscScalar    re,im,*Zr,*Zi,*X;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = DSGetDimensions(eps->ds,&n,NULL,&l,NULL,NULL);CHKERRQ(ierr);
  for (i=l;i<n;i++) {
    re = eps->eigr[i];
    im = eps->eigi[i];
    ierr = STBackTransform(eps->st,1,&re,&im);CHKERRQ(ierr);
    newi = i;
    ierr = DSVectors(eps->ds,DS_MAT_X,&newi,NULL);CHKERRQ(ierr);
    ierr = DSGetArray(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
    Zr = X+i*ld;
    if (newi==i+1) Zi = X+newi*ld;
    else Zi = NULL;
    ierr = EPSComputeRitzVector(eps,Zr,Zi,eps->V,xr,xi);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
    ierr = (*eps->arbitrary)(re,im,xr,xi,rr+i,ri+i,eps->arbitraryctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_KrylovSchur"
PetscErrorCode EPSSetUp_KrylovSchur(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  enum { EPS_KS_DEFAULT,EPS_KS_SYMM,EPS_KS_SLICE,EPS_KS_INDEF } variant;

  PetscFunctionBegin;
  /* spectrum slicing requires special treatment of default values */
  if (eps->which==EPS_ALL) {
    ierr = EPSSetUp_KrylovSchur_Slice(eps);CHKERRQ(ierr);
  } else {
    ierr = EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd);CHKERRQ(ierr);
    if (eps->ncv>eps->nev+eps->mpd) SETERRQ(PetscObjectComm((PetscObject)eps),1,"The value of ncv must not be larger than nev+mpd");
    if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
    if (!eps->which) { ierr = EPSSetWhichEigenpairs_Default(eps);CHKERRQ(ierr); }
  }

  if (eps->isgeneralized && eps->ishermitian && !eps->ispositive && eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not implemented for indefinite problems");
  if (eps->ishermitian && eps->ispositive && (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY)) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");

  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ && eps->extraction!=EPS_HARMONIC)
    SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");

  if (!ctx->keep) ctx->keep = 0.5;

  ierr = EPSAllocateSolution(eps,1);CHKERRQ(ierr);
  ierr = EPS_SetInnerProduct(eps);CHKERRQ(ierr);
  if (eps->arbitrary) {
    ierr = EPSSetWorkVecs(eps,2);CHKERRQ(ierr);
  } else if (eps->ishermitian && !eps->ispositive){
    ierr = EPSSetWorkVecs(eps,1);CHKERRQ(ierr);
  }

  /* dispatch solve method */
  if (eps->ishermitian) {
    if (eps->which==EPS_ALL) {
      if (eps->isgeneralized && !eps->ispositive) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Spectrum slicing not implemented for indefinite problems");
      else variant = EPS_KS_SLICE;
    } else if (eps->isgeneralized && !eps->ispositive) {
      variant = EPS_KS_INDEF;
    } else {
      switch (eps->extraction) {
        case EPS_RITZ:     variant = EPS_KS_SYMM; break;
        case EPS_HARMONIC: variant = EPS_KS_DEFAULT; break;
        default: SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");
      }
    }
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
      ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
      ierr = DSAllocate(eps->ds,eps->ncv+1);CHKERRQ(ierr);
      break;
    case EPS_KS_SYMM:
      eps->ops->solve = EPSSolve_KrylovSchur_Symm;
      eps->ops->computevectors = EPSComputeVectors_Hermitian;
      ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
      ierr = DSSetCompact(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
      ierr = DSSetExtraRow(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
      ierr = DSAllocate(eps->ds,eps->ncv+1);CHKERRQ(ierr);
      break;
    case EPS_KS_SLICE:
      eps->ops->solve = EPSSolve_KrylovSchur_Slice;
      eps->ops->computevectors = NULL;
      ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
      ierr = DSSetCompact(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
      ierr = DSAllocate(eps->ds,ctx->ncv+1);CHKERRQ(ierr);
      break;
    case EPS_KS_INDEF:
      eps->ops->solve = EPSSolve_KrylovSchur_Indefinite;
      eps->ops->computevectors = EPSComputeVectors_Indefinite;
      ierr = DSSetType(eps->ds,DSGHIEP);CHKERRQ(ierr);
      ierr = DSSetCompact(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
      ierr = DSAllocate(eps->ds,eps->ncv+1);CHKERRQ(ierr);
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)eps),1,"Unexpected error");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_KrylovSchur_Default"
PetscErrorCode EPSSolve_KrylovSchur_Default(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i,j,*pj,k,l,nv,ld;
  Mat             U;
  PetscScalar     *S,*Q,*g;
  PetscReal       beta,gamma=1.0;
  PetscBool       breakdown,harmonic;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  harmonic = (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC)?PETSC_TRUE:PETSC_FALSE;
  if (harmonic) { ierr = PetscMalloc1(ld,&g);CHKERRQ(ierr); }
  if (eps->arbitrary) pj = &j;
  else pj = NULL;

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,NULL);CHKERRQ(ierr);
  l = 0;

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = DSGetArray(eps->ds,DS_MAT_A,&S);CHKERRQ(ierr);
    ierr = EPSBasicArnoldi(eps,PETSC_FALSE,S,ld,eps->nconv+l,&nv,&beta,&breakdown);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_A,&S);CHKERRQ(ierr);
    ierr = DSSetDimensions(eps->ds,nv,0,eps->nconv,eps->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(eps->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(eps->V,eps->nconv,nv);CHKERRQ(ierr);

    /* Compute translation of Krylov decomposition if harmonic extraction used */
    if (harmonic) {
      ierr = DSTranslateHarmonic(eps->ds,eps->target,beta,PETSC_FALSE,g,&gamma);CHKERRQ(ierr);
    }

    /* Solve projected problem */
    ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
    if (eps->arbitrary) {
      ierr = EPSGetArbitraryValues(eps,eps->rr,eps->ri);CHKERRQ(ierr);
      j=1;
    }
    ierr = DSSort(eps->ds,eps->eigr,eps->eigi,eps->rr,eps->ri,pj);CHKERRQ(ierr);

    /* Check convergence */
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,nv-eps->nconv,beta,gamma,&k);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
#if !defined(PETSC_USE_COMPLEX)
      ierr = DSGetArray(eps->ds,DS_MAT_A,&S);CHKERRQ(ierr);
      if (S[k+l+(k+l-1)*ld] != 0.0) {
        if (k+l<nv-1) l = l+1;
        else l = l-1;
      }
      ierr = DSRestoreArray(eps->ds,DS_MAT_A,&S);CHKERRQ(ierr);
#endif
    }

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Arnoldi factorization */
        ierr = PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%g)\n",eps->its,(double)beta);CHKERRQ(ierr);
        if (k<eps->nev) {
          ierr = EPSGetStartVector(eps,k,&breakdown);CHKERRQ(ierr);
          if (breakdown) {
            eps->reason = EPS_DIVERGED_BREAKDOWN;
            ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
          }
        }
      } else {
        /* Undo translation of Krylov decomposition */
        if (harmonic) {
          ierr = DSSetDimensions(eps->ds,nv,0,k,l);CHKERRQ(ierr);
          ierr = DSTranslateHarmonic(eps->ds,0.0,beta,PETSC_TRUE,g,&gamma);CHKERRQ(ierr);
          /* gamma u^ = u - U*g~ */
          ierr = BVMultColumn(eps->V,-1.0,1.0,nv,g);CHKERRQ(ierr);
          ierr = BVScaleColumn(eps->V,nv,1.0/gamma);CHKERRQ(ierr);
        }
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSGetArray(eps->ds,DS_MAT_A,&S);CHKERRQ(ierr);
        ierr = DSGetArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        for (i=k;i<k+l;i++) {
          S[k+l+i*ld] = Q[nv-1+i*ld]*beta*gamma;
        }
        ierr = DSRestoreArray(eps->ds,DS_MAT_A,&S);CHKERRQ(ierr);
        ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = DSGetMat(eps->ds,DS_MAT_Q,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(eps->V,U,eps->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);

    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = BVCopyColumn(eps->V,nv,k+l);CHKERRQ(ierr);
    }
    eps->nconv = k;
    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
  }

  if (harmonic) { ierr = PetscFree(g);CHKERRQ(ierr); }
  /* truncate Schur decomposition and change the state to raw so that
     PSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(eps->ds,eps->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurSetRestart_KrylovSchur"
static PetscErrorCode EPSKrylovSchurSetRestart_KrylovSchur(EPS eps,PetscReal keep)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) ctx->keep = 0.5;
  else {
    if (keep<0.1 || keep>0.9) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument must be in the range [0.1,0.9]");
    ctx->keep = keep;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurSetRestart"
/*@
   EPSKrylovSchurSetRestart - Sets the restart parameter for the Krylov-Schur
   method, in particular the proportion of basis vectors that must be kept
   after restart.

   Logically Collective on EPS

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,keep,2);
  ierr = PetscTryMethod(eps,"EPSKrylovSchurSetRestart_C",(EPS,PetscReal),(eps,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurGetRestart_KrylovSchur"
static PetscErrorCode EPSKrylovSchurGetRestart_KrylovSchur(EPS eps,PetscReal *keep)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  *keep = ctx->keep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurGetRestart"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(keep,2);
  ierr = PetscTryMethod(eps,"EPSKrylovSchurGetRestart_C",(EPS,PetscReal*),(eps,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurSetDimensions_KrylovSchur"
static PetscErrorCode EPSKrylovSchurSetDimensions_KrylovSchur(EPS eps,PetscInt nev,PetscInt ncv,PetscInt mpd)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  if (nev<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of nev. Must be > 0");
  ctx->nev = nev;
  if (ncv == PETSC_DECIDE || ncv == PETSC_DEFAULT) {
    ctx->ncv = 0;
  } else {
    if (ncv<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
    ctx->ncv = ncv;
  }
  if (mpd == PETSC_DECIDE || mpd == PETSC_DEFAULT) {
    ctx->mpd = 0;
  } else {
    if (mpd<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of mpd. Must be > 0");
    ctx->mpd = mpd;
  }
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurSetDimensions"
/*@
   EPSKrylovSchurSetDimensions - Sets the dimensions used for each subsolve
   step in case of doing spectrum slicing for a computational interval.
   The meaning of the parameters is the same as in EPSSetDimensions().

   Logically Collective on EPS

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,nev,2);
  PetscValidLogicalCollectiveInt(eps,ncv,3);
  PetscValidLogicalCollectiveInt(eps,mpd,4);
  ierr = PetscTryMethod(eps,"EPSKrylovSchurSetDimensions_C",(EPS,PetscInt,PetscInt,PetscInt),(eps,nev,ncv,mpd));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurGetDimensions_KrylovSchur"
static PetscErrorCode EPSKrylovSchurGetDimensions_KrylovSchur(EPS eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  if (nev) *nev = ctx->nev;
  if (ncv) *ncv = ctx->ncv;
  if (mpd) *mpd = ctx->mpd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurGetDimensions"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscTryMethod(eps,"EPSKrylovSchurGetDimensions_C",(EPS,PetscInt*,PetscInt*,PetscInt*),(eps,nev,ncv,mpd));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define SWAP(a,b,t) {t=a;a=b;b=t;}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurGetInertias_KrylovSchur"
static PetscErrorCode EPSKrylovSchurGetInertias_KrylovSchur(EPS eps,PetscInt *n,PetscReal **shifts,PetscInt **inertias)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i=0,j,tmpi;
  PetscReal       v,tmpr;
  EPS_shift       s;

  PetscFunctionBegin;
  if (!eps->state) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Must call EPSSetUp() first");
  if (!ctx->sr) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Only available in interval computations, see EPSSetInterval()");
  if (!ctx->sr->s0) {  /* EPSSolve not called yet */
    *n = 2;
  } else {
    *n = 1;
    s = ctx->sr->s0;
    while (s) {
      (*n)++;
      s = s->neighb[1];
    }
  }
  ierr = PetscMalloc1(*n,shifts);CHKERRQ(ierr);
  ierr = PetscMalloc1(*n,inertias);CHKERRQ(ierr);
  if (!ctx->sr->s0) {  /* EPSSolve not called yet */
    (*shifts)[0]   = ctx->sr->int0;
    (*shifts)[1]   = ctx->sr->int1;
    (*inertias)[0] = ctx->sr->inertia0;
    (*inertias)[1] = ctx->sr->inertia1;
  } else {
    s = ctx->sr->s0;
    while (s) {
      (*shifts)[i]     = s->value;
      (*inertias)[i++] = s->inertia;
      s = s->neighb[1];
    }
    (*shifts)[i]   = ctx->sr->int1;
    (*inertias)[i] = ctx->sr->inertia1;
  }
  /* sort result */
  for (i=0;i<*n;i++) {
    v = (*shifts)[i];
    for (j=i+1;j<*n;j++) {
      if (v > (*shifts)[j]) {
        SWAP((*shifts)[i],(*shifts)[j],tmpr);
        SWAP((*inertias)[i],(*inertias)[j],tmpi);
        v = (*shifts)[i];
      }
    }
  }
  /* remove possible duplicate in last position */
  if ((*shifts)[(*n)-1]==(*shifts)[(*n)-2]) (*n)--;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurGetInertias"
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
   endpoints is available.

   This function is only available for spectrum slicing runs, see EPSSetInterval().

   The returned arrays should be freed by the user.

   Level: advanced
@*/
PetscErrorCode EPSKrylovSchurGetInertias(EPS eps,PetscInt *n,PetscReal **shifts,PetscInt **inertias)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(n,2);
  ierr = PetscTryMethod(eps,"EPSKrylovSchurGetInertias_C",(EPS,PetscInt*,PetscReal**,PetscInt**),(eps,n,shifts,inertias));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetFromOptions_KrylovSchur"
PetscErrorCode EPSSetFromOptions_KrylovSchur(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscReal      keep;
  PetscInt       i,j,k;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EPS Krylov-Schur Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps_krylovschur_restart","Proportion of vectors kept after restart","EPSKrylovSchurSetRestart",0.5,&keep,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = EPSKrylovSchurSetRestart(eps,keep);CHKERRQ(ierr);
  }
  i = 1;
  j = k = PETSC_DECIDE;
  ierr = PetscOptionsInt("-eps_krylovschur_nev","Number of eigenvalues to compute in each subsolve (only for spectrum slicing)","EPSKrylovSchurSetDimensions",40,&i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_krylovschur_ncv","Number of basis vectors in each subsolve (only for spectrum slicing)","EPSKrylovSchurSetDimensions",80,&j,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_krylovschur_mpd","Maximum dimension of projected problem in each subsolve (only for spectrum slicing)","EPSKrylovSchurSetDimensions",80,&k,NULL);CHKERRQ(ierr);
  ierr = EPSKrylovSchurSetDimensions(eps,i,j,k);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSView_KrylovSchur"
PetscErrorCode EPSView_KrylovSchur(EPS eps,PetscViewer viewer)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscBool       isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Krylov-Schur: %d%% of basis vectors kept after restart\n",(int)(100*ctx->keep));CHKERRQ(ierr);
    if (eps->which==EPS_ALL) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Krylov-Schur: doing spectrum slicing with nev=%D, ncv=%D, mpd=%D\n",ctx->nev,ctx->ncv,ctx->mpd);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset_KrylovSchur"
PetscErrorCode EPSReset_KrylovSchur(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  EPS_shift       s;

  PetscFunctionBegin;
  /* Reviewing list of shifts to free memory */
  if (ctx->sr) {
    s = ctx->sr->s0;
    if (s) {
      while (s->neighb[1]) {
        s = s->neighb[1];
        ierr = PetscFree(s->neighb[0]);CHKERRQ(ierr);
      }
      ierr = PetscFree(s);CHKERRQ(ierr);
    }
    ierr = PetscFree(ctx->sr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_KrylovSchur"
PetscErrorCode EPSDestroy_KrylovSchur(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetDimensions_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetDimensions_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetInertias_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_KrylovSchur"
PETSC_EXTERN PetscErrorCode EPSCreate_KrylovSchur(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;
  ctx->nev = 1;

  eps->ops->setup          = EPSSetUp_KrylovSchur;
  eps->ops->setfromoptions = EPSSetFromOptions_KrylovSchur;
  eps->ops->destroy        = EPSDestroy_KrylovSchur;
  eps->ops->reset          = EPSReset_KrylovSchur;
  eps->ops->view           = EPSView_KrylovSchur;
  eps->ops->backtransform  = EPSBackTransform_Default;
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetRestart_C",EPSKrylovSchurSetRestart_KrylovSchur);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetRestart_C",EPSKrylovSchurGetRestart_KrylovSchur);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetDimensions_C",EPSKrylovSchurSetDimensions_KrylovSchur);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetDimensions_C",EPSKrylovSchurGetDimensions_KrylovSchur);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetInertias_C",EPSKrylovSchurGetInertias_KrylovSchur);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

