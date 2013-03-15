/*

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur

   Algorithm:

       Single-vector Krylov-Schur method for non-symmetric problems,
       including harmonic extraction.

   References:

       [1] "Krylov-Schur Methods in SLEPc", SLEPc Technical Report STR-7,
           available at http://www.grycap.upv.es/slepc.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Anal. App. 23(3):601-614, 2001.

       [3] "Practical Implementation of Harmonic Krylov-Schur", SLEPc Technical
            Report STR-9, available at http://www.grycap.upv.es/slepc.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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
#include <slepcblaslapack.h>
#include "krylovschur.h"

#undef __FUNCT__
#define __FUNCT__ "EPSGetArbitraryValues"
PetscErrorCode EPSGetArbitraryValues(EPS eps,PetscScalar *rr,PetscScalar *ri)
{
  PetscErrorCode ierr;
  PetscInt       i,newi,ld,n,l;
  Vec            xr=eps->work[1],xi=eps->work[2];
  PetscScalar    re,im,*Zr,*Zi,*X;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = DSGetDimensions(eps->ds,&n,NULL,&l,NULL);CHKERRQ(ierr);
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
    ierr = EPSComputeRitzVector(eps,Zr,Zi,eps->V,n,xr,xi);CHKERRQ(ierr);
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
  PetscBool       issinv;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  enum { EPS_KS_DEFAULT,EPS_KS_SYMM,EPS_KS_SLICE,EPS_KS_INDEF } variant;

  PetscFunctionBegin;
  /* spectrum slicing requires special treatment of default values */
  if (eps->which==EPS_ALL) {
    if (eps->inta==0.0 && eps->intb==0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"Must define a computational interval when using EPS_ALL");
    if (!eps->ishermitian) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Spectrum slicing only available for symmetric/Hermitian eigenproblems");
    if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs cannot be used with spectrum slicing");
    if (!((PetscObject)(eps->st))->type_name) { /* default to shift-and-invert */
      ierr = STSetType(eps->st,STSINVERT);CHKERRQ(ierr);
    }
    ierr = PetscObjectTypeCompareAny((PetscObject)eps->st,&issinv,STSINVERT,STCAYLEY,"");CHKERRQ(ierr);
    if (!issinv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Shift-and-invert or Cayley ST is needed for spectrum slicing");
#if defined(PETSC_USE_REAL_DOUBLE)
    if (eps->tol==PETSC_DEFAULT) eps->tol = 1e-10;  /* use tighter tolerance */
#endif
    if (eps->intb >= PETSC_MAX_REAL) { /* right-open interval */
      if (eps->inta <= PETSC_MIN_REAL) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"The defined computational interval should have at least one of their sides bounded");
      ierr = STSetDefaultShift(eps->st,eps->inta);CHKERRQ(ierr);
    } else {
      ierr = STSetDefaultShift(eps->st,eps->intb);CHKERRQ(ierr);
    }

    if (eps->nev==1) eps->nev = 40;  /* nev not set, use default value */
    if (eps->nev<10) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"nev cannot be less than 10 in spectrum slicing runs");
    eps->ops->backtransform = NULL;
  }

  if (eps->isgeneralized && eps->ishermitian && !eps->ispositive && eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not implemented for indefinite problems");

  /* proceed with the general case */
  if (eps->ncv) { /* ncv set */
    if (eps->ncv<eps->nev) SETERRQ(PetscObjectComm((PetscObject)eps),1,"The value of ncv must be at least nev");
  } else if (eps->mpd) { /* mpd set */
    eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (eps->nev<500) eps->ncv = PetscMin(eps->n,PetscMax(2*eps->nev,eps->nev+15));
    else {
      eps->mpd = 500;
      eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd);
    }
  }
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (eps->ncv>eps->nev+eps->mpd) SETERRQ(PetscObjectComm((PetscObject)eps),1,"The value of ncv must not be larger than nev+mpd");
  if (!eps->max_it) {
    if (eps->which==EPS_ALL) eps->max_it = 100;  /* special case for spectrum slicing */
    else eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  }
  if (!eps->which) { ierr = EPSSetWhichEigenpairs_Default(eps);CHKERRQ(ierr); }
  if (eps->ishermitian && (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY)) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");

  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ && eps->extraction!=EPS_HARMONIC)
    SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");

  if (!ctx->keep) ctx->keep = 0.5;

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  if (eps->arbitrary) {
    ierr = EPSSetWorkVecs(eps,3);CHKERRQ(ierr);
  } else {
    ierr = EPSSetWorkVecs(eps,1);CHKERRQ(ierr);
  }

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Left vectors not supported in this solver");
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
      ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
      break;
    case EPS_KS_SYMM:
      eps->ops->solve = EPSSolve_KrylovSchur_Symm;
      ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
      ierr = DSSetCompact(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
      ierr = DSSetExtraRow(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
      break;
    case EPS_KS_SLICE:
      eps->ops->solve = EPSSolve_KrylovSchur_Slice;
      ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
      ierr = DSSetCompact(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
      break;
    case EPS_KS_INDEF:
      eps->ops->solve = EPSSolve_KrylovSchur_Indefinite;
      ierr = DSSetType(eps->ds,DSGHIEP);CHKERRQ(ierr);
      ierr = DSSetCompact(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)eps),1,"Unexpected error");
  }
  ierr = DSAllocate(eps->ds,eps->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_KrylovSchur_Default"
PetscErrorCode EPSSolve_KrylovSchur_Default(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i,j,*pj,k,l,nv,ld;
  Vec             u=eps->work[0];
  PetscScalar     *S,*Q,*g;
  PetscReal       beta,gamma=1.0;
  PetscBool       breakdown,harmonic;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  harmonic = (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC)?PETSC_TRUE:PETSC_FALSE;
  if (harmonic) { ierr = PetscMalloc(ld*sizeof(PetscScalar),&g);CHKERRQ(ierr); }
  if (eps->arbitrary) pj = &j;
  else pj = NULL;

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],NULL);CHKERRQ(ierr);
  l = 0;

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = DSGetArray(eps->ds,DS_MAT_A,&S);CHKERRQ(ierr);
    ierr = EPSBasicArnoldi(eps,PETSC_FALSE,S,ld,eps->V,eps->nconv+l,&nv,u,&beta,&breakdown);CHKERRQ(ierr);
    ierr = VecScale(u,1.0/beta);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_A,&S);CHKERRQ(ierr);
    ierr = DSSetDimensions(eps->ds,nv,0,eps->nconv,eps->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(eps->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

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
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,nv-eps->nconv,eps->V,nv,beta,gamma,&k);CHKERRQ(ierr);
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
        ierr = PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%G)\n",eps->its,beta);CHKERRQ(ierr);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
        }
      } else {
        /* Undo translation of Krylov decomposition */
        if (harmonic) {
          ierr = DSSetDimensions(eps->ds,nv,0,k,l);CHKERRQ(ierr);
          ierr = DSTranslateHarmonic(eps->ds,0.0,beta,PETSC_TRUE,g,&gamma);CHKERRQ(ierr);
          /* gamma u^ = u - U*g~ */
          ierr = SlepcVecMAXPBY(u,1.0,-1.0,nv,g,eps->V);CHKERRQ(ierr);
          ierr = VecScale(u,1.0/gamma);CHKERRQ(ierr);
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
    ierr = DSGetArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv,eps->V,eps->nconv,k+l,Q,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);

    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecCopy(u,eps->V[k+l]);CHKERRQ(ierr);
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
#define __FUNCT__ "EPSSetFromOptions_KrylovSchur"
PetscErrorCode EPSSetFromOptions_KrylovSchur(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscReal      keep;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EPS Krylov-Schur Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps_krylovschur_restart","Proportion of vectors kept after restart","EPSKrylovSchurSetRestart",0.5,&keep,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = EPSKrylovSchurSetRestart(eps,keep);CHKERRQ(ierr);
  }
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
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset_KrylovSchur"
PetscErrorCode EPSReset_KrylovSchur(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  ctx->keep = 0.0;
  ierr = EPSReset_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_KrylovSchur"
PetscErrorCode EPSDestroy_KrylovSchur(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetRestart_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetRestart_C","",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_KrylovSchur"
PETSC_EXTERN PetscErrorCode EPSCreate_KrylovSchur(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,EPS_KRYLOVSCHUR,&eps->data);CHKERRQ(ierr);
  eps->ops->setup          = EPSSetUp_KrylovSchur;
  eps->ops->setfromoptions = EPSSetFromOptions_KrylovSchur;
  eps->ops->destroy        = EPSDestroy_KrylovSchur;
  eps->ops->reset          = EPSReset_KrylovSchur;
  eps->ops->view           = EPSView_KrylovSchur;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->computevectors = EPSComputeVectors_Schur;
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurSetRestart_C","EPSKrylovSchurSetRestart_KrylovSchur",EPSKrylovSchurSetRestart_KrylovSchur);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSKrylovSchurGetRestart_C","EPSKrylovSchurGetRestart_KrylovSchur",EPSKrylovSchurGetRestart_KrylovSchur);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

